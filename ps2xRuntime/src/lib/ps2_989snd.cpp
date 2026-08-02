#include "ps2_989snd.h"

#include "runtime/ps2_audio.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

/*
 * The SBlk layout and 240 Hz grain sequencing behavior are based in part on
 * the OpenGOAL 989snd implementation.
 *
 * Copyright (c) 2020-2026 OpenGOAL Team
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
 * OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
 * CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

namespace ps2_989snd
{
    namespace
    {
        constexpr uint32_t kSfxBlockDataId = 0x6B6C4253u; // "SBlk"
        constexpr uint32_t kContainerType1 = 1u;
        constexpr uint32_t kContainerType3 = 3u;
        constexpr uint32_t kSfxVersion1 = 1u;
        constexpr size_t kContainerPrefixSize = 8u;
        constexpr size_t kChunkDescriptorSize = 8u;
        constexpr size_t kSfxHeaderMinimumSize = 0x34u;
        constexpr size_t kSoundRecordSize = 0x0Cu;
        constexpr size_t kVersion1GrainSize = 0x28u;
        constexpr size_t kSpuAdpcmBlockSize = 0x10u;
        constexpr uint32_t kOutputSampleRate = 48'000u;
        constexpr uint32_t kHandlerRate = 240u;
        constexpr uint32_t kFramesPerHandlerTick =
            kOutputSampleRate / kHandlerRate;
        constexpr uint32_t kMaximumSounds = 4096u;
        constexpr uint32_t kMaximumGrains = 65'536u;
        constexpr uint32_t kMaximumImmediateGrains = 4096u;
        constexpr int32_t kVolumeDontChange = 0x7FFFFFFF;
        constexpr int32_t kPanReset = -1;
        constexpr int32_t kPanDontChange = -2;
        constexpr double kPi = 3.14159265358979323846;

        enum class GrainType : uint32_t
        {
            Null = 0u,
            Tone = 1u,
            XrefId = 2u,
            XrefNum = 3u,
            LfoSettings = 4u,
            StartChildSound = 5u,
            StopChildSound = 6u,
            PluginMessage = 7u,
            Branch = 8u,
            Tone2 = 9u,
            ControlNull = 20u,
            LoopStart = 21u,
            LoopEnd = 22u,
            LoopContinue = 23u,
            Stop = 24u,
            RandomPlay = 25u,
            RandomDelay = 26u,
            RandomPitchBend = 27u,
            PitchBend = 28u,
            AddPitchBend = 29u,
            SetRegister = 30u,
            SetRegisterRandom = 31u,
            IncrementRegister = 32u,
            DecrementRegister = 33u,
            TestRegister = 34u,
            Marker = 35u,
            GotoMarker = 36u,
            GotoRandomMarker = 37u,
            WaitForAllVoices = 38u,
            PlayCycle = 39u,
            AddRegister = 40u,
            KeyOffVoices = 41u,
            KillVoices = 42u,
            OnStopMarker = 43u,
            CopyRegister = 44u,
        };

        enum class LfoShape : uint8_t
        {
            Off = 0u,
            Sine = 1u,
            Square = 2u,
            Triangle = 3u,
            Saw = 4u,
            Random = 5u,
        };

        enum class LfoTarget : uint8_t
        {
            None = 0u,
            Volume = 1u,
            Pan = 2u,
            PitchModifier = 3u,
            PitchBend = 4u,
        };

        struct Chunk
        {
            uint32_t offset = 0u;
            uint32_t size = 0u;
        };

        struct Tone
        {
            int8_t priority = 0;
            int8_t volume = 0;
            int8_t centerNote = 0;
            int8_t centerFine = 0;
            int16_t pan = 0;
            int8_t mapLow = 0;
            int8_t mapHigh = 0;
            int8_t pitchBendLow = 0;
            int8_t pitchBendHigh = 0;
            uint16_t adsr1 = 0u;
            uint16_t adsr2 = 0u;
            uint16_t flags = 0u;
            uint32_t sampleOffset = 0u;
        };

        struct Grain
        {
            GrainType type = GrainType::Null;
            int32_t delay = 0;
            std::array<uint8_t, 32u> parameters{};
        };

        struct Sound
        {
            int8_t volume = 0;
            int8_t volumeGroup = 0;
            int16_t pan = 0;
            int8_t instanceLimit = 0;
            uint16_t flags = 0u;
            std::vector<Grain> grains;
        };

        struct DecodedSample
        {
            std::vector<int16_t> pcm;
            size_t loopStart = 0u;
            bool repeats = false;
        };

        struct Bank
        {
            uint32_t id = 0u;
            std::vector<uint8_t> sampleData;
            std::vector<Sound> sounds;
            std::unordered_map<uint32_t, DecodedSample> decodedSamples;
        };

        bool containsRange(size_t total, size_t offset, size_t size)
        {
            return offset <= total && size <= total - offset;
        }

        bool readU16(std::span<const uint8_t> bytes,
                     size_t offset,
                     uint16_t &value)
        {
            if (!containsRange(bytes.size(), offset, sizeof(uint16_t)))
                return false;
            value = static_cast<uint16_t>(bytes[offset]) |
                    static_cast<uint16_t>(bytes[offset + 1u] << 8u);
            return true;
        }

        bool readS16(std::span<const uint8_t> bytes,
                     size_t offset,
                     int16_t &value)
        {
            uint16_t raw = 0u;
            if (!readU16(bytes, offset, raw))
                return false;
            std::memcpy(&value, &raw, sizeof(value));
            return true;
        }

        bool readU32(std::span<const uint8_t> bytes,
                     size_t offset,
                     uint32_t &value)
        {
            if (!containsRange(bytes.size(), offset, sizeof(uint32_t)))
                return false;
            value = static_cast<uint32_t>(bytes[offset]) |
                    (static_cast<uint32_t>(bytes[offset + 1u]) << 8u) |
                    (static_cast<uint32_t>(bytes[offset + 2u]) << 16u) |
                    (static_cast<uint32_t>(bytes[offset + 3u]) << 24u);
            return true;
        }

        bool readS32(std::span<const uint8_t> bytes,
                     size_t offset,
                     int32_t &value)
        {
            uint32_t raw = 0u;
            if (!readU32(bytes, offset, raw))
                return false;
            std::memcpy(&value, &raw, sizeof(value));
            return true;
        }

        int16_t controlParameter(const Grain &grain, size_t index)
        {
            int16_t value = 0;
            const std::span<const uint8_t> bytes(grain.parameters);
            (void)readS16(bytes, index * sizeof(int16_t), value);
            return value;
        }

        void setControlParameter(Grain &grain, size_t index, int16_t value)
        {
            const size_t offset = index * sizeof(int16_t);
            if (!containsRange(grain.parameters.size(), offset, sizeof(value)))
                return;
            const uint16_t raw = static_cast<uint16_t>(value);
            grain.parameters[offset] = static_cast<uint8_t>(raw);
            grain.parameters[offset + 1u] = static_cast<uint8_t>(raw >> 8u);
        }

        int32_t signedParameter(const Grain &grain, size_t offset = 0u)
        {
            int32_t value = 0;
            const std::span<const uint8_t> bytes(grain.parameters);
            (void)readS32(bytes, offset, value);
            return value;
        }

        bool parseTone(const Grain &grain, Tone &tone)
        {
            const std::span<const uint8_t> bytes(grain.parameters);
            if (bytes.size() < 24u)
                return false;

            tone.priority = static_cast<int8_t>(bytes[0u]);
            tone.volume = static_cast<int8_t>(bytes[1u]);
            tone.centerNote = static_cast<int8_t>(bytes[2u]);
            tone.centerFine = static_cast<int8_t>(bytes[3u]);
            tone.mapLow = static_cast<int8_t>(bytes[6u]);
            tone.mapHigh = static_cast<int8_t>(bytes[7u]);
            tone.pitchBendLow = static_cast<int8_t>(bytes[8u]);
            tone.pitchBendHigh = static_cast<int8_t>(bytes[9u]);
            return readS16(bytes, 4u, tone.pan) &&
                   readU16(bytes, 10u, tone.adsr1) &&
                   readU16(bytes, 12u, tone.adsr2) &&
                   readU16(bytes, 14u, tone.flags) &&
                   readU32(bytes, 16u, tone.sampleOffset);
        }

        bool decodeSample(std::span<const uint8_t> data,
                          uint32_t sampleOffset,
                          DecodedSample &sample)
        {
            if ((sampleOffset & (kSpuAdpcmBlockSize - 1u)) != 0u ||
                !containsRange(data.size(), sampleOffset, kSpuAdpcmBlockSize))
            {
                return false;
            }

            ps2_spu_adpcm::DecoderState decoder{};
            bool foundEnd = false;
            bool foundLoopStart = false;
            for (size_t offset = sampleOffset;
                 containsRange(data.size(), offset, kSpuAdpcmBlockSize);
                 offset += kSpuAdpcmBlockSize)
            {
                const uint8_t flags = data[offset + 1u];
                if ((flags & 0x04u) != 0u)
                {
                    sample.loopStart = sample.pcm.size();
                    foundLoopStart = true;
                }
                if (!ps2_spu_adpcm::decodeBlocks(
                        data.data() + offset,
                        static_cast<uint32_t>(kSpuAdpcmBlockSize),
                        decoder,
                        sample.pcm))
                {
                    return false;
                }
                if ((flags & 0x01u) != 0u)
                {
                    sample.repeats = (flags & 0x02u) != 0u;
                    foundEnd = true;
                    break;
                }
            }

            if (!foundEnd || sample.pcm.empty())
                return false;
            if (!foundLoopStart)
                sample.loopStart = 0u;
            return !sample.repeats || sample.loopStart < sample.pcm.size();
        }

        bool parseBank(uint32_t bankHandle,
                       const uint8_t *containerData,
                       size_t containerSize,
                       Bank &bank)
        {
            if (!containerData || containerSize < kContainerPrefixSize)
                return false;

            const std::span<const uint8_t> container(containerData, containerSize);
            uint32_t type = 0u;
            uint32_t chunkCount = 0u;
            if (!readU32(container, 0u, type) ||
                !readU32(container, 4u, chunkCount) ||
                (type != kContainerType1 && type != kContainerType3) ||
                chunkCount < 2u || chunkCount > 3u)
            {
                return false;
            }

            const size_t tableSize =
                kContainerPrefixSize + static_cast<size_t>(chunkCount) * kChunkDescriptorSize;
            if (tableSize > container.size())
                return false;

            std::vector<Chunk> chunks;
            chunks.reserve(chunkCount);
            uint64_t previousEnd = tableSize;
            for (uint32_t index = 0u; index < chunkCount; ++index)
            {
                Chunk chunk{};
                const size_t descriptor =
                    kContainerPrefixSize + static_cast<size_t>(index) * kChunkDescriptorSize;
                if (!readU32(container, descriptor, chunk.offset) ||
                    !readU32(container, descriptor + 4u, chunk.size))
                {
                    return false;
                }
                const uint64_t end = static_cast<uint64_t>(chunk.offset) + chunk.size;
                if (chunk.size == 0u || chunk.offset < previousEnd ||
                    end > container.size())
                {
                    return false;
                }
                chunks.push_back(chunk);
                previousEnd = end;
            }

            const Chunk metadataChunk = chunks[0u];
            const Chunk samplesChunk = chunks[1u];
            const std::span<const uint8_t> metadata =
                container.subspan(metadataChunk.offset, metadataChunk.size);
            const std::span<const uint8_t> samples =
                container.subspan(samplesChunk.offset, samplesChunk.size);
            if (metadata.size() < kSfxHeaderMinimumSize)
                return false;

            uint32_t dataId = 0u;
            uint32_t version = 0u;
            uint32_t declaredBankId = 0u;
            uint16_t soundCount = 0u;
            uint16_t grainCount = 0u;
            uint16_t sampleCount = 0u;
            uint32_t firstSound = 0u;
            uint32_t firstGrain = 0u;
            if (!readU32(metadata, 0x00u, dataId) || dataId != kSfxBlockDataId ||
                !readU32(metadata, 0x04u, version) || version != kSfxVersion1 ||
                !readU32(metadata, 0x0Cu, declaredBankId) ||
                !readU16(metadata, 0x16u, soundCount) ||
                !readU16(metadata, 0x18u, grainCount) ||
                !readU16(metadata, 0x1Au, sampleCount) ||
                !readU32(metadata, 0x1Cu, firstSound) ||
                !readU32(metadata, 0x20u, firstGrain) ||
                soundCount == 0u || soundCount > kMaximumSounds ||
                grainCount > kMaximumGrains ||
                !containsRange(metadata.size(),
                               firstSound,
                               static_cast<size_t>(soundCount) * kSoundRecordSize) ||
                !containsRange(metadata.size(),
                               firstGrain,
                               static_cast<size_t>(grainCount) * kVersion1GrainSize))
            {
                return false;
            }
            (void)sampleCount;

            bank.id = declaredBankId != 0u ? declaredBankId : bankHandle;
            bank.sampleData.assign(samples.begin(), samples.end());
            bank.sounds.reserve(soundCount);
            for (uint32_t soundIndex = 0u; soundIndex < soundCount; ++soundIndex)
            {
                const size_t record =
                    firstSound + static_cast<size_t>(soundIndex) * kSoundRecordSize;
                Sound sound{};
                sound.volume = static_cast<int8_t>(metadata[record + 0u]);
                sound.volumeGroup = static_cast<int8_t>(metadata[record + 1u]);
                int16_t pan = 0;
                uint16_t flags = 0u;
                uint32_t firstSoundGrain = 0u;
                const int8_t count = static_cast<int8_t>(metadata[record + 4u]);
                sound.instanceLimit = static_cast<int8_t>(metadata[record + 5u]);
                if (count < 0 ||
                    !readS16(metadata, record + 2u, pan) ||
                    !readU16(metadata, record + 6u, flags) ||
                    !readU32(metadata, record + 8u, firstSoundGrain) ||
                    (firstSoundGrain % kVersion1GrainSize) != 0u ||
                    !containsRange(static_cast<size_t>(grainCount) * kVersion1GrainSize,
                                   firstSoundGrain,
                                   static_cast<size_t>(count) * kVersion1GrainSize))
                {
                    return false;
                }

                sound.pan = pan;
                sound.flags = flags;
                sound.grains.reserve(static_cast<size_t>(count));
                for (int32_t grainIndex = 0; grainIndex < count; ++grainIndex)
                {
                    const size_t grainOffset =
                        firstGrain + firstSoundGrain +
                        static_cast<size_t>(grainIndex) * kVersion1GrainSize;
                    uint32_t rawType = 0u;
                    Grain grain{};
                    if (!readU32(metadata, grainOffset, rawType) ||
                        !readS32(metadata, grainOffset + 4u, grain.delay))
                    {
                        return false;
                    }
                    grain.type = static_cast<GrainType>(rawType);
                    std::copy_n(metadata.data() + grainOffset + 8u,
                                grain.parameters.size(),
                                grain.parameters.begin());
                    sound.grains.push_back(grain);
                }
                bank.sounds.push_back(std::move(sound));
            }

            for (const Sound &sound : bank.sounds)
            {
                for (const Grain &grain : sound.grains)
                {
                    if (grain.type != GrainType::Tone &&
                        grain.type != GrainType::Tone2)
                    {
                        continue;
                    }
                    Tone tone{};
                    if (!parseTone(grain, tone))
                        return false;
                    if (bank.decodedSamples.find(tone.sampleOffset) !=
                        bank.decodedSamples.end())
                    {
                        continue;
                    }
                    DecodedSample decoded{};
                    if (!decodeSample(bank.sampleData, tone.sampleOffset, decoded))
                        return false;
                    bank.decodedSamples.emplace(tone.sampleOffset, std::move(decoded));
                }
            }

            return true;
        }

        class AdsrEnvelope
        {
        public:
            enum class Phase
            {
                Attack,
                Decay,
                Sustain,
                Release,
                Stopped,
            };

            void setRegisters(uint16_t adsr1, uint16_t adsr2)
            {
                m_register = static_cast<uint32_t>(adsr1) |
                             (static_cast<uint32_t>(adsr2) << 16u);
            }

            void attack()
            {
                m_phase = Phase::Attack;
                m_level = 0;
                m_counter = 0u;
                updateSettings();
            }

            void release()
            {
                if (m_phase == Phase::Stopped)
                    return;
                m_phase = Phase::Release;
                m_counter = 0u;
                updateSettings();
            }

            void stop()
            {
                m_phase = Phase::Stopped;
                m_level = 0;
            }

            void run()
            {
                if (m_phase == Phase::Stopped)
                    return;

                uint32_t counterStep = 0x800000u;
                const int32_t shift = m_shift - 11;
                if (shift > 0)
                    counterStep >>= shift;
                const int32_t stepShift = std::max(0, 11 - m_shift);
                int16_t step = static_cast<int16_t>(
                    static_cast<int32_t>(m_step) * (1 << stepShift));

                if (m_exponential)
                {
                    if (!m_decrease && m_level > 0x6000)
                        counterStep >>= 2u;
                    if (m_decrease)
                    {
                        step = static_cast<int16_t>(
                            (static_cast<int32_t>(step) * m_level) >> 15);
                    }
                }

                m_counter += counterStep;
                if (m_counter >= 0x800000u)
                {
                    m_counter = 0u;
                    m_level = std::clamp<int32_t>(m_level + step, 0, 0x7FFF);
                }

                if (m_phase == Phase::Sustain)
                    return;
                if ((!m_decrease && m_level >= m_target) ||
                    (m_decrease && m_level <= m_target))
                {
                    if (m_phase == Phase::Attack)
                        m_phase = Phase::Decay;
                    else if (m_phase == Phase::Decay)
                        m_phase = Phase::Sustain;
                    else if (m_phase == Phase::Release)
                        m_phase = Phase::Stopped;
                    updateSettings();
                }
            }

            [[nodiscard]] int32_t level() const { return m_level; }
            [[nodiscard]] bool stopped() const { return m_phase == Phase::Stopped; }

        private:
            void updateSettings()
            {
                switch (m_phase)
                {
                case Phase::Attack:
                    m_exponential = ((m_register >> 15u) & 1u) != 0u;
                    m_decrease = false;
                    m_shift = static_cast<int32_t>((m_register >> 10u) & 0x1Fu);
                    m_step = static_cast<int8_t>(7 - ((m_register >> 8u) & 0x03u));
                    m_target = 0x7FFF;
                    break;
                case Phase::Decay:
                    m_exponential = true;
                    m_decrease = true;
                    m_shift = static_cast<int32_t>((m_register >> 4u) & 0x0Fu);
                    m_step = -8;
                    m_target = static_cast<int32_t>((m_register & 0x0Fu) + 1u) << 11;
                    break;
                case Phase::Sustain:
                    m_exponential = ((m_register >> 31u) & 1u) != 0u;
                    m_decrease = ((m_register >> 30u) & 1u) != 0u;
                    m_shift = static_cast<int32_t>((m_register >> 24u) & 0x1Fu);
                    m_step = m_decrease
                                 ? static_cast<int8_t>(-8 + ((m_register >> 22u) & 0x03u))
                                 : static_cast<int8_t>(7 - ((m_register >> 22u) & 0x03u));
                    m_target = 0;
                    break;
                case Phase::Release:
                    m_exponential = ((m_register >> 21u) & 1u) != 0u;
                    m_decrease = true;
                    m_shift = static_cast<int32_t>((m_register >> 16u) & 0x1Fu);
                    m_step = -8;
                    m_target = 0;
                    break;
                case Phase::Stopped:
                    break;
                }
            }

            Phase m_phase = Phase::Stopped;
            uint32_t m_register = 0u;
            uint32_t m_counter = 0u;
            int32_t m_level = 0;
            int32_t m_target = 0;
            int32_t m_shift = 0;
            int8_t m_step = 0;
            bool m_exponential = false;
            bool m_decrease = false;
        };

        struct Lfo
        {
            LfoShape shape = LfoShape::Off;
            LfoTarget target = LfoTarget::None;
            uint8_t setupFlags = 0u;
            int16_t depth = 0;
            uint32_t nextStep = 0u;
            uint32_t stepSize = 0u;
            int32_t state1 = 0;
            int32_t state2 = 0;
            uint32_t tick = 0u;
        };

        struct Handler
        {
            uint32_t bankHandle = 0u;
            uint32_t soundHandle = 0u;
            uint32_t originalSoundId = 0u;
            uint32_t currentSoundId = 0u;
            uint32_t nextGrain = 0u;
            int32_t countdown = 0;
            int32_t originalVolume = 0;
            int32_t originalPan = 0;
            int32_t applicationVolume = 1024;
            int32_t applicationPan = 0;
            int32_t applicationPitchModifier = 0;
            int32_t applicationPitchBend = 0;
            int32_t currentVolume = 0;
            int32_t currentPan = 0;
            int32_t currentPitchModifier = 0;
            int32_t currentPitchBend = 0;
            int32_t lfoVolume = 0;
            int32_t lfoPan = 0;
            int32_t lfoPitchModifier = 0;
            int32_t lfoPitchBend = 0;
            uint8_t group = 0u;
            std::array<int8_t, 4u> registers{};
            std::array<Lfo, 4u> lfos{};
            uint32_t grainsToPlay = 0u;
            uint32_t grainsToSkip = 0u;
            bool skipGrains = false;
            bool paused = false;
            bool done = false;
        };

        struct Voice
        {
            uint32_t owner = 0u;
            uint32_t bankHandle = 0u;
            const DecodedSample *sample = nullptr;
            Tone tone{};
            double position = 0.0;
            double step = 1.0;
            double leftGain = 0.0;
            double rightGain = 0.0;
            int32_t grainVolume = 0;
            int32_t grainPan = 0;
            AdsrEnvelope envelope{};
            bool dead = false;
        };

        int16_t clampSample(int64_t value)
        {
            return static_cast<int16_t>(std::clamp<int64_t>(
                value,
                std::numeric_limits<int16_t>::min(),
                std::numeric_limits<int16_t>::max()));
        }
    }

    struct Engine::Impl
    {
        mutable std::mutex mutex;
        std::unordered_map<uint32_t, std::unique_ptr<Bank>> banks;
        std::map<uint32_t, Handler> handlers;
        std::vector<Voice> voices;
        std::array<int32_t, 32u> masterVolume{};
        std::array<int8_t, 32u> globalRegisters{};
        DebugSnapshot debug{};
        uint32_t randomState = 0x9895A17Du;
        uint32_t framesUntilHandlerTick = 0u;

        Impl()
        {
            masterVolume.fill(1024);
        }

        uint32_t random()
        {
            randomState ^= randomState << 13u;
            randomState ^= randomState >> 17u;
            randomState ^= randomState << 5u;
            return randomState & 0x7FFFFFFFu;
        }

        int32_t randomBelow(int32_t upperExclusive)
        {
            return upperExclusive > 0
                       ? static_cast<int32_t>(random() % static_cast<uint32_t>(upperExclusive))
                       : 0;
        }

        Bank *findBank(uint32_t handle)
        {
            const auto bank = banks.find(handle);
            return bank == banks.end() ? nullptr : bank->second.get();
        }

        const Bank *findBank(uint32_t handle) const
        {
            const auto bank = banks.find(handle);
            return bank == banks.end() ? nullptr : bank->second.get();
        }

        Sound *findSound(Handler &handler)
        {
            Bank *bank = findBank(handler.bankHandle);
            if (!bank || handler.currentSoundId >= bank->sounds.size())
                return nullptr;
            return &bank->sounds[handler.currentSoundId];
        }

        bool hasOwnerVoices(uint32_t owner) const
        {
            return std::any_of(voices.begin(), voices.end(), [&](const Voice &voice)
                               { return !voice.dead && voice.owner == owner; });
        }

        void keyOffOwnerVoices(uint32_t owner, bool kill)
        {
            for (Voice &voice : voices)
            {
                if (voice.owner != owner || voice.dead)
                    continue;
                if (kill)
                {
                    voice.envelope.stop();
                    voice.dead = true;
                }
                else
                {
                    voice.envelope.release();
                }
            }
        }

        int32_t resolveVolume(int32_t volume, const Handler &handler)
        {
            if (volume >= 0)
                return std::clamp(volume, 0, 127);
            if (volume >= -4)
                return std::max<int32_t>(handler.registers[-volume - 1], 0);
            if (volume == -5)
                return randomBelow(127);
            const size_t index = static_cast<size_t>(-volume - 6);
            return index < globalRegisters.size()
                       ? std::max<int32_t>(globalRegisters[index], 0)
                       : 0;
        }

        int32_t resolvePan(int32_t pan, const Handler &handler)
        {
            if (pan >= 0)
                return pan % 360;
            int32_t value = 0;
            if (pan >= -4)
                value = handler.registers[-pan - 1];
            else if (pan == -5)
                return randomBelow(360);
            else
            {
                const size_t index = static_cast<size_t>(-pan - 6);
                value = index < globalRegisters.size() ? globalRegisters[index] : 0;
            }
            return (360 * std::min(std::abs(value), 127)) / 127;
        }

        void updateVoice(Voice &voice, const Handler &handler)
        {
            const int32_t combinedPan =
                ((handler.currentPan + voice.grainPan) % 360 + 360) % 360;
            const double panRadians =
                static_cast<double>(combinedPan) * kPi / 180.0;
            const double panPosition = std::sin(panRadians);
            const double leftPan = std::sqrt(std::max(0.0, (1.0 - panPosition) * 0.5));
            const double rightPan = std::sqrt(std::max(0.0, (1.0 + panPosition) * 0.5));
            const int32_t groupVolume =
                handler.group < masterVolume.size() ? masterVolume[handler.group] : 1024;
            const double amplitude =
                (static_cast<double>(handler.currentVolume) / 127.0) *
                (static_cast<double>(voice.grainVolume) / 127.0) *
                (static_cast<double>(std::clamp(groupVolume, 0, 1024)) / 1024.0);
            voice.leftGain = amplitude * leftPan;
            voice.rightGain = amplitude * rightPan;

            int32_t noteFine = 60 * 128 + handler.currentPitchModifier;
            if (handler.currentPitchBend >= 0)
            {
                noteFine += static_cast<int32_t>(voice.tone.pitchBendHigh) *
                            ((handler.currentPitchBend * 128) / 0x7FFF);
            }
            else
            {
                noteFine += static_cast<int32_t>(voice.tone.pitchBendLow) *
                            ((handler.currentPitchBend * 128) / 0x8000);
            }
            const int32_t centerNote = std::abs(static_cast<int32_t>(voice.tone.centerNote));
            const double semitones =
                static_cast<double>(noteFine -
                                    centerNote * 128 -
                                    static_cast<int32_t>(voice.tone.centerFine)) /
                128.0;
            const double sourceRate = voice.tone.centerNote >= 0 ? 44'100.0 : 48'000.0;
            voice.step = std::clamp(
                (sourceRate / static_cast<double>(kOutputSampleRate)) *
                    std::pow(2.0, semitones / 12.0),
                1.0 / 4096.0,
                4.0);
        }

        void recomputeHandler(Handler &handler)
        {
            handler.currentVolume = std::clamp(
                ((handler.applicationVolume * handler.originalVolume) >> 10) +
                    handler.lfoVolume,
                0,
                127);
            handler.currentPan = handler.applicationPan + handler.lfoPan;
            while (handler.currentPan >= 360)
                handler.currentPan -= 360;
            while (handler.currentPan < 0)
                handler.currentPan += 360;
            handler.currentPitchModifier =
                handler.applicationPitchModifier + handler.lfoPitchModifier;
            handler.currentPitchBend = std::clamp(
                handler.applicationPitchBend + handler.lfoPitchBend,
                static_cast<int32_t>(std::numeric_limits<int16_t>::min()),
                static_cast<int32_t>(std::numeric_limits<int16_t>::max()));

            for (Voice &voice : voices)
            {
                if (!voice.dead && voice.owner == handler.soundHandle)
                    updateVoice(voice, handler);
            }
        }

        bool startTone(Handler &handler, const Grain &grain)
        {
            Tone tone{};
            if (!parseTone(grain, tone) || (tone.flags & 0x08u) != 0u)
                return false;
            Bank *bank = findBank(handler.bankHandle);
            if (!bank)
                return false;
            const auto sample = bank->decodedSamples.find(tone.sampleOffset);
            if (sample == bank->decodedSamples.end())
                return false;

            Voice voice{};
            voice.owner = handler.soundHandle;
            voice.bankHandle = handler.bankHandle;
            voice.sample = &sample->second;
            voice.tone = tone;
            voice.grainVolume = resolveVolume(tone.volume, handler);
            voice.grainPan = resolvePan(tone.pan, handler);
            voice.envelope.setRegisters(tone.adsr1, tone.adsr2);
            voice.envelope.attack();
            updateVoice(voice, handler);
            voices.push_back(std::move(voice));
            return true;
        }

        int32_t lfoValue(Lfo &lfo)
        {
            const int32_t step = static_cast<int32_t>(lfo.nextStep >> 16u);
            lfo.nextStep = (lfo.nextStep + 2u * lfo.stepSize) & 0x07FFFFFFu;
            int32_t value = 0;
            switch (lfo.shape)
            {
            case LfoShape::Sine:
                value = static_cast<int32_t>(
                    std::sin(static_cast<double>(step) * 2.0 * kPi / 2048.0) * 32767.0);
                break;
            case LfoShape::Square:
                value = step >= lfo.state1 ? -32767 : 32767;
                break;
            case LfoShape::Triangle:
                if (step < 512)
                    value = 0x7FFF * step / 512;
                else if (step >= 1536)
                    value = 0x7FFF * (step - 1536) / 512 - 0x7FFF;
                else
                    value = 0x7FFF - 65534 * (step - 512) / 1024;
                break;
            case LfoShape::Saw:
                value = step >= 1024
                            ? 0x7FFF * (step - 1024) / 1024 - 0x7FFF
                            : 0x7FFF * step / 1023;
                break;
            case LfoShape::Random:
                if (step >= 1024 && lfo.state2 == 1)
                {
                    lfo.state2 = 0;
                    lfo.state1 = 2 * (randomBelow(0x8000) - 0x3FFF);
                }
                else if (step < 1024 && lfo.state2 == 0)
                {
                    lfo.state2 = 1;
                    lfo.state1 = -randomBelow(0x8000) * randomBelow(2);
                }
                value = lfo.state1;
                break;
            case LfoShape::Off:
                break;
            }
            return (lfo.setupFlags & 1u) != 0u ? -value : value;
        }

        void tickLfos(Handler &handler)
        {
            Sound *sound = findSound(handler);
            if (!sound)
                return;
            bool changed = false;
            for (Lfo &lfo : handler.lfos)
            {
                ++lfo.tick;
                if (lfo.target == LfoTarget::None || (lfo.tick & 1u) == 0u)
                    continue;
                const int32_t value = lfoValue(lfo);
                switch (lfo.target)
                {
                case LfoTarget::Volume:
                    handler.lfoVolume =
                        (((static_cast<int32_t>(sound->volume) * lfo.depth) >> 10) *
                         (value - 0x7FFF)) >>
                        16;
                    changed = true;
                    break;
                case LfoTarget::Pan:
                    handler.lfoPan =
                        (((180 * lfo.depth) >> 10) * value) >> 15;
                    changed = true;
                    break;
                case LfoTarget::PitchModifier:
                    handler.lfoPitchModifier =
                        (((6096 * lfo.depth) >> 10) * value) >> 15;
                    changed = true;
                    break;
                case LfoTarget::PitchBend:
                    handler.lfoPitchBend =
                        (((0x7FFF * lfo.depth) >> 10) * value) >> 15;
                    changed = true;
                    break;
                case LfoTarget::None:
                    break;
                }
            }
            if (changed)
                recomputeHandler(handler);
        }

        void configureLfo(Handler &handler, const Grain &grain)
        {
            const auto &p = grain.parameters;
            const uint8_t index = p[0u];
            if (index >= handler.lfos.size())
                return;
            Lfo &lfo = handler.lfos[index];
            lfo.target = static_cast<LfoTarget>(p[1u]);
            lfo.shape = static_cast<LfoShape>(p[3u]);
            uint16_t duty = 0u;
            uint16_t depth = 0u;
            uint16_t flags = 0u;
            uint16_t startOffset = 0u;
            uint32_t stepSize = 0u;
            const std::span<const uint8_t> bytes(p);
            (void)readU16(bytes, 4u, duty);
            (void)readU16(bytes, 6u, depth);
            (void)readU16(bytes, 8u, flags);
            (void)readU16(bytes, 10u, startOffset);
            (void)readU32(bytes, 12u, stepSize);
            lfo.depth = static_cast<int16_t>(depth);
            lfo.setupFlags = static_cast<uint8_t>(flags);
            lfo.nextStep = (flags & 2u) != 0u
                               ? (random() & 0x7FFu) << 16u
                               : static_cast<uint32_t>(startOffset) << 16u;
            lfo.stepSize = stepSize;
            lfo.state1 = lfo.shape == LfoShape::Square ? duty : 0;
            lfo.state2 = 0;
            lfo.tick = 0u;
            if (lfo.shape == LfoShape::Random)
            {
                lfo.state1 = -randomBelow(0x8000) * randomBelow(2);
                lfo.state2 = 1;
            }
        }

        int32_t registerValue(const Handler &handler, int32_t index) const
        {
            if (index < 0)
            {
                const size_t global = static_cast<size_t>(-index - 1);
                return global < globalRegisters.size() ? globalRegisters[global] : 0;
            }
            return static_cast<size_t>(index) < handler.registers.size()
                       ? handler.registers[static_cast<size_t>(index)]
                       : 0;
        }

        void setRegisterValue(Handler &handler, int32_t index, int32_t value)
        {
            const int8_t clamped = static_cast<int8_t>(
                std::clamp(value,
                           static_cast<int32_t>(std::numeric_limits<int8_t>::min()),
                           static_cast<int32_t>(std::numeric_limits<int8_t>::max())));
            if (index < 0)
            {
                const size_t global = static_cast<size_t>(-index - 1);
                if (global < globalRegisters.size())
                    globalRegisters[global] = clamped;
                return;
            }
            if (static_cast<size_t>(index) < handler.registers.size())
                handler.registers[static_cast<size_t>(index)] = clamped;
        }

        int32_t executeGrain(Handler &handler, Grain &grain)
        {
            Sound *sound = findSound(handler);
            if (!sound)
            {
                handler.done = true;
                return 0;
            }

            switch (grain.type)
            {
            case GrainType::Null:
            case GrainType::ControlNull:
            case GrainType::LoopStart:
            case GrainType::Marker:
                return 0;
            case GrainType::Tone:
            case GrainType::Tone2:
                (void)startTone(handler, grain);
                return 0;
            case GrainType::LfoSettings:
                configureLfo(handler, grain);
                return 0;
            case GrainType::LoopEnd:
                for (int32_t index = static_cast<int32_t>(handler.nextGrain) - 1;
                     index >= 0;
                     --index)
                {
                    if (sound->grains[static_cast<size_t>(index)].type == GrainType::LoopStart)
                    {
                        handler.nextGrain = static_cast<uint32_t>(std::max(index - 1, -1));
                        if (index == 0)
                            handler.nextGrain = std::numeric_limits<uint32_t>::max();
                        return 0;
                    }
                }
                return 0;
            case GrainType::LoopContinue:
                for (uint32_t index = handler.nextGrain + 1u;
                     index < sound->grains.size();
                     ++index)
                {
                    if (sound->grains[index].type == GrainType::LoopEnd)
                    {
                        handler.nextGrain = index;
                        break;
                    }
                }
                return 0;
            case GrainType::Stop:
                handler.done = true;
                return 0;
            case GrainType::RandomPlay:
            {
                const int32_t options = controlParameter(grain, 0u);
                const int32_t count = controlParameter(grain, 1u);
                if (options <= 0 || count <= 0)
                    return 0;
                int32_t choice = randomBelow(options);
                const int32_t previous = controlParameter(grain, 2u);
                if (choice == previous)
                    choice = (choice + 1) % options;
                setControlParameter(grain, 2u, static_cast<int16_t>(choice));
                handler.nextGrain += static_cast<uint32_t>(choice * count);
                handler.grainsToPlay = static_cast<uint32_t>(count + 1);
                handler.grainsToSkip = static_cast<uint32_t>((options - 1 - choice) * count);
                handler.skipGrains = true;
                return 0;
            }
            case GrainType::RandomDelay:
            {
                const int32_t amount = signedParameter(grain);
                return randomBelow(amount);
            }
            case GrainType::RandomPitchBend:
            {
                const int32_t amount = controlParameter(grain, 0u);
                const int32_t randomValue = randomBelow(0x7FFF);
                handler.applicationPitchBend =
                    amount * (((0xFFFF * randomValue) / 0x7FFF) - 0x8000) / 100;
                recomputeHandler(handler);
                return 0;
            }
            case GrainType::PitchBend:
            {
                const int32_t bend = controlParameter(grain, 0u);
                handler.applicationPitchBend = bend >= 0
                                                   ? 0x7FFF * bend / 127
                                                   : -0x8000 * bend / -128;
                recomputeHandler(handler);
                return 0;
            }
            case GrainType::AddPitchBend:
            {
                const int32_t bend = controlParameter(grain, 0u);
                handler.applicationPitchBend = std::clamp(
                    handler.currentPitchBend + 0x7FFF * bend / 127,
                    static_cast<int32_t>(std::numeric_limits<int16_t>::min()),
                    static_cast<int32_t>(std::numeric_limits<int16_t>::max()));
                recomputeHandler(handler);
                return 0;
            }
            case GrainType::SetRegister:
                setRegisterValue(handler,
                                 controlParameter(grain, 0u),
                                 controlParameter(grain, 1u));
                return 0;
            case GrainType::SetRegisterRandom:
            {
                const int32_t low = controlParameter(grain, 1u);
                const int32_t high = controlParameter(grain, 2u);
                const int32_t range = high - low + 1;
                if (range > 0)
                {
                    setRegisterValue(handler,
                                     controlParameter(grain, 0u),
                                     low + randomBelow(range));
                }
                return 0;
            }
            case GrainType::IncrementRegister:
            {
                const int32_t index = controlParameter(grain, 0u);
                setRegisterValue(handler, index, registerValue(handler, index) + 1);
                return 0;
            }
            case GrainType::DecrementRegister:
            {
                const int32_t index = controlParameter(grain, 0u);
                setRegisterValue(handler, index, registerValue(handler, index) - 1);
                return 0;
            }
            case GrainType::TestRegister:
            {
                const int32_t value = registerValue(handler, controlParameter(grain, 0u));
                const int32_t operation = controlParameter(grain, 1u);
                const int32_t comparison = controlParameter(grain, 2u);
                const bool skip = operation == 0
                                      ? value >= comparison
                                      : operation == 1
                                            ? value != comparison
                                            : comparison >= value;
                if (skip)
                    ++handler.nextGrain;
                return 0;
            }
            case GrainType::GotoMarker:
            {
                const int32_t marker = controlParameter(grain, 0u);
                for (uint32_t index = 0u; index < sound->grains.size(); ++index)
                {
                    if (sound->grains[index].type == GrainType::Marker &&
                        controlParameter(sound->grains[index], 0u) == marker)
                    {
                        handler.nextGrain = index == 0u
                                                ? std::numeric_limits<uint32_t>::max()
                                                : index - 1u;
                        break;
                    }
                }
                return 0;
            }
            case GrainType::GotoRandomMarker:
            {
                const int32_t low = controlParameter(grain, 0u);
                const int32_t high = controlParameter(grain, 1u);
                const int32_t range = high - low + 1;
                const int32_t marker = range > 0 ? low + randomBelow(range) : low;
                for (uint32_t index = 0u; index < sound->grains.size(); ++index)
                {
                    if (sound->grains[index].type == GrainType::Marker &&
                        controlParameter(sound->grains[index], 0u) == marker)
                    {
                        handler.nextGrain = index == 0u
                                                ? std::numeric_limits<uint32_t>::max()
                                                : index - 1u;
                        break;
                    }
                }
                return 0;
            }
            case GrainType::WaitForAllVoices:
                if (hasOwnerVoices(handler.soundHandle))
                {
                    --handler.nextGrain;
                    return 1;
                }
                return 0;
            case GrainType::PlayCycle:
            {
                const int32_t groupSize = controlParameter(grain, 0u);
                const int32_t groupCount = controlParameter(grain, 1u);
                int32_t index = controlParameter(grain, 2u);
                if (groupSize <= 0 || groupCount <= 0)
                    return 0;
                const int32_t selected = index++;
                if (index == groupSize)
                    index = 0;
                setControlParameter(grain, 2u, static_cast<int16_t>(index));
                handler.nextGrain += static_cast<uint32_t>(groupCount * selected);
                handler.grainsToPlay = static_cast<uint32_t>(groupCount + 1);
                handler.grainsToSkip =
                    static_cast<uint32_t>((groupSize - 1 - selected) * groupCount);
                handler.skipGrains = true;
                return 0;
            }
            case GrainType::AddRegister:
            {
                const int32_t value = controlParameter(grain, 0u);
                const int32_t index = controlParameter(grain, 1u);
                setRegisterValue(handler, index, registerValue(handler, index) + value);
                return 0;
            }
            case GrainType::KeyOffVoices:
                keyOffOwnerVoices(handler.soundHandle, false);
                return 0;
            case GrainType::KillVoices:
                keyOffOwnerVoices(handler.soundHandle, true);
                return 0;
            case GrainType::OnStopMarker:
                handler.nextGrain = static_cast<uint32_t>(sound->grains.size() - 1u);
                return 0;
            case GrainType::CopyRegister:
                setRegisterValue(handler,
                                 controlParameter(grain, 1u),
                                 registerValue(handler, controlParameter(grain, 0u)));
                return 0;
            case GrainType::XrefId:
            case GrainType::XrefNum:
            case GrainType::StartChildSound:
            case GrainType::StopChildSound:
            case GrainType::PluginMessage:
            case GrainType::Branch:
                ++debug.unsupportedGrains;
                return 0;
            }
            ++debug.unsupportedGrains;
            return 0;
        }

        void doGrain(Handler &handler)
        {
            Sound *sound = findSound(handler);
            if (!sound || handler.nextGrain >= sound->grains.size())
            {
                handler.done = true;
                return;
            }

            Grain &grain = sound->grains[handler.nextGrain];
            const int32_t extraDelay = executeGrain(handler, grain);
            if (handler.skipGrains)
            {
                if (handler.grainsToPlay > 0u)
                    --handler.grainsToPlay;
                if (handler.grainsToPlay == 0u)
                {
                    handler.nextGrain += handler.grainsToSkip;
                    handler.skipGrains = false;
                }
            }

            ++handler.nextGrain;
            sound = findSound(handler);
            if (handler.done || !sound || handler.nextGrain >= sound->grains.size())
            {
                handler.done = true;
                return;
            }
            handler.countdown = sound->grains[handler.nextGrain].delay + extraDelay;
        }

        void processImmediateGrains(Handler &handler)
        {
            uint32_t count = 0u;
            while (handler.countdown <= 0 && !handler.done &&
                   count++ < kMaximumImmediateGrains)
            {
                doGrain(handler);
            }
            if (count >= kMaximumImmediateGrains)
            {
                handler.done = true;
                ++debug.rejectedCommands;
            }
        }

        void tickHandlers()
        {
            for (auto handler = handlers.begin(); handler != handlers.end();)
            {
                Handler &state = handler->second;
                tickLfos(state);
                if (!state.paused && !state.done)
                {
                    --state.countdown;
                    processImmediateGrains(state);
                }
                if (state.done && !hasOwnerVoices(state.soundHandle))
                    handler = handlers.erase(handler);
                else
                    ++handler;
            }
        }

        bool addHandler(uint32_t bankHandle,
                        uint32_t soundId,
                        uint32_t soundHandle,
                        int32_t volume,
                        int32_t pan,
                        int32_t pitchModifier,
                        int32_t pitchBend)
        {
            Bank *bank = findBank(bankHandle);
            if (!bank || soundHandle == 0u || soundId >= bank->sounds.size() ||
                bank->sounds[soundId].grains.empty())
            {
                ++debug.rejectedCommands;
                return false;
            }

            const auto previousHandler = handlers.find(soundHandle);
            if (previousHandler != handlers.end())
                handlers.erase(previousHandler);
            keyOffOwnerVoices(soundHandle, true);
            voices.erase(
                std::remove_if(voices.begin(), voices.end(),
                               [soundHandle](const Voice &voice)
                               { return voice.owner == soundHandle; }),
                voices.end());
            const Sound &sound = bank->sounds[soundId];
            Handler handler{};
            handler.bankHandle = bankHandle;
            handler.soundHandle = soundHandle;
            handler.originalSoundId = soundId;
            handler.currentSoundId = soundId;
            handler.originalVolume = sound.volume;
            handler.originalPan = sound.pan;
            handler.applicationVolume =
                volume == kVolumeDontChange ? 1024 : volume;
            handler.applicationPan =
                pan == kPanReset || pan == kPanDontChange ? sound.pan : pan;
            handler.applicationPitchModifier = pitchModifier;
            handler.applicationPitchBend = pitchBend;
            handler.group = static_cast<uint8_t>(sound.volumeGroup);
            handler.countdown = sound.grains.front().delay;
            recomputeHandler(handler);
            auto [inserted, accepted] = handlers.emplace(soundHandle, std::move(handler));
            if (!accepted)
                return false;
            processImmediateGrains(inserted->second);
            return true;
        }

        void stopSound(uint32_t soundHandle)
        {
            const auto handler = handlers.find(soundHandle);
            if (handler != handlers.end())
                handler->second.done = true;
            keyOffOwnerVoices(soundHandle, false);
        }

        void renderVoice(Voice &voice, int64_t &left, int64_t &right)
        {
            if (voice.dead || !voice.sample || voice.sample->pcm.empty())
            {
                voice.dead = true;
                return;
            }

            const DecodedSample &sample = *voice.sample;
            size_t first = static_cast<size_t>(voice.position);
            while (first >= sample.pcm.size())
            {
                if (!sample.repeats || sample.loopStart >= sample.pcm.size())
                {
                    voice.envelope.stop();
                    voice.dead = true;
                    return;
                }
                const size_t loopLength = sample.pcm.size() - sample.loopStart;
                voice.position = static_cast<double>(sample.loopStart) +
                                 std::fmod(voice.position - sample.loopStart,
                                           static_cast<double>(loopLength));
                first = static_cast<size_t>(voice.position);
            }
            size_t second = first + 1u;
            if (second >= sample.pcm.size())
                second = sample.repeats ? sample.loopStart : first;
            const double fraction = voice.position - static_cast<double>(first);
            const double pcm =
                static_cast<double>(sample.pcm[first]) * (1.0 - fraction) +
                static_cast<double>(sample.pcm[second]) * fraction;
            const double envelope = static_cast<double>(voice.envelope.level()) / 32767.0;
            left += static_cast<int64_t>(pcm * envelope * voice.leftGain);
            right += static_cast<int64_t>(pcm * envelope * voice.rightGain);
            voice.position += voice.step;
            voice.envelope.run();
            if (voice.envelope.stopped())
                voice.dead = true;
        }

        void render(int16_t *output, size_t frameCount)
        {
            if (!output)
                return;
            for (size_t frame = 0u; frame < frameCount; ++frame)
            {
                if (framesUntilHandlerTick == 0u)
                {
                    tickHandlers();
                    framesUntilHandlerTick = kFramesPerHandlerTick;
                }
                --framesUntilHandlerTick;

                int64_t left = 0;
                int64_t right = 0;
                for (Voice &voice : voices)
                    renderVoice(voice, left, right);
                const int16_t leftSample = clampSample(left);
                const int16_t rightSample = clampSample(right);
                output[frame * 2u] = leftSample;
                output[frame * 2u + 1u] = rightSample;
                ++debug.renderedFrames;
                if (leftSample != 0 || rightSample != 0)
                    ++debug.nonzeroFrames;

                voices.erase(
                    std::remove_if(voices.begin(), voices.end(),
                                   [](const Voice &voice)
                                   { return voice.dead; }),
                    voices.end());
            }
        }

        DebugSnapshot snapshot() const
        {
            DebugSnapshot snapshot = debug;
            snapshot.bankCount = banks.size();
            snapshot.activeHandlerCount = handlers.size();
            snapshot.activeVoiceCount = voices.size();
            for (const auto &[handle, bank] : banks)
            {
                (void)handle;
                snapshot.soundCount += bank->sounds.size();
                snapshot.decodedSampleCount += bank->decodedSamples.size();
            }
            return snapshot;
        }
    };

    Engine::Engine() : m_impl(std::make_unique<Impl>())
    {
    }

    Engine::~Engine() = default;

    bool Engine::loadBank(uint32_t bankHandle,
                          const uint8_t *container,
                          size_t containerSize)
    {
        if (bankHandle == 0u)
            return false;
        auto bank = std::make_unique<Bank>();
        if (!parseBank(bankHandle, container, containerSize, *bank))
        {
            std::lock_guard<std::mutex> lock(m_impl->mutex);
            ++m_impl->debug.rejectedBanks;
            return false;
        }

        std::lock_guard<std::mutex> lock(m_impl->mutex);
        m_impl->banks[bankHandle] = std::move(bank);
        return true;
    }

    void Engine::unloadBank(uint32_t bankHandle)
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        for (auto handler = m_impl->handlers.begin();
             handler != m_impl->handlers.end();)
        {
            if (handler->second.bankHandle == bankHandle)
                handler = m_impl->handlers.erase(handler);
            else
                ++handler;
        }
        m_impl->voices.erase(
            std::remove_if(m_impl->voices.begin(), m_impl->voices.end(),
                           [bankHandle](const Voice &voice)
                           { return voice.bankHandle == bankHandle; }),
            m_impl->voices.end());
        m_impl->banks.erase(bankHandle);
    }

    bool Engine::playSound(uint32_t bankHandle,
                           uint32_t soundId,
                           uint32_t soundHandle,
                           int32_t volume,
                           int32_t pan,
                           int32_t pitchModifier,
                           int32_t pitchBend)
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        return m_impl->addHandler(bankHandle,
                                  soundId,
                                  soundHandle,
                                  volume,
                                  pan,
                                  pitchModifier,
                                  pitchBend);
    }

    void Engine::stopSound(uint32_t soundHandle)
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        m_impl->stopSound(soundHandle);
    }

    void Engine::stopAllSounds()
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        for (Voice &voice : m_impl->voices)
        {
            voice.envelope.stop();
            voice.dead = true;
        }
        m_impl->voices.clear();
        m_impl->handlers.clear();
    }

    void Engine::setMasterVolume(uint32_t group, int32_t volume)
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        if (group >= m_impl->masterVolume.size())
            return;
        m_impl->masterVolume[group] = std::clamp(volume, 0, 1024);
        for (auto &[handle, handler] : m_impl->handlers)
        {
            (void)handle;
            if (handler.group == group)
                m_impl->recomputeHandler(handler);
        }
    }

    void Engine::setSoundParameters(uint32_t soundHandle,
                                    uint32_t mask,
                                    int32_t volume,
                                    int32_t pan,
                                    int32_t pitchModifier,
                                    int32_t pitchBend)
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        const auto found = m_impl->handlers.find(soundHandle);
        if (found == m_impl->handlers.end())
            return;
        Handler &handler = found->second;
        if ((mask & 0x01u) != 0u && volume != kVolumeDontChange)
            handler.applicationVolume = volume;
        if ((mask & 0x02u) != 0u)
        {
            if (pan == kPanReset)
                handler.applicationPan = handler.originalPan;
            else if (pan != kPanDontChange)
                handler.applicationPan = pan;
        }
        if ((mask & 0x04u) != 0u)
            handler.applicationPitchModifier = pitchModifier;
        if ((mask & 0x08u) != 0u)
            handler.applicationPitchBend = pitchBend;
        m_impl->recomputeHandler(handler);
    }

    bool Engine::soundActive(uint32_t soundHandle) const
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        return m_impl->handlers.find(soundHandle) != m_impl->handlers.end() ||
               m_impl->hasOwnerVoices(soundHandle);
    }

    void Engine::render(int16_t *stereoOutput, size_t frameCount)
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        m_impl->render(stereoOutput, frameCount);
    }

    DebugSnapshot Engine::debugSnapshot() const
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        return m_impl->snapshot();
    }
}
