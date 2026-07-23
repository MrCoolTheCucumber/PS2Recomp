#include "runtime/ps2_audio.h"
#include "runtime/ps2_memory.h"
#include "ps2_host_backend.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <deque>
#include <limits>
#include <vector>

namespace
{
    std::vector<uint8_t> buildWavFromPcm(const int16_t *pcm, size_t sampleCount, uint32_t sampleRate)
    {
        const uint32_t dataSize = static_cast<uint32_t>(sampleCount * 2);
        const uint32_t fileSize = 36 + dataSize;
        std::vector<uint8_t> wav(8 + fileSize);

        uint8_t *p = wav.data();
        p[0] = 'R';
        p[1] = 'I';
        p[2] = 'F';
        p[3] = 'F';
        p[4] = static_cast<uint8_t>(fileSize);
        p[5] = static_cast<uint8_t>(fileSize >> 8);
        p[6] = static_cast<uint8_t>(fileSize >> 16);
        p[7] = static_cast<uint8_t>(fileSize >> 24);
        p[8] = 'W';
        p[9] = 'A';
        p[10] = 'V';
        p[11] = 'E';
        p[12] = 'f';
        p[13] = 'm';
        p[14] = 't';
        p[15] = ' ';
        p[16] = 16;
        p[17] = 0;
        p[18] = 0;
        p[19] = 0;
        p[20] = 1;
        p[21] = 0;
        p[22] = 1;
        p[23] = 0;
        p[24] = static_cast<uint8_t>(sampleRate);
        p[25] = static_cast<uint8_t>(sampleRate >> 8);
        p[26] = static_cast<uint8_t>(sampleRate >> 16);
        p[27] = static_cast<uint8_t>(sampleRate >> 24);
        const uint32_t byteRate = sampleRate * 2;
        p[28] = static_cast<uint8_t>(byteRate);
        p[29] = static_cast<uint8_t>(byteRate >> 8);
        p[30] = static_cast<uint8_t>(byteRate >> 16);
        p[31] = static_cast<uint8_t>(byteRate >> 24);
        p[32] = 2;
        p[33] = 0;
        p[34] = 16;
        p[35] = 0;
        p[36] = 'd';
        p[37] = 'a';
        p[38] = 't';
        p[39] = 'a';
        p[40] = static_cast<uint8_t>(dataSize);
        p[41] = static_cast<uint8_t>(dataSize >> 8);
        p[42] = static_cast<uint8_t>(dataSize >> 16);
        p[43] = static_cast<uint8_t>(dataSize >> 24);
        std::memcpy(p + 44, pcm, dataSize);
        return wav;
    }
}

namespace ps2_vag
{
    bool decode(const uint8_t *data, uint32_t sizeBytes,
                std::vector<int16_t> &outPcm, uint32_t &outSampleRate);
}

struct PS2AudioBackend::Impl
{
    struct TrackedSound
    {
        Sound snd;
        uint32_t sampleKey;
    };

    struct SpuAdpcmStream
    {
        std::mutex mutex;
        std::array<ps2_spu_adpcm::DecoderState, 2> decoders{};
        std::array<std::deque<int16_t>, 2> channelPcm;
        std::deque<int16_t> interleavedPcm;
        AudioStream hostStream{};
        uint32_t channelBufferSize = 0;
        uint32_t sampleRate = 0;
        uint32_t channelCount = 0;
        bool opened = false;
        bool hostStreamLoaded = false;
        bool playing = false;
    };

    std::vector<TrackedSound> activeSounds;
    SpuAdpcmStream spuAdpcmStream;

    inline static std::atomic<SpuAdpcmStream *> activeCallbackStream{nullptr};

    static void flushChannelPcm(SpuAdpcmStream &stream)
    {
        if (stream.channelCount == 1u)
        {
            auto &mono = stream.channelPcm[0];
            while (!mono.empty())
            {
                stream.interleavedPcm.push_back(mono.front());
                mono.pop_front();
            }
        }
        else if (stream.channelCount == 2u)
        {
            auto &left = stream.channelPcm[0];
            auto &right = stream.channelPcm[1];
            while (!left.empty() && !right.empty())
            {
                stream.interleavedPcm.push_back(left.front());
                stream.interleavedPcm.push_back(right.front());
                left.pop_front();
                right.pop_front();
            }
        }

        if (stream.channelCount != 0u && stream.sampleRate != 0u)
        {
            const size_t maximumSamples =
                static_cast<size_t>(stream.sampleRate) * stream.channelCount * 2u;
            while (stream.interleavedPcm.size() > maximumSamples)
            {
                for (uint32_t channel = 0u;
                     channel < stream.channelCount && !stream.interleavedPcm.empty();
                     ++channel)
                {
                    stream.interleavedPcm.pop_front();
                }
            }
        }
    }

    static void streamCallback(void *bufferData, unsigned int frameCount)
    {
        SpuAdpcmStream *const stream =
            activeCallbackStream.load(std::memory_order_acquire);
        if (!stream || !bufferData)
            return;

        std::lock_guard<std::mutex> lock(stream->mutex);
        const uint32_t channels = stream->channelCount;
        if (channels == 0u || channels > 2u)
            return;

        int16_t *const output = static_cast<int16_t *>(bufferData);
        const size_t requestedSamples = static_cast<size_t>(frameCount) * channels;
        size_t written = 0u;
        while (written < requestedSamples && !stream->interleavedPcm.empty())
        {
            output[written++] = stream->interleavedPcm.front();
            stream->interleavedPcm.pop_front();
        }
        std::fill(output + written, output + requestedSamples, 0);
    }
};

PS2AudioBackend::PS2AudioBackend() : m_impl(std::make_unique<Impl>())
{
}

PS2AudioBackend::~PS2AudioBackend()
{
    if (m_impl)
        stopAll();
}

void PS2AudioBackend::onVagTransfer(const uint8_t *rdram, uint32_t srcAddr, uint32_t sizeBytes)
{
    if (!rdram || sizeBytes < 48)
        return;

    const uint32_t physAddr = srcAddr & PS2_RAM_MASK;
    if (physAddr + sizeBytes > PS2_RAM_SIZE)
        return;

    std::vector<int16_t> pcm;
    uint32_t sampleRate = 44100;
    if (!ps2_vag::decode(rdram + physAddr, sizeBytes, pcm, sampleRate))
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample sample;
    sample.pcm = std::move(pcm);
    sample.sampleRate = sampleRate;
    m_sampleBank[physAddr] = std::move(sample);
    m_mostRecentSampleKey = physAddr;
}

void PS2AudioBackend::onVagTransferFromBuffer(const uint8_t *data, uint32_t sizeBytes, uint32_t keyAddr)
{
    if (!data || sizeBytes < 48)
        return;

    std::vector<int16_t> pcm;
    uint32_t sampleRate = 44100;
    if (!ps2_vag::decode(data, sizeBytes, pcm, sampleRate))
        return;

    const uint32_t physAddr = keyAddr & PS2_RAM_MASK;
    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample sample;
    sample.pcm = std::move(pcm);
    sample.sampleRate = sampleRate;
    m_sampleBank[physAddr] = sample;
    m_mostRecentSampleKey = physAddr;
    m_loadOrderSamples.push_back(std::move(sample));
    m_loadOrderSampleKeys.push_back(physAddr);
    constexpr size_t kMaxLoadOrderSamples = 32;
    if (m_loadOrderSamples.size() > kMaxLoadOrderSamples)
    {
        m_loadOrderSamples.erase(m_loadOrderSamples.begin());
        m_loadOrderSampleKeys.erase(m_loadOrderSampleKeys.begin());
    }
}

namespace
{
    constexpr uint32_t LIBSD_CMD_SET_VOICE = 0x8010u;
    constexpr uint32_t SONY_989SND_SID = 0x00123456u;
    constexpr uint32_t SONY_989SND_OPEN_STREAM = 0x3Bu;
    constexpr uint32_t SONY_989SND_CLOSE_STREAM = 0x3Cu;
    constexpr uint32_t SONY_989SND_STOP_STREAM = 0x3Du;
    constexpr uint32_t SONY_989SND_START_STREAM = 0x3Eu;
    constexpr uint32_t SONY_989SND_SUBMIT_STREAM = 0x5Au;
}

void PS2AudioBackend::onSoundCommand(uint32_t sid, uint32_t rpcNum,
                                     const uint8_t *sendBuf, uint32_t sendSize,
                                     uint8_t *recvBuf, uint32_t recvSize)
{
    if (sid == SONY_989SND_SID)
    {
        handleSony989sndCommand(rpcNum, sendBuf, sendSize, recvBuf, recvSize);
        return;
    }

    if (sid != 0x80000701u)
        return;

    if ((rpcNum == LIBSD_CMD_SET_VOICE || (rpcNum & 0xFF00u) == 0x8100u) &&
        sendBuf && sendSize >= 20)
    {
        uint32_t sampleAddr = 0;
        uint32_t voiceIndex = 0xFFFFFFFFu;
        for (int vo = 4; vo >= 0 && voiceIndex == 0xFFFFFFFFu; vo -= 4)
        {
            if (vo < static_cast<int>(sendSize))
            {
                uint32_t v = 0;
                std::memcpy(&v, sendBuf + vo, sizeof(v));
                if (v < 24u)
                    voiceIndex = v;
            }
        }

        constexpr uint32_t kMinPlausibleAddr = 0x1000u;
        for (int off = 12; off <= 24 && sampleAddr == 0; off += 4)
        {
            if (sendSize >= static_cast<uint32_t>(off + 4))
            {
                uint32_t cand = 0;
                std::memcpy(&cand, sendBuf + off, sizeof(cand));
                if (cand >= kMinPlausibleAddr && (cand <= PS2_RAM_MASK || (cand & ~PS2_RAM_MASK) == 0))
                    sampleAddr = cand;
            }
        }
        if (sampleAddr == 0)
            sampleAddr = m_mostRecentSampleKey;

        float pitch = 1.0f;
        if (sendSize >= 12)
        {
            uint16_t pitchHalf = 0;
            std::memcpy(&pitchHalf, sendBuf + 8, sizeof(pitchHalf));
            if (pitchHalf != 0)
                pitch = 4096.0f / static_cast<float>(pitchHalf);
        }
        play(sampleAddr, pitch, 1.0f, voiceIndex);
    }
}

void PS2AudioBackend::destroySpuAdpcmHostStream()
{
    Impl::SpuAdpcmStream &stream = m_impl->spuAdpcmStream;
    AudioStream hostStream{};
    bool loaded = false;
    {
        std::lock_guard<std::mutex> lock(stream.mutex);
        hostStream = stream.hostStream;
        loaded = stream.hostStreamLoaded;
        stream.hostStreamLoaded = false;
        stream.playing = false;
        stream.hostStream = {};
    }

    Impl::SpuAdpcmStream *expected = &stream;
    (void)Impl::activeCallbackStream.compare_exchange_strong(
        expected, nullptr, std::memory_order_acq_rel);

#if !defined(PLATFORM_VITA)
    if (loaded && IsAudioDeviceReady())
    {
        SetAudioStreamCallback(hostStream, nullptr);
        StopAudioStream(hostStream);
        UnloadAudioStream(hostStream);
    }
#else
    (void)hostStream;
    (void)loaded;
#endif
}

void PS2AudioBackend::closeSpuAdpcmStream()
{
    destroySpuAdpcmHostStream();

    Impl::SpuAdpcmStream &stream = m_impl->spuAdpcmStream;
    std::lock_guard<std::mutex> lock(stream.mutex);
    stream.decoders = {};
    stream.channelPcm = {};
    stream.interleavedPcm.clear();
    stream.channelBufferSize = 0u;
    stream.sampleRate = 0u;
    stream.channelCount = 0u;
    stream.opened = false;
}

void PS2AudioBackend::handleSony989sndCommand(uint32_t rpcNum,
                                               const uint8_t *sendBuf,
                                               uint32_t sendSize,
                                               const uint8_t *streamData,
                                               uint32_t streamDataSize)
{
    Impl::SpuAdpcmStream &stream = m_impl->spuAdpcmStream;

    if (rpcNum == SONY_989SND_OPEN_STREAM)
    {
        closeSpuAdpcmStream();
        if (!sendBuf || sendSize < 6u * sizeof(uint32_t) ||
            !streamData || streamDataSize == 0u)
        {
            return;
        }

        std::array<uint32_t, 6> configuration{};
        std::memcpy(configuration.data(), sendBuf, sizeof(configuration));
        if (configuration[0] != streamDataSize || configuration[1] == 0u)
            return;

        std::lock_guard<std::mutex> lock(stream.mutex);
        stream.channelBufferSize = configuration[1];
        stream.opened = true;
        return;
    }

    if (rpcNum == SONY_989SND_CLOSE_STREAM)
    {
        closeSpuAdpcmStream();
        return;
    }

    if (rpcNum == SONY_989SND_STOP_STREAM)
    {
        AudioStream hostStream{};
        bool loaded = false;
        {
            std::lock_guard<std::mutex> lock(stream.mutex);
            hostStream = stream.hostStream;
            loaded = stream.hostStreamLoaded;
            stream.playing = false;
        }
#if !defined(PLATFORM_VITA)
        if (loaded && IsAudioDeviceReady())
            PauseAudioStream(hostStream);
#else
        (void)hostStream;
        (void)loaded;
#endif
        return;
    }

    if (rpcNum == SONY_989SND_SUBMIT_STREAM)
    {
        if (!sendBuf || sendSize < 2u * sizeof(uint32_t) ||
            !streamData || streamDataSize == 0u)
        {
            return;
        }

        std::array<uint32_t, 2> transfer{};
        std::memcpy(transfer.data(), sendBuf, sizeof(transfer));
        const uint32_t byteCount = transfer[0];
        const uint32_t destinationOffset = transfer[1];
        if (byteCount == 0u || byteCount > streamDataSize ||
            (byteCount % 16u) != 0u)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(stream.mutex);
        if (!stream.opened || stream.channelBufferSize == 0u)
            return;

        const uint32_t channel = destinationOffset / stream.channelBufferSize;
        const uint32_t channelOffset = destinationOffset % stream.channelBufferSize;
        if (channel >= stream.channelPcm.size() ||
            channelOffset > stream.channelBufferSize ||
            byteCount > stream.channelBufferSize - channelOffset)
        {
            return;
        }

        std::vector<int16_t> decoded;
        if (!ps2_spu_adpcm::decodeBlocks(streamData,
                                         byteCount,
                                         stream.decoders[channel],
                                         decoded))
        {
            return;
        }
        stream.channelPcm[channel].insert(stream.channelPcm[channel].end(),
                                          decoded.begin(), decoded.end());
        Impl::flushChannelPcm(stream);
        return;
    }

    if (rpcNum != SONY_989SND_START_STREAM ||
        !sendBuf || sendSize < 5u * sizeof(uint32_t))
    {
        return;
    }

    std::array<uint32_t, 5> playback{};
    std::memcpy(playback.data(), sendBuf, sizeof(playback));
    const uint32_t sampleRate = playback[3];
    const uint32_t channelCount = playback[4];
    if (sampleRate == 0u || sampleRate > 192000u ||
        channelCount == 0u || channelCount > 2u)
    {
        return;
    }

    destroySpuAdpcmHostStream();
    {
        std::lock_guard<std::mutex> lock(stream.mutex);
        if (!stream.opened)
            return;
        stream.sampleRate = sampleRate;
        stream.channelCount = channelCount;
        Impl::flushChannelPcm(stream);
    }

#if !defined(PLATFORM_VITA)
    if (!m_audioReady || !IsAudioDeviceReady())
        return;

    AudioStream hostStream = LoadAudioStream(sampleRate, 16u, channelCount);
    if (!IsAudioStreamValid(hostStream))
        return;

    {
        std::lock_guard<std::mutex> lock(stream.mutex);
        stream.hostStream = hostStream;
        stream.hostStreamLoaded = true;
        stream.playing = true;
    }
    Impl::activeCallbackStream.store(&stream, std::memory_order_release);
    SetAudioStreamCallback(hostStream, &Impl::streamCallback);
    PlayAudioStream(hostStream);
#endif
}

void PS2AudioBackend::play(uint32_t sampleAddr, float pitch, float volume, uint32_t voiceIndex)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample *sampleToPlay = nullptr;
    uint32_t sampleKey = 0;

    auto it = m_sampleBank.find(sampleAddr & PS2_RAM_MASK);
    if (it != m_sampleBank.end())
    {
        sampleToPlay = &it->second;
        sampleKey = it->first;
    }
    else if (voiceIndex != 0xFFFFFFFFu &&
             voiceIndex < m_loadOrderSamples.size() &&
             voiceIndex < m_loadOrderSampleKeys.size())
    {
        sampleToPlay = &m_loadOrderSamples[voiceIndex];
        sampleKey = m_loadOrderSampleKeys[voiceIndex];
    }
    else
    {
        it = m_sampleBank.find(m_mostRecentSampleKey);
        if (it == m_sampleBank.end())
            return;
        sampleToPlay = &it->second;
        sampleKey = it->first;
    }
    if (!sampleToPlay || sampleToPlay->pcm.empty())
        return;

    const bool isBgm = (sampleToPlay->pcm.size() > static_cast<size_t>(sampleToPlay->sampleRate * 5));
    playDecodedSample(sampleKey, *sampleToPlay, pitch, volume, isBgm);
}

void PS2AudioBackend::pruneFinishedSounds()
{
#if defined(PLATFORM_VITA)
    return;
#else
    auto &sounds = m_impl->activeSounds;
    auto it = sounds.begin();
    while (it != sounds.end())
    {
        if (!IsSoundPlaying(it->snd))
        {
            UnloadSound(it->snd);
            it = sounds.erase(it);
        }
        else
        {
            ++it;
        }
    }
#endif
}

void PS2AudioBackend::playDecodedSample(uint32_t sampleKey, DecodedSample &sample, float pitch, float volume,
                                        bool isBgm)
{
#if defined(PLATFORM_VITA)
    (void)sampleKey;
    (void)sample;
    (void)pitch;
    (void)volume;
    (void)isBgm;
    return;
#else
    if (!m_audioReady || sample.pcm.empty())
        return;

    pruneFinishedSounds();

    for (const auto &t : m_impl->activeSounds)
    {
        if (t.sampleKey == sampleKey && IsSoundPlaying(t.snd))
            return;
    }

    auto &sounds = m_impl->activeSounds;
    if (isBgm)
    {
        for (auto it = sounds.begin(); it != sounds.end();)
        {
            if (IsSoundPlaying(it->snd))
            {
                StopSound(it->snd);
                UnloadSound(it->snd);
                it = sounds.erase(it);
            }
            else
                ++it;
        }
    }

    constexpr int kMaxConcurrentSounds = 4;
    while (static_cast<int>(sounds.size()) >= kMaxConcurrentSounds)
    {
        StopSound(sounds.front().snd);
        UnloadSound(sounds.front().snd);
        sounds.erase(sounds.begin());
    }

    std::vector<uint8_t> wav = buildWavFromPcm(sample.pcm.data(), sample.pcm.size(), sample.sampleRate);
    Wave wave = LoadWaveFromMemory(".wav", wav.data(), static_cast<int>(wav.size()));
    if (wave.frameCount <= 0)
        return;
    Sound snd = LoadSoundFromWave(wave);
    UnloadWave(wave);
    SetSoundPitch(snd, pitch);
    SetSoundVolume(snd, volume);
    m_impl->activeSounds.push_back({snd, sampleKey});
    PlaySound(snd);
#endif
}

void PS2AudioBackend::stop(uint32_t voiceId)
{
    (void)voiceId;
}

void PS2AudioBackend::stopAll()
{
    closeSpuAdpcmStream();

    std::lock_guard<std::mutex> lock(m_mutex);
#if defined(PLATFORM_VITA)
    return;
#else
    for (auto &t : m_impl->activeSounds)
    {
        StopSound(t.snd);
        UnloadSound(t.snd);
    }
    m_impl->activeSounds.clear();
#endif
}
