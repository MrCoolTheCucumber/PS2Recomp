#include "MiniTest.h"
#include "ps2_989snd.h"
#include "runtime/ps2_audio.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace
{
    void writeU16(std::vector<uint8_t> &bytes, size_t offset, uint16_t value)
    {
        bytes[offset] = static_cast<uint8_t>(value);
        bytes[offset + 1u] = static_cast<uint8_t>(value >> 8u);
    }

    void writeU32(std::vector<uint8_t> &bytes, size_t offset, uint32_t value)
    {
        bytes[offset] = static_cast<uint8_t>(value);
        bytes[offset + 1u] = static_cast<uint8_t>(value >> 8u);
        bytes[offset + 2u] = static_cast<uint8_t>(value >> 16u);
        bytes[offset + 3u] = static_cast<uint8_t>(value >> 24u);
    }

    std::vector<uint8_t> buildSyntheticSfxBank()
    {
        constexpr size_t kMetadataOffset = 0x18u;
        constexpr size_t kMetadataSize = 0x68u;
        constexpr size_t kSamplesOffset = 0x80u;
        constexpr size_t kSamplesSize = 0x10u;
        std::vector<uint8_t> container(kSamplesOffset + kSamplesSize, 0u);

        writeU32(container, 0x00u, 3u);
        writeU32(container, 0x04u, 2u);
        writeU32(container, 0x08u, kMetadataOffset);
        writeU32(container, 0x0Cu, kMetadataSize);
        writeU32(container, 0x10u, kSamplesOffset);
        writeU32(container, 0x14u, kSamplesSize);

        const size_t metadata = kMetadataOffset;
        writeU32(container, metadata + 0x00u, 0x6B6C4253u); // "SBlk"
        writeU32(container, metadata + 0x04u, 1u);
        writeU16(container, metadata + 0x16u, 1u);
        writeU16(container, metadata + 0x18u, 1u);
        writeU16(container, metadata + 0x1Au, 1u);
        writeU32(container, metadata + 0x1Cu, 0x34u);
        writeU32(container, metadata + 0x20u, 0x40u);
        writeU32(container, metadata + 0x28u, kSamplesSize);
        writeU32(container, metadata + 0x2Cu, kSamplesSize);

        const size_t sound = metadata + 0x34u;
        container[sound + 0u] = 127u;
        container[sound + 1u] = 0u;
        container[sound + 4u] = 1u;
        writeU32(container, sound + 8u, 0u);

        const size_t grain = metadata + 0x40u;
        writeU32(container, grain + 0u, 1u); // Tone.
        writeU32(container, grain + 4u, 0u);
        container[grain + 8u + 1u] = 127u;
        container[grain + 8u + 2u] = static_cast<uint8_t>(-60);
        writeU16(container, grain + 8u + 10u, 0x80FFu);
        writeU16(container, grain + 8u + 12u, 0x9FC0u);
        writeU32(container, grain + 8u + 16u, 0u);

        container[kSamplesOffset + 0u] = 0x00u;
        container[kSamplesOffset + 1u] = 0x07u; // Loop start, repeat, and end.
        std::fill(container.begin() + kSamplesOffset + 2u,
                  container.end(),
                  0x77u);
        return container;
    }
}

struct PS2AudioBackendTestAccess
{
    static void renderFrames(
        PS2AudioBackend &backend,
        void *bufferData,
        unsigned int frameCount)
    {
        backend.renderSpuAdpcmFramesForTesting(
            bufferData, frameCount);
    }

    static void renderSfxFrames(
        PS2AudioBackend &backend,
        void *bufferData,
        unsigned int frameCount)
    {
        backend.renderSony989sndFramesForTesting(
            bufferData, frameCount);
    }
};

void register_ps2_audio_tests()
{
    MiniTest::Case("PS2Audio", [](TestCase &tc)
                   {
        tc.Run("raw SPU ADPCM blocks decode in nibble order", [](TestCase &t)
               {
                   std::array<uint8_t, 16> block{};
                   block[0] = 0x0Cu; // Predictor 0, shift 12.
                   block[1] = 0x06u; // Flags do not replace encoded samples.
                   for (size_t index = 2u; index < block.size(); ++index)
                       block[index] = 0xF1u;

                   ps2_spu_adpcm::DecoderState state{};
                   std::vector<int16_t> pcm;
                   t.IsTrue(ps2_spu_adpcm::decodeBlocks(
                                block.data(), block.size(), state, pcm),
                            "one complete SPU ADPCM block should decode");
                   t.Equals(pcm.size(), static_cast<size_t>(28u),
                            "one SPU ADPCM block should produce 28 samples");
                   for (size_t index = 0u; index < pcm.size(); ++index)
                   {
                       t.Equals(pcm[index],
                                static_cast<int16_t>((index & 1u) == 0u ? 1 : -1),
                                "SPU ADPCM should decode the low nibble before the high nibble");
                   }
               });

        tc.Run("raw SPU ADPCM predictor history crosses submissions", [](TestCase &t)
               {
                   std::array<uint8_t, 16> first{};
                   first[0] = 0x0Cu; // Predictor 0, shift 12.
                   std::fill(first.begin() + 2, first.end(), 0x11u);

                   std::array<uint8_t, 16> second{};
                   second[0] = 0x1Cu; // Predictor 1, shift 12.

                   ps2_spu_adpcm::DecoderState state{};
                   std::vector<int16_t> pcm;
                   t.IsTrue(ps2_spu_adpcm::decodeBlocks(
                                first.data(), first.size(), state, pcm),
                            "the first submission should decode");
                   t.IsTrue(ps2_spu_adpcm::decodeBlocks(
                                second.data(), second.size(), state, pcm),
                            "the second submission should decode with retained history");
                   t.Equals(pcm.size(), static_cast<size_t>(56u),
                            "two submitted blocks should append 56 samples");
                   t.Equals(pcm[28u], static_cast<int16_t>(1),
                            "the second block should use the first block's predictor history");
                   t.Equals(state.previous1, static_cast<int32_t>(1),
                            "decoder history should retain the latest clamped sample");
               });

        tc.Run("raw SPU ADPCM rejects partial blocks without changing state", [](TestCase &t)
               {
                   std::array<uint8_t, 15> truncated{};
                   ps2_spu_adpcm::DecoderState state{123, -456};
                   std::vector<int16_t> pcm{7};

                   t.IsFalse(ps2_spu_adpcm::decodeBlocks(
                                 truncated.data(), truncated.size(), state, pcm),
                             "a partial 16-byte block must be rejected");
                   t.Equals(state.previous1, static_cast<int32_t>(123),
                            "rejected input should preserve the first predictor sample");
                   t.Equals(state.previous2, static_cast<int32_t>(-456),
                            "rejected input should preserve the second predictor sample");
                   t.Equals(pcm.size(), static_cast<size_t>(1u),
                            "rejected input should not append PCM");
               });

        tc.Run("989snd parses and renders a looping version-one SBlk tone", [](TestCase &t)
               {
                   constexpr uint32_t kBankHandle = 0x00100010u;
                   constexpr uint32_t kSoundHandle = 0x00100020u;
                   std::vector<uint8_t> bank = buildSyntheticSfxBank();
                   ps2_989snd::Engine engine;

                   t.IsTrue(engine.loadBank(kBankHandle, bank.data(), bank.size()),
                            "a bounded v1 SBlk container should load");
                   ps2_989snd::DebugSnapshot snapshot = engine.debugSnapshot();
                   t.Equals(snapshot.bankCount, size_t{1u},
                            "the parsed bank should be retained by its opaque handle");
                   t.Equals(snapshot.soundCount, size_t{1u},
                            "the SBlk sound table should expose one sound");
                   t.Equals(snapshot.decodedSampleCount, size_t{1u},
                            "the tone's headerless SPU sample should be decoded once");

                   t.IsTrue(engine.playSound(kBankHandle,
                                             0u,
                                             kSoundHandle,
                                             0x400,
                                             0,
                                             0,
                                             0),
                            "a valid SBlk sound should create its requested host handle");
                   std::array<int16_t, 512u * 2u> output{};
                   engine.render(output.data(), 512u);
                   t.IsTrue(std::any_of(output.begin(), output.end(),
                                        [](int16_t sample)
                                        { return sample != 0; }),
                            "the looping tone should render nonzero stereo PCM");
                   snapshot = engine.debugSnapshot();
                   t.Equals(snapshot.activeVoiceCount, size_t{1u},
                            "a repeating ADPCM sample should remain active past its end flag");
                   t.IsTrue(snapshot.nonzeroFrames > 28u,
                            "loop flags should produce audio beyond one encoded block");

                   engine.setMasterVolume(0u, 0);
                   std::array<int16_t, 32u * 2u> muted{};
                   muted.fill(1);
                   engine.render(muted.data(), 32u);
                   t.IsTrue(std::all_of(muted.begin(), muted.end(),
                                       [](int16_t sample)
                                       { return sample == 0; }),
                            "the sound group master volume should gate active voices");

                   engine.stopAllSounds();
                   t.IsFalse(engine.soundActive(kSoundHandle),
                             "stop-all should retire both handlers and voices");
                   engine.unloadBank(kBankHandle);
                   t.Equals(engine.debugSnapshot().bankCount, size_t{0u},
                            "unload should release decoded bank and sample state");
               });

        tc.Run("989snd rejects malformed bank metadata before publishing state", [](TestCase &t)
               {
                   std::vector<uint8_t> bank = buildSyntheticSfxBank();
                   bank[0x18u + 0x04u] = 2u;
                   ps2_989snd::Engine engine;
                   t.IsFalse(engine.loadBank(0x00100010u, bank.data(), bank.size()),
                             "a v2 metadata block must not be decoded as the v1 layout");
                   const ps2_989snd::DebugSnapshot snapshot = engine.debugSnapshot();
                   t.Equals(snapshot.bankCount, size_t{0u},
                            "a rejected bank should not leave partial parser state");
                   t.Equals(snapshot.rejectedBanks, uint64_t{1u},
                            "malformed bank diagnostics should be countable");
               });

        tc.Run("989snd backend routes bank and play commands to the mixer", [](TestCase &t)
               {
                   constexpr uint32_t kStreamingSid = 0x00123456u;
                   constexpr uint32_t kBankHandle = 0x00100010u;
                   uint32_t soundHandle = 0x00100020u;
                   std::vector<uint8_t> bank = buildSyntheticSfxBank();
                   PS2AudioBackend backend;

                   backend.onSoundBankLoaded(
                       kStreamingSid,
                       kBankHandle,
                       bank.data(),
                       bank.size());
                   const std::array<uint32_t, 6u> play{
                       kBankHandle,
                       0u,
                       0x400u,
                       0u,
                       0u,
                       0u,
                   };
                   backend.onSoundCommand(
                       kStreamingSid,
                       0x11u,
                       reinterpret_cast<const uint8_t *>(play.data()),
                       sizeof(play),
                       reinterpret_cast<uint8_t *>(&soundHandle),
                       sizeof(soundHandle));

                   std::array<int16_t, 512u * 2u> output{};
                   PS2AudioBackendTestAccess::renderSfxFrames(
                       backend, output.data(), 512u);
                   t.IsTrue(std::any_of(output.begin(), output.end(),
                                        [](int16_t sample)
                                        { return sample != 0; }),
                            "the runtime command bridge should produce stereo PCM");

                   PS2AudioStreamDebugSnapshot snapshot =
                       backend.streamDebugSnapshot();
                   t.Equals(snapshot.sfxBankCount, size_t{1u},
                            "loaded 989snd banks should be visible in diagnostics");
                   t.Equals(snapshot.sfxActiveHandlerCount, size_t{1u},
                            "the IOP-provided sound handle should remain active");
                   t.IsTrue(snapshot.sfxNonzeroFrames > 0u,
                            "diagnostics should prove that the mixer produced audio");

                   const std::array<uint32_t, 6u> mute{
                       soundHandle,
                       0x01u,
                       0u,
                       0u,
                       0u,
                       0u,
                   };
                   backend.onSoundCommand(
                       kStreamingSid,
                       0x21u,
                       reinterpret_cast<const uint8_t *>(mute.data()),
                       sizeof(mute),
                       nullptr,
                       0u);
                   std::array<int16_t, 32u * 2u> muted{};
                   muted.fill(1);
                   PS2AudioBackendTestAccess::renderSfxFrames(
                       backend, muted.data(), 32u);
                   t.IsTrue(std::all_of(muted.begin(), muted.end(),
                                        [](int16_t sample)
                                        { return sample == 0; }),
                            "command 0x21 volume updates should reach active voices");

                   backend.onSoundBankUnloaded(kStreamingSid, kBankHandle);
                   snapshot = backend.streamDebugSnapshot();
                   t.Equals(snapshot.sfxBankCount, size_t{0u},
                            "the host unload callback should release mixer state");
                   t.Equals(snapshot.sfxActiveHandlerCount, size_t{0u},
                            "unloading a bank should stop its active handlers");
               });

        tc.Run("stream diagnostics count produced consumed and zero-filled frames", [](TestCase &t)
               {
                   constexpr uint32_t kStreamingSid = 0x00123456u;
                   constexpr uint32_t kChannelBufferSize = 128u;
                   std::array<uint8_t, kChannelBufferSize * 2u> workArea{};
                   const std::array<uint32_t, 6> open{
                       static_cast<uint32_t>(workArea.size()),
                       kChannelBufferSize,
                       static_cast<uint32_t>(workArea.size()),
                       0u,
                       5u,
                       3u,
                   };

                   PS2AudioBackend backend;
                   backend.onSoundCommand(
                       kStreamingSid,
                       0x3Bu,
                       reinterpret_cast<const uint8_t *>(open.data()),
                       sizeof(open),
                       workArea.data(),
                       static_cast<uint32_t>(workArea.size()));

                   std::array<uint8_t, 16u> encoded{};
                   encoded[0] = 0x0Cu;
                   std::fill(
                       encoded.begin() + 2u,
                       encoded.end(),
                       0x11u);
                   for (uint32_t channel = 0u; channel < 2u; ++channel)
                   {
                       const std::array<uint32_t, 2> submit{
                           static_cast<uint32_t>(encoded.size()),
                           channel * kChannelBufferSize,
                       };
                       backend.onSoundCommand(
                           kStreamingSid,
                           0x5Au,
                           reinterpret_cast<const uint8_t *>(
                               submit.data()),
                           sizeof(submit),
                           encoded.data(),
                           static_cast<uint32_t>(encoded.size()));
                   }

                   const std::array<uint32_t, 5> start{
                       0u,
                       0u,
                       0u,
                       44'100u,
                       2u,
                   };
                   backend.onSoundCommand(
                       kStreamingSid,
                       0x3Eu,
                       reinterpret_cast<const uint8_t *>(start.data()),
                       sizeof(start),
                       nullptr,
                       0u);

                   PS2AudioStreamDebugSnapshot snapshot =
                       backend.streamDebugSnapshot();
                   t.Equals(
                       snapshot.producedFrames,
                       uint64_t{28u},
                       "one block per stereo channel should produce 28 complete frames");
                   t.Equals(
                       snapshot.queuedSamples,
                       static_cast<size_t>(56u),
                       "the producer should queue 28 interleaved stereo frames");

                   std::array<int16_t, 32u * 2u> output{};
                   output.fill(static_cast<int16_t>(0x1234));
                   PS2AudioBackendTestAccess::renderFrames(
                       backend,
                       output.data(),
                       32u);

                   snapshot = backend.streamDebugSnapshot();
                   t.Equals(
                       snapshot.requestedFrames,
                       uint64_t{32u},
                       "the callback request should be counted in sample frames");
                   t.Equals(
                       snapshot.consumedFrames,
                       uint64_t{28u},
                       "the callback should consume every produced frame");
                   t.Equals(
                       snapshot.zeroFilledFrames,
                       uint64_t{4u},
                       "the callback should count the four-frame underrun");
                   t.Equals(
                       snapshot.queuedSamples,
                       static_cast<size_t>(0u),
                       "the callback should drain the interleaved queue");
                   for (size_t sample = 56u;
                        sample < output.size();
                        ++sample)
                   {
                       t.Equals(
                           output[sample],
                           static_cast<int16_t>(0),
                           "every sample in an underrun frame should be zero-filled");
                   }
               });
    });
}
