#include "MiniTest.h"
#include "runtime/ps2_audio.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

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
