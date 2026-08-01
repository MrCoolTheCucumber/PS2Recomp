#include "MiniTest.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_gs_memory.h"
#include "runtime/ps2_gs_vulkan.h"
#include "runtime/ps2_gs_vulkan_backend.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
    struct PsmSpec
    {
        GSMem::PixelStorageMode psm;
        uint32_t pageWidth;
        uint32_t pageHeight;
        const char *name;
    };

    constexpr std::array<PsmSpec, 13> kPsmSpecs{{
        {GSMem::C32, 64u, 32u, "C32"},
        {GSMem::C24, 64u, 32u, "C24"},
        {GSMem::C16, 64u, 64u, "C16"},
        {GSMem::C16S, 64u, 64u, "C16S"},
        {GSMem::P8, 128u, 64u, "P8"},
        {GSMem::P4, 128u, 128u, "P4"},
        {GSMem::P8H, 64u, 32u, "P8H"},
        {GSMem::P4HL, 64u, 32u, "P4HL"},
        {GSMem::P4HH, 64u, 32u, "P4HH"},
        {GSMem::Z32, 64u, 32u, "Z32"},
        {GSMem::Z24, 64u, 32u, "Z24"},
        {GSMem::Z16, 64u, 64u, "Z16"},
        {GSMem::Z16S, 64u, 64u, "Z16S"},
    }};

    uint32_t nextRandom(uint32_t &state)
    {
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        return state;
    }

    std::vector<uint8_t> makeVramPattern(uint32_t seed)
    {
        std::vector<uint8_t> pattern(GS_VULKAN_VRAM_SIZE);
        uint32_t state = seed;
        for (uint8_t &byte : pattern)
        {
            nextRandom(state);
            byte = static_cast<uint8_t>(state >> 24u);
        }
        return pattern;
    }

    uint32_t cpuReadPixel(
        GSMem::PixelStorageMode psm, uint8_t *vram,
        uint32_t bp, uint32_t bw, uint32_t x, uint32_t y)
    {
        switch (psm)
        {
        case GSMem::C32:
            return GSMem::ReadCT32(vram, bp, bw, x, y);
        case GSMem::C24:
            return GSMem::ReadCT24(vram, bp, bw, x, y);
        case GSMem::C16:
            return GSMem::ReadCT16(vram, bp, bw, x, y);
        case GSMem::C16S:
            return GSMem::ReadCT16S(vram, bp, bw, x, y);
        case GSMem::P8:
            return GSMem::ReadP8(vram, bp, bw, x, y);
        case GSMem::P4:
            return GSMem::ReadP4(vram, bp, bw, x, y);
        case GSMem::P8H:
            return GSMem::ReadP8H(vram, bp, bw, x, y);
        case GSMem::P4HL:
            return GSMem::ReadP4HL(vram, bp, bw, x, y);
        case GSMem::P4HH:
            return GSMem::ReadP4HH(vram, bp, bw, x, y);
        case GSMem::Z32:
            return GSMem::ReadZ32(vram, bp, bw, x, y);
        case GSMem::Z24:
            return GSMem::ReadZ24(vram, bp, bw, x, y);
        case GSMem::Z16:
            return GSMem::ReadZ16(vram, bp, bw, x, y);
        case GSMem::Z16S:
            return GSMem::ReadZ16S(vram, bp, bw, x, y);
        default:
            return 0u;
        }
    }

    void cpuWritePixel(
        GSMem::PixelStorageMode psm, uint8_t *vram,
        uint32_t bp, uint32_t bw, uint32_t x, uint32_t y,
        uint32_t value)
    {
        switch (psm)
        {
        case GSMem::C32:
            GSMem::WriteCT32(vram, bp, bw, x, y, value);
            break;
        case GSMem::C24:
            GSMem::WriteCT24(vram, bp, bw, x, y, value);
            break;
        case GSMem::C16:
            GSMem::WriteCT16(vram, bp, bw, x, y, value);
            break;
        case GSMem::C16S:
            GSMem::WriteCT16S(vram, bp, bw, x, y, value);
            break;
        case GSMem::P8:
            GSMem::WriteP8(vram, bp, bw, x, y, value);
            break;
        case GSMem::P4:
            GSMem::WriteP4(vram, bp, bw, x, y, value);
            break;
        case GSMem::P8H:
            GSMem::WriteP8H(vram, bp, bw, x, y, value);
            break;
        case GSMem::P4HL:
            GSMem::WriteP4HL(vram, bp, bw, x, y, value);
            break;
        case GSMem::P4HH:
            GSMem::WriteP4HH(vram, bp, bw, x, y, value);
            break;
        case GSMem::Z32:
            GSMem::WriteZ32(vram, bp, bw, x, y, value);
            break;
        case GSMem::Z24:
            GSMem::WriteZ24(vram, bp, bw, x, y, value);
            break;
        case GSMem::Z16:
            GSMem::WriteZ16(vram, bp, bw, x, y, value);
            break;
        case GSMem::Z16S:
            GSMem::WriteZ16S(vram, bp, bw, x, y, value);
            break;
        default:
            break;
        }
    }

    std::vector<GsVulkanMemoryCase> makeMemoryReadCorpus()
    {
        std::vector<GsVulkanMemoryCase> cases;
        cases.reserve(61952u);
        for (const PsmSpec &spec : kPsmSpecs)
        {
            const uint32_t rawPsm =
                static_cast<uint32_t>(spec.psm);
            uint32_t ordinal = 0u;
            for (uint32_t y = 0u; y < spec.pageHeight; ++y)
            {
                for (uint32_t x = 0u; x < spec.pageWidth;
                     ++x, ++ordinal)
                {
                    const uint32_t block =
                        (ordinal * 17u + rawPsm) & 31u;
                    const uint32_t bp = ordinal % 17u == 0u
                        ? 0x3FE0u + block
                        : block;
                    cases.push_back({
                        rawPsm,
                        bp,
                        1u + ((ordinal * 11u + rawPsm) % 63u),
                        x,
                        y,
                        GsVulkanMemoryOperation::Read,
                        0u,
                        0u,
                    });
                }
            }

            uint32_t randomState =
                0x9E3779B9u ^ (rawPsm * 0x45D9F3Bu);
            for (uint32_t sample = 0u; sample < 512u; ++sample)
            {
                uint32_t bp = nextRandom(randomState) & 0x3FFFu;
                if ((sample & 3u) == 0u)
                    bp = 0x3FE0u + (bp & 31u);
                const uint32_t bw =
                    1u + (nextRandom(randomState) % 63u);
                const uint32_t x = nextRandom(randomState) & 0x7FFu;
                const uint32_t y = nextRandom(randomState) & 0x7FFu;
                cases.push_back({
                    rawPsm,
                    bp,
                    bw,
                    x,
                    y,
                    GsVulkanMemoryOperation::Read,
                    0u,
                    0u,
                });
            }
        }
        return cases;
    }

    std::vector<uint32_t> makeAddressTokenPattern(
        uint32_t bitWidth, uint32_t chunkOffset)
    {
        std::vector<uint32_t> words(
            GS_VULKAN_VRAM_SIZE / sizeof(uint32_t));
        const uint32_t mask = bitWidth == 32u
            ? 0xFFFFFFFFu
            : (1u << bitWidth) - 1u;
        const uint32_t laneStride = bitWidth == 24u
            ? 32u
            : bitWidth;
        for (uint32_t wordIndex = 0u;
             wordIndex < words.size(); ++wordIndex)
        {
            uint32_t encoded = 0u;
            for (uint32_t bitShift = 0u; bitShift < 32u;
                 bitShift += laneStride)
            {
                const uint32_t addressToken =
                    (wordIndex << 5u) | bitShift;
                encoded |=
                    ((addressToken >> chunkOffset) & mask) <<
                    bitShift;
            }
            words[wordIndex] = encoded;
        }
        return words;
    }

    std::string describeMemoryCase(
        const GsVulkanMemoryCase &memoryCase, size_t index)
    {
        std::ostringstream message;
        message << "case " << index
                << " psm=0x" << std::hex
                << memoryCase.pixelStorageMode
                << " bp=0x" << memoryCase.baseBlock
                << " bw=0x" << memoryCase.bufferWidth
                << " x=0x" << memoryCase.x
                << " y=0x" << memoryCase.y;
        return message.str();
    }

    GsDrawCommand makeCt32SpriteCommand(
        uint64_t sequence,
        uint32_t framebufferPage,
        uint8_t framebufferWidth,
        GSScissorReg scissor,
        GSXYOffsetReg xyoffset,
        uint16_t rawX0,
        uint16_t rawY0,
        uint16_t rawX1,
        uint16_t rawY1,
        uint32_t rgba)
    {
        GSContext context{};
        context.frame = {
            framebufferPage, framebufferWidth,
            GS_PSM_CT32, 0u};
        context.scissor = scissor;
        context.xyoffset = xyoffset;
        context.zbuf = {0u, GS_PSM_Z32, true};
        context.test = 0x30000u; // ZTE=1, ZTST=ALWAYS, ZMSK=1.

        GSPrimReg primitive{};
        primitive.type = GS_PRIM_SPRITE;
        std::array<GSVertex, 2> vertices{};
        vertices[0].x12_4 = rawX0;
        vertices[0].y12_4 = rawY0;
        vertices[0].r = 0x11u;
        vertices[0].g = 0x22u;
        vertices[0].b = 0x33u;
        vertices[0].a = 0x44u;
        vertices[1].x12_4 = rawX1;
        vertices[1].y12_4 = rawY1;
        vertices[1].r = static_cast<uint8_t>(rgba);
        vertices[1].g = static_cast<uint8_t>(rgba >> 8u);
        vertices[1].b = static_cast<uint8_t>(rgba >> 16u);
        vertices[1].a = static_cast<uint8_t>(rgba >> 24u);
        return buildGsDrawCommand(
            sequence, primitive, context,
            std::span<const GSVertex>(vertices),
            GsDrawGlobalState{});
    }

    void applyCt32SpriteCpu(
        std::vector<uint8_t> &vram,
        const GsVulkanCt32Sprite &sprite)
    {
        for (uint32_t y = sprite.y0; y < sprite.y1; ++y)
        {
            for (uint32_t x = sprite.x0; x < sprite.x1; ++x)
            {
                GSMem::WriteCT32(
                    vram.data(),
                    sprite.framebufferBaseBlock,
                    sprite.framebufferWidth,
                    x, y, sprite.rgba);
            }
        }
    }

    uint64_t packXyz2(
        uint16_t x12_4, uint16_t y12_4, uint32_t z = 0u)
    {
        return static_cast<uint64_t>(x12_4) |
               (static_cast<uint64_t>(y12_4) << 16u) |
               (static_cast<uint64_t>(z) << 32u);
    }

    void configureFlatCt32Draws(
        GS &gs,
        uint32_t framebufferPage = 3u,
        uint32_t framebufferWidth = 2u)
    {
        const uint64_t frame =
            static_cast<uint64_t>(framebufferPage) |
            (static_cast<uint64_t>(framebufferWidth) << 16u) |
            (static_cast<uint64_t>(GS_PSM_CT32) << 24u);
        const uint64_t scissor =
            (127ull << 16u) |
            (63ull << 48u);
        gs.writeRegister(GS_REG_FRAME_1, frame);
        gs.writeRegister(GS_REG_ZBUF_1, 1ull << 32u);
        gs.writeRegister(GS_REG_SCISSOR_1, scissor);
        gs.writeRegister(GS_REG_XYOFFSET_1, 0u);
        gs.writeRegister(GS_REG_TEST_1, 0x30000u);
        gs.writeRegister(GS_REG_FBA_1, 0u);
        gs.writeRegister(GS_REG_SCANMSK, 0u);
        gs.writeRegister(GS_REG_DTHE, 0u);
    }

    void drawFlatCt32Sprite(
        GS &gs,
        uint16_t x0,
        uint16_t y0,
        uint16_t x1,
        uint16_t y1,
        uint32_t rgba)
    {
        gs.writeRegister(
            GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_SPRITE));
        gs.writeRegister(GS_REG_RGBAQ, rgba);
        gs.writeRegister(GS_REG_XYZ2, packXyz2(x0, y0));
        gs.writeRegister(GS_REG_XYZ2, packXyz2(x1, y1));
    }

    void drawFlatCt32Point(
        GS &gs, uint16_t x, uint16_t y, uint32_t rgba)
    {
        gs.writeRegister(
            GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT));
        gs.writeRegister(GS_REG_RGBAQ, rgba);
        gs.writeRegister(GS_REG_XYZ2, packXyz2(x, y));
    }

    void uploadCt32Pixels(
        GS &gs,
        uint32_t destinationBlock,
        uint32_t destinationWidth,
        uint16_t x,
        uint16_t y,
        std::span<const uint32_t> pixels)
    {
        const uint64_t bitbltbuf =
            (static_cast<uint64_t>(destinationBlock) << 32u) |
            (static_cast<uint64_t>(destinationWidth) << 48u) |
            (static_cast<uint64_t>(GS_PSM_CT32) << 56u);
        const uint64_t trxpos =
            (static_cast<uint64_t>(x) << 32u) |
            (static_cast<uint64_t>(y) << 48u);
        const uint64_t trxreg =
            static_cast<uint64_t>(pixels.size()) |
            (1ull << 32u);
        gs.uploadImageNative(
            bitbltbuf, trxpos, trxreg, 0u,
            reinterpret_cast<const uint8_t *>(pixels.data()),
            static_cast<uint32_t>(pixels.size_bytes()));
    }

    GsVulkanServiceConfig makeRendererServiceConfig(
        GsVulkanCapabilityReport &preflight)
    {
        GsVulkanServiceConfig config{};
        config.probe.enableValidation = true;
        preflight = probeGsVulkanCapabilities(config.probe);
        if (preflight.status ==
            GsVulkanProbeStatus::ValidationUnavailable)
        {
            config.probe.enableValidation = false;
            preflight = probeGsVulkanCapabilities(config.probe);
        }
        return config;
    }

    class FakeCt32Executor final : public IGsVulkanDrawExecutor
    {
    public:
        enum class Behavior
        {
            Exact,
            Noop,
            Fail,
            InvalidOutput,
        };

        explicit FakeCt32Executor(Behavior behavior_)
            : behavior(behavior_)
        {
            report.compiled = true;
            report.status = GsVulkanProbeStatus::Ready;
            report.selectedDeviceIndex = 0;
            report.devices.push_back({});
            report.devices[0].name = "deterministic fake executor";
            report.devices[0].suitable = true;
        }

        bool executeCt32Sprite(
            std::span<const uint8_t> input,
            const GsVulkanCt32Sprite &sprite,
            std::vector<uint8_t> &output,
            std::string *error) override
        {
            if (!isHealthy || behavior == Behavior::Fail)
            {
                ++serviceStatistics.spriteDrawsFailed;
                if (error)
                    *error = "injected CT32 executor failure";
                return false;
            }

            output.assign(input.begin(), input.end());
            if (behavior == Behavior::Exact)
                applyCt32SpriteCpu(output, sprite);
            else if (behavior == Behavior::InvalidOutput)
                output.resize(1u);
            ++serviceStatistics.spriteDrawsCompleted;
            if (error)
                error->clear();
            return true;
        }

        void shutdown() noexcept override
        {
            isHealthy = false;
        }

        GsVulkanCapabilityReport capabilities() const override
        {
            return report;
        }

        GsVulkanServiceStatistics statistics() const override
        {
            return serviceStatistics;
        }

        bool healthy() const override
        {
            return isHealthy;
        }

    private:
        Behavior behavior;
        GsVulkanCapabilityReport report;
        GsVulkanServiceStatistics serviceStatistics;
        bool isHealthy = true;
    };

    class ScopedArtifactDirectory final
    {
    public:
        ScopedArtifactDirectory()
        {
            static std::atomic<uint64_t> sequence{0u};
            const uint64_t unique =
                static_cast<uint64_t>(
                    std::chrono::steady_clock::now()
                        .time_since_epoch().count()) ^
                sequence.fetch_add(1u, std::memory_order_relaxed);
            path = std::filesystem::temp_directory_path() /
                   ("ps2-gs-vulkan-verify-" +
                    std::to_string(unique));
            std::filesystem::create_directories(path);
        }

        ~ScopedArtifactDirectory()
        {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }

        ScopedArtifactDirectory(const ScopedArtifactDirectory &) = delete;
        ScopedArtifactDirectory &operator=(
            const ScopedArtifactDirectory &) = delete;

        std::filesystem::path path;
    };
}

void register_ps2_gs_vulkan_tests()
{
    MiniTest::Case("PS2GSVulkan", [](TestCase &tc)
    {
        tc.Run("Vulkan capability metadata is stable", [](TestCase &t)
        {
            constexpr uint32_t version =
                (1u << 22u) | (4u << 12u) | 350u;
            t.Equals(gsVulkanVersionString(version),
                     std::string("1.4.350"),
                     "packed Vulkan versions should have a stable text form");
            t.Equals(
                gsVulkanProbeStatusName(GsVulkanProbeStatus::Ready),
                std::string_view("ready"),
                "probe statuses should have stable diagnostic names");
            t.Equals(
                gsVulkanDeviceKindName(GsVulkanDeviceKind::DiscreteGpu),
                std::string_view("discrete-gpu"),
                "device kinds should have stable diagnostic names");
            t.Equals(GS_VULKAN_NOOP_GROUP_COUNT, 16384u,
                     "the fixed 64-thread kernel should cover every 4 MiB word");
            t.Equals(GS_VULKAN_MAX_MEMORY_CASES, 65536u,
                     "memory conformance batches should have a fixed host bound");
        });

        tc.Run("CT32 sprite preparation applies exact GS bounds and eligibility", [](TestCase &t)
        {
            const GsDrawCommand command = makeCt32SpriteCommand(
                17u,
                511u,
                0u,
                {4u, 15u, 3u, 12u},
                {32u, 16u},
                289u, 209u,
                65u, 33u,
                0xD4C3B2A1u);

            GsVulkanCt32Sprite sprite{
                1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
            const GsBackendDecision decision =
                prepareGsVulkanCt32Sprite(command, sprite);
            t.IsTrue(decision.supported,
                     "the narrow opaque CT32 command should be eligible");
            t.Equals(decision.reason, GsFallbackReason::Supported,
                     "eligible preparation should retain the canonical reason");
            t.Equals(sprite.framebufferBaseBlock, 0x3FE0u,
                     "FRAME page units should become 256-byte block units");
            t.Equals(sprite.framebufferWidth, 1u,
                     "FRAME width zero should use the software oracle's one-unit normalization");
            t.Equals(sprite.x0, 4u,
                     "reversed fractional X endpoints should ceil then clip at scissor minimum");
            t.Equals(sprite.y0, 3u,
                     "XYOFFSET-adjusted Y should clip at scissor minimum");
            t.Equals(sprite.x1, 16u,
                     "sprite X maximum should remain exclusive after scissor clipping");
            t.Equals(sprite.y1, 13u,
                     "sprite Y maximum should remain exclusive after scissor clipping");
            t.Equals(sprite.rgba, 0xD4C3B2A1u,
                     "flat sprite color should come from the second GS vertex in RGBA byte order");
            t.Equals(sprite.reserved, 0u,
                     "prepared shader ABI records should clear reserved data");

            GSPrimReg textured = command.primitive();
            textured.tme = true;
            const GsDrawCommand rejected = buildGsDrawCommand(
                18u, textured, command.context(),
                std::span<const GSVertex>(
                    command.vertices().data(), command.vertexCount()),
                command.globalState());
            const GsVulkanCt32Sprite sentinel = sprite;
            const GsBackendDecision rejectedDecision =
                prepareGsVulkanCt32Sprite(rejected, sprite);
            t.IsFalse(rejectedDecision.supported,
                      "texturing should be rejected before a shader record is published");
            t.Equals(rejectedDecision.reason, GsFallbackReason::Textured,
                     "preparation should expose the canonical first fallback reason");
            t.IsTrue(sprite == sentinel,
                     "rejected preparation must preserve the caller's record");

            const GsDrawCommand empty = makeCt32SpriteCommand(
                19u, 0u, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                123u, 456u, 123u, 456u,
                0x01020304u);
            const GsBackendDecision emptyDecision =
                prepareGsVulkanCt32Sprite(empty, sprite);
            t.Equals(emptyDecision.reason, GsFallbackReason::EmptyBounds,
                     "degenerate endpoints should fail before any GPU submission");
            t.IsTrue(sprite == sentinel,
                     "degenerate preparation must preserve the caller's record");
        });

        tc.Run("Vulkan raster backend verifies commits and fails before mutation", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const GsDrawCommand command = makeCt32SpriteCommand(
                41u, 9u, 2u,
                {0u, 63u, 0u, 63u}, {0u, 0u},
                3u * 16u, 5u * 16u,
                19u * 16u, 14u * 16u,
                0xA1B2C3D4u);
            const auto commands =
                std::span<const GsDrawCommand>(&command, 1u);

            std::vector<uint8_t> vram = makeVramPattern(0xBACC3EEDu);
            const std::vector<uint8_t> initial = vram;
            std::vector<uint8_t> expected = initial;
            GsVulkanCt32Sprite prepared{};
            t.IsTrue(prepareGsVulkanCt32Sprite(command, prepared).supported,
                     "the backend fixture should satisfy the canonical predicate");
            applyCt32SpriteCpu(expected, prepared);

            uint64_t softwareCalls = 0u;
            uint64_t commitCalls = 0u;
            const auto softwareOracle =
                [&](const GsDrawCommand &draw)
            {
                ++softwareCalls;
                GsVulkanCt32Sprite sprite{};
                if (prepareGsVulkanCt32Sprite(draw, sprite).supported)
                    applyCt32SpriteCpu(vram, sprite);
            };
            const auto acceleratedCommit =
                [&](const GsDrawCommand &)
            {
                ++commitCalls;
            };

            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::Verify;
            std::string creationError;
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram, softwareOracle,
                    acceleratedCommit, &creationError);
            t.IsNotNull(backend.get(),
                        "a healthy injected executor should create the backend");
            t.IsTrue(creationError.empty(),
                     "successful injected creation should clear its diagnostic");
            if (!backend)
                return;

            backend->submit(commands);
            t.IsTrue(vram == expected,
                     "verify mode should retain the agreed software image");
            t.Equals(softwareCalls, 1ull,
                     "verify mode should execute the software oracle once");
            t.Equals(commitCalls, 0ull,
                     "verify mode must not publish the duplicate GPU image");
            GsVulkanRasterBackendStatistics statistics =
                backend->backendStatistics();
            t.Equals(statistics.commandsAttempted, 1ull,
                     "verify should record one attempted command");
            t.Equals(statistics.commandsCompleted, 1ull,
                     "an agreeing command should complete");
            t.Equals(statistics.verifiedCommands, 1ull,
                     "the independent comparison should be counted");
            t.Equals(statistics.bytesCompared,
                     static_cast<uint64_t>(GS_VULKAN_VRAM_SIZE),
                     "verification should compare all 4 MiB");

            t.IsTrue(backend->setMode(GsRendererMode::GpuStrict),
                     "the synchronized backend should accept strict mode");
            t.IsFalse(backend->setMode(GsRendererMode::Software),
                      "the accelerated backend should reject software mode");
            std::copy(initial.begin(), initial.end(), vram.begin());
            backend->submit(commands);
            t.IsTrue(vram == expected,
                     "strict mode should publish the exact GPU result");
            t.Equals(softwareCalls, 1ull,
                     "strict execution must not invoke the software oracle");
            t.Equals(commitCalls, 1ull,
                     "strict execution should publish one commit callback");
            statistics = backend->backendStatistics();
            t.Equals(statistics.committedGpuCommands, 1ull,
                     "strict publication should be counted exactly once");

            GSPrimReg textured = command.primitive();
            textured.tme = true;
            const GsDrawCommand unsupported = buildGsDrawCommand(
                42u, textured, command.context(),
                std::span<const GSVertex>(
                    command.vertices().data(), command.vertexCount()),
                command.globalState());
            t.Equals(backend->classify(unsupported).reason,
                     GsFallbackReason::Textured,
                     "the backend should reuse the canonical reason");
            const std::vector<uint8_t> unsupportedSentinel = vram;
            bool unsupportedThrew = false;
            try
            {
                backend->submit(
                    std::span<const GsDrawCommand>(&unsupported, 1u));
            }
            catch (const std::logic_error &)
            {
                unsupportedThrew = true;
            }
            t.IsTrue(unsupportedThrew,
                     "direct unsupported submission should fail as a router bug");
            t.IsTrue(vram == unsupportedSentinel,
                     "unsupported direct submission must not mutate VRAM");

            std::copy(initial.begin(), initial.end(), vram.begin());
            const std::array<GsDrawCommand, 2> mixedBatch{
                command, unsupported};
            bool batchThrew = false;
            try
            {
                backend->submit(mixedBatch);
            }
            catch (const std::logic_error &)
            {
                batchThrew = true;
            }
            t.IsTrue(batchThrew,
                     "an unsupported batch member should reject the whole span");
            t.IsTrue(vram == initial,
                     "batch preclassification must precede every mutation");
            t.Equals(backend->backendStatistics().commandsAttempted, 2ull,
                     "rejected spans should not count partial attempts");

            std::vector<uint8_t> failureVram = initial;
            uint64_t failedSoftwareCalls = 0u;
            std::unique_ptr<GsVulkanRasterBackend> failingBackend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Fail),
                    GsVulkanRasterBackendConfig{
                        GsRendererMode::GpuStrict, {}},
                    failureVram,
                    [&](const GsDrawCommand &)
                    {
                        ++failedSoftwareCalls;
                    },
                    {}, nullptr);
            t.IsNotNull(failingBackend.get(),
                        "an initially healthy failing executor should construct");
            bool executionThrew = false;
            if (failingBackend)
            {
                try
                {
                    failingBackend->submit(commands);
                }
                catch (const std::runtime_error &error)
                {
                    executionThrew =
                        std::string(error.what()).find(
                            "before canonical VRAM mutation") !=
                        std::string::npos;
                }
                t.Equals(
                    failingBackend->backendStatistics().gpuRequestsFailed,
                    1ull,
                    "executor failure should be counted once");
            }
            t.IsTrue(executionThrew,
                     "executor failure should report the synchronized boundary");
            t.IsTrue(failureVram == initial,
                     "executor failure must retain pre-draw canonical VRAM");
            t.Equals(failedSoftwareCalls, 0ull,
                     "GPU failure should occur before the oracle mutates VRAM");

            backend->flush(GsFlushReason::Shutdown);
            t.IsFalse(backend->healthy(),
                      "shutdown should close the injected executor once");
            t.Equals(backend->classify(command).reason,
                     GsFallbackReason::BackendUnavailable,
                     "post-shutdown classification should fail closed");
            t.Equals(backend->pendingCommandCount(), static_cast<size_t>(0u),
                     "Phase 3 requests should remain synchronous");
        });

        tc.Run("Vulkan verify mismatch writes a complete bounded reproducer", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            ScopedArtifactDirectory artifacts;
            const std::filesystem::path collidingBundle =
                artifacts.path / "draw-00000000000000000091";
            std::filesystem::create_directory(collidingBundle);
            const GsDrawCommand command = makeCt32SpriteCommand(
                91u, 5u, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                4u * 16u, 6u * 16u,
                11u * 16u, 13u * 16u,
                0x44332211u);
            std::vector<uint8_t> vram(GS_VULKAN_VRAM_SIZE, 0u);
            const std::vector<uint8_t> initial = vram;

            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::Verify;
            config.verificationArtifactDirectory =
                artifacts.path.string();
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Noop),
                    config, vram,
                    [&](const GsDrawCommand &draw)
                    {
                        GsVulkanCt32Sprite sprite{};
                        if (prepareGsVulkanCt32Sprite(
                                draw, sprite).supported)
                        {
                            applyCt32SpriteCpu(vram, sprite);
                        }
                    },
                    {}, nullptr);
            t.IsNotNull(backend.get(),
                        "the mismatch fixture backend should construct");
            if (!backend)
                return;

            std::string mismatchText;
            try
            {
                backend->submit(
                    std::span<const GsDrawCommand>(&command, 1u));
            }
            catch (const std::runtime_error &error)
            {
                mismatchText = error.what();
            }
            t.IsFalse(mismatchText.empty(),
                      "an injected no-op GPU result should disagree");
            t.IsTrue(mismatchText.find("draw 91") != std::string::npos,
                     "the failure should identify the exact sequence");
            t.IsTrue(mismatchText.find("artifact=") != std::string::npos,
                     "the failure should publish its reproducer path");

            const GsVulkanRasterBackendStatistics statistics =
                backend->backendStatistics();
            t.Equals(statistics.commandsAttempted, 1ull,
                     "the mismatching command should be attempted once");
            t.Equals(statistics.commandsCompleted, 0ull,
                     "a mismatch must not be reported as completed");
            t.Equals(statistics.verificationMismatches, 1ull,
                     "the first disagreement should stop verification");
            t.Equals(statistics.bytesCompared,
                     static_cast<uint64_t>(GS_VULKAN_VRAM_SIZE),
                     "the mismatch search should cover canonical VRAM");

            const std::filesystem::path bundle =
                statistics.lastVerificationArtifact;
            t.IsTrue(std::filesystem::is_directory(bundle),
                     "the reproducer should be atomically published as a directory");
            t.IsTrue(bundle.filename() ==
                         "draw-00000000000000000091-1",
                     "an existing sequence bundle should select a stable suffix");
            const auto expectFullVram = [&](const char *name)
            {
                std::error_code error;
                const uintmax_t size = std::filesystem::file_size(
                    bundle / name, error);
                t.IsFalse(static_cast<bool>(error),
                          std::string(name) + " should be readable");
                t.Equals(size,
                         static_cast<uintmax_t>(GS_VULKAN_VRAM_SIZE),
                         std::string(name) + " should retain all 4 MiB");
            };
            expectFullVram("initial-vram.bin");
            expectFullVram("software-vram.bin");
            expectFullVram("gpu-vram.bin");

            std::ifstream manifest(bundle / "command.json");
            const std::string manifestText{
                std::istreambuf_iterator<char>(manifest),
                std::istreambuf_iterator<char>()};
            t.IsTrue(manifest.good() || manifest.eof(),
                     "the command manifest should be readable");
            t.IsTrue(manifestText.find(
                         "ps2-gs-vulkan-verification-mismatch") !=
                         std::string::npos,
                     "the manifest should identify its stable schema kind");
            t.IsTrue(manifestText.find("\"sequence\": 91") !=
                         std::string::npos,
                     "the manifest should retain the command identity");
            t.IsTrue(manifestText.find("\"vertices\"") !=
                         std::string::npos,
                     "the manifest should retain raw vertex state");
            t.IsTrue(manifestText.find("\"write_pages\"") !=
                         std::string::npos,
                     "the manifest should retain conservative resource pages");

            std::vector<uint8_t> expected = initial;
            GsVulkanCt32Sprite sprite{};
            (void)prepareGsVulkanCt32Sprite(command, sprite);
            applyCt32SpriteCpu(expected, sprite);
            t.IsTrue(vram == expected,
                     "the software oracle should remain canonical on mismatch");
        });

        tc.Run("GS Vulkan verify and hybrid preserve fallback transfer ordering", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            std::vector<uint8_t> softwareVram =
                makeVramPattern(0x52545231u);
            std::vector<uint8_t> acceleratedVram = softwareVram;
            GS software;
            GS accelerated;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()), nullptr);
            accelerated.init(
                acceleratedVram.data(),
                static_cast<uint32_t>(acceleratedVram.size()), nullptr);
            configureFlatCt32Draws(software);
            configureFlatCt32Draws(accelerated);

            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig config =
                makeRendererServiceConfig(preflight);
            ScopedArtifactDirectory artifacts;
            t.IsTrue(accelerated.configureVulkanRenderer(
                         config, artifacts.path.string()),
                     "a software-mode GS should accept Vulkan configuration before creation");
            if (!preflight.ready())
            {
                t.IsFalse(accelerated.setRendererMode(
                              GsRendererMode::Verify),
                          "an unavailable host should decline verify mode cleanly");
                t.Equals(accelerated.rendererMode(),
                         GsRendererMode::Software,
                         "failed opt-in selection must preserve software mode");
                t.IsFalse(accelerated.rendererDiagnostic().empty(),
                          "failed renderer creation should retain a diagnostic");
                return;
            }

            t.IsTrue(accelerated.setRendererMode(
                         GsRendererMode::Verify),
                     "a capable host should create the integrated verify backend");
            if (accelerated.rendererMode() != GsRendererMode::Verify)
                return;
            t.IsTrue(accelerated.rendererDiagnostic().empty(),
                     "successful renderer selection should clear diagnostics");
            t.IsFalse(accelerated.configureVulkanRenderer(config),
                      "renderer configuration should lock after backend creation");

            accelerated.setBackendCountersEnabled(true);
            accelerated.resetBackendCounters();

            drawFlatCt32Sprite(
                software, 4u * 16u, 5u * 16u,
                26u * 16u, 18u * 16u, 0xA0123456u);
            drawFlatCt32Sprite(
                accelerated, 4u * 16u, 5u * 16u,
                26u * 16u, 18u * 16u, 0xA0123456u);
            t.IsTrue(acceleratedVram == softwareVram,
                     "an integrated verified sprite should preserve the complete software image");

            drawFlatCt32Point(
                software, 9u * 16u, 7u * 16u, 0xB0654321u);
            drawFlatCt32Point(
                accelerated, 9u * 16u, 7u * 16u, 0xB0654321u);
            t.IsTrue(acceleratedVram == softwareVram,
                     "verify fallback should execute after the synchronized GPU draw");

            constexpr std::array<uint32_t, 3> transferPixels{
                0xC0010203u, 0xD0040506u, 0xE0070809u};
            constexpr uint32_t framebufferBlock = 3u << 5u;
            uploadCt32Pixels(
                software, framebufferBlock, 2u, 12u, 9u,
                transferPixels);
            uploadCt32Pixels(
                accelerated, framebufferBlock, 2u, 12u, 9u,
                transferPixels);
            t.IsTrue(acceleratedVram == softwareVram,
                     "a host transfer after GPU-to-software fallback should preserve canonical VRAM");

            drawFlatCt32Sprite(
                software, 10u * 16u, 8u * 16u,
                31u * 16u, 23u * 16u, 0xF0ABCDEFu);
            drawFlatCt32Sprite(
                accelerated, 10u * 16u, 8u * 16u,
                31u * 16u, 23u * 16u, 0xF0ABCDEFu);
            (void)software.getDebugSnapshot();
            (void)accelerated.getDebugSnapshot();
            t.IsTrue(acceleratedVram == softwareVram,
                     "the post-transfer verified overlap should remain exact at a forced observation");

            const GsBackendCounters verifyCounters =
                accelerated.backendCounters();
            t.Equals(verifyCounters.commands, 3ull,
                     "the verify sequence should route all three assembled draws");
            t.Equals(verifyCounters.acceleratedCommands, 2ull,
                     "both eligible sprites should use the Vulkan backend");
            t.Equals(verifyCounters.verifiedCommands, 2ull,
                     "both Vulkan sprites should be independently verified");
            t.Equals(verifyCounters.softwareCommands, 1ull,
                     "the unsupported point should execute once in software");
            t.Equals(verifyCounters.fallbackCommands, 1ull,
                     "the unsupported point should be an explicit fallback");
            t.Equals(verifyCounters.backendSwitches, 1ull,
                     "the adjacent supported-to-fallback transition should synchronize once");
            t.Equals(
                verifyCounters.flushReasons[static_cast<size_t>(
                    GsFlushReason::Transfer)],
                1ull,
                "the host upload should remain a named transfer boundary");

            t.IsTrue(accelerated.setRendererMode(
                         GsRendererMode::Hybrid),
                     "the proven synchronized backend should opt into hybrid mode");
            t.Equals(accelerated.rendererMode(), GsRendererMode::Hybrid,
                     "hybrid selection should be externally observable");
            accelerated.resetBackendCounters();

            drawFlatCt32Sprite(
                software, 2u * 16u, 24u * 16u,
                18u * 16u, 35u * 16u, 0x81726354u);
            drawFlatCt32Sprite(
                accelerated, 2u * 16u, 24u * 16u,
                18u * 16u, 35u * 16u, 0x81726354u);
            drawFlatCt32Point(
                software, 17u * 16u, 34u * 16u, 0x91FEDCBAu);
            drawFlatCt32Point(
                accelerated, 17u * 16u, 34u * 16u, 0x91FEDCBAu);
            (void)software.getDebugSnapshot();
            (void)accelerated.getDebugSnapshot();
            t.IsTrue(acceleratedVram == softwareVram,
                     "hybrid GPU and software work should share exact canonical VRAM");

            const GsBackendCounters hybridCounters =
                accelerated.backendCounters();
            t.Equals(hybridCounters.commands, 2ull,
                     "hybrid should observe both commands");
            t.Equals(hybridCounters.acceleratedCommands, 1ull,
                     "hybrid should accelerate the eligible sprite");
            t.Equals(hybridCounters.softwareCommands, 1ull,
                     "hybrid should retain transparent software fallback");
            t.Equals(hybridCounters.fallbackCommands, 1ull,
                     "hybrid fallback should remain explicit");
            t.Equals(hybridCounters.backendSwitches, 1ull,
                     "hybrid should synchronize the adjacent backend transition");

            const GsVulkanRasterBackendStatistics backendStatistics =
                accelerated.vulkanRendererBackendStatistics();
            t.Equals(backendStatistics.commandsCompleted, 3ull,
                     "three total GPU commands should complete across verify and hybrid");
            t.Equals(backendStatistics.verifiedCommands, 2ull,
                     "verify should retain its two complete comparisons");
            t.Equals(backendStatistics.committedGpuCommands, 1ull,
                     "hybrid should publish one GPU result to canonical VRAM");
            t.Equals(backendStatistics.verificationMismatches, 0ull,
                     "the integrated sequence should have no byte mismatch");
            const GsVulkanServiceStatistics serviceStatistics =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(serviceStatistics.spriteDrawsCompleted, 3ull,
                     "the service should execute exactly the eligible draws");
            t.Equals(serviceStatistics.validationErrors, 0u,
                     "integrated transitions should remain validation-clean");
            t.Equals(serviceStatistics.validationWarnings, 0u,
                     "integrated transitions should emit no validation warnings");
        });

        tc.Run("GS Vulkan strict rejects before mutation and survives reset", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            std::vector<uint8_t> softwareVram =
                makeVramPattern(0x53545231u);
            std::vector<uint8_t> strictVram = softwareVram;
            GS software;
            GS strict;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()), nullptr);
            strict.init(
                strictVram.data(),
                static_cast<uint32_t>(strictVram.size()), nullptr);
            configureFlatCt32Draws(software);
            configureFlatCt32Draws(strict);

            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig config =
                makeRendererServiceConfig(preflight);
            t.IsTrue(strict.configureVulkanRenderer(config),
                     "strict fixture configuration should succeed before creation");
            if (!preflight.ready())
            {
                t.IsFalse(strict.setRendererMode(
                              GsRendererMode::GpuStrict),
                          "an unavailable host should decline strict mode cleanly");
                t.Equals(strict.rendererMode(), GsRendererMode::Software,
                         "failed strict selection must preserve software mode");
                return;
            }

            t.IsTrue(strict.setRendererMode(
                         GsRendererMode::GpuStrict),
                     "a capable host should create the strict Vulkan backend");
            if (strict.rendererMode() != GsRendererMode::GpuStrict)
                return;
            strict.setBackendCountersEnabled(true);
            strict.resetBackendCounters();

            drawFlatCt32Sprite(
                software, 6u * 16u, 4u * 16u,
                23u * 16u, 16u * 16u, 0xC0112233u);
            drawFlatCt32Sprite(
                strict, 6u * 16u, 4u * 16u,
                23u * 16u, 16u * 16u, 0xC0112233u);
            (void)strict.getDebugSnapshot();
            t.IsTrue(strictVram == softwareVram,
                     "strict GPU publication should match the software oracle");

            const std::vector<uint8_t> rejectionSentinel = strictVram;
            std::string pointFailure;
            try
            {
                drawFlatCt32Point(
                    strict, 9u * 16u, 9u * 16u, 0xD0445566u);
            }
            catch (const std::runtime_error &error)
            {
                pointFailure = error.what();
            }
            t.IsTrue(pointFailure.find("unsupported-primitive") !=
                         std::string::npos,
                     "strict should name the first unsupported primitive");
            t.IsTrue(strictVram == rejectionSentinel,
                     "strict point rejection must occur before VRAM mutation");

            std::string emptyFailure;
            try
            {
                strict.writeRegister(
                    GS_REG_PRIM,
                    static_cast<uint64_t>(GS_PRIM_SPRITE));
                strict.writeRegister(GS_REG_RGBAQ, 0xE0778899u);
                strict.writeRegister(
                    GS_REG_XYZ2, packXyz2(14u * 16u, 12u * 16u));
                strict.writeRegister(
                    GS_REG_XYZ2, packXyz2(14u * 16u, 12u * 16u));
            }
            catch (const std::runtime_error &error)
            {
                emptyFailure = error.what();
            }
            t.IsTrue(emptyFailure.find("empty-bounds") !=
                         std::string::npos,
                     "strict should reject empty bounds before submission");
            t.IsTrue(strictVram == rejectionSentinel,
                     "strict empty rejection must preserve all canonical VRAM");

            strict.reset();
            software.reset();
            t.Equals(strict.rendererMode(), GsRendererMode::GpuStrict,
                     "GS reset should preserve the selected renderer service and mode");
            configureFlatCt32Draws(software);
            configureFlatCt32Draws(strict);
            drawFlatCt32Sprite(
                software, 19u * 16u, 20u * 16u,
                39u * 16u, 31u * 16u, 0xF0AABBCCu);
            drawFlatCt32Sprite(
                strict, 19u * 16u, 20u * 16u,
                39u * 16u, 31u * 16u, 0xF0AABBCCu);
            strict.writeRegister(GS_REG_FINISH, 0u);
            (void)strict.getDebugSnapshot();
            t.IsTrue(strictVram == softwareVram,
                     "strict execution after reset and forced drain should remain exact");

            const GsBackendCounters counters = strict.backendCounters();
            t.Equals(counters.commands, 4ull,
                     "strict should classify two accepted and two rejected commands");
            t.Equals(counters.acceleratedCommands, 2ull,
                     "both eligible strict sprites should reach Vulkan");
            t.Equals(counters.softwareCommands, 0ull,
                     "strict must never hide rejection behind software");
            t.Equals(counters.strictFailures, 2ull,
                     "both pre-mutation rejections should be explicit");
            t.Equals(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::Reset)],
                1ull,
                "reset should drain the active accelerated backend once");
            t.Equals(
                counters.decisions[static_cast<size_t>(
                    GsFallbackReason::UnsupportedPrimitive)],
                1ull,
                "the strict point should retain its canonical reason");
            t.Equals(
                counters.decisions[static_cast<size_t>(
                    GsFallbackReason::EmptyBounds)],
                1ull,
                "the strict degenerate sprite should retain its canonical reason");

            const GsVulkanRasterBackendStatistics backendStatistics =
                strict.vulkanRendererBackendStatistics();
            t.Equals(backendStatistics.commandsCompleted, 2ull,
                     "only accepted strict draws should execute");
            t.Equals(backendStatistics.committedGpuCommands, 2ull,
                     "both accepted strict draws should publish GPU VRAM");
            const GsVulkanServiceStatistics serviceStatistics =
                strict.vulkanRendererServiceStatistics();
            t.Equals(serviceStatistics.spriteDrawsCompleted, 2ull,
                     "strict rejections should not become service requests");
            t.Equals(serviceStatistics.validationErrors, 0u,
                     "strict reset and drain should remain validation-clean");
            t.Equals(serviceStatistics.validationWarnings, 0u,
                     "strict reset and drain should emit no validation warnings");
        });

        tc.Run("Compact PSM addresses match the CPU memory oracle", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const std::vector<GsVulkanMemoryCase> cases =
                makeMemoryReadCorpus();
            t.Equals(cases.size(), static_cast<size_t>(61952u),
                     "the corpus should cover every page pixel plus fixed-seed off-page cases");
            t.IsTrue(cases.size() <= GS_VULKAN_MAX_MEMORY_CASES,
                     "the exhaustive corpus must fit one bounded GPU dispatch");

            std::vector<GSMem::PixelAddress> addresses;
            addresses.reserve(cases.size());
            for (size_t index = 0u; index < cases.size(); ++index)
            {
                const GsVulkanMemoryCase &memoryCase = cases[index];
                GSMem::PixelAddress address{};
                const auto psm = static_cast<GSMem::PixelStorageMode>(
                    memoryCase.pixelStorageMode);
                if (!GSMem::ResolvePixelAddress(
                        psm,
                        memoryCase.baseBlock,
                        memoryCase.bufferWidth,
                        memoryCase.x,
                        memoryCase.y,
                        address))
                {
                    t.Fail("compact resolver rejected " +
                           describeMemoryCase(memoryCase, index));
                    return;
                }
                const uint32_t expectedWidth =
                    static_cast<uint32_t>(GSMem::BitsPerPixel(psm));
                if (address.packed_bit_width != expectedWidth ||
                    address.bit_shift + expectedWidth > 32u)
                {
                    t.Fail("compact resolver returned an invalid field for " +
                           describeMemoryCase(memoryCase, index));
                    return;
                }
                addresses.push_back(address);
            }

            constexpr std::array<uint32_t, 5> bitWidths{
                32u, 24u, 16u, 8u, 4u};
            for (uint32_t bitWidth : bitWidths)
            {
                const uint32_t passCount =
                    (25u + bitWidth - 1u) / bitWidth;
                const uint32_t valueMask = bitWidth == 32u
                    ? 0xFFFFFFFFu
                    : (1u << bitWidth) - 1u;
                for (uint32_t pass = 0u; pass < passCount; ++pass)
                {
                    const uint32_t chunkOffset = pass * bitWidth;
                    std::vector<uint32_t> encodedWords =
                        makeAddressTokenPattern(bitWidth, chunkOffset);
                    uint8_t *encodedVram = reinterpret_cast<uint8_t *>(
                        encodedWords.data());
                    for (size_t index = 0u;
                         index < cases.size(); ++index)
                    {
                        const auto psm =
                            static_cast<GSMem::PixelStorageMode>(
                                cases[index].pixelStorageMode);
                        if (GSMem::BitsPerPixel(psm) != bitWidth)
                            continue;
                        const GSMem::PixelAddress &address =
                            addresses[index];
                        const uint32_t addressToken =
                            (address.word_index << 5u) |
                            address.bit_shift;
                        const uint32_t expected =
                            (addressToken >> chunkOffset) & valueMask;
                        const uint32_t actual = cpuReadPixel(
                            psm, encodedVram,
                            cases[index].baseBlock,
                            cases[index].bufferWidth,
                            cases[index].x,
                            cases[index].y);
                        if (actual != expected)
                        {
                            std::ostringstream failure;
                            failure << "CPU lookup disagrees with compact address at "
                                    << describeMemoryCase(
                                           cases[index], index)
                                    << " pass=" << std::dec << pass
                                    << " expected=0x" << std::hex
                                    << expected << " actual=0x" << actual;
                            t.Fail(failure.str());
                            return;
                        }
                    }
                }
            }

            GSMem::PixelAddress invalid{
                0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu};
            t.IsFalse(GSMem::ResolvePixelAddress(
                          static_cast<GSMem::PixelStorageMode>(0x3Fu),
                          0u, 1u, 0u, 0u, invalid),
                      "unsupported PSMs must fail compact resolution");
            t.Equals(invalid.word_index, 0u,
                     "failed compact resolution should clear its result");
        });

        tc.Run("An unavailable Vulkan loader fails closed", [](TestCase &t)
        {
            GsVulkanProbeConfig config{};
            config.loaderPath =
                "/ps2recomp-test-loader-does-not-exist/libvulkan.so";
            const GsVulkanCapabilityReport report =
                probeGsVulkanCapabilities(config);

#if PS2X_HAS_GS_VULKAN
            t.IsTrue(report.compiled,
                     "a header-enabled build should report Vulkan as compiled");
            t.IsFalse(report.loaderAvailable,
                      "an explicit missing loader must not fall back to a system loader");
            t.Equals(report.status,
                     GsVulkanProbeStatus::LoaderUnavailable,
                     "a missing runtime loader should be a clean capability result");
#else
            t.IsFalse(report.compiled,
                      "a disabled build should expose the compiled-out capability");
            t.Equals(report.status,
                     GsVulkanProbeStatus::CompiledOut,
                     "a disabled build must remain a clean software-only build");
#endif
            t.IsFalse(report.ready(),
                      "a missing or compiled-out loader can never select GPU work");
            t.IsNull(report.selectedDevice(),
                     "failed capability discovery must not retain a selected device");

            GsVulkanServiceConfig serviceConfig{};
            serviceConfig.probe = config;
            GsVulkanCapabilityReport serviceReport{};
            std::string serviceError;
            const std::unique_ptr<GsVulkanService> service =
                GsVulkanService::create(
                    serviceConfig, &serviceReport, &serviceError);
            t.IsNull(service.get(),
                     "a missing explicit loader must not create a service");
            t.Equals(serviceReport.status, report.status,
                     "the service should preserve the capability failure status");
            t.IsFalse(serviceError.empty(),
                      "service creation failure should retain a diagnostic");
        });

        tc.Run("Headless Vulkan device selection is internally consistent", [](TestCase &t)
        {
            const GsVulkanCapabilityReport report =
                probeGsVulkanCapabilities();

#if PS2X_HAS_GS_VULKAN
            t.IsTrue(report.compiled,
                     "the compiled Vulkan path should identify itself");
            if (report.ready())
            {
                const GsVulkanDeviceReport *selected =
                    report.selectedDevice();
                t.IsNotNull(selected,
                            "a ready report must identify its selected device");
                if (selected)
                {
                    t.IsTrue(selected->suitable,
                             "the selected device must satisfy every gate");
                    t.IsTrue(selected->exactVramStorage,
                             "the selected device must address exact raw VRAM storage");
                    t.IsTrue(selected->computeQueue,
                             "the selected device must expose a compute queue");
                    t.IsTrue(
                        selected->maxStorageBufferRange >=
                            GS_VULKAN_VRAM_SIZE,
                        "the selected storage-buffer range must cover all GS VRAM");
                    t.IsFalse(
                        selected->kind == GsVulkanDeviceKind::Cpu,
                        "a CPU Vulkan implementation is not a hardware renderer");
                }
            }
            else
            {
                t.IsNull(report.selectedDevice(),
                         "an unavailable host must remain software-only");
            }
#else
            t.Equals(report.status,
                     GsVulkanProbeStatus::CompiledOut,
                     "the generic probe must be callable in software-only builds");
#endif

            for (const GsVulkanDeviceReport &device : report.devices)
            {
                if (!device.suitable)
                    continue;
                t.IsTrue(device.exactVramStorage && device.computeQueue &&
                             device.deviceLocalMemory &&
                             device.hostVisibleMemory,
                         "every suitable device should satisfy the permanent capability contract");
                t.IsTrue(device.rejectionReason.empty(),
                         "suitable devices should not retain a rejection reason");
            }
        });

        tc.Run("Requested Vulkan validation is explicit and fail-closed", [](TestCase &t)
        {
            GsVulkanProbeConfig config{};
            config.enableValidation = true;
            const GsVulkanCapabilityReport report =
                probeGsVulkanCapabilities(config);

            t.IsTrue(report.validationRequested,
                     "the report should retain the validation request");
            if (report.ready())
            {
                t.IsTrue(report.validationLayerAvailable &&
                             report.debugUtilsAvailable &&
                             report.validationEnabled,
                         "a validated ready device must have the layer, extension, and messenger");
                t.Equals(report.validationErrors, 0u,
                         "a ready validated probe must contain no validation errors");
            }
            if (report.status ==
                GsVulkanProbeStatus::ValidationUnavailable)
            {
                t.IsFalse(report.ready(),
                          "missing requested validation must not silently continue");
            }
        });

        tc.Run("Vulkan service rejects an unbounded fence configuration", [](TestCase &t)
        {
            GsVulkanServiceConfig config{};
            config.probe.enableValidation = false;
            config.fenceTimeoutNanoseconds = 0u;
            const GsVulkanCapabilityReport preflight =
                probeGsVulkanCapabilities(config.probe);
            GsVulkanCapabilityReport report{};
            std::string error;
            const std::unique_ptr<GsVulkanService> service =
                GsVulkanService::create(config, &report, &error);
            t.IsNull(service.get(),
                     "a zero timeout must never create an execution service");
            t.IsFalse(error.empty(),
                      "the rejected timeout should retain a diagnostic");
            if (preflight.ready())
            {
                t.Equals(report.status,
                         GsVulkanProbeStatus::ResourceCreationFailed,
                         "a capable host should reject the unbounded service contract itself");
            }
            else
            {
                t.Equals(report.status, preflight.status,
                         "an unavailable host should retain its earlier capability failure");
            }
        });

        tc.Run("Vulkan service preserves exact 4 MiB VRAM", [](TestCase &t)
        {
            GsVulkanServiceConfig config{};
            config.probe.enableValidation = true;
            GsVulkanCapabilityReport preflight =
                probeGsVulkanCapabilities(config.probe);
            if (preflight.status ==
                GsVulkanProbeStatus::ValidationUnavailable)
            {
                config.probe.enableValidation = false;
                preflight = probeGsVulkanCapabilities(config.probe);
            }

            GsVulkanCapabilityReport creationReport{};
            std::string creationError;
            std::unique_ptr<GsVulkanService> service =
                GsVulkanService::create(
                    config, &creationReport, &creationError);
            if (!preflight.ready())
            {
                t.IsNull(service.get(),
                         "an unavailable host should remain software-only");
                t.IsFalse(creationReport.ready(),
                          "an unavailable service must preserve its capability result");
                return;
            }

            t.IsNotNull(service.get(),
                        "a suitable device should create the execution service");
            t.IsTrue(creationReport.ready(),
                     "a created service should retain a ready report");
            t.IsTrue(creationError.empty(),
                     "successful service creation should clear its diagnostic");
            if (!service)
                return;

            std::vector<std::vector<uint8_t>> patterns;
            patterns.emplace_back(GS_VULKAN_VRAM_SIZE, 0u);
            patterns.emplace_back(GS_VULKAN_VRAM_SIZE, 0xFFu);
            patterns.emplace_back(GS_VULKAN_VRAM_SIZE);
            for (size_t index = 0u; index < patterns.back().size(); ++index)
            {
                patterns.back()[index] =
                    static_cast<uint8_t>(index);
            }
            patterns.push_back(makeVramPattern(0x52414331u));

            for (const std::vector<uint8_t> &input : patterns)
            {
                std::vector<uint8_t> output = {0xA5u, 0x5Au};
                std::string error;
                const bool succeeded =
                    service->roundTripVram(input, output, &error);
                t.IsTrue(succeeded,
                         "each fixed-size service request should complete");
                t.IsTrue(error.empty(),
                         "a successful request should clear its diagnostic");
                t.IsTrue(output == input,
                         "upload, shader access, and download must preserve every VRAM byte");
            }

            const std::vector<uint8_t> invalid(
                GS_VULKAN_VRAM_SIZE - 1u, 0x11u);
            std::vector<uint8_t> invalidOutput = {0xBEu, 0xEFu};
            const std::vector<uint8_t> invalidSentinel = invalidOutput;
            std::string invalidError;
            t.IsFalse(service->roundTripVram(
                          invalid, invalidOutput, &invalidError),
                      "an input shorter than exact GS VRAM must fail closed");
            t.IsTrue(invalidOutput == invalidSentinel,
                     "a rejected request must not alter caller output");
            t.IsFalse(invalidError.empty(),
                      "a rejected request should explain the exact-size contract");
            t.IsTrue(service->healthy(),
                     "caller input rejection must not poison the worker");

            const std::vector<uint8_t> concurrentA =
                makeVramPattern(0x13579BDFu);
            const std::vector<uint8_t> concurrentB =
                makeVramPattern(0x2468ACE1u);
            std::vector<uint8_t> outputA;
            std::vector<uint8_t> outputB;
            std::string errorA;
            std::string errorB;
            bool succeededA = false;
            bool succeededB = false;
            std::thread callerA([&]
            {
                succeededA = service->roundTripVram(
                    concurrentA, outputA, &errorA);
            });
            std::thread callerB([&]
            {
                succeededB = service->roundTripVram(
                    concurrentB, outputB, &errorB);
            });
            callerA.join();
            callerB.join();
            t.IsTrue(succeededA && succeededB,
                     "concurrent callers should serialize through the bounded slot");
            t.IsTrue(errorA.empty() && errorB.empty(),
                     "serialized callers should complete without diagnostics");
            t.IsTrue(outputA == concurrentA && outputB == concurrentB,
                     "serialized requests must retain their own exact output");

            const GsVulkanCapabilityReport finalReport =
                service->capabilities();
            const GsVulkanServiceStatistics statistics =
                service->statistics();
            t.IsTrue(finalReport.ready() && service->healthy(),
                     "successful work should leave the service selectable");
            t.Equals(statistics.roundTripsCompleted, 6ull,
                     "all accepted requests should complete exactly once");
            t.Equals(statistics.roundTripsFailed, 0ull,
                     "caller-side size rejection should not submit failed GPU work");
            t.Equals(statistics.queueSubmissions, 6ull,
                     "each request should use one bounded queue submission");
            t.Equals(statistics.shaderDispatches, 6ull,
                     "each request should execute the storage-buffer shader");
            t.Equals(statistics.bytesUploaded,
                     6ull * GS_VULKAN_VRAM_SIZE,
                     "statistics should account for every uploaded byte");
            t.Equals(statistics.bytesDownloaded,
                     6ull * GS_VULKAN_VRAM_SIZE,
                     "statistics should account for every downloaded byte");
            t.Equals(statistics.validationErrors, 0u,
                     "validated service work must remain error-free");
            t.IsFalse(statistics.deviceLost,
                      "successful service work must not report device loss");

            service->shutdown();
            service->shutdown();
            t.IsFalse(service->healthy(),
                      "explicit shutdown should be synchronous and idempotent");
            const GsVulkanServiceStatistics shutdownStatistics =
                service->statistics();
            t.Equals(shutdownStatistics.roundTripsCompleted, 6ull,
                     "shutdown must drain without duplicating accepted work");
            t.Equals(shutdownStatistics.validationErrors, 0u,
                     "resource destruction must remain validation-clean");

            std::vector<uint8_t> shutdownOutput = {0x42u};
            const std::vector<uint8_t> shutdownSentinel = shutdownOutput;
            std::string shutdownError;
            t.IsFalse(service->roundTripVram(
                          patterns.front(), shutdownOutput,
                          &shutdownError),
                      "a stopped service must reject later work");
            t.IsTrue(shutdownOutput == shutdownSentinel,
                     "post-shutdown rejection must preserve caller output");
        });

        tc.Run("Vulkan CT32 sprites match CPU bounds overlap and wrap", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            std::vector<GsVulkanCt32Sprite> sprites;
            const auto addSprite = [&](GsDrawCommand command)
            {
                GsVulkanCt32Sprite sprite{};
                const GsBackendDecision decision =
                    prepareGsVulkanCt32Sprite(command, sprite);
                if (!decision.supported)
                {
                    t.Fail(
                        "synthetic CT32 sprite was rejected as " +
                        std::string(gsFallbackReasonName(
                            decision.reason)));
                    return false;
                }
                sprites.push_back(sprite);
                return true;
            };

            if (!addSprite(makeCt32SpriteCommand(
                    1u, 0u, 1u,
                    {0u, 63u, 0u, 63u}, {0u, 0u},
                    7u * 16u, 9u * 16u,
                    8u * 16u, 10u * 16u,
                    0x80112233u)) ||
                !addSprite(makeCt32SpriteCommand(
                    2u, 3u, 2u,
                    {4u, 31u, 3u, 23u}, {32u, 16u},
                    35u, 18u,
                    497u, 402u,
                    0xA0445566u)) ||
                !addSprite(makeCt32SpriteCommand(
                    3u, 511u, 1u,
                    {0u, 127u, 0u, 63u}, {0u, 0u},
                    60u * 16u, 28u * 16u,
                    68u * 16u, 36u * 16u,
                    0xC0778899u)) ||
                !addSprite(makeCt32SpriteCommand(
                    4u, 7u, 2u,
                    {0u, 63u, 0u, 63u}, {0u, 0u},
                    7u * 16u, 11u * 16u,
                    29u * 16u, 27u * 16u,
                    0xE0AABBCCu)) ||
                !addSprite(makeCt32SpriteCommand(
                    5u, 7u, 2u,
                    {0u, 63u, 0u, 63u}, {0u, 0u},
                    15u * 16u, 19u * 16u,
                    36u * 16u, 35u * 16u,
                    0xF0DDEEFFu)))
            {
                return;
            }

            t.Equals(sprites.size(), static_cast<size_t>(5u),
                     "the synthetic corpus should retain every prepared sprite");
            t.Equals(sprites[1].x0, 4u,
                     "fractional XYOFFSET geometry should clip to X scissor minimum");
            t.Equals(sprites[1].y0, 3u,
                     "fractional XYOFFSET geometry should clip to Y scissor minimum");
            t.Equals(sprites[2].framebufferBaseBlock, 0x3FE0u,
                     "the wrap fixture should begin in the last GS page");

            GsVulkanServiceConfig config{};
            config.probe.enableValidation = true;
            GsVulkanCapabilityReport preflight =
                probeGsVulkanCapabilities(config.probe);
            if (preflight.status ==
                GsVulkanProbeStatus::ValidationUnavailable)
            {
                config.probe.enableValidation = false;
                preflight = probeGsVulkanCapabilities(config.probe);
            }

            GsVulkanCapabilityReport creationReport{};
            std::string creationError;
            std::unique_ptr<GsVulkanService> service =
                GsVulkanService::create(
                    config, &creationReport, &creationError);
            if (!preflight.ready())
            {
                t.IsNull(service.get(),
                         "an unavailable host should skip CT32 shader execution cleanly");
                t.IsFalse(creationReport.ready(),
                          "the skipped sprite service must retain its capability result");
                return;
            }

            t.IsNotNull(service.get(),
                        "a suitable device should create the CT32 sprite service");
            t.IsTrue(creationError.empty(),
                     "successful sprite service creation should clear its diagnostic");
            if (!service)
                return;

            const std::vector<uint8_t> initial =
                makeVramPattern(0x53505231u);
            std::vector<uint8_t> gpu = initial;
            uint64_t expectedPixels = 0u;
            for (size_t index = 0u; index < sprites.size(); ++index)
            {
                const GsVulkanCt32Sprite &sprite = sprites[index];
                std::vector<uint8_t> expected = gpu;
                applyCt32SpriteCpu(expected, sprite);
                std::vector<uint8_t> actual = {0xA5u};
                std::string error;
                if (!service->executeCt32Sprite(
                        gpu, sprite, actual, &error))
                {
                    t.Fail(
                        "GPU CT32 sprite " +
                        std::to_string(index) + " failed: " + error);
                    return;
                }
                if (actual != expected)
                {
                    t.Fail(
                        "GPU CT32 sprite " +
                        std::to_string(index) +
                        " disagreed with the complete CPU VRAM image");
                    return;
                }
                gpu = std::move(actual);
                expectedPixels +=
                    static_cast<uint64_t>(sprite.x1 - sprite.x0) *
                    static_cast<uint64_t>(sprite.y1 - sprite.y0);
            }

            // Repeat the wrapping dispatch from the same input to prove that
            // physical-address overlap is deterministic under atomic writes.
            std::vector<uint8_t> repeatedExpected = initial;
            applyCt32SpriteCpu(repeatedExpected, sprites[2]);
            std::vector<uint8_t> repeatedOutput = {0x5Au};
            std::string repeatedError;
            t.IsTrue(service->executeCt32Sprite(
                         initial, sprites[2], repeatedOutput,
                         &repeatedError),
                     "the repeated wrap dispatch should complete");
            t.IsTrue(repeatedOutput == repeatedExpected,
                     "the repeated wrap dispatch should remain byte-exact");
            expectedPixels +=
                static_cast<uint64_t>(sprites[2].x1 - sprites[2].x0) *
                static_cast<uint64_t>(sprites[2].y1 - sprites[2].y0);

            const auto expectRejected =
                [&](std::span<const uint8_t> rejectedInput,
                    GsVulkanCt32Sprite rejectedSprite,
                    const std::string &label)
            {
                std::vector<uint8_t> output = {0x12u, 0x34u};
                const std::vector<uint8_t> sentinel = output;
                std::string error;
                t.IsFalse(service->executeCt32Sprite(
                              rejectedInput, rejectedSprite,
                              output, &error),
                          label + " should fail closed");
                t.IsTrue(output == sentinel,
                         label + " must preserve caller output");
                t.IsFalse(error.empty(),
                          label + " should retain a diagnostic");
            };

            const std::vector<uint8_t> shortInput(
                GS_VULKAN_VRAM_SIZE - 1u, 0u);
            expectRejected(shortInput, sprites.front(),
                           "short sprite VRAM input");
            GsVulkanCt32Sprite invalid = sprites.front();
            invalid.framebufferBaseBlock = 0x4000u;
            expectRejected(initial, invalid,
                           "out-of-range sprite framebuffer base");
            invalid = sprites.front();
            invalid.framebufferWidth = 0u;
            expectRejected(initial, invalid,
                           "zero sprite framebuffer width");
            invalid = sprites.front();
            invalid.x1 = invalid.x0;
            expectRejected(initial, invalid, "empty sprite bounds");
            invalid = sprites.front();
            invalid.x1 = 2049u;
            expectRejected(initial, invalid,
                           "out-of-range sprite bounds");
            invalid = sprites.front();
            invalid.reserved = 1u;
            expectRejected(initial, invalid,
                           "non-zero sprite reserved data");

            const GsVulkanServiceStatistics statistics =
                service->statistics();
            constexpr uint64_t acceptedDraws = 6u;
            t.Equals(statistics.spriteDrawsCompleted, acceptedDraws,
                     "every accepted sprite should complete exactly once");
            t.Equals(statistics.spriteDrawsFailed, 0ull,
                     "caller-side sprite rejection should not count as failed GPU work");
            t.Equals(statistics.spritePixelsExecuted, expectedPixels,
                     "sprite statistics should count exact covered pixels");
            t.Equals(statistics.queueSubmissions, acceptedDraws,
                     "each Phase 3 sprite should use one queue submission");
            t.Equals(statistics.shaderDispatches, acceptedDraws,
                     "each Phase 3 sprite should use one compute dispatch");
            t.Equals(statistics.bytesUploaded,
                     acceptedDraws * GS_VULKAN_VRAM_SIZE,
                     "each sprite should upload canonical CPU VRAM");
            t.Equals(statistics.bytesDownloaded,
                     acceptedDraws * GS_VULKAN_VRAM_SIZE,
                     "each sprite should download canonical CPU VRAM");
            t.Equals(statistics.validationErrors, 0u,
                     "CT32 sprite execution must remain validation-clean");
            t.Equals(statistics.validationWarnings, 0u,
                     "CT32 sprite execution should emit no validation warnings");
            t.IsTrue(service->healthy(),
                     "successful and rejected sprites should leave the service healthy");

            service->shutdown();
            service->shutdown();
            const GsVulkanServiceStatistics shutdownStatistics =
                service->statistics();
            t.Equals(shutdownStatistics.validationErrors, 0u,
                     "sprite pipeline destruction must remain validation-clean");
            t.Equals(shutdownStatistics.validationWarnings, 0u,
                     "sprite pipeline destruction should emit no validation warnings");

            std::vector<uint8_t> shutdownOutput = {0xABu};
            const std::vector<uint8_t> shutdownSentinel = shutdownOutput;
            std::string shutdownError;
            t.IsFalse(service->executeCt32Sprite(
                          initial, sprites.front(), shutdownOutput,
                          &shutdownError),
                      "a stopped service must reject sprite work");
            t.IsTrue(shutdownOutput == shutdownSentinel,
                     "post-shutdown sprite rejection must preserve output");
        });

        tc.Run("Vulkan PSM helpers match CPU addresses reads and writes", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const std::vector<GsVulkanMemoryCase> readCases =
                makeMemoryReadCorpus();
            const std::vector<uint8_t> input =
                makeVramPattern(0x50534D31u);

            GsVulkanServiceConfig config{};
            config.probe.enableValidation = true;
            GsVulkanCapabilityReport preflight =
                probeGsVulkanCapabilities(config.probe);
            if (preflight.status ==
                GsVulkanProbeStatus::ValidationUnavailable)
            {
                config.probe.enableValidation = false;
                preflight = probeGsVulkanCapabilities(config.probe);
            }

            GsVulkanCapabilityReport creationReport{};
            std::string creationError;
            std::unique_ptr<GsVulkanService> service =
                GsVulkanService::create(
                    config, &creationReport, &creationError);
            if (!preflight.ready())
            {
                t.IsNull(service.get(),
                         "an unavailable host should skip shader PSM conformance cleanly");
                t.IsFalse(creationReport.ready(),
                          "the skipped conformance service must retain its capability result");
                return;
            }

            t.IsNotNull(service.get(),
                        "a suitable device should create the PSM conformance service");
            t.IsTrue(creationError.empty(),
                     "successful PSM service creation should clear its diagnostic");
            if (!service)
                return;

            std::vector<uint8_t> readOutput = {0xA5u};
            std::vector<GsVulkanMemoryResult> readResults = {{
                0xAAAAAAAAu, 0xBBBBBBBBu,
                0xCCCCCCCCu, 0xDDDDDDDDu}};
            std::string readError;
            if (!service->executeMemoryCases(
                    input, readCases, readOutput, readResults,
                    &readError))
            {
                t.Fail("exhaustive GPU PSM read batch failed: " +
                       readError);
                return;
            }
            t.IsTrue(readOutput == input,
                     "read-only PSM cases must preserve every VRAM byte");
            t.Equals(readResults.size(), readCases.size(),
                     "one GPU result must be returned for every read case");
            if (readResults.size() != readCases.size())
                return;

            std::vector<uint8_t> mutableInput = input;
            for (size_t index = 0u; index < readCases.size(); ++index)
            {
                const GsVulkanMemoryCase &memoryCase = readCases[index];
                const auto psm = static_cast<GSMem::PixelStorageMode>(
                    memoryCase.pixelStorageMode);
                GSMem::PixelAddress address{};
                const bool resolved = GSMem::ResolvePixelAddress(
                    psm,
                    memoryCase.baseBlock,
                    memoryCase.bufferWidth,
                    memoryCase.x,
                    memoryCase.y,
                    address);
                const uint32_t expectedValue = cpuReadPixel(
                    psm, mutableInput.data(),
                    memoryCase.baseBlock,
                    memoryCase.bufferWidth,
                    memoryCase.x,
                    memoryCase.y);
                const GsVulkanMemoryResult &actual =
                    readResults[index];
                if (!resolved ||
                    actual.wordIndex != address.word_index ||
                    actual.bitShift != address.bit_shift ||
                    actual.valueBefore != expectedValue ||
                    actual.valueAfter != expectedValue)
                {
                    std::ostringstream failure;
                    failure << "GPU read mismatch at "
                            << describeMemoryCase(memoryCase, index)
                            << " expected(word=0x" << std::hex
                            << address.word_index << ", shift=0x"
                            << address.bit_shift << ", value=0x"
                            << expectedValue << ") actual(word=0x"
                            << actual.wordIndex << ", shift=0x"
                            << actual.bitShift << ", before=0x"
                            << actual.valueBefore << ", after=0x"
                            << actual.valueAfter << ')';
                    t.Fail(failure.str());
                    return;
                }
            }

            for (size_t index = 0u; index < kPsmSpecs.size(); ++index)
            {
                const PsmSpec &spec = kPsmSpecs[index];
                GsVulkanMemoryCase writeCase{};
                writeCase.pixelStorageMode =
                    static_cast<uint32_t>(spec.psm);
                writeCase.baseBlock =
                    0x3FE0u +
                    ((static_cast<uint32_t>(index) * 5u + 31u) &
                     31u);
                writeCase.bufferWidth =
                    2u + 2u * (static_cast<uint32_t>(index) % 31u);
                writeCase.x =
                    spec.pageWidth *
                        (2u + static_cast<uint32_t>(index % 3u)) -
                    1u;
                writeCase.y =
                    spec.pageHeight *
                        (2u + static_cast<uint32_t>(index % 2u)) -
                    1u;
                writeCase.operation =
                    GsVulkanMemoryOperation::Write;
                writeCase.value =
                    0xDEADBEEFu ^
                    (static_cast<uint32_t>(index) * 0x10204081u);

                std::vector<uint8_t> expected = input;
                const uint32_t expectedBefore = cpuReadPixel(
                    spec.psm, expected.data(),
                    writeCase.baseBlock,
                    writeCase.bufferWidth,
                    writeCase.x,
                    writeCase.y);
                cpuWritePixel(
                    spec.psm, expected.data(),
                    writeCase.baseBlock,
                    writeCase.bufferWidth,
                    writeCase.x,
                    writeCase.y,
                    writeCase.value);
                const uint32_t expectedAfter = cpuReadPixel(
                    spec.psm, expected.data(),
                    writeCase.baseBlock,
                    writeCase.bufferWidth,
                    writeCase.x,
                    writeCase.y);
                GSMem::PixelAddress expectedAddress{};
                const bool resolved = GSMem::ResolvePixelAddress(
                    spec.psm,
                    writeCase.baseBlock,
                    writeCase.bufferWidth,
                    writeCase.x,
                    writeCase.y,
                    expectedAddress);

                std::vector<uint8_t> writeOutput = {0x5Au};
                std::vector<GsVulkanMemoryResult> writeResults = {{
                    0x11111111u, 0x22222222u,
                    0x33333333u, 0x44444444u}};
                std::string writeError;
                if (!service->executeMemoryCases(
                        input,
                        std::span<const GsVulkanMemoryCase>(
                            &writeCase, 1u),
                        writeOutput, writeResults, &writeError))
                {
                    t.Fail(std::string("GPU write failed for ") +
                           spec.name + ": " + writeError);
                    return;
                }
                if (!resolved || writeOutput != expected ||
                    writeResults.size() != 1u ||
                    writeResults[0].wordIndex !=
                        expectedAddress.word_index ||
                    writeResults[0].bitShift !=
                        expectedAddress.bit_shift ||
                    writeResults[0].valueBefore != expectedBefore ||
                    writeResults[0].valueAfter != expectedAfter)
                {
                    std::ostringstream failure;
                    failure << "GPU write disagrees with CPU RMW for "
                            << spec.name << " expected(word=0x"
                            << std::hex << expectedAddress.word_index
                            << ", shift=0x"
                            << expectedAddress.bit_shift
                            << ", before=0x" << expectedBefore
                            << ", after=0x" << expectedAfter << ')';
                    t.Fail(failure.str());
                    return;
                }
            }

            const auto expectRejected =
                [&](std::span<const uint8_t> rejectedInput,
                    std::span<const GsVulkanMemoryCase> rejectedCases,
                    const std::string &label)
            {
                std::vector<uint8_t> output = {0x12u, 0x34u};
                const std::vector<uint8_t> outputSentinel = output;
                std::vector<GsVulkanMemoryResult> results = {{
                    1u, 2u, 3u, 4u}};
                const std::vector<GsVulkanMemoryResult> resultSentinel =
                    results;
                std::string error;
                t.IsFalse(service->executeMemoryCases(
                              rejectedInput, rejectedCases,
                              output, results, &error),
                          label + " should fail closed");
                t.IsTrue(output == outputSentinel,
                         label + " must preserve VRAM output");
                t.IsTrue(results == resultSentinel,
                         label + " must preserve result output");
                t.IsFalse(error.empty(),
                          label + " should retain a diagnostic");
            };

            const std::vector<uint8_t> shortInput(
                GS_VULKAN_VRAM_SIZE - 1u, 0u);
            expectRejected(shortInput,
                           std::span<const GsVulkanMemoryCase>(
                               readCases.data(), 1u),
                           "short VRAM input");
            expectRejected(input, {}, "empty memory batch");
            std::vector<GsVulkanMemoryCase> tooMany(
                GS_VULKAN_MAX_MEMORY_CASES + 1u);
            expectRejected(input, tooMany, "oversized memory batch");

            GsVulkanMemoryCase invalidCase = readCases.front();
            invalidCase.pixelStorageMode = 0x3Fu;
            expectRejected(
                input,
                std::span<const GsVulkanMemoryCase>(&invalidCase, 1u),
                "unsupported PSM");
            invalidCase = readCases.front();
            invalidCase.operation =
                static_cast<GsVulkanMemoryOperation>(2u);
            expectRejected(
                input,
                std::span<const GsVulkanMemoryCase>(&invalidCase, 1u),
                "unsupported memory operation");
            invalidCase = readCases.front();
            invalidCase.reserved = 1u;
            expectRejected(
                input,
                std::span<const GsVulkanMemoryCase>(&invalidCase, 1u),
                "non-zero reserved data");

            const GsVulkanServiceStatistics statistics =
                service->statistics();
            constexpr uint64_t expectedBatches =
                1u + kPsmSpecs.size();
            t.Equals(statistics.memoryBatchesCompleted,
                     expectedBatches,
                     "all accepted PSM batches should complete exactly once");
            t.Equals(statistics.memoryBatchesFailed, 0ull,
                     "caller-side rejection should not count as failed GPU work");
            t.Equals(statistics.memoryCasesExecuted,
                     static_cast<uint64_t>(readCases.size()) +
                         kPsmSpecs.size(),
                     "memory case statistics should count every invocation");
            t.Equals(statistics.queueSubmissions, expectedBatches,
                     "each PSM batch should use one queue submission");
            t.Equals(statistics.shaderDispatches, expectedBatches,
                     "each PSM batch should use one compute dispatch");
            t.Equals(statistics.bytesUploaded,
                     expectedBatches * GS_VULKAN_VRAM_SIZE,
                     "each PSM batch should upload exact GS VRAM");
            t.Equals(statistics.bytesDownloaded,
                     expectedBatches * GS_VULKAN_VRAM_SIZE,
                     "each PSM batch should download exact GS VRAM");
            t.Equals(statistics.validationErrors, 0u,
                     "PSM shader execution must remain validation-clean");
            t.Equals(statistics.validationWarnings, 0u,
                     "PSM shader execution should emit no validation warnings");
            t.IsTrue(service->healthy(),
                     "successful and caller-rejected cases must leave the service healthy");

            service->shutdown();
            service->shutdown();
            const GsVulkanServiceStatistics shutdownStatistics =
                service->statistics();
            t.Equals(shutdownStatistics.validationErrors, 0u,
                     "PSM pipeline destruction must remain validation-clean");
            t.Equals(shutdownStatistics.validationWarnings, 0u,
                     "PSM pipeline destruction should emit no validation warnings");

            std::vector<uint8_t> shutdownOutput = {0xABu};
            const std::vector<uint8_t> shutdownOutputSentinel =
                shutdownOutput;
            std::vector<GsVulkanMemoryResult> shutdownResults = {{
                5u, 6u, 7u, 8u}};
            const std::vector<GsVulkanMemoryResult>
                shutdownResultSentinel = shutdownResults;
            std::string shutdownError;
            t.IsFalse(service->executeMemoryCases(
                          input,
                          std::span<const GsVulkanMemoryCase>(
                              readCases.data(), 1u),
                          shutdownOutput, shutdownResults,
                          &shutdownError),
                      "a stopped service must reject PSM work");
            t.IsTrue(shutdownOutput == shutdownOutputSentinel &&
                         shutdownResults == shutdownResultSentinel,
                     "post-shutdown PSM rejection must preserve both outputs");
        });
    });
}
