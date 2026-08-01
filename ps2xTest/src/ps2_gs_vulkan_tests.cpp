#include "MiniTest.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_gs_memory.h"
#include "runtime/ps2_gs_vulkan.h"
#include "runtime/ps2_gs_vulkan_backend.h"
#include "runtime/ps2_memory.h"

#include <algorithm>
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

    GsDrawCommand makeCt32TriangleCommand(
        uint64_t sequence,
        uint32_t framebufferPage,
        uint8_t framebufferWidth,
        GSScissorReg scissor,
        GSXYOffsetReg xyoffset,
        const std::array<uint16_t, 3> &rawX,
        const std::array<uint16_t, 3> &rawY,
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
        primitive.type = GS_PRIM_TRIANGLE;
        std::array<GSVertex, 3> vertices{};
        for (size_t index = 0u; index < vertices.size(); ++index)
        {
            vertices[index].x12_4 = rawX[index];
            vertices[index].y12_4 = rawY[index];
            vertices[index].r = static_cast<uint8_t>(
                0x10u + index);
            vertices[index].g = static_cast<uint8_t>(
                0x20u + index);
            vertices[index].b = static_cast<uint8_t>(
                0x30u + index);
            vertices[index].a = static_cast<uint8_t>(
                0x40u + index);
        }
        vertices[2].r = static_cast<uint8_t>(rgba);
        vertices[2].g = static_cast<uint8_t>(rgba >> 8u);
        vertices[2].b = static_cast<uint8_t>(rgba >> 16u);
        vertices[2].a = static_cast<uint8_t>(rgba >> 24u);
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

    int64_t ct32TriangleEdge(
        int32_t ax,
        int32_t ay,
        int32_t bx,
        int32_t by,
        int32_t sampleX,
        int32_t sampleY)
    {
        return
            (static_cast<int64_t>(bx) - ax) *
                (static_cast<int64_t>(sampleY) - ay) -
            (static_cast<int64_t>(by) - ay) *
                (static_cast<int64_t>(sampleX) - ax);
    }

    void applyCt32TriangleCpu(
        std::vector<uint8_t> &vram,
        const GsVulkanCt32Triangle &triangle)
    {
        for (uint32_t y = triangle.boundsY0;
             y < triangle.boundsY1; ++y)
        {
            for (uint32_t x = triangle.boundsX0;
                 x < triangle.boundsX1; ++x)
            {
                const int32_t sampleX = static_cast<int32_t>(x * 16u);
                const int32_t sampleY = static_cast<int32_t>(y * 16u);
                const std::array<int64_t, 3> edges{{
                    ct32TriangleEdge(
                        triangle.vertex1X12_4,
                        triangle.vertex1Y12_4,
                        triangle.vertex2X12_4,
                        triangle.vertex2Y12_4,
                        sampleX,
                        sampleY),
                    ct32TriangleEdge(
                        triangle.vertex2X12_4,
                        triangle.vertex2Y12_4,
                        triangle.vertex0X12_4,
                        triangle.vertex0Y12_4,
                        sampleX,
                        sampleY),
                    ct32TriangleEdge(
                        triangle.vertex0X12_4,
                        triangle.vertex0Y12_4,
                        triangle.vertex1X12_4,
                        triangle.vertex1Y12_4,
                        sampleX,
                        sampleY),
                }};
                bool covered = true;
                for (uint32_t edge = 0u; edge < edges.size(); ++edge)
                {
                    if (edges[edge] < 0 ||
                        (edges[edge] == 0 &&
                         (triangle.topLeftEdgeMask & (1u << edge)) == 0u))
                    {
                        covered = false;
                        break;
                    }
                }
                if (!covered)
                    continue;
                GSMem::WriteCT32(
                    vram.data(),
                    triangle.framebufferBaseBlock,
                    triangle.framebufferWidth,
                    x,
                    y,
                    triangle.rgba);
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

    void drawRecursiveCt32Sprite(
        GS &gs,
        uint16_t x0,
        uint16_t y0,
        uint16_t x1,
        uint16_t y1,
        uint32_t rgba)
    {
        gs.writeRegister(
            GS_REG_PRIM,
            static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4u) |  // TME
                (1ull << 8u));  // FST
        gs.writeRegister(GS_REG_RGBAQ, rgba);
        gs.writeRegister(GS_REG_UV, 0u);
        gs.writeRegister(GS_REG_XYZ2, packXyz2(x0, y0));
        gs.writeRegister(
            GS_REG_UV,
            static_cast<uint64_t>(x1 - x0) |
                (static_cast<uint64_t>(y1 - y0) << 16u));
        gs.writeRegister(GS_REG_XYZ2, packXyz2(x1, y1));
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

    void copyCt32Pixels(
        GS &gs,
        uint32_t sourceBlock,
        uint32_t destinationBlock,
        uint32_t bufferWidth,
        uint16_t sourceX,
        uint16_t sourceY,
        uint16_t destinationX,
        uint16_t destinationY,
        uint16_t width,
        uint16_t height)
    {
        const uint64_t bitbltbuf =
            static_cast<uint64_t>(sourceBlock) |
            (static_cast<uint64_t>(bufferWidth) << 16u) |
            (static_cast<uint64_t>(GS_PSM_CT32) << 24u) |
            (static_cast<uint64_t>(destinationBlock) << 32u) |
            (static_cast<uint64_t>(bufferWidth) << 48u) |
            (static_cast<uint64_t>(GS_PSM_CT32) << 56u);
        const uint64_t trxpos =
            static_cast<uint64_t>(sourceX) |
            (static_cast<uint64_t>(sourceY) << 16u) |
            (static_cast<uint64_t>(destinationX) << 32u) |
            (static_cast<uint64_t>(destinationY) << 48u);
        const uint64_t trxreg =
            static_cast<uint64_t>(width) |
            (static_cast<uint64_t>(height) << 32u);
        gs.writeRegister(GS_REG_BITBLTBUF, bitbltbuf);
        gs.writeRegister(GS_REG_TRXPOS, trxpos);
        gs.writeRegister(GS_REG_TRXREG, trxreg);
        gs.writeRegister(GS_REG_TRXDIR, 2u);
    }

    std::vector<uint8_t> readCt32Pixels(
        GS &gs,
        uint32_t sourceBlock,
        uint32_t sourceWidth,
        uint16_t x,
        uint16_t y,
        uint16_t width,
        uint16_t height)
    {
        const uint64_t bitbltbuf =
            static_cast<uint64_t>(sourceBlock) |
            (static_cast<uint64_t>(sourceWidth) << 16u) |
            (static_cast<uint64_t>(GS_PSM_CT32) << 24u);
        const uint64_t trxpos =
            static_cast<uint64_t>(x) |
            (static_cast<uint64_t>(y) << 16u);
        const uint64_t trxreg =
            static_cast<uint64_t>(width) |
            (static_cast<uint64_t>(height) << 32u);
        gs.writeRegister(GS_REG_BITBLTBUF, bitbltbuf);
        gs.writeRegister(GS_REG_TRXPOS, trxpos);
        gs.writeRegister(GS_REG_TRXREG, trxreg);
        gs.writeRegister(GS_REG_TRXDIR, 1u);

        std::vector<uint8_t> bytes(
            static_cast<size_t>(width) * height * sizeof(uint32_t));
        const uint32_t consumed = gs.consumeLocalToHostBytes(
            bytes.data(), static_cast<uint32_t>(bytes.size()));
        bytes.resize(consumed);
        return bytes;
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

    void configureCt32Display(
        GSRegisters &registers,
        uint32_t framebufferPage,
        uint32_t width = 64u,
        uint32_t height = 64u)
    {
        registers.pmode = 1ull;
        registers.dispfb1 =
            framebufferPage |
            (1ull << 9u) |
            (static_cast<uint64_t>(GS_PSM_CT32) << 15u);
        registers.display1 =
            (static_cast<uint64_t>(width - 1u) << 32u) |
            (static_cast<uint64_t>(height - 1u) << 44u);
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
            ++serviceStatistics.queueSubmissions;
            ++serviceStatistics.shaderDispatches;
            serviceStatistics.pipelineBarriers += 4u;
            ++serviceStatistics.pipelineBinds;
            ++serviceStatistics.pipelineCacheHits;
            ++serviceStatistics.fenceWaits;
            if (error)
                error->clear();
            return true;
        }

        bool uploadVramPages(
            std::span<const uint8_t> source,
            const GsVramPageMask &pages,
            std::string *error) override
        {
            if (!isHealthy || behavior == Behavior::Fail ||
                source.size() != GS_VULKAN_VRAM_SIZE || !pages.any())
            {
                ++serviceStatistics.pageUploadOperationsFailed;
                if (error)
                    *error = "injected page upload failure";
                return false;
            }
            for (size_t page = 0u; page < GS_VRAM_PAGE_COUNT; ++page)
            {
                if (!pages.test(page))
                    continue;
                const size_t offset = page * GS_VRAM_PAGE_SIZE;
                std::memcpy(
                    residentVram.data() + offset,
                    source.data() + offset,
                    GS_VRAM_PAGE_SIZE);
            }
            ++serviceStatistics.pageUploadOperationsCompleted;
            ++serviceStatistics.queueSubmissions;
            serviceStatistics.pipelineBarriers += 3u;
            ++serviceStatistics.fenceWaits;
            serviceStatistics.pagesUploaded += pages.count();
            serviceStatistics.bytesUploaded +=
                pages.count() * GS_VRAM_PAGE_SIZE;
            if (error)
                error->clear();
            return true;
        }

        bool downloadVramPages(
            std::span<uint8_t> destination,
            const GsVramPageMask &pages,
            std::string *error) override
        {
            if (!isHealthy || behavior == Behavior::Fail ||
                destination.size() != GS_VULKAN_VRAM_SIZE || !pages.any())
            {
                ++serviceStatistics.pageDownloadOperationsFailed;
                if (error)
                    *error = "injected page download failure";
                return false;
            }
            for (size_t page = 0u; page < GS_VRAM_PAGE_COUNT; ++page)
            {
                if (!pages.test(page))
                    continue;
                const size_t offset = page * GS_VRAM_PAGE_SIZE;
                std::memcpy(
                    destination.data() + offset,
                    residentVram.data() + offset,
                    GS_VRAM_PAGE_SIZE);
            }
            ++serviceStatistics.pageDownloadOperationsCompleted;
            ++serviceStatistics.queueSubmissions;
            serviceStatistics.pipelineBarriers += 3u;
            ++serviceStatistics.fenceWaits;
            serviceStatistics.pagesDownloaded += pages.count();
            serviceStatistics.bytesDownloaded +=
                pages.count() * GS_VRAM_PAGE_SIZE;
            if (error)
                error->clear();
            return true;
        }

        bool executeResidentCt32Sprite(
            const GsVulkanCt32Sprite &sprite,
            std::string *error) override
        {
            return executeResidentCt32Sprites(
                std::span<const GsVulkanCt32Sprite>(&sprite, 1u),
                error);
        }

        bool executeResidentCt32Sprites(
            std::span<const GsVulkanCt32Sprite> sprites,
            std::string *error) override
        {
            if (!isHealthy || behavior == Behavior::Fail ||
                behavior == Behavior::InvalidOutput ||
                sprites.empty() ||
                sprites.size() > GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH)
            {
                serviceStatistics.spriteDrawsFailed += sprites.size();
                ++serviceStatistics.residentSpriteBatchesFailed;
                if (error)
                    *error = "injected resident CT32 executor failure";
                return false;
            }
            for (const GsVulkanCt32Sprite &sprite : sprites)
            {
                if (behavior == Behavior::Exact)
                    applyCt32SpriteCpu(residentVram, sprite);
                serviceStatistics.spritePixelsExecuted +=
                    static_cast<uint64_t>(sprite.x1 - sprite.x0) *
                    static_cast<uint64_t>(sprite.y1 - sprite.y0);
            }
            serviceStatistics.spriteDrawsCompleted += sprites.size();
            ++serviceStatistics.residentSpriteBatchesCompleted;
            serviceStatistics.largestResidentSpriteBatch = std::max(
                serviceStatistics.largestResidentSpriteBatch,
                static_cast<uint64_t>(sprites.size()));
            ++serviceStatistics.queueSubmissions;
            serviceStatistics.shaderDispatches += sprites.size();
            serviceStatistics.pipelineBarriers += 2u;
            ++serviceStatistics.pipelineBinds;
            ++serviceStatistics.pipelineCacheHits;
            ++serviceStatistics.fenceWaits;
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
        std::vector<uint8_t> residentVram =
            std::vector<uint8_t>(GS_VULKAN_VRAM_SIZE);
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
            t.Equals(GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH, size_t{64u},
                     "resident sprite batches should have a fixed host bound");
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

        tc.Run("CT32 triangle preparation normalizes exact fixed-point edges", [](TestCase &t)
        {
            const GsDrawCommand command = makeCt32TriangleCommand(
                27u,
                511u,
                0u,
                {4u, 15u, 3u, 12u},
                {32u, 16u},
                {289u, 209u, 65u},
                {33u, 65u, 209u},
                0xD4C3B2A1u);

            GsVulkanCt32Triangle triangle{
                1u, 2u, 3u, 4u, 5u, 6u,
                7, 8, 9, 10, 11, 12,
                13u, 14u, 15u, 16u};
            const GsBackendDecision decision =
                prepareGsVulkanCt32Triangle(command, triangle);
            t.IsTrue(decision.supported,
                     "the flat opaque CT32 triangle should be eligible");
            t.Equals(decision.reason, GsFallbackReason::Supported,
                     "eligible triangle preparation should retain the canonical reason");
            t.Equals(triangle.framebufferBaseBlock, 0x3FE0u,
                     "triangle FRAME page units should become 256-byte blocks");
            t.Equals(triangle.framebufferWidth, 1u,
                     "triangle FRAME width zero should use one-unit normalization");
            t.Equals(triangle.boundsX0, 4u,
                     "triangle minimum X should be ceil-rounded then scissor clipped");
            t.Equals(triangle.boundsY0, 3u,
                     "triangle minimum Y should be ceil-rounded then scissor clipped");
            t.Equals(triangle.boundsX1, 16u,
                     "triangle maximum X should remain exclusive after clipping");
            t.Equals(triangle.boundsY1, 13u,
                     "triangle maximum Y should remain exclusive after clipping");
            t.Equals(triangle.vertex0X12_4, 257,
                     "triangle preparation should retain signed 12.4 X after XYOFFSET");
            t.Equals(triangle.vertex0Y12_4, 17,
                     "triangle preparation should retain signed 12.4 Y after XYOFFSET");
            t.Equals(triangle.vertex1X12_4, 33,
                     "negative winding should swap the second geometric vertex");
            t.Equals(triangle.vertex1Y12_4, 193,
                     "winding normalization should preserve the swapped Y coordinate");
            t.Equals(triangle.vertex2X12_4, 177,
                     "negative winding should swap the third geometric vertex");
            t.Equals(triangle.vertex2Y12_4, 49,
                     "winding normalization should preserve every fractional coordinate");
            t.Equals(triangle.rgba, 0xD4C3B2A1u,
                     "flat color should come from the original third GS vertex");
            t.Equals(triangle.topLeftEdgeMask, 3u,
                     "prepared edges should name the two inclusive top-left sides");
            t.Equals(triangle.reserved0, 0u,
                     "prepared triangle records should clear reserved data");
            t.Equals(triangle.reserved1, 0u,
                     "all prepared triangle padding should be deterministic");

            GSPrimReg textured = command.primitive();
            textured.tme = true;
            const GsDrawCommand rejected = buildGsDrawCommand(
                28u, textured, command.context(),
                std::span<const GSVertex>(command.vertices()),
                command.globalState());
            const GsVulkanCt32Triangle sentinel = triangle;
            const GsBackendDecision rejectedDecision =
                prepareGsVulkanCt32Triangle(rejected, triangle);
            t.Equals(rejectedDecision.reason, GsFallbackReason::Textured,
                     "triangle preparation should preserve ordered fallback reasons");
            t.IsTrue(triangle == sentinel,
                     "rejected triangle preparation must leave the record untouched");

            std::array<GSVertex, 3> degenerateVertices = command.vertices();
            degenerateVertices[0].x12_4 = 64u;
            degenerateVertices[0].y12_4 = 64u;
            degenerateVertices[1].x12_4 = 128u;
            degenerateVertices[1].y12_4 = 128u;
            degenerateVertices[2].x12_4 = 192u;
            degenerateVertices[2].y12_4 = 192u;
            const GsDrawCommand degenerate = buildGsDrawCommand(
                29u, command.primitive(), command.context(),
                std::span<const GSVertex>(degenerateVertices),
                command.globalState());
            t.Equals(
                prepareGsVulkanCt32Triangle(degenerate, triangle).reason,
                GsFallbackReason::EmptyBounds,
                "collinear triangles should fail before record publication");
            t.IsTrue(triangle == sentinel,
                     "degenerate triangle rejection must preserve the prior record");

            std::array<GSVertex, 2> incompleteVertices{{
                command.vertices()[0], command.vertices()[1]}};
            incompleteVertices[0].x12_4 = 64u;
            incompleteVertices[0].y12_4 = 64u;
            incompleteVertices[1].x12_4 = 192u;
            incompleteVertices[1].y12_4 = 80u;
            const GsDrawCommand incomplete = buildGsDrawCommand(
                30u, command.primitive(), command.context(),
                std::span<const GSVertex>(incompleteVertices),
                command.globalState());
            t.Equals(classifyGsFlatCt32Triangle(incomplete).reason,
                     GsFallbackReason::UnsupportedPrimitiveState,
                     "a nondegenerate partial command must not synthesize a third vertex");
        });

        tc.Run("CT32 triangle preparation matches exhaustive software edge coverage", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            std::vector<uint8_t> actual(GS_VULKAN_VRAM_SIZE, 0u);
            std::vector<uint8_t> expected(GS_VULKAN_VRAM_SIZE, 0u);
            GS gs;
            gs.init(
                actual.data(),
                static_cast<uint32_t>(actual.size()),
                nullptr);
            gs.setDebugHistoryPaused(true);

            using FixedVertex = std::array<int32_t, 2>;
            using FixedTriangle = std::array<FixedVertex, 3>;
            const std::array<FixedTriangle, 3> shapes{{
                {{{-64, 17}, {239, 49}, {33, 255}}},
                {{{32, 32}, {256, 32}, {32, 256}}},
                {{{65, 65}, {81, 241}, {257, 97}}},
            }};
            constexpr std::array<std::array<uint8_t, 3>, 6>
                permutations{{
                    {{0u, 1u, 2u}},
                    {{0u, 2u, 1u}},
                    {{1u, 0u, 2u}},
                    {{1u, 2u, 0u}},
                    {{2u, 0u, 1u}},
                    {{2u, 1u, 0u}},
                }};
            constexpr std::array<GSScissorReg, 4> scissors{{
                {0u, 31u, 0u, 31u},
                {4u, 13u, 3u, 14u},
                {0u, 7u, 6u, 20u},
                {8u, 8u, 0u, 31u},
            }};
            constexpr GSXYOffsetReg xyoffset{128u, 128u};
            constexpr uint32_t rgba = 0xA4B3C2D1u;
            constexpr uint8_t initialByte = 0x5Au;
            uint64_t sequence = 1000u;
            uint64_t preparedCases = 0u;
            uint64_t emptyCases = 0u;

            for (const FixedTriangle &shape : shapes)
            {
                for (const GSScissorReg &scissor : scissors)
                {
                    for (uint32_t phaseY = 0u; phaseY < 16u; ++phaseY)
                    {
                        for (uint32_t phaseX = 0u; phaseX < 16u; ++phaseX)
                        {
                            for (const auto &permutation : permutations)
                            {
                                std::array<uint16_t, 3> rawX{};
                                std::array<uint16_t, 3> rawY{};
                                for (size_t vertex = 0u;
                                     vertex < rawX.size(); ++vertex)
                                {
                                    const FixedVertex &source =
                                        shape[permutation[vertex]];
                                    rawX[vertex] = static_cast<uint16_t>(
                                        source[0] + phaseX + xyoffset.ofx);
                                    rawY[vertex] = static_cast<uint16_t>(
                                        source[1] + phaseY + xyoffset.ofy);
                                }

                                const GsDrawCommand command =
                                    makeCt32TriangleCommand(
                                        sequence++, 0u, 1u,
                                        scissor, xyoffset,
                                        rawX, rawY, rgba);
                                GsVulkanCt32Triangle triangle{};
                                const GsBackendDecision decision =
                                    prepareGsVulkanCt32Triangle(
                                        command, triangle);
                                if (decision.supported)
                                {
                                    ++preparedCases;
                                }
                                else if (decision.reason ==
                                         GsFallbackReason::EmptyBounds)
                                {
                                    ++emptyCases;
                                }
                                else
                                {
                                    t.Fail(
                                        "eligible edge corpus produced an unexpected triangle fallback");
                                    return;
                                }

                                std::fill_n(
                                    actual.begin(),
                                    GS_VRAM_PAGE_SIZE,
                                    initialByte);
                                std::fill_n(
                                    expected.begin(),
                                    GS_VRAM_PAGE_SIZE,
                                    initialByte);
                                if (decision.supported)
                                    applyCt32TriangleCpu(expected, triangle);

                                const uint64_t frame =
                                    (1ull << 16u) |
                                    (static_cast<uint64_t>(GS_PSM_CT32) << 24u);
                                const uint64_t scissorValue =
                                    static_cast<uint64_t>(scissor.x0) |
                                    (static_cast<uint64_t>(scissor.x1) << 16u) |
                                    (static_cast<uint64_t>(scissor.y0) << 32u) |
                                    (static_cast<uint64_t>(scissor.y1) << 48u);
                                const uint64_t xyoffsetValue =
                                    static_cast<uint64_t>(xyoffset.ofx) |
                                    (static_cast<uint64_t>(xyoffset.ofy) << 32u);
                                gs.writeRegister(GS_REG_FRAME_1, frame);
                                gs.writeRegister(GS_REG_ZBUF_1, 1ull << 32u);
                                gs.writeRegister(GS_REG_SCISSOR_1, scissorValue);
                                gs.writeRegister(GS_REG_XYOFFSET_1, xyoffsetValue);
                                gs.writeRegister(GS_REG_TEST_1, 0x30000u);
                                gs.writeRegister(GS_REG_FBA_1, 0u);
                                gs.writeRegister(GS_REG_SCANMSK, 0u);
                                gs.writeRegister(GS_REG_DTHE, 0u);
                                gs.writeRegister(
                                    GS_REG_PRIM,
                                    static_cast<uint64_t>(GS_PRIM_TRIANGLE));
                                for (size_t vertex = 0u;
                                     vertex < rawX.size(); ++vertex)
                                {
                                    gs.writeRegister(GS_REG_RGBAQ, rgba);
                                    gs.writeRegister(
                                        GS_REG_XYZ2,
                                        packXyz2(rawX[vertex], rawY[vertex]));
                                }
                                gs.flushRenderBatch();

                                if (!std::equal(
                                        actual.begin(),
                                        actual.begin() + GS_VRAM_PAGE_SIZE,
                                        expected.begin()))
                                {
                                    std::ostringstream message;
                                    message
                                        << "triangle edge mismatch at sequence "
                                        << command.sequence()
                                        << " phase=(" << phaseX << ','
                                        << phaseY << ") scissor=("
                                        << scissor.x0 << ',' << scissor.y0
                                        << ")-(" << scissor.x1 << ','
                                        << scissor.y1 << ')';
                                    t.Fail(message.str());
                                    return;
                                }
                            }
                        }
                    }
                }
            }

            t.Equals(preparedCases, 18'432ull,
                     "all winding, fractional-phase, and scissor cases should remain drawable");
            t.Equals(emptyCases, 0ull,
                     "the bounded edge corpus should not accidentally become fully clipped");
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
            GsVramPageMask allPages;
            allPages.setAll();
            backend->noteCpuVramWrite(allPages);
            backend->submit(commands);
            t.IsTrue(vram == initial,
                     "strict mode should leave resident GPU-newer pages off the CPU hot path");
            const GsVramPageMask writtenPages =
                command.resources().writePages;
            t.Equals(backend->pendingCommandCount(), static_cast<size_t>(1u),
                     "strict submission should remain in the bounded resident queue");
            t.Equals(
                backend->backendStatistics().pageOwnership.gpuNewerPages,
                static_cast<size_t>(0u),
                "queued work should not publish ownership before execution");
            backend->prepareCpuVramAccess(
                writtenPages, GsFlushReason::CpuReadback);
            t.IsTrue(vram == expected,
                     "an overlapping CPU observation should publish the exact resident result");
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
            backend->noteCpuVramWrite(allPages);
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
                failingBackend->submit(commands);
                t.Equals(failingBackend->pendingCommandCount(),
                         static_cast<size_t>(1u),
                         "a deferred failing request should first enter the queue");
                try
                {
                    failingBackend->flush(GsFlushReason::Explicit);
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
                     "executor failure should report the draining boundary");
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
                     "shutdown should leave no accepted resident work");
        });

        tc.Run("Vulkan hybrid backend keeps disjoint pages resident until scoped CPU access", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const GsDrawCommand first = makeCt32SpriteCommand(
                101u, 5u, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                2u * 16u, 3u * 16u,
                11u * 16u, 13u * 16u,
                0x44332211u);
            const GsDrawCommand second = makeCt32SpriteCommand(
                102u, 511u, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                4u * 16u, 5u * 16u,
                14u * 16u, 15u * 16u,
                0x88776655u);
            const GsDrawCommand firstAgain = makeCt32SpriteCommand(
                103u, 5u, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                6u * 16u, 7u * 16u,
                15u * 16u, 16u * 16u,
                0xCCBBAA99u);

            const GsVramPageMask firstPages =
                first.resources().writePages;
            const GsVramPageMask secondPages =
                second.resources().writePages;
            t.Equals(firstPages.count(), static_cast<size_t>(1u),
                     "the first resident fixture should own one page");
            t.IsTrue(firstPages.test(5u),
                     "the first fixture should resolve FRAME page 5");
            t.Equals(secondPages.count(), static_cast<size_t>(1u),
                     "the wrap-edge resident fixture should own one page");
            t.IsTrue(secondPages.test(511u),
                     "the second fixture should resolve FRAME page 511");

            std::vector<uint8_t> vram = makeVramPattern(0xC0E3E17u);
            const std::vector<uint8_t> initial = vram;
            std::vector<uint8_t> expected = initial;
            for (const GsDrawCommand *draw :
                 std::array<const GsDrawCommand *, 3>{
                     &first, &second, &firstAgain})
            {
                GsVulkanCt32Sprite sprite{};
                t.IsTrue(prepareGsVulkanCt32Sprite(*draw, sprite).supported,
                         "every resident fixture should satisfy the exact predicate");
                applyCt32SpriteCpu(expected, sprite);
            }

            uint64_t softwareCalls = 0u;
            uint64_t commitCalls = 0u;
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    GsVulkanRasterBackendConfig{
                        GsRendererMode::Hybrid, {}},
                    vram,
                    [&](const GsDrawCommand &)
                    {
                        ++softwareCalls;
                    },
                    [&](const GsDrawCommand &)
                    {
                        ++commitCalls;
                    },
                    nullptr);
            t.IsNotNull(backend.get(),
                        "the resident fake backend should construct");
            if (!backend)
                return;

            backend->submit(std::span<const GsDrawCommand>(&first, 1u));
            backend->submit(std::span<const GsDrawCommand>(&second, 1u));
            backend->submit(std::span<const GsDrawCommand>(&firstAgain, 1u));
            t.IsTrue(vram == initial,
                     "three resident draws should avoid every implicit CPU publication");
            t.Equals(softwareCalls, 0ull,
                     "hybrid resident execution must not invoke the oracle");
            t.Equals(commitCalls, 2ull,
                     "the disjoint prefix should publish when the repeated page forces a drain");
            t.Equals(backend->pendingCommandCount(), static_cast<size_t>(1u),
                     "the post-hazard command should remain pending");

            GsVulkanServiceStatistics service = backend->serviceStatistics();
            t.Equals(service.pageUploadOperationsCompleted, 1ull,
                     "the disjoint prefix should upload both pages in one operation");
            t.Equals(service.pagesUploaded, 2ull,
                     "resident setup should transfer exactly two 8 KiB pages");
            t.Equals(service.pageDownloadOperationsCompleted, 0ull,
                     "no CPU observer means no resident page download");
            t.Equals(service.spriteDrawsCompleted, 2ull,
                     "only the drained disjoint prefix should have executed");
            t.Equals(service.residentSpriteBatchesCompleted, 1ull,
                     "the disjoint prefix should be one service batch");
            t.Equals(service.largestResidentSpriteBatch, 2ull,
                     "the first resident batch should contain both disjoint draws");

            GsVulkanRasterBackendStatistics statistics =
                backend->backendStatistics();
            t.Equals(statistics.residentCommands, 2ull,
                     "backend counters should count only executed resident commands");
            t.Equals(statistics.resourceHazardDrains, 1ull,
                     "the repeated physical page should force one conservative drain");
            t.Equals(statistics.largestResidentBatch, 2ull,
                     "backend diagnostics should retain the disjoint prefix size");
            t.Equals(statistics.pageOwnership.gpuNewerPages,
                     static_cast<size_t>(2u),
                     "both touched pages should remain GPU-newer");
            t.Equals(statistics.pageOwnership.cpuNewerPages,
                     static_cast<size_t>(510u),
                     "untouched initial pages should remain explicitly CPU-owned");

            backend->prepareCpuVramAccess(
                firstPages, GsFlushReason::CpuReadback);
            t.Equals(commitCalls, 3ull,
                     "the CPU boundary should publish the remaining metadata commit");
            t.Equals(backend->pendingCommandCount(), static_cast<size_t>(0u),
                     "CPU access should drain all accepted resident work");
            const size_t firstOffset = 5u * GS_VRAM_PAGE_SIZE;
            const size_t secondOffset = 511u * GS_VRAM_PAGE_SIZE;
            t.IsTrue(std::equal(
                         vram.begin() + firstOffset,
                         vram.begin() + firstOffset + GS_VRAM_PAGE_SIZE,
                         expected.begin() + firstOffset),
                     "a scoped readback should publish the requested GPU-newer page");
            t.IsTrue(std::equal(
                         vram.begin() + secondOffset,
                         vram.begin() + secondOffset + GS_VRAM_PAGE_SIZE,
                         initial.begin() + secondOffset),
                     "a scoped readback must leave an unrelated GPU-newer page stale on CPU");

            GsVramPageMask allPages;
            allPages.setAll();
            backend->prepareCpuVramAccess(
                allPages, GsFlushReason::DebuggerObservation);
            t.IsTrue(vram == expected,
                     "a forced full observation should reconstruct the exact CPU image");

            service = backend->serviceStatistics();
            t.Equals(service.pageDownloadOperationsCompleted, 2ull,
                     "two differently scoped observers should issue two downloads");
            t.Equals(service.pagesDownloaded, 2ull,
                     "each GPU-newer page should download exactly once");
            t.Equals(service.spriteDrawsCompleted, 3ull,
                     "the CPU boundary should execute the final queued draw");
            t.Equals(service.residentSpriteBatchesCompleted, 2ull,
                     "the hazard prefix and remaining draw should form two batches");
            statistics = backend->backendStatistics();
            t.Equals(statistics.coherency.gpuWriteOperations, 3ull,
                     "every resident command should advance GPU ownership");
            t.Equals(statistics.coherency.cpuToGpuPages, 2ull,
                     "only first-use CPU-newer pages should become resident");
            t.Equals(statistics.coherency.gpuToCpuPages, 2ull,
                     "forced observations should publish only GPU-newer pages");
            t.Equals(statistics.pageOwnership.gpuNewerPages,
                     static_cast<size_t>(0u),
                     "the final observation should leave no hidden GPU writer");
        });

        tc.Run("Vulkan resident queue drains at its configured bound", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const std::array<GsDrawCommand, 3> commands{
                makeCt32SpriteCommand(
                    201u, 3u, 1u,
                    {0u, 31u, 0u, 31u}, {0u, 0u},
                    1u * 16u, 2u * 16u,
                    9u * 16u, 8u * 16u, 0x44332211u),
                makeCt32SpriteCommand(
                    202u, 7u, 1u,
                    {0u, 31u, 0u, 31u}, {0u, 0u},
                    3u * 16u, 4u * 16u,
                    12u * 16u, 11u * 16u, 0x88776655u),
                makeCt32SpriteCommand(
                    203u, 11u, 1u,
                    {0u, 31u, 0u, 31u}, {0u, 0u},
                    5u * 16u, 6u * 16u,
                    15u * 16u, 13u * 16u, 0xCCBBAA99u),
            };

            std::vector<uint8_t> vram = makeVramPattern(0x424F554Eu);
            const std::vector<uint8_t> initial = vram;
            std::vector<uint8_t> expected = initial;
            for (const GsDrawCommand &command : commands)
            {
                GsVulkanCt32Sprite sprite{};
                t.IsTrue(prepareGsVulkanCt32Sprite(
                             command, sprite).supported,
                         "each bounded-queue fixture should be GPU eligible");
                applyCt32SpriteCpu(expected, sprite);
            }

            uint64_t commitCalls = 0u;
            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::Hybrid;
            config.maximumResidentBatchCommands = 2u;
            config.minimumHybridSpritePixels = 0u;
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram,
                    [](const GsDrawCommand &) {},
                    [&](const GsDrawCommand &) { ++commitCalls; },
                    nullptr);
            t.IsNotNull(backend.get(),
                        "a two-command resident queue should construct");
            if (!backend)
                return;

            for (const GsDrawCommand &command : commands)
            {
                backend->submit(
                    std::span<const GsDrawCommand>(&command, 1u));
            }
            t.Equals(commitCalls, 2ull,
                     "the third command should drain the full two-command prefix");
            t.Equals(backend->pendingCommandCount(), static_cast<size_t>(1u),
                     "the third command should occupy the newly empty queue");
            t.IsTrue(vram == initial,
                     "backpressure execution should not imply CPU publication");

            GsVulkanRasterBackendStatistics backendStatistics =
                backend->backendStatistics();
            t.Equals(backendStatistics.commandsAttempted, 3ull,
                     "all three commands should be accepted once");
            t.Equals(backendStatistics.commandsCompleted, 2ull,
                     "only the backpressure-drained prefix should be complete");
            t.Equals(backendStatistics.queueBackpressureDrains, 1ull,
                     "the configured bound should force one named drain");
            t.Equals(backendStatistics.residentBatchesCompleted, 1ull,
                     "the full prefix should use one resident batch");
            t.Equals(backendStatistics.largestResidentBatch, 2ull,
                     "the backend should retain its configured peak batch size");

            GsVulkanServiceStatistics service =
                backend->serviceStatistics();
            t.Equals(service.pageUploadOperationsCompleted, 1ull,
                     "the full prefix should upload its pages together");
            t.Equals(service.pagesUploaded, 2ull,
                     "the first batch should upload two distinct pages");
            t.Equals(service.spriteDrawsCompleted, 2ull,
                     "the service should execute the full prefix only");
            t.Equals(service.residentSpriteBatchesCompleted, 1ull,
                     "the service should observe one full prefix batch");

            backend->flush(GsFlushReason::Explicit);
            t.Equals(commitCalls, 3ull,
                     "an explicit drain should publish the remaining commit");
            t.Equals(backend->pendingCommandCount(), static_cast<size_t>(0u),
                     "the explicit drain should empty the bounded queue");
            t.IsTrue(vram == initial,
                     "an executor drain should keep GPU-newer bytes resident");

            GsVramPageMask allPages;
            allPages.setAll();
            backend->prepareCpuVramAccess(
                allPages, GsFlushReason::DebuggerObservation);
            t.IsTrue(vram == expected,
                     "the bounded queue should match sequential CPU execution exactly");
            backendStatistics = backend->backendStatistics();
            t.Equals(backendStatistics.commandsCompleted, 3ull,
                     "all accepted commands should complete after the drain");
            t.Equals(backendStatistics.residentBatchesCompleted, 2ull,
                     "the full prefix and tail should form two batches");

            GsVulkanRasterBackendConfig invalidConfig = config;
            invalidConfig.maximumResidentBatchCommands = 0u;
            std::string invalidError;
            std::unique_ptr<GsVulkanRasterBackend> invalidBackend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    invalidConfig, vram,
                    [](const GsDrawCommand &) {}, {}, &invalidError);
            t.IsNull(invalidBackend.get(),
                     "a zero-sized resident queue should fail closed");
            t.IsFalse(invalidError.empty(),
                      "an invalid queue bound should retain a diagnostic");
        });

        tc.Run("Vulkan hybrid cost policy keeps tiny sprites in software", [](TestCase &t)
        {
            class ImmediateSoftwareBackend final : public IGsRasterBackend
            {
            public:
                explicit ImmediateSoftwareBackend(
                    std::vector<uint8_t> &vram_) noexcept
                    : vram(vram_)
                {
                }

                GsBackendDecision classify(
                    const GsDrawCommand &) const override
                {
                    return {true, GsFallbackReason::Supported};
                }

                void submit(
                    std::span<const GsDrawCommand> commands) override
                {
                    for (const GsDrawCommand &command : commands)
                    {
                        GsVulkanCt32Sprite sprite{};
                        if (prepareGsVulkanCt32Sprite(
                                command, sprite).supported)
                        {
                            applyCt32SpriteCpu(vram, sprite);
                        }
                    }
                }

                void flush(GsFlushReason) override {}

                size_t pendingCommandCount() const noexcept override
                {
                    return 0u;
                }

            private:
                std::vector<uint8_t> &vram;
            };

            GSMem::InitLookupTables();
            const GsDrawCommand tiny = makeCt32SpriteCommand(
                301u, 2u, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                1u * 16u, 1u * 16u,
                5u * 16u, 5u * 16u, 0x44332211u);
            const GsDrawCommand threshold = makeCt32SpriteCommand(
                302u, 6u, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                2u * 16u, 2u * 16u,
                10u * 16u, 10u * 16u, 0x88776655u);

            std::vector<uint8_t> vram = makeVramPattern(0x434F5354u);
            std::vector<uint8_t> expected = vram;
            for (const GsDrawCommand *command :
                 std::array<const GsDrawCommand *, 2>{&tiny, &threshold})
            {
                GsVulkanCt32Sprite sprite{};
                t.IsTrue(prepareGsVulkanCt32Sprite(
                             *command, sprite).supported,
                         "cost fixtures should satisfy the semantic predicate");
                applyCt32SpriteCpu(expected, sprite);
            }

            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::Hybrid;
            config.minimumHybridSpritePixels = 64u;
            uint64_t commitCalls = 0u;
            std::unique_ptr<GsVulkanRasterBackend> accelerated =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram,
                    [](const GsDrawCommand &) {},
                    [&](const GsDrawCommand &) { ++commitCalls; }, nullptr);
            t.IsNotNull(accelerated.get(),
                        "the cost-policy fake backend should construct");
            if (!accelerated)
                return;

            t.Equals(accelerated->classify(tiny).reason,
                     GsFallbackReason::CostModel,
                     "a 16-pixel hybrid draw should fall below the threshold");
            t.IsTrue(accelerated->classify(threshold).supported,
                     "a draw exactly at the threshold should stay eligible");
            t.IsTrue(accelerated->setMode(GsRendererMode::GpuStrict),
                     "the fixture should switch to strict mode");
            t.IsTrue(accelerated->classify(tiny).supported,
                     "strict mode should ignore the hybrid performance policy");
            t.IsTrue(accelerated->setMode(GsRendererMode::Verify),
                     "the fixture should switch to verify mode");
            t.IsTrue(accelerated->classify(tiny).supported,
                     "verify mode should exercise every semantic candidate");
            t.IsTrue(accelerated->setMode(GsRendererMode::Hybrid),
                     "the fixture should restore hybrid policy routing");

            ImmediateSoftwareBackend software(vram);
            GsBackendRouter router(software);
            router.setAcceleratedBackend(accelerated.get());
            t.IsTrue(router.setMode(GsRendererMode::Hybrid),
                     "the cost-policy backend should attach to the router");
            router.setCountersEnabled(true);
            const GsSubmissionResult tinyResult = router.submit(tiny);
            const GsSubmissionResult thresholdResult =
                router.submit(threshold);
            t.IsTrue(tinyResult.usedSoftware &&
                         tinyResult.decision.reason ==
                             GsFallbackReason::CostModel,
                     "hybrid should route the tiny draw through software");
            t.IsTrue(thresholdResult.usedAccelerated,
                     "hybrid should queue the threshold-sized draw on the GPU");
            t.Equals(accelerated->pendingCommandCount(), static_cast<size_t>(1u),
                     "the threshold-sized draw should remain pending until observation");

            router.flush(GsFlushReason::DebuggerObservation);
            t.IsTrue(vram == expected,
                     "mixed cost fallback and GPU execution should remain byte-exact");
            t.Equals(commitCalls, 1ull,
                     "only the accelerated threshold draw should publish a commit");

            const GsBackendCounters &counters = router.counters();
            t.Equals(counters.commands, 2ull,
                     "the cost sequence should contain two commands");
            t.Equals(counters.softwareCommands, 1ull,
                     "the tiny command should use software once");
            t.Equals(counters.fallbackCommands, 1ull,
                     "the tiny command should be one explicit fallback");
            t.Equals(counters.acceleratedCommands, 1ull,
                     "the threshold command should be accelerated once");
            t.Equals(counters.drawPixels, 80ull,
                     "pixel counters should include both 4x4 and 8x8 draws");
            t.Equals(counters.softwarePixels, 16ull,
                     "software cost should retain the tiny draw's 16 pixels");
            t.Equals(counters.fallbackPixels, 16ull,
                     "cost fallback should retain its exact pixel count");
            t.Equals(counters.acceleratedPixels, 64ull,
                     "accelerated cost should retain the threshold draw's pixels");
            t.Equals(
                counters.decisions[static_cast<size_t>(
                    GsFallbackReason::CostModel)],
                1ull,
                "the named cost-model decision should be counted once");
            t.Equals(counters.queueHighWatermark, 1ull,
                     "the cost sequence should expose one queued GPU draw");

            const GsVulkanServiceStatistics service =
                accelerated->serviceStatistics();
            t.Equals(service.spriteDrawsCompleted, 1ull,
                     "cost fallback should not become a GPU request");
            t.Equals(service.pipelineBarriers, 8ull,
                     "one upload draw and download should expose eight barriers");
            t.Equals(service.pipelineBinds, 1ull,
                     "the one accelerated draw should bind one fixed pipeline");
            t.Equals(service.pipelineCacheHits, 1ull,
                     "the accelerated draw should reuse one fixed pipeline");
            t.Equals(service.fenceWaits, 3ull,
                     "upload draw and observation should each wait once");
        });

        tc.Run("GS Vulkan presentation latch publishes a complete CPU frame checkpoint", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            std::vector<uint8_t> softwareVram =
                makeVramPattern(0x4652414Du);
            std::vector<uint8_t> acceleratedVram = softwareVram;
            const std::vector<uint8_t> initial = softwareVram;

            constexpr uint32_t displayPage = 5u;
            constexpr uint32_t offscreenPage = 20u;
            GSRegisters softwareRegisters{};
            GSRegisters acceleratedRegisters{};
            configureCt32Display(softwareRegisters, displayPage);
            configureCt32Display(acceleratedRegisters, displayPage);

            GS software;
            GS accelerated;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()),
                &softwareRegisters);
            accelerated.init(
                acceleratedVram.data(),
                static_cast<uint32_t>(acceleratedVram.size()),
                &acceleratedRegisters);

            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig config =
                makeRendererServiceConfig(preflight);
            t.IsTrue(accelerated.configureVulkanRenderer(config),
                     "the frame checkpoint fixture should accept Vulkan configuration");
            if (!preflight.ready())
            {
                t.IsFalse(accelerated.setRendererMode(
                              GsRendererMode::Hybrid),
                          "an unavailable host should skip the frame checkpoint cleanly");
                return;
            }

            t.IsTrue(accelerated.setRendererMode(
                         GsRendererMode::Hybrid),
                     "a capable host should create the hybrid frame fixture");
            if (accelerated.rendererMode() != GsRendererMode::Hybrid)
                return;
            accelerated.setBackendCountersEnabled(true);
            accelerated.resetBackendCounters();

            configureFlatCt32Draws(software, displayPage, 1u);
            configureFlatCt32Draws(accelerated, displayPage, 1u);
            drawFlatCt32Sprite(
                software, 2u * 16u, 3u * 16u,
                18u * 16u, 19u * 16u, 0xA043210Fu);
            drawFlatCt32Sprite(
                accelerated, 2u * 16u, 3u * 16u,
                18u * 16u, 19u * 16u, 0xA043210Fu);

            configureFlatCt32Draws(software, offscreenPage, 1u);
            configureFlatCt32Draws(accelerated, offscreenPage, 1u);
            drawFlatCt32Sprite(
                software, 7u * 16u, 9u * 16u,
                23u * 16u, 25u * 16u, 0xB0876543u);
            drawFlatCt32Sprite(
                accelerated, 7u * 16u, 9u * 16u,
                23u * 16u, 25u * 16u, 0xB0876543u);

            GsBackendCounters counters = accelerated.backendCounters();
            t.Equals(counters.queueDepth, 2ull,
                     "both disjoint frame draws should remain queued before the latch");
            t.Equals(counters.queueHighWatermark, 2ull,
                     "the frame fixture should expose its compatible-run depth");
            GsVulkanRasterBackendStatistics backend =
                accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsCompleted, 0ull,
                     "the frame boundary should own execution of both pending draws");
            const uint64_t cpuPreparationsBeforeFrame =
                backend.cpuAccessPreparations;
            t.IsTrue(std::equal(
                         acceleratedVram.begin() +
                             offscreenPage * GS_VRAM_PAGE_SIZE,
                         acceleratedVram.begin() +
                             (offscreenPage + 1u) * GS_VRAM_PAGE_SIZE,
                         initial.begin() +
                             offscreenPage * GS_VRAM_PAGE_SIZE),
                     "canonical CPU VRAM should still hide the off-display pending draw");

            software.latchHostPresentationFrame();
            accelerated.latchHostPresentationFrame();
            t.IsTrue(acceleratedVram == softwareVram,
                     "the frame checkpoint should publish every GPU-newer page");

            std::vector<uint8_t> softwareFrame;
            std::vector<uint8_t> acceleratedFrame;
            uint32_t softwareWidth = 0u;
            uint32_t softwareHeight = 0u;
            uint32_t acceleratedWidth = 0u;
            uint32_t acceleratedHeight = 0u;
            uint32_t displayFbp = 0u;
            uint32_t sourceFbp = 0u;
            bool usedPreferred = true;
            t.IsTrue(software.copyLatchedHostPresentationFrame(
                         softwareFrame, softwareWidth, softwareHeight),
                     "the software oracle should produce a latched frame");
            t.IsTrue(accelerated.copyLatchedHostPresentationFrame(
                         acceleratedFrame,
                         acceleratedWidth, acceleratedHeight,
                         &displayFbp, &sourceFbp, &usedPreferred),
                     "the synchronized hybrid image should produce a latched frame");
            t.IsTrue(acceleratedFrame == softwareFrame,
                     "hybrid presentation bytes should equal the software frame");
            t.Equals(acceleratedWidth, 64u,
                     "the frame fixture should retain its bounded display width");
            t.Equals(acceleratedHeight, 64u,
                     "the frame fixture should retain its bounded display height");
            t.Equals(acceleratedWidth, softwareWidth,
                     "hybrid and software widths should agree");
            t.Equals(acceleratedHeight, softwareHeight,
                     "hybrid and software heights should agree");
            t.Equals(displayFbp, displayPage,
                     "the latch should retain the configured display page");
            t.Equals(sourceFbp, displayPage,
                     "direct presentation should read the configured display source");
            t.IsFalse(usedPreferred,
                      "the direct frame checkpoint should not require a copy shortcut");
            const size_t drawnPixel =
                (3u * acceleratedWidth + 2u) * 4u;
            t.Equals(static_cast<uint32_t>(
                         acceleratedFrame[drawnPixel + 0u]),
                     0x0Fu,
                     "the latched frame should expose the GPU draw's red byte");
            t.Equals(static_cast<uint32_t>(
                         acceleratedFrame[drawnPixel + 1u]),
                     0x21u,
                     "the latched frame should expose the GPU draw's green byte");
            t.Equals(static_cast<uint32_t>(
                         acceleratedFrame[drawnPixel + 2u]),
                     0x43u,
                     "the latched frame should expose the GPU draw's blue byte");

            const GsVulkanServiceStatistics service =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(service.pageUploadOperationsCompleted, 1ull,
                     "the frame checkpoint should upload both CPU-newer pages once");
            t.Equals(service.pagesUploaded, 2ull,
                     "the frame upload should contain only the two draw pages");
            t.Equals(service.residentSpriteBatchesCompleted, 1ull,
                     "the frame checkpoint should submit one compatible batch");
            t.Equals(service.largestResidentSpriteBatch, 2ull,
                     "the frame batch should retain both disjoint draws");
            t.Equals(service.pageDownloadOperationsCompleted, 1ull,
                     "the frame checkpoint should publish GPU-newer pages once");
            t.Equals(service.pagesDownloaded, 2ull,
                     "presentation should publish display and off-display GPU pages");
            t.Equals(service.queueSubmissions, 3ull,
                     "one upload batch and download should use three submissions");
            t.Equals(service.shaderDispatches, 2ull,
                     "the frame batch should execute both draw dispatches");
            t.Equals(service.pipelineBarriers, 8ull,
                     "the frame checkpoint should expose upload draw and download barriers");
            t.Equals(service.pipelineCacheHits, 1ull,
                     "the frame batch should reuse the fixed sprite pipeline once");
            t.Equals(service.fenceWaits, 3ull,
                     "each submitted frame operation should wait on one fence");

            backend = accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsCompleted, 2ull,
                     "the frame checkpoint should complete both accepted draws");
            t.Equals(backend.residentBatchesCompleted, 1ull,
                     "the backend should expose one frame-boundary batch");
            t.Equals(backend.cpuAccessPreparations -
                         cpuPreparationsBeforeFrame,
                     1ull,
                     "the first frame should request one full CPU publication");
            t.Equals(backend.pageOwnership.gpuNewerPages,
                     static_cast<size_t>(0u),
                     "no hidden GPU writer may survive a frame checkpoint");
            t.Equals(backend.coherency.cpuToGpuPages, 2ull,
                     "coherency should record the two uploaded draw pages");
            t.Equals(backend.coherency.gpuToCpuPages, 2ull,
                     "coherency should record display and off-display publication");
            counters = accelerated.backendCounters();
            t.Equals(counters.queueDepth, 0ull,
                     "the presentation boundary should leave no pending draw");
            t.Equals(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::PresentationLatch)],
                1ull,
                "the first frame should retain one named presentation latch");

            accelerated.latchHostPresentationFrame();
            const GsVulkanServiceStatistics repeatedService =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(repeatedService.queueSubmissions,
                     service.queueSubmissions,
                     "an unchanged second frame should submit no Vulkan work");
            t.Equals(repeatedService.pagesDownloaded,
                     service.pagesDownloaded,
                     "an unchanged second frame should download no page twice");
            backend = accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.cpuAccessPreparations -
                         cpuPreparationsBeforeFrame,
                     2ull,
                     "each conservative frame should still prepare CPU visibility");
            counters = accelerated.backendCounters();
            t.Equals(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::PresentationLatch)],
                2ull,
                "both frame boundaries should remain observable");
            t.Equals(repeatedService.validationErrors, 0u,
                     "frame synchronization should remain validation-clean");
            t.Equals(repeatedService.validationWarnings, 0u,
                     "frame synchronization should emit no validation warnings");
        });

        tc.Run("GS Vulkan fixed-seed transition streams match software at every observation", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            std::vector<uint8_t> softwareVram =
                makeVramPattern(0x5452414Eu);
            std::vector<uint8_t> acceleratedVram = softwareVram;

            constexpr std::array<uint32_t, 8> pages{
                1u, 5u, 9u, 13u, 17u, 21u, 25u, 29u};
            GSRegisters softwareRegisters{};
            GSRegisters acceleratedRegisters{};
            configureCt32Display(softwareRegisters, pages[0]);
            configureCt32Display(acceleratedRegisters, pages[0]);

            GS software;
            GS accelerated;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()),
                &softwareRegisters);
            accelerated.init(
                acceleratedVram.data(),
                static_cast<uint32_t>(acceleratedVram.size()),
                &acceleratedRegisters);

            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig serviceConfig =
                makeRendererServiceConfig(preflight);
            GsVulkanRasterBackendConfig backendConfig{};
            backendConfig.maximumResidentBatchCommands = 8u;
            backendConfig.minimumHybridSpritePixels = 64u;
            t.IsTrue(accelerated.configureVulkanRenderer(
                         serviceConfig, backendConfig),
                     "the randomized transition fixture should accept Vulkan configuration");
            if (!preflight.ready())
            {
                t.IsFalse(accelerated.setRendererMode(
                              GsRendererMode::Hybrid),
                          "an unavailable host should skip randomized GPU transitions cleanly");
                return;
            }

            t.IsTrue(accelerated.setRendererMode(
                         GsRendererMode::Hybrid),
                     "a capable host should create the randomized hybrid fixture");
            if (accelerated.rendererMode() != GsRendererMode::Hybrid)
                return;
            accelerated.setBackendCountersEnabled(true);
            accelerated.resetBackendCounters();

            uint32_t randomState = 0x6D2B79F5u;
            const auto randomBelow = [&](uint32_t bound)
            {
                return nextRandom(randomState) % bound;
            };
            const auto randomPage = [&]()
            {
                return pages[randomBelow(
                    static_cast<uint32_t>(pages.size()))];
            };
            const auto drawEligiblePair = [&](uint32_t page)
            {
                const uint16_t x0 = static_cast<uint16_t>(
                    randomBelow(40u));
                const uint16_t y0 = static_cast<uint16_t>(
                    randomBelow(16u));
                const uint16_t width = static_cast<uint16_t>(
                    8u + randomBelow(9u));
                const uint16_t height = static_cast<uint16_t>(
                    8u + randomBelow(5u));
                const uint32_t color =
                    0x80000000u | nextRandom(randomState);
                configureFlatCt32Draws(software, page, 1u);
                configureFlatCt32Draws(accelerated, page, 1u);
                drawFlatCt32Sprite(
                    software,
                    static_cast<uint16_t>(x0 * 16u),
                    static_cast<uint16_t>(y0 * 16u),
                    static_cast<uint16_t>((x0 + width) * 16u),
                    static_cast<uint16_t>((y0 + height) * 16u),
                    color);
                drawFlatCt32Sprite(
                    accelerated,
                    static_cast<uint16_t>(x0 * 16u),
                    static_cast<uint16_t>(y0 * 16u),
                    static_cast<uint16_t>((x0 + width) * 16u),
                    static_cast<uint16_t>((y0 + height) * 16u),
                    color);
            };

            uint64_t forcedObservations = 0u;
            uint64_t frameObservations = 0u;
            const auto compareFullVram = [&](
                const char *boundary,
                size_t operationIndex) -> bool
            {
                ++forcedObservations;
                const auto difference = std::mismatch(
                    softwareVram.begin(), softwareVram.end(),
                    acceleratedVram.begin(), acceleratedVram.end());
                if (difference.first == softwareVram.end())
                    return true;
                const size_t byteOffset = static_cast<size_t>(
                    difference.first - softwareVram.begin());
                t.IsTrue(
                    false,
                    std::string("fixed-seed VRAM mismatch after ") +
                        boundary + " at operation " +
                        std::to_string(operationIndex) +
                        ", byte " + std::to_string(byteOffset) +
                        ", page " + std::to_string(
                            byteOffset / GS_VRAM_PAGE_SIZE));
                return false;
            };
            const auto forceObservation = [&](
                uint32_t kind,
                size_t operationIndex) -> bool
            {
                const char *name = nullptr;
                switch (kind)
                {
                case 0u:
                    (void)software.getDebugSnapshot();
                    (void)accelerated.getDebugSnapshot();
                    name = "debugger observation";
                    break;
                case 1u:
                {
                    software.latchHostPresentationFrame();
                    accelerated.latchHostPresentationFrame();
                    std::vector<uint8_t> softwareFrame;
                    std::vector<uint8_t> acceleratedFrame;
                    uint32_t softwareWidth = 0u;
                    uint32_t softwareHeight = 0u;
                    uint32_t acceleratedWidth = 0u;
                    uint32_t acceleratedHeight = 0u;
                    const bool softwareFrameReady =
                        software.copyLatchedHostPresentationFrame(
                            softwareFrame,
                            softwareWidth, softwareHeight);
                    const bool acceleratedFrameReady =
                        accelerated.copyLatchedHostPresentationFrame(
                            acceleratedFrame,
                            acceleratedWidth, acceleratedHeight);
                    if (!softwareFrameReady ||
                        acceleratedFrameReady != softwareFrameReady ||
                        acceleratedWidth != softwareWidth ||
                        acceleratedHeight != softwareHeight ||
                        acceleratedFrame != softwareFrame)
                    {
                        t.IsTrue(
                            false,
                            "fixed-seed frame bytes diverged at operation " +
                                std::to_string(operationIndex));
                        return false;
                    }
                    ++frameObservations;
                    name = "presentation latch";
                    break;
                }
                case 2u:
                    software.writeRegister(GS_REG_FINISH, 0u);
                    accelerated.writeRegister(GS_REG_FINISH, 0u);
                    name = "FINISH";
                    break;
                default:
                    (void)software.captureReplayState();
                    (void)accelerated.captureReplayState();
                    name = "save-state observation";
                    break;
                }
                return compareFullVram(name, operationIndex);
            };

            enum class TransitionAction : uint8_t
            {
                GpuDraw,
                PointFallback,
                HostUpload,
                CpuReadback,
                CostFallback,
                ClutLoad,
                Feedback,
                LocalCopy,
                CpuClear,
                Count,
            };
            constexpr size_t actionCount =
                static_cast<size_t>(TransitionAction::Count);
            std::array<uint64_t, actionCount> actionCounts{};
            constexpr size_t cycleCount = 8u;
            size_t operationIndex = 0u;
            for (size_t cycle = 0u; cycle < cycleCount; ++cycle)
            {
                std::array<TransitionAction, actionCount> order{
                    TransitionAction::GpuDraw,
                    TransitionAction::PointFallback,
                    TransitionAction::HostUpload,
                    TransitionAction::CpuReadback,
                    TransitionAction::CostFallback,
                    TransitionAction::ClutLoad,
                    TransitionAction::Feedback,
                    TransitionAction::LocalCopy,
                    TransitionAction::CpuClear,
                };
                for (size_t index = order.size() - 1u;
                     index > 0u; --index)
                {
                    const size_t other = randomBelow(
                        static_cast<uint32_t>(index + 1u));
                    std::swap(order[index], order[other]);
                }

                for (TransitionAction action : order)
                {
                    ++operationIndex;
                    ++actionCounts[static_cast<size_t>(action)];
                    const uint32_t page = randomPage();
                    switch (action)
                    {
                    case TransitionAction::GpuDraw:
                        drawEligiblePair(page);
                        break;
                    case TransitionAction::PointFallback:
                    {
                        configureFlatCt32Draws(software, page, 1u);
                        configureFlatCt32Draws(accelerated, page, 1u);
                        const uint16_t x = static_cast<uint16_t>(
                            randomBelow(64u) * 16u);
                        const uint16_t y = static_cast<uint16_t>(
                            randomBelow(32u) * 16u);
                        const uint32_t color =
                            0x80000000u | nextRandom(randomState);
                        drawFlatCt32Point(software, x, y, color);
                        drawFlatCt32Point(accelerated, x, y, color);
                        break;
                    }
                    case TransitionAction::HostUpload:
                    {
                        std::array<uint32_t, 4> pixels{};
                        for (uint32_t &pixel : pixels)
                            pixel = nextRandom(randomState);
                        const size_t pixelCount =
                            1u + randomBelow(
                                static_cast<uint32_t>(pixels.size()));
                        const uint16_t x = static_cast<uint16_t>(
                            randomBelow(60u));
                        const uint16_t y = static_cast<uint16_t>(
                            randomBelow(32u));
                        const std::span<const uint32_t> payload(
                            pixels.data(), pixelCount);
                        uploadCt32Pixels(
                            software, page * 32u, 1u,
                            x, y, payload);
                        uploadCt32Pixels(
                            accelerated, page * 32u, 1u,
                            x, y, payload);
                        break;
                    }
                    case TransitionAction::CpuReadback:
                    {
                        const uint16_t x = static_cast<uint16_t>(
                            randomBelow(61u));
                        const uint16_t y = static_cast<uint16_t>(
                            randomBelow(29u));
                        const std::vector<uint8_t> softwareBytes =
                            readCt32Pixels(
                                software, page * 32u, 1u,
                                x, y, 3u, 3u);
                        const std::vector<uint8_t> acceleratedBytes =
                            readCt32Pixels(
                                accelerated, page * 32u, 1u,
                                x, y, 3u, 3u);
                        if (acceleratedBytes != softwareBytes)
                        {
                            t.IsTrue(
                                false,
                                "fixed-seed scoped readback diverged at operation " +
                                    std::to_string(operationIndex));
                            return;
                        }
                        break;
                    }
                    case TransitionAction::CostFallback:
                    {
                        configureFlatCt32Draws(software, page, 1u);
                        configureFlatCt32Draws(accelerated, page, 1u);
                        const uint16_t x0 = static_cast<uint16_t>(
                            randomBelow(60u));
                        const uint16_t y0 = static_cast<uint16_t>(
                            randomBelow(28u));
                        const uint32_t color =
                            0x80000000u | nextRandom(randomState);
                        drawFlatCt32Sprite(
                            software,
                            static_cast<uint16_t>(x0 * 16u),
                            static_cast<uint16_t>(y0 * 16u),
                            static_cast<uint16_t>((x0 + 4u) * 16u),
                            static_cast<uint16_t>((y0 + 4u) * 16u),
                            color);
                        drawFlatCt32Sprite(
                            accelerated,
                            static_cast<uint16_t>(x0 * 16u),
                            static_cast<uint16_t>(y0 * 16u),
                            static_cast<uint16_t>((x0 + 4u) * 16u),
                            static_cast<uint16_t>((y0 + 4u) * 16u),
                            color);
                        break;
                    }
                    case TransitionAction::ClutLoad:
                    {
                        drawEligiblePair(page);
                        const uint64_t indexedTex0 =
                            (1ull << 14u) |
                            (static_cast<uint64_t>(GS_PSM_T8) << 20u) |
                            (5ull << 26u) |
                            (5ull << 30u) |
                            (1ull << 34u) |
                            (static_cast<uint64_t>(page * 32u) << 37u) |
                            (static_cast<uint64_t>(GS_PSM_CT32) << 51u) |
                            (1ull << 61u);
                        software.writeRegister(GS_REG_TEX0_1, indexedTex0);
                        accelerated.writeRegister(
                            GS_REG_TEX0_1, indexedTex0);
                        break;
                    }
                    case TransitionAction::Feedback:
                    {
                        drawEligiblePair(page);
                        const uint64_t feedbackTex0 =
                            static_cast<uint64_t>(page * 32u) |
                            (1ull << 14u) |
                            (static_cast<uint64_t>(GS_PSM_CT32) << 20u) |
                            (5ull << 26u) |
                            (5ull << 30u) |
                            (1ull << 34u) |
                            (1ull << 35u);
                        software.writeRegister(GS_REG_TEX0_1, feedbackTex0);
                        accelerated.writeRegister(
                            GS_REG_TEX0_1, feedbackTex0);
                        software.beginRenderBatch();
                        accelerated.beginRenderBatch();
                        const uint16_t x0 = static_cast<uint16_t>(
                            randomBelow(48u));
                        const uint16_t y0 = static_cast<uint16_t>(
                            randomBelow(16u));
                        drawRecursiveCt32Sprite(
                            software,
                            static_cast<uint16_t>(x0 * 16u),
                            static_cast<uint16_t>(y0 * 16u),
                            static_cast<uint16_t>((x0 + 8u) * 16u),
                            static_cast<uint16_t>((y0 + 8u) * 16u),
                            0x80808080u);
                        drawRecursiveCt32Sprite(
                            accelerated,
                            static_cast<uint16_t>(x0 * 16u),
                            static_cast<uint16_t>(y0 * 16u),
                            static_cast<uint16_t>((x0 + 8u) * 16u),
                            static_cast<uint16_t>((y0 + 8u) * 16u),
                            0x80808080u);
                        software.endRenderBatch();
                        accelerated.endRenderBatch();
                        if (!compareFullVram(
                                "feedback explicit boundary",
                                operationIndex))
                        {
                            return;
                        }
                        break;
                    }
                    case TransitionAction::LocalCopy:
                    {
                        const size_t sourceIndex = randomBelow(
                            static_cast<uint32_t>(pages.size()));
                        const size_t destinationIndex =
                            (sourceIndex + 1u + randomBelow(
                                static_cast<uint32_t>(pages.size() - 1u))) %
                            pages.size();
                        const uint32_t sourcePage = pages[sourceIndex];
                        const uint32_t destinationPage =
                            pages[destinationIndex];
                        drawEligiblePair(sourcePage);
                        drawEligiblePair(destinationPage);
                        const uint16_t sourceX = static_cast<uint16_t>(
                            randomBelow(57u));
                        const uint16_t sourceY = static_cast<uint16_t>(
                            randomBelow(29u));
                        const uint16_t destinationX =
                            static_cast<uint16_t>(randomBelow(57u));
                        const uint16_t destinationY =
                            static_cast<uint16_t>(randomBelow(29u));
                        copyCt32Pixels(
                            software,
                            sourcePage * 32u,
                            destinationPage * 32u,
                            1u,
                            sourceX, sourceY,
                            destinationX, destinationY,
                            4u, 3u);
                        copyCt32Pixels(
                            accelerated,
                            sourcePage * 32u,
                            destinationPage * 32u,
                            1u,
                            sourceX, sourceY,
                            destinationX, destinationY,
                            4u, 3u);
                        break;
                    }
                    case TransitionAction::CpuClear:
                    {
                        configureFlatCt32Draws(software, page, 1u);
                        configureFlatCt32Draws(accelerated, page, 1u);
                        const uint32_t color =
                            0x80000000u | nextRandom(randomState);
                        const bool softwareCleared =
                            software.clearFramebufferContext(0u, color);
                        const bool acceleratedCleared =
                            accelerated.clearFramebufferContext(0u, color);
                        if (!softwareCleared ||
                            acceleratedCleared != softwareCleared)
                        {
                            t.IsTrue(
                                false,
                                "fixed-seed CPU clear failed at operation " +
                                    std::to_string(operationIndex));
                            return;
                        }
                        break;
                    }
                    case TransitionAction::Count:
                        break;
                    }

                    if ((operationIndex & 3u) == 0u)
                    {
                        const uint32_t observationKind =
                            static_cast<uint32_t>(
                                (operationIndex / 4u - 1u) % 4u);
                        if (!forceObservation(
                                observationKind, operationIndex))
                        {
                            return;
                        }
                    }
                }
            }

            (void)software.getDebugSnapshot();
            (void)accelerated.getDebugSnapshot();
            if (!compareFullVram(
                    "final debugger observation", operationIndex))
            {
                return;
            }

            for (size_t index = 0u; index < actionCounts.size(); ++index)
            {
                t.Equals(actionCounts[index],
                         static_cast<uint64_t>(cycleCount),
                         "every transition action should execute once per shuffled cycle");
            }
            t.Equals(operationIndex, cycleCount * actionCount,
                     "the randomized transition stream should remain bounded");
            t.Equals(forcedObservations, 27ull,
                     "scheduled feedback and final boundaries should all compare VRAM");
            t.Equals(frameObservations, 5ull,
                     "five scheduled frame observations should compare host pixels");

            const GsBackendCounters counters =
                accelerated.backendCounters();
            t.Equals(counters.commands, 64ull,
                     "the shuffled stream should submit its exact draw count");
            t.Equals(counters.acceleratedCommands, 40ull,
                     "all semantically eligible non-tiny sprites should accelerate");
            t.Equals(counters.softwareCommands, 24ull,
                     "point cost and feedback commands should use software");
            t.Equals(counters.fallbackCommands, 24ull,
                     "every software command should retain an explicit fallback");
            t.Equals(
                counters.decisions[static_cast<size_t>(
                    GsFallbackReason::UnsupportedPrimitive)],
                8ull,
                "each shuffled cycle should contain one point fallback");
            t.Equals(
                counters.decisions[static_cast<size_t>(
                    GsFallbackReason::CostModel)],
                8ull,
                "each shuffled cycle should contain one tiny cost fallback");
            t.Equals(
                counters.decisions[static_cast<size_t>(
                    GsFallbackReason::Textured)],
                8ull,
                "each shuffled cycle should contain one feedback fallback");
            t.IsTrue(counters.queueHighWatermark >= 2ull,
                     "local-copy setup should expose compatible GPU queueing");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::Transfer)] >= 24ull,
                "uploads local copies and clears should cross transfer boundaries");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::CpuReadback)] >= 8ull,
                "every readback action should retain its scoped boundary");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::ClutHazard)] >= 8ull,
                "every CLUT action should retain its sampled-page boundary");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::FeedbackSnapshot)] >= 8ull,
                "every recursive draw should retain its feedback boundary");

            const GsVulkanRasterBackendStatistics backend =
                accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsCompleted, 40ull,
                     "every accelerated transition command should complete");
            t.Equals(backend.pageOwnership.gpuNewerPages,
                     static_cast<size_t>(0u),
                     "the final observation should publish every GPU writer");
            t.Equals(backend.coherency.rejectedTransitions, 0ull,
                     "the randomized stream should never create competing writers");
            t.IsTrue(backend.residentBatchesCompleted > 0ull,
                     "the randomized stream should exercise resident batches");
            const GsVulkanServiceStatistics service =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(service.spriteDrawsCompleted, 40ull,
                     "the service should execute every accepted sprite once");
            t.IsTrue(service.pageUploadOperationsCompleted > 0ull,
                     "the randomized stream should upload CPU-newer pages");
            t.IsTrue(service.pageDownloadOperationsCompleted > 0ull,
                     "forced observations should publish GPU-newer pages");
            t.Equals(service.fenceTimeouts, 0ull,
                     "the bounded transition stream should not time out");
            t.Equals(service.validationErrors, 0u,
                     "randomized transitions should remain validation-clean");
            t.Equals(service.validationWarnings, 0u,
                     "randomized transitions should emit no validation warnings");
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

        tc.Run("GS Vulkan hybrid scopes host transfers and readbacks to overlapping pages", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            std::vector<uint8_t> softwareVram =
                makeVramPattern(0x5452414Eu);
            std::vector<uint8_t> acceleratedVram = softwareVram;
            const std::vector<uint8_t> initial = softwareVram;
            GS software;
            GS accelerated;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()), nullptr);
            accelerated.init(
                acceleratedVram.data(),
                static_cast<uint32_t>(acceleratedVram.size()), nullptr);

            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig config =
                makeRendererServiceConfig(preflight);
            t.IsTrue(accelerated.configureVulkanRenderer(config),
                     "the scoped transfer fixture should accept Vulkan configuration");
            if (!preflight.ready())
            {
                t.IsFalse(accelerated.setRendererMode(
                              GsRendererMode::Hybrid),
                          "an unavailable host should skip scoped transfer execution cleanly");
                return;
            }

            t.IsTrue(accelerated.setRendererMode(
                         GsRendererMode::Hybrid),
                     "a capable host should create the hybrid transfer fixture");
            if (accelerated.rendererMode() != GsRendererMode::Hybrid)
                return;
            accelerated.setBackendCountersEnabled(true);
            accelerated.resetBackendCounters();

            constexpr uint32_t firstPage = 5u;
            constexpr uint32_t secondPage = 20u;
            configureFlatCt32Draws(software, firstPage, 1u);
            configureFlatCt32Draws(accelerated, firstPage, 1u);
            drawFlatCt32Sprite(
                software, 2u * 16u, 3u * 16u,
                15u * 16u, 14u * 16u, 0xA043210Fu);
            drawFlatCt32Sprite(
                accelerated, 2u * 16u, 3u * 16u,
                15u * 16u, 14u * 16u, 0xA043210Fu);

            configureFlatCt32Draws(software, secondPage, 1u);
            configureFlatCt32Draws(accelerated, secondPage, 1u);
            drawFlatCt32Sprite(
                software, 4u * 16u, 5u * 16u,
                18u * 16u, 16u * 16u, 0xB0876543u);
            drawFlatCt32Sprite(
                accelerated, 4u * 16u, 5u * 16u,
                18u * 16u, 16u * 16u, 0xB0876543u);

            GsVulkanServiceStatistics service =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(service.pageUploadOperationsCompleted, 0ull,
                     "compatible draws should defer their upload until a boundary");
            t.Equals(service.pagesUploaded, 0ull,
                     "pending draws should not transfer pages early");
            t.Equals(service.pageDownloadOperationsCompleted, 0ull,
                     "resident draws should not publish either page to CPU");
            GsVulkanRasterBackendStatistics backend =
                accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsAttempted, 2ull,
                     "both compatible draws should be accepted immediately");
            t.Equals(backend.commandsCompleted, 0ull,
                     "accepted draws should remain incomplete before a boundary");
            GsBackendCounters counters = accelerated.backendCounters();
            t.Equals(counters.queueDepth, 2ull,
                     "the router should expose both pending compatible draws");
            t.Equals(counters.queueHighWatermark, 2ull,
                     "the router should retain the compatible-run high watermark");

            constexpr std::array<uint32_t, 3> uploadPixels{
                0xC0010203u, 0xD0040506u, 0xE0070809u};
            uploadCt32Pixels(
                software, firstPage * 32u, 1u, 6u, 8u,
                uploadPixels);
            uploadCt32Pixels(
                accelerated, firstPage * 32u, 1u, 6u, 8u,
                uploadPixels);

            service = accelerated.vulkanRendererServiceStatistics();
            t.Equals(service.pageUploadOperationsCompleted, 1ull,
                     "the transfer boundary should upload both draw pages together");
            t.Equals(service.pagesUploaded, 2ull,
                     "the batched upload should contain exactly two 8 KiB pages");
            t.Equals(service.residentSpriteBatchesCompleted, 1ull,
                     "the transfer boundary should execute one resident draw batch");
            t.Equals(service.largestResidentSpriteBatch, 2ull,
                     "the resident service should observe both compatible draws");
            t.Equals(service.queueSubmissions, 3ull,
                     "one upload one draw batch and one download should require three submissions");
            t.Equals(service.shaderDispatches, 2ull,
                     "both batched sprites should retain exact dispatch accounting");
            t.Equals(service.pipelineBarriers, 8ull,
                     "upload batch and download should expose all submitted barriers");
            t.Equals(service.pipelineBinds, 1ull,
                     "the two compatible draws should bind their fixed pipeline once");
            t.Equals(service.pipelineCacheHits, 1ull,
                     "the compatible batch should reuse the fixed sprite pipeline");
            t.Equals(service.fenceWaits, 3ull,
                     "upload batch and download should each wait on one fence");
            t.Equals(service.pageDownloadOperationsCompleted, 1ull,
                     "a partial host upload should preserve one overlapping GPU page");
            t.Equals(service.pagesDownloaded, 1ull,
                     "the upload must not publish an unrelated resident page");
            backend = accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsCompleted, 2ull,
                     "the transfer boundary should complete both accepted draws");
            t.Equals(backend.residentBatchesCompleted, 1ull,
                     "the backend should submit one compatible resident batch");
            t.Equals(backend.pageOwnership.gpuNewerPages,
                     static_cast<size_t>(1u),
                     "the disjoint draw page should remain GPU-newer after the upload");
            t.IsTrue(std::equal(
                         acceleratedVram.begin() +
                             firstPage * GS_VRAM_PAGE_SIZE,
                         acceleratedVram.begin() +
                             (firstPage + 1u) * GS_VRAM_PAGE_SIZE,
                         softwareVram.begin() +
                             firstPage * GS_VRAM_PAGE_SIZE),
                     "the overlapping page should contain the GPU draw plus host upload");
            t.IsTrue(std::equal(
                         acceleratedVram.begin() +
                             secondPage * GS_VRAM_PAGE_SIZE,
                         acceleratedVram.begin() +
                             (secondPage + 1u) * GS_VRAM_PAGE_SIZE,
                         initial.begin() +
                             secondPage * GS_VRAM_PAGE_SIZE),
                     "the unrelated GPU-newer page should remain stale on canonical CPU VRAM");

            const std::vector<uint8_t> softwareReadback =
                readCt32Pixels(
                    software, secondPage * 32u, 1u,
                    6u, 7u, 3u, 2u);
            const std::vector<uint8_t> acceleratedReadback =
                readCt32Pixels(
                    accelerated, secondPage * 32u, 1u,
                    6u, 7u, 3u, 2u);
            t.IsTrue(acceleratedReadback == softwareReadback,
                     "local-to-host should observe the resident GPU result exactly");
            t.IsTrue(acceleratedVram == softwareVram,
                     "the two scoped CPU accesses should reconstruct the complete image");

            service = accelerated.vulkanRendererServiceStatistics();
            t.Equals(service.pageDownloadOperationsCompleted, 2ull,
                     "upload preservation and readback should issue separate downloads");
            t.Equals(service.pagesDownloaded, 2ull,
                     "each disjoint GPU-newer page should publish exactly once");
            backend = accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.pageOwnership.gpuNewerPages,
                     static_cast<size_t>(0u),
                     "the final scoped readback should leave no hidden GPU writer");
            t.Equals(backend.coherency.gpuToCpuPages, 2ull,
                     "coherency accounting should match the exact observed pages");

            counters = accelerated.backendCounters();
            t.Equals(counters.queueDepth, 0ull,
                     "the scoped CPU boundaries should leave no pending draw");
            t.Equals(counters.queueHighWatermark, 2ull,
                     "queue drainage should retain the observed high watermark");
            t.Equals(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::Transfer)],
                1ull,
                "the host upload should retain one named transfer boundary");
            t.Equals(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::CpuReadback)],
                1ull,
                "the local-to-host operation should retain one named readback boundary");
            t.Equals(service.validationErrors, 0u,
                     "scoped transfer synchronization should remain validation-clean");
            t.Equals(service.validationWarnings, 0u,
                     "scoped transfer synchronization should emit no validation warnings");
        });

        tc.Run("GS Vulkan hybrid scopes CLUT and feedback snapshots to sampled pages", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            std::vector<uint8_t> softwareVram =
                makeVramPattern(0x434C5554u);
            std::vector<uint8_t> acceleratedVram = softwareVram;
            const std::vector<uint8_t> initial = softwareVram;
            GS software;
            GS accelerated;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()), nullptr);
            accelerated.init(
                acceleratedVram.data(),
                static_cast<uint32_t>(acceleratedVram.size()), nullptr);

            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig config =
                makeRendererServiceConfig(preflight);
            t.IsTrue(accelerated.configureVulkanRenderer(config),
                     "the scoped sampler fixture should accept Vulkan configuration");
            if (!preflight.ready())
            {
                t.IsFalse(accelerated.setRendererMode(
                              GsRendererMode::Hybrid),
                          "an unavailable host should skip scoped sampler execution cleanly");
                return;
            }

            t.IsTrue(accelerated.setRendererMode(
                         GsRendererMode::Hybrid),
                     "a capable host should create the hybrid sampler fixture");
            if (accelerated.rendererMode() != GsRendererMode::Hybrid)
                return;
            accelerated.setBackendCountersEnabled(true);
            accelerated.resetBackendCounters();

            constexpr uint32_t clutPage = 9u;
            constexpr uint32_t feedbackPage = 27u;
            configureFlatCt32Draws(software, clutPage, 1u);
            configureFlatCt32Draws(accelerated, clutPage, 1u);
            drawFlatCt32Sprite(
                software, 0u, 0u,
                16u * 16u, 16u * 16u, 0xA0112233u);
            drawFlatCt32Sprite(
                accelerated, 0u, 0u,
                16u * 16u, 16u * 16u, 0xA0112233u);

            configureFlatCt32Draws(software, feedbackPage, 1u);
            configureFlatCt32Draws(accelerated, feedbackPage, 1u);
            drawFlatCt32Sprite(
                software, 3u * 16u, 4u * 16u,
                19u * 16u, 18u * 16u, 0xB0445566u);
            drawFlatCt32Sprite(
                accelerated, 3u * 16u, 4u * 16u,
                19u * 16u, 18u * 16u, 0xB0445566u);

            const uint64_t indexedTex0 =
                (1ull << 14u) |
                (static_cast<uint64_t>(GS_PSM_T8) << 20u) |
                (5ull << 26u) |
                (5ull << 30u) |
                (1ull << 34u) |
                (static_cast<uint64_t>(clutPage * 32u) << 37u) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 51u) |
                (1ull << 61u);
            software.writeRegister(GS_REG_TEX0_1, indexedTex0);
            accelerated.writeRegister(GS_REG_TEX0_1, indexedTex0);

            GsVulkanServiceStatistics service =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(service.pageDownloadOperationsCompleted, 1ull,
                     "CLUT loading should publish its one overlapping page");
            t.Equals(service.pagesDownloaded, 1ull,
                     "CLUT loading must leave the feedback page resident");
            t.IsTrue(std::equal(
                         acceleratedVram.begin() +
                             clutPage * GS_VRAM_PAGE_SIZE,
                         acceleratedVram.begin() +
                             (clutPage + 1u) * GS_VRAM_PAGE_SIZE,
                         softwareVram.begin() +
                             clutPage * GS_VRAM_PAGE_SIZE),
                     "the CLUT page should expose the completed GPU draw");
            t.IsTrue(std::equal(
                         acceleratedVram.begin() +
                             feedbackPage * GS_VRAM_PAGE_SIZE,
                         acceleratedVram.begin() +
                             (feedbackPage + 1u) * GS_VRAM_PAGE_SIZE,
                         initial.begin() +
                             feedbackPage * GS_VRAM_PAGE_SIZE),
                     "CLUT loading should not publish an unrelated GPU page");

            const uint64_t feedbackTex0 =
                static_cast<uint64_t>(feedbackPage * 32u) |
                (1ull << 14u) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 20u) |
                (5ull << 26u) |
                (5ull << 30u) |
                (1ull << 34u) |
                (1ull << 35u);
            software.writeRegister(GS_REG_TEX0_1, feedbackTex0);
            accelerated.writeRegister(GS_REG_TEX0_1, feedbackTex0);
            software.beginRenderBatch();
            accelerated.beginRenderBatch();
            drawRecursiveCt32Sprite(
                software, 5u * 16u, 6u * 16u,
                13u * 16u, 12u * 16u, 0x80808080u);
            drawRecursiveCt32Sprite(
                accelerated, 5u * 16u, 6u * 16u,
                13u * 16u, 12u * 16u, 0x80808080u);

            service = accelerated.vulkanRendererServiceStatistics();
            t.Equals(service.pageDownloadOperationsCompleted, 2ull,
                     "recursive feedback should publish its sampled page once");
            t.Equals(service.pagesDownloaded, 2ull,
                     "CLUT and feedback should publish exactly two disjoint pages");
            GsVulkanRasterBackendStatistics backend =
                accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.pageOwnership.gpuNewerPages,
                     static_cast<size_t>(0u),
                     "feedback fallback should leave no unrelated GPU writer");

            software.endRenderBatch();
            accelerated.endRenderBatch();
            t.IsTrue(acceleratedVram == softwareVram,
                     "scoped CLUT and feedback boundaries should retain exact software semantics");

            const GsBackendCounters counters =
                accelerated.backendCounters();
            t.Equals(counters.fallbackCommands, 1ull,
                     "the recursive textured draw should remain one explicit fallback");
            t.Equals(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::ClutHazard)],
                1ull,
                "the indexed TEX0 write should retain one named CLUT boundary");
            t.Equals(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::FeedbackSnapshot)],
                1ull,
                "snapshot creation should retain one named feedback boundary");
            t.Equals(service.validationErrors, 0u,
                     "scoped sampler synchronization should remain validation-clean");
            t.Equals(service.validationWarnings, 0u,
                     "scoped sampler synchronization should emit no validation warnings");
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

        tc.Run("Vulkan resident page operations preserve unselected bytes", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig config =
                makeRendererServiceConfig(preflight);
            GsVulkanCapabilityReport creationReport{};
            std::string creationError;
            std::unique_ptr<GsVulkanService> service =
                GsVulkanService::create(
                    config, &creationReport, &creationError);
            if (!preflight.ready())
            {
                t.IsNull(service.get(),
                         "an unavailable host should skip resident page execution cleanly");
                t.IsFalse(creationReport.ready(),
                          "a skipped resident service should retain its capability result");
                return;
            }
            t.IsNotNull(service.get(),
                        "a suitable device should create the resident page service");
            t.IsTrue(creationError.empty(),
                     "successful resident page service creation should clear its diagnostic");
            if (!service)
                return;

            GsVramPageMask allPages;
            allPages.setAll();
            GsVramPageMask sparsePages;
            for (const size_t page : {0u, 1u, 7u, 255u, 256u, 511u})
                sparsePages.set(page);
            const auto countRuns = [](const GsVramPageMask &pages)
            {
                uint64_t runs = 0u;
                bool insideRun = false;
                for (size_t page = 0u; page < GS_VRAM_PAGE_COUNT; ++page)
                {
                    const bool selected = pages.test(page);
                    if (selected && !insideRun)
                        ++runs;
                    insideRun = selected;
                }
                return runs;
            };
            const auto copyPages = [](
                std::vector<uint8_t> &destination,
                const std::vector<uint8_t> &source,
                const GsVramPageMask &pages)
            {
                for (size_t page = 0u; page < GS_VRAM_PAGE_COUNT; ++page)
                {
                    if (!pages.test(page))
                        continue;
                    const size_t offset = page * GS_VRAM_PAGE_SIZE;
                    std::memcpy(
                        destination.data() + offset,
                        source.data() + offset,
                        GS_VRAM_PAGE_SIZE);
                }
            };

            const std::vector<uint8_t> initial =
                makeVramPattern(0x50414731u);
            const std::vector<uint8_t> replacement =
                makeVramPattern(0x50414732u);
            std::vector<uint8_t> expected = initial;
            copyPages(expected, replacement, sparsePages);
            std::string operationError;
            t.IsTrue(service->uploadVramPages(
                         initial, allPages, &operationError),
                     "the initial full-page upload should establish resident VRAM");
            t.IsTrue(operationError.empty(),
                     "the initial upload should clear its diagnostic");
            t.IsTrue(service->uploadVramPages(
                         replacement, sparsePages, &operationError),
                     "a sparse upload should accept first last and adjacent pages");
            t.IsTrue(operationError.empty(),
                     "the sparse upload should clear its diagnostic");

            std::vector<uint8_t> sparseDownload(
                GS_VULKAN_VRAM_SIZE, 0xA5u);
            std::vector<uint8_t> expectedSparse = sparseDownload;
            copyPages(expectedSparse, expected, sparsePages);
            t.IsTrue(service->downloadVramPages(
                         sparseDownload, sparsePages, &operationError),
                     "a sparse download should complete");
            t.IsTrue(sparseDownload == expectedSparse,
                     "a sparse download must leave every unselected byte untouched");

            std::vector<uint8_t> residentBefore(
                GS_VULKAN_VRAM_SIZE, 0u);
            t.IsTrue(service->downloadVramPages(
                         residentBefore, allPages, &operationError),
                     "a full observation should reconstruct resident VRAM");
            t.IsTrue(residentBefore == expected,
                     "sparse upload ranges must preserve every other resident page");

            const GsDrawCommand command = makeCt32SpriteCommand(
                91u, 511u, 1u,
                {0u, 127u, 0u, 63u}, {0u, 0u},
                60u * 16u, 28u * 16u,
                68u * 16u, 36u * 16u,
                0xD1C2B3A4u);
            GsVulkanCt32Sprite sprite{};
            const GsBackendDecision decision =
                prepareGsVulkanCt32Sprite(command, sprite);
            t.IsTrue(decision.supported,
                     "the resident wrap fixture should satisfy the exact predicate");
            if (!decision.supported)
                return;
            std::vector<uint8_t> expectedAfter = expected;
            applyCt32SpriteCpu(expectedAfter, sprite);
            t.IsTrue(service->executeResidentCt32Sprite(
                         sprite, &operationError),
                     "a resident sprite should execute without implicit transfers");
            t.IsTrue(operationError.empty(),
                     "resident sprite execution should clear its diagnostic");

            const GsVramPageMask writePages =
                command.resources().writePages;
            t.IsTrue(writePages.any(),
                     "the resident sprite should expose conservative write pages");
            std::vector<uint8_t> writePageObservation = expected;
            t.IsTrue(service->downloadVramPages(
                         writePageObservation, writePages,
                         &operationError),
                     "downloading only sprite write pages should complete");
            t.IsTrue(writePageObservation == expectedAfter,
                     "the command write mask must cover every resident sprite byte");

            std::vector<uint8_t> residentAfter(
                GS_VULKAN_VRAM_SIZE, 0x5Au);
            t.IsTrue(service->downloadVramPages(
                         residentAfter, allPages, &operationError),
                     "a full post-draw observation should complete");
            t.IsTrue(residentAfter == expectedAfter,
                     "resident upload draw and download must match the CPU oracle exactly");

            const std::vector<uint8_t> shortSource(
                GS_VULKAN_VRAM_SIZE - 1u, 0u);
            GsVramPageMask emptyPages;
            t.IsFalse(service->uploadVramPages(
                          shortSource, sparsePages, &operationError),
                      "a short page-upload source should fail closed");
            t.IsFalse(service->uploadVramPages(
                          initial, emptyPages, &operationError),
                      "an empty page upload should fail closed");
            std::vector<uint8_t> shortDestination(
                GS_VULKAN_VRAM_SIZE - 1u, 0x33u);
            const std::vector<uint8_t> shortSentinel = shortDestination;
            t.IsFalse(service->downloadVramPages(
                          shortDestination, sparsePages,
                          &operationError),
                      "a short page-download destination should fail closed");
            t.IsTrue(shortDestination == shortSentinel,
                     "a rejected page download must preserve its destination");
            std::vector<uint8_t> emptyDestination(
                GS_VULKAN_VRAM_SIZE, 0x44u);
            const std::vector<uint8_t> emptySentinel = emptyDestination;
            t.IsFalse(service->downloadVramPages(
                          emptyDestination, emptyPages,
                          &operationError),
                      "an empty page download should fail closed");
            t.IsTrue(emptyDestination == emptySentinel,
                     "an empty page download must preserve its destination");
            GsVulkanCt32Sprite invalidSprite = sprite;
            invalidSprite.reserved = 1u;
            t.IsFalse(service->executeResidentCt32Sprite(
                          invalidSprite, &operationError),
                      "an invalid resident sprite should fail before submission");

            const GsVulkanServiceStatistics statistics =
                service->statistics();
            const uint64_t writePageCount = writePages.count();
            t.Equals(statistics.pageUploadOperationsCompleted, 2ull,
                     "both accepted page uploads should complete once");
            t.Equals(statistics.pageUploadOperationsFailed, 0ull,
                     "caller-side upload rejection should not count as worker failure");
            t.Equals(statistics.pageDownloadOperationsCompleted, 4ull,
                     "all accepted page downloads should complete once");
            t.Equals(statistics.pageDownloadOperationsFailed, 0ull,
                     "caller-side download rejection should not count as worker failure");
            t.Equals(statistics.pagesUploaded,
                     512ull + sparsePages.count(),
                     "page upload statistics should count selected physical pages");
            t.Equals(statistics.pagesDownloaded,
                     512ull + sparsePages.count() +
                         writePageCount + 512ull,
                     "page download statistics should count selected physical pages");
            t.Equals(statistics.pageUploadRegions,
                     1ull + countRuns(sparsePages),
                     "adjacent upload pages should coalesce into copy regions");
            t.Equals(statistics.pageDownloadRegions,
                     countRuns(sparsePages) + 1ull +
                         countRuns(writePages) + 1ull,
                     "adjacent download pages should coalesce into copy regions");
            t.Equals(statistics.bytesUploaded,
                     (512ull + sparsePages.count()) *
                         GS_VRAM_PAGE_SIZE,
                     "page uploads should account only selected bytes");
            t.Equals(statistics.bytesDownloaded,
                     (512ull + sparsePages.count() +
                          writePageCount + 512ull) *
                         GS_VRAM_PAGE_SIZE,
                     "page downloads should account only selected bytes");
            t.Equals(statistics.spriteDrawsCompleted, 1ull,
                     "the resident sprite should complete exactly once");
            t.Equals(statistics.spriteDrawsFailed, 0ull,
                     "caller-side resident rejection should not count as worker failure");
            t.Equals(statistics.queueSubmissions, 7ull,
                     "each accepted transfer or draw should use one bounded submission");
            t.Equals(statistics.shaderDispatches, 1ull,
                     "page transfers should not execute a compute shader");
            t.Equals(statistics.validationErrors, 0u,
                     "resident page operations must remain validation-clean");
            t.Equals(statistics.validationWarnings, 0u,
                     "resident page operations should emit no validation warnings");
            t.IsTrue(service->healthy(),
                     "successful and caller-rejected page work should leave the service healthy");

            service->shutdown();
            service->shutdown();
            std::vector<uint8_t> shutdownDestination(
                GS_VULKAN_VRAM_SIZE, 0x6Bu);
            const std::vector<uint8_t> shutdownSentinel =
                shutdownDestination;
            t.IsFalse(service->downloadVramPages(
                          shutdownDestination, sparsePages,
                          &operationError),
                      "a stopped service must reject page downloads");
            t.IsTrue(shutdownDestination == shutdownSentinel,
                     "post-shutdown page rejection must preserve output");
        });

        tc.Run("Vulkan resident sprite batches use one exact submission", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig config =
                makeRendererServiceConfig(preflight);
            GsVulkanCapabilityReport creationReport{};
            std::string creationError;
            std::unique_ptr<GsVulkanService> service =
                GsVulkanService::create(
                    config, &creationReport, &creationError);
            if (!preflight.ready())
            {
                t.IsNull(service.get(),
                         "an unavailable host should skip resident batching cleanly");
                t.IsFalse(creationReport.ready(),
                          "a skipped batch service should retain its capability result");
                return;
            }
            t.IsNotNull(service.get(),
                        "a suitable device should create the resident batch service");
            t.IsTrue(creationError.empty(),
                     "successful batch service creation should clear its diagnostic");
            if (!service)
                return;

            const std::vector<uint8_t> initial =
                makeVramPattern(0x42415431u);
            GsVramPageMask allPages;
            allPages.setAll();
            std::string operationError;
            t.IsTrue(service->uploadVramPages(
                         initial, allPages, &operationError),
                     "the batch fixture should establish resident VRAM");
            if (!operationError.empty())
                t.Fail("resident batch upload failed: " + operationError);

            const std::array<GsVulkanCt32Sprite, 4> sprites{{
                {5u * 32u, 1u, 1u, 2u, 9u, 8u, 0x10203040u, 0u},
                {41u * 32u, 1u, 3u, 4u, 14u, 13u, 0x50607080u, 0u},
                {197u * 32u, 1u, 8u, 1u, 21u, 12u, 0x90A0B0C0u, 0u},
                {509u * 32u, 1u, 17u, 15u, 31u, 27u, 0xD0E0F001u, 0u},
            }};

            const GsVulkanServiceStatistics beforeRejections =
                service->statistics();
            const auto expectRejected =
                [&](std::span<const GsVulkanCt32Sprite> rejected,
                    const std::string &label)
            {
                operationError.clear();
                t.IsFalse(service->executeResidentCt32Sprites(
                              rejected, &operationError),
                          label + " should fail before worker submission");
                t.IsFalse(operationError.empty(),
                          label + " should retain a diagnostic");
            };
            expectRejected({}, "empty resident batch");
            std::vector<GsVulkanCt32Sprite> oversized(
                GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH + 1u,
                sprites.front());
            expectRejected(oversized, "oversized resident batch");
            std::vector<GsVulkanCt32Sprite> invalid(
                sprites.begin(), sprites.end());
            invalid[2].reserved = 1u;
            expectRejected(invalid, "invalid resident batch member");
            const std::array<GsVulkanCt32Sprite, 2> overlapping{{
                sprites.front(),
                {5u * 32u, 1u, 32u, 1u, 40u, 7u,
                 0x55667788u, 0u},
            }};
            expectRejected(overlapping, "physical-page-overlapping batch");

            const GsVulkanServiceStatistics afterRejections =
                service->statistics();
            t.Equals(afterRejections.queueSubmissions,
                     beforeRejections.queueSubmissions,
                     "caller-rejected batches must not submit GPU work");
            t.Equals(afterRejections.spriteDrawsFailed,
                     beforeRejections.spriteDrawsFailed,
                     "caller-rejected batches must not count worker failures");
            t.Equals(afterRejections.residentSpriteBatchesFailed,
                     beforeRejections.residentSpriteBatchesFailed,
                     "caller-rejected batches must not count failed worker batches");

            std::vector<uint8_t> expected = initial;
            uint64_t expectedPixels = 0u;
            for (const GsVulkanCt32Sprite &sprite : sprites)
            {
                applyCt32SpriteCpu(expected, sprite);
                expectedPixels +=
                    static_cast<uint64_t>(sprite.x1 - sprite.x0) *
                    static_cast<uint64_t>(sprite.y1 - sprite.y0);
            }
            const GsVulkanServiceStatistics beforeBatch =
                service->statistics();
            operationError.clear();
            t.IsTrue(service->executeResidentCt32Sprites(
                         sprites, &operationError),
                     "four disjoint resident sprites should execute as one batch");
            t.IsTrue(operationError.empty(),
                     "successful resident batching should clear its diagnostic");
            const GsVulkanServiceStatistics afterBatch =
                service->statistics();
            t.Equals(afterBatch.queueSubmissions -
                         beforeBatch.queueSubmissions,
                     1ull,
                     "one resident batch should use one Vulkan queue submission");
            t.Equals(afterBatch.shaderDispatches -
                         beforeBatch.shaderDispatches,
                     static_cast<uint64_t>(sprites.size()),
                     "each resident batch member should retain one compute dispatch");
            t.Equals(afterBatch.pipelineBarriers -
                         beforeBatch.pipelineBarriers,
                     2ull,
                     "one resident batch should use one prepare and one completion barrier");
            t.Equals(afterBatch.pipelineBinds -
                         beforeBatch.pipelineBinds,
                     1ull,
                     "one resident batch should bind the fixed CT32 pipeline once");
            t.Equals(afterBatch.pipelineCacheHits -
                         beforeBatch.pipelineCacheHits,
                     1ull,
                     "one resident batch should reuse its prebuilt pipeline once");
            t.Equals(afterBatch.fenceWaits - beforeBatch.fenceWaits,
                     1ull,
                     "one resident batch should wait on one submitted fence");
            t.Equals(afterBatch.spriteDrawsCompleted -
                         beforeBatch.spriteDrawsCompleted,
                     static_cast<uint64_t>(sprites.size()),
                     "batch statistics should count every completed sprite");
            t.Equals(afterBatch.spritePixelsExecuted -
                         beforeBatch.spritePixelsExecuted,
                     expectedPixels,
                     "batch statistics should count every exact covered pixel");
            t.Equals(afterBatch.residentSpriteBatchesCompleted -
                         beforeBatch.residentSpriteBatchesCompleted,
                     1ull,
                     "the worker should complete one resident batch");
            t.Equals(afterBatch.largestResidentSpriteBatch,
                     static_cast<uint64_t>(sprites.size()),
                     "the service should retain its largest accepted batch");

            std::vector<uint8_t> actual(
                GS_VULKAN_VRAM_SIZE, 0xA5u);
            t.IsTrue(service->downloadVramPages(
                         actual, allPages, &operationError),
                     "the completed resident batch should be observable");
            t.IsTrue(actual == expected,
                     "one submitted resident batch must match sequential CPU draws exactly");
            const GsVulkanServiceStatistics finalStatistics =
                service->statistics();
            t.Equals(finalStatistics.validationErrors, 0u,
                     "resident batching must remain validation-clean");
            t.Equals(finalStatistics.validationWarnings, 0u,
                     "resident batching should emit no validation warnings");
            t.IsTrue(service->healthy(),
                     "accepted and caller-rejected batches should leave the service healthy");
            service->shutdown();
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
