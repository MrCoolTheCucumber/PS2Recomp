#include "MiniTest.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_gs_memory.h"
#include "runtime/ps2_gs_vulkan.h"
#include "runtime/ps2_gs_vulkan_backend.h"
#include "runtime/ps2_memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
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

    GsDrawCommand makeSourceCopyAlphaCt32SpriteCommand(
        uint64_t sequence,
        uint32_t framebufferPage,
        uint8_t framebufferWidth,
        GSScissorReg scissor,
        GSXYOffsetReg xyoffset,
        uint16_t rawX0,
        uint16_t rawY0,
        uint16_t rawX1,
        uint16_t rawY1,
        uint32_t rgba,
        uint64_t alpha = 0u,
        bool pabe = false)
    {
        const GsDrawCommand opaque = makeCt32SpriteCommand(
            sequence,
            framebufferPage,
            framebufferWidth,
            scissor,
            xyoffset,
            rawX0,
            rawY0,
            rawX1,
            rawY1,
            rgba);
        GSPrimReg primitive = opaque.primitive();
        primitive.abe = true;
        GSContext context = opaque.context();
        context.alpha = alpha;
        GsDrawGlobalState global = opaque.globalState();
        global.pabe = pabe;
        return buildGsDrawCommand(
            sequence,
            primitive,
            context,
            std::span<const GSVertex>(opaque.vertices()).first(2u),
            global);
    }

    GsDrawCommand makeDepthCt32SpriteCommand(
        uint64_t sequence,
        uint32_t framebufferPage,
        uint8_t framebufferWidth,
        uint32_t depthPage,
        uint8_t depthPsm,
        bool depthMask,
        uint8_t depthTestMethod,
        GSScissorReg scissor,
        GSXYOffsetReg xyoffset,
        uint16_t rawX0,
        uint16_t rawY0,
        uint16_t rawX1,
        uint16_t rawY1,
        uint32_t rgba,
        uint32_t depth)
    {
        const GsDrawCommand flat = makeCt32SpriteCommand(
            sequence,
            framebufferPage,
            framebufferWidth,
            scissor,
            xyoffset,
            rawX0,
            rawY0,
            rawX1,
            rawY1,
            rgba);
        GSContext context = flat.context();
        context.zbuf = {depthPage, depthPsm, depthMask};
        context.test =
            (1ull << 16u) |
            (static_cast<uint64_t>(depthTestMethod & 0x3u) << 17u);
        std::array<GSVertex, 3> vertices = flat.vertices();
        vertices[1].z = static_cast<double>(depth);
        vertices[1].zInteger = depth;
        return buildGsDrawCommand(
            sequence,
            flat.primitive(),
            context,
            std::span<const GSVertex>(vertices).first(2u),
            flat.globalState());
    }

    std::array<GsDrawCommand, 6> makeOrderedDepthCt32SpriteCommands(
        uint64_t sequenceBase)
    {
        return {{
            makeDepthCt32SpriteCommand(
                sequenceBase, 40u, 1u, 200u,
                GS_PSM_Z24, false, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                17u, 17u, 257u, 257u,
                0x10203040u, 0x00ABCDEFu),
            makeDepthCt32SpriteCommand(
                sequenceBase + 1u, 40u, 1u, 200u,
                GS_PSM_Z24, false, 2u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                17u, 17u, 257u, 257u,
                0x50607080u, 0x00BCDEF0u),
            makeDepthCt32SpriteCommand(
                sequenceBase + 2u, 41u, 1u, 201u,
                GS_PSM_Z32, false, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                17u, 17u, 257u, 257u,
                0x90A0B0C0u, 0x7FFFF000u),
            makeDepthCt32SpriteCommand(
                sequenceBase + 3u, 41u, 1u, 201u,
                GS_PSM_Z32, false, 3u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                17u, 17u, 257u, 257u,
                0xD0E0F001u, 0x80000000u),
            makeDepthCt32SpriteCommand(
                sequenceBase + 4u, 201u, 1u, 202u,
                GS_PSM_Z24, false, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                17u, 17u, 257u, 257u,
                0x12345678u, 0x00555555u),
            makeDepthCt32SpriteCommand(
                sequenceBase + 5u, 42u, 1u, 202u,
                GS_PSM_Z24, true, 2u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                17u, 17u, 257u, 257u,
                0x89ABCDEFu, 0x00555555u),
        }};
    }

    GsDrawCommand makeNearestCt32SpriteCommand(
        uint64_t sequence,
        uint32_t framebufferPage,
        uint8_t framebufferWidth,
        uint32_t textureBaseBlock,
        uint8_t textureWidth,
        uint8_t textureWidthLog2,
        uint8_t textureHeightLog2,
        GSScissorReg scissor,
        GSXYOffsetReg xyoffset,
        const std::array<uint16_t, 2> &rawX,
        const std::array<uint16_t, 2> &rawY,
        const std::array<uint16_t, 2> &rawU,
        const std::array<uint16_t, 2> &rawV,
        uint8_t textureWrapModeU = 0u,
        uint8_t textureWrapModeV = 0u,
        uint16_t textureRegionMinU = 0x155u,
        uint16_t textureRegionMaxU = 0x2AAu,
        uint16_t textureRegionMinV = 0x133u,
        uint16_t textureRegionMaxV = 0x266u)
    {
        GSContext context{};
        context.frame = {
            framebufferPage, framebufferWidth,
            GS_PSM_CT32, 0u};
        context.scissor = scissor;
        context.xyoffset = xyoffset;
        context.zbuf = {0u, GS_PSM_Z32, true};
        context.test = 0x30000u; // ZTE=1, ZTST=ALWAYS, ZMSK=1.
        context.tex0.tbp0 = textureBaseBlock;
        context.tex0.tbw = textureWidth;
        context.tex0.psm = GS_PSM_CT32;
        context.tex0.tw = textureWidthLog2;
        context.tex0.th = textureHeightLog2;
        context.tex0.tcc = 1u;
        context.tex0.tfx = 1u;
        // LCM, MTBA, L, and K are semantically irrelevant when MXL, MMAG,
        // and MMIN are zero. Keep them non-zero to make that contract visible.
        context.tex1 = 1ull | (1ull << 9u) |
                       (3ull << 19u) | (0x321ull << 32u);
        context.miptbp1 = 0x0123456789ABCDEFull;
        context.miptbp2 = 0x0FEDCBA987654321ull;
        context.clamp =
            (static_cast<uint64_t>(textureRegionMinU) << 4u) |
            (static_cast<uint64_t>(textureRegionMaxU) << 14u) |
            (static_cast<uint64_t>(textureRegionMinV) << 24u) |
            (static_cast<uint64_t>(textureRegionMaxV) << 34u) |
            (textureWrapModeU & 0x3u) |
            ((textureWrapModeV & 0x3u) << 2u);

        GSPrimReg primitive{};
        primitive.type = GS_PRIM_SPRITE;
        primitive.tme = true;
        primitive.fst = true;
        std::array<GSVertex, 2> vertices{};
        for (size_t index = 0u; index < vertices.size(); ++index)
        {
            vertices[index].x12_4 = rawX[index];
            vertices[index].y12_4 = rawY[index];
            vertices[index].u = rawU[index];
            vertices[index].v = rawV[index];
            vertices[index].r = static_cast<uint8_t>(0x10u + index);
            vertices[index].g = static_cast<uint8_t>(0x20u + index);
            vertices[index].b = static_cast<uint8_t>(0x30u + index);
            vertices[index].a = static_cast<uint8_t>(0x40u + index);
        }
        return buildGsDrawCommand(
            sequence, primitive, context,
            std::span<const GSVertex>(vertices),
            GsDrawGlobalState{});
    }

    GsDrawCommand makeLinearCt32SpriteCommand(
        uint64_t sequence,
        uint32_t framebufferPage,
        uint8_t framebufferWidth,
        uint32_t textureBaseBlock,
        uint8_t textureWidth,
        uint8_t textureWidthLog2,
        uint8_t textureHeightLog2,
        GSScissorReg scissor,
        GSXYOffsetReg xyoffset,
        const std::array<uint16_t, 2> &rawX,
        const std::array<uint16_t, 2> &rawY,
        const std::array<uint16_t, 2> &rawU,
        const std::array<uint16_t, 2> &rawV,
        uint8_t textureWrapModeU,
        uint8_t textureWrapModeV)
    {
        const GsDrawCommand nearest = makeNearestCt32SpriteCommand(
            sequence,
            framebufferPage,
            framebufferWidth,
            textureBaseBlock,
            textureWidth,
            textureWidthLog2,
            textureHeightLog2,
            scissor,
            xyoffset,
            rawX,
            rawY,
            rawU,
            rawV,
            textureWrapModeU,
            textureWrapModeV);
        GSContext context = nearest.context();
        context.tex1 &=
            ~((0x7ull << 2u) | (1ull << 5u) | (0x7ull << 6u));
        context.tex1 |= (1ull << 5u) | (1ull << 6u);
        return buildGsDrawCommand(
            sequence,
            nearest.primitive(),
            context,
            std::span<const GSVertex>(nearest.vertices()).first(2u),
            nearest.globalState());
    }

    GsDrawCommand makeLinearCt32RepeatSpriteCommand(
        uint64_t sequence,
        uint32_t framebufferPage,
        uint8_t framebufferWidth,
        uint32_t textureBaseBlock,
        uint8_t textureWidth,
        uint8_t textureWidthLog2,
        uint8_t textureHeightLog2,
        GSScissorReg scissor,
        GSXYOffsetReg xyoffset,
        const std::array<uint16_t, 2> &rawX,
        const std::array<uint16_t, 2> &rawY,
        const std::array<uint16_t, 2> &rawU,
        const std::array<uint16_t, 2> &rawV)
    {
        return makeLinearCt32SpriteCommand(
            sequence,
            framebufferPage,
            framebufferWidth,
            textureBaseBlock,
            textureWidth,
            textureWidthLog2,
            textureHeightLog2,
            scissor,
            xyoffset,
            rawX,
            rawY,
            rawU,
            rawV,
            0u,
            0u);
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

    void applyDepthCt32SpriteCpu(
        std::vector<uint8_t> &vram,
        const GsVulkanDepthCt32Sprite &sprite)
    {
        const auto readDepth = [&](uint32_t x, uint32_t y)
        {
            return sprite.depthPsm == GS_PSM_Z24
                ? GSMem::ReadZ24(
                      vram.data(), sprite.depthBaseBlock,
                      sprite.framebufferWidth, x, y)
                : GSMem::ReadZ32(
                      vram.data(), sprite.depthBaseBlock,
                      sprite.framebufferWidth, x, y);
        };
        const auto writeDepth = [&](uint32_t x, uint32_t y)
        {
            if (sprite.depthPsm == GS_PSM_Z24)
            {
                GSMem::WriteZ24(
                    vram.data(), sprite.depthBaseBlock,
                    sprite.framebufferWidth, x, y, sprite.depth);
            }
            else
            {
                GSMem::WriteZ32(
                    vram.data(), sprite.depthBaseBlock,
                    sprite.framebufferWidth, x, y, sprite.depth);
            }
        };

        for (uint32_t y = sprite.boundsY0; y < sprite.boundsY1; ++y)
        {
            for (uint32_t x = sprite.boundsX0; x < sprite.boundsX1; ++x)
            {
                const uint32_t currentDepth =
                    sprite.depthTestMethod >= 2u ? readDepth(x, y) : 0u;
                const bool passes =
                    sprite.depthTestMethod == 1u ||
                    (sprite.depthTestMethod == 2u &&
                     sprite.depth >= currentDepth) ||
                    (sprite.depthTestMethod == 3u &&
                     sprite.depth > currentDepth);
                if (!passes)
                    continue;
                GSMem::WriteCT32(
                    vram.data(),
                    sprite.framebufferBaseBlock,
                    sprite.framebufferWidth,
                    x, y, sprite.rgba);
                if (sprite.depthWrite != 0u)
                    writeDepth(x, y);
            }
        }
    }

    void applyNearestCt32SpriteCpu(
        std::vector<uint8_t> &vram,
        const GsVulkanNearestCt32Sprite &sprite)
    {
        const auto wrapCoordinate = [](
            int32_t coordinate,
            uint32_t mask,
            uint32_t wrap)
        {
            const uint8_t mode = gsVulkanTextureWrapMode(wrap);
            if (mode == 0u)
                return static_cast<uint32_t>(coordinate) & mask;
            if (mode == 1u)
            {
                return static_cast<uint32_t>(std::clamp(
                    coordinate, 0, static_cast<int32_t>(mask)));
            }
            if (mode == 3u)
            {
                const uint32_t regionMask =
                    gsVulkanTextureRegionMin(wrap) & mask;
                return
                    (static_cast<uint32_t>(coordinate) & regionMask) |
                    gsVulkanTextureRegionMax(wrap);
            }
            const int32_t minimum =
                gsVulkanTextureRegionMin(wrap);
            const int32_t maximum =
                gsVulkanTextureRegionMax(wrap);
            if (coordinate < minimum)
                return static_cast<uint32_t>(minimum);
            if (coordinate > maximum)
                return static_cast<uint32_t>(maximum);
            return static_cast<uint32_t>(coordinate);
        };
        for (uint32_t y = sprite.boundsY0;
             y < sprite.boundsY1; ++y)
        {
            const int32_t textureV =
                sprite.textureOriginV +
                static_cast<int32_t>(y - sprite.boundsY0) *
                    sprite.textureStepV;
            for (uint32_t x = sprite.boundsX0;
                 x < sprite.boundsX1; ++x)
            {
                const int32_t textureU =
                    sprite.textureOriginU +
                    static_cast<int32_t>(x - sprite.boundsX0) *
                        sprite.textureStepU;
                const uint32_t texel = GSMem::ReadCT32(
                    vram.data(),
                    sprite.textureBaseBlock,
                    sprite.textureWidth,
                    wrapCoordinate(
                        textureU, sprite.textureMaskU,
                        sprite.textureWrapU),
                    wrapCoordinate(
                        textureV, sprite.textureMaskV,
                        sprite.textureWrapV));
                GSMem::WriteCT32(
                    vram.data(),
                    sprite.framebufferBaseBlock,
                    sprite.framebufferWidth,
                    x, y, texel);
            }
        }
    }

    int floorLinearFixed16_16(int32_t value)
    {
        int quotient = value / 65536;
        if (value < 0 && (value % 65536) != 0)
            --quotient;
        return quotient;
    }

    int linearShiftRight4(int value)
    {
        return value >= 0 ? value / 16 : -((-value + 15) / 16);
    }

    uint32_t linearColor4ForRecord(
        uint32_t from,
        uint32_t to,
        uint8_t weight)
    {
        uint32_t result = 0u;
        for (uint32_t shift = 0u; shift < 32u; shift += 8u)
        {
            const int fromChannel =
                static_cast<int>((from >> shift) & 0xFFu);
            const int toChannel =
                static_cast<int>((to >> shift) & 0xFFu);
            const int channel = std::clamp(
                fromChannel + linearShiftRight4(
                    (toChannel - fromChannel) * weight),
                0,
                255);
            result |= static_cast<uint32_t>(channel) << shift;
        }
        return result;
    }

    uint32_t bilinearColor4ForRecord(
        uint32_t c00,
        uint32_t c10,
        uint32_t c01,
        uint32_t c11,
        uint8_t weightU,
        uint8_t weightV)
    {
        return linearColor4ForRecord(
            linearColor4ForRecord(c00, c10, weightU),
            linearColor4ForRecord(c01, c11, weightU),
            weightV);
    }

    void applyLinearCt32SpriteCpu(
        std::vector<uint8_t> &vram,
        const GsVulkanLinearCt32Sprite &sprite)
    {
        const auto wrapCoordinate = [](
            int32_t coordinate,
            uint32_t mask,
            uint32_t wrap)
        {
            const uint8_t mode = gsVulkanTextureWrapMode(wrap);
            if (mode == 0u)
                return static_cast<uint32_t>(coordinate) & mask;
            if (mode == 1u)
            {
                return static_cast<uint32_t>(std::clamp(
                    coordinate, 0, static_cast<int32_t>(mask)));
            }
            if (mode == 3u)
            {
                const uint32_t regionMask =
                    gsVulkanTextureRegionMin(wrap) & mask;
                return
                    (static_cast<uint32_t>(coordinate) & regionMask) |
                    gsVulkanTextureRegionMax(wrap);
            }
            const int32_t minimum =
                gsVulkanTextureRegionMin(wrap);
            const int32_t maximum =
                gsVulkanTextureRegionMax(wrap);
            if (coordinate < minimum)
                return static_cast<uint32_t>(minimum);
            if (coordinate > maximum)
                return static_cast<uint32_t>(maximum);
            return static_cast<uint32_t>(coordinate);
        };
        const int alignedDrawX =
            static_cast<int>(sprite.boundsX0) & ~7;
        float fixedScanV =
            std::bit_cast<float>(sprite.fixedScanVBits);
        const float fixedStepV =
            std::bit_cast<float>(sprite.fixedStepVBits);
        for (uint32_t y = sprite.boundsY0;
             y < sprite.boundsY1;
             ++y)
        {
            const int32_t fixedV = static_cast<int32_t>(fixedScanV);
            const uint8_t weightV = static_cast<uint8_t>(
                (static_cast<uint32_t>(fixedV) >> 12u) & 0xFu);
            const int32_t unwrappedV0 = floorLinearFixed16_16(fixedV);
            const uint32_t v0 = wrapCoordinate(
                unwrappedV0,
                sprite.textureMaskV,
                sprite.textureWrapV);
            const uint32_t v1 = weightV == 0u
                ? v0
                : wrapCoordinate(
                      unwrappedV0 + 1,
                      sprite.textureMaskV,
                      sprite.textureWrapV);

            for (uint32_t x = sprite.boundsX0;
                 x < sprite.boundsX1;
                 ++x)
            {
                const int block =
                    (static_cast<int>(x) - alignedDrawX) / 8;
                const int32_t fixedU = static_cast<int32_t>(
                    static_cast<uint32_t>(sprite.fixedBaseU) +
                    static_cast<uint32_t>(sprite.fixedLaneU[x & 7u]) +
                    static_cast<uint32_t>(sprite.fixedBlockStepU) *
                        static_cast<uint32_t>(block));
                const uint8_t weightU = static_cast<uint8_t>(
                    (static_cast<uint32_t>(fixedU) >> 12u) & 0xFu);
                const int32_t unwrappedU0 =
                    floorLinearFixed16_16(fixedU);
                const uint32_t u0 = wrapCoordinate(
                    unwrappedU0,
                    sprite.textureMaskU,
                    sprite.textureWrapU);
                const uint32_t u1 = weightU == 0u
                    ? u0
                    : wrapCoordinate(
                          unwrappedU0 + 1,
                          sprite.textureMaskU,
                          sprite.textureWrapU);

                const uint32_t c00 = GSMem::ReadCT32(
                    vram.data(),
                    sprite.textureBaseBlock,
                    sprite.textureWidth,
                    u0,
                    v0);
                uint32_t color = c00;
                if (weightU == 0u)
                {
                    if (weightV != 0u)
                    {
                        color = linearColor4ForRecord(
                            c00,
                            GSMem::ReadCT32(
                                vram.data(),
                                sprite.textureBaseBlock,
                                sprite.textureWidth,
                                u0,
                                v1),
                            weightV);
                    }
                }
                else if (weightV == 0u)
                {
                    color = linearColor4ForRecord(
                        c00,
                        GSMem::ReadCT32(
                            vram.data(),
                            sprite.textureBaseBlock,
                            sprite.textureWidth,
                            u1,
                            v0),
                        weightU);
                }
                else
                {
                    color = bilinearColor4ForRecord(
                        c00,
                        GSMem::ReadCT32(
                            vram.data(),
                            sprite.textureBaseBlock,
                            sprite.textureWidth,
                            u1,
                            v0),
                        GSMem::ReadCT32(
                            vram.data(),
                            sprite.textureBaseBlock,
                            sprite.textureWidth,
                            u0,
                            v1),
                        GSMem::ReadCT32(
                            vram.data(),
                            sprite.textureBaseBlock,
                            sprite.textureWidth,
                            u1,
                            v1),
                        weightU,
                        weightV);
                }
                GSMem::WriteCT32(
                    vram.data(),
                    sprite.framebufferBaseBlock,
                    sprite.framebufferWidth,
                    x,
                    y,
                    color);
            }
            fixedScanV += fixedStepV;
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
        uint32_t framebufferWidth = 2u,
        uint32_t scissorX1 = 127u,
        uint32_t scissorY1 = 63u)
    {
        const uint64_t frame =
            static_cast<uint64_t>(framebufferPage) |
            (static_cast<uint64_t>(framebufferWidth) << 16u) |
            (static_cast<uint64_t>(GS_PSM_CT32) << 24u);
        const uint64_t scissor =
            (static_cast<uint64_t>(scissorX1) << 16u) |
            (static_cast<uint64_t>(scissorY1) << 48u);
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

    void drawFlatCt32Triangle(
        GS &gs,
        const std::array<uint16_t, 3> &x,
        const std::array<uint16_t, 3> &y,
        uint32_t rgba)
    {
        gs.writeRegister(
            GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_TRIANGLE));
        gs.writeRegister(GS_REG_RGBAQ, rgba);
        for (size_t index = 0u; index < x.size(); ++index)
            gs.writeRegister(GS_REG_XYZ2, packXyz2(x[index], y[index]));
    }

    void drawFlatCt32TriangleCommand(
        GS &gs,
        const GsDrawCommand &command,
        uint32_t rgba)
    {
        const GSContext &context = command.context();
        const GsDrawGlobalState &global = command.globalState();
        const uint64_t frame =
            static_cast<uint64_t>(context.frame.fbp) |
            (static_cast<uint64_t>(context.frame.fbw) << 16u) |
            (static_cast<uint64_t>(context.frame.psm) << 24u) |
            (static_cast<uint64_t>(context.frame.fbmsk) << 32u);
        const uint64_t zbuf =
            static_cast<uint64_t>(context.zbuf.zbp) |
            (static_cast<uint64_t>(context.zbuf.psm) << 24u) |
            (static_cast<uint64_t>(context.zbuf.zmask) << 32u);
        const uint64_t scissor =
            static_cast<uint64_t>(context.scissor.x0) |
            (static_cast<uint64_t>(context.scissor.x1) << 16u) |
            (static_cast<uint64_t>(context.scissor.y0) << 32u) |
            (static_cast<uint64_t>(context.scissor.y1) << 48u);
        const uint64_t xyoffset =
            static_cast<uint64_t>(context.xyoffset.ofx) |
            (static_cast<uint64_t>(context.xyoffset.ofy) << 32u);
        gs.writeRegister(GS_REG_FRAME_1, frame);
        gs.writeRegister(GS_REG_ZBUF_1, zbuf);
        gs.writeRegister(GS_REG_SCISSOR_1, scissor);
        gs.writeRegister(GS_REG_XYOFFSET_1, xyoffset);
        gs.writeRegister(GS_REG_TEST_1, context.test);
        gs.writeRegister(GS_REG_FBA_1, context.fba);
        gs.writeRegister(GS_REG_SCANMSK, global.scanMask);
        gs.writeRegister(GS_REG_DTHE, global.dither ? 1u : 0u);

        std::array<uint16_t, 3> rawX{};
        std::array<uint16_t, 3> rawY{};
        for (size_t index = 0u; index < rawX.size(); ++index)
        {
            rawX[index] = command.vertices()[index].x12_4;
            rawY[index] = command.vertices()[index].y12_4;
        }
        drawFlatCt32Triangle(gs, rawX, rawY, rgba);
    }

    void configureNearestCt32SpriteCommand(
        GS &gs,
        const GsDrawCommand &command)
    {
        const GSContext &context = command.context();
        const GsDrawGlobalState &global = command.globalState();
        const GSTex0Reg &texture = context.tex0;
        const uint64_t frame =
            static_cast<uint64_t>(context.frame.fbp) |
            (static_cast<uint64_t>(context.frame.fbw) << 16u) |
            (static_cast<uint64_t>(context.frame.psm) << 24u) |
            (static_cast<uint64_t>(context.frame.fbmsk) << 32u);
        const uint64_t zbuf =
            static_cast<uint64_t>(context.zbuf.zbp) |
            (static_cast<uint64_t>(context.zbuf.psm) << 24u) |
            (static_cast<uint64_t>(context.zbuf.zmask) << 32u);
        const uint64_t scissor =
            static_cast<uint64_t>(context.scissor.x0) |
            (static_cast<uint64_t>(context.scissor.x1) << 16u) |
            (static_cast<uint64_t>(context.scissor.y0) << 32u) |
            (static_cast<uint64_t>(context.scissor.y1) << 48u);
        const uint64_t xyoffset =
            static_cast<uint64_t>(context.xyoffset.ofx) |
            (static_cast<uint64_t>(context.xyoffset.ofy) << 32u);
        const uint64_t tex0 =
            static_cast<uint64_t>(texture.tbp0) |
            (static_cast<uint64_t>(texture.tbw) << 14u) |
            (static_cast<uint64_t>(texture.psm) << 20u) |
            (static_cast<uint64_t>(texture.tw) << 26u) |
            (static_cast<uint64_t>(texture.th) << 30u) |
            (static_cast<uint64_t>(texture.tcc) << 34u) |
            (static_cast<uint64_t>(texture.tfx) << 35u) |
            (static_cast<uint64_t>(texture.cbp) << 37u) |
            (static_cast<uint64_t>(texture.cpsm) << 51u) |
            (static_cast<uint64_t>(texture.csm) << 55u) |
            (static_cast<uint64_t>(texture.csa) << 56u) |
            (static_cast<uint64_t>(texture.cld) << 61u);
        const GSPrimReg &primitive = command.primitive();
        const uint64_t prim =
            static_cast<uint64_t>(primitive.type) |
            (static_cast<uint64_t>(primitive.iip) << 3u) |
            (static_cast<uint64_t>(primitive.tme) << 4u) |
            (static_cast<uint64_t>(primitive.fge) << 5u) |
            (static_cast<uint64_t>(primitive.abe) << 6u) |
            (static_cast<uint64_t>(primitive.aa1) << 7u) |
            (static_cast<uint64_t>(primitive.fst) << 8u) |
            (static_cast<uint64_t>(primitive.ctxt) << 9u) |
            (static_cast<uint64_t>(primitive.fix) << 10u);

        gs.writeRegister(GS_REG_FRAME_1, frame);
        gs.writeRegister(GS_REG_ZBUF_1, zbuf);
        gs.writeRegister(GS_REG_SCISSOR_1, scissor);
        gs.writeRegister(GS_REG_XYOFFSET_1, xyoffset);
        gs.writeRegister(GS_REG_TEST_1, context.test);
        gs.writeRegister(GS_REG_FBA_1, context.fba);
        gs.writeRegister(GS_REG_SCANMSK, global.scanMask);
        gs.writeRegister(GS_REG_DTHE, global.dither ? 1u : 0u);
        gs.writeRegister(GS_REG_TEX0_1, tex0);
        gs.writeRegister(GS_REG_TEX1_1, context.tex1);
        gs.writeRegister(GS_REG_MIPTBP1_1, context.miptbp1);
        gs.writeRegister(GS_REG_MIPTBP2_1, context.miptbp2);
        gs.writeRegister(GS_REG_CLAMP_1, context.clamp);
        gs.writeRegister(GS_REG_PRIM, prim);
    }

    void writeNearestCt32SpriteVertex(
        GS &gs,
        const GSVertex &vertex)
    {
        const uint32_t rgba =
            static_cast<uint32_t>(vertex.r) |
            (static_cast<uint32_t>(vertex.g) << 8u) |
            (static_cast<uint32_t>(vertex.b) << 16u) |
            (static_cast<uint32_t>(vertex.a) << 24u);
        const uint64_t uv =
            static_cast<uint64_t>(vertex.u) |
            (static_cast<uint64_t>(vertex.v) << 16u);
        gs.writeRegister(GS_REG_RGBAQ, rgba);
        gs.writeRegister(GS_REG_UV, uv);
        gs.writeRegister(
            GS_REG_XYZ2,
            packXyz2(vertex.x12_4, vertex.y12_4,
                     vertex.zInteger));
    }

    void drawNearestCt32SpriteCommand(
        GS &gs,
        const GsDrawCommand &command)
    {
        configureNearestCt32SpriteCommand(gs, command);
        for (size_t index = 0u; index < command.vertexCount(); ++index)
        {
            writeNearestCt32SpriteVertex(
                gs, command.vertices()[index]);
        }
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
            FailResidentDraw,
            InvalidOutput,
        };

        explicit FakeCt32Executor(
            Behavior behavior_,
            bool exactCt32Triangle_ = true,
            bool exactNearestCt32Sprite_ = true,
            bool exactLinearCt32Sprite_ = true,
            bool exactDepthCt32Sprite_ = true)
            : behavior(behavior_)
        {
            report.compiled = true;
            report.status = GsVulkanProbeStatus::Ready;
            report.selectedDeviceIndex = 0;
            report.devices.push_back({});
            report.devices[0].name = "deterministic fake executor";
            report.devices[0].suitable = true;
            report.devices[0].shaderInt64 = exactCt32Triangle_;
            report.devices[0].exactCt32Triangle = exactCt32Triangle_;
            report.devices[0].exactNearestCt32Sprite =
                exactNearestCt32Sprite_;
            report.devices[0].exactLinearCt32Sprite =
                exactLinearCt32Sprite_;
            report.devices[0].exactDepthCt32Sprite =
                exactDepthCt32Sprite_;
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

            residentVram.assign(input.begin(), input.end());
            if (behavior == Behavior::Exact)
                applyCt32SpriteCpu(residentVram, sprite);
            output = residentVram;
            if (behavior == Behavior::InvalidOutput)
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

        bool executeCt32Triangle(
            std::span<const uint8_t> input,
            const GsVulkanCt32Triangle &triangle,
            std::vector<uint8_t> &output,
            std::string *error) override
        {
            if (!isHealthy || behavior == Behavior::Fail)
            {
                ++serviceStatistics.triangleDrawsFailed;
                if (error)
                    *error = "injected CT32 triangle executor failure";
                return false;
            }

            residentVram.assign(input.begin(), input.end());
            if (behavior == Behavior::Exact)
                applyCt32TriangleCpu(residentVram, triangle);
            output = residentVram;
            if (behavior == Behavior::InvalidOutput)
                output.resize(1u);
            ++serviceStatistics.triangleDrawsCompleted;
            serviceStatistics.triangleCandidatePixelsExecuted +=
                static_cast<uint64_t>(
                    triangle.boundsX1 - triangle.boundsX0) *
                static_cast<uint64_t>(
                    triangle.boundsY1 - triangle.boundsY0);
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

        bool executeDepthCt32Sprite(
            std::span<const uint8_t> input,
            const GsVulkanDepthCt32Sprite &sprite,
            std::vector<uint8_t> &output,
            std::string *error) override
        {
            if (!isHealthy || behavior == Behavior::Fail ||
                !report.devices[0].exactDepthCt32Sprite)
            {
                ++serviceStatistics.depthCt32SpriteDrawsFailed;
                if (error)
                    *error = "injected depth CT32 executor failure";
                return false;
            }

            residentVram.assign(input.begin(), input.end());
            if (behavior == Behavior::Exact)
                applyDepthCt32SpriteCpu(residentVram, sprite);
            output = residentVram;
            if (behavior == Behavior::InvalidOutput)
                output.resize(1u);
            ++serviceStatistics.depthCt32SpriteDrawsCompleted;
            serviceStatistics.depthCt32SpritePixelsExecuted +=
                static_cast<uint64_t>(
                    sprite.boundsX1 - sprite.boundsX0) *
                static_cast<uint64_t>(
                    sprite.boundsY1 - sprite.boundsY0);
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

        bool executeNearestCt32Sprite(
            std::span<const uint8_t> input,
            const GsVulkanNearestCt32Sprite &sprite,
            std::vector<uint8_t> &output,
            std::string *error) override
        {
            if (!isHealthy || behavior == Behavior::Fail ||
                !report.devices[0].exactNearestCt32Sprite)
            {
                ++serviceStatistics.nearestCt32SpriteDrawsFailed;
                if (error)
                    *error = "injected nearest CT32 executor failure";
                return false;
            }

            residentVram.assign(input.begin(), input.end());
            if (behavior == Behavior::Exact)
                applyNearestCt32SpriteCpu(residentVram, sprite);
            output = residentVram;
            if (behavior == Behavior::InvalidOutput)
                output.resize(1u);
            ++serviceStatistics.nearestCt32SpriteDrawsCompleted;
            serviceStatistics.nearestCt32SpritePixelsExecuted +=
                static_cast<uint64_t>(
                    sprite.boundsX1 - sprite.boundsX0) *
                static_cast<uint64_t>(
                    sprite.boundsY1 - sprite.boundsY0);
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

        bool executeLinearCt32Sprite(
            std::span<const uint8_t> input,
            const GsVulkanLinearCt32Sprite &sprite,
            std::vector<uint8_t> &output,
            std::string *error) override
        {
            if (!isHealthy || behavior == Behavior::Fail ||
                !report.devices[0].exactLinearCt32Sprite)
            {
                ++serviceStatistics.linearCt32SpriteDrawsFailed;
                if (error)
                    *error = "injected linear CT32 executor failure";
                return false;
            }

            residentVram.assign(input.begin(), input.end());
            if (behavior == Behavior::Exact)
                applyLinearCt32SpriteCpu(residentVram, sprite);
            output = residentVram;
            if (behavior == Behavior::InvalidOutput)
                output.resize(1u);
            ++serviceStatistics.linearCt32SpriteDrawsCompleted;
            serviceStatistics.linearCt32SpritePixelsExecuted +=
                static_cast<uint64_t>(
                    sprite.boundsX1 - sprite.boundsX0) *
                static_cast<uint64_t>(
                    sprite.boundsY1 - sprite.boundsY0);
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
                behavior == Behavior::FailResidentDraw ||
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

        bool executeResidentNearestCt32Sprite(
            const GsVulkanNearestCt32Sprite &sprite,
            std::string *error) override
        {
            return executeResidentNearestCt32Sprites(
                std::span<const GsVulkanNearestCt32Sprite>(
                    &sprite, 1u),
                error);
        }

        bool executeResidentNearestCt32Sprites(
            std::span<const GsVulkanNearestCt32Sprite> sprites,
            std::string *error) override
        {
            if (!isHealthy || behavior == Behavior::Fail ||
                behavior == Behavior::FailResidentDraw ||
                behavior == Behavior::InvalidOutput ||
                sprites.empty() ||
                sprites.size() >
                    GS_VULKAN_MAX_RESIDENT_NEAREST_CT32_BATCH ||
                !report.devices[0].exactNearestCt32Sprite)
            {
                serviceStatistics.nearestCt32SpriteDrawsFailed +=
                    sprites.size();
                ++serviceStatistics
                      .residentNearestCt32SpriteBatchesFailed;
                if (error)
                {
                    *error =
                        "injected resident nearest CT32 executor failure";
                }
                return false;
            }
            for (const GsVulkanNearestCt32Sprite &sprite : sprites)
            {
                if (behavior == Behavior::Exact)
                    applyNearestCt32SpriteCpu(residentVram, sprite);
                serviceStatistics.nearestCt32SpritePixelsExecuted +=
                    static_cast<uint64_t>(
                        sprite.boundsX1 - sprite.boundsX0) *
                    static_cast<uint64_t>(
                        sprite.boundsY1 - sprite.boundsY0);
            }
            serviceStatistics.nearestCt32SpriteDrawsCompleted +=
                sprites.size();
            ++serviceStatistics
                  .residentNearestCt32SpriteBatchesCompleted;
            serviceStatistics.largestResidentNearestCt32SpriteBatch =
                std::max(
                    serviceStatistics
                        .largestResidentNearestCt32SpriteBatch,
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

        bool executeResidentDepthCt32Sprite(
            const GsVulkanDepthCt32Sprite &sprite,
            std::string *error) override
        {
            return executeResidentDepthCt32Sprites(
                std::span<const GsVulkanDepthCt32Sprite>(
                    &sprite, 1u),
                error);
        }

        bool executeResidentDepthCt32Sprites(
            std::span<const GsVulkanDepthCt32Sprite> sprites,
            std::string *error) override
        {
            if (!isHealthy || behavior == Behavior::Fail ||
                behavior == Behavior::FailResidentDraw ||
                behavior == Behavior::InvalidOutput ||
                sprites.empty() ||
                sprites.size() >
                    GS_VULKAN_MAX_RESIDENT_DEPTH_CT32_BATCH ||
                !report.devices[0].exactDepthCt32Sprite)
            {
                serviceStatistics.depthCt32SpriteDrawsFailed +=
                    sprites.size();
                ++serviceStatistics
                      .residentDepthCt32SpriteBatchesFailed;
                if (error)
                {
                    *error =
                        "injected resident depth CT32 executor failure";
                }
                return false;
            }
            for (const GsVulkanDepthCt32Sprite &sprite : sprites)
            {
                if (behavior == Behavior::Exact)
                    applyDepthCt32SpriteCpu(residentVram, sprite);
                serviceStatistics.depthCt32SpritePixelsExecuted +=
                    static_cast<uint64_t>(
                        sprite.boundsX1 - sprite.boundsX0) *
                    static_cast<uint64_t>(
                        sprite.boundsY1 - sprite.boundsY0);
            }
            serviceStatistics.depthCt32SpriteDrawsCompleted +=
                sprites.size();
            ++serviceStatistics
                  .residentDepthCt32SpriteBatchesCompleted;
            serviceStatistics.largestResidentDepthCt32SpriteBatch =
                std::max(
                    serviceStatistics
                        .largestResidentDepthCt32SpriteBatch,
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

        bool executeResidentLinearCt32Sprite(
            const GsVulkanLinearCt32Sprite &sprite,
            std::string *error) override
        {
            return executeResidentLinearCt32Sprites(
                std::span<const GsVulkanLinearCt32Sprite>(
                    &sprite, 1u),
                error);
        }

        bool executeResidentLinearCt32Sprites(
            std::span<const GsVulkanLinearCt32Sprite> sprites,
            std::string *error) override
        {
            if (!isHealthy || behavior == Behavior::Fail ||
                behavior == Behavior::FailResidentDraw ||
                behavior == Behavior::InvalidOutput ||
                sprites.empty() ||
                sprites.size() >
                    GS_VULKAN_MAX_RESIDENT_LINEAR_CT32_BATCH ||
                !report.devices[0].exactLinearCt32Sprite)
            {
                serviceStatistics.linearCt32SpriteDrawsFailed +=
                    sprites.size();
                ++serviceStatistics
                      .residentLinearCt32SpriteBatchesFailed;
                if (error)
                {
                    *error =
                        "injected resident linear CT32 executor failure";
                }
                return false;
            }
            for (const GsVulkanLinearCt32Sprite &sprite : sprites)
            {
                if (behavior == Behavior::Exact)
                    applyLinearCt32SpriteCpu(residentVram, sprite);
                serviceStatistics.linearCt32SpritePixelsExecuted +=
                    static_cast<uint64_t>(
                        sprite.boundsX1 - sprite.boundsX0) *
                    static_cast<uint64_t>(
                        sprite.boundsY1 - sprite.boundsY0);
            }
            serviceStatistics.linearCt32SpriteDrawsCompleted +=
                sprites.size();
            ++serviceStatistics
                  .residentLinearCt32SpriteBatchesCompleted;
            serviceStatistics.largestResidentLinearCt32SpriteBatch =
                std::max(
                    serviceStatistics
                        .largestResidentLinearCt32SpriteBatch,
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

        bool executeResidentCt32Triangle(
            const GsVulkanCt32Triangle &triangle,
            std::string *error) override
        {
            return executeResidentCt32Triangles(
                std::span<const GsVulkanCt32Triangle>(&triangle, 1u),
                error);
        }

        bool executeResidentCt32Triangles(
            std::span<const GsVulkanCt32Triangle> triangles,
            std::string *error) override
        {
            if (!isHealthy || behavior == Behavior::Fail ||
                behavior == Behavior::FailResidentDraw ||
                behavior == Behavior::InvalidOutput ||
                triangles.empty() ||
                triangles.size() >
                    GS_VULKAN_MAX_RESIDENT_TRIANGLE_BATCH ||
                !report.devices[0].exactCt32Triangle)
            {
                serviceStatistics.triangleDrawsFailed += triangles.size();
                ++serviceStatistics.residentTriangleBatchesFailed;
                if (error)
                    *error = "injected resident CT32 triangle executor failure";
                return false;
            }
            for (const GsVulkanCt32Triangle &triangle : triangles)
            {
                if (behavior == Behavior::Exact)
                    applyCt32TriangleCpu(residentVram, triangle);
                serviceStatistics.triangleCandidatePixelsExecuted +=
                    static_cast<uint64_t>(
                        triangle.boundsX1 - triangle.boundsX0) *
                    static_cast<uint64_t>(
                        triangle.boundsY1 - triangle.boundsY0);
            }
            serviceStatistics.triangleDrawsCompleted += triangles.size();
            ++serviceStatistics.residentTriangleBatchesCompleted;
            serviceStatistics.largestResidentTriangleBatch = std::max(
                serviceStatistics.largestResidentTriangleBatch,
                static_cast<uint64_t>(triangles.size()));
            ++serviceStatistics.queueSubmissions;
            serviceStatistics.shaderDispatches += triangles.size();
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
            t.Equals(GS_VULKAN_MAX_RESIDENT_NEAREST_CT32_BATCH,
                     GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH,
                     "resident texture batches should share the bounded sprite limit");
            t.Equals(GS_VULKAN_MAX_RESIDENT_LINEAR_CT32_BATCH,
                     GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH,
                     "resident linear batches should share the bounded sprite limit");
            t.Equals(GS_VULKAN_MAX_RESIDENT_TRIANGLE_BATCH,
                     GS_VRAM_PAGE_COUNT,
                     "disjoint triangle batches cannot exceed physical pages");
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

        tc.Run("source-copy alpha CT32 sprites reuse the opaque record exactly", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const GsDrawCommand opaque = makeCt32SpriteCommand(
                20u,
                0u,
                8u,
                {0u, 511u, 0u, 511u},
                {0u, 0u},
                7u * 16u,
                9u * 16u,
                35u * 16u,
                521u * 16u,
                0x2B000000u);

            const auto withAlpha = [&opaque](
                                       uint64_t sequence,
                                       uint64_t alpha,
                                       bool pabe = false)
            {
                GSPrimReg primitive = opaque.primitive();
                primitive.abe = true;
                GSContext context = opaque.context();
                context.alpha = alpha;
                GsDrawGlobalState global = opaque.globalState();
                global.pabe = pabe;
                return buildGsDrawCommand(
                    sequence,
                    primitive,
                    context,
                    std::span<const GSVertex>(opaque.vertices()).first(2u),
                    global);
            };

            const std::array<GsDrawCommand, 3> sourceCopies{{
                withAlpha(21u, 0u),
                // A=Cd, B=Cd, C=Ad, D=Cs. FIX is deliberately nonzero.
                withAlpha(22u, 0xA500000015ull),
                // A=0, B=0, C=FIX, D=Cs. PABE cannot change the result.
                withAlpha(23u, 0x7F0000002Aull, true),
            }};

            GsVulkanCt32Sprite opaqueRecord{};
            t.IsTrue(
                prepareGsVulkanCt32Sprite(opaque, opaqueRecord).supported,
                "the comparison record should prepare");
            for (const GsDrawCommand &command : sourceCopies)
            {
                const GsDrawResources resources = command.resources();
                t.IsFalse(
                    resources.framebufferReadPages.any(),
                    "a cancelling blend equation must not claim a framebuffer read");
                t.IsFalse(
                    resources.readsDestination,
                    "a source-copy blend has no effective destination dependency");

                GsVulkanCt32Sprite record{};
                const GsBackendDecision decision =
                    prepareGsVulkanCt32Sprite(command, record);
                t.IsTrue(
                    decision.supported,
                    "A=B and D=Cs should reuse the exact flat CT32 path");
                t.Equals(
                    decision.reason,
                    GsFallbackReason::Supported,
                    "a source-copy blend should retain the supported reason");
                t.IsTrue(
                    record == opaqueRecord,
                    "blend cancellation should publish the identical shader record");
            }

            const std::array<uint64_t, 3> rejectedAlpha{{
                0x44u, // (Cs-Cd)*As/128+Cd: genuine source-over.
                0x40u, // A=B but D=Cd: destination copy.
                0x0Fu, // Reserved A/B selectors are not an exact contract.
            }};
            for (uint64_t alpha : rejectedAlpha)
            {
                const GsDrawCommand rejected = withAlpha(24u, alpha);
                t.IsTrue(
                    rejected.resources().framebufferReadPages.any(),
                    "non-source-copy blending should retain its conservative read");
                GsVulkanCt32Sprite sentinel{
                    1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
                const GsBackendDecision decision =
                    prepareGsVulkanCt32Sprite(rejected, sentinel);
                t.IsFalse(
                    decision.supported,
                    "destination-dependent or reserved equations must stay closed");
                t.Equals(
                    decision.reason,
                    GsFallbackReason::AlphaBlend,
                    "rejected equations should retain the alpha-blend reason");
                t.IsTrue(
                    sentinel == GsVulkanCt32Sprite{
                        1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u},
                    "rejected alpha preparation must preserve caller output");
            }

            std::vector<uint8_t> actual = makeVramPattern(0xA17A0000u);
            std::vector<uint8_t> expected = actual;
            applyCt32SpriteCpu(expected, opaqueRecord);

            GS gs;
            gs.init(
                actual.data(),
                static_cast<uint32_t>(actual.size()),
                nullptr);
            gs.setDebugHistoryPaused(true);
            drawNearestCt32SpriteCommand(gs, sourceCopies.front());
            gs.flushRenderBatch();
            t.IsTrue(
                actual == expected,
                "the retained ALPHA=0 software draw must equal the opaque record over all VRAM");
        });

        tc.Run("depth CT32 sprite preparation publishes exact Z32 and Z24 state", [](TestCase &t)
        {
            const GsDrawCommand command = makeDepthCt32SpriteCommand(
                20u,
                511u,
                0u,
                200u,
                GS_PSM_Z24,
                false,
                1u,
                {4u, 15u, 3u, 12u},
                {32u, 16u},
                289u, 209u,
                65u, 33u,
                0xD4C3B2A1u,
                0xFEDCBA98u);

            GsVulkanDepthCt32Sprite sprite{
                1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u,
                9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u};
            const GsBackendDecision decision =
                prepareGsVulkanDepthCt32Sprite(command, sprite);
            t.IsTrue(decision.supported,
                     "an opaque CT32 sprite with an ALWAYS Z24 write should be eligible");
            t.Equals(decision.reason, GsFallbackReason::Supported,
                     "eligible depth preparation should retain the canonical reason");
            t.Equals(sprite.framebufferBaseBlock, 0x3FE0u,
                     "depth records should retain converted FRAME block units");
            t.Equals(sprite.framebufferWidth, 1u,
                     "zero FRAME width should use the software one-unit normalization");
            t.Equals(sprite.depthBaseBlock, 6400u,
                     "ZBUF page units should become 256-byte block units");
            t.Equals(sprite.depthPsm, static_cast<uint32_t>(GS_PSM_Z24),
                     "the record should retain packed Z24 format explicitly");
            t.Equals(sprite.boundsX0, 4u,
                     "depth records should retain the clipped inclusive X minimum");
            t.Equals(sprite.boundsY0, 3u,
                     "depth records should retain the clipped inclusive Y minimum");
            t.Equals(sprite.boundsX1, 16u,
                     "depth records should retain the exclusive X maximum");
            t.Equals(sprite.boundsY1, 13u,
                     "depth records should retain the exclusive Y maximum");
            t.Equals(sprite.rgba, 0xD4C3B2A1u,
                     "depth records should use the sprite provoking color");
            t.Equals(sprite.depth, 0xFEDCBA98u,
                     "depth records must use the exact integer XYZ payload");
            t.Equals(sprite.depthTestMethod, 1u,
                     "the GS ALWAYS method should remain explicit");
            t.Equals(sprite.depthWrite, 1u,
                     "an unmasked depth surface should remain writable");
            t.Equals(sprite.reserved0 | sprite.reserved1 |
                         sprite.reserved2 | sprite.reserved3,
                     0u,
                     "all reserved depth ABI words should be deterministic");
            t.IsTrue(command.resources().depthReadPages.any(),
                     "a packed Z24 write should retain its preservation read dependency");
            t.IsTrue(command.resources().depthWritePages.any(),
                     "an unmasked Z24 write should name its write pages");

            const GsDrawCommand gequalReadOnly = makeDepthCt32SpriteCommand(
                21u, 40u, 2u, 200u, GS_PSM_Z32, true, 2u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                16u, 16u, 272u, 272u,
                0x88776655u, 0x80000000u);
            t.IsTrue(
                prepareGsVulkanDepthCt32Sprite(gequalReadOnly, sprite).supported,
                "masked Z32 GEQUAL should remain a useful comparison-only class");
            t.Equals(sprite.depthPsm, static_cast<uint32_t>(GS_PSM_Z32),
                     "Z32 and Z24 should share one explicit prepared contract");
            t.Equals(sprite.depthTestMethod, 2u,
                     "GEQUAL should remain distinct from ALWAYS");
            t.Equals(sprite.depthWrite, 0u,
                     "ZMASK should suppress only the prepared depth write");

            const GsVulkanDepthCt32Sprite sentinel = sprite;
            const auto expectRejected = [&](const GsDrawCommand &rejected,
                                            GsFallbackReason reason,
                                            const char *message)
            {
                const GsBackendDecision rejectedDecision =
                    prepareGsVulkanDepthCt32Sprite(rejected, sprite);
                t.IsFalse(rejectedDecision.supported, message);
                t.Equals(rejectedDecision.reason, reason,
                         "depth rejection should retain its ordered reason");
                t.IsTrue(sprite == sentinel,
                         "rejected depth preparation must preserve the caller record");
            };

            expectRejected(
                makeDepthCt32SpriteCommand(
                    22u, 40u, 2u, 200u, GS_PSM_Z16, false, 1u,
                    {0u, 31u, 0u, 31u}, {0u, 0u},
                    16u, 16u, 272u, 272u,
                    0x01020304u, 7u),
                GsFallbackReason::UnsupportedDepthFormat,
                "Z16 should remain outside the first depth record");
            expectRejected(
                makeDepthCt32SpriteCommand(
                    23u, 40u, 2u, 200u, GS_PSM_Z32, false, 0u,
                    {0u, 31u, 0u, 31u}, {0u, 0u},
                    16u, 16u, 272u, 272u,
                    0x01020304u, 7u),
                GsFallbackReason::UnsupportedDepthFunction,
                "ZTST NEVER should not publish a record that can write nothing");
            GSContext disabledContext = gequalReadOnly.context();
            disabledContext.test = 0u;
            const GsDrawCommand disabled = buildGsDrawCommand(
                24u,
                gequalReadOnly.primitive(),
                disabledContext,
                std::span<const GSVertex>(gequalReadOnly.vertices()).first(2u),
                gequalReadOnly.globalState());
            expectRejected(
                disabled,
                GsFallbackReason::UnsupportedDepthFunction,
                "disabled depth testing belongs to the established no-depth path");
            expectRejected(
                makeDepthCt32SpriteCommand(
                    25u, 40u, 2u, 200u, GS_PSM_Z32, true, 1u,
                    {0u, 31u, 0u, 31u}, {0u, 0u},
                    16u, 16u, 272u, 272u,
                    0x01020304u, 7u),
                GsFallbackReason::UnsupportedDepthFunction,
                "ALWAYS plus ZMASK belongs to the established no-depth class");
            expectRejected(
                makeDepthCt32SpriteCommand(
                    26u, 40u, 2u, 40u, GS_PSM_Z32, false, 3u,
                    {0u, 31u, 0u, 31u}, {0u, 0u},
                    16u, 16u, 272u, 272u,
                    0x01020304u, 7u),
                GsFallbackReason::ResourceAlias,
                "framebuffer/depth aliases should remain outside the independent kernel");
        });

        tc.Run("depth CT32 prepared record matches the production software GS", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const std::array<GsDrawCommand, 4> commands{{
                makeDepthCt32SpriteCommand(
                    26u, 40u, 2u, 200u, GS_PSM_Z24, false, 1u,
                    {3u, 30u, 2u, 25u}, {16u, 32u},
                    49u, 65u, 449u, 353u,
                    0x88776655u, 0xFEDCBA98u),
                makeDepthCt32SpriteCommand(
                    27u, 40u, 2u, 200u, GS_PSM_Z32, false, 1u,
                    {3u, 30u, 2u, 25u}, {16u, 32u},
                    449u, 353u, 49u, 65u,
                    0x10203040u, 0x12345678u),
                makeDepthCt32SpriteCommand(
                    28u, 40u, 2u, 200u, GS_PSM_Z24, true, 2u,
                    {3u, 30u, 2u, 25u}, {16u, 32u},
                    49u, 65u, 449u, 353u,
                    0xA1B2C3D4u, 0x007FFFFFu),
                makeDepthCt32SpriteCommand(
                    29u, 40u, 2u, 200u, GS_PSM_Z32, false, 3u,
                    {3u, 30u, 2u, 25u}, {16u, 32u},
                    49u, 65u, 449u, 353u,
                    0x0A0B0C0Du, 0x80000000u),
            }};

            for (size_t index = 0u; index < commands.size(); ++index)
            {
                GsVulkanDepthCt32Sprite sprite{};
                if (!prepareGsVulkanDepthCt32Sprite(
                        commands[index], sprite).supported)
                {
                    t.Fail("the depth software-equivalence corpus should prepare");
                    return;
                }

                std::vector<uint8_t> actual = makeVramPattern(
                    0x5A170000u + static_cast<uint32_t>(index));
                std::vector<uint8_t> expected = actual;
                applyDepthCt32SpriteCpu(expected, sprite);

                GS gs;
                gs.init(
                    actual.data(),
                    static_cast<uint32_t>(actual.size()),
                    nullptr);
                gs.setDebugHistoryPaused(true);
                drawNearestCt32SpriteCommand(gs, commands[index]);
                gs.flushRenderBatch();

                if (actual != expected)
                {
                    std::ostringstream message;
                    message << "depth prepared record diverged from software case "
                            << index;
                    t.Fail(message.str());
                    return;
                }
            }
        });

        tc.Run("depth CT32 record survives fixed-seed edge and transition corpus", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            std::vector<uint8_t> actual = makeVramPattern(0xD37A5EEDu);
            std::vector<uint8_t> expected = actual;
            GS gs;
            gs.init(
                actual.data(),
                static_cast<uint32_t>(actual.size()),
                nullptr);
            gs.setDebugHistoryPaused(true);

            uint32_t random = 0x5A24C732u;
            constexpr size_t kCases = 4096u;
            uint64_t candidatePixels = 0u;
            size_t z24Cases = 0u;
            size_t z32Cases = 0u;
            std::array<size_t, 4> methods{};
            size_t maskedComparisons = 0u;
            for (size_t index = 0u; index < kCases; ++index)
            {
                const uint32_t bits = nextRandom(random);
                const uint8_t depthPsm =
                    (bits & 1u) != 0u ? GS_PSM_Z24 : GS_PSM_Z32;
                const uint8_t method = static_cast<uint8_t>(1u +
                    ((bits >> 1u) % 3u));
                const bool depthMask =
                    method >= 2u && ((bits >> 3u) & 1u) != 0u;
                const uint8_t framebufferWidth = static_cast<uint8_t>(
                    1u + ((bits >> 4u) % 4u));
                const uint32_t framebufferPage =
                    index % 97u == 0u ? 511u : (bits >> 7u) % 96u;
                const uint32_t depthPage =
                    index % 89u == 0u ? 255u : 256u + ((bits >> 14u) % 96u);

                const uint32_t width = 1u + (nextRandom(random) % 17u);
                const uint32_t height = 1u + (nextRandom(random) % 13u);
                const uint32_t rawOriginX = (bits >> 21u) % 96u;
                const uint32_t x0 = framebufferWidth == 1u
                    ? rawOriginX % (64u - width)
                    : rawOriginX;
                const uint32_t y0 = (nextRandom(random) >> 23u) % 48u;
                const uint32_t phaseX = nextRandom(random) & 0xFu;
                const uint32_t phaseY = nextRandom(random) & 0xFu;
                const uint32_t offsetX = nextRandom(random) & 0x3Fu;
                const uint32_t offsetY = nextRandom(random) & 0x3Fu;
                uint16_t rawX0 = static_cast<uint16_t>(
                    x0 * 16u + phaseX + offsetX);
                uint16_t rawY0 = static_cast<uint16_t>(
                    y0 * 16u + phaseY + offsetY);
                uint16_t rawX1 = static_cast<uint16_t>(
                    (x0 + width) * 16u + phaseX + offsetX);
                uint16_t rawY1 = static_cast<uint16_t>(
                    (y0 + height) * 16u + phaseY + offsetY);
                if ((nextRandom(random) & 1u) != 0u)
                    std::swap(rawX0, rawX1);
                if ((nextRandom(random) & 1u) != 0u)
                    std::swap(rawY0, rawY1);

                const uint32_t roundedX0 =
                    x0 + (phaseX != 0u ? 1u : 0u);
                const uint32_t roundedY0 =
                    y0 + (phaseY != 0u ? 1u : 0u);
                const uint32_t scissorInsetX = std::min(
                    nextRandom(random) % 3u, width - 1u);
                const uint32_t scissorInsetY = std::min(
                    nextRandom(random) % 3u, height - 1u);
                const GSScissorReg scissor{
                    static_cast<uint16_t>(roundedX0 + scissorInsetX),
                    static_cast<uint16_t>(roundedX0 + width - 1u),
                    static_cast<uint16_t>(roundedY0 + scissorInsetY),
                    static_cast<uint16_t>(roundedY0 + height - 1u),
                };
                const GsDrawCommand command = makeDepthCt32SpriteCommand(
                    1000u + index,
                    framebufferPage,
                    framebufferWidth,
                    depthPage,
                    depthPsm,
                    depthMask,
                    method,
                    scissor,
                    {static_cast<uint16_t>(offsetX),
                     static_cast<uint16_t>(offsetY)},
                    rawX0,
                    rawY0,
                    rawX1,
                    rawY1,
                    nextRandom(random),
                    nextRandom(random));
                GsVulkanDepthCt32Sprite sprite{};
                const GsBackendDecision decision =
                    prepareGsVulkanDepthCt32Sprite(command, sprite);
                if (!decision.supported)
                {
                    std::ostringstream message;
                    message << "fixed-seed depth case " << index
                            << " rejected as "
                            << gsFallbackReasonName(decision.reason);
                    t.Fail(message.str());
                    return;
                }

                applyDepthCt32SpriteCpu(expected, sprite);
                drawNearestCt32SpriteCommand(gs, command);
                candidatePixels +=
                    static_cast<uint64_t>(sprite.boundsX1 - sprite.boundsX0) *
                    (sprite.boundsY1 - sprite.boundsY0);
                ++methods[method];
                if (depthMask)
                    ++maskedComparisons;
                if (depthPsm == GS_PSM_Z24)
                    ++z24Cases;
                else
                    ++z32Cases;
            }
            gs.flushRenderBatch();

            t.IsTrue(actual == expected,
                     "all fixed-seed Z32/Z24 transitions should match the independent record oracle");
            t.Equals(z24Cases + z32Cases, kCases,
                     "the corpus should account every accepted depth format");
            t.IsTrue(z24Cases > 1000u && z32Cases > 1000u,
                     "both packed and full-word depth formats should have broad coverage");
            t.IsTrue(methods[1] > 1000u && methods[2] > 1000u &&
                         methods[3] > 1000u,
                     "ALWAYS, GEQUAL, and GREATER should each have broad coverage");
            t.IsTrue(maskedComparisons > 1000u,
                     "comparison-only masked depth draws should be common in the corpus");
            t.IsTrue(candidatePixels > 100'000u,
                     "the edge corpus should execute substantial exact pixel work");
        });

        tc.Run("Nearest CT32 texture preparation publishes exact integer sampling", [](TestCase &t)
        {
            const GsDrawCommand command = makeNearestCt32SpriteCommand(
                21u,
                40u,
                0u,
                64u,
                2u,
                6u,
                5u,
                {6u, 15u, 5u, 12u},
                {32u, 16u},
                {352u, 96u},
                {48u, 304u},
                {480u, 224u},
                {64u, 320u});

            GsVulkanNearestCt32Sprite sprite{
                1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u,
                9u, 10u, 11, 12, 13, 14, 15u, 16u};
            const GsBackendDecision decision =
                prepareGsVulkanNearestCt32Sprite(command, sprite);
            t.IsTrue(decision.supported,
                     "one-to-one point-sampled CT32 sprites should be eligible");
            t.Equals(decision.reason, GsFallbackReason::Supported,
                     "eligible texture preparation should retain the canonical reason");
            t.Equals(sprite.framebufferBaseBlock, 1280u,
                     "FRAME pages should become raw-VRAM block units");
            t.Equals(sprite.framebufferWidth, 1u,
                     "FRAME width zero should retain the software normalization");
            t.Equals(sprite.boundsX0, 6u,
                     "the texture record should start at the clipped X bound");
            t.Equals(sprite.boundsY0, 5u,
                     "the texture record should start at the clipped Y bound");
            t.Equals(sprite.boundsX1, 16u,
                     "the texture record should retain exclusive X bounds");
            t.Equals(sprite.boundsY1, 13u,
                     "the texture record should retain exclusive Y bounds");
            t.Equals(sprite.textureBaseBlock, 64u,
                     "TEX0 block units should remain native raw-VRAM units");
            t.Equals(sprite.textureWidth, 2u,
                     "the shader record should retain the exact texture stride");
            t.Equals(sprite.textureMaskU, 63u,
                     "TW should become the power-of-two U extent mask");
            t.Equals(sprite.textureMaskV, 31u,
                     "TH should become the power-of-two V extent mask");
            t.Equals(sprite.textureOriginU, 16,
                     "reversed screen endpoints and scissor clipping should advance U");
            t.Equals(sprite.textureOriginV, 7,
                     "scissor clipping should advance V from the geometric origin");
            t.Equals(sprite.textureStepU, 1,
                     "normalization should preserve positive U progression");
            t.Equals(sprite.textureStepV, 1,
                     "normalization should preserve positive V progression");
            t.Equals(gsVulkanTextureWrapMode(sprite.textureWrapU), 0u,
                     "repeat U should be explicit in the texture record");
            t.Equals(gsVulkanTextureWrapMode(sprite.textureWrapV), 0u,
                     "repeat V should be explicit in the texture record");
            t.Equals(gsVulkanTextureRegionMin(sprite.textureWrapU), 0x155u,
                     "the fixed record should retain raw MINU even when REPEAT ignores it");
            t.Equals(gsVulkanTextureRegionMax(sprite.textureWrapV), 0x266u,
                     "the fixed record should retain raw MAXV even when REPEAT ignores it");

            const GsVulkanNearestCt32Sprite sentinel = sprite;
            auto expectRejected =
                [&](const GsDrawCommand &rejected,
                    GsFallbackReason reason,
                    const char *message)
            {
                const GsBackendDecision rejectedDecision =
                    prepareGsVulkanNearestCt32Sprite(rejected, sprite);
                t.IsFalse(rejectedDecision.supported, message);
                t.Equals(rejectedDecision.reason, reason, message);
                t.IsTrue(sprite == sentinel,
                         "rejected texture preparation must preserve the caller's record");
            };
            auto rebuild =
                [&](uint64_t sequence,
                    const GSPrimReg &primitive,
                    const GSContext &context,
                    std::span<const GSVertex> vertices)
            {
                return buildGsDrawCommand(
                    sequence, primitive, context, vertices,
                    command.globalState());
            };

            GSPrimReg untextured = command.primitive();
            untextured.tme = false;
            expectRejected(
                rebuild(22u, untextured, command.context(),
                        std::span<const GSVertex>(command.vertices()).first(2u)),
                GsFallbackReason::UnsupportedTextureState,
                "the texture capability should require TME");

            GSContext format = command.context();
            format.tex0.psm = GS_PSM_T8;
            expectRejected(
                rebuild(23u, command.primitive(), format,
                        std::span<const GSVertex>(command.vertices()).first(2u)),
                GsFallbackReason::UnsupportedTextureFormat,
                "the first raw texture slice should reject indexed sources");

            GSContext function = command.context();
            function.tex0.tfx = 0u;
            expectRejected(
                rebuild(24u, command.primitive(), function,
                        std::span<const GSVertex>(command.vertices()).first(2u)),
                GsFallbackReason::UnsupportedTextureFunction,
                "the first texture slice should require direct DECAL RGBA");

            GSPrimReg stq = command.primitive();
            stq.fst = false;
            expectRejected(
                rebuild(25u, stq, command.context(),
                        std::span<const GSVertex>(command.vertices()).first(2u)),
                GsFallbackReason::UnsupportedTextureCoordinates,
                "STQ interpolation should remain outside the integer FST slice");

            std::array<GSVertex, 3> fractional = command.vertices();
            ++fractional[0].u;
            expectRejected(
                rebuild(26u, command.primitive(), command.context(),
                        std::span<const GSVertex>(fractional).first(2u)),
                GsFallbackReason::UnsupportedTextureCoordinates,
                "fractional UV endpoints should not publish an integer record");

            std::array<GSVertex, 3> scaled = command.vertices();
            scaled[0].u = static_cast<uint16_t>(scaled[0].u + 16u);
            expectRejected(
                rebuild(27u, command.primitive(), command.context(),
                        std::span<const GSVertex>(scaled).first(2u)),
                GsFallbackReason::UnsupportedTextureCoordinates,
                "integer UV scaling should remain outside the one-to-one slice");

            GSContext filtered = command.context();
            filtered.tex1 |= 1ull << 5u;
            expectRejected(
                rebuild(28u, command.primitive(), filtered,
                        std::span<const GSVertex>(command.vertices()).first(2u)),
                GsFallbackReason::UnsupportedTextureFilter,
                "linear magnification should remain outside the point slice");

            GSContext mipmapped = command.context();
            mipmapped.tex1 |= 1ull << 2u;
            expectRejected(
                rebuild(29u, command.primitive(), mipmapped,
                        std::span<const GSVertex>(command.vertices()).first(2u)),
                GsFallbackReason::UnsupportedTextureFilter,
                "non-zero maximum mip level should remain outside the point slice");

            GSContext clamped = command.context();
            clamped.clamp = (clamped.clamp & ~0xFull) | 0x5u;
            GsVulkanNearestCt32Sprite clampedSprite{};
            const GsBackendDecision clampedDecision =
                prepareGsVulkanNearestCt32Sprite(
                    rebuild(
                        30u, command.primitive(), clamped,
                        std::span<const GSVertex>(command.vertices()).first(2u)),
                    clampedSprite);
            t.IsTrue(clampedDecision.supported,
                     "standard clamp should be eligible on both axes");
            t.Equals(
                gsVulkanTextureWrapMode(clampedSprite.textureWrapU), 1u,
                "standard U clamp should reach the shader record");
            t.Equals(
                gsVulkanTextureWrapMode(clampedSprite.textureWrapV), 1u,
                "standard V clamp should reach the shader record");

            GSContext regionClamped = command.context();
            regionClamped.clamp =
                2ull | (1ull << 2u) |
                (70ull << 4u) | (72ull << 14u) |
                (40ull << 24u) | (42ull << 34u);
            GsVulkanNearestCt32Sprite regionClampedSprite{};
            const GsBackendDecision regionClampedDecision =
                prepareGsVulkanNearestCt32Sprite(
                    rebuild(
                        35u, command.primitive(), regionClamped,
                        std::span<const GSVertex>(command.vertices()).first(2u)),
                    regionClampedSprite);
            t.IsTrue(regionClampedDecision.supported,
                     "REGION_CLAMP should be eligible independently of REGION_REPEAT");
            t.Equals(
                gsVulkanTextureWrapMode(regionClampedSprite.textureWrapU), 2u,
                "REGION_CLAMP U should reach the shader record");
            t.Equals(
                gsVulkanTextureRegionMin(regionClampedSprite.textureWrapU),
                70u,
                "REGION_CLAMP U should retain raw MINU beyond nominal width");
            t.Equals(
                gsVulkanTextureRegionMax(regionClampedSprite.textureWrapU),
                72u,
                "REGION_CLAMP U should retain raw MAXU beyond nominal width");
            t.Equals(
                gsVulkanTextureWrapMode(regionClampedSprite.textureWrapV), 1u,
                "standard V clamp should remain independent");

            GSContext reversedRegionClamp = regionClamped;
            reversedRegionClamp.clamp &=
                ~((0x3FFull << 4u) | (0x3FFull << 14u));
            reversedRegionClamp.clamp |=
                (73ull << 4u) | (72ull << 14u);
            expectRejected(
                rebuild(
                    37u, command.primitive(), reversedRegionClamp,
                    std::span<const GSVertex>(command.vertices()).first(2u)),
                GsFallbackReason::UnsupportedTextureWrap,
                "unproven reversed REGION_CLAMP bounds should remain closed");

            GSContext regionRepeated = command.context();
            regionRepeated.clamp =
                3ull | (3ull << 2u) |
                (127ull << 4u) | (128ull << 14u) |
                (63ull << 24u) | (64ull << 34u);
            GsVulkanNearestCt32Sprite regionRepeatedSprite{};
            const GsBackendDecision regionRepeatedDecision =
                prepareGsVulkanNearestCt32Sprite(
                    rebuild(
                        36u, command.primitive(), regionRepeated,
                        std::span<const GSVertex>(command.vertices()).first(2u)),
                    regionRepeatedSprite);
            t.IsTrue(regionRepeatedDecision.supported,
                     "REGION_REPEAT should be independently eligible");
            t.Equals(
                gsVulkanTextureWrapMode(regionRepeatedSprite.textureWrapU), 3u,
                "REGION_REPEAT U should reach the shader record");
            t.Equals(
                gsVulkanTextureRegionMin(regionRepeatedSprite.textureWrapU),
                127u,
                "REGION_REPEAT U should retain raw MINU before nominal masking");
            t.Equals(
                gsVulkanTextureRegionMax(regionRepeatedSprite.textureWrapU),
                128u,
                "REGION_REPEAT U should retain its raw MAXU offset");
            t.Equals(
                gsVulkanTextureWrapMode(regionRepeatedSprite.textureWrapV), 3u,
                "REGION_REPEAT V should remain independent");

            GSContext zeroStride = command.context();
            zeroStride.tex0.tbw = 0u;
            expectRejected(
                rebuild(31u, command.primitive(), zeroStride,
                        std::span<const GSVertex>(command.vertices()).first(2u)),
                GsFallbackReason::UnsupportedTextureState,
                "zero texture stride should remain an explicit later capability");

            GSContext oversized = command.context();
            oversized.tex0.tw = 11u;
            expectRejected(
                rebuild(32u, command.primitive(), oversized,
                        std::span<const GSVertex>(command.vertices()).first(2u)),
                GsFallbackReason::UnsupportedTextureState,
                "texture exponents above the GS UV range should fail closed");

            GSContext aliased = command.context();
            aliased.tex0.tbp0 = command.context().frame.fbp << 5u;
            expectRejected(
                rebuild(33u, command.primitive(), aliased,
                        std::span<const GSVertex>(command.vertices()).first(2u)),
                GsFallbackReason::ResourceAlias,
                "raw texture reads must not overlap framebuffer writes");

            const GsDrawCommand selfAliased =
                makeNearestCt32SpriteCommand(
                    34u, 40u, 1u, 64u, 2u, 7u, 0u,
                    {0u, 64u, 0u, 0u}, {0u, 0u},
                    {0u, 65u * 16u}, {0u, 16u},
                    {0u, 65u * 16u}, {0u, 16u});
            expectRejected(
                selfAliased,
                GsFallbackReason::UnknownMemoryLayout,
                "textured destinations must not wrap onto their own CT32 words");

            t.Equals(
                gsFallbackReasonName(
                    GsFallbackReason::UnsupportedTextureCoordinates),
                std::string_view("unsupported-texture-coordinates"),
                "texture fallbacks should have stable diagnostic names");
        });

        tc.Run("Nearest CT32 texture records match software point sampling", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            struct TextureCase
            {
                GSScissorReg scissor;
                GSXYOffsetReg xyoffset;
                std::array<uint16_t, 2> x;
                std::array<uint16_t, 2> y;
                std::array<uint16_t, 2> u;
                std::array<uint16_t, 2> v;
                uint8_t tw;
                uint8_t th;
                uint8_t wrapModeU;
                uint8_t wrapModeV;
                uint16_t regionMinU = 0x155u;
                uint16_t regionMaxU = 0x2AAu;
                uint16_t regionMinV = 0x133u;
                uint16_t regionMaxV = 0x266u;
            };
            const std::array<TextureCase, 14> cases{{
                {
                    {6u, 15u, 5u, 12u}, {32u, 16u},
                    {352u, 96u}, {48u, 304u},
                    {480u, 224u}, {64u, 320u}, 6u, 5u, 0u, 0u,
                },
                {
                    {3u, 14u, 2u, 6u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {992u, 1248u}, {480u, 608u}, 6u, 5u, 0u, 0u,
                },
                {
                    {10u, 20u, 7u, 17u}, {16u, 32u},
                    {400u, 144u}, {352u, 96u},
                    {80u, 336u}, {32u, 288u}, 5u, 5u, 0u, 0u,
                },
                {
                    {5u, 10u, 6u, 10u}, {48u, 16u},
                    {112u, 240u}, {208u, 80u},
                    {320u, 192u}, {320u, 192u}, 5u, 4u, 0u, 0u,
                },
                {
                    {1u, 3u, 1u, 3u}, {0u, 0u},
                    {0u, 64u}, {0u, 64u},
                    {160u, 224u}, {320u, 384u}, 0u, 0u, 0u, 0u,
                },
                {
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 3u, 2u, 1u, 0u,
                },
                {
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 3u, 2u, 0u, 1u,
                },
                {
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 3u, 2u, 1u, 1u,
                },
                {
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 3u, 2u, 2u, 0u,
                    10u, 12u, 0u, 0u,
                },
                {
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 3u, 2u, 0u, 2u,
                    0u, 0u, 6u, 9u,
                },
                {
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 3u, 2u, 2u, 2u,
                    10u, 12u, 6u, 9u,
                },
                {
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 3u, 2u, 3u, 0u,
                    15u, 16u, 0u, 0u,
                },
                {
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 3u, 2u, 0u, 3u,
                    0u, 0u, 15u, 16u,
                },
                {
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 3u, 2u, 3u, 3u,
                    15u, 32u, 15u, 24u,
                },
            }};

            uint64_t sequence = 100u;
            for (size_t index = 0u; index < cases.size(); ++index)
            {
                const TextureCase &textureCase = cases[index];
                const GsDrawCommand command = makeNearestCt32SpriteCommand(
                    sequence++, 40u, 2u, 64u, 2u,
                    textureCase.tw, textureCase.th,
                    textureCase.scissor, textureCase.xyoffset,
                    textureCase.x, textureCase.y,
                    textureCase.u, textureCase.v,
                    textureCase.wrapModeU,
                    textureCase.wrapModeV,
                    textureCase.regionMinU,
                    textureCase.regionMaxU,
                    textureCase.regionMinV,
                    textureCase.regionMaxV);
                GsVulkanNearestCt32Sprite sprite{};
                const GsBackendDecision decision =
                    prepareGsVulkanNearestCt32Sprite(command, sprite);
                if (!decision.supported)
                {
                    t.Fail(
                        "the software-equivalence corpus must satisfy the narrow texture predicate");
                    return;
                }

                std::vector<uint8_t> actual = makeVramPattern(
                    0x54455830u + static_cast<uint32_t>(index));
                std::vector<uint8_t> expected = actual;
                applyNearestCt32SpriteCpu(expected, sprite);

                GS gs;
                gs.init(
                    actual.data(),
                    static_cast<uint32_t>(actual.size()),
                    nullptr);
                gs.setDebugHistoryPaused(true);
                drawNearestCt32SpriteCommand(gs, command);
                gs.flushRenderBatch();

                if (actual != expected)
                {
                    std::ostringstream message;
                    message << "nearest CT32 record diverged from software case "
                            << index;
                    t.Fail(message.str());
                    return;
                }
            }
        });

        tc.Run("Linear CT32 repeat preparation retains the exact software DDA", [](TestCase &t)
        {
            const GsDrawCommand command =
                makeLinearCt32RepeatSpriteCommand(
                    140u,
                    0u,
                    8u,
                    3584u,
                    8u,
                    10u,
                    10u,
                    {0u, 511u, 0u, 447u},
                    {28672u, 29184u},
                    {28664u, 29176u},
                    {29176u, 36344u},
                    {0u, 512u},
                    {0u, 6656u});

            GsVulkanLinearCt32Sprite sprite{};
            sprite.framebufferBaseBlock = 0xDEADu;
            const GsVulkanLinearCt32Sprite sentinel = sprite;
            const GsBackendDecision decision =
                prepareGsVulkanLinearCt32Sprite(command, sprite);
            t.IsTrue(
                decision.supported,
                "the retained 32x448 / 32x416 linear repeat sprite should be eligible");
            if (!decision.supported)
                return;

            t.Equals(decision.reason, GsFallbackReason::Supported,
                     "eligible linear preparation should retain the canonical reason");
            t.Equals(sprite.framebufferBaseBlock, 0u,
                     "FRAME page zero should become raw block zero");
            t.Equals(sprite.framebufferWidth, 8u,
                     "the title framebuffer stride should remain exact");
            t.Equals(sprite.boundsX0, 0u,
                     "the half-pixel left edge should ceil to zero");
            t.Equals(sprite.boundsY0, 0u,
                     "the half-pixel top edge should ceil to zero");
            t.Equals(sprite.boundsX1, 32u,
                     "the retained title width should be exact");
            t.Equals(sprite.boundsY1, 448u,
                     "the retained title height should be exact");
            t.Equals(sprite.textureBaseBlock, 3584u,
                     "the title texture block should remain native");
            t.Equals(sprite.textureWidth, 8u,
                     "the title texture stride should remain native");
            t.Equals(sprite.textureMaskU, 1023u,
                     "TW=10 should publish its exact repeat mask");
            t.Equals(sprite.textureMaskV, 1023u,
                     "TH=10 should publish its exact repeat mask");
            t.Equals(sprite.fixedBaseU, 0,
                     "half-texel bias and half-pixel prestep should center the first U sample");
            t.Equals(sprite.fixedBlockStepU, 8 * 65536,
                     "one-to-one U should retain the eight-pixel GS block step");
            for (size_t lane = 0u; lane < sprite.fixedLaneU.size(); ++lane)
            {
                t.Equals(
                    sprite.fixedLaneU[lane],
                    static_cast<int32_t>(lane * 65536u),
                    "the prepared record should retain every truncated GS U lane");
            }
            t.Equals(sprite.fixedScanVBits, 0xC5124928u,
                     "the clipped first V scanline must retain its exact binary32 seed");
            t.Equals(sprite.fixedStepVBits, 0x476DB6DBu,
                     "the 416/448 V scale must retain its exact binary32 step");
            t.Equals(gsVulkanTextureWrapMode(sprite.textureWrapU), 0u,
                     "the initial linear record should be repeat-only on U");
            t.Equals(gsVulkanTextureWrapMode(sprite.textureWrapV), 0u,
                     "the initial linear record should be repeat-only on V");

            const auto rebuild = [&command](
                uint64_t sequence,
                const GSPrimReg &primitive,
                const GSContext &context)
            {
                return buildGsDrawCommand(
                    sequence,
                    primitive,
                    context,
                    std::span<const GSVertex>(command.vertices()).first(2u),
                    command.globalState());
            };
            const auto expectRejected = [&t, &sprite, &rebuild](
                const GsDrawCommand &rejected,
                GsFallbackReason expectedReason,
                const char *message)
            {
                const GsVulkanLinearCt32Sprite prior = sprite;
                const GsBackendDecision rejectedDecision =
                    prepareGsVulkanLinearCt32Sprite(rejected, sprite);
                t.IsFalse(rejectedDecision.supported, message);
                t.Equals(rejectedDecision.reason, expectedReason, message);
                t.IsTrue(sprite == prior,
                         "linear rejection must preserve the caller's record");
            };

            GSContext point = command.context();
            point.tex1 &= ~((1ull << 5u) | (0x7ull << 6u));
            expectRejected(
                rebuild(141u, command.primitive(), point),
                GsFallbackReason::UnsupportedTextureFilter,
                "point filtering must remain on the nearest pipeline");

            GSContext clamp = command.context();
            clamp.clamp = (clamp.clamp & ~0xFull) | 0x5u;
            const GsDrawCommand clamped =
                rebuild(142u, command.primitive(), clamp);
            GsVulkanLinearCt32Sprite clampedSprite{};
            const GsBackendDecision clampedDecision =
                prepareGsVulkanLinearCt32Sprite(
                    clamped, clampedSprite);
            t.IsTrue(
                clampedDecision.supported,
                "standard clamp should extend the same exact linear DDA");
            t.Equals(
                gsVulkanTextureWrapMode(clampedSprite.textureWrapU),
                1u,
                "linear U clamp should reach the prepared record");
            t.Equals(
                gsVulkanTextureWrapMode(clampedSprite.textureWrapV),
                1u,
                "linear V clamp should reach the prepared record");

            GSContext regionClamp = command.context();
            regionClamp.clamp =
                (regionClamp.clamp & ~0xFull) | 0xAu;
            expectRejected(
                rebuild(143u, command.primitive(), regionClamp),
                GsFallbackReason::UnsupportedTextureWrap,
                "linear region clamp should remain a later slice");

            GSPrimReg stq = command.primitive();
            stq.fst = false;
            expectRejected(
                rebuild(144u, stq, command.context()),
                GsFallbackReason::UnsupportedTextureCoordinates,
                "linear STQ should remain outside the prepared FST DDA");

            GSContext aliased = command.context();
            aliased.tex0.tbp0 = command.context().frame.fbp << 5u;
            expectRejected(
                rebuild(145u, command.primitive(), aliased),
                GsFallbackReason::ResourceAlias,
                "linear raw texture reads must remain disjoint from writes");

            t.IsFalse(sprite == sentinel,
                      "successful preparation should replace the initial sentinel");
        });

        tc.Run("Linear CT32 repeat records match software bilinear sampling", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            struct LinearCase
            {
                uint32_t framebufferPage;
                uint8_t framebufferWidth;
                uint32_t textureBaseBlock;
                uint8_t textureWidth;
                uint8_t tw;
                uint8_t th;
                GSScissorReg scissor;
                GSXYOffsetReg xyoffset;
                std::array<uint16_t, 2> x;
                std::array<uint16_t, 2> y;
                std::array<uint16_t, 2> u;
                std::array<uint16_t, 2> v;
            };
            const std::array<LinearCase, 4> cases{{
                {
                    0u, 8u, 3584u, 8u, 10u, 10u,
                    {0u, 511u, 0u, 447u},
                    {28672u, 29184u},
                    {28664u, 29176u},
                    {29176u, 36344u},
                    {0u, 512u},
                    {0u, 6656u},
                },
                {
                    0u, 2u, 512u, 2u, 6u, 5u,
                    {3u, 12u, 2u, 13u},
                    {128u, 96u},
                    {120u, 376u},
                    {88u, 344u},
                    {0u, 249u},
                    {0u, 137u},
                },
                {
                    0u, 2u, 512u, 2u, 6u, 5u,
                    {4u, 14u, 3u, 12u},
                    {128u, 96u},
                    {400u, 144u},
                    {352u, 96u},
                    {9u, 521u},
                    {17u, 273u},
                },
                {
                    0u, 1u, 512u, 1u, 3u, 3u,
                    {0u, 7u, 0u, 7u},
                    {128u, 96u},
                    {120u, 248u},
                    {88u, 216u},
                    {0u, 128u},
                    {0u, 128u},
                },
            }};

            uint64_t sequence = 150u;
            for (size_t index = 0u; index < cases.size(); ++index)
            {
                const LinearCase &linearCase = cases[index];
                const GsDrawCommand command =
                    makeLinearCt32RepeatSpriteCommand(
                        sequence++,
                        linearCase.framebufferPage,
                        linearCase.framebufferWidth,
                        linearCase.textureBaseBlock,
                        linearCase.textureWidth,
                        linearCase.tw,
                        linearCase.th,
                        linearCase.scissor,
                        linearCase.xyoffset,
                        linearCase.x,
                        linearCase.y,
                        linearCase.u,
                        linearCase.v);
                GsVulkanLinearCt32Sprite sprite{};
                const GsBackendDecision decision =
                    prepareGsVulkanLinearCt32Sprite(command, sprite);
                if (!decision.supported)
                {
                    t.Fail(
                        "the linear software-equivalence corpus must satisfy the narrow predicate");
                    return;
                }

                std::vector<uint8_t> actual = makeVramPattern(
                    0x4C494E30u + static_cast<uint32_t>(index));
                std::vector<uint8_t> expected = actual;
                applyLinearCt32SpriteCpu(expected, sprite);

                GS gs;
                gs.init(
                    actual.data(),
                    static_cast<uint32_t>(actual.size()),
                    nullptr);
                gs.setDebugHistoryPaused(true);
                drawNearestCt32SpriteCommand(gs, command);
                gs.flushRenderBatch();

                if (actual != expected)
                {
                    std::ostringstream message;
                    message
                        << "linear CT32 record diverged from software case "
                        << index;
                    t.Fail(message.str());
                    return;
                }
            }
        });

        tc.Run("Linear CT32 standard clamp records match software bilinear edges", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            struct ClampCase
            {
                uint8_t wrapU;
                uint8_t wrapV;
                std::array<uint16_t, 2> u;
                std::array<uint16_t, 2> v;
            };
            const std::array<ClampCase, 4> cases{{
                {1u, 0u, {0u, 128u}, {0u, 128u}},
                {0u, 1u, {0u, 128u}, {0u, 128u}},
                {1u, 1u, {0u, 128u}, {0u, 128u}},
                {1u, 1u, {128u, 256u}, {128u, 256u}},
            }};

            uint64_t sequence = 160u;
            for (size_t index = 0u; index < cases.size(); ++index)
            {
                const ClampCase &clampCase = cases[index];
                const GsDrawCommand command = makeLinearCt32SpriteCommand(
                    sequence++,
                    0u,
                    1u,
                    512u,
                    1u,
                    3u,
                    3u,
                    {0u, 7u, 0u, 7u},
                    {128u, 96u},
                    {128u, 256u},
                    {96u, 224u},
                    clampCase.u,
                    clampCase.v,
                    clampCase.wrapU,
                    clampCase.wrapV);
                GsVulkanLinearCt32Sprite sprite{};
                const GsBackendDecision decision =
                    prepareGsVulkanLinearCt32Sprite(command, sprite);
                if (!decision.supported)
                {
                    t.Fail(
                        "the standard-clamp edge corpus must satisfy the linear predicate");
                    return;
                }

                t.Equals(
                    gsVulkanTextureWrapMode(sprite.textureWrapU),
                    clampCase.wrapU,
                    "the clamp corpus should retain its U mode");
                t.Equals(
                    gsVulkanTextureWrapMode(sprite.textureWrapV),
                    clampCase.wrapV,
                    "the clamp corpus should retain its V mode");

                std::vector<uint8_t> actual = makeVramPattern(
                    0x434C4D50u + static_cast<uint32_t>(index));
                std::vector<uint8_t> expected = actual;
                applyLinearCt32SpriteCpu(expected, sprite);

                GS gs;
                gs.init(
                    actual.data(),
                    static_cast<uint32_t>(actual.size()),
                    nullptr);
                gs.setDebugHistoryPaused(true);
                drawNearestCt32SpriteCommand(gs, command);
                gs.flushRenderBatch();

                if (actual != expected)
                {
                    std::ostringstream message;
                    message
                        << "linear CT32 clamp record diverged from software case "
                        << index;
                    t.Fail(message.str());
                    return;
                }
            }
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
                     GsFallbackReason::UnsupportedTextureFunction,
                     "strict should retain the first unmet texture invariant");
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

        tc.Run("Vulkan raster backend verifies nearest CT32 textures behind capability", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const GsDrawCommand command = makeNearestCt32SpriteCommand(
                43u, 40u, 2u, 64u, 2u, 6u, 5u,
                {6u, 15u, 5u, 12u}, {32u, 16u},
                {352u, 96u}, {48u, 304u},
                {480u, 224u}, {64u, 320u}, 3u, 3u,
                15u, 128u, 15u, 64u);
            GsVulkanNearestCt32Sprite prepared{};
            t.IsTrue(
                prepareGsVulkanNearestCt32Sprite(
                    command, prepared).supported,
                "the REGION_REPEAT verification fixture should satisfy the nearest texture predicate");

            std::vector<uint8_t> vram = makeVramPattern(0x54585631u);
            const std::vector<uint8_t> initial = vram;
            std::vector<uint8_t> expected = initial;
            applyNearestCt32SpriteCpu(expected, prepared);
            uint64_t softwareCalls = 0u;
            uint64_t commitCalls = 0u;
            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::Verify;
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram,
                    [&](const GsDrawCommand &draw)
                    {
                        ++softwareCalls;
                        GsVulkanNearestCt32Sprite sprite{};
                        if (prepareGsVulkanNearestCt32Sprite(
                                draw, sprite).supported)
                        {
                            applyNearestCt32SpriteCpu(vram, sprite);
                        }
                    },
                    [&](const GsDrawCommand &) { ++commitCalls; },
                    nullptr);
            t.IsNotNull(backend.get(),
                        "an exact texture executor should create Verify");
            if (!backend)
                return;

            t.IsTrue(backend->classify(command).supported,
                     "Verify should expose the capability-gated nearest texture class");
            backend->submit(
                std::span<const GsDrawCommand>(&command, 1u));
            t.IsTrue(vram == expected,
                     "Verify should retain the agreed textured software image");
            t.Equals(softwareCalls, 1ull,
                     "texture Verify should run the software oracle once");
            const GsVulkanRasterBackendStatistics statistics =
                backend->backendStatistics();
            t.Equals(statistics.commandsAttempted, 1ull,
                     "texture Verify should attempt one command");
            t.Equals(statistics.commandsCompleted, 1ull,
                     "an agreeing texture draw should complete once");
            t.Equals(statistics.verifiedCommands, 1ull,
                     "the full textured comparison should be counted");
            t.Equals(statistics.bytesCompared,
                     static_cast<uint64_t>(GS_VULKAN_VRAM_SIZE),
                     "texture Verify should compare all 4 MiB");
            const GsVulkanServiceStatistics serviceStatistics =
                backend->serviceStatistics();
            t.Equals(
                serviceStatistics.nearestCt32SpriteDrawsCompleted,
                1ull,
                "the executor should receive one nearest texture request");
            t.Equals(serviceStatistics.spriteDrawsCompleted, 0ull,
                     "texture Verify must not use the flat sprite kernel");

            t.IsTrue(backend->setMode(GsRendererMode::GpuStrict),
                     "the synchronized backend should accept strict mode");
            t.IsTrue(backend->classify(command).supported,
                     "strict should expose the qualified nearest texture class");
            std::copy(initial.begin(), initial.end(), vram.begin());
            GsVramPageMask allPages;
            allPages.setAll();
            backend->noteCpuVramWrite(allPages);
            backend->submit(
                std::span<const GsDrawCommand>(&command, 1u));
            t.IsTrue(vram == initial,
                     "strict texture queueing should leave canonical VRAM untouched");
            t.Equals(softwareCalls, 1ull,
                     "strict texture execution must not call the software oracle");
            t.Equals(commitCalls, 0ull,
                     "queued strict textures should not publish commit metadata early");
            t.Equals(backend->pendingCommandCount(), size_t{1u},
                     "strict textures should enter the bounded resident queue");
            const GsDrawResources resources = command.resources();
            GsVramPageMask strictAccessPages = resources.readPages;
            strictAccessPages.unionWith(resources.writePages);
            backend->prepareCpuVramAccess(
                resources.writePages, GsFlushReason::CpuReadback);
            t.IsTrue(vram == expected,
                     "an overlapping CPU observation should publish the exact texture result");
            t.Equals(commitCalls, 1ull,
                     "drained strict texture execution should publish one commit callback");
            t.Equals(backend->pendingCommandCount(), size_t{0u},
                     "the observation boundary should drain resident texture work");
            const GsVulkanRasterBackendStatistics strictStatistics =
                backend->backendStatistics();
            t.Equals(strictStatistics.commandsAttempted, 2ull,
                     "Verify and strict should each attempt the texture draw");
            t.Equals(strictStatistics.commandsCompleted, 2ull,
                     "Verify and strict should each complete the texture draw");
            t.Equals(strictStatistics.verifiedCommands, 1ull,
                     "strict should not increment verification accounting");
            t.Equals(strictStatistics.committedGpuCommands, 1ull,
                     "strict should commit the GPU result exactly once");
            t.Equals(strictStatistics.residentCommands, 1ull,
                     "strict should count the drained resident texture once");
            t.Equals(strictStatistics.pageOwnership.cpuNewerPages,
                     GS_VRAM_PAGE_COUNT - strictAccessPages.count(),
                     "resident strict should upload only conservative texture access pages");
            t.Equals(strictStatistics.pageOwnership.gpuNewerPages,
                     size_t{0u},
                     "the scoped observation should publish the texture destination");
            const GsVulkanServiceStatistics strictServiceStatistics =
                backend->serviceStatistics();
            t.Equals(
                strictServiceStatistics.nearestCt32SpriteDrawsCompleted,
                2ull,
                "Verify and resident strict should each execute one texture draw");
            t.Equals(
                strictServiceStatistics
                    .residentNearestCt32SpriteBatchesCompleted,
                1ull,
                "strict should execute one resident texture batch");
            t.IsTrue(backend->setMode(GsRendererMode::Hybrid),
                     "the synchronized backend should accept Hybrid mode");
            t.Equals(backend->classify(command).reason,
                     GsFallbackReason::CostModel,
                     "Hybrid should keep the small qualified texture on the CPU");

            std::vector<uint8_t> unavailableVram = initial;
            std::unique_ptr<GsVulkanRasterBackend> unavailable =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact,
                        true, false),
                    config, unavailableVram,
                    [](const GsDrawCommand &) {}, {}, nullptr);
            t.IsNotNull(unavailable.get(),
                        "a base-capable executor should retain texture fallback");
            if (unavailable)
            {
                t.IsTrue(unavailable->setMode(GsRendererMode::Hybrid),
                         "the unavailable fixture should exercise Hybrid ordering");
                t.Equals(unavailable->classify(command).reason,
                         GsFallbackReason::BackendUnavailable,
                         "missing exact texture capability should precede cost fallback");
                t.IsTrue(unavailable->setMode(GsRendererMode::GpuStrict),
                         "the base-capable executor should still enter strict mode");
                t.Equals(unavailable->classify(command).reason,
                         GsFallbackReason::BackendUnavailable,
                         "strict should preserve the missing texture capability reason");
                t.Equals(
                    unavailable->serviceStatistics()
                        .nearestCt32SpriteDrawsFailed,
                    0ull,
                         "capability fallback must not post texture work");
            }

            std::vector<uint8_t> failureVram = initial;
            uint64_t failedSoftwareCalls = 0u;
            std::unique_ptr<GsVulkanRasterBackend> failing =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::FailResidentDraw),
                    GsVulkanRasterBackendConfig{
                        GsRendererMode::GpuStrict, {}},
                    failureVram,
                    [&](const GsDrawCommand &)
                    {
                        ++failedSoftwareCalls;
                    },
                    {}, nullptr);
            t.IsNotNull(failing.get(),
                        "an initially healthy failing texture executor should construct");
            bool executionThrew = false;
            if (failing)
            {
                failing->submit(
                    std::span<const GsDrawCommand>(&command, 1u));
                t.Equals(failing->pendingCommandCount(), size_t{1u},
                         "a failing resident texture should first enter the queue");
                try
                {
                    failing->flush(GsFlushReason::Explicit);
                }
                catch (const std::runtime_error &error)
                {
                    executionThrew =
                        std::string(error.what()).find(
                            "before canonical VRAM mutation") !=
                        std::string::npos;
                }
                t.Equals(
                    failing->backendStatistics().gpuRequestsFailed,
                    1ull,
                    "strict texture execution failure should be counted once");
                t.Equals(
                    failing->serviceStatistics()
                        .residentNearestCt32SpriteBatchesFailed,
                    1ull,
                    "the deferred texture failure should count one resident batch");
            }
            t.IsTrue(executionThrew,
                     "strict texture failure should identify its atomic boundary");
            t.IsTrue(failureVram == initial,
                     "strict texture failure must preserve canonical VRAM");
            t.Equals(failedSoftwareCalls, 0ull,
                     "strict texture failure must not invoke software fallback");
        });

        tc.Run("Vulkan raster backend verifies linear CT32 repeat and clamp behind capability", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const GsDrawCommand command =
                makeLinearCt32RepeatSpriteCommand(
                    98u, 0u, 8u, 3584u, 8u, 10u, 10u,
                    {0u, 511u, 0u, 447u},
                    {28672u, 29184u},
                    {28664u, 29176u},
                    {29176u, 36344u},
                    {0u, 512u},
                    {0u, 6656u});
            GsVulkanLinearCt32Sprite prepared{};
            t.IsTrue(
                prepareGsVulkanLinearCt32Sprite(
                    command, prepared).supported,
                "the retained title linear sprite should satisfy its prepared-DDA predicate");

            const std::vector<uint8_t> initial =
                makeVramPattern(0x4C565246u);
            std::vector<uint8_t> expected = initial;
            applyLinearCt32SpriteCpu(expected, prepared);
            std::vector<uint8_t> vram = initial;
            uint64_t softwareCalls = 0u;
            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::Verify;
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram,
                    [&](const GsDrawCommand &draw)
                    {
                        ++softwareCalls;
                        GsVulkanLinearCt32Sprite sprite{};
                        if (prepareGsVulkanLinearCt32Sprite(
                                draw, sprite).supported)
                        {
                            applyLinearCt32SpriteCpu(vram, sprite);
                        }
                    },
                    {}, nullptr);
            t.IsNotNull(backend.get(),
                        "an exact linear executor should create Verify");
            if (!backend)
                return;

            const GsDrawCommand clampedCommand =
                makeLinearCt32SpriteCommand(
                    99u, 0u, 8u, 3584u, 8u, 10u, 10u,
                    {0u, 511u, 0u, 447u},
                    {28672u, 29184u},
                    {28664u, 29176u},
                    {29176u, 36344u},
                    {0u, 512u},
                    {0u, 6656u},
                    1u, 1u);
            GsVulkanLinearCt32Sprite clampedPrepared{};
            t.IsTrue(
                prepareGsVulkanLinearCt32Sprite(
                    clampedCommand, clampedPrepared).supported,
                "standard clamp should be valid at the prepared-record seam");
            const GsBackendDecision clampedRoute =
                backend->classify(clampedCommand);
            t.IsTrue(
                clampedRoute.supported,
                "Verify should expose qualified linear standard clamp");
            if (!clampedRoute.supported)
                return;

            t.IsTrue(backend->classify(command).supported,
                     "Verify should expose capability-gated linear repeat sprites");
            backend->submit(
                std::span<const GsDrawCommand>(&command, 1u));
            t.IsTrue(vram == expected,
                     "linear Verify should retain the agreed software image");
            t.Equals(softwareCalls, 1ull,
                     "linear Verify should run its independent oracle once");
            const GsVulkanRasterBackendStatistics statistics =
                backend->backendStatistics();
            t.Equals(statistics.commandsAttempted, 1ull,
                     "linear Verify should attempt one command");
            t.Equals(statistics.commandsCompleted, 1ull,
                     "an agreeing linear command should complete once");
            t.Equals(statistics.verifiedCommands, 1ull,
                     "the full linear comparison should be counted");
            t.Equals(statistics.bytesCompared,
                     static_cast<uint64_t>(GS_VULKAN_VRAM_SIZE),
                     "linear Verify should compare all 4 MiB");
            const GsVulkanServiceStatistics serviceStatistics =
                backend->serviceStatistics();
            t.Equals(serviceStatistics.linearCt32SpriteDrawsCompleted,
                     1ull,
                     "the executor should receive one linear request");
            t.Equals(serviceStatistics.nearestCt32SpriteDrawsCompleted,
                     0ull,
                     "linear Verify must not alias the nearest request");

            std::vector<uint8_t> clampedExpected = expected;
            applyLinearCt32SpriteCpu(
                clampedExpected, clampedPrepared);
            backend->submit(
                std::span<const GsDrawCommand>(&clampedCommand, 1u));
            t.IsTrue(
                vram == clampedExpected,
                "linear clamp Verify should retain the agreed software image");
            t.Equals(
                softwareCalls,
                2ull,
                "linear clamp Verify should run its independent oracle once");
            const GsVulkanRasterBackendStatistics clampedStatistics =
                backend->backendStatistics();
            t.Equals(clampedStatistics.commandsAttempted, 2ull,
                     "repeat and clamp should each reach Verify");
            t.Equals(clampedStatistics.commandsCompleted, 2ull,
                     "repeat and clamp should each complete once");
            t.Equals(clampedStatistics.verifiedCommands, 2ull,
                     "repeat and clamp should each compare independently");
            t.Equals(
                clampedStatistics.bytesCompared,
                2ull * GS_VULKAN_VRAM_SIZE,
                "two linear draws should compare two complete VRAM images");
            t.Equals(
                backend->serviceStatistics()
                    .linearCt32SpriteDrawsCompleted,
                2ull,
                "the executor should receive repeat and clamp requests");

            t.IsTrue(backend->setMode(GsRendererMode::GpuStrict),
                     "the Verify fixture should enter strict mode cleanly");
            t.IsTrue(
                backend->classify(clampedCommand).supported,
                "strict should expose resident-qualified linear clamp");
            t.IsTrue(backend->setMode(GsRendererMode::Verify),
                     "the fixture should restore Verify after its strict gate");

            std::vector<uint8_t> unavailableVram = initial;
            std::unique_ptr<GsVulkanRasterBackend> unavailable =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact,
                        true, true, false),
                    config, unavailableVram,
                    [](const GsDrawCommand &) {}, {}, nullptr);
            t.IsNotNull(unavailable.get(),
                        "a base-capable executor should retain linear fallback");
            if (unavailable)
            {
                t.Equals(unavailable->classify(command).reason,
                         GsFallbackReason::BackendUnavailable,
                         "a missing exact linear capability should fail closed");
                t.Equals(
                    unavailable->serviceStatistics()
                        .linearCt32SpriteDrawsFailed,
                    0ull,
                    "capability fallback must not post linear work");
            }

            std::vector<uint8_t> failureVram = initial;
            uint64_t failedSoftwareCalls = 0u;
            std::unique_ptr<GsVulkanRasterBackend> failing =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Fail),
                    config, failureVram,
                    [&](const GsDrawCommand &)
                    {
                        ++failedSoftwareCalls;
                    },
                    {}, nullptr);
            t.IsNotNull(failing.get(),
                        "an initially healthy failing linear executor should construct");
            bool executionThrew = false;
            if (failing)
            {
                try
                {
                    failing->submit(
                        std::span<const GsDrawCommand>(&command, 1u));
                }
                catch (const std::runtime_error &error)
                {
                    executionThrew =
                        std::string(error.what()).find(
                            "before canonical VRAM mutation") !=
                        std::string::npos;
                }
                t.Equals(failing->backendStatistics().gpuRequestsFailed,
                         1ull,
                         "linear executor failure should be counted once");
                t.Equals(
                    failing->serviceStatistics()
                        .linearCt32SpriteDrawsFailed,
                    1ull,
                    "the failed linear request should retain its class counter");
            }
            t.IsTrue(executionThrew,
                     "linear executor failure should identify its atomic boundary");
            t.IsTrue(failureVram == initial,
                     "linear executor failure must preserve canonical VRAM");
            t.Equals(failedSoftwareCalls, 0ull,
                     "linear executor failure must precede the software oracle");

            ScopedArtifactDirectory artifacts;
            std::vector<uint8_t> mismatchVram = initial;
            GsVulkanRasterBackendConfig mismatchConfig = config;
            mismatchConfig.verificationArtifactDirectory =
                artifacts.path.string();
            std::unique_ptr<GsVulkanRasterBackend> mismatchBackend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Noop),
                    mismatchConfig, mismatchVram,
                    [&](const GsDrawCommand &draw)
                    {
                        GsVulkanLinearCt32Sprite sprite{};
                        if (prepareGsVulkanLinearCt32Sprite(
                                draw, sprite).supported)
                        {
                            applyLinearCt32SpriteCpu(
                                mismatchVram, sprite);
                        }
                    },
                    {}, nullptr);
            t.IsNotNull(mismatchBackend.get(),
                        "the linear mismatch backend should construct");
            bool mismatchThrew = false;
            if (mismatchBackend)
            {
                try
                {
                    mismatchBackend->submit(
                        std::span<const GsDrawCommand>(&command, 1u));
                }
                catch (const std::runtime_error &)
                {
                    mismatchThrew = true;
                }
                const GsVulkanRasterBackendStatistics mismatchStatistics =
                    mismatchBackend->backendStatistics();
                t.Equals(mismatchStatistics.verificationMismatches,
                         1ull,
                         "the injected no-op linear result should disagree once");
                const std::filesystem::path bundle =
                    mismatchStatistics.lastVerificationArtifact;
                t.IsTrue(std::filesystem::is_directory(bundle),
                         "the linear reproducer should be published atomically");
                std::ifstream manifest(bundle / "command.json");
                const std::string manifestText{
                    std::istreambuf_iterator<char>(manifest),
                    std::istreambuf_iterator<char>()};
                t.IsTrue(
                    manifestText.find("\"linear_ct32_sprite\"") !=
                        std::string::npos,
                    "the manifest should identify the linear record");
                t.IsTrue(
                    manifestText.find(
                        "\"fixed_lane_u\":[0,65536,131072,196608,262144,327680,393216,458752]") !=
                        std::string::npos,
                    "the manifest should retain all eight prepared U lanes");
                t.IsTrue(
                    manifestText.find(
                        "\"fixed_scan_v_bits\":" +
                        std::to_string(prepared.fixedScanVBits)) !=
                        std::string::npos,
                    "the manifest should retain the exact binary32 V seed");
                t.IsTrue(
                    manifestText.find(
                        "\"fixed_step_v_bits\":" +
                        std::to_string(prepared.fixedStepVBits)) !=
                        std::string::npos,
                    "the manifest should retain the exact binary32 V step");
            }
            t.IsTrue(mismatchThrew,
                     "the injected no-op linear result should stop Verify");
            t.IsTrue(mismatchVram == expected,
                     "the software oracle should remain canonical on mismatch");
        });

        tc.Run("Vulkan raster backend verifies exact CT32 depth sprites behind capability", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const std::array<GsDrawCommand, 2> commands{{
                makeDepthCt32SpriteCommand(
                    30'100u, 40u, 2u, 200u, GS_PSM_Z24, false, 1u,
                    {3u, 30u, 2u, 25u}, {16u, 32u},
                    49u, 65u, 449u, 353u,
                    0x88776655u, 0xFEDCBA98u),
                makeDepthCt32SpriteCommand(
                    30'101u, 41u, 2u, 201u, GS_PSM_Z32, true, 2u,
                    {4u, 27u, 3u, 22u}, {0u, 0u},
                    65u, 49u, 433u, 337u,
                    0x10203040u, 0x7FFFFF00u),
            }};
            std::array<GsVulkanDepthCt32Sprite, 2> prepared{};
            for (size_t index = 0u; index < commands.size(); ++index)
            {
                t.IsTrue(
                    prepareGsVulkanDepthCt32Sprite(
                        commands[index], prepared[index]).supported,
                    "the Verify depth fixture should satisfy its narrow predicate");
            }

            std::vector<uint8_t> initial =
                makeVramPattern(0x44565246u);
            for (uint32_t y = prepared[1].boundsY0;
                 y < prepared[1].boundsY1; ++y)
            {
                for (uint32_t x = prepared[1].boundsX0;
                     x < prepared[1].boundsX1; ++x)
                {
                    const uint32_t selector = (x + y) % 3u;
                    const uint32_t current = selector == 0u
                        ? prepared[1].depth - 1u
                        : selector == 1u
                            ? prepared[1].depth
                            : prepared[1].depth + 1u;
                    GSMem::WriteZ32(
                        initial.data(), prepared[1].depthBaseBlock,
                        prepared[1].framebufferWidth, x, y, current);
                }
            }
            std::vector<uint8_t> expected = initial;
            for (const GsVulkanDepthCt32Sprite &sprite : prepared)
                applyDepthCt32SpriteCpu(expected, sprite);

            std::vector<uint8_t> vram = initial;
            uint64_t softwareCalls = 0u;
            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::Verify;
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram,
                    [&](const GsDrawCommand &draw)
                    {
                        ++softwareCalls;
                        GsVulkanDepthCt32Sprite sprite{};
                        if (prepareGsVulkanDepthCt32Sprite(
                                draw, sprite).supported)
                        {
                            applyDepthCt32SpriteCpu(vram, sprite);
                        }
                    },
                    {}, nullptr);
            t.IsNotNull(backend.get(),
                        "an exact depth executor should create Verify");
            if (!backend)
                return;

            for (const GsDrawCommand &command : commands)
            {
                const GsBackendDecision route = backend->classify(command);
                t.IsTrue(route.supported,
                         "Verify should expose capability-gated depth sprites");
                if (!route.supported)
                    return;
            }
            backend->submit(commands);
            t.IsTrue(vram == expected,
                     "depth Verify should retain the agreed software image");
            t.Equals(softwareCalls, 2ull,
                     "depth Verify should run each independent oracle once");
            const GsVulkanRasterBackendStatistics statistics =
                backend->backendStatistics();
            t.Equals(statistics.commandsAttempted, 2ull,
                     "depth Verify should attempt both commands");
            t.Equals(statistics.commandsCompleted, 2ull,
                     "agreeing depth commands should complete once each");
            t.Equals(statistics.verifiedCommands, 2ull,
                     "both full depth comparisons should be counted");
            t.Equals(statistics.bytesCompared,
                     2ull * GS_VULKAN_VRAM_SIZE,
                     "depth Verify should compare two complete VRAM images");
            const GsVulkanServiceStatistics serviceStatistics =
                backend->serviceStatistics();
            t.Equals(serviceStatistics.depthCt32SpriteDrawsCompleted,
                     2ull,
                     "the executor should receive two depth requests");
            t.Equals(serviceStatistics.spriteDrawsCompleted, 0ull,
                     "depth Verify must not alias the no-depth request");

            t.IsTrue(backend->setMode(GsRendererMode::GpuStrict),
                     "the synchronized Verify fixture should enter strict mode");
            t.IsTrue(backend->classify(commands.front()).supported,
                     "strict should expose the resident-qualified depth class");
            t.IsTrue(backend->setMode(GsRendererMode::Hybrid),
                     "the fixture should also enter Hybrid cleanly");
            t.IsFalse(backend->classify(commands.front()).supported,
                      "Hybrid routing should remain closed before measurement");
            t.IsTrue(backend->setMode(GsRendererMode::Verify),
                     "the fixture should restore Verify for failure checks");

            std::vector<uint8_t> unavailableVram = initial;
            std::unique_ptr<GsVulkanRasterBackend> unavailable =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact,
                        true, true, true, false),
                    config, unavailableVram,
                    [](const GsDrawCommand &) {}, {}, nullptr);
            t.IsNotNull(unavailable.get(),
                        "a base-capable executor should retain depth fallback");
            if (unavailable)
            {
                t.Equals(unavailable->classify(commands.front()).reason,
                         GsFallbackReason::BackendUnavailable,
                         "a missing exact depth capability should fail closed");
                t.Equals(
                    unavailable->serviceStatistics()
                        .depthCt32SpriteDrawsFailed,
                    0ull,
                    "capability fallback must not post depth work");
            }

            std::vector<uint8_t> failureVram = initial;
            uint64_t failedSoftwareCalls = 0u;
            std::unique_ptr<GsVulkanRasterBackend> failing =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Fail),
                    config, failureVram,
                    [&](const GsDrawCommand &)
                    {
                        ++failedSoftwareCalls;
                    },
                    {}, nullptr);
            t.IsNotNull(failing.get(),
                        "an initially healthy failing depth executor should construct");
            bool executionThrew = false;
            if (failing)
            {
                try
                {
                    failing->submit(std::span<const GsDrawCommand>(
                        &commands.front(), 1u));
                }
                catch (const std::runtime_error &error)
                {
                    executionThrew =
                        std::string(error.what()).find(
                            "before canonical VRAM mutation") !=
                        std::string::npos;
                }
                t.Equals(failing->backendStatistics().gpuRequestsFailed,
                         1ull,
                         "depth executor failure should be counted once");
                t.Equals(
                    failing->serviceStatistics()
                        .depthCt32SpriteDrawsFailed,
                    1ull,
                    "the failed depth request should retain its class counter");
            }
            t.IsTrue(executionThrew,
                     "depth executor failure should identify its atomic boundary");
            t.IsTrue(failureVram == initial,
                     "depth executor failure must preserve canonical VRAM");
            t.Equals(failedSoftwareCalls, 0ull,
                     "depth executor failure must precede the software oracle");

            ScopedArtifactDirectory artifacts;
            std::vector<uint8_t> mismatchVram = initial;
            std::vector<uint8_t> mismatchExpected = initial;
            applyDepthCt32SpriteCpu(mismatchExpected, prepared.front());
            GsVulkanRasterBackendConfig mismatchConfig = config;
            mismatchConfig.verificationArtifactDirectory =
                artifacts.path.string();
            std::unique_ptr<GsVulkanRasterBackend> mismatchBackend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Noop),
                    mismatchConfig, mismatchVram,
                    [&](const GsDrawCommand &draw)
                    {
                        GsVulkanDepthCt32Sprite sprite{};
                        if (prepareGsVulkanDepthCt32Sprite(
                                draw, sprite).supported)
                        {
                            applyDepthCt32SpriteCpu(
                                mismatchVram, sprite);
                        }
                    },
                    {}, nullptr);
            t.IsNotNull(mismatchBackend.get(),
                        "the depth mismatch backend should construct");
            bool mismatchThrew = false;
            if (mismatchBackend)
            {
                try
                {
                    mismatchBackend->submit(std::span<const GsDrawCommand>(
                        &commands.front(), 1u));
                }
                catch (const std::runtime_error &)
                {
                    mismatchThrew = true;
                }
                const GsVulkanRasterBackendStatistics mismatchStatistics =
                    mismatchBackend->backendStatistics();
                t.Equals(mismatchStatistics.verificationMismatches,
                         1ull,
                         "the injected no-op depth result should disagree once");
                const std::filesystem::path bundle =
                    mismatchStatistics.lastVerificationArtifact;
                t.IsTrue(std::filesystem::is_directory(bundle),
                         "the depth reproducer should be published atomically");
                std::ifstream manifest(bundle / "command.json");
                const std::string manifestText{
                    std::istreambuf_iterator<char>(manifest),
                    std::istreambuf_iterator<char>()};
                t.IsTrue(
                    manifestText.find("\"depth_ct32_sprite\"") !=
                        std::string::npos,
                    "the manifest should identify the depth record");
                t.IsTrue(
                    manifestText.find(
                        "\"depth_psm\":" +
                        std::to_string(prepared.front().depthPsm)) !=
                        std::string::npos,
                    "the manifest should retain the exact depth format");
                t.IsTrue(
                    manifestText.find(
                        "\"depth\":" +
                        std::to_string(prepared.front().depth)) !=
                        std::string::npos,
                    "the manifest should retain the integer Z payload");
                t.IsTrue(
                    manifestText.find("\"depth_test_method\":1") !=
                        std::string::npos &&
                    manifestText.find("\"depth_write\":1") !=
                        std::string::npos,
                    "the manifest should retain ALWAYS plus write state");
            }
            t.IsTrue(mismatchThrew,
                     "the injected no-op depth result should stop Verify");
            t.IsTrue(mismatchVram == mismatchExpected,
                     "the software depth oracle should remain canonical on mismatch");
        });

        tc.Run("Vulkan strict backend keeps ordered depth CT32 sprites resident", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const std::array<GsDrawCommand, 6> commands =
                makeOrderedDepthCt32SpriteCommands(39'000u);
            std::array<GsVulkanDepthCt32Sprite, 6> prepared{};
            for (size_t index = 0u; index < commands.size(); ++index)
            {
                t.IsTrue(
                    prepareGsVulkanDepthCt32Sprite(
                        commands[index], prepared[index]).supported,
                    "the strict depth fixture should satisfy its narrow predicate");
            }

            const std::vector<uint8_t> initial =
                makeVramPattern(0x44535431u);
            std::vector<uint8_t> expected = initial;
            uint64_t expectedPixels = 0u;
            for (const GsVulkanDepthCt32Sprite &sprite : prepared)
            {
                applyDepthCt32SpriteCpu(expected, sprite);
                expectedPixels +=
                    static_cast<uint64_t>(
                        sprite.boundsX1 - sprite.boundsX0) *
                    static_cast<uint64_t>(
                        sprite.boundsY1 - sprite.boundsY0);
            }

            std::vector<uint8_t> vram = initial;
            uint64_t softwareCalls = 0u;
            uint64_t commitCalls = 0u;
            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::GpuStrict;
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram,
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
                        "an exact depth executor should create strict mode");
            if (!backend)
                return;

            const GsBackendDecision firstDecision =
                backend->classify(commands.front());
            t.IsTrue(firstDecision.supported,
                     "strict should expose the capability-gated depth class");
            if (!firstDecision.supported)
                return;
            for (const GsDrawCommand &command :
                 std::span<const GsDrawCommand>(commands).subspan(1u))
            {
                t.IsTrue(backend->classify(command).supported,
                         "strict should accept every ordered depth record");
            }

            backend->submit(commands);
            t.IsTrue(vram == initial,
                     "queued strict depth draws should not mutate canonical VRAM");
            t.Equals(softwareCalls, 0ull,
                     "strict depth execution must not call the software oracle");
            t.Equals(commitCalls, 0ull,
                     "pending strict depth draws must not publish commit metadata");
            t.Equals(backend->pendingCommandCount(), commands.size(),
                     "dependent and disjoint depth draws should share one ordered queue");

            GsVramPageMask allPages;
            allPages.setAll();
            backend->prepareCpuVramAccess(
                allPages, GsFlushReason::CpuReadback);
            t.IsTrue(vram == expected,
                     "strict depth observation should publish the exact resident result");
            t.Equals(softwareCalls, 0ull,
                     "strict depth publication must not invoke software fallback");
            t.Equals(
                commitCalls,
                static_cast<uint64_t>(commands.size()),
                "draining strict depth work should publish every commit");
            t.Equals(backend->pendingCommandCount(), size_t{0u},
                     "the observation boundary should drain resident depth work");

            const GsVulkanRasterBackendStatistics firstStatistics =
                backend->backendStatistics();
            t.Equals(
                firstStatistics.commandsCompleted,
                static_cast<uint64_t>(commands.size()),
                "strict should complete every ordered depth draw");
            t.Equals(
                firstStatistics.committedGpuCommands,
                static_cast<uint64_t>(commands.size()),
                "strict should commit every ordered depth result");
            t.Equals(
                firstStatistics.residentCommands,
                static_cast<uint64_t>(commands.size()),
                "strict should count every resident depth draw");
            t.Equals(firstStatistics.residentBatchesCompleted, 1ull,
                     "overlapping depth draws should share one backend batch");
            t.Equals(firstStatistics.resourceHazardDrains, 0ull,
                     "the ordered depth service should absorb resource dependencies");
            t.Equals(
                firstStatistics.largestResidentBatch,
                static_cast<uint64_t>(commands.size()),
                "the backend should retain the ordered depth batch size");
            const GsVulkanServiceStatistics firstServiceStatistics =
                backend->serviceStatistics();
            t.Equals(
                firstServiceStatistics.depthCt32SpriteDrawsCompleted,
                static_cast<uint64_t>(commands.size()),
                "the resident executor should complete every depth draw");
            t.Equals(
                firstServiceStatistics.depthCt32SpritePixelsExecuted,
                expectedPixels,
                "the resident executor should retain depth pixel accounting");
            t.Equals(
                firstServiceStatistics
                    .residentDepthCt32SpriteBatchesCompleted,
                1ull,
                "strict should execute one resident depth service batch");
            t.Equals(
                firstServiceStatistics
                    .largestResidentDepthCt32SpriteBatch,
                static_cast<uint64_t>(commands.size()),
                "the service should retain the ordered depth batch size");

            const GsDrawCommand flat = makeCt32SpriteCommand(
                39'010u, 43u, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                17u, 17u, 257u, 257u, 0x0BADF00Du);
            GsVulkanCt32Sprite flatPrepared{};
            t.IsTrue(prepareGsVulkanCt32Sprite(
                         flat, flatPrepared).supported,
                     "the mixed-pipeline fixture should retain one flat draw");
            std::vector<uint8_t> mixedExpected = expected;
            applyDepthCt32SpriteCpu(mixedExpected, prepared[0]);
            applyCt32SpriteCpu(mixedExpected, flatPrepared);
            applyDepthCt32SpriteCpu(mixedExpected, prepared[2]);

            backend->submit(std::span<const GsDrawCommand>(
                &commands[0], 1u));
            backend->submit(std::span<const GsDrawCommand>(&flat, 1u));
            backend->submit(std::span<const GsDrawCommand>(
                &commands[2], 1u));
            t.Equals(backend->pendingCommandCount(), size_t{1u},
                     "depth-flat-depth transitions should leave the final draw pending");
            backend->prepareCpuVramAccess(
                allPages, GsFlushReason::DebuggerObservation);
            t.IsTrue(vram == mixedExpected,
                     "mixed strict pipeline transitions should preserve guest order");
            t.Equals(commitCalls, 9ull,
                     "mixed pipeline drains should publish each commit once");

            const GsVulkanRasterBackendStatistics mixedStatistics =
                backend->backendStatistics();
            t.Equals(mixedStatistics.commandsCompleted, 9ull,
                     "the strict backend should complete both depth groups and flat draw");
            t.Equals(mixedStatistics.residentBatchesCompleted, 4ull,
                     "the initial depth batch and three mixed groups should complete");
            t.Equals(mixedStatistics.pipelineChangeDrains, 2ull,
                     "depth-flat-depth should cross two pipeline boundaries");
            t.Equals(mixedStatistics.coherency.rejectedTransitions, 0ull,
                     "strict depth routing should preserve page ownership");
            t.Equals(mixedStatistics.pageOwnership.gpuNewerPages, size_t{0u},
                     "debug observation should publish every resident depth writer");
            const GsVulkanServiceStatistics mixedServiceStatistics =
                backend->serviceStatistics();
            t.Equals(mixedServiceStatistics.depthCt32SpriteDrawsCompleted,
                     8ull,
                     "the service should execute both follow-up depth draws");
            t.Equals(
                mixedServiceStatistics
                    .residentDepthCt32SpriteBatchesCompleted,
                3ull,
                "the service should retain three resident depth batches");
            t.Equals(mixedServiceStatistics.spriteDrawsCompleted, 1ull,
                     "the mixed transition should execute one flat sprite");

            t.IsTrue(backend->setMode(GsRendererMode::Hybrid),
                     "a synchronized strict depth backend should enter Hybrid");
            t.Equals(backend->classify(commands.front()).reason,
                     GsFallbackReason::CostModel,
                     "small strict fixtures should stay on the CPU in Hybrid");

            std::vector<uint8_t> unavailableVram = initial;
            std::unique_ptr<GsVulkanRasterBackend> unavailable =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact,
                        true, true, true, false),
                    config, unavailableVram,
                    [](const GsDrawCommand &) {}, {}, nullptr);
            t.IsNotNull(unavailable.get(),
                        "a base-capable executor should retain strict depth fallback");
            if (unavailable)
            {
                t.Equals(unavailable->classify(commands.front()).reason,
                         GsFallbackReason::BackendUnavailable,
                         "missing depth capability should reject before queueing");
                t.Equals(unavailable->pendingCommandCount(), size_t{0u},
                         "capability fallback must not retain depth work");
                t.Equals(
                    unavailable->serviceStatistics()
                        .residentDepthCt32SpriteBatchesFailed,
                    0ull,
                    "capability fallback must not post a resident depth batch");
            }

            std::vector<uint8_t> failureVram = initial;
            uint64_t failedSoftwareCalls = 0u;
            std::unique_ptr<GsVulkanRasterBackend> failing =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::FailResidentDraw),
                    config, failureVram,
                    [&](const GsDrawCommand &)
                    {
                        ++failedSoftwareCalls;
                    },
                    {}, nullptr);
            t.IsNotNull(failing.get(),
                        "an initially healthy failing depth executor should construct");
            bool executionThrew = false;
            if (failing)
            {
                failing->submit(std::span<const GsDrawCommand>(
                    &commands.front(), 1u));
                t.Equals(failing->pendingCommandCount(), size_t{1u},
                         "a failing strict depth draw should first enter the queue");
                try
                {
                    failing->flush(GsFlushReason::Explicit);
                }
                catch (const std::runtime_error &error)
                {
                    executionThrew =
                        std::string(error.what()).find(
                            "before canonical VRAM mutation") !=
                        std::string::npos;
                }
                t.Equals(failing->backendStatistics().gpuRequestsFailed,
                         1ull,
                         "resident depth execution failure should be counted once");
                t.Equals(
                    failing->serviceStatistics()
                        .residentDepthCt32SpriteBatchesFailed,
                    1ull,
                    "the deferred depth failure should count one resident batch");
            }
            t.IsTrue(executionThrew,
                     "strict depth failure should identify its atomic boundary");
            t.IsTrue(failureVram == initial,
                     "strict depth failure must preserve canonical VRAM");
            t.Equals(failedSoftwareCalls, 0ull,
                     "strict depth failure must not invoke software fallback");
        });

        tc.Run("Vulkan Hybrid admits measured depth CT32 work", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            constexpr uint64_t thresholdPixels = 262'144u;
            const GsDrawCommand belowThreshold =
                makeDepthCt32SpriteCommand(
                    39'099u, 0u, 8u, 256u,
                    GS_PSM_Z24, false, 1u,
                    {0u, 511u, 0u, 510u}, {0u, 0u},
                    0u, 0u, 512u * 16u, 511u * 16u,
                    0x40302010u, 0x00444444u);
            const GsDrawCommand atThreshold = makeDepthCt32SpriteCommand(
                39'100u, 0u, 8u, 256u,
                GS_PSM_Z24, false, 1u,
                {0u, 511u, 0u, 511u}, {0u, 0u},
                0u, 0u, 512u * 16u, 512u * 16u,
                0x80706050u, 0x00555555u);
            const GsDrawCommand retainedTitle =
                makeDepthCt32SpriteCommand(
                    39'101u, 112u, 8u, 216u,
                    GS_PSM_Z24, false, 1u,
                    {0u, 511u, 0u, 415u},
                    {1792u * 16u, 1840u * 16u},
                    1792u * 16u - 8u, 1840u * 16u - 8u,
                    (1792u + 32u) * 16u - 8u,
                    (1840u + 416u) * 16u - 8u,
                    0x80000000u, 0u);
            const std::array<GsDrawCommand, 5> admittedStates{{
                atThreshold,
                makeDepthCt32SpriteCommand(
                    39'102u, 0u, 8u, 256u,
                    GS_PSM_Z32, false, 1u,
                    {0u, 511u, 0u, 511u}, {0u, 0u},
                    0u, 0u, 512u * 16u, 512u * 16u,
                    0x90807060u, 0x55555555u),
                makeDepthCt32SpriteCommand(
                    39'103u, 0u, 8u, 256u,
                    GS_PSM_Z24, true, 2u,
                    {0u, 511u, 0u, 511u}, {0u, 0u},
                    0u, 0u, 512u * 16u, 512u * 16u,
                    0xA0908070u, 0x007FFFFFu),
                makeDepthCt32SpriteCommand(
                    39'104u, 0u, 8u, 256u,
                    GS_PSM_Z32, true, 3u,
                    {0u, 511u, 0u, 511u}, {0u, 0u},
                    0u, 0u, 512u * 16u, 512u * 16u,
                    0xB0A09080u, 0x7FFFFFFFu),
                makeDepthCt32SpriteCommand(
                    39'105u, 0u, 8u, 256u,
                    GS_PSM_Z24, false, 2u,
                    {0u, 511u, 0u, 511u}, {0u, 0u},
                    0u, 0u, 512u * 16u, 512u * 16u,
                    0xC0B0A090u, 0x007FFFFFu),
            }};
            std::array<GsVulkanDepthCt32Sprite, 5> prepared{};
            for (size_t index = 0u; index < admittedStates.size(); ++index)
            {
                t.IsTrue(
                    prepareGsVulkanDepthCt32Sprite(
                        admittedStates[index], prepared[index]).supported,
                    "every measured depth state should remain exact");
            }
            GsVulkanDepthCt32Sprite belowPrepared{};
            GsVulkanDepthCt32Sprite retainedPrepared{};
            t.IsTrue(prepareGsVulkanDepthCt32Sprite(
                         belowThreshold, belowPrepared).supported,
                     "the below-threshold depth fixture should be semantic");
            t.IsTrue(prepareGsVulkanDepthCt32Sprite(
                         retainedTitle, retainedPrepared).supported,
                     "the retained title depth fixture should be semantic");
            const auto pixels = [](const GsVulkanDepthCt32Sprite &sprite)
            {
                return static_cast<uint64_t>(
                           sprite.boundsX1 - sprite.boundsX0) *
                       static_cast<uint64_t>(
                           sprite.boundsY1 - sprite.boundsY0);
            };
            t.Equals(pixels(belowPrepared), 261'632ull,
                     "the smaller fixture should be one row below policy");
            t.Equals(pixels(prepared.front()), thresholdPixels,
                     "the admitted fixture should meet policy exactly");
            t.Equals(pixels(retainedPrepared), 13'312ull,
                     "the retained title fixture should preserve exact work");

            const std::vector<uint8_t> initial =
                makeVramPattern(0x4448434Fu);
            std::vector<uint8_t> expected = initial;
            applyDepthCt32SpriteCpu(expected, prepared.front());
            std::vector<uint8_t> vram = initial;
            uint64_t softwareCalls = 0u;
            uint64_t commitCalls = 0u;
            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::Hybrid;
            t.Equals(config.minimumHybridDepthCt32SpritePixels,
                     thresholdPixels,
                     "depth Hybrid should retain the measured default");
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram,
                    [&](const GsDrawCommand &) { ++softwareCalls; },
                    [&](const GsDrawCommand &) { ++commitCalls; }, nullptr);
            t.IsNotNull(backend.get(),
                        "an exact executor should create depth Hybrid mode");
            if (!backend)
                return;
            t.Equals(backend->classify(belowThreshold).reason,
                     GsFallbackReason::CostModel,
                     "depth work below the measured envelope should use CPU");
            t.Equals(backend->classify(retainedTitle).reason,
                     GsFallbackReason::CostModel,
                     "the retained 13312-pixel title draw should stay on CPU");
            for (const GsDrawCommand &command : admittedStates)
            {
                t.IsTrue(backend->classify(command).supported,
                         "every exact state at the threshold should use Vulkan");
            }
            t.IsTrue(backend->setMode(GsRendererMode::GpuStrict),
                     "the depth policy fixture should enter strict mode");
            t.IsTrue(backend->classify(belowThreshold).supported,
                     "strict mode should ignore the depth cost policy");
            t.IsTrue(backend->setMode(GsRendererMode::Verify),
                     "the depth policy fixture should enter Verify");
            t.IsTrue(backend->classify(retainedTitle).supported,
                     "Verify should exercise the retained depth candidate");
            t.IsTrue(backend->setMode(GsRendererMode::Hybrid),
                     "the depth policy fixture should restore Hybrid");

            backend->submit(
                std::span<const GsDrawCommand>(&atThreshold, 1u));
            t.Equals(backend->pendingCommandCount(), size_t{1u},
                     "the admitted depth draw should remain resident");
            backend->flush(GsFlushReason::Explicit);
            backend->prepareCpuVramAccess(
                atThreshold.resources().writePages,
                GsFlushReason::CpuReadback);
            t.IsTrue(vram == expected,
                     "the admitted Hybrid depth draw should remain exact");
            t.Equals(softwareCalls, 0ull,
                     "admitted Hybrid depth must not call its oracle");
            t.Equals(commitCalls, 1ull,
                     "admitted Hybrid depth should commit once");
            const GsVulkanServiceStatistics service =
                backend->serviceStatistics();
            t.Equals(service.depthCt32SpriteDrawsCompleted, 1ull,
                     "the admitted draw should reach the depth executor");
            t.Equals(service.residentDepthCt32SpriteBatchesCompleted, 1ull,
                     "the admitted draw should use resident depth execution");

            std::vector<uint8_t> unavailableVram = initial;
            std::unique_ptr<GsVulkanRasterBackend> unavailable =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact,
                        true, true, true, false),
                    config, unavailableVram,
                    [](const GsDrawCommand &) {}, {}, nullptr);
            t.IsNotNull(unavailable.get(),
                        "a base-capable executor should retain depth fallback");
            if (unavailable)
            {
                t.Equals(unavailable->classify(belowThreshold).reason,
                         GsFallbackReason::BackendUnavailable,
                         "missing depth capability should precede cost policy");
                t.Equals(
                    unavailable->serviceStatistics()
                        .residentDepthCt32SpriteBatchesFailed,
                    0ull,
                    "capability fallback must not post depth work");
            }

            GsVulkanRasterBackendConfig disabledConfig = config;
            disabledConfig.minimumHybridDepthCt32SpritePixels = 0u;
            std::vector<uint8_t> disabledVram = initial;
            std::unique_ptr<GsVulkanRasterBackend> disabled =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    disabledConfig, disabledVram,
                    [](const GsDrawCommand &) {}, {}, nullptr);
            t.IsNotNull(disabled.get(),
                        "a zero-threshold depth Hybrid should construct");
            if (disabled)
            {
                t.IsTrue(disabled->classify(retainedTitle).supported,
                         "zero should disable only the depth cost policy");
            }
        });

        tc.Run("Vulkan strict backend keeps linear CT32 textures resident", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const auto makeLinear = [](uint64_t sequence,
                                       uint32_t framebufferPage,
                                       uint8_t wrapU = 0u,
                                       uint8_t wrapV = 0u)
            {
                return makeLinearCt32SpriteCommand(
                    sequence, framebufferPage, 2u,
                    512u, 2u, 6u, 5u,
                    {3u, 12u, 2u, 13u}, {128u, 96u},
                    {120u, 376u}, {88u, 344u},
                    {0u, 249u}, {0u, 137u}, wrapU, wrapV);
            };
            const std::array<GsDrawCommand, 4> commands{{
                makeLinear(101u, 40u),
                makeLinear(102u, 41u),
                makeLinear(103u, 42u, 1u, 0u),
                makeLinear(104u, 43u, 1u, 1u),
            }};
            std::array<GsVulkanLinearCt32Sprite, 4> prepared{};
            for (size_t index = 0u; index < commands.size(); ++index)
            {
                t.IsTrue(
                    prepareGsVulkanLinearCt32Sprite(
                        commands[index], prepared[index]).supported,
                    "the strict linear fixture should satisfy its narrow predicate");
            }

            const std::vector<uint8_t> initial =
                makeVramPattern(0x4C535431u);
            std::vector<uint8_t> expected = initial;
            for (const GsVulkanLinearCt32Sprite &sprite : prepared)
                applyLinearCt32SpriteCpu(expected, sprite);

            std::vector<uint8_t> vram = initial;
            uint64_t softwareCalls = 0u;
            uint64_t commitCalls = 0u;
            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::GpuStrict;
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram,
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
                        "an exact linear executor should create strict mode");
            if (!backend)
                return;

            for (const GsDrawCommand &command : commands)
            {
                t.IsTrue(backend->classify(command).supported,
                         "strict should expose the capability-gated linear class");
            }
            backend->submit(commands);
            t.IsTrue(vram == initial,
                     "queued strict linear draws should not mutate canonical VRAM");
            t.Equals(softwareCalls, 0ull,
                     "strict linear execution must not call the software oracle");
            t.Equals(commitCalls, 0ull,
                     "pending strict linear draws must not publish commit metadata");
            t.Equals(backend->pendingCommandCount(), commands.size(),
                     "shared-read disjoint linear draws should remain in one queue");

            GsVramPageMask allPages;
            allPages.setAll();
            backend->prepareCpuVramAccess(
                allPages, GsFlushReason::CpuReadback);
            t.IsTrue(vram == expected,
                     "strict linear observation should publish the exact resident result");
            t.Equals(softwareCalls, 0ull,
                     "strict linear publication must not invoke software fallback");
            t.Equals(
                commitCalls,
                static_cast<uint64_t>(commands.size()),
                "draining strict linear work should publish every commit");
            t.Equals(backend->pendingCommandCount(), size_t{0u},
                     "the observation boundary should drain resident linear work");

            const GsVulkanRasterBackendStatistics statistics =
                backend->backendStatistics();
            t.Equals(
                statistics.commandsAttempted,
                static_cast<uint64_t>(commands.size()),
                "strict should attempt every linear draw");
            t.Equals(
                statistics.commandsCompleted,
                static_cast<uint64_t>(commands.size()),
                "strict should complete every linear draw");
            t.Equals(
                statistics.committedGpuCommands,
                static_cast<uint64_t>(commands.size()),
                "strict should commit every GPU result");
            t.Equals(
                statistics.residentCommands,
                static_cast<uint64_t>(commands.size()),
                "strict should count every resident linear draw");
            t.Equals(statistics.residentBatchesCompleted, 1ull,
                     "the mixed-wrap linear draws should share one backend batch");
            t.Equals(
                statistics.largestResidentBatch,
                static_cast<uint64_t>(commands.size()),
                "the backend should retain the mixed-wrap linear batch size");
            t.Equals(statistics.gpuRequestsFailed, 0ull,
                     "the exact strict linear batch should not fail");
            const GsVulkanServiceStatistics serviceStatistics =
                backend->serviceStatistics();
            t.Equals(
                serviceStatistics.linearCt32SpriteDrawsCompleted,
                static_cast<uint64_t>(commands.size()),
                "the resident executor should complete every linear draw");
            t.Equals(
                serviceStatistics.residentLinearCt32SpriteBatchesCompleted,
                1ull,
                "strict should execute one resident linear service batch");
            t.Equals(
                serviceStatistics.largestResidentLinearCt32SpriteBatch,
                static_cast<uint64_t>(commands.size()),
                "the service should retain the mixed-wrap linear batch size");
            t.Equals(serviceStatistics.nearestCt32SpriteDrawsCompleted,
                     0ull,
                     "strict linear routing must not alias nearest textures");

            t.IsTrue(backend->setMode(GsRendererMode::Hybrid),
                     "a synchronized strict backend should enter Hybrid");
            t.IsFalse(backend->classify(commands.front()).supported,
                      "Hybrid should keep the small repeat fixture on software");
            t.Equals(
                backend->classify(commands.back()).reason,
                GsFallbackReason::CostModel,
                "Hybrid should keep the small clamp fixture on software");

            std::vector<uint8_t> unavailableVram = initial;
            std::unique_ptr<GsVulkanRasterBackend> unavailable =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact,
                        true, true, false),
                    config, unavailableVram,
                    [](const GsDrawCommand &) {}, {}, nullptr);
            t.IsNotNull(unavailable.get(),
                        "a base-capable executor should retain strict linear fallback");
            if (unavailable)
            {
                t.Equals(unavailable->classify(commands.front()).reason,
                         GsFallbackReason::BackendUnavailable,
                         "missing linear capability should reject before queueing");
                t.Equals(unavailable->pendingCommandCount(), size_t{0u},
                         "capability fallback must not retain linear work");
                t.Equals(
                    unavailable->serviceStatistics()
                        .residentLinearCt32SpriteBatchesFailed,
                    0ull,
                    "capability fallback must not post a resident batch");
            }

            std::vector<uint8_t> failureVram = initial;
            uint64_t failedSoftwareCalls = 0u;
            std::unique_ptr<GsVulkanRasterBackend> failing =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::FailResidentDraw),
                    config, failureVram,
                    [&](const GsDrawCommand &)
                    {
                        ++failedSoftwareCalls;
                    },
                    {}, nullptr);
            t.IsNotNull(failing.get(),
                        "an initially healthy failing linear executor should construct");
            bool executionThrew = false;
            if (failing)
            {
                failing->submit(
                    std::span<const GsDrawCommand>(
                        &commands.front(), 1u));
                t.Equals(failing->pendingCommandCount(), size_t{1u},
                         "a failing strict linear draw should first enter the queue");
                try
                {
                    failing->flush(GsFlushReason::Explicit);
                }
                catch (const std::runtime_error &error)
                {
                    executionThrew =
                        std::string(error.what()).find(
                            "before canonical VRAM mutation") !=
                        std::string::npos;
                }
                t.Equals(failing->backendStatistics().gpuRequestsFailed,
                         1ull,
                         "resident linear execution failure should be counted once");
                t.Equals(
                    failing->serviceStatistics()
                        .residentLinearCt32SpriteBatchesFailed,
                    1ull,
                    "the deferred linear failure should count one resident batch");
            }
            t.IsTrue(executionThrew,
                     "strict linear failure should identify its atomic boundary");
            t.IsTrue(failureVram == initial,
                     "strict linear failure must preserve canonical VRAM");
            t.Equals(failedSoftwareCalls, 0ull,
                     "strict linear failure must not invoke software fallback");
        });

        tc.Run("Vulkan strict texture queue shares reads and splits dependencies", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const auto makeTexture =
                [](uint64_t sequence,
                   uint32_t framebufferPage,
                   uint32_t texturePage)
            {
                return makeNearestCt32SpriteCommand(
                    sequence, framebufferPage, 2u,
                    texturePage * 32u, 2u, 6u, 5u,
                    {0u, 31u, 0u, 31u}, {0u, 0u},
                    {0u, 8u * 16u}, {0u, 8u * 16u},
                    {0u, 8u * 16u}, {0u, 8u * 16u});
            };
            const std::array<GsDrawCommand, 5> commands{{
                makeTexture(44u, 40u, 2u),
                makeTexture(45u, 41u, 2u),
                makeTexture(46u, 42u, 40u),
                makeTexture(47u, 40u, 60u),
                makeTexture(48u, 40u, 61u),
            }};

            std::vector<uint8_t> vram = makeVramPattern(0x54515231u);
            const std::vector<uint8_t> initial = vram;
            std::vector<uint8_t> expected = initial;
            for (size_t index = 0u; index < commands.size(); ++index)
            {
                GsVulkanNearestCt32Sprite prepared{};
                const GsBackendDecision decision =
                    prepareGsVulkanNearestCt32Sprite(
                        commands[index], prepared);
                if (!decision.supported)
                {
                    t.Fail(
                        "strict texture dependency fixture " +
                        std::to_string(index) + " was rejected as " +
                        std::string(gsFallbackReasonName(decision.reason)));
                    return;
                }
                const GsDrawResources resources =
                    commands[index].resources();
                t.Equals(resources.readPages.count(), size_t{1u},
                         "each dependency fixture should read one physical page");
                t.Equals(resources.writePages.count(), size_t{1u},
                         "each dependency fixture should write one physical page");
                t.IsFalse(
                    resources.readPages.intersects(resources.writePages),
                    "each dependency fixture must be individually non-aliasing");
                applyNearestCt32SpriteCpu(expected, prepared);
            }

            uint64_t softwareCalls = 0u;
            uint64_t commitCalls = 0u;
            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::GpuStrict;
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram,
                    [&](const GsDrawCommand &) { ++softwareCalls; },
                    [&](const GsDrawCommand &) { ++commitCalls; },
                    nullptr);
            t.IsNotNull(backend.get(),
                        "an exact executor should create strict texture residency");
            if (!backend)
                return;

            backend->submit(
                std::span<const GsDrawCommand>(&commands[0], 1u));
            backend->submit(
                std::span<const GsDrawCommand>(&commands[1], 1u));
            t.Equals(backend->pendingCommandCount(), size_t{2u},
                     "two shared-source reads should remain in one resident batch");
            t.Equals(commitCalls, 0ull,
                     "shared-read queueing should not execute the batch early");

            backend->submit(
                std::span<const GsDrawCommand>(&commands[2], 1u));
            t.Equals(commitCalls, 2ull,
                     "a write-then-read dependency should drain the shared prefix");
            t.Equals(backend->pendingCommandCount(), size_t{1u},
                     "the dependent reader should begin a new batch");

            backend->submit(
                std::span<const GsDrawCommand>(&commands[3], 1u));
            t.Equals(commitCalls, 3ull,
                     "a read-then-write dependency should drain its reader");
            t.Equals(backend->pendingCommandCount(), size_t{1u},
                     "the dependent writer should begin a new batch");

            backend->submit(
                std::span<const GsDrawCommand>(&commands[4], 1u));
            t.Equals(commitCalls, 4ull,
                     "a write/write dependency should drain the earlier writer");
            t.Equals(backend->pendingCommandCount(), size_t{1u},
                     "the final writer should remain pending");
            t.IsTrue(vram == initial,
                     "dependency drains should retain every result on resident VRAM");
            t.Equals(softwareCalls, 0ull,
                     "strict texture dependencies must not invoke software fallback");

            GsVulkanRasterBackendStatistics statistics =
                backend->backendStatistics();
            t.Equals(statistics.resourceHazardDrains, 3ull,
                     "all three unordered dependency directions should drain once");
            t.Equals(statistics.pipelineChangeDrains, 0ull,
                     "homogeneous texture dependencies are not pipeline changes");
            t.Equals(statistics.residentBatchesCompleted, 3ull,
                     "three dependency drains should complete three prefixes");
            t.Equals(statistics.largestResidentBatch, 2ull,
                     "the shared-read prefix should establish the peak batch size");

            backend->flush(GsFlushReason::Explicit);
            t.Equals(commitCalls, 5ull,
                     "the explicit boundary should execute the final writer");
            t.Equals(backend->pendingCommandCount(), size_t{0u},
                     "the explicit boundary should empty the texture queue");
            GsVramPageMask allPages;
            allPages.setAll();
            backend->prepareCpuVramAccess(
                allPages, GsFlushReason::DebuggerObservation);
            t.IsTrue(vram == expected,
                     "hazard-split resident textures should match sequential CPU execution");

            statistics = backend->backendStatistics();
            t.Equals(statistics.commandsAttempted, 5ull,
                     "every texture dependency fixture should be attempted once");
            t.Equals(statistics.commandsCompleted, 5ull,
                     "every texture dependency fixture should complete once");
            t.Equals(statistics.committedGpuCommands, 5ull,
                     "every drained texture should publish one commit");
            t.Equals(statistics.residentCommands, 5ull,
                     "all dependency fixtures should use resident execution");
            t.Equals(statistics.residentBatchesCompleted, 4ull,
                     "the three hazard prefixes and final tail should form four batches");
            t.Equals(statistics.pageOwnership.gpuNewerPages, size_t{0u},
                     "the final observation should publish every texture writer");
            t.Equals(statistics.coherency.rejectedTransitions, 0ull,
                     "hazard splitting should preserve single-writer ownership");

            const GsVulkanServiceStatistics service =
                backend->serviceStatistics();
            t.Equals(service.nearestCt32SpriteDrawsCompleted, 5ull,
                     "the service should execute every resident texture once");
            t.Equals(service.residentNearestCt32SpriteBatchesCompleted,
                     4ull,
                     "the service should receive each hazard-split texture batch");
            t.Equals(service.largestResidentNearestCt32SpriteBatch, 2ull,
                     "the service should retain the shared-read prefix size");
            t.Equals(service.pageUploadOperationsCompleted, 4ull,
                     "each resident texture batch should upload its CPU-newer inputs");
            t.Equals(service.pagesUploaded, 6ull,
                     "texture residency should upload only first-use source and destination pages");
            t.Equals(service.pageDownloadOperationsCompleted, 1ull,
                     "one final observation should download resident texture outputs");
            t.Equals(service.pagesDownloaded, 3ull,
                     "only the three distinct texture destinations should download");
            t.Equals(service.spriteDrawsCompleted, 0ull,
                     "resident textures must not alias flat-sprite accounting");
            t.Equals(service.triangleDrawsCompleted, 0ull,
                     "resident textures must not alias triangle accounting");
        });

        tc.Run("Vulkan raster backend routes exact triangles behind capability", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const GsDrawCommand command = makeCt32TriangleCommand(
                43u, 9u, 2u,
                {0u, 63u, 0u, 63u}, {0u, 0u},
                {49u, 335u, 103u},
                {83u, 139u, 309u},
                0xD4C3B2A1u);
            GsVulkanCt32Triangle prepared{};
            t.IsTrue(
                prepareGsVulkanCt32Triangle(command, prepared).supported,
                "the verification fixture should satisfy the triangle predicate");

            std::vector<uint8_t> vram = makeVramPattern(0x54525631u);
            const std::vector<uint8_t> initial = vram;
            std::vector<uint8_t> expected = initial;
            applyCt32TriangleCpu(expected, prepared);
            uint64_t softwareCalls = 0u;
            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::Verify;
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram,
                    [&](const GsDrawCommand &draw)
                    {
                        ++softwareCalls;
                        GsVulkanCt32Triangle triangle{};
                        if (prepareGsVulkanCt32Triangle(
                                draw, triangle).supported)
                        {
                            applyCt32TriangleCpu(vram, triangle);
                        }
                    },
                    {}, nullptr);
            t.IsNotNull(backend.get(),
                        "an exact triangle executor should create Verify");
            if (!backend)
                return;

            t.IsTrue(backend->classify(command).supported,
                     "Verify should expose the capability-gated triangle class");
            backend->submit(
                std::span<const GsDrawCommand>(&command, 1u));
            t.IsTrue(vram == expected,
                     "Verify should retain the agreed independent CPU image");
            t.Equals(softwareCalls, 1ull,
                     "triangle Verify should run the software oracle once");
            const GsVulkanRasterBackendStatistics statistics =
                backend->backendStatistics();
            t.Equals(statistics.commandsAttempted, 1ull,
                     "triangle Verify should attempt one command");
            t.Equals(statistics.commandsCompleted, 1ull,
                     "an agreeing triangle should complete once");
            t.Equals(statistics.verifiedCommands, 1ull,
                     "the full triangle comparison should be counted");
            t.Equals(statistics.bytesCompared,
                     static_cast<uint64_t>(GS_VULKAN_VRAM_SIZE),
                     "triangle Verify should compare all 4 MiB");
            const GsVulkanServiceStatistics serviceStatistics =
                backend->serviceStatistics();
            t.Equals(serviceStatistics.triangleDrawsCompleted, 1ull,
                     "the executor should receive one triangle request");
            t.Equals(serviceStatistics.spriteDrawsCompleted, 0ull,
                     "triangle Verify must not use the sprite kernel");

            t.IsTrue(backend->setMode(GsRendererMode::Hybrid),
                     "the synchronized backend should accept Hybrid mode");
            t.Equals(backend->classify(command).reason,
                     GsFallbackReason::CostModel,
                     "Hybrid should retain a small qualified triangle on the CPU");
            t.IsTrue(backend->setMode(GsRendererMode::GpuStrict),
                     "the synchronized backend should accept strict mode");
            t.IsTrue(backend->classify(command).supported,
                     "strict mode should expose the exhaustively qualified class");
            backend->submit(
                std::span<const GsDrawCommand>(&command, 1u));
            t.Equals(backend->pendingCommandCount(), size_t{1u},
                     "strict triangles should enter the resident queue");
            backend->flush(GsFlushReason::Explicit);
            t.Equals(backend->pendingCommandCount(), size_t{0u},
                     "an explicit checkpoint should drain the triangle queue");
            backend->prepareCpuVramAccess(
                command.resources().writePages,
                GsFlushReason::CpuReadback);
            t.IsTrue(vram == expected,
                     "strict resident execution should preserve exact VRAM");
            t.Equals(softwareCalls, 1ull,
                     "strict triangle execution must not call the software oracle");
            const GsVulkanRasterBackendStatistics strictStatistics =
                backend->backendStatistics();
            t.Equals(strictStatistics.committedGpuCommands, 1ull,
                     "the strict triangle should commit one resident command");
            t.Equals(strictStatistics.residentBatchesCompleted, 1ull,
                     "the strict triangle should complete one resident batch");
            const GsVulkanServiceStatistics strictService =
                backend->serviceStatistics();
            t.Equals(strictService.triangleDrawsCompleted, 2ull,
                     "Verify and strict should use the triangle pipeline once each");
            t.Equals(strictService.residentTriangleBatchesCompleted, 1ull,
                     "strict should use the resident triangle request");

            std::vector<uint8_t> unavailableVram = initial;
            std::unique_ptr<GsVulkanRasterBackend> unavailable =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact, false),
                    config, unavailableVram,
                    [](const GsDrawCommand &) {}, {}, nullptr);
            t.IsNotNull(unavailable.get(),
                        "a base-capable executor should remain generally usable");
            if (unavailable)
            {
                t.IsTrue(unavailable->setMode(GsRendererMode::Hybrid),
                         "the base service should still enter Hybrid mode");
                t.Equals(unavailable->classify(command).reason,
                         GsFallbackReason::BackendUnavailable,
                         "Hybrid should name capability failure before cost");
                t.IsTrue(unavailable->setMode(GsRendererMode::GpuStrict),
                         "the base service should still enter strict mode");
                t.Equals(unavailable->classify(command).reason,
                         GsFallbackReason::BackendUnavailable,
                         "strict should name the missing triangle capability");
                t.IsTrue(unavailableVram == initial,
                         "capability classification must not mutate VRAM");
                t.Equals(
                    unavailable->serviceStatistics().triangleDrawsFailed,
                    0ull,
                    "capability fallback must not post triangle work");
            }
        });

        tc.Run("Vulkan strict backend splits resident pipeline classes", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const GsDrawCommand sprite = makeCt32SpriteCommand(
                51u, 5u, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                2u * 16u, 3u * 16u,
                12u * 16u, 14u * 16u,
                0x44332211u);
            const GsDrawCommand firstTriangle = makeCt32TriangleCommand(
                52u, 41u, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                {3u * 16u + 1u, 24u * 16u + 7u, 7u * 16u + 3u},
                {4u * 16u + 5u, 9u * 16u + 1u, 26u * 16u + 9u},
                0x88776655u);
            const GsDrawCommand secondTriangle = makeCt32TriangleCommand(
                53u, 197u, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                {5u * 16u + 2u, 27u * 16u + 6u, 11u * 16u + 14u},
                {2u * 16u + 3u, 12u * 16u + 11u, 28u * 16u + 4u},
                0xCCBBAA99u);

            GsVulkanCt32Sprite preparedSprite{};
            GsVulkanCt32Triangle preparedFirst{};
            GsVulkanCt32Triangle preparedSecond{};
            t.IsTrue(prepareGsVulkanCt32Sprite(
                         sprite, preparedSprite).supported,
                     "the strict sprite fixture should be eligible");
            t.IsTrue(prepareGsVulkanCt32Triangle(
                         firstTriangle, preparedFirst).supported,
                     "the first strict triangle fixture should be eligible");
            t.IsTrue(prepareGsVulkanCt32Triangle(
                         secondTriangle, preparedSecond).supported,
                     "the second strict triangle fixture should be eligible");

            std::vector<uint8_t> vram = makeVramPattern(0x53545231u);
            std::vector<uint8_t> expected = vram;
            applyCt32SpriteCpu(expected, preparedSprite);
            applyCt32TriangleCpu(expected, preparedFirst);
            applyCt32TriangleCpu(expected, preparedSecond);
            uint64_t softwareCalls = 0u;
            uint64_t committedCalls = 0u;
            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::GpuStrict;
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram,
                    [&](const GsDrawCommand &) { ++softwareCalls; },
                    [&](const GsDrawCommand &) { ++committedCalls; },
                    nullptr);
            t.IsNotNull(backend.get(),
                        "an exact executor should create strict mixed routing");
            if (!backend)
                return;

            backend->submit(
                std::span<const GsDrawCommand>(&sprite, 1u));
            t.Equals(backend->pendingCommandCount(), size_t{1u},
                     "the sprite should begin one homogeneous batch");
            backend->submit(
                std::span<const GsDrawCommand>(&firstTriangle, 1u));
            t.Equals(backend->pendingCommandCount(), size_t{1u},
                     "a pipeline change should drain then queue the triangle");
            backend->submit(
                std::span<const GsDrawCommand>(&secondTriangle, 1u));
            t.Equals(backend->pendingCommandCount(), size_t{2u},
                     "same-pipeline disjoint triangles should batch together");
            backend->flush(GsFlushReason::Explicit);
            GsVramPageMask affected = sprite.resources().writePages;
            affected.unionWith(firstTriangle.resources().writePages);
            affected.unionWith(secondTriangle.resources().writePages);
            backend->prepareCpuVramAccess(
                affected, GsFlushReason::CpuReadback);

            t.IsTrue(vram == expected,
                     "split strict batches should match sequential CPU records");
            t.Equals(softwareCalls, 0ull,
                     "strict mixed routing must not call the software oracle");
            t.Equals(committedCalls, 3ull,
                     "every completed resident draw should publish its commit callback");
            const GsVulkanRasterBackendStatistics statistics =
                backend->backendStatistics();
            t.Equals(statistics.pipelineChangeDrains, 1ull,
                     "sprite-to-triangle should retain its explicit drain reason");
            t.Equals(statistics.resourceHazardDrains, 0ull,
                     "disjoint class changes should not be mislabeled as hazards");
            t.Equals(statistics.residentBatchesCompleted, 2ull,
                     "the two homogeneous classes should use two backend batches");
            t.Equals(statistics.largestResidentBatch, 2ull,
                     "the triangle suffix should establish the peak batch size");
            const GsVulkanServiceStatistics service =
                backend->serviceStatistics();
            t.Equals(service.residentSpriteBatchesCompleted, 1ull,
                     "the sprite prefix should use one sprite service batch");
            t.Equals(service.residentTriangleBatchesCompleted, 1ull,
                     "the triangle suffix should use one triangle service batch");
            t.Equals(service.spriteDrawsCompleted, 1ull,
                     "the sprite pipeline should execute one draw");
            t.Equals(service.triangleDrawsCompleted, 2ull,
                     "the triangle pipeline should execute two draws");
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

        tc.Run("Vulkan hybrid source-copy alpha policy uses measured work", [](TestCase &t)
        {
            const auto sourceCopy = [](const GsDrawCommand &opaque,
                                       uint64_t sequence,
                                       uint64_t alpha = 0u)
            {
                GSPrimReg primitive = opaque.primitive();
                primitive.abe = true;
                GSContext context = opaque.context();
                context.alpha = alpha;
                return buildGsDrawCommand(
                    sequence,
                    primitive,
                    context,
                    std::span<const GSVertex>(opaque.vertices()).first(2u),
                    opaque.globalState());
            };
            const GsDrawCommand below = sourceCopy(
                makeCt32SpriteCommand(
                    350u, 4u, 1u,
                    {0u, 63u, 0u, 63u}, {0u, 0u},
                    0u, 0u, 64u * 16u, 64u * 16u,
                    0x20000000u),
                351u);
            const GsDrawCommand threshold = sourceCopy(
                makeCt32SpriteCommand(
                    352u, 8u, 2u,
                    {0u, 127u, 0u, 63u}, {0u, 0u},
                    0u, 0u, 128u * 16u, 64u * 16u,
                    0x40000000u),
                353u);
            const GsDrawCommand sourceOver = sourceCopy(
                threshold, 354u, 0x44u);

            std::vector<uint8_t> vram = makeVramPattern(0x53434150u);
            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::Hybrid;
            t.Equals(
                config.minimumHybridSourceCopyAlphaSpritePixels,
                8'192ull,
                "source-copy alpha should retain the measured default floor");
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config,
                    vram,
                    [](const GsDrawCommand &) {},
                    {},
                    nullptr);
            t.IsNotNull(
                backend.get(),
                "the source-copy policy fixture should construct");
            if (!backend)
                return;

            t.Equals(
                backend->classify(below).reason,
                GsFallbackReason::CostModel,
                "4,096 source-copy pixels should remain below the crossover");
            t.IsTrue(
                backend->classify(threshold).supported,
                "8,192 source-copy pixels should meet the crossover exactly");
            t.Equals(
                backend->classify(sourceOver).reason,
                GsFallbackReason::AlphaBlend,
                "the policy must not turn source-over into a semantic candidate");

            t.IsTrue(
                backend->setMode(GsRendererMode::GpuStrict),
                "the source-copy fixture should enter strict mode");
            t.IsTrue(
                backend->classify(below).supported,
                "strict mode should ignore the source-copy cost gate");
            t.IsTrue(
                backend->setMode(GsRendererMode::Verify),
                "the source-copy fixture should enter Verify mode");
            t.IsTrue(
                backend->classify(below).supported,
                "Verify should exercise every source-copy candidate");

            config.minimumHybridSourceCopyAlphaSpritePixels = 0u;
            std::unique_ptr<GsVulkanRasterBackend> disabled =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config,
                    vram,
                    [](const GsDrawCommand &) {},
                    {},
                    nullptr);
            t.IsNotNull(
                disabled.get(),
                "a disabled source-copy cost gate should construct");
            if (disabled)
            {
                t.IsTrue(
                    disabled->classify(below).supported,
                    "zero should disable the source-copy Hybrid floor");
            }
        });

        tc.Run("Vulkan hybrid nearest CT32 cost policy uses sample bounds", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            constexpr uint64_t thresholdPixels = 8'192u;
            const GsDrawCommand belowThreshold =
                makeNearestCt32SpriteCommand(
                    303u, 100u, 2u, 64u, 1u, 6u, 5u,
                    {0u, 127u, 0u, 63u}, {0u, 0u},
                    {0u, 127u * 16u}, {0u, 64u * 16u},
                    {0u, 127u * 16u}, {0u, 64u * 16u});
            const GsDrawCommand atThreshold =
                makeNearestCt32SpriteCommand(
                    304u, 110u, 2u, 64u, 1u, 6u, 5u,
                    {0u, 127u, 0u, 63u}, {0u, 0u},
                    {0u, 128u * 16u}, {0u, 64u * 16u},
                    {0u, 128u * 16u}, {0u, 64u * 16u},
                    3u, 3u, 15u, 16u, 15u, 16u);

            GsVulkanNearestCt32Sprite belowPrepared{};
            GsVulkanNearestCt32Sprite thresholdPrepared{};
            t.IsTrue(prepareGsVulkanNearestCt32Sprite(
                         belowThreshold, belowPrepared).supported,
                     "the below-threshold texture should be semantically eligible");
            t.IsTrue(prepareGsVulkanNearestCt32Sprite(
                         atThreshold, thresholdPrepared).supported,
                     "the threshold texture should be semantically eligible");
            const auto samplePixels = [](
                const GsVulkanNearestCt32Sprite &sprite)
            {
                return
                    static_cast<uint64_t>(
                        sprite.boundsX1 - sprite.boundsX0) *
                    static_cast<uint64_t>(
                        sprite.boundsY1 - sprite.boundsY0);
            };
            t.Equals(samplePixels(belowPrepared), 8'128ull,
                     "the smaller texture should be 64 samples short");
            t.Equals(samplePixels(thresholdPrepared), thresholdPixels,
                     "the larger texture should meet the policy exactly");
            t.Equals(gsVulkanTextureWrapMode(
                         thresholdPrepared.textureWrapU), 3u,
                     "Hybrid cost routing should preserve REGION_REPEAT U");
            t.Equals(gsVulkanTextureWrapMode(
                         thresholdPrepared.textureWrapV), 3u,
                     "Hybrid cost routing should preserve REGION_REPEAT V");

            std::vector<uint8_t> vram = makeVramPattern(0x5458434Fu);
            std::vector<uint8_t> expected = vram;
            applyNearestCt32SpriteCpu(expected, thresholdPrepared);
            uint64_t softwareCalls = 0u;
            uint64_t commitCalls = 0u;
            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::Hybrid;
            t.Equals(config.minimumHybridNearestCt32SpritePixels,
                     thresholdPixels,
                     "the measured nearest-texture threshold should be the default");
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram,
                    [&](const GsDrawCommand &) { ++softwareCalls; },
                    [&](const GsDrawCommand &) { ++commitCalls; }, nullptr);
            t.IsNotNull(backend.get(),
                        "an exact executor should create Hybrid texture policy");
            if (!backend)
                return;

            t.Equals(backend->classify(belowThreshold).reason,
                     GsFallbackReason::CostModel,
                     "a texture below the measured sample count should stay on the CPU");
            t.IsTrue(backend->classify(atThreshold).supported,
                     "a texture exactly at the measured sample count should use the GPU");
            t.IsTrue(backend->setMode(GsRendererMode::GpuStrict),
                     "the fixture should switch to strict mode");
            t.IsTrue(backend->classify(belowThreshold).supported,
                     "strict mode should ignore the texture cost policy");
            t.IsTrue(backend->setMode(GsRendererMode::Verify),
                     "the fixture should switch to Verify mode");
            t.IsTrue(backend->classify(belowThreshold).supported,
                     "Verify should exercise every semantic texture candidate");
            t.IsTrue(backend->setMode(GsRendererMode::Hybrid),
                     "the fixture should restore Hybrid policy routing");

            backend->submit(
                std::span<const GsDrawCommand>(&atThreshold, 1u));
            t.Equals(backend->pendingCommandCount(), size_t{1u},
                     "the threshold texture should remain resident until observation");
            backend->flush(GsFlushReason::Explicit);
            backend->prepareCpuVramAccess(
                atThreshold.resources().writePages,
                GsFlushReason::CpuReadback);
            t.IsTrue(vram == expected,
                     "the accepted threshold texture should remain byte-exact");
            t.Equals(softwareCalls, 0ull,
                     "an accepted Hybrid texture must not call the oracle");
            t.Equals(commitCalls, 1ull,
                     "the accepted Hybrid texture should publish one commit");
            const GsVulkanServiceStatistics service =
                backend->serviceStatistics();
            t.Equals(service.nearestCt32SpriteDrawsCompleted, 1ull,
                     "the accepted Hybrid texture should reach the executor once");
            t.Equals(service.residentNearestCt32SpriteBatchesCompleted, 1ull,
                     "the accepted Hybrid texture should use one resident batch");

            std::vector<uint8_t> unavailableVram =
                makeVramPattern(0x54584355u);
            std::unique_ptr<GsVulkanRasterBackend> unavailable =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact, true, false),
                    config, unavailableVram,
                    [](const GsDrawCommand &) {}, {}, nullptr);
            t.IsNotNull(unavailable.get(),
                        "a base-capable executor should retain Hybrid texture fallback");
            if (unavailable)
            {
                t.Equals(unavailable->classify(belowThreshold).reason,
                         GsFallbackReason::BackendUnavailable,
                         "missing exact texture coverage should take priority over cost");
                t.Equals(
                    unavailable->serviceStatistics()
                        .nearestCt32SpriteDrawsFailed,
                    0ull,
                    "capability fallback must not post texture work");
            }

            GsVulkanRasterBackendConfig disabledConfig = config;
            disabledConfig.minimumHybridNearestCt32SpritePixels = 0u;
            std::vector<uint8_t> disabledVram =
                makeVramPattern(0x5458435Au);
            std::unique_ptr<GsVulkanRasterBackend> disabled =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    disabledConfig, disabledVram,
                    [](const GsDrawCommand &) {}, {}, nullptr);
            t.IsNotNull(disabled.get(),
                        "a zero-threshold Hybrid texture backend should construct");
            if (disabled)
            {
                t.IsTrue(disabled->classify(belowThreshold).supported,
                         "zero should disable the Hybrid texture cost policy");
            }
        });

        tc.Run("Vulkan hybrid linear CT32 cost policy uses exact work", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            constexpr uint64_t thresholdPixels = 131'072u;
            constexpr uint32_t width = 136u;
            const auto makeCommand = [](uint64_t sequence, uint32_t height)
            {
                return makeLinearCt32RepeatSpriteCommand(
                    sequence, 0u, 3u, 400u * 32u, 1u, 6u, 5u,
                    {0u, static_cast<uint16_t>(width - 1u),
                     0u, static_cast<uint16_t>(height - 1u)},
                    {0u, 0u},
                    {0u, static_cast<uint16_t>(width * 16u)},
                    {0u, static_cast<uint16_t>(height * 16u)},
                    {3u, static_cast<uint16_t>(width * 16u + 3u)},
                    {5u, static_cast<uint16_t>(height * 16u + 5u)});
            };
            const GsDrawCommand belowThreshold = makeCommand(305u, 963u);
            const GsDrawCommand admitted = makeCommand(306u, 964u);

            std::array<GsVulkanLinearCt32Sprite, 2> prepared{};
            const std::array<const GsDrawCommand *, 2> commands{{
                &belowThreshold, &admitted,
            }};
            for (size_t index = 0u; index < commands.size(); ++index)
            {
                t.IsTrue(
                    prepareGsVulkanLinearCt32Sprite(
                        *commands[index], prepared[index]).supported,
                    "every linear cost fixture should be semantically eligible");
            }
            const auto pixels = [](const GsVulkanLinearCt32Sprite &sprite)
            {
                return static_cast<uint64_t>(
                           sprite.boundsX1 - sprite.boundsX0) *
                       static_cast<uint64_t>(
                           sprite.boundsY1 - sprite.boundsY0);
            };
            t.Equals(pixels(prepared[0]), 130'968ull,
                     "the smaller fixture should be 104 pixels below policy");
            t.Equals(pixels(prepared[1]), 131'104ull,
                     "the first integral row above policy should be admitted");

            std::vector<uint8_t> vram = makeVramPattern(0x4C48434Fu);
            std::vector<uint8_t> expected = vram;
            applyLinearCt32SpriteCpu(expected, prepared[1]);
            uint64_t softwareCalls = 0u;
            uint64_t commitCalls = 0u;
            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::Hybrid;
            t.Equals(config.minimumHybridLinearCt32SpritePixels,
                     thresholdPixels,
                     "the measured linear work threshold should be the default");
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram,
                    [&](const GsDrawCommand &) { ++softwareCalls; },
                    [&](const GsDrawCommand &) { ++commitCalls; }, nullptr);
            t.IsNotNull(backend.get(),
                        "an exact executor should create Hybrid linear policy");
            if (!backend)
                return;

            t.Equals(backend->classify(belowThreshold).reason,
                     GsFallbackReason::CostModel,
                     "insufficient linear work should stay on the CPU");
            t.IsTrue(backend->classify(admitted).supported,
                     "the measured linear envelope should use the GPU");
            t.IsTrue(backend->setMode(GsRendererMode::GpuStrict),
                     "the fixture should switch to strict mode");
            t.IsTrue(backend->classify(belowThreshold).supported,
                     "strict mode should ignore linear cost policy");
            t.IsTrue(backend->setMode(GsRendererMode::Verify),
                     "the fixture should switch to Verify mode");
            t.IsTrue(backend->classify(belowThreshold).supported,
                     "Verify should exercise every semantic linear candidate");
            t.IsTrue(backend->setMode(GsRendererMode::Hybrid),
                     "the fixture should restore Hybrid linear routing");

            backend->submit(std::span<const GsDrawCommand>(&admitted, 1u));
            t.Equals(backend->pendingCommandCount(), size_t{1u},
                     "the admitted linear draw should remain resident");
            backend->flush(GsFlushReason::Explicit);
            backend->prepareCpuVramAccess(
                admitted.resources().writePages,
                GsFlushReason::CpuReadback);
            t.IsTrue(vram == expected,
                     "the admitted Hybrid linear draw should remain byte-exact");
            t.Equals(softwareCalls, 0ull,
                     "an admitted Hybrid linear draw must not call the oracle");
            t.Equals(commitCalls, 1ull,
                     "the admitted Hybrid linear draw should commit once");
            const GsVulkanServiceStatistics service =
                backend->serviceStatistics();
            t.Equals(service.linearCt32SpriteDrawsCompleted, 1ull,
                     "the admitted Hybrid draw should reach the linear executor");
            t.Equals(service.residentLinearCt32SpriteBatchesCompleted, 1ull,
                     "the admitted Hybrid draw should use one resident batch");

            std::vector<uint8_t> unavailableVram =
                makeVramPattern(0x4C484355u);
            std::unique_ptr<GsVulkanRasterBackend> unavailable =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact,
                        true, true, false),
                    config, unavailableVram,
                    [](const GsDrawCommand &) {}, {}, nullptr);
            t.IsNotNull(unavailable.get(),
                        "a base-capable executor should retain linear fallback");
            if (unavailable)
            {
                t.Equals(unavailable->classify(belowThreshold).reason,
                         GsFallbackReason::BackendUnavailable,
                         "missing linear capability should precede cost policy");
                t.Equals(
                    unavailable->serviceStatistics()
                        .linearCt32SpriteDrawsFailed,
                    0ull,
                    "capability fallback must not post linear work");
            }

            GsVulkanRasterBackendConfig disabledConfig = config;
            disabledConfig.minimumHybridLinearCt32SpritePixels = 0u;
            std::vector<uint8_t> disabledVram =
                makeVramPattern(0x4C48435Au);
            std::unique_ptr<GsVulkanRasterBackend> disabled =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    disabledConfig, disabledVram,
                    [](const GsDrawCommand &) {}, {}, nullptr);
            t.IsNotNull(disabled.get(),
                        "a zero-threshold Hybrid linear backend should construct");
            if (disabled)
            {
                t.IsTrue(disabled->classify(belowThreshold).supported,
                         "zero should disable the Hybrid linear cost policy");
            }
        });

        tc.Run("Vulkan hybrid linear CT32 clamp uses its measured work", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            constexpr uint64_t thresholdPixels = 8'192u;
            constexpr uint32_t width = 32u;
            const auto makeCommand = [](
                uint64_t sequence, uint32_t height,
                uint8_t wrapU, uint8_t wrapV)
            {
                return makeLinearCt32SpriteCommand(
                    sequence, 0u, 1u, 400u * 32u, 1u, 6u, 5u,
                    {0u, static_cast<uint16_t>(width - 1u),
                     0u, static_cast<uint16_t>(height - 1u)},
                    {0u, 0u},
                    {0u, static_cast<uint16_t>(width * 16u)},
                    {0u, static_cast<uint16_t>(height * 16u)},
                    {3u, static_cast<uint16_t>(width * 16u + 3u)},
                    {5u, static_cast<uint16_t>(height * 16u + 5u)},
                    wrapU, wrapV);
            };
            const GsDrawCommand belowThreshold =
                makeCommand(307u, 255u, 1u, 1u);
            const GsDrawCommand admitted =
                makeCommand(308u, 256u, 1u, 1u);
            const GsDrawCommand repeatAtClampThreshold =
                makeCommand(309u, 256u, 0u, 0u);

            std::array<GsVulkanLinearCt32Sprite, 2> prepared{};
            const std::array<const GsDrawCommand *, 2> clampCommands{{
                &belowThreshold, &admitted,
            }};
            for (size_t index = 0u; index < clampCommands.size(); ++index)
            {
                t.IsTrue(
                    prepareGsVulkanLinearCt32Sprite(
                        *clampCommands[index], prepared[index]).supported,
                    "every clamp cost fixture should be semantically eligible");
            }
            const auto pixels = [](const GsVulkanLinearCt32Sprite &sprite)
            {
                return static_cast<uint64_t>(
                           sprite.boundsX1 - sprite.boundsX0) *
                       static_cast<uint64_t>(
                           sprite.boundsY1 - sprite.boundsY0);
            };
            t.Equals(pixels(prepared[0]), 8'160ull,
                     "the smaller clamp fixture should be one row below policy");
            t.Equals(pixels(prepared[1]), thresholdPixels,
                     "the admitted clamp fixture should land on policy");

            std::vector<uint8_t> vram = makeVramPattern(0x4C48434Cu);
            std::vector<uint8_t> expected = vram;
            applyLinearCt32SpriteCpu(expected, prepared[1]);
            uint64_t softwareCalls = 0u;
            uint64_t commitCalls = 0u;
            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::Hybrid;
            t.Equals(config.minimumHybridLinearCt32ClampSpritePixels,
                     thresholdPixels,
                     "standard clamp should retain its measured default");
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram,
                    [&](const GsDrawCommand &) { ++softwareCalls; },
                    [&](const GsDrawCommand &) { ++commitCalls; }, nullptr);
            t.IsNotNull(backend.get(),
                        "an exact executor should create Hybrid clamp policy");
            if (!backend)
                return;

            t.Equals(backend->classify(belowThreshold).reason,
                     GsFallbackReason::CostModel,
                     "insufficient clamp work should stay on the CPU");
            t.IsTrue(backend->classify(admitted).supported,
                     "the measured clamp envelope should use the GPU");
            t.Equals(backend->classify(repeatAtClampThreshold).reason,
                     GsFallbackReason::CostModel,
                     "clamp policy must not lower the REPEAT threshold");
            t.IsTrue(backend->setMode(GsRendererMode::GpuStrict),
                     "the clamp fixture should switch to strict mode");
            t.IsTrue(backend->classify(belowThreshold).supported,
                     "strict mode should ignore clamp cost policy");
            t.IsTrue(backend->setMode(GsRendererMode::Verify),
                     "the clamp fixture should switch to Verify mode");
            t.IsTrue(backend->classify(belowThreshold).supported,
                     "Verify should exercise every semantic clamp candidate");
            t.IsTrue(backend->setMode(GsRendererMode::Hybrid),
                     "the clamp fixture should restore Hybrid routing");

            backend->submit(std::span<const GsDrawCommand>(&admitted, 1u));
            t.Equals(backend->pendingCommandCount(), size_t{1u},
                     "the admitted clamp draw should remain resident");
            backend->flush(GsFlushReason::Explicit);
            backend->prepareCpuVramAccess(
                admitted.resources().writePages,
                GsFlushReason::CpuReadback);
            t.IsTrue(vram == expected,
                     "the admitted Hybrid clamp draw should remain byte-exact");
            t.Equals(softwareCalls, 0ull,
                     "an admitted Hybrid clamp draw must not call the oracle");
            t.Equals(commitCalls, 1ull,
                     "the admitted Hybrid clamp draw should commit once");
            const GsVulkanServiceStatistics service =
                backend->serviceStatistics();
            t.Equals(service.linearCt32SpriteDrawsCompleted, 1ull,
                     "the admitted clamp draw should reach the linear executor");
            t.Equals(service.residentLinearCt32SpriteBatchesCompleted, 1ull,
                     "the admitted clamp draw should use one resident batch");

            std::vector<uint8_t> unavailableVram =
                makeVramPattern(0x4C484343u);
            std::unique_ptr<GsVulkanRasterBackend> unavailable =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact,
                        true, true, false),
                    config, unavailableVram,
                    [](const GsDrawCommand &) {}, {}, nullptr);
            t.IsNotNull(unavailable.get(),
                        "a base-capable executor should retain clamp fallback");
            if (unavailable)
            {
                t.Equals(unavailable->classify(admitted).reason,
                         GsFallbackReason::BackendUnavailable,
                         "missing linear capability should precede clamp cost");
            }

            GsVulkanRasterBackendConfig disabledConfig = config;
            disabledConfig.minimumHybridLinearCt32ClampSpritePixels = 0u;
            std::vector<uint8_t> disabledVram =
                makeVramPattern(0x4C48435Cu);
            std::unique_ptr<GsVulkanRasterBackend> disabled =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    disabledConfig, disabledVram,
                    [](const GsDrawCommand &) {}, {}, nullptr);
            t.IsNotNull(disabled.get(),
                        "a zero-threshold Hybrid clamp backend should construct");
            if (disabled)
            {
                t.IsTrue(disabled->classify(belowThreshold).supported,
                         "zero should disable only the clamp cost policy");
            }
        });

        tc.Run("Vulkan hybrid triangle cost policy uses candidate bounds", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            constexpr uint64_t thresholdPixels = 32'768u;
            const GsDrawCommand belowThreshold = makeCt32TriangleCommand(
                303u, 27u, 4u,
                {0u, 255u, 0u, 127u}, {0u, 0u},
                {0u, 255u * 16u, 0u},
                {0u, 0u, 128u * 16u},
                0x44332211u);
            const GsDrawCommand atThreshold = makeCt32TriangleCommand(
                304u, 59u, 4u,
                {0u, 255u, 0u, 127u}, {0u, 0u},
                {0u, 256u * 16u, 0u},
                {0u, 0u, 128u * 16u},
                0x88776655u);

            GsVulkanCt32Triangle belowPrepared{};
            GsVulkanCt32Triangle thresholdPrepared{};
            t.IsTrue(prepareGsVulkanCt32Triangle(
                         belowThreshold, belowPrepared).supported,
                     "the below-threshold triangle should be semantically eligible");
            t.IsTrue(prepareGsVulkanCt32Triangle(
                         atThreshold, thresholdPrepared).supported,
                     "the threshold triangle should be semantically eligible");
            const auto candidatePixels = [](const GsVulkanCt32Triangle &triangle)
            {
                return
                    static_cast<uint64_t>(
                        triangle.boundsX1 - triangle.boundsX0) *
                    static_cast<uint64_t>(
                        triangle.boundsY1 - triangle.boundsY0);
            };
            t.Equals(candidatePixels(belowPrepared), 32'640ull,
                     "the smaller conservative box should be 128 pixels short");
            t.Equals(candidatePixels(thresholdPrepared), thresholdPixels,
                     "the larger conservative box should meet the policy exactly");

            std::vector<uint8_t> vram = makeVramPattern(0x5452434Fu);
            std::vector<uint8_t> expected = vram;
            applyCt32TriangleCpu(expected, thresholdPrepared);
            uint64_t softwareCalls = 0u;
            uint64_t commitCalls = 0u;
            GsVulkanRasterBackendConfig config{};
            config.mode = GsRendererMode::Hybrid;
            t.Equals(config.minimumHybridTriangleCandidatePixels,
                     thresholdPixels,
                     "the measured conservative threshold should be the default");
            std::unique_ptr<GsVulkanRasterBackend> backend =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact),
                    config, vram,
                    [&](const GsDrawCommand &) { ++softwareCalls; },
                    [&](const GsDrawCommand &) { ++commitCalls; }, nullptr);
            t.IsNotNull(backend.get(),
                        "an exact executor should create Hybrid triangle policy");
            if (!backend)
                return;

            t.Equals(backend->classify(belowThreshold).reason,
                     GsFallbackReason::CostModel,
                     "a triangle below the measured box should remain on the CPU");
            t.IsTrue(backend->classify(atThreshold).supported,
                     "a triangle exactly at the measured box should use the GPU");
            t.IsTrue(backend->setMode(GsRendererMode::GpuStrict),
                     "the fixture should switch to strict mode");
            t.IsTrue(backend->classify(belowThreshold).supported,
                     "strict mode should ignore the triangle cost policy");
            t.IsTrue(backend->setMode(GsRendererMode::Verify),
                     "the fixture should switch to Verify mode");
            t.IsTrue(backend->classify(belowThreshold).supported,
                     "Verify should exercise every semantic triangle candidate");
            t.IsTrue(backend->setMode(GsRendererMode::Hybrid),
                     "the fixture should restore Hybrid policy routing");

            backend->submit(
                std::span<const GsDrawCommand>(&atThreshold, 1u));
            t.Equals(backend->pendingCommandCount(), size_t{1u},
                     "the threshold triangle should remain resident until observation");
            backend->flush(GsFlushReason::Explicit);
            backend->prepareCpuVramAccess(
                atThreshold.resources().writePages,
                GsFlushReason::CpuReadback);
            t.IsTrue(vram == expected,
                     "the accepted threshold triangle should remain byte-exact");
            t.Equals(softwareCalls, 0ull,
                     "an accepted Hybrid triangle must not call the oracle");
            t.Equals(commitCalls, 1ull,
                     "the accepted Hybrid triangle should publish one commit");
            const GsVulkanServiceStatistics service =
                backend->serviceStatistics();
            t.Equals(service.triangleDrawsCompleted, 1ull,
                     "the accepted Hybrid triangle should reach the executor once");
            t.Equals(service.residentTriangleBatchesCompleted, 1ull,
                     "the accepted Hybrid triangle should use one resident batch");

            std::vector<uint8_t> unavailableVram =
                makeVramPattern(0x54524355u);
            std::unique_ptr<GsVulkanRasterBackend> unavailable =
                GsVulkanRasterBackend::createWithExecutor(
                    std::make_unique<FakeCt32Executor>(
                        FakeCt32Executor::Behavior::Exact, false),
                    config, unavailableVram,
                    [](const GsDrawCommand &) {}, {}, nullptr);
            t.IsNotNull(unavailable.get(),
                        "a base-capable executor should retain Hybrid fallback");
            if (unavailable)
            {
                t.Equals(unavailable->classify(atThreshold).reason,
                         GsFallbackReason::BackendUnavailable,
                         "missing exact coverage should take priority over cost");
                t.Equals(
                    unavailable->serviceStatistics().triangleDrawsFailed,
                    0ull,
                    "capability fallback must not post triangle work");
            }
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
                0ull,
                "qualified nearest feedback should no longer use generic texture fallback");
            t.Equals(
                counters.decisions[static_cast<size_t>(
                    GsFallbackReason::ResourceAlias)],
                8ull,
                "each shuffled feedback draw should name its source-destination alias");
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

        tc.Run("GS Vulkan strict geometry and texture transitions match every observation", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            std::vector<uint8_t> softwareVram =
                makeVramPattern(0x54525331u);
            std::vector<uint8_t> acceleratedVram = softwareVram;

            constexpr std::array<uint32_t, 8> pages{
                1u, 5u, 9u, 13u, 17u, 21u, 25u, 29u};
            const GsDrawCommand textureCommand =
                makeNearestCt32SpriteCommand(
                    80u, 40u, 2u, 64u, 2u, 6u, 5u,
                    {6u, 15u, 5u, 12u}, {32u, 16u},
                    {352u, 96u}, {48u, 304u},
                    {480u, 224u}, {64u, 320u}, 3u, 3u,
                    15u, 128u, 15u, 64u);
            GsVulkanNearestCt32Sprite preparedTexture{};
            t.IsTrue(
                prepareGsVulkanNearestCt32Sprite(
                    textureCommand, preparedTexture).supported,
                "the randomized strict REGION_REPEAT texture fixture should be eligible");
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
            t.IsTrue(accelerated.configureVulkanRenderer(
                         serviceConfig, backendConfig),
                     "the strict transition fixture should accept Vulkan configuration");
            if (!preflight.ready())
            {
                t.IsFalse(accelerated.setRendererMode(
                              GsRendererMode::GpuStrict),
                          "an unavailable host should skip strict transitions cleanly");
                return;
            }
            const GsVulkanDeviceReport *selected =
                preflight.selectedDevice();
            t.IsNotNull(selected,
                        "a ready strict transition probe should select one device");
            if (!selected)
                return;
            t.IsTrue(selected->exactCt32Triangle,
                     "the strict transition device should support exact triangles");
            t.IsTrue(selected->exactNearestCt32Sprite,
                     "the strict transition device should support exact textures");
            if (!selected->exactCt32Triangle ||
                !selected->exactNearestCt32Sprite)
                return;

            t.IsTrue(accelerated.setRendererMode(
                         GsRendererMode::GpuStrict),
                     "a qualified host should create the strict transition fixture");
            if (accelerated.rendererMode() !=
                GsRendererMode::GpuStrict)
            {
                return;
            }
            accelerated.setBackendCountersEnabled(true);
            accelerated.resetBackendCounters();

            uint32_t randomState = 0x13D4A7C9u;
            const auto randomBelow = [&](uint32_t bound)
            {
                return nextRandom(randomState) % bound;
            };
            const auto randomPageIndex = [&]()
            {
                return static_cast<size_t>(
                    randomBelow(static_cast<uint32_t>(pages.size())));
            };
            const auto distinctPageIndex = [&](size_t first)
            {
                return (first + 1u +
                        randomBelow(static_cast<uint32_t>(
                            pages.size() - 1u))) %
                       pages.size();
            };
            const auto drawTrianglePair = [&](uint32_t page)
            {
                const uint16_t baseX = static_cast<uint16_t>(
                    randomBelow(20u));
                const uint16_t baseY = static_cast<uint16_t>(
                    randomBelow(7u));
                const std::array<uint16_t, 3> x{{
                    static_cast<uint16_t>(
                        (baseX + 1u) * 16u + randomBelow(16u)),
                    static_cast<uint16_t>(
                        (baseX + 25u) * 16u + randomBelow(16u)),
                    static_cast<uint16_t>(
                        (baseX + 6u) * 16u + randomBelow(16u)),
                }};
                const std::array<uint16_t, 3> y{{
                    static_cast<uint16_t>(
                        (baseY + 1u) * 16u + randomBelow(16u)),
                    static_cast<uint16_t>(
                        (baseY + 6u) * 16u + randomBelow(16u)),
                    static_cast<uint16_t>(
                        (baseY + 23u) * 16u + randomBelow(16u)),
                }};
                const uint32_t color =
                    0x80000000u | nextRandom(randomState);
                configureFlatCt32Draws(software, page, 1u);
                configureFlatCt32Draws(accelerated, page, 1u);
                drawFlatCt32Triangle(software, x, y, color);
                drawFlatCt32Triangle(accelerated, x, y, color);
            };
            const auto drawSpritePair = [&](uint32_t page)
            {
                const uint16_t x0 = static_cast<uint16_t>(
                    randomBelow(32u));
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
            const auto drawTexturePair = [&]()
            {
                drawNearestCt32SpriteCommand(
                    software, textureCommand);
                drawNearestCt32SpriteCommand(
                    accelerated, textureCommand);
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
                    std::string("strict transition VRAM mismatch after ") +
                        boundary + " at operation " +
                        std::to_string(operationIndex) + ", byte " +
                        std::to_string(byteOffset) + ", page " +
                        std::to_string(
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
                            "strict transition frame diverged at operation " +
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

            enum class StrictTransitionAction : uint8_t
            {
                TriangleDraw,
                TriangleBatch,
                PipelineChange,
                HostUpload,
                CpuReadback,
                LocalCopy,
                SoftwareModeTriangle,
                TextureDraw,
                Observation,
                Count,
            };
            constexpr size_t actionCount =
                static_cast<size_t>(StrictTransitionAction::Count);
            std::array<uint64_t, actionCount> actionCounts{};
            constexpr size_t cycleCount = 8u;
            size_t operationIndex = 0u;
            for (size_t cycle = 0u; cycle < cycleCount; ++cycle)
            {
                std::array<StrictTransitionAction, actionCount> order{
                    StrictTransitionAction::TriangleDraw,
                    StrictTransitionAction::TriangleBatch,
                    StrictTransitionAction::PipelineChange,
                    StrictTransitionAction::HostUpload,
                    StrictTransitionAction::CpuReadback,
                    StrictTransitionAction::LocalCopy,
                    StrictTransitionAction::SoftwareModeTriangle,
                    StrictTransitionAction::TextureDraw,
                    StrictTransitionAction::Observation,
                };
                for (size_t index = order.size() - 1u;
                     index > 0u; --index)
                {
                    const size_t other = randomBelow(
                        static_cast<uint32_t>(index + 1u));
                    std::swap(order[index], order[other]);
                }

                for (StrictTransitionAction action : order)
                {
                    ++operationIndex;
                    ++actionCounts[static_cast<size_t>(action)];
                    const size_t firstPageIndex = randomPageIndex();
                    const size_t secondPageIndex =
                        distinctPageIndex(firstPageIndex);
                    const uint32_t page = pages[firstPageIndex];
                    switch (action)
                    {
                    case StrictTransitionAction::TriangleDraw:
                        drawTrianglePair(page);
                        break;
                    case StrictTransitionAction::TriangleBatch:
                        drawTrianglePair(page);
                        drawTrianglePair(pages[secondPageIndex]);
                        break;
                    case StrictTransitionAction::PipelineChange:
                        drawTrianglePair(page);
                        drawSpritePair(pages[secondPageIndex]);
                        break;
                    case StrictTransitionAction::HostUpload:
                    {
                        std::array<uint32_t, 4> pixels{};
                        for (uint32_t &pixel : pixels)
                            pixel = nextRandom(randomState);
                        const size_t pixelCount =
                            1u + randomBelow(static_cast<uint32_t>(
                                     pixels.size()));
                        const uint16_t x = static_cast<uint16_t>(
                            randomBelow(61u));
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
                    case StrictTransitionAction::CpuReadback:
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
                                "strict scoped readback diverged at operation " +
                                    std::to_string(operationIndex));
                            return;
                        }
                        break;
                    }
                    case StrictTransitionAction::LocalCopy:
                    {
                        const uint16_t sourceX =
                            static_cast<uint16_t>(randomBelow(57u));
                        const uint16_t sourceY =
                            static_cast<uint16_t>(randomBelow(25u));
                        const uint16_t destinationX =
                            static_cast<uint16_t>(randomBelow(57u));
                        const uint16_t destinationY =
                            static_cast<uint16_t>(randomBelow(25u));
                        copyCt32Pixels(
                            software,
                            page * 32u,
                            pages[secondPageIndex] * 32u,
                            1u,
                            sourceX, sourceY,
                            destinationX, destinationY,
                            8u, 8u);
                        copyCt32Pixels(
                            accelerated,
                            page * 32u,
                            pages[secondPageIndex] * 32u,
                            1u,
                            sourceX, sourceY,
                            destinationX, destinationY,
                            8u, 8u);
                        break;
                    }
                    case StrictTransitionAction::SoftwareModeTriangle:
                        if (!accelerated.setRendererMode(
                                GsRendererMode::Software))
                        {
                            t.IsTrue(
                                false,
                                "strict transition stream failed to enter software mode");
                            return;
                        }
                        drawTrianglePair(page);
                        if (!accelerated.setRendererMode(
                                GsRendererMode::GpuStrict))
                        {
                            t.IsTrue(
                                false,
                                "strict transition stream failed to restore strict mode");
                            return;
                        }
                        if (!compareFullVram(
                                "software-to-strict mode switch",
                                operationIndex))
                        {
                            return;
                        }
                        break;
                    case StrictTransitionAction::TextureDraw:
                        drawTexturePair();
                        break;
                    case StrictTransitionAction::Observation:
                        if (!forceObservation(
                                static_cast<uint32_t>(cycle % 4u),
                                operationIndex))
                        {
                            return;
                        }
                        break;
                    case StrictTransitionAction::Count:
                        break;
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
                         "every strict transition action should run once per cycle");
            }
            t.Equals(operationIndex, cycleCount * actionCount,
                     "the strict transition stream should remain bounded");
            t.Equals(forcedObservations, 17ull,
                     "mode switches scheduled observations and final state should compare VRAM");
            t.Equals(frameObservations, 2ull,
                     "both scheduled frame observations should compare host pixels");

            const GsBackendCounters counters =
                accelerated.backendCounters();
            t.Equals(counters.commands, 56ull,
                     "the strict stream should assemble its exact draw count");
            t.Equals(counters.acceleratedCommands, 48ull,
                     "every strict-mode draw should reach Vulkan");
            t.Equals(counters.softwareCommands, 8ull,
                     "only explicit software-mode triangles should use the CPU");
            t.Equals(counters.fallbackCommands, 0ull,
                     "strict transitions should contain no implicit fallback");
            t.Equals(counters.strictFailures, 0ull,
                     "every strict draw should satisfy the qualified envelope");
            t.IsTrue(counters.queueHighWatermark >= 2ull,
                     "each explicit triangle pair should expose resident queueing");
            t.Equals(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::BackendSwitch)],
                16ull,
                "each explicit software interval should cross two backend boundaries");

            const GsVulkanRasterBackendStatistics backend =
                accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsCompleted, 48ull,
                     "every strict GPU command should complete once");
            t.Equals(backend.committedGpuCommands, 48ull,
                     "every strict geometry and texture draw should commit once");
            t.Equals(backend.residentCommands, 48ull,
                     "every qualified strict class should use resident execution");
            t.Equals(backend.pageOwnership.gpuNewerPages,
                     static_cast<size_t>(0u),
                     "the final observation should publish every strict writer");
            t.Equals(backend.coherency.rejectedTransitions, 0ull,
                     "the strict stream should never create competing writers");
            t.IsTrue(backend.residentBatchesCompleted > 0ull,
                     "strict transitions should exercise resident batches");
            t.IsTrue(backend.pipelineChangeDrains >= 8ull,
                     "each triangle-to-sprite pair should retain a pipeline drain");

            const GsVulkanServiceStatistics service =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(service.triangleDrawsCompleted, 32ull,
                     "the service should execute every strict triangle once");
            t.Equals(service.spriteDrawsCompleted, 8ull,
                     "the service should execute every pipeline-change sprite once");
            t.Equals(service.nearestCt32SpriteDrawsCompleted, 8ull,
                     "the service should execute every shuffled strict texture once");
            t.Equals(
                service.nearestCt32SpritePixelsExecuted,
                8ull * static_cast<uint64_t>(
                    preparedTexture.boundsX1 - preparedTexture.boundsX0) *
                    static_cast<uint64_t>(
                        preparedTexture.boundsY1 - preparedTexture.boundsY0),
                "the randomized stream should retain exact texture pixel accounting");
            t.IsTrue(service.residentTriangleBatchesCompleted > 0ull,
                     "strict triangles should use the resident triangle service");
            t.Equals(
                service.residentNearestCt32SpriteBatchesCompleted,
                8ull,
                "every shuffled strict texture should use a resident batch");
            t.Equals(service.largestResidentNearestCt32SpriteBatch,
                     1ull,
                     "repeated texture destinations should remain dependency-split");
            t.IsTrue(service.largestResidentTriangleBatch >= 2ull,
                     "the explicit triangle pairs should share a resident batch");
            t.IsTrue(service.pageUploadOperationsCompleted > 0ull,
                     "strict transitions should upload CPU-newer pages");
            t.IsTrue(service.pageDownloadOperationsCompleted > 0ull,
                     "strict observations should publish GPU-newer pages");
            t.Equals(service.fenceTimeouts, 0ull,
                     "the bounded strict transition stream should not time out");
            t.Equals(service.validationErrors, 0u,
                     "strict triangle transitions should remain validation-clean");
            t.Equals(service.validationWarnings, 0u,
                     "strict triangle transitions should emit no validation warnings");
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

        tc.Run("Vulkan triangle mismatch artifact retains exact shader record", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            ScopedArtifactDirectory artifacts;
            const GsDrawCommand command = makeCt32TriangleCommand(
                92u, 5u, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                {65u, 337u, 113u},
                {81u, 145u, 321u},
                0x88776655u);
            std::vector<uint8_t> vram(GS_VULKAN_VRAM_SIZE, 0u);

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
                        GsVulkanCt32Triangle triangle{};
                        if (prepareGsVulkanCt32Triangle(
                                draw, triangle).supported)
                        {
                            applyCt32TriangleCpu(vram, triangle);
                        }
                    },
                    {}, nullptr);
            t.IsNotNull(backend.get(),
                        "the triangle mismatch backend should construct");
            if (!backend)
                return;

            bool mismatch = false;
            try
            {
                backend->submit(
                    std::span<const GsDrawCommand>(&command, 1u));
            }
            catch (const std::runtime_error &)
            {
                mismatch = true;
            }
            t.IsTrue(mismatch,
                     "the injected no-op triangle should disagree with software");
            const std::filesystem::path bundle =
                backend->backendStatistics().lastVerificationArtifact;
            t.IsTrue(std::filesystem::is_directory(bundle),
                     "the triangle reproducer should be published atomically");
            std::ifstream manifest(bundle / "command.json");
            const std::string manifestText{
                std::istreambuf_iterator<char>(manifest),
                std::istreambuf_iterator<char>()};
            t.IsTrue(manifestText.find("\"triangle\"") !=
                         std::string::npos,
                     "the manifest should identify the triangle record");
            t.IsTrue(manifestText.find("\"vertex0_x_12_4\"") !=
                         std::string::npos,
                     "the manifest should retain signed fixed-point vertices");
            t.IsTrue(manifestText.find("\"top_left_edge_mask\"") !=
                         std::string::npos,
                     "the manifest should retain exact edge ownership");
        });

        tc.Run("Vulkan texture mismatch artifact retains exact sampling record", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            ScopedArtifactDirectory artifacts;
            const GsDrawCommand command = makeNearestCt32SpriteCommand(
                93u, 40u, 2u, 64u, 2u, 6u, 5u,
                {6u, 15u, 5u, 12u}, {32u, 16u},
                {352u, 96u}, {48u, 304u},
                {480u, 224u}, {64u, 320u}, 3u, 1u,
                15u, 128u, 40u, 42u);
            std::vector<uint8_t> vram = makeVramPattern(0x54584D31u);

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
                        GsVulkanNearestCt32Sprite sprite{};
                        if (prepareGsVulkanNearestCt32Sprite(
                                draw, sprite).supported)
                        {
                            applyNearestCt32SpriteCpu(vram, sprite);
                        }
                    },
                    {}, nullptr);
            t.IsNotNull(backend.get(),
                        "the texture mismatch backend should construct");
            if (!backend)
                return;

            bool mismatch = false;
            try
            {
                backend->submit(
                    std::span<const GsDrawCommand>(&command, 1u));
            }
            catch (const std::runtime_error &)
            {
                mismatch = true;
            }
            t.IsTrue(mismatch,
                     "the injected no-op texture result should disagree with software");
            const GsVulkanRasterBackendStatistics statistics =
                backend->backendStatistics();
            t.Equals(statistics.verificationMismatches, 1ull,
                     "the texture mismatch should stop at its first draw");
            const std::filesystem::path bundle =
                statistics.lastVerificationArtifact;
            t.IsTrue(std::filesystem::is_directory(bundle),
                     "the texture reproducer should be published atomically");
            std::ifstream manifest(bundle / "command.json");
            const std::string manifestText{
                std::istreambuf_iterator<char>(manifest),
                std::istreambuf_iterator<char>()};
            t.IsTrue(manifestText.find("\"nearest_ct32_sprite\"") !=
                         std::string::npos,
                     "the manifest should identify the texture record");
            t.IsTrue(manifestText.find("\"texture_base_block\":64") !=
                         std::string::npos,
                     "the manifest should retain the raw texture base");
            t.IsTrue(manifestText.find("\"texture_mask_u\":63") !=
                         std::string::npos,
                     "the manifest should retain the exact texture extent mask");
            t.IsTrue(manifestText.find("\"texture_origin_u\":16") !=
                         std::string::npos,
                     "the manifest should retain the clipped texture origin");
            t.IsTrue(manifestText.find("\"texture_step_u\":1") !=
                         std::string::npos,
                     "the manifest should retain signed sampling direction");
            t.IsTrue(manifestText.find("\"texture_wrap_mode_u\":3") !=
                         std::string::npos,
                     "the manifest should retain REGION_REPEAT U");
            t.IsTrue(manifestText.find("\"texture_wrap_mode_v\":1") !=
                         std::string::npos,
                     "the manifest should retain standard V clamp independently");
            t.IsTrue(manifestText.find("\"texture_region_min_u\":15") !=
                         std::string::npos,
                     "the manifest should retain raw MINU before nominal masking");
            t.IsTrue(manifestText.find("\"texture_region_max_u\":128") !=
                         std::string::npos,
                     "the manifest should retain the raw MAXU offset");
            t.IsTrue(manifestText.find("\"texture_region_min_v\":40") !=
                         std::string::npos,
                     "the manifest should retain raw MINV independently");
            t.IsTrue(manifestText.find("\"texture_region_max_v\":42") !=
                         std::string::npos,
                     "the manifest should retain raw MAXV independently");
        });

        tc.Run("GS Vulkan verifies nearest CT32 texture sprites end to end", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const GsDrawCommand command = makeNearestCt32SpriteCommand(
                94u, 40u, 2u, 64u, 2u, 6u, 5u,
                {6u, 15u, 5u, 12u}, {32u, 16u},
                {352u, 96u}, {48u, 304u},
                {480u, 224u}, {64u, 320u});
            const GsDrawCommand clampedCommand =
                makeNearestCt32SpriteCommand(
                    95u, 41u, 2u, 64u, 2u, 3u, 2u,
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 1u, 1u);
            const GsDrawCommand regionClampedCommand =
                makeNearestCt32SpriteCommand(
                    96u, 42u, 2u, 64u, 2u, 3u, 2u,
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 2u, 2u,
                    70u, 72u, 40u, 42u);
            const GsDrawCommand regionRepeatedCommand =
                makeNearestCt32SpriteCommand(
                    97u, 43u, 2u, 64u, 2u, 3u, 2u,
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 3u, 3u,
                    15u, 128u, 15u, 64u);
            GsVulkanNearestCt32Sprite prepared{};
            GsVulkanNearestCt32Sprite clampedPrepared{};
            GsVulkanNearestCt32Sprite regionClampedPrepared{};
            GsVulkanNearestCt32Sprite regionRepeatedPrepared{};
            t.IsTrue(
                prepareGsVulkanNearestCt32Sprite(
                    command, prepared).supported,
                "the integrated fixture should satisfy the nearest texture predicate");
            t.IsTrue(
                prepareGsVulkanNearestCt32Sprite(
                    clampedCommand, clampedPrepared).supported,
                "the integrated standard-clamp fixture should be eligible");
            t.IsTrue(
                prepareGsVulkanNearestCt32Sprite(
                    regionClampedCommand,
                    regionClampedPrepared).supported,
                "the integrated REGION_CLAMP fixture should be eligible");
            t.IsTrue(
                prepareGsVulkanNearestCt32Sprite(
                    regionRepeatedCommand,
                    regionRepeatedPrepared).supported,
                "the integrated REGION_REPEAT fixture should be eligible");

            std::vector<uint8_t> softwareVram =
                makeVramPattern(0x54585632u);
            std::vector<uint8_t> acceleratedVram = softwareVram;
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
            ScopedArtifactDirectory artifacts;
            t.IsTrue(accelerated.configureVulkanRenderer(
                         config, artifacts.path.string()),
                     "the texture fixture should accept Vulkan configuration");
            if (!preflight.ready())
            {
                t.IsFalse(accelerated.setRendererMode(
                              GsRendererMode::Verify),
                          "an unavailable host should decline texture Verify cleanly");
                return;
            }

            const GsVulkanDeviceReport *selected =
                preflight.selectedDevice();
            t.IsNotNull(selected,
                        "a ready texture preflight should select one device");
            if (!selected)
                return;
            t.IsTrue(selected->exactNearestCt32Sprite,
                     "the selected raw-VRAM device should expose the exact texture kernel");
            if (!selected->exactNearestCt32Sprite)
                return;

            t.IsTrue(accelerated.setRendererMode(
                         GsRendererMode::Verify),
                     "the capable host should create texture Verify");
            if (accelerated.rendererMode() != GsRendererMode::Verify)
                return;
            accelerated.setBackendCountersEnabled(true);
            accelerated.resetBackendCounters();

            drawNearestCt32SpriteCommand(software, command);
            drawNearestCt32SpriteCommand(accelerated, command);
            drawNearestCt32SpriteCommand(software, clampedCommand);
            drawNearestCt32SpriteCommand(accelerated, clampedCommand);
            drawNearestCt32SpriteCommand(software, regionClampedCommand);
            drawNearestCt32SpriteCommand(accelerated, regionClampedCommand);
            drawNearestCt32SpriteCommand(software, regionRepeatedCommand);
            drawNearestCt32SpriteCommand(accelerated, regionRepeatedCommand);
            (void)software.getDebugSnapshot();
            (void)accelerated.getDebugSnapshot();
            t.IsTrue(acceleratedVram == softwareVram,
                     "real Vulkan texture Verify should match all canonical VRAM");

            const GsBackendCounters counters =
                accelerated.backendCounters();
            t.Equals(counters.commands, 4ull,
                     "the texture fixture should assemble all four wrap classes");
            t.Equals(counters.acceleratedCommands, 4ull,
                     "all exact texture draws should use Vulkan Verify");
            t.Equals(counters.verifiedCommands, 4ull,
                     "all texture draws should record verification");
            t.Equals(counters.softwareCommands, 0ull,
                     "the routed texture draw should not use fallback");
            t.Equals(counters.fallbackCommands, 0ull,
                     "the exact texture draw should have no fallback decision");
            t.Equals(
                counters.decisions[static_cast<size_t>(
                    GsFallbackReason::Supported)],
                4ull,
                "the router should retain every supported texture decision");

            const GsVulkanRasterBackendStatistics backend =
                accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsAttempted, 4ull,
                     "the integrated backend should attempt all texture draws");
            t.Equals(backend.commandsCompleted, 4ull,
                     "all matching texture draws should complete once");
            t.Equals(backend.verifiedCommands, 4ull,
                     "the backend should compare all four wrap results");
            t.Equals(backend.bytesCompared,
                     4ull * GS_VULKAN_VRAM_SIZE,
                     "each texture Verify should compare the complete 4 MiB image");
            t.Equals(backend.verificationMismatches, 0ull,
                     "the real texture kernel should have no byte mismatch");

            const GsVulkanServiceStatistics service =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(service.nearestCt32SpriteDrawsCompleted, 4ull,
                     "the service should execute all four wrap classes");
            t.Equals(service.nearestCt32SpriteDrawsFailed, 0ull,
                     "the real texture request should not fail");
            t.Equals(
                service.nearestCt32SpritePixelsExecuted,
                static_cast<uint64_t>(
                    prepared.boundsX1 - prepared.boundsX0) *
                    static_cast<uint64_t>(
                        prepared.boundsY1 - prepared.boundsY0) +
                    static_cast<uint64_t>(
                        clampedPrepared.boundsX1 -
                        clampedPrepared.boundsX0) *
                    static_cast<uint64_t>(
                        clampedPrepared.boundsY1 -
                        clampedPrepared.boundsY0) +
                    static_cast<uint64_t>(
                        regionClampedPrepared.boundsX1 -
                        regionClampedPrepared.boundsX0) *
                    static_cast<uint64_t>(
                        regionClampedPrepared.boundsY1 -
                        regionClampedPrepared.boundsY0) +
                    static_cast<uint64_t>(
                        regionRepeatedPrepared.boundsX1 -
                        regionRepeatedPrepared.boundsX0) *
                    static_cast<uint64_t>(
                        regionRepeatedPrepared.boundsY1 -
                        regionRepeatedPrepared.boundsY0),
                "the service should retain exact texture pixel accounting");
            t.Equals(service.spriteDrawsCompleted, 0ull,
                     "texture routing must not alias the flat sprite request");
            t.Equals(service.triangleDrawsCompleted, 0ull,
                     "texture routing must not alias the triangle request");
            t.Equals(service.validationErrors, 0u,
                     "integrated texture Verify should remain validation-clean");
            t.Equals(service.validationWarnings, 0u,
                     "integrated texture Verify should emit no validation warnings");
        });

        tc.Run("GS Vulkan verifies linear CT32 repeat and clamp end to end", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const GsDrawCommand command =
                makeLinearCt32RepeatSpriteCommand(
                    99u, 0u, 8u, 3584u, 8u, 10u, 10u,
                    {0u, 511u, 0u, 447u},
                    {28672u, 29184u},
                    {28664u, 29176u},
                    {29176u, 36344u},
                    {0u, 512u},
                    {0u, 6656u});
            const GsDrawCommand clampedCommand =
                makeLinearCt32SpriteCommand(
                    100u, 0u, 8u, 3584u, 8u, 10u, 10u,
                    {0u, 511u, 0u, 447u},
                    {28672u, 29184u},
                    {28664u, 29176u},
                    {29176u, 36344u},
                    {0u, 512u},
                    {0u, 6656u},
                    1u, 1u);
            GsVulkanLinearCt32Sprite prepared{};
            GsVulkanLinearCt32Sprite clampedPrepared{};
            t.IsTrue(
                prepareGsVulkanLinearCt32Sprite(
                    command, prepared).supported,
                "the integrated title fixture should satisfy the linear predicate");
            t.IsTrue(
                prepareGsVulkanLinearCt32Sprite(
                    clampedCommand, clampedPrepared).supported,
                "the integrated clamp title fixture should satisfy the linear predicate");

            std::vector<uint8_t> softwareVram =
                makeVramPattern(0x4C565247u);
            std::vector<uint8_t> acceleratedVram = softwareVram;
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
            ScopedArtifactDirectory artifacts;
            t.IsTrue(accelerated.configureVulkanRenderer(
                         config, artifacts.path.string()),
                     "the linear fixture should accept Vulkan configuration");
            if (!preflight.ready())
            {
                t.IsFalse(accelerated.setRendererMode(
                              GsRendererMode::Verify),
                          "an unavailable host should decline linear Verify cleanly");
                return;
            }

            const GsVulkanDeviceReport *selected =
                preflight.selectedDevice();
            t.IsNotNull(selected,
                        "a ready linear preflight should select one device");
            if (!selected)
                return;
            t.IsTrue(selected->exactLinearCt32Sprite,
                     "the selected raw-VRAM device should expose the exact linear kernel");
            if (!selected->exactLinearCt32Sprite)
                return;

            t.IsTrue(accelerated.setRendererMode(
                         GsRendererMode::Verify),
                     "the capable host should create linear Verify");
            if (accelerated.rendererMode() != GsRendererMode::Verify)
                return;
            accelerated.setBackendCountersEnabled(true);
            accelerated.resetBackendCounters();

            drawNearestCt32SpriteCommand(software, command);
            drawNearestCt32SpriteCommand(accelerated, command);
            drawNearestCt32SpriteCommand(software, clampedCommand);
            drawNearestCt32SpriteCommand(accelerated, clampedCommand);
            (void)software.getDebugSnapshot();
            (void)accelerated.getDebugSnapshot();
            t.IsTrue(acceleratedVram == softwareVram,
                     "real Vulkan linear Verify should match canonical VRAM");

            const GsBackendCounters counters =
                accelerated.backendCounters();
            t.Equals(counters.commands, 2ull,
                     "the linear title fixture should assemble two draws");
            t.Equals(counters.acceleratedCommands, 2ull,
                     "both exact linear draws should use Vulkan Verify");
            t.Equals(counters.verifiedCommands, 2ull,
                     "both routed linear draws should record verification");
            t.Equals(counters.softwareCommands, 0ull,
                     "the routed linear draw should not use fallback");
            t.Equals(counters.fallbackCommands, 0ull,
                     "the exact linear draw should have no fallback decision");
            t.Equals(
                counters.decisions[static_cast<size_t>(
                    GsFallbackReason::Supported)],
                2ull,
                "the router should retain both supported linear decisions");

            const GsVulkanRasterBackendStatistics backend =
                accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsAttempted, 2ull,
                     "the integrated backend should attempt both linear draws");
            t.Equals(backend.commandsCompleted, 2ull,
                     "both matching linear draws should complete once");
            t.Equals(backend.verifiedCommands, 2ull,
                     "the backend should compare both linear results");
            t.Equals(backend.bytesCompared,
                     2ull * GS_VULKAN_VRAM_SIZE,
                     "linear Verify should compare two complete VRAM images");
            t.Equals(backend.verificationMismatches, 0ull,
                     "the real linear kernel should have no byte mismatch");

            const GsVulkanServiceStatistics service =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(service.linearCt32SpriteDrawsCompleted, 2ull,
                     "the service should execute repeat and clamp draws");
            t.Equals(service.linearCt32SpriteDrawsFailed, 0ull,
                     "the real linear request should not fail");
            t.Equals(
                service.linearCt32SpritePixelsExecuted,
                static_cast<uint64_t>(
                    prepared.boundsX1 - prepared.boundsX0) *
                    static_cast<uint64_t>(
                        prepared.boundsY1 - prepared.boundsY0) +
                    static_cast<uint64_t>(
                        clampedPrepared.boundsX1 -
                        clampedPrepared.boundsX0) *
                    static_cast<uint64_t>(
                        clampedPrepared.boundsY1 -
                        clampedPrepared.boundsY0),
                "the service should retain exact linear pixel accounting");
            t.Equals(service.nearestCt32SpriteDrawsCompleted, 0ull,
                     "linear routing must not alias the nearest request");
            t.Equals(service.validationErrors, 0u,
                     "integrated linear Verify should remain validation-clean");
            t.Equals(service.validationWarnings, 0u,
                     "integrated linear Verify should emit no validation warnings");
        });

        tc.Run("GS Vulkan Hybrid qualifies nearest CT32 textures at 8192 samples", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            constexpr uint64_t thresholdPixels = 8'192u;
            const GsDrawCommand belowThreshold =
                makeNearestCt32SpriteCommand(
                    95u, 100u, 2u, 64u, 1u, 6u, 5u,
                    {0u, 127u, 0u, 63u}, {0u, 0u},
                    {0u, 127u * 16u}, {0u, 64u * 16u},
                    {0u, 127u * 16u}, {0u, 64u * 16u});
            const GsDrawCommand atThreshold =
                makeNearestCt32SpriteCommand(
                    96u, 110u, 2u, 64u, 1u, 6u, 5u,
                    {0u, 127u, 0u, 63u}, {0u, 0u},
                    {0u, 128u * 16u}, {0u, 64u * 16u},
                    {0u, 128u * 16u}, {0u, 64u * 16u},
                    3u, 3u, 15u, 16u, 15u, 16u);
            GsVulkanNearestCt32Sprite belowPrepared{};
            GsVulkanNearestCt32Sprite thresholdPrepared{};
            t.IsTrue(prepareGsVulkanNearestCt32Sprite(
                         belowThreshold, belowPrepared).supported,
                     "the real below-threshold fixture should be eligible");
            t.IsTrue(prepareGsVulkanNearestCt32Sprite(
                         atThreshold, thresholdPrepared).supported,
                     "the real threshold fixture should be eligible");
            t.Equals(gsVulkanTextureWrapMode(
                         thresholdPrepared.textureWrapU), 3u,
                     "real Hybrid should retain REGION_REPEAT U");
            t.Equals(gsVulkanTextureWrapMode(
                         thresholdPrepared.textureWrapV), 3u,
                     "real Hybrid should retain REGION_REPEAT V");
            t.Equals(gsVulkanTextureRegionMax(
                         thresholdPrepared.textureWrapU), 16u,
                     "real Hybrid should retain the REGION_REPEAT U offset");
            t.Equals(gsVulkanTextureRegionMax(
                         thresholdPrepared.textureWrapV), 16u,
                     "real Hybrid should retain the REGION_REPEAT V offset");

            std::vector<uint8_t> softwareVram =
                makeVramPattern(0x54584859u);
            std::vector<uint8_t> acceleratedVram = softwareVram;
            GS software;
            GS accelerated;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()), nullptr);
            accelerated.init(
                acceleratedVram.data(),
                static_cast<uint32_t>(acceleratedVram.size()), nullptr);

            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig serviceConfig =
                makeRendererServiceConfig(preflight);
            GsVulkanRasterBackendConfig backendConfig{};
            t.Equals(backendConfig.minimumHybridNearestCt32SpritePixels,
                     thresholdPixels,
                     "the integrated fixture should use the measured default");
            t.IsTrue(accelerated.configureVulkanRenderer(
                         serviceConfig, backendConfig),
                     "the Hybrid texture fixture should accept Vulkan configuration");
            if (!preflight.ready())
            {
                t.IsFalse(accelerated.setRendererMode(
                              GsRendererMode::Hybrid),
                          "an unavailable host should decline texture Hybrid cleanly");
                return;
            }

            const GsVulkanDeviceReport *selected =
                preflight.selectedDevice();
            t.IsNotNull(selected,
                        "a ready Hybrid texture preflight should select one device");
            if (!selected)
                return;
            t.IsTrue(selected->exactNearestCt32Sprite,
                     "the selected device should expose exact nearest textures");
            if (!selected->exactNearestCt32Sprite)
                return;

            t.IsTrue(accelerated.setRendererMode(
                         GsRendererMode::Hybrid),
                     "the qualified host should enter texture Hybrid mode");
            if (accelerated.rendererMode() != GsRendererMode::Hybrid)
                return;
            accelerated.setBackendCountersEnabled(true);
            accelerated.resetBackendCounters();

            drawNearestCt32SpriteCommand(software, belowThreshold);
            drawNearestCt32SpriteCommand(accelerated, belowThreshold);
            drawNearestCt32SpriteCommand(software, atThreshold);
            drawNearestCt32SpriteCommand(accelerated, atThreshold);

            GsBackendCounters counters = accelerated.backendCounters();
            t.Equals(counters.queueDepth, 1ull,
                     "only the threshold texture should remain GPU-resident");
            t.Equals(counters.softwareCommands, 1ull,
                     "the smaller texture should execute once in software");
            t.Equals(counters.acceleratedCommands, 1ull,
                     "the threshold texture should route to Vulkan once");

            (void)software.getDebugSnapshot();
            (void)accelerated.getDebugSnapshot();
            t.IsTrue(acceleratedVram == softwareVram,
                     "mixed Hybrid texture routing should match complete software VRAM");

            counters = accelerated.backendCounters();
            t.Equals(counters.commands, 2ull,
                     "Hybrid should classify both texture fixtures");
            t.Equals(counters.softwareCommands, 1ull,
                     "only the below-threshold texture should use software");
            t.Equals(counters.fallbackCommands, 1ull,
                     "the smaller texture should record one fallback");
            t.Equals(counters.acceleratedCommands, 1ull,
                     "only the threshold texture should use Vulkan");
            t.Equals(counters.drawPixels, 16'320ull,
                     "draw accounting should include both exact sample rectangles");
            t.Equals(counters.softwarePixels, 8'128ull,
                     "software accounting should retain the smaller rectangle");
            t.Equals(counters.fallbackPixels, 8'128ull,
                     "fallback accounting should retain the smaller rectangle");
            t.Equals(counters.acceleratedPixels, thresholdPixels,
                     "accelerated accounting should retain the threshold rectangle");
            t.Equals(counters.decisions[static_cast<size_t>(
                         GsFallbackReason::CostModel)],
                     1ull,
                     "Hybrid should retain one named texture cost decision");
            t.Equals(counters.decisions[static_cast<size_t>(
                         GsFallbackReason::Supported)],
                     1ull,
                     "Hybrid should retain one supported texture decision");

            const GsVulkanRasterBackendStatistics backend =
                accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsAttempted, 1ull,
                     "cost fallback must not become a backend attempt");
            t.Equals(backend.commandsCompleted, 1ull,
                     "the threshold texture should complete once");
            t.Equals(backend.committedGpuCommands, 1ull,
                     "the threshold texture should publish one GPU result");
            t.Equals(backend.residentCommands, 1ull,
                     "the accepted Hybrid texture should remain resident");
            t.Equals(backend.pageOwnership.gpuNewerPages, size_t{0u},
                     "debug observation should publish the texture destination");

            const GsVulkanServiceStatistics service =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(service.nearestCt32SpriteDrawsCompleted, 1ull,
                     "the real service should execute only the threshold texture");
            t.Equals(service.nearestCt32SpriteDrawsFailed, 0ull,
                     "real Hybrid texture execution should not fail");
            t.Equals(service.nearestCt32SpritePixelsExecuted,
                     thresholdPixels,
                     "the service should retain the threshold sample count");
            t.Equals(service.residentNearestCt32SpriteBatchesCompleted, 1ull,
                     "the accepted texture should use one resident service batch");
            t.Equals(service.largestResidentNearestCt32SpriteBatch, 1ull,
                     "the real cost fixture should expose a one-draw batch");
            t.Equals(service.validationErrors, 0u,
                     "real Hybrid texture routing should remain validation-clean");
            t.Equals(service.validationWarnings, 0u,
                     "real Hybrid texture routing should emit no validation warnings");
        });

        tc.Run("GS Vulkan Hybrid qualifies linear CT32 textures at measured work", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            constexpr uint64_t thresholdPixels = 131'072u;
            constexpr uint32_t width = 136u;
            const auto makeCommand = [](
                uint64_t sequence, uint32_t height,
                uint16_t uFraction = 3u)
            {
                return makeLinearCt32RepeatSpriteCommand(
                    sequence, 0u, 3u, 400u * 32u, 1u, 6u, 5u,
                    {0u, static_cast<uint16_t>(width - 1u),
                     0u, static_cast<uint16_t>(height - 1u)},
                    {0u, 0u},
                    {0u, static_cast<uint16_t>(width * 16u)},
                    {0u, static_cast<uint16_t>(height * 16u)},
                    {uFraction,
                     static_cast<uint16_t>(width * 16u + uFraction)},
                    {5u, static_cast<uint16_t>(height * 16u + 5u)});
            };
            const GsDrawCommand belowThreshold = makeCommand(97u, 963u);
            const GsDrawCommand admittedA = makeCommand(98u, 964u);
            const GsDrawCommand admittedB = makeCommand(99u, 964u, 11u);
            GsVulkanLinearCt32Sprite belowPrepared{};
            GsVulkanLinearCt32Sprite admittedPrepared{};
            t.IsTrue(
                prepareGsVulkanLinearCt32Sprite(
                    belowThreshold, belowPrepared).supported,
                "the real below-threshold linear fixture should be eligible");
            t.IsTrue(
                prepareGsVulkanLinearCt32Sprite(
                    admittedA, admittedPrepared).supported,
                "the real admitted linear fixture should be eligible");
            const auto pixels = [](const GsVulkanLinearCt32Sprite &sprite)
            {
                return static_cast<uint64_t>(
                           sprite.boundsX1 - sprite.boundsX0) *
                       static_cast<uint64_t>(
                           sprite.boundsY1 - sprite.boundsY0);
            };
            t.Equals(pixels(belowPrepared), 130'968ull,
                     "the real fallback should remain below measured work");
            t.Equals(pixels(admittedPrepared), 131'104ull,
                     "the first integral admitted row should clear measured work");

            std::vector<uint8_t> softwareVram =
                makeVramPattern(0x4C484759u);
            std::vector<uint8_t> acceleratedVram = softwareVram;
            GS software;
            GS accelerated;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()), nullptr);
            accelerated.init(
                acceleratedVram.data(),
                static_cast<uint32_t>(acceleratedVram.size()), nullptr);

            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig serviceConfig =
                makeRendererServiceConfig(preflight);
            GsVulkanRasterBackendConfig backendConfig{};
            t.Equals(backendConfig.minimumHybridLinearCt32SpritePixels,
                     thresholdPixels,
                     "the integrated linear fixture should use measured work");
            t.IsTrue(accelerated.configureVulkanRenderer(
                         serviceConfig, backendConfig),
                     "the Hybrid linear fixture should accept Vulkan configuration");
            if (!preflight.ready())
            {
                t.IsFalse(accelerated.setRendererMode(
                              GsRendererMode::Hybrid),
                          "an unavailable host should decline linear Hybrid cleanly");
                return;
            }

            const GsVulkanDeviceReport *selected =
                preflight.selectedDevice();
            t.IsNotNull(selected,
                        "a ready Hybrid linear preflight should select one device");
            if (!selected)
                return;
            t.IsTrue(selected->exactLinearCt32Sprite,
                     "the selected device should expose exact linear textures");
            if (!selected->exactLinearCt32Sprite)
                return;

            t.IsTrue(accelerated.setRendererMode(GsRendererMode::Hybrid),
                     "the qualified host should enter linear Hybrid mode");
            if (accelerated.rendererMode() != GsRendererMode::Hybrid)
                return;
            accelerated.setBackendCountersEnabled(true);
            accelerated.resetBackendCounters();

            drawNearestCt32SpriteCommand(software, belowThreshold);
            drawNearestCt32SpriteCommand(accelerated, belowThreshold);
            drawNearestCt32SpriteCommand(software, admittedA);
            drawNearestCt32SpriteCommand(accelerated, admittedA);
            drawNearestCt32SpriteCommand(software, admittedB);
            drawNearestCt32SpriteCommand(accelerated, admittedB);

            GsBackendCounters counters = accelerated.backendCounters();
            t.Equals(counters.queueDepth, 2ull,
                     "ordered admitted linear draws should remain resident together");
            t.Equals(counters.softwareCommands, 1ull,
                     "the smaller linear draw should execute in software");
            t.Equals(counters.acceleratedCommands, 2ull,
                     "both measured linear draws should route to Vulkan");

            (void)software.getDebugSnapshot();
            (void)accelerated.getDebugSnapshot();
            t.IsTrue(acceleratedVram == softwareVram,
                     "mixed Hybrid linear routing should match all 4 MiB");

            counters = accelerated.backendCounters();
            t.Equals(counters.commands, 3ull,
                     "Hybrid should classify all three linear fixtures");
            t.Equals(counters.fallbackCommands, 1ull,
                     "the smaller linear draw should record one fallback");
            t.Equals(counters.softwarePixels, 130'968ull,
                     "software accounting should retain sub-threshold work");
            t.Equals(counters.acceleratedPixels, 262'208ull,
                     "accelerated accounting should retain both admitted draws");
            t.Equals(counters.decisions[static_cast<size_t>(
                         GsFallbackReason::CostModel)],
                     1ull,
                     "Hybrid should retain one named linear cost decision");
            t.Equals(counters.decisions[static_cast<size_t>(
                         GsFallbackReason::Supported)],
                     2ull,
                     "Hybrid should retain two supported linear decisions");

            const GsVulkanRasterBackendStatistics backend =
                accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsAttempted, 2ull,
                     "linear cost fallback must not reach the backend");
            t.Equals(backend.committedGpuCommands, 2ull,
                     "both admitted linear draws should publish once");
            t.Equals(backend.residentCommands, 2ull,
                     "both admitted linear draws should remain resident");
            t.Equals(backend.resourceHazardDrains, 0ull,
                     "ordered linear writes should not force a queue drain");
            t.Equals(backend.coherency.rejectedTransitions, 0ull,
                     "mixed linear routing should preserve page ownership");

            const GsVulkanServiceStatistics service =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(service.linearCt32SpriteDrawsCompleted, 2ull,
                     "the service should execute both admitted linear draws");
            t.Equals(service.linearCt32SpritePixelsExecuted, 262'208ull,
                     "the service should retain both admitted pixel counts");
            t.Equals(service.residentLinearCt32SpriteBatchesCompleted, 1ull,
                     "ordered linear writes should use one service batch");
            t.Equals(service.largestResidentLinearCt32SpriteBatch, 2ull,
                     "the real linear cost fixture should expose batch size two");
            t.Equals(service.nearestCt32SpriteDrawsCompleted, 0ull,
                     "linear Hybrid must not alias nearest execution");
            t.Equals(service.validationErrors, 0u,
                     "real Hybrid linear routing should remain validation-clean");
            t.Equals(service.validationWarnings, 0u,
                     "real Hybrid linear routing should emit no warnings");
        });

        tc.Run("GS Vulkan Hybrid qualifies linear CT32 clamp at measured work", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            constexpr uint64_t thresholdPixels = 8'192u;
            constexpr uint32_t width = 32u;
            const auto makeCommand = [](
                uint64_t sequence, uint32_t framebufferPage,
                uint32_t height, uint16_t uFraction)
            {
                return makeLinearCt32SpriteCommand(
                    sequence, framebufferPage, 1u,
                    400u * 32u, 1u, 6u, 5u,
                    {0u, static_cast<uint16_t>(width - 1u),
                     0u, static_cast<uint16_t>(height - 1u)},
                    {0u, 0u},
                    {0u, static_cast<uint16_t>(width * 16u)},
                    {0u, static_cast<uint16_t>(height * 16u)},
                    {uFraction,
                     static_cast<uint16_t>(width * 16u + uFraction)},
                    {5u, static_cast<uint16_t>(height * 16u + 5u)},
                    1u, 1u);
            };
            const GsDrawCommand belowThreshold =
                makeCommand(100u, 0u, 255u, 3u);
            const GsDrawCommand admittedA =
                makeCommand(101u, 8u, 256u, 3u);
            const GsDrawCommand admittedB =
                makeCommand(102u, 16u, 256u, 11u);
            GsVulkanLinearCt32Sprite belowPrepared{};
            GsVulkanLinearCt32Sprite admittedPrepared{};
            t.IsTrue(
                prepareGsVulkanLinearCt32Sprite(
                    belowThreshold, belowPrepared).supported,
                "the real below-threshold clamp fixture should be eligible");
            t.IsTrue(
                prepareGsVulkanLinearCt32Sprite(
                    admittedA, admittedPrepared).supported,
                "the real admitted clamp fixture should be eligible");
            const auto pixels = [](const GsVulkanLinearCt32Sprite &sprite)
            {
                return static_cast<uint64_t>(
                           sprite.boundsX1 - sprite.boundsX0) *
                       static_cast<uint64_t>(
                           sprite.boundsY1 - sprite.boundsY0);
            };
            t.Equals(pixels(belowPrepared), 8'160ull,
                     "the real clamp fallback should remain below policy");
            t.Equals(pixels(admittedPrepared), thresholdPixels,
                     "the first admitted clamp row should land on policy");

            std::vector<uint8_t> softwareVram =
                makeVramPattern(0x4C48434Du);
            std::vector<uint8_t> acceleratedVram = softwareVram;
            GS software;
            GS accelerated;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()), nullptr);
            accelerated.init(
                acceleratedVram.data(),
                static_cast<uint32_t>(acceleratedVram.size()), nullptr);

            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig serviceConfig =
                makeRendererServiceConfig(preflight);
            GsVulkanRasterBackendConfig backendConfig{};
            t.Equals(
                backendConfig.minimumHybridLinearCt32ClampSpritePixels,
                thresholdPixels,
                "the integrated clamp fixture should use measured work");
            t.IsTrue(accelerated.configureVulkanRenderer(
                         serviceConfig, backendConfig),
                     "the Hybrid clamp fixture should accept Vulkan configuration");
            if (!preflight.ready())
            {
                t.IsFalse(accelerated.setRendererMode(
                              GsRendererMode::Hybrid),
                          "an unavailable host should decline clamp Hybrid cleanly");
                return;
            }

            const GsVulkanDeviceReport *selected =
                preflight.selectedDevice();
            t.IsNotNull(selected,
                        "a ready Hybrid clamp preflight should select one device");
            if (!selected)
                return;
            t.IsTrue(selected->exactLinearCt32Sprite,
                     "the selected device should expose exact linear clamp");
            if (!selected->exactLinearCt32Sprite)
                return;

            t.IsTrue(accelerated.setRendererMode(GsRendererMode::Hybrid),
                     "the qualified host should enter clamp Hybrid mode");
            if (accelerated.rendererMode() != GsRendererMode::Hybrid)
                return;
            accelerated.setBackendCountersEnabled(true);
            accelerated.resetBackendCounters();

            drawNearestCt32SpriteCommand(software, belowThreshold);
            drawNearestCt32SpriteCommand(accelerated, belowThreshold);
            drawNearestCt32SpriteCommand(software, admittedA);
            drawNearestCt32SpriteCommand(accelerated, admittedA);
            drawNearestCt32SpriteCommand(software, admittedB);
            drawNearestCt32SpriteCommand(accelerated, admittedB);

            GsBackendCounters counters = accelerated.backendCounters();
            t.Equals(counters.queueDepth, 2ull,
                     "admitted clamp draws should remain resident together");
            t.Equals(counters.softwareCommands, 1ull,
                     "the smaller clamp draw should execute in software");
            t.Equals(counters.acceleratedCommands, 2ull,
                     "both measured clamp draws should route to Vulkan");

            (void)software.getDebugSnapshot();
            (void)accelerated.getDebugSnapshot();
            t.IsTrue(acceleratedVram == softwareVram,
                     "mixed Hybrid clamp routing should match all 4 MiB");

            counters = accelerated.backendCounters();
            t.Equals(counters.commands, 3ull,
                     "Hybrid should classify all three clamp fixtures");
            t.Equals(counters.fallbackCommands, 1ull,
                     "the smaller clamp draw should record one fallback");
            t.Equals(counters.softwarePixels, 8'160ull,
                     "software accounting should retain sub-threshold clamp work");
            t.Equals(counters.acceleratedPixels, 16'384ull,
                     "accelerated accounting should retain admitted clamp work");
            t.Equals(counters.decisions[static_cast<size_t>(
                         GsFallbackReason::CostModel)],
                     1ull,
                     "Hybrid should retain one named clamp cost decision");
            t.Equals(counters.decisions[static_cast<size_t>(
                         GsFallbackReason::Supported)],
                     2ull,
                     "Hybrid should retain two supported clamp decisions");

            const GsVulkanRasterBackendStatistics backend =
                accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsAttempted, 2ull,
                     "clamp cost fallback must not reach the backend");
            t.Equals(backend.committedGpuCommands, 2ull,
                     "both admitted clamp draws should publish once");
            t.Equals(backend.residentCommands, 2ull,
                     "both admitted clamp draws should remain resident");
            t.Equals(backend.resourceHazardDrains, 0ull,
                     "disjoint clamp writes should not force a queue drain");
            t.Equals(backend.coherency.rejectedTransitions, 0ull,
                     "mixed clamp routing should preserve page ownership");

            const GsVulkanServiceStatistics service =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(service.linearCt32SpriteDrawsCompleted, 2ull,
                     "the service should execute both admitted clamp draws");
            t.Equals(service.linearCt32SpritePixelsExecuted, 16'384ull,
                     "the service should retain admitted clamp pixel counts");
            t.Equals(service.residentLinearCt32SpriteBatchesCompleted, 1ull,
                     "disjoint clamp writes should use one service batch");
            t.Equals(service.largestResidentLinearCt32SpriteBatch, 2ull,
                     "the real clamp cost fixture should expose batch size two");
            t.Equals(service.nearestCt32SpriteDrawsCompleted, 0ull,
                     "linear clamp Hybrid must not alias nearest execution");
            t.Equals(service.validationErrors, 0u,
                     "real Hybrid clamp routing should remain validation-clean");
            t.Equals(service.validationWarnings, 0u,
                     "real Hybrid clamp routing should emit no warnings");
        });

        tc.Run("GS Vulkan routes nearest CT32 texture sprites in strict mode", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const GsDrawCommand command = makeNearestCt32SpriteCommand(
                95u, 40u, 2u, 64u, 2u, 6u, 5u,
                {6u, 15u, 5u, 12u}, {32u, 16u},
                {352u, 96u}, {48u, 304u},
                {480u, 224u}, {64u, 320u});
            const GsDrawCommand secondCommand =
                makeNearestCt32SpriteCommand(
                    96u, 41u, 2u, 64u, 2u, 6u, 5u,
                    {6u, 15u, 5u, 12u}, {32u, 16u},
                    {352u, 96u}, {48u, 304u},
                    {480u, 224u}, {64u, 320u}, 3u, 3u,
                    15u, 16u, 15u, 16u);
            const GsDrawCommand clampedCommand =
                makeNearestCt32SpriteCommand(
                    97u, 42u, 2u, 64u, 2u, 3u, 2u,
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 1u, 1u);
            const GsDrawCommand regionClampedCommand =
                makeNearestCt32SpriteCommand(
                    98u, 43u, 2u, 64u, 2u, 3u, 2u,
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 2u, 2u,
                    70u, 72u, 40u, 42u);
            GsVulkanNearestCt32Sprite prepared{};
            GsVulkanNearestCt32Sprite secondPrepared{};
            GsVulkanNearestCt32Sprite clampedPrepared{};
            GsVulkanNearestCt32Sprite regionClampedPrepared{};
            t.IsTrue(
                prepareGsVulkanNearestCt32Sprite(
                    command, prepared).supported,
                "the strict frontend fixture should satisfy the texture predicate");
            t.IsTrue(
                prepareGsVulkanNearestCt32Sprite(
                    secondCommand, secondPrepared).supported,
                "the shared-source strict REGION_REPEAT fixture should satisfy the texture predicate");
            t.IsTrue(
                prepareGsVulkanNearestCt32Sprite(
                    clampedCommand, clampedPrepared).supported,
                "the shared-source strict clamp fixture should be eligible");
            t.IsTrue(
                prepareGsVulkanNearestCt32Sprite(
                    regionClampedCommand,
                    regionClampedPrepared).supported,
                "the shared-source strict REGION_CLAMP fixture should be eligible");

            GSContext filteredContext = command.context();
            filteredContext.tex1 |= 1ull << 5u;
            const GsDrawCommand filtered = buildGsDrawCommand(
                99u, command.primitive(), filteredContext,
                std::span<const GSVertex>(
                    command.vertices().data(), command.vertexCount()),
                command.globalState());
            t.Equals(
                prepareGsVulkanNearestCt32Sprite(
                    filtered, prepared).reason,
                GsFallbackReason::UnsupportedTextureFilter,
                "the rejected strict fixture should name its first invariant");

            std::vector<uint8_t> softwareVram =
                makeVramPattern(0x54585331u);
            std::vector<uint8_t> strictVram = softwareVram;
            GS software;
            GS strict;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()), nullptr);
            strict.init(
                strictVram.data(),
                static_cast<uint32_t>(strictVram.size()), nullptr);

            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig config =
                makeRendererServiceConfig(preflight);
            t.IsTrue(strict.configureVulkanRenderer(config),
                     "the strict texture fixture should accept Vulkan configuration");
            if (!preflight.ready())
            {
                t.IsFalse(strict.setRendererMode(
                              GsRendererMode::GpuStrict),
                          "an unavailable host should decline texture strict cleanly");
                return;
            }

            const GsVulkanDeviceReport *selected =
                preflight.selectedDevice();
            t.IsNotNull(selected,
                        "a ready strict texture preflight should select one device");
            if (!selected)
                return;
            t.IsTrue(selected->exactNearestCt32Sprite,
                     "the selected device should expose the exact texture kernel");
            if (!selected->exactNearestCt32Sprite)
                return;

            t.IsTrue(strict.setRendererMode(
                         GsRendererMode::GpuStrict),
                     "the qualified host should enter texture strict mode");
            if (strict.rendererMode() != GsRendererMode::GpuStrict)
                return;
            strict.setBackendCountersEnabled(true);
            strict.resetBackendCounters();

            drawNearestCt32SpriteCommand(software, command);
            drawNearestCt32SpriteCommand(strict, command);
            drawNearestCt32SpriteCommand(software, secondCommand);
            drawNearestCt32SpriteCommand(strict, secondCommand);
            drawNearestCt32SpriteCommand(software, clampedCommand);
            drawNearestCt32SpriteCommand(strict, clampedCommand);
            drawNearestCt32SpriteCommand(software, regionClampedCommand);
            drawNearestCt32SpriteCommand(strict, regionClampedCommand);
            (void)software.getDebugSnapshot();
            (void)strict.getDebugSnapshot();
            t.IsTrue(strictVram == softwareVram,
                     "shared-read strict texture batching should match complete software VRAM");

            const std::vector<uint8_t> rejectionSentinel = strictVram;
            std::string rejection;
            try
            {
                drawNearestCt32SpriteCommand(strict, filtered);
            }
            catch (const std::runtime_error &error)
            {
                rejection = error.what();
            }
            t.IsTrue(rejection.find("unsupported-texture-filter") !=
                         std::string::npos,
                     "strict should expose the exact unsupported filter reason");
            t.IsTrue(strictVram == rejectionSentinel,
                     "strict texture rejection must precede canonical mutation");

            const GsBackendCounters counters = strict.backendCounters();
            t.Equals(counters.commands, 5ull,
                     "strict should classify four accepted and one rejected texture draw");
            t.Equals(counters.acceleratedCommands, 4ull,
                     "all qualified strict textures should reach Vulkan");
            t.Equals(counters.softwareCommands, 0ull,
                     "strict texture routing must never use software fallback");
            t.Equals(counters.fallbackCommands, 0ull,
                     "strict rejection should not be counted as Hybrid fallback");
            t.Equals(counters.strictFailures, 1ull,
                     "the unsupported filter should fail strict exactly once");
            t.Equals(counters.decisions[static_cast<size_t>(
                         GsFallbackReason::UnsupportedTextureFilter)],
                     1ull,
                     "the router should retain the exact strict texture reason");

            const GsVulkanRasterBackendStatistics backend =
                strict.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsAttempted, 4ull,
                     "only the supported strict textures should be submitted");
            t.Equals(backend.commandsCompleted, 4ull,
                     "all strict textures should complete exactly once");
            t.Equals(backend.committedGpuCommands, 4ull,
                     "all strict textures should publish one GPU result");
            t.Equals(backend.verifiedCommands, 0ull,
                     "strict texture routing should not run the Verify oracle");
            t.Equals(backend.bytesCompared, 0ull,
                     "strict texture routing should not claim a comparison");
            t.Equals(backend.residentCommands, 4ull,
                     "strict texture routing should batch resident execution");
            t.Equals(backend.pageOwnership.gpuNewerPages, size_t{0u},
                     "debug observation should publish resident texture VRAM");
            t.Equals(backend.coherency.rejectedTransitions, 0ull,
                     "strict texture publication should preserve page ownership");

            const GsVulkanServiceStatistics service =
                strict.vulkanRendererServiceStatistics();
            t.Equals(service.nearestCt32SpriteDrawsCompleted, 4ull,
                     "the service should execute repeat and both clamp classes");
            t.Equals(service.nearestCt32SpriteDrawsFailed, 0ull,
                     "classification rejection must not post service work");
            t.Equals(service.residentNearestCt32SpriteBatchesCompleted,
                     1ull,
                     "strict texture routing should complete one resident batch");
            t.Equals(service.largestResidentNearestCt32SpriteBatch,
                     4ull,
                     "shared texture reads should remain in one service batch");
            t.Equals(
                service.nearestCt32SpritePixelsExecuted,
                static_cast<uint64_t>(
                    prepared.boundsX1 - prepared.boundsX0) *
                    static_cast<uint64_t>(
                        prepared.boundsY1 - prepared.boundsY0) +
                    static_cast<uint64_t>(
                        secondPrepared.boundsX1 - secondPrepared.boundsX0) *
                    static_cast<uint64_t>(
                        secondPrepared.boundsY1 - secondPrepared.boundsY0) +
                    static_cast<uint64_t>(
                        clampedPrepared.boundsX1 -
                        clampedPrepared.boundsX0) *
                    static_cast<uint64_t>(
                        clampedPrepared.boundsY1 -
                        clampedPrepared.boundsY0) +
                    static_cast<uint64_t>(
                        regionClampedPrepared.boundsX1 -
                        regionClampedPrepared.boundsX0) *
                    static_cast<uint64_t>(
                        regionClampedPrepared.boundsY1 -
                        regionClampedPrepared.boundsY0),
                "the service should retain strict texture pixel accounting");
            t.Equals(service.spriteDrawsCompleted, 0ull,
                     "strict texture routing must not alias the flat sprite request");
            t.Equals(service.triangleDrawsCompleted, 0ull,
                     "strict texture routing must not alias the triangle request");
            t.Equals(service.validationErrors, 0u,
                     "strict texture routing should remain validation-clean");
            t.Equals(service.validationWarnings, 0u,
                     "strict texture routing should emit no validation warnings");
        });

        tc.Run("GS Vulkan strict depth CT32 survives ownership and frame checkpoints", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const std::array<GsDrawCommand, 6> commands =
                makeOrderedDepthCt32SpriteCommands(40'000u);
            std::array<GsVulkanDepthCt32Sprite, 6> prepared{};
            for (size_t index = 0u; index < commands.size(); ++index)
            {
                t.IsTrue(
                    prepareGsVulkanDepthCt32Sprite(
                        commands[index], prepared[index]).supported,
                    "every strict depth checkpoint should satisfy the narrow predicate");
            }
            const GsDrawCommand flat = makeCt32SpriteCommand(
                40'010u, 43u, 1u,
                {0u, 31u, 0u, 31u}, {0u, 0u},
                17u, 17u, 257u, 257u, 0x0D15EA5Eu);
            GsVulkanCt32Sprite flatPrepared{};
            t.IsTrue(prepareGsVulkanCt32Sprite(
                         flat, flatPrepared).supported,
                     "the strict depth checkpoint should retain one flat pipeline draw");

            const std::vector<uint8_t> initial =
                makeVramPattern(0x44504331u);
            std::vector<uint8_t> softwareVram = initial;
            std::vector<uint8_t> strictVram = initial;
            GSRegisters softwareRegisters{};
            GSRegisters strictRegisters{};
            configureCt32Display(softwareRegisters, 40u, 64u, 32u);
            configureCt32Display(strictRegisters, 40u, 64u, 32u);
            GS software;
            GS strict;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()),
                &softwareRegisters);
            strict.init(
                strictVram.data(),
                static_cast<uint32_t>(strictVram.size()),
                &strictRegisters);

            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig config =
                makeRendererServiceConfig(preflight);
            t.IsTrue(strict.configureVulkanRenderer(config),
                     "the strict depth checkpoint fixture should accept Vulkan configuration");
            if (!preflight.ready())
            {
                t.IsFalse(strict.setRendererMode(
                              GsRendererMode::GpuStrict),
                          "an unavailable host should decline strict depth checkpoints cleanly");
                return;
            }
            const GsVulkanDeviceReport *selected =
                preflight.selectedDevice();
            t.IsNotNull(selected,
                        "a ready strict depth preflight should select one device");
            if (!selected)
                return;
            t.IsTrue(selected->exactDepthCt32Sprite,
                     "the strict checkpoint device should expose the depth kernel");
            if (!selected->exactDepthCt32Sprite)
                return;

            t.IsTrue(strict.setRendererMode(
                         GsRendererMode::GpuStrict),
                     "the qualified host should enter strict depth mode");
            if (strict.rendererMode() != GsRendererMode::GpuStrict)
                return;
            strict.setBackendCountersEnabled(true);
            strict.resetBackendCounters();

            const auto drawPair = [&](const GsDrawCommand &command)
            {
                drawNearestCt32SpriteCommand(software, command);
                drawNearestCt32SpriteCommand(strict, command);
            };
            const auto compareFullVram = [&](const char *boundary)
            {
                const auto difference = std::mismatch(
                    softwareVram.begin(), softwareVram.end(),
                    strictVram.begin(), strictVram.end());
                if (difference.first == softwareVram.end())
                    return true;
                const size_t byteOffset = static_cast<size_t>(
                    difference.first - softwareVram.begin());
                t.IsTrue(
                    false,
                    std::string("strict depth VRAM mismatch after ") +
                        boundary + " at byte " +
                        std::to_string(byteOffset) + ", page " +
                        std::to_string(
                            byteOffset / GS_VRAM_PAGE_SIZE));
                return false;
            };

            drawPair(commands[0]);
            drawPair(commands[1]);
            t.IsTrue(strictVram == initial,
                     "overlapping strict depth draws should remain resident before observation");
            (void)software.getDebugSnapshot();
            (void)strict.getDebugSnapshot();
            if (!compareFullVram("overlapping Z24 debugger checkpoint"))
                return;

            drawPair(commands[2]);
            drawPair(commands[3]);
            software.writeRegister(GS_REG_FINISH, 0u);
            strict.writeRegister(GS_REG_FINISH, 0u);
            if (!compareFullVram("overlapping Z32 FINISH checkpoint"))
                return;

            drawPair(commands[0]);
            constexpr std::array<uint32_t, 4> uploadedPixels{{
                0xA0010203u,
                0xA1040506u,
                0xA2070809u,
                0xA30A0B0Cu,
            }};
            uploadCt32Pixels(
                software, prepared[0].framebufferBaseBlock,
                prepared[0].framebufferWidth,
                static_cast<uint16_t>(prepared[0].boundsX0),
                static_cast<uint16_t>(prepared[0].boundsY0),
                uploadedPixels);
            uploadCt32Pixels(
                strict, prepared[0].framebufferBaseBlock,
                prepared[0].framebufferWidth,
                static_cast<uint16_t>(prepared[0].boundsX0),
                static_cast<uint16_t>(prepared[0].boundsY0),
                uploadedPixels);
            drawPair(commands[1]);
            const std::vector<uint8_t> softwareReadback =
                readCt32Pixels(
                    software, prepared[1].framebufferBaseBlock,
                    prepared[1].framebufferWidth,
                    static_cast<uint16_t>(prepared[1].boundsX0),
                    static_cast<uint16_t>(prepared[1].boundsY0),
                    4u, 1u);
            const std::vector<uint8_t> strictReadback =
                readCt32Pixels(
                    strict, prepared[1].framebufferBaseBlock,
                    prepared[1].framebufferWidth,
                    static_cast<uint16_t>(prepared[1].boundsX0),
                    static_cast<uint16_t>(prepared[1].boundsY0),
                    4u, 1u);
            t.IsTrue(strictReadback == softwareReadback,
                     "strict depth should upload CPU-newer color and publish scoped readback");
            (void)software.getDebugSnapshot();
            (void)strict.getDebugSnapshot();
            if (!compareFullVram("transfer and scoped readback checkpoint"))
                return;

            drawPair(commands[0]);
            software.latchHostPresentationFrame();
            strict.latchHostPresentationFrame();
            std::vector<uint8_t> softwareFrame;
            std::vector<uint8_t> strictFrame;
            uint32_t softwareWidth = 0u;
            uint32_t softwareHeight = 0u;
            uint32_t strictWidth = 0u;
            uint32_t strictHeight = 0u;
            t.IsTrue(
                software.copyLatchedHostPresentationFrame(
                    softwareFrame, softwareWidth, softwareHeight),
                "the software depth checkpoint should publish one frame");
            t.IsTrue(
                strict.copyLatchedHostPresentationFrame(
                    strictFrame, strictWidth, strictHeight),
                "strict depth should publish one frame");
            t.Equals(strictWidth, softwareWidth,
                     "strict depth presentation should preserve frame width");
            t.Equals(strictHeight, softwareHeight,
                     "strict depth presentation should preserve frame height");
            t.IsTrue(strictFrame == softwareFrame,
                     "strict depth presentation bytes should match software");
            if (!compareFullVram("presentation checkpoint"))
                return;

            drawPair(commands[2]);
            configureNearestCt32SpriteCommand(software, flat);
            configureNearestCt32SpriteCommand(strict, flat);
            for (const GSVertex &vertex : flat.vertices())
            {
                writeNearestCt32SpriteVertex(software, vertex);
                writeNearestCt32SpriteVertex(strict, vertex);
            }
            drawPair(commands[3]);
            (void)software.getDebugSnapshot();
            (void)strict.getDebugSnapshot();
            if (!compareFullVram("depth-flat-depth pipeline checkpoint"))
                return;

            configureNearestCt32SpriteCommand(software, commands[0]);
            configureNearestCt32SpriteCommand(strict, commands[0]);
            writeNearestCt32SpriteVertex(
                software, commands[0].vertices()[0]);
            writeNearestCt32SpriteVertex(
                strict, commands[0].vertices()[0]);
            const GsReplayState softwareState =
                software.captureReplayState();
            const GsReplayState strictState =
                strict.captureReplayState();
            t.Equals(softwareState.vertexCount, 1,
                     "the software depth checkpoint should retain one vertex");
            t.Equals(strictState.vertexCount, 1,
                     "the strict depth checkpoint should retain one vertex");
            std::vector<uint8_t> softwareEncoded;
            std::vector<uint8_t> strictEncoded;
            std::string stateError;
            t.IsTrue(
                encodeGsReplayState(
                    softwareState, softwareEncoded, &stateError),
                "the software depth checkpoint should encode");
            t.IsTrue(
                encodeGsReplayState(
                    strictState, strictEncoded, &stateError),
                "the strict depth checkpoint should encode");
            t.IsTrue(strictEncoded == softwareEncoded,
                     "strict depth save-state should preserve frontend state");
            software.writeRegister(
                GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT));
            strict.writeRegister(
                GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT));
            t.IsTrue(software.restoreReplayState(softwareState),
                     "the software depth checkpoint should restore");
            t.IsTrue(strict.restoreReplayState(strictState),
                     "the strict depth checkpoint should restore");
            writeNearestCt32SpriteVertex(
                software, commands[0].vertices()[1]);
            writeNearestCt32SpriteVertex(
                strict, commands[0].vertices()[1]);
            (void)software.getDebugSnapshot();
            (void)strict.getDebugSnapshot();
            if (!compareFullVram("save-state restore checkpoint"))
                return;

            drawPair(commands[1]);
            software.reset();
            strict.reset();
            t.Equals(strict.rendererMode(), GsRendererMode::GpuStrict,
                     "reset should preserve strict depth mode and service");
            if (!compareFullVram("reset checkpoint"))
                return;
            drawPair(commands[0]);
            (void)software.getDebugSnapshot();
            (void)strict.getDebugSnapshot();
            if (!compareFullVram("post-reset depth draw"))
                return;

            drawPair(commands[2]);
            t.IsTrue(strict.setRendererMode(GsRendererMode::Software),
                     "the depth checkpoint should enter explicit software mode");
            if (!compareFullVram("strict-to-software switch"))
                return;
            drawPair(commands[3]);
            if (!compareFullVram("explicit software depth draw"))
                return;
            configureFlatCt32Draws(software, 44u, 1u);
            configureFlatCt32Draws(strict, 44u, 1u);
            const uint64_t feedbackTex0 =
                static_cast<uint64_t>(44u * 32u) |
                (1ull << 14u) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 20u) |
                (5ull << 26u) |
                (5ull << 30u) |
                (1ull << 34u) |
                (1ull << 35u);
            software.writeRegister(GS_REG_TEX0_1, feedbackTex0);
            strict.writeRegister(GS_REG_TEX0_1, feedbackTex0);
            software.beginRenderBatch();
            strict.beginRenderBatch();
            drawRecursiveCt32Sprite(
                software, 2u * 16u, 3u * 16u,
                10u * 16u, 11u * 16u, 0x80808080u);
            drawRecursiveCt32Sprite(
                strict, 2u * 16u, 3u * 16u,
                10u * 16u, 11u * 16u, 0x80808080u);
            software.endRenderBatch();
            strict.endRenderBatch();
            if (!compareFullVram("software feedback snapshot interval"))
                return;
            t.IsTrue(strict.setRendererMode(GsRendererMode::GpuStrict),
                     "the depth checkpoint should restore strict mode");
            drawPair(commands[0]);
            (void)software.getDebugSnapshot();
            (void)strict.getDebugSnapshot();
            if (!compareFullVram("software-to-strict switch"))
                return;

            drawPair(commands[0]);
            software.flushRenderBatch();
            strict.flushRenderBatch();
            if (!compareFullVram("explicit flush checkpoint"))
                return;

            drawPair(commands[0]);
            const uint64_t indexedTex0 =
                (1ull << 14u) |
                (static_cast<uint64_t>(GS_PSM_T8) << 20u) |
                (5ull << 26u) |
                (5ull << 30u) |
                (1ull << 34u) |
                (static_cast<uint64_t>(
                     prepared[0].depthBaseBlock) << 37u) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 51u) |
                (1ull << 61u);
            software.writeRegister(GS_REG_TEX0_1, indexedTex0);
            strict.writeRegister(GS_REG_TEX0_1, indexedTex0);
            if (!compareFullVram("CLUT hazard checkpoint"))
                return;
            t.IsTrue(
                strict.vulkanRendererBackendStatistics()
                        .pageOwnership.gpuNewerPages > 0u,
                "the scoped CLUT read should leave unrelated color writes resident");
            (void)software.getDebugSnapshot();
            (void)strict.getDebugSnapshot();
            if (!compareFullVram("final debugger checkpoint"))
                return;

            const GsBackendCounters counters = strict.backendCounters();
            t.Equals(counters.commands, 19ull,
                     "the strict depth checkpoint should assemble its exact draw count");
            t.Equals(counters.acceleratedCommands, 17ull,
                     "every strict-mode depth or flat draw should reach Vulkan");
            t.Equals(counters.softwareCommands, 2ull,
                     "only the explicit software interval should use the CPU");
            t.Equals(counters.fallbackCommands, 0ull,
                     "strict depth checkpoints should have no implicit fallback");
            t.Equals(counters.strictFailures, 0ull,
                     "every strict depth checkpoint should remain supported");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::Explicit)] >= 1ull,
                "explicit flush should retain its named boundary");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::Transfer)] >= 1ull,
                "the host upload should retain its transfer boundary");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::CpuReadback)] >= 1ull,
                "the scoped readback should retain its named boundary");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::ClutHazard)] >= 1ull,
                "the depth-written CLUT page should retain its hazard boundary");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::FeedbackSnapshot)] >= 1ull,
                "the software interval should retain its feedback snapshot boundary");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::Finish)] >= 1ull,
                "FINISH should remain an explicit checkpoint");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::PresentationLatch)] >= 1ull,
                "presentation should remain an explicit checkpoint");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::DebuggerObservation)] >= 1ull,
                "debugger snapshots should retain their observation boundary");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::SaveLoad)] >= 2ull,
                "capture and restore should each cross the save-state boundary");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::Reset)] >= 1ull,
                "reset should retain its named boundary");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::BackendSwitch)] >= 2ull,
                "the explicit software interval should cross both backend boundaries");

            const GsVulkanRasterBackendStatistics backend =
                strict.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsCompleted, 17ull,
                     "every strict depth checkpoint draw should complete once");
            t.Equals(backend.committedGpuCommands, 17ull,
                     "every strict depth checkpoint should publish once");
            t.Equals(backend.residentCommands, 17ull,
                     "every accelerated checkpoint should remain resident");
            t.Equals(backend.pipelineChangeDrains, 2ull,
                     "depth-flat-depth should retain two pipeline drains");
            t.Equals(backend.coherency.rejectedTransitions, 0ull,
                     "strict depth checkpoints should preserve page ownership");
            t.Equals(backend.pageOwnership.gpuNewerPages, size_t{0u},
                     "the final debugger observation should publish all GPU writers");

            const GsVulkanServiceStatistics service =
                strict.vulkanRendererServiceStatistics();
            t.Equals(service.depthCt32SpriteDrawsCompleted, 16ull,
                     "the service should execute every strict depth draw");
            t.Equals(service.depthCt32SpriteDrawsFailed, 0ull,
                     "strict depth checkpoints should not fail service work");
            t.Equals(service.residentDepthCt32SpriteBatchesCompleted,
                     14ull,
                     "only the two overlapping pairs should share depth batches");
            t.Equals(service.largestResidentDepthCt32SpriteBatch, 2ull,
                     "overlapping Z24 and Z32 pairs should establish batch size two");
            t.Equals(service.spriteDrawsCompleted, 1ull,
                     "the pipeline transition should execute one flat sprite");
            t.IsTrue(service.pageUploadOperationsCompleted >= 2ull,
                     "strict depth should upload initial and CPU-newer pages");
            t.IsTrue(service.pageDownloadOperationsCompleted >= 1ull,
                     "strict observations should publish GPU-newer pages");
            t.Equals(service.fenceTimeouts, 0ull,
                     "bounded strict depth checkpoints should not time out");
            t.Equals(service.validationErrors, 0u,
                     "strict depth checkpoints should remain validation-clean");
            t.Equals(service.validationWarnings, 0u,
                     "strict depth checkpoints should emit no validation warnings");
        });

        tc.Run("GS Vulkan source-copy alpha survives Verify strict and Hybrid", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const GsDrawCommand below =
                makeSourceCopyAlphaCt32SpriteCommand(
                    40'900u, 20u, 1u,
                    {0u, 63u, 0u, 63u}, {0u, 0u},
                    0u, 0u, 64u * 16u, 64u * 16u,
                    0x20000000u);
            const GsDrawCommand threshold =
                makeSourceCopyAlphaCt32SpriteCommand(
                    40'901u, 100u, 2u,
                    {0u, 127u, 0u, 63u}, {0u, 0u},
                    0u, 0u, 128u * 16u, 64u * 16u,
                    0x40000000u);
            const GsDrawCommand retained =
                makeSourceCopyAlphaCt32SpriteCommand(
                    40'902u, 300u, 8u,
                    {0u, 511u, 0u, 447u},
                    {1792u * 16u, 1824u * 16u},
                    1792u * 16u - 8u,
                    1824u * 16u - 8u,
                    (1792u + 32u) * 16u - 8u,
                    (1824u + 448u) * 16u - 8u,
                    0x08000000u);
            const auto pixels = [](const GsDrawCommand &command)
            {
                return static_cast<uint64_t>(
                           command.bounds().x1 - command.bounds().x0) *
                       static_cast<uint64_t>(
                           command.bounds().y1 - command.bounds().y0);
            };
            t.Equals(pixels(below), 4'096ull,
                     "the small source-copy fixture should stay below policy");
            t.Equals(pixels(threshold), 8'192ull,
                     "the threshold fixture should meet policy exactly");
            t.Equals(pixels(retained), 14'336ull,
                     "the retained fixture should reproduce one RAC1 strip");

            const std::vector<uint8_t> initial =
                makeVramPattern(0x53434152u);
            std::vector<uint8_t> softwareVram = initial;
            std::vector<uint8_t> acceleratedVram = initial;
            GS software;
            GS accelerated;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()), nullptr);
            accelerated.init(
                acceleratedVram.data(),
                static_cast<uint32_t>(acceleratedVram.size()), nullptr);

            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig serviceConfig =
                makeRendererServiceConfig(preflight);
            GsVulkanRasterBackendConfig config{};
            t.Equals(config.minimumHybridSourceCopyAlphaSpritePixels,
                     8'192ull,
                     "the integrated fixture should use measured policy");
            t.IsTrue(accelerated.configureVulkanRenderer(
                         serviceConfig, config),
                     "source-copy integration should accept configuration");
            if (!preflight.ready())
            {
                t.IsFalse(accelerated.setRendererMode(
                              GsRendererMode::Verify),
                          "an unavailable host should decline Verify");
                return;
            }
            const GsVulkanDeviceReport *selected =
                preflight.selectedDevice();
            t.IsNotNull(selected,
                        "a ready source-copy preflight should select a device");
            if (!selected)
                return;
            t.IsTrue(selected->exactVramStorage,
                     "the selected device should expose exact raw VRAM");
            if (!selected->exactVramStorage)
                return;

            t.IsTrue(accelerated.setRendererMode(GsRendererMode::Verify),
                     "the qualified host should enter Verify");
            drawNearestCt32SpriteCommand(software, below);
            drawNearestCt32SpriteCommand(accelerated, below);
            (void)software.getDebugSnapshot();
            (void)accelerated.getDebugSnapshot();
            t.IsTrue(acceleratedVram == softwareVram,
                     "source-copy Verify should match all software VRAM");

            t.IsTrue(accelerated.setRendererMode(GsRendererMode::GpuStrict),
                     "the qualified host should enter strict mode");
            drawNearestCt32SpriteCommand(software, threshold);
            drawNearestCt32SpriteCommand(accelerated, threshold);
            (void)software.getDebugSnapshot();
            (void)accelerated.getDebugSnapshot();
            t.IsTrue(acceleratedVram == softwareVram,
                     "resident strict source-copy should publish exact VRAM");

            t.IsTrue(accelerated.setRendererMode(GsRendererMode::Hybrid),
                     "the qualified host should enter Hybrid");
            accelerated.setBackendCountersEnabled(true);
            accelerated.resetBackendCounters();
            for (const GsDrawCommand *command :
                 std::array<const GsDrawCommand *, 3>{
                     &below, &threshold, &retained})
            {
                drawNearestCt32SpriteCommand(software, *command);
                drawNearestCt32SpriteCommand(accelerated, *command);
            }
            (void)software.getDebugSnapshot();
            (void)accelerated.getDebugSnapshot();
            t.IsTrue(acceleratedVram == softwareVram,
                     "mixed source-copy Hybrid routing should remain exact");

            const GsBackendCounters counters =
                accelerated.backendCounters();
            t.Equals(counters.commands, 3ull,
                     "Hybrid should classify the complete source-copy stream");
            t.Equals(counters.softwareCommands, 1ull,
                     "only below-threshold source-copy work should use CPU");
            t.Equals(counters.fallbackCommands, 1ull,
                     "the small draw should retain one cost fallback");
            t.Equals(counters.acceleratedCommands, 2ull,
                     "threshold and retained source-copy work should use GPU");
            t.Equals(counters.decisions[static_cast<size_t>(
                         GsFallbackReason::CostModel)],
                     1ull,
                     "the mixed stream should expose one cost decision");

            const GsVulkanRasterBackendStatistics backend =
                accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsCompleted, 4ull,
                     "Verify strict and two Hybrid GPU draws should complete");
            t.Equals(backend.verifiedCommands, 1ull,
                     "the first source-copy draw should be verified once");
            t.Equals(backend.residentCommands, 3ull,
                     "strict and Hybrid source-copy work should stay resident");
            t.Equals(backend.coherency.rejectedTransitions, 0ull,
                     "source-copy routing should preserve page ownership");
            t.Equals(backend.pageOwnership.gpuNewerPages, size_t{0u},
                     "the final observation should publish every GPU page");

            const GsVulkanServiceStatistics service =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(service.spriteDrawsCompleted, 4ull,
                     "the flat service should execute every accelerated copy");
            t.Equals(service.spriteDrawsFailed, 0ull,
                     "source-copy service work should not fail");
            t.Equals(service.residentSpriteBatchesCompleted, 2ull,
                     "strict and two-draw Hybrid work should form two batches");
            t.Equals(service.largestResidentSpriteBatch, 2ull,
                     "disjoint threshold and retained work should batch together");
            t.Equals(service.validationErrors, 0u,
                     "source-copy integration should remain validation-clean");
            t.Equals(service.validationWarnings, 0u,
                     "source-copy integration should emit no warnings");
        });

        tc.Run("GS Vulkan Hybrid keeps retained depth on CPU and orders admitted work", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            constexpr uint64_t thresholdPixels = 262'144u;
            const GsDrawCommand retainedTitle =
                makeDepthCt32SpriteCommand(
                    41'000u, 112u, 8u, 216u,
                    GS_PSM_Z24, false, 1u,
                    {0u, 511u, 0u, 415u},
                    {1792u * 16u, 1840u * 16u},
                    1792u * 16u - 8u, 1840u * 16u - 8u,
                    (1792u + 32u) * 16u - 8u,
                    (1840u + 416u) * 16u - 8u,
                    0x80000000u, 0u);
            const GsDrawCommand admitted =
                makeDepthCt32SpriteCommand(
                    41'001u, 0u, 8u, 256u,
                    GS_PSM_Z32, false, 1u,
                    {0u, 511u, 0u, 511u}, {0u, 0u},
                    0u, 0u, 512u * 16u, 512u * 16u,
                    0xC0A08060u, 0x55667788u);
            GsVulkanDepthCt32Sprite retainedPrepared{};
            GsVulkanDepthCt32Sprite admittedPrepared{};
            t.IsTrue(prepareGsVulkanDepthCt32Sprite(
                         retainedTitle, retainedPrepared).supported,
                     "the retained Hybrid title fixture should be exact");
            t.IsTrue(prepareGsVulkanDepthCt32Sprite(
                         admitted, admittedPrepared).supported,
                     "the admitted Hybrid depth fixture should be exact");
            const auto pixels = [](const GsVulkanDepthCt32Sprite &sprite)
            {
                return static_cast<uint64_t>(
                           sprite.boundsX1 - sprite.boundsX0) *
                       static_cast<uint64_t>(
                           sprite.boundsY1 - sprite.boundsY0);
            };
            t.Equals(pixels(retainedPrepared), 13'312ull,
                     "the retained fixture should preserve title work");
            t.Equals(pixels(admittedPrepared), thresholdPixels,
                     "the broad fixture should meet depth policy exactly");

            const std::vector<uint8_t> initial =
                makeVramPattern(0x44485942u);
            std::vector<uint8_t> softwareVram = initial;
            std::vector<uint8_t> defaultVram = initial;
            std::vector<uint8_t> forcedVram = initial;
            GS software;
            GS defaultHybrid;
            GS forcedHybrid;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()), nullptr);
            defaultHybrid.init(
                defaultVram.data(),
                static_cast<uint32_t>(defaultVram.size()), nullptr);
            forcedHybrid.init(
                forcedVram.data(),
                static_cast<uint32_t>(forcedVram.size()), nullptr);

            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig serviceConfig =
                makeRendererServiceConfig(preflight);
            GsVulkanRasterBackendConfig defaultConfig{};
            GsVulkanRasterBackendConfig forcedConfig{};
            forcedConfig.minimumHybridDepthCt32SpritePixels = 0u;
            t.Equals(defaultConfig.minimumHybridDepthCt32SpritePixels,
                     thresholdPixels,
                     "the integrated depth fixture should use measured policy");
            t.IsTrue(defaultHybrid.configureVulkanRenderer(
                         serviceConfig, defaultConfig),
                     "the default depth Hybrid should accept configuration");
            t.IsTrue(forcedHybrid.configureVulkanRenderer(
                         serviceConfig, forcedConfig),
                     "the forced depth Hybrid should accept configuration");
            if (!preflight.ready())
            {
                t.IsFalse(defaultHybrid.setRendererMode(
                              GsRendererMode::Hybrid),
                          "an unavailable host should decline default Hybrid");
                t.IsFalse(forcedHybrid.setRendererMode(
                              GsRendererMode::Hybrid),
                          "an unavailable host should decline forced Hybrid");
                return;
            }
            const GsVulkanDeviceReport *selected =
                preflight.selectedDevice();
            t.IsNotNull(selected,
                        "a ready depth Hybrid preflight should select a device");
            if (!selected)
                return;
            t.IsTrue(selected->exactDepthCt32Sprite,
                     "the selected device should expose exact depth sprites");
            if (!selected->exactDepthCt32Sprite)
                return;

            t.IsTrue(defaultHybrid.setRendererMode(
                         GsRendererMode::Hybrid),
                     "the qualified host should enter default depth Hybrid");
            t.IsTrue(forcedHybrid.setRendererMode(
                         GsRendererMode::Hybrid),
                     "the qualified host should enter forced depth Hybrid");
            if (defaultHybrid.rendererMode() != GsRendererMode::Hybrid ||
                forcedHybrid.rendererMode() != GsRendererMode::Hybrid)
            {
                return;
            }
            defaultHybrid.setBackendCountersEnabled(true);
            defaultHybrid.resetBackendCounters();
            forcedHybrid.setBackendCountersEnabled(true);
            forcedHybrid.resetBackendCounters();

            const auto drawAll = [&](const GsDrawCommand &command)
            {
                drawNearestCt32SpriteCommand(software, command);
                drawNearestCt32SpriteCommand(defaultHybrid, command);
                drawNearestCt32SpriteCommand(forcedHybrid, command);
            };
            drawAll(retainedTitle);
            drawAll(admitted);
            drawAll(retainedTitle);
            (void)software.getDebugSnapshot();
            (void)defaultHybrid.getDebugSnapshot();
            (void)forcedHybrid.getDebugSnapshot();
            t.IsTrue(defaultVram == softwareVram,
                     "default mixed depth routing should match full software VRAM");
            t.IsTrue(forcedVram == softwareVram,
                     "forced resident depth routing should match full software VRAM");

            const GsBackendCounters defaultCounters =
                defaultHybrid.backendCounters();
            t.Equals(defaultCounters.commands, 3ull,
                     "default Hybrid should classify all three depth draws");
            t.Equals(defaultCounters.softwareCommands, 2ull,
                     "both retained title draws should use software");
            t.Equals(defaultCounters.fallbackCommands, 2ull,
                     "both retained draws should record cost fallback");
            t.Equals(defaultCounters.acceleratedCommands, 1ull,
                     "only measured broad work should use default Vulkan");
            t.Equals(defaultCounters.decisions[static_cast<size_t>(
                         GsFallbackReason::CostModel)],
                     2ull,
                     "default Hybrid should retain both cost decisions");
            t.Equals(defaultCounters.decisions[static_cast<size_t>(
                         GsFallbackReason::Supported)],
                     1ull,
                     "default Hybrid should retain the admitted decision");

            const GsBackendCounters forcedCounters =
                forcedHybrid.backendCounters();
            t.Equals(forcedCounters.commands, 3ull,
                     "forced Hybrid should classify all depth draws");
            t.Equals(forcedCounters.softwareCommands, 0ull,
                     "zero threshold should avoid software depth work");
            t.Equals(forcedCounters.fallbackCommands, 0ull,
                     "zero threshold should avoid cost fallback");
            t.Equals(forcedCounters.acceleratedCommands, 3ull,
                     "zero threshold should accelerate the complete sequence");

            const GsVulkanRasterBackendStatistics defaultBackend =
                defaultHybrid.vulkanRendererBackendStatistics();
            t.Equals(defaultBackend.commandsCompleted, 1ull,
                     "default Hybrid should complete one GPU depth draw");
            t.Equals(defaultBackend.committedGpuCommands, 1ull,
                     "default Hybrid should commit one GPU depth draw");
            t.Equals(defaultBackend.coherency.rejectedTransitions, 0ull,
                     "mixed default depth ownership should remain valid");
            const GsVulkanRasterBackendStatistics forcedBackend =
                forcedHybrid.vulkanRendererBackendStatistics();
            t.Equals(forcedBackend.commandsCompleted, 3ull,
                     "forced Hybrid should complete every depth draw");
            t.Equals(forcedBackend.committedGpuCommands, 3ull,
                     "forced Hybrid should commit every depth draw");
            t.Equals(forcedBackend.residentBatchesCompleted, 1ull,
                     "the ordered depth service should retain one batch");
            t.Equals(forcedBackend.largestResidentBatch, 3ull,
                     "the ordered batch should retain all dependencies");
            t.Equals(forcedBackend.coherency.rejectedTransitions, 0ull,
                     "forced depth ownership should remain valid");

            const GsVulkanServiceStatistics defaultService =
                defaultHybrid.vulkanRendererServiceStatistics();
            t.Equals(defaultService.depthCt32SpriteDrawsCompleted, 1ull,
                     "default Hybrid should execute one depth dispatch");
            const GsVulkanServiceStatistics forcedService =
                forcedHybrid.vulkanRendererServiceStatistics();
            t.Equals(forcedService.depthCt32SpriteDrawsCompleted, 3ull,
                     "forced Hybrid should execute all depth dispatches");
            t.Equals(
                forcedService.residentDepthCt32SpriteBatchesCompleted,
                1ull,
                "forced Hybrid should use one ordered service batch");
            t.Equals(forcedService.largestResidentDepthCt32SpriteBatch,
                     3ull,
                     "the service should retain the complete depth batch");
            t.Equals(forcedService.validationErrors, 0u,
                     "depth Hybrid transitions should remain validation-clean");
            t.Equals(forcedService.validationWarnings, 0u,
                     "depth Hybrid transitions should emit no warnings");
        });

        tc.Run("GS Vulkan strict linear CT32 survives frame and reset checkpoints", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const GsDrawCommand retained =
                makeLinearCt32RepeatSpriteCommand(
                    103u, 0u, 8u, 3584u, 8u, 10u, 10u,
                    {0u, 511u, 0u, 447u},
                    {28672u, 29184u},
                    {28664u, 29176u},
                    {29176u, 36344u},
                    {0u, 512u},
                    {0u, 6656u});
            const GsDrawCommand retainedClamp =
                makeLinearCt32SpriteCommand(
                    104u, 0u, 8u, 3584u, 8u, 10u, 10u,
                    {0u, 511u, 0u, 447u},
                    {28672u, 29184u},
                    {28664u, 29176u},
                    {29176u, 36344u},
                    {0u, 512u},
                    {0u, 6656u},
                    1u, 1u);
            const auto makeSmall = [](uint64_t sequence,
                                      uint32_t framebufferPage,
                                      uint8_t wrapU = 0u,
                                      uint8_t wrapV = 0u)
            {
                return makeLinearCt32SpriteCommand(
                    sequence, framebufferPage, 2u,
                    512u, 2u, 6u, 5u,
                    {3u, 12u, 2u, 13u}, {128u, 96u},
                    {120u, 376u}, {88u, 344u},
                    {0u, 249u}, {0u, 137u}, wrapU, wrapV);
            };
            const GsDrawCommand smallA = makeSmall(105u, 40u);
            const GsDrawCommand smallB = makeSmall(106u, 41u, 1u, 1u);
            for (const GsDrawCommand *command :
                 std::array<const GsDrawCommand *, 4>{{
                     &retained, &retainedClamp, &smallA, &smallB,
                 }})
            {
                GsVulkanLinearCt32Sprite prepared{};
                t.IsTrue(
                    prepareGsVulkanLinearCt32Sprite(
                        *command, prepared).supported,
                    "every strict checkpoint command should satisfy the linear predicate");
            }

            const std::vector<uint8_t> initial =
                makeVramPattern(0x4C535432u);
            std::vector<uint8_t> softwareVram = initial;
            std::vector<uint8_t> strictVram = initial;
            GSRegisters softwareRegisters{};
            GSRegisters strictRegisters{};
            configureCt32Display(softwareRegisters, 0u);
            configureCt32Display(strictRegisters, 0u);
            GS software;
            GS strict;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()),
                &softwareRegisters);
            strict.init(
                strictVram.data(),
                static_cast<uint32_t>(strictVram.size()),
                &strictRegisters);

            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig config =
                makeRendererServiceConfig(preflight);
            t.IsTrue(strict.configureVulkanRenderer(config),
                     "the strict linear checkpoint fixture should accept Vulkan configuration");
            if (!preflight.ready())
            {
                t.IsFalse(strict.setRendererMode(
                              GsRendererMode::GpuStrict),
                          "an unavailable host should decline strict linear checkpoints cleanly");
                return;
            }
            const GsVulkanDeviceReport *selected =
                preflight.selectedDevice();
            t.IsNotNull(selected,
                        "a ready strict linear preflight should select one device");
            if (!selected)
                return;
            t.IsTrue(selected->exactLinearCt32Sprite,
                     "the strict checkpoint device should expose the linear kernel");
            if (!selected->exactLinearCt32Sprite)
                return;

            t.IsTrue(strict.setRendererMode(
                         GsRendererMode::GpuStrict),
                     "the qualified host should enter strict linear mode");
            if (strict.rendererMode() != GsRendererMode::GpuStrict)
                return;
            strict.setBackendCountersEnabled(true);
            strict.resetBackendCounters();

            const auto compareFullVram = [&](const char *boundary)
            {
                const auto difference = std::mismatch(
                    softwareVram.begin(), softwareVram.end(),
                    strictVram.begin(), strictVram.end());
                if (difference.first == softwareVram.end())
                    return true;
                const size_t byteOffset = static_cast<size_t>(
                    difference.first - softwareVram.begin());
                t.IsTrue(
                    false,
                    std::string("strict linear VRAM mismatch after ") +
                        boundary + " at byte " +
                        std::to_string(byteOffset));
                return false;
            };

            drawNearestCt32SpriteCommand(software, smallA);
            drawNearestCt32SpriteCommand(strict, smallA);
            drawNearestCt32SpriteCommand(software, smallB);
            drawNearestCt32SpriteCommand(strict, smallB);
            t.IsTrue(strictVram == initial,
                     "two pending strict linear draws should not publish CPU VRAM");
            (void)software.getDebugSnapshot();
            (void)strict.getDebugSnapshot();
            if (!compareFullVram("shared-read debugger checkpoint"))
                return;

            const std::vector<uint8_t> beforeRetained = strictVram;
            drawNearestCt32SpriteCommand(software, retainedClamp);
            drawNearestCt32SpriteCommand(strict, retainedClamp);
            t.IsTrue(strictVram == beforeRetained,
                     "the retained title draw should stay resident before presentation");
            software.latchHostPresentationFrame();
            strict.latchHostPresentationFrame();
            std::vector<uint8_t> softwareFrame;
            std::vector<uint8_t> strictFrame;
            uint32_t softwareWidth = 0u;
            uint32_t softwareHeight = 0u;
            uint32_t strictWidth = 0u;
            uint32_t strictHeight = 0u;
            t.IsTrue(
                software.copyLatchedHostPresentationFrame(
                    softwareFrame, softwareWidth, softwareHeight),
                "the software linear checkpoint should publish one frame");
            t.IsTrue(
                strict.copyLatchedHostPresentationFrame(
                    strictFrame, strictWidth, strictHeight),
                "strict linear presentation should publish one frame");
            t.Equals(strictWidth, softwareWidth,
                     "strict linear presentation should preserve frame width");
            t.Equals(strictHeight, softwareHeight,
                     "strict linear presentation should preserve frame height");
            t.IsTrue(strictFrame == softwareFrame,
                     "strict linear presentation bytes should match software");
            if (!compareFullVram("presentation checkpoint"))
                return;

            drawNearestCt32SpriteCommand(software, smallA);
            drawNearestCt32SpriteCommand(strict, smallA);
            software.writeRegister(GS_REG_FINISH, 0u);
            strict.writeRegister(GS_REG_FINISH, 0u);
            if (!compareFullVram("FINISH checkpoint"))
                return;

            drawNearestCt32SpriteCommand(software, smallA);
            drawNearestCt32SpriteCommand(strict, smallA);
            configureFlatCt32Draws(software, 42u, 1u);
            configureFlatCt32Draws(strict, 42u, 1u);
            drawFlatCt32Sprite(
                software, 3u * 16u, 2u * 16u,
                15u * 16u, 11u * 16u, 0xD0112233u);
            drawFlatCt32Sprite(
                strict, 3u * 16u, 2u * 16u,
                15u * 16u, 11u * 16u, 0xD0112233u);
            drawNearestCt32SpriteCommand(software, smallB);
            drawNearestCt32SpriteCommand(strict, smallB);
            (void)software.getDebugSnapshot();
            (void)strict.getDebugSnapshot();
            if (!compareFullVram("pipeline-change checkpoint"))
                return;

            drawNearestCt32SpriteCommand(software, smallA);
            drawNearestCt32SpriteCommand(strict, smallA);
            const GsReplayState softwareState =
                software.captureReplayState();
            const GsReplayState strictState =
                strict.captureReplayState();
            std::vector<uint8_t> softwareEncoded;
            std::vector<uint8_t> strictEncoded;
            std::string stateError;
            t.IsTrue(
                encodeGsReplayState(
                    softwareState, softwareEncoded, &stateError),
                "the software linear checkpoint should encode");
            t.IsTrue(
                encodeGsReplayState(
                    strictState, strictEncoded, &stateError),
                "the strict linear checkpoint should encode");
            t.IsTrue(strictEncoded == softwareEncoded,
                     "strict linear save-state should preserve frontend state");
            if (!compareFullVram("save-state checkpoint"))
                return;

            drawNearestCt32SpriteCommand(software, smallB);
            drawNearestCt32SpriteCommand(strict, smallB);
            software.reset();
            strict.reset();
            t.Equals(strict.rendererMode(), GsRendererMode::GpuStrict,
                     "reset should preserve strict linear mode and service");
            if (!compareFullVram("reset checkpoint"))
                return;

            drawNearestCt32SpriteCommand(software, retained);
            drawNearestCt32SpriteCommand(strict, retained);
            (void)software.getDebugSnapshot();
            (void)strict.getDebugSnapshot();
            if (!compareFullVram("post-reset repeat title draw"))
                return;

            const GsBackendCounters counters = strict.backendCounters();
            t.Equals(counters.commands, 10ull,
                     "the checkpoint stream should assemble nine linear and one flat draw");
            t.Equals(counters.acceleratedCommands, 10ull,
                     "every checkpoint draw should reach strict Vulkan");
            t.Equals(counters.softwareCommands, 0ull,
                     "strict linear checkpoints must not use software fallback");
            t.Equals(counters.fallbackCommands, 0ull,
                     "strict linear checkpoints should have no fallback decision");
            t.Equals(counters.strictFailures, 0ull,
                     "every checkpoint command should remain supported");
            t.Equals(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::Reset)],
                1ull,
                "reset should drain pending strict linear work once");

            const GsVulkanRasterBackendStatistics backend =
                strict.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsCompleted, 10ull,
                     "every strict checkpoint draw should complete once");
            t.Equals(backend.committedGpuCommands, 10ull,
                     "every strict checkpoint draw should publish once");
            t.Equals(backend.residentCommands, 10ull,
                     "all checkpoint draws should use resident execution");
            t.Equals(backend.pipelineChangeDrains, 2ull,
                     "linear/flat/linear transitions should drain twice");
            t.Equals(backend.coherency.rejectedTransitions, 0ull,
                     "strict linear checkpoints should preserve page ownership");
            t.Equals(backend.pageOwnership.gpuNewerPages, size_t{0u},
                     "the final debugger checkpoint should publish all GPU writers");

            const GsVulkanServiceStatistics service =
                strict.vulkanRendererServiceStatistics();
            t.Equals(service.linearCt32SpriteDrawsCompleted, 9ull,
                     "the service should execute every strict linear draw");
            t.Equals(service.linearCt32SpriteDrawsFailed, 0ull,
                     "strict checkpoint routing should not fail linear work");
            t.Equals(service.residentLinearCt32SpriteBatchesCompleted,
                     8ull,
                     "only the initial shared-read pair should share a linear batch");
            t.Equals(service.largestResidentLinearCt32SpriteBatch, 2ull,
                     "the initial disjoint pair should establish batch size two");
            t.Equals(service.spriteDrawsCompleted, 1ull,
                     "the pipeline transition should execute one flat sprite");
            t.Equals(service.nearestCt32SpriteDrawsCompleted, 0ull,
                     "strict linear checkpoints must not alias nearest textures");
            t.Equals(service.validationErrors, 0u,
                     "strict linear checkpoints should remain validation-clean");
            t.Equals(service.validationWarnings, 0u,
                     "strict linear checkpoints should emit no validation warnings");
        });

        tc.Run("GS Vulkan nearest CT32 Verify survives every frame checkpoint", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            constexpr uint32_t framebufferPage = 40u;
            constexpr uint32_t framebufferBlock = framebufferPage * 32u;
            constexpr uint32_t textureBlock = 64u;
            constexpr uint32_t bufferWidth = 2u;
            const GsDrawCommand command = makeNearestCt32SpriteCommand(
                95u, framebufferPage, bufferWidth,
                textureBlock, bufferWidth, 6u, 5u,
                {6u, 15u, 5u, 12u}, {32u, 16u},
                {352u, 96u}, {48u, 304u},
                {480u, 224u}, {64u, 320u}, 3u, 3u,
                15u, 70u, 8u, 40u);
            GsVulkanNearestCt32Sprite prepared{};
            t.IsTrue(
                prepareGsVulkanNearestCt32Sprite(
                    command, prepared).supported,
                "the REGION_REPEAT transition fixture should satisfy the nearest texture predicate");

            std::vector<uint8_t> softwareVram =
                makeVramPattern(0x54584350u);
            std::vector<uint8_t> acceleratedVram = softwareVram;
            GSRegisters softwareRegisters{};
            GSRegisters acceleratedRegisters{};
            configureCt32Display(
                softwareRegisters, framebufferPage, 64u, 32u);
            configureCt32Display(
                acceleratedRegisters, framebufferPage, 64u, 32u);
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
            ScopedArtifactDirectory artifacts;
            t.IsTrue(accelerated.configureVulkanRenderer(
                         config, artifacts.path.string()),
                     "the transition fixture should accept Vulkan configuration");
            if (!preflight.ready())
            {
                t.IsFalse(accelerated.setRendererMode(
                              GsRendererMode::Verify),
                          "an unavailable host should skip texture checkpoints cleanly");
                return;
            }
            const GsVulkanDeviceReport *selected =
                preflight.selectedDevice();
            t.IsNotNull(selected,
                        "a ready checkpoint preflight should select one device");
            if (!selected)
                return;
            t.IsTrue(selected->exactNearestCt32Sprite,
                     "the checkpoint device should expose the exact texture kernel");
            if (!selected->exactNearestCt32Sprite)
                return;

            t.IsTrue(accelerated.setRendererMode(
                         GsRendererMode::Verify),
                     "the qualified host should create texture Verify checkpoints");
            if (accelerated.rendererMode() != GsRendererMode::Verify)
                return;
            accelerated.setBackendCountersEnabled(true);
            accelerated.resetBackendCounters();

            const auto compareFullVram = [&](const char *boundary) -> bool
            {
                const auto difference = std::mismatch(
                    softwareVram.begin(), softwareVram.end(),
                    acceleratedVram.begin(), acceleratedVram.end());
                if (difference.first == softwareVram.end())
                    return true;
                const size_t byteOffset = static_cast<size_t>(
                    difference.first - softwareVram.begin());
                t.IsTrue(
                    false,
                    std::string("texture Verify VRAM mismatch after ") +
                        boundary + ", byte " +
                        std::to_string(byteOffset) + ", page " +
                        std::to_string(
                            byteOffset / GS_VRAM_PAGE_SIZE));
                return false;
            };

            drawNearestCt32SpriteCommand(software, command);
            drawNearestCt32SpriteCommand(accelerated, command);
            if (!compareFullVram("initial eligible draw"))
                return;

            drawFlatCt32Point(
                software, 10u * 16u + 32u,
                8u * 16u + 16u, 0xC0112233u);
            drawFlatCt32Point(
                accelerated, 10u * 16u + 32u,
                8u * 16u + 16u, 0xC0112233u);
            if (!compareFullVram("eligible to fallback adjacency"))
                return;
            drawNearestCt32SpriteCommand(software, command);
            drawNearestCt32SpriteCommand(accelerated, command);
            if (!compareFullVram("eligible fallback eligible adjacency"))
                return;

            constexpr std::array<uint32_t, 4> sourcePixels{{
                0xD0010203u,
                0xD1040506u,
                0xD2070809u,
                0xD30A0B0Cu,
            }};
            uploadCt32Pixels(
                software, textureBlock, bufferWidth,
                70u, 40u, sourcePixels);
            uploadCt32Pixels(
                accelerated, textureBlock, bufferWidth,
                70u, 40u, sourcePixels);
            drawNearestCt32SpriteCommand(software, command);
            drawNearestCt32SpriteCommand(accelerated, command);

            const std::vector<uint8_t> softwareReadback =
                readCt32Pixels(
                    software, framebufferBlock, bufferWidth,
                    prepared.boundsX0, prepared.boundsY0, 3u, 2u);
            const std::vector<uint8_t> acceleratedReadback =
                readCt32Pixels(
                    accelerated, framebufferBlock, bufferWidth,
                    prepared.boundsX0, prepared.boundsY0, 3u, 2u);
            t.IsTrue(acceleratedReadback == softwareReadback,
                     "the destination readback should match after the source upload");
            t.Equals(
                GSMem::ReadCT32(
                    acceleratedVram.data(), framebufferBlock,
                    bufferWidth, prepared.boundsX0,
                    prepared.boundsY0),
                sourcePixels[0],
                "the verified draw should consume the newly uploaded source texel");
            if (!compareFullVram("source upload and destination readback"))
                return;

            drawNearestCt32SpriteCommand(software, command);
            drawNearestCt32SpriteCommand(accelerated, command);
            software.writeRegister(GS_REG_FINISH, 0u);
            accelerated.writeRegister(GS_REG_FINISH, 0u);
            if (!compareFullVram("FINISH"))
                return;

            software.latchHostPresentationFrame();
            accelerated.latchHostPresentationFrame();
            std::vector<uint8_t> softwareFrame;
            std::vector<uint8_t> acceleratedFrame;
            uint32_t softwareWidth = 0u;
            uint32_t softwareHeight = 0u;
            uint32_t acceleratedWidth = 0u;
            uint32_t acceleratedHeight = 0u;
            t.IsTrue(
                software.copyLatchedHostPresentationFrame(
                    softwareFrame, softwareWidth, softwareHeight),
                "the software checkpoint should publish one host frame");
            t.IsTrue(
                accelerated.copyLatchedHostPresentationFrame(
                    acceleratedFrame,
                    acceleratedWidth, acceleratedHeight),
                "texture Verify should publish one host frame");
            t.Equals(acceleratedWidth, softwareWidth,
                     "texture Verify should preserve the frame width");
            t.Equals(acceleratedHeight, softwareHeight,
                     "texture Verify should preserve the frame height");
            t.IsTrue(acceleratedFrame == softwareFrame,
                     "texture Verify presentation bytes should match software");
            if (!compareFullVram("presentation latch"))
                return;

            configureNearestCt32SpriteCommand(software, command);
            configureNearestCt32SpriteCommand(accelerated, command);
            writeNearestCt32SpriteVertex(
                software, command.vertices()[0]);
            writeNearestCt32SpriteVertex(
                accelerated, command.vertices()[0]);
            const GsReplayState softwareState =
                software.captureReplayState();
            const GsReplayState acceleratedState =
                accelerated.captureReplayState();
            t.Equals(softwareState.vertexCount, 1,
                     "the software checkpoint should retain one sprite vertex");
            t.Equals(acceleratedState.vertexCount, 1,
                     "the Vulkan checkpoint should retain one sprite vertex");
            std::vector<uint8_t> softwareEncoded;
            std::vector<uint8_t> acceleratedEncoded;
            std::string stateError;
            t.IsTrue(
                encodeGsReplayState(
                    softwareState, softwareEncoded, &stateError),
                "the software texture checkpoint should encode");
            t.IsTrue(
                encodeGsReplayState(
                    acceleratedState, acceleratedEncoded, &stateError),
                "the Vulkan texture checkpoint should encode");
            t.IsTrue(acceleratedEncoded == softwareEncoded,
                     "Verify should preserve the exact canonical frontend state");

            software.writeRegister(
                GS_REG_PRIM,
                static_cast<uint64_t>(GS_PRIM_POINT));
            accelerated.writeRegister(
                GS_REG_PRIM,
                static_cast<uint64_t>(GS_PRIM_POINT));
            t.IsTrue(software.restoreReplayState(softwareState),
                     "the software texture checkpoint should restore");
            t.IsTrue(accelerated.restoreReplayState(acceleratedState),
                     "the Vulkan texture checkpoint should restore");
            writeNearestCt32SpriteVertex(
                software, command.vertices()[1]);
            writeNearestCt32SpriteVertex(
                accelerated, command.vertices()[1]);
            (void)software.getDebugSnapshot();
            (void)accelerated.getDebugSnapshot();
            if (!compareFullVram("exact-state restore"))
                return;

            software.reset();
            accelerated.reset();
            t.Equals(accelerated.rendererMode(), GsRendererMode::Verify,
                     "reset should preserve the texture Verify service and mode");
            drawNearestCt32SpriteCommand(software, command);
            drawNearestCt32SpriteCommand(accelerated, command);
            (void)software.getDebugSnapshot();
            (void)accelerated.getDebugSnapshot();
            if (!compareFullVram("post-reset draw"))
                return;

            const GsBackendCounters counters =
                accelerated.backendCounters();
            t.Equals(counters.commands, 7ull,
                     "the checkpoint stream should assemble six textures and one point");
            t.Equals(counters.acceleratedCommands, 6ull,
                     "every eligible texture should reach Verify");
            t.Equals(counters.verifiedCommands, 6ull,
                     "every accelerated texture should compare completely");
            t.Equals(counters.softwareCommands, 1ull,
                     "only the unsupported point should use software");
            t.Equals(counters.fallbackCommands, 1ull,
                     "the unsupported point should remain explicit fallback");
            t.Equals(counters.backendSwitches, 2ull,
                     "the eligible fallback eligible run should cross both backends");
            t.Equals(
                counters.decisions[static_cast<size_t>(
                    GsFallbackReason::UnsupportedPrimitive)],
                1ull,
                "the fallback should retain the point reason");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::Transfer)] >= 1ull,
                "the source upload should retain a transfer boundary");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::CpuReadback)] >= 1ull,
                "the destination readback should retain its named boundary");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::Finish)] >= 1ull,
                "FINISH should remain an explicit checkpoint");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::PresentationLatch)] >= 1ull,
                "presentation should remain an explicit checkpoint");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::SaveLoad)] >= 2ull,
                "capture and restore should each cross the save-state boundary");
            t.IsTrue(
                counters.flushReasons[static_cast<size_t>(
                    GsFlushReason::Reset)] >= 1ull,
                "reset should retain its named boundary");

            const GsVulkanRasterBackendStatistics backend =
                accelerated.vulkanRendererBackendStatistics();
            t.Equals(backend.commandsCompleted, 6ull,
                     "all six texture checkpoints should complete once");
            t.Equals(backend.verifiedCommands, 6ull,
                     "all six texture checkpoints should remain verified");
            t.Equals(
                backend.bytesCompared,
                6ull * static_cast<uint64_t>(GS_VULKAN_VRAM_SIZE),
                "each texture checkpoint should compare all 4 MiB");
            t.Equals(backend.verificationMismatches, 0ull,
                     "no texture checkpoint should disagree");
            const GsVulkanServiceStatistics service =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(service.nearestCt32SpriteDrawsCompleted, 6ull,
                     "the service should execute every eligible texture once");
            t.Equals(service.nearestCt32SpriteDrawsFailed, 0ull,
                     "no texture checkpoint should fail execution");
            t.Equals(
                service.nearestCt32SpritePixelsExecuted,
                6ull * static_cast<uint64_t>(
                    prepared.boundsX1 - prepared.boundsX0) *
                    static_cast<uint64_t>(
                        prepared.boundsY1 - prepared.boundsY0),
                "the checkpoint stream should retain exact pixel accounting");
            t.Equals(service.validationErrors, 0u,
                     "texture checkpoints should remain validation-clean");
            t.Equals(service.validationWarnings, 0u,
                     "texture checkpoints should emit no validation warnings");
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

        tc.Run("GS Vulkan routes qualified flat CT32 triangles by mode", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            std::vector<uint8_t> softwareVram =
                makeVramPattern(0x54524956u);
            std::vector<uint8_t> acceleratedVram = softwareVram;
            GS software;
            GS accelerated;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()), nullptr);
            accelerated.init(
                acceleratedVram.data(),
                static_cast<uint32_t>(acceleratedVram.size()), nullptr);
            configureFlatCt32Draws(software, 80u, 4u, 255u, 127u);
            configureFlatCt32Draws(accelerated, 80u, 4u, 255u, 127u);

            GsVulkanCapabilityReport preflight{};
            const GsVulkanServiceConfig config =
                makeRendererServiceConfig(preflight);
            t.IsTrue(accelerated.configureVulkanRenderer(config),
                     "a software-mode GS should accept triangle renderer configuration");
            if (!preflight.ready())
            {
                t.IsFalse(accelerated.setRendererMode(
                              GsRendererMode::Verify),
                          "an unavailable host should decline triangle Verify cleanly");
                return;
            }

            t.IsTrue(accelerated.setRendererMode(
                         GsRendererMode::Verify),
                     "a generally capable host should create Verify");
            if (accelerated.rendererMode() != GsRendererMode::Verify)
                return;
            accelerated.setBackendCountersEnabled(true);
            accelerated.resetBackendCounters();

            constexpr std::array<uint16_t, 3> firstX{{
                3u * 16u + 1u,
                27u * 16u + 15u,
                8u * 16u + 7u,
            }};
            constexpr std::array<uint16_t, 3> firstY{{
                5u * 16u + 3u,
                9u * 16u + 11u,
                24u * 16u + 5u,
            }};
            drawFlatCt32Triangle(
                software, firstX, firstY, 0xD4C3B2A1u);
            drawFlatCt32Triangle(
                accelerated, firstX, firstY, 0xD4C3B2A1u);
            (void)software.getDebugSnapshot();
            (void)accelerated.getDebugSnapshot();
            t.IsTrue(acceleratedVram == softwareVram,
                     "Verify triangle routing should retain the complete software image");

            const GsVulkanDeviceReport *selected =
                preflight.selectedDevice();
            const bool exactTriangle =
                selected && selected->exactCt32Triangle;
            const GsBackendCounters verifyCounters =
                accelerated.backendCounters();
            t.Equals(verifyCounters.commands, 1ull,
                     "the Verify fixture should assemble one triangle");
            t.Equals(verifyCounters.acceleratedCommands,
                     exactTriangle ? 1ull : 0ull,
                     "only the explicit triangle capability should select GPU Verify");
            t.Equals(verifyCounters.verifiedCommands,
                     exactTriangle ? 1ull : 0ull,
                     "only a GPU-routed triangle should count as verified");
            t.Equals(verifyCounters.softwareCommands,
                     exactTriangle ? 0ull : 1ull,
                     "missing triangle capability should retain software fallback");
            t.Equals(
                verifyCounters.decisions[static_cast<size_t>(
                    exactTriangle
                        ? GsFallbackReason::Supported
                        : GsFallbackReason::BackendUnavailable)],
                1ull,
                "Verify should record the exact capability decision");
            const GsVulkanRasterBackendStatistics verifyBackend =
                accelerated.vulkanRendererBackendStatistics();
            t.Equals(verifyBackend.verifiedCommands,
                     exactTriangle ? 1ull : 0ull,
                     "the backend should compare only a capable triangle");
            const GsVulkanServiceStatistics verifyService =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(verifyService.triangleDrawsCompleted,
                     exactTriangle ? 1ull : 0ull,
                     "the service should receive only capability-approved triangles");
            t.Equals(verifyService.spriteDrawsCompleted, 0ull,
                     "triangle routing must not alias the sprite request");

            t.IsTrue(accelerated.setRendererMode(
                         GsRendererMode::Hybrid),
                     "the synchronized Verify backend should enter Hybrid");
            accelerated.resetBackendCounters();
            constexpr std::array<uint16_t, 3> secondX{{
                35u * 16u,
                55u * 16u,
                39u * 16u,
            }};
            constexpr std::array<uint16_t, 3> secondY{{
                7u * 16u,
                12u * 16u,
                29u * 16u,
            }};
            drawFlatCt32Triangle(
                software, secondX, secondY, 0x88776655u);
            drawFlatCt32Triangle(
                accelerated, secondX, secondY, 0x88776655u);
            (void)software.getDebugSnapshot();
            (void)accelerated.getDebugSnapshot();
            t.IsTrue(acceleratedVram == softwareVram,
                     "Hybrid should retain exact software triangle fallback");
            const GsBackendCounters hybridCounters =
                accelerated.backendCounters();
            t.Equals(hybridCounters.acceleratedCommands, 0ull,
                     "Hybrid should not accelerate a triangle below its cost threshold");
            t.Equals(hybridCounters.softwareCommands, 1ull,
                     "Hybrid should execute the triangle once in software");
            t.Equals(hybridCounters.fallbackCommands, 1ull,
                     "Hybrid triangle fallback should remain explicit");
            t.Equals(
                hybridCounters.decisions[static_cast<size_t>(
                    exactTriangle
                        ? GsFallbackReason::CostModel
                        : GsFallbackReason::BackendUnavailable)],
                1ull,
                "Hybrid should record the capability or measured cost gate");
            const GsVulkanServiceStatistics finalService =
                accelerated.vulkanRendererServiceStatistics();
            t.Equals(finalService.triangleDrawsCompleted,
                     exactTriangle ? 1ull : 0ull,
                     "Hybrid fallback must not add a triangle dispatch");
            t.Equals(finalService.validationErrors, 0u,
                     "integrated triangle routing should remain validation-clean");
            t.Equals(finalService.validationWarnings, 0u,
                     "integrated triangle routing should emit no validation warnings");

            if (exactTriangle)
            {
                accelerated.resetBackendCounters();
                constexpr std::array<uint16_t, 3> hybridX{{
                    0u,
                    256u * 16u,
                    0u,
                }};
                constexpr std::array<uint16_t, 3> hybridY{{
                    0u,
                    0u,
                    128u * 16u,
                }};
                drawFlatCt32Triangle(
                    software, hybridX, hybridY, 0x6C5B4A39u);
                drawFlatCt32Triangle(
                    accelerated, hybridX, hybridY, 0x6C5B4A39u);
                (void)software.getDebugSnapshot();
                (void)accelerated.getDebugSnapshot();
                t.IsTrue(acceleratedVram == softwareVram,
                         "a threshold-sized Hybrid triangle should match software VRAM");
                const GsBackendCounters eligibleHybridCounters =
                    accelerated.backendCounters();
                t.Equals(eligibleHybridCounters.acceleratedCommands, 1ull,
                         "Hybrid should accelerate the measured triangle envelope");
                t.Equals(eligibleHybridCounters.softwareCommands, 0ull,
                         "an eligible Hybrid triangle should not execute in software");
                t.Equals(eligibleHybridCounters.fallbackCommands, 0ull,
                         "an eligible Hybrid triangle should not record fallback");
                const GsVulkanRasterBackendStatistics hybridBackend =
                    accelerated.vulkanRendererBackendStatistics();
                t.Equals(hybridBackend.committedGpuCommands, 1ull,
                         "Hybrid should commit one resident triangle");
                t.Equals(hybridBackend.residentBatchesCompleted, 1ull,
                         "the Hybrid checkpoint should drain one triangle batch");
                const GsVulkanServiceStatistics hybridService =
                    accelerated.vulkanRendererServiceStatistics();
                t.Equals(hybridService.triangleDrawsCompleted, 2ull,
                         "Verify and Hybrid should each complete one triangle");
                t.Equals(hybridService.residentTriangleBatchesCompleted, 1ull,
                         "Hybrid should reach the resident triangle request");

                t.IsTrue(accelerated.setRendererMode(
                             GsRendererMode::GpuStrict),
                         "the qualified host should enter strict triangle mode");
                accelerated.resetBackendCounters();
                constexpr std::array<uint16_t, 3> thirdX{{
                    61u * 16u + 1u,
                    83u * 16u + 13u,
                    67u * 16u + 9u,
                }};
                constexpr std::array<uint16_t, 3> thirdY{{
                    31u * 16u + 5u,
                    37u * 16u + 3u,
                    55u * 16u + 15u,
                }};
                drawFlatCt32Triangle(
                    software, thirdX, thirdY, 0xC4B3A291u);
                drawFlatCt32Triangle(
                    accelerated, thirdX, thirdY, 0xC4B3A291u);
                (void)software.getDebugSnapshot();
                (void)accelerated.getDebugSnapshot();
                t.IsTrue(acceleratedVram == softwareVram,
                         "strict triangle routing should match complete software VRAM");
                const GsBackendCounters strictCounters =
                    accelerated.backendCounters();
                t.Equals(strictCounters.acceleratedCommands, 1ull,
                         "strict should accelerate the qualified triangle");
                t.Equals(strictCounters.softwareCommands, 0ull,
                         "strict should not execute the qualified triangle in software");
                t.Equals(strictCounters.strictFailures, 0ull,
                         "the qualified strict triangle should not fail routing");
                const GsVulkanRasterBackendStatistics strictBackend =
                    accelerated.vulkanRendererBackendStatistics();
                t.Equals(strictBackend.committedGpuCommands, 2ull,
                         "Hybrid and strict should each commit a resident triangle");
                t.Equals(strictBackend.residentBatchesCompleted, 2ull,
                         "Hybrid and strict checkpoints should each drain a triangle batch");
                const GsVulkanServiceStatistics strictService =
                    accelerated.vulkanRendererServiceStatistics();
                t.Equals(strictService.triangleDrawsCompleted, 3ull,
                         "Verify, Hybrid, and strict should each complete one triangle");
                t.Equals(strictService.residentTriangleBatchesCompleted, 2ull,
                         "Hybrid and strict should use resident triangle requests");
                t.Equals(strictService.validationErrors, 0u,
                         "strict triangle routing should remain validation-clean");
                t.Equals(strictService.validationWarnings, 0u,
                         "strict triangle routing should emit no validation warnings");
            }
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
            const GsDrawCommand textureCommand =
                makeNearestCt32SpriteCommand(
                    101u, 40u, 2u, 64u, 2u, 6u, 5u,
                    {6u, 15u, 5u, 12u}, {32u, 16u},
                    {352u, 96u}, {48u, 304u},
                    {480u, 224u}, {64u, 320u}, 3u, 3u,
                    15u, 128u, 15u, 64u);
            GsVulkanNearestCt32Sprite preparedTexture{};
            t.IsTrue(
                prepareGsVulkanNearestCt32Sprite(
                    textureCommand, preparedTexture).supported,
                "the strict reset REGION_REPEAT texture fixture should be eligible");
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
            const GsVulkanDeviceReport *selected =
                preflight.selectedDevice();
            const bool exactTriangle =
                selected && selected->exactCt32Triangle;
            const bool exactTexture =
                selected && selected->exactNearestCt32Sprite;
            t.IsTrue(exactTexture,
                     "the strict reset device should expose the texture kernel");
            if (!exactTexture)
                return;

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

            drawNearestCt32SpriteCommand(
                software, textureCommand);
            drawNearestCt32SpriteCommand(
                strict, textureCommand);
            (void)strict.getDebugSnapshot();
            t.IsTrue(strictVram == softwareVram,
                     "strict texture execution before reset should remain exact");

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
            if (exactTriangle)
            {
                constexpr std::array<uint16_t, 3> x{{
                    19u * 16u + 1u,
                    43u * 16u + 11u,
                    25u * 16u + 7u,
                }};
                constexpr std::array<uint16_t, 3> y{{
                    7u * 16u + 3u,
                    13u * 16u + 9u,
                    29u * 16u + 5u,
                }};
                drawFlatCt32Triangle(
                    software, x, y, 0xF0AABBCCu);
                drawFlatCt32Triangle(
                    strict, x, y, 0xF0AABBCCu);
            }
            else
            {
                drawFlatCt32Sprite(
                    software, 19u * 16u, 20u * 16u,
                    39u * 16u, 31u * 16u, 0xF0AABBCCu);
                drawFlatCt32Sprite(
                    strict, 19u * 16u, 20u * 16u,
                    39u * 16u, 31u * 16u, 0xF0AABBCCu);
            }
            drawNearestCt32SpriteCommand(
                software, textureCommand);
            drawNearestCt32SpriteCommand(
                strict, textureCommand);
            strict.writeRegister(GS_REG_FINISH, 0u);
            (void)strict.getDebugSnapshot();
            t.IsTrue(strictVram == softwareVram,
                     "strict execution after reset and forced drain should remain exact");

            const GsBackendCounters counters = strict.backendCounters();
            t.Equals(counters.commands, 6ull,
                     "strict should classify four accepted and two rejected commands");
            t.Equals(counters.acceleratedCommands, 4ull,
                     "all eligible strict draws should reach Vulkan");
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
            t.Equals(backendStatistics.commandsCompleted, 4ull,
                     "only accepted strict draws should execute");
            t.Equals(backendStatistics.committedGpuCommands, 4ull,
                     "all accepted strict draws should publish GPU VRAM");
            t.Equals(backendStatistics.residentCommands, 4ull,
                     "all accepted strict draws should use resident execution");
            const GsVulkanServiceStatistics serviceStatistics =
                strict.vulkanRendererServiceStatistics();
            t.Equals(serviceStatistics.spriteDrawsCompleted,
                     exactTriangle ? 1ull : 2ull,
                     "strict reset should preserve exact sprite accounting");
            t.Equals(serviceStatistics.triangleDrawsCompleted,
                     exactTriangle ? 1ull : 0ull,
                     "a qualified host should execute the post-reset triangle");
            t.Equals(serviceStatistics.nearestCt32SpriteDrawsCompleted,
                     2ull,
                     "strict reset should preserve texture execution on both sides");
            t.Equals(
                serviceStatistics
                    .residentNearestCt32SpriteBatchesCompleted,
                2ull,
                "strict reset should preserve resident texture execution on both sides");
            t.Equals(
                serviceStatistics.nearestCt32SpritePixelsExecuted,
                2ull * static_cast<uint64_t>(
                    preparedTexture.boundsX1 - preparedTexture.boundsX0) *
                    static_cast<uint64_t>(
                        preparedTexture.boundsY1 - preparedTexture.boundsY0),
                "strict reset should preserve texture pixel accounting");
            t.Equals(serviceStatistics.residentTriangleBatchesCompleted,
                     exactTriangle ? 1ull : 0ull,
                     "the qualified post-reset triangle should use resident execution");
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
                t.Equals(
                    device.exactCt32Triangle,
                    device.suitable && device.shaderInt64,
                    "the exact triangle capability should expose its complete hard gate");
                t.Equals(
                    device.exactDepthCt32Sprite,
                    device.suitable,
                    "the depth CT32 capability should expose its raw-VRAM hard gate");
                t.Equals(
                    device.exactNearestCt32Sprite,
                    device.suitable,
                    "the nearest CT32 capability should expose its raw-VRAM hard gate");
                t.Equals(
                    device.exactLinearCt32Sprite,
                    device.suitable,
                    "the linear CT32 capability should expose its raw-VRAM hard gate");
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

        tc.Run("Vulkan nearest CT32 sprites match raw VRAM point sampling", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            std::vector<GsVulkanNearestCt32Sprite> sprites;
            const auto addSprite = [&](GsDrawCommand command)
            {
                GsVulkanNearestCt32Sprite sprite{};
                const GsBackendDecision decision =
                    prepareGsVulkanNearestCt32Sprite(command, sprite);
                if (!decision.supported)
                {
                    t.Fail(
                        "synthetic nearest CT32 sprite was rejected as " +
                        std::string(gsFallbackReasonName(
                            decision.reason)));
                    return false;
                }
                sprites.push_back(sprite);
                return true;
            };

            if (!addSprite(makeNearestCt32SpriteCommand(
                    30'000u, 40u, 2u, 64u, 2u, 6u, 5u,
                    {6u, 15u, 5u, 12u}, {32u, 16u},
                    {352u, 96u}, {48u, 304u},
                    {480u, 224u}, {64u, 320u})) ||
                !addSprite(makeNearestCt32SpriteCommand(
                    30'001u, 40u, 2u, 64u, 2u, 6u, 5u,
                    {3u, 14u, 2u, 6u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {992u, 1248u}, {480u, 608u})) ||
                !addSprite(makeNearestCt32SpriteCommand(
                    30'002u, 40u, 2u, 64u, 2u, 5u, 5u,
                    {10u, 20u, 7u, 17u}, {16u, 32u},
                    {400u, 144u}, {352u, 96u},
                    {80u, 336u}, {32u, 288u})) ||
                !addSprite(makeNearestCt32SpriteCommand(
                    30'003u, 40u, 2u, 64u, 2u, 5u, 4u,
                    {5u, 10u, 6u, 10u}, {48u, 16u},
                    {112u, 240u}, {208u, 80u},
                    {320u, 192u}, {320u, 192u})) ||
                !addSprite(makeNearestCt32SpriteCommand(
                    30'004u, 40u, 2u, 0x3FFFu, 1u, 0u, 0u,
                    {7u, 7u, 11u, 11u}, {0u, 0u},
                    {112u, 128u}, {176u, 192u},
                    {0u, 16u}, {0u, 16u})) ||
                !addSprite(makeNearestCt32SpriteCommand(
                    30'005u, 511u, 2u, 256u, 4u, 7u, 6u,
                    {60u, 67u, 29u, 35u}, {0u, 0u},
                    {960u, 1088u}, {464u, 576u},
                    {1984u, 2112u}, {992u, 1104u})) ||
                !addSprite(makeNearestCt32SpriteCommand(
                    30'006u, 100u, 4u, 512u, 8u, 8u, 7u,
                    {20u, 31u, 14u, 22u}, {0u, 0u},
                    {320u, 512u}, {224u, 368u},
                    {4000u, 4192u}, {2000u, 2144u})) ||
                !addSprite(makeNearestCt32SpriteCommand(
                    30'007u, 101u, 2u, 64u, 2u, 3u, 2u,
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 1u, 0u)) ||
                !addSprite(makeNearestCt32SpriteCommand(
                    30'008u, 102u, 2u, 64u, 2u, 3u, 2u,
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 0u, 1u)) ||
                !addSprite(makeNearestCt32SpriteCommand(
                    30'009u, 103u, 2u, 64u, 2u, 3u, 2u,
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 1u, 1u)) ||
                !addSprite(makeNearestCt32SpriteCommand(
                    30'010u, 104u, 2u, 64u, 2u, 3u, 2u,
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 2u, 0u,
                    70u, 72u, 0u, 0u)) ||
                !addSprite(makeNearestCt32SpriteCommand(
                    30'011u, 105u, 2u, 64u, 2u, 3u, 2u,
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 0u, 2u,
                    0u, 0u, 40u, 42u)) ||
                !addSprite(makeNearestCt32SpriteCommand(
                    30'012u, 106u, 2u, 64u, 2u, 3u, 2u,
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 2u, 2u,
                    70u, 72u, 40u, 42u)) ||
                !addSprite(makeNearestCt32SpriteCommand(
                    30'013u, 107u, 2u, 64u, 2u, 3u, 2u,
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 3u, 0u,
                    15u, 16u, 0u, 0u)) ||
                !addSprite(makeNearestCt32SpriteCommand(
                    30'014u, 108u, 2u, 64u, 2u, 3u, 2u,
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 0u, 3u,
                    0u, 0u, 15u, 16u)) ||
                !addSprite(makeNearestCt32SpriteCommand(
                    30'015u, 109u, 2u, 64u, 2u, 3u, 2u,
                    {0u, 15u, 0u, 7u}, {0u, 0u},
                    {0u, 256u}, {0u, 128u},
                    {0u, 256u}, {0u, 128u}, 3u, 3u,
                    15u, 128u, 15u, 64u)))
            {
                return;
            }

            t.Equals(sprites.size(), static_cast<size_t>(16u),
                     "the GPU texture corpus should retain every semantic fixture");
            t.Equals(sprites[4].textureBaseBlock, 0x3FFFu,
                     "the source-wrap fixture should begin at the last GS block");
            t.Equals(sprites[5].framebufferBaseBlock, 0x3FE0u,
                     "the destination-wrap fixture should begin at the last GS page");
            t.Equals(sprites[5].textureMaskU, 127u,
                     "the varied-stride fixture should retain its larger U mask");
            t.Equals(gsVulkanTextureWrapMode(sprites[7].textureWrapU), 1u,
                     "the GPU corpus should retain standard U clamp");
            t.Equals(gsVulkanTextureWrapMode(sprites[7].textureWrapV), 0u,
                     "standard U clamp should preserve repeat V");
            t.Equals(gsVulkanTextureWrapMode(sprites[8].textureWrapU), 0u,
                     "standard V clamp should preserve repeat U");
            t.Equals(gsVulkanTextureWrapMode(sprites[8].textureWrapV), 1u,
                     "the GPU corpus should retain standard V clamp");
            t.Equals(gsVulkanTextureWrapMode(sprites[9].textureWrapU), 1u,
                     "the GPU corpus should combine U and V clamp");
            t.Equals(gsVulkanTextureWrapMode(sprites[9].textureWrapV), 1u,
                     "the GPU corpus should combine V and U clamp");
            t.Equals(gsVulkanTextureWrapMode(sprites[10].textureWrapU), 2u,
                     "the GPU corpus should retain REGION_CLAMP U");
            t.Equals(gsVulkanTextureRegionMin(sprites[10].textureWrapU), 70u,
                     "REGION_CLAMP U should retain raw MINU beyond nominal width");
            t.Equals(gsVulkanTextureRegionMax(sprites[10].textureWrapU), 72u,
                     "REGION_CLAMP U should retain raw MAXU beyond nominal width");
            t.Equals(gsVulkanTextureWrapMode(sprites[11].textureWrapV), 2u,
                     "the GPU corpus should retain REGION_CLAMP V");
            t.Equals(gsVulkanTextureRegionMin(sprites[11].textureWrapV), 40u,
                     "REGION_CLAMP V should retain raw MINV beyond nominal height");
            t.Equals(gsVulkanTextureRegionMax(sprites[11].textureWrapV), 42u,
                     "REGION_CLAMP V should retain raw MAXV beyond nominal height");
            t.Equals(gsVulkanTextureWrapMode(sprites[12].textureWrapU), 2u,
                     "the GPU corpus should combine U and V REGION_CLAMP");
            t.Equals(gsVulkanTextureWrapMode(sprites[12].textureWrapV), 2u,
                     "the GPU corpus should combine V and U REGION_CLAMP");
            t.Equals(gsVulkanTextureWrapMode(sprites[13].textureWrapU), 3u,
                     "the GPU corpus should retain REGION_REPEAT U");
            t.Equals(gsVulkanTextureRegionMin(sprites[13].textureWrapU), 15u,
                     "REGION_REPEAT U should retain raw MINU before masking");
            t.Equals(gsVulkanTextureRegionMax(sprites[13].textureWrapU), 16u,
                     "REGION_REPEAT U should retain its raw MAXU offset");
            t.Equals(gsVulkanTextureWrapMode(sprites[14].textureWrapV), 3u,
                     "the GPU corpus should retain REGION_REPEAT V");
            t.Equals(gsVulkanTextureWrapMode(sprites[15].textureWrapU), 3u,
                     "the GPU corpus should combine U and V REGION_REPEAT");
            t.Equals(gsVulkanTextureWrapMode(sprites[15].textureWrapV), 3u,
                     "the GPU corpus should combine V and U REGION_REPEAT");

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
                         "an unavailable host should skip nearest CT32 execution cleanly");
                t.IsFalse(creationReport.ready(),
                          "the skipped texture service must retain its capability result");
                return;
            }

            t.IsNotNull(service.get(),
                        "a suitable device should create the nearest CT32 service");
            t.IsTrue(creationError.empty(),
                     "successful texture service creation should clear its diagnostic");
            if (!service)
                return;
            const GsVulkanDeviceReport *selected =
                creationReport.selectedDevice();
            t.IsNotNull(selected,
                        "the texture service should retain its selected device");
            if (selected)
            {
                t.IsTrue(selected->exactNearestCt32Sprite,
                         "the selected device should publish the exact texture capability");
            }

            uint64_t expectedPixels = 0u;
            for (size_t index = 0u; index < sprites.size(); ++index)
            {
                const GsVulkanNearestCt32Sprite &sprite =
                    sprites[index];
                const std::vector<uint8_t> input = makeVramPattern(
                    0x54455830u + static_cast<uint32_t>(index));
                std::vector<uint8_t> expected = input;
                applyNearestCt32SpriteCpu(expected, sprite);
                std::vector<uint8_t> actual = {0xA5u};
                std::string error;
                if (!service->executeNearestCt32Sprite(
                        input, sprite, actual, &error))
                {
                    t.Fail(
                        "GPU nearest CT32 sprite " +
                        std::to_string(index) + " failed: " + error);
                    return;
                }
                if (actual != expected)
                {
                    t.Fail(
                        "GPU nearest CT32 sprite " +
                        std::to_string(index) +
                        " disagreed with the complete CPU VRAM image");
                    return;
                }
                expectedPixels +=
                    static_cast<uint64_t>(
                        sprite.boundsX1 - sprite.boundsX0) *
                    static_cast<uint64_t>(
                        sprite.boundsY1 - sprite.boundsY0);
            }

            const std::vector<uint8_t> repeatedInput =
                makeVramPattern(0x52505431u);
            std::vector<uint8_t> repeatedExpected = repeatedInput;
            applyNearestCt32SpriteCpu(repeatedExpected, sprites[5]);
            std::vector<uint8_t> repeatedOutput = {0x5Au};
            std::string repeatedError;
            t.IsTrue(service->executeNearestCt32Sprite(
                         repeatedInput, sprites[5], repeatedOutput,
                         &repeatedError),
                     "the repeated destination-wrap dispatch should complete");
            t.IsTrue(repeatedOutput == repeatedExpected,
                     "repeated raw texture execution should remain byte-exact");
            expectedPixels +=
                static_cast<uint64_t>(
                    sprites[5].boundsX1 - sprites[5].boundsX0) *
                static_cast<uint64_t>(
                    sprites[5].boundsY1 - sprites[5].boundsY0);

            const std::vector<uint8_t> validInput =
                makeVramPattern(0x56414C31u);
            const auto expectRejected =
                [&](std::span<const uint8_t> rejectedInput,
                    GsVulkanNearestCt32Sprite rejectedSprite,
                    const std::string &label)
            {
                std::vector<uint8_t> output = {0x12u, 0x34u};
                const std::vector<uint8_t> sentinel = output;
                std::string error;
                t.IsFalse(service->executeNearestCt32Sprite(
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
                           "short texture VRAM input");
            GsVulkanNearestCt32Sprite invalid = sprites.front();
            invalid.textureWrapU |= 1u << 31u;
            expectRejected(validInput, invalid,
                           "texture wrap descriptor reserved bits");
            invalid = sprites[10];
            invalid.textureWrapU =
                packGsVulkanTextureWrap(2u, 73u, 72u);
            expectRejected(validInput, invalid,
                           "reversed texture region-clamp bounds");
            invalid = sprites.front();
            invalid.textureMaskU = 2u;
            expectRejected(validInput, invalid,
                           "non-power-of-two texture extent mask");
            invalid = sprites.front();
            invalid.textureStepU = 0;
            expectRejected(validInput, invalid,
                           "zero texture coordinate step");
            invalid = sprites.front();
            invalid.boundsX1 = invalid.boundsX0 +
                               invalid.framebufferWidth * 64u + 1u;
            expectRejected(validInput, invalid,
                           "self-aliasing texture destination");
            invalid = sprites.front();
            invalid.textureBaseBlock = invalid.framebufferBaseBlock;
            expectRejected(validInput, invalid,
                           "aliased texture source and destination");
            GSMem::PixelAddress regionAddress{};
            t.IsTrue(GSMem::ResolvePixelAddress(
                         GSMem::PixelStorageMode::C32,
                         sprites[12].textureBaseBlock,
                         sprites[12].textureWidth,
                         72u, 42u, regionAddress),
                     "the canonical resolver should accept the raw region maximum");
            const uint32_t regionPage = static_cast<uint32_t>(
                (static_cast<uint64_t>(regionAddress.word_index) *
                 sizeof(uint32_t)) / GS_VRAM_PAGE_SIZE);
            t.IsTrue(regionPage != 2u,
                     "the raw region fixture should extend beyond its nominal source page");
            invalid = sprites[12];
            invalid.framebufferBaseBlock = regionPage * 32u;
            expectRejected(validInput, invalid,
                           "region-expanded texture source and destination alias");
            GSMem::PixelAddress repeatAddress{};
            t.IsTrue(GSMem::ResolvePixelAddress(
                         GSMem::PixelStorageMode::C32,
                         sprites[15].textureBaseBlock,
                         sprites[15].textureWidth,
                         135u, 67u, repeatAddress),
                     "the canonical resolver should accept the expanded repeat maximum");
            const uint32_t repeatPage = static_cast<uint32_t>(
                (static_cast<uint64_t>(repeatAddress.word_index) *
                 sizeof(uint32_t)) / GS_VRAM_PAGE_SIZE);
            invalid = sprites[15];
            invalid.framebufferBaseBlock = repeatPage * 32u;
            expectRejected(validInput, invalid,
                           "region-repeat-expanded source and destination alias");

            const GsVulkanServiceStatistics statistics =
                service->statistics();
            const uint64_t acceptedDraws = sprites.size() + 1u;
            t.Equals(statistics.nearestCt32SpriteDrawsCompleted,
                     acceptedDraws,
                     "every accepted nearest CT32 sprite should complete exactly once");
            t.Equals(statistics.nearestCt32SpriteDrawsFailed, 0ull,
                     "caller-side texture rejection should not count as failed GPU work");
            t.Equals(statistics.nearestCt32SpritePixelsExecuted,
                     expectedPixels,
                     "texture statistics should count exact covered pixels");
            t.Equals(statistics.spriteDrawsCompleted, 0ull,
                     "textured execution should not pollute flat-sprite counters");
            t.Equals(statistics.queueSubmissions, acceptedDraws,
                     "each texture request should use one queue submission");
            t.Equals(statistics.shaderDispatches, acceptedDraws,
                     "each texture request should use one compute dispatch");
            t.Equals(statistics.pipelineBarriers, acceptedDraws * 4u,
                     "each raw texture request should retain the bounded barrier plan");
            t.Equals(statistics.bytesUploaded,
                     acceptedDraws * GS_VULKAN_VRAM_SIZE,
                     "each texture request should upload canonical CPU VRAM");
            t.Equals(statistics.bytesDownloaded,
                     acceptedDraws * GS_VULKAN_VRAM_SIZE,
                     "each texture request should download canonical CPU VRAM");
            t.Equals(statistics.validationErrors, 0u,
                     "nearest CT32 execution must remain validation-clean");
            t.Equals(statistics.validationWarnings, 0u,
                     "nearest CT32 execution should emit no validation warnings");
            t.IsTrue(service->healthy(),
                     "successful and rejected texture draws should leave the service healthy");

            service->shutdown();
            service->shutdown();
            const GsVulkanServiceStatistics shutdownStatistics =
                service->statistics();
            t.Equals(shutdownStatistics.validationErrors, 0u,
                     "texture pipeline destruction must remain validation-clean");
            t.Equals(shutdownStatistics.validationWarnings, 0u,
                     "texture pipeline destruction should emit no validation warnings");

            std::vector<uint8_t> shutdownOutput = {0xABu};
            const std::vector<uint8_t> shutdownSentinel = shutdownOutput;
            std::string shutdownError;
            t.IsFalse(service->executeNearestCt32Sprite(
                          validInput, sprites.front(), shutdownOutput,
                          &shutdownError),
                      "a stopped service must reject nearest CT32 work");
            t.IsTrue(shutdownOutput == shutdownSentinel,
                     "post-shutdown texture rejection must preserve output");
        });

        tc.Run("Vulkan depth CT32 sprites match Z32 Z24 compare and packed writes", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const std::array<GsDrawCommand, 6> commands{{
                makeDepthCt32SpriteCommand(
                    30'000u, 511u, 1u, 200u, GS_PSM_Z24, false, 1u,
                    {1u, 16u, 2u, 13u}, {0u, 0u},
                    17u, 33u, 273u, 225u,
                    0x11223344u, 0xFEDCBA98u),
                makeDepthCt32SpriteCommand(
                    30'001u, 40u, 2u, 511u, GS_PSM_Z32, false, 1u,
                    {3u, 20u, 1u, 15u}, {0u, 0u},
                    321u, 241u, 49u, 17u,
                    0x55667788u, 0x89ABCDEFu),
                makeDepthCt32SpriteCommand(
                    30'002u, 41u, 2u, 201u, GS_PSM_Z24, true, 2u,
                    {2u, 18u, 3u, 17u}, {0u, 0u},
                    33u, 49u, 305u, 289u,
                    0x99AABBCCu, 0x007FFF00u),
                makeDepthCt32SpriteCommand(
                    30'003u, 42u, 2u, 202u, GS_PSM_Z24, false, 3u,
                    {4u, 19u, 2u, 14u}, {0u, 0u},
                    65u, 33u, 321u, 241u,
                    0xDDEEFF10u, 0x00800100u),
                makeDepthCt32SpriteCommand(
                    30'004u, 43u, 2u, 203u, GS_PSM_Z32, true, 2u,
                    {1u, 17u, 4u, 18u}, {0u, 0u},
                    17u, 65u, 289u, 305u,
                    0x20304050u, 0x7FFFFF00u),
                makeDepthCt32SpriteCommand(
                    30'005u, 44u, 2u, 204u, GS_PSM_Z32, false, 3u,
                    {5u, 21u, 3u, 16u}, {0u, 0u},
                    81u, 49u, 353u, 273u,
                    0x60708090u, 0x80000100u),
            }};

            std::vector<GsVulkanDepthCt32Sprite> sprites;
            for (const GsDrawCommand &command : commands)
            {
                GsVulkanDepthCt32Sprite sprite{};
                const GsBackendDecision decision =
                    prepareGsVulkanDepthCt32Sprite(command, sprite);
                if (!decision.supported)
                {
                    t.Fail(
                        "synthetic depth CT32 sprite was rejected as " +
                        std::string(gsFallbackReasonName(decision.reason)));
                    return;
                }
                sprites.push_back(sprite);
            }

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
                         "an unavailable host should skip the depth shader cleanly");
                return;
            }
            t.IsNotNull(service.get(),
                        "a suitable device should create the depth sprite service");
            if (!service)
                return;
            const GsVulkanDeviceReport *selected =
                creationReport.selectedDevice();
            t.IsNotNull(selected,
                        "the depth service should retain its selected device");
            if (!selected)
                return;
            t.IsTrue(selected->exactDepthCt32Sprite,
                     "the selected device should publish exact Z32/Z24 depth");
            if (!selected->exactDepthCt32Sprite)
                return;

            std::vector<uint8_t> gpu = makeVramPattern(0x44335054u);
            uint64_t expectedPixels = 0u;
            bool coveredEquality = false;
            bool coveredPass = false;
            bool coveredFail = false;
            for (size_t index = 0u; index < sprites.size(); ++index)
            {
                const GsVulkanDepthCt32Sprite &sprite = sprites[index];
                if (sprite.depthTestMethod >= 2u)
                {
                    for (uint32_t y = sprite.boundsY0;
                         y < sprite.boundsY1; ++y)
                    {
                        for (uint32_t x = sprite.boundsX0;
                             x < sprite.boundsX1; ++x)
                        {
                            const uint32_t selector = (x + y) % 3u;
                            const uint32_t current = selector == 0u
                                ? sprite.depth - 1u
                                : selector == 1u
                                    ? sprite.depth
                                    : sprite.depth + 1u;
                            if (sprite.depthPsm == GS_PSM_Z24)
                            {
                                GSMem::WriteZ24(
                                    gpu.data(), sprite.depthBaseBlock,
                                    sprite.framebufferWidth, x, y, current);
                            }
                            else
                            {
                                GSMem::WriteZ32(
                                    gpu.data(), sprite.depthBaseBlock,
                                    sprite.framebufferWidth, x, y, current);
                            }
                            coveredPass = coveredPass || selector == 0u;
                            coveredEquality = coveredEquality || selector == 1u;
                            coveredFail = coveredFail || selector == 2u;
                        }
                    }
                }

                std::vector<uint8_t> expected = gpu;
                applyDepthCt32SpriteCpu(expected, sprite);
                std::vector<uint8_t> actual = {0xA5u};
                std::string error;
                if (!service->executeDepthCt32Sprite(
                        gpu, sprite, actual, &error))
                {
                    t.Fail(
                        "GPU depth CT32 sprite " +
                        std::to_string(index) + " failed: " + error);
                    return;
                }
                if (actual != expected)
                {
                    t.Fail(
                        "GPU depth CT32 sprite " +
                        std::to_string(index) +
                        " disagreed with the complete CPU VRAM image");
                    return;
                }
                gpu = std::move(actual);
                expectedPixels +=
                    static_cast<uint64_t>(
                        sprite.boundsX1 - sprite.boundsX0) *
                    static_cast<uint64_t>(
                        sprite.boundsY1 - sprite.boundsY0);
            }
            t.IsTrue(coveredEquality && coveredPass && coveredFail,
                     "the depth corpus should cover equality, pass, and fail inputs");

            const auto expectRejected = [&t, &service](
                std::span<const uint8_t> input,
                const GsVulkanDepthCt32Sprite &sprite,
                const std::string &label)
            {
                std::vector<uint8_t> output = {0x12u, 0x34u};
                const std::vector<uint8_t> sentinel = output;
                std::string error;
                t.IsFalse(
                    service->executeDepthCt32Sprite(
                        input, sprite, output, &error),
                    label + " should fail closed");
                t.IsTrue(output == sentinel,
                         label + " must preserve caller output");
                t.IsFalse(error.empty(),
                          label + " should retain a diagnostic");
            };
            const std::vector<uint8_t> shortInput(
                GS_VULKAN_VRAM_SIZE - 1u, 0u);
            expectRejected(shortInput, sprites.front(),
                           "short depth sprite VRAM input");
            GsVulkanDepthCt32Sprite invalid = sprites.front();
            invalid.reserved0 = 1u;
            expectRejected(gpu, invalid, "depth sprite reserved data");
            invalid = sprites.front();
            invalid.depthPsm = GS_PSM_Z16;
            expectRejected(gpu, invalid, "unsupported depth sprite format");
            invalid = sprites.front();
            invalid.depthTestMethod = 0u;
            expectRejected(gpu, invalid, "unsupported depth sprite method");
            invalid = sprites.front();
            invalid.depthWrite = 0u;
            expectRejected(gpu, invalid, "redundant masked ALWAYS depth sprite");
            invalid = sprites.front();
            invalid.depthBaseBlock = invalid.framebufferBaseBlock;
            expectRejected(gpu, invalid, "aliased depth and color surfaces");

            const GsVulkanServiceStatistics statistics =
                service->statistics();
            t.Equals(
                statistics.depthCt32SpriteDrawsCompleted,
                static_cast<uint64_t>(sprites.size()),
                "every depth sprite should complete exactly once");
            t.Equals(statistics.depthCt32SpriteDrawsFailed, 0ull,
                     "caller-side depth rejection should not count as GPU failure");
            t.Equals(statistics.depthCt32SpritePixelsExecuted,
                     expectedPixels,
                     "depth statistics should count exact candidate pixels");
            t.Equals(statistics.queueSubmissions,
                     static_cast<uint64_t>(sprites.size()),
                     "each depth request should use one queue submission");
            t.Equals(statistics.shaderDispatches,
                     static_cast<uint64_t>(sprites.size()),
                     "each depth request should use one compute dispatch");
            t.Equals(statistics.pipelineBarriers,
                     static_cast<uint64_t>(sprites.size()) * 4u,
                     "each depth request should retain the bounded barrier plan");
            t.Equals(statistics.validationErrors, 0u,
                     "depth shader execution must remain validation-clean");
            t.Equals(statistics.validationWarnings, 0u,
                     "depth shader execution should emit no validation warnings");
            t.IsTrue(service->healthy(),
                     "successful and rejected depth work should leave the service healthy");

            service->shutdown();
            service->shutdown();
            const GsVulkanServiceStatistics shutdownStatistics =
                service->statistics();
            t.Equals(shutdownStatistics.validationErrors, 0u,
                     "depth pipeline destruction must remain validation-clean");
            std::vector<uint8_t> shutdownOutput = {0xABu};
            const std::vector<uint8_t> shutdownSentinel = shutdownOutput;
            std::string shutdownError;
            t.IsFalse(
                service->executeDepthCt32Sprite(
                    gpu, sprites.front(), shutdownOutput, &shutdownError),
                "a stopped service must reject depth sprite work");
            t.IsTrue(shutdownOutput == shutdownSentinel,
                     "post-shutdown depth rejection must preserve output");
        });

        tc.Run("Vulkan linear CT32 repeat and clamp sprites match the prepared DDA", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const std::array<GsDrawCommand, 8> commands{{
                makeLinearCt32RepeatSpriteCommand(
                    31'000u, 0u, 8u, 3584u, 8u, 10u, 10u,
                    {0u, 511u, 0u, 447u}, {28672u, 29184u},
                    {28664u, 29176u}, {29176u, 36344u},
                    {0u, 512u}, {0u, 6656u}),
                makeLinearCt32RepeatSpriteCommand(
                    31'001u, 40u, 2u, 512u, 2u, 6u, 5u,
                    {3u, 12u, 2u, 13u}, {128u, 96u},
                    {120u, 376u}, {88u, 344u},
                    {0u, 249u}, {0u, 137u}),
                makeLinearCt32RepeatSpriteCommand(
                    31'002u, 41u, 2u, 512u, 2u, 6u, 5u,
                    {4u, 14u, 3u, 12u}, {128u, 96u},
                    {400u, 144u}, {352u, 96u},
                    {9u, 521u}, {17u, 273u}),
                makeLinearCt32RepeatSpriteCommand(
                    31'003u, 42u, 1u, 512u, 1u, 3u, 3u,
                    {0u, 7u, 0u, 7u}, {128u, 96u},
                    {120u, 248u}, {88u, 216u},
                    {0u, 128u}, {0u, 128u}),
                makeLinearCt32SpriteCommand(
                    31'004u, 0u, 8u, 3584u, 8u, 10u, 10u,
                    {0u, 511u, 0u, 447u}, {28672u, 29184u},
                    {28664u, 29176u}, {29176u, 36344u},
                    {0u, 512u}, {0u, 6656u}, 1u, 1u),
                makeLinearCt32SpriteCommand(
                    31'005u, 44u, 1u, 512u, 1u, 3u, 3u,
                    {0u, 7u, 0u, 7u}, {128u, 96u},
                    {128u, 256u}, {96u, 224u},
                    {0u, 128u}, {0u, 128u}, 1u, 0u),
                makeLinearCt32SpriteCommand(
                    31'006u, 45u, 1u, 512u, 1u, 3u, 3u,
                    {0u, 7u, 0u, 7u}, {128u, 96u},
                    {128u, 256u}, {96u, 224u},
                    {0u, 128u}, {0u, 128u}, 0u, 1u),
                makeLinearCt32SpriteCommand(
                    31'007u, 46u, 1u, 512u, 1u, 3u, 3u,
                    {0u, 7u, 0u, 7u}, {128u, 96u},
                    {128u, 256u}, {96u, 224u},
                    {128u, 256u}, {128u, 256u}, 1u, 1u),
            }};
            std::vector<GsVulkanLinearCt32Sprite> sprites;
            for (const GsDrawCommand &command : commands)
            {
                GsVulkanLinearCt32Sprite sprite{};
                const GsBackendDecision decision =
                    prepareGsVulkanLinearCt32Sprite(command, sprite);
                if (!decision.supported)
                {
                    t.Fail(
                        "synthetic linear CT32 sprite was rejected as " +
                        std::string(gsFallbackReasonName(decision.reason)));
                    return;
                }
                sprites.push_back(sprite);
            }

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
                         "an unavailable host should skip the linear shader cleanly");
                return;
            }
            t.IsNotNull(service.get(),
                        "a suitable device should create the linear sprite service");
            if (!service)
                return;
            const GsVulkanDeviceReport *selected =
                creationReport.selectedDevice();
            t.IsNotNull(selected,
                        "the linear service should retain its selected device");
            if (!selected)
                return;
            t.IsTrue(selected->exactLinearCt32Sprite,
                     "the selected device should publish exact linear sampling");

            std::vector<uint8_t> gpu = makeVramPattern(0x4C494E47u);
            uint64_t expectedPixels = 0u;
            for (size_t index = 0u; index < sprites.size(); ++index)
            {
                const GsVulkanLinearCt32Sprite &sprite = sprites[index];
                std::vector<uint8_t> expected = gpu;
                applyLinearCt32SpriteCpu(expected, sprite);
                std::vector<uint8_t> actual = {0xA5u};
                std::string error;
                if (!service->executeLinearCt32Sprite(
                        gpu, sprite, actual, &error))
                {
                    t.Fail(
                        "GPU linear CT32 sprite " +
                        std::to_string(index) + " failed: " + error);
                    return;
                }
                if (actual != expected)
                {
                    t.Fail(
                        "GPU linear CT32 sprite " +
                        std::to_string(index) +
                        " disagreed with the complete CPU VRAM image");
                    return;
                }
                gpu = std::move(actual);
                expectedPixels +=
                    static_cast<uint64_t>(
                        sprite.boundsX1 - sprite.boundsX0) *
                    static_cast<uint64_t>(
                        sprite.boundsY1 - sprite.boundsY0);
            }

            const auto expectRejected = [&t, &service](
                std::span<const uint8_t> input,
                const GsVulkanLinearCt32Sprite &sprite,
                const std::string &label)
            {
                std::vector<uint8_t> output = {0x12u, 0x34u};
                const std::vector<uint8_t> sentinel = output;
                std::string error;
                t.IsFalse(
                    service->executeLinearCt32Sprite(
                        input, sprite, output, &error),
                    label + " should fail closed");
                t.IsTrue(output == sentinel,
                         label + " must preserve caller output");
                t.IsFalse(error.empty(),
                          label + " should retain a diagnostic");
            };
            const std::vector<uint8_t> shortInput(
                GS_VULKAN_VRAM_SIZE - 1u, 0u);
            expectRejected(shortInput, sprites.front(),
                           "short linear sprite VRAM input");
            GsVulkanLinearCt32Sprite invalid = sprites.front();
            invalid.textureWrapU = packGsVulkanTextureWrap(2u, 0u, 0u);
            expectRejected(gpu, invalid,
                           "unsupported linear region wrap");
            invalid = sprites.front();
            invalid.textureMaskU = 6u;
            expectRejected(gpu, invalid,
                           "non-power-of-two linear sprite mask");
            invalid = sprites.front();
            invalid.fixedStepVBits = 0x7FC00000u;
            expectRejected(gpu, invalid,
                           "non-finite linear sprite V step");
            invalid = sprites.front();
            invalid.textureBaseBlock = invalid.framebufferBaseBlock;
            expectRejected(gpu, invalid,
                           "aliased linear sprite source");

            const GsVulkanServiceStatistics statistics =
                service->statistics();
            t.Equals(
                statistics.linearCt32SpriteDrawsCompleted,
                static_cast<uint64_t>(sprites.size()),
                "every linear sprite should complete exactly once");
            t.Equals(statistics.linearCt32SpriteDrawsFailed, 0ull,
                     "caller-side linear rejection should not count as GPU failure");
            t.Equals(statistics.linearCt32SpritePixelsExecuted,
                     expectedPixels,
                     "linear statistics should count exact output pixels");
            t.Equals(statistics.validationErrors, 0u,
                     "linear shader execution must remain validation-clean");
            t.Equals(statistics.validationWarnings, 0u,
                     "linear shader execution should emit no validation warnings");

            service->shutdown();
            service->shutdown();
            std::vector<uint8_t> shutdownOutput = {0xABu};
            const std::vector<uint8_t> shutdownSentinel = shutdownOutput;
            std::string shutdownError;
            t.IsFalse(
                service->executeLinearCt32Sprite(
                    gpu, sprites.front(), shutdownOutput, &shutdownError),
                "a stopped service must reject linear sprite work");
            t.IsTrue(shutdownOutput == shutdownSentinel,
                     "post-shutdown linear rejection must preserve output");
        });

        tc.Run("Vulkan CT32 triangles match CPU fixed-point edges and wrap", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            std::vector<GsVulkanCt32Triangle> triangles;
            const auto addTriangle = [&](const GsDrawCommand &command)
            {
                GsVulkanCt32Triangle triangle{};
                const GsBackendDecision decision =
                    prepareGsVulkanCt32Triangle(command, triangle);
                if (!decision.supported)
                {
                    t.Fail(
                        "synthetic CT32 triangle was rejected as " +
                        std::string(gsFallbackReasonName(
                            decision.reason)));
                    return false;
                }
                triangles.push_back(triangle);
                return true;
            };

            using FixedVertex = std::array<int32_t, 2>;
            using FixedTriangle = std::array<FixedVertex, 3>;
            constexpr std::array<FixedTriangle, 3> shapes{{
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
            constexpr std::array<GSScissorReg, 3> scissors{{
                {0u, 31u, 0u, 31u},
                {4u, 13u, 3u, 14u},
                {0u, 7u, 6u, 20u},
            }};
            constexpr std::array<uint32_t, 6> phases{{
                0u, 1u, 7u, 8u, 14u, 15u,
            }};
            constexpr GSXYOffsetReg xyoffset{128u, 128u};
            uint64_t sequence = 20'000u;
            for (size_t shapeIndex = 0u;
                 shapeIndex < shapes.size(); ++shapeIndex)
            {
                for (size_t permutationIndex = 0u;
                     permutationIndex < permutations.size();
                     ++permutationIndex)
                {
                    std::array<uint16_t, 3> rawX{};
                    std::array<uint16_t, 3> rawY{};
                    for (size_t vertex = 0u;
                         vertex < rawX.size(); ++vertex)
                    {
                        const FixedVertex &source = shapes[shapeIndex]
                            [permutations[permutationIndex][vertex]];
                        rawX[vertex] = static_cast<uint16_t>(
                            source[0] + phases[permutationIndex] +
                            xyoffset.ofx);
                        rawY[vertex] = static_cast<uint16_t>(
                            source[1] +
                            phases[permutations.size() - 1u -
                                   permutationIndex] +
                            xyoffset.ofy);
                    }
                    if (!addTriangle(makeCt32TriangleCommand(
                            sequence++,
                            static_cast<uint32_t>(3u + shapeIndex),
                            static_cast<uint8_t>(1u + shapeIndex),
                            scissors[permutationIndex % scissors.size()],
                            xyoffset, rawX, rawY,
                            0x80000000u |
                                static_cast<uint32_t>(
                                    shapeIndex * permutations.size() +
                                    permutationIndex + 1u))))
                    {
                        return;
                    }
                }
            }

            if (!addTriangle(makeCt32TriangleCommand(
                    sequence++, 511u, 1u,
                    {0u, 127u, 0u, 63u}, {0u, 0u},
                    {60u * 16u, 68u * 16u, 60u * 16u},
                    {28u * 16u, 28u * 16u, 36u * 16u},
                    0xC0778899u)) ||
                !addTriangle(makeCt32TriangleCommand(
                    sequence++, 7u, 2u,
                    {0u, 15u, 0u, 15u}, {32768u, 32768u},
                    {0u, 65535u, 0u},
                    {0u, 0u, 65535u},
                    0xE0AABBCCu)))
            {
                return;
            }

            t.Equals(triangles.size(), static_cast<size_t>(20u),
                     "the GPU corpus should retain every prepared triangle");
            t.Equals(triangles[18].framebufferBaseBlock, 0x3FE0u,
                     "the triangle wrap fixture should begin in the last GS page");
            t.IsTrue(
                ct32TriangleEdge(
                    triangles[19].vertex2X12_4,
                    triangles[19].vertex2Y12_4,
                    triangles[19].vertex0X12_4,
                    triangles[19].vertex0Y12_4,
                    15 * 16, 15 * 16) >
                    std::numeric_limits<int32_t>::max(),
                "the extreme fixture should require signed 64-bit edge arithmetic");

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
                         "an unavailable host should skip triangle execution cleanly");
                t.IsFalse(creationReport.ready(),
                          "a skipped triangle service must retain its capability result");
                return;
            }

            t.IsNotNull(service.get(),
                        "a generally suitable device should create the GS service");
            t.IsTrue(creationError.empty(),
                     "successful triangle service creation should clear its diagnostic");
            if (!service)
                return;
            const GsVulkanDeviceReport *selected =
                creationReport.selectedDevice();
            t.IsNotNull(selected,
                        "the triangle service should retain its selected device");
            if (!selected)
                return;

            const std::vector<uint8_t> initial =
                makeVramPattern(0x54524931u);
            if (!selected->exactCt32Triangle)
            {
                std::vector<uint8_t> output = {0xA5u, 0x5Au};
                const std::vector<uint8_t> sentinel = output;
                std::string error;
                t.IsFalse(service->executeCt32Triangle(
                              initial, triangles.front(), output, &error),
                          "a device without shaderInt64 must reject exact triangles");
                t.IsTrue(output == sentinel,
                         "capability rejection must preserve caller output");
                t.IsFalse(error.empty(),
                          "capability rejection should retain a diagnostic");
                t.Equals(service->statistics().triangleDrawsFailed, 0ull,
                         "capability rejection must not enter the worker");
                t.IsTrue(service->healthy(),
                         "an unsupported triangle must not poison the base service");
                service->shutdown();
                return;
            }
            t.IsTrue(selected->shaderInt64 && selected->suitable,
                     "an exact triangle device must expose every prerequisite");

            std::vector<uint8_t> gpu = initial;
            uint64_t expectedCandidatePixels = 0u;
            for (size_t index = 0u; index < triangles.size(); ++index)
            {
                const GsVulkanCt32Triangle &triangle = triangles[index];
                std::vector<uint8_t> expected = gpu;
                applyCt32TriangleCpu(expected, triangle);
                std::vector<uint8_t> actual = {0xA5u};
                std::string error;
                if (!service->executeCt32Triangle(
                        gpu, triangle, actual, &error))
                {
                    t.Fail(
                        "GPU CT32 triangle " +
                        std::to_string(index) + " failed: " + error);
                    return;
                }
                if (actual != expected)
                {
                    const auto mismatch = std::mismatch(
                        actual.begin(), actual.end(), expected.begin());
                    t.Fail(
                        "GPU CT32 triangle " +
                        std::to_string(index) +
                        " first disagreed with CPU VRAM at byte " +
                        std::to_string(static_cast<size_t>(
                            mismatch.first - actual.begin())));
                    return;
                }
                gpu = std::move(actual);
                expectedCandidatePixels +=
                    static_cast<uint64_t>(
                        triangle.boundsX1 - triangle.boundsX0) *
                    static_cast<uint64_t>(
                        triangle.boundsY1 - triangle.boundsY0);
            }

            const auto expectRejected =
                [&](std::span<const uint8_t> rejectedInput,
                    GsVulkanCt32Triangle rejectedTriangle,
                    const std::string &label)
            {
                std::vector<uint8_t> output = {0x12u, 0x34u};
                const std::vector<uint8_t> sentinel = output;
                std::string error;
                t.IsFalse(service->executeCt32Triangle(
                              rejectedInput, rejectedTriangle,
                              output, &error),
                          label + " should fail closed");
                t.IsTrue(output == sentinel,
                         label + " must preserve caller output");
                t.IsFalse(error.empty(),
                          label + " should retain a diagnostic");
            };

            const std::vector<uint8_t> shortInput(
                GS_VULKAN_VRAM_SIZE - 1u, 0u);
            expectRejected(shortInput, triangles.front(),
                           "short triangle VRAM input");
            GsVulkanCt32Triangle invalid = triangles.front();
            invalid.framebufferBaseBlock = 0x4000u;
            expectRejected(initial, invalid,
                           "out-of-range triangle framebuffer base");
            invalid = triangles.front();
            invalid.framebufferWidth = 0u;
            expectRejected(initial, invalid,
                           "zero triangle framebuffer width");
            invalid = triangles.front();
            invalid.boundsX1 = invalid.boundsX0;
            expectRejected(initial, invalid, "empty triangle bounds");
            invalid = triangles.front();
            invalid.boundsX1 = 2049u;
            expectRejected(initial, invalid,
                           "out-of-range triangle bounds");
            invalid = triangles.front();
            invalid.vertex0X12_4 = 65536;
            expectRejected(initial, invalid,
                           "out-of-range triangle vertex");
            invalid = triangles.front();
            std::swap(invalid.vertex1X12_4, invalid.vertex2X12_4);
            std::swap(invalid.vertex1Y12_4, invalid.vertex2Y12_4);
            expectRejected(initial, invalid,
                           "negative triangle winding");
            invalid = triangles.front();
            invalid.topLeftEdgeMask ^= 1u;
            expectRejected(initial, invalid,
                           "inconsistent triangle edge mask");
            invalid = triangles.front();
            invalid.topLeftEdgeMask |= 8u;
            expectRejected(initial, invalid,
                           "reserved triangle edge mask bit");
            invalid = triangles.front();
            invalid.reserved1 = 1u;
            expectRejected(initial, invalid,
                           "non-zero triangle reserved data");

            const GsVulkanServiceStatistics statistics =
                service->statistics();
            const uint64_t acceptedDraws = triangles.size();
            t.Equals(statistics.triangleDrawsCompleted, acceptedDraws,
                     "every accepted triangle should complete exactly once");
            t.Equals(statistics.triangleDrawsFailed, 0ull,
                     "caller-side triangle rejection should not count as failed GPU work");
            t.Equals(statistics.triangleCandidatePixelsExecuted,
                     expectedCandidatePixels,
                     "triangle statistics should count conservative dispatch pixels");
            t.Equals(statistics.queueSubmissions, acceptedDraws,
                     "each independent triangle should use one queue submission");
            t.Equals(statistics.shaderDispatches, acceptedDraws,
                     "each independent triangle should use one compute dispatch");
            t.Equals(statistics.bytesUploaded,
                     acceptedDraws * GS_VULKAN_VRAM_SIZE,
                     "each triangle should upload canonical CPU VRAM");
            t.Equals(statistics.bytesDownloaded,
                     acceptedDraws * GS_VULKAN_VRAM_SIZE,
                     "each triangle should download canonical CPU VRAM");
            t.Equals(statistics.validationErrors, 0u,
                     "CT32 triangle execution must remain validation-clean");
            t.Equals(statistics.validationWarnings, 0u,
                     "CT32 triangle execution should emit no validation warnings");
            t.IsTrue(service->healthy(),
                     "successful and rejected triangles should leave the service healthy");

            service->shutdown();
            service->shutdown();
            const GsVulkanServiceStatistics shutdownStatistics =
                service->statistics();
            t.Equals(shutdownStatistics.validationErrors, 0u,
                     "triangle pipeline destruction must remain validation-clean");
            t.Equals(shutdownStatistics.validationWarnings, 0u,
                     "triangle pipeline destruction should emit no validation warnings");

            std::vector<uint8_t> shutdownOutput = {0xABu};
            const std::vector<uint8_t> shutdownSentinel = shutdownOutput;
            std::string shutdownError;
            t.IsFalse(service->executeCt32Triangle(
                          initial, triangles.front(), shutdownOutput,
                          &shutdownError),
                      "a stopped service must reject triangle work");
            t.IsTrue(shutdownOutput == shutdownSentinel,
                     "post-shutdown triangle rejection must preserve output");
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

        tc.Run("Vulkan resident depth CT32 batches preserve ordered Z32 Z24 dependencies", [](TestCase &t)
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
                         "an unavailable host should skip resident depth batching cleanly");
                t.IsFalse(creationReport.ready(),
                          "a skipped depth batch service should retain its capability result");
                return;
            }
            t.IsNotNull(service.get(),
                        "a suitable device should create the resident depth batch service");
            t.IsTrue(creationError.empty(),
                     "successful depth batch service creation should clear its diagnostic");
            if (!service)
                return;

            const std::array<GsDrawCommand, 6> commands =
                makeOrderedDepthCt32SpriteCommands(38'000u);
            std::vector<GsVulkanDepthCt32Sprite> sprites;
            for (const GsDrawCommand &command : commands)
            {
                GsVulkanDepthCt32Sprite sprite{};
                const GsBackendDecision decision =
                    prepareGsVulkanDepthCt32Sprite(command, sprite);
                if (!decision.supported)
                {
                    t.Fail(
                        "resident depth CT32 fixture was rejected as " +
                        std::string(gsFallbackReasonName(decision.reason)));
                    return;
                }
                sprites.push_back(sprite);
            }

            const auto colorPages = [](
                const GsVulkanDepthCt32Sprite &sprite)
            {
                return gsVramPagesForSurfaceRect(
                    sprite.framebufferBaseBlock,
                    sprite.framebufferWidth,
                    static_cast<uint8_t>(GSMem::C32),
                    sprite.boundsX0, sprite.boundsY0,
                    sprite.boundsX1 - sprite.boundsX0,
                    sprite.boundsY1 - sprite.boundsY0);
            };
            const auto depthPages = [](
                const GsVulkanDepthCt32Sprite &sprite)
            {
                return gsVramPagesForSurfaceRect(
                    sprite.depthBaseBlock,
                    sprite.framebufferWidth,
                    static_cast<uint8_t>(sprite.depthPsm),
                    sprite.boundsX0, sprite.boundsY0,
                    sprite.boundsX1 - sprite.boundsX0,
                    sprite.boundsY1 - sprite.boundsY0);
            };
            t.IsTrue(colorPages(sprites[0]).intersects(colorPages(sprites[1])) &&
                         depthPages(sprites[0]).intersects(depthPages(sprites[1])),
                     "the Z24 pair should overlap both raw surfaces");
            t.IsFalse(colorPages(sprites[1]).intersects(colorPages(sprites[2])) ||
                          depthPages(sprites[1]).intersects(depthPages(sprites[2])),
                      "the Z24 and Z32 pairs should contain a disjoint segment");
            t.IsTrue(colorPages(sprites[2]).intersects(colorPages(sprites[3])) &&
                         depthPages(sprites[2]).intersects(depthPages(sprites[3])),
                     "the Z32 pair should overlap both raw surfaces");
            t.IsTrue(depthPages(sprites[3]).intersects(colorPages(sprites[4])),
                     "the cross-view draw should write an earlier Z32 page as CT32");
            t.IsTrue(depthPages(sprites[4]).intersects(depthPages(sprites[5])),
                     "the masked comparison should read the preceding Z24 write");

            std::string operationError;
            const GsVulkanServiceStatistics beforeRejections =
                service->statistics();
            const auto expectRejected =
                [&](std::span<const GsVulkanDepthCt32Sprite> rejected,
                    const std::string &label)
            {
                operationError.clear();
                t.IsFalse(service->executeResidentDepthCt32Sprites(
                              rejected, &operationError),
                          label + " should fail before worker submission");
                t.IsFalse(operationError.empty(),
                          label + " should retain a diagnostic");
            };
            expectRejected({}, "empty resident depth batch");
            std::vector<GsVulkanDepthCt32Sprite> oversized(
                GS_VULKAN_MAX_RESIDENT_DEPTH_CT32_BATCH + 1u,
                sprites.front());
            expectRejected(oversized, "oversized resident depth batch");
            std::vector<GsVulkanDepthCt32Sprite> invalid = sprites;
            invalid[3].reserved0 = 1u;
            expectRejected(invalid, "invalid resident depth batch member");
            const GsVulkanServiceStatistics afterRejections =
                service->statistics();
            t.Equals(afterRejections.queueSubmissions,
                     beforeRejections.queueSubmissions,
                     "caller-rejected depth batches must not submit GPU work");
            t.Equals(afterRejections.depthCt32SpriteDrawsFailed,
                     beforeRejections.depthCt32SpriteDrawsFailed,
                     "caller-rejected depth batches must not count worker failures");
            t.Equals(
                afterRejections.residentDepthCt32SpriteBatchesFailed,
                beforeRejections.residentDepthCt32SpriteBatchesFailed,
                "caller-rejected depth batches must not count failed worker batches");

            const GsVulkanDeviceReport *selected =
                creationReport.selectedDevice();
            t.IsNotNull(selected,
                        "the depth batch service should retain its selected device");
            if (!selected)
                return;
            if (!selected->exactDepthCt32Sprite)
            {
                expectRejected(sprites,
                               "capability-gated resident depth batch");
                t.Equals(service->statistics().depthCt32SpriteDrawsFailed,
                         0ull,
                         "depth capability rejection must not enter the worker");
                t.IsTrue(service->healthy(),
                         "unsupported depth batching must not poison the base service");
                service->shutdown();
                return;
            }

            const std::vector<uint8_t> initial =
                makeVramPattern(0x44505448u);
            std::vector<uint8_t> expected = initial;
            uint64_t expectedPixels = 0u;
            for (const GsVulkanDepthCt32Sprite &sprite : sprites)
            {
                applyDepthCt32SpriteCpu(expected, sprite);
                expectedPixels +=
                    static_cast<uint64_t>(
                        sprite.boundsX1 - sprite.boundsX0) *
                    static_cast<uint64_t>(
                        sprite.boundsY1 - sprite.boundsY0);
            }
            GsVramPageMask allPages;
            allPages.setAll();
            t.IsTrue(service->uploadVramPages(
                         initial, allPages, &operationError),
                     "the depth batch fixture should establish resident VRAM");
            if (!operationError.empty())
                t.Fail("resident depth batch upload failed: " + operationError);

            const GsVulkanServiceStatistics beforeBatch =
                service->statistics();
            operationError.clear();
            if (!service->executeResidentDepthCt32Sprites(
                    sprites, &operationError))
            {
                t.Fail(
                    "ordered resident depth batch failed: " +
                    operationError);
                service->shutdown();
                return;
            }
            const GsVulkanServiceStatistics afterBatch =
                service->statistics();
            t.IsTrue(operationError.empty(),
                     "successful resident depth batching should clear its diagnostic");
            t.Equals(afterBatch.queueSubmissions -
                         beforeBatch.queueSubmissions,
                     1ull,
                     "one depth batch should use one Vulkan queue submission");
            t.Equals(afterBatch.shaderDispatches -
                         beforeBatch.shaderDispatches,
                     static_cast<uint64_t>(sprites.size()),
                     "each depth batch member should retain one compute dispatch");
            t.Equals(afterBatch.pipelineBarriers -
                         beforeBatch.pipelineBarriers,
                     6ull,
                     "four dependencies should join the two batch-envelope barriers");
            t.Equals(afterBatch.pipelineBinds - beforeBatch.pipelineBinds,
                     1ull,
                     "one depth batch should bind the exact pipeline once");
            t.Equals(afterBatch.pipelineCacheHits -
                         beforeBatch.pipelineCacheHits,
                     1ull,
                     "one depth batch should reuse its prebuilt pipeline once");
            t.Equals(afterBatch.fenceWaits - beforeBatch.fenceWaits,
                     1ull,
                     "one depth batch should wait on one submitted fence");
            t.Equals(afterBatch.depthCt32SpriteDrawsCompleted -
                         beforeBatch.depthCt32SpriteDrawsCompleted,
                     static_cast<uint64_t>(sprites.size()),
                     "depth batch statistics should count every completed sprite");
            t.Equals(afterBatch.depthCt32SpritePixelsExecuted -
                         beforeBatch.depthCt32SpritePixelsExecuted,
                     expectedPixels,
                     "depth batch statistics should count every candidate pixel");
            t.Equals(
                afterBatch.residentDepthCt32SpriteBatchesCompleted -
                    beforeBatch.residentDepthCt32SpriteBatchesCompleted,
                1ull,
                "the worker should complete one resident depth batch");
            t.Equals(afterBatch.largestResidentDepthCt32SpriteBatch,
                     static_cast<uint64_t>(sprites.size()),
                     "the service should retain its largest depth batch");

            std::vector<uint8_t> actual(
                GS_VULKAN_VRAM_SIZE, 0xA5u);
            t.IsTrue(service->downloadVramPages(
                         actual, allPages, &operationError),
                     "the completed resident depth batch should be observable");
            t.IsTrue(actual == expected,
                     "ordered resident depth draws must match the sequential CPU oracle");

            std::vector<uint8_t> singletonExpected = expected;
            applyDepthCt32SpriteCpu(singletonExpected, sprites.front());
            operationError.clear();
            t.IsTrue(service->executeResidentDepthCt32Sprite(
                         sprites.front(), &operationError),
                     "the resident depth singleton wrapper should execute");
            std::fill(actual.begin(), actual.end(), 0xA5u);
            t.IsTrue(service->downloadVramPages(
                         actual, allPages, &operationError),
                     "the resident depth singleton should be observable");
            t.IsTrue(actual == singletonExpected,
                     "the singleton wrapper should preserve exact depth semantics");

            const GsVulkanServiceStatistics finalStatistics =
                service->statistics();
            t.Equals(finalStatistics.validationErrors, 0u,
                     "resident depth batching must remain validation-clean");
            t.Equals(finalStatistics.validationWarnings, 0u,
                     "resident depth batching should emit no validation warnings");
            t.IsTrue(service->healthy(),
                     "accepted and rejected depth batches should leave the service healthy");

            service->shutdown();
            operationError.clear();
            t.IsFalse(service->executeResidentDepthCt32Sprites(
                          sprites, &operationError),
                      "a stopped service must reject resident depth batches");
            t.IsFalse(operationError.empty(),
                      "post-shutdown depth rejection should retain a diagnostic");
        });

        tc.Run("Vulkan resident nearest CT32 batches preserve texture dependencies", [](TestCase &t)
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
                         "an unavailable host should skip resident texture batching cleanly");
                t.IsFalse(creationReport.ready(),
                          "a skipped texture batch service should retain its capability result");
                return;
            }
            t.IsNotNull(service.get(),
                        "a suitable device should create the resident texture batch service");
            t.IsTrue(creationError.empty(),
                     "successful texture batch service creation should clear its diagnostic");
            if (!service)
                return;

            constexpr std::array<uint32_t, 8> framebufferPages{{
                40u, 41u, 197u, 509u, 200u, 201u, 202u, 203u,
            }};
            std::vector<GsVulkanNearestCt32Sprite> sprites;
            for (size_t index = 0u; index < framebufferPages.size(); ++index)
            {
                const GsDrawCommand command = makeNearestCt32SpriteCommand(
                    35'000u + index, framebufferPages[index], 2u,
                    64u, 2u, 6u, 5u,
                    {6u, 15u, 5u, 12u}, {32u, 16u},
                    {352u, 96u}, {48u, 304u},
                    {480u, 224u}, {64u, 320u},
                    index == 4u
                        ? 2u
                        : (index == 6u
                            ? 3u
                            : static_cast<uint8_t>(index & 1u)),
                    index == 5u
                        ? 2u
                        : (index == 7u
                            ? 3u
                            : static_cast<uint8_t>((index >> 1u) & 1u)),
                    index == 4u ? 70u : (index == 6u ? 15u : 0u),
                    index == 4u ? 72u : (index == 6u ? 16u : 0u),
                    index == 5u ? 40u : (index == 7u ? 15u : 0u),
                    index == 5u ? 42u : (index == 7u ? 16u : 0u));
                GsVulkanNearestCt32Sprite sprite{};
                const GsBackendDecision decision =
                    prepareGsVulkanNearestCt32Sprite(command, sprite);
                if (!decision.supported)
                {
                    t.Fail(
                        "resident nearest CT32 fixture was rejected as " +
                        std::string(gsFallbackReasonName(decision.reason)));
                    return;
                }
                sprites.push_back(sprite);
            }
            t.Equals(gsVulkanTextureWrapMode(sprites[4].textureWrapU), 2u,
                     "resident batching should retain REGION_CLAMP U");
            t.Equals(gsVulkanTextureRegionMax(sprites[4].textureWrapU), 72u,
                     "resident batching should retain raw MAXU beyond nominal width");
            t.Equals(gsVulkanTextureWrapMode(sprites[5].textureWrapV), 2u,
                     "resident batching should retain REGION_CLAMP V");
            t.Equals(gsVulkanTextureRegionMax(sprites[5].textureWrapV), 42u,
                     "resident batching should retain raw MAXV beyond nominal height");
            t.Equals(gsVulkanTextureWrapMode(sprites[6].textureWrapU), 3u,
                     "resident batching should retain REGION_REPEAT U");
            t.Equals(gsVulkanTextureRegionMax(sprites[6].textureWrapU), 16u,
                     "resident batching should retain REGION_REPEAT U offset");
            t.Equals(gsVulkanTextureWrapMode(sprites[7].textureWrapV), 3u,
                     "resident batching should retain REGION_REPEAT V");
            t.Equals(gsVulkanTextureRegionMax(sprites[7].textureWrapV), 16u,
                     "resident batching should retain REGION_REPEAT V offset");

            const GsVramPageMask sharedReadPages =
                gsVramPagesForSurfaceRect(
                    sprites.front().textureBaseBlock,
                    sprites.front().textureWidth,
                    static_cast<uint8_t>(GSMem::C32),
                    0u, 0u,
                    sprites.front().textureMaskU + 1u,
                    sprites.front().textureMaskV + 1u);
            t.Equals(sharedReadPages.count(), size_t{1u},
                     "the shared texture fixture should conservatively read one page");
            t.IsTrue(sharedReadPages.test(2u),
                     "the shared texture fixture should read its selected physical page");

            std::string operationError;
            const GsVulkanServiceStatistics beforeRejections =
                service->statistics();
            const auto expectRejected =
                [&](std::span<const GsVulkanNearestCt32Sprite> rejected,
                    const std::string &label)
            {
                operationError.clear();
                t.IsFalse(service->executeResidentNearestCt32Sprites(
                              rejected, &operationError),
                          label + " should fail before worker submission");
                t.IsFalse(operationError.empty(),
                          label + " should retain a diagnostic");
            };
            const auto expectDependencyRejected =
                [&](std::span<const GsVulkanNearestCt32Sprite> rejected,
                    const std::string &label)
            {
                expectRejected(rejected, label);
                t.IsTrue(
                    operationError.find("memory dependency") !=
                        std::string::npos,
                    label + " should identify an inter-draw dependency");
            };
            expectRejected({}, "empty resident texture batch");
            std::vector<GsVulkanNearestCt32Sprite> oversized(
                GS_VULKAN_MAX_RESIDENT_NEAREST_CT32_BATCH + 1u,
                sprites.front());
            expectRejected(oversized, "oversized resident texture batch");
            std::vector<GsVulkanNearestCt32Sprite> invalid = sprites;
            invalid[2].textureWrapU |= 1u << 31u;
            expectRejected(invalid, "invalid resident texture batch member");

            const std::array<GsVulkanNearestCt32Sprite, 2> writeWrite{{
                sprites.front(), sprites.front(),
            }};
            expectDependencyRejected(
                writeWrite, "write/write-dependent texture batch");

            GsVulkanNearestCt32Sprite writeThenRead = sprites[1];
            writeThenRead.textureBaseBlock =
                sprites.front().framebufferBaseBlock;
            const std::array<GsVulkanNearestCt32Sprite, 2>
                writeRead{{sprites.front(), writeThenRead}};
            expectDependencyRejected(
                writeRead, "write/read-dependent texture batch");

            GsVulkanNearestCt32Sprite readThenWrite = sprites[1];
            readThenWrite.framebufferBaseBlock =
                sprites.front().textureBaseBlock;
            readThenWrite.textureBaseBlock = 60u * 32u;
            const std::array<GsVulkanNearestCt32Sprite, 2>
                readWrite{{sprites.front(), readThenWrite}};
            expectDependencyRejected(
                readWrite, "read/write-dependent texture batch");

            const GsVulkanServiceStatistics afterRejections =
                service->statistics();
            t.Equals(afterRejections.queueSubmissions,
                     beforeRejections.queueSubmissions,
                     "caller-rejected texture batches must not submit GPU work");
            t.Equals(afterRejections.nearestCt32SpriteDrawsFailed,
                     beforeRejections.nearestCt32SpriteDrawsFailed,
                     "caller-rejected texture batches must not count worker failures");
            t.Equals(
                afterRejections.residentNearestCt32SpriteBatchesFailed,
                beforeRejections.residentNearestCt32SpriteBatchesFailed,
                "caller-rejected texture batches must not count failed worker batches");

            const GsVulkanDeviceReport *selected =
                creationReport.selectedDevice();
            t.IsNotNull(selected,
                        "the texture batch service should retain its selected device");
            if (!selected)
                return;
            if (!selected->exactNearestCt32Sprite)
            {
                expectRejected(sprites,
                               "capability-gated resident texture batch");
                t.Equals(
                    service->statistics().nearestCt32SpriteDrawsFailed,
                    0ull,
                    "capability rejection must not enter the worker");
                t.IsTrue(service->healthy(),
                         "unsupported texture batching must not poison the base service");
                service->shutdown();
                return;
            }

            const std::vector<uint8_t> initial =
                makeVramPattern(0x52545831u);
            std::vector<uint8_t> expected = initial;
            uint64_t expectedPixels = 0u;
            for (const GsVulkanNearestCt32Sprite &sprite : sprites)
            {
                applyNearestCt32SpriteCpu(expected, sprite);
                expectedPixels +=
                    static_cast<uint64_t>(
                        sprite.boundsX1 - sprite.boundsX0) *
                    static_cast<uint64_t>(
                        sprite.boundsY1 - sprite.boundsY0);
            }
            GsVramPageMask allPages;
            allPages.setAll();
            t.IsTrue(service->uploadVramPages(
                         initial, allPages, &operationError),
                     "the texture batch fixture should establish resident VRAM");
            if (!operationError.empty())
                t.Fail("resident texture batch upload failed: " + operationError);

            const GsVulkanServiceStatistics beforeBatch =
                service->statistics();
            operationError.clear();
            t.IsTrue(service->executeResidentNearestCt32Sprites(
                         sprites, &operationError),
                     "shared-read resident textures should execute as one batch");
            t.IsTrue(operationError.empty(),
                     "successful resident texture batching should clear its diagnostic");
            const GsVulkanServiceStatistics afterBatch =
                service->statistics();
            t.Equals(afterBatch.queueSubmissions -
                         beforeBatch.queueSubmissions,
                     1ull,
                     "one texture batch should use one Vulkan queue submission");
            t.Equals(afterBatch.shaderDispatches -
                         beforeBatch.shaderDispatches,
                     static_cast<uint64_t>(sprites.size()),
                     "each texture batch member should retain one compute dispatch");
            t.Equals(afterBatch.pipelineBarriers -
                         beforeBatch.pipelineBarriers,
                     2ull,
                     "one texture batch should use one prepare and one completion barrier");
            t.Equals(afterBatch.pipelineBinds - beforeBatch.pipelineBinds,
                     1ull,
                     "one texture batch should bind the exact pipeline once");
            t.Equals(afterBatch.pipelineCacheHits -
                         beforeBatch.pipelineCacheHits,
                     1ull,
                     "one texture batch should reuse its prebuilt pipeline once");
            t.Equals(afterBatch.fenceWaits - beforeBatch.fenceWaits,
                     1ull,
                     "one texture batch should wait on one submitted fence");
            t.Equals(afterBatch.nearestCt32SpriteDrawsCompleted -
                         beforeBatch.nearestCt32SpriteDrawsCompleted,
                     static_cast<uint64_t>(sprites.size()),
                     "texture batch statistics should count every completed sprite");
            t.Equals(afterBatch.nearestCt32SpritePixelsExecuted -
                         beforeBatch.nearestCt32SpritePixelsExecuted,
                     expectedPixels,
                     "texture batch statistics should count every covered pixel");
            t.Equals(
                afterBatch.residentNearestCt32SpriteBatchesCompleted -
                    beforeBatch.residentNearestCt32SpriteBatchesCompleted,
                1ull,
                "the worker should complete one resident texture batch");
            t.Equals(afterBatch.largestResidentNearestCt32SpriteBatch,
                     static_cast<uint64_t>(sprites.size()),
                     "the service should retain its largest texture batch");

            std::vector<uint8_t> actual(
                GS_VULKAN_VRAM_SIZE, 0xA5u);
            t.IsTrue(service->downloadVramPages(
                         actual, allPages, &operationError),
                     "the completed resident texture batch should be observable");
            t.IsTrue(actual == expected,
                     "one resident texture batch must match sequential CPU draws exactly");
            const GsVulkanServiceStatistics finalStatistics =
                service->statistics();
            t.Equals(finalStatistics.validationErrors, 0u,
                     "resident texture batching must remain validation-clean");
            t.Equals(finalStatistics.validationWarnings, 0u,
                     "resident texture batching should emit no validation warnings");
            t.IsTrue(service->healthy(),
                     "accepted and rejected texture batches should leave the service healthy");

            service->shutdown();
            operationError.clear();
            t.IsFalse(service->executeResidentNearestCt32Sprites(
                          sprites, &operationError),
                      "a stopped service must reject resident texture batches");
            t.IsFalse(operationError.empty(),
                      "post-shutdown texture rejection should retain a diagnostic");
        });

        tc.Run("Vulkan resident linear CT32 batches preserve texture dependencies", [](TestCase &t)
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
                         "an unavailable host should skip resident linear batching cleanly");
                t.IsFalse(creationReport.ready(),
                          "a skipped linear batch service should retain its capability result");
                return;
            }
            t.IsNotNull(service.get(),
                        "a suitable device should create the resident linear batch service");
            t.IsTrue(creationError.empty(),
                     "successful linear batch service creation should clear its diagnostic");
            if (!service)
                return;

            constexpr std::array<uint32_t, 4> framebufferPages{{
                40u, 41u, 197u, 509u,
            }};
            std::vector<GsVulkanLinearCt32Sprite> sprites;
            for (size_t index = 0u; index < framebufferPages.size(); ++index)
            {
                const GsDrawCommand command =
                    makeLinearCt32RepeatSpriteCommand(
                        36'000u + index,
                        framebufferPages[index], 2u,
                        512u, 2u, 6u, 5u,
                        {3u, 12u, 2u, 13u}, {128u, 96u},
                        {120u, 376u}, {88u, 344u},
                        {0u, 249u}, {0u, 137u});
                GsVulkanLinearCt32Sprite sprite{};
                const GsBackendDecision decision =
                    prepareGsVulkanLinearCt32Sprite(command, sprite);
                if (!decision.supported)
                {
                    t.Fail(
                        "resident linear CT32 fixture was rejected as " +
                        std::string(gsFallbackReasonName(decision.reason)));
                    return;
                }
                sprites.push_back(sprite);
            }

            const GsVramPageMask sharedReadPages =
                gsVramPagesForSurfaceRect(
                    sprites.front().textureBaseBlock,
                    sprites.front().textureWidth,
                    static_cast<uint8_t>(GSMem::C32),
                    0u, 0u,
                    sprites.front().textureMaskU + 1u,
                    sprites.front().textureMaskV + 1u);
            t.Equals(sharedReadPages.count(), size_t{1u},
                     "the shared linear texture should conservatively read one page");
            t.IsTrue(sharedReadPages.test(16u),
                     "the shared linear texture should retain its physical page");

            std::string operationError;
            const GsVulkanServiceStatistics beforeRejections =
                service->statistics();
            const auto expectRejected =
                [&](std::span<const GsVulkanLinearCt32Sprite> rejected,
                    const std::string &label)
            {
                operationError.clear();
                t.IsFalse(service->executeResidentLinearCt32Sprites(
                              rejected, &operationError),
                          label + " should fail before worker submission");
                t.IsFalse(operationError.empty(),
                          label + " should retain a diagnostic");
            };
            expectRejected({}, "empty resident linear batch");
            std::vector<GsVulkanLinearCt32Sprite> oversized(
                GS_VULKAN_MAX_RESIDENT_LINEAR_CT32_BATCH + 1u,
                sprites.front());
            expectRejected(oversized, "oversized resident linear batch");
            std::vector<GsVulkanLinearCt32Sprite> invalid = sprites;
            invalid[2].textureWrapU |= 1u << 31u;
            expectRejected(invalid, "invalid resident linear batch member");

            const GsVulkanServiceStatistics afterRejections =
                service->statistics();
            t.Equals(afterRejections.queueSubmissions,
                     beforeRejections.queueSubmissions,
                     "caller-rejected linear batches must not submit GPU work");
            t.Equals(afterRejections.linearCt32SpriteDrawsFailed,
                     beforeRejections.linearCt32SpriteDrawsFailed,
                     "caller-rejected linear batches must not count worker failures");
            t.Equals(
                afterRejections.residentLinearCt32SpriteBatchesFailed,
                beforeRejections.residentLinearCt32SpriteBatchesFailed,
                "caller-rejected linear batches must not count failed worker batches");

            const GsVulkanDeviceReport *selected =
                creationReport.selectedDevice();
            t.IsNotNull(selected,
                        "the linear batch service should retain its selected device");
            if (!selected)
                return;
            if (!selected->exactLinearCt32Sprite)
            {
                expectRejected(sprites,
                               "capability-gated resident linear batch");
                t.Equals(
                    service->statistics().linearCt32SpriteDrawsFailed,
                    0ull,
                    "linear capability rejection must not enter the worker");
                t.IsTrue(service->healthy(),
                         "unsupported linear batching must not poison the base service");
                service->shutdown();
                return;
            }

            const std::vector<uint8_t> initial =
                makeVramPattern(0x524C4931u);
            std::vector<uint8_t> expected = initial;
            uint64_t expectedPixels = 0u;
            for (const GsVulkanLinearCt32Sprite &sprite : sprites)
            {
                applyLinearCt32SpriteCpu(expected, sprite);
                expectedPixels +=
                    static_cast<uint64_t>(
                        sprite.boundsX1 - sprite.boundsX0) *
                    static_cast<uint64_t>(
                        sprite.boundsY1 - sprite.boundsY0);
            }
            GsVramPageMask allPages;
            allPages.setAll();
            t.IsTrue(service->uploadVramPages(
                         initial, allPages, &operationError),
                     "the linear batch fixture should establish resident VRAM");
            if (!operationError.empty())
                t.Fail("resident linear batch upload failed: " + operationError);

            const GsVulkanServiceStatistics beforeBatch =
                service->statistics();
            operationError.clear();
            t.IsTrue(service->executeResidentLinearCt32Sprites(
                         sprites, &operationError),
                     "shared-read resident linear sprites should execute as one batch");
            t.IsTrue(operationError.empty(),
                     "successful resident linear batching should clear its diagnostic");
            const GsVulkanServiceStatistics afterBatch =
                service->statistics();
            t.Equals(afterBatch.queueSubmissions -
                         beforeBatch.queueSubmissions,
                     1ull,
                     "one linear batch should use one Vulkan queue submission");
            t.Equals(afterBatch.shaderDispatches -
                         beforeBatch.shaderDispatches,
                     static_cast<uint64_t>(sprites.size()),
                     "each linear batch member should retain one compute dispatch");
            t.Equals(afterBatch.pipelineBarriers -
                         beforeBatch.pipelineBarriers,
                     2ull,
                     "one linear batch should use one prepare and one completion barrier");
            t.Equals(afterBatch.pipelineBinds - beforeBatch.pipelineBinds,
                     1ull,
                     "one linear batch should bind the exact pipeline once");
            t.Equals(afterBatch.pipelineCacheHits -
                         beforeBatch.pipelineCacheHits,
                     1ull,
                     "one linear batch should reuse its prebuilt pipeline once");
            t.Equals(afterBatch.fenceWaits - beforeBatch.fenceWaits,
                     1ull,
                     "one linear batch should wait on one submitted fence");
            t.Equals(afterBatch.linearCt32SpriteDrawsCompleted -
                         beforeBatch.linearCt32SpriteDrawsCompleted,
                     static_cast<uint64_t>(sprites.size()),
                     "linear batch statistics should count every completed sprite");
            t.Equals(afterBatch.linearCt32SpritePixelsExecuted -
                         beforeBatch.linearCt32SpritePixelsExecuted,
                     expectedPixels,
                     "linear batch statistics should count every covered pixel");
            t.Equals(
                afterBatch.residentLinearCt32SpriteBatchesCompleted -
                    beforeBatch.residentLinearCt32SpriteBatchesCompleted,
                1ull,
                "the worker should complete one resident linear batch");
            t.Equals(afterBatch.largestResidentLinearCt32SpriteBatch,
                     static_cast<uint64_t>(sprites.size()),
                     "the service should retain its largest linear batch");

            std::vector<uint8_t> actual(
                GS_VULKAN_VRAM_SIZE, 0xA5u);
            t.IsTrue(service->downloadVramPages(
                         actual, allPages, &operationError),
                     "the completed resident linear batch should be observable");
            t.IsTrue(actual == expected,
                     "one resident linear batch must match sequential CPU draws exactly");

            GsVulkanLinearCt32Sprite secondWrite = sprites.front();
            secondWrite.fixedBaseU += 65'536;
            GsVulkanLinearCt32Sprite writeThenRead = sprites[1];
            writeThenRead.textureBaseBlock =
                sprites.front().framebufferBaseBlock;
            GsVulkanLinearCt32Sprite readThenWrite = sprites[1];
            readThenWrite.framebufferBaseBlock =
                sprites.front().textureBaseBlock;
            readThenWrite.textureBaseBlock = 60u * 32u;
            const std::array<GsVulkanLinearCt32Sprite, 4> ordered{{
                sprites.front(), secondWrite,
                writeThenRead, readThenWrite,
            }};
            std::vector<uint8_t> orderedExpected = initial;
            for (const GsVulkanLinearCt32Sprite &sprite : ordered)
                applyLinearCt32SpriteCpu(orderedExpected, sprite);
            operationError.clear();
            t.IsTrue(service->uploadVramPages(
                         initial, allPages, &operationError),
                     "the ordered linear fixture should restore resident VRAM");
            if (!operationError.empty())
                t.Fail("ordered linear batch upload failed: " + operationError);

            const GsVulkanServiceStatistics beforeOrdered =
                service->statistics();
            operationError.clear();
            t.IsTrue(service->executeResidentLinearCt32Sprites(
                         ordered, &operationError),
                     "dependent linear draws should execute in guest order");
            t.IsTrue(operationError.empty(),
                     "ordered dependent execution should clear its diagnostic");
            const GsVulkanServiceStatistics afterOrdered =
                service->statistics();
            t.Equals(afterOrdered.queueSubmissions -
                         beforeOrdered.queueSubmissions,
                     1ull,
                     "one ordered linear batch should submit once");
            t.Equals(afterOrdered.shaderDispatches -
                         beforeOrdered.shaderDispatches,
                     static_cast<uint64_t>(ordered.size()),
                     "ordered batching should retain one dispatch per draw");
            t.Equals(afterOrdered.pipelineBarriers -
                         beforeOrdered.pipelineBarriers,
                     4ull,
                     "two dependencies should add two barriers to the batch envelope");
            t.Equals(afterOrdered.pipelineBinds -
                         beforeOrdered.pipelineBinds,
                     1ull,
                     "ordered batching should bind the linear pipeline once");
            t.Equals(afterOrdered.fenceWaits - beforeOrdered.fenceWaits,
                     1ull,
                     "ordered batching should wait on one fence");
            std::vector<uint8_t> orderedActual(
                GS_VULKAN_VRAM_SIZE, 0xA5u);
            t.IsTrue(service->downloadVramPages(
                         orderedActual, allPages, &operationError),
                     "the ordered dependent image should be observable");
            t.IsTrue(orderedActual == orderedExpected,
                     "dependency barriers must preserve sequential CPU order");
            const GsVulkanServiceStatistics finalStatistics =
                service->statistics();
            t.Equals(finalStatistics.validationErrors, 0u,
                     "resident linear batching must remain validation-clean");
            t.Equals(finalStatistics.validationWarnings, 0u,
                     "resident linear batching should emit no validation warnings");
            t.IsTrue(service->healthy(),
                     "accepted and rejected linear batches should leave the service healthy");

            service->shutdown();
            operationError.clear();
            t.IsFalse(service->executeResidentLinearCt32Sprites(
                          sprites, &operationError),
                      "a stopped service must reject resident linear batches");
            t.IsFalse(operationError.empty(),
                      "post-shutdown linear rejection should retain a diagnostic");
        });

        tc.Run("Vulkan resident triangle batches require disjoint pages", [](TestCase &t)
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
                         "an unavailable host should skip triangle batching cleanly");
                t.IsFalse(creationReport.ready(),
                          "a skipped triangle batch service should retain its capability result");
                return;
            }
            t.IsNotNull(service.get(),
                        "a generally suitable device should create the triangle batch service");
            t.IsTrue(creationError.empty(),
                     "successful triangle batch service creation should clear its diagnostic");
            if (!service)
                return;

            constexpr std::array<uint32_t, 4> physicalPages{{
                5u, 41u, 197u, 509u,
            }};
            std::vector<GsVulkanCt32Triangle> triangles;
            for (size_t index = 0u; index < physicalPages.size(); ++index)
            {
                const uint16_t phase = static_cast<uint16_t>(index * 3u + 1u);
                const GsDrawCommand command = makeCt32TriangleCommand(
                    30'000u + index, physicalPages[index], 1u,
                    {0u, 31u, 0u, 31u}, {0u, 0u},
                    {
                        static_cast<uint16_t>(2u * 16u + phase),
                        static_cast<uint16_t>(22u * 16u + phase),
                        static_cast<uint16_t>(6u * 16u + phase),
                    },
                    {
                        static_cast<uint16_t>(3u * 16u + phase),
                        static_cast<uint16_t>(8u * 16u + phase),
                        static_cast<uint16_t>(25u * 16u + phase),
                    },
                    0x80402010u + static_cast<uint32_t>(index));
                GsVulkanCt32Triangle triangle{};
                const GsBackendDecision decision =
                    prepareGsVulkanCt32Triangle(command, triangle);
                if (!decision.supported)
                {
                    t.Fail(
                        "disjoint triangle fixture was rejected as " +
                        std::string(gsFallbackReasonName(decision.reason)));
                    return;
                }
                const GsVramPageMask writePages =
                    gsVramPagesForSurfaceRect(
                        triangle.framebufferBaseBlock,
                        triangle.framebufferWidth,
                        static_cast<uint8_t>(GSMem::C32),
                        triangle.boundsX0,
                        triangle.boundsY0,
                        triangle.boundsX1 - triangle.boundsX0,
                        triangle.boundsY1 - triangle.boundsY0);
                t.Equals(writePages.count(), size_t{1u},
                         "each triangle fixture should own exactly one physical page");
                t.IsTrue(writePages.test(physicalPages[index]),
                         "each triangle fixture should map to its selected page");
                triangles.push_back(triangle);
            }

            const GsVulkanServiceStatistics beforeRejections =
                service->statistics();
            std::string operationError;
            const auto expectRejected =
                [&](std::span<const GsVulkanCt32Triangle> rejected,
                    const std::string &label)
            {
                operationError.clear();
                t.IsFalse(service->executeResidentCt32Triangles(
                              rejected, &operationError),
                          label + " should fail before worker submission");
                t.IsFalse(operationError.empty(),
                          label + " should retain a diagnostic");
            };
            expectRejected({}, "empty resident triangle batch");
            std::vector<GsVulkanCt32Triangle> oversized(
                GS_VULKAN_MAX_RESIDENT_TRIANGLE_BATCH + 1u,
                triangles.front());
            expectRejected(oversized, "oversized resident triangle batch");
            std::vector<GsVulkanCt32Triangle> invalid = triangles;
            invalid[2].reserved0 = 1u;
            expectRejected(invalid, "invalid resident triangle batch member");
            const std::array<GsVulkanCt32Triangle, 2> overlapping{{
                triangles.front(), triangles.front(),
            }};
            expectRejected(overlapping,
                           "physical-page-overlapping triangle batch");

            const GsVulkanServiceStatistics afterRejections =
                service->statistics();
            t.Equals(afterRejections.queueSubmissions,
                     beforeRejections.queueSubmissions,
                     "caller-rejected triangle batches must not submit GPU work");
            t.Equals(afterRejections.triangleDrawsFailed,
                     beforeRejections.triangleDrawsFailed,
                     "caller-rejected triangle batches must not count worker failures");
            t.Equals(afterRejections.residentTriangleBatchesFailed,
                     beforeRejections.residentTriangleBatchesFailed,
                     "caller-rejected triangle batches must not count failed worker batches");

            const GsVulkanDeviceReport *selected =
                creationReport.selectedDevice();
            t.IsNotNull(selected,
                        "the triangle batch service should retain its selected device");
            if (!selected)
                return;
            if (!selected->exactCt32Triangle)
            {
                expectRejected(triangles,
                               "capability-gated resident triangle batch");
                t.Equals(service->statistics().triangleDrawsFailed, 0ull,
                         "capability rejection must not enter the worker");
                t.IsTrue(service->healthy(),
                         "unsupported triangle batching must not poison the base service");
                service->shutdown();
                return;
            }

            const std::vector<uint8_t> initial =
                makeVramPattern(0x54524231u);
            std::vector<uint8_t> expected = initial;
            uint64_t expectedCandidatePixels = 0u;
            for (const GsVulkanCt32Triangle &triangle : triangles)
            {
                applyCt32TriangleCpu(expected, triangle);
                expectedCandidatePixels +=
                    static_cast<uint64_t>(
                        triangle.boundsX1 - triangle.boundsX0) *
                    static_cast<uint64_t>(
                        triangle.boundsY1 - triangle.boundsY0);
            }
            GsVramPageMask allPages;
            allPages.setAll();
            t.IsTrue(service->uploadVramPages(
                         initial, allPages, &operationError),
                     "the triangle batch fixture should establish resident VRAM");
            if (!operationError.empty())
                t.Fail("resident triangle batch upload failed: " + operationError);

            const GsVulkanServiceStatistics beforeBatch =
                service->statistics();
            operationError.clear();
            t.IsTrue(service->executeResidentCt32Triangles(
                         triangles, &operationError),
                     "four disjoint resident triangles should execute as one batch");
            t.IsTrue(operationError.empty(),
                     "successful resident triangle batching should clear its diagnostic");
            const GsVulkanServiceStatistics afterBatch =
                service->statistics();
            t.Equals(afterBatch.queueSubmissions -
                         beforeBatch.queueSubmissions,
                     1ull,
                     "one triangle batch should use one Vulkan queue submission");
            t.Equals(afterBatch.shaderDispatches -
                         beforeBatch.shaderDispatches,
                     static_cast<uint64_t>(triangles.size()),
                     "each triangle batch member should retain one compute dispatch");
            t.Equals(afterBatch.pipelineBarriers -
                         beforeBatch.pipelineBarriers,
                     2ull,
                     "one triangle batch should use one prepare and one completion barrier");
            t.Equals(afterBatch.pipelineBinds - beforeBatch.pipelineBinds,
                     1ull,
                     "one triangle batch should bind the exact pipeline once");
            t.Equals(afterBatch.pipelineCacheHits -
                         beforeBatch.pipelineCacheHits,
                     1ull,
                     "one triangle batch should reuse its prebuilt pipeline once");
            t.Equals(afterBatch.fenceWaits - beforeBatch.fenceWaits,
                     1ull,
                     "one triangle batch should wait on one submitted fence");
            t.Equals(afterBatch.triangleDrawsCompleted -
                         beforeBatch.triangleDrawsCompleted,
                     static_cast<uint64_t>(triangles.size()),
                     "batch statistics should count every completed triangle");
            t.Equals(afterBatch.triangleCandidatePixelsExecuted -
                         beforeBatch.triangleCandidatePixelsExecuted,
                     expectedCandidatePixels,
                     "batch statistics should count every conservative candidate pixel");
            t.Equals(afterBatch.residentTriangleBatchesCompleted -
                         beforeBatch.residentTriangleBatchesCompleted,
                     1ull,
                     "the worker should complete one resident triangle batch");
            t.Equals(afterBatch.largestResidentTriangleBatch,
                     static_cast<uint64_t>(triangles.size()),
                     "the service should retain its largest triangle batch");

            std::vector<uint8_t> actual(
                GS_VULKAN_VRAM_SIZE, 0xA5u);
            t.IsTrue(service->downloadVramPages(
                         actual, allPages, &operationError),
                     "the completed resident triangle batch should be observable");
            t.IsTrue(actual == expected,
                     "one triangle batch must match sequential CPU draws exactly");
            const GsVulkanServiceStatistics finalStatistics =
                service->statistics();
            t.Equals(finalStatistics.validationErrors, 0u,
                     "resident triangle batching must remain validation-clean");
            t.Equals(finalStatistics.validationWarnings, 0u,
                     "resident triangle batching should emit no validation warnings");
            t.IsTrue(service->healthy(),
                     "accepted and rejected triangle batches should leave the service healthy");
            service->shutdown();
        });

        tc.Run("Vulkan CT32 triangles match exhaustive and random software batches", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            using FixedVertex = std::array<int32_t, 2>;
            using FixedTriangle = std::array<FixedVertex, 3>;
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
            constexpr GSXYOffsetReg edgeXyoffset{128u, 128u};

            // Degenerate geometry never reaches a GPU record. Cover repeated,
            // horizontal, vertical, and diagonal collinearity across every
            // fractional phase, scissor, and vertex order.
            constexpr std::array<FixedTriangle, 4> degenerateShapes{{
                {{{32, 32}, {32, 32}, {32, 32}}},
                {{{32, 48}, {256, 48}, {144, 48}}},
                {{{64, 16}, {64, 256}, {64, 128}}},
                {{{16, 16}, {256, 256}, {128, 128}}},
            }};
            uint64_t sequence = 40'000u;
            uint64_t degenerateCases = 0u;
            for (const FixedTriangle &shape : degenerateShapes)
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
                                        source[0] + phaseX +
                                        edgeXyoffset.ofx);
                                    rawY[vertex] = static_cast<uint16_t>(
                                        source[1] + phaseY +
                                        edgeXyoffset.ofy);
                                }
                                const GsDrawCommand command =
                                    makeCt32TriangleCommand(
                                        sequence++, 0u, 1u,
                                        scissor, edgeXyoffset,
                                        rawX, rawY, 0xD06E0000u);
                                GsVulkanCt32Triangle record{
                                    1u, 2u, 3u, 4u, 5u, 6u,
                                    7, 8, 9, 10, 11, 12,
                                    13u, 14u, 15u, 16u};
                                const GsVulkanCt32Triangle sentinel = record;
                                const GsBackendDecision decision =
                                    prepareGsVulkanCt32Triangle(
                                        command, record);
                                if (decision.supported ||
                                    decision.reason !=
                                        GsFallbackReason::EmptyBounds ||
                                    !(record == sentinel))
                                {
                                    t.Fail(
                                        "degenerate triangle corpus did not fail as an empty bound");
                                    return;
                                }
                                ++degenerateCases;
                            }
                        }
                    }
                }
            }
            t.Equals(degenerateCases, 24'576ull,
                     "the exhaustive degenerate corpus should retain every case");

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
                         "an unavailable host should skip the GPU triangle corpus cleanly");
                t.IsFalse(creationReport.ready(),
                          "a skipped corpus service should retain its capability result");
                return;
            }
            t.IsNotNull(service.get(),
                        "a generally suitable device should create the triangle corpus service");
            t.IsTrue(creationError.empty(),
                     "successful corpus service creation should clear its diagnostic");
            if (!service)
                return;
            const GsVulkanDeviceReport *selected =
                creationReport.selectedDevice();
            t.IsNotNull(selected,
                        "the triangle corpus service should retain its selected device");
            if (!selected)
                return;
            if (!selected->exactCt32Triangle)
            {
                service->shutdown();
                return;
            }

            std::vector<uint8_t> softwareVram(
                GS_VULKAN_VRAM_SIZE, 0u);
            GS software;
            software.init(
                softwareVram.data(),
                static_cast<uint32_t>(softwareVram.size()), nullptr);
            software.setDebugHistoryPaused(true);

            GsVramPageMask allPages;
            allPages.setAll();
            std::vector<GsVulkanCt32Triangle> batch;
            batch.reserve(GS_VULKAN_MAX_RESIDENT_TRIANGLE_BATCH);
            std::vector<uint8_t> initial;
            uint64_t acceptedCases = 0u;
            uint64_t candidatePixels = 0u;
            uint64_t completedBatches = 0u;
            std::string operationError;

            const auto beginBatch = [&]()
            {
                initial = makeVramPattern(
                    0x54514331u ^
                    static_cast<uint32_t>(completedBatches *
                                          0x9E3779B9u));
                std::copy(
                    initial.begin(), initial.end(), softwareVram.begin());
            };
            const auto flushBatch = [&]()
            {
                if (batch.empty())
                    return true;
                if (batch.size() !=
                    GS_VULKAN_MAX_RESIDENT_TRIANGLE_BATCH)
                {
                    t.Fail("triangle qualification produced a partial batch");
                    return false;
                }

                software.flushRenderBatch();
                operationError.clear();
                if (!service->uploadVramPages(
                        initial, allPages, &operationError))
                {
                    t.Fail(
                        "triangle qualification upload failed: " +
                        operationError);
                    return false;
                }
                operationError.clear();
                if (!service->executeResidentCt32Triangles(
                        batch, &operationError))
                {
                    t.Fail(
                        "triangle qualification batch failed: " +
                        operationError);
                    return false;
                }

                std::vector<uint8_t> actual(
                    GS_VULKAN_VRAM_SIZE, 0xA5u);
                operationError.clear();
                if (!service->downloadVramPages(
                        actual, allPages, &operationError))
                {
                    t.Fail(
                        "triangle qualification download failed: " +
                        operationError);
                    return false;
                }
                if (actual != softwareVram)
                {
                    const auto mismatch = std::mismatch(
                        actual.begin(), actual.end(),
                        softwareVram.begin());
                    const size_t byte = static_cast<size_t>(
                        mismatch.first - actual.begin());
                    const size_t page = byte / GS_VRAM_PAGE_SIZE;
                    const uint64_t firstCase =
                        acceptedCases - batch.size();
                    std::ostringstream message;
                    message
                        << "triangle qualification batch "
                        << completedBatches
                        << " first disagreed at byte " << byte
                        << " physical page " << page
                        << " corpus case " << (firstCase + page);
                    t.Fail(message.str());
                    return false;
                }
                ++completedBatches;
                batch.clear();
                return true;
            };
            const auto appendCase =
                [&](const GsDrawCommand &command,
                    const GsVulkanCt32Triangle &triangle)
            {
                if (batch.empty())
                    beginBatch();
                const size_t physicalPage = batch.size();
                const GsVramPageMask writePages =
                    gsVramPagesForSurfaceRect(
                        triangle.framebufferBaseBlock,
                        triangle.framebufferWidth,
                        static_cast<uint8_t>(GSMem::C32),
                        triangle.boundsX0,
                        triangle.boundsY0,
                        triangle.boundsX1 - triangle.boundsX0,
                        triangle.boundsY1 - triangle.boundsY0);
                if (writePages.count() != 1u ||
                    !writePages.test(physicalPage))
                {
                    t.Fail(
                        "triangle qualification record escaped its assigned page");
                    return false;
                }
                drawFlatCt32TriangleCommand(
                    software, command, triangle.rgba);
                batch.push_back(triangle);
                ++acceptedCases;
                candidatePixels +=
                    static_cast<uint64_t>(
                        triangle.boundsX1 - triangle.boundsX0) *
                    static_cast<uint64_t>(
                        triangle.boundsY1 - triangle.boundsY0);
                return batch.size() !=
                           GS_VULKAN_MAX_RESIDENT_TRIANGLE_BATCH ||
                       flushBatch();
            };

            constexpr std::array<FixedTriangle, 3> edgeShapes{{
                {{{-64, 17}, {239, 49}, {33, 255}}},
                {{{32, 32}, {256, 32}, {32, 256}}},
                {{{65, 65}, {81, 241}, {257, 97}}},
            }};
            uint64_t exhaustiveCases = 0u;
            uint32_t colorState = 0xE6A1C4D3u;
            for (const FixedTriangle &shape : edgeShapes)
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
                                        source[0] + phaseX +
                                        edgeXyoffset.ofx);
                                    rawY[vertex] = static_cast<uint16_t>(
                                        source[1] + phaseY +
                                        edgeXyoffset.ofy);
                                }
                                const uint32_t physicalPage =
                                    static_cast<uint32_t>(batch.size());
                                const GsDrawCommand command =
                                    makeCt32TriangleCommand(
                                        sequence++, physicalPage, 1u,
                                        scissor, edgeXyoffset,
                                        rawX, rawY,
                                        nextRandom(colorState));
                                GsVulkanCt32Triangle triangle{};
                                const GsBackendDecision decision =
                                    prepareGsVulkanCt32Triangle(
                                        command, triangle);
                                if (!decision.supported)
                                {
                                    t.Fail(
                                        "exhaustive GPU edge case was rejected as " +
                                        std::string(gsFallbackReasonName(
                                            decision.reason)));
                                    return;
                                }
                                if (!appendCase(command, triangle))
                                    return;
                                ++exhaustiveCases;
                            }
                        }
                    }
                }
            }
            t.Equals(exhaustiveCases, 18'432ull,
                     "the GPU edge corpus should retain every exhaustive case");
            t.Equals(completedBatches, 36ull,
                     "the exhaustive corpus should use 36 full-page batches");

            constexpr uint64_t randomTarget = 4'096u;
            constexpr uint32_t randomSeed = 0xC001D00Du;
            uint32_t randomState = randomSeed;
            uint64_t randomAttempts = 0u;
            uint64_t randomCases = 0u;
            uint64_t randomEmpty = 0u;
            while (randomCases < randomTarget &&
                   randomAttempts < 100'000u)
            {
                ++randomAttempts;
                const GSXYOffsetReg xyoffset{
                    static_cast<uint16_t>(
                        0x4000u + (nextRandom(randomState) & 0x0FFFu)),
                    static_cast<uint16_t>(
                        0x4000u + (nextRandom(randomState) & 0x0FFFu)),
                };
                std::array<uint16_t, 3> rawX{};
                std::array<uint16_t, 3> rawY{};
                for (size_t vertex = 0u;
                     vertex < rawX.size(); ++vertex)
                {
                    const int32_t fixedX =
                        static_cast<int32_t>(
                            nextRandom(randomState) % (80u * 16u)) -
                        8 * 16;
                    const int32_t fixedY =
                        static_cast<int32_t>(
                            nextRandom(randomState) % (48u * 16u)) -
                        8 * 16;
                    rawX[vertex] = static_cast<uint16_t>(
                        fixedX + xyoffset.ofx);
                    rawY[vertex] = static_cast<uint16_t>(
                        fixedY + xyoffset.ofy);
                }
                const GSScissorReg scissor{
                    static_cast<uint16_t>(nextRandom(randomState) % 16u),
                    static_cast<uint16_t>(
                        48u + nextRandom(randomState) % 16u),
                    static_cast<uint16_t>(nextRandom(randomState) % 8u),
                    static_cast<uint16_t>(
                        24u + nextRandom(randomState) % 8u),
                };
                const uint32_t physicalPage =
                    static_cast<uint32_t>(batch.size());
                const GsDrawCommand command = makeCt32TriangleCommand(
                    sequence++, physicalPage, 1u,
                    scissor, xyoffset, rawX, rawY,
                    nextRandom(randomState));
                GsVulkanCt32Triangle triangle{};
                const GsBackendDecision decision =
                    prepareGsVulkanCt32Triangle(command, triangle);
                if (!decision.supported)
                {
                    if (decision.reason != GsFallbackReason::EmptyBounds)
                    {
                        t.Fail(
                            "seeded random triangle was rejected as " +
                            std::string(gsFallbackReasonName(
                                decision.reason)));
                        return;
                    }
                    ++randomEmpty;
                    continue;
                }
                if (!appendCase(command, triangle))
                    return;
                ++randomCases;
            }
            t.Equals(randomCases, randomTarget,
                     "the fixed seed should produce every requested random draw");
            t.IsTrue(randomEmpty != 0u,
                     "the fixed seed should exercise clipped or degenerate fallback");
            t.Equals(batch.size(), size_t{0u},
                     "the complete qualification corpus should end on a full batch");
            t.Equals(completedBatches, 44ull,
                     "exhaustive and random cases should use 44 full batches");
            t.Equals(acceptedCases,
                     exhaustiveCases + randomCases,
                     "every accepted record should be assigned to one batch");

            const GsVulkanServiceStatistics statistics =
                service->statistics();
            t.Equals(statistics.triangleDrawsCompleted, acceptedCases,
                     "the GPU should complete every qualification record");
            t.Equals(statistics.triangleDrawsFailed, 0ull,
                     "the qualification corpus should have no worker failures");
            t.Equals(statistics.triangleCandidatePixelsExecuted,
                     candidatePixels,
                     "qualification statistics should retain all candidate pixels");
            t.Equals(statistics.residentTriangleBatchesCompleted,
                     completedBatches,
                     "the worker should complete every qualification batch");
            t.Equals(statistics.residentTriangleBatchesFailed, 0ull,
                     "no qualification batch should fail");
            t.Equals(statistics.largestResidentTriangleBatch,
                     static_cast<uint64_t>(
                         GS_VULKAN_MAX_RESIDENT_TRIANGLE_BATCH),
                     "qualification should exercise the maximum safe batch");
            t.Equals(statistics.queueSubmissions,
                     completedBatches * 3u,
                     "each corpus batch should use upload draw and download submissions");
            t.Equals(statistics.shaderDispatches, acceptedCases,
                     "every accepted record should use one production dispatch");
            t.Equals(statistics.pagesUploaded,
                     completedBatches * GS_VRAM_PAGE_COUNT,
                     "each corpus batch should reset every independent page");
            t.Equals(statistics.pagesDownloaded,
                     completedBatches * GS_VRAM_PAGE_COUNT,
                     "each corpus batch should compare every independent page");
            t.Equals(statistics.validationErrors, 0u,
                     "the full triangle corpus must remain validation-clean");
            t.Equals(statistics.validationWarnings, 0u,
                     "the full triangle corpus should emit no validation warnings");
            t.IsTrue(service->healthy(),
                     "the exhaustive and random corpus should leave the service healthy");
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
