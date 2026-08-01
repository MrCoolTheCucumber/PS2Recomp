#include "MiniTest.h"
#include "runtime/ps2_memory.h"
#include "ps2_runtime.h"
#include "ps2_stubs.h"
#include "ps2_syscalls.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_gs_backend.h"
#include "runtime/ps2_gs_memory.h"
#include "runtime/ps2_gs_rasterizer.h"
#include "ps2_gs_rasterizer_detail.h"
#include "runtime/ps2_gs_psmct32.h"
#include "runtime/ps2_gs_psmt4.h"
#include "runtime/ps2_gs_psmt8.h"
#include "Stubs/Helpers/Support.h"
#include "Stubs/GS.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

using namespace ps2_syscalls;

namespace
{
    std::atomic<uint32_t> g_gsSyncCallbackHits{0u};
    std::atomic<uint32_t> g_gsSyncCallbackLastTick{0u};

    static_assert(sizeof(GsImageMem) == 12, "GsImageMem size mismatch");

    void setRegU32(R5900Context &ctx, int reg, uint32_t value)
    {
        ctx.r[reg] = _mm_set_epi64x(0, static_cast<int64_t>(value));
    }

    void setRegU64(R5900Context &ctx, int reg, uint64_t value)
    {
        ctx.r[reg] = _mm_set_epi64x(0, static_cast<int64_t>(value));
    }

    uint32_t getRegU32Test(const R5900Context &ctx, int reg)
    {
        return ::getRegU32(&ctx, reg);
    }

    uint64_t getReturnU64(const R5900Context &ctx)
    {
        const uint64_t lo = static_cast<uint64_t>(getRegU32Test(ctx, 2));
        const uint64_t hi = static_cast<uint64_t>(getRegU32Test(ctx, 3));
        return lo | (hi << 32);
    }

    uint64_t makeGifTag(uint16_t nloop, uint8_t flg, uint8_t nreg, bool eop = true)
    {
        uint64_t tag = static_cast<uint64_t>(nloop & 0x7FFFu);
        if (eop)
            tag |= (1ull << 15);
        tag |= (static_cast<uint64_t>(flg & 0x3u) << 58);
        tag |= (static_cast<uint64_t>(nreg & 0xFu) << 60);
        return tag;
    }

    void appendU64(std::vector<uint8_t> &dst, uint64_t value)
    {
        const size_t pos = dst.size();
        dst.resize(pos + sizeof(uint64_t));
        std::memcpy(dst.data() + pos, &value, sizeof(uint64_t));
    }

    void appendGifAd(std::vector<uint8_t> &dst, uint64_t value, uint64_t reg)
    {
        appendU64(dst, value);
        appendU64(dst, reg);
    }

    template <typename Predicate>
    bool waitUntil(Predicate pred, std::chrono::milliseconds timeout)
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (pred())
            {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        return pred();
    }

    void testGsSyncVCallback(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;

        g_gsSyncCallbackLastTick.store(getRegU32(ctx, 4), std::memory_order_relaxed);
        g_gsSyncCallbackHits.fetch_add(1u, std::memory_order_relaxed);
        ctx->pc = 0u;
    }

    void writeGsImageTest(uint8_t *rdram, uint32_t addr, const GsImageMem &image)
    {
        std::memcpy(rdram + addr, &image, sizeof(image));
    }

    void writeGsImageTest(std::vector<uint8_t> &rdram, uint32_t addr, const GsImageMem &image)
    {
        writeGsImageTest(rdram.data(), addr, image);
    }

    void writePSMT4Texel(std::vector<uint8_t> &vram, uint32_t tbp, uint32_t tbw, uint32_t x, uint32_t y, uint8_t index)
    {
        const uint32_t nibbleAddr = GSPSMT4::addrPSMT4(tbp, tbw, x, y);
        const uint32_t byteOff = nibbleAddr >> 1;
        uint8_t &packed = vram[byteOff];
        if ((nibbleAddr & 1u) != 0u)
        {
            packed = static_cast<uint8_t>((packed & 0x0Fu) | ((index & 0x0Fu) << 4));
        }
        else
        {
            packed = static_cast<uint8_t>((packed & 0xF0u) | (index & 0x0Fu));
        }
    }

    uint32_t referenceAddrPSMT4(uint32_t block, uint32_t width, uint32_t x, uint32_t y)
    {
        static constexpr uint8_t kBlockTable4[8][4] = {
            {0, 2, 8, 10},
            {1, 3, 9, 11},
            {4, 6, 12, 14},
            {5, 7, 13, 15},
            {16, 18, 24, 26},
            {17, 19, 25, 27},
            {20, 22, 28, 30},
            {21, 23, 29, 31},
        };

        static constexpr uint16_t kColumnTable4[16][32] = {
            {0, 8, 32, 40, 64, 72, 96, 104, 2, 10, 34, 42, 66, 74, 98, 106, 4, 12, 36, 44, 68, 76, 100, 108, 6, 14, 38, 46, 70, 78, 102, 110},
            {16, 24, 48, 56, 80, 88, 112, 120, 18, 26, 50, 58, 82, 90, 114, 122, 20, 28, 52, 60, 84, 92, 116, 124, 22, 30, 54, 62, 86, 94, 118, 126},
            {65, 73, 97, 105, 1, 9, 33, 41, 67, 75, 99, 107, 3, 11, 35, 43, 69, 77, 101, 109, 5, 13, 37, 45, 71, 79, 103, 111, 7, 15, 39, 47},
            {81, 89, 113, 121, 17, 25, 49, 57, 83, 91, 115, 123, 19, 27, 51, 59, 85, 93, 117, 125, 21, 29, 53, 61, 87, 95, 119, 127, 23, 31, 55, 63},
            {192, 200, 224, 232, 128, 136, 160, 168, 194, 202, 226, 234, 130, 138, 162, 170, 196, 204, 228, 236, 132, 140, 164, 172, 198, 206, 230, 238, 134, 142, 166, 174},
            {208, 216, 240, 248, 144, 152, 176, 184, 210, 218, 242, 250, 146, 154, 178, 186, 212, 220, 244, 252, 148, 156, 180, 188, 214, 222, 246, 254, 150, 158, 182, 190},
            {129, 137, 161, 169, 193, 201, 225, 233, 131, 139, 163, 171, 195, 203, 227, 235, 133, 141, 165, 173, 197, 205, 229, 237, 135, 143, 167, 175, 199, 207, 231, 239},
            {145, 153, 177, 185, 209, 217, 241, 249, 147, 155, 179, 187, 211, 219, 243, 251, 149, 157, 181, 189, 213, 221, 245, 253, 151, 159, 183, 191, 215, 223, 247, 255},
            {256, 264, 288, 296, 320, 328, 352, 360, 258, 266, 290, 298, 322, 330, 354, 362, 260, 268, 292, 300, 324, 332, 356, 364, 262, 270, 294, 302, 326, 334, 358, 366},
            {272, 280, 304, 312, 336, 344, 368, 376, 274, 282, 306, 314, 338, 346, 370, 378, 276, 284, 308, 316, 340, 348, 372, 380, 278, 286, 310, 318, 342, 350, 374, 382},
            {321, 329, 353, 361, 257, 265, 289, 297, 323, 331, 355, 363, 259, 267, 291, 299, 325, 333, 357, 365, 261, 269, 293, 301, 327, 335, 359, 367, 263, 271, 295, 303},
            {337, 345, 369, 377, 273, 281, 305, 313, 339, 347, 371, 379, 275, 283, 307, 315, 341, 349, 373, 381, 277, 285, 309, 317, 343, 351, 375, 383, 279, 287, 311, 319},
            {448, 456, 480, 488, 384, 392, 416, 424, 450, 458, 482, 490, 386, 394, 418, 426, 452, 460, 484, 492, 388, 396, 420, 428, 454, 462, 486, 494, 390, 398, 422, 430},
            {464, 472, 496, 504, 400, 408, 432, 440, 466, 474, 498, 506, 402, 410, 434, 442, 468, 476, 500, 508, 404, 412, 436, 444, 470, 478, 502, 510, 406, 414, 438, 446},
            {385, 393, 417, 425, 449, 457, 481, 489, 387, 395, 419, 427, 451, 459, 483, 491, 389, 397, 421, 429, 453, 461, 485, 493, 391, 399, 423, 431, 455, 463, 487, 495},
            {401, 409, 433, 441, 465, 473, 497, 505, 403, 411, 435, 443, 467, 475, 499, 507, 405, 413, 437, 445, 469, 477, 501, 509, 407, 415, 439, 447, 471, 479, 503, 511},
        };

        const uint32_t pagesPerRow = ((width >> 1u) != 0u) ? (width >> 1u) : 1u;
        const uint32_t page = (block >> 5u) + (y >> 7u) * pagesPerRow + (x >> 7u);
        const uint32_t blockId = (block & 0x1Fu) + kBlockTable4[(y >> 4u) & 7u][(x >> 5u) & 3u];
        const uint32_t pageOffset = (blockId >> 5u) << 14u;
        const uint32_t localBlock = blockId & 0x1Fu;
        return (page << 14u) + pageOffset + localBlock * 512u + kColumnTable4[y & 0x0Fu][x & 0x1Fu];
    }

    void writeReferencePSMT4Texel(std::vector<uint8_t> &vram, uint32_t tbp, uint32_t tbw, uint32_t x, uint32_t y, uint8_t index)
    {
        const uint32_t nibbleAddr = referenceAddrPSMT4(tbp, tbw, x, y);
        const uint32_t byteOff = nibbleAddr >> 1;
        uint8_t &packed = vram[byteOff];
        if ((nibbleAddr & 1u) != 0u)
        {
            packed = static_cast<uint8_t>((packed & 0x0Fu) | ((index & 0x0Fu) << 4));
        }
        else
        {
            packed = static_cast<uint8_t>((packed & 0xF0u) | (index & 0x0Fu));
        }
    }

    uint32_t referenceAddrPSMT8(uint32_t block, uint32_t width, uint32_t x, uint32_t y)
    {
        static constexpr uint8_t kBlockTable8[4][8] = {
            {0, 1, 4, 5, 16, 17, 20, 21},
            {2, 3, 6, 7, 18, 19, 22, 23},
            {8, 9, 12, 13, 24, 25, 28, 29},
            {10, 11, 14, 15, 26, 27, 30, 31},
        };

        static constexpr uint8_t kColumnTable8[16][16] = {
            {0, 4, 16, 20, 32, 36, 48, 52, 2, 6, 18, 22, 34, 38, 50, 54},
            {8, 12, 24, 28, 40, 44, 56, 60, 10, 14, 26, 30, 42, 46, 58, 62},
            {33, 37, 49, 53, 1, 5, 17, 21, 35, 39, 51, 55, 3, 7, 19, 23},
            {41, 45, 57, 61, 9, 13, 25, 29, 43, 47, 59, 63, 11, 15, 27, 31},
            {96, 100, 112, 116, 64, 68, 80, 84, 98, 102, 114, 118, 66, 70, 82, 86},
            {104, 108, 120, 124, 72, 76, 88, 92, 106, 110, 122, 126, 74, 78, 90, 94},
            {65, 69, 81, 85, 97, 101, 113, 117, 67, 71, 83, 87, 99, 103, 115, 119},
            {73, 77, 89, 93, 105, 109, 121, 125, 75, 79, 91, 95, 107, 111, 123, 127},
            {128, 132, 144, 148, 160, 164, 176, 180, 130, 134, 146, 150, 162, 166, 178, 182},
            {136, 140, 152, 156, 168, 172, 184, 188, 138, 142, 154, 158, 170, 174, 186, 190},
            {161, 165, 177, 181, 129, 133, 145, 149, 163, 167, 179, 183, 131, 135, 147, 151},
            {169, 173, 185, 189, 137, 141, 153, 157, 171, 175, 187, 191, 139, 143, 155, 159},
            {224, 228, 240, 244, 192, 196, 208, 212, 226, 230, 242, 246, 194, 198, 210, 214},
            {232, 236, 248, 252, 200, 204, 216, 220, 234, 238, 250, 254, 202, 206, 218, 222},
            {193, 197, 209, 213, 225, 229, 241, 245, 195, 199, 211, 215, 227, 231, 243, 247},
            {201, 205, 217, 221, 233, 237, 249, 253, 203, 207, 219, 223, 235, 239, 251, 255},
        };

        const uint32_t pagesPerRow = ((width >> 1u) != 0u) ? (width >> 1u) : 1u;
        const uint32_t page = (block >> 5u) + (y >> 6u) * pagesPerRow + (x >> 7u);
        const uint32_t blockId = (block & 0x1Fu) + kBlockTable8[(y >> 4u) & 3u][(x >> 4u) & 7u];
        const uint32_t pageOffset = (blockId >> 5u) << 13u;
        const uint32_t localBlock = blockId & 0x1Fu;
        return (page << 13u) + pageOffset + localBlock * 256u + kColumnTable8[y & 0x0Fu][x & 0x0Fu];
    }

    uint32_t referenceAddrPSMCT32(uint32_t block, uint32_t width, uint32_t x, uint32_t y)
    {
        static constexpr uint8_t kBlockTable32[4][8] = {
            {0, 1, 4, 5, 16, 17, 20, 21},
            {2, 3, 6, 7, 18, 19, 22, 23},
            {8, 9, 12, 13, 24, 25, 28, 29},
            {10, 11, 14, 15, 26, 27, 30, 31},
        };

        static constexpr uint8_t kColumnTable32[8][8] = {
            {0, 1, 4, 5, 8, 9, 12, 13},
            {2, 3, 6, 7, 10, 11, 14, 15},
            {16, 17, 20, 21, 24, 25, 28, 29},
            {18, 19, 22, 23, 26, 27, 30, 31},
            {32, 33, 36, 37, 40, 41, 44, 45},
            {34, 35, 38, 39, 42, 43, 46, 47},
            {48, 49, 52, 53, 56, 57, 60, 61},
            {50, 51, 54, 55, 58, 59, 62, 63},
        };

        const uint32_t pagesPerRow = (width != 0u) ? width : 1u;
        const uint32_t page = (block >> 5u) + (y >> 5u) * pagesPerRow + (x >> 6u);
        const uint32_t blockId = (block & 0x1Fu) + kBlockTable32[(y >> 3u) & 3u][(x >> 3u) & 7u];
        const uint32_t pageOffset = (blockId >> 5u) << 13u;
        const uint32_t localBlock = blockId & 0x1Fu;
        return (page << 13u) + pageOffset + localBlock * 256u +
               static_cast<uint32_t>(kColumnTable32[y & 0x7u][x & 0x7u]) * 4u;
    }

    void writeReferencePSMCT32Pixel(std::vector<uint8_t> &vram,
                                    uint32_t fbp,
                                    uint32_t fbw,
                                    uint32_t x,
                                    uint32_t y,
                                    uint32_t pixel)
    {
        const uint32_t off = referenceAddrPSMCT32(fbp, (fbw != 0u) ? fbw : 1u, x, y);
        std::memcpy(vram.data() + off, &pixel, sizeof(pixel));
    }

    uint32_t readReferencePSMCT32Pixel(const std::vector<uint8_t> &vram,
                                       uint32_t fbp,
                                       uint32_t fbw,
                                       uint32_t x,
                                       uint32_t y)
    {
        const uint32_t off = referenceAddrPSMCT32(fbp, (fbw != 0u) ? fbw : 1u, x, y);
        uint32_t pixel = 0u;
        std::memcpy(&pixel, vram.data() + off, sizeof(pixel));
        return pixel;
    }

    uint32_t frameBaseToBlock(uint32_t fbp)
    {
        return fbp << 5u;
    }

    void writeReferenceFramePSMCT32Pixel(std::vector<uint8_t> &vram,
                                         uint32_t fbp,
                                         uint32_t fbw,
                                         uint32_t x,
                                         uint32_t y,
                                         uint32_t pixel)
    {
        writeReferencePSMCT32Pixel(vram, frameBaseToBlock(fbp), fbw, x, y, pixel);
    }

    uint32_t readReferenceFramePSMCT32Pixel(const std::vector<uint8_t> &vram,
                                            uint32_t fbp,
                                            uint32_t fbw,
                                            uint32_t x,
                                            uint32_t y)
    {
        return readReferencePSMCT32Pixel(vram, frameBaseToBlock(fbp), fbw, x, y);
    }

    uint32_t referenceBilinearColor4(uint32_t c00,
                                     uint32_t c10,
                                     uint32_t c01,
                                     uint32_t c11,
                                     uint8_t weightU,
                                     uint8_t weightV)
    {
        auto shiftRight4 = [](int value)
        {
            return value >= 0 ? value / 16 : -((-value + 15) / 16);
        };
        auto interpolate = [&](int from, int to, int weight)
        {
            return from + shiftRight4((to - from) * weight);
        };
        auto channel = [&](uint8_t shift)
        {
            const int top = interpolate(
                static_cast<uint8_t>(c00 >> shift),
                static_cast<uint8_t>(c10 >> shift),
                weightU);
            const int bottom = interpolate(
                static_cast<uint8_t>(c01 >> shift),
                static_cast<uint8_t>(c11 >> shift),
                weightU);
            return static_cast<uint8_t>(
                interpolate(top, bottom, weightV));
        };

        return static_cast<uint32_t>(channel(0u)) |
               (static_cast<uint32_t>(channel(8u)) << 8u) |
               (static_cast<uint32_t>(channel(16u)) << 16u) |
               (static_cast<uint32_t>(channel(24u)) << 24u);
    }

    struct PackedSpriteTestConfiguration
    {
        uint64_t frame =
            (0ull << 0) |
            (2ull << 16) |
            (static_cast<uint64_t>(GS_PSM_CT32) << 24);
        uint64_t zbuf =
            (3ull << 0) |
            (1ull << 32);
        uint64_t scissor =
            (0ull << 0) |
            (127ull << 16) |
            (0ull << 32) |
            (63ull << 48);
        uint64_t tex0 =
            (96ull << 0) |
            (2ull << 14) |
            (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
            (7ull << 26) |
            (6ull << 30) |
            (1ull << 34) |
            (1ull << 35);
        uint64_t tex1 =
            (1ull << 5) |
            (1ull << 6);
        uint64_t clamp = 0ull;
        uint64_t test =
            (1ull << 16) |
            (1ull << 17);
        uint64_t fba = 0ull;
        uint64_t prim =
            static_cast<uint64_t>(GS_PRIM_SPRITE) |
            (1ull << 4) |
            (1ull << 8);
        uint64_t rgbaq = 0x3F80000010203040ull;
        uint16_t x0 = 60u * 16u;
        uint16_t y0 = 28u * 16u;
        uint16_t x1 = 76u * 16u;
        uint16_t y1 = 40u * 16u;
        uint16_t u0 = 60u * 16u;
        uint16_t v0 = 28u * 16u;
        uint16_t u1 = 76u * 16u;
        uint16_t v1 = 40u * 16u;
    };

    struct PackedSpriteRenderResult
    {
        std::vector<uint8_t> vram;
        uint64_t packedDispatches = 0u;
        GSRasterizerDetail::PackedSpriteKernelImplementation
            implementation =
                GSRasterizerDetail::
                    PackedSpriteKernelImplementation::Scalar;
        uint64_t vectorGroups = 0u;
    };

    PackedSpriteRenderResult renderPackedSpriteTest(
        const PackedSpriteTestConfiguration &configuration,
        GSRasterizerDetail::PackedSpriteKernelOverride overrideMode)
    {
        std::vector<uint8_t> vram(
            PS2_GS_VRAM_SIZE, 0x5Au);
        GS gs;
        gs.init(
            vram.data(),
            static_cast<uint32_t>(vram.size()),
            nullptr);

        const uint32_t textureBase =
            static_cast<uint32_t>(
                configuration.tex0 & 0x3FFFu);
        const uint32_t textureWidth =
            static_cast<uint32_t>(
                (configuration.tex0 >> 14u) & 0x3Fu);
        const uint32_t textureSizeU =
            1u << ((configuration.tex0 >> 26u) & 0xFu);
        const uint32_t textureSizeV =
            1u << ((configuration.tex0 >> 30u) & 0xFu);
        for (uint32_t y = 0u; y < textureSizeV; ++y)
        {
            for (uint32_t x = 0u; x < textureSizeU; ++x)
            {
                const uint32_t color =
                    ((0x40u + x * 17u + y * 3u) & 0xFFu) |
                    (((0x20u + x * 5u + y * 29u) & 0xFFu)
                     << 8u) |
                    (((0xE0u - x * 11u - y * 7u) & 0xFFu)
                     << 16u) |
                    (((0x80u + x * 9u + y * 13u) & 0xFFu)
                     << 24u);
                gs.WriteVram(
                    GS_PSM_CT32,
                    textureBase,
                    textureWidth,
                    x,
                    y,
                    color);
            }
        }

        GSRasterizerDetail::setPackedSpriteKernelOverride(
            overrideMode);
        GSRasterizerDetail::resetPackedSpriteKernelDispatchCount();
        GSRasterizerDetail::resetPackedSpriteVectorGroupCount();

        gs.writeRegister(GS_REG_FRAME_1, configuration.frame);
        gs.writeRegister(GS_REG_ZBUF_1, configuration.zbuf);
        gs.writeRegister(
            GS_REG_SCISSOR_1, configuration.scissor);
        gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
        gs.writeRegister(GS_REG_TEST_1, configuration.test);
        gs.writeRegister(GS_REG_FBA_1, configuration.fba);
        gs.writeRegister(GS_REG_TEX0_1, configuration.tex0);
        gs.writeRegister(GS_REG_TEX1_1, configuration.tex1);
        gs.writeRegister(GS_REG_CLAMP_1, configuration.clamp);
        gs.writeRegister(GS_REG_PRIM, configuration.prim);
        gs.writeRegister(GS_REG_RGBAQ, configuration.rgbaq);
        gs.writeRegister(
            GS_REG_UV,
            static_cast<uint64_t>(configuration.u0) |
                (static_cast<uint64_t>(configuration.v0) << 16u));
        gs.writeRegister(
            GS_REG_XYZ2,
            static_cast<uint64_t>(configuration.x0) |
                (static_cast<uint64_t>(configuration.y0) << 16u));
        gs.writeRegister(
            GS_REG_UV,
            static_cast<uint64_t>(configuration.u1) |
                (static_cast<uint64_t>(configuration.v1) << 16u));
        gs.writeRegister(
            GS_REG_XYZ2,
            static_cast<uint64_t>(configuration.x1) |
                (static_cast<uint64_t>(configuration.y1) << 16u));

        return {
            std::move(vram),
            GSRasterizerDetail::packedSpriteKernelDispatchCount(),
            GSRasterizerDetail::
                packedSpriteLastKernelImplementation(),
            GSRasterizerDetail::packedSpriteVectorGroupCount(),
        };
    }

    void expectGuestHeapReusable(TestCase &t, PS2Runtime &runtime, const char *message)
    {
        const uint32_t expectedBase = runtime.guestHeapBase();
        const uint32_t probe = runtime.guestMalloc(16u, 16u);
        t.Equals(probe, expectedBase, message);
        runtime.guestFree(probe);
    }
}

void register_ps2_gs_tests()
{
    MiniTest::Case("PS2GS", [](TestCase &tc)
    {
        tc.Run("GS draw commands preserve raw vertices and stable state signatures", [](TestCase &t)
        {
            GSContext context{};
            context.frame.fbp = 7u;
            context.frame.fbw = 2u;
            context.frame.psm = GS_PSM_CT32;
            context.scissor = {0u, 127u, 0u, 63u};
            context.xyoffset = {16u, 32u};
            context.zbuf = {20u, GS_PSM_Z32, true};
            context.test = 0x30000u;

            GSPrimReg primitive{};
            primitive.type = GS_PRIM_SPRITE;

            std::array<GSVertex, 2> vertices{};
            vertices[0].x = 999.0f;
            vertices[0].y = 999.0f;
            vertices[0].x12_4 = 32u;
            vertices[0].y12_4 = 48u;
            vertices[0].zInteger = 0xFEDCBA98u;
            vertices[1].x12_4 = 160u;
            vertices[1].y12_4 = 176u;

            const GsDrawGlobalState global{};
            const GsDrawCommand first = buildGsDrawCommand(
                41u,
                primitive,
                context,
                std::span<const GSVertex>(vertices),
                global);

            t.Equals(first.sequence(), 41ull,
                     "draw sequence should be retained verbatim");
            t.Equals(first.vertexCount(), static_cast<uint8_t>(2u),
                     "sprite command should own two submitted vertices");
            t.Equals(first.vertices()[0].zInteger, 0xFEDCBA98u,
                     "integer Z should survive without a floating-point round trip");
            t.Equals(first.fixedX()[0], 16,
                     "fixed X should come from raw 12.4 payload minus XYOFFSET");
            t.Equals(first.fixedY()[0], 16,
                     "fixed Y should come from raw 12.4 payload minus XYOFFSET");
            t.Equals(first.bounds().x0, 1,
                     "decoded bounds should use raw fixed-point coordinates");
            t.Equals(first.bounds().y0, 1,
                     "decoded bounds should include the raw XYOFFSET subtraction");

            vertices[0].x12_4 = 48u;
            t.Equals(first.vertices()[0].x12_4,
                     static_cast<uint16_t>(32u),
                     "command vertices should not observe later assembly mutations");
            const GsDrawCommand sameState = buildGsDrawCommand(
                42u,
                primitive,
                context,
                std::span<const GSVertex>(vertices),
                global);
            t.Equals(first.stateSignature(), sameState.stateSignature(),
                     "state signature should group compatible draws independently of geometry and sequence");

            GSContext changedContext = context;
            changedContext.frame.fbp = 8u;
            const GsDrawCommand changedState = buildGsDrawCommand(
                43u,
                primitive,
                changedContext,
                std::span<const GSVertex>(vertices),
                global);
            t.IsTrue(first.stateSignature() != changedState.stateSignature(),
                     "state signature should change when a resource-defining register changes");
        });

        tc.Run("GS draw resource masks conservatively wrap the 4 MiB page ring", [](TestCase &t)
        {
            GSContext context{};
            context.frame.fbp = 511u;
            context.frame.fbw = 1u;
            context.frame.psm = GS_PSM_CT32;
            context.scissor = {0u, 63u, 0u, 63u};
            context.zbuf = {100u, GS_PSM_Z32, true};

            GSPrimReg primitive{};
            primitive.type = GS_PRIM_SPRITE;
            std::array<GSVertex, 2> vertices{};
            vertices[1].x12_4 = 64u * 16u;
            vertices[1].y12_4 = 64u * 16u;

            const GsDrawCommand command = buildGsDrawCommand(
                1u,
                primitive,
                context,
                std::span<const GSVertex>(vertices),
                GsDrawGlobalState{});
            const GsDrawResources resources = command.resources();
            const GsVramPageMask &writes =
                resources.framebufferWritePages;
            t.Equals(writes.count(), static_cast<size_t>(2u),
                     "64x64 CT32 bounds should conservatively span two 8 KiB pages");
            t.IsTrue(writes.test(511u),
                     "resource mask should retain the final physical GS page");
            t.IsTrue(writes.test(0u),
                     "resource mask should wrap past the end of 4 MiB VRAM");
            t.IsFalse(resources.readPages.any(),
                      "opaque CT32 draw with disabled depth should not claim a read dependency");
        });

        tc.Run("GS surface rectangle masks contain every canonical PSM address", [](TestCase &t)
        {
            struct SurfaceSpec
            {
                uint8_t psm;
                uint32_t pageWidth;
                uint32_t pageHeight;
            };
            constexpr std::array<SurfaceSpec, 13> specs{{
                {GS_PSM_CT32, 64u, 32u},
                {GS_PSM_CT24, 64u, 32u},
                {GS_PSM_CT16, 64u, 64u},
                {GS_PSM_CT16S, 64u, 64u},
                {GS_PSM_T8, 128u, 64u},
                {GS_PSM_T4, 128u, 128u},
                {GS_PSM_T8H, 64u, 32u},
                {GS_PSM_T4HL, 64u, 32u},
                {GS_PSM_T4HH, 64u, 32u},
                {GS_PSM_Z32, 64u, 32u},
                {GS_PSM_Z24, 64u, 32u},
                {GS_PSM_Z16, 64u, 64u},
                {GS_PSM_Z16S, 64u, 64u},
            }};
            constexpr std::array<uint32_t, 4> widths{{0u, 1u, 2u, 63u}};

            uint64_t resolvedPixels = 0u;
            for (const SurfaceSpec &spec : specs)
            {
                const uint32_t x = spec.pageWidth - 3u;
                const uint32_t y = spec.pageHeight - 2u;
                const uint32_t rectangleWidth = spec.pageWidth + 7u;
                const uint32_t rectangleHeight = spec.pageHeight + 5u;
                for (uint32_t basePhase = 0u;
                     basePhase < 32u; ++basePhase)
                {
                    const uint32_t bp =
                        (basePhase & 1u) != 0u
                            ? 37u * 32u + basePhase
                            : 0x3FE0u + basePhase;
                    for (uint32_t bw : widths)
                    {
                        const GsVramPageMask pages =
                            gsVramPagesForSurfaceRect(
                                bp, bw, spec.psm,
                                x, y,
                                rectangleWidth, rectangleHeight);
                        t.IsFalse(pages.all(),
                                  "a bounded valid rectangle should retain a partial mask");
                        for (uint32_t py = y;
                             py < y + rectangleHeight; ++py)
                        {
                            for (uint32_t px = x;
                                 px < x + rectangleWidth; ++px)
                            {
                                GSMem::PixelAddress address{};
                                if (!GSMem::ResolvePixelAddress(
                                        static_cast<GSMem::PixelStorageMode>(
                                            spec.psm),
                                        bp, bw, px, py, address))
                                {
                                    t.Fail("canonical resolver rejected a supported surface PSM");
                                    return;
                                }
                                const size_t physicalPage =
                                    (static_cast<size_t>(address.word_index) *
                                     sizeof(uint32_t)) /
                                    GS_VRAM_PAGE_SIZE;
                                if (!pages.test(physicalPage))
                                {
                                    t.Fail("surface mask omitted a canonical physical page");
                                    return;
                                }
                                ++resolvedPixels;
                            }
                        }
                    }
                }
            }
            t.IsTrue(resolvedPixels > 5'000'000ull,
                     "the containment corpus should cover millions of packed addresses");

            const GsVramPageMask wrap =
                gsVramPagesForSurfaceRect(
                    511u * 32u, 1u, GS_PSM_CT32,
                    0u, 0u, 64u, 64u);
            t.Equals(wrap.count(), static_cast<size_t>(2u),
                     "an aligned two-page CT32 rectangle should stay exact");
            t.IsTrue(wrap.test(511u) && wrap.test(0u),
                     "the exact rectangle should wrap across the 4 MiB ring");

            const GsVramPageMask phased =
                gsVramPagesForSurfaceRect(
                    31u, 1u, GS_PSM_T4HH,
                    0u, 0u, 1u, 1u);
            t.IsTrue(phased.test(0u) && phased.test(1u),
                     "a non-page-aligned high-plane surface should include a possible carry page");

            t.IsTrue(gsVramPagesForSurfaceRect(
                         0u, 1u, 0x3Fu,
                         0u, 0u, 1u, 1u).all(),
                     "an unknown layout should fail closed to all pages");
            t.IsFalse(gsVramPagesForSurfaceRect(
                          0u, 1u, 0x3Fu,
                          0u, 0u, 0u, 1u).any(),
                      "an empty rectangle should remain empty even with unknown state");
            t.IsTrue(gsVramPagesForSurfaceRect(
                         0u, 1u, GS_PSM_CT32,
                         UINT32_MAX, 0u, 2u, 1u).all(),
                     "coordinate overflow should fail closed to all pages");
        });

        tc.Run("GS draw resource masks fail closed for unknown memory layouts", [](TestCase &t)
        {
            GSContext context{};
            context.frame.fbw = 1u;
            context.frame.psm = 0x3Fu;
            context.scissor = {0u, 15u, 0u, 15u};

            GSPrimReg primitive{};
            primitive.type = GS_PRIM_SPRITE;
            std::array<GSVertex, 2> vertices{};
            vertices[1].x12_4 = 16u * 16u;
            vertices[1].y12_4 = 16u * 16u;

            const GsDrawCommand command = buildGsDrawCommand(
                1u,
                primitive,
                context,
                std::span<const GSVertex>(vertices),
                GsDrawGlobalState{});
            t.IsTrue(command.resources().unknownMemoryLayout,
                     "unknown PSM should be explicit in the resource description");
            t.IsTrue(command.resources().framebufferWritePages.all(),
                     "unknown framebuffer layout should conservatively touch all 512 pages");
        });

        tc.Run("initial CT32 sprite classifier distinguishes depth access from ZTST ALWAYS", [](TestCase &t)
        {
            GSContext context{};
            context.frame.fbw = 1u;
            context.frame.psm = GS_PSM_CT32;
            context.scissor = {0u, 15u, 0u, 15u};
            context.zbuf = {32u, GS_PSM_Z32, true};
            context.test = 0x30000u; // ZTE=1, ZTST=ALWAYS.

            GSPrimReg primitive{};
            primitive.type = GS_PRIM_SPRITE;
            std::array<GSVertex, 2> vertices{};
            vertices[1].x12_4 = 16u * 16u;
            vertices[1].y12_4 = 16u * 16u;

            const auto makeCommand = [&](const GSContext &drawContext)
            {
                return buildGsDrawCommand(
                    1u,
                    primitive,
                    drawContext,
                    std::span<const GSVertex>(vertices),
                    GsDrawGlobalState{});
            };

            const GsDrawCommand alwaysMasked = makeCommand(context);
            t.IsFalse(alwaysMasked.resources().depthReadPages.any(),
                      "ZTST ALWAYS should not read the depth surface");
            t.IsFalse(alwaysMasked.resources().depthWritePages.any(),
                      "ZMSK should suppress the depth write dependency");
            t.Equals(classifyGsInitialCt32Sprite(alwaysMasked).reason,
                     GsFallbackReason::Supported,
                     "depth-always with masked writes belongs in the no-depth initial subset");

            context.zbuf.zmask = false;
            const GsDrawCommand depthWrite = makeCommand(context);
            t.IsTrue(depthWrite.resources().depthWritePages.any(),
                     "unmasked depth should create a write dependency");
            t.Equals(classifyGsInitialCt32Sprite(depthWrite).reason,
                     GsFallbackReason::DepthWrite,
                     "initial subset should name depth writes precisely");

            context.zbuf.zmask = true;
            context.test = 0x50000u; // ZTE=1, ZTST=GEQUAL.
            const GsDrawCommand depthRead = makeCommand(context);
            t.IsTrue(depthRead.resources().depthReadPages.any(),
                     "GEQUAL should create a depth read dependency even with writes masked");
            t.Equals(classifyGsInitialCt32Sprite(depthRead).reason,
                     GsFallbackReason::DepthRead,
                     "initial subset should name depth comparisons precisely");
        });

        tc.Run("GS backend router classifies before mutation and records synchronized fallback", [](TestCase &t)
        {
            class RecordingBackend final : public IGsRasterBackend
            {
            public:
                GsBackendDecision decision{
                    true, GsFallbackReason::Supported};
                std::vector<uint64_t> sequences;
                std::vector<GsFlushReason> flushReasons;
                std::vector<GsVramPageMask> cpuAccessPages;
                std::vector<GsFlushReason> cpuAccessReasons;
                std::vector<GsVramPageMask> cpuWritePages;
                size_t pending = 0u;

                [[nodiscard]] GsBackendDecision classify(
                    const GsDrawCommand &) const override
                {
                    return decision;
                }

                void submit(
                    std::span<const GsDrawCommand> commands) override
                {
                    for (const GsDrawCommand &command : commands)
                        sequences.push_back(command.sequence());
                    pending += commands.size();
                }

                void flush(GsFlushReason reason) override
                {
                    flushReasons.push_back(reason);
                    pending = 0u;
                }

                [[nodiscard]] size_t pendingCommandCount()
                    const noexcept override
                {
                    return pending;
                }

                void prepareCpuVramAccess(
                    const GsVramPageMask &pages,
                    GsFlushReason reason) override
                {
                    cpuAccessPages.push_back(pages);
                    cpuAccessReasons.push_back(reason);
                }

                void noteCpuVramWrite(
                    const GsVramPageMask &pages) override
                {
                    cpuWritePages.push_back(pages);
                }
            } software, accelerated;

            GSContext context{};
            context.frame.fbw = 1u;
            context.frame.psm = GS_PSM_CT32;
            context.scissor = {0u, 15u, 0u, 15u};
            GSPrimReg primitive{};
            primitive.type = GS_PRIM_SPRITE;
            std::array<GSVertex, 2> vertices{};
            vertices[1].x12_4 = 16u * 16u;
            vertices[1].y12_4 = 16u * 16u;
            const GsDrawCommand command = buildGsDrawCommand(
                7u,
                primitive,
                context,
                std::span<const GSVertex>(vertices),
                GsDrawGlobalState{});

            GsBackendRouter router(software);
            t.IsFalse(router.setMode(GsRendererMode::Hybrid),
                      "accelerated modes should not be selectable without a backend");
            t.Equals(router.mode(), GsRendererMode::Software,
                     "failed selection should retain the synchronized software mode");

            router.setCountersEnabled(true);
            const GsSubmissionResult direct = router.submit(command);
            t.IsTrue(direct.submitted && direct.usedSoftware,
                     "software mode should submit through the permanent adapter");
            t.Equals(software.sequences.size(), static_cast<size_t>(1u),
                     "software adapter should receive the command once");
            t.Equals(router.counters().queueHighWatermark, 1ull,
                     "router should expose backend queue high-water state");
            t.Equals(router.counters().drawPixels, 256ull,
                     "router counters should measure exact command bounds");
            t.Equals(router.counters().softwarePixels, 256ull,
                     "software routing should account its covered pixels");

            router.setAcceleratedBackend(&accelerated);
            t.IsTrue(router.setMode(GsRendererMode::Hybrid),
                     "hybrid mode should become selectable after backend attachment");
            accelerated.cpuAccessPages.clear();
            accelerated.cpuAccessReasons.clear();
            accelerated.cpuWritePages.clear();
            accelerated.decision = {
                false, GsFallbackReason::Textured};
            const GsSubmissionResult fallback = router.submit(command);
            t.IsTrue(fallback.submitted && fallback.usedSoftware,
                     "hybrid rejection should fall back before accelerated submission");
            t.Equals(accelerated.sequences.size(), static_cast<size_t>(0u),
                     "unsupported command must not partially reach the accelerated backend");
            t.Equals(router.counters().fallbackCommands, 1ull,
                     "fallback should be represented in structured counters");
            t.Equals(router.counters().fallbackPixels, 256ull,
                     "fallback counters should retain covered-pixel cost");
            t.Equals(
                router.counters().decisions[
                    static_cast<size_t>(GsFallbackReason::Textured)],
                1ull,
                "the classifier's single reason should be counted");
            const GsDrawResources commandResources = command.resources();
            GsVramPageMask commandAccess = commandResources.readPages;
            commandAccess.unionWith(commandResources.writePages);
            t.Equals(accelerated.cpuAccessPages.size(),
                     static_cast<size_t>(1u),
                     "software fallback should prepare one scoped CPU access");
            t.IsTrue(accelerated.cpuAccessPages[0] == commandAccess,
                     "fallback should synchronize exactly its conservative resource pages");
            t.Equals(accelerated.cpuAccessReasons[0],
                     GsFlushReason::BackendSwitch,
                     "draw fallback should retain the backend-switch transition reason");
            t.Equals(accelerated.cpuWritePages.size(),
                     static_cast<size_t>(1u),
                     "software fallback should publish one CPU writer mask");
            t.IsTrue(accelerated.cpuWritePages[0] ==
                         commandResources.writePages,
                     "software publication should exclude read-only resource pages");

            accelerated.decision = {
                true, GsFallbackReason::Supported};
            const GsSubmissionResult acceleratedResult =
                router.submit(command);
            t.IsTrue(
                acceleratedResult.submitted &&
                    acceleratedResult.usedAccelerated,
                "supported hybrid command should route to the accelerated backend");
            t.Equals(accelerated.sequences.size(), static_cast<size_t>(1u),
                     "accelerated backend should receive a supported command once");
            t.IsTrue(!software.flushReasons.empty() &&
                         software.flushReasons.back() ==
                             GsFlushReason::BackendSwitch,
                     "switching away from queued software work should synchronize it first");

            t.IsTrue(router.setMode(GsRendererMode::GpuStrict),
                     "strict mode should be selectable with an accelerated backend");
            accelerated.decision = {
                false, GsFallbackReason::AlphaBlend};
            const size_t softwareBefore = software.sequences.size();
            const size_t acceleratedBefore = accelerated.sequences.size();
            const GsSubmissionResult strict = router.submit(command);
            t.IsFalse(strict.submitted,
                      "strict mode should fail the first unsupported command");
            t.Equals(software.sequences.size(), softwareBefore,
                     "strict rejection must not silently fall back to software");
            t.Equals(accelerated.sequences.size(), acceleratedBefore,
                     "strict rejection must occur before accelerated mutation");
            t.Equals(router.counters().strictFailures, 1ull,
                     "strict failures should be explicit in counters");

            t.IsTrue(router.setMode(GsRendererMode::Verify),
                     "verify mode should share the synchronized accelerated slot");
            accelerated.decision = {
                true, GsFallbackReason::Supported};
            const GsSubmissionResult verified = router.submit(command);
            t.IsTrue(verified.submitted && verified.usedAccelerated,
                     "verification backend should receive supported work");
            t.Equals(router.counters().verifiedCommands, 1ull,
                     "verify submissions should be counted separately");
            t.Equals(router.counters().drawPixels, 1280ull,
                     "all five routing decisions should contribute pixel work");
            t.Equals(router.counters().softwarePixels, 512ull,
                     "direct software and fallback pixels should be separated");
            t.Equals(router.counters().acceleratedPixels, 512ull,
                     "hybrid and verify submissions should count accelerated pixels");
            t.Equals(router.counters().verifiedPixels, 256ull,
                     "verify pixels should be a named accelerated subset");
            t.Equals(router.counters().strictFailurePixels, 256ull,
                     "strict rejection should retain its avoided pixel work");

            accelerated.cpuAccessPages.clear();
            accelerated.cpuAccessReasons.clear();
            accelerated.cpuWritePages.clear();
            GsVramPageMask cpuReads;
            cpuReads.set(7u);
            GsVramPageMask cpuWrites;
            cpuWrites.set(511u);
            router.beginCpuVramAccess(
                cpuReads, cpuWrites, GsFlushReason::Transfer);
            router.endCpuVramAccess(cpuWrites);
            GsVramPageMask combinedCpuAccess = cpuReads;
            combinedCpuAccess.unionWith(cpuWrites);
            t.Equals(accelerated.cpuAccessPages.size(),
                     static_cast<size_t>(1u),
                     "an external CPU transaction should prepare one access");
            t.IsTrue(accelerated.cpuAccessPages[0] == combinedCpuAccess,
                     "external CPU access should synchronize the read/write union only");
            t.Equals(accelerated.cpuAccessReasons[0],
                     GsFlushReason::Transfer,
                     "external CPU access should preserve its named boundary");
            t.Equals(accelerated.cpuWritePages.size(),
                     static_cast<size_t>(1u),
                     "external CPU transaction should publish its writer mask once");
            t.IsTrue(accelerated.cpuWritePages[0] == cpuWrites,
                     "external CPU publication should not dirty read-only pages");

            accelerated.cpuAccessPages.clear();
            accelerated.cpuAccessReasons.clear();
            accelerated.flushReasons.clear();
            const GsSubmissionResult frameDraw = router.submit(command);
            t.IsTrue(frameDraw.submitted && frameDraw.usedAccelerated,
                     "the frame checkpoint fixture should begin with pending GPU work");
            t.Equals(router.counters().queueDepth, 1ull,
                     "the frame checkpoint should observe pending accelerated work");
            router.flush(GsFlushReason::PresentationLatch);
            t.Equals(accelerated.cpuAccessPages.size(),
                     static_cast<size_t>(1u),
                     "a presentation latch should request one CPU publication");
            t.IsTrue(accelerated.cpuAccessPages[0].all(),
                     "a conservative frame checkpoint should publish all 512 pages");
            t.Equals(accelerated.cpuAccessReasons[0],
                     GsFlushReason::PresentationLatch,
                     "frame publication should retain its named boundary");
            t.Equals(accelerated.flushReasons.size(),
                     static_cast<size_t>(1u),
                     "the active accelerated backend should drain once at the latch");
            t.Equals(accelerated.flushReasons[0],
                     GsFlushReason::PresentationLatch,
                     "accelerated drainage should retain the frame reason");
            t.Equals(router.counters().queueDepth, 0ull,
                     "the frame checkpoint should leave no queued command");
            t.Equals(
                router.counters().flushReasons[static_cast<size_t>(
                    GsFlushReason::PresentationLatch)],
                1ull,
                "the router should count the conservative frame checkpoint");
        });

        tc.Run("GS frontend routes draws and visibility boundaries through the software backend", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(
                vram.data(),
                static_cast<uint32_t>(vram.size()),
                nullptr);

            t.Equals(gs.rendererMode(), GsRendererMode::Software,
                     "software should remain the permanent default mode");
            GsVulkanServiceConfig unavailableConfig{};
            unavailableConfig.probe.loaderPath =
                "/ps2recomp-test-loader-does-not-exist/libvulkan.so";
            t.IsTrue(gs.configureVulkanRenderer(unavailableConfig),
                     "the frontend should accept an explicit unavailable backend fixture");
            t.IsFalse(gs.setRendererMode(GsRendererMode::Hybrid),
                      "an unavailable accelerated backend should fail selection cleanly");

            gs.setBackendCountersEnabled(true);
            gs.writeRegister(GS_REG_FRAME_1, 1ull << 16u);
            gs.writeRegister(GS_REG_ZBUF_1, 1ull << 32u);
            gs.writeRegister(GS_REG_SCISSOR_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(
                GS_REG_PRIM,
                static_cast<uint64_t>(GS_PRIM_POINT));
            gs.writeRegister(GS_REG_RGBAQ, 0x80112233ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);

            const GsBackendCounters drawCounters =
                gs.backendCounters();
            t.Equals(drawCounters.commands, 1ull,
                     "frontend should submit each assembled draw to the router");
            t.Equals(drawCounters.softwareCommands, 1ull,
                     "default routing should reach the software adapter once");
            t.IsTrue(
                drawCounters.softwareRasterHostNanoseconds > 0u,
                "enabled counters should measure the software raster kernel");

            gs.writeRegister(GS_REG_FINISH, 0ull);
            const GsBackendCounters finishCounters =
                gs.backendCounters();
            t.Equals(
                finishCounters.flushReasons[
                    static_cast<size_t>(GsFlushReason::Finish)],
                1ull,
                "FINISH visibility should carry a structured flush reason");

            (void)gs.getDebugSnapshot();
            const GsBackendCounters debugCounters =
                gs.backendCounters();
            t.Equals(
                debugCounters.flushReasons[
                    static_cast<size_t>(
                        GsFlushReason::DebuggerObservation)],
                1ull,
                "debugger state observation should be an explicit synchronization boundary");

            gs.resetBackendCounters();
            t.Equals(
                gs.backendCounters().softwareRasterHostNanoseconds,
                0ull,
                "resetting backend counters should clear software raster time");
        });

        tc.Run("GS draw command limit pauses within a GIF packet", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(
                vram.data(),
                static_cast<uint32_t>(vram.size()),
                nullptr);

            auto configurePointDraws = [](GS &target)
            {
                target.writeRegister(GS_REG_FRAME_1, 1ull << 16u);
                target.writeRegister(GS_REG_ZBUF_1, 1ull << 32u);
                target.writeRegister(GS_REG_SCISSOR_1, 1ull << 16u);
                target.writeRegister(GS_REG_TEST_1, 0x30000ull);
                target.writeRegister(
                    GS_REG_PRIM,
                    static_cast<uint64_t>(GS_PRIM_POINT));
            };
            configurePointDraws(gs);

            constexpr uint32_t kFirstColor = 0x80112233u;
            constexpr uint32_t kSecondColor = 0x80445566u;
            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(4u, GIF_FMT_PACKED, 1u));
            appendU64(packet, 0x0Eull);
            appendGifAd(packet, kFirstColor, GS_REG_RGBAQ);
            appendGifAd(packet, 0ull, GS_REG_XYZ2);
            appendGifAd(packet, kSecondColor, GS_REG_RGBAQ);
            appendGifAd(packet, 16ull, GS_REG_XYZ2);

            gs.setDrawCommandLimit(1u);
            gs.processGIFPacket(
                packet.data(),
                static_cast<uint32_t>(packet.size()));

            t.IsTrue(gs.drawCommandLimitReached(),
                     "the configured draw boundary should be observable");
            t.Equals(gs.submittedDrawCommandCount(), 1ull,
                     "only the first complete command should be submitted");
            t.Equals(
                gs.ReadVram(GS_PSM_CT32, 0u, 1u, 0u, 0u),
                kFirstColor,
                "the boundary draw should complete before packet decoding pauses");
            t.Equals(
                gs.ReadVram(GS_PSM_CT32, 0u, 1u, 1u, 0u),
                0u,
                "registers after the boundary should not partially execute");

            std::vector<uint8_t> resumePacket;
            appendU64(
                resumePacket,
                makeGifTag(2u, GIF_FMT_PACKED, 1u));
            appendU64(resumePacket, 0x0Eull);
            appendGifAd(resumePacket, kSecondColor, GS_REG_RGBAQ);
            appendGifAd(resumePacket, 16ull, GS_REG_XYZ2);

            gs.clearDrawCommandLimit();
            gs.processGIFPacket(
                resumePacket.data(),
                static_cast<uint32_t>(resumePacket.size()));
            t.IsFalse(gs.drawCommandLimitReached(),
                      "clearing the limit should resume packet decoding");
            t.Equals(gs.submittedDrawCommandCount(), 2ull,
                     "resumed decoding should submit the next command");
            t.Equals(
                gs.ReadVram(GS_PSM_CT32, 0u, 1u, 1u, 0u),
                kSecondColor,
                "resumed decoding should render subsequent commands");

            std::vector<uint8_t> nativeVram(PS2_GS_VRAM_SIZE, 0u);
            GS nativeGs;
            nativeGs.init(
                nativeVram.data(),
                static_cast<uint32_t>(nativeVram.size()),
                nullptr);
            configurePointDraws(nativeGs);
            nativeGs.setDrawCommandLimit(1u);
            t.IsTrue(
                nativeGs.processNativePackedGIFPacket(
                    packet.data(),
                    static_cast<uint32_t>(packet.size())),
                "the native packed path should claim an intentional bounded stop");
            t.IsTrue(nativeGs.drawCommandLimitReached(),
                     "the native packed path should observe the same boundary");
            t.Equals(nativeGs.submittedDrawCommandCount(), 1ull,
                     "native packet decoding should stop after the boundary command");
            t.Equals(
                nativeGs.ReadVram(GS_PSM_CT32, 0u, 1u, 1u, 0u),
                0u,
                "native packet decoding should not execute the packet suffix");
        });

        tc.Run("GS replay state preserves partial primitive assembly", [](TestCase &t)
        {
            std::vector<uint8_t> originalVram(PS2_GS_VRAM_SIZE, 0u);
            GS original;
            original.init(
                originalVram.data(),
                static_cast<uint32_t>(originalVram.size()),
                nullptr);
            original.writeRegister(GS_REG_FRAME_1, 1ull << 16u);
            original.writeRegister(GS_REG_ZBUF_1, 1ull << 32u);
            original.writeRegister(
                GS_REG_SCISSOR_1,
                (15ull << 16u) | (15ull << 48u));
            original.writeRegister(GS_REG_TEST_1, 0x30000ull);
            original.writeRegister(
                GS_REG_PRIM,
                static_cast<uint64_t>(GS_PRIM_TRISTRIP));
            original.writeRegister(GS_REG_RGBAQ, 0x80443322ull);
            original.writeRegister(
                GS_REG_XYZ2,
                16ull | (16ull << 16u));
            original.writeRegister(
                GS_REG_XYZ2,
                192ull | (16ull << 16u));

            const GsReplayState captured =
                original.captureReplayState();
            t.Equals(captured.vertexCount, 2,
                     "two strip vertices should remain pending at the boundary");

            std::vector<uint8_t> encoded;
            std::string stateError;
            t.IsTrue(
                encodeGsReplayState(captured, encoded, &stateError),
                "the synchronized frontend state should encode");
            GsReplayState decoded{};
            t.IsTrue(
                decodeGsReplayState(encoded, decoded, &stateError),
                "the encoded frontend state should decode");
            std::vector<uint8_t> canonical;
            t.IsTrue(
                encodeGsReplayState(decoded, canonical, &stateError),
                "decoded frontend state should re-encode");
            t.IsTrue(encoded == canonical,
                     "GS replay state encoding should be canonical");
            if (!encoded.empty())
            {
                std::vector<uint8_t> truncated = encoded;
                truncated.pop_back();
                GsReplayState rejected{};
                t.IsFalse(
                    decodeGsReplayState(
                        truncated, rejected, &stateError),
                    "truncated GS replay state should fail closed");
            }

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(1u, GIF_FMT_PACKED, 1u));
            appendU64(packet, 0x0Eull);
            appendGifAd(
                packet,
                16ull | (192ull << 16u),
                GS_REG_XYZ2);

            const std::vector<uint8_t> initialVram = originalVram;
            original.processGIFPacket(
                packet.data(),
                static_cast<uint32_t>(packet.size()));
            t.Equals(original.submittedDrawCommandCount(), 1ull,
                     "the third strip vertex should complete one command");
            t.IsFalse(originalVram == initialVram,
                      "the completed strip should mutate the framebuffer");

            std::vector<uint8_t> replayVram = initialVram;
            GS replay;
            replay.init(
                replayVram.data(),
                static_cast<uint32_t>(replayVram.size()),
                nullptr);
            t.IsTrue(replay.restoreReplayState(decoded),
                     "the decoded frontend state should restore");
            replay.processGIFPacket(
                packet.data(),
                static_cast<uint32_t>(packet.size()));
            t.Equals(replay.submittedDrawCommandCount(), 1ull,
                     "restored strip assembly should complete one command");
            t.IsTrue(replayVram == originalVram,
                     "restored partial assembly should reproduce all VRAM bytes");

            GsReplayState invalidFeedback = captured;
            invalidFeedback.rasterizer.feedbackSnapshotValid = true;
            invalidFeedback.rasterizer.feedbackVram.assign(1u, 0u);
            t.IsFalse(
                encodeGsReplayState(
                    invalidFeedback, canonical, &stateError),
                "partial feedback snapshots should fail closed");
            t.IsTrue(canonical.empty(),
                     "a failed state encoding should not leave stale bytes");
            invalidFeedback.rasterizer.feedbackSnapshotValid = false;
            t.IsFalse(
                encodeGsReplayState(
                    invalidFeedback, canonical, &stateError),
                "inactive feedback snapshots should not carry stale VRAM");

            GsReplayState invalidFrontend = captured;
            invalidFrontend.vertexCount =
                static_cast<int32_t>(
                    GS_REPLAY_VERTEX_QUEUE_CAPACITY + 1u);
            t.IsFalse(
                encodeGsReplayState(
                    invalidFrontend, canonical, &stateError),
                "oversized primitive assembly should fail closed");
            invalidFrontend = captured;
            invalidFrontend.prim.type =
                static_cast<GSPrimType>(0xFFu);
            t.IsFalse(
                encodeGsReplayState(
                    invalidFrontend, canonical, &stateError),
                "invalid primitive types should fail closed");
        });

        tc.Run("GS replay state resumes partial host-to-local transfers", [](TestCase &t)
        {
            std::vector<uint8_t> originalVram(PS2_GS_VRAM_SIZE, 0u);
            GS original;
            original.init(
                originalVram.data(),
                static_cast<uint32_t>(originalVram.size()),
                nullptr);

            original.writeRegister(
                GS_REG_BITBLTBUF,
                1ull << 48u); // DBW=1, DPSM=CT32.
            original.writeRegister(GS_REG_TRXPOS, 0ull);
            original.writeRegister(
                GS_REG_TRXREG,
                4ull | (1ull << 32u));
            original.writeRegister(GS_REG_TRXDIR, 0ull);

            constexpr uint32_t kPixel0 = 0x80112233u;
            constexpr uint32_t kPixel1 = 0x80445566u;
            constexpr uint32_t kPixel2 = 0x80778899u;
            constexpr uint32_t kPixel3 = 0x80AABBCCu;
            original.writeRegister(
                GS_REG_HWREG,
                static_cast<uint64_t>(kPixel0) |
                    (static_cast<uint64_t>(kPixel1) << 32u));

            const GsReplayState captured =
                original.captureReplayState();
            t.Equals(captured.transfer.copiedPixels, 2u,
                     "the capture should retain transfer progress");
            const std::vector<uint8_t> initialVram = originalVram;

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(1u, GIF_FMT_PACKED, 1u));
            appendU64(packet, 0x0Eull);
            appendGifAd(
                packet,
                static_cast<uint64_t>(kPixel2) |
                    (static_cast<uint64_t>(kPixel3) << 32u),
                GS_REG_HWREG);
            original.processGIFPacket(
                packet.data(),
                static_cast<uint32_t>(packet.size()));

            std::vector<uint8_t> replayVram = initialVram;
            GS replay;
            replay.init(
                replayVram.data(),
                static_cast<uint32_t>(replayVram.size()),
                nullptr);
            t.IsTrue(replay.restoreReplayState(captured),
                     "the partial transfer state should restore");
            replay.processGIFPacket(
                packet.data(),
                static_cast<uint32_t>(packet.size()));
            t.IsTrue(replayVram == originalVram,
                     "restored transfer progress should reproduce all VRAM bytes");
            t.Equals(
                replay.ReadVram(GS_PSM_CT32, 0u, 1u, 2u, 0u),
                kPixel2,
                "the resumed transfer should continue at the third pixel");
            t.Equals(
                replay.ReadVram(GS_PSM_CT32, 0u, 1u, 3u, 0u),
                kPixel3,
                "the resumed transfer should complete at the fourth pixel");
        });

        tc.Run("GS CSR/IMR support coherent 64-bit and 32-bit access", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kGsCsr = 0x12001000u;
            constexpr uint32_t kGsImr = 0x12001010u;

            const uint64_t csrPattern = 0xA1B2C3D4E5F60718ull;
            mem.write64(kGsCsr, csrPattern);
            t.Equals(mem.read64(kGsCsr), csrPattern, "64-bit CSR read should match prior 64-bit write");
            t.Equals(mem.read32(kGsCsr), static_cast<uint32_t>(csrPattern & 0xFFFFFFFFull), "CSR low dword read should match");
            t.Equals(mem.read32(kGsCsr + 4u), static_cast<uint32_t>(csrPattern >> 32), "CSR high dword read should match");

            mem.write32(kGsCsr, 0x11223344u);
            t.Equals(mem.read64(kGsCsr), 0xA1B2C3D411223344ull, "32-bit low write should preserve CSR high dword");

            mem.write32(kGsCsr + 4u, 0x55667788u);
            t.Equals(mem.read64(kGsCsr), 0x5566778811223344ull, "32-bit high write should preserve CSR low dword");

            const uint64_t imrPattern = 0x0123456789ABCDEFull;
            mem.write64(kGsImr, imrPattern);
            t.Equals(mem.read64(kGsImr), imrPattern, "IMR 64-bit read should match prior write");
            t.Equals(mem.read32(kGsImr), 0x89ABCDEFu, "IMR low dword should match");
            t.Equals(mem.read32(kGsImr + 4u), 0x01234567u, "IMR high dword should match");
        });

        tc.Run("unknown GS privileged offsets are no-op and read as zero", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kKnownBusdir = 0x12001040u;
            constexpr uint32_t kUnknown = 0x12001008u; // inside GS priv range, but not mapped by gsRegPtr.

            mem.write64(kKnownBusdir, 0xCAFEBABE12345678ull);
            const uint64_t before = mem.read64(kKnownBusdir);
            mem.write32(kUnknown, 0xDEADBEEFu);
            t.Equals(mem.read32(kUnknown), 0u, "unknown GS offset should read as zero");
            t.Equals(mem.read64(kKnownBusdir), before, "unknown GS writes should not corrupt mapped GS registers");
        });

        tc.Run("GS writeIORegister increments GS write counter", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kGsPmode = 0x12000000u;
            constexpr uint32_t kGsImr = 0x12001010u;

            const uint64_t countBefore = mem.gsWriteCount();
            t.IsTrue(mem.writeIORegister(kGsPmode, 0x11u), "writeIORegister PMODE should succeed");
            t.IsTrue(mem.writeIORegister(kGsImr, 0x22u), "writeIORegister IMR should succeed");
            t.Equals(mem.gsWriteCount(), countBefore + 2ull, "GS IO writes should increment GS write counter");

            t.Equals(mem.readIORegister(kGsPmode), 0x11u, "writeIORegister PMODE value should be readable");
            t.Equals(mem.readIORegister(kGsImr), 0x22u, "writeIORegister IMR value should be readable");
        });

        tc.Run("GsPutIMR and GsGetIMR roundtrip old and new values", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "runtime memory initialize should succeed");
            runtime.memory().gs().imr = 0xAAAABBBBCCCCDDDDull;

            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};

            setRegU64(ctx, 4, 0x3333444411112222ull);
            GsPutIMR(rdram.data(), &ctx, &runtime);

            const uint64_t oldImr = getReturnU64(ctx);
            t.Equals(oldImr, 0xAAAABBBBCCCCDDDDull, "GsPutIMR should return previous IMR");
            t.Equals(runtime.memory().gs().imr, 0x3333444411112222ull, "GsPutIMR should update GS IMR");

            std::memset(&ctx, 0, sizeof(ctx));
            GsGetIMR(rdram.data(), &ctx, &runtime);
            const uint64_t currentImr = getReturnU64(ctx);
            t.Equals(currentImr, 0x3333444411112222ull, "GsGetIMR should return current GS IMR");
        });

        tc.Run("GsSetCrt updates SMODE2 for host presentation mode", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "runtime memory initialize should succeed");

            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};
            setRegU32(ctx, 4, 1u); // interlaced
            setRegU32(ctx, 5, 0u); // NTSC
            setRegU32(ctx, 6, 0u); // field mode

            runtime.memory().gs().pmode = 0u;
            runtime.memory().gs().smode2 = 0u;
            GsSetCrt(rdram.data(), &ctx, &runtime);

            t.Equals(runtime.memory().gs().smode2, 0x1ull,
                     "GsSetCrt should publish interlaced field mode through SMODE2");
            t.Equals(runtime.memory().gs().pmode & 0x3ull, 0x1ull,
                     "GsSetCrt should leave CRT1 enabled for presentation");
            t.Equals(getRegU32Test(ctx, 2), 0u,
                     "GsSetCrt should return success");
        });

        tc.Run("sceGsSetDefDBuffDc seeds display envs and swap applies the selected page", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "runtime memory initialize should succeed");

            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            constexpr uint32_t kEnvAddr = 0x4000u;
            constexpr uint32_t kDispEnvSize = 40u;
            constexpr uint32_t kDBuffSize = 0x330u;
            constexpr uint32_t kDispFbOffset = 16u;
            constexpr uint32_t kDisplayOffset = 24u;
            constexpr uint32_t kDraw01Offset = 0x60u;
            constexpr uint32_t kFrame1Offset = kDraw01Offset + 0x00u;
            constexpr uint32_t kFrame1AddrOffset = kDraw01Offset + 0x08u;
            constexpr uint32_t kXYOffset1Offset = kDraw01Offset + 0x20u;
            constexpr uint32_t kXYOffset1AddrOffset = kDraw01Offset + 0x28u;

            R5900Context ctx{};
            setRegU32(ctx, 4, kEnvAddr);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 640u);
            setRegU32(ctx, 7, 448u);
            std::memset(rdram.data() + kEnvAddr, 0xCD, kDBuffSize);
            ps2_stubs::sceGsSetDefDBuffDc(rdram.data(), &ctx, &runtime);

            uint64_t dispfb0 = 0u;
            uint64_t display0 = 0u;
            uint64_t frame10 = 0u;
            uint64_t frame10Addr = 0u;
            uint64_t xyoffset10 = 0u;
            uint64_t xyoffset10Addr = 0u;
            std::memcpy(&dispfb0, rdram.data() + kEnvAddr + kDispFbOffset, sizeof(dispfb0));
            std::memcpy(&display0, rdram.data() + kEnvAddr + kDisplayOffset, sizeof(display0));
            std::memcpy(&frame10, rdram.data() + kEnvAddr + kFrame1Offset, sizeof(frame10));
            std::memcpy(&frame10Addr, rdram.data() + kEnvAddr + kFrame1AddrOffset, sizeof(frame10Addr));
            std::memcpy(&xyoffset10, rdram.data() + kEnvAddr + kXYOffset1Offset, sizeof(xyoffset10));
            std::memcpy(&xyoffset10Addr, rdram.data() + kEnvAddr + kXYOffset1AddrOffset, sizeof(xyoffset10Addr));

            t.Equals((dispfb0 >> 9) & 0x3Fu, 10ull, "dbuff display env should seed FBW from width");
            t.Equals((display0 >> 32) & 0x0FFFull, 639ull, "dbuff display env should seed DW from width");
            t.Equals((display0 >> 44) & 0x07FFull, 447ull, "dbuff display env should seed DH from height");
            t.Equals((frame10 >> 16) & 0x3Full, 10ull, "dbuff draw env should seed FRAME FBW from width");
            t.Equals(frame10Addr, 0x4Cull, "dbuff draw env should seed FRAME_1 register id");
            t.Equals(xyoffset10 & 0xFFFFull, 0x6C00ull, "dbuff draw env should seed OFX in 12.4 fixed point");
            t.Equals((xyoffset10 >> 32) & 0xFFFFull, 0x7200ull, "dbuff draw env should seed OFY in 12.4 fixed point");
            t.Equals(xyoffset10Addr, 0x18ull, "dbuff draw env should seed XYOFFSET_1 register id");

            dispfb0 = (dispfb0 & ~0x1FFull) | 150ull;
            std::memcpy(rdram.data() + kEnvAddr + kDispFbOffset, &dispfb0, sizeof(dispfb0));

            uint64_t dispfb1 = dispfb0;
            dispfb1 = (dispfb1 & ~0x1FFull) | 151ull;
            std::memcpy(rdram.data() + kEnvAddr + kDispEnvSize + kDispFbOffset, &dispfb1, sizeof(dispfb1));

            runtime.memory().gs().dispfb1 = 0xDEADBEEFDEADBEEFull;
            runtime.memory().gs().display1 = 0xCAFEF00DCAFEF00Dull;

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kEnvAddr);
            setRegU32(ctx, 5, 1u);
            ps2_stubs::sceGsSwapDBuffDc(rdram.data(), &ctx, &runtime);

            t.Equals(runtime.memory().gs().dispfb2 & 0x1FFull, 151ull,
                     "sceGsSwapDBuffDc should program GS to the selected display page");
            t.Equals((runtime.memory().gs().display2 >> 32) & 0x0FFFull, 639ull,
                     "sceGsSwapDBuffDc should preserve the display width from the seeded env");
            t.Equals(runtime.memory().gs().pmode & 0x3ull, 0x2ull,
                     "sceGsSwapDBuffDc should enable read circuit 2 only");
            t.Equals(runtime.memory().gs().dispfb1, 0xDEADBEEFDEADBEEFull,
                     "sceGsSwapDBuffDc should preserve circuit-1 DISPFB1");
            t.Equals(runtime.memory().gs().display1, 0xCAFEF00DCAFEF00Dull,
                     "sceGsSwapDBuffDc should preserve circuit-1 DISPLAY1");
        });

        tc.Run("sceGsSetDefDispEnv initializes the complete Sony libgraph display environment", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "runtime memory initialize should succeed");
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            R5900Context resetCtx{};
            setRegU32(resetCtx, 4, 0u);
            setRegU32(resetCtx, 5, 1u); // interlaced
            setRegU32(resetCtx, 6, 2u); // NTSC
            setRegU32(resetCtx, 7, 1u); // frame mode
            ps2_stubs::sceGsResetGraph(rdram.data(), &resetCtx, &runtime);

            constexpr uint32_t kEnvAddr = 0x6000u;
            std::memset(rdram.data() + kEnvAddr, 0xCD, sizeof(GsDispEnvMem));

            R5900Context ctx{};
            setRegU32(ctx, 4, kEnvAddr);
            setRegU32(ctx, 5, 0u);   // PSMCT32
            setRegU32(ctx, 6, 512u); // width
            setRegU32(ctx, 7, 448u); // height
            setRegU32(ctx, 8, 4u);   // dx
            setRegU32(ctx, 9, 5u);   // dy
            ps2_stubs::sceGsSetDefDispEnv(rdram.data(), &ctx, &runtime);

            GsDispEnvMem env{};
            std::memcpy(&env, rdram.data() + kEnvAddr, sizeof(env));
            t.Equals(env.pmode, 0x66ull,
                     "sceGsSetDefDispEnv should seed circuit-2 PMODE mixing");
            t.Equals(env.smode2, 0x3ull,
                     "sceGsSetDefDispEnv should seed interlaced frame-mode SMODE2");
            t.Equals((env.dispfb >> 9) & 0x3Full, 8ull,
                     "sceGsSetDefDispEnv should derive FBW from the requested width");
            t.Equals(env.display & 0xFFFull, 4ull,
                     "sceGsSetDefDispEnv should preserve dx");
            t.Equals((env.display >> 12) & 0x7FFull, 5ull,
                     "sceGsSetDefDispEnv should preserve dy");
            t.Equals((env.display >> 32) & 0xFFFull, 511ull,
                     "sceGsSetDefDispEnv should encode display width");
            t.Equals((env.display >> 44) & 0x7FFull, 447ull,
                     "sceGsSetDefDispEnv should encode display height");
            t.Equals(env.bgcolor, 0ull,
                     "sceGsSetDefDispEnv should initialize BGCOLOR");
            t.Equals(getRegU32Test(ctx, 2), 0u,
                     "sceGsSetDefDispEnv should return success");
        });

        tc.Run("sceGsPutDispEnv programs read circuit 2 without clobbering circuit 1", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "runtime memory initialize should succeed");
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            constexpr uint32_t kEnvAddr = 0x6800u;
            const GsDispEnvMem env{0x66ull, 0x3ull, 0x1000ull,
                                   0x1BF1FF00000000ull, 0x00445566ull};
            std::memcpy(rdram.data() + kEnvAddr, &env, sizeof(env));

            runtime.memory().gs().dispfb1 = 0xDEADBEEFDEADBEEFull;
            runtime.memory().gs().display1 = 0xCAFEF00DCAFEF00Dull;

            R5900Context ctx{};
            setRegU32(ctx, 4, kEnvAddr);
            ps2_stubs::sceGsPutDispEnv(rdram.data(), &ctx, &runtime);

            t.Equals(runtime.memory().gs().pmode, env.pmode,
                     "sceGsPutDispEnv should program PMODE");
            t.Equals(runtime.memory().gs().smode2, env.smode2,
                     "sceGsPutDispEnv should program SMODE2");
            t.Equals(runtime.memory().gs().dispfb2, env.dispfb,
                     "sceGsPutDispEnv should program circuit-2 DISPFB2");
            t.Equals(runtime.memory().gs().display2, env.display,
                     "sceGsPutDispEnv should program circuit-2 DISPLAY2");
            t.Equals(runtime.memory().gs().bgcolor, env.bgcolor,
                     "sceGsPutDispEnv should program BGCOLOR");
            t.Equals(runtime.memory().gs().dispfb1, 0xDEADBEEFDEADBEEFull,
                     "sceGsPutDispEnv should preserve circuit-1 DISPFB1");
            t.Equals(runtime.memory().gs().display1, 0xCAFEF00DCAFEF00Dull,
                     "sceGsPutDispEnv should preserve circuit-1 DISPLAY1");
        });

        tc.Run("sceGsSetDefDBuffDc seeds a clear packet and swap clears the draw buffer", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "runtime memory initialize should succeed");

            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            constexpr uint32_t kEnvAddr = 0x5000u;
            constexpr uint32_t kDBuffSize = 0x330u;
            constexpr uint32_t kClear0Offset = 0x160u;
            constexpr uint32_t kTestAAddrOffset = kClear0Offset + 0x08u;
            constexpr uint32_t kPrimAddrOffset = kClear0Offset + 0x18u;
            constexpr uint32_t kRgbaqOffset = kClear0Offset + 0x20u;
            constexpr uint32_t kRgbaqAddrOffset = kClear0Offset + 0x28u;
            constexpr uint32_t kXyz2AAddrOffset = kClear0Offset + 0x38u;
            constexpr uint32_t kXyz2BAddrOffset = kClear0Offset + 0x48u;
            constexpr uint32_t kTestBAddrOffset = kClear0Offset + 0x58u;
            constexpr uint32_t kClearColor = 0x80402010u;
            constexpr uint32_t kStackAddr = 0x900u;
            const uint32_t kZTest = 2u;
            const uint32_t kEnableClear = 1u;

            R5900Context ctx{};
            setRegU32(ctx, 4, kEnvAddr);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 640u);
            setRegU32(ctx, 7, 448u);
            setRegU32(ctx, 29, kStackAddr);
            std::memset(rdram.data() + kEnvAddr, 0xCD, kDBuffSize);
            std::memcpy(rdram.data() + kStackAddr + 16u, &kZTest, sizeof(kZTest));
            std::memcpy(rdram.data() + kStackAddr + 24u, &kEnableClear, sizeof(kEnableClear));
            std::memset(runtime.memory().getGSVRAM(), 0xAB, 16u);
            ps2_stubs::sceGsSetDefDBuffDc(rdram.data(), &ctx, &runtime);

            uint64_t testAAddr = 0u;
            uint64_t primAddr = 0u;
            uint64_t rgbaqAddr = 0u;
            uint64_t xyz2AAddr = 0u;
            uint64_t xyz2BAddr = 0u;
            uint64_t testBAddr = 0u;
            std::memcpy(&testAAddr, rdram.data() + kEnvAddr + kTestAAddrOffset, sizeof(testAAddr));
            std::memcpy(&primAddr, rdram.data() + kEnvAddr + kPrimAddrOffset, sizeof(primAddr));
            std::memcpy(&rgbaqAddr, rdram.data() + kEnvAddr + kRgbaqAddrOffset, sizeof(rgbaqAddr));
            std::memcpy(&xyz2AAddr, rdram.data() + kEnvAddr + kXyz2AAddrOffset, sizeof(xyz2AAddr));
            std::memcpy(&xyz2BAddr, rdram.data() + kEnvAddr + kXyz2BAddrOffset, sizeof(xyz2BAddr));
            std::memcpy(&testBAddr, rdram.data() + kEnvAddr + kTestBAddrOffset, sizeof(testBAddr));

            t.Equals(testAAddr, 0x47ull, "dbuff clear packet should program TEST_1 before clearing");
            t.Equals(primAddr, 0x00ull, "dbuff clear packet should program PRIM before clearing");
            t.Equals(rgbaqAddr, 0x01ull, "dbuff clear packet should expose RGBAQ for runtime color updates");
            t.Equals(xyz2AAddr, 0x05ull, "dbuff clear packet should seed the first clear vertex as XYZ2");
            t.Equals(xyz2BAddr, 0x05ull, "dbuff clear packet should seed the second clear vertex as XYZ2");
            t.Equals(testBAddr, 0x47ull, "dbuff clear packet should restore TEST_1 after clearing");

            uint64_t rgbaq = static_cast<uint64_t>(kClearColor);
            std::memcpy(rdram.data() + kEnvAddr + kRgbaqOffset, &rgbaq, sizeof(rgbaq));

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kEnvAddr);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::sceGsSwapDBuffDc(rdram.data(), &ctx, &runtime);

            uint32_t clearedPixel = 0u;
            std::memcpy(&clearedPixel, runtime.memory().getGSVRAM(), sizeof(clearedPixel));
            t.Equals(clearedPixel, kClearColor,
                     "sceGsSwapDBuffDc should execute the seeded clear packet against the active draw buffer");

            constexpr uint32_t kMidX = 320u;
            constexpr uint32_t kMidY = 200u;
            const uint32_t kMidOffset = ((kMidY * 640u) + kMidX) * 4u;
            uint32_t clearedMidPixel = 0u;
            std::memcpy(&clearedMidPixel, runtime.memory().getGSVRAM() + kMidOffset, sizeof(clearedMidPixel));
            t.Equals(clearedMidPixel, kClearColor,
                     "sceGsSwapDBuffDc should clear the interior of the active draw buffer, not just the first pixel");
        });

        tc.Run("sceGsSetDefDBuffDc accepts trailing args from the recompiler register ABI", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "runtime memory initialize should succeed");

            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            constexpr uint32_t kEnvAddr = 0x5400u;
            constexpr uint32_t kDBuffSize = 0x330u;
            constexpr uint32_t kClear0Offset = 0x160u;
            constexpr uint32_t kRgbaqOffset = kClear0Offset + 0x20u;
            constexpr uint32_t kRgbaqAddrOffset = kClear0Offset + 0x28u;
            constexpr uint32_t kStackAddr = 0xA00u;
            constexpr uint32_t kClearColor = 0x40201008u;

            R5900Context ctx{};
            setRegU32(ctx, 4, kEnvAddr);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 640u);
            setRegU32(ctx, 7, 448u);
            setRegU32(ctx, 8, 2u);
            setRegU32(ctx, 9, 58u);
            setRegU32(ctx, 10, 1u);
            setRegU32(ctx, 29, kStackAddr);
            std::memset(rdram.data() + kEnvAddr, 0xCD, kDBuffSize);
            std::memset(runtime.memory().getGSVRAM(), 0xAB, 16u);

            ps2_stubs::sceGsSetDefDBuffDc(rdram.data(), &ctx, &runtime);

            uint64_t rgbaqAddr = 0u;
            std::memcpy(&rgbaqAddr, rdram.data() + kEnvAddr + kRgbaqAddrOffset, sizeof(rgbaqAddr));
            t.Equals(rgbaqAddr, 0x01ull,
                     "sceGsSetDefDBuffDc should seed the clear packet when trailing args arrive in t0-t2");

            const uint64_t rgbaq = static_cast<uint64_t>(kClearColor);
            std::memcpy(rdram.data() + kEnvAddr + kRgbaqOffset, &rgbaq, sizeof(rgbaq));

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kEnvAddr);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::sceGsSwapDBuffDc(rdram.data(), &ctx, &runtime);

            uint32_t clearedPixel = 0u;
            std::memcpy(&clearedPixel, runtime.memory().getGSVRAM(), sizeof(clearedPixel));
            t.Equals(clearedPixel, kClearColor,
                     "sceGsSwapDBuffDc should honor a clear packet seeded from register-based trailing args");
        });

        tc.Run("clearFramebufferContext clears the requested context even if another context is active", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kCtx0Color = 0x11223344u;
            constexpr uint32_t kCtx1Sentinel = 0xAABBCCDDu;

            gs.writeRegister(GS_REG_FRAME_1, (1ull << 16)); // FBP=0, FBW=1, PSMCT32
            gs.writeRegister(GS_REG_SCISSOR_1, (0ull << 0) | (0ull << 16) | (1ull << 32) | (1ull << 48));
            gs.writeRegister(GS_REG_FRAME_2, 150ull | (1ull << 16));
            gs.writeRegister(GS_REG_SCISSOR_2, (0ull << 0) | (0ull << 16) | (1ull << 32) | (1ull << 48));
            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT) | (1ull << 9));

            writeReferenceFramePSMCT32Pixel(vram, 150u, 1u, 0u, 1u, kCtx1Sentinel);

            t.IsTrue(gs.clearFramebufferContext(0u, kCtx0Color),
                     "context-targeted clear should succeed for a configured CT32 framebuffer");

            const uint32_t ctx0Pixel = readReferenceFramePSMCT32Pixel(vram, 0u, 1u, 0u, 1u);
            t.Equals(ctx0Pixel, kCtx0Color,
                     "context-targeted clear should write the requested context even when PRIM.ctxt points elsewhere");

            const uint32_t ctx1Pixel = readReferenceFramePSMCT32Pixel(vram, 150u, 1u, 0u, 1u);
            t.Equals(ctx1Pixel, kCtx1Sentinel,
                     "context-targeted clear should leave the other context framebuffer untouched");
        });

        tc.Run("PABE bypasses alpha blend for low-alpha source pixels", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            gs.writeRegister(GS_REG_FRAME_1, (1ull << 16)); // FBW=1, PSMCT32, FBP=0
            gs.writeRegister(GS_REG_ZBUF_1, (1ull << 32));
            gs.writeRegister(GS_REG_SCISSOR_1, 0ull);
            gs.writeRegister(GS_REG_ALPHA_1, 0x6000000064ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT) | (1ull << 6));

            const uint32_t dstWhite = 0xFFFFFFFFu;
            std::memcpy(vram.data(), &dstWhite, sizeof(dstWhite));

            gs.writeRegister(GS_REG_PABE, 0ull);
            gs.writeRegister(GS_REG_RGBAQ, 0x01000000ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);

            uint32_t blendedPixel = 0u;
            std::memcpy(&blendedPixel, vram.data(), sizeof(blendedPixel));
            t.Equals(blendedPixel, 0x013F3F3Fu,
                     "without PABE, low-alpha fullscreen copies should still apply ALPHA blending");

            std::memcpy(vram.data(), &dstWhite, sizeof(dstWhite));

            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT) | (1ull << 6));
            gs.writeRegister(GS_REG_PABE, 1ull);
            gs.writeRegister(GS_REG_RGBAQ, 0x01000000ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);

            uint32_t pabeBypassedPixel = 0u;
            std::memcpy(&pabeBypassedPixel, vram.data(), sizeof(pabeBypassedPixel));
            t.Equals(pabeBypassedPixel, 0x01000000u,
                     "with PABE enabled, low-alpha source pixels should bypass ALPHA blending and overwrite the destination");

            std::memcpy(vram.data(), &dstWhite, sizeof(dstWhite));

            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT) | (1ull << 6));
            gs.writeRegister(GS_REG_PABE, 1ull);
            gs.writeRegister(GS_REG_RGBAQ, 0x80000000ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);

            uint32_t highAlphaPixel = 0u;
            std::memcpy(&highAlphaPixel, vram.data(), sizeof(highAlphaPixel));
            t.Equals(highAlphaPixel, 0x803F3F3Fu,
                     "with PABE enabled, high-alpha source pixels should still use the configured ALPHA blend");
        });

        tc.Run("FBA forces the framebuffer alpha high bit on CT32 writes", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            gs.writeRegister(GS_REG_FRAME_1, (1ull << 16));
            gs.writeRegister(GS_REG_ZBUF_1, (1ull) << 32);
            gs.writeRegister(GS_REG_SCISSOR_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);

            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT));
            gs.writeRegister(GS_REG_FBA_1, 0ull);
            gs.writeRegister(GS_REG_RGBAQ, 0x01112233ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);

            uint32_t pixelWithoutFba = 0u;
            std::memcpy(&pixelWithoutFba, vram.data(), sizeof(pixelWithoutFba));
            t.Equals(pixelWithoutFba, 0x01112233u,
                     "without FBA, CT32 writes should preserve the source alpha byte");

            std::memset(vram.data(), 0, sizeof(uint32_t));

            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT));
            gs.writeRegister(GS_REG_FBA_1, 1ull);
            gs.writeRegister(GS_REG_RGBAQ, 0x01112233ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);

            uint32_t pixelWithFba = 0u;
            std::memcpy(&pixelWithFba, vram.data(), sizeof(pixelWithFba));
            t.Equals(pixelWithFba, 0x81112233u,
                     "with FBA enabled, CT32 writes should force the framebuffer alpha high bit");
        });

        tc.Run("CT32 raster writes alias cleanly into later CT32 texture sampling", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint64_t kFrame1 =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf1 = (1ull << 32);
            constexpr uint64_t kScissor1 =
                (0ull << 0) |
                (1ull << 16) |
                (0ull << 32) |
                (1ull << 48);
            constexpr uint64_t kFrame2 =
                (150ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf2 = (1ull << 32);
            constexpr uint64_t kScissor2 = 0ull;
            constexpr uint64_t kTex0_2 =
                (0ull << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
                (0ull << 26) |
                (1ull << 30) |
                (1ull << 34) |
                (1ull << 35);
            constexpr uint64_t kPointPrim = static_cast<uint64_t>(GS_PRIM_POINT);
            constexpr uint64_t kCopyPrim =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |
                (1ull << 8) |
                (1ull << 9);
            constexpr uint64_t kSourceColor = 0x80665544ull;
            constexpr uint64_t kPointXyz =
                (0ull << 0) |
                (16ull << 16);
            constexpr uint64_t kUvRow1 =
                (0ull << 0) |
                (16ull << 16);

            gs.writeRegister(GS_REG_FRAME_1, kFrame1);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf1);
            gs.writeRegister(GS_REG_SCISSOR_1, kScissor1);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_PRIM, kPointPrim);
            gs.writeRegister(GS_REG_RGBAQ, kSourceColor);
            gs.writeRegister(GS_REG_XYZ2, kPointXyz);

            gs.writeRegister(GS_REG_FRAME_2, kFrame2);
            gs.writeRegister(GS_REG_ZBUF_2, kZbuf2);
            gs.writeRegister(GS_REG_SCISSOR_2, kScissor2);
            gs.writeRegister(GS_REG_XYOFFSET_2, 0ull);
            gs.writeRegister(GS_REG_TEST_2, 0x30000ull);
            gs.writeRegister(GS_REG_TEX0_2, kTex0_2);
            gs.writeRegister(GS_REG_PRIM, kCopyPrim);
            gs.writeRegister(GS_REG_RGBAQ, 0x80808080ull);
            gs.writeRegister(GS_REG_UV, kUvRow1);
            gs.writeRegister(GS_REG_XYZ2, 0ull);
            gs.writeRegister(GS_REG_UV, kUvRow1);
            gs.writeRegister(GS_REG_XYZ2, (16ull << 0) | (16ull << 16));

            const uint32_t dstPixel = readReferenceFramePSMCT32Pixel(vram, 150u, 1u, 0u, 0u);
            t.Equals(dstPixel, static_cast<uint32_t>(kSourceColor),
                     "CT32 primitives should land in the same local-memory layout that later CT32 texture sampling expects");
        });

        tc.Run("FST sprite 1:1 CT32 copies preserve source texels at the right and bottom edges", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kTexTbp = 64u;
            constexpr uint64_t kFrame =
                (150ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (3ull << 16) |
                (0ull << 32) |
                (3ull << 48);
            constexpr uint64_t kTex0 =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
                (2ull << 26) |
                (2ull << 30) |
                (1ull << 34);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |
                (1ull << 8);
            constexpr uint64_t kXyz0 = 0ull;
            constexpr uint64_t kXyz1 =
                (static_cast<uint64_t>(4u << 4) << 0) |
                (static_cast<uint64_t>(4u << 4) << 16);
            constexpr uint64_t kUv0 = 0ull;
            constexpr uint64_t kUv1 =
                ((4ull * 16ull) << 0) |
                ((4ull * 16ull) << 16);
            constexpr uint32_t kSourcePixels[4] = {
                0x800000FFu,
                0x8000FF00u,
                0x80FF0000u,
                0x80FFFFFFu,
            };

            for (uint32_t y = 0u; y < 4u; ++y)
            {
                for (uint32_t x = 0u; x < 4u; ++x)
                {
                    writeReferencePSMCT32Pixel(vram, kTexTbp, 1u, x, y, kSourcePixels[x]);
                }
            }

            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_ALPHA_1, 0ull);
            gs.writeRegister(GS_REG_TEX0_1, kTex0);
            gs.writeRegister(GS_REG_TEX1_1, 0ull);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, 0x80808080ull);
            gs.writeRegister(GS_REG_UV, kUv0);
            gs.writeRegister(GS_REG_XYZ2, kXyz0);
            gs.writeRegister(GS_REG_UV, kUv1);
            gs.writeRegister(GS_REG_XYZ2, kXyz1);

            for (uint32_t y = 0u; y < 4u; ++y)
            {
                for (uint32_t x = 0u; x < 4u; ++x)
                {
                    const uint32_t pixel = readReferenceFramePSMCT32Pixel(vram, 150u, 1u, x, y);
                    t.Equals(pixel, kSourcePixels[x],
                             "1:1 FST sprite copies should preserve each source texel without off-by-one edge skew");
                }
            }
        });

        tc.Run("fullscreen display copy tracks the preferred presentation source frame", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint64_t kFrame2 =
                150ull |
                (10ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kScissor2 =
                (0ull << 0) |
                (639ull << 16) |
                (0ull << 32) |
                (479ull << 48);
            constexpr uint64_t kXYOffset2 =
                (static_cast<uint64_t>(1728u << 4) << 0) |
                (static_cast<uint64_t>(1808u << 4) << 32);
            constexpr uint64_t kAlpha2 = 0x6000000064ull;
            constexpr uint64_t kTest2 = 0x30000ull;
            constexpr uint64_t kTex0_2 =
                (0ull << 0) |
                (10ull << 14) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
                (10ull << 26) |
                (9ull << 30) |
                (1ull << 34);
            constexpr uint64_t kPrimCopy =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |
                (1ull << 6) |
                (1ull << 8) |
                (1ull << 9);
            constexpr uint64_t kXyz0 =
                (static_cast<uint64_t>(1728u << 4) << 0) |
                (static_cast<uint64_t>(1808u << 4) << 16) |
                (256ull << 32);
            constexpr uint64_t kXyz1 =
                (static_cast<uint64_t>(2368u << 4) << 0) |
                (static_cast<uint64_t>(2288u << 4) << 16) |
                (256ull << 32);
            constexpr uint64_t kUv0 =
                (8ull << 0) |
                (8ull << 16);
            constexpr uint64_t kUv1 =
                ((8ull + (640ull * 16ull)) << 0) |
                ((8ull + (480ull * 16ull)) << 16);

            gs.writeRegister(GS_REG_FRAME_2, kFrame2);
            gs.writeRegister(GS_REG_SCISSOR_2, kScissor2);
            gs.writeRegister(GS_REG_XYOFFSET_2, kXYOffset2);
            gs.writeRegister(GS_REG_ALPHA_2, kAlpha2);
            gs.writeRegister(GS_REG_TEST_2, kTest2);
            gs.writeRegister(GS_REG_TEX0_2, kTex0_2);
            gs.writeRegister(GS_REG_PRIM, kPrimCopy);
            gs.writeRegister(GS_REG_RGBAQ, 0x80808080ull);
            gs.writeRegister(GS_REG_UV, kUv0);
            gs.writeRegister(GS_REG_XYZ2, kXyz0);
            gs.writeRegister(GS_REG_UV, kUv1);
            gs.writeRegister(GS_REG_XYZ2, kXyz1);

            GSFrameReg preferredSource{};
            uint32_t preferredDestFbp = 0u;
            t.IsTrue(gs.getPreferredDisplaySource(preferredSource, preferredDestFbp),
                     "fullscreen display-copy sprites should record their source frame for host presentation");
            t.Equals(preferredDestFbp, 150u,
                     "preferred presentation tracking should target the copied display page");
            t.Equals(preferredSource.fbp, 0u,
                     "preferred presentation tracking should expose the copy source frame base");
            t.Equals(preferredSource.fbw, 10u,
                     "preferred presentation tracking should expose the copy source width");
            t.Equals(static_cast<uint32_t>(preferredSource.psm), static_cast<uint32_t>(GS_PSM_CT32),
                     "preferred presentation tracking should expose the copy source format");

            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT) | (1ull << 9));
            gs.writeRegister(GS_REG_RGBAQ, 0xFFFFFFFFull);
            gs.writeRegister(GS_REG_XYZ2, kXyz0);

            t.IsFalse(gs.getPreferredDisplaySource(preferredSource, preferredDestFbp),
                      "non-copy primitives that touch the display target should invalidate the preferred presentation source");
        });

        tc.Run("latched host presentation frame stays stable until the next latch", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GSRegisters regs{};
            regs.pmode = 1ull;
            regs.dispfb1 =
                150ull |
                (10ull << 9) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 15);
            regs.display1 =
                (639ull << 32) |
                (447ull << 44);

            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), &regs);

            constexpr uint32_t kDisplayPixel = 0x00332211u;
            constexpr uint32_t kSourcePixel = 0x00665544u;
            constexpr uint32_t kUpdatedSourcePixel = 0x00998877u;
            writeReferenceFramePSMCT32Pixel(vram, 150u, 10u, 0u, 0u, kDisplayPixel);
            std::memcpy(vram.data() + 0u, &kSourcePixel, sizeof(kSourcePixel));

            constexpr uint64_t kFrame2 =
                150ull |
                (10ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf2 = (1ull << 32);
            constexpr uint64_t kScissor2 =
                (0ull << 0) |
                (639ull << 16) |
                (0ull << 32) |
                (479ull << 48);
            constexpr uint64_t kXYOffset2 =
                (static_cast<uint64_t>(1728u << 4) << 0) |
                (static_cast<uint64_t>(1808u << 4) << 32);
            constexpr uint64_t kAlpha2 = 0x6000000064ull;
            constexpr uint64_t kTest2 = 0x30000ull;
            constexpr uint64_t kTex0_2 =
                (0ull << 0) |
                (10ull << 14) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
                (10ull << 26) |
                (9ull << 30) |
                (1ull << 34);
            constexpr uint64_t kPrimCopy =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |
                (1ull << 6) |
                (1ull << 8) |
                (1ull << 9);
            constexpr uint64_t kXyz0 =
                (static_cast<uint64_t>(1728u << 4) << 0) |
                (static_cast<uint64_t>(1808u << 4) << 16) |
                (256ull << 32);
            constexpr uint64_t kXyz1 =
                (static_cast<uint64_t>(2368u << 4) << 0) |
                (static_cast<uint64_t>(2288u << 4) << 16) |
                (256ull << 32);
            constexpr uint64_t kUv0 =
                (8ull << 0) |
                (8ull << 16);
            constexpr uint64_t kUv1 =
                ((8ull + (640ull * 16ull)) << 0) |
                ((8ull + (480ull * 16ull)) << 16);

            gs.writeRegister(GS_REG_FRAME_2, kFrame2);
            gs.writeRegister(GS_REG_ZBUF_2, kZbuf2);
            gs.writeRegister(GS_REG_SCISSOR_2, kScissor2);
            gs.writeRegister(GS_REG_XYOFFSET_2, kXYOffset2);
            gs.writeRegister(GS_REG_ALPHA_2, kAlpha2);
            gs.writeRegister(GS_REG_TEST_2, kTest2);
            gs.writeRegister(GS_REG_TEX0_2, kTex0_2);
            gs.writeRegister(GS_REG_PRIM, kPrimCopy);
            gs.writeRegister(GS_REG_RGBAQ, 0x80808080ull);
            gs.writeRegister(GS_REG_UV, kUv0);
            gs.writeRegister(GS_REG_XYZ2, kXyz0);
            gs.writeRegister(GS_REG_UV, kUv1);
            gs.writeRegister(GS_REG_XYZ2, kXyz1);

            gs.latchHostPresentationFrame();

            std::vector<uint8_t> latchedFrame;
            uint32_t latchedWidth = 0u;
            uint32_t latchedHeight = 0u;
            uint32_t displayFbp = 0u;
            uint32_t sourceFbp = 0u;
            bool usedPreferred = false;
            t.IsTrue(gs.copyLatchedHostPresentationFrame(latchedFrame,
                                                         latchedWidth,
                                                         latchedHeight,
                                                         &displayFbp,
                                                         &sourceFbp,
                                                         &usedPreferred),
                     "host presentation latch should produce a readable frame");
            t.Equals(displayFbp, 150u,
                     "latched host presentation should remember the selected display page");
            t.Equals(sourceFbp, 0u,
                     "latched host presentation should switch to the fullscreen copy source");
            t.IsTrue(usedPreferred,
                     "latched host presentation should record when it used the preferred copy source");
            t.Equals(latchedWidth, 640u,
                     "latched host presentation should preserve display width");
            t.Equals(latchedHeight, 448u,
                     "latched host presentation should preserve display height");
            t.Equals(static_cast<uint32_t>(latchedFrame[0]), 0x44u,
                     "latched host presentation should expose the source frame RGB data");
            t.Equals(static_cast<uint32_t>(latchedFrame[1]), 0x55u,
                     "latched host presentation should preserve the source frame green channel");
            t.Equals(static_cast<uint32_t>(latchedFrame[2]), 0x66u,
                     "latched host presentation should preserve the source frame blue channel");
            t.Equals(static_cast<uint32_t>(latchedFrame[3]), 0xFFu,
                     "latched host presentation should normalize framebuffer alpha for host upload");

            std::memcpy(vram.data() + 0u, &kUpdatedSourcePixel, sizeof(kUpdatedSourcePixel));

            std::vector<uint8_t> staleFrame;
            uint32_t staleWidth = 0u;
            uint32_t staleHeight = 0u;
            t.IsTrue(gs.copyLatchedHostPresentationFrame(staleFrame, staleWidth, staleHeight),
                     "latched host presentation should remain readable without relatching");
            t.Equals(static_cast<uint32_t>(staleFrame[0]), 0x44u,
                     "latched host presentation should stay stable until the next latch");

            gs.latchHostPresentationFrame();

            std::vector<uint8_t> refreshedFrame;
            uint32_t refreshedWidth = 0u;
            uint32_t refreshedHeight = 0u;
            t.IsTrue(gs.copyLatchedHostPresentationFrame(refreshedFrame, refreshedWidth, refreshedHeight),
                     "latched host presentation should refresh after a new latch");
            t.Equals(static_cast<uint32_t>(refreshedFrame[0]), 0x77u,
                     "relatching should pick up the updated source frame contents");
        });

        tc.Run("latched host presentation frame is returned tightly packed for narrower display widths", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GSRegisters regs{};
            regs.pmode = 1ull;
            regs.dispfb1 =
                150ull |
                (1ull << 9) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 15);
            regs.display1 =
                (63ull << 32) |
                (63ull << 44);

            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), &regs);

            constexpr uint32_t kTopLeft = 0xFF332211u;
            constexpr uint32_t kTopRight = 0xFF665544u;
            constexpr uint32_t kBottomLeft = 0xFF998877u;
            constexpr uint32_t kBottomRight = 0xFFCCBBAAu;
            writeReferenceFramePSMCT32Pixel(vram, 150u, 1u, 0u, 0u, kTopLeft);
            writeReferenceFramePSMCT32Pixel(vram, 150u, 1u, 1u, 0u, kTopRight);
            writeReferenceFramePSMCT32Pixel(vram, 150u, 1u, 0u, 1u, kBottomLeft);
            writeReferenceFramePSMCT32Pixel(vram, 150u, 1u, 1u, 1u, kBottomRight);

            gs.latchHostPresentationFrame();

            std::vector<uint8_t> latchedFrame;
            uint32_t latchedWidth = 0u;
            uint32_t latchedHeight = 0u;
            t.IsTrue(gs.copyLatchedHostPresentationFrame(latchedFrame, latchedWidth, latchedHeight),
                     "latched host presentation should be readable for narrow display widths");
            t.Equals(latchedWidth, 64u,
                     "latched host presentation should preserve the decoded display width");
            t.Equals(latchedHeight, 64u,
                     "latched host presentation should preserve the decoded display height");
            t.Equals(static_cast<uint32_t>(latchedFrame.size()), 64u * 64u * 4u,
                     "latched host presentation should return a tightly packed RGBA buffer");

            uint32_t pixel = 0u;
            std::memcpy(&pixel, latchedFrame.data() + 0u, sizeof(pixel));
            t.Equals(pixel, kTopLeft,
                     "latched host presentation should keep the first row intact");
            std::memcpy(&pixel, latchedFrame.data() + 4u, sizeof(pixel));
            t.Equals(pixel, kTopRight,
                     "latched host presentation should pack the first row contiguously");
            std::memcpy(&pixel, latchedFrame.data() + (64u * 4u), sizeof(pixel));
            t.Equals(pixel, kBottomLeft,
                     "latched host presentation should start the second row immediately after the first");
            std::memcpy(&pixel, latchedFrame.data() + (64u * 4u) + 4u, sizeof(pixel));
            t.Equals(pixel, kBottomRight,
                     "latched host presentation should preserve subsequent rows without the internal 640-pixel stride");
        });

        tc.Run("latched host presentation reads preferred CT32 source with GS swizzle", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GSRegisters regs{};
            regs.pmode = 1ull;
            regs.dispfb1 =
                150ull |
                (10ull << 9) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 15);
            regs.display1 =
                (639ull << 32) |
                (447ull << 44);

            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), &regs);

            constexpr uint32_t kSourceTbp0 = 64u;
            constexpr uint32_t kSourcePixelRow1 = 0x00665544u;
            constexpr uint32_t kDisplayPixelRow1 = 0x00CCBBAAu;
            const uint32_t swizzledSourceOff = GSPSMCT32::addrPSMCT32(kSourceTbp0, 10u, 0u, 1u);
            std::memcpy(vram.data() + swizzledSourceOff, &kSourcePixelRow1, sizeof(kSourcePixelRow1));
            writeReferenceFramePSMCT32Pixel(vram, 150u, 10u, 0u, 1u, kDisplayPixelRow1);

            constexpr uint64_t kFrame2 =
                150ull |
                (10ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf2 = (static_cast<uint64_t>(1) << 32);
            constexpr uint64_t kScissor2 =
                (0ull << 0) |
                (639ull << 16) |
                (0ull << 32) |
                (479ull << 48);
            constexpr uint64_t kXYOffset2 =
                (static_cast<uint64_t>(1728u << 4) << 0) |
                (static_cast<uint64_t>(1808u << 4) << 32);
            constexpr uint64_t kAlpha2 = 0x6000000064ull;
            constexpr uint64_t kTest2 = 0x30000ull;
            constexpr uint64_t kTex0_2 =
                (static_cast<uint64_t>(kSourceTbp0) << 0) |
                (10ull << 14) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
                (10ull << 26) |
                (9ull << 30) |
                (1ull << 34);
            constexpr uint64_t kPrimCopy =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |
                (1ull << 6) |
                (1ull << 8) |
                (1ull << 9);
            constexpr uint64_t kXyz0 =
                (static_cast<uint64_t>(1728u << 4) << 0) |
                (static_cast<uint64_t>(1808u << 4) << 16) |
                (256ull << 32);
            constexpr uint64_t kXyz1 =
                (static_cast<uint64_t>(2368u << 4) << 0) |
                (static_cast<uint64_t>(2288u << 4) << 16) |
                (256ull << 32);
            constexpr uint64_t kUv0 =
                (8ull << 0) |
                (8ull << 16);
            constexpr uint64_t kUv1 =
                ((8ull + (640ull * 16ull)) << 0) |
                ((8ull + (480ull * 16ull)) << 16);

            gs.writeRegister(GS_REG_FRAME_2, kFrame2);
            gs.writeRegister(GS_REG_ZBUF_2, kZbuf2);
            gs.writeRegister(GS_REG_SCISSOR_2, kScissor2);
            gs.writeRegister(GS_REG_XYOFFSET_2, kXYOffset2);
            gs.writeRegister(GS_REG_ALPHA_2, kAlpha2);
            gs.writeRegister(GS_REG_TEST_2, kTest2);
            gs.writeRegister(GS_REG_TEX0_2, kTex0_2);
            gs.writeRegister(GS_REG_PRIM, kPrimCopy);
            gs.writeRegister(GS_REG_RGBAQ, 0x80808080ull);
            gs.writeRegister(GS_REG_UV, kUv0);
            gs.writeRegister(GS_REG_XYZ2, kXyz0);
            gs.writeRegister(GS_REG_UV, kUv1);
            gs.writeRegister(GS_REG_XYZ2, kXyz1);

            gs.latchHostPresentationFrame();

            std::vector<uint8_t> latchedFrame;
            uint32_t latchedWidth = 0u;
            uint32_t latchedHeight = 0u;
            uint32_t displayFbp = 0u;
            uint32_t sourceFbp = 0u;
            bool usedPreferred = false;
            t.IsTrue(gs.copyLatchedHostPresentationFrame(latchedFrame,
                                                         latchedWidth,
                                                         latchedHeight,
                                                         &displayFbp,
                                                         &sourceFbp,
                                                         &usedPreferred),
                     "preferred-source presentation should produce a host frame");
            t.IsTrue(usedPreferred,
                     "preferred-source presentation should use the fullscreen copy source");
            t.Equals(displayFbp, 150u,
                     "preferred-source presentation should still target the display page");
            t.Equals(sourceFbp, kSourceTbp0,
                     "preferred-source presentation should report the CT32 source frame");

            const size_t row1Off = 640u * 4u;
            t.Equals(static_cast<uint32_t>(latchedFrame[row1Off + 0u]), 0x44u,
                     "preferred-source presentation should read row 1 red from the swizzled CT32 source");
            t.Equals(static_cast<uint32_t>(latchedFrame[row1Off + 1u]), 0x55u,
                     "preferred-source presentation should read row 1 green from the swizzled CT32 source");
            t.Equals(static_cast<uint32_t>(latchedFrame[row1Off + 2u]), 0x66u,
                     "preferred-source presentation should read row 1 blue from the swizzled CT32 source");
            t.Equals(static_cast<uint32_t>(latchedFrame[row1Off + 3u]), 0xFFu,
                     "preferred-source presentation should normalize row 1 alpha for the host frame");
        });

        tc.Run("latched host presentation reads direct CT32 display frames with GS swizzle", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GSRegisters regs{};
            regs.pmode = 0x0001ull;
            regs.dispfb1 =
                150ull |
                (10ull << 9) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 15);
            regs.display1 =
                (639ull << 32) |
                (447ull << 44);

            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), &regs);

            constexpr uint32_t kRow1Pixel =
                0x44u |
                (0x55u << 8) |
                (0x66u << 16) |
                (0x77u << 24);
            constexpr uint32_t kLinearGarbageRow1 =
                0xAAu |
                (0xBBu << 8) |
                (0xCCu << 16) |
                (0xDDu << 24);
            constexpr size_t kHostRow1Off = 640u * 4u;

            writeReferenceFramePSMCT32Pixel(vram, 150u, 10u, 0u, 1u, kRow1Pixel);
            std::memcpy(vram.data() + (150u * 8192u) + kHostRow1Off, &kLinearGarbageRow1, sizeof(kLinearGarbageRow1));

            gs.latchHostPresentationFrame();

            std::vector<uint8_t> latchedFrame;
            uint32_t latchedWidth = 0u;
            uint32_t latchedHeight = 0u;
            uint32_t displayFbp = 0u;
            bool usedPreferred = false;
            t.IsTrue(gs.copyLatchedHostPresentationFrame(latchedFrame,
                                                         latchedWidth,
                                                         latchedHeight,
                                                         &displayFbp,
                                                         nullptr,
                                                         &usedPreferred),
                     "direct CT32 display presentation should produce a host frame");
            t.Equals(displayFbp, 150u,
                     "direct CT32 presentation should report the active display page");
            t.IsFalse(usedPreferred,
                      "direct CT32 presentation should not claim it used a preferred copy source");
            t.Equals(static_cast<uint32_t>(latchedFrame[kHostRow1Off + 0u]), 0x44u,
                     "direct CT32 presentation should read row 1 red from the GS-swizzled display page");
            t.Equals(static_cast<uint32_t>(latchedFrame[kHostRow1Off + 1u]), 0x55u,
                     "direct CT32 presentation should read row 1 green from the GS-swizzled display page");
            t.Equals(static_cast<uint32_t>(latchedFrame[kHostRow1Off + 2u]), 0x66u,
                     "direct CT32 presentation should read row 1 blue from the GS-swizzled display page");
            t.Equals(static_cast<uint32_t>(latchedFrame[kHostRow1Off + 3u]), 0xFFu,
                     "direct CT32 presentation should normalize row 1 alpha for the host frame");
        });

        tc.Run("latched host presentation merges both enabled PMODE circuits", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GSRegisters regs{};
            regs.pmode = 0x8007ull;
            regs.dispfb1 =
                150ull |
                (10ull << 9) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 15);
            regs.display1 =
                (639ull << 32) |
                (447ull << 44);
            regs.dispfb2 =
                0ull |
                (10ull << 9) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 15);
            regs.display2 = regs.display1;

            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), &regs);

            constexpr uint32_t kCircuit1Pixel =
                200u |
                (0u << 8) |
                (0u << 16) |
                (64u << 24);
            constexpr uint32_t kCircuit2Pixel =
                0u |
                (0u << 8) |
                (200u << 16) |
                (255u << 24);

            writeReferenceFramePSMCT32Pixel(vram, 150u, 10u, 0u, 0u, kCircuit1Pixel);
            std::memcpy(vram.data(), &kCircuit2Pixel, sizeof(kCircuit2Pixel));

            gs.latchHostPresentationFrame();

            std::vector<uint8_t> latchedFrame;
            uint32_t latchedWidth = 0u;
            uint32_t latchedHeight = 0u;
            uint32_t displayFbp = 0u;
            bool usedPreferred = false;
            t.IsTrue(gs.copyLatchedHostPresentationFrame(latchedFrame,
                                                         latchedWidth,
                                                         latchedHeight,
                                                         &displayFbp,
                                                         nullptr,
                                                         &usedPreferred),
                     "dual-circuit PMODE presentation should produce a host frame");
            t.Equals(displayFbp, 150u,
                     "dual-circuit presentation should still report the primary display page");
            t.IsFalse(usedPreferred,
                      "dual-circuit PMODE presentation should not bypass the first circuit with the preferred-copy shortcut");
            t.Equals(latchedWidth, 640u,
                     "dual-circuit presentation should preserve the display width");
            t.Equals(latchedHeight, 448u,
                     "dual-circuit presentation should preserve the display height");
            t.Equals(static_cast<uint32_t>(latchedFrame[0]), 100u,
                     "dual-circuit presentation should blend the first circuit red channel over the second circuit");
            t.Equals(static_cast<uint32_t>(latchedFrame[1]), 0u,
                     "dual-circuit presentation should preserve a zero green channel");
            t.Equals(static_cast<uint32_t>(latchedFrame[2]), 100u,
                     "dual-circuit presentation should blend the second circuit blue channel under the first circuit");
            t.Equals(static_cast<uint32_t>(latchedFrame[3]), 0xFFu,
                     "dual-circuit presentation should normalize the final host alpha");
        });

        tc.Run("latched host presentation normalizes alpha for single-circuit display", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GSRegisters regs{};
            regs.pmode = 0x0001ull;
            regs.dispfb1 =
                150ull |
                (10ull << 9) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 15);
            regs.display1 =
                (639ull << 32) |
                (447ull << 44);

            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), &regs);

            constexpr uint32_t kPixel =
                0x22u |
                (0x44u << 8) |
                (0x66u << 16) |
                (0x01u << 24);
            writeReferenceFramePSMCT32Pixel(vram, 150u, 10u, 0u, 0u, kPixel);

            gs.latchHostPresentationFrame();

            std::vector<uint8_t> latchedFrame;
            uint32_t latchedWidth = 0u;
            uint32_t latchedHeight = 0u;
            t.IsTrue(gs.copyLatchedHostPresentationFrame(latchedFrame, latchedWidth, latchedHeight),
                     "single-circuit presentation should produce a host frame");
            t.Equals(static_cast<uint32_t>(latchedFrame[0]), 0x22u,
                     "single-circuit presentation should preserve the red channel");
            t.Equals(static_cast<uint32_t>(latchedFrame[1]), 0x44u,
                     "single-circuit presentation should preserve the green channel");
            t.Equals(static_cast<uint32_t>(latchedFrame[2]), 0x66u,
                     "single-circuit presentation should preserve the blue channel");
            t.Equals(static_cast<uint32_t>(latchedFrame[3]), 0xFFu,
                     "single-circuit presentation should upload an opaque host alpha");
        });

        tc.Run("latched host presentation preserves 480-line display height", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GSRegisters regs{};
            regs.pmode = 0x0001ull;
            regs.dispfb1 =
                150ull |
                (10ull << 9) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 15);
            regs.display1 =
                (639ull << 32) |
                (479ull << 44);

            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), &regs);

            constexpr uint32_t kLastRowPixel =
                0x12u |
                (0x34u << 8) |
                (0x56u << 16) |
                (0x78u << 24);
            writeReferenceFramePSMCT32Pixel(vram, 150u, 10u, 0u, 479u, kLastRowPixel);

            gs.latchHostPresentationFrame();

            std::vector<uint8_t> latchedFrame;
            uint32_t latchedWidth = 0u;
            uint32_t latchedHeight = 0u;
            t.IsTrue(gs.copyLatchedHostPresentationFrame(latchedFrame, latchedWidth, latchedHeight),
                     "480-line single-circuit presentation should produce a host frame");
            t.Equals(latchedWidth, 640u,
                     "480-line presentation should preserve the display width");
            t.Equals(latchedHeight, 480u,
                     "480-line presentation should preserve the full display height");

            const size_t lastRowOffset = static_cast<size_t>(479u) * 640u * 4u;
            t.Equals(static_cast<uint32_t>(latchedFrame[lastRowOffset + 0u]), 0x12u,
                     "480-line presentation should keep the last row red channel");
            t.Equals(static_cast<uint32_t>(latchedFrame[lastRowOffset + 1u]), 0x34u,
                     "480-line presentation should keep the last row green channel");
            t.Equals(static_cast<uint32_t>(latchedFrame[lastRowOffset + 2u]), 0x56u,
                     "480-line presentation should keep the last row blue channel");
            t.Equals(static_cast<uint32_t>(latchedFrame[lastRowOffset + 3u]), 0xFFu,
                     "single-circuit presentation should normalize the last row alpha");
        });

        tc.Run("latched host presentation preserves both interlaced field scanlines", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GSRegisters regs{};
            regs.pmode = 0x0001ull;
            regs.smode2 = 0x0001ull; // interlaced, field mode
            regs.dispfb1 =
                (10ull << 9) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 15);
            regs.display1 =
                (639ull << 32) |
                (447ull << 44);

            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), &regs);

            constexpr uint32_t kLine0 = 0x000000FFu;
            constexpr uint32_t kLine1 = 0x0000FF00u;
            constexpr uint32_t kLine2 = 0x00FF0000u;
            constexpr uint32_t kLine3 = 0x00FFFF00u;
            writeReferenceFramePSMCT32Pixel(vram, 0u, 10u, 0u, 0u, kLine0);
            writeReferenceFramePSMCT32Pixel(vram, 0u, 10u, 0u, 1u, kLine1);
            writeReferenceFramePSMCT32Pixel(vram, 0u, 10u, 0u, 2u, kLine2);
            writeReferenceFramePSMCT32Pixel(vram, 0u, 10u, 0u, 3u, kLine3);

            gs.latchHostPresentationFrame();

            std::vector<uint8_t> latchedFrame;
            uint32_t latchedWidth = 0u;
            uint32_t latchedHeight = 0u;
            t.IsTrue(gs.copyLatchedHostPresentationFrame(latchedFrame, latchedWidth, latchedHeight),
                     "interlaced field presentation should produce a host frame");
            t.Equals(latchedWidth, 640u,
                     "field presentation should preserve display width");
            t.Equals(latchedHeight, 448u,
                     "field presentation should preserve display height");

            auto pixelAtRow = [&](uint32_t row) -> uint32_t
            {
                const size_t off = static_cast<size_t>(row) * latchedWidth * 4u;
                return static_cast<uint32_t>(latchedFrame[off + 0u]) |
                       (static_cast<uint32_t>(latchedFrame[off + 1u]) << 8) |
                       (static_cast<uint32_t>(latchedFrame[off + 2u]) << 16);
            };

            t.Equals(pixelAtRow(0u), kLine0,
                     "field presentation should retain the first even scanline");
            t.Equals(pixelAtRow(1u), kLine1,
                     "field presentation should retain the first odd scanline");
            t.Equals(pixelAtRow(2u), kLine2,
                     "field presentation should retain the second even scanline");
            t.Equals(pixelAtRow(3u), kLine3,
                     "field presentation should retain the second odd scanline");
        });

        tc.Run("latched host presentation applies DISPLAY vertical magnification", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GSRegisters regs{};
            regs.pmode = 0x0001ull;
            regs.dispfb1 =
                (10ull << 9) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 15);
            regs.display1 =
                (1ull << 27) |
                (639ull << 32) |
                (127ull << 44);

            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), &regs);

            constexpr uint32_t kLastVisiblePixel = 0x00665544u;
            writeReferenceFramePSMCT32Pixel(
                vram, 0u, 10u, 0u, 63u, kLastVisiblePixel);

            gs.latchHostPresentationFrame();

            std::vector<uint8_t> latchedFrame;
            uint32_t latchedWidth = 0u;
            uint32_t latchedHeight = 0u;
            t.IsTrue(gs.copyLatchedHostPresentationFrame(
                         latchedFrame, latchedWidth, latchedHeight),
                     "vertically magnified presentation should produce a host frame");
            t.Equals(latchedWidth, 640u,
                     "vertical magnification should not change source width");
            t.Equals(latchedHeight, 64u,
                     "MAGV=1 should reduce 128 display lines to 64 source rows");
            t.Equals(latchedFrame.size(), static_cast<size_t>(640u * 64u * 4u),
                     "vertically magnified presentation should stay tightly packed");

            uint32_t lastPixel = 0u;
            std::memcpy(
                &lastPixel,
                latchedFrame.data() +
                    (static_cast<size_t>(latchedHeight - 1u) * latchedWidth * 4u),
                sizeof(lastPixel));
            t.Equals(lastPixel, kLastVisiblePixel | 0xFF000000u,
                     "vertical magnification should retain the last source row");
        });

        tc.Run("latched host presentation supports display widths above 640 pixels", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GSRegisters regs{};
            regs.pmode = 0x0001ull;
            regs.dispfb1 =
                (13ull << 9) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 15);
            regs.display1 =
                (799ull << 32) |
                (63ull << 44);

            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), &regs);

            constexpr uint32_t kRightmostPixel = 0x00332211u;
            writeReferenceFramePSMCT32Pixel(
                vram, 0u, 13u, 799u, 0u, kRightmostPixel);

            gs.latchHostPresentationFrame();

            std::vector<uint8_t> latchedFrame;
            uint32_t latchedWidth = 0u;
            uint32_t latchedHeight = 0u;
            t.IsTrue(gs.copyLatchedHostPresentationFrame(
                         latchedFrame, latchedWidth, latchedHeight),
                     "wide presentation should produce a host frame");
            t.Equals(latchedWidth, 800u,
                     "DISPLAY should preserve source widths above the NTSC maximum");
            t.Equals(latchedHeight, 64u,
                     "wide presentation should preserve display height");
            t.Equals(latchedFrame.size(), static_cast<size_t>(800u * 64u * 4u),
                     "wide presentation should stay tightly packed");

            uint32_t rightmostPixel = 0u;
            std::memcpy(
                &rightmostPixel,
                latchedFrame.data() + (799u * 4u),
                sizeof(rightmostPixel));
            t.Equals(rightmostPixel, kRightmostPixel | 0xFF000000u,
                     "wide presentation should retain pixels beyond column 639");
        });

        tc.Run("latched host presentation halves interlaced frame-mode source height", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GSRegisters regs{};
            regs.pmode = 0x0001ull;
            regs.smode2 = 0x0003ull; // interlaced, frame mode
            regs.dispfb1 =
                (10ull << 9) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 15);
            regs.display1 =
                (639ull << 32) |
                (447ull << 44);

            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), &regs);

            constexpr uint32_t kLastFieldPixel = 0x00654321u;
            writeReferenceFramePSMCT32Pixel(
                vram, 0u, 10u, 0u, 223u, kLastFieldPixel);

            gs.latchHostPresentationFrame();

            std::vector<uint8_t> latchedFrame;
            uint32_t latchedWidth = 0u;
            uint32_t latchedHeight = 0u;
            t.IsTrue(gs.copyLatchedHostPresentationFrame(
                         latchedFrame, latchedWidth, latchedHeight),
                     "interlaced frame mode should produce a host field");
            t.Equals(latchedWidth, 640u,
                     "interlaced frame mode should preserve source width");
            t.Equals(latchedHeight, 224u,
                     "interlaced frame mode should read half the display height");

            uint32_t lastPixel = 0u;
            std::memcpy(
                &lastPixel,
                latchedFrame.data() +
                    (static_cast<size_t>(latchedHeight - 1u) * latchedWidth * 4u),
                sizeof(lastPixel));
            t.Equals(lastPixel, kLastFieldPixel | 0xFF000000u,
                     "interlaced frame mode should retain the last field row");
        });

        tc.Run("GIF PACKED A+D ignores reserved register addresses", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GSRegisters regs{};
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), &regs);

            regs.dispfb1 = 0x0123456789ABCDEFull;
            regs.display1 = 0x1111222233334444ull;
            regs.dispfb2 = 0x2222333344445555ull;
            regs.display2 = 0x6666777788889999ull;
            regs.bgcolor = 0xAAAABBBBCCCCDDDDull;

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(5u, GIF_FMT_PACKED, 1u, true));
            appendU64(packet, 0x0Eull); // REGS[0] = A+D

            appendU64(packet, 0x1000000000000059ull);
            appendU64(packet, 0x59ull);
            appendU64(packet, 0x200000000000005Aull);
            appendU64(packet, 0x5Aull);
            appendU64(packet, 0x300000000000005Bull);
            appendU64(packet, 0x5Bull);
            appendU64(packet, 0x400000000000005Cull);
            appendU64(packet, 0x5Cull);
            appendU64(packet, 0x500000000000005Full);
            appendU64(packet, 0x5Full);

            gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));

            t.Equals(regs.dispfb1, 0x0123456789ABCDEFull,
                     "reserved A+D address 0x59 should not change DISPFB1");
            t.Equals(regs.display1, 0x1111222233334444ull,
                     "reserved A+D address 0x5A should not change DISPLAY1");
            t.Equals(regs.dispfb2, 0x2222333344445555ull,
                     "reserved A+D address 0x5B should not change DISPFB2");
            t.Equals(regs.display2, 0x6666777788889999ull,
                     "reserved A+D address 0x5C should not change DISPLAY2");
            t.Equals(regs.bgcolor, 0xAAAABBBBCCCCDDDDull,
                     "reserved A+D address 0x5F should not change BGCOLOR");
        });

        tc.Run("PSMT4 address mapping matches GS manual layout", [](TestCase &t)
        {
            constexpr uint32_t kBaseBlock = 0u;
            constexpr uint32_t kWidth = 2u; // One 128x128 PSMT4 page.

            t.Equals(GSPSMT4::addrPSMT4(kBaseBlock, kWidth, 0u, 0u), 0u,
                     "PSMT4 origin should map to nibble offset 0");
            t.Equals(GSPSMT4::addrPSMT4(kBaseBlock, kWidth, 1u, 0u), 8u,
                     "PSMT4 x=1 should advance to the next packed nibble group");
            t.Equals(GSPSMT4::addrPSMT4(kBaseBlock, kWidth, 0u, 1u), 16u,
                     "PSMT4 second source row should follow the manual's row packing");
            t.Equals(GSPSMT4::addrPSMT4(kBaseBlock, kWidth, 0u, 2u), 65u,
                     "PSMT4 third source row should include the manual's odd-row permutation");
            t.Equals(GSPSMT4::addrPSMT4(kBaseBlock, kWidth, 0u, 3u), 81u,
                     "PSMT4 fourth source row should stay in the first block's manual column layout");
            t.Equals(GSPSMT4::addrPSMT4(kBaseBlock, kWidth, 31u, 15u), 511u,
                     "PSMT4 final texel in the first 32x16 block should land at the end of the block");
            t.Equals(GSPSMT4::addrPSMT4(kBaseBlock, kWidth, 32u, 0u), 1024u,
                     "PSMT4 x=32 should advance to the next swizzled block in the page");
            t.Equals(GSPSMT4::addrPSMT4(kBaseBlock, kWidth, 32u, 16u), 1536u,
                     "PSMT4 x=32,y=16 should follow the manual's second block-row permutation");
            t.Equals(GSPSMT4::addrPSMT4(kBaseBlock, kWidth, 64u, 0u), 4096u,
                     "PSMT4 x=64 should advance to the third swizzled block column in the page");
            t.Equals(GSPSMT4::addrPSMT4(kBaseBlock, kWidth, 96u, 112u), 15872u,
                     "PSMT4 bottom-right block origin should match the manual's page permutation");
            t.Equals(GSPSMT4::addrPSMT4(kBaseBlock, kWidth, 127u, 127u), 16383u,
                     "PSMT4 final texel in a 128x128 page should land at the end of the page");
            t.Equals(GSPSMT4::addrPSMT4(kBaseBlock, kWidth, 128u, 0u), 16384u,
                     "PSMT4 x=128 should advance to the next page of nibble addresses");
        });

        tc.Run("PSMT4 large atlases keep manual page layout across 512x512 textures", [](TestCase &t)
        {
            constexpr uint32_t kBaseBlock = 64u;
            constexpr uint32_t kWidth = 8u; // 512 pixel-wide T4 atlas, like Veronica font pages.
            constexpr uint32_t kCoords[][2] = {
                {0u, 0u},
                {31u, 15u},
                {32u, 0u},
                {95u, 31u},
                {127u, 127u},
                {128u, 0u},
                {255u, 127u},
                {256u, 0u},
                {383u, 127u},
                {384u, 128u},
                {511u, 511u},
            };

            for (const auto &coord : kCoords)
            {
                const uint32_t x = coord[0];
                const uint32_t y = coord[1];
                t.Equals(GSPSMT4::addrPSMT4(kBaseBlock, kWidth, x, y),
                         referenceAddrPSMT4(kBaseBlock, kWidth, x, y),
                         "PSMT4 512x512 atlas mapping should match the GS manual for every sampled page boundary");
            }
        });

        tc.Run("GS T4 triangle sampling reads manual-layout texels from a 512x512 atlas", [](TestCase &t)
        {
            constexpr uint32_t kTexTbp = 64u;
            constexpr uint32_t kClutCbp = 128u;
            constexpr uint64_t kFrame =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (4ull << 16) |
                (0ull << 32) |
                (4ull << 48);
            constexpr uint64_t kTex0 =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (8ull << 14) |
                (static_cast<uint64_t>(GS_PSM_T4) << 20) |
                (9ull << 26) |
                (9ull << 30) |
                (1ull << 34) |
                (1ull << 35) |
                (static_cast<uint64_t>(kClutCbp) << 37) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 51) |
                (1ull << 55) |
                (1ull << 61);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_TRIANGLE) |
                (1ull << 4);
            constexpr uint64_t kRgbaq = 0x3F80000080808080ull;

            auto packFloat = [](float value) -> uint32_t
            {
                uint32_t bits = 0u;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            };

            auto packSt = [&](float s, float tVal) -> uint64_t
            {
                return static_cast<uint64_t>(packFloat(s)) |
                       (static_cast<uint64_t>(packFloat(tVal)) << 32);
            };

            const struct SampleCase
            {
                uint32_t x;
                uint32_t y;
                uint8_t index;
                uint32_t color;
            } cases[] = {
                {5u, 5u, 1u, 0xFF0000FFu},
                {129u, 5u, 2u, 0xFF00FF00u},
                {257u, 5u, 3u, 0xFFFF0000u},
                {385u, 129u, 4u, 0xFFFFFFFFu},
            };

            for (const auto &sample : cases)
            {
                std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
                GS gs;
                gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

                writeReferencePSMT4Texel(vram, kTexTbp, 8u, sample.x, sample.y, sample.index);
                const uint32_t clutOff = referenceAddrPSMCT32(kClutCbp, 1u, sample.index, 0u);
                std::memcpy(vram.data() + clutOff, &sample.color, sizeof(sample.color));

                gs.writeRegister(GS_REG_FRAME_1, kFrame);
                gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
                gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
                gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
                gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
                gs.writeRegister(GS_REG_ALPHA_1, 0ull);
                gs.writeRegister(GS_REG_TEX0_1, kTex0);
                gs.writeRegister(GS_REG_TEX1_1, 0ull);
                gs.writeRegister(GS_REG_PRIM, kPrim);
                gs.writeRegister(GS_REG_RGBAQ, kRgbaq);

                const float s = (static_cast<float>(sample.x) + 0.25f) / 512.0f;
                const float tVal = (static_cast<float>(sample.y) + 0.25f) / 512.0f;
                gs.writeRegister(GS_REG_ST, packSt(s, tVal));
                gs.writeRegister(GS_REG_XYZ2, 0ull);
                gs.writeRegister(GS_REG_ST, packSt(s, tVal));
                gs.writeRegister(GS_REG_XYZ2, (64ull << 0) | (0ull << 16));
                gs.writeRegister(GS_REG_ST, packSt(s, tVal));
                gs.writeRegister(GS_REG_XYZ2, (0ull << 0) | (64ull << 16));

                const uint32_t pixel = readReferencePSMCT32Pixel(vram, 0u, 1u, 1u, 1u);
                t.Equals(pixel, sample.color,
                         "T4 triangle sampling should fetch the manual-layout atlas texel from the correct 128x128 page");
            }
        });

        tc.Run("PSMT8 address mapping matches Veronica Conv8to32 layout", [](TestCase &t)
        {
            constexpr uint32_t kBaseBlock = 0u;
            constexpr uint32_t kWidth = 2u; // One 128x64 PSMT8 page.

            t.Equals(GSPSMT8::addrPSMT8(kBaseBlock, kWidth, 0u, 0u), 0u,
                     "PSMT8 origin should map to byte offset 0");
            t.Equals(GSPSMT8::addrPSMT8(kBaseBlock, kWidth, 1u, 0u), 4u,
                     "PSMT8 x=1 should follow Veronica's Conv8to32 byte interleave");
            t.Equals(GSPSMT8::addrPSMT8(kBaseBlock, kWidth, 0u, 1u), 8u,
                     "PSMT8 second source row should land on the next Conv8to32 row stride");
            t.Equals(GSPSMT8::addrPSMT8(kBaseBlock, kWidth, 0u, 2u), 33u,
                     "PSMT8 third source row should preserve Veronica's odd-row shuffle");
            t.Equals(GSPSMT8::addrPSMT8(kBaseBlock, kWidth, 0u, 3u), 41u,
                     "PSMT8 fourth source row should preserve Veronica's alternating block rows");
            t.Equals(GSPSMT8::addrPSMT8(kBaseBlock, kWidth, 15u, 15u), 255u,
                     "PSMT8 final texel in the first 16x16 block should end at byte 255");
            t.Equals(GSPSMT8::addrPSMT8(kBaseBlock, kWidth, 16u, 0u), 256u,
                     "PSMT8 x=16 should advance to the next 16x16 block");
            t.Equals(GSPSMT8::addrPSMT8(kBaseBlock, kWidth, 16u, 16u), 768u,
                     "PSMT8 x=16,y=16 should include both block-column and block-row offsets");
            t.Equals(GSPSMT8::addrPSMT8(kBaseBlock, kWidth, 32u, 0u), 1024u,
                     "PSMT8 x=32 should advance to the third block column in the page");
            t.Equals(GSPSMT8::addrPSMT8(kBaseBlock, kWidth, 64u, 0u), 4096u,
                     "PSMT8 x=64 should advance to the second page half");
            t.Equals(GSPSMT8::addrPSMT8(kBaseBlock, kWidth, 96u, 48u), 7680u,
                     "PSMT8 lower-right interior block should follow Veronica's page permutation");
            t.Equals(GSPSMT8::addrPSMT8(kBaseBlock, kWidth, 127u, 63u), 8191u,
                     "PSMT8 final texel in a 128x64 page should land at the final byte");
        });

        tc.Run("GIF REGLIST with odd register count consumes 128-bit padding before next tag", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            const uint64_t bitblt =
                (static_cast<uint64_t>(0u) << 0) |
                (static_cast<uint64_t>(1u) << 16) |
                (static_cast<uint64_t>(0u) << 24) |
                (static_cast<uint64_t>(0u) << 32) |
                (static_cast<uint64_t>(1u) << 48) |
                (static_cast<uint64_t>(0u) << 56);
            gs.writeRegister(GS_REG_BITBLTBUF, bitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, (4ull << 0) | (1ull << 32));
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(1u, GIF_FMT_REGLIST, 1u, false));
            appendU64(packet, 0x0ull); // REGS[0] = PRIM
            appendU64(packet, 0x0000000000000006ull); // PRIM write
            appendU64(packet, 0xDEADBEEFCAFEBABEull); // required REGLIST pad qword

            appendU64(packet, makeGifTag(1u, GIF_FMT_IMAGE, 0u, true));
            appendU64(packet, 0ull);
            const uint8_t payload[16] = {
                0x31u, 0x32u, 0x33u, 0x34u,
                0x35u, 0x36u, 0x37u, 0x38u,
                0x39u, 0x3Au, 0x3Bu, 0x3Cu,
                0x3Du, 0x3Eu, 0x3Fu, 0x40u,
            };
            packet.insert(packet.end(), payload, payload + sizeof(payload));

            gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));

            bool imageOk = true;
            for (uint32_t x = 0; x < 4u && imageOk; ++x)
            {
                const uint32_t off = referenceAddrPSMCT32(0u, 1u, x, 0u);
                for (uint32_t c = 0; c < 4u; ++c)
                {
                    if (vram[off + c] != payload[x * 4u + c])
                    {
                        imageOk = false;
                        break;
                    }
                }
            }
            t.IsTrue(imageOk, "odd REGLIST payload should not corrupt alignment of the following IMAGE tag");
        });

        tc.Run("GIF REGLIST NREG=0 is treated as sixteen descriptors", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            const uint64_t bitblt =
                (static_cast<uint64_t>(0u) << 0) |
                (static_cast<uint64_t>(1u) << 16) |
                (static_cast<uint64_t>(0u) << 24) |
                (static_cast<uint64_t>(0u) << 32) |
                (static_cast<uint64_t>(1u) << 48) |
                (static_cast<uint64_t>(0u) << 56);
            gs.writeRegister(GS_REG_BITBLTBUF, bitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, (4ull << 0) | (1ull << 32));
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(1u, GIF_FMT_REGLIST, 0u, false)); // NREG=0 -> 16 regs
            appendU64(packet, 0ull); // 16x PRIM descriptors
            for (uint32_t i = 0; i < 16u; ++i)
            {
                appendU64(packet, static_cast<uint64_t>(i));
            }

            appendU64(packet, makeGifTag(1u, GIF_FMT_IMAGE, 0u, true));
            appendU64(packet, 0ull);
            const uint8_t payload[16] = {
                0x51u, 0x52u, 0x53u, 0x54u,
                0x55u, 0x56u, 0x57u, 0x58u,
                0x59u, 0x5Au, 0x5Bu, 0x5Cu,
                0x5Du, 0x5Eu, 0x5Fu, 0x60u,
            };
            packet.insert(packet.end(), payload, payload + sizeof(payload));

            gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));

            bool imageOk = true;
            for (uint32_t x = 0; x < 4u && imageOk; ++x)
            {
                const uint32_t off = referenceAddrPSMCT32(0u, 1u, x, 0u);
                for (uint32_t c = 0; c < 4u; ++c)
                {
                    if (vram[off + c] != payload[x * 4u + c])
                    {
                        imageOk = false;
                        break;
                    }
                }
            }
            t.IsTrue(imageOk, "NREG=0 REGLIST should consume 16 data words and keep following tag aligned");
        });

        tc.Run("GS SIGNAL and FINISH set CSR bits that clear by CSR write-one acknowledge", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            GS gs;
            gs.init(mem.getGSVRAM(), static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &mem.gs());

            const uint64_t signalValue = (0xFFFFFFFFull << 32) | 0x11223344ull;
            gs.writeRegister(GS_REG_SIGNAL, signalValue);
            gs.writeRegister(GS_REG_FINISH, 0u);

            t.IsTrue((mem.gs().csr & 0x1ull) != 0ull, "SIGNAL should raise CSR.SIGNAL");
            t.IsTrue((mem.gs().csr & 0x2ull) != 0ull, "FINISH should raise CSR.FINISH");
            t.Equals(static_cast<uint32_t>(mem.gs().siglblid & 0xFFFFFFFFull), 0x11223344u, "SIGNAL should update SIGLBLID low dword");

            mem.write64(0x12001000u, 0x1ull);
            t.IsTrue((mem.gs().csr & 0x1ull) == 0ull, "writing CSR bit0 should acknowledge SIGNAL");
            t.IsTrue((mem.gs().csr & 0x2ull) != 0ull, "acknowledging SIGNAL should not clear FINISH");

            mem.write32(0x12001000u, 0x2u);
            t.IsTrue((mem.gs().csr & 0x2ull) == 0ull, "writing CSR bit1 should acknowledge FINISH");
        });

        tc.Run("GIF IMAGE packet writes host-to-local data into GS VRAM", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            // Setup for host->local transfer to DBP=0, DBW=1, PSMCT32, rect 2x2.
            const uint64_t bitblt =
                (static_cast<uint64_t>(0u) << 0) |      // SBP
                (static_cast<uint64_t>(1u) << 16) |     // SBW
                (static_cast<uint64_t>(0u) << 24) |     // SPSM
                (static_cast<uint64_t>(0u) << 32) |     // DBP
                (static_cast<uint64_t>(1u) << 48) |     // DBW
                (static_cast<uint64_t>(0u) << 56);      // DPSM (CT32)
            gs.writeRegister(GS_REG_BITBLTBUF, bitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, (2ull << 0) | (2ull << 32));
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(1u, GIF_FMT_IMAGE, 0u, true));
            appendU64(packet, 0ull);

            const uint8_t payload[16] = {
                0x10u, 0x11u, 0x12u, 0x13u,
                0x20u, 0x21u, 0x22u, 0x23u,
                0x30u, 0x31u, 0x32u, 0x33u,
                0x40u, 0x41u, 0x42u, 0x43u,
            };
            packet.insert(packet.end(), payload, payload + sizeof(payload));

            gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));

            bool same = true;
            for (uint32_t y = 0; y < 2u && same; ++y)
            {
                for (uint32_t x = 0; x < 2u; ++x)
                {
                    const uint32_t pixelIndex = y * 2u + x;
                    const uint32_t off = referenceAddrPSMCT32(0u, 1u, x, y);
                    for (uint32_t c = 0; c < 4u; ++c)
                    {
                        if (vram[off + c] != payload[pixelIndex * 4u + c])
                        {
                            same = false;
                            break;
                        }
                    }
                    if (!same)
                        break;
                }
            }
            t.IsTrue(same, "GIF IMAGE transfer should write payload bytes into GS VRAM");
        });

        tc.Run("GIF load-image packet uses native upload fast path", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            const uint64_t bitblt =
                (static_cast<uint64_t>(0u) << 0) |
                (static_cast<uint64_t>(1u) << 16) |
                (static_cast<uint64_t>(0u) << 24) |
                (static_cast<uint64_t>(0u) << 32) |
                (static_cast<uint64_t>(1u) << 48) |
                (static_cast<uint64_t>(0u) << 56);
            const uint64_t trxpos = 0ull;
            const uint64_t trxreg = (2ull << 0) | (2ull << 32);
            const uint64_t trxdir = 0ull;

            const uint8_t payload[16] = {
                0x10u, 0x11u, 0x12u, 0x13u,
                0x20u, 0x21u, 0x22u, 0x23u,
                0x30u, 0x31u, 0x32u, 0x33u,
                0x40u, 0x41u, 0x42u, 0x43u,
            };

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(4u, GIF_FMT_PACKED, 1u, false));
            appendU64(packet, 0x0Eull);
            appendGifAd(packet, bitblt, GS_REG_BITBLTBUF);
            appendGifAd(packet, trxpos, GS_REG_TRXPOS);
            appendGifAd(packet, trxreg, GS_REG_TRXREG);
            appendGifAd(packet, trxdir, GS_REG_TRXDIR);
            appendU64(packet, makeGifTag(1u, GIF_FMT_IMAGE, 0u, true));
            appendU64(packet, 0ull);
            packet.insert(packet.end(), payload, payload + sizeof(payload));

            gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));

            t.Equals(gs.nativeImageUploadCount(), 1ull, "load-image packet should use the native image upload fast path");

            bool same = true;
            for (uint32_t y = 0; y < 2u && same; ++y)
            {
                for (uint32_t x = 0; x < 2u; ++x)
                {
                    const uint32_t pixelIndex = y * 2u + x;
                    const uint32_t off = referenceAddrPSMCT32(0u, 1u, x, y);
                    for (uint32_t c = 0; c < 4u; ++c)
                    {
                        if (vram[off + c] != payload[pixelIndex * 4u + c])
                        {
                            same = false;
                            break;
                        }
                    }
                    if (!same)
                        break;
                }
            }
            t.IsTrue(same, "native load-image upload should preserve pixel payload");
        });

        tc.Run("GS local-to-host transfer supports partial incremental reads", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            for (uint32_t x = 0; x < 4u; ++x)
            {
                const uint32_t off = referenceAddrPSMCT32(0u, 1u, x, 0u);
                for (uint32_t c = 0; c < 4u; ++c)
                {
                    vram[off + c] = static_cast<uint8_t>(0xA0u + x * 4u + c);
                }
            }

            const uint64_t bitblt =
                (static_cast<uint64_t>(0u) << 0) |      // SBP
                (static_cast<uint64_t>(1u) << 16) |     // SBW
                (static_cast<uint64_t>(0u) << 24) |     // SPSM (CT32)
                (static_cast<uint64_t>(0u) << 32) |
                (static_cast<uint64_t>(1u) << 48) |
                (static_cast<uint64_t>(0u) << 56);
            gs.writeRegister(GS_REG_BITBLTBUF, bitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, (4ull << 0) | (1ull << 32)); // 4 pixels, 1 row -> 16 bytes
            gs.writeRegister(GS_REG_TRXDIR, 1ull);

            uint8_t bufA[8] = {};
            uint8_t bufB[16] = {};

            const uint32_t nA = gs.consumeLocalToHostBytes(bufA, 6u);
            const uint32_t nB = gs.consumeLocalToHostBytes(bufB, 16u);
            const uint32_t nC = gs.consumeLocalToHostBytes(bufB, 4u);

            t.Equals(nA, 6u, "first partial read should consume requested bytes");
            t.Equals(nB, 10u, "second read should consume the remaining bytes");
            t.Equals(nC, 0u, "buffer should be empty after all bytes are consumed");

            bool bytesOk = true;
            for (uint32_t i = 0; i < 6u; ++i)
            {
                if (bufA[i] != static_cast<uint8_t>(0xA0u + i))
                    bytesOk = false;
            }
            for (uint32_t i = 0; i < 10u; ++i)
            {
                if (bufB[i] != static_cast<uint8_t>(0xA6u + i))
                    bytesOk = false;
            }
            t.IsTrue(bytesOk, "partial reads should return local->host data in-order");
        });

        tc.Run("GS CT24 host-local-host transfer preserves 24-bit RGB payload", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            const uint64_t bitblt =
                (static_cast<uint64_t>(0u) << 0) |      // SBP
                (static_cast<uint64_t>(1u) << 16) |     // SBW
                (static_cast<uint64_t>(1u) << 24) |     // SPSM CT24
                (static_cast<uint64_t>(0u) << 32) |     // DBP
                (static_cast<uint64_t>(1u) << 48) |     // DBW
                (static_cast<uint64_t>(1u) << 56);      // DPSM CT24
            gs.writeRegister(GS_REG_BITBLTBUF, bitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, (2ull << 0) | (1ull << 32)); // 2 pixels
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(1u, GIF_FMT_IMAGE, 0u, true));
            appendU64(packet, 0ull);
            const uint8_t rgbData[16] = {
                0x11u, 0x22u, 0x33u,
                0x44u, 0x55u, 0x66u,
                0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
            };
            packet.insert(packet.end(), rgbData, rgbData + sizeof(rgbData));
            gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));

            // Read back from local to host in CT24.
            gs.writeRegister(GS_REG_TRXDIR, 1ull);
            uint8_t out[16] = {};
            const uint32_t outBytes = gs.consumeLocalToHostBytes(out, sizeof(out));

            t.Equals(outBytes, 6u, "CT24 local->host read should output 3 bytes per pixel");
            t.Equals(out[0], static_cast<uint8_t>(0x11u), "pixel0 R should roundtrip");
            t.Equals(out[1], static_cast<uint8_t>(0x22u), "pixel0 G should roundtrip");
            t.Equals(out[2], static_cast<uint8_t>(0x33u), "pixel0 B should roundtrip");
            t.Equals(out[3], static_cast<uint8_t>(0x44u), "pixel1 R should roundtrip");
            t.Equals(out[4], static_cast<uint8_t>(0x55u), "pixel1 G should roundtrip");
            t.Equals(out[5], static_cast<uint8_t>(0x66u), "pixel1 B should roundtrip");
        });

        tc.Run("GS PSMT4 host-local-host keeps nibble packing stable", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            const uint64_t bitblt =
                (static_cast<uint64_t>(0u) << 0) |      // SBP
                (static_cast<uint64_t>(1u) << 16) |     // SBW
                (static_cast<uint64_t>(20u) << 24) |    // SPSM PSMT4
                (static_cast<uint64_t>(0u) << 32) |     // DBP
                (static_cast<uint64_t>(1u) << 48) |     // DBW
                (static_cast<uint64_t>(20u) << 56);     // DPSM PSMT4
            gs.writeRegister(GS_REG_BITBLTBUF, bitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, (4ull << 0) | (1ull << 32)); // 4 texels => 2 bytes
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(1u, GIF_FMT_IMAGE, 0u, true));
            appendU64(packet, 0ull);
            const uint8_t nibbleData[16] = {0x21u, 0x43u};
            packet.insert(packet.end(), nibbleData, nibbleData + sizeof(nibbleData));
            gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));

            gs.writeRegister(GS_REG_TRXDIR, 1ull);
            uint8_t out[8] = {};
            const uint32_t outBytes = gs.consumeLocalToHostBytes(out, sizeof(out));

            t.Equals(outBytes, 2u, "PSMT4 local->host should return packed nibble bytes");
            t.Equals(out[0], static_cast<uint8_t>(0x21u), "packed nibble byte 0 should roundtrip");
            t.Equals(out[1], static_cast<uint8_t>(0x43u), "packed nibble byte 1 should roundtrip");
        });

        tc.Run("GS PSMT4 host-local upload keeps position across split IMAGE packets", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            const uint64_t bitblt =
                (static_cast<uint64_t>(0u) << 0) |
                (static_cast<uint64_t>(1u) << 16) |
                (static_cast<uint64_t>(GS_PSM_T4) << 24) |
                (static_cast<uint64_t>(0u) << 32) |
                (static_cast<uint64_t>(1u) << 48) |
                (static_cast<uint64_t>(GS_PSM_T4) << 56);
            gs.writeRegister(GS_REG_BITBLTBUF, bitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, (8ull << 0) | (8ull << 32)); // 64 texels => 32 bytes
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            uint8_t packedSource[32] = {};
            for (uint32_t i = 0; i < 32u; ++i)
            {
                packedSource[i] = static_cast<uint8_t>(0x10u + i);
            }

            std::vector<uint8_t> packetA;
            appendU64(packetA, makeGifTag(1u, GIF_FMT_IMAGE, 0u, true));
            appendU64(packetA, 0ull);
            packetA.insert(packetA.end(), packedSource, packedSource + 16u);

            std::vector<uint8_t> packetB;
            appendU64(packetB, makeGifTag(1u, GIF_FMT_IMAGE, 0u, true));
            appendU64(packetB, 0ull);
            packetB.insert(packetB.end(), packedSource + 16u, packedSource + 32u);

            gs.processGIFPacket(packetA.data(), static_cast<uint32_t>(packetA.size()));
            gs.processGIFPacket(packetB.data(), static_cast<uint32_t>(packetB.size()));

            gs.writeRegister(GS_REG_TRXDIR, 1ull);
            uint8_t out[32] = {};
            const uint32_t outBytes = gs.consumeLocalToHostBytes(out, sizeof(out));

            t.Equals(outBytes, 32u, "split T4 IMAGE upload should fill the full packed byte range");
            for (uint32_t i = 0; i < 32u; ++i)
            {
                t.Equals(out[i], packedSource[i], "split T4 IMAGE upload should preserve packed nibble order");
            }
        });

        tc.Run("GS CT32 upload aliases cleanly into later PSMT8 sampling", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kTexWidth = 128u;
            constexpr uint32_t kTexHeight = 64u;
            constexpr uint32_t kUploadWidth = 64u;
            constexpr uint32_t kUploadHeight = 32u;
            constexpr uint32_t kTexTbp = 0u;
            constexpr uint32_t kTexTbw = 2u;

            std::vector<uint8_t> source(kTexWidth * kTexHeight, 0u);
            for (uint32_t i = 0; i < source.size(); ++i)
            {
                source[i] = static_cast<uint8_t>((i * 37u + 11u) & 0xFFu);
            }

            std::vector<uint16_t> rawToUpload(8192u, 0xFFFFu);
            for (uint32_t y = 0; y < kUploadHeight; ++y)
            {
                for (uint32_t x = 0; x < kUploadWidth; ++x)
                {
                    const uint32_t rawBase = referenceAddrPSMCT32(kTexTbp, 1u, x, y);
                    const uint32_t uploadBase = ((y * kUploadWidth) + x) * 4u;
                    for (uint32_t c = 0; c < 4u; ++c)
                    {
                        rawToUpload[rawBase + c] = static_cast<uint16_t>(uploadBase + c);
                    }
                }
            }

            bool inverseComplete = true;
            for (uint16_t byteOff : rawToUpload)
            {
                if (byteOff == 0xFFFFu)
                {
                    inverseComplete = false;
                    break;
                }
            }
            t.IsTrue(inverseComplete,
                     "reference CT32 raw-to-upload map should cover every byte in a 64x32 CT32 page");

            std::vector<uint8_t> upload(kUploadWidth * kUploadHeight * 4u, 0u);
            for (uint32_t y = 0; y < kTexHeight; ++y)
            {
                for (uint32_t x = 0; x < kTexWidth; ++x)
                {
                    const uint32_t texelIndex = y * kTexWidth + x;
                    const uint32_t rawOff = referenceAddrPSMT8(kTexTbp, kTexTbw, x, y);
                    const uint32_t uploadOff = rawToUpload[rawOff];
                    upload[uploadOff] = source[texelIndex];
                }
            }

            const uint64_t bitblt =
                (static_cast<uint64_t>(0u) << 0) |
                (static_cast<uint64_t>(1u) << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24) |
                (static_cast<uint64_t>(kTexTbp) << 32) |
                (static_cast<uint64_t>(1u) << 48) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 56);
            gs.writeRegister(GS_REG_BITBLTBUF, bitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, (static_cast<uint64_t>(kUploadWidth) << 0) |
                                            (static_cast<uint64_t>(kUploadHeight) << 32));
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(static_cast<uint16_t>(upload.size() / 16u), GIF_FMT_IMAGE, 0u, true));
            appendU64(packet, 0ull);
            packet.insert(packet.end(), upload.begin(), upload.end());
            gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));

            bool aliasOk = true;
            uint32_t badX = 0u;
            uint32_t badY = 0u;
            uint32_t got = 0u;
            uint32_t expected = 0u;
            for (uint32_t y = 0; y < kTexHeight && aliasOk; ++y)
            {
                for (uint32_t x = 0; x < kTexWidth; ++x)
                {
                    const uint32_t texelOff = GSPSMT8::addrPSMT8(kTexTbp, kTexTbw, x, y);
                    got = vram[texelOff];
                    expected = source[y * kTexWidth + x];
                    if (got != expected)
                    {
                        aliasOk = false;
                        badX = x;
                        badY = y;
                        break;
                    }
                }
            }

            if (!aliasOk)
            {
                t.Fail("CT32 image upload should preserve Veronica's later PSMT8 sampling layout "
                       "(first mismatch at x=" + std::to_string(badX) +
                       ", y=" + std::to_string(badY) +
                       ", got " + std::to_string(got) +
                       ", expected " + std::to_string(expected) + ")");
            }
        });

        tc.Run("GS PSMT4 local-local copy respects swizzled page layout", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kSrcBp = 64u;
            constexpr uint32_t kDstBp = 96u;
            constexpr uint64_t kUploadBitblt =
                (static_cast<uint64_t>(0u) << 0) |
                (static_cast<uint64_t>(2u) << 16) |
                (static_cast<uint64_t>(GS_PSM_T4) << 24) |
                (static_cast<uint64_t>(kSrcBp) << 32) |
                (static_cast<uint64_t>(2u) << 48) |
                (static_cast<uint64_t>(GS_PSM_T4) << 56);
            constexpr uint64_t kCopyBitblt =
                (static_cast<uint64_t>(kSrcBp) << 0) |
                (static_cast<uint64_t>(2u) << 16) |
                (static_cast<uint64_t>(GS_PSM_T4) << 24) |
                (static_cast<uint64_t>(kDstBp) << 32) |
                (static_cast<uint64_t>(2u) << 48) |
                (static_cast<uint64_t>(GS_PSM_T4) << 56);
            constexpr uint64_t kCopyPos =
                (static_cast<uint64_t>(0u) << 0) |
                (static_cast<uint64_t>(0u) << 16) |
                (static_cast<uint64_t>(32u) << 32) |
                (static_cast<uint64_t>(16u) << 48);
            constexpr uint64_t kReadBitblt =
                (static_cast<uint64_t>(kDstBp) << 0) |
                (static_cast<uint64_t>(2u) << 16) |
                (static_cast<uint64_t>(GS_PSM_T4) << 24) |
                (static_cast<uint64_t>(0u) << 32) |
                (static_cast<uint64_t>(2u) << 48) |
                (static_cast<uint64_t>(GS_PSM_T4) << 56);
            constexpr uint64_t kReadPos =
                (static_cast<uint64_t>(32u) << 0) |
                (static_cast<uint64_t>(16u) << 16);
            constexpr uint64_t kRect = (8ull << 0) | (4ull << 32);
            const uint8_t packedSource[16] = {
                0x10u, 0x32u, 0x54u, 0x76u,
                0x98u, 0xBAu, 0xDCu, 0xFEu,
                0x01u, 0x23u, 0x45u, 0x67u,
                0x89u, 0xABu, 0xCDu, 0xEFu
            };

            gs.writeRegister(GS_REG_BITBLTBUF, kUploadBitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, kRect);
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(1u, GIF_FMT_IMAGE, 0u, true));
            appendU64(packet, 0ull);
            packet.insert(packet.end(), packedSource, packedSource + sizeof(packedSource));
            gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));

            gs.writeRegister(GS_REG_BITBLTBUF, kCopyBitblt);
            gs.writeRegister(GS_REG_TRXPOS, kCopyPos);
            gs.writeRegister(GS_REG_TRXREG, kRect);
            gs.writeRegister(GS_REG_TRXDIR, 2ull);

            gs.writeRegister(GS_REG_BITBLTBUF, kReadBitblt);
            gs.writeRegister(GS_REG_TRXPOS, kReadPos);
            gs.writeRegister(GS_REG_TRXREG, kRect);
            gs.writeRegister(GS_REG_TRXDIR, 1ull);

            uint8_t out[16] = {};
            const uint32_t outBytes = gs.consumeLocalToHostBytes(out, sizeof(out));
            t.Equals(outBytes, 16u, "PSMT4 local-local copy should preserve the full packed byte count");
            for (size_t i = 0; i < sizeof(packedSource); ++i)
            {
                t.Equals(out[i], packedSource[i], "PSMT4 local-local copy should preserve packed nibble order");
            }
        });

        tc.Run("GS T4 CSM1 lookup matches Veronica ClutCopy layout", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kTexTbp = 64u;
            constexpr uint32_t kClutCbp = 128u;
            constexpr uint32_t kFrameReg =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kTex0 =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_T4) << 20) |
                (0ull << 26) |
                (0ull << 30) |
                (1ull << 34) |
                (1ull << 35) |
                (static_cast<uint64_t>(kClutCbp) << 37) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 51) |
                (1ull << 61);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |  // TME
                (1ull << 8);   // FST
            constexpr uint32_t kExpectedColor = 0x800000FFu; // RGBA = (255,0,0,128)
            constexpr uint32_t kWrongColor = 0x8000FF00u;    // RGBA = (0,255,0,128)

            const uint32_t texNibbleAddr = GSPSMT4::addrPSMT4(kTexTbp, 1u, 0u, 0u);
            const uint32_t texByteOff = texNibbleAddr >> 1;
            vram[texByteOff] = static_cast<uint8_t>((vram[texByteOff] & 0xF0u) | 0x08u);

            // Veronica uploads CSM1 CLUT rows with a 64-pixel GS stride, so logical entry 8
            // resolves to row 1, column 0 after the CSM1 swizzle.
            const uint32_t wrongClutOff = GSPSMCT32::addrPSMCT32(kClutCbp, 1u, 8u, 0u);
            const uint32_t expectedClutOff = GSPSMCT32::addrPSMCT32(kClutCbp, 1u, 0u, 1u);
            std::memcpy(vram.data() + wrongClutOff, &kWrongColor, sizeof(kWrongColor));
            std::memcpy(vram.data() + expectedClutOff, &kExpectedColor, sizeof(kExpectedColor));

            gs.writeRegister(GS_REG_FRAME_1, kFrameReg);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, 0ull);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_ALPHA_1, 0ull);
            gs.writeRegister(GS_REG_TEX0_1, kTex0);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, 0x80808080ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, (16ull << 0) | (16ull << 16));

            uint32_t pixel = 0u;
            std::memcpy(&pixel, vram.data(), sizeof(pixel));
            t.Equals(pixel, kExpectedColor,
                     "T4 CSM1 lookup should follow Veronica's swizzled CLUT row layout for logical index 8");
        });

        tc.Run("GS T8 CT32-uploaded CSM1 CLUT follows swizzled palette layout", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kTexTbp = 64u;
            constexpr uint32_t kClutCbp = 128u;
            constexpr uint64_t kFrameReg =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kTex0 =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_T8) << 20) |
                (0ull << 26) |
                (0ull << 30) |
                (1ull << 34) |
                (1ull << 35) |
                (static_cast<uint64_t>(kClutCbp) << 37) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 51) |
                (1ull << 61);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |  // TME
                (1ull << 8);   // FST
            constexpr uint64_t kClutBitblt =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24) |
                (static_cast<uint64_t>(kClutCbp) << 32) |
                (1ull << 48) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 56);
            constexpr uint64_t kClutRect =
                (16ull << 0) |
                (2ull << 32);
            constexpr uint32_t kExpectedColor = 0x80FFFFFFu;

            const uint32_t texOff = GSPSMT8::addrPSMT8(kTexTbp, 1u, 0u, 0u);
            vram[texOff] = 8u;

            std::vector<uint32_t> clut(32u, 0u);
            clut[16] = kExpectedColor;

            gs.writeRegister(GS_REG_BITBLTBUF, kClutBitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, kClutRect);
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            std::vector<uint8_t> packet;
            appendU64(packet,
                      makeGifTag(static_cast<uint16_t>((clut.size() * sizeof(uint32_t)) / 16u),
                                 GIF_FMT_IMAGE,
                                 0u,
                                 true));
            appendU64(packet, 0ull);
            const size_t payloadOffset = packet.size();
            packet.resize(payloadOffset + clut.size() * sizeof(uint32_t));
            std::memcpy(packet.data() + payloadOffset, clut.data(), clut.size() * sizeof(uint32_t));
            gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));

            gs.writeRegister(GS_REG_FRAME_1, kFrameReg);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, 0ull);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_ALPHA_1, 0ull);
            gs.writeRegister(GS_REG_TEX0_1, kTex0);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, 0x80808080ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, (16ull << 0) | (16ull << 16));

            uint32_t pixel = 0u;
            std::memcpy(&pixel, vram.data(), sizeof(pixel));
            t.Equals(pixel, kExpectedColor,
                     "T8 CSM1 CLUT sampling should read CT32-uploaded palette entries through GS swizzled addressing");
        });

        tc.Run("GS TEX2 updates CLUT state independently from TEX0", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kTexTbp = 64u;
            constexpr uint32_t kWrongClutCbp = 128u;
            constexpr uint32_t kExpectedClutCbp = 192u;
            constexpr uint64_t kFrameReg =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kTex0 =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_T8) << 20) |
                (0ull << 26) |
                (0ull << 30) |
                (1ull << 34) |
                (1ull << 35) |
                (static_cast<uint64_t>(kWrongClutCbp) << 37) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 51) |
                (1ull << 55) |
                (1ull << 61);
            constexpr uint64_t kTex2 =
                (static_cast<uint64_t>(GS_PSM_T8) << 20) |
                (static_cast<uint64_t>(kExpectedClutCbp) << 37) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 51) |
                (1ull << 55) |
                (1ull << 61);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |
                (1ull << 8);
            constexpr uint32_t kWrongColor = 0xFF00FF00u;
            constexpr uint32_t kExpectedColor = 0xFF0000FFu;

            const uint32_t texOff = GSPSMT8::addrPSMT8(kTexTbp, 1u, 0u, 0u);
            vram[texOff] = 8u;

            const uint32_t wrongClutOff = GSPSMCT32::addrPSMCT32(kWrongClutCbp, 1u, 8u, 0u);
            const uint32_t expectedClutOff = GSPSMCT32::addrPSMCT32(kExpectedClutCbp, 1u, 8u, 0u);
            std::memcpy(vram.data() + wrongClutOff, &kWrongColor, sizeof(kWrongColor));
            std::memcpy(vram.data() + expectedClutOff, &kExpectedColor, sizeof(kExpectedColor));

            gs.writeRegister(GS_REG_FRAME_1, kFrameReg);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, 0ull);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_ALPHA_1, 0ull);
            gs.writeRegister(GS_REG_TEX0_1, kTex0);
            gs.writeRegister(GS_REG_TEX2_1, kTex2);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, 0x80808080ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, (16ull << 0) | (16ull << 16));

            uint32_t pixel = 0u;
            std::memcpy(&pixel, vram.data(), sizeof(pixel));
            t.Equals(pixel, kExpectedColor,
                     "TEX2 should override the active CLUT base and format state without requiring a new TEX0 write");
        });

        tc.Run("GS TEXCLUT offsets CSM2 T8 CLUT fetch coordinates", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kTexTbp = 64u;
            constexpr uint32_t kClutCbp = 128u;
            constexpr uint64_t kFrameReg =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kTex0 =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_T8) << 20) |
                (0ull << 26) |
                (0ull << 30) |
                (1ull << 34) |
                (1ull << 35) |
                (static_cast<uint64_t>(kClutCbp) << 37) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 51) |
                (1ull << 55) |
                (1ull << 61);
            constexpr uint64_t kTexClut =
                (1ull << 0) |
                (3ull << 6) |
                (2ull << 12);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |
                (1ull << 8);
            constexpr uint32_t kWrongColor = 0xFF00FF00u;
            constexpr uint32_t kExpectedColor = 0xFF3366CCu;

            const uint32_t texOff = GSPSMT8::addrPSMT8(kTexTbp, 1u, 0u, 0u);
            vram[texOff] = 0u;

            const uint32_t wrongClutOff = GSPSMCT32::addrPSMCT32(kClutCbp, 1u, 0u, 0u);
            const uint32_t expectedClutOff = GSPSMCT32::addrPSMCT32(kClutCbp, 1u, 48u, 2u);
            std::memcpy(vram.data() + wrongClutOff, &kWrongColor, sizeof(kWrongColor));
            std::memcpy(vram.data() + expectedClutOff, &kExpectedColor, sizeof(kExpectedColor));

            gs.writeRegister(GS_REG_FRAME_1, kFrameReg);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, 0ull);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_ALPHA_1, 0ull);
            gs.writeRegister(GS_REG_TEXCLUT, kTexClut);
            gs.writeRegister(GS_REG_TEX0_1, kTex0);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, 0x80808080ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, (16ull << 0) | (16ull << 16));

            uint32_t pixel = 0u;
            std::memcpy(&pixel, vram.data(), sizeof(pixel));
            t.Equals(pixel, kExpectedColor,
                     "CSM2 should apply TEXCLUT.COU in units of 16 pixels");
        });

        tc.Run("GS CSM1 CLUT lookup ignores TEXCLUT coordinates", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kTexTbp = 64u;
            constexpr uint32_t kClutCbp = 128u;
            constexpr uint64_t kFrameReg =
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kTex0 =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_T8) << 20) |
                (1ull << 34) |
                (1ull << 35) |
                (static_cast<uint64_t>(kClutCbp) << 37) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 51) |
                (1ull << 61); // CSM1, CLD=LOAD
            constexpr uint64_t kTexClut =
                (1ull << 0) |
                (3ull << 6) |
                (2ull << 12);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |
                (1ull << 8);
            constexpr uint32_t kExpectedColor = 0x803366CCu;
            constexpr uint32_t kWrongColor = 0x8000FF00u;

            const uint32_t texOff = GSPSMT8::addrPSMT8(kTexTbp, 1u, 0u, 0u);
            vram[texOff] = 0u;

            const uint32_t expectedClutOff =
                GSPSMCT32::addrPSMCT32(kClutCbp, 1u, 0u, 0u);
            const uint32_t wrongClutOff =
                GSPSMCT32::addrPSMCT32(kClutCbp, 1u, 48u, 2u);
            std::memcpy(vram.data() + expectedClutOff, &kExpectedColor, sizeof(kExpectedColor));
            std::memcpy(vram.data() + wrongClutOff, &kWrongColor, sizeof(kWrongColor));

            gs.writeRegister(GS_REG_FRAME_1, kFrameReg);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, 0ull);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_TEX0_1, kTex0);
            gs.writeRegister(GS_REG_TEXCLUT, kTexClut);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, 0x80808080ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, (16ull << 0) | (16ull << 16));

            uint32_t pixel = 0u;
            std::memcpy(&pixel, vram.data(), sizeof(pixel));
            t.Equals(pixel, kExpectedColor,
                     "CSM1 should fetch from TEX0.CBP regardless of TEXCLUT");
        });

        tc.Run("GS TEXA expands CT24 alpha and honors AEM for black texels", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kTexTbp = 64u;
            constexpr uint64_t kFrameReg =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (1ull << 16) |
                (0ull << 32) |
                (0ull << 48);
            constexpr uint64_t kTex0 =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_CT24) << 20) |
                (1ull << 26) |
                (0ull << 30) |
                (1ull << 34) |
                (1ull << 35);
            constexpr uint64_t kTexa =
                (0x55ull << 0) |
                (1ull << 15) |
                (0xAAull << 32);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |
                (1ull << 8);
            constexpr uint32_t kExpectedRed = 0x550000FFu;
            constexpr uint32_t kExpectedBlack = 0x00000000u;

            const uint32_t redOff = GSPSMCT32::addrPSMCT32(kTexTbp, 1u, 0u, 0u);
            vram[redOff + 0u] = 0xFFu;
            vram[redOff + 1u] = 0x00u;
            vram[redOff + 2u] = 0x00u;
            vram[redOff + 3u] = 0x00u;

            const uint32_t blackOff = GSPSMCT32::addrPSMCT32(kTexTbp, 1u, 1u, 0u);
            vram[blackOff + 0u] = 0x00u;
            vram[blackOff + 1u] = 0x00u;
            vram[blackOff + 2u] = 0x00u;
            vram[blackOff + 3u] = 0x00u;

            gs.writeRegister(GS_REG_FRAME_1, kFrameReg);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_ALPHA_1, 0ull);
            gs.writeRegister(GS_REG_TEX0_1, kTex0);
            gs.writeRegister(GS_REG_TEXA, kTexa);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, 0x80808080ull);

            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, (16ull << 0) | (16ull << 16));

            gs.writeRegister(GS_REG_UV, (16ull << 0));
            gs.writeRegister(GS_REG_XYZ2, (16ull << 0));
            gs.writeRegister(GS_REG_UV, (16ull << 0));
            gs.writeRegister(GS_REG_XYZ2, (32ull << 0) | (16ull << 16));

            const uint32_t redPixel = readReferencePSMCT32Pixel(vram, 0u, 1u, 0u, 0u);
            const uint32_t blackPixel = readReferencePSMCT32Pixel(vram, 0u, 1u, 1u, 0u);
            t.Equals(redPixel, kExpectedRed,
                     "TEXA should supply TA0 as the alpha for non-alpha CT24 texels");
            t.Equals(blackPixel, kExpectedBlack,
                     "TEXA AEM should force zero alpha when a CT24 texel is RGB=0");
        });

        tc.Run("GS TCC=0 MODULATE uses texture RGB and keeps vertex alpha", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kTexTbp = 64u;
            constexpr uint64_t kFrameReg =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (0ull << 16) |
                (0ull << 32) |
                (0ull << 48);
            constexpr uint64_t kTex0 =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
                (0ull << 26) |
                (0ull << 30) |
                (0ull << 34) |
                (0ull << 35);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |
                (1ull << 8);
            constexpr uint32_t kTexturePixel =
                0x12u |
                (0x34u << 8) |
                (0x56u << 16) |
                (0x78u << 24);
            constexpr uint32_t kExpectedPixel =
                0x12u |
                (0x34u << 8) |
                (0x56u << 16) |
                (0x44u << 24);

            const uint32_t texOff = GSPSMCT32::addrPSMCT32(kTexTbp, 1u, 0u, 0u);
            std::memcpy(vram.data() + texOff, &kTexturePixel, sizeof(kTexturePixel));

            gs.writeRegister(GS_REG_FRAME_1, kFrameReg);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_ALPHA_1, 0ull);
            gs.writeRegister(GS_REG_TEX0_1, kTex0);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, 0x44808080ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, (16ull << 0) | (16ull << 16));

            const uint32_t pixel = readReferencePSMCT32Pixel(vram, 0u, 1u, 0u, 0u);
            t.Equals(pixel, kExpectedPixel,
                     "TCC=0 MODULATE should still use texture RGB while sourcing alpha from the shaded vertex");
        });

        tc.Run("GS HIGHLIGHT adds vertex alpha into RGB and texture alpha into A", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kTexTbp = 64u;
            constexpr uint64_t kFrameReg =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (0ull << 16) |
                (0ull << 32) |
                (0ull << 48);
            constexpr uint64_t kTex0 =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
                (0ull << 26) |
                (0ull << 30) |
                (1ull << 34) |
                (2ull << 35);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |
                (1ull << 8);
            constexpr uint32_t kTexturePixel =
                0x20u |
                (0x40u << 8) |
                (0x60u << 16) |
                (0x10u << 24);
            constexpr uint32_t kExpectedPixel =
                0x40u |
                (0x60u << 8) |
                (0x80u << 16) |
                (0x30u << 24);

            const uint32_t texOff = GSPSMCT32::addrPSMCT32(kTexTbp, 1u, 0u, 0u);
            std::memcpy(vram.data() + texOff, &kTexturePixel, sizeof(kTexturePixel));

            gs.writeRegister(GS_REG_FRAME_1, kFrameReg);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_ALPHA_1, 0ull);
            gs.writeRegister(GS_REG_TEX0_1, kTex0);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, 0x20808080ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, (16ull << 0) | (16ull << 16));

            const uint32_t pixel = readReferencePSMCT32Pixel(vram, 0u, 1u, 0u, 0u);
            t.Equals(pixel, kExpectedPixel,
                     "HIGHLIGHT should add the shaded vertex alpha into RGB and accumulate texture plus vertex alpha");
        });

        tc.Run("GS HIGHLIGHT2 keeps texture alpha while adding vertex alpha into RGB", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kTexTbp = 64u;
            constexpr uint64_t kFrameReg =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (0ull << 16) |
                (0ull << 32) |
                (0ull << 48);
            constexpr uint64_t kTex0 =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
                (0ull << 26) |
                (0ull << 30) |
                (1ull << 34) |
                (3ull << 35);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |
                (1ull << 8);
            constexpr uint32_t kTexturePixel =
                0x20u |
                (0x40u << 8) |
                (0x60u << 16) |
                (0x10u << 24);
            constexpr uint32_t kExpectedPixel =
                0x40u |
                (0x60u << 8) |
                (0x80u << 16) |
                (0x10u << 24);

            const uint32_t texOff = GSPSMCT32::addrPSMCT32(kTexTbp, 1u, 0u, 0u);
            std::memcpy(vram.data() + texOff, &kTexturePixel, sizeof(kTexturePixel));

            gs.writeRegister(GS_REG_FRAME_1, kFrameReg);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_ALPHA_1, 0ull);
            gs.writeRegister(GS_REG_TEX0_1, kTex0);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, 0x20808080ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, (16ull << 0) | (16ull << 16));

            const uint32_t pixel = readReferencePSMCT32Pixel(vram, 0u, 1u, 0u, 0u);
            t.Equals(pixel, kExpectedPixel,
                     "HIGHLIGHT2 should add the shaded vertex alpha into RGB while preserving the texture alpha");
        });

        tc.Run("GS linear CT32 and T8 sampling skips zero-weight taps", [](TestCase &t)
        {
            constexpr uint32_t kTextureBase = 64u;
            constexpr uint32_t kClutBase = 128u;
            constexpr uint32_t kColors[4] = {
                0xF02080E0u,
                0x1080F020u,
                0xC0E010A0u,
                0x40A0D060u,
            };

            for (uint8_t psm : {GS_PSM_CT32, GS_PSM_T8})
            {
                std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
                GS gs;
                gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

                const uint64_t tex0 =
                    (static_cast<uint64_t>(kTextureBase) << 0) |
                    (1ull << 14) |
                    (static_cast<uint64_t>(psm) << 20) |
                    (2ull << 26) |
                    (2ull << 30) |
                    (1ull << 34) |
                    (1ull << 35) |
                    (static_cast<uint64_t>(kClutBase) << 37) |
                    (static_cast<uint64_t>(GS_PSM_CT32) << 51) |
                    (1ull << 61);

                if (psm == GS_PSM_CT32)
                {
                    gs.WriteVram(psm, kTextureBase, 1u, 0u, 0u, kColors[0]);
                    gs.WriteVram(psm, kTextureBase, 1u, 1u, 0u, kColors[1]);
                    gs.WriteVram(psm, kTextureBase, 1u, 0u, 1u, kColors[2]);
                    gs.WriteVram(psm, kTextureBase, 1u, 1u, 1u, kColors[3]);
                }
                else
                {
                    const uint8_t indices[4] = {1u, 2u, 3u, 4u};
                    const uint32_t x[4] = {0u, 1u, 0u, 1u};
                    const uint32_t y[4] = {0u, 0u, 1u, 1u};
                    for (size_t index = 0u; index < 4u; ++index)
                    {
                        vram[GSPSMT8::addrPSMT8(
                            kTextureBase, 1u, x[index], y[index])] =
                            indices[index];
                        writeReferencePSMCT32Pixel(
                            vram,
                            kClutBase,
                            1u,
                            indices[index],
                            0u,
                            kColors[index]);
                    }
                }

                constexpr uint64_t kLinearTex1 =
                    (1ull << 5) |
                    (1ull << 6);
                constexpr uint64_t kPrim =
                    static_cast<uint64_t>(GS_PRIM_SPRITE) |
                    (1ull << 4) |
                    (1ull << 8);
                gs.writeRegister(GS_REG_CLAMP_1, 5ull);
                gs.writeRegister(GS_REG_TEX0_1, tex0);
                gs.writeRegister(GS_REG_TEX1_1, kLinearTex1);
                gs.writeRegister(GS_REG_PRIM, kPrim);

                GSRasterizer rasterizer;
                for (uint8_t weightV = 0u; weightV < 16u; ++weightV)
                {
                    for (uint8_t weightU = 0u; weightU < 16u; ++weightU)
                    {
                        size_t requestedReads = 0u;
                        const auto selectedTaps =
                            GSRasterizerDetail::readRequiredBilinearTaps(
                                weightU,
                                weightV,
                                [&](int uIndex, int vIndex)
                                {
                                    ++requestedReads;
                                    const size_t index = static_cast<size_t>(
                                        vIndex * 2 + uIndex);
                                    return GSRasterizerDetail::LinearTextureTap{
                                        static_cast<uint32_t>(index),
                                        kColors[index],
                                    };
                                });
                        const size_t expectedReads =
                            (weightU == 0u ? 1u : 2u) *
                            (weightV == 0u ? 1u : 2u);
                        const uint32_t expectedColor =
                            referenceBilinearColor4(
                                kColors[0],
                                kColors[1],
                                kColors[2],
                                kColors[3],
                                weightU,
                                weightV);
                        const uint16_t u =
                            static_cast<uint16_t>(8u + weightU);
                        const uint16_t v =
                            static_cast<uint16_t>(8u + weightV);
                        const uint32_t publicColor = rasterizer.sampleTexture(
                            &gs, 0.0f, 0.0f, 1.0f, u, v);
                        const std::string format =
                            psm == GS_PSM_CT32 ? "CT32" : "T8";
                        t.Equals(
                            requestedReads,
                            expectedReads,
                            format + " linear tap selection should request only nonzero-weight taps");
                        t.Equals(
                            selectedTaps[0].color,
                            kColors[0],
                            format + " linear tap selection should preserve the base sample");
                        t.Equals(
                            publicColor,
                            expectedColor,
                            format + " linear sampling should match signed nested interpolation");
                    }
                }
            }
        });

        tc.Run("GS packed CT32 DECAL sprite kernel matches the reference path", [](TestCase &t)
        {
            const PackedSpriteTestConfiguration configuration;
            const PackedSpriteRenderResult reference =
                renderPackedSpriteTest(
                    configuration,
                    GSRasterizerDetail::
                        PackedSpriteKernelOverride::
                            ForceReference);
            const PackedSpriteRenderResult optimized =
                renderPackedSpriteTest(
                    configuration,
                    GSRasterizerDetail::
                        PackedSpriteKernelOverride::
                            ForceOptimized);
            GSRasterizerDetail::setPackedSpriteKernelOverride(
                GSRasterizerDetail::PackedSpriteKernelOverride::
                    Automatic);

            t.Equals(
                reference.packedDispatches,
                uint64_t{0u},
                "the forced reference path must not dispatch the packed kernel");
            t.Equals(
                optimized.packedDispatches,
                uint64_t{1u},
                "the eligible forced optimized draw must dispatch the packed kernel once");
            t.IsTrue(
                optimized.vram == reference.vram,
                "the packed CT32 DECAL draw must match full reference VRAM");
        });

        tc.Run("GS packed sprite SIMD kernels match scalar vertical interpolation", [](TestCase &t)
        {
            struct ImplementationCase
            {
                const char *name;
                GSRasterizerDetail::PackedSpriteKernelOverride
                    overrideMode;
                GSRasterizerDetail::PackedSpriteKernelImplementation
                    implementation;
            };
            const ImplementationCase cases[] = {
                {
                    "SSE4.1",
                    GSRasterizerDetail::
                        PackedSpriteKernelOverride::ForceSse41,
                    GSRasterizerDetail::
                        PackedSpriteKernelImplementation::Sse41,
                },
                {
                    "AVX2",
                    GSRasterizerDetail::
                        PackedSpriteKernelOverride::ForceAvx2,
                    GSRasterizerDetail::
                        PackedSpriteKernelImplementation::Avx2,
                },
            };

            for (uint16_t weight = 0u; weight < 16u; ++weight)
            {
                PackedSpriteTestConfiguration configuration;
                configuration.x0 -= 8u;
                configuration.x1 -= 8u;
                configuration.v0 += weight;
                configuration.v1 += weight;

                const PackedSpriteRenderResult scalar =
                    renderPackedSpriteTest(
                        configuration,
                        GSRasterizerDetail::
                            PackedSpriteKernelOverride::
                                ForceScalar);
                t.Equals(
                    scalar.packedDispatches,
                    uint64_t{1u},
                    "the forced scalar draw must dispatch the packed kernel once");
                t.Equals(
                    scalar.implementation,
                    GSRasterizerDetail::
                        PackedSpriteKernelImplementation::Scalar,
                    "the forced scalar draw must report the scalar implementation");
                t.Equals(
                    scalar.vectorGroups,
                    uint64_t{0u},
                    "the scalar implementation must not report vector groups");

                for (const ImplementationCase &testCase : cases)
                {
                    if (!GSRasterizerDetail::
                            packedSpriteKernelImplementationAvailable(
                                testCase.implementation))
                    {
                        continue;
                    }

                    const PackedSpriteRenderResult vector =
                        renderPackedSpriteTest(
                            configuration,
                            testCase.overrideMode);
                    const std::string prefix =
                        std::string(testCase.name) +
                        " weight " +
                        std::to_string(weight) + ": ";
                    t.Equals(
                        vector.packedDispatches,
                        uint64_t{1u},
                        prefix +
                            "the forced vector draw must dispatch once");
                    t.Equals(
                        vector.implementation,
                        testCase.implementation,
                        prefix +
                            "the requested implementation must be selected");
                    if (weight != 8u)
                    {
                        t.IsTrue(
                            vector.vectorGroups != 0u,
                            prefix +
                                "full vertical-filter groups must enter the vector path");
                    }
                    t.IsTrue(
                        vector.vram == scalar.vram,
                        prefix +
                            "vector and scalar packed paths must match full VRAM");
                }
            }

            GSRasterizerDetail::setPackedSpriteKernelOverride(
                GSRasterizerDetail::PackedSpriteKernelOverride::
                    Automatic);
        });

        tc.Run("GS packed sprite selector rejects states with extra pixel work", [](TestCase &t)
        {
            struct RejectionCase
            {
                const char *name;
                void (*mutate)(
                    PackedSpriteTestConfiguration &configuration);
            };
            const RejectionCase cases[] = {
                {
                    "point filter",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.tex1 = 0ull;
                    },
                },
                {
                    "mip level",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.tex1 |= 1ull << 2;
                    },
                },
                {
                    "U clamp",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.clamp = 1ull;
                    },
                },
                {
                    "U region clamp",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.clamp =
                            2ull |
                            (2ull << 4) |
                            (100ull << 14);
                    },
                },
                {
                    "U region repeat",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.clamp =
                            3ull |
                            (0x7Full << 4) |
                            (1ull << 14);
                    },
                },
                {
                    "V clamp",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.clamp = 1ull << 2;
                    },
                },
                {
                    "T8 texture",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.tex0 =
                            (configuration.tex0 &
                             ~(0x3Full << 20)) |
                            (static_cast<uint64_t>(GS_PSM_T8)
                             << 20);
                    },
                },
                {
                    "texture alpha disabled",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.tex0 &=
                            ~(1ull << 34);
                    },
                },
                {
                    "MODULATE",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.tex0 &=
                            ~(0x3ull << 35);
                    },
                },
                {
                    "STQ coordinates",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.prim &=
                            ~(1ull << 8);
                    },
                },
                {
                    "fog",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.prim |=
                            1ull << 5;
                    },
                },
                {
                    "alpha blend",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.prim |=
                            1ull << 6;
                    },
                },
                {
                    "alpha test",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.test |=
                            3ull;
                    },
                },
                {
                    "destination alpha test",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.test |=
                            1ull << 14;
                    },
                },
                {
                    "depth comparison",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.test =
                            (configuration.test &
                             ~(0x3ull << 17)) |
                            (2ull << 17);
                    },
                },
                {
                    "depth write",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.zbuf &=
                            ~(1ull << 32);
                    },
                },
                {
                    "framebuffer mask",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.frame |=
                            0x00FF0000ull << 32;
                    },
                },
                {
                    "FBA",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.fba = 1ull;
                    },
                },
                {
                    "CT16 framebuffer",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.frame =
                            (configuration.frame &
                             ~(0x3Full << 24)) |
                            (static_cast<uint64_t>(
                                 GS_PSM_CT16)
                             << 24);
                    },
                },
                {
                    "untextured sprite",
                    [](PackedSpriteTestConfiguration &configuration)
                    {
                        configuration.prim &=
                            ~(1ull << 4);
                    },
                },
            };

            for (const RejectionCase &testCase : cases)
            {
                PackedSpriteTestConfiguration configuration;
                testCase.mutate(configuration);
                const PackedSpriteRenderResult reference =
                    renderPackedSpriteTest(
                        configuration,
                        GSRasterizerDetail::
                            PackedSpriteKernelOverride::
                                ForceReference);
                const PackedSpriteRenderResult optimized =
                    renderPackedSpriteTest(
                        configuration,
                        GSRasterizerDetail::
                            PackedSpriteKernelOverride::
                                ForceOptimized);
                const std::string prefix =
                    std::string(testCase.name) + ": ";
                t.Equals(
                    optimized.packedDispatches,
                    uint64_t{0u},
                    prefix +
                        "the selector must reject this state before dispatch");
                t.IsTrue(
                    optimized.vram == reference.vram,
                    prefix +
                        "forced optimized mode must fall back to exact reference output");
            }
            GSRasterizerDetail::setPackedSpriteKernelOverride(
                GSRasterizerDetail::PackedSpriteKernelOverride::
                    Automatic);
        });

        tc.Run("GS packed sprite kernel covers exact tap and overlap boundaries", [](TestCase &t)
        {
            struct DifferentialCase
            {
                const char *name;
                PackedSpriteTestConfiguration configuration;
            };
            std::array<DifferentialCase, 6> cases;

            cases[0].name = "four taps with negative biased origin";
            cases[0].configuration =
                PackedSpriteTestConfiguration{};

            cases[1] = cases[0];
            cases[1].name = "zero U weight";
            cases[1].configuration.x0 -= 8u;
            cases[1].configuration.x1 -= 8u;

            cases[2] = cases[0];
            cases[2].name = "zero V weight";
            cases[2].configuration.y0 -= 8u;
            cases[2].configuration.y1 -= 8u;

            cases[3] = cases[0];
            cases[3].name = "one tap";
            cases[3].configuration.x0 -= 8u;
            cases[3].configuration.x1 -= 8u;
            cases[3].configuration.y0 -= 8u;
            cases[3].configuration.y1 -= 8u;

            cases[4] = cases[0];
            cases[4].name =
                "clipped partial lane groups across CT32 pages";
            cases[4].configuration.scissor =
                (63ull << 0) |
                (70ull << 16) |
                (29ull << 32) |
                (37ull << 48);

            cases[5] = cases[0];
            cases[5].name =
                "recursive framebuffer texture overlap";
            cases[5].configuration.tex0 &=
                ~0x3FFFull;
            cases[5].configuration.x0 = 1u * 16u;
            cases[5].configuration.x1 = 17u * 16u;
            cases[5].configuration.y0 = 1u * 16u;
            cases[5].configuration.y1 = 9u * 16u;
            cases[5].configuration.u0 = 0u;
            cases[5].configuration.u1 = 16u * 16u;
            cases[5].configuration.v0 = 1u * 16u;
            cases[5].configuration.v1 = 9u * 16u;

            for (const DifferentialCase &testCase : cases)
            {
                const PackedSpriteRenderResult reference =
                    renderPackedSpriteTest(
                        testCase.configuration,
                        GSRasterizerDetail::
                            PackedSpriteKernelOverride::
                                ForceReference);
                const PackedSpriteRenderResult optimized =
                    renderPackedSpriteTest(
                        testCase.configuration,
                        GSRasterizerDetail::
                            PackedSpriteKernelOverride::
                                ForceOptimized);
                const std::string prefix =
                    std::string(testCase.name) + ": ";
                t.Equals(
                    optimized.packedDispatches,
                    uint64_t{1u},
                    prefix +
                        "the eligible draw must dispatch exactly once");
                t.IsTrue(
                    optimized.vram == reference.vram,
                    prefix +
                        "packed and reference paths must match full VRAM");
            }
            GSRasterizerDetail::setPackedSpriteKernelOverride(
                GSRasterizerDetail::PackedSpriteKernelOverride::
                    Automatic);
        });

        tc.Run("GS TEX1 linear filter blends T4 STQ triangle samples", [](TestCase &t)
        {
            auto renderSamplePixel = [](uint64_t tex1Reg) -> uint32_t
            {
                std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
                GS gs;
                gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

                constexpr uint32_t kTexTbp = 64u;
                constexpr uint32_t kClutCbp = 128u;
                constexpr uint64_t kFrame =
                    (0ull << 0) |
                    (1ull << 16) |
                    (static_cast<uint64_t>(GS_PSM_CT32) << 24);
                constexpr uint64_t kZbuf = (1ull << 32);
                constexpr uint64_t kScissor =
                    (0ull << 0) |
                    (4ull << 16) |
                    (0ull << 32) |
                    (4ull << 48);
                constexpr uint64_t kTex0 =
                    (static_cast<uint64_t>(kTexTbp) << 0) |
                    (1ull << 14) |
                    (static_cast<uint64_t>(GS_PSM_T4) << 20) |
                    (1ull << 26) |
                    (0ull << 30) |
                    (1ull << 34) |
                    (1ull << 35) |
                    (static_cast<uint64_t>(kClutCbp) << 37) |
                    (static_cast<uint64_t>(GS_PSM_CT32) << 51) |
                    (1ull << 61);
                constexpr uint64_t kPrim =
                    static_cast<uint64_t>(GS_PRIM_TRIANGLE) |
                    (1ull << 4) |
                    (0ull << 8);
                constexpr uint64_t kRgbaq = 0x3F80000080808080ull;
                constexpr uint32_t kBlack = 0x80000000u;
                constexpr uint32_t kWhite = 0x80FFFFFFu;

                writePSMT4Texel(vram, kTexTbp, 1u, 0u, 0u, 0u);
                writePSMT4Texel(vram, kTexTbp, 1u, 1u, 0u, 1u);
                std::memcpy(vram.data() + kClutCbp * 256u + 0u * 4u, &kBlack, sizeof(kBlack));
                std::memcpy(vram.data() + kClutCbp * 256u + 1u * 4u, &kWhite, sizeof(kWhite));

                auto packFloat = [](float value) -> uint32_t
                {
                    uint32_t bits = 0u;
                    std::memcpy(&bits, &value, sizeof(bits));
                    return bits;
                };

                auto packSt = [&](float s, float tVal) -> uint64_t
                {
                    return static_cast<uint64_t>(packFloat(s)) |
                           (static_cast<uint64_t>(packFloat(tVal)) << 32);
                };

                gs.writeRegister(GS_REG_FRAME_1, kFrame);
                gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
                gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
                gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
                gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
                gs.writeRegister(GS_REG_ALPHA_1, 0ull);
                gs.writeRegister(GS_REG_TEX0_1, kTex0);
                gs.writeRegister(GS_REG_TEX1_1, tex1Reg);
                gs.writeRegister(GS_REG_PRIM, kPrim);
                gs.writeRegister(GS_REG_RGBAQ, kRgbaq);
                gs.writeRegister(GS_REG_ST, packSt(0.0f, 0.0f));
                gs.writeRegister(GS_REG_XYZ2, 0ull);
                // GS pixels are sampled at integer window coordinates. S=1.5
                // puts pixel (1,1) between the two texels for the linear case.
                gs.writeRegister(GS_REG_ST, packSt(1.5f, 0.0f));
                gs.writeRegister(GS_REG_XYZ2, (64ull << 0) | (0ull << 16));
                gs.writeRegister(GS_REG_ST, packSt(0.0f, 0.0f));
                gs.writeRegister(GS_REG_XYZ2, (0ull << 0) | (64ull << 16));

                return readReferencePSMCT32Pixel(vram, 0u, 1u, 1u, 1u);
            };

            constexpr uint64_t kTex1Linear =
                (1ull << 5) |
                (1ull << 6);

            const uint32_t nearestPixel = renderSamplePixel(0ull);
            const uint32_t linearPixel = renderSamplePixel(kTex1Linear);

            t.Equals(nearestPixel, 0x80000000u,
                     "point sampling should keep the sampled STQ triangle pixel on texel 0");

            const uint8_t linearR = static_cast<uint8_t>(linearPixel & 0xFFu);
            const uint8_t linearA = static_cast<uint8_t>((linearPixel >> 24) & 0xFFu);
            t.IsTrue(linearR > 0x10u && linearR < 0x70u,
                     "linear filtering should blend the STQ triangle sample between black and white T4 texels");
            t.Equals(linearA, static_cast<uint8_t>(0x80u),
                     "linear filtering should preserve the shared opaque alpha from the CLUT entries");
        });

        tc.Run("GS recursive sprites retain the cached texture source across draws", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kOriginal0 = 0x80112233u;
            constexpr uint32_t kOriginal1 = 0x80445566u;
            constexpr uint64_t kFrame =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = 1ull << 32;
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (3ull << 16) |
                (0ull << 32) |
                (0ull << 48);
            constexpr uint64_t kTex0 =
                (0ull << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
                (6ull << 26) |
                (0ull << 30) |
                (1ull << 34) |
                (1ull << 35);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |
                (1ull << 8);
            constexpr uint64_t kRgbaq =
                0x80ull |
                (0x80ull << 8) |
                (0x80ull << 16) |
                (0x80ull << 24) |
                (0x3F800000ull << 32);

            gs.WriteVram(GS_PSM_CT32, 0u, 1u, 0u, 0u, kOriginal0);
            gs.WriteVram(GS_PSM_CT32, 0u, 1u, 1u, 0u, kOriginal1);
            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_TEX0_1, kTex0);
            gs.writeRegister(GS_REG_TEX1_1, 0ull);
            gs.writeRegister(GS_REG_CLAMP_1, 5ull);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, kRgbaq);

            // First overwrite texel 1 with texel 0.
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, 16ull);
            gs.writeRegister(GS_REG_UV, 16ull);
            gs.writeRegister(GS_REG_XYZ2, 32ull | (16ull << 16));

            const GsReplayState captured = gs.captureReplayState();
            t.IsTrue(captured.rasterizer.feedbackSnapshotValid,
                     "the first recursive draw should retain its source snapshot");
            t.Equals(
                captured.rasterizer.feedbackVram.size(),
                PS2_GS_VRAM_SIZE,
                "a recursive replay boundary should retain all feedback VRAM");
            const std::vector<uint8_t> replayInitialVram = vram;
            std::vector<uint8_t> encodedState;
            std::string stateError;
            t.IsTrue(
                encodeGsReplayState(
                    captured, encodedState, &stateError),
                "recursive feedback state should encode");
            GsReplayState decodedState{};
            t.IsTrue(
                decodeGsReplayState(
                    encodedState, decodedState, &stateError),
                "recursive feedback state should decode");

            // Then copy original texel 1 to texel 2. Recursive drawing reads
            // through the texture cache, so this must not observe the first
            // draw's framebuffer write.
            gs.writeRegister(GS_REG_UV, 16ull);
            gs.writeRegister(GS_REG_XYZ2, 32ull);
            gs.writeRegister(GS_REG_UV, 32ull);
            gs.writeRegister(GS_REG_XYZ2, 48ull | (16ull << 16));

            std::vector<uint8_t> replayVram = replayInitialVram;
            GS replay;
            replay.init(
                replayVram.data(),
                static_cast<uint32_t>(replayVram.size()),
                nullptr);
            t.IsTrue(
                replay.restoreReplayState(decodedState),
                "recursive feedback state should restore");
            replay.writeRegister(GS_REG_UV, 16ull);
            replay.writeRegister(GS_REG_XYZ2, 32ull);
            replay.writeRegister(GS_REG_UV, 32ull);
            replay.writeRegister(
                GS_REG_XYZ2,
                48ull | (16ull << 16));

            t.Equals(gs.ReadVram(GS_PSM_CT32, 0u, 1u, 1u, 0u),
                     kOriginal0,
                     "the first recursive sprite should update its destination");
            t.Equals(gs.ReadVram(GS_PSM_CT32, 0u, 1u, 2u, 0u),
                     kOriginal1,
                     "a consecutive recursive sprite should sample the cached pre-draw surface");
            t.IsTrue(
                replayVram == vram,
                "restored feedback state should reproduce every GS VRAM byte");
        });

        tc.Run("GS render batches match sequential overlapping triangle draws", [](TestCase &t)
        {
            auto render = [=](bool batched)
            {
                std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
                GS gs;
                gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

                constexpr uint64_t kFrame =
                    (0ull << 0) |
                    (1ull << 16) |
                    (static_cast<uint64_t>(GS_PSM_CT32) << 24);
                constexpr uint64_t kZbuf = 1ull << 32;
                constexpr uint64_t kScissor =
                    (0ull << 0) |
                    (63ull << 16) |
                    (0ull << 32) |
                    (127ull << 48);
                constexpr uint64_t kAlpha =
                    (0ull << 0) |
                    (1ull << 2) |
                    (0ull << 4) |
                    (1ull << 6);
                constexpr uint64_t kPrim =
                    static_cast<uint64_t>(GS_PRIM_TRIANGLE) |
                    (1ull << 6);
                auto xyz = [](uint16_t x, uint16_t y) -> uint64_t
                {
                    return static_cast<uint64_t>(x * 16u) |
                           (static_cast<uint64_t>(y * 16u) << 16);
                };

                gs.writeRegister(GS_REG_FRAME_1, kFrame);
                gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
                gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
                gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
                gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
                gs.writeRegister(GS_REG_ALPHA_1, kAlpha);
                gs.writeRegister(GS_REG_PRIM, kPrim);

                if (batched)
                    gs.beginRenderBatch();
                for (uint32_t pass = 0u; pass < 16u; ++pass)
                {
                    const uint64_t rgbaq =
                        static_cast<uint64_t>((pass * 17u + 3u) & 0xFFu) |
                        (static_cast<uint64_t>((pass * 29u + 5u) & 0xFFu) << 8) |
                        (static_cast<uint64_t>((pass * 43u + 7u) & 0xFFu) << 16) |
                        (0x40ull << 24) |
                        (0x3F800000ull << 32);
                    gs.writeRegister(GS_REG_RGBAQ, rgbaq);
                    gs.writeRegister(GS_REG_XYZ2, xyz(0u, 0u));
                    gs.writeRegister(GS_REG_XYZ2, xyz(64u, 0u));
                    gs.writeRegister(GS_REG_XYZ2, xyz(0u, 128u));
                    gs.writeRegister(GS_REG_XYZ2, xyz(64u, 0u));
                    gs.writeRegister(GS_REG_XYZ2, xyz(64u, 128u));
                    gs.writeRegister(GS_REG_XYZ2, xyz(0u, 128u));
                }
                if (batched)
                    gs.endRenderBatch();
                return vram;
            };

            const std::vector<uint8_t> sequential = render(false);
            const std::vector<uint8_t> batched = render(true);
            t.IsTrue(batched == sequential,
                     "interleaved scanline workers must preserve draw order for overlapping blended primitives");
        });

        tc.Run("GS render batches retain each indexed draw's decoded palette", [](TestCase &t)
        {
            constexpr uint32_t kTextureBase = 256u;
            constexpr uint32_t kPaletteBaseA = 512u;
            constexpr uint32_t kPaletteBaseB = 544u;
            constexpr uint32_t kColorA = 0x80112233u;
            constexpr uint32_t kColorB = 0x80445566u;

            auto render = [=](bool batched)
            {
                std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
                const uint32_t paletteOffsetA =
                    GSPSMCT32::addrPSMCT32(kPaletteBaseA, 1u, 0u, 0u);
                const uint32_t paletteOffsetB =
                    GSPSMCT32::addrPSMCT32(kPaletteBaseB, 1u, 0u, 0u);
                std::memcpy(vram.data() + paletteOffsetA,
                            &kColorA,
                            sizeof(kColorA));
                std::memcpy(vram.data() + paletteOffsetB,
                            &kColorB,
                            sizeof(kColorB));

                GS gs;
                gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);
                constexpr uint64_t kFrame =
                    (0ull << 0) |
                    (1ull << 16) |
                    (static_cast<uint64_t>(GS_PSM_CT32) << 24);
                constexpr uint64_t kZbuf = 1ull << 32;
                constexpr uint64_t kScissor =
                    (0ull << 0) |
                    (63ull << 16) |
                    (0ull << 32) |
                    (63ull << 48);
                constexpr uint64_t kPrim =
                    static_cast<uint64_t>(GS_PRIM_SPRITE) |
                    (1ull << 4) |
                    (1ull << 8);
                auto tex0 = [](uint32_t paletteBase) -> uint64_t
                {
                    return
                        (static_cast<uint64_t>(kTextureBase) << 0) |
                        (1ull << 14) |
                        (static_cast<uint64_t>(GS_PSM_T8) << 20) |
                        (6ull << 26) |
                        (0ull << 30) |
                        (1ull << 34) |
                        (1ull << 35) |
                        (static_cast<uint64_t>(paletteBase) << 37) |
                        (static_cast<uint64_t>(GS_PSM_CT32) << 51) |
                        (1ull << 61);
                };
                auto uv = [](uint16_t x, uint16_t y) -> uint64_t
                {
                    return static_cast<uint64_t>(x * 16u) |
                           (static_cast<uint64_t>(y * 16u) << 16);
                };

                gs.writeRegister(GS_REG_FRAME_1, kFrame);
                gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
                gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
                gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
                gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
                gs.writeRegister(GS_REG_CLAMP_1, 5ull);
                gs.writeRegister(GS_REG_PRIM, kPrim);
                gs.writeRegister(GS_REG_RGBAQ, 0x3F80000080808080ull);

                if (batched)
                    gs.beginRenderBatch();
                for (uint16_t y = 0u; y < 64u; ++y)
                {
                    gs.writeRegister(
                        GS_REG_TEX0_1,
                        tex0((y & 1u) == 0u
                                 ? kPaletteBaseA
                                 : kPaletteBaseB));
                    gs.writeRegister(GS_REG_UV, uv(0u, 0u));
                    gs.writeRegister(GS_REG_XYZ2, uv(0u, y));
                    gs.writeRegister(GS_REG_UV, uv(64u, 1u));
                    gs.writeRegister(GS_REG_XYZ2, uv(64u, y + 1u));
                }
                if (batched)
                    gs.endRenderBatch();
                return vram;
            };

            const std::vector<uint8_t> sequential = render(false);
            const std::vector<uint8_t> batched = render(true);
            t.IsTrue(batched == sequential,
                     "queued indexed draws must keep the CLUT decoded when each draw was submitted");
            t.Equals(readReferencePSMCT32Pixel(batched, 0u, 1u, 0u, 0u),
                     kColorA,
                     "the first palette should shade the first row");
            t.Equals(readReferencePSMCT32Pixel(batched, 0u, 1u, 0u, 1u),
                     kColorB,
                     "the second palette should shade the next row");
        });

        tc.Run("GS render batches preserve recursive framebuffer snapshots", [](TestCase &t)
        {
            auto render = [](bool batched)
            {
                std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
                GS gs;
                gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

                for (uint32_t y = 0u; y < 64u; ++y)
                {
                    gs.WriteVram(
                        GS_PSM_CT32, 0u, 1u, 0u, y,
                        0x80000000u | (y * 0x00010101u));
                    gs.WriteVram(
                        GS_PSM_CT32, 0u, 1u, 1u, y,
                        0x80800000u | (y * 0x00000101u));
                }

                constexpr uint64_t kFrame =
                    (0ull << 0) |
                    (1ull << 16) |
                    (static_cast<uint64_t>(GS_PSM_CT32) << 24);
                constexpr uint64_t kZbuf = 1ull << 32;
                constexpr uint64_t kScissor =
                    (0ull << 0) |
                    (2ull << 16) |
                    (0ull << 32) |
                    (63ull << 48);
                constexpr uint64_t kTex0 =
                    (0ull << 0) |
                    (1ull << 14) |
                    (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
                    (6ull << 26) |
                    (6ull << 30) |
                    (1ull << 34) |
                    (1ull << 35);
                constexpr uint64_t kPrim =
                    static_cast<uint64_t>(GS_PRIM_SPRITE) |
                    (1ull << 4) |
                    (1ull << 8);
                auto uv = [](uint16_t x, uint16_t y) -> uint64_t
                {
                    return static_cast<uint64_t>(x * 16u) |
                           (static_cast<uint64_t>(y * 16u) << 16);
                };

                gs.writeRegister(GS_REG_FRAME_1, kFrame);
                gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
                gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
                gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
                gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
                gs.writeRegister(GS_REG_TEX0_1, kTex0);
                gs.writeRegister(GS_REG_TEX1_1, 0ull);
                gs.writeRegister(GS_REG_CLAMP_1, 5ull);
                gs.writeRegister(GS_REG_PRIM, kPrim);
                gs.writeRegister(GS_REG_RGBAQ, 0x3F80000080808080ull);

                if (batched)
                    gs.beginRenderBatch();
                for (uint16_t y = 0u; y < 64u; ++y)
                {
                    gs.writeRegister(GS_REG_UV, uv(0u, y));
                    gs.writeRegister(GS_REG_XYZ2, uv(1u, y));
                    gs.writeRegister(GS_REG_UV, uv(1u, y + 1u));
                    gs.writeRegister(GS_REG_XYZ2, uv(2u, y + 1u));

                    gs.writeRegister(GS_REG_UV, uv(1u, y));
                    gs.writeRegister(GS_REG_XYZ2, uv(2u, y));
                    gs.writeRegister(GS_REG_UV, uv(2u, y + 1u));
                    gs.writeRegister(GS_REG_XYZ2, uv(3u, y + 1u));
                }
                if (batched)
                    gs.endRenderBatch();
                return vram;
            };

            const std::vector<uint8_t> sequential = render(false);
            const std::vector<uint8_t> batched = render(true);
            t.IsTrue(batched == sequential,
                     "queued recursive draws must sample the same immutable pre-draw surface as sequential draws");
            t.Equals(readReferencePSMCT32Pixel(batched, 0u, 1u, 1u, 31u),
                     0x801F1F1Fu,
                     "the first recursive copy should use source column zero");
            t.Equals(readReferencePSMCT32Pixel(batched, 0u, 1u, 2u, 31u),
                     0x80801F1Fu,
                     "the second recursive copy should still use the original source column one");
        });

        tc.Run("GS FST sprites preserve accumulated 16.16 vertical precision", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kTexTbp = 64u;
            constexpr uint32_t kFramePage = 2u;
            constexpr uint64_t kFrame =
                (static_cast<uint64_t>(kFramePage) << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = 1ull << 32;
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (0ull << 16) |
                (0ull << 32) |
                (414ull << 48);
            constexpr uint64_t kTex0 =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
                (0ull << 26) |
                (9ull << 30) |
                (1ull << 34) |
                (1ull << 35);
            constexpr uint64_t kTex1 =
                (1ull << 5) |
                (1ull << 6);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |
                (1ull << 8);
            constexpr uint64_t kRgbaq =
                0x80ull |
                (0x80ull << 8) |
                (0x80ull << 16) |
                (0x80ull << 24) |
                (0x3F800000ull << 32);

            gs.WriteVram(GS_PSM_CT32, kTexTbp, 1u, 0u, 363u,
                         0x80000000u);
            gs.WriteVram(GS_PSM_CT32, kTexTbp, 1u, 0u, 364u,
                         0x800000FFu);
            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_TEX0_1, kTex0);
            gs.writeRegister(GS_REG_TEX1_1, kTex1);
            gs.writeRegister(GS_REG_CLAMP_1, 5ull);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, kRgbaq);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);
            gs.writeRegister(GS_REG_UV,
                             (16ull << 0) |
                             (static_cast<uint64_t>(416u * 16u) << 16));
            gs.writeRegister(GS_REG_XYZ2,
                             (16ull << 0) |
                             (static_cast<uint64_t>(415u * 16u) << 16));

            const uint32_t pixel = gs.ReadVram(
                GS_PSM_CT32,
                kFramePage << 5u,
                1u,
                0u,
                363u);
            t.Equals(pixel, 0x8000005Fu,
                     "row 363 should retain bilinear weight 6 after 363 vertical DDA steps");
        });

        tc.Run("GS FST triangles retain low 16.16 bits through scanline setup", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kTexTbp = 64u;
            constexpr uint32_t kFramePage = 2u;
            constexpr uint64_t kFrame =
                (static_cast<uint64_t>(kFramePage) << 0) |
                (4ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = 1ull << 32;
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (255ull << 16) |
                (0ull << 32) |
                (127ull << 48);
            constexpr uint64_t kTex0 =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (4ull << 14) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
                (8ull << 26) |
                (7ull << 30) |
                (1ull << 34) |
                (1ull << 35);
            constexpr uint64_t kTex1 =
                (1ull << 5) |
                (1ull << 6);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_TRIANGLE) |
                (1ull << 4) |
                (1ull << 8);
            constexpr uint64_t kRgbaq =
                0x80ull |
                (0x80ull << 8) |
                (0x80ull << 16) |
                (0x80ull << 24) |
                (0x3F800000ull << 32);
            auto xyz = [](uint16_t x, uint16_t y) -> uint64_t
            {
                return static_cast<uint64_t>(x) |
                       (static_cast<uint64_t>(y) << 16);
            };
            auto uv = [](uint16_t u, uint16_t v) -> uint64_t
            {
                return static_cast<uint64_t>(u) |
                       (static_cast<uint64_t>(v) << 16);
            };

            for (uint32_t y : {57u, 58u})
            {
                gs.WriteVram(GS_PSM_CT32, kTexTbp, 4u, 45u, y,
                             0x80000000u);
                gs.WriteVram(GS_PSM_CT32, kTexTbp, 4u, 46u, y,
                             0x800000FFu);
            }

            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_TEX0_1, kTex0);
            gs.writeRegister(GS_REG_TEX1_1, kTex1);
            gs.writeRegister(GS_REG_CLAMP_1, 5ull);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, kRgbaq);

            gs.writeRegister(GS_REG_UV, uv(0u, 0u));
            gs.writeRegister(GS_REG_XYZ2, xyz(120u, 120u));
            gs.writeRegister(GS_REG_UV, uv(256u * 16u, 0u));
            gs.writeRegister(GS_REG_XYZ2, xyz(3192u, 120u));
            gs.writeRegister(GS_REG_UV, uv(0u, 128u * 16u));
            gs.writeRegister(GS_REG_XYZ2, xyz(120u, 1656u));

            const uint32_t pixel = gs.ReadVram(
                GS_PSM_CT32,
                kFramePage << 5u,
                4u,
                42u,
                51u);
            t.Equals(pixel, 0x8000006Fu,
                     "independent lane and block truncation should produce bilinear U weight 7");
        });

        tc.Run("GS CLAMP modes transform out-of-range texture coordinates", [](TestCase &t)
        {
            auto renderSamplePixel = [](uint64_t clampReg) -> uint32_t
            {
                std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
                GS gs;
                gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

                constexpr uint32_t kTexTbp = 64u;
                constexpr uint64_t kFrame =
                    (0ull << 0) |
                    (1ull << 16) |
                    (static_cast<uint64_t>(GS_PSM_CT32) << 24);
                constexpr uint64_t kZbuf = (1ull << 32);
                constexpr uint64_t kScissor = 0ull;
                constexpr uint64_t kTex0 =
                    (static_cast<uint64_t>(kTexTbp) << 0) |
                    (1ull << 14) |
                    (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
                    (3ull << 26) |
                    (0ull << 30) |
                    (1ull << 34) |
                    (1ull << 35);
                constexpr uint64_t kPrim =
                    static_cast<uint64_t>(GS_PRIM_TRIANGLE) |
                    (1ull << 4) |
                    (1ull << 8);
                constexpr uint64_t kRgbaq = 0x3F80000080808080ull;
                constexpr uint16_t kOutOfRangeU = 10u * 16u;

                for (uint32_t x = 0u; x < 8u; ++x)
                {
                    const uint32_t color = 0x80000010u + x;
                    const uint32_t offset = GSPSMCT32::addrPSMCT32(kTexTbp, 1u, x, 0u);
                    std::memcpy(vram.data() + offset, &color, sizeof(color));
                }

                gs.writeRegister(GS_REG_FRAME_1, kFrame);
                gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
                gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
                gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
                gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
                gs.writeRegister(GS_REG_TEX0_1, kTex0);
                gs.writeRegister(GS_REG_CLAMP_1, clampReg);
                gs.writeRegister(GS_REG_PRIM, kPrim);
                gs.writeRegister(GS_REG_RGBAQ, kRgbaq);

                gs.writeRegister(GS_REG_UV, kOutOfRangeU);
                gs.writeRegister(GS_REG_XYZ2, 0ull);
                gs.writeRegister(GS_REG_UV, kOutOfRangeU);
                gs.writeRegister(GS_REG_XYZ2, 32ull);
                gs.writeRegister(GS_REG_UV, kOutOfRangeU);
                gs.writeRegister(GS_REG_XYZ2, (32ull << 16));

                return readReferencePSMCT32Pixel(vram, 0u, 1u, 0u, 0u);
            };

            constexpr uint64_t kRepeat = 0ull;
            constexpr uint64_t kClamp = 1ull;
            constexpr uint64_t kRegionClamp =
                2ull |
                (3ull << 4) |
                (5ull << 14);
            constexpr uint64_t kRegionRepeat =
                3ull |
                (6ull << 4) |
                (1ull << 14);

            t.Equals(renderSamplePixel(kRepeat), 0x80000012u,
                     "REPEAT should wrap coordinate 10 to texel 2 in an eight-wide texture");
            t.Equals(renderSamplePixel(kClamp), 0x80000017u,
                     "CLAMP should saturate coordinate 10 to the last texel");
            t.Equals(renderSamplePixel(kRegionClamp), 0x80000015u,
                     "REGION_CLAMP should saturate coordinate 10 to MAXU");
            t.Equals(renderSamplePixel(kRegionRepeat), 0x80000013u,
                     "REGION_REPEAT should apply (coordinate & MINU) | MAXU");
        });

        tc.Run("GS STQ triangles interpolate S T and Q before perspective division", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kTexTbp = 64u;
            constexpr uint64_t kFrame =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (3ull << 16) |
                (0ull << 32) |
                (3ull << 48);
            constexpr uint64_t kTex0 =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 20) |
                (2ull << 26) |
                (0ull << 30) |
                (1ull << 34) |
                (1ull << 35);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_TRIANGLE) |
                (1ull << 4);

            for (uint32_t x = 0u; x < 4u; ++x)
            {
                const uint32_t color = 0x80000020u + x;
                const uint32_t offset = GSPSMCT32::addrPSMCT32(kTexTbp, 1u, x, 0u);
                std::memcpy(vram.data() + offset, &color, sizeof(color));
            }

            auto packFloat = [](float value) -> uint32_t
            {
                uint32_t bits = 0u;
                std::memcpy(&bits, &value, sizeof(bits));
                return bits;
            };
            auto packSt = [&](float s, float tValue) -> uint64_t
            {
                return static_cast<uint64_t>(packFloat(s)) |
                       (static_cast<uint64_t>(packFloat(tValue)) << 32);
            };
            auto packRgbaq = [&](float q) -> uint64_t
            {
                return 0x80808080ull |
                       (static_cast<uint64_t>(packFloat(q)) << 32);
            };

            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_TEX0_1, kTex0);
            gs.writeRegister(GS_REG_CLAMP_1, 5ull);
            gs.writeRegister(GS_REG_PRIM, kPrim);

            gs.writeRegister(GS_REG_RGBAQ, packRgbaq(1.0f));
            gs.writeRegister(GS_REG_ST, packSt(0.0f, 0.0f));
            gs.writeRegister(GS_REG_XYZ2, 0ull);

            gs.writeRegister(GS_REG_RGBAQ, packRgbaq(4.0f));
            gs.writeRegister(GS_REG_ST, packSt(4.0f, 0.0f));
            gs.writeRegister(GS_REG_XYZ2, 64ull);

            gs.writeRegister(GS_REG_RGBAQ, packRgbaq(1.0f));
            gs.writeRegister(GS_REG_ST, packSt(0.0f, 0.0f));
            gs.writeRegister(GS_REG_XYZ2, (64ull << 16));

            const uint32_t pixel = readReferencePSMCT32Pixel(vram, 0u, 1u, 1u, 0u);
            t.Equals(pixel, 0x80000022u,
                     "pixel (1,0) should sample texel 2 after interpolating S=1.5 and Q=2.125");
        });

        tc.Run("GS disabled depth testing passes pixels without updating depth", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kZPage = 1u;
            constexpr uint32_t kInitialZ = 0x12345678u;
            constexpr uint64_t kFrame =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf =
                (static_cast<uint64_t>(kZPage) << 0) |
                (0ull << 24) |         // PSMZ32
                (0ull << 32);          // ZMSK disabled
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (0ull << 16) |
                (0ull << 32) |
                (0ull << 48);
            constexpr uint64_t kTest = 0ull; // ZTE disabled; ZTST bits are ignored.
            constexpr uint64_t kPrim = static_cast<uint64_t>(GS_PRIM_POINT);
            constexpr uint64_t kRgbaq =
                (0x12ull << 0) |
                (0x34ull << 8) |
                (0x56ull << 16) |
                (0x78ull << 24) |
                (0x3F800000ull << 32);

            const uint32_t zBase = kZPage << 5u;
            gs.WriteVram(GS_PSM_Z32, zBase, 1u, 0u, 0u, kInitialZ);

            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
            gs.writeRegister(GS_REG_TEST_1, kTest);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, kRgbaq);
            gs.writeRegister(GS_REG_XYZ2, 0ull);

            const uint32_t pixel = gs.ReadVram(GS_PSM_CT32, 0u, 1u, 0u, 0u);
            const uint32_t depth = gs.ReadVram(GS_PSM_Z32, zBase, 1u, 0u, 0u);
            t.Equals(pixel, 0x78563412u,
                     "ZTE=0 should make the depth comparison pass");
            t.Equals(depth, kInitialZ,
                     "ZTE=0 should suppress depth-buffer writes");
        });

        tc.Run("GS destination alpha test masks 32-bit framebuffer and depth writes", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kZPage = 1u;
            constexpr uint32_t kInitialZ = 0x11223344u;
            constexpr uint32_t kNewZ = 0x55667788u;
            constexpr uint32_t kAlphaClear = 0x00112233u;
            constexpr uint32_t kAlphaSet = 0x80112233u;
            constexpr uint64_t kFrame =
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf =
                (static_cast<uint64_t>(kZPage) << 0) |
                (static_cast<uint64_t>(GS_PSM_Z32) << 24);
            constexpr uint64_t kTest =
                (1ull << 14) | // DATE
                (1ull << 15) | // DATM: destination alpha MSB must be set
                (1ull << 16) | // ZTE
                (1ull << 17);  // ZTST = ALWAYS
            constexpr uint64_t kRgbaq =
                0x12ull |
                (0x34ull << 8) |
                (0x56ull << 16) |
                (0x78ull << 24) |
                (0x3F800000ull << 32);

            const uint32_t zBase = kZPage << 5u;
            gs.WriteVram(GS_PSM_CT32, 0u, 1u, 0u, 0u, kAlphaClear);
            gs.WriteVram(GS_PSM_CT32, 0u, 1u, 1u, 0u, kAlphaSet);
            gs.WriteVram(GS_PSM_Z32, zBase, 1u, 0u, 0u, kInitialZ);
            gs.WriteVram(GS_PSM_Z32, zBase, 1u, 1u, 0u, kInitialZ);

            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, 1ull << 16);
            gs.writeRegister(GS_REG_TEST_1, kTest);
            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT));
            gs.writeRegister(GS_REG_RGBAQ, kRgbaq);
            gs.writeRegister(GS_REG_XYZ2, static_cast<uint64_t>(kNewZ) << 32);
            gs.writeRegister(GS_REG_XYZ2,
                             16ull | (static_cast<uint64_t>(kNewZ) << 32));

            t.Equals(gs.ReadVram(GS_PSM_CT32, 0u, 1u, 0u, 0u),
                     kAlphaClear,
                     "DATM=1 should reject a PSMCT32 destination with alpha MSB clear");
            t.Equals(gs.ReadVram(GS_PSM_Z32, zBase, 1u, 0u, 0u),
                     kInitialZ,
                     "destination alpha failure should also suppress the depth write");
            t.Equals(gs.ReadVram(GS_PSM_CT32, 0u, 1u, 1u, 0u),
                     0x78563412u,
                     "DATM=1 should accept a PSMCT32 destination with alpha MSB set");
            t.Equals(gs.ReadVram(GS_PSM_Z32, zBase, 1u, 1u, 0u),
                     kNewZ,
                     "a passing destination alpha test should allow the depth write");
        });

        tc.Run("GS destination alpha test uses the PSMCT16 alpha bit", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint16_t kAlphaClear = 0x001Fu;
            constexpr uint16_t kAlphaSet = 0x801Fu;
            constexpr uint64_t kFrame =
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT16) << 24);
            constexpr uint64_t kZbuf = 1ull << 32;
            constexpr uint64_t kTest =
                (1ull << 14) | // DATE
                (0ull << 15);  // DATM: destination alpha bit must be clear
            constexpr uint64_t kRgbaq =
                0x20ull |
                (0x40ull << 8) |
                (0x60ull << 16) |
                (0x80ull << 24) |
                (0x3F800000ull << 32);

            gs.WriteVram(GS_PSM_CT16, 0u, 1u, 0u, 0u, kAlphaClear);
            gs.WriteVram(GS_PSM_CT16, 0u, 1u, 1u, 0u, kAlphaSet);
            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, 1ull << 16);
            gs.writeRegister(GS_REG_TEST_1, kTest);
            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT));
            gs.writeRegister(GS_REG_RGBAQ, kRgbaq);
            gs.writeRegister(GS_REG_XYZ2, 0ull);
            gs.writeRegister(GS_REG_XYZ2, 16ull);

            t.Equals(gs.ReadVram(GS_PSM_CT16, 0u, 1u, 0u, 0u),
                     static_cast<uint32_t>(0xB104u),
                     "DATM=0 should accept a PSMCT16 destination with alpha clear");
            t.Equals(gs.ReadVram(GS_PSM_CT16, 0u, 1u, 1u, 0u),
                     static_cast<uint32_t>(kAlphaSet),
                     "DATM=0 should reject a PSMCT16 destination with alpha set");
        });

        tc.Run("GS fog blends RGB after shading and preserves alpha", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint64_t kFrame =
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = 1ull << 32;
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_POINT) |
                (1ull << 5); // FGE
            constexpr uint64_t kRgbaq =
                200ull |
                (100ull << 8) |
                (40ull << 16) |
                (0xA5ull << 24) |
                (0x3F800000ull << 32);
            constexpr uint64_t kFogColor =
                20ull |
                (60ull << 8) |
                (100ull << 16);
            constexpr uint8_t kFog = 64u;
            constexpr uint64_t kXyzf2 =
                static_cast<uint64_t>(kFog) << 56;

            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0ull);
            gs.writeRegister(GS_REG_FOGCOL, kFogColor);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, kRgbaq);
            gs.writeRegister(GS_REG_XYZF2, kXyzf2);

            const uint32_t pixel = gs.ReadVram(GS_PSM_CT32, 0u, 1u, 0u, 0u);
            t.Equals(pixel,
                     0xA5554641u,
                     "F=64 should mix source RGB toward FOGCOL with 8-bit fixed-point weights while leaving alpha unchanged");
        });

        tc.Run("GS sprite fog uses the second vertex", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint64_t kFrame =
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = 1ull << 32;
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 5); // FGE
            constexpr uint64_t kRgbaq =
                0xC0ull |
                (0x80ull << 8) |
                (0x40ull << 16) |
                (0x7Eull << 24) |
                (0x3F800000ull << 32);
            constexpr uint64_t kFogColor =
                0x10ull |
                (0x20ull << 8) |
                (0x30ull << 16);

            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, 1ull << 16);
            gs.writeRegister(GS_REG_TEST_1, 0ull);
            gs.writeRegister(GS_REG_FOGCOL, kFogColor);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, kRgbaq);
            gs.writeRegister(GS_REG_XYZF2, 255ull << 56);
            gs.writeRegister(GS_REG_XYZF2,
                             16ull |
                                 (16ull << 16) |
                                 (0ull << 32) |
                                 (0ull << 56));

            const uint32_t pixel = gs.ReadVram(GS_PSM_CT32, 0u, 1u, 0u, 0u);
            t.Equals(pixel,
                     0x7E302010u,
                     "a sprite must use its second vertex's fog coefficient for every fragment");
        });

        tc.Run("GS alpha test AFAIL framebuffer-only still writes the pixel", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kZPage = 1u;
            constexpr uint32_t kInitialZ = 0x11223344u;
            constexpr uint32_t kNewZ = 0x55667788u;
            constexpr uint64_t kFrame =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf =
                (static_cast<uint64_t>(kZPage) << 0) |
                (static_cast<uint64_t>(GS_PSM_Z32) << 24);
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (0ull << 16) |
                (0ull << 32) |
                (0ull << 48);
            constexpr uint64_t kTest =
                1ull |                  // ATE
                (5ull << 1) |          // ATST = GEQUAL
                (0x80ull << 4) |       // AREF
                (1ull << 12) |         // AFAIL = FB_ONLY
                (1ull << 16) |         // ZTE
                (1ull << 17);          // ZTST = ALWAYS
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_POINT);
            constexpr uint64_t kRgbaq =
                (0x12ull << 0) |
                (0x34ull << 8) |
                (0x56ull << 16) |
                (0x00ull << 24) |
                (0x3F800000ull << 32); // q = 1.0f

            const uint32_t zBase = kZPage << 5u;
            gs.WriteVram(GS_PSM_Z32, zBase, 1u, 0u, 0u, kInitialZ);
            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
            gs.writeRegister(GS_REG_TEST_1, kTest);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, kRgbaq);
            gs.writeRegister(GS_REG_XYZ2, static_cast<uint64_t>(kNewZ) << 32);

            uint32_t pixel = 0u;
            std::memcpy(&pixel, vram.data(), sizeof(pixel));
            const uint32_t depth = gs.ReadVram(GS_PSM_Z32, zBase, 1u, 0u, 0u);
            t.Equals(pixel, 0x00563412u,
                     "AFAIL=FB_ONLY should still update the framebuffer when the alpha test fails");
            t.Equals(depth, kInitialZ,
                     "AFAIL=FB_ONLY should suppress the depth-buffer write");
        });

        tc.Run("GS alpha test AFAIL RGB-only preserves destination alpha", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kZPage = 1u;
            constexpr uint32_t kInitialZ = 0x11223344u;
            constexpr uint32_t kNewZ = 0x55667788u;
            constexpr uint64_t kFrame =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf =
                (static_cast<uint64_t>(kZPage) << 0) |
                (static_cast<uint64_t>(GS_PSM_Z32) << 24);
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (0ull << 16) |
                (0ull << 32) |
                (0ull << 48);
            constexpr uint64_t kTest =
                1ull |                // ATE
                (5ull << 1) |         // ATST = GEQUAL
                (0x80ull << 4) |      // AREF
                (3ull << 12) |        // AFAIL = RGB_ONLY
                (1ull << 16) |        // ZTE
                (1ull << 17);         // ZTST = ALWAYS
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_POINT);
            constexpr uint64_t kRgbaq =
                (0x12ull << 0) |
                (0x34ull << 8) |
                (0x56ull << 16) |
                (0x00ull << 24) |
                (0x3F800000ull << 32); // q = 1.0f
            constexpr uint32_t kExisting = 0xAB030201u;

            std::memcpy(vram.data(), &kExisting, sizeof(kExisting));
            const uint32_t zBase = kZPage << 5u;
            gs.WriteVram(GS_PSM_Z32, zBase, 1u, 0u, 0u, kInitialZ);

            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
            gs.writeRegister(GS_REG_TEST_1, kTest);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, kRgbaq);
            gs.writeRegister(GS_REG_XYZ2, static_cast<uint64_t>(kNewZ) << 32);

            uint32_t pixel = 0u;
            std::memcpy(&pixel, vram.data(), sizeof(pixel));
            const uint32_t depth = gs.ReadVram(GS_PSM_Z32, zBase, 1u, 0u, 0u);
            t.Equals(pixel, 0xAB563412u,
                     "AFAIL=RGB_ONLY should update RGB while preserving destination alpha");
            t.Equals(depth, kInitialZ,
                     "AFAIL=RGB_ONLY should suppress the depth-buffer write");
        });

        tc.Run("GS alpha test AFAIL Z-buffer-only preserves color and writes depth", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kZPage = 1u;
            constexpr uint32_t kInitialZ = 0x11223344u;
            constexpr uint32_t kNewZ = 0x55667788u;
            constexpr uint32_t kExisting = 0xAB030201u;
            constexpr uint64_t kFrame =
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf =
                (static_cast<uint64_t>(kZPage) << 0) |
                (static_cast<uint64_t>(GS_PSM_Z32) << 24);
            constexpr uint64_t kTest =
                1ull |                // ATE
                (5ull << 1) |         // ATST = GEQUAL
                (0x80ull << 4) |      // AREF
                (2ull << 12) |        // AFAIL = ZB_ONLY
                (1ull << 16) |        // ZTE
                (1ull << 17);         // ZTST = ALWAYS
            constexpr uint64_t kRgbaq =
                0x12ull |
                (0x34ull << 8) |
                (0x56ull << 16) |
                (0x3F800000ull << 32);

            std::memcpy(vram.data(), &kExisting, sizeof(kExisting));
            const uint32_t zBase = kZPage << 5u;
            gs.WriteVram(GS_PSM_Z32, zBase, 1u, 0u, 0u, kInitialZ);
            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, kTest);
            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT));
            gs.writeRegister(GS_REG_RGBAQ, kRgbaq);
            gs.writeRegister(GS_REG_XYZ2, static_cast<uint64_t>(kNewZ) << 32);

            uint32_t pixel = 0u;
            std::memcpy(&pixel, vram.data(), sizeof(pixel));
            const uint32_t depth = gs.ReadVram(GS_PSM_Z32, zBase, 1u, 0u, 0u);
            t.Equals(pixel, kExisting,
                     "AFAIL=ZB_ONLY should preserve the framebuffer");
            t.Equals(depth, kNewZ,
                     "AFAIL=ZB_ONLY should update the depth buffer");
        });

        tc.Run("GS alpha test AFAIL keep preserves color and depth", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kZPage = 1u;
            constexpr uint32_t kInitialZ = 0x11223344u;
            constexpr uint32_t kNewZ = 0x55667788u;
            constexpr uint32_t kExisting = 0xAB030201u;
            constexpr uint64_t kFrame =
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf =
                (static_cast<uint64_t>(kZPage) << 0) |
                (static_cast<uint64_t>(GS_PSM_Z32) << 24);
            constexpr uint64_t kTest =
                1ull |                // ATE
                (5ull << 1) |         // ATST = GEQUAL
                (0x80ull << 4) |      // AREF
                (0ull << 12) |        // AFAIL = KEEP
                (1ull << 16) |        // ZTE
                (1ull << 17);         // ZTST = ALWAYS
            constexpr uint64_t kRgbaq =
                0x12ull |
                (0x34ull << 8) |
                (0x56ull << 16) |
                (0x3F800000ull << 32);

            std::memcpy(vram.data(), &kExisting, sizeof(kExisting));
            const uint32_t zBase = kZPage << 5u;
            gs.WriteVram(GS_PSM_Z32, zBase, 1u, 0u, 0u, kInitialZ);
            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, kTest);
            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_POINT));
            gs.writeRegister(GS_REG_RGBAQ, kRgbaq);
            gs.writeRegister(GS_REG_XYZ2, static_cast<uint64_t>(kNewZ) << 32);

            uint32_t pixel = 0u;
            std::memcpy(&pixel, vram.data(), sizeof(pixel));
            const uint32_t depth = gs.ReadVram(GS_PSM_Z32, zBase, 1u, 0u, 0u);
            t.Equals(pixel, kExisting,
                     "AFAIL=KEEP should preserve the framebuffer");
            t.Equals(depth, kInitialZ,
                     "AFAIL=KEEP should preserve the depth buffer");
        });

        tc.Run("GS triangle fan subpixel quad fills rows without interior holes", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint64_t kFrame =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (31ull << 16) |
                (0ull << 32) |
                (31ull << 48);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_TRIFAN);
            constexpr uint64_t kRgbaq =
                0xFFull |
                (0xFFull << 8) |
                (0xFFull << 16) |
                (0x80ull << 24) |
                (0x3F800000ull << 32); // q = 1.0f
            auto makeXyzf = [](uint16_t x, uint16_t y) -> uint64_t
            {
                return static_cast<uint64_t>(x) |
                       (static_cast<uint64_t>(y) << 16);
            };

            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, kRgbaq);
            gs.writeRegister(GS_REG_XYZF2, makeXyzf(102u, 102u));
            gs.writeRegister(GS_REG_XYZF2, makeXyzf(420u, 102u));
            gs.writeRegister(GS_REG_XYZF2, makeXyzf(420u, 420u));
            gs.writeRegister(GS_REG_XYZF2, makeXyzf(102u, 420u));

            bool sawFilledRow = false;
            for (uint32_t y = 6u; y <= 26u; ++y)
            {
                int first = -1;
                int last = -1;
                for (uint32_t x = 6u; x <= 26u; ++x)
                {
                    const uint32_t pixel = gs.ReadVram(GS_PSM_CT32, 0u, 1u, x, y);
                    if ((pixel & 0x00FFFFFFu) != 0u)
                    {
                        if (first < 0)
                        {
                            first = static_cast<int>(x);
                        }
                        last = static_cast<int>(x);
                    }
                }

                if (first < 0 || last < 0)
                {
                    continue;
                }

                sawFilledRow = true;
                for (int x = first; x <= last; ++x)
                {
                    const uint32_t pixel =
                        gs.ReadVram(GS_PSM_CT32, 0u, 1u, static_cast<uint32_t>(x), y);
                    if ((pixel & 0x00FFFFFFu) == 0u)
                    {
                        t.Fail("triangle fan quad should not leave interior holes within a covered row");
                        break;
                    }
                }
            }

            t.IsTrue(sawFilledRow,
                     "triangle fan quad should light at least one framebuffer row");
        });

        tc.Run("GS top-left triangle rule draws shared blended edges exactly once", [](TestCase &t)
        {
            constexpr uint64_t kFrame =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (7ull << 16) |
                (0ull << 32) |
                (7ull << 48);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_TRIANGLE) |
                (1ull << 6);
            constexpr uint64_t kAlpha =
                (0ull << 0) | // A = source
                (1ull << 2) | // B = destination
                (0ull << 4) | // C = source alpha
                (1ull << 6);  // D = destination
            constexpr uint64_t kRgbaq =
                0xFFull |
                (0xFFull << 8) |
                (0xFFull << 16) |
                (0x40ull << 24) |
                (0x3F800000ull << 32);
            constexpr uint32_t kOnceBlended = 0x407F7F7Fu;

            auto makeXyz = [](uint16_t x, uint16_t y) -> uint64_t
            {
                return static_cast<uint64_t>(x * 16u) |
                       (static_cast<uint64_t>(y * 16u) << 16);
            };

            const uint64_t vertices[2][6] = {
                {
                    makeXyz(0u, 0u), makeXyz(4u, 0u), makeXyz(0u, 4u),
                    makeXyz(4u, 0u), makeXyz(4u, 4u), makeXyz(0u, 4u),
                },
                {
                    makeXyz(0u, 4u), makeXyz(4u, 0u), makeXyz(0u, 0u),
                    makeXyz(0u, 4u), makeXyz(4u, 4u), makeXyz(4u, 0u),
                },
            };

            for (const auto &winding : vertices)
            {
                std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
                GS gs;
                gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

                gs.writeRegister(GS_REG_FRAME_1, kFrame);
                gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
                gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
                gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
                gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
                gs.writeRegister(GS_REG_ALPHA_1, kAlpha);
                gs.writeRegister(GS_REG_PRIM, kPrim);
                gs.writeRegister(GS_REG_RGBAQ, kRgbaq);

                for (uint64_t xyz : winding)
                    gs.writeRegister(GS_REG_XYZ2, xyz);

                for (uint32_t y = 0u; y < 4u; ++y)
                {
                    for (uint32_t x = 0u; x < 4u; ++x)
                    {
                        t.Equals(gs.ReadVram(GS_PSM_CT32, 0u, 1u, x, y),
                                 kOnceBlended,
                                 "every pixel in two side-sharing triangles should be blended exactly once");
                    }
                }

                t.Equals(gs.ReadVram(GS_PSM_CT32, 0u, 1u, 4u, 0u), 0u,
                         "the right edge should be excluded");
                t.Equals(gs.ReadVram(GS_PSM_CT32, 0u, 1u, 0u, 4u), 0u,
                         "the bottom edge should be excluded");
            }
        });

        tc.Run("GS skipped triangle-strip vertices advance primitive assembly", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint64_t kFrame =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kScissor =
                (0ull << 0) |
                (31ull << 16) |
                (0ull << 32) |
                (31ull << 48);
            constexpr uint64_t kRgbaq =
                0xFFull |
                (0xFFull << 8) |
                (0xFFull << 16) |
                (0x80ull << 24) |
                (0x3F800000ull << 32);
            auto makeXyz = [](uint16_t x, uint16_t y) -> uint64_t
            {
                return static_cast<uint64_t>(x * 16u) |
                       (static_cast<uint64_t>(y * 16u) << 16);
            };

            gs.writeRegister(GS_REG_FRAME_1, kFrame);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, kScissor);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_TRISTRIP));
            gs.writeRegister(GS_REG_RGBAQ, kRgbaq);

            // A-B-C would cover the upper-right half of the square. XYZ3
            // suppresses that triangle, but C must remain in the strip so the
            // following kick draws B-C-D (the lower-right half).
            gs.writeRegister(GS_REG_XYZ2, makeXyz(1u, 1u));   // A
            gs.writeRegister(GS_REG_XYZ2, makeXyz(20u, 1u));  // B
            gs.writeRegister(GS_REG_XYZ3, makeXyz(20u, 20u)); // C, ADC/skip

            t.Equals(gs.ReadVram(GS_PSM_CT32, 0u, 1u, 16u, 3u), 0u,
                     "XYZ3 should suppress the completed strip triangle");

            gs.writeRegister(GS_REG_XYZ2, makeXyz(1u, 20u)); // D

            const uint32_t upperRight = gs.ReadVram(GS_PSM_CT32, 0u, 1u, 16u, 3u);
            const uint32_t lowerRight = gs.ReadVram(GS_PSM_CT32, 0u, 1u, 16u, 16u);
            t.Equals(upperRight, 0u,
                     "the skipped A-B-C triangle should remain suppressed");
            t.IsTrue((lowerRight & 0x00FFFFFFu) != 0u,
                     "the next strip triangle should use B-C-D after the skipped kick");
        });

        tc.Run("sceGs load-image stubs use the Sony setup-packet ABI and literal DBP units", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "runtime memory initialize should succeed");
            uint8_t *const rdram = runtime.memory().getRDRAM();
            constexpr uint32_t kPacketAddr = 0x4000u;
            constexpr uint32_t kSrcAddr = 0x5000u;
            constexpr uint32_t kDstAddr = 0x6000u;
            constexpr uint32_t kStoreImageAddr = 0x7000u;
            constexpr uint32_t kDbp = 8u;

            const uint8_t pixels[16] = {
                0x10u, 0x20u, 0x30u, 0x40u,
                0x50u, 0x60u, 0x70u, 0x80u,
                0x90u, 0xA0u, 0xB0u, 0xC0u,
                0xD0u, 0xE0u, 0xF0u, 0xFFu,
            };

            R5900Context defineCtx{};
            setRegU32(defineCtx, 4, kPacketAddr);
            setRegU32(defineCtx, 5, kDbp);
            setRegU32(defineCtx, 6, 1u);
            setRegU32(defineCtx, 7, GS_PSM_CT32);
            setRegU32(defineCtx, 8, 0u);
            setRegU32(defineCtx, 9, 0u);
            setRegU32(defineCtx, 10, 2u);
            setRegU32(defineCtx, 11, 2u);
            ps2_stubs::sceGsSetDefLoadImage(rdram, &defineCtx, &runtime);
            t.Equals(static_cast<int32_t>(getRegU32Test(defineCtx, 2)), 6,
                     "sceGsSetDefLoadImage should return the six-QW setup packet size");

            uint64_t setupTag = 0u;
            uint64_t bitbltbuf = 0u;
            uint64_t bitbltbufReg = 0u;
            uint64_t trxposReg = 0u;
            uint64_t trxregReg = 0u;
            uint64_t trxdirReg = 0u;
            uint64_t imageTag = 0u;
            std::memcpy(&setupTag, rdram + kPacketAddr, sizeof(setupTag));
            std::memcpy(&bitbltbuf, rdram + kPacketAddr + 16u, sizeof(bitbltbuf));
            std::memcpy(&bitbltbufReg, rdram + kPacketAddr + 24u, sizeof(bitbltbufReg));
            std::memcpy(&trxposReg, rdram + kPacketAddr + 40u, sizeof(trxposReg));
            std::memcpy(&trxregReg, rdram + kPacketAddr + 56u, sizeof(trxregReg));
            std::memcpy(&trxdirReg, rdram + kPacketAddr + 72u, sizeof(trxdirReg));
            std::memcpy(&imageTag, rdram + kPacketAddr + 80u, sizeof(imageTag));

            t.Equals(setupTag, 0x1000000000008004ull,
                     "load-image setup should begin with four packed A+D writes");
            t.Equals(bitbltbuf,
                     (static_cast<uint64_t>(kDbp) << 32u) | (1ull << 48u),
                     "BITBLTBUF should encode DBP in native 256-byte GS units");
            t.Equals(bitbltbufReg, 0x50ull, "load-image setup should write BITBLTBUF");
            t.Equals(trxposReg, 0x51ull, "load-image setup should write TRXPOS");
            t.Equals(trxregReg, 0x52ull, "load-image setup should write TRXREG");
            t.Equals(trxdirReg, 0x53ull, "load-image setup should write TRXDIR");
            t.Equals(imageTag, 0x0800000000008001ull,
                     "load-image setup should end with a one-QW IMAGE tag");

            std::memcpy(rdram + kSrcAddr, pixels, sizeof(pixels));

            R5900Context loadCtx{};
            setRegU32(loadCtx, 4, kPacketAddr);
            setRegU32(loadCtx, 5, kSrcAddr);
            ps2_stubs::sceGsExecLoadImage(rdram, &loadCtx, &runtime);
            t.Equals(static_cast<int32_t>(getRegU32Test(loadCtx, 2)), 0,
                     "sceGsExecLoadImage should succeed for a simple CT32 upload");

            t.Equals(runtime.gs().ReadVram(GS_PSM_CT32, kDbp, 1u, 0u, 0u), 0x40302010u,
                     "the first uploaded pixel should land at the literal DBP");
            t.Equals(runtime.gs().ReadVram(GS_PSM_CT32, kDbp, 1u, 1u, 0u), 0x80706050u,
                     "the second uploaded pixel should preserve byte order");
            t.Equals(runtime.gs().ReadVram(GS_PSM_CT32, kDbp, 1u, 0u, 1u), 0xC0B0A090u,
                     "the split IMAGE DMA should preserve its second row");
            t.Equals(runtime.gs().ReadVram(GS_PSM_CT32, kDbp, 1u, 1u, 1u), 0xFFF0E0D0u,
                     "the final uploaded pixel should complete the IMAGE payload");

            // Store-image still uses its direct readback representation. Keep
            // that existing coverage while the load path validates the SDK ABI.
            const GsImageMem storeImage{0u, 0u, 2u, 2u, 1u, 1u, 0u};
            writeGsImageTest(rdram, kStoreImageAddr, storeImage);
            R5900Context storeCtx{};
            setRegU32(storeCtx, 4, kStoreImageAddr);
            setRegU32(storeCtx, 5, kDstAddr);
            ps2_stubs::sceGsExecStoreImage(rdram, &storeCtx, &runtime);
            t.Equals(static_cast<int32_t>(getRegU32Test(storeCtx, 2)), 0,
                     "sceGsExecStoreImage should succeed for a matching CT32 readback");
            uint64_t storeTag = 0u;
            std::memcpy(&storeTag, rdram + runtime.guestHeapBase(), sizeof(storeTag));
            t.Equals(storeTag, 0x1000000000008004ull,
                     "sceGsExecStoreImage should populate the packed A+D GIF tag in guest RAM");
            uint64_t storeReg1 = 0u;
            uint64_t storeReg2 = 0u;
            uint64_t storeReg3 = 0u;
            uint64_t storeReg4 = 0u;
            std::memcpy(&storeReg1, rdram + runtime.guestHeapBase() + 24u, sizeof(storeReg1));
            std::memcpy(&storeReg2, rdram + runtime.guestHeapBase() + 40u, sizeof(storeReg2));
            std::memcpy(&storeReg3, rdram + runtime.guestHeapBase() + 56u, sizeof(storeReg3));
            std::memcpy(&storeReg4, rdram + runtime.guestHeapBase() + 72u, sizeof(storeReg4));
            t.Equals(storeReg1, 0x50ull, "sceGsExecStoreImage should encode BITBLTBUF as A+D register 0x50");
            t.Equals(storeReg2, 0x51ull, "sceGsExecStoreImage should encode TRXPOS as A+D register 0x51");
            t.Equals(storeReg3, 0x52ull, "sceGsExecStoreImage should encode TRXREG as A+D register 0x52");
            t.Equals(storeReg4, 0x53ull, "sceGsExecStoreImage should encode TRXDIR as A+D register 0x53");
            expectGuestHeapReusable(t, runtime,
                                    "sceGsExecStoreImage should free its temporary GIF packet");

            bool roundtripOk = true;
            size_t mismatchIndex = 0u;
            for (size_t i = 0; i < sizeof(pixels); ++i)
            {
                if (rdram[kDstAddr + i] != pixels[i])
                {
                    roundtripOk = false;
                    mismatchIndex = i;
                    break;
                }
            }
            if (!roundtripOk)
            {
                t.Fail("load-image packet DMA and store-image readback should roundtrip CT32 pixel data "
                       "(first mismatch at byte " + std::to_string(mismatchIndex) +
                       ", got " + std::to_string(rdram[kDstAddr + mismatchIndex]) +
                       ", expected " + std::to_string(pixels[mismatchIndex]) + ")");
            }
        });

        tc.Run("sceGifPkRefLoadImage seeds A+D GIFtag nloop once (no double-count)", [](TestCase &t)
        {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            PS2Runtime runtime;
            R5900Context ctx{};

            constexpr uint32_t stateAddr = 0x1000u;
            constexpr uint32_t baseAddr = 0x2000u;
            constexpr uint32_t spAddr = 0x8000u;

            setRegU32(ctx, 4, stateAddr);
            setRegU32(ctx, 5, baseAddr);
            ps2_stubs::sceGifPkInit(rdram.data(), &ctx, &runtime);

            setRegU32(ctx, 29, spAddr);
            const uint32_t width = 16u;
            const uint32_t height = 1u;
            std::memcpy(rdram.data() + spAddr, &width, sizeof(width));
            std::memcpy(rdram.data() + spAddr + 8u, &height, sizeof(height));

            const uint32_t dbp = 0x3fc0u;
            const uint32_t dpsm = 0u;
            const uint32_t dbw = 1u;
            const uint32_t dataAddr = 0u;
            const uint32_t dsax = 0u;
            const uint32_t dsay = 0u;

            setRegU32(ctx, 4, stateAddr);
            setRegU32(ctx, 5, dbp);
            setRegU32(ctx, 6, dpsm);
            setRegU32(ctx, 7, dbw);
            setRegU32(ctx, 8, dataAddr);
            setRegU32(ctx, 9, 0u); // qwcRemaining: setup only, no image body
            setRegU32(ctx, 10, dsax);
            setRegU32(ctx, 11, dsay);
            ps2_stubs::sceGifPkRefLoadImage(rdram.data(), &ctx, &runtime);

            uint64_t tagLo = 0u;
            uint64_t tagHi = 0u;
            std::memcpy(&tagLo, rdram.data() + baseAddr + 16u, sizeof(tagLo));
            std::memcpy(&tagHi, rdram.data() + baseAddr + 24u, sizeof(tagHi));
            t.Equals(tagLo, static_cast<uint64_t>(0x1000000000000004ULL),
                      "header GIFtag lo must be nloop=4 nreg=1 A+D eop=0 (double-count would give ...0008)");
            t.Equals(tagHi, static_cast<uint64_t>(0xEULL),
                      "header GIFtag hi must be A+D register descriptor 0xE");

            uint64_t reg1Desc = 0u;
            uint64_t reg2Desc = 0u;
            uint64_t reg3Desc = 0u;
            uint64_t reg4Desc = 0u;
            std::memcpy(&reg1Desc, rdram.data() + baseAddr + 32u + 0u * 16u + 8u, sizeof(reg1Desc));
            std::memcpy(&reg2Desc, rdram.data() + baseAddr + 32u + 1u * 16u + 8u, sizeof(reg2Desc));
            std::memcpy(&reg3Desc, rdram.data() + baseAddr + 32u + 2u * 16u + 8u, sizeof(reg3Desc));
            std::memcpy(&reg4Desc, rdram.data() + baseAddr + 32u + 3u * 16u + 8u, sizeof(reg4Desc));
            t.Equals(reg1Desc, static_cast<uint64_t>(0x50ULL), "first register qword should be BITBLTBUF (0x50)");
            t.Equals(reg2Desc, static_cast<uint64_t>(0x51ULL), "second register qword should be TRXPOS (0x51)");
            t.Equals(reg3Desc, static_cast<uint64_t>(0x52ULL), "third register qword should be TRXREG (0x52)");
            t.Equals(reg4Desc, static_cast<uint64_t>(0x53ULL), "fourth register qword should be TRXDIR (0x53)");

            uint64_t reg1Payload = 0u;
            uint64_t reg2Payload = 0u;
            uint64_t reg3Payload = 0u;
            uint64_t reg4Payload = 0u;
            std::memcpy(&reg1Payload, rdram.data() + baseAddr + 32u + 0u * 16u + 0u, sizeof(reg1Payload));
            std::memcpy(&reg2Payload, rdram.data() + baseAddr + 32u + 1u * 16u + 0u, sizeof(reg2Payload));
            std::memcpy(&reg3Payload, rdram.data() + baseAddr + 32u + 2u * 16u + 0u, sizeof(reg3Payload));
            std::memcpy(&reg4Payload, rdram.data() + baseAddr + 32u + 3u * 16u + 0u, sizeof(reg4Payload));
            // BITBLTBUF: dbp=0x3fc0 (bits 32-45), dbw=1 (bits 48-53), dpsm=0 (bits 56-61).
            t.Equals(reg1Payload, static_cast<uint64_t>(0x00013FC000000000ULL),
                      "BITBLTBUF payload must encode dbp=0x3fc0, dbw=1, dpsm=0");
            // TRXPOS: dsax=0, dsay=0.
            t.Equals(reg2Payload, static_cast<uint64_t>(0x0ULL),
                      "TRXPOS payload must encode dsax=0, dsay=0");
            // TRXREG: width=16 (bits 0-31), height=1 (bits 32-63).
            t.Equals(reg3Payload, static_cast<uint64_t>(0x0000000100000010ULL),
                      "TRXREG payload must encode width=16, height=1");
            // TRXDIR: host-to-local transfer, dir=0.
            t.Equals(reg4Payload, static_cast<uint64_t>(0x0ULL),
                      "TRXDIR payload must encode dir=0 (host-to-local)");
        });

        tc.Run("sceGsResetGraph frees its temporary GIF packet", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "runtime memory initialize should succeed");

            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};
            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 1u);
            setRegU32(ctx, 6, 2u);
            setRegU32(ctx, 7, 1u);
            ps2_stubs::sceGsResetGraph(rdram.data(), &ctx, &runtime);

            t.Equals(static_cast<int32_t>(getRegU32Test(ctx, 2)), 0,
                     "sceGsResetGraph should succeed in reset mode");
            expectGuestHeapReusable(t, runtime,
                                    "sceGsResetGraph should free its temporary GIF packet");
        });

        tc.Run("sceGsSyncV waits on VBlank and reports interlaced field parity", [](TestCase &t)
        {
            notifyRuntimeStop();

            PS2Runtime runtime;
            ps2_stubs::resetGsSyncVCallbackState(
                &runtime);
            t.IsTrue(runtime.memory().initialize(), "runtime memory initialize should succeed");
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            R5900Context resetCtx{};
            setRegU32(resetCtx, 4, 0u);
            setRegU32(resetCtx, 5, 1u);
            setRegU32(resetCtx, 6, 2u);
            setRegU32(resetCtx, 7, 1u);
            ps2_stubs::sceGsResetGraph(rdram.data(), &resetCtx, &runtime);

            R5900Context sync0{};
            ps2_stubs::sceGsSyncV(rdram.data(), &sync0, &runtime);
            t.Equals(static_cast<int32_t>(getRegU32Test(sync0, 2)), 0, "first interlaced sceGsSyncV should report even field");

            R5900Context sync1{};
            ps2_stubs::sceGsSyncV(rdram.data(), &sync1, &runtime);
            t.Equals(static_cast<int32_t>(getRegU32Test(sync1, 2)), 1, "second interlaced sceGsSyncV should report odd field");

            R5900Context resetProgCtx{};
            setRegU32(resetProgCtx, 4, 0u);
            setRegU32(resetProgCtx, 5, 0u);
            setRegU32(resetProgCtx, 6, 2u);
            setRegU32(resetProgCtx, 7, 1u);
            ps2_stubs::sceGsResetGraph(rdram.data(), &resetProgCtx, &runtime);

            R5900Context syncProg{};
            ps2_stubs::sceGsSyncV(rdram.data(), &syncProg, &runtime);
            t.Equals(static_cast<int32_t>(getRegU32Test(syncProg, 2)), 1, "progressive sceGsSyncV should always return one");

            runtime.requestStop();
            notifyRuntimeStop();
            ps2_stubs::resetGsSyncVCallbackState(
                &runtime);
        });


        tc.Run("GS T4HL/T4HH shared-plane upload preserves both index planes via RMW", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kDbp = 64u;
            constexpr uint32_t kDbw = 1u;
            constexpr uint32_t kRrw = 8u;
            constexpr uint32_t kRrh = 8u;
            constexpr uint64_t kRect = (static_cast<uint64_t>(kRrw) << 0) | (static_cast<uint64_t>(kRrh) << 32);

            // Two independent, differing index patterns for the T4HL and T4HH planes.
            auto indexA = [](uint32_t x, uint32_t y) -> uint8_t
            {
                return static_cast<uint8_t>((x * 3u + y * 5u + 1u) & 0xFu);
            };
            auto indexB = [](uint32_t x, uint32_t y) -> uint8_t
            {
                return static_cast<uint8_t>((x * 7u + y * 2u + 9u) & 0xFu);
            };

            auto buildPacked = [&](const auto &indexFn) -> std::vector<uint8_t>
            {
                std::vector<uint8_t> packed((kRrw * kRrh) / 2u, 0u);
                for (uint32_t y = 0; y < kRrh; ++y)
                {
                    for (uint32_t x = 0; x < kRrw; x += 2u)
                    {
                        const uint8_t lo = indexFn(x, y) & 0xFu;
                        const uint8_t hi = indexFn(x + 1u, y) & 0xFu;
                        packed[(y * kRrw + x) / 2u] = static_cast<uint8_t>(lo | (hi << 4));
                    }
                }
                return packed;
            };

            const std::vector<uint8_t> packedA = buildPacked(indexA);
            const std::vector<uint8_t> packedB = buildPacked(indexB);

            constexpr uint64_t kUploadHLBitblt =
                (static_cast<uint64_t>(kDbp) << 32) |
                (static_cast<uint64_t>(kDbw) << 48) |
                (static_cast<uint64_t>(GS_PSM_T4HL) << 56);
            constexpr uint64_t kUploadHHBitblt =
                (static_cast<uint64_t>(kDbp) << 32) |
                (static_cast<uint64_t>(kDbw) << 48) |
                (static_cast<uint64_t>(GS_PSM_T4HH) << 56);

            gs.writeRegister(GS_REG_BITBLTBUF, kUploadHLBitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, kRect);
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            std::vector<uint8_t> packetA;
            appendU64(packetA, makeGifTag(static_cast<uint16_t>(packedA.size() / 16u), GIF_FMT_IMAGE, 0u, true));
            appendU64(packetA, 0ull);
            packetA.insert(packetA.end(), packedA.begin(), packedA.end());
            gs.processGIFPacket(packetA.data(), static_cast<uint32_t>(packetA.size()));

            gs.writeRegister(GS_REG_BITBLTBUF, kUploadHHBitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, kRect);
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            std::vector<uint8_t> packetB;
            appendU64(packetB, makeGifTag(static_cast<uint16_t>(packedB.size() / 16u), GIF_FMT_IMAGE, 0u, true));
            appendU64(packetB, 0ull);
            packetB.insert(packetB.end(), packedB.begin(), packedB.end());
            gs.processGIFPacket(packetB.data(), static_cast<uint32_t>(packetB.size()));

            bool planesMatch = true;
            bool memReadersMatch = true;
            for (uint32_t y = 0; y < kRrh; ++y)
            {
                for (uint32_t x = 0; x < kRrw; ++x)
                {
                    const uint32_t off = GSPSMCT32::addrPSMCT32(kDbp, kDbw, x, y);
                    uint32_t word = 0u;
                    std::memcpy(&word, vram.data() + off, sizeof(word));

                    const uint8_t expectedA = indexA(x, y);
                    const uint8_t expectedB = indexB(x, y);
                    const uint8_t gotA = static_cast<uint8_t>((word >> 24) & 0xFu);
                    const uint8_t gotB = static_cast<uint8_t>((word >> 28) & 0xFu);
                    if (gotA != expectedA || gotB != expectedB)
                        planesMatch = false;

                    const uint32_t memA = GSMem::ReadP4HL(vram.data(), kDbp, kDbw, x, y);
                    const uint32_t memB = GSMem::ReadP4HH(vram.data(), kDbp, kDbw, x, y);
                    if (memA != expectedA || memB != expectedB)
                        memReadersMatch = false;
                }
            }
            t.IsTrue(planesMatch,
                     "T4HL and T4HH uploads to the same shared CT32 word must not clobber each other's nibble");
            t.IsTrue(memReadersMatch,
                     "GSMem::ReadP4HL/ReadP4HH should agree with the raw shared-word nibble extraction");

            // --- T8H coverage: full-byte upload, round-trip via GSMem::ReadP8H, and the
            // --- clobber interaction when a later T4HL nibble upload lands on the same word.

            constexpr uint32_t kDbpT8H = 128u;
            constexpr uint32_t kDbwT8H = 1u;

            // Full 0..255 range so both nibbles of the uploaded byte vary independently.
            auto byteT8H = [](uint32_t x, uint32_t y) -> uint8_t
            {
                return static_cast<uint8_t>((x * 11u + y * 13u + 7u) & 0xFFu);
            };

            std::vector<uint8_t> packedT8H(kRrw * kRrh, 0u);
            for (uint32_t y = 0; y < kRrh; ++y)
            {
                for (uint32_t x = 0; x < kRrw; ++x)
                {
                    packedT8H[y * kRrw + x] = byteT8H(x, y);
                }
            }

            constexpr uint64_t kUploadT8HBitblt =
                (static_cast<uint64_t>(kDbpT8H) << 32) |
                (static_cast<uint64_t>(kDbwT8H) << 48) |
                (static_cast<uint64_t>(GS_PSM_T8H) << 56);

            gs.writeRegister(GS_REG_BITBLTBUF, kUploadT8HBitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, kRect);
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            std::vector<uint8_t> packetT8H;
            appendU64(packetT8H, makeGifTag(static_cast<uint16_t>(packedT8H.size() / 16u), GIF_FMT_IMAGE, 0u, true));
            appendU64(packetT8H, 0ull);
            packetT8H.insert(packetT8H.end(), packedT8H.begin(), packedT8H.end());
            gs.processGIFPacket(packetT8H.data(), static_cast<uint32_t>(packetT8H.size()));

            bool t8hByteMatches = true;
            bool t8hMemReaderMatches = true;
            for (uint32_t y = 0; y < kRrh; ++y)
            {
                for (uint32_t x = 0; x < kRrw; ++x)
                {
                    const uint32_t off = GSPSMCT32::addrPSMCT32(kDbpT8H, kDbwT8H, x, y);
                    uint32_t word = 0u;
                    std::memcpy(&word, vram.data() + off, sizeof(word));

                    const uint8_t expected = byteT8H(x, y);
                    const uint8_t got = static_cast<uint8_t>((word >> 24) & 0xFFu);
                    if (got != expected)
                        t8hByteMatches = false;

                    const uint32_t memByte = GSMem::ReadP8H(vram.data(), kDbpT8H, kDbwT8H, x, y);
                    if (memByte != expected)
                        t8hMemReaderMatches = false;
                }
            }
            t.IsTrue(t8hByteMatches,
                     "T8H upload must land the full byte in bits 24-31 of the shared CT32 word");
            t.IsTrue(t8hMemReaderMatches,
                     "GSMem::ReadP8H should agree with the raw shared-word byte extraction after a T8H upload");

            // Clobber interaction: upload a T8H byte plane, then upload a T4HL nibble plane to
            // the same shared word. WriteP4HL's nibble RMW should overwrite bits 24-27 with the
            // new nibble while preserving bits 28-31 (the T8H byte's high nibble).
            constexpr uint32_t kDbpMix = 192u;
            constexpr uint32_t kDbwMix = 1u;

            auto byteMix = [](uint32_t x, uint32_t y) -> uint8_t
            {
                return static_cast<uint8_t>((x * 7u + y * 5u + 3u) & 0xFFu);
            };
            auto nibbleN = [](uint32_t x, uint32_t y) -> uint8_t
            {
                return static_cast<uint8_t>((x * 3u + y + 1u) & 0xFu);
            };

            std::vector<uint8_t> packedMixT8H(kRrw * kRrh, 0u);
            for (uint32_t y = 0; y < kRrh; ++y)
            {
                for (uint32_t x = 0; x < kRrw; ++x)
                {
                    packedMixT8H[y * kRrw + x] = byteMix(x, y);
                }
            }

            constexpr uint64_t kUploadMixT8HBitblt =
                (static_cast<uint64_t>(kDbpMix) << 32) |
                (static_cast<uint64_t>(kDbwMix) << 48) |
                (static_cast<uint64_t>(GS_PSM_T8H) << 56);

            gs.writeRegister(GS_REG_BITBLTBUF, kUploadMixT8HBitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, kRect);
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            std::vector<uint8_t> packetMixT8H;
            appendU64(packetMixT8H,
                      makeGifTag(static_cast<uint16_t>(packedMixT8H.size() / 16u), GIF_FMT_IMAGE, 0u, true));
            appendU64(packetMixT8H, 0ull);
            packetMixT8H.insert(packetMixT8H.end(), packedMixT8H.begin(), packedMixT8H.end());
            gs.processGIFPacket(packetMixT8H.data(), static_cast<uint32_t>(packetMixT8H.size()));

            std::vector<uint8_t> packedMixNibble((kRrw * kRrh) / 2u, 0u);
            for (uint32_t y = 0; y < kRrh; ++y)
            {
                for (uint32_t x = 0; x < kRrw; x += 2u)
                {
                    const uint8_t lo = nibbleN(x, y) & 0xFu;
                    const uint8_t hi = nibbleN(x + 1u, y) & 0xFu;
                    packedMixNibble[(y * kRrw + x) / 2u] = static_cast<uint8_t>(lo | (hi << 4));
                }
            }

            constexpr uint64_t kUploadMixHLBitblt =
                (static_cast<uint64_t>(kDbpMix) << 32) |
                (static_cast<uint64_t>(kDbwMix) << 48) |
                (static_cast<uint64_t>(GS_PSM_T4HL) << 56);

            gs.writeRegister(GS_REG_BITBLTBUF, kUploadMixHLBitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, kRect);
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            std::vector<uint8_t> packetMixNibble;
            appendU64(packetMixNibble,
                      makeGifTag(static_cast<uint16_t>(packedMixNibble.size() / 16u), GIF_FMT_IMAGE, 0u, true));
            appendU64(packetMixNibble, 0ull);
            packetMixNibble.insert(packetMixNibble.end(), packedMixNibble.begin(), packedMixNibble.end());
            gs.processGIFPacket(packetMixNibble.data(), static_cast<uint32_t>(packetMixNibble.size()));

            bool mixClobberMatches = true;
            for (uint32_t y = 0; y < kRrh; ++y)
            {
                for (uint32_t x = 0; x < kRrw; ++x)
                {
                    const uint32_t off = GSPSMCT32::addrPSMCT32(kDbpMix, kDbwMix, x, y);
                    uint32_t word = 0u;
                    std::memcpy(&word, vram.data() + off, sizeof(word));

                    const uint8_t gotLow = static_cast<uint8_t>((word >> 24) & 0xFu);
                    const uint8_t gotHigh = static_cast<uint8_t>((word >> 28) & 0xFu);
                    const uint8_t expectedLow = nibbleN(x, y);
                    const uint8_t expectedHigh = static_cast<uint8_t>((byteMix(x, y) >> 4) & 0xFu);
                    if (gotLow != expectedLow || gotHigh != expectedHigh)
                        mixClobberMatches = false;
                }
            }
            t.IsTrue(mixClobberMatches,
                     "T4HL nibble upload over a T8H byte must overwrite bits 24-27 with the nibble and preserve "
                     "bits 28-31 from the T8H byte's high nibble");
        });

        tc.Run("GS T4HL/T4HH sampling reads only its own plane through independent CLUTs", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kTexTbp = 64u;
            constexpr uint32_t kClutCbpA = 128u;
            constexpr uint32_t kClutCbpB = 192u;
            constexpr uint8_t kIndexA = 4u; // T4HL plane index at the sampled texel (0..7 -> identity swizzle)
            constexpr uint8_t kIndexB = 0u; // T4HH plane index at the sampled texel; must differ from kIndexA

            // Shared CT32 word at texel (0,0): T4HL nibble occupies bits 24-27, T4HH bits 28-31.
            const uint32_t sharedWordOff = GSPSMCT32::addrPSMCT32(kTexTbp, 1u, 0u, 0u);
            const uint32_t sharedWord =
                (static_cast<uint32_t>(kIndexB) << 28) | (static_cast<uint32_t>(kIndexA) << 24);
            std::memcpy(vram.data() + sharedWordOff, &sharedWord, sizeof(sharedWord));

            constexpr uint32_t kExpectedColorA = 0x800000FFu; // RGBA = (255,0,0,128)
            constexpr uint32_t kExpectedColorB = 0x8000FF00u; // RGBA = (0,255,0,128)
            constexpr uint32_t kDistractorColor = 0x800000AAu;

            // Place each plane's expected color at its own CLUT's entry for the sampled index.
            const uint32_t clutAOff = GSPSMCT32::addrPSMCT32(kClutCbpA, 1u, kIndexA, 0u);
            const uint32_t clutBOff = GSPSMCT32::addrPSMCT32(kClutCbpB, 1u, kIndexB, 0u);
            std::memcpy(vram.data() + clutAOff, &kExpectedColorA, sizeof(kExpectedColorA));
            std::memcpy(vram.data() + clutBOff, &kExpectedColorB, sizeof(kExpectedColorB));

            // Seed distractor entries at the *other* plane's index in each CLUT so that a
            // cross-plane nibble read (a bug reading the wrong plane, or the wrong CLUT) would
            // resolve to a non-matching color instead of accidentally matching by coincidence.
            const uint32_t clutADistractorOff = GSPSMCT32::addrPSMCT32(kClutCbpA, 1u, kIndexB, 0u);
            const uint32_t clutBDistractorOff = GSPSMCT32::addrPSMCT32(kClutCbpB, 1u, kIndexA, 0u);
            std::memcpy(vram.data() + clutADistractorOff, &kDistractorColor, sizeof(kDistractorColor));
            std::memcpy(vram.data() + clutBDistractorOff, &kDistractorColor, sizeof(kDistractorColor));

            constexpr uint64_t kFrameReg =
                (0ull << 0) |
                (1ull << 16) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 24);
            constexpr uint64_t kZbuf = (1ull << 32);
            constexpr uint64_t kPrim =
                static_cast<uint64_t>(GS_PRIM_SPRITE) |
                (1ull << 4) |  // TME
                (1ull << 8);   // FST

            const uint64_t kTex0HL =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_T4HL) << 20) |
                (0ull << 26) |
                (0ull << 30) |
                (1ull << 34) |
                (1ull << 35) |
                (static_cast<uint64_t>(kClutCbpA) << 37) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 51) |
                (1ull << 61);

            gs.writeRegister(GS_REG_FRAME_1, kFrameReg);
            gs.writeRegister(GS_REG_ZBUF_1, kZbuf);
            gs.writeRegister(GS_REG_SCISSOR_1, 0ull);
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);
            gs.writeRegister(GS_REG_ALPHA_1, 0ull);
            gs.writeRegister(GS_REG_TEX0_1, kTex0HL);
            gs.writeRegister(GS_REG_PRIM, kPrim);
            gs.writeRegister(GS_REG_RGBAQ, 0x80808080ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, (16ull << 0) | (16ull << 16));

            uint32_t pixelHL = 0u;
            std::memcpy(&pixelHL, vram.data(), sizeof(pixelHL));
            t.Equals(pixelHL, kExpectedColorA,
                     "T4HL sampling should resolve through its own CLUT plane, unaffected by the co-resident T4HH nibble");

            const uint64_t kTex0HH =
                (static_cast<uint64_t>(kTexTbp) << 0) |
                (1ull << 14) |
                (static_cast<uint64_t>(GS_PSM_T4HH) << 20) |
                (0ull << 26) |
                (0ull << 30) |
                (1ull << 34) |
                (1ull << 35) |
                (static_cast<uint64_t>(kClutCbpB) << 37) |
                (static_cast<uint64_t>(GS_PSM_CT32) << 51) |
                (1ull << 61);

            gs.writeRegister(GS_REG_TEX0_1, kTex0HH);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, 0ull);
            gs.writeRegister(GS_REG_UV, 0ull);
            gs.writeRegister(GS_REG_XYZ2, (16ull << 0) | (16ull << 16));

            uint32_t pixelHH = 0u;
            std::memcpy(&pixelHH, vram.data(), sizeof(pixelHH));
            t.Equals(pixelHH, kExpectedColorB,
                     "T4HH sampling should resolve through its own CLUT plane, unaffected by the co-resident T4HL nibble");
        });

        tc.Run("GS T4HL upload deactivates the transfer at total_pixels and discards excess bytes", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            constexpr uint32_t kDbw = 1u;
            constexpr uint32_t kRrw = 8u;
            constexpr uint32_t kRrh = 8u;
            constexpr uint64_t kRect = (static_cast<uint64_t>(kRrw) << 0) | (static_cast<uint64_t>(kRrh) << 32);

            auto buildPacked = [&](const auto &indexFn) -> std::vector<uint8_t>
            {
                std::vector<uint8_t> packed((kRrw * kRrh) / 2u, 0u);
                for (uint32_t y = 0; y < kRrh; ++y)
                {
                    for (uint32_t x = 0; x < kRrw; x += 2u)
                    {
                        const uint8_t lo = indexFn(x, y) & 0xFu;
                        const uint8_t hi = indexFn(x + 1u, y) & 0xFu;
                        packed[(y * kRrw + x) / 2u] = static_cast<uint8_t>(lo | (hi << 4));
                    }
                }
                return packed;
            };

            // --- First transfer: an ~8x oversized IMAGE payload (256 bytes / 16 qwords) for a
            // rect that only needs 32 bytes (64 texels). The first 32 bytes carry a known
            // pattern; the remaining 224 bytes are a 0xFF sentinel that must be discarded.
            constexpr uint32_t kDbp1 = 0u;
            constexpr uint64_t kUploadBitblt1 =
                (static_cast<uint64_t>(kDbp1) << 32) |
                (static_cast<uint64_t>(kDbw) << 48) |
                (static_cast<uint64_t>(GS_PSM_T4HL) << 56);

            auto indexPattern1 = [](uint32_t x, uint32_t y) -> uint8_t
            {
                return static_cast<uint8_t>((x + y * 3u + 2u) & 0xFu);
            };
            const std::vector<uint8_t> packed1 = buildPacked(indexPattern1);
            t.Equals(packed1.size(), static_cast<size_t>(32), "sanity: packed rect should be 32 bytes (64 texels)");

            gs.writeRegister(GS_REG_BITBLTBUF, kUploadBitblt1);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, kRect);
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            constexpr uint32_t kOversizedBytes = 256u; // 16 qwords, 8x the required 32 bytes
            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(static_cast<uint16_t>(kOversizedBytes / 16u), GIF_FMT_IMAGE, 0u, true));
            appendU64(packet, 0ull);
            const size_t payloadOffset = packet.size();
            packet.resize(payloadOffset + kOversizedBytes, 0xFFu);
            std::memcpy(packet.data() + payloadOffset, packed1.data(), packed1.size());
            gs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));

            const GSDebugSnapshot snap1 = gs.getDebugSnapshot();
            t.Equals(snap1.transferCopiedPixels, 64u, "T4HL transfer should stop after copying exactly rrw*rrh texels");
            t.Equals(snap1.trxdir, 3u, "T4HL transfer should deactivate (trxdir=3) once total_pixels is reached");

            bool pattern1Ok = true;
            for (uint32_t y = 0; y < kRrh; ++y)
            {
                for (uint32_t x = 0; x < kRrw; ++x)
                {
                    const uint32_t off = GSPSMCT32::addrPSMCT32(kDbp1, kDbw, x, y);
                    uint32_t word = 0u;
                    std::memcpy(&word, vram.data() + off, sizeof(word));
                    if (((word >> 24) & 0xFu) != indexPattern1(x, y))
                        pattern1Ok = false;
                }
            }
            t.IsTrue(pattern1Ok,
                     "the first 64 texels of the oversized T4HL transfer should match the known pattern; sentinel bytes must not leak in");

            // --- Second, correctly-sized transfer to a different DBP: proves the discarded
            // excess bytes from the first transfer were not mis-accounted into later state.
            constexpr uint32_t kDbp2 = 128u;
            constexpr uint64_t kUploadBitblt2 =
                (static_cast<uint64_t>(kDbp2) << 32) |
                (static_cast<uint64_t>(kDbw) << 48) |
                (static_cast<uint64_t>(GS_PSM_T4HL) << 56);

            auto indexPattern2 = [](uint32_t x, uint32_t y) -> uint8_t
            {
                return static_cast<uint8_t>((x * 5u + y + 3u) & 0xFu);
            };
            const std::vector<uint8_t> packed2 = buildPacked(indexPattern2);

            gs.writeRegister(GS_REG_BITBLTBUF, kUploadBitblt2);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, kRect);
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            std::vector<uint8_t> packet2;
            appendU64(packet2, makeGifTag(static_cast<uint16_t>(packed2.size() / 16u), GIF_FMT_IMAGE, 0u, true));
            appendU64(packet2, 0ull);
            packet2.insert(packet2.end(), packed2.begin(), packed2.end());
            gs.processGIFPacket(packet2.data(), static_cast<uint32_t>(packet2.size()));

            const GSDebugSnapshot snap2 = gs.getDebugSnapshot();
            t.Equals(snap2.transferCopiedPixels, 64u, "second, correctly-sized T4HL transfer should copy exactly rrw*rrh texels");
            t.Equals(snap2.trxdir, 3u, "second T4HL transfer should also deactivate cleanly");

            bool pattern2Ok = true;
            for (uint32_t y = 0; y < kRrh; ++y)
            {
                for (uint32_t x = 0; x < kRrw; ++x)
                {
                    const uint32_t off = GSPSMCT32::addrPSMCT32(kDbp2, kDbw, x, y);
                    uint32_t word = 0u;
                    std::memcpy(&word, vram.data() + off, sizeof(word));
                    if (((word >> 24) & 0xFu) != indexPattern2(x, y))
                        pattern2Ok = false;
                }
            }
            t.IsTrue(pattern2Ok,
                     "second T4HL transfer to a different DBP should be byte-correct, proving the discarded excess bytes from the first transfer did not leak into subsequent transfer state");
        });
    });
}
