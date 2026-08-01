#include "runtime/ps2_gs_rasterizer.h"
#include "runtime/ps2_gs_backend.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_gs_common.h"
#include "runtime/ps2_gs_psmct16.h"
#include "runtime/ps2_gs_psmct32.h"
#include "runtime/ps2_gs_psmt4.h"
#include "runtime/ps2_gs_psmt8.h"
#include "runtime/ps2_gs_memory.h"
#include "runtime/ps2_gs_vulkan_backend.h"
#include "ps2_gs_rasterizer_detail.h"
#include "ps2_gs_packed_sprite_kernel.h"
#include "ps2_log.h"
#include <array>
#include <atomic>
#include <algorithm>
#include <bitset>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(__i386__) || defined(__x86_64__) || \
    defined(_M_IX86) || defined(_M_X64)
#define PS2X_GS_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#else
#define PS2X_GS_X86 0
#endif

#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif

#if defined(__SSE4_1__)
#define PS2X_GS_HAS_SSE41_VARIANT 1
#else
#define PS2X_GS_HAS_SSE41_VARIANT 0
#endif

#ifndef PS2X_GS_HAS_AVX2_KERNEL
#define PS2X_GS_HAS_AVX2_KERNEL 0
#endif

#if defined(_MSC_VER)
#define PS2X_GS_ALWAYS_INLINE __forceinline
#define PS2X_GS_LAMBDA_ALWAYS_INLINE
#define PS2X_GS_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define PS2X_GS_ALWAYS_INLINE inline __attribute__((always_inline))
#define PS2X_GS_LAMBDA_ALWAYS_INLINE __attribute__((always_inline))
#define PS2X_GS_NOINLINE __attribute__((noinline))
#else
#define PS2X_GS_ALWAYS_INLINE inline
#define PS2X_GS_LAMBDA_ALWAYS_INLINE
#define PS2X_GS_NOINLINE
#endif

using namespace GSInternal;

namespace
{
    bool hostSupportsAvx2()
    {
#if PS2X_GS_HAS_AVX2_KERNEL && PS2X_GS_X86 && \
    (defined(__GNUC__) || defined(__clang__))
        static const bool supported = []
        {
            __builtin_cpu_init();
            return __builtin_cpu_supports("avx2") != 0;
        }();
        return supported;
#elif PS2X_GS_HAS_AVX2_KERNEL && PS2X_GS_X86 && \
    defined(_MSC_VER)
        static const bool supported = []
        {
            int registers[4]{};
            __cpuid(registers, 0);
            if (registers[0] < 7)
                return false;

            __cpuidex(registers, 1, 0);
            constexpr int kOsXsave = 1 << 27;
            constexpr int kAvx = 1 << 28;
            if ((registers[2] & (kOsXsave | kAvx)) !=
                (kOsXsave | kAvx))
            {
                return false;
            }
            if ((_xgetbv(0) & 0x6u) != 0x6u)
                return false;

            __cpuidex(registers, 7, 0);
            constexpr int kAvx2 = 1 << 5;
            return (registers[1] & kAvx2) != 0;
        }();
        return supported;
#else
        return false;
#endif
    }

    GSRasterizerDetail::PackedSpriteKernelOverride
        environmentPackedSpriteKernelOverride()
    {
        using Override =
            GSRasterizerDetail::PackedSpriteKernelOverride;
        static const Override overrideMode = []
        {
            const char *value =
                std::getenv("PS2X_GS_PACKED_SPRITE_KERNEL");
            if (value == nullptr)
                return Override::Automatic;
            if (std::strcmp(value, "scalar") == 0)
                return Override::ForceScalar;
            if (std::strcmp(value, "sse41") == 0)
                return Override::ForceSse41;
            if (std::strcmp(value, "avx2") == 0)
                return Override::ForceAvx2;
            return Override::Automatic;
        }();
        return overrideMode;
    }
}

namespace GSRasterizerDetail
{
    namespace
    {
        std::atomic<PackedSpriteKernelOverride>
            s_packedSpriteKernelOverride{
                PackedSpriteKernelOverride::Automatic};
        std::atomic<uint64_t>
            s_packedSpriteKernelDispatchCount{0u};
        std::atomic<PackedSpriteKernelImplementation>
            s_packedSpriteLastKernelImplementation{
                PackedSpriteKernelImplementation::Scalar};
        std::atomic<uint64_t>
            s_packedSpriteVectorGroupCount{0u};
    }

    void setPackedSpriteKernelOverride(
        PackedSpriteKernelOverride overrideMode)
    {
        s_packedSpriteKernelOverride.store(
            overrideMode, std::memory_order_relaxed);
    }

    PackedSpriteKernelOverride packedSpriteKernelOverride()
    {
        return s_packedSpriteKernelOverride.load(
            std::memory_order_relaxed);
    }

    bool packedSpriteKernelImplementationAvailable(
        PackedSpriteKernelImplementation implementation)
    {
        switch (implementation)
        {
        case PackedSpriteKernelImplementation::Scalar:
            return true;
        case PackedSpriteKernelImplementation::Sse41:
            return PS2X_GS_HAS_SSE41_VARIANT != 0;
        case PackedSpriteKernelImplementation::Avx2:
            return hostSupportsAvx2();
        }
        return false;
    }

    void resetPackedSpriteKernelDispatchCount()
    {
        s_packedSpriteKernelDispatchCount.store(
            0u, std::memory_order_relaxed);
        s_packedSpriteLastKernelImplementation.store(
            PackedSpriteKernelImplementation::Scalar,
            std::memory_order_relaxed);
    }

    uint64_t packedSpriteKernelDispatchCount()
    {
        return s_packedSpriteKernelDispatchCount.load(
            std::memory_order_relaxed);
    }

    PackedSpriteKernelImplementation
        packedSpriteLastKernelImplementation()
    {
        return s_packedSpriteLastKernelImplementation.load(
            std::memory_order_relaxed);
    }

    void recordPackedSpriteKernelDispatch(
        PackedSpriteKernelImplementation implementation)
    {
        s_packedSpriteLastKernelImplementation.store(
            implementation, std::memory_order_relaxed);
        s_packedSpriteKernelDispatchCount.fetch_add(
            1u, std::memory_order_relaxed);
    }

    void resetPackedSpriteVectorGroupCount()
    {
        s_packedSpriteVectorGroupCount.store(
            0u, std::memory_order_relaxed);
    }

    uint64_t packedSpriteVectorGroupCount()
    {
        return s_packedSpriteVectorGroupCount.load(
            std::memory_order_relaxed);
    }

    void recordPackedSpriteVectorGroup()
    {
        s_packedSpriteVectorGroupCount.fetch_add(
            1u, std::memory_order_relaxed);
    }
}

namespace
{
    struct FixedPointVertex
    {
        int32_t x;
        int32_t y;
    };

    int ceilFixed12_4(int32_t value)
    {
        // C++ integer division truncates toward zero, which is ceil for
        // negative values. Positive values need the fractional bias.
        return (value >= 0) ? ((value + 15) / 16) : (value / 16);
    }

    int64_t triangleEdge(const FixedPointVertex &a,
                         const FixedPointVertex &b,
                         const FixedPointVertex &p)
    {
        return static_cast<int64_t>(b.x - a.x) * static_cast<int64_t>(p.y - a.y) -
               static_cast<int64_t>(b.y - a.y) * static_cast<int64_t>(p.x - a.x);
    }

    bool isTopLeftEdge(const FixedPointVertex &a,
                       const FixedPointVertex &b,
                       bool positiveArea)
    {
        int32_t dx = b.x - a.x;
        int32_t dy = b.y - a.y;
        if (!positiveArea)
        {
            dx = -dx;
            dy = -dy;
        }

        // Window Y increases downward. For a consistently oriented triangle,
        // an upward edge is a left edge and a rightward horizontal edge is a
        // top edge.
        return dy < 0 || (dy == 0 && dx > 0);
    }

    float fabsQ(float q)
    {
        return (std::fabs(q) > 1.0e-8f) ? q : 1.0f;
    }

    u16 Rgba8888ToRgba5551(u32 c)
    {
        uint32_t r = ((c >> 0)  & 0xFF) >> 3;
        uint32_t g = ((c >> 8)  & 0xFF) >> 3;
        uint32_t b = ((c >> 16) & 0xFF) >> 3;
        uint32_t a = ((c >> 24) & 0xFF) >> 7;

        return (r | (g << 5) | (b << 10) | (a << 15));
    }

    u32 Rgba5551ToRgba8888(u16 c)
    {
        u32 r = ((c >> 0)  & 0x1F) << 3;
        u32 g = ((c >> 5)  & 0x1F) << 3;
        u32 b = ((c >> 10) & 0x1F) << 3;
        u32 a = ((c >> 15) & 0x01) << 7;

        return (r | (g << 8) | (b << 16) | (a << 24));
    }

    u32 pack32(u8 r, u8 g, u8 b, u8 a)
    {
        return static_cast<u32>(r) | (g << 8) | (b << 16) | (a << 24);
    }

    uint8_t blendFogChannel(uint8_t source, uint8_t fog, uint8_t factor)
    {
        // GS fog is an 8-bit fixed-point interpolation. Using a 256-wide
        // complement makes F=0 exactly the distant fog colour, matching the
        // documented endpoint and PCSX2's software renderer.
        const uint32_t sourceWeight = static_cast<uint32_t>(factor);
        const uint32_t fogWeight = 256u - sourceWeight;
        return static_cast<uint8_t>(
            (static_cast<uint32_t>(source) * sourceWeight +
             static_cast<uint32_t>(fog) * fogWeight) >>
            8u);
    }

    PS2X_GS_ALWAYS_INLINE uint8_t blendFogChannelFixed(
        uint8_t source,
        uint8_t fog,
        uint16_t factor)
    {
        const int difference =
            (static_cast<int>(source) - static_cast<int>(fog)) *
            static_cast<int>(factor);
        const int delta =
            difference >= 0
                ? difference / 32768
                : -((-difference + 32767) / 32768);
        return clampU8(static_cast<int>(fog) + delta);
    }

    void applyFog(uint32_t fogColor,
                  uint8_t factor,
                  uint8_t &r,
                  uint8_t &g,
                  uint8_t &b)
    {
        r = blendFogChannel(r,
                            static_cast<uint8_t>(fogColor),
                            factor);
        g = blendFogChannel(g,
                            static_cast<uint8_t>(fogColor >> 8u),
                            factor);
        b = blendFogChannel(b,
                            static_cast<uint8_t>(fogColor >> 16u),
                            factor);
    }

    PS2X_GS_ALWAYS_INLINE void applyFogFixed(
        uint32_t fogColor,
        uint16_t factor,
        uint8_t &r,
        uint8_t &g,
        uint8_t &b)
    {
        r = blendFogChannelFixed(
            r, static_cast<uint8_t>(fogColor), factor);
        g = blendFogChannelFixed(
            g,
            static_cast<uint8_t>(fogColor >> 8u),
            factor);
        b = blendFogChannelFixed(
            b,
            static_cast<uint8_t>(fogColor >> 16u),
            factor);
    }

    uint32_t applyTexa(const GSTexaReg &texa, uint8_t psm, uint32_t texel)
    {
        if (psm == GS_PSM_CT32)
            return texel;

        const uint8_t r = static_cast<uint8_t>(texel & 0xFFu);
        const uint8_t g = static_cast<uint8_t>((texel >> 8) & 0xFFu);
        const uint8_t b = static_cast<uint8_t>((texel >> 16) & 0xFFu);
        const bool rgbZero = r == 0u && g == 0u && b == 0u;
        uint8_t a = static_cast<uint8_t>((texel >> 24) & 0xFFu);

        switch (psm)
        {
        case GS_PSM_CT24:
            a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            if ((a & 0x80u) != 0u)
                a = texa.ta1;
            else
                a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        default:
            break;
        }

        return (texel & 0x00FFFFFFu) | (static_cast<uint32_t>(a) << 24);
    }

    uint32_t addrPSMCT16Family(uint32_t basePtr, uint32_t width, uint8_t psm, uint32_t x, uint32_t y)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
            return GSPSMCT16::addrPSMCT16(basePtr, width, x, y);
        case GS_PSM_CT16S:
            return GSPSMCT16::addrPSMCT16S(basePtr, width, x, y);
        case GS_PSM_Z16:
            return GSPSMCT16::addrPSMZ16(basePtr, width, x, y);
        case GS_PSM_Z16S:
            return GSPSMCT16::addrPSMZ16S(basePtr, width, x, y);
        default:
            return 0u;
        }
    }

    std::atomic<uint32_t> s_debugPrimitiveCount{0};
    std::atomic<uint32_t> s_debugPixelCount{0};
    std::atomic<uint32_t> s_debugContext1PrimitiveCount{0};
    std::atomic<uint32_t> s_debugFbp150PixelCount{0};

    constexpr size_t kGsPageCount =
        GSMem::MEMORY_SIZE / GSMem::GS_PAGE_SIZE;

    uint32_t debugPageId(uint8_t psm,
                         uint32_t base,
                         uint32_t width,
                         uint32_t x,
                         uint32_t y)
    {
        size_t page = std::numeric_limits<size_t>::max();
        switch (psm)
        {
        case GS_PSM_CT32:
            page = GSMem::PixelStorageTraits<GSMem::C32>::PageId(
                base, width, x, y);
            break;
        case GS_PSM_CT24:
            page = GSMem::PixelStorageTraits<GSMem::C24>::PageId(
                base, width, x, y);
            break;
        case GS_PSM_CT16:
            page = GSMem::PixelStorageTraits<GSMem::C16>::PageId(
                base, width, x, y);
            break;
        case GS_PSM_CT16S:
            page = GSMem::PixelStorageTraits<GSMem::C16S>::PageId(
                base, width, x, y);
            break;
        case GS_PSM_T8:
            page = GSMem::PixelStorageTraits<GSMem::P8>::PageId(
                base, width, x, y);
            break;
        case GS_PSM_T4:
            page = GSMem::PixelStorageTraits<GSMem::P4>::PageId(
                base, width, x, y);
            break;
        case GS_PSM_T8H:
            page = GSMem::PixelStorageTraits<GSMem::P8H>::PageId(
                base, width, x, y);
            break;
        case GS_PSM_T4HL:
            page = GSMem::PixelStorageTraits<GSMem::P4HL>::PageId(
                base, width, x, y);
            break;
        case GS_PSM_T4HH:
            page = GSMem::PixelStorageTraits<GSMem::P4HH>::PageId(
                base, width, x, y);
            break;
        case GS_PSM_Z32:
            page = GSMem::PixelStorageTraits<GSMem::Z32>::PageId(
                base, width, x, y);
            break;
        case GS_PSM_Z24:
            page = GSMem::PixelStorageTraits<GSMem::Z24>::PageId(
                base, width, x, y);
            break;
        case GS_PSM_Z16:
            page = GSMem::PixelStorageTraits<GSMem::Z16>::PageId(
                base, width, x, y);
            break;
        case GS_PSM_Z16S:
            page = GSMem::PixelStorageTraits<GSMem::Z16S>::PageId(
                base, width, x, y);
            break;
        default:
            return std::numeric_limits<uint32_t>::max();
        }
        return static_cast<uint32_t>(page % kGsPageCount);
    }

    uint64_t debugFnv1a64(const uint8_t *data, size_t size)
    {
        uint64_t hash = 14695981039346656037ull;
        for (size_t index = 0u; index < size; ++index)
        {
            hash ^= data[index];
            hash *= 1099511628211ull;
        }
        return hash;
    }

    std::string debugPageList(const std::bitset<kGsPageCount> &pages)
    {
        std::ostringstream output;
        bool first = true;
        for (size_t page = 0u; page < pages.size(); ++page)
        {
            if (!pages.test(page))
                continue;
            if (!first)
                output << ';';
            first = false;
            output << page;
        }
        return output.str();
    }

    std::string debugPageList(const GsVramPageMask &pages)
    {
        std::ostringstream output;
        bool first = true;
        for (size_t page = 0u; page < GS_VRAM_PAGE_COUNT; ++page)
        {
            if (!pages.test(page))
                continue;
            if (!first)
                output << ';';
            first = false;
            output << page;
        }
        return output.str();
    }

    struct GSDrawTraceState
    {
        std::ofstream output;
        uint64_t index = 0u;
        uint64_t written = 0u;
        uint64_t limit = 1000000u;
        uint32_t framebufferFilter = 0u;
        bool filterFramebuffer = false;
        uint64_t candidates = 0u;
        uint64_t scissorRejects = 0u;
        uint64_t alphaRejects = 0u;
        uint64_t destinationAlphaRejects = 0u;
        uint64_t depthRejects = 0u;
        uint64_t writes = 0u;
        uint64_t framebufferChanges = 0u;
        uint64_t depthWrites = 0u;
        uint64_t depthChanges = 0u;
        uint8_t minR = 0xFFu;
        uint8_t maxR = 0u;
        uint8_t minG = 0xFFu;
        uint8_t maxG = 0u;
        uint8_t minB = 0xFFu;
        uint8_t maxB = 0u;
        uint8_t minA = 0xFFu;
        uint8_t maxA = 0u;
        uint64_t textureSamples = 0u;
        uint8_t minTextureIndex = 0xFFu;
        uint8_t maxTextureIndex = 0u;
        uint8_t minTextureAlpha = 0xFFu;
        uint8_t maxTextureAlpha = 0u;
        std::bitset<kGsPageCount> framebufferPages;
        std::bitset<kGsPageCount> depthPages;
        std::bitset<kGsPageCount> texturePages;
        bool feedback = false;
        bool hashVram = false;
        bool initialized = false;
        bool capturing = false;

        void begin(const GsDrawCommand &command)
        {
            const GSContext &ctx = command.context();
            if (!initialized)
            {
                initialized = true;
                const char *path = std::getenv("PS2X_GS_DRAW_TRACE");
                if (path && path[0] != '\0')
                {
                    if (const char *limitText = std::getenv("PS2X_GS_DRAW_TRACE_LIMIT"))
                    {
                        char *end = nullptr;
                        const unsigned long long parsed = std::strtoull(limitText, &end, 0);
                        if (end != limitText && end && *end == '\0' && parsed != 0u)
                            limit = parsed;
                    }
                    if (const char *fbpText = std::getenv("PS2X_GS_DRAW_TRACE_FBP"))
                    {
                        char *end = nullptr;
                        const unsigned long parsed = std::strtoul(fbpText, &end, 0);
                        if (end != fbpText && end && *end == '\0' && parsed <= 0x1FFu)
                        {
                            framebufferFilter = static_cast<uint32_t>(parsed);
                            filterFramebuffer = true;
                        }
                    }
                    if (const char *hashText =
                            std::getenv("PS2X_GS_DRAW_TRACE_VRAM_HASH"))
                    {
                        hashVram =
                            hashText[0] != '\0' &&
                            std::strcmp(hashText, "0") != 0;
                    }
                    output.open(path, std::ios::out | std::ios::trunc);
                    if (output)
                    {
                        output << "index,type,iip,tme,fge,abe,fst,ctxt,fogcol,fbp,fbw,fpsm,fbmsk,zbp,zpsm,zmask,test,"
                                  "alpha,tbp0,tbw,tpsm,tw,th,tcc,tfx,cbp,cpsm,csm,csa,cld,"
                                  "tex1,miptbp1,miptbp2,clamp,"
                                  "texclut_cbw,texclut_cou,texclut_cov,"
                                  "ofx,ofy,sx0,sx1,sy0,sy1,"
                                  "x0,y0,z0,x1,y1,z1,x2,y2,z2,"
                                  "s0,t0,q0,u0,v0,fog0,r0,g0,b0,a0,"
                                  "s1,t1,q1,u1,v1,fog1,r1,g1,b1,a1,"
                                  "s2,t2,q2,u2,v2,fog2,r2,g2,b2,a2,"
                                  "candidates,scissor_rejects,alpha_rejects,destination_alpha_rejects,"
                                  "depth_rejects,writes,"
                                  "framebuffer_changes,depth_writes,depth_changes,"
                                  "min_r,max_r,min_g,max_g,min_b,max_b,min_a,max_a,"
                                  "texture_samples,min_texture_index,max_texture_index,"
                                  "min_texture_alpha,max_texture_alpha,"
                                  "framebuffer_pages,depth_pages,texture_pages,"
                                  "feedback,vram_fnv1a64,"
                                  "sequence,state_signature,draw_x0,draw_y0,draw_x1,draw_y1,bounds_exact,"
                                  "raw_x0,raw_y0,integer_z0,raw_x1,raw_y1,integer_z1,raw_x2,raw_y2,integer_z2,"
                                  "aa1,fix,pabe,texa_ta0,texa_aem,texa_ta1,scanmsk,dimx,dthe,colclamp,fba,"
                                  "framebuffer_read_pages,framebuffer_write_pages,"
                                  "depth_read_pages,depth_write_pages,texture_read_pages,mip_read_pages,clut_read_pages,"
                                  "reads_destination,framebuffer_depth_alias,framebuffer_texture_alias,framebuffer_clut_alias,"
                                  "fallback_reason\n";
                    }
                }
            }

            capturing = output.is_open() &&
                        written < limit &&
                        (!filterFramebuffer || ctx.frame.fbp == framebufferFilter);
            if (!capturing)
                return;

            candidates = 0u;
            scissorRejects = 0u;
            alphaRejects = 0u;
            destinationAlphaRejects = 0u;
            depthRejects = 0u;
            writes = 0u;
            framebufferChanges = 0u;
            depthWrites = 0u;
            depthChanges = 0u;
            minR = minG = minB = minA = 0xFFu;
            maxR = maxG = maxB = maxA = 0u;
            textureSamples = 0u;
            minTextureIndex = minTextureAlpha = 0xFFu;
            maxTextureIndex = maxTextureAlpha = 0u;
            framebufferPages.reset();
            depthPages.reset();
            texturePages.reset();
            feedback = false;
        }

        void recordFramebufferPage(uint8_t psm,
                                   uint32_t base,
                                   uint32_t width,
                                   uint32_t x,
                                   uint32_t y)
        {
            recordPage(framebufferPages, psm, base, width, x, y);
        }

        void recordDepthPage(uint8_t psm,
                             uint32_t base,
                             uint32_t width,
                             uint32_t x,
                             uint32_t y)
        {
            recordPage(depthPages, psm, base, width, x, y);
        }

        void recordTexturePage(uint8_t psm,
                               uint32_t base,
                               uint32_t width,
                               uint32_t x,
                               uint32_t y)
        {
            recordPage(texturePages, psm, base, width, x, y);
        }

    private:
        void recordPage(std::bitset<kGsPageCount> &pages,
                        uint8_t psm,
                        uint32_t base,
                        uint32_t width,
                        uint32_t x,
                        uint32_t y)
        {
            if (!capturing)
                return;
            const uint32_t page =
                debugPageId(psm, base, std::max(width, 1u), x, y);
            if (page < pages.size())
                pages.set(page);
        }
    };

    GSDrawTraceState &drawTrace()
    {
        static GSDrawTraceState trace;
        return trace;
    }

    struct GSFeedbackTraceState
    {
        std::ofstream output;
        uint64_t index = 0u;
        bool initialized = false;

        void initialize()
        {
            if (initialized)
                return;

            initialized = true;
            const char *path = std::getenv("PS2X_GS_FEEDBACK_TRACE");
            if (!path || path[0] == '\0')
                return;

            output.open(path, std::ios::out | std::ios::trunc);
            if (output)
            {
                output << "index,type,iip,abe,fst,ctxt,"
                          "fbp,fbw,fpsm,fbmsk,zbp,zpsm,zmask,"
                          "tbp0,tbw,tpsm,tw,th,tcc,tfx,"
                          "alpha,test,ofx,ofy,sx0,sx1,sy0,sy1,"
                          "x0,y0,z0,u0,v0,x1,y1,z1,u1,v1,x2,y2,z2,u2,v2\n";
            }
        }

        void record(const GSPrimReg &prim, const GSContext &ctx, const GSVertex *vertices)
        {
            initialize();
            if (!output || !prim.tme)
                return;

            const bool directColorOrDepthTexture =
                ctx.tex0.psm == GS_PSM_CT32 ||
                ctx.tex0.psm == GS_PSM_CT24 ||
                ctx.tex0.psm == GS_PSM_CT16 ||
                ctx.tex0.psm == GS_PSM_CT16S ||
                ctx.tex0.psm == GS_PSM_Z32 ||
                ctx.tex0.psm == GS_PSM_Z24 ||
                ctx.tex0.psm == GS_PSM_Z16 ||
                ctx.tex0.psm == GS_PSM_Z16S;
            if (!directColorOrDepthTexture)
                return;

            const GSVertex &v0 = vertices[0];
            const GSVertex &v1 = vertices[1];
            const GSVertex &v2 = vertices[2];
            output << index++ << ','
                   << static_cast<uint32_t>(prim.type) << ','
                   << static_cast<uint32_t>(prim.iip) << ','
                   << static_cast<uint32_t>(prim.abe) << ','
                   << static_cast<uint32_t>(prim.fst) << ','
                   << static_cast<uint32_t>(prim.ctxt) << ','
                   << ctx.frame.fbp << ','
                   << ctx.frame.fbw << ','
                   << static_cast<uint32_t>(ctx.frame.psm) << ','
                   << ctx.frame.fbmsk << ','
                   << ctx.zbuf.zbp << ','
                   << static_cast<uint32_t>(ctx.zbuf.psm) << ','
                   << static_cast<uint32_t>(ctx.zbuf.zmask) << ','
                   << ctx.tex0.tbp0 << ','
                   << static_cast<uint32_t>(ctx.tex0.tbw) << ','
                   << static_cast<uint32_t>(ctx.tex0.psm) << ','
                   << static_cast<uint32_t>(ctx.tex0.tw) << ','
                   << static_cast<uint32_t>(ctx.tex0.th) << ','
                   << static_cast<uint32_t>(ctx.tex0.tcc) << ','
                   << static_cast<uint32_t>(ctx.tex0.tfx) << ','
                   << ctx.alpha << ','
                   << ctx.test << ','
                   << (ctx.xyoffset.ofx >> 4) << ','
                   << (ctx.xyoffset.ofy >> 4) << ','
                   << ctx.scissor.x0 << ','
                   << ctx.scissor.x1 << ','
                   << ctx.scissor.y0 << ','
                   << ctx.scissor.y1 << ','
                   << v0.x << ',' << v0.y << ',' << v0.z << ','
                   << (v0.u >> 4) << ',' << (v0.v >> 4) << ','
                   << v1.x << ',' << v1.y << ',' << v1.z << ','
                   << (v1.u >> 4) << ',' << (v1.v >> 4) << ','
                   << v2.x << ',' << v2.y << ',' << v2.z << ','
                   << (v2.u >> 4) << ',' << (v2.v >> 4) << '\n';
            output.flush();
        }
    };

    GSFeedbackTraceState &feedbackTrace()
    {
        static GSFeedbackTraceState trace;
        return trace;
    }

    struct GSClutTraceState
    {
        std::ofstream output;
        uint64_t index = 0u;
        uint64_t limit = 100000u;
        bool initialized = false;

        void initialize()
        {
            if (initialized)
                return;

            initialized = true;
            const char *path = std::getenv("PS2X_GS_CLUT_TRACE");
            if (!path || path[0] == '\0')
                return;

            if (const char *limitText = std::getenv("PS2X_GS_CLUT_TRACE_LIMIT"))
            {
                char *end = nullptr;
                const unsigned long long parsed = std::strtoull(limitText, &end, 0);
                if (end != limitText && end && *end == '\0' && parsed != 0u)
                    limit = parsed;
            }

            output.open(path, std::ios::out | std::ios::trunc);
            if (output)
            {
                output << "index,context,tbp0,tpsm,cbp,cpsm,csm,csa,cld,"
                          "raw_nonzero_bytes,raw_max_byte,decoded_nonzero_entries,"
                          "decoded_alpha_nonzero_entries,decoded_max_alpha\n";
            }
        }

        bool enabled()
        {
            initialize();
            return output.is_open() && index < limit;
        }
    };

    GSClutTraceState &clutTrace()
    {
        static GSClutTraceState trace;
        return trace;
    }

    struct GSDrawDumpState
    {
        std::string directory;
        std::vector<uint64_t> indices;
        bool dumpVram = false;
        bool initialized = false;

        void initialize()
        {
            if (initialized)
                return;

            initialized = true;
            const char *directoryText = std::getenv("PS2X_GS_DRAW_DUMP_DIR");
            const char *indicesText = std::getenv("PS2X_GS_DRAW_DUMP_INDICES");
            if (!directoryText || directoryText[0] == '\0' ||
                !indicesText || indicesText[0] == '\0')
            {
                return;
            }

            directory = directoryText;
            if (const char *dumpVramText = std::getenv("PS2X_GS_DRAW_DUMP_VRAM"))
            {
                dumpVram = dumpVramText[0] != '\0' &&
                           std::strcmp(dumpVramText, "0") != 0;
            }
            std::stringstream stream(indicesText);
            std::string token;
            while (std::getline(stream, token, ','))
            {
                char *end = nullptr;
                const unsigned long long parsed = std::strtoull(token.c_str(), &end, 0);
                if (end != token.c_str() && end && *end == '\0')
                    indices.push_back(static_cast<uint64_t>(parsed));
            }
        }

        bool contains(uint64_t index)
        {
            initialize();
            if (indices.empty())
                return false;
            return std::find(indices.begin(), indices.end(), index) != indices.end();
        }
    };

    GSDrawDumpState &drawDump()
    {
        static GSDrawDumpState dump;
        return dump;
    }

    struct GSPixelTraceTarget
    {
        bool enabled = false;
        int x = 0;
        int y = 0;
    };

    const GSPixelTraceTarget &pixelTraceTarget()
    {
        static const GSPixelTraceTarget target = []
        {
            GSPixelTraceTarget value;
            if (const char *text = std::getenv("PS2X_GS_TRACE_PIXEL"))
            {
                value.enabled =
                    std::sscanf(
                        text,
                        "%d,%d",
                        &value.x,
                        &value.y) == 2;
            }
            return value;
        }();
        return target;
    }

    bool tracePixelMatches(int x, int y)
    {
        const GSPixelTraceTarget &target = pixelTraceTarget();
        return target.enabled && x == target.x && y == target.y;
    }

    void dumpDrawFramebuffer(GS *gs,
                             const GSContext &ctx,
                             uint64_t index,
                             const char *phase)
    {
        GSDrawDumpState &dump = drawDump();
        if (!dump.contains(index))
            return;

        constexpr uint32_t width = 640u;
        constexpr uint32_t height = 448u;
        std::ostringstream path;
        path << dump.directory
             << "/draw-" << index
             << '-' << phase
             << "-fbp-" << ctx.frame.fbp
             << ".ppm";
        std::ofstream output(path.str(), std::ios::out | std::ios::binary | std::ios::trunc);
        if (!output)
            return;

        std::ostringstream alphaPath;
        alphaPath << dump.directory
                  << "/draw-" << index
                  << '-' << phase
                  << "-fbp-" << ctx.frame.fbp
                  << "-alpha.pgm";
        std::ofstream alphaOutput(alphaPath.str(),
                                  std::ios::out | std::ios::binary | std::ios::trunc);

        output << "P6\n" << width << ' ' << height << "\n255\n";
        if (alphaOutput)
            alphaOutput << "P5\n" << width << ' ' << height << "\n255\n";

        const uint32_t base = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
        const uint32_t bufferWidth = std::max<uint32_t>(ctx.frame.fbw, 1u);
        for (uint32_t y = 0; y < height; ++y)
        {
            for (uint32_t x = 0; x < width; ++x)
            {
                uint32_t rgba = gs->ReadVram(ctx.frame.psm, base, bufferWidth, x, y);
                if (bitsPerPixel(ctx.frame.psm) == 16)
                    rgba = Rgba5551ToRgba8888(static_cast<uint16_t>(rgba));
                const uint8_t rgb[3] = {
                    static_cast<uint8_t>(rgba),
                    static_cast<uint8_t>(rgba >> 8),
                    static_cast<uint8_t>(rgba >> 16)};
                output.write(reinterpret_cast<const char *>(rgb), sizeof(rgb));
                if (alphaOutput)
                {
                    const uint8_t alpha = static_cast<uint8_t>(rgba >> 24);
                    alphaOutput.write(reinterpret_cast<const char *>(&alpha),
                                      sizeof(alpha));
                }
            }
        }
    }

    PS2X_GS_ALWAYS_INLINE bool passesAlphaTest(
        uint64_t testReg,
        uint8_t alpha)
    {
        if ((testReg & 0x1u) == 0u)
            return true;

        const uint8_t atst = static_cast<uint8_t>((testReg >> 1) & 0x7u);
        const uint8_t aref = static_cast<uint8_t>((testReg >> 4) & 0xFFu);

        switch (atst)
        {
        case 0:
            return false;
        case 1:
            return true;
        case 2:
            return alpha < aref;
        case 3:
            return alpha <= aref;
        case 4:
            return alpha == aref;
        case 5:
            return alpha >= aref;
        case 6:
            return alpha > aref;
        case 7:
            return alpha != aref;
        default:
            return true;
        }
    }

    struct AlphaTestResult
    {
        bool writeFramebuffer;
        bool writeDepth;
        bool preserveDestinationAlpha;
    };

    PS2X_GS_ALWAYS_INLINE AlphaTestResult classifyAlphaTest(
        uint64_t testReg,
        uint8_t alpha)
    {
        const bool pass = passesAlphaTest(testReg, alpha);
        if (pass)
            return {true, true, false};

        // TEST.AFAIL controls what happens when the alpha comparison fails.
        switch (static_cast<uint8_t>((testReg >> 12) & 0x3u))
        {
        case 1: // FB_ONLY
            return {true, false, false};
        case 2: // ZB_ONLY
            return {false, true, false};
        case 3: // RGB_ONLY
            return {true, false, true};
        case 0: // KEEP
        default:
            return {false, false, false};
        }
    }

    bool destinationAlphaTestEnabled(uint64_t testReg, uint8_t framePsm)
    {
        return ((testReg >> 14u) & 1u) != 0u &&
               framePsm != GS_PSM_CT24;
    }

    bool passesDestinationAlphaTest(uint64_t testReg,
                                    uint8_t framePsm,
                                    uint32_t framebufferValue)
    {
        if (!destinationAlphaTestEnabled(testReg, framePsm))
            return true;

        const uint32_t expected = (testReg >> 15u) & 1u;
        switch (framePsm)
        {
        case GS_PSM_CT32:
            return ((framebufferValue >> 31u) & 1u) == expected;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            return ((framebufferValue >> 15u) & 1u) == expected;
        default:
            return true;
        }
    }

    struct TextureCombineResult
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };

    PS2X_GS_ALWAYS_INLINE uint8_t modulateTextureChannelFixed(
        uint8_t texture,
        int vertexFixed)
    {
        // The GS color interpolator carries seven fractional bits into the
        // texture function. Truncating the vertex color to eight bits before
        // modulation loses visible precision on Gouraud-shaded primitives.
        const int fixed = clampInt(vertexFixed, 0, 0x7FFF);
        return clampU8((static_cast<int>(texture) * fixed) >> 14);
    }

    PS2X_GS_ALWAYS_INLINE TextureCombineResult combineTextureFixed(
        const GSTex0Reg &tex,
        int vr,
        int vg,
        int vb,
        int va,
        uint8_t tr,
        uint8_t tg,
        uint8_t tb,
        uint8_t ta)
    {
        const bool textureHasAlpha = tex.tcc != 0u;
        const uint8_t vertexAlpha = clampU8(va >> 7);
        TextureCombineResult out{
            tr, tg, tb, textureHasAlpha ? ta : vertexAlpha};

        switch (tex.tfx)
        {
        case 0: // MODULATE
            out.r = modulateTextureChannelFixed(tr, vr);
            out.g = modulateTextureChannelFixed(tg, vg);
            out.b = modulateTextureChannelFixed(tb, vb);
            out.a = textureHasAlpha
                        ? modulateTextureChannelFixed(ta, va)
                        : vertexAlpha;
            break;
        case 1: // DECAL
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : vertexAlpha;
            break;
        case 2: // HIGHLIGHT
            out.r = clampU8(
                modulateTextureChannelFixed(tr, vr) + vertexAlpha);
            out.g = clampU8(
                modulateTextureChannelFixed(tg, vg) + vertexAlpha);
            out.b = clampU8(
                modulateTextureChannelFixed(tb, vb) + vertexAlpha);
            out.a = textureHasAlpha
                        ? clampU8(ta + vertexAlpha)
                        : vertexAlpha;
            break;
        case 3: // HIGHLIGHT2
            out.r = clampU8(
                modulateTextureChannelFixed(tr, vr) + vertexAlpha);
            out.g = clampU8(
                modulateTextureChannelFixed(tg, vg) + vertexAlpha);
            out.b = clampU8(
                modulateTextureChannelFixed(tb, vb) + vertexAlpha);
            out.a = textureHasAlpha ? ta : vertexAlpha;
            break;
        default:
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : vertexAlpha;
            break;
        }

        return out;
    }

    TextureCombineResult combineTexture(const GSTex0Reg &tex,
                                        uint8_t vr,
                                        uint8_t vg,
                                        uint8_t vb,
                                        uint8_t va,
                                        uint8_t tr,
                                        uint8_t tg,
                                        uint8_t tb,
                                        uint8_t ta)
    {
        return combineTextureFixed(
            tex,
            static_cast<int>(vr) << 7,
            static_cast<int>(vg) << 7,
            static_cast<int>(vb) << 7,
            static_cast<int>(va) << 7,
            tr,
            tg,
            tb,
            ta);
    }

    uint32_t swizzleClutIndexCSM1(uint32_t index)
    {
        return (index & 0xE7u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
    }

    // TODO: clut cache
    uint32_t resolveClutIndex(uint8_t index, uint8_t csm, uint8_t csa, uint8_t sourcePsm)
    {
        uint32_t clutIndex = static_cast<uint32_t>(index);

        switch (sourcePsm)
        {
        case GS_PSM_T4:
        case GS_PSM_T4HH:
        case GS_PSM_T4HL:
        {
            clutIndex = (static_cast<uint32_t>(csa) << 4u) | (clutIndex & 0x0Fu);

            if (csm == 0u)
                clutIndex = swizzleClutIndexCSM1(clutIndex);
        }
        break;
        case GS_PSM_T8:
        case GS_PSM_T8H:
            if (csm == 0)
                clutIndex = swizzleClutIndexCSM1(clutIndex);
            break;
        default:
            break;
        }

        return clutIndex;
    }

    bool tex1UsesLinearFilter(uint64_t tex1)
    {
        const uint8_t mmag = static_cast<uint8_t>((tex1 >> 5) & 0x1u);
        const uint8_t mmin = static_cast<uint8_t>((tex1 >> 6) & 0x7u);
        return mmag != 0u || mmin == 1u || (mmin & 0x4u) != 0u;
    }

    int wrapTextureCoordinate(int coordinate,
                              int textureSize,
                              uint8_t mode,
                              uint16_t regionMin,
                              uint16_t regionMax)
    {
        const int size = std::max(textureSize, 1);
        const uint32_t textureMask = static_cast<uint32_t>(size - 1);

        switch (mode)
        {
        case 0: // REPEAT
            return static_cast<int>(static_cast<uint32_t>(coordinate) & textureMask);
        case 1: // CLAMP
            return clampInt(coordinate, 0, size - 1);
        case 2: // REGION_CLAMP
            return clampInt(coordinate, regionMin, regionMax);
        case 3: // REGION_REPEAT
        {
            const uint32_t mask = static_cast<uint32_t>(regionMin) & textureMask;
            return static_cast<int>((static_cast<uint32_t>(coordinate) & mask) |
                                    static_cast<uint32_t>(regionMax));
        }
        default:
            return clampInt(coordinate, 0, size - 1);
        }
    }

    int arithmeticShiftRight4(int value)
    {
        // The GS filter uses signed 16-bit arithmetic shifts. Spell out floor
        // division so the result does not depend on the implementation's
        // handling of right shifts of negative integers.
        return value >= 0 ? (value / 16) : -((-value + 15) / 16);
    }

    int floorFixed16_16(int32_t value)
    {
        int quotient = value / 65536;
        if (value < 0 && (value % 65536) != 0)
            --quotient;
        return quotient;
    }

    float approximateLog2Precision3(float value)
    {
        // The GS software reference uses a deliberately low-order mantissa
        // polynomial for LOD. Near a half-level boundary, std::log2 can pick
        // the adjacent mip even though both values look nearly identical.
        uint32_t bits = 0u;
        std::memcpy(&bits, &value, sizeof(bits));
        bits &= 0x7FFFFFFFu;

        const int exponent =
            static_cast<int>((bits >> 23u) & 0xFFu) - 127;
        const uint32_t mantissaBits =
            (bits & 0x007FFFFFu) | 0x3F800000u;
        float mantissa = 1.0f;
        std::memcpy(
            &mantissa, &mantissaBits, sizeof(mantissa));

        float polynomial =
            0.204446009836232697516f * mantissa +
            -1.04913055217340124191f;
        polynomial =
            polynomial * mantissa +
            2.28330284476918490682f;
        polynomial *= mantissa - 1.0f;
        return polynomial + static_cast<float>(exponent);
    }

    uint8_t lerpChannel4(uint8_t c00,
                         uint8_t c10,
                         uint8_t c01,
                         uint8_t c11,
                         uint8_t weightU,
                         uint8_t weightV)
    {
        auto lerp = [](int from, int to, int weight)
        {
            return from + arithmeticShiftRight4((to - from) * weight);
        };

        const int top = lerp(c00, c10, weightU);
        const int bottom = lerp(c01, c11, weightU);
        return clampU8(lerp(top, bottom, weightV));
    }

    uint32_t linearColor4(uint32_t from,
                          uint32_t to,
                          uint8_t weight)
    {
#if defined(__SSE4_1__)
        const __m128i fromChannels =
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(from));
        const __m128i toChannels =
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(to));
        __m128i result = _mm_add_epi32(
            fromChannels,
            _mm_srai_epi32(
                _mm_mullo_epi32(
                    _mm_sub_epi32(toChannels, fromChannels),
                    _mm_set1_epi32(weight)),
                4));
        result = _mm_min_epi32(
            _mm_max_epi32(result, _mm_setzero_si128()),
            _mm_set1_epi32(255));
        return static_cast<uint32_t>(_mm_cvtsi128_si32(
            _mm_packus_epi16(
                _mm_packus_epi32(result, result),
                _mm_setzero_si128())));
#else
        const uint8_t r = clampU8(
            static_cast<uint8_t>(from) +
            arithmeticShiftRight4(
                (static_cast<uint8_t>(to) -
                 static_cast<uint8_t>(from)) *
                weight));
        const uint8_t g = clampU8(
            static_cast<uint8_t>(from >> 8u) +
            arithmeticShiftRight4(
                (static_cast<uint8_t>(to >> 8u) -
                 static_cast<uint8_t>(from >> 8u)) *
                weight));
        const uint8_t b = clampU8(
            static_cast<uint8_t>(from >> 16u) +
            arithmeticShiftRight4(
                (static_cast<uint8_t>(to >> 16u) -
                 static_cast<uint8_t>(from >> 16u)) *
                weight));
        const uint8_t a = clampU8(
            static_cast<uint8_t>(from >> 24u) +
            arithmeticShiftRight4(
                (static_cast<uint8_t>(to >> 24u) -
                 static_cast<uint8_t>(from >> 24u)) *
                weight));
        return static_cast<uint32_t>(r) |
               (static_cast<uint32_t>(g) << 8u) |
               (static_cast<uint32_t>(b) << 16u) |
               (static_cast<uint32_t>(a) << 24u);
#endif
    }

    uint32_t bilinearColor4(uint32_t c00,
                            uint32_t c10,
                            uint32_t c01,
                            uint32_t c11,
                            uint8_t weightU,
                            uint8_t weightV)
    {
#if defined(__SSE4_1__)
        const __m128i zero = _mm_setzero_si128();
        const __m128i maximum = _mm_set1_epi32(255);
        const __m128i u = _mm_set1_epi32(weightU);
        const __m128i v = _mm_set1_epi32(weightV);
        const __m128i color00 =
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(c00));
        const __m128i color10 =
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(c10));
        const __m128i color01 =
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(c01));
        const __m128i color11 =
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(c11));
        const __m128i top = _mm_add_epi32(
            color00,
            _mm_srai_epi32(
                _mm_mullo_epi32(
                    _mm_sub_epi32(color10, color00), u),
                4));
        const __m128i bottom = _mm_add_epi32(
            color01,
            _mm_srai_epi32(
                _mm_mullo_epi32(
                    _mm_sub_epi32(color11, color01), u),
                4));
        __m128i result = _mm_add_epi32(
            top,
            _mm_srai_epi32(
                _mm_mullo_epi32(_mm_sub_epi32(bottom, top), v),
                4));
        result = _mm_min_epi32(
            _mm_max_epi32(result, zero), maximum);
        return static_cast<uint32_t>(_mm_cvtsi128_si32(
            _mm_packus_epi16(
                _mm_packus_epi32(result, result), zero)));
#else
        const uint8_t r = lerpChannel4(
            static_cast<uint8_t>(c00),
            static_cast<uint8_t>(c10),
            static_cast<uint8_t>(c01),
            static_cast<uint8_t>(c11),
            weightU,
            weightV);
        const uint8_t g = lerpChannel4(
            static_cast<uint8_t>(c00 >> 8u),
            static_cast<uint8_t>(c10 >> 8u),
            static_cast<uint8_t>(c01 >> 8u),
            static_cast<uint8_t>(c11 >> 8u),
            weightU,
            weightV);
        const uint8_t b = lerpChannel4(
            static_cast<uint8_t>(c00 >> 16u),
            static_cast<uint8_t>(c10 >> 16u),
            static_cast<uint8_t>(c01 >> 16u),
            static_cast<uint8_t>(c11 >> 16u),
            weightU,
            weightV);
        const uint8_t a = lerpChannel4(
            static_cast<uint8_t>(c00 >> 24u),
            static_cast<uint8_t>(c10 >> 24u),
            static_cast<uint8_t>(c01 >> 24u),
            static_cast<uint8_t>(c11 >> 24u),
            weightU,
            weightV);

        return static_cast<uint32_t>(r) |
               (static_cast<uint32_t>(g) << 8u) |
               (static_cast<uint32_t>(b) << 16u) |
               (static_cast<uint32_t>(a) << 24u);
#endif
    }

    PS2X_GS_ALWAYS_INLINE uint32_t blendSourceOverRgb(
        uint32_t source,
        uint32_t destination,
        uint8_t alpha)
    {
#if defined(__SSE4_1__)
        const __m128i sourceChannels =
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(source));
        const __m128i destinationChannels =
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(destination));
        __m128i result = _mm_add_epi32(
            destinationChannels,
            _mm_srai_epi32(
                _mm_mullo_epi32(
                    _mm_sub_epi32(
                        sourceChannels, destinationChannels),
                    _mm_set1_epi32(alpha)),
                7));
        result = _mm_min_epi32(
            _mm_max_epi32(result, _mm_setzero_si128()),
            _mm_set1_epi32(255));
        return static_cast<uint32_t>(_mm_cvtsi128_si32(
            _mm_packus_epi16(
                _mm_packus_epi32(result, result),
                _mm_setzero_si128())));
#else
        uint32_t result = source & 0xFF000000u;
        for (uint32_t shift = 0u; shift < 24u; shift += 8u)
        {
            const int sourceChannel =
                static_cast<int>((source >> shift) & 0xFFu);
            const int destinationChannel =
                static_cast<int>((destination >> shift) & 0xFFu);
            const int blended = clampInt(
                (((sourceChannel - destinationChannel) * alpha) >> 7) +
                    destinationChannel,
                0,
                255);
            result |= static_cast<uint32_t>(blended) << shift;
        }
        return result;
#endif
    }

    constexpr uint32_t kGsBlockCount = (4u * 1024u * 1024u) / 256u;
    constexpr uint32_t kRasterStripeHeight = 16u;

    struct PackedSpriteKernelSelection
    {
        GSRasterizerDetail::PackedSpriteKernelImplementation
            implementation =
                GSRasterizerDetail::
                    PackedSpriteKernelImplementation::Scalar;
        GSRasterizerPacked::DrawKernel kernel =
            GSRasterizerPacked::drawScalar;
    };

    PackedSpriteKernelSelection selectPackedSpriteKernel(
        GSRasterizerDetail::PackedSpriteKernelOverride overrideMode)
    {
        using Implementation =
            GSRasterizerDetail::PackedSpriteKernelImplementation;
        using Override =
            GSRasterizerDetail::PackedSpriteKernelOverride;

        if (overrideMode == Override::Automatic)
        {
            overrideMode =
                environmentPackedSpriteKernelOverride();
        }

        if (overrideMode == Override::ForceScalar ||
            overrideMode == Override::ForceReference)
        {
            return {
                Implementation::Scalar,
                GSRasterizerPacked::drawScalar,
            };
        }

#if PS2X_GS_HAS_SSE41_VARIANT
        if (overrideMode == Override::ForceSse41)
        {
            return {
                Implementation::Sse41,
                GSRasterizerPacked::drawSse41,
            };
        }
#endif

#if PS2X_GS_HAS_AVX2_KERNEL
        if (overrideMode == Override::ForceAvx2 &&
            hostSupportsAvx2())
        {
            return {
                Implementation::Avx2,
                GSRasterizerPacked::drawAvx2,
            };
        }
#endif

        if (overrideMode == Override::ForceSse41 ||
            overrideMode == Override::ForceAvx2)
        {
            return {
                Implementation::Scalar,
                GSRasterizerPacked::drawScalar,
            };
        }

#if PS2X_GS_HAS_AVX2_KERNEL
        if (hostSupportsAvx2())
        {
            return {
                Implementation::Avx2,
                GSRasterizerPacked::drawAvx2,
            };
        }
#endif

#if PS2X_GS_HAS_SSE41_VARIANT
        return {
            Implementation::Sse41,
            GSRasterizerPacked::drawSse41,
        };
#else
        return {
            Implementation::Scalar,
            GSRasterizerPacked::drawScalar,
        };
#endif
    }

    using PreparedPackedSprite =
        GSRasterizerPacked::PreparedSprite;

    struct GSBlockRange
    {
        uint32_t start = 0u;
        uint32_t count = 0u;
    };

    bool blockRangesOverlap(GSBlockRange lhs, GSBlockRange rhs)
    {
        if (lhs.count == 0u || rhs.count == 0u)
            return false;
        if (lhs.count >= kGsBlockCount || rhs.count >= kGsBlockCount)
            return true;

        lhs.start %= kGsBlockCount;
        rhs.start %= kGsBlockCount;

        struct LinearRange
        {
            uint32_t begin;
            uint32_t end;
        };
        auto split = [](GSBlockRange range,
                        std::array<LinearRange, 2> &out) -> uint32_t
        {
            const uint32_t end = range.start + range.count;
            if (end <= kGsBlockCount)
            {
                out[0] = {range.start, end};
                return 1u;
            }

            out[0] = {range.start, kGsBlockCount};
            out[1] = {0u, end - kGsBlockCount};
            return 2u;
        };

        std::array<LinearRange, 2> lhsParts{};
        std::array<LinearRange, 2> rhsParts{};
        const uint32_t lhsCount = split(lhs, lhsParts);
        const uint32_t rhsCount = split(rhs, rhsParts);
        for (uint32_t i = 0u; i < lhsCount; ++i)
        {
            for (uint32_t j = 0u; j < rhsCount; ++j)
            {
                if (lhsParts[i].begin < rhsParts[j].end &&
                    rhsParts[j].begin < lhsParts[i].end)
                {
                    return true;
                }
            }
        }
        return false;
    }

    uint32_t surfaceBlockCount(uint8_t psm,
                               uint32_t widthUnits,
                               uint32_t maximumX,
                               uint32_t maximumY)
    {
        uint32_t pageWidth = 64u;
        uint32_t pageHeight = 32u;
        uint32_t pagesPerRow = std::max(widthUnits, 1u);
        switch (psm)
        {
        case GS_PSM_T8:
            pageWidth = 128u;
            pageHeight = 64u;
            pagesPerRow = std::max(widthUnits >> 1u, 1u);
            break;
        case GS_PSM_CT32:
        case GS_PSM_CT24:
        case GS_PSM_Z32:
        case GS_PSM_Z24:
        default:
            break;
        }

        const uint64_t maximumPage =
            static_cast<uint64_t>(maximumY / pageHeight) *
                pagesPerRow +
            maximumX / pageWidth;
        return static_cast<uint32_t>(std::min<uint64_t>(
            (maximumPage + 1u) * 32u,
            kGsBlockCount));
    }

    GSBlockRange framebufferRange(const GSContext &ctx)
    {
        return {
            GSInternal::framePageBaseToBlock(ctx.frame.fbp),
            surfaceBlockCount(ctx.frame.psm,
                              ctx.frame.fbw,
                              ctx.scissor.x1,
                              ctx.scissor.y1),
        };
    }

    GSBlockRange depthRange(const GSContext &ctx)
    {
        return {
            GSInternal::framePageBaseToBlock(ctx.zbuf.zbp),
            surfaceBlockCount(ctx.zbuf.psm,
                              ctx.frame.fbw,
                              ctx.scissor.x1,
                              ctx.scissor.y1),
        };
    }

    GSBlockRange textureRange(const GSContext &ctx, uint8_t level)
    {
        uint32_t base = ctx.tex0.tbp0;
        uint8_t width = ctx.tex0.tbw;
        if (level != 0u)
        {
            const uint64_t mipRegister =
                level <= 3u ? ctx.miptbp1 : ctx.miptbp2;
            const uint8_t slot =
                static_cast<uint8_t>((level - 1u) % 3u);
            const uint8_t shift = static_cast<uint8_t>(slot * 20u);
            base = static_cast<uint32_t>(
                (mipRegister >> shift) & 0x3FFFu);
            width = static_cast<uint8_t>(
                (mipRegister >> (shift + 14u)) & 0x3Fu);
        }

        const uint32_t textureWidth =
            std::max(1u, (1u << ctx.tex0.tw) >> level);
        const uint32_t textureHeight =
            std::max(1u, (1u << ctx.tex0.th) >> level);
        const uint64_t clamp = ctx.clamp;
        const uint32_t maximumTextureX =
            GSInternal::maximumWrappedTextureCoordinate(
                textureWidth,
                static_cast<uint8_t>(clamp & 0x3u),
                static_cast<uint16_t>(
                    ((clamp >> 4u) & 0x3FFu) >> level),
                static_cast<uint16_t>(
                    ((clamp >> 14u) & 0x3FFu) >> level));
        const uint32_t maximumTextureY =
            GSInternal::maximumWrappedTextureCoordinate(
                textureHeight,
                static_cast<uint8_t>((clamp >> 2u) & 0x3u),
                static_cast<uint16_t>(
                    ((clamp >> 24u) & 0x3FFu) >> level),
                static_cast<uint16_t>(
                    ((clamp >> 34u) & 0x3FFu) >> level));
        return {
            base,
            surfaceBlockCount(ctx.tex0.psm,
                              width,
                              maximumTextureX,
                              maximumTextureY),
        };
    }

    bool rasterizerInstrumentationRequested()
    {
#if AGRESSIVE_LOGS
        return true;
#else
        static const bool requested = []
        {
            constexpr const char *names[] = {
                "PS2X_GS_DRAW_TRACE",
                "PS2X_GS_FEEDBACK_TRACE",
                "PS2X_GS_CLUT_TRACE",
                "PS2X_GS_DRAW_DUMP_DIR",
                "PS2X_GS_TRACE_PIXEL",
            };
            for (const char *name : names)
            {
                const char *value = std::getenv(name);
                if (value && value[0] != '\0')
                    return true;
            }
            return false;
        }();
        return requested;
#endif
    }

    uint32_t requestedRasterThreadCount()
    {
        const char *disabled =
            std::getenv("PS2X_GS_DISABLE_PARALLEL_RASTERIZER");
        if (disabled && disabled[0] != '\0' &&
            std::strcmp(disabled, "0") != 0)
        {
            return 1u;
        }

        const uint32_t hardwareThreads =
            std::max(std::thread::hardware_concurrency(), 1u);
        uint32_t count = std::min(hardwareThreads, 8u);
        if (const char *text =
                std::getenv("PS2X_GS_RASTER_THREADS"))
        {
            char *end = nullptr;
            const unsigned long parsed = std::strtoul(text, &end, 0);
            if (end != text && end && *end == '\0' && parsed != 0u)
            {
                count = static_cast<uint32_t>(
                    std::min<unsigned long>(parsed, 16u));
            }
        }
        return count;
    }
}

struct GSRasterizer::ParallelState
{
    struct DecodedPalette
    {
        std::array<uint32_t, 256> colors{};
        uint64_t generation = 0u;
        uint64_t serial = 0u;
        uint16_t texa = 0u;
        uint8_t sourcePsm = 0u;
        uint8_t csm = 0u;
        uint8_t csa = 0u;
    };

    struct Command
    {
        explicit Command(const GsDrawCommand &draw_)
            : draw(draw_)
        {
        }

        GsDrawCommand draw;
        bool feedbackSnapshot = false;
        size_t paletteIndex = SIZE_MAX;
        uint32_t workerMask = 0u;
    };

    explicit ParallelState(GSRasterizer *owner_, uint32_t count)
        : owner(owner_), workerCount(count)
    {
        const char *stats =
            std::getenv("PS2X_GS_RASTER_BATCH_STATS");
        statsEnabled =
            stats && stats[0] != '\0' &&
            std::strcmp(stats, "0") != 0;
        commands.reserve(1024u);
        palettes.reserve(16u);
        workerCommands.resize(workerCount);
        workerDrawTotals.resize(workerCount);
        workerNanoseconds.resize(workerCount);
        renderGs.reserve(workerCount);
        for (uint32_t i = 0u; i < workerCount; ++i)
            renderGs.emplace_back(std::make_unique<GS>());

        threads.reserve(workerCount - 1u);
        for (uint32_t i = 1u; i < workerCount; ++i)
        {
            threads.emplace_back([this, i]
            {
                workerLoop(i);
            });
        }
    }

    ~ParallelState()
    {
        {
            std::lock_guard<std::mutex> lock(workMutex);
            stopping = true;
            ++generation;
        }
        workReady.notify_all();
        for (std::thread &thread : threads)
        {
            if (thread.joinable())
                thread.join();
        }

        if (statsEnabled)
        {
            std::cerr
                << "[gs:raster-batches] draws=" << queuedDraws
                << " batches=" << batches
                << " parallel=" << parallelBatches
                << " sequential=" << sequentialBatches
                << " parallel-draws=" << parallelDraws
                << " sequential-draws=" << sequentialDraws
                << " max-draws=" << maximumBatchDraws
                << '\n';
            for (uint32_t i = 0u; i < workerCount; ++i)
            {
                std::cerr
                    << "[gs:raster-worker] index=" << i
                    << " draws=" << workerDrawTotals[i]
                    << " milliseconds="
                    << static_cast<double>(workerNanoseconds[i]) /
                           1000000.0
                    << '\n';
            }
        }
    }

    void workerLoop(uint32_t workerIndex)
    {
        uint64_t observedGeneration = 0u;
        std::unique_lock<std::mutex> lock(workMutex);
        for (;;)
        {
            workReady.wait(lock, [&]
            {
                return stopping || generation != observedGeneration;
            });
            if (stopping)
                return;

            observedGeneration = generation;
            lock.unlock();
            const auto start = std::chrono::steady_clock::now();
            for (size_t commandIndex : workerCommands[workerIndex])
            {
                owner->renderQueuedPrimitive(
                    renderGs[workerIndex].get(),
                    commandIndex,
                    workerIndex,
                    workerCount);
            }
            if (statsEnabled)
            {
                workerDrawTotals[workerIndex] +=
                    workerCommands[workerIndex].size();
                workerNanoseconds[workerIndex] +=
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<
                            std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() -
                            start)
                            .count());
            }
            lock.lock();
            if (--workersRemaining == 0u)
                workComplete.notify_one();
        }
    }

    GSRasterizer *owner = nullptr;
    GS *primaryGs = nullptr;
    uint32_t workerCount = 1u;
    bool batchActive = false;
    bool eligibilityCacheValid = false;
    bool cachedEligible = false;
    bool cachedRecursiveFeedback = false;
    GSContext eligibilityContext{};
    GSPrimReg eligibilityPrim{};
    GSBlockRange eligibleFrameRange{};
    GSBlockRange eligibleDepthRange{};
    bool outputGroupValid = false;
    GSFrameReg groupFrame{};
    GSZbufReg groupDepth{};
    GSBlockRange groupFrameRange{};
    GSBlockRange groupDepthRange{};
    std::vector<Command> commands;
    std::vector<DecodedPalette> palettes;
    std::vector<std::vector<size_t>> workerCommands;
    std::vector<std::unique_ptr<GS>> renderGs;
    std::vector<std::thread> threads;
    std::vector<uint64_t> workerDrawTotals;
    std::vector<uint64_t> workerNanoseconds;
    std::mutex workMutex;
    std::condition_variable workReady;
    std::condition_variable workComplete;
    uint64_t generation = 0u;
    uint64_t nextPaletteSerial = 1u;
    uint32_t workersRemaining = 0u;
    bool stopping = false;
    bool statsEnabled = false;
    uint64_t queuedDraws = 0u;
    uint64_t batches = 0u;
    uint64_t parallelBatches = 0u;
    uint64_t sequentialBatches = 0u;
    uint64_t parallelDraws = 0u;
    uint64_t sequentialDraws = 0u;
    size_t maximumBatchDraws = 0u;
};

struct GSRasterizer::BackendState
{
    class SoftwareBackend final : public IGsRasterBackend
    {
    public:
        SoftwareBackend(GSRasterizer &rasterizer_, GS *owner_) noexcept
            : rasterizer(rasterizer_), owner(owner_)
        {
        }

        [[nodiscard]] GsBackendDecision classify(
            const GsDrawCommand &) const override
        {
            return {true, GsFallbackReason::Supported};
        }

        void submit(std::span<const GsDrawCommand> commands) override
        {
            for (const GsDrawCommand &command : commands)
                rasterizer.submitSoftwareCommand(owner, command);
        }

        void flush(GsFlushReason) override
        {
            rasterizer.flushSoftwareDrawBatch(owner);
        }

        [[nodiscard]] size_t pendingCommandCount() const noexcept override
        {
            return rasterizer.softwarePendingCommandCount();
        }

    private:
        GSRasterizer &rasterizer;
        GS *owner = nullptr;
    };

    BackendState(GSRasterizer &rasterizer, GS *owner) noexcept
        : software(rasterizer, owner), router(software)
    {
    }

    SoftwareBackend software;
    std::unique_ptr<GsVulkanRasterBackend> accelerated;
    GsBackendRouter router;
    GsVulkanServiceConfig serviceConfig{};
    GsVulkanRasterBackendConfig vulkanBackendConfig{};
    GsVulkanCapabilityReport capabilityReport{};
    std::string diagnostic;
};

GSRasterizer::GSRasterizer(GS *owner)
    : m_owner(owner),
      m_backendState(std::make_unique<BackendState>(*this, owner))
{
}
GSRasterizer::~GSRasterizer() = default;

GSRasterizer::DebugProgressScope::DebugProgressScope(
    GSRasterizer &rasterizer, GS *owner)
    : m_rasterizer(rasterizer)
{
    m_rasterizer.beginDebugProgress(owner);
}

GSRasterizer::DebugProgressScope::~DebugProgressScope()
{
    m_rasterizer.endDebugProgress();
}

void GSRasterizer::beginDebugProgress(GS *owner)
{
    m_debugProgressOwner = owner;
    m_debugCandidatePixelBatch = 0u;
    m_trackDebugProgress =
        owner &&
        owner->m_progressTrackingEnabled.load(std::memory_order_relaxed);
    if (!m_trackDebugProgress)
    {
        return;
    }

    owner->m_progressDrawsStarted.fetch_add(1u, std::memory_order_relaxed);
    owner->m_progressActiveDraws.fetch_add(1u, std::memory_order_relaxed);
    owner->m_progressActivePrimitive.store(
        static_cast<uint32_t>(owner->m_prim.type), std::memory_order_relaxed);
}

void GSRasterizer::endDebugProgress()
{
    GS *const owner = m_debugProgressOwner;
    if (m_trackDebugProgress && owner)
    {
        if (m_debugCandidatePixelBatch != 0u)
        {
            owner->m_progressCandidatePixels.fetch_add(
                m_debugCandidatePixelBatch, std::memory_order_relaxed);
        }
        owner->m_progressDrawsCompleted.fetch_add(1u, std::memory_order_relaxed);
        owner->m_progressActiveDraws.fetch_sub(1u, std::memory_order_relaxed);
    }
    m_debugProgressOwner = nullptr;
    m_debugCandidatePixelBatch = 0u;
    m_trackDebugProgress = false;
}

bool GSRasterizer::beginDrawBatch(GS *gs)
{
    if (!gs || rasterizerInstrumentationRequested() ||
        !gs->m_debugHistoryPaused)
    {
        return false;
    }

    if (!m_parallelState)
    {
        const uint32_t threadCount = requestedRasterThreadCount();
        if (threadCount <= 1u)
            return false;
        m_parallelState =
            std::make_unique<ParallelState>(this, threadCount);
    }

    if (m_parallelState->batchActive)
        return false;

    m_parallelState->primaryGs = gs;
    m_parallelState->batchActive = true;
    return true;
}

void GSRasterizer::flushSoftwareDrawBatch(GS *gs)
{
    ParallelState *state = m_parallelState.get();
    if (!state || state->commands.empty())
    {
        if (state)
            state->outputGroupValid = false;
        return;
    }

    if (gs && state->primaryGs && gs != state->primaryGs)
        return;

    ++state->batches;
    state->queuedDraws += state->commands.size();
    state->maximumBatchDraws =
        std::max(state->maximumBatchDraws, state->commands.size());

    uint32_t populatedWorkers = 0u;
    for (const auto &commands : state->workerCommands)
        populatedWorkers += commands.empty() ? 0u : 1u;

    constexpr size_t kMinimumParallelDraws = 8u;
    const bool useWorkers =
        state->commands.size() >= kMinimumParallelDraws &&
        populatedWorkers >= 2u;
    if (!useWorkers)
    {
        ++state->sequentialBatches;
        state->sequentialDraws += state->commands.size();
        for (size_t commandIndex = 0u;
             commandIndex < state->commands.size();
             ++commandIndex)
        {
            renderQueuedPrimitive(
                state->renderGs[0].get(),
                commandIndex,
                0u,
                1u);
        }
    }
    else
    {
        ++state->parallelBatches;
        state->parallelDraws += state->commands.size();
        {
            std::lock_guard<std::mutex> lock(state->workMutex);
            state->workersRemaining = state->workerCount - 1u;
            ++state->generation;
        }
        state->workReady.notify_all();

        const auto mainStart =
            std::chrono::steady_clock::now();
        for (size_t commandIndex : state->workerCommands[0])
        {
            renderQueuedPrimitive(
                state->renderGs[0].get(),
                commandIndex,
                0u,
                state->workerCount);
        }
        if (state->statsEnabled)
        {
            state->workerDrawTotals[0] +=
                state->workerCommands[0].size();
            state->workerNanoseconds[0] +=
                static_cast<uint64_t>(
                    std::chrono::duration_cast<
                        std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() -
                        mainStart)
                        .count());
        }

        std::unique_lock<std::mutex> lock(state->workMutex);
        state->workComplete.wait(lock, [&]
        {
            return state->workersRemaining == 0u;
        });
    }

    state->commands.clear();
    state->palettes.clear();
    for (auto &commands : state->workerCommands)
        commands.clear();
    state->outputGroupValid = false;
}

void GSRasterizer::endDrawBatch(GS *gs)
{
    flushDrawBatch(gs, GsFlushReason::Explicit);
    if (m_parallelState)
    {
        m_parallelState->batchActive = false;
        m_parallelState->primaryGs = nullptr;
    }
}

bool GSRasterizer::tryQueuePrimitive(
    GS *gs,
    const GsDrawCommand &command)
{
    ParallelState *state = m_parallelState.get();
    if (!state || !state->batchActive || state->primaryGs != gs)
        return false;

    const GSPrimReg &primitive = command.primitive();
    switch (primitive.type)
    {
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
    case GS_PRIM_SPRITE:
        break;
    default:
        return false;
    }

    const GSContext &ctx = command.context();
    auto sameEligibilityState =
        [&](const GSContext &cached,
            const GSPrimReg &cachedPrim)
    {
        return
            cached.frame.fbp == ctx.frame.fbp &&
            cached.frame.fbw == ctx.frame.fbw &&
            cached.frame.psm == ctx.frame.psm &&
            cached.frame.fbmsk == ctx.frame.fbmsk &&
            cached.scissor.x0 == ctx.scissor.x0 &&
            cached.scissor.x1 == ctx.scissor.x1 &&
            cached.scissor.y0 == ctx.scissor.y0 &&
            cached.scissor.y1 == ctx.scissor.y1 &&
            cached.zbuf.zbp == ctx.zbuf.zbp &&
            cached.zbuf.psm == ctx.zbuf.psm &&
            cached.zbuf.zmask == ctx.zbuf.zmask &&
            cached.tex0.tbp0 == ctx.tex0.tbp0 &&
            cached.tex0.tbw == ctx.tex0.tbw &&
            cached.tex0.psm == ctx.tex0.psm &&
            cached.tex0.tw == ctx.tex0.tw &&
            cached.tex0.th == ctx.tex0.th &&
            cached.tex1 == ctx.tex1 &&
            cached.miptbp1 == ctx.miptbp1 &&
            cached.miptbp2 == ctx.miptbp2 &&
            cached.clamp == ctx.clamp &&
            cached.test == ctx.test &&
            cached.alpha == ctx.alpha &&
            cachedPrim.tme == primitive.tme &&
            cachedPrim.abe == primitive.abe;
    };
    if (!state->eligibilityCacheValid ||
        !sameEligibilityState(
            state->eligibilityContext,
            state->eligibilityPrim))
    {
        state->eligibilityCacheValid = true;
        state->eligibilityContext = ctx;
        state->eligibilityPrim = primitive;

        bool eligible =
            ctx.frame.psm == GS_PSM_CT32 &&
            ctx.frame.fbmsk == 0u &&
            !destinationAlphaTestEnabled(
                ctx.test, GS_PSM_CT32) &&
            (ctx.zbuf.psm == GS_PSM_Z24 ||
             ctx.zbuf.psm == GS_PSM_Z32) &&
            (!primitive.abe ||
             (ctx.alpha & 0xFFu) == 0x44u) &&
            (!primitive.tme ||
             ctx.tex0.psm == GS_PSM_T8 ||
             ctx.tex0.psm == GS_PSM_CT32);

        const GSBlockRange frame = framebufferRange(ctx);
        const bool depthTestEnabled =
            ((ctx.test >> 16u) & 1u) != 0u;
        const uint8_t depthTestMethod =
            static_cast<uint8_t>((ctx.test >> 17u) & 0x3u);
        const bool depthSurfaceUsed =
            depthTestEnabled &&
            (depthTestMethod >= 2u ||
             (depthTestMethod != 0u && !ctx.zbuf.zmask));
        const GSBlockRange depth =
            depthSurfaceUsed ? depthRange(ctx) : GSBlockRange{};
        const bool recursiveFeedback =
            primitive.tme &&
            ctx.tex0.tbp0 == frame.start;
        eligible =
            eligible && !blockRangesOverlap(frame, depth);
        if (eligible && primitive.tme)
        {
            uint8_t maximumLevel = 0u;
            const uint8_t minificationFilter =
                static_cast<uint8_t>(
                    (ctx.tex1 >> 6u) & 0x7u);
            if (minificationFilter >= 2u &&
                minificationFilter <= 5u)
            {
                maximumLevel = static_cast<uint8_t>(
                    std::min<uint64_t>(
                        (ctx.tex1 >> 2u) & 0x7u, 6u));
            }
            for (uint8_t level = 0u;
                 level <= maximumLevel;
                 ++level)
            {
                const GSBlockRange texture =
                    textureRange(ctx, level);
                if (!recursiveFeedback &&
                    (blockRangesOverlap(texture, frame) ||
                     blockRangesOverlap(texture, depth)))
                {
                    eligible = false;
                    break;
                }
            }
        }
        state->cachedEligible = eligible;
        state->cachedRecursiveFeedback =
            eligible && recursiveFeedback;
        state->eligibleFrameRange = frame;
        state->eligibleDepthRange = depth;
    }
    if (!state->cachedEligible)
        return false;

    const GSBlockRange frame = state->eligibleFrameRange;
    const GSBlockRange depth = state->eligibleDepthRange;

    const GsDrawBounds &bounds = command.bounds();
    const int minimumX = bounds.x0;
    const int maximumX = bounds.x1;
    const int minimumY = bounds.y0;
    const int maximumY = bounds.y1;

    const bool recursiveTextureDraw =
        state->cachedRecursiveFeedback;
    if (gs->m_hasPreferredDisplaySource &&
        ctx.frame.fbp == gs->m_preferredDisplayDestFbp)
    {
        gs->m_hasPreferredDisplaySource = false;
    }
    if (bounds.empty())
    {
        return true;
    }

    const bool sameOutputGroup =
        state->outputGroupValid &&
        state->groupFrame.fbp == ctx.frame.fbp &&
        state->groupFrame.fbw == ctx.frame.fbw &&
        state->groupFrame.psm == ctx.frame.psm &&
        state->groupDepth.zbp == ctx.zbuf.zbp &&
        state->groupDepth.psm == ctx.zbuf.psm;
    if (state->outputGroupValid && !sameOutputGroup)
        flushDrawBatch(gs, GsFlushReason::ResourceHazard);
    if (!state->outputGroupValid)
    {
        state->groupFrame = ctx.frame;
        state->groupDepth = ctx.zbuf;
        state->groupFrameRange = frame;
        state->groupDepthRange = depth;
        state->outputGroupValid = true;
    }

    uint32_t workerMask = 0u;
    const uint32_t firstStripe =
        static_cast<uint32_t>(minimumY) / kRasterStripeHeight;
    const uint32_t lastStripe =
        static_cast<uint32_t>(maximumY - 1) / kRasterStripeHeight;
    for (uint32_t stripe = firstStripe;
         stripe <= lastStripe;
         ++stripe)
    {
        workerMask |=
            1u << (stripe % state->workerCount);
    }

    ParallelState::Command queuedCommand(command);
    queuedCommand.feedbackSnapshot = recursiveTextureDraw;
    if (primitive.tme && ctx.tex0.psm == GS_PSM_T8)
    {
        prepareDecodedClut(gs);
        size_t paletteIndex = SIZE_MAX;
        if (!state->palettes.empty())
        {
            const ParallelState::DecodedPalette &palette =
                state->palettes.back();
            if (palette.generation == m_decodedClutGeneration &&
                palette.texa == m_decodedClutTexa &&
                palette.sourcePsm == m_decodedClutSourcePsm &&
                palette.csm == m_decodedClutCsm &&
                palette.csa == m_decodedClutCsa)
            {
                paletteIndex = state->palettes.size() - 1u;
            }
        }
        if (paletteIndex == SIZE_MAX)
        {
            ParallelState::DecodedPalette palette;
            palette.colors = m_decodedClut;
            palette.generation = m_decodedClutGeneration;
            palette.serial = state->nextPaletteSerial++;
            palette.texa = m_decodedClutTexa;
            palette.sourcePsm = m_decodedClutSourcePsm;
            palette.csm = m_decodedClutCsm;
            palette.csa = m_decodedClutCsa;
            paletteIndex = state->palettes.size();
            state->palettes.emplace_back(std::move(palette));
        }
        queuedCommand.paletteIndex = paletteIndex;
    }
    queuedCommand.workerMask = workerMask;

    const size_t commandIndex = state->commands.size();
    state->commands.emplace_back(std::move(queuedCommand));
    for (uint32_t worker = 0u;
         worker < state->workerCount;
         ++worker)
    {
        if ((workerMask & (1u << worker)) != 0u)
            state->workerCommands[worker].push_back(commandIndex);
    }
    return true;
}

void GSRasterizer::renderQueuedPrimitive(GS *renderGs,
                                         size_t commandIndex,
                                         uint32_t workerIndex,
                                         uint32_t workerCount)
{
    const ParallelState::Command &command =
        m_parallelState->commands[commandIndex];
    const GsDrawCommand &draw = command.draw;
    const GSPrimReg &primitive = draw.primitive();
    const GSContext &context = draw.context();
    const GsDrawGlobalState &global = draw.globalState();
    renderGs->m_vram = m_parallelState->primaryGs->m_vram;
    renderGs->m_vramSize = m_parallelState->primaryGs->m_vramSize;
    renderGs->m_ctx[primitive.ctxt ? 1 : 0] = context;
    renderGs->m_prim = primitive;
    std::copy_n(draw.vertices().data(), 3u, renderGs->m_vtxQueue);
    renderGs->m_vtxCount = draw.vertexCount();
    renderGs->m_texa = global.texa;
    renderGs->m_texclut = global.texclut;
    renderGs->m_fogColor = global.fogColor;
    renderGs->m_prmodecont = global.prmodecont;
    renderGs->m_pabe = global.pabe;
    renderGs->m_scanMask = global.scanMask;
    renderGs->m_dimx = global.dimx;
    renderGs->m_dither = global.dither;
    renderGs->m_colorClamp = global.colorClamp;

    GSRasterizer &rasterizer = renderGs->m_rasterizer;
    DebugProgressScope progress(
        rasterizer, m_parallelState->primaryGs);
    rasterizer.m_scanlineWorkerIndex = workerIndex;
    rasterizer.m_scanlineWorkerCount = workerCount;
    std::copy_n(draw.fixedX().data(), 3u, rasterizer.m_queuedFixedX);
    std::copy_n(draw.fixedY().data(), 3u, rasterizer.m_queuedFixedY);
    rasterizer.m_queuedFixedVerticesValid = true;
    rasterizer.m_textureReadVram =
        command.feedbackSnapshot
            ? m_textureSnapshot.data()
            : nullptr;
    rasterizer.m_feedbackSnapshotValid = false;
    if (command.paletteIndex != SIZE_MAX)
    {
        const ParallelState::DecodedPalette &palette =
            m_parallelState->palettes[command.paletteIndex];
        if (rasterizer.m_queuedPaletteSerial != palette.serial)
        {
            rasterizer.m_decodedClut = palette.colors;
            rasterizer.m_queuedPaletteSerial = palette.serial;
        }
        rasterizer.m_decodedClutGeneration = palette.generation;
        rasterizer.m_decodedClutTexa = palette.texa;
        rasterizer.m_decodedClutSourcePsm = palette.sourcePsm;
        rasterizer.m_decodedClutCsm = palette.csm;
        rasterizer.m_decodedClutCsa = palette.csa;
        rasterizer.m_decodedClutActive = true;
    }
    else
    {
        rasterizer.m_decodedClutActive = false;
    }
    const bool measureRaster =
        m_backendTimingEnabled.load(std::memory_order_relaxed);
    const auto rasterStart = measureRaster
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    if (primitive.type == GS_PRIM_SPRITE)
        rasterizer.drawSprite(renderGs);
    else
        rasterizer.drawTriangle(renderGs);
    if (measureRaster)
    {
        m_softwareRasterHostNanoseconds.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - rasterStart)
                    .count()),
            std::memory_order_relaxed);
    }
}

bool GSRasterizer::ownsScanline(int y) const
{
    if (m_scanlineWorkerCount <= 1u)
        return true;
    return (static_cast<uint32_t>(y) / kRasterStripeHeight) %
               m_scanlineWorkerCount ==
           m_scanlineWorkerIndex;
}

void GSRasterizer::drawPrimitive(GS *gs)
{
    const GsDrawGlobalState globalState{
        .texa = gs->m_texa,
        .texclut = gs->m_texclut,
        .fogColor = gs->m_fogColor,
        .dimx = gs->m_dimx,
        .scanMask = gs->m_scanMask,
        .prmodecont = gs->m_prmodecont,
        .pabe = gs->m_pabe,
        .dither = gs->m_dither,
        .colorClamp = gs->m_colorClamp,
    };
    const GsDrawCommand command = buildGsDrawCommand(
        gs->m_nextDrawSequence++,
        gs->m_prim,
        gs->activeContext(),
        std::span<const GSVertex>(gs->m_vtxQueue, 3u),
        globalState);
    prepareFeedbackSnapshot(gs, command);
    const GsSubmissionResult result =
        m_backendState->router.submit(command);
    if (!result.submitted)
    {
        throw std::runtime_error(
            std::string("GS gpu-strict rejected draw: ") +
            std::string(gsFallbackReasonName(result.decision.reason)));
    }
}

void GSRasterizer::prepareFeedbackSnapshot(
    GS *gs,
    const GsDrawCommand &command)
{
    m_textureReadVram = nullptr;
    const GSContext &context = command.context();
    const uint32_t frameBase =
        GSInternal::framePageBaseToBlock(context.frame.fbp);
    const bool recursiveTextureDraw =
        command.primitive().tme &&
        context.tex0.tbp0 == frameBase &&
        gs && gs->m_vram && gs->m_vramSize != 0u;
    if (!recursiveTextureDraw)
    {
        if (m_feedbackSnapshotValid)
            flushDrawBatch(gs, GsFlushReason::FeedbackSnapshot);
        m_feedbackSnapshotValid = false;
        return;
    }

    const bool sameFeedbackSurface =
        m_feedbackSnapshotValid &&
        m_feedbackTextureBase == context.tex0.tbp0 &&
        m_feedbackFrameBase == frameBase &&
        m_feedbackTexturePsm == context.tex0.psm &&
        m_feedbackFramePsm == context.frame.psm &&
        m_feedbackTextureWidth == context.tex0.tbw &&
        m_feedbackFrameWidth == context.frame.fbw;
    if (sameFeedbackSurface)
        return;

    const GsDrawResources resources = command.resources();
    beginCpuVramAccess(
        gs,
        resources.readPages,
        {},
        GsFlushReason::FeedbackSnapshot);
    m_textureSnapshot.resize(gs->m_vramSize);
    std::memcpy(
        m_textureSnapshot.data(),
        gs->m_vram,
        gs->m_vramSize);
    m_feedbackTextureBase = context.tex0.tbp0;
    m_feedbackFrameBase = frameBase;
    m_feedbackTexturePsm = context.tex0.psm;
    m_feedbackFramePsm = context.frame.psm;
    m_feedbackTextureWidth = context.tex0.tbw;
    m_feedbackFrameWidth = context.frame.fbw;
    m_feedbackSnapshotValid = true;
}

void GSRasterizer::submitSoftwareCommand(
    GS *gs,
    const GsDrawCommand &command)
{
    if (tryQueuePrimitive(gs, command))
        return;
    flushSoftwareDrawBatch(gs);
    renderSoftwarePrimitive(gs, command);
}

void GSRasterizer::recordAcceleratedCommit(
    GS *gs,
    const GsDrawCommand &command)
{
    DebugProgressScope progress(*this, gs);
    const GsDrawBounds &bounds = command.bounds();
    if (!bounds.empty())
    {
        m_debugCandidatePixelBatch +=
            static_cast<uint64_t>(bounds.x1 - bounds.x0) *
            static_cast<uint64_t>(bounds.y1 - bounds.y0);
    }

    // Snapshot lifetime follows frontend command order rather than completion
    // callbacks: a recursive feedback run must keep sampling the image captured
    // before its first draw even while strict/hybrid writes become GPU-newer.
    m_textureReadVram = nullptr;
    if (gs->m_hasPreferredDisplaySource &&
        command.context().frame.fbp ==
            gs->m_preferredDisplayDestFbp)
    {
        gs->m_hasPreferredDisplaySource = false;
    }
}

void GSRasterizer::flushDrawBatch(
    GS *gs,
    GsFlushReason reason)
{
    if (gs && gs != m_owner)
        return;
    m_backendState->router.flush(reason);
}

void GSRasterizer::beginCpuVramAccess(
    GS *gs,
    const GsVramPageMask &readPages,
    const GsVramPageMask &writePages,
    GsFlushReason reason)
{
    if (gs && gs != m_owner)
        return;
    m_backendState->router.beginCpuVramAccess(
        readPages, writePages, reason);
}

void GSRasterizer::endCpuVramAccess(
    GS *gs,
    const GsVramPageMask &writePages)
{
    if (gs && gs != m_owner)
        return;
    m_backendState->router.endCpuVramAccess(writePages);
}

size_t GSRasterizer::softwarePendingCommandCount() const noexcept
{
    return m_parallelState ? m_parallelState->commands.size() : 0u;
}

bool GSRasterizer::configureVulkanRenderer(
    const GsVulkanServiceConfig &config,
    std::string verificationArtifactDirectory)
{
    GsVulkanRasterBackendConfig backendConfig{};
    backendConfig.verificationArtifactDirectory =
        std::move(verificationArtifactDirectory);
    return configureVulkanRenderer(config, std::move(backendConfig));
}

bool GSRasterizer::configureVulkanRenderer(
    const GsVulkanServiceConfig &config,
    GsVulkanRasterBackendConfig backendConfig)
{
    BackendState &state = *m_backendState;
    if (state.router.mode() != GsRendererMode::Software ||
        state.accelerated)
    {
        state.diagnostic =
            "Vulkan renderer configuration is locked after backend creation";
        return false;
    }

    state.serviceConfig = config;
    state.vulkanBackendConfig = std::move(backendConfig);
    state.capabilityReport = {};
    state.diagnostic.clear();
    return true;
}

bool GSRasterizer::setRendererMode(GsRendererMode mode)
{
    BackendState &state = *m_backendState;
    if (mode == GsRendererMode::Software)
    {
        const bool selected = state.router.setMode(mode);
        if (selected)
            state.diagnostic.clear();
        return selected;
    }
    if (mode != GsRendererMode::Hybrid &&
        mode != GsRendererMode::Verify &&
        mode != GsRendererMode::GpuStrict)
    {
        state.diagnostic = "unknown GS renderer mode";
        return false;
    }

    if (!state.accelerated)
    {
        if (!m_owner || !m_owner->m_vram ||
            m_owner->m_vramSize != GS_VULKAN_VRAM_SIZE)
        {
            state.diagnostic =
                "Vulkan renderer requires initialized exact 4 MiB GS VRAM";
            return false;
        }

        GsVulkanRasterBackendConfig backendConfig =
            state.vulkanBackendConfig;
        backendConfig.mode = mode;
        std::unique_ptr<GsVulkanRasterBackend> accelerated =
            GsVulkanRasterBackend::create(
                state.serviceConfig, backendConfig,
                std::span<uint8_t>(
                    m_owner->m_vram, m_owner->m_vramSize),
                [this](const GsDrawCommand &command)
                {
                    renderSoftwarePrimitive(m_owner, command);
                },
                [this](const GsDrawCommand &command)
                {
                    recordAcceleratedCommit(m_owner, command);
                },
                &state.capabilityReport,
                &state.diagnostic);
        if (!accelerated)
            return false;
        state.accelerated = std::move(accelerated);
        state.router.setAcceleratedBackend(state.accelerated.get());
    }

    if (!state.accelerated->healthy())
    {
        state.diagnostic = "Vulkan renderer backend is not healthy";
        return false;
    }
    if (!state.accelerated->setMode(mode) ||
        !state.router.setMode(mode))
    {
        state.diagnostic = "failed to select synchronized Vulkan renderer mode";
        return false;
    }
    state.diagnostic.clear();
    return true;
}

GsRendererMode GSRasterizer::rendererMode() const noexcept
{
    return m_backendState->router.mode();
}

std::string GSRasterizer::rendererDiagnostic() const
{
    return m_backendState->diagnostic;
}

GsVulkanCapabilityReport
GSRasterizer::vulkanRendererCapabilities() const
{
    const BackendState &state = *m_backendState;
    return state.accelerated
        ? state.accelerated->capabilities()
        : state.capabilityReport;
}

GsVulkanServiceStatistics
GSRasterizer::vulkanRendererServiceStatistics() const
{
    const BackendState &state = *m_backendState;
    return state.accelerated
        ? state.accelerated->serviceStatistics()
        : GsVulkanServiceStatistics{};
}

GsVulkanRasterBackendStatistics
GSRasterizer::vulkanRendererBackendStatistics() const
{
    const BackendState &state = *m_backendState;
    return state.accelerated
        ? state.accelerated->backendStatistics()
        : GsVulkanRasterBackendStatistics{};
}

void GSRasterizer::setBackendCountersEnabled(bool enabled) noexcept
{
    m_backendTimingEnabled.store(enabled, std::memory_order_relaxed);
    m_backendState->router.setCountersEnabled(enabled);
}

GsBackendCounters GSRasterizer::backendCounters() const noexcept
{
    GsBackendCounters counters = m_backendState->router.counters();
    counters.softwareRasterHostNanoseconds =
        m_softwareRasterHostNanoseconds.load(std::memory_order_relaxed);
    return counters;
}

void GSRasterizer::resetBackendCounters() noexcept
{
    m_backendState->router.resetCounters();
    m_softwareRasterHostNanoseconds.store(0u, std::memory_order_relaxed);
}

GsReplayRasterizerState GSRasterizer::captureReplayState() const
{
    GsReplayRasterizerState state{};
    state.feedbackTextureBase = m_feedbackTextureBase;
    state.feedbackFrameBase = m_feedbackFrameBase;
    state.feedbackTexturePsm = m_feedbackTexturePsm;
    state.feedbackFramePsm = m_feedbackFramePsm;
    state.feedbackTextureWidth = m_feedbackTextureWidth;
    state.feedbackFrameWidth = m_feedbackFrameWidth;
    state.feedbackSnapshotValid = m_feedbackSnapshotValid;
    if (m_feedbackSnapshotValid)
        state.feedbackVram = m_textureSnapshot;
    state.decodedClut = m_decodedClut;
    state.decodedClutGeneration = m_decodedClutGeneration;
    state.decodedClutTexa = m_decodedClutTexa;
    state.decodedClutSourcePsm = m_decodedClutSourcePsm;
    state.decodedClutCsm = m_decodedClutCsm;
    state.decodedClutCsa = m_decodedClutCsa;
    state.decodedClutActive = m_decodedClutActive;
    return state;
}

bool GSRasterizer::restoreReplayState(
    const GsReplayRasterizerState &state,
    uint32_t vramSize)
{
    if (state.feedbackSnapshotValid &&
        state.feedbackVram.size() != vramSize)
    {
        return false;
    }
    if (!state.feedbackSnapshotValid && !state.feedbackVram.empty())
        return false;

    m_textureSnapshot = state.feedbackVram;
    m_textureReadVram = nullptr;
    m_feedbackTextureBase = state.feedbackTextureBase;
    m_feedbackFrameBase = state.feedbackFrameBase;
    m_feedbackTexturePsm = state.feedbackTexturePsm;
    m_feedbackFramePsm = state.feedbackFramePsm;
    m_feedbackTextureWidth = state.feedbackTextureWidth;
    m_feedbackFrameWidth = state.feedbackFrameWidth;
    m_feedbackSnapshotValid = state.feedbackSnapshotValid;
    m_decodedClut = state.decodedClut;
    m_decodedClutGeneration = state.decodedClutGeneration;
    m_decodedClutTexa = state.decodedClutTexa;
    m_decodedClutSourcePsm = state.decodedClutSourcePsm;
    m_decodedClutCsm = state.decodedClutCsm;
    m_decodedClutCsa = state.decodedClutCsa;
    m_decodedClutActive = state.decodedClutActive;
    m_queuedPaletteSerial = UINT64_MAX;
    m_queuedFixedVerticesValid = false;

    if (m_parallelState)
    {
        m_parallelState->commands.clear();
        m_parallelState->palettes.clear();
        for (auto &commands : m_parallelState->workerCommands)
            commands.clear();
        m_parallelState->batchActive = false;
        m_parallelState->primaryGs = nullptr;
        m_parallelState->eligibilityCacheValid = false;
        m_parallelState->outputGroupValid = false;
    }
    return true;
}

void GSRasterizer::renderSoftwarePrimitive(
    GS *gs,
    const GsDrawCommand &command)
{
    DebugProgressScope progress(*this, gs);

    const GSContext &ctx = command.context();
    const GSPrimReg &primitive = command.primitive();
    prepareDecodedClut(gs);
    m_textureReadVram = nullptr;
    const uint32_t frameBase =
        GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    const bool recursiveTextureDraw =
        primitive.tme &&
        ctx.tex0.tbp0 == frameBase &&
        gs->m_vram &&
        gs->m_vramSize != 0u;
    if (recursiveTextureDraw)
    {
        if (!m_feedbackSnapshotValid ||
            m_textureSnapshot.size() != gs->m_vramSize)
        {
            throw std::logic_error(
                "GS recursive feedback draw has no prepared snapshot");
        }
        m_textureReadVram = m_textureSnapshot.data();
    }
    feedbackTrace().record(
        primitive, ctx, command.vertices().data());
    GSDrawTraceState &trace = drawTrace();
    trace.begin(command);
    trace.feedback = recursiveTextureDraw;
    dumpDrawFramebuffer(gs, ctx, trace.index, "before");
    GSDrawDumpState &dump = drawDump();
    if (dump.dumpVram && dump.contains(trace.index) &&
        gs->m_vram && gs->m_vramSize != 0u)
    {
        std::ostringstream path;
        path << dump.directory
             << "/draw-" << trace.index
             << "-before-vram.bin";
        std::ofstream output(path.str(),
                             std::ios::out | std::ios::binary | std::ios::trunc);
        if (output)
        {
            output.write(reinterpret_cast<const char *>(gs->m_vram),
                         static_cast<std::streamsize>(gs->m_vramSize));
        }
    }

    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t primitiveIndex = s_debugPrimitiveCount.fetch_add(1u, std::memory_order_relaxed);
        if (primitiveIndex < 64u)
        {
            std::cout << "[gs:prim] idx=" << primitiveIndex
                      << " type=" << static_cast<uint32_t>(gs->m_prim.type)
                      << " tme=" << static_cast<uint32_t>(gs->m_prim.tme)
                      << " abe=" << static_cast<uint32_t>(gs->m_prim.abe)
                      << " fst=" << static_cast<uint32_t>(gs->m_prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(gs->m_prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tw=" << static_cast<uint32_t>(ctx.tex0.tw)
                      << " th=" << static_cast<uint32_t>(ctx.tex0.th)
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(gs->m_texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(gs->m_texclut.cou)
                      << " cov=" << gs->m_texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec
                      << " v0=(" << gs->m_vtxQueue[0].x << "," << gs->m_vtxQueue[0].y << ")"
                      << " uv0=(" << (gs->m_vtxQueue[0].u >> 4) << "," << (gs->m_vtxQueue[0].v >> 4) << ")"
                      << " stq0=(" << gs->m_vtxQueue[0].s << "," << gs->m_vtxQueue[0].t << "," << gs->m_vtxQueue[0].q << ")"
                      << " v1=(" << gs->m_vtxQueue[1].x << "," << gs->m_vtxQueue[1].y << ")"
                      << " uv1=(" << (gs->m_vtxQueue[1].u >> 4) << "," << (gs->m_vtxQueue[1].v >> 4) << ")"
                      << " stq1=(" << gs->m_vtxQueue[1].s << "," << gs->m_vtxQueue[1].t << "," << gs->m_vtxQueue[1].q << ")"
                      << " v2=(" << gs->m_vtxQueue[2].x << "," << gs->m_vtxQueue[2].y << ")"
                      << " uv2=(" << (gs->m_vtxQueue[2].u >> 4) << "," << (gs->m_vtxQueue[2].v >> 4) << ")"
                      << " stq2=(" << gs->m_vtxQueue[2].s << "," << gs->m_vtxQueue[2].t << "," << gs->m_vtxQueue[2].q << ")"
                      << " rgba0=(" << static_cast<uint32_t>(gs->m_vtxQueue[0].r) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[0].g) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[0].b) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[0].a) << ")"
                      << " rgba1=(" << static_cast<uint32_t>(gs->m_vtxQueue[1].r) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[1].g) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[1].b) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[1].a) << ")"
                      << " rgba2=(" << static_cast<uint32_t>(gs->m_vtxQueue[2].r) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[2].g) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[2].b) << ","
                      << static_cast<uint32_t>(gs->m_vtxQueue[2].a) << ")"
                      << std::endl;
        }
    });

    PS2_IF_AGRESSIVE_LOGS({
        if ((gs->m_prim.ctxt != 0u || ctx.frame.fbp == 150u) &&
            s_debugContext1PrimitiveCount.fetch_add(1u, std::memory_order_relaxed) < 32u)
        {
            std::cout << "[gs:copy-prim]"
                      << " type=" << static_cast<uint32_t>(gs->m_prim.type)
                      << " tme=" << static_cast<uint32_t>(gs->m_prim.tme)
                      << " abe=" << static_cast<uint32_t>(gs->m_prim.abe)
                      << " fst=" << static_cast<uint32_t>(gs->m_prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(gs->m_prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(gs->m_texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(gs->m_texclut.cou)
                      << " cov=" << gs->m_texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec << std::endl;
        }
    });

    if (gs->m_hasPreferredDisplaySource && ctx.frame.fbp == gs->m_preferredDisplayDestFbp)
    {
        gs->m_hasPreferredDisplaySource = false;
    }

    std::copy_n(command.fixedX().data(), 3u, m_queuedFixedX);
    std::copy_n(command.fixedY().data(), 3u, m_queuedFixedY);
    m_queuedFixedVerticesValid = true;
    const bool measureRaster =
        m_backendTimingEnabled.load(std::memory_order_relaxed);
    const auto rasterStart = measureRaster
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    switch (primitive.type)
    {
    case GS_PRIM_SPRITE:
        drawSprite(gs);
        break;
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
        drawTriangle(gs);
        break;
    case GS_PRIM_LINE:
    case GS_PRIM_LINESTRIP:
        drawLine(gs);
        break;
    case GS_PRIM_POINT:
    {
        const GSVertex &v = command.vertices()[0];
        int px = static_cast<int>(v.x) - (ctx.xyoffset.ofx >> 4);
        int py = static_cast<int>(v.y) - (ctx.xyoffset.ofy >> 4);
        uint8_t r = v.r;
        uint8_t g = v.g;
        uint8_t b = v.b;
        if (primitive.fge)
            applyFog(command.globalState().fogColor, v.fog, r, g, b);
        writePixel(gs, px, py, static_cast<u32>(v.z), r, g, b, v.a);
        break;
    }
    default:
        break;
    }
    if (measureRaster)
    {
        m_softwareRasterHostNanoseconds.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - rasterStart)
                    .count()),
            std::memory_order_relaxed);
    }
    m_queuedFixedVerticesValid = false;
    m_textureReadVram = nullptr;

    dumpDrawFramebuffer(gs, ctx, trace.index, "after");

    if (trace.capturing)
    {
        const GSVertex &v0 = command.vertices()[0];
        const GSVertex &v1 = command.vertices()[1];
        const GSVertex &v2 = command.vertices()[2];
        const GsDrawGlobalState &global = command.globalState();
        const GsDrawBounds &bounds = command.bounds();
        const GsDrawResources resources = command.resources();
        const GsBackendDecision decision =
            classifyGsInitialCt32Sprite(command);
        trace.output
            << trace.index << ','
            << static_cast<uint32_t>(primitive.type) << ','
            << static_cast<uint32_t>(primitive.iip) << ','
            << static_cast<uint32_t>(primitive.tme) << ','
            << static_cast<uint32_t>(primitive.fge) << ','
            << static_cast<uint32_t>(primitive.abe) << ','
            << static_cast<uint32_t>(primitive.fst) << ','
            << static_cast<uint32_t>(primitive.ctxt) << ','
            << global.fogColor << ','
            << ctx.frame.fbp << ','
            << ctx.frame.fbw << ','
            << static_cast<uint32_t>(ctx.frame.psm) << ','
            << ctx.frame.fbmsk << ','
            << ctx.zbuf.zbp << ','
            << static_cast<uint32_t>(ctx.zbuf.psm) << ','
            << static_cast<uint32_t>(ctx.zbuf.zmask) << ','
            << ctx.test << ','
            << ctx.alpha << ','
            << ctx.tex0.tbp0 << ','
            << static_cast<uint32_t>(ctx.tex0.tbw) << ','
            << static_cast<uint32_t>(ctx.tex0.psm) << ','
            << static_cast<uint32_t>(ctx.tex0.tw) << ','
            << static_cast<uint32_t>(ctx.tex0.th) << ','
            << static_cast<uint32_t>(ctx.tex0.tcc) << ','
            << static_cast<uint32_t>(ctx.tex0.tfx) << ','
            << ctx.tex0.cbp << ','
            << static_cast<uint32_t>(ctx.tex0.cpsm) << ','
            << static_cast<uint32_t>(ctx.tex0.csm) << ','
            << static_cast<uint32_t>(ctx.tex0.csa) << ','
            << static_cast<uint32_t>(ctx.tex0.cld) << ','
            << ctx.tex1 << ','
            << ctx.miptbp1 << ','
            << ctx.miptbp2 << ','
            << ctx.clamp << ','
            << static_cast<uint32_t>(global.texclut.cbw) << ','
            << static_cast<uint32_t>(global.texclut.cou) << ','
            << global.texclut.cov << ','
            << (ctx.xyoffset.ofx >> 4) << ','
            << (ctx.xyoffset.ofy >> 4) << ','
            << ctx.scissor.x0 << ','
            << ctx.scissor.x1 << ','
            << ctx.scissor.y0 << ','
            << ctx.scissor.y1 << ','
            << v0.x << ',' << v0.y << ',' << v0.z << ','
            << v1.x << ',' << v1.y << ',' << v1.z << ','
            << v2.x << ',' << v2.y << ',' << v2.z << ','
            << v0.s << ',' << v0.t << ',' << v0.q << ','
            << v0.u << ',' << v0.v << ','
            << static_cast<uint32_t>(v0.fog) << ','
            << static_cast<uint32_t>(v0.r) << ','
            << static_cast<uint32_t>(v0.g) << ','
            << static_cast<uint32_t>(v0.b) << ','
            << static_cast<uint32_t>(v0.a) << ','
            << v1.s << ',' << v1.t << ',' << v1.q << ','
            << v1.u << ',' << v1.v << ','
            << static_cast<uint32_t>(v1.fog) << ','
            << static_cast<uint32_t>(v1.r) << ','
            << static_cast<uint32_t>(v1.g) << ','
            << static_cast<uint32_t>(v1.b) << ','
            << static_cast<uint32_t>(v1.a) << ','
            << v2.s << ',' << v2.t << ',' << v2.q << ','
            << v2.u << ',' << v2.v << ','
            << static_cast<uint32_t>(v2.fog) << ','
            << static_cast<uint32_t>(v2.r) << ','
            << static_cast<uint32_t>(v2.g) << ','
            << static_cast<uint32_t>(v2.b) << ','
            << static_cast<uint32_t>(v2.a) << ','
            << trace.candidates << ','
            << trace.scissorRejects << ','
            << trace.alphaRejects << ','
            << trace.destinationAlphaRejects << ','
            << trace.depthRejects << ','
            << trace.writes << ','
            << trace.framebufferChanges << ','
            << trace.depthWrites << ','
            << trace.depthChanges << ','
            << static_cast<uint32_t>(trace.minR) << ','
            << static_cast<uint32_t>(trace.maxR) << ','
            << static_cast<uint32_t>(trace.minG) << ','
            << static_cast<uint32_t>(trace.maxG) << ','
            << static_cast<uint32_t>(trace.minB) << ','
            << static_cast<uint32_t>(trace.maxB) << ','
            << static_cast<uint32_t>(trace.minA) << ','
            << static_cast<uint32_t>(trace.maxA) << ','
            << trace.textureSamples << ','
            << static_cast<uint32_t>(trace.minTextureIndex) << ','
            << static_cast<uint32_t>(trace.maxTextureIndex) << ','
            << static_cast<uint32_t>(trace.minTextureAlpha) << ','
            << static_cast<uint32_t>(trace.maxTextureAlpha) << ','
            << debugPageList(trace.framebufferPages) << ','
            << debugPageList(trace.depthPages) << ','
            << debugPageList(trace.texturePages) << ','
            << static_cast<uint32_t>(trace.feedback) << ',';
        if (trace.hashVram && gs->m_vram && gs->m_vramSize != 0u)
        {
            trace.output
                << std::hex << std::setw(16) << std::setfill('0')
                << debugFnv1a64(gs->m_vram, gs->m_vramSize)
                << std::dec;
        }
        trace.output
            << ',' << command.sequence()
            << ',' << command.stateSignature()
            << ',' << bounds.x0
            << ',' << bounds.y0
            << ',' << bounds.x1
            << ',' << bounds.y1
            << ',' << static_cast<uint32_t>(bounds.exact)
            << ',' << v0.x12_4
            << ',' << v0.y12_4
            << ',' << v0.zInteger
            << ',' << v1.x12_4
            << ',' << v1.y12_4
            << ',' << v1.zInteger
            << ',' << v2.x12_4
            << ',' << v2.y12_4
            << ',' << v2.zInteger
            << ',' << static_cast<uint32_t>(primitive.aa1)
            << ',' << static_cast<uint32_t>(primitive.fix)
            << ',' << static_cast<uint32_t>(global.pabe)
            << ',' << static_cast<uint32_t>(global.texa.ta0)
            << ',' << static_cast<uint32_t>(global.texa.aem)
            << ',' << static_cast<uint32_t>(global.texa.ta1)
            << ',' << static_cast<uint32_t>(global.scanMask)
            << ',' << global.dimx
            << ',' << static_cast<uint32_t>(global.dither)
            << ',' << static_cast<uint32_t>(global.colorClamp)
            << ',' << ctx.fba
            << ',' << debugPageList(resources.framebufferReadPages)
            << ',' << debugPageList(resources.framebufferWritePages)
            << ',' << debugPageList(resources.depthReadPages)
            << ',' << debugPageList(resources.depthWritePages)
            << ',' << debugPageList(resources.texturePages)
            << ',' << debugPageList(resources.mipPages)
            << ',' << debugPageList(resources.clutPages)
            << ',' << static_cast<uint32_t>(resources.readsDestination)
            << ',' << static_cast<uint32_t>(resources.framebufferDepthAlias)
            << ',' << static_cast<uint32_t>(resources.framebufferTextureAlias)
            << ',' << static_cast<uint32_t>(resources.framebufferClutAlias)
            << ',' << gsFallbackReasonName(decision.reason);
        trace.output << '\n';
        ++trace.written;
        if (trace.written == trace.limit)
        {
            // ps2EntryRunner performs an explicit std::_Exit after runtime
            // shutdown, so static stream destructors cannot commit the tail
            // of a bounded trace. The limit is the natural durable boundary.
            trace.output.flush();
        }
    }
    ++trace.index;
}

void GSRasterizer::writePixel(GS *gs,
                              int x,
                              int y,
                              int z,
                              uint8_t r,
                              uint8_t g,
                              uint8_t b,
                              uint8_t a,
                              bool forceAlphaBlend,
                              bool suppressDepthWrite)
{
    if (m_trackDebugProgress && m_debugProgressOwner)
    {
        ++m_debugCandidatePixelBatch;
        if (m_debugCandidatePixelBatch == 1024u)
        {
            m_debugProgressOwner->m_progressCandidatePixels.fetch_add(
                m_debugCandidatePixelBatch, std::memory_order_relaxed);
            m_debugCandidatePixelBatch = 0u;
        }
    }

    const auto &ctx = gs->activeContext();
    GSDrawTraceState &trace = drawTrace();
    if (trace.capturing)
    {
        ++trace.candidates;
        trace.minR = std::min(trace.minR, r);
        trace.maxR = std::max(trace.maxR, r);
        trace.minG = std::min(trace.minG, g);
        trace.maxG = std::max(trace.maxG, g);
        trace.minB = std::min(trace.minB, b);
        trace.maxB = std::max(trace.maxB, b);
        trace.minA = std::min(trace.minA, a);
        trace.maxA = std::max(trace.maxA, a);
    }

    if (x < ctx.scissor.x0 || x > ctx.scissor.x1 ||
        y < ctx.scissor.y0 || y > ctx.scissor.y1)
    {
        if (trace.capturing)
            ++trace.scissorRejects;
        return;
    }

    const AlphaTestResult alphaTest = classifyAlphaTest(ctx.test, a);

    if (!alphaTest.writeFramebuffer && !alphaTest.writeDepth)
    {
        if (trace.capturing)
            ++trace.alphaRejects;
        return;
    }

    u8* vram = gs->m_vram;

    const u32 fbp  = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    const u32 fbw  = std::max<u32>(ctx.frame.fbw, 1u);
    const u32 fpsm = ctx.frame.psm;
    const u32 fmsk = ctx.frame.fbmsk;
    const u32 zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
    const u32 zpsm = ctx.zbuf.psm;
    auto readFramebuffer = [&]() -> u32
    {
        if (fpsm == GS_PSM_CT32)
            return GSMem::ReadCT32(vram, fbp, fbw, x, y);
        return gs->ReadVram(fpsm, fbp, fbw, x, y);
    };
    auto writeFramebuffer = [&](u32 value)
    {
        if (fpsm == GS_PSM_CT32)
            GSMem::WriteCT32(vram, fbp, fbw, x, y, value);
        else
            gs->WriteVram(fpsm, fbp, fbw, x, y, value);
    };
    auto readDepth = [&]() -> u32
    {
        if (zpsm == GS_PSM_Z24)
            return GSMem::ReadZ24(vram, zbp, fbw, x, y);
        if (zpsm == GS_PSM_Z32)
            return GSMem::ReadZ32(vram, zbp, fbw, x, y);
        return gs->ReadVram(zpsm, zbp, fbw, x, y);
    };
    auto writeDepth = [&](u32 value)
    {
        if (zpsm == GS_PSM_Z24)
            GSMem::WriteZ24(vram, zbp, fbw, x, y, value);
        else if (zpsm == GS_PSM_Z32)
            GSMem::WriteZ32(vram, zbp, fbw, x, y, value);
        else
            gs->WriteVram(zpsm, zbp, fbw, x, y, value);
    };

    const bool alphaBlendEnabled =
        alphaTest.writeFramebuffer &&
        (gs->m_prim.abe || forceAlphaBlend);
    const bool destinationAlpha  = alphaTest.preserveDestinationAlpha;
    const bool destinationAlphaTest =
        destinationAlphaTestEnabled(ctx.test, static_cast<uint8_t>(fpsm));

    // small optimization, avoid reading the framebuffer for simple draws
    // TODO: only one address lookup for rmw
    const bool frmw = alphaTest.writeFramebuffer &&
                      ((ctx.frame.fbmsk != 0) || alphaBlendEnabled || destinationAlpha);

    u32 rawFramebufferValue = 0;
    u32 fbrgba = 0;
    if (frmw || destinationAlphaTest)
    {
        rawFramebufferValue = readFramebuffer();
        fbrgba = rawFramebufferValue;

        if (bitsPerPixel(fpsm) == 16)
        {
            fbrgba = Rgba5551ToRgba8888(fbrgba);
        }
    }

    if (!passesDestinationAlphaTest(ctx.test,
                                    static_cast<uint8_t>(fpsm),
                                    rawFramebufferValue))
    {
        if (trace.capturing)
            ++trace.destinationAlphaRejects;
        return;
    }

    const bool ztestEnabled = ((ctx.test >> 16) & 1u) != 0u;
    const uint ztestMethod = (ctx.test >> 17) & 3u;

    // When TEST.ZTE is clear the ZTST field is ignored and the depth
    // comparison always passes. The GS also suppresses depth writes in this
    // mode, even when ZBUF.ZMSK is clear.
    bool zpass = !ztestEnabled;
    switch (ztestEnabled ? ztestMethod : 1u)
    {
    case 0:
        zpass = false;
        break;
    case 1:
        zpass = true;
        break;
    case 2:
        zpass = z >= readDepth();
        break;
    case 3:
        zpass = z > readDepth();
        break;
    }

    if (!zpass)
    {
        if (trace.capturing)
            ++trace.depthRejects;
        return;
    }

    if (alphaTest.writeFramebuffer)
    {
        const u32 priorFramebufferValue =
            trace.capturing ? readFramebuffer() : 0u;
        const u8 srcR = r;
        const u8 srcG = g;
        const u8 srcB = b;

        if (alphaBlendEnabled)
        {
            uint8_t dr = fbrgba & 0xFF;
            uint8_t dg = (fbrgba >> 8) & 0xFF;
            uint8_t db = (fbrgba >> 16) & 0xFF;
            uint8_t da = (fbrgba >> 24) & 0xFF;

            // PABE disables alpha blending when the source alpha MSB is clear.
            if (!(gs->m_pabe && (a & 0x80u) == 0u))
            {
                uint64_t alphaReg = ctx.alpha;
                if ((alphaReg & 0xFFu) == 0x44u)
                {
                    const uint32_t blended = blendSourceOverRgb(
                        pack32(r, g, b, a),
                        fbrgba,
                        a);
                    r = static_cast<uint8_t>(blended);
                    g = static_cast<uint8_t>(blended >> 8u);
                    b = static_cast<uint8_t>(blended >> 16u);
                }
                else
                {
                uint8_t asel = alphaReg & 3;
                uint8_t bsel = (alphaReg >> 2) & 3;
                uint8_t csel = (alphaReg >> 4) & 3;
                uint8_t dsel = (alphaReg >> 6) & 3;
                uint8_t fix = static_cast<uint8_t>((alphaReg >> 32) & 0xFF);

                auto pickRGB = [&](uint8_t sel, int cs, int cd) -> int
                {
                    if (sel == 0)
                        return cs;
                    if (sel == 1)
                        return cd;
                    return 0;
                };
                int cAlpha = (csel == 0) ? a : (csel == 1) ? da
                                                           : fix;

                r = clampU8(((pickRGB(asel, r, dr) - pickRGB(bsel, r, dr)) * cAlpha >> 7) + pickRGB(dsel, r, dr));
                g = clampU8(((pickRGB(asel, g, dg) - pickRGB(bsel, g, dg)) * cAlpha >> 7) + pickRGB(dsel, g, dg));
                b = clampU8(((pickRGB(asel, b, db) - pickRGB(bsel, b, db)) * cAlpha >> 7) + pickRGB(dsel, b, db));
                }
            }
            else
            {
                r = srcR;
                g = srcG;
                b = srcB;
            }
        }

        u32 fbmask = ctx.frame.fbmsk;

        if (!alphaTest.preserveDestinationAlpha &&
            (ctx.fba & 0x1ull) != 0ull &&
            ctx.frame.psm != GS_PSM_CT24)
        {
            a = static_cast<uint8_t>(a | 0x80u);
        }

        u32 pixel = pack32(r, g, b, a);

        if (fbmask != 0)
        {
            pixel = (pixel & ~fbmask) | (fbrgba & fbmask);
        }

        if (alphaTest.preserveDestinationAlpha)
        {
            pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
        }

        // format conversion
        if (bitsPerPixel(fpsm) == 16)
        {
            pixel = Rgba8888ToRgba5551(pixel);
        }

        if (tracePixelMatches(x, y))
        {
            std::cerr
                << "[gs:write-pixel] draw=" << trace.index
                << " xy=(" << x << ',' << y << ')'
                << " z=" << z
                << " src=("
                << static_cast<unsigned>(srcR) << ','
                << static_cast<unsigned>(srcG) << ','
                << static_cast<unsigned>(srcB) << ','
                << static_cast<unsigned>(a) << ')'
                << " dst=0x" << std::hex << fbrgba
                << " alpha=0x" << ctx.alpha
                << " fbmask=0x" << fbmask
                << " pixel=0x" << pixel << std::dec
                << " blend=" << alphaBlendEnabled
                << '\n';
        }

        writeFramebuffer(pixel);
        if (trace.capturing)
        {
            trace.recordFramebufferPage(
                static_cast<uint8_t>(fpsm), fbp, fbw, x, y);
            ++trace.writes;
            if (readFramebuffer() != priorFramebufferValue)
                ++trace.framebufferChanges;
        }
    }

    if (alphaTest.writeDepth &&
        ztestEnabled &&
        !ctx.zbuf.zmask &&
        !suppressDepthWrite)
    {
        const u32 priorDepthValue =
            trace.capturing ? readDepth() : 0u;
        writeDepth(z);
        if (trace.capturing)
        {
            trace.recordDepthPage(
                static_cast<uint8_t>(zpsm), zbp, fbw, x, y);
            ++trace.depthWrites;
            if (readDepth() != priorDepthValue)
                ++trace.depthChanges;
        }
    }
}

uint32_t GSRasterizer::lookupCLUT(GS *gs,
                                  uint8_t index,
                                  uint32_t cbp,
                                  uint8_t cpsm,
                                  uint8_t csm,
                                  uint8_t csa,
                                  uint8_t sourcePsm)
{
    const uint32_t clutIndex = resolveClutIndex(index, csm, csa, sourcePsm);
    (void)cbp;
    (void)cpsm;

    if (!gs->m_clutCacheValid[clutIndex])
        return 0u;

    const uint32_t cached = gs->m_clutCache[clutIndex];
    const uint8_t cachedFormat = gs->m_clutCacheFormat[clutIndex];

    switch (cachedFormat)
    {
    case GS_PSM_CT32:
        return applyTexa(gs->m_texa, cachedFormat, cached);
    case GS_PSM_CT24:
        return applyTexa(gs->m_texa, cachedFormat, cached);
    case GS_PSM_CT16:
        return applyTexa(gs->m_texa, cachedFormat, Rgba5551ToRgba8888(cached));
    case GS_PSM_CT16S:
        return applyTexa(gs->m_texa, cachedFormat, Rgba5551ToRgba8888(cached));
    default:
        break;
    }

    return 0xFFFF00FFu;
}

void GSRasterizer::prepareDecodedClut(GS *gs)
{
    const GSTex0Reg &tex = gs->activeContext().tex0;
    const bool indexed =
        tex.psm == GS_PSM_T8 ||
        tex.psm == GS_PSM_T8H ||
        tex.psm == GS_PSM_T4 ||
        tex.psm == GS_PSM_T4HL ||
        tex.psm == GS_PSM_T4HH;
    m_decodedClutActive = indexed && gs->m_prim.tme;
    if (!m_decodedClutActive)
        return;

    const uint16_t texa =
        static_cast<uint16_t>(gs->m_texa.ta0) |
        (static_cast<uint16_t>(gs->m_texa.aem) << 8u) |
        (static_cast<uint16_t>(gs->m_texa.ta1) << 9u);
    if (m_decodedClutGeneration == gs->m_clutCacheGeneration &&
        m_decodedClutTexa == texa &&
        m_decodedClutSourcePsm == tex.psm &&
        m_decodedClutCsm == tex.csm &&
        m_decodedClutCsa == tex.csa)
    {
        return;
    }

    for (uint32_t index = 0u; index < m_decodedClut.size(); ++index)
    {
        m_decodedClut[index] = lookupCLUT(
            gs,
            static_cast<uint8_t>(index),
            tex.cbp,
            tex.cpsm,
            tex.csm,
            tex.csa,
            tex.psm);
    }
    m_decodedClutGeneration = gs->m_clutCacheGeneration;
    m_decodedClutTexa = texa;
    m_decodedClutSourcePsm = tex.psm;
    m_decodedClutCsm = tex.csm;
    m_decodedClutCsa = tex.csa;
}

void GSRasterizer::updateClutCache(GS *gs, int contextIndex)
{
    if (!gs || !gs->m_vram)
        return;

    const GSTex0Reg &tex = gs->m_ctx[(contextIndex != 0) ? 1 : 0].tex0;
    const bool indexed =
        tex.psm == GS_PSM_T8 ||
        tex.psm == GS_PSM_T8H ||
        tex.psm == GS_PSM_T4 ||
        tex.psm == GS_PSM_T4HL ||
        tex.psm == GS_PSM_T4HH;
    if (!indexed)
        return;

    bool load = false;
    switch (tex.cld)
    {
    case 1:
        load = true;
        break;
    case 2:
        gs->m_clutCbp[0] = tex.cbp;
        load = true;
        break;
    case 3:
        gs->m_clutCbp[1] = tex.cbp;
        load = true;
        break;
    case 4:
        if (gs->m_clutCbp[0] != tex.cbp)
        {
            gs->m_clutCbp[0] = tex.cbp;
            load = true;
        }
        break;
    case 5:
        if (gs->m_clutCbp[1] != tex.cbp)
        {
            gs->m_clutCbp[1] = tex.cbp;
            load = true;
        }
        break;
    case 0:
    case 6:
    case 7:
    default:
        break;
    }

    if (!load)
        return;

    const bool fourBit =
        tex.psm == GS_PSM_T4 ||
        tex.psm == GS_PSM_T4HL ||
        tex.psm == GS_PSM_T4HH;
    const uint32_t entryCount = fourBit ? 16u : 256u;
    const bool csm2 = tex.csm != 0u;
    const uint32_t clutWidth =
        csm2 && gs->m_texclut.cbw != 0u
            ? static_cast<uint32_t>(gs->m_texclut.cbw)
            : 1u;

    const uint32_t maximumClutX =
        csm2
            ? (static_cast<uint32_t>(gs->m_texclut.cou) << 4u) +
                  15u
            : 15u;
    const uint32_t maximumClutY =
        csm2
            ? static_cast<uint32_t>(gs->m_texclut.cov) +
                  ((entryCount - 1u) >> 4u)
            : 15u;
    const GsVramPageMask clutPages =
        gsVramPagesForSurfaceRect(
            tex.cbp,
            clutWidth,
            tex.cpsm,
            0u,
            0u,
            maximumClutX + 1u,
            maximumClutY + 1u);
    beginCpuVramAccess(
        gs, clutPages, {}, GsFlushReason::ClutHazard);

    GSClutTraceState &trace = clutTrace();
    uint32_t rawNonzeroBytes = 0u;
    uint8_t rawMaxByte = 0u;
    const bool traceEnabled = trace.enabled();
    if (traceEnabled)
    {
        constexpr uint32_t kRawClutWindowBytes = 1024u;
        const uint32_t rawBase =
            (tex.cbp * 256u) % std::max<uint32_t>(gs->m_vramSize, 1u);
        for (uint32_t offset = 0u; offset < kRawClutWindowBytes; ++offset)
        {
            const uint32_t address = (rawBase + offset) % gs->m_vramSize;
            const uint8_t value = gs->m_vram[address];
            rawNonzeroBytes += value != 0u ? 1u : 0u;
            rawMaxByte = std::max(rawMaxByte, value);
        }
    }

    for (uint32_t rawIndex = 0u; rawIndex < entryCount; ++rawIndex)
    {
        const uint32_t cacheIndex =
            resolveClutIndex(static_cast<uint8_t>(rawIndex),
                             tex.csm,
                             tex.csa,
                             tex.psm);
        const uint32_t sourceIndex = csm2 ? rawIndex : cacheIndex;
        const uint32_t clutX =
            (csm2 ? (static_cast<uint32_t>(gs->m_texclut.cou) << 4u) : 0u) +
            (sourceIndex & 0x0Fu);
        const uint32_t clutY =
            (csm2 ? static_cast<uint32_t>(gs->m_texclut.cov) : 0u) +
            (sourceIndex >> 4u);

        uint32_t value = 0u;
        switch (tex.cpsm)
        {
        case GS_PSM_CT32:
            value = GSMem::ReadCT32(gs->m_vram, tex.cbp, clutWidth, clutX, clutY);
            break;
        case GS_PSM_CT24:
            value = GSMem::ReadCT24(gs->m_vram, tex.cbp, clutWidth, clutX, clutY);
            break;
        case GS_PSM_CT16:
            value = GSMem::ReadCT16(gs->m_vram, tex.cbp, clutWidth, clutX, clutY);
            break;
        case GS_PSM_CT16S:
            value = GSMem::ReadCT16S(gs->m_vram, tex.cbp, clutWidth, clutX, clutY);
            break;
        default:
            continue;
        }

        gs->m_clutCache[cacheIndex] = value;
        gs->m_clutCacheFormat[cacheIndex] = tex.cpsm;
        gs->m_clutCacheValid[cacheIndex] = true;
    }
    ++gs->m_clutCacheGeneration;

    if (traceEnabled)
    {
        uint32_t decodedNonzeroEntries = 0u;
        uint32_t decodedAlphaNonzeroEntries = 0u;
        uint8_t decodedMaxAlpha = 0u;
        for (uint32_t rawIndex = 0u; rawIndex < entryCount; ++rawIndex)
        {
            const uint32_t cacheIndex =
                resolveClutIndex(static_cast<uint8_t>(rawIndex),
                                 tex.csm,
                                 tex.csa,
                                 tex.psm);
            const uint32_t value = gs->m_clutCache[cacheIndex];
            decodedNonzeroEntries += value != 0u ? 1u : 0u;
            const uint8_t alpha = static_cast<uint8_t>(value >> 24u);
            decodedAlphaNonzeroEntries += alpha != 0u ? 1u : 0u;
            decodedMaxAlpha = std::max(decodedMaxAlpha, alpha);
        }

        trace.output << trace.index++ << ','
                     << ((contextIndex != 0) ? 1 : 0) << ','
                     << tex.tbp0 << ','
                     << static_cast<uint32_t>(tex.psm) << ','
                     << tex.cbp << ','
                     << static_cast<uint32_t>(tex.cpsm) << ','
                     << static_cast<uint32_t>(tex.csm) << ','
                     << static_cast<uint32_t>(tex.csa) << ','
                     << static_cast<uint32_t>(tex.cld) << ','
                     << rawNonzeroBytes << ','
                     << static_cast<uint32_t>(rawMaxByte) << ','
                     << decodedNonzeroEntries << ','
                     << decodedAlphaNonzeroEntries << ','
                     << static_cast<uint32_t>(decodedMaxAlpha) << '\n';
        trace.output.flush();
    }
}

uint32_t GSRasterizer::sampleTexture(GS *gs, float s, float t, float q, uint16_t u, uint16_t v)
{
    const auto &ctx = gs->activeContext();
    const auto &tex = ctx.tex0;

    int32_t fixedU;
    int32_t fixedV;
    if (gs->m_prim.fst)
    {
        fixedU = static_cast<int32_t>(u) << 12;
        fixedV = static_cast<int32_t>(v) << 12;
    }
    else
    {
        const float invQ = 1.0f / fabsQ(q);
        const float textureScaleU =
            static_cast<float>(65536u << tex.tw);
        const float textureScaleV =
            static_cast<float>(65536u << tex.th);
        fixedU = static_cast<int32_t>(s * invQ * textureScaleU);
        fixedV = static_cast<int32_t>(t * invQ * textureScaleV);
    }

    return sampleTextureFixed(gs, fixedU, fixedV, false, q);
}

uint32_t GSRasterizer::sampleTextureLinearLevel(
    GS *gs,
    int32_t fixedU,
    int32_t fixedV,
    bool linearBiasApplied,
    uint8_t level)
{
    const GSContext &ctx = gs->activeContext();
    const GSTex0Reg &tex = ctx.tex0;
    if (level != 0u)
    {
        const int32_t divisor = 1 << level;
        int32_t quotientU = fixedU / divisor;
        int32_t quotientV = fixedV / divisor;
        if (fixedU < 0 && (fixedU % divisor) != 0)
            --quotientU;
        if (fixedV < 0 && (fixedV % divisor) != 0)
            --quotientV;
        fixedU = quotientU;
        fixedV = quotientV;
    }
    if (!linearBiasApplied)
    {
        fixedU = static_cast<int32_t>(
            static_cast<uint32_t>(fixedU) - 0x8000u);
        fixedV = static_cast<int32_t>(
            static_cast<uint32_t>(fixedV) - 0x8000u);
    }

    const uint8_t weightU = static_cast<uint8_t>(
        (static_cast<uint32_t>(fixedU) >> 12u) & 0xFu);
    const uint8_t weightV = static_cast<uint8_t>(
        (static_cast<uint32_t>(fixedV) >> 12u) & 0xFu);
    const uint8_t sampledTapMask =
        GSRasterizerDetail::requiredBilinearTapMask(weightU, weightV);

    const int u0 = floorFixed16_16(fixedU);
    const int v0 = floorFixed16_16(fixedV);
    const uint64_t clamp = ctx.clamp;
    const int texW = std::max(1, (1 << tex.tw) >> level);
    const int texH = std::max(1, (1 << tex.th) >> level);
    auto wrapU = [&](int coordinate)
    {
        return wrapTextureCoordinate(
            coordinate,
            texW,
            static_cast<uint8_t>(clamp & 0x3u),
            static_cast<uint16_t>(
                ((clamp >> 4u) & 0x3FFu) >> level),
            static_cast<uint16_t>(
                ((clamp >> 14u) & 0x3FFu) >> level));
    };
    auto wrapV = [&](int coordinate)
    {
        return wrapTextureCoordinate(
            coordinate,
            texH,
            static_cast<uint8_t>((clamp >> 2u) & 0x3u),
            static_cast<uint16_t>(
                ((clamp >> 24u) & 0x3FFu) >> level),
            static_cast<uint16_t>(
                ((clamp >> 34u) & 0x3FFu) >> level));
    };

    int wrappedU[2] = {wrapU(u0), 0};
    int wrappedV[2] = {wrapV(v0), 0};
    wrappedU[1] = weightU != 0u ? wrapU(u0 + 1) : wrappedU[0];
    wrappedV[1] = weightV != 0u ? wrapV(v0 + 1) : wrappedV[0];
    uint32_t tbp = tex.tbp0;
    uint8_t tbw = tex.tbw;
    if (level != 0u)
    {
        const uint64_t mipRegister =
            level <= 3u ? ctx.miptbp1 : ctx.miptbp2;
        const uint8_t slot =
            static_cast<uint8_t>((level - 1u) % 3u);
        const uint8_t shift = static_cast<uint8_t>(slot * 20u);
        tbp = static_cast<uint32_t>(
            (mipRegister >> shift) & 0x3FFFu);
        tbw = static_cast<uint8_t>(
            (mipRegister >> (shift + 14u)) & 0x3Fu);
    }
    const uint8_t *textureVram =
        m_textureReadVram ? m_textureReadVram : gs->m_vram;
    std::array<GSRasterizerDetail::LinearTextureTap, 4> taps;
    if (tex.psm == GS_PSM_T8)
    {
        taps = GSRasterizerDetail::readRequiredBilinearTaps(
            weightU,
            weightV,
            [&](int uIndex, int vIndex)
            {
                const uint32_t raw = GSMem::ReadP8(
                    const_cast<uint8_t *>(textureVram),
                    tbp, tbw, wrappedU[uIndex], wrappedV[vIndex]);
                const uint32_t color =
                    m_decodedClutActive
                        ? m_decodedClut[static_cast<uint8_t>(raw)]
                        : lookupCLUT(
                              gs,
                              static_cast<uint8_t>(raw),
                              tex.cbp,
                              tex.cpsm,
                              tex.csm,
                              tex.csa,
                              tex.psm);
                return GSRasterizerDetail::LinearTextureTap{raw, color};
            });
    }
    else
    {
        taps = GSRasterizerDetail::readRequiredBilinearTaps(
            weightU,
            weightV,
            [&](int uIndex, int vIndex)
            {
                const uint32_t color = GSMem::ReadCT32(
                    const_cast<uint8_t *>(textureVram),
                    tbp, tbw, wrappedU[uIndex], wrappedV[vIndex]);
                return GSRasterizerDetail::LinearTextureTap{color, color};
            });
    }

    GSDrawTraceState &trace = drawTrace();
    if (trace.capturing)
    {
        // textureSamples describes physical reads, so replicated taps do not
        // contribute to the trace count, page set, or extrema.
        for (int sample = 0; sample < 4; ++sample)
        {
            if ((sampledTapMask & (1u << sample)) == 0u)
                continue;
            trace.recordTexturePage(
                tex.psm,
                tbp,
                tbw,
                static_cast<uint32_t>(
                    std::max(wrappedU[sample & 1], 0)),
                static_cast<uint32_t>(
                    std::max(wrappedV[sample >> 1], 0)));
            ++trace.textureSamples;
            const uint8_t textureIndex =
                static_cast<uint8_t>(taps[sample].raw);
            trace.minTextureIndex =
                std::min(trace.minTextureIndex, textureIndex);
            trace.maxTextureIndex =
                std::max(trace.maxTextureIndex, textureIndex);
            const uint8_t alpha =
                static_cast<uint8_t>(taps[sample].color >> 24u);
            trace.minTextureAlpha =
                std::min(trace.minTextureAlpha, alpha);
            trace.maxTextureAlpha =
                std::max(trace.maxTextureAlpha, alpha);
        }
    }

    if (weightU == 0u)
    {
        if (weightV == 0u)
            return taps[0].color;
        return linearColor4(taps[0].color, taps[2].color, weightV);
    }
    if (weightV == 0u)
        return linearColor4(taps[0].color, taps[1].color, weightU);
    return bilinearColor4(
        taps[0].color,
        taps[1].color,
        taps[2].color,
        taps[3].color,
        weightU,
        weightV);
}

uint32_t GSRasterizer::sampleTextureFixed(GS *gs,
                                          int32_t fixedU,
                                          int32_t fixedV,
                                          bool linearBiasApplied,
                                          float lodQ)
{
    const auto &ctx = gs->activeContext();
    const auto &tex = ctx.tex0;
    const uint8_t maximumLevel =
        static_cast<uint8_t>((ctx.tex1 >> 2u) & 0x7u);
    const uint8_t magnificationFilter =
        static_cast<uint8_t>((ctx.tex1 >> 5u) & 0x1u);
    const uint8_t minificationFilter =
        static_cast<uint8_t>((ctx.tex1 >> 6u) & 0x7u);
    const bool mipmapsEnabled =
        maximumLevel != 0u &&
        minificationFilter >= 2u &&
        minificationFilter <= 5u;
    const bool linearFilter = tex1UsesLinearFilter(ctx.tex1);
    const bool supportsLinearFastPath =
        tex.psm == GS_PSM_T8 ||
        tex.psm == GS_PSM_CT32;
    if (!mipmapsEnabled &&
        linearFilter &&
        supportsLinearFastPath)
    {
        return sampleTextureLinearLevel(
            gs, fixedU, fixedV, linearBiasApplied, 0u);
    }

    auto shiftFixedForLevel = [](int32_t value,
                                 uint8_t level) -> int32_t
    {
        if (level == 0u)
            return value;
        const int32_t divisor = 1 << level;
        int32_t quotient = value / divisor;
        if (value < 0 && (value % divisor) != 0)
            --quotient;
        return quotient;
    };

    auto mipAddress = [&](uint8_t level,
                          uint32_t &tbp,
                          uint8_t &tbw)
    {
        if (level == 0u)
        {
            tbp = tex.tbp0;
            tbw = tex.tbw;
            return;
        }

        const uint64_t mipRegister =
            level <= 3u ? ctx.miptbp1 : ctx.miptbp2;
        const uint8_t slot =
            static_cast<uint8_t>((level - 1u) % 3u);
        const uint8_t shift = static_cast<uint8_t>(slot * 20u);
        tbp = static_cast<uint32_t>(
            (mipRegister >> shift) & 0x3FFFu);
        tbw = static_cast<uint8_t>(
            (mipRegister >> (shift + 14u)) & 0x3Fu);
    };

    auto sampleLevel = [&](uint8_t level,
                           bool linearFilter,
                           bool biasAlreadyApplied) -> uint32_t
    {
        const int texW =
            std::max(1, (1 << tex.tw) >> level);
        const int texH =
            std::max(1, (1 << tex.th) >> level);
        int32_t levelU = shiftFixedForLevel(fixedU, level);
        int32_t levelV = shiftFixedForLevel(fixedV, level);

        if (linearFilter && !biasAlreadyApplied)
        {
            levelU = static_cast<int32_t>(
                static_cast<uint32_t>(levelU) - 0x8000u);
            levelV = static_cast<int32_t>(
                static_cast<uint32_t>(levelV) - 0x8000u);
        }

        uint32_t tbp = 0u;
        uint8_t tbw = 0u;
        mipAddress(level, tbp, tbw);

        auto samplePoint = [&](int sampleU,
                               int sampleV) -> uint32_t
        {
            const uint64_t clamp = ctx.clamp;
            sampleU = wrapTextureCoordinate(
                sampleU,
                texW,
                static_cast<uint8_t>(clamp & 0x3u),
                static_cast<uint16_t>(
                    ((clamp >> 4u) & 0x3FFu) >> level),
                static_cast<uint16_t>(
                    ((clamp >> 14u) & 0x3FFu) >> level));
            sampleV = wrapTextureCoordinate(
                sampleV,
                texH,
                static_cast<uint8_t>((clamp >> 2u) & 0x3u),
                static_cast<uint16_t>(
                    ((clamp >> 24u) & 0x3FFu) >> level),
                static_cast<uint16_t>(
                    ((clamp >> 34u) & 0x3FFu) >> level));

            const u8 *textureVram =
                m_textureReadVram
                    ? m_textureReadVram
                    : gs->m_vram;
            const u32 out = gs->ReadVramFrom(
                textureVram,
                tex.psm,
                tbp,
                tbw,
                sampleU,
                sampleV);
            const uint8_t textureIndex =
                static_cast<uint8_t>(out);
            auto recordSample =
                [&](uint32_t color) -> uint32_t
            {
                GSDrawTraceState &trace = drawTrace();
                if (trace.capturing)
                {
                    trace.recordTexturePage(
                        tex.psm,
                        tbp,
                        tbw,
                        static_cast<uint32_t>(std::max(sampleU, 0)),
                        static_cast<uint32_t>(std::max(sampleV, 0)));
                    ++trace.textureSamples;
                    trace.minTextureIndex =
                        std::min(trace.minTextureIndex,
                                 textureIndex);
                    trace.maxTextureIndex =
                        std::max(trace.maxTextureIndex,
                                 textureIndex);
                    const uint8_t alpha =
                        static_cast<uint8_t>(color >> 24u);
                    trace.minTextureAlpha =
                        std::min(trace.minTextureAlpha, alpha);
                    trace.maxTextureAlpha =
                        std::max(trace.maxTextureAlpha, alpha);
                }
                return color;
            };

            switch (tex.psm)
            {
            case GS_PSM_CT32:
            case GS_PSM_Z32:
            case GS_PSM_CT24:
            case GS_PSM_Z24:
                return recordSample(
                    applyTexa(gs->m_texa, tex.psm, out));
            case GS_PSM_CT16:
            case GS_PSM_CT16S:
            case GS_PSM_Z16:
            case GS_PSM_Z16S:
                return recordSample(applyTexa(
                    gs->m_texa,
                    tex.psm,
                    Rgba5551ToRgba8888(out)));
            case GS_PSM_T8:
            case GS_PSM_T8H:
            case GS_PSM_T4:
            case GS_PSM_T4HL:
            case GS_PSM_T4HH:
                return recordSample(
                    m_decodedClutActive
                        ? m_decodedClut[static_cast<uint8_t>(out)]
                        : lookupCLUT(
                              gs,
                              static_cast<u8>(out),
                              tex.cbp,
                              tex.cpsm,
                              tex.csm,
                              tex.csa,
                              tex.psm));
            }

            return recordSample(0xFFFF00FFu);
        };

        const int u0 = floorFixed16_16(levelU);
        const int v0 = floorFixed16_16(levelV);
        if (!linearFilter)
            return samplePoint(u0, v0);

        const int u1 = u0 + 1;
        const int v1 = v0 + 1;
        const uint8_t weightU = static_cast<uint8_t>(
            (static_cast<uint32_t>(levelU) >> 12u) & 0xFu);
        const uint8_t weightV = static_cast<uint8_t>(
            (static_cast<uint32_t>(levelV) >> 12u) & 0xFu);

        const uint32_t c00 = samplePoint(u0, v0);
        const uint32_t c10 = samplePoint(u1, v0);
        const uint32_t c01 = samplePoint(u0, v1);
        const uint32_t c11 = samplePoint(u1, v1);

        return bilinearColor4(
            c00, c10, c01, c11, weightU, weightV);
    };

    if (!mipmapsEnabled)
    {
        return sampleLevel(0u,
                           linearFilter,
                           linearBiasApplied);
    }

    const int32_t rawK =
        static_cast<int32_t>((ctx.tex1 >> 32u) & 0xFFFu);
    const int32_t signedK =
        (rawK & 0x800) != 0 ? rawK - 0x1000 : rawK;
    int32_t lodFixed;
    if ((ctx.tex1 & 0x1u) == 0u)
    {
        const uint8_t l =
            static_cast<uint8_t>((ctx.tex1 >> 19u) & 0x3u);
        const float absoluteQ =
            std::max(std::fabs(lodQ),
                     std::numeric_limits<float>::min());
        const float lodScale =
            -static_cast<float>((1u << l) * 65536u);
        const float lodBias =
            static_cast<float>(signedK * 4096);
        const float lodValue =
            approximateLog2Precision3(absoluteQ) * lodScale +
            lodBias;
        const float clampedLod = std::clamp(
            lodValue,
            0.0f,
            static_cast<float>(maximumLevel) * 65536.0f);
        lodFixed = static_cast<int32_t>(
            std::nearbyint(clampedLod));
    }
    else
    {
        lodFixed = std::clamp(
            signedK * 4096,
            0,
            static_cast<int32_t>(maximumLevel) * 65536);
    }

    if (lodFixed <= 0)
    {
        if (magnificationFilter != 0u &&
            supportsLinearFastPath)
        {
            return sampleTextureLinearLevel(
                gs, fixedU, fixedV, false, 0u);
        }
        return sampleLevel(0u,
                           magnificationFilter != 0u,
                           false);
    }

    const bool linearWithinLevel =
        minificationFilter >= 4u;
    const bool interpolateLevels =
        (minificationFilter & 0x1u) != 0u;
    if (!interpolateLevels)
    {
        const int32_t roundedLod =
            lodFixed + 0x8000;
        const uint8_t level = static_cast<uint8_t>(
            std::min<int>(
                static_cast<int>(maximumLevel),
                roundedLod >> 16));
        if (linearWithinLevel &&
            supportsLinearFastPath)
        {
            return sampleTextureLinearLevel(
                gs, fixedU, fixedV, false, level);
        }
        return sampleLevel(level,
                           linearWithinLevel,
                           false);
    }

    const uint8_t level0 = static_cast<uint8_t>(
        std::min<int>(
            static_cast<int>(maximumLevel),
            lodFixed >> 16));
    const uint8_t level1 = std::min<uint8_t>(
        static_cast<uint8_t>(level0 + 1u),
        maximumLevel);
    const uint32_t color0 =
        linearWithinLevel && supportsLinearFastPath
            ? sampleTextureLinearLevel(
                  gs, fixedU, fixedV, false, level0)
            : sampleLevel(level0, linearWithinLevel, false);
    if (level0 == level1)
        return color0;
    const uint32_t color1 =
        linearWithinLevel && supportsLinearFastPath
            ? sampleTextureLinearLevel(
                  gs, fixedU, fixedV, false, level1)
            : sampleLevel(level1, linearWithinLevel, false);
    const int weight =
        (lodFixed & 0xFFFF) >> 1;
    auto interpolateChannel = [&](uint8_t from,
                                  uint8_t to) -> uint8_t
    {
        const int difference =
            (static_cast<int>(to) -
             static_cast<int>(from)) *
            weight;
        const int delta =
            difference >= 0
                ? difference / 32768
                : -((-difference + 32767) / 32768);
        return clampU8(static_cast<int>(from) + delta);
    };

    return static_cast<uint32_t>(interpolateChannel(
               static_cast<uint8_t>(color0),
               static_cast<uint8_t>(color1))) |
           (static_cast<uint32_t>(interpolateChannel(
                static_cast<uint8_t>(color0 >> 8u),
                static_cast<uint8_t>(color1 >> 8u)))
            << 8u) |
           (static_cast<uint32_t>(interpolateChannel(
                static_cast<uint8_t>(color0 >> 16u),
                static_cast<uint8_t>(color1 >> 16u)))
            << 16u) |
           (static_cast<uint32_t>(interpolateChannel(
                static_cast<uint8_t>(color0 >> 24u),
                static_cast<uint8_t>(color1 >> 24u)))
            << 24u);
}

void GSRasterizer::drawSprite(GS *gs)
{
    const GSVertex &v0 = gs->m_vtxQueue[0];
    const GSVertex &v1 = gs->m_vtxQueue[1];
    const auto &ctx = gs->activeContext();

    int32_t fixedX0 =
        m_queuedFixedVerticesValid
            ? m_queuedFixedX[0]
            : static_cast<int32_t>(std::lround(v0.x * 16.0f)) -
                  static_cast<int32_t>(ctx.xyoffset.ofx);
    int32_t fixedY0 =
        m_queuedFixedVerticesValid
            ? m_queuedFixedY[0]
            : static_cast<int32_t>(std::lround(v0.y * 16.0f)) -
                  static_cast<int32_t>(ctx.xyoffset.ofy);
    int32_t fixedX1 =
        m_queuedFixedVerticesValid
            ? m_queuedFixedX[1]
            : static_cast<int32_t>(std::lround(v1.x * 16.0f)) -
                  static_cast<int32_t>(ctx.xyoffset.ofx);
    int32_t fixedY1 =
        m_queuedFixedVerticesValid
            ? m_queuedFixedY[1]
            : static_cast<int32_t>(std::lround(v1.y * 16.0f)) -
                  static_cast<int32_t>(ctx.xyoffset.ofy);
    u32 z1 = static_cast<u32>(v1.z);

    float u0f, v0f, u1f, v1f;
    const auto &tex = ctx.tex0;
    const int texW = 1 << tex.tw;
    const int texH = 1 << tex.th;
    if (gs->m_prim.fst)
    {
        u0f = static_cast<float>(v0.u) / 16.0f;
        v0f = static_cast<float>(v0.v) / 16.0f;
        u1f = static_cast<float>(v1.u) / 16.0f;
        v1f = static_cast<float>(v1.v) / 16.0f;
    }
    else
    {
        const float q0 = fabsQ(v0.q);
        const float q1 = fabsQ(v1.q);
        u0f = (v0.s / q0) * static_cast<float>(texW);
        v0f = (v0.t / q0) * static_cast<float>(texH);
        u1f = (v1.s / q1) * static_cast<float>(texW);
        v1f = (v1.t / q1) * static_cast<float>(texH);
    }

    if (fixedX0 > fixedX1)
    {
        std::swap(fixedX0, fixedX1);
        std::swap(u0f, u1f);
    }
    if (fixedY0 > fixedY1)
    {
        std::swap(fixedY0, fixedY1);
        std::swap(v0f, v1f);
    }

    const int unclippedX0 = ceilFixed12_4(fixedX0);
    const int unclippedY0 = ceilFixed12_4(fixedY0);
    const int unclippedX1Exclusive = ceilFixed12_4(fixedX1);
    const int unclippedY1Exclusive = ceilFixed12_4(fixedY1);
    if (unclippedX0 >= unclippedX1Exclusive ||
        unclippedY0 >= unclippedY1Exclusive)
    {
        return;
    }

    // If the sprite rectangle is fully outside scissor, nothing should render.
    if (unclippedX1Exclusive <= ctx.scissor.x0 ||
        unclippedX0 > ctx.scissor.x1 ||
        unclippedY1Exclusive <= ctx.scissor.y0 ||
        unclippedY0 > ctx.scissor.y1)
    {
        // maybe a log here idk ?
        return;
    }

    const int drawX0 = clampInt(unclippedX0, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY0 = clampInt(unclippedY0, ctx.scissor.y0, ctx.scissor.y1);
    const int drawX1 = clampInt(unclippedX1Exclusive - 1,
                                ctx.scissor.x0, ctx.scissor.x1);
    const int drawY1 = clampInt(unclippedY1Exclusive - 1,
                                ctx.scissor.y0, ctx.scissor.y1);

    const bool useFastDirectSprite =
        !drawTrace().capturing &&
        ctx.frame.psm == GS_PSM_CT32 &&
        ctx.frame.fbmsk == 0u &&
        !gs->m_prim.abe &&
        (ctx.test & 1u) == 0u &&
        !destinationAlphaTestEnabled(
            ctx.test, GS_PSM_CT32) &&
        ((ctx.test >> 16u) & 1u) != 0u &&
        ((ctx.test >> 17u) & 3u) == 1u &&
        (ctx.zbuf.psm == GS_PSM_Z24 ||
         ctx.zbuf.psm == GS_PSM_Z32);
    const uint32_t fastSpriteFbp =
        GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    const uint32_t fastSpriteZbp =
        GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
    const uint32_t fastSpriteFbw =
        std::max<uint32_t>(ctx.frame.fbw, 1u);
    auto writeSpritePixel =
        [&](int x,
            int y,
            uint8_t pixelR,
            uint8_t pixelG,
            uint8_t pixelB,
            uint8_t pixelA)
    {
        if (!useFastDirectSprite ||
            tracePixelMatches(x, y))
        {
            writePixel(
                gs,
                x,
                y,
                z1,
                pixelR,
                pixelG,
                pixelB,
                pixelA);
            return;
        }

        if ((ctx.fba & 1u) != 0u)
            pixelA = static_cast<uint8_t>(pixelA | 0x80u);
        GSMem::WriteCT32(
            gs->m_vram,
            fastSpriteFbp,
            fastSpriteFbw,
            x,
            y,
            pack32(pixelR,
                   pixelG,
                   pixelB,
                   pixelA));
        if (!ctx.zbuf.zmask)
        {
            if (ctx.zbuf.psm == GS_PSM_Z24)
            {
                GSMem::WriteZ24(
                    gs->m_vram,
                    fastSpriteZbp,
                    fastSpriteFbw,
                    x,
                    y,
                    z1);
            }
            else
            {
                GSMem::WriteZ32(
                    gs->m_vram,
                    fastSpriteZbp,
                    fastSpriteFbw,
                    x,
                    y,
                    z1);
            }
        }
    };

    const uint64_t alphaReg = ctx.alpha;
    const uint8_t alphaMode = static_cast<uint8_t>(alphaReg & 0xFFu);
    const uint8_t alphaFix = static_cast<uint8_t>((alphaReg >> 32) & 0xFFu);
    const bool looksLikeDisplayCopy =
        gs->m_prim.tme &&
        gs->m_prim.abe &&
        gs->m_prim.fst &&
        gs->m_prim.ctxt &&
        ctx.frame.fbp != ctx.tex0.tbp0 &&
        alphaMode == 0x64u &&
        (alphaFix == 0x60u || alphaFix == 0x80u) &&
        unclippedX0 <= 0 &&
        unclippedY0 <= 0 &&
        unclippedX1Exclusive > 639 &&
        unclippedY1Exclusive > 447;
    if (looksLikeDisplayCopy)
    {
        gs->m_preferredDisplaySourceFrame = {ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, 0u};
        gs->m_preferredDisplayDestFbp = ctx.frame.fbp;
        gs->m_hasPreferredDisplaySource = true;
    }

    uint8_t r = v1.r, g = v1.g, b = v1.b, a = v1.a;
    const uint8_t fog = v1.fog;

    if (gs->m_prim.tme)
    {
        const float windowX0 = static_cast<float>(fixedX0) / 16.0f;
        const float windowY0 = static_cast<float>(fixedY0) / 16.0f;
        const float spriteW =
            static_cast<float>(fixedX1 - fixedX0) / 16.0f;
        const float spriteH =
            static_cast<float>(fixedY1 - fixedY0) / 16.0f;
        const bool fixedTextureDda = gs->m_prim.fst;
        const bool linearFilter =
            fixedTextureDda && tex1UsesLinearFilter(ctx.tex1);
        const float linearBias = linearFilter ? 32768.0f : 0.0f;
        const float fixedU0 =
            u0f * 65536.0f - linearBias;
        const float fixedV0 =
            v0f * 65536.0f - linearBias;
        const float fixedU1 =
            u1f * 65536.0f - linearBias;
        const float fixedV1 =
            v1f * 65536.0f - linearBias;
        const float fixedStepU =
            (fixedU1 - fixedU0) / spriteW;
        const float fixedStepV =
            (fixedV1 - fixedV0) / spriteH;
        const float fixedPrestepX =
            static_cast<float>(drawX0) - windowX0;
        const float fixedPrestepY =
            static_cast<float>(drawY0) - windowY0;
        const float fixedScanU =
            fixedU0 + fixedStepU * fixedPrestepX;
        float fixedScanV =
            fixedV0 + fixedStepV * fixedPrestepY;
        constexpr int kPixelsPerLaneGroup = 8;
        const int alignedDrawX =
            drawX0 & ~(kPixelsPerLaneGroup - 1);
        const int laneSkip =
            drawX0 & (kPixelsPerLaneGroup - 1);
        const int32_t fixedBlockStepU =
            static_cast<int32_t>(
                fixedStepU *
                static_cast<float>(kPixelsPerLaneGroup));

        const auto packedOverride =
            GSRasterizerDetail::packedSpriteKernelOverride();
        const bool depthTestEnabled =
            ((ctx.test >> 16u) & 1u) != 0u;
        const uint8_t depthTestMethod =
            static_cast<uint8_t>((ctx.test >> 17u) & 0x3u);
        const bool depthAlwaysPassesWithoutWrite =
            !depthTestEnabled ||
            (depthTestMethod == 1u && ctx.zbuf.zmask);
        const GSPixelTraceTarget &pixelTrace =
            pixelTraceTarget();
        const bool usePackedSprite =
            packedOverride !=
                GSRasterizerDetail::PackedSpriteKernelOverride::
                    ForceReference &&
            fixedTextureDda &&
            linearFilter &&
            ((ctx.tex1 >> 2u) & 0x7u) == 0u &&
            tex.psm == GS_PSM_CT32 &&
            tex.tcc == 1u &&
            tex.tfx == 1u &&
            (ctx.clamp & 0xFu) == 0u &&
            ctx.frame.psm == GS_PSM_CT32 &&
            ctx.frame.fbmsk == 0u &&
            (ctx.fba & 1u) == 0u &&
            !gs->m_prim.fge &&
            !gs->m_prim.abe &&
            (ctx.test & 1u) == 0u &&
            !destinationAlphaTestEnabled(
                ctx.test, GS_PSM_CT32) &&
            depthAlwaysPassesWithoutWrite &&
            !drawTrace().capturing &&
            !pixelTrace.enabled;
        if (usePackedSprite)
        {
            const PackedSpriteKernelSelection kernelSelection =
                selectPackedSpriteKernel(packedOverride);
            PreparedPackedSprite packedSprite{
                .textureVram =
                    m_textureReadVram
                        ? m_textureReadVram
                        : gs->m_vram,
                .framebufferVram = gs->m_vram,
                .textureBase = tex.tbp0,
                .textureWidth = tex.tbw,
                .textureMaskU =
                    static_cast<uint32_t>(texW - 1),
                .textureMaskV =
                    static_cast<uint32_t>(texH - 1),
                .framebufferBase =
                    GSInternal::framePageBaseToBlock(
                        ctx.frame.fbp),
                .framebufferWidth =
                    std::max<uint32_t>(
                        ctx.frame.fbw, 1u),
                .drawX0 = drawX0,
                .drawX1 = drawX1,
                .drawY0 = drawY0,
                .drawY1 = drawY1,
                .alignedDrawX = alignedDrawX,
                .fixedBaseU =
                    static_cast<int32_t>(fixedScanU),
                .fixedBlockStepU = fixedBlockStepU,
                .fixedScanV = fixedScanV,
                .fixedStepV = fixedStepV,
                .scanlineWorkerIndex =
                    m_scanlineWorkerIndex,
                .scanlineWorkerCount =
                    m_scanlineWorkerCount,
                .recordVectorGroups =
                    packedOverride !=
                    GSRasterizerDetail::
                        PackedSpriteKernelOverride::Automatic,
            };
            for (int lane = 0;
                 lane < kPixelsPerLaneGroup;
                 ++lane)
            {
                packedSprite.fixedLaneU[lane] =
                    static_cast<int32_t>(
                        fixedStepU *
                        static_cast<float>(
                            lane - laneSkip));
            }

            if (packedOverride !=
                GSRasterizerDetail::PackedSpriteKernelOverride::
                    Automatic)
            {
                GSRasterizerDetail::
                    recordPackedSpriteKernelDispatch(
                        kernelSelection.implementation);
            }

            kernelSelection.kernel(packedSprite);
            return;
        }

        for (int y = drawY0; y <= drawY1; ++y)
        {
            if (!ownsScanline(y))
            {
                if (fixedTextureDda)
                    fixedScanV += fixedStepV;
                continue;
            }

            const float ty =
                (static_cast<float>(y) - windowY0) / spriteH;
            float texVf = v0f + (v1f - v0f) * ty;

            // GS scanlines evaluate eight pixels in parallel. In particular,
            // all texture reads for a lane group happen before any of that
            // group's framebuffer writes. This ordering is observable for
            // legal recursive draws where TEX0 and FRAME overlap.
            struct PendingSpritePixel
            {
                int x;
                uint8_t r;
                uint8_t g;
                uint8_t b;
                uint8_t a;
            };
            const int firstGroup =
                drawX0 & ~(kPixelsPerLaneGroup - 1);
            for (int groupX = firstGroup;
                 groupX <= drawX1;
                 groupX += kPixelsPerLaneGroup)
            {
                std::array<PendingSpritePixel,
                           kPixelsPerLaneGroup> pending{};
                int pendingCount = 0;
                const int firstX = std::max(groupX, drawX0);
                const int lastX = std::min(
                    groupX + kPixelsPerLaneGroup - 1, drawX1);
                for (int x = firstX; x <= lastX; ++x)
                {
                    const float tx =
                        (static_cast<float>(x) - windowX0) / spriteW;
                    float texUf = u0f + (u1f - u0f) * tx;
                    uint32_t texel = 0xFFFF00FFu;
                    int traceFixedU = 0;
                    int traceFixedV = 0;
                    if (gs->m_prim.fst)
                    {
                        // The GS retains 16.16 texture precision for sprite
                        // interpolation. Its scanline setup converts a base,
                        // per-lane delta, and eight-pixel step independently;
                        // the vertical edge value is then advanced once per
                        // row. Recomputing from normalized UVs loses both the
                        // low twelve bits and the observable accumulated
                        // floating-point rounding.
                        const int block =
                            (x - alignedDrawX) /
                            kPixelsPerLaneGroup;
                        const int laneOffset =
                            (x & (kPixelsPerLaneGroup - 1)) -
                            laneSkip;
                        const int32_t fixedBaseU =
                            static_cast<int32_t>(fixedScanU);
                        const int32_t fixedLaneU =
                            static_cast<int32_t>(
                                fixedStepU *
                                static_cast<float>(laneOffset));
                        const int32_t fixedU =
                            static_cast<int32_t>(
                                static_cast<uint32_t>(fixedBaseU) +
                                static_cast<uint32_t>(fixedLaneU) +
                                static_cast<uint32_t>(fixedBlockStepU) *
                                    static_cast<uint32_t>(block));
                        const int32_t fixedV =
                            static_cast<int32_t>(fixedScanV);
                        traceFixedU = fixedU;
                        traceFixedV = fixedV;
                        texUf =
                            static_cast<float>(
                                static_cast<uint32_t>(fixedU) +
                                static_cast<uint32_t>(
                                    linearFilter ? 0x8000 : 0)) /
                            65536.0f;
                        texVf =
                            static_cast<float>(
                                static_cast<uint32_t>(fixedV) +
                                static_cast<uint32_t>(
                                    linearFilter ? 0x8000 : 0)) /
                            65536.0f;
                        texel = sampleTextureFixed(
                            gs,
                            fixedU,
                            fixedV,
                            linearFilter,
                            1.0f);
                    }
                    else
                    {
                        texel = sampleTexture(
                            gs,
                            texUf / static_cast<float>(texW),
                            texVf / static_cast<float>(texH),
                            1.0f,
                            0u,
                            0u);
                    }

                    const uint8_t tr =
                        static_cast<uint8_t>(texel & 0xFF);
                    const uint8_t tg =
                        static_cast<uint8_t>((texel >> 8) & 0xFF);
                    const uint8_t tb =
                        static_cast<uint8_t>((texel >> 16) & 0xFF);
                    const uint8_t ta =
                        static_cast<uint8_t>((texel >> 24) & 0xFF);

                    const TextureCombineResult color =
                        combineTexture(
                            tex, r, g, b, a, tr, tg, tb, ta);
                    if (tracePixelMatches(x, y))
                    {
                        std::cerr
                            << "[gs:sprite-pixel] draw="
                            << drawTrace().index
                            << " xy=(" << x << ',' << y << ')'
                            << " texf=(" << texUf << ',' << texVf << ')'
                            << " fixed=(" << traceFixedU << ','
                            << traceFixedV << ')'
                            << " texel=0x" << std::hex << texel
                            << std::dec
                            << " out=("
                            << static_cast<unsigned>(color.r) << ','
                            << static_cast<unsigned>(color.g) << ','
                            << static_cast<unsigned>(color.b) << ','
                            << static_cast<unsigned>(color.a) << ')'
                            << '\n';
                    }
                    PendingSpritePixel &pixel =
                        pending[pendingCount++];
                    pixel.x = x;
                    pixel.r = color.r;
                    pixel.g = color.g;
                    pixel.b = color.b;
                    pixel.a = color.a;
                    if (gs->m_prim.fge)
                    {
                        applyFog(
                            gs->m_fogColor,
                            fog,
                            pixel.r,
                            pixel.g,
                            pixel.b);
                    }
                }

                for (int index = 0; index < pendingCount; ++index)
                {
                    const PendingSpritePixel &pixel = pending[index];
                    writeSpritePixel(
                        pixel.x,
                        y,
                        pixel.r,
                        pixel.g,
                        pixel.b,
                        pixel.a);
                }
            }

            if (fixedTextureDda)
                fixedScanV += fixedStepV;
        }
    }
    else
    {
        if (gs->m_prim.fge)
            applyFog(gs->m_fogColor, fog, r, g, b);
        for (int y = drawY0; y <= drawY1; ++y)
        {
            if (!ownsScanline(y))
                continue;
            for (int x = drawX0; x <= drawX1; ++x)
                writeSpritePixel(x, y, r, g, b, a);
        }
    }
}

#if (defined(__x86_64__) || defined(__i386__)) && \
    defined(__GNUC__) && !defined(__clang__) && \
    !defined(__SANITIZE_THREAD__)
__attribute__((target_clones("fma", "default"),
               optimize("fp-contract=off")))
#endif
void GSRasterizer::drawTriangle(GS *gs)
{
    const GSVertex &v0 = gs->m_vtxQueue[0];
    const GSVertex &v1 = gs->m_vtxQueue[1];
    const GSVertex &v2 = gs->m_vtxQueue[2];
    const auto &ctx = gs->activeContext();

    // GS vertex and offset coordinates are both 12.4 fixed point. Preserve
    // all four fractional bits when mapping to window coordinates.
    const FixedPointVertex p0{
        m_queuedFixedVerticesValid
            ? m_queuedFixedX[0]
            : static_cast<int32_t>(std::lround(v0.x * 16.0f)) -
                  static_cast<int32_t>(ctx.xyoffset.ofx),
        m_queuedFixedVerticesValid
            ? m_queuedFixedY[0]
            : static_cast<int32_t>(std::lround(v0.y * 16.0f)) -
                  static_cast<int32_t>(ctx.xyoffset.ofy),
    };
    const FixedPointVertex p1{
        m_queuedFixedVerticesValid
            ? m_queuedFixedX[1]
            : static_cast<int32_t>(std::lround(v1.x * 16.0f)) -
                  static_cast<int32_t>(ctx.xyoffset.ofx),
        m_queuedFixedVerticesValid
            ? m_queuedFixedY[1]
            : static_cast<int32_t>(std::lround(v1.y * 16.0f)) -
                  static_cast<int32_t>(ctx.xyoffset.ofy),
    };
    const FixedPointVertex p2{
        m_queuedFixedVerticesValid
            ? m_queuedFixedX[2]
            : static_cast<int32_t>(std::lround(v2.x * 16.0f)) -
                  static_cast<int32_t>(ctx.xyoffset.ofx),
        m_queuedFixedVerticesValid
            ? m_queuedFixedY[2]
            : static_cast<int32_t>(std::lround(v2.y * 16.0f)) -
                  static_cast<int32_t>(ctx.xyoffset.ofy),
    };

    const int64_t signedArea = triangleEdge(p0, p1, p2);
    if (signedArea == 0)
        return;

    const bool positiveArea = signedArea > 0;
    const int64_t area = positiveArea ? signedArea : -signedArea;
    const float invArea = 1.0f / static_cast<float>(area);

    // Reject clipped or sub-pixel triangles before preparing their color,
    // depth, and texture gradients. Triangle strips commonly submit many
    // valid but fully clipped primitives.
    int minX = ceilFixed12_4(std::min({p0.x, p1.x, p2.x}));
    int maxX = ceilFixed12_4(std::max({p0.x, p1.x, p2.x}));
    int minY = ceilFixed12_4(std::min({p0.y, p1.y, p2.y}));
    int maxY = ceilFixed12_4(std::max({p0.y, p1.y, p2.y}));

    minX = std::max(minX, static_cast<int>(ctx.scissor.x0));
    maxX = std::min(maxX, static_cast<int>(ctx.scissor.x1) + 1);
    minY = std::max(minY, static_cast<int>(ctx.scissor.y0));
    maxY = std::min(maxY, static_cast<int>(ctx.scissor.y1) + 1);
    if (minX >= maxX || minY >= maxY)
        return;

    const bool topLeft0 = isTopLeftEdge(p1, p2, positiveArea);
    const bool topLeft1 = isTopLeftEdge(p2, p0, positiveArea);
    const bool topLeft2 = isTopLeftEdge(p0, p1, positiveArea);

    struct SortedVertex
    {
        const GSVertex *vertex;
        float x;
        float y;
    };
    SortedVertex sorted[3] = {
        {&v0, static_cast<float>(p0.x) / 16.0f,
         static_cast<float>(p0.y) / 16.0f},
        {&v1, static_cast<float>(p1.x) / 16.0f,
         static_cast<float>(p1.y) / 16.0f},
        {&v2, static_cast<float>(p2.x) / 16.0f,
         static_cast<float>(p2.y) / 16.0f},
    };
    // A strict three-element sorting network is stable for equal Y values
    // and avoids repeating the same sort for each attribute interpolator.
    if (sorted[1].y < sorted[0].y)
        std::swap(sorted[0], sorted[1]);
    if (sorted[2].y < sorted[1].y)
        std::swap(sorted[1], sorted[2]);
    if (sorted[1].y < sorted[0].y)
        std::swap(sorted[0], sorted[1]);
    const float gradientDx01 = sorted[1].x - sorted[0].x;
    const float gradientDy01 = sorted[1].y - sorted[0].y;
    const float gradientDx02 = sorted[2].x - sorted[0].x;
    const float gradientDy02 = sorted[2].y - sorted[0].y;
    const float gradientCross = std::fma(
        gradientDy01,
        gradientDx02,
        -(gradientDx01 * gradientDy02));
    const bool gradientValid = gradientCross != 0.0f;
    const float gradientX01OverCross =
        gradientValid ? gradientDx01 / gradientCross : 0.0f;
    const float gradientY01OverCross =
        gradientValid ? gradientDy01 / gradientCross : 0.0f;
    const float gradientX02OverCross =
        gradientValid ? gradientDx02 / gradientCross : 0.0f;
    const float gradientY02OverCross =
        gradientValid ? gradientDy02 / gradientCross : 0.0f;

    struct TriangleColorDda
    {
        const GSVertex *top;
        const GSVertex *middle;
        const GSVertex *bottom;
        float topX;
        float topY;
        float middleX;
        float middleY;
        float bottomX;
        float bottomY;
        float dx[5];
        float dy[5];
        bool valid;
    };

    TriangleColorDda colorDda{};
    auto prepareColorDda =
        [&]() PS2X_GS_LAMBDA_ALWAYS_INLINE
    {
        if (gs->m_prim.iip || gs->m_prim.fge)
        {
            colorDda.top = sorted[0].vertex;
            colorDda.middle = sorted[1].vertex;
            colorDda.bottom = sorted[2].vertex;
            colorDda.topX = sorted[0].x;
            colorDda.topY = sorted[0].y;
            colorDda.middleX = sorted[1].x;
            colorDda.middleY = sorted[1].y;
            colorDda.bottomX = sorted[2].x;
            colorDda.bottomY = sorted[2].y;

            colorDda.valid = gradientValid;
            if (colorDda.valid)
            {
                const float topColor[5] = {
                    static_cast<float>(colorDda.top->r) * 128.0f,
                    static_cast<float>(colorDda.top->g) * 128.0f,
                    static_cast<float>(colorDda.top->b) * 128.0f,
                    static_cast<float>(colorDda.top->a) * 128.0f,
                    static_cast<float>(colorDda.top->fog) * 128.0f,
                };
                const float middleColor[5] = {
                    static_cast<float>(colorDda.middle->r) * 128.0f,
                    static_cast<float>(colorDda.middle->g) * 128.0f,
                    static_cast<float>(colorDda.middle->b) * 128.0f,
                    static_cast<float>(colorDda.middle->a) * 128.0f,
                    static_cast<float>(colorDda.middle->fog) * 128.0f,
                };
                const float bottomColor[5] = {
                    static_cast<float>(colorDda.bottom->r) * 128.0f,
                    static_cast<float>(colorDda.bottom->g) * 128.0f,
                    static_cast<float>(colorDda.bottom->b) * 128.0f,
                    static_cast<float>(colorDda.bottom->a) * 128.0f,
                    static_cast<float>(colorDda.bottom->fog) * 128.0f,
                };
                for (int channel = 0; channel < 5; ++channel)
                {
                    const float delta01 =
                        middleColor[channel] - topColor[channel];
                    const float delta02 =
                        bottomColor[channel] - topColor[channel];
                    colorDda.dx[channel] = std::fma(
                        delta02,
                        gradientY01OverCross,
                        -(delta01 * gradientY02OverCross));
                    colorDda.dy[channel] = std::fma(
                        delta01,
                        gradientX02OverCross,
                        -(delta02 * gradientX01OverCross));
                }
            }
        }
    };

    struct TriangleDepthDda
    {
        const GSVertex *top;
        const GSVertex *middle;
        const GSVertex *bottom;
        float topX;
        float topY;
        float middleX;
        float middleY;
        float bottomX;
        float bottomY;
        double dx;
        double dy;
        bool valid;
    };

    TriangleDepthDda depthDda{};
    {
        depthDda.top = sorted[0].vertex;
        depthDda.middle = sorted[1].vertex;
        depthDda.bottom = sorted[2].vertex;
        depthDda.topX = sorted[0].x;
        depthDda.topY = sorted[0].y;
        depthDda.middleX = sorted[1].x;
        depthDda.middleY = sorted[1].y;
        depthDda.bottomX = sorted[2].x;
        depthDda.bottomY = sorted[2].y;

        // Match the GS-style setup path's single-precision coordinate
        // gradients and double-precision Z arithmetic.
        depthDda.valid = gradientValid;
        if (depthDda.valid)
        {
            const double delta01 =
                static_cast<double>(depthDda.middle->z) -
                static_cast<double>(depthDda.top->z);
            const double delta02 =
                static_cast<double>(depthDda.bottom->z) -
                static_cast<double>(depthDda.top->z);
            depthDda.dx = std::fma(
                delta02,
                static_cast<double>(gradientY01OverCross),
                -(delta01 *
                  static_cast<double>(gradientY02OverCross)));
            depthDda.dy = std::fma(
                delta01,
                static_cast<double>(gradientX02OverCross),
                -(delta02 *
                  static_cast<double>(gradientX01OverCross)));
        }
    }

    struct TriangleTextureDda
    {
        const GSVertex *top;
        const GSVertex *middle;
        const GSVertex *bottom;
        float topX;
        float topY;
        float middleX;
        float middleY;
        float bottomX;
        float bottomY;
        float dx[3];
        float dy[3];
        bool constantQFixed;
        bool linearFilter;
        bool valid;
    };

    TriangleTextureDda textureDda{};
    auto prepareTextureDda =
        [&]() PS2X_GS_LAMBDA_ALWAYS_INLINE
    {
        if (gs->m_prim.tme)
        {
            textureDda.top = sorted[0].vertex;
            textureDda.middle = sorted[1].vertex;
            textureDda.bottom = sorted[2].vertex;
            textureDda.topX = sorted[0].x;
            textureDda.topY = sorted[0].y;
            textureDda.middleX = sorted[1].x;
            textureDda.middleY = sorted[1].y;
            textureDda.bottomX = sorted[2].x;
            textureDda.bottomY = sorted[2].y;
            const uint8_t minificationFilter =
                static_cast<uint8_t>((ctx.tex1 >> 6u) & 0x7u);
            const bool mipmapsEnabled =
                ((ctx.tex1 >> 2u) & 0x7u) != 0u &&
                minificationFilter >= 2u &&
                minificationFilter <= 5u;
            textureDda.linearFilter =
                tex1UsesLinearFilter(ctx.tex1);
            textureDda.constantQFixed =
                gs->m_prim.fst ||
                (!mipmapsEnabled &&
                 v0.q == v1.q &&
                 v1.q == v2.q);

            textureDda.valid = gradientValid;
            if (textureDda.valid)
            {
                const float textureScaleU =
                    static_cast<float>(65536u << ctx.tex0.tw);
                const float textureScaleV =
                    static_cast<float>(65536u << ctx.tex0.th);
                auto textureValues =
                    [&](const GSVertex &vertex, float values[3])
                {
                    if (gs->m_prim.fst)
                    {
                        const float filterBias =
                            textureDda.linearFilter ? 32768.0f : 0.0f;
                        values[0] =
                            static_cast<float>(vertex.u) * 4096.0f -
                            filterBias;
                        values[1] =
                            static_cast<float>(vertex.v) * 4096.0f -
                            filterBias;
                        values[2] = 1.0f;
                    }
                    else
                    {
                        if (textureDda.constantQFixed)
                        {
                            const float q = fabsQ(vertex.q);
                            const float filterBias =
                                textureDda.linearFilter ? 32768.0f : 0.0f;
                            values[0] =
                                (vertex.s / q) * textureScaleU - filterBias;
                            values[1] =
                                (vertex.t / q) * textureScaleV - filterBias;
                            values[2] = 1.0f;
                        }
                        else
                        {
                            values[0] = vertex.s * textureScaleU;
                            values[1] = vertex.t * textureScaleV;
                            values[2] = vertex.q;
                        }
                    }
                };

                float topTexture[3];
                float middleTexture[3];
                float bottomTexture[3];
                textureValues(*textureDda.top, topTexture);
                textureValues(*textureDda.middle, middleTexture);
                textureValues(*textureDda.bottom, bottomTexture);
                for (int component = 0; component < 3; ++component)
                {
                    const float delta01 =
                        middleTexture[component] - topTexture[component];
                    const float delta02 =
                        bottomTexture[component] - topTexture[component];
                    textureDda.dx[component] = std::fma(
                        delta02,
                        gradientY01OverCross,
                        -(delta01 * gradientY02OverCross));
                    textureDda.dy[component] = std::fma(
                        delta01,
                        gradientX02OverCross,
                        -(delta02 * gradientX01OverCross));
                }
            }
        }
    };
    bool attributeDdasPrepared = false;

    // The GS manual defines pixel centers at integer window coordinates.
    // Rasterize top/left edges inclusively and bottom/right edges exclusively,
    // so triangles which share an edge neither overlap nor leave a gap.
    const FixedPointVertex firstSample{minX * 16, minY * 16};
    int64_t rowEdge0 = triangleEdge(p1, p2, firstSample);
    int64_t rowEdge1 = triangleEdge(p2, p0, firstSample);
    int64_t rowEdge2 = triangleEdge(p0, p1, firstSample);
    int64_t edge0StepX =
        -static_cast<int64_t>(p2.y - p1.y) * 16;
    int64_t edge1StepX =
        -static_cast<int64_t>(p0.y - p2.y) * 16;
    int64_t edge2StepX =
        -static_cast<int64_t>(p1.y - p0.y) * 16;
    int64_t edge0StepY =
        static_cast<int64_t>(p2.x - p1.x) * 16;
    int64_t edge1StepY =
        static_cast<int64_t>(p0.x - p2.x) * 16;
    int64_t edge2StepY =
        static_cast<int64_t>(p1.x - p0.x) * 16;
    if (!positiveArea)
    {
        rowEdge0 = -rowEdge0;
        rowEdge1 = -rowEdge1;
        rowEdge2 = -rowEdge2;
        edge0StepX = -edge0StepX;
        edge1StepX = -edge1StepX;
        edge2StepX = -edge2StepX;
        edge0StepY = -edge0StepY;
        edge1StepY = -edge1StepY;
        edge2StepY = -edge2StepY;
    }

    const bool useFastPixelPath =
        !drawTrace().capturing &&
        ctx.frame.psm == GS_PSM_CT32 &&
        ctx.frame.fbmsk == 0u &&
        !destinationAlphaTestEnabled(
            ctx.test, GS_PSM_CT32) &&
        (ctx.zbuf.psm == GS_PSM_Z24 ||
         ctx.zbuf.psm == GS_PSM_Z32) &&
        (!gs->m_prim.abe ||
         (ctx.alpha & 0xFFu) == 0x44u);
    const GSPixelTraceTarget &pixelTrace =
        pixelTraceTarget();
    const uint32_t fastFbp =
        GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    const uint32_t fastZbp =
        GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
    const uint32_t fastFbw =
        std::max<uint32_t>(ctx.frame.fbw, 1u);
    const bool fastZtestEnabled =
        ((ctx.test >> 16u) & 1u) != 0u;
    const uint32_t fastZtestMethod =
        static_cast<uint32_t>((ctx.test >> 17u) & 3u);
    auto readFastDepth = [&](int x, int y)
    {
        return ctx.zbuf.psm == GS_PSM_Z24
                   ? GSMem::ReadZ24(
                         gs->m_vram, fastZbp, fastFbw, x, y)
                   : GSMem::ReadZ32(
                         gs->m_vram, fastZbp, fastFbw, x, y);
    };
    auto passesFastDepthTest =
        [&](int x, int y, uint32_t z)
    {
        if (!fastZtestEnabled)
            return true;
        if (fastZtestMethod == 0u)
            return false;
        if (fastZtestMethod == 1u)
            return true;

        const uint32_t destinationZ =
            readFastDepth(x, y);
        return fastZtestMethod == 2u
                   ? z >= destinationZ
                   : z > destinationZ;
    };
    auto writeTrianglePixelFast =
        [&](int x,
            int y,
            uint32_t z,
            uint8_t r,
            uint8_t g,
            uint8_t b,
            uint8_t a,
            bool depthPretested) PS2X_GS_LAMBDA_ALWAYS_INLINE
    {
        const AlphaTestResult alphaTest =
            classifyAlphaTest(ctx.test, a);
        if (!alphaTest.writeFramebuffer &&
            !alphaTest.writeDepth)
        {
            return;
        }

        if (!depthPretested &&
            !passesFastDepthTest(x, y, z))
        {
            return;
        }

        if (alphaTest.writeFramebuffer)
        {
            const bool alphaBlendEnabled = gs->m_prim.abe;
            uint32_t destination = 0u;
            if (alphaBlendEnabled ||
                alphaTest.preserveDestinationAlpha)
            {
                destination = GSMem::ReadCT32(
                    gs->m_vram,
                    fastFbp,
                    fastFbw,
                    x,
                    y);
            }
            if (alphaBlendEnabled &&
                !(gs->m_pabe && (a & 0x80u) == 0u))
            {
                const uint32_t blended =
                    blendSourceOverRgb(
                        pack32(r, g, b, a),
                        destination,
                        a);
                r = static_cast<uint8_t>(blended);
                g = static_cast<uint8_t>(blended >> 8u);
                b = static_cast<uint8_t>(blended >> 16u);
            }
            if (!alphaTest.preserveDestinationAlpha &&
                (ctx.fba & 1u) != 0u)
            {
                a = static_cast<uint8_t>(a | 0x80u);
            }
            uint32_t pixel = pack32(r, g, b, a);
            if (alphaTest.preserveDestinationAlpha)
            {
                pixel =
                    (pixel & 0x00FFFFFFu) |
                    (destination & 0xFF000000u);
            }
            GSMem::WriteCT32(
                gs->m_vram,
                fastFbp,
                fastFbw,
                x,
                y,
                pixel);
        }

        if (alphaTest.writeDepth &&
            fastZtestEnabled &&
            !ctx.zbuf.zmask)
        {
            if (ctx.zbuf.psm == GS_PSM_Z24)
                GSMem::WriteZ24(
                    gs->m_vram,
                    fastZbp,
                    fastFbw,
                    x,
                    y,
                    z);
            else
                GSMem::WriteZ32(
                    gs->m_vram,
                    fastZbp,
                    fastFbw,
                    x,
                    y,
                    z);
        }
    };

    for (int y = minY; y < maxY; ++y)
    {
        if (!ownsScanline(y))
        {
            rowEdge0 += edge0StepY;
            rowEdge1 += edge1StepY;
            rowEdge2 += edge2StepY;
            continue;
        }

        int spanStart = 0;
        int spanEnd = maxX - minX;
        auto clipSpanToEdge =
            [&](int64_t edge,
                int64_t step,
                bool topLeft)
        {
            const int64_t threshold = topLeft ? 0 : 1;
            if (step > 0)
            {
                if (edge < threshold)
                {
                    const int64_t delta = threshold - edge;
                    const int64_t first =
                        (delta + step - 1) / step;
                    if (first >= spanEnd)
                        spanStart = spanEnd;
                    else
                        spanStart = std::max(
                            spanStart,
                            static_cast<int>(first));
                }
            }
            else if (step < 0)
            {
                if (edge < threshold)
                {
                    spanEnd = 0;
                }
                else
                {
                    const int64_t last =
                        (edge - threshold) / -step;
                    if (last < spanEnd - 1)
                    {
                        spanEnd =
                            static_cast<int>(last + 1);
                    }
                }
            }
            else if (edge < threshold)
            {
                spanEnd = 0;
            }
        };
        clipSpanToEdge(rowEdge0, edge0StepX, topLeft0);
        clipSpanToEdge(rowEdge1, edge1StepX, topLeft1);
        clipSpanToEdge(rowEdge2, edge2StepX, topLeft2);

        const int scanlineMinX = minX + spanStart;
        const int scanlineMaxX = minX + spanEnd;
        int64_t edge0 =
            rowEdge0 + edge0StepX * spanStart;
        int64_t edge1 =
            rowEdge1 + edge1StepX * spanStart;
        int64_t edge2 =
            rowEdge2 + edge2StepX * spanStart;
        int scanlineLeft = 0;
        int scanlineColor[5] = {};
        float scanlineColorRaw[5] = {};
        double scanlineDepth = 0.0;
        float scanlineTexture[3] = {};
        constexpr int kDdaPixelsPerStep = 8;
        int scanlineDdaSkip = 0;
        bool scanlineStarted = false;
        bool scanlineAttributesStarted = false;
        for (int x = scanlineMinX; x < scanlineMaxX; ++x)
        {
            if (!scanlineStarted)
            {
                scanlineStarted = true;
                scanlineLeft = x;
                scanlineDdaSkip =
                    scanlineLeft & (kDdaPixelsPerStep - 1);
                if (depthDda.valid)
                {
                    const GSVertex *base = depthDda.top;
                    float baseX = depthDda.topX;
                    float baseY = depthDda.topY;
                    if (depthDda.topY == depthDda.middleY)
                    {
                        const bool negativeCross =
                            (depthDda.middleY - depthDda.topY) *
                                    (depthDda.bottomX - depthDda.topX) -
                                (depthDda.middleX - depthDda.topX) *
                                    (depthDda.bottomY - depthDda.topY) <
                            0.0f;
                        if (!negativeCross)
                        {
                            base = depthDda.middle;
                            baseX = depthDda.middleX;
                            baseY = depthDda.middleY;
                        }
                    }
                    else if (y >= static_cast<int>(
                                      std::ceil(depthDda.middleY)))
                    {
                        base = depthDda.middle;
                        baseX = depthDda.middleX;
                        baseY = depthDda.middleY;
                    }

                    const float deltaY =
                        static_cast<float>(y) - baseY;
                    const float prestep =
                        static_cast<float>(scanlineLeft) - baseX;
                    const double edgeDepth = std::fma(
                        depthDda.dy,
                        static_cast<double>(deltaY),
                        static_cast<double>(base->z));
                    scanlineDepth = std::fma(
                        depthDda.dx,
                        static_cast<double>(prestep),
                        edgeDepth);
                }
            }

            // Express interpolation as a base value plus two deltas. Besides
            // reducing accumulated error, this guarantees that a constant
            // vertex attribute remains bit-exact across the whole triangle.
            auto interpolate = [&](float a, float b, float c)
            {
                const float w1 =
                    static_cast<float>(edge1) * invArea;
                const float w2 =
                    static_cast<float>(edge2) * invArea;
                return a + (b - a) * w1 + (c - a) * w2;
            };

            const int ddaAlignedLeft =
                scanlineLeft & ~(kDdaPixelsPerStep - 1);
            const int ddaBlock =
                (x - ddaAlignedLeft) / kDdaPixelsPerStep;
            const int ddaLane =
                x & (kDdaPixelsPerStep - 1);
            const int ddaLaneOffset =
                ddaLane - scanlineDdaSkip;

            double z;
            if (depthDda.valid)
            {
                const float laneDelta =
                    static_cast<float>(depthDda.dx) *
                    static_cast<float>(ddaLaneOffset);
                const double blockDelta =
                    depthDda.dx *
                    static_cast<double>(kDdaPixelsPerStep);
                z = scanlineDepth +
                    static_cast<double>(laneDelta);
                for (int blockIndex = 0;
                     blockIndex < ddaBlock;
                     ++blockIndex)
                {
                    z += blockDelta;
                }
            }
            else
            {
                const float w1 =
                    static_cast<float>(edge1) * invArea;
                const float w2 =
                    static_cast<float>(edge2) * invArea;
                z =
                    v0.z +
                    (v1.z - v0.z) * static_cast<double>(w1) +
                    (v2.z - v0.z) * static_cast<double>(w2);
            }

            const uint32_t integerZ =
                static_cast<uint32_t>(z);
            const bool debugPixel =
                pixelTrace.enabled &&
                x == pixelTrace.x &&
                y == pixelTrace.y;
            const bool depthPretested =
                useFastPixelPath && !debugPixel;
            if (depthPretested &&
                !passesFastDepthTest(x, y, integerZ))
            {
                edge0 += edge0StepX;
                edge1 += edge1StepX;
                edge2 += edge2StepX;
                continue;
            }

            if (!scanlineAttributesStarted)
            {
                scanlineAttributesStarted = true;
                if (!attributeDdasPrepared)
                {
                    attributeDdasPrepared = true;
                    prepareColorDda();
                    prepareTextureDda();
                }
                if (colorDda.valid)
                {
                    const GSVertex *base = colorDda.top;
                    float baseX = colorDda.topX;
                    float baseY = colorDda.topY;
                    if (colorDda.topY == colorDda.middleY)
                    {
                        const bool negativeCross =
                            (colorDda.middleY - colorDda.topY) *
                                    (colorDda.bottomX - colorDda.topX) -
                                (colorDda.middleX - colorDda.topX) *
                                    (colorDda.bottomY - colorDda.topY) <
                            0.0f;
                        if (!negativeCross)
                        {
                            base = colorDda.middle;
                            baseX = colorDda.middleX;
                            baseY = colorDda.middleY;
                        }
                    }
                    else if (y >= static_cast<int>(
                                      std::ceil(colorDda.middleY)))
                    {
                        base = colorDda.middle;
                        baseX = colorDda.middleX;
                        baseY = colorDda.middleY;
                    }

                    const float baseColor[5] = {
                        static_cast<float>(base->r) * 128.0f,
                        static_cast<float>(base->g) * 128.0f,
                        static_cast<float>(base->b) * 128.0f,
                        static_cast<float>(base->a) * 128.0f,
                        static_cast<float>(base->fog) * 128.0f,
                    };
                    const float deltaY =
                        static_cast<float>(y) - baseY;
                    const float prestep =
                        static_cast<float>(scanlineLeft) - baseX;
                    for (int channel = 0; channel < 5; ++channel)
                    {
                        const float edgeValue = std::fma(
                            colorDda.dy[channel],
                            deltaY,
                            baseColor[channel]);
                        scanlineColorRaw[channel] = std::fma(
                            colorDda.dx[channel],
                            prestep,
                            edgeValue);
                        scanlineColor[channel] =
                            static_cast<int>(scanlineColorRaw[channel]);
                    }
                }
                if (textureDda.valid)
                {
                    const GSVertex *base = textureDda.top;
                    float baseX = textureDda.topX;
                    float baseY = textureDda.topY;
                    if (textureDda.topY == textureDda.middleY)
                    {
                        const bool negativeCross =
                            (textureDda.middleY - textureDda.topY) *
                                    (textureDda.bottomX - textureDda.topX) -
                                (textureDda.middleX - textureDda.topX) *
                                    (textureDda.bottomY - textureDda.topY) <
                            0.0f;
                        if (!negativeCross)
                        {
                            base = textureDda.middle;
                            baseX = textureDda.middleX;
                            baseY = textureDda.middleY;
                        }
                    }
                    else if (y >= static_cast<int>(
                                      std::ceil(textureDda.middleY)))
                    {
                        base = textureDda.middle;
                        baseX = textureDda.middleX;
                        baseY = textureDda.middleY;
                    }

                    const float textureScaleU =
                        static_cast<float>(65536u << ctx.tex0.tw);
                    const float textureScaleV =
                        static_cast<float>(65536u << ctx.tex0.th);
                    float baseTexture[3];
                    if (gs->m_prim.fst)
                    {
                        const float filterBias =
                            textureDda.linearFilter ? 32768.0f : 0.0f;
                        baseTexture[0] =
                            static_cast<float>(base->u) * 4096.0f -
                            filterBias;
                        baseTexture[1] =
                            static_cast<float>(base->v) * 4096.0f -
                            filterBias;
                        baseTexture[2] = 1.0f;
                    }
                    else
                    {
                        if (textureDda.constantQFixed)
                        {
                            const float q = fabsQ(base->q);
                            const float filterBias =
                                textureDda.linearFilter ? 32768.0f : 0.0f;
                            baseTexture[0] =
                                (base->s / q) * textureScaleU -
                                filterBias;
                            baseTexture[1] =
                                (base->t / q) * textureScaleV -
                                filterBias;
                            baseTexture[2] = 1.0f;
                        }
                        else
                        {
                            baseTexture[0] = base->s * textureScaleU;
                            baseTexture[1] = base->t * textureScaleV;
                            baseTexture[2] = base->q;
                        }
                    }
                    const float deltaY =
                        static_cast<float>(y) - baseY;
                    const float prestep =
                        static_cast<float>(scanlineLeft) - baseX;
                    for (int component = 0; component < 3; ++component)
                    {
                        const float edgeValue = std::fma(
                            textureDda.dy[component],
                            deltaY,
                            baseTexture[component]);
                        scanlineTexture[component] = std::fma(
                            textureDda.dx[component],
                            prestep,
                            edgeValue);
                    }
                }
            }

            auto interpolateDdaFixed = [&](int channel)
            {
                const int laneDelta = static_cast<int>(
                    colorDda.dx[channel] *
                    static_cast<float>(ddaLaneOffset));
                const float blockDeltaFloat =
                    colorDda.dx[channel] *
                    static_cast<float>(kDdaPixelsPerStep);
                // PCSX2's GS setup path uses the hardware-equivalent
                // round-to-nearest conversion for the eight-pixel fog
                // step, while colour steps and all per-lane offsets use
                // truncation.
                const int blockDelta =
                    channel == 4
                        ? static_cast<int>(std::nearbyint(blockDeltaFloat))
                        : static_cast<int>(blockDeltaFloat);
                if ((channel == 0 || channel == 4) &&
                    debugPixel)
                {
                    std::cerr
                        << (channel == 4
                                ? "[gs:fog-dda] draw="
                                : "[gs:color-dda] draw=")
                        << drawTrace().index
                        << " xy=(" << x << ',' << y << ')'
                        << " scan=" << scanlineColor[channel]
                        << " scan-raw=" << std::setprecision(9)
                        << scanlineColorRaw[channel]
                        << " dx=" << colorDda.dx[channel]
                        << " dy=" << colorDda.dy[channel]
                        << " skip=" << scanlineDdaSkip
                        << " lane=" << ddaLaneOffset
                        << " lane-delta=" << laneDelta
                        << " step=" << blockDelta
                        << " block=" << ddaBlock
                        << " value="
                        << scanlineColor[channel] +
                               laneDelta +
                               ddaBlock * blockDelta
                        << '\n';
                }
                return scanlineColor[channel] +
                       laneDelta +
                       ddaBlock * blockDelta;
            };

            uint8_t r, g, b, a;
            int traceFixedShade[4] = {};
            if (gs->m_prim.iip && colorDda.valid)
            {
#if defined(__SSE4_1__)
                const __m128 dx =
                    _mm_loadu_ps(colorDda.dx);
                const __m128i laneDelta =
                    _mm_cvttps_epi32(_mm_mul_ps(
                        dx,
                        _mm_set1_ps(
                            static_cast<float>(ddaLaneOffset))));
                const __m128i blockDelta =
                    _mm_cvttps_epi32(_mm_mul_ps(
                        dx,
                        _mm_set1_ps(
                            static_cast<float>(
                                kDdaPixelsPerStep))));
                __m128i fixed = _mm_add_epi32(
                    _mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(
                            scanlineColor)),
                    laneDelta);
                fixed = _mm_add_epi32(
                    fixed,
                    _mm_mullo_epi32(
                        blockDelta,
                        _mm_set1_epi32(ddaBlock)));
                _mm_storeu_si128(
                    reinterpret_cast<__m128i *>(traceFixedShade),
                    fixed);
                __m128i color = _mm_srai_epi32(fixed, 7);
                color = _mm_min_epi32(
                    _mm_max_epi32(color, _mm_setzero_si128()),
                    _mm_set1_epi32(255));
                const uint32_t packed =
                    static_cast<uint32_t>(_mm_cvtsi128_si32(
                        _mm_packus_epi16(
                            _mm_packus_epi32(color, color),
                            _mm_setzero_si128())));
                r = static_cast<uint8_t>(packed);
                g = static_cast<uint8_t>(packed >> 8u);
                b = static_cast<uint8_t>(packed >> 16u);
                a = static_cast<uint8_t>(packed >> 24u);
                if (debugPixel)
                    (void)interpolateDdaFixed(0);
#else
                auto shadeChannel = [&](int channel)
                {
                    const int fixed = interpolateDdaFixed(channel);
                    traceFixedShade[channel] = fixed;
                    return clampU8(fixed >> 7);
                };
                r = shadeChannel(0);
                g = shadeChannel(1);
                b = shadeChannel(2);
                a = shadeChannel(3);
#endif
            }
            else if (gs->m_prim.iip)
            {
                r = clampU8(static_cast<int>(interpolate(
                    static_cast<float>(v0.r),
                    static_cast<float>(v1.r),
                    static_cast<float>(v2.r))));
                g = clampU8(static_cast<int>(interpolate(
                    static_cast<float>(v0.g),
                    static_cast<float>(v1.g),
                    static_cast<float>(v2.g))));
                b = clampU8(static_cast<int>(interpolate(
                    static_cast<float>(v0.b),
                    static_cast<float>(v1.b),
                    static_cast<float>(v2.b))));
                a = clampU8(static_cast<int>(interpolate(
                    static_cast<float>(v0.a),
                    static_cast<float>(v1.a),
                    static_cast<float>(v2.a))));
            }
            else
            {
                r = v2.r;
                g = v2.g;
                b = v2.b;
                a = v2.a;
            }
            if (!(gs->m_prim.iip && colorDda.valid))
            {
                traceFixedShade[0] = static_cast<int>(r) << 7;
                traceFixedShade[1] = static_cast<int>(g) << 7;
                traceFixedShade[2] = static_cast<int>(b) << 7;
                traceFixedShade[3] = static_cast<int>(a) << 7;
            }

            const uint8_t initialR = r;
            const uint8_t initialG = g;
            const uint8_t initialB = b;
            const uint8_t initialA = a;
            uint32_t traceTexel = 0u;
            if (gs->m_prim.tme)
            {
                float is, it, iq;
                uint16_t iu, iv;
                bool useFixedTextureCoordinates = false;
                bool fixedTextureLinearBiasApplied = false;
                int32_t fixedTextureU = 0;
                int32_t fixedTextureV = 0;
                if (textureDda.valid)
                {
                    float interpolated[3] = {};
                    if (textureDda.constantQFixed)
                    {
                        int32_t fixed[2] = {};
                        for (int component = 0; component < 2; ++component)
                        {
                            const int32_t scanlineBase =
                                static_cast<int32_t>(
                                    scanlineTexture[component]);
                            const int32_t laneDelta =
                                static_cast<int32_t>(
                                    textureDda.dx[component] *
                                    static_cast<float>(
                                        ddaLaneOffset));
                            const int32_t blockDelta =
                                static_cast<int32_t>(
                                    textureDda.dx[component] *
                                    static_cast<float>(
                                        kDdaPixelsPerStep));
                            const uint32_t value =
                                static_cast<uint32_t>(scanlineBase) +
                                static_cast<uint32_t>(laneDelta) +
                                static_cast<uint32_t>(blockDelta) *
                                    static_cast<uint32_t>(
                                        ddaBlock);
                            fixed[component] =
                                static_cast<int32_t>(value);
                        }

                        useFixedTextureCoordinates = true;
                        fixedTextureLinearBiasApplied =
                            textureDda.linearFilter;
                        fixedTextureU = fixed[0];
                        fixedTextureV = fixed[1];
                        const int32_t filterBias =
                            textureDda.linearFilter ? 0x8000 : 0;
                        const int32_t unbiasedU = static_cast<int32_t>(
                            static_cast<uint32_t>(fixed[0]) +
                            static_cast<uint32_t>(filterBias));
                        const int32_t unbiasedV = static_cast<int32_t>(
                            static_cast<uint32_t>(fixed[1]) +
                            static_cast<uint32_t>(filterBias));
                        const float textureScaleU =
                            static_cast<float>(65536u << ctx.tex0.tw);
                        const float textureScaleV =
                            static_cast<float>(65536u << ctx.tex0.th);
                        is = static_cast<float>(unbiasedU) /
                             textureScaleU;
                        it = static_cast<float>(unbiasedV) /
                             textureScaleV;
                        iq = 1.0f;
                        iu = 0u;
                        iv = 0u;
                    }
                    else
                    {
                        for (int component = 0;
                             component < 3;
                             ++component)
                        {
                            interpolated[component] =
                                scanlineTexture[component] +
                                textureDda.dx[component] *
                                    static_cast<float>(
                                        ddaLaneOffset);
                            const float blockStep =
                                textureDda.dx[component] *
                                static_cast<float>(
                                    kDdaPixelsPerStep);
                            for (int blockIndex = 0;
                                 blockIndex < ddaBlock;
                                 ++blockIndex)
                            {
                                interpolated[component] +=
                                    blockStep;
                            }
                        }

                        if (gs->m_prim.fst)
                        {
                            iu = static_cast<uint16_t>(clampInt(
                                static_cast<int>(
                                    interpolated[0] / 4096.0f),
                                0, 0xFFFF));
                            iv = static_cast<uint16_t>(clampInt(
                                static_cast<int>(
                                    interpolated[1] / 4096.0f),
                                0, 0xFFFF));
                            is = 0.0f;
                            it = 0.0f;
                            iq = 1.0f;
                        }
                        else
                        {
                            const float textureScaleU =
                                static_cast<float>(
                                    65536u << ctx.tex0.tw);
                            const float textureScaleV =
                                static_cast<float>(
                                    65536u << ctx.tex0.th);
                            is = interpolated[0] / textureScaleU;
                            it = interpolated[1] / textureScaleV;
                            iq = interpolated[2];
                            // The GS scanline path converts S/Q and T/Q
                            // directly from the already texture-scaled
                            // interpolants. Normalizing S/T and multiplying
                            // by the texture size again can move an exact
                            // fixed-point boundary down by one ULP.
                            useFixedTextureCoordinates = true;
                            fixedTextureLinearBiasApplied = false;
                            fixedTextureU = static_cast<int32_t>(
                                interpolated[0] / fabsQ(iq));
                            fixedTextureV = static_cast<int32_t>(
                                interpolated[1] / fabsQ(iq));
                            iu = 0u;
                            iv = 0u;
                        }
                    }
                }
                else if (gs->m_prim.fst)
                {
                    iu = static_cast<uint16_t>(clampInt(
                        static_cast<int>(interpolate(
                            static_cast<float>(v0.u),
                            static_cast<float>(v1.u),
                            static_cast<float>(v2.u))),
                        0, 0xFFFF));
                    iv = static_cast<uint16_t>(clampInt(
                        static_cast<int>(interpolate(
                            static_cast<float>(v0.v),
                            static_cast<float>(v1.v),
                            static_cast<float>(v2.v))),
                        0, 0xFFFF));
                    is = 0.0f;
                    it = 0.0f;
                    iq = 1.0f;
                }
                else
                {
                    is = interpolate(v0.s, v1.s, v2.s);
                    it = interpolate(v0.t, v1.t, v2.t);
                    iq = interpolate(v0.q, v1.q, v2.q);
                    iu = 0;
                    iv = 0;
                }

                uint32_t texel =
                    useFixedTextureCoordinates
                        ? sampleTextureFixed(gs,
                                             fixedTextureU,
                                             fixedTextureV,
                                             fixedTextureLinearBiasApplied,
                                             iq)
                        : sampleTexture(gs, is, it, iq, iu, iv);
                traceTexel = texel;
                if (debugPixel)
                {
                    const double traceU =
                        useFixedTextureCoordinates
                            ? (static_cast<double>(fixedTextureU) +
                               (fixedTextureLinearBiasApplied
                                    ? 32768.0
                                    : 0.0)) /
                                  65536.0
                        : gs->m_prim.fst
                            ? static_cast<double>(iu) / 16.0
                            : (static_cast<double>(is) /
                               static_cast<double>(fabsQ(iq))) *
                                  static_cast<double>(
                                      1 << ctx.tex0.tw);
                    const double traceV =
                        useFixedTextureCoordinates
                            ? (static_cast<double>(fixedTextureV) +
                               (fixedTextureLinearBiasApplied
                                    ? 32768.0
                                    : 0.0)) /
                                  65536.0
                        : gs->m_prim.fst
                            ? static_cast<double>(iv) / 16.0
                            : (static_cast<double>(it) /
                               static_cast<double>(fabsQ(iq))) *
                                  static_cast<double>(
                                      1 << ctx.tex0.th);
                    const int traceU0 =
                        static_cast<int>(std::floor(traceU - 0.5));
                    const int traceV0 =
                        static_cast<int>(std::floor(traceV - 0.5));
                    uint32_t traceColors[4] = {};
                    for (int sampleY = 0; sampleY < 2; ++sampleY)
                    {
                        for (int sampleX = 0; sampleX < 2; ++sampleX)
                        {
                            const uint8_t index = static_cast<uint8_t>(
                                gs->ReadVram(ctx.tex0.psm,
                                             ctx.tex0.tbp0,
                                             ctx.tex0.tbw,
                                             traceU0 + sampleX,
                                             traceV0 + sampleY));
                            traceColors[sampleY * 2 + sampleX] =
                                lookupCLUT(gs,
                                           index,
                                           ctx.tex0.cbp,
                                           ctx.tex0.cpsm,
                                           ctx.tex0.csm,
                                           ctx.tex0.csa,
                                           ctx.tex0.psm);
                        }
                    }
                    std::ostringstream message;
                    message
                        << "[gs:pixel] draw=" << drawTrace().index
                        << " xy=(" << x << ',' << y << ')'
                        << " left=" << scanlineLeft
                        << " stq=(" << is << ',' << it << ',' << iq << ')'
                        << " uv=(" << iu << ',' << iv << ')'
                        << " scan=(" << scanlineTexture[0] << ','
                        << scanlineTexture[1] << ','
                        << scanlineTexture[2] << ')'
                        << " d=(" << textureDda.dx[0] << ','
                        << textureDda.dx[1] << ','
                        << textureDda.dx[2] << ')'
                        << "/(" << textureDda.dy[0] << ','
                        << textureDda.dy[1] << ','
                        << textureDda.dy[2] << ')';
                    if (textureDda.constantQFixed)
                    {
                        constexpr int kTraceDdaPixelsPerStep = 8;
                        const int traceAlignedLeft =
                            scanlineLeft &
                            ~(kTraceDdaPixelsPerStep - 1);
                        const int traceSkip =
                            scanlineLeft &
                            (kTraceDdaPixelsPerStep - 1);
                        const int traceBlock =
                            (x - traceAlignedLeft) /
                            kTraceDdaPixelsPerStep;
                        const int traceLaneOffset =
                            (x & (kTraceDdaPixelsPerStep - 1)) -
                            traceSkip;
                        int32_t traceFixed[2] = {};
                        message << " fixed_parts=";
                        for (int component = 0;
                             component < 2;
                             ++component)
                        {
                            const int32_t base = static_cast<int32_t>(
                                scanlineTexture[component]);
                            const int32_t lane = static_cast<int32_t>(
                                textureDda.dx[component] *
                                static_cast<float>(
                                    traceLaneOffset));
                            const int32_t step = static_cast<int32_t>(
                                textureDda.dx[component] *
                                static_cast<float>(
                                    kTraceDdaPixelsPerStep));
                            traceFixed[component] =
                                static_cast<int32_t>(
                                    static_cast<uint32_t>(base) +
                                    static_cast<uint32_t>(lane) +
                                    static_cast<uint32_t>(step) *
                                        static_cast<uint32_t>(
                                            traceBlock));
                            message << (component == 0 ? "(" : "/(")
                                    << base << ',' << lane << ','
                                    << step << ','
                                    << traceFixed[component] << ')';
                        }
                    }
                    message
                        << " texcoord=(" << traceU << ','
                        << traceV << ')'
                        << " taps=(" << std::hex
                        << traceColors[0] << ','
                        << traceColors[1] << ','
                        << traceColors[2] << ','
                        << traceColors[3] << ')'
                        << " texel=0x" << std::hex << texel;
                    std::cerr << message.str() << '\n';
                }

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const auto &tex = ctx.tex0;
                const TextureCombineResult color = combineTextureFixed(
                    tex,
                    traceFixedShade[0],
                    traceFixedShade[1],
                    traceFixedShade[2],
                    traceFixedShade[3],
                    tr,
                    tg,
                    tb,
                    ta);

                r = color.r;
                g = color.g;
                b = color.b;
                a = color.a;
            }

            const uint8_t textureR = r;
            const uint8_t textureG = g;
            const uint8_t textureB = b;
            const uint8_t textureA = a;
            int traceFixedFog = -1;
            if (gs->m_prim.fge)
            {
                if (colorDda.valid)
                {
                    const uint16_t fog = static_cast<uint16_t>(
                        clampInt(interpolateDdaFixed(4),
                                 0,
                                 255 * 128));
                    traceFixedFog = fog;
                    applyFogFixed(
                        gs->m_fogColor, fog, r, g, b);
                }
                else
                {
                    const uint8_t fog = clampU8(
                        static_cast<int>(interpolate(
                            static_cast<float>(v0.fog),
                            static_cast<float>(v1.fog),
                            static_cast<float>(v2.fog))));
                    applyFog(gs->m_fogColor, fog, r, g, b);
                }
            }

            if (debugPixel)
            {
                std::cerr
                    << "[gs:pixel-stages] draw=" << drawTrace().index
                    << " xy=(" << x << ',' << y << ')'
                    << " fixed_rgba=("
                    << traceFixedShade[0] << ','
                    << traceFixedShade[1] << ','
                    << traceFixedShade[2] << ','
                    << traceFixedShade[3] << ')'
                    << " shade=("
                    << static_cast<unsigned>(initialR) << ','
                    << static_cast<unsigned>(initialG) << ','
                    << static_cast<unsigned>(initialB) << ','
                    << static_cast<unsigned>(initialA) << ')'
                    << " texel=0x" << std::hex << traceTexel << std::dec
                    << " texture=("
                    << static_cast<unsigned>(textureR) << ','
                    << static_cast<unsigned>(textureG) << ','
                    << static_cast<unsigned>(textureB) << ','
                    << static_cast<unsigned>(textureA) << ')'
                    << " fog=" << traceFixedFog
                    << " fogged=("
                    << static_cast<unsigned>(r) << ','
                    << static_cast<unsigned>(g) << ','
                    << static_cast<unsigned>(b) << ','
                    << static_cast<unsigned>(a) << ')'
                    << '\n';
            }

            // GS depth interpolation is converted to the integer Z format by
            // truncation. Rounding to nearest makes sloped triangles one unit
            // too near or far at roughly half their pixels.
            if (useFastPixelPath && !debugPixel)
            {
                writeTrianglePixelFast(
                    x,
                    y,
                    integerZ,
                    r,
                    g,
                    b,
                    a,
                    depthPretested);
            }
            else
            {
                writePixel(
                    gs,
                    x,
                    y,
                    integerZ,
                    r,
                    g,
                    b,
                    a);
            }
            edge0 += edge0StepX;
            edge1 += edge1StepX;
            edge2 += edge2StepX;
        }
        rowEdge0 += edge0StepY;
        rowEdge1 += edge1StepY;
        rowEdge2 += edge2StepY;
    }
}

void GSRasterizer::drawLine(GS *gs)
{
    const GSVertex &v0 = gs->m_vtxQueue[0];
    const GSVertex &v1 = gs->m_vtxQueue[1];
    const auto &ctx = gs->activeContext();

    const float x0 =
        (static_cast<float>(std::lround(v0.x * 16.0f)) -
         static_cast<float>(ctx.xyoffset.ofx)) /
        16.0f;
    const float y0 =
        (static_cast<float>(std::lround(v0.y * 16.0f)) -
         static_cast<float>(ctx.xyoffset.ofy)) /
        16.0f;
    const float x1 =
        (static_cast<float>(std::lround(v1.x * 16.0f)) -
         static_cast<float>(ctx.xyoffset.ofx)) /
        16.0f;
    const float y1 =
        (static_cast<float>(std::lround(v1.y * 16.0f)) -
         static_cast<float>(ctx.xyoffset.ofy)) /
        16.0f;

    const float deltaX = x1 - x0;
    const float deltaY = y1 - y0;
    const bool stepX = std::fabs(deltaX) >= std::fabs(deltaY);
    const bool positiveX = deltaX >= 0.0f;
    const bool positiveY = deltaY >= 0.0f;
    const int directionX = positiveX ? 1 : -1;
    const int directionY = positiveY ? 1 : -1;

    float roundedX0 = std::floor(x0 + 0.5f);
    float roundedY0 = std::floor(y0 + 0.5f);
    float roundedX1 = std::floor(x1 + 0.5f);
    float roundedY1 = std::floor(y1 + 0.5f);

    // GS lines use the diamond-exit rule. It determines whether each
    // endpoint's pixel is owned by this segment and keeps joined line strips
    // from drawing an endpoint twice.
    auto exitsEndpointDiamond = [&](float dx, float dy)
    {
        const float distance = std::fabs(dx) + std::fabs(dy);
        if (distance < 0.5f)
            return false;

        if (stepX)
        {
            const bool exitsInDirection =
                positiveX ? (dx > 0.0f) : (dx < 0.0f);
            return exitsInDirection &&
                   (distance > 0.5f || dy >= 0.0f);
        }

        const bool exitsInDirection =
            positiveY ? (dy > 0.0f) : (dy < 0.0f);
        return exitsInDirection &&
               (distance > 0.5f || dx >= 0.0f);
    };

    const bool drawFirst =
        !exitsEndpointDiamond(x0 - roundedX0, y0 - roundedY0);
    const bool drawLast =
        exitsEndpointDiamond(x1 - roundedX1, y1 - roundedY1);
    if (!drawFirst)
    {
        roundedX0 += stepX ? directionX : 0;
        roundedY0 += stepX ? 0 : directionY;
    }
    if (!drawLast)
    {
        roundedX1 -= stepX ? directionX : 0;
        roundedY1 -= stepX ? 0 : directionY;
    }

    if ((stepX
             ? directionX * (roundedX1 - roundedX0)
             : directionY * (roundedY1 - roundedY0)) < 0.0f)
    {
        return;
    }

    const float drivingDelta = std::fabs(stepX ? deltaX : deltaY);
    if (drivingDelta == 0.0f)
        return;

    int pixelX = static_cast<int>(roundedX0);
    int pixelY = static_cast<int>(roundedY0);
    const int lastX = static_cast<int>(roundedX1);
    const int lastY = static_cast<int>(roundedY1);

    const int decisionScale = static_cast<int>(
        2.0f * 16.0f * 16.0f * drivingDelta);
    const int decisionStep = static_cast<int>(
        2.0f * 16.0f * 16.0f * (stepX ? deltaY : deltaX));
    int decision = static_cast<int>(
        decisionScale *
        (stepX ? (y0 - roundedY0) : (x0 - roundedX0)));

    const float prestep = stepX
                              ? directionX * (roundedX0 - x0)
                              : directionY * (roundedY0 - y0);
    decision += static_cast<int>(decisionStep * prestep);

    auto stepDependent = [&](int direction)
    {
        decision -= decisionScale * direction;
        pixelX += stepX ? 0 : direction;
        pixelY += stepX ? direction : 0;
    };

    while (decision >= decisionScale / 2)
        stepDependent(1);
    while (decision < -decisionScale / 2)
        stepDependent(-1);

    int step = 0;
    for (;;)
    {
        const float t = (prestep + static_cast<float>(step)) / drivingDelta;
        uint8_t r, g, b, a;
        if (gs->m_prim.iip)
        {
            r = clampU8(static_cast<int>(v0.r + (v1.r - v0.r) * t));
            g = clampU8(static_cast<int>(v0.g + (v1.g - v0.g) * t));
            b = clampU8(static_cast<int>(v0.b + (v1.b - v0.b) * t));
            a = clampU8(static_cast<int>(v0.a + (v1.a - v0.a) * t));
        }
        else
        {
            r = v1.r;
            g = v1.g;
            b = v1.b;
            a = v1.a;
        }

        if (gs->m_prim.tme)
        {
            uint32_t texel = 0u;
            if (gs->m_prim.fst)
            {
                const uint16_t u = static_cast<uint16_t>(clampInt(
                    static_cast<int>(
                        v0.u + (static_cast<float>(v1.u) - v0.u) * t),
                    0, 0xFFFF));
                const uint16_t v = static_cast<uint16_t>(clampInt(
                    static_cast<int>(
                        v0.v + (static_cast<float>(v1.v) - v0.v) * t),
                    0, 0xFFFF));
                texel = sampleTexture(gs, 0.0f, 0.0f, 1.0f, u, v);
            }
            else
            {
                const float s = v0.s + (v1.s - v0.s) * t;
                const float textureT = v0.t + (v1.t - v0.t) * t;
                const float q = v0.q + (v1.q - v0.q) * t;
                texel = sampleTexture(gs, s, textureT, q, 0u, 0u);
            }

            const TextureCombineResult color = combineTexture(
                ctx.tex0,
                r, g, b, a,
                static_cast<uint8_t>(texel),
                static_cast<uint8_t>(texel >> 8u),
                static_cast<uint8_t>(texel >> 16u),
                static_cast<uint8_t>(texel >> 24u));
            r = color.r;
            g = color.g;
            b = color.b;
            a = color.a;
        }

        if (gs->m_prim.fge)
        {
            const uint8_t fog = clampU8(static_cast<int>(
                v0.fog + (v1.fog - v0.fog) * t));
            applyFog(gs->m_fogColor, fog, r, g, b);
        }

        const double z = v0.z + (v1.z - v0.z) * t;
        if (gs->m_prim.aa1)
        {
            const float coverageDistance =
                65535.0f *
                std::fabs(static_cast<float>(decision) /
                          static_cast<float>(decisionScale));
            const int secondaryCoverage = clampInt(
                static_cast<int>(coverageDistance), 0, 0xFFFF);
            const int secondaryOffset = decision >= 0 ? 1 : -1;

            auto drawAntialiasedPixel =
                [&](int x, int y, int coverage)
            {
                uint8_t outputAlpha = a;
                if (!gs->m_prim.abe || a == 0x80u)
                {
                    outputAlpha = static_cast<uint8_t>(
                        clampInt(coverage, 0, 0xFFFF) >> 9);
                }
                writePixel(gs,
                           x,
                           y,
                           static_cast<u32>(z),
                           r,
                           g,
                           b,
                           outputAlpha,
                           true,
                           true);
            };

            drawAntialiasedPixel(
                pixelX,
                pixelY,
                0xFFFF - secondaryCoverage);
            drawAntialiasedPixel(
                pixelX + (stepX ? 0 : secondaryOffset),
                pixelY + (stepX ? secondaryOffset : 0),
                secondaryCoverage);
        }
        else
        {
            writePixel(gs,
                       pixelX,
                       pixelY,
                       static_cast<u32>(z),
                       r,
                       g,
                       b,
                       a);
        }

        if (stepX ? (pixelX == lastX) : (pixelY == lastY))
            break;

        decision += decisionStep;
        pixelX += stepX ? directionX : 0;
        pixelY += stepX ? 0 : directionY;
        if (stepX ? positiveY : positiveX)
        {
            if (decision >= decisionScale / 2)
                stepDependent(1);
        }
        else
        {
            if (decision < -decisionScale / 2)
                stepDependent(-1);
        }
        ++step;
    }
}
