#include "runtime/ps2_gs_backend.h"
#include "runtime/ps2_gs_common.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace
{
    constexpr uint32_t kGsBlockCount =
        static_cast<uint32_t>(GS_VRAM_PAGE_COUNT * 32u);

    bool isSourceCopyAlphaBlend(const GSPrimReg &primitive,
                                const GSContext &context) noexcept
    {
        if (!primitive.abe)
            return false;

        const uint8_t a = static_cast<uint8_t>(context.alpha & 0x3u);
        const uint8_t b = static_cast<uint8_t>(
            (context.alpha >> 2u) & 0x3u);
        const uint8_t d = static_cast<uint8_t>(
            (context.alpha >> 6u) & 0x3u);
        // GS ALPHA computes (A-B)*C/128+D for RGB. Equal defined A/B
        // selectors cancel exactly, and D=Cs makes the result a plain source
        // copy. Alpha itself is never blended. Keep selector 3 closed because
        // it is reserved rather than relying on the software fallback's zero
        // interpretation.
        return a <= 2u && a == b && d == 0u;
    }

    bool isSourceOverAlphaBlend(
        const GSPrimReg &primitive,
        const GSContext &context,
        const GsDrawGlobalState &global) noexcept
    {
        // A=Cs, B=Cd, C=As, D=Cd. FIX is irrelevant because C does not
        // select it. Keep PABE and wraparound color arithmetic outside this
        // first exact destination-dependent contract.
        return primitive.abe &&
               (context.alpha & 0xFFu) == 0x44u &&
               !global.pabe && global.colorClamp;
    }

    struct PageGeometry
    {
        uint32_t width;
        uint32_t height;
    };

    bool pageGeometry(uint8_t psm, PageGeometry &geometry) noexcept
    {
        switch (psm)
        {
        case GS_PSM_CT32:
        case GS_PSM_CT24:
        case GS_PSM_Z32:
        case GS_PSM_Z24:
        case GS_PSM_T8H:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
            geometry = {64u, 32u};
            return true;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            geometry = {64u, 64u};
            return true;
        case GS_PSM_T8:
            geometry = {128u, 64u};
            return true;
        case GS_PSM_T4:
            geometry = {128u, 128u};
            return true;
        default:
            return false;
        }
    }

    void addBlockRange(GsVramPageMask &mask,
                       uint32_t startBlock,
                       uint32_t blockCount) noexcept
    {
        if (blockCount == 0u)
            return;
        if (blockCount >= kGsBlockCount)
        {
            mask.setAll();
            return;
        }

        startBlock %= kGsBlockCount;
        const uint32_t firstPage = startBlock / 32u;
        const uint32_t lastBlock = startBlock + blockCount - 1u;
        const uint32_t pageCount = lastBlock / 32u - firstPage + 1u;
        for (uint32_t page = 0u; page < pageCount; ++page)
        {
            mask.set((firstPage + page) % GS_VRAM_PAGE_COUNT);
        }
    }

    bool addSurfaceRange(GsVramPageMask &mask,
                         uint32_t baseBlock,
                         uint32_t widthUnits,
                         uint8_t psm,
                         uint32_t maximumX,
                         uint32_t maximumY) noexcept
    {
        PageGeometry geometry{};
        if (!pageGeometry(psm, geometry))
        {
            mask.setAll();
            return false;
        }

        const uint64_t bufferWidth =
            static_cast<uint64_t>(std::max(widthUnits, 1u)) * 64u;
        const uint64_t pagesPerRow = std::max<uint64_t>(
            (bufferWidth + geometry.width - 1u) / geometry.width, 1u);
        const uint64_t maximumPage =
            static_cast<uint64_t>(maximumY / geometry.height) * pagesPerRow +
            maximumX / geometry.width;
        const uint64_t blockCount = std::min<uint64_t>(
            (maximumPage + 1u) * 32u, kGsBlockCount);
        addBlockRange(mask,
                      baseBlock,
                      static_cast<uint32_t>(blockCount));
        return true;
    }

    int32_t ceilFixed12_4(int32_t value) noexcept
    {
        return value >= 0 ? (value + 15) / 16 : value / 16;
    }

    int64_t triangleEdge(const std::array<int32_t, 3> &x,
                         const std::array<int32_t, 3> &y) noexcept
    {
        return static_cast<int64_t>(x[1] - x[0]) *
                   static_cast<int64_t>(y[2] - y[0]) -
               static_cast<int64_t>(y[1] - y[0]) *
                   static_cast<int64_t>(x[2] - x[0]);
    }

    uint8_t expectedVertexCount(GSPrimType type) noexcept
    {
        switch (type)
        {
        case GS_PRIM_POINT:
            return 1u;
        case GS_PRIM_LINE:
        case GS_PRIM_LINESTRIP:
        case GS_PRIM_SPRITE:
            return 2u;
        case GS_PRIM_TRIANGLE:
        case GS_PRIM_TRISTRIP:
        case GS_PRIM_TRIFAN:
            return 3u;
        default:
            return 0u;
        }
    }

    GsDrawBounds decodeBounds(const GSPrimReg &primitive,
                              const GSContext &context,
                              const std::array<GSVertex, 3> &vertices,
                              const std::array<int32_t, 3> &fixedX,
                              const std::array<int32_t, 3> &fixedY,
                              uint8_t vertexCount) noexcept
    {
        GsDrawBounds bounds{};
        if (vertexCount == 0u || context.scissor.x0 > context.scissor.x1 ||
            context.scissor.y0 > context.scissor.y1)
        {
            return bounds;
        }

        switch (primitive.type)
        {
        case GS_PRIM_POINT:
        {
            const int32_t x =
                static_cast<int32_t>(vertices[0].x12_4 >> 4u) -
                static_cast<int32_t>(context.xyoffset.ofx >> 4u);
            const int32_t y =
                static_cast<int32_t>(vertices[0].y12_4 >> 4u) -
                static_cast<int32_t>(context.xyoffset.ofy >> 4u);
            if (x < context.scissor.x0 || x > context.scissor.x1 ||
                y < context.scissor.y0 || y > context.scissor.y1)
            {
                return bounds;
            }
            bounds = {x, y, x + 1, y + 1, true};
            return bounds;
        }
        case GS_PRIM_SPRITE:
            bounds.x0 = ceilFixed12_4(std::min(fixedX[0], fixedX[1]));
            bounds.x1 = ceilFixed12_4(std::max(fixedX[0], fixedX[1]));
            bounds.y0 = ceilFixed12_4(std::min(fixedY[0], fixedY[1]));
            bounds.y1 = ceilFixed12_4(std::max(fixedY[0], fixedY[1]));
            break;
        case GS_PRIM_TRIANGLE:
        case GS_PRIM_TRISTRIP:
        case GS_PRIM_TRIFAN:
            if (triangleEdge(fixedX, fixedY) == 0)
                return bounds;
            bounds.x0 = ceilFixed12_4(
                std::min({fixedX[0], fixedX[1], fixedX[2]}));
            bounds.x1 = ceilFixed12_4(
                std::max({fixedX[0], fixedX[1], fixedX[2]}));
            bounds.y0 = ceilFixed12_4(
                std::min({fixedY[0], fixedY[1], fixedY[2]}));
            bounds.y1 = ceilFixed12_4(
                std::max({fixedY[0], fixedY[1], fixedY[2]}));
            break;
        case GS_PRIM_LINE:
        case GS_PRIM_LINESTRIP:
            // This contains every pixel the current diamond-exit line path can
            // visit. Exact line ownership remains a software-only capability.
            bounds.x0 = std::min(fixedX[0], fixedX[1]) / 16 - 1;
            bounds.x1 = std::max(fixedX[0], fixedX[1]) / 16 + 2;
            bounds.y0 = std::min(fixedY[0], fixedY[1]) / 16 - 1;
            bounds.y1 = std::max(fixedY[0], fixedY[1]) / 16 + 2;
            bounds.exact = false;
            break;
        default:
            bounds = {
                static_cast<int32_t>(context.scissor.x0),
                static_cast<int32_t>(context.scissor.y0),
                static_cast<int32_t>(context.scissor.x1) + 1,
                static_cast<int32_t>(context.scissor.y1) + 1,
                false};
            break;
        }

        bounds.x0 = std::max<int32_t>(bounds.x0, context.scissor.x0);
        bounds.y0 = std::max<int32_t>(bounds.y0, context.scissor.y0);
        bounds.x1 = std::min<int32_t>(
            bounds.x1, static_cast<int32_t>(context.scissor.x1) + 1);
        bounds.y1 = std::min<int32_t>(
            bounds.y1, static_cast<int32_t>(context.scissor.y1) + 1);
        if (bounds.empty())
            return {};
        return bounds;
    }

    bool isIndexedTexture(uint8_t psm) noexcept
    {
        return psm == GS_PSM_T8 || psm == GS_PSM_T4 ||
               psm == GS_PSM_T8H || psm == GS_PSM_T4HL ||
               psm == GS_PSM_T4HH;
    }

    uint8_t maximumMipLevel(const GSContext &context) noexcept
    {
        const uint8_t minificationFilter =
            static_cast<uint8_t>((context.tex1 >> 6u) & 0x7u);
        if (minificationFilter < 2u || minificationFilter > 5u)
            return 0u;
        return static_cast<uint8_t>(
            std::min<uint64_t>((context.tex1 >> 2u) & 0x7u, 6u));
    }

    void mipSurface(const GSContext &context,
                    uint8_t level,
                    uint32_t &base,
                    uint8_t &width) noexcept
    {
        if (level == 0u)
        {
            base = context.tex0.tbp0;
            width = context.tex0.tbw;
            return;
        }

        const uint64_t mipRegister =
            level <= 3u ? context.miptbp1 : context.miptbp2;
        const uint8_t slot = static_cast<uint8_t>((level - 1u) % 3u);
        const uint8_t shift = static_cast<uint8_t>(slot * 20u);
        base = static_cast<uint32_t>((mipRegister >> shift) & 0x3FFFu);
        width = static_cast<uint8_t>(
            (mipRegister >> (shift + 14u)) & 0x3Fu);
    }

    GsDrawResources describeResources(const GSPrimReg &primitive,
                                      const GSContext &context,
                                      const GsDrawGlobalState &globalState,
                                      const GsDrawBounds &bounds) noexcept
    {
        GsDrawResources resources{};
        if (bounds.empty())
            return resources;

        const uint32_t maximumX = static_cast<uint32_t>(bounds.x1 - 1);
        const uint32_t maximumY = static_cast<uint32_t>(bounds.y1 - 1);
        const uint32_t framebufferBase = context.frame.fbp << 5u;
        GsVramPageMask framebufferSurface;
        if (!addSurfaceRange(framebufferSurface,
                             framebufferBase,
                             context.frame.fbw,
                             context.frame.psm,
                             maximumX,
                             maximumY))
        {
            resources.unknownMemoryLayout = true;
        }
        resources.framebufferWritePages = framebufferSurface;

        const bool alphaTestEnabled = (context.test & 1u) != 0u;
        const bool preserveDestinationAlpha =
            alphaTestEnabled && ((context.test >> 12u) & 0x3u) == 3u;
        const bool destinationAlphaTest =
            ((context.test >> 14u) & 1u) != 0u &&
            context.frame.psm != GS_PSM_CT24;
        const bool alphaBlendReadsDestination =
            primitive.abe &&
            !isSourceCopyAlphaBlend(primitive, context);
        const bool framebufferReadModifyWrite =
            context.frame.psm == GS_PSM_CT24 ||
            context.frame.fbmsk != 0u || alphaBlendReadsDestination ||
            primitive.aa1 ||
            destinationAlphaTest || preserveDestinationAlpha;
        if (framebufferReadModifyWrite)
            resources.framebufferReadPages = framebufferSurface;

        const bool depthTestEnabled =
            ((context.test >> 16u) & 1u) != 0u;
        const uint8_t depthTestMethod =
            static_cast<uint8_t>((context.test >> 17u) & 0x3u);
        const bool depthRead =
            depthTestEnabled && depthTestMethod >= 2u;
        const bool depthWrite = depthTestEnabled && !context.zbuf.zmask;
        if (depthRead || depthWrite)
        {
            GsVramPageMask depthSurface;
            if (!addSurfaceRange(depthSurface,
                                 context.zbuf.zbp << 5u,
                                 context.frame.fbw,
                                 context.zbuf.psm,
                                 maximumX,
                                 maximumY))
            {
                resources.unknownMemoryLayout = true;
            }
            if (depthRead ||
                (depthWrite && context.zbuf.psm == GS_PSM_Z24))
            {
                resources.depthReadPages = depthSurface;
            }
            if (depthWrite)
                resources.depthWritePages = depthSurface;
        }

        if (primitive.tme)
        {
            const uint8_t lastLevel = maximumMipLevel(context);
            for (uint8_t level = 0u; level <= lastLevel; ++level)
            {
                uint32_t base = 0u;
                uint8_t width = 0u;
                mipSurface(context, level, base, width);
                const uint32_t textureWidth =
                    std::max(1u, (1u << context.tex0.tw) >> level);
                const uint32_t textureHeight =
                    std::max(1u, (1u << context.tex0.th) >> level);
                const uint64_t clamp = context.clamp;
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
                GsVramPageMask &pages =
                    level == 0u ? resources.texturePages : resources.mipPages;
                if (!addSurfaceRange(pages,
                                     base,
                                     width,
                                     context.tex0.psm,
                                     maximumTextureX,
                                     maximumTextureY))
                {
                    resources.unknownMemoryLayout = true;
                }
            }

            if (isIndexedTexture(context.tex0.psm))
            {
                const bool fourBit =
                    context.tex0.psm == GS_PSM_T4 ||
                    context.tex0.psm == GS_PSM_T4HL ||
                    context.tex0.psm == GS_PSM_T4HH;
                const uint32_t entryCount = fourBit ? 16u : 256u;
                const bool csm2 = context.tex0.csm != 0u;
                const uint32_t clutWidth =
                    csm2 && globalState.texclut.cbw != 0u
                        ? globalState.texclut.cbw
                        : 1u;
                const uint32_t maximumClutX =
                    csm2
                        ? (static_cast<uint32_t>(globalState.texclut.cou) << 4u) + 15u
                        : 15u;
                const uint32_t maximumClutY =
                    csm2
                        ? static_cast<uint32_t>(globalState.texclut.cov) +
                              ((entryCount - 1u) >> 4u)
                        : 15u;
                if (!addSurfaceRange(resources.clutPages,
                                     context.tex0.cbp,
                                     clutWidth,
                                     context.tex0.cpsm,
                                     maximumClutX,
                                     maximumClutY))
                {
                    resources.unknownMemoryLayout = true;
                }
            }
        }

        resources.readPages = resources.framebufferReadPages;
        resources.readPages.unionWith(resources.depthReadPages);
        resources.readPages.unionWith(resources.texturePages);
        resources.readPages.unionWith(resources.mipPages);
        resources.readPages.unionWith(resources.clutPages);
        resources.writePages = resources.framebufferWritePages;
        resources.writePages.unionWith(resources.depthWritePages);

        resources.readsDestination =
            resources.framebufferReadPages.intersects(
                resources.framebufferWritePages);
        resources.framebufferDepthAlias =
            resources.framebufferWritePages.intersects(resources.depthReadPages) ||
            resources.framebufferWritePages.intersects(resources.depthWritePages);
        resources.framebufferTextureAlias =
            resources.framebufferWritePages.intersects(resources.texturePages) ||
            resources.framebufferWritePages.intersects(resources.mipPages);
        resources.framebufferClutAlias =
            resources.framebufferWritePages.intersects(resources.clutPages);
        return resources;
    }

    class StableHash
    {
    public:
        template <typename T>
        void append(T value) noexcept
        {
            static_assert(std::is_integral_v<T> || std::is_enum_v<T>);
            if constexpr (std::is_enum_v<T>)
            {
                append(static_cast<std::underlying_type_t<T>>(value));
            }
            else if constexpr (std::is_same_v<T, bool>)
            {
                append(static_cast<uint8_t>(value));
            }
            else
            {
                using Unsigned = std::make_unsigned_t<T>;
                Unsigned bits = static_cast<Unsigned>(value);
                for (size_t i = 0; i < sizeof(Unsigned); ++i)
                {
                    m_value ^= static_cast<uint8_t>(bits >> (i * 8u));
                    m_value *= 1099511628211ull;
                }
            }
        }

        [[nodiscard]] uint64_t value() const noexcept { return m_value; }

    private:
        uint64_t m_value = 14695981039346656037ull;
    };

    uint64_t stateSignature(const GSPrimReg &p,
                            const GSContext &c,
                            const GsDrawGlobalState &g) noexcept
    {
        StableHash hash;
        hash.append(p.type);
        hash.append(p.iip);
        hash.append(p.tme);
        hash.append(p.fge);
        hash.append(p.abe);
        hash.append(p.aa1);
        hash.append(p.fst);
        hash.append(p.ctxt);
        hash.append(p.fix);
        hash.append(c.frame.fbp);
        hash.append(c.frame.fbw);
        hash.append(c.frame.psm);
        hash.append(c.frame.fbmsk);
        hash.append(c.scissor.x0);
        hash.append(c.scissor.x1);
        hash.append(c.scissor.y0);
        hash.append(c.scissor.y1);
        hash.append(c.tex0.tbp0);
        hash.append(c.tex0.tbw);
        hash.append(c.tex0.psm);
        hash.append(c.tex0.tw);
        hash.append(c.tex0.th);
        hash.append(c.tex0.tcc);
        hash.append(c.tex0.tfx);
        hash.append(c.tex0.cbp);
        hash.append(c.tex0.cpsm);
        hash.append(c.tex0.csm);
        hash.append(c.tex0.csa);
        hash.append(c.tex0.cld);
        hash.append(c.xyoffset.ofx);
        hash.append(c.xyoffset.ofy);
        hash.append(c.zbuf.zbp);
        hash.append(c.zbuf.psm);
        hash.append(c.zbuf.zmask);
        hash.append(c.tex1);
        hash.append(c.miptbp1);
        hash.append(c.miptbp2);
        hash.append(c.clamp);
        hash.append(c.alpha);
        hash.append(c.test);
        hash.append(c.fba);
        hash.append(g.texa.ta0);
        hash.append(g.texa.aem);
        hash.append(g.texa.ta1);
        hash.append(g.texclut.cbw);
        hash.append(g.texclut.cou);
        hash.append(g.texclut.cov);
        hash.append(g.fogColor);
        hash.append(g.dimx);
        hash.append(g.scanMask);
        hash.append(g.prmodecont);
        hash.append(g.pabe);
        hash.append(g.dither);
        hash.append(g.colorClamp);
        return hash.value();
    }

    bool flushRequiresCanonicalCpuVram(GsFlushReason reason) noexcept
    {
        switch (reason)
        {
        case GsFlushReason::Explicit:
        case GsFlushReason::Finish:
        case GsFlushReason::PresentationLatch:
        case GsFlushReason::DebuggerObservation:
        case GsFlushReason::Reset:
        case GsFlushReason::SaveLoad:
        case GsFlushReason::Shutdown:
            return true;
        case GsFlushReason::BackendSwitch:
        case GsFlushReason::QueueBackpressure:
        case GsFlushReason::ResourceHazard:
        case GsFlushReason::PipelineChange:
        case GsFlushReason::Transfer:
        case GsFlushReason::CpuReadback:
        case GsFlushReason::FeedbackSnapshot:
        case GsFlushReason::ClutHazard:
        case GsFlushReason::Count:
            return false;
        }
        return true;
    }

    uint64_t drawPixelCount(const GsDrawCommand &command) noexcept
    {
        const GsDrawBounds &bounds = command.bounds();
        if (bounds.empty())
            return 0u;
        const uint64_t width = static_cast<uint64_t>(
            static_cast<int64_t>(bounds.x1) - bounds.x0);
        const uint64_t height = static_cast<uint64_t>(
            static_cast<int64_t>(bounds.y1) - bounds.y0);
        return width * height;
    }
}

void GsVramPageMask::clear() noexcept
{
    m_words.fill(0u);
}

void GsVramPageMask::set(size_t page) noexcept
{
    if (page < kPageCount)
        m_words[page / 64u] |= 1ull << (page % 64u);
}

void GsVramPageMask::setAll() noexcept
{
    m_words.fill(std::numeric_limits<uint64_t>::max());
}

void GsVramPageMask::unionWith(const GsVramPageMask &other) noexcept
{
    for (size_t i = 0; i < kWordCount; ++i)
        m_words[i] |= other.m_words[i];
}

bool GsVramPageMask::test(size_t page) const noexcept
{
    return page < kPageCount &&
           (m_words[page / 64u] & (1ull << (page % 64u))) != 0u;
}

bool GsVramPageMask::any() const noexcept
{
    return std::any_of(m_words.begin(), m_words.end(),
                       [](uint64_t word) { return word != 0u; });
}

bool GsVramPageMask::all() const noexcept
{
    return std::all_of(m_words.begin(), m_words.end(),
                       [](uint64_t word)
                       {
                           return word == std::numeric_limits<uint64_t>::max();
                       });
}

size_t GsVramPageMask::count() const noexcept
{
    size_t result = 0u;
    for (uint64_t word : m_words)
        result += std::popcount(word);
    return result;
}

bool GsVramPageMask::intersects(const GsVramPageMask &other) const noexcept
{
    for (size_t i = 0; i < kWordCount; ++i)
    {
        if ((m_words[i] & other.m_words[i]) != 0u)
            return true;
    }
    return false;
}

const std::array<uint64_t, GsVramPageMask::kWordCount> &
GsVramPageMask::words() const noexcept
{
    return m_words;
}

GsVramPageMask gsVramPagesForSurfaceRect(
    uint32_t bp,
    uint32_t bw,
    uint8_t psm,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height) noexcept
{
    GsVramPageMask pages;
    if (width == 0u || height == 0u)
        return pages;

    PageGeometry geometry{};
    if (!pageGeometry(psm, geometry))
    {
        pages.setAll();
        return pages;
    }

    const uint64_t lastX =
        static_cast<uint64_t>(x) + width - 1u;
    const uint64_t lastY =
        static_cast<uint64_t>(y) + height - 1u;
    if (lastX > std::numeric_limits<uint32_t>::max() ||
        lastY > std::numeric_limits<uint32_t>::max())
    {
        pages.setAll();
        return pages;
    }

    const uint64_t firstPageX = x / geometry.width;
    const uint64_t lastPageX = lastX / geometry.width;
    const uint64_t firstPageY = y / geometry.height;
    const uint64_t lastPageY = lastY / geometry.height;
    const uint64_t pageColumns = lastPageX - firstPageX + 1u;
    const uint64_t pageRows = lastPageY - firstPageY + 1u;
    if (pageColumns > GS_VRAM_PAGE_COUNT ||
        pageRows > GS_VRAM_PAGE_COUNT ||
        pageColumns * pageRows > GS_VRAM_PAGE_COUNT)
    {
        pages.setAll();
        return pages;
    }

    bp %= kGsBlockCount;
    const uint64_t basePage = bp / 32u;
    const uint64_t baseBlockPhase = bp % 32u;
    const uint64_t rowStridePages =
        (static_cast<uint64_t>(bw) * 64u) / geometry.width;
    for (uint64_t pageY = firstPageY;
         pageY <= lastPageY; ++pageY)
    {
        for (uint64_t pageX = firstPageX;
             pageX <= lastPageX; ++pageX)
        {
            const size_t physicalPage = static_cast<size_t>(
                (basePage +
                 (pageY % GS_VRAM_PAGE_COUNT) *
                     (rowStridePages % GS_VRAM_PAGE_COUNT) +
                 pageX) % GS_VRAM_PAGE_COUNT);
            pages.set(physicalPage);
            // The GS base pointer may begin part-way through a page. A block
            // selected late in the swizzle then carries into the next physical
            // page; include it without depending on format-specific block
            // permutation tables.
            if (baseBlockPhase != 0u)
            {
                pages.set(
                    (physicalPage + 1u) % GS_VRAM_PAGE_COUNT);
            }
        }
    }
    return pages;
}

GsDrawCommand::GsDrawCommand(
    uint64_t sequence,
    GSPrimReg primitive,
    GSContext context,
    std::array<GSVertex, 3> vertices,
    uint8_t vertexCount,
    std::array<int32_t, 3> fixedX,
    std::array<int32_t, 3> fixedY,
    GsDrawBounds bounds,
    GsDrawGlobalState globalState) noexcept
    : m_sequence(sequence),
      m_primitive(primitive),
      m_context(context),
      m_vertices(vertices),
      m_vertexCount(vertexCount),
      m_fixedX(fixedX),
      m_fixedY(fixedY),
      m_bounds(bounds),
      m_globalState(globalState)
{
}

uint64_t GsDrawCommand::stateSignature() const noexcept
{
    return ::stateSignature(m_primitive, m_context, m_globalState);
}

GsDrawResources GsDrawCommand::resources() const noexcept
{
    return describeResources(
        m_primitive, m_context, m_globalState, m_bounds);
}

GsDrawCommand buildGsDrawCommand(
    uint64_t sequence,
    const GSPrimReg &primitive,
    const GSContext &context,
    std::span<const GSVertex> vertices,
    const GsDrawGlobalState &globalState)
{
    std::array<GSVertex, 3> copiedVertices{};
    const uint8_t vertexCount = static_cast<uint8_t>(std::min<size_t>(
        {vertices.size(), copiedVertices.size(), expectedVertexCount(primitive.type)}));
    std::copy_n(vertices.begin(), vertexCount, copiedVertices.begin());

    std::array<int32_t, 3> fixedX{};
    std::array<int32_t, 3> fixedY{};
    for (uint8_t i = 0u; i < vertexCount; ++i)
    {
        fixedX[i] = static_cast<int32_t>(copiedVertices[i].x12_4) -
                    static_cast<int32_t>(context.xyoffset.ofx);
        fixedY[i] = static_cast<int32_t>(copiedVertices[i].y12_4) -
                    static_cast<int32_t>(context.xyoffset.ofy);
    }
    if (primitive.type == GS_PRIM_SPRITE && vertexCount == 2u)
    {
        fixedX[2] = fixedX[1];
        fixedY[2] = fixedY[1];
    }

    const GsDrawBounds bounds = decodeBounds(
        primitive, context, copiedVertices, fixedX, fixedY, vertexCount);
    return GsDrawCommand(
        sequence,
        primitive,
        context,
        copiedVertices,
        vertexCount,
        fixedX,
        fixedY,
        bounds,
        globalState);
}

namespace
{
    enum class TextureRequirement : uint8_t
    {
        Disabled,
        NearestCt32,
        LinearCt32,
    };

    enum class DepthRequirement : uint8_t
    {
        None,
        Z32OrZ24,
    };

    enum class AlphaRequirement : uint8_t
    {
        Disabled,
        DisabledOrSourceCopy,
        SourceOver,
    };

    enum class AliasRequirement : uint8_t
    {
        Disallowed,
        ExactFramebufferTextureFeedback,
    };

    bool hasOneToOneIntegerTextureAxis(
        int32_t fixed0,
        int32_t fixed1,
        uint16_t texture0,
        uint16_t texture1) noexcept
    {
        if ((fixed0 % 16) != 0 || (fixed1 % 16) != 0 ||
            (texture0 % 16u) != 0u || (texture1 % 16u) != 0u)
        {
            return false;
        }

        const int32_t screenDelta = fixed1 - fixed0;
        const int32_t textureDelta =
            static_cast<int32_t>(texture1) -
            static_cast<int32_t>(texture0);
        return screenDelta != 0 &&
               std::abs(textureDelta) == std::abs(screenDelta);
    }

    bool ct32RectangleHasUniqueWords(
        const GsDrawBounds &bounds,
        uint32_t framebufferWidth) noexcept
    {
        const uint32_t x0 = static_cast<uint32_t>(bounds.x0);
        const uint32_t y0 = static_cast<uint32_t>(bounds.y0);
        const uint32_t x1 = static_cast<uint32_t>(bounds.x1 - 1);
        const uint32_t y1 = static_cast<uint32_t>(bounds.y1 - 1);
        const uint32_t minimumColumn = x0 / 64u;
        const uint32_t maximumColumn = x1 / 64u;
        const uint32_t minimumRow = y0 / 32u;
        const uint32_t maximumRow = y1 / 32u;
        const uint32_t columns = maximumColumn - minimumColumn + 1u;
        if (columns > framebufferWidth)
            return false;

        const uint64_t logicalPageSpan =
            static_cast<uint64_t>(maximumRow - minimumRow) *
                framebufferWidth +
            (maximumColumn - minimumColumn);
        return logicalPageSpan < GS_VRAM_PAGE_COUNT;
    }

    GsBackendDecision classifyFlatCt32State(
        const GsDrawCommand &command,
        GSPrimType expectedPrimitive,
        uint8_t expectedVertices,
        TextureRequirement textureRequirement,
        DepthRequirement depthRequirement = DepthRequirement::None,
        AlphaRequirement alphaRequirement =
            AlphaRequirement::Disabled,
        AliasRequirement aliasRequirement =
            AliasRequirement::Disallowed) noexcept
    {
        const GSPrimReg &primitive = command.primitive();
        const GSContext &context = command.context();
        const GsDrawGlobalState &global = command.globalState();

        if (command.bounds().empty())
            return {false, GsFallbackReason::EmptyBounds};
        if (!command.bounds().exact)
            return {false, GsFallbackReason::InexactBounds};
        if (primitive.type != expectedPrimitive)
            return {false, GsFallbackReason::UnsupportedPrimitive};
        if (command.vertexCount() != expectedVertices ||
            primitive.aa1 || primitive.fix)
        {
            return {false, GsFallbackReason::UnsupportedPrimitiveState};
        }
        if (context.frame.psm != GS_PSM_CT32)
            return {false, GsFallbackReason::UnsupportedFramebufferFormat};
        if (textureRequirement == TextureRequirement::Disabled &&
            primitive.tme)
        {
            return {false, GsFallbackReason::Textured};
        }
        if (textureRequirement != TextureRequirement::Disabled &&
            !primitive.tme)
        {
            return {false, GsFallbackReason::UnsupportedTextureState};
        }
        if (primitive.iip)
            return {false, GsFallbackReason::GouraudShading};
        if (primitive.fge)
            return {false, GsFallbackReason::Fog};
        const bool acceptedAlpha =
            alphaRequirement == AlphaRequirement::SourceOver
                ? isSourceOverAlphaBlend(primitive, context, global)
                : !primitive.abe ||
                      (alphaRequirement ==
                           AlphaRequirement::DisabledOrSourceCopy &&
                       isSourceCopyAlphaBlend(primitive, context));
        if (!acceptedAlpha)
            return {false, GsFallbackReason::AlphaBlend};
        if ((context.test & 1u) != 0u)
            return {false, GsFallbackReason::AlphaTest};
        if (((context.test >> 14u) & 1u) != 0u)
            return {false, GsFallbackReason::DestinationAlphaTest};
        if (context.frame.fbmsk != 0u)
            return {false, GsFallbackReason::FramebufferMask};
        if ((context.fba & 1u) != 0u)
        {
            return {
                false,
                GsFallbackReason::FramebufferAlphaCorrection};
        }
        if (global.dither)
            return {false, GsFallbackReason::Dither};
        if (global.scanMask != 0u)
            return {false, GsFallbackReason::ScanMask};

        if (textureRequirement != TextureRequirement::Disabled)
        {
            if (context.tex0.psm != GS_PSM_CT32)
            {
                return {
                    false,
                    GsFallbackReason::UnsupportedTextureFormat};
            }
            if (context.tex0.tcc != 1u || context.tex0.tfx != 1u)
            {
                return {
                    false,
                    GsFallbackReason::UnsupportedTextureFunction};
            }
            if (!primitive.fst)
            {
                return {
                    false,
                    GsFallbackReason::UnsupportedTextureCoordinates};
            }
            if (textureRequirement == TextureRequirement::NearestCt32 &&
                (!hasOneToOneIntegerTextureAxis(
                     command.fixedX()[0], command.fixedX()[1],
                     command.vertices()[0].u, command.vertices()[1].u) ||
                 !hasOneToOneIntegerTextureAxis(
                     command.fixedY()[0], command.fixedY()[1],
                     command.vertices()[0].v, command.vertices()[1].v)))
            {
                return {
                    false,
                    GsFallbackReason::UnsupportedTextureCoordinates};
            }
            if (context.tex0.tbw == 0u || context.tex0.tbw > 0x3Fu ||
                context.tex0.tw > 10u || context.tex0.th > 10u)
            {
                return {
                    false,
                    GsFallbackReason::UnsupportedTextureState};
            }
            const uint8_t maximumMipLevel =
                static_cast<uint8_t>((context.tex1 >> 2u) & 0x7u);
            const uint8_t magnificationFilter =
                static_cast<uint8_t>((context.tex1 >> 5u) & 0x1u);
            const uint8_t minificationFilter =
                static_cast<uint8_t>((context.tex1 >> 6u) & 0x7u);
            const bool supportedFilter =
                textureRequirement == TextureRequirement::NearestCt32
                    ? maximumMipLevel == 0u &&
                          magnificationFilter == 0u &&
                          minificationFilter == 0u
                    : maximumMipLevel == 0u &&
                          magnificationFilter == 1u &&
                          minificationFilter == 1u;
            if (!supportedFilter)
            {
                return {
                    false,
                    GsFallbackReason::UnsupportedTextureFilter};
            }
            const uint8_t wrapModeU =
                static_cast<uint8_t>(context.clamp & 0x3u);
            const uint8_t wrapModeV =
                static_cast<uint8_t>((context.clamp >> 2u) & 0x3u);
            const uint16_t regionMinU = static_cast<uint16_t>(
                (context.clamp >> 4u) & 0x3FFu);
            const uint16_t regionMaxU = static_cast<uint16_t>(
                (context.clamp >> 14u) & 0x3FFu);
            const uint16_t regionMinV = static_cast<uint16_t>(
                (context.clamp >> 24u) & 0x3FFu);
            const uint16_t regionMaxV = static_cast<uint16_t>(
                (context.clamp >> 34u) & 0x3FFu);
            if (textureRequirement == TextureRequirement::LinearCt32 &&
                (wrapModeU > 1u || wrapModeV > 1u))
            {
                return {
                    false,
                    GsFallbackReason::UnsupportedTextureWrap};
            }
            if ((wrapModeU == 2u && regionMinU > regionMaxU) ||
                (wrapModeV == 2u && regionMinV > regionMaxV))
            {
                return {
                    false,
                    GsFallbackReason::UnsupportedTextureWrap};
            }
            if (!ct32RectangleHasUniqueWords(
                    command.bounds(),
                    std::max<uint32_t>(context.frame.fbw, 1u)))
            {
                return {
                    false,
                    GsFallbackReason::UnknownMemoryLayout};
            }
        }

        const GsDrawResources resources = command.resources();
        if (resources.unknownMemoryLayout)
            return {false, GsFallbackReason::UnknownMemoryLayout};
        if (depthRequirement == DepthRequirement::None)
        {
            if (resources.depthReadPages.any())
                return {false, GsFallbackReason::DepthRead};
            if (resources.depthWritePages.any())
                return {false, GsFallbackReason::DepthWrite};
        }
        else
        {
            if (context.zbuf.psm != GS_PSM_Z32 &&
                context.zbuf.psm != GS_PSM_Z24)
            {
                return {false, GsFallbackReason::UnsupportedDepthFormat};
            }
            const bool depthTestEnabled =
                ((context.test >> 16u) & 1u) != 0u;
            const uint8_t depthTestMethod =
                static_cast<uint8_t>((context.test >> 17u) & 0x3u);
            if (!depthTestEnabled || depthTestMethod == 0u ||
                (depthTestMethod == 1u && context.zbuf.zmask))
            {
                return {false, GsFallbackReason::UnsupportedDepthFunction};
            }
            if (!ct32RectangleHasUniqueWords(
                    command.bounds(),
                    std::max<uint32_t>(context.frame.fbw, 1u)))
            {
                return {false, GsFallbackReason::UnknownMemoryLayout};
            }
        }
        if (resources.readsDestination &&
            alphaRequirement != AlphaRequirement::SourceOver)
            return {false, GsFallbackReason::DestinationRead};
        if (aliasRequirement ==
            AliasRequirement::ExactFramebufferTextureFeedback)
        {
            const bool exactFeedbackSurface =
                resources.framebufferTextureAlias &&
                !resources.framebufferDepthAlias &&
                !resources.framebufferClutAlias &&
                context.tex0.tbp0 == (context.frame.fbp << 5u) &&
                context.tex0.psm == context.frame.psm &&
                context.tex0.tbw ==
                    std::max<uint32_t>(context.frame.fbw, 1u);
            if (!exactFeedbackSurface)
                return {false, GsFallbackReason::ResourceAlias};
        }
        else if (resources.aliasesAnotherView())
        {
            return {false, GsFallbackReason::ResourceAlias};
        }
        return {true, GsFallbackReason::Supported};
    }
}

GsBackendDecision classifyGsInitialCt32Sprite(
    const GsDrawCommand &command) noexcept
{
    return classifyFlatCt32State(
        command, GS_PRIM_SPRITE, 2u,
        TextureRequirement::Disabled,
        DepthRequirement::None,
        AlphaRequirement::DisabledOrSourceCopy);
}

GsBackendDecision classifyGsDepthCt32Sprite(
    const GsDrawCommand &command) noexcept
{
    return classifyFlatCt32State(
        command, GS_PRIM_SPRITE, 2u,
        TextureRequirement::Disabled,
        DepthRequirement::Z32OrZ24,
        AlphaRequirement::DisabledOrSourceCopy);
}

GsBackendDecision classifyGsSourceOverDepthCt32Sprite(
    const GsDrawCommand &command) noexcept
{
    return classifyFlatCt32State(
        command, GS_PRIM_SPRITE, 2u,
        TextureRequirement::Disabled,
        DepthRequirement::Z32OrZ24,
        AlphaRequirement::SourceOver);
}

GsBackendDecision classifyGsNearestCt32TexturedSprite(
    const GsDrawCommand &command) noexcept
{
    return classifyFlatCt32State(
        command, GS_PRIM_SPRITE, 2u,
        TextureRequirement::NearestCt32);
}

GsBackendDecision classifyGsLinearCt32TexturedSprite(
    const GsDrawCommand &command) noexcept
{
    return classifyFlatCt32State(
        command, GS_PRIM_SPRITE, 2u,
        TextureRequirement::LinearCt32);
}

GsBackendDecision classifyGsFeedbackLinearDepthCt32Sprite(
    const GsDrawCommand &command) noexcept
{
    return classifyFlatCt32State(
        command, GS_PRIM_SPRITE, 2u,
        TextureRequirement::LinearCt32,
        DepthRequirement::Z32OrZ24,
        AlphaRequirement::Disabled,
        AliasRequirement::ExactFramebufferTextureFeedback);
}

GsBackendDecision classifyGsFlatCt32Triangle(
    const GsDrawCommand &command) noexcept
{
    return classifyFlatCt32State(
        command, GS_PRIM_TRIANGLE, 3u,
        TextureRequirement::Disabled);
}

GsBackendRouter::GsBackendRouter(
    IGsRasterBackend &softwareBackend) noexcept
    : m_softwareBackend(&softwareBackend)
{
}

void GsBackendRouter::setAcceleratedBackend(
    IGsRasterBackend *backend)
{
    if (m_acceleratedBackend == backend)
        return;

    flush(GsFlushReason::BackendSwitch);
    if (m_acceleratedBackend)
    {
        GsVramPageMask allPages;
        allPages.setAll();
        synchronizeCpuVram(allPages, GsFlushReason::BackendSwitch);
    }
    m_acceleratedBackend = backend;
    if (!m_acceleratedBackend && m_mode != GsRendererMode::Software)
        m_mode = GsRendererMode::Software;
}

bool GsBackendRouter::setMode(GsRendererMode mode)
{
    if (m_mode == mode)
        return true;
    if (mode != GsRendererMode::Software && !m_acceleratedBackend)
        return false;

    flush(GsFlushReason::BackendSwitch);
    GsVramPageMask allPages;
    allPages.setAll();
    synchronizeCpuVram(allPages, GsFlushReason::BackendSwitch);
    m_mode = mode;
    return true;
}

GsRendererMode GsBackendRouter::mode() const noexcept
{
    return m_mode;
}

bool GsBackendRouter::hasAcceleratedBackend() const noexcept
{
    return m_acceleratedBackend != nullptr;
}

GsSubmissionResult GsBackendRouter::submit(
    const GsDrawCommand &command)
{
    uint64_t pixels = 0u;
    if (m_countersEnabled)
    {
        pixels = drawPixelCount(command);
        ++m_counters.commands;
        m_counters.drawPixels += pixels;
    }

    if (m_mode == GsRendererMode::Software)
    {
        transitionTo(ActiveBackend::Software);
        const GsDrawResources resources = command.resources();
        GsVramPageMask accessPages = resources.readPages;
        accessPages.unionWith(resources.writePages);
        synchronizeCpuVram(
            accessPages, GsFlushReason::BackendSwitch);
        m_softwareBackend->submit(
            std::span<const GsDrawCommand>(&command, 1u));
        if (m_acceleratedBackend)
            m_acceleratedBackend->noteCpuVramWrite(resources.writePages);
        m_activeBackend = ActiveBackend::Software;
        if (m_countersEnabled)
        {
            ++m_counters.softwareCommands;
            m_counters.softwarePixels += pixels;
            recordDecision(GsFallbackReason::Supported);
            updateQueueDepth(*m_softwareBackend);
        }
        return {
            true,
            true,
            false,
            {true, GsFallbackReason::Supported}};
    }

    const GsBackendDecision decision =
        m_acceleratedBackend
            ? m_acceleratedBackend->classify(command)
            : GsBackendDecision{
                  false, GsFallbackReason::BackendUnavailable};

    GsHybridBatchPolicy hybridPolicy{};
    if (m_mode == GsRendererMode::Hybrid && decision.supported &&
        m_acceleratedBackend)
    {
        hybridPolicy =
            m_acceleratedBackend->hybridBatchPolicy(command);
    }

    if (m_admittedHybridTail)
    {
        const bool compatible =
            hybridPolicy.deferred() &&
            hybridPolicy.minimumPixels ==
                m_admittedHybridPolicy.minimumPixels &&
            hybridPolicy.maximumCommands ==
                m_admittedHybridPolicy.maximumCommands &&
            m_acceleratedBackend &&
            m_acceleratedBackend->hybridBatchCompatible(
                *m_admittedHybridTail, command);
        if (compatible)
        {
            if (m_countersEnabled)
                recordDecision(decision.reason);
            transitionTo(ActiveBackend::Accelerated);
            m_acceleratedBackend->submit(
                std::span<const GsDrawCommand>(&command, 1u));
            m_activeBackend = ActiveBackend::Accelerated;
            m_admittedHybridTail = command;
            if (m_countersEnabled)
            {
                ++m_counters.acceleratedCommands;
                m_counters.acceleratedPixels += pixels;
                updateQueueDepth(*m_acceleratedBackend);
            }
            return {true, false, true, decision};
        }
        m_admittedHybridTail.reset();
        m_admittedHybridPolicy = {};
    }

    if (!m_pendingHybridCommands.empty())
    {
        const bool compatible =
            hybridPolicy.deferred() &&
            hybridPolicy.minimumPixels ==
                m_pendingHybridPolicy.minimumPixels &&
            hybridPolicy.maximumCommands ==
                m_pendingHybridPolicy.maximumCommands &&
            m_acceleratedBackend &&
            m_acceleratedBackend->hybridBatchCompatible(
                m_pendingHybridCommands.back(), command);
        if (!compatible)
            resolvePendingHybrid(false);
    }

    if (hybridPolicy.deferred())
    {
        if (!m_countersEnabled)
            pixels = drawPixelCount(command);
        if (m_pendingHybridCommands.empty())
        {
            m_pendingHybridPolicy = hybridPolicy;
            m_pendingHybridPixels = 0u;
        }
        m_pendingHybridCommands.push_back(command);
        if (pixels <=
            std::numeric_limits<uint64_t>::max() -
                m_pendingHybridPixels)
            m_pendingHybridPixels += pixels;
        else
            m_pendingHybridPixels =
                std::numeric_limits<uint64_t>::max();
        updateDeferredQueueDepth();

        if (m_pendingHybridPixels >=
            m_pendingHybridPolicy.minimumPixels)
        {
            resolvePendingHybrid(true);
            return {true, false, true, decision};
        }
        if (m_pendingHybridCommands.size() >=
            m_pendingHybridPolicy.maximumCommands)
        {
            resolvePendingHybrid(false);
            return {
                true,
                true,
                false,
                {false, GsFallbackReason::CostModel}};
        }
        return {true, false, false, decision};
    }

    if (decision.supported)
    {
        if (m_countersEnabled)
            recordDecision(decision.reason);
        transitionTo(ActiveBackend::Accelerated);
        m_acceleratedBackend->submit(
            std::span<const GsDrawCommand>(&command, 1u));
        m_activeBackend = ActiveBackend::Accelerated;
        if (m_countersEnabled)
        {
            ++m_counters.acceleratedCommands;
            m_counters.acceleratedPixels += pixels;
            if (m_mode == GsRendererMode::Verify)
            {
                ++m_counters.verifiedCommands;
                m_counters.verifiedPixels += pixels;
            }
            updateQueueDepth(*m_acceleratedBackend);
        }
        return {true, false, true, decision};
    }

    if (m_mode == GsRendererMode::GpuStrict)
    {
        if (m_countersEnabled)
        {
            recordDecision(decision.reason);
            ++m_counters.strictFailures;
            m_counters.strictFailurePixels += pixels;
        }
        return {false, false, false, decision};
    }

    transitionTo(ActiveBackend::Software);
    const GsDrawResources resources = command.resources();
    GsVramPageMask accessPages = resources.readPages;
    accessPages.unionWith(resources.writePages);
    synchronizeCpuVram(
        accessPages, GsFlushReason::BackendSwitch);
    m_softwareBackend->submit(
        std::span<const GsDrawCommand>(&command, 1u));
    if (m_acceleratedBackend)
        m_acceleratedBackend->noteCpuVramWrite(resources.writePages);
    m_activeBackend = ActiveBackend::Software;
    if (m_countersEnabled)
    {
        recordDecision(decision.reason);
        ++m_counters.softwareCommands;
        ++m_counters.fallbackCommands;
        m_counters.softwarePixels += pixels;
        m_counters.fallbackPixels += pixels;
        updateQueueDepth(*m_softwareBackend);
    }
    return {true, true, false, decision};
}

void GsBackendRouter::flush(GsFlushReason reason)
{
    resolvePendingHybrid(false);
    m_admittedHybridTail.reset();
    m_admittedHybridPolicy = {};

    if (reason == GsFlushReason::Shutdown)
    {
        if (m_activeBackend == ActiveBackend::Software)
            m_softwareBackend->flush(reason);
        if (m_acceleratedBackend)
            m_acceleratedBackend->flush(reason);

        m_activeBackend = ActiveBackend::None;
        if (m_countersEnabled)
        {
            m_counters.queueDepth = 0u;
            recordFlush(reason);
        }
        return;
    }

    const bool requiresCpuVram =
        flushRequiresCanonicalCpuVram(reason);

    // Software commands must materialize their reserved CPU writes before a
    // device-to-CPU copy can publish unrelated GPU-newer pages. Accelerated
    // synchronization drains its own work as part of prepareCpuVramAccess.
    if (requiresCpuVram && m_activeBackend == ActiveBackend::Software)
        m_softwareBackend->flush(reason);

    if (requiresCpuVram)
    {
        GsVramPageMask allPages;
        allPages.setAll();
        synchronizeCpuVram(allPages, reason);
        if (m_activeBackend == ActiveBackend::Accelerated &&
            m_acceleratedBackend)
        {
            m_acceleratedBackend->flush(reason);
        }
    }
    else
    {
        drainActive(reason);
    }

    m_activeBackend = ActiveBackend::None;
    if (m_countersEnabled)
    {
        m_counters.queueDepth = 0u;
        recordFlush(reason);
    }
}

void GsBackendRouter::beginCpuVramAccess(
    const GsVramPageMask &readPages,
    const GsVramPageMask &writePages,
    GsFlushReason reason)
{
    resolvePendingHybrid(false);
    m_admittedHybridTail.reset();
    m_admittedHybridPolicy = {};
    drainActive(reason);
    m_activeBackend = ActiveBackend::None;

    GsVramPageMask accessPages = readPages;
    accessPages.unionWith(writePages);
    synchronizeCpuVram(accessPages, reason);

    if (m_countersEnabled)
    {
        m_counters.queueDepth = 0u;
        recordFlush(reason);
    }
}

void GsBackendRouter::endCpuVramAccess(
    const GsVramPageMask &writePages)
{
    if (m_acceleratedBackend)
        m_acceleratedBackend->noteCpuVramWrite(writePages);
}

void GsBackendRouter::setCountersEnabled(bool enabled) noexcept
{
    m_countersEnabled = enabled;
}

bool GsBackendRouter::countersEnabled() const noexcept
{
    return m_countersEnabled;
}

const GsBackendCounters &GsBackendRouter::counters() const noexcept
{
    return m_counters;
}

void GsBackendRouter::resetCounters() noexcept
{
    m_counters = {};
}

void GsBackendRouter::transitionTo(ActiveBackend backend)
{
    if (m_activeBackend == ActiveBackend::None ||
        m_activeBackend == backend)
    {
        m_activeBackend = backend;
        return;
    }

    drainActive(GsFlushReason::BackendSwitch);
    m_activeBackend = backend;
    if (m_countersEnabled)
    {
        ++m_counters.backendSwitches;
        m_counters.queueDepth = 0u;
        recordFlush(GsFlushReason::BackendSwitch);
    }
}

void GsBackendRouter::drainActive(GsFlushReason reason)
{
    switch (m_activeBackend)
    {
    case ActiveBackend::Software:
        m_softwareBackend->flush(reason);
        break;
    case ActiveBackend::Accelerated:
        if (m_acceleratedBackend)
            m_acceleratedBackend->flush(reason);
        break;
    case ActiveBackend::None:
        break;
    }
}

void GsBackendRouter::synchronizeCpuVram(
    const GsVramPageMask &pages,
    GsFlushReason reason)
{
    if (m_acceleratedBackend && pages.any())
        m_acceleratedBackend->prepareCpuVramAccess(pages, reason);
}

void GsBackendRouter::resolvePendingHybrid(bool accelerate)
{
    if (m_pendingHybridCommands.empty())
        return;

    const size_t commandCount = m_pendingHybridCommands.size();
    const uint64_t pixels = m_pendingHybridPixels;
    const std::span<const GsDrawCommand> commands(
        m_pendingHybridCommands.data(),
        m_pendingHybridCommands.size());

    if (accelerate)
    {
        if (!m_acceleratedBackend)
            throw std::logic_error(
                "deferred Hybrid run lost its accelerated backend");
        transitionTo(ActiveBackend::Accelerated);
        m_acceleratedBackend->submit(commands);
        m_activeBackend = ActiveBackend::Accelerated;
        m_admittedHybridTail = commands.back();
        m_admittedHybridPolicy = m_pendingHybridPolicy;
    }
    else
    {
        transitionTo(ActiveBackend::Software);
        GsVramPageMask accessPages;
        GsVramPageMask writePages;
        for (const GsDrawCommand &command : commands)
        {
            const GsDrawResources resources = command.resources();
            accessPages.unionWith(resources.readPages);
            accessPages.unionWith(resources.writePages);
            writePages.unionWith(resources.writePages);
        }
        synchronizeCpuVram(
            accessPages, GsFlushReason::BackendSwitch);
        m_softwareBackend->submit(commands);
        if (m_acceleratedBackend)
            m_acceleratedBackend->noteCpuVramWrite(writePages);
        m_activeBackend = ActiveBackend::Software;
        m_admittedHybridTail.reset();
        m_admittedHybridPolicy = {};
    }

    m_pendingHybridCommands.clear();
    m_pendingHybridPixels = 0u;
    m_pendingHybridPolicy = {};

    if (!m_countersEnabled)
        return;
    for (size_t index = 0u; index < commandCount; ++index)
    {
        recordDecision(
            accelerate
                ? GsFallbackReason::Supported
                : GsFallbackReason::CostModel);
    }
    if (accelerate)
    {
        m_counters.acceleratedCommands += commandCount;
        m_counters.acceleratedPixels += pixels;
        updateQueueDepth(*m_acceleratedBackend);
    }
    else
    {
        m_counters.softwareCommands += commandCount;
        m_counters.fallbackCommands += commandCount;
        m_counters.softwarePixels += pixels;
        m_counters.fallbackPixels += pixels;
        updateQueueDepth(*m_softwareBackend);
    }
}

void GsBackendRouter::updateDeferredQueueDepth() noexcept
{
    if (!m_countersEnabled)
        return;
    uint64_t pending = m_pendingHybridCommands.size();
    if (m_activeBackend == ActiveBackend::Software)
        pending += m_softwareBackend->pendingCommandCount();
    else if (m_activeBackend == ActiveBackend::Accelerated &&
             m_acceleratedBackend)
    {
        pending += m_acceleratedBackend->pendingCommandCount();
    }
    m_counters.queueDepth = pending;
    m_counters.queueHighWatermark = std::max(
        m_counters.queueHighWatermark,
        m_counters.queueDepth);
}

void GsBackendRouter::recordDecision(
    GsFallbackReason reason) noexcept
{
    const size_t index = static_cast<size_t>(reason);
    if (index < m_counters.decisions.size())
        ++m_counters.decisions[index];
}

void GsBackendRouter::recordFlush(GsFlushReason reason) noexcept
{
    ++m_counters.flushes;
    const size_t index = static_cast<size_t>(reason);
    if (index < m_counters.flushReasons.size())
        ++m_counters.flushReasons[index];
}

void GsBackendRouter::updateQueueDepth(
    const IGsRasterBackend &backend) noexcept
{
    m_counters.queueDepth = backend.pendingCommandCount();
    m_counters.queueHighWatermark = std::max(
        m_counters.queueHighWatermark,
        m_counters.queueDepth);
}

std::string_view gsFallbackReasonName(GsFallbackReason reason) noexcept
{
    switch (reason)
    {
    case GsFallbackReason::Supported: return "supported";
    case GsFallbackReason::BackendUnavailable: return "backend-unavailable";
    case GsFallbackReason::EmptyBounds: return "empty-bounds";
    case GsFallbackReason::InexactBounds: return "inexact-bounds";
    case GsFallbackReason::UnsupportedPrimitive: return "unsupported-primitive";
    case GsFallbackReason::UnsupportedPrimitiveState: return "unsupported-primitive-state";
    case GsFallbackReason::UnsupportedFramebufferFormat: return "unsupported-framebuffer-format";
    case GsFallbackReason::Textured: return "textured";
    case GsFallbackReason::UnsupportedTextureState: return "unsupported-texture-state";
    case GsFallbackReason::UnsupportedTextureFormat: return "unsupported-texture-format";
    case GsFallbackReason::UnsupportedTextureFunction: return "unsupported-texture-function";
    case GsFallbackReason::UnsupportedTextureCoordinates: return "unsupported-texture-coordinates";
    case GsFallbackReason::UnsupportedTextureFilter: return "unsupported-texture-filter";
    case GsFallbackReason::UnsupportedTextureWrap: return "unsupported-texture-wrap";
    case GsFallbackReason::GouraudShading: return "gouraud-shading";
    case GsFallbackReason::Fog: return "fog";
    case GsFallbackReason::AlphaBlend: return "alpha-blend";
    case GsFallbackReason::AlphaTest: return "alpha-test";
    case GsFallbackReason::DestinationAlphaTest: return "destination-alpha-test";
    case GsFallbackReason::FramebufferMask: return "framebuffer-mask";
    case GsFallbackReason::FramebufferAlphaCorrection: return "framebuffer-alpha-correction";
    case GsFallbackReason::Dither: return "dither";
    case GsFallbackReason::ScanMask: return "scan-mask";
    case GsFallbackReason::UnsupportedDepthFormat: return "unsupported-depth-format";
    case GsFallbackReason::UnsupportedDepthFunction: return "unsupported-depth-function";
    case GsFallbackReason::DepthRead: return "depth-read";
    case GsFallbackReason::DepthWrite: return "depth-write";
    case GsFallbackReason::DestinationRead: return "destination-read";
    case GsFallbackReason::ResourceAlias: return "resource-alias";
    case GsFallbackReason::UnknownMemoryLayout: return "unknown-memory-layout";
    case GsFallbackReason::CostModel: return "cost-model";
    case GsFallbackReason::Count: break;
    }
    return "unknown";
}

std::string_view gsFlushReasonName(GsFlushReason reason) noexcept
{
    switch (reason)
    {
    case GsFlushReason::Explicit: return "explicit";
    case GsFlushReason::Transfer: return "transfer";
    case GsFlushReason::CpuReadback: return "cpu-readback";
    case GsFlushReason::FeedbackSnapshot: return "feedback-snapshot";
    case GsFlushReason::ClutHazard: return "clut-hazard";
    case GsFlushReason::Finish: return "finish";
    case GsFlushReason::PresentationLatch: return "presentation-latch";
    case GsFlushReason::DebuggerObservation: return "debugger-observation";
    case GsFlushReason::BackendSwitch: return "backend-switch";
    case GsFlushReason::Reset: return "reset";
    case GsFlushReason::SaveLoad: return "save-load";
    case GsFlushReason::Shutdown: return "shutdown";
    case GsFlushReason::QueueBackpressure: return "queue-backpressure";
    case GsFlushReason::ResourceHazard: return "resource-hazard";
    case GsFlushReason::PipelineChange: return "pipeline-change";
    case GsFlushReason::Count: break;
    }
    return "unknown";
}

std::string_view gsRendererModeName(GsRendererMode mode) noexcept
{
    switch (mode)
    {
    case GsRendererMode::Software: return "software";
    case GsRendererMode::Hybrid: return "hybrid";
    case GsRendererMode::Verify: return "verify";
    case GsRendererMode::GpuStrict: return "gpu-strict";
    }
    return "unknown";
}
