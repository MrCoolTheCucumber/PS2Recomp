#include "runtime/ps2_gs_backend.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <type_traits>

namespace
{
    constexpr uint32_t kGsBlockCount =
        static_cast<uint32_t>(GS_VRAM_PAGE_COUNT * 32u);

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
        const bool framebufferReadModifyWrite =
            context.frame.psm == GS_PSM_CT24 ||
            context.frame.fbmsk != 0u || primitive.abe || primitive.aa1 ||
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
                GsVramPageMask &pages =
                    level == 0u ? resources.texturePages : resources.mipPages;
                if (!addSurfaceRange(pages,
                                     base,
                                     width,
                                     context.tex0.psm,
                                     textureWidth - 1u,
                                     textureHeight - 1u))
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

GsBackendDecision classifyGsInitialCt32Sprite(
    const GsDrawCommand &command) noexcept
{
    const GSPrimReg &primitive = command.primitive();
    const GSContext &context = command.context();
    const GsDrawGlobalState &global = command.globalState();
    const GsDrawResources resources = command.resources();

    if (command.bounds().empty())
        return {false, GsFallbackReason::EmptyBounds};
    if (!command.bounds().exact)
        return {false, GsFallbackReason::InexactBounds};
    if (primitive.type != GS_PRIM_SPRITE)
        return {false, GsFallbackReason::UnsupportedPrimitive};
    if (primitive.aa1 || primitive.fix)
        return {false, GsFallbackReason::UnsupportedPrimitiveState};
    if (context.frame.psm != GS_PSM_CT32)
        return {false, GsFallbackReason::UnsupportedFramebufferFormat};
    if (primitive.tme)
        return {false, GsFallbackReason::Textured};
    if (primitive.iip)
        return {false, GsFallbackReason::GouraudShading};
    if (primitive.fge)
        return {false, GsFallbackReason::Fog};
    if (primitive.abe)
        return {false, GsFallbackReason::AlphaBlend};
    if ((context.test & 1u) != 0u)
        return {false, GsFallbackReason::AlphaTest};
    if (((context.test >> 14u) & 1u) != 0u)
        return {false, GsFallbackReason::DestinationAlphaTest};
    if (context.frame.fbmsk != 0u)
        return {false, GsFallbackReason::FramebufferMask};
    if ((context.fba & 1u) != 0u)
        return {false, GsFallbackReason::FramebufferAlphaCorrection};
    if (global.dither)
        return {false, GsFallbackReason::Dither};
    if (global.scanMask != 0u)
        return {false, GsFallbackReason::ScanMask};
    if (resources.unknownMemoryLayout)
        return {false, GsFallbackReason::UnknownMemoryLayout};
    if (resources.depthReadPages.any())
        return {false, GsFallbackReason::DepthRead};
    if (resources.depthWritePages.any())
        return {false, GsFallbackReason::DepthWrite};
    if (resources.readsDestination)
        return {false, GsFallbackReason::DestinationRead};
    if (resources.aliasesAnotherView())
        return {false, GsFallbackReason::ResourceAlias};
    return {true, GsFallbackReason::Supported};
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
    if (m_countersEnabled)
        ++m_counters.commands;

    if (m_mode == GsRendererMode::Software)
    {
        transitionTo(ActiveBackend::Software);
        m_softwareBackend->submit(
            std::span<const GsDrawCommand>(&command, 1u));
        m_activeBackend = ActiveBackend::Software;
        if (m_countersEnabled)
        {
            ++m_counters.softwareCommands;
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
    if (m_countersEnabled)
        recordDecision(decision.reason);

    if (decision.supported)
    {
        transitionTo(ActiveBackend::Accelerated);
        m_acceleratedBackend->submit(
            std::span<const GsDrawCommand>(&command, 1u));
        m_activeBackend = ActiveBackend::Accelerated;
        if (m_countersEnabled)
        {
            ++m_counters.acceleratedCommands;
            if (m_mode == GsRendererMode::Verify)
                ++m_counters.verifiedCommands;
            updateQueueDepth(*m_acceleratedBackend);
        }
        return {true, false, true, decision};
    }

    if (m_mode == GsRendererMode::GpuStrict)
    {
        if (m_countersEnabled)
            ++m_counters.strictFailures;
        return {false, false, false, decision};
    }

    transitionTo(ActiveBackend::Software);
    m_softwareBackend->submit(
        std::span<const GsDrawCommand>(&command, 1u));
    m_activeBackend = ActiveBackend::Software;
    if (m_countersEnabled)
    {
        ++m_counters.softwareCommands;
        ++m_counters.fallbackCommands;
        updateQueueDepth(*m_softwareBackend);
    }
    return {true, true, false, decision};
}

void GsBackendRouter::flush(GsFlushReason reason)
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

    m_activeBackend = ActiveBackend::None;
    if (m_countersEnabled)
    {
        m_counters.queueDepth = 0u;
        recordFlush(reason);
    }
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

    if (m_activeBackend == ActiveBackend::Software)
    {
        m_softwareBackend->flush(GsFlushReason::BackendSwitch);
    }
    else if (m_acceleratedBackend)
    {
        m_acceleratedBackend->flush(GsFlushReason::BackendSwitch);
    }
    m_activeBackend = backend;
    if (m_countersEnabled)
    {
        ++m_counters.backendSwitches;
        m_counters.queueDepth = 0u;
        recordFlush(GsFlushReason::BackendSwitch);
    }
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
    case GsFallbackReason::GouraudShading: return "gouraud-shading";
    case GsFallbackReason::Fog: return "fog";
    case GsFallbackReason::AlphaBlend: return "alpha-blend";
    case GsFallbackReason::AlphaTest: return "alpha-test";
    case GsFallbackReason::DestinationAlphaTest: return "destination-alpha-test";
    case GsFallbackReason::FramebufferMask: return "framebuffer-mask";
    case GsFallbackReason::FramebufferAlphaCorrection: return "framebuffer-alpha-correction";
    case GsFallbackReason::Dither: return "dither";
    case GsFallbackReason::ScanMask: return "scan-mask";
    case GsFallbackReason::DepthRead: return "depth-read";
    case GsFallbackReason::DepthWrite: return "depth-write";
    case GsFallbackReason::DestinationRead: return "destination-read";
    case GsFallbackReason::ResourceAlias: return "resource-alias";
    case GsFallbackReason::UnknownMemoryLayout: return "unknown-memory-layout";
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
