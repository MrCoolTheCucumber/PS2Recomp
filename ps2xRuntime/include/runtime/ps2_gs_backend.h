#pragma once

#include "runtime/ps2_gs_types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

inline constexpr size_t GS_VRAM_PAGE_SIZE = 8u * 1024u;
inline constexpr size_t GS_VRAM_PAGE_COUNT = 512u;

class GsVramPageMask
{
public:
    static constexpr size_t kPageCount = GS_VRAM_PAGE_COUNT;
    static constexpr size_t kWordCount = kPageCount / 64u;

    void clear() noexcept;
    void set(size_t page) noexcept;
    void setAll() noexcept;
    void unionWith(const GsVramPageMask &other) noexcept;

    [[nodiscard]] bool test(size_t page) const noexcept;
    [[nodiscard]] bool any() const noexcept;
    [[nodiscard]] bool all() const noexcept;
    [[nodiscard]] size_t count() const noexcept;
    [[nodiscard]] bool intersects(const GsVramPageMask &other) const noexcept;
    [[nodiscard]] const std::array<uint64_t, kWordCount> &words() const noexcept;

    bool operator==(const GsVramPageMask &) const noexcept = default;

private:
    std::array<uint64_t, kWordCount> m_words{};
};

struct GsDrawBounds
{
    // Inclusive minimum and exclusive maximum in framebuffer coordinates.
    int32_t x0 = 0;
    int32_t y0 = 0;
    int32_t x1 = 0;
    int32_t y1 = 0;
    bool exact = true;

    [[nodiscard]] bool empty() const noexcept
    {
        return x0 >= x1 || y0 >= y1;
    }
};

struct GsDrawResources
{
    GsVramPageMask framebufferReadPages;
    GsVramPageMask framebufferWritePages;
    GsVramPageMask depthReadPages;
    GsVramPageMask depthWritePages;
    GsVramPageMask texturePages;
    GsVramPageMask mipPages;
    GsVramPageMask clutPages;
    GsVramPageMask readPages;
    GsVramPageMask writePages;

    bool readsDestination = false;
    bool framebufferDepthAlias = false;
    bool framebufferTextureAlias = false;
    bool framebufferClutAlias = false;
    bool unknownMemoryLayout = false;

    [[nodiscard]] bool aliasesAnotherView() const noexcept
    {
        return framebufferDepthAlias ||
               framebufferTextureAlias ||
               framebufferClutAlias;
    }
};

struct GsDrawGlobalState
{
    GSTexaReg texa{};
    GSTexClutReg texclut{};
    uint32_t fogColor = 0;
    uint64_t dimx = 0;
    uint8_t scanMask = 0;
    bool prmodecont = true;
    bool pabe = false;
    bool dither = false;
    bool colorClamp = false;
};

// A command owns a value snapshot of every input needed to classify a draw.
// Its fields are private and only const access is exposed so queued backends
// cannot observe later GS register or vertex-assembly mutations.
class GsDrawCommand final
{
public:
    GsDrawCommand(const GsDrawCommand &) = default;
    GsDrawCommand(GsDrawCommand &&) noexcept = default;
    GsDrawCommand &operator=(const GsDrawCommand &) = default;
    GsDrawCommand &operator=(GsDrawCommand &&) noexcept = default;

    [[nodiscard]] uint64_t sequence() const noexcept { return m_sequence; }
    // Derived diagnostics are intentionally evaluated on demand. Software-only
    // execution never consumes them, which keeps the permanent command seam
    // out of the rasterizer's hot path while retaining one canonical analysis.
    [[nodiscard]] uint64_t stateSignature() const noexcept;
    [[nodiscard]] const GSPrimReg &primitive() const noexcept { return m_primitive; }
    [[nodiscard]] const GSContext &context() const noexcept { return m_context; }
    [[nodiscard]] const std::array<GSVertex, 3> &vertices() const noexcept { return m_vertices; }
    [[nodiscard]] uint8_t vertexCount() const noexcept { return m_vertexCount; }
    [[nodiscard]] const std::array<int32_t, 3> &fixedX() const noexcept { return m_fixedX; }
    [[nodiscard]] const std::array<int32_t, 3> &fixedY() const noexcept { return m_fixedY; }
    [[nodiscard]] const GsDrawBounds &bounds() const noexcept { return m_bounds; }
    [[nodiscard]] const GsDrawGlobalState &globalState() const noexcept { return m_globalState; }
    [[nodiscard]] GsDrawResources resources() const noexcept;

private:
    GsDrawCommand(uint64_t sequence,
                  GSPrimReg primitive,
                  GSContext context,
                  std::array<GSVertex, 3> vertices,
                  uint8_t vertexCount,
                  std::array<int32_t, 3> fixedX,
                  std::array<int32_t, 3> fixedY,
                  GsDrawBounds bounds,
                  GsDrawGlobalState globalState) noexcept;

    friend GsDrawCommand buildGsDrawCommand(
        uint64_t,
        const GSPrimReg &,
        const GSContext &,
        std::span<const GSVertex>,
        const GsDrawGlobalState &);

    uint64_t m_sequence = 0;
    GSPrimReg m_primitive{};
    GSContext m_context{};
    std::array<GSVertex, 3> m_vertices{};
    uint8_t m_vertexCount = 0;
    std::array<int32_t, 3> m_fixedX{};
    std::array<int32_t, 3> m_fixedY{};
    GsDrawBounds m_bounds{};
    GsDrawGlobalState m_globalState{};
};

enum class GsRendererMode : uint8_t
{
    Software,
    Hybrid,
    Verify,
    GpuStrict,
};

enum class GsFlushReason : uint8_t
{
    Explicit,
    Transfer,
    CpuReadback,
    FeedbackSnapshot,
    ClutHazard,
    Finish,
    PresentationLatch,
    DebuggerObservation,
    BackendSwitch,
    Reset,
    SaveLoad,
    Shutdown,
    QueueBackpressure,
};

enum class GsFallbackReason : uint8_t
{
    Supported,
    EmptyBounds,
    InexactBounds,
    UnsupportedPrimitive,
    UnsupportedPrimitiveState,
    UnsupportedFramebufferFormat,
    Textured,
    GouraudShading,
    Fog,
    AlphaBlend,
    AlphaTest,
    DestinationAlphaTest,
    FramebufferMask,
    FramebufferAlphaCorrection,
    Dither,
    ScanMask,
    DepthRead,
    DepthWrite,
    DestinationRead,
    ResourceAlias,
    UnknownMemoryLayout,
};

struct GsBackendDecision
{
    bool supported = false;
    GsFallbackReason reason = GsFallbackReason::UnsupportedPrimitive;
};

class IGsRasterBackend
{
public:
    virtual ~IGsRasterBackend() = default;

    [[nodiscard]] virtual GsBackendDecision
    classify(const GsDrawCommand &command) const = 0;
    virtual void submit(std::span<const GsDrawCommand> commands) = 0;
    virtual void flush(GsFlushReason reason) = 0;
};

[[nodiscard]] GsDrawCommand buildGsDrawCommand(
    uint64_t sequence,
    const GSPrimReg &primitive,
    const GSContext &context,
    std::span<const GSVertex> vertices,
    const GsDrawGlobalState &globalState);

// Phase 3's deliberately narrow first GPU predicate. Keeping it as a pure
// classifier makes the Phase 0 census and future backend use the same reason.
[[nodiscard]] GsBackendDecision classifyGsInitialCt32Sprite(
    const GsDrawCommand &command) noexcept;

[[nodiscard]] std::string_view gsFallbackReasonName(
    GsFallbackReason reason) noexcept;
