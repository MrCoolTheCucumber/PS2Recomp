#pragma once

#include "runtime/ps2_gs_types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

inline constexpr size_t GS_VRAM_PAGE_SIZE = 8u * 1024u;
inline constexpr size_t GS_VRAM_PAGE_COUNT = 512u;

class GsVramPageMask
{
public:
    static constexpr size_t kPageCount = GS_VRAM_PAGE_COUNT;
    static constexpr size_t kWordCount = kPageCount / 64u;

    void clear() noexcept;
    void set(size_t page) noexcept;
    // Sets a consecutive range in the circular 4 MiB page space. Large
    // conservative surfaces commonly cover hundreds of pages, so populate
    // whole mask words instead of paying one read/modify/write per page.
    void setRange(size_t firstPage, size_t pageCount) noexcept;
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
    ResourceHazard,
    PipelineChange,
    Count,
};

enum class GsFallbackReason : uint8_t
{
    Supported,
    BackendUnavailable,
    EmptyBounds,
    InexactBounds,
    UnsupportedPrimitive,
    UnsupportedPrimitiveState,
    UnsupportedFramebufferFormat,
    Textured,
    UnsupportedTextureState,
    UnsupportedTextureFormat,
    UnsupportedTextureFunction,
    UnsupportedTextureCoordinates,
    UnsupportedTextureFilter,
    UnsupportedTextureWrap,
    GouraudShading,
    Fog,
    AlphaBlend,
    AlphaTest,
    DestinationAlphaTest,
    FramebufferMask,
    FramebufferAlphaCorrection,
    Dither,
    ScanMask,
    UnsupportedDepthFormat,
    UnsupportedDepthFunction,
    DepthRead,
    DepthWrite,
    DestinationRead,
    ResourceAlias,
    UnknownMemoryLayout,
    CostModel,
    Count,
};

inline constexpr size_t GS_FLUSH_REASON_COUNT =
    static_cast<size_t>(GsFlushReason::Count);
inline constexpr size_t GS_FALLBACK_REASON_COUNT =
    static_cast<size_t>(GsFallbackReason::Count);

struct GsBackendDecision
{
    bool supported = false;
    GsFallbackReason reason = GsFallbackReason::UnsupportedPrimitive;
};

struct GsSubmissionResult
{
    bool submitted = false;
    bool usedSoftware = false;
    bool usedAccelerated = false;
    GsBackendDecision decision{};
};

// Some exact GPU classes only cross over after several compatible commands
// share one synchronization boundary. A non-zero policy lets Hybrid defer the
// backend choice until the run reaches minimumPixels. maximumCommands keeps
// the undecided queue bounded; a shorter full queue falls back to software.
struct GsHybridBatchPolicy
{
    uint64_t minimumPixels = 0u;
    size_t maximumCommands = 0u;

    [[nodiscard]] bool deferred() const noexcept
    {
        return minimumPixels != 0u && maximumCommands != 0u;
    }
};

struct GsBackendCounters
{
    uint64_t commands = 0;
    uint64_t noopCommands = 0;
    uint64_t softwareCommands = 0;
    uint64_t acceleratedCommands = 0;
    uint64_t verifiedCommands = 0;
    uint64_t fallbackCommands = 0;
    uint64_t strictFailures = 0;
    uint64_t flushes = 0;
    uint64_t backendSwitches = 0;
    uint64_t queueDepth = 0;
    uint64_t queueHighWatermark = 0;
    uint64_t drawPixels = 0;
    uint64_t softwarePixels = 0;
    uint64_t acceleratedPixels = 0;
    uint64_t verifiedPixels = 0;
    uint64_t fallbackPixels = 0;
    uint64_t strictFailurePixels = 0;
    // Sum of host elapsed time spent in software raster kernels. Parallel
    // worker durations are accumulated, so this is a CPU-cost signal rather
    // than a batch wall-clock measurement.
    uint64_t softwareRasterHostNanoseconds = 0;
    std::array<uint64_t, GS_FALLBACK_REASON_COUNT> decisions{};
    std::array<uint64_t, GS_FLUSH_REASON_COUNT> flushReasons{};
};

class IGsRasterBackend
{
public:
    virtual ~IGsRasterBackend() = default;

    [[nodiscard]] virtual GsBackendDecision
    classify(const GsDrawCommand &command) const = 0;
    virtual void submit(std::span<const GsDrawCommand> commands) = 0;
    // Exact empty bounds are a semantic no-op. The router uses this optional
    // notification to retain frontend/debug accounting without changing the
    // active renderer or entering either backend's work queue.
    virtual void recordNoop(const GsDrawCommand &command)
    {
        (void)command;
    }
    virtual void flush(GsFlushReason reason) = 0;
    [[nodiscard]] virtual size_t pendingCommandCount() const noexcept = 0;

    // The default preserves immediate per-command routing. Backends opting
    // into deferred Hybrid admission must also prove compatibility between
    // every adjacent command retained in one candidate run.
    [[nodiscard]] virtual GsHybridBatchPolicy hybridBatchPolicy(
        const GsDrawCommand &) const noexcept
    {
        return {};
    }
    [[nodiscard]] virtual bool hybridBatchCompatible(
        const GsDrawCommand &,
        const GsDrawCommand &) const noexcept
    {
        return false;
    }

    // Recursive texture draws normally require the raster frontend to copy
    // an immutable CPU snapshot before routing. An accelerated backend may
    // opt one exact command into a resident device-to-device snapshot instead.
    // Verify deliberately remains on the CPU-snapshot path so its independent
    // software oracle observes the same immutable bytes.
    [[nodiscard]] virtual bool canCaptureFeedbackSnapshotOnDevice(
        const GsDrawCommand &) const noexcept
    {
        return false;
    }

    // Device-backed implementations use these hooks to keep their private GS
    // memory image coherent with the canonical CPU image. The default no-op
    // keeps software-only and diagnostic backends source-compatible. A CPU
    // access includes write pages because page-granular ownership must preserve
    // every byte not overwritten by the eventual operation.
    virtual void prepareCpuVramAccess(
        const GsVramPageMask &pages,
        GsFlushReason reason)
    {
        (void)pages;
        (void)reason;
    }
    // The split form lets a device backend distinguish CPU reads (which only
    // conflict with pending GPU writes) from CPU writes (which also conflict
    // with pending GPU reads). Legacy backends retain conservative behavior
    // through the union-forwarding default.
    virtual void prepareCpuVramAccess(
        const GsVramPageMask &readPages,
        const GsVramPageMask &writePages,
        GsFlushReason reason)
    {
        GsVramPageMask pages = readPages;
        pages.unionWith(writePages);
        prepareCpuVramAccess(pages, reason);
    }
    virtual void noteCpuVramWrite(const GsVramPageMask &pages)
    {
        (void)pages;
    }
};

// Selects a backend before submission, so unsupported commands cannot
// partially execute. The accelerated slot may later be a GPU or verification
// service; in Verify mode that service owns the independent software/GPU
// images and comparison. The permanent software backend is always present.
class GsBackendRouter final
{
public:
    explicit GsBackendRouter(IGsRasterBackend &softwareBackend) noexcept;

    void setAcceleratedBackend(IGsRasterBackend *backend);
    [[nodiscard]] bool setMode(GsRendererMode mode);
    [[nodiscard]] GsRendererMode mode() const noexcept;
    [[nodiscard]] bool hasAcceleratedBackend() const noexcept;
    [[nodiscard]] bool canCaptureFeedbackSnapshotOnDevice(
        const GsDrawCommand &command) const noexcept;

    [[nodiscard]] GsSubmissionResult submit(
        const GsDrawCommand &command);
    void flush(GsFlushReason reason);
    // Completes the current command stream like a normal flush, but only
    // publishes the requested GPU-newer pages to canonical CPU VRAM. This is
    // intended for bounded observers such as host presentation; full-state
    // observations continue to use flush(reason).
    void flushForCpuReadback(
        const GsVramPageMask &readPages,
        GsFlushReason reason);

    // Brackets a non-rasterizer CPU access to GS local memory. begin drains
    // already-submitted work, downloads only affected device-newer pages, and
    // leaves both copies safe for the access. end publishes the pages actually
    // written by the CPU. Empty write masks are valid for read-only observers.
    void beginCpuVramAccess(
        const GsVramPageMask &readPages,
        const GsVramPageMask &writePages,
        GsFlushReason reason);
    void endCpuVramAccess(const GsVramPageMask &writePages);

    void setCountersEnabled(bool enabled) noexcept;
    [[nodiscard]] bool countersEnabled() const noexcept;
    [[nodiscard]] const GsBackendCounters &counters() const noexcept;
    void resetCounters() noexcept;

private:
    enum class ActiveBackend : uint8_t
    {
        None,
        Software,
        Accelerated,
    };

    void transitionTo(ActiveBackend backend);
    void drainActive(GsFlushReason reason);
    void synchronizeCpuVram(
        const GsVramPageMask &pages,
        GsFlushReason reason);
    void synchronizeCpuVram(
        const GsVramPageMask &readPages,
        const GsVramPageMask &writePages,
        GsFlushReason reason);
    void resolvePendingHybrid(bool accelerate);
    void updateDeferredQueueDepth() noexcept;
    void recordDecision(GsFallbackReason reason) noexcept;
    void recordFlush(GsFlushReason reason) noexcept;
    void updateQueueDepth(const IGsRasterBackend &backend) noexcept;

    IGsRasterBackend *m_softwareBackend = nullptr;
    IGsRasterBackend *m_acceleratedBackend = nullptr;
    GsRendererMode m_mode = GsRendererMode::Software;
    ActiveBackend m_activeBackend = ActiveBackend::None;
    std::vector<GsDrawCommand> m_pendingHybridCommands;
    uint64_t m_pendingHybridPixels = 0u;
    GsHybridBatchPolicy m_pendingHybridPolicy{};
    std::optional<GsDrawCommand> m_admittedHybridTail;
    GsHybridBatchPolicy m_admittedHybridPolicy{};
    bool m_countersEnabled = false;
    GsBackendCounters m_counters{};
};

[[nodiscard]] GsDrawCommand buildGsDrawCommand(
    uint64_t sequence,
    const GSPrimReg &primitive,
    const GSContext &context,
    std::span<const GSVertex> vertices,
    const GsDrawGlobalState &globalState);

// Returns a bounded conservative mask for one GS surface rectangle. bp is in
// 256-byte blocks and bw is the GS buffer-width field in 64-pixel units. The
// physical 4 MiB ring, non-page-aligned bases, packed/high-plane formats, and
// zero row strides are all handled. Unknown formats conservatively touch all
// pages; an empty rectangle touches none.
[[nodiscard]] GsVramPageMask gsVramPagesForSurfaceRect(
    uint32_t bp,
    uint32_t bw,
    uint8_t psm,
    uint32_t x,
    uint32_t y,
    uint32_t width,
    uint32_t height) noexcept;

// Exact flat, untextured CT32 sprite predicate. In addition to opaque writes,
// it admits alpha equations whose defined equal A/B selectors cancel and whose
// D selector copies the source; those equations publish the same raster record.
// Keeping this as a pure classifier gives census and every backend one reason.
[[nodiscard]] GsBackendDecision classifyGsInitialCt32Sprite(
    const GsDrawCommand &command) noexcept;

// Exact flat CT32 sprite subset with an enabled Z32/Z24 depth test. Opaque and
// cancelling source-copy alpha writes publish the same record. Packed Z24
// read-modify-write dependencies remain explicit, and framebuffer/depth aliases
// are rejected before a device record can be published.
[[nodiscard]] GsBackendDecision classifyGsDepthCt32Sprite(
    const GsDrawCommand &command) noexcept;

// Exact flat CT32 depth subset for the standard GS source-over equation
// (Cs-Cd)*As/128+Cd. Unlike source-copy alpha, this contract retains the real
// framebuffer read. Its first slice requires PABE disabled and COLCLAMP
// enabled so the prepared record has one unambiguous integer blend operation.
[[nodiscard]] GsBackendDecision classifyGsSourceOverDepthCt32Sprite(
    const GsDrawCommand &command) noexcept;

// Exact flat CT32 depth subset where TEST always fails alpha and AFAIL keeps
// framebuffer writes only. The admitted blend is (Cs-Cd)*FIX/128+Cd, with
// optional CT32 destination-alpha testing. Depth comparison is ALWAYS and the
// failed alpha test suppresses every depth write regardless of ZMASK.
[[nodiscard]] GsBackendDecision
classifyGsFramebufferOnlyAlphaFailDepthCt32Sprite(
    const GsDrawCommand &command) noexcept;

// Phase 6's first texture predicate deliberately covers only direct CT32
// point sampling whose integer FST coordinates advance exactly one texel per
// framebuffer pixel. Repeat wrap and disjoint source/destination pages keep
// the eventual raw-VRAM executor free of filtering and feedback hazards.
[[nodiscard]] GsBackendDecision classifyGsNearestCt32TexturedSprite(
    const GsDrawCommand &command) noexcept;

// Exact linear-filtered texture predicate. It retains the software FST sprite
// DDA and admits only direct CT32 DECAL RGBA with the explicit
// MXL=0/MMAG=1/MMIN=1 filter state and REPEAT or standard CLAMP on each axis.
[[nodiscard]] GsBackendDecision classifyGsLinearCt32TexturedSprite(
    const GsDrawCommand &command) noexcept;

// Exact recursive-feedback slice for direct CT32 DECAL RGBA with the linear
// FST DDA, standard CLAMP/REPEAT, and Z32/Z24 depth. The texture base must name
// the same complete surface as FRAME. Execution must sample one immutable
// snapshot shared by the consecutive same-surface feedback run; canonical
// color/depth writes are never a texture source for later draws in that run.
[[nodiscard]] GsBackendDecision
classifyGsFeedbackLinearDepthCt32Sprite(
    const GsDrawCommand &command) noexcept;

// Exact recursive-feedback triangle slice selected from retained RAC1
// gameplay. It covers flat STQ, direct CT32 nearest MODULATE sampling where
// FRAME and TEX0 name the same surface. Alpha test NEVER with FB_ONLY and an
// RGB framebuffer mask reduces the architectural write to source alpha only;
// Z32/Z24 GEQUAL is still evaluated, but the failed alpha test suppresses the
// depth write. The optional resource result removes that architecturally
// suppressed depth write from resident coherency accounting.
[[nodiscard]] GsBackendDecision
classifyGsFeedbackNearestDepthCt32Triangle(
    const GsDrawCommand &command,
    GsDrawResources *supportedResources = nullptr) noexcept;

// Phase 5's first triangle predicate deliberately admits only flat,
// untextured CT32 triangles with no destination or depth dependency. It shares
// the ordered state checks above with the sprite slice but remains a separate
// capability until an exact triangle executor is installed.
[[nodiscard]] GsBackendDecision classifyGsFlatCt32Triangle(
    const GsDrawCommand &command) noexcept;

// Exact untextured Gouraud CT32 triangle slice selected by the title census.
// It accepts the standard source-over equation only when all three source
// alpha values are 128, which reduces RGB exactly to a source copy while
// preserving the source alpha byte. Z32/Z24 must be a constant, unmasked
// ALWAYS write; varying color remains part of the contract.
[[nodiscard]] GsBackendDecision
classifyGsGouraudSourceOverDepthCt32Triangle(
    const GsDrawCommand &command) noexcept;

// Exact perspective T8 Gouraud/fog triangle slice used by the dominant RAC1
// pipeline. The contract consumes the frontend's decoded CSM1 CT32 palette,
// supports REPEAT/CLAMP and nearest-mip linear filtering, and retains exact
// source-over or additive-source-alpha blending, alpha GEQUAL with FB_ONLY or
// RGB_ONLY failure, and Z ALWAYS/GEQUAL semantics with optional writes.
// TRIANGLE, TRISTRIP, and TRIFAN commands share the same assembled
// three-vertex raster operation and are classified identically.
[[nodiscard]] GsBackendDecision
classifyGsT8GouraudDepthCt32Triangle(
    const GsDrawCommand &command,
    GsDrawResources *supportedResources = nullptr) noexcept;

[[nodiscard]] std::string_view gsFallbackReasonName(
    GsFallbackReason reason) noexcept;

[[nodiscard]] std::string_view gsFlushReasonName(
    GsFlushReason reason) noexcept;

[[nodiscard]] std::string_view gsRendererModeName(
    GsRendererMode mode) noexcept;
