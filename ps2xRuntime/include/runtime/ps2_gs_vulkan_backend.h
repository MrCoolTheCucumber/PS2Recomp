#pragma once

#include "runtime/ps2_gs_backend.h"
#include "runtime/ps2_gs_coherency.h"
#include "runtime/ps2_gs_vulkan.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>

struct GsVulkanRasterBackendConfig
{
    GsRendererMode mode = GsRendererMode::Verify;
    // Empty selects PS2X_GS_VERIFY_DUMP_DIR, then the bounded default
    // "gs-vulkan-verify-failures" below the process working directory.
    std::string verificationArtifactDirectory;
    size_t maximumResidentBatchCommands =
        GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH;
    // Hybrid keeps smaller draws on the CPU. Zero disables these policies.
    // Verify and strict modes always exercise every semantically eligible draw.
    uint64_t minimumHybridSpritePixels = 64u;
    // Source-copy alpha equations reuse the flat sprite kernel but retain a
    // separate crossover because the software blend path is more expensive.
    uint64_t minimumHybridSourceCopyAlphaSpritePixels = 8'192u;
    // Source-over without depth reuses the depth-capable blend kernel but
    // avoids all Z traffic. It retains a separate crossover from depth work.
    uint64_t minimumHybridSourceOverSpritePixels = 8'192u;
    // Depth work includes one CT32 write plus an exact Z32/Z24 test/write. The
    // default is the conservative isolated-draw crossover across both formats.
    uint64_t minimumHybridDepthCt32SpritePixels = 262'144u;
    // Framebuffer-only alpha-fail draws perform no depth I/O and are commonly
    // emitted as compatible runs. Admission therefore uses aggregate run work
    // so one resident submission can amortize its queue and fence overhead.
    uint64_t
        minimumHybridFramebufferOnlyAlphaFailDepthCt32RunPixels =
            212'480u;
    // Nearest texture work is the exact clipped sample/write rectangle.
    uint64_t minimumHybridNearestCt32SpritePixels = 8'192u;
    // REPEAT linear texture work is the exact clipped sample/write rectangle.
    // Zero disables the policy; tiled execution preserves the binary32 V DDA.
    uint64_t minimumHybridLinearCt32SpritePixels = 131'072u;
    // Standard CLAMP is materially more expensive in the software oracle, so
    // it has a separately measured Hybrid crossover. Zero disables the policy.
    uint64_t minimumHybridLinearCt32ClampSpritePixels = 8'192u;
    // Recursive feedback pays for one immutable 4 MiB snapshot per compatible
    // frontend run. The default is the measured 16-draw title-shaped crossover,
    // so admission is deferred and based on aggregate run pixels. Zero admits
    // every semantically exact feedback draw immediately.
    uint64_t minimumHybridFeedbackLinearDepthCt32RunPixels = 212'480u;
    // Triangle work is the conservative clipped bounding box dispatched by the
    // compute kernel, including samples rejected by its exact edge tests.
    uint64_t minimumHybridTriangleCandidatePixels = 32'768u;
    // Small Gouraud/depth triangles are emitted as one overlapping run. A
    // resident ordered-tile dispatch amortizes setup using aggregate clipped
    // bounds instead of applying the flat triangle's isolated-draw gate.
    uint64_t minimumHybridGouraudDepthCt32TriangleRunPixels = 262'144u;
};

struct GsVulkanRasterBackendStatistics
{
    uint64_t commandsAttempted = 0u;
    uint64_t commandsCompleted = 0u;
    uint64_t gpuRequestsFailed = 0u;
    uint64_t verifiedCommands = 0u;
    uint64_t committedGpuCommands = 0u;
    uint64_t verificationMismatches = 0u;
    uint64_t bytesCompared = 0u;
    uint64_t flushes = 0u;
    uint64_t residentCommands = 0u;
    uint64_t residentBatchesCompleted = 0u;
    uint64_t largestResidentBatch = 0u;
    uint64_t resourceHazardDrains = 0u;
    uint64_t queueBackpressureDrains = 0u;
    uint64_t pipelineChangeDrains = 0u;
    uint64_t cpuAccessPreparations = 0u;
    std::array<uint64_t, GS_FLUSH_REASON_COUNT>
        pageDownloadOperationsByReason{};
    std::array<uint64_t, GS_FLUSH_REASON_COUNT>
        pagesDownloadedByReason{};
    GsVramCoherencySummary pageOwnership{};
    GsVramCoherencyStatistics coherency{};
    std::string lastVerificationArtifact;
};

// Production completion accounting only needs aggregate frontend metadata.
// Keeping it separate from GsDrawCommand lets a resident batch avoid retaining
// and copying complete register/vertex snapshots after GPU preparation.
struct GsVulkanAcceleratedCommitBatch
{
    uint64_t commandCount = 0u;
    uint64_t candidatePixels = 0u;
    // FRAME.FBP is a 9-bit GS page index, so the physical page-mask shape is
    // also an exact compact set representation for touched framebuffer bases.
    GsVramPageMask framebufferBasePages;
};

struct GsVulkanDecodedPalette
{
    std::span<const uint32_t> colors;
    uint64_t generation = 0u;
    uint16_t texa = 0u;
    uint8_t sourcePsm = 0u;
    uint8_t csm = 0u;
    uint8_t csa = 0u;
};

struct GsVulkanFeedbackSnapshotIdentity
{
    uint64_t generation = 0u;
    bool deviceResident = false;
};

// Verify mode retains Phase 3's independent whole-image transaction and full
// 4 MiB comparison. Hybrid and strict use the page-coherency hooks inherited
// from IGsRasterBackend, so canonical CPU VRAM may be stale until the router
// prepares an overlapping CPU access. The caller must keep the canonical VRAM
// address and extent stable for the backend's lifetime.
class GsVulkanRasterBackend final : public IGsRasterBackend
{
public:
    using DrawCallback = std::function<void(const GsDrawCommand &)>;
    using AcceleratedBatchCommitCallback =
        std::function<void(const GsVulkanAcceleratedCommitBatch &)>;
    // Verify invokes this synchronously once per admitted feedback draw;
    // resident modes copy it once when opening an admitted feedback batch. The
    // caller owns the returned bytes and must keep them stable only for the
    // callback.
    using FeedbackSnapshotCallback =
        std::function<std::span<const uint8_t>()>;
    // Monotonically identifies the frontend's current immutable feedback run.
    // Device-resident snapshots reuse bytes only while this identity matches.
    using FeedbackSnapshotGenerationCallback =
        std::function<GsVulkanFeedbackSnapshotIdentity()>;
    // The callback refreshes and exposes the frontend's architectural decoded
    // CLUT. Its exact identity fields must change whenever the decoded colors
    // can change. The backend copies all 256 entries synchronously only when a
    // resident batch observes a new identity.
    using DecodedPaletteCallback =
        std::function<GsVulkanDecodedPalette()>;

    [[nodiscard]] static std::unique_ptr<GsVulkanRasterBackend> create(
        const GsVulkanServiceConfig &serviceConfig,
        const GsVulkanRasterBackendConfig &backendConfig,
        std::span<uint8_t> canonicalVram,
        DrawCallback softwareOracle,
        DrawCallback acceleratedCommit,
        GsVulkanCapabilityReport *report = nullptr,
        std::string *error = nullptr,
        FeedbackSnapshotCallback feedbackSnapshot = {},
        DecodedPaletteCallback decodedPalette = {},
        AcceleratedBatchCommitCallback acceleratedBatchCommit = {},
        FeedbackSnapshotGenerationCallback
            feedbackSnapshotGeneration = {});

    // Dependency-injected construction keeps mismatch, execution-failure, and
    // lifecycle tests deterministic even on compiled-out or GPU-less hosts.
    [[nodiscard]] static std::unique_ptr<GsVulkanRasterBackend>
    createWithExecutor(
        std::unique_ptr<IGsVulkanDrawExecutor> executor,
        const GsVulkanRasterBackendConfig &backendConfig,
        std::span<uint8_t> canonicalVram,
        DrawCallback softwareOracle,
        DrawCallback acceleratedCommit,
        std::string *error = nullptr,
        FeedbackSnapshotCallback feedbackSnapshot = {},
        DecodedPaletteCallback decodedPalette = {},
        AcceleratedBatchCommitCallback acceleratedBatchCommit = {},
        FeedbackSnapshotGenerationCallback
            feedbackSnapshotGeneration = {});

    ~GsVulkanRasterBackend() override;

    GsVulkanRasterBackend(const GsVulkanRasterBackend &) = delete;
    GsVulkanRasterBackend &operator=(
        const GsVulkanRasterBackend &) = delete;

    [[nodiscard]] bool setMode(GsRendererMode mode) noexcept;
    [[nodiscard]] GsRendererMode mode() const noexcept;

    [[nodiscard]] GsBackendDecision classify(
        const GsDrawCommand &command) const override;
    [[nodiscard]] GsHybridBatchPolicy hybridBatchPolicy(
        const GsDrawCommand &command) const noexcept override;
    [[nodiscard]] bool hybridBatchCompatible(
        const GsDrawCommand &first,
        const GsDrawCommand &next) const noexcept override;
    [[nodiscard]] bool canCaptureFeedbackSnapshotOnDevice(
        const GsDrawCommand &command) const noexcept override;
    void submit(std::span<const GsDrawCommand> commands) override;
    void flush(GsFlushReason reason) override;
    [[nodiscard]] size_t pendingCommandCount() const noexcept override;
    void prepareCpuVramAccess(
        const GsVramPageMask &pages,
        GsFlushReason reason) override;
    void prepareCpuVramAccess(
        const GsVramPageMask &readPages,
        const GsVramPageMask &writePages,
        GsFlushReason reason) override;
    void noteCpuVramWrite(const GsVramPageMask &pages) override;

    // Flushes pending resident work and copies the immutable device feedback
    // image without changing canonical CPU/GPU VRAM ownership.
    [[nodiscard]] bool materializeFeedbackSnapshot(
        std::span<uint8_t> destination,
        std::string *error = nullptr);

    [[nodiscard]] GsVulkanRasterBackendStatistics
    backendStatistics() const;
    [[nodiscard]] GsVulkanCapabilityReport capabilities() const;
    [[nodiscard]] GsVulkanServiceStatistics serviceStatistics() const;
    [[nodiscard]] bool healthy() const;

private:
    struct Impl;
    explicit GsVulkanRasterBackend(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> m_impl;
};
