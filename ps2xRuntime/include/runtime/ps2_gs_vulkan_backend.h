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
    // Nearest texture work is the exact clipped sample/write rectangle.
    uint64_t minimumHybridNearestCt32SpritePixels = 8'192u;
    // Linear texture work is the exact clipped sample/write rectangle. Zero
    // disables the policy; tiled execution preserves the binary32 V DDA.
    uint64_t minimumHybridLinearCt32SpritePixels = 131'072u;
    // Triangle work is the conservative clipped bounding box dispatched by the
    // compute kernel, including samples rejected by its exact edge tests.
    uint64_t minimumHybridTriangleCandidatePixels = 32'768u;
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
    GsVramCoherencySummary pageOwnership{};
    GsVramCoherencyStatistics coherency{};
    std::string lastVerificationArtifact;
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

    [[nodiscard]] static std::unique_ptr<GsVulkanRasterBackend> create(
        const GsVulkanServiceConfig &serviceConfig,
        const GsVulkanRasterBackendConfig &backendConfig,
        std::span<uint8_t> canonicalVram,
        DrawCallback softwareOracle,
        DrawCallback acceleratedCommit,
        GsVulkanCapabilityReport *report = nullptr,
        std::string *error = nullptr);

    // Dependency-injected construction keeps mismatch, execution-failure, and
    // lifecycle tests deterministic even on compiled-out or GPU-less hosts.
    [[nodiscard]] static std::unique_ptr<GsVulkanRasterBackend>
    createWithExecutor(
        std::unique_ptr<IGsVulkanDrawExecutor> executor,
        const GsVulkanRasterBackendConfig &backendConfig,
        std::span<uint8_t> canonicalVram,
        DrawCallback softwareOracle,
        DrawCallback acceleratedCommit,
        std::string *error = nullptr);

    ~GsVulkanRasterBackend() override;

    GsVulkanRasterBackend(const GsVulkanRasterBackend &) = delete;
    GsVulkanRasterBackend &operator=(
        const GsVulkanRasterBackend &) = delete;

    [[nodiscard]] bool setMode(GsRendererMode mode) noexcept;
    [[nodiscard]] GsRendererMode mode() const noexcept;

    [[nodiscard]] GsBackendDecision classify(
        const GsDrawCommand &command) const override;
    void submit(std::span<const GsDrawCommand> commands) override;
    void flush(GsFlushReason reason) override;
    [[nodiscard]] size_t pendingCommandCount() const noexcept override;
    void prepareCpuVramAccess(
        const GsVramPageMask &pages,
        GsFlushReason reason) override;
    void noteCpuVramWrite(const GsVramPageMask &pages) override;

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
