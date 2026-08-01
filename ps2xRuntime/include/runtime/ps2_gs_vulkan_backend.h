#pragma once

#include "runtime/ps2_gs_backend.h"
#include "runtime/ps2_gs_vulkan.h"

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
    std::string lastVerificationArtifact;
};

// Phase 3 backend: each command starts from canonical CPU VRAM and completes a
// synchronous whole-buffer GPU transaction. Verify mode executes the supplied
// software oracle only after the independent GPU result is complete, compares
// all 4 MiB, and leaves the software result canonical on agreement.
// The caller must keep the canonical VRAM address and extent stable for the
// backend's lifetime.
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
