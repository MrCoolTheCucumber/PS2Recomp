#ifndef PS2_GS_RASTERIZER_H
#define PS2_GS_RASTERIZER_H

#include "runtime/ps2_gs_backend.h"
#include "runtime/ps2_gs_replay.h"
#include "runtime/ps2_gs_vulkan_backend.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class GS;

class GSRasterizer
{
public:
    explicit GSRasterizer(GS *owner = nullptr);
    ~GSRasterizer();

    GSRasterizer(const GSRasterizer &) = delete;
    GSRasterizer &operator=(const GSRasterizer &) = delete;

    bool beginDrawBatch(GS *gs);
    void flushDrawBatch(
        GS *gs,
        GsFlushReason reason = GsFlushReason::Explicit);
    void beginCpuVramAccess(
        GS *gs,
        const GsVramPageMask &readPages,
        const GsVramPageMask &writePages,
        GsFlushReason reason);
    void endCpuVramAccess(
        GS *gs,
        const GsVramPageMask &writePages);
    void endDrawBatch(GS *gs);
    void drawPrimitive(GS *gs);
    [[nodiscard]] bool configureVulkanRenderer(
        const GsVulkanServiceConfig &config,
        std::string verificationArtifactDirectory = {});
    [[nodiscard]] bool configureVulkanRenderer(
        const GsVulkanServiceConfig &config,
        GsVulkanRasterBackendConfig backendConfig);
    [[nodiscard]] bool setRendererMode(GsRendererMode mode);
    [[nodiscard]] GsRendererMode rendererMode() const noexcept;
    [[nodiscard]] std::string rendererDiagnostic() const;
    [[nodiscard]] GsVulkanCapabilityReport
    vulkanRendererCapabilities() const;
    [[nodiscard]] GsVulkanServiceStatistics
    vulkanRendererServiceStatistics() const;
    [[nodiscard]] GsVulkanRasterBackendStatistics
    vulkanRendererBackendStatistics() const;
    void setBackendCountersEnabled(bool enabled) noexcept;
    [[nodiscard]] GsBackendCounters backendCounters() const noexcept;
    void resetBackendCounters() noexcept;
    [[nodiscard]] GsReplayRasterizerState captureReplayState() const;
    [[nodiscard]] bool restoreReplayState(
        const GsReplayRasterizerState &state,
        uint32_t vramSize);
    void writePixel(GS *gs,
                    int x,
                    int y,
                    int z,
                    uint8_t r,
                    uint8_t g,
                    uint8_t b,
                    uint8_t a,
                    bool forceAlphaBlend = false,
                    bool suppressDepthWrite = false);
    uint32_t sampleTexture(GS *gs, float s, float t, float q, uint16_t u, uint16_t v);
    uint32_t lookupCLUT(GS *gs, uint8_t index, uint32_t cbp, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm);
    void updateClutCache(GS *gs, int contextIndex);

private:
    struct ParallelState;
    struct BackendState;
    class DebugProgressScope
    {
    public:
        DebugProgressScope(GSRasterizer &rasterizer,
                           GS *owner,
                           uint64_t drawCount = 1u);
        ~DebugProgressScope();

        DebugProgressScope(const DebugProgressScope &) = delete;
        DebugProgressScope &operator=(const DebugProgressScope &) = delete;

    private:
        GSRasterizer &m_rasterizer;
    };

    bool tryQueuePrimitive(GS *gs, const GsDrawCommand &command);
    void prepareFeedbackSnapshot(
        GS *gs,
        const GsDrawCommand &command);
    void submitSoftwareCommand(GS *gs, const GsDrawCommand &command);
    void recordAcceleratedCommit(
        GS *gs,
        const GsDrawCommand &command);
    void flushSoftwareDrawBatch(GS *gs);
    [[nodiscard]] size_t softwarePendingCommandCount() const noexcept;
    void renderSoftwarePrimitive(GS *gs, const GsDrawCommand &command);
    void beginDebugProgress(GS *owner, uint64_t drawCount);
    void endDebugProgress();
    void renderQueuedPrimitive(GS *renderGs,
                               size_t commandIndex,
                               uint32_t workerIndex,
                               uint32_t workerCount);
    bool ownsScanline(int y) const;
    uint32_t sampleTextureFixed(GS *gs,
                                int32_t fixedU,
                                int32_t fixedV,
                                bool linearBiasApplied,
                                float lodQ);
    uint32_t sampleTextureLinearLevel(GS *gs,
                                      int32_t fixedU,
                                      int32_t fixedV,
                                      bool linearBiasApplied,
                                      uint8_t level);
    void prepareDecodedClut(GS *gs);
    void drawSprite(GS *gs);
    void drawTriangle(GS *gs);
    void drawLine(GS *gs);

    std::vector<uint8_t> m_textureSnapshot;
    const uint8_t *m_textureReadVram = nullptr;
    uint32_t m_feedbackTextureBase = 0;
    uint32_t m_feedbackFrameBase = 0;
    uint8_t m_feedbackTexturePsm = 0;
    uint8_t m_feedbackFramePsm = 0;
    uint8_t m_feedbackTextureWidth = 0;
    uint8_t m_feedbackFrameWidth = 0;
    bool m_feedbackSnapshotValid = false;
    std::array<uint32_t, 256> m_decodedClut{};
    uint64_t m_decodedClutGeneration = UINT64_MAX;
    uint16_t m_decodedClutTexa = 0u;
    uint8_t m_decodedClutSourcePsm = 0u;
    uint8_t m_decodedClutCsm = 0u;
    uint8_t m_decodedClutCsa = 0u;
    bool m_decodedClutActive = false;
    uint64_t m_queuedPaletteSerial = UINT64_MAX;
    int32_t m_queuedFixedX[3]{};
    int32_t m_queuedFixedY[3]{};
    bool m_queuedFixedVerticesValid = false;
    GS *m_owner = nullptr;
    std::unique_ptr<ParallelState> m_parallelState;
    std::unique_ptr<BackendState> m_backendState;
    uint32_t m_scanlineWorkerIndex = 0u;
    uint32_t m_scanlineWorkerCount = 1u;
    GS *m_debugProgressOwner = nullptr;
    uint64_t m_debugCandidatePixelBatch = 0u;
    uint64_t m_debugDrawCount = 0u;
    uint32_t m_debugActiveDrawCount = 0u;
    bool m_trackDebugProgress = false;
    std::atomic<bool> m_backendTimingEnabled{false};
    std::atomic<uint64_t> m_softwareRasterHostNanoseconds{0u};
};

#endif
