#ifndef PS2_GS_RASTERIZER_H
#define PS2_GS_RASTERIZER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

class GS;

class GSRasterizer
{
public:
    GSRasterizer();
    ~GSRasterizer();

    GSRasterizer(const GSRasterizer &) = delete;
    GSRasterizer &operator=(const GSRasterizer &) = delete;

    bool beginDrawBatch(GS *gs);
    void flushDrawBatch(GS *gs);
    void endDrawBatch(GS *gs);
    void drawPrimitive(GS *gs);
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
    class DebugProgressScope
    {
    public:
        DebugProgressScope(GSRasterizer &rasterizer, GS *owner);
        ~DebugProgressScope();

        DebugProgressScope(const DebugProgressScope &) = delete;
        DebugProgressScope &operator=(const DebugProgressScope &) = delete;

    private:
        GSRasterizer &m_rasterizer;
    };

    bool tryQueuePrimitive(GS *gs);
    void beginDebugProgress(GS *owner);
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
    std::unique_ptr<ParallelState> m_parallelState;
    uint32_t m_scanlineWorkerIndex = 0u;
    uint32_t m_scanlineWorkerCount = 1u;
    GS *m_debugProgressOwner = nullptr;
    uint64_t m_debugCandidatePixelBatch = 0u;
    bool m_trackDebugProgress = false;
};

#endif
