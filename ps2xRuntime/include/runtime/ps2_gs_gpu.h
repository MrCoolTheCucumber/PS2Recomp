#ifndef PS2_GS_GPU_H
#define PS2_GS_GPU_H

#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <vector>

#include "ps2_gs_types.h"
#include "ps2_gs_rasterizer.h"
#include "ps2_gs_replay.h"
#include "ps2_gs_memory.h"

struct GSDebugSnapshot
{
    GSContext ctx[2]{};
    GSPrimReg prim{};
    GSTexaReg texa{};
    GSTexClutReg texclut{};
    uint32_t fogColor = 0;
    bool prmodecont = true;
    bool pabe = false;
    GSBitBltBuf bitbltbuf{};
    GSTrxPos trxpos{};
    GSTrxReg trxreg{};
    uint32_t trxdir = 0;
    uint32_t transferX = 0;
    uint32_t transferY = 0;
    uint32_t transferTotalPixels = 0;
    uint32_t transferCopiedPixels = 0;
    uint32_t lastDisplayBaseBytes = 0;
    GSFrameReg preferredDisplaySourceFrame{};
    uint32_t preferredDisplayDestFbp = 0;
    bool hasPreferredDisplaySource = false;
    uint32_t hostPresentationWidth = 0;
    uint32_t hostPresentationHeight = 0;
    uint32_t hostPresentationDisplayFbp = 0;
    uint32_t hostPresentationSourceFbp = 0;
    bool hostPresentationUsedPreferred = false;
    bool hasHostPresentationFrame = false;
    size_t localToHostPendingBytes = 0;
};

struct GSProgressSnapshot
{
    bool enabled = false;
    uint64_t drawsStarted = 0;
    uint64_t drawsCompleted = 0;
    uint64_t candidatePixels = 0;
    uint64_t presentations = 0;
    uint32_t activeDraws = 0;
    uint32_t activePrimitive = 0;
};

enum class GSDebugEventKind : uint8_t
{
    GifTag = 0,
    Register = 1,
    Draw = 2,
    Transfer = 3,
    Present = 4,
};

struct GSDebugHistoryEntry
{
    uint64_t seq = 0;
    uint64_t vsyncTick = 0;
    uint32_t frameIndex = 0;
    GSDebugEventKind kind = GSDebugEventKind::Register;

    uint8_t reg = 0;
    uint64_t regValue = 0;

    uint32_t gifSizeBytes = 0;
    uint32_t gifNloop = 0;
    uint8_t gifFlg = 0;
    uint8_t gifNreg = 0;

    GSPrimReg prim{};
    GSFrameReg frame{};
    GSZbufReg zbuf{};
    GSTex0Reg tex0{};
    GSScissorReg scissor{};
    uint64_t test = 0;
    uint64_t alpha = 0;

    uint32_t vertexCount = 0;
    float xMin = 0.0f;
    float xMax = 0.0f;
    float yMin = 0.0f;
    float yMax = 0.0f;
    double zMin = 0.0;
    double zMax = 0.0;
    uint8_t aMin = 0;
    uint8_t aMax = 0;

    GSBitBltBuf bitbltbuf{};
    GSTrxPos trxpos{};
    GSTrxReg trxreg{};
    uint32_t trxdir = 0;
    uint32_t transferPixels = 0;

    uint32_t displayFbp = 0;
    uint32_t sourceFbp = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    bool usedPreferred = false;
};

class GSRasterizer;

class GS
{
    friend class GSRasterizer;

public:
    GS();
    ~GS();

    void init(uint8_t *vram, uint32_t vramSize, struct GSRegisters *privRegs = nullptr);
    void reset();
    void setVSyncTickProvider(
        std::function<uint64_t()> provider);

    void processGIFPacket(const uint8_t *data, uint32_t sizeBytes);
    bool processNativePackedGIFPacket(const uint8_t *data, uint32_t sizeBytes);
    void beginRenderBatch();
    void flushRenderBatch();
    void endRenderBatch();
    void uploadImageNative(uint64_t bitbltbuf,
                           uint64_t trxpos,
                           uint64_t trxreg,
                           uint64_t trxdir,
                           const uint8_t *data,
                           uint32_t sizeBytes);
    void writeRegister(uint8_t regAddr, uint64_t value);

    const uint8_t *lockDisplaySnapshot(uint32_t &outSize);
    void unlockDisplaySnapshot();
    uint32_t getLastDisplayBaseBytes() const;
    const GSFrameReg &getContextFrame(int index) const
    {
        return m_ctx[(index != 0) ? 1 : 0].frame;
    }
    GSDebugSnapshot getDebugSnapshot() const;
    GSProgressSnapshot getProgressSnapshot() const;
    void setProgressTrackingEnabled(bool enabled);
    std::vector<GSDebugHistoryEntry> getDebugHistory() const;
    void copyRecentGifPackets(size_t limit,
                              std::vector<uint8_t> &outStream,
                              std::vector<uint32_t> &outSizes,
                              std::vector<uint8_t> &outInitialVram,
                              GsReplayState *outInitialState) const;
    [[nodiscard]] GsReplayState captureReplayState() const;
    [[nodiscard]] bool restoreReplayState(
        const GsReplayState &state);
    void clearDebugHistory();
    bool isDebugHistoryPaused() const;
    void setDebugHistoryPaused(bool paused);
    bool getPreferredDisplaySource(GSFrameReg &outSource, uint32_t &outDestFbp) const;
    void latchHostPresentationFrame();
    bool copyLatchedHostPresentationFrame(std::vector<uint8_t> &outPixels,
                                          uint32_t &outWidth,
                                          uint32_t &outHeight,
                                          uint32_t *outDisplayFbp = nullptr,
                                          uint32_t *outSourceFbp = nullptr,
                                          bool *outUsedPreferred = nullptr) const;
    bool clearFramebufferContext(uint32_t contextIndex, uint32_t rgba);
    bool clearActiveFramebuffer(uint32_t rgba);
    uint64_t nativeImageUploadCount() const { return m_nativeImageUploadCount; }
    uint64_t nativePackedGIFPacketCount() const { return m_nativePackedGIFPacketCount; }
    [[nodiscard]] bool configureVulkanRenderer(
        const GsVulkanServiceConfig &config,
        std::string verificationArtifactDirectory = {});
    [[nodiscard]] bool configureVulkanRenderer(
        const GsVulkanServiceConfig &config,
        GsVulkanRasterBackendConfig backendConfig);
    [[nodiscard]] bool setRendererMode(GsRendererMode mode);
    [[nodiscard]] GsRendererMode rendererMode() const;
    [[nodiscard]] std::string rendererDiagnostic() const;
    [[nodiscard]] GsVulkanCapabilityReport
    vulkanRendererCapabilities() const;
    [[nodiscard]] GsVulkanServiceStatistics
    vulkanRendererServiceStatistics() const;
    [[nodiscard]] GsVulkanRasterBackendStatistics
    vulkanRendererBackendStatistics() const;
    void setBackendCountersEnabled(bool enabled);
    [[nodiscard]] GsBackendCounters backendCounters() const;
    void resetBackendCounters();
    // Replay control: pause GIF decoding immediately after this many
    // draw commands have been submitted. A limit of zero pauses before input.
    void setDrawCommandLimit(uint64_t maximumCommands);
    void clearDrawCommandLimit();
    [[nodiscard]] bool drawCommandLimitReached() const;
    [[nodiscard]] uint64_t submittedDrawCommandCount() const;

    uint32_t consumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes);

    void refreshDisplaySnapshot();

    inline void WriteVram(u32 psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);
    inline u32 ReadVram(u32 psm, u32 base, u32 bw, u32 x, u32 y) const;
    inline u32 ReadVramFrom(const u8 *vram,
                            u32 psm,
                            u32 base,
                            u32 bw,
                            u32 x,
                            u32 y) const;

private:
    void snapshotVRAM();
    void writeRegisterPacked(uint8_t regDesc, uint64_t lo, uint64_t hi);
    void vertexKick(bool drawing);
    void latchHostPresentationFrameUnlocked();
    uint64_t currentVSyncTickUnlocked() const;
    void flushForObservation(GsFlushReason reason) const;

    void recordDebugEventUnlocked(GSDebugHistoryEntry entry);
    GSDebugHistoryEntry makeDebugEventUnlocked(GSDebugEventKind kind) const;
    void recordGifTagDebugEventUnlocked(uint32_t sizeBytes, uint32_t nloop, uint8_t flg, uint32_t nreg);
    void recordRegisterDebugEventUnlocked(uint8_t regAddr, uint64_t value);
    void recordDrawDebugEventUnlocked(int vertexCount);
    void recordTransferDebugEventUnlocked();
    void recordPresentDebugEventUnlocked(uint32_t displayFbp, uint32_t sourceFbp, uint32_t width, uint32_t height, bool usedPreferred);

    void processImageData(const uint8_t *data, uint32_t sizeBytes);
    [[nodiscard]] GsReplayState captureReplayStateUnlocked() const;
    bool tryProcessNativeImageUploadPacket(const uint8_t *data, uint32_t sizeBytes);
    void performLocalToLocalTransfer();
    void performLocalToHostToBuffer();
    bool copyFrameToHostRgbaUnlocked(const GSFrameReg &frame,
                                     uint32_t width,
                                     uint32_t height,
                                     std::vector<uint8_t> &outPixels,
                                     bool preserveAlpha = false,
                                     bool useLocalMemoryLayout = false,
                                     bool frameBaseIsPages = true,
                                     uint32_t sourceOriginX = 0u,
                                     uint32_t sourceOriginY = 0u) const;

    GSContext &activeContext();

    uint8_t *m_vram = nullptr;
    uint32_t m_vramSize = 0;
    struct GSRegisters *m_privRegs = nullptr;
    mutable std::recursive_mutex m_stateMutex;
    std::function<uint64_t()> m_vsyncTickProvider;

    GSContext m_ctx[2];
    GSPrimReg m_prim{};

    uint8_t m_curR = 0x80, m_curG = 0x80, m_curB = 0x80, m_curA = 0x80;
    float m_curQ = 1.0f;
    float m_curS = 0.0f, m_curT = 0.0f;
    uint16_t m_curU = 0, m_curV = 0;
    uint8_t m_curFog = 0;
    uint32_t m_fogColor = 0;

    bool m_prmodecont = true;
    bool m_pabe = false;
    uint8_t m_scanMask = 0;
    uint64_t m_dimx = 0;
    bool m_dither = false;
    bool m_colorClamp = false;
    GSTexaReg m_texa{0u, false, 0u};
    GSTexClutReg m_texclut{0u, 0u, 0u};
    uint32_t m_clutCache[256]{};
    uint8_t m_clutCacheFormat[256]{};
    bool m_clutCacheValid[256]{};
    uint32_t m_clutCbp[2]{};
    uint64_t m_clutCacheGeneration = 0u;

    GSBitBltBuf m_bitbltbuf{};
    GSTrxPos m_trxpos{};
    GSTrxReg m_trxreg{};
    uint32_t m_trxdir = 3;

    struct
    {
        uint32_t x{ 0 };
        uint32_t y{ 0 };
        uint32_t total_pixels{ 0 };
        uint32_t copied_pixels{ 0 };
    } m_transferState;

    static constexpr int kMaxVerts = 6;
    static_assert(
        kMaxVerts == static_cast<int>(GS_REPLAY_VERTEX_QUEUE_CAPACITY));
    GSVertex m_vtxQueue[kMaxVerts];
    int m_vtxCount = 0;
    int m_vtxIndex = 0;
    uint64_t m_nextDrawSequence = 1;
    uint64_t m_drawCommandLimit = UINT64_MAX;
    bool m_drawCommandLimitReached = false;

    std::vector<uint8_t> m_displaySnapshot;
    std::mutex m_snapshotMutex;
    uint32_t m_lastDisplayBaseBytes = 0;
    GSFrameReg m_preferredDisplaySourceFrame{};
    uint32_t m_preferredDisplayDestFbp = 0;
    bool m_hasPreferredDisplaySource = false;
    std::vector<uint8_t> m_hostPresentationFrame;
    uint32_t m_hostPresentationWidth = 0;
    uint32_t m_hostPresentationHeight = 0;
    uint32_t m_hostPresentationDisplayFbp = 0;
    uint32_t m_hostPresentationSourceFbp = 0;
    bool m_hostPresentationUsedPreferred = false;
    bool m_hasHostPresentationFrame = false;
    uint64_t m_nativeImageUploadCount = 0;
    uint64_t m_nativePackedGIFPacketCount = 0;
    std::atomic<bool> m_progressTrackingEnabled{false};
    std::atomic<uint64_t> m_progressDrawsStarted{0};
    std::atomic<uint64_t> m_progressDrawsCompleted{0};
    std::atomic<uint64_t> m_progressCandidatePixels{0};
    std::atomic<uint64_t> m_progressPresentations{0};
    std::atomic<uint32_t> m_progressActiveDraws{0};
    std::atomic<uint32_t> m_progressActivePrimitive{0};

    std::vector<uint8_t> m_localToHostBuffer;
    size_t m_localToHostReadPos = 0;

    static constexpr size_t kDebugHistoryCapacity = 512;
    static constexpr size_t kDebugGifCaptureBytes = 16u * 1024u * 1024u;
    std::array<GSDebugHistoryEntry, kDebugHistoryCapacity> m_debugHistory{};
    std::deque<std::vector<uint8_t>> m_debugGifPackets;
    size_t m_debugGifPacketBytes = 0;
    std::vector<uint8_t> m_debugGifInitialVram;
    GsReplayState m_debugGifInitialState{};
    size_t m_debugHistoryWrite = 0;
    size_t m_debugHistoryCount = 0;
    uint64_t m_debugNextSeq = 1;
    uint32_t m_debugFrameIndex = 0;
    uint64_t m_debugLastVsyncTick = UINT64_MAX;
    bool m_debugHistoryPaused = true;

    GSRasterizer m_rasterizer;

    using WriteVramFunc =
        void (*)(u8*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using ReadVramFunc =
        u32 (*)(u8*, u32, u32, u32, u32);

    std::array<ReadVramFunc, 0x3F> m_read_vram_funcs{ };
    std::array<WriteVramFunc, 0x3F> m_write_vram_funcs{ };
};

inline u32 GS::ReadVram(u32 psm, u32 base, u32 bw, u32 x, u32 y) const
{
    return m_read_vram_funcs[psm & 0x3F](m_vram, base, bw, x, y);
}

inline u32 GS::ReadVramFrom(const u8 *vram,
                            u32 psm,
                            u32 base,
                            u32 bw,
                            u32 x,
                            u32 y) const
{
    return m_read_vram_funcs[psm & 0x3F](
        const_cast<u8 *>(vram), base, bw, x, y);
}

inline void GS::WriteVram(u32 psm, u32 base, u32 bw, u32 x, u32 y, u32 value)
{
    m_write_vram_funcs[psm & 0x3F](m_vram, base, bw, x, y, value);
}

#endif
