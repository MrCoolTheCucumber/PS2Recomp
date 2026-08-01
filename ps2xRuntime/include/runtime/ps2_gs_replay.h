#pragma once

#include "runtime/ps2_gs_types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

inline constexpr uint32_t GS_REPLAY_STATE_VERSION = 1u;
inline constexpr size_t GS_REPLAY_VERTEX_QUEUE_CAPACITY = 6u;
inline constexpr size_t GS_REPLAY_CLUT_ENTRY_COUNT = 256u;
inline constexpr size_t GS_REPLAY_VRAM_SIZE = 4u * 1024u * 1024u;

struct GsReplayTransferState
{
    uint32_t x = 0u;
    uint32_t y = 0u;
    uint32_t totalPixels = 0u;
    uint32_t copiedPixels = 0u;
};

struct GsReplayRasterizerState
{
    std::vector<uint8_t> feedbackVram;
    uint32_t feedbackTextureBase = 0u;
    uint32_t feedbackFrameBase = 0u;
    uint8_t feedbackTexturePsm = 0u;
    uint8_t feedbackFramePsm = 0u;
    uint8_t feedbackTextureWidth = 0u;
    uint8_t feedbackFrameWidth = 0u;
    bool feedbackSnapshotValid = false;

    std::array<uint32_t, GS_REPLAY_CLUT_ENTRY_COUNT> decodedClut{};
    uint64_t decodedClutGeneration = UINT64_MAX;
    uint16_t decodedClutTexa = 0u;
    uint8_t decodedClutSourcePsm = 0u;
    uint8_t decodedClutCsm = 0u;
    uint8_t decodedClutCsa = 0u;
    bool decodedClutActive = false;
};

// Complete GS state needed to continue decoding GIF packets at a synchronized
// packet boundary. Diagnostic counters, debug history, and host presentation
// caches are deliberately excluded because they cannot affect future GS VRAM.
struct GsReplayState
{
    GSContext ctx[2]{};
    GSPrimReg prim{};

    uint8_t currentR = 0x80u;
    uint8_t currentG = 0x80u;
    uint8_t currentB = 0x80u;
    uint8_t currentA = 0x80u;
    float currentQ = 1.0f;
    float currentS = 0.0f;
    float currentT = 0.0f;
    uint16_t currentU = 0u;
    uint16_t currentV = 0u;
    uint8_t currentFog = 0u;

    uint32_t fogColor = 0u;
    bool prmodecont = true;
    bool pabe = false;
    uint8_t scanMask = 0u;
    uint64_t dimx = 0u;
    bool dither = false;
    bool colorClamp = false;
    GSTexaReg texa{};
    GSTexClutReg texclut{};

    std::array<uint32_t, GS_REPLAY_CLUT_ENTRY_COUNT> clutCache{};
    std::array<uint8_t, GS_REPLAY_CLUT_ENTRY_COUNT> clutCacheFormat{};
    std::array<uint8_t, GS_REPLAY_CLUT_ENTRY_COUNT> clutCacheValid{};
    std::array<uint32_t, 2> clutCbp{};
    uint64_t clutCacheGeneration = 0u;

    GSBitBltBuf bitbltbuf{};
    GSTrxPos trxpos{};
    GSTrxReg trxreg{};
    uint32_t trxdir = 3u;
    GsReplayTransferState transfer{};

    std::array<GSVertex, GS_REPLAY_VERTEX_QUEUE_CAPACITY> vertexQueue{};
    int32_t vertexCount = 0;

    GsReplayRasterizerState rasterizer{};
};

[[nodiscard]] bool encodeGsReplayState(
    const GsReplayState &state,
    std::vector<uint8_t> &output,
    std::string *error = nullptr);

[[nodiscard]] bool decodeGsReplayState(
    std::span<const uint8_t> input,
    GsReplayState &state,
    std::string *error = nullptr);
