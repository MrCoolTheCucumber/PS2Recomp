#pragma once

#include "runtime/ps2_gs_backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#ifndef PS2X_HAS_GS_VULKAN
#define PS2X_HAS_GS_VULKAN 0
#endif

inline constexpr size_t GS_VULKAN_VRAM_SIZE = 4u * 1024u * 1024u;
inline constexpr uint32_t GS_VULKAN_NOOP_LOCAL_SIZE = 64u;
inline constexpr uint32_t GS_VULKAN_NOOP_GROUP_COUNT =
    static_cast<uint32_t>(
        GS_VULKAN_VRAM_SIZE / sizeof(uint32_t) /
        GS_VULKAN_NOOP_LOCAL_SIZE);
inline constexpr uint32_t GS_VULKAN_MAX_MEMORY_CASES = 65536u;
inline constexpr size_t GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH = 64u;
inline constexpr size_t GS_VULKAN_MAX_RESIDENT_DEPTH_CT32_BATCH =
    GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH;
inline constexpr size_t GS_VULKAN_MAX_RESIDENT_NEAREST_CT32_BATCH =
    GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH;
inline constexpr size_t GS_VULKAN_MAX_RESIDENT_LINEAR_CT32_BATCH =
    GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH;
inline constexpr size_t GS_VULKAN_MAX_RESIDENT_TRIANGLE_BATCH =
    GS_VRAM_PAGE_COUNT;

enum class GsVulkanProbeStatus : uint8_t
{
    CompiledOut,
    LoaderUnavailable,
    LoaderInvalid,
    ValidationUnavailable,
    InstanceCreationFailed,
    DeviceEnumerationFailed,
    NoPhysicalDevices,
    NoSuitableDevice,
    DeviceCreationFailed,
    ResourceCreationFailed,
    ExecutionTimeout,
    DeviceLost,
    ValidationError,
    Ready,
};

enum class GsVulkanDeviceKind : uint8_t
{
    Other,
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
};

struct GsVulkanProbeConfig
{
    // Validation is opt-in for deterministic headless tools. The future live
    // backend enables it by default in development builds.
    bool enableValidation = false;

    // An explicit loader path is tried exclusively. This is useful both for
    // deployment overrides and for proving that a missing loader fails closed.
    std::string loaderPath;

    // Zero selects any suitable device. A non-zero value is a required match,
    // so a stale override cannot silently select a different GPU.
    uint32_t preferredVendorId = 0u;
    uint32_t preferredDeviceId = 0u;
};

struct GsVulkanDeviceReport
{
    std::string name;
    uint32_t vendorId = 0u;
    uint32_t deviceId = 0u;
    uint32_t apiVersion = 0u;
    uint32_t driverVersion = 0u;
    GsVulkanDeviceKind kind = GsVulkanDeviceKind::Other;

    uint64_t maxStorageBufferRange = 0u;
    uint32_t maxComputeWorkGroupCountX = 0u;
    uint32_t maxComputeWorkGroupInvocations = 0u;
    uint32_t maxComputeWorkGroupSizeX = 0u;
    uint32_t queueFamilyIndex = std::numeric_limits<uint32_t>::max();

    bool computeQueue = false;
    bool dedicatedComputeQueue = false;
    bool deviceLocalMemory = false;
    bool hostVisibleMemory = false;
    bool shaderInt16 = false;
    bool shaderInt64 = false;

    // Vulkan 1.0 compute shaders have exact 32-bit integer storage-buffer
    // access. This flag also requires the queue, limits, and memory classes
    // needed by the fixed 4 MiB raw-GS-buffer design.
    bool exactVramStorage = false;

    // The initial exact triangle kernel additionally requires native signed
    // 64-bit shader arithmetic for full-range GS 12.4 edge equations.
    bool exactCt32Triangle = false;

    // Exact flat CT32 plus Z32/Z24 depth uses only the permanent raw-VRAM
    // contract and 32-bit storage-buffer atomics.
    bool exactDepthCt32Sprite = false;

    // The first textured-sprite kernel needs only the permanent exact raw-VRAM
    // storage contract; the separate bit lets routing fail closed if that
    // semantic pipeline is unavailable on a future platform.
    bool exactNearestCt32Sprite = false;

    // The first exact linear pipeline uses the same raw-VRAM contract and a
    // 96-byte push-constant record, below Vulkan 1.0's required minimum.
    bool exactLinearCt32Sprite = false;
    bool suitable = false;
    std::string rejectionReason;
};

struct GsVulkanCapabilityReport
{
    GsVulkanProbeStatus status = GsVulkanProbeStatus::CompiledOut;
    bool compiled = false;
    bool loaderAvailable = false;
    uint32_t loaderApiVersion = 0u;
    bool validationRequested = false;
    bool validationLayerAvailable = false;
    bool debugUtilsAvailable = false;
    bool validationEnabled = false;
    uint32_t validationWarnings = 0u;
    uint32_t validationErrors = 0u;
    std::vector<GsVulkanDeviceReport> devices;
    int32_t selectedDeviceIndex = -1;
    std::string detail;

    [[nodiscard]] bool ready() const noexcept
    {
        return status == GsVulkanProbeStatus::Ready &&
               selectedDeviceIndex >= 0;
    }

    [[nodiscard]] const GsVulkanDeviceReport *selectedDevice() const noexcept
    {
        if (selectedDeviceIndex < 0 ||
            static_cast<size_t>(selectedDeviceIndex) >= devices.size())
        {
            return nullptr;
        }
        return &devices[static_cast<size_t>(selectedDeviceIndex)];
    }
};

#if defined(NDEBUG)
inline constexpr bool GS_VULKAN_DEFAULT_VALIDATION = false;
#else
inline constexpr bool GS_VULKAN_DEFAULT_VALIDATION = true;
#endif

struct GsVulkanServiceConfig
{
    GsVulkanServiceConfig()
    {
        probe.enableValidation = GS_VULKAN_DEFAULT_VALIDATION;
    }

    GsVulkanProbeConfig probe;
    uint64_t fenceTimeoutNanoseconds = 30'000'000'000ull;
};

struct GsVulkanServiceStatistics
{
    uint64_t roundTripsCompleted = 0u;
    uint64_t roundTripsFailed = 0u;
    uint64_t queueSubmissions = 0u;
    uint64_t shaderDispatches = 0u;
    uint64_t pipelineBarriers = 0u;
    uint64_t pipelineBinds = 0u;
    uint64_t pipelineCacheHits = 0u;
    uint64_t pipelineCacheMisses = 0u;
    uint64_t bytesUploaded = 0u;
    uint64_t bytesDownloaded = 0u;
    uint64_t fenceWaits = 0u;
    uint64_t fenceWaitNanoseconds = 0u;
    uint64_t fenceTimeouts = 0u;
    uint64_t memoryBatchesCompleted = 0u;
    uint64_t memoryBatchesFailed = 0u;
    uint64_t memoryCasesExecuted = 0u;
    uint64_t spriteDrawsCompleted = 0u;
    uint64_t spriteDrawsFailed = 0u;
    uint64_t spritePixelsExecuted = 0u;
    uint64_t depthCt32SpriteDrawsCompleted = 0u;
    uint64_t depthCt32SpriteDrawsFailed = 0u;
    uint64_t depthCt32SpritePixelsExecuted = 0u;
    uint64_t residentDepthCt32SpriteBatchesCompleted = 0u;
    uint64_t residentDepthCt32SpriteBatchesFailed = 0u;
    uint64_t largestResidentDepthCt32SpriteBatch = 0u;
    uint64_t nearestCt32SpriteDrawsCompleted = 0u;
    uint64_t nearestCt32SpriteDrawsFailed = 0u;
    uint64_t nearestCt32SpritePixelsExecuted = 0u;
    uint64_t linearCt32SpriteDrawsCompleted = 0u;
    uint64_t linearCt32SpriteDrawsFailed = 0u;
    uint64_t linearCt32SpritePixelsExecuted = 0u;
    uint64_t residentLinearCt32SpriteBatchesCompleted = 0u;
    uint64_t residentLinearCt32SpriteBatchesFailed = 0u;
    uint64_t largestResidentLinearCt32SpriteBatch = 0u;
    uint64_t residentNearestCt32SpriteBatchesCompleted = 0u;
    uint64_t residentNearestCt32SpriteBatchesFailed = 0u;
    uint64_t largestResidentNearestCt32SpriteBatch = 0u;
    uint64_t triangleDrawsCompleted = 0u;
    uint64_t triangleDrawsFailed = 0u;
    uint64_t triangleCandidatePixelsExecuted = 0u;
    uint64_t residentSpriteBatchesCompleted = 0u;
    uint64_t residentSpriteBatchesFailed = 0u;
    uint64_t largestResidentSpriteBatch = 0u;
    uint64_t residentTriangleBatchesCompleted = 0u;
    uint64_t residentTriangleBatchesFailed = 0u;
    uint64_t largestResidentTriangleBatch = 0u;
    uint64_t pageUploadOperationsCompleted = 0u;
    uint64_t pageUploadOperationsFailed = 0u;
    uint64_t pageDownloadOperationsCompleted = 0u;
    uint64_t pageDownloadOperationsFailed = 0u;
    uint64_t pagesUploaded = 0u;
    uint64_t pagesDownloaded = 0u;
    uint64_t pageUploadRegions = 0u;
    uint64_t pageDownloadRegions = 0u;
    uint32_t validationWarnings = 0u;
    uint32_t validationErrors = 0u;
    bool deviceLost = false;
};

enum class GsVulkanMemoryOperation : uint32_t
{
    Read = 0u,
    Write = 1u,
};

// Fixed std430 records used by the address/value conformance kernel. The same
// helpers are included by later draw shaders, so this is a permanent backend
// diagnostic rather than a separate test implementation.
struct alignas(16) GsVulkanMemoryCase
{
    uint32_t pixelStorageMode = 0u;
    uint32_t baseBlock = 0u;
    uint32_t bufferWidth = 0u;
    uint32_t x = 0u;
    uint32_t y = 0u;
    GsVulkanMemoryOperation operation = GsVulkanMemoryOperation::Read;
    uint32_t value = 0u;
    uint32_t reserved = 0u;

    bool operator==(const GsVulkanMemoryCase &) const = default;
};

struct alignas(16) GsVulkanMemoryResult
{
    uint32_t wordIndex = 0u;
    uint32_t bitShift = 0u;
    uint32_t valueBefore = 0u;
    uint32_t valueAfter = 0u;

    bool operator==(const GsVulkanMemoryResult &) const = default;
};

static_assert(sizeof(GsVulkanMemoryCase) == 32u);
static_assert(sizeof(GsVulkanMemoryResult) == 16u);
static_assert(std::is_standard_layout_v<GsVulkanMemoryCase>);
static_assert(std::is_trivially_copyable_v<GsVulkanMemoryCase>);
static_assert(offsetof(GsVulkanMemoryCase, pixelStorageMode) == 0u);
static_assert(offsetof(GsVulkanMemoryCase, baseBlock) == 4u);
static_assert(offsetof(GsVulkanMemoryCase, bufferWidth) == 8u);
static_assert(offsetof(GsVulkanMemoryCase, x) == 12u);
static_assert(offsetof(GsVulkanMemoryCase, y) == 16u);
static_assert(offsetof(GsVulkanMemoryCase, operation) == 20u);
static_assert(offsetof(GsVulkanMemoryCase, value) == 24u);
static_assert(offsetof(GsVulkanMemoryCase, reserved) == 28u);
static_assert(std::is_standard_layout_v<GsVulkanMemoryResult>);
static_assert(std::is_trivially_copyable_v<GsVulkanMemoryResult>);
static_assert(offsetof(GsVulkanMemoryResult, wordIndex) == 0u);
static_assert(offsetof(GsVulkanMemoryResult, bitShift) == 4u);
static_assert(offsetof(GsVulkanMemoryResult, valueBefore) == 8u);
static_assert(offsetof(GsVulkanMemoryResult, valueAfter) == 12u);

// Fixed push-constant ABI for Phase 3's first exact draw kernel. Bounds are
// inclusive minimum and exclusive maximum framebuffer coordinates after the
// backend-neutral command builder has applied XYOFFSET, endpoint, and scissor
// rules. The FRAME base is expressed in 256-byte GS blocks.
struct alignas(16) GsVulkanCt32Sprite
{
    uint32_t framebufferBaseBlock = 0u;
    uint32_t framebufferWidth = 0u;
    uint32_t x0 = 0u;
    uint32_t y0 = 0u;
    uint32_t x1 = 0u;
    uint32_t y1 = 0u;
    uint32_t rgba = 0u;
    uint32_t reserved = 0u;

    bool operator==(const GsVulkanCt32Sprite &) const = default;
};

static_assert(sizeof(GsVulkanCt32Sprite) == 32u);
static_assert(std::is_standard_layout_v<GsVulkanCt32Sprite>);
static_assert(std::is_trivially_copyable_v<GsVulkanCt32Sprite>);
static_assert(offsetof(GsVulkanCt32Sprite, framebufferBaseBlock) == 0u);
static_assert(offsetof(GsVulkanCt32Sprite, framebufferWidth) == 4u);
static_assert(offsetof(GsVulkanCt32Sprite, x0) == 8u);
static_assert(offsetof(GsVulkanCt32Sprite, y0) == 12u);
static_assert(offsetof(GsVulkanCt32Sprite, x1) == 16u);
static_assert(offsetof(GsVulkanCt32Sprite, y1) == 20u);
static_assert(offsetof(GsVulkanCt32Sprite, rgba) == 24u);
static_assert(offsetof(GsVulkanCt32Sprite, reserved) == 28u);

// Applies the canonical eligibility predicate before publishing a shader ABI
// record. Rejected commands leave the caller's record untouched.
[[nodiscard]] GsBackendDecision prepareGsVulkanCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanCt32Sprite &sprite) noexcept;

// Fixed record for exact flat CT32 color plus Z32/Z24 depth execution. The
// depth method uses GS ZTST values (ALWAYS=1, GEQUAL=2, GREATER=3); depthWrite
// is normalized from ZMASK. Z24 writes preserve the unrelated high byte.
struct alignas(16) GsVulkanDepthCt32Sprite
{
    uint32_t framebufferBaseBlock = 0u;
    uint32_t framebufferWidth = 0u;
    uint32_t depthBaseBlock = 0u;
    uint32_t depthPsm = 0u;
    uint32_t boundsX0 = 0u;
    uint32_t boundsY0 = 0u;
    uint32_t boundsX1 = 0u;
    uint32_t boundsY1 = 0u;
    uint32_t rgba = 0u;
    uint32_t depth = 0u;
    uint32_t depthTestMethod = 0u;
    uint32_t depthWrite = 0u;
    uint32_t reserved0 = 0u;
    uint32_t reserved1 = 0u;
    uint32_t reserved2 = 0u;
    uint32_t reserved3 = 0u;

    bool operator==(const GsVulkanDepthCt32Sprite &) const = default;
};

static_assert(sizeof(GsVulkanDepthCt32Sprite) == 64u);
static_assert(std::is_standard_layout_v<GsVulkanDepthCt32Sprite>);
static_assert(std::is_trivially_copyable_v<GsVulkanDepthCt32Sprite>);
static_assert(offsetof(GsVulkanDepthCt32Sprite, framebufferBaseBlock) == 0u);
static_assert(offsetof(GsVulkanDepthCt32Sprite, framebufferWidth) == 4u);
static_assert(offsetof(GsVulkanDepthCt32Sprite, depthBaseBlock) == 8u);
static_assert(offsetof(GsVulkanDepthCt32Sprite, depthPsm) == 12u);
static_assert(offsetof(GsVulkanDepthCt32Sprite, boundsX0) == 16u);
static_assert(offsetof(GsVulkanDepthCt32Sprite, boundsY0) == 20u);
static_assert(offsetof(GsVulkanDepthCt32Sprite, boundsX1) == 24u);
static_assert(offsetof(GsVulkanDepthCt32Sprite, boundsY1) == 28u);
static_assert(offsetof(GsVulkanDepthCt32Sprite, rgba) == 32u);
static_assert(offsetof(GsVulkanDepthCt32Sprite, depth) == 36u);
static_assert(offsetof(GsVulkanDepthCt32Sprite, depthTestMethod) == 40u);
static_assert(offsetof(GsVulkanDepthCt32Sprite, depthWrite) == 44u);
static_assert(offsetof(GsVulkanDepthCt32Sprite, reserved0) == 48u);
static_assert(offsetof(GsVulkanDepthCt32Sprite, reserved3) == 60u);

// Rejection leaves the caller's record untouched.
[[nodiscard]] GsBackendDecision prepareGsVulkanDepthCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanDepthCt32Sprite &sprite) noexcept;

// Fixed record for Phase 6's first textured-sprite semantic slice. Texture
// coordinates name the texel sampled at boundsX0/boundsY0 and advance by one
// signed texel per output pixel. Power-of-two masks and packed per-axis wrap
// descriptors implement all four GS wrap modes before raw CT32 local-memory
// addressing. Each descriptor stores the two-bit mode followed by ten-bit MIN
// and MAX values. Source and destination are guaranteed disjoint by the
// backend-neutral classifier.
inline constexpr uint32_t GS_VULKAN_TEXTURE_WRAP_DESCRIPTOR_MASK =
    0x003FFFFFu;

inline constexpr uint32_t packGsVulkanTextureWrap(
    uint8_t mode,
    uint16_t regionMin,
    uint16_t regionMax) noexcept
{
    return
        (static_cast<uint32_t>(mode) & 0x3u) |
        ((static_cast<uint32_t>(regionMin) & 0x3FFu) << 2u) |
        ((static_cast<uint32_t>(regionMax) & 0x3FFu) << 12u);
}

inline constexpr uint8_t gsVulkanTextureWrapMode(
    uint32_t descriptor) noexcept
{
    return static_cast<uint8_t>(descriptor & 0x3u);
}

inline constexpr uint16_t gsVulkanTextureRegionMin(
    uint32_t descriptor) noexcept
{
    return static_cast<uint16_t>((descriptor >> 2u) & 0x3FFu);
}

inline constexpr uint16_t gsVulkanTextureRegionMax(
    uint32_t descriptor) noexcept
{
    return static_cast<uint16_t>((descriptor >> 12u) & 0x3FFu);
}

struct alignas(16) GsVulkanNearestCt32Sprite
{
    uint32_t framebufferBaseBlock = 0u;
    uint32_t framebufferWidth = 0u;
    uint32_t boundsX0 = 0u;
    uint32_t boundsY0 = 0u;
    uint32_t boundsX1 = 0u;
    uint32_t boundsY1 = 0u;
    uint32_t textureBaseBlock = 0u;
    uint32_t textureWidth = 0u;
    uint32_t textureMaskU = 0u;
    uint32_t textureMaskV = 0u;
    int32_t textureOriginU = 0;
    int32_t textureOriginV = 0;
    int32_t textureStepU = 0;
    int32_t textureStepV = 0;
    uint32_t textureWrapU = 0u;
    uint32_t textureWrapV = 0u;

    bool operator==(const GsVulkanNearestCt32Sprite &) const = default;
};

static_assert(sizeof(GsVulkanNearestCt32Sprite) == 64u);
static_assert(std::is_standard_layout_v<GsVulkanNearestCt32Sprite>);
static_assert(std::is_trivially_copyable_v<GsVulkanNearestCt32Sprite>);
static_assert(offsetof(GsVulkanNearestCt32Sprite, framebufferBaseBlock) == 0u);
static_assert(offsetof(GsVulkanNearestCt32Sprite, framebufferWidth) == 4u);
static_assert(offsetof(GsVulkanNearestCt32Sprite, boundsX0) == 8u);
static_assert(offsetof(GsVulkanNearestCt32Sprite, boundsY0) == 12u);
static_assert(offsetof(GsVulkanNearestCt32Sprite, boundsX1) == 16u);
static_assert(offsetof(GsVulkanNearestCt32Sprite, boundsY1) == 20u);
static_assert(offsetof(GsVulkanNearestCt32Sprite, textureBaseBlock) == 24u);
static_assert(offsetof(GsVulkanNearestCt32Sprite, textureWidth) == 28u);
static_assert(offsetof(GsVulkanNearestCt32Sprite, textureMaskU) == 32u);
static_assert(offsetof(GsVulkanNearestCt32Sprite, textureMaskV) == 36u);
static_assert(offsetof(GsVulkanNearestCt32Sprite, textureOriginU) == 40u);
static_assert(offsetof(GsVulkanNearestCt32Sprite, textureOriginV) == 44u);
static_assert(offsetof(GsVulkanNearestCt32Sprite, textureStepU) == 48u);
static_assert(offsetof(GsVulkanNearestCt32Sprite, textureStepV) == 52u);
static_assert(offsetof(GsVulkanNearestCt32Sprite, textureWrapU) == 56u);
static_assert(offsetof(GsVulkanNearestCt32Sprite, textureWrapV) == 60u);

// Publishes only fully validated records. Rejection leaves the caller's
// record untouched, matching the established sprite/triangle preparation
// contract.
[[nodiscard]] GsBackendDecision prepareGsVulkanNearestCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanNearestCt32Sprite &sprite) noexcept;

// Fixed prepared-DDA record for the first exact linear-filtered texture
// slice. U retains the software GS eight-lane setup verbatim. V retains the
// binary32 seed and step used by its sequential scanline recurrence; the bit
// representation keeps the executor ABI independent of host float layout
// spelling. The preparation contract admits REPEAT or standard CLAMP on each
// axis; execution capabilities may qualify those modes incrementally.
struct alignas(16) GsVulkanLinearCt32Sprite
{
    uint32_t framebufferBaseBlock = 0u;
    uint32_t framebufferWidth = 0u;
    uint32_t boundsX0 = 0u;
    uint32_t boundsY0 = 0u;
    uint32_t boundsX1 = 0u;
    uint32_t boundsY1 = 0u;
    uint32_t textureBaseBlock = 0u;
    uint32_t textureWidth = 0u;
    uint32_t textureMaskU = 0u;
    uint32_t textureMaskV = 0u;
    int32_t fixedBaseU = 0;
    int32_t fixedBlockStepU = 0;
    std::array<int32_t, 8> fixedLaneU{};
    uint32_t fixedScanVBits = 0u;
    uint32_t fixedStepVBits = 0u;
    uint32_t textureWrapU = 0u;
    uint32_t textureWrapV = 0u;

    bool operator==(const GsVulkanLinearCt32Sprite &) const = default;
};

static_assert(sizeof(GsVulkanLinearCt32Sprite) == 96u);
static_assert(std::is_standard_layout_v<GsVulkanLinearCt32Sprite>);
static_assert(std::is_trivially_copyable_v<GsVulkanLinearCt32Sprite>);
static_assert(offsetof(GsVulkanLinearCt32Sprite, framebufferBaseBlock) == 0u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, framebufferWidth) == 4u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, boundsX0) == 8u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, boundsY0) == 12u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, boundsX1) == 16u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, boundsY1) == 20u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, textureBaseBlock) == 24u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, textureWidth) == 28u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, textureMaskU) == 32u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, textureMaskV) == 36u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, fixedBaseU) == 40u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, fixedBlockStepU) == 44u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, fixedLaneU) == 48u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, fixedScanVBits) == 80u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, fixedStepVBits) == 84u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, textureWrapU) == 88u);
static_assert(offsetof(GsVulkanLinearCt32Sprite, textureWrapV) == 92u);

// Publishes only a fully validated repeat/clamp linear CT32 record. Rejected
// commands leave the caller's record untouched.
[[nodiscard]] GsBackendDecision prepareGsVulkanLinearCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanLinearCt32Sprite &sprite) noexcept;

// Phase 5's first exact triangle record. Vertex coordinates retain the signed
// 12.4 window-space values after XYOFFSET. Preparation normalizes the winding
// to positive area; topLeftEdgeMask bits 0..2 describe the edges opposite
// vertices 0..2 respectively. Bounds are inclusive minimum / exclusive
// maximum after scissor clipping.
struct alignas(16) GsVulkanCt32Triangle
{
    uint32_t framebufferBaseBlock = 0u;
    uint32_t framebufferWidth = 0u;
    uint32_t boundsX0 = 0u;
    uint32_t boundsY0 = 0u;
    uint32_t boundsX1 = 0u;
    uint32_t boundsY1 = 0u;
    int32_t vertex0X12_4 = 0;
    int32_t vertex0Y12_4 = 0;
    int32_t vertex1X12_4 = 0;
    int32_t vertex1Y12_4 = 0;
    int32_t vertex2X12_4 = 0;
    int32_t vertex2Y12_4 = 0;
    uint32_t rgba = 0u;
    uint32_t topLeftEdgeMask = 0u;
    uint32_t reserved0 = 0u;
    uint32_t reserved1 = 0u;

    bool operator==(const GsVulkanCt32Triangle &) const = default;
};

static_assert(sizeof(GsVulkanCt32Triangle) == 64u);
static_assert(std::is_standard_layout_v<GsVulkanCt32Triangle>);
static_assert(std::is_trivially_copyable_v<GsVulkanCt32Triangle>);
static_assert(offsetof(GsVulkanCt32Triangle, framebufferBaseBlock) == 0u);
static_assert(offsetof(GsVulkanCt32Triangle, framebufferWidth) == 4u);
static_assert(offsetof(GsVulkanCt32Triangle, boundsX0) == 8u);
static_assert(offsetof(GsVulkanCt32Triangle, boundsY0) == 12u);
static_assert(offsetof(GsVulkanCt32Triangle, boundsX1) == 16u);
static_assert(offsetof(GsVulkanCt32Triangle, boundsY1) == 20u);
static_assert(offsetof(GsVulkanCt32Triangle, vertex0X12_4) == 24u);
static_assert(offsetof(GsVulkanCt32Triangle, vertex0Y12_4) == 28u);
static_assert(offsetof(GsVulkanCt32Triangle, vertex1X12_4) == 32u);
static_assert(offsetof(GsVulkanCt32Triangle, vertex1Y12_4) == 36u);
static_assert(offsetof(GsVulkanCt32Triangle, vertex2X12_4) == 40u);
static_assert(offsetof(GsVulkanCt32Triangle, vertex2Y12_4) == 44u);
static_assert(offsetof(GsVulkanCt32Triangle, rgba) == 48u);
static_assert(offsetof(GsVulkanCt32Triangle, topLeftEdgeMask) == 52u);
static_assert(offsetof(GsVulkanCt32Triangle, reserved0) == 56u);
static_assert(offsetof(GsVulkanCt32Triangle, reserved1) == 60u);

// Applies the triangle predicate and publishes a normalized shader record
// only after every invariant has been validated. Rejection leaves the caller's
// record untouched.
[[nodiscard]] GsBackendDecision prepareGsVulkanCt32Triangle(
    const GsDrawCommand &command,
    GsVulkanCt32Triangle &triangle) noexcept;

// Narrow injectable execution seam used by the Phase 3 verification backend.
// Production requests still enter through the single-owner service below.
class IGsVulkanDrawExecutor
{
public:
    virtual ~IGsVulkanDrawExecutor() = default;

    [[nodiscard]] virtual bool executeCt32Sprite(
        std::span<const uint8_t> input,
        const GsVulkanCt32Sprite &sprite,
        std::vector<uint8_t> &output,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeDepthCt32Sprite(
        std::span<const uint8_t> input,
        const GsVulkanDepthCt32Sprite &sprite,
        std::vector<uint8_t> &output,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeNearestCt32Sprite(
        std::span<const uint8_t> input,
        const GsVulkanNearestCt32Sprite &sprite,
        std::vector<uint8_t> &output,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeLinearCt32Sprite(
        std::span<const uint8_t> input,
        const GsVulkanLinearCt32Sprite &sprite,
        std::vector<uint8_t> &output,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeCt32Triangle(
        std::span<const uint8_t> input,
        const GsVulkanCt32Triangle &triangle,
        std::vector<uint8_t> &output,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool uploadVramPages(
        std::span<const uint8_t> source,
        const GsVramPageMask &pages,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool downloadVramPages(
        std::span<uint8_t> destination,
        const GsVramPageMask &pages,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeResidentCt32Sprite(
        const GsVulkanCt32Sprite &sprite,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeResidentCt32Sprites(
        std::span<const GsVulkanCt32Sprite> sprites,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeResidentDepthCt32Sprite(
        const GsVulkanDepthCt32Sprite &sprite,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeResidentDepthCt32Sprites(
        std::span<const GsVulkanDepthCt32Sprite> sprites,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeResidentNearestCt32Sprite(
        const GsVulkanNearestCt32Sprite &sprite,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeResidentNearestCt32Sprites(
        std::span<const GsVulkanNearestCt32Sprite> sprites,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeResidentLinearCt32Sprite(
        const GsVulkanLinearCt32Sprite &sprite,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeResidentLinearCt32Sprites(
        std::span<const GsVulkanLinearCt32Sprite> sprites,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeResidentCt32Triangle(
        const GsVulkanCt32Triangle &triangle,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeResidentCt32Triangles(
        std::span<const GsVulkanCt32Triangle> triangles,
        std::string *error = nullptr) = 0;
    virtual void shutdown() noexcept = 0;
    [[nodiscard]] virtual GsVulkanCapabilityReport
    capabilities() const = 0;
    [[nodiscard]] virtual GsVulkanServiceStatistics
    statistics() const = 0;
    [[nodiscard]] virtual bool healthy() const = 0;
};

// Owns all Vulkan API objects on one worker thread. Callers synchronously post
// fixed-size jobs through a single bounded slot; no Vulkan object escapes to
// an EE, replay, or presentation thread.
class GsVulkanService final : public IGsVulkanDrawExecutor
{
public:
    [[nodiscard]] static std::unique_ptr<GsVulkanService> create(
        const GsVulkanServiceConfig &config = {},
        GsVulkanCapabilityReport *report = nullptr,
        std::string *error = nullptr);

    ~GsVulkanService() override;

    GsVulkanService(const GsVulkanService &) = delete;
    GsVulkanService &operator=(const GsVulkanService &) = delete;

    // The output is replaced only after a complete, validated operation.
    // Input must contain exactly the PS2's 4 MiB of raw GS local memory.
    [[nodiscard]] bool roundTripVram(
        std::span<const uint8_t> input,
        std::vector<uint8_t> &output,
        std::string *error = nullptr);

    // Runs bounded PSM read/write cases in parallel through the reusable shader
    // memory helpers. Overlapping writes have no ordering guarantee. Both
    // outputs are replaced only after the whole batch completes.
    [[nodiscard]] bool executeMemoryCases(
        std::span<const uint8_t> input,
        std::span<const GsVulkanMemoryCase> cases,
        std::vector<uint8_t> &output,
        std::vector<GsVulkanMemoryResult> &results,
        std::string *error = nullptr);

    // Uploads canonical 4 MiB CPU VRAM, executes one already-classified flat
    // CT32 sprite, and publishes the complete synchronized image on success.
    [[nodiscard]] bool executeCt32Sprite(
        std::span<const uint8_t> input,
        const GsVulkanCt32Sprite &sprite,
        std::vector<uint8_t> &output,
        std::string *error = nullptr) override;

    // Uploads canonical VRAM, executes one exact flat CT32 plus Z32/Z24
    // depth sprite, and publishes the complete synchronized image.
    [[nodiscard]] bool executeDepthCt32Sprite(
        std::span<const uint8_t> input,
        const GsVulkanDepthCt32Sprite &sprite,
        std::vector<uint8_t> &output,
        std::string *error = nullptr) override;

    // Uploads canonical VRAM, executes one prepared nearest CT32 texture
    // sprite through raw GS-local-memory reads, and publishes the synchronized
    // 4 MiB result.
    [[nodiscard]] bool executeNearestCt32Sprite(
        std::span<const uint8_t> input,
        const GsVulkanNearestCt32Sprite &sprite,
        std::vector<uint8_t> &output,
        std::string *error = nullptr) override;

    // Uploads canonical VRAM and executes one exact prepared-DDA linear CT32
    // repeat sprite.
    [[nodiscard]] bool executeLinearCt32Sprite(
        std::span<const uint8_t> input,
        const GsVulkanLinearCt32Sprite &sprite,
        std::vector<uint8_t> &output,
        std::string *error = nullptr) override;

    // Uploads canonical 4 MiB CPU VRAM, executes one prepared exact flat
    // CT32 triangle, and publishes the complete synchronized image. Devices
    // without the explicit 64-bit triangle capability reject before posting.
    [[nodiscard]] bool executeCt32Triangle(
        std::span<const uint8_t> input,
        const GsVulkanCt32Triangle &triangle,
        std::vector<uint8_t> &output,
        std::string *error = nullptr) override;

    // Copies only the selected 8 KiB physical pages between canonical CPU
    // storage and the persistent device-local VRAM allocation. The source and
    // destination still describe the exact 4 MiB GS address space; page data
    // is compacted through the bounded worker slot. A successful download
    // leaves every unselected destination byte untouched.
    [[nodiscard]] bool uploadVramPages(
        std::span<const uint8_t> source,
        const GsVramPageMask &pages,
        std::string *error = nullptr) override;
    [[nodiscard]] bool downloadVramPages(
        std::span<uint8_t> destination,
        const GsVramPageMask &pages,
        std::string *error = nullptr) override;

    // Executes against already-resident device VRAM without an implicit
    // upload or download. Page ownership remains the caller's responsibility.
    [[nodiscard]] bool executeResidentCt32Sprite(
        const GsVulkanCt32Sprite &sprite,
        std::string *error = nullptr) override;

    // Records a bounded set of non-overlapping resident draws into one command
    // buffer and submits it once. The caller remains responsible for page
    // ownership; the service rejects empty, oversized, invalid, or physically
    // overlapping batches before they enter the worker slot.
    [[nodiscard]] bool executeResidentCt32Sprites(
        std::span<const GsVulkanCt32Sprite> sprites,
        std::string *error = nullptr) override;

    // Records a bounded depth batch in guest order. Color and depth surfaces
    // share raw VRAM; dependent dispatch segments are separated explicitly.
    [[nodiscard]] bool executeResidentDepthCt32Sprite(
        const GsVulkanDepthCt32Sprite &sprite,
        std::string *error = nullptr) override;
    [[nodiscard]] bool executeResidentDepthCt32Sprites(
        std::span<const GsVulkanDepthCt32Sprite> sprites,
        std::string *error = nullptr) override;

    // Records a bounded nearest-texture batch against resident raw VRAM.
    // Shared read-only texture pages are allowed; any pairwise read/write or
    // write/write dependency is rejected before the worker slot.
    [[nodiscard]] bool executeResidentNearestCt32Sprite(
        const GsVulkanNearestCt32Sprite &sprite,
        std::string *error = nullptr) override;
    [[nodiscard]] bool executeResidentNearestCt32Sprites(
        std::span<const GsVulkanNearestCt32Sprite> sprites,
        std::string *error = nullptr) override;

    // Records a bounded linear-texture batch against resident raw VRAM.
    // Records execute in guest order; the service inserts barriers between
    // dependent dispatch segments while allowing independent work to batch.
    [[nodiscard]] bool executeResidentLinearCt32Sprite(
        const GsVulkanLinearCt32Sprite &sprite,
        std::string *error = nullptr) override;
    [[nodiscard]] bool executeResidentLinearCt32Sprites(
        std::span<const GsVulkanLinearCt32Sprite> sprites,
        std::string *error = nullptr) override;

    // Uses the exact 64-bit triangle pipeline against already-resident VRAM.
    // As with sprite batches, records must have pairwise-disjoint conservative
    // physical write-page masks so no inter-dispatch ordering is assumed.
    [[nodiscard]] bool executeResidentCt32Triangle(
        const GsVulkanCt32Triangle &triangle,
        std::string *error = nullptr) override;
    [[nodiscard]] bool executeResidentCt32Triangles(
        std::span<const GsVulkanCt32Triangle> triangles,
        std::string *error = nullptr) override;

    // Idempotently drains accepted work and destroys every Vulkan object on
    // the owning worker before returning. The diagnostic snapshots remain
    // readable after shutdown.
    void shutdown() noexcept override;

    [[nodiscard]] GsVulkanCapabilityReport capabilities() const override;
    [[nodiscard]] GsVulkanServiceStatistics statistics() const override;
    [[nodiscard]] bool healthy() const override;

private:
    struct Impl;
    explicit GsVulkanService(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> m_impl;
};

[[nodiscard]] GsVulkanCapabilityReport probeGsVulkanCapabilities(
    const GsVulkanProbeConfig &config = {});

[[nodiscard]] std::string_view gsVulkanProbeStatusName(
    GsVulkanProbeStatus status) noexcept;
[[nodiscard]] std::string_view gsVulkanDeviceKindName(
    GsVulkanDeviceKind kind) noexcept;
[[nodiscard]] std::string gsVulkanVersionString(uint32_t version);
