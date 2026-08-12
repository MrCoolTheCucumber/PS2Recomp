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
inline constexpr size_t GS_VULKAN_AUXILIARY_STORAGE_SIZE =
    8u * 1024u * 1024u;
// Resident commands share one bounded host-visible record allocation and one
// command buffer. A larger bound amortizes queue/fence overhead for the many
// small primitives emitted by PS2 renderers while remaining below the fixed
// 2 MiB auxiliary allocation even for the largest exact draw record.
inline constexpr size_t GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH = 1024u;
inline constexpr size_t GS_VULKAN_MAX_RESIDENT_DEPTH_CT32_BATCH =
    GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH;
inline constexpr size_t GS_VULKAN_MAX_RESIDENT_NEAREST_CT32_BATCH =
    GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH;
inline constexpr size_t GS_VULKAN_MAX_RESIDENT_LINEAR_CT32_BATCH =
    GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH;
inline constexpr size_t
    GS_VULKAN_MAX_RESIDENT_FEEDBACK_LINEAR_DEPTH_CT32_BATCH =
        GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH;
inline constexpr size_t GS_VULKAN_MAX_RESIDENT_TRIANGLE_BATCH =
    GS_VRAM_PAGE_COUNT;
inline constexpr size_t GS_VULKAN_MAX_RESIDENT_T8_TRIANGLE_BATCH =
    6144u;

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
    bool shaderFloat64 = false;

    // Vulkan 1.0 compute shaders have exact 32-bit integer storage-buffer
    // access. This flag also requires the queue, limits, and memory classes
    // needed by the fixed 4 MiB raw-GS-buffer design.
    bool exactVramStorage = false;

    // The initial exact triangle kernel additionally requires native signed
    // 64-bit shader arithmetic for full-range GS 12.4 edge equations.
    bool exactCt32Triangle = false;

    // Gouraud CT32 plus constant Z32/Z24 depth retains the same signed
    // 64-bit coverage requirement and additionally reproduces the software
    // rasterizer's single-precision eight-pixel color DDA.
    bool exactGouraudDepthCt32Triangle = false;

    // Perspective T8 Gouraud/depth triangles additionally reproduce the
    // software rasterizer's binary64 Z DDA. Their decoded palette and setup
    // record are supplied through the bounded auxiliary storage buffer.
    bool exactT8GouraudDepthCt32Triangle = false;

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

    // Recursive linear/depth execution additionally requires a separately
    // bound immutable 4 MiB texture snapshot and the exact 128-byte record.
    bool exactFeedbackLinearDepthCt32Sprite = false;
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
    uint64_t feedbackLinearDepthCt32SpriteDrawsCompleted = 0u;
    uint64_t feedbackLinearDepthCt32SpriteDrawsFailed = 0u;
    uint64_t feedbackLinearDepthCt32SpritePixelsExecuted = 0u;
    uint64_t
        residentFeedbackLinearDepthCt32SpriteBatchesCompleted = 0u;
    uint64_t residentFeedbackLinearDepthCt32SpriteBatchesFailed = 0u;
    uint64_t largestResidentFeedbackLinearDepthCt32SpriteBatch = 0u;
    uint64_t residentLinearCt32SpriteBatchesCompleted = 0u;
    uint64_t residentLinearCt32SpriteBatchesFailed = 0u;
    uint64_t largestResidentLinearCt32SpriteBatch = 0u;
    uint64_t residentNearestCt32SpriteBatchesCompleted = 0u;
    uint64_t residentNearestCt32SpriteBatchesFailed = 0u;
    uint64_t largestResidentNearestCt32SpriteBatch = 0u;
    uint64_t triangleDrawsCompleted = 0u;
    uint64_t triangleDrawsFailed = 0u;
    uint64_t triangleCandidatePixelsExecuted = 0u;
    uint64_t gouraudDepthCt32TriangleDrawsCompleted = 0u;
    uint64_t gouraudDepthCt32TriangleDrawsFailed = 0u;
    uint64_t gouraudDepthCt32TriangleCandidatePixelsExecuted = 0u;
    uint64_t t8GouraudDepthCt32TriangleDrawsCompleted = 0u;
    uint64_t t8GouraudDepthCt32TriangleDrawsFailed = 0u;
    uint64_t t8GouraudDepthCt32TriangleCandidatePixelsExecuted = 0u;
    uint64_t residentT8GouraudDepthCt32TriangleBatchesCompleted = 0u;
    uint64_t residentT8GouraudDepthCt32TriangleBatchesFailed = 0u;
    uint64_t largestResidentT8GouraudDepthCt32TriangleBatch = 0u;
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
// colorBlendMode is a packed color-operation descriptor. Source-copy and
// source-over use their literal operation values. FIXED_ALPHA additionally
// stores FIX in bits 8..15 and optional CT32 DATE/DATM state in bits 16..17.
// Every operation writes source alpha; alpha blending only changes RGB.
inline constexpr uint32_t GS_VULKAN_DEPTH_CT32_COLOR_SOURCE_COPY = 0u;
inline constexpr uint32_t GS_VULKAN_DEPTH_CT32_COLOR_SOURCE_OVER = 1u;
inline constexpr uint32_t GS_VULKAN_DEPTH_CT32_COLOR_FIXED_ALPHA = 2u;
inline constexpr uint32_t GS_VULKAN_DEPTH_CT32_COLOR_OPERATION_MASK = 0x3u;
inline constexpr uint32_t GS_VULKAN_DEPTH_CT32_COLOR_FIXED_ALPHA_SHIFT = 8u;
inline constexpr uint32_t GS_VULKAN_DEPTH_CT32_COLOR_FIXED_ALPHA_MASK =
    0xFFu << GS_VULKAN_DEPTH_CT32_COLOR_FIXED_ALPHA_SHIFT;
inline constexpr uint32_t GS_VULKAN_DEPTH_CT32_COLOR_DATE = 1u << 16u;
inline constexpr uint32_t GS_VULKAN_DEPTH_CT32_COLOR_DATM = 1u << 17u;
inline constexpr uint32_t GS_VULKAN_DEPTH_CT32_COLOR_DESCRIPTOR_MASK =
    GS_VULKAN_DEPTH_CT32_COLOR_OPERATION_MASK |
    GS_VULKAN_DEPTH_CT32_COLOR_FIXED_ALPHA_MASK |
    GS_VULKAN_DEPTH_CT32_COLOR_DATE |
    GS_VULKAN_DEPTH_CT32_COLOR_DATM;

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
    uint32_t colorBlendMode = GS_VULKAN_DEPTH_CT32_COLOR_SOURCE_COPY;
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
static_assert(offsetof(GsVulkanDepthCt32Sprite, colorBlendMode) == 48u);
static_assert(offsetof(GsVulkanDepthCt32Sprite, reserved3) == 60u);

// Rejection leaves the caller's record untouched.
[[nodiscard]] GsBackendDecision prepareGsVulkanDepthCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanDepthCt32Sprite &sprite) noexcept;

// Publishes the same depth ABI with its exact source-over color operation.
// Rejection leaves the caller's record untouched. Device execution remains
// independently capability-gated by the Vulkan service.
[[nodiscard]] GsBackendDecision prepareGsVulkanSourceOverDepthCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanDepthCt32Sprite &sprite) noexcept;

// Publishes the exact framebuffer-only alpha-fail operation through the same
// depth sprite ABI. The record carries FIX and optional DATE/DATM state while
// forcing depthWrite to zero. Rejection preserves the caller's record.
[[nodiscard]] GsBackendDecision
prepareGsVulkanFramebufferOnlyAlphaFailDepthCt32Sprite(
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

// Fixed record for the first recursive framebuffer-feedback contract. Its
// leading 96 bytes retain the exact linear CT32 DDA ABI. The remaining words
// describe the independent Z32/Z24 operation and require texture reads from a
// separately supplied immutable 4 MiB feedback snapshot. Snapshot creation,
// same-surface reuse, and invalidation are ordered by the raster frontend;
// this record never permits in-place texture reads from writable resident VRAM.
inline constexpr uint32_t GS_VULKAN_TEXTURE_SOURCE_FEEDBACK_SNAPSHOT = 1u;

struct alignas(16) GsVulkanFeedbackLinearDepthCt32Sprite
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
    uint32_t depthBaseBlock = 0u;
    uint32_t depthPsm = 0u;
    uint32_t depth = 0u;
    uint32_t depthTestMethod = 0u;
    uint32_t depthWrite = 0u;
    uint32_t textureSource =
        GS_VULKAN_TEXTURE_SOURCE_FEEDBACK_SNAPSHOT;
    uint32_t reserved0 = 0u;
    uint32_t reserved1 = 0u;

    bool operator==(
        const GsVulkanFeedbackLinearDepthCt32Sprite &) const = default;
};

static_assert(sizeof(GsVulkanFeedbackLinearDepthCt32Sprite) == 128u);
static_assert(std::is_standard_layout_v<
              GsVulkanFeedbackLinearDepthCt32Sprite>);
static_assert(std::is_trivially_copyable_v<
              GsVulkanFeedbackLinearDepthCt32Sprite>);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    framebufferBaseBlock) == 0u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    framebufferWidth) == 4u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    boundsX0) == 8u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    boundsY0) == 12u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    boundsX1) == 16u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    boundsY1) == 20u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    textureBaseBlock) == 24u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    textureWidth) == 28u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    textureMaskU) == 32u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    textureMaskV) == 36u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    fixedBaseU) == 40u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    fixedBlockStepU) == 44u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    fixedLaneU) == 48u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    fixedScanVBits) == 80u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    fixedStepVBits) == 84u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    textureWrapU) == 88u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    textureWrapV) == 92u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    depthBaseBlock) == 96u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    depthPsm) == 100u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    depth) == 104u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    depthTestMethod) == 108u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    depthWrite) == 112u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    textureSource) == 116u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    reserved0) == 120u);
static_assert(offsetof(
    GsVulkanFeedbackLinearDepthCt32Sprite,
    reserved1) == 124u);

// Rejection leaves the caller's record untouched. A successful record still
// requires the matching immutable feedback snapshot at execution time.
[[nodiscard]] GsBackendDecision
prepareGsVulkanFeedbackLinearDepthCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanFeedbackLinearDepthCt32Sprite &sprite) noexcept;

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

// Exact Phase 5 Gouraud/depth triangle record. Unlike the flat record, vertex
// order and attributes remain paired exactly as submitted; positiveArea tells
// the executor how to normalize edge signs without perturbing equal-Y DDA
// ordering. The first contract has constant Z and alpha 128 at every vertex,
// but retains independently varying RGB for exact color interpolation.
struct alignas(16) GsVulkanGouraudDepthCt32Triangle
{
    uint32_t framebufferBaseBlock = 0u;
    uint32_t framebufferWidth = 0u;
    uint32_t boundsX0 = 0u;
    uint32_t boundsY0 = 0u;
    uint32_t boundsX1 = 0u;
    uint32_t boundsY1 = 0u;
    uint32_t depthBaseBlock = 0u;
    uint32_t depthPsm = 0u;
    uint32_t depth = 0u;
    uint32_t positiveArea = 0u;
    int32_t vertex0X12_4 = 0;
    int32_t vertex0Y12_4 = 0;
    int32_t vertex1X12_4 = 0;
    int32_t vertex1Y12_4 = 0;
    int32_t vertex2X12_4 = 0;
    int32_t vertex2Y12_4 = 0;
    uint32_t rgba0 = 0u;
    uint32_t rgba1 = 0u;
    uint32_t rgba2 = 0u;
    uint32_t topLeftEdgeMask = 0u;
    // Exact host float bit patterns from the software rasterizer's setup.
    // Shipping setup results avoids relying on device-specific division and
    // setup contraction while retaining per-pixel fused scanline evaluation.
    std::array<uint32_t, 4> colorDxBits{};
    std::array<uint32_t, 4> colorDyBits{};

    bool operator==(
        const GsVulkanGouraudDepthCt32Triangle &) const = default;
};

static_assert(sizeof(GsVulkanGouraudDepthCt32Triangle) == 112u);
static_assert(std::is_standard_layout_v<
              GsVulkanGouraudDepthCt32Triangle>);
static_assert(std::is_trivially_copyable_v<
              GsVulkanGouraudDepthCt32Triangle>);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    framebufferBaseBlock) == 0u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    framebufferWidth) == 4u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    boundsX0) == 8u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    boundsY0) == 12u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    boundsX1) == 16u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    boundsY1) == 20u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    depthBaseBlock) == 24u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    depthPsm) == 28u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    depth) == 32u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    positiveArea) == 36u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    vertex0X12_4) == 40u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    vertex0Y12_4) == 44u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    vertex1X12_4) == 48u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    vertex1Y12_4) == 52u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    vertex2X12_4) == 56u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    vertex2Y12_4) == 60u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    rgba0) == 64u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    rgba1) == 68u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    rgba2) == 72u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    topLeftEdgeMask) == 76u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    colorDxBits) == 80u);
static_assert(offsetof(
    GsVulkanGouraudDepthCt32Triangle,
    colorDyBits) == 96u);

// Applies the complete semantic predicate and publishes the raw ordered
// vertices/attributes only after every invariant has been validated.
// Rejection leaves the caller's record untouched.
[[nodiscard]] GsBackendDecision
prepareGsVulkanGouraudSourceOverDepthCt32Triangle(
    const GsDrawCommand &command,
    GsVulkanGouraudDepthCt32Triangle &triangle) noexcept;

// Storage-buffer record for the dominant perspective T8 triangle contract.
// Every floating-point field stores the exact host setup bit pattern. The
// embedded palette is the GS frontend's decoded cached CLUT at submission
// time; reading palette source VRAM later would be architecturally incorrect.
struct alignas(16) GsVulkanT8GouraudDepthCt32Triangle
{
    uint32_t framebufferBaseBlock = 0u;
    uint32_t framebufferWidth = 0u;
    uint32_t boundsX0 = 0u;
    uint32_t boundsY0 = 0u;
    uint32_t boundsX1 = 0u;
    uint32_t boundsY1 = 0u;
    uint32_t depthBaseBlock = 0u;
    uint32_t depthPsm = 0u;
    uint32_t positiveArea = 0u;
    uint32_t topLeftEdgeMask = 0u;
    std::array<int32_t, 3> vertexX12_4{};
    std::array<int32_t, 3> vertexY12_4{};
    std::array<uint32_t, 3> vertexZ{};
    std::array<uint32_t, 3> rgba{};
    uint32_t vertexFog = 0u;
    uint32_t fogColor = 0u;
    std::array<uint32_t, 3> sBits{};
    std::array<uint32_t, 3> tBits{};
    std::array<uint32_t, 3> qBits{};
    std::array<uint32_t, 4> colorDxBits{};
    std::array<uint32_t, 4> colorDyBits{};
    uint32_t fogDxBits = 0u;
    uint32_t fogDyBits = 0u;
    std::array<uint32_t, 3> textureDxBits{};
    std::array<uint32_t, 3> textureDyBits{};
    std::array<uint32_t, 2> depthDxBits{};
    std::array<uint32_t, 2> depthDyBits{};
    // Zero for a standalone record; resident records index their separately
    // captured palette table here.
    uint32_t paletteIndex = 0u;
    uint32_t rasterFlags = 0u;
    uint32_t reserved = 0u;
    std::array<uint32_t, 8> textureBaseBlocks{};
    std::array<uint32_t, 8> textureWidths{};
    uint32_t textureWidthLog2 = 0u;
    uint32_t textureHeightLog2 = 0u;
    uint32_t maximumMipLevel = 0u;
    uint32_t textureWrapU = 0u;
    uint32_t textureWrapV = 0u;
    int32_t lodK = 0;
    uint32_t lodL = 0u;
    uint32_t alphaReference = 0u;
    std::array<uint32_t, 256> palette{};

    bool operator==(
        const GsVulkanT8GouraudDepthCt32Triangle &) const = default;
};

static_assert(sizeof(GsVulkanT8GouraudDepthCt32Triangle) == 1344u);
static_assert(std::is_standard_layout_v<
              GsVulkanT8GouraudDepthCt32Triangle>);
static_assert(std::is_trivially_copyable_v<
              GsVulkanT8GouraudDepthCt32Triangle>);
static_assert(offsetof(
    GsVulkanT8GouraudDepthCt32Triangle,
    textureBaseBlocks) == 224u);
static_assert(offsetof(
    GsVulkanT8GouraudDepthCt32Triangle,
    palette) == 320u);

using GsVulkanT8Palette = std::array<uint32_t, 256>;

inline constexpr uint32_t GS_VULKAN_T8_GOURAUD_FLAG_FOG = 1u << 0u;
inline constexpr uint32_t GS_VULKAN_T8_GOURAUD_FLAG_DEPTH_GEQUAL =
    1u << 1u;
inline constexpr uint32_t GS_VULKAN_T8_GOURAUD_FLAG_DEPTH_WRITE =
    1u << 2u;
inline constexpr uint32_t GS_VULKAN_T8_GOURAUD_FLAG_ALPHA_FAIL_RGB_ONLY =
    1u << 3u;
inline constexpr uint32_t GS_VULKAN_T8_GOURAUD_FLAG_CONSTANT_Q_FIXED =
    1u << 4u;
inline constexpr uint32_t GS_VULKAN_T8_GOURAUD_VALID_FLAGS =
    GS_VULKAN_T8_GOURAUD_FLAG_FOG |
    GS_VULKAN_T8_GOURAUD_FLAG_DEPTH_GEQUAL |
    GS_VULKAN_T8_GOURAUD_FLAG_DEPTH_WRITE |
    GS_VULKAN_T8_GOURAUD_FLAG_ALPHA_FAIL_RGB_ONLY |
    GS_VULKAN_T8_GOURAUD_FLAG_CONSTANT_Q_FIXED;

// Compact resident record for the same exact T8 contract. Palettes are
// captured once per distinct consecutive CLUT image and indexed explicitly
// by each record. This keeps a 1 KiB immutable palette from
// being copied and uploaded once for every tiny strip triangle.
struct alignas(16) GsVulkanResidentT8GouraudDepthCt32Triangle
{
    uint32_t framebufferBaseBlock = 0u;
    uint32_t framebufferWidth = 0u;
    uint32_t boundsX0 = 0u;
    uint32_t boundsY0 = 0u;
    uint32_t boundsX1 = 0u;
    uint32_t boundsY1 = 0u;
    uint32_t depthBaseBlock = 0u;
    uint32_t depthPsm = 0u;
    uint32_t positiveArea = 0u;
    uint32_t topLeftEdgeMask = 0u;
    std::array<int32_t, 3> vertexX12_4{};
    std::array<int32_t, 3> vertexY12_4{};
    std::array<uint32_t, 3> vertexZ{};
    std::array<uint32_t, 3> rgba{};
    uint32_t vertexFog = 0u;
    uint32_t fogColor = 0u;
    std::array<uint32_t, 3> sBits{};
    std::array<uint32_t, 3> tBits{};
    std::array<uint32_t, 3> qBits{};
    std::array<uint32_t, 4> colorDxBits{};
    std::array<uint32_t, 4> colorDyBits{};
    uint32_t fogDxBits = 0u;
    uint32_t fogDyBits = 0u;
    std::array<uint32_t, 3> textureDxBits{};
    std::array<uint32_t, 3> textureDyBits{};
    std::array<uint32_t, 2> depthDxBits{};
    std::array<uint32_t, 2> depthDyBits{};
    uint32_t paletteIndex = 0u;
    // Per-record state permits exact batching of nearby GS contracts without
    // multiplying pipelines. An unset depth-comparison bit means ALWAYS.
    uint32_t rasterFlags = 0u;
    uint32_t reserved = 0u;
    std::array<uint32_t, 8> textureBaseBlocks{};
    std::array<uint32_t, 8> textureWidths{};
    uint32_t textureWidthLog2 = 0u;
    uint32_t textureHeightLog2 = 0u;
    uint32_t maximumMipLevel = 0u;
    uint32_t textureWrapU = 0u;
    uint32_t textureWrapV = 0u;
    int32_t lodK = 0;
    uint32_t lodL = 0u;
    uint32_t alphaReference = 0u;

    bool operator==(
        const GsVulkanResidentT8GouraudDepthCt32Triangle &) const = default;
};

static_assert(
    sizeof(GsVulkanResidentT8GouraudDepthCt32Triangle) == 320u);
static_assert(std::is_standard_layout_v<
              GsVulkanResidentT8GouraudDepthCt32Triangle>);
static_assert(std::is_trivially_copyable_v<
              GsVulkanResidentT8GouraudDepthCt32Triangle>);
static_assert(offsetof(
    GsVulkanResidentT8GouraudDepthCt32Triangle,
    paletteIndex) == offsetof(
        GsVulkanT8GouraudDepthCt32Triangle, paletteIndex));
static_assert(offsetof(
    GsVulkanResidentT8GouraudDepthCt32Triangle,
    textureBaseBlocks) == 224u);

[[nodiscard]] GsBackendDecision
prepareGsVulkanT8GouraudSourceOverDepthCt32Triangle(
    const GsDrawCommand &command,
    std::span<const uint32_t> decodedPalette,
    GsVulkanT8GouraudDepthCt32Triangle &triangle,
    const GsDrawResources *classifiedResources = nullptr) noexcept;

[[nodiscard]] GsBackendDecision
prepareGsVulkanResidentT8GouraudSourceOverDepthCt32Triangle(
    const GsDrawCommand &command,
    uint32_t paletteIndex,
    GsVulkanResidentT8GouraudDepthCt32Triangle &triangle,
    const GsDrawResources *classifiedResources = nullptr) noexcept;

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
    [[nodiscard]] virtual bool executeFeedbackLinearDepthCt32Sprite(
        std::span<const uint8_t> input,
        std::span<const uint8_t> feedbackSnapshot,
        const GsVulkanFeedbackLinearDepthCt32Sprite &sprite,
        std::vector<uint8_t> &output,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeCt32Triangle(
        std::span<const uint8_t> input,
        const GsVulkanCt32Triangle &triangle,
        std::vector<uint8_t> &output,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeGouraudDepthCt32Triangle(
        std::span<const uint8_t> input,
        const GsVulkanGouraudDepthCt32Triangle &triangle,
        std::vector<uint8_t> &output,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool executeT8GouraudDepthCt32Triangle(
        std::span<const uint8_t>,
        const GsVulkanT8GouraudDepthCt32Triangle &,
        std::vector<uint8_t> &,
        std::string *error = nullptr)
    {
        if (error)
            *error = "T8 Gouraud depth triangle execution is unavailable";
        return false;
    }
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
    [[nodiscard]] virtual bool
    executeResidentFeedbackLinearDepthCt32Sprite(
        std::span<const uint8_t> feedbackSnapshot,
        const GsVulkanFeedbackLinearDepthCt32Sprite &sprite,
        std::string *error = nullptr) = 0;
    [[nodiscard]] virtual bool
    executeResidentFeedbackLinearDepthCt32Sprites(
        std::span<const uint8_t> feedbackSnapshot,
        std::span<const GsVulkanFeedbackLinearDepthCt32Sprite> sprites,
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
    [[nodiscard]] virtual bool executeResidentT8GouraudDepthCt32Triangles(
        std::span<const GsVulkanResidentT8GouraudDepthCt32Triangle>,
        std::span<const GsVulkanT8Palette>,
        std::string *error = nullptr)
    {
        if (error)
            *error = "resident T8 Gouraud depth triangle execution is unavailable";
        return false;
    }
    // The raster backend has already classified resource aliasing for these
    // immutable records. Generic executors may retain the full public
    // validation path; the production service can omit that duplicate page-
    // range reconstruction while still validating every raw field.
    [[nodiscard]] virtual bool
    executePreparedResidentT8GouraudDepthCt32Triangles(
        std::vector<GsVulkanResidentT8GouraudDepthCt32Triangle> triangles,
        std::vector<GsVulkanT8Palette> palettes,
        std::string *error = nullptr)
    {
        return executeResidentT8GouraudDepthCt32Triangles(
            std::span<const GsVulkanResidentT8GouraudDepthCt32Triangle>(
                triangles),
            std::span<const GsVulkanT8Palette>(palettes), error);
    }
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

    // Uploads canonical writable VRAM and a distinct immutable texture
    // snapshot, then executes one recursive linear CT32 plus depth sprite.
    [[nodiscard]] bool executeFeedbackLinearDepthCt32Sprite(
        std::span<const uint8_t> input,
        std::span<const uint8_t> feedbackSnapshot,
        const GsVulkanFeedbackLinearDepthCt32Sprite &sprite,
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

    // Uploads canonical 4 MiB CPU VRAM, executes one prepared Gouraud CT32
    // triangle with constant ALWAYS-write Z32/Z24 depth, and publishes the
    // complete synchronized image.
    [[nodiscard]] bool executeGouraudDepthCt32Triangle(
        std::span<const uint8_t> input,
        const GsVulkanGouraudDepthCt32Triangle &triangle,
        std::vector<uint8_t> &output,
        std::string *error = nullptr) override;

    [[nodiscard]] bool executeT8GouraudDepthCt32Triangle(
        std::span<const uint8_t> input,
        const GsVulkanT8GouraudDepthCt32Triangle &triangle,
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

    // Uploads one immutable texture snapshot and records a bounded recursive
    // linear/depth batch against already-resident canonical VRAM. Records run
    // in guest order; canonical page ownership remains the caller's concern.
    [[nodiscard]] bool executeResidentFeedbackLinearDepthCt32Sprite(
        std::span<const uint8_t> feedbackSnapshot,
        const GsVulkanFeedbackLinearDepthCt32Sprite &sprite,
        std::string *error = nullptr) override;
    [[nodiscard]] bool executeResidentFeedbackLinearDepthCt32Sprites(
        std::span<const uint8_t> feedbackSnapshot,
        std::span<const GsVulkanFeedbackLinearDepthCt32Sprite> sprites,
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

    [[nodiscard]] bool executeResidentT8GouraudDepthCt32Triangles(
        std::span<const GsVulkanResidentT8GouraudDepthCt32Triangle> triangles,
        std::span<const GsVulkanT8Palette> palettes,
        std::string *error = nullptr) override;
    [[nodiscard]] bool
    executePreparedResidentT8GouraudDepthCt32Triangles(
        std::vector<GsVulkanResidentT8GouraudDepthCt32Triangle> triangles,
        std::vector<GsVulkanT8Palette> palettes,
        std::string *error = nullptr) override;

    // Idempotently drains accepted work and destroys every Vulkan object on
    // the owning worker before returning. The diagnostic snapshots remain
    // readable after shutdown.
    void shutdown() noexcept override;

    [[nodiscard]] GsVulkanCapabilityReport capabilities() const override;
    [[nodiscard]] GsVulkanServiceStatistics statistics() const override;
    [[nodiscard]] bool healthy() const override;

private:
    [[nodiscard]] bool executeResidentT8GouraudDepthCt32TrianglesImpl(
        std::vector<GsVulkanResidentT8GouraudDepthCt32Triangle> triangles,
        std::vector<GsVulkanT8Palette> palettes,
        bool validateResourceViews,
        std::string *error);
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
