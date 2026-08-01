#pragma once

#include "runtime/ps2_gs_backend.h"

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
    uint64_t bytesUploaded = 0u;
    uint64_t bytesDownloaded = 0u;
    uint64_t fenceWaitNanoseconds = 0u;
    uint64_t memoryBatchesCompleted = 0u;
    uint64_t memoryBatchesFailed = 0u;
    uint64_t memoryCasesExecuted = 0u;
    uint64_t spriteDrawsCompleted = 0u;
    uint64_t spriteDrawsFailed = 0u;
    uint64_t spritePixelsExecuted = 0u;
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
