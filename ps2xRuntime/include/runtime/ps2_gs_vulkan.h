#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
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
    uint32_t validationWarnings = 0u;
    uint32_t validationErrors = 0u;
    bool deviceLost = false;
};

// Owns all Vulkan API objects on one worker thread. Callers synchronously post
// fixed-size jobs through a single bounded slot; no Vulkan object escapes to
// an EE, replay, or presentation thread.
class GsVulkanService final
{
public:
    [[nodiscard]] static std::unique_ptr<GsVulkanService> create(
        const GsVulkanServiceConfig &config = {},
        GsVulkanCapabilityReport *report = nullptr,
        std::string *error = nullptr);

    ~GsVulkanService();

    GsVulkanService(const GsVulkanService &) = delete;
    GsVulkanService &operator=(const GsVulkanService &) = delete;

    // The output is replaced only after a complete, validated operation.
    // Input must contain exactly the PS2's 4 MiB of raw GS local memory.
    [[nodiscard]] bool roundTripVram(
        std::span<const uint8_t> input,
        std::vector<uint8_t> &output,
        std::string *error = nullptr);

    // Idempotently drains accepted work and destroys every Vulkan object on
    // the owning worker before returning. The diagnostic snapshots remain
    // readable after shutdown.
    void shutdown() noexcept;

    [[nodiscard]] GsVulkanCapabilityReport capabilities() const;
    [[nodiscard]] GsVulkanServiceStatistics statistics() const;
    [[nodiscard]] bool healthy() const;

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
