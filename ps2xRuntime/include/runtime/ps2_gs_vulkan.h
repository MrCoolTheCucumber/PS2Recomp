#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
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

[[nodiscard]] GsVulkanCapabilityReport probeGsVulkanCapabilities(
    const GsVulkanProbeConfig &config = {});

[[nodiscard]] std::string_view gsVulkanProbeStatusName(
    GsVulkanProbeStatus status) noexcept;
[[nodiscard]] std::string_view gsVulkanDeviceKindName(
    GsVulkanDeviceKind kind) noexcept;
[[nodiscard]] std::string gsVulkanVersionString(uint32_t version);
