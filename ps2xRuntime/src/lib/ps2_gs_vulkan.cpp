#include "runtime/ps2_gs_vulkan.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <sstream>
#include <tuple>
#include <utility>

#if PS2X_HAS_GS_VULKAN
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

std::string_view gsVulkanProbeStatusName(
    GsVulkanProbeStatus status) noexcept
{
    switch (status)
    {
    case GsVulkanProbeStatus::CompiledOut:
        return "compiled-out";
    case GsVulkanProbeStatus::LoaderUnavailable:
        return "loader-unavailable";
    case GsVulkanProbeStatus::LoaderInvalid:
        return "loader-invalid";
    case GsVulkanProbeStatus::ValidationUnavailable:
        return "validation-unavailable";
    case GsVulkanProbeStatus::InstanceCreationFailed:
        return "instance-creation-failed";
    case GsVulkanProbeStatus::DeviceEnumerationFailed:
        return "device-enumeration-failed";
    case GsVulkanProbeStatus::NoPhysicalDevices:
        return "no-physical-devices";
    case GsVulkanProbeStatus::NoSuitableDevice:
        return "no-suitable-device";
    case GsVulkanProbeStatus::ValidationError:
        return "validation-error";
    case GsVulkanProbeStatus::Ready:
        return "ready";
    }
    return "unknown";
}

std::string_view gsVulkanDeviceKindName(
    GsVulkanDeviceKind kind) noexcept
{
    switch (kind)
    {
    case GsVulkanDeviceKind::Other:
        return "other";
    case GsVulkanDeviceKind::IntegratedGpu:
        return "integrated-gpu";
    case GsVulkanDeviceKind::DiscreteGpu:
        return "discrete-gpu";
    case GsVulkanDeviceKind::VirtualGpu:
        return "virtual-gpu";
    case GsVulkanDeviceKind::Cpu:
        return "cpu";
    }
    return "unknown";
}

std::string gsVulkanVersionString(uint32_t version)
{
    const uint32_t variant = version >> 29u;
    const uint32_t major = (version >> 22u) & 0x7Fu;
    const uint32_t minor = (version >> 12u) & 0x3FFu;
    const uint32_t patch = version & 0xFFFu;
    std::ostringstream output;
    if (variant != 0u)
        output << variant << ':';
    output << major << '.' << minor << '.' << patch;
    return output.str();
}

#if PS2X_HAS_GS_VULKAN
namespace
{
    class DynamicLibrary final
    {
    public:
        DynamicLibrary() = default;
        ~DynamicLibrary()
        {
            close();
        }

        DynamicLibrary(const DynamicLibrary &) = delete;
        DynamicLibrary &operator=(const DynamicLibrary &) = delete;

        bool open(const std::string &path, std::string &error)
        {
            close();
#if defined(_WIN32)
            m_handle = LoadLibraryA(path.c_str());
            if (!m_handle)
            {
                error = "LoadLibrary failed with code " +
                        std::to_string(GetLastError());
                return false;
            }
#else
            dlerror();
            m_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!m_handle)
            {
                const char *message = dlerror();
                error = message ? message : "dlopen failed";
                return false;
            }
#endif
            m_path = path;
            error.clear();
            return true;
        }

        [[nodiscard]] void *symbol(const char *name) const noexcept
        {
#if defined(_WIN32)
            return m_handle
                ? reinterpret_cast<void *>(GetProcAddress(m_handle, name))
                : nullptr;
#else
            return m_handle ? dlsym(m_handle, name) : nullptr;
#endif
        }

        [[nodiscard]] const std::string &path() const noexcept
        {
            return m_path;
        }

    private:
        void close() noexcept
        {
            if (!m_handle)
                return;
#if defined(_WIN32)
            FreeLibrary(m_handle);
#else
            dlclose(m_handle);
#endif
            m_handle = nullptr;
            m_path.clear();
        }

#if defined(_WIN32)
        HMODULE m_handle = nullptr;
#else
        void *m_handle = nullptr;
#endif
        std::string m_path;
    };

    struct ValidationCounters
    {
        std::atomic<uint32_t> warnings{0u};
        std::atomic<uint32_t> errors{0u};
    };

    VKAPI_ATTR VkBool32 VKAPI_CALL validationCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT,
        const VkDebugUtilsMessengerCallbackDataEXT *,
        void *userData)
    {
        auto *counters = static_cast<ValidationCounters *>(userData);
        if (!counters)
            return VK_FALSE;
        if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0u)
            counters->errors.fetch_add(1u, std::memory_order_relaxed);
        else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0u)
            counters->warnings.fetch_add(1u, std::memory_order_relaxed);
        return VK_FALSE;
    }

    VkDebugUtilsMessengerCreateInfoEXT makeDebugCreateInfo(
        ValidationCounters &counters) noexcept
    {
        VkDebugUtilsMessengerCreateInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        info.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        info.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        info.pfnUserCallback = validationCallback;
        info.pUserData = &counters;
        return info;
    }

    std::string resultDescription(VkResult result)
    {
        switch (result)
        {
        case VK_SUCCESS:
            return "VK_SUCCESS";
        case VK_NOT_READY:
            return "VK_NOT_READY";
        case VK_TIMEOUT:
            return "VK_TIMEOUT";
        case VK_ERROR_OUT_OF_HOST_MEMORY:
            return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:
            return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST:
            return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_LAYER_NOT_PRESENT:
            return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT:
            return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER:
            return "VK_ERROR_INCOMPATIBLE_DRIVER";
        default:
            return "VkResult(" + std::to_string(static_cast<int32_t>(result)) + ")";
        }
    }

    template <typename T, typename Enumerator>
    VkResult enumerateValues(Enumerator &&enumerator,
                             std::vector<T> &values)
    {
        for (uint32_t attempt = 0u; attempt < 4u; ++attempt)
        {
            uint32_t count = 0u;
            VkResult result = enumerator(&count, nullptr);
            if (result != VK_SUCCESS && result != VK_INCOMPLETE)
                return result;
            values.resize(count);
            if (count == 0u)
                return VK_SUCCESS;
            result = enumerator(&count, values.data());
            values.resize(count);
            if (result == VK_SUCCESS)
                return result;
            if (result != VK_INCOMPLETE)
                return result;
        }
        return VK_INCOMPLETE;
    }

    template <typename T>
    T loadInstanceProc(PFN_vkGetInstanceProcAddr getInstanceProcAddr,
                       VkInstance instance,
                       const char *name) noexcept
    {
        return reinterpret_cast<T>(getInstanceProcAddr(instance, name));
    }

    bool openVulkanLoader(const GsVulkanProbeConfig &config,
                          DynamicLibrary &library,
                          std::string &error)
    {
        if (!config.loaderPath.empty())
        {
            if (library.open(config.loaderPath, error))
                return true;
            error = "failed to load explicit Vulkan loader '" +
                    config.loaderPath + "': " + error;
            return false;
        }

#if defined(_WIN32)
        const char *candidates[] = {"vulkan-1.dll"};
#elif defined(__APPLE__)
        const char *candidates[] = {
            "libvulkan.1.dylib",
            "libvulkan.dylib",
            "libMoltenVK.dylib",
        };
#else
        const char *candidates[] = {
            "libvulkan.so.1",
            "libvulkan.so",
        };
#endif
        std::string lastError;
        for (const char *candidate : candidates)
        {
            if (library.open(candidate, lastError))
                return true;
        }
        error = "no Vulkan loader could be opened";
        if (!lastError.empty())
            error += ": " + lastError;
        return false;
    }

    GsVulkanDeviceKind deviceKind(VkPhysicalDeviceType type) noexcept
    {
        switch (type)
        {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return GsVulkanDeviceKind::IntegratedGpu;
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return GsVulkanDeviceKind::DiscreteGpu;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return GsVulkanDeviceKind::VirtualGpu;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            return GsVulkanDeviceKind::Cpu;
        default:
            return GsVulkanDeviceKind::Other;
        }
    }

    int deviceScore(const GsVulkanDeviceReport &device) noexcept
    {
        int score = 0;
        switch (device.kind)
        {
        case GsVulkanDeviceKind::DiscreteGpu:
            score = 400;
            break;
        case GsVulkanDeviceKind::IntegratedGpu:
            score = 300;
            break;
        case GsVulkanDeviceKind::VirtualGpu:
            score = 200;
            break;
        case GsVulkanDeviceKind::Other:
            score = 100;
            break;
        case GsVulkanDeviceKind::Cpu:
            break;
        }
        if (device.dedicatedComputeQueue)
            score += 10;
        return score;
    }

    bool deviceMatchesPreference(const GsVulkanDeviceReport &device,
                                 const GsVulkanProbeConfig &config) noexcept
    {
        return (config.preferredVendorId == 0u ||
                device.vendorId == config.preferredVendorId) &&
               (config.preferredDeviceId == 0u ||
                device.deviceId == config.preferredDeviceId);
    }

    int32_t selectDevice(const std::vector<GsVulkanDeviceReport> &devices,
                         const GsVulkanProbeConfig &config)
    {
        int32_t selected = -1;
        int selectedScore = -1;
        for (size_t index = 0u; index < devices.size(); ++index)
        {
            const GsVulkanDeviceReport &candidate = devices[index];
            if (!candidate.suitable ||
                !deviceMatchesPreference(candidate, config))
            {
                continue;
            }

            const int score = deviceScore(candidate);
            bool replace = score > selectedScore;
            if (!replace && score == selectedScore && selected >= 0)
            {
                const GsVulkanDeviceReport &current =
                    devices[static_cast<size_t>(selected)];
                replace = std::tie(candidate.vendorId,
                                   candidate.deviceId,
                                   candidate.name) <
                          std::tie(current.vendorId,
                                   current.deviceId,
                                   current.name);
            }
            if (replace)
            {
                selected = static_cast<int32_t>(index);
                selectedScore = score;
            }
        }
        return selected;
    }

    struct InstanceOwner
    {
        VkInstance instance = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
        PFN_vkDestroyDebugUtilsMessengerEXT destroyMessenger = nullptr;
        PFN_vkDestroyInstance destroyInstance = nullptr;

        void reset() noexcept
        {
            if (messenger != VK_NULL_HANDLE && destroyMessenger)
                destroyMessenger(instance, messenger, nullptr);
            messenger = VK_NULL_HANDLE;
            if (instance != VK_NULL_HANDLE && destroyInstance)
                destroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }

        ~InstanceOwner()
        {
            reset();
        }
    };
}
#endif

GsVulkanCapabilityReport probeGsVulkanCapabilities(
    const GsVulkanProbeConfig &config)
{
    GsVulkanCapabilityReport report{};
    report.validationRequested = config.enableValidation;

#if !PS2X_HAS_GS_VULKAN
    report.status = GsVulkanProbeStatus::CompiledOut;
    report.detail = "Vulkan support was disabled or its headers were unavailable at build time";
    return report;
#else
    report.compiled = true;

    DynamicLibrary library;
    std::string loaderError;
    if (!openVulkanLoader(config, library, loaderError))
    {
        report.status = GsVulkanProbeStatus::LoaderUnavailable;
        report.detail = std::move(loaderError);
        return report;
    }
    report.loaderAvailable = true;

    const auto getInstanceProcAddr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            library.symbol("vkGetInstanceProcAddr"));
    if (!getInstanceProcAddr)
    {
        report.status = GsVulkanProbeStatus::LoaderInvalid;
        report.detail = "Vulkan loader does not export vkGetInstanceProcAddr";
        return report;
    }

    const auto enumerateInstanceVersion =
        loadInstanceProc<PFN_vkEnumerateInstanceVersion>(
            getInstanceProcAddr, VK_NULL_HANDLE,
            "vkEnumerateInstanceVersion");
    report.loaderApiVersion = VK_API_VERSION_1_0;
    if (enumerateInstanceVersion)
    {
        const VkResult versionResult =
            enumerateInstanceVersion(&report.loaderApiVersion);
        if (versionResult != VK_SUCCESS)
        {
            report.status = GsVulkanProbeStatus::LoaderInvalid;
            report.detail = "vkEnumerateInstanceVersion failed: " +
                            resultDescription(versionResult);
            return report;
        }
    }

    const auto enumerateLayers =
        loadInstanceProc<PFN_vkEnumerateInstanceLayerProperties>(
            getInstanceProcAddr, VK_NULL_HANDLE,
            "vkEnumerateInstanceLayerProperties");
    const auto enumerateExtensions =
        loadInstanceProc<PFN_vkEnumerateInstanceExtensionProperties>(
            getInstanceProcAddr, VK_NULL_HANDLE,
            "vkEnumerateInstanceExtensionProperties");
    const auto createInstance =
        loadInstanceProc<PFN_vkCreateInstance>(
            getInstanceProcAddr, VK_NULL_HANDLE, "vkCreateInstance");
    if (!enumerateLayers || !enumerateExtensions || !createInstance)
    {
        report.status = GsVulkanProbeStatus::LoaderInvalid;
        report.detail = "Vulkan loader is missing required global entry points";
        return report;
    }

    std::vector<VkLayerProperties> layers;
    VkResult result = enumerateValues<VkLayerProperties>(
        [&](uint32_t *count, VkLayerProperties *values)
        {
            return enumerateLayers(count, values);
        },
        layers);
    if (result != VK_SUCCESS)
    {
        report.status = GsVulkanProbeStatus::LoaderInvalid;
        report.detail = "layer enumeration failed: " +
                        resultDescription(result);
        return report;
    }
    report.validationLayerAvailable = std::any_of(
        layers.begin(), layers.end(), [](const VkLayerProperties &layer)
        {
            return std::strcmp(layer.layerName,
                               "VK_LAYER_KHRONOS_validation") == 0;
        });

    std::vector<VkExtensionProperties> extensions;
    result = enumerateValues<VkExtensionProperties>(
        [&](uint32_t *count, VkExtensionProperties *values)
        {
            return enumerateExtensions(nullptr, count, values);
        },
        extensions);
    if (result != VK_SUCCESS)
    {
        report.status = GsVulkanProbeStatus::LoaderInvalid;
        report.detail = "instance-extension enumeration failed: " +
                        resultDescription(result);
        return report;
    }
    report.debugUtilsAvailable = std::any_of(
        extensions.begin(), extensions.end(),
        [](const VkExtensionProperties &extension)
        {
            return std::strcmp(extension.extensionName,
                               VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0;
        });
#if defined(VK_KHR_portability_enumeration)
    const bool portabilityEnumerationAvailable = std::any_of(
        extensions.begin(), extensions.end(),
        [](const VkExtensionProperties &extension)
        {
            return std::strcmp(
                       extension.extensionName,
                       VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
        });
#endif

    if (config.enableValidation &&
        (!report.validationLayerAvailable || !report.debugUtilsAvailable))
    {
        report.status = GsVulkanProbeStatus::ValidationUnavailable;
        report.detail = "validation requested but the Khronos layer or debug-utils extension is unavailable";
        return report;
    }

    ValidationCounters validationCounters;
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo =
        makeDebugCreateInfo(validationCounters);
    const char *validationLayer = "VK_LAYER_KHRONOS_validation";
    std::vector<const char *> enabledExtensions;
    if (config.enableValidation)
        enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#if defined(VK_KHR_portability_enumeration)
    if (portabilityEnumerationAvailable)
    {
        enabledExtensions.push_back(
            VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }
#endif

    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "PS2Recomp GS capability probe";
    applicationInfo.applicationVersion = 1u;
    applicationInfo.pEngineName = "PS2Recomp";
    applicationInfo.engineVersion = 1u;
    applicationInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &applicationInfo;
    if (config.enableValidation)
    {
        instanceInfo.pNext = &debugCreateInfo;
        instanceInfo.enabledLayerCount = 1u;
        instanceInfo.ppEnabledLayerNames = &validationLayer;
    }
    instanceInfo.enabledExtensionCount =
        static_cast<uint32_t>(enabledExtensions.size());
    instanceInfo.ppEnabledExtensionNames = enabledExtensions.empty()
        ? nullptr
        : enabledExtensions.data();
#if defined(VK_KHR_portability_enumeration)
    if (portabilityEnumerationAvailable)
    {
        instanceInfo.flags |=
            VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
#endif

    InstanceOwner owner;
    result = createInstance(&instanceInfo, nullptr, &owner.instance);
    if (result != VK_SUCCESS)
    {
        report.status = GsVulkanProbeStatus::InstanceCreationFailed;
        report.detail = "vkCreateInstance failed: " +
                        resultDescription(result);
        report.validationWarnings =
            validationCounters.warnings.load(std::memory_order_relaxed);
        report.validationErrors =
            validationCounters.errors.load(std::memory_order_relaxed);
        return report;
    }

    owner.destroyInstance = loadInstanceProc<PFN_vkDestroyInstance>(
        getInstanceProcAddr, owner.instance, "vkDestroyInstance");
    if (!owner.destroyInstance)
    {
        report.status = GsVulkanProbeStatus::LoaderInvalid;
        report.detail = "Vulkan loader is missing vkDestroyInstance";
        return report;
    }

    auto finish = [&](GsVulkanProbeStatus status,
                      std::string detail) mutable
    {
        report.status = status;
        report.detail = std::move(detail);
        owner.reset();
        report.validationWarnings =
            validationCounters.warnings.load(std::memory_order_relaxed);
        report.validationErrors =
            validationCounters.errors.load(std::memory_order_relaxed);
        if (report.status == GsVulkanProbeStatus::Ready &&
            report.validationErrors != 0u)
        {
            report.status = GsVulkanProbeStatus::ValidationError;
            report.detail = "Vulkan validation reported errors during capability discovery";
            report.selectedDeviceIndex = -1;
        }
        return std::move(report);
    };

    if (config.enableValidation)
    {
        const auto createDebugMessenger =
            loadInstanceProc<PFN_vkCreateDebugUtilsMessengerEXT>(
                getInstanceProcAddr, owner.instance,
                "vkCreateDebugUtilsMessengerEXT");
        owner.destroyMessenger =
            loadInstanceProc<PFN_vkDestroyDebugUtilsMessengerEXT>(
                getInstanceProcAddr, owner.instance,
                "vkDestroyDebugUtilsMessengerEXT");
        if (!createDebugMessenger || !owner.destroyMessenger)
        {
            return finish(
                GsVulkanProbeStatus::LoaderInvalid,
                "debug-utils extension entry points are unavailable");
        }
        result = createDebugMessenger(
            owner.instance, &debugCreateInfo, nullptr, &owner.messenger);
        if (result != VK_SUCCESS)
        {
            return finish(
                GsVulkanProbeStatus::InstanceCreationFailed,
                "debug messenger creation failed: " +
                    resultDescription(result));
        }
        report.validationEnabled = true;
    }

    const auto enumeratePhysicalDevices =
        loadInstanceProc<PFN_vkEnumeratePhysicalDevices>(
            getInstanceProcAddr, owner.instance,
            "vkEnumeratePhysicalDevices");
    const auto getPhysicalDeviceProperties =
        loadInstanceProc<PFN_vkGetPhysicalDeviceProperties>(
            getInstanceProcAddr, owner.instance,
            "vkGetPhysicalDeviceProperties");
    const auto getPhysicalDeviceFeatures =
        loadInstanceProc<PFN_vkGetPhysicalDeviceFeatures>(
            getInstanceProcAddr, owner.instance,
            "vkGetPhysicalDeviceFeatures");
    const auto getQueueFamilyProperties =
        loadInstanceProc<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            getInstanceProcAddr, owner.instance,
            "vkGetPhysicalDeviceQueueFamilyProperties");
    const auto getMemoryProperties =
        loadInstanceProc<PFN_vkGetPhysicalDeviceMemoryProperties>(
            getInstanceProcAddr, owner.instance,
            "vkGetPhysicalDeviceMemoryProperties");
    if (!enumeratePhysicalDevices || !getPhysicalDeviceProperties ||
        !getPhysicalDeviceFeatures || !getQueueFamilyProperties ||
        !getMemoryProperties)
    {
        return finish(
            GsVulkanProbeStatus::LoaderInvalid,
            "Vulkan instance is missing required physical-device entry points");
    }

    std::vector<VkPhysicalDevice> physicalDevices;
    result = enumerateValues<VkPhysicalDevice>(
        [&](uint32_t *count, VkPhysicalDevice *values)
        {
            return enumeratePhysicalDevices(owner.instance, count, values);
        },
        physicalDevices);
    if (result != VK_SUCCESS)
    {
        return finish(
            GsVulkanProbeStatus::DeviceEnumerationFailed,
            "physical-device enumeration failed: " +
                resultDescription(result));
    }
    if (physicalDevices.empty())
    {
        return finish(
            GsVulkanProbeStatus::NoPhysicalDevices,
            "Vulkan loader exposed no physical devices");
    }

    report.devices.reserve(physicalDevices.size());
    for (VkPhysicalDevice physicalDevice : physicalDevices)
    {
        VkPhysicalDeviceProperties properties{};
        VkPhysicalDeviceFeatures features{};
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        getPhysicalDeviceProperties(physicalDevice, &properties);
        getPhysicalDeviceFeatures(physicalDevice, &features);
        getMemoryProperties(physicalDevice, &memoryProperties);

        uint32_t queueCount = 0u;
        getQueueFamilyProperties(physicalDevice, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queueCount);
        if (queueCount != 0u)
        {
            getQueueFamilyProperties(
                physicalDevice, &queueCount, queues.data());
            queues.resize(queueCount);
        }

        GsVulkanDeviceReport device{};
        device.name = properties.deviceName;
        device.vendorId = properties.vendorID;
        device.deviceId = properties.deviceID;
        device.apiVersion = properties.apiVersion;
        device.driverVersion = properties.driverVersion;
        device.kind = deviceKind(properties.deviceType);
        device.maxStorageBufferRange =
            properties.limits.maxStorageBufferRange;
        device.maxComputeWorkGroupCountX =
            properties.limits.maxComputeWorkGroupCount[0];
        device.maxComputeWorkGroupInvocations =
            properties.limits.maxComputeWorkGroupInvocations;
        device.maxComputeWorkGroupSizeX =
            properties.limits.maxComputeWorkGroupSize[0];
        device.shaderInt16 = features.shaderInt16 == VK_TRUE;
        device.shaderInt64 = features.shaderInt64 == VK_TRUE;

        for (uint32_t index = 0u; index < queues.size(); ++index)
        {
            const VkQueueFlags flags = queues[index].queueFlags;
            if (queues[index].queueCount == 0u ||
                (flags & VK_QUEUE_COMPUTE_BIT) == 0u)
            {
                continue;
            }
            const bool dedicated = (flags & VK_QUEUE_GRAPHICS_BIT) == 0u;
            if (!device.computeQueue || dedicated)
            {
                device.computeQueue = true;
                device.dedicatedComputeQueue = dedicated;
                device.queueFamilyIndex = index;
            }
            if (dedicated)
                break;
        }

        for (uint32_t index = 0u;
             index < memoryProperties.memoryTypeCount;
             ++index)
        {
            const VkMemoryType &type = memoryProperties.memoryTypes[index];
            if (type.heapIndex >= memoryProperties.memoryHeapCount ||
                memoryProperties.memoryHeaps[type.heapIndex].size <
                    GS_VULKAN_VRAM_SIZE)
            {
                continue;
            }
            device.deviceLocalMemory =
                device.deviceLocalMemory ||
                (type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0u;
            device.hostVisibleMemory =
                device.hostVisibleMemory ||
                (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0u;
        }

        const bool storageRange =
            device.maxStorageBufferRange >= GS_VULKAN_VRAM_SIZE;
        const bool dispatchLimits =
            device.maxComputeWorkGroupCountX >=
                GS_VULKAN_NOOP_GROUP_COUNT &&
            device.maxComputeWorkGroupInvocations >=
                GS_VULKAN_NOOP_LOCAL_SIZE &&
            device.maxComputeWorkGroupSizeX >=
                GS_VULKAN_NOOP_LOCAL_SIZE;
        device.exactVramStorage =
            device.computeQueue && storageRange && dispatchLimits &&
            device.deviceLocalMemory && device.hostVisibleMemory;
        device.suitable =
            device.exactVramStorage &&
            device.kind != GsVulkanDeviceKind::Cpu;

        if (device.kind == GsVulkanDeviceKind::Cpu)
            device.rejectionReason = "CPU Vulkan implementations are not hardware-GS targets";
        else if (!device.computeQueue)
            device.rejectionReason = "no compute-capable queue family";
        else if (!storageRange)
            device.rejectionReason = "maxStorageBufferRange is smaller than 4 MiB";
        else if (!dispatchLimits)
            device.rejectionReason = "compute dispatch limits cannot cover 4 MiB with the fixed kernel";
        else if (!device.deviceLocalMemory)
            device.rejectionReason = "no device-local memory heap can hold 4 MiB";
        else if (!device.hostVisibleMemory)
            device.rejectionReason = "no host-visible staging heap can hold 4 MiB";

        report.devices.push_back(std::move(device));
    }

    report.selectedDeviceIndex = selectDevice(report.devices, config);
    if (report.selectedDeviceIndex < 0)
    {
        const bool preferenceSpecified =
            config.preferredVendorId != 0u ||
            config.preferredDeviceId != 0u;
        return finish(
            GsVulkanProbeStatus::NoSuitableDevice,
            preferenceSpecified
                ? "no suitable Vulkan device matches the required vendor/device override"
                : "no hardware Vulkan device satisfies the raw 4 MiB storage requirements");
    }

    const GsVulkanDeviceReport *selected = report.selectedDevice();
    return finish(
        GsVulkanProbeStatus::Ready,
        selected
            ? "selected " + selected->name
            : "selected a suitable Vulkan device");
#endif
}
