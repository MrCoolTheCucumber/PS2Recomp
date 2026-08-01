#include "runtime/ps2_gs_vulkan.h"
#include "runtime/ps2_gs_memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <sstream>
#include <thread>
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
    case GsVulkanProbeStatus::DeviceCreationFailed:
        return "device-creation-failed";
    case GsVulkanProbeStatus::ResourceCreationFailed:
        return "resource-creation-failed";
    case GsVulkanProbeStatus::ExecutionTimeout:
        return "execution-timeout";
    case GsVulkanProbeStatus::DeviceLost:
        return "device-lost";
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

namespace
{
    const char *ct32SpriteValidationError(
        const GsVulkanCt32Sprite &sprite) noexcept
    {
        if (sprite.reserved != 0u)
            return "Vulkan CT32 sprite has non-zero reserved data";
        if (sprite.framebufferBaseBlock > 0x3FFFu)
            return "Vulkan CT32 sprite framebuffer base is outside GS VRAM";
        if (sprite.framebufferWidth == 0u ||
            sprite.framebufferWidth > 0x3Fu)
            return "Vulkan CT32 sprite framebuffer width is outside FRAME range";
        if (sprite.x0 >= sprite.x1 || sprite.y0 >= sprite.y1)
            return "Vulkan CT32 sprite bounds are empty";
        if (sprite.x1 > 2048u || sprite.y1 > 2048u)
            return "Vulkan CT32 sprite bounds are outside GS scissor range";
        return nullptr;
    }

    struct GsVulkanElementParameters
    {
        uint32_t elementCount;
        uint32_t reserved;
    };

    static_assert(sizeof(GsVulkanElementParameters) == 8u);
}

GsBackendDecision prepareGsVulkanCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanCt32Sprite &sprite) noexcept
{
    const GsBackendDecision decision =
        classifyGsInitialCt32Sprite(command);
    if (!decision.supported)
        return decision;

    const GSContext &context = command.context();
    const GsDrawBounds &bounds = command.bounds();
    const GSVertex &color = command.vertices()[1];
    GsVulkanCt32Sprite prepared{};
    prepared.framebufferBaseBlock = context.frame.fbp << 5u;
    prepared.framebufferWidth =
        std::max<uint32_t>(context.frame.fbw, 1u);
    prepared.x0 = static_cast<uint32_t>(bounds.x0);
    prepared.y0 = static_cast<uint32_t>(bounds.y0);
    prepared.x1 = static_cast<uint32_t>(bounds.x1);
    prepared.y1 = static_cast<uint32_t>(bounds.y1);
    prepared.rgba =
        static_cast<uint32_t>(color.r) |
        (static_cast<uint32_t>(color.g) << 8u) |
        (static_cast<uint32_t>(color.b) << 16u) |
        (static_cast<uint32_t>(color.a) << 24u);
    if (ct32SpriteValidationError(prepared))
    {
        return {false, GsFallbackReason::UnknownMemoryLayout};
    }

    sprite = prepared;
    return decision;
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

#if PS2X_HAS_GS_VULKAN
namespace
{
#include "shaders/ps2_gs_ct32_sprite_spv.inc"
#include "shaders/ps2_gs_memory_cases_spv.inc"
#include "shaders/ps2_gs_vram_noop_spv.inc"

    static_assert(sizeof(kGsCt32SpriteShaderSpv) == 7536u);
    static_assert(kGsCt32SpriteShaderSpv[0] == 0x07230203u);
    static_assert(sizeof(kGsMemoryCasesShaderSpv) == 10988u);
    static_assert(kGsMemoryCasesShaderSpv[0] == 0x07230203u);
    static_assert(sizeof(kGsVramNoopShaderSpv) == 1112u);
    static_assert(kGsVramNoopShaderSpv[0] == 0x07230203u);

    template <typename T>
    T loadDeviceProc(PFN_vkGetDeviceProcAddr getDeviceProcAddr,
                     VkDevice device,
                     const char *name) noexcept
    {
        return reinterpret_cast<T>(getDeviceProcAddr(device, name));
    }

    struct DeviceFunctions
    {
        PFN_vkDestroyDevice destroyDevice = nullptr;
        PFN_vkDeviceWaitIdle deviceWaitIdle = nullptr;
        PFN_vkGetDeviceQueue getDeviceQueue = nullptr;
        PFN_vkCreateBuffer createBuffer = nullptr;
        PFN_vkDestroyBuffer destroyBuffer = nullptr;
        PFN_vkGetBufferMemoryRequirements getBufferMemoryRequirements = nullptr;
        PFN_vkAllocateMemory allocateMemory = nullptr;
        PFN_vkFreeMemory freeMemory = nullptr;
        PFN_vkBindBufferMemory bindBufferMemory = nullptr;
        PFN_vkMapMemory mapMemory = nullptr;
        PFN_vkUnmapMemory unmapMemory = nullptr;
        PFN_vkFlushMappedMemoryRanges flushMappedMemoryRanges = nullptr;
        PFN_vkInvalidateMappedMemoryRanges invalidateMappedMemoryRanges = nullptr;
        PFN_vkCreateDescriptorSetLayout createDescriptorSetLayout = nullptr;
        PFN_vkDestroyDescriptorSetLayout destroyDescriptorSetLayout = nullptr;
        PFN_vkCreatePipelineLayout createPipelineLayout = nullptr;
        PFN_vkDestroyPipelineLayout destroyPipelineLayout = nullptr;
        PFN_vkCreateShaderModule createShaderModule = nullptr;
        PFN_vkDestroyShaderModule destroyShaderModule = nullptr;
        PFN_vkCreateComputePipelines createComputePipelines = nullptr;
        PFN_vkDestroyPipeline destroyPipeline = nullptr;
        PFN_vkCreateDescriptorPool createDescriptorPool = nullptr;
        PFN_vkDestroyDescriptorPool destroyDescriptorPool = nullptr;
        PFN_vkAllocateDescriptorSets allocateDescriptorSets = nullptr;
        PFN_vkUpdateDescriptorSets updateDescriptorSets = nullptr;
        PFN_vkCreateCommandPool createCommandPool = nullptr;
        PFN_vkDestroyCommandPool destroyCommandPool = nullptr;
        PFN_vkAllocateCommandBuffers allocateCommandBuffers = nullptr;
        PFN_vkResetCommandBuffer resetCommandBuffer = nullptr;
        PFN_vkBeginCommandBuffer beginCommandBuffer = nullptr;
        PFN_vkEndCommandBuffer endCommandBuffer = nullptr;
        PFN_vkCreateFence createFence = nullptr;
        PFN_vkDestroyFence destroyFence = nullptr;
        PFN_vkResetFences resetFences = nullptr;
        PFN_vkWaitForFences waitForFences = nullptr;
        PFN_vkQueueSubmit queueSubmit = nullptr;
        PFN_vkCmdPipelineBarrier cmdPipelineBarrier = nullptr;
        PFN_vkCmdCopyBuffer cmdCopyBuffer = nullptr;
        PFN_vkCmdBindPipeline cmdBindPipeline = nullptr;
        PFN_vkCmdBindDescriptorSets cmdBindDescriptorSets = nullptr;
        PFN_vkCmdPushConstants cmdPushConstants = nullptr;
        PFN_vkCmdDispatch cmdDispatch = nullptr;

        bool load(PFN_vkGetDeviceProcAddr getDeviceProcAddr,
                  VkDevice device,
                  std::string &error)
        {
#define PS2X_LOAD_DEVICE_PROC(member, type, name)                         \
            member = loadDeviceProc<type>(                               \
                getDeviceProcAddr, device, #name);                       \
            if (!member)                                                 \
            {                                                            \
                error = "Vulkan device is missing " #name;              \
                return false;                                            \
            }
            PS2X_LOAD_DEVICE_PROC(destroyDevice, PFN_vkDestroyDevice,
                                  vkDestroyDevice)
            PS2X_LOAD_DEVICE_PROC(deviceWaitIdle, PFN_vkDeviceWaitIdle,
                                  vkDeviceWaitIdle)
            PS2X_LOAD_DEVICE_PROC(getDeviceQueue, PFN_vkGetDeviceQueue,
                                  vkGetDeviceQueue)
            PS2X_LOAD_DEVICE_PROC(createBuffer, PFN_vkCreateBuffer,
                                  vkCreateBuffer)
            PS2X_LOAD_DEVICE_PROC(destroyBuffer, PFN_vkDestroyBuffer,
                                  vkDestroyBuffer)
            PS2X_LOAD_DEVICE_PROC(
                getBufferMemoryRequirements,
                PFN_vkGetBufferMemoryRequirements,
                vkGetBufferMemoryRequirements)
            PS2X_LOAD_DEVICE_PROC(allocateMemory, PFN_vkAllocateMemory,
                                  vkAllocateMemory)
            PS2X_LOAD_DEVICE_PROC(freeMemory, PFN_vkFreeMemory,
                                  vkFreeMemory)
            PS2X_LOAD_DEVICE_PROC(bindBufferMemory, PFN_vkBindBufferMemory,
                                  vkBindBufferMemory)
            PS2X_LOAD_DEVICE_PROC(mapMemory, PFN_vkMapMemory, vkMapMemory)
            PS2X_LOAD_DEVICE_PROC(unmapMemory, PFN_vkUnmapMemory,
                                  vkUnmapMemory)
            PS2X_LOAD_DEVICE_PROC(
                flushMappedMemoryRanges,
                PFN_vkFlushMappedMemoryRanges,
                vkFlushMappedMemoryRanges)
            PS2X_LOAD_DEVICE_PROC(
                invalidateMappedMemoryRanges,
                PFN_vkInvalidateMappedMemoryRanges,
                vkInvalidateMappedMemoryRanges)
            PS2X_LOAD_DEVICE_PROC(
                createDescriptorSetLayout,
                PFN_vkCreateDescriptorSetLayout,
                vkCreateDescriptorSetLayout)
            PS2X_LOAD_DEVICE_PROC(
                destroyDescriptorSetLayout,
                PFN_vkDestroyDescriptorSetLayout,
                vkDestroyDescriptorSetLayout)
            PS2X_LOAD_DEVICE_PROC(createPipelineLayout,
                                  PFN_vkCreatePipelineLayout,
                                  vkCreatePipelineLayout)
            PS2X_LOAD_DEVICE_PROC(destroyPipelineLayout,
                                  PFN_vkDestroyPipelineLayout,
                                  vkDestroyPipelineLayout)
            PS2X_LOAD_DEVICE_PROC(createShaderModule,
                                  PFN_vkCreateShaderModule,
                                  vkCreateShaderModule)
            PS2X_LOAD_DEVICE_PROC(destroyShaderModule,
                                  PFN_vkDestroyShaderModule,
                                  vkDestroyShaderModule)
            PS2X_LOAD_DEVICE_PROC(createComputePipelines,
                                  PFN_vkCreateComputePipelines,
                                  vkCreateComputePipelines)
            PS2X_LOAD_DEVICE_PROC(destroyPipeline, PFN_vkDestroyPipeline,
                                  vkDestroyPipeline)
            PS2X_LOAD_DEVICE_PROC(createDescriptorPool,
                                  PFN_vkCreateDescriptorPool,
                                  vkCreateDescriptorPool)
            PS2X_LOAD_DEVICE_PROC(destroyDescriptorPool,
                                  PFN_vkDestroyDescriptorPool,
                                  vkDestroyDescriptorPool)
            PS2X_LOAD_DEVICE_PROC(allocateDescriptorSets,
                                  PFN_vkAllocateDescriptorSets,
                                  vkAllocateDescriptorSets)
            PS2X_LOAD_DEVICE_PROC(updateDescriptorSets,
                                  PFN_vkUpdateDescriptorSets,
                                  vkUpdateDescriptorSets)
            PS2X_LOAD_DEVICE_PROC(createCommandPool,
                                  PFN_vkCreateCommandPool,
                                  vkCreateCommandPool)
            PS2X_LOAD_DEVICE_PROC(destroyCommandPool,
                                  PFN_vkDestroyCommandPool,
                                  vkDestroyCommandPool)
            PS2X_LOAD_DEVICE_PROC(allocateCommandBuffers,
                                  PFN_vkAllocateCommandBuffers,
                                  vkAllocateCommandBuffers)
            PS2X_LOAD_DEVICE_PROC(resetCommandBuffer,
                                  PFN_vkResetCommandBuffer,
                                  vkResetCommandBuffer)
            PS2X_LOAD_DEVICE_PROC(beginCommandBuffer,
                                  PFN_vkBeginCommandBuffer,
                                  vkBeginCommandBuffer)
            PS2X_LOAD_DEVICE_PROC(endCommandBuffer,
                                  PFN_vkEndCommandBuffer,
                                  vkEndCommandBuffer)
            PS2X_LOAD_DEVICE_PROC(createFence, PFN_vkCreateFence,
                                  vkCreateFence)
            PS2X_LOAD_DEVICE_PROC(destroyFence, PFN_vkDestroyFence,
                                  vkDestroyFence)
            PS2X_LOAD_DEVICE_PROC(resetFences, PFN_vkResetFences,
                                  vkResetFences)
            PS2X_LOAD_DEVICE_PROC(waitForFences, PFN_vkWaitForFences,
                                  vkWaitForFences)
            PS2X_LOAD_DEVICE_PROC(queueSubmit, PFN_vkQueueSubmit,
                                  vkQueueSubmit)
            PS2X_LOAD_DEVICE_PROC(cmdPipelineBarrier,
                                  PFN_vkCmdPipelineBarrier,
                                  vkCmdPipelineBarrier)
            PS2X_LOAD_DEVICE_PROC(cmdCopyBuffer, PFN_vkCmdCopyBuffer,
                                  vkCmdCopyBuffer)
            PS2X_LOAD_DEVICE_PROC(cmdBindPipeline,
                                  PFN_vkCmdBindPipeline,
                                  vkCmdBindPipeline)
            PS2X_LOAD_DEVICE_PROC(cmdBindDescriptorSets,
                                  PFN_vkCmdBindDescriptorSets,
                                  vkCmdBindDescriptorSets)
            PS2X_LOAD_DEVICE_PROC(cmdPushConstants,
                                  PFN_vkCmdPushConstants,
                                  vkCmdPushConstants)
            PS2X_LOAD_DEVICE_PROC(cmdDispatch, PFN_vkCmdDispatch,
                                  vkCmdDispatch)
#undef PS2X_LOAD_DEVICE_PROC
            return true;
        }
    };

    struct BufferAllocation
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize allocationSize = 0u;
        bool coherent = false;
    };

    class VulkanExecutionContext final
    {
    public:
        VulkanExecutionContext() = default;
        ~VulkanExecutionContext()
        {
            shutdown();
        }

        VulkanExecutionContext(const VulkanExecutionContext &) = delete;
        VulkanExecutionContext &operator=(
            const VulkanExecutionContext &) = delete;

        bool initialize(const GsVulkanServiceConfig &config,
                        GsVulkanCapabilityReport &report,
                        std::string &error);
        bool roundTrip(std::span<const uint8_t> input,
                       std::vector<uint8_t> &output,
                       GsVulkanCapabilityReport &report,
                       GsVulkanServiceStatistics &statistics,
                       std::string &error);
        bool executeMemoryCases(
            std::span<const uint8_t> input,
            std::span<const GsVulkanMemoryCase> cases,
            std::vector<uint8_t> &output,
            std::vector<GsVulkanMemoryResult> &results,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeCt32Sprite(
            std::span<const uint8_t> input,
            const GsVulkanCt32Sprite &sprite,
            std::vector<uint8_t> &output,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        void refreshDiagnostics(GsVulkanCapabilityReport &report,
                                GsVulkanServiceStatistics &statistics) const;
        void shutdown() noexcept;
        [[nodiscard]] bool healthy() const noexcept
        {
            return m_healthy;
        }

    private:
        bool createInstance(const GsVulkanProbeConfig &config,
                            GsVulkanCapabilityReport &report,
                            std::string &error);
        bool createDevice(const GsVulkanDeviceReport &selected,
                          GsVulkanCapabilityReport &report,
                          std::string &error);
        bool createResources(GsVulkanCapabilityReport &report,
                             std::string &error);
        bool createBuffer(VkDeviceSize size,
                          VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags required,
                          VkMemoryPropertyFlags preferred,
                          BufferAllocation &allocation,
                          GsVulkanCapabilityReport &report,
                          std::string &error);
        bool executeKernel(
            std::span<const uint8_t> input,
            std::span<const GsVulkanMemoryCase> cases,
            VkPipeline pipeline,
            uint32_t groupCountX,
            uint32_t groupCountY,
            const void *pushConstants,
            uint32_t pushConstantSize,
            std::vector<uint8_t> &output,
            std::vector<GsVulkanMemoryResult> *results,
            std::string_view operationName,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        int32_t findMemoryType(uint32_t typeBits,
                               VkMemoryPropertyFlags required,
                               VkMemoryPropertyFlags preferred) const noexcept;
        bool checkResult(VkResult result,
                         std::string_view operation,
                         GsVulkanCapabilityReport &report,
                         std::string &error,
                         GsVulkanProbeStatus failureStatus);

        DynamicLibrary m_library;
        ValidationCounters m_validation;
        InstanceOwner m_instance;
        PFN_vkGetInstanceProcAddr m_getInstanceProcAddr = nullptr;
        PFN_vkGetDeviceProcAddr m_getDeviceProcAddr = nullptr;
        VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
        VkPhysicalDeviceMemoryProperties m_memoryProperties{};
        VkDevice m_device = VK_NULL_HANDLE;
        DeviceFunctions m_functions{};
        VkQueue m_queue = VK_NULL_HANDLE;
        uint32_t m_queueFamilyIndex = 0u;
        BufferAllocation m_vram;
        BufferAllocation m_staging;
        BufferAllocation m_memoryCases;
        BufferAllocation m_memoryResults;
        void *m_stagingMap = nullptr;
        void *m_memoryCasesMap = nullptr;
        void *m_memoryResultsMap = nullptr;
        VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
        VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
        VkShaderModule m_noopShaderModule = VK_NULL_HANDLE;
        VkPipeline m_noopPipeline = VK_NULL_HANDLE;
        VkShaderModule m_memoryShaderModule = VK_NULL_HANDLE;
        VkPipeline m_memoryPipeline = VK_NULL_HANDLE;
        VkShaderModule m_spriteShaderModule = VK_NULL_HANDLE;
        VkPipeline m_spritePipeline = VK_NULL_HANDLE;
        VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
        VkCommandPool m_commandPool = VK_NULL_HANDLE;
        VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
        VkFence m_fence = VK_NULL_HANDLE;
        uint64_t m_fenceTimeoutNanoseconds = 0u;
        bool m_healthy = false;
    };

    bool VulkanExecutionContext::checkResult(
        VkResult result,
        std::string_view operation,
        GsVulkanCapabilityReport &report,
        std::string &error,
        GsVulkanProbeStatus failureStatus)
    {
        if (result == VK_SUCCESS)
            return true;
        error = std::string(operation) + " failed: " +
                resultDescription(result);
        report.status = result == VK_ERROR_DEVICE_LOST
            ? GsVulkanProbeStatus::DeviceLost
            : failureStatus;
        report.detail = error;
        if (result == VK_ERROR_DEVICE_LOST)
            m_healthy = false;
        return false;
    }

    bool VulkanExecutionContext::createInstance(
        const GsVulkanProbeConfig &config,
        GsVulkanCapabilityReport &report,
        std::string &error)
    {
        if (!openVulkanLoader(config, m_library, error))
        {
            report.status = GsVulkanProbeStatus::LoaderUnavailable;
            report.detail = error;
            return false;
        }

        m_getInstanceProcAddr =
            reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                m_library.symbol("vkGetInstanceProcAddr"));
        if (!m_getInstanceProcAddr)
        {
            error = "Vulkan loader does not export vkGetInstanceProcAddr";
            report.status = GsVulkanProbeStatus::LoaderInvalid;
            report.detail = error;
            return false;
        }

        const auto enumerateExtensions =
            loadInstanceProc<PFN_vkEnumerateInstanceExtensionProperties>(
                m_getInstanceProcAddr, VK_NULL_HANDLE,
                "vkEnumerateInstanceExtensionProperties");
        const auto createInstanceProc =
            loadInstanceProc<PFN_vkCreateInstance>(
                m_getInstanceProcAddr, VK_NULL_HANDLE,
                "vkCreateInstance");
        if (!enumerateExtensions || !createInstanceProc)
        {
            error = "Vulkan loader is missing instance-creation entry points";
            report.status = GsVulkanProbeStatus::LoaderInvalid;
            report.detail = error;
            return false;
        }

        std::vector<VkExtensionProperties> extensions;
        const VkResult extensionResult =
            enumerateValues<VkExtensionProperties>(
                [&](uint32_t *count, VkExtensionProperties *values)
                {
                    return enumerateExtensions(nullptr, count, values);
                },
                extensions);
        if (!checkResult(extensionResult,
                         "instance-extension enumeration",
                         report, error,
                         GsVulkanProbeStatus::LoaderInvalid))
        {
            return false;
        }

        const bool debugUtilsAvailable = std::any_of(
            extensions.begin(), extensions.end(),
            [](const VkExtensionProperties &extension)
            {
                return std::strcmp(
                           extension.extensionName,
                           VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0;
            });
        if (config.enableValidation && !debugUtilsAvailable)
        {
            error = "validation requested but debug-utils is unavailable during service creation";
            report.status = GsVulkanProbeStatus::ValidationUnavailable;
            report.detail = error;
            return false;
        }

        std::vector<const char *> enabledExtensions;
        if (config.enableValidation)
            enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#if defined(VK_KHR_portability_enumeration)
        const bool portabilityEnumerationAvailable = std::any_of(
            extensions.begin(), extensions.end(),
            [](const VkExtensionProperties &extension)
            {
                return std::strcmp(
                           extension.extensionName,
                           VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
            });
        if (portabilityEnumerationAvailable)
        {
            enabledExtensions.push_back(
                VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        }
#endif

        VkApplicationInfo applicationInfo{};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "PS2Recomp GS Vulkan service";
        applicationInfo.applicationVersion = 1u;
        applicationInfo.pEngineName = "PS2Recomp";
        applicationInfo.engineVersion = 1u;
        applicationInfo.apiVersion = VK_API_VERSION_1_0;

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo =
            makeDebugCreateInfo(m_validation);
        const char *validationLayer = "VK_LAYER_KHRONOS_validation";
        VkInstanceCreateInfo instanceInfo{};
        instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceInfo.pApplicationInfo = &applicationInfo;
        instanceInfo.enabledExtensionCount =
            static_cast<uint32_t>(enabledExtensions.size());
        instanceInfo.ppEnabledExtensionNames = enabledExtensions.empty()
            ? nullptr
            : enabledExtensions.data();
        if (config.enableValidation)
        {
            instanceInfo.pNext = &debugCreateInfo;
            instanceInfo.enabledLayerCount = 1u;
            instanceInfo.ppEnabledLayerNames = &validationLayer;
        }
#if defined(VK_KHR_portability_enumeration)
        if (portabilityEnumerationAvailable)
        {
            instanceInfo.flags |=
                VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
#endif

        const VkResult instanceResult = createInstanceProc(
            &instanceInfo, nullptr, &m_instance.instance);
        if (!checkResult(instanceResult, "vkCreateInstance",
                         report, error,
                         GsVulkanProbeStatus::InstanceCreationFailed))
        {
            return false;
        }
        m_instance.destroyInstance =
            loadInstanceProc<PFN_vkDestroyInstance>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkDestroyInstance");
        if (!m_instance.destroyInstance)
        {
            error = "Vulkan instance is missing vkDestroyInstance";
            report.status = GsVulkanProbeStatus::LoaderInvalid;
            report.detail = error;
            return false;
        }

        if (config.enableValidation)
        {
            const auto createDebugMessenger =
                loadInstanceProc<PFN_vkCreateDebugUtilsMessengerEXT>(
                    m_getInstanceProcAddr, m_instance.instance,
                    "vkCreateDebugUtilsMessengerEXT");
            m_instance.destroyMessenger =
                loadInstanceProc<PFN_vkDestroyDebugUtilsMessengerEXT>(
                    m_getInstanceProcAddr, m_instance.instance,
                    "vkDestroyDebugUtilsMessengerEXT");
            if (!createDebugMessenger || !m_instance.destroyMessenger)
            {
                error = "Vulkan instance is missing debug-utils entry points";
                report.status = GsVulkanProbeStatus::LoaderInvalid;
                report.detail = error;
                return false;
            }
            const VkResult debugResult = createDebugMessenger(
                m_instance.instance, &debugCreateInfo, nullptr,
                &m_instance.messenger);
            if (!checkResult(debugResult, "debug messenger creation",
                             report, error,
                             GsVulkanProbeStatus::InstanceCreationFailed))
            {
                return false;
            }
            report.validationEnabled = true;
        }
        return true;
    }

    bool VulkanExecutionContext::createDevice(
        const GsVulkanDeviceReport &selected,
        GsVulkanCapabilityReport &report,
        std::string &error)
    {
        const auto enumeratePhysicalDevices =
            loadInstanceProc<PFN_vkEnumeratePhysicalDevices>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkEnumeratePhysicalDevices");
        const auto getPhysicalDeviceProperties =
            loadInstanceProc<PFN_vkGetPhysicalDeviceProperties>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkGetPhysicalDeviceProperties");
        const auto getQueueFamilyProperties =
            loadInstanceProc<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkGetPhysicalDeviceQueueFamilyProperties");
        const auto getMemoryProperties =
            loadInstanceProc<PFN_vkGetPhysicalDeviceMemoryProperties>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkGetPhysicalDeviceMemoryProperties");
        const auto enumerateDeviceExtensions =
            loadInstanceProc<PFN_vkEnumerateDeviceExtensionProperties>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkEnumerateDeviceExtensionProperties");
        const auto createDeviceProc =
            loadInstanceProc<PFN_vkCreateDevice>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkCreateDevice");
        m_getDeviceProcAddr =
            loadInstanceProc<PFN_vkGetDeviceProcAddr>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkGetDeviceProcAddr");
        if (!enumeratePhysicalDevices || !getPhysicalDeviceProperties ||
            !getQueueFamilyProperties || !getMemoryProperties ||
            !enumerateDeviceExtensions || !createDeviceProc ||
            !m_getDeviceProcAddr)
        {
            error = "Vulkan instance is missing logical-device entry points";
            report.status = GsVulkanProbeStatus::LoaderInvalid;
            report.detail = error;
            return false;
        }

        std::vector<VkPhysicalDevice> physicalDevices;
        const VkResult enumerateResult =
            enumerateValues<VkPhysicalDevice>(
                [&](uint32_t *count, VkPhysicalDevice *values)
                {
                    return enumeratePhysicalDevices(
                        m_instance.instance, count, values);
                },
                physicalDevices);
        if (!checkResult(enumerateResult,
                         "physical-device enumeration",
                         report, error,
                         GsVulkanProbeStatus::DeviceEnumerationFailed))
        {
            return false;
        }

        for (VkPhysicalDevice candidate : physicalDevices)
        {
            VkPhysicalDeviceProperties properties{};
            getPhysicalDeviceProperties(candidate, &properties);
            if (properties.vendorID == selected.vendorId &&
                properties.deviceID == selected.deviceId &&
                selected.name == properties.deviceName)
            {
                m_physicalDevice = candidate;
                break;
            }
        }
        if (m_physicalDevice == VK_NULL_HANDLE)
        {
            error = "the selected Vulkan device disappeared before service creation";
            report.status = GsVulkanProbeStatus::NoSuitableDevice;
            report.detail = error;
            return false;
        }

        uint32_t queueCount = 0u;
        getQueueFamilyProperties(m_physicalDevice, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queueCount);
        if (queueCount != 0u)
        {
            getQueueFamilyProperties(
                m_physicalDevice, &queueCount, queues.data());
            queues.resize(queueCount);
        }
        m_queueFamilyIndex = selected.queueFamilyIndex;
        if (m_queueFamilyIndex >= queues.size() ||
            queues[m_queueFamilyIndex].queueCount == 0u ||
            (queues[m_queueFamilyIndex].queueFlags &
             VK_QUEUE_COMPUTE_BIT) == 0u)
        {
            error = "the selected Vulkan compute queue is no longer available";
            report.status = GsVulkanProbeStatus::NoSuitableDevice;
            report.detail = error;
            return false;
        }

        std::vector<VkExtensionProperties> deviceExtensions;
        const VkResult extensionResult =
            enumerateValues<VkExtensionProperties>(
                [&](uint32_t *count, VkExtensionProperties *values)
                {
                    return enumerateDeviceExtensions(
                        m_physicalDevice, nullptr, count, values);
                },
                deviceExtensions);
        if (!checkResult(extensionResult,
                         "device-extension enumeration",
                         report, error,
                         GsVulkanProbeStatus::DeviceCreationFailed))
        {
            return false;
        }

        std::vector<const char *> enabledDeviceExtensions;
#if defined(VK_KHR_portability_subset)
        const bool portabilitySubsetAvailable = std::any_of(
            deviceExtensions.begin(), deviceExtensions.end(),
            [](const VkExtensionProperties &extension)
            {
                return std::strcmp(
                           extension.extensionName,
                           VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) == 0;
            });
        if (portabilitySubsetAvailable)
        {
            enabledDeviceExtensions.push_back(
                VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
        }
#endif

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = m_queueFamilyIndex;
        queueInfo.queueCount = 1u;
        queueInfo.pQueuePriorities = &priority;

        VkPhysicalDeviceFeatures enabledFeatures{};
        VkDeviceCreateInfo deviceInfo{};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.queueCreateInfoCount = 1u;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        deviceInfo.enabledExtensionCount =
            static_cast<uint32_t>(enabledDeviceExtensions.size());
        deviceInfo.ppEnabledExtensionNames =
            enabledDeviceExtensions.empty()
                ? nullptr
                : enabledDeviceExtensions.data();
        deviceInfo.pEnabledFeatures = &enabledFeatures;

        const VkResult deviceResult = createDeviceProc(
            m_physicalDevice, &deviceInfo, nullptr, &m_device);
        if (!checkResult(deviceResult, "vkCreateDevice",
                         report, error,
                         GsVulkanProbeStatus::DeviceCreationFailed))
        {
            return false;
        }
        if (!m_functions.load(
                m_getDeviceProcAddr, m_device, error))
        {
            report.status = GsVulkanProbeStatus::LoaderInvalid;
            report.detail = error;
            return false;
        }
        m_functions.getDeviceQueue(
            m_device, m_queueFamilyIndex, 0u, &m_queue);
        if (m_queue == VK_NULL_HANDLE)
        {
            error = "vkGetDeviceQueue returned a null compute queue";
            report.status = GsVulkanProbeStatus::DeviceCreationFailed;
            report.detail = error;
            return false;
        }
        getMemoryProperties(m_physicalDevice, &m_memoryProperties);
        return true;
    }

    int32_t VulkanExecutionContext::findMemoryType(
        uint32_t typeBits,
        VkMemoryPropertyFlags required,
        VkMemoryPropertyFlags preferred) const noexcept
    {
        int32_t bestIndex = -1;
        uint32_t bestScore = 0u;
        for (uint32_t index = 0u;
             index < m_memoryProperties.memoryTypeCount;
             ++index)
        {
            if ((typeBits & (1u << index)) == 0u)
                continue;
            const VkMemoryPropertyFlags flags =
                m_memoryProperties.memoryTypes[index].propertyFlags;
            if ((flags & required) != required)
                continue;

            uint32_t score = 0u;
            VkMemoryPropertyFlags remaining = flags & preferred;
            while (remaining != 0u)
            {
                score += remaining & 1u;
                remaining >>= 1u;
            }
            if (bestIndex < 0 || score > bestScore)
            {
                bestIndex = static_cast<int32_t>(index);
                bestScore = score;
            }
        }
        return bestIndex;
    }

    bool VulkanExecutionContext::createBuffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags required,
        VkMemoryPropertyFlags preferred,
        BufferAllocation &allocation,
        GsVulkanCapabilityReport &report,
        std::string &error)
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult result = m_functions.createBuffer(
            m_device, &bufferInfo, nullptr, &allocation.buffer);
        if (!checkResult(result, "vkCreateBuffer", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
            return false;

        VkMemoryRequirements requirements{};
        m_functions.getBufferMemoryRequirements(
            m_device, allocation.buffer, &requirements);
        const int32_t memoryType = findMemoryType(
            requirements.memoryTypeBits, required, preferred);
        if (memoryType < 0)
        {
            error = "no compatible Vulkan memory type satisfies the buffer requirements";
            return false;
        }

        VkMemoryAllocateInfo allocationInfo{};
        allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocationInfo.allocationSize = requirements.size;
        allocationInfo.memoryTypeIndex =
            static_cast<uint32_t>(memoryType);
        result = m_functions.allocateMemory(
            m_device, &allocationInfo, nullptr, &allocation.memory);
        if (!checkResult(result, "vkAllocateMemory", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
            return false;
        allocation.allocationSize = requirements.size;
        allocation.coherent =
            (m_memoryProperties
                 .memoryTypes[static_cast<uint32_t>(memoryType)]
                 .propertyFlags &
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u;

        result = m_functions.bindBufferMemory(
            m_device, allocation.buffer, allocation.memory, 0u);
        if (!checkResult(result, "vkBindBufferMemory", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
            return false;
        return true;
    }

    bool VulkanExecutionContext::createResources(
        GsVulkanCapabilityReport &report,
        std::string &error)
    {
        constexpr VkDeviceSize memoryCaseBytes =
            sizeof(GsVulkanMemoryCase) *
            static_cast<VkDeviceSize>(GS_VULKAN_MAX_MEMORY_CASES);
        constexpr VkDeviceSize memoryResultBytes =
            sizeof(GsVulkanMemoryResult) *
            static_cast<VkDeviceSize>(GS_VULKAN_MAX_MEMORY_CASES);
        if (!createBuffer(
                GS_VULKAN_VRAM_SIZE,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                m_vram, report, error) ||
            !createBuffer(
                GS_VULKAN_VRAM_SIZE,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                m_staging, report, error) ||
            !createBuffer(
                memoryCaseBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                m_memoryCases, report, error) ||
            !createBuffer(
                memoryResultBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                m_memoryResults, report, error))
        {
            if (report.status != GsVulkanProbeStatus::DeviceLost)
                report.status =
                    GsVulkanProbeStatus::ResourceCreationFailed;
            report.detail = error;
            return false;
        }

        VkResult result = m_functions.mapMemory(
            m_device, m_staging.memory, 0u, VK_WHOLE_SIZE, 0u,
            &m_stagingMap);
        if (!checkResult(result, "vkMapMemory", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }
        result = m_functions.mapMemory(
            m_device, m_memoryCases.memory, 0u, VK_WHOLE_SIZE, 0u,
            &m_memoryCasesMap);
        if (!checkResult(result, "vkMapMemory(memory cases)",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }
        result = m_functions.mapMemory(
            m_device, m_memoryResults.memory, 0u, VK_WHOLE_SIZE, 0u,
            &m_memoryResultsMap);
        if (!checkResult(result, "vkMapMemory(memory results)",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }

        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        for (uint32_t index = 0u; index < bindings.size(); ++index)
        {
            bindings[index].binding = index;
            bindings[index].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[index].descriptorCount = 1u;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo{};
        descriptorLayoutInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorLayoutInfo.bindingCount =
            static_cast<uint32_t>(bindings.size());
        descriptorLayoutInfo.pBindings = bindings.data();
        result = m_functions.createDescriptorSetLayout(
            m_device, &descriptorLayoutInfo, nullptr,
            &m_descriptorSetLayout);
        if (!checkResult(result, "vkCreateDescriptorSetLayout",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0u;
        pushConstantRange.size = sizeof(GsVulkanCt32Sprite);
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1u;
        pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1u;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        result = m_functions.createPipelineLayout(
            m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout);
        if (!checkResult(result, "vkCreatePipelineLayout",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }

        const auto createComputePipeline =
            [&](const uint32_t *code, size_t codeSize,
                VkShaderModule &shaderModule, VkPipeline &pipeline,
                std::string_view label)
        {
            VkShaderModuleCreateInfo shaderInfo{};
            shaderInfo.sType =
                VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shaderInfo.codeSize = codeSize;
            shaderInfo.pCode = code;
            VkResult createResult = m_functions.createShaderModule(
                m_device, &shaderInfo, nullptr, &shaderModule);
            if (!checkResult(
                    createResult,
                    std::string("vkCreateShaderModule(") +
                        std::string(label) + ')',
                    report, error,
                    GsVulkanProbeStatus::ResourceCreationFailed))
            {
                return false;
            }

            VkPipelineShaderStageCreateInfo stageInfo{};
            stageInfo.sType =
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = shaderModule;
            stageInfo.pName = "main";
            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType =
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.stage = stageInfo;
            pipelineInfo.layout = m_pipelineLayout;
            createResult = m_functions.createComputePipelines(
                m_device, VK_NULL_HANDLE, 1u, &pipelineInfo, nullptr,
                &pipeline);
            return checkResult(
                createResult,
                std::string("vkCreateComputePipelines(") +
                    std::string(label) + ')',
                report, error,
                GsVulkanProbeStatus::ResourceCreationFailed);
        };
        if (!createComputePipeline(
                kGsVramNoopShaderSpv,
                sizeof(kGsVramNoopShaderSpv),
                m_noopShaderModule, m_noopPipeline, "VRAM no-op") ||
            !createComputePipeline(
                kGsMemoryCasesShaderSpv,
                sizeof(kGsMemoryCasesShaderSpv),
                m_memoryShaderModule, m_memoryPipeline,
                "memory cases") ||
            !createComputePipeline(
                kGsCt32SpriteShaderSpv,
                sizeof(kGsCt32SpriteShaderSpv),
                m_spriteShaderModule, m_spritePipeline,
                "CT32 sprite"))
        {
            return false;
        }

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 3u;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1u;
        poolInfo.poolSizeCount = 1u;
        poolInfo.pPoolSizes = &poolSize;
        result = m_functions.createDescriptorPool(
            m_device, &poolInfo, nullptr, &m_descriptorPool);
        if (!checkResult(result, "vkCreateDescriptorPool",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }

        VkDescriptorSetAllocateInfo descriptorSetInfo{};
        descriptorSetInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetInfo.descriptorPool = m_descriptorPool;
        descriptorSetInfo.descriptorSetCount = 1u;
        descriptorSetInfo.pSetLayouts = &m_descriptorSetLayout;
        result = m_functions.allocateDescriptorSets(
            m_device, &descriptorSetInfo, &m_descriptorSet);
        if (!checkResult(result, "vkAllocateDescriptorSets",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }

        std::array<VkDescriptorBufferInfo, 3> descriptorBuffers{};
        descriptorBuffers[0].buffer = m_vram.buffer;
        descriptorBuffers[0].range = GS_VULKAN_VRAM_SIZE;
        descriptorBuffers[1].buffer = m_memoryCases.buffer;
        descriptorBuffers[1].range = memoryCaseBytes;
        descriptorBuffers[2].buffer = m_memoryResults.buffer;
        descriptorBuffers[2].range = memoryResultBytes;
        std::array<VkWriteDescriptorSet, 3> descriptorWrites{};
        for (uint32_t index = 0u;
             index < descriptorWrites.size(); ++index)
        {
            descriptorWrites[index].sType =
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = m_descriptorSet;
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].descriptorCount = 1u;
            descriptorWrites[index].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[index].pBufferInfo =
                &descriptorBuffers[index];
        }
        m_functions.updateDescriptorSets(
            m_device,
            static_cast<uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(), 0u, nullptr);

        VkCommandPoolCreateInfo commandPoolInfo{};
        commandPoolInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolInfo.flags =
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolInfo.queueFamilyIndex = m_queueFamilyIndex;
        result = m_functions.createCommandPool(
            m_device, &commandPoolInfo, nullptr, &m_commandPool);
        if (!checkResult(result, "vkCreateCommandPool",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }

        VkCommandBufferAllocateInfo commandBufferInfo{};
        commandBufferInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferInfo.commandPool = m_commandPool;
        commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferInfo.commandBufferCount = 1u;
        result = m_functions.allocateCommandBuffers(
            m_device, &commandBufferInfo, &m_commandBuffer);
        if (!checkResult(result, "vkAllocateCommandBuffers",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        result = m_functions.createFence(
            m_device, &fenceInfo, nullptr, &m_fence);
        if (!checkResult(result, "vkCreateFence", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }
        return true;
    }

    bool VulkanExecutionContext::initialize(
        const GsVulkanServiceConfig &config,
        GsVulkanCapabilityReport &report,
        std::string &error)
    {
        const GsVulkanDeviceReport *selectedDevice =
            report.selectedDevice();
        if (!report.ready() || !selectedDevice)
        {
            error = report.detail.empty()
                ? "Vulkan capability discovery did not select a device"
                : report.detail;
            return false;
        }
        if (config.fenceTimeoutNanoseconds == 0u)
        {
            error = "the Vulkan fence timeout must be non-zero";
            report.status = GsVulkanProbeStatus::ResourceCreationFailed;
            report.detail = error;
            return false;
        }
        const GsVulkanDeviceReport selected = *selectedDevice;
        m_fenceTimeoutNanoseconds = config.fenceTimeoutNanoseconds;
        if (!createInstance(config.probe, report, error) ||
            !createDevice(selected, report, error) ||
            !createResources(report, error))
        {
            return false;
        }

        GsVulkanServiceStatistics diagnostics{};
        refreshDiagnostics(report, diagnostics);
        if (report.validationErrors != 0u)
        {
            error = "Vulkan validation reported errors during service creation";
            report.status = GsVulkanProbeStatus::ValidationError;
            report.detail = error;
            return false;
        }
        report.status = GsVulkanProbeStatus::Ready;
        report.detail = "Vulkan GS service ready on " + selected.name;
        m_healthy = true;
        return true;
    }

    void VulkanExecutionContext::refreshDiagnostics(
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics) const
    {
        report.validationWarnings =
            m_validation.warnings.load(std::memory_order_relaxed);
        report.validationErrors =
            m_validation.errors.load(std::memory_order_relaxed);
        statistics.validationWarnings = report.validationWarnings;
        statistics.validationErrors = report.validationErrors;
        statistics.deviceLost =
            report.status == GsVulkanProbeStatus::DeviceLost;
    }

    bool VulkanExecutionContext::roundTrip(
        std::span<const uint8_t> input,
        std::vector<uint8_t> &output,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (input.size() != GS_VULKAN_VRAM_SIZE)
        {
            error = "Vulkan VRAM round trip requires exactly 4 MiB";
            return false;
        }

        const GsVulkanElementParameters parameters{
            static_cast<uint32_t>(
                GS_VULKAN_VRAM_SIZE / sizeof(uint32_t)),
            0u,
        };
        return executeKernel(
            input, {}, m_noopPipeline,
            GS_VULKAN_NOOP_GROUP_COUNT, 1u,
            &parameters, sizeof(parameters),
            output, nullptr, "VRAM round trip",
            report, statistics, error);
    }

    bool VulkanExecutionContext::executeMemoryCases(
        std::span<const uint8_t> input,
        std::span<const GsVulkanMemoryCase> cases,
        std::vector<uint8_t> &output,
        std::vector<GsVulkanMemoryResult> &results,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (input.size() != GS_VULKAN_VRAM_SIZE)
        {
            error = "Vulkan memory cases require exactly 4 MiB of VRAM";
            return false;
        }
        if (cases.empty() ||
            cases.size() > GS_VULKAN_MAX_MEMORY_CASES)
        {
            error = "Vulkan memory cases require between 1 and " +
                    std::to_string(GS_VULKAN_MAX_MEMORY_CASES) +
                    " records";
            return false;
        }

        const uint32_t caseCount =
            static_cast<uint32_t>(cases.size());
        const uint32_t groupCount =
            (caseCount + GS_VULKAN_NOOP_LOCAL_SIZE - 1u) /
            GS_VULKAN_NOOP_LOCAL_SIZE;
        const GsVulkanElementParameters parameters{caseCount, 0u};
        return executeKernel(
            input, cases, m_memoryPipeline, groupCount, 1u,
            &parameters, sizeof(parameters),
            output, &results, "memory-case batch",
            report, statistics, error);
    }

    bool VulkanExecutionContext::executeCt32Sprite(
        std::span<const uint8_t> input,
        const GsVulkanCt32Sprite &sprite,
        std::vector<uint8_t> &output,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (input.size() != GS_VULKAN_VRAM_SIZE)
        {
            error = "Vulkan CT32 sprite requires exactly 4 MiB of VRAM";
            return false;
        }
        if (const char *validationError =
                ct32SpriteValidationError(sprite))
        {
            error = validationError;
            return false;
        }

        constexpr uint32_t localSize = 8u;
        const uint32_t width = sprite.x1 - sprite.x0;
        const uint32_t height = sprite.y1 - sprite.y0;
        const uint32_t groupCountX =
            (width + localSize - 1u) / localSize;
        const uint32_t groupCountY =
            (height + localSize - 1u) / localSize;
        return executeKernel(
            input, {}, m_spritePipeline,
            groupCountX, groupCountY,
            &sprite, sizeof(sprite), output, nullptr,
            "CT32 sprite", report, statistics, error);
    }

    bool VulkanExecutionContext::executeKernel(
        std::span<const uint8_t> input,
        std::span<const GsVulkanMemoryCase> cases,
        VkPipeline pipeline,
        uint32_t groupCountX,
        uint32_t groupCountY,
        const void *pushConstants,
        uint32_t pushConstantSize,
        std::vector<uint8_t> &output,
        std::vector<GsVulkanMemoryResult> *results,
        std::string_view operationName,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        const bool memoryOperation = results != nullptr;
        if (pipeline == VK_NULL_HANDLE ||
            groupCountX == 0u || groupCountY == 0u ||
            !pushConstants || pushConstantSize == 0u ||
            pushConstantSize > sizeof(GsVulkanCt32Sprite) ||
            (pushConstantSize & 3u) != 0u ||
            memoryOperation != !cases.empty())
        {
            error = "invalid Vulkan GS kernel request";
            return false;
        }

        const uint32_t validationErrorsBefore =
            m_validation.errors.load(std::memory_order_relaxed);
        std::memcpy(m_stagingMap, input.data(), input.size());
        const VkDeviceSize caseBytes =
            sizeof(GsVulkanMemoryCase) * cases.size();
        const VkDeviceSize resultBytes =
            sizeof(GsVulkanMemoryResult) * cases.size();
        if (memoryOperation)
        {
            std::memcpy(
                m_memoryCasesMap, cases.data(),
                static_cast<size_t>(caseBytes));
        }

        const auto flushMappedAllocation =
            [&](const BufferAllocation &allocation,
                std::string_view label)
        {
            if (allocation.coherent)
                return true;
            VkMappedMemoryRange mappedRange{};
            mappedRange.sType =
                VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            mappedRange.memory = allocation.memory;
            mappedRange.offset = 0u;
            mappedRange.size = VK_WHOLE_SIZE;
            const VkResult flushResult =
                m_functions.flushMappedMemoryRanges(
                    m_device, 1u, &mappedRange);
            const std::string operation =
                "vkFlushMappedMemoryRanges(" +
                std::string(label) + ')';
            return checkResult(
                flushResult, operation, report, error,
                GsVulkanProbeStatus::ResourceCreationFailed);
        };
        if (!flushMappedAllocation(m_staging, "VRAM") ||
            (memoryOperation &&
             !flushMappedAllocation(m_memoryCases, "memory cases")))
        {
            m_healthy = false;
            return false;
        }

        VkResult result = m_functions.resetFences(
            m_device, 1u, &m_fence);
        if (!checkResult(result, "vkResetFences", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            m_healthy = false;
            return false;
        }
        result = m_functions.resetCommandBuffer(m_commandBuffer, 0u);
        if (!checkResult(result, "vkResetCommandBuffer", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            m_healthy = false;
            return false;
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = m_functions.beginCommandBuffer(
            m_commandBuffer, &beginInfo);
        if (!checkResult(result, "vkBeginCommandBuffer", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            m_healthy = false;
            return false;
        }

        VkBufferMemoryBarrier stagingUploadBarrier{};
        stagingUploadBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        stagingUploadBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        stagingUploadBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        stagingUploadBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        stagingUploadBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        stagingUploadBarrier.buffer = m_staging.buffer;
        stagingUploadBarrier.offset = 0u;
        stagingUploadBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0u, 0u, nullptr, 1u, &stagingUploadBarrier,
            0u, nullptr);

        if (memoryOperation)
        {
            VkBufferMemoryBarrier caseUploadBarrier{};
            caseUploadBarrier.sType =
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            caseUploadBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            caseUploadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            caseUploadBarrier.srcQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            caseUploadBarrier.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            caseUploadBarrier.buffer = m_memoryCases.buffer;
            caseUploadBarrier.offset = 0u;
            caseUploadBarrier.size = caseBytes;
            m_functions.cmdPipelineBarrier(
                m_commandBuffer,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0u, 0u, nullptr, 1u, &caseUploadBarrier,
                0u, nullptr);
        }

        VkBufferCopy copyRegion{};
        copyRegion.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdCopyBuffer(
            m_commandBuffer, m_staging.buffer, m_vram.buffer,
            1u, &copyRegion);

        VkBufferMemoryBarrier uploadComputeBarrier{};
        uploadComputeBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        uploadComputeBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        uploadComputeBarrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        uploadComputeBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        uploadComputeBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        uploadComputeBarrier.buffer = m_vram.buffer;
        uploadComputeBarrier.offset = 0u;
        uploadComputeBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr, 1u, &uploadComputeBarrier,
            0u, nullptr);

        m_functions.cmdBindPipeline(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline);
        m_functions.cmdBindDescriptorSets(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelineLayout, 0u, 1u, &m_descriptorSet,
            0u, nullptr);
        m_functions.cmdPushConstants(
            m_commandBuffer, m_pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0u,
            pushConstantSize, pushConstants);
        m_functions.cmdDispatch(
            m_commandBuffer, groupCountX, groupCountY, 1u);

        VkBufferMemoryBarrier computeDownloadBarrier{};
        computeDownloadBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        computeDownloadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        computeDownloadBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        computeDownloadBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        computeDownloadBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        computeDownloadBarrier.buffer = m_vram.buffer;
        computeDownloadBarrier.offset = 0u;
        computeDownloadBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0u, 0u, nullptr, 1u, &computeDownloadBarrier,
            0u, nullptr);

        if (memoryOperation)
        {
            VkBufferMemoryBarrier resultDownloadBarrier{};
            resultDownloadBarrier.sType =
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            resultDownloadBarrier.srcAccessMask =
                VK_ACCESS_SHADER_WRITE_BIT;
            resultDownloadBarrier.dstAccessMask =
                VK_ACCESS_HOST_READ_BIT;
            resultDownloadBarrier.srcQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            resultDownloadBarrier.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            resultDownloadBarrier.buffer = m_memoryResults.buffer;
            resultDownloadBarrier.offset = 0u;
            resultDownloadBarrier.size = resultBytes;
            m_functions.cmdPipelineBarrier(
                m_commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0u, 0u, nullptr, 1u, &resultDownloadBarrier,
                0u, nullptr);
        }

        m_functions.cmdCopyBuffer(
            m_commandBuffer, m_vram.buffer, m_staging.buffer,
            1u, &copyRegion);

        VkBufferMemoryBarrier stagingDownloadBarrier{};
        stagingDownloadBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        stagingDownloadBarrier.srcAccessMask =
            VK_ACCESS_TRANSFER_WRITE_BIT;
        stagingDownloadBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        stagingDownloadBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        stagingDownloadBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        stagingDownloadBarrier.buffer = m_staging.buffer;
        stagingDownloadBarrier.offset = 0u;
        stagingDownloadBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0u, 0u, nullptr, 1u, &stagingDownloadBarrier,
            0u, nullptr);

        result = m_functions.endCommandBuffer(m_commandBuffer);
        if (!checkResult(result, "vkEndCommandBuffer", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            m_healthy = false;
            return false;
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1u;
        submitInfo.pCommandBuffers = &m_commandBuffer;
        result = m_functions.queueSubmit(
            m_queue, 1u, &submitInfo, m_fence);
        if (!checkResult(result, "vkQueueSubmit", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            m_healthy = false;
            return false;
        }
        ++statistics.queueSubmissions;
        ++statistics.shaderDispatches;

        const auto waitStart = std::chrono::steady_clock::now();
        result = m_functions.waitForFences(
            m_device, 1u, &m_fence, VK_TRUE,
            m_fenceTimeoutNanoseconds);
        const auto waitEnd = std::chrono::steady_clock::now();
        statistics.fenceWaitNanoseconds +=
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    waitEnd - waitStart)
                    .count());
        if (result == VK_TIMEOUT)
        {
            error = "Vulkan GS " + std::string(operationName) +
                    " fence timed out";
            report.status = GsVulkanProbeStatus::ExecutionTimeout;
            report.detail = error;
            m_healthy = false;
            return false;
        }
        if (!checkResult(result, "vkWaitForFences", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            m_healthy = false;
            return false;
        }

        const auto invalidateMappedAllocation =
            [&](const BufferAllocation &allocation,
                std::string_view label)
        {
            if (allocation.coherent)
                return true;
            VkMappedMemoryRange mappedRange{};
            mappedRange.sType =
                VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            mappedRange.memory = allocation.memory;
            mappedRange.offset = 0u;
            mappedRange.size = VK_WHOLE_SIZE;
            const VkResult invalidateResult =
                m_functions.invalidateMappedMemoryRanges(
                    m_device, 1u, &mappedRange);
            const std::string operation =
                "vkInvalidateMappedMemoryRanges(" +
                std::string(label) + ')';
            return checkResult(
                invalidateResult, operation, report, error,
                GsVulkanProbeStatus::ResourceCreationFailed);
        };
        if (!invalidateMappedAllocation(m_staging, "VRAM") ||
            (memoryOperation &&
             !invalidateMappedAllocation(
                 m_memoryResults, "memory results")))
        {
            m_healthy = false;
            return false;
        }

        refreshDiagnostics(report, statistics);
        if (report.validationErrors != validationErrorsBefore)
        {
            error =
                "Vulkan validation reported an error during the " +
                std::string(operationName);
            report.status = GsVulkanProbeStatus::ValidationError;
            report.detail = error;
            m_healthy = false;
            return false;
        }

        std::vector<uint8_t> completedVram(GS_VULKAN_VRAM_SIZE);
        std::memcpy(
            completedVram.data(), m_stagingMap,
            completedVram.size());
        std::vector<GsVulkanMemoryResult> completedResults;
        if (memoryOperation)
        {
            completedResults.resize(cases.size());
            std::memcpy(
                completedResults.data(), m_memoryResultsMap,
                static_cast<size_t>(resultBytes));
        }

        output = std::move(completedVram);
        if (results)
            *results = std::move(completedResults);
        statistics.bytesUploaded += GS_VULKAN_VRAM_SIZE;
        statistics.bytesDownloaded += GS_VULKAN_VRAM_SIZE;
        error.clear();
        return true;
    }

    void VulkanExecutionContext::shutdown() noexcept
    {
        if (m_device != VK_NULL_HANDLE && m_functions.deviceWaitIdle)
            (void)m_functions.deviceWaitIdle(m_device);
        if (m_memoryResultsMap && m_functions.unmapMemory)
        {
            m_functions.unmapMemory(
                m_device, m_memoryResults.memory);
        }
        m_memoryResultsMap = nullptr;
        if (m_memoryCasesMap && m_functions.unmapMemory)
            m_functions.unmapMemory(m_device, m_memoryCases.memory);
        m_memoryCasesMap = nullptr;
        if (m_stagingMap && m_functions.unmapMemory)
            m_functions.unmapMemory(m_device, m_staging.memory);
        m_stagingMap = nullptr;
        if (m_fence != VK_NULL_HANDLE && m_functions.destroyFence)
            m_functions.destroyFence(m_device, m_fence, nullptr);
        m_fence = VK_NULL_HANDLE;
        if (m_commandPool != VK_NULL_HANDLE &&
            m_functions.destroyCommandPool)
        {
            m_functions.destroyCommandPool(
                m_device, m_commandPool, nullptr);
        }
        m_commandPool = VK_NULL_HANDLE;
        m_commandBuffer = VK_NULL_HANDLE;
        if (m_descriptorPool != VK_NULL_HANDLE &&
            m_functions.destroyDescriptorPool)
        {
            m_functions.destroyDescriptorPool(
                m_device, m_descriptorPool, nullptr);
        }
        m_descriptorPool = VK_NULL_HANDLE;
        m_descriptorSet = VK_NULL_HANDLE;
        if (m_spritePipeline != VK_NULL_HANDLE &&
            m_functions.destroyPipeline)
        {
            m_functions.destroyPipeline(
                m_device, m_spritePipeline, nullptr);
        }
        m_spritePipeline = VK_NULL_HANDLE;
        if (m_spriteShaderModule != VK_NULL_HANDLE &&
            m_functions.destroyShaderModule)
        {
            m_functions.destroyShaderModule(
                m_device, m_spriteShaderModule, nullptr);
        }
        m_spriteShaderModule = VK_NULL_HANDLE;
        if (m_memoryPipeline != VK_NULL_HANDLE &&
            m_functions.destroyPipeline)
        {
            m_functions.destroyPipeline(
                m_device, m_memoryPipeline, nullptr);
        }
        m_memoryPipeline = VK_NULL_HANDLE;
        if (m_memoryShaderModule != VK_NULL_HANDLE &&
            m_functions.destroyShaderModule)
        {
            m_functions.destroyShaderModule(
                m_device, m_memoryShaderModule, nullptr);
        }
        m_memoryShaderModule = VK_NULL_HANDLE;
        if (m_noopPipeline != VK_NULL_HANDLE &&
            m_functions.destroyPipeline)
        {
            m_functions.destroyPipeline(
                m_device, m_noopPipeline, nullptr);
        }
        m_noopPipeline = VK_NULL_HANDLE;
        if (m_noopShaderModule != VK_NULL_HANDLE &&
            m_functions.destroyShaderModule)
        {
            m_functions.destroyShaderModule(
                m_device, m_noopShaderModule, nullptr);
        }
        m_noopShaderModule = VK_NULL_HANDLE;
        if (m_pipelineLayout != VK_NULL_HANDLE &&
            m_functions.destroyPipelineLayout)
        {
            m_functions.destroyPipelineLayout(
                m_device, m_pipelineLayout, nullptr);
        }
        m_pipelineLayout = VK_NULL_HANDLE;
        if (m_descriptorSetLayout != VK_NULL_HANDLE &&
            m_functions.destroyDescriptorSetLayout)
        {
            m_functions.destroyDescriptorSetLayout(
                m_device, m_descriptorSetLayout, nullptr);
        }
        m_descriptorSetLayout = VK_NULL_HANDLE;

        auto destroyAllocation = [&](BufferAllocation &allocation)
        {
            if (allocation.buffer != VK_NULL_HANDLE &&
                m_functions.destroyBuffer)
            {
                m_functions.destroyBuffer(
                    m_device, allocation.buffer, nullptr);
            }
            allocation.buffer = VK_NULL_HANDLE;
            if (allocation.memory != VK_NULL_HANDLE &&
                m_functions.freeMemory)
            {
                m_functions.freeMemory(
                    m_device, allocation.memory, nullptr);
            }
            allocation.memory = VK_NULL_HANDLE;
            allocation.allocationSize = 0u;
            allocation.coherent = false;
        };
        destroyAllocation(m_memoryResults);
        destroyAllocation(m_memoryCases);
        destroyAllocation(m_staging);
        destroyAllocation(m_vram);

        if (m_device != VK_NULL_HANDLE && m_functions.destroyDevice)
            m_functions.destroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
        m_queue = VK_NULL_HANDLE;
        m_physicalDevice = VK_NULL_HANDLE;
        m_instance.reset();
        m_healthy = false;
    }
}
#endif

#if PS2X_HAS_GS_VULKAN
enum class GsVulkanRequestKind : uint8_t
{
    RoundTrip,
    MemoryCases,
    Ct32Sprite,
};

struct GsVulkanService::Impl final
{
    explicit Impl(const GsVulkanServiceConfig &serviceConfig)
        : config(serviceConfig)
    {
    }

    void threadMain()
    {
        try
        {
            threadMainImpl();
        }
        catch (const std::exception &exception)
        {
            failUnexpectedWorkerExit(
                std::string("Vulkan GS worker exception: ") +
                exception.what());
        }
        catch (...)
        {
            failUnexpectedWorkerExit(
                "Vulkan GS worker exited with an unknown exception");
        }
    }

    void failUnexpectedWorkerExit(std::string message)
    {
        {
            std::lock_guard lock(stateMutex);
            capabilities.compiled = true;
            capabilities.status =
                GsVulkanProbeStatus::ResourceCreationFailed;
            capabilities.detail = message;
            initializationError = message;
            healthy = false;
            initialized = true;
            workerFinished = true;
            requestPending = false;
            if (requestInFlight)
            {
                if (activeRequestKind ==
                    GsVulkanRequestKind::MemoryCases)
                {
                    ++statistics.memoryBatchesFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::Ct32Sprite)
                {
                    ++statistics.spriteDrawsFailed;
                }
                else
                {
                    ++statistics.roundTripsFailed;
                }
                responseOutput.clear();
                responseResults.clear();
                responseError = message;
                responseSucceeded = false;
                responseReady = true;
                requestInFlight = false;
            }
        }
        stateChanged.notify_all();
    }

    void threadMainImpl()
    {
        VulkanExecutionContext context;
        GsVulkanCapabilityReport localCapabilities =
            probeGsVulkanCapabilities(config.probe);
        GsVulkanServiceStatistics localStatistics{};
        std::string localError = localCapabilities.ready()
            ? std::string{}
            : localCapabilities.detail;
        const bool initializedSuccessfully =
            localCapabilities.ready() &&
            context.initialize(config, localCapabilities, localError);

        if (!initializedSuccessfully)
        {
            context.shutdown();
            context.refreshDiagnostics(
                localCapabilities, localStatistics);
        }

        {
            std::lock_guard lock(stateMutex);
            capabilities = localCapabilities;
            statistics = localStatistics;
            initializationError = localError;
            healthy = initializedSuccessfully;
            initialized = true;
            if (!initializedSuccessfully)
                workerFinished = true;
        }
        stateChanged.notify_all();
        if (!initializedSuccessfully)
            return;

        for (;;)
        {
            std::vector<uint8_t> input;
            std::vector<GsVulkanMemoryCase> memoryCases;
            GsVulkanCt32Sprite sprite{};
            GsVulkanRequestKind kind =
                GsVulkanRequestKind::RoundTrip;
            {
                std::unique_lock lock(stateMutex);
                stateChanged.wait(lock, [this]
                {
                    return stopping || requestPending;
                });
                if (stopping && !requestPending)
                    break;
                input = std::move(requestInput);
                memoryCases = std::move(requestMemoryCases);
                sprite = requestSprite;
                kind = requestKind;
                activeRequestKind = kind;
                requestPending = false;
                requestInFlight = true;
            }

            std::vector<uint8_t> output;
            std::vector<GsVulkanMemoryResult> memoryResults;
            std::string operationError;
            bool succeeded = false;
            if (kind == GsVulkanRequestKind::MemoryCases)
            {
                succeeded = context.executeMemoryCases(
                    input, memoryCases, output, memoryResults,
                    localCapabilities, localStatistics,
                    operationError);
            }
            else if (kind == GsVulkanRequestKind::Ct32Sprite)
            {
                succeeded = context.executeCt32Sprite(
                    input, sprite, output,
                    localCapabilities, localStatistics,
                    operationError);
            }
            else
            {
                succeeded = context.roundTrip(
                    input, output, localCapabilities,
                    localStatistics, operationError);
            }
            if (kind == GsVulkanRequestKind::MemoryCases)
            {
                if (succeeded)
                {
                    ++localStatistics.memoryBatchesCompleted;
                    localStatistics.memoryCasesExecuted +=
                        memoryCases.size();
                }
                else
                {
                    ++localStatistics.memoryBatchesFailed;
                }
            }
            else if (kind == GsVulkanRequestKind::Ct32Sprite)
            {
                if (succeeded)
                {
                    ++localStatistics.spriteDrawsCompleted;
                    localStatistics.spritePixelsExecuted +=
                        static_cast<uint64_t>(sprite.x1 - sprite.x0) *
                        static_cast<uint64_t>(sprite.y1 - sprite.y0);
                }
                else
                {
                    ++localStatistics.spriteDrawsFailed;
                }
            }
            else if (succeeded)
            {
                ++localStatistics.roundTripsCompleted;
            }
            else
            {
                ++localStatistics.roundTripsFailed;
            }
            context.refreshDiagnostics(
                localCapabilities, localStatistics);

            {
                std::lock_guard lock(stateMutex);
                capabilities = localCapabilities;
                statistics = localStatistics;
                healthy = context.healthy();
                responseOutput = std::move(output);
                responseResults = std::move(memoryResults);
                responseError = std::move(operationError);
                responseSucceeded = succeeded;
                responseReady = true;
                requestInFlight = false;
            }
            stateChanged.notify_all();
        }

        context.refreshDiagnostics(localCapabilities, localStatistics);
        context.shutdown();
        context.refreshDiagnostics(localCapabilities, localStatistics);
        if (localCapabilities.status == GsVulkanProbeStatus::Ready &&
            localStatistics.validationErrors != 0u)
        {
            localCapabilities.status =
                GsVulkanProbeStatus::ValidationError;
            localCapabilities.detail =
                "Vulkan validation reported an error during service shutdown";
        }
        {
            std::lock_guard lock(stateMutex);
            capabilities = std::move(localCapabilities);
            statistics = localStatistics;
            healthy = false;
            workerFinished = true;
        }
        stateChanged.notify_all();
    }

    bool executeRequest(
        GsVulkanRequestKind kind,
        std::vector<uint8_t> input,
        std::vector<GsVulkanMemoryCase> memoryCases,
        GsVulkanCt32Sprite sprite,
        std::vector<uint8_t> &output,
        std::vector<GsVulkanMemoryResult> *memoryResults,
        std::string *error)
    {
        std::lock_guard callLock(callMutex);
        std::unique_lock stateLock(stateMutex);
        if (!healthy || stopping || workerFinished)
        {
            if (error)
            {
                if (stopping || workerFinished)
                    *error = "Vulkan GS service is shut down";
                else
                {
                    *error = capabilities.detail.empty()
                        ? "Vulkan GS service is not healthy"
                        : capabilities.detail;
                }
            }
            return false;
        }

        requestKind = kind;
        requestInput = std::move(input);
        requestMemoryCases = std::move(memoryCases);
        requestSprite = sprite;
        responseOutput.clear();
        responseResults.clear();
        responseError.clear();
        responseSucceeded = false;
        responseReady = false;
        requestPending = true;
        stateLock.unlock();
        stateChanged.notify_all();
        stateLock.lock();
        stateChanged.wait(stateLock, [this]
        {
            return responseReady || workerFinished;
        });

        if (!responseReady)
        {
            if (error)
            {
                *error =
                    "Vulkan GS worker stopped before completing the request";
            }
            return false;
        }
        const bool succeeded = responseSucceeded;
        std::vector<uint8_t> completedOutput =
            std::move(responseOutput);
        std::vector<GsVulkanMemoryResult> completedResults =
            std::move(responseResults);
        std::string operationError = std::move(responseError);
        responseReady = false;
        stateLock.unlock();

        if (!succeeded)
        {
            if (error)
                *error = std::move(operationError);
            return false;
        }
        output = std::move(completedOutput);
        if (memoryResults)
            *memoryResults = std::move(completedResults);
        if (error)
            error->clear();
        return true;
    }

    void stopAndJoin() noexcept
    {
        {
            std::lock_guard lock(stateMutex);
            stopping = true;
        }
        stateChanged.notify_all();
        if (worker.joinable())
            worker.join();
    }

    GsVulkanServiceConfig config;
    mutable std::mutex stateMutex;
    std::mutex callMutex;
    std::condition_variable stateChanged;
    std::thread worker;
    GsVulkanCapabilityReport capabilities;
    GsVulkanServiceStatistics statistics;
    std::string initializationError;
    std::vector<uint8_t> requestInput;
    std::vector<GsVulkanMemoryCase> requestMemoryCases;
    GsVulkanCt32Sprite requestSprite{};
    std::vector<uint8_t> responseOutput;
    std::vector<GsVulkanMemoryResult> responseResults;
    std::string responseError;
    GsVulkanRequestKind requestKind =
        GsVulkanRequestKind::RoundTrip;
    GsVulkanRequestKind activeRequestKind =
        GsVulkanRequestKind::RoundTrip;
    bool initialized = false;
    bool healthy = false;
    bool stopping = false;
    bool workerFinished = false;
    bool requestPending = false;
    bool requestInFlight = false;
    bool responseReady = false;
    bool responseSucceeded = false;
};
#else
struct GsVulkanService::Impl final
{
};
#endif

GsVulkanService::GsVulkanService(std::unique_ptr<Impl> impl)
    : m_impl(std::move(impl))
{
}

GsVulkanService::~GsVulkanService()
{
    shutdown();
}

void GsVulkanService::shutdown() noexcept
{
#if PS2X_HAS_GS_VULKAN
    if (m_impl)
        m_impl->stopAndJoin();
#endif
}

std::unique_ptr<GsVulkanService> GsVulkanService::create(
    const GsVulkanServiceConfig &config,
    GsVulkanCapabilityReport *report,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    GsVulkanCapabilityReport unavailable =
        probeGsVulkanCapabilities(config.probe);
    if (report)
        *report = unavailable;
    if (error)
        *error = unavailable.detail;
    return nullptr;
#else
    auto impl = std::make_unique<Impl>(config);
    try
    {
        impl->worker = std::thread(&Impl::threadMain, impl.get());
    }
    catch (const std::exception &exception)
    {
        GsVulkanCapabilityReport failed{};
        failed.compiled = true;
        failed.status = GsVulkanProbeStatus::ResourceCreationFailed;
        failed.detail =
            std::string("failed to start Vulkan GS worker: ") +
            exception.what();
        if (report)
            *report = failed;
        if (error)
            *error = failed.detail;
        return nullptr;
    }

    GsVulkanCapabilityReport initializedCapabilities;
    std::string initializationError;
    bool initializedSuccessfully = false;
    {
        std::unique_lock lock(impl->stateMutex);
        impl->stateChanged.wait(lock, [&impl]
        {
            return impl->initialized;
        });
        initializedCapabilities = impl->capabilities;
        initializationError = impl->initializationError;
        initializedSuccessfully = impl->healthy;
    }

    if (report)
        *report = initializedCapabilities;
    if (error)
        *error = initializationError;
    if (!initializedSuccessfully)
    {
        impl->stopAndJoin();
        return nullptr;
    }
    return std::unique_ptr<GsVulkanService>(
        new GsVulkanService(std::move(impl)));
#endif
}

bool GsVulkanService::roundTripVram(
    std::span<const uint8_t> input,
    std::vector<uint8_t> &output,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)input;
    (void)output;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (input.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
            *error = "Vulkan VRAM round trip requires exactly 4 MiB";
        return false;
    }

    return m_impl->executeRequest(
        GsVulkanRequestKind::RoundTrip,
        std::vector<uint8_t>(input.begin(), input.end()), {},
        {}, output, nullptr, error);
#endif
}

bool GsVulkanService::executeMemoryCases(
    std::span<const uint8_t> input,
    std::span<const GsVulkanMemoryCase> cases,
    std::vector<uint8_t> &output,
    std::vector<GsVulkanMemoryResult> &results,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)input;
    (void)cases;
    (void)output;
    (void)results;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (input.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
            *error =
                "Vulkan memory cases require exactly 4 MiB of VRAM";
        return false;
    }
    if (cases.empty() ||
        cases.size() > GS_VULKAN_MAX_MEMORY_CASES)
    {
        if (error)
        {
            *error = "Vulkan memory cases require between 1 and " +
                     std::to_string(GS_VULKAN_MAX_MEMORY_CASES) +
                     " records";
        }
        return false;
    }

    for (size_t index = 0u; index < cases.size(); ++index)
    {
        const GsVulkanMemoryCase &memoryCase = cases[index];
        const auto psm = static_cast<GSMem::PixelStorageMode>(
            memoryCase.pixelStorageMode);
        if (!GSMem::IsValidPsm(psm))
        {
            if (error)
            {
                *error = "Vulkan memory case " +
                         std::to_string(index) +
                         " has an unsupported pixel storage mode";
            }
            return false;
        }
        if (memoryCase.operation !=
                GsVulkanMemoryOperation::Read &&
            memoryCase.operation !=
                GsVulkanMemoryOperation::Write)
        {
            if (error)
            {
                *error = "Vulkan memory case " +
                         std::to_string(index) +
                         " has an unsupported operation";
            }
            return false;
        }
        if (memoryCase.reserved != 0u)
        {
            if (error)
            {
                *error = "Vulkan memory case " +
                         std::to_string(index) +
                         " has non-zero reserved data";
            }
            return false;
        }
    }

    return m_impl->executeRequest(
        GsVulkanRequestKind::MemoryCases,
        std::vector<uint8_t>(input.begin(), input.end()),
        std::vector<GsVulkanMemoryCase>(
            cases.begin(), cases.end()),
        {}, output, &results, error);
#endif
}

bool GsVulkanService::executeCt32Sprite(
    std::span<const uint8_t> input,
    const GsVulkanCt32Sprite &sprite,
    std::vector<uint8_t> &output,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)input;
    (void)sprite;
    (void)output;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (input.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
            *error =
                "Vulkan CT32 sprite requires exactly 4 MiB of VRAM";
        return false;
    }
    if (const char *validationError =
            ct32SpriteValidationError(sprite))
    {
        if (error)
            *error = validationError;
        return false;
    }

    return m_impl->executeRequest(
        GsVulkanRequestKind::Ct32Sprite,
        std::vector<uint8_t>(input.begin(), input.end()), {},
        sprite, output, nullptr, error);
#endif
}

GsVulkanCapabilityReport GsVulkanService::capabilities() const
{
#if !PS2X_HAS_GS_VULKAN
    return probeGsVulkanCapabilities();
#else
    std::lock_guard lock(m_impl->stateMutex);
    return m_impl->capabilities;
#endif
}

GsVulkanServiceStatistics GsVulkanService::statistics() const
{
#if !PS2X_HAS_GS_VULKAN
    return {};
#else
    std::lock_guard lock(m_impl->stateMutex);
    return m_impl->statistics;
#endif
}

bool GsVulkanService::healthy() const
{
#if !PS2X_HAS_GS_VULKAN
    return false;
#else
    std::lock_guard lock(m_impl->stateMutex);
    return m_impl->healthy && !m_impl->stopping &&
           !m_impl->workerFinished;
#endif
}
