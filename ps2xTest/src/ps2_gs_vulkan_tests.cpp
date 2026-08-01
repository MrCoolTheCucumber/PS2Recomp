#include "MiniTest.h"
#include "runtime/ps2_gs_vulkan.h"

#include <cstdint>
#include <string>

void register_ps2_gs_vulkan_tests()
{
    MiniTest::Case("PS2GSVulkan", [](TestCase &tc)
    {
        tc.Run("Vulkan capability metadata is stable", [](TestCase &t)
        {
            constexpr uint32_t version =
                (1u << 22u) | (4u << 12u) | 350u;
            t.Equals(gsVulkanVersionString(version),
                     std::string("1.4.350"),
                     "packed Vulkan versions should have a stable text form");
            t.Equals(
                gsVulkanProbeStatusName(GsVulkanProbeStatus::Ready),
                std::string_view("ready"),
                "probe statuses should have stable diagnostic names");
            t.Equals(
                gsVulkanDeviceKindName(GsVulkanDeviceKind::DiscreteGpu),
                std::string_view("discrete-gpu"),
                "device kinds should have stable diagnostic names");
            t.Equals(GS_VULKAN_NOOP_GROUP_COUNT, 16384u,
                     "the fixed 64-thread kernel should cover every 4 MiB word");
        });

        tc.Run("An unavailable Vulkan loader fails closed", [](TestCase &t)
        {
            GsVulkanProbeConfig config{};
            config.loaderPath =
                "/ps2recomp-test-loader-does-not-exist/libvulkan.so";
            const GsVulkanCapabilityReport report =
                probeGsVulkanCapabilities(config);

#if PS2X_HAS_GS_VULKAN
            t.IsTrue(report.compiled,
                     "a header-enabled build should report Vulkan as compiled");
            t.IsFalse(report.loaderAvailable,
                      "an explicit missing loader must not fall back to a system loader");
            t.Equals(report.status,
                     GsVulkanProbeStatus::LoaderUnavailable,
                     "a missing runtime loader should be a clean capability result");
#else
            t.IsFalse(report.compiled,
                      "a disabled build should expose the compiled-out capability");
            t.Equals(report.status,
                     GsVulkanProbeStatus::CompiledOut,
                     "a disabled build must remain a clean software-only build");
#endif
            t.IsFalse(report.ready(),
                      "a missing or compiled-out loader can never select GPU work");
            t.IsNull(report.selectedDevice(),
                     "failed capability discovery must not retain a selected device");
        });

        tc.Run("Headless Vulkan device selection is internally consistent", [](TestCase &t)
        {
            const GsVulkanCapabilityReport report =
                probeGsVulkanCapabilities();

#if PS2X_HAS_GS_VULKAN
            t.IsTrue(report.compiled,
                     "the compiled Vulkan path should identify itself");
            if (report.ready())
            {
                const GsVulkanDeviceReport *selected =
                    report.selectedDevice();
                t.IsNotNull(selected,
                            "a ready report must identify its selected device");
                if (selected)
                {
                    t.IsTrue(selected->suitable,
                             "the selected device must satisfy every gate");
                    t.IsTrue(selected->exactVramStorage,
                             "the selected device must address exact raw VRAM storage");
                    t.IsTrue(selected->computeQueue,
                             "the selected device must expose a compute queue");
                    t.IsTrue(
                        selected->maxStorageBufferRange >=
                            GS_VULKAN_VRAM_SIZE,
                        "the selected storage-buffer range must cover all GS VRAM");
                    t.IsFalse(
                        selected->kind == GsVulkanDeviceKind::Cpu,
                        "a CPU Vulkan implementation is not a hardware renderer");
                }
            }
            else
            {
                t.IsNull(report.selectedDevice(),
                         "an unavailable host must remain software-only");
            }
#else
            t.Equals(report.status,
                     GsVulkanProbeStatus::CompiledOut,
                     "the generic probe must be callable in software-only builds");
#endif

            for (const GsVulkanDeviceReport &device : report.devices)
            {
                if (!device.suitable)
                    continue;
                t.IsTrue(device.exactVramStorage && device.computeQueue &&
                             device.deviceLocalMemory &&
                             device.hostVisibleMemory,
                         "every suitable device should satisfy the permanent capability contract");
                t.IsTrue(device.rejectionReason.empty(),
                         "suitable devices should not retain a rejection reason");
            }
        });

        tc.Run("Requested Vulkan validation is explicit and fail-closed", [](TestCase &t)
        {
            GsVulkanProbeConfig config{};
            config.enableValidation = true;
            const GsVulkanCapabilityReport report =
                probeGsVulkanCapabilities(config);

            t.IsTrue(report.validationRequested,
                     "the report should retain the validation request");
            if (report.ready())
            {
                t.IsTrue(report.validationLayerAvailable &&
                             report.debugUtilsAvailable &&
                             report.validationEnabled,
                         "a validated ready device must have the layer, extension, and messenger");
                t.Equals(report.validationErrors, 0u,
                         "a ready validated probe must contain no validation errors");
            }
            if (report.status ==
                GsVulkanProbeStatus::ValidationUnavailable)
            {
                t.IsFalse(report.ready(),
                          "missing requested validation must not silently continue");
            }
        });
    });
}
