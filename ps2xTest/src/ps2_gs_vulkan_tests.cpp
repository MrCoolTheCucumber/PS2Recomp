#include "MiniTest.h"
#include "runtime/ps2_gs_vulkan.h"

#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace
{
    std::vector<uint8_t> makeVramPattern(uint32_t seed)
    {
        std::vector<uint8_t> pattern(GS_VULKAN_VRAM_SIZE);
        uint32_t state = seed;
        for (uint8_t &byte : pattern)
        {
            state ^= state << 13u;
            state ^= state >> 17u;
            state ^= state << 5u;
            byte = static_cast<uint8_t>(state >> 24u);
        }
        return pattern;
    }
}

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

            GsVulkanServiceConfig serviceConfig{};
            serviceConfig.probe = config;
            GsVulkanCapabilityReport serviceReport{};
            std::string serviceError;
            const std::unique_ptr<GsVulkanService> service =
                GsVulkanService::create(
                    serviceConfig, &serviceReport, &serviceError);
            t.IsNull(service.get(),
                     "a missing explicit loader must not create a service");
            t.Equals(serviceReport.status, report.status,
                     "the service should preserve the capability failure status");
            t.IsFalse(serviceError.empty(),
                      "service creation failure should retain a diagnostic");
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

        tc.Run("Vulkan service rejects an unbounded fence configuration", [](TestCase &t)
        {
            GsVulkanServiceConfig config{};
            config.probe.enableValidation = false;
            config.fenceTimeoutNanoseconds = 0u;
            const GsVulkanCapabilityReport preflight =
                probeGsVulkanCapabilities(config.probe);
            GsVulkanCapabilityReport report{};
            std::string error;
            const std::unique_ptr<GsVulkanService> service =
                GsVulkanService::create(config, &report, &error);
            t.IsNull(service.get(),
                     "a zero timeout must never create an execution service");
            t.IsFalse(error.empty(),
                      "the rejected timeout should retain a diagnostic");
            if (preflight.ready())
            {
                t.Equals(report.status,
                         GsVulkanProbeStatus::ResourceCreationFailed,
                         "a capable host should reject the unbounded service contract itself");
            }
            else
            {
                t.Equals(report.status, preflight.status,
                         "an unavailable host should retain its earlier capability failure");
            }
        });

        tc.Run("Vulkan service preserves exact 4 MiB VRAM", [](TestCase &t)
        {
            GsVulkanServiceConfig config{};
            config.probe.enableValidation = true;
            GsVulkanCapabilityReport preflight =
                probeGsVulkanCapabilities(config.probe);
            if (preflight.status ==
                GsVulkanProbeStatus::ValidationUnavailable)
            {
                config.probe.enableValidation = false;
                preflight = probeGsVulkanCapabilities(config.probe);
            }

            GsVulkanCapabilityReport creationReport{};
            std::string creationError;
            std::unique_ptr<GsVulkanService> service =
                GsVulkanService::create(
                    config, &creationReport, &creationError);
            if (!preflight.ready())
            {
                t.IsNull(service.get(),
                         "an unavailable host should remain software-only");
                t.IsFalse(creationReport.ready(),
                          "an unavailable service must preserve its capability result");
                return;
            }

            t.IsNotNull(service.get(),
                        "a suitable device should create the execution service");
            t.IsTrue(creationReport.ready(),
                     "a created service should retain a ready report");
            t.IsTrue(creationError.empty(),
                     "successful service creation should clear its diagnostic");
            if (!service)
                return;

            std::vector<std::vector<uint8_t>> patterns;
            patterns.emplace_back(GS_VULKAN_VRAM_SIZE, 0u);
            patterns.emplace_back(GS_VULKAN_VRAM_SIZE, 0xFFu);
            patterns.emplace_back(GS_VULKAN_VRAM_SIZE);
            for (size_t index = 0u; index < patterns.back().size(); ++index)
            {
                patterns.back()[index] =
                    static_cast<uint8_t>(index);
            }
            patterns.push_back(makeVramPattern(0x52414331u));

            for (const std::vector<uint8_t> &input : patterns)
            {
                std::vector<uint8_t> output = {0xA5u, 0x5Au};
                std::string error;
                const bool succeeded =
                    service->roundTripVram(input, output, &error);
                t.IsTrue(succeeded,
                         "each fixed-size service request should complete");
                t.IsTrue(error.empty(),
                         "a successful request should clear its diagnostic");
                t.IsTrue(output == input,
                         "upload, shader access, and download must preserve every VRAM byte");
            }

            const std::vector<uint8_t> invalid(
                GS_VULKAN_VRAM_SIZE - 1u, 0x11u);
            std::vector<uint8_t> invalidOutput = {0xBEu, 0xEFu};
            const std::vector<uint8_t> invalidSentinel = invalidOutput;
            std::string invalidError;
            t.IsFalse(service->roundTripVram(
                          invalid, invalidOutput, &invalidError),
                      "an input shorter than exact GS VRAM must fail closed");
            t.IsTrue(invalidOutput == invalidSentinel,
                     "a rejected request must not alter caller output");
            t.IsFalse(invalidError.empty(),
                      "a rejected request should explain the exact-size contract");
            t.IsTrue(service->healthy(),
                     "caller input rejection must not poison the worker");

            const std::vector<uint8_t> concurrentA =
                makeVramPattern(0x13579BDFu);
            const std::vector<uint8_t> concurrentB =
                makeVramPattern(0x2468ACE1u);
            std::vector<uint8_t> outputA;
            std::vector<uint8_t> outputB;
            std::string errorA;
            std::string errorB;
            bool succeededA = false;
            bool succeededB = false;
            std::thread callerA([&]
            {
                succeededA = service->roundTripVram(
                    concurrentA, outputA, &errorA);
            });
            std::thread callerB([&]
            {
                succeededB = service->roundTripVram(
                    concurrentB, outputB, &errorB);
            });
            callerA.join();
            callerB.join();
            t.IsTrue(succeededA && succeededB,
                     "concurrent callers should serialize through the bounded slot");
            t.IsTrue(errorA.empty() && errorB.empty(),
                     "serialized callers should complete without diagnostics");
            t.IsTrue(outputA == concurrentA && outputB == concurrentB,
                     "serialized requests must retain their own exact output");

            const GsVulkanCapabilityReport finalReport =
                service->capabilities();
            const GsVulkanServiceStatistics statistics =
                service->statistics();
            t.IsTrue(finalReport.ready() && service->healthy(),
                     "successful work should leave the service selectable");
            t.Equals(statistics.roundTripsCompleted, 6ull,
                     "all accepted requests should complete exactly once");
            t.Equals(statistics.roundTripsFailed, 0ull,
                     "caller-side size rejection should not submit failed GPU work");
            t.Equals(statistics.queueSubmissions, 6ull,
                     "each request should use one bounded queue submission");
            t.Equals(statistics.shaderDispatches, 6ull,
                     "each request should execute the storage-buffer shader");
            t.Equals(statistics.bytesUploaded,
                     6ull * GS_VULKAN_VRAM_SIZE,
                     "statistics should account for every uploaded byte");
            t.Equals(statistics.bytesDownloaded,
                     6ull * GS_VULKAN_VRAM_SIZE,
                     "statistics should account for every downloaded byte");
            t.Equals(statistics.validationErrors, 0u,
                     "validated service work must remain error-free");
            t.IsFalse(statistics.deviceLost,
                      "successful service work must not report device loss");

            service->shutdown();
            service->shutdown();
            t.IsFalse(service->healthy(),
                      "explicit shutdown should be synchronous and idempotent");
            const GsVulkanServiceStatistics shutdownStatistics =
                service->statistics();
            t.Equals(shutdownStatistics.roundTripsCompleted, 6ull,
                     "shutdown must drain without duplicating accepted work");
            t.Equals(shutdownStatistics.validationErrors, 0u,
                     "resource destruction must remain validation-clean");

            std::vector<uint8_t> shutdownOutput = {0x42u};
            const std::vector<uint8_t> shutdownSentinel = shutdownOutput;
            std::string shutdownError;
            t.IsFalse(service->roundTripVram(
                          patterns.front(), shutdownOutput,
                          &shutdownError),
                      "a stopped service must reject later work");
            t.IsTrue(shutdownOutput == shutdownSentinel,
                     "post-shutdown rejection must preserve caller output");
        });
    });
}
