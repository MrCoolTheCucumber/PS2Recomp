#include "MiniTest.h"
#include "runtime/ps2_gs_memory.h"
#include "runtime/ps2_gs_vulkan.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{
    struct PsmSpec
    {
        GSMem::PixelStorageMode psm;
        uint32_t pageWidth;
        uint32_t pageHeight;
        const char *name;
    };

    constexpr std::array<PsmSpec, 13> kPsmSpecs{{
        {GSMem::C32, 64u, 32u, "C32"},
        {GSMem::C24, 64u, 32u, "C24"},
        {GSMem::C16, 64u, 64u, "C16"},
        {GSMem::C16S, 64u, 64u, "C16S"},
        {GSMem::P8, 128u, 64u, "P8"},
        {GSMem::P4, 128u, 128u, "P4"},
        {GSMem::P8H, 64u, 32u, "P8H"},
        {GSMem::P4HL, 64u, 32u, "P4HL"},
        {GSMem::P4HH, 64u, 32u, "P4HH"},
        {GSMem::Z32, 64u, 32u, "Z32"},
        {GSMem::Z24, 64u, 32u, "Z24"},
        {GSMem::Z16, 64u, 64u, "Z16"},
        {GSMem::Z16S, 64u, 64u, "Z16S"},
    }};

    uint32_t nextRandom(uint32_t &state)
    {
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        return state;
    }

    std::vector<uint8_t> makeVramPattern(uint32_t seed)
    {
        std::vector<uint8_t> pattern(GS_VULKAN_VRAM_SIZE);
        uint32_t state = seed;
        for (uint8_t &byte : pattern)
        {
            nextRandom(state);
            byte = static_cast<uint8_t>(state >> 24u);
        }
        return pattern;
    }

    uint32_t cpuReadPixel(
        GSMem::PixelStorageMode psm, uint8_t *vram,
        uint32_t bp, uint32_t bw, uint32_t x, uint32_t y)
    {
        switch (psm)
        {
        case GSMem::C32:
            return GSMem::ReadCT32(vram, bp, bw, x, y);
        case GSMem::C24:
            return GSMem::ReadCT24(vram, bp, bw, x, y);
        case GSMem::C16:
            return GSMem::ReadCT16(vram, bp, bw, x, y);
        case GSMem::C16S:
            return GSMem::ReadCT16S(vram, bp, bw, x, y);
        case GSMem::P8:
            return GSMem::ReadP8(vram, bp, bw, x, y);
        case GSMem::P4:
            return GSMem::ReadP4(vram, bp, bw, x, y);
        case GSMem::P8H:
            return GSMem::ReadP8H(vram, bp, bw, x, y);
        case GSMem::P4HL:
            return GSMem::ReadP4HL(vram, bp, bw, x, y);
        case GSMem::P4HH:
            return GSMem::ReadP4HH(vram, bp, bw, x, y);
        case GSMem::Z32:
            return GSMem::ReadZ32(vram, bp, bw, x, y);
        case GSMem::Z24:
            return GSMem::ReadZ24(vram, bp, bw, x, y);
        case GSMem::Z16:
            return GSMem::ReadZ16(vram, bp, bw, x, y);
        case GSMem::Z16S:
            return GSMem::ReadZ16S(vram, bp, bw, x, y);
        default:
            return 0u;
        }
    }

    void cpuWritePixel(
        GSMem::PixelStorageMode psm, uint8_t *vram,
        uint32_t bp, uint32_t bw, uint32_t x, uint32_t y,
        uint32_t value)
    {
        switch (psm)
        {
        case GSMem::C32:
            GSMem::WriteCT32(vram, bp, bw, x, y, value);
            break;
        case GSMem::C24:
            GSMem::WriteCT24(vram, bp, bw, x, y, value);
            break;
        case GSMem::C16:
            GSMem::WriteCT16(vram, bp, bw, x, y, value);
            break;
        case GSMem::C16S:
            GSMem::WriteCT16S(vram, bp, bw, x, y, value);
            break;
        case GSMem::P8:
            GSMem::WriteP8(vram, bp, bw, x, y, value);
            break;
        case GSMem::P4:
            GSMem::WriteP4(vram, bp, bw, x, y, value);
            break;
        case GSMem::P8H:
            GSMem::WriteP8H(vram, bp, bw, x, y, value);
            break;
        case GSMem::P4HL:
            GSMem::WriteP4HL(vram, bp, bw, x, y, value);
            break;
        case GSMem::P4HH:
            GSMem::WriteP4HH(vram, bp, bw, x, y, value);
            break;
        case GSMem::Z32:
            GSMem::WriteZ32(vram, bp, bw, x, y, value);
            break;
        case GSMem::Z24:
            GSMem::WriteZ24(vram, bp, bw, x, y, value);
            break;
        case GSMem::Z16:
            GSMem::WriteZ16(vram, bp, bw, x, y, value);
            break;
        case GSMem::Z16S:
            GSMem::WriteZ16S(vram, bp, bw, x, y, value);
            break;
        default:
            break;
        }
    }

    std::vector<GsVulkanMemoryCase> makeMemoryReadCorpus()
    {
        std::vector<GsVulkanMemoryCase> cases;
        cases.reserve(61952u);
        for (const PsmSpec &spec : kPsmSpecs)
        {
            const uint32_t rawPsm =
                static_cast<uint32_t>(spec.psm);
            uint32_t ordinal = 0u;
            for (uint32_t y = 0u; y < spec.pageHeight; ++y)
            {
                for (uint32_t x = 0u; x < spec.pageWidth;
                     ++x, ++ordinal)
                {
                    const uint32_t block =
                        (ordinal * 17u + rawPsm) & 31u;
                    const uint32_t bp = ordinal % 17u == 0u
                        ? 0x3FE0u + block
                        : block;
                    cases.push_back({
                        rawPsm,
                        bp,
                        1u + ((ordinal * 11u + rawPsm) % 63u),
                        x,
                        y,
                        GsVulkanMemoryOperation::Read,
                        0u,
                        0u,
                    });
                }
            }

            uint32_t randomState =
                0x9E3779B9u ^ (rawPsm * 0x45D9F3Bu);
            for (uint32_t sample = 0u; sample < 512u; ++sample)
            {
                uint32_t bp = nextRandom(randomState) & 0x3FFFu;
                if ((sample & 3u) == 0u)
                    bp = 0x3FE0u + (bp & 31u);
                const uint32_t bw =
                    1u + (nextRandom(randomState) % 63u);
                const uint32_t x = nextRandom(randomState) & 0x7FFu;
                const uint32_t y = nextRandom(randomState) & 0x7FFu;
                cases.push_back({
                    rawPsm,
                    bp,
                    bw,
                    x,
                    y,
                    GsVulkanMemoryOperation::Read,
                    0u,
                    0u,
                });
            }
        }
        return cases;
    }

    std::vector<uint32_t> makeAddressTokenPattern(
        uint32_t bitWidth, uint32_t chunkOffset)
    {
        std::vector<uint32_t> words(
            GS_VULKAN_VRAM_SIZE / sizeof(uint32_t));
        const uint32_t mask = bitWidth == 32u
            ? 0xFFFFFFFFu
            : (1u << bitWidth) - 1u;
        const uint32_t laneStride = bitWidth == 24u
            ? 32u
            : bitWidth;
        for (uint32_t wordIndex = 0u;
             wordIndex < words.size(); ++wordIndex)
        {
            uint32_t encoded = 0u;
            for (uint32_t bitShift = 0u; bitShift < 32u;
                 bitShift += laneStride)
            {
                const uint32_t addressToken =
                    (wordIndex << 5u) | bitShift;
                encoded |=
                    ((addressToken >> chunkOffset) & mask) <<
                    bitShift;
            }
            words[wordIndex] = encoded;
        }
        return words;
    }

    std::string describeMemoryCase(
        const GsVulkanMemoryCase &memoryCase, size_t index)
    {
        std::ostringstream message;
        message << "case " << index
                << " psm=0x" << std::hex
                << memoryCase.pixelStorageMode
                << " bp=0x" << memoryCase.baseBlock
                << " bw=0x" << memoryCase.bufferWidth
                << " x=0x" << memoryCase.x
                << " y=0x" << memoryCase.y;
        return message.str();
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
            t.Equals(GS_VULKAN_MAX_MEMORY_CASES, 65536u,
                     "memory conformance batches should have a fixed host bound");
        });

        tc.Run("Compact PSM addresses match the CPU memory oracle", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const std::vector<GsVulkanMemoryCase> cases =
                makeMemoryReadCorpus();
            t.Equals(cases.size(), static_cast<size_t>(61952u),
                     "the corpus should cover every page pixel plus fixed-seed off-page cases");
            t.IsTrue(cases.size() <= GS_VULKAN_MAX_MEMORY_CASES,
                     "the exhaustive corpus must fit one bounded GPU dispatch");

            std::vector<GSMem::PixelAddress> addresses;
            addresses.reserve(cases.size());
            for (size_t index = 0u; index < cases.size(); ++index)
            {
                const GsVulkanMemoryCase &memoryCase = cases[index];
                GSMem::PixelAddress address{};
                const auto psm = static_cast<GSMem::PixelStorageMode>(
                    memoryCase.pixelStorageMode);
                if (!GSMem::ResolvePixelAddress(
                        psm,
                        memoryCase.baseBlock,
                        memoryCase.bufferWidth,
                        memoryCase.x,
                        memoryCase.y,
                        address))
                {
                    t.Fail("compact resolver rejected " +
                           describeMemoryCase(memoryCase, index));
                    return;
                }
                const uint32_t expectedWidth =
                    static_cast<uint32_t>(GSMem::BitsPerPixel(psm));
                if (address.packed_bit_width != expectedWidth ||
                    address.bit_shift + expectedWidth > 32u)
                {
                    t.Fail("compact resolver returned an invalid field for " +
                           describeMemoryCase(memoryCase, index));
                    return;
                }
                addresses.push_back(address);
            }

            constexpr std::array<uint32_t, 5> bitWidths{
                32u, 24u, 16u, 8u, 4u};
            for (uint32_t bitWidth : bitWidths)
            {
                const uint32_t passCount =
                    (25u + bitWidth - 1u) / bitWidth;
                const uint32_t valueMask = bitWidth == 32u
                    ? 0xFFFFFFFFu
                    : (1u << bitWidth) - 1u;
                for (uint32_t pass = 0u; pass < passCount; ++pass)
                {
                    const uint32_t chunkOffset = pass * bitWidth;
                    std::vector<uint32_t> encodedWords =
                        makeAddressTokenPattern(bitWidth, chunkOffset);
                    uint8_t *encodedVram = reinterpret_cast<uint8_t *>(
                        encodedWords.data());
                    for (size_t index = 0u;
                         index < cases.size(); ++index)
                    {
                        const auto psm =
                            static_cast<GSMem::PixelStorageMode>(
                                cases[index].pixelStorageMode);
                        if (GSMem::BitsPerPixel(psm) != bitWidth)
                            continue;
                        const GSMem::PixelAddress &address =
                            addresses[index];
                        const uint32_t addressToken =
                            (address.word_index << 5u) |
                            address.bit_shift;
                        const uint32_t expected =
                            (addressToken >> chunkOffset) & valueMask;
                        const uint32_t actual = cpuReadPixel(
                            psm, encodedVram,
                            cases[index].baseBlock,
                            cases[index].bufferWidth,
                            cases[index].x,
                            cases[index].y);
                        if (actual != expected)
                        {
                            std::ostringstream failure;
                            failure << "CPU lookup disagrees with compact address at "
                                    << describeMemoryCase(
                                           cases[index], index)
                                    << " pass=" << std::dec << pass
                                    << " expected=0x" << std::hex
                                    << expected << " actual=0x" << actual;
                            t.Fail(failure.str());
                            return;
                        }
                    }
                }
            }

            GSMem::PixelAddress invalid{
                0xAAAAAAAAu, 0xBBBBBBBBu, 0xCCCCCCCCu};
            t.IsFalse(GSMem::ResolvePixelAddress(
                          static_cast<GSMem::PixelStorageMode>(0x3Fu),
                          0u, 1u, 0u, 0u, invalid),
                      "unsupported PSMs must fail compact resolution");
            t.Equals(invalid.word_index, 0u,
                     "failed compact resolution should clear its result");
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

        tc.Run("Vulkan PSM helpers match CPU addresses reads and writes", [](TestCase &t)
        {
            GSMem::InitLookupTables();
            const std::vector<GsVulkanMemoryCase> readCases =
                makeMemoryReadCorpus();
            const std::vector<uint8_t> input =
                makeVramPattern(0x50534D31u);

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
                         "an unavailable host should skip shader PSM conformance cleanly");
                t.IsFalse(creationReport.ready(),
                          "the skipped conformance service must retain its capability result");
                return;
            }

            t.IsNotNull(service.get(),
                        "a suitable device should create the PSM conformance service");
            t.IsTrue(creationError.empty(),
                     "successful PSM service creation should clear its diagnostic");
            if (!service)
                return;

            std::vector<uint8_t> readOutput = {0xA5u};
            std::vector<GsVulkanMemoryResult> readResults = {{
                0xAAAAAAAAu, 0xBBBBBBBBu,
                0xCCCCCCCCu, 0xDDDDDDDDu}};
            std::string readError;
            if (!service->executeMemoryCases(
                    input, readCases, readOutput, readResults,
                    &readError))
            {
                t.Fail("exhaustive GPU PSM read batch failed: " +
                       readError);
                return;
            }
            t.IsTrue(readOutput == input,
                     "read-only PSM cases must preserve every VRAM byte");
            t.Equals(readResults.size(), readCases.size(),
                     "one GPU result must be returned for every read case");
            if (readResults.size() != readCases.size())
                return;

            std::vector<uint8_t> mutableInput = input;
            for (size_t index = 0u; index < readCases.size(); ++index)
            {
                const GsVulkanMemoryCase &memoryCase = readCases[index];
                const auto psm = static_cast<GSMem::PixelStorageMode>(
                    memoryCase.pixelStorageMode);
                GSMem::PixelAddress address{};
                const bool resolved = GSMem::ResolvePixelAddress(
                    psm,
                    memoryCase.baseBlock,
                    memoryCase.bufferWidth,
                    memoryCase.x,
                    memoryCase.y,
                    address);
                const uint32_t expectedValue = cpuReadPixel(
                    psm, mutableInput.data(),
                    memoryCase.baseBlock,
                    memoryCase.bufferWidth,
                    memoryCase.x,
                    memoryCase.y);
                const GsVulkanMemoryResult &actual =
                    readResults[index];
                if (!resolved ||
                    actual.wordIndex != address.word_index ||
                    actual.bitShift != address.bit_shift ||
                    actual.valueBefore != expectedValue ||
                    actual.valueAfter != expectedValue)
                {
                    std::ostringstream failure;
                    failure << "GPU read mismatch at "
                            << describeMemoryCase(memoryCase, index)
                            << " expected(word=0x" << std::hex
                            << address.word_index << ", shift=0x"
                            << address.bit_shift << ", value=0x"
                            << expectedValue << ") actual(word=0x"
                            << actual.wordIndex << ", shift=0x"
                            << actual.bitShift << ", before=0x"
                            << actual.valueBefore << ", after=0x"
                            << actual.valueAfter << ')';
                    t.Fail(failure.str());
                    return;
                }
            }

            for (size_t index = 0u; index < kPsmSpecs.size(); ++index)
            {
                const PsmSpec &spec = kPsmSpecs[index];
                GsVulkanMemoryCase writeCase{};
                writeCase.pixelStorageMode =
                    static_cast<uint32_t>(spec.psm);
                writeCase.baseBlock =
                    0x3FE0u +
                    ((static_cast<uint32_t>(index) * 5u + 31u) &
                     31u);
                writeCase.bufferWidth =
                    2u + 2u * (static_cast<uint32_t>(index) % 31u);
                writeCase.x =
                    spec.pageWidth *
                        (2u + static_cast<uint32_t>(index % 3u)) -
                    1u;
                writeCase.y =
                    spec.pageHeight *
                        (2u + static_cast<uint32_t>(index % 2u)) -
                    1u;
                writeCase.operation =
                    GsVulkanMemoryOperation::Write;
                writeCase.value =
                    0xDEADBEEFu ^
                    (static_cast<uint32_t>(index) * 0x10204081u);

                std::vector<uint8_t> expected = input;
                const uint32_t expectedBefore = cpuReadPixel(
                    spec.psm, expected.data(),
                    writeCase.baseBlock,
                    writeCase.bufferWidth,
                    writeCase.x,
                    writeCase.y);
                cpuWritePixel(
                    spec.psm, expected.data(),
                    writeCase.baseBlock,
                    writeCase.bufferWidth,
                    writeCase.x,
                    writeCase.y,
                    writeCase.value);
                const uint32_t expectedAfter = cpuReadPixel(
                    spec.psm, expected.data(),
                    writeCase.baseBlock,
                    writeCase.bufferWidth,
                    writeCase.x,
                    writeCase.y);
                GSMem::PixelAddress expectedAddress{};
                const bool resolved = GSMem::ResolvePixelAddress(
                    spec.psm,
                    writeCase.baseBlock,
                    writeCase.bufferWidth,
                    writeCase.x,
                    writeCase.y,
                    expectedAddress);

                std::vector<uint8_t> writeOutput = {0x5Au};
                std::vector<GsVulkanMemoryResult> writeResults = {{
                    0x11111111u, 0x22222222u,
                    0x33333333u, 0x44444444u}};
                std::string writeError;
                if (!service->executeMemoryCases(
                        input,
                        std::span<const GsVulkanMemoryCase>(
                            &writeCase, 1u),
                        writeOutput, writeResults, &writeError))
                {
                    t.Fail(std::string("GPU write failed for ") +
                           spec.name + ": " + writeError);
                    return;
                }
                if (!resolved || writeOutput != expected ||
                    writeResults.size() != 1u ||
                    writeResults[0].wordIndex !=
                        expectedAddress.word_index ||
                    writeResults[0].bitShift !=
                        expectedAddress.bit_shift ||
                    writeResults[0].valueBefore != expectedBefore ||
                    writeResults[0].valueAfter != expectedAfter)
                {
                    std::ostringstream failure;
                    failure << "GPU write disagrees with CPU RMW for "
                            << spec.name << " expected(word=0x"
                            << std::hex << expectedAddress.word_index
                            << ", shift=0x"
                            << expectedAddress.bit_shift
                            << ", before=0x" << expectedBefore
                            << ", after=0x" << expectedAfter << ')';
                    t.Fail(failure.str());
                    return;
                }
            }

            const auto expectRejected =
                [&](std::span<const uint8_t> rejectedInput,
                    std::span<const GsVulkanMemoryCase> rejectedCases,
                    const std::string &label)
            {
                std::vector<uint8_t> output = {0x12u, 0x34u};
                const std::vector<uint8_t> outputSentinel = output;
                std::vector<GsVulkanMemoryResult> results = {{
                    1u, 2u, 3u, 4u}};
                const std::vector<GsVulkanMemoryResult> resultSentinel =
                    results;
                std::string error;
                t.IsFalse(service->executeMemoryCases(
                              rejectedInput, rejectedCases,
                              output, results, &error),
                          label + " should fail closed");
                t.IsTrue(output == outputSentinel,
                         label + " must preserve VRAM output");
                t.IsTrue(results == resultSentinel,
                         label + " must preserve result output");
                t.IsFalse(error.empty(),
                          label + " should retain a diagnostic");
            };

            const std::vector<uint8_t> shortInput(
                GS_VULKAN_VRAM_SIZE - 1u, 0u);
            expectRejected(shortInput,
                           std::span<const GsVulkanMemoryCase>(
                               readCases.data(), 1u),
                           "short VRAM input");
            expectRejected(input, {}, "empty memory batch");
            std::vector<GsVulkanMemoryCase> tooMany(
                GS_VULKAN_MAX_MEMORY_CASES + 1u);
            expectRejected(input, tooMany, "oversized memory batch");

            GsVulkanMemoryCase invalidCase = readCases.front();
            invalidCase.pixelStorageMode = 0x3Fu;
            expectRejected(
                input,
                std::span<const GsVulkanMemoryCase>(&invalidCase, 1u),
                "unsupported PSM");
            invalidCase = readCases.front();
            invalidCase.operation =
                static_cast<GsVulkanMemoryOperation>(2u);
            expectRejected(
                input,
                std::span<const GsVulkanMemoryCase>(&invalidCase, 1u),
                "unsupported memory operation");
            invalidCase = readCases.front();
            invalidCase.reserved = 1u;
            expectRejected(
                input,
                std::span<const GsVulkanMemoryCase>(&invalidCase, 1u),
                "non-zero reserved data");

            const GsVulkanServiceStatistics statistics =
                service->statistics();
            constexpr uint64_t expectedBatches =
                1u + kPsmSpecs.size();
            t.Equals(statistics.memoryBatchesCompleted,
                     expectedBatches,
                     "all accepted PSM batches should complete exactly once");
            t.Equals(statistics.memoryBatchesFailed, 0ull,
                     "caller-side rejection should not count as failed GPU work");
            t.Equals(statistics.memoryCasesExecuted,
                     static_cast<uint64_t>(readCases.size()) +
                         kPsmSpecs.size(),
                     "memory case statistics should count every invocation");
            t.Equals(statistics.queueSubmissions, expectedBatches,
                     "each PSM batch should use one queue submission");
            t.Equals(statistics.shaderDispatches, expectedBatches,
                     "each PSM batch should use one compute dispatch");
            t.Equals(statistics.bytesUploaded,
                     expectedBatches * GS_VULKAN_VRAM_SIZE,
                     "each PSM batch should upload exact GS VRAM");
            t.Equals(statistics.bytesDownloaded,
                     expectedBatches * GS_VULKAN_VRAM_SIZE,
                     "each PSM batch should download exact GS VRAM");
            t.Equals(statistics.validationErrors, 0u,
                     "PSM shader execution must remain validation-clean");
            t.Equals(statistics.validationWarnings, 0u,
                     "PSM shader execution should emit no validation warnings");
            t.IsTrue(service->healthy(),
                     "successful and caller-rejected cases must leave the service healthy");

            service->shutdown();
            service->shutdown();
            const GsVulkanServiceStatistics shutdownStatistics =
                service->statistics();
            t.Equals(shutdownStatistics.validationErrors, 0u,
                     "PSM pipeline destruction must remain validation-clean");
            t.Equals(shutdownStatistics.validationWarnings, 0u,
                     "PSM pipeline destruction should emit no validation warnings");

            std::vector<uint8_t> shutdownOutput = {0xABu};
            const std::vector<uint8_t> shutdownOutputSentinel =
                shutdownOutput;
            std::vector<GsVulkanMemoryResult> shutdownResults = {{
                5u, 6u, 7u, 8u}};
            const std::vector<GsVulkanMemoryResult>
                shutdownResultSentinel = shutdownResults;
            std::string shutdownError;
            t.IsFalse(service->executeMemoryCases(
                          input,
                          std::span<const GsVulkanMemoryCase>(
                              readCases.data(), 1u),
                          shutdownOutput, shutdownResults,
                          &shutdownError),
                      "a stopped service must reject PSM work");
            t.IsTrue(shutdownOutput == shutdownOutputSentinel &&
                         shutdownResults == shutdownResultSentinel,
                     "post-shutdown PSM rejection must preserve both outputs");
        });
    });
}
