#include "runtime/ps2_gs_vulkan_backend.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

namespace
{
    namespace fs = std::filesystem;

    constexpr const char *kDefaultVerificationDirectory =
        "gs-vulkan-verify-failures";

    uint64_t fnv1a64(std::span<const uint8_t> bytes) noexcept
    {
        uint64_t hash = 14695981039346656037ull;
        for (uint8_t byte : bytes)
        {
            hash ^= byte;
            hash *= 1099511628211ull;
        }
        return hash;
    }

    bool writeBytes(
        const fs::path &path,
        std::span<const uint8_t> bytes,
        std::string &error)
    {
        std::ofstream output(
            path, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!output)
        {
            error = "failed to create " + path.string();
            return false;
        }
        output.write(
            reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
        if (!output)
        {
            error = "failed to write " + path.string();
            return false;
        }
        return true;
    }

    template <typename Value>
    std::string exactFloatingPoint(Value value)
    {
        std::ostringstream output;
        output << std::hexfloat
               << std::setprecision(
                      std::numeric_limits<Value>::max_digits10)
               << value;
        return output.str();
    }

    void writePageList(
        std::ostream &output,
        const GsVramPageMask &pages)
    {
        output << '[';
        bool first = true;
        for (size_t page = 0u; page < GS_VRAM_PAGE_COUNT; ++page)
        {
            if (!pages.test(page))
                continue;
            if (!first)
                output << ',';
            output << page;
            first = false;
        }
        output << ']';
    }

    bool writeCommandManifest(
        const fs::path &path,
        const GsDrawCommand &command,
        const GsVulkanCt32Sprite &sprite,
        size_t firstDifference,
        uint8_t softwareByte,
        uint8_t gpuByte,
        std::span<const uint8_t> initial,
        std::span<const uint8_t> software,
        std::span<const uint8_t> gpu,
        std::string &error)
    {
        std::ofstream output(path, std::ios::out | std::ios::trunc);
        if (!output)
        {
            error = "failed to create " + path.string();
            return false;
        }

        const GSPrimReg &primitive = command.primitive();
        const GSContext &context = command.context();
        const GsDrawGlobalState &global = command.globalState();
        const GsDrawBounds &bounds = command.bounds();
        const GsDrawResources resources = command.resources();
        output
            << "{\n"
            << "  \"schema_version\": 1,\n"
            << "  \"kind\": \"ps2-gs-vulkan-verification-mismatch\",\n"
            << "  \"sequence\": " << command.sequence() << ",\n"
            << "  \"state_signature\": \"0x" << std::hex
            << std::setw(16) << std::setfill('0')
            << command.stateSignature() << std::dec << "\",\n"
            << "  \"first_differing_byte\": "
            << firstDifference << ",\n"
            << "  \"first_differing_page\": "
            << firstDifference / GS_VRAM_PAGE_SIZE << ",\n"
            << "  \"first_differing_page_byte\": "
            << firstDifference % GS_VRAM_PAGE_SIZE << ",\n"
            << "  \"software_byte\": "
            << static_cast<uint32_t>(softwareByte) << ",\n"
            << "  \"gpu_byte\": "
            << static_cast<uint32_t>(gpuByte) << ",\n"
            << "  \"initial_fnv1a64\": \"0x" << std::hex
            << std::setw(16) << std::setfill('0') << fnv1a64(initial)
            << "\",\n"
            << "  \"software_fnv1a64\": \"0x"
            << std::setw(16) << std::setfill('0') << fnv1a64(software)
            << "\",\n"
            << "  \"gpu_fnv1a64\": \"0x"
            << std::setw(16) << std::setfill('0') << fnv1a64(gpu)
            << std::dec << "\",\n"
            << "  \"files\": {\"initial\": \"initial-vram.bin\", "
               "\"software\": \"software-vram.bin\", "
               "\"gpu\": \"gpu-vram.bin\"},\n"
            << "  \"sprite\": {"
            << "\"framebuffer_base_block\":"
            << sprite.framebufferBaseBlock << ','
            << "\"framebuffer_width\":"
            << sprite.framebufferWidth << ','
            << "\"x0\":" << sprite.x0 << ','
            << "\"y0\":" << sprite.y0 << ','
            << "\"x1\":" << sprite.x1 << ','
            << "\"y1\":" << sprite.y1 << ','
            << "\"rgba\":" << sprite.rgba << "},\n"
            << "  \"bounds\": {"
            << "\"x0\":" << bounds.x0 << ','
            << "\"y0\":" << bounds.y0 << ','
            << "\"x1\":" << bounds.x1 << ','
            << "\"y1\":" << bounds.y1 << ','
            << "\"exact\":" << (bounds.exact ? "true" : "false")
            << "},\n"
            << "  \"primitive\": {"
            << "\"type\":" << static_cast<uint32_t>(primitive.type) << ','
            << "\"iip\":" << (primitive.iip ? "true" : "false") << ','
            << "\"tme\":" << (primitive.tme ? "true" : "false") << ','
            << "\"fge\":" << (primitive.fge ? "true" : "false") << ','
            << "\"abe\":" << (primitive.abe ? "true" : "false") << ','
            << "\"aa1\":" << (primitive.aa1 ? "true" : "false") << ','
            << "\"fst\":" << (primitive.fst ? "true" : "false") << ','
            << "\"ctxt\":" << (primitive.ctxt ? "true" : "false") << ','
            << "\"fix\":" << (primitive.fix ? "true" : "false")
            << "},\n"
            << "  \"context\": {"
            << "\"frame\":{\"fbp\":" << context.frame.fbp
            << ",\"fbw\":" << context.frame.fbw
            << ",\"psm\":" << static_cast<uint32_t>(context.frame.psm)
            << ",\"fbmsk\":" << context.frame.fbmsk << "},"
            << "\"scissor\":{\"x0\":" << context.scissor.x0
            << ",\"x1\":" << context.scissor.x1
            << ",\"y0\":" << context.scissor.y0
            << ",\"y1\":" << context.scissor.y1 << "},"
            << "\"xyoffset\":{\"ofx\":" << context.xyoffset.ofx
            << ",\"ofy\":" << context.xyoffset.ofy << "},"
            << "\"zbuf\":{\"zbp\":" << context.zbuf.zbp
            << ",\"psm\":" << static_cast<uint32_t>(context.zbuf.psm)
            << ",\"zmask\":"
            << (context.zbuf.zmask ? "true" : "false") << "},"
            << "\"tex0\":{\"tbp0\":" << context.tex0.tbp0
            << ",\"tbw\":" << static_cast<uint32_t>(context.tex0.tbw)
            << ",\"psm\":" << static_cast<uint32_t>(context.tex0.psm)
            << ",\"tw\":" << static_cast<uint32_t>(context.tex0.tw)
            << ",\"th\":" << static_cast<uint32_t>(context.tex0.th)
            << ",\"tcc\":" << static_cast<uint32_t>(context.tex0.tcc)
            << ",\"tfx\":" << static_cast<uint32_t>(context.tex0.tfx)
            << ",\"cbp\":" << context.tex0.cbp
            << ",\"cpsm\":" << static_cast<uint32_t>(context.tex0.cpsm)
            << ",\"csm\":" << static_cast<uint32_t>(context.tex0.csm)
            << ",\"csa\":" << static_cast<uint32_t>(context.tex0.csa)
            << ",\"cld\":" << static_cast<uint32_t>(context.tex0.cld)
            << "},"
            << "\"tex1\":" << context.tex1 << ','
            << "\"miptbp1\":" << context.miptbp1 << ','
            << "\"miptbp2\":" << context.miptbp2 << ','
            << "\"clamp\":" << context.clamp << ','
            << "\"alpha\":" << context.alpha << ','
            << "\"test\":" << context.test << ','
            << "\"fba\":" << context.fba << "},\n"
            << "  \"global\": {"
            << "\"texa\":{\"ta0\":"
            << static_cast<uint32_t>(global.texa.ta0)
            << ",\"aem\":" << (global.texa.aem ? "true" : "false")
            << ",\"ta1\":" << static_cast<uint32_t>(global.texa.ta1)
            << "},\"texclut\":{\"cbw\":"
            << static_cast<uint32_t>(global.texclut.cbw)
            << ",\"cou\":" << static_cast<uint32_t>(global.texclut.cou)
            << ",\"cov\":" << global.texclut.cov << "},"
            << "\"fog_color\":" << global.fogColor << ','
            << "\"dimx\":" << global.dimx << ','
            << "\"scan_mask\":" << static_cast<uint32_t>(global.scanMask)
            << ",\"prmodecont\":"
            << (global.prmodecont ? "true" : "false")
            << ",\"pabe\":" << (global.pabe ? "true" : "false")
            << ",\"dither\":" << (global.dither ? "true" : "false")
            << ",\"color_clamp\":"
            << (global.colorClamp ? "true" : "false") << "},\n"
            << "  \"vertices\": [";

        for (size_t index = 0u; index < command.vertexCount(); ++index)
        {
            if (index != 0u)
                output << ',';
            const GSVertex &vertex = command.vertices()[index];
            output
                << "{\"x12_4\":" << vertex.x12_4
                << ",\"y12_4\":" << vertex.y12_4
                << ",\"z_integer\":" << vertex.zInteger
                << ",\"r\":" << static_cast<uint32_t>(vertex.r)
                << ",\"g\":" << static_cast<uint32_t>(vertex.g)
                << ",\"b\":" << static_cast<uint32_t>(vertex.b)
                << ",\"a\":" << static_cast<uint32_t>(vertex.a)
                << ",\"u\":" << vertex.u
                << ",\"v\":" << vertex.v
                << ",\"fog\":" << static_cast<uint32_t>(vertex.fog)
                << ",\"x_float\":\""
                << exactFloatingPoint(vertex.x)
                << "\",\"y_float\":\""
                << exactFloatingPoint(vertex.y)
                << "\",\"z_float\":\""
                << exactFloatingPoint(vertex.z)
                << "\",\"q\":\""
                << exactFloatingPoint(vertex.q)
                << "\",\"s\":\""
                << exactFloatingPoint(vertex.s)
                << "\",\"t\":\""
                << exactFloatingPoint(vertex.t) << "\"}";
        }
        output << "],\n  \"fixed_x\": [";
        for (size_t index = 0u; index < command.vertexCount(); ++index)
        {
            if (index != 0u)
                output << ',';
            output << command.fixedX()[index];
        }
        output << "],\n  \"fixed_y\": [";
        for (size_t index = 0u; index < command.vertexCount(); ++index)
        {
            if (index != 0u)
                output << ',';
            output << command.fixedY()[index];
        }
        output << "],\n  \"read_pages\": ";
        writePageList(output, resources.readPages);
        output << ",\n  \"write_pages\": ";
        writePageList(output, resources.writePages);
        output << "\n}\n";
        if (!output)
        {
            error = "failed to write " + path.string();
            return false;
        }
        return true;
    }

    bool chooseArtifactPaths(
        const std::string &configuredDirectory,
        uint64_t sequence,
        fs::path &partial,
        fs::path &completed,
        std::string &error)
    {
        fs::path root;
        if (!configuredDirectory.empty())
        {
            root = configuredDirectory;
        }
        else if (const char *environment =
                     std::getenv("PS2X_GS_VERIFY_DUMP_DIR");
                 environment && environment[0] != '\0')
        {
            root = environment;
        }
        else
        {
            root = kDefaultVerificationDirectory;
        }

        std::error_code filesystemError;
        fs::create_directories(root, filesystemError);
        if (filesystemError)
        {
            error = "failed to create verification artifact root " +
                    root.string() + ": " + filesystemError.message();
            return false;
        }

        for (uint32_t suffix = 0u; suffix < 10000u; ++suffix)
        {
            std::ostringstream leaf;
            leaf << "draw-" << std::setw(20) << std::setfill('0')
                 << sequence;
            if (suffix != 0u)
                leaf << '-' << suffix;
            completed = root / leaf.str();
            partial = root / (leaf.str() + ".partial");
            filesystemError.clear();
            const bool completedExists =
                fs::exists(completed, filesystemError);
            if (filesystemError)
            {
                error = "failed to inspect verification artifact " +
                        completed.string() + ": " +
                        filesystemError.message();
                return false;
            }
            if (completedExists)
                continue;
            if (fs::create_directory(partial, filesystemError))
                return true;
            if (filesystemError &&
                filesystemError != std::errc::file_exists)
            {
                error = "failed to create verification artifact " +
                        partial.string() + ": " +
                        filesystemError.message();
                return false;
            }
        }

        error = "verification artifact suffix space is exhausted";
        return false;
    }

    bool writeVerificationArtifact(
        const std::string &configuredDirectory,
        const GsDrawCommand &command,
        const GsVulkanCt32Sprite &sprite,
        size_t firstDifference,
        std::span<const uint8_t> initial,
        std::span<const uint8_t> software,
        std::span<const uint8_t> gpu,
        std::string &artifactPath,
        std::string &error)
    {
        fs::path partial;
        fs::path completed;
        if (!chooseArtifactPaths(
                configuredDirectory, command.sequence(),
                partial, completed, error))
        {
            return false;
        }

        artifactPath = partial.string();
        if (!writeBytes(partial / "initial-vram.bin", initial, error) ||
            !writeBytes(partial / "software-vram.bin", software, error) ||
            !writeBytes(partial / "gpu-vram.bin", gpu, error) ||
            !writeCommandManifest(
                partial / "command.json", command, sprite,
                firstDifference, software[firstDifference],
                gpu[firstDifference], initial, software, gpu, error))
        {
            return false;
        }

        std::error_code filesystemError;
        fs::rename(partial, completed, filesystemError);
        if (filesystemError)
        {
            error = "failed to publish verification artifact " +
                    completed.string() + ": " +
                    filesystemError.message();
            return false;
        }
        artifactPath = completed.string();
        error.clear();
        return true;
    }

    bool isAcceleratedMode(GsRendererMode mode) noexcept
    {
        return mode == GsRendererMode::Hybrid ||
               mode == GsRendererMode::Verify ||
               mode == GsRendererMode::GpuStrict;
    }
}

struct GsVulkanRasterBackend::Impl final
{
    struct PendingResidentCommand
    {
        GsDrawCommand command;
        GsVulkanCt32Sprite sprite;
        GsDrawResources resources;
    };

    std::unique_ptr<IGsVulkanDrawExecutor> executor;
    GsVulkanRasterBackendConfig config;
    std::span<uint8_t> canonicalVram;
    DrawCallback softwareOracle;
    DrawCallback acceleratedCommit;
    GsVulkanRasterBackendStatistics statistics;
    GsVramCoherency coherency;
    std::vector<PendingResidentCommand> pendingResidentCommands;
    GsVramPageMask pendingResidentAccessPages;
    bool failed = false;
    bool shutDown = false;

    [[noreturn]] void failRequest(
        std::string_view operation,
        std::string detail)
    {
        ++statistics.gpuRequestsFailed;
        failed = true;
        if (detail.empty())
            detail = "executor rejected the request";
        throw std::runtime_error(
            "Vulkan GS " + std::string(operation) +
            " failed before canonical VRAM mutation: " + detail);
    }

    void uploadCpuNewer(
        const GsVramPageMask &within,
        std::string_view operation)
    {
        const GsVramPageMask pages = coherency.cpuNewerPages(within);
        if (!pages.any())
            return;

        std::string executionError;
        if (!executor->uploadVramPages(
                canonicalVram, pages, &executionError))
        {
            failRequest(operation, std::move(executionError));
        }
        coherency.completeCpuToGpu(pages);
    }

    void downloadGpuNewer(
        const GsVramPageMask &within,
        GsFlushReason reason)
    {
        const GsVramPageMask pages = coherency.gpuNewerPages(within);
        if (!pages.any())
            return;
        if (failed || shutDown || !executor->healthy())
        {
            failRequest(
                "page download at " +
                    std::string(gsFlushReasonName(reason)),
                "resident GPU-newer VRAM is unavailable");
        }

        std::string executionError;
        if (!executor->downloadVramPages(
                canonicalVram, pages, &executionError))
        {
            failRequest(
                "page download at " +
                    std::string(gsFlushReasonName(reason)),
                std::move(executionError));
        }
        coherency.completeGpuToCpu(pages);
    }

    void clearPendingResidentCommands() noexcept
    {
        pendingResidentCommands.clear();
        pendingResidentAccessPages.clear();
    }

    void drainPendingResidentCommands(GsFlushReason reason)
    {
        if (pendingResidentCommands.empty())
            return;

        const size_t commandCount = pendingResidentCommands.size();
        if (reason == GsFlushReason::ResourceHazard)
            ++statistics.resourceHazardDrains;
        else if (reason == GsFlushReason::QueueBackpressure)
            ++statistics.queueBackpressureDrains;

        std::vector<GsVulkanCt32Sprite> sprites;
        sprites.reserve(commandCount);
        for (const PendingResidentCommand &pending :
             pendingResidentCommands)
        {
            sprites.push_back(pending.sprite);
        }

        try
        {
            uploadCpuNewer(
                pendingResidentAccessPages,
                "page upload for resident CT32 batch at " +
                    std::string(gsFlushReasonName(reason)));
        }
        catch (...)
        {
            clearPendingResidentCommands();
            throw;
        }

        std::string executionError;
        if (!executor->executeResidentCt32Sprites(
                sprites, &executionError))
        {
            clearPendingResidentCommands();
            failRequest(
                "resident CT32 batch at " +
                    std::string(gsFlushReasonName(reason)),
                std::move(executionError));
        }

        for (const PendingResidentCommand &pending :
             pendingResidentCommands)
        {
            coherency.noteGpuWrite(pending.resources.writePages);
        }
        statistics.committedGpuCommands += commandCount;
        statistics.residentCommands += commandCount;
        statistics.commandsCompleted += commandCount;
        ++statistics.residentBatchesCompleted;
        statistics.largestResidentBatch = std::max(
            statistics.largestResidentBatch,
            static_cast<uint64_t>(commandCount));
        try
        {
            if (acceleratedCommit)
            {
                for (const PendingResidentCommand &pending :
                     pendingResidentCommands)
                {
                    acceleratedCommit(pending.command);
                }
            }
        }
        catch (...)
        {
            clearPendingResidentCommands();
            throw;
        }
        clearPendingResidentCommands();
    }
};

GsVulkanRasterBackend::GsVulkanRasterBackend(
    std::unique_ptr<Impl> impl)
    : m_impl(std::move(impl))
{
}

GsVulkanRasterBackend::~GsVulkanRasterBackend()
{
    if (m_impl && !m_impl->shutDown)
    {
        try
        {
            flush(GsFlushReason::Shutdown);
        }
        catch (...)
        {
            m_impl->executor->shutdown();
            m_impl->shutDown = true;
        }
    }
}

std::unique_ptr<GsVulkanRasterBackend>
GsVulkanRasterBackend::create(
    const GsVulkanServiceConfig &serviceConfig,
    const GsVulkanRasterBackendConfig &backendConfig,
    std::span<uint8_t> canonicalVram,
    DrawCallback softwareOracle,
    DrawCallback acceleratedCommit,
    GsVulkanCapabilityReport *report,
    std::string *error)
{
    std::unique_ptr<GsVulkanService> service =
        GsVulkanService::create(serviceConfig, report, error);
    if (!service)
        return nullptr;
    return createWithExecutor(
        std::move(service), backendConfig, canonicalVram,
        std::move(softwareOracle), std::move(acceleratedCommit), error);
}

std::unique_ptr<GsVulkanRasterBackend>
GsVulkanRasterBackend::createWithExecutor(
    std::unique_ptr<IGsVulkanDrawExecutor> executor,
    const GsVulkanRasterBackendConfig &backendConfig,
    std::span<uint8_t> canonicalVram,
    DrawCallback softwareOracle,
    DrawCallback acceleratedCommit,
    std::string *error)
{
    const auto reject = [&](std::string message)
    {
        if (error)
            *error = std::move(message);
        return std::unique_ptr<GsVulkanRasterBackend>{};
    };
    if (!executor)
        return reject("Vulkan raster backend requires an executor");
    if (!executor->healthy())
        return reject("Vulkan raster backend executor is not healthy");
    if (canonicalVram.size() != GS_VULKAN_VRAM_SIZE)
    {
        return reject(
            "Vulkan raster backend requires exactly 4 MiB of canonical VRAM");
    }
    if (!softwareOracle)
        return reject("Vulkan raster backend requires a software oracle");
    if (!isAcceleratedMode(backendConfig.mode))
        return reject("Vulkan raster backend requires an accelerated mode");
    if (backendConfig.maximumResidentBatchCommands == 0u ||
        backendConfig.maximumResidentBatchCommands >
            GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH)
    {
        return reject(
            "Vulkan raster backend resident batch bound must be between 1 and " +
            std::to_string(GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH));
    }

    auto impl = std::make_unique<Impl>();
    impl->executor = std::move(executor);
    impl->config = backendConfig;
    impl->canonicalVram = canonicalVram;
    impl->softwareOracle = std::move(softwareOracle);
    impl->acceleratedCommit = std::move(acceleratedCommit);
    impl->pendingResidentCommands.reserve(
        backendConfig.maximumResidentBatchCommands);
    GsVramPageMask allPages;
    allPages.setAll();
    // The canonical image predates the device allocation. Start with explicit
    // CPU ownership so the first resident draw uploads only what it can touch.
    impl->coherency.noteCpuWrite(allPages);
    impl->coherency.resetStatistics();
    if (error)
        error->clear();
    return std::unique_ptr<GsVulkanRasterBackend>(
        new GsVulkanRasterBackend(std::move(impl)));
}

bool GsVulkanRasterBackend::setMode(GsRendererMode mode) noexcept
{
    if (!isAcceleratedMode(mode) || m_impl->shutDown || m_impl->failed)
        return false;
    m_impl->config.mode = mode;
    return true;
}

GsRendererMode GsVulkanRasterBackend::mode() const noexcept
{
    return m_impl->config.mode;
}

GsBackendDecision GsVulkanRasterBackend::classify(
    const GsDrawCommand &command) const
{
    if (m_impl->shutDown || m_impl->failed ||
        !m_impl->executor->healthy())
        return {false, GsFallbackReason::BackendUnavailable};
    GsVulkanCt32Sprite sprite{};
    const GsBackendDecision decision =
        prepareGsVulkanCt32Sprite(command, sprite);
    if (!decision.supported ||
        m_impl->config.mode != GsRendererMode::Hybrid)
    {
        return decision;
    }

    const uint64_t pixels =
        static_cast<uint64_t>(sprite.x1 - sprite.x0) *
        static_cast<uint64_t>(sprite.y1 - sprite.y0);
    if (pixels < m_impl->config.minimumHybridSpritePixels)
        return {false, GsFallbackReason::CostModel};
    return decision;
}

void GsVulkanRasterBackend::submit(
    std::span<const GsDrawCommand> commands)
{
    // Preserve the router contract for callers that submit a future batch:
    // one unsupported member rejects the whole span before canonical VRAM is
    // changed by an earlier member.
    for (const GsDrawCommand &command : commands)
    {
        const GsBackendDecision decision = classify(command);
        if (!decision.supported)
        {
            throw std::logic_error(
                "Vulkan raster backend received unsupported draw " +
                std::to_string(command.sequence()) + ": " +
                std::string(gsFallbackReasonName(decision.reason)));
        }
    }

    for (const GsDrawCommand &command : commands)
    {
        GsVulkanCt32Sprite sprite{};
        (void)prepareGsVulkanCt32Sprite(command, sprite);
        const GsDrawResources resources = command.resources();

        ++m_impl->statistics.commandsAttempted;
        if (m_impl->config.mode == GsRendererMode::Verify)
        {
            GsVramPageMask allPages;
            allPages.setAll();
            prepareCpuVramAccess(
                allPages, GsFlushReason::BackendSwitch);
            std::vector<uint8_t> initial(
                m_impl->canonicalVram.begin(),
                m_impl->canonicalVram.end());
            std::vector<uint8_t> gpuOutput;
            std::string executionError;
            if (!m_impl->executor->executeCt32Sprite(
                    initial, sprite, gpuOutput, &executionError) ||
                gpuOutput.size() != GS_VULKAN_VRAM_SIZE)
            {
                if (executionError.empty())
                {
                    executionError =
                        "executor returned an invalid VRAM image";
                }
                m_impl->failRequest(
                    "CT32 verification draw " +
                        std::to_string(command.sequence()),
                    std::move(executionError));
            }

            // executeCt32Sprite uploaded the entire initial image. Record that
            // fact before the independent CPU oracle creates its newer result.
            const GsVramPageMask initiallyCpuNewer =
                m_impl->coherency.cpuNewerPages(allPages);
            m_impl->coherency.completeCpuToGpu(initiallyCpuNewer);
            m_impl->softwareOracle(command);
            m_impl->coherency.noteCpuWrite(resources.writePages);
            m_impl->statistics.bytesCompared += GS_VULKAN_VRAM_SIZE;
            const auto difference = std::mismatch(
                m_impl->canonicalVram.begin(),
                m_impl->canonicalVram.end(),
                gpuOutput.begin(), gpuOutput.end());
            if (difference.first != m_impl->canonicalVram.end())
            {
                const size_t firstDifference =
                    static_cast<size_t>(
                        difference.first -
                        m_impl->canonicalVram.begin());
                ++m_impl->statistics.verificationMismatches;
                m_impl->failed = true;
                std::string artifactPath;
                std::string artifactError;
                const bool artifactWritten = writeVerificationArtifact(
                    m_impl->config.verificationArtifactDirectory,
                    command, sprite, firstDifference,
                    initial, m_impl->canonicalVram, gpuOutput,
                    artifactPath, artifactError);
                m_impl->statistics.lastVerificationArtifact = artifactPath;
                std::ostringstream message;
                message
                    << "Vulkan GS verification mismatch for draw "
                    << command.sequence() << " at byte "
                    << firstDifference << " (page "
                    << firstDifference / GS_VRAM_PAGE_SIZE
                    << ", page byte "
                    << firstDifference % GS_VRAM_PAGE_SIZE
                    << "): software="
                    << static_cast<uint32_t>(*difference.first)
                    << " gpu="
                    << static_cast<uint32_t>(*difference.second);
                if (artifactWritten)
                    message << "; artifact=" << artifactPath;
                else
                    message << "; artifact failure=" << artifactError;
                throw std::runtime_error(message.str());
            }
            // A complete byte comparison proves that the independently written
            // GPU pages now contain the same generation as the CPU oracle.
            m_impl->coherency.completeCpuToGpu(resources.writePages);
            ++m_impl->statistics.verifiedCommands;
        }
        else
        {
            GsVramPageMask accessPages = resources.readPages;
            accessPages.unionWith(resources.writePages);
            if (m_impl->pendingResidentAccessPages.intersects(
                    accessPages))
            {
                m_impl->drainPendingResidentCommands(
                    GsFlushReason::ResourceHazard);
            }
            if (m_impl->pendingResidentCommands.size() >=
                m_impl->config.maximumResidentBatchCommands)
            {
                m_impl->drainPendingResidentCommands(
                    GsFlushReason::QueueBackpressure);
            }

            m_impl->pendingResidentCommands.push_back(
                {command, sprite, resources});
            m_impl->pendingResidentAccessPages.unionWith(accessPages);
        }
        if (m_impl->config.mode == GsRendererMode::Verify)
            ++m_impl->statistics.commandsCompleted;
    }
}

void GsVulkanRasterBackend::flush(GsFlushReason reason)
{
    ++m_impl->statistics.flushes;
    if (reason == GsFlushReason::Shutdown && !m_impl->shutDown)
    {
        GsVramPageMask allPages;
        allPages.setAll();
        try
        {
            prepareCpuVramAccess(allPages, reason);
        }
        catch (...)
        {
            // Shutdown is commonly reached from a noexcept owner destructor.
            // The failed flag and request counters retain the diagnosis; an
            // unavailable GPU-newer image cannot be repaired during teardown.
        }
        m_impl->executor->shutdown();
        m_impl->shutDown = true;
    }
    else
    {
        m_impl->drainPendingResidentCommands(reason);
    }
}

size_t GsVulkanRasterBackend::pendingCommandCount() const noexcept
{
    return m_impl->pendingResidentCommands.size();
}

void GsVulkanRasterBackend::prepareCpuVramAccess(
    const GsVramPageMask &pages,
    GsFlushReason reason)
{
    if (!pages.any())
        return;
    m_impl->drainPendingResidentCommands(reason);
    ++m_impl->statistics.cpuAccessPreparations;
    m_impl->downloadGpuNewer(pages, reason);
}

void GsVulkanRasterBackend::noteCpuVramWrite(
    const GsVramPageMask &pages)
{
    m_impl->coherency.noteCpuWrite(pages);
}

GsVulkanRasterBackendStatistics
GsVulkanRasterBackend::backendStatistics() const
{
    GsVulkanRasterBackendStatistics statistics = m_impl->statistics;
    statistics.pageOwnership = m_impl->coherency.summary();
    statistics.coherency = m_impl->coherency.statistics();
    return statistics;
}

GsVulkanCapabilityReport GsVulkanRasterBackend::capabilities() const
{
    return m_impl->executor->capabilities();
}

GsVulkanServiceStatistics
GsVulkanRasterBackend::serviceStatistics() const
{
    return m_impl->executor->statistics();
}

bool GsVulkanRasterBackend::healthy() const
{
    return !m_impl->shutDown && !m_impl->failed &&
           m_impl->executor->healthy();
}
