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

    void writePreparedRecord(
        std::ostream &output,
        const GsVulkanCt32Sprite &sprite)
    {
        output
            << "  \"sprite\": {"
            << "\"framebuffer_base_block\":"
            << sprite.framebufferBaseBlock << ','
            << "\"framebuffer_width\":"
            << sprite.framebufferWidth << ','
            << "\"x0\":" << sprite.x0 << ','
            << "\"y0\":" << sprite.y0 << ','
            << "\"x1\":" << sprite.x1 << ','
            << "\"y1\":" << sprite.y1 << ','
            << "\"rgba\":" << sprite.rgba << '}';
    }

    void writePreparedRecord(
        std::ostream &output,
        const GsVulkanCt32Triangle &triangle)
    {
        output
            << "  \"triangle\": {"
            << "\"framebuffer_base_block\":"
            << triangle.framebufferBaseBlock << ','
            << "\"framebuffer_width\":"
            << triangle.framebufferWidth << ','
            << "\"bounds_x0\":" << triangle.boundsX0 << ','
            << "\"bounds_y0\":" << triangle.boundsY0 << ','
            << "\"bounds_x1\":" << triangle.boundsX1 << ','
            << "\"bounds_y1\":" << triangle.boundsY1 << ','
            << "\"vertex0_x_12_4\":" << triangle.vertex0X12_4 << ','
            << "\"vertex0_y_12_4\":" << triangle.vertex0Y12_4 << ','
            << "\"vertex1_x_12_4\":" << triangle.vertex1X12_4 << ','
            << "\"vertex1_y_12_4\":" << triangle.vertex1Y12_4 << ','
            << "\"vertex2_x_12_4\":" << triangle.vertex2X12_4 << ','
            << "\"vertex2_y_12_4\":" << triangle.vertex2Y12_4 << ','
            << "\"rgba\":" << triangle.rgba << ','
            << "\"top_left_edge_mask\":"
            << triangle.topLeftEdgeMask << '}';
    }

    void writePreparedRecord(
        std::ostream &output,
        const GsVulkanNearestCt32Sprite &sprite)
    {
        output
            << "  \"nearest_ct32_sprite\": {"
            << "\"framebuffer_base_block\":"
            << sprite.framebufferBaseBlock << ','
            << "\"framebuffer_width\":"
            << sprite.framebufferWidth << ','
            << "\"bounds_x0\":" << sprite.boundsX0 << ','
            << "\"bounds_y0\":" << sprite.boundsY0 << ','
            << "\"bounds_x1\":" << sprite.boundsX1 << ','
            << "\"bounds_y1\":" << sprite.boundsY1 << ','
            << "\"texture_base_block\":"
            << sprite.textureBaseBlock << ','
            << "\"texture_width\":" << sprite.textureWidth << ','
            << "\"texture_mask_u\":" << sprite.textureMaskU << ','
            << "\"texture_mask_v\":" << sprite.textureMaskV << ','
            << "\"texture_origin_u\":" << sprite.textureOriginU << ','
            << "\"texture_origin_v\":" << sprite.textureOriginV << ','
            << "\"texture_step_u\":" << sprite.textureStepU << ','
            << "\"texture_step_v\":" << sprite.textureStepV << ','
            << "\"texture_wrap_mode_u\":"
            << static_cast<uint32_t>(
                   gsVulkanTextureWrapMode(sprite.textureWrapU)) << ','
            << "\"texture_wrap_mode_v\":"
            << static_cast<uint32_t>(
                   gsVulkanTextureWrapMode(sprite.textureWrapV)) << ','
            << "\"texture_region_min_u\":"
            << gsVulkanTextureRegionMin(sprite.textureWrapU) << ','
            << "\"texture_region_max_u\":"
            << gsVulkanTextureRegionMax(sprite.textureWrapU) << ','
            << "\"texture_region_min_v\":"
            << gsVulkanTextureRegionMin(sprite.textureWrapV) << ','
            << "\"texture_region_max_v\":"
            << gsVulkanTextureRegionMax(sprite.textureWrapV) << '}';
    }

    void writePreparedRecord(
        std::ostream &output,
        const GsVulkanLinearCt32Sprite &sprite)
    {
        output
            << "  \"linear_ct32_sprite\": {"
            << "\"framebuffer_base_block\":"
            << sprite.framebufferBaseBlock << ','
            << "\"framebuffer_width\":"
            << sprite.framebufferWidth << ','
            << "\"bounds_x0\":" << sprite.boundsX0 << ','
            << "\"bounds_y0\":" << sprite.boundsY0 << ','
            << "\"bounds_x1\":" << sprite.boundsX1 << ','
            << "\"bounds_y1\":" << sprite.boundsY1 << ','
            << "\"texture_base_block\":"
            << sprite.textureBaseBlock << ','
            << "\"texture_width\":" << sprite.textureWidth << ','
            << "\"texture_mask_u\":" << sprite.textureMaskU << ','
            << "\"texture_mask_v\":" << sprite.textureMaskV << ','
            << "\"fixed_base_u\":" << sprite.fixedBaseU << ','
            << "\"fixed_block_step_u\":"
            << sprite.fixedBlockStepU << ','
            << "\"fixed_lane_u\":[";
        for (size_t lane = 0u; lane < sprite.fixedLaneU.size(); ++lane)
        {
            if (lane != 0u)
                output << ',';
            output << sprite.fixedLaneU[lane];
        }
        output
            << "],\"fixed_scan_v_bits\":"
            << sprite.fixedScanVBits << ','
            << "\"fixed_step_v_bits\":"
            << sprite.fixedStepVBits << ','
            << "\"texture_wrap_u\":" << sprite.textureWrapU << ','
            << "\"texture_wrap_v\":" << sprite.textureWrapV << '}';
    }

    template <typename PreparedRecord>
    bool writeCommandManifest(
        const fs::path &path,
        const GsDrawCommand &command,
        const PreparedRecord &record,
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
               "\"gpu\": \"gpu-vram.bin\"},\n";
        writePreparedRecord(output, record);
        output
            << ",\n  \"bounds\": {"
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

    template <typename PreparedRecord>
    bool writeVerificationArtifact(
        const std::string &configuredDirectory,
        const GsDrawCommand &command,
        const PreparedRecord &record,
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
                partial / "command.json", command, record,
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
    enum class ResidentPipeline : uint8_t
    {
        Ct32Sprite,
        NearestCt32Sprite,
        LinearCt32Sprite,
        Ct32Triangle,
    };

    struct PendingResidentCommand
    {
        GsDrawCommand command;
        GsVulkanCt32Sprite sprite;
        GsVulkanNearestCt32Sprite nearestCt32Sprite;
        GsVulkanLinearCt32Sprite linearCt32Sprite;
        GsVulkanCt32Triangle triangle;
        GsDrawResources resources;
        ResidentPipeline pipeline = ResidentPipeline::Ct32Sprite;
    };

    std::unique_ptr<IGsVulkanDrawExecutor> executor;
    GsVulkanRasterBackendConfig config;
    std::span<uint8_t> canonicalVram;
    DrawCallback softwareOracle;
    DrawCallback acceleratedCommit;
    GsVulkanRasterBackendStatistics statistics;
    GsVramCoherency coherency;
    std::vector<PendingResidentCommand> pendingResidentCommands;
    GsVramPageMask pendingResidentReadPages;
    GsVramPageMask pendingResidentWritePages;
    bool exactCt32Triangle = false;
    bool exactNearestCt32Sprite = false;
    bool exactLinearCt32Sprite = false;
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
        pendingResidentReadPages.clear();
        pendingResidentWritePages.clear();
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
        else if (reason == GsFlushReason::PipelineChange)
            ++statistics.pipelineChangeDrains;

        const ResidentPipeline pipeline =
            pendingResidentCommands.front().pipeline;
        std::vector<GsVulkanCt32Sprite> sprites;
        std::vector<GsVulkanNearestCt32Sprite> nearestCt32Sprites;
        std::vector<GsVulkanLinearCt32Sprite> linearCt32Sprites;
        std::vector<GsVulkanCt32Triangle> triangles;
        if (pipeline == ResidentPipeline::Ct32Triangle)
            triangles.reserve(commandCount);
        else if (pipeline == ResidentPipeline::LinearCt32Sprite)
            linearCt32Sprites.reserve(commandCount);
        else if (pipeline == ResidentPipeline::NearestCt32Sprite)
            nearestCt32Sprites.reserve(commandCount);
        else
            sprites.reserve(commandCount);
        for (const PendingResidentCommand &pending :
             pendingResidentCommands)
        {
            if (pending.pipeline != pipeline)
            {
                clearPendingResidentCommands();
                failRequest(
                    "resident CT32 batch assembly",
                    "mixed pipelines reached a homogeneous service batch");
            }
            if (pipeline == ResidentPipeline::Ct32Triangle)
                triangles.push_back(pending.triangle);
            else if (pipeline == ResidentPipeline::LinearCt32Sprite)
                linearCt32Sprites.push_back(
                    pending.linearCt32Sprite);
            else if (pipeline == ResidentPipeline::NearestCt32Sprite)
                nearestCt32Sprites.push_back(
                    pending.nearestCt32Sprite);
            else
                sprites.push_back(pending.sprite);
        }

        const char *batchName = "resident CT32 sprite batch";
        if (pipeline == ResidentPipeline::NearestCt32Sprite)
            batchName = "resident nearest CT32 sprite batch";
        else if (pipeline == ResidentPipeline::LinearCt32Sprite)
            batchName = "resident linear CT32 sprite batch";
        else if (pipeline == ResidentPipeline::Ct32Triangle)
            batchName = "resident CT32 triangle batch";
        GsVramPageMask accessPages = pendingResidentReadPages;
        accessPages.unionWith(pendingResidentWritePages);

        try
        {
            uploadCpuNewer(
                accessPages,
                "page upload for " + std::string(batchName) + " at " +
                    std::string(gsFlushReasonName(reason)));
        }
        catch (...)
        {
            clearPendingResidentCommands();
            throw;
        }

        std::string executionError;
        bool executed = false;
        if (pipeline == ResidentPipeline::Ct32Triangle)
        {
            executed = executor->executeResidentCt32Triangles(
                triangles, &executionError);
        }
        else if (pipeline == ResidentPipeline::NearestCt32Sprite)
        {
            executed = executor->executeResidentNearestCt32Sprites(
                nearestCt32Sprites, &executionError);
        }
        else if (pipeline == ResidentPipeline::LinearCt32Sprite)
        {
            executed = executor->executeResidentLinearCt32Sprites(
                linearCt32Sprites, &executionError);
        }
        else
        {
            executed = executor->executeResidentCt32Sprites(
                sprites, &executionError);
        }
        if (!executed)
        {
            clearPendingResidentCommands();
            failRequest(
                std::string(batchName) + " at " +
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
    const GsVulkanCapabilityReport executorCapabilities =
        impl->executor->capabilities();
    const GsVulkanDeviceReport *selectedDevice =
        executorCapabilities.selectedDevice();
    impl->exactCt32Triangle =
        selectedDevice && selectedDevice->exactCt32Triangle;
    impl->exactNearestCt32Sprite =
        selectedDevice && selectedDevice->exactNearestCt32Sprite;
    impl->exactLinearCt32Sprite =
        selectedDevice && selectedDevice->exactLinearCt32Sprite;
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
    if (decision.supported)
    {
        if (m_impl->config.mode != GsRendererMode::Hybrid)
            return decision;

        const uint64_t pixels =
            static_cast<uint64_t>(sprite.x1 - sprite.x0) *
            static_cast<uint64_t>(sprite.y1 - sprite.y0);
        if (pixels < m_impl->config.minimumHybridSpritePixels)
            return {false, GsFallbackReason::CostModel};
        return decision;
    }

    if (command.primitive().type == GS_PRIM_SPRITE &&
        command.primitive().tme &&
        (m_impl->config.mode == GsRendererMode::Hybrid ||
         m_impl->config.mode == GsRendererMode::Verify ||
         m_impl->config.mode == GsRendererMode::GpuStrict))
    {
        GsVulkanNearestCt32Sprite texturedSprite{};
        const GsBackendDecision textureDecision =
            prepareGsVulkanNearestCt32Sprite(
                command, texturedSprite);
        if (textureDecision.supported)
        {
            if (!m_impl->exactNearestCt32Sprite)
                return {false, GsFallbackReason::BackendUnavailable};

            if (m_impl->config.mode == GsRendererMode::Hybrid)
            {
                const uint64_t pixels =
                    static_cast<uint64_t>(
                        texturedSprite.boundsX1 - texturedSprite.boundsX0) *
                    static_cast<uint64_t>(
                        texturedSprite.boundsY1 - texturedSprite.boundsY0);
                if (pixels <
                    m_impl->config.minimumHybridNearestCt32SpritePixels)
                {
                    return {false, GsFallbackReason::CostModel};
                }
            }
            return textureDecision;
        }

        const uint64_t textureFilter = command.context().tex1;
        const bool requestsExactLinearFilter =
            ((textureFilter >> 2u) & 0x7u) == 0u &&
            ((textureFilter >> 5u) & 0x1u) == 1u &&
            ((textureFilter >> 6u) & 0x7u) == 1u;
        if (!requestsExactLinearFilter)
            return textureDecision;

        GsVulkanLinearCt32Sprite linearSprite{};
        const GsBackendDecision linearDecision =
            prepareGsVulkanLinearCt32Sprite(command, linearSprite);
        if (!linearDecision.supported)
            return linearDecision;
        const bool usesStandardClamp =
            gsVulkanTextureWrapMode(linearSprite.textureWrapU) != 0u ||
            gsVulkanTextureWrapMode(linearSprite.textureWrapV) != 0u;
        if (usesStandardClamp &&
            m_impl->config.mode != GsRendererMode::Verify)
        {
            return {false, GsFallbackReason::UnsupportedTextureWrap};
        }
        if (!m_impl->exactLinearCt32Sprite)
            return {false, GsFallbackReason::BackendUnavailable};
        if (m_impl->config.mode == GsRendererMode::Hybrid)
        {
            const uint32_t columns =
                linearSprite.boundsX1 - linearSprite.boundsX0;
            const uint64_t pixels =
                static_cast<uint64_t>(columns) *
                static_cast<uint64_t>(
                    linearSprite.boundsY1 - linearSprite.boundsY0);
            if (m_impl->config.minimumHybridLinearCt32SpritePixels != 0u &&
                pixels <
                    m_impl->config.minimumHybridLinearCt32SpritePixels)
            {
                return {false, GsFallbackReason::CostModel};
            }
        }
        return linearDecision;
    }

    if (command.primitive().type != GS_PRIM_TRIANGLE ||
        (m_impl->config.mode != GsRendererMode::Hybrid &&
         m_impl->config.mode != GsRendererMode::Verify &&
         m_impl->config.mode != GsRendererMode::GpuStrict))
    {
        return decision;
    }

    GsVulkanCt32Triangle triangle{};
    const GsBackendDecision triangleDecision =
        prepareGsVulkanCt32Triangle(command, triangle);
    if (!triangleDecision.supported)
        return triangleDecision;
    if (!m_impl->exactCt32Triangle)
        return {false, GsFallbackReason::BackendUnavailable};

    if (m_impl->config.mode == GsRendererMode::Hybrid)
    {
        const uint64_t candidatePixels =
            static_cast<uint64_t>(triangle.boundsX1 - triangle.boundsX0) *
            static_cast<uint64_t>(triangle.boundsY1 - triangle.boundsY0);
        if (candidatePixels <
            m_impl->config.minimumHybridTriangleCandidatePixels)
        {
            return {false, GsFallbackReason::CostModel};
        }
    }
    return triangleDecision;
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
        GsVulkanNearestCt32Sprite texturedSprite{};
        GsVulkanLinearCt32Sprite linearTexturedSprite{};
        GsVulkanCt32Triangle triangle{};
        const bool isTriangle =
            command.primitive().type == GS_PRIM_TRIANGLE;
        const bool isTexturedSprite =
            command.primitive().type == GS_PRIM_SPRITE &&
            command.primitive().tme;
        bool isLinearTexturedSprite = false;
        if (isTriangle)
            (void)prepareGsVulkanCt32Triangle(command, triangle);
        else if (isTexturedSprite)
        {
            const GsBackendDecision textureDecision =
                prepareGsVulkanNearestCt32Sprite(
                    command, texturedSprite);
            if (!textureDecision.supported)
            {
                (void)prepareGsVulkanLinearCt32Sprite(
                    command, linearTexturedSprite);
                isLinearTexturedSprite = true;
            }
        }
        else
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
            bool executed = false;
            if (isTriangle)
            {
                executed = m_impl->executor->executeCt32Triangle(
                    initial, triangle, gpuOutput, &executionError);
            }
            else if (isLinearTexturedSprite)
            {
                executed =
                    m_impl->executor->executeLinearCt32Sprite(
                        initial, linearTexturedSprite, gpuOutput,
                        &executionError);
            }
            else if (isTexturedSprite)
            {
                executed =
                    m_impl->executor->executeNearestCt32Sprite(
                        initial, texturedSprite, gpuOutput,
                        &executionError);
            }
            else
            {
                executed = m_impl->executor->executeCt32Sprite(
                    initial, sprite, gpuOutput, &executionError);
            }
            if (!executed ||
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

            // The verification request uploaded the entire initial image.
            // Record that fact before the independent CPU oracle creates its
            // newer result.
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
                bool artifactWritten = false;
                if (isTriangle)
                {
                    artifactWritten = writeVerificationArtifact(
                        m_impl->config.verificationArtifactDirectory,
                        command, triangle, firstDifference,
                        initial, m_impl->canonicalVram, gpuOutput,
                        artifactPath, artifactError);
                }
                else if (isLinearTexturedSprite)
                {
                    artifactWritten = writeVerificationArtifact(
                        m_impl->config.verificationArtifactDirectory,
                        command, linearTexturedSprite, firstDifference,
                        initial, m_impl->canonicalVram, gpuOutput,
                        artifactPath, artifactError);
                }
                else if (isTexturedSprite)
                {
                    artifactWritten = writeVerificationArtifact(
                        m_impl->config.verificationArtifactDirectory,
                        command, texturedSprite, firstDifference,
                        initial, m_impl->canonicalVram, gpuOutput,
                        artifactPath, artifactError);
                }
                else
                {
                    artifactWritten = writeVerificationArtifact(
                        m_impl->config.verificationArtifactDirectory,
                        command, sprite, firstDifference,
                        initial, m_impl->canonicalVram, gpuOutput,
                        artifactPath, artifactError);
                }
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
            Impl::ResidentPipeline pipeline =
                Impl::ResidentPipeline::Ct32Sprite;
            if (isTriangle)
                pipeline = Impl::ResidentPipeline::Ct32Triangle;
            else if (isLinearTexturedSprite)
                pipeline = Impl::ResidentPipeline::LinearCt32Sprite;
            else if (isTexturedSprite)
                pipeline = Impl::ResidentPipeline::NearestCt32Sprite;
            if (!m_impl->pendingResidentCommands.empty() &&
                m_impl->pendingResidentCommands.front().pipeline !=
                    pipeline)
            {
                m_impl->drainPendingResidentCommands(
                    GsFlushReason::PipelineChange);
            }
            const bool hasDependency =
                m_impl->pendingResidentWritePages.intersects(
                    resources.readPages) ||
                m_impl->pendingResidentWritePages.intersects(
                    resources.writePages) ||
                m_impl->pendingResidentReadPages.intersects(
                    resources.writePages);
            const bool ordersDependenciesInBatch =
                pipeline == Impl::ResidentPipeline::LinearCt32Sprite;
            if (hasDependency && !ordersDependenciesInBatch)
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
                {command, sprite, texturedSprite, linearTexturedSprite,
                 triangle,
                 resources, pipeline});
            m_impl->pendingResidentReadPages.unionWith(
                resources.readPages);
            m_impl->pendingResidentWritePages.unionWith(
                resources.writePages);
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
