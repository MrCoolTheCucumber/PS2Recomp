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
        const GsVulkanDepthCt32Sprite &sprite)
    {
        output
            << "  \"depth_ct32_sprite\": {"
            << "\"framebuffer_base_block\":"
            << sprite.framebufferBaseBlock << ','
            << "\"framebuffer_width\":"
            << sprite.framebufferWidth << ','
            << "\"depth_base_block\":"
            << sprite.depthBaseBlock << ','
            << "\"depth_psm\":" << sprite.depthPsm << ','
            << "\"bounds_x0\":" << sprite.boundsX0 << ','
            << "\"bounds_y0\":" << sprite.boundsY0 << ','
            << "\"bounds_x1\":" << sprite.boundsX1 << ','
            << "\"bounds_y1\":" << sprite.boundsY1 << ','
            << "\"rgba\":" << sprite.rgba << ','
            << "\"depth\":" << sprite.depth << ','
            << "\"depth_test_method\":"
            << sprite.depthTestMethod << ','
            << "\"depth_write\":" << sprite.depthWrite << ','
            << "\"color_blend_mode\":"
            << sprite.colorBlendMode << '}';
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
        const GsVulkanGouraudDepthCt32Triangle &triangle)
    {
        output
            << "  \"gouraud_depth_ct32_triangle\": {"
            << "\"framebuffer_base_block\":"
            << triangle.framebufferBaseBlock << ','
            << "\"framebuffer_width\":"
            << triangle.framebufferWidth << ','
            << "\"bounds_x0\":" << triangle.boundsX0 << ','
            << "\"bounds_y0\":" << triangle.boundsY0 << ','
            << "\"bounds_x1\":" << triangle.boundsX1 << ','
            << "\"bounds_y1\":" << triangle.boundsY1 << ','
            << "\"depth_base_block\":"
            << triangle.depthBaseBlock << ','
            << "\"depth_psm\":" << triangle.depthPsm << ','
            << "\"depth\":" << triangle.depth << ','
            << "\"positive_area\":" << triangle.positiveArea << ','
            << "\"vertex0_x_12_4\":" << triangle.vertex0X12_4 << ','
            << "\"vertex0_y_12_4\":" << triangle.vertex0Y12_4 << ','
            << "\"vertex1_x_12_4\":" << triangle.vertex1X12_4 << ','
            << "\"vertex1_y_12_4\":" << triangle.vertex1Y12_4 << ','
            << "\"vertex2_x_12_4\":" << triangle.vertex2X12_4 << ','
            << "\"vertex2_y_12_4\":" << triangle.vertex2Y12_4 << ','
            << "\"rgba0\":" << triangle.rgba0 << ','
            << "\"rgba1\":" << triangle.rgba1 << ','
            << "\"rgba2\":" << triangle.rgba2 << ','
            << "\"top_left_edge_mask\":"
            << triangle.topLeftEdgeMask
            << ",\"color_dx_bits\":[";
        for (size_t index = 0u;
             index < triangle.colorDxBits.size(); ++index)
        {
            if (index != 0u)
                output << ',';
            output << triangle.colorDxBits[index];
        }
        output << "],\"color_dy_bits\":[";
        for (size_t index = 0u;
             index < triangle.colorDyBits.size(); ++index)
        {
            if (index != 0u)
                output << ',';
            output << triangle.colorDyBits[index];
        }
        output << "]}";
    }

    void writePreparedRecord(
        std::ostream &output,
        const GsVulkanT8GouraudDepthCt32Triangle &triangle)
    {
        output
            << "  \"t8_gouraud_depth_ct32_triangle\": {"
            << "\"framebuffer_base_block\":"
            << triangle.framebufferBaseBlock << ','
            << "\"framebuffer_width\":"
            << triangle.framebufferWidth << ','
            << "\"bounds_x0\":" << triangle.boundsX0 << ','
            << "\"bounds_y0\":" << triangle.boundsY0 << ','
            << "\"bounds_x1\":" << triangle.boundsX1 << ','
            << "\"bounds_y1\":" << triangle.boundsY1 << ','
            << "\"depth_base_block\":"
            << triangle.depthBaseBlock << ','
            << "\"depth_psm\":" << triangle.depthPsm << ','
            << "\"positive_area\":" << triangle.positiveArea << ','
            << "\"top_left_edge_mask\":"
            << triangle.topLeftEdgeMask << ','
            << "\"raster_flags\":" << triangle.rasterFlags << ','
            << "\"maximum_mip_level\":"
            << triangle.maximumMipLevel << ','
            << "\"texture_wrap_u\":" << triangle.textureWrapU << ','
            << "\"texture_wrap_v\":" << triangle.textureWrapV << ','
            << "\"lod_k\":" << triangle.lodK << ','
            << "\"lod_l\":" << triangle.lodL << ','
            << "\"alpha_reference\":"
            << triangle.alphaReference << ",\"s_bits\":[";
        for (size_t index = 0u; index < triangle.sBits.size(); ++index)
        {
            if (index != 0u)
                output << ',';
            output << triangle.sBits[index];
        }
        output << "],\"t_bits\":[";
        for (size_t index = 0u; index < triangle.tBits.size(); ++index)
        {
            if (index != 0u)
                output << ',';
            output << triangle.tBits[index];
        }
        output << "],\"q_bits\":[";
        for (size_t index = 0u; index < triangle.qBits.size(); ++index)
        {
            if (index != 0u)
                output << ',';
            output << triangle.qBits[index];
        }
        output << "],\"color_dx_bits\":[";
        for (size_t index = 0u; index < triangle.colorDxBits.size(); ++index)
        {
            if (index != 0u)
                output << ',';
            output << triangle.colorDxBits[index];
        }
        output << "],\"color_dy_bits\":[";
        for (size_t index = 0u; index < triangle.colorDyBits.size(); ++index)
        {
            if (index != 0u)
                output << ',';
            output << triangle.colorDyBits[index];
        }
        output << "],\"texture_dx_bits\":[";
        for (size_t index = 0u; index < triangle.textureDxBits.size(); ++index)
        {
            if (index != 0u)
                output << ',';
            output << triangle.textureDxBits[index];
        }
        output << "],\"texture_dy_bits\":[";
        for (size_t index = 0u; index < triangle.textureDyBits.size(); ++index)
        {
            if (index != 0u)
                output << ',';
            output << triangle.textureDyBits[index];
        }
        output << "],\"palette\":[";
        for (size_t index = 0u; index < triangle.palette.size(); ++index)
        {
            if (index != 0u)
                output << ',';
            output << triangle.palette[index];
        }
        output << "]}";
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

    void writePreparedRecord(
        std::ostream &output,
        const GsVulkanFeedbackLinearDepthCt32Sprite &sprite)
    {
        output
            << "  \"feedback_linear_depth_ct32_sprite\": {"
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
            << "\"texture_wrap_v\":" << sprite.textureWrapV << ','
            << "\"depth_base_block\":" << sprite.depthBaseBlock << ','
            << "\"depth_psm\":" << sprite.depthPsm << ','
            << "\"depth\":" << sprite.depth << ','
            << "\"depth_test_method\":"
            << sprite.depthTestMethod << ','
            << "\"depth_write\":" << sprite.depthWrite << ','
            << "\"texture_source\":" << sprite.textureSource << '}';
    }

    void writePreparedRecord(
        std::ostream &output,
        const GsVulkanFeedbackNearestDepthCt32Triangle &triangle)
    {
        output
            << "  \"feedback_nearest_depth_ct32_triangle\": {"
            << "\"framebuffer_base_block\":"
            << triangle.framebufferBaseBlock << ','
            << "\"framebuffer_width\":"
            << triangle.framebufferWidth << ','
            << "\"bounds_x0\":" << triangle.boundsX0 << ','
            << "\"bounds_y0\":" << triangle.boundsY0 << ','
            << "\"bounds_x1\":" << triangle.boundsX1 << ','
            << "\"bounds_y1\":" << triangle.boundsY1 << ','
            << "\"depth_base_block\":"
            << triangle.depthBaseBlock << ','
            << "\"depth_psm\":" << triangle.depthPsm << ','
            << "\"rgba\":" << triangle.rgba << ','
            << "\"texture_base_block\":"
            << triangle.textureBaseBlock << ','
            << "\"texture_width\":" << triangle.textureWidth << ','
            << "\"texture_width_log2\":"
            << triangle.textureWidthLog2 << ','
            << "\"texture_height_log2\":"
            << triangle.textureHeightLog2 << ','
            << "\"texture_wrap_u\":"
            << triangle.textureWrapU << ','
            << "\"texture_wrap_v\":"
            << triangle.textureWrapV << ','
            << "\"texture_source\":"
            << triangle.textureSource << '}';
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
        std::span<const uint8_t> feedbackSnapshot,
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
            << std::dec << '"';
        if (!feedbackSnapshot.empty())
        {
            output
                << ",\n  \"feedback_snapshot_fnv1a64\": \"0x"
                << std::hex << std::setw(16) << std::setfill('0')
                << fnv1a64(feedbackSnapshot) << std::dec << '"';
        }
        output
            << ",\n  \"files\": {\"initial\": \"initial-vram.bin\", "
               "\"software\": \"software-vram.bin\", "
               "\"gpu\": \"gpu-vram.bin\"";
        if (!feedbackSnapshot.empty())
        {
            output
                << ", \"feedback_snapshot\": "
                   "\"feedback-snapshot-vram.bin\"";
        }
        output << "},\n";
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
        std::string &error,
        std::span<const uint8_t> feedbackSnapshot = {})
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
            (!feedbackSnapshot.empty() &&
             !writeBytes(
                 partial / "feedback-snapshot-vram.bin",
                 feedbackSnapshot, error)) ||
            !writeCommandManifest(
                partial / "command.json", command, record,
                firstDifference, software[firstDifference],
                gpu[firstDifference], initial, software, gpu,
                feedbackSnapshot, error))
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
        DepthCt32Sprite,
        NearestCt32Sprite,
        LinearCt32Sprite,
        FeedbackLinearDepthCt32Sprite,
        FeedbackNearestDepthCt32Triangle,
        Ct32Triangle,
        GouraudDepthCt32Triangle,
        T8GouraudDepthCt32Triangle,
    };

    std::unique_ptr<IGsVulkanDrawExecutor> executor;
    GsVulkanRasterBackendConfig config;
    std::span<uint8_t> canonicalVram;
    DrawCallback softwareOracle;
    DrawCallback acceleratedCommit;
    AcceleratedBatchCommitCallback acceleratedBatchCommit;
    FeedbackSnapshotCallback feedbackSnapshot;
    FeedbackSnapshotGenerationCallback feedbackSnapshotGeneration;
    DecodedPaletteCallback decodedPalette;
    GsVulkanT8Palette cachedDecodedPaletteColors{};
    GsVulkanDecodedPalette cachedDecodedPalette;
    bool cachedDecodedPaletteValid = false;
    GsVulkanRasterBackendStatistics statistics;
    GsVramCoherency coherency;
    ResidentPipeline pendingResidentPipeline = ResidentPipeline::Ct32Sprite;
    size_t pendingResidentCommandTotal = 0u;
    std::vector<GsVulkanCt32Sprite> pendingSprites;
    std::vector<GsVulkanDepthCt32Sprite> pendingDepthCt32Sprites;
    std::vector<GsVulkanNearestCt32Sprite> pendingNearestCt32Sprites;
    std::vector<GsVulkanLinearCt32Sprite> pendingLinearCt32Sprites;
    std::vector<GsVulkanFeedbackLinearDepthCt32Sprite>
        pendingFeedbackLinearDepthCt32Sprites;
    std::vector<GsVulkanFeedbackNearestDepthCt32Triangle>
        pendingFeedbackNearestDepthCt32Triangles;
    std::vector<GsVulkanCt32Triangle> pendingTriangles;
    std::vector<GsVulkanGouraudDepthCt32Triangle>
        pendingGouraudDepthTriangles;
    std::vector<GsVulkanResidentT8GouraudDepthCt32Triangle>
        pendingT8GouraudDepthTriangles;
    // The legacy per-draw callback is retained for diagnostics/tests, but
    // production uses the compact batch callback and never fills this vector.
    std::vector<GsDrawCommand> pendingCommitCommands;
    std::vector<GsVulkanT8Palette> pendingT8Palettes;
    uint64_t pendingT8PaletteGeneration = 0u;
    uint16_t pendingT8PaletteTexa = 0u;
    uint8_t pendingT8PaletteSourcePsm = 0u;
    uint8_t pendingT8PaletteCsm = 0u;
    uint8_t pendingT8PaletteCsa = 0u;
    bool pendingT8PaletteIdentityValid = false;
    GsVramPageMask pendingResidentReadPages;
    GsVramPageMask pendingResidentWritePages;
    GsVramPageMask pendingResidentTextureReadPages;
    uint64_t pendingResidentGpuWritePageTouches = 0u;
    GsVulkanAcceleratedCommitBatch pendingBatchCommit;
    std::vector<uint8_t> pendingFeedbackSnapshot;
    GsVulkanFeedbackSnapshotMode pendingFeedbackSnapshotMode =
        GsVulkanFeedbackSnapshotMode::UploadHost;
    uint64_t pendingFeedbackSnapshotGeneration = 0u;
    uint64_t residentFeedbackSnapshotGeneration = 0u;
    bool residentFeedbackSnapshotValid = false;
    const GsDrawCommand *classifiedT8Command = nullptr;
    uint64_t classifiedT8Sequence = 0u;
    GsDrawResources classifiedT8Resources;
    const GsDrawCommand *classifiedFeedbackNearestCommand = nullptr;
    uint64_t classifiedFeedbackNearestSequence = 0u;
    GsDrawResources classifiedFeedbackNearestResources;
    GsDrawResourceCache drawResourceCache;
    std::vector<GsDrawResources> submissionResourcesScratch;
    std::vector<uint8_t> submissionT8Scratch;
    std::vector<uint8_t> submissionFeedbackNearestScratch;
    bool exactCt32Triangle = false;
    bool exactGouraudDepthCt32Triangle = false;
    bool exactT8GouraudDepthCt32Triangle = false;
    bool exactDepthCt32Sprite = false;
    bool exactNearestCt32Sprite = false;
    bool exactLinearCt32Sprite = false;
    bool exactFeedbackLinearDepthCt32Sprite = false;
    bool exactFeedbackNearestDepthCt32Triangle = false;
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
        const size_t reasonIndex = static_cast<size_t>(reason);
        if (reasonIndex < GS_FLUSH_REASON_COUNT)
        {
            ++statistics.pageDownloadOperationsByReason[reasonIndex];
            statistics.pagesDownloadedByReason[reasonIndex] +=
                pages.count();
        }
        coherency.completeGpuToCpu(pages);
    }

    void clearPendingResidentCommands() noexcept
    {
        pendingResidentCommandTotal = 0u;
        pendingSprites.clear();
        pendingDepthCt32Sprites.clear();
        pendingNearestCt32Sprites.clear();
        pendingLinearCt32Sprites.clear();
        pendingFeedbackLinearDepthCt32Sprites.clear();
        pendingFeedbackNearestDepthCt32Triangles.clear();
        pendingTriangles.clear();
        pendingGouraudDepthTriangles.clear();
        pendingT8GouraudDepthTriangles.clear();
        pendingCommitCommands.clear();
        pendingResidentReadPages.clear();
        pendingResidentWritePages.clear();
        pendingResidentTextureReadPages.clear();
        pendingResidentGpuWritePageTouches = 0u;
        pendingBatchCommit = {};
        pendingFeedbackSnapshot.clear();
        pendingFeedbackSnapshotMode =
            GsVulkanFeedbackSnapshotMode::UploadHost;
        pendingFeedbackSnapshotGeneration = 0u;
        pendingT8Palettes.clear();
        pendingT8PaletteIdentityValid = false;
    }

    [[nodiscard]] size_t pendingRecordCount() const noexcept
    {
        switch (pendingResidentPipeline)
        {
        case ResidentPipeline::Ct32Sprite:
            return pendingSprites.size();
        case ResidentPipeline::DepthCt32Sprite:
            return pendingDepthCt32Sprites.size();
        case ResidentPipeline::NearestCt32Sprite:
            return pendingNearestCt32Sprites.size();
        case ResidentPipeline::LinearCt32Sprite:
            return pendingLinearCt32Sprites.size();
        case ResidentPipeline::FeedbackLinearDepthCt32Sprite:
            return pendingFeedbackLinearDepthCt32Sprites.size();
        case ResidentPipeline::FeedbackNearestDepthCt32Triangle:
            return pendingFeedbackNearestDepthCt32Triangles.size();
        case ResidentPipeline::Ct32Triangle:
            return pendingTriangles.size();
        case ResidentPipeline::GouraudDepthCt32Triangle:
            return pendingGouraudDepthTriangles.size();
        case ResidentPipeline::T8GouraudDepthCt32Triangle:
            return pendingT8GouraudDepthTriangles.size();
        }
        return 0u;
    }

    void drainPendingResidentCommands(GsFlushReason reason)
    {
        if (pendingResidentCommandTotal == 0u)
            return;

        const size_t commandCount = pendingResidentCommandTotal;
        if (reason == GsFlushReason::ResourceHazard)
            ++statistics.resourceHazardDrains;
        else if (reason == GsFlushReason::QueueBackpressure)
            ++statistics.queueBackpressureDrains;
        else if (reason == GsFlushReason::PipelineChange)
            ++statistics.pipelineChangeDrains;

        const ResidentPipeline pipeline = pendingResidentPipeline;
        if (pendingRecordCount() != commandCount ||
            (acceleratedCommit &&
             pendingCommitCommands.size() != commandCount))
        {
            clearPendingResidentCommands();
            failRequest(
                "resident CT32 batch assembly",
                "resident metadata disagrees with its homogeneous record queue");
        }

        const char *batchName = "resident CT32 sprite batch";
        if (pipeline == ResidentPipeline::DepthCt32Sprite)
            batchName = "resident depth CT32 sprite batch";
        else if (pipeline == ResidentPipeline::NearestCt32Sprite)
            batchName = "resident nearest CT32 sprite batch";
        else if (pipeline == ResidentPipeline::LinearCt32Sprite)
            batchName = "resident linear CT32 sprite batch";
        else if (pipeline ==
                 ResidentPipeline::FeedbackLinearDepthCt32Sprite)
        {
            batchName =
                "resident feedback linear depth CT32 sprite batch";
        }
        else if (pipeline ==
                 ResidentPipeline::FeedbackNearestDepthCt32Triangle)
        {
            batchName =
                "resident feedback nearest depth CT32 triangle batch";
        }
        else if (pipeline == ResidentPipeline::Ct32Triangle)
            batchName = "resident CT32 triangle batch";
        else if (pipeline == ResidentPipeline::GouraudDepthCt32Triangle)
            batchName = "resident Gouraud depth CT32 triangle batch";
        else if (pipeline ==
                 ResidentPipeline::T8GouraudDepthCt32Triangle)
        {
            batchName =
                "resident T8 Gouraud depth CT32 triangle batch";
        }
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
                pendingTriangles, &executionError);
        }
        else if (pipeline == ResidentPipeline::GouraudDepthCt32Triangle)
        {
            executed = executor->executeResidentGouraudDepthCt32Triangles(
                pendingGouraudDepthTriangles, &executionError);
        }
        else if (pipeline ==
                 ResidentPipeline::T8GouraudDepthCt32Triangle)
        {
            executed = executor
                ->executePreparedResidentT8GouraudDepthCt32Triangles(
                    std::move(pendingT8GouraudDepthTriangles),
                    std::move(pendingT8Palettes),
                    &executionError);
        }
        else if (pipeline == ResidentPipeline::DepthCt32Sprite)
        {
            executed = executor->executeResidentDepthCt32Sprites(
                pendingDepthCt32Sprites, &executionError);
        }
        else if (pipeline == ResidentPipeline::NearestCt32Sprite)
        {
            executed = executor->executeResidentNearestCt32Sprites(
                pendingNearestCt32Sprites, &executionError);
        }
        else if (pipeline == ResidentPipeline::LinearCt32Sprite)
        {
            executed = executor->executeResidentLinearCt32Sprites(
                pendingLinearCt32Sprites, &executionError);
        }
        else if (pipeline ==
                 ResidentPipeline::FeedbackLinearDepthCt32Sprite)
        {
            if (pendingFeedbackSnapshot.size() !=
                GS_VULKAN_VRAM_SIZE)
            {
                clearPendingResidentCommands();
                failRequest(
                    std::string(batchName) + " at " +
                        std::string(gsFlushReasonName(reason)),
                    "resident batch lost its exact 4 MiB snapshot");
            }
            executed = executor
                ->executeResidentFeedbackLinearDepthCt32Sprites(
                    pendingFeedbackSnapshot,
                    pendingFeedbackLinearDepthCt32Sprites,
                    &executionError);
        }
        else if (pipeline ==
                 ResidentPipeline::FeedbackNearestDepthCt32Triangle)
        {
            const bool needsHostSnapshot =
                pendingFeedbackSnapshotMode ==
                GsVulkanFeedbackSnapshotMode::UploadHost;
            if (needsHostSnapshot &&
                pendingFeedbackSnapshot.size() !=
                    GS_VULKAN_VRAM_SIZE)
            {
                clearPendingResidentCommands();
                failRequest(
                    std::string(batchName) + " at " +
                        std::string(gsFlushReasonName(reason)),
                    "resident batch lost its exact host snapshot");
            }
            executed = executor
                ->executeResidentFeedbackNearestDepthCt32Triangles(
                    pendingFeedbackSnapshot,
                    pendingFeedbackSnapshotMode,
                    pendingFeedbackNearestDepthCt32Triangles,
                    &executionError);
        }
        else
        {
            executed = executor->executeResidentCt32Sprites(
                pendingSprites, &executionError);
        }
        if (!executed)
        {
            clearPendingResidentCommands();
            failRequest(
                std::string(batchName) + " at " +
                    std::string(gsFlushReasonName(reason)),
                std::move(executionError));
        }

        if (pipeline ==
            ResidentPipeline::FeedbackNearestDepthCt32Triangle)
        {
            residentFeedbackSnapshotGeneration =
                pendingFeedbackSnapshotGeneration;
            residentFeedbackSnapshotValid = true;
        }
        else if (pipeline ==
                 ResidentPipeline::FeedbackLinearDepthCt32Sprite)
        {
            // This legacy path uploads through the shared immutable-snapshot
            // allocation without publishing a generation identity.
            residentFeedbackSnapshotValid = false;
        }

        coherency.noteGpuWriteBatch(
            pendingResidentWritePages,
            static_cast<uint64_t>(commandCount),
            pendingResidentGpuWritePageTouches);
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
                for (const GsDrawCommand &command : pendingCommitCommands)
                    acceleratedCommit(command);
            }
            if (acceleratedBatchCommit)
                acceleratedBatchCommit(pendingBatchCommit);
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
    std::string *error,
    FeedbackSnapshotCallback feedbackSnapshot,
    DecodedPaletteCallback decodedPalette,
    AcceleratedBatchCommitCallback acceleratedBatchCommit,
    FeedbackSnapshotGenerationCallback feedbackSnapshotGeneration)
{
    std::unique_ptr<GsVulkanService> service =
        GsVulkanService::create(serviceConfig, report, error);
    if (!service)
        return nullptr;
    return createWithExecutor(
        std::move(service), backendConfig, canonicalVram,
        std::move(softwareOracle), std::move(acceleratedCommit), error,
        std::move(feedbackSnapshot), std::move(decodedPalette),
        std::move(acceleratedBatchCommit),
        std::move(feedbackSnapshotGeneration));
}

std::unique_ptr<GsVulkanRasterBackend>
GsVulkanRasterBackend::createWithExecutor(
    std::unique_ptr<IGsVulkanDrawExecutor> executor,
    const GsVulkanRasterBackendConfig &backendConfig,
    std::span<uint8_t> canonicalVram,
    DrawCallback softwareOracle,
    DrawCallback acceleratedCommit,
    std::string *error,
    FeedbackSnapshotCallback feedbackSnapshot,
    DecodedPaletteCallback decodedPalette,
    AcceleratedBatchCommitCallback acceleratedBatchCommit,
    FeedbackSnapshotGenerationCallback feedbackSnapshotGeneration)
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
    impl->exactGouraudDepthCt32Triangle =
        selectedDevice &&
        selectedDevice->exactGouraudDepthCt32Triangle;
    impl->exactT8GouraudDepthCt32Triangle =
        selectedDevice &&
        selectedDevice->exactT8GouraudDepthCt32Triangle;
    impl->exactDepthCt32Sprite =
        selectedDevice && selectedDevice->exactDepthCt32Sprite;
    impl->exactNearestCt32Sprite =
        selectedDevice && selectedDevice->exactNearestCt32Sprite;
    impl->exactLinearCt32Sprite =
        selectedDevice && selectedDevice->exactLinearCt32Sprite;
    impl->exactFeedbackLinearDepthCt32Sprite =
        selectedDevice &&
        selectedDevice->exactFeedbackLinearDepthCt32Sprite;
    impl->exactFeedbackNearestDepthCt32Triangle =
        selectedDevice &&
        selectedDevice->exactFeedbackNearestDepthCt32Triangle;
    impl->config = backendConfig;
    impl->canonicalVram = canonicalVram;
    impl->softwareOracle = std::move(softwareOracle);
    impl->acceleratedCommit = std::move(acceleratedCommit);
    impl->acceleratedBatchCommit = std::move(acceleratedBatchCommit);
    impl->feedbackSnapshot = std::move(feedbackSnapshot);
    impl->feedbackSnapshotGeneration =
        std::move(feedbackSnapshotGeneration);
    impl->decodedPalette = std::move(decodedPalette);
    impl->pendingSprites.reserve(backendConfig.maximumResidentBatchCommands);
    impl->pendingDepthCt32Sprites.reserve(
        backendConfig.maximumResidentBatchCommands);
    impl->pendingNearestCt32Sprites.reserve(
        backendConfig.maximumResidentBatchCommands);
    impl->pendingLinearCt32Sprites.reserve(
        backendConfig.maximumResidentBatchCommands);
    impl->pendingFeedbackLinearDepthCt32Sprites.reserve(
        backendConfig.maximumResidentBatchCommands);
    impl->pendingFeedbackNearestDepthCt32Triangles.reserve(
        GS_VULKAN_MAX_RESIDENT_FEEDBACK_NEAREST_DEPTH_CT32_TRIANGLE_BATCH);
    impl->pendingTriangles.reserve(backendConfig.maximumResidentBatchCommands);
    impl->pendingGouraudDepthTriangles.reserve(
        GS_VULKAN_MAX_RESIDENT_GOURAUD_DEPTH_CT32_TRIANGLE_BATCH);
    impl->pendingT8GouraudDepthTriangles.reserve(
        GS_VULKAN_MAX_RESIDENT_T8_TRIANGLE_BATCH);
    if (impl->acceleratedCommit)
    {
        impl->pendingCommitCommands.reserve(
            std::max(
                backendConfig.maximumResidentBatchCommands,
                std::max(
                    GS_VULKAN_MAX_RESIDENT_GOURAUD_DEPTH_CT32_TRIANGLE_BATCH,
                    GS_VULKAN_MAX_RESIDENT_T8_TRIANGLE_BATCH)));
    }
    impl->pendingT8Palettes.reserve(16u);
    impl->submissionResourcesScratch.reserve(1u);
    impl->submissionT8Scratch.reserve(1u);
    impl->submissionFeedbackNearestScratch.reserve(1u);
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
    m_impl->classifiedT8Command = nullptr;
    m_impl->classifiedFeedbackNearestCommand = nullptr;
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
        const uint64_t minimumPixels = command.primitive().abe
            ? m_impl->config.minimumHybridSourceCopyAlphaSpritePixels
            : m_impl->config.minimumHybridSpritePixels;
        if (minimumPixels != 0u && pixels < minimumPixels)
            return {false, GsFallbackReason::CostModel};
        return decision;
    }

    // Depth is exact through full-image Verify and ordered resident execution.
    // Hybrid uses the conservative isolated-draw crossover measured across
    // Z32/Z24 ALWAYS and comparison states.
    if (command.primitive().type == GS_PRIM_SPRITE &&
        !command.primitive().tme &&
        (m_impl->config.mode == GsRendererMode::Hybrid ||
         m_impl->config.mode == GsRendererMode::Verify ||
         m_impl->config.mode == GsRendererMode::GpuStrict))
    {
        GsVulkanDepthCt32Sprite depthSprite{};
        bool framebufferOnlyAlphaFail = false;
        GsBackendDecision depthDecision =
            prepareGsVulkanDepthCt32Sprite(command, depthSprite);
        if (!depthDecision.supported &&
            depthDecision.reason == GsFallbackReason::AlphaBlend)
        {
            depthDecision =
                prepareGsVulkanSourceOverDepthCt32Sprite(
                    command, depthSprite);
        }
        if (!depthDecision.supported)
        {
            const GsBackendDecision framebufferOnlyDecision =
                prepareGsVulkanFramebufferOnlyAlphaFailDepthCt32Sprite(
                    command, depthSprite);
            if (framebufferOnlyDecision.supported)
            {
                depthDecision = framebufferOnlyDecision;
                framebufferOnlyAlphaFail = true;
            }
        }
        if (!depthDecision.supported)
            return depthDecision;
        if (!m_impl->exactDepthCt32Sprite)
            return {false, GsFallbackReason::BackendUnavailable};
        if (m_impl->config.mode == GsRendererMode::Hybrid &&
            !framebufferOnlyAlphaFail)
        {
            const uint64_t pixels =
                static_cast<uint64_t>(
                    depthSprite.boundsX1 - depthSprite.boundsX0) *
                static_cast<uint64_t>(
                    depthSprite.boundsY1 - depthSprite.boundsY0);
            if (pixels <
                m_impl->config.minimumHybridDepthCt32SpritePixels)
            {
                return {false, GsFallbackReason::CostModel};
            }
        }
        return depthDecision;
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

        const GsDrawResources resources = command.resources();
        if ((m_impl->config.mode == GsRendererMode::Hybrid ||
             m_impl->config.mode == GsRendererMode::Verify ||
             m_impl->config.mode == GsRendererMode::GpuStrict) &&
            resources.framebufferTextureAlias)
        {
            GsVulkanFeedbackLinearDepthCt32Sprite feedbackSprite{};
            const GsBackendDecision feedbackDecision =
                prepareGsVulkanFeedbackLinearDepthCt32Sprite(
                    command, feedbackSprite);
            if (!feedbackDecision.supported)
                return feedbackDecision;
            if (!m_impl->exactFeedbackLinearDepthCt32Sprite ||
                !m_impl->feedbackSnapshot)
            {
                return {false, GsFallbackReason::BackendUnavailable};
            }
            return feedbackDecision;
        }

        GsVulkanLinearCt32Sprite linearSprite{};
        const GsBackendDecision linearDecision =
            prepareGsVulkanLinearCt32Sprite(command, linearSprite);
        if (!linearDecision.supported)
            return linearDecision;
        const bool usesStandardClamp =
            gsVulkanTextureWrapMode(linearSprite.textureWrapU) != 0u ||
            gsVulkanTextureWrapMode(linearSprite.textureWrapV) != 0u;
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
            const uint64_t minimumPixels = usesStandardClamp
                ? m_impl->config.minimumHybridLinearCt32ClampSpritePixels
                : m_impl->config.minimumHybridLinearCt32SpritePixels;
            if (minimumPixels != 0u && pixels < minimumPixels)
            {
                return {false, GsFallbackReason::CostModel};
            }
        }
        return linearDecision;
    }

    const bool acceleratedMode =
        m_impl->config.mode == GsRendererMode::Hybrid ||
        m_impl->config.mode == GsRendererMode::Verify ||
        m_impl->config.mode == GsRendererMode::GpuStrict;
    const bool assembledTriangle =
        command.primitive().type == GS_PRIM_TRIANGLE ||
        command.primitive().type == GS_PRIM_TRISTRIP ||
        command.primitive().type == GS_PRIM_TRIFAN;
    if (assembledTriangle && acceleratedMode)
    {
        const GsBackendDecision feedbackNearestDecision =
            classifyGsFeedbackNearestDepthCt32Triangle(
                command,
                &m_impl->classifiedFeedbackNearestResources,
                &m_impl->drawResourceCache);
        if (feedbackNearestDecision.supported)
        {
            m_impl->classifiedFeedbackNearestCommand = &command;
            m_impl->classifiedFeedbackNearestSequence =
                command.sequence();
            if (!m_impl->exactFeedbackNearestDepthCt32Triangle ||
                (m_impl->config.mode == GsRendererMode::Verify &&
                 !m_impl->feedbackSnapshot))
            {
                return {false, GsFallbackReason::BackendUnavailable};
            }
            return feedbackNearestDecision;
        }

        const GsBackendDecision t8Decision =
            classifyGsT8GouraudDepthCt32Triangle(
                command,
                &m_impl->classifiedT8Resources,
                &m_impl->drawResourceCache);
        if (t8Decision.supported)
        {
            m_impl->classifiedT8Command = &command;
            m_impl->classifiedT8Sequence = command.sequence();
            if (!m_impl->exactT8GouraudDepthCt32Triangle ||
                (command.primitive().tme && !m_impl->decodedPalette))
            {
                return {false, GsFallbackReason::BackendUnavailable};
            }
            return t8Decision;
        }
        if (command.primitive().type != GS_PRIM_TRIANGLE)
            return t8Decision;
    }

    if (command.primitive().type != GS_PRIM_TRIANGLE ||
        !acceleratedMode)
    {
        return decision;
    }

    GsVulkanCt32Triangle triangle{};
    const GsBackendDecision triangleDecision =
        prepareGsVulkanCt32Triangle(command, triangle);
    if (triangleDecision.supported)
    {
        if (!m_impl->exactCt32Triangle)
            return {false, GsFallbackReason::BackendUnavailable};

        if (m_impl->config.mode == GsRendererMode::Hybrid)
        {
            const uint64_t candidatePixels =
                static_cast<uint64_t>(
                    triangle.boundsX1 - triangle.boundsX0) *
                static_cast<uint64_t>(
                    triangle.boundsY1 - triangle.boundsY0);
            if (candidatePixels <
                m_impl->config.minimumHybridTriangleCandidatePixels)
            {
                return {false, GsFallbackReason::CostModel};
            }
        }
        return triangleDecision;
    }

    if (triangleDecision.reason != GsFallbackReason::GouraudShading)
    {
        return triangleDecision;
    }

    GsVulkanGouraudDepthCt32Triangle gouraudDepthTriangle{};
    const GsBackendDecision gouraudDepthDecision =
        prepareGsVulkanGouraudSourceOverDepthCt32Triangle(
            command, gouraudDepthTriangle);
    if (!gouraudDepthDecision.supported)
        return triangleDecision;
    if (!m_impl->exactGouraudDepthCt32Triangle)
        return {false, GsFallbackReason::BackendUnavailable};
    return gouraudDepthDecision;
}

GsHybridBatchPolicy GsVulkanRasterBackend::hybridBatchPolicy(
    const GsDrawCommand &command) const noexcept
{
    if (m_impl->config.mode != GsRendererMode::Hybrid ||
        m_impl->shutDown || m_impl->failed)
    {
        return {};
    }

    if (m_impl->config
                .minimumHybridFramebufferOnlyAlphaFailDepthCt32RunPixels !=
            0u &&
        m_impl->exactDepthCt32Sprite)
    {
        GsVulkanDepthCt32Sprite framebufferOnlySprite{};
        if (prepareGsVulkanFramebufferOnlyAlphaFailDepthCt32Sprite(
                command, framebufferOnlySprite).supported)
        {
            return {
                m_impl->config
                    .minimumHybridFramebufferOnlyAlphaFailDepthCt32RunPixels,
                m_impl->config.maximumResidentBatchCommands};
        }
    }

    if (m_impl->config
                .minimumHybridGouraudDepthCt32TriangleRunPixels != 0u &&
        m_impl->exactGouraudDepthCt32Triangle)
    {
        GsVulkanGouraudDepthCt32Triangle triangle{};
        if (prepareGsVulkanGouraudSourceOverDepthCt32Triangle(
                command, triangle).supported)
        {
            return {
                m_impl->config
                    .minimumHybridGouraudDepthCt32TriangleRunPixels,
                GS_VULKAN_MAX_RESIDENT_GOURAUD_DEPTH_CT32_TRIANGLE_BATCH};
        }
    }

    if (m_impl->config.minimumHybridFeedbackLinearDepthCt32RunPixels == 0u ||
        !m_impl->exactFeedbackLinearDepthCt32Sprite ||
        !m_impl->feedbackSnapshot)
    {
        return {};
    }

    GsVulkanFeedbackLinearDepthCt32Sprite feedbackSprite{};
    if (!prepareGsVulkanFeedbackLinearDepthCt32Sprite(
             command, feedbackSprite).supported)
    {
        return {};
    }
    return {
        m_impl->config.minimumHybridFeedbackLinearDepthCt32RunPixels,
        m_impl->config.maximumResidentBatchCommands};
}

bool GsVulkanRasterBackend::hybridBatchCompatible(
    const GsDrawCommand &first,
    const GsDrawCommand &next) const noexcept
{
    if (m_impl->config.mode != GsRendererMode::Hybrid ||
        m_impl->shutDown || m_impl->failed)
    {
        return false;
    }

    if (m_impl->exactDepthCt32Sprite)
    {
        GsVulkanDepthCt32Sprite firstFramebufferOnly{};
        GsVulkanDepthCt32Sprite nextFramebufferOnly{};
        if (prepareGsVulkanFramebufferOnlyAlphaFailDepthCt32Sprite(
                first, firstFramebufferOnly).supported &&
            prepareGsVulkanFramebufferOnlyAlphaFailDepthCt32Sprite(
                next, nextFramebufferOnly).supported)
        {
            return true;
        }
    }

    if (m_impl->exactGouraudDepthCt32Triangle)
    {
        GsVulkanGouraudDepthCt32Triangle firstTriangle{};
        GsVulkanGouraudDepthCt32Triangle nextTriangle{};
        if (prepareGsVulkanGouraudSourceOverDepthCt32Triangle(
                first, firstTriangle).supported &&
            prepareGsVulkanGouraudSourceOverDepthCt32Triangle(
                next, nextTriangle).supported)
        {
            return firstTriangle.framebufferBaseBlock ==
                       nextTriangle.framebufferBaseBlock &&
                   firstTriangle.framebufferWidth ==
                       nextTriangle.framebufferWidth &&
                   firstTriangle.depthBaseBlock ==
                       nextTriangle.depthBaseBlock &&
                   firstTriangle.depthPsm == nextTriangle.depthPsm;
        }
    }

    if (!m_impl->exactFeedbackLinearDepthCt32Sprite ||
        !m_impl->feedbackSnapshot)
    {
        return false;
    }

    GsVulkanFeedbackLinearDepthCt32Sprite firstSprite{};
    GsVulkanFeedbackLinearDepthCt32Sprite nextSprite{};
    if (!prepareGsVulkanFeedbackLinearDepthCt32Sprite(
             first, firstSprite).supported ||
        !prepareGsVulkanFeedbackLinearDepthCt32Sprite(
             next, nextSprite).supported)
    {
        return false;
    }

    // These are exactly the surface fields used by the raster frontend to
    // retain or invalidate its immutable feedback snapshot. Depth state,
    // coordinates, and sampling DDA may vary within the same snapshot run.
    return firstSprite.framebufferBaseBlock ==
               nextSprite.framebufferBaseBlock &&
           firstSprite.framebufferWidth == nextSprite.framebufferWidth &&
           firstSprite.textureBaseBlock == nextSprite.textureBaseBlock &&
           firstSprite.textureWidth == nextSprite.textureWidth;
}

bool GsVulkanRasterBackend::canCaptureFeedbackSnapshotOnDevice(
    const GsDrawCommand &command) const noexcept
{
    if ((m_impl->config.mode != GsRendererMode::Hybrid &&
         m_impl->config.mode != GsRendererMode::GpuStrict) ||
        m_impl->shutDown || m_impl->failed ||
        !m_impl->exactFeedbackNearestDepthCt32Triangle)
    {
        return false;
    }
    return classifyGsFeedbackNearestDepthCt32Triangle(command).supported;
}

void GsVulkanRasterBackend::submit(
    std::span<const GsDrawCommand> commands)
{
    // Preserve the router contract for callers that submit a future batch:
    // one unsupported member rejects the whole span before canonical VRAM is
    // changed by an earlier member.
    std::vector<GsDrawResources> &commandResources =
        m_impl->submissionResourcesScratch;
    std::vector<uint8_t> &commandIsT8 =
        m_impl->submissionT8Scratch;
    std::vector<uint8_t> &commandIsFeedbackNearest =
        m_impl->submissionFeedbackNearestScratch;
    commandResources.clear();
    commandIsT8.clear();
    commandIsFeedbackNearest.clear();
    const bool singleton = commands.size() == 1u;
    GsDrawResources singletonResources;
    bool singletonIsT8 = false;
    bool singletonIsFeedbackNearest = false;
    if (!singleton)
    {
        commandResources.resize(commands.size());
        commandIsT8.assign(commands.size(), 0u);
        commandIsFeedbackNearest.assign(commands.size(), 0u);
    }
    for (size_t index = 0u; index < commands.size(); ++index)
    {
        const GsDrawCommand &command = commands[index];
        const bool cachedT8 =
            m_impl->classifiedT8Command == &command &&
            m_impl->classifiedT8Sequence == command.sequence();
        const bool cachedFeedbackNearest =
            m_impl->classifiedFeedbackNearestCommand == &command &&
            m_impl->classifiedFeedbackNearestSequence ==
                command.sequence();
        const GsBackendDecision decision =
            (cachedT8 || cachedFeedbackNearest)
            ? GsBackendDecision{true, GsFallbackReason::Supported}
            : classify(command);
        if (!decision.supported)
        {
            throw std::logic_error(
                "Vulkan raster backend received unsupported draw " +
                std::to_string(command.sequence()) + ": " +
                std::string(gsFallbackReasonName(decision.reason)));
        }
        if (m_impl->classifiedFeedbackNearestCommand == &command &&
            m_impl->classifiedFeedbackNearestSequence ==
                command.sequence())
        {
            if (singleton)
            {
                singletonIsFeedbackNearest = true;
            }
            else
            {
                commandResources[index] =
                    m_impl->classifiedFeedbackNearestResources;
                commandIsFeedbackNearest[index] = 1u;
            }
        }
        else if (m_impl->classifiedT8Command == &command &&
            m_impl->classifiedT8Sequence == command.sequence())
        {
            if (singleton)
            {
                singletonIsT8 = true;
            }
            else
            {
                commandResources[index] = m_impl->classifiedT8Resources;
                commandIsT8[index] = 1u;
            }
        }
        else if (singleton)
            command.describeResources(singletonResources);
        else
            commandResources[index] = command.resources();
    }
    m_impl->classifiedT8Command = nullptr;
    m_impl->classifiedFeedbackNearestCommand = nullptr;

    for (size_t commandIndex = 0u;
         commandIndex < commands.size(); ++commandIndex)
    {
        const GsDrawCommand &command = commands[commandIndex];
        GsVulkanCt32Sprite sprite{};
        GsVulkanDepthCt32Sprite depthSprite{};
        GsVulkanNearestCt32Sprite texturedSprite{};
        GsVulkanLinearCt32Sprite linearTexturedSprite{};
        GsVulkanFeedbackLinearDepthCt32Sprite
            feedbackLinearDepthSprite{};
        GsVulkanFeedbackNearestDepthCt32Triangle
            feedbackNearestDepthTriangle{};
        GsVulkanCt32Triangle triangle{};
        GsVulkanGouraudDepthCt32Triangle
            gouraudDepthTriangle{};
        GsVulkanResidentT8GouraudDepthCt32Triangle
            t8GouraudDepthTriangle{};
        GsVulkanT8GouraudDepthCt32Triangle
            verifiedT8GouraudDepthTriangle{};
        GsVulkanDecodedPalette t8Palette;
        const GsDrawResources &resources = singleton
            ? (singletonIsFeedbackNearest
                   ? m_impl->classifiedFeedbackNearestResources
               : singletonIsT8
                   ? m_impl->classifiedT8Resources
                   : singletonResources)
            : commandResources[commandIndex];
        const bool isTriangle =
            command.primitive().type == GS_PRIM_TRIANGLE;
        const bool isAssembledTriangle =
            isTriangle ||
            command.primitive().type == GS_PRIM_TRISTRIP ||
            command.primitive().type == GS_PRIM_TRIFAN;
        const bool isTexturedSprite =
            command.primitive().type == GS_PRIM_SPRITE &&
            command.primitive().tme;
        bool isDepthSprite = false;
        bool isLinearTexturedSprite = false;
        bool isFeedbackLinearDepthSprite = false;
        bool isFeedbackNearestDepthTriangle = false;
        bool isGouraudDepthTriangle = false;
        bool isT8GouraudDepthTriangle = false;
        if (isAssembledTriangle &&
            (singleton
                 ? singletonIsFeedbackNearest
                 : commandIsFeedbackNearest[commandIndex] != 0u))
        {
            const GsBackendDecision prepared =
                prepareGsVulkanFeedbackNearestDepthCt32Triangle(
                    command,
                    feedbackNearestDepthTriangle,
                    &resources);
            if (!prepared.supported)
            {
                m_impl->failRequest(
                    "feedback nearest triangle preparation for draw " +
                        std::to_string(command.sequence()),
                    "frontend did not provide a valid recursive setup record");
            }
            isFeedbackNearestDepthTriangle = true;
        }
        else if (isAssembledTriangle &&
                 (singleton
                      ? singletonIsT8
                      : commandIsT8[commandIndex] != 0u))
        {
            const bool textured = command.primitive().tme;
            if (textured)
            {
                const GSTex0Reg &tex0 = command.context().tex0;
                const GSTexaReg &texa = command.globalState().texa;
                const uint16_t packedTexa =
                    static_cast<uint16_t>(texa.ta0) |
                    (static_cast<uint16_t>(texa.aem) << 8u) |
                    (static_cast<uint16_t>(texa.ta1) << 9u);
                const bool samePaletteInputs =
                    m_impl->cachedDecodedPaletteValid &&
                    m_impl->cachedDecodedPalette.texa == packedTexa &&
                    m_impl->cachedDecodedPalette.sourcePsm == tex0.psm &&
                    m_impl->cachedDecodedPalette.csm == tex0.csm &&
                    m_impl->cachedDecodedPalette.csa == tex0.csa;
                if (!samePaletteInputs)
                {
                    const GsVulkanDecodedPalette decoded =
                        m_impl->decodedPalette();
                    if (decoded.colors.size() != 256u)
                    {
                        m_impl->failRequest(
                            "T8 Gouraud triangle palette capture for draw " +
                                std::to_string(command.sequence()),
                            "frontend did not provide an exact 256-entry decoded palette");
                    }
                    std::copy(
                        decoded.colors.begin(),
                        decoded.colors.end(),
                        m_impl->cachedDecodedPaletteColors.begin());
                    m_impl->cachedDecodedPalette = {
                        std::span<const uint32_t>(
                            m_impl->cachedDecodedPaletteColors),
                        decoded.generation,
                        decoded.texa,
                        decoded.sourcePsm,
                        decoded.csm,
                        decoded.csa};
                    m_impl->cachedDecodedPaletteValid = true;
                }
                t8Palette = m_impl->cachedDecodedPalette;
            }
            const GsBackendDecision prepared =
                prepareGsVulkanResidentT8GouraudDepthCt32Triangle(
                    command, 0u, t8GouraudDepthTriangle,
                    &resources);
            if (!prepared.supported)
            {
                m_impl->failRequest(
                    "T8 Gouraud triangle preparation for draw " +
                        std::to_string(command.sequence()),
                    "frontend did not provide a valid decoded palette or setup record");
            }
            if (m_impl->config.mode == GsRendererMode::Verify &&
                !prepareGsVulkanT8GouraudDepthCt32Triangle(
                    command, t8Palette.colors,
                    verifiedT8GouraudDepthTriangle,
                    &resources).supported)
            {
                m_impl->failRequest(
                    "T8 Gouraud verification preparation for draw " +
                        std::to_string(command.sequence()),
                    "frontend did not provide a valid decoded palette or setup record");
            }
            isT8GouraudDepthTriangle = true;
        }
        else if (isTriangle)
        {
            const GsBackendDecision triangleDecision =
                prepareGsVulkanCt32Triangle(command, triangle);
            if (!triangleDecision.supported)
            {
                isGouraudDepthTriangle =
                    prepareGsVulkanGouraudSourceOverDepthCt32Triangle(
                        command, gouraudDepthTriangle).supported;
            }
        }
        else if (isTexturedSprite)
        {
            const GsBackendDecision textureDecision =
                prepareGsVulkanNearestCt32Sprite(
                    command, texturedSprite);
            if (!textureDecision.supported)
            {
                if ((m_impl->config.mode == GsRendererMode::Hybrid ||
                     m_impl->config.mode == GsRendererMode::Verify ||
                     m_impl->config.mode == GsRendererMode::GpuStrict) &&
                    resources.framebufferTextureAlias)
                {
                    const GsBackendDecision feedbackDecision =
                        prepareGsVulkanFeedbackLinearDepthCt32Sprite(
                            command, feedbackLinearDepthSprite);
                    isFeedbackLinearDepthSprite =
                        feedbackDecision.supported;
                }
                if (!isFeedbackLinearDepthSprite)
                {
                    (void)prepareGsVulkanLinearCt32Sprite(
                        command, linearTexturedSprite);
                    isLinearTexturedSprite = true;
                }
            }
        }
        else
        {
            const GsBackendDecision spriteDecision =
                prepareGsVulkanCt32Sprite(command, sprite);
            if (!spriteDecision.supported)
            {
                GsBackendDecision depthDecision =
                    prepareGsVulkanDepthCt32Sprite(
                        command, depthSprite);
                if (!depthDecision.supported &&
                    depthDecision.reason ==
                        GsFallbackReason::AlphaBlend)
                {
                    depthDecision =
                        prepareGsVulkanSourceOverDepthCt32Sprite(
                            command, depthSprite);
                }
                if (!depthDecision.supported)
                {
                    const GsBackendDecision framebufferOnlyDecision =
                        prepareGsVulkanFramebufferOnlyAlphaFailDepthCt32Sprite(
                            command, depthSprite);
                    if (framebufferOnlyDecision.supported)
                        depthDecision = framebufferOnlyDecision;
                }
                isDepthSprite = depthDecision.supported;
            }
        }
        ++m_impl->statistics.commandsAttempted;
        if (m_impl->config.mode == GsRendererMode::Verify)
        {
            std::span<const uint8_t> feedbackSnapshot;
            if (isFeedbackLinearDepthSprite ||
                isFeedbackNearestDepthTriangle)
            {
                feedbackSnapshot = m_impl->feedbackSnapshot();
                if (feedbackSnapshot.size() != GS_VULKAN_VRAM_SIZE)
                {
                    m_impl->failRequest(
                        "feedback snapshot for verification draw " +
                            std::to_string(command.sequence()),
                        "frontend did not provide an exact 4 MiB snapshot");
                }
            }
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
            if (isFeedbackNearestDepthTriangle)
            {
                executed = m_impl->executor
                    ->executeFeedbackNearestDepthCt32Triangle(
                        initial, feedbackSnapshot,
                        feedbackNearestDepthTriangle,
                        gpuOutput, &executionError);
            }
            else if (isT8GouraudDepthTriangle)
            {
                executed = m_impl->executor
                    ->executeT8GouraudDepthCt32Triangle(
                        initial, verifiedT8GouraudDepthTriangle,
                        gpuOutput, &executionError);
            }
            else if (isGouraudDepthTriangle)
            {
                executed = m_impl->executor
                    ->executeGouraudDepthCt32Triangle(
                        initial, gouraudDepthTriangle,
                        gpuOutput, &executionError);
            }
            else if (isTriangle)
            {
                executed = m_impl->executor->executeCt32Triangle(
                    initial, triangle, gpuOutput, &executionError);
            }
            else if (isFeedbackLinearDepthSprite)
            {
                executed = m_impl->executor
                    ->executeFeedbackLinearDepthCt32Sprite(
                        initial, feedbackSnapshot,
                        feedbackLinearDepthSprite, gpuOutput,
                        &executionError);
            }
            else if (isDepthSprite)
            {
                executed = m_impl->executor->executeDepthCt32Sprite(
                    initial, depthSprite, gpuOutput, &executionError);
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
                if (isFeedbackNearestDepthTriangle)
                {
                    artifactWritten = writeVerificationArtifact(
                        m_impl->config.verificationArtifactDirectory,
                        command, feedbackNearestDepthTriangle,
                        firstDifference, initial,
                        m_impl->canonicalVram, gpuOutput,
                        artifactPath, artifactError,
                        feedbackSnapshot);
                }
                else if (isT8GouraudDepthTriangle)
                {
                    artifactWritten = writeVerificationArtifact(
                        m_impl->config.verificationArtifactDirectory,
                        command, verifiedT8GouraudDepthTriangle,
                        firstDifference, initial,
                        m_impl->canonicalVram, gpuOutput,
                        artifactPath, artifactError);
                }
                else if (isGouraudDepthTriangle)
                {
                    artifactWritten = writeVerificationArtifact(
                        m_impl->config.verificationArtifactDirectory,
                        command, gouraudDepthTriangle,
                        firstDifference, initial,
                        m_impl->canonicalVram, gpuOutput,
                        artifactPath, artifactError);
                }
                else if (isTriangle)
                {
                    artifactWritten = writeVerificationArtifact(
                        m_impl->config.verificationArtifactDirectory,
                        command, triangle, firstDifference,
                        initial, m_impl->canonicalVram, gpuOutput,
                        artifactPath, artifactError);
                }
                else if (isFeedbackLinearDepthSprite)
                {
                    artifactWritten = writeVerificationArtifact(
                        m_impl->config.verificationArtifactDirectory,
                        command, feedbackLinearDepthSprite,
                        firstDifference, initial,
                        m_impl->canonicalVram, gpuOutput,
                        artifactPath, artifactError,
                        feedbackSnapshot);
                }
                else if (isDepthSprite)
                {
                    artifactWritten = writeVerificationArtifact(
                        m_impl->config.verificationArtifactDirectory,
                        command, depthSprite, firstDifference,
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
            if (isFeedbackNearestDepthTriangle)
            {
                pipeline = Impl::ResidentPipeline::
                    FeedbackNearestDepthCt32Triangle;
            }
            else if (isT8GouraudDepthTriangle)
            {
                pipeline = Impl::ResidentPipeline::
                    T8GouraudDepthCt32Triangle;
            }
            else if (isGouraudDepthTriangle)
            {
                pipeline = Impl::ResidentPipeline::
                    GouraudDepthCt32Triangle;
            }
            else if (isTriangle)
                pipeline = Impl::ResidentPipeline::Ct32Triangle;
            else if (isDepthSprite)
                pipeline = Impl::ResidentPipeline::DepthCt32Sprite;
            else if (isFeedbackLinearDepthSprite)
            {
                pipeline = Impl::ResidentPipeline::
                    FeedbackLinearDepthCt32Sprite;
            }
            else if (isLinearTexturedSprite)
                pipeline = Impl::ResidentPipeline::LinearCt32Sprite;
            else if (isTexturedSprite)
                pipeline = Impl::ResidentPipeline::NearestCt32Sprite;

            GsVramPageMask residentReadPages = resources.readPages;
            if (isFeedbackLinearDepthSprite ||
                isFeedbackNearestDepthTriangle)
            {
                // Texture, mip, and CLUT reads consume the independently
                // copied snapshot rather than canonical resident VRAM.
                residentReadPages = resources.framebufferReadPages;
                residentReadPages.unionWith(
                    resources.depthReadPages);
            }
            else if (isT8GouraudDepthTriangle)
            {
                // The shader consumes the submission-time decoded palette
                // captured in its immutable batch table. CLUT source VRAM is
                // therefore not a resident device dependency; retaining it
                // here would make a later architectural CLUT load drain
                // unrelated queued draws.
                residentReadPages = resources.framebufferReadPages;
                residentReadPages.unionWith(
                    resources.depthReadPages);
                residentReadPages.unionWith(
                    resources.texturePages);
                residentReadPages.unionWith(
                    resources.mipPages);
            }

            std::vector<uint8_t> nextFeedbackSnapshot;
            const bool opensFeedbackBatch =
                isFeedbackLinearDepthSprite &&
                (m_impl->pendingResidentCommandTotal == 0u ||
                 m_impl->pendingResidentPipeline != pipeline ||
                 m_impl->pendingResidentCommandTotal >=
                     m_impl->config.maximumResidentBatchCommands);
            if (opensFeedbackBatch)
            {
                const std::span<const uint8_t> snapshot =
                    m_impl->feedbackSnapshot();
                if (snapshot.size() != GS_VULKAN_VRAM_SIZE)
                {
                    m_impl->failRequest(
                        "feedback snapshot for strict draw " +
                            std::to_string(command.sequence()),
                        "frontend did not provide an exact 4 MiB snapshot");
                }
                if (m_impl->pendingResidentCommandTotal == 0u)
                {
                    m_impl->pendingFeedbackSnapshot.assign(
                        snapshot.begin(), snapshot.end());
                }
                else
                {
                    nextFeedbackSnapshot.assign(
                        snapshot.begin(), snapshot.end());
                }
            }
            if (m_impl->pendingResidentCommandTotal != 0u &&
                m_impl->pendingResidentPipeline != pipeline)
            {
                m_impl->drainPendingResidentCommands(
                    GsFlushReason::PipelineChange);
            }
            if (pipeline == Impl::ResidentPipeline::
                    FeedbackNearestDepthCt32Triangle &&
                m_impl->pendingResidentCommandTotal != 0u)
            {
                const auto &first =
                    m_impl->pendingFeedbackNearestDepthCt32Triangles.front();
                const bool sameRenderTarget =
                    first.framebufferBaseBlock ==
                        feedbackNearestDepthTriangle.framebufferBaseBlock &&
                    first.framebufferWidth ==
                        feedbackNearestDepthTriangle.framebufferWidth &&
                    first.depthBaseBlock ==
                        feedbackNearestDepthTriangle.depthBaseBlock &&
                    first.depthPsm ==
                        feedbackNearestDepthTriangle.depthPsm &&
                    first.textureBaseBlock ==
                        feedbackNearestDepthTriangle.textureBaseBlock &&
                    first.textureWidth ==
                        feedbackNearestDepthTriangle.textureWidth;
                const GsVulkanFeedbackSnapshotIdentity identity =
                    m_impl->feedbackSnapshotGeneration
                        ? m_impl->feedbackSnapshotGeneration()
                        : GsVulkanFeedbackSnapshotIdentity{};
                if (!sameRenderTarget ||
                    identity.generation !=
                        m_impl->pendingFeedbackSnapshotGeneration)
                {
                    m_impl->drainPendingResidentCommands(
                        GsFlushReason::PipelineChange);
                    m_impl->residentFeedbackSnapshotValid = false;
                }
            }
            if (pipeline == Impl::ResidentPipeline::
                    T8GouraudDepthCt32Triangle &&
                m_impl->pendingResidentCommandTotal != 0u)
            {
                const auto &first =
                    m_impl->pendingT8GouraudDepthTriangles.front();
                const bool sameRenderTarget =
                    first.framebufferBaseBlock ==
                        t8GouraudDepthTriangle.framebufferBaseBlock &&
                    first.framebufferWidth ==
                        t8GouraudDepthTriangle.framebufferWidth &&
                    first.depthBaseBlock ==
                        t8GouraudDepthTriangle.depthBaseBlock &&
                    first.depthPsm ==
                        t8GouraudDepthTriangle.depthPsm;
                GsVramPageMask textureReadPages =
                    resources.texturePages;
                textureReadPages.unionWith(
                    resources.mipPages);
                const bool textureDependency =
                    m_impl->pendingResidentWritePages.intersects(
                        textureReadPages) ||
                    m_impl->pendingResidentTextureReadPages.intersects(
                        resources.writePages);
                if (!sameRenderTarget)
                {
                    m_impl->drainPendingResidentCommands(
                        GsFlushReason::PipelineChange);
                }
                else if (textureDependency)
                {
                    // Ordered tiles serialize destination color/depth for one
                    // render target. A texture dependency can address an
                    // arbitrary pixel, so it still requires a global batch
                    // boundary before either side mutates the sampled pages.
                    m_impl->drainPendingResidentCommands(
                        GsFlushReason::ResourceHazard);
                }
            }
            else if (pipeline == Impl::ResidentPipeline::
                         GouraudDepthCt32Triangle &&
                     m_impl->pendingResidentCommandTotal != 0u)
            {
                const auto &first =
                    m_impl->pendingGouraudDepthTriangles.front();
                const bool sameRenderTarget =
                    first.framebufferBaseBlock ==
                        gouraudDepthTriangle.framebufferBaseBlock &&
                    first.framebufferWidth ==
                        gouraudDepthTriangle.framebufferWidth &&
                    first.depthBaseBlock ==
                        gouraudDepthTriangle.depthBaseBlock &&
                    first.depthPsm == gouraudDepthTriangle.depthPsm;
                if (!sameRenderTarget)
                {
                    m_impl->drainPendingResidentCommands(
                        GsFlushReason::PipelineChange);
                }
            }
            const bool hasDependency =
                m_impl->pendingResidentWritePages.intersects(
                    residentReadPages) ||
                m_impl->pendingResidentWritePages.intersects(
                    resources.writePages) ||
                m_impl->pendingResidentReadPages.intersects(
                    resources.writePages);
            const bool ordersDependenciesInBatch =
                pipeline == Impl::ResidentPipeline::LinearCt32Sprite ||
                pipeline == Impl::ResidentPipeline::DepthCt32Sprite ||
                pipeline == Impl::ResidentPipeline::
                    FeedbackLinearDepthCt32Sprite ||
                pipeline == Impl::ResidentPipeline::
                    FeedbackNearestDepthCt32Triangle ||
                pipeline == Impl::ResidentPipeline::
                    GouraudDepthCt32Triangle ||
                pipeline == Impl::ResidentPipeline::
                    T8GouraudDepthCt32Triangle;
            if (hasDependency && !ordersDependenciesInBatch)
            {
                m_impl->drainPendingResidentCommands(
                    GsFlushReason::ResourceHazard);
            }
            const size_t maximumBatchCommands =
                pipeline == Impl::ResidentPipeline::
                                FeedbackNearestDepthCt32Triangle
                    ? GS_VULKAN_MAX_RESIDENT_FEEDBACK_NEAREST_DEPTH_CT32_TRIANGLE_BATCH
                : pipeline == Impl::ResidentPipeline::
                                T8GouraudDepthCt32Triangle
                    ? GS_VULKAN_MAX_RESIDENT_T8_TRIANGLE_BATCH
                : pipeline == Impl::ResidentPipeline::
                                GouraudDepthCt32Triangle
                    ? GS_VULKAN_MAX_RESIDENT_GOURAUD_DEPTH_CT32_TRIANGLE_BATCH
                    : m_impl->config.maximumResidentBatchCommands;
            if (m_impl->pendingResidentCommandTotal >=
                maximumBatchCommands)
            {
                m_impl->drainPendingResidentCommands(
                    GsFlushReason::QueueBackpressure);
            }

            if (isT8GouraudDepthTriangle)
            {
                if (!command.primitive().tme)
                {
                    if (m_impl->pendingT8Palettes.empty())
                        m_impl->pendingT8Palettes.emplace_back();
                    t8GouraudDepthTriangle.paletteIndex = 0u;
                }
                else
                {
                    const bool repeatsPalette =
                        m_impl->pendingT8PaletteIdentityValid &&
                        t8Palette.generation ==
                            m_impl->pendingT8PaletteGeneration &&
                        t8Palette.texa == m_impl->pendingT8PaletteTexa &&
                        t8Palette.sourcePsm ==
                            m_impl->pendingT8PaletteSourcePsm &&
                        t8Palette.csm == m_impl->pendingT8PaletteCsm &&
                        t8Palette.csa == m_impl->pendingT8PaletteCsa;
                    if (!repeatsPalette)
                    {
                        GsVulkanT8Palette captured{};
                        std::copy(
                            t8Palette.colors.begin(),
                            t8Palette.colors.end(),
                            captured.begin());
                        m_impl->pendingT8Palettes.push_back(
                            std::move(captured));
                        m_impl->pendingT8PaletteGeneration =
                            t8Palette.generation;
                        m_impl->pendingT8PaletteTexa = t8Palette.texa;
                        m_impl->pendingT8PaletteSourcePsm =
                            t8Palette.sourcePsm;
                        m_impl->pendingT8PaletteCsm = t8Palette.csm;
                        m_impl->pendingT8PaletteCsa = t8Palette.csa;
                        m_impl->pendingT8PaletteIdentityValid = true;
                    }
                    t8GouraudDepthTriangle.paletteIndex =
                        static_cast<uint32_t>(
                            m_impl->pendingT8Palettes.size() - 1u);
                }
            }

            if (isFeedbackLinearDepthSprite &&
                m_impl->pendingFeedbackSnapshot.size() !=
                    GS_VULKAN_VRAM_SIZE)
            {
                if (nextFeedbackSnapshot.size() == GS_VULKAN_VRAM_SIZE)
                {
                    m_impl->pendingFeedbackSnapshot =
                        std::move(nextFeedbackSnapshot);
                }
                else
                {
                    m_impl->failRequest(
                        "resident feedback batch assembly",
                        "pending batch has no exact 4 MiB snapshot");
                }
            }

            if (isFeedbackNearestDepthTriangle &&
                m_impl->pendingResidentCommandTotal == 0u)
            {
                const GsVulkanFeedbackSnapshotIdentity identity =
                    m_impl->feedbackSnapshotGeneration
                        ? m_impl->feedbackSnapshotGeneration()
                        : GsVulkanFeedbackSnapshotIdentity{};
                m_impl->pendingFeedbackSnapshotGeneration =
                    identity.generation;
                if (identity.deviceResident)
                {
                    const bool canReuse =
                        m_impl->residentFeedbackSnapshotValid &&
                        m_impl->residentFeedbackSnapshotGeneration ==
                            identity.generation;
                    m_impl->pendingFeedbackSnapshotMode = canReuse
                        ? GsVulkanFeedbackSnapshotMode::ReuseResident
                        : GsVulkanFeedbackSnapshotMode::CaptureResident;
                    m_impl->pendingFeedbackSnapshot.clear();
                }
                else
                {
                    if (!m_impl->feedbackSnapshot)
                    {
                        m_impl->failRequest(
                            "resident feedback nearest batch assembly",
                            "frontend did not provide a host snapshot callback");
                    }
                    const std::span<const uint8_t> snapshot =
                        m_impl->feedbackSnapshot();
                    if (snapshot.size() != GS_VULKAN_VRAM_SIZE)
                    {
                        m_impl->failRequest(
                            "resident feedback nearest batch assembly",
                            "frontend did not provide an exact 4 MiB snapshot");
                    }
                    m_impl->pendingFeedbackSnapshotMode =
                        GsVulkanFeedbackSnapshotMode::UploadHost;
                    m_impl->pendingFeedbackSnapshot.assign(
                        snapshot.begin(), snapshot.end());
                }
            }

            if (m_impl->pendingResidentCommandTotal == 0u)
                m_impl->pendingResidentPipeline = pipeline;
            switch (pipeline)
            {
            case Impl::ResidentPipeline::Ct32Sprite:
                m_impl->pendingSprites.push_back(sprite);
                break;
            case Impl::ResidentPipeline::DepthCt32Sprite:
                m_impl->pendingDepthCt32Sprites.push_back(depthSprite);
                break;
            case Impl::ResidentPipeline::NearestCt32Sprite:
                m_impl->pendingNearestCt32Sprites.push_back(texturedSprite);
                break;
            case Impl::ResidentPipeline::LinearCt32Sprite:
                m_impl->pendingLinearCt32Sprites.push_back(
                    linearTexturedSprite);
                break;
            case Impl::ResidentPipeline::FeedbackLinearDepthCt32Sprite:
                m_impl->pendingFeedbackLinearDepthCt32Sprites.push_back(
                    feedbackLinearDepthSprite);
                break;
            case Impl::ResidentPipeline::FeedbackNearestDepthCt32Triangle:
                m_impl->pendingFeedbackNearestDepthCt32Triangles.push_back(
                    feedbackNearestDepthTriangle);
                break;
            case Impl::ResidentPipeline::Ct32Triangle:
                m_impl->pendingTriangles.push_back(triangle);
                break;
            case Impl::ResidentPipeline::GouraudDepthCt32Triangle:
                m_impl->pendingGouraudDepthTriangles.push_back(
                    gouraudDepthTriangle);
                break;
            case Impl::ResidentPipeline::T8GouraudDepthCt32Triangle:
                m_impl->pendingT8GouraudDepthTriangles.push_back(
                    t8GouraudDepthTriangle);
                break;
            }
            ++m_impl->pendingResidentCommandTotal;
            if (m_impl->acceleratedCommit)
                m_impl->pendingCommitCommands.push_back(command);
            if (m_impl->acceleratedBatchCommit)
            {
                ++m_impl->pendingBatchCommit.commandCount;
                const GsDrawBounds &bounds = command.bounds();
                m_impl->pendingBatchCommit.candidatePixels +=
                    static_cast<uint64_t>(bounds.x1 - bounds.x0) *
                    static_cast<uint64_t>(bounds.y1 - bounds.y0);
                m_impl->pendingBatchCommit.framebufferBasePages.set(
                    command.context().frame.fbp);
            }
            if (pipeline == Impl::ResidentPipeline::
                    FeedbackNearestDepthCt32Triangle &&
                m_impl->pendingFeedbackSnapshotMode ==
                    GsVulkanFeedbackSnapshotMode::CaptureResident)
            {
                // CaptureResident copies the complete immutable 4 MiB image,
                // and that identity can outlive the currently sampled pages.
                // GPU-newer pages are already correct in resident VRAM; make
                // sure every CPU-newer page is uploaded before the copy too.
                residentReadPages.setAll();
            }
            m_impl->pendingResidentReadPages.unionWith(
                residentReadPages);
            m_impl->pendingResidentWritePages.unionWith(
                resources.writePages);
            m_impl->pendingResidentGpuWritePageTouches +=
                resources.writePages.count();
            if (pipeline == Impl::ResidentPipeline::
                    T8GouraudDepthCt32Triangle)
            {
                m_impl->pendingResidentTextureReadPages.unionWith(
                    resources.texturePages);
                m_impl->pendingResidentTextureReadPages.unionWith(
                    resources.mipPages);
            }
        }
        if (m_impl->config.mode == GsRendererMode::Verify)
            ++m_impl->statistics.commandsCompleted;
    }
}

void GsVulkanRasterBackend::flush(GsFlushReason reason)
{
    ++m_impl->statistics.flushes;
    if (reason == GsFlushReason::Reset ||
        reason == GsFlushReason::SaveLoad)
    {
        m_impl->cachedDecodedPaletteValid = false;
    }
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
    return m_impl->pendingResidentCommandTotal;
}

void GsVulkanRasterBackend::prepareCpuVramAccess(
    const GsVramPageMask &pages,
    GsFlushReason reason)
{
    prepareCpuVramAccess(pages, pages, reason);
}

void GsVulkanRasterBackend::prepareCpuVramAccess(
    const GsVramPageMask &readPages,
    const GsVramPageMask &writePages,
    GsFlushReason reason)
{
    if (!readPages.any() && !writePages.any())
        return;
    // CLUT loads enter through this boundary as read-only CPU accesses, while
    // transfers and restore paths can alter their source. The cached span is
    // still owned by the frontend; force its identity/decode callback before
    // the next indexed draw after any external VRAM observation.
    m_impl->cachedDecodedPaletteValid = false;
    const bool conflicts =
        m_impl->pendingResidentWritePages.intersects(readPages) ||
        m_impl->pendingResidentReadPages.intersects(writePages) ||
        m_impl->pendingResidentWritePages.intersects(writePages);
    if (conflicts)
        m_impl->drainPendingResidentCommands(reason);
    ++m_impl->statistics.cpuAccessPreparations;
    GsVramPageMask pages = readPages;
    pages.unionWith(writePages);
    m_impl->downloadGpuNewer(pages, reason);
}

void GsVulkanRasterBackend::noteCpuVramWrite(
    const GsVramPageMask &pages)
{
    m_impl->cachedDecodedPaletteValid = false;
    m_impl->coherency.noteCpuWrite(pages);
}

bool GsVulkanRasterBackend::materializeFeedbackSnapshot(
    std::span<uint8_t> destination,
    std::string *error)
{
    if (destination.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
            *error = "feedback snapshot materialization requires exactly 4 MiB";
        return false;
    }
    if (m_impl->shutDown || m_impl->failed ||
        !m_impl->executor->healthy())
    {
        if (error)
            *error = "Vulkan GS backend is not healthy";
        return false;
    }

    m_impl->drainPendingResidentCommands(GsFlushReason::SaveLoad);
    if (!m_impl->residentFeedbackSnapshotValid)
    {
        if (error)
            *error = "no immutable resident feedback snapshot is available";
        return false;
    }
    return m_impl->executor->downloadFeedbackSnapshot(
        destination, error);
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
