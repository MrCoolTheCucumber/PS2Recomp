#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_gs_command_stream.h"
#include "runtime/ps2_gs_vulkan.h"
#include "runtime/ps2_memory.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
    bool readFile(const std::string &path, std::vector<uint8_t> &data)
    {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input)
            return false;

        const std::streampos end = input.tellg();
        if (end < 0)
            return false;

        data.resize(static_cast<size_t>(end));
        input.seekg(0, std::ios::beg);
        if (!data.empty())
            input.read(reinterpret_cast<char *>(data.data()),
                       static_cast<std::streamsize>(data.size()));
        return input.good() ||
               static_cast<size_t>(input.gcount()) == data.size();
    }

    bool writeFile(const std::string &path,
                   const uint8_t *data,
                   size_t size)
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output)
            return false;

        output.write(reinterpret_cast<const char *>(data),
                     static_cast<std::streamsize>(size));
        return output.good();
    }

    void printUsage()
    {
        std::cerr
            << "usage: gs_replay --vulkan-info [--vulkan-validation] "
               "[--vulkan-loader FILE] [--vulkan-vendor ID] "
               "[--vulkan-device ID]\n"
            << "usage: gs_replay --vulkan-roundtrip COUNT "
               "[--vram-in FILE] [--vram-out FILE] "
               "[--vulkan-validation] [--vulkan-loader FILE] "
               "[--vulkan-vendor ID] [--vulkan-device ID]\n"
            << "usage: gs_replay [--vram-in FILE] [--vram-out FILE] "
               "[--register ADDRESS=VALUE] [--register-file FILE] "
               "[--state-in FILE] "
               "[--packet-sizes FILE] [--hash-trace FILE] "
               "[--renderer software|hybrid|verify|gpu-strict] "
               "[--gs-execution inline|threaded-sync|threaded-async] "
               "[--verify-dump-dir DIRECTORY] "
               "[--vulkan-max-resident-batch COUNT] "
               "[--vulkan-min-hybrid-pixels COUNT] "
               "[--vulkan-min-hybrid-source-copy-alpha-pixels COUNT] "
               "[--vulkan-min-hybrid-depth-ct32-pixels COUNT] "
               "[--vulkan-min-hybrid-alpha-fail-depth-run-pixels COUNT] "
               "[--vulkan-min-hybrid-nearest-ct32-pixels COUNT] "
               "[--vulkan-min-hybrid-linear-ct32-pixels COUNT] "
               "[--vulkan-min-hybrid-linear-ct32-clamp-pixels COUNT] "
               "[--vulkan-min-hybrid-feedback-ct32-run-pixels COUNT] "
               "[--vulkan-min-hybrid-triangle-pixels COUNT] "
               "[--vulkan-min-hybrid-gouraud-depth-run-pixels COUNT] "
               "[--vulkan-validation] [--vulkan-loader FILE] "
               "[--vulkan-vendor ID] [--vulkan-device ID] "
               "[--backend-stats] [--stop-after-command COUNT] "
               "[--stop-after-packet COUNT] [--compare-vram FILE] "
               "[--batch-stream | --batch-drains] "
               "GIF_PACKET [GIF_PACKET ...]\n";
    }

    bool parseUnsigned(const std::string &text, uint64_t &value)
    {
        errno = 0;
        char *end = nullptr;
        const unsigned long long parsed =
            std::strtoull(text.c_str(), &end, 0);
        if (errno != 0 || end == text.c_str() || !end || *end != '\0')
            return false;
        value = static_cast<uint64_t>(parsed);
        return true;
    }

    bool parseCount(const std::string &text, uint64_t &value)
    {
        return !text.empty() && text[0] != '-' &&
               parseUnsigned(text, value);
    }

    bool parseRendererMode(
        const std::string &text,
        GsRendererMode &mode)
    {
        if (text == "software")
            mode = GsRendererMode::Software;
        else if (text == "hybrid")
            mode = GsRendererMode::Hybrid;
        else if (text == "verify")
            mode = GsRendererMode::Verify;
        else if (text == "gpu-strict")
            mode = GsRendererMode::GpuStrict;
        else
            return false;
        return true;
    }

    bool parseGsExecutionMode(
        const std::string &text,
        GsExecutionMode &mode)
    {
        if (text == "inline")
            mode = GsExecutionMode::Inline;
        else if (text == "threaded-sync")
            mode = GsExecutionMode::ThreadedSynchronous;
        else if (text == "threaded-async")
            mode = GsExecutionMode::ThreadedAsync;
        else
            return false;
        return true;
    }

    struct VramDifference
    {
        bool matches = true;
        size_t byteOffset = 0u;
        size_t page = 0u;
        uint8_t expected = 0u;
        uint8_t actual = 0u;
    };

    VramDifference compareVram(
        const uint8_t *actual,
        const std::vector<uint8_t> &expected)
    {
        for (size_t offset = 0u; offset < expected.size(); ++offset)
        {
            if (actual[offset] == expected[offset])
                continue;
            return {
                false,
                offset,
                offset / GS_VRAM_PAGE_SIZE,
                expected[offset],
                actual[offset],
            };
        }
        return {};
    }

    void writeBackendCounters(
        std::ostream &output,
        const GsBackendCounters &counters)
    {
        output << "{\"commands\":" << counters.commands
               << ",\"noop_commands\":" << counters.noopCommands
               << ",\"software_commands\":" << counters.softwareCommands
               << ",\"accelerated_commands\":" << counters.acceleratedCommands
               << ",\"verified_commands\":" << counters.verifiedCommands
               << ",\"fallback_commands\":" << counters.fallbackCommands
               << ",\"strict_failures\":" << counters.strictFailures
               << ",\"flushes\":" << counters.flushes
               << ",\"backend_switches\":" << counters.backendSwitches
               << ",\"queue_depth\":" << counters.queueDepth
               << ",\"queue_high_watermark\":"
               << counters.queueHighWatermark
               << ",\"draw_pixels\":" << counters.drawPixels
               << ",\"software_pixels\":" << counters.softwarePixels
               << ",\"accelerated_pixels\":" << counters.acceleratedPixels
               << ",\"verified_pixels\":" << counters.verifiedPixels
               << ",\"fallback_pixels\":" << counters.fallbackPixels
               << ",\"strict_failure_pixels\":"
               << counters.strictFailurePixels
               << ",\"software_raster_host_nanoseconds\":"
               << counters.softwareRasterHostNanoseconds
               << ",\"decisions\":{";

        for (size_t index = 0u;
             index < GS_FALLBACK_REASON_COUNT;
             ++index)
        {
            if (index != 0u)
                output << ',';
            const auto reason = static_cast<GsFallbackReason>(index);
            output << '\"' << gsFallbackReasonName(reason) << "\":"
                   << counters.decisions[index];
        }

        output << "},\"decision_pixels\":{";
        for (size_t index = 0u;
             index < GS_FALLBACK_REASON_COUNT;
             ++index)
        {
            if (index != 0u)
                output << ',';
            const auto reason = static_cast<GsFallbackReason>(index);
            output << '\"' << gsFallbackReasonName(reason) << "\":"
                   << counters.decisionPixels[index];
        }

        output << "},\"flush_reasons\":{";
        for (size_t index = 0u;
             index < GS_FLUSH_REASON_COUNT;
             ++index)
        {
            if (index != 0u)
                output << ',';
            const auto reason = static_cast<GsFlushReason>(index);
            output << '\"' << gsFlushReasonName(reason) << "\":"
                   << counters.flushReasons[index];
        }
        output << "}}";
    }

    void writeJsonString(std::ostream &output, std::string_view value)
    {
        static constexpr char hex[] = "0123456789abcdef";
        output << '"';
        for (const unsigned char ch : value)
        {
            switch (ch)
            {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (ch < 0x20u)
                {
                    output << "\\u00" << hex[ch >> 4u]
                           << hex[ch & 0xFu];
                }
                else
                {
                    output << static_cast<char>(ch);
                }
                break;
            }
        }
        output << '"';
    }

    void writeVulkanCapabilityFields(
        std::ostream &output,
        const GsVulkanCapabilityReport &report)
    {
        output << "\"status\":";
        writeJsonString(output, gsVulkanProbeStatusName(report.status));
        output << ",\"compiled\":"
               << (report.compiled ? "true" : "false")
               << ",\"loader_available\":"
               << (report.loaderAvailable ? "true" : "false")
               << ",\"loader_api_version\":"
               << report.loaderApiVersion
               << ",\"loader_api_version_text\":";
        writeJsonString(
            output, gsVulkanVersionString(report.loaderApiVersion));
        output << ",\"validation_requested\":"
               << (report.validationRequested ? "true" : "false")
               << ",\"validation_layer_available\":"
               << (report.validationLayerAvailable ? "true" : "false")
               << ",\"debug_utils_available\":"
               << (report.debugUtilsAvailable ? "true" : "false")
               << ",\"validation_enabled\":"
               << (report.validationEnabled ? "true" : "false")
               << ",\"validation_warnings\":"
               << report.validationWarnings
               << ",\"validation_errors\":"
               << report.validationErrors
               << ",\"selected_device_index\":"
               << report.selectedDeviceIndex
               << ",\"devices\":[";

        for (size_t index = 0u; index < report.devices.size(); ++index)
        {
            if (index != 0u)
                output << ',';
            const GsVulkanDeviceReport &device = report.devices[index];
            output << "{\"name\":";
            writeJsonString(output, device.name);
            output << ",\"vendor_id\":" << device.vendorId
                   << ",\"device_id\":" << device.deviceId
                   << ",\"api_version\":" << device.apiVersion
                   << ",\"api_version_text\":";
            writeJsonString(
                output, gsVulkanVersionString(device.apiVersion));
            output << ",\"driver_version\":" << device.driverVersion
                   << ",\"kind\":";
            writeJsonString(output, gsVulkanDeviceKindName(device.kind));
            output << ",\"max_storage_buffer_range\":"
                   << device.maxStorageBufferRange
                   << ",\"max_compute_work_group_count_x\":"
                   << device.maxComputeWorkGroupCountX
                   << ",\"max_compute_work_group_invocations\":"
                   << device.maxComputeWorkGroupInvocations
                   << ",\"max_compute_work_group_size_x\":"
                   << device.maxComputeWorkGroupSizeX
                   << ",\"queue_family_index\":"
                   << device.queueFamilyIndex
                   << ",\"compute_queue\":"
                   << (device.computeQueue ? "true" : "false")
                   << ",\"dedicated_compute_queue\":"
                   << (device.dedicatedComputeQueue ? "true" : "false")
                   << ",\"device_local_memory\":"
                   << (device.deviceLocalMemory ? "true" : "false")
                   << ",\"host_visible_memory\":"
                   << (device.hostVisibleMemory ? "true" : "false")
                   << ",\"shader_int16\":"
                   << (device.shaderInt16 ? "true" : "false")
                   << ",\"shader_int64\":"
                   << (device.shaderInt64 ? "true" : "false")
                   << ",\"shader_float64\":"
                   << (device.shaderFloat64 ? "true" : "false")
                   << ",\"exact_vram_storage\":"
                   << (device.exactVramStorage ? "true" : "false")
                   << ",\"exact_ct32_triangle\":"
                   << (device.exactCt32Triangle ? "true" : "false")
                   << ",\"exact_gouraud_depth_ct32_triangle\":"
                   << (device.exactGouraudDepthCt32Triangle
                           ? "true"
                           : "false")
                   << ",\"exact_t8_gouraud_depth_ct32_triangle\":"
                   << (device.exactT8GouraudDepthCt32Triangle
                           ? "true"
                           : "false")
                   << ",\"exact_depth_ct32_sprite\":"
                   << (device.exactDepthCt32Sprite ? "true" : "false")
                   << ",\"exact_nearest_ct32_sprite\":"
                   << (device.exactNearestCt32Sprite ? "true" : "false")
                   << ",\"exact_linear_ct32_sprite\":"
                   << (device.exactLinearCt32Sprite ? "true" : "false")
                   << ",\"exact_feedback_linear_depth_ct32_sprite\":"
                   << (device.exactFeedbackLinearDepthCt32Sprite
                           ? "true"
                           : "false")
                   << ",\"exact_feedback_nearest_depth_ct32_triangle\":"
                   << (device.exactFeedbackNearestDepthCt32Triangle
                           ? "true"
                           : "false")
                   << ",\"suitable\":"
                   << (device.suitable ? "true" : "false")
                   << ",\"rejection_reason\":";
            writeJsonString(output, device.rejectionReason);
            output << '}';
        }

        output << "],\"detail\":";
        writeJsonString(output, report.detail);
    }

    void writeVulkanCapabilityReport(
        std::ostream &output,
        const GsVulkanCapabilityReport &report)
    {
        output << "{\"schema_version\":1,\"mode\":\"vulkan-info\",";
        writeVulkanCapabilityFields(output, report);
        output << "}\n";
    }

    void writeVulkanServiceStatistics(
        std::ostream &output,
        const GsVulkanServiceStatistics &statistics)
    {
        output << "{\"round_trips_completed\":"
               << statistics.roundTripsCompleted
               << ",\"round_trips_failed\":"
               << statistics.roundTripsFailed
               << ",\"request_waits\":"
               << statistics.requestWaits
               << ",\"request_wait_nanoseconds\":"
               << statistics.requestWaitNanoseconds
               << ",\"maximum_request_wait_nanoseconds\":"
               << statistics.maximumRequestWaitNanoseconds
               << ",\"requests_by_kind\":{";
        for (size_t index = 0u;
             index < statistics.requestsByKind.size(); ++index)
        {
            if (index != 0u)
                output << ',';
            writeJsonString(
                output,
                gsVulkanRequestKindName(
                    static_cast<GsVulkanRequestKind>(index)));
            const GsVulkanRequestStatistics &request =
                statistics.requestsByKind[index];
            output << ":{\"request_waits\":"
                   << request.requestWaits
                   << ",\"request_wait_nanoseconds\":"
                   << request.requestWaitNanoseconds
                   << ",\"fence_waits\":"
                   << request.fenceWaits
                   << ",\"fence_wait_nanoseconds\":"
                   << request.fenceWaitNanoseconds
                   << '}';
        }
        output << "},\"queue_submissions\":"
               << statistics.queueSubmissions
               << ",\"shader_dispatches\":"
               << statistics.shaderDispatches
               << ",\"pipeline_barriers\":"
               << statistics.pipelineBarriers
               << ",\"pipeline_binds\":"
               << statistics.pipelineBinds
               << ",\"pipeline_cache_hits\":"
               << statistics.pipelineCacheHits
               << ",\"pipeline_cache_misses\":"
               << statistics.pipelineCacheMisses
               << ",\"bytes_uploaded\":"
               << statistics.bytesUploaded
               << ",\"bytes_downloaded\":"
               << statistics.bytesDownloaded
               << ",\"fence_waits\":"
               << statistics.fenceWaits
               << ",\"fence_wait_nanoseconds\":"
               << statistics.fenceWaitNanoseconds
               << ",\"fence_timeouts\":"
               << statistics.fenceTimeouts
               << ",\"memory_batches_completed\":"
               << statistics.memoryBatchesCompleted
               << ",\"memory_batches_failed\":"
               << statistics.memoryBatchesFailed
               << ",\"memory_cases_executed\":"
               << statistics.memoryCasesExecuted
               << ",\"sprite_draws_completed\":"
               << statistics.spriteDrawsCompleted
               << ",\"sprite_draws_failed\":"
               << statistics.spriteDrawsFailed
               << ",\"sprite_pixels_executed\":"
               << statistics.spritePixelsExecuted
               << ",\"depth_ct32_sprite_draws_completed\":"
               << statistics.depthCt32SpriteDrawsCompleted
               << ",\"depth_ct32_sprite_draws_failed\":"
               << statistics.depthCt32SpriteDrawsFailed
               << ",\"depth_ct32_sprite_pixels_executed\":"
               << statistics.depthCt32SpritePixelsExecuted
               << ",\"resident_depth_ct32_sprite_batches_completed\":"
               << statistics.residentDepthCt32SpriteBatchesCompleted
               << ",\"resident_depth_ct32_sprite_batches_failed\":"
               << statistics.residentDepthCt32SpriteBatchesFailed
               << ",\"largest_resident_depth_ct32_sprite_batch\":"
               << statistics.largestResidentDepthCt32SpriteBatch
               << ",\"nearest_ct32_sprite_draws_completed\":"
               << statistics.nearestCt32SpriteDrawsCompleted
               << ",\"nearest_ct32_sprite_draws_failed\":"
               << statistics.nearestCt32SpriteDrawsFailed
               << ",\"nearest_ct32_sprite_pixels_executed\":"
               << statistics.nearestCt32SpritePixelsExecuted
               << ",\"resident_nearest_ct32_sprite_batches_completed\":"
               << statistics.residentNearestCt32SpriteBatchesCompleted
               << ",\"resident_nearest_ct32_sprite_batches_failed\":"
               << statistics.residentNearestCt32SpriteBatchesFailed
               << ",\"largest_resident_nearest_ct32_sprite_batch\":"
               << statistics.largestResidentNearestCt32SpriteBatch
               << ",\"linear_ct32_sprite_draws_completed\":"
               << statistics.linearCt32SpriteDrawsCompleted
               << ",\"linear_ct32_sprite_draws_failed\":"
               << statistics.linearCt32SpriteDrawsFailed
               << ",\"linear_ct32_sprite_pixels_executed\":"
               << statistics.linearCt32SpritePixelsExecuted
               << ",\"feedback_linear_depth_ct32_sprite_draws_completed\":"
               << statistics.feedbackLinearDepthCt32SpriteDrawsCompleted
               << ",\"feedback_linear_depth_ct32_sprite_draws_failed\":"
               << statistics.feedbackLinearDepthCt32SpriteDrawsFailed
               << ",\"feedback_linear_depth_ct32_sprite_pixels_executed\":"
               << statistics.feedbackLinearDepthCt32SpritePixelsExecuted
               << ",\"resident_feedback_linear_depth_ct32_sprite_batches_completed\":"
               << statistics
                      .residentFeedbackLinearDepthCt32SpriteBatchesCompleted
               << ",\"resident_feedback_linear_depth_ct32_sprite_batches_failed\":"
               << statistics
                      .residentFeedbackLinearDepthCt32SpriteBatchesFailed
               << ",\"largest_resident_feedback_linear_depth_ct32_sprite_batch\":"
               << statistics
                      .largestResidentFeedbackLinearDepthCt32SpriteBatch
               << ",\"feedback_nearest_depth_ct32_triangle_draws_completed\":"
               << statistics
                      .feedbackNearestDepthCt32TriangleDrawsCompleted
               << ",\"feedback_nearest_depth_ct32_triangle_draws_failed\":"
               << statistics
                      .feedbackNearestDepthCt32TriangleDrawsFailed
               << ",\"feedback_nearest_depth_ct32_triangle_candidate_pixels_executed\":"
               << statistics
                      .feedbackNearestDepthCt32TriangleCandidatePixelsExecuted
               << ",\"resident_feedback_nearest_depth_ct32_triangle_batches_completed\":"
               << statistics
                      .residentFeedbackNearestDepthCt32TriangleBatchesCompleted
               << ",\"resident_feedback_nearest_depth_ct32_triangle_batches_failed\":"
               << statistics
                      .residentFeedbackNearestDepthCt32TriangleBatchesFailed
               << ",\"largest_resident_feedback_nearest_depth_ct32_triangle_batch\":"
               << statistics
                      .largestResidentFeedbackNearestDepthCt32TriangleBatch
               << ",\"resident_feedback_snapshots_captured\":"
               << statistics.residentFeedbackSnapshotsCaptured
               << ",\"resident_feedback_snapshots_reused\":"
               << statistics.residentFeedbackSnapshotsReused
               << ",\"resident_feedback_snapshots_downloaded\":"
               << statistics.residentFeedbackSnapshotsDownloaded
               << ",\"resident_feedback_snapshot_downloads_failed\":"
               << statistics.residentFeedbackSnapshotDownloadsFailed
               << ",\"resident_linear_ct32_sprite_batches_completed\":"
               << statistics.residentLinearCt32SpriteBatchesCompleted
               << ",\"resident_linear_ct32_sprite_batches_failed\":"
               << statistics.residentLinearCt32SpriteBatchesFailed
               << ",\"largest_resident_linear_ct32_sprite_batch\":"
               << statistics.largestResidentLinearCt32SpriteBatch
               << ",\"triangle_draws_completed\":"
               << statistics.triangleDrawsCompleted
               << ",\"triangle_draws_failed\":"
               << statistics.triangleDrawsFailed
               << ",\"triangle_candidate_pixels_executed\":"
               << statistics.triangleCandidatePixelsExecuted
               << ",\"gouraud_depth_ct32_triangle_draws_completed\":"
               << statistics.gouraudDepthCt32TriangleDrawsCompleted
               << ",\"gouraud_depth_ct32_triangle_draws_failed\":"
               << statistics.gouraudDepthCt32TriangleDrawsFailed
               << ",\"gouraud_depth_ct32_triangle_candidate_pixels_executed\":"
               << statistics
                      .gouraudDepthCt32TriangleCandidatePixelsExecuted
               << ",\"resident_gouraud_depth_ct32_triangle_batches_completed\":"
               << statistics
                      .residentGouraudDepthCt32TriangleBatchesCompleted
               << ",\"resident_gouraud_depth_ct32_triangle_batches_failed\":"
               << statistics
                      .residentGouraudDepthCt32TriangleBatchesFailed
               << ",\"largest_resident_gouraud_depth_ct32_triangle_batch\":"
               << statistics
                      .largestResidentGouraudDepthCt32TriangleBatch
               << ",\"t8_gouraud_depth_ct32_triangle_draws_completed\":"
               << statistics.t8GouraudDepthCt32TriangleDrawsCompleted
               << ",\"t8_gouraud_depth_ct32_triangle_draws_failed\":"
               << statistics.t8GouraudDepthCt32TriangleDrawsFailed
               << ",\"t8_gouraud_depth_ct32_triangle_candidate_pixels_executed\":"
               << statistics
                      .t8GouraudDepthCt32TriangleCandidatePixelsExecuted
               << ",\"resident_t8_gouraud_depth_ct32_triangle_batches_completed\":"
               << statistics
                      .residentT8GouraudDepthCt32TriangleBatchesCompleted
               << ",\"resident_t8_gouraud_depth_ct32_triangle_batches_failed\":"
               << statistics
                      .residentT8GouraudDepthCt32TriangleBatchesFailed
               << ",\"largest_resident_t8_gouraud_depth_ct32_triangle_batch\":"
               << statistics
                      .largestResidentT8GouraudDepthCt32TriangleBatch
               << ",\"resident_sprite_batches_completed\":"
               << statistics.residentSpriteBatchesCompleted
               << ",\"resident_sprite_batches_failed\":"
               << statistics.residentSpriteBatchesFailed
               << ",\"largest_resident_sprite_batch\":"
               << statistics.largestResidentSpriteBatch
               << ",\"resident_triangle_batches_completed\":"
               << statistics.residentTriangleBatchesCompleted
               << ",\"resident_triangle_batches_failed\":"
               << statistics.residentTriangleBatchesFailed
               << ",\"largest_resident_triangle_batch\":"
               << statistics.largestResidentTriangleBatch
               << ",\"page_upload_operations_completed\":"
               << statistics.pageUploadOperationsCompleted
               << ",\"page_upload_operations_failed\":"
               << statistics.pageUploadOperationsFailed
               << ",\"page_download_operations_completed\":"
               << statistics.pageDownloadOperationsCompleted
               << ",\"page_download_operations_failed\":"
               << statistics.pageDownloadOperationsFailed
               << ",\"pages_uploaded\":"
               << statistics.pagesUploaded
               << ",\"pages_downloaded\":"
               << statistics.pagesDownloaded
               << ",\"page_upload_regions\":"
               << statistics.pageUploadRegions
               << ",\"page_download_regions\":"
               << statistics.pageDownloadRegions
               << ",\"validation_warnings\":"
               << statistics.validationWarnings
               << ",\"validation_errors\":"
               << statistics.validationErrors
               << ",\"device_lost\":"
               << (statistics.deviceLost ? "true" : "false")
               << '}';
    }

    void writeVulkanRasterBackendStatistics(
        std::ostream &output,
        const GsVulkanRasterBackendStatistics &statistics)
    {
        output << "{\"commands_attempted\":"
               << statistics.commandsAttempted
               << ",\"commands_completed\":"
               << statistics.commandsCompleted
               << ",\"gpu_requests_failed\":"
               << statistics.gpuRequestsFailed
               << ",\"verified_commands\":"
               << statistics.verifiedCommands
               << ",\"committed_gpu_commands\":"
               << statistics.committedGpuCommands
               << ",\"verification_mismatches\":"
               << statistics.verificationMismatches
               << ",\"bytes_compared\":"
               << statistics.bytesCompared
               << ",\"flushes\":" << statistics.flushes
               << ",\"resident_commands\":"
               << statistics.residentCommands
               << ",\"resident_batches_completed\":"
               << statistics.residentBatchesCompleted
               << ",\"largest_resident_batch\":"
               << statistics.largestResidentBatch
               << ",\"resource_hazard_drains\":"
               << statistics.resourceHazardDrains
               << ",\"queue_backpressure_drains\":"
               << statistics.queueBackpressureDrains
               << ",\"pipeline_change_drains\":"
               << statistics.pipelineChangeDrains
               << ",\"cpu_access_preparations\":"
               << statistics.cpuAccessPreparations
               << ",\"page_downloads_by_reason\":{";
        for (size_t index = 0u;
             index < GS_FLUSH_REASON_COUNT;
             ++index)
        {
            if (index != 0u)
                output << ',';
            writeJsonString(
                output,
                gsFlushReasonName(
                    static_cast<GsFlushReason>(index)));
            output << ":{\"operations\":"
                   << statistics
                          .pageDownloadOperationsByReason[index]
                   << ",\"pages\":"
                   << statistics.pagesDownloadedByReason[index]
                   << '}';
        }
        output << "},\"page_ownership\":{\"synchronized_pages\":"
               << statistics.pageOwnership.synchronizedPages
               << ",\"cpu_newer_pages\":"
               << statistics.pageOwnership.cpuNewerPages
               << ",\"gpu_newer_pages\":"
               << statistics.pageOwnership.gpuNewerPages
               << "},\"coherency\":{\"cpu_write_operations\":"
               << statistics.coherency.cpuWriteOperations
               << ",\"cpu_write_pages\":"
               << statistics.coherency.cpuWritePages
               << ",\"gpu_write_operations\":"
               << statistics.coherency.gpuWriteOperations
               << ",\"gpu_write_pages\":"
               << statistics.coherency.gpuWritePages
               << ",\"cpu_to_gpu_operations\":"
               << statistics.coherency.cpuToGpuOperations
               << ",\"cpu_to_gpu_pages\":"
               << statistics.coherency.cpuToGpuPages
               << ",\"gpu_to_cpu_operations\":"
               << statistics.coherency.gpuToCpuOperations
               << ",\"gpu_to_cpu_pages\":"
               << statistics.coherency.gpuToCpuPages
               << ",\"rejected_transitions\":"
               << statistics.coherency.rejectedTransitions << '}'
               << ",\"last_verification_artifact\":";
        writeJsonString(
            output, statistics.lastVerificationArtifact);
        output << '}';
    }

    bool readPacketSizes(const std::string &path,
                         std::vector<size_t> &packetSizes)
    {
        std::ifstream input(path);
        if (!input)
            return false;

        std::string line;
        while (std::getline(input, line))
        {
            if (const size_t comment = line.find('#');
                comment != std::string::npos)
            {
                line.erase(comment);
            }

            const size_t first = line.find_first_not_of(" \t\r");
            if (first == std::string::npos)
                continue;
            const size_t last = line.find_last_not_of(" \t\r");
            const std::string valueText =
                line.substr(first, last - first + 1u);

            errno = 0;
            char *end = nullptr;
            const unsigned long long parsed =
                std::strtoull(valueText.c_str(), &end, 0);
            if (errno != 0 || end == valueText.c_str() ||
                !end || *end != '\0' ||
                parsed > std::numeric_limits<size_t>::max())
            {
                return false;
            }
            packetSizes.push_back(static_cast<size_t>(parsed));
        }
        return input.eof() && !packetSizes.empty();
    }

    uint64_t fnv1a64(const uint8_t *data, size_t size)
    {
        uint64_t hash = 14695981039346656037ull;
        for (size_t index = 0; index < size; ++index)
        {
            hash ^= data[index];
            hash *= 1099511628211ull;
        }
        return hash;
    }

    void appendU64ToFnv(uint64_t &hash, uint64_t value) noexcept
    {
        for (uint32_t shift = 0u; shift < 64u; shift += 8u)
        {
            hash ^= static_cast<uint8_t>(value >> shift);
            hash *= 1099511628211ull;
        }
    }

    void fillDeterministicVram(std::vector<uint8_t> &vram)
    {
        vram.resize(GS_VULKAN_VRAM_SIZE);
        uint32_t state = 0x52414331u;
        for (uint8_t &byte : vram)
        {
            state ^= state << 13u;
            state ^= state >> 17u;
            state ^= state << 5u;
            byte = static_cast<uint8_t>(state >> 24u);
        }
    }

    void writeHex64(std::ostream &output, uint64_t value)
    {
        output << "\"0x" << std::hex << std::setw(16)
               << std::setfill('0') << value << '"' << std::dec
               << std::setfill(' ');
    }

    void writeVulkanRoundTripReport(
        std::ostream &output,
        const GsVulkanCapabilityReport &capabilities,
        const GsVulkanServiceStatistics &statistics,
        uint64_t requested,
        uint64_t completed,
        bool exact,
        bool serviceHealthyAfterWork,
        bool shutdownComplete,
        uint64_t inputHash,
        const std::vector<uint8_t> *completedOutput,
        uint64_t failureIteration,
        const VramDifference *difference,
        std::string_view error)
    {
        output << "{\"schema_version\":1,"
               << "\"mode\":\"vulkan-roundtrip\",";
        writeVulkanCapabilityFields(output, capabilities);
        output << ",\"vram_bytes\":" << GS_VULKAN_VRAM_SIZE
               << ",\"iterations_requested\":" << requested
               << ",\"iterations_completed\":" << completed
               << ",\"exact\":" << (exact ? "true" : "false")
               << ",\"service_healthy_after_work\":"
               << (serviceHealthyAfterWork ? "true" : "false")
               << ",\"shutdown_complete\":"
               << (shutdownComplete ? "true" : "false")
               << ",\"input_fnv1a64\":";
        writeHex64(output, inputHash);
        output << ",\"output_fnv1a64\":";
        if (completedOutput)
        {
            writeHex64(output, fnv1a64(
                completedOutput->data(), completedOutput->size()));
        }
        else
        {
            output << "null";
        }
        output << ",\"failure_iteration\":";
        if (failureIteration != 0u)
            output << failureIteration;
        else
            output << "null";
        if (difference && !difference->matches)
        {
            output << ",\"first_differing_byte_offset\":"
                   << difference->byteOffset
                   << ",\"first_differing_page\":"
                   << difference->page
                   << ",\"expected_byte\":"
                   << static_cast<uint32_t>(difference->expected)
                   << ",\"actual_byte\":"
                   << static_cast<uint32_t>(difference->actual);
        }
        output << ",\"statistics\":";
        writeVulkanServiceStatistics(output, statistics);
        output << ",\"error\":";
        writeJsonString(output, error);
        output << "}\n";
    }

    bool parseRegisterAssignment(
        const std::string &text,
        std::pair<uint8_t, uint64_t> &assignment)
    {
        const size_t equals = text.find('=');
        if (equals == std::string::npos ||
            equals == 0u ||
            equals + 1u == text.size())
        {
            return false;
        }

        uint64_t address = 0u;
        uint64_t value = 0u;
        if (!parseUnsigned(text.substr(0u, equals), address) ||
            !parseUnsigned(text.substr(equals + 1u), value) ||
            address > 0xFFu)
        {
            return false;
        }

        assignment = {
            static_cast<uint8_t>(address),
            value,
        };
        return true;
    }

    bool readRegisterFile(
        const std::string &path,
        std::vector<std::pair<uint8_t, uint64_t>> &assignments,
        std::string &statePath)
    {
        std::ifstream input(path);
        if (!input)
            return false;

        std::string line;
        while (std::getline(input, line))
        {
            if (const size_t comment = line.find('#');
                comment != std::string::npos)
            {
                line.erase(comment);
            }

            const size_t first = line.find_first_not_of(" \t\r");
            if (first == std::string::npos)
                continue;
            const size_t last = line.find_last_not_of(" \t\r");
            const std::string content =
                line.substr(first, last - first + 1u);

            constexpr std::string_view statePrefix = "@state-file=";
            if (content.starts_with(statePrefix))
            {
                const std::string value =
                    content.substr(statePrefix.size());
                if (value.empty() || !statePath.empty())
                    return false;
                std::filesystem::path resolved(value);
                if (resolved.is_relative())
                {
                    resolved =
                        std::filesystem::path(path).parent_path() /
                        resolved;
                }
                statePath = resolved.lexically_normal().string();
                continue;
            }

            std::pair<uint8_t, uint64_t> assignment{};
            if (!parseRegisterAssignment(content, assignment))
            {
                return false;
            }
            assignments.push_back(assignment);
        }
        return input.eof();
    }
}

int main(int argc, char **argv)
{
    std::string vramInputPath;
    std::string vramOutputPath;
    std::string packetSizesPath;
    std::string hashTracePath;
    std::string compareVramPath;
    std::string explicitStatePath;
    std::string registerStatePath;
    std::string verificationArtifactDirectory;
    bool batchStream = false;
    bool batchDrains = false;
    bool backendStats = false;
    bool replayOptionUsed = false;
    bool gifReplayOptionUsed = false;
    bool vulkanInfo = false;
    bool vulkanRoundTrip = false;
    bool vulkanOptionUsed = false;
    uint64_t vulkanRoundTripCount = 0u;
    GsVulkanProbeConfig vulkanConfig{};
    GsVulkanRasterBackendConfig vulkanBackendConfig{};
    bool commandLimitSet = false;
    bool packetLimitSet = false;
    uint64_t commandLimit = 0u;
    uint64_t packetLimit = 0u;
    GsRendererMode rendererMode = GsRendererMode::Software;
    GsExecutionMode executionMode = GsExecutionMode::Inline;
    std::vector<std::pair<uint8_t, uint64_t>> fileRegisters;
    std::vector<std::pair<uint8_t, uint64_t>> overrideRegisters;
    std::vector<std::string> packetPaths;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--batch-stream")
        {
            batchStream = true;
            replayOptionUsed = true;
            gifReplayOptionUsed = true;
        }
        else if (argument == "--batch-drains")
        {
            batchDrains = true;
            replayOptionUsed = true;
            gifReplayOptionUsed = true;
        }
        else if (argument == "--backend-stats")
        {
            backendStats = true;
            replayOptionUsed = true;
            gifReplayOptionUsed = true;
        }
        else if (argument == "--vulkan-info")
        {
            vulkanInfo = true;
        }
        else if (argument == "--vulkan-roundtrip")
        {
            if (++index >= argc || vulkanRoundTrip)
            {
                printUsage();
                return 2;
            }
            constexpr uint64_t maxRoundTrips = 1024u;
            if (!parseCount(argv[index], vulkanRoundTripCount) ||
                vulkanRoundTripCount == 0u ||
                vulkanRoundTripCount > maxRoundTrips)
            {
                std::cerr << "Vulkan round-trip count must be between 1 and "
                          << maxRoundTrips << '\n';
                return 2;
            }
            vulkanRoundTrip = true;
        }
        else if (argument == "--vulkan-validation")
        {
            vulkanConfig.enableValidation = true;
            vulkanOptionUsed = true;
        }
        else if (argument == "--vram-in" ||
            argument == "--vram-out" ||
            argument == "--register" ||
            argument == "--register-file" ||
            argument == "--state-in" ||
            argument == "--packet-sizes" ||
            argument == "--hash-trace" ||
            argument == "--renderer" ||
            argument == "--gs-execution" ||
            argument == "--stop-after-command" ||
            argument == "--stop-after-packet" ||
            argument == "--compare-vram" ||
            argument == "--verify-dump-dir" ||
            argument == "--vulkan-max-resident-batch" ||
            argument == "--vulkan-min-hybrid-pixels" ||
            argument ==
                "--vulkan-min-hybrid-source-copy-alpha-pixels" ||
            argument == "--vulkan-min-hybrid-depth-ct32-pixels" ||
            argument ==
                "--vulkan-min-hybrid-alpha-fail-depth-run-pixels" ||
            argument == "--vulkan-min-hybrid-nearest-ct32-pixels" ||
            argument == "--vulkan-min-hybrid-linear-ct32-pixels" ||
            argument == "--vulkan-min-hybrid-linear-ct32-clamp-pixels" ||
            argument ==
                "--vulkan-min-hybrid-feedback-ct32-run-pixels" ||
            argument == "--vulkan-min-hybrid-triangle-pixels" ||
            argument ==
                "--vulkan-min-hybrid-gouraud-depth-run-pixels" ||
            argument == "--vulkan-loader" ||
            argument == "--vulkan-vendor" ||
            argument == "--vulkan-device")
        {
            if (++index >= argc)
            {
                printUsage();
                return 2;
            }

            if (argument == "--vulkan-loader")
            {
                vulkanConfig.loaderPath = argv[index];
                vulkanOptionUsed = true;
            }
            else if (argument == "--vulkan-vendor" ||
                     argument == "--vulkan-device")
            {
                uint64_t id = 0u;
                if (!parseCount(argv[index], id) ||
                    id > std::numeric_limits<uint32_t>::max())
                {
                    std::cerr << "invalid Vulkan device ID: "
                              << argv[index] << '\n';
                    return 2;
                }
                if (argument == "--vulkan-vendor")
                    vulkanConfig.preferredVendorId =
                        static_cast<uint32_t>(id);
                else
                    vulkanConfig.preferredDeviceId =
                        static_cast<uint32_t>(id);
                vulkanOptionUsed = true;
            }
            else if (argument == "--vram-in")
            {
                vramInputPath = argv[index];
                replayOptionUsed = true;
            }
            else if (argument == "--verify-dump-dir")
            {
                verificationArtifactDirectory = argv[index];
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument == "--vulkan-max-resident-batch")
            {
                uint64_t maximum = 0u;
                if (!parseCount(argv[index], maximum) || maximum == 0u ||
                    maximum > GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH)
                {
                    std::cerr
                        << "Vulkan resident batch bound must be between 1 and "
                        << GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH << '\n';
                    return 2;
                }
                vulkanBackendConfig.maximumResidentBatchCommands =
                    static_cast<size_t>(maximum);
                vulkanOptionUsed = true;
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument == "--vulkan-min-hybrid-pixels")
            {
                uint64_t minimum = 0u;
                constexpr uint64_t maximumPixels = 2048ull * 2048ull;
                if (!parseCount(argv[index], minimum) ||
                    minimum > maximumPixels)
                {
                    std::cerr
                        << "Vulkan hybrid pixel threshold must be between 0 and "
                        << maximumPixels << '\n';
                    return 2;
                }
                vulkanBackendConfig.minimumHybridSpritePixels = minimum;
                vulkanOptionUsed = true;
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument ==
                     "--vulkan-min-hybrid-source-copy-alpha-pixels")
            {
                uint64_t minimum = 0u;
                constexpr uint64_t maximumPixels = 2048ull * 2048ull;
                if (!parseCount(argv[index], minimum) ||
                    minimum > maximumPixels)
                {
                    std::cerr
                        << "Vulkan hybrid source-copy alpha pixel threshold "
                           "must be between 0 and "
                        << maximumPixels << '\n';
                    return 2;
                }
                vulkanBackendConfig
                    .minimumHybridSourceCopyAlphaSpritePixels = minimum;
                vulkanOptionUsed = true;
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument ==
                     "--vulkan-min-hybrid-depth-ct32-pixels")
            {
                uint64_t minimum = 0u;
                constexpr uint64_t maximumPixels = 2048ull * 2048ull;
                if (!parseCount(argv[index], minimum) ||
                    minimum > maximumPixels)
                {
                    std::cerr
                        << "Vulkan hybrid depth CT32 pixel threshold "
                           "must be between 0 and "
                        << maximumPixels << '\n';
                    return 2;
                }
                vulkanBackendConfig.minimumHybridDepthCt32SpritePixels =
                    minimum;
                vulkanOptionUsed = true;
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument ==
                     "--vulkan-min-hybrid-alpha-fail-depth-run-pixels")
            {
                uint64_t minimum = 0u;
                constexpr uint64_t maximumPixels =
                    2048ull * 2048ull *
                    GS_VULKAN_MAX_RESIDENT_DEPTH_CT32_BATCH;
                if (!parseCount(argv[index], minimum) ||
                    minimum > maximumPixels)
                {
                    std::cerr
                        << "Vulkan hybrid alpha-fail depth run-pixel threshold "
                           "must be between 0 and "
                        << maximumPixels << '\n';
                    return 2;
                }
                vulkanBackendConfig
                    .minimumHybridFramebufferOnlyAlphaFailDepthCt32RunPixels =
                    minimum;
                vulkanOptionUsed = true;
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument ==
                     "--vulkan-min-hybrid-nearest-ct32-pixels")
            {
                uint64_t minimum = 0u;
                constexpr uint64_t maximumPixels = 2048ull * 2048ull;
                if (!parseCount(argv[index], minimum) ||
                    minimum > maximumPixels)
                {
                    std::cerr
                        << "Vulkan hybrid nearest CT32 pixel threshold "
                           "must be between 0 and "
                        << maximumPixels << '\n';
                    return 2;
                }
                vulkanBackendConfig.minimumHybridNearestCt32SpritePixels =
                    minimum;
                vulkanOptionUsed = true;
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument ==
                     "--vulkan-min-hybrid-linear-ct32-pixels")
            {
                uint64_t minimum = 0u;
                constexpr uint64_t maximumPixels = 2048ull * 2048ull;
                if (!parseCount(argv[index], minimum) ||
                    minimum > maximumPixels)
                {
                    std::cerr
                        << "Vulkan hybrid linear CT32 pixel threshold "
                           "must be between 0 and "
                        << maximumPixels << '\n';
                    return 2;
                }
                vulkanBackendConfig.minimumHybridLinearCt32SpritePixels =
                    minimum;
                vulkanOptionUsed = true;
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument ==
                     "--vulkan-min-hybrid-linear-ct32-clamp-pixels")
            {
                uint64_t minimum = 0u;
                constexpr uint64_t maximumPixels = 2048ull * 2048ull;
                if (!parseCount(argv[index], minimum) ||
                    minimum > maximumPixels)
                {
                    std::cerr
                        << "Vulkan hybrid linear CT32 clamp pixel threshold "
                           "must be between 0 and "
                        << maximumPixels << '\n';
                    return 2;
                }
                vulkanBackendConfig
                    .minimumHybridLinearCt32ClampSpritePixels = minimum;
                vulkanOptionUsed = true;
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument ==
                     "--vulkan-min-hybrid-feedback-ct32-run-pixels")
            {
                uint64_t minimum = 0u;
                constexpr uint64_t maximumPixels =
                    2048ull * 2048ull *
                    GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH;
                if (!parseCount(argv[index], minimum) ||
                    minimum > maximumPixels)
                {
                    std::cerr
                        << "Vulkan hybrid feedback CT32 run-pixel threshold "
                           "must be between 0 and "
                        << maximumPixels << '\n';
                    return 2;
                }
                vulkanBackendConfig
                    .minimumHybridFeedbackLinearDepthCt32RunPixels =
                    minimum;
                vulkanOptionUsed = true;
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument ==
                     "--vulkan-min-hybrid-triangle-pixels")
            {
                uint64_t minimum = 0u;
                constexpr uint64_t maximumPixels = 2048ull * 2048ull;
                if (!parseCount(argv[index], minimum) ||
                    minimum > maximumPixels)
                {
                    std::cerr
                        << "Vulkan hybrid triangle candidate-pixel threshold "
                           "must be between 0 and "
                        << maximumPixels << '\n';
                    return 2;
                }
                vulkanBackendConfig.minimumHybridTriangleCandidatePixels =
                    minimum;
                vulkanOptionUsed = true;
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument ==
                     "--vulkan-min-hybrid-gouraud-depth-run-pixels")
            {
                uint64_t minimum = 0u;
                constexpr uint64_t maximumPixels =
                    2048ull * 2048ull *
                    GS_VULKAN_MAX_RESIDENT_GOURAUD_DEPTH_CT32_TRIANGLE_BATCH;
                if (!parseCount(argv[index], minimum) ||
                    minimum > maximumPixels)
                {
                    std::cerr
                        << "Vulkan hybrid Gouraud depth triangle run-pixel threshold "
                           "must be between 0 and "
                        << maximumPixels << '\n';
                    return 2;
                }
                vulkanBackendConfig
                    .minimumHybridGouraudDepthCt32TriangleRunPixels =
                    minimum;
                vulkanOptionUsed = true;
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument == "--vram-out")
            {
                vramOutputPath = argv[index];
                replayOptionUsed = true;
            }
            else if (argument == "--packet-sizes")
            {
                packetSizesPath = argv[index];
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument == "--hash-trace")
            {
                hashTracePath = argv[index];
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument == "--state-in")
            {
                explicitStatePath = argv[index];
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument == "--compare-vram")
            {
                compareVramPath = argv[index];
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
            }
            else if (argument == "--renderer")
            {
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
                if (!parseRendererMode(argv[index], rendererMode))
                {
                    std::cerr << "invalid renderer mode: "
                              << argv[index] << '\n';
                    return 2;
                }
            }
            else if (argument == "--gs-execution")
            {
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
                if (!parseGsExecutionMode(
                        argv[index], executionMode))
                {
                    std::cerr << "invalid GS execution mode: "
                              << argv[index] << '\n';
                    return 2;
                }
            }
            else if (argument == "--stop-after-command")
            {
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
                if (!parseCount(argv[index], commandLimit))
                {
                    std::cerr << "invalid command limit: "
                              << argv[index] << '\n';
                    return 2;
                }
                commandLimitSet = true;
            }
            else if (argument == "--stop-after-packet")
            {
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
                if (!parseCount(argv[index], packetLimit))
                {
                    std::cerr << "invalid packet limit: "
                              << argv[index] << '\n';
                    return 2;
                }
                packetLimitSet = true;
            }
            else if (argument == "--register-file")
            {
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
                if (!readRegisterFile(
                        argv[index],
                        fileRegisters,
                        registerStatePath))
                {
                    std::cerr << "failed to read register file: "
                              << argv[index] << '\n';
                    return 2;
                }
            }
            else
            {
                replayOptionUsed = true;
                gifReplayOptionUsed = true;
                std::pair<uint8_t, uint64_t> assignment{};
                if (!parseRegisterAssignment(argv[index], assignment))
                {
                    std::cerr << "invalid register assignment: "
                              << argv[index] << '\n';
                    return 2;
                }
                overrideRegisters.push_back(assignment);
            }
        }
        else if (argument == "--help" || argument == "-h")
        {
            printUsage();
            return 0;
        }
        else if (!argument.empty() && argument[0] == '-')
        {
            std::cerr << "unknown option: " << argument << '\n';
            printUsage();
            return 2;
        }
        else
        {
            packetPaths.push_back(argument);
        }
    }

    if (vulkanInfo && vulkanRoundTrip)
    {
        std::cerr << "--vulkan-info and --vulkan-roundtrip are mutually exclusive\n";
        return 2;
    }
    if (vulkanInfo)
    {
        if (replayOptionUsed || !packetPaths.empty())
        {
            std::cerr << "--vulkan-info cannot be combined with GIF replay inputs\n";
            return 2;
        }
        const GsVulkanCapabilityReport report =
            probeGsVulkanCapabilities(vulkanConfig);
        writeVulkanCapabilityReport(std::cout, report);
        return report.ready() ? 0 : 1;
    }
    if (vulkanRoundTrip)
    {
        if (gifReplayOptionUsed || !packetPaths.empty())
        {
            std::cerr << "--vulkan-roundtrip cannot be combined with GIF replay inputs\n";
            return 2;
        }

        std::vector<uint8_t> inputVram;
        if (vramInputPath.empty())
        {
            fillDeterministicVram(inputVram);
        }
        else if (!readFile(vramInputPath, inputVram))
        {
            std::cerr << "failed to read initial GS memory: "
                      << vramInputPath << '\n';
            return 2;
        }
        if (inputVram.size() != GS_VULKAN_VRAM_SIZE)
        {
            std::cerr << "initial GS memory must be exactly "
                      << GS_VULKAN_VRAM_SIZE << " bytes\n";
            return 2;
        }

        GsVulkanServiceConfig serviceConfig{};
        serviceConfig.probe = vulkanConfig;
        GsVulkanCapabilityReport report{};
        std::string operationError;
        std::unique_ptr<GsVulkanService> service =
            GsVulkanService::create(
                serviceConfig, &report, &operationError);
        GsVulkanServiceStatistics statistics{};
        std::vector<uint8_t> completedOutput;
        bool outputAvailable = false;
        bool serviceHealthyAfterWork = false;
        bool shutdownComplete = false;
        bool exact = false;
        uint64_t iterationsCompleted = 0u;
        uint64_t failureIteration = 0u;
        VramDifference difference{};
        bool hasDifference = false;

        if (service)
        {
            operationError.clear();
            for (uint64_t iteration = 0u;
                 iteration < vulkanRoundTripCount;
                 ++iteration)
            {
                std::vector<uint8_t> iterationOutput;
                if (!service->roundTripVram(
                        inputVram, iterationOutput, &operationError))
                {
                    failureIteration = iteration + 1u;
                    break;
                }
                iterationsCompleted = iteration + 1u;
                completedOutput = std::move(iterationOutput);
                outputAvailable = true;
                if (completedOutput != inputVram)
                {
                    difference = compareVram(
                        completedOutput.data(), inputVram);
                    hasDifference = true;
                    failureIteration = iteration + 1u;
                    operationError =
                        "Vulkan round trip changed GS VRAM";
                    break;
                }
            }
            exact = iterationsCompleted == vulkanRoundTripCount &&
                    !hasDifference && operationError.empty();
            serviceHealthyAfterWork = service->healthy();
            service->shutdown();
            shutdownComplete = true;
            report = service->capabilities();
            statistics = service->statistics();
            if (exact &&
                (!report.ready() || statistics.validationErrors != 0u ||
                 statistics.deviceLost))
            {
                exact = false;
                operationError = report.detail.empty()
                    ? "Vulkan service failed during shutdown"
                    : report.detail;
            }
        }

        if (!vramOutputPath.empty() && outputAvailable &&
            !writeFile(vramOutputPath,
                       completedOutput.data(), completedOutput.size()))
        {
            std::cerr << "failed to write final GS memory: "
                      << vramOutputPath << '\n';
            return 2;
        }

        writeVulkanRoundTripReport(
            std::cout, report, statistics,
            vulkanRoundTripCount, iterationsCompleted,
            exact, serviceHealthyAfterWork, shutdownComplete,
            fnv1a64(inputVram.data(), inputVram.size()),
            outputAvailable ? &completedOutput : nullptr,
            failureIteration,
            hasDifference ? &difference : nullptr,
            operationError);
        return exact ? 0 : 1;
    }
    if (vulkanOptionUsed && rendererMode == GsRendererMode::Software)
    {
        std::cerr << "Vulkan options require a Vulkan probe, round trip, "
                     "or accelerated renderer mode\n";
        return 2;
    }
    if (!verificationArtifactDirectory.empty() &&
        rendererMode != GsRendererMode::Verify)
    {
        std::cerr << "--verify-dump-dir requires --renderer verify\n";
        return 2;
    }

    if (packetPaths.empty())
    {
        printUsage();
        return 2;
    }

    std::vector<uint8_t> expectedVram;
    if (!compareVramPath.empty())
    {
        if (!readFile(compareVramPath, expectedVram))
        {
            std::cerr << "failed to read comparison GS memory: "
                      << compareVramPath << '\n';
            return 2;
        }
        if (expectedVram.size() != PS2_GS_VRAM_SIZE)
        {
            std::cerr << "comparison GS memory must be exactly "
                      << PS2_GS_VRAM_SIZE << " bytes\n";
            return 2;
        }
    }

    const std::string statePath = explicitStatePath.empty()
        ? registerStatePath
        : explicitStatePath;
    GsReplayState replayState{};
    bool replayStateLoaded = false;
    if (!statePath.empty())
    {
        std::vector<uint8_t> encodedState;
        if (!readFile(statePath, encodedState))
        {
            std::cerr << "failed to read initial GS replay state: "
                      << statePath << '\n';
            return 2;
        }
        std::string stateError;
        if (!decodeGsReplayState(
                encodedState, replayState, &stateError))
        {
            std::cerr << "failed to decode initial GS replay state: "
                      << stateError << '\n';
            return 2;
        }
        replayStateLoaded = true;
    }

    PS2Memory memory;
    if (!memory.initialize())
    {
        std::cerr << "failed to initialize memory\n";
        return 2;
    }

    GS gs;
    gs.init(
        memory.getGSVRAM(),
        static_cast<uint32_t>(PS2_GS_VRAM_SIZE),
        executionMode == GsExecutionMode::ThreadedAsync
            ? nullptr
            : &memory.gs());
    GsCommandProcessor gsProcessor(gs, true);
    std::unique_ptr<GsCommandExecutor> gsExecutor;
    switch (executionMode)
    {
    case GsExecutionMode::Inline:
        gsExecutor = std::make_unique<InlineGsExecutor>(
            gsProcessor);
        break;
    case GsExecutionMode::ThreadedSynchronous:
        gsExecutor = std::make_unique<ThreadedGsExecutor>(
            gsProcessor);
        break;
    case GsExecutionMode::ThreadedAsync:
        gsExecutor = std::make_unique<ThreadedGsExecutor>(
            gsProcessor);
        break;
    }
    auto *const threadedGsExecutor =
        dynamic_cast<ThreadedGsExecutor *>(
            gsExecutor.get());
    uint64_t commandDigest = 14695981039346656037ull;
    uint64_t gsCommands = 0u;
    std::deque<GsCommandSubmission> pendingGsCommands;
    auto recordGsResult = [&](const GsCommandResult &result)
    {
        publishGsPrivilegedSideEffects(
            memory.gs(), result.privilegedEffects);
        appendU64ToFnv(
            commandDigest,
            gsCommandDigestHash(result.digest));
        ++gsCommands;
    };
    auto reapPendingGs = [&](bool waitAll)
    {
        while (!pendingGsCommands.empty() &&
               (waitAll || pendingGsCommands.front().ready()))
        {
            GsCommandResult result =
                pendingGsCommands.front().wait();
            pendingGsCommands.pop_front();
            recordGsResult(result);
        }
    };
    auto submitGs = [&](GsCommandPayload payload)
    {
        if (executionMode == GsExecutionMode::ThreadedAsync)
            reapPendingGs(true);
        GsCommandResult result =
            gsExecutor->submit(std::move(payload));
        recordGsResult(result);
        return result;
    };
    auto submitGsDrain = [&](GsDrainBatchCommand batch)
    {
        if (executionMode != GsExecutionMode::ThreadedAsync)
        {
            (void)submitGs(std::move(batch));
            return;
        }
        pendingGsCommands.push_back(
            threadedGsExecutor->submitAsync(
                std::move(batch)));
        reapPendingGs(false);
    };

    if (rendererMode != GsRendererMode::Software)
    {
        GsVulkanServiceConfig serviceConfig{};
        serviceConfig.probe = vulkanConfig;
        vulkanBackendConfig.verificationArtifactDirectory =
            verificationArtifactDirectory;
        if (!takeGsCommandResult<GsBooleanResult>(
                submitGs(
                    GsConfigureVulkanCommand{
                        .service = std::move(serviceConfig),
                        .backend = std::move(vulkanBackendConfig)}))
                 .value)
        {
            const GsRendererStatusResult status =
                takeGsCommandResult<GsRendererStatusResult>(
                    submitGs(GsRendererStatusCommand{}));
            std::cerr << "failed to configure Vulkan renderer: "
                      << status.diagnostic << '\n';
            return 2;
        }
    }
    if (!takeGsCommandResult<GsBooleanResult>(
            submitGs(
                GsSetRendererModeCommand{
                    .mode = rendererMode}))
             .value)
    {
        std::cerr << "renderer mode is unavailable: "
                  << gsRendererModeName(rendererMode);
        const std::string diagnostic =
            takeGsCommandResult<GsRendererStatusResult>(
                submitGs(GsRendererStatusCommand{}))
                .diagnostic;
        if (!diagnostic.empty())
            std::cerr << ": " << diagnostic;
        std::cerr << '\n';
        return 2;
    }

    if (!vramInputPath.empty())
    {
        std::vector<uint8_t> initialVram;
        if (!readFile(vramInputPath, initialVram))
        {
            std::cerr << "failed to read initial GS memory: "
                      << vramInputPath << '\n';
            return 2;
        }
        if (initialVram.size() != PS2_GS_VRAM_SIZE)
        {
            std::cerr << "initial GS memory must be exactly "
                      << PS2_GS_VRAM_SIZE << " bytes\n";
            return 2;
        }

        std::memcpy(memory.getGSVRAM(), initialVram.data(),
                    initialVram.size());
    }

    if (replayStateLoaded)
    {
        if (!takeGsCommandResult<GsBooleanResult>(
                submitGs(
                    GsRestoreReplayStateCommand{
                        .state = std::move(replayState)}))
                 .value)
        {
            std::cerr << "initial GS replay state is incompatible "
                         "with this runtime\n";
            return 2;
        }
    }
    else
    {
        for (const auto &[address, value] : fileRegisters)
        {
            (void)submitGs(
                GsWriteRegisterCommand{
                    .address = address,
                    .value = value});
        }
    }
    for (const auto &[address, value] : overrideRegisters)
    {
        (void)submitGs(
            GsWriteRegisterCommand{
                .address = address,
                .value = value});
    }

    if (batchStream && batchDrains)
    {
        std::cerr << "--batch-stream and --batch-drains are mutually exclusive\n";
        return 2;
    }

    if (backendStats)
    {
        (void)submitGs(GsResetBackendCountersCommand{});
        (void)submitGs(
            GsSetBackendCountersEnabledCommand{
                .enabled = true});
    }
    if (commandLimitSet)
    {
        (void)submitGs(
            GsSetDrawCommandLimitCommand{
                .maximumCommands = commandLimit});
    }

    struct RenderBatchScope
    {
        GsCommandExecutor *executor = nullptr;
        ~RenderBatchScope()
        {
            if (!executor)
                return;
            try
            {
                (void)executor->submit(
                    GsEndRenderBatchCommand{});
            }
            catch (...)
            {
            }
        }
    } renderBatchScope{};

    std::ofstream hashTrace;
    if (!hashTracePath.empty())
    {
        hashTrace.open(hashTracePath, std::ios::out | std::ios::trunc);
        if (!hashTrace)
        {
            std::cerr << "failed to open hash trace: "
                      << hashTracePath << '\n';
            return 2;
        }
        hashTrace << "index,size,input_fnv1a64,vram_fnv1a64\n";
    }

    std::vector<size_t> packetSizes;
    if (!packetSizesPath.empty() &&
        !readPacketSizes(packetSizesPath, packetSizes))
    {
        std::cerr << "failed to read packet sizes: "
                  << packetSizesPath << '\n';
        return 2;
    }
    if (!packetSizes.empty() && packetPaths.size() != 1u)
    {
        std::cerr << "--packet-sizes requires exactly one GIF_PACKET\n";
        return 2;
    }

    uint64_t totalBytes = 0u;
    uint64_t packetIndex = 0u;
    std::string executionError;
    bool stopped = false;
    bool stoppedWithinPacket = false;
    const char *stopReason = "complete";
    if (batchStream)
    {
        (void)submitGs(GsBeginRenderBatchCommand{});
        renderBatchScope.executor = gsExecutor.get();
    }
    if (commandLimitSet &&
        takeGsCommandResult<GsDrawStatusResult>(
            submitGs(GsDrawStatusCommand{}))
            .limitReached)
    {
        stopped = true;
        stopReason = "command-limit";
    }
    else if (packetLimitSet && packetLimit == 0u)
    {
        stopped = true;
        stopReason = "packet-limit";
    }

    auto processPacket = [&](const uint8_t *data, size_t size) -> bool
    {
        if (size > std::numeric_limits<uint32_t>::max())
        {
            std::cerr << "GIF packet is too large\n";
            return false;
        }

        try
        {
            GsDrainBatchCommand batch{};
            batch.beginSubmissionBatch = batchDrains;
            batch.endSubmissionBatch = batchDrains;
            GifArbiterDrainBatch::Packet packet{};
            packet.pathId = GifPathId::Path3;
            packet.storageIndex = 3u;
            packet.size = size;
            batch.batch.storage[3].assign(data, data + size);
            batch.batch.packets.push_back(std::move(packet));
            submitGsDrain(std::move(batch));
        }
        catch (const std::exception &error)
        {
            executionError =
                "GS replay failed at packet " +
                std::to_string(packetIndex) + ": " + error.what();
            std::cerr << executionError << '\n';
            return false;
        }
        totalBytes += size;
        if (hashTrace.is_open())
        {
            (void)submitGs(GsFlushRenderBatchCommand{});
            hashTrace << packetIndex << ',' << size << ','
                      << std::hex << std::setw(16) << std::setfill('0')
                      << fnv1a64(data, size) << ','
                      << std::hex << std::setw(16) << std::setfill('0')
                      << fnv1a64(memory.getGSVRAM(), PS2_GS_VRAM_SIZE)
                      << std::dec << '\n';
        }
        ++packetIndex;
        if (commandLimitSet &&
            takeGsCommandResult<GsDrawStatusResult>(
                submitGs(GsDrawStatusCommand{}))
                .limitReached)
        {
            stopped = true;
            stoppedWithinPacket = true;
            stopReason = "command-limit";
        }
        else if (packetLimitSet && packetIndex >= packetLimit)
        {
            stopped = true;
            stopReason = "packet-limit";
        }
        return true;
    };

    for (const std::string &packetPath : packetPaths)
    {
        if (stopped)
            break;

        std::vector<uint8_t> packet;
        if (!readFile(packetPath, packet))
        {
            std::cerr << "failed to read GIF packet: " << packetPath << '\n';
            return 2;
        }

        if (packetSizes.empty())
        {
            if (!processPacket(packet.data(), packet.size()))
                return executionError.empty() ? 2 : 1;
            continue;
        }

        size_t offset = 0u;
        for (size_t size : packetSizes)
        {
            if (offset > packet.size() || size > packet.size() - offset)
            {
                std::cerr << "packet sizes exceed GIF stream length\n";
                return 2;
            }
            if (!processPacket(packet.data() + offset, size))
                return executionError.empty() ? 2 : 1;
            offset += size;
            if (stopped)
                break;
        }
        if (!stopped && offset != packet.size())
        {
            std::cerr << "packet sizes do not consume GIF stream\n";
            return 2;
        }
    }

    if (batchStream)
    {
        (void)submitGs(GsEndRenderBatchCommand{});
        renderBatchScope.executor = nullptr;
    }

    // Final comparison, hashing, and output are CPU observations. Hybrid may
    // deliberately leave newer pages resident on the GPU after the last
    // packet, so use the ordinary save/load observation boundary to publish
    // canonical CPU VRAM before reading it directly below.
    GsRendererStatusResult rendererStatus =
        takeGsCommandResult<GsRendererStatusResult>(
            submitGs(GsRendererStatusCommand{}));
    if (rendererStatus.mode != GsRendererMode::Software)
    {
        (void)submitGs(GsCaptureReplayStateCommand{});
        rendererStatus =
            takeGsCommandResult<GsRendererStatusResult>(
                submitGs(GsRendererStatusCommand{}));
    }
    const GsDrawStatusResult drawStatus =
        takeGsCommandResult<GsDrawStatusResult>(
            submitGs(GsDrawStatusCommand{}));
    GsBackendCountersResult backendCounterResult{};
    if (backendStats)
    {
        backendCounterResult =
            takeGsCommandResult<GsBackendCountersResult>(
                submitGs(GsBackendCountersCommand{}));
    }
    const ThreadedGsExecutorStatistics ownerStatistics =
        threadedGsExecutor
            ? threadedGsExecutor->statistics()
            : ThreadedGsExecutorStatistics{};

    const VramDifference difference = expectedVram.empty()
        ? VramDifference{}
        : compareVram(memory.getGSVRAM(), expectedVram);

    if (!vramOutputPath.empty() &&
        !writeFile(vramOutputPath, memory.getGSVRAM(), PS2_GS_VRAM_SIZE))
    {
        std::cerr << "failed to write final GS memory: "
                  << vramOutputPath << '\n';
        return 2;
    }

    std::cout << "{\"schema_version\":1,\"renderer\":\""
              << gsRendererModeName(rendererStatus.mode)
              << "\",\"gs_execution\":\""
              << gsExecutionModeName(executionMode)
              << "\",\"packets\":" << packetIndex
              << ",\"bytes\":" << totalBytes
              << ",\"commands\":" << drawStatus.submittedCommands
              << ",\"gs_stream_commands\":" << gsCommands
              << ",\"gs_command_digest_fnv1a64\":\"0x"
              << std::hex << std::setw(16) << std::setfill('0')
              << commandDigest << '\"' << std::dec
              << ",\"state_restored\":"
              << (replayStateLoaded ? "true" : "false")
              << ",\"stopped\":" << (stopped ? "true" : "false")
              << ",\"stop_reason\":\"" << stopReason << '\"'
              << ",\"stopped_within_packet\":"
              << (stoppedWithinPacket ? "true" : "false")
              << ",\"final_vram_fnv1a64\":\"0x"
              << std::hex << std::setw(16) << std::setfill('0')
              << fnv1a64(memory.getGSVRAM(), PS2_GS_VRAM_SIZE)
              << '\"' << std::dec;

    if (threadedGsExecutor)
    {
        std::cout
            << ",\"gs_owner_statistics\":{"
            << "\"submitted_sequence\":"
            << ownerStatistics.submittedSequence
            << ",\"completed_sequence\":"
            << ownerStatistics.completedSequence
            << ",\"queue_high_water\":"
            << ownerStatistics.queueHighWater
            << ",\"payload_high_water_bytes\":"
            << ownerStatistics.payloadHighWaterBytes
            << ",\"producer_block_count\":"
            << ownerStatistics.producerBlockCount
            << ",\"producer_blocked_nanoseconds\":"
            << ownerStatistics.producerBlockedNanoseconds
            << ",\"producer_slot_wait_count\":"
            << ownerStatistics.producerSlotWaitCount
            << ",\"producer_slot_wait_nanoseconds\":"
            << ownerStatistics.producerSlotWaitNanoseconds
            << ",\"producer_payload_wait_count\":"
            << ownerStatistics.producerPayloadWaitCount
            << ",\"producer_payload_wait_nanoseconds\":"
            << ownerStatistics.producerPayloadWaitNanoseconds
            << ",\"worker_active_nanoseconds\":"
            << ownerStatistics.workerActiveNanoseconds
            << ",\"worker_idle_nanoseconds\":"
            << ownerStatistics.workerIdleNanoseconds
            << ",\"barrier_wait_count\":"
            << ownerStatistics.barrierWaitCount
            << ",\"barrier_wait_nanoseconds\":"
            << ownerStatistics.barrierWaitNanoseconds
            << '}';
    }

    if (!expectedVram.empty())
    {
        std::cout << ",\"vram_matches\":"
                  << (difference.matches ? "true" : "false");
        if (!difference.matches)
        {
            std::cout << ",\"first_differing_page\":"
                      << difference.page
                      << ",\"first_differing_byte_offset\":"
                      << difference.byteOffset
                      << ",\"first_differing_page_byte_offset\":"
                      << difference.byteOffset % GS_VRAM_PAGE_SIZE
                      << ",\"expected_byte\":"
                      << static_cast<uint32_t>(difference.expected)
                      << ",\"actual_byte\":"
                      << static_cast<uint32_t>(difference.actual);
        }
    }

    if (backendStats)
    {
        std::cout << ",\"backend_counters\":";
        writeBackendCounters(
            std::cout, backendCounterResult.counters);
        if (rendererStatus.mode != GsRendererMode::Software)
        {
            std::cout << ",\"vulkan_capabilities\":{";
            writeVulkanCapabilityFields(
                std::cout, rendererStatus.capabilities);
            std::cout << "},\"vulkan_service_statistics\":";
            writeVulkanServiceStatistics(
                std::cout,
                rendererStatus.serviceStatistics);
            std::cout << ",\"vulkan_backend_statistics\":";
            writeVulkanRasterBackendStatistics(
                std::cout,
                rendererStatus.backendStatistics);
        }
    }

    std::cout << "}\n";
    return !expectedVram.empty() && !difference.matches ? 1 : 0;
}
