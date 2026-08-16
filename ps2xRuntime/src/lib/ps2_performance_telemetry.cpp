#include "runtime/ps2_performance_telemetry.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace ps2x::performance {
namespace {
using JsonWriter = rapidjson::Writer<rapidjson::StringBuffer>;

void setError(std::string *error, std::string message) {
  if (error)
    *error = std::move(message);
}

void writeUint64(JsonWriter &writer, const char *name, uint64_t value) {
  writer.Key(name);
  writer.Uint64(value);
}

void writeBool(JsonWriter &writer, const char *name, bool value) {
  writer.Key(name);
  writer.Bool(value);
}

void writeString(JsonWriter &writer, const char *name,
                 const std::string &value) {
  writer.Key(name);
  writer.String(value.data(), static_cast<rapidjson::SizeType>(value.size()));
}

bool readObject(const rapidjson::Value &parent, const char *name,
                const rapidjson::Value *&output, std::string *error) {
  const auto it = parent.FindMember(name);
  if (it == parent.MemberEnd() || !it->value.IsObject()) {
    setError(error, std::string("missing object: ") + name);
    return false;
  }
  output = &it->value;
  return true;
}

bool readUint64(const rapidjson::Value &parent, const char *name,
                uint64_t &output, std::string *error) {
  const auto it = parent.FindMember(name);
  if (it == parent.MemberEnd() || !it->value.IsUint64()) {
    setError(error, std::string("missing uint64: ") + name);
    return false;
  }
  output = it->value.GetUint64();
  return true;
}

bool readInt64(const rapidjson::Value &parent, const char *name,
               int64_t &output, std::string *error) {
  const auto it = parent.FindMember(name);
  if (it == parent.MemberEnd() || !it->value.IsInt64()) {
    setError(error, std::string("missing int64: ") + name);
    return false;
  }
  output = it->value.GetInt64();
  return true;
}

bool readBool(const rapidjson::Value &parent, const char *name, bool &output,
              std::string *error) {
  const auto it = parent.FindMember(name);
  if (it == parent.MemberEnd() || !it->value.IsBool()) {
    setError(error, std::string("missing bool: ") + name);
    return false;
  }
  output = it->value.GetBool();
  return true;
}

bool readString(const rapidjson::Value &parent, const char *name,
                std::string &output, std::string *error) {
  const auto it = parent.FindMember(name);
  if (it == parent.MemberEnd() || !it->value.IsString()) {
    setError(error, std::string("missing string: ") + name);
    return false;
  }
  output.assign(it->value.GetString(), it->value.GetStringLength());
  return true;
}

bool counterDelta(uint64_t previous, uint64_t current,
                  uint64_t &delta) noexcept {
  if (current < previous)
    return false;
  delta = current - previous;
  return true;
}

double percentOfElapsed(uint64_t nanoseconds, double seconds) noexcept {
  if (seconds <= 0.0)
    return 0.0;
  return static_cast<double>(nanoseconds) / (seconds * 1'000'000'000.0) * 100.0;
}

double busyPercent(uint64_t previousActive, uint64_t currentActive,
                   uint64_t previousIdle, uint64_t currentIdle) noexcept {
  uint64_t active = 0u;
  uint64_t idle = 0u;
  if (!counterDelta(previousActive, currentActive, active) ||
      !counterDelta(previousIdle, currentIdle, idle) || active + idle == 0u) {
    return 0.0;
  }
  return static_cast<double>(active) * 100.0 /
         static_cast<double>(active + idle);
}

void writeGuest(JsonWriter &writer, const GuestTelemetry &guest) {
  writer.Key("guest");
  writer.StartObject();
  writeUint64(writer, "ee_tick", guest.eeTick);
  writeUint64(writer, "vsync_fields", guest.vsyncFields);
  writeUint64(writer, "presentations", guest.presentations);
  writeUint64(writer, "vsync_period_cycles", guest.vsyncPeriodCycles);
  writer.EndObject();
}

void writeGs(JsonWriter &writer, const GsTelemetry &gs) {
  writer.Key("gs");
  writer.StartObject();
  writeBool(writer, "async_enabled", gs.asyncEnabled);
  writeUint64(writer, "pending_completions", gs.pendingCompletions);
  writeUint64(writer, "pending_high_water", gs.pendingHighWater);
  writeUint64(writer, "fields_submitted", gs.fieldsSubmitted);
  writeUint64(writer, "fields_completed", gs.fieldsCompleted);
  writeUint64(writer, "field_lead_block_count", gs.fieldLeadBlockCount);
  writeUint64(writer, "field_lead_blocked_ns", gs.fieldLeadBlockedNanoseconds);

  writer.Key("owner");
  writer.StartObject();
  writeUint64(writer, "queue_capacity", gs.ownerQueueCapacity);
  writeUint64(writer, "queue_depth", gs.ownerQueueDepth);
  writeUint64(writer, "queue_high_water", gs.ownerQueueHighWater);
  writeUint64(writer, "queued_payload_bytes", gs.ownerQueuedPayloadBytes);
  writeUint64(writer, "payload_capacity_bytes", gs.ownerPayloadCapacityBytes);
  writeUint64(writer, "worker_active_ns", gs.ownerWorkerActiveNanoseconds);
  writeUint64(writer, "worker_idle_ns", gs.ownerWorkerIdleNanoseconds);
  writeUint64(writer, "producer_block_count", gs.ownerProducerBlockCount);
  writeUint64(writer, "producer_blocked_ns",
              gs.ownerProducerBlockedNanoseconds);
  writer.EndObject();

  writer.Key("routing");
  writer.StartObject();
  writeUint64(writer, "commands", gs.routingCommands);
  writeUint64(writer, "noop_commands", gs.routingNoopCommands);
  writeUint64(writer, "software_commands", gs.routingSoftwareCommands);
  writeUint64(writer, "accelerated_commands", gs.routingAcceleratedCommands);
  writeUint64(writer, "fallback_commands", gs.routingFallbackCommands);
  writeUint64(writer, "draw_pixels", gs.routingDrawPixels);
  writeUint64(writer, "software_pixels", gs.routingSoftwarePixels);
  writeUint64(writer, "accelerated_pixels", gs.routingAcceleratedPixels);
  writeUint64(writer, "fallback_pixels", gs.routingFallbackPixels);
  writeUint64(writer, "software_raster_host_ns",
              gs.softwareRasterHostNanoseconds);
  writer.EndObject();

  writer.Key("vulkan");
  writer.StartObject();
  writeUint64(writer, "round_trips_completed", gs.vulkanRoundTripsCompleted);
  writeUint64(writer, "round_trips_failed", gs.vulkanRoundTripsFailed);
  writeUint64(writer, "request_waits", gs.vulkanRequestWaits);
  writeUint64(writer, "request_wait_ns", gs.vulkanRequestWaitNanoseconds);
  writeUint64(writer, "queue_submissions", gs.vulkanQueueSubmissions);
  writeUint64(writer, "shader_dispatches", gs.vulkanShaderDispatches);
  writeUint64(writer, "bytes_uploaded", gs.vulkanBytesUploaded);
  writeUint64(writer, "bytes_downloaded", gs.vulkanBytesDownloaded);
  writeUint64(writer, "fence_waits", gs.vulkanFenceWaits);
  writeUint64(writer, "fence_wait_ns", gs.vulkanFenceWaitNanoseconds);
  writeUint64(writer, "committed_gpu_commands", gs.vulkanCommittedGpuCommands);
  writeUint64(writer, "verification_mismatches",
              gs.vulkanVerificationMismatches);
  writer.EndObject();
  writer.EndObject();
}

void writeVu1(JsonWriter &writer, const Vu1Telemetry &vu1) {
  writer.Key("vu1");
  writer.StartObject();
  writeBool(writer, "async_enabled", vu1.asyncEnabled);
  writeBool(writer, "pending_slice", vu1.pendingSlice);
  writeBool(writer, "deferred_slice", vu1.deferredSlice);
  writeUint64(writer, "slices_submitted", vu1.slicesSubmitted);
  writeUint64(writer, "slices_published", vu1.slicesPublished);
  writeUint64(writer, "results_ready_at_event", vu1.resultsReadyAtEvent);
  writeUint64(writer, "results_late_at_event", vu1.resultsLateAtEvent);
  writeUint64(writer, "event_wait_count", vu1.eventWaitCount);
  writeUint64(writer, "event_wait_ns", vu1.eventWaitNanoseconds);
  writeUint64(writer, "maximum_event_wait_ns", vu1.maximumEventWaitNanoseconds);
  writeUint64(writer, "hazard_barrier_count", vu1.hazardBarrierCount);
  writeUint64(writer, "hazard_wait_ns", vu1.hazardWaitNanoseconds);
  writeUint64(writer, "budget_fallback_count", vu1.budgetFallbackCount);
  writeUint64(writer, "budget_fallback_wait_ns",
              vu1.budgetFallbackWaitNanoseconds);

  writer.Key("owner");
  writer.StartObject();
  writeUint64(writer, "queue_capacity", vu1.ownerQueueCapacity);
  writeUint64(writer, "queue_depth", vu1.ownerQueueDepth);
  writeUint64(writer, "queue_high_water", vu1.ownerQueueHighWater);
  writeUint64(writer, "queued_payload_bytes", vu1.ownerQueuedPayloadBytes);
  writeUint64(writer, "payload_capacity_bytes", vu1.ownerPayloadCapacityBytes);
  writeUint64(writer, "worker_active_ns", vu1.ownerWorkerActiveNanoseconds);
  writeUint64(writer, "worker_idle_ns", vu1.ownerWorkerIdleNanoseconds);
  writeUint64(writer, "producer_block_count", vu1.ownerProducerBlockCount);
  writeUint64(writer, "producer_blocked_ns",
              vu1.ownerProducerBlockedNanoseconds);
  writeUint64(writer, "result_wait_count", vu1.ownerResultWaitCount);
  writeUint64(writer, "result_wait_ns", vu1.ownerResultWaitNanoseconds);
  writeUint64(writer, "work_notification_count",
              vu1.ownerWorkNotificationCount);
  writeUint64(writer, "worker_wake_count", vu1.ownerWorkerWakeCount);
  writer.EndObject();
  writer.EndObject();
}

bool readGuest(const rapidjson::Value &root, GuestTelemetry &guest,
               std::string *error) {
  const rapidjson::Value *value = nullptr;
  return readObject(root, "guest", value, error) &&
         readUint64(*value, "ee_tick", guest.eeTick, error) &&
         readUint64(*value, "vsync_fields", guest.vsyncFields, error) &&
         readUint64(*value, "presentations", guest.presentations, error) &&
         readUint64(*value, "vsync_period_cycles", guest.vsyncPeriodCycles,
                    error);
}

bool readGs(const rapidjson::Value &root, GsTelemetry &gs, std::string *error) {
  const rapidjson::Value *value = nullptr;
  const rapidjson::Value *owner = nullptr;
  const rapidjson::Value *routing = nullptr;
  const rapidjson::Value *vulkan = nullptr;
  return readObject(root, "gs", value, error) &&
         readBool(*value, "async_enabled", gs.asyncEnabled, error) &&
         readUint64(*value, "pending_completions", gs.pendingCompletions,
                    error) &&
         readUint64(*value, "pending_high_water", gs.pendingHighWater, error) &&
         readUint64(*value, "fields_submitted", gs.fieldsSubmitted, error) &&
         readUint64(*value, "fields_completed", gs.fieldsCompleted, error) &&
         readUint64(*value, "field_lead_block_count", gs.fieldLeadBlockCount,
                    error) &&
         readUint64(*value, "field_lead_blocked_ns",
                    gs.fieldLeadBlockedNanoseconds, error) &&
         readObject(*value, "owner", owner, error) &&
         readUint64(*owner, "queue_capacity", gs.ownerQueueCapacity, error) &&
         readUint64(*owner, "queue_depth", gs.ownerQueueDepth, error) &&
         readUint64(*owner, "queue_high_water", gs.ownerQueueHighWater,
                    error) &&
         readUint64(*owner, "queued_payload_bytes", gs.ownerQueuedPayloadBytes,
                    error) &&
         readUint64(*owner, "payload_capacity_bytes",
                    gs.ownerPayloadCapacityBytes, error) &&
         readUint64(*owner, "worker_active_ns", gs.ownerWorkerActiveNanoseconds,
                    error) &&
         readUint64(*owner, "worker_idle_ns", gs.ownerWorkerIdleNanoseconds,
                    error) &&
         readUint64(*owner, "producer_block_count", gs.ownerProducerBlockCount,
                    error) &&
         readUint64(*owner, "producer_blocked_ns",
                    gs.ownerProducerBlockedNanoseconds, error) &&
         readObject(*value, "routing", routing, error) &&
         readUint64(*routing, "commands", gs.routingCommands, error) &&
         readUint64(*routing, "noop_commands", gs.routingNoopCommands, error) &&
         readUint64(*routing, "software_commands", gs.routingSoftwareCommands,
                    error) &&
         readUint64(*routing, "accelerated_commands",
                    gs.routingAcceleratedCommands, error) &&
         readUint64(*routing, "fallback_commands", gs.routingFallbackCommands,
                    error) &&
         readUint64(*routing, "draw_pixels", gs.routingDrawPixels, error) &&
         readUint64(*routing, "software_pixels", gs.routingSoftwarePixels,
                    error) &&
         readUint64(*routing, "accelerated_pixels", gs.routingAcceleratedPixels,
                    error) &&
         readUint64(*routing, "fallback_pixels", gs.routingFallbackPixels,
                    error) &&
         readUint64(*routing, "software_raster_host_ns",
                    gs.softwareRasterHostNanoseconds, error) &&
         readObject(*value, "vulkan", vulkan, error) &&
         readUint64(*vulkan, "round_trips_completed",
                    gs.vulkanRoundTripsCompleted, error) &&
         readUint64(*vulkan, "round_trips_failed", gs.vulkanRoundTripsFailed,
                    error) &&
         readUint64(*vulkan, "request_waits", gs.vulkanRequestWaits, error) &&
         readUint64(*vulkan, "request_wait_ns", gs.vulkanRequestWaitNanoseconds,
                    error) &&
         readUint64(*vulkan, "queue_submissions", gs.vulkanQueueSubmissions,
                    error) &&
         readUint64(*vulkan, "shader_dispatches", gs.vulkanShaderDispatches,
                    error) &&
         readUint64(*vulkan, "bytes_uploaded", gs.vulkanBytesUploaded, error) &&
         readUint64(*vulkan, "bytes_downloaded", gs.vulkanBytesDownloaded,
                    error) &&
         readUint64(*vulkan, "fence_waits", gs.vulkanFenceWaits, error) &&
         readUint64(*vulkan, "fence_wait_ns", gs.vulkanFenceWaitNanoseconds,
                    error) &&
         readUint64(*vulkan, "committed_gpu_commands",
                    gs.vulkanCommittedGpuCommands, error) &&
         readUint64(*vulkan, "verification_mismatches",
                    gs.vulkanVerificationMismatches, error);
}

bool readVu1(const rapidjson::Value &root, Vu1Telemetry &vu1,
             std::string *error) {
  const rapidjson::Value *value = nullptr;
  const rapidjson::Value *owner = nullptr;
  return readObject(root, "vu1", value, error) &&
         readBool(*value, "async_enabled", vu1.asyncEnabled, error) &&
         readBool(*value, "pending_slice", vu1.pendingSlice, error) &&
         readBool(*value, "deferred_slice", vu1.deferredSlice, error) &&
         readUint64(*value, "slices_submitted", vu1.slicesSubmitted, error) &&
         readUint64(*value, "slices_published", vu1.slicesPublished, error) &&
         readUint64(*value, "results_ready_at_event", vu1.resultsReadyAtEvent,
                    error) &&
         readUint64(*value, "results_late_at_event", vu1.resultsLateAtEvent,
                    error) &&
         readUint64(*value, "event_wait_count", vu1.eventWaitCount, error) &&
         readUint64(*value, "event_wait_ns", vu1.eventWaitNanoseconds, error) &&
         readUint64(*value, "maximum_event_wait_ns",
                    vu1.maximumEventWaitNanoseconds, error) &&
         readUint64(*value, "hazard_barrier_count", vu1.hazardBarrierCount,
                    error) &&
         readUint64(*value, "hazard_wait_ns", vu1.hazardWaitNanoseconds,
                    error) &&
         readUint64(*value, "budget_fallback_count", vu1.budgetFallbackCount,
                    error) &&
         readUint64(*value, "budget_fallback_wait_ns",
                    vu1.budgetFallbackWaitNanoseconds, error) &&
         readObject(*value, "owner", owner, error) &&
         readUint64(*owner, "queue_capacity", vu1.ownerQueueCapacity, error) &&
         readUint64(*owner, "queue_depth", vu1.ownerQueueDepth, error) &&
         readUint64(*owner, "queue_high_water", vu1.ownerQueueHighWater,
                    error) &&
         readUint64(*owner, "queued_payload_bytes", vu1.ownerQueuedPayloadBytes,
                    error) &&
         readUint64(*owner, "payload_capacity_bytes",
                    vu1.ownerPayloadCapacityBytes, error) &&
         readUint64(*owner, "worker_active_ns",
                    vu1.ownerWorkerActiveNanoseconds, error) &&
         readUint64(*owner, "worker_idle_ns", vu1.ownerWorkerIdleNanoseconds,
                    error) &&
         readUint64(*owner, "producer_block_count", vu1.ownerProducerBlockCount,
                    error) &&
         readUint64(*owner, "producer_blocked_ns",
                    vu1.ownerProducerBlockedNanoseconds, error) &&
         readUint64(*owner, "result_wait_count", vu1.ownerResultWaitCount,
                    error) &&
         readUint64(*owner, "result_wait_ns", vu1.ownerResultWaitNanoseconds,
                    error) &&
         readUint64(*owner, "work_notification_count",
                    vu1.ownerWorkNotificationCount, error) &&
         readUint64(*owner, "worker_wake_count", vu1.ownerWorkerWakeCount,
                    error);
}
} // namespace

TelemetryRates
calculateTelemetryRates(const TelemetrySnapshot &previous,
                        const TelemetrySnapshot &current) noexcept {
  TelemetryRates rates{};
  if (current.monotonicNanoseconds <= previous.monotonicNanoseconds)
    return rates;

  const uint64_t elapsedNanoseconds =
      current.monotonicNanoseconds - previous.monotonicNanoseconds;
  rates.elapsedSeconds =
      static_cast<double>(elapsedNanoseconds) / 1'000'000'000.0;
  rates.valid = rates.elapsedSeconds > 0.0;
  if (!rates.valid)
    return rates;

  uint64_t delta = 0u;
  if (counterDelta(previous.guest.presentations, current.guest.presentations,
                   delta)) {
    rates.framesPerSecond = static_cast<double>(delta) / rates.elapsedSeconds;
  }
  if (current.guest.vsyncPeriodCycles != 0u &&
      current.guest.vsyncPeriodCycles == previous.guest.vsyncPeriodCycles &&
      counterDelta(previous.guest.vsyncFields, current.guest.vsyncFields,
                   delta)) {
    const double expectedFields =
        rates.elapsedSeconds * static_cast<double>(kEeCyclesPerSecond) /
        static_cast<double>(current.guest.vsyncPeriodCycles);
    if (expectedFields > 0.0) {
      rates.speedPercent = static_cast<double>(delta) * 100.0 / expectedFields;
    }
  }

  rates.gsOwnerBusyPercent =
      busyPercent(previous.gs.ownerWorkerActiveNanoseconds,
                  current.gs.ownerWorkerActiveNanoseconds,
                  previous.gs.ownerWorkerIdleNanoseconds,
                  current.gs.ownerWorkerIdleNanoseconds);
  rates.vu1OwnerBusyPercent =
      busyPercent(previous.vu1.ownerWorkerActiveNanoseconds,
                  current.vu1.ownerWorkerActiveNanoseconds,
                  previous.vu1.ownerWorkerIdleNanoseconds,
                  current.vu1.ownerWorkerIdleNanoseconds);

  if (counterDelta(previous.gs.ownerProducerBlockedNanoseconds,
                   current.gs.ownerProducerBlockedNanoseconds, delta)) {
    rates.gsProducerBlockedPercent =
        percentOfElapsed(delta, rates.elapsedSeconds);
  }
  if (counterDelta(previous.vu1.ownerProducerBlockedNanoseconds,
                   current.vu1.ownerProducerBlockedNanoseconds, delta)) {
    rates.vu1ProducerBlockedPercent =
        percentOfElapsed(delta, rates.elapsedSeconds);
  }
  if (counterDelta(previous.vu1.eventWaitNanoseconds,
                   current.vu1.eventWaitNanoseconds, delta)) {
    rates.vu1EventWaitPercent = percentOfElapsed(delta, rates.elapsedSeconds);
  }
  if (counterDelta(previous.vu1.hazardWaitNanoseconds,
                   current.vu1.hazardWaitNanoseconds, delta)) {
    rates.vu1HazardWaitPercent = percentOfElapsed(delta, rates.elapsedSeconds);
  }
  if (counterDelta(previous.gs.softwareRasterHostNanoseconds,
                   current.gs.softwareRasterHostNanoseconds, delta)) {
    rates.softwareRasterCorePercent =
        percentOfElapsed(delta, rates.elapsedSeconds);
  }

  uint64_t softwareCommands = 0u;
  uint64_t acceleratedCommands = 0u;
  uint64_t noopCommands = 0u;
  uint64_t fallbackCommands = 0u;
  uint64_t softwarePixels = 0u;
  uint64_t acceleratedPixels = 0u;
  rates.routingValid =
      counterDelta(previous.gs.routingSoftwareCommands,
                   current.gs.routingSoftwareCommands, softwareCommands) &&
      counterDelta(previous.gs.routingAcceleratedCommands,
                   current.gs.routingAcceleratedCommands,
                   acceleratedCommands) &&
      counterDelta(previous.gs.routingNoopCommands,
                   current.gs.routingNoopCommands, noopCommands) &&
      counterDelta(previous.gs.routingFallbackCommands,
                   current.gs.routingFallbackCommands, fallbackCommands) &&
      counterDelta(previous.gs.routingSoftwarePixels,
                   current.gs.routingSoftwarePixels, softwarePixels) &&
      counterDelta(previous.gs.routingAcceleratedPixels,
                   current.gs.routingAcceleratedPixels, acceleratedPixels);
  if (rates.routingValid) {
    const uint64_t routedCommands =
        softwareCommands + acceleratedCommands + noopCommands;
    const uint64_t routedPixels = softwarePixels + acceleratedPixels;
    if (routedCommands != 0u) {
      rates.softwareCommandPercent = static_cast<double>(softwareCommands) *
                                     100.0 /
                                     static_cast<double>(routedCommands);
      rates.acceleratedCommandPercent =
          static_cast<double>(acceleratedCommands) * 100.0 /
          static_cast<double>(routedCommands);
      rates.noopCommandPercent = static_cast<double>(noopCommands) * 100.0 /
                                 static_cast<double>(routedCommands);
    }
    if (routedPixels != 0u) {
      rates.softwarePixelPercent = static_cast<double>(softwarePixels) * 100.0 /
                                   static_cast<double>(routedPixels);
      rates.acceleratedPixelPercent = static_cast<double>(acceleratedPixels) *
                                      100.0 / static_cast<double>(routedPixels);
    }
    rates.fallbackCommandsPerSecond =
        static_cast<double>(fallbackCommands) / rates.elapsedSeconds;
    rates.softwareCommandsPerSecond =
        static_cast<double>(softwareCommands) / rates.elapsedSeconds;
    rates.acceleratedCommandsPerSecond =
        static_cast<double>(acceleratedCommands) / rates.elapsedSeconds;
    rates.softwarePixelsPerSecond =
        static_cast<double>(softwarePixels) / rates.elapsedSeconds;
    rates.acceleratedPixelsPerSecond =
        static_cast<double>(acceleratedPixels) / rates.elapsedSeconds;
  }

  if (counterDelta(previous.gs.vulkanQueueSubmissions,
                   current.gs.vulkanQueueSubmissions, delta)) {
    rates.vulkanSubmissionsPerSecond =
        static_cast<double>(delta) / rates.elapsedSeconds;
  }
  if (counterDelta(previous.gs.vulkanShaderDispatches,
                   current.gs.vulkanShaderDispatches, delta)) {
    rates.vulkanDispatchesPerSecond =
        static_cast<double>(delta) / rates.elapsedSeconds;
  }
  if (counterDelta(previous.gs.vulkanRequestWaitNanoseconds,
                   current.gs.vulkanRequestWaitNanoseconds, delta)) {
    rates.vulkanRequestWaitPercent =
        percentOfElapsed(delta, rates.elapsedSeconds);
  }
  if (counterDelta(previous.gs.vulkanFenceWaitNanoseconds,
                   current.gs.vulkanFenceWaitNanoseconds, delta)) {
    rates.vulkanFenceWaitPercent =
        percentOfElapsed(delta, rates.elapsedSeconds);
  }
  if (counterDelta(previous.gs.vulkanBytesUploaded,
                   current.gs.vulkanBytesUploaded, delta)) {
    rates.vulkanUploadBytesPerSecond =
        static_cast<double>(delta) / rates.elapsedSeconds;
  }
  if (counterDelta(previous.gs.vulkanBytesDownloaded,
                   current.gs.vulkanBytesDownloaded, delta)) {
    rates.vulkanDownloadBytesPerSecond =
        static_cast<double>(delta) / rates.elapsedSeconds;
  }

  return rates;
}

std::string makeSnapshotRequest(uint64_t requestId) {
  rapidjson::StringBuffer buffer;
  JsonWriter writer(buffer);
  writer.StartObject();
  writer.Key("jsonrpc");
  writer.String("2.0");
  writeUint64(writer, "id", requestId);
  writer.Key("method");
  writer.String("performance.snapshot");
  writer.EndObject();
  std::string output(buffer.GetString(), buffer.GetSize());
  output.push_back('\n');
  return output;
}

bool parseSnapshotRequest(std::string_view json, uint64_t &requestId,
                          std::string *error) {
  if (error)
    error->clear();
  rapidjson::Document document;
  document.Parse(json.data(), json.size());
  if (document.HasParseError() || !document.IsObject()) {
    setError(error, "invalid JSON-RPC request");
    return false;
  }
  const auto version = document.FindMember("jsonrpc");
  const auto method = document.FindMember("method");
  if (version == document.MemberEnd() || !version->value.IsString() ||
      std::string_view(version->value.GetString(),
                       version->value.GetStringLength()) != "2.0" ||
      method == document.MemberEnd() || !method->value.IsString() ||
      std::string_view(method->value.GetString(),
                       method->value.GetStringLength()) !=
          "performance.snapshot" ||
      !readUint64(document, "id", requestId, error)) {
    if (error && error->empty())
      *error = "unsupported JSON-RPC request";
    return false;
  }
  return true;
}

std::string makeSnapshotResponse(uint64_t requestId,
                                 const TelemetrySnapshot &snapshot) {
  rapidjson::StringBuffer buffer;
  JsonWriter writer(buffer);
  writer.StartObject();
  writer.Key("jsonrpc");
  writer.String("2.0");
  writeUint64(writer, "id", requestId);
  writer.Key("result");
  writer.StartObject();
  writer.Key("schema");
  writer.Uint(snapshot.schemaVersion);
  writeUint64(writer, "sequence", snapshot.sequence);
  writeUint64(writer, "monotonic_ns", snapshot.monotonicNanoseconds);
  writer.Key("pid");
  writer.Int64(snapshot.processId);
  writeBool(writer, "desired_visible", snapshot.desiredVisible);
  writeBool(writer, "detailed", snapshot.detailed);
  writer.Key("modes");
  writer.StartObject();
  writeString(writer, "gs", snapshot.gsExecutionMode);
  writeString(writer, "vu1", snapshot.vu1ExecutionMode);
  writeString(writer, "renderer", snapshot.rendererMode);
  writeString(writer, "renderer_diagnostic", snapshot.rendererDiagnostic);
  writer.EndObject();
  writeGuest(writer, snapshot.guest);
  writeGs(writer, snapshot.gs);
  writeVu1(writer, snapshot.vu1);
  writer.EndObject();
  writer.EndObject();
  std::string output(buffer.GetString(), buffer.GetSize());
  output.push_back('\n');
  return output;
}

std::string makeErrorResponse(uint64_t requestId, int code,
                              std::string_view message) {
  rapidjson::StringBuffer buffer;
  JsonWriter writer(buffer);
  writer.StartObject();
  writer.Key("jsonrpc");
  writer.String("2.0");
  writeUint64(writer, "id", requestId);
  writer.Key("error");
  writer.StartObject();
  writer.Key("code");
  writer.Int(code);
  writer.Key("message");
  writer.String(message.data(),
                static_cast<rapidjson::SizeType>(message.size()));
  writer.EndObject();
  writer.EndObject();
  std::string output(buffer.GetString(), buffer.GetSize());
  output.push_back('\n');
  return output;
}

bool parseSnapshotResponse(std::string_view json, uint64_t &requestId,
                           TelemetrySnapshot &snapshot, std::string *error) {
  if (error)
    error->clear();
  rapidjson::Document document;
  document.Parse(json.data(), json.size());
  if (document.HasParseError() || !document.IsObject()) {
    setError(error, "invalid JSON-RPC response");
    return false;
  }
  if (!readUint64(document, "id", requestId, error))
    return false;
  const auto remoteError = document.FindMember("error");
  if (remoteError != document.MemberEnd()) {
    if (remoteError->value.IsObject()) {
      const auto message = remoteError->value.FindMember("message");
      if (message != remoteError->value.MemberEnd() &&
          message->value.IsString()) {
        setError(error, std::string(message->value.GetString(),
                                    message->value.GetStringLength()));
        return false;
      }
    }
    setError(error, "remote telemetry error");
    return false;
  }

  const rapidjson::Value *root = nullptr;
  const rapidjson::Value *modes = nullptr;
  uint64_t schema = 0u;
  if (!readObject(document, "result", root, error) ||
      !readUint64(*root, "schema", schema, error) ||
      schema != kTelemetrySchemaVersion) {
    if (schema != 0u && schema != kTelemetrySchemaVersion) {
      setError(error, "unsupported telemetry schema " + std::to_string(schema));
    }
    return false;
  }
  snapshot = {};
  snapshot.schemaVersion = static_cast<uint32_t>(schema);
  if (!readUint64(*root, "sequence", snapshot.sequence, error) ||
      !readUint64(*root, "monotonic_ns", snapshot.monotonicNanoseconds,
                  error) ||
      !readInt64(*root, "pid", snapshot.processId, error) ||
      !readBool(*root, "desired_visible", snapshot.desiredVisible, error) ||
      !readBool(*root, "detailed", snapshot.detailed, error) ||
      !readObject(*root, "modes", modes, error) ||
      !readString(*modes, "gs", snapshot.gsExecutionMode, error) ||
      !readString(*modes, "vu1", snapshot.vu1ExecutionMode, error) ||
      !readString(*modes, "renderer", snapshot.rendererMode, error) ||
      !readString(*modes, "renderer_diagnostic", snapshot.rendererDiagnostic,
                  error) ||
      !readGuest(*root, snapshot.guest, error) ||
      !readGs(*root, snapshot.gs, error) ||
      !readVu1(*root, snapshot.vu1, error)) {
    return false;
  }
  return true;
}
} // namespace ps2x::performance
