#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ps2x::performance {
inline constexpr uint32_t kTelemetrySchemaVersion = 2u;
inline constexpr uint64_t kEeCyclesPerSecond = 294'912'000u;

// Matches the semantic order of the runtime's GsFlushReason without exposing
// renderer headers to the standalone telemetry client.
enum class TelemetryFlushReason : size_t {
  Explicit,
  Transfer,
  CpuReadback,
  FeedbackSnapshot,
  ClutHazard,
  Finish,
  PresentationLatch,
  DebuggerObservation,
  BackendSwitch,
  Reset,
  SaveLoad,
  Shutdown,
  QueueBackpressure,
  ResourceHazard,
  PipelineChange,
  Count,
};

inline constexpr size_t kTelemetryFlushReasonCount =
    static_cast<size_t>(TelemetryFlushReason::Count);
using FlushReasonCounters = std::array<uint64_t, kTelemetryFlushReasonCount>;
using FlushReasonRates = std::array<double, kTelemetryFlushReasonCount>;

[[nodiscard]] constexpr size_t
flushReasonIndex(TelemetryFlushReason reason) noexcept {
  return static_cast<size_t>(reason);
}

struct GuestTelemetry {
  uint64_t eeTick = 0u;
  uint64_t vsyncFields = 0u;
  uint64_t presentations = 0u;
  uint64_t vsyncPeriodCycles = 0u;
};

struct GsTelemetry {
  bool asyncEnabled = false;
  uint64_t pendingCompletions = 0u;
  uint64_t pendingHighWater = 0u;
  uint64_t fieldsSubmitted = 0u;
  uint64_t fieldsCompleted = 0u;
  uint64_t fieldLeadBlockCount = 0u;
  uint64_t fieldLeadBlockedNanoseconds = 0u;

  uint64_t ownerQueueCapacity = 0u;
  uint64_t ownerQueueDepth = 0u;
  uint64_t ownerQueueHighWater = 0u;
  uint64_t ownerQueuedPayloadBytes = 0u;
  uint64_t ownerPayloadCapacityBytes = 0u;
  uint64_t ownerWorkerActiveNanoseconds = 0u;
  uint64_t ownerWorkerIdleNanoseconds = 0u;
  uint64_t ownerProducerBlockCount = 0u;
  uint64_t ownerProducerBlockedNanoseconds = 0u;

  uint64_t routingCommands = 0u;
  uint64_t routingNoopCommands = 0u;
  uint64_t routingSoftwareCommands = 0u;
  uint64_t routingAcceleratedCommands = 0u;
  uint64_t routingFallbackCommands = 0u;
  uint64_t routingDrawPixels = 0u;
  uint64_t routingSoftwarePixels = 0u;
  uint64_t routingAcceleratedPixels = 0u;
  uint64_t routingFallbackPixels = 0u;
  uint64_t softwareRasterHostNanoseconds = 0u;
  uint64_t routingFlushes = 0u;
  uint64_t routingBackendSwitches = 0u;
  FlushReasonCounters routingFlushReasons{};

  uint64_t vulkanRoundTripsCompleted = 0u;
  uint64_t vulkanRoundTripsFailed = 0u;
  uint64_t vulkanRequestWaits = 0u;
  uint64_t vulkanRequestWaitNanoseconds = 0u;
  uint64_t vulkanQueueSubmissions = 0u;
  uint64_t vulkanShaderDispatches = 0u;
  uint64_t vulkanBytesUploaded = 0u;
  uint64_t vulkanBytesDownloaded = 0u;
  uint64_t vulkanFenceWaits = 0u;
  uint64_t vulkanFenceWaitNanoseconds = 0u;
  uint64_t vulkanCommittedGpuCommands = 0u;
  uint64_t vulkanVerificationMismatches = 0u;
  uint64_t vulkanResourceHazardDrains = 0u;
  uint64_t vulkanQueueBackpressureDrains = 0u;
  uint64_t vulkanPipelineChangeDrains = 0u;
  uint64_t vulkanCpuAccessPreparations = 0u;
  uint64_t vulkanCpuToGpuOperations = 0u;
  uint64_t vulkanCpuToGpuPages = 0u;
  uint64_t vulkanGpuToCpuOperations = 0u;
  uint64_t vulkanGpuToCpuPages = 0u;
  FlushReasonCounters vulkanDownloadOperationsByReason{};
  FlushReasonCounters vulkanDownloadedPagesByReason{};
};

struct Vu1Telemetry {
  bool asyncEnabled = false;
  bool pendingSlice = false;
  bool deferredSlice = false;
  uint64_t slicesSubmitted = 0u;
  uint64_t slicesPublished = 0u;
  uint64_t resultsReadyAtEvent = 0u;
  uint64_t resultsLateAtEvent = 0u;
  uint64_t eventWaitCount = 0u;
  uint64_t eventWaitNanoseconds = 0u;
  uint64_t maximumEventWaitNanoseconds = 0u;
  uint64_t hazardBarrierCount = 0u;
  uint64_t hazardWaitNanoseconds = 0u;
  uint64_t budgetFallbackCount = 0u;
  uint64_t budgetFallbackWaitNanoseconds = 0u;

  uint64_t ownerQueueCapacity = 0u;
  uint64_t ownerQueueDepth = 0u;
  uint64_t ownerQueueHighWater = 0u;
  uint64_t ownerQueuedPayloadBytes = 0u;
  uint64_t ownerPayloadCapacityBytes = 0u;
  uint64_t ownerWorkerActiveNanoseconds = 0u;
  uint64_t ownerWorkerIdleNanoseconds = 0u;
  uint64_t ownerProducerBlockCount = 0u;
  uint64_t ownerProducerBlockedNanoseconds = 0u;
  uint64_t ownerResultWaitCount = 0u;
  uint64_t ownerResultWaitNanoseconds = 0u;
  uint64_t ownerWorkNotificationCount = 0u;
  uint64_t ownerWorkerWakeCount = 0u;
};

struct TelemetrySnapshot {
  uint32_t schemaVersion = kTelemetrySchemaVersion;
  uint64_t sequence = 0u;
  uint64_t monotonicNanoseconds = 0u;
  int64_t processId = 0;
  bool desiredVisible = false;
  bool detailed = false;
  std::string gsExecutionMode;
  std::string vu1ExecutionMode;
  std::string rendererMode;
  std::string rendererDiagnostic;
  GuestTelemetry guest{};
  GsTelemetry gs{};
  Vu1Telemetry vu1{};
};

struct TelemetryRates {
  bool valid = false;
  bool routingValid = false;
  bool presentationValid = false;
  bool batchingValid = false;
  bool coherencyValid = false;
  double elapsedSeconds = 0.0;
  double framesPerSecond = 0.0;
  double speedPercent = 0.0;
  double gsOwnerBusyPercent = 0.0;
  double vu1OwnerBusyPercent = 0.0;
  double gsProducerBlockedPercent = 0.0;
  double vu1ProducerBlockedPercent = 0.0;
  double vu1EventWaitPercent = 0.0;
  double vu1HazardWaitPercent = 0.0;
  double softwareRasterCorePercent = 0.0;
  double softwareCommandPercent = 0.0;
  double acceleratedCommandPercent = 0.0;
  double noopCommandPercent = 0.0;
  double softwarePixelPercent = 0.0;
  double acceleratedPixelPercent = 0.0;
  double fallbackCommandsPerSecond = 0.0;
  double softwareCommandsPerSecond = 0.0;
  double acceleratedCommandsPerSecond = 0.0;
  double noopCommandsPerSecond = 0.0;
  double softwarePixelsPerSecond = 0.0;
  double acceleratedPixelsPerSecond = 0.0;
  double fallbackCommandsPerPresentation = 0.0;
  double softwareCommandsPerPresentation = 0.0;
  double acceleratedCommandsPerPresentation = 0.0;
  double noopCommandsPerPresentation = 0.0;
  double backendSwitchesPerSecond = 0.0;
  double backendSwitchesPerPresentation = 0.0;
  double routingFlushesPerSecond = 0.0;
  double routingFlushesPerPresentation = 0.0;
  FlushReasonRates routingFlushReasonsPerSecond{};
  FlushReasonRates routingFlushReasonsPerPresentation{};
  double vulkanSubmissionsPerSecond = 0.0;
  double vulkanDispatchesPerSecond = 0.0;
  double vulkanSubmissionsPerPresentation = 0.0;
  double vulkanDispatchesPerPresentation = 0.0;
  double vulkanRequestWaitPercent = 0.0;
  double vulkanFenceWaitPercent = 0.0;
  double vulkanUploadBytesPerSecond = 0.0;
  double vulkanDownloadBytesPerSecond = 0.0;
  double vulkanUploadBytesPerPresentation = 0.0;
  double vulkanDownloadBytesPerPresentation = 0.0;
  double vulkanResourceHazardDrainsPerPresentation = 0.0;
  double vulkanQueueBackpressureDrainsPerPresentation = 0.0;
  double vulkanPipelineChangeDrainsPerPresentation = 0.0;
  double vulkanCpuAccessPreparationsPerPresentation = 0.0;
  double vulkanCpuToGpuOperationsPerPresentation = 0.0;
  double vulkanCpuToGpuPagesPerPresentation = 0.0;
  double vulkanGpuToCpuOperationsPerPresentation = 0.0;
  double vulkanGpuToCpuPagesPerPresentation = 0.0;
  FlushReasonRates vulkanDownloadOperationsByReasonPerPresentation{};
  FlushReasonRates vulkanDownloadedPagesByReasonPerPresentation{};
};

[[nodiscard]] TelemetryRates
calculateTelemetryRates(const TelemetrySnapshot &previous,
                        const TelemetrySnapshot &current) noexcept;

[[nodiscard]] std::string makeSnapshotRequest(uint64_t requestId);
[[nodiscard]] bool parseSnapshotRequest(std::string_view json,
                                        uint64_t &requestId,
                                        std::string *error = nullptr);
[[nodiscard]] std::string
makeSnapshotResponse(uint64_t requestId, const TelemetrySnapshot &snapshot);
[[nodiscard]] std::string makeErrorResponse(uint64_t requestId, int code,
                                            std::string_view message);
[[nodiscard]] bool parseSnapshotResponse(std::string_view json,
                                         uint64_t &requestId,
                                         TelemetrySnapshot &snapshot,
                                         std::string *error = nullptr);
} // namespace ps2x::performance
