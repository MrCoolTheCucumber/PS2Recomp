#include "runtime/ps2_performance_telemetry.h"
#include "runtime/ps2_performance_threads_linux.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <unistd.h>

namespace {
int failures = 0;

void check(bool condition, const char *message) {
  if (condition)
    return;
  std::cerr << "FAIL: " << message << '\n';
  ++failures;
}

void checkNear(double actual, double expected, const char *message) {
  check(std::abs(actual - expected) < 0.0001, message);
}

ps2x::performance::TelemetrySnapshot populatedSnapshot() {
  using namespace ps2x::performance;
  TelemetrySnapshot snapshot{};
  snapshot.sequence = 17u;
  snapshot.monotonicNanoseconds = 123'456'789u;
  snapshot.processId = 9012;
  snapshot.desiredVisible = true;
  snapshot.detailed = true;
  snapshot.gsExecutionMode = "threaded-async";
  snapshot.vu1ExecutionMode = "threaded-coarse";
  snapshot.rendererMode = "hybrid";
  snapshot.rendererDiagnostic = "exact synthetic renderer status";
  snapshot.guest = GuestTelemetry{
      .eeTick = 11u,
      .vsyncFields = 12u,
      .presentations = 13u,
      .vsyncPeriodCycles = 14u,
  };
  snapshot.gs.asyncEnabled = true;
  snapshot.gs.pendingCompletions = 21u;
  snapshot.gs.pendingHighWater = 22u;
  snapshot.gs.fieldsSubmitted = 23u;
  snapshot.gs.fieldsCompleted = 24u;
  snapshot.gs.fieldLeadBlockCount = 25u;
  snapshot.gs.fieldLeadBlockedNanoseconds = 26u;
  snapshot.gs.ownerQueueCapacity = 27u;
  snapshot.gs.ownerQueueDepth = 28u;
  snapshot.gs.ownerQueueHighWater = 29u;
  snapshot.gs.ownerQueuedPayloadBytes = 30u;
  snapshot.gs.ownerPayloadCapacityBytes = 31u;
  snapshot.gs.ownerWorkerActiveNanoseconds = 32u;
  snapshot.gs.ownerWorkerIdleNanoseconds = 33u;
  snapshot.gs.ownerProducerBlockCount = 34u;
  snapshot.gs.ownerProducerBlockedNanoseconds = 35u;
  snapshot.gs.routingCommands = 36u;
  snapshot.gs.routingNoopCommands = 37u;
  snapshot.gs.routingSoftwareCommands = 38u;
  snapshot.gs.routingAcceleratedCommands = 39u;
  snapshot.gs.routingFallbackCommands = 40u;
  snapshot.gs.routingDrawPixels = 41u;
  snapshot.gs.routingSoftwarePixels = 42u;
  snapshot.gs.routingAcceleratedPixels = 43u;
  snapshot.gs.routingFallbackPixels = 44u;
  snapshot.gs.softwareRasterHostNanoseconds = 45u;
  snapshot.gs.vulkanRoundTripsCompleted = 46u;
  snapshot.gs.vulkanRoundTripsFailed = 47u;
  snapshot.gs.vulkanRequestWaits = 48u;
  snapshot.gs.vulkanRequestWaitNanoseconds = 49u;
  snapshot.gs.vulkanQueueSubmissions = 50u;
  snapshot.gs.vulkanShaderDispatches = 51u;
  snapshot.gs.vulkanBytesUploaded = 52u;
  snapshot.gs.vulkanBytesDownloaded = 53u;
  snapshot.gs.vulkanFenceWaits = 54u;
  snapshot.gs.vulkanFenceWaitNanoseconds = 55u;
  snapshot.gs.vulkanCommittedGpuCommands = 56u;
  snapshot.gs.vulkanVerificationMismatches = 57u;
  snapshot.vu1.asyncEnabled = true;
  snapshot.vu1.pendingSlice = true;
  snapshot.vu1.deferredSlice = true;
  snapshot.vu1.slicesSubmitted = 61u;
  snapshot.vu1.slicesPublished = 62u;
  snapshot.vu1.resultsReadyAtEvent = 63u;
  snapshot.vu1.resultsLateAtEvent = 64u;
  snapshot.vu1.eventWaitCount = 65u;
  snapshot.vu1.eventWaitNanoseconds = 66u;
  snapshot.vu1.maximumEventWaitNanoseconds = 67u;
  snapshot.vu1.hazardBarrierCount = 68u;
  snapshot.vu1.hazardWaitNanoseconds = 69u;
  snapshot.vu1.budgetFallbackCount = 70u;
  snapshot.vu1.budgetFallbackWaitNanoseconds = 71u;
  snapshot.vu1.ownerQueueCapacity = 72u;
  snapshot.vu1.ownerQueueDepth = 73u;
  snapshot.vu1.ownerQueueHighWater = 74u;
  snapshot.vu1.ownerQueuedPayloadBytes = 75u;
  snapshot.vu1.ownerPayloadCapacityBytes = 76u;
  snapshot.vu1.ownerWorkerActiveNanoseconds = 77u;
  snapshot.vu1.ownerWorkerIdleNanoseconds = 78u;
  snapshot.vu1.ownerProducerBlockCount = 79u;
  snapshot.vu1.ownerProducerBlockedNanoseconds = 80u;
  snapshot.vu1.ownerResultWaitCount = 81u;
  snapshot.vu1.ownerResultWaitNanoseconds = 82u;
  snapshot.vu1.ownerWorkNotificationCount = 83u;
  snapshot.vu1.ownerWorkerWakeCount = 84u;
  return snapshot;
}

void testProtocolRoundTrip() {
  using namespace ps2x::performance;
  const TelemetrySnapshot original = populatedSnapshot();
  const std::string encoded = makeSnapshotResponse(99u, original);
  uint64_t responseId = 0u;
  TelemetrySnapshot decoded{};
  std::string error;
  check(parseSnapshotResponse(encoded, responseId, decoded, &error),
        "snapshot response parses");
  check(error.empty(), "snapshot response has no parser error");
  check(responseId == 99u, "snapshot response preserves request id");
  check(makeSnapshotResponse(responseId, decoded) == encoded,
        "snapshot response round-trips every serialized field");

  const std::string request = makeSnapshotRequest(123u);
  uint64_t requestId = 0u;
  check(parseSnapshotRequest(request, requestId, &error),
        "snapshot request parses");
  check(requestId == 123u, "snapshot request preserves id");
  check(!parseSnapshotRequest(
            R"({"jsonrpc":"2.0","id":1,"method":"control.pause"})", requestId,
            &error),
        "protocol rejects control methods");

  std::string wrongSchema = encoded;
  const size_t schema = wrongSchema.find("\"schema\":1");
  check(schema != std::string::npos, "encoded schema field is present");
  if (schema != std::string::npos)
    wrongSchema.replace(schema, 10u, "\"schema\":2");
  check(!parseSnapshotResponse(wrongSchema, responseId, decoded, &error),
        "protocol rejects an unknown schema");
}

void testRatesAndResetHandling() {
  using namespace ps2x::performance;
  TelemetrySnapshot previous{};
  previous.monotonicNanoseconds = 1'000'000'000u;
  previous.guest.vsyncPeriodCycles = 4'915'200u;
  previous.guest.presentations = 100u;
  previous.guest.vsyncFields = 200u;
  previous.gs.ownerWorkerActiveNanoseconds = 10u;
  previous.gs.ownerWorkerIdleNanoseconds = 20u;
  previous.vu1.ownerWorkerActiveNanoseconds = 30u;
  previous.vu1.ownerWorkerIdleNanoseconds = 40u;
  previous.gs.routingSoftwareCommands = 100u;
  previous.gs.routingAcceleratedCommands = 200u;
  previous.gs.routingNoopCommands = 300u;
  previous.gs.routingFallbackCommands = 10u;
  previous.gs.routingSoftwarePixels = 1'000u;
  previous.gs.routingAcceleratedPixels = 2'000u;
  previous.gs.softwareRasterHostNanoseconds = 50u;

  TelemetrySnapshot current = previous;
  current.monotonicNanoseconds = 2'000'000'000u;
  current.guest.presentations += 30u;
  current.guest.vsyncFields += 60u;
  current.gs.ownerWorkerActiveNanoseconds += 800u;
  current.gs.ownerWorkerIdleNanoseconds += 200u;
  current.vu1.ownerWorkerActiveNanoseconds += 500u;
  current.vu1.ownerWorkerIdleNanoseconds += 500u;
  current.gs.routingSoftwareCommands += 10u;
  current.gs.routingAcceleratedCommands += 30u;
  current.gs.routingNoopCommands += 10u;
  current.gs.routingFallbackCommands += 2u;
  current.gs.routingSoftwarePixels += 250u;
  current.gs.routingAcceleratedPixels += 750u;
  current.gs.softwareRasterHostNanoseconds += 2'000'000'000u;

  const TelemetryRates rates = calculateTelemetryRates(previous, current);
  check(rates.valid, "telemetry rate interval is valid");
  check(rates.routingValid, "routing interval is valid");
  checkNear(rates.framesPerSecond, 30.0, "presentation FPS");
  checkNear(rates.speedPercent, 100.0, "VSync speed percentage");
  checkNear(rates.gsOwnerBusyPercent, 80.0, "GS owner busy ratio");
  checkNear(rates.vu1OwnerBusyPercent, 50.0, "VU1 owner busy ratio");
  checkNear(rates.softwareCommandPercent, 20.0, "software command share");
  checkNear(rates.acceleratedCommandPercent, 60.0, "accelerated command share");
  checkNear(rates.noopCommandPercent, 20.0, "no-op command share");
  checkNear(rates.softwarePixelPercent, 25.0, "software pixel share");
  checkNear(rates.acceleratedPixelPercent, 75.0, "accelerated pixel share");
  checkNear(rates.softwareRasterCorePercent, 200.0,
            "parallel software raster core cost may exceed one core");

  current.monotonicNanoseconds += 1'000'000'000u;
  current.gs.routingSoftwareCommands = 0u;
  const TelemetryRates resetRates = calculateTelemetryRates(previous, current);
  check(!resetRates.routingValid,
        "routing counter reset never underflows into a bogus rate");
}

void testLinuxThreadParsing() {
  using namespace ps2x::performance;
  LinuxThreadCounters counters{};
  check(parseLinuxSchedstat(" 123  456\t789 999\n", counters),
        "schedstat parser accepts normal whitespace and extra fields");
  check(counters.runNanoseconds == 123u, "schedstat run time");
  check(counters.waitNanoseconds == 456u, "schedstat wait time");
  check(counters.timeslices == 789u, "schedstat timeslices");
  check(!parseLinuxSchedstat("123 invalid 789", counters),
        "schedstat parser rejects malformed counters");
  check(classifyLinuxThread(50, 50, "name with spaces") ==
            LinuxThreadRole::Main,
        "main thread classification uses TID, not comm parsing");
  check(classifyLinuxThread(50, 51, "EeExecutor") ==
            LinuxThreadRole::EeExecutor,
        "EE role classification");
  check(classifyLinuxThread(50, 52, "PS2GsRaster") == LinuxThreadRole::GsRaster,
        "raster worker role classification");
  check(classifyLinuxThread(50, 53, "worker with spaces") ==
            LinuxThreadRole::Other,
        "thread names with spaces remain intact and classify safely");

  LinuxThreadSampler sampler(::getpid());
  const auto samples = sampler.sample(1'000'000'000u);
  bool foundMain = false;
  for (const LinuxThreadSample &sample : samples) {
    if (sample.tid == ::getpid()) {
      foundMain = sample.role == LinuxThreadRole::Main && !sample.name.empty();
      break;
    }
  }
  check(foundMain, "Linux sampler discovers the current main thread");
}
} // namespace

int main() {
  testProtocolRoundTrip();
  testRatesAndResetHandling();
  testLinuxThreadParsing();
  if (failures != 0) {
    std::cerr << failures << " performance telemetry test(s) failed\n";
    return 1;
  }
  std::cout << "performance telemetry tests passed\n";
  return 0;
}
