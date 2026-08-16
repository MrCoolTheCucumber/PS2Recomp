#include "ps2_performance_hud.h"

#include "ps2_runtime.h"
#include "runtime/ps2_performance_telemetry.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace {
using namespace std::chrono_literals;

uint64_t monotonicNanoseconds() noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

bool ensurePrivateDirectory(const std::filesystem::path &directory,
                            std::string &error) {
  std::error_code filesystemError;
  std::filesystem::create_directories(directory, filesystemError);
  if (filesystemError) {
    error = "could not create " + directory.string() + ": " +
            filesystemError.message();
    return false;
  }

  struct stat status{};
  if (::lstat(directory.c_str(), &status) != 0) {
    error =
        "could not inspect " + directory.string() + ": " + std::strerror(errno);
    return false;
  }
  if (!S_ISDIR(status.st_mode) || status.st_uid != ::geteuid()) {
    error = "runtime directory is not a private user-owned directory: " +
            directory.string();
    return false;
  }
  if (::chmod(directory.c_str(), S_IRWXU) != 0) {
    error =
        "could not protect " + directory.string() + ": " + std::strerror(errno);
    return false;
  }
  return true;
}

std::filesystem::path socketDirectory() {
  if (const char *const runtimeDirectory = std::getenv("XDG_RUNTIME_DIR")) {
    if (runtimeDirectory[0] == '/') {
      return std::filesystem::path(runtimeDirectory) / "ps2recomp";
    }
  }
  return std::filesystem::path("/tmp") /
         ("ps2recomp-" + std::to_string(::geteuid()));
}

bool sendAll(int socket, std::string_view bytes) {
  size_t offset = 0u;
  while (offset < bytes.size()) {
    const ssize_t sent = ::send(socket, bytes.data() + offset,
                                bytes.size() - offset, MSG_NOSIGNAL);
    if (sent > 0) {
      offset += static_cast<size_t>(sent);
      continue;
    }
    if (sent < 0 && errno == EINTR)
      continue;
    return false;
  }
  return true;
}
} // namespace

class PS2PerformanceHudImpl {
public:
  explicit PS2PerformanceHudImpl(PS2Runtime &runtime) : m_runtime(runtime) {}

  ~PS2PerformanceHudImpl() { stop(); }

  void toggle() {
    update();
    if (m_childPid <= 0) {
      if (!m_serverStarted && !startServer())
        return;
      if (!spawnCompanion()) {
        stopServer();
        return;
      }
      setRoutingCountersEnabled(true);
      m_desiredVisible.store(true, std::memory_order_release);
      return;
    }

    const bool show = !m_desiredVisible.load(std::memory_order_acquire);
    if (show) {
      setRoutingCountersEnabled(true);
      m_desiredVisible.store(true, std::memory_order_release);
    } else {
      m_desiredVisible.store(false, std::memory_order_release);
      setRoutingCountersEnabled(false);
    }
  }

  void update() {
    if (m_childPid <= 0)
      return;
    const auto now = std::chrono::steady_clock::now();
    if (now < m_nextChildPoll)
      return;
    m_nextChildPoll = now + 250ms;

    int status = 0;
    const pid_t result = ::waitpid(m_childPid, &status, WNOHANG);
    if (result == m_childPid || (result < 0 && errno == ECHILD)) {
      m_childPid = -1;
      m_expectedPeerPid.store(-1, std::memory_order_release);
      m_desiredVisible.store(false, std::memory_order_release);
      setRoutingCountersEnabled(false);
      stopServer();
    }
  }

  void stop() {
    if (m_stopped.exchange(true, std::memory_order_acq_rel))
      return;

    m_desiredVisible.store(false, std::memory_order_release);
    setRoutingCountersEnabled(false);
    stopServer();

    if (m_childPid > 0) {
      (void)::kill(m_childPid, SIGTERM);
      const auto deadline = std::chrono::steady_clock::now() + 500ms;
      int status = 0;
      while (std::chrono::steady_clock::now() < deadline) {
        const pid_t result = ::waitpid(m_childPid, &status, WNOHANG);
        if (result == m_childPid || (result < 0 && errno == ECHILD)) {
          m_childPid = -1;
          break;
        }
        std::this_thread::sleep_for(10ms);
      }
      if (m_childPid > 0) {
        (void)::kill(m_childPid, SIGKILL);
        while (::waitpid(m_childPid, &status, 0) < 0 && errno == EINTR) {
        }
        m_childPid = -1;
      }
    }
    m_expectedPeerPid.store(-1, std::memory_order_release);
  }

private:
  bool startServer() {
    std::string error;
    const std::filesystem::path directory = socketDirectory();
    if (!ensurePrivateDirectory(directory, error)) {
      reportError(error);
      return false;
    }
    m_socketPath =
        (directory / ("perf-hud-" + std::to_string(::getpid()) + ".sock"))
            .string();
    if (m_socketPath.size() >=
        sizeof(static_cast<sockaddr_un *>(nullptr)->sun_path)) {
      reportError("telemetry socket path is too long: " + m_socketPath);
      return false;
    }

    struct stat existing{};
    if (::lstat(m_socketPath.c_str(), &existing) == 0) {
      if (!S_ISSOCK(existing.st_mode) || existing.st_uid != ::geteuid()) {
        reportError("refusing to replace non-socket telemetry path: " +
                    m_socketPath);
        return false;
      }
      if (::unlink(m_socketPath.c_str()) != 0) {
        reportError("could not remove stale telemetry socket: " +
                    std::string(std::strerror(errno)));
        return false;
      }
    } else if (errno != ENOENT) {
      reportError("could not inspect telemetry socket: " +
                  std::string(std::strerror(errno)));
      return false;
    }

    const int listener = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (listener < 0) {
      reportError("could not create telemetry socket: " +
                  std::string(std::strerror(errno)));
      return false;
    }
    const int flags = ::fcntl(listener, F_GETFL, 0);
    if (flags < 0 || ::fcntl(listener, F_SETFL, flags | O_NONBLOCK) != 0) {
      reportError("could not configure telemetry socket: " +
                  std::string(std::strerror(errno)));
      (void)::close(listener);
      return false;
    }

    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, m_socketPath.c_str(),
                m_socketPath.size() + 1u);
    const socklen_t addressSize = static_cast<socklen_t>(
        offsetof(sockaddr_un, sun_path) + m_socketPath.size() + 1u);
    if (::bind(listener, reinterpret_cast<const sockaddr *>(&address),
               addressSize) != 0 ||
        ::chmod(m_socketPath.c_str(), S_IRUSR | S_IWUSR) != 0 ||
        ::listen(listener, 1) != 0) {
      reportError("could not publish telemetry socket: " +
                  std::string(std::strerror(errno)));
      (void)::close(listener);
      (void)::unlink(m_socketPath.c_str());
      return false;
    }

    m_listener = listener;
    m_serverStop.store(false, std::memory_order_release);
    m_serverThread = std::thread([this] { serverLoop(); });
    m_serverStarted = true;
    return true;
  }

  bool spawnCompanion() {
    std::error_code error;
    const std::filesystem::path executable =
        std::filesystem::read_symlink("/proc/self/exe", error);
    if (error) {
      reportError("could not resolve the runner executable: " +
                  error.message());
      return false;
    }
    const std::filesystem::path companion =
        executable.parent_path() / "ps2DebugHud";
    if (::access(companion.c_str(), X_OK) != 0) {
      reportError("performance HUD executable is unavailable: " +
                  companion.string());
      return false;
    }

    std::string executableArgument = companion.string();
    std::string pidArgument = std::to_string(::getpid());
    std::string socketArgument = m_socketPath;
    std::array<char *, 6u> arguments{
        executableArgument.data(), const_cast<char *>("--pid"),
        pidArgument.data(),        const_cast<char *>("--socket"),
        socketArgument.data(),     nullptr,
    };
    pid_t child = -1;
    const int spawnResult =
        ::posix_spawn(&child, executableArgument.c_str(), nullptr, nullptr,
                      arguments.data(), environ);
    if (spawnResult != 0) {
      reportError("could not launch the performance HUD: " +
                  std::string(std::strerror(spawnResult)));
      return false;
    }
    m_childPid = child;
    m_expectedPeerPid.store(child, std::memory_order_release);
    m_nextChildPoll = std::chrono::steady_clock::now() + 250ms;
    return true;
  }

  void stopServer() {
    if (!m_serverStarted)
      return;
    m_serverStop.store(true, std::memory_order_release);
    if (m_serverThread.joinable())
      m_serverThread.join();
    if (!m_socketPath.empty())
      (void)::unlink(m_socketPath.c_str());
    m_serverStarted = false;
    m_listener = -1;
  }

  void setRoutingCountersEnabled(bool enabled) {
    if (enabled == m_routingCountersEnabled)
      return;
    try {
      if (enabled) {
        (void)m_runtime.submitGsCommand(GsResetBackendCountersCommand{});
        (void)m_runtime.submitGsCommand(
            GsSetBackendCountersEnabledCommand{true});
      } else {
        (void)m_runtime.submitGsCommand(
            GsSetBackendCountersEnabledCommand{false});
      }
      m_routingCountersEnabled = enabled;
    } catch (const std::exception &exception) {
      reportError(std::string("could not ") + (enabled ? "enable" : "disable") +
                  " GS routing counters: " + exception.what());
    }
  }

  ps2x::performance::TelemetrySnapshot makeSnapshot() {
    using namespace ps2x::performance;
    TelemetrySnapshot snapshot{};
    snapshot.sequence =
        m_sequence.fetch_add(1u, std::memory_order_relaxed) + 1u;
    snapshot.monotonicNanoseconds = monotonicNanoseconds();
    snapshot.processId = ::getpid();
    snapshot.desiredVisible = m_desiredVisible.load(std::memory_order_acquire);
    snapshot.detailed = snapshot.desiredVisible;
    snapshot.gsExecutionMode = gsExecutionModeName(m_runtime.m_gsExecutionMode);
    snapshot.vu1ExecutionMode =
        vu1ExecutionModeName(m_runtime.m_vu1ExecutionMode);
    snapshot.rendererMode = m_lastRendererMode;
    snapshot.rendererDiagnostic = m_lastRendererDiagnostic;
    snapshot.guest.eeTick = m_runtime.currentEeTick().raw();
    snapshot.guest.vsyncFields =
        m_runtime.m_debugVSyncFields.load(std::memory_order_relaxed);
    snapshot.guest.vsyncPeriodCycles =
        m_runtime.m_debugVSyncPeriodCycles.load(std::memory_order_relaxed);

    if (!snapshot.detailed)
      return snapshot;

    const GsAsyncRuntimeStatistics gsAsync = m_runtime.gsAsyncStatistics();
    snapshot.gs.asyncEnabled = gsAsync.enabled;
    snapshot.gs.pendingCompletions = gsAsync.pendingCompletions;
    snapshot.gs.pendingHighWater = gsAsync.pendingHighWater;
    snapshot.gs.fieldsSubmitted = gsAsync.fieldsSubmitted;
    snapshot.gs.fieldsCompleted = gsAsync.fieldsCompleted;
    snapshot.gs.fieldLeadBlockCount = gsAsync.fieldLeadBlockCount;
    snapshot.gs.fieldLeadBlockedNanoseconds =
        gsAsync.fieldLeadBlockedNanoseconds;
    snapshot.gs.ownerQueueCapacity = gsAsync.owner.queueCapacity;
    snapshot.gs.ownerQueueDepth = gsAsync.owner.queueDepth;
    snapshot.gs.ownerQueueHighWater = gsAsync.owner.queueHighWater;
    snapshot.gs.ownerQueuedPayloadBytes = gsAsync.owner.queuedPayloadBytes;
    snapshot.gs.ownerPayloadCapacityBytes = gsAsync.owner.payloadCapacityBytes;
    snapshot.gs.ownerWorkerActiveNanoseconds =
        gsAsync.owner.workerActiveNanoseconds;
    snapshot.gs.ownerWorkerIdleNanoseconds =
        gsAsync.owner.workerIdleNanoseconds;
    snapshot.gs.ownerProducerBlockCount = gsAsync.owner.producerBlockCount;
    snapshot.gs.ownerProducerBlockedNanoseconds =
        gsAsync.owner.producerBlockedNanoseconds;

    const Vu1AsyncRuntimeStatistics vu1Async = m_runtime.vu1AsyncStatistics();
    const ThreadedVu1ExecutorStatistics vu1Owner =
        m_runtime.vu1OwnerStatistics();
    snapshot.vu1.asyncEnabled = vu1Async.enabled;
    snapshot.vu1.pendingSlice = vu1Async.pendingSlice;
    snapshot.vu1.deferredSlice = vu1Async.deferredSlice;
    snapshot.vu1.slicesSubmitted = vu1Async.slicesSubmitted;
    snapshot.vu1.slicesPublished = vu1Async.slicesPublished;
    snapshot.vu1.resultsReadyAtEvent = vu1Async.resultsReadyAtEvent;
    snapshot.vu1.resultsLateAtEvent = vu1Async.resultsLateAtEvent;
    snapshot.vu1.eventWaitCount = vu1Async.eventWaitCount;
    snapshot.vu1.eventWaitNanoseconds = vu1Async.eventWaitNanoseconds;
    snapshot.vu1.maximumEventWaitNanoseconds =
        vu1Async.maximumEventWaitNanoseconds;
    snapshot.vu1.hazardBarrierCount = vu1Async.hazardBarrierCount;
    snapshot.vu1.hazardWaitNanoseconds = vu1Async.hazardWaitNanoseconds;
    snapshot.vu1.budgetFallbackCount = vu1Async.budgetFallbackCount;
    snapshot.vu1.budgetFallbackWaitNanoseconds =
        vu1Async.budgetFallbackWaitNanoseconds;
    snapshot.vu1.ownerQueueCapacity = vu1Owner.queueCapacity;
    snapshot.vu1.ownerQueueDepth = vu1Owner.queueDepth;
    snapshot.vu1.ownerQueueHighWater = vu1Owner.queueHighWater;
    snapshot.vu1.ownerQueuedPayloadBytes = vu1Owner.queuedPayloadBytes;
    snapshot.vu1.ownerPayloadCapacityBytes = vu1Owner.payloadCapacityBytes;
    snapshot.vu1.ownerWorkerActiveNanoseconds =
        vu1Owner.workerActiveNanoseconds;
    snapshot.vu1.ownerWorkerIdleNanoseconds = vu1Owner.workerIdleNanoseconds;
    snapshot.vu1.ownerProducerBlockCount = vu1Owner.producerBlockCount;
    snapshot.vu1.ownerProducerBlockedNanoseconds =
        vu1Owner.producerBlockedNanoseconds;
    snapshot.vu1.ownerResultWaitCount = vu1Owner.resultWaitCount;
    snapshot.vu1.ownerResultWaitNanoseconds = vu1Owner.resultWaitNanoseconds;
    snapshot.vu1.ownerWorkNotificationCount = vu1Owner.workNotificationCount;
    snapshot.vu1.ownerWorkerWakeCount = vu1Owner.workerWakeCount;

    const GsProgressSnapshotResult progress =
        takeGsCommandResult<GsProgressSnapshotResult>(
            m_runtime.submitGsCommand(GsProgressSnapshotCommand{}));
    snapshot.guest.presentations = progress.snapshot.presentations;

    const GsRendererStatusResult renderer =
        takeGsCommandResult<GsRendererStatusResult>(
            m_runtime.submitGsCommand(GsRendererStatusCommand{}));
    m_lastRendererMode = std::string(gsRendererModeName(renderer.mode));
    m_lastRendererDiagnostic = renderer.diagnostic;
    snapshot.rendererMode = m_lastRendererMode;
    snapshot.rendererDiagnostic = m_lastRendererDiagnostic;
    snapshot.gs.vulkanRoundTripsCompleted =
        renderer.serviceStatistics.roundTripsCompleted;
    snapshot.gs.vulkanRoundTripsFailed =
        renderer.serviceStatistics.roundTripsFailed;
    snapshot.gs.vulkanRequestWaits = renderer.serviceStatistics.requestWaits;
    snapshot.gs.vulkanRequestWaitNanoseconds =
        renderer.serviceStatistics.requestWaitNanoseconds;
    snapshot.gs.vulkanQueueSubmissions =
        renderer.serviceStatistics.queueSubmissions;
    snapshot.gs.vulkanShaderDispatches =
        renderer.serviceStatistics.shaderDispatches;
    snapshot.gs.vulkanBytesUploaded = renderer.serviceStatistics.bytesUploaded;
    snapshot.gs.vulkanBytesDownloaded =
        renderer.serviceStatistics.bytesDownloaded;
    snapshot.gs.vulkanFenceWaits = renderer.serviceStatistics.fenceWaits;
    snapshot.gs.vulkanFenceWaitNanoseconds =
        renderer.serviceStatistics.fenceWaitNanoseconds;
    snapshot.gs.vulkanCommittedGpuCommands =
        renderer.backendStatistics.committedGpuCommands;
    snapshot.gs.vulkanVerificationMismatches =
        renderer.backendStatistics.verificationMismatches;

    const GsBackendCountersResult routing =
        takeGsCommandResult<GsBackendCountersResult>(
            m_runtime.submitGsCommand(GsBackendCountersCommand{}));
    snapshot.gs.routingCommands = routing.counters.commands;
    snapshot.gs.routingNoopCommands = routing.counters.noopCommands;
    snapshot.gs.routingSoftwareCommands = routing.counters.softwareCommands;
    snapshot.gs.routingAcceleratedCommands =
        routing.counters.acceleratedCommands;
    snapshot.gs.routingFallbackCommands = routing.counters.fallbackCommands;
    snapshot.gs.routingDrawPixels = routing.counters.drawPixels;
    snapshot.gs.routingSoftwarePixels = routing.counters.softwarePixels;
    snapshot.gs.routingAcceleratedPixels = routing.counters.acceleratedPixels;
    snapshot.gs.routingFallbackPixels = routing.counters.fallbackPixels;
    snapshot.gs.softwareRasterHostNanoseconds =
        routing.counters.softwareRasterHostNanoseconds;
    return snapshot;
  }

  void serverLoop() {
    int client = -1;
    std::string input;
    while (!m_serverStop.load(std::memory_order_acquire)) {
      std::array<pollfd, 2u> descriptors{};
      descriptors[0] = pollfd{
          .fd = m_listener,
          .events = POLLIN,
          .revents = 0,
      };
      descriptors[1] = pollfd{
          .fd = client,
          .events = static_cast<short>(POLLIN | POLLHUP | POLLERR),
          .revents = 0,
      };
      const nfds_t count = client >= 0 ? 2u : 1u;
      const int pollResult = ::poll(descriptors.data(), count, 100);
      if (pollResult < 0) {
        if (errno == EINTR)
          continue;
        break;
      }

      if ((descriptors[0].revents & POLLIN) != 0) {
        const int accepted =
            ::accept4(m_listener, nullptr, nullptr, SOCK_CLOEXEC);
        if (accepted >= 0) {
          ucred credentials{};
          socklen_t credentialSize = sizeof(credentials);
          const pid_t expected =
              m_expectedPeerPid.load(std::memory_order_acquire);
          const bool trusted =
              ::getsockopt(accepted, SOL_SOCKET, SO_PEERCRED, &credentials,
                           &credentialSize) == 0 &&
              credentials.uid == ::geteuid() &&
              (expected <= 0 || credentials.pid == expected);
          if (!trusted || client >= 0) {
            (void)::close(accepted);
          } else {
            client = accepted;
            input.clear();
          }
        }
      }

      if (client < 0)
        continue;
      const short events = descriptors[1].revents;
      if ((events & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
        (void)::close(client);
        client = -1;
        input.clear();
        continue;
      }
      if ((events & POLLIN) == 0)
        continue;

      std::array<char, 4096u> buffer{};
      const ssize_t received = ::recv(client, buffer.data(), buffer.size(), 0);
      if (received <= 0) {
        if (received < 0 && errno == EINTR)
          continue;
        (void)::close(client);
        client = -1;
        input.clear();
        continue;
      }
      input.append(buffer.data(), static_cast<size_t>(received));
      if (input.size() > 64u * 1024u) {
        (void)::close(client);
        client = -1;
        input.clear();
        continue;
      }

      size_t newline = 0u;
      while (client >= 0 && (newline = input.find('\n')) != std::string::npos) {
        std::string request = input.substr(0u, newline);
        input.erase(0u, newline + 1u);
        uint64_t requestId = 0u;
        std::string parseError;
        std::string response;
        if (!ps2x::performance::parseSnapshotRequest(request, requestId,
                                                     &parseError)) {
          response = ps2x::performance::makeErrorResponse(requestId, -32600,
                                                          parseError);
        } else {
          try {
            response = ps2x::performance::makeSnapshotResponse(requestId,
                                                               makeSnapshot());
          } catch (const std::exception &exception) {
            response = ps2x::performance::makeErrorResponse(requestId, -32000,
                                                            exception.what());
          } catch (...) {
            response = ps2x::performance::makeErrorResponse(
                requestId, -32000, "unknown telemetry snapshot failure");
          }
        }
        if (!sendAll(client, response)) {
          (void)::close(client);
          client = -1;
          input.clear();
        }
      }
    }

    if (client >= 0)
      (void)::close(client);
    if (m_listener >= 0) {
      (void)::close(m_listener);
      m_listener = -1;
    }
  }

  void reportError(const std::string &message) {
    std::cerr << "[performance-hud] " << message << std::endl;
  }

  PS2Runtime &m_runtime;
  std::atomic<bool> m_desiredVisible{false};
  std::atomic<bool> m_serverStop{false};
  std::atomic<bool> m_stopped{false};
  std::atomic<pid_t> m_expectedPeerPid{-1};
  std::atomic<uint64_t> m_sequence{0u};
  std::thread m_serverThread;
  int m_listener = -1;
  pid_t m_childPid = -1;
  bool m_serverStarted = false;
  bool m_routingCountersEnabled = false;
  std::string m_socketPath;
  std::string m_lastRendererMode{"unavailable"};
  std::string m_lastRendererDiagnostic;
  std::chrono::steady_clock::time_point m_nextChildPoll{};
};

PS2PerformanceHud::PS2PerformanceHud(PS2Runtime &runtime)
    : m_impl(std::make_unique<PS2PerformanceHudImpl>(runtime)) {}

PS2PerformanceHud::~PS2PerformanceHud() = default;

void PS2PerformanceHud::toggle() { m_impl->toggle(); }

void PS2PerformanceHud::update() { m_impl->update(); }

void PS2PerformanceHud::stop() { m_impl->stop(); }
