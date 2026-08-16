#include "runtime/ps2_performance_telemetry.h"
#include "runtime/ps2_performance_threads_linux.h"

#include "imgui.h"
#include "raylib.h"
#include "rlImGui.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <poll.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

namespace {
using namespace std::chrono_literals;
using ps2x::performance::LinuxThreadRole;
using ps2x::performance::LinuxThreadSample;
using ps2x::performance::TelemetryRates;
using ps2x::performance::TelemetrySnapshot;

struct Arguments {
  int64_t processId = 0;
  std::string socketPath;
};

uint64_t monotonicNanoseconds() noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

bool parseProcessId(std::string_view text, int64_t &value) noexcept {
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size() &&
         value > 0;
}

std::optional<Arguments> parseArguments(int argc, char **argv) {
  Arguments arguments{};
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--pid" && index + 1 < argc) {
      if (!parseProcessId(argv[++index], arguments.processId))
        return std::nullopt;
    } else if (argument == "--socket" && index + 1 < argc) {
      arguments.socketPath = argv[++index];
    } else {
      return std::nullopt;
    }
  }
  if (arguments.processId <= 0 || arguments.socketPath.empty() ||
      arguments.socketPath.size() >=
          sizeof(static_cast<sockaddr_un *>(nullptr)->sun_path)) {
    return std::nullopt;
  }
  return arguments;
}

bool processExists(int64_t processId) noexcept {
  if (::kill(static_cast<pid_t>(processId), 0) == 0)
    return true;
  return errno == EPERM;
}

bool sendAll(int socket, std::string_view bytes) {
  size_t offset = 0u;
  while (offset < bytes.size()) {
    const ssize_t result = ::send(socket, bytes.data() + offset,
                                  bytes.size() - offset, MSG_NOSIGNAL);
    if (result > 0) {
      offset += static_cast<size_t>(result);
      continue;
    }
    if (result < 0 && errno == EINTR)
      continue;
    return false;
  }
  return true;
}

class TelemetryClient {
public:
  struct State {
    bool connected = false;
    bool parentGone = false;
    uint64_t revision = 0u;
    std::optional<TelemetrySnapshot> snapshot;
    std::string error;
  };

  TelemetryClient(int64_t processId, std::string socketPath)
      : m_processId(processId), m_socketPath(std::move(socketPath)),
        m_thread([this] { run(); }) {}

  ~TelemetryClient() {
    m_stop.store(true, std::memory_order_release);
    if (m_thread.joinable())
      m_thread.join();
  }

  [[nodiscard]] State state() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_state;
  }

private:
  int connectToServer() {
    const int socket = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (socket < 0)
      return -1;

    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, m_socketPath.c_str(),
                m_socketPath.size() + 1u);
    const socklen_t addressSize = static_cast<socklen_t>(
        offsetof(sockaddr_un, sun_path) + m_socketPath.size() + 1u);
    if (::connect(socket, reinterpret_cast<const sockaddr *>(&address),
                  addressSize) != 0) {
      (void)::close(socket);
      return -1;
    }

    ucred credentials{};
    socklen_t credentialSize = sizeof(credentials);
    if (::getsockopt(socket, SOL_SOCKET, SO_PEERCRED, &credentials,
                     &credentialSize) != 0 ||
        credentials.pid != static_cast<pid_t>(m_processId) ||
        credentials.uid != ::geteuid()) {
      (void)::close(socket);
      return -1;
    }
    return socket;
  }

  bool receiveLine(int socket, std::string &line) {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!m_stop.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
      const size_t newline = m_input.find('\n');
      if (newline != std::string::npos) {
        line = m_input.substr(0u, newline);
        m_input.erase(0u, newline + 1u);
        return true;
      }

      pollfd descriptor{
          .fd = socket,
          .events = static_cast<short>(POLLIN | POLLHUP | POLLERR),
          .revents = 0,
      };
      const int pollResult = ::poll(&descriptor, 1u, 100);
      if (pollResult < 0) {
        if (errno == EINTR)
          continue;
        return false;
      }
      if (pollResult == 0)
        continue;
      if ((descriptor.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
        return false;
      }
      if ((descriptor.revents & POLLIN) == 0)
        continue;

      std::array<char, 8192u> buffer{};
      const ssize_t received = ::recv(socket, buffer.data(), buffer.size(), 0);
      if (received > 0) {
        m_input.append(buffer.data(), static_cast<size_t>(received));
        if (m_input.size() > 256u * 1024u)
          return false;
        continue;
      }
      if (received < 0 && errno == EINTR)
        continue;
      return false;
    }
    return false;
  }

  bool waitInterruptibly(std::chrono::milliseconds duration) {
    const auto deadline = std::chrono::steady_clock::now() + duration;
    while (std::chrono::steady_clock::now() < deadline) {
      if (m_stop.load(std::memory_order_acquire))
        return false;
      std::this_thread::sleep_for(20ms);
    }
    return !m_stop.load(std::memory_order_acquire);
  }

  void publishDisconnected(std::string error, bool parentGone) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.connected = false;
    m_state.parentGone = parentGone;
    m_state.error = std::move(error);
    ++m_state.revision;
  }

  void publishSnapshot(TelemetrySnapshot snapshot) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_state.connected = true;
    m_state.parentGone = false;
    m_state.snapshot = std::move(snapshot);
    m_state.error.clear();
    ++m_state.revision;
  }

  void run() {
    int socket = -1;
    uint64_t requestId = 0u;
    bool visible = true;
    while (!m_stop.load(std::memory_order_acquire)) {
      if (socket < 0) {
        socket = connectToServer();
        if (socket < 0) {
          const bool gone = !processExists(m_processId);
          publishDisconnected(gone ? "game process exited"
                                   : "waiting for telemetry socket",
                              gone);
          if (gone || !waitInterruptibly(100ms))
            break;
          continue;
        }
        m_input.clear();
      }

      const uint64_t expectedId = ++requestId;
      const std::string request =
          ps2x::performance::makeSnapshotRequest(expectedId);
      std::string response;
      if (!sendAll(socket, request) || !receiveLine(socket, response)) {
        (void)::close(socket);
        socket = -1;
        publishDisconnected("telemetry connection lost", false);
        continue;
      }

      uint64_t responseId = 0u;
      TelemetrySnapshot snapshot{};
      std::string parseError;
      if (!ps2x::performance::parseSnapshotResponse(response, responseId,
                                                    snapshot, &parseError) ||
          responseId != expectedId) {
        publishDisconnected(
            parseError.empty() ? "telemetry response id mismatch" : parseError,
            false);
        if (!waitInterruptibly(250ms))
          break;
        continue;
      }
      visible = snapshot.desiredVisible;
      publishSnapshot(std::move(snapshot));
      if (!waitInterruptibly(visible ? 250ms : 100ms))
        break;
    }
    if (socket >= 0)
      (void)::close(socket);
  }

  int64_t m_processId = 0;
  std::string m_socketPath;
  std::atomic<bool> m_stop{false};
  mutable std::mutex m_mutex;
  State m_state{};
  std::thread m_thread;
  std::string m_input;
};

struct WindowPlacement {
  int x = 80;
  int y = 80;
  int width = 1000;
  int height = 720;
};

std::filesystem::path placementPath() {
  if (const char *const configHome = std::getenv("XDG_CONFIG_HOME")) {
    if (configHome[0] != '\0') {
      return std::filesystem::path(configHome) / "ps2recomp" / "perf-hud.ini";
    }
  }
  if (const char *const userHome = std::getenv("HOME")) {
    if (userHome[0] != '\0') {
      return std::filesystem::path(userHome) / ".config" / "ps2recomp" /
             "perf-hud.ini";
    }
  }
  return {};
}

WindowPlacement loadPlacement() {
  WindowPlacement placement{};
  const std::filesystem::path path = placementPath();
  if (path.empty())
    return placement;
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    const size_t separator = line.find('=');
    if (separator == std::string::npos)
      continue;
    const std::string_view name(line.data(), separator);
    const std::string_view value(line.data() + separator + 1u,
                                 line.size() - separator - 1u);
    int parsed = 0;
    const auto result =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
      continue;
    }
    if (name == "x")
      placement.x = parsed;
    else if (name == "y")
      placement.y = parsed;
    else if (name == "width")
      placement.width = parsed;
    else if (name == "height")
      placement.height = parsed;
  }
  placement.width = std::clamp(placement.width, 720, 7680);
  placement.height = std::clamp(placement.height, 480, 4320);
  return placement;
}

WindowPlacement clampPlacement(WindowPlacement placement) {
  const int monitorCount = GetMonitorCount();
  if (monitorCount <= 0)
    return placement;

  int bestMonitor = 0;
  int64_t bestOverlap = -1;
  for (int monitor = 0; monitor < monitorCount; ++monitor) {
    const Vector2 position = GetMonitorPosition(monitor);
    const int monitorX = static_cast<int>(position.x);
    const int monitorY = static_cast<int>(position.y);
    const int monitorWidth = GetMonitorWidth(monitor);
    const int monitorHeight = GetMonitorHeight(monitor);
    const int overlapWidth = std::max(
        0, std::min(placement.x + placement.width, monitorX + monitorWidth) -
               std::max(placement.x, monitorX));
    const int overlapHeight = std::max(
        0, std::min(placement.y + placement.height, monitorY + monitorHeight) -
               std::max(placement.y, monitorY));
    const int64_t overlap = static_cast<int64_t>(overlapWidth) * overlapHeight;
    if (overlap > bestOverlap) {
      bestOverlap = overlap;
      bestMonitor = monitor;
    }
  }

  const Vector2 monitorPosition = GetMonitorPosition(bestMonitor);
  const int monitorX = static_cast<int>(monitorPosition.x);
  const int monitorY = static_cast<int>(monitorPosition.y);
  const int monitorWidth = GetMonitorWidth(bestMonitor);
  const int monitorHeight = GetMonitorHeight(bestMonitor);
  placement.width =
      std::clamp(placement.width, 720, std::max(720, monitorWidth));
  placement.height =
      std::clamp(placement.height, 480, std::max(480, monitorHeight));
  const int visibleWidth = std::min(120, placement.width);
  const int visibleHeight = std::min(80, placement.height);
  placement.x =
      std::clamp(placement.x, monitorX - placement.width + visibleWidth,
                 monitorX + monitorWidth - visibleWidth);
  placement.y = std::clamp(placement.y, monitorY,
                           monitorY + monitorHeight - visibleHeight);
  return placement;
}

void savePlacement() {
  const std::filesystem::path path = placementPath();
  if (path.empty() || !IsWindowReady() || IsWindowHidden())
    return;
  const Vector2 position = GetWindowPosition();
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error)
    return;
  std::ofstream file(path, std::ios::trunc);
  if (!file)
    return;
  file << "version=1\n"
       << "x=" << static_cast<int>(position.x) << '\n'
       << "y=" << static_cast<int>(position.y) << '\n'
       << "width=" << GetScreenWidth() << '\n'
       << "height=" << GetScreenHeight() << '\n';
}

class HistorySeries {
public:
  void push(float value) {
    if (!std::isfinite(value))
      return;
    if (m_values.size() == kCapacity)
      m_values.erase(m_values.begin());
    m_values.push_back(value);
  }

  [[nodiscard]] const std::vector<float> &values() const noexcept {
    return m_values;
  }

private:
  static constexpr size_t kCapacity = 240u;
  std::vector<float> m_values;
};

class ThreadWindow {
public:
  std::vector<LinuxThreadSample>
  update(std::vector<LinuxThreadSample> samples) {
    std::unordered_map<int64_t, std::deque<LinuxThreadSample>> next;
    std::vector<LinuxThreadSample> smoothed;
    smoothed.reserve(samples.size());
    for (LinuxThreadSample &sample : samples) {
      auto prior = m_samples.find(sample.tid);
      std::deque<LinuxThreadSample> history =
          prior != m_samples.end() ? std::move(prior->second)
                                   : std::deque<LinuxThreadSample>{};
      history.push_back(sample);
      while (history.size() > 4u)
        history.pop_front();

      LinuxThreadSample average = sample;
      average.ratesValid = false;
      average.cpuPercent = 0.0;
      average.runQueueWaitPercent = 0.0;
      average.timeslicesPerSecond = 0.0;
      average.averageRunMicroseconds = 0.0;
      size_t valid = 0u;
      double weightedRun = 0.0;
      double sliceWeight = 0.0;
      for (const LinuxThreadSample &entry : history) {
        if (!entry.ratesValid)
          continue;
        ++valid;
        average.cpuPercent += entry.cpuPercent;
        average.runQueueWaitPercent += entry.runQueueWaitPercent;
        average.timeslicesPerSecond += entry.timeslicesPerSecond;
        weightedRun += entry.averageRunMicroseconds * entry.timeslicesPerSecond;
        sliceWeight += entry.timeslicesPerSecond;
      }
      if (valid != 0u) {
        average.ratesValid = true;
        average.cpuPercent /= static_cast<double>(valid);
        average.runQueueWaitPercent /= static_cast<double>(valid);
        average.timeslicesPerSecond /= static_cast<double>(valid);
        if (sliceWeight > 0.0) {
          average.averageRunMicroseconds = weightedRun / sliceWeight;
        }
      }
      smoothed.push_back(std::move(average));
      next.emplace(sample.tid, std::move(history));
    }
    m_samples = std::move(next);
    std::sort(
        smoothed.begin(), smoothed.end(),
        [](const LinuxThreadSample &left, const LinuxThreadSample &right) {
          if (left.cpuPercent != right.cpuPercent)
            return left.cpuPercent > right.cpuPercent;
          return left.tid < right.tid;
        });
    return smoothed;
  }

  void reset() { m_samples.clear(); }

private:
  std::unordered_map<int64_t, std::deque<LinuxThreadSample>> m_samples;
};

struct RoleAggregate {
  LinuxThreadRole role = LinuxThreadRole::Other;
  size_t count = 0u;
  bool valid = false;
  double cpuPercent = 0.0;
  double waitPercent = 0.0;
  double timeslicesPerSecond = 0.0;
  double averageRunMicroseconds = 0.0;
};

std::array<RoleAggregate, 6u>
aggregateRoles(const std::vector<LinuxThreadSample> &threads) {
  std::array<RoleAggregate, 6u> roles{
      RoleAggregate{.role = LinuxThreadRole::Main},
      RoleAggregate{.role = LinuxThreadRole::EeExecutor},
      RoleAggregate{.role = LinuxThreadRole::Vu1Owner},
      RoleAggregate{.role = LinuxThreadRole::GsOwner},
      RoleAggregate{.role = LinuxThreadRole::GsVulkan},
      RoleAggregate{.role = LinuxThreadRole::GsRaster},
  };
  for (const LinuxThreadSample &thread : threads) {
    const auto role = std::find_if(
        roles.begin(), roles.end(),
        [&](const RoleAggregate &entry) { return entry.role == thread.role; });
    if (role == roles.end())
      continue;
    ++role->count;
    if (!thread.ratesValid)
      continue;
    role->valid = true;
    role->cpuPercent += thread.cpuPercent;
    role->waitPercent += thread.runQueueWaitPercent;
    role->timeslicesPerSecond += thread.timeslicesPerSecond;
    role->averageRunMicroseconds +=
        thread.averageRunMicroseconds * thread.timeslicesPerSecond;
  }
  for (RoleAggregate &role : roles) {
    if (role.timeslicesPerSecond > 0.0) {
      role.averageRunMicroseconds /= role.timeslicesPerSecond;
    }
  }
  return roles;
}

double totalCpuPercent(const std::vector<LinuxThreadSample> &threads) noexcept {
  double total = 0.0;
  for (const LinuxThreadSample &thread : threads) {
    if (thread.ratesValid)
      total += thread.cpuPercent;
  }
  return total;
}

std::string formatNumber(double value) {
  std::array<char, 64u> buffer{};
  const double magnitude = std::abs(value);
  if (magnitude >= 1'000'000'000.0)
    std::snprintf(buffer.data(), buffer.size(), "%.2fG", value / 1e9);
  else if (magnitude >= 1'000'000.0)
    std::snprintf(buffer.data(), buffer.size(), "%.2fM", value / 1e6);
  else if (magnitude >= 1'000.0)
    std::snprintf(buffer.data(), buffer.size(), "%.2fk", value / 1e3);
  else
    std::snprintf(buffer.data(), buffer.size(), "%.1f", value);
  return buffer.data();
}

std::string formatBytes(double value) {
  std::array<char, 64u> buffer{};
  constexpr std::array<const char *, 5u> units{"B", "KiB", "MiB", "GiB", "TiB"};
  size_t unit = 0u;
  while (std::abs(value) >= 1024.0 && unit + 1u < units.size()) {
    value /= 1024.0;
    ++unit;
  }
  std::snprintf(buffer.data(), buffer.size(), "%.2f %s", value, units[unit]);
  return buffer.data();
}

std::string formatPerPresentation(double value) {
  if (std::abs(value) >= 1'000.0)
    return formatNumber(value);
  std::array<char, 64u> buffer{};
  const double magnitude = std::abs(value);
  if (magnitude >= 100.0)
    std::snprintf(buffer.data(), buffer.size(), "%.1f", value);
  else if (magnitude >= 1.0)
    std::snprintf(buffer.data(), buffer.size(), "%.2f", value);
  else
    std::snprintf(buffer.data(), buffer.size(), "%.3f", value);
  return buffer.data();
}

double flushRate(const ps2x::performance::FlushReasonRates &rates,
                 ps2x::performance::TelemetryFlushReason reason) noexcept {
  return rates[ps2x::performance::flushReasonIndex(reason)];
}

void metricCell(const char *label, const std::string &value,
                const char *detail = nullptr) {
  ImGui::TextDisabled("%s", label);
  ImGui::SetWindowFontScale(1.35f);
  ImGui::TextUnformatted(value.c_str());
  ImGui::SetWindowFontScale(1.0f);
  if (detail) {
    ImGui::TextDisabled("%s", detail);
  }
}

void drawHistory(const char *label, const HistorySeries &history, float minimum,
                 float maximum, const char *overlay) {
  const std::vector<float> &values = history.values();
  ImGui::TextDisabled("%s", label);
  ImGui::PlotLines((std::string("##") + label).c_str(),
                   values.empty() ? nullptr : values.data(),
                   static_cast<int>(values.size()), 0, overlay, minimum,
                   maximum, ImVec2(-1.0f, 58.0f));
}

void drawThreadBar(const RoleAggregate &role) {
  ImGui::TableNextRow();
  ImGui::TableSetColumnIndex(0);
  ImGui::TextUnformatted(ps2x::performance::linuxThreadRoleName(role.role));
  if (role.role == LinuxThreadRole::GsRaster && role.count > 1u) {
    ImGui::SameLine();
    ImGui::TextDisabled("(%zu threads)", role.count);
  }
  ImGui::TableSetColumnIndex(1);
  if (!role.valid) {
    ImGui::TextDisabled(role.count == 0u ? "not running" : "warming");
  } else {
    std::array<char, 64u> overlay{};
    std::snprintf(overlay.data(), overlay.size(), "%.1f%% of one core",
                  role.cpuPercent);
    ImGui::ProgressBar(
        static_cast<float>(std::clamp(role.cpuPercent / 100.0, 0.0, 1.0)),
        ImVec2(-1.0f, 0.0f), overlay.data());
  }
  ImGui::TableSetColumnIndex(2);
  if (role.valid)
    ImGui::Text("%.1f%%", role.waitPercent);
  else
    ImGui::TextDisabled("-");
  ImGui::TableSetColumnIndex(3);
  if (role.valid)
    ImGui::Text("%.0f/s", role.timeslicesPerSecond);
  else
    ImGui::TextDisabled("-");
}

void drawKeyThreads(const std::vector<LinuxThreadSample> &threads) {
  ImGui::SeparatorText("Key host threads (trailing 1 s)");
  ImGui::TextDisabled(
      "CPU is schedstat run time as a percentage of one logical core. "
      "Run-queue wait is time runnable but not scheduled.");
  const auto roles = aggregateRoles(threads);
  if (ImGui::BeginTable("key-threads", 4,
                        ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableSetupColumn("Role", ImGuiTableColumnFlags_WidthFixed, 160.0f);
    ImGui::TableSetupColumn("Busy", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Run-q wait", ImGuiTableColumnFlags_WidthFixed,
                            85.0f);
    ImGui::TableSetupColumn("Slices", ImGuiTableColumnFlags_WidthFixed, 75.0f);
    ImGui::TableHeadersRow();
    for (const RoleAggregate &role : roles)
      drawThreadBar(role);
    ImGui::EndTable();
  }
}

void drawRendererSplit(const TelemetrySnapshot &snapshot,
                       const TelemetryRates &rates) {
  ImGui::SeparatorText("Renderer workload split");
  ImGui::Text("Mode: %s", snapshot.rendererMode.c_str());
  ImGui::SameLine();
  ImGui::TextDisabled(
      "| accelerated workload share, not vendor GPU utilization");

  if (!rates.routingValid) {
    ImGui::ProgressBar(0.0f, ImVec2(-1.0f, 0.0f),
                       "routing counters warming up");
    ImGui::ProgressBar(0.0f, ImVec2(-1.0f, 0.0f), "pixel split warming up");
  } else {
    std::array<char, 128u> commands{};
    std::snprintf(
        commands.data(), commands.size(),
        "commands: %.1f%% accelerated | %.1f%% software | %.1f%% no-op",
        rates.acceleratedCommandPercent, rates.softwareCommandPercent,
        rates.noopCommandPercent);
    ImGui::ProgressBar(
        static_cast<float>(rates.acceleratedCommandPercent / 100.0),
        ImVec2(-1.0f, 0.0f), commands.data());
    std::array<char, 128u> pixels{};
    std::snprintf(pixels.data(), pixels.size(),
                  "candidate pixels: %.1f%% accelerated | %.1f%% software",
                  rates.acceleratedPixelPercent, rates.softwarePixelPercent);
    ImGui::ProgressBar(
        static_cast<float>(rates.acceleratedPixelPercent / 100.0),
        ImVec2(-1.0f, 0.0f), pixels.data());
  }

  if (ImGui::BeginTable("renderer-rates", 2,
                        ImGuiTableFlags_BordersInnerH |
                            ImGuiTableFlags_RowBg)) {
    const auto row = [](const char *name, const std::string &value) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::TextUnformatted(name);
      ImGui::TableSetColumnIndex(1);
      ImGui::TextUnformatted(value.c_str());
    };
    row("Software raster host cost",
        formatNumber(rates.softwareRasterCorePercent) + "% of one core");
    row("Software / accelerated draws",
        formatNumber(rates.softwareCommandsPerSecond) + " / " +
            formatNumber(rates.acceleratedCommandsPerSecond) + "/s");
    row("Fallback draws", formatNumber(rates.fallbackCommandsPerSecond) + "/s");
    if (rates.batchingValid) {
      row("Backend switches / router flushes",
          formatNumber(rates.backendSwitchesPerSecond) + " / " +
              formatNumber(rates.routingFlushesPerSecond) + "/s");
    } else {
      row("Backend switches / router flushes", "warming");
    }
    if (rates.routingValid && rates.presentationValid) {
      row("Draws / presentation",
          "sw " + formatPerPresentation(rates.softwareCommandsPerPresentation) +
              " | accel " +
              formatPerPresentation(rates.acceleratedCommandsPerPresentation) +
              " | no-op " +
              formatPerPresentation(rates.noopCommandsPerPresentation));
    } else {
      row("Draws / presentation", "warming or no presentation");
    }
    if (rates.routingValid && rates.batchingValid && rates.presentationValid) {
      row("Fallback / switch / flush",
          formatPerPresentation(rates.fallbackCommandsPerPresentation) + " / " +
              formatPerPresentation(rates.backendSwitchesPerPresentation) +
              " / " +
              formatPerPresentation(rates.routingFlushesPerPresentation) +
              " per presentation");
      using ps2x::performance::TelemetryFlushReason;
      const auto &flushes = rates.routingFlushReasonsPerPresentation;
      const double hazards =
          flushRate(flushes, TelemetryFlushReason::QueueBackpressure) +
          flushRate(flushes, TelemetryFlushReason::ResourceHazard) +
          flushRate(flushes, TelemetryFlushReason::PipelineChange);
      row("Router flush causes / presentation",
          "switch " +
              formatPerPresentation(
                  flushRate(flushes, TelemetryFlushReason::BackendSwitch)) +
              " | transfer " +
              formatPerPresentation(
                  flushRate(flushes, TelemetryFlushReason::Transfer)) +
              " | readback " +
              formatPerPresentation(
                  flushRate(flushes, TelemetryFlushReason::CpuReadback)) +
              " | hazard " + formatPerPresentation(hazards));
    } else {
      row("Fallback / switch / flush", "warming or no presentation");
    }
    if (rates.coherencyValid && rates.presentationValid) {
      row("Vulkan drains / presentation",
          "resource " +
              formatPerPresentation(
                  rates.vulkanResourceHazardDrainsPerPresentation) +
              " | queue " +
              formatPerPresentation(
                  rates.vulkanQueueBackpressureDrainsPerPresentation) +
              " | pipeline " +
              formatPerPresentation(
                  rates.vulkanPipelineChangeDrainsPerPresentation));
      row("Coherency ops / presentation",
          "CPU access " +
              formatPerPresentation(
                  rates.vulkanCpuAccessPreparationsPerPresentation) +
              " | up " +
              formatPerPresentation(
                  rates.vulkanCpuToGpuOperationsPerPresentation) +
              " | down " +
              formatPerPresentation(
                  rates.vulkanGpuToCpuOperationsPerPresentation));
      row("Coherency pages / presentation",
          "up " +
              formatPerPresentation(rates.vulkanCpuToGpuPagesPerPresentation) +
              " | down " +
              formatPerPresentation(rates.vulkanGpuToCpuPagesPerPresentation));
      const size_t switchIndex = ps2x::performance::flushReasonIndex(
          ps2x::performance::TelemetryFlushReason::BackendSwitch);
      row("Switch-caused downloads",
          formatPerPresentation(
              rates.vulkanDownloadOperationsByReasonPerPresentation
                  [switchIndex]) +
              " ops | " +
              formatPerPresentation(
                  rates.vulkanDownloadedPagesByReasonPerPresentation
                      [switchIndex]) +
              " pages / presentation");
    } else {
      row("Coherency / presentation", "warming or unavailable");
    }
    row("Vulkan submissions / dispatches",
        formatNumber(rates.vulkanSubmissionsPerSecond) + " / " +
            formatNumber(rates.vulkanDispatchesPerSecond) + "/s");
    if (rates.presentationValid) {
      row("Vulkan submit / dispatch per presentation",
          formatPerPresentation(rates.vulkanSubmissionsPerPresentation) +
              " / " +
              formatPerPresentation(rates.vulkanDispatchesPerPresentation));
    }
    row("Vulkan upload / download",
        formatBytes(rates.vulkanUploadBytesPerSecond) + "/s / " +
            formatBytes(rates.vulkanDownloadBytesPerSecond) + "/s");
    if (rates.presentationValid) {
      row("Vulkan transfer / presentation",
          formatBytes(rates.vulkanUploadBytesPerPresentation) + " up / " +
              formatBytes(rates.vulkanDownloadBytesPerPresentation) + " down");
    }
    row("Committed accelerated commands",
        std::to_string(snapshot.gs.vulkanCommittedGpuCommands));
    ImGui::EndTable();
  }
  if (!snapshot.rendererDiagnostic.empty() &&
      ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    ImGui::SetTooltip("%s", snapshot.rendererDiagnostic.c_str());
  }
}

void drawQueuesAndWaits(const TelemetrySnapshot &snapshot,
                        const TelemetryRates &rates) {
  ImGui::SeparatorText("Queues, ownership and waits");
  if (!ImGui::BeginTable("queues-waits", 5,
                         ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                             ImGuiTableFlags_SizingStretchProp)) {
    return;
  }
  ImGui::TableSetupColumn("Stage");
  ImGui::TableSetupColumn("Current / capacity");
  ImGui::TableSetupColumn("High water");
  ImGui::TableSetupColumn("Owner busy");
  ImGui::TableSetupColumn("Producer/wait cost");
  ImGui::TableHeadersRow();

  ImGui::TableNextRow();
  ImGui::TableSetColumnIndex(0);
  ImGui::TextUnformatted("GS owner");
  ImGui::TableSetColumnIndex(1);
  ImGui::Text("%llu / %llu",
              static_cast<unsigned long long>(snapshot.gs.ownerQueueDepth),
              static_cast<unsigned long long>(snapshot.gs.ownerQueueCapacity));
  ImGui::TableSetColumnIndex(2);
  ImGui::Text("%llu",
              static_cast<unsigned long long>(snapshot.gs.ownerQueueHighWater));
  ImGui::TableSetColumnIndex(3);
  ImGui::Text("%.1f%%", rates.gsOwnerBusyPercent);
  ImGui::TableSetColumnIndex(4);
  ImGui::Text("%.1f%% blocked", rates.gsProducerBlockedPercent);

  ImGui::TableNextRow();
  ImGui::TableSetColumnIndex(0);
  ImGui::TextUnformatted("VU1 owner");
  ImGui::TableSetColumnIndex(1);
  ImGui::Text("%llu / %llu",
              static_cast<unsigned long long>(snapshot.vu1.ownerQueueDepth),
              static_cast<unsigned long long>(snapshot.vu1.ownerQueueCapacity));
  ImGui::TableSetColumnIndex(2);
  ImGui::Text("%llu", static_cast<unsigned long long>(
                          snapshot.vu1.ownerQueueHighWater));
  ImGui::TableSetColumnIndex(3);
  ImGui::Text("%.1f%%", rates.vu1OwnerBusyPercent);
  ImGui::TableSetColumnIndex(4);
  ImGui::Text("%.1f%% producer | %.1f%% event", rates.vu1ProducerBlockedPercent,
              rates.vu1EventWaitPercent);

  ImGui::TableNextRow();
  ImGui::TableSetColumnIndex(0);
  ImGui::TextUnformatted("GS field overlap");
  ImGui::TableSetColumnIndex(1);
  ImGui::Text("%llu pending",
              static_cast<unsigned long long>(snapshot.gs.pendingCompletions));
  ImGui::TableSetColumnIndex(2);
  ImGui::Text("%llu",
              static_cast<unsigned long long>(snapshot.gs.pendingHighWater));
  ImGui::TableSetColumnIndex(3);
  ImGui::TextDisabled("-");
  ImGui::TableSetColumnIndex(4);
  ImGui::Text("%llu lead blocks",
              static_cast<unsigned long long>(snapshot.gs.fieldLeadBlockCount));

  ImGui::TableNextRow();
  ImGui::TableSetColumnIndex(0);
  ImGui::TextUnformatted("Vulkan service");
  ImGui::TableSetColumnIndex(1);
  ImGui::Text("%llu round trips", static_cast<unsigned long long>(
                                      snapshot.gs.vulkanRoundTripsCompleted));
  ImGui::TableSetColumnIndex(2);
  ImGui::Text("%llu failed", static_cast<unsigned long long>(
                                 snapshot.gs.vulkanRoundTripsFailed));
  ImGui::TableSetColumnIndex(3);
  ImGui::Text("%.1f%% fence", rates.vulkanFenceWaitPercent);
  ImGui::TableSetColumnIndex(4);
  ImGui::Text("%.1f%% request", rates.vulkanRequestWaitPercent);
  ImGui::EndTable();
}

void drawAllThreads(std::vector<LinuxThreadSample> threads) {
  if (!ImGui::CollapsingHeader("All game-process threads"))
    return;
  ImGui::TextDisabled(
      "Read from /proc/PID/task/*/{comm,schedstat}; sorted by trailing "
      "1-second CPU activity.");
  std::sort(threads.begin(), threads.end(),
            [](const LinuxThreadSample &left, const LinuxThreadSample &right) {
              if (left.cpuPercent != right.cpuPercent)
                return left.cpuPercent > right.cpuPercent;
              return left.tid < right.tid;
            });
  if (!ImGui::BeginTable("all-threads", 7,
                         ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                             ImGuiTableFlags_Resizable |
                             ImGuiTableFlags_ScrollY,
                         ImVec2(0.0f, 260.0f))) {
    return;
  }
  ImGui::TableSetupScrollFreeze(0, 1);
  ImGui::TableSetupColumn("TID", ImGuiTableColumnFlags_WidthFixed, 72.0f);
  ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableSetupColumn("Role", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableSetupColumn("CPU/core", ImGuiTableColumnFlags_WidthFixed, 82.0f);
  ImGui::TableSetupColumn("Run-q wait", ImGuiTableColumnFlags_WidthFixed,
                          88.0f);
  ImGui::TableSetupColumn("Slices/s", ImGuiTableColumnFlags_WidthFixed, 76.0f);
  ImGui::TableSetupColumn("Mean run", ImGuiTableColumnFlags_WidthFixed, 82.0f);
  ImGui::TableHeadersRow();
  for (const LinuxThreadSample &thread : threads) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("%lld", static_cast<long long>(thread.tid));
    ImGui::TableSetColumnIndex(1);
    ImGui::TextUnformatted(thread.name.c_str());
    ImGui::TableSetColumnIndex(2);
    ImGui::TextUnformatted(ps2x::performance::linuxThreadRoleName(thread.role));
    ImGui::TableSetColumnIndex(3);
    if (thread.ratesValid)
      ImGui::Text("%.1f%%", thread.cpuPercent);
    else
      ImGui::TextDisabled("warming");
    ImGui::TableSetColumnIndex(4);
    if (thread.ratesValid)
      ImGui::Text("%.1f%%", thread.runQueueWaitPercent);
    else
      ImGui::TextDisabled("-");
    ImGui::TableSetColumnIndex(5);
    if (thread.ratesValid)
      ImGui::Text("%.0f", thread.timeslicesPerSecond);
    else
      ImGui::TextDisabled("-");
    ImGui::TableSetColumnIndex(6);
    if (thread.ratesValid)
      ImGui::Text("%.1f us", thread.averageRunMicroseconds);
    else
      ImGui::TextDisabled("-");
  }
  ImGui::EndTable();
}

struct DashboardHistory {
  HistorySeries framesPerSecond;
  HistorySeries speedPercent;
  HistorySeries processCpuPercent;
  HistorySeries acceleratedPixelPercent;
};

void drawDashboard(const TelemetryClient::State &client,
                   const TelemetrySnapshot &snapshot,
                   const TelemetryRates &rates,
                   const std::vector<LinuxThreadSample> &threads,
                   const DashboardHistory &history) {
  ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
  ImGui::SetNextWindowSize(ImVec2(static_cast<float>(GetScreenWidth()),
                                  static_cast<float>(GetScreenHeight())));
  const ImGuiWindowFlags flags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
  if (!ImGui::Begin("##performance-dashboard", nullptr, flags)) {
    ImGui::End();
    return;
  }

  const double ageMilliseconds =
      snapshot.monotonicNanoseconds <= monotonicNanoseconds()
          ? static_cast<double>(monotonicNanoseconds() -
                                snapshot.monotonicNanoseconds) /
                1'000'000.0
          : 0.0;
  const ImVec4 liveColor = client.connected ? ImVec4(0.35f, 0.90f, 0.48f, 1.0f)
                                            : ImVec4(1.0f, 0.55f, 0.25f, 1.0f);
  ImGui::TextColored(liveColor, client.connected ? "LIVE" : "DISCONNECTED");
  ImGui::SameLine();
  ImGui::Text(
      "RAC1 performance HUD  |  PID %lld  |  sample %llu  |  %.0f ms old",
      static_cast<long long>(snapshot.processId),
      static_cast<unsigned long long>(snapshot.sequence), ageMilliseconds);
  ImGui::SameLine();
  ImGui::TextDisabled("  F12 in the game toggles this window");
  if (!client.error.empty())
    ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.25f, 1.0f), "%s",
                       client.error.c_str());

  const double processCpu = totalCpuPercent(threads);
  if (ImGui::BeginTable("headline-metrics", 4,
                        ImGuiTableFlags_BordersInnerV |
                            ImGuiTableFlags_SizingStretchSame)) {
    ImGui::TableNextColumn();
    metricCell("GAME PRESENTATION",
               formatNumber(rates.framesPerSecond) + " FPS",
               "guest presentations / host second");
    ImGui::TableNextColumn();
    metricCell("EMULATION SPEED", formatNumber(rates.speedPercent) + "%",
               "scheduled VSync field rate");
    ImGui::TableNextColumn();
    metricCell("GAME PROCESS CPU", formatNumber(processCpu / 100.0) + " cores",
               "sum of sampled game threads");
    ImGui::TableNextColumn();
    metricCell("ACCELERATED PIXEL SHARE",
               rates.routingValid
                   ? formatNumber(rates.acceleratedPixelPercent) + "%"
                   : "warming",
               "workload share; not GPU utilization");
    ImGui::EndTable();
  }

  if (ImGui::BeginTable("history-plots", 4,
                        ImGuiTableFlags_SizingStretchSame)) {
    ImGui::TableNextColumn();
    drawHistory("FPS - last 60 s", history.framesPerSecond, 0.0f, 60.0f,
                nullptr);
    ImGui::TableNextColumn();
    drawHistory("Speed - last 60 s", history.speedPercent, 0.0f, 110.0f,
                nullptr);
    ImGui::TableNextColumn();
    drawHistory("Game CPU - last 60 s", history.processCpuPercent, 0.0f,
                std::max(400.0f, static_cast<float>(processCpu * 1.1)),
                nullptr);
    ImGui::TableNextColumn();
    drawHistory("Accelerated pixels - last 60 s",
                history.acceleratedPixelPercent, 0.0f, 100.0f, nullptr);
    ImGui::EndTable();
  }

  if (ImGui::BeginTable("primary-sections", 2,
                        ImGuiTableFlags_SizingStretchSame |
                            ImGuiTableFlags_BordersInnerV)) {
    ImGui::TableNextColumn();
    drawKeyThreads(threads);
    ImGui::TableNextColumn();
    drawRendererSplit(snapshot, rates);
    ImGui::EndTable();
  }

  drawQueuesAndWaits(snapshot, rates);
  ImGui::TextDisabled(
      "Modes: GS %s | VU1 %s. Vendor GPU engine occupancy and VRAM "
      "utilization are intentionally not reported in v1.",
      snapshot.gsExecutionMode.c_str(), snapshot.vu1ExecutionMode.c_str());
  drawAllThreads(threads);
  ImGui::End();
}

void applyStyle() {
  ImGui::StyleColorsDark();
  ImGuiStyle &style = ImGui::GetStyle();
  style.WindowRounding = 0.0f;
  style.FrameRounding = 3.0f;
  style.GrabRounding = 3.0f;
  style.CellPadding = ImVec2(7.0f, 5.0f);
  style.WindowPadding = ImVec2(12.0f, 10.0f);
  style.ItemSpacing = ImVec2(8.0f, 6.0f);
}
} // namespace

int main(int argc, char **argv) {
  const std::optional<Arguments> arguments = parseArguments(argc, argv);
  if (!arguments) {
    std::fprintf(stderr, "usage: ps2DebugHud --pid PID --socket PATH\n");
    return 2;
  }

  (void)::prctl(PR_SET_NAME, "PS2PerfHud", 0, 0, 0);
  TelemetryClient client(arguments->processId, arguments->socketPath);
  ps2x::performance::LinuxThreadSampler threadSampler(arguments->processId);
  ThreadWindow threadWindow;

  SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_ALWAYS_RUN);
  const WindowPlacement loadedPlacement = loadPlacement();
  const std::string title =
      "PS2Recomp Performance HUD - PID " + std::to_string(arguments->processId);
  InitWindow(loadedPlacement.width, loadedPlacement.height, title.c_str());
  SetWindowMinSize(720, 480);
  const WindowPlacement placement = clampPlacement(loadedPlacement);
  SetWindowSize(placement.width, placement.height);
  SetWindowPosition(placement.x, placement.y);
  SetTargetFPS(0);
  rlImGuiSetup(true);
  applyStyle();

  bool hidden = false;
  bool haveSnapshot = false;
  uint64_t lastSequence = 0u;
  uint64_t lastHistorySequence = 0u;
  std::optional<TelemetrySnapshot> previousSnapshot;
  TelemetrySnapshot snapshot{};
  TelemetryRates rates{};
  std::vector<LinuxThreadSample> threads;
  DashboardHistory history{};
  auto nextThreadSample = std::chrono::steady_clock::now();
  auto nextPlacementSave = std::chrono::steady_clock::now() + 2s;
  auto nextFrame = std::chrono::steady_clock::now();

  while (true) {
    const TelemetryClient::State state = client.state();
    if (state.parentGone)
      break;
    if (state.snapshot && state.snapshot->sequence != lastSequence) {
      snapshot = *state.snapshot;
      haveSnapshot = true;
      lastSequence = snapshot.sequence;
      if (snapshot.detailed && previousSnapshot && previousSnapshot->detailed) {
        rates = ps2x::performance::calculateTelemetryRates(*previousSnapshot,
                                                           snapshot);
      } else {
        rates = {};
      }
      if (snapshot.detailed)
        previousSnapshot = snapshot;
      else
        previousSnapshot.reset();
    }

    const bool shouldBeVisible = !haveSnapshot || snapshot.desiredVisible;
    if (!shouldBeVisible && !hidden) {
      savePlacement();
      SetWindowState(FLAG_WINDOW_HIDDEN);
      hidden = true;
      threadSampler.reset();
      threadWindow.reset();
      threads.clear();
    } else if (shouldBeVisible && hidden) {
      ClearWindowState(FLAG_WINDOW_HIDDEN);
      RestoreWindow();
      hidden = false;
      threadSampler.reset();
      threadWindow.reset();
      nextThreadSample = std::chrono::steady_clock::now();
    }

    if (hidden) {
      PollInputEvents();
      std::this_thread::sleep_for(30ms);
      continue;
    }
    if (WindowShouldClose())
      break;

    const auto now = std::chrono::steady_clock::now();
    if (now >= nextThreadSample) {
      threads = threadWindow.update(threadSampler.sample());
      nextThreadSample = now + 250ms;
      if (rates.valid && snapshot.sequence != lastHistorySequence) {
        lastHistorySequence = snapshot.sequence;
        history.framesPerSecond.push(static_cast<float>(rates.framesPerSecond));
        history.speedPercent.push(static_cast<float>(rates.speedPercent));
        history.processCpuPercent.push(
            static_cast<float>(totalCpuPercent(threads)));
        if (rates.routingValid) {
          history.acceleratedPixelPercent.push(
              static_cast<float>(rates.acceleratedPixelPercent));
        }
      }
    }
    if (now >= nextPlacementSave) {
      savePlacement();
      nextPlacementSave = now + 2s;
    }

    BeginDrawing();
    ClearBackground(Color{18, 20, 24, 255});
    rlImGuiBegin();
    if (haveSnapshot) {
      drawDashboard(state, snapshot, rates, threads, history);
    } else {
      ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
      ImGui::SetNextWindowSize(ImVec2(static_cast<float>(GetScreenWidth()),
                                      static_cast<float>(GetScreenHeight())));
      ImGui::Begin("##connecting", nullptr,
                   ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                       ImGuiWindowFlags_NoResize);
      ImGui::TextUnformatted("Connecting to RAC1 telemetry...");
      if (!state.error.empty())
        ImGui::TextDisabled("%s", state.error.c_str());
      ImGui::End();
    }
    rlImGuiEnd();
    EndDrawing();

    nextFrame += 33ms;
    const auto frameNow = std::chrono::steady_clock::now();
    if (nextFrame > frameNow)
      std::this_thread::sleep_until(nextFrame);
    else if (frameNow - nextFrame > 33ms)
      nextFrame = frameNow;
  }

  if (!hidden)
    savePlacement();
  rlImGuiShutdown();
  CloseWindow();
  return 0;
}
