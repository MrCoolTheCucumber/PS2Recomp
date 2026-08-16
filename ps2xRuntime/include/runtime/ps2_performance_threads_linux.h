#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ps2x::performance {
enum class LinuxThreadRole : uint8_t {
  Main,
  EeExecutor,
  Vu1Owner,
  GsOwner,
  GsVulkan,
  GsRaster,
  Other,
};

struct LinuxThreadCounters {
  uint64_t runNanoseconds = 0u;
  uint64_t waitNanoseconds = 0u;
  uint64_t timeslices = 0u;
};

struct LinuxThreadSample {
  int64_t tid = 0;
  std::string name;
  LinuxThreadRole role = LinuxThreadRole::Other;
  bool pinned = false;
  bool ratesValid = false;
  double cpuPercent = 0.0;
  double runQueueWaitPercent = 0.0;
  double timeslicesPerSecond = 0.0;
  double averageRunMicroseconds = 0.0;
  LinuxThreadCounters counters{};
};

[[nodiscard]] bool parseLinuxSchedstat(std::string_view text,
                                       LinuxThreadCounters &counters) noexcept;
[[nodiscard]] LinuxThreadRole
classifyLinuxThread(int64_t processId, int64_t threadId,
                    std::string_view name) noexcept;
[[nodiscard]] const char *linuxThreadRoleName(LinuxThreadRole role) noexcept;

class LinuxThreadSampler {
public:
  explicit LinuxThreadSampler(int64_t processId);

  [[nodiscard]] std::vector<LinuxThreadSample>
  sample(uint64_t monotonicNanoseconds = 0u);
  void reset() noexcept;

private:
  struct PreviousSample {
    LinuxThreadCounters counters{};
    uint64_t monotonicNanoseconds = 0u;
  };

  int64_t m_processId = 0;
  std::unordered_map<int64_t, PreviousSample> m_previous;
};
} // namespace ps2x::performance
