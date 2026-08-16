#include "runtime/ps2_performance_threads_linux.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <system_error>

namespace ps2x::performance {
namespace {
uint64_t steadyClockNanoseconds() noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

bool parseUnsigned(std::string_view text, size_t &offset,
                   uint64_t &value) noexcept {
  while (offset < text.size() &&
         (text[offset] == ' ' || text[offset] == '\t')) {
    ++offset;
  }
  if (offset == text.size())
    return false;
  const char *const first = text.data() + offset;
  const char *const last = text.data() + text.size();
  const auto result = std::from_chars(first, last, value);
  if (result.ec != std::errc{} || result.ptr == first)
    return false;
  offset = static_cast<size_t>(result.ptr - text.data());
  return true;
}

bool parseThreadId(std::string_view name, int64_t &threadId) noexcept {
  if (name.empty())
    return false;
  const auto result =
      std::from_chars(name.data(), name.data() + name.size(), threadId);
  return result.ec == std::errc{} && result.ptr == name.data() + name.size() &&
         threadId > 0;
}

bool readTextFile(const std::filesystem::path &path, std::string &output) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    return false;
  output.assign(std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>());
  return file.good() || file.eof();
}

void trimLineEnding(std::string &value) {
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
}
} // namespace

bool parseLinuxSchedstat(std::string_view text,
                         LinuxThreadCounters &counters) noexcept {
  LinuxThreadCounters parsed{};
  size_t offset = 0u;
  if (!parseUnsigned(text, offset, parsed.runNanoseconds) ||
      !parseUnsigned(text, offset, parsed.waitNanoseconds) ||
      !parseUnsigned(text, offset, parsed.timeslices)) {
    return false;
  }
  counters = parsed;
  return true;
}

LinuxThreadRole classifyLinuxThread(int64_t processId, int64_t threadId,
                                    std::string_view name) noexcept {
  if (threadId == processId)
    return LinuxThreadRole::Main;
  if (name == "EeExecutor")
    return LinuxThreadRole::EeExecutor;
  if (name == "PS2Vu1Owner")
    return LinuxThreadRole::Vu1Owner;
  if (name == "PS2GsOwner")
    return LinuxThreadRole::GsOwner;
  if (name == "PS2GsVulkan")
    return LinuxThreadRole::GsVulkan;
  if (name.starts_with("PS2GsRaster"))
    return LinuxThreadRole::GsRaster;
  return LinuxThreadRole::Other;
}

const char *linuxThreadRoleName(LinuxThreadRole role) noexcept {
  switch (role) {
  case LinuxThreadRole::Main:
    return "Main / presentation";
  case LinuxThreadRole::EeExecutor:
    return "EE executor";
  case LinuxThreadRole::Vu1Owner:
    return "VU1 owner";
  case LinuxThreadRole::GsOwner:
    return "GS owner";
  case LinuxThreadRole::GsVulkan:
    return "GS Vulkan";
  case LinuxThreadRole::GsRaster:
    return "GS software raster";
  case LinuxThreadRole::Other:
    return "Other";
  }
  return "Other";
}

LinuxThreadSampler::LinuxThreadSampler(int64_t processId)
    : m_processId(processId) {}

std::vector<LinuxThreadSample>
LinuxThreadSampler::sample(uint64_t monotonicNanoseconds) {
  if (monotonicNanoseconds == 0u)
    monotonicNanoseconds = steadyClockNanoseconds();

  std::vector<LinuxThreadSample> samples;
  std::unordered_map<int64_t, PreviousSample> nextPrevious;
  const std::filesystem::path taskDirectory =
      std::filesystem::path("/proc") / std::to_string(m_processId) / "task";
  std::error_code error;
  std::filesystem::directory_iterator iterator(taskDirectory, error);
  const std::filesystem::directory_iterator end;
  while (!error && iterator != end) {
    const std::filesystem::directory_entry entry = *iterator;
    iterator.increment(error);
    int64_t threadId = 0;
    const std::string filename = entry.path().filename().string();
    if (!parseThreadId(filename, threadId))
      continue;

    std::string name;
    std::string schedstat;
    LinuxThreadCounters counters{};
    if (!readTextFile(entry.path() / "comm", name) ||
        !readTextFile(entry.path() / "schedstat", schedstat) ||
        !parseLinuxSchedstat(schedstat, counters)) {
      continue;
    }
    trimLineEnding(name);

    LinuxThreadSample sample{};
    sample.tid = threadId;
    sample.name = std::move(name);
    sample.role = classifyLinuxThread(m_processId, threadId, sample.name);
    sample.pinned = sample.role != LinuxThreadRole::Other;
    sample.counters = counters;

    const auto previous = m_previous.find(threadId);
    if (previous != m_previous.end() &&
        monotonicNanoseconds > previous->second.monotonicNanoseconds &&
        counters.runNanoseconds >= previous->second.counters.runNanoseconds &&
        counters.waitNanoseconds >= previous->second.counters.waitNanoseconds &&
        counters.timeslices >= previous->second.counters.timeslices) {
      const uint64_t elapsedNanoseconds =
          monotonicNanoseconds - previous->second.monotonicNanoseconds;
      const uint64_t runNanoseconds =
          counters.runNanoseconds - previous->second.counters.runNanoseconds;
      const uint64_t waitNanoseconds =
          counters.waitNanoseconds - previous->second.counters.waitNanoseconds;
      const uint64_t timeslices =
          counters.timeslices - previous->second.counters.timeslices;
      const double elapsedSeconds =
          static_cast<double>(elapsedNanoseconds) / 1'000'000'000.0;
      sample.ratesValid = elapsedNanoseconds != 0u;
      if (sample.ratesValid) {
        sample.cpuPercent = static_cast<double>(runNanoseconds) * 100.0 /
                            static_cast<double>(elapsedNanoseconds);
        sample.runQueueWaitPercent = static_cast<double>(waitNanoseconds) *
                                     100.0 /
                                     static_cast<double>(elapsedNanoseconds);
        sample.timeslicesPerSecond =
            static_cast<double>(timeslices) / elapsedSeconds;
        if (timeslices != 0u) {
          sample.averageRunMicroseconds = static_cast<double>(runNanoseconds) /
                                          static_cast<double>(timeslices) /
                                          1'000.0;
        }
      }
    }

    nextPrevious.emplace(threadId,
                         PreviousSample{
                             .counters = counters,
                             .monotonicNanoseconds = monotonicNanoseconds,
                         });
    samples.push_back(std::move(sample));
  }

  m_previous = std::move(nextPrevious);
  std::sort(samples.begin(), samples.end(),
            [](const LinuxThreadSample &left, const LinuxThreadSample &right) {
              if (left.pinned != right.pinned)
                return left.pinned > right.pinned;
              if (left.cpuPercent != right.cpuPercent)
                return left.cpuPercent > right.cpuPercent;
              return left.tid < right.tid;
            });
  return samples;
}

void LinuxThreadSampler::reset() noexcept { m_previous.clear(); }
} // namespace ps2x::performance
