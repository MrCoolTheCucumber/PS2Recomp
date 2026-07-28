#ifndef PS2_PERF_JITDUMP_H
#define PS2_PERF_JITDUMP_H

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

// Minimal Linux perf jitdump writer. Each code load is flushed immediately so
// a runner that terminates without C++ static destruction still leaves a
// complete, injectable prefix of the dump.
class PS2PerfJitDumpWriter
{
public:
    PS2PerfJitDumpWriter();
    ~PS2PerfJitDumpWriter();

    PS2PerfJitDumpWriter(
        const PS2PerfJitDumpWriter &) = delete;
    PS2PerfJitDumpWriter &operator=(
        const PS2PerfJitDumpWriter &) = delete;
    PS2PerfJitDumpWriter(
        PS2PerfJitDumpWriter &&) = delete;
    PS2PerfJitDumpWriter &operator=(
        PS2PerfJitDumpWriter &&) = delete;

    [[nodiscard]] static bool supported();

    [[nodiscard]] bool open(
        const std::filesystem::path &outputPath,
        std::string *diagnostic = nullptr);
    [[nodiscard]] bool registerCode(
        const void *code, size_t sizeBytes,
        std::string_view symbol, uint64_t *codeIndex = nullptr,
        std::string *diagnostic = nullptr);
    [[nodiscard]] bool close(
        std::string *diagnostic = nullptr);

    [[nodiscard]] bool isOpen() const;
    [[nodiscard]] std::filesystem::path outputPath() const;

private:
    struct Implementation;
    std::unique_ptr<Implementation> m_implementation;
};

// Environment-backed process writer used by runtime JITs. Set
// PS2X_PERF_JITDUMP=1 to enable it and optionally set
// PS2X_PERF_JITDUMP_DIR to choose the output directory.
[[nodiscard]] bool
ps2PerfJitDumpEnvironmentRequested() noexcept;
[[nodiscard]] bool ps2PerfJitDumpInitialize(
    std::string *diagnostic = nullptr);
[[nodiscard]] bool ps2PerfJitDumpRegisterCode(
    const void *code, size_t sizeBytes,
    std::string_view symbol, uint64_t *codeIndex = nullptr,
    std::string *diagnostic = nullptr);

#endif
