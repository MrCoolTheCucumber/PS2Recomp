#include "runtime/ps2_perf_jitdump.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <system_error>
#include <utility>

#if defined(__linux__)
#include <elf.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif

namespace
{
    constexpr uint32_t kJitDumpMagic = 0x4a695444u;
    constexpr uint32_t kJitDumpVersion = 1u;
    constexpr uint32_t kJitCodeLoad = 0u;
    constexpr uint32_t kJitCodeClose = 3u;

#pragma pack(push, 1)
    struct JitDumpHeader
    {
        uint32_t magic = kJitDumpMagic;
        uint32_t version = kJitDumpVersion;
        uint32_t totalSize = sizeof(JitDumpHeader);
        uint32_t elfMachine = 0u;
        uint32_t padding = 0u;
        uint32_t processId = 0u;
        uint64_t timestamp = 0u;
        uint64_t flags = 0u;
    };

    struct JitDumpRecordHeader
    {
        uint32_t id = 0u;
        uint32_t totalSize = 0u;
        uint64_t timestamp = 0u;
    };

    struct JitDumpCodeLoad
    {
        JitDumpRecordHeader header{};
        uint32_t processId = 0u;
        uint32_t threadId = 0u;
        uint64_t virtualAddress = 0u;
        uint64_t codeAddress = 0u;
        uint64_t codeSize = 0u;
        uint64_t codeIndex = 0u;
    };
#pragma pack(pop)

    static_assert(sizeof(JitDumpHeader) == 40u);
    static_assert(sizeof(JitDumpRecordHeader) == 16u);
    static_assert(sizeof(JitDumpCodeLoad) == 56u);

    bool fail(
        std::string message, std::string *diagnostic)
    {
        if (diagnostic)
            *diagnostic = std::move(message);
        return false;
    }

    bool environmentFlagEnabled(const char *value) noexcept
    {
        if (!value || value[0] == '\0')
            return false;
        return
            std::strcmp(value, "0") != 0 &&
            std::strcmp(value, "false") != 0 &&
            std::strcmp(value, "FALSE") != 0 &&
            std::strcmp(value, "off") != 0 &&
            std::strcmp(value, "OFF") != 0 &&
            std::strcmp(value, "no") != 0 &&
            std::strcmp(value, "NO") != 0;
    }

#if defined(__linux__)
    uint32_t hostElfMachine()
    {
#if defined(__x86_64__)
        return EM_X86_64;
#elif defined(__aarch64__)
        return EM_AARCH64;
#else
        return 0u;
#endif
    }

    uint64_t monotonicNanoseconds()
    {
        timespec value{};
        if (clock_gettime(CLOCK_MONOTONIC, &value) != 0)
            return 0u;
        return
            static_cast<uint64_t>(value.tv_sec) *
                1000000000ull +
            static_cast<uint64_t>(value.tv_nsec);
    }

    std::string errnoMessage(std::string_view operation)
    {
        return
            std::string(operation) + ": " +
            std::strerror(errno);
    }
#endif
}

struct PS2PerfJitDumpWriter::Implementation
{
    std::mutex mutex;
    std::filesystem::path outputPath;
    std::FILE *file = nullptr;
    void *marker = nullptr;
    size_t markerSize = 0u;
    uint64_t nextCodeIndex = 1u;
    uint64_t lastTimestamp = 0u;
    bool failed = false;

    uint64_t timestamp()
    {
#if defined(__linux__)
        uint64_t value = monotonicNanoseconds();
        if (value == 0u)
            return 0u;
        if (value <= lastTimestamp)
        {
            if (lastTimestamp ==
                std::numeric_limits<uint64_t>::max())
            {
                return 0u;
            }
            value = lastTimestamp + 1u;
        }
        lastTimestamp = value;
        return value;
#else
        return 0u;
#endif
    }
};

PS2PerfJitDumpWriter::PS2PerfJitDumpWriter()
    : m_implementation(
          std::make_unique<Implementation>())
{
}

PS2PerfJitDumpWriter::~PS2PerfJitDumpWriter()
{
    (void)close();
}

bool PS2PerfJitDumpWriter::supported()
{
#if defined(__linux__) && \
    (defined(__x86_64__) || defined(__aarch64__))
    return true;
#else
    return false;
#endif
}

bool PS2PerfJitDumpWriter::open(
    const std::filesystem::path &outputPath,
    std::string *diagnostic)
{
    std::lock_guard<std::mutex> lock(
        m_implementation->mutex);
    Implementation &state = *m_implementation;
    if (state.file)
    {
        return fail(
            "perf jitdump writer is already open",
            diagnostic);
    }
    if (!supported())
    {
        return fail(
            "perf jitdump is unsupported on this host",
            diagnostic);
    }
    if (outputPath.empty())
    {
        return fail(
            "perf jitdump output path is empty",
            diagnostic);
    }

#if defined(__linux__)
    const int descriptor = ::open(
        outputPath.c_str(),
        O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC,
        0666);
    if (descriptor < 0)
    {
        return fail(
            errnoMessage("opening perf jitdump"),
            diagnostic);
    }
    std::FILE *const file = fdopen(descriptor, "w+b");
    if (!file)
    {
        const std::string message =
            errnoMessage("opening perf jitdump stream");
        ::close(descriptor);
        return fail(message, diagnostic);
    }

    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0)
    {
        const std::string message =
            errnoMessage("reading host page size");
        std::fclose(file);
        return fail(message, diagnostic);
    }
    void *const marker = mmap(
        nullptr, static_cast<size_t>(pageSize),
        PROT_READ | PROT_EXEC,
        MAP_PRIVATE, descriptor, 0);
    if (marker == MAP_FAILED)
    {
        const std::string message =
            errnoMessage("mapping perf jitdump marker");
        std::fclose(file);
        return fail(message, diagnostic);
    }

    const uint64_t timestamp = state.timestamp();
    if (timestamp == 0u)
    {
        (void)munmap(
            marker, static_cast<size_t>(pageSize));
        std::fclose(file);
        return fail(
            "reading CLOCK_MONOTONIC for perf jitdump failed",
            diagnostic);
    }
    const JitDumpHeader header{
        .elfMachine = hostElfMachine(),
        .processId =
            static_cast<uint32_t>(getpid()),
        .timestamp = timestamp,
    };
    if (std::fwrite(
            &header, sizeof(header), 1u, file) != 1u ||
        std::fflush(file) != 0)
    {
        const std::string message =
            errnoMessage("writing perf jitdump header");
        (void)munmap(
            marker, static_cast<size_t>(pageSize));
        std::fclose(file);
        return fail(message, diagnostic);
    }

    state.outputPath = outputPath;
    state.file = file;
    state.marker = marker;
    state.markerSize =
        static_cast<size_t>(pageSize);
    state.nextCodeIndex = 1u;
    state.failed = false;
    if (diagnostic)
        diagnostic->clear();
    return true;
#else
    (void)outputPath;
    return fail(
        "perf jitdump is unsupported on this host",
        diagnostic);
#endif
}

bool PS2PerfJitDumpWriter::registerCode(
    const void *code, size_t sizeBytes,
    std::string_view symbol, uint64_t *codeIndex,
    std::string *diagnostic)
{
    if (codeIndex)
        *codeIndex = 0u;
    std::lock_guard<std::mutex> lock(
        m_implementation->mutex);
    Implementation &state = *m_implementation;
    if (!state.file)
    {
        return fail(
            "perf jitdump writer is not open",
            diagnostic);
    }
    if (state.failed)
    {
        return fail(
            "perf jitdump writer failed previously",
            diagnostic);
    }
    if (!code || sizeBytes == 0u)
    {
        return fail(
            "perf jitdump code range must be nonempty",
            diagnostic);
    }
    if (symbol.empty() ||
        symbol.find('\0') != std::string_view::npos)
    {
        return fail(
            "perf jitdump symbol must be a nonempty C string",
            diagnostic);
    }
    constexpr size_t kMaximumPayload =
        static_cast<size_t>(
            std::numeric_limits<uint32_t>::max()) -
        sizeof(JitDumpCodeLoad);
    if (symbol.size() + 1u > kMaximumPayload ||
        sizeBytes >
            kMaximumPayload - (symbol.size() + 1u))
    {
        return fail(
            "perf jitdump code-load record is too large",
            diagnostic);
    }
    if (state.nextCodeIndex == 0u)
    {
        return fail(
            "perf jitdump code-index space is exhausted",
            diagnostic);
    }

#if defined(__linux__)
    const uint64_t timestamp = state.timestamp();
    if (timestamp == 0u)
    {
        return fail(
            "reading CLOCK_MONOTONIC for perf jitdump failed",
            diagnostic);
    }
    const uint64_t assignedIndex =
        state.nextCodeIndex++;
    const uintptr_t address =
        reinterpret_cast<uintptr_t>(code);
    const JitDumpCodeLoad record{
        .header = {
            .id = kJitCodeLoad,
            .totalSize = static_cast<uint32_t>(
                sizeof(JitDumpCodeLoad) +
                symbol.size() + 1u + sizeBytes),
            .timestamp = timestamp,
        },
        .processId =
            static_cast<uint32_t>(getpid()),
        .threadId = static_cast<uint32_t>(
            syscall(SYS_gettid)),
        .virtualAddress =
            static_cast<uint64_t>(address),
        .codeAddress =
            static_cast<uint64_t>(address),
        .codeSize =
            static_cast<uint64_t>(sizeBytes),
        .codeIndex = assignedIndex,
    };
    const char terminator = '\0';
    const bool written =
        std::fwrite(
            &record, sizeof(record), 1u,
            state.file) == 1u &&
        std::fwrite(
            symbol.data(), symbol.size(), 1u,
            state.file) == 1u &&
        std::fwrite(
            &terminator, sizeof(terminator), 1u,
            state.file) == 1u &&
        std::fwrite(
            code, sizeBytes, 1u,
            state.file) == 1u &&
        std::fflush(state.file) == 0;
    if (!written)
    {
        state.failed = true;
        return fail(
            errnoMessage("writing perf jitdump code load"),
            diagnostic);
    }
    if (codeIndex)
        *codeIndex = assignedIndex;
    if (diagnostic)
        diagnostic->clear();
    return true;
#else
    (void)code;
    (void)sizeBytes;
    (void)symbol;
    return fail(
        "perf jitdump is unsupported on this host",
        diagnostic);
#endif
}

bool PS2PerfJitDumpWriter::close(
    std::string *diagnostic)
{
    std::lock_guard<std::mutex> lock(
        m_implementation->mutex);
    Implementation &state = *m_implementation;
    if (!state.file)
    {
        if (diagnostic)
            diagnostic->clear();
        return true;
    }

    bool success = !state.failed;
#if defined(__linux__)
    if (success)
    {
        const uint64_t timestamp = state.timestamp();
        const JitDumpRecordHeader record{
            .id = kJitCodeClose,
            .totalSize =
                static_cast<uint32_t>(
                    sizeof(JitDumpRecordHeader)),
            .timestamp = timestamp,
        };
        success =
            timestamp != 0u &&
            std::fwrite(
                &record, sizeof(record), 1u,
                state.file) == 1u &&
            std::fflush(state.file) == 0;
    }
#endif
    if (std::fclose(state.file) != 0)
        success = false;
    state.file = nullptr;
#if defined(__linux__)
    if (state.marker)
    {
        if (munmap(
                state.marker,
                state.markerSize) != 0)
        {
            success = false;
        }
        state.marker = nullptr;
        state.markerSize = 0u;
    }
#endif
    if (!success)
    {
        state.failed = true;
        return fail(
            "closing perf jitdump failed",
            diagnostic);
    }
    if (diagnostic)
        diagnostic->clear();
    return true;
}

bool PS2PerfJitDumpWriter::isOpen() const
{
    std::lock_guard<std::mutex> lock(
        m_implementation->mutex);
    return m_implementation->file != nullptr;
}

std::filesystem::path
PS2PerfJitDumpWriter::outputPath() const
{
    std::lock_guard<std::mutex> lock(
        m_implementation->mutex);
    return m_implementation->outputPath;
}

bool ps2PerfJitDumpEnvironmentRequested() noexcept
{
    return environmentFlagEnabled(
        std::getenv("PS2X_PERF_JITDUMP"));
}

namespace
{
    struct ProcessWriter
    {
        PS2PerfJitDumpWriter writer;
        std::string error;
        bool ready = false;

        ProcessWriter()
        {
            if (!ps2PerfJitDumpEnvironmentRequested())
            {
                error =
                    "PS2X_PERF_JITDUMP is not enabled";
                return;
            }
            std::filesystem::path directory =
                std::filesystem::current_path();
            if (const char *const configured =
                    std::getenv(
                        "PS2X_PERF_JITDUMP_DIR");
                configured && configured[0] != '\0')
            {
                directory = configured;
            }
            std::error_code directoryError;
            std::filesystem::create_directories(
                directory, directoryError);
            if (directoryError)
            {
                error =
                    "creating perf jitdump directory: " +
                    directoryError.message();
                return;
            }
#if defined(__linux__)
            const std::filesystem::path output =
                directory /
                ("jit-" +
                 std::to_string(
                     static_cast<uint64_t>(
                         getpid())) +
                 ".dump");
#else
            const std::filesystem::path output =
                directory / "jit-unsupported.dump";
#endif
            ready = writer.open(output, &error);
        }
    };

    ProcessWriter &processWriter()
    {
        static ProcessWriter writer;
        return writer;
    }
}

bool ps2PerfJitDumpInitialize(
    std::string *diagnostic)
{
    ProcessWriter &process = processWriter();
    if (!process.ready)
        return fail(process.error, diagnostic);
    if (diagnostic)
        diagnostic->clear();
    return true;
}

bool ps2PerfJitDumpRegisterCode(
    const void *code, size_t sizeBytes,
    std::string_view symbol, uint64_t *codeIndex,
    std::string *diagnostic)
{
    ProcessWriter &process = processWriter();
    if (!process.ready)
        return fail(process.error, diagnostic);
    return process.writer.registerCode(
        code, sizeBytes, symbol,
        codeIndex, diagnostic);
}
