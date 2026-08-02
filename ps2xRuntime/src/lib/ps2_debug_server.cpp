#include "runtime/ps2_debug_server.h"

#include "ThreadNaming.h"
#include "Kernel/Stubs/Pad.h"
#include "Kernel/Syscalls/Helpers/State.h"
#include "ps2_runtime.h"
#include "ps2_syscalls.h"
#include "runtime/ps2_vu_program_cache.h"
#include "runtime/ps2_vu_recompiler.h"
#include "runtime/ps2_watchdog.h"

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "raylib.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <condition_variable>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace
{
    constexpr uint32_t kProtocolVersion = 1u;
    constexpr size_t kMaxMessageBytes = 16u * 1024u * 1024u;
    constexpr uint64_t kMaxTimeoutMs = 86400000u;
    constexpr uint64_t kDefaultWatchdogIntervalMs = 1000u;
    constexpr uint64_t kDefaultWatchdogThresholdMs = 15000u;
    constexpr size_t kStatusBlockProfileRecords = 16u;
    constexpr size_t kMaximumBlockProfilePageRecords = 256u;

    using Document = rapidjson::Document;
    using Value = rapidjson::Value;
    using Allocator = Document::AllocatorType;

    class RequestError final : public std::runtime_error
    {
    public:
        RequestError(int codeValue, std::string message)
            : std::runtime_error(std::move(message)), code(codeValue)
        {
        }

        int code;
    };

    Value makeString(std::string_view text, Allocator &allocator)
    {
        return Value(text.data(), static_cast<rapidjson::SizeType>(text.size()), allocator);
    }

    void addString(Value &object,
                   const char *key,
                   std::string_view text,
                   Allocator &allocator)
    {
        object.AddMember(Value(key, allocator), makeString(text, allocator), allocator);
    }

    std::string serialize(const Document &document)
    {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        document.Accept(writer);
        return std::string(buffer.GetString(), buffer.GetSize());
    }

    std::string addressString(uint32_t address)
    {
        std::ostringstream output;
        output << "0x" << std::hex << std::setw(8) << std::setfill('0') << address;
        return output.str();
    }

    std::string hostAddressString(uint64_t address)
    {
        std::ostringstream output;
        output
            << "0x" << std::hex << std::setw(16)
            << std::setfill('0') << address;
        return output.str();
    }

    std::string wordString(const uint32_t *words, size_t count)
    {
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (size_t index = 0u; index < count; ++index)
        {
            output << std::setw(8) << words[index];
        }
        return output.str();
    }

    std::string halfwordString(const uint16_t *words, size_t count)
    {
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (size_t index = 0u; index < count; ++index)
        {
            output << std::setw(4)
                   << static_cast<uint32_t>(words[index]);
        }
        return output.str();
    }

    std::string hexBytes(const void *data, size_t size)
    {
        static constexpr char digits[] = "0123456789abcdef";
        const auto *bytes = static_cast<const uint8_t *>(data);
        std::string output(size * 2u, '\0');
        for (size_t index = 0; index < size; ++index)
        {
            output[index * 2u] = digits[bytes[index] >> 4u];
            output[index * 2u + 1u] = digits[bytes[index] & 0x0fu];
        }
        return output;
    }

    class Sha256
    {
    public:
        Sha256()
            : m_state{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u}
        {
        }

        void update(const uint8_t *data, size_t size)
        {
            m_total += size;
            while (size != 0u)
            {
                const size_t count = std::min(size, m_block.size() - m_used);
                std::memcpy(m_block.data() + m_used, data, count);
                m_used += count;
                data += count;
                size -= count;
                if (m_used == m_block.size())
                {
                    transform(m_block.data());
                    m_used = 0u;
                }
            }
        }

        std::array<uint8_t, 32> finish()
        {
            const uint64_t bitCount = static_cast<uint64_t>(m_total) * 8u;
            m_block[m_used++] = 0x80u;
            if (m_used > 56u)
            {
                std::fill(m_block.begin() + static_cast<std::ptrdiff_t>(m_used),
                          m_block.end(), 0u);
                transform(m_block.data());
                m_used = 0u;
            }
            std::fill(m_block.begin() + static_cast<std::ptrdiff_t>(m_used),
                      m_block.begin() + 56, 0u);
            for (size_t index = 0; index < 8u; ++index)
            {
                m_block[63u - index] =
                    static_cast<uint8_t>(bitCount >> (index * 8u));
            }
            transform(m_block.data());

            std::array<uint8_t, 32> output{};
            for (size_t index = 0; index < m_state.size(); ++index)
            {
                output[index * 4u] = static_cast<uint8_t>(m_state[index] >> 24u);
                output[index * 4u + 1u] = static_cast<uint8_t>(m_state[index] >> 16u);
                output[index * 4u + 2u] = static_cast<uint8_t>(m_state[index] >> 8u);
                output[index * 4u + 3u] = static_cast<uint8_t>(m_state[index]);
            }
            return output;
        }

    private:
        static uint32_t rotateRight(uint32_t value, uint32_t count)
        {
            return (value >> count) | (value << (32u - count));
        }

        void transform(const uint8_t *block)
        {
            static constexpr std::array<uint32_t, 64> constants = {
                0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
                0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
                0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
                0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
                0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
                0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
                0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
                0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
                0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
                0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
                0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
                0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
                0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
                0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
                0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
                0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

            std::array<uint32_t, 64> words{};
            for (size_t index = 0; index < 16u; ++index)
            {
                words[index] =
                    (static_cast<uint32_t>(block[index * 4u]) << 24u) |
                    (static_cast<uint32_t>(block[index * 4u + 1u]) << 16u) |
                    (static_cast<uint32_t>(block[index * 4u + 2u]) << 8u) |
                    static_cast<uint32_t>(block[index * 4u + 3u]);
            }
            for (size_t index = 16u; index < words.size(); ++index)
            {
                const uint32_t s0 =
                    rotateRight(words[index - 15u], 7u) ^
                    rotateRight(words[index - 15u], 18u) ^
                    (words[index - 15u] >> 3u);
                const uint32_t s1 =
                    rotateRight(words[index - 2u], 17u) ^
                    rotateRight(words[index - 2u], 19u) ^
                    (words[index - 2u] >> 10u);
                words[index] =
                    words[index - 16u] + s0 + words[index - 7u] + s1;
            }

            uint32_t a = m_state[0], b = m_state[1], c = m_state[2], d = m_state[3];
            uint32_t e = m_state[4], f = m_state[5], g = m_state[6], h = m_state[7];
            for (size_t index = 0; index < words.size(); ++index)
            {
                const uint32_t sum1 =
                    rotateRight(e, 6u) ^ rotateRight(e, 11u) ^ rotateRight(e, 25u);
                const uint32_t temp1 =
                    h + sum1 + ((e & f) ^ (~e & g)) + constants[index] + words[index];
                const uint32_t sum0 =
                    rotateRight(a, 2u) ^ rotateRight(a, 13u) ^ rotateRight(a, 22u);
                const uint32_t temp2 =
                    sum0 + ((a & b) ^ (a & c) ^ (b & c));
                h = g;
                g = f;
                f = e;
                e = d + temp1;
                d = c;
                c = b;
                b = a;
                a = temp1 + temp2;
            }
            m_state[0] += a;
            m_state[1] += b;
            m_state[2] += c;
            m_state[3] += d;
            m_state[4] += e;
            m_state[5] += f;
            m_state[6] += g;
            m_state[7] += h;
        }

        std::array<uint32_t, 8> m_state;
        std::array<uint8_t, 64> m_block{};
        size_t m_used = 0u;
        size_t m_total = 0u;
    };

    std::string digest(const void *data, size_t size)
    {
        Sha256 hash;
        hash.update(static_cast<const uint8_t *>(data), size);
        const auto result = hash.finish();
        return hexBytes(result.data(), result.size());
    }

    std::string fileDigest(const std::filesystem::path &path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
        {
            throw RequestError(-32003, "cannot read " + path.string());
        }
        Sha256 hash;
        std::array<uint8_t, 1024u * 1024u> buffer{};
        while (input)
        {
            input.read(reinterpret_cast<char *>(buffer.data()),
                       static_cast<std::streamsize>(buffer.size()));
            if (input.gcount() > 0)
            {
                hash.update(buffer.data(), static_cast<size_t>(input.gcount()));
            }
        }
        const auto result = hash.finish();
        return hexBytes(result.data(), result.size());
    }

    void writeBytes(const std::filesystem::path &path,
                    const void *data,
                    size_t size)
    {
        if (size != 0u && data == nullptr)
        {
            throw RequestError(
                -32003, "cannot write unavailable data to " + path.string());
        }
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output)
        {
            throw RequestError(-32003, "cannot create " + path.string());
        }
        output.write(static_cast<const char *>(data),
                     static_cast<std::streamsize>(size));
        if (!output)
        {
            throw RequestError(-32003, "cannot write " + path.string());
        }
    }

    void writeJson(const std::filesystem::path &path, const Document &document)
    {
        const std::string text = serialize(document) + "\n";
        writeBytes(path, text.data(), text.size());
    }

    const Value *paramsFor(const Document &request)
    {
        const auto member = request.FindMember("params");
        if (member == request.MemberEnd())
        {
            return nullptr;
        }
        if (!member->value.IsObject())
        {
            throw RequestError(-32602, "params must be an object");
        }
        return &member->value;
    }

    const Value &required(const Value *params, const char *name)
    {
        if (!params)
        {
            throw RequestError(-32602, std::string("missing parameter ") + name);
        }
        const auto member = params->FindMember(name);
        if (member == params->MemberEnd())
        {
            throw RequestError(-32602, std::string("missing parameter ") + name);
        }
        return member->value;
    }

    std::string requiredString(const Value *params, const char *name)
    {
        const Value &value = required(params, name);
        if (!value.IsString())
        {
            throw RequestError(-32602, std::string(name) + " must be a string");
        }
        return std::string(value.GetString(), value.GetStringLength());
    }

    uint64_t requiredUnsigned(const Value *params, const char *name)
    {
        const Value &value = required(params, name);
        if (!value.IsUint64())
        {
            throw RequestError(-32602,
                               std::string(name) + " must be an unsigned integer");
        }
        return value.GetUint64();
    }

    uint32_t parseAddress(std::string_view text)
    {
        if (text.empty())
        {
            throw RequestError(-32602, "address must not be empty");
        }
        std::string copy(text);
        char *end = nullptr;
        errno = 0;
        const unsigned long long parsed = std::strtoull(copy.c_str(), &end, 0);
        if (errno != 0 || end != copy.c_str() + copy.size() ||
            parsed > std::numeric_limits<uint32_t>::max())
        {
            throw RequestError(-32602, "invalid 32-bit address " + copy);
        }
        return static_cast<uint32_t>(parsed);
    }

    uint64_t timeoutMs(const Value *params)
    {
        const Value &bounds = required(params, "bounds");
        if (!bounds.IsObject())
        {
            throw RequestError(-32602, "bounds must be an object");
        }
        const auto timeout = bounds.FindMember("timeout_ms");
        if (timeout == bounds.MemberEnd())
        {
            return 30000u;
        }
        if (!timeout->value.IsUint64())
        {
            throw RequestError(-32602, "timeout_ms must be an unsigned integer");
        }
        return std::clamp<uint64_t>(timeout->value.GetUint64(), 1u, kMaxTimeoutMs);
    }

    std::string gprValue(const __m128i &value)
    {
        alignas(16) std::array<uint64_t, 2> lanes{};
        _mm_storeu_si128(reinterpret_cast<__m128i *>(lanes.data()), value);
        std::ostringstream output;
        output << "0x" << std::hex << std::setw(16) << std::setfill('0') << lanes[1]
               << std::setw(16) << lanes[0];
        return output.str();
    }

    const char *gsEventName(GSDebugEventKind kind)
    {
        switch (kind)
        {
        case GSDebugEventKind::GifTag:
            return "gif-tag";
        case GSDebugEventKind::Register:
            return "register";
        case GSDebugEventKind::Draw:
            return "draw";
        case GSDebugEventKind::Transfer:
            return "transfer";
        case GSDebugEventKind::Present:
            return "present";
        default:
            return "unknown";
        }
    }

    uint64_t boundedEnvironmentInteger(const char *name,
                                       uint64_t fallback,
                                       uint64_t minimum,
                                       uint64_t maximum)
    {
        const char *value = std::getenv(name);
        if (!value || value[0] == '\0')
        {
            return fallback;
        }
        char *end = nullptr;
        errno = 0;
        const unsigned long long parsed = std::strtoull(value, &end, 0);
        if (errno != 0 || end == value || *end != '\0')
        {
            return fallback;
        }
        return std::clamp<uint64_t>(
            static_cast<uint64_t>(parsed), minimum, maximum);
    }

    const char *threadStatusName(int status)
    {
        switch (status)
        {
        case 0x01:
            return "running";
        case 0x02:
            return "ready";
        case 0x04:
            return "waiting";
        case 0x08:
            return "suspended";
        case 0x0c:
            return "waiting-suspended";
        case 0x10:
            return "dormant";
        default:
            return "unknown";
        }
    }

    const char *threadWaitName(int waitType)
    {
        switch (waitType)
        {
        case 0:
            return "none";
        case 1:
            return "sleep";
        case 2:
            return "semaphore";
        case 3:
            return "event";
        default:
            return "unknown";
        }
    }

    const char *branchKindName(PS2Runtime::GuestBranchKind kind)
    {
        switch (kind)
        {
        case PS2Runtime::GuestBranchKind::DirectJump:
            return "direct-jump";
        case PS2Runtime::GuestBranchKind::DirectCall:
            return "direct-call";
        case PS2Runtime::GuestBranchKind::IndirectJump:
            return "indirect-jump";
        case PS2Runtime::GuestBranchKind::IndirectCall:
            return "indirect-call";
        case PS2Runtime::GuestBranchKind::Return:
            return "return";
        default:
            return "unknown";
        }
    }
}

struct PS2DebugServer::Impl
{
    explicit Impl(PS2Runtime &runtimeValue) : runtime(runtimeValue)
    {
    }

    PS2Runtime &runtime;
    std::thread thread;
    std::thread watchdogThread;
    std::atomic<bool> stopped{true};
    std::string socketPath;
    std::atomic<int> serverSocket{-1};
    std::atomic<int> clientSocket{-1};
    std::mutex socketMutex;
    std::mutex captureMutex;
    std::mutex watchdogWaitMutex;
    std::condition_variable watchdogWaitCv;
    mutable std::mutex watchdogStateMutex;
    PS2WatchdogSample watchdogSample{};
    PS2WatchdogAssessment watchdogAssessment{};
    uint64_t watchdogSampleSequence = 0u;
    uint64_t watchdogNoPresentationMs = 0u;
    uint64_t watchdogLastCapturedFault = 0u;
    std::string watchdogLastBundle;
    std::string watchdogLastError;

    struct QuiesceGuard
    {
        explicit QuiesceGuard(PS2Runtime &runtimeValue,
                              std::chrono::milliseconds timeout =
                                  std::chrono::seconds(30))
            : runtime(runtimeValue), resumeOnExit(!runtimeValue.debugIsPaused())
        {
            if (!runtime.debugPause(timeout))
            {
                throw RequestError(-32002, "timed out waiting for a guest safe point");
            }
        }

        ~QuiesceGuard()
        {
            if (resumeOnExit)
            {
                runtime.debugResume();
            }
        }

        PS2Runtime &runtime;
        bool resumeOnExit;
    };

    bool environmentFlagEnabled(const char *name) const
    {
        const char *value = std::getenv(name);
        if (!value || value[0] == '\0')
        {
            return false;
        }
        std::string normalized(value);
        std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                       [](unsigned char ch)
                       { return static_cast<char>(std::tolower(ch)); });
        return normalized != "0" && normalized != "false" && normalized != "off";
    }

    bool enabled() const
    {
        if (const char *path = std::getenv("PS2DBG_SOCKET"); path && path[0] != '\0')
        {
            return true;
        }
        return environmentFlagEnabled("PS2DBG_ENABLE");
    }

    std::string defaultSocketPath() const
    {
        if (const char *path = std::getenv("PS2DBG_SOCKET"); path && path[0] != '\0')
        {
            return path;
        }
        const char *runtimeDirectory = std::getenv("XDG_RUNTIME_DIR");
        const std::filesystem::path base =
            runtimeDirectory && runtimeDirectory[0] != '\0'
                ? std::filesystem::path(runtimeDirectory)
                : std::filesystem::path("/tmp");
        return (base / "ps2dbg" / "recomp.sock").string();
    }

    std::filesystem::path crashDirectory() const
    {
        if (const char *path = std::getenv("PS2DBG_CRASH_DIR");
            path && path[0] != '\0')
        {
            return std::filesystem::absolute(path);
        }
        return std::filesystem::temp_directory_path() /
               "ps2dbg" / "crashes";
    }

    PS2WatchdogSample takeWatchdogSample()
    {
        const PS2Runtime::DebugRuntimeProgress core =
            runtime.debugRuntimeProgress();
        const GSProgressSnapshot gs = runtime.gs().getProgressSnapshot();
        const VuProgressSnapshot vu0 =
            runtime.vu0().getProgressSnapshot();
        const VuProgressSnapshot vu1 =
            runtime.vu1().getProgressSnapshot();
        const auto threads =
            ps2_syscalls::debugThreadSnapshots(&runtime);

        PS2WatchdogSample sample{};
        sample.dispatches = core.dispatches;
        sample.eeInstructions = core.eeInstructions;
        sample.pc = core.pc;
        sample.dmaStarts = runtime.memory().dmaStartCount();
        sample.gifCopies = runtime.memory().gifCopyCount();
        sample.gsWrites = runtime.memory().gsWriteCount();
        sample.vifWrites = runtime.memory().vifWriteCount();
        sample.presentations = gs.presentations;
        sample.gsDrawsStarted = gs.drawsStarted;
        sample.gsDrawsCompleted = gs.drawsCompleted;
        sample.gsCandidatePixels = gs.candidatePixels;
        sample.gsActiveDraws = gs.activeDraws;
        sample.vu0Cycles = vu0.cycles;
        sample.vu1Cycles = vu1.cycles;
        sample.vu0Active = vu0.active;
        sample.vu1Active = vu1.active;
        sample.guestExecutionWaiters = core.guestExecutionWaiters;
        for (const auto &guestThread : threads)
        {
            if (guestThread.terminated || guestThread.status == 0x10)
            {
                continue;
            }
            ++sample.guestThreadCount;
            if (guestThread.status == 0x04 ||
                guestThread.status == 0x0c)
            {
                ++sample.waitingThreadCount;
            }
        }
        return sample;
    }

    Value watchdogStatus(Allocator &allocator)
    {
        PS2WatchdogSample sample{};
        PS2WatchdogAssessment assessment{};
        uint64_t sequence = 0u;
        uint64_t noPresentationMs = 0u;
        std::string lastBundle;
        std::string lastError;
        {
            std::lock_guard<std::mutex> lock(watchdogStateMutex);
            sample = watchdogSample;
            assessment = watchdogAssessment;
            sequence = watchdogSampleSequence;
            noPresentationMs = watchdogNoPresentationMs;
            lastBundle = watchdogLastBundle;
            lastError = watchdogLastError;
        }

        Value result(rapidjson::kObjectType);
        result.AddMember("sample_sequence", sequence, allocator);
        addString(result, "classification", assessment.name, allocator);
        result.AddMember("made_progress", assessment.madeProgress, allocator);
        addString(result, "hot_operation", assessment.hotOperation, allocator);
        result.AddMember("no_presentation_ms", noPresentationMs, allocator);
        result.AddMember("dispatches", sample.dispatches, allocator);
        result.AddMember("ee_instructions", sample.eeInstructions, allocator);
        addString(result, "pc", addressString(sample.pc), allocator);
        result.AddMember("presentations", sample.presentations, allocator);
        result.AddMember("gs_draws_started", sample.gsDrawsStarted, allocator);
        result.AddMember("gs_draws_completed", sample.gsDrawsCompleted, allocator);
        result.AddMember("gs_active_draws", sample.gsActiveDraws, allocator);
        result.AddMember("gs_candidate_pixels", sample.gsCandidatePixels, allocator);
        result.AddMember("vu0_cycles", sample.vu0Cycles, allocator);
        result.AddMember("vu1_cycles", sample.vu1Cycles, allocator);
        result.AddMember("vu0_active", sample.vu0Active, allocator);
        result.AddMember("vu1_active", sample.vu1Active, allocator);
        result.AddMember("guest_threads", sample.guestThreadCount, allocator);
        result.AddMember("waiting_threads", sample.waitingThreadCount, allocator);
        result.AddMember(
            "guest_execution_waiters", sample.guestExecutionWaiters, allocator);
        addString(result, "last_bundle", lastBundle, allocator);
        addString(result, "last_error", lastError, allocator);
        return result;
    }

    Value blockProfileRecord(
        const PS2Runtime::DebugVuRecompilerDiagnostics::
            BlockProfile &block,
        Allocator &allocator)
    {
        Value result(rapidjson::kObjectType);
        result.AddMember(
            "compilation_id",
            block.compilationIdentity,
            allocator);
        result.AddMember(
            "code_content_identity",
            block.codeContentIdentity,
            allocator);
        result.AddMember(
            "compilation_generation",
            block.compilationGeneration,
            allocator);
        addString(
            result, "native_address",
            hostAddressString(block.nativeAddress),
            allocator);
        result.AddMember(
            "native_bytes",
            block.nativeBytes,
            allocator);
        addString(
            result, "entry_pc",
            addressString(block.entryPc),
            allocator);
        addString(
            result, "first_pc",
            addressString(block.firstPc),
            allocator);
        addString(
            result, "last_pc",
            addressString(block.lastPc),
            allocator);
        result.AddMember(
            "block_pairs",
            block.blockPairs,
            allocator);
        result.AddMember(
            "fixed_cycles",
            block.fixedCycles,
            allocator);
        addString(
            result, "compilation_mode",
            block.compilationMode,
            allocator);
        addString(
            result, "block_form",
            block.blockForm,
            allocator);
        addString(
            result, "block_exit",
            block.blockExit,
            allocator);
        result.AddMember(
            "resident",
            block.resident,
            allocator);
        result.AddMember(
            "executions",
            block.executions,
            allocator);
        result.AddMember(
            "guest_pairs",
            block.guestPairs,
            allocator);
        result.AddMember(
            "full_budget_entries",
            block.fullBudgetEntries,
            allocator);
        result.AddMember(
            "bounded_entries",
            block.boundedEntries,
            allocator);
        result.AddMember(
            "linked_edges",
            block.linkedEdges,
            allocator);
        result.AddMember(
            "helper_barriers",
            block.helperBarriers,
            allocator);
        result.AddMember(
            "resident_vf_registers",
            block.residentVfRegisters,
            allocator);
        result.AddMember(
            "resident_vf_dirty_registers",
            block.residentVfDirtyRegisters,
            allocator);
        result.AddMember(
            "resident_vf_dirty_lanes",
            block.residentVfDirtyLanes,
            allocator);
        result.AddMember(
            "maximum_live_vf_registers",
            block.maximumLiveVfRegisters,
            allocator);
        result.AddMember(
            "vf_accesses",
            block.vfAccesses,
            allocator);
        result.AddMember(
            "allocated_vf_accesses",
            block.allocatedVfAccesses,
            allocator);
        result.AddMember(
            "vf_register_loads",
            block.vfRegisterLoads,
            allocator);
        result.AddMember(
            "vf_register_stores",
            block.vfRegisterStores,
            allocator);
        result.AddMember(
            "vf_register_spills",
            block.vfRegisterSpills,
            allocator);
        result.AddMember(
            "vf_register_reloads",
            block.vfRegisterReloads,
            allocator);

        Value materializations(
            rapidjson::kObjectType);
        static_assert(
            kVuRegisterMaterializationCauseCount ==
            std::tuple_size_v<
                decltype(
                    block.registerMaterializations)>);
        for (size_t cause = 0u;
             cause <
                 block.registerMaterializations.size();
             ++cause)
        {
            Value causeName =
                makeString(
                    vuRegisterMaterializationCauseName(
                        static_cast<
                            VuRegisterMaterializationCause>(
                            cause)),
                    allocator);
            materializations.AddMember(
                causeName,
                block.registerMaterializations[cause],
                allocator);
        }
        result.AddMember(
            "register_materializations",
            materializations,
            allocator);

        Value exits(rapidjson::kObjectType);
        static_assert(
            kVuNativeBlockExitCount ==
            std::tuple_size_v<
                decltype(block.exitReasons)>);
        for (size_t exit = 0u;
             exit < block.exitReasons.size();
             ++exit)
        {
            Value exitName =
                makeString(
                    vuNativeBlockExitName(
                        static_cast<
                            VuNativeBlockExit>(
                            exit)),
                    allocator);
            exits.AddMember(
                exitName,
                block.exitReasons[exit],
                allocator);
        }
        result.AddMember(
            "exit_reasons", exits,
            allocator);

        Value opcodes(rapidjson::kArrayType);
        for (const auto &opcode :
             block.opcodeCounts)
        {
            Value opcodeValue(
                rapidjson::kObjectType);
            addString(
                opcodeValue, "name",
                opcode.name, allocator);
            opcodeValue.AddMember(
                "operations",
                opcode.operations,
                allocator);
            opcodes.PushBack(
                opcodeValue, allocator);
        }
        result.AddMember(
            "opcodes", opcodes,
            allocator);

        Value registrations(
            rapidjson::kArrayType);
        for (const auto &registration :
             block.jitRegistrations)
        {
            Value registrationValue(
                rapidjson::kObjectType);
            registrationValue.AddMember(
                "generation",
                registration.generation,
                allocator);
            registrationValue.AddMember(
                "code_index",
                registration.codeIndex,
                allocator);
            registrations.PushBack(
                registrationValue,
                allocator);
        }
        result.AddMember(
            "jit_registrations",
            registrations, allocator);
        return result;
    }

    Value status(Allocator &allocator)
    {
        const PS2Runtime::DebugRuntimeProgress core =
            runtime.debugRuntimeProgress();
        const GSProgressSnapshot gsProgress =
            runtime.gs().getProgressSnapshot();
        const VuProgressSnapshot vu0Progress =
            runtime.vu0().getProgressSnapshot();
        const VuProgressSnapshot vu1Progress =
            runtime.vu1().getProgressSnapshot();
        const auto vuDiagnostics =
            runtime.debugVuBackendDiagnosticsSnapshot();
        Value result(rapidjson::kObjectType);
        addString(result, "backend", "recomp", allocator);
        addString(
            result,
            "ee_execution_backend",
            runtime.eeExecutionBackendName(),
            allocator);
        result.AddMember(
            "ee_execution_threads",
            static_cast<uint64_t>(
                runtime
                    .managedEeExecutionThreadCountForTesting()),
            allocator);
        const char *state = runtime.isStopRequested()
                                ? "stopped"
                                : (runtime.debugIsPaused() ? "paused" : "running");
        addString(result, "state", state, allocator);
        result.AddMember("protocol_version", kProtocolVersion, allocator);
        result.AddMember("vm_valid", runtime.memory().getRDRAM() != nullptr, allocator);
        addString(result, "pc",
                  addressString(core.pc),
                  allocator);
        addString(result, "symbol",
                  PS2Runtime::formatGuestPc(core.pc),
                  allocator);
        result.AddMember("function_table_base",
                         g_ps2RecompiledFunctionTableBase, allocator);
        result.AddMember("function_table_end",
                         g_ps2RecompiledFunctionTableEnd, allocator);
        result.AddMember("function_table_slots",
                         g_ps2RecompiledFunctionTableSlotCount, allocator);
        result.AddMember("guest_symbols", g_ps2GuestFunctionSymbolCount, allocator);

        Value progress(rapidjson::kObjectType);
        progress.AddMember("dispatches", core.dispatches, allocator);
        progress.AddMember("ee_instructions", core.eeInstructions, allocator);
        progress.AddMember("dma_starts", runtime.memory().dmaStartCount(), allocator);
        progress.AddMember("gif_copies", runtime.memory().gifCopyCount(), allocator);
        progress.AddMember("gs_writes", runtime.memory().gsWriteCount(), allocator);
        progress.AddMember("vif_writes", runtime.memory().vifWriteCount(), allocator);
        progress.AddMember("presentations", gsProgress.presentations, allocator);
        progress.AddMember("gs_draws_started", gsProgress.drawsStarted, allocator);
        progress.AddMember("gs_draws_completed", gsProgress.drawsCompleted, allocator);
        progress.AddMember("gs_active_draws", gsProgress.activeDraws, allocator);
        progress.AddMember("gs_candidate_pixels", gsProgress.candidatePixels, allocator);
        progress.AddMember("vu0_cycles", vu0Progress.cycles, allocator);
        progress.AddMember("vu1_cycles", vu1Progress.cycles, allocator);
        result.AddMember("progress", progress, allocator);

        const auto vuBackendStatus =
            [&](const VuUnit &unit,
                const PS2Runtime::DebugVuBackendDiagnostics &diagnostics)
        {
            Value value(rapidjson::kObjectType);
            addString(
                value, "requested",
                vuBackendKindName(unit.requestedBackend()), allocator);
            addString(
                value, "resolved",
                vuBackendKindName(unit.resolvedBackend()), allocator);
            addString(value, "name", unit.backendName(), allocator);
            value.AddMember("active", unit.isActive(), allocator);
            value.AddMember(
                "native_instrumentation",
                unit.nativeInstrumentationEnabled(), allocator);
            value.AddMember(
                "diagnostics_captured",
                diagnostics.captured, allocator);
            if (!diagnostics.captured)
                return value;

            value.AddMember(
                "diagnostics_snapshot_sequence",
                diagnostics.snapshotSequence, allocator);
            addString(
                value, "pc",
                addressString(diagnostics.pc), allocator);
            addString(
                value, "last_exit_reason",
                vuExitReasonName(diagnostics.lastExitReason),
                allocator);
            value.AddMember(
                "issued_cycles",
                diagnostics.issuedCycles, allocator);
            if (diagnostics.cacheCreated)
            {
                const auto &source = diagnostics.cache;
                Value cache(rapidjson::kObjectType);
                cache.AddMember("hits", source.hits, allocator);
                cache.AddMember("misses", source.misses, allocator);
                cache.AddMember(
                    "compilations", source.compilations, allocator);
                cache.AddMember(
                    "invalidations", source.invalidations, allocator);
                cache.AddMember(
                    "invalidated_programs",
                    source.invalidatedPrograms, allocator);
                cache.AddMember(
                    "generation_retentions",
                    source.generationRetentions, allocator);
                cache.AddMember(
                    "retained_programs",
                    source.retainedPrograms, allocator);
                cache.AddMember(
                    "cross_generation_hits",
                    source.crossGenerationHits, allocator);
                cache.AddMember(
                    "eviction_flushes",
                    source.evictionFlushes, allocator);
                cache.AddMember(
                    "evicted_programs",
                    source.evictedPrograms, allocator);
                cache.AddMember(
                    "manual_flushes",
                    source.manualFlushes, allocator);
                cache.AddMember(
                    "rejected_programs",
                    source.rejectedPrograms, allocator);
                cache.AddMember(
                    "link_resolutions",
                    source.linkResolutions, allocator);
                cache.AddMember(
                    "link_resolution_failures",
                    source.linkResolutionFailures,
                    allocator);
                cache.AddMember(
                    "link_invalidations",
                    source.linkInvalidations, allocator);
                cache.AddMember(
                    "generated_bytes",
                    source.generatedBytes, allocator);
                cache.AddMember(
                    "compilation_time_ns",
                    source.compilationNanoseconds, allocator);
                cache.AddMember(
                    "resident_programs",
                    source.residentPrograms, allocator);
                cache.AddMember(
                    "resident_executable_bytes",
                    source.residentExecutableBytes, allocator);
                cache.AddMember(
                    "high_water_programs",
                    source.highWaterPrograms, allocator);
                cache.AddMember(
                    "high_water_executable_bytes",
                    source.highWaterExecutableBytes, allocator);
                value.AddMember("cache", cache, allocator);
            }
            if (diagnostics.recompilerCreated)
            {
                const auto &source = diagnostics.recompiler;
                Value recompiler(rapidjson::kObjectType);
                recompiler.AddMember(
                    "block_linking_enabled",
                    source.blockLinkingEnabled,
                    allocator);
                recompiler.AddMember(
                    "block_budget_guards_enabled",
                    source.blockBudgetGuardsEnabled,
                    allocator);
                recompiler.AddMember(
                    "block_local_vf_registers_enabled",
                    source.blockLocalVfRegistersEnabled,
                    allocator);
                recompiler.AddMember(
                    "block_local_vf_registers_automatic",
                    source.blockLocalVfRegistersAutomatic,
                    allocator);
                recompiler.AddMember(
                    "inline_xgkick_enabled",
                    source.inlineXgkickEnabled,
                    allocator);
                recompiler.AddMember(
                    "native_entries",
                    source.nativeEntries, allocator);
                recompiler.AddMember(
                    "native_blocks",
                    source.nativeBlocks, allocator);
                recompiler.AddMember(
                    "native_pairs",
                    source.nativePairs, allocator);
                recompiler.AddMember(
                    "instrumented_native_entries",
                    source.instrumentedNativeEntries,
                    allocator);
                recompiler.AddMember(
                    "instrumented_native_blocks",
                    source.instrumentedNativeBlocks,
                    allocator);
                recompiler.AddMember(
                    "instrumented_native_pairs",
                    source.instrumentedNativePairs,
                    allocator);
                recompiler.AddMember(
                    "inline_pairs",
                    source.inlinePairs, allocator);
                recompiler.AddMember(
                    "helper_pairs",
                    source.helperPairs, allocator);
                recompiler.AddMember(
                    "linked_edges",
                    source.linkedEdges, allocator);
                recompiler.AddMember(
                    "slow_link_exits",
                    source.slowLinkExits, allocator);
                recompiler.AddMember(
                    "resolved_links",
                    source.resolvedLinks, allocator);
                recompiler.AddMember(
                    "abandoned_link_resolutions",
                    source.abandonedLinkResolutions,
                    allocator);
                recompiler.AddMember(
                    "incompatible_link_exits",
                    source.incompatibleLinkExits,
                    allocator);
                recompiler.AddMember(
                    "full_block_guards",
                    source.fullBlockGuards,
                    allocator);
                recompiler.AddMember(
                    "precise_tail_entries",
                    source.preciseTailEntries,
                    allocator);
                recompiler.AddMember(
                    "full_guard_pairs",
                    source.fullGuardPairs,
                    allocator);
                recompiler.AddMember(
                    "precise_tail_pairs",
                    source.preciseTailPairs,
                    allocator);
                recompiler.AddMember(
                    "budget_guard_comparisons",
                    source.budgetGuardComparisons,
                    allocator);
                recompiler.AddMember(
                    "native_budget_exits",
                    source.nativeBudgetExits,
                    allocator);
                recompiler.AddMember(
                    "block_completes",
                    source.blockCompletes, allocator);
                recompiler.AddMember(
                    "cycle_budget_exits",
                    source.cycleBudgetExits, allocator);
                recompiler.AddMember(
                    "xgkick_exits",
                    source.xgkickExits, allocator);
                recompiler.AddMember(
                    "xgkick_advance_helper_calls",
                    source.xgkickAdvanceHelperCalls, allocator);
                recompiler.AddMember(
                    "unsupported_exits",
                    source.unsupportedExits, allocator);
                recompiler.AddMember(
                    "code_invalidation_exits",
                    source.codeInvalidationExits, allocator);
                recompiler.AddMember(
                    "fault_exits",
                    source.faultExits, allocator);
                Value nativeExits(
                    rapidjson::kObjectType);
                static_assert(
                    kVuNativeBlockExitCount ==
                    std::tuple_size_v<
                        decltype(
                            source.nativeExitReasons)>);
                for (size_t exit = 0u;
                     exit <
                         source.nativeExitReasons.size();
                     ++exit)
                {
                    Value exitName =
                        makeString(
                            vuNativeBlockExitName(
                                static_cast<
                                    VuNativeBlockExit>(
                                    exit)),
                            allocator);
                    nativeExits.AddMember(
                        exitName,
                        source.nativeExitReasons[exit],
                        allocator);
                }
                recompiler.AddMember(
                    "native_exit_reasons",
                    nativeExits, allocator);
                recompiler.AddMember(
                    "interpreter_instrumentation_fallbacks",
                    source.interpreterInstrumentationFallbacks,
                    allocator);
                recompiler.AddMember(
                    "interpreter_fallback_pairs",
                    source.interpreterFallbackPairs, allocator);
                recompiler.AddMember(
                    "code_image_identities",
                    source.codeImageIdentities, allocator);
                recompiler.AddMember(
                    "code_image_reuses",
                    source.codeImageReuses, allocator);
                recompiler.AddMember(
                    "code_image_catalog_evictions",
                    source.codeImageCatalogEvictions,
                    allocator);
                recompiler.AddMember(
                    "jitdump_registrations",
                    source.jitDumpRegistrations,
                    allocator);
                recompiler.AddMember(
                    "jitdump_failures",
                    source.jitDumpFailures,
                    allocator);
                if (!source.lastJitDiagnostic.empty())
                {
                    addString(
                        recompiler,
                        "last_jitdump_diagnostic",
                        source.lastJitDiagnostic,
                        allocator);
                }

                Value blockProfile(
                    rapidjson::kObjectType);
                blockProfile.AddMember(
                    "enabled",
                    source.blockProfileEnabled,
                    allocator);
                blockProfile.AddMember(
                    "maximum_records",
                    source.blockProfileMaximumRecords,
                    allocator);
                blockProfile.AddMember(
                    "dropped_records",
                    source.blockProfileDroppedRecords,
                    allocator);
                blockProfile.AddMember(
                    "record_count",
                    static_cast<uint64_t>(
                        source.blockProfiles.size()),
                    allocator);
                const size_t returnedRecords =
                    std::min(
                        source.blockProfiles.size(),
                        kStatusBlockProfileRecords);
                blockProfile.AddMember(
                    "returned_records",
                    static_cast<uint64_t>(
                        returnedRecords),
                    allocator);
                blockProfile.AddMember(
                    "records_truncated",
                    returnedRecords <
                        source.blockProfiles.size(),
                    allocator);
                Value blocks(rapidjson::kArrayType);
                for (size_t index = 0u;
                     index < returnedRecords;
                     ++index)
                {
                    blocks.PushBack(
                        blockProfileRecord(
                            source.blockProfiles[index],
                            allocator),
                        allocator);
                }
                blockProfile.AddMember(
                    "blocks", blocks, allocator);
                recompiler.AddMember(
                    "block_profile",
                    blockProfile, allocator);
                value.AddMember(
                    "recompiler", recompiler, allocator);
            }
            const auto &verifySource =
                diagnostics.verify;
            Value verify(rapidjson::kObjectType);
            verify.AddMember(
                "runs", verifySource.runs, allocator);
            verify.AddMember(
                "compared_pairs",
                verifySource.comparedPairs, allocator);
            verify.AddMember(
                "published_pairs",
                verifySource.publishedPairs, allocator);
            verify.AddMember(
                "published_path1_packets",
                verifySource.publishedPath1Packets,
                allocator);
            verify.AddMember(
                "mismatches",
                verifySource.mismatches, allocator);
            if (!verifySource.lastMismatch.empty())
            {
                addString(
                    verify, "last_mismatch",
                    verifySource.lastMismatch, allocator);
            }
            value.AddMember(
                "verify", verify, allocator);
            return value;
        };
        Value vuBackends(rapidjson::kObjectType);
        vuBackends.AddMember(
            "vu0",
            vuBackendStatus(runtime.vu0(), vuDiagnostics[0u]),
            allocator);
        vuBackends.AddMember(
            "vu1",
            vuBackendStatus(runtime.vu1(), vuDiagnostics[1u]),
            allocator);
        result.AddMember("vu_backends", vuBackends, allocator);

        const Vu1WorkloadProfileSnapshot vu1Profile =
            runtime.memory().vu1WorkloadProfileSnapshot();
        Value profile(rapidjson::kObjectType);
        profile.AddMember("enabled", vu1Profile.enabled, allocator);
        profile.AddMember(
            "measurement_complete",
            vu1Profile.measurementComplete, allocator);
        profile.AddMember(
            "warmup_pairs", vu1Profile.warmupPairs, allocator);
        profile.AddMember(
            "pair_limit", vu1Profile.pairLimit, allocator);
        profile.AddMember(
            "observed_pairs", vu1Profile.observedPairs, allocator);
        profile.AddMember(
            "measured_pairs", vu1Profile.measuredPairs, allocator);
        profile.AddMember(
            "invocations", vu1Profile.invocations, allocator);
        profile.AddMember(
            "code_uploads", vu1Profile.codeUploads, allocator);
        profile.AddMember(
            "identical_code_uploads",
            vu1Profile.identicalCodeUploads, allocator);
        profile.AddMember(
            "code_upload_bytes",
            vu1Profile.codeUploadBytes, allocator);
        profile.AddMember(
            "identical_code_upload_bytes",
            vu1Profile.identicalCodeUploadBytes, allocator);
        result.AddMember("vu1_workload_profile", profile, allocator);
        result.AddMember("watchdog", watchdogStatus(allocator), allocator);

        const PS2Runtime::DebugFaultInfo fault = runtime.debugFaultSnapshot();
        if (fault.active)
        {
            Value faultValue(rapidjson::kObjectType);
            faultValue.AddMember("sequence", fault.sequence, allocator);
            addString(faultValue, "type", fault.type, allocator);
            addString(faultValue, "operation", fault.operation, allocator);
            addString(faultValue, "target",
                      addressString(fault.targetPc), allocator);
            result.AddMember("fault", faultValue, allocator);
        }
        return result;
    }

    Value vuBlockProfile(
        const Value *params,
        Allocator &allocator)
    {
        if (!runtime.debugIsPaused())
        {
            throw RequestError(
                -32002,
                "VU block profiles require a paused guest");
        }
        const std::string unitName =
            requiredString(params, "unit");
        size_t unitIndex = 0u;
        if (unitName == "vu0")
        {
            unitIndex = 0u;
        }
        else if (unitName == "vu1")
        {
            unitIndex = 1u;
        }
        else
        {
            throw RequestError(
                -32602,
                "unit must be vu0 or vu1");
        }
        const uint64_t offsetValue =
            requiredUnsigned(params, "offset");
        const uint64_t limitValue =
            requiredUnsigned(params, "limit");
        if (offsetValue >
                std::numeric_limits<size_t>::max() ||
            limitValue == 0u ||
            limitValue >
                kMaximumBlockProfilePageRecords)
        {
            throw RequestError(
                -32602,
                "offset is too large or limit is outside 1..256");
        }

        const auto snapshots =
            runtime.debugVuBackendDiagnosticsSnapshot();
        const auto &diagnostics =
            snapshots[unitIndex];
        if (!diagnostics.captured ||
            !diagnostics.recompilerCreated)
        {
            throw RequestError(
                -32003,
                "VU recompiler diagnostics are unavailable");
        }
        const auto &source =
            diagnostics.recompiler;
        const size_t offset =
            static_cast<size_t>(offsetValue);
        if (offset > source.blockProfiles.size())
        {
            throw RequestError(
                -32602,
                "offset exceeds the block profile record count");
        }
        const size_t returned =
            std::min(
                static_cast<size_t>(limitValue),
                source.blockProfiles.size() -
                    offset);
        const size_t nextOffset =
            offset + returned;

        Value result(rapidjson::kObjectType);
        result.AddMember(
            "schema_version", 1u, allocator);
        addString(
            result, "unit", unitName,
            allocator);
        result.AddMember(
            "snapshot_sequence",
            diagnostics.snapshotSequence,
            allocator);
        result.AddMember(
            "enabled",
            source.blockProfileEnabled,
            allocator);
        result.AddMember(
            "maximum_records",
            source.blockProfileMaximumRecords,
            allocator);
        result.AddMember(
            "dropped_records",
            source.blockProfileDroppedRecords,
            allocator);
        result.AddMember(
            "record_count",
            static_cast<uint64_t>(
                source.blockProfiles.size()),
            allocator);
        result.AddMember(
            "offset",
            static_cast<uint64_t>(offset),
            allocator);
        result.AddMember(
            "returned_records",
            static_cast<uint64_t>(returned),
            allocator);
        result.AddMember(
            "next_offset",
            static_cast<uint64_t>(nextOffset),
            allocator);
        result.AddMember(
            "complete",
            nextOffset ==
                source.blockProfiles.size(),
            allocator);
        Value blocks(rapidjson::kArrayType);
        for (size_t index = offset;
             index < nextOffset;
             ++index)
        {
            blocks.PushBack(
                blockProfileRecord(
                    source.blockProfiles[index],
                    allocator),
                allocator);
        }
        result.AddMember(
            "blocks", blocks, allocator);
        return result;
    }

    Value hello(Allocator &allocator)
    {
        Value result = status(allocator);
        Value capabilities(rapidjson::kArrayType);
        for (const char *capability : {
                 "system.status", "execution.pause", "execution.resume",
                 "execution.shutdown", "execution.runUntil:ee.pc",
                 "execution.step:dispatch", "state.registers:ee",
                 "state.registers:iop", "state.registers:vu0",
                 "state.registers:vu1", "memory.read", "memory.hash",
                 "breakpoint:ee", "watchpoint:ee", "capture",
                 "trace.vu0-sync", "trace.vu0-instruction",
                 "trace.ee-events",
                 "diagnostics.status", "watchdog.status",
                 "input.pad.status", "input.pad.override",
                 "vu.backend.set", "vu.block-profile"})
        {
            capabilities.PushBack(Value(capability, allocator), allocator);
        }
        result.AddMember("capabilities", capabilities, allocator);
        return result;
    }

    std::string functionMapDigest() const
    {
        std::ostringstream canonical;
        canonical << "base=" << g_ps2RecompiledFunctionTableBase << '\n'
                  << "end=" << g_ps2RecompiledFunctionTableEnd << '\n'
                  << "slots=" << g_ps2RecompiledFunctionTableSlotCount << '\n';
        for (uint32_t slot = 0u;
             slot < g_ps2RecompiledFunctionTableSlotCount;
             ++slot)
        {
            canonical << (g_ps2RecompiledFunctionTable[slot] ? '1' : '0');
        }
        canonical << '\n';
        for (uint32_t index = 0u;
             index < g_ps2GuestFunctionSymbolCount;
             ++index)
        {
            const PS2GuestFunctionSymbol &symbol =
                g_ps2GuestFunctionSymbols[index];
            canonical << symbol.start << ',' << symbol.end << ','
                      << (symbol.name ? symbol.name : "") << '\n';
        }
        const std::string text = canonical.str();
        return digest(text.data(), text.size());
    }

    std::string executableDigest() const
    {
        const std::filesystem::path path =
            runtime.ioPaths().elfPath;
        std::error_code error;
        if (path.empty() || !std::filesystem::is_regular_file(path, error) ||
            error)
        {
            return {};
        }
        try
        {
            return fileDigest(path);
        }
        catch (...)
        {
            return {};
        }
    }

    Value diagnosticsValue(std::string_view reason,
                           bool quiescent,
                           const PS2WatchdogSample &sample,
                           const PS2WatchdogAssessment &assessment,
                           Allocator &allocator)
    {
        const PS2Runtime::DebugRuntimeProgress core =
            runtime.debugRuntimeProgress();
        const PS2Runtime::DebugEeTiming eeTiming =
            runtime.debugEeTimingSnapshot();
        const PS2Runtime::DebugEeThreadDiagnostics
            eeThreadDiagnostics =
                runtime.debugEeThreadDiagnosticsSnapshot();
        const PS2Runtime::DebugFaultInfo fault =
            runtime.debugFaultSnapshot();
        const auto branches = runtime.debugBranchHistory(256u);
        auto threads =
            ps2_syscalls::debugThreadSnapshots(&runtime);
        const PS2AudioStreamDebugSnapshot audio =
            runtime.audioBackend().streamDebugSnapshot();

        Value result(rapidjson::kObjectType);
        result.AddMember("schema_version", 1, allocator);
        addString(result, "backend", "recomp", allocator);
        addString(
            result,
            "ee_execution_backend",
            runtime.eeExecutionBackendName(),
            allocator);
        result.AddMember(
            "ee_execution_threads",
            static_cast<uint64_t>(
                runtime
                    .managedEeExecutionThreadCountForTesting()),
            allocator);
        addString(result, "reason", reason, allocator);
        result.AddMember("quiescent", quiescent, allocator);
        addString(result, "classification", assessment.name, allocator);
        addString(result, "hot_operation", assessment.hotOperation, allocator);
        addString(result, "pc", addressString(core.pc), allocator);
        addString(result, "symbol", PS2Runtime::formatGuestPc(core.pc), allocator);
        addString(result, "ra", addressString(core.ra), allocator);
        addString(result, "sp", addressString(core.sp), allocator);
        addString(result, "gp", addressString(core.gp), allocator);
        addString(result, "function_map_sha256", functionMapDigest(), allocator);
        addString(result, "executable_sha256", executableDigest(), allocator);
        addString(result, "executable",
                  runtime.ioPaths().elfPath.string(),
                  allocator);

        Value progress(rapidjson::kObjectType);
        progress.AddMember("dispatches", sample.dispatches, allocator);
        progress.AddMember("ee_instructions", sample.eeInstructions, allocator);
        progress.AddMember("ee_tick", eeTiming.currentTick, allocator);
        progress.AddMember("ee_cycle", eeTiming.currentCycle, allocator);
        progress.AddMember("vsync_fields", core.vsyncFields, allocator);
        progress.AddMember(
            "mpeg_pictures_served",
            core.mpegPicturesServed,
            allocator);
        progress.AddMember(
            "mpeg_unique_pictures_served",
            core.mpegUniquePicturesServed,
            allocator);
        progress.AddMember(
            "mpeg_repeated_pictures_served",
            core.mpegRepeatedPicturesServed,
            allocator);
        progress.AddMember("dma_starts", sample.dmaStarts, allocator);
        progress.AddMember("gif_copies", sample.gifCopies, allocator);
        progress.AddMember("gs_writes", sample.gsWrites, allocator);
        progress.AddMember("vif_writes", sample.vifWrites, allocator);
        progress.AddMember("presentations", sample.presentations, allocator);
        progress.AddMember("gs_draws_started", sample.gsDrawsStarted, allocator);
        progress.AddMember("gs_draws_completed", sample.gsDrawsCompleted, allocator);
        progress.AddMember("gs_active_draws", sample.gsActiveDraws, allocator);
        progress.AddMember("gs_candidate_pixels", sample.gsCandidatePixels, allocator);
        progress.AddMember("vu0_cycles", sample.vu0Cycles, allocator);
        progress.AddMember("vu1_cycles", sample.vu1Cycles, allocator);
        progress.AddMember("vu0_active", sample.vu0Active, allocator);
        progress.AddMember("vu1_active", sample.vu1Active, allocator);
        progress.AddMember("guest_execution_waiters",
                           sample.guestExecutionWaiters, allocator);
        progress.AddMember("guest_execution_handoff_timeouts",
                           core.guestExecutionHandoffTimeouts, allocator);
        result.AddMember("progress", progress, allocator);

        Value eeThreadValue(rapidjson::kObjectType);
        eeThreadValue.AddMember(
            "enabled", eeThreadDiagnostics.enabled, allocator);
        eeThreadValue.AddMember(
            "guest_lock_requests",
            eeThreadDiagnostics.guestLockRequests,
            allocator);
        eeThreadValue.AddMember(
            "guest_lock_acquisitions",
            eeThreadDiagnostics.guestLockAcquisitions,
            allocator);
        eeThreadValue.AddMember(
            "guest_lock_contentions",
            eeThreadDiagnostics.guestLockContentions,
            allocator);
        eeThreadValue.AddMember(
            "outer_guest_execution_acquisitions",
            eeThreadDiagnostics.outerGuestExecutionAcquisitions,
            allocator);
        eeThreadValue.AddMember(
            "guest_context_changes",
            eeThreadDiagnostics.guestContextChanges,
            allocator);
        eeThreadValue.AddMember(
            "handoff_notifications",
            eeThreadDiagnostics.handoffNotifications,
            allocator);
        eeThreadValue.AddMember(
            "handoff_wait_requests",
            eeThreadDiagnostics.handoffWaitRequests,
            allocator);
        eeThreadValue.AddMember(
            "handoff_wait_fast_paths",
            eeThreadDiagnostics.handoffWaitFastPaths,
            allocator);
        eeThreadValue.AddMember(
            "handoff_cv_waits",
            eeThreadDiagnostics.handoffCvWaits,
            allocator);
        eeThreadValue.AddMember(
            "handoff_completions",
            eeThreadDiagnostics.handoffCompletions,
            allocator);
        eeThreadValue.AddMember(
            "handoff_timeouts",
            eeThreadDiagnostics.handoffTimeouts,
            allocator);
        eeThreadValue.AddMember(
            "yield_requests",
            eeThreadDiagnostics.yieldRequests,
            allocator);
        eeThreadValue.AddMember(
            "deferred_yields",
            eeThreadDiagnostics.deferredYields,
            allocator);
        eeThreadValue.AddMember(
            "host_thread_yields",
            eeThreadDiagnostics.hostThreadYields,
            allocator);
        eeThreadValue.AddMember(
            "requested_guest_switches",
            eeThreadDiagnostics.requestedGuestSwitches,
            allocator);
        eeThreadValue.AddMember(
            "guest_switch_cv_waits",
            eeThreadDiagnostics.guestSwitchCvWaits,
            allocator);
        eeThreadValue.AddMember(
            "completed_guest_switches",
            eeThreadDiagnostics.completedGuestSwitches,
            allocator);
        eeThreadValue.AddMember(
            "guest_switch_timeouts",
            eeThreadDiagnostics.guestSwitchTimeouts,
            allocator);
        eeThreadValue.AddMember(
            "rotation_requests",
            eeThreadDiagnostics.rotationRequests,
            allocator);
        eeThreadValue.AddMember(
            "accepted_rotation_requests",
            eeThreadDiagnostics.acceptedRotationRequests,
            allocator);
        eeThreadValue.AddMember(
            "rejected_rotation_requests",
            eeThreadDiagnostics.rejectedRotationRequests,
            allocator);
        eeThreadValue.AddMember(
            "priority_zero_rotation_requests",
            eeThreadDiagnostics.priorityZeroRotationRequests,
            allocator);
        eeThreadValue.AddMember(
            "untracked_thread_rotation_requests",
            eeThreadDiagnostics.untrackedThreadRotationRequests,
            allocator);

        Value rotationsByPriority(rapidjson::kArrayType);
        for (size_t priority = 0u;
             priority <
                 eeThreadDiagnostics.acceptedRotationsByPriority.size();
             ++priority)
        {
            const uint64_t count =
                eeThreadDiagnostics
                    .acceptedRotationsByPriority[priority];
            if (count == 0u)
            {
                continue;
            }
            Value entry(rapidjson::kObjectType);
            entry.AddMember(
                "priority",
                static_cast<uint32_t>(priority),
                allocator);
            entry.AddMember("count", count, allocator);
            rotationsByPriority.PushBack(entry, allocator);
        }
        eeThreadValue.AddMember(
            "accepted_rotations_by_priority",
            rotationsByPriority,
            allocator);

        Value rotationsByThread(rapidjson::kArrayType);
        for (size_t thread = 0u;
             thread <
                 eeThreadDiagnostics.acceptedRotationsByThread.size();
             ++thread)
        {
            const uint64_t count =
                eeThreadDiagnostics
                    .acceptedRotationsByThread[thread];
            if (count == 0u)
            {
                continue;
            }
            Value entry(rapidjson::kObjectType);
            entry.AddMember(
                "thread_id",
                static_cast<uint32_t>(thread),
                allocator);
            entry.AddMember("count", count, allocator);
            rotationsByThread.PushBack(entry, allocator);
        }
        eeThreadValue.AddMember(
            "accepted_rotations_by_thread",
            rotationsByThread,
            allocator);
        result.AddMember(
            "ee_thread_diagnostics",
            eeThreadValue,
            allocator);

        Value faultValue(rapidjson::kObjectType);
        faultValue.AddMember("active", fault.active, allocator);
        if (fault.active)
        {
            faultValue.AddMember("sequence", fault.sequence, allocator);
            addString(faultValue, "type", fault.type, allocator);
            addString(faultValue, "operation", fault.operation, allocator);
            addString(faultValue, "branch_kind",
                      branchKindName(fault.branchKind), allocator);
            addString(faultValue, "source",
                      addressString(fault.sourcePc), allocator);
            addString(faultValue, "target",
                      addressString(fault.targetPc), allocator);
            addString(faultValue, "pc", addressString(fault.pc), allocator);
            addString(faultValue, "ra", addressString(fault.ra), allocator);
            addString(faultValue, "sp", addressString(fault.sp), allocator);
            addString(faultValue, "gp", addressString(fault.gp), allocator);
            addString(faultValue, "a0", addressString(fault.a0), allocator);
            addString(faultValue, "a1", addressString(fault.a1), allocator);
            addString(faultValue, "v0", addressString(fault.v0), allocator);
            addString(faultValue, "v1", addressString(fault.v1), allocator);
            faultValue.AddMember("code_region", fault.codeRegion, allocator);
            faultValue.AddMember(
                "policy", static_cast<uint32_t>(fault.policy), allocator);
        }
        result.AddMember("fault", faultValue, allocator);

        Value branchValues(rapidjson::kArrayType);
        for (const PS2Runtime::DebugBranchEntry &branch : branches)
        {
            Value entry(rapidjson::kObjectType);
            entry.AddMember("sequence", branch.sequence, allocator);
            addString(entry, "pc", addressString(branch.pc), allocator);
            addString(entry, "symbol",
                      PS2Runtime::formatGuestPc(branch.pc), allocator);
            branchValues.PushBack(entry, allocator);
        }
        result.AddMember("branches", branchValues, allocator);

        Value rpcValue(rapidjson::kObjectType);
        Value rpcEvents(rapidjson::kArrayType);
        std::vector<SifRpcDebugEvent> rpcHistory;
        uint64_t nextRpcSequence = 0u;
        {
            EeRpcRuntimeState &rpcState =
                runtime.eeRpcRuntimeState();
            std::lock_guard<std::mutex> lock(rpcState.rpcMutex);
            nextRpcSequence = rpcState.nextDebugSequence;
            rpcHistory.reserve(kSifRpcDebugHistoryCount);
            for (const SifRpcDebugEvent &event : rpcState.debugHistory)
            {
                if (event.seq != 0u)
                {
                    rpcHistory.push_back(event);
                }
            }
        }
        std::sort(
            rpcHistory.begin(), rpcHistory.end(),
            [](const SifRpcDebugEvent &left,
               const SifRpcDebugEvent &right)
            {
                return left.seq < right.seq;
            });
        for (const SifRpcDebugEvent &event : rpcHistory)
        {
            Value entry(rapidjson::kObjectType);
            entry.AddMember("sequence", event.seq, allocator);
            addString(entry, "operation",
                      event.op ? event.op : "", allocator);
            addString(entry, "pc", addressString(event.pc), allocator);
            addString(entry, "ra", addressString(event.ra), allocator);
            entry.AddMember("thread_id", event.threadId, allocator);
            entry.AddMember("sid", event.sid, allocator);
            entry.AddMember("rpc", event.rpcNum, allocator);
            addString(entry, "client",
                      addressString(event.clientPtr), allocator);
            addString(entry, "server",
                      addressString(event.serverPtr), allocator);
            addString(entry, "send_buffer",
                      addressString(event.sendBuf), allocator);
            entry.AddMember("send_size", event.sendSize, allocator);
            addString(entry, "receive_buffer",
                      addressString(event.recvBuf), allocator);
            entry.AddMember("receive_size", event.recvSize, allocator);
            addString(entry, "result_pointer",
                      addressString(event.resultPtr), allocator);
            entry.AddMember("mode", event.mode, allocator);
            addString(entry, "end_function",
                      addressString(event.endFunc), allocator);
            addString(entry, "end_parameter",
                      addressString(event.endParam), allocator);
            entry.AddMember("semaphore_id", event.semaId, allocator);
            entry.AddMember("flags", event.flags, allocator);
            entry.AddMember("result", event.result, allocator);
            addString(
                entry, "send_preview",
                hexBytes(event.sendPreview, event.sendPreviewSize),
                allocator);
            addString(
                entry, "receive_preview",
                hexBytes(event.recvPreview, event.recvPreviewSize),
                allocator);
            rpcEvents.PushBack(entry, allocator);
        }
        rpcValue.AddMember(
            "next_sequence", nextRpcSequence, allocator);
        rpcValue.AddMember("events", rpcEvents, allocator);
        result.AddMember("sif_rpc", rpcValue, allocator);

        bool hasMainThread = false;
        Value threadValues(rapidjson::kArrayType);
        for (const ps2_syscalls::GuestThreadDebugSnapshot &thread : threads)
        {
            hasMainThread = hasMainThread || thread.id == 1;
            Value entry(rapidjson::kObjectType);
            entry.AddMember("id", thread.id, allocator);
            addString(entry, "pc", addressString(thread.pc), allocator);
            addString(entry, "entry", addressString(thread.entry), allocator);
            addString(entry, "stack", addressString(thread.stack), allocator);
            entry.AddMember("stack_size", thread.stackSize, allocator);
            addString(entry, "gp", addressString(thread.gp), allocator);
            entry.AddMember("status_code", thread.status, allocator);
            addString(entry, "status",
                      threadStatusName(thread.status), allocator);
            entry.AddMember("wait_type_code", thread.waitType, allocator);
            addString(entry, "wait_type",
                      threadWaitName(thread.waitType), allocator);
            entry.AddMember("wait_id", thread.waitId, allocator);
            entry.AddMember("wakeup_count", thread.wakeupCount, allocator);
            entry.AddMember("priority", thread.currentPriority, allocator);
            entry.AddMember("suspend_count", thread.suspendCount, allocator);
            entry.AddMember("wait_queue_code", thread.waitQueue, allocator);
            entry.AddMember("wait_queue_id", thread.waitQueueId, allocator);
            entry.AddMember(
                "wait_completion_pending",
                thread.waitCompletionPending,
                allocator);
            entry.AddMember(
                "state_revision",
                thread.stateRevision,
                allocator);
            entry.AddMember("state_valid", thread.stateValid, allocator);
            entry.AddMember("started", thread.started, allocator);
            entry.AddMember("force_release", thread.forceRelease, allocator);
            entry.AddMember("terminated", thread.terminated, allocator);
            threadValues.PushBack(entry, allocator);
        }
        if (!hasMainThread)
        {
            Value main(rapidjson::kObjectType);
            main.AddMember("id", 1, allocator);
            addString(main, "pc", addressString(core.pc), allocator);
            addString(main, "entry", addressString(0u), allocator);
            addString(main, "stack", addressString(core.sp), allocator);
            addString(main, "gp", addressString(core.gp), allocator);
            addString(main, "status",
                      runtime.debugIsPaused() ? "paused" : "running",
                      allocator);
            main.AddMember("state_valid", false, allocator);
            main.AddMember("synthetic", true, allocator);
            threadValues.PushBack(main, allocator);
        }
        result.AddMember("threads", threadValues, allocator);

        Value audioValue(rapidjson::kObjectType);
        audioValue.AddMember("opened", audio.opened, allocator);
        audioValue.AddMember("playing", audio.playing, allocator);
        audioValue.AddMember("debugger_paused", audio.debuggerPaused, allocator);
        audioValue.AddMember("sample_rate", audio.sampleRate, allocator);
        audioValue.AddMember("channels", audio.channelCount, allocator);
        audioValue.AddMember("channel_buffer_size",
                             audio.channelBufferSize, allocator);
        audioValue.AddMember("submissions", audio.submissionCount, allocator);
        audioValue.AddMember("submitted_bytes", audio.submittedBytes, allocator);
        audioValue.AddMember("last_submission_hash",
                             audio.lastSubmissionHash, allocator);
        audioValue.AddMember(
            "produced_frames", audio.producedFrames, allocator);
        audioValue.AddMember(
            "requested_frames", audio.requestedFrames, allocator);
        audioValue.AddMember(
            "consumed_frames", audio.consumedFrames, allocator);
        audioValue.AddMember(
            "zero_filled_frames", audio.zeroFilledFrames, allocator);
        audioValue.AddMember(
            "queued_samples", static_cast<uint64_t>(audio.queuedSamples), allocator);
        Value sfxValue(rapidjson::kObjectType);
        sfxValue.AddMember("opened", audio.sfxOpened, allocator);
        sfxValue.AddMember("playing", audio.sfxPlaying, allocator);
        sfxValue.AddMember(
            "banks", static_cast<uint64_t>(audio.sfxBankCount), allocator);
        sfxValue.AddMember(
            "sounds", static_cast<uint64_t>(audio.sfxSoundCount), allocator);
        sfxValue.AddMember(
            "decoded_samples",
            static_cast<uint64_t>(audio.sfxDecodedSampleCount),
            allocator);
        sfxValue.AddMember(
            "active_handlers",
            static_cast<uint64_t>(audio.sfxActiveHandlerCount),
            allocator);
        sfxValue.AddMember(
            "active_voices",
            static_cast<uint64_t>(audio.sfxActiveVoiceCount),
            allocator);
        sfxValue.AddMember("rendered_frames", audio.sfxRenderedFrames, allocator);
        sfxValue.AddMember("nonzero_frames", audio.sfxNonzeroFrames, allocator);
        sfxValue.AddMember("rejected_banks", audio.sfxRejectedBanks, allocator);
        sfxValue.AddMember(
            "rejected_commands", audio.sfxRejectedCommands, allocator);
        sfxValue.AddMember(
            "unsupported_grains", audio.sfxUnsupportedGrains, allocator);
        audioValue.AddMember("sfx", sfxValue, allocator);
        result.AddMember("audio", audioValue, allocator);
        return result;
    }

    Value diagnostics(Allocator &allocator)
    {
        PS2WatchdogSample sample{};
        PS2WatchdogAssessment assessment{};
        {
            std::lock_guard<std::mutex> lock(watchdogStateMutex);
            sample = watchdogSample;
            assessment = watchdogAssessment;
        }
        return diagnosticsValue(
            "status", runtime.debugIsPaused(), sample, assessment, allocator);
    }

    Value eeEventDeviceStateValue(
        const PS2Runtime::DebugEeEventDeviceState &state,
        Allocator &allocator)
    {
        if (state.kind ==
            PS2Runtime::DebugEeEventDeviceKind::None)
        {
            return Value(rapidjson::kNullType);
        }

        Value value(rapidjson::kObjectType);
        addString(
            value, "kind",
            PS2Runtime::debugEeEventDeviceKindName(
                state.kind),
            allocator);
        value.AddMember(
            "operation_generation",
            state.operationGeneration, allocator);
        value.AddMember("active", state.active, allocator);
        value.AddMember("phase", state.phase, allocator);
        value.AddMember("stall", state.stall, allocator);
        value.AddMember("tadr", state.tadr, allocator);
        value.AddMember("madr", state.madr, allocator);
        value.AddMember("qwc", state.qwc, allocator);
        value.AddMember(
            "tags_processed", state.tagsProcessed, allocator);
        value.AddMember("pc", state.pc, allocator);
        value.AddMember(
            "last_advanced_tick",
            state.lastAdvancedTick, allocator);
        value.AddMember(
            "total_advanced_cycles",
            state.totalAdvancedCycles, allocator);
        return value;
    }

    Value eeSchedulerStatisticsValue(
        const ps2x::timing::EeEventSchedulerStatistics &statistics,
        Allocator &allocator)
    {
        Value value(rapidjson::kObjectType);
        value.AddMember("scheduled", statistics.scheduled, allocator);
        value.AddMember("replaced", statistics.replaced, allocator);
        value.AddMember("cancelled", statistics.cancelled, allocator);
        value.AddMember("serviced", statistics.serviced, allocator);
        value.AddMember("late", statistics.late, allocator);
        value.AddMember(
            "same_tick_reschedules",
            statistics.sameTickReschedules, allocator);
        value.AddMember(
            "service_limit_hits",
            statistics.serviceLimitHits, allocator);
        return value;
    }

    Value eeSchedulerSlotsValue(
        const PS2Runtime::DebugEeScheduler &scheduler,
        Allocator &allocator)
    {
        Value slots(rapidjson::kArrayType);
        for (const PS2Runtime::DebugEeEventSlot &slot :
             scheduler.slots)
        {
            Value value(rapidjson::kObjectType);
            addString(
                value, "source",
                ps2x::timing::eeEventSourceName(slot.source),
                allocator);
            value.AddMember("pending", slot.pending, allocator);
            value.AddMember("generation", slot.generation, allocator);
            value.AddMember("sequence", slot.sequence, allocator);
            if (slot.pending)
            {
                value.AddMember(
                    "deadline_tick", slot.deadlineTick, allocator);
            }
            else
            {
                value.AddMember(
                    "deadline_tick",
                    Value(rapidjson::kNullType), allocator);
            }
            value.AddMember(
                "device",
                eeEventDeviceStateValue(slot.device, allocator),
                allocator);
            slots.PushBack(value, allocator);
        }
        return slots;
    }

    Value eeSchedulerValue(
        const PS2Runtime::DebugEeScheduler &scheduler,
        Allocator &allocator)
    {
        Value value(rapidjson::kObjectType);
        addString(
            value, "mode", "event",
            allocator);
        value.AddMember(
            "current_tick", scheduler.currentTick, allocator);
        if (scheduler.hasNextDeadline)
        {
            value.AddMember(
                "next_deadline_tick",
                scheduler.nextDeadlineTick, allocator);
        }
        else
        {
            value.AddMember(
                "next_deadline_tick",
                Value(rapidjson::kNullType), allocator);
        }
        value.AddMember(
            "statistics",
            eeSchedulerStatisticsValue(
                scheduler.statistics, allocator),
            allocator);
        value.AddMember(
            "slots",
            eeSchedulerSlotsValue(scheduler, allocator),
            allocator);
        return value;
    }

    Value eeRegisters(Allocator &allocator)
    {
        const R5900Context context = runtime.debugCpuSnapshot();
        const PS2Runtime::DebugEeTiming timing =
            runtime.debugEeTimingSnapshot();
        const PS2Runtime::DebugEeScheduler scheduler =
            runtime.debugEeSchedulerSnapshot();
        static constexpr std::array<const char *, 32> names = {
            "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3",
            "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
            "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
            "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra"};

        Value result(rapidjson::kObjectType);
        addString(result, "cpu", "ee", allocator);
        addString(result, "pc", addressString(context.pc), allocator);
        result.AddMember("cycles", timing.currentCycle, allocator);
        result.AddMember(
            "cycle_ticks", timing.currentTick, allocator);
        result.AddMember(
            "vsync_tick",
            ps2_syscalls::GetCurrentVSyncTick(
                &runtime),
            allocator);
        result.AddMember(
            "local_block_cycle_ticks",
            timing.localBlockTicks, allocator);
        result.AddMember(
            "local_block_active",
            timing.localBlockActive, allocator);
        result.AddMember(
            "timing_context_bound",
            timing.contextBound, allocator);
        addString(
            result, "scheduler_mode", "event",
            allocator);
        if (scheduler.hasNextDeadline)
        {
            result.AddMember(
                "next_event_cycle_tick",
                scheduler.nextDeadlineTick, allocator);
        }
        else
        {
            result.AddMember(
                "next_event_cycle_tick",
                Value(rapidjson::kNullType), allocator);
        }
        result.AddMember(
            "scheduler_statistics",
            eeSchedulerStatisticsValue(
                scheduler.statistics, allocator),
            allocator);
        result.AddMember(
            "scheduler_slots",
            eeSchedulerSlotsValue(scheduler, allocator),
            allocator);
        result.AddMember("instructions", context.insn_count, allocator);

        Value categories(rapidjson::kArrayType);
        Value gprCategory(rapidjson::kObjectType);
        addString(gprCategory, "name", "gpr", allocator);
        gprCategory.AddMember("bits", 128, allocator);
        Value registers(rapidjson::kArrayType);
        for (size_t index = 0u; index < names.size(); ++index)
        {
            Value reg(rapidjson::kObjectType);
            addString(reg, "name", names[index], allocator);
            addString(reg, "value", gprValue(context.r[index]), allocator);
            registers.PushBack(reg, allocator);
        }
        gprCategory.AddMember("registers", registers, allocator);
        categories.PushBack(gprCategory, allocator);

        Value specialCategory(rapidjson::kObjectType);
        addString(specialCategory, "name", "special", allocator);
        specialCategory.AddMember("bits", 64, allocator);
        Value special(rapidjson::kArrayType);
        const auto addSpecial = [&](const char *name, uint64_t value)
        {
            Value reg(rapidjson::kObjectType);
            addString(reg, "name", name, allocator);
            std::ostringstream formatted;
            formatted << "0x" << std::hex << std::setw(16) << std::setfill('0')
                      << value;
            addString(reg, "value", formatted.str(), allocator);
            special.PushBack(reg, allocator);
        };
        addSpecial("hi", context.hi);
        addSpecial("lo", context.lo);
        addSpecial("hi1", context.hi1);
        addSpecial("lo1", context.lo1);
        addSpecial("sa", context.sa);
        specialCategory.AddMember("registers", special, allocator);
        categories.PushBack(specialCategory, allocator);
        result.AddMember("categories", categories, allocator);
        return result;
    }

    Value vuRegisters(bool vu1, Allocator &allocator)
    {
        QuiesceGuard guard(runtime);
        const VuExecutionState state =
            vu1 ? runtime.vu1().state() : runtime.vu0().state();
        const auto floatValue = [](float value)
        {
            Value result;
            if (std::isfinite(value))
            {
                result.SetFloat(value);
            }
            else
            {
                result.SetNull();
            }
            return result;
        };
        const auto floatBits = [](float value)
        {
            uint32_t bits = 0u;
            std::memcpy(&bits, &value, sizeof(bits));
            return bits;
        };
        Value result(rapidjson::kObjectType);
        addString(result, "cpu", vu1 ? "vu1" : "vu0", allocator);
        addString(result, "pc", addressString(state.pc), allocator);
        Value vi(rapidjson::kArrayType);
        for (size_t index = 0u; index < 16u; ++index)
        {
            Value reg(rapidjson::kObjectType);
            std::ostringstream name;
            name << "vi" << index;
            addString(reg, "name", name.str(), allocator);
            addString(reg, "value",
                      addressString(static_cast<uint32_t>(state.vi[index])),
                      allocator);
            vi.PushBack(reg, allocator);
        }
        result.AddMember("integer_registers", vi, allocator);
        Value vf(rapidjson::kArrayType);
        Value vfBits(rapidjson::kArrayType);
        for (size_t index = 0u; index < 32u; ++index)
        {
            Value vector(rapidjson::kArrayType);
            Value vectorBits(rapidjson::kArrayType);
            for (float lane : state.vf[index])
            {
                vector.PushBack(floatValue(lane), allocator);
                vectorBits.PushBack(
                    makeString(addressString(floatBits(lane)), allocator),
                    allocator);
            }
            vf.PushBack(vector, allocator);
            vfBits.PushBack(vectorBits, allocator);
        }
        result.AddMember("vector_registers", vf, allocator);
        result.AddMember("vector_register_bits", vfBits, allocator);

        Value accumulator(rapidjson::kArrayType);
        Value accumulatorBits(rapidjson::kArrayType);
        for (float lane : state.acc)
        {
            accumulator.PushBack(floatValue(lane), allocator);
            accumulatorBits.PushBack(
                makeString(addressString(floatBits(lane)), allocator),
                allocator);
        }
        result.AddMember("accumulator", accumulator, allocator);
        result.AddMember("accumulator_bits", accumulatorBits, allocator);

        result.AddMember("q", floatValue(state.q), allocator);
        result.AddMember("p", floatValue(state.p), allocator);
        result.AddMember("i", floatValue(state.i), allocator);
        addString(result, "q_bits", addressString(floatBits(state.q)), allocator);
        addString(result, "p_bits", addressString(floatBits(state.p)), allocator);
        addString(result, "i_bits", addressString(floatBits(state.i)), allocator);
        result.AddMember("mac", state.mac, allocator);
        result.AddMember("clip", state.clip, allocator);
        result.AddMember("status", state.status, allocator);
        result.AddMember("top", state.top, allocator);
        result.AddMember("itop", state.itop, allocator);
        return result;
    }

    Value iopRegisters(Allocator &allocator)
    {
        const ps2x::iop::DebugSnapshot snapshot = runtime.iopDebugSnapshot();
        Value result(rapidjson::kObjectType);
        addString(result, "cpu", "iop", allocator);
        result.AddMember("available", false, allocator);
        addString(result, "reason",
                  "PS2Recomp uses a high-level IOP subsystem without an R3000 register file",
                  allocator);
        addString(result, "active_profile", snapshot.activeProfile, allocator);
        addString(result, "active_provider", snapshot.activeProvider, allocator);
        Value diagnostics(rapidjson::kArrayType);
        for (const std::string &diagnostic : snapshot.diagnostics)
        {
            diagnostics.PushBack(makeString(diagnostic, allocator), allocator);
        }
        result.AddMember("diagnostics", diagnostics, allocator);
        return result;
    }

    Value registers(const Value *params, Allocator &allocator)
    {
        const std::string cpu = requiredString(params, "cpu");
        if (cpu == "ee")
        {
            return eeRegisters(allocator);
        }
        if (cpu == "iop")
        {
            return iopRegisters(allocator);
        }
        if (cpu == "vu0")
        {
            return vuRegisters(false, allocator);
        }
        if (cpu == "vu1")
        {
            return vuRegisters(true, allocator);
        }
        throw RequestError(-32602, "unsupported CPU " + cpu);
    }

    Value vu0SyncTraceValue(bool stop, Allocator &allocator)
    {
        const PS2Runtime::DebugVu0SyncTrace trace =
            runtime.debugVu0SyncTraceSnapshot(stop);
        Value result(rapidjson::kObjectType);
        result.AddMember("enabled", trace.enabled, allocator);
        result.AddMember("triggered", trace.triggered, allocator);
        result.AddMember("stop_on_full", trace.stopOnFull, allocator);
        if (trace.triggerEePc.has_value())
        {
            addString(
                result, "trigger_ee_pc",
                addressString(*trace.triggerEePc), allocator);
        }
        else
        {
            result.AddMember(
                "trigger_ee_pc", Value(rapidjson::kNullType), allocator);
        }
        if (trace.hasTriggerSchedulerSnapshot)
        {
            result.AddMember(
                "trigger_vsync_tick",
                trace.triggerVsyncTick, allocator);
            result.AddMember(
                "trigger_scheduler",
                eeSchedulerValue(
                    trace.triggerScheduler, allocator),
                allocator);
        }
        else
        {
            result.AddMember(
                "trigger_vsync_tick",
                Value(rapidjson::kNullType), allocator);
            result.AddMember(
                "trigger_scheduler",
                Value(rapidjson::kNullType), allocator);
        }
        result.AddMember("total_entries", trace.totalEntries, allocator);
        result.AddMember("dropped_entries", trace.droppedEntries, allocator);
        Value entries(rapidjson::kArrayType);
        for (const PS2Runtime::DebugVu0SyncEntry &source : trace.entries)
        {
            Value entry(rapidjson::kObjectType);
            entry.AddMember("sequence", source.sequence, allocator);
            entry.AddMember("invocation", source.invocation, allocator);
            entry.AddMember(
                "invocation_instruction",
                source.invocationInstruction, allocator);
            entry.AddMember("ee_cycle_ticks", source.eeCycleTicks, allocator);
            entry.AddMember("vsync_tick", source.vsyncTick, allocator);
            entry.AddMember("vu_cycle_ticks", source.vuCycleTicks, allocator);
            entry.AddMember(
                "next_event_cycle_ticks",
                source.nextEventCycleTicks, allocator);
            addString(entry, "ee_pc", addressString(source.eePc), allocator);
            entry.AddMember("cycle_budget", source.cycleBudget, allocator);
            addString(
                entry, "vu_pc_before", addressString(source.vuPcBefore),
                allocator);
            addString(
                entry, "vu_pc_after", addressString(source.vuPcAfter),
                allocator);
            addString(
                entry, "pending_vf_mask",
                addressString(source.pendingVfMask), allocator);
            addString(
                entry, "pending_vi_mask",
                addressString(source.pendingViMask), allocator);
            entry.AddMember("context_vi1", source.contextVi1, allocator);
            entry.AddMember("context_vi2", source.contextVi2, allocator);
            entry.AddMember("vu_vi1_before", source.vuVi1Before, allocator);
            entry.AddMember("vu_vi2_before", source.vuVi2Before, allocator);
            entry.AddMember("vu_vi1_after", source.vuVi1After, allocator);
            entry.AddMember("vu_vi2_after", source.vuVi2After, allocator);
            entry.AddMember("interlocked", source.interlocked, allocator);
            entry.AddMember(
                "block_boundary", source.blockBoundary, allocator);
            entry.AddMember("event_due", source.eventDue, allocator);
            entry.AddMember("active_before", source.activeBefore, allocator);
            entry.AddMember("active_after", source.activeAfter, allocator);
            entries.PushBack(entry, allocator);
        }
        result.AddMember("entries", entries, allocator);
        return result;
    }

    Value vu0SyncTraceStart(const Value *params, Allocator &allocator)
    {
        const uint64_t maximumEntries =
            requiredUnsigned(params, "maximum_entries");
        if (maximumEntries == 0u || maximumEntries > 16384u)
        {
            throw RequestError(
                -32602, "maximum_entries must be between 1 and 16384");
        }
        std::optional<uint32_t> triggerEePc;
        bool stopOnFull = false;
        if (params)
        {
            const auto trigger = params->FindMember("trigger_ee_pc");
            if (trigger != params->MemberEnd())
            {
                if (!trigger->value.IsString())
                {
                    throw RequestError(
                        -32602, "trigger_ee_pc must be an address string");
                }
                triggerEePc = parseAddress(std::string_view(
                    trigger->value.GetString(),
                    trigger->value.GetStringLength()));
            }
            const auto stop = params->FindMember("stop_on_full");
            if (stop != params->MemberEnd())
            {
                if (!stop->value.IsBool())
                {
                    throw RequestError(
                        -32602, "stop_on_full must be a boolean");
                }
                stopOnFull = stop->value.GetBool();
            }
        }
        runtime.debugStartVu0SyncTrace(
            static_cast<size_t>(maximumEntries), triggerEePc, stopOnFull);
        return vu0SyncTraceValue(false, allocator);
    }

    Value vu0InstructionTraceValue(bool stop, Allocator &allocator)
    {
        const PS2Runtime::DebugVu0InstructionTrace trace =
            runtime.debugVu0InstructionTraceSnapshot(stop);
        Value result(rapidjson::kObjectType);
        result.AddMember("enabled", trace.enabled, allocator);
        result.AddMember("triggered", trace.triggered, allocator);
        result.AddMember("stop_on_full", trace.stopOnFull, allocator);
        if (trace.triggerEePc.has_value())
        {
            addString(
                result, "trigger_ee_pc",
                addressString(*trace.triggerEePc), allocator);
        }
        else
        {
            result.AddMember(
                "trigger_ee_pc", Value(rapidjson::kNullType), allocator);
        }
        result.AddMember("total_entries", trace.totalEntries, allocator);
        result.AddMember("dropped_entries", trace.droppedEntries, allocator);
        Value entries(rapidjson::kArrayType);
        for (const PS2Runtime::DebugVu0InstructionEntry &source :
             trace.entries)
        {
            Value entry(rapidjson::kObjectType);
            entry.AddMember("sequence", source.sequence, allocator);
            entry.AddMember("invocation", source.invocation, allocator);
            entry.AddMember(
                "invocation_instruction",
                source.invocationInstruction, allocator);
            addString(entry, "pc", addressString(source.pc), allocator);
            addString(entry, "lower", addressString(source.lower), allocator);
            addString(entry, "upper", addressString(source.upper), allocator);
            addString(
                entry, "vi",
                halfwordString(source.vi.data(), source.vi.size()),
                allocator);
            addString(
                entry, "vf",
                wordString(source.vf.data(), source.vf.size()),
                allocator);
            addString(
                entry, "acc",
                wordString(source.acc.data(), source.acc.size()),
                allocator);
            addString(entry, "q", addressString(source.q), allocator);
            addString(entry, "p", addressString(source.p), allocator);
            addString(entry, "i", addressString(source.i), allocator);
            addString(
                entry, "status", addressString(source.status), allocator);
            addString(entry, "mac", addressString(source.mac), allocator);
            addString(entry, "clip", addressString(source.clip), allocator);
            addString(entry, "top", addressString(source.top), allocator);
            addString(entry, "itop", addressString(source.itop), allocator);
            entry.AddMember(
                "branch_pending", source.branchPending, allocator);
            addString(
                entry, "branch_target",
                addressString(source.branchTarget), allocator);
            entry.AddMember(
                "branch_delay", source.branchDelay, allocator);
            entry.AddMember(
                "vi_backup_cycles", source.viBackupCycles, allocator);
            entry.AddMember(
                "vi_backup_register", source.viBackupRegister, allocator);
            addString(
                entry, "vi_backup_value",
                addressString(source.viBackupValue), allocator);
            entries.PushBack(entry, allocator);
        }
        result.AddMember("entries", entries, allocator);
        return result;
    }

    Value vu0InstructionTraceStart(
        const Value *params, Allocator &allocator)
    {
        if (runtime.vu0().resolvedBackend() ==
            VuBackendKind::Verify)
        {
            throw RequestError(
                -32002,
                "VU0 instruction tracing requires the interpreter or an instrumented recompiler; select the interpreter while paused");
        }
        const uint64_t maximumEntries =
            requiredUnsigned(params, "maximum_entries");
        if (maximumEntries == 0u || maximumEntries > 8192u)
        {
            throw RequestError(
                -32602, "maximum_entries must be between 1 and 8192");
        }
        std::optional<uint32_t> triggerEePc;
        bool stopOnFull = false;
        if (params)
        {
            const auto trigger = params->FindMember("trigger_ee_pc");
            if (trigger != params->MemberEnd())
            {
                if (!trigger->value.IsString())
                {
                    throw RequestError(
                        -32602, "trigger_ee_pc must be an address string");
                }
                triggerEePc = parseAddress(std::string_view(
                    trigger->value.GetString(),
                    trigger->value.GetStringLength()));
            }
            const auto stop = params->FindMember("stop_on_full");
            if (stop != params->MemberEnd())
            {
                if (!stop->value.IsBool())
                {
                    throw RequestError(
                        -32602, "stop_on_full must be a boolean");
                }
                stopOnFull = stop->value.GetBool();
            }
        }
        runtime.debugStartVu0InstructionTrace(
            static_cast<size_t>(maximumEntries), triggerEePc, stopOnFull);
        return vu0InstructionTraceValue(false, allocator);
    }

    Value eeEventTraceValue(bool stop, Allocator &allocator)
    {
        const PS2Runtime::DebugEeEventTrace trace =
            runtime.debugEeEventTraceSnapshot(stop);
        Value result(rapidjson::kObjectType);
        result.AddMember("enabled", trace.enabled, allocator);
        result.AddMember("triggered", trace.triggered, allocator);
        result.AddMember(
            "stop_on_full", trace.stopOnFull, allocator);
        if (trace.triggerEePc.has_value())
        {
            addString(
                result, "trigger_ee_pc",
                addressString(*trace.triggerEePc), allocator);
        }
        else
        {
            result.AddMember(
                "trigger_ee_pc",
                Value(rapidjson::kNullType), allocator);
        }
        result.AddMember(
            "total_entries", trace.totalEntries, allocator);
        result.AddMember(
            "dropped_entries", trace.droppedEntries, allocator);
        Value entries(rapidjson::kArrayType);
        for (const PS2Runtime::DebugEeEventEntry &source :
             trace.entries)
        {
            Value entry(rapidjson::kObjectType);
            entry.AddMember(
                "sequence", source.sequence, allocator);
            addString(
                entry, "source",
                ps2x::timing::eeEventSourceName(source.source),
                allocator);
            entry.AddMember(
                "event_sequence",
                source.eventSequence, allocator);
            entry.AddMember(
                "generation", source.generation, allocator);
            entry.AddMember(
                "scheduled_tick",
                source.scheduledTick, allocator);
            entry.AddMember(
                "service_tick",
                source.serviceTick, allocator);
            entry.AddMember(
                "lateness_ticks",
                source.latenessTicks, allocator);
            addString(
                entry, "ee_pc",
                addressString(source.eePc), allocator);
            entry.AddMember(
                "device_before",
                eeEventDeviceStateValue(
                    source.deviceBefore, allocator),
                allocator);
            entry.AddMember(
                "device_after",
                eeEventDeviceStateValue(
                    source.deviceAfter, allocator),
                allocator);
            entry.AddMember(
                "rescheduled", source.rescheduled, allocator);
            if (source.hasFollowup)
            {
                entry.AddMember(
                    "followup_tick",
                    source.followupTick, allocator);
            }
            else
            {
                entry.AddMember(
                    "followup_tick",
                    Value(rapidjson::kNullType), allocator);
            }
            entries.PushBack(entry, allocator);
        }
        result.AddMember("entries", entries, allocator);
        return result;
    }

    Value eeEventTraceStart(
        const Value *params, Allocator &allocator)
    {
        const uint64_t maximumEntries =
            requiredUnsigned(params, "maximum_entries");
        if (maximumEntries == 0u || maximumEntries > 16384u)
        {
            throw RequestError(
                -32602,
                "maximum_entries must be between 1 and 16384");
        }

        std::optional<uint32_t> triggerEePc;
        bool stopOnFull = false;
        if (params)
        {
            const auto trigger =
                params->FindMember("trigger_ee_pc");
            if (trigger != params->MemberEnd())
            {
                if (!trigger->value.IsString())
                {
                    throw RequestError(
                        -32602,
                        "trigger_ee_pc must be an address string");
                }
                triggerEePc = parseAddress(std::string_view(
                    trigger->value.GetString(),
                    trigger->value.GetStringLength()));
            }
            const auto stop =
                params->FindMember("stop_on_full");
            if (stop != params->MemberEnd())
            {
                if (!stop->value.IsBool())
                {
                    throw RequestError(
                        -32602,
                        "stop_on_full must be a boolean");
                }
                stopOnFull = stop->value.GetBool();
            }
        }
        runtime.debugStartEeEventTrace(
            static_cast<size_t>(maximumEntries),
            triggerEePc, stopOnFull);
        return eeEventTraceValue(false, allocator);
    }

    Value memory(const Value *params, bool hashOnly, Allocator &allocator)
    {
        const uint32_t address =
            parseAddress(requiredString(params, "address"));
        const uint64_t requestedSize = requiredUnsigned(params, "size");
        if (requestedSize > kMaxMessageBytes ||
            static_cast<uint64_t>(address) + requestedSize > 0x100000000ull)
        {
            throw RequestError(-32602, "invalid or excessive memory range");
        }
        std::vector<uint8_t> bytes;
        if (!runtime.debugReadMemory(
                address, static_cast<uint32_t>(requestedSize), bytes))
        {
            throw RequestError(-32001,
                               "guest range is not contiguous readable memory");
        }
        Value result(rapidjson::kObjectType);
        addString(result, "address", addressString(address), allocator);
        result.AddMember("size", requestedSize, allocator);
        if (hashOnly)
        {
            addString(result, "algorithm", "sha256", allocator);
            addString(result, "digest", digest(bytes.data(), bytes.size()), allocator);
        }
        else
        {
            addString(result, "encoding", "hex", allocator);
            addString(result, "data", hexBytes(bytes.data(), bytes.size()), allocator);
        }
        return result;
    }

    Value padStatus(Allocator &allocator)
    {
        const ps2_stubs::PadDebugSnapshot snapshot =
            ps2_stubs::getPadDebugSnapshot(&runtime);
        Value result(rapidjson::kObjectType);
        result.AddMember("override_enabled", snapshot.overrideEnabled, allocator);
        result.AddMember("active_low", true, allocator);
        result.AddMember(
            "buttons", static_cast<uint32_t>(snapshot.overrideButtons), allocator);
        result.AddMember(
            "lx", static_cast<uint32_t>(snapshot.overrideLx), allocator);
        result.AddMember(
            "ly", static_cast<uint32_t>(snapshot.overrideLy), allocator);
        result.AddMember(
            "rx", static_cast<uint32_t>(snapshot.overrideRx), allocator);
        result.AddMember(
            "ry", static_cast<uint32_t>(snapshot.overrideRy), allocator);
        return result;
    }

    Value vuBackendSet(
        const Value *params, Allocator &allocator)
    {
        const std::string unitText =
            requiredString(params, "unit");
        VuUnitId unit;
        if (unitText == "vu0")
            unit = VuUnitId::Vu0;
        else if (unitText == "vu1")
            unit = VuUnitId::Vu1;
        else
        {
            throw RequestError(
                -32602, "unit must be vu0 or vu1");
        }

        const std::string backendText =
            requiredString(params, "backend");
        VuBackendKind backend;
        if (!parseVuBackendKind(backendText, backend))
        {
            throw RequestError(
                -32602,
                "backend must be auto, interpreter, recompiler, or verify");
        }
        std::string diagnostic;
        if (!runtime.debugSetVuBackend(
                unit, backend, &diagnostic))
        {
            throw RequestError(
                -32002,
                diagnostic.empty()
                    ? "VU backend change was rejected"
                    : diagnostic);
        }

        const VuUnit &selected =
            unit == VuUnitId::Vu0
                ? runtime.vu0()
                : runtime.vu1();
        Value result(rapidjson::kObjectType);
        addString(result, "unit", unitText, allocator);
        addString(
            result, "requested",
            vuBackendKindName(
                selected.requestedBackend()),
            allocator);
        addString(
            result, "resolved",
            vuBackendKindName(
                selected.resolvedBackend()),
            allocator);
        addString(
            result, "name",
            selected.backendName(), allocator);
        result.AddMember(
            "active", selected.isActive(), allocator);
        return result;
    }

    Value padControl(std::string_view method,
                     const Value *params,
                     Allocator &allocator)
    {
        if (method == "input.pad.clear")
        {
            ps2_stubs::clearPadOverrideState(&runtime);
            return padStatus(allocator);
        }

        const uint64_t buttons = requiredUnsigned(params, "buttons");
        const uint64_t lx = requiredUnsigned(params, "lx");
        const uint64_t ly = requiredUnsigned(params, "ly");
        const uint64_t rx = requiredUnsigned(params, "rx");
        const uint64_t ry = requiredUnsigned(params, "ry");
        if (buttons > std::numeric_limits<uint16_t>::max() ||
            lx > std::numeric_limits<uint8_t>::max() ||
            ly > std::numeric_limits<uint8_t>::max() ||
            rx > std::numeric_limits<uint8_t>::max() ||
            ry > std::numeric_limits<uint8_t>::max())
        {
            throw RequestError(-32602, "pad buttons or axes are out of range");
        }

        ps2_stubs::setPadOverrideState(
            &runtime,
            static_cast<uint16_t>(buttons),
            static_cast<uint8_t>(lx),
            static_cast<uint8_t>(ly),
            static_cast<uint8_t>(rx),
            static_cast<uint8_t>(ry));
        return padStatus(allocator);
    }

    Value breakpointList(Allocator &allocator)
    {
        Value result(rapidjson::kArrayType);
        for (uint32_t address : runtime.debugBreakpoints())
        {
            Value item(rapidjson::kObjectType);
            addString(item, "cpu", "ee", allocator);
            addString(item, "address", addressString(address), allocator);
            item.AddMember("enabled", true, allocator);
            result.PushBack(item, allocator);
        }
        return result;
    }

    Value breakpointChange(std::string_view method,
                           const Value *params,
                           Allocator &allocator)
    {
        const std::string cpu = requiredString(params, "cpu");
        if (cpu != "ee")
        {
            throw RequestError(-32001,
                               "PS2Recomp v1 only supports EE breakpoints");
        }
        const uint32_t address =
            parseAddress(requiredString(params, "address"));
        if (method == "breakpoint.add")
        {
            runtime.debugAddBreakpoint(address);
        }
        else
        {
            runtime.debugRemoveBreakpoint(address);
        }
        Value result(rapidjson::kObjectType);
        addString(result, "cpu", "ee", allocator);
        addString(result, "address", addressString(address), allocator);
        addString(result, "operation",
                  method == "breakpoint.add" ? "added" : "removed", allocator);
        return result;
    }

    static const char *accessName(PS2Runtime::DebugMemoryAccess access)
    {
        switch (access)
        {
        case PS2Runtime::DebugMemoryAccess::Read:
            return "read";
        case PS2Runtime::DebugMemoryAccess::Write:
            return "write";
        case PS2Runtime::DebugMemoryAccess::ReadWrite:
            return "readwrite";
        default:
            return "unknown";
        }
    }

    Value watchpointList(Allocator &allocator)
    {
        Value result(rapidjson::kArrayType);
        for (const PS2Runtime::DebugWatchpoint &watchpoint :
             runtime.debugWatchpoints())
        {
            Value item(rapidjson::kObjectType);
            addString(item, "id", std::to_string(watchpoint.id), allocator);
            addString(item, "cpu", "ee", allocator);
            addString(item, "start", addressString(watchpoint.start), allocator);
            item.AddMember("size", watchpoint.size, allocator);
            addString(item, "access", accessName(watchpoint.access), allocator);
            result.PushBack(item, allocator);
        }
        return result;
    }

    Value watchpointAdd(const Value *params, Allocator &allocator)
    {
        if (requiredString(params, "cpu") != "ee")
        {
            throw RequestError(-32001,
                               "PS2Recomp v1 only supports EE watchpoints");
        }
        const uint32_t start =
            parseAddress(requiredString(params, "start"));
        const uint64_t size = requiredUnsigned(params, "size");
        if (size == 0u || size > std::numeric_limits<uint32_t>::max() ||
            static_cast<uint64_t>(start) + size > 0x100000000ull)
        {
            throw RequestError(-32602, "invalid watchpoint range");
        }
        const std::string accessText = requiredString(params, "access");
        const PS2Runtime::DebugMemoryAccess access =
            accessText == "read"
                ? PS2Runtime::DebugMemoryAccess::Read
                : accessText == "write"
                      ? PS2Runtime::DebugMemoryAccess::Write
                      : accessText == "readwrite"
                            ? PS2Runtime::DebugMemoryAccess::ReadWrite
                            : static_cast<PS2Runtime::DebugMemoryAccess>(0u);
        if (static_cast<uint8_t>(access) == 0u)
        {
            throw RequestError(-32602, "invalid watchpoint access");
        }
        const uint64_t id = runtime.debugAddWatchpoint(
            start, static_cast<uint32_t>(size), access);
        if (id == 0u)
        {
            throw RequestError(-32602, "invalid watchpoint range");
        }
        Value result(rapidjson::kObjectType);
        addString(result, "id", std::to_string(id), allocator);
        return result;
    }

    Value watchpointRemove(const Value *params, Allocator &allocator)
    {
        const std::string idText = requiredString(params, "id");
        char *end = nullptr;
        errno = 0;
        const unsigned long long id =
            std::strtoull(idText.c_str(), &end, 10);
        if (errno != 0 || end != idText.c_str() + idText.size() ||
            !runtime.debugRemoveWatchpoint(id))
        {
            throw RequestError(-32602, "unknown watchpoint ID");
        }
        Value result(rapidjson::kObjectType);
        addString(result, "id", idText, allocator);
        addString(result, "operation", "removed", allocator);
        return result;
    }

    Value runUntil(const Value *params, Allocator &allocator)
    {
        std::string predicate = requiredString(params, "predicate");
        const size_t equals = predicate.find("==");
        if (equals == std::string::npos)
        {
            throw RequestError(-32602, "v1 requires 'ee.pc == <address>'");
        }
        auto trim = [](std::string_view value)
        {
            while (!value.empty() &&
                   std::isspace(static_cast<unsigned char>(value.front())))
            {
                value.remove_prefix(1u);
            }
            while (!value.empty() &&
                   std::isspace(static_cast<unsigned char>(value.back())))
            {
                value.remove_suffix(1u);
            }
            return value;
        };
        const std::string_view field =
            trim(std::string_view(predicate).substr(0u, equals));
        const std::string_view value =
            trim(std::string_view(predicate).substr(equals + 2u));
        if (field != "ee.pc")
        {
            throw RequestError(-32001,
                               "PS2Recomp v1 supports only ee.pc predicates");
        }
        const uint32_t target = parseAddress(value);
        const uint64_t timeout = timeoutMs(params);
        const PS2Runtime::DebugStopInfo stop = runtime.debugRunUntilPc(
            target, std::chrono::milliseconds(timeout));
        if (!stop.completed)
        {
            throw RequestError(-32002, "run-until " + stop.reason);
        }
        Value result(rapidjson::kObjectType);
        addString(result, "reason", stop.reason, allocator);
        addString(result, "cpu", "ee", allocator);
        addString(result, "pc", addressString(stop.pc), allocator);
        result.AddMember("sequence", stop.sequence, allocator);
        result.AddMember("timeout_ms", timeout, allocator);
        return result;
    }

    Value step(const Value *params, Allocator &allocator)
    {
        const std::string granularity =
            requiredString(params, "granularity");
        if (granularity != "dispatch" && granularity != "block")
        {
            throw RequestError(
                -32001,
                "PS2Recomp v1 can step recompiled dispatches/blocks, not individual instructions");
        }
        const uint64_t count = requiredUnsigned(params, "count");
        if (count == 0u || count > 100000u)
        {
            throw RequestError(-32602,
                               "step count must be between 1 and 100000");
        }
        const PS2Runtime::DebugStopInfo stop = runtime.debugStepDispatches(
            count, std::chrono::seconds(30));
        if (!stop.completed)
        {
            throw RequestError(-32002, "step " + stop.reason);
        }
        Value result(rapidjson::kObjectType);
        addString(result, "cpu", "ee", allocator);
        addString(result, "granularity", "dispatch", allocator);
        result.AddMember("count", count, allocator);
        addString(result, "pc", addressString(stop.pc), allocator);
        result.AddMember("sequence", stop.sequence, allocator);
        return result;
    }

    void addArtifact(Value &artifacts,
                     const std::filesystem::path &path,
                     std::string_view kind,
                     Allocator &allocator)
    {
        Value artifact(rapidjson::kObjectType);
        addString(artifact, "path", path.filename().string(), allocator);
        addString(artifact, "kind", kind, allocator);
        artifact.AddMember(
            "size", static_cast<uint64_t>(std::filesystem::file_size(path)),
            allocator);
        addString(artifact, "sha256", fileDigest(path), allocator);
        artifacts.PushBack(artifact, allocator);
    }

    void writeRuntimeDiagnostics(const std::filesystem::path &root,
                                 std::string_view reason,
                                 bool quiescent,
                                 const PS2WatchdogSample &sample,
                                 const PS2WatchdogAssessment &assessment,
                                 Value &artifacts,
                                 Allocator &allocator)
    {
        Document document;
        document.SetObject();
        document.CopyFrom(
            diagnosticsValue(
                reason, quiescent, sample, assessment,
                document.GetAllocator()),
            document.GetAllocator());
        const auto path = root / "runtime-diagnostics.json";
        writeJson(path, document);
        addArtifact(artifacts, path, "runtime-diagnostics", allocator);
    }

    void writeCodeWindow(const std::filesystem::path &root,
                         Value &artifacts,
                         Allocator &allocator)
    {
        const PS2Runtime::DebugFaultInfo fault =
            runtime.debugFaultSnapshot();
        const PS2Runtime::DebugRuntimeProgress core =
            runtime.debugRuntimeProgress();
        uint32_t center = fault.active ? fault.targetPc : core.pc;
        uint32_t physical = 0u;
        if (!ps2ResolveDirectRdramOffset(center, physical))
        {
            return;
        }

        constexpr uint32_t kWindowRadius = 128u;
        const uint32_t start =
            (physical > kWindowRadius ? physical - kWindowRadius : 0u) & ~3u;
        const uint32_t end = std::min<uint32_t>(
            PS2_RAM_SIZE, physical + kWindowRadius);
        if (end <= start)
        {
            return;
        }

        std::vector<uint8_t> bytes;
        if (!runtime.debugReadRdram(start, end - start, bytes))
        {
            return;
        }
        const auto binary = root / "code-window.bin";
        writeBytes(binary, bytes.data(), bytes.size());
        addArtifact(artifacts, binary, "code-window", allocator);

        Document metadata;
        metadata.SetObject();
        addString(metadata, "center", addressString(physical),
                  metadata.GetAllocator());
        addString(metadata, "start", addressString(start),
                  metadata.GetAllocator());
        metadata.AddMember(
            "size", static_cast<uint64_t>(bytes.size()),
            metadata.GetAllocator());
        const auto json = root / "code-window.json";
        writeJson(json, metadata);
        addArtifact(artifacts, json, "code-window-metadata", allocator);
    }

    void writeBranchEvents(const std::filesystem::path &root,
                           uint64_t historyLimit,
                           Value &artifacts,
                           Allocator &allocator)
    {
        const auto branches = runtime.debugBranchHistory(
            static_cast<size_t>(std::min<uint64_t>(
                historyLimit, 256u)));
        std::ostringstream lines;
        uint64_t outputSequence = 1u;
        for (const PS2Runtime::DebugBranchEntry &branch : branches)
        {
            Document event;
            event.SetObject();
            Allocator &eventAllocator = event.GetAllocator();
            event.AddMember("schema_version", 1u, eventAllocator);
            event.AddMember("sequence", outputSequence++, eventAllocator);
            event.AddMember("guest_cycle", branch.sequence, eventAllocator);
            event.AddMember("frame", 0u, eventAllocator);
            addString(event, "thread", "ee", eventAllocator);
            addString(event, "pc", addressString(branch.pc), eventAllocator);
            addString(event, "symbol",
                      PS2Runtime::formatGuestPc(branch.pc), eventAllocator);
            addString(event, "subsystem", "ee", eventAllocator);
            addString(event, "event", "dispatch", eventAllocator);
            Value payload(rapidjson::kObjectType);
            payload.AddMember(
                "dispatch_sequence", branch.sequence, eventAllocator);
            Document payloadDocument;
            payloadDocument.SetObject();
            payloadDocument.CopyFrom(payload, payloadDocument.GetAllocator());
            const std::string payloadText = serialize(payloadDocument);
            addString(event, "payload_sha256",
                      digest(payloadText.data(), payloadText.size()),
                      eventAllocator);
            event.AddMember("payload", payload, eventAllocator);
            lines << serialize(event) << '\n';
        }
        const std::string text = lines.str();
        const auto path = root / "events.jsonl";
        writeBytes(path, text.data(), text.size());
        addArtifact(artifacts, path, "events", allocator);
    }

    void writeEeState(const std::filesystem::path &root,
                      Value &artifacts,
                      Allocator &allocator)
    {
        const R5900Context context = runtime.debugCpuSnapshot();
        const auto binary = root / "ee-state.bin";
        writeBytes(binary, &context, sizeof(context));
        addArtifact(artifacts, binary, "ee-state", allocator);

        Document document;
        document.SetObject();
        document.CopyFrom(eeRegisters(document.GetAllocator()),
                          document.GetAllocator());
        const auto json = root / "ee-state.json";
        writeJson(json, document);
        addArtifact(artifacts, json, "ee-registers", allocator);
    }

    bool writeVuState(const std::filesystem::path &root,
                      bool vu1,
                      Value &artifacts,
                      Allocator &allocator)
    {
        Document document;
        document.SetObject();
        document.CopyFrom(vuRegisters(vu1, document.GetAllocator()),
                          document.GetAllocator());
        const std::string prefix = vu1 ? "vu1" : "vu0";
        const auto statePath = root / (prefix + "-state.json");
        writeJson(statePath, document);
        addArtifact(artifacts, statePath, prefix + "-registers", allocator);

        if (vu1)
        {
            const VuExecutionState state = runtime.vu1().state();
            std::array<uint32_t, 159> words{};
            words[0] = 0x31555652u; // "RVU1"
            words[1] = 1u;
            words[2] = state.pc;
            words[3] = state.top;
            words[4] = state.itop;
            size_t cursor = 5u;
            std::memcpy(
                words.data() + cursor, state.vf, sizeof(state.vf));
            cursor += 32u * 4u;
            for (size_t index = 0u; index < 16u; ++index)
                words[cursor++] = static_cast<uint32_t>(state.vi[index]);
            std::memcpy(
                words.data() + cursor, state.acc, sizeof(state.acc));
            cursor += 4u;
            std::memcpy(&words[cursor++], &state.q, sizeof(state.q));
            std::memcpy(&words[cursor++], &state.p, sizeof(state.p));
            std::memcpy(&words[cursor++], &state.i, sizeof(state.i));
            words[cursor++] = state.status;
            words[cursor++] = state.mac;
            words[cursor++] = state.clip;

            const auto replayStatePath =
                root / "vu1-replay-state.bin";
            writeBytes(
                replayStatePath, words.data(), sizeof(words));
            addArtifact(
                artifacts,
                replayStatePath,
                "vu1-replay-state",
                allocator);
        }

        const uint8_t *code =
            vu1 ? runtime.memory().getVU1Code() : runtime.memory().getVU0Code();
        const uint8_t *data =
            vu1 ? runtime.memory().getVU1Data() : runtime.memory().getVU0Data();
        if (code == nullptr || data == nullptr)
        {
            return false;
        }
        const uint32_t codeSize = vu1 ? PS2_VU1_CODE_SIZE : PS2_VU0_CODE_SIZE;
        const uint32_t dataSize = vu1 ? PS2_VU1_DATA_SIZE : PS2_VU0_DATA_SIZE;
        const auto codePath = root / (prefix + "-code.bin");
        const auto dataPath = root / (prefix + "-data.bin");
        writeBytes(codePath, code, codeSize);
        writeBytes(dataPath, data, dataSize);
        addArtifact(artifacts, codePath, prefix + "-code", allocator);
        addArtifact(artifacts, dataPath, prefix + "-data", allocator);
        return true;
    }

    void writeIopState(const std::filesystem::path &root,
                       Value &artifacts,
                       Allocator &allocator)
    {
        Document document;
        document.SetObject();
        document.CopyFrom(iopRegisters(document.GetAllocator()),
                          document.GetAllocator());
        const auto path = root / "iop-state.json";
        writeJson(path, document);
        addArtifact(artifacts, path, "iop-registers", allocator);
    }

    void writeGsState(const std::filesystem::path &root,
                      Value &artifacts,
                      Allocator &allocator)
    {
        const GSDebugSnapshot snapshot = runtime.gs().getDebugSnapshot();
        Document document;
        document.SetObject();
        Allocator &stateAllocator = document.GetAllocator();
        document.AddMember("prim_type",
                           static_cast<uint32_t>(snapshot.prim.type),
                           stateAllocator);
        document.AddMember("prim_iip", snapshot.prim.iip, stateAllocator);
        document.AddMember("prim_tme", snapshot.prim.tme, stateAllocator);
        document.AddMember("prim_abe", snapshot.prim.abe, stateAllocator);
        document.AddMember("prim_fst", snapshot.prim.fst, stateAllocator);
        document.AddMember("prim_context", snapshot.prim.ctxt, stateAllocator);
        Value contexts(rapidjson::kArrayType);
        for (const GSContext &context : snapshot.ctx)
        {
            Value item(rapidjson::kObjectType);
            item.AddMember("frame_fbp", context.frame.fbp, stateAllocator);
            item.AddMember("frame_fbw", context.frame.fbw, stateAllocator);
            item.AddMember("frame_psm",
                           static_cast<uint32_t>(context.frame.psm),
                           stateAllocator);
            item.AddMember("frame_mask", context.frame.fbmsk, stateAllocator);
            item.AddMember("zbuf_zbp", context.zbuf.zbp, stateAllocator);
            item.AddMember("zbuf_psm",
                           static_cast<uint32_t>(context.zbuf.psm),
                           stateAllocator);
            item.AddMember("zbuf_mask", context.zbuf.zmask, stateAllocator);
            item.AddMember("tex0_tbp0", context.tex0.tbp0, stateAllocator);
            item.AddMember("tex0_tbw", context.tex0.tbw, stateAllocator);
            item.AddMember("tex0_psm",
                           static_cast<uint32_t>(context.tex0.psm),
                           stateAllocator);
            item.AddMember("test", context.test, stateAllocator);
            item.AddMember("alpha", context.alpha, stateAllocator);
            contexts.PushBack(item, stateAllocator);
        }
        document.AddMember("contexts", contexts, stateAllocator);
        document.AddMember("transfer_x", snapshot.transferX, stateAllocator);
        document.AddMember("transfer_y", snapshot.transferY, stateAllocator);
        document.AddMember("transfer_total_pixels",
                           snapshot.transferTotalPixels, stateAllocator);
        document.AddMember("transfer_copied_pixels",
                           snapshot.transferCopiedPixels, stateAllocator);
        document.AddMember("presentation_width",
                           snapshot.hostPresentationWidth, stateAllocator);
        document.AddMember("presentation_height",
                           snapshot.hostPresentationHeight, stateAllocator);
        document.AddMember("presentation_display_fbp",
                           snapshot.hostPresentationDisplayFbp, stateAllocator);
        document.AddMember("presentation_source_fbp",
                           snapshot.hostPresentationSourceFbp, stateAllocator);
        document.AddMember("presentation_preferred",
                           snapshot.hostPresentationUsedPreferred, stateAllocator);

        const auto path = root / "gs-registers.json";
        writeJson(path, document);
        addArtifact(artifacts, path, "gs-registers", allocator);
    }

    void writeEvents(const std::filesystem::path &root,
                     uint64_t historyLimit,
                     Value &artifacts,
                     Allocator &allocator)
    {
        std::vector<GSDebugHistoryEntry> history = runtime.gs().getDebugHistory();
        if (history.size() > historyLimit)
        {
            history.erase(history.begin(),
                          history.end() - static_cast<std::ptrdiff_t>(historyLimit));
        }

        std::ostringstream lines;
        for (const GSDebugHistoryEntry &entry : history)
        {
            Document event;
            event.SetObject();
            Allocator &eventAllocator = event.GetAllocator();
            event.AddMember("schema_version", 1u, eventAllocator);
            event.AddMember("sequence", entry.seq, eventAllocator);
            event.AddMember("guest_cycle", entry.vsyncTick, eventAllocator);
            event.AddMember("frame", entry.frameIndex, eventAllocator);
            addString(event, "thread", "gs", eventAllocator);
            addString(event, "pc",
                      addressString(runtime.m_debugPc.load(std::memory_order_relaxed)),
                      eventAllocator);
            addString(event, "symbol",
                      PS2Runtime::formatGuestPc(
                          runtime.m_debugPc.load(std::memory_order_relaxed)),
                      eventAllocator);
            addString(event, "subsystem", "gs", eventAllocator);
            addString(event, "event", gsEventName(entry.kind), eventAllocator);

            Value payload(rapidjson::kObjectType);
            payload.AddMember("gif_size", entry.gifSizeBytes, eventAllocator);
            payload.AddMember("gif_nloop", entry.gifNloop, eventAllocator);
            payload.AddMember("gif_flg", entry.gifFlg, eventAllocator);
            payload.AddMember("gif_nreg", entry.gifNreg, eventAllocator);
            payload.AddMember("register", entry.reg, eventAllocator);
            payload.AddMember("register_value", entry.regValue, eventAllocator);
            payload.AddMember("prim_type",
                              static_cast<uint32_t>(entry.prim.type),
                              eventAllocator);
            payload.AddMember("frame_fbp", entry.frame.fbp, eventAllocator);
            payload.AddMember("frame_fbw", entry.frame.fbw, eventAllocator);
            payload.AddMember("frame_psm",
                              static_cast<uint32_t>(entry.frame.psm),
                              eventAllocator);
            payload.AddMember("frame_mask", entry.frame.fbmsk, eventAllocator);
            payload.AddMember("vertex_count", entry.vertexCount, eventAllocator);
            payload.AddMember("x_min", entry.xMin, eventAllocator);
            payload.AddMember("x_max", entry.xMax, eventAllocator);
            payload.AddMember("y_min", entry.yMin, eventAllocator);
            payload.AddMember("y_max", entry.yMax, eventAllocator);
            payload.AddMember("transfer_width", entry.trxreg.rrw, eventAllocator);
            payload.AddMember("transfer_height", entry.trxreg.rrh, eventAllocator);
            payload.AddMember("transfer_pixels", entry.transferPixels, eventAllocator);
            payload.AddMember("present_width", entry.width, eventAllocator);
            payload.AddMember("present_height", entry.height, eventAllocator);
            payload.AddMember("display_fbp", entry.displayFbp, eventAllocator);
            payload.AddMember("source_fbp", entry.sourceFbp, eventAllocator);

            Document payloadDocument;
            payloadDocument.SetObject();
            payloadDocument.CopyFrom(payload, payloadDocument.GetAllocator());
            const std::string payloadText = serialize(payloadDocument);
            addString(event, "payload_sha256",
                      digest(payloadText.data(), payloadText.size()),
                      eventAllocator);
            event.AddMember("payload", payload, eventAllocator);
            lines << serialize(event) << '\n';
        }

        const std::string text = lines.str();
        const auto path = root / "events.jsonl";
        writeBytes(path, text.data(), text.size());
        addArtifact(artifacts, path, "events", allocator);
    }

    std::string gsReplayRegisterFile(
        const GsReplayState &snapshot,
        std::string_view stateFile)
    {
        std::ostringstream output;
        if (!stateFile.empty())
            output << "@state-file=" << stateFile << '\n';
        const auto append = [&output](uint8_t address, uint64_t value)
        {
            output << "0x" << std::hex << std::setw(2)
                   << std::setfill('0') << static_cast<uint32_t>(address)
                   << "=0x" << std::setw(16) << value << std::dec << '\n';
        };
        const auto packPrim = [](const GSPrimReg &prim)
        {
            return static_cast<uint64_t>(prim.type) |
                   (static_cast<uint64_t>(prim.iip) << 3u) |
                   (static_cast<uint64_t>(prim.tme) << 4u) |
                   (static_cast<uint64_t>(prim.fge) << 5u) |
                   (static_cast<uint64_t>(prim.abe) << 6u) |
                   (static_cast<uint64_t>(prim.aa1) << 7u) |
                   (static_cast<uint64_t>(prim.fst) << 8u) |
                   (static_cast<uint64_t>(prim.ctxt) << 9u) |
                   (static_cast<uint64_t>(prim.fix) << 10u);
        };
        const auto packTex0 = [](const GSTex0Reg &tex)
        {
            return static_cast<uint64_t>(tex.tbp0) |
                   (static_cast<uint64_t>(tex.tbw) << 14u) |
                   (static_cast<uint64_t>(tex.psm) << 20u) |
                   (static_cast<uint64_t>(tex.tw) << 26u) |
                   (static_cast<uint64_t>(tex.th) << 30u) |
                   (static_cast<uint64_t>(tex.tcc) << 34u) |
                   (static_cast<uint64_t>(tex.tfx) << 35u) |
                   (static_cast<uint64_t>(tex.cbp) << 37u) |
                   (static_cast<uint64_t>(tex.cpsm) << 51u) |
                   (static_cast<uint64_t>(tex.csm) << 55u) |
                   (static_cast<uint64_t>(tex.csa) << 56u) |
                   (static_cast<uint64_t>(tex.cld) << 61u);
        };
        const auto packFrame = [](const GSFrameReg &frame)
        {
            return static_cast<uint64_t>(frame.fbp) |
                   (static_cast<uint64_t>(frame.fbw) << 16u) |
                   (static_cast<uint64_t>(frame.psm) << 24u) |
                   (static_cast<uint64_t>(frame.fbmsk) << 32u);
        };
        const auto packZbuf = [](const GSZbufReg &zbuf)
        {
            return static_cast<uint64_t>(zbuf.zbp) |
                   (static_cast<uint64_t>(zbuf.psm & 0xFu) << 24u) |
                   (static_cast<uint64_t>(zbuf.zmask) << 32u);
        };
        const auto packScissor = [](const GSScissorReg &scissor)
        {
            return static_cast<uint64_t>(scissor.x0) |
                   (static_cast<uint64_t>(scissor.x1) << 16u) |
                   (static_cast<uint64_t>(scissor.y0) << 32u) |
                   (static_cast<uint64_t>(scissor.y1) << 48u);
        };
        const auto packOffset = [](const GSXYOffsetReg &offset)
        {
            return static_cast<uint64_t>(offset.ofx) |
                   (static_cast<uint64_t>(offset.ofy) << 32u);
        };

        append(GS_REG_PRMODECONT, snapshot.prmodecont ? 1u : 0u);
        append(
            GS_REG_TEXCLUT,
            static_cast<uint64_t>(snapshot.texclut.cbw) |
                (static_cast<uint64_t>(snapshot.texclut.cou) << 6u) |
                (static_cast<uint64_t>(snapshot.texclut.cov) << 12u));
        append(
            GS_REG_TEXA,
            static_cast<uint64_t>(snapshot.texa.ta0) |
                (static_cast<uint64_t>(snapshot.texa.aem) << 15u) |
                (static_cast<uint64_t>(snapshot.texa.ta1) << 32u));
        append(GS_REG_FOGCOL, snapshot.fogColor);
        append(GS_REG_PABE, snapshot.pabe ? 1u : 0u);
        append(GS_REG_SCANMSK, snapshot.scanMask);
        append(GS_REG_DIMX, snapshot.dimx);
        append(GS_REG_DTHE, snapshot.dither ? 1u : 0u);
        append(GS_REG_COLCLAMP, snapshot.colorClamp ? 1u : 0u);
        for (uint32_t index = 0u; index < 2u; ++index)
        {
            const GSContext &context = snapshot.ctx[index];
            append(
                static_cast<uint8_t>(GS_REG_TEX0_1 + index),
                packTex0(context.tex0));
            append(
                static_cast<uint8_t>(GS_REG_CLAMP_1 + index),
                context.clamp);
            append(
                static_cast<uint8_t>(GS_REG_TEX1_1 + index),
                context.tex1);
            append(
                static_cast<uint8_t>(GS_REG_XYOFFSET_1 + index),
                packOffset(context.xyoffset));
            append(
                static_cast<uint8_t>(GS_REG_MIPTBP1_1 + index),
                context.miptbp1);
            append(
                static_cast<uint8_t>(GS_REG_MIPTBP2_1 + index),
                context.miptbp2);
            append(
                static_cast<uint8_t>(GS_REG_SCISSOR_1 + index),
                packScissor(context.scissor));
            append(
                static_cast<uint8_t>(GS_REG_ALPHA_1 + index),
                context.alpha);
            append(
                static_cast<uint8_t>(GS_REG_TEST_1 + index),
                context.test);
            append(
                static_cast<uint8_t>(GS_REG_FBA_1 + index),
                context.fba);
            append(
                static_cast<uint8_t>(GS_REG_FRAME_1 + index),
                packFrame(context.frame));
            append(
                static_cast<uint8_t>(GS_REG_ZBUF_1 + index),
                packZbuf(context.zbuf));
        }
        append(
            GS_REG_BITBLTBUF,
            static_cast<uint64_t>(snapshot.bitbltbuf.sbp) |
                (static_cast<uint64_t>(snapshot.bitbltbuf.sbw) << 16u) |
                (static_cast<uint64_t>(snapshot.bitbltbuf.spsm) << 24u) |
                (static_cast<uint64_t>(snapshot.bitbltbuf.dbp) << 32u) |
                (static_cast<uint64_t>(snapshot.bitbltbuf.dbw) << 48u) |
                (static_cast<uint64_t>(snapshot.bitbltbuf.dpsm) << 56u));
        append(
            GS_REG_TRXPOS,
            static_cast<uint64_t>(snapshot.trxpos.ssax) |
                (static_cast<uint64_t>(snapshot.trxpos.ssay) << 16u) |
                (static_cast<uint64_t>(snapshot.trxpos.dsax) << 32u) |
                (static_cast<uint64_t>(snapshot.trxpos.dsay) << 48u) |
                (static_cast<uint64_t>(snapshot.trxpos.dir) << 59u));
        append(
            GS_REG_TRXREG,
            static_cast<uint64_t>(snapshot.trxreg.rrw) |
                (static_cast<uint64_t>(snapshot.trxreg.rrh) << 32u));
        append(GS_REG_TRXDIR, snapshot.trxdir);
        append(GS_REG_PRIM, packPrim(snapshot.prim));
        append(
            GS_REG_RGBAQ,
            static_cast<uint64_t>(snapshot.currentR) |
                (static_cast<uint64_t>(snapshot.currentG) << 8u) |
                (static_cast<uint64_t>(snapshot.currentB) << 16u) |
                (static_cast<uint64_t>(snapshot.currentA) << 24u) |
                (static_cast<uint64_t>(
                     std::bit_cast<uint32_t>(snapshot.currentQ))
                 << 32u));
        append(
            GS_REG_ST,
            static_cast<uint64_t>(
                std::bit_cast<uint32_t>(snapshot.currentS)) |
                (static_cast<uint64_t>(
                     std::bit_cast<uint32_t>(snapshot.currentT))
                 << 32u));
        append(
            GS_REG_UV,
            static_cast<uint64_t>(snapshot.currentU) |
                (static_cast<uint64_t>(snapshot.currentV) << 16u));
        append(
            GS_REG_FOG,
            static_cast<uint64_t>(snapshot.currentFog) << 56u);
        return output.str();
    }

    void writeGifStream(const std::filesystem::path &root,
                        uint64_t historyLimit,
                        Value &artifacts,
                        Allocator &allocator)
    {
        std::vector<uint8_t> stream;
        std::vector<uint32_t> sizes;
        std::vector<uint8_t> initialVram;
        GsReplayState initialState{};
        runtime.gs().copyRecentGifPackets(
            static_cast<size_t>(std::min<uint64_t>(
                historyLimit, std::numeric_limits<size_t>::max())),
            stream,
            sizes,
            initialVram,
            &initialState);
        if (sizes.empty())
            return;

        const auto streamPath = root / "gif-stream.bin";
        writeBytes(streamPath, stream.data(), stream.size());
        addArtifact(artifacts, streamPath, "gif-stream", allocator);

        std::ostringstream sizeText;
        for (uint32_t size : sizes)
            sizeText << size << '\n';
        const std::string text = sizeText.str();
        const auto sizesPath = root / "gif-packet-sizes.txt";
        writeBytes(sizesPath, text.data(), text.size());
        addArtifact(
            artifacts, sizesPath, "gif-packet-sizes", allocator);

        if (!initialVram.empty())
        {
            const auto vramPath = root / "initial-vram.bin";
            writeBytes(vramPath, initialVram.data(), initialVram.size());
            addArtifact(artifacts, vramPath, "initial-vram", allocator);

            std::vector<uint8_t> stateBytes;
            std::string stateError;
            if (!encodeGsReplayState(
                    initialState, stateBytes, &stateError))
            {
                throw std::runtime_error(
                    "failed to encode initial GS replay state: " +
                    stateError);
            }
            const auto statePath = root / "initial-gs-state.bin";
            writeBytes(
                statePath, stateBytes.data(), stateBytes.size());
            addArtifact(
                artifacts,
                statePath,
                "initial-gs-state",
                allocator);

            const std::string registers =
                gsReplayRegisterFile(
                    initialState,
                    statePath.filename().string());
            const auto registersPath =
                root / "initial-gs-registers.txt";
            writeBytes(
                registersPath, registers.data(), registers.size());
            addArtifact(
                artifacts,
                registersPath,
                "initial-gs-register-file",
                allocator);
        }
    }

    bool writeScreenshot(const std::filesystem::path &root,
                         Value &artifacts,
                         Allocator &allocator)
    {
        std::vector<uint8_t> pixels;
        uint32_t width = 0u;
        uint32_t height = 0u;
        if (!runtime.gs().copyLatchedHostPresentationFrame(
                pixels, width, height) ||
            width == 0u || height == 0u ||
            pixels.size() < static_cast<size_t>(width) * height * 4u)
        {
            return false;
        }

        Image image{};
        image.data = pixels.data();
        image.width = static_cast<int>(width);
        image.height = static_cast<int>(height);
        image.mipmaps = 1;
        image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
        const auto path = root / "screenshot.png";
        if (!ExportImage(image, path.string().c_str()))
        {
            throw RequestError(-32003, "cannot encode screenshot");
        }
        addArtifact(artifacts, path, "screenshot", allocator);
        return true;
    }

    void createAutomaticBundle(std::string_view reason,
                               uint64_t identifier,
                               const PS2WatchdogSample &sample,
                               const PS2WatchdogAssessment &assessment)
    {
        std::lock_guard<std::mutex> captureLock(captureMutex);
        std::filesystem::path partial;
        bool resumeOnExit = false;
        try
        {
            const std::filesystem::path directory = crashDirectory();
            std::filesystem::create_directories(directory);
            chmod(directory.c_str(), 0700);

            std::ostringstream leaf;
            leaf << "recomp-" << getpid() << '-' << reason << '-'
                 << identifier;
            std::filesystem::path output = directory / leaf.str();
            for (uint32_t suffix = 2u;
                 std::filesystem::exists(output);
                 ++suffix)
            {
                output = directory /
                         (leaf.str() + "-" + std::to_string(suffix));
            }
            partial = output.string() + ".partial";
            if (std::filesystem::exists(partial))
            {
                std::filesystem::remove_all(partial);
            }

            const bool alreadyPaused = runtime.debugIsPaused();
            const bool quiescent =
                alreadyPaused ||
                runtime.debugPause(std::chrono::milliseconds(1500));
            resumeOnExit = quiescent && !alreadyPaused;

            std::filesystem::create_directories(partial);
            Document manifest;
            manifest.SetObject();
            Allocator &manifestAllocator = manifest.GetAllocator();
            manifest.AddMember("schema_version", 1, manifestAllocator);
            addString(manifest, "backend", "recomp", manifestAllocator);
            manifest.AddMember(
                "protocol_version", kProtocolVersion, manifestAllocator);
            addString(manifest, "reason", reason, manifestAllocator);
            addString(
                manifest, "classification", assessment.name, manifestAllocator);
            manifest.AddMember("quiescent", quiescent, manifestAllocator);
            Value artifacts(rapidjson::kArrayType);
            Value unsupported(rapidjson::kArrayType);

            writeRuntimeDiagnostics(
                partial, reason, quiescent, sample, assessment,
                artifacts, manifestAllocator);
            if (quiescent)
            {
                writeEvents(partial, 256u, artifacts, manifestAllocator);
                writeGifStream(
                    partial, 256u, artifacts, manifestAllocator);
                writeEeState(partial, artifacts, manifestAllocator);
                writeIopState(partial, artifacts, manifestAllocator);
                if (!writeVuState(
                        partial, false, artifacts, manifestAllocator))
                {
                    unsupported.PushBack(
                        Value("vu0-memory", manifestAllocator),
                        manifestAllocator);
                }
                if (!writeVuState(
                        partial, true, artifacts, manifestAllocator))
                {
                    unsupported.PushBack(
                        Value("vu1-memory", manifestAllocator),
                        manifestAllocator);
                }
                writeGsState(partial, artifacts, manifestAllocator);
                std::vector<uint8_t> vram;
                if (runtime.debugCopyGsVram(vram))
                {
                    const auto path = partial / "final-vram.bin";
                    writeBytes(path, vram.data(), vram.size());
                    addArtifact(
                        artifacts, path, "final-vram", manifestAllocator);
                }
                else
                {
                    unsupported.PushBack(
                        Value("vram", manifestAllocator), manifestAllocator);
                }
                if (!writeScreenshot(
                        partial, artifacts, manifestAllocator))
                {
                    unsupported.PushBack(
                        Value("screenshot", manifestAllocator),
                        manifestAllocator);
                }
                writeCodeWindow(partial, artifacts, manifestAllocator);
            }
            else
            {
                writeBranchEvents(
                    partial, 256u, artifacts, manifestAllocator);
                for (const char *scope : {
                         "ee-state", "iop-state", "vu0-state", "vu1-state",
                         "gs-state", "vram", "screenshot", "code-window"})
                {
                    unsupported.PushBack(
                        Value(scope, manifestAllocator), manifestAllocator);
                }
            }

            manifest.AddMember("artifacts", artifacts, manifestAllocator);
            manifest.AddMember(
                "unsupported_scopes", unsupported, manifestAllocator);
            writeJson(partial / "manifest.json", manifest);
            std::filesystem::rename(partial, output);

            if (resumeOnExit)
            {
                runtime.debugResume();
                resumeOnExit = false;
            }
            {
                std::lock_guard<std::mutex> lock(watchdogStateMutex);
                watchdogLastBundle = output.string();
                watchdogLastError.clear();
            }
            std::cerr << "[ps2dbg] automatic diagnostic bundle: "
                      << output << '\n';
        }
        catch (const std::exception &error)
        {
            if (resumeOnExit)
            {
                runtime.debugResume();
            }
            if (!partial.empty())
            {
                std::error_code cleanupError;
                std::filesystem::remove_all(partial, cleanupError);
            }
            std::lock_guard<std::mutex> lock(watchdogStateMutex);
            watchdogLastError = error.what();
        }
        catch (...)
        {
            if (resumeOnExit)
            {
                runtime.debugResume();
            }
            if (!partial.empty())
            {
                std::error_code cleanupError;
                std::filesystem::remove_all(partial, cleanupError);
            }
            std::lock_guard<std::mutex> lock(watchdogStateMutex);
            watchdogLastError = "unknown automatic capture error";
        }
    }

    Value capture(const Value *params, Allocator &allocator)
    {
        std::lock_guard<std::mutex> captureLock(captureMutex);
        const std::filesystem::path output = requiredString(params, "output");
        if (!output.is_absolute())
        {
            throw RequestError(-32602, "capture output must be absolute");
        }
        if (std::filesystem::exists(output))
        {
            throw RequestError(-32003, "capture output already exists");
        }
        const Value &requestedScopes = required(params, "scopes");
        if (!requestedScopes.IsArray())
        {
            throw RequestError(-32602, "scopes must be an array");
        }
        std::vector<std::string> scopes;
        for (const Value &scope : requestedScopes.GetArray())
        {
            if (!scope.IsString())
            {
                throw RequestError(-32602, "capture scope must be a string");
            }
            scopes.emplace_back(scope.GetString(), scope.GetStringLength());
        }
        const auto selected = [&](std::string_view scope)
        {
            return std::find(scopes.begin(), scopes.end(), scope) != scopes.end();
        };
        uint64_t historyLimit = 256u;
        if (params)
        {
            const auto history = params->FindMember("history_events");
            if (history != params->MemberEnd())
            {
                if (!history->value.IsUint64() || history->value.GetUint64() == 0u)
                {
                    throw RequestError(-32602,
                                       "history_events must be a positive integer");
                }
                historyLimit = history->value.GetUint64();
            }
        }

        const std::filesystem::path partial =
            output.string() + ".partial-" + std::to_string(getpid());
        if (std::filesystem::exists(partial))
        {
            throw RequestError(-32003, "capture partial output already exists");
        }

        QuiesceGuard guard(runtime);
        std::filesystem::create_directories(partial);
        try
        {
            Document manifest;
            manifest.SetObject();
            Allocator &manifestAllocator = manifest.GetAllocator();
            manifest.AddMember("schema_version", 1, manifestAllocator);
            addString(manifest, "backend", "recomp", manifestAllocator);
            manifest.AddMember("protocol_version",
                               kProtocolVersion, manifestAllocator);
            Value artifacts(rapidjson::kArrayType);
            Value unsupported(rapidjson::kArrayType);

            writeEvents(partial, historyLimit, artifacts, manifestAllocator);
            writeGifStream(
                partial, historyLimit, artifacts, manifestAllocator);
            PS2WatchdogSample sample{};
            PS2WatchdogAssessment assessment{};
            {
                std::lock_guard<std::mutex> lock(watchdogStateMutex);
                sample = watchdogSample;
                assessment = watchdogAssessment;
            }
            writeRuntimeDiagnostics(
                partial, "manual-capture", true, sample, assessment,
                artifacts, manifestAllocator);
            if (selected("ee"))
            {
                writeEeState(partial, artifacts, manifestAllocator);
                writeCodeWindow(partial, artifacts, manifestAllocator);
            }
            if (selected("iop"))
            {
                writeIopState(partial, artifacts, manifestAllocator);
            }
            if (selected("vu0"))
            {
                if (!writeVuState(
                        partial, false, artifacts, manifestAllocator))
                {
                    unsupported.PushBack(
                        Value("vu0-memory", manifestAllocator),
                        manifestAllocator);
                }
            }
            if (selected("vu1"))
            {
                if (!writeVuState(
                        partial, true, artifacts, manifestAllocator))
                {
                    unsupported.PushBack(
                        Value("vu1-memory", manifestAllocator),
                        manifestAllocator);
                }
            }
            if (selected("gs"))
            {
                writeGsState(partial, artifacts, manifestAllocator);
            }
            if (selected("vram"))
            {
                std::vector<uint8_t> vram;
                if (!runtime.debugCopyGsVram(vram))
                {
                    throw RequestError(-32003, "GS VRAM is unavailable");
                }
                const auto path = partial / "final-vram.bin";
                writeBytes(path, vram.data(), vram.size());
                addArtifact(artifacts, path, "final-vram", manifestAllocator);
            }
            if (selected("screenshot") &&
                !writeScreenshot(partial, artifacts, manifestAllocator))
            {
                unsupported.PushBack(Value("screenshot", manifestAllocator),
                                     manifestAllocator);
            }

            manifest.AddMember("artifacts", artifacts, manifestAllocator);
            manifest.AddMember("unsupported_scopes", unsupported,
                               manifestAllocator);
            writeJson(partial / "manifest.json", manifest);
            std::filesystem::rename(partial, output);

            Value result(rapidjson::kObjectType);
            addString(result, "output", output.string(), allocator);
            result.AddMember("complete", true, allocator);
            result.AddMember(
                "artifact_count",
                static_cast<uint64_t>(manifest["artifacts"].Size()), allocator);
            return result;
        }
        catch (...)
        {
            std::error_code cleanupError;
            std::filesystem::remove_all(partial, cleanupError);
            throw;
        }
    }

    Value dispatch(std::string_view method,
                   const Value *params,
                   Allocator &allocator)
    {
        if (method == "system.hello")
        {
            return hello(allocator);
        }
        if (method == "system.status")
        {
            return status(allocator);
        }
        if (method == "diagnostics.status")
        {
            return diagnostics(allocator);
        }
        if (method == "watchdog.status")
        {
            return watchdogStatus(allocator);
        }
        if (method == "execution.pause")
        {
            if (!runtime.debugPause())
            {
                throw RequestError(-32002,
                                   "timed out waiting for a guest safe point");
            }
            return status(allocator);
        }
        if (method == "execution.resume")
        {
            runtime.debugResume();
            return status(allocator);
        }
        if (method == "execution.shutdown")
        {
            runtime.requestStop();
            Value result(rapidjson::kObjectType);
            result.AddMember("requested", true, allocator);
            return result;
        }
        if (method == "state.registers")
        {
            return registers(params, allocator);
        }
        if (method == "vu.backend.set")
        {
            return vuBackendSet(params, allocator);
        }
        if (method == "vu.block-profile")
        {
            return vuBlockProfile(params, allocator);
        }
        if (method == "trace.vu0-sync.start")
        {
            return vu0SyncTraceStart(params, allocator);
        }
        if (method == "trace.vu0-sync.status")
        {
            return vu0SyncTraceValue(false, allocator);
        }
        if (method == "trace.vu0-sync.stop")
        {
            return vu0SyncTraceValue(true, allocator);
        }
        if (method == "trace.vu0-instruction.start")
        {
            return vu0InstructionTraceStart(params, allocator);
        }
        if (method == "trace.vu0-instruction.status")
        {
            return vu0InstructionTraceValue(false, allocator);
        }
        if (method == "trace.vu0-instruction.stop")
        {
            return vu0InstructionTraceValue(true, allocator);
        }
        if (method == "trace.ee-events.start")
        {
            return eeEventTraceStart(params, allocator);
        }
        if (method == "trace.ee-events.status")
        {
            return eeEventTraceValue(false, allocator);
        }
        if (method == "trace.ee-events.stop")
        {
            return eeEventTraceValue(true, allocator);
        }
        if (method == "memory.read")
        {
            return memory(params, false, allocator);
        }
        if (method == "memory.hash")
        {
            return memory(params, true, allocator);
        }
        if (method == "input.pad.status")
        {
            return padStatus(allocator);
        }
        if (method == "input.pad.set" || method == "input.pad.clear")
        {
            return padControl(method, params, allocator);
        }
        if (method == "breakpoint.list")
        {
            return breakpointList(allocator);
        }
        if (method == "breakpoint.add" || method == "breakpoint.remove")
        {
            return breakpointChange(method, params, allocator);
        }
        if (method == "watchpoint.list")
        {
            return watchpointList(allocator);
        }
        if (method == "watchpoint.add")
        {
            return watchpointAdd(params, allocator);
        }
        if (method == "watchpoint.remove")
        {
            return watchpointRemove(params, allocator);
        }
        if (method == "execution.runUntil")
        {
            return runUntil(params, allocator);
        }
        if (method == "execution.step")
        {
            return step(params, allocator);
        }
        if (method == "capture.create")
        {
            return capture(params, allocator);
        }
        throw RequestError(-32601,
                           "method not found: " + std::string(method));
    }

    std::string errorResponse(const Value *id,
                              int code,
                              std::string_view message)
    {
        Document response;
        response.SetObject();
        Allocator &allocator = response.GetAllocator();
        addString(response, "jsonrpc", "2.0", allocator);
        if (id)
        {
            response.AddMember("id", Value(*id, allocator), allocator);
        }
        else
        {
            response.AddMember("id", Value(rapidjson::kNullType), allocator);
        }
        Value error(rapidjson::kObjectType);
        error.AddMember("code", code, allocator);
        addString(error, "message", message, allocator);
        response.AddMember("error", error, allocator);
        return serialize(response) + "\n";
    }

    std::string handle(std::string_view text)
    {
        Document request;
        request.Parse(text.data(), text.size());
        if (request.HasParseError() || !request.IsObject())
        {
            return errorResponse(nullptr, -32700, "parse error");
        }
        const Value *id = nullptr;
        if (const auto member = request.FindMember("id");
            member != request.MemberEnd())
        {
            id = &member->value;
        }

        try
        {
            const auto version = request.FindMember("jsonrpc");
            const auto method = request.FindMember("method");
            if (!id ||
                version == request.MemberEnd() ||
                !version->value.IsString() ||
                std::string_view(version->value.GetString(),
                                 version->value.GetStringLength()) != "2.0" ||
                method == request.MemberEnd() ||
                !method->value.IsString())
            {
                throw RequestError(-32600, "invalid request");
            }

            Document response;
            response.SetObject();
            Allocator &allocator = response.GetAllocator();
            addString(response, "jsonrpc", "2.0", allocator);
            response.AddMember("id", Value(*id, allocator), allocator);
            const std::string_view name(method->value.GetString(),
                                        method->value.GetStringLength());
            response.AddMember(
                "result", dispatch(name, paramsFor(request), allocator), allocator);
            return serialize(response) + "\n";
        }
        catch (const RequestError &error)
        {
            return errorResponse(id, error.code, error.what());
        }
        catch (const std::exception &error)
        {
            return errorResponse(id, -32603, error.what());
        }
        catch (...)
        {
            return errorResponse(id, -32603, "unknown backend error");
        }
    }

    bool readLine(int socketValue, std::string &output)
    {
        output.clear();
        std::array<char, 4096> buffer{};
        while (!stopped.load(std::memory_order_acquire))
        {
            const ssize_t count =
                recv(socketValue, buffer.data(), buffer.size(), 0);
            if (count <= 0)
            {
                return false;
            }
            const char *newline = static_cast<const char *>(
                std::memchr(buffer.data(), '\n', static_cast<size_t>(count)));
            const size_t bytes =
                newline ? static_cast<size_t>(newline - buffer.data())
                        : static_cast<size_t>(count);
            output.append(buffer.data(), bytes);
            if (output.size() > kMaxMessageBytes)
            {
                return false;
            }
            if (newline)
            {
                return true;
            }
        }
        return false;
    }

    void mainLoop()
    {
        ThreadNaming::SetCurrentThreadName("PS2DebugServer");
        while (!stopped.load(std::memory_order_acquire))
        {
            const int server =
                serverSocket.load(std::memory_order_acquire);
            if (server < 0)
            {
                break;
            }
            const int client = accept(server, nullptr, nullptr);
            if (client < 0)
            {
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(socketMutex);
                if (stopped.load(std::memory_order_acquire))
                {
                    close(client);
                    continue;
                }
                clientSocket.store(client, std::memory_order_release);
            }

            std::string request;
            if (readLine(client, request))
            {
                const std::string response = handle(request);
                size_t sent = 0u;
                while (sent < response.size())
                {
                    const ssize_t count = send(
                        client, response.data() + sent,
                        response.size() - sent, MSG_NOSIGNAL);
                    if (count <= 0)
                    {
                        break;
                    }
                    sent += static_cast<size_t>(count);
                }
            }
            {
                std::lock_guard<std::mutex> lock(socketMutex);
                close(client);
                clientSocket.store(-1, std::memory_order_release);
            }
        }
    }

    void watchdogLoop()
    {
        ThreadNaming::SetCurrentThreadName("PS2DbgWatchdog");
        const uint64_t intervalMs = boundedEnvironmentInteger(
            "PS2DBG_WATCHDOG_INTERVAL_MS",
            kDefaultWatchdogIntervalMs, 100u, 60000u);
        uint64_t thresholdMs = boundedEnvironmentInteger(
            "PS2DBG_WATCHDOG_THRESHOLD_MS",
            kDefaultWatchdogThresholdMs, 1000u, 3600000u);
        if (const char *seconds = std::getenv("PS2DBG_WATCHDOG_SECONDS");
            seconds && seconds[0] != '\0')
        {
            thresholdMs =
                boundedEnvironmentInteger(
                    "PS2DBG_WATCHDOG_SECONDS",
                    thresholdMs / 1000u, 1u, 3600u) *
                1000u;
        }

        PS2WatchdogSample previous = takeWatchdogSample();
        uint64_t noPresentationMs = 0u;
        bool capturedForStall = false;
        {
            std::lock_guard<std::mutex> lock(watchdogStateMutex);
            watchdogSample = previous;
            watchdogAssessment = {};
            watchdogAssessment.name = "initializing";
            watchdogAssessment.hotOperation = "";
            watchdogNoPresentationMs = 0u;
            watchdogSampleSequence = 1u;
        }

        while (!stopped.load(std::memory_order_acquire))
        {
            {
                std::unique_lock<std::mutex> lock(watchdogWaitMutex);
                if (watchdogWaitCv.wait_for(
                        lock, std::chrono::milliseconds(intervalMs),
                        [this]()
                        {
                            return stopped.load(std::memory_order_acquire);
                        }))
                {
                    break;
                }
            }

            const PS2WatchdogSample current = takeWatchdogSample();
            PS2WatchdogAssessment assessment =
                ps2ClassifyWatchdogSample(previous, current);
            const bool presentationAdvanced =
                current.presentations > previous.presentations;
            if (presentationAdvanced)
            {
                noPresentationMs = 0u;
                capturedForStall = false;
            }
            else
            {
                noPresentationMs =
                    std::min<uint64_t>(
                        noPresentationMs + intervalMs, 3600000u);
            }

            if (runtime.debugIsPaused())
            {
                assessment.classification =
                    PS2WatchdogClassification::StalledNoProgress;
                assessment.madeProgress = false;
                assessment.name = "paused";
                assessment.hotOperation = "";
            }

            uint64_t sampleSequence = 0u;
            {
                std::lock_guard<std::mutex> lock(watchdogStateMutex);
                watchdogSample = current;
                watchdogAssessment = assessment;
                watchdogNoPresentationMs = noPresentationMs;
                sampleSequence = ++watchdogSampleSequence;
            }

            const PS2Runtime::DebugFaultInfo fault =
                runtime.debugFaultSnapshot();
            if (fault.active &&
                fault.sequence > watchdogLastCapturedFault)
            {
                watchdogLastCapturedFault = fault.sequence;
                capturedForStall = true;
                createAutomaticBundle(
                    "fault", fault.sequence, current, assessment);
            }
            else
            {
                const bool hasGuestActivity =
                    current.dispatches != 0u ||
                    current.eeInstructions != 0u ||
                    current.gsDrawsStarted != 0u ||
                    current.dmaStarts != 0u;
                if (!capturedForStall &&
                    hasGuestActivity &&
                    !runtime.debugIsPaused() &&
                    !runtime.isStopRequested() &&
                    noPresentationMs >= thresholdMs)
                {
                    capturedForStall = true;
                    createAutomaticBundle(
                        "stall", sampleSequence, current, assessment);
                }
            }
            previous = current;
        }
    }

    bool start()
    {
        if (!stopped.load(std::memory_order_acquire))
        {
            return true;
        }
        if (!enabled())
        {
            return false;
        }

        socketPath = defaultSocketPath();
        const std::filesystem::path path(socketPath);
        std::error_code directoryError;
        std::filesystem::create_directories(path.parent_path(), directoryError);
        if (directoryError)
        {
            std::cerr << "[ps2dbg] cannot create socket directory: "
                      << directoryError.message() << '\n';
            return false;
        }
        chmod(path.parent_path().c_str(), 0700);

        const int server = socket(AF_UNIX, SOCK_STREAM, 0);
        if (server < 0)
        {
            return false;
        }
        serverSocket.store(server, std::memory_order_release);
        unlink(socketPath.c_str());

        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        if (socketPath.size() >= sizeof(address.sun_path))
        {
            close(server);
            serverSocket.store(-1, std::memory_order_release);
            return false;
        }
        std::memcpy(address.sun_path, socketPath.c_str(), socketPath.size() + 1u);
        if (bind(server, reinterpret_cast<sockaddr *>(&address),
                 sizeof(address)) != 0 ||
            chmod(socketPath.c_str(), 0600) != 0 ||
            listen(server, 8) != 0)
        {
            close(server);
            serverSocket.store(-1, std::memory_order_release);
            unlink(socketPath.c_str());
            std::cerr << "[ps2dbg] cannot bind " << socketPath << '\n';
            return false;
        }

        stopped.store(false, std::memory_order_release);
        runtime.gs().clearDebugHistory();
        runtime.gs().setDebugHistoryPaused(
            environmentFlagEnabled("PS2DBG_PAUSE_GS_HISTORY"));
        runtime.gs().setProgressTrackingEnabled(true);
        runtime.vu0().setProgressTrackingEnabled(true);
        runtime.vu1().setProgressTrackingEnabled(true);
        thread = std::thread([this]()
                             { mainLoop(); });
        watchdogThread = std::thread([this]()
                                     { watchdogLoop(); });
        if (environmentFlagEnabled("PS2DBG_START_PAUSED"))
        {
            (void)runtime.debugPause(std::chrono::milliseconds(0));
        }
        std::cout << "[ps2dbg] listening on " << socketPath << '\n';
        return true;
    }

    void stop()
    {
        if (stopped.exchange(true, std::memory_order_acq_rel))
        {
            return;
        }
        watchdogWaitCv.notify_all();
        const int server =
            serverSocket.exchange(-1, std::memory_order_acq_rel);
        if (server >= 0)
        {
            shutdown(server, SHUT_RDWR);
            close(server);
        }
        {
            std::lock_guard<std::mutex> lock(socketMutex);
            const int client =
                clientSocket.load(std::memory_order_acquire);
            if (client >= 0)
            {
                shutdown(client, SHUT_RDWR);
            }
        }
        if (thread.joinable())
        {
            thread.join();
        }
        clientSocket.store(-1, std::memory_order_release);
        if (watchdogThread.joinable())
        {
            watchdogThread.join();
        }
        runtime.gs().setProgressTrackingEnabled(false);
        runtime.vu0().setProgressTrackingEnabled(false);
        runtime.vu1().setProgressTrackingEnabled(false);
        runtime.gs().setDebugHistoryPaused(true);
        if (!socketPath.empty())
        {
            unlink(socketPath.c_str());
            socketPath.clear();
        }
    }
};

PS2DebugServer::PS2DebugServer(PS2Runtime &runtime)
    : m_impl(std::make_unique<Impl>(runtime))
{
}

PS2DebugServer::~PS2DebugServer()
{
    stop();
}

bool PS2DebugServer::start()
{
    return m_impl->start();
}

void PS2DebugServer::stop()
{
    m_impl->stop();
}

bool PS2DebugServer::isRunning() const
{
    return !m_impl->stopped.load(std::memory_order_acquire);
}
