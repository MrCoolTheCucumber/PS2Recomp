#include "runtime/ps2_memory.h"
#include "runtime/ps2_address.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_vu1.h"
#include "ps2_log.h"
#include <array>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{
    struct DmaChainState
    {
        uint32_t tagAddress;
        uint32_t asr0;
        uint32_t asr1;
        uint32_t asp;

        bool operator==(const DmaChainState &other) const
        {
            return tagAddress == other.tagAddress &&
                   asr0 == other.asr0 &&
                   asr1 == other.asr1 &&
                   asp == other.asp;
        }
    };

    struct DmaChainStateHash
    {
        size_t operator()(const DmaChainState &state) const
        {
            size_t hash = static_cast<size_t>(state.tagAddress);
            hash ^= static_cast<size_t>(state.asr0) +
                    static_cast<size_t>(0x9E3779B9u) + (hash << 6u) + (hash >> 2u);
            hash ^= static_cast<size_t>(state.asr1) +
                    static_cast<size_t>(0x9E3779B9u) + (hash << 6u) + (hash >> 2u);
            hash ^= static_cast<size_t>(state.asp) +
                    static_cast<size_t>(0x9E3779B9u) + (hash << 6u) + (hash >> 2u);
            return hash;
        }
    };

    inline void inRange(uint32_t offset, size_t bytes, size_t regionSize, const char *op, uint32_t address)
    {
        if (static_cast<uint64_t>(offset) + static_cast<uint64_t>(bytes) > static_cast<uint64_t>(regionSize))
        {
            throw std::runtime_error(std::string(op) + " out-of-bounds at address: 0x" + std::to_string(address));
        }
    }

    template <typename T>
    inline T loadScalar(const uint8_t *base, uint32_t offset, size_t regionSize, const char *op, uint32_t address)
    {
        inRange(offset, sizeof(T), regionSize, op, address);
        T value{};
        std::memcpy(&value, base + offset, sizeof(T));
        return value;
    }

    template <typename T>
    inline void storeScalar(uint8_t *base, uint32_t offset, size_t regionSize, T value, const char *op, uint32_t address)
    {
        inRange(offset, sizeof(T), regionSize, op, address);
        std::memcpy(base + offset, &value, sizeof(T));
    }

    inline bool isGsPrivReg(uint32_t addr)
    {
        return Ps2AddressInRange(addr, PS2_GS_PRIV_REG_BASE, PS2_GS_PRIV_REG_SIZE);
    }

    inline bool isIoRegister(uint32_t addr)
    {
        return Ps2AddressInRange(addr, PS2_IO_BASE, PS2_IO_SIZE);
    }

    inline uint64_t *gsRegPtr(GSRegisters &gs, uint32_t addr)
    {
        // Support both 64-bit base offsets and +4 dword aliases.
        uint32_t off = (addr - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        switch (off)
        {
        case 0x0000:
            return &gs.pmode;
        case 0x0010:
            return &gs.smode1;
        case 0x0020:
            return &gs.smode2;
        case 0x0030:
            return &gs.srfsh;
        case 0x0040:
            return &gs.synch1;
        case 0x0050:
            return &gs.synch2;
        case 0x0060:
            return &gs.syncv;
        case 0x0070:
            return &gs.dispfb1;
        case 0x0080:
            return &gs.display1;
        case 0x0090:
            return &gs.dispfb2;
        case 0x00A0:
            return &gs.display2;
        case 0x00B0:
            return &gs.extbuf;
        case 0x00C0:
            return &gs.extdata;
        case 0x00D0:
            return &gs.extwrite;
        case 0x00E0:
            return &gs.bgcolor;
        // CSR (offset 0x1000) is intentionally not handled here: it is
        // std::atomic<uint64_t> and no longer converts to uint64_t*. Callers must
        // check for offset 0x1000 themselves and go through writeCsrHalf/
        // writeCsrFull/gs.csr.load() instead of gsRegPtr().
        case 0x1010:
            return &gs.imr;
        case 0x1040:
            return &gs.busdir;
        case 0x1080:
            return &gs.siglblid;
        default:
            return nullptr;
        }
    }

    constexpr uint32_t kGsCsrRegOffset = 0x1000u;

    // Atomically apply a 32-bit write to one half (off=0 low dword, off=4 high
    // dword) of the GS CSR register. Bits 0..1 of the low dword (SIGNAL/FINISH) are
    // write-one-to-clear; everything else is a plain merge. Uses compare_exchange
    // so the whole read-modify-write is a single atomic step -- this register is
    // also touched by the vsync worker (FIELD bit) and the GIF (SIGNAL/FINISH) on
    // other threads, so a load-then-store here would race with them.
    inline void writeCsrHalf(std::atomic<uint64_t> &csr, uint32_t off, uint32_t value)
    {
        constexpr uint32_t kW1cMask = 0x3u;
        uint64_t expected = csr.load();
        uint64_t desired;
        do
        {
            if (off == 0u)
            {
                uint32_t oldLow = static_cast<uint32_t>(expected & 0xFFFFFFFFull);
                uint32_t mergedLow = (oldLow & kW1cMask) | (value & ~kW1cMask);
                desired = (expected & 0xFFFFFFFF00000000ull) | static_cast<uint64_t>(mergedLow);
                desired &= ~static_cast<uint64_t>(value & kW1cMask);
            }
            else
            {
                uint64_t mask = 0xFFFFFFFFull << (off * 8u);
                desired = (expected & ~mask) | (static_cast<uint64_t>(value) << (off * 8u));
            }
        } while (!csr.compare_exchange_weak(expected, desired));
    }

    // Same as writeCsrHalf but for a full 64-bit CSR write (bits 0..1 are still
    // write-one-to-clear against the current value).
    inline void writeCsrFull(std::atomic<uint64_t> &csr, uint64_t value)
    {
        constexpr uint64_t kW1cMask = 0x3ull;
        uint64_t expected = csr.load();
        uint64_t desired;
        do
        {
            desired = (expected & kW1cMask) | (value & ~kW1cMask);
            desired &= ~(value & kW1cMask);
        } while (!csr.compare_exchange_weak(expected, desired));
    }

    constexpr uint32_t kEeTimer0Count = 0x10000000u;
    constexpr uint32_t kEeTimer0Mode = 0x10000010u;
    constexpr uint32_t kEeTimer0Compare = 0x10000020u;
    constexpr uint32_t kEeTimer0Hold = 0x10000030u;
    constexpr uint32_t kEeTimerModeCue = 1u << 7;
    constexpr uint64_t kEeTimer0TicksPerSecond = 15720ull;
    constexpr uint64_t kNanosecondsPerSecond = 1000000000ull;

    inline bool isEeTimer0Register(uint32_t address)
    {
        return address == kEeTimer0Count ||
               address == kEeTimer0Mode ||
               address == kEeTimer0Compare ||
               address == kEeTimer0Hold;
    }

    inline uint64_t steadyClockNs()
    {
        using namespace std::chrono;
        return static_cast<uint64_t>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
    }

    struct DmaTagView
    {
        uint16_t qwc = 0;
        uint8_t id = 0;
        bool irq = false;
        uint32_t addr = 0;
        uint32_t upper = 0;
    };

    inline DmaTagView decodeDmaTag(uint64_t tag)
    {
        DmaTagView out{};
        out.qwc = static_cast<uint16_t>(tag & 0xFFFFu);
        out.id = static_cast<uint8_t>((tag >> 28u) & 0x7u);
        out.irq = ((tag >> 31u) & 0x1ull) != 0ull;
        out.addr = static_cast<uint32_t>((tag >> 32u) & 0x7FFFFFFFu);
        out.upper = static_cast<uint32_t>((tag >> 16u) & 0xFFFFu);
        return out;
    }

    inline uint32_t gifTagNloop(uint64_t tagLo)
    {
        return static_cast<uint32_t>(tagLo & 0x7FFFu);
    }

    inline uint8_t gifTagFlg(uint64_t tagLo)
    {
        return static_cast<uint8_t>((tagLo >> 58u) & 0x3u);
    }

    inline uint32_t gifTagNreg(uint64_t tagLo)
    {
        uint32_t nreg = static_cast<uint32_t>((tagLo >> 60u) & 0xFu);
        return nreg == 0u ? 16u : nreg;
    }

}

std::atomic<bool> g_ps2GsDmaWriteTraceEnabled{false};
std::atomic<bool> g_ps2GuestWriteTraceEnabled{[]
{
    const char *path = std::getenv("PS2X_GUEST_WRITE_TRACE");
    return path && *path;
}()};
namespace
{
    Ps2GsDmaTraceState *g_ps2GsDmaTraceState = nullptr;

    struct GuestWriteTraceState
    {
        std::FILE *output = nullptr;
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        uint64_t sequence = 0u;
        std::mutex mutex;

        GuestWriteTraceState()
        {
            const char *path = std::getenv("PS2X_GUEST_WRITE_TRACE");
            if (!path || !*path)
                return;

            output = std::fopen(path, "wb");
            if (!output)
            {
                std::fprintf(stderr,
                             "Guest write trace: cannot open '%s': %s\n",
                             path, std::strerror(errno));
                return;
            }

            const char *rangesText = std::getenv("PS2X_GUEST_WRITE_RANGES");
            if (!parseRanges(rangesText))
            {
                std::fprintf(stderr,
                             "Guest write trace: invalid PS2X_GUEST_WRITE_RANGES '%s'\n",
                             rangesText ? rangesText : "");
                std::fclose(output);
                output = nullptr;
                ranges.clear();
                return;
            }

            std::fprintf(output,
                         "{\"schema_version\":1,\"event\":\"configuration\","
                         "\"ranges\":\"%s\"}\n",
                         rangesText ? rangesText : "");
            std::fflush(output);
        }

        ~GuestWriteTraceState()
        {
            if (output)
                std::fclose(output);
        }

        bool parseRanges(const char *text)
        {
            if (!text || !*text)
                return false;

            const std::string rangesText(text);
            size_t position = 0u;
            while (position < rangesText.size())
            {
                const size_t comma = rangesText.find(',', position);
                const size_t end = comma == std::string::npos ? rangesText.size() : comma;
                const std::string range = rangesText.substr(position, end - position);
                const size_t dash = range.find('-');
                if (dash == std::string::npos)
                    return false;

                auto parseAddress = [](const std::string &addressText,
                                       uint32_t &address)
                {
                    errno = 0;
                    char *parseEnd = nullptr;
                    const unsigned long long parsed =
                        std::strtoull(addressText.c_str(), &parseEnd, 0);
                    if (errno != 0 || parseEnd == addressText.c_str() ||
                        *parseEnd != '\0' ||
                        parsed > std::numeric_limits<uint32_t>::max())
                    {
                        return false;
                    }
                    address = static_cast<uint32_t>(parsed);
                    return true;
                };

                uint32_t rangeBegin = 0u;
                uint32_t rangeEnd = 0u;
                if (!parseAddress(range.substr(0u, dash), rangeBegin) ||
                    !parseAddress(range.substr(dash + 1u), rangeEnd) ||
                    rangeBegin >= rangeEnd)
                {
                    return false;
                }
                ranges.emplace_back(rangeBegin, rangeEnd);
                position = end + (comma == std::string::npos ? 0u : 1u);
            }
            return !ranges.empty();
        }

        void record(uint32_t guestAddr,
                    uint32_t size,
                    uint64_t valueLo,
                    uint64_t valueHi,
                    const char *op,
                    uint32_t writerPc,
                    uint32_t returnAddress)
        {
            if (!output || size == 0u)
                return;

            uint32_t physicalAddress = 0u;
            if (!ps2ResolveDirectRdramOffset(guestAddr, physicalAddress))
                return;

            const uint64_t writeEnd =
                static_cast<uint64_t>(physicalAddress) + size;
            bool overlaps = false;
            for (const auto &[rangeBegin, rangeEnd] : ranges)
            {
                if (physicalAddress < rangeEnd && writeEnd > rangeBegin)
                {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps)
                return;

            std::lock_guard<std::mutex> lock(mutex);
            std::fprintf(
                output,
                "{\"schema_version\":1,\"event\":\"guest-write\","
                "\"sequence\":%" PRIu64 ",\"pc\":\"0x%08" PRIx32 "\","
                "\"ra\":\"0x%08" PRIx32 "\",\"address\":\"0x%08" PRIx32 "\","
                "\"size\":%" PRIu32 ",\"op\":\"%s\","
                "\"value_lo\":\"0x%016" PRIx64 "\","
                "\"value_hi\":\"0x%016" PRIx64 "\"}\n",
                sequence++, writerPc, returnAddress, physicalAddress,
                size, op ? op : "", valueLo, valueHi);
            std::fflush(output);
        }
    };

    GuestWriteTraceState &getGuestWriteTraceState()
    {
        static GuestWriteTraceState trace;
        return trace;
    }
}

void ps2RecordGuestWriteTrace(uint32_t guestAddr,
                              uint32_t size,
                              uint64_t valueLo,
                              uint64_t valueHi,
                              const char *op,
                              uint32_t writerPc,
                              uint32_t returnAddress)
{
    getGuestWriteTraceState().record(
        guestAddr, size, valueLo, valueHi, op, writerPc, returnAddress);
}

struct Ps2GsDmaTraceState
{
    static constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ull;
    static constexpr uint64_t FNV_PRIME = 1099511628211ull;
    static constexpr uint64_t MAX_BYTES = 64ull * 1024ull * 1024ull;
    static constexpr size_t PREFIX_SIZE = 64u;

    struct Digest
    {
        uint64_t hash = FNV_OFFSET_BASIS;
        uint64_t bytes = 0u;
        bool truncated = false;
        std::array<uint8_t, PREFIX_SIZE> prefix{};
        size_t prefixSize = 0u;

        void append(const uint8_t *data, uint64_t size)
        {
            if (!data || bytes >= MAX_BYTES)
            {
                truncated = truncated || size != 0u;
                return;
            }

            const uint64_t available = MAX_BYTES - bytes;
            const uint64_t boundedSize = std::min(size, available);
            for (uint64_t i = 0; i < boundedSize; ++i)
            {
                const uint8_t value = data[i];
                hash ^= value;
                hash *= FNV_PRIME;
                if (prefixSize < prefix.size())
                    prefix[prefixSize++] = value;
            }
            bytes += boundedSize;
            truncated = truncated || boundedSize != size;
        }
    };

    struct EeWriteSummary
    {
        uint64_t calls = 0u;
        uint64_t bytes = 0u;
        uint32_t firstAddress = 0u;
        uint32_t lastAddress = 0u;
        uint32_t minAddress = std::numeric_limits<uint32_t>::max();
        uint32_t maxAddress = 0u;
    };

    struct Vu1RunPrefix
    {
        struct VifCommand
        {
            uint32_t command = 0u;
            uint32_t cycle = 0u;
            uint32_t mode = 0u;
            uint32_t tops = 0u;
            std::array<uint8_t, 64u> following{};
            size_t followingSize = 0u;
        };

        uint64_t ordinal = 0u;
        uint32_t startPc = 0u;
        uint32_t top = 0u;
        uint32_t itop = 0u;
        uint32_t finalPc = 0u;
        uint64_t microHash = FNV_OFFSET_BASIS;
        uint64_t instructionCount = 0u;
        bool ended = false;
        bool hitCycleLimit = false;
        std::array<int32_t, 16u> initialVi{};
        std::array<int32_t, 16u> finalVi{};
        std::array<uint8_t, 256u> topData{};
        std::vector<VifCommand> recentVifCommands;
        std::vector<uint32_t> pcs;
        std::vector<uint32_t> lower;
        std::vector<uint32_t> upper;
    };

    bool configured = false;
    bool enabled = false;
    bool active = false;
    std::FILE *output = nullptr;
    uint64_t sequence = 0u;
    uint64_t activeSequence = 0u;
    uint64_t limit = std::numeric_limits<uint64_t>::max();
    uint32_t chcr = 0u;
    uint32_t madr = 0u;
    uint32_t qwc = 0u;
    uint32_t tadr = 0u;
    uint32_t asr0 = 0u;
    uint32_t asr1 = 0u;
    uint32_t sadr = 0u;
    uint32_t tagCount = 0u;
    uint64_t path1Calls = 0u;
    uint64_t vu1Runs = 0u;
    uint64_t vu1Resumes = 0u;
    uint64_t vu1Pairs = 0u;
    uint64_t vu1Xgkicks = 0u;
    uint64_t vu1EndedRuns = 0u;
    uint64_t vu1CycleLimitRuns = 0u;
    uint32_t vu1LastPc = 0u;
    uint32_t vu1CurrentStartPc = 0u;
    uint32_t vu1CurrentTop = 0u;
    uint32_t vu1CurrentItop = 0u;
    uint64_t vu1CurrentMicroHash = FNV_OFFSET_BASIS;
    uint64_t vu1CurrentPath1Calls = 0u;
    Digest vu1CurrentPath1{};
    uint64_t vu1StepSequence = std::numeric_limits<uint64_t>::max();
    uint64_t vu1StepOrdinal = std::numeric_limits<uint64_t>::max();
    uint64_t vu1StepEvents = 0u;
    std::array<int32_t, 16u> vu1CurrentInitialVi{};
    std::array<uint8_t, 256u> vu1CurrentTopData{};
    std::array<uint64_t, 2048u> vu1RunStarts{};
    std::array<uint64_t, 2048u> vu1EndedStarts{};
    std::array<uint64_t, 2048u> vu1CycleLimitStarts{};
    std::array<uint64_t, 128u> vu1LowerOps{};
    std::array<uint64_t, 128u> vu1LowerSpecialOps{};
    std::array<uint64_t, 64u> vu1UpperOps{};
    std::array<uint64_t, 128u> vu1UpperSpecialOps{};
    std::vector<uint32_t> vu1StartPcs;
    std::vector<uint32_t> vu1KickAddresses;
    std::vector<Vu1RunPrefix> vu1RunPrefix;
    std::vector<Vu1RunPrefix::VifCommand> recentVifCommands;
    std::array<uint32_t, 16u> vu1CurrentPcs{};
    std::array<uint32_t, 16u> vu1CurrentLower{};
    std::array<uint32_t, 16u> vu1CurrentUpper{};
    size_t vu1CurrentInstructionCount = 0u;
    bool vu1FirstCycleLimitCaptured = false;
    uint32_t vu1FirstCycleLimitStartPc = 0u;
    uint32_t vu1FirstCycleLimitTop = 0u;
    uint32_t vu1FirstCycleLimitItop = 0u;
    uint32_t vu1FirstCycleLimitFinalPc = 0u;
    uint64_t vu1FirstCycleLimitOrdinal = 0u;
    uint64_t vu1FirstCycleLimitMicroHash = FNV_OFFSET_BASIS;
    std::array<int32_t, 16u> vu1FirstCycleLimitInitialVi{};
    std::array<int32_t, 16u> vu1FirstCycleLimitVi{};
    std::array<uint8_t, 256u> vu1FirstCycleLimitTopData{};
    std::vector<uint32_t> vu1FirstCycleLimitPcs;
    std::vector<uint32_t> vu1FirstCycleLimitLower;
    std::vector<uint32_t> vu1FirstCycleLimitUpper;
    Digest packet{};
    Digest vifInput{};
    Digest path1{};
    std::string path1DumpDirectory;
    std::string vif1InputDumpDirectory;
    std::string vu1DumpDirectory;
    std::vector<std::pair<uint32_t, uint32_t>> writeRanges;
    std::vector<uint32_t> watchAddresses;
    std::vector<uint32_t> watchValues;
    std::unordered_map<uint64_t, EeWriteSummary> eeWrites;

    ~Ps2GsDmaTraceState()
    {
        if (g_ps2GsDmaTraceState == this)
        {
            g_ps2GsDmaWriteTraceEnabled.store(false, std::memory_order_relaxed);
            g_ps2GsDmaTraceState = nullptr;
        }
        if (output)
            std::fclose(output);
    }

    bool parseWriteRanges(const char *text)
    {
        if (!text || !*text)
            return true;

        const std::string rangesText(text);
        size_t position = 0u;
        while (position < rangesText.size())
        {
            const size_t comma = rangesText.find(',', position);
            const size_t end = comma == std::string::npos ? rangesText.size() : comma;
            const std::string range = rangesText.substr(position, end - position);
            const size_t dash = range.find('-');
            if (dash == std::string::npos)
                return false;

            auto parseAddress = [](const std::string &addressText, uint32_t &address)
            {
                errno = 0;
                char *parseEnd = nullptr;
                const unsigned long long parsed =
                    std::strtoull(addressText.c_str(), &parseEnd, 0);
                if (errno != 0 || parseEnd == addressText.c_str() ||
                    *parseEnd != '\0' || parsed > std::numeric_limits<uint32_t>::max())
                {
                    return false;
                }
                address = static_cast<uint32_t>(parsed);
                return true;
            };

            uint32_t beginAddress = 0u;
            uint32_t endAddress = 0u;
            if (!parseAddress(range.substr(0u, dash), beginAddress) ||
                !parseAddress(range.substr(dash + 1u), endAddress) ||
                beginAddress >= endAddress)
            {
                return false;
            }
            writeRanges.emplace_back(beginAddress, endAddress);
            position = end + (comma == std::string::npos ? 0u : 1u);
        }
        return true;
    }

    bool parseWatchAddresses(const char *text)
    {
        if (!text || !*text)
            return true;

        const char *cursor = text;
        while (*cursor)
        {
            errno = 0;
            char *end = nullptr;
            const unsigned long long parsed = std::strtoull(cursor, &end, 0);
            if (errno != 0 || end == cursor ||
                (*end != '\0' && *end != ',') ||
                parsed > std::numeric_limits<uint32_t>::max())
            {
                return false;
            }
            watchAddresses.push_back(static_cast<uint32_t>(parsed));
            if (*end == '\0')
                break;
            cursor = end + 1;
            if (*cursor == '\0')
                return false;
        }
        return true;
    }

    void configure()
    {
        if (configured)
            return;
        configured = true;

        const char *path = std::getenv("PS2X_GS_DMA_TRACE");
        if (!path || !*path)
            return;

        const char *limitText = std::getenv("PS2X_GS_DMA_TRACE_LIMIT");
        if (limitText && *limitText)
        {
            errno = 0;
            char *end = nullptr;
            const unsigned long long parsed = std::strtoull(limitText, &end, 0);
            if (errno != 0 || end == limitText || *end != '\0' || parsed == 0u)
            {
                std::fprintf(stderr,
                             "PS2 GS DMA trace: invalid PS2X_GS_DMA_TRACE_LIMIT '%s'\n",
                             limitText);
                return;
            }
            limit = static_cast<uint64_t>(parsed);
        }

        output = std::fopen(path, "wb");
        if (!output)
        {
            std::fprintf(stderr, "PS2 GS DMA trace: cannot open '%s': %s\n",
                         path, std::strerror(errno));
            return;
        }

        enabled = true;
        auto parseOptionalIndex = [](const char *name, uint64_t &value)
        {
            const char *text = std::getenv(name);
            if (!text || !*text)
                return true;
            errno = 0;
            char *end = nullptr;
            const unsigned long long parsed = std::strtoull(text, &end, 0);
            if (errno != 0 || end == text || *end != '\0')
            {
                std::fprintf(stderr, "PS2 GS DMA trace: invalid %s '%s'\n",
                             name, text);
                return false;
            }
            value = static_cast<uint64_t>(parsed);
            return true;
        };
        if (!parseOptionalIndex("PS2X_VU1_STEP_SEQUENCE", vu1StepSequence) ||
            !parseOptionalIndex("PS2X_VU1_STEP_ORDINAL", vu1StepOrdinal))
        {
            return;
        }
        if (const char *path1DumpDirectoryText =
                std::getenv("PS2X_PATH1_DUMP_DIR");
            path1DumpDirectoryText && *path1DumpDirectoryText)
        {
            path1DumpDirectory = path1DumpDirectoryText;
        }
        if (const char *vif1InputDumpDirectoryText =
                std::getenv("PS2X_VIF1_INPUT_DUMP_DIR");
            vif1InputDumpDirectoryText && *vif1InputDumpDirectoryText)
        {
            vif1InputDumpDirectory = vif1InputDumpDirectoryText;
        }
        if (const char *vu1DumpDirectoryText =
                std::getenv("PS2X_VU1_DUMP_DIR");
            vu1DumpDirectoryText && *vu1DumpDirectoryText)
        {
            vu1DumpDirectory = vu1DumpDirectoryText;
        }
        const char *writeRangesText = std::getenv("PS2X_GS_DMA_WRITE_RANGES");
        if (!parseWriteRanges(writeRangesText))
        {
            std::fprintf(stderr,
                         "PS2 GS DMA trace: invalid PS2X_GS_DMA_WRITE_RANGES '%s'\n",
                         writeRangesText ? writeRangesText : "");
            writeRanges.clear();
        }
        if (!writeRanges.empty())
        {
            g_ps2GsDmaTraceState = this;
            g_ps2GsDmaWriteTraceEnabled.store(true, std::memory_order_relaxed);
        }
        const char *watchAddressesText =
            std::getenv("PS2X_GS_DMA_WATCH_ADDRESSES");
        if (!parseWatchAddresses(watchAddressesText))
        {
            std::fprintf(stderr,
                         "PS2 GS DMA trace: invalid PS2X_GS_DMA_WATCH_ADDRESSES '%s'\n",
                         watchAddressesText ? watchAddressesText : "");
            watchAddresses.clear();
        }
        std::fprintf(output,
                     "{\"schema_version\":1,\"event\":\"configuration\","
                     "\"producer\":\"PS2Recomp\",\"limit\":%" PRIu64 ","
                     "\"ee_write_ranges\":\"%s\","
                     "\"watch_addresses\":\"%s\"}\n",
                     limit, writeRangesText ? writeRangesText : "",
                     watchAddressesText ? watchAddressesText : "");
        std::fflush(output);
    }

    static void writeHex(std::FILE *file, const Digest &digest)
    {
        for (size_t i = 0; i < digest.prefixSize; ++i)
            std::fprintf(file, "%02" PRIx8, digest.prefix[i]);
    }

    void recordEeWrite(uint32_t guestAddr,
                       uint32_t size,
                       uint32_t writerPc,
                       uint32_t returnAddress)
    {
        uint32_t physicalAddress = 0u;
        if (size == 0u || !ps2ResolveDirectRdramOffset(guestAddr, physicalAddress))
            return;

        const uint64_t writeEnd = static_cast<uint64_t>(physicalAddress) + size;
        for (const auto &[rangeBegin, rangeEnd] : writeRanges)
        {
            if (physicalAddress >= rangeEnd || writeEnd <= rangeBegin)
                continue;

            const uint32_t clippedBegin = std::max(physicalAddress, rangeBegin);
            const uint32_t clippedEnd = static_cast<uint32_t>(
                std::min<uint64_t>(writeEnd, static_cast<uint64_t>(rangeEnd)));
            const uint64_t key =
                (static_cast<uint64_t>(writerPc) << 32u) | returnAddress;
            EeWriteSummary &summary = eeWrites[key];
            if (summary.calls == 0u)
                summary.firstAddress = clippedBegin;
            summary.lastAddress = clippedBegin;
            summary.minAddress = std::min(summary.minAddress, clippedBegin);
            summary.maxAddress = std::max(summary.maxAddress, clippedEnd - 1u);
            ++summary.calls;
            summary.bytes += clippedEnd - clippedBegin;
        }
    }

    void flushEeWrites(uint64_t dmacSequence)
    {
        if (eeWrites.empty() || !output)
            return;

        std::vector<std::pair<uint64_t, EeWriteSummary>> summaries(
            eeWrites.begin(), eeWrites.end());
        std::sort(summaries.begin(), summaries.end(),
                  [](const auto &left, const auto &right)
                  {
                      return left.first < right.first;
                  });

        for (const auto &[key, summary] : summaries)
        {
            const uint32_t writerPc = static_cast<uint32_t>(key >> 32u);
            const uint32_t returnAddress = static_cast<uint32_t>(key);
            std::fprintf(output,
                         "{\"schema_version\":1,\"event\":\"ee-write-summary\","
                         "\"dmac_sequence\":%" PRIu64 ","
                         "\"writer_pc\":\"0x%08" PRIx32 "\","
                         "\"ra\":\"0x%08" PRIx32 "\","
                         "\"calls\":%" PRIu64 ",\"bytes\":%" PRIu64 ","
                         "\"first_address\":\"0x%08" PRIx32 "\","
                         "\"last_address\":\"0x%08" PRIx32 "\","
                         "\"min_address\":\"0x%08" PRIx32 "\","
                         "\"max_address\":\"0x%08" PRIx32 "\"}\n",
                         dmacSequence, writerPc, returnAddress,
                         summary.calls, summary.bytes,
                         summary.firstAddress, summary.lastAddress,
                         summary.minAddress, summary.maxAddress);
        }
        std::fflush(output);
        eeWrites.clear();
    }

    void captureWatches(const uint8_t *rdram, const uint8_t *scratchpad)
    {
        watchValues.clear();
        watchValues.reserve(watchAddresses.size());
        for (uint32_t address : watchAddresses)
        {
            uint32_t physicalAddress = 0u;
            uint32_t value = 0u;
            if (scratchpad && ps2IsScratchpadAddress(address))
            {
                physicalAddress = ps2ScratchpadOffset(address);
                if (physicalAddress <= PS2_SCRATCHPAD_SIZE - sizeof(value))
                    std::memcpy(&value, scratchpad + physicalAddress, sizeof(value));
            }
            else if (rdram &&
                ps2ResolveDirectRdramOffset(address, physicalAddress) &&
                physicalAddress <= PS2_RAM_SIZE - sizeof(value))
            {
                std::memcpy(&value, rdram + physicalAddress, sizeof(value));
            }
            watchValues.push_back(value);
        }
    }

    void finish()
    {
        if (!active || !output)
            return;

        auto writeSparseCounts = [&](const auto &counts)
        {
            bool first = true;
            for (size_t index = 0u; index < counts.size(); ++index)
            {
                if (counts[index] == 0u)
                    continue;
                std::fprintf(output, "%s\"0x%02zx\":%" PRIu64,
                             first ? "" : ",", index, counts[index]);
                first = false;
            }
        };
        auto writeSparsePcCounts = [&](const auto &counts)
        {
            bool first = true;
            for (size_t index = 0u; index < counts.size(); ++index)
            {
                if (counts[index] == 0u)
                    continue;
                std::fprintf(output, "%s\"0x%04zx\":%" PRIu64,
                             first ? "" : ",", index * 8u, counts[index]);
                first = false;
            }
        };

        std::fprintf(output,
                     "{\"schema_version\":1,\"event\":\"vif1-segment\","
                     "\"sequence\":%" PRIu64 ",\"chcr_write\":\"0x%08" PRIx32 "\","
                     "\"mode\":%" PRIu32 ",\"tte\":%s,"
                     "\"madr\":\"0x%08" PRIx32 "\",\"qwc\":%" PRIu32 ","
                     "\"tadr\":\"0x%08" PRIx32 "\","
                     "\"asr0\":\"0x%08" PRIx32 "\","
                     "\"asr1\":\"0x%08" PRIx32 "\","
                     "\"sadr\":\"0x%08" PRIx32 "\","
                     "\"packet_hash_fnv1a64\":\"0x%016" PRIx64 "\","
                     "\"packet_bytes\":%" PRIu64 ",\"tag_count\":%" PRIu32 ","
                     "\"packet_truncated\":%s,\"packet_prefix\":\"",
                     activeSequence, chcr, (chcr >> 2u) & 0x3u,
                     (chcr & (1u << 6u)) != 0u ? "true" : "false",
                     madr, qwc, tadr, asr0, asr1, sadr,
                     packet.hash, packet.bytes, tagCount,
                     packet.truncated ? "true" : "false");
        writeHex(output, packet);
        std::fprintf(output,
                     "\",\"vif_hash_fnv1a64\":\"0x%016" PRIx64 "\","
                     "\"vif_bytes\":%" PRIu64 ",\"vif_truncated\":%s,"
                     "\"vif_prefix\":\"",
                     vifInput.hash, vifInput.bytes,
                     vifInput.truncated ? "true" : "false");
        writeHex(output, vifInput);
        std::fprintf(output,
                     "\",\"path1_calls\":%" PRIu64 ","
                     "\"path1_hash_fnv1a64\":\"0x%016" PRIx64 "\","
                     "\"path1_bytes\":%" PRIu64 ",\"path1_truncated\":%s,"
                     "\"path1_prefix\":\"",
                     path1Calls, path1.hash, path1.bytes,
                     path1.truncated ? "true" : "false");
        writeHex(output, path1);
        std::fprintf(output,
                     "\",\"vu1_runs\":%" PRIu64 ",\"vu1_resumes\":%" PRIu64 ","
                     "\"vu1_pairs\":%" PRIu64 ",\"vu1_xgkicks\":%" PRIu64 ","
                     "\"vu1_ended_runs\":%" PRIu64 ","
                     "\"vu1_cycle_limit_runs\":%" PRIu64 ","
                     "\"vu1_last_pc\":\"0x%04" PRIx32 "\","
                     "\"vu1_start_pcs\":[",
                     vu1Runs, vu1Resumes, vu1Pairs, vu1Xgkicks,
                     vu1EndedRuns, vu1CycleLimitRuns, vu1LastPc);
        for (size_t i = 0u; i < vu1StartPcs.size(); ++i)
        {
            std::fprintf(output, "%s\"0x%04" PRIx32 "\"",
                         i == 0u ? "" : ",", vu1StartPcs[i]);
        }
        std::fputs("],\"vu1_kick_addresses\":[", output);
        for (size_t i = 0u; i < vu1KickAddresses.size(); ++i)
        {
            std::fprintf(output, "%s\"0x%03" PRIx32 "\"",
                         i == 0u ? "" : ",", vu1KickAddresses[i]);
        }
        std::fputs("],\"vu1_run_prefix\":[", output);
        for (size_t runIndex = 0u; runIndex < vu1RunPrefix.size(); ++runIndex)
        {
            const Vu1RunPrefix &run = vu1RunPrefix[runIndex];
            std::fprintf(output,
                         "%s{\"ordinal\":%" PRIu64 ","
                         "\"start_pc\":\"0x%04" PRIx32 "\","
                         "\"top\":\"0x%03" PRIx32 "\","
                         "\"itop\":\"0x%03" PRIx32 "\","
                         "\"final_pc\":\"0x%04" PRIx32 "\","
                         "\"pairs\":%" PRIu64 ",\"ended\":%s,"
                         "\"cycle_limit\":%s,"
                         "\"micro_hash_fnv1a64\":\"0x%016" PRIx64 "\","
                         "\"initial_vi\":[",
                         runIndex == 0u ? "" : ",", run.ordinal,
                         run.startPc, run.top, run.itop, run.finalPc,
                         run.instructionCount,
                         run.ended ? "true" : "false",
                         run.hitCycleLimit ? "true" : "false",
                         run.microHash);
            for (size_t i = 0u; i < run.initialVi.size(); ++i)
            {
                std::fprintf(output, "%s\"0x%04" PRIx32 "\"",
                             i == 0u ? "" : ",",
                             static_cast<uint32_t>(run.initialVi[i]) & 0xFFFFu);
            }
            std::fputs("],\"final_vi\":[", output);
            for (size_t i = 0u; i < run.finalVi.size(); ++i)
            {
                std::fprintf(output, "%s\"0x%04" PRIx32 "\"",
                             i == 0u ? "" : ",",
                             static_cast<uint32_t>(run.finalVi[i]) & 0xFFFFu);
            }
            std::fputs("],\"top_data\":\"", output);
            for (uint8_t value : run.topData)
                std::fprintf(output, "%02" PRIx8, value);
            std::fputs("\",\"recent_vif\":[", output);
            for (size_t commandIndex = 0u;
                 commandIndex < run.recentVifCommands.size(); ++commandIndex)
            {
                const auto &command = run.recentVifCommands[commandIndex];
                std::fprintf(output,
                             "%s{\"command\":\"0x%08" PRIx32 "\","
                             "\"cycle\":\"0x%04" PRIx32 "\","
                             "\"mode\":%" PRIu32 ","
                             "\"tops\":\"0x%03" PRIx32 "\","
                             "\"following\":\"",
                             commandIndex == 0u ? "" : ",",
                             command.command, command.cycle,
                             command.mode, command.tops);
                for (size_t i = 0u; i < command.followingSize; ++i)
                    std::fprintf(output, "%02" PRIx8, command.following[i]);
                std::fputs("\"}", output);
            }
            std::fputs("],\"instructions\":[", output);
            for (size_t i = 0u; i < run.pcs.size(); ++i)
            {
                std::fprintf(output,
                             "%s{\"pc\":\"0x%04" PRIx32 "\","
                             "\"lower\":\"0x%08" PRIx32 "\","
                             "\"upper\":\"0x%08" PRIx32 "\"}",
                             i == 0u ? "" : ",",
                             run.pcs[i], run.lower[i], run.upper[i]);
            }
            std::fputs("]}", output);
        }
        std::fputs("],\"vu1_run_starts\":{", output);
        writeSparsePcCounts(vu1RunStarts);
        std::fputs("},\"vu1_ended_starts\":{", output);
        writeSparsePcCounts(vu1EndedStarts);
        std::fputs("},\"vu1_cycle_limit_starts\":{", output);
        writeSparsePcCounts(vu1CycleLimitStarts);
        std::fputs("},\"vu1_first_cycle_limit\":", output);
        if (!vu1FirstCycleLimitCaptured)
        {
            std::fputs("null", output);
        }
        else
        {
            std::fprintf(output,
                         "{\"start_pc\":\"0x%04" PRIx32 "\","
                         "\"ordinal\":%" PRIu64 ","
                         "\"top\":\"0x%03" PRIx32 "\","
                         "\"itop\":\"0x%03" PRIx32 "\","
                         "\"final_pc\":\"0x%04" PRIx32 "\","
                         "\"micro_hash_fnv1a64\":\"0x%016" PRIx64 "\","
                         "\"initial_vi\":[",
                         vu1FirstCycleLimitStartPc, vu1FirstCycleLimitOrdinal,
                         vu1FirstCycleLimitTop,
                         vu1FirstCycleLimitItop, vu1FirstCycleLimitFinalPc,
                         vu1FirstCycleLimitMicroHash);
            for (size_t i = 0u; i < vu1FirstCycleLimitInitialVi.size(); ++i)
            {
                std::fprintf(output, "%s\"0x%04" PRIx32 "\"",
                             i == 0u ? "" : ",",
                             static_cast<uint32_t>(vu1FirstCycleLimitInitialVi[i]) & 0xFFFFu);
            }
            std::fputs("],\"final_vi\":[", output);
            for (size_t i = 0u; i < vu1FirstCycleLimitVi.size(); ++i)
            {
                std::fprintf(output, "%s\"0x%04" PRIx32 "\"",
                             i == 0u ? "" : ",",
                             static_cast<uint32_t>(vu1FirstCycleLimitVi[i]) & 0xFFFFu);
            }
            std::fputs("],\"top_data\":\"", output);
            for (uint8_t value : vu1FirstCycleLimitTopData)
                std::fprintf(output, "%02" PRIx8, value);
            std::fputs("\",\"tail\":[", output);
            for (size_t i = 0u; i < vu1FirstCycleLimitPcs.size(); ++i)
            {
                std::fprintf(output,
                             "%s{\"pc\":\"0x%04" PRIx32 "\","
                             "\"lower\":\"0x%08" PRIx32 "\","
                             "\"upper\":\"0x%08" PRIx32 "\"}",
                             i == 0u ? "" : ",",
                             vu1FirstCycleLimitPcs[i],
                             vu1FirstCycleLimitLower[i],
                             vu1FirstCycleLimitUpper[i]);
            }
            std::fputs("]}", output);
        }
        std::fputs(",\"vu1_lower_ops\":{", output);
        writeSparseCounts(vu1LowerOps);
        std::fputs("},\"vu1_lower_special_ops\":{", output);
        writeSparseCounts(vu1LowerSpecialOps);
        std::fputs("},\"vu1_upper_ops\":{", output);
        writeSparseCounts(vu1UpperOps);
        std::fputs("},\"vu1_upper_special_ops\":{", output);
        writeSparseCounts(vu1UpperSpecialOps);
        std::fputs("},\"watches\":{", output);
        for (size_t i = 0; i < watchAddresses.size(); ++i)
        {
            std::fprintf(output, "%s\"0x%08" PRIx32 "\":\"0x%08" PRIx32 "\"",
                         i == 0u ? "" : ",",
                         watchAddresses[i],
                         i < watchValues.size() ? watchValues[i] : 0u);
        }
        std::fputs("}}\n", output);
        std::fflush(output);
        active = false;
    }
};

void ps2RecordGsDmaWriteTrace(uint32_t guestAddr,
                              uint32_t size,
                              uint32_t writerPc,
                              uint32_t returnAddress)
{
    if (g_ps2GsDmaTraceState)
    {
        g_ps2GsDmaTraceState->recordEeWrite(
            guestAddr, size, writerPc, returnAddress);
    }
}

// Helpers for GS VRAM addressing (PSMCT32 path).
static inline uint32_t gs_vram_offset(uint32_t basePage, uint32_t x, uint32_t y, uint32_t fbw)
{
    // basePage is in 2048-byte units; fbw is in blocks of 64 pixels.
    uint32_t strideBytes = fbw * 64 * 4;
    return basePage * 2048 + y * strideBytes + x * 4;
}

PS2Memory::PS2Memory()
    : m_rdram(nullptr), m_scratchpad(nullptr), iop_ram(nullptr), m_seenGifCopy(false), m_gsVRAM(nullptr)
{
    ps2SetScratchpadHostPtr(nullptr);
}

PS2Memory::~PS2Memory()
{
    flushCodeInvalidationLog();
    finishVif1DmaTrace();
    delete m_gsDmaTrace;
    m_gsDmaTrace = nullptr;

    if (m_rdram)
    {
        delete[] m_rdram;
        m_rdram = nullptr;
    }

    if (m_scratchpad)
    {
        ps2SetScratchpadHostPtr(nullptr);
        delete[] m_scratchpad;
        m_scratchpad = nullptr;
    }

    if (m_gsVRAM)
    {
        delete[] m_gsVRAM;
        m_gsVRAM = nullptr;
    }

    if (m_vu1Code)
    {
        delete[] m_vu1Code;
        m_vu1Code = nullptr;
    }
    if (m_vu1Data)
    {
        delete[] m_vu1Data;
        m_vu1Data = nullptr;
    }
    if (m_vu0Code)
    {
        delete[] m_vu0Code;
        m_vu0Code = nullptr;
    }
    if (m_vu0Data)
    {
        delete[] m_vu0Data;
        m_vu0Data = nullptr;
    }

    if (iop_ram)
    {
        delete[] iop_ram;
        iop_ram = nullptr;
    }
}

void PS2Memory::beginVif1DmaTrace(uint32_t chcr,
                                  uint32_t madr,
                                  uint32_t qwc,
                                  uint32_t tadr,
                                  uint32_t asr0,
                                  uint32_t asr1,
                                  uint32_t sadr)
{
    if (!m_gsDmaTrace)
        m_gsDmaTrace = new Ps2GsDmaTraceState();

    Ps2GsDmaTraceState &trace = *m_gsDmaTrace;
    trace.configure();
    trace.finish();
    trace.flushEeWrites(trace.sequence);
    if (!trace.enabled || trace.sequence >= trace.limit)
    {
        if (g_ps2GsDmaTraceState == &trace)
            g_ps2GsDmaWriteTraceEnabled.store(false, std::memory_order_relaxed);
        return;
    }

    trace.active = true;
    trace.activeSequence = trace.sequence++;
    trace.chcr = chcr;
    trace.madr = madr;
    trace.qwc = qwc;
    trace.tadr = tadr;
    trace.asr0 = asr0;
    trace.asr1 = asr1;
    trace.sadr = sadr;
    trace.tagCount = 0u;
    trace.path1Calls = 0u;
    trace.vu1Runs = 0u;
    trace.vu1Resumes = 0u;
    trace.vu1Pairs = 0u;
    trace.vu1Xgkicks = 0u;
    trace.vu1EndedRuns = 0u;
    trace.vu1CycleLimitRuns = 0u;
    trace.vu1LastPc = 0u;
    trace.vu1CurrentStartPc = 0u;
    trace.vu1CurrentTop = 0u;
    trace.vu1CurrentItop = 0u;
    trace.vu1CurrentMicroHash = Ps2GsDmaTraceState::FNV_OFFSET_BASIS;
    trace.vu1CurrentPath1Calls = 0u;
    trace.vu1CurrentPath1 = {};
    trace.vu1CurrentInitialVi.fill(0);
    trace.vu1CurrentTopData.fill(0u);
    trace.vu1RunStarts.fill(0u);
    trace.vu1EndedStarts.fill(0u);
    trace.vu1CycleLimitStarts.fill(0u);
    trace.vu1LowerOps.fill(0u);
    trace.vu1LowerSpecialOps.fill(0u);
    trace.vu1UpperOps.fill(0u);
    trace.vu1UpperSpecialOps.fill(0u);
    trace.vu1StartPcs.clear();
    trace.vu1KickAddresses.clear();
    trace.vu1RunPrefix.clear();
    trace.recentVifCommands.clear();
    trace.vu1CurrentInstructionCount = 0u;
    trace.vu1FirstCycleLimitCaptured = false;
    trace.vu1FirstCycleLimitStartPc = 0u;
    trace.vu1FirstCycleLimitTop = 0u;
    trace.vu1FirstCycleLimitItop = 0u;
    trace.vu1FirstCycleLimitFinalPc = 0u;
    trace.vu1FirstCycleLimitOrdinal = 0u;
    trace.vu1FirstCycleLimitMicroHash = Ps2GsDmaTraceState::FNV_OFFSET_BASIS;
    trace.vu1FirstCycleLimitInitialVi.fill(0);
    trace.vu1FirstCycleLimitVi.fill(0);
    trace.vu1FirstCycleLimitTopData.fill(0u);
    trace.vu1FirstCycleLimitPcs.clear();
    trace.vu1FirstCycleLimitLower.clear();
    trace.vu1FirstCycleLimitUpper.clear();
    trace.packet = {};
    trace.vifInput = {};
    trace.path1 = {};
    trace.captureWatches(m_rdram, m_scratchpad);
}

void PS2Memory::traceVif1DmaTag(const uint8_t *data)
{
    if (!m_gsDmaTrace || !m_gsDmaTrace->active)
        return;
    m_gsDmaTrace->packet.append(data, 16u);
    ++m_gsDmaTrace->tagCount;
}

void PS2Memory::traceVif1DmaPayload(const uint8_t *data, uint32_t sizeBytes)
{
    if (!m_gsDmaTrace || !m_gsDmaTrace->active)
        return;
    m_gsDmaTrace->packet.append(data, sizeBytes);
}

void PS2Memory::traceVif1Input(const uint8_t *data, uint32_t sizeBytes)
{
    if (!m_gsDmaTrace || !m_gsDmaTrace->active)
        return;
    m_gsDmaTrace->vifInput.append(data, sizeBytes);
    if (!m_gsDmaTrace->vif1InputDumpDirectory.empty())
    {
        char filename[96];
        std::snprintf(filename, sizeof(filename),
                      "/sequence-%04" PRIu64 ".bin",
                      m_gsDmaTrace->activeSequence);
        const std::string path =
            m_gsDmaTrace->vif1InputDumpDirectory + filename;
        if (std::FILE *dump = std::fopen(path.c_str(), "ab"))
        {
            std::fwrite(data, 1u, sizeBytes, dump);
            std::fclose(dump);
        }
    }
}

void PS2Memory::traceVif1Command(uint32_t command, const uint8_t *followingData,
                                 uint32_t followingBytes)
{
    if (!isVif1DmaTraceActive())
        return;

    Ps2GsDmaTraceState::Vu1RunPrefix::VifCommand record{};
    record.command = command;
    record.cycle = vif1_regs.cycle;
    record.mode = vif1_regs.mode;
    record.tops = vif1_regs.tops;
    record.followingSize =
        std::min<size_t>(record.following.size(), followingBytes);
    if (followingData && record.followingSize != 0u)
    {
        std::copy_n(followingData, record.followingSize,
                    record.following.begin());
    }
    if (m_gsDmaTrace->recentVifCommands.size() == 16u)
        m_gsDmaTrace->recentVifCommands.erase(
            m_gsDmaTrace->recentVifCommands.begin());
    m_gsDmaTrace->recentVifCommands.push_back(record);
}

void PS2Memory::traceVif1NormalTransfer(uint32_t srcAddr, uint32_t qwc)
{
    if (!m_gsDmaTrace || !m_gsDmaTrace->active || qwc == 0u)
        return;

    const bool scratch = isScratchpad(srcAddr);
    uint32_t src = 0u;
    try
    {
        src = translateAddress(srcAddr);
    }
    catch (const std::exception &)
    {
        m_gsDmaTrace->packet.truncated = true;
        m_gsDmaTrace->vifInput.truncated = true;
        return;
    }

    const uint8_t *base = scratch ? m_scratchpad : m_rdram;
    const uint32_t limit = scratch ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
    const uint64_t requestedBytes = static_cast<uint64_t>(qwc) * 16ull;
    uint64_t bytes = std::min(requestedBytes, Ps2GsDmaTraceState::MAX_BYTES);
    if (bytes != requestedBytes)
    {
        m_gsDmaTrace->packet.truncated = true;
        m_gsDmaTrace->vifInput.truncated = true;
    }

    while (bytes > 0u)
    {
        if (src >= limit)
            src = 0u;
        const uint32_t chunk = static_cast<uint32_t>(
            std::min<uint64_t>(bytes, static_cast<uint64_t>(limit - src)));
        if (chunk == 0u)
            break;
        m_gsDmaTrace->packet.append(base + src, chunk);
        m_gsDmaTrace->vifInput.append(base + src, chunk);
        src += chunk;
        bytes -= chunk;
    }
}

void PS2Memory::tracePath1Packet(const uint8_t *data, uint32_t sizeBytes)
{
    if (!m_gsDmaTrace || !m_gsDmaTrace->active)
        return;
    ++m_gsDmaTrace->path1Calls;
    m_gsDmaTrace->path1.append(data, sizeBytes);
    if (m_gsDmaTrace->vu1Runs != 0u)
    {
        ++m_gsDmaTrace->vu1CurrentPath1Calls;
        m_gsDmaTrace->vu1CurrentPath1.append(data, sizeBytes);
    }
    if (!m_gsDmaTrace->path1DumpDirectory.empty() && data && sizeBytes != 0u)
    {
        char filename[64];
        std::snprintf(filename, sizeof(filename),
                      "/sequence-%04" PRIu64 ".bin",
                      m_gsDmaTrace->activeSequence);
        const std::string path = m_gsDmaTrace->path1DumpDirectory + filename;
        if (std::FILE *dump = std::fopen(
                path.c_str(), m_gsDmaTrace->path1Calls == 1u ? "wb" : "ab"))
        {
            std::fwrite(data, 1u, sizeBytes, dump);
            std::fclose(dump);
        }
    }
}

bool PS2Memory::isVif1DmaTraceActive() const
{
    return m_gsDmaTrace && m_gsDmaTrace->active;
}

void PS2Memory::traceVu1Invocation(uint32_t startPc, uint32_t top, uint32_t itop, bool resume,
                                   const VU1State &state)
{
    if (!isVif1DmaTraceActive())
        return;

    const uint64_t ordinal = m_gsDmaTrace->vu1Runs;
    ++m_gsDmaTrace->vu1Runs;
    if (resume)
        ++m_gsDmaTrace->vu1Resumes;
    m_gsDmaTrace->vu1CurrentStartPc = startPc & 0x3FFFu;
    m_gsDmaTrace->vu1CurrentTop = top & 0x3FFu;
    m_gsDmaTrace->vu1CurrentItop = itop & 0x3FFu;
    m_gsDmaTrace->vu1CurrentInstructionCount = 0u;
    m_gsDmaTrace->vu1CurrentPath1Calls = 0u;
    m_gsDmaTrace->vu1CurrentPath1 = {};
    m_gsDmaTrace->vu1CurrentInitialVi.fill(0);
    std::copy_n(state.vi, m_gsDmaTrace->vu1CurrentInitialVi.size(),
                m_gsDmaTrace->vu1CurrentInitialVi.begin());
    m_gsDmaTrace->vu1CurrentTopData.fill(0u);
    if (m_vu1Data)
    {
        const uint32_t start = (top & 0x3FFu) * 16u;
        for (size_t i = 0u; i < m_gsDmaTrace->vu1CurrentTopData.size(); ++i)
        {
            m_gsDmaTrace->vu1CurrentTopData[i] =
                m_vu1Data[(start + static_cast<uint32_t>(i)) % PS2_VU1_DATA_SIZE];
        }
    }
    m_gsDmaTrace->vu1CurrentMicroHash = Ps2GsDmaTraceState::FNV_OFFSET_BASIS;
    if (m_vu1Code)
    {
        for (size_t i = 0u; i < PS2_VU1_CODE_SIZE; ++i)
        {
            m_gsDmaTrace->vu1CurrentMicroHash ^= m_vu1Code[i];
            m_gsDmaTrace->vu1CurrentMicroHash *= Ps2GsDmaTraceState::FNV_PRIME;
        }
    }

    if (!resume &&
        m_gsDmaTrace->activeSequence == m_gsDmaTrace->vu1StepSequence &&
        ordinal == m_gsDmaTrace->vu1StepOrdinal &&
        !m_gsDmaTrace->vu1DumpDirectory.empty() &&
        m_vu1Data && m_vu1Code)
    {
        char filename[96];
        auto writeFile = [&](const char *suffix, const void *data, size_t size)
        {
            std::snprintf(filename, sizeof(filename),
                          "/sequence-%04" PRIu64 "-ordinal-%04" PRIu64 "-%s.bin",
                          m_gsDmaTrace->activeSequence, ordinal, suffix);
            const std::string path = m_gsDmaTrace->vu1DumpDirectory + filename;
            if (std::FILE *dump = std::fopen(path.c_str(), "wb"))
            {
                std::fwrite(data, 1u, size, dump);
                std::fclose(dump);
            }
            else
            {
                std::fprintf(stderr, "PS2 VU1 trace: cannot open '%s': %s\n",
                             path.c_str(), std::strerror(errno));
            }
        };

        writeFile("data", m_vu1Data, PS2_VU1_DATA_SIZE);
        writeFile("micro", m_vu1Code, PS2_VU1_CODE_SIZE);

        std::array<uint32_t, 159u> words{};
        size_t cursor = 0u;
        words[cursor++] = 0x31555652u;
        words[cursor++] = 1u;
        words[cursor++] = startPc & 0x3FFFu;
        words[cursor++] = top & 0x3FFu;
        words[cursor++] = itop & 0x3FFu;
        std::memcpy(words.data() + cursor, state.vf, sizeof(state.vf));
        cursor += 32u * 4u;
        for (int32_t value : state.vi)
            words[cursor++] = static_cast<uint32_t>(value);
        std::memcpy(words.data() + cursor, state.acc, sizeof(state.acc));
        cursor += 4u;
        std::memcpy(&words[cursor++], &state.q, sizeof(state.q));
        std::memcpy(&words[cursor++], &state.p, sizeof(state.p));
        std::memcpy(&words[cursor++], &state.i, sizeof(state.i));
        words[cursor++] = state.status;
        words[cursor++] = state.mac;
        words[cursor++] = state.clip;
        writeFile("state", words.data(), words.size() * sizeof(words[0]));
    }
    ++m_gsDmaTrace->vu1RunStarts[(startPc & 0x3FFFu) / 8u];
    if (m_gsDmaTrace->vu1StartPcs.size() < 64u)
        m_gsDmaTrace->vu1StartPcs.push_back(startPc & 0x3FFFu);
    const bool selectedRun =
        m_gsDmaTrace->activeSequence == m_gsDmaTrace->vu1StepSequence &&
        ordinal == m_gsDmaTrace->vu1StepOrdinal;
    if (m_gsDmaTrace->vu1RunPrefix.size() < 8u ||
        (selectedRun && ordinal >= 8u))
    {
        Ps2GsDmaTraceState::Vu1RunPrefix run{};
        run.ordinal = m_gsDmaTrace->vu1Runs - 1u;
        run.startPc = m_gsDmaTrace->vu1CurrentStartPc;
        run.top = m_gsDmaTrace->vu1CurrentTop;
        run.itop = m_gsDmaTrace->vu1CurrentItop;
        run.microHash = m_gsDmaTrace->vu1CurrentMicroHash;
        run.initialVi = m_gsDmaTrace->vu1CurrentInitialVi;
        run.topData = m_gsDmaTrace->vu1CurrentTopData;
        run.recentVifCommands = m_gsDmaTrace->recentVifCommands;
        m_gsDmaTrace->vu1RunPrefix.push_back(run);
    }

    std::fprintf(m_gsDmaTrace->output,
                 "{\"schema_version\":1,\"event\":\"vu1-start\","
                 "\"sequence\":%" PRIu64 ",\"ordinal\":%" PRIu64 ","
                 "\"start_pc\":\"0x%04" PRIx32 "\","
                 "\"top\":\"0x%03" PRIx32 "\",\"itop\":\"0x%03" PRIx32 "\","
                 "\"micro_hash_fnv1a64\":\"0x%016" PRIx64 "\","
                 "\"vi\":[",
                 m_gsDmaTrace->activeSequence, m_gsDmaTrace->vu1Runs - 1u,
                 m_gsDmaTrace->vu1CurrentStartPc,
                 m_gsDmaTrace->vu1CurrentTop,
                 m_gsDmaTrace->vu1CurrentItop,
                 m_gsDmaTrace->vu1CurrentMicroHash);
    for (size_t index = 0u;
         index < m_gsDmaTrace->vu1CurrentInitialVi.size(); ++index)
    {
        std::fprintf(m_gsDmaTrace->output, "%s\"0x%04" PRIx32 "\"",
                     index == 0u ? "" : ",",
                     static_cast<uint32_t>(
                         m_gsDmaTrace->vu1CurrentInitialVi[index]) &
                         0xFFFFu);
    }
    std::fputs("],\"top_data\":\"", m_gsDmaTrace->output);
    for (uint8_t value : m_gsDmaTrace->vu1CurrentTopData)
        std::fprintf(m_gsDmaTrace->output, "%02" PRIx8, value);
    std::fputs("\"}\n", m_gsDmaTrace->output);
}

void PS2Memory::traceVu1Instruction(uint32_t pc, uint32_t lower, uint32_t upper,
                                    const VU1State &state)
{
    if (!isVif1DmaTraceActive())
        return;

    Ps2GsDmaTraceState &trace = *m_gsDmaTrace;
    if (trace.activeSequence == trace.vu1StepSequence &&
        trace.vu1Runs != 0u &&
        trace.vu1Runs - 1u == trace.vu1StepOrdinal &&
        trace.vu1StepEvents < 4096u)
    {
        std::fprintf(trace.output,
                     "{\"schema_version\":1,\"event\":\"vu1-step\","
                     "\"sequence\":%" PRIu64 ",\"ordinal\":%" PRIu64 ","
                     "\"index\":%" PRIu64 ",\"pc\":\"0x%04" PRIx32 "\","
                     "\"lower\":\"0x%08" PRIx32 "\","
                     "\"upper\":\"0x%08" PRIx32 "\",\"vi\":[",
                     trace.activeSequence, trace.vu1Runs - 1u,
                     trace.vu1StepEvents, pc & 0x3FFFu, lower, upper);
        for (size_t index = 0u; index < std::size(state.vi); ++index)
        {
            std::fprintf(trace.output, "%s\"0x%04" PRIx32 "\"",
                         index == 0u ? "" : ",",
                         static_cast<uint32_t>(state.vi[index]) & 0xFFFFu);
        }
        std::fputs("],\"vf\":[", trace.output);
        const float *vf = &state.vf[0][0];
        for (size_t index = 0u; index < 32u * 4u; ++index)
        {
            uint32_t bits = 0u;
            std::memcpy(&bits, &vf[index], sizeof(bits));
            std::fprintf(trace.output, "%s\"0x%08" PRIx32 "\"",
                         index == 0u ? "" : ",", bits);
        }
        std::fputs("],\"acc\":[", trace.output);
        for (size_t index = 0u; index < std::size(state.acc); ++index)
        {
            uint32_t bits = 0u;
            std::memcpy(&bits, &state.acc[index], sizeof(bits));
            std::fprintf(trace.output, "%s\"0x%08" PRIx32 "\"",
                         index == 0u ? "" : ",", bits);
        }
        uint32_t q = 0u;
        uint32_t p = 0u;
        uint32_t i = 0u;
        std::memcpy(&q, &state.q, sizeof(q));
        std::memcpy(&p, &state.p, sizeof(p));
        std::memcpy(&i, &state.i, sizeof(i));
        std::fprintf(trace.output,
                     "],\"q\":\"0x%08" PRIx32 "\","
                     "\"p\":\"0x%08" PRIx32 "\","
                     "\"i\":\"0x%08" PRIx32 "\","
                     "\"status\":\"0x%08" PRIx32 "\","
                     "\"mac\":\"0x%08" PRIx32 "\","
                     "\"clip\":\"0x%08" PRIx32 "\"}\n",
                     q, p, i, state.status, state.mac, state.clip);
        std::fflush(trace.output);
        ++trace.vu1StepEvents;
    }
    ++trace.vu1Pairs;
    trace.vu1LastPc = pc & 0x3FFFu;
    const size_t instructionIndex =
        trace.vu1CurrentInstructionCount % trace.vu1CurrentPcs.size();
    trace.vu1CurrentPcs[instructionIndex] = pc & 0x3FFFu;
    trace.vu1CurrentLower[instructionIndex] = lower;
    trace.vu1CurrentUpper[instructionIndex] = upper;
    ++trace.vu1CurrentInstructionCount;
    if (!trace.vu1RunPrefix.empty() &&
        trace.vu1RunPrefix.back().ordinal == trace.vu1Runs - 1u &&
        trace.vu1RunPrefix.back().pcs.size() < 256u)
    {
        trace.vu1RunPrefix.back().pcs.push_back(pc & 0x3FFFu);
        trace.vu1RunPrefix.back().lower.push_back(lower);
        trace.vu1RunPrefix.back().upper.push_back(upper);
    }

    const uint8_t upperOp = static_cast<uint8_t>(upper & 0x3Fu);
    ++trace.vu1UpperOps[upperOp];
    if (upperOp >= 0x3Cu)
    {
        const uint8_t upperSpecial =
            static_cast<uint8_t>((upper & 0x3u) | ((upper >> 4u) & 0x7Cu));
        ++trace.vu1UpperSpecialOps[upperSpecial];
    }

    if ((upper & 0x80000000u) != 0u)
        return;

    const uint8_t lowerOp = static_cast<uint8_t>((lower >> 25u) & 0x7Fu);
    ++trace.vu1LowerOps[lowerOp];
    if (lowerOp == 0x40u)
    {
        const uint8_t function = static_cast<uint8_t>(lower & 0x3Fu);
        if (function >= 0x3Cu)
        {
            const uint8_t lowerSpecial =
                static_cast<uint8_t>((lower & 0x3u) | ((lower >> 4u) & 0x7Cu));
            ++trace.vu1LowerSpecialOps[lowerSpecial];
        }
    }
}

void PS2Memory::traceVu1Xgkick(uint32_t sourceQword)
{
    if (!isVif1DmaTraceActive())
        return;

    ++m_gsDmaTrace->vu1Xgkicks;
    if (m_gsDmaTrace->vu1KickAddresses.size() < 64u)
        m_gsDmaTrace->vu1KickAddresses.push_back(sourceQword & 0x3FFu);
}

void PS2Memory::traceVu1InvocationEnd(uint32_t finalPc, bool ended, bool hitCycleLimit,
                                      const int32_t *viRegisters, size_t viRegisterCount)
{
    if (!isVif1DmaTraceActive())
        return;

    m_gsDmaTrace->vu1LastPc = finalPc & 0x3FFFu;
    if (!m_gsDmaTrace->vu1RunPrefix.empty() &&
        m_gsDmaTrace->vu1RunPrefix.back().ordinal == m_gsDmaTrace->vu1Runs - 1u)
    {
        Ps2GsDmaTraceState::Vu1RunPrefix &run =
            m_gsDmaTrace->vu1RunPrefix.back();
        run.finalPc = finalPc & 0x3FFFu;
        run.instructionCount = m_gsDmaTrace->vu1CurrentInstructionCount;
        run.ended = ended;
        run.hitCycleLimit = hitCycleLimit;
        if (viRegisters)
        {
            const size_t count =
                std::min(viRegisterCount, run.finalVi.size());
            std::copy_n(viRegisters, count, run.finalVi.begin());
        }
    }
    if (ended)
    {
        ++m_gsDmaTrace->vu1EndedRuns;
        ++m_gsDmaTrace->vu1EndedStarts[m_gsDmaTrace->vu1CurrentStartPc / 8u];
    }
    if (hitCycleLimit)
    {
        ++m_gsDmaTrace->vu1CycleLimitRuns;
        ++m_gsDmaTrace->vu1CycleLimitStarts[m_gsDmaTrace->vu1CurrentStartPc / 8u];

        if (!m_gsDmaTrace->vu1FirstCycleLimitCaptured)
        {
            Ps2GsDmaTraceState &trace = *m_gsDmaTrace;
            trace.vu1FirstCycleLimitCaptured = true;
            trace.vu1FirstCycleLimitStartPc = trace.vu1CurrentStartPc;
            trace.vu1FirstCycleLimitOrdinal = trace.vu1Runs - 1u;
            trace.vu1FirstCycleLimitTop = trace.vu1CurrentTop;
            trace.vu1FirstCycleLimitItop = trace.vu1CurrentItop;
            trace.vu1FirstCycleLimitFinalPc = finalPc & 0x3FFFu;
            trace.vu1FirstCycleLimitMicroHash = trace.vu1CurrentMicroHash;
            trace.vu1FirstCycleLimitInitialVi = trace.vu1CurrentInitialVi;
            trace.vu1FirstCycleLimitTopData = trace.vu1CurrentTopData;
            if (viRegisters)
            {
                const size_t count =
                    std::min(viRegisterCount, trace.vu1FirstCycleLimitVi.size());
                std::copy_n(viRegisters, count, trace.vu1FirstCycleLimitVi.begin());
            }

            const size_t tailCount = std::min(
                trace.vu1CurrentInstructionCount, trace.vu1CurrentPcs.size());
            const size_t firstIndex =
                (trace.vu1CurrentInstructionCount - tailCount) %
                trace.vu1CurrentPcs.size();
            for (size_t i = 0u; i < tailCount; ++i)
            {
                const size_t index = (firstIndex + i) % trace.vu1CurrentPcs.size();
                trace.vu1FirstCycleLimitPcs.push_back(trace.vu1CurrentPcs[index]);
                trace.vu1FirstCycleLimitLower.push_back(trace.vu1CurrentLower[index]);
                trace.vu1FirstCycleLimitUpper.push_back(trace.vu1CurrentUpper[index]);
            }
        }
    }

    Ps2GsDmaTraceState &trace = *m_gsDmaTrace;
    std::fprintf(trace.output,
                 "{\"schema_version\":1,\"event\":\"vu1-end\","
                 "\"sequence\":%" PRIu64 ",\"ordinal\":%" PRIu64 ","
                 "\"final_pc\":\"0x%04" PRIx32 "\","
                 "\"pairs\":%" PRIu64 ",\"ended\":%s,"
                 "\"cycle_limit\":%s,\"vi\":[",
                 trace.activeSequence, trace.vu1Runs - 1u,
                 finalPc & 0x3FFFu, trace.vu1CurrentInstructionCount,
                 ended ? "true" : "false",
                 hitCycleLimit ? "true" : "false");
    for (size_t index = 0u; index < 16u; ++index)
    {
        const uint32_t value =
            viRegisters && index < viRegisterCount
                ? static_cast<uint32_t>(viRegisters[index]) & 0xFFFFu
                : 0u;
        std::fprintf(trace.output, "%s\"0x%04" PRIx32 "\"",
                     index == 0u ? "" : ",", value);
    }
    std::fprintf(trace.output,
                 "],\"path1_calls\":%" PRIu64 ","
                 "\"path1_bytes\":%" PRIu64 ","
                 "\"path1_hash_fnv1a64\":\"0x%016" PRIx64 "\"}\n",
                 trace.vu1CurrentPath1Calls,
                 trace.vu1CurrentPath1.bytes,
                 trace.vu1CurrentPath1.hash);
}

void PS2Memory::finishVif1DmaTrace()
{
    if (m_gsDmaTrace)
        m_gsDmaTrace->finish();
}

bool PS2Memory::initialize(size_t ramSize)
{
    auto cleanup = [this]()
    {
        delete[] m_rdram;
        delete[] m_scratchpad;
        delete[] iop_ram;
        delete[] m_gsVRAM;
        delete[] m_vu0Code;
        delete[] m_vu0Data;
        delete[] m_vu1Code;
        delete[] m_vu1Data;
        m_rdram = nullptr;
        m_scratchpad = nullptr;
        ps2SetScratchpadHostPtr(nullptr);
        iop_ram = nullptr;
        m_gsVRAM = nullptr;
        m_vu0Code = nullptr;
        m_vu0Data = nullptr;
        m_vu1Code = nullptr;
        m_vu1Data = nullptr;
    };

    cleanup();
    m_seenGifCopy = false;
    m_dmaStartCount.store(0, std::memory_order_relaxed);
    m_gifCopyCount.store(0, std::memory_order_relaxed);
    m_gsWriteCount.store(0, std::memory_order_relaxed);
    m_vifWriteCount.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(m_dmacMutex);
        m_dmacChannels = {};
        m_readyDmacCompletions.clear();
        m_pendingDmacInterrupts.clear();
        m_dmacReadySequence = 0u;
        m_dmacPublicationSequence = 0u;
    }
    m_pendingGifTransfers.clear();
    m_pendingVif0Transfers.clear();
    m_pendingVif1Transfers.clear();
    m_codeRegions.clear();
    m_path3Masked = false;
    m_path3MaskedFifo.clear();
    m_vif1PendingPath2DirectQwc = 0u;
    m_vif1PendingPath2DirectHl = false;
    m_vif1WaitingForVu = false;
    m_vif1DeferredData.clear();
    m_vif1DeferredDmacCompletion.reset();
    m_timer0LastHostNs = 0;
    m_timer0FractionNs = 0;

    try
    {
        // Allocate main RAM
        m_rdram = new uint8_t[ramSize];
        std::memset(m_rdram, 0, ramSize);

        // Allocate scratchpad
        m_scratchpad = new uint8_t[PS2_SCRATCHPAD_SIZE];
        std::memset(m_scratchpad, 0, PS2_SCRATCHPAD_SIZE);
        ps2SetScratchpadHostPtr(m_scratchpad);

        // Initialize EE TLB entries (R5900 has 48 entries).
        m_tlbEntries.assign(48, TLBEntry{0, 0, 0, false});

        // Allocate IOP RAM
        iop_ram = new uint8_t[PS2_IOP_RAM_SIZE];

        // Initialize IOP RAM with zeros
        std::memset(iop_ram, 0, PS2_IOP_RAM_SIZE);

        // Initialize I/O registers
        m_ioRegisters.clear();

        // Initialize GS registers
        memset(&gs_regs, 0, sizeof(gs_regs));
        // memset zero-fills std::atomic<uint64_t>::csr's bytes, which is not itself
        // a guaranteed-valid atomic store; make the zero-initialization explicit.
        gs_regs.csr.store(0);
        gs_regs.dispfb1 = (0ULL << 0) | (10ULL << 9) | (0ULL << 15) | (0ULL << 32) | (0ULL << 43);
        gs_regs.display1 = (0ULL << 0) | (0ULL << 12) | (0ULL << 23) | (0ULL << 27) | (639ULL << 32) | (447ULL << 44);
        gs_regs.dispfb2 = gs_regs.dispfb1;
        gs_regs.display2 = gs_regs.display1;

        // Allocate GS VRAM (4MB)
        m_gsVRAM = new uint8_t[PS2_GS_VRAM_SIZE];
        std::memset(m_gsVRAM, 0, PS2_GS_VRAM_SIZE);

        m_vu0Code = new uint8_t[PS2_VU0_CODE_SIZE];
        m_vu0Data = new uint8_t[PS2_VU0_DATA_SIZE];
        std::memset(m_vu0Code, 0, PS2_VU0_CODE_SIZE);
        std::memset(m_vu0Data, 0, PS2_VU0_DATA_SIZE);
        markVU0CodeModified();

        m_vu1Code = new uint8_t[PS2_VU1_CODE_SIZE];
        m_vu1Data = new uint8_t[PS2_VU1_DATA_SIZE];
        std::memset(m_vu1Code, 0, PS2_VU1_CODE_SIZE);
        std::memset(m_vu1Data, 0, PS2_VU1_DATA_SIZE);
        markVU1CodeModified();

        // Initialize VIF registers
        memset(&vif0_regs, 0, sizeof(vif0_regs));
        memset(&vif1_regs, 0, sizeof(vif1_regs));

        // Initialize DMA registers
        memset(dma_regs, 0, sizeof(dma_regs));

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error initializing PS2 memory: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

void PS2Memory::updateEeTimer0Counter()
{
    const uint64_t nowNs = steadyClockNs();
    if (m_timer0LastHostNs == 0u)
    {
        m_timer0LastHostNs = nowNs;
        return;
    }

    const uint32_t mode = m_ioRegisters.count(kEeTimer0Mode) ? m_ioRegisters[kEeTimer0Mode] : 0u;
    if ((mode & kEeTimerModeCue) == 0u)
    {
        m_timer0LastHostNs = nowNs;
        m_timer0FractionNs = 0u;
        return;
    }

    const uint64_t elapsedNs = nowNs - m_timer0LastHostNs;
    m_timer0LastHostNs = nowNs;
    if (elapsedNs == 0u)
    {
        return;
    }

    const uint64_t scaled = elapsedNs * kEeTimer0TicksPerSecond + m_timer0FractionNs;
    const uint64_t ticks = scaled / kNanosecondsPerSecond;
    m_timer0FractionNs = scaled % kNanosecondsPerSecond;
    if (ticks != 0u)
    {
        m_ioRegisters[kEeTimer0Count] = m_ioRegisters[kEeTimer0Count] + static_cast<uint32_t>(ticks);
    }
}

bool PS2Memory::isScratchpad(uint32_t address) const
{
    return ps2IsScratchpadAddress(address);
}

uint8_t *PS2Memory::mapVuMemory(uint32_t physAddr, uint32_t size, uint32_t &offset, uint32_t &limit)
{
    return const_cast<uint8_t *>(static_cast<const PS2Memory *>(this)->mapVuMemory(physAddr, size, offset, limit));
}

const uint8_t *PS2Memory::mapVuMemory(uint32_t physAddr, uint32_t size, uint32_t &offset, uint32_t &limit) const
{
    auto mapRange = [&](uint32_t base, uint32_t rangeSize, const uint8_t *ptr) -> const uint8_t *
    {
        if (!ptr || physAddr < base)
        {
            return nullptr;
        }
        const uint32_t local = physAddr - base;
        if (local >= rangeSize || size > (rangeSize - local))
        {
            return nullptr;
        }
        offset = local;
        limit = rangeSize;
        return ptr;
    };

    if (const uint8_t *ptr = mapRange(PS2_VU0_CODE_BASE, PS2_VU0_CODE_SIZE, m_vu0Code))
    {
        return ptr;
    }
    if (const uint8_t *ptr = mapRange(PS2_VU0_DATA_BASE, PS2_VU0_DATA_SIZE, m_vu0Data))
    {
        return ptr;
    }
    if (const uint8_t *ptr = mapRange(PS2_VU1_CODE_BASE, PS2_VU1_CODE_SIZE, m_vu1Code))
    {
        return ptr;
    }
    return mapRange(PS2_VU1_DATA_BASE, PS2_VU1_DATA_SIZE, m_vu1Data);
}

uint32_t PS2Memory::translateAddress(uint32_t virtualAddress)
{
    if (isScratchpad(virtualAddress))
    {
        return ps2ScratchpadOffset(virtualAddress);
    }

    // EE TLB aliases of main RAM installed by the kernel.
    if (Ps2IsUncachedRamMirrorAddress(virtualAddress))
    {
        return virtualAddress & PS2_RAM_MASK;
    }
    if (Ps2IsAcceleratedRamMirrorAddress(virtualAddress))
    {
        return virtualAddress & PS2_RAM_MASK;
    }

    // KSEG0/KSEG1 direct-mapped window.
    if (Ps2IsKseg01Address(virtualAddress))
    {
        return Ps2DirectMappedPhysicalAddress(virtualAddress);
    }

    // In this runtime, low segments are treated as physical-style addresses already.
    if (virtualAddress < 0x80000000)
    {
        return virtualAddress;
    }

    // KSEG2/KSEG3 are TLB mapped.
    if (Ps2IsKseg23Address(virtualAddress))
    {
        for (const auto &entry : m_tlbEntries)
        {
            if (entry.valid)
            {
                // PageMask uses bits [24:13]. Build an address-level mask (plus 4KB base page bits).
                const uint32_t mask = entry.mask & 0x01FFE000u;
                const uint32_t compareMask = ~(mask | 0xFFFu);
                if ((virtualAddress & compareMask) == (entry.vpn & compareMask))
                {
                    // TLB hit
                    const uint32_t pageOffsetMask = mask | 0xFFFu;
                    const uint32_t physBase = entry.pfn << 12;
                    return physBase | (virtualAddress & pageOffsetMask);
                }
            }
        }
        throw PS2TlbMissException(virtualAddress);
    }

    return virtualAddress;
}

bool PS2Memory::tlbRead(uint32_t index, uint32_t &vpn, uint32_t &pfn, uint32_t &mask, bool &valid) const
{
    if (index >= m_tlbEntries.size())
    {
        return false;
    }

    const TLBEntry &entry = m_tlbEntries[index];
    vpn = entry.vpn;
    pfn = entry.pfn;
    mask = entry.mask;
    valid = entry.valid;
    return true;
}

bool PS2Memory::tlbWrite(uint32_t index, uint32_t vpn, uint32_t pfn, uint32_t mask, bool valid)
{
    if (index >= m_tlbEntries.size())
    {
        return false;
    }

    TLBEntry &entry = m_tlbEntries[index];
    entry.vpn = vpn & 0xFFFFF000u;
    entry.pfn = pfn & 0x000FFFFFu;
    entry.mask = mask & 0x01FFE000u;
    entry.valid = valid;
    return true;
}

int32_t PS2Memory::tlbProbe(uint32_t vpn) const
{
    const uint32_t normalizedVpn = vpn & 0xFFFFF000u;
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_tlbEntries.size()); ++i)
    {
        const TLBEntry &entry = m_tlbEntries[i];
        if (!entry.valid)
        {
            continue;
        }

        const uint32_t mask = entry.mask & 0x01FFE000u;
        const uint32_t compareMask = ~(mask | 0xFFFu);
        if ((normalizedVpn & compareMask) == (entry.vpn & compareMask))
        {
            return static_cast<int32_t>(i);
        }
    }

    return -1;
}

uint8_t PS2Memory::read8(uint32_t address)
{
    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return m_scratchpad[physAddr];
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return m_rdram[physAddr];
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint8_t), vuOffset, vuLimit))
    {
        (void)vuLimit;
        return vuMem[vuOffset];
    }
    else if (isIoRegister(physAddr))
    {
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t value = readIORegister(regAddr);
        uint32_t shift = (physAddr & 3) * 8;
        return static_cast<uint8_t>((value >> shift) & 0xFF);
    }

    return 0;
}

uint16_t PS2Memory::read16(uint32_t address)
{
    if (address & 1)
    {
        throw std::runtime_error("Unaligned 16-bit read at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return loadScalar<uint16_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, "read16 scratchpad", address);
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return loadScalar<uint16_t>(m_rdram, physAddr, PS2_RAM_SIZE, "read16 rdram", address);
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint16_t), vuOffset, vuLimit))
    {
        return loadScalar<uint16_t>(vuMem, vuOffset, vuLimit, "read16 vu", address);
    }
    else if (isIoRegister(physAddr))
    {
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t value = readIORegister(regAddr);
        uint32_t shift = (physAddr & 2) * 8;
        return static_cast<uint16_t>((value >> shift) & 0xFFFF);
    }

    return 0;
}

uint32_t PS2Memory::read32(uint32_t address)
{
    if (address & 3)
    {
        throw std::runtime_error("Unaligned 32-bit read at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        uint32_t off = address & 7;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            uint64_t val = gs_regs.csr.load();
            return (uint32_t)(val >> (off * 8));
        }
        uint64_t *reg = gsRegPtr(gs_regs, address);
        if (!reg)
            return 0;
        uint64_t val = *reg;
        return (uint32_t)(val >> (off * 8));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return loadScalar<uint32_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, "read32 scratchpad", address);
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return loadScalar<uint32_t>(m_rdram, physAddr, PS2_RAM_SIZE, "read32 rdram", address);
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint32_t), vuOffset, vuLimit))
    {
        return loadScalar<uint32_t>(vuMem, vuOffset, vuLimit, "read32 vu", address);
    }
    else if (isIoRegister(physAddr))
    {
        return readIORegister(physAddr);
    }

    return 0;
}

uint64_t PS2Memory::read64(uint32_t address)
{
    if (address & 7)
    {
        throw std::runtime_error("Unaligned 64-bit read at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            return gs_regs.csr.load();
        }
        uint64_t *reg = gsRegPtr(gs_regs, address);
        return reg ? *reg : 0;
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return loadScalar<uint64_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, "read64 scratchpad", address);
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return loadScalar<uint64_t>(m_rdram, physAddr, PS2_RAM_SIZE, "read64 rdram", address);
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint64_t), vuOffset, vuLimit))
    {
        return loadScalar<uint64_t>(vuMem, vuOffset, vuLimit, "read64 vu", address);
    }

    // 64-bit IO read: compose from the two adjacent 32-bit IO register slots
    // to avoid any side-effects from read32 handlers.
    if (isIoRegister(address))
    {
        uint32_t lo = m_ioRegisters.count(address) ? m_ioRegisters[address] : 0u;
        uint32_t hi = m_ioRegisters.count(address + 4) ? m_ioRegisters[address + 4] : 0u;
        return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    }
    return (uint64_t)read32(address) | ((uint64_t)read32(address + 4) << 32);
}

__m128i PS2Memory::read128(uint32_t address)
{
    if (address & 15)
    {
        throw std::runtime_error("Unaligned 128-bit read at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        inRange(physAddr, sizeof(__m128i), PS2_SCRATCHPAD_SIZE, "read128 scratchpad", address);
        return _mm_loadu_si128(reinterpret_cast<__m128i *>(&m_scratchpad[physAddr]));
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        inRange(physAddr, sizeof(__m128i), PS2_RAM_SIZE, "read128 rdram", address);
        return _mm_loadu_si128(reinterpret_cast<__m128i *>(&m_rdram[physAddr]));
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(__m128i), vuOffset, vuLimit))
    {
        inRange(vuOffset, sizeof(__m128i), vuLimit, "read128 vu", address);
        return _mm_loadu_si128(reinterpret_cast<const __m128i *>(vuMem + vuOffset));
    }

    // 128-bit reads are primarily for quad-word loads in the EE, which are only valid for RAM areas
    // Return zeroes for unsupported areas
    return _mm_setzero_si128();
}

void PS2Memory::write8(uint32_t address, uint8_t value, uint32_t writerPc)
{
    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        m_scratchpad[physAddr] = value;
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        markModified(physAddr, sizeof(value), writerPc);
        m_rdram[physAddr] = value;
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint8_t), vuOffset, vuLimit))
        {
            (void)vuLimit;
            vuMem[vuOffset] = value;
            markVUCodeModified(vuMem);
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        // IO registers - handle byte writes by modifying the appropriate byte in the word
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t shift = (physAddr & 3) * 8;
        uint32_t mask = ~(0xFF << shift);
        uint32_t newValue = (m_ioRegisters[regAddr] & mask) | ((uint32_t)value << shift);
        writeIORegister(regAddr, newValue);
    }
}

void PS2Memory::write16(uint32_t address, uint16_t value, uint32_t writerPc)
{
    if (address & 1)
    {
        throw std::runtime_error("Unaligned 16-bit write at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        storeScalar<uint16_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, value, "write16 scratchpad", address);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        markModified(physAddr, sizeof(value), writerPc);
        storeScalar<uint16_t>(m_rdram, physAddr, PS2_RAM_SIZE, value, "write16 rdram", address);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint16_t), vuOffset, vuLimit))
        {
            storeScalar<uint16_t>(vuMem, vuOffset, vuLimit, value, "write16 vu", address);
            markVUCodeModified(vuMem);
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t shift = (physAddr & 2) * 8;
        uint32_t mask = ~(0xFFFF << shift);
        uint32_t newValue = (m_ioRegisters[regAddr] & mask) | ((uint32_t)value << shift);
        writeIORegister(regAddr, newValue);
    }
}

void PS2Memory::write32(uint32_t address, uint32_t value, uint32_t writerPc)
{
    if (address & 3)
    {
        throw std::runtime_error("Unaligned 32-bit write at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        uint32_t off = address & 7;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            // CSR: bits 0..1 of the low dword are write-one-to-clear status bits.
            // Done as a single atomic RMW -- see writeCsrHalf's comment.
            writeCsrHalf(gs_regs.csr, off, value);
        }
        else if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            uint64_t mask = 0xFFFFFFFFULL << (off * 8);
            uint64_t newVal = (*reg & ~mask) | ((uint64_t)value << (off * 8));
            *reg = newVal;
        }
        return;
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        storeScalar<uint32_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, value, "write32 scratchpad", address);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        // Check if this might be code modification
        markModified(physAddr, sizeof(value), writerPc);

        storeScalar<uint32_t>(m_rdram, physAddr, PS2_RAM_SIZE, value, "write32 rdram", address);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint32_t), vuOffset, vuLimit))
        {
            storeScalar<uint32_t>(vuMem, vuOffset, vuLimit, value, "write32 vu", address);
            markVUCodeModified(vuMem);
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        writeIORegister(physAddr, value);
    }
}

void PS2Memory::write64(uint32_t address, uint64_t value, uint32_t writerPc)
{
    if (address & 7)
    {
        throw std::runtime_error("Unaligned 64-bit write at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            // CSR: bits 0..1 are write-one-to-clear status bits. Done as a single
            // atomic RMW -- see writeCsrFull's comment.
            writeCsrFull(gs_regs.csr, value);
        }
        else if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            *reg = value;
        }
        return;
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        storeScalar<uint64_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, value, "write64 scratchpad", address);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        markModified(physAddr, sizeof(value), writerPc);
        storeScalar<uint64_t>(m_rdram, physAddr, PS2_RAM_SIZE, value, "write64 rdram", address);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint64_t), vuOffset, vuLimit))
        {
            storeScalar<uint64_t>(vuMem, vuOffset, vuLimit, value, "write64 vu", address);
            markVUCodeModified(vuMem);
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        write32(address, (uint32_t)value, writerPc);
        write32(address + 4, (uint32_t)(value >> 32), writerPc);
    }
}

void PS2Memory::write128(uint32_t address, __m128i value, uint32_t writerPc)
{
    if (address & 15)
    {
        throw std::runtime_error("Unaligned 128-bit write at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        inRange(physAddr, sizeof(__m128i), PS2_SCRATCHPAD_SIZE, "write128 scratchpad", address);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(&m_scratchpad[physAddr]), value);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        markModified(physAddr, sizeof(__m128i), writerPc);
        inRange(physAddr, sizeof(__m128i), PS2_RAM_SIZE, "write128 rdram", address);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(&m_rdram[physAddr]), value);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(__m128i), vuOffset, vuLimit))
        {
            inRange(vuOffset, sizeof(__m128i), vuLimit, "write128 vu", address);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(vuMem + vuOffset), value);
            markVUCodeModified(vuMem);
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        // Non-RAM 128-bit stores are modeled as two 64-bit stores.
        uint64_t lo = _mm_extract_epi64(value, 0);
        uint64_t hi = _mm_extract_epi64(value, 1);

        write64(address, lo, writerPc);
        write64(address + 8, hi, writerPc);
    }
}

bool PS2Memory::writeIORegister(uint32_t address, uint32_t value)
{
    if (isEeTimer0Register(address))
    {
        if (address == kEeTimer0Count)
        {
            m_ioRegisters[address] = value;
            m_timer0LastHostNs = steadyClockNs();
            m_timer0FractionNs = 0u;
            return true;
        }

        updateEeTimer0Counter();
        m_ioRegisters[address] = value;
        m_timer0LastHostNs = steadyClockNs();
        if (address == kEeTimer0Mode)
        {
            m_timer0FractionNs = 0u;
        }
        return true;
    }

    if (isGsPrivReg(address))
    {
        // NB: unreachable from write8/16/32/64 today since those all funnel IO
        // register writes through addresses in PS2_IO_BASE's range, which is
        // disjoint from PS2_GS_PRIV_REG_BASE; kept correct for direct callers.
        m_ioRegisters[address] = value;
        const uint32_t off = address & 7u;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            writeCsrHalf(gs_regs.csr, off, value);
        }
        else if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            const uint64_t mask = 0xFFFFFFFFull << (off * 8u);
            *reg = (*reg & ~mask) | (static_cast<uint64_t>(value) << (off * 8u));
        }
        m_gsWriteCount.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    if (address >= 0x10002000 && address <= 0x10002030)
    {
        if (address == 0x10002010)
        {
            m_ioRegisters[address] = value & ~(1u << 31);
            if (value & (1u << 30))
            {
                m_ioRegisters[0x10002000] = 0;
                m_ioRegisters[0x10002020] = 0;
                m_ioRegisters[0x10002030] = 0;
            }
        }
        else
        {
            m_ioRegisters[address] = value;
        }
        return true;
    }

    if (address == 0x1000E010u)
    {
        const uint32_t current = m_ioRegisters.count(address) ? m_ioRegisters[address] : 0u;
        uint32_t status = current & 0x3FFu;
        uint32_t mask = (current >> 16) & 0x3FFu;

        // D_STAT low bits are W1C status, high bits [16..25] toggle masks on write-one.
        status &= ~(value & 0x3FFu);
        mask ^= ((value >> 16) & 0x3FFu);

        uint32_t next = (current & ~((0x3FFu) | (0x3FFu << 16) | (1u << 31)));
        next |= status | (mask << 16);
        if ((status & mask) != 0u)
            next |= (1u << 31);
        m_ioRegisters[address] = next;
        if (m_dmacInterruptStateCallback)
        {
            m_dmacInterruptStateCallback();
        }
        return true;
    }

    m_ioRegisters[address] = value;

    if (address >= 0x10003C00u && address < 0x10003E00u)
    {
        m_vifWriteCount.fetch_add(1, std::memory_order_relaxed);

        switch (address)
        {
        case 0x10003C10u:     // VIF1_FBRST
            if (value & 0x1u) // RST
            {
                std::memset(&vif1_regs, 0, sizeof(vif1_regs));
                m_vif1PendingPath2DirectQwc = 0u;
                m_vif1PendingPath2DirectHl = false;
                m_vif1WaitingForVu = false;
                m_vif1DeferredData.clear();
                m_vif1DeferredDmacCompletion.reset();
                (void)cancelDmacTransfer(DmacChannel::Vif1);
                if (m_vif1ResetCallback)
                    m_vif1ResetCallback();
            }
            if (value & 0x8u) // STC
            {
                vif1_regs.stat &= ~((1u << 8) | (1u << 9) | (1u << 10) | (1u << 11) | (1u << 12) | (1u << 13));
            }
            break;
        case 0x10003C30u:
            vif1_regs.mark = value & 0xFFFFu;
            vif1_regs.stat &= ~(1u << 6); // clear MRK flag on CPU write
            break;
        case 0x10003C40u:
            vif1_regs.cycle = value & 0xFFFFu;
            break;
        case 0x10003C50u:
            vif1_regs.mode = value & 0x3u;
            break;
        case 0x10003C60u:
            vif1_regs.num = value & 0xFFu;
            break;
        case 0x10003C70u:
            vif1_regs.mask = value;
            break;
        case 0x10003C80u:
            vif1_regs.code = value;
            break;
        case 0x10003C90u:
            vif1_regs.itops = value & 0x3FFu;
            break;
        case 0x10003CA0u:
            vif1_regs.base = value & 0x3FFu;
            break;
        case 0x10003CB0u:
            vif1_regs.ofst = value & 0x3FFu;
            break;
        case 0x10003CC0u:
            vif1_regs.tops = value & 0x3FFu;
            break;
        case 0x10003CD0u:
            vif1_regs.itop = value & 0x3FFu;
            break;
        case 0x10003CE0u:
            vif1_regs.top = value & 0x3FFu;
            break;
        default:
            break;
        }

        return true;
    }

    if (address >= 0x10003800u && address < 0x10003A00u)
    {
        m_vifWriteCount.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    if (address >= 0x10008000 && address < 0x1000F000)
    {
        const uint32_t channelBase =
            address & 0xFFFFFF00u;
        const std::optional<DmacChannel> channel =
            dmacChannelFromBase(channelBase);
        if ((address & 0xFFu) == 0x00u &&
            channel.has_value() &&
            (value & 0x100u) == 0u)
        {
            (void)cancelDmacTransfer(*channel);
            return true;
        }

        if ((address & 0xFFu) == 0x00u &&
            channel.has_value() &&
            (value & 0x100u) != 0u)
        {
            const auto dctrlIt = m_ioRegisters.find(0x1000E000u);
            const bool dmacEnabled = (dctrlIt == m_ioRegisters.end()) || ((dctrlIt->second & 0x1u) != 0u);
            if (!dmacEnabled)
            {
                return true;
            }

            const DmacTransferToken transfer =
                beginDmacTransfer(*channel);
            const uint32_t madr = m_ioRegisters[channelBase + 0x10];
            const uint32_t qwc = m_ioRegisters[channelBase + 0x20];
            m_dmaStartCount.fetch_add(1, std::memory_order_relaxed);

            if (channelBase == 0x1000D000u || channelBase == 0x1000D400u)
            {
                (void)processScratchpadDma(channelBase);
                return true;
            }

            if ((channelBase == 0x1000A000u || channelBase == 0x10009000u || channelBase == 0x10008000u) &&
                (m_gsVRAM || channelBase == 0x10008000u))
            {
                auto enqueueTransfer = [&](uint32_t srcAddr, uint32_t qwCount)
                {
                    if (qwCount == 0)
                        return;
                    const bool scratch = isScratchpad(srcAddr);
                    PendingTransfer pt;
                    pt.transfer = transfer;
                    pt.fromScratchpad = scratch;
                    pt.srcAddr = srcAddr;
                    pt.qwc = qwCount;
                    if (channelBase == 0x1000A000u)
                        m_pendingGifTransfers.push_back(pt);
                    else if (channelBase == 0x10009000u)
                        m_pendingVif1Transfers.push_back(pt);
                    else if (channelBase == 0x10008000u)
                        m_pendingVif0Transfers.push_back(pt);
                };

                uint32_t chcr = value;
                uint32_t mode = (chcr >> 2) & 0x3;
                if (channelBase == 0x10009000u)
                {
                    beginVif1DmaTrace(
                        chcr, madr, qwc,
                        m_ioRegisters[channelBase + 0x30],
                        m_ioRegisters[channelBase + 0x40],
                        m_ioRegisters[channelBase + 0x50],
                        m_ioRegisters[channelBase + 0x80]);
                }

                if (mode == 0 && qwc > 0)
                {
                    if (channelBase == 0x10009000u)
                        traceVif1NormalTransfer(madr, qwc);
                    enqueueTransfer(madr, qwc);
                }
                else if (mode == 1)
                {
                    uint32_t tagAddr = m_ioRegisters[channelBase + 0x30];
                    uint32_t asr0 = m_ioRegisters[channelBase + 0x40];
                    uint32_t asr1 = m_ioRegisters[channelBase + 0x50];
                    uint32_t asp = (chcr >> 4) & 0x3u;
                    const bool tieEnabled = (chcr & (1u << 7)) != 0u;
                    const bool tagTransferEnabled = (chcr & (1u << 6)) != 0u;
                    constexpr uint32_t kMaxChainTags = 65536u;
                    std::vector<uint8_t> chainBuf;

                    auto appendData = [&](uint32_t srcAddr, uint32_t qwCount)
                    {
                        const uint64_t bytes64 = static_cast<uint64_t>(qwCount) * 16ull;
                        uint32_t bytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
                        const bool scratch = isScratchpad(srcAddr);
                        uint32_t src = 0;
                        src = translateAddress(srcAddr);
                        const uint8_t *base2;
                        uint32_t maxSz2;
                        if (scratch)
                        {
                            base2 = m_scratchpad;
                            maxSz2 = PS2_SCRATCHPAD_SIZE;
                        }
                        else
                        {
                            base2 = m_rdram;
                            maxSz2 = PS2_RAM_SIZE;
                        }

                        while (bytes > 0)
                        {
                            if (src >= maxSz2)
                                src = 0;
                            uint32_t chunk = bytes;
                            if (src + chunk > maxSz2)
                                chunk = maxSz2 - src;
                            if (chunk == 0)
                                break;
                            if (channelBase == 0x10009000u)
                                traceVif1DmaPayload(base2 + src, chunk);
                            chainBuf.insert(chainBuf.end(), base2 + src, base2 + src + chunk);
                            bytes -= chunk;
                            src += chunk;
                        }
                    };

                    uint32_t tagsProcessed = 0u;
                    uint32_t lastTagUpper = (chcr >> 16) & 0xFFFFu;
                    bool chainComplete = false;
                    bool chainFailed = false;
                    const char *chainFailure = nullptr;
                    std::unordered_set<DmaChainState, DmaChainStateHash> visitedStates;
                    visitedStates.reserve(8192u);

                    while (tagsProcessed < kMaxChainTags)
                    {
                        const uint32_t currentTagAddr = tagAddr;
                        const DmaChainState state{currentTagAddr, asr0, asr1, asp};
                        if (!visitedStates.insert(state).second)
                        {
                            chainFailed = true;
                            chainFailure = "repeated chain state";
                            break;
                        }

                        const bool tagInSPR = isScratchpad(tagAddr);
                        uint32_t physTag = 0;
                        try
                        {
                            physTag = translateAddress(tagAddr);
                        }
                        catch (...)
                        {
                            chainFailed = true;
                            chainFailure = "unmapped tag address";
                            break;
                        }
                        const uint8_t *tagBase;
                        uint32_t tagMax;
                        if (tagInSPR)
                        {
                            tagBase = m_scratchpad;
                            tagMax = PS2_SCRATCHPAD_SIZE;
                        }
                        else
                        {
                            tagBase = m_rdram;
                            tagMax = PS2_RAM_SIZE;
                        }
                        if (physTag + 16 > tagMax)
                        {
                            chainFailed = true;
                            chainFailure = "tag crosses memory boundary";
                            break;
                        }

                        const uint8_t *tp = tagBase + physTag;
                        uint64_t tag = loadScalar<uint64_t>(tp, 0, 16, "dma chain tag", tagAddr);
                        if (channelBase == 0x10009000u)
                            traceVif1DmaTag(tp);
                        uint16_t tagQwc = static_cast<uint16_t>(tag & 0xFFFF);
                        uint32_t id = static_cast<uint32_t>((tag >> 28) & 0x7);
                        const bool irq = ((tag >> 31) & 0x1ull) != 0ull;
                        uint32_t addr = static_cast<uint32_t>((tag >> 32) & 0x7FFFFFFF);
                        lastTagUpper = static_cast<uint32_t>((tag >> 16) & 0xFFFFu);
                        ++tagsProcessed;

                        uint32_t dataAddr = 0;
                        bool hasPayload = (tagQwc > 0);
                        bool endChain = false;

                        switch (id)
                        {
                        case 0:
                            dataAddr = addr;
                            tagAddr = tagAddr + 16;
                            endChain = true;
                            break;
                        case 1:
                            dataAddr = tagAddr + 16;
                            tagAddr = dataAddr + static_cast<uint32_t>(tagQwc) * 16u;
                            break;
                        case 2:
                            dataAddr = tagAddr + 16;
                            tagAddr = addr;
                            break;
                        case 3:
                        case 4:
                            dataAddr = addr;
                            tagAddr = tagAddr + 16;
                            break;
                        case 5:
                            dataAddr = tagAddr + 16;
                            {
                                const uint32_t retAddr = dataAddr + static_cast<uint32_t>(tagQwc) * 16u;
                                if (asp == 0u)
                                {
                                    asr0 = retAddr;
                                    asp = 1u;
                                }
                                else if (asp == 1u)
                                {
                                    asr1 = retAddr;
                                    asp = 2u;
                                }
                            }
                            tagAddr = addr;
                            break;
                        case 6:
                            dataAddr = tagAddr + 16;
                            if (asp == 2u)
                            {
                                tagAddr = asr1;
                                asp = 1u;
                            }
                            else if (asp == 1u)
                            {
                                tagAddr = asr0;
                                asp = 0u;
                            }
                            else
                            {
                                endChain = true;
                            }
                            break;
                        case 7:
                            dataAddr = tagAddr + 16;
                            endChain = true;
                            break;
                        default:
                            hasPayload = false;
                            endChain = true;
                            break;
                        }

                        if (channelBase == 0x10009000u &&
                            m_gsDmaTrace && m_gsDmaTrace->active &&
                            m_gsDmaTrace->output)
                        {
                            std::fprintf(
                                m_gsDmaTrace->output,
                                "{\"schema_version\":1,"
                                "\"event\":\"vif1-chain-tag\","
                                "\"sequence\":%" PRIu64 ","
                                "\"index\":%" PRIu32 ","
                                "\"stream_offset\":%zu,"
                                "\"tag_address\":\"0x%08" PRIx32 "\","
                                "\"qwc\":%" PRIu32 ",\"id\":%" PRIu32 ","
                                "\"irq\":%s,"
                                "\"target_address\":\"0x%08" PRIx32 "\","
                                "\"data_address\":\"0x%08" PRIx32 "\"}\n",
                                m_gsDmaTrace->activeSequence,
                                tagsProcessed - 1u, chainBuf.size(),
                                currentTagAddr, static_cast<uint32_t>(tagQwc),
                                id, irq ? "true" : "false", addr, dataAddr);
                        }

                        // VIF0/VIF1 consume the upper 64 bits of every source-chain
                        // tag before its QWC payload when CHCR.TTE is set. This
                        // includes reference tags whose payload lives elsewhere.
                        const bool transferVifTag = tagTransferEnabled &&
                            (channelBase == 0x10009000u || channelBase == 0x10008000u);
                        if (transferVifTag)
                            chainBuf.insert(chainBuf.end(), tp + 8u, tp + 16u);

                        if (hasPayload)
                            appendData(dataAddr, tagQwc);
                        if (irq && tieEnabled)
                            endChain = true;
                        if (endChain)
                        {
                            chainComplete = true;
                            break;
                        }
                    }

                    if (!chainComplete && !chainFailed)
                    {
                        chainFailed = true;
                        chainFailure = "tag safety limit reached";
                    }

                    m_ioRegisters[channelBase + 0x30] = tagAddr;
                    m_ioRegisters[channelBase + 0x40] = asr0;
                    m_ioRegisters[channelBase + 0x50] = asr1;
                    chcr = (chcr & ~(0x3u << 4)) | ((asp & 0x3u) << 4);
                    chcr = (chcr & 0x0000FFFFu) | (lastTagUpper << 16);
                    m_ioRegisters[channelBase + 0x00] = chcr;

                    if (chainFailed)
                    {
                        std::fprintf(stderr,
                                     "DMAC source chain did not terminate: channel=0x%08x "
                                     "tadr=0x%08x tags=%u reason=%s\n",
                                     channelBase, tagAddr, tagsProcessed,
                                     chainFailure ? chainFailure : "unknown");
                    }
                    else if (chainComplete)
                    {
                        if (channelBase == 0x10009000u && !chainBuf.empty())
                            traceVif1Input(chainBuf.data(), static_cast<uint32_t>(chainBuf.size()));
                        PendingTransfer pt;
                        pt.transfer = transfer;
                        pt.fromScratchpad = false;
                        pt.srcAddr = 0;
                        pt.qwc = 0;
                        pt.chainData = std::move(chainBuf);
                        if (channelBase == 0x1000A000)
                        {
                            m_pendingGifTransfers.push_back(std::move(pt));
                        }
                        else if (channelBase == 0x10009000u)
                        {
                            m_pendingVif1Transfers.push_back(std::move(pt));
                        }
                        else if (channelBase == 0x10008000u)
                        {
                            m_pendingVif0Transfers.push_back(std::move(pt));
                        }
                    }
                    // else if (channelBase == 0x10009000u)
                    // {

                    // }
                }
                else if (qwc > 0)
                {
                    enqueueTransfer(madr, qwc);
                }

                const bool autoProcessTransfers =
                    (channelBase == 0x1000A000u) ? (m_gifPacketCallback || m_gifArbiter != nullptr) : true;
                if (autoProcessTransfers)
                {
                    processPendingTransfers();
                }
                if (channelBase == 0x10009000u)
                {
                    const bool cooperativeVu1Active =
                        m_vu1BusyCallback &&
                        m_vu1BusyCallback();
                    if (!cooperativeVu1Active)
                        finishVif1DmaTrace();
                }
            }
        }
        return true;
    }

    if (address >= 0x10000000 && address < 0x10010000)
    {
        if (address >= 0x10000200 && address < 0x10000300)
        {
            return true;
        }
        if (address >= 0x10000000 && address < 0x10000100)
        {
            return true;
        }
    }

    return false;
}

bool PS2Memory::processScratchpadDma(uint32_t channelBase)
{
    constexpr uint32_t kFromScratchpadChannel = 0x1000D000u;
    constexpr uint32_t kToScratchpadChannel = 0x1000D400u;
    constexpr uint32_t kMaxChainTags = 4096u;

    if (channelBase != kFromScratchpadChannel && channelBase != kToScratchpadChannel)
    {
        return false;
    }

    uint32_t chcr = m_ioRegisters[channelBase + 0x00u];
    const uint32_t mode = (chcr >> 2u) & 0x3u;
    const bool toScratchpad = channelBase == kToScratchpadChannel;
    if (mode > 1u || (mode == 1u && !toScratchpad))
    {
        return false;
    }

    uint32_t madr = m_ioRegisters[channelBase + 0x10u] & 0x7FFFFFFFu;
    uint32_t qwc = m_ioRegisters[channelBase + 0x20u] & 0xFFFFu;
    uint32_t tadr = m_ioRegisters[channelBase + 0x30u] & 0x7FFFFFFFu;
    uint32_t asr0 = m_ioRegisters[channelBase + 0x40u] & 0x7FFFFFFFu;
    uint32_t asr1 = m_ioRegisters[channelBase + 0x50u] & 0x7FFFFFFFu;
    uint32_t sadr = m_ioRegisters[channelBase + 0x80u] & 0x3FF0u;

    auto transferQwords = [&](uint32_t &memoryAddress, uint32_t count)
    {
        for (uint32_t index = 0u; index < count; ++index)
        {
            const uint32_t scratchAddress = PS2_SCRATCHPAD_BASE + sadr;
            if (toScratchpad)
            {
                write128(scratchAddress, read128(memoryAddress));
            }
            else
            {
                write128(memoryAddress, read128(scratchAddress));
            }

            memoryAddress = (memoryAddress + 16u) & 0x7FFFFFFFu;
            sadr = (sadr + 16u) & (PS2_SCRATCHPAD_SIZE - 1u);
        }
    };

    try
    {
        const bool resumedChain = mode == 1u && qwc != 0u;
        transferQwords(madr, qwc);
        qwc = 0u;

        bool endChain = false;
        if (resumedChain)
        {
            const uint32_t previousTagId = (chcr >> 28u) & 0x7u;
            const bool previousTagIrq = (chcr & (1u << 31u)) != 0u;
            const bool tieEnabled = (chcr & (1u << 7u)) != 0u;
            endChain = previousTagId == 0u || previousTagId == 7u ||
                       (tieEnabled && previousTagIrq);
        }

        uint32_t tagsProcessed = 0u;
        while (mode == 1u && !endChain && tagsProcessed < kMaxChainTags)
        {
            const uint32_t tagWord0 = read32(tadr + 0u);
            const uint32_t tagAddress = read32(tadr + 4u) & 0x7FFFFFFFu;
            const uint32_t tagQwc = tagWord0 & 0xFFFFu;
            const uint32_t tagId = (tagWord0 >> 28u) & 0x7u;
            const bool tagIrq = (tagWord0 & (1u << 31u)) != 0u;
            const bool tieEnabled = (chcr & (1u << 7u)) != 0u;
            const bool tagTransferEnabled = (chcr & (1u << 6u)) != 0u;
            ++tagsProcessed;

            chcr = (chcr & 0x0000FFFFu) | (tagWord0 & 0xFFFF0000u);

            if (tagTransferEnabled)
            {
                const uint32_t scratchAddress = PS2_SCRATCHPAD_BASE + sadr;
                write128(scratchAddress, read128(tadr));
                sadr = (sadr + 16u) & (PS2_SCRATCHPAD_SIZE - 1u);
            }

            uint32_t dataAddress = tagAddress;
            switch (tagId)
            {
            case 0u: // REFE
                tadr = (tadr + 16u) & 0x7FFFFFFFu;
                endChain = true;
                break;
            case 1u: // CNT
                dataAddress = (tadr + 16u) & 0x7FFFFFFFu;
                tadr = (dataAddress + tagQwc * 16u) & 0x7FFFFFFFu;
                break;
            case 2u: // NEXT
                dataAddress = (tadr + 16u) & 0x7FFFFFFFu;
                tadr = tagAddress;
                break;
            case 3u: // REF
            case 4u: // REFS
                tadr = (tadr + 16u) & 0x7FFFFFFFu;
                break;
            case 5u: // CALL
            {
                dataAddress = (tadr + 16u) & 0x7FFFFFFFu;
                const uint32_t returnAddress =
                    (dataAddress + tagQwc * 16u) & 0x7FFFFFFFu;
                uint32_t asp = (chcr >> 4u) & 0x3u;
                if (asp == 0u)
                {
                    asr0 = returnAddress;
                    asp = 1u;
                }
                else if (asp == 1u)
                {
                    asr1 = returnAddress;
                    asp = 2u;
                }
                else
                {
                    endChain = true;
                }
                chcr = (chcr & ~(0x3u << 4u)) | (asp << 4u);
                tadr = tagAddress;
                break;
            }
            case 6u: // RET
            {
                dataAddress = (tadr + 16u) & 0x7FFFFFFFu;
                uint32_t asp = (chcr >> 4u) & 0x3u;
                if (asp == 2u)
                {
                    tadr = asr1;
                    asr1 = 0u;
                    asp = 1u;
                }
                else if (asp == 1u)
                {
                    tadr = asr0;
                    asr0 = 0u;
                    asp = 0u;
                }
                else
                {
                    endChain = true;
                }
                chcr = (chcr & ~(0x3u << 4u)) | (asp << 4u);
                break;
            }
            case 7u: // END
                dataAddress = (tadr + 16u) & 0x7FFFFFFFu;
                endChain = true;
                break;
            default:
                endChain = true;
                break;
            }

            madr = dataAddress;
            qwc = tagQwc;
            transferQwords(madr, qwc);
            qwc = 0u;

            if (tieEnabled && tagIrq)
            {
                endChain = true;
            }
        }

        if (mode == 1u && !endChain)
        {
            return false;
        }
    }
    catch (const std::exception &error)
    {
        PS2_IF_AGRESSIVE_LOGS({
            RUNTIME_LOG("[dmac:scratchpad] transfer failed channel=0x" << std::hex
                        << channelBase << " madr=0x" << madr
                        << " sadr=0x" << sadr << " qwc=0x" << qwc
                        << std::dec << " error=" << error.what() << std::endl);
        });
        return false;
    }

    m_ioRegisters[channelBase + 0x00u] = chcr;
    m_ioRegisters[channelBase + 0x10u] = madr;
    m_ioRegisters[channelBase + 0x20u] = qwc;
    m_ioRegisters[channelBase + 0x30u] = tadr;
    m_ioRegisters[channelBase + 0x40u] = asr0;
    m_ioRegisters[channelBase + 0x50u] = asr1;
    m_ioRegisters[channelBase + 0x80u] = sadr;
    const std::optional<DmacChannel> channel =
        dmacChannelFromBase(channelBase);
    if (!channel.has_value())
    {
        return false;
    }
    const DmacChannelSnapshot state =
        dmacChannelSnapshot(*channel);
    (void)requestDmacCompletion(
        DmacTransferToken{*channel, state.generation});
    return true;
}

uint64_t PS2Memory::nextDmacSequence(uint64_t value) noexcept
{
    ++value;
    if (value == 0u)
    {
        value = 1u;
    }
    return value;
}

void PS2Memory::discardPendingDmacWork(DmacChannel channel)
{
    const auto isChannel =
        [channel](const PendingTransfer &pending)
        {
            return pending.transfer.channel == channel;
        };
    m_pendingGifTransfers.erase(
        std::remove_if(
            m_pendingGifTransfers.begin(),
            m_pendingGifTransfers.end(), isChannel),
        m_pendingGifTransfers.end());
    m_pendingVif0Transfers.erase(
        std::remove_if(
            m_pendingVif0Transfers.begin(),
            m_pendingVif0Transfers.end(), isChannel),
        m_pendingVif0Transfers.end());
    m_pendingVif1Transfers.erase(
        std::remove_if(
            m_pendingVif1Transfers.begin(),
            m_pendingVif1Transfers.end(), isChannel),
        m_pendingVif1Transfers.end());
}

DmacTransferToken PS2Memory::beginDmacTransfer(DmacChannel channel)
{
    const size_t index = static_cast<size_t>(channel);
    if (index >= m_dmacChannels.size())
    {
        return {};
    }

    discardPendingDmacWork(channel);
    if (channel == DmacChannel::Vif1)
        m_vif1DeferredDmacCompletion.reset();

    DmacTransferToken transfer{channel, 0u};
    {
        std::lock_guard<std::mutex> lock(m_dmacMutex);
        DmacChannelLifecycle &state = m_dmacChannels[index];
        state.generation = nextDmacSequence(state.generation);
        state.phase = DmacChannelPhase::Active;
        transfer.generation = state.generation;

        m_readyDmacCompletions.erase(
            std::remove_if(
                m_readyDmacCompletions.begin(),
                m_readyDmacCompletions.end(),
                [channel](const DmacCompletionRequest &completion)
                {
                    return completion.transfer.channel == channel;
                }),
            m_readyDmacCompletions.end());
    }

    const uint32_t channelBase = dmacChannelBase(channel);
    if (channelBase != 0u)
    {
        m_ioRegisters[channelBase + 0x00u] |= 0x100u;
    }
    return transfer;
}

bool PS2Memory::progressDmacTransfer(
    DmacTransferToken transfer) const
{
    const size_t index =
        static_cast<size_t>(transfer.channel);
    if (index >= m_dmacChannels.size() ||
        transfer.generation == 0u)
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_dmacMutex);
    const DmacChannelLifecycle &state =
        m_dmacChannels[index];
    return state.generation == transfer.generation &&
           state.phase == DmacChannelPhase::Active;
}

bool PS2Memory::requestDmacCompletion(
    DmacTransferToken transfer)
{
    const size_t index =
        static_cast<size_t>(transfer.channel);
    if (index >= m_dmacChannels.size() ||
        transfer.generation == 0u)
    {
        return false;
    }

    DmacCompletionReadyCallback callback;
    {
        std::lock_guard<std::mutex> lock(m_dmacMutex);
        DmacChannelLifecycle &state =
            m_dmacChannels[index];
        if (state.generation != transfer.generation ||
            state.phase != DmacChannelPhase::Active)
        {
            return false;
        }

        state.phase = DmacChannelPhase::CompletionReady;
        m_dmacReadySequence =
            nextDmacSequence(m_dmacReadySequence);
        m_readyDmacCompletions.push_back(
            DmacCompletionRequest{
                transfer, m_dmacReadySequence});
        callback = m_dmacCompletionReadyCallback;
    }

    if (callback)
    {
        callback();
    }
    return true;
}

bool PS2Memory::cancelDmacTransfer(DmacChannel channel)
{
    const size_t index = static_cast<size_t>(channel);
    if (index >= m_dmacChannels.size())
    {
        return false;
    }

    discardPendingDmacWork(channel);
    if (channel == DmacChannel::Vif1)
        m_vif1DeferredDmacCompletion.reset();

    bool cancelled = false;
    {
        std::lock_guard<std::mutex> lock(m_dmacMutex);
        DmacChannelLifecycle &state = m_dmacChannels[index];
        cancelled = state.phase != DmacChannelPhase::Idle;
        state.generation = nextDmacSequence(state.generation);
        state.phase = DmacChannelPhase::Idle;
        m_readyDmacCompletions.erase(
            std::remove_if(
                m_readyDmacCompletions.begin(),
                m_readyDmacCompletions.end(),
                [channel](const DmacCompletionRequest &completion)
                {
                    return completion.transfer.channel == channel;
                }),
            m_readyDmacCompletions.end());
    }

    const uint32_t channelBase = dmacChannelBase(channel);
    if (channelBase != 0u)
    {
        m_ioRegisters[channelBase + 0x00u] &= ~0x100u;
    }
    return cancelled;
}

bool PS2Memory::hasReadyDmacCompletions() const
{
    std::lock_guard<std::mutex> lock(m_dmacMutex);
    return !m_readyDmacCompletions.empty();
}

std::vector<DmacCompletionRequest>
PS2Memory::consumeReadyDmacCompletions()
{
    std::lock_guard<std::mutex> lock(m_dmacMutex);
    std::vector<DmacCompletionRequest> completions;
    completions.swap(m_readyDmacCompletions);
    return completions;
}

bool PS2Memory::publishDmacCompletion(
    DmacCompletionRequest completion,
    uint64_t eventSequence)
{
    constexpr uint32_t kDStat = 0x1000E010u;
    const DmacChannel channel =
        completion.transfer.channel;
    const size_t index = static_cast<size_t>(channel);
    if (index >= m_dmacChannels.size() ||
        completion.transfer.generation == 0u)
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_dmacMutex);
    DmacChannelLifecycle &state = m_dmacChannels[index];
    if (state.generation !=
            completion.transfer.generation ||
        state.phase != DmacChannelPhase::CompletionReady)
    {
        return false;
    }

    state.phase = DmacChannelPhase::Idle;
    const uint32_t channelBase = dmacChannelBase(channel);
    m_ioRegisters[channelBase + 0x00u] &= ~0x100u;
    m_ioRegisters[channelBase + 0x20u] = 0u;

    const uint32_t cause = dmacChannelCause(channel);
    uint32_t dstat =
        m_ioRegisters.count(kDStat)
            ? m_ioRegisters[kDStat]
            : 0u;
    dstat |= (1u << cause);
    const uint32_t status = dstat & 0x3FFu;
    const uint32_t mask =
        (dstat >> 16u) & 0x3FFu;
    if ((status & mask) != 0u)
    {
        dstat |= (1u << 31u);
    }
    else
    {
        dstat &= ~(1u << 31u);
    }
    m_ioRegisters[kDStat] = dstat;

    m_dmacPublicationSequence =
        nextDmacSequence(m_dmacPublicationSequence);
    m_pendingDmacInterrupts.push_back(
        DmacPendingInterrupt{
            channel,
            cause,
            completion.transfer.generation,
            eventSequence,
            m_dmacPublicationSequence});
    return true;
}

size_t PS2Memory::publishReadyDmacCompletions(
    uint64_t eventSequence)
{
    std::vector<DmacCompletionRequest> completions =
        consumeReadyDmacCompletions();
    size_t published = 0u;
    for (const DmacCompletionRequest &completion :
         completions)
    {
        published += publishDmacCompletion(
                         completion, eventSequence)
                         ? 1u
                         : 0u;
    }
    return published;
}

std::vector<DmacPendingInterrupt>
PS2Memory::consumePendingDmacInterrupts()
{
    std::lock_guard<std::mutex> lock(m_dmacMutex);
    std::vector<DmacPendingInterrupt> interrupts;
    interrupts.swap(m_pendingDmacInterrupts);
    return interrupts;
}

DmacChannelSnapshot PS2Memory::dmacChannelSnapshot(
    DmacChannel channel) const
{
    const size_t index = static_cast<size_t>(channel);
    if (index >= m_dmacChannels.size())
    {
        return {};
    }

    std::lock_guard<std::mutex> lock(m_dmacMutex);
    const DmacChannelLifecycle &state =
        m_dmacChannels[index];
    return DmacChannelSnapshot{
        state.phase, state.generation};
}

bool PS2Memory::dmacInterruptStatusLatched(
    DmacChannel channel) const
{
    constexpr uint32_t kDStat = 0x1000E010u;
    const uint32_t cause = dmacChannelCause(channel);
    if (cause >= 10u)
    {
        return false;
    }
    const auto found = m_ioRegisters.find(kDStat);
    return found != m_ioRegisters.end() &&
           (found->second & (1u << cause)) != 0u;
}

bool PS2Memory::dmacInterruptUnmasked(
    DmacChannel channel) const
{
    constexpr uint32_t kDStat = 0x1000E010u;
    const uint32_t cause = dmacChannelCause(channel);
    if (cause >= 10u)
    {
        return false;
    }
    const auto found = m_ioRegisters.find(kDStat);
    return found != m_ioRegisters.end() &&
           (found->second & (1u << (cause + 16u))) != 0u;
}

void PS2Memory::setDmacInterruptMask(
    DmacChannel channel, bool enabled)
{
    constexpr uint32_t kDStat = 0x1000E010u;
    const uint32_t cause = dmacChannelCause(channel);
    if (cause >= 10u)
    {
        return;
    }

    uint32_t &dstat = m_ioRegisters[kDStat];
    const uint32_t maskBit = 1u << (cause + 16u);
    if (enabled)
    {
        dstat |= maskBit;
    }
    else
    {
        dstat &= ~maskBit;
    }

    const uint32_t status = dstat & 0x3FFu;
    const uint32_t mask =
        (dstat >> 16u) & 0x3FFu;
    if ((status & mask) != 0u)
    {
        dstat |= (1u << 31u);
    }
    else
    {
        dstat &= ~(1u << 31u);
    }

    if (m_dmacInterruptStateCallback)
    {
        m_dmacInterruptStateCallback();
    }
}

void PS2Memory::processPendingTransfers()
{
    std::optional<DmacTransferToken> finishedGif;
    for (size_t idx = 0; idx < m_pendingGifTransfers.size(); ++idx)
    {
        auto &p = m_pendingGifTransfers[idx];
        if (!progressDmacTransfer(p.transfer))
        {
            continue;
        }
        finishedGif = p.transfer;
        if (!p.chainData.empty())
        {
            m_seenGifCopy = true;
            m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
            submitGifPacket(GifPathId::Path3, p.chainData.data(), static_cast<uint32_t>(p.chainData.size()), false);
        }
        else if (p.qwc > 0)
        {
            const uint64_t bytes64 = static_cast<uint64_t>(p.qwc) * 16ull;
            uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
            uint32_t srcPhys = 0;
            try
            {
                srcPhys = translateAddress(p.srcAddr);
            }
            catch (const std::exception &)
            {
                continue;
            }
            if (p.fromScratchpad)
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft >= 16)
                {
                    if (srcPhys >= PS2_SCRATCHPAD_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_SCRATCHPAD_SIZE)
                        chunk = PS2_SCRATCHPAD_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    m_seenGifCopy = true;
                    m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
                    submitGifPacket(GifPathId::Path3, m_scratchpad + srcPhys, chunk, false);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
            else
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft >= 16)
                {
                    if (srcPhys >= PS2_RAM_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_RAM_SIZE)
                        chunk = PS2_RAM_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    m_seenGifCopy = true;
                    m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
                    submitGifPacket(GifPathId::Path3, m_rdram + srcPhys, chunk, false);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
        }
    }
    m_pendingGifTransfers.clear();

    std::optional<DmacTransferToken> finishedVif0;
    for (auto &p : m_pendingVif0Transfers)
    {
        if (!progressDmacTransfer(p.transfer))
        {
            continue;
        }
        finishedVif0 = p.transfer;
        if (!p.chainData.empty())
        {
            processVIF0Data(p.chainData.data(), static_cast<uint32_t>(p.chainData.size()));
        }
        else if (p.qwc > 0)
        {
            uint32_t srcPhys = 0;
            const uint64_t bytes64 = static_cast<uint64_t>(p.qwc) * 16ull;
            uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
            try
            {
                srcPhys = translateAddress(p.srcAddr);
            }
            catch (const std::exception &)
            {
                continue;
            }
            if (p.fromScratchpad)
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft > 0)
                {
                    if (srcPhys >= PS2_SCRATCHPAD_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_SCRATCHPAD_SIZE)
                        chunk = PS2_SCRATCHPAD_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    processVIF0Data(m_scratchpad + srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
            else
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft > 0)
                {
                    if (srcPhys >= PS2_RAM_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_RAM_SIZE)
                        chunk = PS2_RAM_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    processVIF0Data(srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
        }
    }
    m_pendingVif0Transfers.clear();

    std::optional<DmacTransferToken> finishedVif1;
    for (auto &p : m_pendingVif1Transfers)
    {
        if (!progressDmacTransfer(p.transfer))
        {
            continue;
        }
        finishedVif1 = p.transfer;
        if (!p.chainData.empty())
        {
            processVIF1Data(p.chainData.data(), static_cast<uint32_t>(p.chainData.size()));
        }
        else if (p.qwc > 0)
        {
            uint32_t srcPhys = 0;
            const uint64_t bytes64 = static_cast<uint64_t>(p.qwc) * 16ull;
            uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
            try
            {
                srcPhys = translateAddress(p.srcAddr);
            }
            catch (const std::exception &)
            {
                continue;
            }
            if (p.fromScratchpad)
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft > 0)
                {
                    if (srcPhys >= PS2_SCRATCHPAD_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_SCRATCHPAD_SIZE)
                        chunk = PS2_SCRATCHPAD_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    processVIF1Data(m_scratchpad + srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
            else
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft > 0)
                {
                    if (srcPhys >= PS2_RAM_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_RAM_SIZE)
                        chunk = PS2_RAM_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    processVIF1Data(srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
        }
    }
    m_pendingVif1Transfers.clear();

    if (m_gifArbiter)
        m_gifArbiter->drain();

    if (finishedGif.has_value())
    {
        (void)requestDmacCompletion(*finishedGif);
    }
    if (finishedVif0.has_value())
    {
        (void)requestDmacCompletion(*finishedVif0);
    }
    if (finishedVif1.has_value())
    {
        if (m_vif1WaitingForVu)
            m_vif1DeferredDmacCompletion = *finishedVif1;
        else
            (void)requestDmacCompletion(*finishedVif1);
    }
}

void PS2Memory::flushMaskedPath3Packets(bool drainImmediately)
{
    if (m_path3Masked || m_path3MaskedFifo.empty())
        return;

    auto emit = [&](const uint8_t *packetData, uint32_t packetSize)
    {
        if (m_gifArbiter)
            m_gifArbiter->submit(GifPathId::Path3, packetData, packetSize, false);
        else if (m_gifPacketCallback)
            m_gifPacketCallback(packetData, packetSize);
    };

    for (const auto &packet : m_path3MaskedFifo)
    {
        if (packet.size() >= 16u)
            emit(packet.data(), static_cast<uint32_t>(packet.size()));
    }
    m_path3MaskedFifo.clear();

    if (m_gifArbiter && drainImmediately)
        m_gifArbiter->drain();
}

void PS2Memory::submitGifPacket(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool drainImmediately, bool path2DirectHl)
{
    if (!data || sizeBytes < 16)
        return;

    if (pathId == GifPathId::Path1)
        tracePath1Packet(data, sizeBytes);

    if (pathId == GifPathId::Path3)
    {
        if (m_path3Masked)
        {
            m_path3MaskedFifo.emplace_back(data, data + sizeBytes);
            return;
        }
        flushMaskedPath3Packets(false);
    }

    if (m_gifArbiter)
        m_gifArbiter->submit(pathId, data, sizeBytes, path2DirectHl);
    else if (m_gifPacketCallback)
        m_gifPacketCallback(data, sizeBytes);

    if (m_gifArbiter && drainImmediately)
        m_gifArbiter->drain();
}

void PS2Memory::processGIFPacket(uint32_t srcPhysAddr, uint32_t qwCount)
{
    if (!m_rdram || qwCount == 0)
        return;
    const uint64_t bytes64 = static_cast<uint64_t>(qwCount) * 16ull;
    uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
    uint32_t bytesLeft = sizeBytes;
    while (bytesLeft >= 16)
    {
        if (srcPhysAddr >= PS2_RAM_SIZE)
            srcPhysAddr = 0;
        uint32_t chunk = bytesLeft;
        if (srcPhysAddr + chunk > PS2_RAM_SIZE)
            chunk = PS2_RAM_SIZE - srcPhysAddr;
        if (chunk == 0)
            break;

        m_seenGifCopy = true;
        m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
        submitGifPacket(GifPathId::Path3, m_rdram + srcPhysAddr, chunk);

        bytesLeft -= chunk;
        srcPhysAddr += chunk;
    }
}

void PS2Memory::processGIFPacket(const uint8_t *data, uint32_t sizeBytes)
{
    if (m_gifArbiter)
        submitGifPacket(GifPathId::Path3, data, sizeBytes);
    else if (m_gifPacketCallback && data && sizeBytes >= 16)
        m_gifPacketCallback(data, sizeBytes);
}

bool PS2Memory::tryProcessNativeGifImageUploadChain(GS &gs, uint32_t tadr, uint32_t chcr)
{
    static constexpr uint32_t GIF_CHANNEL = 0x1000A000u;
    static constexpr uint32_t D_CTRL = 0x1000E000u;

    if (!m_rdram || !m_gsVRAM || m_path3Masked)
        return false;
    if (m_gifArbiter && !m_gifArbiter->empty())
        return false;
    if ((chcr & 0x100u) == 0u || ((chcr >> 2u) & 0x3u) != 1u)
        return false;
    if ((chcr & (1u << 7u)) != 0u || ((chcr >> 4u) & 0x3u) != 0u)
        return false;

    const auto dctrlIt = m_ioRegisters.find(D_CTRL);
    if (dctrlIt != m_ioRegisters.end() && ((dctrlIt->second & 0x1u) == 0u))
        return false;

    auto resolveContiguous = [&](uint32_t guestAddr, uint32_t bytes, const uint8_t *&out) -> bool
    {
        try
        {
            const bool scratch = isScratchpad(guestAddr);
            const uint32_t phys = translateAddress(guestAddr);
            const uint8_t *base = scratch ? m_scratchpad : m_rdram;
            const uint32_t limit = scratch ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
            if (!base || phys > limit || bytes > limit - phys)
                return false;
            out = base + phys;
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    };

    auto loadDmaTagAt = [&](uint32_t guestAddr, DmaTagView &out) -> bool
    {
        const uint8_t *ptr = nullptr;
        if (!resolveContiguous(guestAddr, 16u, ptr))
            return false;
        out = decodeDmaTag(loadScalar<uint64_t>(ptr, 0u, 16u, "native gif dma tag", guestAddr));
        return true;
    };

    auto decodeSetupPayload = [&](const uint8_t *payload, uint64_t (&regs)[4]) -> bool
    {
        const uint64_t tagLo = loadScalar<uint64_t>(payload, 0u, 80u, "native gif setup tag", 0u);
        const uint64_t tagHi = loadScalar<uint64_t>(payload, 8u, 80u, "native gif setup regs", 0u);
        if (gifTagNloop(tagLo) != 4u ||
            gifTagFlg(tagLo) != GIF_FMT_PACKED ||
            gifTagNreg(tagLo) != 1u ||
            (tagHi & 0xFull) != 0x0Eull)
        {
            return false;
        }

        static constexpr uint8_t kExpectedRegs[4] = {
            GS_REG_BITBLTBUF,
            GS_REG_TRXPOS,
            GS_REG_TRXREG,
            GS_REG_TRXDIR,
        };

        uint32_t offset = 16u;
        for (uint32_t i = 0; i < 4u; ++i)
        {
            regs[i] = loadScalar<uint64_t>(payload, offset, 80u, "native gif setup value", 0u);
            const uint64_t reg = loadScalar<uint64_t>(payload, offset + 8u, 80u, "native gif setup register", 0u);
            if ((reg & 0xFFu) != kExpectedRegs[i])
                return false;
            offset += 16u;
        }

        const uint32_t trxdirMode = static_cast<uint32_t>(regs[3] & 0x3ull);
        const uint32_t rrw = static_cast<uint32_t>(regs[2] & 0xFFFull);
        const uint32_t rrh = static_cast<uint32_t>((regs[2] >> 32u) & 0xFFFull);
        return trxdirMode == 0u && rrw != 0u && rrh != 0u;
    };

    DmaTagView setupTag{};
    if (!loadDmaTagAt(tadr, setupTag) ||
        setupTag.id != 1u ||
        setupTag.qwc != 5u ||
        setupTag.irq)
    {
        return false;
    }

    const uint8_t *setupPayload = nullptr;
    const uint32_t setupPayloadAddr = tadr + 16u;
    if (!resolveContiguous(setupPayloadAddr, 5u * 16u, setupPayload))
        return false;

    uint64_t setupRegs[4] = {};
    if (!decodeSetupPayload(setupPayload, setupRegs))
        return false;

    uint32_t imageTagDmaAddr = setupPayloadAddr + 5u * 16u;
    DmaTagView imageTagDma{};
    if (!loadDmaTagAt(imageTagDmaAddr, imageTagDma) ||
        imageTagDma.id != 1u ||
        imageTagDma.qwc != 1u ||
        imageTagDma.irq)
    {
        return false;
    }

    const uint8_t *imageGifTag = nullptr;
    if (!resolveContiguous(imageTagDmaAddr + 16u, 16u, imageGifTag))
        return false;

    const uint64_t imageTagLo = loadScalar<uint64_t>(imageGifTag, 0u, 16u, "native gif image tag", imageTagDmaAddr + 16u);
    if (gifTagFlg(imageTagLo) != GIF_FMT_IMAGE && gifTagFlg(imageTagLo) != GIF_FMT_IMAGE2)
        return false;

    const uint32_t imageQwc = gifTagNloop(imageTagLo);
    if (imageQwc == 0u)
        return false;

    const uint64_t imageBytes64 = static_cast<uint64_t>(imageQwc) * 16ull;
    if (imageBytes64 > 0xFFFFFFFFull)
        return false;
    const uint32_t imageBytes = static_cast<uint32_t>(imageBytes64);

    const uint32_t payloadTagAddr = imageTagDmaAddr + 32u;
    DmaTagView payloadTag{};
    if (!loadDmaTagAt(payloadTagAddr, payloadTag) ||
        payloadTag.qwc != imageQwc ||
        payloadTag.irq)
    {
        return false;
    }

    uint32_t imageDataAddr = 0u;
    uint32_t finalTadr = payloadTagAddr;
    uint32_t lastTagUpper = payloadTag.upper;
    if (payloadTag.id == 3u || payloadTag.id == 4u)
    {
        imageDataAddr = payloadTag.addr;
        const uint32_t terminalTagAddr = payloadTagAddr + 16u;
        DmaTagView terminalTag{};
        if (!loadDmaTagAt(terminalTagAddr, terminalTag) ||
            terminalTag.qwc != 0u ||
            terminalTag.irq ||
            (terminalTag.id != 0u && terminalTag.id != 7u))
        {
            return false;
        }
        finalTadr = (terminalTag.id == 0u) ? (terminalTagAddr + 16u) : terminalTagAddr;
        lastTagUpper = terminalTag.upper;
    }
    else if (payloadTag.id == 7u)
    {
        imageDataAddr = payloadTagAddr + 16u;
        finalTadr = payloadTagAddr;
    }
    else
    {
        return false;
    }

    const uint8_t *imageData = nullptr;
    if (!resolveContiguous(imageDataAddr, imageBytes, imageData))
        return false;

    const DmacTransferToken transfer =
        beginDmacTransfer(DmacChannel::Gif);
    m_dmaStartCount.fetch_add(1, std::memory_order_relaxed);
    m_seenGifCopy = true;
    m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
    gs.uploadImageNative(setupRegs[0], setupRegs[1], setupRegs[2], setupRegs[3], imageData, imageBytes);

    m_ioRegisters[GIF_CHANNEL + 0x30u] = finalTadr;
    m_ioRegisters[GIF_CHANNEL + 0x40u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x50u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x00u] =
        ((chcr & 0x0000FFFFu) |
         (lastTagUpper << 16u)) |
        0x100u;
    m_ioRegisters[GIF_CHANNEL + 0x20u] = 0u;
    (void)requestDmacCompletion(transfer);
    return true;
}

bool PS2Memory::tryProcessNativeGifPackedChain(GS &gs, uint32_t tadr, uint32_t chcr)
{
    static constexpr uint32_t GIF_CHANNEL = 0x1000A000u;
    static constexpr uint32_t D_CTRL = 0x1000E000u;

    if (!m_rdram || !m_gsVRAM || m_path3Masked)
        return false;
    if (m_gifArbiter && !m_gifArbiter->empty())
        return false;
    if ((chcr & 0x100u) == 0u || ((chcr >> 2u) & 0x3u) != 1u)
        return false;
    if ((chcr & (1u << 7u)) != 0u || ((chcr >> 4u) & 0x3u) != 0u)
        return false;

    const auto dctrlIt = m_ioRegisters.find(D_CTRL);
    if (dctrlIt != m_ioRegisters.end() && ((dctrlIt->second & 0x1u) == 0u))
        return false;

    auto resolveContiguous = [&](uint32_t guestAddr, uint32_t bytes, const uint8_t *&out) -> bool
    {
        try
        {
            const bool scratch = isScratchpad(guestAddr);
            const uint32_t phys = translateAddress(guestAddr);
            const uint8_t *base = scratch ? m_scratchpad : m_rdram;
            const uint32_t limit = scratch ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
            if (!base || phys > limit || bytes > limit - phys)
                return false;
            out = base + phys;
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    };

    const uint8_t *tagPtr = nullptr;
    if (!resolveContiguous(tadr, 16u, tagPtr))
        return false;

    const DmaTagView tag = decodeDmaTag(loadScalar<uint64_t>(tagPtr, 0u, 16u, "native packed gif dma tag", tadr));
    if (tag.id != 7u || tag.qwc == 0u || tag.irq)
        return false;

    const uint64_t payloadBytes64 = static_cast<uint64_t>(tag.qwc) * 16ull;
    if (payloadBytes64 > 0xFFFFFFFFull)
        return false;
    const uint32_t payloadBytes = static_cast<uint32_t>(payloadBytes64);

    const uint8_t *payload = nullptr;
    if (!resolveContiguous(tadr + 16u, payloadBytes, payload))
        return false;
    if (!gs.processNativePackedGIFPacket(payload, payloadBytes))
        return false;

    const DmacTransferToken transfer =
        beginDmacTransfer(DmacChannel::Gif);
    m_dmaStartCount.fetch_add(1, std::memory_order_relaxed);
    m_seenGifCopy = true;
    m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);

    m_ioRegisters[GIF_CHANNEL + 0x30u] = tadr;
    m_ioRegisters[GIF_CHANNEL + 0x40u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x50u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x00u] =
        ((chcr & 0x0000FFFFu) |
         (tag.upper << 16u)) |
        0x100u;
    m_ioRegisters[GIF_CHANNEL + 0x20u] = 0u;
    (void)requestDmacCompletion(transfer);
    return true;
}

int PS2Memory::pollDmaRegisters()
{
    return 0;
}

uint32_t PS2Memory::readIORegister(uint32_t address)
{
    if (isGsPrivReg(address))
    {
        // NB: unreachable from read8/16/32/64 today, same reasoning as the write
        // path above; kept correct for direct callers.
        const uint32_t off = address & 7u;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            return static_cast<uint32_t>((gs_regs.csr.load() >> (off * 8u)) & 0xFFFFFFFFull);
        }
        if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            return static_cast<uint32_t>((*reg >> (off * 8u)) & 0xFFFFFFFFull);
        }
        return 0u;
    }

    if (address >= 0x10002000 && address <= 0x10002030)
    {
        uint32_t val = 0;
        switch (address)
        {
        case 0x10002000:
            val = m_ioRegisters[address];
            break;
        case 0x10002010:
            val = m_ioRegisters[address] & ~(1u << 31);
            break;
        case 0x10002020:
        case 0x10002030:
            val = m_ioRegisters[address];
            break;
        default:
            val = 0;
            break;
        }
        return val;
    }
    if (address >= 0x10000000 && address < 0x10010000)
    {
        if (address >= 0x10000000 && address < 0x10000100)
        {
            if (isEeTimer0Register(address))
            {
                if (address == kEeTimer0Count)
                {
                    updateEeTimer0Counter();
                }
                auto timerIt = m_ioRegisters.find(address);
                return timerIt != m_ioRegisters.end() ? timerIt->second : 0u;
            }
        }

        if (address >= 0x10000200 && address < 0x10000300)
        {
            return 0;
        }

        if (address >= 0x1000F200 && address <= 0x1000F260)
        {
            if (address == 0x1000F230)
            {
                return 0x60000;
            }
            if (address == 0x1000F240)
            {
                return 0xF0000002;
            }
            return 0;
        }
    }

    auto it = m_ioRegisters.find(address);
    if (it != m_ioRegisters.end())
    {
        return it->second;
    }

    return 0;
}

void PS2Memory::registerCodeRegion(uint32_t start, uint32_t end)
{
    if (end <= start)
    {
        std::cerr << "Ignoring invalid code region: start=0x" << std::hex << start
                  << " end=0x" << end << std::dec << std::endl;
        return;
    }

    if ((end - start) > PS2_RAM_SIZE)
    {
        std::cerr << "Ignoring oversized code region: start=0x" << std::hex << start
                  << " end=0x" << end << std::dec << std::endl;
        return;
    }

    for (const auto &existing : m_codeRegions)
    {
        if (existing.start == start && existing.end == end)
        {
            return;
        }
    }

    CodeRegion region;
    region.start = start;
    region.end = end;

    size_t sizeInWords = (end - start + 3u) / 4u;
    region.modified.resize(sizeInWords, false);

    m_codeRegions.push_back(region);
    RUNTIME_LOG("Registered code region: " << std::hex << start << " - " << end
                                            << std::dec << std::endl);
}

bool PS2Memory::isAddressInRegion(uint32_t address, const CodeRegion &region)
{
    return (address >= region.start && address < region.end);
}

bool PS2Memory::isCodeAddress(uint32_t address) const
{
    for (const auto &region : m_codeRegions)
    {
        if (address >= region.start && address < region.end)
        {
            return true;
        }
    }
    return false;
}

void PS2Memory::markModified(uint32_t address, uint32_t size, uint32_t writerPc)
{
    if (size == 0)
    {
        return;
    }

    const uint64_t writeEnd = static_cast<uint64_t>(address) + static_cast<uint64_t>(size);
    for (auto &region : m_codeRegions)
    {
        const uint64_t regionStart = region.start;
        const uint64_t regionEnd = region.end;
        if (writeEnd <= regionStart || static_cast<uint64_t>(address) >= regionEnd)
        {
            continue;
        }

        const uint32_t overlapStart =
            static_cast<uint32_t>(std::max<uint64_t>(address, regionStart));
        const uint32_t overlapEnd =
            static_cast<uint32_t>(std::min<uint64_t>(writeEnd, regionEnd));
        const size_t firstWord = (overlapStart - region.start) / 4u;
        const size_t lastWord =
            (static_cast<size_t>(overlapEnd - region.start) + 3u) / 4u;

        CodeInvalidationEvent pending{};
        pending.writerPc = writerPc;
        for (size_t bitIndex = firstWord; bitIndex < lastWord; ++bitIndex)
        {
            if (bitIndex >= region.modified.size() || region.modified[bitIndex])
            {
                continue;
            }

            region.modified[bitIndex] = true;
            const uint32_t wordStart =
                region.start + static_cast<uint32_t>(bitIndex * 4u);
            const uint32_t wordEnd = std::min<uint32_t>(wordStart + 4u, region.end);
            if (pending.words != 0u && wordStart != pending.end)
            {
                std::lock_guard<std::mutex> lock(m_codeInvalidationMutex);
                m_pendingCodeInvalidations.push_back(pending);
                pending = {};
                pending.writerPc = writerPc;
            }

            if (pending.words == 0u)
            {
                pending.start = wordStart;
                pending.end = wordEnd;
            }
            else
            {
                pending.end = wordEnd;
            }
            ++pending.words;
        }

        if (pending.words != 0u)
        {
            std::lock_guard<std::mutex> lock(m_codeInvalidationMutex);
            m_pendingCodeInvalidations.push_back(pending);
        }
    }
}

std::vector<PS2Memory::CodeInvalidationEvent> PS2Memory::takeCodeInvalidationEvents()
{
    std::vector<CodeInvalidationEvent> pending;
    {
        std::lock_guard<std::mutex> lock(m_codeInvalidationMutex);
        pending.swap(m_pendingCodeInvalidations);
    }

    std::sort(pending.begin(), pending.end(),
              [](const CodeInvalidationEvent &lhs, const CodeInvalidationEvent &rhs)
              {
                  if (lhs.writerPc != rhs.writerPc)
                  {
                      return lhs.writerPc < rhs.writerPc;
                  }
                  return lhs.start < rhs.start;
              });

    std::vector<CodeInvalidationEvent> merged;
    merged.reserve(pending.size());
    for (const auto &event : pending)
    {
        if (!merged.empty() &&
            merged.back().writerPc == event.writerPc &&
            event.start <= merged.back().end)
        {
            merged.back().end = std::max(merged.back().end, event.end);
            merged.back().words += event.words;
            continue;
        }
        merged.push_back(event);
    }
    return merged;
}

void PS2Memory::flushCodeInvalidationLog()
{
    for (const auto &event : takeCodeInvalidationEvents())
    {
        RUNTIME_LOG("{\"schema_version\":1,\"event\":\"code-invalidated\","
                    << "\"start\":\"0x" << std::hex << std::setw(8)
                    << std::setfill('0') << event.start
                    << "\",\"end\":\"0x" << std::setw(8) << event.end
                    << "\",\"words\":" << std::dec << event.words
                    << ",\"writer_pc\":\"0x" << std::hex << std::setw(8)
                    << event.writerPc << "\"}" << std::dec << std::setfill(' ')
                    << std::endl);
    }
}

bool PS2Memory::isCodeModified(uint32_t address, uint32_t size)
{
    if (size == 0)
    {
        return false;
    }

    const uint64_t writeEnd = static_cast<uint64_t>(address) + static_cast<uint64_t>(size);
    for (const auto &region : m_codeRegions)
    {
        const uint64_t regionStart = region.start;
        const uint64_t regionEnd = region.end;
        if (writeEnd <= regionStart || static_cast<uint64_t>(address) >= regionEnd)
        {
            continue;
        }

        uint32_t overlapStart = static_cast<uint32_t>(std::max<uint64_t>(address, regionStart));
        uint32_t overlapEnd = static_cast<uint32_t>(std::min<uint64_t>(writeEnd, regionEnd));

        for (uint32_t addr = overlapStart; addr < overlapEnd; addr += 4)
        {
            size_t bitIndex = (addr - region.start) / 4;
            if (bitIndex < region.modified.size() && region.modified[bitIndex])
            {
                return true; // Found modified code
            }
        }
    }

    return false; // No modifications found
}

void PS2Memory::clearModifiedFlag(uint32_t address, uint32_t size)
{
    if (size == 0)
    {
        return;
    }

    const uint64_t writeEnd = static_cast<uint64_t>(address) + static_cast<uint64_t>(size);
    for (auto &region : m_codeRegions)
    {
        const uint64_t regionStart = region.start;
        const uint64_t regionEnd = region.end;
        if (writeEnd <= regionStart || static_cast<uint64_t>(address) >= regionEnd)
        {
            continue;
        }

        uint32_t overlapStart = static_cast<uint32_t>(std::max<uint64_t>(address, regionStart));
        uint32_t overlapEnd = static_cast<uint32_t>(std::min<uint64_t>(writeEnd, regionEnd));

        for (uint32_t addr = overlapStart; addr < overlapEnd; addr += 4)
        {
            size_t bitIndex = (addr - region.start) / 4;
            if (bitIndex < region.modified.size())
            {
                region.modified[bitIndex] = false;
            }
        }
    }
}
