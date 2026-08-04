#include "runtime/ps2_memory.h"
#include "runtime/ps2_address.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_vu1.h"
#include "ps2_log.h"
#include <array>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// These opt-in observers are large and cold in normal execution. Keep LTO
// from folding them into the permanent interpreter's instruction-pair loop.
#if defined(_MSC_VER)
#define PS2X_VU_OBSERVER_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define PS2X_VU_OBSERVER_NOINLINE __attribute__((noinline))
#else
#define PS2X_VU_OBSERVER_NOINLINE
#endif

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

    inline void checkEePhysicalBusError(uint32_t physicalAddress)
    {
        if (Ps2IsEeBusErrorPhysicalAddress(physicalAddress))
        {
            throw PS2BusErrorException(physicalAddress);
        }
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
    inline void writeCsrHalf(
        std::atomic<uint64_t> &csr,
        uint32_t off,
        uint32_t value,
        uint32_t writeMask = 0xFFFFFFFFu)
    {
        constexpr uint32_t kW1cMask = 0x3u;
        uint64_t expected = csr.load();
        uint64_t desired;
        do
        {
            if (off == 0u)
            {
                uint32_t oldLow = static_cast<uint32_t>(expected & 0xFFFFFFFFull);
                const uint32_t ordinaryMask = writeMask & ~kW1cMask;
                uint32_t mergedLow =
                    (oldLow & ~ordinaryMask) |
                    (value & ordinaryMask);
                desired = (expected & 0xFFFFFFFF00000000ull) | static_cast<uint64_t>(mergedLow);
                desired &= ~static_cast<uint64_t>(value & writeMask & kW1cMask);
            }
            else
            {
                const uint64_t mask =
                    static_cast<uint64_t>(writeMask) << (off * 8u);
                desired =
                    (expected & ~mask) |
                    ((static_cast<uint64_t>(value) << (off * 8u)) & mask);
            }
        } while (!csr.compare_exchange_weak(expected, desired));
    }

    // Same as writeCsrHalf but for a full 64-bit CSR write (bits 0..1 are still
    // write-one-to-clear against the current value).
    inline void writeCsrFull(
        std::atomic<uint64_t> &csr,
        uint64_t value,
        uint64_t writeMask = 0xFFFFFFFFFFFFFFFFull)
    {
        constexpr uint64_t kW1cMask = 0x3ull;
        uint64_t expected = csr.load();
        uint64_t desired;
        do
        {
            const uint64_t ordinaryMask = writeMask & ~kW1cMask;
            desired =
                (expected & ~ordinaryMask) |
                (value & ordinaryMask);
            desired &= ~(value & writeMask & kW1cMask);
        } while (!csr.compare_exchange_weak(expected, desired));
    }

    constexpr uint32_t kIntcStat = 0x1000F000u;
    constexpr uint32_t kIntcMask = 0x1000F010u;
    constexpr uint32_t kEeCounterIntcMask =
        (1u << 9u) | (1u << 10u) |
        (1u << 11u) | (1u << 12u);

    std::optional<uint32_t> vifRegisterValue(
        const VIFRegisters &regs,
        uint32_t offset)
    {
        switch (offset)
        {
        case 0x00u:
            return regs.stat;
        case 0x20u:
            return regs.err;
        case 0x30u:
            return regs.mark;
        case 0x40u:
            return regs.cycle;
        case 0x50u:
            return regs.mode;
        case 0x60u:
            return regs.num;
        case 0x70u:
            return regs.mask;
        case 0x80u:
            return regs.code;
        case 0x90u:
            return regs.itops;
        case 0xA0u:
            return regs.base;
        case 0xB0u:
            return regs.ofst;
        case 0xC0u:
            return regs.tops;
        case 0xD0u:
            return regs.itop;
        case 0xE0u:
            return regs.top;
        case 0x100u:
        case 0x110u:
        case 0x120u:
        case 0x130u:
            return regs.row[(offset - 0x100u) / 0x10u];
        case 0x140u:
        case 0x150u:
        case 0x160u:
        case 0x170u:
            return regs.col[(offset - 0x140u) / 0x10u];
        default:
            return std::nullopt;
        }
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
    uint64_t start = 0u;
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
    uint64_t vu1DumpCount = 1u;
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

        const char *startText = std::getenv("PS2X_GS_DMA_TRACE_START");
        if (startText && *startText)
        {
            errno = 0;
            char *end = nullptr;
            const unsigned long long parsed = std::strtoull(startText, &end, 0);
            if (errno != 0 || end == startText || *end != '\0')
            {
                std::fprintf(stderr,
                             "PS2 GS DMA trace: invalid PS2X_GS_DMA_TRACE_START '%s'\n",
                             startText);
                return;
            }
            start = static_cast<uint64_t>(parsed);
        }
        if (start >= limit)
        {
            std::fprintf(stderr,
                         "PS2 GS DMA trace: start must be below limit\n");
            return;
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
            !parseOptionalIndex("PS2X_VU1_STEP_ORDINAL", vu1StepOrdinal) ||
            !parseOptionalIndex("PS2X_VU1_DUMP_COUNT", vu1DumpCount) ||
            vu1DumpCount == 0u)
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
                     "\"producer\":\"PS2Recomp\",\"start\":%" PRIu64 ","
                     "\"limit\":%" PRIu64 ","
                     "\"ee_write_ranges\":\"%s\","
                     "\"watch_addresses\":\"%s\"}\n",
                     start, limit, writeRangesText ? writeRangesText : "",
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

struct Ps2Vu1WorkloadProfileState
{
    static constexpr uint64_t FNV_OFFSET_BASIS =
        14695981039346656037ull;
    static constexpr uint64_t FNV_PRIME = 1099511628211ull;

    struct ProgramKey
    {
        uint64_t epoch = 0u;
        uint64_t microHash = FNV_OFFSET_BASIS;
        uint32_t startPc = 0u;

        bool operator<(const ProgramKey &other) const
        {
            return std::tie(epoch, microHash, startPc) <
                   std::tie(other.epoch, other.microHash, other.startPc);
        }
    };

    struct ProgramStats
    {
        uint64_t invocations = 0u;
        uint64_t completedInvocations = 0u;
        uint64_t abortedInvocations = 0u;
        uint64_t pairs = 0u;
        uint64_t minPairs = std::numeric_limits<uint64_t>::max();
        uint64_t maxPairs = 0u;
        uint64_t measuredPairs = 0u;
        uint64_t measuredInvocationsTouched = 0u;
        uint64_t fullyMeasuredInvocations = 0u;
        uint64_t fullyMeasuredInvocationPairs = 0u;
        uint64_t branchPairs = 0u;
        uint64_t conditionalBranchPairs = 0u;
        uint64_t nonsequentialTransitions = 0u;
        std::array<uint64_t, 128u> lowerOps{};
        std::array<uint64_t, 128u> lowerSpecialOps{};
        std::array<uint64_t, 64u> upperOps{};
        std::array<uint64_t, 128u> upperSpecialOps{};
        std::set<uint64_t> codeGenerations;
        std::set<uint64_t> measuredCodeGenerations;
    };

    struct EpochStats
    {
        uint64_t invocations = 0u;
        uint64_t completedInvocations = 0u;
        uint64_t abortedInvocations = 0u;
        uint64_t pairs = 0u;
        uint64_t measuredPairs = 0u;
    };

    struct UploadKey
    {
        uint64_t payloadHash = FNV_OFFSET_BASIS;
        uint32_t destination = 0u;
        uint32_t sizeBytes = 0u;

        bool operator<(const UploadKey &other) const
        {
            return std::tie(payloadHash, destination, sizeBytes) <
                   std::tie(
                       other.payloadHash,
                       other.destination,
                       other.sizeBytes);
        }
    };

    struct UploadStats
    {
        uint64_t count = 0u;
        uint64_t identicalCount = 0u;
        uint64_t measuredCount = 0u;
        uint64_t measuredIdenticalCount = 0u;
    };

    struct CurrentInvocation
    {
        ProgramStats *program = nullptr;
        EpochStats *epoch = nullptr;
        uint64_t microHash = FNV_OFFSET_BASIS;
        uint64_t codeGeneration = 0u;
        uint64_t pairs = 0u;
        uint64_t measuredPairs = 0u;
        bool active = false;
        bool fullyMeasured = false;
        bool lastPairMeasured = false;
    };

    std::FILE *output = nullptr;
    uint64_t warmupPairs = 0u;
    uint64_t pairLimit = std::numeric_limits<uint64_t>::max();
    uint64_t epoch = 0u;
    std::atomic<bool> enabled{false};
    bool written = false;
    std::atomic<uint64_t> observedPairs{0u};
    std::atomic<uint64_t> measuredPairs{0u};
    std::atomic<uint64_t> invocations{0u};
    std::atomic<uint64_t> codeUploads{0u};
    std::atomic<uint64_t> identicalCodeUploads{0u};
    std::atomic<uint64_t> codeUploadBytes{0u};
    std::atomic<uint64_t> identicalCodeUploadBytes{0u};
    std::map<ProgramKey, ProgramStats> programs;
    std::map<uint64_t, EpochStats> epochs;
    std::map<UploadKey, UploadStats> uploadPayloads;
    std::unordered_map<uint64_t, uint64_t> generationHashes;
    std::set<uint64_t> observedCodeGenerations;
    std::set<uint64_t> observedMicroHashes;
    std::set<uint64_t> measuredCodeGenerations;
    std::set<uint64_t> measuredMicroHashes;
    CurrentInvocation current{};

    Ps2Vu1WorkloadProfileState(
        const char *path,
        uint64_t configuredWarmupPairs,
        uint64_t configuredPairLimit)
        : warmupPairs(configuredWarmupPairs),
          pairLimit(
              configuredPairLimit == 0u
                  ? std::numeric_limits<uint64_t>::max()
                  : configuredPairLimit)
    {
        output = path && *path ? std::fopen(path, "wb") : nullptr;
        if (!output)
        {
            if (path && *path)
            {
                std::fprintf(
                    stderr,
                    "VU1 workload profile: cannot open '%s': %s\n",
                    path, std::strerror(errno));
            }
            return;
        }
        enabled.store(true, std::memory_order_relaxed);
    }

    ~Ps2Vu1WorkloadProfileState()
    {
        write();
        if (output)
            std::fclose(output);
    }

    static uint64_t hashBytes(const uint8_t *data, uint32_t sizeBytes)
    {
        uint64_t hash = FNV_OFFSET_BASIS;
        if (!data)
            return hash;
        for (uint32_t index = 0u; index < sizeBytes; ++index)
        {
            hash ^= data[index];
            hash *= FNV_PRIME;
        }
        return hash;
    }

    uint64_t microHash(
        const uint8_t *code,
        uint32_t codeSize,
        uint64_t generation)
    {
        const auto found = generationHashes.find(generation);
        if (found != generationHashes.end())
            return found->second;
        const uint64_t hash = hashBytes(code, codeSize);
        generationHashes.emplace(generation, hash);
        return hash;
    }

    void beginInvocation(
        uint32_t startPc,
        const uint8_t *code,
        uint32_t codeSize,
        uint64_t generation)
    {
        if (!enabled.load(std::memory_order_relaxed))
            return;
        if (current.active)
            endInvocation(false);

        const uint64_t hash = microHash(code, codeSize, generation);
        ProgramStats &program = programs[ProgramKey{
            epoch, hash, startPc & 0x3FFFu}];
        EpochStats &epochStats = epochs[epoch];
        ++program.invocations;
        ++epochStats.invocations;
        program.codeGenerations.insert(generation);
        observedCodeGenerations.insert(generation);
        observedMicroHashes.insert(hash);
        invocations.fetch_add(1u, std::memory_order_relaxed);

        const uint64_t observed =
            observedPairs.load(std::memory_order_relaxed);
        const uint64_t measured =
            measuredPairs.load(std::memory_order_relaxed);
        current = CurrentInvocation{
            .program = &program,
            .epoch = &epochStats,
            .microHash = hash,
            .codeGeneration = generation,
            .active = true,
            .fullyMeasured =
                observed >= warmupPairs && measured < pairLimit,
        };
    }

    void recordInstruction(
        uint32_t lower,
        uint32_t upper)
    {
        if (!enabled.load(std::memory_order_relaxed) || !current.active)
            return;

        ++current.pairs;
        const uint64_t observed =
            observedPairs.fetch_add(1u, std::memory_order_relaxed) + 1u;
        const uint64_t measured =
            measuredPairs.load(std::memory_order_relaxed);
        const bool capture =
            observed > warmupPairs && measured < pairLimit;
        current.lastPairMeasured = capture;
        if (!capture)
        {
            current.fullyMeasured = false;
            return;
        }

        measuredPairs.fetch_add(1u, std::memory_order_relaxed);
        ++current.measuredPairs;
        ProgramStats &program = *current.program;
        if (current.measuredPairs == 1u)
        {
            ++program.measuredInvocationsTouched;
            program.measuredCodeGenerations.insert(
                current.codeGeneration);
            measuredCodeGenerations.insert(
                current.codeGeneration);
            measuredMicroHashes.insert(current.microHash);
        }
        ++program.measuredPairs;
        ++current.epoch->measuredPairs;

        const uint8_t upperOp = static_cast<uint8_t>(upper & 0x3Fu);
        ++program.upperOps[upperOp];
        if (upperOp >= 0x3Cu)
        {
            const uint8_t special =
                static_cast<uint8_t>(
                    (upper & 0x3u) | ((upper >> 4u) & 0x7Cu));
            ++program.upperSpecialOps[special];
        }

        if ((upper & 0x80000000u) != 0u)
            return;

        const uint8_t lowerOp =
            static_cast<uint8_t>((lower >> 25u) & 0x7Fu);
        ++program.lowerOps[lowerOp];
        if (lowerOp == 0x40u)
        {
            const uint8_t function =
                static_cast<uint8_t>(lower & 0x3Fu);
            if (function >= 0x3Cu)
            {
                const uint8_t special =
                    static_cast<uint8_t>(
                        (lower & 0x3u) |
                        ((lower >> 4u) & 0x7Cu));
                ++program.lowerSpecialOps[special];
            }
        }

        switch (lowerOp)
        {
        case 0x20u: // B
        case 0x21u: // BAL
        case 0x24u: // JR
        case 0x25u: // JALR
            ++program.branchPairs;
            break;
        case 0x28u: // IBEQ
        case 0x29u: // IBNE
        case 0x2Cu: // IBLTZ
        case 0x2Du: // IBGTZ
        case 0x2Eu: // IBLEZ
        case 0x2Fu: // IBGEZ
            ++program.branchPairs;
            ++program.conditionalBranchPairs;
            break;
        default:
            break;
        }
    }

    void recordTransition(uint32_t pc, uint32_t nextPc)
    {
        if (!enabled.load(std::memory_order_relaxed) || !current.active ||
            !current.lastPairMeasured)
        {
            return;
        }
        const uint32_t sequential = (pc + 8u) & 0x3FFFu;
        if ((nextPc & 0x3FFFu) != sequential)
            ++current.program->nonsequentialTransitions;
        if (pairLimit != std::numeric_limits<uint64_t>::max() &&
            measuredPairs.load(std::memory_order_relaxed) >= pairLimit)
        {
            // ps2EntryRunner terminates with std::_Exit after its explicit
            // shutdown sequence, so process/static destructors are not a
            // reliable finalization point. A bounded profile is complete at
            // this exact architectural pair boundary and can be committed
            // immediately.
            write();
        }
    }

    void endInvocation(bool completed)
    {
        if (!current.active)
            return;

        ProgramStats &program = *current.program;
        EpochStats &epochStats = *current.epoch;
        program.pairs += current.pairs;
        program.minPairs = std::min(program.minPairs, current.pairs);
        program.maxPairs = std::max(program.maxPairs, current.pairs);
        epochStats.pairs += current.pairs;
        if (completed)
        {
            ++program.completedInvocations;
            ++epochStats.completedInvocations;
        }
        else
        {
            ++program.abortedInvocations;
            ++epochStats.abortedInvocations;
        }
        if (completed && current.measuredPairs != 0u &&
            current.fullyMeasured)
        {
            ++program.fullyMeasuredInvocations;
            program.fullyMeasuredInvocationPairs += current.measuredPairs;
        }
        current = {};
    }

    void resetEpoch()
    {
        if (!enabled.load(std::memory_order_relaxed))
            return;
        if (current.active)
            endInvocation(false);
        ++epoch;
    }

    void recordCodeUpload(
        uint32_t destination,
        const uint8_t *payload,
        uint32_t sizeBytes,
        bool identical)
    {
        if (!enabled.load(std::memory_order_relaxed))
            return;

        codeUploads.fetch_add(1u, std::memory_order_relaxed);
        codeUploadBytes.fetch_add(sizeBytes, std::memory_order_relaxed);
        if (identical)
        {
            identicalCodeUploads.fetch_add(1u, std::memory_order_relaxed);
            identicalCodeUploadBytes.fetch_add(
                sizeBytes, std::memory_order_relaxed);
        }

        UploadStats &stats = uploadPayloads[UploadKey{
            hashBytes(payload, sizeBytes), destination, sizeBytes}];
        ++stats.count;
        if (identical)
            ++stats.identicalCount;

        const uint64_t observed =
            observedPairs.load(std::memory_order_relaxed);
        const uint64_t measured =
            measuredPairs.load(std::memory_order_relaxed);
        if (observed >= warmupPairs && measured < pairLimit)
        {
            ++stats.measuredCount;
            if (identical)
                ++stats.measuredIdenticalCount;
        }
    }

    template <size_t Size>
    static void writeSparseCounts(
        std::FILE *file,
        const std::array<uint64_t, Size> &counts)
    {
        bool first = true;
        for (size_t index = 0u; index < counts.size(); ++index)
        {
            if (counts[index] == 0u)
                continue;
            std::fprintf(
                file, "%s\"0x%02zx\":%" PRIu64,
                first ? "" : ",", index, counts[index]);
            first = false;
        }
    }

    void write()
    {
        if (written || !output)
            return;
        written = true;
        if (current.active)
            endInvocation(false);

        const uint64_t observed =
            observedPairs.load(std::memory_order_relaxed);
        const uint64_t measured =
            measuredPairs.load(std::memory_order_relaxed);
        const bool bounded =
            pairLimit != std::numeric_limits<uint64_t>::max();
        uint64_t measuredUploads = 0u;
        uint64_t measuredIdenticalUploads = 0u;
        for (const auto &[key, stats] : uploadPayloads)
        {
            (void)key;
            measuredUploads += stats.measuredCount;
            measuredIdenticalUploads +=
                stats.measuredIdenticalCount;
        }
        std::fprintf(
            output,
            "{\"schema_version\":1,\"producer\":\"PS2Recomp\","
            "\"warmup_pairs\":%" PRIu64 ",\"pair_limit\":",
            warmupPairs);
        if (bounded)
            std::fprintf(output, "%" PRIu64, pairLimit);
        else
            std::fputs("null", output);
        std::fprintf(
            output,
            ",\"observed_pairs\":%" PRIu64 ","
            "\"measured_pairs\":%" PRIu64 ","
            "\"measurement_complete\":%s,"
            "\"invocations\":%" PRIu64 ","
            "\"reset_epochs\":%" PRIu64 ","
            "\"code_generations_observed\":%zu,"
            "\"microcode_hashes_observed\":%zu,"
            "\"measured_code_generations\":%zu,"
            "\"measured_microcode_hashes\":%zu,"
            "\"uploads\":{\"count\":%" PRIu64 ","
            "\"identical_count\":%" PRIu64 ","
            "\"measured_count\":%" PRIu64 ","
            "\"measured_identical_count\":%" PRIu64 ","
            "\"bytes\":%" PRIu64 ","
            "\"identical_bytes\":%" PRIu64 ","
            "\"payloads\":[",
            observed, measured,
            bounded && measured >= pairLimit ? "true" : "false",
            invocations.load(std::memory_order_relaxed),
            epoch + 1u,
            observedCodeGenerations.size(),
            observedMicroHashes.size(),
            measuredCodeGenerations.size(),
            measuredMicroHashes.size(),
            codeUploads.load(std::memory_order_relaxed),
            identicalCodeUploads.load(std::memory_order_relaxed),
            measuredUploads,
            measuredIdenticalUploads,
            codeUploadBytes.load(std::memory_order_relaxed),
            identicalCodeUploadBytes.load(std::memory_order_relaxed));

        bool first = true;
        for (const auto &[key, stats] : uploadPayloads)
        {
            std::fprintf(
                output,
                "%s{\"payload_hash_fnv1a64\":\"0x%016" PRIx64 "\","
                "\"destination\":\"0x%04" PRIx32 "\","
                "\"bytes\":%" PRIu32 ",\"count\":%" PRIu64 ","
                "\"identical_count\":%" PRIu64 ","
                "\"measured_count\":%" PRIu64 ","
                "\"measured_identical_count\":%" PRIu64 "}",
                first ? "" : ",",
                key.payloadHash, key.destination, key.sizeBytes,
                stats.count, stats.identicalCount,
                stats.measuredCount, stats.measuredIdenticalCount);
            first = false;
        }
        std::fputs("]},\"epochs\":[", output);

        first = true;
        for (const auto &[epochIndex, stats] : epochs)
        {
            std::fprintf(
                output,
                "%s{\"epoch\":%" PRIu64 ","
                "\"invocations\":%" PRIu64 ","
                "\"completed_invocations\":%" PRIu64 ","
                "\"aborted_invocations\":%" PRIu64 ","
                "\"pairs\":%" PRIu64 ","
                "\"measured_pairs\":%" PRIu64 "}",
                first ? "" : ",", epochIndex,
                stats.invocations, stats.completedInvocations,
                stats.abortedInvocations, stats.pairs,
                stats.measuredPairs);
            first = false;
        }
        std::fputs("],\"programs\":[", output);

        first = true;
        for (const auto &[key, stats] : programs)
        {
            const double averagePairs =
                stats.invocations == 0u
                    ? 0.0
                    : static_cast<double>(stats.pairs) /
                          static_cast<double>(stats.invocations);
            const double fullyMeasuredAverage =
                stats.fullyMeasuredInvocations == 0u
                    ? 0.0
                    : static_cast<double>(
                          stats.fullyMeasuredInvocationPairs) /
                          static_cast<double>(
                              stats.fullyMeasuredInvocations);
            std::fprintf(
                output,
                "%s{\"epoch\":%" PRIu64 ","
                "\"micro_hash_fnv1a64\":\"0x%016" PRIx64 "\","
                "\"start_pc\":\"0x%04" PRIx32 "\","
                "\"invocations\":%" PRIu64 ","
                "\"completed_invocations\":%" PRIu64 ","
                "\"aborted_invocations\":%" PRIu64 ","
                "\"pairs\":%" PRIu64 ","
                "\"average_pairs\":%.6f,"
                "\"min_pairs\":%" PRIu64 ","
                "\"max_pairs\":%" PRIu64 ","
                "\"code_generations\":%zu,"
                "\"measured_code_generations\":%zu,"
                "\"measured_pairs\":%" PRIu64 ","
                "\"measured_invocations_touched\":%" PRIu64 ","
                "\"fully_measured_invocations\":%" PRIu64 ","
                "\"fully_measured_average_pairs\":%.6f,"
                "\"branch_pairs\":%" PRIu64 ","
                "\"conditional_branch_pairs\":%" PRIu64 ","
                "\"nonsequential_transitions\":%" PRIu64 ","
                "\"lower_ops\":{",
                first ? "" : ",", key.epoch, key.microHash,
                key.startPc, stats.invocations,
                stats.completedInvocations,
                stats.abortedInvocations, stats.pairs,
                averagePairs,
                stats.minPairs ==
                        std::numeric_limits<uint64_t>::max()
                    ? 0u
                    : stats.minPairs,
                stats.maxPairs, stats.codeGenerations.size(),
                stats.measuredCodeGenerations.size(),
                stats.measuredPairs,
                stats.measuredInvocationsTouched,
                stats.fullyMeasuredInvocations,
                fullyMeasuredAverage,
                stats.branchPairs,
                stats.conditionalBranchPairs,
                stats.nonsequentialTransitions);
            writeSparseCounts(output, stats.lowerOps);
            std::fputs("},\"lower_special_ops\":{", output);
            writeSparseCounts(output, stats.lowerSpecialOps);
            std::fputs("},\"upper_ops\":{", output);
            writeSparseCounts(output, stats.upperOps);
            std::fputs("},\"upper_special_ops\":{", output);
            writeSparseCounts(output, stats.upperSpecialOps);
            std::fputs("}}", output);
            first = false;
        }
        std::fputs("]}\n", output);
        std::fflush(output);
        enabled.store(false, std::memory_order_relaxed);
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

    const char *profilePath = std::getenv("PS2X_VU1_PROFILE");
    if (!profilePath || !*profilePath)
        return;

    auto parsePairCount = [](
                              const char *name,
                              uint64_t defaultValue,
                              uint64_t &value)
    {
        const char *text = std::getenv(name);
        if (!text || !*text)
        {
            value = defaultValue;
            return true;
        }
        errno = 0;
        char *end = nullptr;
        const unsigned long long parsed =
            std::strtoull(text, &end, 0);
        if (errno != 0 || end == text || *end != '\0')
        {
            std::fprintf(
                stderr,
                "VU1 workload profile: invalid %s '%s'\n",
                name, text);
            return false;
        }
        value = static_cast<uint64_t>(parsed);
        return true;
    };

    uint64_t warmupPairs = 0u;
    uint64_t pairLimit = 0u;
    if (parsePairCount(
            "PS2X_VU1_PROFILE_WARMUP_PAIRS",
            0u, warmupPairs) &&
        parsePairCount(
            "PS2X_VU1_PROFILE_PAIRS",
            0u, pairLimit))
    {
        (void)configureVu1WorkloadProfile(
            profilePath, warmupPairs, pairLimit);
    }
}

PS2Memory::~PS2Memory()
{
    flushCodeInvalidationLog();
    finishVif1DmaTrace();
    delete m_gsDmaTrace;
    m_gsDmaTrace = nullptr;
    finishVu1WorkloadProfile();
    delete m_vu1WorkloadProfile;
    m_vu1WorkloadProfile = nullptr;

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

bool PS2Memory::configureVu1WorkloadProfile(
    const char *outputPath,
    uint64_t warmupPairs,
    uint64_t pairLimit)
{
    finishVu1WorkloadProfile();
    delete m_vu1WorkloadProfile;
    m_vu1WorkloadProfile = new Ps2Vu1WorkloadProfileState(
        outputPath, warmupPairs, pairLimit);
    return m_vu1WorkloadProfile->enabled.load(
        std::memory_order_relaxed);
}

void PS2Memory::finishVu1WorkloadProfile()
{
    if (m_vu1WorkloadProfile)
        m_vu1WorkloadProfile->write();
}

Vu1WorkloadProfileSnapshot
PS2Memory::vu1WorkloadProfileSnapshot() const
{
    Vu1WorkloadProfileSnapshot snapshot{};
    if (!m_vu1WorkloadProfile)
        return snapshot;

    const Ps2Vu1WorkloadProfileState &profile =
        *m_vu1WorkloadProfile;
    snapshot.enabled =
        profile.enabled.load(std::memory_order_relaxed);
    snapshot.warmupPairs = profile.warmupPairs;
    snapshot.pairLimit =
        profile.pairLimit ==
                std::numeric_limits<uint64_t>::max()
            ? 0u
            : profile.pairLimit;
    snapshot.observedPairs =
        profile.observedPairs.load(std::memory_order_relaxed);
    snapshot.measuredPairs =
        profile.measuredPairs.load(std::memory_order_relaxed);
    snapshot.measurementComplete =
        snapshot.pairLimit != 0u &&
        snapshot.measuredPairs >= snapshot.pairLimit;
    snapshot.invocations =
        profile.invocations.load(std::memory_order_relaxed);
    snapshot.codeUploads =
        profile.codeUploads.load(std::memory_order_relaxed);
    snapshot.identicalCodeUploads =
        profile.identicalCodeUploads.load(
            std::memory_order_relaxed);
    snapshot.codeUploadBytes =
        profile.codeUploadBytes.load(std::memory_order_relaxed);
    snapshot.identicalCodeUploadBytes =
        profile.identicalCodeUploadBytes.load(
            std::memory_order_relaxed);
    return snapshot;
}

bool PS2Memory::isVu1WorkloadProfileEnabled() const
{
    return m_vu1WorkloadProfile &&
           m_vu1WorkloadProfile->enabled.load(
               std::memory_order_relaxed);
}

PS2X_VU_OBSERVER_NOINLINE
void PS2Memory::beginVu1WorkloadProfileInvocation(
    uint32_t startPc)
{
    if (!isVu1WorkloadProfileEnabled() || !m_vu1Code)
        return;
    m_vu1WorkloadProfile->beginInvocation(
        startPc, m_vu1Code, PS2_VU1_CODE_SIZE,
        getVU1CodeGeneration());
}

PS2X_VU_OBSERVER_NOINLINE
void PS2Memory::recordVu1WorkloadProfileInstruction(
    uint32_t,
    uint32_t lower,
    uint32_t upper)
{
    if (isVu1WorkloadProfileEnabled())
        m_vu1WorkloadProfile->recordInstruction(lower, upper);
}

PS2X_VU_OBSERVER_NOINLINE
void PS2Memory::recordVu1WorkloadProfileTransition(
    uint32_t pc,
    uint32_t nextPc)
{
    if (isVu1WorkloadProfileEnabled())
        m_vu1WorkloadProfile->recordTransition(pc, nextPc);
}

PS2X_VU_OBSERVER_NOINLINE
void PS2Memory::endVu1WorkloadProfileInvocation(
    bool completed)
{
    if (isVu1WorkloadProfileEnabled())
        m_vu1WorkloadProfile->endInvocation(completed);
}

void PS2Memory::resetVu1WorkloadProfileEpoch()
{
    if (isVu1WorkloadProfileEnabled())
        m_vu1WorkloadProfile->resetEpoch();
}

void PS2Memory::recordVu1WorkloadProfileCodeUpload(
    uint32_t destination,
    const uint8_t *payload,
    uint32_t sizeBytes,
    bool identical,
    uint64_t,
    uint64_t)
{
    if (isVu1WorkloadProfileEnabled())
    {
        m_vu1WorkloadProfile->recordCodeUpload(
            destination, payload, sizeBytes, identical);
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
    if (!trace.enabled)
    {
        if (g_ps2GsDmaTraceState == &trace)
            g_ps2GsDmaWriteTraceEnabled.store(false, std::memory_order_relaxed);
        return;
    }

    const uint64_t sequence = trace.sequence++;
    if (sequence < trace.start || sequence >= trace.limit)
    {
        if (g_ps2GsDmaTraceState == &trace)
            g_ps2GsDmaWriteTraceEnabled.store(false, std::memory_order_relaxed);
        return;
    }

    trace.active = true;
    trace.activeSequence = sequence;
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
                                   const VuExecutionState &state)
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

    const bool selectedDumpRun =
        m_gsDmaTrace->activeSequence == m_gsDmaTrace->vu1StepSequence &&
        ordinal >= m_gsDmaTrace->vu1StepOrdinal &&
        ordinal - m_gsDmaTrace->vu1StepOrdinal <
            m_gsDmaTrace->vu1DumpCount;
    if (!resume && selectedDumpRun &&
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
    // Retain the same bounded run window as the fixture dumps. This keeps the
    // command/state prefix for each adjacent setup invocation available in a
    // single deterministic capture instead of retaining only the first one.
    const bool selectedRun = selectedDumpRun;
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

PS2X_VU_OBSERVER_NOINLINE
void PS2Memory::traceVu1Instruction(
    uint32_t pc, uint32_t lower, uint32_t upper,
    const VuExecutionState &state)
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

PS2X_VU_OBSERVER_NOINLINE
void PS2Memory::traceVu1InvocationEnd(
    uint32_t finalPc, bool ended, bool hitCycleLimit,
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

#undef PS2X_VU_OBSERVER_NOINLINE

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
    resetTlbTranslationCacheStats();
    {
        std::lock_guard<std::mutex> lock(m_dmacMutex);
        m_dmacChannels = {};
        m_readyDmacCompletions.clear();
        m_pendingDmacInterrupts.clear();
        m_dmacReadySequence = 0u;
        m_dmacPublicationSequence = 0u;
    }
    m_vif0Dma = {};
    m_vif1Dma = {};
    m_gifDma = {};
    m_fromIpuDma = {};
    m_toIpuDma = {};
    m_ipuOutputFifo = {};
    m_ipuOutputFifoReadIndex = 0u;
    m_ipuOutputFifoWriteIndex = 0u;
    m_ipuOutputFifoQwc = 0u;
    m_ipuInputFifo = {};
    m_ipuInputFifoReadIndex = 0u;
    m_ipuInputFifoWriteIndex = 0u;
    m_ipuInputFifoQwc = 0u;
    m_scratchpadDma = {};
    m_codeRegions.clear();
    m_path3Masked = false;
    m_gifModePath3Masked = false;
    m_path3MaskedFifo.clear();
    m_vif0WaitingForVu = false;
    m_vif0DeferredData.clear();
    m_vif1PendingPath2DirectQwc = 0u;
    m_vif1PendingPath2DirectHl = false;
    m_vif1PendingIrqAfterCommand = false;
    m_vif1WaitingForVu = false;
    m_vif1DeferredData.clear();
    m_eeCounters.reset();

    try
    {
        // Allocate main RAM
        m_rdram = new uint8_t[ramSize];
        std::memset(m_rdram, 0, ramSize);

        // Allocate scratchpad
        m_scratchpad = new uint8_t[PS2_SCRATCHPAD_SIZE];
        std::memset(m_scratchpad, 0, PS2_SCRATCHPAD_SIZE);
        ps2SetScratchpadHostPtr(m_scratchpad);

        // Initialize EE TLB entries (R5900 has 48 entries). PS2Runtime
        // configures a post-BIOS profile that survives host memory resets;
        // standalone PS2Memory instances retain an empty architectural TLB.
        if (m_postBiosTlbStateConfigured)
        {
            installPostBiosTlbState();
        }
        else
        {
            m_tlbEntries.assign(48, EeTlbEntry{});
            advanceTlbMappingGeneration();
        }

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

uint64_t PS2Memory::currentEeCounterCycle() const noexcept
{
    if (m_eeCounterCycleCallback)
    {
        return m_eeCounterCycleCallback();
    }
    return m_eeCounters.snapshot().currentCycle;
}

uint32_t PS2Memory::intcStatus() const noexcept
{
    const auto it = m_ioRegisters.find(kIntcStat);
    return it != m_ioRegisters.end() ? it->second : 0u;
}

uint32_t PS2Memory::intcMask() const noexcept
{
    const auto it = m_ioRegisters.find(kIntcMask);
    return it != m_ioRegisters.end() ? it->second : 0u;
}

void PS2Memory::setIntcInterruptMask(
    uint32_t cause, bool enabled)
{
    if (cause >= 32u)
    {
        return;
    }

    const uint32_t bit = 1u << cause;
    uint32_t &mask = m_ioRegisters[kIntcMask];
    if (enabled)
    {
        mask |= bit;
    }
    else
    {
        mask &= ~bit;
    }
    publishEeCounterAdvance({}, true);
}

void PS2Memory::publishEeCounterAdvance(
    const ps2x::timing::EeCounterAdvanceResult &result,
    bool interruptStateChanged)
{
    if (result.newlyRaisedInterruptMask != 0u)
    {
        m_ioRegisters[kIntcStat] =
            intcStatus() |
            result.newlyRaisedInterruptMask;
        interruptStateChanged = true;
    }

    if (m_eeCounterScheduleCallback)
    {
        m_eeCounterScheduleCallback(
            m_eeCounters.nextEventCycle());
    }
    if (interruptStateChanged &&
        m_eeCounterInterruptStateCallback)
    {
        m_eeCounterInterruptStateCallback(
            intcStatus(),
            intcMask(),
            result.newlyRaisedInterruptMask);
    }
}

void PS2Memory::resetEeCounters(uint64_t currentCycle)
{
    m_eeCounters.reset(currentCycle);
    m_ioRegisters[kIntcStat] =
        intcStatus() & ~kEeCounterIntcMask;
    publishEeCounterAdvance({}, true);
}

void PS2Memory::configureEeCounterVideoTiming(
    ps2x::timing::EeCounterVideoTiming timing,
    uint64_t currentCycle)
{
    const ps2x::timing::EeCounterAdvanceResult result =
        m_eeCounters.configureVideoTiming(
            timing, currentCycle);
    publishEeCounterAdvance(result);
}

ps2x::timing::EeCounterAdvanceResult
PS2Memory::synchronizeEeCounters(uint64_t currentCycle)
{
    const ps2x::timing::EeCounterAdvanceResult result =
        m_eeCounters.advanceTo(currentCycle);
    publishEeCounterAdvance(result);
    return result;
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

uint32_t PS2Memory::translateAddress(
    uint32_t virtualAddress,
    EeAddressTranslationContext translation,
    bool writeAccess,
    uint32_t accessSize)
{
    return resolveAddress(
               virtualAddress,
               translation,
               writeAccess,
               accessSize)
        .physicalAddress;
}

PS2Memory::EeTranslatedAddress PS2Memory::resolveAddress(
    uint32_t virtualAddress,
    EeAddressTranslationContext translation,
    bool writeAccess,
    uint32_t accessSize)
{
    return translateAddressImpl(
        virtualAddress,
        translation,
        writeAccess,
        accessSize);
}

uint32_t PS2Memory::translateAddressUncached(
    uint32_t virtualAddress,
    EeAddressTranslationContext translation,
    bool writeAccess,
    uint32_t accessSize)
{
    return resolveAddressUncached(
               virtualAddress,
               translation,
               writeAccess,
               accessSize)
        .physicalAddress;
}

PS2Memory::EeTranslatedAddress
PS2Memory::resolveAddressUncached(
    uint32_t virtualAddress,
    EeAddressTranslationContext translation,
    bool writeAccess,
    uint32_t accessSize)
{
    return translateAddressImpl(
        virtualAddress,
        translation,
        writeAccess,
        accessSize,
        false);
}

void PS2Memory::clearTlbTranslationCache() noexcept
{
    m_tlbTranslationCache = {};
    m_tlbTranslationCacheNextWay = {};
}

void PS2Memory::advanceTlbMappingGeneration() noexcept
{
    if (m_tlbMappingGeneration ==
        std::numeric_limits<uint64_t>::max())
    {
        // No stale entry may become current after generation wrap.
        clearTlbTranslationCache();
        m_tlbMappingGeneration = 1u;
        return;
    }
    ++m_tlbMappingGeneration;
}

void PS2Memory::recordTlbTranslationSlowPathFault() noexcept
{
    if (m_tlbTranslationCacheEnabled &&
        m_tlbTranslationCacheDiagnosticsEnabled)
    {
        ++m_tlbTranslationCacheStats.slowPathFaults;
    }
}

std::optional<PS2Memory::EeTranslatedAddress>
PS2Memory::lookupTlbTranslationCache(
    uint32_t virtualAddress,
    EeAddressTranslationContext translation,
    bool writeAccess,
    uint32_t accessSize)
{
    if (!m_tlbTranslationCacheEnabled)
    {
        return std::nullopt;
    }

    constexpr uint32_t pageMask =
        kTlbTranslationCachePageSize - 1u;
    const uint32_t pageOffset = virtualAddress & pageMask;
    const bool supportedAccessSize =
        accessSize != 0u &&
        accessSize <= sizeof(__m128i) &&
        (accessSize & (accessSize - 1u)) == 0u;
    if (!supportedAccessSize)
    {
        if (m_tlbTranslationCacheDiagnosticsEnabled)
        {
            ++m_tlbTranslationCacheStats.misses;
        }
        return std::nullopt;
    }
    if (accessSize >
        kTlbTranslationCachePageSize - pageOffset)
    {
        if (m_tlbTranslationCacheDiagnosticsEnabled)
        {
            ++m_tlbTranslationCacheStats.intervalCrossings;
            ++m_tlbTranslationCacheStats.misses;
        }
        return std::nullopt;
    }
    if ((virtualAddress & (accessSize - 1u)) != 0u)
    {
        if (m_tlbTranslationCacheDiagnosticsEnabled)
        {
            ++m_tlbTranslationCacheStats.misses;
        }
        return std::nullopt;
    }

    const uint32_t virtualPageBase =
        virtualAddress & ~pageMask;
    const size_t setIndex =
        (virtualAddress >> 12u) &
        (kTlbTranslationCacheSetCount - 1u);
    bool generationMiss = false;
    bool permissionMiss = false;
    for (const EeTlbTranslationCacheEntry &entry :
         m_tlbTranslationCache[setIndex])
    {
        if (!entry.valid ||
            entry.virtualPageBase != virtualPageBase ||
            entry.mode != translation.mode ||
            entry.enforceOperatingMode !=
                translation.enforceOperatingMode ||
            entry.errorLevel != translation.errorLevel ||
            (!entry.global && entry.asid != translation.asid))
        {
            continue;
        }
        if (entry.generation != m_tlbMappingGeneration)
        {
            generationMiss = true;
            continue;
        }
        if (writeAccess && !entry.writable)
        {
            permissionMiss = true;
            continue;
        }

        if (m_tlbTranslationCacheDiagnosticsEnabled)
        {
            ++m_tlbTranslationCacheStats.hits;
        }
        return EeTranslatedAddress{
            entry.physicalPageBase + pageOffset,
            entry.scratchpad,
        };
    }

    if (m_tlbTranslationCacheDiagnosticsEnabled)
    {
        ++m_tlbTranslationCacheStats.misses;
        if (generationMiss)
        {
            ++m_tlbTranslationCacheStats.generationMisses;
        }
        if (permissionMiss)
        {
            ++m_tlbTranslationCacheStats.permissionMisses;
        }
    }
    return std::nullopt;
}

void PS2Memory::fillTlbTranslationCache(
    uint32_t virtualAddress,
    const EeTranslatedAddress &translated,
    EeAddressTranslationContext translation,
    uint32_t tlbPageSize,
    bool global,
    bool writable)
{
    if (!m_tlbTranslationCacheEnabled)
    {
        return;
    }

    // The mapped fast path is deliberately RDRAM-only. Scratchpad mappings
    // retain the authoritative translator and local-memory distinction.
    if (translated.scratchpad)
    {
        return;
    }

    constexpr uint32_t pageMask =
        kTlbTranslationCachePageSize - 1u;
    const uint32_t pageOffset = virtualAddress & pageMask;
    if ((translated.physicalAddress & pageMask) != pageOffset)
    {
        return;
    }
    const uint32_t physicalPageBase =
        translated.physicalAddress - pageOffset;
    if (physicalPageBase >
        PS2_RAM_SIZE - kTlbTranslationCachePageSize)
    {
        // Device and bus-error translations retain the authoritative path.
        return;
    }

    const uint32_t virtualPageBase =
        virtualAddress & ~pageMask;
    const size_t setIndex =
        (virtualAddress >> 12u) &
        (kTlbTranslationCacheSetCount - 1u);
    auto &set = m_tlbTranslationCache[setIndex];
    size_t selectedWay = kTlbTranslationCacheWayCount;
    for (size_t way = 0u;
         way < kTlbTranslationCacheWayCount;
         ++way)
    {
        EeTlbTranslationCacheEntry &entry = set[way];
        const bool sameAddressSpace =
            entry.valid &&
            entry.generation == m_tlbMappingGeneration &&
            entry.virtualPageBase == virtualPageBase &&
            entry.mode == translation.mode &&
            entry.enforceOperatingMode ==
                translation.enforceOperatingMode &&
            entry.errorLevel == translation.errorLevel &&
            entry.global == global &&
            (global || entry.asid == translation.asid);
        if (sameAddressSpace)
        {
            selectedWay = way;
            break;
        }
        if (selectedWay == kTlbTranslationCacheWayCount &&
            entry.generation != m_tlbMappingGeneration)
        {
            selectedWay = way;
        }
    }
    if (selectedWay == kTlbTranslationCacheWayCount)
    {
        selectedWay = m_tlbTranslationCacheNextWay[setIndex];
        m_tlbTranslationCacheNextWay[setIndex] =
            static_cast<uint8_t>(
                (selectedWay + 1u) %
                kTlbTranslationCacheWayCount);
    }

    set[selectedWay] = {
        m_tlbMappingGeneration,
        virtualPageBase,
        physicalPageBase,
        tlbPageSize,
        translation.asid,
        translation.mode,
        translation.enforceOperatingMode,
        translation.errorLevel,
        true,
        global,
        writable,
        translated.scratchpad,
    };
}

PS2Memory::EeTranslatedAddress
PS2Memory::translateAddressImpl(
    uint32_t virtualAddress,
    EeAddressTranslationContext translation,
    bool writeAccess,
    uint32_t accessSize,
    bool allowTlbCache)
{
    if (!translation.permits(virtualAddress))
    {
        throw PS2AddressErrorException(virtualAddress);
    }

    if (isScratchpad(virtualAddress))
    {
        return {
            ps2ScratchpadOffset(virtualAddress),
            true,
        };
    }

    // The runtime accepts the EE's numerically addressed physical device
    // windows as compatibility aliases. Keep that explicit exception
    // separate from ordinary kuseg translation.
    if (virtualAddress < 0x80000000u &&
        Ps2IsPhysicalSpecialAddress(virtualAddress))
    {
        return {virtualAddress, false};
    }

    // EE TLB aliases of main RAM installed by the kernel.
    if (Ps2IsUncachedRamMirrorAddress(virtualAddress))
    {
        return {virtualAddress & PS2_RAM_MASK, false};
    }
    if (Ps2IsAcceleratedRamMirrorAddress(virtualAddress))
    {
        return {virtualAddress & PS2_RAM_MASK, false};
    }

    // Status.ERL makes ordinary kuseg uncached and unmapped. Keep this
    // architectural path explicit so normal kuseg translation can move to
    // the TLB independently of the runtime's compatibility aliases above.
    if (translation.errorLevel && virtualAddress < 0x80000000u)
    {
        return {virtualAddress, false};
    }

    // KSEG0/KSEG1 direct-mapped window.
    if (Ps2IsKseg01Address(virtualAddress))
    {
        return {
            Ps2DirectMappedPhysicalAddress(virtualAddress),
            false,
        };
    }

    // Ordinary kuseg/useg/suseg and KSEG2/KSEG3 are TLB mapped. Unchecked
    // runtime-internal low addresses remain physical-style and fall through.
    if (translation.mapsKusegThroughTlb(virtualAddress) ||
        Ps2IsKseg23Address(virtualAddress))
    {
        if (allowTlbCache)
        {
            if (const auto cached =
                    lookupTlbTranslationCache(
                        virtualAddress,
                        translation,
                        writeAccess,
                        accessSize))
            {
                return *cached;
            }
        }
        for (const auto &entry : m_tlbEntries)
        {
            constexpr uint32_t entryHiVpnMask =
                0xffffe000u;
            constexpr uint32_t entryHiAsidMask =
                0x000000ffu;
            constexpr uint32_t entryLoScratchpad =
                0x80000000u;
            constexpr uint32_t entryLoPfnMask =
                0x03ffffc0u;
            constexpr uint32_t entryLoDirty = 0x4u;
            constexpr uint32_t entryLoValid = 0x2u;
            constexpr uint32_t entryLoGlobal = 0x1u;

            const uint32_t pageMask =
                entry.pageMask & 0x01ffe000u;
            const bool scratchpad =
                (entry.entryLo0 & entryLoScratchpad) != 0u;
            const uint32_t pairOffsetMask =
                scratchpad
                    ? 0x00003fffu
                    : pageMask | 0x00001fffu;
            const uint32_t compareMask =
                ~pairOffsetMask;
            if ((virtualAddress & compareMask) !=
                (entry.entryHi & entryHiVpnMask &
                 compareMask))
            {
                continue;
            }

            const bool global =
                (entry.entryLo0 & entryLoGlobal) != 0u;
            if (!global &&
                (entry.entryHi & entryHiAsidMask) !=
                    translation.asid)
            {
                continue;
            }

            const uint32_t pageSize =
                ((pageMask >> 13u) + 1u) << 12u;
            const uint32_t entryLo =
                scratchpad ||
                        (virtualAddress & pageSize) == 0u
                    ? entry.entryLo0
                    : entry.entryLo1;
            if ((entryLo & entryLoValid) == 0u)
            {
                if (allowTlbCache)
                {
                    recordTlbTranslationSlowPathFault();
                }
                throw PS2TlbInvalidException(
                    virtualAddress);
            }
            if (writeAccess &&
                (entryLo & entryLoDirty) == 0u)
            {
                if (allowTlbCache)
                {
                    recordTlbTranslationSlowPathFault();
                }
                throw PS2TlbModifiedException(
                    virtualAddress);
            }

            if (scratchpad)
            {
                const EeTranslatedAddress translated{
                    virtualAddress & 0x00003fffu,
                    true,
                };
                if (allowTlbCache)
                {
                    fillTlbTranslationCache(
                        virtualAddress,
                        translated,
                        translation,
                        0x00004000u,
                        global,
                        (entryLo & entryLoDirty) != 0u);
                }
                return translated;
            }

            const uint32_t pageOffsetMask =
                pageSize - 1u;
            const uint32_t pfnMask =
                pageMask >> 13u;
            const uint32_t pfn =
                ((entryLo & entryLoPfnMask) >> 6u) &
                ~pfnMask;
            const EeTranslatedAddress translated{
                (pfn << 12u) |
                    (virtualAddress & pageOffsetMask),
                false,
            };
            if (allowTlbCache)
            {
                fillTlbTranslationCache(
                    virtualAddress,
                    translated,
                    translation,
                    pageSize,
                    global,
                    (entryLo & entryLoDirty) != 0u);
            }
            return translated;
        }
        if (allowTlbCache)
        {
            recordTlbTranslationSlowPathFault();
        }
        throw PS2TlbMissException(virtualAddress);
    }

    return {virtualAddress, false};
}

bool PS2Memory::tlbRead(
    uint32_t index,
    EeTlbEntry &entry) const
{
    if (index >= m_tlbEntries.size())
    {
        return false;
    }

    entry = m_tlbEntries[index];
    return true;
}

void PS2Memory::installPostBiosTlbState()
{
    // SCPH-39001 BIOS ELF-loader handoff captured through TLBR in strict
    // interpreter mode. These are architectural guest mappings, not host
    // address shortcuts; ordinary kuseg still faults after the guest clears
    // or replaces them.
    static constexpr std::array<EeTlbEntry, 48u> entries{{
        {0x00000000u, 0x70000000u, 0x0000001fu, 0x0000001fu},
        {0x00006000u, 0xffff8000u, 0x00001e1fu, 0x00001f1fu},
        {0x00000000u, 0x10000000u, 0x00400017u, 0x00400053u},
        {0x00000000u, 0x10002000u, 0x00400097u, 0x004000d7u},
        {0x00000000u, 0x10004000u, 0x00400117u, 0x00400157u},
        {0x00000000u, 0x10006000u, 0x00400197u, 0x004001d7u},
        {0x00000000u, 0x10008000u, 0x00400217u, 0x00400257u},
        {0x00000000u, 0x1000a000u, 0x00400297u, 0x004002d7u},
        {0x00000000u, 0x1000c000u, 0x00400313u, 0x00400357u},
        {0x00000000u, 0x1000e000u, 0x00400397u, 0x004003d7u},
        {0x0001e000u, 0x11000000u, 0x00440017u, 0x00440415u},
        {0x0001e000u, 0x12000000u, 0x00480017u, 0x00480415u},
        {0x01ffe000u, 0x1e000000u, 0x00780017u, 0x007c0017u},
        {0x0007e000u, 0x00080000u, 0x0000201fu, 0x0000301fu},
        {0x0007e000u, 0x00100000u, 0x0000401fu, 0x0000501fu},
        {0x0007e000u, 0x00180000u, 0x0000601fu, 0x0000701fu},
        {0x001fe000u, 0x00200000u, 0x0000801fu, 0x0000c01fu},
        {0x001fe000u, 0x00400000u, 0x0001001fu, 0x0001401fu},
        {0x001fe000u, 0x00600000u, 0x0001801fu, 0x0001c01fu},
        {0x007fe000u, 0x00800000u, 0x0002001fu, 0x0003001fu},
        {0x007fe000u, 0x01000000u, 0x0004001fu, 0x0005001fu},
        {0x007fe000u, 0x01800000u, 0x0006001fu, 0x0007001fu},
        {0x0007e000u, 0x20080000u, 0x00002017u, 0x00003017u},
        {0x0007e000u, 0x20100000u, 0x00004017u, 0x00005017u},
        {0x0007e000u, 0x20180000u, 0x00006017u, 0x00007017u},
        {0x001fe000u, 0x20200000u, 0x00008017u, 0x0000c017u},
        {0x001fe000u, 0x20400000u, 0x00010017u, 0x00014017u},
        {0x001fe000u, 0x20600000u, 0x00018017u, 0x0001c017u},
        {0x007fe000u, 0x20800000u, 0x00020017u, 0x00030017u},
        {0x007fe000u, 0x21000000u, 0x00040017u, 0x00050017u},
        {0x007fe000u, 0x21800000u, 0x00060017u, 0x00070017u},
        {0x0007e000u, 0x30100000u, 0x0000403fu, 0x0000503fu},
        {0x0007e000u, 0x30180000u, 0x0000603fu, 0x0000703fu},
        {0x001fe000u, 0x30200000u, 0x0000803fu, 0x0000c03fu},
        {0x001fe000u, 0x30400000u, 0x0001003fu, 0x0001403fu},
        {0x001fe000u, 0x30600000u, 0x0001803fu, 0x0001c03fu},
        {0x007fe000u, 0x30800000u, 0x0002003fu, 0x0003003fu},
        {0x007fe000u, 0x31000000u, 0x0004003fu, 0x0005003fu},
        {0x007fe000u, 0x31800000u, 0x0006003fu, 0x0007003fu},
        {0x00000000u, 0xe004c000u, 0x00000010u, 0x00000010u},
        {0x00000000u, 0xe004e000u, 0x00000010u, 0x00000010u},
        {0x00000000u, 0xe0050000u, 0x00000010u, 0x00000010u},
        {0x00000000u, 0xe0052000u, 0x00000010u, 0x00000010u},
        {0x00000000u, 0xe0054000u, 0x00000010u, 0x00000010u},
        {0x00000000u, 0xe0056000u, 0x00000010u, 0x00000010u},
        {0x00000000u, 0xe0058000u, 0x00000010u, 0x00000010u},
        {0x00000000u, 0xe005a000u, 0x00000010u, 0x00000010u},
        {0x00000000u, 0xe005c000u, 0x00000010u, 0x00000010u},
    }};

    m_postBiosTlbStateConfigured = true;
    m_tlbEntries.assign(entries.begin(), entries.end());
    advanceTlbMappingGeneration();
}

bool PS2Memory::tlbWrite(
    uint32_t index,
    const EeTlbEntry &entry)
{
    if (index >= m_tlbEntries.size())
    {
        return false;
    }

    constexpr uint32_t pageMaskMask = 0x01ffe000u;
    constexpr uint32_t entryHiMask = 0xffffe0ffu;
    constexpr uint32_t entryLo0Mask = 0x83ffffffu;
    constexpr uint32_t entryLo1Mask = 0x03ffffffu;
    constexpr uint32_t entryLoCacheMask = 0x38u;
    constexpr uint32_t entryLoUncached = 2u << 3u;
    constexpr uint32_t entryLoGlobal = 0x1u;
    constexpr uint32_t entryLoScratchpad = 0x80000000u;

    EeTlbEntry normalized{};
    normalized.pageMask =
        entry.pageMask & pageMaskMask;
    normalized.entryLo0 =
        entry.entryLo0 & entryLo0Mask;
    normalized.entryLo1 =
        entry.entryLo1 & entryLo1Mask;

    const bool global =
        (normalized.entryLo0 &
         normalized.entryLo1 &
         entryLoGlobal) != 0u;
    normalized.entryLo0 =
        (normalized.entryLo0 & ~entryLoGlobal) |
        (global ? entryLoGlobal : 0u);
    normalized.entryLo1 =
        (normalized.entryLo1 & ~entryLoGlobal) |
        (global ? entryLoGlobal : 0u);

    if ((normalized.entryLo0 &
         entryLoScratchpad) != 0u)
    {
        normalized.pageMask = 0u;
        normalized.entryLo0 =
            (normalized.entryLo0 &
             ~entryLoCacheMask) |
            entryLoUncached;
        normalized.entryLo1 =
            (normalized.entryLo1 &
             ~entryLoCacheMask) |
            entryLoUncached;
    }

    normalized.entryHi =
        (entry.entryHi & entryHiMask) &
        ~normalized.pageMask;
    m_tlbEntries[index] = normalized;
    advanceTlbMappingGeneration();
    return true;
}

int32_t PS2Memory::tlbProbe(uint32_t entryHi) const
{
    constexpr uint32_t entryHiVpnMask = 0xffffe000u;
    constexpr uint32_t entryHiAsidMask = 0x000000ffu;
    constexpr uint32_t entryLoGlobal = 0x1u;
    const uint32_t probeVpn =
        entryHi & entryHiVpnMask;
    const uint32_t probeAsid =
        entryHi & entryHiAsidMask;
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_tlbEntries.size()); ++i)
    {
        const EeTlbEntry &entry = m_tlbEntries[i];
        const uint32_t pageMask =
            entry.pageMask & 0x01ffe000u;
        const uint32_t normalizedProbeVpn =
            probeVpn & ~pageMask;
        const bool global =
            (entry.entryLo0 & entryLoGlobal) != 0u;
        if (normalizedProbeVpn ==
                (entry.entryHi & entryHiVpnMask) &&
            (global ||
             probeAsid ==
                 (entry.entryHi & entryHiAsidMask)))
        {
            return static_cast<int32_t>(i);
        }
    }

    return -1;
}

uint8_t PS2Memory::read8(
    uint32_t address,
    EeAddressTranslationContext translation)
{
    const EeTranslatedAddress translated =
        translateAddressImpl(
            address, translation, false, sizeof(uint8_t));
    const bool scratch = translated.scratchpad;
    const uint32_t physAddr = translated.physicalAddress;
    checkEePhysicalBusError(physAddr);

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

uint16_t PS2Memory::read16(
    uint32_t address,
    EeAddressTranslationContext translation)
{
    if (address & 1)
    {
        throw std::runtime_error("Unaligned 16-bit read at address: 0x" + std::to_string(address));
    }

    const EeTranslatedAddress translated =
        translateAddressImpl(
            address, translation, false, sizeof(uint16_t));
    const bool scratch = translated.scratchpad;
    const uint32_t physAddr = translated.physicalAddress;
    checkEePhysicalBusError(physAddr);

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

uint32_t PS2Memory::read32(
    uint32_t address,
    EeAddressTranslationContext translation)
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

    const EeTranslatedAddress translated =
        translateAddressImpl(
            address, translation, false, sizeof(uint32_t));
    const bool scratch = translated.scratchpad;
    const uint32_t physAddr = translated.physicalAddress;
    checkEePhysicalBusError(physAddr);

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

uint64_t PS2Memory::read64(
    uint32_t address,
    EeAddressTranslationContext translation)
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

    const EeTranslatedAddress translated =
        translateAddressImpl(
            address, translation, false, sizeof(uint64_t));
    const bool scratch = translated.scratchpad;
    const uint32_t physAddr = translated.physicalAddress;
    checkEePhysicalBusError(physAddr);

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
        const bool counterAccess =
            ps2x::timing::EeCounterBank::isRegisterAddress(
                address) ||
            ps2x::timing::EeCounterBank::isRegisterAddress(
                address + 4u);
        uint32_t lo = counterAccess
                          ? readIORegister(address)
                          : (m_ioRegisters.count(address)
                                 ? m_ioRegisters[address]
                                 : 0u);
        uint32_t hi = counterAccess
                          ? readIORegister(address + 4u)
                          : (m_ioRegisters.count(address + 4u)
                                 ? m_ioRegisters[address + 4u]
                                 : 0u);
        return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    }
    return (uint64_t)read32(address, translation) |
           ((uint64_t)read32(address + 4, translation) << 32);
}

__m128i PS2Memory::read128(
    uint32_t address,
    EeAddressTranslationContext translation)
{
    if (address & 15)
    {
        throw std::runtime_error("Unaligned 128-bit read at address: 0x" + std::to_string(address));
    }

    const EeTranslatedAddress translated =
        translateAddressImpl(
            address, translation, false, sizeof(__m128i));
    const bool scratch = translated.scratchpad;
    const uint32_t physAddr = translated.physicalAddress;
    checkEePhysicalBusError(physAddr);

    if (physAddr == 0x10007000u)
    {
        alignas(16) std::array<uint8_t, 16u> payload{};
        (void)readIpuOutputFifo(payload.data(), 1u);
        return _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(
                payload.data()));
    }

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

void PS2Memory::write8(
    uint32_t address,
    uint8_t value,
    uint32_t writerPc,
    EeAddressTranslationContext translation)
{
    const EeTranslatedAddress translated =
        translateAddressImpl(
            address, translation, true, sizeof(uint8_t));
    const bool scratch = translated.scratchpad;
    const uint32_t physAddr = translated.physicalAddress;
    checkEePhysicalBusError(physAddr);

    if (isGsPrivReg(physAddr))
    {
        const uint32_t shift = (physAddr & 3u) * 8u;
        writeMasked32(
            physAddr & ~3u,
            static_cast<uint32_t>(value) << shift,
            static_cast<uint8_t>(1u << (physAddr & 3u)),
            writerPc);
        return;
    }

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
        const uint32_t regAddr = physAddr & ~3u;
        const uint32_t lane = physAddr & 3u;
        writeIORegisterMasked(
            regAddr,
            static_cast<uint32_t>(value) << (lane * 8u),
            static_cast<uint8_t>(1u << lane));
    }
}

void PS2Memory::write16(
    uint32_t address,
    uint16_t value,
    uint32_t writerPc,
    EeAddressTranslationContext translation)
{
    if (address & 1)
    {
        throw std::runtime_error("Unaligned 16-bit write at address: 0x" + std::to_string(address));
    }

    const EeTranslatedAddress translated =
        translateAddressImpl(
            address, translation, true, sizeof(uint16_t));
    const bool scratch = translated.scratchpad;
    const uint32_t physAddr = translated.physicalAddress;
    checkEePhysicalBusError(physAddr);

    if (isGsPrivReg(physAddr))
    {
        const uint32_t lane = physAddr & 2u;
        writeMasked32(
            physAddr & ~3u,
            static_cast<uint32_t>(value) << (lane * 8u),
            static_cast<uint8_t>(0x3u << lane),
            writerPc);
        return;
    }

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
        const uint32_t regAddr = physAddr & ~3u;
        const uint32_t lane = physAddr & 2u;
        writeIORegisterMasked(
            regAddr,
            static_cast<uint32_t>(value) << (lane * 8u),
            static_cast<uint8_t>(0x3u << lane));
    }
}

void PS2Memory::write32(
    uint32_t address,
    uint32_t value,
    uint32_t writerPc,
    EeAddressTranslationContext translation)
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

    const EeTranslatedAddress translated =
        translateAddressImpl(
            address, translation, true, sizeof(uint32_t));
    const bool scratch = translated.scratchpad;
    const uint32_t physAddr = translated.physicalAddress;
    checkEePhysicalBusError(physAddr);

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

void PS2Memory::write64(
    uint32_t address,
    uint64_t value,
    uint32_t writerPc,
    EeAddressTranslationContext translation)
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

    const EeTranslatedAddress translated =
        translateAddressImpl(
            address, translation, true, sizeof(uint64_t));
    const bool scratch = translated.scratchpad;
    const uint32_t physAddr = translated.physicalAddress;
    checkEePhysicalBusError(physAddr);

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
        write32(address, (uint32_t)value, writerPc, translation);
        write32(
            address + 4,
            (uint32_t)(value >> 32),
            writerPc,
            translation);
    }
}

void PS2Memory::writeMasked32(
    uint32_t address,
    uint32_t value,
    uint8_t byteEnable,
    uint32_t writerPc,
    EeAddressTranslationContext translation)
{
    if (address & 3u)
    {
        throw std::runtime_error(
            "Unaligned masked 32-bit write at address: 0x" +
            std::to_string(address));
    }
    const Ps2ByteEnableSpan span =
        Ps2DecodeByteEnableSpan(byteEnable, 4u);
    if (!span.valid)
    {
        throw std::runtime_error("Invalid masked 32-bit byte enables");
    }
    if (span.size == 0u)
    {
        return;
    }

    const uint32_t writeMask = static_cast<uint32_t>(
        Ps2ExpandByteEnableMask(byteEnable, 4u));
    if (isGsPrivReg(address))
    {
        const uint32_t off = address & 7u;
        const uint32_t regOff =
            (address - PS2_GS_PRIV_REG_BASE) & ~7u;
        if (regOff == kGsCsrRegOffset)
        {
            writeCsrHalf(gs_regs.csr, off, value, writeMask);
        }
        else if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            const uint64_t shiftedMask =
                static_cast<uint64_t>(writeMask) << (off * 8u);
            const uint64_t shiftedValue =
                static_cast<uint64_t>(value) << (off * 8u);
            *reg = (*reg & ~shiftedMask) |
                   (shiftedValue & shiftedMask);
        }
        m_gsWriteCount.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    const EeTranslatedAddress translated =
        translateAddressImpl(
            address, translation, true, sizeof(uint32_t));
    const bool scratch = translated.scratchpad;
    const uint32_t physAddr = translated.physicalAddress;
    checkEePhysicalBusError(physAddr);
    auto writeBytes = [&](uint8_t *destination)
    {
        for (uint32_t index = span.offset;
             index < span.offset + span.size;
             ++index)
        {
            destination[index] = static_cast<uint8_t>(
                value >> (index * 8u));
        }
    };

    if (scratch)
    {
        inRange(
            physAddr, sizeof(value), PS2_SCRATCHPAD_SIZE,
            "writeMasked32 scratchpad", address);
        writeBytes(m_scratchpad + physAddr);
        return;
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        inRange(
            physAddr, sizeof(value), PS2_RAM_SIZE,
            "writeMasked32 rdram", address);
        markModified(
            physAddr + span.offset, span.size, writerPc);
        writeBytes(m_rdram + physAddr);
        return;
    }

    uint32_t vuOffset = 0u;
    uint32_t vuLimit = 0u;
    if (uint8_t *vuMem =
            mapVuMemory(physAddr, sizeof(value), vuOffset, vuLimit))
    {
        inRange(
            vuOffset, sizeof(value), vuLimit,
            "writeMasked32 vu", address);
        writeBytes(vuMem + vuOffset);
        markVUCodeModified(vuMem);
        return;
    }
    if (isIoRegister(physAddr))
    {
        writeIORegisterMasked(physAddr, value, byteEnable);
    }
}

void PS2Memory::writeMasked64(
    uint32_t address,
    uint64_t value,
    uint8_t byteEnable,
    uint32_t writerPc,
    EeAddressTranslationContext translation)
{
    if (address & 7u)
    {
        throw std::runtime_error(
            "Unaligned masked 64-bit write at address: 0x" +
            std::to_string(address));
    }
    const Ps2ByteEnableSpan span =
        Ps2DecodeByteEnableSpan(byteEnable, 8u);
    if (!span.valid)
    {
        throw std::runtime_error("Invalid masked 64-bit byte enables");
    }
    if (span.size == 0u)
    {
        return;
    }

    const uint64_t writeMask =
        Ps2ExpandByteEnableMask(byteEnable, 8u);
    if (isGsPrivReg(address))
    {
        const uint32_t regOff =
            (address - PS2_GS_PRIV_REG_BASE) & ~7u;
        if (regOff == kGsCsrRegOffset)
        {
            writeCsrFull(gs_regs.csr, value, writeMask);
        }
        else if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            *reg = (*reg & ~writeMask) |
                   (value & writeMask);
        }
        m_gsWriteCount.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    const EeTranslatedAddress translated =
        translateAddressImpl(
            address, translation, true, sizeof(uint64_t));
    const bool scratch = translated.scratchpad;
    const uint32_t physAddr = translated.physicalAddress;
    checkEePhysicalBusError(physAddr);
    auto writeBytes = [&](uint8_t *destination)
    {
        for (uint32_t index = span.offset;
             index < span.offset + span.size;
             ++index)
        {
            destination[index] = static_cast<uint8_t>(
                value >> (index * 8u));
        }
    };

    if (scratch)
    {
        inRange(
            physAddr, sizeof(value), PS2_SCRATCHPAD_SIZE,
            "writeMasked64 scratchpad", address);
        writeBytes(m_scratchpad + physAddr);
        return;
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        inRange(
            physAddr, sizeof(value), PS2_RAM_SIZE,
            "writeMasked64 rdram", address);
        markModified(
            physAddr + span.offset, span.size, writerPc);
        writeBytes(m_rdram + physAddr);
        return;
    }

    uint32_t vuOffset = 0u;
    uint32_t vuLimit = 0u;
    if (uint8_t *vuMem =
            mapVuMemory(physAddr, sizeof(value), vuOffset, vuLimit))
    {
        inRange(
            vuOffset, sizeof(value), vuLimit,
            "writeMasked64 vu", address);
        writeBytes(vuMem + vuOffset);
        markVUCodeModified(vuMem);
        return;
    }
    if (isIoRegister(physAddr))
    {
        const uint8_t lowEnable = byteEnable & 0xFu;
        const uint8_t highEnable =
            static_cast<uint8_t>((byteEnable >> 4u) & 0xFu);
        if (lowEnable != 0u)
        {
            writeIORegisterMasked(
                physAddr,
                static_cast<uint32_t>(value),
                lowEnable);
        }
        if (highEnable != 0u)
        {
            writeIORegisterMasked(
                physAddr + 4u,
                static_cast<uint32_t>(value >> 32u),
                highEnable);
        }
    }
}

void PS2Memory::write128(
    uint32_t address,
    __m128i value,
    uint32_t writerPc,
    EeAddressTranslationContext translation)
{
    if (address & 15)
    {
        throw std::runtime_error("Unaligned 128-bit write at address: 0x" + std::to_string(address));
    }

    const EeTranslatedAddress translated =
        translateAddressImpl(
            address, translation, true, sizeof(__m128i));
    const bool scratch = translated.scratchpad;
    const uint32_t physAddr = translated.physicalAddress;
    checkEePhysicalBusError(physAddr);

    if (physAddr == 0x10007010u)
    {
        alignas(16) std::array<uint8_t, 16u> payload{};
        _mm_storeu_si128(
            reinterpret_cast<__m128i *>(payload.data()),
            value);
        (void)writeIpuInputFifo(payload.data(), 1u);
        return;
    }

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

        write64(address, lo, writerPc, translation);
        write64(address + 8, hi, writerPc, translation);
    }
}

bool PS2Memory::writeIORegister(uint32_t address, uint32_t value)
{
    return writeIORegisterMasked(address, value, 0xFu);
}

bool PS2Memory::writeIORegisterMasked(
    uint32_t address,
    uint32_t value,
    uint8_t byteEnable)
{
    const Ps2ByteEnableSpan span =
        Ps2DecodeByteEnableSpan(byteEnable, 4u);
    if (!span.valid)
    {
        return false;
    }
    if (span.size == 0u)
    {
        return true;
    }

    const uint32_t writeMask = static_cast<uint32_t>(
        Ps2ExpandByteEnableMask(byteEnable, 4u));
    const uint32_t writeValue = value & writeMask;
    const auto currentIt = m_ioRegisters.find(address);
    uint32_t currentValue =
        currentIt != m_ioRegisters.end()
            ? currentIt->second
            : 0u;
    if ((address >= 0x10003800u &&
         address < 0x10003A00u) ||
        (address >= 0x10003C00u &&
         address < 0x10003E00u))
    {
        const uint32_t base =
            address < 0x10003C00u
                ? 0x10003800u
                : 0x10003C00u;
        const VIFRegisters &regs =
            base == 0x10003800u
                ? vif0_regs
                : vif1_regs;
        if (const std::optional<uint32_t> vifValue =
                vifRegisterValue(regs, address - base);
            vifValue.has_value())
        {
            currentValue = *vifValue;
        }
    }
    const uint32_t mergedValue =
        (currentValue & ~writeMask) | writeValue;

    ps2x::timing::EeCounterAdvanceResult counterAdvance{};
    if (m_eeCounters.writeRegister(
            address,
            value,
            currentEeCounterCycle(),
            &counterAdvance,
            writeMask))
    {
        publishEeCounterAdvance(counterAdvance);
        return true;
    }

    if (address == kIntcStat)
    {
        m_ioRegisters[kIntcStat] =
            intcStatus() & ~writeValue;
        publishEeCounterAdvance({}, true);
        return true;
    }
    if (address == kIntcMask)
    {
        // EE INTC_MASK bits toggle when written as one.
        m_ioRegisters[kIntcMask] =
            intcMask() ^ writeValue;
        publishEeCounterAdvance({}, true);
        return true;
    }

    if (isGsPrivReg(address))
    {
        // NB: unreachable from write8/16/32/64 today since those all funnel IO
        // register writes through addresses in PS2_IO_BASE's range, which is
        // disjoint from PS2_GS_PRIV_REG_BASE; kept correct for direct callers.
        m_ioRegisters[address] = mergedValue;
        const uint32_t off = address & 7u;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            writeCsrHalf(
                gs_regs.csr, off, value, writeMask);
        }
        else if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            const uint64_t mask =
                static_cast<uint64_t>(writeMask) << (off * 8u);
            *reg =
                (*reg & ~mask) |
                ((static_cast<uint64_t>(value) << (off * 8u)) & mask);
        }
        m_gsWriteCount.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    if (address >= 0x10002000 && address <= 0x10002030)
    {
        if (address == 0x10002010)
        {
            m_ioRegisters[address] =
                mergedValue & ~(1u << 31);
            if (writeValue & (1u << 30))
            {
                clearIpuOutputFifo();
                clearIpuInputFifo();
                m_ioRegisters[0x10002000] = 0;
                m_ioRegisters[0x10002020] = 0;
                m_ioRegisters[0x10002030] = 0;
                if (m_toIpuDma.active &&
                    (m_toIpuDma.stall ==
                         ToIpuDmaStallReason::
                             InputFifoFull ||
                     m_toIpuDma.stall ==
                         ToIpuDmaStallReason::
                             WaitingForInputRequest))
                {
                    (void)wakeToIpuDma(4u);
                }
            }
            else
            {
                publishIpuFifoCounts();
            }
        }
        else
        {
            m_ioRegisters[address] = mergedValue;
        }
        return true;
    }

    if (address >= 0x10003000u &&
        address <= 0x10003020u)
    {
        constexpr uint32_t kGifCtrl = 0x10003000u;
        constexpr uint32_t kGifMode = 0x10003010u;
        constexpr uint32_t kGifStat = 0x10003020u;
        if (address == kGifCtrl)
        {
            const bool wasPaused = isGifPaused();
            m_ioRegisters[kGifCtrl] =
                mergedValue & 0x9u;
            if ((writeValue & 0x1u) != 0u)
            {
                if (m_gifArbiter)
                    m_gifArbiter->reset();
                m_path3MaskedFifo.clear();
                m_gifModePath3Masked = false;
                m_ioRegisters[kGifCtrl] = 0u;
                m_ioRegisters[kGifMode] = 0u;
            }
            updateGifStat();
            if (wasPaused && !isGifPaused() &&
                m_gifDma.active)
            {
                (void)wakeGifDma(16u);
            }
            return true;
        }
        if (address == kGifMode)
        {
            const bool wasM3r =
                m_gifModePath3Masked;
            m_ioRegisters[kGifMode] =
                mergedValue & 0x5u;
            m_gifModePath3Masked =
                (mergedValue & 0x1u) != 0u;
            updateGifStat();
            if (wasM3r &&
                !m_gifModePath3Masked &&
                (m_gifDma.active ||
                 !m_gifDma.fifoData.empty()))
            {
                (void)wakeGifDma(8u);
            }
            return true;
        }
        if (address == kGifStat)
        {
            // GIF_STAT is read-only.
            updateGifStat();
            return true;
        }
    }

    if (address == 0x1000E010u)
    {
        const uint32_t current = m_ioRegisters.count(address) ? m_ioRegisters[address] : 0u;
        uint32_t status = current & 0x3FFu;
        uint32_t mask = (current >> 16) & 0x3FFu;

        // D_STAT low bits are W1C status, high bits [16..25] toggle masks on write-one.
        status &= ~(writeValue & 0x3FFu);
        mask ^= ((writeValue >> 16) & 0x3FFu);

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

    if (address == 0x1000E000u) // D_CTRL
    {
        const bool wasEnabled = isDmacEnabled();
        m_ioRegisters[address] = mergedValue;
        const bool enabled = isDmacEnabled();
        if (m_vif0Dma.active && wasEnabled != enabled)
        {
            if (!enabled)
            {
                m_vif0Dma.stall =
                    Vif0DmaStallReason::DmacDisabled;
                if (m_vif0Dma.eventManaged &&
                    m_vif0DmaCancelCallback)
                {
                    m_vif0DmaCancelCallback();
                }
            }
            else
            {
                (void)wakeVif0Dma();
            }
        }
        if (m_vif1Dma.active && wasEnabled != enabled)
        {
            if (!enabled)
            {
                m_vif1Dma.stall =
                    Vif1DmaStallReason::DmacDisabled;
                if (m_vif1Dma.eventManaged &&
                    m_vif1DmaCancelCallback)
                {
                    m_vif1DmaCancelCallback();
                }
            }
            else
            {
                (void)wakeVif1Dma();
            }
        }
        if (m_gifDma.active && wasEnabled != enabled)
        {
            if (!enabled)
            {
                m_gifDma.stall =
                    GifDmaStallReason::DmacDisabled;
                if (m_gifDma.eventManaged &&
                    m_gifDmaCancelCallback)
                {
                    m_gifDmaCancelCallback();
                }
                m_gifDma.eventManaged = false;
            }
            else
            {
                (void)wakeGifDma(64u);
            }
        }
        if (m_fromIpuDma.active &&
            wasEnabled != enabled)
        {
            if (!enabled)
            {
                m_fromIpuDma.stall =
                    FromIpuDmaStallReason::
                        DmacDisabled;
                if (m_fromIpuDma.eventManaged &&
                    m_fromIpuDmaCancelCallback)
                {
                    m_fromIpuDmaCancelCallback();
                }
                m_fromIpuDma.eventManaged = false;
            }
            else
            {
                (void)wakeFromIpuDma(0u);
            }
        }
        if (m_toIpuDma.active &&
            wasEnabled != enabled)
        {
            if (!enabled)
            {
                m_toIpuDma.stall =
                    ToIpuDmaStallReason::
                        DmacDisabled;
                if (m_toIpuDma.eventManaged &&
                    m_toIpuDmaCancelCallback)
                {
                    m_toIpuDmaCancelCallback();
                }
                m_toIpuDma.eventManaged = false;
            }
            else
            {
                (void)wakeToIpuDma(0u);
            }
        }
        return true;
    }

    m_ioRegisters[address] = mergedValue;

    if (address >= 0x10003C00u && address < 0x10003E00u)
    {
        m_vifWriteCount.fetch_add(1, std::memory_order_relaxed);

        switch (address)
        {
        case 0x10003C10u:     // VIF1_FBRST
            if (writeValue & 0x1u) // RST
            {
                std::memset(&vif1_regs, 0, sizeof(vif1_regs));
                m_vif1PendingPath2DirectQwc = 0u;
                m_vif1PendingPath2DirectHl = false;
                m_vif1PendingIrqAfterCommand = false;
                m_vif1WaitingForVu = false;
                m_vif1DeferredData.clear();
                const bool path3WasMasked =
                    isPath3Masked();
                m_path3Masked = false;
                updateGifStat();
                if (path3WasMasked &&
                    !isPath3Masked())
                {
                    flushMaskedPath3Packets();
                    (void)wakeGifDma(0u);
                }
                (void)cancelDmacTransfer(DmacChannel::Vif1);
                if (m_vif1ResetCallback)
                    m_vif1ResetCallback();
                break;
            }
            if (writeValue & 0x2u) // FBK
            {
                vif1_regs.stat |= (1u << 9u); // VFS
                vif1_regs.stat &= ~0x3u;       // VPS idle
                m_vif1Dma.stall =
                    Vif1DmaStallReason::ForceBreak;
                if (m_vif1Dma.active &&
                    m_vif1Dma.eventManaged &&
                    m_vif1DmaCancelCallback)
                {
                    m_vif1DmaCancelCallback();
                }
            }
            if (writeValue & 0x4u) // STP
            {
                vif1_regs.stat |= (1u << 8u); // VSS
                vif1_regs.stat &= ~0x3u;       // VPS idle
                m_vif1Dma.stall =
                    Vif1DmaStallReason::VifStop;
                if (m_vif1Dma.active &&
                    m_vif1Dma.eventManaged &&
                    m_vif1DmaCancelCallback)
                {
                    m_vif1DmaCancelCallback();
                }
            }
            if (writeValue & 0x8u) // STC
            {
                vif1_regs.stat &= ~((1u << 8) | (1u << 9) | (1u << 10) | (1u << 11) | (1u << 12) | (1u << 13));
                if (m_vif1Dma.active)
                    (void)wakeVif1Dma();
                else
                    (void)processVif1Stream();
            }
            break;
        case 0x10003C30u:
            vif1_regs.mark = mergedValue & 0xFFFFu;
            vif1_regs.stat &= ~(1u << 6); // clear MRK flag on CPU write
            break;
        case 0x10003C40u:
            vif1_regs.cycle = mergedValue & 0xFFFFu;
            break;
        case 0x10003C50u:
            vif1_regs.mode = mergedValue & 0x3u;
            break;
        case 0x10003C60u:
            vif1_regs.num = mergedValue & 0xFFu;
            break;
        case 0x10003C70u:
            vif1_regs.mask = mergedValue;
            break;
        case 0x10003C80u:
            vif1_regs.code = mergedValue;
            break;
        case 0x10003C90u:
            vif1_regs.itops = mergedValue & 0x3FFu;
            break;
        case 0x10003CA0u:
            vif1_regs.base = mergedValue & 0x3FFu;
            break;
        case 0x10003CB0u:
            vif1_regs.ofst = mergedValue & 0x3FFu;
            break;
        case 0x10003CC0u:
            vif1_regs.tops = mergedValue & 0x3FFu;
            break;
        case 0x10003CD0u:
            vif1_regs.itop = mergedValue & 0x3FFu;
            break;
        case 0x10003CE0u:
            vif1_regs.top = mergedValue & 0x3FFu;
            break;
        default:
            break;
        }

        return true;
    }

    if (address >= 0x10003800u && address < 0x10003A00u)
    {
        m_vifWriteCount.fetch_add(1, std::memory_order_relaxed);

        switch (address)
        {
        case 0x10003810u:     // VIF0_FBRST
            if (writeValue & 0x1u) // RST
            {
                std::memset(&vif0_regs, 0, sizeof(vif0_regs));
                m_vif0WaitingForVu = false;
                m_vif0DeferredData.clear();
                (void)cancelDmacTransfer(DmacChannel::Vif0);
                if (m_vif0ResetCallback)
                    m_vif0ResetCallback();
                break;
            }
            if (writeValue & 0x2u) // FBK
            {
                vif0_regs.stat |= (1u << 9u); // VFS
                vif0_regs.stat &= ~0x3u;       // VPS idle
                m_vif0Dma.stall =
                    Vif0DmaStallReason::ForceBreak;
                if (m_vif0Dma.active &&
                    m_vif0Dma.eventManaged &&
                    m_vif0DmaCancelCallback)
                {
                    m_vif0DmaCancelCallback();
                }
            }
            if (writeValue & 0x4u) // STP
            {
                vif0_regs.stat |= (1u << 8u); // VSS
                vif0_regs.stat &= ~0x3u;       // VPS idle
                m_vif0Dma.stall =
                    Vif0DmaStallReason::VifStop;
                if (m_vif0Dma.active &&
                    m_vif0Dma.eventManaged &&
                    m_vif0DmaCancelCallback)
                {
                    m_vif0DmaCancelCallback();
                }
            }
            if (writeValue & 0x8u) // STC
            {
                vif0_regs.stat &=
                    ~((1u << 8u) | (1u << 9u) |
                      (1u << 10u) | (1u << 11u) |
                      (1u << 12u) | (1u << 13u));
                if (m_vif0Dma.active)
                    (void)wakeVif0Dma();
                else
                    (void)processVif0Stream();
            }
            break;
        case 0x10003830u:
            vif0_regs.mark = mergedValue & 0xFFFFu;
            vif0_regs.stat &= ~(1u << 6); // clear MRK
            break;
        case 0x10003840u:
            vif0_regs.cycle = mergedValue & 0xFFFFu;
            break;
        case 0x10003850u:
            vif0_regs.mode = mergedValue & 0x3u;
            break;
        case 0x10003860u:
            vif0_regs.num = mergedValue & 0xFFu;
            break;
        case 0x10003870u:
            vif0_regs.mask = mergedValue;
            break;
        case 0x10003880u:
            vif0_regs.code = mergedValue;
            break;
        case 0x10003890u:
            vif0_regs.itops = mergedValue & 0xFFu;
            break;
        case 0x100038D0u:
            vif0_regs.itop = mergedValue & 0xFFu;
            break;
        default:
            break;
        }
        return true;
    }

    if (address >= 0x10008000 && address < 0x1000F000)
    {
        const uint32_t channelBase =
            address & 0xFFFFFF00u;
        const std::optional<DmacChannel> channel =
            dmacChannelFromBase(channelBase);
        const bool writesStr =
            (writeMask & 0x100u) != 0u;
        if ((address & 0xFFu) == 0x00u &&
            channel.has_value() &&
            writesStr &&
            (mergedValue & 0x100u) == 0u)
        {
            (void)cancelDmacTransfer(*channel);
            return true;
        }

        if ((address & 0xFFu) == 0x00u &&
            channel.has_value() &&
            writesStr &&
            (mergedValue & 0x100u) != 0u)
        {
            const auto dctrlIt = m_ioRegisters.find(0x1000E000u);
            const bool dmacEnabled = (dctrlIt == m_ioRegisters.end()) || ((dctrlIt->second & 0x1u) != 0u);
            if (!dmacEnabled &&
                channelBase != 0x1000B000u &&
                channelBase != 0x1000B400u)
            {
                return true;
            }

            const DmacTransferToken transfer =
                beginDmacTransfer(*channel);
            const uint32_t madr = m_ioRegisters[channelBase + 0x10];
            const uint32_t qwc =
                m_ioRegisters[channelBase + 0x20] & 0xFFFFu;
            m_dmaStartCount.fetch_add(1, std::memory_order_relaxed);

            if (channelBase == 0x1000B000u)
            {
                (void)startFromIpuDma(transfer, mergedValue);
                const FromIpuDmaAdvanceResult initial =
                    advanceFromIpuDma();
                if (initial.active &&
                    !initial.completed &&
                    initial.stall ==
                        FromIpuDmaStallReason::None &&
                    !scheduleFromIpuDma(
                        initial.delayEeCycles))
                {
                    drainFromIpuDmaCompatibility();
                }
                return true;
            }

            if (channelBase == 0x1000B400u)
            {
                (void)startToIpuDma(transfer, mergedValue);
                const ToIpuDmaAdvanceResult initial =
                    advanceToIpuDma();
                if (initial.active &&
                    !initial.completed &&
                    initial.stall ==
                        ToIpuDmaStallReason::None &&
                    !scheduleToIpuDma(
                        initial.delayEeCycles))
                {
                    drainToIpuDmaCompatibility();
                }
                return true;
            }

            if (channelBase == 0x1000D000u || channelBase == 0x1000D400u)
            {
                (void)startScratchpadDma(transfer, mergedValue);
                ScratchpadDmaSnapshot submitted =
                    scratchpadDmaSnapshot(*channel);
                uint32_t initialDelayEeCycles = 0u;
                bool pendingProgress =
                    submitted.active &&
                    submitted.phase ==
                        ScratchpadDmaPhase::Finalize;
                if (submitted.active &&
                    submitted.phase !=
                        ScratchpadDmaPhase::Finalize)
                {
                    const ScratchpadDmaAdvanceResult initial =
                        advanceScratchpadDma(*channel);
                    pendingProgress =
                        initial.active &&
                        !initial.completed &&
                        initial.phase !=
                            ScratchpadDmaPhase::Fault;
                    initialDelayEeCycles =
                        initial.delayEeCycles;
                }
                if (pendingProgress &&
                    !scheduleScratchpadDma(
                        *channel,
                        initialDelayEeCycles))
                {
                    drainScratchpadDmaCompatibility(*channel);
                }
                return true;
            }

            if (channelBase == 0x10009000u)
            {
                m_ioRegisters[channelBase + 0x20u] = qwc;
                beginVif1DmaTrace(
                    mergedValue, madr, qwc,
                    m_ioRegisters[channelBase + 0x30u],
                    m_ioRegisters[channelBase + 0x40u],
                    m_ioRegisters[channelBase + 0x50u],
                    m_ioRegisters[channelBase + 0x80u]);
                (void)startVif1Dma(transfer, mergedValue);
                if (!scheduleVif1Dma(4u))
                    drainVif1DmaCompatibility();
                return true;
            }

            if (channelBase == 0x10008000u)
            {
                m_ioRegisters[channelBase + 0x20u] = qwc;
                (void)startVif0Dma(transfer, mergedValue);
                if (!scheduleVif0Dma(4u))
                    drainVif0DmaCompatibility();
                return true;
            }

            if (channelBase == 0x1000A000u && m_gsVRAM)
            {
                (void)startGifDma(transfer, mergedValue);
                const bool hasGifSink =
                    static_cast<bool>(m_gifPacketCallback) ||
                    m_gifArbiter != nullptr;
                if (hasGifSink)
                {
                    const GifDmaAdvanceResult initial =
                        advanceGifDma();
                    if (initial.active &&
                        initial.stall ==
                            GifDmaStallReason::None &&
                        initial.progressed)
                    {
                        if (!scheduleGifDma(
                                initial.delayEeCycles))
                        {
                            drainGifDmaCompatibility();
                        }
                    }
                    else if (
                        initial.active &&
                        initial.stall !=
                            GifDmaStallReason::
                                Path3Masked &&
                        !scheduleGifDma(
                            initial.delayEeCycles))
                    {
                        drainGifDmaCompatibility();
                    }
                }
                return true;
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

std::optional<size_t> PS2Memory::scratchpadDmaIndex(
    DmacChannel channel) noexcept
{
    switch (channel)
    {
    case DmacChannel::FromScratchpad:
        return 0u;
    case DmacChannel::ToScratchpad:
        return 1u;
    default:
        return std::nullopt;
    }
}

void PS2Memory::publishScratchpadDmaRegisters(
    const ScratchpadDmaState &state)
{
    const uint32_t channelBase =
        dmacChannelBase(state.transfer.channel);
    if (channelBase == 0u)
        return;

    m_ioRegisters[channelBase + 0x00u] = state.chcr;
    m_ioRegisters[channelBase + 0x10u] = state.madr;
    m_ioRegisters[channelBase + 0x20u] =
        state.qwc & 0xFFFFu;
    m_ioRegisters[channelBase + 0x30u] = state.tadr;
    m_ioRegisters[channelBase + 0x40u] = state.asr0;
    m_ioRegisters[channelBase + 0x50u] = state.asr1;
    m_ioRegisters[channelBase + 0x80u] =
        state.sadr & 0x3FF0u;
}

void PS2Memory::clearScratchpadDmaState(
    DmacChannel channel,
    bool notifyRuntime)
{
    const std::optional<size_t> index =
        scratchpadDmaIndex(channel);
    if (!index.has_value())
        return;

    ScratchpadDmaState &state =
        m_scratchpadDma[*index];
    const bool cancelEvent =
        state.active && state.eventManaged;
    state = {};
    if (notifyRuntime && cancelEvent &&
        m_scratchpadDmaCancelCallback)
    {
        m_scratchpadDmaCancelCallback(channel);
    }
}

bool PS2Memory::scheduleScratchpadDma(
    DmacChannel channel,
    uint32_t delayEeCycles)
{
    const std::optional<size_t> index =
        scratchpadDmaIndex(channel);
    if (!index.has_value())
        return false;

    ScratchpadDmaState &state =
        m_scratchpadDma[*index];
    if (!state.active ||
        !m_scratchpadDmaScheduleCallback)
    {
        return false;
    }

    const bool scheduled =
        m_scratchpadDmaScheduleCallback(
            channel, delayEeCycles);
    state.eventManaged = scheduled;
    return scheduled;
}

bool PS2Memory::startScratchpadDma(
    DmacTransferToken transfer,
    uint32_t chcr)
{
    const std::optional<size_t> index =
        scratchpadDmaIndex(transfer.channel);
    if (!index.has_value())
        return false;

    clearScratchpadDmaState(
        transfer.channel, true);
    const uint32_t channelBase =
        dmacChannelBase(transfer.channel);
    ScratchpadDmaState state{};
    state.transfer = transfer;
    state.chcr = chcr | 0x100u;
    state.madr =
        m_ioRegisters[channelBase + 0x10u] &
        0x7FFFFFFFu;
    state.qwc =
        m_ioRegisters[channelBase + 0x20u] &
        0xFFFFu;
    state.tadr =
        m_ioRegisters[channelBase + 0x30u] &
        0x7FFFFFFFu;
    state.asr0 =
        m_ioRegisters[channelBase + 0x40u] &
        0x7FFFFFFFu;
    state.asr1 =
        m_ioRegisters[channelBase + 0x50u] &
        0x7FFFFFFFu;
    state.sadr =
        m_ioRegisters[channelBase + 0x80u] &
        0x3FF0u;
    state.asp =
        static_cast<uint8_t>(
            (state.chcr >> 4u) & 0x3u);
    state.chainMode =
        ((state.chcr >> 2u) & 0x3u) == 1u;
    state.tie =
        (state.chcr & (1u << 7u)) != 0u;
    state.tte =
        (state.chcr & (1u << 6u)) != 0u;
    state.active = true;

    if (((state.chcr >> 2u) & 0x3u) > 1u)
    {
        state.phase = ScratchpadDmaPhase::Fault;
    }
    else if (state.chainMode)
    {
        if (state.qwc != 0u)
        {
            state.tagId = static_cast<uint8_t>(
                (state.chcr >> 28u) & 0x7u);
            state.tagIrq =
                (state.chcr & (1u << 31u)) != 0u;
            state.endAfterPayload =
                transfer.channel ==
                        DmacChannel::ToScratchpad
                    ? state.tagId == 0u ||
                          state.tagId == 7u ||
                          (state.tie && state.tagIrq)
                    : state.tagId == 7u ||
                          (state.tie && state.tagIrq);
            state.phase =
                ScratchpadDmaPhase::TransferPayload;
        }
        else
        {
            state.phase =
                ScratchpadDmaPhase::FetchTag;
        }
    }
    else
    {
        state.endAfterPayload = true;
        state.phase =
            state.qwc != 0u
                ? ScratchpadDmaPhase::TransferPayload
                : ScratchpadDmaPhase::Finalize;
    }

    m_scratchpadDma[*index] = state;
    publishScratchpadDmaRegisters(
        m_scratchpadDma[*index]);
    return true;
}

ScratchpadDmaSnapshot
PS2Memory::scratchpadDmaSnapshot(
    DmacChannel channel) const
{
    ScratchpadDmaSnapshot snapshot{};
    const std::optional<size_t> index =
        scratchpadDmaIndex(channel);
    if (!index.has_value())
        return snapshot;

    const ScratchpadDmaState &state =
        m_scratchpadDma[*index];
    snapshot.transfer = state.transfer;
    snapshot.phase = state.phase;
    snapshot.chcr = state.chcr;
    snapshot.madr = state.madr;
    snapshot.qwc = state.qwc;
    snapshot.tadr = state.tadr;
    snapshot.asr0 = state.asr0;
    snapshot.asr1 = state.asr1;
    snapshot.sadr = state.sadr;
    snapshot.tagsProcessed = state.tagsProcessed;
    snapshot.asp = state.asp;
    snapshot.tagId = state.tagId;
    snapshot.active = state.active;
    snapshot.eventManaged = state.eventManaged;
    snapshot.chainMode = state.chainMode;
    snapshot.tagIrq = state.tagIrq;
    snapshot.endAfterPayload =
        state.endAfterPayload;
    return snapshot;
}

ScratchpadDmaAdvanceResult
PS2Memory::advanceScratchpadDma(
    DmacChannel channel)
{
    constexpr uint32_t kMaximumChainTags = 65536u;
    // PCSX2's SPR engine advances at half EE speed and limits one ordinary
    // payload callback to 0x400 QWC (SPR.cpp, _SPR0chain/_SPR1chain).
    constexpr uint32_t kMaximumSliceQwc = 0x400u;
    constexpr uint32_t kEeCyclesPerQwc = 2u;

    ScratchpadDmaAdvanceResult result{};
    const std::optional<size_t> index =
        scratchpadDmaIndex(channel);
    if (!index.has_value())
        return result;

    ScratchpadDmaState &state =
        m_scratchpadDma[*index];
    result.transfer = state.transfer;
    result.phase = state.phase;
    result.active = state.active;
    if (!state.active)
        return result;

    if (!progressDmacTransfer(state.transfer))
    {
        clearScratchpadDmaState(channel, false);
        result.active = false;
        result.phase = ScratchpadDmaPhase::Idle;
        return result;
    }

    const auto fault =
        [&](const char *reason)
        {
            state.phase = ScratchpadDmaPhase::Fault;
            if (!state.faultReported)
            {
                std::fprintf(
                    stderr,
                    "Scratchpad DMA stalled: channel=%u "
                    "madr=0x%08x qwc=%u tadr=0x%08x "
                    "sadr=0x%04x reason=%s\n",
                    static_cast<unsigned>(channel),
                    state.madr, state.qwc, state.tadr,
                    state.sadr, reason);
                state.faultReported = true;
            }
            result.phase = state.phase;
            result.active = true;
        };

    if (state.phase == ScratchpadDmaPhase::Fault)
    {
        fault("unsupported or invalid state");
        return result;
    }

    if (state.phase ==
        ScratchpadDmaPhase::Finalize)
    {
        state.active = false;
        state.eventManaged = false;
        result.completed =
            requestDmacCompletion(state.transfer);
        result.progressed = result.completed;
        result.active = false;
        result.phase =
            ScratchpadDmaPhase::Finalize;
        return result;
    }

    try
    {
        if (state.phase ==
            ScratchpadDmaPhase::FetchTag)
        {
            if (state.tagsProcessed >=
                kMaximumChainTags)
            {
                fault("tag safety limit reached");
                return result;
            }

            uint32_t tagWord0 = 0u;
            uint32_t tagAddress = 0u;
            if (channel ==
                DmacChannel::ToScratchpad)
            {
                tagWord0 = read32(state.tadr);
                tagAddress =
                    read32(state.tadr + 4u) &
                    0x7FFFFFFFu;
            }
            else
            {
                const uint32_t tagBase =
                    PS2_SCRATCHPAD_BASE +
                    state.sadr;
                tagWord0 = read32(tagBase);
                tagAddress =
                    read32(tagBase + 4u) &
                    0x7FFFFFFFu;
                state.sadr =
                    (state.sadr + 16u) &
                    (PS2_SCRATCHPAD_SIZE - 1u);
            }

            const uint32_t tagQwc =
                tagWord0 & 0xFFFFu;
            state.tagId = static_cast<uint8_t>(
                (tagWord0 >> 28u) & 0x7u);
            state.tagIrq =
                (tagWord0 & (1u << 31u)) != 0u;
            state.chcr =
                (state.chcr & 0x0000FFFFu) |
                (tagWord0 & 0xFFFF0000u);
            ++state.tagsProcessed;

            if (channel ==
                DmacChannel::FromScratchpad)
            {
                if (state.tagId != 0u &&
                    state.tagId != 1u &&
                    state.tagId != 7u)
                {
                    fault("unsupported destination-chain tag");
                    return result;
                }
                state.madr = tagAddress;
                state.qwc = tagQwc;
                state.endAfterPayload =
                    state.tagId == 7u ||
                    (state.tie && state.tagIrq);
            }
            else
            {
                if (state.tte)
                {
                    write128(
                        PS2_SCRATCHPAD_BASE +
                            state.sadr,
                        read128(state.tadr));
                    state.sadr =
                        (state.sadr + 16u) &
                        (PS2_SCRATCHPAD_SIZE - 1u);
                }

                uint32_t dataAddress = tagAddress;
                bool endChain = false;
                switch (state.tagId)
                {
                case 0u: // REFE
                    state.tadr =
                        (state.tadr + 16u) &
                        0x7FFFFFFFu;
                    endChain = true;
                    break;
                case 1u: // CNT
                    dataAddress =
                        (state.tadr + 16u) &
                        0x7FFFFFFFu;
                    state.tadr =
                        (dataAddress +
                         tagQwc * 16u) &
                        0x7FFFFFFFu;
                    break;
                case 2u: // NEXT
                    dataAddress =
                        (state.tadr + 16u) &
                        0x7FFFFFFFu;
                    state.tadr = tagAddress;
                    break;
                case 3u: // REF
                case 4u: // REFS
                    state.tadr =
                        (state.tadr + 16u) &
                        0x7FFFFFFFu;
                    break;
                case 5u: // CALL
                {
                    dataAddress =
                        (state.tadr + 16u) &
                        0x7FFFFFFFu;
                    const uint32_t returnAddress =
                        (dataAddress +
                         tagQwc * 16u) &
                        0x7FFFFFFFu;
                    if (state.asp == 0u)
                    {
                        state.asr0 = returnAddress;
                        state.asp = 1u;
                    }
                    else if (state.asp == 1u)
                    {
                        state.asr1 = returnAddress;
                        state.asp = 2u;
                    }
                    else
                    {
                        endChain = true;
                    }
                    state.tadr = tagAddress;
                    break;
                }
                case 6u: // RET
                    dataAddress =
                        (state.tadr + 16u) &
                        0x7FFFFFFFu;
                    if (state.asp == 2u)
                    {
                        state.tadr = state.asr1;
                        state.asr1 = 0u;
                        state.asp = 1u;
                    }
                    else if (state.asp == 1u)
                    {
                        state.tadr = state.asr0;
                        state.asr0 = 0u;
                        state.asp = 0u;
                    }
                    else
                    {
                        endChain = true;
                    }
                    break;
                case 7u: // END
                    dataAddress =
                        (state.tadr + 16u) &
                        0x7FFFFFFFu;
                    endChain = true;
                    break;
                default:
                    fault("invalid source-chain tag");
                    return result;
                }
                state.chcr =
                    (state.chcr & ~(0x3u << 4u)) |
                    (static_cast<uint32_t>(
                         state.asp)
                     << 4u);
                state.madr = dataAddress;
                state.qwc = tagQwc;
                state.endAfterPayload =
                    endChain ||
                    (state.tie && state.tagIrq);
            }

            state.phase =
                state.qwc != 0u
                    ? ScratchpadDmaPhase::
                          TransferPayload
                    : state.endAfterPayload
                          ? ScratchpadDmaPhase::
                                Finalize
                          : ScratchpadDmaPhase::
                                FetchTag;
            publishScratchpadDmaRegisters(state);

            if (state.qwc == 0u)
            {
                result.progressed = true;
                result.delayEeCycles = 0u;
                result.phase = state.phase;
                result.active = true;
                return result;
            }
        }

        if (state.phase !=
            ScratchpadDmaPhase::TransferPayload)
        {
            fault("unexpected progress phase");
            return result;
        }

        uint32_t sliceQwc =
            std::min(state.qwc, kMaximumSliceQwc);
        if (channel ==
            DmacChannel::FromScratchpad)
        {
            const uint32_t untilWrap =
                0x400u -
                ((state.sadr & 0x3FFFu) >> 4u);
            sliceQwc =
                std::min(sliceQwc, untilWrap);
        }
        if (sliceQwc == 0u)
        {
            fault("zero-sized payload slice");
            return result;
        }

        for (uint32_t offset = 0u;
             offset < sliceQwc; ++offset)
        {
            const uint32_t scratchAddress =
                PS2_SCRATCHPAD_BASE + state.sadr;
            if (channel ==
                DmacChannel::ToScratchpad)
            {
                write128(
                    scratchAddress,
                    read128(state.madr));
            }
            else
            {
                write128(
                    state.madr,
                    read128(scratchAddress));
            }
            state.madr =
                (state.madr + 16u) &
                0x7FFFFFFFu;
            state.sadr =
                (state.sadr + 16u) &
                (PS2_SCRATCHPAD_SIZE - 1u);
        }
        state.qwc -= sliceQwc;
        if (state.qwc == 0u)
        {
            state.phase =
                state.endAfterPayload
                    ? ScratchpadDmaPhase::Finalize
                    : ScratchpadDmaPhase::FetchTag;
        }
        publishScratchpadDmaRegisters(state);

        result.progressed = true;
        result.delayEeCycles =
            sliceQwc >
                    UINT32_MAX /
                        kEeCyclesPerQwc
                ? UINT32_MAX
                : sliceQwc *
                      kEeCyclesPerQwc;
        result.phase = state.phase;
        result.active = true;
        return result;
    }
    catch (const std::exception &error)
    {
        fault(error.what());
        return result;
    }
}

void PS2Memory::drainScratchpadDmaCompatibility(
    DmacChannel channel)
{
    constexpr size_t kMaximumTransitions = 131072u;
    for (size_t transition = 0u;
         transition < kMaximumTransitions;
         ++transition)
    {
        const ScratchpadDmaAdvanceResult result =
            advanceScratchpadDma(channel);
        if (!result.active || result.completed ||
            result.phase == ScratchpadDmaPhase::Fault)
        {
            return;
        }
    }
}

void PS2Memory::updateGifStat()
{
    constexpr uint32_t kGifCtrl = 0x10003000u;
    constexpr uint32_t kGifMode = 0x10003010u;
    constexpr uint32_t kGifStat = 0x10003020u;
    constexpr uint32_t kManagedBits =
        0xFu | (0x1Fu << 24u);

    uint32_t stat = m_ioRegisters[kGifStat] &
                    ~kManagedBits;
    if (m_gifModePath3Masked)
        stat |= 1u << 0u; // M3R
    if (m_path3Masked)
        stat |= 1u << 1u; // M3P
    if ((m_ioRegisters[kGifMode] & 0x4u) != 0u)
        stat |= 1u << 2u; // IMT
    if ((m_ioRegisters[kGifCtrl] & 0x8u) != 0u)
        stat |= 1u << 3u; // PSE
    const uint32_t fifoQwc =
        static_cast<uint32_t>(
            std::min<size_t>(
                m_gifDma.fifoData.size() / 16u,
                16u));
    stat |= fifoQwc << 24u;
    m_ioRegisters[kGifStat] = stat;
}

bool PS2Memory::isGifPaused() const
{
    const auto it =
        m_ioRegisters.find(0x10003000u);
    return it != m_ioRegisters.end() &&
           (it->second & 0x8u) != 0u;
}

void PS2Memory::publishGifDmaRegisters()
{
    constexpr uint32_t kGifBase = 0x1000A000u;
    uint32_t chcr = m_gifDma.chcr;
    if (m_gifDma.active)
        chcr |= 0x100u;
    m_ioRegisters[kGifBase + 0x00u] = chcr;
    m_ioRegisters[kGifBase + 0x10u] =
        m_gifDma.madr;
    m_ioRegisters[kGifBase + 0x20u] =
        m_gifDma.qwc & 0xFFFFu;
    m_ioRegisters[kGifBase + 0x30u] =
        m_gifDma.tadr;
    m_ioRegisters[kGifBase + 0x40u] =
        m_gifDma.asr0;
    m_ioRegisters[kGifBase + 0x50u] =
        m_gifDma.asr1;
    updateGifStat();
}

void PS2Memory::clearGifDmaState(bool notifyRuntime)
{
    if (notifyRuntime &&
        m_gifDma.eventManaged &&
        m_gifDmaCancelCallback)
    {
        m_gifDmaCancelCallback();
    }
    m_gifDma = {};
    updateGifStat();
}

bool PS2Memory::startGifDma(
    DmacTransferToken transfer, uint32_t chcr)
{
    constexpr uint32_t kGifBase = 0x1000A000u;
    clearGifDmaState(false);
    m_gifDma.transfer = transfer;
    m_gifDma.chcr = chcr | 0x100u;
    m_gifDma.madr =
        m_ioRegisters[kGifBase + 0x10u];
    m_gifDma.qwc =
        m_ioRegisters[kGifBase + 0x20u] &
        0xFFFFu;
    m_gifDma.tadr =
        m_ioRegisters[kGifBase + 0x30u];
    m_gifDma.asr0 =
        m_ioRegisters[kGifBase + 0x40u];
    m_gifDma.asr1 =
        m_ioRegisters[kGifBase + 0x50u];
    m_gifDma.asp =
        static_cast<uint8_t>(
            (chcr >> 4u) & 0x3u);
    m_gifDma.chainMode =
        ((chcr >> 2u) & 0x3u) == 1u;
    m_gifDma.tie =
        (chcr & (1u << 7u)) != 0u;
    m_gifDma.active = true;
    m_gifDma.stall =
        !isDmacEnabled()
            ? GifDmaStallReason::DmacDisabled
            : isGifPaused()
                  ? GifDmaStallReason::GifPaused
                  : GifDmaStallReason::None;
    m_gifDma.phase =
        m_gifDma.qwc != 0u
            ? GifDmaPhase::TransferPayload
            : m_gifDma.chainMode
                  ? GifDmaPhase::FetchTag
                  : GifDmaPhase::Finalize;
    m_gifDma.endAfterPayload =
        !m_gifDma.chainMode;
    m_gifDma.visitedStates.reserve(8192u);
    m_gifDma.fifoData.reserve(16u * 16u);
    publishGifDmaRegisters();
    return true;
}

bool PS2Memory::scheduleGifDma(
    uint32_t delayEeCycles)
{
    if (!m_gifDma.active ||
        !m_gifDmaScheduleCallback)
    {
        return false;
    }
    const bool accepted =
        m_gifDmaScheduleCallback(delayEeCycles);
    m_gifDma.eventManaged = accepted;
    return accepted;
}

bool PS2Memory::wakeGifDma(
    uint32_t delayEeCycles)
{
    if (!m_gifDma.active)
        return false;
    if (!isDmacEnabled())
    {
        m_gifDma.stall =
            GifDmaStallReason::DmacDisabled;
        return false;
    }
    if (isGifPaused())
    {
        m_gifDma.stall =
            GifDmaStallReason::GifPaused;
        return false;
    }
    if (isPath3Masked())
    {
        m_gifDma.stall =
            GifDmaStallReason::Path3Masked;
        return false;
    }

    m_gifDma.stall = GifDmaStallReason::None;
    if (scheduleGifDma(delayEeCycles))
        return true;
    drainGifDmaCompatibility();
    return !m_gifDma.active;
}

bool PS2Memory::copyGifDmaPayload(
    uint32_t sourceAddress,
    uint32_t qwc,
    bool toFifo)
{
    if (qwc == 0u)
        return true;

    const uint64_t byteCount64 =
        static_cast<uint64_t>(qwc) * 16ull;
    if (byteCount64 >
        std::numeric_limits<uint32_t>::max())
    {
        return false;
    }
    const uint32_t byteCount =
        static_cast<uint32_t>(byteCount64);
    if (toFifo &&
        m_gifDma.fifoData.size() + byteCount >
            16u * 16u)
    {
        return false;
    }

    uint32_t currentAddress = sourceAddress;
    uint32_t bytesLeft = byteCount;
    bool copied = false;
    while (bytesLeft != 0u)
    {
        const bool scratch =
            isScratchpad(currentAddress);
        uint32_t sourceOffset = 0u;
        try
        {
            sourceOffset =
                translateAddress(currentAddress);
        }
        catch (const std::exception &)
        {
            return false;
        }

        const uint8_t *const base =
            scratch ? m_scratchpad : m_rdram;
        const uint32_t limit =
            scratch ? PS2_SCRATCHPAD_SIZE
                    : PS2_RAM_SIZE;
        if (!base || sourceOffset >= limit)
            return false;

        uint32_t chunk =
            std::min(bytesLeft, limit - sourceOffset);
        chunk &= ~0xFu;
        if (chunk == 0u)
            return false;

        if (toFifo)
        {
            m_gifDma.fifoData.insert(
                m_gifDma.fifoData.end(),
                base + sourceOffset,
                base + sourceOffset + chunk);
        }
        else if (!m_gifArbiter &&
                 m_gifPacketCallback)
        {
            m_gifDma.directCallbackData.insert(
                m_gifDma.directCallbackData.end(),
                base + sourceOffset,
                base + sourceOffset + chunk);
        }
        else
        {
            submitGifPacket(
                GifPathId::Path3,
                base + sourceOffset,
                chunk, false);
        }
        copied = true;
        bytesLeft -= chunk;
        if (bytesLeft == 0u)
            break;
        currentAddress =
            scratch ? PS2_SCRATCHPAD_BASE : 0u;
    }

    if (copied && !m_gifDma.copyCounted)
    {
        m_seenGifCopy = true;
        m_gifCopyCount.fetch_add(
            1u, std::memory_order_relaxed);
        m_gifDma.copyCounted = true;
    }
    if (copied && !toFifo && m_gifArbiter)
        m_gifArbiter->drain();
    return copied;
}

GifDmaAdvanceResult PS2Memory::advanceGifDma()
{
    constexpr uint32_t kMaximumChainTags = 65536u;
    GifDmaAdvanceResult result{};
    result.transfer = m_gifDma.transfer;
    result.phase = m_gifDma.phase;
    result.stall = m_gifDma.stall;
    result.active = m_gifDma.active;

    if (!m_gifDma.active)
        return result;
    if (!progressDmacTransfer(m_gifDma.transfer))
    {
        clearGifDmaState(false);
        result.active = false;
        result.phase = GifDmaPhase::Idle;
        return result;
    }

    const auto fault =
        [&](const char *reason)
        {
            m_gifDma.phase = GifDmaPhase::Fault;
            m_gifDma.stall =
                GifDmaStallReason::Fault;
            if (!m_gifDma.faultReported)
            {
                std::fprintf(
                    stderr,
                    "GIF DMA stalled: madr=0x%08x "
                    "qwc=%u tadr=0x%08x tags=%u "
                    "reason=%s\n",
                    m_gifDma.madr, m_gifDma.qwc,
                    m_gifDma.tadr,
                    m_gifDma.tagsProcessed, reason);
                m_gifDma.faultReported = true;
            }
            result.phase = m_gifDma.phase;
            result.stall = m_gifDma.stall;
            result.active = true;
        };

    if (m_gifDma.phase == GifDmaPhase::Fault)
    {
        fault("unsupported or invalid state");
        return result;
    }
    if (!isDmacEnabled())
    {
        m_gifDma.stall =
            GifDmaStallReason::DmacDisabled;
        result.stall = m_gifDma.stall;
        return result;
    }
    if (isGifPaused())
    {
        m_gifDma.stall =
            GifDmaStallReason::GifPaused;
        result.stall = m_gifDma.stall;
        result.delayEeCycles = 16u;
        return result;
    }

    if (!m_gifDma.fifoData.empty())
    {
        if (isPath3Masked())
        {
            m_gifDma.stall =
                GifDmaStallReason::Path3Masked;
            result.stall = m_gifDma.stall;
            return result;
        }

        const uint32_t fifoQwc =
            static_cast<uint32_t>(
                m_gifDma.fifoData.size() / 16u);
        if (!m_gifArbiter &&
            m_gifPacketCallback)
        {
            m_gifDma.directCallbackData.insert(
                m_gifDma.directCallbackData.end(),
                m_gifDma.fifoData.begin(),
                m_gifDma.fifoData.end());
        }
        else
        {
            submitGifPacket(
                GifPathId::Path3,
                m_gifDma.fifoData.data(),
                static_cast<uint32_t>(
                    m_gifDma.fifoData.size()),
                false);
        }
        if (m_gifArbiter)
            m_gifArbiter->drain();
        m_gifDma.fifoData.clear();
        m_gifDma.stall =
            GifDmaStallReason::None;
        updateGifStat();
        result.progressed = true;
        result.transferredQwc = fifoQwc;
        result.delayEeCycles =
            fifoQwc > UINT32_MAX / 2u
                ? UINT32_MAX
                : fifoQwc * 2u;
        result.phase = m_gifDma.phase;
        result.stall = m_gifDma.stall;
        result.active = true;
        return result;
    }

    if (m_gifDma.phase ==
        GifDmaPhase::Finalize)
    {
        if (!m_gifArbiter &&
            m_gifPacketCallback &&
            m_gifDma.directCallbackData.size() >=
                16u)
        {
            m_gifPacketCallback(
                m_gifDma.directCallbackData.data(),
                static_cast<uint32_t>(
                    std::min<size_t>(
                        m_gifDma
                            .directCallbackData
                            .size(),
                        static_cast<size_t>(
                            UINT32_MAX))));
            m_gifDma.directCallbackData.clear();
        }
        m_gifDma.active = false;
        m_gifDma.eventManaged = false;
        m_gifDma.stall =
            GifDmaStallReason::None;
        result.completed =
            requestDmacCompletion(
                m_gifDma.transfer);
        result.progressed = result.completed;
        result.active = false;
        result.phase = GifDmaPhase::Finalize;
        result.stall = GifDmaStallReason::None;
        return result;
    }

    uint32_t tagCycles = 0u;
    while (m_gifDma.phase ==
           GifDmaPhase::FetchTag)
    {
        if (m_gifDma.tagsProcessed >=
            kMaximumChainTags)
        {
            fault("tag safety limit reached");
            return result;
        }
        const DmacSourceChainKey key{
            m_gifDma.tadr,
            m_gifDma.asr0,
            m_gifDma.asr1,
            m_gifDma.asp};
        if (!m_gifDma.visitedStates.insert(key).second)
        {
            fault("repeated chain state");
            return result;
        }

        const uint32_t tagAddress =
            m_gifDma.tadr;
        const bool scratch =
            isScratchpad(tagAddress);
        uint32_t tagOffset = 0u;
        try
        {
            tagOffset = translateAddress(tagAddress);
        }
        catch (const std::exception &)
        {
            fault("unmapped tag address");
            return result;
        }
        const uint8_t *const base =
            scratch ? m_scratchpad : m_rdram;
        const uint32_t limit =
            scratch ? PS2_SCRATCHPAD_SIZE
                    : PS2_RAM_SIZE;
        if (!base || tagOffset > limit ||
            limit - tagOffset < 16u)
        {
            fault("tag crosses memory boundary");
            return result;
        }

        const uint64_t tag =
            loadScalar<uint64_t>(
                base + tagOffset, 0u, 16u,
                "GIF DMA tag", tagAddress);
        const uint32_t tagQwc =
            static_cast<uint32_t>(
                tag & 0xFFFFu);
        const uint32_t tagAddressField =
            static_cast<uint32_t>(
                (tag >> 32u) & 0x7FFFFFFFu);
        m_gifDma.tagId =
            static_cast<uint8_t>(
                (tag >> 28u) & 0x7u);
        m_gifDma.tagIrq =
            ((tag >> 31u) & 0x1u) != 0u;
        m_gifDma.endAfterPayload = false;
        ++m_gifDma.tagsProcessed;
        tagCycles =
            tagCycles > UINT32_MAX - 2u
                ? UINT32_MAX
                : tagCycles + 2u;

        uint32_t dataAddress =
            tagAddressField;
        switch (m_gifDma.tagId)
        {
        case 0u: // REFE
            m_gifDma.tadr =
                (m_gifDma.tadr + 16u) &
                0x7FFFFFFFu;
            m_gifDma.endAfterPayload = true;
            break;
        case 1u: // CNT
            dataAddress =
                (m_gifDma.tadr + 16u) &
                0x7FFFFFFFu;
            m_gifDma.tadr =
                (dataAddress + tagQwc * 16u) &
                0x7FFFFFFFu;
            break;
        case 2u: // NEXT
            dataAddress =
                (m_gifDma.tadr + 16u) &
                0x7FFFFFFFu;
            m_gifDma.tadr = tagAddressField;
            break;
        case 3u: // REF
        case 4u: // REFS
            m_gifDma.tadr =
                (m_gifDma.tadr + 16u) &
                0x7FFFFFFFu;
            break;
        case 5u: // CALL
        {
            dataAddress =
                (m_gifDma.tadr + 16u) &
                0x7FFFFFFFu;
            const uint32_t returnAddress =
                (dataAddress + tagQwc * 16u) &
                0x7FFFFFFFu;
            if (m_gifDma.asp == 0u)
            {
                m_gifDma.asr0 = returnAddress;
                m_gifDma.asp = 1u;
            }
            else if (m_gifDma.asp == 1u)
            {
                m_gifDma.asr1 = returnAddress;
                m_gifDma.asp = 2u;
            }
            else
            {
                fault("CALL stack overflow");
                return result;
            }
            m_gifDma.tadr = tagAddressField;
            break;
        }
        case 6u: // RET
            dataAddress =
                (m_gifDma.tadr + 16u) &
                0x7FFFFFFFu;
            if (m_gifDma.asp == 2u)
            {
                m_gifDma.tadr =
                    m_gifDma.asr1;
                m_gifDma.asr1 = 0u;
                m_gifDma.asp = 1u;
            }
            else if (m_gifDma.asp == 1u)
            {
                m_gifDma.tadr =
                    m_gifDma.asr0;
                m_gifDma.asr0 = 0u;
                m_gifDma.asp = 0u;
            }
            else
            {
                m_gifDma.endAfterPayload = true;
            }
            break;
        case 7u: // END
            dataAddress =
                (m_gifDma.tadr + 16u) &
                0x7FFFFFFFu;
            m_gifDma.endAfterPayload = true;
            break;
        default:
            fault("invalid tag ID");
            return result;
        }

        if (m_gifDma.tie &&
            m_gifDma.tagIrq)
        {
            m_gifDma.endAfterPayload = true;
        }
        m_gifDma.chcr =
            (m_gifDma.chcr & 0x0000FFFFu) |
            (static_cast<uint32_t>(
                 (tag >> 16u) & 0xFFFFu)
             << 16u);
        m_gifDma.chcr =
            (m_gifDma.chcr & ~(0x3u << 4u)) |
            (static_cast<uint32_t>(
                 m_gifDma.asp)
             << 4u);
        m_gifDma.madr = dataAddress;
        m_gifDma.qwc = tagQwc;
        m_gifDma.phase =
            tagQwc != 0u
                ? GifDmaPhase::TransferPayload
                : m_gifDma.endAfterPayload
                      ? GifDmaPhase::Finalize
                      : GifDmaPhase::FetchTag;
        publishGifDmaRegisters();

        if (m_gifDma.phase ==
            GifDmaPhase::Finalize)
        {
            result.progressed = true;
            result.delayEeCycles = 16u;
            result.phase = m_gifDma.phase;
            result.stall =
                GifDmaStallReason::None;
            result.active = true;
            return result;
        }
    }

    if (m_gifDma.phase !=
        GifDmaPhase::TransferPayload)
    {
        fault("unexpected progress phase");
        return result;
    }

    if (isPath3Masked())
    {
        const uint32_t fifoQwc =
            static_cast<uint32_t>(
                m_gifDma.fifoData.size() / 16u);
        const uint32_t availableQwc =
            fifoQwc < 16u
                ? 16u - fifoQwc
                : 0u;
        const uint32_t sliceQwc =
            std::min(m_gifDma.qwc, availableQwc);
        if (sliceQwc != 0u)
        {
            if (!copyGifDmaPayload(
                    m_gifDma.madr,
                    sliceQwc, true))
            {
                fault("unmapped payload address");
                return result;
            }
            m_gifDma.madr =
                (m_gifDma.madr +
                 sliceQwc * 16u) &
                0x7FFFFFFFu;
            m_gifDma.qwc -= sliceQwc;
            if (m_gifDma.qwc == 0u)
            {
                m_gifDma.phase =
                    m_gifDma.endAfterPayload
                        ? GifDmaPhase::Finalize
                        : GifDmaPhase::FetchTag;
            }
            result.progressed = true;
            result.transferredQwc = sliceQwc;
        }
        m_gifDma.stall =
            GifDmaStallReason::Path3Masked;
        publishGifDmaRegisters();
        result.phase = m_gifDma.phase;
        result.stall = m_gifDma.stall;
        result.active = true;
        return result;
    }

    uint32_t sliceQwc = m_gifDma.qwc;
    const bool intermittent =
        (m_ioRegisters[0x10003010u] &
         0x4u) != 0u;
    if (intermittent)
    {
        sliceQwc =
            sliceQwc > 64u
                ? std::max(1u, sliceQwc / 2u)
                : std::min(sliceQwc, 8u);
    }
    else if (sliceQwc > 8u)
    {
        sliceQwc -= 8u;
    }
    if (sliceQwc == 0u)
    {
        fault("zero-sized payload slice");
        return result;
    }
    if (!copyGifDmaPayload(
            m_gifDma.madr,
            sliceQwc, false))
    {
        fault("unmapped payload address");
        return result;
    }

    m_gifDma.madr =
        (m_gifDma.madr +
         sliceQwc * 16u) &
        0x7FFFFFFFu;
    m_gifDma.qwc -= sliceQwc;
    if (m_gifDma.qwc == 0u)
    {
        m_gifDma.phase =
            m_gifDma.endAfterPayload
                ? GifDmaPhase::Finalize
                : GifDmaPhase::FetchTag;
    }
    m_gifDma.stall = GifDmaStallReason::None;
    publishGifDmaRegisters();

    const uint64_t delay =
        static_cast<uint64_t>(tagCycles) +
        static_cast<uint64_t>(sliceQwc) * 2ull;
    result.progressed = true;
    result.transferredQwc = sliceQwc;
    result.delayEeCycles =
        static_cast<uint32_t>(
            std::min<uint64_t>(
                delay,
                std::numeric_limits<uint32_t>::
                    max()));
    result.phase = m_gifDma.phase;
    result.stall = m_gifDma.stall;
    result.active = true;
    return result;
}

GifDmaSnapshot PS2Memory::gifDmaSnapshot() const
{
    GifDmaSnapshot snapshot{};
    snapshot.transfer = m_gifDma.transfer;
    snapshot.phase = m_gifDma.phase;
    snapshot.stall = m_gifDma.stall;
    snapshot.chcr = m_gifDma.chcr;
    snapshot.madr = m_gifDma.madr;
    snapshot.qwc = m_gifDma.qwc;
    snapshot.tadr = m_gifDma.tadr;
    snapshot.asr0 = m_gifDma.asr0;
    snapshot.asr1 = m_gifDma.asr1;
    snapshot.fifoQwc =
        static_cast<uint32_t>(
            m_gifDma.fifoData.size() / 16u);
    snapshot.tagsProcessed =
        m_gifDma.tagsProcessed;
    snapshot.asp = m_gifDma.asp;
    snapshot.tagId = m_gifDma.tagId;
    snapshot.active = m_gifDma.active;
    snapshot.eventManaged =
        m_gifDma.eventManaged;
    snapshot.chainMode =
        m_gifDma.chainMode;
    snapshot.tagIrq = m_gifDma.tagIrq;
    snapshot.endAfterPayload =
        m_gifDma.endAfterPayload;
    return snapshot;
}

void PS2Memory::drainGifDmaCompatibility()
{
    constexpr uint32_t kMaximumTransitions =
        131072u;
    for (uint32_t transition = 0u;
         transition < kMaximumTransitions &&
         m_gifDma.active;
         ++transition)
    {
        const GifDmaAdvanceResult advance =
            advanceGifDma();
        if (!advance.active || advance.completed ||
            advance.phase == GifDmaPhase::Fault)
        {
            return;
        }
        if (advance.stall !=
            GifDmaStallReason::None)
        {
            return;
        }
        if (!advance.progressed)
            return;
    }
}

void PS2Memory::publishIpuFifoCounts()
{
    constexpr uint32_t kIpuCtrl = 0x10002010u;
    m_ioRegisters[kIpuCtrl] =
        (m_ioRegisters[kIpuCtrl] & ~0xFFu) |
        (m_ipuInputFifoQwc & 0xFu) |
        ((m_ipuOutputFifoQwc & 0xFu) << 4u);
}

void PS2Memory::clearIpuOutputFifo()
{
    m_ipuOutputFifo = {};
    m_ipuOutputFifoReadIndex = 0u;
    m_ipuOutputFifoWriteIndex = 0u;
    m_ipuOutputFifoQwc = 0u;
    publishIpuFifoCounts();
}

uint32_t PS2Memory::readIpuOutputFifo(
    uint8_t *data, uint32_t qwc)
{
    if (!data || qwc == 0u)
        return 0u;

    const uint32_t accepted = std::min(
        qwc, m_ipuOutputFifoQwc);
    for (uint32_t index = 0u;
         index < accepted; ++index)
    {
        std::memcpy(
            data + static_cast<size_t>(index) * 16u,
            m_ipuOutputFifo[
                m_ipuOutputFifoReadIndex]
                .data(),
            16u);
        m_ipuOutputFifoReadIndex =
            (m_ipuOutputFifoReadIndex + 1u) %
            kIpuOutputFifoCapacityQwc;
    }
    m_ipuOutputFifoQwc -= accepted;
    publishIpuFifoCounts();
    return accepted;
}

uint32_t PS2Memory::writeIpuOutputFifo(
    const uint8_t *data, uint32_t qwc)
{
    if (!data || qwc == 0u)
        return 0u;

    const uint32_t available =
        kIpuOutputFifoCapacityQwc -
        std::min(
            m_ipuOutputFifoQwc,
            kIpuOutputFifoCapacityQwc);
    const uint32_t accepted =
        std::min(qwc, available);
    for (uint32_t index = 0u;
         index < accepted; ++index)
    {
        std::memcpy(
            m_ipuOutputFifo[
                m_ipuOutputFifoWriteIndex]
                .data(),
            data + static_cast<size_t>(index) * 16u,
            16u);
        m_ipuOutputFifoWriteIndex =
            (m_ipuOutputFifoWriteIndex + 1u) %
            kIpuOutputFifoCapacityQwc;
    }
    m_ipuOutputFifoQwc += accepted;
    publishIpuFifoCounts();
    if (accepted != 0u &&
        m_fromIpuDma.active &&
        m_fromIpuDma.phase ==
            FromIpuDmaPhase::TransferPayload)
    {
        (void)wakeFromIpuDma(1u);
    }
    return accepted;
}

void PS2Memory::clearIpuInputFifo()
{
    m_ipuInputFifo = {};
    m_ipuInputFifoReadIndex = 0u;
    m_ipuInputFifoWriteIndex = 0u;
    m_ipuInputFifoQwc = 0u;
    publishIpuFifoCounts();
}

uint32_t PS2Memory::writeIpuInputFifo(
    const uint8_t *data, uint32_t qwc)
{
    if (!data || qwc == 0u)
        return 0u;

    const uint32_t available =
        kIpuInputFifoCapacityQwc -
        std::min(
            m_ipuInputFifoQwc,
            kIpuInputFifoCapacityQwc);
    const uint32_t accepted =
        std::min(qwc, available);
    for (uint32_t index = 0u;
         index < accepted; ++index)
    {
        std::memcpy(
            m_ipuInputFifo[
                m_ipuInputFifoWriteIndex]
                .data(),
            data + static_cast<size_t>(index) * 16u,
            16u);
        m_ipuInputFifoWriteIndex =
            (m_ipuInputFifoWriteIndex + 1u) %
            kIpuInputFifoCapacityQwc;
    }
    m_ipuInputFifoQwc += accepted;
    publishIpuFifoCounts();
    return accepted;
}

bool PS2Memory::copyFromIpuDmaPayload(
    uint32_t destinationAddress, uint32_t qwc)
{
    if (qwc == 0u)
        return true;
    if (qwc > m_ipuOutputFifoQwc ||
        qwc > kIpuOutputFifoCapacityQwc)
    {
        return false;
    }

    uint32_t currentAddress = destinationAddress;
    for (uint32_t index = 0u;
         index < qwc; ++index)
    {
        const bool scratchFlag =
            (currentAddress & 0x80000000u) != 0u;
        const uint32_t physicalAddress =
            currentAddress & 0x1FFFFFF0u;
        const bool scratchAlias =
            physicalAddress >= 0x10000000u &&
            physicalAddress < 0x10004000u;
        const bool scratch =
            scratchFlag || scratchAlias;
        if (scratch)
        {
            if (!m_scratchpad)
                return false;
        }
        else if (physicalAddress < PS2_RAM_SIZE)
        {
            if (!m_rdram ||
                physicalAddress >
                    PS2_RAM_SIZE - 16u)
            {
                return false;
            }
        }
        else if (physicalAddress >= 0x10000000u)
        {
            return false;
        }
        currentAddress += 16u;
    }

    std::array<
        uint8_t,
        kIpuOutputFifoCapacityQwc * 16u>
        payload{};
    if (readIpuOutputFifo(
            payload.data(), qwc) != qwc)
    {
        return false;
    }

    currentAddress = destinationAddress;
    for (uint32_t index = 0u;
         index < qwc; ++index)
    {
        const uint32_t physicalAddress =
            currentAddress & 0x1FFFFFF0u;
        const bool scratch =
            (currentAddress & 0x80000000u) != 0u ||
            (physicalAddress >= 0x10000000u &&
             physicalAddress < 0x10004000u);
        if (scratch)
        {
            const uint32_t offset =
                currentAddress & 0x3FF0u;
            std::memcpy(
                m_scratchpad + offset,
                payload.data() +
                    static_cast<size_t>(index) * 16u,
                16u);
        }
        else if (physicalAddress < PS2_RAM_SIZE)
        {
            markModified(
                physicalAddress, 16u, 0u);
            std::memcpy(
                m_rdram + physicalAddress,
                payload.data() +
                    static_cast<size_t>(index) * 16u,
                16u);
        }
        // DMAC writes to unpopulated EE physical memory use the zero-write
        // page and are intentionally discarded.
        currentAddress += 16u;
    }
    return true;
}

void PS2Memory::publishFromIpuDmaRegisters()
{
    constexpr uint32_t kFromIpuBase = 0x1000B000u;
    constexpr uint32_t kDctrl = 0x1000E000u;
    constexpr uint32_t kStadr = 0x1000E060u;
    uint32_t chcr = m_fromIpuDma.chcr;
    if (m_fromIpuDma.active)
        chcr |= 0x100u;
    else
        chcr &= ~0x100u;
    m_ioRegisters[kFromIpuBase + 0x00u] = chcr;
    m_ioRegisters[kFromIpuBase + 0x10u] =
        m_fromIpuDma.madr;
    m_ioRegisters[kFromIpuBase + 0x20u] =
        m_fromIpuDma.qwc & 0xFFFFu;
    const auto dctrlIt = m_ioRegisters.find(kDctrl);
    const uint32_t dctrl =
        dctrlIt != m_ioRegisters.end()
            ? dctrlIt->second
            : 0u;
    if (((dctrl >> 4u) & 0x3u) == 0x3u)
    {
        m_ioRegisters[kStadr] =
            m_fromIpuDma.madr;
    }
    publishIpuFifoCounts();
}

void PS2Memory::clearFromIpuDmaState(
    bool notifyRuntime)
{
    if (notifyRuntime &&
        m_fromIpuDma.eventManaged &&
        m_fromIpuDmaCancelCallback)
    {
        m_fromIpuDmaCancelCallback();
    }
    m_fromIpuDma = {};
}

bool PS2Memory::startFromIpuDma(
    DmacTransferToken transfer, uint32_t chcr)
{
    constexpr uint32_t kFromIpuBase = 0x1000B000u;
    clearFromIpuDmaState(false);
    m_fromIpuDma.transfer = transfer;
    m_fromIpuDma.chcr = chcr | 0x100u;
    m_fromIpuDma.madr =
        m_ioRegisters[kFromIpuBase + 0x10u] &
        0xFFFFFFF0u;
    const uint32_t registerQwc =
        m_ioRegisters[kFromIpuBase + 0x20u] &
        0xFFFFu;
    m_fromIpuDma.zeroQwcStart =
        registerQwc == 0u;
    m_fromIpuDma.qwc =
        m_fromIpuDma.zeroQwcStart
            ? 0x10000u
            : registerQwc;
    m_fromIpuDma.normalMode =
        ((chcr >> 2u) & 0x3u) == 0u;
    m_fromIpuDma.active = true;
    m_fromIpuDma.phase =
        m_fromIpuDma.normalMode
            ? FromIpuDmaPhase::TransferPayload
            : FromIpuDmaPhase::Fault;
    if (!isDmacEnabled())
    {
        m_fromIpuDma.stall =
            FromIpuDmaStallReason::DmacDisabled;
    }
    else if (m_fromIpuDma.normalMode)
    {
        m_fromIpuDma.stall =
            FromIpuDmaStallReason::None;
    }
    else
    {
        m_fromIpuDma.stall =
            FromIpuDmaStallReason::UnsupportedMode;
    }
    publishFromIpuDmaRegisters();
    return true;
}

bool PS2Memory::scheduleFromIpuDma(
    uint32_t delayEeCycles)
{
    if (!m_fromIpuDma.active ||
        !m_fromIpuDmaScheduleCallback)
    {
        return false;
    }
    const bool accepted =
        m_fromIpuDmaScheduleCallback(delayEeCycles);
    m_fromIpuDma.eventManaged = accepted;
    return accepted;
}

bool PS2Memory::wakeFromIpuDma(
    uint32_t delayEeCycles)
{
    if (!m_fromIpuDma.active)
        return false;
    if (!isDmacEnabled())
    {
        m_fromIpuDma.stall =
            FromIpuDmaStallReason::DmacDisabled;
        return false;
    }
    if (!m_fromIpuDma.normalMode)
    {
        m_fromIpuDma.stall =
            FromIpuDmaStallReason::UnsupportedMode;
        return false;
    }
    if (m_fromIpuDma.phase ==
            FromIpuDmaPhase::TransferPayload &&
        m_ipuOutputFifoQwc == 0u &&
        !(m_fromIpuDma.zeroQwcStart &&
          m_fromIpuDma.qwc == 0x10000u))
    {
        m_fromIpuDma.stall =
            FromIpuDmaStallReason::OutputFifoEmpty;
        return false;
    }

    m_fromIpuDma.stall =
        FromIpuDmaStallReason::None;
    if (m_fromIpuDma.eventManaged)
        return true;
    if (scheduleFromIpuDma(delayEeCycles))
        return true;
    drainFromIpuDmaCompatibility();
    return !m_fromIpuDma.active;
}

void PS2Memory::setFromIpuDmaEventPending(
    bool pending) noexcept
{
    if (m_fromIpuDma.active)
        m_fromIpuDma.eventManaged = pending;
}

FromIpuDmaAdvanceResult
PS2Memory::advanceFromIpuDma()
{
    FromIpuDmaAdvanceResult result{};
    result.transfer = m_fromIpuDma.transfer;
    result.phase = m_fromIpuDma.phase;
    result.stall = m_fromIpuDma.stall;
    result.active = m_fromIpuDma.active;

    if (!m_fromIpuDma.active)
        return result;
    if (!progressDmacTransfer(
            m_fromIpuDma.transfer))
    {
        clearFromIpuDmaState(false);
        result.active = false;
        result.phase = FromIpuDmaPhase::Idle;
        return result;
    }
    m_fromIpuDma.eventManaged = false;

    const auto fault =
        [&](const char *reason)
        {
            m_fromIpuDma.phase =
                FromIpuDmaPhase::Fault;
            m_fromIpuDma.stall =
                FromIpuDmaStallReason::Fault;
            if (!m_fromIpuDma.faultReported)
            {
                std::fprintf(
                    stderr,
                    "From-IPU DMA stalled: madr=0x%08x "
                    "qwc=%u fifo=%u reason=%s\n",
                    m_fromIpuDma.madr,
                    m_fromIpuDma.qwc,
                    m_ipuOutputFifoQwc,
                    reason);
                m_fromIpuDma.faultReported = true;
            }
            result.phase = m_fromIpuDma.phase;
            result.stall = m_fromIpuDma.stall;
            result.active = true;
        };

    if (!m_fromIpuDma.normalMode)
    {
        m_fromIpuDma.phase =
            FromIpuDmaPhase::Fault;
        m_fromIpuDma.stall =
            FromIpuDmaStallReason::UnsupportedMode;
        if (!m_fromIpuDma.faultReported)
        {
            std::fprintf(
                stderr,
                "From-IPU DMA stalled: unsupported "
                "CHCR mode (chcr=0x%08x)\n",
                m_fromIpuDma.chcr);
            m_fromIpuDma.faultReported = true;
        }
        result.phase = m_fromIpuDma.phase;
        result.stall = m_fromIpuDma.stall;
        return result;
    }
    if (m_fromIpuDma.phase ==
        FromIpuDmaPhase::Fault)
    {
        fault("invalid state");
        return result;
    }
    if (!isDmacEnabled())
    {
        m_fromIpuDma.stall =
            FromIpuDmaStallReason::DmacDisabled;
        result.stall = m_fromIpuDma.stall;
        return result;
    }

    m_fromIpuDma.stall =
        FromIpuDmaStallReason::None;
    if (m_fromIpuDma.phase ==
        FromIpuDmaPhase::Finalize)
    {
        m_fromIpuDma.active = false;
        result.completed = requestDmacCompletion(
            m_fromIpuDma.transfer);
        result.progressed = result.completed;
        result.active = false;
        result.phase = FromIpuDmaPhase::Finalize;
        return result;
    }
    if (m_fromIpuDma.phase !=
        FromIpuDmaPhase::TransferPayload)
    {
        fault("invalid progress phase");
        return result;
    }

    if (m_ipuOutputFifoQwc == 0u)
    {
        if (m_fromIpuDma.zeroQwcStart &&
            m_fromIpuDma.qwc == 0x10000u)
        {
            m_fromIpuDma.qwc = 0u;
            m_fromIpuDma.phase =
                FromIpuDmaPhase::Finalize;
            publishFromIpuDmaRegisters();
            m_fromIpuDma.active = false;
            result.completed = requestDmacCompletion(
                m_fromIpuDma.transfer);
            result.progressed = result.completed;
            result.active = false;
            result.phase =
                FromIpuDmaPhase::Finalize;
            return result;
        }
        m_fromIpuDma.stall =
            FromIpuDmaStallReason::OutputFifoEmpty;
        result.stall = m_fromIpuDma.stall;
        result.active = true;
        return result;
    }

    const uint32_t sliceQwc = std::min(
        m_fromIpuDma.qwc,
        m_ipuOutputFifoQwc);
    if (sliceQwc == 0u)
    {
        fault("zero-sized payload slice");
        return result;
    }
    if (!copyFromIpuDmaPayload(
            m_fromIpuDma.madr, sliceQwc))
    {
        fault("unmapped payload address");
        return result;
    }

    m_fromIpuDma.zeroQwcStart = false;
    m_fromIpuDma.madr += sliceQwc * 16u;
    m_fromIpuDma.qwc -= sliceQwc;
    result.progressed = true;
    result.transferredQwc = sliceQwc;
    if (m_fromIpuDma.qwc == 0u)
    {
        m_fromIpuDma.phase =
            FromIpuDmaPhase::Finalize;
        result.delayEeCycles =
            sliceQwc > UINT32_MAX / 2u
                ? UINT32_MAX
                : sliceQwc * 2u;
    }
    else
    {
        m_fromIpuDma.stall =
            FromIpuDmaStallReason::OutputFifoEmpty;
    }
    publishFromIpuDmaRegisters();

    result.phase = m_fromIpuDma.phase;
    result.stall = m_fromIpuDma.stall;
    result.active = true;
    return result;
}

FromIpuDmaSnapshot
PS2Memory::fromIpuDmaSnapshot() const
{
    FromIpuDmaSnapshot snapshot{};
    snapshot.transfer = m_fromIpuDma.transfer;
    snapshot.phase = m_fromIpuDma.phase;
    snapshot.stall = m_fromIpuDma.stall;
    snapshot.chcr = m_fromIpuDma.chcr;
    snapshot.madr = m_fromIpuDma.madr;
    snapshot.qwc = m_fromIpuDma.qwc;
    snapshot.fifoQwc = m_ipuOutputFifoQwc;
    snapshot.active = m_fromIpuDma.active;
    snapshot.eventManaged =
        m_fromIpuDma.eventManaged;
    snapshot.normalMode =
        m_fromIpuDma.normalMode;
    return snapshot;
}

void PS2Memory::drainFromIpuDmaCompatibility()
{
    constexpr uint32_t kMaximumTransitions =
        131072u;
    for (uint32_t transition = 0u;
         transition < kMaximumTransitions &&
         m_fromIpuDma.active;
         ++transition)
    {
        const FromIpuDmaAdvanceResult advance =
            advanceFromIpuDma();
        if (!advance.active || advance.completed ||
            advance.phase ==
                FromIpuDmaPhase::Fault)
        {
            return;
        }
        if (advance.stall !=
            FromIpuDmaStallReason::None)
        {
            return;
        }
        if (!advance.progressed)
            return;
    }
}

bool PS2Memory::copyToIpuDmaPayload(
    uint32_t sourceAddress, uint32_t qwc)
{
    if (qwc == 0u)
        return true;
    if (qwc >
        kIpuInputFifoCapacityQwc -
            std::min(
                m_ipuInputFifoQwc,
                kIpuInputFifoCapacityQwc))
    {
        return false;
    }

    std::array<
        uint8_t,
        kIpuInputFifoCapacityQwc * 16u>
        payload{};
    uint32_t currentAddress = sourceAddress;
    for (uint32_t index = 0u;
         index < qwc; ++index)
    {
        const bool scratchFlag =
            (currentAddress & 0x80000000u) != 0u;
        const uint32_t physicalAddress =
            currentAddress & 0x1FFFFFF0u;
        const bool scratchAlias =
            physicalAddress >= 0x10000000u &&
            physicalAddress < 0x10004000u;
        const bool scratch =
            scratchFlag || scratchAlias;
        uint32_t sourceOffset = 0u;
        if (scratch)
        {
            sourceOffset =
                currentAddress & 0x3FF0u;
        }
        else if (physicalAddress < PS2_RAM_SIZE)
        {
            sourceOffset = physicalAddress;
        }
        else if (physicalAddress < 0x10000000u)
        {
            // Unpopulated EE physical memory reads as zero to DMAC.
            currentAddress += 16u;
            continue;
        }
        else
        {
            return false;
        }

        const uint8_t *const base =
            scratch ? m_scratchpad : m_rdram;
        const uint32_t limit =
            scratch ? PS2_SCRATCHPAD_SIZE
                    : PS2_RAM_SIZE;
        if (!base ||
            sourceOffset > limit - 16u)
        {
            return false;
        }
        std::memcpy(
            payload.data() +
                static_cast<size_t>(index) * 16u,
            base + sourceOffset,
            16u);
        currentAddress += 16u;
    }

    return writeIpuInputFifo(
               payload.data(), qwc) == qwc;
}

void PS2Memory::publishToIpuDmaRegisters()
{
    constexpr uint32_t kToIpuBase = 0x1000B400u;
    uint32_t chcr = m_toIpuDma.chcr;
    if (m_toIpuDma.active)
        chcr |= 0x100u;
    m_ioRegisters[kToIpuBase + 0x00u] = chcr;
    m_ioRegisters[kToIpuBase + 0x10u] =
        m_toIpuDma.madr;
    m_ioRegisters[kToIpuBase + 0x20u] =
        m_toIpuDma.qwc & 0xFFFFu;
    m_ioRegisters[kToIpuBase + 0x30u] =
        m_toIpuDma.tadr;
    publishIpuFifoCounts();
}

void PS2Memory::clearToIpuDmaState(
    bool notifyRuntime)
{
    if (notifyRuntime &&
        m_toIpuDma.eventManaged &&
        m_toIpuDmaCancelCallback)
    {
        m_toIpuDmaCancelCallback();
    }
    m_toIpuDma = {};
}

bool PS2Memory::startToIpuDma(
    DmacTransferToken transfer, uint32_t chcr)
{
    constexpr uint32_t kToIpuBase = 0x1000B400u;
    clearToIpuDmaState(false);
    m_toIpuDma.transfer = transfer;
    m_toIpuDma.chcr = chcr | 0x100u;
    m_toIpuDma.madr =
        m_ioRegisters[kToIpuBase + 0x10u] &
        0xFFFFFFF0u;
    m_toIpuDma.tadr =
        m_ioRegisters[kToIpuBase + 0x30u] &
        0xFFFFFFF0u;
    const uint32_t mode = (chcr >> 2u) & 0x3u;
    m_toIpuDma.normalMode = mode == 0u;
    m_toIpuDma.chainMode = mode == 1u;
    m_toIpuDma.tie =
        (m_toIpuDma.chcr & (1u << 7u)) != 0u;
    m_toIpuDma.tagId = static_cast<uint8_t>(
        (m_toIpuDma.chcr >> 28u) & 0x7u);
    m_toIpuDma.tagIrq =
        (m_toIpuDma.chcr & (1u << 31u)) != 0u;
    const uint32_t registerQwc =
        m_ioRegisters[kToIpuBase + 0x20u] &
        0xFFFFu;
    // Normal-mode DMAC starts are transfer-first: a zero QWC underflows
    // after the first QW and therefore represents 0x10000 QW in flight.
    m_toIpuDma.qwc =
        m_toIpuDma.normalMode && registerQwc == 0u
            ? 0x10000u
            : registerQwc;
    m_toIpuDma.endAfterPayload =
        m_toIpuDma.normalMode ||
        (m_toIpuDma.chainMode &&
         registerQwc != 0u &&
         (m_toIpuDma.tagId == 0u ||
          m_toIpuDma.tagId == 7u ||
          (m_toIpuDma.tie &&
           m_toIpuDma.tagIrq)));
    m_toIpuDma.active = true;
    if (m_toIpuDma.normalMode)
    {
        m_toIpuDma.phase =
            ToIpuDmaPhase::TransferPayload;
    }
    else if (m_toIpuDma.chainMode)
    {
        m_toIpuDma.phase =
            registerQwc != 0u
                ? ToIpuDmaPhase::TransferPayload
                : ToIpuDmaPhase::FetchTag;
    }
    else
    {
        m_toIpuDma.phase = ToIpuDmaPhase::Fault;
    }

    if (!isDmacEnabled())
    {
        m_toIpuDma.stall =
            ToIpuDmaStallReason::DmacDisabled;
    }
    else if (m_toIpuDma.normalMode ||
             m_toIpuDma.chainMode)
    {
        m_toIpuDma.stall =
            ToIpuDmaStallReason::None;
    }
    else
    {
        m_toIpuDma.stall =
            ToIpuDmaStallReason::UnsupportedMode;
    }
    publishToIpuDmaRegisters();
    return true;
}

bool PS2Memory::scheduleToIpuDma(
    uint32_t delayEeCycles)
{
    if (!m_toIpuDma.active ||
        !m_toIpuDmaScheduleCallback)
    {
        return false;
    }
    const bool accepted =
        m_toIpuDmaScheduleCallback(delayEeCycles);
    m_toIpuDma.eventManaged = accepted;
    return accepted;
}

bool PS2Memory::wakeToIpuDma(
    uint32_t delayEeCycles)
{
    if (!m_toIpuDma.active)
        return false;
    if (!isDmacEnabled())
    {
        m_toIpuDma.stall =
            ToIpuDmaStallReason::DmacDisabled;
        return false;
    }
    if (!m_toIpuDma.normalMode &&
        !m_toIpuDma.chainMode)
    {
        m_toIpuDma.stall =
            ToIpuDmaStallReason::UnsupportedMode;
        return false;
    }
    if (m_toIpuDma.phase ==
            ToIpuDmaPhase::TransferPayload &&
        m_toIpuDma.qwc != 0u &&
        m_ipuInputFifoQwc >=
            kIpuInputFifoCapacityQwc)
    {
        m_toIpuDma.stall =
            ToIpuDmaStallReason::InputFifoFull;
        return false;
    }

    m_toIpuDma.stall =
        ToIpuDmaStallReason::None;
    if (scheduleToIpuDma(delayEeCycles))
        return true;
    drainToIpuDmaCompatibility();
    return !m_toIpuDma.active;
}

void PS2Memory::setToIpuDmaEventPending(
    bool pending) noexcept
{
    if (m_toIpuDma.active)
        m_toIpuDma.eventManaged = pending;
}

ToIpuDmaAdvanceResult PS2Memory::advanceToIpuDma()
{
    constexpr uint32_t kMaximumChainTags = 65536u;
    ToIpuDmaAdvanceResult result{};
    result.transfer = m_toIpuDma.transfer;
    result.phase = m_toIpuDma.phase;
    result.stall = m_toIpuDma.stall;
    result.active = m_toIpuDma.active;

    if (!m_toIpuDma.active)
        return result;
    if (!progressDmacTransfer(
            m_toIpuDma.transfer))
    {
        clearToIpuDmaState(false);
        result.active = false;
        result.phase = ToIpuDmaPhase::Idle;
        return result;
    }
    // The retained deadline has been consumed. A follow-up callback sets this
    // again after the transition returns; a dormant transition must not keep
    // reporting stale event ownership.
    m_toIpuDma.eventManaged = false;

    const auto fault =
        [&](const char *reason)
        {
            m_toIpuDma.phase =
                ToIpuDmaPhase::Fault;
            m_toIpuDma.stall =
                ToIpuDmaStallReason::Fault;
            if (!m_toIpuDma.faultReported)
            {
                std::fprintf(
                    stderr,
                    "To-IPU DMA stalled: madr=0x%08x "
                    "qwc=%u tadr=0x%08x tags=%u "
                    "fifo=%u reason=%s\n",
                    m_toIpuDma.madr,
                    m_toIpuDma.qwc,
                    m_toIpuDma.tadr,
                    m_toIpuDma.tagsProcessed,
                    m_ipuInputFifoQwc,
                    reason);
                m_toIpuDma.faultReported = true;
            }
            result.phase = m_toIpuDma.phase;
            result.stall = m_toIpuDma.stall;
            result.active = true;
        };

    if (!m_toIpuDma.normalMode &&
        !m_toIpuDma.chainMode)
    {
        m_toIpuDma.phase =
            ToIpuDmaPhase::Fault;
        m_toIpuDma.stall =
            ToIpuDmaStallReason::UnsupportedMode;
        if (!m_toIpuDma.faultReported)
        {
            std::fprintf(
                stderr,
                "To-IPU DMA stalled: unsupported "
                "CHCR mode (chcr=0x%08x)\n",
                m_toIpuDma.chcr);
            m_toIpuDma.faultReported = true;
        }
        result.phase = m_toIpuDma.phase;
        result.stall = m_toIpuDma.stall;
        return result;
    }
    if (m_toIpuDma.phase ==
        ToIpuDmaPhase::Fault)
    {
        fault("invalid state");
        return result;
    }
    if (!isDmacEnabled())
    {
        m_toIpuDma.stall =
            ToIpuDmaStallReason::DmacDisabled;
        result.stall = m_toIpuDma.stall;
        return result;
    }

    m_toIpuDma.stall =
        ToIpuDmaStallReason::None;

    if (m_toIpuDma.phase ==
        ToIpuDmaPhase::Finalize)
    {
        m_toIpuDma.active = false;
        m_toIpuDma.eventManaged = false;
        m_toIpuDma.stall =
            ToIpuDmaStallReason::None;
        result.completed =
            requestDmacCompletion(
                m_toIpuDma.transfer);
        result.progressed = result.completed;
        result.active = false;
        result.phase = ToIpuDmaPhase::Finalize;
        result.stall =
            ToIpuDmaStallReason::None;
        return result;
    }

    // A full input FIFO withdraws the IPU data request before the channel
    // reads another source-chain tag. It remains dormant until the IPU
    // consumes or resets the FIFO.
    if (m_ipuInputFifoQwc >=
        kIpuInputFifoCapacityQwc)
    {
        m_toIpuDma.stall =
            ToIpuDmaStallReason::InputFifoFull;
        result.stall = m_toIpuDma.stall;
        result.active = true;
        return result;
    }

    uint32_t tagCycles = 0u;
    if (m_toIpuDma.phase ==
        ToIpuDmaPhase::FetchTag)
    {
        if (!m_toIpuDma.chainMode)
        {
            fault("normal transfer entered tag phase");
            return result;
        }
        if (m_toIpuDma.tagsProcessed >=
            kMaximumChainTags)
        {
            fault("tag safety limit reached");
            return result;
        }

        const uint32_t tagAddress =
            m_toIpuDma.tadr & 0xFFFFFFF0u;
        const uint32_t physicalAddress =
            tagAddress & 0x1FFFFFF0u;
        const bool scratch =
            (tagAddress & 0x80000000u) != 0u ||
            (physicalAddress >= 0x10000000u &&
             physicalAddress < 0x10004000u);
        static constexpr std::array<uint8_t, 16u>
            kDmacZeroRead{};
        const uint8_t *base = nullptr;
        uint32_t tagOffset = 0u;
        uint32_t limit = 0u;
        if (scratch)
        {
            base = m_scratchpad;
            tagOffset = tagAddress & 0x3FF0u;
            limit = PS2_SCRATCHPAD_SIZE;
        }
        else if (physicalAddress < PS2_RAM_SIZE)
        {
            base = m_rdram;
            tagOffset = physicalAddress;
            limit = PS2_RAM_SIZE;
        }
        else if (physicalAddress < 0x10000000u)
        {
            // Unpopulated EE physical memory uses DMAC's zero-read page.
            base = kDmacZeroRead.data();
            tagOffset = 0u;
            limit =
                static_cast<uint32_t>(
                    kDmacZeroRead.size());
        }
        if (!base || tagOffset > limit ||
            limit - tagOffset < 16u)
        {
            fault("tag crosses memory boundary");
            return result;
        }

        const uint64_t tag =
            loadScalar<uint64_t>(
                base + tagOffset, 0u, 16u,
                "To-IPU DMA tag", tagAddress);
        const uint32_t tagQwc =
            static_cast<uint32_t>(
                tag & 0xFFFFu);
        const uint32_t tagAddressField =
            static_cast<uint32_t>(
                tag >> 32u) &
            0xFFFFFFF0u;
        m_toIpuDma.tagId =
            static_cast<uint8_t>(
                (tag >> 28u) & 0x7u);
        m_toIpuDma.tagIrq =
            ((tag >> 31u) & 0x1u) != 0u;
        m_toIpuDma.endAfterPayload = false;
        ++m_toIpuDma.tagsProcessed;
        tagCycles = 1u;

        uint32_t dataAddress =
            tagAddressField;
        switch (m_toIpuDma.tagId)
        {
        case 0u: // REFE
            m_toIpuDma.tadr += 16u;
            m_toIpuDma.endAfterPayload = true;
            break;
        case 1u: // CNT
            dataAddress = m_toIpuDma.tadr + 16u;
            m_toIpuDma.tadr = dataAddress;
            break;
        case 2u: // NEXT
            dataAddress = m_toIpuDma.tadr + 16u;
            m_toIpuDma.tadr = tagAddressField;
            break;
        case 3u: // REF
        case 4u: // REFS
            m_toIpuDma.tadr += 16u;
            break;
        case 5u: // CALL is terminal on the IPU source chain.
        case 6u: // RET is terminal on the IPU source chain.
            // PCSX2 intentionally uses the tag address as payload and does
            // not apply the general source-chain CALL/RET stack here.
            m_toIpuDma.endAfterPayload = true;
            break;
        case 7u: // END
            dataAddress = m_toIpuDma.tadr + 16u;
            m_toIpuDma.endAfterPayload = true;
            break;
        default:
            fault("invalid tag ID");
            return result;
        }

        if (m_toIpuDma.tie &&
            m_toIpuDma.tagIrq)
        {
            m_toIpuDma.endAfterPayload = true;
        }
        m_toIpuDma.chcr =
            (m_toIpuDma.chcr & 0x0000FFFFu) |
            (static_cast<uint32_t>(
                 (tag >> 16u) & 0xFFFFu)
             << 16u);
        m_toIpuDma.madr = dataAddress;
        m_toIpuDma.qwc = tagQwc;
        m_toIpuDma.phase =
            tagQwc != 0u
                ? ToIpuDmaPhase::TransferPayload
                : m_toIpuDma.endAfterPayload
                      ? ToIpuDmaPhase::Finalize
                      : ToIpuDmaPhase::FetchTag;
        publishToIpuDmaRegisters();
        result.progressed = true;
    }

    // A zero-QWC chain tag always yields before the next tag or final
    // publication. PCSX2 charges one tag cycle in addition to its four-cycle
    // minimum, then converts the combined work at two EE cycles per unit.
    if (tagCycles != 0u &&
        (m_toIpuDma.phase ==
             ToIpuDmaPhase::FetchTag ||
         m_toIpuDma.phase ==
             ToIpuDmaPhase::Finalize))
    {
        result.delayEeCycles =
            (4u + tagCycles) * 2u;
        result.phase = m_toIpuDma.phase;
        result.stall =
            ToIpuDmaStallReason::None;
        result.active = true;
        return result;
    }

    if (m_toIpuDma.phase !=
        ToIpuDmaPhase::TransferPayload)
    {
        fault("invalid progress phase");
        return result;
    }

    const uint32_t available =
        kIpuInputFifoCapacityQwc -
        std::min(
            m_ipuInputFifoQwc,
            kIpuInputFifoCapacityQwc);
    const uint32_t sliceQwc =
        std::min(m_toIpuDma.qwc, available);
    if (sliceQwc != 0u &&
        !copyToIpuDmaPayload(
            m_toIpuDma.madr, sliceQwc))
    {
        fault("unmapped payload address");
        return result;
    }

    m_toIpuDma.madr += sliceQwc * 16u;
    m_toIpuDma.qwc -= sliceQwc;
    if (m_toIpuDma.chainMode &&
        m_toIpuDma.tagId == 1u)
    {
        // IPU CNT chains publish TADR alongside partial MADR progress.
        m_toIpuDma.tadr = m_toIpuDma.madr;
    }
    result.progressed =
        result.progressed || sliceQwc != 0u;
    result.transferredQwc = sliceQwc;
    if (m_toIpuDma.qwc == 0u)
    {
        m_toIpuDma.phase =
            m_toIpuDma.endAfterPayload
                ? ToIpuDmaPhase::Finalize
                : ToIpuDmaPhase::FetchTag;
    }

    const bool terminalPayloadComplete =
        m_toIpuDma.phase ==
        ToIpuDmaPhase::Finalize;
    if (sliceQwc == 0u ||
        terminalPayloadComplete)
    {
        const uint32_t delayUnits =
            std::max(4u, sliceQwc) +
            tagCycles;
        result.delayEeCycles =
            delayUnits * 2u;
        m_toIpuDma.stall =
            ToIpuDmaStallReason::None;
    }
    else
    {
        m_toIpuDma.stall =
            m_toIpuDma.qwc != 0u &&
                    m_ipuInputFifoQwc >=
                        kIpuInputFifoCapacityQwc
                ? ToIpuDmaStallReason::
                      InputFifoFull
                : ToIpuDmaStallReason::
                      WaitingForInputRequest;
    }
    publishToIpuDmaRegisters();

    result.phase = m_toIpuDma.phase;
    result.stall = m_toIpuDma.stall;
    result.active = true;
    return result;
}

ToIpuDmaSnapshot
PS2Memory::toIpuDmaSnapshot() const
{
    ToIpuDmaSnapshot snapshot{};
    snapshot.transfer = m_toIpuDma.transfer;
    snapshot.phase = m_toIpuDma.phase;
    snapshot.stall = m_toIpuDma.stall;
    snapshot.chcr = m_toIpuDma.chcr;
    snapshot.madr = m_toIpuDma.madr;
    snapshot.qwc = m_toIpuDma.qwc;
    snapshot.tadr = m_toIpuDma.tadr;
    snapshot.fifoQwc = m_ipuInputFifoQwc;
    snapshot.tagsProcessed =
        m_toIpuDma.tagsProcessed;
    snapshot.tagId = m_toIpuDma.tagId;
    snapshot.active = m_toIpuDma.active;
    snapshot.eventManaged =
        m_toIpuDma.eventManaged;
    snapshot.normalMode =
        m_toIpuDma.normalMode;
    snapshot.chainMode =
        m_toIpuDma.chainMode;
    snapshot.tagIrq = m_toIpuDma.tagIrq;
    snapshot.endAfterPayload =
        m_toIpuDma.endAfterPayload;
    return snapshot;
}

void PS2Memory::drainToIpuDmaCompatibility()
{
    constexpr uint32_t kMaximumTransitions =
        131072u;
    for (uint32_t transition = 0u;
         transition < kMaximumTransitions &&
         m_toIpuDma.active;
         ++transition)
    {
        const ToIpuDmaAdvanceResult advance =
            advanceToIpuDma();
        if (!advance.active || advance.completed ||
            advance.phase == ToIpuDmaPhase::Fault)
        {
            return;
        }
        if (advance.stall !=
            ToIpuDmaStallReason::None)
        {
            return;
        }
        if (!advance.progressed)
            return;
    }
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
    if (channel == DmacChannel::Gif)
        clearGifDmaState(true);
    if (channel == DmacChannel::FromIpu)
        clearFromIpuDmaState(true);
    if (channel == DmacChannel::ToIpu)
        clearToIpuDmaState(true);
    if (channel == DmacChannel::Vif0)
        clearVif0DmaState(true);
    if (channel == DmacChannel::Vif1)
        clearVif1DmaState(true);
    if (channel == DmacChannel::FromScratchpad ||
        channel == DmacChannel::ToScratchpad)
    {
        clearScratchpadDmaState(channel, true);
    }
}

DmacTransferToken PS2Memory::beginDmacTransfer(DmacChannel channel)
{
    const size_t index = static_cast<size_t>(channel);
    if (index >= m_dmacChannels.size())
    {
        return {};
    }

    discardPendingDmacWork(channel);

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

bool PS2Memory::isDmacEnabled() const
{
    constexpr uint32_t kDctrl = 0x1000E000u;
    const auto found = m_ioRegisters.find(kDctrl);
    return found == m_ioRegisters.end() ||
           (found->second & 0x1u) != 0u;
}

bool PS2Memory::isVif1GifPathAvailable(
    bool directHl) const
{
    return !m_vif1GifPathAvailableCallback ||
           m_vif1GifPathAvailableCallback(directHl);
}

void PS2Memory::updateVif0Fqc(uint32_t qwc)
{
    constexpr uint32_t kFqcMask = 0xFu << 24u;
    const uint32_t fqc = std::min(qwc, 8u);
    vif0_regs.stat =
        (vif0_regs.stat & ~kFqcMask) |
        (fqc << 24u);
}

void PS2Memory::publishVif0DmaRegisters()
{
    constexpr uint32_t kVif0Base = 0x10008000u;
    m_ioRegisters[kVif0Base + 0x10u] =
        m_vif0Dma.madr;
    m_ioRegisters[kVif0Base + 0x20u] =
        m_vif0Dma.qwc;
    m_ioRegisters[kVif0Base + 0x30u] =
        m_vif0Dma.tadr;
    m_ioRegisters[kVif0Base + 0x40u] =
        m_vif0Dma.asr0;
    m_ioRegisters[kVif0Base + 0x50u] =
        m_vif0Dma.asr1;
    updateVif0Fqc(m_vif0Dma.qwc);
}

void PS2Memory::clearVif0DmaState(
    bool notifyRuntime)
{
    const bool wasEventManaged =
        m_vif0Dma.active &&
        m_vif0Dma.eventManaged;
    m_vif0Dma = {};
    updateVif0Fqc(0u);
    if (notifyRuntime && wasEventManaged &&
        m_vif0DmaCancelCallback)
    {
        m_vif0DmaCancelCallback();
    }
}

bool PS2Memory::scheduleVif0Dma(
    uint32_t delayEeCycles)
{
    if (!m_vif0Dma.active ||
        !m_vif0DmaScheduleCallback)
    {
        return false;
    }

    const bool scheduled =
        m_vif0DmaScheduleCallback(delayEeCycles);
    m_vif0Dma.eventManaged = scheduled;
    return scheduled;
}

bool PS2Memory::startVif0Dma(
    DmacTransferToken transfer, uint32_t chcr)
{
    constexpr uint32_t kVif0Base = 0x10008000u;

    clearVif0DmaState(true);
    Vif0DmaState state{};
    state.transfer = transfer;
    state.madr = m_ioRegisters[kVif0Base + 0x10u];
    state.qwc =
        m_ioRegisters[kVif0Base + 0x20u] & 0xFFFFu;
    state.tadr = m_ioRegisters[kVif0Base + 0x30u];
    state.asr0 = m_ioRegisters[kVif0Base + 0x40u];
    state.asr1 = m_ioRegisters[kVif0Base + 0x50u];
    state.asp =
        static_cast<uint8_t>((chcr >> 4u) & 0x3u);
    state.chainMode =
        ((chcr >> 2u) & 0x3u) == 1u;
    state.tie = (chcr & (1u << 7u)) != 0u;
    state.tte = (chcr & (1u << 6u)) != 0u;
    state.active = true;

    if (state.chainMode)
    {
        if (state.qwc != 0u)
        {
            state.tagId =
                static_cast<uint8_t>(
                    (chcr >> 28u) & 0x7u);
            state.tagIrq =
                (chcr & (1u << 31u)) != 0u;
            state.endAfterPayload =
                state.tagId == 0u ||
                state.tagId == 7u ||
                (state.tie && state.tagIrq);
            state.phase =
                Vif0DmaPhase::TransferPayload;
        }
        else
        {
            state.phase = Vif0DmaPhase::FetchTag;
        }
    }
    else
    {
        state.endAfterPayload = true;
        state.phase =
            state.qwc != 0u
                ? Vif0DmaPhase::TransferPayload
                : Vif0DmaPhase::Finalize;
    }

    m_vif0Dma = std::move(state);
    publishVif0DmaRegisters();
    return true;
}

Vif0DmaStallReason
PS2Memory::currentVif0DmaStall() const
{
    if (m_vif0Dma.phase == Vif0DmaPhase::Fault)
        return Vif0DmaStallReason::Fault;
    if (!isDmacEnabled())
        return Vif0DmaStallReason::DmacDisabled;
    if ((vif0_regs.stat & (1u << 9u)) != 0u)
        return Vif0DmaStallReason::ForceBreak;
    if ((vif0_regs.stat & (1u << 8u)) != 0u)
        return Vif0DmaStallReason::VifStop;
    if ((vif0_regs.stat & (1u << 10u)) != 0u)
        return Vif0DmaStallReason::VifInterrupt;
    if (m_vif0WaitingForVu)
        return Vif0DmaStallReason::WaitingForVu;
    return Vif0DmaStallReason::None;
}

Vif0DmaSnapshot PS2Memory::vif0DmaSnapshot() const
{
    Vif0DmaSnapshot snapshot{};
    snapshot.transfer = m_vif0Dma.transfer;
    snapshot.phase = m_vif0Dma.phase;
    snapshot.stall = m_vif0Dma.stall;
    snapshot.madr = m_vif0Dma.madr;
    snapshot.qwc = m_vif0Dma.qwc;
    snapshot.tadr = m_vif0Dma.tadr;
    snapshot.asr0 = m_vif0Dma.asr0;
    snapshot.asr1 = m_vif0Dma.asr1;
    snapshot.parserBytes =
        static_cast<uint32_t>(
            std::min<size_t>(
                m_vif0DeferredData.size(),
                static_cast<size_t>(UINT32_MAX)));
    snapshot.bufferedPayloadBytes =
        m_vif0Dma.bufferedPayloadBytes;
    snapshot.payloadByteOffset =
        m_vif0Dma.payloadByteOffset;
    snapshot.tagsProcessed =
        m_vif0Dma.tagsProcessed;
    snapshot.asp = m_vif0Dma.asp;
    snapshot.tagId = m_vif0Dma.tagId;
    snapshot.active = m_vif0Dma.active;
    snapshot.eventManaged =
        m_vif0Dma.eventManaged;
    snapshot.chainMode = m_vif0Dma.chainMode;
    snapshot.tagIrq = m_vif0Dma.tagIrq;
    snapshot.endAfterPayload =
        m_vif0Dma.endAfterPayload;
    return snapshot;
}

bool PS2Memory::wakeVif0Dma()
{
    if (!m_vif0Dma.active)
        return false;

    m_vif0Dma.stall = Vif0DmaStallReason::None;
    if (m_vif0Dma.eventManaged &&
        scheduleVif0Dma(0u))
    {
        return true;
    }

    drainVif0DmaCompatibility();
    return true;
}

Vif0DmaAdvanceResult PS2Memory::advanceVif0Dma()
{
    constexpr uint32_t kVif0Base = 0x10008000u;
    constexpr uint32_t kMaximumChainTags = 65536u;

    Vif0DmaAdvanceResult result{};
    result.transfer = m_vif0Dma.transfer;
    result.phase = m_vif0Dma.phase;
    result.active = m_vif0Dma.active;

    if (!m_vif0Dma.active)
        return result;

    if (!progressDmacTransfer(m_vif0Dma.transfer))
    {
        clearVif0DmaState(false);
        result.active = false;
        result.phase = Vif0DmaPhase::Idle;
        return result;
    }

    m_vif0Dma.stall = currentVif0DmaStall();
    if (m_vif0Dma.stall !=
        Vif0DmaStallReason::None)
    {
        result.stall = m_vif0Dma.stall;
        return result;
    }

    const auto parserStall =
        [](Vif0ParserDisposition disposition)
        {
            switch (disposition)
            {
            case Vif0ParserDisposition::WaitingForVu:
                return Vif0DmaStallReason::WaitingForVu;
            case Vif0ParserDisposition::VifInterrupt:
                return Vif0DmaStallReason::VifInterrupt;
            case Vif0ParserDisposition::Drained:
            case Vif0ParserDisposition::NeedMoreData:
                return Vif0DmaStallReason::None;
            }
            return Vif0DmaStallReason::None;
        };

    const auto fault =
        [&](const char *reason)
        {
            m_vif0Dma.phase = Vif0DmaPhase::Fault;
            m_vif0Dma.stall =
                Vif0DmaStallReason::Fault;
            if (!m_vif0Dma.faultReported)
            {
                std::fprintf(
                    stderr,
                    "VIF0 DMA stalled: tadr=0x%08x "
                    "madr=0x%08x tags=%u reason=%s\n",
                    m_vif0Dma.tadr, m_vif0Dma.madr,
                    m_vif0Dma.tagsProcessed, reason);
                m_vif0Dma.faultReported = true;
            }
            result.phase = m_vif0Dma.phase;
            result.stall = m_vif0Dma.stall;
            result.active = true;
        };

    if (m_vif0Dma.phase == Vif0DmaPhase::Finalize)
    {
        updateVif0Fqc(0u);
        m_vif0Dma.active = false;
        m_vif0Dma.eventManaged = false;
        m_vif0Dma.stall =
            Vif0DmaStallReason::None;
        result.completed =
            requestDmacCompletion(m_vif0Dma.transfer);
        result.progressed = result.completed;
        result.active = false;
        result.phase = Vif0DmaPhase::Finalize;
        return result;
    }

    if (m_vif0Dma.phase == Vif0DmaPhase::FetchTag)
    {
        if (m_vif0Dma.tagsProcessed >=
            kMaximumChainTags)
        {
            fault("tag safety limit reached");
            return result;
        }

        const DmacSourceChainKey key{
            m_vif0Dma.tadr,
            m_vif0Dma.asr0,
            m_vif0Dma.asr1,
            m_vif0Dma.asp};
        if (!m_vif0Dma.visitedStates.insert(key).second)
        {
            fault("repeated chain state");
            return result;
        }

        const uint32_t tagAddress = m_vif0Dma.tadr;
        const bool scratch = isScratchpad(tagAddress);
        uint32_t tagOffset = 0u;
        try
        {
            tagOffset = translateAddress(tagAddress);
        }
        catch (const std::exception &)
        {
            fault("unmapped tag address");
            return result;
        }

        const uint8_t *const base =
            scratch ? m_scratchpad : m_rdram;
        const uint32_t limit =
            scratch ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
        if (!base || tagOffset > limit ||
            limit - tagOffset < 16u)
        {
            fault("tag crosses memory boundary");
            return result;
        }

        const uint8_t *const tagData = base + tagOffset;
        const uint64_t tag = loadScalar<uint64_t>(
            tagData, 0u, 16u, "VIF0 DMA tag",
            tagAddress);
        const uint32_t tagQwc =
            static_cast<uint32_t>(tag & 0xFFFFu);
        const uint32_t tagAddressField =
            static_cast<uint32_t>(
                (tag >> 32u) & 0x7FFFFFFFu);
        m_vif0Dma.tagId =
            static_cast<uint8_t>((tag >> 28u) & 0x7u);
        m_vif0Dma.tagIrq =
            ((tag >> 31u) & 0x1u) != 0u;
        m_vif0Dma.endAfterPayload = false;
        ++m_vif0Dma.tagsProcessed;

        uint32_t dataAddress = 0u;
        switch (m_vif0Dma.tagId)
        {
        case 0u: // REFE
            dataAddress = tagAddressField;
            m_vif0Dma.tadr += 16u;
            m_vif0Dma.endAfterPayload = true;
            break;
        case 1u: // CNT
            m_vif0Dma.tadr += 16u;
            dataAddress = m_vif0Dma.tadr;
            break;
        case 2u: // NEXT
            dataAddress = m_vif0Dma.tadr + 16u;
            m_vif0Dma.tadr = tagAddressField;
            break;
        case 3u: // REF
        case 4u: // REFS
            dataAddress = tagAddressField;
            m_vif0Dma.tadr += 16u;
            break;
        case 5u: // CALL
        {
            dataAddress = m_vif0Dma.tadr + 16u;
            const uint32_t returnAddress =
                dataAddress + tagQwc * 16u;
            if (m_vif0Dma.asp == 0u)
            {
                m_vif0Dma.asr0 = returnAddress;
                m_vif0Dma.asp = 1u;
            }
            else if (m_vif0Dma.asp == 1u)
            {
                m_vif0Dma.asr1 = returnAddress;
                m_vif0Dma.asp = 2u;
            }
            else
            {
                fault("CALL stack overflow");
                return result;
            }
            m_vif0Dma.tadr = tagAddressField;
            break;
        }
        case 6u: // RET
            dataAddress = m_vif0Dma.tadr + 16u;
            if (m_vif0Dma.asp == 2u)
            {
                m_vif0Dma.tadr = m_vif0Dma.asr1;
                m_vif0Dma.asr1 = 0u;
                m_vif0Dma.asp = 1u;
            }
            else if (m_vif0Dma.asp == 1u)
            {
                m_vif0Dma.tadr = m_vif0Dma.asr0;
                m_vif0Dma.asr0 = 0u;
                m_vif0Dma.asp = 0u;
            }
            else
            {
                m_vif0Dma.endAfterPayload = true;
            }
            break;
        case 7u: // END
            dataAddress = m_vif0Dma.tadr + 16u;
            m_vif0Dma.endAfterPayload = true;
            break;
        default:
            fault("invalid tag ID");
            return result;
        }

        if (m_vif0Dma.tie && m_vif0Dma.tagIrq)
            m_vif0Dma.endAfterPayload = true;

        m_vif0Dma.madr = dataAddress;
        m_vif0Dma.qwc = tagQwc;
        m_vif0Dma.payloadByteOffset = 0u;
        m_vif0Dma.bufferedPayloadBytes = 0u;
        uint32_t chcr = m_ioRegisters[kVif0Base];
        chcr = (chcr & 0x0000FFFFu) |
               (static_cast<uint32_t>(
                    (tag >> 16u) & 0xFFFFu)
                << 16u);
        chcr = (chcr & ~(0x3u << 4u)) |
               (static_cast<uint32_t>(
                    m_vif0Dma.asp)
                << 4u);
        m_ioRegisters[kVif0Base] = chcr;
        publishVif0DmaRegisters();

        uint32_t delay = 1u;
        if (m_vif0Dma.tte)
        {
            appendVif0Stream(tagData + 8u, 8u);
            const Vif0ParserResult parsed =
                processVif0Stream();
            m_vif0Dma.stall =
                parserStall(parsed.disposition);
            ++delay;
        }

        m_vif0Dma.phase =
            m_vif0Dma.qwc != 0u
                ? Vif0DmaPhase::TransferPayload
                : (m_vif0Dma.endAfterPayload
                       ? Vif0DmaPhase::Finalize
                       : Vif0DmaPhase::FetchTag);
        result.progressed = true;
        result.delayEeCycles = delay;
    }
    else if (
        m_vif0Dma.phase ==
        Vif0DmaPhase::TransferPayload)
    {
        const uint32_t priorByteOffset =
            m_vif0Dma.payloadByteOffset;
        const uint64_t remainingBytes64 =
            static_cast<uint64_t>(m_vif0Dma.qwc) *
                16ull -
            priorByteOffset;
        if (remainingBytes64 >
            std::numeric_limits<uint32_t>::max())
        {
            fault("payload byte count overflow");
            return result;
        }
        const uint32_t remainingBytes =
            static_cast<uint32_t>(remainingBytes64);

        uint32_t acceptedBytes = 0u;
        if (m_vif0Dma.bufferedPayloadBytes != 0u)
        {
            const Vif0ParserResult parsed =
                processVif0Stream();
            m_vif0Dma.stall =
                parserStall(parsed.disposition);
            if (m_vif0Dma.stall ==
                Vif0DmaStallReason::None)
            {
                acceptedBytes =
                    m_vif0Dma.bufferedPayloadBytes;
            }
            else
            {
                acceptedBytes = std::min(
                    parsed.consumedBytes,
                    m_vif0Dma.bufferedPayloadBytes);
            }
            m_vif0Dma.bufferedPayloadBytes -=
                acceptedBytes;
        }
        else if (remainingBytes != 0u)
        {
            const size_t oldParserBytes =
                m_vif0DeferredData.size();
            uint32_t sourceAddress =
                m_vif0Dma.madr + priorByteOffset;
            bool scratch = isScratchpad(sourceAddress);
            uint32_t sourceOffset = 0u;
            try
            {
                sourceOffset =
                    translateAddress(sourceAddress);
            }
            catch (const std::exception &)
            {
                fault("unmapped payload address");
                return result;
            }

            const uint8_t *const base =
                scratch ? m_scratchpad : m_rdram;
            const uint32_t limit =
                scratch ? PS2_SCRATCHPAD_SIZE
                        : PS2_RAM_SIZE;
            uint32_t bytesLeft = remainingBytes;
            while (bytesLeft != 0u)
            {
                if (sourceOffset >= limit)
                    sourceOffset = 0u;
                const uint32_t chunk = std::min(
                    bytesLeft, limit - sourceOffset);
                if (chunk == 0u)
                {
                    fault("empty payload memory region");
                    return result;
                }
                appendVif0Stream(
                    base + sourceOffset, chunk);
                sourceOffset += chunk;
                bytesLeft -= chunk;
            }

            const Vif0ParserResult parsed =
                processVif0Stream();
            m_vif0Dma.stall =
                parserStall(parsed.disposition);
            if (m_vif0Dma.stall ==
                Vif0DmaStallReason::None)
            {
                acceptedBytes = remainingBytes;
            }
            else
            {
                const uint32_t oldBytes =
                    static_cast<uint32_t>(
                        std::min<size_t>(
                            oldParserBytes,
                            static_cast<size_t>(
                                UINT32_MAX)));
                acceptedBytes =
                    parsed.consumedBytes > oldBytes
                        ? parsed.consumedBytes -
                              oldBytes
                        : 0u;
                acceptedBytes =
                    std::min(
                        acceptedBytes, remainingBytes);
            }
            m_vif0Dma.bufferedPayloadBytes =
                remainingBytes - acceptedBytes;
        }

        const uint64_t acceptedWithOffset =
            static_cast<uint64_t>(priorByteOffset) +
            acceptedBytes;
        const uint32_t completedQwc =
            static_cast<uint32_t>(
                acceptedWithOffset / 16u);
        if (completedQwc > m_vif0Dma.qwc)
        {
            fault("payload accounting exceeded QWC");
            return result;
        }
        m_vif0Dma.madr += completedQwc * 16u;
        m_vif0Dma.qwc -= completedQwc;
        m_vif0Dma.payloadByteOffset =
            static_cast<uint32_t>(
                acceptedWithOffset % 16u);
        if (m_vif0Dma.qwc == 0u &&
            m_vif0Dma.payloadByteOffset != 0u)
        {
            fault("terminal payload is not QW aligned");
            return result;
        }
        if (m_vif0Dma.qwc == 0u)
        {
            if (m_vif0Dma.chainMode &&
                m_vif0Dma.tagId == 1u)
            {
                m_vif0Dma.tadr = m_vif0Dma.madr;
            }
            m_vif0Dma.phase =
                m_vif0Dma.endAfterPayload
                    ? Vif0DmaPhase::Finalize
                    : Vif0DmaPhase::FetchTag;
        }
        publishVif0DmaRegisters();

        result.progressed = acceptedBytes != 0u;
        result.acceptedBytes = acceptedBytes;
        if (result.progressed)
        {
            const uint32_t transferredWords =
                (priorByteOffset + acceptedBytes) / 4u;
            result.delayEeCycles =
                std::max(1u, transferredWords / 2u);
        }
    }
    else
    {
        fault("invalid transfer phase");
        return result;
    }

    result.phase = m_vif0Dma.phase;
    result.stall = m_vif0Dma.stall;
    result.active = m_vif0Dma.active;
    return result;
}

void PS2Memory::drainVif0DmaCompatibility()
{
    constexpr uint32_t kMaximumSteps =
        (65536u * 2u) + 4u;
    for (uint32_t step = 0u;
         step < kMaximumSteps && m_vif0Dma.active;
         ++step)
    {
        const Vif0DmaAdvanceResult advance =
            advanceVif0Dma();
        if (advance.completed || !advance.active)
            break;
        if (advance.stall !=
            Vif0DmaStallReason::None)
        {
            break;
        }
        if (!advance.progressed)
            break;
    }
}

void PS2Memory::updateVif1Fqc(uint32_t qwc)
{
    constexpr uint32_t kFqcMask = 0x1Fu << 24u;
    const uint32_t fqc = std::min(qwc, 16u);
    vif1_regs.stat =
        (vif1_regs.stat & ~kFqcMask) |
        (fqc << 24u);
}

void PS2Memory::clearVif1DmaState(
    bool notifyRuntime)
{
    const bool wasActive = m_vif1Dma.active;
    const bool wasEventManaged =
        wasActive &&
        m_vif1Dma.eventManaged;
    m_vif1Dma = {};
    updateVif1Fqc(0u);
    if (wasActive)
        finishVif1DmaTrace();
    if (notifyRuntime && wasEventManaged &&
        m_vif1DmaCancelCallback)
    {
        m_vif1DmaCancelCallback();
    }
}

bool PS2Memory::scheduleVif1Dma(
    uint32_t delayEeCycles)
{
    if (!m_vif1Dma.active ||
        !m_vif1DmaScheduleCallback)
    {
        return false;
    }

    const bool scheduled =
        m_vif1DmaScheduleCallback(delayEeCycles);
    m_vif1Dma.eventManaged = scheduled;
    return scheduled;
}

bool PS2Memory::startVif1Dma(
    DmacTransferToken transfer, uint32_t chcr)
{
    constexpr uint32_t kVif1Base = 0x10009000u;

    clearVif1DmaState(true);
    Vif1DmaState state{};
    state.transfer = transfer;
    state.madr = m_ioRegisters[kVif1Base + 0x10u];
    state.qwc =
        m_ioRegisters[kVif1Base + 0x20u] & 0xFFFFu;
    state.tadr = m_ioRegisters[kVif1Base + 0x30u];
    state.asr0 = m_ioRegisters[kVif1Base + 0x40u];
    state.asr1 = m_ioRegisters[kVif1Base + 0x50u];
    state.asp = static_cast<uint8_t>((chcr >> 4u) & 0x3u);
    state.chainMode = ((chcr >> 2u) & 0x3u) == 1u;
    state.tie = (chcr & (1u << 7u)) != 0u;
    state.tte = (chcr & (1u << 6u)) != 0u;
    state.active = true;

    if (state.chainMode)
    {
        if (state.qwc != 0u)
        {
            state.tagId =
                static_cast<uint8_t>((chcr >> 28u) & 0x7u);
            state.tagIrq =
                (chcr & (1u << 31u)) != 0u;
            state.endAfterPayload =
                state.tagId == 0u ||
                state.tagId == 7u ||
                (state.tie && state.tagIrq);
            state.phase = Vif1DmaPhase::TransferPayload;
        }
        else
        {
            state.phase = Vif1DmaPhase::FetchTag;
        }
    }
    else
    {
        state.endAfterPayload = true;
        state.phase = state.qwc != 0u
                          ? Vif1DmaPhase::TransferPayload
                          : Vif1DmaPhase::Finalize;
    }

    m_vif1Dma = std::move(state);
    m_ioRegisters[kVif1Base + 0x20u] =
        m_vif1Dma.qwc;
    updateVif1Fqc(m_vif1Dma.qwc);
    return true;
}

Vif1DmaStallReason
PS2Memory::currentVif1DmaStall() const
{
    if (m_vif1Dma.phase == Vif1DmaPhase::Fault)
        return Vif1DmaStallReason::Fault;
    if (!isDmacEnabled())
        return Vif1DmaStallReason::DmacDisabled;
    if ((vif1_regs.stat & (1u << 9u)) != 0u)
        return Vif1DmaStallReason::ForceBreak;
    if ((vif1_regs.stat & (1u << 8u)) != 0u)
        return Vif1DmaStallReason::VifStop;
    if ((vif1_regs.stat & (1u << 10u)) != 0u)
        return Vif1DmaStallReason::VifInterrupt;
    if (m_vif1WaitingForVu)
        return Vif1DmaStallReason::WaitingForVu;
    return Vif1DmaStallReason::None;
}

Vif1DmaSnapshot PS2Memory::vif1DmaSnapshot() const
{
    Vif1DmaSnapshot snapshot{};
    snapshot.transfer = m_vif1Dma.transfer;
    snapshot.phase = m_vif1Dma.phase;
    snapshot.stall = m_vif1Dma.stall;
    snapshot.madr = m_vif1Dma.madr;
    snapshot.qwc = m_vif1Dma.qwc;
    snapshot.tadr = m_vif1Dma.tadr;
    snapshot.asr0 = m_vif1Dma.asr0;
    snapshot.asr1 = m_vif1Dma.asr1;
    snapshot.parserBytes =
        static_cast<uint32_t>(
            std::min<size_t>(
                m_vif1DeferredData.size(),
                static_cast<size_t>(UINT32_MAX)));
    snapshot.tagsProcessed =
        m_vif1Dma.tagsProcessed;
    snapshot.asp = m_vif1Dma.asp;
    snapshot.tagId = m_vif1Dma.tagId;
    snapshot.active = m_vif1Dma.active;
    snapshot.eventManaged =
        m_vif1Dma.eventManaged;
    snapshot.chainMode = m_vif1Dma.chainMode;
    snapshot.tagIrq = m_vif1Dma.tagIrq;
    snapshot.endAfterPayload =
        m_vif1Dma.endAfterPayload;
    return snapshot;
}

bool PS2Memory::wakeVif1Dma()
{
    if (!m_vif1Dma.active)
        return false;

    m_vif1Dma.stall = Vif1DmaStallReason::None;
    if (m_vif1Dma.eventManaged &&
        scheduleVif1Dma(0u))
    {
        return true;
    }

    drainVif1DmaCompatibility();
    return true;
}

Vif1DmaAdvanceResult PS2Memory::advanceVif1Dma()
{
    constexpr uint32_t kVif1Base = 0x10009000u;
    constexpr uint32_t kMaximumChainTags = 65536u;

    Vif1DmaAdvanceResult result{};
    result.transfer = m_vif1Dma.transfer;
    result.phase = m_vif1Dma.phase;
    result.active = m_vif1Dma.active;

    if (!m_vif1Dma.active)
        return result;

    if (!progressDmacTransfer(m_vif1Dma.transfer))
    {
        clearVif1DmaState(false);
        result.active = false;
        result.phase = Vif1DmaPhase::Idle;
        return result;
    }

    m_vif1Dma.stall = currentVif1DmaStall();
    if (m_vif1Dma.stall !=
        Vif1DmaStallReason::None)
    {
        result.stall = m_vif1Dma.stall;
        return result;
    }

    const auto parserStall =
        [&]() -> Vif1DmaStallReason
        {
            switch (processVif1Stream())
            {
            case Vif1ParserDisposition::WaitingForVu:
                return Vif1DmaStallReason::WaitingForVu;
            case Vif1ParserDisposition::VifInterrupt:
                return Vif1DmaStallReason::VifInterrupt;
            case Vif1ParserDisposition::GifPath:
                return Vif1DmaStallReason::GifPath;
            case Vif1ParserDisposition::Drained:
            case Vif1ParserDisposition::NeedMoreData:
                return Vif1DmaStallReason::None;
            }
            return Vif1DmaStallReason::None;
        };

    if (!m_vif1DeferredData.empty())
    {
        m_vif1Dma.stall = parserStall();
        if (m_vif1Dma.stall !=
            Vif1DmaStallReason::None)
        {
            result.stall = m_vif1Dma.stall;
            result.delayEeCycles =
                m_vif1Dma.stall ==
                        Vif1DmaStallReason::GifPath
                    ? 128u
                    : 0u;
            return result;
        }
    }

    const auto fault =
        [&](const char *reason)
        {
            m_vif1Dma.phase = Vif1DmaPhase::Fault;
            m_vif1Dma.stall = Vif1DmaStallReason::Fault;
            if (!m_vif1Dma.faultReported)
            {
                std::fprintf(
                    stderr,
                    "VIF1 DMA stalled: tadr=0x%08x "
                    "madr=0x%08x tags=%u reason=%s\n",
                    m_vif1Dma.tadr, m_vif1Dma.madr,
                    m_vif1Dma.tagsProcessed, reason);
                m_vif1Dma.faultReported = true;
            }
            result.phase = m_vif1Dma.phase;
            result.stall = m_vif1Dma.stall;
            result.active = true;
        };

    if (m_vif1Dma.phase == Vif1DmaPhase::Finalize)
    {
        updateVif1Fqc(0u);
        m_vif1Dma.active = false;
        m_vif1Dma.eventManaged = false;
        m_vif1Dma.stall = Vif1DmaStallReason::None;
        result.completed =
            requestDmacCompletion(m_vif1Dma.transfer);
        result.progressed = result.completed;
        result.active = false;
        result.phase = Vif1DmaPhase::Finalize;
        finishVif1DmaTrace();
        return result;
    }

    if (m_vif1Dma.phase == Vif1DmaPhase::FetchTag)
    {
        if (m_vif1Dma.tagsProcessed >=
            kMaximumChainTags)
        {
            fault("tag safety limit reached");
            return result;
        }

        const DmacSourceChainKey key{
            m_vif1Dma.tadr,
            m_vif1Dma.asr0,
            m_vif1Dma.asr1,
            m_vif1Dma.asp};
        if (!m_vif1Dma.visitedStates.insert(key).second)
        {
            fault("repeated chain state");
            return result;
        }

        const uint32_t tagAddress = m_vif1Dma.tadr;
        const bool scratch = isScratchpad(tagAddress);
        uint32_t tagOffset = 0u;
        try
        {
            tagOffset = translateAddress(tagAddress);
        }
        catch (const std::exception &)
        {
            fault("unmapped tag address");
            return result;
        }

        const uint8_t *const base =
            scratch ? m_scratchpad : m_rdram;
        const uint32_t limit =
            scratch ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
        if (!base || tagOffset > limit ||
            limit - tagOffset < 16u)
        {
            fault("tag crosses memory boundary");
            return result;
        }

        const uint8_t *const tagData = base + tagOffset;
        const uint64_t tag = loadScalar<uint64_t>(
            tagData, 0u, 16u, "VIF1 DMA tag",
            tagAddress);
        const uint32_t tagQwc =
            static_cast<uint32_t>(tag & 0xFFFFu);
        const uint32_t tagAddressField =
            static_cast<uint32_t>(
                (tag >> 32u) & 0x7FFFFFFFu);
        m_vif1Dma.tagId =
            static_cast<uint8_t>((tag >> 28u) & 0x7u);
        m_vif1Dma.tagIrq =
            ((tag >> 31u) & 0x1u) != 0u;
        m_vif1Dma.endAfterPayload = false;
        const uint32_t tagIndex =
            m_vif1Dma.tagsProcessed;
        const size_t streamOffset =
            m_gsDmaTrace && m_gsDmaTrace->active
                ? static_cast<size_t>(
                      m_gsDmaTrace->vifInput.bytes)
                : 0u;
        ++m_vif1Dma.tagsProcessed;
        traceVif1DmaTag(tagData);

        uint32_t dataAddress = 0u;
        switch (m_vif1Dma.tagId)
        {
        case 0u: // REFE
            dataAddress = tagAddressField;
            m_vif1Dma.tadr += 16u;
            m_vif1Dma.endAfterPayload = true;
            break;
        case 1u: // CNT
            m_vif1Dma.tadr += 16u;
            dataAddress = m_vif1Dma.tadr;
            break;
        case 2u: // NEXT
            dataAddress = m_vif1Dma.tadr + 16u;
            m_vif1Dma.tadr = tagAddressField;
            break;
        case 3u: // REF
        case 4u: // REFS
            dataAddress = tagAddressField;
            m_vif1Dma.tadr += 16u;
            break;
        case 5u: // CALL
        {
            dataAddress = m_vif1Dma.tadr + 16u;
            const uint32_t returnAddress =
                dataAddress + tagQwc * 16u;
            if (m_vif1Dma.asp == 0u)
            {
                m_vif1Dma.asr0 = returnAddress;
                m_vif1Dma.asp = 1u;
            }
            else if (m_vif1Dma.asp == 1u)
            {
                m_vif1Dma.asr1 = returnAddress;
                m_vif1Dma.asp = 2u;
            }
            else
            {
                fault("CALL stack overflow");
                return result;
            }
            m_vif1Dma.tadr = tagAddressField;
            break;
        }
        case 6u: // RET
            dataAddress = m_vif1Dma.tadr + 16u;
            if (m_vif1Dma.asp == 2u)
            {
                m_vif1Dma.tadr = m_vif1Dma.asr1;
                m_vif1Dma.asr1 = 0u;
                m_vif1Dma.asp = 1u;
            }
            else if (m_vif1Dma.asp == 1u)
            {
                m_vif1Dma.tadr = m_vif1Dma.asr0;
                m_vif1Dma.asr0 = 0u;
                m_vif1Dma.asp = 0u;
            }
            else
            {
                m_vif1Dma.endAfterPayload = true;
            }
            break;
        case 7u: // END
            dataAddress = m_vif1Dma.tadr + 16u;
            m_vif1Dma.endAfterPayload = true;
            break;
        default:
            fault("invalid tag ID");
            return result;
        }

        if (m_gsDmaTrace &&
            m_gsDmaTrace->active &&
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
                tagIndex, streamOffset, tagAddress,
                tagQwc,
                static_cast<uint32_t>(
                    m_vif1Dma.tagId),
                m_vif1Dma.tagIrq ? "true" : "false",
                tagAddressField, dataAddress);
        }

        if (m_vif1Dma.tie && m_vif1Dma.tagIrq)
            m_vif1Dma.endAfterPayload = true;

        m_vif1Dma.madr = dataAddress;
        m_vif1Dma.qwc = tagQwc;
        uint32_t chcr = m_ioRegisters[kVif1Base];
        chcr = (chcr & 0x0000FFFFu) |
               (static_cast<uint32_t>(
                    (tag >> 16u) & 0xFFFFu)
                << 16u);
        chcr = (chcr & ~(0x3u << 4u)) |
               (static_cast<uint32_t>(
                    m_vif1Dma.asp)
                << 4u);
        m_ioRegisters[kVif1Base] = chcr;
        m_ioRegisters[kVif1Base + 0x10u] =
            m_vif1Dma.madr;
        m_ioRegisters[kVif1Base + 0x20u] =
            m_vif1Dma.qwc;
        m_ioRegisters[kVif1Base + 0x30u] =
            m_vif1Dma.tadr;
        m_ioRegisters[kVif1Base + 0x40u] =
            m_vif1Dma.asr0;
        m_ioRegisters[kVif1Base + 0x50u] =
            m_vif1Dma.asr1;
        updateVif1Fqc(m_vif1Dma.qwc);

        uint32_t delay = 1u;
        if (m_vif1Dma.tte)
        {
            appendVif1Stream(tagData + 8u, 8u);
            traceVif1Input(tagData + 8u, 8u);
            ++delay;
        }

        m_vif1Dma.phase =
            m_vif1Dma.qwc != 0u
                ? Vif1DmaPhase::TransferPayload
                : (m_vif1Dma.endAfterPayload
                       ? Vif1DmaPhase::Finalize
                       : Vif1DmaPhase::FetchTag);
        m_vif1Dma.stall = parserStall();
        result.progressed = true;
        result.delayEeCycles = delay;
    }
    else if (
        m_vif1Dma.phase ==
        Vif1DmaPhase::TransferPayload)
    {
        const uint32_t transferredQwc =
            m_vif1Dma.qwc;
        uint64_t bytesLeft =
            static_cast<uint64_t>(transferredQwc) *
            16ull;
        uint32_t sourceAddress = m_vif1Dma.madr;
        const bool scratch = isScratchpad(sourceAddress);
        uint32_t sourceOffset = 0u;
        try
        {
            sourceOffset = translateAddress(sourceAddress);
        }
        catch (const std::exception &)
        {
            fault("unmapped payload address");
            return result;
        }

        const uint8_t *const base =
            scratch ? m_scratchpad : m_rdram;
        const uint32_t limit =
            scratch ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
        while (bytesLeft != 0u)
        {
            if (sourceOffset >= limit)
                sourceOffset = 0u;
            const uint32_t chunk =
                static_cast<uint32_t>(
                    std::min<uint64_t>(
                        bytesLeft,
                        static_cast<uint64_t>(
                            limit - sourceOffset)));
            if (chunk == 0u)
            {
                fault("empty payload memory region");
                return result;
            }
            appendVif1Stream(base + sourceOffset, chunk);
            traceVif1DmaPayload(
                base + sourceOffset, chunk);
            traceVif1Input(
                base + sourceOffset, chunk);
            sourceOffset += chunk;
            bytesLeft -= chunk;
        }

        m_vif1Dma.madr += transferredQwc * 16u;
        m_vif1Dma.qwc = 0u;
        if (m_vif1Dma.chainMode &&
            m_vif1Dma.tagId == 1u)
        {
            m_vif1Dma.tadr = m_vif1Dma.madr;
        }
        m_ioRegisters[kVif1Base + 0x10u] =
            m_vif1Dma.madr;
        m_ioRegisters[kVif1Base + 0x20u] = 0u;
        m_ioRegisters[kVif1Base + 0x30u] =
            m_vif1Dma.tadr;
        updateVif1Fqc(0u);

        m_vif1Dma.phase =
            m_vif1Dma.endAfterPayload
                ? Vif1DmaPhase::Finalize
                : Vif1DmaPhase::FetchTag;
        m_vif1Dma.stall = parserStall();
        result.progressed = true;
        const uint64_t payloadDelay =
            std::max<uint64_t>(
                1u,
                static_cast<uint64_t>(
                    transferredQwc) *
                    2u);
        result.delayEeCycles =
            static_cast<uint32_t>(
                std::min<uint64_t>(
                    payloadDelay,
                    std::numeric_limits<
                        uint32_t>::max()));
    }
    else
    {
        fault("invalid transfer phase");
        return result;
    }

    if (m_vif1Dma.stall ==
        Vif1DmaStallReason::GifPath)
    {
        result.delayEeCycles = 128u;
    }
    else if (m_vif1Dma.stall !=
             Vif1DmaStallReason::None)
    {
        result.delayEeCycles = 0u;
    }

    result.phase = m_vif1Dma.phase;
    result.stall = m_vif1Dma.stall;
    result.active = m_vif1Dma.active;
    return result;
}

void PS2Memory::drainVif1DmaCompatibility()
{
    constexpr uint32_t kMaximumSteps =
        (65536u * 2u) + 4u;
    for (uint32_t step = 0u;
         step < kMaximumSteps && m_vif1Dma.active;
         ++step)
    {
        const Vif1DmaAdvanceResult advance =
            advanceVif1Dma();
        if (advance.completed || !advance.active)
            break;
        if (advance.stall !=
            Vif1DmaStallReason::None)
        {
            break;
        }
        if (!advance.progressed)
            break;
    }
}

void PS2Memory::processPendingTransfers()
{
    if (m_gifDma.active &&
        !m_gifDma.eventManaged)
        drainGifDmaCompatibility();
    if (m_vif0Dma.active &&
        !m_vif0Dma.eventManaged)
        drainVif0DmaCompatibility();
    if (m_vif1Dma.active &&
        !m_vif1Dma.eventManaged)
        drainVif1DmaCompatibility();
    if (m_fromIpuDma.active &&
        !m_fromIpuDma.eventManaged)
        drainFromIpuDmaCompatibility();
    if (m_toIpuDma.active &&
        !m_toIpuDma.eventManaged)
        drainToIpuDmaCompatibility();
}

void PS2Memory::flushMaskedPath3Packets(bool drainImmediately)
{
    if (isPath3Masked() ||
        m_path3MaskedFifo.empty())
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
        if (isPath3Masked())
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

    if (!m_rdram || !m_gsVRAM ||
        isPath3Masked())
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

    if (!m_rdram || !m_gsVRAM ||
        isPath3Masked())
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

    if ((address >= 0x10003800u &&
         address < 0x10003A00u) ||
        (address >= 0x10003C00u &&
         address < 0x10003E00u))
    {
        const uint32_t base =
            address < 0x10003C00u
                ? 0x10003800u
                : 0x10003C00u;
        const VIFRegisters &regs =
            base == 0x10003800u
                ? vif0_regs
                : vif1_regs;
        const uint32_t offset = address - base;
        if (const std::optional<uint32_t> value =
                vifRegisterValue(regs, offset);
            value.has_value())
        {
            return *value;
        }
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
            val =
                (m_ioRegisters[address] &
                 ~((1u << 31) | 0xFFu)) |
                (m_ipuInputFifoQwc & 0xFu) |
                ((m_ipuOutputFifoQwc & 0xFu)
                 << 4u);
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
        uint32_t counterValue = 0u;
        ps2x::timing::EeCounterAdvanceResult counterAdvance{};
        if (m_eeCounters.readRegister(
                address,
                currentEeCounterCycle(),
                counterValue,
                &counterAdvance))
        {
            publishEeCounterAdvance(counterAdvance);
            return counterValue;
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
