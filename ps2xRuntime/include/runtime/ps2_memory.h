#ifndef PS2_MEMORY_H
#define PS2_MEMORY_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <iostream>
#include <mutex>

#include "ps2_build_config.h"
#include "ee_counters.h"
#include "ps2_gif_arbiter.h"
#include "ps2_vu1.h"
#include "ps2_vu1_command_stream.h"
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(USE_SSE2NEON)
#include "sse2neon.h"
#else
#include <immintrin.h> // For SSE/AVX instructions
#include <smmintrin.h> // For SSE4.1 instructions
#endif

class GsCommandExecutor;
struct Ps2GsDmaTraceState;
struct Ps2Vu1WorkloadProfileState;
struct VuExecutionState;

struct Vif1DmaTraceSnapshot
{
    bool enabled = false;
    bool active = false;
    uint64_t nextSequence = 0u;
    uint64_t startSequence = 0u;
    uint64_t limitSequence = 0u;
    std::string outputPath;
    std::string vif1InputDumpDirectory;
    std::string path1DumpDirectory;
    std::string vu1DumpDirectory;
};

enum class EeAddressSpaceMode : uint8_t
{
    Kernel,
    Supervisor,
    User,
    Reserved,
};

struct EeAddressTranslationContext
{
    EeAddressSpaceMode mode = EeAddressSpaceMode::Kernel;
    bool enforceOperatingMode = false;
    bool errorLevel = false;
    uint8_t asid = 0u;

    static constexpr EeAddressTranslationContext unchecked() noexcept
    {
        return {};
    }

    static constexpr EeAddressTranslationContext fromCop0Status(
        uint32_t status,
        uint8_t asid = 0u) noexcept
    {
        constexpr uint32_t kStatusExl = 0x00000002u;
        constexpr uint32_t kStatusErl = 0x00000004u;
        constexpr uint32_t kStatusKsuMask = 0x00000018u;
        constexpr uint32_t kStatusKsuSupervisor = 0x00000008u;
        constexpr uint32_t kStatusKsuUser = 0x00000010u;

        const bool errorLevelActive =
            (status & kStatusErl) != 0u;
        if ((status & (kStatusExl | kStatusErl)) != 0u)
        {
            return {
                EeAddressSpaceMode::Kernel,
                true,
                errorLevelActive,
                asid,
            };
        }

        switch (status & kStatusKsuMask)
        {
        case 0u:
            return {EeAddressSpaceMode::Kernel, true, false, asid};
        case kStatusKsuSupervisor:
            return {EeAddressSpaceMode::Supervisor, true, false, asid};
        case kStatusKsuUser:
            return {EeAddressSpaceMode::User, true, false, asid};
        default:
            return {EeAddressSpaceMode::Reserved, true, false, asid};
        }
    }

    constexpr bool permits(uint32_t virtualAddress) const noexcept
    {
        if (!enforceOperatingMode ||
            mode == EeAddressSpaceMode::Kernel)
        {
            return true;
        }
        if (virtualAddress < 0x80000000u)
        {
            return true;
        }
        return mode == EeAddressSpaceMode::Supervisor &&
               virtualAddress >= 0xC0000000u &&
               virtualAddress < 0xE0000000u;
    }

    constexpr bool mapsKusegThroughTlb(
        uint32_t virtualAddress) const noexcept
    {
        return enforceOperatingMode &&
               !errorLevel &&
               virtualAddress < 0x80000000u;
    }
};

class PS2AddressErrorException final : public std::exception
{
public:
    explicit PS2AddressErrorException(
        uint32_t virtualAddress) noexcept
        : m_virtualAddress(virtualAddress)
    {
    }

    const char *what() const noexcept override
    {
        return "EE address error";
    }

    uint32_t virtualAddress() const noexcept
    {
        return m_virtualAddress;
    }

private:
    uint32_t m_virtualAddress;
};

enum class PS2TlbFaultKind : uint8_t
{
    Refill,
    Invalid,
    Modified,
};

class PS2TlbFaultException : public std::exception
{
public:
    PS2TlbFaultException(
        uint32_t virtualAddress,
        PS2TlbFaultKind kind) noexcept
        : m_virtualAddress(virtualAddress),
          m_kind(kind)
    {
    }

    const char *what() const noexcept override
    {
        switch (m_kind)
        {
        case PS2TlbFaultKind::Refill:
            return "EE TLB refill";
        case PS2TlbFaultKind::Invalid:
            return "EE TLB invalid";
        case PS2TlbFaultKind::Modified:
            return "EE TLB modified";
        }
        return "EE TLB fault";
    }

    uint32_t virtualAddress() const noexcept
    {
        return m_virtualAddress;
    }

    PS2TlbFaultKind kind() const noexcept
    {
        return m_kind;
    }

private:
    uint32_t m_virtualAddress;
    PS2TlbFaultKind m_kind;
};

class PS2TlbMissException final : public PS2TlbFaultException
{
public:
    explicit PS2TlbMissException(uint32_t virtualAddress) noexcept
        : PS2TlbFaultException(
              virtualAddress,
              PS2TlbFaultKind::Refill)
    {
    }
};

class PS2TlbInvalidException final : public PS2TlbFaultException
{
public:
    explicit PS2TlbInvalidException(uint32_t virtualAddress) noexcept
        : PS2TlbFaultException(
              virtualAddress,
              PS2TlbFaultKind::Invalid)
    {
    }
};

class PS2TlbModifiedException final : public PS2TlbFaultException
{
public:
    explicit PS2TlbModifiedException(uint32_t virtualAddress) noexcept
        : PS2TlbFaultException(
              virtualAddress,
              PS2TlbFaultKind::Modified)
    {
    }
};

struct EeTlbEntry
{
    uint32_t pageMask = 0u;
    uint32_t entryHi = 0u;
    uint32_t entryLo0 = 0u;
    uint32_t entryLo1 = 0u;
};

struct EeTlbTranslationCacheStats
{
    uint64_t hits = 0u;
    uint64_t misses = 0u;
    uint64_t permissionMisses = 0u;
    uint64_t generationMisses = 0u;
    uint64_t intervalCrossings = 0u;
    uint64_t slowPathFaults = 0u;
    uint64_t fills = 0u;
    uint64_t replacements = 0u;
    uint64_t mappingChanges = 0u;
    uint64_t fastHits = 0u;
    uint64_t fastMisses = 0u;
    uint64_t backingHits = 0u;
    uint64_t backingMisses = 0u;
    uint64_t fastFills = 0u;
    uint64_t fastReplacements = 0u;
};

class PS2BusErrorException final : public std::exception
{
public:
    explicit PS2BusErrorException(uint32_t physicalAddress) noexcept
        : m_physicalAddress(physicalAddress)
    {
    }

    const char *what() const noexcept override
    {
        return "EE physical bus error";
    }

    uint32_t physicalAddress() const noexcept
    {
        return m_physicalAddress;
    }

private:
    uint32_t m_physicalAddress;
};

constexpr uint32_t PS2_RAM_SIZE = 32u * 1024u * 1024u; // 32MB
constexpr uint32_t PS2_RAM_MASK = PS2_RAM_SIZE - 1u;   // Mask for 32MB alignment
constexpr uint32_t PS2_RAM_BASE = 0x00000000;          // Physical base of RDRAM
constexpr uint32_t PS2_IOP_RAM_SIZE = 2u * 1024u * 1024u; // 2MB
constexpr uint32_t PS2_EE_UNCACHED_RAM_MIRROR_BASE = 0x20000000u;
constexpr uint32_t PS2_EE_UNCACHED_RAM_MIRROR_SIZE = PS2_RAM_SIZE;
constexpr uint32_t PS2_EE_ACCELERATED_RAM_MIRROR_BASE = 0x30100000u;
constexpr uint32_t PS2_EE_ACCELERATED_RAM_MIRROR_SIZE = PS2_RAM_SIZE - 0x00100000u;
constexpr uint32_t PS2_SCRATCHPAD_BASE = 0x70000000;
constexpr uint32_t PS2_SCRATCHPAD_ALIAS_BASE = 0xF0000000;
constexpr uint32_t PS2_SCRATCHPAD_SIZE = 16u * 1024u;  // 16KB
constexpr uint32_t PS2_IO_BASE = 0x10000000;           // Base for many I/O regs (Timers, DMAC, INTC)
constexpr uint32_t PS2_IO_SIZE = 0x10000;              // 64KB
constexpr uint32_t PS2_BIOS_BASE = 0x1FC00000;         // Or BFC00000 depending on KSEG
constexpr uint32_t PS2_BIOS_SIZE = 4u * 1024u * 1024u; // 4MB

constexpr uint32_t PS2_VU0_CODE_BASE = 0x11000000; // Base address as seen from EE
constexpr uint32_t PS2_VU0_DATA_BASE = 0x11004000;
constexpr uint32_t PS2_VU0_CODE_SIZE = 4u * 1024u; // 4KB Micro Memory
constexpr uint32_t PS2_VU0_DATA_SIZE = 4u * 1024u; // 4KB Data Memory (VU Mem)

constexpr uint32_t PS2_VU1_CODE_BASE = 0x11008000;
constexpr uint32_t PS2_VU1_DATA_BASE = 0x1100C000;
constexpr uint32_t PS2_VU1_MEM_BASE = PS2_VU1_CODE_BASE; // Alias used by older code paths
constexpr uint32_t PS2_VU1_CODE_SIZE = 16u * 1024u;      // 16KB Micro Memory
constexpr uint32_t PS2_VU1_DATA_SIZE = 16u * 1024u;      // 16KB Data Memory (VU Mem)

constexpr uint32_t PS2_GS_BASE = 0x12000000;
constexpr uint32_t PS2_GS_PRIV_REG_BASE = PS2_GS_BASE; // GS Privileged Registers
constexpr uint32_t PS2_GS_PRIV_REG_SIZE = 0x2000;
constexpr size_t PS2_GS_VRAM_SIZE = 4u * 1024u * 1024u; // 4MB GS VRAM

// These regions are explicitly marked as BUSERR by the EE memory map. Callers
// must resolve virtual aliases first: PS2Recomp's configured RDRAM mirrors
// overlap the nominal 0x20000000 physical range but translate back to RDRAM.
inline constexpr bool Ps2IsEeBusErrorPhysicalAddress(
    uint32_t physicalAddress) noexcept
{
    return (physicalAddress >= 0x11010000u &&
            physicalAddress < 0x12000000u) ||
           (physicalAddress >= 0x12010000u &&
            physicalAddress < 0x14000000u) ||
           (physicalAddress >= 0x20000000u &&
            physicalAddress < 0x40000000u) ||
           physicalAddress >= 0x80000000u;
}

// EE left/right stores select one contiguous span of little-endian byte lanes.
// A zero byte enable is a valid no-op; non-contiguous enables are rejected.
struct Ps2ByteEnableSpan
{
    bool valid = false;
    uint32_t offset = 0u;
    uint32_t size = 0u;
};

inline constexpr Ps2ByteEnableSpan Ps2DecodeByteEnableSpan(
    uint8_t byteEnable,
    uint32_t widthBytes) noexcept
{
    if (widthBytes == 0u || widthBytes > 8u)
    {
        return {};
    }

    const uint32_t allowed = (1u << widthBytes) - 1u;
    if ((static_cast<uint32_t>(byteEnable) & ~allowed) != 0u)
    {
        return {};
    }
    if (byteEnable == 0u)
    {
        return {true, 0u, 0u};
    }

    uint32_t first = 0u;
    while ((byteEnable & (1u << first)) == 0u)
    {
        ++first;
    }
    uint32_t last = widthBytes - 1u;
    while ((byteEnable & (1u << last)) == 0u)
    {
        --last;
    }
    const uint32_t contiguous =
        ((1u << (last - first + 1u)) - 1u) << first;
    if (static_cast<uint32_t>(byteEnable) != contiguous)
    {
        return {};
    }
    return {true, first, last - first + 1u};
}

inline constexpr uint64_t Ps2ExpandByteEnableMask(
    uint8_t byteEnable,
    uint32_t widthBytes) noexcept
{
    if (widthBytes > 8u)
    {
        return 0u;
    }

    uint64_t mask = 0u;
    for (uint32_t index = 0u; index < widthBytes; ++index)
    {
        if ((byteEnable & (1u << index)) != 0u)
        {
            mask |= 0xFFull << (index * 8u);
        }
    }
    return mask;
}

inline constexpr uint32_t PS2_FIO_O_RDONLY = 0x0001;
inline constexpr uint32_t PS2_FIO_O_WRONLY = 0x0002;
inline constexpr uint32_t PS2_FIO_O_RDWR = 0x0003;
inline constexpr uint32_t PS2_FIO_O_NBLOCK = 0x0010;
inline constexpr uint32_t PS2_FIO_O_APPEND = 0x0100;
inline constexpr uint32_t PS2_FIO_O_CREAT = 0x0200;
inline constexpr uint32_t PS2_FIO_O_TRUNC = 0x0400;
inline constexpr uint32_t PS2_FIO_O_EXCL = 0x0800;
inline constexpr uint32_t PS2_FIO_O_NOWAIT = 0x8000;

inline constexpr uint32_t PS2_FIO_SEEK_SET = 0;
inline constexpr uint32_t PS2_FIO_SEEK_CUR = 1;
inline constexpr uint32_t PS2_FIO_SEEK_END = 2;

inline constexpr uint32_t PS2_FIO_S_IFDIR = 0x1000;
inline constexpr uint32_t PS2_FIO_S_IFREG = 0x2000;

static_assert((PS2_RAM_SIZE & (PS2_RAM_SIZE - 1u)) == 0u, "PS2_RAM_SIZE must be a power of two");
static_assert(PS2_RAM_MASK == (PS2_RAM_SIZE - 1u), "PS2_RAM_MASK must match PS2_RAM_SIZE");

inline std::atomic<uint8_t *> &ps2ScratchpadHostPtrStorage()
{
    static std::atomic<uint8_t *> ptr{nullptr};
    return ptr;
}

inline void ps2SetScratchpadHostPtr(uint8_t *ptr)
{
    ps2ScratchpadHostPtrStorage().store(ptr, std::memory_order_relaxed);
}

inline uint8_t *ps2GetScratchpadHostPtr()
{
    return ps2ScratchpadHostPtrStorage().load(std::memory_order_relaxed);
}

inline bool ps2IsScratchpadAddress(uint32_t addr)
{
    if (addr >= PS2_SCRATCHPAD_BASE && addr < (PS2_SCRATCHPAD_BASE + PS2_SCRATCHPAD_SIZE))
    {
        return true;
    }

    if ((addr & 0x80000000u) != 0u)
    {
        const uint32_t lower = addr & 0x7FFFFFFFu;
        return lower >= PS2_SCRATCHPAD_BASE &&
               lower < (PS2_SCRATCHPAD_BASE + PS2_SCRATCHPAD_SIZE);
    }

    return false;
}

inline uint32_t ps2ScratchpadOffset(uint32_t addr)
{
    if (addr >= PS2_SCRATCHPAD_BASE && addr < (PS2_SCRATCHPAD_BASE + PS2_SCRATCHPAD_SIZE))
    {
        return addr - PS2_SCRATCHPAD_BASE;
    }

    const uint32_t lower = addr & 0x7FFFFFFFu;
    return lower - PS2_SCRATCHPAD_BASE;
}

inline constexpr bool ps2ResolveDirectRdramOffset(uint32_t addr, uint32_t &offset)
{
    const bool physical = addr < PS2_RAM_SIZE;
    const bool uncached =
        (addr - PS2_EE_UNCACHED_RAM_MIRROR_BASE) < PS2_EE_UNCACHED_RAM_MIRROR_SIZE;
    const bool accelerated =
        (addr - PS2_EE_ACCELERATED_RAM_MIRROR_BASE) < PS2_EE_ACCELERATED_RAM_MIRROR_SIZE;
    const bool kseg01 = (addr - 0x80000000u) < 0x40000000u;

    if (physical || uncached || accelerated ||
        (kseg01 && ((addr & 0x1FFFFFFFu) < PS2_RAM_SIZE)))
    {
        offset = addr & PS2_RAM_MASK;
        return true;
    }

    offset = 0u;
    return false;
}

inline bool ps2ResolveGuestPointer(uint32_t addr, uint32_t &offset, bool &scratch)
{
    if (ps2IsScratchpadAddress(addr))
    {
        scratch = true;
        offset = ps2ScratchpadOffset(addr);
        return true;
    }

    if (!ps2ResolveDirectRdramOffset(addr, offset))
    {
        scratch = false;
        return false;
    }

    scratch = false;
    return true;
}
inline uint8_t *getMemPtr(uint8_t *rdram, uint32_t addr)
{
    if (rdram == nullptr)
    {
        return nullptr;
    }

    uint32_t offset = 0;
    bool scratch = false;
    if (!ps2ResolveGuestPointer(addr, offset, scratch))
    {
        return nullptr;
    }

    if (scratch)
    {
        uint8_t *scratchpad = ps2GetScratchpadHostPtr();
        return scratchpad ? (scratchpad + offset) : nullptr;
    }
    return rdram + offset;
}

inline const uint8_t *getConstMemPtr(const uint8_t *rdram, uint32_t addr)
{
    if (rdram == nullptr)
    {
        return nullptr;
    }

    uint32_t offset = 0;
    bool scratch = false;
    if (!ps2ResolveGuestPointer(addr, offset, scratch))
    {
        return nullptr;
    }

    if (scratch)
    {
        const uint8_t *scratchpad = ps2GetScratchpadHostPtr();
        return scratchpad ? (scratchpad + offset) : nullptr;
    }
    return rdram + offset;
}

// PS2 GS (Graphics Synthesizer) registers
struct GSRegisters
{
    uint64_t pmode;    // Pixel mode
    uint64_t smode1;   // Sync mode 1
    uint64_t smode2;   // Sync mode 2
    uint64_t srfsh;    // Refresh control
    uint64_t synch1;   // Synchronization control 1
    uint64_t synch2;   // Synchronization control 2
    uint64_t syncv;    // Synchronization control V
    uint64_t dispfb1;  // Display buffer 1
    uint64_t display1; // Display area 1
    uint64_t dispfb2;  // Display buffer 2
    uint64_t display2; // Display area 2
    uint64_t extbuf;   // External buffer
    uint64_t extdata;  // External data
    uint64_t extwrite; // External write
    uint64_t bgcolor;  // Background color
    // EE-owned status mirror. EE VSync updates FIELD, guest MMIO performs
    // write-one-to-clear, and ordered GS results publish SIGNAL/FINISH. Keep
    // atomic RMWs so low-frequency host observation remains race-free too.
    std::atomic<uint64_t> csr;
    uint64_t imr;      // Interrupt mask
    uint64_t busdir;   // Bus direction
    uint64_t siglblid; // Signal label ID
};
static_assert(sizeof(GSRegisters) == (19u * sizeof(uint64_t)), "GSRegisters layout changed unexpectedly");
static_assert(alignof(GSRegisters) == alignof(uint64_t), "GSRegisters alignment must remain 64-bit");
// CSR can also be observed from host diagnostics while the EE updates it.
static_assert(std::atomic<uint64_t>::is_always_lock_free, "GS CSR atomic must be lock-free on all supported targets");

// PS2 VIF (VPU Interface) registers
struct VIFRegisters
{
    uint32_t stat;   // Status
    uint32_t fbrst;  // VIF Force Break
    uint32_t err;    // Error status
    uint32_t mark;   // Interrupt control
    uint32_t cycle;  // Transfer mode
    uint32_t mode;   // Mode control
    uint32_t num;    // Data amount counter
    uint32_t mask;   // Data mask
    uint32_t code;   // VIFcode
    uint32_t itops;  // ITOP save
    uint32_t base;   // Base address
    uint32_t ofst;   // Offset
    uint32_t tops;   // TOPS
    uint32_t itop;   // ITOP
    uint32_t top;    // TOP
    uint32_t row[4]; // Transfer row data
    uint32_t col[4]; // Transfer column data
};
static_assert(sizeof(VIFRegisters) == (23u * sizeof(uint32_t)), "VIFRegisters layout changed unexpectedly");

// PS2 DMA registers
struct DMARegisters
{
    uint32_t chcr; // Channel control
    uint32_t madr; // Memory address
    uint32_t qwc;  // Quadword count
    uint32_t tadr; // Tag address
    uint32_t asr0; // Address stack 0
    uint32_t asr1; // Address stack 1
    uint32_t sadr; // Source address
};
static_assert(sizeof(DMARegisters) == (7u * sizeof(uint32_t)), "DMARegisters layout changed unexpectedly");

enum class DmacChannel : uint8_t
{
    Vif0 = 0u,
    Vif1,
    Gif,
    FromIpu,
    ToIpu,
    Sif0,
    Sif1,
    Sif2,
    FromScratchpad,
    ToScratchpad,
    Count,
};

inline constexpr size_t PS2_DMAC_CHANNEL_COUNT =
    static_cast<size_t>(DmacChannel::Count);

inline constexpr uint32_t dmacChannelCause(DmacChannel channel) noexcept
{
    return static_cast<uint32_t>(channel);
}

inline constexpr uint32_t dmacChannelBase(DmacChannel channel) noexcept
{
    constexpr std::array<uint32_t, PS2_DMAC_CHANNEL_COUNT> bases = {
        0x10008000u,
        0x10009000u,
        0x1000A000u,
        0x1000B000u,
        0x1000B400u,
        0x1000C000u,
        0x1000C400u,
        0x1000C800u,
        0x1000D000u,
        0x1000D400u,
    };
    const size_t index = static_cast<size_t>(channel);
    return index < bases.size() ? bases[index] : 0u;
}

inline constexpr std::optional<DmacChannel>
dmacChannelFromBase(uint32_t channelBase) noexcept
{
    for (size_t index = 0u; index < PS2_DMAC_CHANNEL_COUNT; ++index)
    {
        const DmacChannel channel = static_cast<DmacChannel>(index);
        if (dmacChannelBase(channel) == channelBase)
        {
            return channel;
        }
    }
    return std::nullopt;
}

enum class DmacChannelPhase : uint8_t
{
    Idle = 0u,
    Active,
    CompletionReady,
};

struct DmacTransferToken
{
    DmacChannel channel = DmacChannel::Vif0;
    uint64_t generation = 0u;

    friend constexpr bool operator==(
        DmacTransferToken, DmacTransferToken) noexcept = default;
};

struct DmacCompletionRequest
{
    DmacTransferToken transfer{};
    uint64_t readySequence = 0u;
};

struct DmacPendingInterrupt
{
    DmacChannel source = DmacChannel::Vif0;
    uint32_t cause = 0u;
    uint64_t channelGeneration = 0u;
    uint64_t eventSequence = 0u;
    uint64_t publicationSequence = 0u;
};

struct DmacChannelSnapshot
{
    DmacChannelPhase phase = DmacChannelPhase::Idle;
    uint64_t generation = 0u;
};

enum class GifDmaPhase : uint8_t
{
    Idle = 0u,
    FetchTag,
    TransferPayload,
    Finalize,
    Fault,
};

enum class GifDmaStallReason : uint8_t
{
    None = 0u,
    Path3Masked,
    GifPaused,
    DmacDisabled,
    Fault,
};

struct GifDmaAdvanceResult
{
    DmacTransferToken transfer{};
    GifDmaPhase phase = GifDmaPhase::Idle;
    GifDmaStallReason stall = GifDmaStallReason::None;
    uint32_t delayEeCycles = 0u;
    uint32_t transferredQwc = 0u;
    bool progressed = false;
    bool completed = false;
    bool active = false;
};

struct GifDmaSnapshot
{
    DmacTransferToken transfer{};
    GifDmaPhase phase = GifDmaPhase::Idle;
    GifDmaStallReason stall = GifDmaStallReason::None;
    uint32_t chcr = 0u;
    uint32_t madr = 0u;
    uint32_t qwc = 0u;
    uint32_t tadr = 0u;
    uint32_t asr0 = 0u;
    uint32_t asr1 = 0u;
    uint32_t fifoQwc = 0u;
    uint32_t tagsProcessed = 0u;
    uint8_t asp = 0u;
    uint8_t tagId = 0u;
    bool active = false;
    bool eventManaged = false;
    bool chainMode = false;
    bool tagIrq = false;
    bool endAfterPayload = false;
};

enum class FromIpuDmaPhase : uint8_t
{
    Idle = 0u,
    TransferPayload,
    Finalize,
    Fault,
};

enum class FromIpuDmaStallReason : uint8_t
{
    None = 0u,
    OutputFifoEmpty,
    DmacDisabled,
    UnsupportedMode,
    Fault,
};

struct FromIpuDmaAdvanceResult
{
    DmacTransferToken transfer{};
    FromIpuDmaPhase phase = FromIpuDmaPhase::Idle;
    FromIpuDmaStallReason stall =
        FromIpuDmaStallReason::None;
    uint32_t delayEeCycles = 0u;
    uint32_t transferredQwc = 0u;
    bool progressed = false;
    bool completed = false;
    bool active = false;
};

struct FromIpuDmaSnapshot
{
    DmacTransferToken transfer{};
    FromIpuDmaPhase phase = FromIpuDmaPhase::Idle;
    FromIpuDmaStallReason stall =
        FromIpuDmaStallReason::None;
    uint32_t chcr = 0u;
    uint32_t madr = 0u;
    uint32_t qwc = 0u;
    uint32_t fifoQwc = 0u;
    bool active = false;
    bool eventManaged = false;
    bool normalMode = false;
};

enum class ToIpuDmaPhase : uint8_t
{
    Idle = 0u,
    FetchTag,
    TransferPayload,
    Finalize,
    Fault,
};

enum class ToIpuDmaStallReason : uint8_t
{
    None = 0u,
    InputFifoFull,
    WaitingForInputRequest,
    DmacDisabled,
    UnsupportedMode,
    Fault,
};

struct ToIpuDmaAdvanceResult
{
    DmacTransferToken transfer{};
    ToIpuDmaPhase phase = ToIpuDmaPhase::Idle;
    ToIpuDmaStallReason stall =
        ToIpuDmaStallReason::None;
    uint32_t delayEeCycles = 0u;
    uint32_t transferredQwc = 0u;
    bool progressed = false;
    bool completed = false;
    bool active = false;
};

struct ToIpuDmaSnapshot
{
    DmacTransferToken transfer{};
    ToIpuDmaPhase phase = ToIpuDmaPhase::Idle;
    ToIpuDmaStallReason stall =
        ToIpuDmaStallReason::None;
    uint32_t chcr = 0u;
    uint32_t madr = 0u;
    uint32_t qwc = 0u;
    uint32_t tadr = 0u;
    uint32_t fifoQwc = 0u;
    uint32_t tagsProcessed = 0u;
    uint8_t tagId = 0u;
    bool active = false;
    bool eventManaged = false;
    bool normalMode = false;
    bool chainMode = false;
    bool tagIrq = false;
    bool endAfterPayload = false;
};

enum class Vif0DmaPhase : uint8_t
{
    Idle = 0u,
    FetchTag,
    TransferPayload,
    Finalize,
    Fault,
};

enum class Vif0DmaStallReason : uint8_t
{
    None = 0u,
    WaitingForVu,
    VifStop,
    VifInterrupt,
    ForceBreak,
    DmacDisabled,
    Fault,
};

struct Vif0DmaAdvanceResult
{
    DmacTransferToken transfer{};
    Vif0DmaPhase phase = Vif0DmaPhase::Idle;
    Vif0DmaStallReason stall = Vif0DmaStallReason::None;
    uint32_t delayEeCycles = 0u;
    uint32_t acceptedBytes = 0u;
    bool progressed = false;
    bool completed = false;
    bool active = false;
};

struct Vif0DmaSnapshot
{
    DmacTransferToken transfer{};
    Vif0DmaPhase phase = Vif0DmaPhase::Idle;
    Vif0DmaStallReason stall = Vif0DmaStallReason::None;
    uint32_t madr = 0u;
    uint32_t qwc = 0u;
    uint32_t tadr = 0u;
    uint32_t asr0 = 0u;
    uint32_t asr1 = 0u;
    uint32_t parserBytes = 0u;
    uint32_t bufferedPayloadBytes = 0u;
    uint32_t payloadByteOffset = 0u;
    uint32_t tagsProcessed = 0u;
    uint8_t asp = 0u;
    uint8_t tagId = 0u;
    bool active = false;
    bool eventManaged = false;
    bool chainMode = false;
    bool tagIrq = false;
    bool endAfterPayload = false;
};

enum class Vif1DmaPhase : uint8_t
{
    Idle = 0u,
    FetchTag,
    TransferPayload,
    Finalize,
    Fault,
};

enum class Vif1DmaStallReason : uint8_t
{
    None = 0u,
    WaitingForVu,
    VifStop,
    VifInterrupt,
    ForceBreak,
    GifPath,
    DmacDisabled,
    Fault,
};

struct Vif1DmaAdvanceResult
{
    DmacTransferToken transfer{};
    Vif1DmaPhase phase = Vif1DmaPhase::Idle;
    Vif1DmaStallReason stall = Vif1DmaStallReason::None;
    uint32_t delayEeCycles = 0u;
    bool progressed = false;
    bool completed = false;
    bool active = false;
};

struct Vif1DmaSnapshot
{
    DmacTransferToken transfer{};
    Vif1DmaPhase phase = Vif1DmaPhase::Idle;
    Vif1DmaStallReason stall = Vif1DmaStallReason::None;
    uint32_t madr = 0u;
    uint32_t qwc = 0u;
    uint32_t tadr = 0u;
    uint32_t asr0 = 0u;
    uint32_t asr1 = 0u;
    uint32_t parserBytes = 0u;
    uint32_t tagsProcessed = 0u;
    uint8_t asp = 0u;
    uint8_t tagId = 0u;
    bool active = false;
    bool eventManaged = false;
    bool chainMode = false;
    bool tagIrq = false;
    bool endAfterPayload = false;
};

enum class ScratchpadDmaPhase : uint8_t
{
    Idle = 0u,
    FetchTag,
    TransferPayload,
    Finalize,
    Fault,
};

struct ScratchpadDmaAdvanceResult
{
    DmacTransferToken transfer{};
    ScratchpadDmaPhase phase = ScratchpadDmaPhase::Idle;
    uint32_t delayEeCycles = 0u;
    bool progressed = false;
    bool completed = false;
    bool active = false;
};

struct ScratchpadDmaSnapshot
{
    DmacTransferToken transfer{};
    ScratchpadDmaPhase phase = ScratchpadDmaPhase::Idle;
    uint32_t chcr = 0u;
    uint32_t madr = 0u;
    uint32_t qwc = 0u;
    uint32_t tadr = 0u;
    uint32_t asr0 = 0u;
    uint32_t asr1 = 0u;
    uint32_t sadr = 0u;
    uint32_t tagsProcessed = 0u;
    uint8_t asp = 0u;
    uint8_t tagId = 0u;
    bool active = false;
    bool eventManaged = false;
    bool chainMode = false;
    bool tagIrq = false;
    bool endAfterPayload = false;
};

struct JumpTable
{
    uint32_t address = 0;          // Base address of the jump table
    uint32_t baseRegister = 0;     // Register used for index
    std::vector<uint32_t> targets; // Jump targets
};

struct Vu1WorkloadProfileSnapshot
{
    bool enabled = false;
    bool measurementComplete = false;
    uint64_t warmupPairs = 0u;
    uint64_t pairLimit = 0u;
    uint64_t observedPairs = 0u;
    uint64_t measuredPairs = 0u;
    uint64_t invocations = 0u;
    uint64_t codeUploads = 0u;
    uint64_t identicalCodeUploads = 0u;
    uint64_t codeUploadBytes = 0u;
    uint64_t identicalCodeUploadBytes = 0u;
};

class PS2Memory : public IVuExecutionObserver
{
public:
    struct EeTranslatedAddress
    {
        uint32_t physicalAddress = 0u;
        bool scratchpad = false;

        constexpr bool operator==(
            const EeTranslatedAddress &) const noexcept = default;
    };

    struct CodeInvalidationEvent
    {
        uint32_t start = 0;
        uint32_t end = 0;
        uint32_t words = 0;
        uint32_t writerPc = 0;
    };

    PS2Memory();
    ~PS2Memory();

    PS2Memory(const PS2Memory &) = delete;
    PS2Memory &operator=(const PS2Memory &) = delete;
    PS2Memory(PS2Memory &&) = delete;
    PS2Memory &operator=(PS2Memory &&) = delete;

    // Initialize memory
    bool initialize(size_t ramSize = PS2_RAM_SIZE);

    // Memory access methods
    uint8_t *getRDRAM() { return m_rdram; }
    const uint8_t *getRDRAM() const { return m_rdram; }
    uint8_t *getScratchpad() { return m_scratchpad; }
    uint8_t *getIOPRAM() { return iop_ram; }
    uint64_t dmaStartCount() const { return m_dmaStartCount.load(std::memory_order_relaxed); }
    uint64_t gifCopyCount() const { return m_gifCopyCount.load(std::memory_order_relaxed); }
    uint64_t gsWriteCount() const { return m_gsWriteCount.load(std::memory_order_relaxed); }
    uint64_t vifWriteCount() const { return m_vifWriteCount.load(std::memory_order_relaxed); }
    uint64_t vif1DmaTraceSequence() const
    {
        return m_vif1DmaTraceSequence.load(std::memory_order_relaxed);
    }
    bool debugStartVif1DmaTrace(
        const std::string &outputPath,
        uint64_t sequenceCount,
        const std::string &vif1InputDumpDirectory,
        const std::string &path1DumpDirectory,
        const std::string &vu1DumpDirectory,
        std::string *diagnostic = nullptr);
    void debugStopVif1DmaTrace();
    [[nodiscard]] Vif1DmaTraceSnapshot
    debugVif1DmaTraceSnapshot() const;
    uint64_t getVU0CodeGeneration() const { return m_vu0CodeGeneration.load(std::memory_order_relaxed); }
    uint64_t getVU1CodeGeneration() const { return m_vu1CodeGeneration.load(std::memory_order_relaxed); }
    [[nodiscard]] bool vuTraceEnabled() const override;
    [[nodiscard]] bool vuWorkloadProfileEnabled() const override;
    [[nodiscard]] uint64_t currentVuCodeGeneration(
        VuUnitId unit) const override;
    void traceVuInvocation(
        uint32_t startPc, uint32_t top, uint32_t itop,
        bool resume, const VuExecutionState &state) override;
    void traceVuInstruction(
        uint32_t pc, uint32_t lower, uint32_t upper,
        const VuExecutionState &state) override;
    void traceVuXgkick(uint32_t sourceQword) override;
    void traceVuInvocationEnd(
        uint32_t finalPc, bool ended, bool hitCycleLimit,
        const int32_t *viRegisters,
        size_t viRegisterCount) override;
    void beginVuWorkloadProfileInvocation(
        uint32_t startPc, const uint8_t *code,
        uint32_t codeSize, uint64_t generation) override;
    void recordVuWorkloadProfileInstruction(
        uint32_t pc, uint32_t lower, uint32_t upper) override;
    void recordVuWorkloadProfileTransition(
        uint32_t pc, uint32_t nextPc) override;
    void endVuWorkloadProfileInvocation(bool completed) override;
    void resetVuWorkloadProfileEpoch() override;
    bool configureVu1WorkloadProfile(
        const char *outputPath,
        uint64_t warmupPairs = 0u,
        uint64_t pairLimit = 0u);
    void finishVu1WorkloadProfile();
    [[nodiscard]] Vu1WorkloadProfileSnapshot
    vu1WorkloadProfileSnapshot() const;
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    [[nodiscard]] bool isVu1WorkloadProfileEnabled() const;
    void beginVu1WorkloadProfileInvocation(uint32_t startPc);
    void recordVu1WorkloadProfileInstruction(
        uint32_t pc, uint32_t lower, uint32_t upper);
    void recordVu1WorkloadProfileTransition(
        uint32_t pc, uint32_t nextPc);
    void endVu1WorkloadProfileInvocation(bool completed);
    void resetVu1WorkloadProfileEpoch();
#else
    [[nodiscard]] bool
    isVu1WorkloadProfileEnabled() const
    {
        return false;
    }
    void beginVu1WorkloadProfileInvocation(uint32_t)
    {
    }
    void recordVu1WorkloadProfileInstruction(
        uint32_t, uint32_t, uint32_t)
    {
    }
    void recordVu1WorkloadProfileTransition(
        uint32_t, uint32_t)
    {
    }
    void endVu1WorkloadProfileInvocation(bool)
    {
    }
    void resetVu1WorkloadProfileEpoch()
    {
    }
#endif

    // Read/write memory
    uint8_t read8(
        uint32_t address,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked());
    uint16_t read16(
        uint32_t address,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked());
    uint32_t read32(
        uint32_t address,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked());
    uint64_t read64(
        uint32_t address,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked());
    __m128i read128(
        uint32_t address,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked());

    void write8(
        uint32_t address,
        uint8_t value,
        uint32_t writerPc = 0,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked());
    void write16(
        uint32_t address,
        uint16_t value,
        uint32_t writerPc = 0,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked());
    void write32(
        uint32_t address,
        uint32_t value,
        uint32_t writerPc = 0,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked());
    void write64(
        uint32_t address,
        uint64_t value,
        uint32_t writerPc = 0,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked());
    void write128(
        uint32_t address,
        __m128i value,
        uint32_t writerPc = 0,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked());
    void writeMasked32(
        uint32_t address,
        uint32_t value,
        uint8_t byteEnable,
        uint32_t writerPc = 0,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked());
    void writeMasked64(
        uint32_t address,
        uint64_t value,
        uint8_t byteEnable,
        uint32_t writerPc = 0,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked());

    // Publish the non-memory side effects of a write whose destination has
    // already been proven to be a physical RDRAM interval. Generated direct
    // and cached-mapping stores share this with the authoritative store path.
    inline void observeRdramWrite(
        uint32_t physicalAddress,
        uint32_t size,
        uint32_t writerPc)
    {
        if (!m_hasCodeRegions)
        {
            return;
        }
        observeRdramWriteSlow(
            physicalAddress, size, writerPc);
    }

    // Reject the common non-code case inline for generated fixed-width
    // stores. Accesses of at most 16 bytes can touch no more than two RDRAM
    // pages, while the slow path retains the exact region intersection and
    // invalidation aggregation behavior for pages that may contain code.
    template <uint32_t AccessSize>
    inline void observeRdramWriteFixed(
        uint32_t physicalAddress,
        uint32_t writerPc)
    {
        static_assert(
            AccessSize != 0u && AccessSize <= 16u,
            "fixed RDRAM writes must be between 1 and 16 bytes");

        if (!m_hasCodeRegions ||
            physicalAddress > PS2_RAM_SIZE - AccessSize)
        {
            return;
        }

        const uint32_t firstPage =
            physicalAddress >> kCodeRegionPageShift;
        const uint32_t lastPage =
            (physicalAddress + AccessSize - 1u) >>
            kCodeRegionPageShift;
        const auto pageMayContainCode = [this](uint32_t page)
        {
            const uint64_t pageBit =
                uint64_t{1u} << (page & 63u);
            return (m_codeRegionPages[page >> 6u] & pageBit) != 0u;
        };

        if (!pageMayContainCode(firstPage) &&
            (lastPage == firstPage ||
             !pageMayContainCode(lastPage)))
        {
            return;
        }

        observeRdramWriteSlow(
            physicalAddress, AccessSize, writerPc);
    }
    void observeRdramWriteSlow(
        uint32_t physicalAddress,
        uint32_t size,
        uint32_t writerPc);

    // TLB handling
    uint32_t translateAddress(
        uint32_t virtualAddress,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked(),
        bool writeAccess = false,
        uint32_t accessSize = 1u);
    EeTranslatedAddress resolveAddress(
        uint32_t virtualAddress,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked(),
        bool writeAccess = false,
        uint32_t accessSize = 1u);
    uint32_t translateAddressUncached(
        uint32_t virtualAddress,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked(),
        bool writeAccess = false,
        uint32_t accessSize = 1u);
    EeTranslatedAddress resolveAddressUncached(
        uint32_t virtualAddress,
        EeAddressTranslationContext translation =
            EeAddressTranslationContext::unchecked(),
        bool writeAccess = false,
        uint32_t accessSize = 1u);
    bool tlbRead(uint32_t index, EeTlbEntry &entry) const;
    bool tlbWrite(uint32_t index, const EeTlbEntry &entry);
    int32_t tlbProbe(uint32_t entryHi) const;
    void installPostBiosTlbState();
    size_t tlbEntryCount() const { return m_tlbEntries.size(); }
    uint64_t tlbMappingGeneration() const noexcept
    {
        return m_tlbMappingGeneration;
    }
    void setTlbTranslationCacheEnabled(bool enabled) noexcept
    {
        if (m_tlbTranslationCacheEnabled == enabled)
        {
            return;
        }
        m_tlbTranslationCacheEnabled = enabled;
        m_eeInstructionFetchPageCache = {};
    }
    bool tlbTranslationCacheEnabled() const noexcept
    {
        return m_tlbTranslationCacheEnabled;
    }
    void setTlbTranslationCacheDiagnosticsEnabled(
        bool enabled) noexcept
    {
        if (m_tlbTranslationCacheDiagnosticsEnabled == enabled)
        {
            return;
        }
        m_tlbTranslationCacheDiagnosticsEnabled = enabled;
        m_eeInstructionFetchPageCache = {};
    }
    bool tlbTranslationCacheDiagnosticsEnabled() const noexcept
    {
        return m_tlbTranslationCacheDiagnosticsEnabled;
    }
    bool tryReuseValidatedInstructionFetchRdramPage(
        uint32_t translationPageKey) const noexcept
    {
        return m_eeInstructionFetchPageCache.tlbMappingGeneration ==
                   m_tlbMappingGeneration &&
               m_eeInstructionFetchPageCache.translationPageKey ==
                   translationPageKey;
    }
    void rememberValidatedInstructionFetchRdramPage(
        uint32_t translationPageKey) noexcept
    {
        if (!m_tlbTranslationCacheEnabled ||
            m_tlbTranslationCacheDiagnosticsEnabled)
        {
            return;
        }
        m_eeInstructionFetchPageCache = {
            m_tlbMappingGeneration,
            translationPageKey,
        };
    }
    void resetTlbTranslationCacheStats() noexcept
    {
        m_tlbTranslationCacheStats = {};
    }
    EeTlbTranslationCacheStats
    tlbTranslationCacheStats() const noexcept
    {
        return m_tlbTranslationCacheStats;
    }

    // Cache-only lookup for generated accesses. A false result deliberately
    // says nothing about the architectural outcome: callers must use the
    // authoritative runtime path to fill the cache or raise the exact fault.
    bool tryResolveCachedTlbRdramOffset(
        uint32_t virtualAddress,
        EeAddressTranslationContext translation,
        bool writeAccess,
        uint32_t accessSize,
        uint32_t &physicalOffset) noexcept;

    template <uint32_t AccessSize, bool WriteAccess>
    bool tryResolveCachedTlbRdramOffsetFixed(
        uint32_t virtualAddress,
        EeAddressTranslationContext translation,
        uint32_t &physicalOffset) noexcept
    {
        static_assert(
            AccessSize != 0u &&
            AccessSize <= sizeof(__m128i) &&
            (AccessSize & (AccessSize - 1u)) == 0u);

        if (!m_tlbTranslationCacheEnabled)
        {
            return false;
        }

        constexpr uint32_t pageMask =
            kTlbTranslationCachePageSize - 1u;
        const uint32_t pageOffset = virtualAddress & pageMask;
        if (!translation.permits(virtualAddress))
        {
            if (m_tlbTranslationCacheDiagnosticsEnabled)
            {
                recordTlbFastCacheRejectedAccess(false);
            }
            return false;
        }
        if (pageOffset >
            kTlbTranslationCachePageSize - AccessSize)
        {
            if (m_tlbTranslationCacheDiagnosticsEnabled)
            {
                recordTlbFastCacheRejectedAccess(true);
            }
            return false;
        }
        if ((virtualAddress & (AccessSize - 1u)) != 0u)
        {
            if (m_tlbTranslationCacheDiagnosticsEnabled)
            {
                recordTlbFastCacheRejectedAccess(false);
            }
            return false;
        }

        const size_t setIndex =
            (virtualAddress >> 12u) &
            (kTlbFastCacheSetCount - 1u);
        const EeTlbFastCacheEntry &entry =
            m_tlbFastCache[setIndex];
        const uint64_t requestKey = makeTlbFastCacheKey(
            virtualAddress,
            translation,
            WriteAccess);
        const bool generationMatches =
            entry.generation == m_tlbMappingGeneration;
        const bool keyMatches =
            ((requestKey ^ entry.key) & entry.matchMask) == 0u;
        const uint32_t resolved =
            entry.physicalPageBase + pageOffset;
        if (generationMatches &&
            keyMatches &&
            resolved <= PS2_RAM_SIZE - AccessSize)
        {
            if (m_tlbTranslationCacheDiagnosticsEnabled)
            {
                recordTlbFastCacheHit();
            }
            physicalOffset = resolved;
            return true;
        }

        if (m_tlbTranslationCacheDiagnosticsEnabled)
        {
            recordTlbFastCacheMiss(
                entry,
                requestKey,
                WriteAccess);
        }
        return false;
    }

    // Hardware register interface
    bool writeIORegister(uint32_t address, uint32_t value);
    bool writeIORegisterMasked(uint32_t address, uint32_t value, uint8_t byteEnable);
    uint32_t readIORegister(uint32_t address);

    using EeCounterCycleCallback = std::function<uint64_t()>;
    using EeCounterScheduleCallback =
        std::function<void(std::optional<uint64_t>)>;
    using EeCounterInterruptStateCallback =
        std::function<void(
            uint32_t status,
            uint32_t mask,
            uint32_t newlyRaised)>;
    void setEeCounterCycleCallback(
        EeCounterCycleCallback callback)
    {
        m_eeCounterCycleCallback = std::move(callback);
    }
    void setEeCounterScheduleCallback(
        EeCounterScheduleCallback callback)
    {
        m_eeCounterScheduleCallback =
            std::move(callback);
    }
    void setEeCounterInterruptStateCallback(
        EeCounterInterruptStateCallback callback)
    {
        m_eeCounterInterruptStateCallback =
            std::move(callback);
    }
    void resetEeCounters(uint64_t currentCycle = 0u);
    void configureEeCounterVideoTiming(
        ps2x::timing::EeCounterVideoTiming timing,
        uint64_t currentCycle);
    ps2x::timing::EeCounterAdvanceResult
    synchronizeEeCounters(uint64_t currentCycle);
    [[nodiscard]] ps2x::timing::EeCounterBankSnapshot
    eeCounterSnapshot() const noexcept
    {
        return m_eeCounters.snapshot();
    }
    [[nodiscard]] uint32_t intcStatus() const noexcept;
    [[nodiscard]] uint32_t intcMask() const noexcept;
    void setIntcInterruptMask(
        uint32_t cause, bool enabled);

    using GifPacketCallback = std::function<void(const uint8_t *, uint32_t)>;
    void setGifPacketCallback(GifPacketCallback cb) { m_gifPacketCallback = std::move(cb); }
    void setGifArbiter(GifArbiter *arbiter) { m_gifArbiter = arbiter; }
    using GsPrivilegedObservationCallback =
        std::function<void(uint32_t address, bool write)>;
    void setGsPrivilegedObservationCallback(
        GsPrivilegedObservationCallback callback)
    {
        m_gsPrivilegedObservationCallback =
            std::move(callback);
    }

    using Vu0MscalCallback =
        std::function<void(uint32_t startPC, uint32_t itop)>;
    void setVu0MscalCallback(Vu0MscalCallback cb)
    {
        m_vu0MscalCallback = std::move(cb);
    }
    using Vu0MscntCallback = std::function<void(uint32_t itop)>;
    void setVu0MscntCallback(Vu0MscntCallback cb)
    {
        m_vu0MscntCallback = std::move(cb);
    }
    using Vu0BusyCallback = std::function<bool()>;
    void setVu0BusyCallback(Vu0BusyCallback cb)
    {
        m_vu0BusyCallback = std::move(cb);
    }
    using Vif0ResetCallback = std::function<void()>;
    void setVif0ResetCallback(Vif0ResetCallback cb)
    {
        m_vif0ResetCallback = std::move(cb);
    }
    using Vif0DmaScheduleCallback =
        std::function<bool(uint32_t delayEeCycles)>;
    void setVif0DmaScheduleCallback(
        Vif0DmaScheduleCallback cb)
    {
        m_vif0DmaScheduleCallback = std::move(cb);
    }
    using Vif0DmaCancelCallback = std::function<void()>;
    void setVif0DmaCancelCallback(Vif0DmaCancelCallback cb)
    {
        m_vif0DmaCancelCallback = std::move(cb);
    }

    using Vu1MscalCallback = std::function<void(
        uint32_t startPC, uint32_t top, uint32_t itop,
        bool flushBeforeStart)>;
    void setVu1MscalCallback(Vu1MscalCallback cb) { m_vu1MscalCallback = std::move(cb); }
    using Vu1MscntCallback = std::function<void(uint32_t top, uint32_t itop)>;
    void setVu1MscntCallback(Vu1MscntCallback cb) { m_vu1MscntCallback = std::move(cb); }
    using Vu1BusyCallback = std::function<bool()>;
    void setVu1BusyCallback(Vu1BusyCallback cb) { m_vu1BusyCallback = std::move(cb); }
    using Vu1CommandCallback =
        std::function<Vu1CommandResult(Vu1CommandPayload)>;
    void setVu1CommandCallback(Vu1CommandCallback cb)
    {
        m_vu1CommandCallback = std::move(cb);
    }
    using Vu1DecodedUnpackCallback =
        std::function<Vu1CommandResult(Vu1DecodedUnpackCommand)>;
    void setVu1DecodedUnpackCallback(
        Vu1DecodedUnpackCallback cb)
    {
        m_vu1DecodedUnpackCallback = std::move(cb);
    }
    using Vu1VifStateUpdateCallback =
        std::function<Vu1CommandResult(Vu1VifStateUpdateCommand)>;
    void setVu1VifStateUpdateCallback(
        Vu1VifStateUpdateCallback cb)
    {
        m_vu1VifStateUpdateCallback = std::move(cb);
    }
    using Vu1MemoryObservationCallback = std::function<void()>;
    void setVu1MemoryObservationCallback(
        Vu1MemoryObservationCallback callback)
    {
        m_vu1MemoryObservationCallback =
            std::move(callback);
    }
    void publishVu1MemorySnapshot(
        const Vu1Snapshot &snapshot);
    using Vif1ResetCallback = std::function<void()>;
    void setVif1ResetCallback(Vif1ResetCallback cb) { m_vif1ResetCallback = std::move(cb); }
    using Vif1DmaScheduleCallback =
        std::function<bool(uint32_t delayEeCycles)>;
    void setVif1DmaScheduleCallback(
        Vif1DmaScheduleCallback cb)
    {
        m_vif1DmaScheduleCallback = std::move(cb);
    }
    using Vif1DmaCancelCallback = std::function<void()>;
    void setVif1DmaCancelCallback(Vif1DmaCancelCallback cb)
    {
        m_vif1DmaCancelCallback = std::move(cb);
    }
    using ScratchpadDmaScheduleCallback =
        std::function<bool(
            DmacChannel channel,
            uint32_t delayEeCycles)>;
    void setScratchpadDmaScheduleCallback(
        ScratchpadDmaScheduleCallback cb)
    {
        m_scratchpadDmaScheduleCallback =
            std::move(cb);
    }
    using ScratchpadDmaCancelCallback =
        std::function<void(DmacChannel channel)>;
    void setScratchpadDmaCancelCallback(
        ScratchpadDmaCancelCallback cb)
    {
        m_scratchpadDmaCancelCallback =
            std::move(cb);
    }
    using GifDmaScheduleCallback =
        std::function<bool(uint32_t delayEeCycles)>;
    void setGifDmaScheduleCallback(
        GifDmaScheduleCallback cb)
    {
        m_gifDmaScheduleCallback = std::move(cb);
    }
    using GifDmaCancelCallback = std::function<void()>;
    void setGifDmaCancelCallback(
        GifDmaCancelCallback cb)
    {
        m_gifDmaCancelCallback = std::move(cb);
    }
    using FromIpuDmaScheduleCallback =
        std::function<bool(uint32_t delayEeCycles)>;
    void setFromIpuDmaScheduleCallback(
        FromIpuDmaScheduleCallback cb)
    {
        m_fromIpuDmaScheduleCallback = std::move(cb);
    }
    using FromIpuDmaCancelCallback = std::function<void()>;
    void setFromIpuDmaCancelCallback(
        FromIpuDmaCancelCallback cb)
    {
        m_fromIpuDmaCancelCallback = std::move(cb);
    }
    using ToIpuDmaScheduleCallback =
        std::function<bool(uint32_t delayEeCycles)>;
    void setToIpuDmaScheduleCallback(
        ToIpuDmaScheduleCallback cb)
    {
        m_toIpuDmaScheduleCallback = std::move(cb);
    }
    using ToIpuDmaCancelCallback = std::function<void()>;
    void setToIpuDmaCancelCallback(
        ToIpuDmaCancelCallback cb)
    {
        m_toIpuDmaCancelCallback = std::move(cb);
    }
    using Vif1GifPathAvailableCallback =
        std::function<bool(bool directHl)>;
    void setVif1GifPathAvailableCallback(
        Vif1GifPathAvailableCallback cb)
    {
        m_vif1GifPathAvailableCallback = std::move(cb);
    }

    [[nodiscard]] bool vif0WaitingForVu() const { return m_vif0WaitingForVu; }
    [[nodiscard]] size_t vif0DeferredByteCount() const { return m_vif0DeferredData.size(); }
    bool resumeVIF0AfterVu();
    void cancelVIF0VuWait();
    bool wakeVif0Dma();
    Vif0DmaAdvanceResult advanceVif0Dma();
    [[nodiscard]] Vif0DmaSnapshot vif0DmaSnapshot() const;

    [[nodiscard]] bool vif1WaitingForVu() const { return m_vif1WaitingForVu; }
    [[nodiscard]] size_t vif1DeferredByteCount() const { return m_vif1DeferredData.size(); }
    bool resumeVIF1AfterVu();
    void cancelVIF1VuWait();
    bool wakeVif1Dma();
    Vif1DmaAdvanceResult advanceVif1Dma();
    [[nodiscard]] Vif1DmaSnapshot vif1DmaSnapshot() const;
    bool wakeGifDma(uint32_t delayEeCycles);
    GifDmaAdvanceResult advanceGifDma();
    [[nodiscard]] GifDmaSnapshot gifDmaSnapshot() const;
    uint32_t writeIpuOutputFifo(
        const uint8_t *data, uint32_t qwc);
    bool wakeFromIpuDma(uint32_t delayEeCycles);
    void setFromIpuDmaEventPending(
        bool pending) noexcept;
    FromIpuDmaAdvanceResult advanceFromIpuDma();
    [[nodiscard]] FromIpuDmaSnapshot
    fromIpuDmaSnapshot() const;
    bool wakeToIpuDma(uint32_t delayEeCycles);
    void setToIpuDmaEventPending(
        bool pending) noexcept;
    ToIpuDmaAdvanceResult advanceToIpuDma();
    [[nodiscard]] ToIpuDmaSnapshot
    toIpuDmaSnapshot() const;
    ScratchpadDmaAdvanceResult advanceScratchpadDma(
        DmacChannel channel);
    [[nodiscard]] ScratchpadDmaSnapshot
    scratchpadDmaSnapshot(DmacChannel channel) const;

    uint8_t *getVU1Code()
    {
        observeVu1Memory();
        return m_vu1Code;
    }
    const uint8_t *getVU1Code() const
    {
        observeVu1Memory();
        return m_vu1Code;
    }
    uint8_t *getVU1Data()
    {
        observeVu1Memory();
        return m_vu1Data;
    }
    const uint8_t *getVU1Data() const
    {
        observeVu1Memory();
        return m_vu1Data;
    }
    uint8_t *getVU0Code() { return m_vu0Code; }
    const uint8_t *getVU0Code() const { return m_vu0Code; }
    uint8_t *getVU0Data() { return m_vu0Data; }
    const uint8_t *getVU0Data() const { return m_vu0Data; }

    bool isPath3Masked() const
    {
        return m_path3Masked ||
               m_gifModePath3Masked;
    }
    void flushMaskedPath3Packets(bool drainImmediately = true);

    void submitGifPacket(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool drainImmediately = true, bool path2DirectHl = false);
    void processGIFPacket(uint32_t srcPhysAddr, uint32_t qwCount);
    void processGIFPacket(const uint8_t *data, uint32_t sizeBytes);
    // Native DMA shortcuts are fixture-only entry points. Their decoded
    // payloads still cross the canonical GS command boundary, so they cannot
    // retain guest-memory pointers or mutate GS state directly.
    bool tryProcessNativeGifImageUploadChain(
        GsCommandExecutor &executor, uint32_t tadr, uint32_t chcr);
    bool tryProcessNativeGifPackedChain(
        GsCommandExecutor &executor, uint32_t tadr, uint32_t chcr);
    void processVIF0Data(uint32_t srcPhysAddr, uint32_t sizeBytes);
    void processVIF0Data(const uint8_t *data, uint32_t sizeBytes);
    void processVIF1Data(uint32_t srcPhysAddr, uint32_t sizeBytes);
    void processVIF1Data(const uint8_t *data, uint32_t sizeBytes);
    void processPendingTransfers();

    // DMAC work and completion publication intentionally have different
    // owners. Device code begins/progresses/finishes a generation-tagged
    // operation; only completion service mutates completion-visible channel
    // state and publishes an interrupt.
    DmacTransferToken beginDmacTransfer(DmacChannel channel);
    bool progressDmacTransfer(DmacTransferToken transfer) const;
    bool requestDmacCompletion(DmacTransferToken transfer);
    bool cancelDmacTransfer(DmacChannel channel);
    bool hasReadyDmacCompletions() const;
    std::vector<DmacCompletionRequest> consumeReadyDmacCompletions();
    bool publishDmacCompletion(
        DmacCompletionRequest completion,
        uint64_t eventSequence);
    size_t publishReadyDmacCompletions(uint64_t eventSequence);
    std::vector<DmacPendingInterrupt> consumePendingDmacInterrupts();
    DmacChannelSnapshot dmacChannelSnapshot(DmacChannel channel) const;
    bool dmacInterruptStatusLatched(DmacChannel channel) const;
    bool dmacInterruptUnmasked(DmacChannel channel) const;
    void setDmacInterruptMask(DmacChannel channel, bool enabled);

    using DmacCompletionReadyCallback = std::function<void()>;
    void setDmacCompletionReadyCallback(
        DmacCompletionReadyCallback callback)
    {
        m_dmacCompletionReadyCallback = std::move(callback);
    }
    using DmacInterruptStateCallback = std::function<void()>;
    void setDmacInterruptStateCallback(
        DmacInterruptStateCallback callback)
    {
        m_dmacInterruptStateCallback = std::move(callback);
    }

    int pollDmaRegisters();

    // Track code modifications for self-modifying code
    void registerCodeRegion(uint32_t start, uint32_t end);
    bool isCodeAddress(uint32_t address) const;
    bool isCodeModified(uint32_t address, uint32_t size);
    std::vector<CodeInvalidationEvent> takeCodeInvalidationEvents();
    void flushCodeInvalidationLog();
    void clearModifiedFlag(uint32_t address, uint32_t size);

    // GS register accessors
    GSRegisters &gs() { return gs_regs; }
    const GSRegisters &gs() const { return gs_regs; }
    uint8_t *getGSVRAM() { return m_gsVRAM; }
    const uint8_t *getGSVRAM() const { return m_gsVRAM; }
    bool hasSeenGifCopy() const { return m_seenGifCopy; }
    // Main RAM (32MB)
    uint8_t *m_rdram;

    // Scratchpad memory (16KB)
    uint8_t *m_scratchpad;

    // IOP RAM (2MB)
    uint8_t *iop_ram;

    bool m_seenGifCopy;
    std::atomic<uint64_t> m_dmaStartCount{0};
    std::atomic<uint64_t> m_gifCopyCount{0};
    std::atomic<uint64_t> m_gsWriteCount{0};
    std::atomic<uint64_t> m_vifWriteCount{0};
    std::atomic<uint64_t> m_vu0CodeGeneration{0};
    std::atomic<uint64_t> m_vu1CodeGeneration{0};
    // I/O registers
    std::unordered_map<uint32_t, uint32_t> m_ioRegisters;

    // Registers
    GSRegisters gs_regs;
    uint8_t *m_gsVRAM;
    VIFRegisters vif0_regs;
    VIFRegisters vif1_regs;
    DMARegisters dma_regs[10]; // 10 DMA channels

    static constexpr size_t kTlbTranslationCacheSetCount = 32u;
    static constexpr size_t kTlbTranslationCacheWayCount = 4u;
    static constexpr size_t kTlbFastCacheSetCount = 64u;
    static constexpr uint32_t kTlbTranslationCachePageSize = 0x1000u;
    static_assert(
        (kTlbTranslationCacheSetCount &
         (kTlbTranslationCacheSetCount - 1u)) == 0u);

    struct EeTlbTranslationCacheEntry
    {
        uint64_t generation = 0u;
        uint32_t virtualPageBase = 0u;
        uint32_t physicalPageBase = 0u;
        uint32_t tlbPageSize = 0u;
        uint8_t asid = 0u;
        EeAddressSpaceMode mode = EeAddressSpaceMode::Kernel;
        bool enforceOperatingMode = false;
        bool errorLevel = false;
        bool valid = false;
        bool global = false;
        bool writable = false;
        bool scratchpad = false;
    };

    struct EeTlbFastCacheEntry
    {
        uint64_t generation = 0u;
        uint64_t key = 0u;
        uint64_t matchMask = 0u;
        uint32_t physicalPageBase = 0u;
    };

    static uint64_t makeTlbFastCacheKey(
        uint32_t virtualAddress,
        EeAddressTranslationContext translation,
        bool writeAccess) noexcept
    {
        return
            static_cast<uint64_t>(
                virtualAddress &
                ~(kTlbTranslationCachePageSize - 1u)) |
            static_cast<uint64_t>(translation.asid) |
            (static_cast<uint64_t>(translation.mode) << 8u) |
            (static_cast<uint64_t>(
                 translation.enforceOperatingMode)
             << 10u) |
            (static_cast<uint64_t>(translation.errorLevel)
             << 11u) |
            (static_cast<uint64_t>(writeAccess) << 32u);
    }

    EeTranslatedAddress translateAddressImpl(
        uint32_t virtualAddress,
        EeAddressTranslationContext translation,
        bool writeAccess,
        uint32_t accessSize = 1u,
        bool allowTlbCache = true);
    std::optional<EeTranslatedAddress>
    lookupTlbTranslationCache(
        uint32_t virtualAddress,
        EeAddressTranslationContext translation,
        bool writeAccess,
        uint32_t accessSize);
    void fillTlbTranslationCache(
        uint32_t virtualAddress,
        const EeTranslatedAddress &translated,
        EeAddressTranslationContext translation,
        uint32_t tlbPageSize,
        bool global,
        bool writable);
    void fillTlbFastCache(
        uint32_t virtualPageBase,
        uint32_t physicalPageBase,
        EeAddressTranslationContext translation,
        bool global,
        bool writable) noexcept;
    void recordTlbFastCacheRejectedAccess(
        bool intervalCrossing) noexcept;
    void recordTlbFastCacheHit() noexcept;
    void recordTlbFastCacheMiss(
        const EeTlbFastCacheEntry &entry,
        uint64_t requestKey,
        bool writeAccess) noexcept;
    void advanceTlbMappingGeneration() noexcept;
    void clearTlbTranslationCache() noexcept;
    void recordTlbTranslationSlowPathFault() noexcept;

    std::vector<EeTlbEntry> m_tlbEntries;
    bool m_postBiosTlbStateConfigured = false;
    uint64_t m_tlbMappingGeneration = 1u;
    bool m_tlbTranslationCacheEnabled = true;
    bool m_tlbTranslationCacheDiagnosticsEnabled = false;
    // A successful aligned instruction fetch authorizes its complete minimum
    // 4 KiB translation page. Keeping this beside the TLB caches lets the
    // generation-wrap clear invalidate every translation cache together.
    struct EeInstructionFetchPageCache
    {
        uint64_t tlbMappingGeneration = 0u;
        uint32_t translationPageKey = 0u;
    };
    EeInstructionFetchPageCache
        m_eeInstructionFetchPageCache{};
    EeTlbTranslationCacheStats m_tlbTranslationCacheStats{};
    std::array<
        std::array<
            EeTlbTranslationCacheEntry,
            kTlbTranslationCacheWayCount>,
        kTlbTranslationCacheSetCount>
        m_tlbTranslationCache{};
    std::array<uint8_t, kTlbTranslationCacheSetCount>
        m_tlbTranslationCacheNextWay{};
    std::array<EeTlbFastCacheEntry, kTlbFastCacheSetCount>
        m_tlbFastCache{};

    GifPacketCallback m_gifPacketCallback;
    GifArbiter *m_gifArbiter = nullptr;
    GsPrivilegedObservationCallback
        m_gsPrivilegedObservationCallback;
    Vu0MscalCallback m_vu0MscalCallback;
    Vu0MscntCallback m_vu0MscntCallback;
    Vu0BusyCallback m_vu0BusyCallback;
    Vif0ResetCallback m_vif0ResetCallback;
    Vif0DmaScheduleCallback m_vif0DmaScheduleCallback;
    Vif0DmaCancelCallback m_vif0DmaCancelCallback;
    Vu1MscalCallback m_vu1MscalCallback;
    Vu1MscntCallback m_vu1MscntCallback;
    Vu1BusyCallback m_vu1BusyCallback;
    Vu1CommandCallback m_vu1CommandCallback;
    Vu1DecodedUnpackCallback m_vu1DecodedUnpackCallback;
    Vu1VifStateUpdateCallback m_vu1VifStateUpdateCallback;
    Vu1MemoryObservationCallback
        m_vu1MemoryObservationCallback;
    Vif1ResetCallback m_vif1ResetCallback;
    Vif1DmaScheduleCallback m_vif1DmaScheduleCallback;
    Vif1DmaCancelCallback m_vif1DmaCancelCallback;
    ScratchpadDmaScheduleCallback
        m_scratchpadDmaScheduleCallback;
    ScratchpadDmaCancelCallback
        m_scratchpadDmaCancelCallback;
    GifDmaScheduleCallback m_gifDmaScheduleCallback;
    GifDmaCancelCallback m_gifDmaCancelCallback;
    FromIpuDmaScheduleCallback
        m_fromIpuDmaScheduleCallback;
    FromIpuDmaCancelCallback
        m_fromIpuDmaCancelCallback;
    ToIpuDmaScheduleCallback
        m_toIpuDmaScheduleCallback;
    ToIpuDmaCancelCallback m_toIpuDmaCancelCallback;
    Vif1GifPathAvailableCallback
        m_vif1GifPathAvailableCallback;

    uint8_t *m_vu0Code = nullptr;
    uint8_t *m_vu0Data = nullptr;
    uint8_t *m_vu1Code = nullptr;
    uint8_t *m_vu1Data = nullptr;
    // PATH3 has two independent masks: VIF1 MSKPATH3 publishes M3P while
    // GIF_MODE publishes M3R.
    bool m_path3Masked = false;
    bool m_gifModePath3Masked = false;
    bool m_vif0WaitingForVu = false;
    std::vector<uint8_t> m_vif0DeferredData;
    uint32_t m_vif1PendingPath2DirectQwc = 0u;
    bool m_vif1PendingPath2DirectHl = false;
    bool m_vif1PendingIrqAfterCommand = false;
    bool m_vif1WaitingForVu = false;
    std::vector<uint8_t> m_vif1DeferredData;
    struct Vif1StreamSegment
    {
        size_t byteCount = 0u;
        uint8_t firstByteOffset = 0u;
    };
    std::deque<Vif1StreamSegment>
        m_vif1DeferredSegments;
    uint8_t m_vif1NextByteOffset = 0u;
    std::vector<std::vector<uint8_t>> m_path3MaskedFifo;

    struct DmacSourceChainKey
    {
        uint32_t tadr = 0u;
        uint32_t asr0 = 0u;
        uint32_t asr1 = 0u;
        uint8_t asp = 0u;

        friend bool operator==(
            const DmacSourceChainKey &,
            const DmacSourceChainKey &) = default;
    };

    struct DmacSourceChainKeyHash
    {
        size_t operator()(
            const DmacSourceChainKey &key) const noexcept
        {
            size_t hash = static_cast<size_t>(key.tadr);
            hash ^= static_cast<size_t>(key.asr0) +
                    0x9e3779b9u + (hash << 6u) +
                    (hash >> 2u);
            hash ^= static_cast<size_t>(key.asr1) +
                    0x9e3779b9u + (hash << 6u) +
                    (hash >> 2u);
            hash ^= static_cast<size_t>(key.asp) +
                    0x9e3779b9u + (hash << 6u) +
                    (hash >> 2u);
            return hash;
        }
    };

    struct Vif0DmaState
    {
        DmacTransferToken transfer{};
        Vif0DmaPhase phase = Vif0DmaPhase::Idle;
        Vif0DmaStallReason stall =
            Vif0DmaStallReason::None;
        uint32_t madr = 0u;
        uint32_t qwc = 0u;
        uint32_t tadr = 0u;
        uint32_t asr0 = 0u;
        uint32_t asr1 = 0u;
        uint32_t bufferedPayloadBytes = 0u;
        uint32_t payloadByteOffset = 0u;
        uint32_t tagsProcessed = 0u;
        uint8_t asp = 0u;
        uint8_t tagId = 0u;
        bool active = false;
        bool eventManaged = false;
        bool chainMode = false;
        bool tie = false;
        bool tte = false;
        bool tagIrq = false;
        bool endAfterPayload = false;
        bool faultReported = false;
        std::unordered_set<
            DmacSourceChainKey,
            DmacSourceChainKeyHash>
            visitedStates;
    };
    Vif0DmaState m_vif0Dma{};

    struct Vif1DmaState
    {
        DmacTransferToken transfer{};
        Vif1DmaPhase phase = Vif1DmaPhase::Idle;
        Vif1DmaStallReason stall =
            Vif1DmaStallReason::None;
        uint32_t madr = 0u;
        uint32_t qwc = 0u;
        uint32_t tadr = 0u;
        uint32_t asr0 = 0u;
        uint32_t asr1 = 0u;
        uint32_t tagsProcessed = 0u;
        uint8_t asp = 0u;
        uint8_t tagId = 0u;
        bool active = false;
        bool eventManaged = false;
        bool chainMode = false;
        bool tie = false;
        bool tte = false;
        bool tagIrq = false;
        bool endAfterPayload = false;
        bool faultReported = false;
        std::unordered_set<
            DmacSourceChainKey,
            DmacSourceChainKeyHash>
            visitedStates;
    };
    Vif1DmaState m_vif1Dma{};

    struct GifDmaState
    {
        DmacTransferToken transfer{};
        GifDmaPhase phase = GifDmaPhase::Idle;
        GifDmaStallReason stall =
            GifDmaStallReason::None;
        uint32_t chcr = 0u;
        uint32_t madr = 0u;
        uint32_t qwc = 0u;
        uint32_t tadr = 0u;
        uint32_t asr0 = 0u;
        uint32_t asr1 = 0u;
        uint32_t tagsProcessed = 0u;
        uint8_t asp = 0u;
        uint8_t tagId = 0u;
        bool active = false;
        bool eventManaged = false;
        bool chainMode = false;
        bool tie = false;
        bool tagIrq = false;
        bool endAfterPayload = false;
        bool faultReported = false;
        bool copyCounted = false;
        std::vector<uint8_t> fifoData;
        // A raw callback has no path-stream reassembler. Retain the complete
        // DMA stream so compatibility users observe the historical one-call
        // packet while the runtime arbiter still receives timed slices.
        std::vector<uint8_t> directCallbackData;
        std::unordered_set<
            DmacSourceChainKey,
            DmacSourceChainKeyHash>
            visitedStates;
    };
    GifDmaState m_gifDma{};

    struct FromIpuDmaState
    {
        DmacTransferToken transfer{};
        FromIpuDmaPhase phase =
            FromIpuDmaPhase::Idle;
        FromIpuDmaStallReason stall =
            FromIpuDmaStallReason::None;
        uint32_t chcr = 0u;
        uint32_t madr = 0u;
        uint32_t qwc = 0u;
        bool active = false;
        bool eventManaged = false;
        bool normalMode = false;
        bool zeroQwcStart = false;
        bool faultReported = false;
    };
    FromIpuDmaState m_fromIpuDma{};

    struct ToIpuDmaState
    {
        DmacTransferToken transfer{};
        ToIpuDmaPhase phase =
            ToIpuDmaPhase::Idle;
        ToIpuDmaStallReason stall =
            ToIpuDmaStallReason::None;
        uint32_t chcr = 0u;
        uint32_t madr = 0u;
        uint32_t qwc = 0u;
        uint32_t tadr = 0u;
        uint32_t tagsProcessed = 0u;
        uint8_t tagId = 0u;
        bool active = false;
        bool eventManaged = false;
        bool normalMode = false;
        bool chainMode = false;
        bool tie = false;
        bool tagIrq = false;
        bool endAfterPayload = false;
        bool faultReported = false;
    };
    ToIpuDmaState m_toIpuDma{};

    static constexpr uint32_t
        kIpuOutputFifoCapacityQwc = 8u;
    std::array<
        std::array<uint8_t, 16u>,
        kIpuOutputFifoCapacityQwc>
        m_ipuOutputFifo{};
    uint32_t m_ipuOutputFifoReadIndex = 0u;
    uint32_t m_ipuOutputFifoWriteIndex = 0u;
    uint32_t m_ipuOutputFifoQwc = 0u;

    static constexpr uint32_t
        kIpuInputFifoCapacityQwc = 8u;
    std::array<
        std::array<uint8_t, 16u>,
        kIpuInputFifoCapacityQwc>
        m_ipuInputFifo{};
    uint32_t m_ipuInputFifoReadIndex = 0u;
    uint32_t m_ipuInputFifoWriteIndex = 0u;
    uint32_t m_ipuInputFifoQwc = 0u;

    struct ScratchpadDmaState
    {
        DmacTransferToken transfer{};
        ScratchpadDmaPhase phase =
            ScratchpadDmaPhase::Idle;
        uint32_t chcr = 0u;
        uint32_t madr = 0u;
        uint32_t qwc = 0u;
        uint32_t tadr = 0u;
        uint32_t asr0 = 0u;
        uint32_t asr1 = 0u;
        uint32_t sadr = 0u;
        uint32_t tagsProcessed = 0u;
        uint8_t asp = 0u;
        uint8_t tagId = 0u;
        bool active = false;
        bool eventManaged = false;
        bool chainMode = false;
        bool tie = false;
        bool tte = false;
        bool tagIrq = false;
        bool endAfterPayload = false;
        bool faultReported = false;
    };
    std::array<ScratchpadDmaState, 2u>
        m_scratchpadDma{};

    struct DmacChannelLifecycle
    {
        DmacChannelPhase phase = DmacChannelPhase::Idle;
        uint64_t generation = 0u;
    };
    mutable std::mutex m_dmacMutex;
    std::array<DmacChannelLifecycle, PS2_DMAC_CHANNEL_COUNT>
        m_dmacChannels{};
    std::vector<DmacCompletionRequest> m_readyDmacCompletions;
    std::vector<DmacPendingInterrupt> m_pendingDmacInterrupts;
    uint64_t m_dmacReadySequence = 0u;
    uint64_t m_dmacPublicationSequence = 0u;
    DmacCompletionReadyCallback m_dmacCompletionReadyCallback;
    DmacInterruptStateCallback m_dmacInterruptStateCallback;

    struct CodeRegion
    {
        uint32_t start;
        uint32_t end;
        std::vector<bool> modified; // Bitmap of modified 4-byte blocks
    };
    static constexpr uint32_t kCodeRegionPageShift = 12u;
    static constexpr size_t kCodeRegionPageCount =
        PS2_RAM_SIZE >> kCodeRegionPageShift;
    static constexpr size_t kCodeRegionPageWordCount =
        (kCodeRegionPageCount + 63u) / 64u;
    std::vector<CodeRegion> m_codeRegions;
    bool m_hasCodeRegions = false;
    std::array<uint64_t, kCodeRegionPageWordCount>
        m_codeRegionPages{};
    std::mutex m_codeInvalidationMutex;
    std::vector<CodeInvalidationEvent> m_pendingCodeInvalidations;

    bool isAddressInRegion(uint32_t address, const CodeRegion &region);
    void markModified(uint32_t address, uint32_t size, uint32_t writerPc);
    void markVU0CodeModified() { m_vu0CodeGeneration.fetch_add(1, std::memory_order_relaxed); }
    void markVU1CodeModified();
    void markVUCodeModified(const uint8_t *vuMem)
    {
        if (vuMem == m_vu0Code)
            markVU0CodeModified();
        else if (vuMem == m_vu1Code)
            markVU1CodeModified();
    }
    [[nodiscard]] bool submitVu1OwnerCommand(
        Vu1CommandPayload payload);
    [[nodiscard]] bool submitVu1OwnerCommand(
        Vu1DecodedUnpackCommand command);
    [[nodiscard]] bool submitVu1OwnerCommand(
        Vu1VifStateUpdateCommand command);
    [[nodiscard]] bool writeVu1OwnerMemory(
        uint8_t *vuMemory, uint32_t offset,
        const void *bytes, size_t size,
        bool wrap = false);
    [[nodiscard]] Vu1VifState captureVu1VifState() const;
    void publishVu1VifState(
        const Vu1VifState &state) noexcept;
    void submitVu1VifStateUpdate();
    void observeVu1Memory() const;
    bool isScratchpad(uint32_t address) const;
    uint8_t *mapVuMemory(uint32_t physAddr, uint32_t size, uint32_t &offset, uint32_t &limit);
    const uint8_t *mapVuMemory(uint32_t physAddr, uint32_t size, uint32_t &offset, uint32_t &limit) const;
    [[nodiscard]] uint64_t currentEeCounterCycle() const noexcept;
    void publishEeCounterAdvance(
        const ps2x::timing::EeCounterAdvanceResult &result,
        bool interruptStateChanged = false);
    bool startScratchpadDma(
        DmacTransferToken transfer, uint32_t chcr);
    bool scheduleScratchpadDma(
        DmacChannel channel,
        uint32_t delayEeCycles);
    void clearScratchpadDmaState(
        DmacChannel channel,
        bool notifyRuntime);
    void drainScratchpadDmaCompatibility(
        DmacChannel channel);
    [[nodiscard]] static std::optional<size_t>
    scratchpadDmaIndex(DmacChannel channel) noexcept;
    void publishScratchpadDmaRegisters(
        const ScratchpadDmaState &state);
    bool startGifDma(
        DmacTransferToken transfer, uint32_t chcr);
    bool scheduleGifDma(uint32_t delayEeCycles);
    void clearGifDmaState(bool notifyRuntime);
    void drainGifDmaCompatibility();
    void publishGifDmaRegisters();
    void updateGifStat();
    [[nodiscard]] bool isGifPaused() const;
    bool copyGifDmaPayload(
        uint32_t sourceAddress,
        uint32_t qwc,
        bool toFifo);
    bool startFromIpuDma(
        DmacTransferToken transfer, uint32_t chcr);
    bool scheduleFromIpuDma(uint32_t delayEeCycles);
    void clearFromIpuDmaState(bool notifyRuntime);
    void drainFromIpuDmaCompatibility();
    void publishFromIpuDmaRegisters();
    uint32_t readIpuOutputFifo(
        uint8_t *data, uint32_t qwc);
    void clearIpuOutputFifo();
    bool copyFromIpuDmaPayload(
        uint32_t destinationAddress, uint32_t qwc);
    bool startToIpuDma(
        DmacTransferToken transfer, uint32_t chcr);
    bool scheduleToIpuDma(uint32_t delayEeCycles);
    void clearToIpuDmaState(bool notifyRuntime);
    void drainToIpuDmaCompatibility();
    void publishToIpuDmaRegisters();
    void publishIpuFifoCounts();
    void clearIpuInputFifo();
    uint32_t writeIpuInputFifo(
        const uint8_t *data, uint32_t qwc);
    bool copyToIpuDmaPayload(
        uint32_t sourceAddress, uint32_t qwc);
    void discardPendingDmacWork(DmacChannel channel);
    bool startVif0Dma(
        DmacTransferToken transfer, uint32_t chcr);
    void clearVif0DmaState(bool notifyRuntime);
    bool scheduleVif0Dma(uint32_t delayEeCycles);
    void drainVif0DmaCompatibility();
    void updateVif0Fqc(uint32_t qwc);
    [[nodiscard]] Vif0DmaStallReason
    currentVif0DmaStall() const;
    enum class Vif0ParserDisposition : uint8_t
    {
        Drained = 0u,
        NeedMoreData,
        WaitingForVu,
        VifInterrupt,
    };
    struct Vif0ParserResult
    {
        Vif0ParserDisposition disposition =
            Vif0ParserDisposition::Drained;
        uint32_t consumedBytes = 0u;
    };
    Vif0ParserResult processVif0Stream();
    void appendVif0Stream(
        const uint8_t *data, uint32_t sizeBytes);
    void publishVif0DmaRegisters();
    bool startVif1Dma(
        DmacTransferToken transfer, uint32_t chcr);
    void clearVif1DmaState(bool notifyRuntime);
    bool scheduleVif1Dma(uint32_t delayEeCycles);
    void drainVif1DmaCompatibility();
    void updateVif1Fqc(uint32_t qwc);
    [[nodiscard]] bool isDmacEnabled() const;
    [[nodiscard]] bool isVif1GifPathAvailable(
        bool directHl) const;
    enum class Vif1ParserDisposition : uint8_t
    {
        Drained = 0u,
        NeedMoreData,
        WaitingForVu,
        VifInterrupt,
        GifPath,
    };
    Vif1ParserDisposition processVif1Stream();
    void appendVif1Stream(
        const uint8_t *data, uint32_t sizeBytes,
        std::optional<uint8_t> firstByteOffset =
            std::nullopt);
    void consumeVif1StreamSegments(size_t sizeBytes);
    [[nodiscard]] uint8_t vif1StreamByteOffset(
        size_t position) const;
    void clearVif1Stream();
    [[nodiscard]] Vif1DmaStallReason
    currentVif1DmaStall() const;
    static uint64_t nextDmacSequence(uint64_t value) noexcept;
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    void beginVif1DmaTrace(uint32_t chcr,
                           uint32_t madr,
                           uint32_t qwc,
                           uint32_t tadr,
                           uint32_t asr0,
                           uint32_t asr1,
                           uint32_t sadr);
    void traceVif1DmaTag(const uint8_t *data);
    void traceVif1DmaPayload(
        const uint8_t *data, uint32_t sizeBytes);
    void traceVif1Input(
        const uint8_t *data, uint32_t sizeBytes);
    void traceVif1Command(
        uint32_t command,
        const uint8_t *followingData,
        uint32_t followingBytes);
    void tracePath1Packet(
        const uint8_t *data, uint32_t sizeBytes);
    [[nodiscard]] bool isVif1DmaTraceActive() const;
    void traceVu1Invocation(
        uint32_t startPc, uint32_t top, uint32_t itop,
        bool resume, const VuExecutionState &state);
    void traceVu1Instruction(
        uint32_t pc, uint32_t lower, uint32_t upper,
        const VuExecutionState &state);
    void traceVu1Xgkick(uint32_t sourceQword);
    void traceVu1InvocationEnd(
        uint32_t finalPc, bool ended, bool hitCycleLimit,
        const int32_t *viRegisters,
        size_t viRegisterCount);
    void finishVif1DmaTrace();
#else
    void beginVif1DmaTrace(
        uint32_t, uint32_t, uint32_t, uint32_t,
        uint32_t, uint32_t, uint32_t)
    {
    }
    void traceVif1DmaTag(const uint8_t *)
    {
    }
    void traceVif1DmaPayload(const uint8_t *, uint32_t)
    {
    }
    void traceVif1Input(const uint8_t *, uint32_t)
    {
    }
    void traceVif1Command(
        uint32_t, const uint8_t *, uint32_t)
    {
    }
    void tracePath1Packet(const uint8_t *, uint32_t)
    {
    }
    [[nodiscard]] bool
    isVif1DmaTraceActive() const
    {
        return false;
    }
    void traceVu1Invocation(
        uint32_t, uint32_t, uint32_t, bool,
        const VuExecutionState &)
    {
    }
    void traceVu1Instruction(
        uint32_t, uint32_t, uint32_t,
        const VuExecutionState &)
    {
    }
    void traceVu1Xgkick(uint32_t)
    {
    }
    void traceVu1InvocationEnd(
        uint32_t, bool, bool, const int32_t *, size_t)
    {
    }
    void finishVif1DmaTrace()
    {
    }
#endif
    Ps2GsDmaTraceState *m_gsDmaTrace = nullptr;
    std::atomic<uint64_t> m_vif1DmaTraceSequence{0u};
#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
    void recordVu1WorkloadProfileCodeUpload(
        uint32_t destination,
        const uint8_t *payload,
        uint32_t sizeBytes,
        bool identical,
        uint64_t generationBefore,
        uint64_t generationAfter);
#else
    void recordVu1WorkloadProfileCodeUpload(
        uint32_t, const uint8_t *, uint32_t, bool,
        uint64_t, uint64_t)
    {
    }
#endif
    Ps2Vu1WorkloadProfileState *m_vu1WorkloadProfile = nullptr;
    ps2x::timing::EeCounterBank m_eeCounters{};
    EeCounterCycleCallback m_eeCounterCycleCallback;
    EeCounterScheduleCallback m_eeCounterScheduleCallback;
    EeCounterInterruptStateCallback
        m_eeCounterInterruptStateCallback;
};

#endif // PS2_MEMORY_H
