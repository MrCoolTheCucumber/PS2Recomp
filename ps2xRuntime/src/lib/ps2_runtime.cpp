#include "ps2_runtime.h"
#include "ps2_log.h"
#include "ps2_stubs.h"
#include "ps2_syscalls.h"
#include "game_overrides.h"
#include "ps2_runtime_macros.h"
#include "runtime/ps2_gs_gpu.h"
#include "ThreadNaming.h"
#include "Kernel/Stubs/Audio.h"
#include "Kernel/Stubs/GS.h"
#include "Kernel/Stubs/MPEG.h"
#include "ps2_host_backend.h"
#include "ps2_iop_host.h"
#include "ps2x/iop/iop_subsystem.h"
#if defined(PS2X_ENABLE_DEBUG_SERVER) && PS2X_ENABLE_DEBUG_SERVER
#include "runtime/ps2_debug_server.h"
#endif

#include <iostream>
#include <fstream>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <chrono>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <sstream>

namespace ps2_stubs
{
    void resetSifState();
}

#define ELF_MAGIC 0x464C457F // "\x7FELF" in little endian
#define ET_EXEC 2            // Executable file
#define EM_MIPS 8            // MIPS architecture
#define PT_LOAD 1            // Loadable segment

static constexpr int FB_WIDTH = 640;
static constexpr int FB_HEIGHT = 512;
static constexpr int DEFAULT_DISPLAY_HEIGHT = 448;
static constexpr uint32_t DEFAULT_FB_SIZE = FB_WIDTH * FB_HEIGHT * 4;
static constexpr uint32_t DEFAULT_FB_ADDR = (PS2_RAM_SIZE - DEFAULT_FB_SIZE - 0x10000u);
static constexpr const char *GS_HISTORY_DUMP_ENV = "PS2X_GS_HISTORY_DUMP";
#if defined(PLATFORM_VITA)
static constexpr int HOST_WINDOW_WIDTH = 960;
static constexpr int HOST_WINDOW_HEIGHT = 544;
#else
static constexpr int HOST_WINDOW_WIDTH = FB_WIDTH;
static constexpr int HOST_WINDOW_HEIGHT = DEFAULT_DISPLAY_HEIGHT;
#endif
struct ElfHeader
{
    uint32_t magic;
    uint8_t elf_class;
    uint8_t endianness;
    uint8_t version;
    uint8_t os_abi;
    uint8_t abi_version;
    uint8_t padding[7];
    uint16_t type;
    uint16_t machine;
    uint32_t version2;
    uint32_t entry;
    uint32_t phoff;
    uint32_t shoff;
    uint32_t flags;
    uint16_t ehsize;
    uint16_t phentsize;
    uint16_t phnum;
    uint16_t shentsize;
    uint16_t shnum;
    uint16_t shstrndx;
};

struct ProgramHeader
{
    uint32_t type;
    uint32_t offset;
    uint32_t vaddr;
    uint32_t paddr;
    uint32_t filesz;
    uint32_t memsz;
    uint32_t flags;
    uint32_t align;
};

struct SectionHeader
{
    uint32_t name;
    uint32_t type;
    uint32_t flags;
    uint32_t addr;
    uint32_t offset;
    uint32_t size;
    uint32_t link;
    uint32_t info;
    uint32_t addralign;
    uint32_t entsize;
};

static_assert(sizeof(ElfHeader) == 52u);
static_assert(sizeof(ProgramHeader) == 32u);
static_assert(sizeof(SectionHeader) == 40u);

namespace
{
    constexpr uint32_t kGuestHeapDefaultBase = 0x00100000u;
    constexpr uint32_t kGuestHeapDefaultAlignment = 16u;
    constexpr uint32_t kGuestHeapSafetyPad = 0x1000u;
    constexpr uint32_t kGuestHeapHardLimit = 0x01F00000u;
    constexpr uint32_t kElfSectionAlloc = 0x2u;
    constexpr uint32_t kElfSectionExecInstr = 0x4u;
    constexpr uint32_t kVu0SyncThresholdCycles = 4u;
    constexpr uint32_t kVu0MinimumRunCycles = 16u;
    // PCSX2 starts non-interlocked VU work in a minimum 16-cycle batch and
    // revisits an active VU1 from VIF_VU1_FINISH after 128 cycles. Event mode
    // places that startup batch at a future EE boundary so submission itself
    // performs no VU work.
    constexpr uint32_t kVu1StartupEventCycles = 16u;
    constexpr uint32_t kVu1FollowupEventCycles = 128u;
    constexpr uint32_t kVu1MaximumAdvanceCycles = 65536u;
    constexpr uint32_t kVu1BusyMask = 1u << 8u;
    // PCSX2 uses a 48-cycle rapid-event deadline when the EE must revisit
    // asynchronous work in short order. VU0 is advanced only when an emitted
    // EE block crosses this deadline, rather than at every branch.
    constexpr uint32_t kEeRapidEventCycles = 48u;
    constexpr uint32_t kVu0InitialEventDelayCycles =
        kVu0MinimumRunCycles + 1u;

    constexpr uint32_t COP0_CAUSE_EXCCODE_MASK = 0x0000007Cu;
    constexpr uint32_t COP0_CAUSE_BD = 0x80000000u;
    constexpr uint32_t COP0_STATUS_EXL = 0x00000002u;
    constexpr uint32_t COP0_STATUS_BEV = 0x00400000u;
    constexpr uint32_t EXCEPTION_VECTOR_NORMAL_BASE = 0x80000000u;
    constexpr uint32_t EXCEPTION_VECTOR_BOOT_BASE = 0xBFC00200u;
    constexpr uint32_t EXCEPTION_VECTOR_COMMON_OFFSET = 0x180u;
    constexpr uint32_t EXCEPTION_VECTOR_INTERRUPT_OFFSET = 0x200u;

    struct DispatchHistory
    {
        std::array<uint32_t, 64> pcs{};
        uint32_t next = 0u;
        bool wrapped = false;
    };

    thread_local DispatchHistory g_dispatchHistory;
    thread_local std::unordered_map<PS2Runtime *, uint32_t> g_guestExecutionDepths;
    thread_local uint32_t g_deferredGuestYieldDepth = 0u;
    thread_local bool g_deferredGuestYieldPending = false;

    struct Vif0MmioTrace
    {
        std::FILE *output = nullptr;
        uint64_t sequence = 0u;

        ~Vif0MmioTrace()
        {
            if (output)
                std::fclose(output);
        }
    };

    Vif0MmioTrace &getVif0MmioTrace()
    {
        static Vif0MmioTrace trace = []
        {
            Vif0MmioTrace result;
            if (const char *path = std::getenv("PS2X_VIF0_MMIO_TRACE");
                path && *path)
            {
                result.output = std::fopen(path, "wb");
                if (!result.output)
                {
                    std::fprintf(stderr,
                                 "VIF0 MMIO trace: cannot open '%s': %s\n",
                                 path, std::strerror(errno));
                }
            }
            return result;
        }();
        return trace;
    }

    void traceVif0MmioWrite(PS2Memory &memory,
                            const R5900Context *ctx,
                            uint32_t address,
                            uint32_t size,
                            uint64_t valueLo,
                            uint64_t valueHi)
    {
        Vif0MmioTrace &trace = getVif0MmioTrace();
        if (!trace.output)
            return;

        constexpr uint32_t VIF0_CHANNEL = 0x10008000u;
        constexpr uint32_t VIF0_CHANNEL_END = VIF0_CHANNEL + 0x60u;
        const uint32_t physicalAddress = address & 0x1FFFFFFFu;
        const uint64_t writeEnd = static_cast<uint64_t>(physicalAddress) + size;
        if (physicalAddress >= VIF0_CHANNEL_END || writeEnd <= VIF0_CHANNEL)
            return;

        std::fprintf(
            trace.output,
            "{\"schema_version\":1,\"event\":\"vif0-mmio-write\","
            "\"sequence\":%" PRIu64 ",\"pc\":\"0x%08" PRIx32 "\","
            "\"ra\":\"0x%08" PRIx32 "\",\"address\":\"0x%08" PRIx32 "\","
            "\"size\":%" PRIu32 ",\"value_lo\":\"0x%016" PRIx64 "\","
            "\"value_hi\":\"0x%016" PRIx64 "\"",
            trace.sequence++,
            ctx ? ctx->pc : 0u,
            ctx ? getRegU32(ctx, 31) : 0u,
            physicalAddress, size, valueLo, valueHi);

        const uint32_t chcrOffset = VIF0_CHANNEL - physicalAddress;
        if (physicalAddress <= VIF0_CHANNEL &&
            static_cast<uint64_t>(chcrOffset) + sizeof(uint32_t) <= size)
        {
            uint32_t chcr = 0u;
            if (chcrOffset < 8u)
                chcr = static_cast<uint32_t>(valueLo >> (chcrOffset * 8u));
            else
                chcr = static_cast<uint32_t>(valueHi >> ((chcrOffset - 8u) * 8u));

            if ((chcr & 0x100u) != 0u)
            {
                const uint32_t madr = memory.readIORegister(VIF0_CHANNEL + 0x10u);
                const uint32_t qwc = memory.readIORegister(VIF0_CHANNEL + 0x20u);
                const uint32_t tadr = memory.readIORegister(VIF0_CHANNEL + 0x30u);
                const uint32_t asr0 = memory.readIORegister(VIF0_CHANNEL + 0x40u);
                const uint32_t asr1 = memory.readIORegister(VIF0_CHANNEL + 0x50u);
                uint32_t tagWords[4]{};
                bool tagValid = false;
                try
                {
                    for (uint32_t index = 0u; index < 4u; ++index)
                        tagWords[index] = memory.read32(tadr + index * 4u);
                    tagValid = true;
                }
                catch (...)
                {
                }

                std::fprintf(
                    trace.output,
                    ",\"start\":{\"chcr\":\"0x%08" PRIx32 "\","
                    "\"madr\":\"0x%08" PRIx32 "\",\"qwc\":\"0x%08" PRIx32 "\","
                    "\"tadr\":\"0x%08" PRIx32 "\",\"asr0\":\"0x%08" PRIx32 "\","
                    "\"asr1\":\"0x%08" PRIx32 "\",\"tag_valid\":%s,"
                    "\"tag\":[\"0x%08" PRIx32 "\",\"0x%08" PRIx32 "\","
                    "\"0x%08" PRIx32 "\",\"0x%08" PRIx32 "\"]}",
                    chcr, madr, qwc, tadr, asr0, asr1,
                    tagValid ? "true" : "false",
                    tagWords[0], tagWords[1], tagWords[2], tagWords[3]);
            }
        }

        std::fputs("}\n", trace.output);
        std::fflush(trace.output);
    }

    struct ElfAddressRange
    {
        uint32_t start = 0u;
        uint32_t end = 0u;
    };

    bool containsRange(const ElfAddressRange &outer, const ElfAddressRange &inner)
    {
        return inner.start >= outer.start && inner.end <= outer.end;
    }

    void mergeAddressRanges(std::vector<ElfAddressRange> &ranges)
    {
        std::sort(ranges.begin(), ranges.end(), [](const ElfAddressRange &lhs, const ElfAddressRange &rhs)
        {
            return lhs.start < rhs.start || (lhs.start == rhs.start && lhs.end < rhs.end);
        });

        std::vector<ElfAddressRange> merged;
        merged.reserve(ranges.size());
        for (const ElfAddressRange &range : ranges)
        {
            if (range.end <= range.start)
            {
                continue;
            }

            if (!merged.empty() && range.start <= merged.back().end)
            {
                merged.back().end = std::max(merged.back().end, range.end);
                continue;
            }

            merged.push_back(range);
        }
        ranges = std::move(merged);
    }

    bool readExecutableSectionRanges(std::ifstream &file,
                                     uint64_t fileSize,
                                     const ElfHeader &header,
                                     const std::vector<ElfAddressRange> &executableLoadRanges,
                                     std::vector<ElfAddressRange> &sectionRanges)
    {
        sectionRanges.clear();
        if (header.shoff == 0u || header.shnum == 0u ||
            header.shentsize < sizeof(SectionHeader))
        {
            return false;
        }

        const uint64_t tableEnd =
            static_cast<uint64_t>(header.shoff) +
            static_cast<uint64_t>(header.shnum) * static_cast<uint64_t>(header.shentsize);
        if (tableEnd > fileSize)
        {
            return false;
        }

        for (uint16_t i = 0; i < header.shnum; ++i)
        {
            const uint64_t sectionOffset =
                static_cast<uint64_t>(header.shoff) +
                static_cast<uint64_t>(i) * static_cast<uint64_t>(header.shentsize);

            SectionHeader section{};
            file.clear();
            file.seekg(static_cast<std::streamoff>(sectionOffset), std::ios::beg);
            if (!file.read(reinterpret_cast<char *>(&section), sizeof(section)))
            {
                sectionRanges.clear();
                return false;
            }

            constexpr uint32_t requiredFlags = kElfSectionAlloc | kElfSectionExecInstr;
            if ((section.flags & requiredFlags) != requiredFlags || section.size == 0u)
            {
                continue;
            }

            const uint64_t sectionEnd =
                static_cast<uint64_t>(section.addr) + static_cast<uint64_t>(section.size);
            if (sectionEnd > std::numeric_limits<uint32_t>::max())
            {
                continue;
            }

            const ElfAddressRange candidate{
                section.addr,
                static_cast<uint32_t>(sectionEnd),
            };
            const bool isFileBackedExecutableRange =
                std::any_of(executableLoadRanges.begin(), executableLoadRanges.end(),
                            [&candidate](const ElfAddressRange &loadRange)
                            {
                                return containsRange(loadRange, candidate);
                            });
            if (isFileBackedExecutableRange)
            {
                sectionRanges.push_back(candidate);
            }
        }

        if (sectionRanges.empty())
        {
            return false;
        }

        mergeAddressRanges(sectionRanges);
        return true;
    }

    bool computeFileCrc32(const std::string &path, uint32_t &crcOut)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }

        static const std::array<uint32_t, 256> table = []
        {
            std::array<uint32_t, 256> values{};
            for (uint32_t i = 0; i < values.size(); ++i)
            {
                uint32_t value = i;
                for (uint32_t bit = 0; bit < 8; ++bit)
                {
                    value = (value & 1u) ? (0xEDB88320u ^ (value >> 1u)) : (value >> 1u);
                }
                values[i] = value;
            }
            return values;
        }();

        uint32_t crc = 0xFFFFFFFFu;
        std::array<uint8_t, 16 * 1024> buffer{};
        while (file.good())
        {
            file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
            const std::streamsize count = file.gcount();
            for (std::streamsize i = 0; i < count; ++i)
            {
                crc = table[(crc ^ buffer[static_cast<size_t>(i)]) & 0xFFu] ^ (crc >> 8u);
            }
        }
        if (file.bad())
        {
            return false;
        }
        crcOut = ~crc;
        return true;
    }

    void pushDispatchPc(uint32_t pc)
    {
        DispatchHistory &h = g_dispatchHistory;
        h.pcs[h.next] = pc;
        h.next = (h.next + 1u) % static_cast<uint32_t>(h.pcs.size());
        if (h.next == 0u)
        {
            h.wrapped = true;
        }
    }

    std::string formatDispatchHistory()
    {
        const DispatchHistory &h = g_dispatchHistory;
        const uint32_t count = h.wrapped ? static_cast<uint32_t>(h.pcs.size()) : h.next;
        if (count == 0u)
        {
            return "(empty)";
        }

        std::ostringstream oss;
        bool first = true;
        for (uint32_t i = 0u; i < count; ++i)
        {
            const uint32_t idx = (h.next + h.pcs.size() - count + i) % static_cast<uint32_t>(h.pcs.size());
            if (!first)
            {
                oss << " -> ";
            }
            first = false;
            oss << PS2Runtime::formatGuestPc(h.pcs[idx]);
        }
        return oss.str();
    }

    std::string escapeJsonString(std::string_view text)
    {
        std::ostringstream output;
        for (const unsigned char ch : text)
        {
            switch (ch)
            {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (ch < 0x20u)
                {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<uint32_t>(ch)
                           << std::dec << std::setfill(' ');
                }
                else
                {
                    output << static_cast<char>(ch);
                }
                break;
            }
        }
        return output.str();
    }

    std::string rawGuestAddress(uint32_t address)
    {
        std::ostringstream output;
        output << "0x" << std::hex << std::setw(8) << std::setfill('0')
               << address;
        return output.str();
    }

    uint32_t selectExceptionVector(const R5900Context *ctx,
                                   uint32_t exceptionCode,
                                   bool alreadyInException)
    {
        uint32_t offset = EXCEPTION_VECTOR_COMMON_OFFSET;
        if (!alreadyInException)
        {
            if (exceptionCode == EXCEPTION_TLB_REFILL_LOAD ||
                exceptionCode == EXCEPTION_TLB_REFILL_STORE)
            {
                offset = 0u;
            }
            else if (exceptionCode == EXCEPTION_INTERRUPT)
            {
                offset = EXCEPTION_VECTOR_INTERRUPT_OFFSET;
            }
        }

        const uint32_t base = (ctx->cop0_status & COP0_STATUS_BEV)
                                  ? EXCEPTION_VECTOR_BOOT_BASE
                                  : EXCEPTION_VECTOR_NORMAL_BASE;
        return base + offset;
    }

    void seedVu0IdleSuccess(R5900Context *ctx)
    {
        if (!ctx)
        {
            return;
        }

        ctx->vu0_clip_flags = 0;
        ctx->vu0_clip_flags2 = 0;
        ctx->vu0_mac_flags = 0;
        ctx->vu0_status = 0;
        ctx->vu0_q = 1.0f;
        ctx->vu0_vpu_stat = 0;
        ctx->vu0_vpu_stat2 = 0;
    }

    void copyVu0ContextRegistersToState(
        const R5900Context *ctx, VU1State &state)
    {
        for (uint32_t i = 0; i < 32u; ++i)
        {
            _mm_storeu_ps(state.vf[i], ctx->vu0_vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            state.vi[i] = static_cast<int16_t>(ctx->vi[i]);
        }

        _mm_storeu_ps(state.acc, ctx->vu0_acc);
        state.q = ctx->vu0_q;
        state.p = ctx->vu0_p;
        state.i = ctx->vu0_i;
        state.mac = ctx->vu0_mac_flags;
        state.clip = ctx->vu0_clip_flags;
        state.status = ctx->vu0_status;
        state.top = ctx->vu0_top;
        state.itop = ctx->vu0_itop;

        state.vf[0][0] = 0.0f;
        state.vf[0][1] = 0.0f;
        state.vf[0][2] = 0.0f;
        state.vf[0][3] = 1.0f;
        state.vi[0] = 0;
    }

    void copyVu0ContextToState(const R5900Context *ctx, VU1State &state)
    {
        std::memset(&state, 0, sizeof(state));
        copyVu0ContextRegistersToState(ctx, state);
        state.pc = ctx->vu0_pc;
    }

    void copyVu0StateToContext(const VU1State &state, R5900Context *ctx)
    {
        for (uint32_t i = 0; i < 32u; ++i)
        {
            ctx->vu0_vf[i] = _mm_loadu_ps(state.vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            ctx->vi[i] = static_cast<uint16_t>(state.vi[i]);
        }

        ctx->vu0_acc = _mm_loadu_ps(state.acc);
        ctx->vu0_q = state.q;
        ctx->vu0_p = state.p;
        ctx->vu0_i = state.i;
        ctx->vu0_mac_flags = state.mac;
        ctx->vu0_clip_flags = state.clip;
        ctx->vu0_clip_flags2 = state.clip;
        ctx->vu0_status = static_cast<uint16_t>(state.status);
        ctx->vu0_itop = state.itop;
        ctx->vu0_pc = state.pc;
        ctx->vu0_tpc = state.pc;
        ctx->enforceVu0RegisterInvariants();
    }

    [[noreturn]] void raiseCop0Exception(R5900Context *ctx, uint32_t exceptionCode)
    {
        const bool alreadyInException = (ctx->cop0_status & COP0_STATUS_EXL) != 0u;
        if (!alreadyInException && ctx->in_delay_slot)
        {
            ctx->cop0_epc = ctx->branch_pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~COP0_CAUSE_EXCCODE_MASK) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK) |
                              COP0_CAUSE_BD;
        }
        else if (!alreadyInException)
        {
            ctx->cop0_epc = ctx->pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~(COP0_CAUSE_EXCCODE_MASK | COP0_CAUSE_BD)) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK);
        }
        else
        {
            // Nested level-1 exceptions update the cause but preserve the
            // original restart address and branch-delay state.
            ctx->cop0_cause = (ctx->cop0_cause & ~COP0_CAUSE_EXCCODE_MASK) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK);
        }

        ctx->cop0_status |= COP0_STATUS_EXL;
        ctx->pc = selectExceptionVector(ctx, exceptionCode, alreadyInException);
        ctx->in_delay_slot = false;
        throw PS2GuestException{};
    }

    std::filesystem::path normalizeAbsolutePath(const std::filesystem::path &path)
    {
        if (path.empty())
        {
            return {};
        }

#if defined(PLATFORM_VITA)
        const std::string generic = path.generic_string();
        const std::size_t colon = generic.find(':');
        if (colon != std::string::npos && colon != 0u)
        {
            const std::size_t slash = generic.find_first_of("/\\");
            if (slash == std::string::npos || colon < slash)
            {
                return path.lexically_normal();
            }
        }
#endif

        std::error_code ec;
        const std::filesystem::path absolute = std::filesystem::absolute(path, ec);
        if (ec)
        {
            return path.lexically_normal();
        }
        return absolute.lexically_normal();
    }

    PS2Runtime::IoPaths &runtimeIoPaths()
    {
        static PS2Runtime::IoPaths paths = []()
        {
            PS2Runtime::IoPaths defaults;
            std::error_code ec;
            const std::filesystem::path cwd = std::filesystem::current_path(ec);
            defaults.elfDirectory = ec ? std::filesystem::path(".") : cwd.lexically_normal();
            defaults.hostRoot = defaults.elfDirectory;
            defaults.cdRoot = defaults.elfDirectory;
            defaults.mcRoot = defaults.elfDirectory / "mc0";
            return defaults;
        }();

        return paths;
    }

    std::string readGuestPrintableString(const uint8_t *rdram, uint32_t addr, size_t maxLen)
    {
        std::string out;
        if (!rdram || maxLen == 0)
        {
            return out;
        }

        out.reserve(std::min<size_t>(maxLen, 64));
        for (size_t i = 0; i < maxLen; ++i)
        {
            const char ch = static_cast<char>(rdram[(addr + static_cast<uint32_t>(i)) & PS2_RAM_MASK]);
            if (ch == '\0')
            {
                break;
            }
            if (ch >= 0x20 && ch < 0x7F)
            {
                out.push_back(ch);
            }
            else
            {
                out.push_back('.');
            }
        }
        return out;
    }
}

PS2Runtime::GuestExecutionScope::GuestExecutionScope(
    PS2Runtime *runtime, R5900Context *context) noexcept
    : m_runtime(runtime),
      m_context(context)
{
    if (m_runtime)
    {
        m_previousContext =
            m_runtime->enterGuestExecution(m_context);
    }
}

PS2Runtime::GuestExecutionScope::~GuestExecutionScope()
{
    if (m_runtime)
    {
        m_runtime->leaveGuestExecution(
            m_context, m_previousContext);
    }
}

PS2Runtime::GuestExecutionReleaseScope::GuestExecutionReleaseScope(PS2Runtime *runtime) noexcept
    : m_runtime(runtime)
{
    if (m_runtime)
    {
        m_depth =
            m_runtime->releaseGuestExecution(m_context);
    }
}

PS2Runtime::GuestExecutionReleaseScope::~GuestExecutionReleaseScope()
{
    if (m_runtime && m_depth != 0u)
    {
        m_runtime->reacquireGuestExecution(
            m_depth, m_context);
    }
}

static void UploadFrame(Texture2D &tex, PS2Runtime *rt, uint32_t &outWidth, uint32_t &outHeight)
{
    static uint64_t s_lastPresentationTick = std::numeric_limits<uint64_t>::max();
    static bool s_hasLatchedInitialFrame = false;
    static uint32_t s_lastDisplayFbp = std::numeric_limits<uint32_t>::max();
    static uint32_t s_lastSourceFbp = std::numeric_limits<uint32_t>::max();
    static bool s_lastPreferred = false;
    static uint32_t s_lastWidth = 0u;
    static uint32_t s_lastHeight = 0u;
    static bool s_hasUploadedFrame = false;
    static std::vector<uint8_t> s_scratch;
    static std::vector<uint8_t> s_uploadBuffer(DEFAULT_FB_SIZE, 0u);

    const uint64_t currentTick = ps2_syscalls::GetCurrentVSyncTick();
    const bool needsLatch = !s_hasLatchedInitialFrame || currentTick != s_lastPresentationTick;
    if (needsLatch)
    {
        rt->gs().latchHostPresentationFrame();
        s_lastPresentationTick = currentTick;
        s_hasLatchedInitialFrame = true;
    }
    else if (s_hasUploadedFrame)
    {
        outWidth = (s_lastWidth != 0u) ? s_lastWidth : FB_WIDTH;
        outHeight = (s_lastHeight != 0u) ? s_lastHeight : DEFAULT_DISPLAY_HEIGHT;
        return;
    }

    s_scratch.clear();
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t displayFbp = 0u;
    uint32_t sourceFbp = 0u;
    bool usedPreferredDisplaySource = false;
    if (!rt->gs().copyLatchedHostPresentationFrame(s_scratch,
                                                   width,
                                                   height,
                                                   &displayFbp,
                                                   &sourceFbp,
                                                   &usedPreferredDisplaySource))
    {
        Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, MAGENTA);
        UpdateTexture(tex, blank.data);
        UnloadImage(blank);
        outWidth = FB_WIDTH;
        outHeight = DEFAULT_DISPLAY_HEIGHT;
        s_lastWidth = outWidth;
        s_lastHeight = outHeight;
        s_hasUploadedFrame = true;
        return;
    }

    PS2_IF_AGRESSIVE_LOGS({
        static uint32_t s_uploadDebugCount = 0u;
        if (s_uploadDebugCount < 128u ||
            displayFbp != s_lastDisplayFbp ||
            sourceFbp != s_lastSourceFbp ||
            usedPreferredDisplaySource != s_lastPreferred ||
            width != s_lastWidth ||
            height != s_lastHeight)
        {
            std::cout << "[frame:upload] idx=" << s_uploadDebugCount
                      << " tick=" << currentTick
                      << " displayFbp=" << displayFbp
                      << " sourceFbp=" << sourceFbp
                      << " size=" << width << "x" << height
                      << " preferred=" << static_cast<uint32_t>(usedPreferredDisplaySource ? 1u : 0u)
                      << std::endl;
        }
        ++s_uploadDebugCount;
    });
    s_lastDisplayFbp = displayFbp;
    s_lastSourceFbp = sourceFbp;
    s_lastPreferred = usedPreferredDisplaySource;
    s_lastWidth = width;
    s_lastHeight = height;

    std::fill(s_uploadBuffer.begin(), s_uploadBuffer.end(), 0u);
    if (!s_scratch.empty() && width != 0u && height != 0u)
    {
        const uint32_t copyWidth = std::min<uint32_t>(width, FB_WIDTH);
        const uint32_t copyHeight = std::min<uint32_t>(height, FB_HEIGHT);
        const size_t srcRowBytes = static_cast<size_t>(width) * 4u;
        const size_t dstRowBytes = static_cast<size_t>(FB_WIDTH) * 4u;
        const size_t copyRowBytes = static_cast<size_t>(copyWidth) * 4u;
        for (uint32_t y = 0; y < copyHeight; ++y)
        {
            const size_t srcOffset = static_cast<size_t>(y) * srcRowBytes;
            const size_t dstOffset = static_cast<size_t>(y) * dstRowBytes;
            if (srcOffset + copyRowBytes > s_scratch.size() ||
                dstOffset + copyRowBytes > s_uploadBuffer.size())
            {
                break;
            }
            std::memcpy(s_uploadBuffer.data() + dstOffset, s_scratch.data() + srcOffset, copyRowBytes);
        }
    }

    UpdateTexture(tex, s_uploadBuffer.data());
    outWidth = width;
    outHeight = height;
    s_hasUploadedFrame = true;
}

PS2Runtime::PS2Runtime()
{
    if (const char *const mode =
            std::getenv("PS2X_EE_SCHEDULER_MODE");
        mode)
    {
        if (std::strcmp(mode, "shadow") == 0)
        {
            m_eeSchedulingMode =
                ps2x::timing::EeSchedulingMode::Shadow;
        }
        else if (std::strcmp(mode, "event") == 0)
        {
            m_eeSchedulingMode =
                ps2x::timing::EeSchedulingMode::Event;
        }
        else if (std::strcmp(mode, "legacy") != 0)
        {
            std::cerr
                << "Ignoring unknown PS2X_EE_SCHEDULER_MODE='"
                << mode << "'" << std::endl;
        }
    }

    m_memory.setDmacCompletionReadyCallback(
        [this]()
        {
            onDmacCompletionReady();
        });
    m_memory.setDmacInterruptStateCallback(
        [this]()
        {
            onDmacInterruptStateChanged();
        });
    m_memory.setVif1DmaScheduleCallback(
        [this](uint32_t delayEeCycles)
        {
            return scheduleVif1DmaFromMemory(
                delayEeCycles);
        });
    m_memory.setVif1DmaCancelCallback(
        [this]()
        {
            cancelVif1DmaEvent();
        });
    m_memory.setScratchpadDmaScheduleCallback(
        [this](
            DmacChannel channel,
            uint32_t delayEeCycles)
        {
            return scheduleScratchpadDmaFromMemory(
                channel, delayEeCycles);
        });
    m_memory.setScratchpadDmaCancelCallback(
        [this](DmacChannel channel)
        {
            cancelScratchpadDmaEvent(channel);
        });

    m_iopHost = std::make_unique<PS2IopHostAdapter>(*this);
    m_iopSubsystem = std::make_unique<ps2x::iop::IopSubsystem>(*m_iopHost);
    m_vu0.setInstructionObserver(
        [this](uint64_t, uint32_t pc, uint32_t lower, uint32_t upper,
               const VU1State &state)
        {
            if (!m_debugVu0InstructionTraceEnabled.load(
                    std::memory_order_relaxed) ||
                !m_debugVu0InstructionTraceTriggered.load(
                    std::memory_order_acquire))
            {
                return;
            }

            DebugVu0InstructionEntry entry{};
            entry.invocation =
                m_vu0CurrentInvocation.load(std::memory_order_relaxed);
            entry.invocationInstruction =
                m_vu0CurrentInvocationInstruction.fetch_add(
                    1u, std::memory_order_relaxed);
            entry.pc = pc;
            entry.lower = lower;
            entry.upper = upper;
            for (size_t index = 0u; index < entry.vi.size(); ++index)
            {
                entry.vi[index] =
                    static_cast<uint16_t>(state.vi[index]);
            }
            std::memcpy(
                entry.vf.data(), state.vf, sizeof(state.vf));
            std::memcpy(
                entry.acc.data(), state.acc, sizeof(state.acc));
            std::memcpy(&entry.q, &state.q, sizeof(entry.q));
            std::memcpy(&entry.p, &state.p, sizeof(entry.p));
            std::memcpy(&entry.i, &state.i, sizeof(entry.i));
            entry.status = state.status;
            entry.mac = state.mac;
            entry.clip = state.clip;
            entry.top = state.top;
            entry.itop = state.itop;
            entry.branchPending = state.branchPending;
            entry.branchTarget = state.branchTarget;
            entry.branchDelay = state.branchDelay;
            entry.viBackupCycles = state.viBackupCycles;
            entry.viBackupRegister = state.viBackupRegister;
            entry.viBackupValue =
                static_cast<uint16_t>(state.viBackupValue);
            debugRecordVu0Instruction(std::move(entry));
        });
    m_vu0.setInstructionObserverEnabled(false);
#if defined(PS2X_IOP_ENABLE_PLUGINS) && PS2X_IOP_ENABLE_PLUGINS && \
    !defined(PLATFORM_VITA) && (defined(_WIN32) || defined(__linux__))
    if (const char *applicationDirectory = GetApplicationDirectory();
        applicationDirectory && applicationDirectory[0] != '\0')
    {
        m_iopSubsystem->setPluginSearchPaths({std::filesystem::path(applicationDirectory) / "iop_plugins"});
    }
#endif

    std::memset(&m_cpuContext, 0, sizeof(m_cpuContext));

    // R0 is always zero in MIPS
    m_cpuContext.r[0] = _mm_set1_epi32(0);
    // Recompiled ELFs enter after the console BIOS has enabled normal
    // dual-issue, cache, non-blocking-load, and branch-prediction modes.
    m_cpuContext.cop0_config = 0x00073443u;

    // Stack pointer (SP) and global pointer (GP) will be set by the loaded ELF

    m_loadedModules.clear();
    m_guestHeapBlocks.clear();
    m_guestHeapBase = kGuestHeapDefaultBase;
    m_guestHeapEnd = kGuestHeapDefaultBase;
    m_guestHeapLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    m_guestHeapSuggestedBase = kGuestHeapDefaultBase;
    m_guestHeapConfigured = false;
    m_asyncCallbackStackFloor = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    m_asyncCallbackStackTop = PS2_RAM_SIZE;
}

void PS2Runtime::setDebugUiCallbacks(DebugUiCallback initCallback,
                                     DebugUiCallback drawCallback,
                                     DebugUiCallback shutdownCallback,
                                     void *userData)
{
    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }

    m_debugUiInitCallback = initCallback;
    m_debugUiDrawCallback = drawCallback;
    m_debugUiShutdownCallback = shutdownCallback;
    m_debugUiUserData = userData;
}

PS2Runtime::~PS2Runtime()
{
    try
    {
        requestStop();
#if defined(PS2X_ENABLE_DEBUG_SERVER) && PS2X_ENABLE_DEBUG_SERVER
        if (m_debugServer)
        {
            m_debugServer->stop();
            m_debugServer.reset();
        }
#endif
        ps2_syscalls::detachAllGuestHostThreads();
        m_iopSubsystem.reset();
        m_iopHost.reset();
#if defined(PLATFORM_VITA)
        m_audioBackend.stopAll();
        m_audioBackend.setAudioReady(false);
#else
        m_audioBackend.stopAll();
        m_audioBackend.setAudioReady(false);
        if (IsAudioDeviceReady())
        {
            CloseAudioDevice();
        }
#endif
        if (m_debugUiInitialized && m_debugUiShutdownCallback)
        {
            m_debugUiShutdownCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = false;
        }

        if (IsWindowReady())
        {
            CloseWindow();
        }

        m_loadedModules.clear();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: unknown" << std::endl;
    }
}

void PS2Runtime::setIopPluginSearchPaths(std::vector<std::filesystem::path> paths)
{
    m_iopSubsystem->setPluginSearchPaths(std::move(paths));
}

ps2x::iop::RpcAbi PS2Runtime::selectIopRpcAbi(const ps2x::iop::RpcAbiRequest &request) const
{
    return m_iopSubsystem->selectRpcAbi(request);
}

ps2x::iop::RpcResult PS2Runtime::handleIopRpc(uint8_t *rdram, R5900Context *ctx, ps2x::iop::RpcRequest request)
{
    auto scope = m_iopHost->enterCall(ctx, rdram);
    request.callToken = scope.token();
    return m_iopSubsystem->handleRpc(request);
}

void PS2Runtime::notifyIopSifTransfer(uint8_t *rdram, const ps2x::iop::SifTransfer &transfer)
{
    auto scope = m_iopHost->enterCall(nullptr, rdram);
    m_iopSubsystem->onSifTransfer(transfer);
}

void PS2Runtime::resetIop()
{
    m_iopSubsystem->reset();
}

ps2x::timing::EeTick PS2Runtime::currentEeTick() const noexcept
{
    return m_eeTimeline.now();
}

ps2x::timing::EeTick PS2Runtime::publishEeElapsed(
    ps2x::timing::EeTickDelta elapsed) noexcept
{
    return m_eeTimeline.advance(elapsed);
}

ps2x::timing::EeTick PS2Runtime::commitEeContextProgress(
    R5900Context *context) noexcept
{
    if (!context)
    {
        return currentEeTick();
    }
    return publishEeElapsed(context->commitEeBlockCycles());
}

ps2x::timing::EeTick PS2Runtime::finishEeContextBlock(
    R5900Context *context) noexcept
{
    if (!context)
    {
        return currentEeTick();
    }
    return publishEeElapsed(context->finishEeBasicBlock());
}

void PS2Runtime::resetEeTimingUnlocked(
    R5900Context *context) noexcept
{
    cancelVif1DmaEvent();
    if (m_memory.vif1DmaSnapshot().active)
    {
        (void)m_memory.cancelDmacTransfer(
            DmacChannel::Vif1);
    }
    for (const DmacChannel channel :
         {DmacChannel::FromScratchpad,
          DmacChannel::ToScratchpad})
    {
        cancelScratchpadDmaEvent(channel);
        if (m_memory
                .scratchpadDmaSnapshot(channel)
                .active)
        {
            (void)m_memory.cancelDmacTransfer(
                channel);
        }
    }
    m_eeTimeline.reset();
    m_eeEventScheduler.reset();
    cancelVU1Execution(true);
    m_vu0CycleTick = {};
    m_vu0NextEventCycleTick = {};
    m_eeSchedulerShadowMismatch = false;
    m_eeSchedulerShadowMismatchTick = {};
    m_eeSchedulerShadowLegacyDeadline = {};
    m_eeSchedulerShadowDeadline = {};
    m_eeSchedulerShadowLegacyPending = false;
    m_eeSchedulerShadowPending = false;
    m_dmacCompatibilityEventSequence = 0u;
    m_guestExecutionPreemptionRequested.store(
        false, std::memory_order_release);
    if (m_eeSchedulingMode ==
            ps2x::timing::EeSchedulingMode::Event &&
        m_dmacCompletionReady.load(
            std::memory_order_acquire))
    {
        (void)m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::
                DmacCompletion,
            currentEeTick());
    }

    m_cpuContext.resetEeBlockTiming();
    if (m_boundEeContext && m_boundEeContext != &m_cpuContext)
    {
        m_boundEeContext->resetEeBlockTiming();
    }
    if (context &&
        context != &m_cpuContext &&
        context != m_boundEeContext)
    {
        context->resetEeBlockTiming();
    }
}

void PS2Runtime::resetEeTiming(R5900Context *context)
{
    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    resetEeTimingUnlocked(context);
}

void PS2Runtime::setEeSchedulingMode(
    ps2x::timing::EeSchedulingMode mode)
{
    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    if (m_eeSchedulingMode == mode)
    {
        return;
    }

    cancelVif1DmaEvent();
    if (m_memory.vif1DmaSnapshot().active)
    {
        (void)m_memory.cancelDmacTransfer(
            DmacChannel::Vif1);
    }
    for (const DmacChannel channel :
         {DmacChannel::FromScratchpad,
          DmacChannel::ToScratchpad})
    {
        cancelScratchpadDmaEvent(channel);
        if (m_memory
                .scratchpadDmaSnapshot(channel)
                .active)
        {
            (void)m_memory.cancelDmacTransfer(
                channel);
        }
    }
    m_eeSchedulingMode = mode;
    m_eeEventScheduler.reset();
    cancelVU1Execution(true);
    m_eeSchedulerShadowMismatch = false;
    m_eeSchedulerShadowMismatchTick = {};
    m_eeSchedulerShadowLegacyDeadline = {};
    m_eeSchedulerShadowDeadline = {};
    m_eeSchedulerShadowLegacyPending = false;
    m_eeSchedulerShadowPending = false;

    if (mode ==
            ps2x::timing::EeSchedulingMode::Shadow &&
        m_vu0.isActive() &&
        m_vu0NextEventCycleTick !=
            ps2x::timing::EeTick{})
    {
        (void)m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::
                Vu0PeriodicCompatibility,
            m_vu0NextEventCycleTick);
    }
    if (mode == ps2x::timing::EeSchedulingMode::Event &&
        m_dmacCompletionReady.load(
            std::memory_order_acquire))
    {
        (void)m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::
                DmacCompletion,
            currentEeTick());
    }
}

ps2x::timing::EeSchedulingMode
PS2Runtime::eeSchedulingMode() const noexcept
{
    return m_eeSchedulingMode;
}

void PS2Runtime::onDmacCompletionReady()
{
    m_dmacCompletionReady.store(
        true, std::memory_order_release);
    if (m_eeSchedulingMode ==
        ps2x::timing::EeSchedulingMode::Event)
    {
        (void)m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::
                DmacCompletion,
            currentEeTick());
    }
}

void PS2Runtime::requestDmacCompletion(
    DmacChannel channel)
{
    const DmacTransferToken transfer =
        m_memory.beginDmacTransfer(channel);
    (void)m_memory.requestDmacCompletion(transfer);
}

void PS2Runtime::collectPendingDmacInterrupts()
{
    std::vector<DmacPendingInterrupt> pending =
        m_memory.consumePendingDmacInterrupts();
    if (pending.empty())
    {
        return;
    }
    m_pendingDmacInterrupts.insert(
        m_pendingDmacInterrupts.end(),
        std::make_move_iterator(pending.begin()),
        std::make_move_iterator(pending.end()));
}

void PS2Runtime::onDmacInterruptStateChanged()
{
    collectPendingDmacInterrupts();
    for (const DmacPendingInterrupt &interrupt :
         m_pendingDmacInterrupts)
    {
        if (m_memory.dmacInterruptStatusLatched(
                interrupt.source) &&
            m_memory.dmacInterruptUnmasked(
                interrupt.source) &&
            ps2_syscalls::isDmacCauseEnabled(
                interrupt.cause))
        {
            requestGuestPreemption();
            return;
        }
    }
}

size_t PS2Runtime::publishReadyDmacCompletions(
    uint64_t eventSequence)
{
    const size_t published =
        m_memory.publishReadyDmacCompletions(
            eventSequence);
    m_dmacCompletionReady.store(
        m_memory.hasReadyDmacCompletions(),
        std::memory_order_release);
    collectPendingDmacInterrupts();

    if (published != 0u)
    {
        onDmacInterruptStateChanged();
    }
    return published;
}

void PS2Runtime::publishCompatibilityDmacCompletions()
{
    if (!m_dmacCompletionReady.load(
            std::memory_order_acquire))
    {
        return;
    }

    ++m_dmacCompatibilityEventSequence;
    if (m_dmacCompatibilityEventSequence == 0u)
    {
        m_dmacCompatibilityEventSequence = 1u;
    }
    (void)publishReadyDmacCompletions(
        m_dmacCompatibilityEventSequence);
}

ps2x::iop::DebugSnapshot PS2Runtime::iopDebugSnapshot() const
{
    return m_iopSubsystem->debugSnapshot();
}

bool PS2Runtime::syncCoreSubsystems()
{
    uint8_t *const rdram = m_memory.getRDRAM();
    uint8_t *const gsVram = m_memory.getGSVRAM();
    if (!rdram || !gsVram)
    {
        return false;
    }

    if (m_boundRdram == rdram && m_boundGSVram == gsVram)
    {
        return true;
    }

    m_gs.init(gsVram, static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &m_memory.gs());
    m_gifArbiter.reset();
    m_gifArbiter.setProcessPacketFn([this](const uint8_t *data, uint32_t size)
                                    { m_gs.processGIFPacket(data, size); });
    m_gifArbiter.setDrainCallbacks([this]
                                   { m_gs.beginRenderBatch(); },
                                   [this]
                                   { m_gs.endRenderBatch(); });
    m_memory.setGifArbiter(&m_gifArbiter);
    m_memory.setVif1GifPathAvailableCallback(
        [this](bool directHl)
        {
            if (m_eeSchedulingMode !=
                ps2x::timing::EeSchedulingMode::Event)
            {
                return true;
            }
            return m_gifArbiter.canAcceptPath2(
                directHl);
        });
    m_memory.setVu1MscalCallback(
        [this](uint32_t startPC, uint32_t top, uint32_t itop)
        {
            startVU1FromVif(startPC, top, itop);
        });
    m_memory.setVu1MscntCallback(
        [this](uint32_t top, uint32_t itop)
        {
            resumeVU1FromVif(top, itop);
        });
    m_memory.setVu1BusyCallback(
        [this]()
        {
            return m_eeSchedulingMode ==
                       ps2x::timing::EeSchedulingMode::Event &&
                   m_vu1.isActive();
        });
    m_memory.setVif1ResetCallback(
        [this]()
        {
            cancelVU1Execution(true);
        });
    resetIop();
    m_vu0.reset();
    m_vu1.reset();
    resetEeTimingUnlocked(&m_cpuContext);
    m_vu0InvocationSequence.store(0u, std::memory_order_relaxed);
    m_vu0CurrentInvocation.store(0u, std::memory_order_relaxed);
    m_vu0CurrentInvocationInstruction.store(
        0u, std::memory_order_relaxed);

    m_boundRdram = rdram;
    m_boundGSVram = gsVram;
    return true;
}

void PS2Runtime::setVU1BusyFlag(
    R5900Context *context, bool busy) noexcept
{
    const auto update =
        [busy](R5900Context *target)
        {
            if (!target)
                return;
            if (busy)
                target->vu0_vpu_stat |= kVu1BusyMask;
            else
                target->vu0_vpu_stat &= ~kVu1BusyMask;
        };

    update(&m_cpuContext);
    if (m_boundEeContext != &m_cpuContext)
        update(m_boundEeContext);
    if (context != &m_cpuContext &&
        context != m_boundEeContext)
    {
        update(context);
    }
}

void PS2Runtime::scheduleVU1Event(
    ps2x::timing::EeTick deadline) noexcept
{
    m_vu1ExecutionTiming.eventToken =
        m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::VifVu1Finish,
            deadline);
}

bool PS2Runtime::scheduleVif1DmaFromMemory(
    uint32_t delayEeCycles)
{
    if (m_eeSchedulingMode !=
        ps2x::timing::EeSchedulingMode::Event)
    {
        return false;
    }

    const ps2x::timing::EeTick now =
        commitEeContextProgress(m_boundEeContext);
    scheduleVif1DmaEvent(
        ps2x::timing::saturatingAdd(
            now,
            ps2x::timing::eeCyclesToTicks(
                delayEeCycles)));
    return m_vif1DmaEventToken.generation != 0u;
}

void PS2Runtime::scheduleVif1DmaEvent(
    ps2x::timing::EeTick deadline) noexcept
{
    m_vif1DmaEventToken =
        m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::DmacVif1,
            deadline);
}

void PS2Runtime::cancelVif1DmaEvent() noexcept
{
    if (m_vif1DmaEventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_vif1DmaEventToken);
    }
    m_vif1DmaEventToken = {
        ps2x::timing::EeEventSource::DmacVif1,
        0u};
}

void PS2Runtime::serviceVif1DmaAtEvent(
    const ps2x::timing::EeEventService &service)
{
    if (m_vif1DmaEventToken.generation == 0u ||
        service.generation !=
            m_vif1DmaEventToken.generation)
    {
        return;
    }

    m_vif1DmaEventToken = {
        ps2x::timing::EeEventSource::DmacVif1,
        0u};
    const Vif1DmaAdvanceResult advance =
        m_memory.advanceVif1Dma();
    if (!advance.active || advance.completed)
        return;

    if (advance.stall !=
            Vif1DmaStallReason::None &&
        advance.stall !=
            Vif1DmaStallReason::GifPath)
    {
        return;
    }

    scheduleVif1DmaEvent(
        ps2x::timing::saturatingAdd(
            service.serviceTick,
            ps2x::timing::eeCyclesToTicks(
                advance.delayEeCycles)));
}

ps2x::timing::EeEventSource
PS2Runtime::scratchpadDmaEventSource(
    DmacChannel channel) noexcept
{
    switch (channel)
    {
    case DmacChannel::FromScratchpad:
        return ps2x::timing::EeEventSource::
            DmacFromScratchpad;
    case DmacChannel::ToScratchpad:
        return ps2x::timing::EeEventSource::
            DmacToScratchpad;
    default:
        return ps2x::timing::EeEventSource::Count;
    }
}

bool PS2Runtime::scheduleScratchpadDmaFromMemory(
    DmacChannel channel,
    uint32_t delayEeCycles)
{
    if (m_eeSchedulingMode !=
            ps2x::timing::EeSchedulingMode::Event ||
        scratchpadDmaEventSource(channel) ==
            ps2x::timing::EeEventSource::Count)
    {
        return false;
    }

    const ps2x::timing::EeTick now =
        commitEeContextProgress(m_boundEeContext);
    scheduleScratchpadDmaEvent(
        channel,
        ps2x::timing::saturatingAdd(
            now,
            ps2x::timing::eeCyclesToTicks(
                delayEeCycles)));
    const ps2x::timing::EeEventToken &token =
        channel == DmacChannel::FromScratchpad
            ? m_fromScratchpadDmaEventToken
            : m_toScratchpadDmaEventToken;
    return token.generation != 0u;
}

void PS2Runtime::scheduleScratchpadDmaEvent(
    DmacChannel channel,
    ps2x::timing::EeTick deadline) noexcept
{
    const ps2x::timing::EeEventSource source =
        scratchpadDmaEventSource(channel);
    if (source == ps2x::timing::EeEventSource::Count)
        return;

    const ps2x::timing::EeEventToken token =
        m_eeEventScheduler.scheduleAbsolute(
            source, deadline);
    if (channel == DmacChannel::FromScratchpad)
        m_fromScratchpadDmaEventToken = token;
    else
        m_toScratchpadDmaEventToken = token;
}

void PS2Runtime::cancelScratchpadDmaEvent(
    DmacChannel channel) noexcept
{
    ps2x::timing::EeEventToken *token = nullptr;
    if (channel == DmacChannel::FromScratchpad)
        token = &m_fromScratchpadDmaEventToken;
    else if (channel == DmacChannel::ToScratchpad)
        token = &m_toScratchpadDmaEventToken;
    if (!token)
        return;

    if (token->generation != 0u)
        (void)m_eeEventScheduler.cancel(*token);
    *token = {
        scratchpadDmaEventSource(channel), 0u};
}

void PS2Runtime::serviceScratchpadDmaAtEvent(
    DmacChannel channel,
    const ps2x::timing::EeEventService &service)
{
    ps2x::timing::EeEventToken *token =
        channel == DmacChannel::FromScratchpad
            ? &m_fromScratchpadDmaEventToken
            : &m_toScratchpadDmaEventToken;
    if (service.source !=
            scratchpadDmaEventSource(channel) ||
        token->generation == 0u ||
        service.generation != token->generation)
    {
        return;
    }

    *token = {
        scratchpadDmaEventSource(channel), 0u};
    const ScratchpadDmaAdvanceResult advance =
        m_memory.advanceScratchpadDma(channel);
    if (!advance.active || advance.completed ||
        advance.phase == ScratchpadDmaPhase::Fault)
    {
        return;
    }

    scheduleScratchpadDmaEvent(
        channel,
        ps2x::timing::saturatingAdd(
            service.serviceTick,
            ps2x::timing::eeCyclesToTicks(
                advance.delayEeCycles)));
}

void PS2Runtime::cancelVU1Execution(bool resetInterpreter)
{
    if (m_vu1ExecutionTiming.eventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_vu1ExecutionTiming.eventToken);
    }
    m_vu1ExecutionTiming.eventToken = {
        ps2x::timing::EeEventSource::VifVu1Finish, 0u};
    ++m_vu1ExecutionTiming.generation;
    if (m_vu1ExecutionTiming.generation == 0u)
        ++m_vu1ExecutionTiming.generation;
    m_vu1ExecutionTiming.lastAdvancedTick = currentEeTick();
    m_vu1ExecutionTiming.totalAdvancedCycles = 0u;
    if (resetInterpreter)
        m_vu1.reset();
    m_memory.cancelVIF1VuWait();
    m_memory.finishVif1DmaTrace();
    setVU1BusyFlag(nullptr, false);
}

void PS2Runtime::startVU1FromVif(
    uint32_t startPC, uint32_t top, uint32_t itop)
{
    if (m_eeSchedulingMode !=
        ps2x::timing::EeSchedulingMode::Event)
    {
        m_vu1.execute(
            m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
            m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
            m_gs, &m_memory, startPC, top, itop,
            kVu1MaximumAdvanceCycles);
        return;
    }

    const ps2x::timing::EeTick startTick =
        commitEeContextProgress(m_boundEeContext);
    if (m_vu1ExecutionTiming.eventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_vu1ExecutionTiming.eventToken);
    }
    ++m_vu1ExecutionTiming.generation;
    if (m_vu1ExecutionTiming.generation == 0u)
        ++m_vu1ExecutionTiming.generation;
    m_vu1ExecutionTiming.lastAdvancedTick = startTick;
    m_vu1ExecutionTiming.totalAdvancedCycles = 0u;
    m_vu1.start(startPC, top, itop, &m_memory);
    setVU1BusyFlag(nullptr, true);
    scheduleVU1Event(
        ps2x::timing::saturatingAdd(
            startTick,
            ps2x::timing::vuCyclesToEeTicks(
                kVu1StartupEventCycles)));
}

void PS2Runtime::resumeVU1FromVif(
    uint32_t top, uint32_t itop)
{
    if (m_eeSchedulingMode !=
        ps2x::timing::EeSchedulingMode::Event)
    {
        m_vu1.resume(
            m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
            m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
            m_gs, &m_memory, top, itop,
            kVu1MaximumAdvanceCycles);
        return;
    }

    const ps2x::timing::EeTick startTick =
        commitEeContextProgress(m_boundEeContext);
    if (m_vu1ExecutionTiming.eventToken.generation != 0u)
    {
        (void)m_eeEventScheduler.cancel(
            m_vu1ExecutionTiming.eventToken);
    }
    ++m_vu1ExecutionTiming.generation;
    if (m_vu1ExecutionTiming.generation == 0u)
        ++m_vu1ExecutionTiming.generation;
    m_vu1ExecutionTiming.lastAdvancedTick = startTick;
    m_vu1ExecutionTiming.totalAdvancedCycles = 0u;
    m_vu1.resumeState(top, itop, &m_memory);
    setVU1BusyFlag(nullptr, true);
    scheduleVU1Event(
        ps2x::timing::saturatingAdd(
            startTick,
            ps2x::timing::vuCyclesToEeTicks(
                kVu1StartupEventCycles)));
}

void PS2Runtime::serviceVU1AtEvent(
    R5900Context *ctx,
    const ps2x::timing::EeEventService &service)
{
    if (m_vu1ExecutionTiming.eventToken.generation == 0u ||
        service.generation !=
            m_vu1ExecutionTiming.eventToken.generation)
    {
        return;
    }
    m_vu1ExecutionTiming.eventToken = {
        ps2x::timing::EeEventSource::VifVu1Finish, 0u};

    if (!m_vu1.isActive())
    {
        setVU1BusyFlag(ctx, false);
        (void)m_memory.resumeVIF1AfterVu();
        if (!m_vu1.isActive() &&
            !m_memory.vif1DmaSnapshot().active)
            m_memory.finishVif1DmaTrace();
        return;
    }

    setVU1BusyFlag(ctx, true);
    const uint64_t elapsedCycles =
        ps2x::timing::eeTicksToVuCyclesFloor(
            ps2x::timing::elapsedEeTicks(
                m_vu1ExecutionTiming.lastAdvancedTick,
                service.serviceTick));
    const uint32_t cycleBudget =
        static_cast<uint32_t>(
            std::min<uint64_t>(
                elapsedCycles,
                kVu1MaximumAdvanceCycles));
    if (cycleBudget == 0u)
    {
        scheduleVU1Event(
            ps2x::timing::saturatingAdd(
                service.serviceTick,
                ps2x::timing::vuCyclesToEeTicks(1u)));
        return;
    }

    const VU1AdvanceResult advance =
        m_vu1.advance(
            m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
            m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
            m_gs, &m_memory, cycleBudget);
    m_vu1ExecutionTiming.totalAdvancedCycles +=
        advance.executedCycles;

    if (m_vu1.isActive())
    {
        m_vu1ExecutionTiming.lastAdvancedTick =
            ps2x::timing::saturatingAdd(
                m_vu1ExecutionTiming.lastAdvancedTick,
                ps2x::timing::vuCyclesToEeTicks(
                    advance.executedCycles));
        scheduleVU1Event(
            ps2x::timing::saturatingAdd(
                service.serviceTick,
                ps2x::timing::vuCyclesToEeTicks(
                    kVu1FollowupEventCycles)));
        return;
    }

    m_vu1ExecutionTiming.lastAdvancedTick =
        service.serviceTick;
    setVU1BusyFlag(ctx, false);
    (void)m_memory.resumeVIF1AfterVu();
    if (!m_vu1.isActive() &&
        !m_memory.vif1DmaSnapshot().active)
        m_memory.finishVif1DmaTrace();
}

bool PS2Runtime::initialize(const char *title)
{
    try
    {
        if (!m_memory.initialize())
        {
            std::cerr << "Failed to initialize PS2 memory" << std::endl;
            return false;
        }

        if (!syncCoreSubsystems())
        {
            std::cerr << "Failed to bind runtime core subsystems" << std::endl;
            return false;
        }
#if defined(PS2X_IOP_ENABLE_PLUGINS) && PS2X_IOP_ENABLE_PLUGINS && \
    !defined(PLATFORM_VITA) && (defined(_WIN32) || defined(__linux__))
        std::string pluginError;
        if (!m_iopSubsystem->loadPlugins(&pluginError))
        {
            std::cerr << "Failed to load IOP plugins: " << pluginError << std::endl;
            return false;
        }
#endif
#if defined(PLATFORM_VITA)
        InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, title); // raylib vita does not support audio
#else
        SetConfigFlags(FLAG_WINDOW_RESIZABLE);
        InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, title);
        InitAudioDevice();
        m_audioBackend.setAudioReady(IsAudioDeviceReady());
#endif
        SetTargetFPS(60);
        if (m_debugUiInitCallback)
        {
            m_debugUiInitCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = true;
        }

#if defined(PS2X_ENABLE_DEBUG_SERVER) && PS2X_ENABLE_DEBUG_SERVER
        m_debugServer = std::make_unique<PS2DebugServer>(*this);
        (void)m_debugServer->start();
#endif
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to initialize PS2 runtime: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Failed to initialize PS2 runtime: unknown exception" << std::endl;
    }

    return false;
}

bool PS2Runtime::loadELF(const std::string &elfPath)
{
    configureIoPathsFromElf(elfPath);

    std::ifstream file(elfPath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open ELF file: " << elfPath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    const std::streamoff fileSize = file.tellg();
    if (fileSize < static_cast<std::streamoff>(sizeof(ElfHeader)))
    {
        std::cerr << "ELF file is too small: " << elfPath << std::endl;
        return false;
    }
    file.seekg(0, std::ios::beg);

    ElfHeader header{};
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header)))
    {
        std::cerr << "Failed to read ELF header from: " << elfPath << std::endl;
        return false;
    }

    if (header.magic != ELF_MAGIC)
    {
        std::cerr << "Invalid ELF magic number" << std::endl;
        return false;
    }

    if (header.elf_class != 1u || header.endianness != 1u)
    {
        std::cerr << "Unsupported ELF format (expected 32-bit little-endian)." << std::endl;
        return false;
    }

    if (header.machine != EM_MIPS || header.type != ET_EXEC)
    {
        std::cerr << "Not a MIPS executable ELF file" << std::endl;
        return false;
    }

    if (header.phnum != 0u && header.phentsize < sizeof(ProgramHeader))
    {
        std::cerr << "Unsupported ELF program-header entry size: " << header.phentsize << std::endl;
        return false;
    }

    const uint64_t programHeaderTableEnd =
        static_cast<uint64_t>(header.phoff) +
        static_cast<uint64_t>(header.phnum) * static_cast<uint64_t>(header.phentsize);
    if (programHeaderTableEnd > static_cast<uint64_t>(fileSize))
    {
        std::cerr << "ELF program-header table is out of range." << std::endl;
        return false;
    }

    m_cpuContext.pc = header.entry;
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);

    uint32_t maxLoadedRdramEnd = kGuestHeapDefaultBase;
    uint32_t moduleBase = std::numeric_limits<uint32_t>::max();
    uint32_t moduleEnd = 0u;
    bool loadedAnySegment = false;
    std::vector<ElfAddressRange> executableLoadRanges;

    for (uint16_t i = 0; i < header.phnum; i++)
    {
        const uint64_t phOffset =
            static_cast<uint64_t>(header.phoff) +
            static_cast<uint64_t>(i) * static_cast<uint64_t>(header.phentsize);
        if (phOffset + sizeof(ProgramHeader) > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF program header " << i << " is out of range." << std::endl;
            return false;
        }

        ProgramHeader ph{};
        file.seekg(static_cast<std::streamoff>(phOffset), std::ios::beg);
        if (!file.read(reinterpret_cast<char *>(&ph), sizeof(ph)))
        {
            std::cerr << "Failed to read ELF program header " << i << std::endl;
            return false;
        }

        if (ph.type != PT_LOAD || ph.memsz == 0u)
        {
            continue;
        }

        if (ph.filesz > ph.memsz)
        {
            std::cerr << "ELF segment " << i << " has filesz > memsz." << std::endl;
            return false;
        }

        const uint64_t segmentFileEnd = static_cast<uint64_t>(ph.offset) + static_cast<uint64_t>(ph.filesz);
        if (segmentFileEnd > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF segment " << i << " exceeds file bounds." << std::endl;
            return false;
        }

        const bool scratch =
            ph.vaddr >= PS2_SCRATCHPAD_BASE &&
            ph.vaddr < (PS2_SCRATCHPAD_BASE + PS2_SCRATCHPAD_SIZE);

        uint32_t physAddr = 0u;
        try
        {
            physAddr = m_memory.translateAddress(ph.vaddr);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to translate ELF segment " << i
                      << " virtual address 0x" << std::hex << ph.vaddr
                      << std::dec << ": " << e.what() << std::endl;
            return false;
        }
        const uint64_t regionSize = scratch ? static_cast<uint64_t>(PS2_SCRATCHPAD_SIZE)
                                            : static_cast<uint64_t>(PS2_RAM_SIZE);
        const uint64_t segmentMemEnd = static_cast<uint64_t>(physAddr) + static_cast<uint64_t>(ph.memsz);
        if (segmentMemEnd > regionSize)
        {
            std::cerr << "ELF segment " << i << " exceeds "
                      << (scratch ? "scratchpad" : "RDRAM")
                      << " bounds (vaddr=0x" << std::hex << ph.vaddr
                      << " memsz=0x" << ph.memsz << std::dec << ")." << std::endl;
            return false;
        }

        uint8_t *destBase = scratch ? m_memory.getScratchpad() : m_memory.getRDRAM();
        if (!destBase)
        {
            std::cerr << "ELF segment " << i << " has no destination memory backing." << std::endl;
            return false;
        }

        uint8_t *dest = destBase + physAddr;
        if (ph.filesz > 0u)
        {
            file.seekg(static_cast<std::streamoff>(ph.offset), std::ios::beg);
            if (!file.read(reinterpret_cast<char *>(dest), ph.filesz))
            {
                std::cerr << "Failed to read ELF segment " << i << " payload." << std::endl;
                return false;
            }
        }

        if (ph.memsz > ph.filesz)
        {
            std::memset(dest + ph.filesz, 0, ph.memsz - ph.filesz);
        }

        RUNTIME_LOG("Loading segment: 0x" << std::hex << ph.vaddr
                                          << " - 0x" << (static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz))
                                          << " (filesz: 0x" << ph.filesz
                                          << ", memsz: 0x" << ph.memsz << ")"
                                          << std::dec << std::endl);

        if (!scratch)
        {
            maxLoadedRdramEnd = std::max(maxLoadedRdramEnd, static_cast<uint32_t>(segmentMemEnd));
        }

        if (ph.flags & 0x1u) // PF_X
        {
            const uint64_t execEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.filesz);
            if (execEnd <= std::numeric_limits<uint32_t>::max() && execEnd > ph.vaddr)
            {
                executableLoadRanges.push_back({
                    ph.vaddr,
                    static_cast<uint32_t>(execEnd),
                });
            }
        }

        loadedAnySegment = true;
        moduleBase = std::min(moduleBase, ph.vaddr);
        const uint64_t segmentVirtualEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz);
        const uint32_t clampedVirtualEnd =
            (segmentVirtualEnd > std::numeric_limits<uint32_t>::max())
                ? std::numeric_limits<uint32_t>::max()
                : static_cast<uint32_t>(segmentVirtualEnd);
        moduleEnd = std::max(moduleEnd, clampedVirtualEnd);
    }

    if (!loadedAnySegment)
    {
        std::cerr << "ELF contains no loadable PT_LOAD segments." << std::endl;
        return false;
    }

    mergeAddressRanges(executableLoadRanges);
    std::vector<ElfAddressRange> executableSectionRanges;
    const bool hasExecutableSectionRanges =
        readExecutableSectionRanges(file,
                                    static_cast<uint64_t>(fileSize),
                                    header,
                                    executableLoadRanges,
                                    executableSectionRanges);
    const std::vector<ElfAddressRange> &codeRanges =
        hasExecutableSectionRanges ? executableSectionRanges : executableLoadRanges;
    for (const ElfAddressRange &range : codeRanges)
    {
        m_memory.registerCodeRegion(range.start, range.end);
    }
    RUNTIME_LOG("Registered " << codeRanges.size()
                              << (hasExecutableSectionRanges
                                      ? " executable ELF section range(s)"
                                      : " executable ELF segment fallback range(s)")
                              << std::endl);

    if (maxLoadedRdramEnd > PS2_RAM_SIZE)
    {
        maxLoadedRdramEnd = PS2_RAM_SIZE;
    }

    const uint32_t paddedEnd = (maxLoadedRdramEnd > (PS2_RAM_SIZE - kGuestHeapSafetyPad))
                                   ? PS2_RAM_SIZE
                                   : (maxLoadedRdramEnd + kGuestHeapSafetyPad);
    const uint32_t suggestedHeapBase = alignGuestHeapValue(paddedEnd, kGuestHeapDefaultAlignment);
    {
        std::lock_guard<std::mutex> lock(m_guestHeapMutex);
        if (!m_guestHeapConfigured)
        {
            const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
            m_guestHeapSuggestedBase = std::min(suggestedHeapBase, hardLimit);
            m_guestHeapBase = m_guestHeapSuggestedBase;
            m_guestHeapEnd = m_guestHeapSuggestedBase;
            m_guestHeapLimit = hardLimit;
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
        const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
        m_asyncCallbackStackFloor = std::min(std::max(hardLimit, suggestedHeapBase), PS2_RAM_SIZE);
        m_asyncCallbackStackTop = PS2_RAM_SIZE;
    }

    LoadedModule module;
    module.name = elfPath.substr(elfPath.find_last_of("/\\") + 1);
    module.baseAddress = (moduleBase == std::numeric_limits<uint32_t>::max()) ? 0x00100000u : moduleBase;
    module.size = (moduleEnd > module.baseAddress) ? static_cast<size_t>(moduleEnd - module.baseAddress) : 0u;
    module.active = true;

    m_loadedModules.push_back(module);

    uint32_t elfCrc32 = 0u;
    const bool elfCrc32Valid = computeFileCrc32(elfPath, elfCrc32);
    if (!elfCrc32Valid)
    {
        std::cerr << "[ps2xIOP] failed to compute ELF CRC32 for '" << elfPath << "'" << std::endl;
    }
    ps2x::iop::GameIdentity identity;
    identity.elfName = module.name;
    identity.entryPoint = m_cpuContext.pc;
    identity.crc32 = elfCrc32;
    std::string iopError;
    if (!m_iopSubsystem->configure(identity, &iopError))
    {
        std::cerr << "[ps2xIOP] failed to configure profile: " << iopError << std::endl;
        return false;
    }

    ps2_game_overrides::applyMatching(*this,
                                      elfPath,
                                      m_cpuContext.pc,
                                      elfCrc32,
                                      elfCrc32Valid);

    RUNTIME_LOG("ELF file loaded successfully. Entry point: 0x" << std::hex << m_cpuContext.pc << std::dec);
    return true;
}

const PS2Runtime::IoPaths &PS2Runtime::getIoPaths()
{
    return runtimeIoPaths();
}

void PS2Runtime::setIoPaths(const IoPaths &paths)
{
    IoPaths normalized = paths;
    normalized.elfPath = normalizeAbsolutePath(normalized.elfPath);
    normalized.elfDirectory = normalizeAbsolutePath(normalized.elfDirectory);
    normalized.hostRoot = normalizeAbsolutePath(normalized.hostRoot);
    normalized.cdRoot = normalizeAbsolutePath(normalized.cdRoot);
    normalized.mcRoot = normalizeAbsolutePath(normalized.mcRoot);
    normalized.cdImage = normalizeAbsolutePath(normalized.cdImage);

    if (normalized.elfDirectory.empty() && !normalized.elfPath.empty())
    {
        normalized.elfDirectory = normalized.elfPath.parent_path();
    }

    if (normalized.hostRoot.empty())
    {
        normalized.hostRoot = normalized.elfDirectory;
    }
    if (normalized.cdRoot.empty())
    {
        normalized.cdRoot = normalized.elfDirectory;
    }
    if (normalized.mcRoot.empty())
    {
        normalized.mcRoot = normalized.elfDirectory / "mc0";
    }

    runtimeIoPaths() = normalized;
}

void PS2Runtime::configureIoPathsFromElf(const std::string &elfPath)
{
    IoPaths paths = runtimeIoPaths();
    paths.elfPath = normalizeAbsolutePath(std::filesystem::path(elfPath));
    if (!paths.elfPath.empty())
    {
        paths.elfDirectory = paths.elfPath.parent_path();
    }

    if (!paths.elfDirectory.empty())
    {
        paths.hostRoot = paths.elfDirectory;
        paths.cdRoot = paths.elfDirectory;
        paths.mcRoot = paths.elfDirectory / "mc0";
    }

    setIoPaths(paths);
}

namespace
{
    uint32_t normalizeGuestFunctionAddress(uint32_t address)
    {
        uint32_t offset = 0u;
        if (ps2ResolveDirectRdramOffset(address, offset))
        {
            return offset;
        }
        return address;
    }

    bool generatedFunctionTableSlot(uint32_t address, uint32_t &slot)
    {
        address = normalizeGuestFunctionAddress(address);
        if ((address & 3u) != 0u || g_ps2RecompiledFunctionTableSlotCount == 0u)
        {
            return false;
        }

        if (address < g_ps2RecompiledFunctionTableBase || address >= g_ps2RecompiledFunctionTableEnd)
        {
            return false;
        }

        const uint32_t offset = address - g_ps2RecompiledFunctionTableBase;
        slot = offset >> 2;
        return slot < g_ps2RecompiledFunctionTableSlotCount;
    }

    const PS2GuestFunctionSymbol *findGuestFunctionSymbol(uint32_t address)
    {
        address = normalizeGuestFunctionAddress(address);
        uint32_t first = 0u;
        uint32_t last = g_ps2GuestFunctionSymbolCount;
        while (first < last)
        {
            const uint32_t middle = first + ((last - first) / 2u);
            if (g_ps2GuestFunctionSymbols[middle].start <= address)
            {
                first = middle + 1u;
            }
            else
            {
                last = middle;
            }
        }

        if (first == 0u)
        {
            return nullptr;
        }

        const PS2GuestFunctionSymbol &symbol = g_ps2GuestFunctionSymbols[first - 1u];
        if (address < symbol.end && symbol.name != nullptr && symbol.name[0] != '\0')
        {
            return &symbol;
        }
        return nullptr;
    }
}

std::string PS2Runtime::formatGuestPc(uint32_t address)
{
    std::ostringstream oss;
    oss << "0x" << std::hex << address;

    const PS2GuestFunctionSymbol *symbol = findGuestFunctionSymbol(address);
    if (symbol != nullptr)
    {
        const uint32_t normalizedAddress = normalizeGuestFunctionAddress(address);
        const uint32_t offset = normalizedAddress - symbol->start;
        oss << '<' << symbol->name;
        if (offset != 0u)
        {
            oss << "+0x" << offset;
        }
        oss << '>';
    }

    return oss.str();
}

bool PS2Runtime::replaceFunction(uint32_t address, RecompiledFunction func)
{
    uint32_t slot = 0u;
    if (!generatedFunctionTableSlot(address, slot))
    {
        std::cerr << "[function-table] cannot replace guest PC 0x" << std::hex << address
                  << ": outside generated dense table [0x" << g_ps2RecompiledFunctionTableBase
                  << ", 0x" << g_ps2RecompiledFunctionTableEnd << ")"
                  << std::dec << std::endl;
        return false;
    }

    g_ps2RecompiledFunctionTable[slot] = func;
    return true;
}

bool PS2Runtime::registerFunction(uint32_t address, RecompiledFunction func)
{
    return replaceFunction(address, func);
}

bool PS2Runtime::hasFunction(uint32_t address) const
{
    uint32_t slot = 0u;
    return generatedFunctionTableSlot(address, slot) && g_ps2RecompiledFunctionTable[slot] != nullptr;
}

const char *describeGuestBranchKind(PS2Runtime::GuestBranchKind kind)
{
    switch (kind)
    {
    case PS2Runtime::GuestBranchKind::DirectJump:
        return "DirectJump";
    case PS2Runtime::GuestBranchKind::DirectCall:
        return "DirectCall";
    case PS2Runtime::GuestBranchKind::IndirectJump:
        return "IndirectJump";
    case PS2Runtime::GuestBranchKind::IndirectCall:
        return "IndirectCall";
    case PS2Runtime::GuestBranchKind::Return:
        return "Return";
    default:
        return "Unknown";
    }
}

PS2Runtime::RecompiledFunction PS2Runtime::lookupFunction(uint32_t address)
{
    pushDispatchPc(address);
    debugRecordBranch(address);

    const uint32_t normalizedAddress = normalizeGuestFunctionAddress(address);
    m_debugPc.store(normalizedAddress, std::memory_order_relaxed);
    uint32_t slot = 0u;
    if (generatedFunctionTableSlot(normalizedAddress, slot))
    {
        RecompiledFunction fn = g_ps2RecompiledFunctionTable[slot];
        if (fn != nullptr)
        {
            if (normalizedAddress != address)
            {
                static RecompiledFunction directMappedAlias =
                    [](uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
                {
                    if (!ctx || !runtime)
                    {
                        return;
                    }

                    ctx->pc = normalizeGuestFunctionAddress(ctx->pc);
                    RecompiledFunction target = runtime->lookupFunction(ctx->pc);
                    target(rdram, ctx, runtime);
                };
                return directMappedAlias;
            }
            return fn;
        }
    }

    static RecompiledFunction missingFunction = [](uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t badPc = ctx->pc;
        runtime->reportMissingFunction(rdram,
                                       ctx,
                                       badPc,
                                       0u,
                                       PS2Runtime::GuestBranchKind::IndirectJump,
                                       "dispatch");
    };

    return missingFunction;
}

void PS2Runtime::setMissingFunctionPolicy(MissingFunctionPolicy policy)
{
    m_missingFunctionPolicy.store(static_cast<uint32_t>(policy), std::memory_order_release);
}

PS2Runtime::MissingFunctionPolicy PS2Runtime::missingFunctionPolicy() const
{
    return static_cast<MissingFunctionPolicy>(m_missingFunctionPolicy.load(std::memory_order_acquire));
}

void PS2Runtime::resetMissingFunctionReportOnce()
{
    m_missingFunctionReported.store(false, std::memory_order_release);
    debugClearFault();
}

void PS2Runtime::reportMissingFunction(uint8_t *rdram,
                                       R5900Context *ctx,
                                       uint32_t targetPc,
                                       uint32_t sourcePc,
                                       GuestBranchKind kind,
                                       const char *debugName)
{
    (void)rdram;
    const MissingFunctionPolicy policy = missingFunctionPolicy();
    const bool firstReport = !m_missingFunctionReported.exchange(true, std::memory_order_acq_rel);

    const uint32_t pc =
        ctx ? ctx->pc : m_debugPc.load(std::memory_order_relaxed);
    const uint32_t ra =
        ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0))
            : m_debugRa.load(std::memory_order_relaxed);
    const uint32_t sp =
        ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0))
            : m_debugSp.load(std::memory_order_relaxed);
    const uint32_t gp =
        ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0))
            : m_debugGp.load(std::memory_order_relaxed);
    const uint32_t a0 =
        ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[4], 0)) : 0u;
    const uint32_t a1 =
        ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[5], 0)) : 0u;
    const uint32_t v0 =
        ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[2], 0)) : 0u;
    const uint32_t v1 =
        ctx ? static_cast<uint32_t>(_mm_extract_epi32(ctx->r[3], 0)) : 0u;

    if (firstReport)
    {
        DebugFaultInfo fault{};
        fault.active = true;
        fault.type = "missing-recompiled-target";
        fault.operation = debugName ? debugName : "";
        fault.branchKind = kind;
        fault.sourcePc = sourcePc;
        fault.targetPc = targetPc;
        fault.pc = pc;
        fault.ra = ra;
        fault.sp = sp;
        fault.gp = gp;
        fault.a0 = a0;
        fault.a1 = a1;
        fault.v0 = v0;
        fault.v1 = v1;
        fault.codeRegion = m_memory.isCodeAddress(targetPc);
        fault.policy = policy;
        debugRecordFault(fault);
        fault = debugFaultSnapshot();

        std::ostringstream oss;
        oss << "{\"schema_version\":1,"
            << "\"event\":\"missing-recompiled-target\","
            << "\"sequence\":" << fault.sequence << ','
            << "\"kind\":\"" << describeGuestBranchKind(kind) << "\","
            << "\"operation\":\""
            << escapeJsonString(debugName ? debugName : "") << "\","
            << "\"source\":\"" << rawGuestAddress(sourcePc) << "\","
            << "\"target\":\"" << rawGuestAddress(targetPc) << "\","
            << "\"pc\":\"" << rawGuestAddress(pc) << "\","
            << "\"ra\":\"" << rawGuestAddress(ra) << "\","
            << "\"sp\":\"" << rawGuestAddress(sp) << "\","
            << "\"gp\":\"" << rawGuestAddress(gp) << "\","
            << "\"code_region\":" << (fault.codeRegion ? "true" : "false")
            << ",\"policy\":" << static_cast<uint32_t>(policy)
            << ",\"trace_entries\":"
            << debugBranchHistory(kDebugBranchHistoryCapacity).size()
            << '}';

        static std::mutex s_missingFunctionLogMutex;
        {
            std::lock_guard<std::mutex> lock(s_missingFunctionLogMutex);
            std::cerr << oss.str() << std::endl;
        }
    }

    if (firstReport && policy == MissingFunctionPolicy::BreakOnce)
    {
#if defined(_MSC_VER)
        __debugbreak();
#endif // TODO others breakpoints
    }

    if (ctx)
    {
        ctx->pc = targetPc;
    }

    if (policy == MissingFunctionPolicy::Stop)
    {
        requestStop();
    }
}

bool PS2Runtime::dispatchGuestBranch(uint8_t *rdram,
                                     R5900Context *ctx,
                                     uint32_t targetPc,
                                     uint32_t sourcePc,
                                     uint32_t fallthroughPc,
                                     GuestBranchKind kind,
                                     const char *debugName)
{
    targetPc = normalizeGuestFunctionAddress(targetPc);
    ctx->pc = targetPc;
    const bool isCall = (kind == GuestBranchKind::DirectCall || kind == GuestBranchKind::IndirectCall);

    if (kind == GuestBranchKind::Return)
    {
        if (!hasFunction(targetPc))
        {
            reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);
        }

        // Prevent nested dispatch.
        ctx->pc = targetPc;
        return false;
    }

    if (!hasFunction(targetPc))
    {
        reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);

        const MissingFunctionPolicy policy = missingFunctionPolicy();

        if (policy == MissingFunctionPolicy::SkipCallDebug && isCall)
        {
            ctx->pc = fallthroughPc;
            return true;
        }

        if (policy == MissingFunctionPolicy::ContinueToTarget)
        {
            ctx->pc = targetPc;
            return true;
        }

        return false;
    }

    RecompiledFunction targetFn = lookupFunction(targetPc);
    const uint32_t entryPc = ctx->pc;
    const uint64_t preemptionEpoch =
        m_guestExecutionPreemptionEpoch.load(std::memory_order_acquire);
    targetFn(rdram, ctx, this);

    if (isStopRequested() || ctx->pc == 0u)
    {
        return false;
    }

    // A generated back edge returns to the outer dispatcher when the runtime
    // requests cooperative preemption.  In a nested call, the resume PC can
    // legitimately equal the callee entry, which is otherwise also how an
    // empty implicit-return handler is represented.  Preserve the resume PC
    // instead of converting it to the call fallthrough when any nested target
    // observed a preemption request.
    if (m_guestExecutionPreemptionEpoch.load(std::memory_order_acquire) !=
        preemptionEpoch)
    {
        return false;
    }

    if (!isCall)
    {
        return false;
    }

    if (ctx->pc == entryPc)
    {
        ctx->pc = fallthroughPc;
    }

    return ctx->pc == fallthroughPc;
}

[[noreturn]] void PS2Runtime::SignalException(R5900Context *ctx, PS2Exception exception)
{
    if (exception == EXCEPTION_INTEGER_OVERFLOW)
    {
        HandleIntegerOverflow(ctx);
    }

    raiseCop0Exception(ctx, static_cast<uint32_t>(exception));
}

[[noreturn]] void PS2Runtime::SignalMemoryException(R5900Context *ctx,
                                                    PS2Exception exception,
                                                    uint32_t badVAddr)
{
    ctx->cop0_badvaddr = badVAddr;

    if (exception == EXCEPTION_TLB_MODIFIED ||
        exception == EXCEPTION_TLB_REFILL_LOAD ||
        exception == EXCEPTION_TLB_REFILL_STORE)
    {
        ctx->cop0_context = (ctx->cop0_context & 0xFF80000Fu) |
                            ((badVAddr >> 9u) & 0x007FFFF0u);
        ctx->cop0_entryhi = (badVAddr & 0xFFFFE000u) |
                            (ctx->cop0_entryhi & 0x00001FFFu);
    }

    SignalException(ctx, exception);
}

void PS2Runtime::debugArmVu0Traces(const R5900Context *ctx)
{
    if (!ctx)
    {
        return;
    }

    if (m_debugVu0SyncTraceEnabled.load(std::memory_order_acquire) &&
        !m_debugVu0SyncTraceTriggered.load(std::memory_order_acquire) &&
        m_debugVu0SyncTraceHasTrigger.load(std::memory_order_relaxed) &&
        ctx->pc ==
            m_debugVu0SyncTraceTriggerEePc.load(std::memory_order_relaxed))
    {
        // VCALLMS synchronizes before the new microprogram is active. Arm at
        // that boundary so the following invocation is retained without
        // fabricating an inactive sync entry.
        m_debugVu0SyncTraceTriggered.store(true, std::memory_order_release);
    }

    if (m_debugVu0InstructionTraceEnabled.load(
            std::memory_order_acquire) &&
        !m_debugVu0InstructionTraceTriggered.load(
            std::memory_order_acquire) &&
        m_debugVu0InstructionTraceHasTrigger.load(
            std::memory_order_relaxed) &&
        ctx->pc ==
            m_debugVu0InstructionTraceTriggerEePc.load(
                std::memory_order_relaxed))
    {
        m_debugVu0InstructionTraceTriggered.store(
            true, std::memory_order_release);
    }
}

void PS2Runtime::beginVu0Invocation()
{
    const uint64_t invocation =
        m_vu0InvocationSequence.fetch_add(
            1u, std::memory_order_relaxed) +
        1u;
    m_vu0CurrentInvocation.store(
        invocation, std::memory_order_relaxed);
    m_vu0CurrentInvocationInstruction.store(
        0u, std::memory_order_relaxed);
}

void PS2Runtime::executeVU0Microprogram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    (void)rdram;

    uint8_t *const vu0Code = m_memory.getVU0Code();
    uint8_t *const vu0Data = m_memory.getVU0Data();
    const uint32_t startPC = address & ~0x7u;

    if (!vu0Code || !vu0Data || startPC + 8u > PS2_VU0_CODE_SIZE)
    {
        seedVu0IdleSuccess(ctx);
        return;
    }

    m_vu0.reset();
    copyVu0ContextToState(ctx, m_vu0.state());
    beginVu0Invocation();
    m_vu0.execute(vu0Code, PS2_VU0_CODE_SIZE,
                  vu0Data, PS2_VU0_DATA_SIZE,
                  m_gs, &m_memory,
                  startPC, 0u, ctx->vu0_itop);
    copyVu0StateToContext(m_vu0.state(), ctx);
    ctx->vu0_vpu_stat =
        (ctx->vu0_vpu_stat & ~1u) | (m_vu0.isActive() ? 1u : 0u);
    const ps2x::timing::EeTick eeCycleTick = currentEeTick();
    m_vu0CycleTick = eeCycleTick;
    cancelVu0CompatibilityEvent();
}

void PS2Runtime::vu0StartMicroProgram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    (void)rdram;

    synchronizeVU0Microprogram(rdram, ctx, true);

    uint8_t *const vu0Code = m_memory.getVU0Code();
    uint8_t *const vu0Data = m_memory.getVU0Data();
    const uint32_t startPC = address & ~0x7u;

    if (!vu0Code || !vu0Data || startPC + 8u > PS2_VU0_CODE_SIZE)
    {
        seedVu0IdleSuccess(ctx);
        return;
    }

    m_vu0.reset();
    copyVu0ContextToState(ctx, m_vu0.state());
    beginVu0Invocation();
    m_vu0.execute(vu0Code, PS2_VU0_CODE_SIZE,
                  vu0Data, PS2_VU0_DATA_SIZE,
                  m_gs, &m_memory,
                  startPC, 0u, ctx->vu0_itop, kVu0MinimumRunCycles);
    copyVu0StateToContext(m_vu0.state(), ctx);
    ctx->vu0_vpu_stat =
        (ctx->vu0_vpu_stat & ~1u) | (m_vu0.isActive() ? 1u : 0u);
    const ps2x::timing::EeTick eeCycleTick = currentEeTick();
    m_vu0CycleTick = eeCycleTick;
    if (m_vu0.isActive())
    {
        m_vu0CycleTick = ps2x::timing::saturatingAdd(
            m_vu0CycleTick,
            ps2x::timing::vuCyclesToEeTicks(
                kVu0MinimumRunCycles));
    }
    if (m_vu0.isActive())
    {
        scheduleVu0CompatibilityEvent(
            ps2x::timing::saturatingAdd(
                eeCycleTick,
                ps2x::timing::eeCyclesToTicks(
                    kVu0InitialEventDelayCycles)));
    }
    else
    {
        cancelVu0CompatibilityEvent();
    }
}

void PS2Runtime::synchronizeVU0Microprogram(
    uint8_t *rdram, R5900Context *ctx, bool interlocked)
{
    debugArmVu0Traces(ctx);
    if (!ctx)
    {
        return;
    }

    const ps2x::timing::EeTick eeCycleTick =
        commitEeContextProgress(ctx);
    if (m_eeSchedulingMode ==
            ps2x::timing::EeSchedulingMode::Event &&
        m_eeEventScheduler.deadlineDue(eeCycleTick))
    {
        // A COP2 synchronization is also a safe EE event boundary. Service
        // the complete due device batch before publishing VU0 state, just
        // as PCSX2 does when an architectural access reaches an event test.
        serviceEeEventsAtTick(rdram, ctx, eeCycleTick);
    }
    if (!m_vu0.isActive())
    {
        ctx->vu0_vpu_stat &= ~1u;
        m_vu0CycleTick = eeCycleTick;
        cancelVu0CompatibilityEvent();
        return;
    }

    ps2x::timing::EeTick nextEventTick{};
    if (m_eeSchedulingMode ==
        ps2x::timing::EeSchedulingMode::Event)
    {
        const auto scheduled =
            m_eeEventScheduler.nextDeadline();
        if (scheduled.has_value())
        {
            nextEventTick = *scheduled;
        }
    }
    else
    {
        nextEventTick = m_vu0NextEventCycleTick;
    }

    synchronizeVU0MicroprogramAtTick(
        rdram, ctx, interlocked, false, false,
        eeCycleTick, nextEventTick);
    if (!m_vu0.isActive())
    {
        cancelVu0CompatibilityEvent();
    }
}

void PS2Runtime::advanceVU0AtEeBlockBoundary(
    uint8_t *rdram, R5900Context *ctx)
{
    debugArmVu0Traces(ctx);
    if (!ctx)
    {
        return;
    }
    advanceVU0LegacyAtBlockBoundary(
        rdram, ctx, finishEeContextBlock(ctx));
}

void PS2Runtime::serviceEeEventsAtBlockBoundary(
    uint8_t *rdram, R5900Context *ctx)
{
    debugArmVu0Traces(ctx);
    if (!ctx)
    {
        return;
    }

    const ps2x::timing::EeTick eeCycleTick =
        finishEeContextBlock(ctx);
    if (m_eeSchedulingMode !=
        ps2x::timing::EeSchedulingMode::Event)
    {
        publishCompatibilityDmacCompletions();
    }
    switch (m_eeSchedulingMode)
    {
    case ps2x::timing::EeSchedulingMode::Legacy:
        advanceVU0LegacyAtBlockBoundary(
            rdram, ctx, eeCycleTick);
        break;
    case ps2x::timing::EeSchedulingMode::Shadow:
    {
        if (!checkEeSchedulerShadow(eeCycleTick))
        {
            return;
        }

        advanceVU0LegacyAtBlockBoundary(
            rdram, ctx, eeCycleTick);
        const ps2x::timing::EeEventServiceResult result =
            m_eeEventScheduler.serviceDue(
                eeCycleTick,
                [&](const ps2x::timing::EeEventService &service)
                {
                    if (service.source ==
                            ps2x::timing::EeEventSource::
                                Vu0PeriodicCompatibility &&
                        m_vu0.isActive())
                    {
                        (void)m_eeEventScheduler.scheduleAbsolute(
                            service.source,
                            nextVu0CompatibilityDeadline(
                                service.scheduledTick,
                                service.serviceTick));
                    }

                    DebugEeEventEntry entry{};
                    entry.source = service.source;
                    entry.eventSequence = service.sequence;
                    entry.generation = service.generation;
                    entry.scheduledTick =
                        service.scheduledTick.raw();
                    entry.serviceTick =
                        service.serviceTick.raw();
                    entry.latenessTicks =
                        service.latenessTicks.raw();
                    const auto followup =
                        m_eeEventScheduler.event(service.source);
                    if (followup.has_value())
                    {
                        entry.hasFollowup = true;
                        entry.rescheduled = true;
                        entry.followupTick =
                            followup->deadline.raw();
                    }
                    debugRecordEeEvent(std::move(entry));
                });
        if (!m_vu0.isActive())
        {
            (void)m_eeEventScheduler.cancel(
                ps2x::timing::EeEventSource::
                    Vu0PeriodicCompatibility);
        }
        if (result.limitExceeded)
        {
            std::cerr
                << "EE scheduler service limit exceeded in shadow mode"
                << std::endl;
            requestStop();
            return;
        }
        (void)checkEeSchedulerShadow(eeCycleTick);
        break;
    }
    case ps2x::timing::EeSchedulingMode::Event:
        serviceEeEventsAtTick(rdram, ctx, eeCycleTick);
        break;
    }
}

ps2x::timing::EeTick
PS2Runtime::nextVu0CompatibilityDeadline(
    ps2x::timing::EeTick scheduledTick,
    ps2x::timing::EeTick serviceTick) const noexcept
{
    const ps2x::timing::EeTickDelta intervalTicks =
        ps2x::timing::eeCyclesToTicks(
            kEeRapidEventCycles);
    ps2x::timing::EeTick next = scheduledTick;
    do
    {
        const ps2x::timing::EeTick previous = next;
        next = ps2x::timing::saturatingAdd(
            next, intervalTicks);
        if (next == previous)
        {
            break;
        }
    } while (next <= serviceTick);
    return next;
}

void PS2Runtime::scheduleVu0CompatibilityEvent(
    ps2x::timing::EeTick deadline) noexcept
{
    m_vu0NextEventCycleTick = deadline;
    if (m_eeSchedulingMode ==
        ps2x::timing::EeSchedulingMode::Shadow)
    {
        (void)m_eeEventScheduler.scheduleAbsolute(
            ps2x::timing::EeEventSource::
                Vu0PeriodicCompatibility,
            deadline);
    }
}

void PS2Runtime::cancelVu0CompatibilityEvent() noexcept
{
    m_vu0NextEventCycleTick = {};
    (void)m_eeEventScheduler.cancel(
        ps2x::timing::EeEventSource::
            Vu0PeriodicCompatibility);
}

void PS2Runtime::advanceVU0LegacyAtBlockBoundary(
    uint8_t *rdram,
    R5900Context *ctx,
    ps2x::timing::EeTick eeCycleTick)
{
    if (!m_vu0.isActive())
    {
        ctx->vu0_vpu_stat &= ~1u;
        m_vu0CycleTick = eeCycleTick;
        m_vu0NextEventCycleTick = {};
        return;
    }

    const ps2x::timing::EeTick scheduledTick =
        m_vu0NextEventCycleTick;
    const bool eventDue =
        scheduledTick != ps2x::timing::EeTick{} &&
        eeCycleTick >= scheduledTick;
    synchronizeVU0MicroprogramAtTick(
        rdram, ctx, false, true, eventDue,
        eeCycleTick, scheduledTick);
    if (!m_vu0.isActive())
    {
        m_vu0NextEventCycleTick = {};
    }
    else if (eventDue)
    {
        m_vu0NextEventCycleTick =
            nextVu0CompatibilityDeadline(
                scheduledTick, eeCycleTick);
    }
}

void PS2Runtime::dispatchEeEvent(
    uint8_t *rdram,
    R5900Context *ctx,
    const ps2x::timing::EeEventService &service)
{
    switch (service.source)
    {
    case ps2x::timing::EeEventSource::
        DmacCompletion:
        (void)publishReadyDmacCompletions(
            service.sequence);
        if (m_dmacCompletionReady.load(
                std::memory_order_acquire))
        {
            (void)m_eeEventScheduler.scheduleAbsolute(
                ps2x::timing::EeEventSource::
                    DmacCompletion,
                service.serviceTick);
        }
        return;
    case ps2x::timing::EeEventSource::
        Vu0PeriodicCompatibility:
        if (!m_vu0.isActive())
        {
            ctx->vu0_vpu_stat &= ~1u;
            m_vu0CycleTick = service.serviceTick;
            cancelVu0CompatibilityEvent();
            return;
        }
        scheduleVu0CompatibilityEvent(
            nextVu0CompatibilityDeadline(
                service.scheduledTick,
                service.serviceTick));
        return;
    case ps2x::timing::EeEventSource::DmacVif1:
        serviceVif1DmaAtEvent(service);
        return;
    case ps2x::timing::EeEventSource::
        DmacFromScratchpad:
        serviceScratchpadDmaAtEvent(
            DmacChannel::FromScratchpad, service);
        return;
    case ps2x::timing::EeEventSource::
        DmacToScratchpad:
        serviceScratchpadDmaAtEvent(
            DmacChannel::ToScratchpad, service);
        return;
    case ps2x::timing::EeEventSource::VifVu1Finish:
        serviceVU1AtEvent(ctx, service);
        return;
    case ps2x::timing::EeEventSource::Count:
        return;
    }
}

void PS2Runtime::serviceEeEventsAtTick(
    uint8_t *rdram,
    R5900Context *ctx,
    ps2x::timing::EeTick eeCycleTick)
{
    if (!m_eeEventScheduler.deadlineDue(eeCycleTick))
    {
        return;
    }

    const ps2x::timing::EeEventServiceResult result =
        m_eeEventScheduler.serviceDue(
            eeCycleTick,
            [&](const ps2x::timing::EeEventService &service)
            {
                const bool traceEnabled =
                    m_debugEeEventTraceEnabled.load(
                        std::memory_order_relaxed);
                DebugEeEventEntry entry{};
                if (traceEnabled)
                {
                    entry.source = service.source;
                    entry.eventSequence = service.sequence;
                    entry.generation = service.generation;
                    entry.scheduledTick =
                        service.scheduledTick.raw();
                    entry.serviceTick =
                        service.serviceTick.raw();
                    entry.latenessTicks =
                        service.latenessTicks.raw();
                    entry.eePc = ctx ? ctx->pc : 0u;
                    entry.deviceBefore =
                        debugEeEventDeviceState(
                            service.source);
                }

                dispatchEeEvent(rdram, ctx, service);

                if (!traceEnabled)
                    return;

                entry.deviceAfter =
                    debugEeEventDeviceState(service.source);
                const auto followup =
                    m_eeEventScheduler.event(service.source);
                if (followup.has_value())
                {
                    entry.hasFollowup = true;
                    entry.rescheduled = true;
                    entry.followupTick =
                        followup->deadline.raw();
                }
                debugRecordEeEvent(std::move(entry));
            });
    if (result.limitExceeded)
    {
        std::cerr << "EE scheduler service limit exceeded";
        if (result.offendingSource.has_value())
        {
            std::cerr << " for "
                      << ps2x::timing::eeEventSourceName(
                             *result.offendingSource);
        }
        std::cerr << std::endl;
        requestStop();
        return;
    }

    // PCSX2 services the complete fixed-priority device batch before its
    // ordinary VU0 catch-up. A same-tick callback scheduled by another
    // callback therefore remains part of this batch, and VU0 advances only
    // once to the actual shared-event service time.
    if (result.serviced != 0u)
    {
        ps2x::timing::EeTick nextEventTick{};
        const auto nextDeadline =
            m_eeEventScheduler.nextDeadline();
        if (nextDeadline.has_value())
            nextEventTick = *nextDeadline;

        if (m_vu0.isActive())
        {
            synchronizeVU0MicroprogramAtTick(
                rdram, ctx, false, true, true,
                eeCycleTick, nextEventTick);
        }
        else
        {
            ctx->vu0_vpu_stat &= ~1u;
            m_vu0CycleTick = eeCycleTick;
        }
    }
}

PS2Runtime::DebugEeEventDeviceState
PS2Runtime::debugEeEventDeviceState(
    ps2x::timing::EeEventSource source) const noexcept
{
    DebugEeEventDeviceState result{};
    switch (source)
    {
    case ps2x::timing::EeEventSource::DmacVif1:
    {
        const Vif1DmaSnapshot state =
            m_memory.vif1DmaSnapshot();
        result.kind = DebugEeEventDeviceKind::Vif1Dma;
        result.operationGeneration =
            state.transfer.generation;
        result.phase =
            static_cast<uint32_t>(state.phase);
        result.stall =
            static_cast<uint32_t>(state.stall);
        result.tadr = state.tadr;
        result.madr = state.madr;
        result.qwc = state.qwc;
        result.tagsProcessed = state.tagsProcessed;
        result.active = state.active;
        return result;
    }
    case ps2x::timing::EeEventSource::
        DmacFromScratchpad:
    case ps2x::timing::EeEventSource::
        DmacToScratchpad:
    {
        const DmacChannel channel =
            source ==
                    ps2x::timing::EeEventSource::
                        DmacFromScratchpad
                ? DmacChannel::FromScratchpad
                : DmacChannel::ToScratchpad;
        const ScratchpadDmaSnapshot state =
            m_memory.scratchpadDmaSnapshot(channel);
        result.kind =
            DebugEeEventDeviceKind::ScratchpadDma;
        result.operationGeneration =
            state.transfer.generation;
        result.phase =
            static_cast<uint32_t>(state.phase);
        result.tadr = state.tadr;
        result.madr = state.madr;
        result.qwc = state.qwc;
        result.tagsProcessed = state.tagsProcessed;
        result.active = state.active;
        return result;
    }
    case ps2x::timing::EeEventSource::VifVu1Finish:
        result.kind = DebugEeEventDeviceKind::Vu1;
        result.operationGeneration =
            m_vu1ExecutionTiming.generation;
        result.lastAdvancedTick =
            m_vu1ExecutionTiming.lastAdvancedTick.raw();
        result.totalAdvancedCycles =
            m_vu1ExecutionTiming.totalAdvancedCycles;
        result.pc = m_vu1.state().pc;
        result.active = m_vu1.isActive();
        return result;
    case ps2x::timing::EeEventSource::DmacCompletion:
    case ps2x::timing::EeEventSource::
        Vu0PeriodicCompatibility:
    case ps2x::timing::EeEventSource::Count:
        return result;
    }
    return result;
}

bool PS2Runtime::checkEeSchedulerShadow(
    ps2x::timing::EeTick now)
{
    if (m_eeSchedulerShadowMismatch)
    {
        return false;
    }

    const bool legacyPending =
        m_vu0NextEventCycleTick !=
        ps2x::timing::EeTick{};
    const auto scheduled = m_eeEventScheduler.event(
        ps2x::timing::EeEventSource::
            Vu0PeriodicCompatibility);
    const bool schedulerPending = scheduled.has_value();
    const ps2x::timing::EeTick schedulerDeadline =
        schedulerPending ? scheduled->deadline
                         : ps2x::timing::EeTick{};
    if (legacyPending == schedulerPending &&
        (!legacyPending ||
         m_vu0NextEventCycleTick == schedulerDeadline))
    {
        return true;
    }

    m_eeSchedulerShadowMismatch = true;
    m_eeSchedulerShadowMismatchTick = now;
    m_eeSchedulerShadowLegacyDeadline =
        m_vu0NextEventCycleTick;
    m_eeSchedulerShadowDeadline = schedulerDeadline;
    m_eeSchedulerShadowLegacyPending = legacyPending;
    m_eeSchedulerShadowPending = schedulerPending;
    requestStop();
    return false;
}

void PS2Runtime::synchronizeVU0MicroprogramAtTick(
    uint8_t *rdram, R5900Context *ctx, bool interlocked,
    bool blockBoundary, bool eventDue,
    ps2x::timing::EeTick eeCycleTick,
    ps2x::timing::EeTick nextEventTick)
{
    (void)rdram;

    const bool traceEnabled =
        m_debugVu0SyncTraceEnabled.load(std::memory_order_acquire);
    DebugVu0SyncEntry traceEntry{};
    bool traceSync =
        traceEnabled &&
        m_debugVu0SyncTraceTriggered.load(std::memory_order_acquire);
    if (traceEnabled && !traceSync &&
        m_debugVu0SyncTraceHasTrigger.load(std::memory_order_relaxed) &&
        ctx->pc ==
            m_debugVu0SyncTraceTriggerEePc.load(std::memory_order_relaxed))
    {
        m_debugVu0SyncTraceTriggered.store(true, std::memory_order_release);
        traceSync = true;
    }
    if (traceSync)
    {
        const VU1State &state = m_vu0.state();
        traceEntry.invocation =
            m_vu0CurrentInvocation.load(std::memory_order_relaxed);
        traceEntry.invocationInstruction =
            m_vu0CurrentInvocationInstruction.load(
                std::memory_order_relaxed);
        traceEntry.eeCycleTicks = eeCycleTick.raw();
        traceEntry.vuCycleTicks = m_vu0CycleTick.raw();
        traceEntry.nextEventCycleTicks =
            nextEventTick.raw();
        traceEntry.eePc = ctx->pc;
        traceEntry.vuPcBefore = state.pc;
        traceEntry.contextVi1 = ctx->vi[1];
        traceEntry.contextVi2 = ctx->vi[2];
        traceEntry.vuVi1Before = static_cast<uint16_t>(state.vi[1]);
        traceEntry.vuVi2Before = static_cast<uint16_t>(state.vi[2]);
        traceEntry.interlocked = interlocked;
        traceEntry.blockBoundary = blockBoundary;
        traceEntry.activeBefore = m_vu0.isActive();
        for (size_t index = 0u; index < 16u; ++index)
        {
            if (ctx->vi[index] != static_cast<uint16_t>(state.vi[index]))
            {
                traceEntry.pendingViMask |=
                    static_cast<uint16_t>(1u << index);
            }
        }
        for (size_t index = 0u; index < 32u; ++index)
        {
            if (std::memcmp(
                    &ctx->vu0_vf[index], state.vf[index],
                    sizeof(state.vf[index])) != 0)
            {
                traceEntry.pendingVfMask |= 1u << index;
            }
        }
    }

    uint32_t cycleBudget = 0u;
    if (interlocked)
    {
        cycleBudget = 65536u;
    }
    else if (eeCycleTick > m_vu0CycleTick)
    {
        const uint64_t elapsedCycles =
            ps2x::timing::eeTicksToVuCyclesFloor(
                ps2x::timing::elapsedEeTicks(
                    m_vu0CycleTick, eeCycleTick));
        if (blockBoundary)
        {
            if (eventDue)
            {
                cycleBudget = static_cast<uint32_t>(
                    std::min<uint64_t>(
                        std::max<uint64_t>(
                            elapsedCycles,
                            kVu0MinimumRunCycles),
                        65536u));
            }
        }
        else if (elapsedCycles >= kVu0SyncThresholdCycles)
        {
            cycleBudget = static_cast<uint32_t>(
                std::min<uint64_t>(
                    std::max<uint64_t>(
                        elapsedCycles,
                        kVu0MinimumRunCycles),
                    65536u));
        }
    }
    copyVu0ContextRegistersToState(ctx, m_vu0.state());
    if (cycleBudget != 0u)
    {
        m_vu0.continueExecution(
            m_memory.getVU0Code(), PS2_VU0_CODE_SIZE,
            m_memory.getVU0Data(), PS2_VU0_DATA_SIZE,
            m_gs, &m_memory, cycleBudget);
    }
    copyVu0StateToContext(m_vu0.state(), ctx);
    ctx->vu0_vpu_stat =
        (ctx->vu0_vpu_stat & ~1u) | (m_vu0.isActive() ? 1u : 0u);
    if (interlocked || !m_vu0.isActive())
    {
        m_vu0CycleTick = eeCycleTick;
    }
    else
    {
        m_vu0CycleTick = ps2x::timing::saturatingAdd(
            m_vu0CycleTick,
            ps2x::timing::vuCyclesToEeTicks(cycleBudget));
    }
    if (traceSync)
    {
        const VU1State &state = m_vu0.state();
        traceEntry.cycleBudget = cycleBudget;
        traceEntry.eventDue = eventDue;
        traceEntry.vuPcAfter = state.pc;
        traceEntry.vuVi1After = static_cast<uint16_t>(state.vi[1]);
        traceEntry.vuVi2After = static_cast<uint16_t>(state.vi[2]);
        traceEntry.activeAfter = m_vu0.isActive();
        debugRecordVu0Sync(std::move(traceEntry));
    }
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx)
{
    handleSyscall(rdram, ctx, 0);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx, uint32_t encodedSyscallId)
{
    if (ctx->in_delay_slot)
    {
        throw std::runtime_error("Attempted to execute a syscall inside a branch delay slot! "
                                 "This breaks the atomic basic block model and is structurally unsupported by the emulator.");
    }

    const uint32_t syscallId = (encodedSyscallId != 0u)
                                   ? encodedSyscallId
                                   : getRegU32(ctx, 3); // $v1 / $3 is the EE kernel syscall number

    if (ps2_syscalls::dispatchNumericSyscall(syscallId, rdram, ctx, this))
    {
        return;
    }

    // God help you
    ps2_syscalls::TODO(rdram, ctx, this, encodedSyscallId);
}

void PS2Runtime::handleBreak(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_BREAKPOINT);
}

void PS2Runtime::drainCompletedDmacHandlers(uint8_t *rdram)
{
    // Older generated output and direct HLE test calls may reach the outer
    // safe boundary without executing a generated event-test helper. Publish
    // their current-tick completion here, but never invoke a handler from the
    // device work call itself.
    if (m_dmacCompletionReady.load(
            std::memory_order_acquire))
    {
        (void)m_eeEventScheduler.cancel(
            ps2x::timing::EeEventSource::
                DmacCompletion);
        publishCompatibilityDmacCompletions();
    }
    collectPendingDmacInterrupts();

    std::vector<DmacPendingInterrupt> retained;
    retained.reserve(m_pendingDmacInterrupts.size());
    for (const DmacPendingInterrupt &interrupt :
         m_pendingDmacInterrupts)
    {
        if (!m_memory.dmacInterruptStatusLatched(
                interrupt.source))
        {
            // The guest acknowledged the level before handler delivery.
            continue;
        }
        if (!m_memory.dmacInterruptUnmasked(
                interrupt.source) ||
            !ps2_syscalls::isDmacCauseEnabled(
                interrupt.cause))
        {
            retained.push_back(interrupt);
            continue;
        }
        ps2_syscalls::dispatchDmacHandlersForCause(
            rdram, this, interrupt.cause);
    }
    m_pendingDmacInterrupts = std::move(retained);
}

void PS2Runtime::handleTrap(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_TRAP);
}

void PS2Runtime::handleTLBR(uint8_t *rdram, R5900Context *ctx)
{
    uint32_t vpn = 0;
    uint32_t pfn = 0;
    uint32_t mask = 0;
    bool valid = false;

    const uint32_t index = ctx->cop0_index & 0x3Fu;
    if (!m_memory.tlbRead(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Preserve low ASID bits in EntryHi.
    ctx->cop0_entryhi = (ctx->cop0_entryhi & 0x00000FFFu) | (vpn & 0xFFFFF000u);
    ctx->cop0_entrylo0 = (ctx->cop0_entrylo0 & ~0x03FFFFC2u) |
                         ((pfn & 0x000FFFFFu) << 6) |
                         (valid ? 0x2u : 0u);
    ctx->cop0_pagemask = mask & 0x01FFE000u;
}

void PS2Runtime::handleTLBWI(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t index = ctx->cop0_index & 0x3Fu;
    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
    }
}

void PS2Runtime::handleTLBWR(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t entryCount = static_cast<uint32_t>(m_memory.tlbEntryCount());
    if (entryCount == 0)
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    const uint32_t wired = std::min(ctx->cop0_wired, entryCount - 1);
    uint32_t random = ctx->cop0_random % entryCount;
    if (random < wired)
    {
        random = wired;
    }

    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(random, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Keep COP0 bookkeeping in sync with the selected slot.
    ctx->cop0_index = (ctx->cop0_index & ~0x3Fu) | (random & 0x3Fu);
    ctx->cop0_random = (random <= wired) ? (entryCount - 1) : (random - 1);
}

void PS2Runtime::handleTLBP(uint8_t *rdram, R5900Context *ctx)
{
    const int32_t index = m_memory.tlbProbe(ctx->cop0_entryhi & 0xFFFFF000u);
    if (index >= 0)
    {
        ctx->cop0_index = (ctx->cop0_index & ~0x8000003Fu) |
                          (static_cast<uint32_t>(index) & 0x3Fu);
    }
    else
    {
        // MIPS sets probe failure bit (P) in Index[31].
        ctx->cop0_index |= 0x80000000u;
    }
}

void PS2Runtime::clearLLBit(R5900Context *ctx)
{
    // LL/SC reservation is tracked separately from COP0 Status.
    ctx->llbit = 0;
    ctx->lladdr = 0;
}

uint32_t PS2Runtime::alignGuestHeapValue(uint32_t value, uint32_t alignment)
{
    if (alignment == 0)
    {
        return value;
    }

    const uint32_t mask = alignment - 1u;
    if (value > (std::numeric_limits<uint32_t>::max() - mask))
    {
        return std::numeric_limits<uint32_t>::max();
    }
    return (value + mask) & ~mask;
}

bool PS2Runtime::isGuestHeapAlignmentValid(uint32_t alignment)
{
    return alignment != 0u && (alignment & (alignment - 1u)) == 0u;
}

uint32_t PS2Runtime::normalizeGuestHeapAlignment(uint32_t alignment)
{
    if (!isGuestHeapAlignmentValid(alignment))
    {
        return kGuestHeapDefaultAlignment;
    }
    return std::max(alignment, kGuestHeapDefaultAlignment);
}

uint32_t PS2Runtime::clampGuestHeapBase(uint32_t guestBase) const
{
    uint32_t normalized = guestBase;
    if (normalized >= PS2_RAM_SIZE)
    {
        normalized &= PS2_RAM_MASK;
    }
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    return std::min(normalized, hardLimit);
}

uint32_t PS2Runtime::clampGuestHeapLimit(uint32_t guestLimit) const
{
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    if (guestLimit == 0u || guestLimit > hardLimit)
    {
        return hardLimit;
    }
    return guestLimit;
}

void PS2Runtime::resetGuestHeapLocked(uint32_t guestBase, uint32_t guestLimit)
{
    uint32_t base = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    uint32_t limit = clampGuestHeapLimit(guestLimit);
    if (base == 0u)
    {
        const uint32_t fallbackBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
        base = alignGuestHeapValue(clampGuestHeapBase(fallbackBase), kGuestHeapDefaultAlignment);
    }

    if (limit <= base)
    {
        base = alignGuestHeapValue(clampGuestHeapBase(m_guestHeapSuggestedBase), kGuestHeapDefaultAlignment);
        limit = clampGuestHeapLimit(0u);
    }

    if (limit <= base)
    {
        base = 0u;
        limit = 0u;
    }

    m_guestHeapBlocks.clear();
    if (limit > base)
    {
        m_guestHeapBlocks.push_back({base, limit - base, true});
    }

    m_guestHeapBase = base;
    m_guestHeapEnd = base;
    m_guestHeapLimit = limit;
    m_guestHeapConfigured = true;
}

void PS2Runtime::ensureGuestHeapInitializedLocked()
{
    if (m_guestHeapConfigured)
    {
        return;
    }

    const uint32_t suggested = (m_guestHeapSuggestedBase == 0u) ? kGuestHeapDefaultBase : m_guestHeapSuggestedBase;
    resetGuestHeapLocked(suggested, clampGuestHeapLimit(0u));
}

int32_t PS2Runtime::findGuestHeapBlockIndexLocked(uint32_t guestAddr) const
{
    const uint32_t normalizedAddr = guestAddr & PS2_RAM_MASK;
    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock &block = m_guestHeapBlocks[i];
        if (!block.free && block.addr == normalizedAddr)
        {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

uint32_t PS2Runtime::allocateGuestBlockLocked(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    if (size > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock block = m_guestHeapBlocks[i];
        if (!block.free)
        {
            continue;
        }

        const uint64_t blockStart = block.addr;
        const uint64_t blockEnd = blockStart + static_cast<uint64_t>(block.size);
        const uint32_t alignedAddr = alignGuestHeapValue(block.addr, normalizedAlignment);
        if (alignedAddr < block.addr)
        {
            continue;
        }

        const uint64_t alignedStart = alignedAddr;
        if (alignedStart > blockEnd)
        {
            continue;
        }

        const uint64_t allocEnd = alignedStart + static_cast<uint64_t>(allocSize);
        if (allocEnd > blockEnd)
        {
            continue;
        }

        const uint32_t prefixSize = static_cast<uint32_t>(alignedStart - blockStart);
        const uint32_t suffixSize = static_cast<uint32_t>(blockEnd - allocEnd);

        std::vector<GuestHeapBlock> replacement;
        replacement.reserve(3);
        if (prefixSize > 0u)
        {
            replacement.push_back({block.addr, prefixSize, true});
        }
        replacement.push_back({alignedAddr, allocSize, false});
        if (suffixSize > 0u)
        {
            replacement.push_back({static_cast<uint32_t>(allocEnd), suffixSize, true});
        }

        m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
        m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i),
                                 replacement.begin(),
                                 replacement.end());

        m_guestHeapEnd = std::max(m_guestHeapEnd, static_cast<uint32_t>(allocEnd));
        return alignedAddr;
    }

    return 0u;
}

void PS2Runtime::coalesceGuestHeapLocked()
{
    if (m_guestHeapBlocks.empty())
    {
        return;
    }

    size_t i = 1;
    while (i < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &prev = m_guestHeapBlocks[i - 1];
        GuestHeapBlock &curr = m_guestHeapBlocks[i];
        const uint64_t prevEnd = static_cast<uint64_t>(prev.addr) + static_cast<uint64_t>(prev.size);
        if (prev.free && curr.free && prevEnd == curr.addr)
        {
            prev.size += curr.size;
            m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        ++i;
    }
}

void PS2Runtime::freeGuestBlockLocked(uint32_t guestAddr)
{
    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return;
    }

    m_guestHeapBlocks[static_cast<size_t>(index)].free = true;
    coalesceGuestHeapLocked();
}

void PS2Runtime::configureGuestHeap(uint32_t guestBase, uint32_t guestLimit)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    uint32_t normalizedBase = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    if (normalizedBase == 0u)
    {
        normalizedBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
    }
    m_guestHeapSuggestedBase = normalizedBase;
    resetGuestHeapLocked(normalizedBase, guestLimit);
}

uint32_t PS2Runtime::guestMalloc(uint32_t size, uint32_t alignment)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    return allocateGuestBlockLocked(size, alignment);
}

uint32_t PS2Runtime::guestCalloc(uint32_t count, uint32_t size, uint32_t alignment)
{
    if (count == 0u || size == 0u)
    {
        return 0u;
    }
    if (count > (std::numeric_limits<uint32_t>::max() / size))
    {
        return 0u;
    }

    const uint32_t totalSize = count * size;
    const uint32_t guestAddr = guestMalloc(totalSize, alignment);
    if (guestAddr != 0u)
    {
        uint8_t *rdram = m_memory.getRDRAM();
        if (rdram)
        {
            uint32_t physAddr = guestAddr & PS2_RAM_MASK;
            if (physAddr + totalSize <= PS2_RAM_SIZE)
                std::memset(rdram + physAddr, 0, totalSize);
        }
    }

    return guestAddr;
}

uint32_t PS2Runtime::guestRealloc(uint32_t guestAddr, uint32_t newSize, uint32_t alignment)
{
    if (guestAddr == 0u)
    {
        return guestMalloc(newSize, alignment);
    }
    if (newSize == 0u)
    {
        guestFree(guestAddr);
        return 0u;
    }

    if (newSize > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t requestedSize = alignGuestHeapValue(newSize, kGuestHeapDefaultAlignment);

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();

    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return 0u;
    }

    const size_t blockIndex = static_cast<size_t>(index);
    const uint32_t oldAddr = m_guestHeapBlocks[blockIndex].addr;
    const uint32_t oldSize = m_guestHeapBlocks[blockIndex].size;

    if (requestedSize <= oldSize)
    {
        if (requestedSize < oldSize)
        {
            const uint32_t tailAddr = oldAddr + requestedSize;
            const uint32_t tailSize = oldSize - requestedSize;
            m_guestHeapBlocks[blockIndex].size = requestedSize;
            m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u),
                                     GuestHeapBlock{tailAddr, tailSize, true});
            coalesceGuestHeapLocked();
        }
        return oldAddr;
    }

    if (blockIndex + 1u < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &next = m_guestHeapBlocks[blockIndex + 1u];
        const uint64_t blockEnd = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].addr) +
                                  static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size);
        if (next.free && blockEnd == next.addr)
        {
            const uint64_t combined = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size) +
                                      static_cast<uint64_t>(next.size);
            if (combined >= requestedSize)
            {
                const uint32_t extraNeeded = requestedSize - m_guestHeapBlocks[blockIndex].size;
                m_guestHeapBlocks[blockIndex].size = requestedSize;
                if (next.size == extraNeeded)
                {
                    m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u));
                }
                else
                {
                    next.addr += extraNeeded;
                    next.size -= extraNeeded;
                }
                m_guestHeapEnd = std::max(m_guestHeapEnd, oldAddr + requestedSize);
                return oldAddr;
            }
        }
    }

    const uint32_t newAddr = allocateGuestBlockLocked(newSize, normalizedAlignment);
    if (newAddr == 0u)
    {
        return 0u;
    }

    uint8_t *rdram = m_memory.getRDRAM();
    if (rdram)
    {
        const uint32_t copyBytes = std::min(oldSize, newSize);
        uint32_t dstPhys = newAddr & PS2_RAM_MASK;
        uint32_t srcPhys = oldAddr & PS2_RAM_MASK;
        if (dstPhys + copyBytes <= PS2_RAM_SIZE && srcPhys + copyBytes <= PS2_RAM_SIZE)
            std::memmove(rdram + dstPhys, rdram + srcPhys, copyBytes);
    }

    freeGuestBlockLocked(oldAddr);
    return newAddr;
}

void PS2Runtime::guestFree(uint32_t guestAddr)
{
    if (guestAddr == 0u)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    freeGuestBlockLocked(guestAddr);
}

uint32_t PS2Runtime::guestHeapBase() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapBase : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapEnd() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapEnd : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapLimit() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapLimit : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::reserveAsyncCallbackStack(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
    uint32_t top = m_asyncCallbackStackTop;
    if (top > PS2_RAM_SIZE)
    {
        top = PS2_RAM_SIZE;
    }
    top &= ~(kGuestHeapDefaultAlignment - 1u);

    if (top <= allocSize)
    {
        return 0u;
    }

    uint32_t base = top - allocSize;
    base &= ~(normalizedAlignment - 1u);
    if (base < m_asyncCallbackStackFloor || base >= top)
    {
        return 0u;
    }

    m_asyncCallbackStackTop = base;
    return top - 0x10u;
}

void PS2Runtime::executeGuestStep(uint8_t *rdram,
                                  R5900Context *ctx,
                                  RecompiledFunction function)
{
    struct CodeInvalidationFlush
    {
        PS2Memory &memory;
        ~CodeInvalidationFlush()
        {
            try
            {
                memory.flushCodeInvalidationLog();
            }
            catch (...)
            {
            }
        }
    } flush{m_memory};

    debugBeforeGuestStep(ctx);
    if (isStopRequested())
    {
        return;
    }

    try
    {
        function(rdram, ctx, this);
    }
    catch (const PS2GuestException &)
    {
    }
    catch (...)
    {
        debugAfterGuestStep(ctx);
        throw;
    }
    debugAfterGuestStep(ctx);
}

void PS2Runtime::dispatchLoop(uint8_t *rdram, R5900Context *ctx)
{
    uint32_t lastPc = std::numeric_limits<uint32_t>::max();
    uint32_t samePcCount = 0;
    constexpr uint32_t kSamePcYieldInterval = 0x4000u;

    while (!isStopRequested())
    {
        const uint32_t pc = normalizeGuestFunctionAddress(ctx->pc);
        ctx->pc = pc;

        if (pc == lastPc)
        {
            ++samePcCount;
            if ((samePcCount % kSamePcYieldInterval) == 0u)
            {
                PS2_IF_AGRESSIVE_LOGS({
                    RUNTIME_LOG("CPU is doing some work at PC " << formatGuestPc(pc) << ". PC not updating.");
                });
                std::this_thread::yield();
            }
        }
        else
        {
            samePcCount = 0;
            lastPc = pc;
        }

        m_debugPc.store(pc, std::memory_order_relaxed);
        m_debugRa.store(static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)), std::memory_order_relaxed);
        m_debugSp.store(static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0)), std::memory_order_relaxed);
        m_debugGp.store(static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0)), std::memory_order_relaxed);

        RecompiledFunction fn = lookupFunction(pc);
        const uint32_t dispatchedPc = pc;
        const uint32_t dispatchedRa = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0));

        uint64_t handoffBaseline = 0u;
        {
            GuestExecutionScope guestExecution(this, ctx);
            executeGuestStep(rdram, ctx, fn);
            handoffBaseline = guestExecutionHandoffEpochSnapshot();
        }

        waitForGuestExecutionHandoff(handoffBaseline);

        if (ctx->pc == 0u)
        {
            const uint32_t ra = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0));
            const uint32_t sp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0));
            const uint32_t gp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0));
            PS2_IF_AGRESSIVE_LOGS({
                std::cerr << "[dispatch:pc-zero] from=" << formatGuestPc(dispatchedPc)
                          << " fromRa=" << formatGuestPc(dispatchedRa)
                          << " ra=" << formatGuestPc(ra)
                          << " sp=0x" << std::hex << sp
                          << " gp=0x" << gp
                          << " trace=" << formatDispatchHistory()
                          << std::dec << std::endl;
            });

            // PC=0 means this guest thread returned (usually via jr $ra with RA=0).
            // Do not request a global runtime stop here: other guest threads may still run.
            break;
        }
    }
}

void PS2Runtime::switchGuestExecutionContext(
    R5900Context *context) noexcept
{
    if (!context || context == m_boundEeContext)
    {
        return;
    }

    (void)finishEeContextBlock(m_boundEeContext);
    context->resetEeBlockTiming();
    m_boundEeContext = context;
}

void PS2Runtime::restoreGuestExecutionContext(
    R5900Context *context,
    R5900Context *previousContext) noexcept
{
    if (!context ||
        context == previousContext ||
        m_boundEeContext != context)
    {
        return;
    }

    (void)finishEeContextBlock(context);
    m_boundEeContext = previousContext;
}

R5900Context *PS2Runtime::enterGuestExecution(
    R5900Context *context)
{
    uint32_t &depth = g_guestExecutionDepths[this];

    if (depth != 0u)
    {
        m_guestExecutionMutex.lock();
        ++depth;
        R5900Context *const previousContext =
            m_boundEeContext;
        switchGuestExecutionContext(context);
        return previousContext;
    }

    if (!m_debugControlActive.load(std::memory_order_acquire))
    {
        m_guestExecutionWaiters.fetch_add(1u, std::memory_order_acq_rel);
        m_guestExecutionMutex.lock();
        m_guestExecutionWaiters.fetch_sub(1u, std::memory_order_acq_rel);
        depth = 1u;
        markGuestExecutionAcquired();
        R5900Context *const previousContext =
            m_boundEeContext;
        switchGuestExecutionContext(context);
        return previousContext;
    }

    for (;;)
    {
        {
            std::unique_lock<std::mutex> controlLock(m_debugControlMutex);
            m_debugControlCv.wait(controlLock, [this]()
                                  { return !m_debugPauseRequested.load(std::memory_order_acquire) ||
                                           isStopRequested(); });
        }

        m_guestExecutionWaiters.fetch_add(1u, std::memory_order_acq_rel);
        m_guestExecutionMutex.lock();
        m_guestExecutionWaiters.fetch_sub(1u, std::memory_order_acq_rel);

        // Close the race where a pause is requested after the condition check
        // but before this thread acquires the guest-execution mutex.
        if (!m_debugPauseRequested.load(std::memory_order_acquire) ||
            isStopRequested())
        {
            break;
        }
        m_guestExecutionMutex.unlock();
    }
    depth = 1u;
    markGuestExecutionAcquired();
    R5900Context *const previousContext =
        m_boundEeContext;
    switchGuestExecutionContext(context);
    return previousContext;
}

void PS2Runtime::leaveGuestExecution(
    R5900Context *context,
    R5900Context *previousContext)
{
    auto it = g_guestExecutionDepths.find(this);
    if (it == g_guestExecutionDepths.end() || it->second == 0u)
    {
        return;
    }

    restoreGuestExecutionContext(context, previousContext);

    // A completed DMA may not interrupt the guest from inside the instruction
    // or library call that started it. Deliver queued completions only when the
    // outermost recompiled invocation has returned, so guest code can publish
    // its post-submit state before its DMAC handler observes it.
    if (it->second == 1u)
    {
        drainCompletedDmacHandlers(m_memory.getRDRAM());
        m_guestExecutionPreemptionRequested.store(
            false, std::memory_order_release);
        it = g_guestExecutionDepths.find(this);
        if (it == g_guestExecutionDepths.end() || it->second == 0u)
        {
            return;
        }
    }

    --it->second;
    m_guestExecutionMutex.unlock();
    if (it->second == 0u)
    {
        g_guestExecutionDepths.erase(it);
    }
}

uint32_t PS2Runtime::releaseGuestExecution(
    R5900Context *&context)
{
    context = nullptr;
    auto it = g_guestExecutionDepths.find(this);
    if (it == g_guestExecutionDepths.end() || it->second == 0u)
    {
        return 0u;
    }

    context = m_boundEeContext;
    (void)finishEeContextBlock(m_boundEeContext);
    m_boundEeContext = nullptr;

    const uint32_t depth = it->second;
    for (uint32_t i = 0; i < depth; ++i)
    {
        m_guestExecutionMutex.unlock();
    }
    g_guestExecutionDepths.erase(it);
    return depth;
}

void PS2Runtime::reacquireGuestExecution(
    uint32_t depth,
    R5900Context *context)
{
    if (depth == 0u)
    {
        return;
    }

    uint32_t &heldDepth = g_guestExecutionDepths[this];
    uint32_t remaining = depth;

    if (heldDepth == 0u)
    {
        (void)enterGuestExecution(context);
        heldDepth = g_guestExecutionDepths[this];
        --remaining;
    }
    else
    {
        switchGuestExecutionContext(context);
    }

    for (uint32_t i = 0; i < remaining; ++i)
    {
        m_guestExecutionMutex.lock();
        ++heldDepth;
    }
}

void PS2Runtime::markGuestExecutionAcquired()
{
    {
        std::lock_guard<std::mutex> lock(m_guestExecutionHandoffMutex);
        m_guestExecutionHandoffEpoch.fetch_add(1u, std::memory_order_acq_rel);
    }
    m_guestExecutionHandoffCv.notify_all();
}

void PS2Runtime::waitForGuestExecutionHandoff()
{
    waitForGuestExecutionHandoff(guestExecutionHandoffEpochSnapshot());
}

void PS2Runtime::waitForGuestExecutionHandoff(uint64_t baselineEpoch)
{
    // Lock-free fast path
    if (m_guestExecutionWaiters.load(std::memory_order_acquire) == 0u)
    {
        return;
    }

    std::unique_lock<std::mutex> lock(m_guestExecutionHandoffMutex);

    if (m_guestExecutionWaiters.load(std::memory_order_acquire) == 0u)
    {
        return;
    }

    const bool handedOff = m_guestExecutionHandoffCv.wait_for(
        lock,
        std::chrono::milliseconds(2),
        [&]()
        {
            return m_guestExecutionWaiters.load(std::memory_order_acquire) == 0u ||
                   m_guestExecutionHandoffEpoch.load(std::memory_order_relaxed) != baselineEpoch ||
                   isStopRequested();
        });

    if (!handedOff)
    {
        m_guestExecutionHandoffTimeouts.fetch_add(1u, std::memory_order_relaxed);
    }
}

void PS2Runtime::debugRecordStopLocked(const char *reason, uint32_t pc)
{
    ++m_debugStopSequence;
    m_debugLastStop.completed = true;
    m_debugLastStop.reason = reason ? reason : "unknown";
    m_debugLastStop.pc = normalizeGuestFunctionAddress(pc);
    m_debugLastStop.sequence = m_debugStopSequence;
}

void PS2Runtime::debugRefreshControlActiveLocked()
{
    const bool active =
        m_debugPauseRequested.load(std::memory_order_relaxed) ||
        m_debugRunUntilActive ||
        m_debugStepActive ||
        !m_debugBreakpoints.empty() ||
        !m_debugWatchpoints.empty();
    m_debugControlActive.store(active, std::memory_order_release);
}

void PS2Runtime::debugWaitUntilResumed()
{
    std::unique_lock<std::mutex> lock(m_debugControlMutex);
    m_debugControlCv.wait(lock, [this]()
                          { return !m_debugPauseRequested.load(std::memory_order_acquire) ||
                                   isStopRequested(); });
}

void PS2Runtime::debugBlockGuestAtBoundary(R5900Context *ctx, const char *reason)
{
    const uint32_t pc = ctx ? ctx->pc : m_debugPc.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        m_debugPauseRequested.store(true, std::memory_order_release);
        m_debugRunUntilActive = false;
        m_debugStepActive = false;
        m_debugStepRemaining = 0u;
        debugRecordStopLocked(reason, pc);
        debugRefreshControlActiveLocked();
    }
    m_debugControlCv.notify_all();
    m_audioBackend.setDebuggerPaused(true);

    // executeGuestStep() is called while this host thread owns the recursive
    // guest-execution mutex. Release all levels so debugger snapshots can be
    // proven quiescent while the guest remains stopped.
    R5900Context *releasedContext = nullptr;
    const uint32_t depth =
        releaseGuestExecution(releasedContext);
    debugWaitUntilResumed();
    if (depth != 0u)
    {
        reacquireGuestExecution(depth, releasedContext);
    }
}

void PS2Runtime::debugBeforeGuestStep(R5900Context *ctx)
{
    if (!ctx)
    {
        return;
    }

    const uint32_t pc = normalizeGuestFunctionAddress(ctx->pc);
    ctx->pc = pc;
    m_debugPc.store(pc, std::memory_order_relaxed);
    m_debugRa.store(static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)),
                    std::memory_order_relaxed);
    m_debugSp.store(static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0)),
                    std::memory_order_relaxed);
    m_debugGp.store(static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0)),
                    std::memory_order_relaxed);
    m_debugEeInstructions.store(ctx->insn_count, std::memory_order_relaxed);
    m_debugDispatches.fetch_add(1u, std::memory_order_relaxed);

    if (!m_debugControlActive.load(std::memory_order_acquire))
    {
        return;
    }

    const char *stopReason = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        if (m_debugPauseRequested.load(std::memory_order_acquire))
        {
            stopReason = "pause";
        }
        else if (m_debugRunUntilActive && pc == m_debugRunUntilPc)
        {
            stopReason = "predicate";
        }
        else
        {
            const bool hasBreakpoint =
                std::binary_search(m_debugBreakpoints.begin(), m_debugBreakpoints.end(), pc);
            if (hasBreakpoint)
            {
                if (m_debugSkipBreakpoint && m_debugSkipBreakpointPc == pc)
                {
                    m_debugSkipBreakpoint = false;
                }
                else
                {
                    stopReason = "breakpoint";
                }
            }
            else if (m_debugSkipBreakpoint)
            {
                m_debugSkipBreakpoint = false;
            }
        }
    }

    if (stopReason)
    {
        debugBlockGuestAtBoundary(ctx, stopReason);
    }
}

void PS2Runtime::debugAfterGuestStep(R5900Context *ctx)
{
    if (ctx)
    {
        m_debugRa.store(
            static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0)),
            std::memory_order_relaxed);
        m_debugSp.store(
            static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0)),
            std::memory_order_relaxed);
        m_debugGp.store(
            static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0)),
            std::memory_order_relaxed);
        m_debugEeInstructions.store(
            ctx->insn_count, std::memory_order_relaxed);
    }

    if (!m_debugControlActive.load(std::memory_order_acquire))
    {
        return;
    }

    bool stopForStep = false;
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        if (m_debugStepActive)
        {
            if (m_debugStepRemaining > 0u)
            {
                --m_debugStepRemaining;
            }
            stopForStep = m_debugStepRemaining == 0u;
        }
    }

    if (stopForStep)
    {
        debugBlockGuestAtBoundary(ctx, "step");
    }
}

bool PS2Runtime::debugPause(std::chrono::milliseconds timeout)
{
    m_debugControlActive.store(true, std::memory_order_release);
    bool newlyRequested = false;
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        if (!m_debugPauseRequested.load(std::memory_order_acquire))
        {
            newlyRequested = true;
            m_debugPauseRequested.store(true, std::memory_order_release);
            m_debugRunUntilActive = false;
            m_debugStepActive = false;
            m_debugStepRemaining = 0u;
            debugRecordStopLocked("pause", m_debugPc.load(std::memory_order_relaxed));
            debugRefreshControlActiveLocked();
        }
    }
    m_debugControlCv.notify_all();
    m_audioBackend.setDebuggerPaused(true);

    const bool locked = m_guestExecutionMutex.try_lock_for(timeout);
    if (locked)
    {
        m_guestExecutionMutex.unlock();
        return true;
    }

    if (newlyRequested)
    {
        {
            std::lock_guard<std::mutex> lock(m_debugControlMutex);
            m_debugPauseRequested.store(false, std::memory_order_release);
            debugRefreshControlActiveLocked();
        }
        m_debugControlCv.notify_all();
        m_audioBackend.setDebuggerPaused(false);
    }
    return false;
}

void PS2Runtime::debugResume()
{
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        if (m_debugLastStop.reason == "breakpoint")
        {
            m_debugSkipBreakpoint = true;
            m_debugSkipBreakpointPc = m_debugLastStop.pc;
        }
        m_debugRunUntilActive = false;
        m_debugStepActive = false;
        m_debugStepRemaining = 0u;
        m_debugPauseRequested.store(false, std::memory_order_release);
        debugRefreshControlActiveLocked();
    }
    m_debugControlCv.notify_all();
    m_audioBackend.setDebuggerPaused(false);
}

bool PS2Runtime::debugIsPaused() const
{
    return m_debugPauseRequested.load(std::memory_order_acquire);
}

PS2Runtime::DebugStopInfo PS2Runtime::debugRunUntilPc(
    uint32_t pc, std::chrono::milliseconds timeout)
{
    m_debugControlActive.store(true, std::memory_order_release);
    pc = normalizeGuestFunctionAddress(pc);
    if (!debugPause(timeout))
    {
        return {false, "pause-timeout", m_debugPc.load(std::memory_order_relaxed), 0u};
    }

    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        if (normalizeGuestFunctionAddress(m_debugPc.load(std::memory_order_relaxed)) == pc)
        {
            debugRecordStopLocked("already-matched", pc);
            return m_debugLastStop;
        }
    }

    uint64_t baseline = 0u;
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        baseline = m_debugStopSequence;
        m_debugRunUntilActive = true;
        m_debugRunUntilPc = pc;
        m_debugStepActive = false;
        m_debugPauseRequested.store(false, std::memory_order_release);
        debugRefreshControlActiveLocked();
    }
    m_debugControlCv.notify_all();
    m_audioBackend.setDebuggerPaused(false);

    {
        std::unique_lock<std::mutex> lock(m_debugControlMutex);
        if (m_debugControlCv.wait_for(lock, timeout, [this, baseline]()
                                      { return (m_debugStopSequence > baseline &&
                                                m_debugPauseRequested.load(std::memory_order_acquire)) ||
                                               isStopRequested(); }))
        {
            if (m_debugStopSequence > baseline)
            {
                return m_debugLastStop;
            }
            return {false, "stopped", m_debugPc.load(std::memory_order_relaxed),
                    m_debugStopSequence};
        }
        m_debugRunUntilActive = false;
    }

    (void)debugPause(timeout);
    return {false, "timeout", m_debugPc.load(std::memory_order_relaxed),
            m_debugStopSequence};
}

PS2Runtime::DebugStopInfo PS2Runtime::debugStepDispatches(
    uint64_t count, std::chrono::milliseconds timeout)
{
    m_debugControlActive.store(true, std::memory_order_release);
    if (count == 0u)
    {
        return {false, "invalid-count", m_debugPc.load(std::memory_order_relaxed), 0u};
    }
    if (!debugPause(timeout))
    {
        return {false, "pause-timeout", m_debugPc.load(std::memory_order_relaxed), 0u};
    }

    uint64_t baseline = 0u;
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        baseline = m_debugStopSequence;
        m_debugRunUntilActive = false;
        m_debugStepActive = true;
        m_debugStepRemaining = count;
        m_debugPauseRequested.store(false, std::memory_order_release);
        debugRefreshControlActiveLocked();
    }
    m_debugControlCv.notify_all();
    m_audioBackend.setDebuggerPaused(false);

    {
        std::unique_lock<std::mutex> lock(m_debugControlMutex);
        if (m_debugControlCv.wait_for(lock, timeout, [this, baseline]()
                                      { return (m_debugStopSequence > baseline &&
                                                m_debugPauseRequested.load(std::memory_order_acquire)) ||
                                               isStopRequested(); }))
        {
            if (m_debugStopSequence > baseline)
            {
                return m_debugLastStop;
            }
            return {false, "stopped", m_debugPc.load(std::memory_order_relaxed),
                    m_debugStopSequence};
        }
        m_debugStepActive = false;
        m_debugStepRemaining = 0u;
    }

    (void)debugPause(timeout);
    return {false, "timeout", m_debugPc.load(std::memory_order_relaxed),
            m_debugStopSequence};
}

R5900Context PS2Runtime::debugCpuSnapshot()
{
    std::lock_guard<std::recursive_timed_mutex> lock(m_guestExecutionMutex);
    return m_cpuContext;
}

PS2Runtime::DebugEeTiming PS2Runtime::debugEeTimingSnapshot()
{
    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    const R5900Context *const context =
        m_boundEeContext ? m_boundEeContext : &m_cpuContext;
    const uint64_t currentTick = currentEeTick().raw();

    DebugEeTiming result{};
    result.currentTick = currentTick;
    result.currentCycle =
        ps2x::timing::eeTickToCyclesFloor(
            currentEeTick());
    result.localBlockTicks = context->ee_block_cycle_ticks;
    result.localBlockActive = context->ee_block_cycle_active;
    result.contextBound = m_boundEeContext != nullptr;
    return result;
}

PS2Runtime::DebugEeScheduler
PS2Runtime::debugEeSchedulerSnapshot()
{
    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    DebugEeScheduler result{};
    result.mode = m_eeSchedulingMode;
    result.currentTick = currentEeTick().raw();
    const auto next = m_eeEventScheduler.nextDeadline();
    if (next.has_value())
    {
        result.hasNextDeadline = true;
        result.nextDeadlineTick = next->raw();
    }
    result.shadowMismatch = m_eeSchedulerShadowMismatch;
    result.shadowMismatchTick =
        m_eeSchedulerShadowMismatchTick.raw();
    result.legacyDeadlineTick =
        m_eeSchedulerShadowLegacyDeadline.raw();
    result.schedulerDeadlineTick =
        m_eeSchedulerShadowDeadline.raw();
    result.legacyPending =
        m_eeSchedulerShadowLegacyPending;
    result.schedulerPending =
        m_eeSchedulerShadowPending;
    result.statistics = m_eeEventScheduler.statistics();

    for (size_t index = 0u;
         index < result.slots.size(); ++index)
    {
        DebugEeEventSlot &target = result.slots[index];
        target.source =
            static_cast<ps2x::timing::EeEventSource>(index);
        target.device =
            debugEeEventDeviceState(target.source);
        const auto source = m_eeEventScheduler.event(
            target.source);
        if (!source.has_value())
        {
            continue;
        }
        target.pending = true;
        target.deadlineTick = source->deadline.raw();
        target.generation = source->token.generation;
        target.sequence = source->sequence;
    }
    return result;
}

PS2Runtime::DebugVu1Timing
PS2Runtime::debugVu1TimingSnapshot()
{
    std::lock_guard<std::recursive_timed_mutex> lock(
        m_guestExecutionMutex);
    DebugVu1Timing result{};
    result.currentTick = currentEeTick().raw();
    result.generation = m_vu1ExecutionTiming.generation;
    result.lastAdvancedTick =
        m_vu1ExecutionTiming.lastAdvancedTick.raw();
    result.totalAdvancedCycles =
        m_vu1ExecutionTiming.totalAdvancedCycles;
    result.pc = m_vu1.state().pc;
    result.active = m_vu1.isActive();
    result.vifWaitingForVu =
        m_memory.vif1WaitingForVu();
    const auto event = m_eeEventScheduler.event(
        ps2x::timing::EeEventSource::VifVu1Finish);
    if (event.has_value())
    {
        result.eventPending = true;
        result.eventGeneration =
            event->token.generation;
        result.eventDeadlineTick =
            event->deadline.raw();
    }
    return result;
}

bool PS2Runtime::debugReadRdram(uint32_t address,
                                uint32_t size,
                                std::vector<uint8_t> &output)
{
    uint32_t offset = 0u;
    if (!ps2ResolveDirectRdramOffset(address, offset) ||
        size > PS2_RAM_SIZE ||
        offset > (PS2_RAM_SIZE - size) ||
        !m_memory.getRDRAM())
    {
        return false;
    }

    std::lock_guard<std::recursive_timed_mutex> lock(m_guestExecutionMutex);
    output.assign(m_memory.getRDRAM() + offset,
                  m_memory.getRDRAM() + offset + size);
    return true;
}

bool PS2Runtime::debugReadMemory(uint32_t address,
                                 uint32_t size,
                                 std::vector<uint8_t> &output)
{
    uint8_t *base = nullptr;
    uint32_t offset = 0u;
    uint32_t limit = 0u;
    if (ps2ResolveDirectRdramOffset(address, offset))
    {
        base = m_memory.getRDRAM();
        limit = PS2_RAM_SIZE;
    }
    else if (ps2IsScratchpadAddress(address))
    {
        base = m_memory.getScratchpad();
        offset = ps2ScratchpadOffset(address);
        limit = PS2_SCRATCHPAD_SIZE;
    }

    if (!base || size > limit || offset > (limit - size))
    {
        return false;
    }

    std::lock_guard<std::recursive_timed_mutex> lock(m_guestExecutionMutex);
    output.assign(base + offset, base + offset + size);
    return true;
}

bool PS2Runtime::debugCopyGsVram(std::vector<uint8_t> &output)
{
    std::lock_guard<std::recursive_timed_mutex> lock(m_guestExecutionMutex);
    if (!m_boundGSVram)
    {
        return false;
    }
    output.assign(m_boundGSVram, m_boundGSVram + PS2_GS_VRAM_SIZE);
    return true;
}

PS2Runtime::DebugRuntimeProgress PS2Runtime::debugRuntimeProgress() const
{
    DebugRuntimeProgress progress{};
    progress.dispatches = m_debugDispatches.load(std::memory_order_relaxed);
    progress.eeInstructions =
        m_debugEeInstructions.load(std::memory_order_relaxed);
    progress.pc = m_debugPc.load(std::memory_order_relaxed);
    progress.ra = m_debugRa.load(std::memory_order_relaxed);
    progress.sp = m_debugSp.load(std::memory_order_relaxed);
    progress.gp = m_debugGp.load(std::memory_order_relaxed);
    progress.guestExecutionWaiters =
        m_guestExecutionWaiters.load(std::memory_order_relaxed);
    progress.guestExecutionHandoffTimeouts =
        m_guestExecutionHandoffTimeouts.load(std::memory_order_relaxed);
    return progress;
}

void PS2Runtime::debugRecordBranch(uint32_t pc)
{
    const uint64_t sequence =
        m_debugBranchSequence.fetch_add(1u, std::memory_order_acq_rel) + 1u;
    DebugBranchSlot &slot =
        m_debugBranchHistory[(sequence - 1u) % kDebugBranchHistoryCapacity];
    slot.pc.store(normalizeGuestFunctionAddress(pc), std::memory_order_relaxed);
    slot.sequence.store(sequence, std::memory_order_release);
}

std::vector<PS2Runtime::DebugBranchEntry> PS2Runtime::debugBranchHistory(
    size_t maximumEntries) const
{
    maximumEntries =
        std::min(maximumEntries, kDebugBranchHistoryCapacity);
    const uint64_t last =
        m_debugBranchSequence.load(std::memory_order_acquire);
    const uint64_t count =
        std::min<uint64_t>(last, static_cast<uint64_t>(maximumEntries));
    const uint64_t first = last - count + 1u;

    std::vector<DebugBranchEntry> entries;
    entries.reserve(static_cast<size_t>(count));
    for (uint64_t sequence = first; sequence <= last && count != 0u;
         ++sequence)
    {
        const DebugBranchSlot &slot =
            m_debugBranchHistory[(sequence - 1u) %
                                 kDebugBranchHistoryCapacity];
        const uint64_t observed =
            slot.sequence.load(std::memory_order_acquire);
        if (observed != sequence)
        {
            continue;
        }
        entries.push_back(
            {sequence, slot.pc.load(std::memory_order_relaxed)});
    }
    return entries;
}

void PS2Runtime::debugRecordFault(const DebugFaultInfo &fault)
{
    DebugFaultInfo copy = fault;
    copy.active = true;
    copy.sequence =
        m_debugFaultSequence.fetch_add(1u, std::memory_order_acq_rel) + 1u;
    std::lock_guard<std::mutex> lock(m_debugFaultMutex);
    m_debugFault = std::move(copy);
}

PS2Runtime::DebugFaultInfo PS2Runtime::debugFaultSnapshot() const
{
    std::lock_guard<std::mutex> lock(m_debugFaultMutex);
    return m_debugFault;
}

void PS2Runtime::debugClearFault()
{
    std::lock_guard<std::mutex> lock(m_debugFaultMutex);
    m_debugFault = {};
}

void PS2Runtime::debugStartVu0SyncTrace(
    size_t maximumEntries, std::optional<uint32_t> triggerEePc,
    bool stopOnFull)
{
    maximumEntries = std::clamp<size_t>(maximumEntries, 1u, 16384u);
    std::lock_guard<std::mutex> lock(m_debugVu0SyncTraceMutex);
    m_debugVu0SyncTraceEnabled.store(false, std::memory_order_release);
    m_debugVu0SyncTrace.clear();
    m_debugVu0SyncTrace.reserve(maximumEntries);
    m_debugVu0SyncTraceCapacity = maximumEntries;
    m_debugVu0SyncTraceNext = 0u;
    m_debugVu0SyncTraceTotal = 0u;
    m_debugVu0SyncTraceStopOnFull = stopOnFull;
    m_debugVu0SyncTraceTriggerEePc.store(
        triggerEePc.value_or(0u), std::memory_order_relaxed);
    m_debugVu0SyncTraceHasTrigger.store(
        triggerEePc.has_value(), std::memory_order_relaxed);
    m_debugVu0SyncTraceTriggered.store(
        !triggerEePc.has_value(), std::memory_order_release);
    m_debugVu0SyncTraceEnabled.store(true, std::memory_order_release);
}

void PS2Runtime::debugRecordVu0Sync(DebugVu0SyncEntry entry)
{
    if (!m_debugVu0SyncTraceEnabled.load(std::memory_order_relaxed))
    {
        return;
    }

    std::lock_guard<std::mutex> lock(m_debugVu0SyncTraceMutex);
    if (!m_debugVu0SyncTraceEnabled.load(std::memory_order_relaxed) ||
        m_debugVu0SyncTraceCapacity == 0u)
    {
        return;
    }

    entry.sequence = ++m_debugVu0SyncTraceTotal;
    if (m_debugVu0SyncTrace.size() < m_debugVu0SyncTraceCapacity)
    {
        m_debugVu0SyncTrace.push_back(std::move(entry));
        if (m_debugVu0SyncTraceStopOnFull &&
            m_debugVu0SyncTrace.size() == m_debugVu0SyncTraceCapacity)
        {
            m_debugVu0SyncTraceEnabled.store(
                false, std::memory_order_release);
        }
        return;
    }

    m_debugVu0SyncTrace[m_debugVu0SyncTraceNext] = std::move(entry);
    m_debugVu0SyncTraceNext =
        (m_debugVu0SyncTraceNext + 1u) % m_debugVu0SyncTraceCapacity;
}

PS2Runtime::DebugVu0SyncTrace PS2Runtime::debugVu0SyncTraceSnapshot(bool stop)
{
    if (stop)
    {
        m_debugVu0SyncTraceEnabled.store(false, std::memory_order_release);
    }

    std::lock_guard<std::mutex> lock(m_debugVu0SyncTraceMutex);
    DebugVu0SyncTrace snapshot{};
    snapshot.enabled =
        m_debugVu0SyncTraceEnabled.load(std::memory_order_acquire);
    snapshot.triggered =
        m_debugVu0SyncTraceTriggered.load(std::memory_order_acquire);
    snapshot.stopOnFull = m_debugVu0SyncTraceStopOnFull;
    if (m_debugVu0SyncTraceHasTrigger.load(std::memory_order_relaxed))
    {
        snapshot.triggerEePc =
            m_debugVu0SyncTraceTriggerEePc.load(std::memory_order_relaxed);
    }
    snapshot.totalEntries = m_debugVu0SyncTraceTotal;
    snapshot.droppedEntries =
        m_debugVu0SyncTraceTotal > m_debugVu0SyncTrace.size()
            ? m_debugVu0SyncTraceTotal - m_debugVu0SyncTrace.size()
            : 0u;
    snapshot.entries.reserve(m_debugVu0SyncTrace.size());
    if (m_debugVu0SyncTrace.size() < m_debugVu0SyncTraceCapacity ||
        m_debugVu0SyncTrace.empty())
    {
        snapshot.entries = m_debugVu0SyncTrace;
        return snapshot;
    }

    for (size_t offset = 0u; offset < m_debugVu0SyncTrace.size(); ++offset)
    {
        snapshot.entries.push_back(
            m_debugVu0SyncTrace[
                (m_debugVu0SyncTraceNext + offset) %
                m_debugVu0SyncTrace.size()]);
    }
    return snapshot;
}

void PS2Runtime::debugStartEeEventTrace(
    size_t maximumEntries, bool stopOnFull)
{
    maximumEntries =
        std::clamp<size_t>(maximumEntries, 1u, 16384u);
    std::lock_guard<std::mutex> lock(
        m_debugEeEventTraceMutex);
    m_debugEeEventTraceEnabled.store(
        false, std::memory_order_release);
    m_debugEeEventTrace.clear();
    m_debugEeEventTrace.reserve(maximumEntries);
    m_debugEeEventTraceCapacity = maximumEntries;
    m_debugEeEventTraceNext = 0u;
    m_debugEeEventTraceTotal = 0u;
    m_debugEeEventTraceStopOnFull = stopOnFull;
    m_debugEeEventTraceEnabled.store(
        true, std::memory_order_release);
}

void PS2Runtime::debugRecordEeEvent(
    DebugEeEventEntry entry)
{
    if (!m_debugEeEventTraceEnabled.load(
            std::memory_order_relaxed))
    {
        return;
    }

    std::lock_guard<std::mutex> lock(
        m_debugEeEventTraceMutex);
    if (!m_debugEeEventTraceEnabled.load(
            std::memory_order_relaxed) ||
        m_debugEeEventTraceCapacity == 0u)
    {
        return;
    }

    entry.sequence = ++m_debugEeEventTraceTotal;
    if (m_debugEeEventTrace.size() <
        m_debugEeEventTraceCapacity)
    {
        m_debugEeEventTrace.push_back(std::move(entry));
        if (m_debugEeEventTraceStopOnFull &&
            m_debugEeEventTrace.size() ==
                m_debugEeEventTraceCapacity)
        {
            m_debugEeEventTraceEnabled.store(
                false, std::memory_order_release);
        }
        return;
    }

    m_debugEeEventTrace[m_debugEeEventTraceNext] =
        std::move(entry);
    m_debugEeEventTraceNext =
        (m_debugEeEventTraceNext + 1u) %
        m_debugEeEventTraceCapacity;
}

PS2Runtime::DebugEeEventTrace
PS2Runtime::debugEeEventTraceSnapshot(bool stop)
{
    if (stop)
    {
        m_debugEeEventTraceEnabled.store(
            false, std::memory_order_release);
    }

    std::lock_guard<std::mutex> lock(
        m_debugEeEventTraceMutex);
    DebugEeEventTrace snapshot{};
    snapshot.enabled = m_debugEeEventTraceEnabled.load(
        std::memory_order_acquire);
    snapshot.stopOnFull = m_debugEeEventTraceStopOnFull;
    snapshot.totalEntries = m_debugEeEventTraceTotal;
    snapshot.droppedEntries =
        m_debugEeEventTraceTotal >
                m_debugEeEventTrace.size()
            ? m_debugEeEventTraceTotal -
                  m_debugEeEventTrace.size()
            : 0u;
    snapshot.entries.reserve(m_debugEeEventTrace.size());
    if (m_debugEeEventTrace.size() <
            m_debugEeEventTraceCapacity ||
        m_debugEeEventTrace.empty())
    {
        snapshot.entries = m_debugEeEventTrace;
        return snapshot;
    }

    for (size_t offset = 0u;
         offset < m_debugEeEventTrace.size(); ++offset)
    {
        snapshot.entries.push_back(
            m_debugEeEventTrace[
                (m_debugEeEventTraceNext + offset) %
                m_debugEeEventTrace.size()]);
    }
    return snapshot;
}

void PS2Runtime::debugStartVu0InstructionTrace(
    size_t maximumEntries, std::optional<uint32_t> triggerEePc,
    bool stopOnFull)
{
    maximumEntries = std::clamp<size_t>(maximumEntries, 1u, 8192u);
    m_vu0.setInstructionObserverEnabled(false);
    std::lock_guard<std::mutex> lock(m_debugVu0InstructionTraceMutex);
    m_debugVu0InstructionTraceEnabled.store(
        false, std::memory_order_release);
    m_debugVu0InstructionTrace.clear();
    m_debugVu0InstructionTrace.reserve(maximumEntries);
    m_debugVu0InstructionTraceCapacity = maximumEntries;
    m_debugVu0InstructionTraceNext = 0u;
    m_debugVu0InstructionTraceTotal = 0u;
    m_debugVu0InstructionTraceStopOnFull = stopOnFull;
    m_debugVu0InstructionTraceTriggerEePc.store(
        triggerEePc.value_or(0u), std::memory_order_relaxed);
    m_debugVu0InstructionTraceHasTrigger.store(
        triggerEePc.has_value(), std::memory_order_relaxed);
    m_debugVu0InstructionTraceTriggered.store(
        !triggerEePc.has_value(), std::memory_order_release);
    m_debugVu0InstructionTraceEnabled.store(
        true, std::memory_order_release);
    m_vu0.setInstructionObserverEnabled(true);
}

void PS2Runtime::debugRecordVu0Instruction(
    DebugVu0InstructionEntry entry)
{
    if (!m_debugVu0InstructionTraceEnabled.load(
            std::memory_order_relaxed))
    {
        return;
    }

    std::lock_guard<std::mutex> lock(m_debugVu0InstructionTraceMutex);
    if (!m_debugVu0InstructionTraceEnabled.load(
            std::memory_order_relaxed) ||
        m_debugVu0InstructionTraceCapacity == 0u)
    {
        return;
    }

    entry.sequence = ++m_debugVu0InstructionTraceTotal;
    if (m_debugVu0InstructionTrace.size() <
        m_debugVu0InstructionTraceCapacity)
    {
        m_debugVu0InstructionTrace.push_back(std::move(entry));
        if (m_debugVu0InstructionTraceStopOnFull &&
            m_debugVu0InstructionTrace.size() ==
                m_debugVu0InstructionTraceCapacity)
        {
            m_debugVu0InstructionTraceEnabled.store(
                false, std::memory_order_release);
        }
        return;
    }

    m_debugVu0InstructionTrace[m_debugVu0InstructionTraceNext] =
        std::move(entry);
    m_debugVu0InstructionTraceNext =
        (m_debugVu0InstructionTraceNext + 1u) %
        m_debugVu0InstructionTraceCapacity;
}

PS2Runtime::DebugVu0InstructionTrace
PS2Runtime::debugVu0InstructionTraceSnapshot(bool stop)
{
    if (stop)
    {
        m_vu0.setInstructionObserverEnabled(false);
        m_debugVu0InstructionTraceEnabled.store(
            false, std::memory_order_release);
    }

    std::lock_guard<std::mutex> lock(m_debugVu0InstructionTraceMutex);
    DebugVu0InstructionTrace snapshot{};
    snapshot.enabled =
        m_debugVu0InstructionTraceEnabled.load(std::memory_order_acquire);
    snapshot.triggered =
        m_debugVu0InstructionTraceTriggered.load(
            std::memory_order_acquire);
    snapshot.stopOnFull = m_debugVu0InstructionTraceStopOnFull;
    if (m_debugVu0InstructionTraceHasTrigger.load(
            std::memory_order_relaxed))
    {
        snapshot.triggerEePc =
            m_debugVu0InstructionTraceTriggerEePc.load(
                std::memory_order_relaxed);
    }
    snapshot.totalEntries = m_debugVu0InstructionTraceTotal;
    snapshot.droppedEntries =
        m_debugVu0InstructionTraceTotal >
                m_debugVu0InstructionTrace.size()
            ? m_debugVu0InstructionTraceTotal -
                  m_debugVu0InstructionTrace.size()
            : 0u;
    snapshot.entries.reserve(m_debugVu0InstructionTrace.size());
    if (m_debugVu0InstructionTrace.size() <
            m_debugVu0InstructionTraceCapacity ||
        m_debugVu0InstructionTrace.empty())
    {
        snapshot.entries = m_debugVu0InstructionTrace;
        return snapshot;
    }

    for (size_t offset = 0u;
         offset < m_debugVu0InstructionTrace.size(); ++offset)
    {
        snapshot.entries.push_back(
            m_debugVu0InstructionTrace[
                (m_debugVu0InstructionTraceNext + offset) %
                m_debugVu0InstructionTrace.size()]);
    }
    return snapshot;
}

std::vector<uint32_t> PS2Runtime::debugBreakpoints() const
{
    std::lock_guard<std::mutex> lock(m_debugControlMutex);
    return m_debugBreakpoints;
}

void PS2Runtime::debugAddBreakpoint(uint32_t address)
{
    m_debugControlActive.store(true, std::memory_order_release);
    address = normalizeGuestFunctionAddress(address);
    std::lock_guard<std::mutex> lock(m_debugControlMutex);
    const auto position =
        std::lower_bound(m_debugBreakpoints.begin(), m_debugBreakpoints.end(), address);
    if (position == m_debugBreakpoints.end() || *position != address)
    {
        m_debugBreakpoints.insert(position, address);
    }
    debugRefreshControlActiveLocked();
}

void PS2Runtime::debugRemoveBreakpoint(uint32_t address)
{
    address = normalizeGuestFunctionAddress(address);
    std::lock_guard<std::mutex> lock(m_debugControlMutex);
    const auto position =
        std::lower_bound(m_debugBreakpoints.begin(), m_debugBreakpoints.end(), address);
    if (position != m_debugBreakpoints.end() && *position == address)
    {
        m_debugBreakpoints.erase(position);
    }
    debugRefreshControlActiveLocked();
}

std::vector<PS2Runtime::DebugWatchpoint> PS2Runtime::debugWatchpoints() const
{
    std::lock_guard<std::mutex> lock(m_debugControlMutex);
    return m_debugWatchpoints;
}

uint64_t PS2Runtime::debugAddWatchpoint(uint32_t start,
                                        uint32_t size,
                                        DebugMemoryAccess access)
{
    if (size == 0u || static_cast<uint64_t>(start) + size > 0x100000000ull)
    {
        return 0u;
    }

    m_debugControlActive.store(true, std::memory_order_release);
    uint32_t normalized = 0u;
    if (ps2ResolveDirectRdramOffset(start, normalized))
    {
        start = normalized;
    }

    std::lock_guard<std::mutex> lock(m_debugControlMutex);
    const uint64_t id = m_debugNextWatchpointId++;
    m_debugWatchpoints.push_back({id, start, size, access});
    m_debugWatchpointsActive.store(true, std::memory_order_release);
    debugRefreshControlActiveLocked();
    return id;
}

bool PS2Runtime::debugRemoveWatchpoint(uint64_t id)
{
    std::lock_guard<std::mutex> lock(m_debugControlMutex);
    const auto position = std::find_if(
        m_debugWatchpoints.begin(), m_debugWatchpoints.end(),
        [id](const DebugWatchpoint &watchpoint)
        { return watchpoint.id == id; });
    if (position == m_debugWatchpoints.end())
    {
        return false;
    }
    m_debugWatchpoints.erase(position);
    m_debugWatchpointsActive.store(!m_debugWatchpoints.empty(),
                                   std::memory_order_release);
    debugRefreshControlActiveLocked();
    return true;
}

bool PS2Runtime::debugWatchpointsEnabled() const
{
    return m_debugWatchpointsActive.load(std::memory_order_acquire);
}

void PS2Runtime::debugObserveMemoryAccess(uint32_t address,
                                          uint32_t size,
                                          DebugMemoryAccess access,
                                          const R5900Context *ctx)
{
    if (!debugWatchpointsEnabled() || size == 0u)
    {
        return;
    }

    uint32_t normalized = 0u;
    if (ps2ResolveDirectRdramOffset(address, normalized))
    {
        address = normalized;
    }

    bool matched = false;
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        const uint64_t accessBegin = address;
        const uint64_t accessEnd = accessBegin + size;
        const uint8_t requested = static_cast<uint8_t>(access);
        for (const DebugWatchpoint &watchpoint : m_debugWatchpoints)
        {
            const uint8_t watched = static_cast<uint8_t>(watchpoint.access);
            const uint64_t watchBegin = watchpoint.start;
            const uint64_t watchEnd = watchBegin + watchpoint.size;
            if ((requested & watched) != 0u &&
                accessBegin < watchEnd && watchBegin < accessEnd)
            {
                matched = true;
                break;
            }
        }
    }

    if (matched)
    {
        debugBlockGuestAtBoundary(const_cast<R5900Context *>(ctx), "watchpoint");
    }
}

PS2Runtime::DeferredGuestYieldScope::DeferredGuestYieldScope(bool &pendingOut) noexcept
    : m_pendingOut(pendingOut)
{
    ++g_deferredGuestYieldDepth;
}

PS2Runtime::DeferredGuestYieldScope::~DeferredGuestYieldScope()
{
    if (--g_deferredGuestYieldDepth == 0u && g_deferredGuestYieldPending)
    {
        g_deferredGuestYieldPending = false;
        m_pendingOut = true;
    }
}

void PS2Runtime::yieldGuestExecutionAfterWake()
{
    if (g_deferredGuestYieldDepth != 0u)
    {
        g_deferredGuestYieldPending = true;
        return;
    }

    auto it = g_guestExecutionDepths.find(this);
    if (it == g_guestExecutionDepths.end() || it->second == 0u)
    {
        std::this_thread::yield();
        return;
    }

    const uint64_t handoffEpoch = m_guestExecutionHandoffEpoch.load(std::memory_order_acquire);
    {
        GuestExecutionReleaseScope releaseGuestExecution(this);
        std::unique_lock<std::mutex> lock(m_guestExecutionHandoffMutex);
        m_guestExecutionHandoffCv.wait_for(lock, std::chrono::milliseconds(2), [&]()
                                           { return m_guestExecutionHandoffEpoch.load(std::memory_order_acquire) != handoffEpoch; });
    }
}

void PS2Runtime::requestGuestPreemption()
{
    m_guestExecutionPreemptionRequested.store(
        true, std::memory_order_release);
    m_guestExecutionPreemptionEpoch.fetch_add(
        1u, std::memory_order_release);
}

bool PS2Runtime::shouldPreemptGuestExecution()
{
    if (m_guestExecutionPreemptionRequested.exchange(
            false, std::memory_order_acq_rel))
    {
        return true;
    }

    thread_local uint32_t s_backEdgeYieldCounter = 0u;
    const uint32_t waiterCount = m_guestExecutionWaiters.load(std::memory_order_acquire);
    const uint32_t yieldInterval = (waiterCount != 0u) ? 64u : 100u;
    if (++s_backEdgeYieldCounter < yieldInterval)
    {
        return false;
    }

    s_backEdgeYieldCounter = 0u;
    if (waiterCount != 0u)
    {
        // Returning to the dispatcher is not itself a fair handoff: the current
        // host thread can release and immediately reacquire the recursive mutex,
        // indefinitely starving a guest thread that is already queued. Transfer
        // ownership here before asking generated code to return to its dispatcher.
        yieldGuestExecutionAfterWake();
    }
    m_guestExecutionPreemptionEpoch.fetch_add(1u, std::memory_order_release);
    return true;
}

uint8_t PS2Runtime::Load8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    debugObserveMemoryAccess(vaddr, 1u, DebugMemoryAccess::Read, ctx);
    try
    {
        return m_memory.read8(vaddr);
    }
    catch (const PS2TlbMissException &fault)
    {
        SignalMemoryException(ctx, EXCEPTION_TLB_REFILL_LOAD, fault.virtualAddress());
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD, vaddr);
    }
}

uint16_t PS2Runtime::Load16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    debugObserveMemoryAccess(vaddr, 2u, DebugMemoryAccess::Read, ctx);
    try
    {
        return m_memory.read16(vaddr);
    }
    catch (const PS2TlbMissException &fault)
    {
        SignalMemoryException(ctx, EXCEPTION_TLB_REFILL_LOAD, fault.virtualAddress());
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD, vaddr);
    }
}

uint32_t PS2Runtime::Load32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    debugObserveMemoryAccess(vaddr, 4u, DebugMemoryAccess::Read, ctx);
    try
    {
        return m_memory.read32(vaddr);
    }
    catch (const PS2TlbMissException &fault)
    {
        SignalMemoryException(ctx, EXCEPTION_TLB_REFILL_LOAD, fault.virtualAddress());
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD, vaddr);
    }
}

uint64_t PS2Runtime::Load64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    debugObserveMemoryAccess(vaddr, 8u, DebugMemoryAccess::Read, ctx);
    try
    {
        return m_memory.read64(vaddr);
    }
    catch (const PS2TlbMissException &fault)
    {
        SignalMemoryException(ctx, EXCEPTION_TLB_REFILL_LOAD, fault.virtualAddress());
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD, vaddr);
    }
}

__m128i PS2Runtime::Load128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    debugObserveMemoryAccess(vaddr, 16u, DebugMemoryAccess::Read, ctx);
    try
    {
        return m_memory.read128(vaddr);
    }
    catch (const PS2TlbMissException &fault)
    {
        SignalMemoryException(ctx, EXCEPTION_TLB_REFILL_LOAD, fault.virtualAddress());
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD, vaddr);
    }
}

void PS2Runtime::Store8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint8_t value)
{
    debugObserveMemoryAccess(vaddr, 1u, DebugMemoryAccess::Write, ctx);
    ps2TraceGuestWrite(rdram, vaddr, 1u, value, 0u, "WRITE8", ctx);
    traceVif0MmioWrite(m_memory, ctx, vaddr, 1u, value, 0u);
    try
    {
        m_memory.write8(vaddr, value, ctx ? ctx->pc : 0u);
    }
    catch (const PS2TlbMissException &fault)
    {
        SignalMemoryException(ctx, EXCEPTION_TLB_REFILL_STORE, fault.virtualAddress());
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_STORE, vaddr);
    }
}

void PS2Runtime::Store16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint16_t value)
{
    debugObserveMemoryAccess(vaddr, 2u, DebugMemoryAccess::Write, ctx);
    ps2TraceGuestWrite(rdram, vaddr, 2u, value, 0u, "WRITE16", ctx);
    traceVif0MmioWrite(m_memory, ctx, vaddr, 2u, value, 0u);
    try
    {
        m_memory.write16(vaddr, value, ctx ? ctx->pc : 0u);
    }
    catch (const PS2TlbMissException &fault)
    {
        SignalMemoryException(ctx, EXCEPTION_TLB_REFILL_STORE, fault.virtualAddress());
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_STORE, vaddr);
    }
}

void PS2Runtime::Store32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint32_t value)
{
    debugObserveMemoryAccess(vaddr, 4u, DebugMemoryAccess::Write, ctx);
    ps2TraceGuestWrite(rdram, vaddr, 4u, value, 0u, "WRITE32", ctx);
    traceVif0MmioWrite(m_memory, ctx, vaddr, 4u, value, 0u);
    try
    {
        m_memory.write32(vaddr, value, ctx ? ctx->pc : 0u);
    }
    catch (const PS2TlbMissException &fault)
    {
        SignalMemoryException(ctx, EXCEPTION_TLB_REFILL_STORE, fault.virtualAddress());
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_STORE, vaddr);
    }
}

void PS2Runtime::Store64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint64_t value)
{
    debugObserveMemoryAccess(vaddr, 8u, DebugMemoryAccess::Write, ctx);
    ps2TraceGuestWrite(rdram, vaddr, 8u, value, 0u, "WRITE64", ctx);
    traceVif0MmioWrite(m_memory, ctx, vaddr, 8u, value, 0u);
    try
    {
        m_memory.write64(vaddr, value, ctx ? ctx->pc : 0u);
    }
    catch (const PS2TlbMissException &fault)
    {
        SignalMemoryException(ctx, EXCEPTION_TLB_REFILL_STORE, fault.virtualAddress());
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_STORE, vaddr);
    }
}

void PS2Runtime::Store128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, __m128i value)
{
    debugObserveMemoryAccess(vaddr, 16u, DebugMemoryAccess::Write, ctx);
    alignas(16) uint64_t _parts[2];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(_parts), value);
    ps2TraceGuestWrite(rdram, vaddr, 16u, _parts[0], _parts[1], "WRITE128", ctx);
    traceVif0MmioWrite(m_memory, ctx, vaddr, 16u, _parts[0], _parts[1]);
    try
    {
        m_memory.write128(vaddr, value, ctx ? ctx->pc : 0u);
    }
    catch (const PS2TlbMissException &fault)
    {
        SignalMemoryException(ctx, EXCEPTION_TLB_REFILL_STORE, fault.virtualAddress());
    }
    catch (const std::exception &)
    {
        SignalMemoryException(ctx, EXCEPTION_ADDRESS_ERROR_STORE, vaddr);
    }
}

void PS2Runtime::kickGifDmaChainFromMMIO(uint8_t *rdram,
                                         R5900Context *ctx,
                                         uint32_t dPcrValue,
                                         uint32_t dStatValue,
                                         uint32_t tadr,
                                         uint32_t chcr)
{
    constexpr uint32_t D_PCR = 0x1000E020u;
    constexpr uint32_t D_STAT = 0x1000E010u;
    constexpr uint32_t GIF_TADR = 0x1000A030u;
    constexpr uint32_t GIF_CHCR = 0x1000A000u;

    ps2TraceGuestWrite(rdram, D_PCR, 4u, dPcrValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_PCR, dPcrValue);
    ps2TraceGuestWrite(rdram, D_STAT, 4u, dStatValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_STAT, dStatValue);
    ps2TraceGuestWrite(rdram, GIF_TADR, 4u, tadr, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(GIF_TADR, tadr);
    ps2TraceGuestWrite(rdram, GIF_CHCR, 4u, chcr, 0u, "WRITE32", ctx);
    if (m_memory.tryProcessNativeGifImageUploadChain(m_gs, tadr, chcr))
    {
        return;
    }
    if (m_memory.tryProcessNativeGifPackedChain(m_gs, tadr, chcr))
    {
        return;
    }
    m_memory.writeIORegister(GIF_CHCR, chcr);
    m_memory.processPendingTransfers();
}

void PS2Runtime::requestStop()
{
    m_stopRequested.store(true, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(m_debugControlMutex);
        m_debugRunUntilActive = false;
        m_debugStepActive = false;
        m_debugStepRemaining = 0u;
        m_debugPauseRequested.store(false, std::memory_order_release);
        debugRefreshControlActiveLocked();
    }
    m_debugControlCv.notify_all();
    m_audioBackend.setDebuggerPaused(false);
    ps2_syscalls::notifyRuntimeStop();
}

bool PS2Runtime::isStopRequested() const
{
    return m_stopRequested.load(std::memory_order_relaxed);
}

[[noreturn]] void PS2Runtime::HandleIntegerOverflow(R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_INTEGER_OVERFLOW);
}

void PS2Runtime::run()
{
    m_stopRequested.store(false, std::memory_order_relaxed);
    ps2_stubs::resetSifState();
    resetIop();
    ps2_stubs::resetAudioStubState();
    ps2_stubs::resetGsSyncVCallbackState();
    ps2_stubs::resetMpegStubState();
    ps2_syscalls::initializeGuestKernelState(m_memory.getRDRAM());
    m_cpuContext.r[4] = _mm_setzero_si128();
    m_cpuContext.r[5] = _mm_setzero_si128();
    m_cpuContext.r[29] = _mm_set_epi64x(0, static_cast<int64_t>(PS2_RAM_SIZE - 0x10u));
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);
    m_debugRa.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[31], 0)), std::memory_order_relaxed);
    m_debugSp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[29], 0)), std::memory_order_relaxed);
    m_debugGp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[28], 0)), std::memory_order_relaxed);

    const char *const gsHistoryDumpPath = std::getenv(GS_HISTORY_DUMP_ENV);
    if (gsHistoryDumpPath && gsHistoryDumpPath[0] != '\0')
    {
        m_gs.clearDebugHistory();
        m_gs.setDebugHistoryPaused(false);
    }

    RUNTIME_LOG("Starting execution at address 0x" << std::hex << m_cpuContext.pc << std::dec);

    // A blank image to use as a framebuffer
    Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, BLANK);
    Texture2D frameTex = LoadTextureFromImage(blank);
    UnloadImage(blank);

    g_activeThreads.store(1, std::memory_order_relaxed);
    std::atomic<bool> gameThreadFinished{false};

    std::thread gameThread([&]()
                           {
        ThreadNaming::SetCurrentThreadName("GameThread");
        try
        {
            dispatchLoop(m_memory.getRDRAM(), &m_cpuContext);
            uint32_t pc = m_debugPc.load(std::memory_order_relaxed);
            RUNTIME_LOG("Game thread returned. PC=0x" << std::hex << pc
                      << " RA=0x" << static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[31], 0)) << std::dec << std::endl);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error during program execution: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Error during program execution: unknown exception" << std::endl;
        }
        g_activeThreads.fetch_sub(1, std::memory_order_relaxed);
        gameThreadFinished.store(true, std::memory_order_release); });

    ps2_syscalls::EnsureVSyncWorkerRunning(m_memory.getRDRAM(), this);

    uint64_t tick = 0;
    while (!isStopRequested() && g_activeThreads.load(std::memory_order_relaxed) > 0)
    {
        PS2_IF_AGRESSIVE_LOGS({
            tick++;
            if ((tick % 120) == 0)
            {
                uint64_t curDma = m_memory.dmaStartCount();
                uint64_t curGif = m_memory.gifCopyCount();
                uint64_t curGs = m_memory.gsWriteCount();
                uint64_t curVif = m_memory.vifWriteCount();
                const GSRegisters &gs = m_memory.gs();
                const uint32_t dbgPc = m_debugPc.load(std::memory_order_relaxed);
                const uint32_t dbgRa = m_debugRa.load(std::memory_order_relaxed);
                const uint32_t dbgSp = m_debugSp.load(std::memory_order_relaxed);
                const uint32_t dbgGp = m_debugGp.load(std::memory_order_relaxed);
                const int activeThreads = g_activeThreads.load(std::memory_order_relaxed);

                RUNTIME_LOG("[run:tick] tick=" << tick
                                               << " pc=0x" << std::hex << dbgPc
                                               << " ra=0x" << dbgRa
                                               << " sp=0x" << dbgSp
                                               << " gp=0x" << dbgGp
                                               << " dispfb1=0x" << gs.dispfb1
                                               << " display1=0x" << gs.display1
                                               << std::dec
                                               << " activeThreads=" << activeThreads
                                               << " dma=" << curDma
                                               << " gif=" << curGif
                                               << " gsw=" << curGs
                                               << " vif=" << curVif
                                               << std::endl);
            }
        });
        uint32_t presentWidth = FB_WIDTH;
        uint32_t presentHeight = DEFAULT_DISPLAY_HEIGHT;
        UploadFrame(frameTex, this, presentWidth, presentHeight);

        BeginDrawing();
        ClearBackground(BLACK);
        const float srcWidth = static_cast<float>(std::max<uint32_t>(1u, presentWidth));
        const float srcHeight = static_cast<float>(std::max<uint32_t>(1u, presentHeight));
        const float screenWidth = static_cast<float>(GetScreenWidth());
        const float screenHeight = static_cast<float>(GetScreenHeight());
        const float scale = std::min(screenWidth / srcWidth, screenHeight / srcHeight);
        const float dstWidth = srcWidth * scale;
        const float dstHeight = srcHeight * scale;
        const Rectangle srcRect{0.0f, 0.0f, srcWidth, srcHeight};
        const Rectangle dstRect{
            (screenWidth - dstWidth) * 0.5f,
            (screenHeight - dstHeight) * 0.5f,
            dstWidth,
            dstHeight};
        DrawTexturePro(frameTex, srcRect, dstRect, Vector2{0.0f, 0.0f}, 0.0f, WHITE);
        if (m_debugUiInitialized && m_debugUiDrawCallback)
        {
            m_debugUiDrawCallback(*this, m_debugUiUserData);
        }
        EndDrawing();

        if (WindowShouldClose())
        {
            RUNTIME_LOG("[run] window close requested, breaking out of loop");
            requestStop();
            break;
        }
    }

    requestStop();

    const auto joinDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!gameThreadFinished.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < joinDeadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (gameThread.joinable())
    {
        if (gameThreadFinished.load(std::memory_order_acquire))
        {
            gameThread.join();
        }
        else
        {
            std::cerr << "[run] game thread did not stop within timeout; detaching" << std::endl;
            gameThread.detach();
        }
    }

    const auto workerDeadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
    while (g_activeThreads.load(std::memory_order_relaxed) > 0 &&
           std::chrono::steady_clock::now() < workerDeadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (g_activeThreads.load(std::memory_order_relaxed) > 0)
    {
        requestStop();
        const auto finalWorkerDeadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
        while (g_activeThreads.load(std::memory_order_relaxed) > 0 &&
               std::chrono::steady_clock::now() < finalWorkerDeadline)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    if (g_activeThreads.load(std::memory_order_relaxed) == 0)
    {
        ps2_syscalls::joinAllGuestHostThreads();
    }
    else
    {
        std::cerr << "[run] guest host threads did not stop within timeout; detaching remaining worker threads"
                  << std::endl;
        ps2_syscalls::detachAllGuestHostThreads();
    }

    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }

    if (gsHistoryDumpPath && gsHistoryDumpPath[0] != '\0')
    {
        m_gs.setDebugHistoryPaused(true);
        const GSDebugSnapshot snapshot = m_gs.getDebugSnapshot();
        const std::vector<GSDebugHistoryEntry> history = m_gs.getDebugHistory();
        std::ofstream out(gsHistoryDumpPath, std::ios::out | std::ios::trunc);
        if (!out)
        {
            std::cerr << "[gs:history] failed to open dump path: "
                      << gsHistoryDumpPath << std::endl;
        }
        else
        {
            out << "snapshot"
                << " ctx0.frame=" << snapshot.ctx[0].frame.fbp
                << "/" << snapshot.ctx[0].frame.fbw
                << "/0x" << std::hex << static_cast<uint32_t>(snapshot.ctx[0].frame.psm)
                << " ctx1.frame=" << std::dec << snapshot.ctx[1].frame.fbp
                << "/" << snapshot.ctx[1].frame.fbw
                << "/0x" << std::hex << static_cast<uint32_t>(snapshot.ctx[1].frame.psm)
                << " present=" << std::dec << snapshot.hostPresentationWidth
                << "x" << snapshot.hostPresentationHeight
                << " display=" << snapshot.hostPresentationDisplayFbp
                << " source=" << snapshot.hostPresentationSourceFbp
                << " preferred=" << snapshot.hostPresentationUsedPreferred
                << '\n';
            out << "seq\tframe\ttick\tkind\tprim\tflags\tframebuf\tzbuf\ttex0"
                   "\ttest\talpha\tbounds\tgif\treg\ttransfer\tpresent\n";
            for (const GSDebugHistoryEntry &entry : history)
            {
                out << entry.seq << '\t'
                    << entry.frameIndex << '\t'
                    << entry.vsyncTick << '\t'
                    << static_cast<uint32_t>(entry.kind) << '\t'
                    << static_cast<uint32_t>(entry.prim.type) << '\t'
                    << "iip=" << entry.prim.iip
                    << "/tme=" << entry.prim.tme
                    << "/abe=" << entry.prim.abe
                    << "/fst=" << entry.prim.fst
                    << "/ctxt=" << entry.prim.ctxt << '\t'
                    << entry.frame.fbp << "/" << entry.frame.fbw
                    << "/0x" << std::hex << static_cast<uint32_t>(entry.frame.psm)
                    << "/0x" << entry.frame.fbmsk << std::dec << '\t'
                    << entry.zbuf.zbp << "/0x" << std::hex
                    << static_cast<uint32_t>(entry.zbuf.psm)
                    << "/m=" << std::dec << entry.zbuf.zmask << '\t'
                    << entry.tex0.tbp0 << "/" << static_cast<uint32_t>(entry.tex0.tbw)
                    << "/0x" << std::hex << static_cast<uint32_t>(entry.tex0.psm)
                    << "/" << std::dec << (1u << entry.tex0.tw)
                    << "x" << (1u << entry.tex0.th)
                    << "/cbp=" << entry.tex0.cbp
                    << "/cpsm=0x" << std::hex << static_cast<uint32_t>(entry.tex0.cpsm)
                    << std::dec << '\t'
                    << "0x" << std::hex << entry.test << '\t'
                    << "0x" << entry.alpha << std::dec << '\t'
                    << entry.xMin << ".." << entry.xMax
                    << "/" << entry.yMin << ".." << entry.yMax
                    << "/z=" << entry.zMin << ".." << entry.zMax
                    << "/a=" << static_cast<uint32_t>(entry.aMin)
                    << ".." << static_cast<uint32_t>(entry.aMax) << '\t'
                    << "nloop=" << entry.gifNloop
                    << "/flg=" << static_cast<uint32_t>(entry.gifFlg)
                    << "/nreg=" << static_cast<uint32_t>(entry.gifNreg)
                    << "/bytes=" << entry.gifSizeBytes << '\t'
                    << "0x" << std::hex << static_cast<uint32_t>(entry.reg)
                    << "=0x" << entry.regValue << std::dec << '\t'
                    << entry.trxreg.rrw << "x" << entry.trxreg.rrh
                    << "/dir=" << entry.trxdir
                    << "/pixels=" << entry.transferPixels << '\t'
                    << entry.width << "x" << entry.height
                    << "/display=" << entry.displayFbp
                    << "/source=" << entry.sourceFbp
                    << "/preferred=" << entry.usedPreferred
                    << '\n';
            }
            std::cout << "[gs:history] wrote " << history.size()
                      << " entries to " << gsHistoryDumpPath << std::endl;
        }
    }

    UnloadTexture(frameTex);
    CloseWindow();

    const int remainingThreads = g_activeThreads.load(std::memory_order_relaxed);
    RUNTIME_LOG("[run] exiting loop, activeThreads=" << remainingThreads);
    if (remainingThreads > 0)
    {
        std::cerr << "[run] warning: " << remainingThreads
                  << " guest worker thread(s) still active during shutdown." << std::endl;
    }
}
