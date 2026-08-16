#include "ps2_runtime.h"

#include "runtime/ps2_save_state.h"
#include "Kernel/Stubs/Helpers/AudioRuntimeState.h"
#include "Kernel/Stubs/Helpers/CdRuntimeState.h"
#include "Kernel/Stubs/Helpers/DmaRuntimeState.h"
#include "Kernel/Stubs/Helpers/LibCRuntimeState.h"
#include "Kernel/Stubs/Helpers/MemoryCardRuntimeState.h"
#include "Kernel/Stubs/Helpers/PadRuntimeState.h"
#include "Kernel/Stubs/Helpers/SifRuntimeState.h"
#include "Kernel/Syscalls/Helpers/Deci2RuntimeState.h"
#include "Kernel/Syscalls/Helpers/FileRuntimeState.h"
#include "Kernel/Syscalls/Helpers/InterruptRuntimeState.h"
#include "Kernel/Syscalls/Helpers/KernelRuntimeState.h"
#include "Kernel/Syscalls/Helpers/RpcRuntimeState.h"
#include "Kernel/Syscalls/Helpers/SyncRuntimeState.h"
#include "Kernel/Syscalls/Helpers/ThreadRuntimeState.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string_view>
#include <tuple>

namespace
{
    using ps2x::savestate::Chunk;
    using ps2x::savestate::ChunkFlags;
    using ps2x::savestate::Document;
    using ps2x::savestate::Reader;
    using ps2x::savestate::Writer;

    constexpr uint32_t kRuntimeStateVersion = 1u;
    constexpr uint32_t kMetaChunk =
        ps2x::savestate::makeChunkId('M', 'E', 'T', 'A');
    constexpr uint32_t kMemoryChunk =
        ps2x::savestate::makeChunkId('M', 'E', 'M', '0');
    constexpr uint32_t kCoreChunk =
        ps2x::savestate::makeChunkId('C', 'O', 'R', 'E');
    constexpr uint32_t kVu0Chunk =
        ps2x::savestate::makeChunkId('V', 'U', '0', '0');
    constexpr uint32_t kVu1Chunk =
        ps2x::savestate::makeChunkId('V', 'U', '1', '0');
    constexpr uint32_t kGsChunk =
        ps2x::savestate::makeChunkId('G', 'S', '0', '0');
    constexpr uint32_t kHleChunk =
        ps2x::savestate::makeChunkId('H', 'L', 'E', '0');
    constexpr uint32_t kMaximumSavedThreads = 256u;
    constexpr uint32_t kMaximumSavedObjects = 65536u;
    constexpr uint32_t kMaximumSavedMapEntries = 1u << 20u;
    constexpr uint32_t kMaximumSavedString = 1u << 20u;
    constexpr uint64_t kMaximumVuPacketBytes = 16u * 1024u * 1024u;

    void setError(std::string *error, std::string_view message)
    {
        if (error)
            *error = message;
    }

    template <typename Enum>
    bool readEnum8(Reader &reader, Enum &value, uint8_t maximum)
    {
        uint8_t encoded = 0u;
        if (!reader.u8(encoded) || encoded > maximum)
            return false;
        value = static_cast<Enum>(encoded);
        return true;
    }

    void writeM128(Writer &writer, __m128i value)
    {
        alignas(16) std::array<uint32_t, 4u> words{};
        _mm_storeu_si128(
            reinterpret_cast<__m128i *>(words.data()), value);
        for (uint32_t word : words)
            writer.u32(word);
    }

    bool readM128(Reader &reader, __m128i &value)
    {
        std::array<uint32_t, 4u> words{};
        for (uint32_t &word : words)
        {
            if (!reader.u32(word))
                return false;
        }
        value = _mm_set_epi32(
            static_cast<int32_t>(words[3]),
            static_cast<int32_t>(words[2]),
            static_cast<int32_t>(words[1]),
            static_cast<int32_t>(words[0]));
        return true;
    }

    void writeM128f(Writer &writer, __m128 value)
    {
        writeM128(writer, _mm_castps_si128(value));
    }

    bool readM128f(Reader &reader, __m128 &value)
    {
        __m128i encoded{};
        if (!readM128(reader, encoded))
            return false;
        value = _mm_castsi128_ps(encoded);
        return true;
    }

    void writeR5900Context(Writer &writer, const R5900Context &state)
    {
        for (const __m128i value : state.r)
            writeM128(writer, value);
        writer.u32(state.pc);
        writer.u64(state.insn_count);
        writer.u32(state.ee_block_cycle_ticks);
        writer.boolean(state.ee_block_cycle_active);
        writer.u64(state.cop0_random_pending_retirements);
        writer.boolean(state.ee_performance_issue_pending);
        writer.u32(state.ee_performance_pending_issue.pipeMask);
        writer.u32(state.ee_performance_pending_issue.gprReadMask);
        writer.u32(state.ee_performance_pending_issue.gprWriteMask);
        writer.u32(state.ee_performance_pending_issue.flags);
        writer.u64(state.hi);
        writer.u64(state.lo);
        writer.u64(state.hi1);
        writer.u64(state.lo1);
        writer.u32(state.sa);

        for (const __m128 value : state.vu0_vf)
            writeM128f(writer, value);
        for (uint16_t value : state.vi)
            writer.u16(value);
        writer.f32(state.vu0_q);
        writer.f32(state.vu0_p);
        writer.f32(state.vu0_i);
        writeM128f(writer, state.vu0_r);
        writeM128f(writer, state.vu0_acc);
        writer.u16(state.vu0_status);
        writer.u32(state.vu0_mac_flags);
        writer.u32(state.vu0_clip_flags);
        writer.u32(state.vu0_clip_flags2);
        writer.u32(state.vu0_cmsar0);
        writer.u32(state.vu0_cmsar1);
        writer.u32(state.vu0_cmsar2);
        writer.u32(state.vu0_cmsar3);
        writer.u32(state.vu0_vpu_stat);
        writer.u32(state.vu0_vpu_stat2);
        writer.u32(state.vu0_vpu_stat3);
        writer.u32(state.vu0_vpu_stat4);
        writer.u32(state.vu0_tpc);
        writer.u32(state.vu0_tpc2);
        writer.u32(state.vu0_fbrst);
        writer.u32(state.vu0_fbrst2);
        writer.u32(state.vu0_fbrst3);
        writer.u32(state.vu0_fbrst4);
        writer.u32(state.vu0_itop);
        writer.u32(state.vu0_top);
        writer.u32(state.vu0_info);
        writer.u32(state.vu0_xitop);
        writer.u32(state.vu0_pc);
        for (float value : state.vu0_cf)
            writer.f32(value);

#define PS2X_WRITE_COP0(field) writer.u32(state.field)
        PS2X_WRITE_COP0(cop0_index);
        PS2X_WRITE_COP0(cop0_random);
        PS2X_WRITE_COP0(cop0_entrylo0);
        PS2X_WRITE_COP0(cop0_entrylo1);
        PS2X_WRITE_COP0(cop0_context);
        PS2X_WRITE_COP0(cop0_pagemask);
        PS2X_WRITE_COP0(cop0_wired);
        PS2X_WRITE_COP0(cop0_badvaddr);
        PS2X_WRITE_COP0(cop0_count);
        PS2X_WRITE_COP0(cop0_entryhi);
        PS2X_WRITE_COP0(cop0_compare);
        PS2X_WRITE_COP0(cop0_status);
        PS2X_WRITE_COP0(cop0_cause);
        PS2X_WRITE_COP0(cop0_epc);
        PS2X_WRITE_COP0(cop0_prid);
        PS2X_WRITE_COP0(cop0_config);
        PS2X_WRITE_COP0(cop0_badpaddr);
        PS2X_WRITE_COP0(cop0_bpc);
        PS2X_WRITE_COP0(cop0_iab);
        PS2X_WRITE_COP0(cop0_iabm);
        PS2X_WRITE_COP0(cop0_dab);
        PS2X_WRITE_COP0(cop0_dabm);
        PS2X_WRITE_COP0(cop0_dvb);
        PS2X_WRITE_COP0(cop0_dvbm);
        PS2X_WRITE_COP0(cop0_perf);
        PS2X_WRITE_COP0(cop0_pcr0);
        PS2X_WRITE_COP0(cop0_pcr1);
        PS2X_WRITE_COP0(cop0_taglo);
        PS2X_WRITE_COP0(cop0_taghi);
        PS2X_WRITE_COP0(cop0_errorepc);
#undef PS2X_WRITE_COP0

        writer.boolean(state.in_delay_slot);
        writer.u32(state.branch_pc);
        for (uint32_t value : state.cop2_ccr)
            writer.u32(value);
        for (float value : state.f)
            writer.f32(value);
        writer.f32(state.f_acc);
        writer.u32(state.fcr31);
    }

    bool readR5900Context(Reader &reader, R5900Context &state)
    {
        R5900Context decoded{};
        for (__m128i &value : decoded.r)
        {
            if (!readM128(reader, value))
                return false;
        }
        if (!reader.u32(decoded.pc) ||
            !reader.u64(decoded.insn_count) ||
            !reader.u32(decoded.ee_block_cycle_ticks) ||
            !reader.boolean(decoded.ee_block_cycle_active) ||
            !reader.u64(decoded.cop0_random_pending_retirements) ||
            !reader.boolean(decoded.ee_performance_issue_pending) ||
            !reader.u32(decoded.ee_performance_pending_issue.pipeMask) ||
            !reader.u32(decoded.ee_performance_pending_issue.gprReadMask) ||
            !reader.u32(decoded.ee_performance_pending_issue.gprWriteMask) ||
            !reader.u32(decoded.ee_performance_pending_issue.flags) ||
            !reader.u64(decoded.hi) || !reader.u64(decoded.lo) ||
            !reader.u64(decoded.hi1) || !reader.u64(decoded.lo1) ||
            !reader.u32(decoded.sa))
        {
            return false;
        }
        for (__m128 &value : decoded.vu0_vf)
        {
            if (!readM128f(reader, value))
                return false;
        }
        for (uint16_t &value : decoded.vi)
        {
            if (!reader.u16(value))
                return false;
        }
        if (!reader.f32(decoded.vu0_q) ||
            !reader.f32(decoded.vu0_p) ||
            !reader.f32(decoded.vu0_i) ||
            !readM128f(reader, decoded.vu0_r) ||
            !readM128f(reader, decoded.vu0_acc) ||
            !reader.u16(decoded.vu0_status) ||
            !reader.u32(decoded.vu0_mac_flags) ||
            !reader.u32(decoded.vu0_clip_flags) ||
            !reader.u32(decoded.vu0_clip_flags2) ||
            !reader.u32(decoded.vu0_cmsar0) ||
            !reader.u32(decoded.vu0_cmsar1) ||
            !reader.u32(decoded.vu0_cmsar2) ||
            !reader.u32(decoded.vu0_cmsar3) ||
            !reader.u32(decoded.vu0_vpu_stat) ||
            !reader.u32(decoded.vu0_vpu_stat2) ||
            !reader.u32(decoded.vu0_vpu_stat3) ||
            !reader.u32(decoded.vu0_vpu_stat4) ||
            !reader.u32(decoded.vu0_tpc) ||
            !reader.u32(decoded.vu0_tpc2) ||
            !reader.u32(decoded.vu0_fbrst) ||
            !reader.u32(decoded.vu0_fbrst2) ||
            !reader.u32(decoded.vu0_fbrst3) ||
            !reader.u32(decoded.vu0_fbrst4) ||
            !reader.u32(decoded.vu0_itop) ||
            !reader.u32(decoded.vu0_top) ||
            !reader.u32(decoded.vu0_info) ||
            !reader.u32(decoded.vu0_xitop) ||
            !reader.u32(decoded.vu0_pc))
        {
            return false;
        }
        for (float &value : decoded.vu0_cf)
        {
            if (!reader.f32(value))
                return false;
        }

#define PS2X_READ_COP0(field) if (!reader.u32(decoded.field)) return false
        PS2X_READ_COP0(cop0_index);
        PS2X_READ_COP0(cop0_random);
        PS2X_READ_COP0(cop0_entrylo0);
        PS2X_READ_COP0(cop0_entrylo1);
        PS2X_READ_COP0(cop0_context);
        PS2X_READ_COP0(cop0_pagemask);
        PS2X_READ_COP0(cop0_wired);
        PS2X_READ_COP0(cop0_badvaddr);
        PS2X_READ_COP0(cop0_count);
        PS2X_READ_COP0(cop0_entryhi);
        PS2X_READ_COP0(cop0_compare);
        PS2X_READ_COP0(cop0_status);
        PS2X_READ_COP0(cop0_cause);
        PS2X_READ_COP0(cop0_epc);
        PS2X_READ_COP0(cop0_prid);
        PS2X_READ_COP0(cop0_config);
        PS2X_READ_COP0(cop0_badpaddr);
        PS2X_READ_COP0(cop0_bpc);
        PS2X_READ_COP0(cop0_iab);
        PS2X_READ_COP0(cop0_iabm);
        PS2X_READ_COP0(cop0_dab);
        PS2X_READ_COP0(cop0_dabm);
        PS2X_READ_COP0(cop0_dvb);
        PS2X_READ_COP0(cop0_dvbm);
        PS2X_READ_COP0(cop0_perf);
        PS2X_READ_COP0(cop0_pcr0);
        PS2X_READ_COP0(cop0_pcr1);
        PS2X_READ_COP0(cop0_taglo);
        PS2X_READ_COP0(cop0_taghi);
        PS2X_READ_COP0(cop0_errorepc);
#undef PS2X_READ_COP0

        if (!reader.boolean(decoded.in_delay_slot) ||
            !reader.u32(decoded.branch_pc))
        {
            return false;
        }
        for (uint32_t &value : decoded.cop2_ccr)
        {
            if (!reader.u32(value))
                return false;
        }
        for (float &value : decoded.f)
        {
            if (!reader.f32(value))
                return false;
        }
        if (!reader.f32(decoded.f_acc) || !reader.u32(decoded.fcr31))
            return false;
        decoded.enforceVu0RegisterInvariants();
        state = decoded;
        return true;
    }

    void writeVuExecutionState(Writer &writer, const VuExecutionState &state)
    {
        for (const auto &reg : state.vf)
            for (float value : reg)
                writer.f32(value);
        for (int32_t value : state.vi)
            writer.i32(value);
        for (float value : state.acc)
            writer.f32(value);
        writer.f32(state.q);
        writer.f32(state.p);
        writer.f32(state.i);
        writer.u32(state.pc);
        writer.u32(state.mac);
        writer.u32(state.clip);
        writer.u32(state.status);
        writer.boolean(state.ebit);
        writer.u32(state.top);
        writer.u32(state.itop);
        writer.boolean(state.branchPending);
        writer.u32(state.branchTarget);
        writer.u32(state.branchDelay);
        writer.u8(state.viBackupCycles);
        writer.u8(state.viBackupRegister);
        writer.i32(state.viBackupValue);
        writer.boolean(state.active);
        writer.u64(state.issuedCycles);

        const VuPipelineState &pipeline = state.pipeline;
        writer.boolean(pipeline.xgkick.active);
        writer.u32(pipeline.xgkick.address);
        writer.u32(pipeline.xgkick.tagBytesRemaining);
        writer.u32(pipeline.xgkick.cycleCredit);
        writer.boolean(pipeline.xgkick.tagEop);
        writer.sizedBytes(std::span<const uint8_t>(
            pipeline.xgkick.packet.data(),
            pipeline.xgkick.packet.size()));
        writer.boolean(pipeline.delayedQ.active);
        writer.f32(pipeline.delayedQ.result);
        writer.u32(pipeline.delayedQ.cyclesRemaining);
        writer.boolean(pipeline.delayedP.active);
        writer.f32(pipeline.delayedP.result);
        writer.u32(pipeline.delayedP.cyclesRemaining);
        for (const auto &flags : pipeline.fmacFlags)
        {
            writer.boolean(flags.active);
            writer.boolean(flags.fmacActive);
            writer.boolean(flags.clipActive);
            writer.u32(flags.mac);
            writer.u32(flags.status);
            writer.u32(flags.clip);
        }
        writer.u8(pipeline.fmacFlagIndex);
        writer.u32(pipeline.workingMac);
        writer.u32(pipeline.workingClip);
    }

    bool readVuExecutionState(Reader &reader, VuExecutionState &state)
    {
        VuExecutionState decoded{};
        for (auto &reg : decoded.vf)
            for (float &value : reg)
                if (!reader.f32(value)) return false;
        for (int32_t &value : decoded.vi)
            if (!reader.i32(value)) return false;
        for (float &value : decoded.acc)
            if (!reader.f32(value)) return false;
        if (!reader.f32(decoded.q) || !reader.f32(decoded.p) ||
            !reader.f32(decoded.i) || !reader.u32(decoded.pc) ||
            !reader.u32(decoded.mac) || !reader.u32(decoded.clip) ||
            !reader.u32(decoded.status) || !reader.boolean(decoded.ebit) ||
            !reader.u32(decoded.top) || !reader.u32(decoded.itop) ||
            !reader.boolean(decoded.branchPending) ||
            !reader.u32(decoded.branchTarget) ||
            !reader.u32(decoded.branchDelay) ||
            !reader.u8(decoded.viBackupCycles) ||
            !reader.u8(decoded.viBackupRegister) ||
            !reader.i32(decoded.viBackupValue) ||
            !reader.boolean(decoded.active) ||
            !reader.u64(decoded.issuedCycles))
        {
            return false;
        }
        VuPipelineState &pipeline = decoded.pipeline;
        std::span<const uint8_t> packet;
        if (!reader.boolean(pipeline.xgkick.active) ||
            !reader.u32(pipeline.xgkick.address) ||
            !reader.u32(pipeline.xgkick.tagBytesRemaining) ||
            !reader.u32(pipeline.xgkick.cycleCredit) ||
            !reader.boolean(pipeline.xgkick.tagEop) ||
            !reader.sizedBytes(packet, kMaximumVuPacketBytes))
        {
            return false;
        }
        pipeline.xgkick.packet.resize(packet.size());
        if (!packet.empty())
        {
            std::memcpy(
                pipeline.xgkick.packet.data(),
                packet.data(), packet.size());
        }
        if (!reader.boolean(pipeline.delayedQ.active) ||
            !reader.f32(pipeline.delayedQ.result) ||
            !reader.u32(pipeline.delayedQ.cyclesRemaining) ||
            !reader.boolean(pipeline.delayedP.active) ||
            !reader.f32(pipeline.delayedP.result) ||
            !reader.u32(pipeline.delayedP.cyclesRemaining))
        {
            return false;
        }
        for (auto &flags : pipeline.fmacFlags)
        {
            if (!reader.boolean(flags.active) ||
                !reader.boolean(flags.fmacActive) ||
                !reader.boolean(flags.clipActive) ||
                !reader.u32(flags.mac) ||
                !reader.u32(flags.status) ||
                !reader.u32(flags.clip))
            {
                return false;
            }
        }
        if (!reader.u8(pipeline.fmacFlagIndex) ||
            pipeline.fmacFlagIndex >= VuPipelineState::kFmacFlagStages ||
            !reader.u32(pipeline.workingMac) ||
            !reader.u32(pipeline.workingClip))
        {
            return false;
        }
        state = std::move(decoded);
        return true;
    }

    void writeVu1VifState(Writer &writer, const Vu1VifState &state)
    {
        writer.u32(state.cycle);
        writer.u32(state.mode);
        writer.u32(state.mask);
        for (uint32_t value : state.row) writer.u32(value);
        for (uint32_t value : state.column) writer.u32(value);
        writer.u32(state.tops);
    }

    bool readVu1VifState(Reader &reader, Vu1VifState &state)
    {
        if (!reader.u32(state.cycle) ||
            !reader.u32(state.mode) ||
            !reader.u32(state.mask))
        {
            return false;
        }
        for (uint32_t &value : state.row)
            if (!reader.u32(value)) return false;
        for (uint32_t &value : state.column)
            if (!reader.u32(value)) return false;
        return reader.u32(state.tops);
    }

    void writeCop0Timing(
        Writer &writer,
        const ps2x::timing::Cop0TimingSnapshot &state)
    {
        writer.u32(state.count);
        writer.u32(state.compare);
        writer.u32(state.pccr);
        for (uint32_t value : state.pcr) writer.u32(value);
        writer.u64(state.lastCountCycle);
        for (uint64_t value : state.lastPerformanceCycle)
            writer.u64(value);
        writer.boolean(state.timerArmed);
        writer.boolean(state.timerPending);
    }

    bool readCop0Timing(
        Reader &reader,
        ps2x::timing::Cop0TimingSnapshot &state)
    {
        if (!reader.u32(state.count) ||
            !reader.u32(state.compare) ||
            !reader.u32(state.pccr))
        {
            return false;
        }
        for (uint32_t &value : state.pcr)
            if (!reader.u32(value)) return false;
        if (!reader.u64(state.lastCountCycle)) return false;
        for (uint64_t &value : state.lastPerformanceCycle)
            if (!reader.u64(value)) return false;
        return reader.boolean(state.timerArmed) &&
               reader.boolean(state.timerPending);
    }

    void writeEventToken(
        Writer &writer, ps2x::timing::EeEventToken token)
    {
        writer.u8(static_cast<uint8_t>(token.source));
        writer.u64(token.generation);
    }

    bool readEventToken(
        Reader &reader, ps2x::timing::EeEventToken &token)
    {
        return readEnum8(
                   reader, token.source,
                   static_cast<uint8_t>(
                       ps2x::timing::EeEventSource::Count) - 1u) &&
               reader.u64(token.generation);
    }

    void writeEventScheduler(
        Writer &writer,
        const ps2x::timing::EeEventSchedulerSnapshot &state)
    {
        for (const auto &slot : state.slots)
        {
            writer.u8(static_cast<uint8_t>(slot.source));
            writer.u64(slot.deadline.raw());
            writer.u64(slot.generation);
            writer.u64(slot.sequence);
            writer.boolean(slot.pending);
        }
        writer.u64(state.nextSequence);
        writer.u64(state.statistics.scheduled);
        writer.u64(state.statistics.replaced);
        writer.u64(state.statistics.cancelled);
        writer.u64(state.statistics.serviced);
        writer.u64(state.statistics.late);
        writer.u64(state.statistics.sameTickReschedules);
        writer.u64(state.statistics.serviceLimitHits);
    }

    bool readEventScheduler(
        Reader &reader,
        ps2x::timing::EeEventSchedulerSnapshot &state)
    {
        for (auto &slot : state.slots)
        {
            uint64_t deadline = 0u;
            if (!readEnum8(
                    reader, slot.source,
                    static_cast<uint8_t>(
                        ps2x::timing::EeEventSource::Count) - 1u) ||
                !reader.u64(deadline) ||
                !reader.u64(slot.generation) ||
                !reader.u64(slot.sequence) ||
                !reader.boolean(slot.pending))
            {
                return false;
            }
            slot.deadline = ps2x::timing::eeTickFromRaw(deadline);
        }
        return reader.u64(state.nextSequence) &&
               reader.u64(state.statistics.scheduled) &&
               reader.u64(state.statistics.replaced) &&
               reader.u64(state.statistics.cancelled) &&
               reader.u64(state.statistics.serviced) &&
               reader.u64(state.statistics.late) &&
               reader.u64(state.statistics.sameTickReschedules) &&
               reader.u64(state.statistics.serviceLimitHits);
    }

    void writeThreadScheduler(
        Writer &writer,
        const ps2x::ee::EeThreadSchedulerSnapshot &state)
    {
        writer.u32(static_cast<uint32_t>(state.threads.size()));
        for (const auto &thread : state.threads)
        {
            writer.i32(thread.id);
            writer.u32(thread.generation);
            writer.i32(thread.priority);
            writer.u8(static_cast<uint8_t>(thread.state));
            writer.u8(static_cast<uint8_t>(thread.wait.kind));
            writer.i32(thread.wait.objectId);
            writer.u8(static_cast<uint8_t>(
                thread.completedWait.wait.kind));
            writer.i32(thread.completedWait.wait.objectId);
            writer.u8(static_cast<uint8_t>(
                thread.completedWait.completion));
            writer.boolean(thread.retainCompletedWaitReason);
            writer.u32(thread.wakeupCount);
            writer.u32(thread.suspendCount);
            writer.u64(thread.queueSequence);
        }
        writer.boolean(state.currentThreadId.has_value());
        if (state.currentThreadId)
            writer.i32(*state.currentThreadId);
        writer.u64(state.nextQueueSequence);
        writer.u64(state.canonicalTick.raw());
    }

    bool readThreadScheduler(
        Reader &reader,
        ps2x::ee::EeThreadSchedulerSnapshot &state)
    {
        uint32_t count = 0u;
        if (!reader.u32(count) || count > kMaximumSavedThreads)
            return false;
        state.threads.clear();
        state.threads.resize(count);
        for (auto &thread : state.threads)
        {
            if (!reader.i32(thread.id) ||
                !reader.u32(thread.generation) ||
                !reader.i32(thread.priority) ||
                !readEnum8(
                    reader, thread.state,
                    static_cast<uint8_t>(
                        ps2x::ee::EeSchedulerThreadState::WaitSuspended)) ||
                !readEnum8(
                    reader, thread.wait.kind,
                    static_cast<uint8_t>(
                        ps2x::ee::EeSchedulerWaitKind::HleSemaphore)) ||
                !reader.i32(thread.wait.objectId) ||
                !readEnum8(
                    reader, thread.completedWait.wait.kind,
                    static_cast<uint8_t>(
                        ps2x::ee::EeSchedulerWaitKind::HleSemaphore)) ||
                !reader.i32(thread.completedWait.wait.objectId) ||
                !readEnum8(
                    reader, thread.completedWait.completion,
                    static_cast<uint8_t>(
                        ps2x::ee::EeSchedulerWaitCompletion::TimedOut)) ||
                !reader.boolean(thread.retainCompletedWaitReason) ||
                !reader.u32(thread.wakeupCount) ||
                !reader.u32(thread.suspendCount) ||
                !reader.u64(thread.queueSequence))
            {
                return false;
            }
        }
        bool hasCurrent = false;
        int current = 0;
        uint64_t canonicalTick = 0u;
        if (!reader.boolean(hasCurrent) ||
            (hasCurrent && !reader.i32(current)) ||
            !reader.u64(state.nextQueueSequence) ||
            !reader.u64(canonicalTick))
        {
            return false;
        }
        state.currentThreadId =
            hasCurrent ? std::optional<int>(current) : std::nullopt;
        state.canonicalTick =
            ps2x::timing::eeTickFromRaw(canonicalTick);
        ps2x::ee::EeThreadScheduler validator;
        return validator.restore(state);
    }

    struct SavedThread
    {
        int id = 0;
        uint32_t entry = 0u;
        uint32_t stack = 0u;
        uint32_t stackSize = 0u;
        uint32_t gp = 0u;
        uint32_t priority = 0u;
        uint32_t attr = 0u;
        uint32_t option = 0u;
        uint32_t arg = 0u;
        uint32_t generation = 0u;
        bool ownsStack = false;
        uint32_t tlsBase = 0u;
        EeThreadGuestStateSnapshot guest{};
        R5900Context context{};
        uint32_t currentPc = 0u;
        ps2x::ee::EeSchedulerWaitCompletion pendingWait =
            ps2x::ee::EeSchedulerWaitCompletion::None;
        uint32_t pendingEventFlagResultBits = 0u;
        bool resumeAfterNativeWait = false;
    };

    void writeSavedThread(Writer &writer, const SavedThread &thread)
    {
        writer.i32(thread.id);
        writer.u32(thread.entry);
        writer.u32(thread.stack);
        writer.u32(thread.stackSize);
        writer.u32(thread.gp);
        writer.u32(thread.priority);
        writer.u32(thread.attr);
        writer.u32(thread.option);
        writer.u32(thread.arg);
        writer.u32(thread.generation);
        writer.boolean(thread.ownsStack);
        writer.u32(thread.tlsBase);
        writer.i32(thread.guest.status);
        writer.i32(thread.guest.waitType);
        writer.i32(thread.guest.waitId);
        writer.i32(thread.guest.wakeupCount);
        writer.i32(thread.guest.currentPriority);
        writer.i32(thread.guest.suspendCount);
        writer.u8(static_cast<uint8_t>(thread.guest.waitQueue));
        writer.i32(thread.guest.waitQueueId);
        writer.boolean(thread.guest.waitCompletionPending);
        writer.boolean(thread.guest.started);
        writer.u64(thread.guest.revision);
        writeR5900Context(writer, thread.context);
        writer.u32(thread.currentPc);
        writer.u8(static_cast<uint8_t>(thread.pendingWait));
        writer.u32(thread.pendingEventFlagResultBits);
        writer.boolean(thread.resumeAfterNativeWait);
    }

    bool readSavedThread(Reader &reader, SavedThread &thread)
    {
        if (!reader.i32(thread.id) ||
            !reader.u32(thread.entry) ||
            !reader.u32(thread.stack) ||
            !reader.u32(thread.stackSize) ||
            !reader.u32(thread.gp) ||
            !reader.u32(thread.priority) ||
            !reader.u32(thread.attr) ||
            !reader.u32(thread.option) ||
            !reader.u32(thread.arg) ||
            !reader.u32(thread.generation) ||
            !reader.boolean(thread.ownsStack) ||
            !reader.u32(thread.tlsBase) ||
            !reader.i32(thread.guest.status) ||
            !reader.i32(thread.guest.waitType) ||
            !reader.i32(thread.guest.waitId) ||
            !reader.i32(thread.guest.wakeupCount) ||
            !reader.i32(thread.guest.currentPriority) ||
            !reader.i32(thread.guest.suspendCount) ||
            !readEnum8(
                reader, thread.guest.waitQueue,
                static_cast<uint8_t>(EeThreadWaitQueue::HleSemaphore)) ||
            !reader.i32(thread.guest.waitQueueId) ||
            !reader.boolean(thread.guest.waitCompletionPending) ||
            !reader.boolean(thread.guest.started) ||
            !reader.u64(thread.guest.revision) ||
            !readR5900Context(reader, thread.context) ||
            !reader.u32(thread.currentPc) ||
            !readEnum8(
                reader, thread.pendingWait,
                static_cast<uint8_t>(
                    ps2x::ee::EeSchedulerWaitCompletion::TimedOut)) ||
            !reader.u32(thread.pendingEventFlagResultBits) ||
            !reader.boolean(thread.resumeAfterNativeWait))
        {
            return false;
        }
        EeThreadGuestState validator;
        return thread.id > 0 && thread.generation != 0u &&
               thread.priority < 128u && validator.restore(thread.guest);
    }

    void recoverNativeWaitContinuation(SavedThread &thread)
    {
        if (!thread.resumeAfterNativeWait)
            return;
        const uint32_t returnAddress = getRegU32(&thread.context, 31);
        if (returnAddress != 0u)
        {
            thread.context.pc = returnAddress;
            setReturnS32(&thread.context, 0);
            thread.context.in_delay_slot = false;
            thread.context.branch_pc = 0u;
            thread.context.resetEeBlockTiming();
        }
    }
}

bool PS2Runtime::encodeRuntimeCoreSaveState(
    const ps2x::ee::EeThreadSchedulerSnapshot &scheduler,
    std::vector<uint8_t> &output,
    std::string *error)
{
    if (!usesDedicatedEeExecutor())
    {
        setError(error,
                 "save states require the dedicated EE fiber executor");
        return false;
    }

    Writer writer;
    writer.u32(kRuntimeStateVersion);
    writeR5900Context(writer, m_cpuContext);
    writeCop0Timing(writer, m_cop0Timing.snapshot());
    writer.u64(m_eeTimeline.now().raw());
    writeEventScheduler(writer, m_eeEventScheduler.snapshot());
    writeThreadScheduler(writer, scheduler);

    std::vector<SavedThread> threads;
    {
        EeThreadRuntimeState &runtimeState = *m_eeThreadRuntimeState;
        std::lock_guard<std::mutex> mapLock(runtimeState.threadMapMutex);
        writer.i32(runtimeState.nextThreadId);
        writer.u32(runtimeState.nextGeneration);
        writer.i32(runtimeState.currentThreadId.load(
            std::memory_order_acquire));
        threads.reserve(runtimeState.threads.size());
        for (const auto &[threadId, info] : runtimeState.threads)
        {
            if (!info)
                continue;
            SavedThread saved{};
            saved.id = threadId;
            std::lock_guard<std::mutex> threadLock(info->m);
            saved.entry = info->entry;
            saved.stack = info->stack;
            saved.stackSize = info->stackSize;
            saved.gp = info->gp;
            saved.priority = info->priority;
            saved.attr = info->attr;
            saved.option = info->option;
            saved.arg = info->arg;
            saved.generation = info->generation;
            saved.ownsStack = info->ownsStack;
            saved.tlsBase = info->tlsBase;
            saved.guest = info->guestState.snapshot();
            saved.context = threadId == 1
                                ? m_cpuContext
                                : info->context;
            saved.currentPc = info->currentPc.load(
                std::memory_order_relaxed);
            saved.pendingWait = info->pendingWaitCompletion;
            saved.pendingEventFlagResultBits =
                info->pendingEventFlagResultBits;
            // A stackful HLE wait cannot be serialized. On a cold load its
            // fresh continuation starts at the guest return address and is
            // kept blocked by the restored scheduler until the object wakes.
            saved.resumeAfterNativeWait =
                saved.guest.status == EeThreadGuestState::kWait ||
                saved.guest.status == EeThreadGuestState::kWaitSuspend;
            threads.push_back(std::move(saved));
        }
    }
    std::sort(
        threads.begin(), threads.end(),
        [](const SavedThread &left, const SavedThread &right)
        { return left.id < right.id; });
    if (threads.size() > kMaximumSavedThreads)
    {
        setError(error, "save state has too many EE threads");
        return false;
    }
    writer.u32(static_cast<uint32_t>(threads.size()));
    for (const SavedThread &thread : threads)
        writeSavedThread(writer, thread);

    writer.u64(m_vu1ExecutionTiming.generation);
    writer.u64(m_vu1ExecutionTiming.lastAdvancedTick.raw());
    writer.u64(m_vu1ExecutionTiming.totalAdvancedCycles);
    writeEventToken(writer, m_vu1ExecutionTiming.eventToken);
    writeEventToken(writer, m_vif0VuFinishEventToken);
    writeEventToken(writer, m_vif0DmaEventToken);
    writeEventToken(writer, m_vif1DmaEventToken);
    writeEventToken(writer, m_gifDmaEventToken);
    writeEventToken(writer, m_fromIpuDmaEventToken);
    writeEventToken(writer, m_toIpuDmaEventToken);
    writeEventToken(writer, m_hleSifDmaEventToken);
    writeEventToken(writer, m_fromScratchpadDmaEventToken);
    writeEventToken(writer, m_toScratchpadDmaEventToken);

    writer.u32(static_cast<uint32_t>(m_hleSifDmaQueueCount));
    for (size_t logical = 0u; logical < m_hleSifDmaQueueCount; ++logical)
    {
        const HleSifDmaOperation &operation =
            m_hleSifDmaQueue[
                (m_hleSifDmaQueueHead + logical) %
                kHleSifDmaQueueCapacity];
        writer.u32(operation.submission.transferId);
        writer.u32(operation.submission.transferCount);
        for (uint32_t index = 0u;
             index < operation.submission.transferCount;
             ++index)
        {
            const HleSifDmaTransfer &transfer =
                operation.submission.transfers[index];
            writer.u32(transfer.src);
            writer.u32(transfer.dest);
            writer.u32(transfer.size);
            writer.u32(transfer.iopOffset);
            writer.u8(static_cast<uint8_t>(transfer.destination));
        }
        writer.u64(operation.operationGeneration);
        writer.u64(operation.iopCycles);
    }
    writer.u64(m_hleSifDmaOperationGeneration);

    writer.u8(static_cast<uint8_t>(m_eeVSyncTiming.phase));
    writer.u64(m_eeVSyncTiming.startTick.raw());
    writeEventToken(writer, m_eeVSyncTiming.eventToken);
    writer.u64(m_eeVSyncTiming.durations.periodCycles);
    writer.u64(m_eeVSyncTiming.durations.renderCycles);
    writer.u64(m_eeVSyncTiming.durations.blankCycles);
    writer.u64(m_eeVSyncTiming.durations.gsBlankCycles);
    writer.u32(m_eeVSyncTiming.durations.hRenderCycles);
    writer.u32(m_eeVSyncTiming.durations.hBlankCycles);
    writer.u32(m_eeVSyncTiming.durations.hSyncErrorCycles);
    writer.u64(m_eeVSyncTiming.currentVSyncTick);
    writer.u32(m_eeVSyncTiming.videoMode);
    writer.u8(static_cast<uint8_t>(m_eeVSyncTiming.videoModeClass));
    writer.boolean(m_eeVSyncTiming.videoModeConfigured);
    writer.boolean(m_eeVSyncTiming.interlaced);
    writer.boolean(m_eeVSyncTiming.enabled);
    writeEventToken(writer, m_eeCounterEventToken);
    writeEventToken(writer, m_cop0TimerEventToken);
    writeEventToken(writer, m_cop0PerformanceEventToken);
    writer.u32(m_pendingEeCounterInterrupts);

    writer.u32(static_cast<uint32_t>(m_pendingVSyncDeliveryCount));
    for (size_t index = 0u;
         index < m_pendingVSyncDeliveryCount;
         ++index)
    {
        const PendingVSyncDelivery &delivery =
            m_pendingVSyncDeliveries[index];
        writer.u8(static_cast<uint8_t>(delivery.kind));
        writer.u64(delivery.tick);
        writer.u64(delivery.eventSequence);
    }
    writer.u64(m_vu0CycleTick.raw());

    writer.u32(static_cast<uint32_t>(m_pendingDmacInterrupts.size()));
    for (const DmacPendingInterrupt &interrupt : m_pendingDmacInterrupts)
    {
        writer.u8(static_cast<uint8_t>(interrupt.source));
        writer.u32(interrupt.cause);
        writer.u64(interrupt.channelGeneration);
        writer.u64(interrupt.eventSequence);
        writer.u64(interrupt.publicationSequence);
    }
    writer.boolean(m_dmacCompletionReady.load(
        std::memory_order_acquire));
    writer.boolean(m_dmacInterruptDeliveryDirty.load(
        std::memory_order_acquire));

    {
        std::lock_guard<std::mutex> lock(m_guestHeapMutex);
        writer.u32(m_guestHeapBase);
        writer.u32(m_guestHeapEnd);
        writer.u32(m_guestHeapLimit);
        writer.u32(m_guestHeapSuggestedBase);
        writer.boolean(m_guestHeapConfigured);
        writer.u32(static_cast<uint32_t>(m_guestHeapBlocks.size()));
        for (const GuestHeapBlock &block : m_guestHeapBlocks)
        {
            writer.u32(block.addr);
            writer.u32(block.size);
            writer.boolean(block.free);
        }
    }

    output = writer.take();
    return true;
}

bool PS2Runtime::decodeRuntimeCoreSaveState(
    std::span<const uint8_t> input,
    std::string *error)
{
    struct SavedSifOperation
    {
        HleSifDmaSubmission submission{};
        uint64_t operationGeneration = 0u;
        uint64_t iopCycles = 0u;
    };

    Reader reader(input);
    uint32_t version = 0u;
    R5900Context cpu{};
    ps2x::timing::Cop0TimingSnapshot cop0{};
    uint64_t timeline = 0u;
    ps2x::timing::EeEventSchedulerSnapshot events{};
    ps2x::ee::EeThreadSchedulerSnapshot scheduler{};
    int nextThreadId = 0;
    uint32_t nextGeneration = 0u;
    int currentThreadId = 0;
    uint32_t threadCount = 0u;
    if (!reader.u32(version) || version != kRuntimeStateVersion ||
        !readR5900Context(reader, cpu) ||
        !readCop0Timing(reader, cop0) ||
        !reader.u64(timeline) ||
        !readEventScheduler(reader, events) ||
        !readThreadScheduler(reader, scheduler) ||
        !reader.i32(nextThreadId) ||
        !reader.u32(nextGeneration) ||
        !reader.i32(currentThreadId) ||
        !reader.u32(threadCount) ||
        threadCount > kMaximumSavedThreads)
    {
        setError(error, "invalid runtime core save-state header");
        return false;
    }
    std::vector<SavedThread> threads(threadCount);
    std::set<int> threadIds;
    for (SavedThread &thread : threads)
    {
        if (!readSavedThread(reader, thread) ||
            !threadIds.insert(thread.id).second)
        {
            setError(error, "invalid or duplicate saved EE thread");
            return false;
        }
        recoverNativeWaitContinuation(thread);
    }

    // HLE-only wait objects (currently MPEG worker semaphores) own native
    // decoder state which v1 intentionally starts cold. Let those callers
    // return successfully on load instead of preserving a wait which no
    // reconstructed host object could ever signal.
    for (auto &scheduled : scheduler.threads)
    {
        const bool hleWait =
            scheduled.wait.kind ==
                ps2x::ee::EeSchedulerWaitKind::HleSemaphore ||
            scheduled.completedWait.wait.kind ==
                ps2x::ee::EeSchedulerWaitKind::HleSemaphore;
        if (!hleWait)
            continue;
        const auto found = std::find_if(
            threads.begin(), threads.end(),
            [&](const SavedThread &thread)
            { return thread.id == scheduled.id; });
        if (found == threads.end())
        {
            setError(error, "HLE wait has no saved EE thread");
            return false;
        }
        scheduled.state =
            scheduled.suspendCount != 0u
                ? ps2x::ee::EeSchedulerThreadState::Suspended
                : ps2x::ee::EeSchedulerThreadState::Ready;
        scheduled.wait = {};
        scheduled.completedWait = {};
        scheduled.retainCompletedWaitReason = false;
        found->guest.status =
            found->guest.suspendCount != 0
                ? EeThreadGuestState::kSuspend
                : EeThreadGuestState::kReady;
        found->guest.waitType = EeThreadGuestState::kWaitNone;
        found->guest.waitId = 0;
        found->guest.waitQueue = EeThreadWaitQueue::None;
        found->guest.waitQueueId = 0;
        found->guest.waitCompletionPending = false;
        found->pendingWait =
            ps2x::ee::EeSchedulerWaitCompletion::None;
        found->pendingEventFlagResultBits = 0u;
        EeThreadGuestState guestValidator;
        if (!guestValidator.restore(found->guest))
        {
            setError(error, "failed to recover an HLE-waiting thread");
            return false;
        }
    }
    {
        ps2x::ee::EeThreadScheduler schedulerValidator;
        if (!schedulerValidator.restore(scheduler))
        {
            setError(error, "failed to normalize restored EE waits");
            return false;
        }
    }
    if (threadIds.find(1) == threadIds.end())
    {
        setError(error, "save state has no main EE thread");
        return false;
    }
    for (const auto &thread : scheduler.threads)
    {
        const auto found = std::find_if(
            threads.begin(), threads.end(),
            [&](const SavedThread &saved)
            {
                return saved.id == thread.id &&
                       saved.generation == thread.generation;
            });
        if (found == threads.end())
        {
            setError(error,
                     "EE scheduler and thread registry do not match");
            return false;
        }
    }

    uint64_t vu1Generation = 0u;
    uint64_t vu1LastTick = 0u;
    uint64_t vu1AdvancedCycles = 0u;
    ps2x::timing::EeEventToken vu1Token{};
    ps2x::timing::EeEventToken vif0VuToken{};
    ps2x::timing::EeEventToken vif0DmaToken{};
    ps2x::timing::EeEventToken vif1DmaToken{};
    ps2x::timing::EeEventToken gifDmaToken{};
    ps2x::timing::EeEventToken fromIpuToken{};
    ps2x::timing::EeEventToken toIpuToken{};
    ps2x::timing::EeEventToken sifToken{};
    ps2x::timing::EeEventToken fromScratchToken{};
    ps2x::timing::EeEventToken toScratchToken{};
    if (!reader.u64(vu1Generation) ||
        !reader.u64(vu1LastTick) ||
        !reader.u64(vu1AdvancedCycles) ||
        !readEventToken(reader, vu1Token) ||
        !readEventToken(reader, vif0VuToken) ||
        !readEventToken(reader, vif0DmaToken) ||
        !readEventToken(reader, vif1DmaToken) ||
        !readEventToken(reader, gifDmaToken) ||
        !readEventToken(reader, fromIpuToken) ||
        !readEventToken(reader, toIpuToken) ||
        !readEventToken(reader, sifToken) ||
        !readEventToken(reader, fromScratchToken) ||
        !readEventToken(reader, toScratchToken))
    {
        setError(error, "invalid runtime event-token state");
        return false;
    }

    uint32_t sifOperationCount = 0u;
    if (!reader.u32(sifOperationCount) ||
        sifOperationCount > kHleSifDmaQueueCapacity)
    {
        setError(error, "invalid HLE SIF DMA queue size");
        return false;
    }
    std::vector<SavedSifOperation> sifOperations(sifOperationCount);
    for (SavedSifOperation &operation : sifOperations)
    {
        if (!reader.u32(operation.submission.transferId) ||
            !reader.u32(operation.submission.transferCount) ||
            operation.submission.transferCount >
                kMaximumHleSifDmaTransfers)
        {
            setError(error, "invalid HLE SIF DMA operation");
            return false;
        }
        for (uint32_t index = 0u;
             index < operation.submission.transferCount;
             ++index)
        {
            HleSifDmaTransfer &transfer =
                operation.submission.transfers[index];
            if (!reader.u32(transfer.src) ||
                !reader.u32(transfer.dest) ||
                !reader.u32(transfer.size) ||
                !reader.u32(transfer.iopOffset) ||
                !readEnum8(
                    reader, transfer.destination,
                    static_cast<uint8_t>(
                        HleSifDmaDestination::SyntheticGuest)))
            {
                setError(error, "invalid HLE SIF DMA transfer");
                return false;
            }
        }
        if (!reader.u64(operation.operationGeneration) ||
            !reader.u64(operation.iopCycles))
        {
            setError(error, "truncated HLE SIF DMA operation");
            return false;
        }
    }
    uint64_t sifOperationGeneration = 0u;
    if (!reader.u64(sifOperationGeneration))
    {
        setError(error, "truncated HLE SIF generation");
        return false;
    }

    EeVSyncTiming vsync{};
    uint64_t vsyncStartTick = 0u;
    if (!readEnum8(
            reader, vsync.phase,
            static_cast<uint8_t>(EeVSyncPhase::End)) ||
        !reader.u64(vsyncStartTick) ||
        !readEventToken(reader, vsync.eventToken) ||
        !reader.u64(vsync.durations.periodCycles) ||
        !reader.u64(vsync.durations.renderCycles) ||
        !reader.u64(vsync.durations.blankCycles) ||
        !reader.u64(vsync.durations.gsBlankCycles) ||
        !reader.u32(vsync.durations.hRenderCycles) ||
        !reader.u32(vsync.durations.hBlankCycles) ||
        !reader.u32(vsync.durations.hSyncErrorCycles) ||
        !reader.u64(vsync.currentVSyncTick) ||
        !reader.u32(vsync.videoMode) ||
        !readEnum8(
            reader, vsync.videoModeClass,
            static_cast<uint8_t>(EeVSyncVideoModeClass::DvdPal)) ||
        !reader.boolean(vsync.videoModeConfigured) ||
        !reader.boolean(vsync.interlaced) ||
        !reader.boolean(vsync.enabled))
    {
        setError(error, "invalid VSync state");
        return false;
    }
    vsync.startTick = ps2x::timing::eeTickFromRaw(vsyncStartTick);
    ps2x::timing::EeEventToken counterToken{};
    ps2x::timing::EeEventToken cop0TimerToken{};
    ps2x::timing::EeEventToken cop0PerformanceToken{};
    uint32_t pendingCounterInterrupts = 0u;
    if (!readEventToken(reader, counterToken) ||
        !readEventToken(reader, cop0TimerToken) ||
        !readEventToken(reader, cop0PerformanceToken) ||
        !reader.u32(pendingCounterInterrupts))
    {
        setError(error, "invalid counter event state");
        return false;
    }

    uint32_t pendingVsyncCount = 0u;
    if (!reader.u32(pendingVsyncCount) ||
        pendingVsyncCount > kMaximumPendingVSyncDeliveries)
    {
        setError(error, "invalid pending VSync delivery count");
        return false;
    }
    std::array<PendingVSyncDelivery,
               kMaximumPendingVSyncDeliveries>
        pendingVsync{};
    for (uint32_t index = 0u; index < pendingVsyncCount; ++index)
    {
        if (!readEnum8(
                reader, pendingVsync[index].kind,
                static_cast<uint8_t>(PendingVSyncKind::End)) ||
            !reader.u64(pendingVsync[index].tick) ||
            !reader.u64(pendingVsync[index].eventSequence))
        {
            setError(error, "invalid pending VSync delivery");
            return false;
        }
    }
    uint64_t vu0Tick = 0u;
    uint32_t pendingDmacCount = 0u;
    if (!reader.u64(vu0Tick) ||
        !reader.u32(pendingDmacCount) ||
        pendingDmacCount > 4096u)
    {
        setError(error, "invalid pending DMAC state");
        return false;
    }
    std::vector<DmacPendingInterrupt> pendingDmac(pendingDmacCount);
    for (DmacPendingInterrupt &interrupt : pendingDmac)
    {
        if (!readEnum8(
                reader, interrupt.source,
                static_cast<uint8_t>(DmacChannel::Count) - 1u) ||
            !reader.u32(interrupt.cause) ||
            !reader.u64(interrupt.channelGeneration) ||
            !reader.u64(interrupt.eventSequence) ||
            !reader.u64(interrupt.publicationSequence))
        {
            setError(error, "invalid pending DMAC interrupt");
            return false;
        }
    }
    bool dmacCompletionReady = false;
    bool dmacInterruptDirty = false;
    if (!reader.boolean(dmacCompletionReady) ||
        !reader.boolean(dmacInterruptDirty))
    {
        setError(error, "truncated DMAC publication state");
        return false;
    }

    uint32_t heapBase = 0u;
    uint32_t heapEnd = 0u;
    uint32_t heapLimit = 0u;
    uint32_t heapSuggestedBase = 0u;
    bool heapConfigured = false;
    uint32_t heapBlockCount = 0u;
    if (!reader.u32(heapBase) || !reader.u32(heapEnd) ||
        !reader.u32(heapLimit) ||
        !reader.u32(heapSuggestedBase) ||
        !reader.boolean(heapConfigured) ||
        !reader.u32(heapBlockCount) ||
        heapBlockCount > kMaximumSavedMapEntries)
    {
        setError(error, "invalid guest heap state");
        return false;
    }
    std::vector<GuestHeapBlock> heapBlocks(heapBlockCount);
    for (GuestHeapBlock &block : heapBlocks)
    {
        if (!reader.u32(block.addr) ||
            !reader.u32(block.size) ||
            !reader.boolean(block.free))
        {
            setError(error, "invalid guest heap block");
            return false;
        }
    }
    if (!reader.atEnd())
    {
        setError(error, "runtime core chunk has trailing data");
        return false;
    }

    // All variable-size and enum validation is complete. Apply the cold
    // architectural image; no native EE continuation exists yet.
    m_cpuContext = cpu;
    m_cop0Timing.restore(cop0);
    m_eeTimeline.restore(ps2x::timing::eeTickFromRaw(timeline));
    if (!m_eeEventScheduler.restore(events))
    {
        setError(error, "failed to restore EE event scheduler");
        return false;
    }

    EeThreadRuntimeState &runtimeState = *m_eeThreadRuntimeState;
    {
        std::lock_guard<std::mutex> mapLock(runtimeState.threadMapMutex);
        runtimeState.threads.clear();
        runtimeState.contextThreadIds.clear();
        runtimeState.nextThreadId = nextThreadId;
        runtimeState.nextGeneration = nextGeneration;
        for (SavedThread &saved : threads)
        {
            auto info = std::make_shared<ThreadInfo>();
            info->entry = saved.entry;
            info->stack = saved.stack;
            info->stackSize = saved.stackSize;
            info->gp = saved.gp;
            info->priority = saved.priority;
            info->attr = saved.attr;
            info->option = saved.option;
            info->arg = saved.arg;
            info->generation = saved.generation;
            info->ownsStack = saved.ownsStack;
            info->tlsBase = saved.tlsBase;
            if (!info->guestState.restore(saved.guest))
            {
                setError(error, "failed to restore EE guest-thread state");
                return false;
            }
            if (saved.id != 1)
                info->context = saved.context;
            info->boundContext =
                saved.id == 1 ? &m_cpuContext : &info->context;
            info->currentPc.store(
                saved.currentPc, std::memory_order_relaxed);
            info->pendingWaitCompletion = saved.pendingWait;
            info->pendingEventFlagResultBits =
                saved.pendingEventFlagResultBits;
            info->legacyAdmissionPending = false;
            info->forceRelease.store(false, std::memory_order_relaxed);
            info->terminated.store(false, std::memory_order_relaxed);
            runtimeState.contextThreadIds[info->boundContext] = saved.id;
            runtimeState.threads.emplace(saved.id, std::move(info));
        }
    }
    for (auto &queue : runtimeState.legacyReadyQueues)
        queue.clear();
    runtimeState.currentThreadId.store(
        currentThreadId, std::memory_order_release);
    runtimeState.activeHostThreads.store(0, std::memory_order_release);
    m_pendingRestoredEeScheduler = std::move(scheduler);
    m_boundEeContext = nullptr;
    m_boundEeThreadId = currentThreadId;

    m_vu1ExecutionTiming.generation = vu1Generation;
    m_vu1ExecutionTiming.lastAdvancedTick =
        ps2x::timing::eeTickFromRaw(vu1LastTick);
    m_vu1ExecutionTiming.totalAdvancedCycles = vu1AdvancedCycles;
    m_vu1ExecutionTiming.eventToken = vu1Token;
    m_vif0VuFinishEventToken = vif0VuToken;
    m_vif0DmaEventToken = vif0DmaToken;
    m_vif1DmaEventToken = vif1DmaToken;
    m_gifDmaEventToken = gifDmaToken;
    m_fromIpuDmaEventToken = fromIpuToken;
    m_toIpuDmaEventToken = toIpuToken;
    m_hleSifDmaEventToken = sifToken;
    m_fromScratchpadDmaEventToken = fromScratchToken;
    m_toScratchpadDmaEventToken = toScratchToken;
    m_hleSifDmaQueue = {};
    m_hleSifDmaQueueHead = 0u;
    m_hleSifDmaQueueCount = sifOperations.size();
    for (size_t index = 0u; index < sifOperations.size(); ++index)
    {
        m_hleSifDmaQueue[index].submission =
            sifOperations[index].submission;
        m_hleSifDmaQueue[index].operationGeneration =
            sifOperations[index].operationGeneration;
        m_hleSifDmaQueue[index].iopCycles =
            sifOperations[index].iopCycles;
    }
    m_hleSifDmaOperationGeneration = sifOperationGeneration;
    m_eeVSyncTiming = vsync;
    m_eeCounterEventToken = counterToken;
    m_cop0TimerEventToken = cop0TimerToken;
    m_cop0PerformanceEventToken = cop0PerformanceToken;
    m_pendingEeCounterInterrupts = pendingCounterInterrupts;
    m_pendingVSyncDeliveries = pendingVsync;
    m_pendingVSyncDeliveryCount = pendingVsyncCount;
    m_vu0CycleTick = ps2x::timing::eeTickFromRaw(vu0Tick);
    m_pendingDmacInterrupts = std::move(pendingDmac);
    m_dmacCompletionReady.store(
        dmacCompletionReady, std::memory_order_release);
    m_dmacInterruptDeliveryDirty.store(
        dmacInterruptDirty, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(m_guestHeapMutex);
        m_guestHeapBase = heapBase;
        m_guestHeapEnd = heapEnd;
        m_guestHeapLimit = heapLimit;
        m_guestHeapSuggestedBase = heapSuggestedBase;
        m_guestHeapConfigured = heapConfigured;
        m_guestHeapBlocks = std::move(heapBlocks);
    }
    m_eeVSyncPacing.hostDeadline = {};
    m_eeVSyncPacing.eeDeadlineTick = {};
    m_eeVSyncPacing.anchored = false;
    m_eeExecutorInterruptPassComplete = false;
    publishCop0TimingMirrors(&m_cpuContext);
    return true;
}

bool PS2Runtime::encodeRuntimeHleSaveState(
    std::vector<uint8_t> &output,
    std::string *error)
{
    Writer writer;
    writer.u32(kRuntimeStateVersion);

    const auto sortedIntegerEntries = []<typename Map>(const Map &map)
    {
        using Entry = typename Map::value_type;
        std::vector<const Entry *> entries;
        entries.reserve(map.size());
        for (const Entry &entry : map)
            entries.push_back(&entry);
        std::sort(
            entries.begin(), entries.end(),
            [](const Entry *left, const Entry *right)
            { return left->first < right->first; });
        return entries;
    };

    EeSyncRuntimeState &sync = *m_eeSyncRuntimeState;
    {
        std::lock_guard<std::mutex> lock(sync.semaMapMutex);
        if (sync.semas.size() > kMaximumSavedObjects)
        {
            setError(error, "too many saved semaphores");
            return false;
        }
        writer.i32(sync.nextSemaId);
        writer.u32(static_cast<uint32_t>(sync.semas.size()));
        for (const auto *entry : sortedIntegerEntries(sync.semas))
        {
            writer.i32(entry->first);
            const std::shared_ptr<SemaInfo> &info = entry->second;
            std::lock_guard<std::mutex> objectLock(info->m);
            writer.i32(info->count);
            writer.i32(info->maxCount);
            writer.i32(info->initCount);
            writer.u32(info->attr);
            writer.u32(info->option);
            writer.i32(info->waiters);
            writer.boolean(info->deleted);
        }
    }
    {
        std::lock_guard<std::mutex> lock(sync.eventFlagMapMutex);
        if (sync.eventFlags.size() > kMaximumSavedObjects)
        {
            setError(error, "too many saved event flags");
            return false;
        }
        writer.i32(sync.nextEventFlagId);
        writer.u32(static_cast<uint32_t>(sync.eventFlags.size()));
        for (const auto *entry : sortedIntegerEntries(sync.eventFlags))
        {
            writer.i32(entry->first);
            const std::shared_ptr<EventFlagInfo> &info = entry->second;
            std::lock_guard<std::mutex> objectLock(info->m);
            writer.u32(info->attr);
            writer.u32(info->option);
            writer.u32(info->initBits);
            writer.u32(info->bits);
            writer.i32(info->waiters);
            writer.boolean(info->deleted);
            writer.u32(static_cast<uint32_t>(
                info->schedulerWaiters.size()));
            for (const auto *waiter :
                 sortedIntegerEntries(info->schedulerWaiters))
            {
                writer.i32(waiter->first);
                writer.u32(waiter->second.generation);
                writer.u32(waiter->second.waitBits);
                writer.u32(waiter->second.mode);
            }
            writer.u32(static_cast<uint32_t>(
                info->legacyWaitOrder.size()));
            for (int id : info->legacyWaitOrder)
                writer.i32(id);
        }
    }

    EeKernelRuntimeState &kernel = *m_eeKernelRuntimeState;
    writer.u32(kernel.threadRootFunction.load(
        std::memory_order_acquire));
    {
        std::lock_guard<std::mutex> lock(kernel.exitHandlerMutex);
        writer.u32(static_cast<uint32_t>(kernel.exitHandlers.size()));
        for (const auto *entry : sortedIntegerEntries(kernel.exitHandlers))
        {
            writer.i32(entry->first);
            writer.u32(static_cast<uint32_t>(entry->second.size()));
            for (const ExitHandlerEntry &handler : entry->second)
            {
                writer.u32(handler.func);
                writer.u32(handler.arg);
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(kernel.bootModeMutex);
        writer.boolean(kernel.bootModeInitialized);
        writer.u32(kernel.bootModePoolOffset);
        writer.u32(static_cast<uint32_t>(
            kernel.bootModeAddresses.size()));
        std::vector<std::pair<uint8_t, uint32_t>> entries(
            kernel.bootModeAddresses.begin(),
            kernel.bootModeAddresses.end());
        std::sort(entries.begin(), entries.end());
        for (const auto &[id, address] : entries)
        {
            writer.u8(id);
            writer.u32(address);
        }
    }
    {
        std::lock_guard<std::mutex> lock(kernel.syscallOverrideMutex);
        writer.u32(static_cast<uint32_t>(
            kernel.syscallOverrides.size()));
        for (const auto *entry :
             sortedIntegerEntries(kernel.syscallOverrides))
        {
            writer.u32(entry->first);
            writer.u32(entry->second);
        }
        std::vector<uint32_t> mirrors(
            kernel.syscallMirrorAddresses.begin(),
            kernel.syscallMirrorAddresses.end());
        std::sort(mirrors.begin(), mirrors.end());
        writer.u32(static_cast<uint32_t>(mirrors.size()));
        for (uint32_t address : mirrors)
            writer.u32(address);
    }
    {
        std::lock_guard<std::mutex> lock(
            kernel.activeSyscallOverrideMutex);
        writer.u32(static_cast<uint32_t>(
            kernel.activeSyscallOverrides.size()));
        for (uint32_t address : kernel.activeSyscallOverrides)
            writer.u32(address);
    }
    {
        std::lock_guard<std::mutex> lock(kernel.tlsMutex);
        writer.u32(kernel.tlsIndex);
    }
    {
        std::lock_guard<std::mutex> lock(kernel.osdMutex);
        writer.boolean(kernel.osdConfigInitialized);
        writer.u32(kernel.osdConfigRaw);
        writer.u32(kernel.osdConfig2Raw);
    }

    EeInterruptRuntimeState &interrupt = *m_eeInterruptRuntimeState;
    {
        std::lock_guard<std::mutex> lock(interrupt.handlerMutex);
        const auto writeHandlers = [&](const auto &handlers)
        {
            writer.u32(static_cast<uint32_t>(handlers.size()));
            for (const auto *entry : sortedIntegerEntries(handlers))
            {
                const IrqHandlerInfo &handler = entry->second;
                writer.i32(entry->first);
                writer.i32(handler.id);
                writer.u32(handler.cause);
                writer.u32(handler.handler);
                writer.u32(handler.arg);
                writer.u32(handler.gp);
                writer.u32(handler.sp);
                writer.boolean(handler.enabled);
                writer.i32(handler.order);
            }
        };
        writeHandlers(interrupt.intcHandlers);
        writeHandlers(interrupt.dmacHandlers);
        writer.i32(interrupt.nextIntcHandlerId);
        writer.i32(interrupt.nextDmacHandlerId);
        writer.i32(interrupt.intcHeadOrder);
        writer.i32(interrupt.intcTailOrder);
        writer.i32(interrupt.dmacHeadOrder);
        writer.i32(interrupt.dmacTailOrder);
        writer.u32(interrupt.enabledIntcMask);
        writer.u32(interrupt.enabledDmacMask);
    }
    {
        std::lock_guard<std::mutex> lock(interrupt.vsyncMutex);
        writer.u32(interrupt.vsyncRegistration.flagAddr);
        writer.u32(interrupt.vsyncRegistration.tickAddr);
        writer.u64(interrupt.vsyncTick);
        writer.u64(interrupt.gsSyncVBaseTick);
        writer.u8(interrupt.gsVideoParameters.interlace);
        writer.u8(interrupt.gsVideoParameters.outputMode);
        writer.u8(interrupt.gsVideoParameters.frameMode);
        writer.u8(interrupt.gsVideoParameters.version);
        writer.u32(interrupt.gsSyncVCallback);
        writer.u32(interrupt.gsSyncVCallbackGp);
        writer.u32(interrupt.gsSyncVCallbackSp);
    }

    EeRpcRuntimeState &rpc = *m_eeRpcRuntimeState;
    {
        std::lock_guard<std::mutex> lock(rpc.rpcMutex);
        writer.boolean(rpc.initialized);
        writer.u32(static_cast<uint32_t>(rpc.servers.size()));
        for (const auto *entry : sortedIntegerEntries(rpc.servers))
        {
            writer.u32(entry->first);
            writer.u32(entry->second.sid);
            writer.u32(entry->second.sd_ptr);
            writer.boolean(entry->second.guestBacked);
        }
        writer.u32(static_cast<uint32_t>(rpc.clients.size()));
        for (const auto *entry : sortedIntegerEntries(rpc.clients))
        {
            writer.u32(entry->first);
            writer.boolean(entry->second.busy);
            writer.u32(entry->second.last_rpc);
            writer.u32(entry->second.sid);
        }
        writer.u64(rpc.latestClientBusyState.load(
            std::memory_order_acquire));
        writer.u64(rpc.nextDebugSequence);
        writer.u32(rpc.nextRequestId);
        writer.u32(rpc.packetIndex);
        writer.u32(rpc.serverIndex);
        writer.u32(rpc.activeQueue);
    }
    {
        std::lock_guard<std::mutex> lock(rpc.moduleMutex);
        writer.i32(rpc.nextModuleId);
        writer.u32(static_cast<uint32_t>(rpc.modulesById.size()));
        for (const auto *entry : sortedIntegerEntries(rpc.modulesById))
        {
            const SifModuleRecord &module = entry->second;
            writer.i32(entry->first);
            writer.i32(module.id);
            writer.string(module.path);
            writer.string(module.pathKey);
            writer.u32(module.refCount);
            writer.boolean(module.loaded);
        }
    }

    Deci2RuntimeState &deci = *m_deci2RuntimeState;
    {
        std::lock_guard<std::mutex> lock(deci.sessionMutex);
        writer.i32(deci.nextSocket);
        writer.u32(static_cast<uint32_t>(deci.sessions.size()));
        for (const auto *entry : sortedIntegerEntries(deci.sessions))
        {
            writer.i32(entry->first);
            writer.u16(entry->second.protocol);
            writer.u32(entry->second.opt);
            writer.u32(entry->second.handler);
            writer.u32(entry->second.userArea);
            writer.boolean(entry->second.locked);
        }
    }

    ps2_stubs::AudioRuntimeState &audio = *m_audioRuntimeState;
    {
        std::lock_guard<std::mutex> lock(audio.mutex);
        writer.boolean(audio.initialized);
        for (const auto &transfer : audio.voiceTransfers)
        {
            writer.u32(transfer.sourceAddress);
            writer.u32(transfer.destinationAddress);
            writer.u32(transfer.size);
            writer.u16(transfer.mode);
            writer.boolean(transfer.completed);
        }
        for (const auto &transfer : audio.blockTransfers)
        {
            writer.u32(transfer.base);
            writer.u32(transfer.size);
            writer.u32(transfer.pauseBase);
            writer.u32(transfer.offset);
            writer.u32(transfer.statusTraceCount);
            writer.u16(transfer.mode);
            writer.boolean(transfer.active);
            writer.boolean(transfer.loop);
        }
    }

    ps2_stubs::CdRuntimeState &cd = *m_cdRuntimeState;
    {
        std::lock_guard<std::mutex> lock(cd.mutex);
        writer.u32(cd.nextPseudoLbn);
        writer.i32(cd.lastError);
        writer.u32(cd.mode);
        writer.u32(cd.streamingLbn);
        writer.u32(cd.streamingEndLbn);
        writer.boolean(cd.initialized);
        writer.string(cd.lastFailedPath);
        writer.u32(cd.samePathFailCount);
    }

    ps2_stubs::DmaRuntimeState &dma = *m_dmaRuntimeState;
    {
        std::lock_guard<std::mutex> lock(dma.environmentMutex);
        writer.u8(dma.currentEnvironment.sts);
        writer.u8(dma.currentEnvironment.std);
        writer.u8(dma.currentEnvironment.mfd);
        writer.u8(dma.currentEnvironment.rele);
        writer.u32(dma.currentEnvironment.pcr);
        writer.u32(dma.currentEnvironment.sqwc);
        writer.u32(dma.currentEnvironment.rbor);
        writer.u32(dma.currentEnvironment.rbsr);
    }
    {
        std::lock_guard<std::mutex> lock(dma.stubMutex);
        writer.u32(static_cast<uint32_t>(dma.pendingPolls.size()));
        for (const auto *entry : sortedIntegerEntries(dma.pendingPolls))
        {
            writer.u32(entry->first);
            writer.u32(entry->second);
        }
    }

    ps2_stubs::LibCRuntimeState &libc = *m_libcRuntimeState;
    {
        std::lock_guard<std::mutex> lock(libc.mutex);
        writer.u32(libc.nextFileHandle);
        writer.u32(libc.randomSeed);
    }

    ps2_stubs::MemoryCardRuntimeState &memoryCard =
        *m_memoryCardRuntimeState;
    {
        std::lock_guard<std::mutex> lock(memoryCard.mutex);
        writer.i32(memoryCard.nextFd);
        writer.i32(memoryCard.lastCmd);
        writer.boolean(memoryCard.commandPending);
        writer.i32(memoryCard.lastResult);
        for (const auto &port : memoryCard.ports)
        {
            writer.string(port.currentDir);
            writer.boolean(port.formatted);
        }
        writer.i32(memoryCard.cvFileCursor);
    }

    ps2_stubs::PadRuntimeState &pad = *m_padRuntimeState;
    {
        std::lock_guard<std::mutex> lock(pad.mutex);
        for (const auto &port : pad.ports)
        {
            writer.boolean(port.open);
            writer.boolean(port.analogMode);
            writer.boolean(port.pressureEnabled);
            writer.boolean(port.lastUsedOverride);
            writer.boolean(port.lastUsedBackend);
            writer.boolean(port.lastReadOk);
            writer.u16(port.buttonMask);
            writer.u32(port.dmaAddr);
            writer.u32(port.reqState);
            writer.u32(port.transientState);
            writer.u16(port.lastInput.buttons);
            writer.u8(port.lastInput.rx);
            writer.u8(port.lastInput.ry);
            writer.u8(port.lastInput.lx);
            writer.u8(port.lastInput.ly);
            writer.bytes(port.lastData);
            writer.u32(port.readCount);
            writer.u32(port.lastReadDataAddr);
        }
        writer.u32(pad.frameCount.load(std::memory_order_relaxed));
    }

    ps2_stubs::SifRuntimeState &sif = *m_sifRuntimeState;
    {
        std::lock_guard<std::mutex> lock(sif.dmaTransferMutex);
        writer.u32(sif.nextDmaTransferId);
    }
    {
        std::lock_guard<std::mutex> lock(sif.commandMutex);
        const auto writeMap = [&](const auto &map)
        {
            writer.u32(static_cast<uint32_t>(map.size()));
            for (const auto *entry : sortedIntegerEntries(map))
            {
                writer.u32(entry->first);
                writer.u32(entry->second);
            }
        };
        writeMap(sif.regs);
        writeMap(sif.sregs);
        writeMap(sif.cmdHandlers);
        writer.u32(sif.cmdBuffer);
        writer.u32(sif.sysCmdBuffer);
        writer.boolean(sif.cmdInitialized);
    }
    {
        std::lock_guard<std::mutex> lock(sif.heapMutex);
        writer.u32(static_cast<uint32_t>(sif.heapAllocations.size()));
        for (const auto &[address, size] : sif.heapAllocations)
        {
            writer.u32(address);
            writer.u32(size);
        }
        writer.u32(sif.iopHeapNext);
    }

    EeFileRuntimeState &files = *m_eeFileRuntimeState;
    {
        std::lock_guard<std::mutex> lock(files.vagMutex);
        writer.u32(static_cast<uint32_t>(files.vagAccum.size()));
        for (const auto *entry : sortedIntegerEntries(files.vagAccum))
        {
            writer.i32(entry->first);
            writer.u32(entry->second.firstBufAddr);
            writer.sizedBytes(entry->second.data);
        }
    }
    {
        std::lock_guard<std::mutex> lock(files.pathMutex);
        writer.boolean(files.pathsInitialized);
        writer.string(files.hostBase.generic_string());
        writer.string(files.cdromBase.generic_string());
        writer.string(files.mcBase.generic_string());
        writer.string(files.hostCwd.generic_string());
        writer.string(files.cdromCwd.generic_string());
        writer.string(files.mcCwd.generic_string());
        writer.string(files.cwdDevice);
    }

    output = writer.take();
    return true;
}

bool PS2Runtime::decodeRuntimeHleSaveState(
    std::span<const uint8_t> input,
    std::string *error)
{
    Reader reader(input);
    uint32_t version = 0u;
    if (!reader.u32(version) || version != kRuntimeStateVersion)
    {
        setError(error, "unsupported HLE save-state version");
        return false;
    }
    const auto readCount = [&](uint32_t &count,
                               uint32_t maximum = kMaximumSavedObjects)
    {
        return reader.u32(count) && count <= maximum;
    };

    EeSyncRuntimeState &sync = *m_eeSyncRuntimeState;
    int nextSemaId = 0;
    uint32_t count = 0u;
    if (!reader.i32(nextSemaId) || !readCount(count))
    {
        setError(error, "invalid saved semaphore registry");
        return false;
    }
    std::unordered_map<int, std::shared_ptr<SemaInfo>> semas;
    for (uint32_t index = 0u; index < count; ++index)
    {
        int id = 0;
        auto info = std::make_shared<SemaInfo>();
        if (!reader.i32(id) || !reader.i32(info->count) ||
            !reader.i32(info->maxCount) ||
            !reader.i32(info->initCount) ||
            !reader.u32(info->attr) || !reader.u32(info->option) ||
            !reader.i32(info->waiters) ||
            !reader.boolean(info->deleted) ||
            !semas.emplace(id, info).second)
        {
            setError(error, "invalid saved semaphore");
            return false;
        }
    }
    int nextEventFlagId = 0;
    if (!reader.i32(nextEventFlagId) || !readCount(count))
    {
        setError(error, "invalid saved event-flag registry");
        return false;
    }
    std::unordered_map<int, std::shared_ptr<EventFlagInfo>> eventFlags;
    for (uint32_t index = 0u; index < count; ++index)
    {
        int id = 0;
        auto info = std::make_shared<EventFlagInfo>();
        uint32_t waiterCount = 0u;
        if (!reader.i32(id) || !reader.u32(info->attr) ||
            !reader.u32(info->option) ||
            !reader.u32(info->initBits) || !reader.u32(info->bits) ||
            !reader.i32(info->waiters) ||
            !reader.boolean(info->deleted) ||
            !readCount(waiterCount, kMaximumSavedThreads))
        {
            setError(error, "invalid saved event flag");
            return false;
        }
        for (uint32_t waiterIndex = 0u;
             waiterIndex < waiterCount; ++waiterIndex)
        {
            int threadId = 0;
            EventFlagInfo::SchedulerWaiter waiter{};
            if (!reader.i32(threadId) ||
                !reader.u32(waiter.generation) ||
                !reader.u32(waiter.waitBits) ||
                !reader.u32(waiter.mode) ||
                !info->schedulerWaiters.emplace(
                    threadId, waiter).second)
            {
                setError(error, "invalid saved event-flag waiter");
                return false;
            }
        }
        uint32_t legacyCount = 0u;
        if (!readCount(legacyCount, kMaximumSavedThreads))
        {
            setError(error, "invalid legacy event wait order");
            return false;
        }
        info->legacyWaitOrder.resize(legacyCount);
        for (int &threadId : info->legacyWaitOrder)
        {
            if (!reader.i32(threadId))
            {
                setError(error, "truncated legacy event wait order");
                return false;
            }
        }
        if (!eventFlags.emplace(id, info).second)
        {
            setError(error, "duplicate saved event flag");
            return false;
        }
    }

    EeKernelRuntimeState &kernel = *m_eeKernelRuntimeState;
    uint32_t threadRoot = 0u;
    if (!reader.u32(threadRoot) || !readCount(count))
    {
        setError(error, "invalid saved kernel state");
        return false;
    }
    std::unordered_map<int, std::vector<ExitHandlerEntry>> exitHandlers;
    for (uint32_t index = 0u; index < count; ++index)
    {
        int threadId = 0;
        uint32_t handlerCount = 0u;
        if (!reader.i32(threadId) ||
            !readCount(handlerCount, kMaximumSavedObjects))
        {
            setError(error, "invalid saved exit-handler list");
            return false;
        }
        auto &handlers = exitHandlers[threadId];
        handlers.resize(handlerCount);
        for (ExitHandlerEntry &handler : handlers)
        {
            if (!reader.u32(handler.func) || !reader.u32(handler.arg))
            {
                setError(error, "truncated saved exit handler");
                return false;
            }
        }
    }
    bool bootInitialized = false;
    uint32_t bootPoolOffset = 0u;
    if (!reader.boolean(bootInitialized) ||
        !reader.u32(bootPoolOffset) || !readCount(count, 256u))
    {
        setError(error, "invalid saved boot-mode state");
        return false;
    }
    std::unordered_map<uint8_t, uint32_t> bootAddresses;
    for (uint32_t index = 0u; index < count; ++index)
    {
        uint8_t id = 0u;
        uint32_t address = 0u;
        if (!reader.u8(id) || !reader.u32(address) ||
            !bootAddresses.emplace(id, address).second)
        {
            setError(error, "invalid saved boot-mode entry");
            return false;
        }
    }
    if (!readCount(count))
    {
        setError(error, "invalid syscall-override count");
        return false;
    }
    std::unordered_map<uint32_t, uint32_t> syscallOverrides;
    for (uint32_t index = 0u; index < count; ++index)
    {
        uint32_t address = 0u, target = 0u;
        if (!reader.u32(address) || !reader.u32(target) ||
            !syscallOverrides.emplace(address, target).second)
        {
            setError(error, "invalid syscall override");
            return false;
        }
    }
    if (!readCount(count))
    {
        setError(error, "invalid syscall-mirror count");
        return false;
    }
    std::unordered_set<uint32_t> syscallMirrors;
    for (uint32_t index = 0u; index < count; ++index)
    {
        uint32_t address = 0u;
        if (!reader.u32(address) ||
            !syscallMirrors.insert(address).second)
        {
            setError(error, "invalid syscall mirror");
            return false;
        }
    }
    if (!readCount(count))
    {
        setError(error, "invalid active syscall-override count");
        return false;
    }
    std::vector<uint32_t> activeOverrides(count);
    for (uint32_t &address : activeOverrides)
        if (!reader.u32(address)) return false;
    uint32_t tlsIndex = 0u;
    bool osdInitialized = false;
    uint32_t osdRaw = 0u, osd2Raw = 0u;
    if (!reader.u32(tlsIndex) ||
        !reader.boolean(osdInitialized) ||
        !reader.u32(osdRaw) || !reader.u32(osd2Raw))
    {
        setError(error, "invalid saved kernel configuration");
        return false;
    }

    EeInterruptRuntimeState &interrupt = *m_eeInterruptRuntimeState;
    const auto readHandlers = [&](auto &handlers) -> bool
    {
        uint32_t handlerCount = 0u;
        if (!readCount(handlerCount)) return false;
        for (uint32_t index = 0u; index < handlerCount; ++index)
        {
            int key = 0;
            IrqHandlerInfo handler{};
            if (!reader.i32(key) || !reader.i32(handler.id) ||
                !reader.u32(handler.cause) ||
                !reader.u32(handler.handler) ||
                !reader.u32(handler.arg) || !reader.u32(handler.gp) ||
                !reader.u32(handler.sp) ||
                !reader.boolean(handler.enabled) ||
                !reader.i32(handler.order) ||
                !handlers.emplace(key, handler).second)
            {
                return false;
            }
        }
        return true;
    };
    std::unordered_map<int, IrqHandlerInfo> intcHandlers;
    std::unordered_map<int, IrqHandlerInfo> dmacHandlers;
    int nextIntc = 0, nextDmac = 0, intcHead = 0, intcTail = 0;
    int dmacHead = 0, dmacTail = 0;
    uint32_t intcMask = 0u, dmacMask = 0u;
    if (!readHandlers(intcHandlers) || !readHandlers(dmacHandlers) ||
        !reader.i32(nextIntc) || !reader.i32(nextDmac) ||
        !reader.i32(intcHead) || !reader.i32(intcTail) ||
        !reader.i32(dmacHead) || !reader.i32(dmacTail) ||
        !reader.u32(intcMask) || !reader.u32(dmacMask))
    {
        setError(error, "invalid interrupt-handler state");
        return false;
    }
    EeVSyncFlagRegistration vsyncRegistration{};
    uint64_t vsyncTick = 0u, gsSyncBase = 0u;
    EeGsVideoParameters video{};
    uint32_t gsCallback = 0u, gsCallbackGp = 0u, gsCallbackSp = 0u;
    if (!reader.u32(vsyncRegistration.flagAddr) ||
        !reader.u32(vsyncRegistration.tickAddr) ||
        !reader.u64(vsyncTick) || !reader.u64(gsSyncBase) ||
        !reader.u8(video.interlace) || !reader.u8(video.outputMode) ||
        !reader.u8(video.frameMode) || !reader.u8(video.version) ||
        !reader.u32(gsCallback) || !reader.u32(gsCallbackGp) ||
        !reader.u32(gsCallbackSp))
    {
        setError(error, "invalid interrupt VSync state");
        return false;
    }

    EeRpcRuntimeState &rpc = *m_eeRpcRuntimeState;
    bool rpcInitialized = false;
    if (!reader.boolean(rpcInitialized) || !readCount(count))
    {
        setError(error, "invalid RPC server state");
        return false;
    }
    std::unordered_map<uint32_t, RpcServerState> servers;
    for (uint32_t index = 0u; index < count; ++index)
    {
        uint32_t key = 0u;
        RpcServerState server{};
        if (!reader.u32(key) || !reader.u32(server.sid) ||
            !reader.u32(server.sd_ptr) ||
            !reader.boolean(server.guestBacked) ||
            !servers.emplace(key, server).second)
        {
            setError(error, "invalid saved RPC server");
            return false;
        }
    }
    if (!readCount(count))
    {
        setError(error, "invalid RPC client state");
        return false;
    }
    std::unordered_map<uint32_t, RpcClientState> clients;
    for (uint32_t index = 0u; index < count; ++index)
    {
        uint32_t key = 0u;
        RpcClientState client{};
        if (!reader.u32(key) || !reader.boolean(client.busy) ||
            !reader.u32(client.last_rpc) || !reader.u32(client.sid) ||
            !clients.emplace(key, client).second)
        {
            setError(error, "invalid saved RPC client");
            return false;
        }
    }
    uint64_t latestBusy = 0u, nextDebug = 0u;
    uint32_t nextRequest = 0u, packetIndex = 0u;
    uint32_t serverIndex = 0u, activeQueue = 0u;
    if (!reader.u64(latestBusy) || !reader.u64(nextDebug) ||
        !reader.u32(nextRequest) || !reader.u32(packetIndex) ||
        !reader.u32(serverIndex) || !reader.u32(activeQueue))
    {
        setError(error, "truncated saved RPC state");
        return false;
    }
    int32_t nextModule = 0;
    if (!reader.i32(nextModule) || !readCount(count))
    {
        setError(error, "invalid saved IOP-module state");
        return false;
    }
    std::unordered_map<int32_t, SifModuleRecord> modules;
    std::unordered_map<std::string, int32_t> modulePaths;
    for (uint32_t index = 0u; index < count; ++index)
    {
        int32_t key = 0;
        SifModuleRecord module{};
        if (!reader.i32(key) || !reader.i32(module.id) ||
            !reader.string(module.path, kMaximumSavedString) ||
            !reader.string(module.pathKey, kMaximumSavedString) ||
            !reader.u32(module.refCount) ||
            !reader.boolean(module.loaded) ||
            !modules.emplace(key, module).second)
        {
            setError(error, "invalid saved IOP module");
            return false;
        }
        modulePaths[module.pathKey] = module.id;
    }

    Deci2RuntimeState &deci = *m_deci2RuntimeState;
    int32_t nextSocket = 0;
    if (!reader.i32(nextSocket) || !readCount(count))
    {
        setError(error, "invalid saved DECI2 state");
        return false;
    }
    std::unordered_map<int32_t, Deci2Session> sessions;
    for (uint32_t index = 0u; index < count; ++index)
    {
        int32_t key = 0;
        Deci2Session session{};
        if (!reader.i32(key) || !reader.u16(session.protocol) ||
            !reader.u32(session.opt) || !reader.u32(session.handler) ||
            !reader.u32(session.userArea) ||
            !reader.boolean(session.locked) ||
            !sessions.emplace(key, session).second)
        {
            setError(error, "invalid saved DECI2 session");
            return false;
        }
    }

    bool audioInitialized = false;
    std::array<ps2_stubs::AudioVoiceTransferState, 2u> voice{};
    std::array<ps2_stubs::AudioBlockTransferState, 2u> block{};
    if (!reader.boolean(audioInitialized)) return false;
    for (auto &transfer : voice)
    {
        if (!reader.u32(transfer.sourceAddress) ||
            !reader.u32(transfer.destinationAddress) ||
            !reader.u32(transfer.size) || !reader.u16(transfer.mode) ||
            !reader.boolean(transfer.completed)) return false;
    }
    for (auto &transfer : block)
    {
        if (!reader.u32(transfer.base) || !reader.u32(transfer.size) ||
            !reader.u32(transfer.pauseBase) ||
            !reader.u32(transfer.offset) ||
            !reader.u32(transfer.statusTraceCount) ||
            !reader.u16(transfer.mode) ||
            !reader.boolean(transfer.active) ||
            !reader.boolean(transfer.loop)) return false;
    }

    uint32_t nextPseudoLbn = 0u, cdMode = 0u;
    uint32_t streamLbn = 0u, streamEnd = 0u;
    int32_t cdError = 0;
    bool cdInitialized = false;
    std::string lastFailedPath;
    uint32_t samePathFails = 0u;
    if (!reader.u32(nextPseudoLbn) || !reader.i32(cdError) ||
        !reader.u32(cdMode) || !reader.u32(streamLbn) ||
        !reader.u32(streamEnd) || !reader.boolean(cdInitialized) ||
        !reader.string(lastFailedPath, kMaximumSavedString) ||
        !reader.u32(samePathFails))
    {
        setError(error, "invalid saved CD state");
        return false;
    }

    ps2_stubs::SceDmaEnv dmaEnvironment{};
    if (!reader.u8(dmaEnvironment.sts) ||
        !reader.u8(dmaEnvironment.std) ||
        !reader.u8(dmaEnvironment.mfd) ||
        !reader.u8(dmaEnvironment.rele) ||
        !reader.u32(dmaEnvironment.pcr) ||
        !reader.u32(dmaEnvironment.sqwc) ||
        !reader.u32(dmaEnvironment.rbor) ||
        !reader.u32(dmaEnvironment.rbsr) || !readCount(count))
    {
        setError(error, "invalid saved DMA stub state");
        return false;
    }
    std::unordered_map<uint32_t, uint32_t> dmaPolls;
    for (uint32_t index = 0u; index < count; ++index)
    {
        uint32_t key = 0u, value = 0u;
        if (!reader.u32(key) || !reader.u32(value) ||
            !dmaPolls.emplace(key, value).second) return false;
    }
    uint32_t nextFileHandle = 0u, randomSeed = 0u;
    if (!reader.u32(nextFileHandle) || !reader.u32(randomSeed))
        return false;

    int32_t nextMcFd = 0, lastMcCommand = 0, lastMcResult = 0;
    bool mcPending = false;
    std::array<ps2_stubs::MemoryCardPortState, 2u> mcPorts{};
    int32_t cvCursor = 0;
    if (!reader.i32(nextMcFd) || !reader.i32(lastMcCommand) ||
        !reader.boolean(mcPending) || !reader.i32(lastMcResult))
        return false;
    for (auto &port : mcPorts)
    {
        if (!reader.string(port.currentDir, kMaximumSavedString) ||
            !reader.boolean(port.formatted)) return false;
    }
    if (!reader.i32(cvCursor)) return false;

    std::array<ps2_stubs::PadPortState,
               ps2_stubs::kPadRuntimePortCount> padPorts{};
    for (auto &port : padPorts)
    {
        std::span<const uint8_t> lastData;
        if (!reader.boolean(port.open) ||
            !reader.boolean(port.analogMode) ||
            !reader.boolean(port.pressureEnabled) ||
            !reader.boolean(port.lastUsedOverride) ||
            !reader.boolean(port.lastUsedBackend) ||
            !reader.boolean(port.lastReadOk) ||
            !reader.u16(port.buttonMask) || !reader.u32(port.dmaAddr) ||
            !reader.u32(port.reqState) ||
            !reader.u32(port.transientState) ||
            !reader.u16(port.lastInput.buttons) ||
            !reader.u8(port.lastInput.rx) || !reader.u8(port.lastInput.ry) ||
            !reader.u8(port.lastInput.lx) || !reader.u8(port.lastInput.ly) ||
            !reader.bytes(port.lastData.size(), lastData) ||
            !reader.u32(port.readCount) ||
            !reader.u32(port.lastReadDataAddr)) return false;
        std::copy(lastData.begin(), lastData.end(), port.lastData.begin());
    }
    uint32_t padFrame = 0u;
    if (!reader.u32(padFrame)) return false;

    uint32_t nextSifDma = 0u;
    if (!reader.u32(nextSifDma)) return false;
    const auto readU32Map = [&](auto &map) -> bool
    {
        uint32_t mapCount = 0u;
        if (!readCount(mapCount)) return false;
        for (uint32_t index = 0u; index < mapCount; ++index)
        {
            uint32_t key = 0u, value = 0u;
            if (!reader.u32(key) || !reader.u32(value) ||
                !map.emplace(key, value).second) return false;
        }
        return true;
    };
    std::unordered_map<uint32_t, uint32_t> sifRegs;
    std::unordered_map<uint32_t, uint32_t> sifSregs;
    std::unordered_map<uint32_t, uint32_t> sifHandlers;
    uint32_t cmdBuffer = 0u, sysCmdBuffer = 0u;
    bool cmdInitialized = false;
    if (!readU32Map(sifRegs) || !readU32Map(sifSregs) ||
        !readU32Map(sifHandlers) || !reader.u32(cmdBuffer) ||
        !reader.u32(sysCmdBuffer) ||
        !reader.boolean(cmdInitialized) || !readCount(count))
    {
        setError(error, "invalid saved SIF state");
        return false;
    }
    std::map<uint32_t, uint32_t> heapAllocations;
    for (uint32_t index = 0u; index < count; ++index)
    {
        uint32_t address = 0u, size = 0u;
        if (!reader.u32(address) || !reader.u32(size) ||
            !heapAllocations.emplace(address, size).second) return false;
    }
    uint32_t iopHeapNext = 0u;
    if (!reader.u32(iopHeapNext) || !readCount(count))
    {
        setError(error, "invalid saved SIF heap state");
        return false;
    }

    std::unordered_map<int, EeFileVagAccumEntry> vagAccum;
    for (uint32_t index = 0u; index < count; ++index)
    {
        int key = 0;
        EeFileVagAccumEntry entry{};
        std::span<const uint8_t> data;
        if (!reader.i32(key) || !reader.u32(entry.firstBufAddr) ||
            !reader.sizedBytes(data, 64u * 1024u * 1024u) ||
            !vagAccum.emplace(key, std::move(entry)).second) return false;
        auto &stored = vagAccum.at(key);
        stored.data.assign(data.begin(), data.end());
    }
    bool pathsInitialized = false;
    std::array<std::string, 6u> pathStrings{};
    std::string cwdDevice;
    if (!reader.boolean(pathsInitialized)) return false;
    for (std::string &path : pathStrings)
        if (!reader.string(path, kMaximumSavedString)) return false;
    if (!reader.string(cwdDevice, kMaximumSavedString) || !reader.atEnd())
    {
        setError(error, "invalid or trailing HLE save-state data");
        return false;
    }

    // Commit only after the complete chunk validates.
    {
        std::lock_guard<std::mutex> lock(sync.semaMapMutex);
        sync.nextSemaId = nextSemaId;
        sync.semas = std::move(semas);
    }
    {
        std::lock_guard<std::mutex> lock(sync.eventFlagMapMutex);
        sync.nextEventFlagId = nextEventFlagId;
        sync.eventFlags = std::move(eventFlags);
    }
    kernel.threadRootFunction.store(threadRoot, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(kernel.exitHandlerMutex);
        kernel.exitHandlers = std::move(exitHandlers);
    }
    {
        std::lock_guard<std::mutex> lock(kernel.bootModeMutex);
        kernel.bootModeInitialized = bootInitialized;
        kernel.bootModePoolOffset = bootPoolOffset;
        kernel.bootModeAddresses = std::move(bootAddresses);
    }
    {
        std::lock_guard<std::mutex> lock(kernel.syscallOverrideMutex);
        kernel.syscallOverrides = std::move(syscallOverrides);
        kernel.syscallMirrorAddresses = std::move(syscallMirrors);
    }
    {
        std::lock_guard<std::mutex> lock(
            kernel.activeSyscallOverrideMutex);
        kernel.activeSyscallOverrides = std::move(activeOverrides);
    }
    {
        std::lock_guard<std::mutex> lock(kernel.tlsMutex);
        kernel.tlsIndex = tlsIndex;
    }
    {
        std::lock_guard<std::mutex> lock(kernel.osdMutex);
        kernel.osdConfigInitialized = osdInitialized;
        kernel.osdConfigRaw = osdRaw;
        kernel.osdConfig2Raw = osd2Raw;
    }
    {
        std::lock_guard<std::mutex> lock(interrupt.handlerMutex);
        interrupt.intcHandlers = std::move(intcHandlers);
        interrupt.dmacHandlers = std::move(dmacHandlers);
        interrupt.nextIntcHandlerId = nextIntc;
        interrupt.nextDmacHandlerId = nextDmac;
        interrupt.intcHeadOrder = intcHead;
        interrupt.intcTailOrder = intcTail;
        interrupt.dmacHeadOrder = dmacHead;
        interrupt.dmacTailOrder = dmacTail;
        interrupt.enabledIntcMask = intcMask;
        interrupt.enabledDmacMask = dmacMask;
    }
    {
        std::lock_guard<std::mutex> lock(interrupt.vsyncMutex);
        interrupt.vsyncRegistration = vsyncRegistration;
        interrupt.vsyncTick = vsyncTick;
        interrupt.gsSyncVBaseTick = gsSyncBase;
        interrupt.gsVideoParameters = video;
        interrupt.gsSyncVCallback = gsCallback;
        interrupt.gsSyncVCallbackGp = gsCallbackGp;
        interrupt.gsSyncVCallbackSp = gsCallbackSp;
    }
    {
        std::lock_guard<std::mutex> lock(interrupt.callbackContextMutex);
        interrupt.callbackContexts.clear();
    }
    {
        std::lock_guard<std::mutex> lock(rpc.rpcMutex);
        rpc.initialized = rpcInitialized;
        rpc.servers = std::move(servers);
        rpc.clients = std::move(clients);
        rpc.latestClientBusyState.store(latestBusy, std::memory_order_release);
        rpc.nextDebugSequence = nextDebug;
        rpc.nextRequestId = nextRequest;
        rpc.packetIndex = packetIndex;
        rpc.serverIndex = serverIndex;
        rpc.activeQueue = activeQueue;
    }
    {
        std::lock_guard<std::mutex> lock(rpc.moduleMutex);
        rpc.nextModuleId = nextModule;
        rpc.modulesById = std::move(modules);
        rpc.moduleIdByPath = std::move(modulePaths);
    }
    {
        std::lock_guard<std::mutex> lock(deci.sessionMutex);
        deci.nextSocket = nextSocket;
        deci.sessions = std::move(sessions);
    }
    {
        auto &audio = *m_audioRuntimeState;
        std::lock_guard<std::mutex> lock(audio.mutex);
        audio.initialized = audioInitialized;
        audio.voiceTransfers = voice;
        audio.blockTransfers = block;
    }
    {
        auto &cd = *m_cdRuntimeState;
        std::lock_guard<std::mutex> lock(cd.mutex);
        cd.resetLocked();
        cd.nextPseudoLbn = nextPseudoLbn;
        cd.lastError = cdError;
        cd.mode = cdMode;
        cd.streamingLbn = streamLbn;
        cd.streamingEndLbn = streamEnd;
        cd.initialized = cdInitialized;
        cd.lastFailedPath = std::move(lastFailedPath);
        cd.samePathFailCount = samePathFails;
    }
    {
        auto &dma = *m_dmaRuntimeState;
        std::lock_guard<std::mutex> lock(dma.environmentMutex);
        dma.currentEnvironment = dmaEnvironment;
    }
    {
        auto &dma = *m_dmaRuntimeState;
        std::lock_guard<std::mutex> lock(dma.stubMutex);
        dma.pendingPolls = std::move(dmaPolls);
        dma.stubLogCount = 0u;
    }
    {
        auto &libc = *m_libcRuntimeState;
        libc.reset();
        std::lock_guard<std::mutex> lock(libc.mutex);
        libc.nextFileHandle = nextFileHandle;
        libc.randomSeed = randomSeed;
    }
    {
        auto &mc = *m_memoryCardRuntimeState;
        mc.reset();
        std::lock_guard<std::mutex> lock(mc.mutex);
        mc.nextFd = nextMcFd;
        mc.lastCmd = lastMcCommand;
        mc.commandPending = mcPending;
        mc.lastResult = lastMcResult;
        mc.ports = std::move(mcPorts);
        mc.cvFileCursor = cvCursor;
    }
    {
        auto &pad = *m_padRuntimeState;
        std::lock_guard<std::mutex> lock(pad.mutex);
        pad.ports = std::move(padPorts);
        pad.overrideEnabled = false;
        pad.overrideInput = {};
        pad.frameCount.store(padFrame, std::memory_order_relaxed);
    }
    {
        auto &sif = *m_sifRuntimeState;
        std::lock_guard<std::mutex> lock(sif.dmaTransferMutex);
        sif.nextDmaTransferId = nextSifDma;
    }
    {
        auto &sif = *m_sifRuntimeState;
        std::lock_guard<std::mutex> lock(sif.commandMutex);
        sif.regs = std::move(sifRegs);
        sif.sregs = std::move(sifSregs);
        sif.cmdHandlers = std::move(sifHandlers);
        sif.cmdBuffer = cmdBuffer;
        sif.sysCmdBuffer = sysCmdBuffer;
        sif.cmdInitialized = cmdInitialized;
    }
    {
        auto &sif = *m_sifRuntimeState;
        std::lock_guard<std::mutex> lock(sif.heapMutex);
        sif.heapAllocations = std::move(heapAllocations);
        sif.iopHeapNext = iopHeapNext;
    }
    {
        auto &files = *m_eeFileRuntimeState;
        files.closeAll();
        {
            std::lock_guard<std::mutex> lock(files.vagMutex);
            files.vagAccum = std::move(vagAccum);
        }
        {
            std::lock_guard<std::mutex> lock(files.pathMutex);
            files.pathsInitialized = pathsInitialized;
            files.hostBase = pathStrings[0];
            files.cdromBase = pathStrings[1];
            files.mcBase = pathStrings[2];
            files.hostCwd = pathStrings[3];
            files.cdromCwd = pathStrings[4];
            files.mcCwd = pathStrings[5];
            files.cwdDevice = std::move(cwdDevice);
        }
    }
    return true;
}

void PS2Runtime::setQuickSaveStatePath(std::filesystem::path path)
{
    if (path.empty())
        throw std::invalid_argument("quick-save state path is empty");
    m_quickSaveStatePath = std::move(path);
}

bool PS2Runtime::saveState(
    const std::filesystem::path &path,
    std::string *error)
{
    if (path.empty())
    {
        setError(error, "save-state path is empty");
        return false;
    }
    if (!m_eeRuntimeExecutor || !m_eeRuntimeExecutor->running())
    {
        setError(error,
                 "save state requires a running dedicated EE executor");
        return false;
    }

    Document document;
    Writer metadata;
    metadata.u32(kRuntimeStateVersion);
    metadata.string(
        m_loadedModules.empty() ? std::string_view{} :
                                  std::string_view(m_loadedModules.front().name));
    metadata.u32(PS2_RAM_SIZE);
    metadata.u32(PS2_GS_VRAM_SIZE);
    document.chunks.push_back(Chunk{
        .id = kMetaChunk,
        .version = kRuntimeStateVersion,
        .flags = ChunkFlags::Required,
        .payload = metadata.take(),
    });

    try
    {
        requestGuestPreemption();
        m_eeRuntimeExecutor->invokeAtBoundary(
            [this, &document, error](
                ps2x::ee::EeThreadScheduler &scheduler,
                IEeExecutionBackend &)
            {
                if (m_gsAsyncEnabled)
                {
                    std::lock_guard<std::mutex> lock(
                        m_gsEeJournalMutex);
                    reapEeGsSubmissions(true);
                }

                std::shared_ptr<const Vu1Snapshot> vu1Owner =
                    captureVu1OwnerSnapshot(false);
                if (!vu1Owner)
                    throw std::runtime_error("VU1 owner returned no snapshot");
                m_memory.publishVu1MemorySnapshot(*vu1Owner);
                m_vu1MemoryMirrorDirty = false;

                const GsReplayState gsState =
                    takeGsCommandResult<GsReplayStateResult>(
                        submitGsCommand(GsCaptureReplayStateCommand{}))
                        .state;
                const GsVramSnapshotResult vram =
                    takeGsCommandResult<GsVramSnapshotResult>(
                        submitGsCommand(GsCopyVramCommand{}));
                if (!vram.available ||
                    vram.bytes.size() != PS2_GS_VRAM_SIZE)
                {
                    throw std::runtime_error(
                        "GS owner returned no canonical VRAM image");
                }
                std::memcpy(
                    m_memory.getGSVRAM(),
                    vram.bytes.data(), vram.bytes.size());

                std::vector<uint8_t> memoryPayload;
                std::string localError;
                if (!m_memory.encodeSaveState(
                        memoryPayload, &localError))
                {
                    throw std::runtime_error(
                        "memory snapshot failed: " + localError);
                }
                document.chunks.push_back(Chunk{
                    .id = kMemoryChunk,
                    .version = kRuntimeStateVersion,
                    .flags = ChunkFlags::Required,
                    .payload = std::move(memoryPayload),
                });

                std::vector<uint8_t> corePayload;
                if (!encodeRuntimeCoreSaveState(
                        scheduler.snapshot(), corePayload, &localError))
                {
                    throw std::runtime_error(
                        "runtime core snapshot failed: " + localError);
                }
                document.chunks.push_back(Chunk{
                    .id = kCoreChunk,
                    .version = kRuntimeStateVersion,
                    .flags = ChunkFlags::Required,
                    .payload = std::move(corePayload),
                });

                // In-flight microprograms may own speculative host work. A
                // cold state drops that one transient activation while
                // retaining all architectural registers and memories; VIF is
                // released after load and the next game packet proceeds.
                VuExecutionState vu0State = m_vu0.state();
                vu0State.active = false;
                vu0State.branchPending = false;
                vu0State.branchDelay = 0u;
                vu0State.pipeline = {};
                Writer vu0Payload;
                vu0Payload.u32(kRuntimeStateVersion);
                writeVuExecutionState(vu0Payload, vu0State);
                document.chunks.push_back(Chunk{
                    .id = kVu0Chunk,
                    .version = kRuntimeStateVersion,
                    .flags = ChunkFlags::Required,
                    .payload = vu0Payload.take(),
                });

                VuExecutionState vu1State = vu1Owner->state;
                vu1State.active = false;
                vu1State.branchPending = false;
                vu1State.branchDelay = 0u;
                vu1State.pipeline = {};
                Writer vu1Payload;
                vu1Payload.u32(kRuntimeStateVersion);
                writeVuExecutionState(vu1Payload, vu1State);
                writeVu1VifState(vu1Payload, vu1Owner->vif);
                vu1Payload.u64(vu1Owner->codeGeneration);
                document.chunks.push_back(Chunk{
                    .id = kVu1Chunk,
                    .version = kRuntimeStateVersion,
                    .flags = ChunkFlags::Required,
                    .payload = vu1Payload.take(),
                });

                std::vector<uint8_t> gsPayload;
                if (!encodeGsReplayState(
                        gsState, gsPayload, &localError))
                {
                    throw std::runtime_error(
                        "GS snapshot failed: " + localError);
                }
                document.chunks.push_back(Chunk{
                    .id = kGsChunk,
                    .version = kRuntimeStateVersion,
                    .flags = ChunkFlags::Required,
                    .payload = std::move(gsPayload),
                });

                std::vector<uint8_t> hlePayload;
                if (!encodeRuntimeHleSaveState(
                        hlePayload, &localError))
                {
                    throw std::runtime_error(
                        "HLE snapshot failed: " + localError);
                }
                document.chunks.push_back(Chunk{
                    .id = kHleChunk,
                    .version = kRuntimeStateVersion,
                    .flags = ChunkFlags::Required,
                    .payload = std::move(hlePayload),
                });
            },
            ps2x::ee::EeSchedulerReschedulePolicy::None);
    }
    catch (const std::exception &exception)
    {
        setError(error, exception.what());
        return false;
    }

    return ps2x::savestate::writeFileAtomically(
        path, document, error);
}

bool PS2Runtime::loadState(
    const std::filesystem::path &path,
    std::string *error)
{
    if (path.empty())
    {
        setError(error, "save-state path is empty");
        return false;
    }
    if (!usesDedicatedEeExecutor())
    {
        setError(error,
                 "save states require the dedicated EE fiber executor");
        return false;
    }
    if (m_eeRuntimeExecutor && m_eeRuntimeExecutor->running())
    {
        setError(error,
                 "v1 save states can only be loaded before run()");
        return false;
    }
    if (!m_memory.getRDRAM() || !m_memory.getGSVRAM())
    {
        setError(error, "runtime memory is not initialized");
        return false;
    }

    Document document;
    if (!ps2x::savestate::readFile(path, document, error))
        return false;
    const auto requiredChunk = [&](uint32_t id) -> const Chunk *
    {
        const Chunk *chunk = document.find(id);
        return chunk && chunk->version == kRuntimeStateVersion
                   ? chunk
                   : nullptr;
    };
    const Chunk *meta = requiredChunk(kMetaChunk);
    const Chunk *memory = requiredChunk(kMemoryChunk);
    const Chunk *core = requiredChunk(kCoreChunk);
    const Chunk *vu0 = requiredChunk(kVu0Chunk);
    const Chunk *vu1 = requiredChunk(kVu1Chunk);
    const Chunk *gs = requiredChunk(kGsChunk);
    const Chunk *hle = requiredChunk(kHleChunk);
    if (!meta || !memory || !core || !vu0 || !vu1 || !gs || !hle)
    {
        setError(error,
                 "save state is missing a required v1 runtime chunk");
        return false;
    }

    Reader metaReader(meta->payload);
    uint32_t metadataVersion = 0u;
    std::string moduleName;
    uint32_t ramSize = 0u, vramSize = 0u;
    if (!metaReader.u32(metadataVersion) ||
        metadataVersion != kRuntimeStateVersion ||
        !metaReader.string(moduleName, kMaximumSavedString) ||
        !metaReader.u32(ramSize) || !metaReader.u32(vramSize) ||
        !metaReader.atEnd() || ramSize != PS2_RAM_SIZE ||
        vramSize != PS2_GS_VRAM_SIZE)
    {
        setError(error, "invalid save-state metadata");
        return false;
    }
    const std::string loadedModule =
        m_loadedModules.empty() ? std::string{} :
                                  m_loadedModules.front().name;
    if (!moduleName.empty() && loadedModule != moduleName)
    {
        setError(error,
                 "save state belongs to a different guest ELF (" +
                     moduleName + ")");
        return false;
    }

    VuExecutionState vu0State{};
    {
        Reader reader(vu0->payload);
        uint32_t version = 0u;
        if (!reader.u32(version) || version != kRuntimeStateVersion ||
            !readVuExecutionState(reader, vu0State) || !reader.atEnd())
        {
            setError(error, "invalid VU0 save-state chunk");
            return false;
        }
    }
    VuExecutionState vu1State{};
    Vu1VifState vu1Vif{};
    uint64_t vu1CodeGeneration = 0u;
    {
        Reader reader(vu1->payload);
        uint32_t version = 0u;
        if (!reader.u32(version) || version != kRuntimeStateVersion ||
            !readVuExecutionState(reader, vu1State) ||
            !readVu1VifState(reader, vu1Vif) ||
            !reader.u64(vu1CodeGeneration) || !reader.atEnd())
        {
            setError(error, "invalid VU1 save-state chunk");
            return false;
        }
    }
    GsReplayState gsState{};
    if (!decodeGsReplayState(gs->payload, gsState, error))
        return false;

    if (!m_memory.decodeSaveState(memory->payload, error) ||
        !decodeRuntimeCoreSaveState(core->payload, error) ||
        !decodeRuntimeHleSaveState(hle->payload, error))
    {
        return false;
    }

    const GsBooleanResult gsRestored =
        takeGsCommandResult<GsBooleanResult>(
            submitGsCommand(GsRestoreReplayStateCommand{
                .state = std::move(gsState)}));
    if (!gsRestored.value)
    {
        setError(error, "GS owner rejected the restored state");
        return false;
    }

    m_vu0.state() = std::move(vu0State);
    m_vu0.requestInterpreterRecovery();
    auto restoredVu1 = std::make_shared<Vu1Snapshot>();
    restoredVu1->state = std::move(vu1State);
    restoredVu1->microMemory.assign(
        m_memory.getVU1Code(),
        m_memory.getVU1Code() + PS2_VU1_CODE_SIZE);
    restoredVu1->dataMemory.assign(
        m_memory.getVU1Data(),
        m_memory.getVU1Data() + PS2_VU1_DATA_SIZE);
    restoredVu1->vif = vu1Vif;
    restoredVu1->codeGeneration = vu1CodeGeneration;
    Vu1CommandResult vu1Result = submitVu1Command(
        Vu1RestoreCommand{
            .snapshot = restoredVu1,
            .interpreterRecovery = true,
        },
        currentEeTick().raw(),
        m_vu1ExecutionTiming.generation);
    if (vu1Result.disposition != Vu1CommandDisposition::Completed)
    {
        setError(error, "VU1 owner rejected the restored state");
        return false;
    }
    publishVu1CommandResult(vu1Result);
    m_memory.publishVu1MemorySnapshot(*restoredVu1);
    m_vu1MemoryMirrorDirty = false;

    // The cold format deliberately does not serialize speculative VU host
    // work. Release any VIF parser which was waiting for that discarded one
    // activation and let the normal DMA scheduler continue from its saved
    // byte-stream position. The first activation on each VU uses the
    // interpreter until an inactive boundary, then the configured backend
    // takes over again. This bounds recovery without filling a native cache
    // from an incomplete mid-program pipeline state.
    (void)m_eeEventScheduler.cancel(
        ps2x::timing::EeEventSource::VifVu0Finish);
    (void)m_eeEventScheduler.cancel(
        ps2x::timing::EeEventSource::VifVu1Finish);
    m_vif0VuFinishEventToken = {
        ps2x::timing::EeEventSource::VifVu0Finish, 0u};
    m_vu1ExecutionTiming.eventToken = {
        ps2x::timing::EeEventSource::VifVu1Finish, 0u};
    m_pendingVu1Slice.reset();
    m_pendingVu1ExecutionEpoch.reset();
    m_pendingVu1CoarseInvocation.reset();
    m_deferredVu1SlicePlan.reset();
    m_vu1SpeculationPublicationState =
        Vu1SpeculationPublicationState::None;
    (void)m_memory.resumeVIF0AfterVu();
    (void)m_memory.resumeVIF1AfterVu();
    constexpr uint32_t kRestoredVu1BusyMask = 1u << 8u;
    m_cpuContext.vu0_vpu_stat &= ~kRestoredVu1BusyMask;
    m_cpuContext.vu0_vpu_stat &= ~1u;
    {
        EeThreadRuntimeState &threads = *m_eeThreadRuntimeState;
        std::lock_guard<std::mutex> lock(threads.threadMapMutex);
        for (const auto &[id, info] : threads.threads)
        {
            if (id != 1 && info)
            {
                info->context.vu0_vpu_stat &= ~kRestoredVu1BusyMask;
                info->context.vu0_vpu_stat &= ~1u;
            }
        }
    }
    m_publishedVu1Active.store(false, std::memory_order_release);
    m_saveStateLoaded = true;
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);
    m_debugRa.store(getRegU32(&m_cpuContext, 31),
                    std::memory_order_relaxed);
    m_debugSp.store(getRegU32(&m_cpuContext, 29),
                    std::memory_order_relaxed);
    m_debugGp.store(getRegU32(&m_cpuContext, 28),
                    std::memory_order_relaxed);
    return true;
}
