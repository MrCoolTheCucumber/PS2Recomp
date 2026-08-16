#include "runtime/ps2_memory.h"

#include "runtime/ps2_save_state.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string_view>
#include <tuple>

namespace
{
    using ps2x::savestate::Reader;
    using ps2x::savestate::Writer;

    constexpr uint32_t kMemoryStateVersion = 1u;
    constexpr uint32_t kMaximumIoRegisterCount = 65536u;
    constexpr uint32_t kMaximumTlbEntryCount = 256u;
    constexpr uint64_t kMaximumDeferredBytes = 64ull * 1024ull * 1024ull;
    constexpr uint32_t kMaximumDeferredSegmentCount = 1u << 20u;
    constexpr uint32_t kMaximumPath3PacketCount = 1u << 16u;
    constexpr uint32_t kMaximumPendingDmacCount = 4096u;

    void setError(std::string *error, std::string_view message)
    {
        if (error)
            *error = message;
    }

    void writeToken(Writer &writer, DmacTransferToken token)
    {
        writer.u8(static_cast<uint8_t>(token.channel));
        writer.u64(token.generation);
    }

    bool readToken(Reader &reader, DmacTransferToken &token)
    {
        uint8_t channel = 0u;
        if (!reader.u8(channel) ||
            channel >= static_cast<uint8_t>(DmacChannel::Count) ||
            !reader.u64(token.generation))
        {
            return false;
        }
        token.channel = static_cast<DmacChannel>(channel);
        return true;
    }

    template <typename Enum>
    bool readEnum(Reader &reader, Enum &value, uint8_t maximum)
    {
        uint8_t encoded = 0u;
        if (!reader.u8(encoded) || encoded > maximum)
            return false;
        value = static_cast<Enum>(encoded);
        return true;
    }

    void writeVifRegisters(Writer &writer, const VIFRegisters &registers)
    {
        writer.u32(registers.stat);
        writer.u32(registers.fbrst);
        writer.u32(registers.err);
        writer.u32(registers.mark);
        writer.u32(registers.cycle);
        writer.u32(registers.mode);
        writer.u32(registers.num);
        writer.u32(registers.mask);
        writer.u32(registers.code);
        writer.u32(registers.itops);
        writer.u32(registers.base);
        writer.u32(registers.ofst);
        writer.u32(registers.tops);
        writer.u32(registers.itop);
        writer.u32(registers.top);
        for (uint32_t value : registers.row)
            writer.u32(value);
        for (uint32_t value : registers.col)
            writer.u32(value);
    }

    bool readVifRegisters(Reader &reader, VIFRegisters &registers)
    {
        if (!reader.u32(registers.stat) ||
            !reader.u32(registers.fbrst) ||
            !reader.u32(registers.err) ||
            !reader.u32(registers.mark) ||
            !reader.u32(registers.cycle) ||
            !reader.u32(registers.mode) ||
            !reader.u32(registers.num) ||
            !reader.u32(registers.mask) ||
            !reader.u32(registers.code) ||
            !reader.u32(registers.itops) ||
            !reader.u32(registers.base) ||
            !reader.u32(registers.ofst) ||
            !reader.u32(registers.tops) ||
            !reader.u32(registers.itop) ||
            !reader.u32(registers.top))
        {
            return false;
        }
        for (uint32_t &value : registers.row)
        {
            if (!reader.u32(value))
                return false;
        }
        for (uint32_t &value : registers.col)
        {
            if (!reader.u32(value))
                return false;
        }
        return true;
    }

    void writeDmaRegisters(Writer &writer, const DMARegisters &registers)
    {
        writer.u32(registers.chcr);
        writer.u32(registers.madr);
        writer.u32(registers.qwc);
        writer.u32(registers.tadr);
        writer.u32(registers.asr0);
        writer.u32(registers.asr1);
        writer.u32(registers.sadr);
    }

    bool readDmaRegisters(Reader &reader, DMARegisters &registers)
    {
        return reader.u32(registers.chcr) &&
               reader.u32(registers.madr) &&
               reader.u32(registers.qwc) &&
               reader.u32(registers.tadr) &&
               reader.u32(registers.asr0) &&
               reader.u32(registers.asr1) &&
               reader.u32(registers.sadr);
    }

    void writeCounterSnapshot(
        Writer &writer,
        const ps2x::timing::EeCounterBankSnapshot &state)
    {
        for (const ps2x::timing::EeCounterSnapshot &counter : state.counters)
        {
            writer.u32(counter.count);
            writer.u16(counter.mode);
            writer.u16(counter.target);
            writer.u16(counter.hold);
            writer.u32(counter.rate);
            writer.u64(counter.startCycle);
            writer.boolean(counter.targetAfterOverflow);
        }
        writer.u64(state.videoTiming.renderCycles);
        writer.u64(state.videoTiming.blankCycles);
        writer.u32(state.videoTiming.hRenderCycles);
        writer.u32(state.videoTiming.hBlankCycles);
        writer.u32(state.videoTiming.hSyncErrorCycles);
        writer.u64(state.currentCycle);
        writer.u64(state.hSyncStartCycle);
        writer.u64(state.vSyncStartCycle);
        writer.u32(state.hSyncDeltaCycles);
        writer.u64(state.vSyncDeltaCycles);
        writer.u8(static_cast<uint8_t>(state.hSyncPhase));
        writer.u8(static_cast<uint8_t>(state.vSyncPhase));
    }

    bool readCounterSnapshot(
        Reader &reader,
        ps2x::timing::EeCounterBankSnapshot &state)
    {
        for (ps2x::timing::EeCounterSnapshot &counter : state.counters)
        {
            if (!reader.u32(counter.count) ||
                !reader.u16(counter.mode) ||
                !reader.u16(counter.target) ||
                !reader.u16(counter.hold) ||
                !reader.u32(counter.rate) ||
                !reader.u64(counter.startCycle) ||
                !reader.boolean(counter.targetAfterOverflow))
            {
                return false;
            }
        }
        return reader.u64(state.videoTiming.renderCycles) &&
               reader.u64(state.videoTiming.blankCycles) &&
               reader.u32(state.videoTiming.hRenderCycles) &&
               reader.u32(state.videoTiming.hBlankCycles) &&
               reader.u32(state.videoTiming.hSyncErrorCycles) &&
               reader.u64(state.currentCycle) &&
               reader.u64(state.hSyncStartCycle) &&
               reader.u64(state.vSyncStartCycle) &&
               reader.u32(state.hSyncDeltaCycles) &&
               reader.u64(state.vSyncDeltaCycles) &&
               readEnum(
                   reader, state.hSyncPhase,
                   static_cast<uint8_t>(
                       ps2x::timing::EeCounterSignalPhase::Blank)) &&
               readEnum(
                   reader, state.vSyncPhase,
                   static_cast<uint8_t>(
                       ps2x::timing::EeCounterSignalPhase::Blank));
    }

    bool readOwnedBytes(
        Reader &reader,
        std::vector<uint8_t> &destination,
        uint64_t maximum = kMaximumDeferredBytes)
    {
        std::span<const uint8_t> bytes;
        if (!reader.sizedBytes(bytes, maximum))
            return false;
        destination.assign(bytes.begin(), bytes.end());
        return true;
    }

    bool readFixedBytes(
        Reader &reader, void *destination, size_t size)
    {
        std::span<const uint8_t> bytes;
        if (!reader.bytes(size, bytes))
            return false;
        std::memcpy(destination, bytes.data(), size);
        return true;
    }
}

bool PS2Memory::encodeSaveState(
    std::vector<uint8_t> &output,
    std::string *error) const
{
    if (!m_rdram || !m_scratchpad || !iop_ram || !m_gsVRAM ||
        !m_vu0Code || !m_vu0Data || !m_vu1Code || !m_vu1Data)
    {
        setError(error, "PS2 memory is not initialized");
        return false;
    }

    Writer writer;
    writer.u32(kMemoryStateVersion);
    writer.bytes(std::span<const uint8_t>(m_rdram, PS2_RAM_SIZE));
    writer.bytes(std::span<const uint8_t>(m_scratchpad, PS2_SCRATCHPAD_SIZE));
    writer.bytes(std::span<const uint8_t>(iop_ram, PS2_IOP_RAM_SIZE));
    writer.bytes(std::span<const uint8_t>(m_gsVRAM, PS2_GS_VRAM_SIZE));
    writer.bytes(std::span<const uint8_t>(m_vu0Code, PS2_VU0_CODE_SIZE));
    writer.bytes(std::span<const uint8_t>(m_vu0Data, PS2_VU0_DATA_SIZE));
    writer.bytes(std::span<const uint8_t>(m_vu1Code, PS2_VU1_CODE_SIZE));
    writer.bytes(std::span<const uint8_t>(m_vu1Data, PS2_VU1_DATA_SIZE));

    writer.boolean(m_seenGifCopy);
    writer.u64(m_dmaStartCount.load(std::memory_order_relaxed));
    writer.u64(m_gifCopyCount.load(std::memory_order_relaxed));
    writer.u64(m_gsWriteCount.load(std::memory_order_relaxed));
    writer.u64(m_vifWriteCount.load(std::memory_order_relaxed));
    writer.u64(m_vu0CodeGeneration.load(std::memory_order_relaxed));
    writer.u64(m_vu1CodeGeneration.load(std::memory_order_relaxed));

    std::vector<std::pair<uint32_t, uint32_t>> ioRegisters(
        m_ioRegisters.begin(), m_ioRegisters.end());
    std::sort(ioRegisters.begin(), ioRegisters.end());
    writer.u32(static_cast<uint32_t>(ioRegisters.size()));
    for (const auto &[address, value] : ioRegisters)
    {
        writer.u32(address);
        writer.u32(value);
    }

    writer.u64(gs_regs.pmode);
    writer.u64(gs_regs.smode1);
    writer.u64(gs_regs.smode2);
    writer.u64(gs_regs.srfsh);
    writer.u64(gs_regs.synch1);
    writer.u64(gs_regs.synch2);
    writer.u64(gs_regs.syncv);
    writer.u64(gs_regs.dispfb1);
    writer.u64(gs_regs.display1);
    writer.u64(gs_regs.dispfb2);
    writer.u64(gs_regs.display2);
    writer.u64(gs_regs.extbuf);
    writer.u64(gs_regs.extdata);
    writer.u64(gs_regs.extwrite);
    writer.u64(gs_regs.bgcolor);
    writer.u64(gs_regs.csr.load(std::memory_order_relaxed));
    writer.u64(gs_regs.imr);
    writer.u64(gs_regs.busdir);
    writer.u64(gs_regs.siglblid);
    writeVifRegisters(writer, vif0_regs);
    writeVifRegisters(writer, vif1_regs);
    for (const DMARegisters &registers : dma_regs)
        writeDmaRegisters(writer, registers);

    writer.u32(static_cast<uint32_t>(m_tlbEntries.size()));
    for (const EeTlbEntry &entry : m_tlbEntries)
    {
        writer.u32(entry.pageMask);
        writer.u32(entry.entryHi);
        writer.u32(entry.entryLo0);
        writer.u32(entry.entryLo1);
    }
    writer.boolean(m_postBiosTlbStateConfigured);
    writer.u64(m_tlbMappingGeneration);

    writer.boolean(m_path3Masked);
    writer.boolean(m_gifModePath3Masked);
    writer.boolean(m_vif0WaitingForVu);
    writer.sizedBytes(m_vif0DeferredData);
    writer.u32(m_vif1PendingPath2DirectQwc);
    writer.boolean(m_vif1PendingPath2DirectHl);
    writer.boolean(m_vif1PendingIrqAfterCommand);
    writer.boolean(m_vif1WaitingForVu);
    writer.sizedBytes(m_vif1DeferredData);
    writer.u32(static_cast<uint32_t>(m_vif1DeferredSegments.size()));
    for (const Vif1StreamSegment &segment : m_vif1DeferredSegments)
    {
        writer.u64(segment.byteCount);
        writer.u8(segment.firstByteOffset);
    }
    writer.u8(m_vif1NextByteOffset);
    writer.u32(static_cast<uint32_t>(m_path3MaskedFifo.size()));
    for (const std::vector<uint8_t> &packet : m_path3MaskedFifo)
        writer.sizedBytes(packet);

    const auto writeVif0Dma = [&writer](const Vif0DmaState &state)
    {
        writeToken(writer, state.transfer);
        writer.u8(static_cast<uint8_t>(state.phase));
        writer.u8(static_cast<uint8_t>(state.stall));
        writer.u32(state.madr);
        writer.u32(state.qwc);
        writer.u32(state.tadr);
        writer.u32(state.asr0);
        writer.u32(state.asr1);
        writer.u32(state.bufferedPayloadBytes);
        writer.u32(state.payloadByteOffset);
        writer.u32(state.tagsProcessed);
        writer.u8(state.asp);
        writer.u8(state.tagId);
        writer.boolean(state.active);
        writer.boolean(state.eventManaged);
        writer.boolean(state.chainMode);
        writer.boolean(state.tie);
        writer.boolean(state.tte);
        writer.boolean(state.tagIrq);
        writer.boolean(state.endAfterPayload);
        writer.boolean(state.faultReported);
    };
    const auto writeVif1Dma = [&writer](const Vif1DmaState &state)
    {
        writeToken(writer, state.transfer);
        writer.u8(static_cast<uint8_t>(state.phase));
        writer.u8(static_cast<uint8_t>(state.stall));
        writer.u32(state.madr);
        writer.u32(state.qwc);
        writer.u32(state.tadr);
        writer.u32(state.asr0);
        writer.u32(state.asr1);
        writer.u32(state.tagsProcessed);
        writer.u8(state.asp);
        writer.u8(state.tagId);
        writer.boolean(state.active);
        writer.boolean(state.eventManaged);
        writer.boolean(state.chainMode);
        writer.boolean(state.tie);
        writer.boolean(state.tte);
        writer.boolean(state.tagIrq);
        writer.boolean(state.endAfterPayload);
        writer.boolean(state.faultReported);
    };
    writeVif0Dma(m_vif0Dma);
    writeVif1Dma(m_vif1Dma);

    writeToken(writer, m_gifDma.transfer);
    writer.u8(static_cast<uint8_t>(m_gifDma.phase));
    writer.u8(static_cast<uint8_t>(m_gifDma.stall));
    writer.u32(m_gifDma.chcr);
    writer.u32(m_gifDma.madr);
    writer.u32(m_gifDma.qwc);
    writer.u32(m_gifDma.tadr);
    writer.u32(m_gifDma.asr0);
    writer.u32(m_gifDma.asr1);
    writer.u32(m_gifDma.tagsProcessed);
    writer.u8(m_gifDma.asp);
    writer.u8(m_gifDma.tagId);
    writer.boolean(m_gifDma.active);
    writer.boolean(m_gifDma.eventManaged);
    writer.boolean(m_gifDma.chainMode);
    writer.boolean(m_gifDma.tie);
    writer.boolean(m_gifDma.tagIrq);
    writer.boolean(m_gifDma.endAfterPayload);
    writer.boolean(m_gifDma.faultReported);
    writer.boolean(m_gifDma.copyCounted);
    writer.sizedBytes(m_gifDma.fifoData);
    writer.sizedBytes(m_gifDma.directCallbackData);

    writeToken(writer, m_fromIpuDma.transfer);
    writer.u8(static_cast<uint8_t>(m_fromIpuDma.phase));
    writer.u8(static_cast<uint8_t>(m_fromIpuDma.stall));
    writer.u32(m_fromIpuDma.chcr);
    writer.u32(m_fromIpuDma.madr);
    writer.u32(m_fromIpuDma.qwc);
    writer.boolean(m_fromIpuDma.active);
    writer.boolean(m_fromIpuDma.eventManaged);
    writer.boolean(m_fromIpuDma.normalMode);
    writer.boolean(m_fromIpuDma.zeroQwcStart);
    writer.boolean(m_fromIpuDma.faultReported);

    writeToken(writer, m_toIpuDma.transfer);
    writer.u8(static_cast<uint8_t>(m_toIpuDma.phase));
    writer.u8(static_cast<uint8_t>(m_toIpuDma.stall));
    writer.u32(m_toIpuDma.chcr);
    writer.u32(m_toIpuDma.madr);
    writer.u32(m_toIpuDma.qwc);
    writer.u32(m_toIpuDma.tadr);
    writer.u32(m_toIpuDma.tagsProcessed);
    writer.u8(m_toIpuDma.tagId);
    writer.boolean(m_toIpuDma.active);
    writer.boolean(m_toIpuDma.eventManaged);
    writer.boolean(m_toIpuDma.normalMode);
    writer.boolean(m_toIpuDma.chainMode);
    writer.boolean(m_toIpuDma.tie);
    writer.boolean(m_toIpuDma.tagIrq);
    writer.boolean(m_toIpuDma.endAfterPayload);
    writer.boolean(m_toIpuDma.faultReported);

    writer.bytes(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t *>(m_ipuOutputFifo.data()),
        sizeof(m_ipuOutputFifo)));
    writer.u32(m_ipuOutputFifoReadIndex);
    writer.u32(m_ipuOutputFifoWriteIndex);
    writer.u32(m_ipuOutputFifoQwc);
    writer.bytes(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t *>(m_ipuInputFifo.data()),
        sizeof(m_ipuInputFifo)));
    writer.u32(m_ipuInputFifoReadIndex);
    writer.u32(m_ipuInputFifoWriteIndex);
    writer.u32(m_ipuInputFifoQwc);

    for (const ScratchpadDmaState &state : m_scratchpadDma)
    {
        writeToken(writer, state.transfer);
        writer.u8(static_cast<uint8_t>(state.phase));
        writer.u32(state.chcr);
        writer.u32(state.madr);
        writer.u32(state.qwc);
        writer.u32(state.tadr);
        writer.u32(state.asr0);
        writer.u32(state.asr1);
        writer.u32(state.sadr);
        writer.u32(state.tagsProcessed);
        writer.u8(state.asp);
        writer.u8(state.tagId);
        writer.boolean(state.active);
        writer.boolean(state.eventManaged);
        writer.boolean(state.chainMode);
        writer.boolean(state.tie);
        writer.boolean(state.tte);
        writer.boolean(state.tagIrq);
        writer.boolean(state.endAfterPayload);
        writer.boolean(state.faultReported);
    }

    {
        std::lock_guard<std::mutex> lock(m_dmacMutex);
        for (const DmacChannelLifecycle &channel : m_dmacChannels)
        {
            writer.u8(static_cast<uint8_t>(channel.phase));
            writer.u64(channel.generation);
        }
        writer.u32(static_cast<uint32_t>(m_readyDmacCompletions.size()));
        for (const DmacCompletionRequest &completion :
             m_readyDmacCompletions)
        {
            writeToken(writer, completion.transfer);
            writer.u64(completion.readySequence);
        }
        writer.u32(static_cast<uint32_t>(m_pendingDmacInterrupts.size()));
        for (const DmacPendingInterrupt &interrupt :
             m_pendingDmacInterrupts)
        {
            writer.u8(static_cast<uint8_t>(interrupt.source));
            writer.u32(interrupt.cause);
            writer.u64(interrupt.channelGeneration);
            writer.u64(interrupt.eventSequence);
            writer.u64(interrupt.publicationSequence);
        }
        writer.u64(m_dmacReadySequence);
        writer.u64(m_dmacPublicationSequence);
    }

    writeCounterSnapshot(writer, m_eeCounters.snapshot());
    output = writer.take();
    return true;
}

bool PS2Memory::decodeSaveState(
    std::span<const uint8_t> input,
    std::string *error)
{
    if (!m_rdram || !m_scratchpad || !iop_ram || !m_gsVRAM ||
        !m_vu0Code || !m_vu0Data || !m_vu1Code || !m_vu1Data)
    {
        setError(error, "PS2 memory is not initialized");
        return false;
    }

    Reader reader(input);
    uint32_t version = 0u;
    if (!reader.u32(version) || version != kMemoryStateVersion)
    {
        setError(error, "unsupported PS2 memory state version");
        return false;
    }
    if (!readFixedBytes(reader, m_rdram, PS2_RAM_SIZE) ||
        !readFixedBytes(reader, m_scratchpad, PS2_SCRATCHPAD_SIZE) ||
        !readFixedBytes(reader, iop_ram, PS2_IOP_RAM_SIZE) ||
        !readFixedBytes(reader, m_gsVRAM, PS2_GS_VRAM_SIZE) ||
        !readFixedBytes(reader, m_vu0Code, PS2_VU0_CODE_SIZE) ||
        !readFixedBytes(reader, m_vu0Data, PS2_VU0_DATA_SIZE) ||
        !readFixedBytes(reader, m_vu1Code, PS2_VU1_CODE_SIZE) ||
        !readFixedBytes(reader, m_vu1Data, PS2_VU1_DATA_SIZE))
    {
        setError(error, "PS2 memory state payload is truncated");
        return false;
    }

    uint64_t dmaStartCount = 0u;
    uint64_t gifCopyCount = 0u;
    uint64_t gsWriteCount = 0u;
    uint64_t vifWriteCount = 0u;
    uint64_t vu0Generation = 0u;
    uint64_t vu1Generation = 0u;
    if (!reader.boolean(m_seenGifCopy) ||
        !reader.u64(dmaStartCount) || !reader.u64(gifCopyCount) ||
        !reader.u64(gsWriteCount) || !reader.u64(vifWriteCount) ||
        !reader.u64(vu0Generation) || !reader.u64(vu1Generation))
    {
        setError(error, "PS2 memory state counters are invalid");
        return false;
    }
    m_dmaStartCount.store(dmaStartCount, std::memory_order_relaxed);
    m_gifCopyCount.store(gifCopyCount, std::memory_order_relaxed);
    m_gsWriteCount.store(gsWriteCount, std::memory_order_relaxed);
    m_vifWriteCount.store(vifWriteCount, std::memory_order_relaxed);
    m_vu0CodeGeneration.store(vu0Generation, std::memory_order_relaxed);
    m_vu1CodeGeneration.store(vu1Generation, std::memory_order_relaxed);

    uint32_t ioCount = 0u;
    if (!reader.u32(ioCount) || ioCount > kMaximumIoRegisterCount)
    {
        setError(error, "PS2 memory state I/O register count is invalid");
        return false;
    }
    std::unordered_map<uint32_t, uint32_t> ioRegisters;
    ioRegisters.reserve(ioCount);
    for (uint32_t index = 0u; index < ioCount; ++index)
    {
        uint32_t address = 0u;
        uint32_t value = 0u;
        if (!reader.u32(address) || !reader.u32(value) ||
            !ioRegisters.emplace(address, value).second)
        {
            setError(error, "PS2 memory state I/O registers are invalid");
            return false;
        }
    }
    m_ioRegisters = std::move(ioRegisters);

    uint64_t csr = 0u;
    if (!reader.u64(gs_regs.pmode) || !reader.u64(gs_regs.smode1) ||
        !reader.u64(gs_regs.smode2) || !reader.u64(gs_regs.srfsh) ||
        !reader.u64(gs_regs.synch1) || !reader.u64(gs_regs.synch2) ||
        !reader.u64(gs_regs.syncv) || !reader.u64(gs_regs.dispfb1) ||
        !reader.u64(gs_regs.display1) || !reader.u64(gs_regs.dispfb2) ||
        !reader.u64(gs_regs.display2) || !reader.u64(gs_regs.extbuf) ||
        !reader.u64(gs_regs.extdata) || !reader.u64(gs_regs.extwrite) ||
        !reader.u64(gs_regs.bgcolor) || !reader.u64(csr) ||
        !reader.u64(gs_regs.imr) || !reader.u64(gs_regs.busdir) ||
        !reader.u64(gs_regs.siglblid) ||
        !readVifRegisters(reader, vif0_regs) ||
        !readVifRegisters(reader, vif1_regs))
    {
        setError(error, "PS2 memory state register payload is invalid");
        return false;
    }
    gs_regs.csr.store(csr, std::memory_order_relaxed);
    for (DMARegisters &registers : dma_regs)
    {
        if (!readDmaRegisters(reader, registers))
        {
            setError(error, "PS2 memory state DMA registers are invalid");
            return false;
        }
    }

    uint32_t tlbCount = 0u;
    if (!reader.u32(tlbCount) || tlbCount > kMaximumTlbEntryCount)
    {
        setError(error, "PS2 memory state TLB entry count is invalid");
        return false;
    }
    std::vector<EeTlbEntry> tlbEntries(tlbCount);
    for (EeTlbEntry &entry : tlbEntries)
    {
        if (!reader.u32(entry.pageMask) || !reader.u32(entry.entryHi) ||
            !reader.u32(entry.entryLo0) || !reader.u32(entry.entryLo1))
        {
            setError(error, "PS2 memory state TLB payload is invalid");
            return false;
        }
    }
    if (!reader.boolean(m_postBiosTlbStateConfigured) ||
        !reader.u64(m_tlbMappingGeneration) ||
        m_tlbMappingGeneration == 0u)
    {
        setError(error, "PS2 memory state TLB metadata is invalid");
        return false;
    }
    m_tlbEntries = std::move(tlbEntries);

    if (!reader.boolean(m_path3Masked) ||
        !reader.boolean(m_gifModePath3Masked) ||
        !reader.boolean(m_vif0WaitingForVu) ||
        !readOwnedBytes(reader, m_vif0DeferredData) ||
        !reader.u32(m_vif1PendingPath2DirectQwc) ||
        !reader.boolean(m_vif1PendingPath2DirectHl) ||
        !reader.boolean(m_vif1PendingIrqAfterCommand) ||
        !reader.boolean(m_vif1WaitingForVu) ||
        !readOwnedBytes(reader, m_vif1DeferredData))
    {
        setError(error, "PS2 memory state VIF stream payload is invalid");
        return false;
    }
    uint32_t segmentCount = 0u;
    if (!reader.u32(segmentCount) ||
        segmentCount > kMaximumDeferredSegmentCount)
    {
        setError(error, "PS2 memory state VIF segment count is invalid");
        return false;
    }
    m_vif1DeferredSegments.clear();
    uint64_t segmentBytes = 0u;
    for (uint32_t index = 0u; index < segmentCount; ++index)
    {
        uint64_t byteCount = 0u;
        uint8_t offset = 0u;
        if (!reader.u64(byteCount) || byteCount > kMaximumDeferredBytes ||
            segmentBytes > kMaximumDeferredBytes - byteCount ||
            !reader.u8(offset) || offset > 0x0fu)
        {
            setError(error, "PS2 memory state VIF segment is invalid");
            return false;
        }
        segmentBytes += byteCount;
        m_vif1DeferredSegments.push_back(Vif1StreamSegment{
            .byteCount = static_cast<size_t>(byteCount),
            .firstByteOffset = offset,
        });
    }
    if (segmentBytes != m_vif1DeferredData.size() ||
        !reader.u8(m_vif1NextByteOffset) ||
        m_vif1NextByteOffset > 0x0fu)
    {
        setError(error, "PS2 memory state VIF segment coverage is invalid");
        return false;
    }
    uint32_t packetCount = 0u;
    if (!reader.u32(packetCount) || packetCount > kMaximumPath3PacketCount)
    {
        setError(error, "PS2 memory state PATH3 packet count is invalid");
        return false;
    }
    m_path3MaskedFifo.clear();
    m_path3MaskedFifo.reserve(packetCount);
    uint64_t path3Bytes = 0u;
    for (uint32_t index = 0u; index < packetCount; ++index)
    {
        std::vector<uint8_t> packet;
        if (!readOwnedBytes(reader, packet) ||
            path3Bytes > kMaximumDeferredBytes - packet.size())
        {
            setError(error, "PS2 memory state PATH3 payload is invalid");
            return false;
        }
        path3Bytes += packet.size();
        m_path3MaskedFifo.push_back(std::move(packet));
    }

    m_vif0Dma = {};
    m_vif1Dma = {};
    const auto readVif0Dma = [&reader](Vif0DmaState &state)
    {
        return readToken(reader, state.transfer) &&
               readEnum(reader, state.phase,
                        static_cast<uint8_t>(Vif0DmaPhase::Fault)) &&
               readEnum(reader, state.stall,
                        static_cast<uint8_t>(Vif0DmaStallReason::Fault)) &&
               reader.u32(state.madr) && reader.u32(state.qwc) &&
               reader.u32(state.tadr) && reader.u32(state.asr0) &&
               reader.u32(state.asr1) &&
               reader.u32(state.bufferedPayloadBytes) &&
               reader.u32(state.payloadByteOffset) &&
               reader.u32(state.tagsProcessed) && reader.u8(state.asp) &&
               reader.u8(state.tagId) && reader.boolean(state.active) &&
               reader.boolean(state.eventManaged) &&
               reader.boolean(state.chainMode) && reader.boolean(state.tie) &&
               reader.boolean(state.tte) && reader.boolean(state.tagIrq) &&
               reader.boolean(state.endAfterPayload) &&
               reader.boolean(state.faultReported);
    };
    const auto readVif1Dma = [&reader](Vif1DmaState &state)
    {
        return readToken(reader, state.transfer) &&
               readEnum(reader, state.phase,
                        static_cast<uint8_t>(Vif1DmaPhase::Fault)) &&
               readEnum(reader, state.stall,
                        static_cast<uint8_t>(Vif1DmaStallReason::Fault)) &&
               reader.u32(state.madr) && reader.u32(state.qwc) &&
               reader.u32(state.tadr) && reader.u32(state.asr0) &&
               reader.u32(state.asr1) && reader.u32(state.tagsProcessed) &&
               reader.u8(state.asp) && reader.u8(state.tagId) &&
               reader.boolean(state.active) &&
               reader.boolean(state.eventManaged) &&
               reader.boolean(state.chainMode) && reader.boolean(state.tie) &&
               reader.boolean(state.tte) && reader.boolean(state.tagIrq) &&
               reader.boolean(state.endAfterPayload) &&
               reader.boolean(state.faultReported);
    };
    if (!readVif0Dma(m_vif0Dma) || !readVif1Dma(m_vif1Dma))
    {
        setError(error, "PS2 memory state VIF DMA payload is invalid");
        return false;
    }

    m_gifDma = {};
    if (!readToken(reader, m_gifDma.transfer) ||
        !readEnum(reader, m_gifDma.phase,
                  static_cast<uint8_t>(GifDmaPhase::Fault)) ||
        !readEnum(reader, m_gifDma.stall,
                  static_cast<uint8_t>(GifDmaStallReason::Fault)) ||
        !reader.u32(m_gifDma.chcr) || !reader.u32(m_gifDma.madr) ||
        !reader.u32(m_gifDma.qwc) || !reader.u32(m_gifDma.tadr) ||
        !reader.u32(m_gifDma.asr0) || !reader.u32(m_gifDma.asr1) ||
        !reader.u32(m_gifDma.tagsProcessed) || !reader.u8(m_gifDma.asp) ||
        !reader.u8(m_gifDma.tagId) || !reader.boolean(m_gifDma.active) ||
        !reader.boolean(m_gifDma.eventManaged) ||
        !reader.boolean(m_gifDma.chainMode) || !reader.boolean(m_gifDma.tie) ||
        !reader.boolean(m_gifDma.tagIrq) ||
        !reader.boolean(m_gifDma.endAfterPayload) ||
        !reader.boolean(m_gifDma.faultReported) ||
        !reader.boolean(m_gifDma.copyCounted) ||
        !readOwnedBytes(reader, m_gifDma.fifoData) ||
        !readOwnedBytes(reader, m_gifDma.directCallbackData))
    {
        setError(error, "PS2 memory state GIF DMA payload is invalid");
        return false;
    }

    m_fromIpuDma = {};
    if (!readToken(reader, m_fromIpuDma.transfer) ||
        !readEnum(reader, m_fromIpuDma.phase,
                  static_cast<uint8_t>(FromIpuDmaPhase::Fault)) ||
        !readEnum(reader, m_fromIpuDma.stall,
                  static_cast<uint8_t>(FromIpuDmaStallReason::Fault)) ||
        !reader.u32(m_fromIpuDma.chcr) || !reader.u32(m_fromIpuDma.madr) ||
        !reader.u32(m_fromIpuDma.qwc) ||
        !reader.boolean(m_fromIpuDma.active) ||
        !reader.boolean(m_fromIpuDma.eventManaged) ||
        !reader.boolean(m_fromIpuDma.normalMode) ||
        !reader.boolean(m_fromIpuDma.zeroQwcStart) ||
        !reader.boolean(m_fromIpuDma.faultReported))
    {
        setError(error, "PS2 memory state from-IPU DMA payload is invalid");
        return false;
    }

    m_toIpuDma = {};
    if (!readToken(reader, m_toIpuDma.transfer) ||
        !readEnum(reader, m_toIpuDma.phase,
                  static_cast<uint8_t>(ToIpuDmaPhase::Fault)) ||
        !readEnum(reader, m_toIpuDma.stall,
                  static_cast<uint8_t>(ToIpuDmaStallReason::Fault)) ||
        !reader.u32(m_toIpuDma.chcr) || !reader.u32(m_toIpuDma.madr) ||
        !reader.u32(m_toIpuDma.qwc) || !reader.u32(m_toIpuDma.tadr) ||
        !reader.u32(m_toIpuDma.tagsProcessed) ||
        !reader.u8(m_toIpuDma.tagId) || !reader.boolean(m_toIpuDma.active) ||
        !reader.boolean(m_toIpuDma.eventManaged) ||
        !reader.boolean(m_toIpuDma.normalMode) ||
        !reader.boolean(m_toIpuDma.chainMode) ||
        !reader.boolean(m_toIpuDma.tie) ||
        !reader.boolean(m_toIpuDma.tagIrq) ||
        !reader.boolean(m_toIpuDma.endAfterPayload) ||
        !reader.boolean(m_toIpuDma.faultReported))
    {
        setError(error, "PS2 memory state to-IPU DMA payload is invalid");
        return false;
    }

    if (!readFixedBytes(reader, m_ipuOutputFifo.data(), sizeof(m_ipuOutputFifo)) ||
        !reader.u32(m_ipuOutputFifoReadIndex) ||
        !reader.u32(m_ipuOutputFifoWriteIndex) ||
        !reader.u32(m_ipuOutputFifoQwc) ||
        m_ipuOutputFifoReadIndex >= kIpuOutputFifoCapacityQwc ||
        m_ipuOutputFifoWriteIndex >= kIpuOutputFifoCapacityQwc ||
        m_ipuOutputFifoQwc > kIpuOutputFifoCapacityQwc ||
        !readFixedBytes(reader, m_ipuInputFifo.data(), sizeof(m_ipuInputFifo)) ||
        !reader.u32(m_ipuInputFifoReadIndex) ||
        !reader.u32(m_ipuInputFifoWriteIndex) ||
        !reader.u32(m_ipuInputFifoQwc) ||
        m_ipuInputFifoReadIndex >= kIpuInputFifoCapacityQwc ||
        m_ipuInputFifoWriteIndex >= kIpuInputFifoCapacityQwc ||
        m_ipuInputFifoQwc > kIpuInputFifoCapacityQwc)
    {
        setError(error, "PS2 memory state IPU FIFO payload is invalid");
        return false;
    }

    m_scratchpadDma = {};
    for (ScratchpadDmaState &state : m_scratchpadDma)
    {
        if (!readToken(reader, state.transfer) ||
            !readEnum(reader, state.phase,
                      static_cast<uint8_t>(ScratchpadDmaPhase::Fault)) ||
            !reader.u32(state.chcr) || !reader.u32(state.madr) ||
            !reader.u32(state.qwc) || !reader.u32(state.tadr) ||
            !reader.u32(state.asr0) || !reader.u32(state.asr1) ||
            !reader.u32(state.sadr) || !reader.u32(state.tagsProcessed) ||
            !reader.u8(state.asp) || !reader.u8(state.tagId) ||
            !reader.boolean(state.active) ||
            !reader.boolean(state.eventManaged) ||
            !reader.boolean(state.chainMode) || !reader.boolean(state.tie) ||
            !reader.boolean(state.tte) || !reader.boolean(state.tagIrq) ||
            !reader.boolean(state.endAfterPayload) ||
            !reader.boolean(state.faultReported))
        {
            setError(error, "PS2 memory state scratchpad DMA payload is invalid");
            return false;
        }
    }

    std::array<DmacChannelLifecycle, PS2_DMAC_CHANNEL_COUNT> channels{};
    for (DmacChannelLifecycle &channel : channels)
    {
        if (!readEnum(reader, channel.phase,
                      static_cast<uint8_t>(DmacChannelPhase::CompletionReady)) ||
            !reader.u64(channel.generation))
        {
            setError(error, "PS2 memory state DMAC lifecycle is invalid");
            return false;
        }
    }
    uint32_t readyCount = 0u;
    if (!reader.u32(readyCount) || readyCount > kMaximumPendingDmacCount)
    {
        setError(error, "PS2 memory state ready-DMAC count is invalid");
        return false;
    }
    std::vector<DmacCompletionRequest> ready(readyCount);
    for (DmacCompletionRequest &completion : ready)
    {
        if (!readToken(reader, completion.transfer) ||
            !reader.u64(completion.readySequence))
        {
            setError(error, "PS2 memory state ready-DMAC payload is invalid");
            return false;
        }
    }
    uint32_t interruptCount = 0u;
    if (!reader.u32(interruptCount) ||
        interruptCount > kMaximumPendingDmacCount)
    {
        setError(error, "PS2 memory state DMAC interrupt count is invalid");
        return false;
    }
    std::vector<DmacPendingInterrupt> interrupts(interruptCount);
    for (DmacPendingInterrupt &interrupt : interrupts)
    {
        uint8_t source = 0u;
        if (!reader.u8(source) ||
            source >= static_cast<uint8_t>(DmacChannel::Count) ||
            !reader.u32(interrupt.cause) ||
            !reader.u64(interrupt.channelGeneration) ||
            !reader.u64(interrupt.eventSequence) ||
            !reader.u64(interrupt.publicationSequence))
        {
            setError(error, "PS2 memory state DMAC interrupt payload is invalid");
            return false;
        }
        interrupt.source = static_cast<DmacChannel>(source);
    }
    uint64_t readySequence = 0u;
    uint64_t publicationSequence = 0u;
    if (!reader.u64(readySequence) || !reader.u64(publicationSequence))
    {
        setError(error, "PS2 memory state DMAC sequence is invalid");
        return false;
    }

    ps2x::timing::EeCounterBankSnapshot counterState{};
    if (!readCounterSnapshot(reader, counterState) ||
        !m_eeCounters.restore(counterState) || !reader.atEnd())
    {
        setError(error, "PS2 memory state counter payload is invalid");
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_dmacMutex);
        m_dmacChannels = channels;
        m_readyDmacCompletions = std::move(ready);
        m_pendingDmacInterrupts = std::move(interrupts);
        m_dmacReadySequence = readySequence;
        m_dmacPublicationSequence = publicationSequence;
    }

    // Rebuild-only state: these caches contain host-derived translations or
    // diagnostics and must not survive a process boundary.
    clearTlbTranslationCache();
    m_eeInstructionFetchPageCache = {};
    m_tlbTranslationCacheStats = {};
    {
        std::lock_guard<std::mutex> lock(m_codeInvalidationMutex);
        m_pendingCodeInvalidations.clear();
    }
    return true;
}
