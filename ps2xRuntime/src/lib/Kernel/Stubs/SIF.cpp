#include "Common.h"
#include "SIF.h"
#include "Helpers/SifRuntimeState.h"
#include "../Syscalls/RPC.h"
#include "../../ps2_iop_transport.h"
#include "runtime/ps2_address.h"

#include <iterator>
#include <map>

namespace ps2_stubs
{
    void sceSifCmdIntrHdlr(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        TODO_NAMED("sceSifCmdIntrHdlr", rdram, ctx, runtime);
    }

    void sceSifLoadModule(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ps2_syscalls::SifLoadModule(rdram, ctx, runtime);
    }

    void sceSifSendCmd(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t srcAddr = getRegU32(ctx, 7); // $a3
        const uint32_t dstAddr = readStackU32(rdram, ctx, 16);
        const uint32_t size = readStackU32(rdram, ctx, 20);
        if (size != 0u && srcAddr != 0u && dstAddr != 0u)
        {
            for (uint32_t i = 0; i < size; ++i)
            {
                const uint8_t *src = getConstMemPtr(rdram, srcAddr + i);
                uint8_t *dst = getMemPtr(rdram, dstAddr + i);
                if (!src || !dst)
                {
                    break;
                }
                *dst = *src;
            }
        }

        setReturnS32(ctx, 1);
    }

    namespace
    {
        struct Ps2SifDmaTransfer
        {
            uint32_t src = 0;
            uint32_t dest = 0;
            int32_t size = 0;
            int32_t attr = 0;
        };
        static_assert(sizeof(Ps2SifDmaTransfer) == 16u, "Unexpected SIF DMA descriptor size");

        enum class SifDmaDestination
        {
            IopRam,
            SyntheticGuest,
        };

        struct PendingSifDmaTransfer
        {
            Ps2SifDmaTransfer transfer{};
            SifDmaDestination destination = SifDmaDestination::IopRam;
            uint32_t iopOffset = 0u;
        };

        SifRuntimeState *getSifRuntimeState(
            PS2Runtime *runtime)
        {
            return runtime
                ? &runtime->sifRuntimeState()
                : nullptr;
        }

        bool shouldTraceSifReg(uint32_t reg)
        {
            switch (reg)
            {
            case 0x2u:
            case 0x4u:
            case 0x80000000u:
            case 0x80000001u:
            case 0x80000002u:
                return true;
            default:
                return false;
            }
        }

        bool claimSifWarning(
            std::atomic<uint32_t> &counter)
        {
            return counter.fetch_add(
                       1u, std::memory_order_relaxed) <
                   32u;
        }

        uint32_t allocateSifDmaTransferId(
            SifRuntimeState &state)
        {
            std::lock_guard<std::mutex> lock(
                state.dmaTransferMutex);
            uint32_t id = state.nextDmaTransferId++;
            if (id == 0u)
            {
                id = state.nextDmaTransferId++;
            }
            return id;
        }

        uint32_t alignIopHeapSize(uint32_t size)
        {
            return (size + (kSifIopHeapAlign - 1u)) &
                   ~(kSifIopHeapAlign - 1u);
        }

        uint32_t allocateSifHeapBlock(
            SifRuntimeState &state,
            uint32_t requestSize)
        {
            const uint32_t alignedSize = alignIopHeapSize(requestSize);
            if (alignedSize == 0u)
            {
                return 0u;
            }

            std::lock_guard<std::mutex> lock(
                state.heapMutex);
            uint32_t candidate = kSifIopHeapBase;
            for (const auto &[addr, size] :
                 state.heapAllocations)
            {
                if (candidate + alignedSize <= addr)
                {
                    break;
                }

                const uint32_t blockEnd = alignIopHeapSize(addr + size);
                if (blockEnd > candidate)
                {
                    candidate = blockEnd;
                }
            }

            if (candidate < kSifIopHeapBase ||
                candidate + alignedSize >
                    kSifIopHeapLimit)
            {
                return 0u;
            }

            state.heapAllocations[candidate] =
                alignedSize;
            state.iopHeapNext =
                candidate + alignedSize;
            return candidate;
        }

        bool freeSifHeapBlock(
            SifRuntimeState &state,
            uint32_t addr)
        {
            std::lock_guard<std::mutex> lock(
                state.heapMutex);
            const auto it =
                state.heapAllocations.find(addr);
            if (it == state.heapAllocations.end())
            {
                return false;
            }

            state.heapAllocations.erase(it);
            if (state.heapAllocations.empty())
            {
                state.iopHeapNext =
                    kSifIopHeapBase;
            }
            return true;
        }

        void resetSifHeapState(
            SifRuntimeState &state)
        {
            std::lock_guard<std::mutex> lock(
                state.heapMutex);
            state.heapAllocations.clear();
            state.iopHeapNext = kSifIopHeapBase;
        }

        bool isCopyableGuestAddress(uint32_t addr)
        {
            return ps2IsScratchpadAddress(addr) || Ps2IsDirectRdramAddress(addr);
        }

        bool canReadGuestByteRange(const uint8_t *rdram, uint32_t srcAddr, uint32_t sizeBytes)
        {
            if (!rdram)
            {
                return false;
            }

            if (sizeBytes == 0u)
            {
                return true;
            }

            if ((sizeBytes - 1u) > (std::numeric_limits<uint32_t>::max() - srcAddr))
            {
                return false;
            }

            for (uint32_t i = 0u; i < sizeBytes; ++i)
            {
                const uint32_t srcByteAddr = srcAddr + i;
                if (!isCopyableGuestAddress(srcByteAddr) ||
                    !getConstMemPtr(rdram, srcByteAddr))
                {
                    return false;
                }
            }

            return true;
        }

        bool canCopyGuestByteRange(const uint8_t *rdram, uint32_t dstAddr, uint32_t srcAddr, uint32_t sizeBytes)
        {
            if (!canReadGuestByteRange(rdram, srcAddr, sizeBytes))
            {
                return false;
            }

            if (sizeBytes == 0u)
            {
                return true;
            }

            if ((sizeBytes - 1u) > (std::numeric_limits<uint32_t>::max() - dstAddr))
            {
                return false;
            }

            for (uint32_t i = 0u; i < sizeBytes; ++i)
            {
                const uint32_t dstByteAddr = dstAddr + i;

                if (!isCopyableGuestAddress(dstByteAddr))
                {
                    return false;
                }

                const uint8_t *dst = getConstMemPtr(rdram, dstByteAddr);
                if (!dst)
                {
                    return false;
                }
            }

            return true;
        }

        bool resolveIopRamRange(uint32_t iopAddr, uint32_t sizeBytes, uint32_t &offset)
        {
            // SIF1 destination tags carry a 24-bit IOP physical address. The
            // upper byte contains tag/control information and is not part of
            // the RAM address.
            offset = iopAddr & 0x00FFFFFFu;
            return offset <= PS2_IOP_RAM_SIZE &&
                   sizeBytes <= (PS2_IOP_RAM_SIZE - offset);
        }

        bool isAllocatedSyntheticIopRange(
            SifRuntimeState &state,
            uint32_t guestAddr,
            uint32_t sizeBytes)
        {
            uint32_t offset = 0u;
            if (!ps2ResolveDirectRdramOffset(guestAddr, offset))
            {
                return false;
            }

            const uint64_t rangeEnd =
                static_cast<uint64_t>(offset) + static_cast<uint64_t>(sizeBytes);
            std::lock_guard<std::mutex> lock(
                state.heapMutex);
            const auto upper =
                state.heapAllocations.upper_bound(offset);
            if (upper == state.heapAllocations.begin())
            {
                return false;
            }

            const auto allocation = std::prev(upper);
            const uint64_t allocationEnd =
                static_cast<uint64_t>(allocation->first) +
                static_cast<uint64_t>(allocation->second);
            return offset >= allocation->first && rangeEnd <= allocationEnd;
        }

        bool copyGuestByteRange(uint8_t *rdram, uint32_t dstAddr, uint32_t srcAddr, uint32_t sizeBytes)
        {
            if (!canCopyGuestByteRange(rdram, dstAddr, srcAddr, sizeBytes))
            {
                return false;
            }

            if (sizeBytes == 0u)
            {
                return true;
            }

            ps2TraceGuestRangeWrite(rdram, dstAddr, sizeBytes, "sifCopyGuestByteRange", nullptr);

            const uint64_t srcBegin = srcAddr;
            const uint64_t srcEnd = srcBegin + static_cast<uint64_t>(sizeBytes);
            const uint64_t dstBegin = dstAddr;
            const bool copyBackward = (dstBegin > srcBegin) && (dstBegin < srcEnd);

            if (copyBackward)
            {
                for (uint32_t i = sizeBytes; i > 0u; --i)
                {
                    const uint32_t index = i - 1u;
                    const uint8_t *src = getConstMemPtr(rdram, srcAddr + index);
                    uint8_t *dst = getMemPtr(rdram, dstAddr + index);
                    if (!src || !dst)
                    {
                        return false;
                    }
                    *dst = *src;
                }
                return true;
            }

            for (uint32_t i = 0; i < sizeBytes; ++i)
            {
                const uint8_t *src = getConstMemPtr(rdram, srcAddr + i);
                uint8_t *dst = getMemPtr(rdram, dstAddr + i);
                if (!src || !dst)
                {
                    return false;
                }
                *dst = *src;
            }
            return true;
        }
    }

    void resetSifState(PS2Runtime *runtime)
    {
        SifRuntimeState *const state =
            getSifRuntimeState(runtime);
        if (!state)
        {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(
                state->commandMutex);
            state->resetCommandStateLocked();
        }
        resetSifHeapState(*state);
        {
            std::lock_guard<std::mutex> lock(
                state->dmaTransferMutex);
            state->nextDmaTransferId = 1u;
        }
        state->oversizedTransferWarnings.store(
            0u, std::memory_order_relaxed);
        state->failedCopyWarnings.store(
            0u, std::memory_order_relaxed);
        state->failedDmaWarnings.store(
            0u, std::memory_order_relaxed);
    }

    void sceSifAddCmdHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t cid = getRegU32(ctx, 4);
        const uint32_t handler = getRegU32(ctx, 5);
        if (SifRuntimeState *const state =
                getSifRuntimeState(runtime))
        {
            std::lock_guard<std::mutex> lock(
                state->commandMutex);
            state->cmdHandlers[cid] = handler;
        }
        setReturnS32(ctx, 0);
    }

    void sceSifAllocIopHeap(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;

        const uint32_t reqSize = getRegU32(ctx, 4);
        SifRuntimeState *const state =
            getSifRuntimeState(runtime);
        setReturnU32(
            ctx,
            state
                ? allocateSifHeapBlock(
                      *state, reqSize)
                : 0u);
    }

    void sceSifAllocSysMemory(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;

        const uint32_t size = getRegU32(ctx, 5);
        SifRuntimeState *const state =
            getSifRuntimeState(runtime);
        setReturnU32(
            ctx,
            state
                ? allocateSifHeapBlock(*state, size)
                : 0u);
    }

    void sceSifBindRpc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ps2_syscalls::SifBindRpc(rdram, ctx, runtime);
    }

    void sceSifCheckStatRpc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ps2_syscalls::SifCheckStatRpc(rdram, ctx, runtime);
    }

    void sceSifDmaStat(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        const uint32_t transferId = getRegU32(ctx, 4);
        setReturnS32(
            ctx,
            runtime
                ? runtime->hleSifDmaStatus(transferId)
                : -1);
    }

    void sceSifExecRequest(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void sceSifExitCmd(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (SifRuntimeState *const state =
                getSifRuntimeState(runtime))
        {
            std::lock_guard<std::mutex> lock(
                state->commandMutex);
            state->resetCommandStateLocked();
        }
        setReturnS32(ctx, 0);
    }

    void sceSifExitRpc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void sceSifFreeIopHeap(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;

        const uint32_t addr = getRegU32(ctx, 4);
        SifRuntimeState *const state =
            getSifRuntimeState(runtime);
        setReturnS32(
            ctx,
            state && freeSifHeapBlock(*state, addr)
                ? 0
                : -1);
    }

    void sceSifFreeSysMemory(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;

        const uint32_t addr = getRegU32(ctx, 4);
        SifRuntimeState *const state =
            getSifRuntimeState(runtime);
        setReturnS32(
            ctx,
            state && freeSifHeapBlock(*state, addr)
                ? 0
                : -1);
    }

    void sceSifGetDataTable(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t cmdBuffer = 0u;
        if (SifRuntimeState *const state =
                getSifRuntimeState(runtime))
        {
            std::lock_guard<std::mutex> lock(
                state->commandMutex);
            cmdBuffer = state->cmdBuffer;
        }
        setReturnU32(ctx, cmdBuffer);
    }

    void sceSifGetIopAddr(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnU32(ctx, getRegU32(ctx, 4));
    }

    void sceSifGetNextRequest(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void sceSifGetOtherData(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SifRuntimeState *const state =
            getSifRuntimeState(runtime);
        const uint32_t rdAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);
        const uint32_t dstAddr = getRegU32(ctx, 6);
        const int32_t sizeSigned = static_cast<int32_t>(getRegU32(ctx, 7));

        if (sizeSigned <= 0)
        {
            setReturnS32(ctx, 0);
            return;
        }

        const uint32_t size = static_cast<uint32_t>(sizeSigned);
        if (size > PS2_RAM_SIZE)
        {
            if (state &&
                claimSifWarning(
                    state->oversizedTransferWarnings))
            {
                std::cerr << "sceSifGetOtherData rejected oversized transfer size=0x"
                          << std::hex << size << std::dec << std::endl;
            }
            setReturnS32(ctx, -1);
            return;
        }

        if (runtime)
        {
            PS2IopTransport::notifyTransfer(runtime, rdram, {
                ps2x::iop::SifTransferKind::GetOtherData,
                ps2x::iop::SifTransferPhase::BeforeCopy,
                srcAddr,
                dstAddr,
                size,
            });
        }

        if (!copyGuestByteRange(rdram, dstAddr, srcAddr, size))
        {
            if (state &&
                claimSifWarning(
                    state->failedCopyWarnings))
            {
                PS2_IF_AGRESSIVE_LOGS({
                    std::cerr << "sceSifGetOtherData copy failed src=0x" << std::hex << srcAddr
                              << " dst=0x" << dstAddr
                              << " size=0x" << size
                              << std::dec << std::endl;
                });
            }
            setReturnS32(ctx, -1);
            return;
        }

        // SifRpcReceiveData_t keeps src/dest/size at offsets 0x10/0x14/0x18.
        if (uint8_t *rd = getMemPtr(rdram, rdAddr))
        {
            std::memcpy(rd + 0x10u, &srcAddr, sizeof(srcAddr));
            std::memcpy(rd + 0x14u, &dstAddr, sizeof(dstAddr));
            std::memcpy(rd + 0x18u, &size, sizeof(size));
        }

        if (runtime)
        {
            PS2IopTransport::notifyTransfer(runtime, rdram, {
                ps2x::iop::SifTransferKind::GetOtherData,
                ps2x::iop::SifTransferPhase::AfterCopy,
                srcAddr,
                dstAddr,
                size,
            });
        }

        setReturnS32(ctx, 0);
    }

    void sceSifGetReg(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SifRuntimeState *const state =
            getSifRuntimeState(runtime);
        const uint32_t reg = getRegU32(ctx, 4);
        uint32_t value = 0u;
        bool shouldLog = false;
        if (state)
        {
            std::lock_guard<std::mutex> lock(
                state->commandMutex);
            auto it = state->regs.find(reg);
            if (it != state->regs.end())
            {
                value = it->second;
            }
            shouldLog =
                shouldTraceSifReg(reg) &&
                state->getRegLogCount < 128u;
            if (shouldLog)
            {
                ++state->getRegLogCount;
            }
        }
        if (shouldLog)
        {
            PS2_IF_AGRESSIVE_LOGS({
                auto flags = std::cerr.flags();
                std::cerr << "[sceSifGetReg] reg=0x" << std::hex << reg
                          << " value=0x" << value
                          << " pc=0x" << (ctx ? ctx->pc : 0u)
                          << " ra=0x" << (ctx ? getRegU32(ctx, 31) : 0u)
                          << std::dec << std::endl;
                std::cerr.flags(flags);
            });
        }
        setReturnU32(ctx, value);
    }

    void sceSifGetSreg(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SifRuntimeState *const state =
            getSifRuntimeState(runtime);
        const uint32_t reg = getRegU32(ctx, 4);
        uint32_t value = 0u;
        if (state)
        {
            std::lock_guard<std::mutex> lock(
                state->commandMutex);
            auto it = state->sregs.find(reg);
            if (it != state->sregs.end())
            {
                value = it->second;
            }
        }
        setReturnU32(ctx, value);
    }

    void sceSifInitCmd(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (SifRuntimeState *const state =
                getSifRuntimeState(runtime))
        {
            std::lock_guard<std::mutex> lock(
                state->commandMutex);
            state->cmdInitialized = true;
        }
        setReturnS32(ctx, 0);
    }

    void sceSifInitIopHeap(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (SifRuntimeState *const state =
                getSifRuntimeState(runtime))
        {
            resetSifHeapState(*state);
        }
        setReturnS32(ctx, 0);
    }

    void sceSifInitRpc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ps2_syscalls::SifInitRpc(rdram, ctx, runtime);
    }

    void sceSifIsAliveIop(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 1);
    }

    void sceSifLoadElf(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ps2_syscalls::sceSifLoadElf(rdram, ctx, runtime);
    }

    void sceSifLoadElfPart(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ps2_syscalls::sceSifLoadElfPart(rdram, ctx, runtime);
    }

    void sceSifLoadFileReset(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void sceSifLoadIopHeap(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void sceSifLoadModuleBuffer(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ps2_syscalls::sceSifLoadModuleBuffer(rdram, ctx, runtime);
    }

    void sceSifRebootIop(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 1);
    }

    void sceSifRegisterRpc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ps2_syscalls::SifRegisterRpc(rdram, ctx, runtime);
    }

    void sceSifRemoveCmdHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t cid = getRegU32(ctx, 4);
        if (SifRuntimeState *const state =
                getSifRuntimeState(runtime))
        {
            std::lock_guard<std::mutex> lock(
                state->commandMutex);
            state->cmdHandlers.erase(cid);
        }
        setReturnS32(ctx, 0);
    }

    void sceSifRemoveRpc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ps2_syscalls::SifRemoveRpc(rdram, ctx, runtime);
    }

    void sceSifRemoveRpcQueue(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ps2_syscalls::SifRemoveRpcQueue(rdram, ctx, runtime);
    }

    void sceSifResetIop(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 1);
    }

    void sceSifRpcLoop(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void sceSifSetCmdBuffer(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SifRuntimeState *const state =
            getSifRuntimeState(runtime);
        const uint32_t newBuffer = getRegU32(ctx, 4);
        uint32_t prev = 0u;
        if (state)
        {
            std::lock_guard<std::mutex> lock(
                state->commandMutex);
            prev = state->cmdBuffer;
            state->cmdBuffer = newBuffer;
        }
        setReturnU32(ctx, prev);
    }

    void isceSifSetDChain(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        sceSifSetDChain(rdram, ctx, runtime);
    }

    void isceSifSetDma(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        sceSifSetDma(rdram, ctx, runtime);
    }

    void sceSifSetDChain(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        setReturnS32(ctx, 0);
    }

    void sceSifSetDma(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SifRuntimeState *const state =
            getSifRuntimeState(runtime);
        const uint32_t dmatAddr = getRegU32(ctx, 4);
        const uint32_t count = getRegU32(ctx, 5);

        const uint32_t listAddr = getRegU32(ctx, 4);
        PS2_IF_AGRESSIVE_LOGS({
            std::cerr << "[sceSifSetDma:CALL] pc=0x" << std::hex << ctx->pc
                      << " ra=0x" << getRegU32(ctx, 31)
                      << " list=0x" << listAddr
                      << " count=" << std::dec << count
                      << std::endl;

            for (uint32_t i = 0; i < count; ++i)
            {
                const uint32_t desc = listAddr + i * 16;
                const uint32_t src = READ32(desc + 0);
                const uint32_t dst = READ32(desc + 4);
                const uint32_t size = READ32(desc + 8);
                const uint32_t attr = READ32(desc + 12);

                std::cerr << "[sceSifSetDma:DESC] i=" << i
                          << " src=0x" << std::hex << src
                          << " dst=0x" << dst
                          << " size=0x" << size
                          << " attr=0x" << attr
                          << " pc=0x" << ctx->pc
                          << " ra=0x" << getRegU32(ctx, 31)
                          << std::dec << std::endl;
            }
        });

        if (!dmatAddr || count == 0u || count > 32u)
        {
            setReturnS32(ctx, 0);
            return;
        }

        std::array<PendingSifDmaTransfer, 32u> pending{};
        uint32_t pendingCount = 0u;
        bool ok = true;
        uint8_t *const iopRam = runtime ? runtime->memory().getIOPRAM() : nullptr;
        for (uint32_t i = 0; i < count; ++i)
        {
            const uint32_t entryAddr = dmatAddr + (i * static_cast<uint32_t>(sizeof(Ps2SifDmaTransfer)));
            const uint8_t *entry = getConstMemPtr(rdram, entryAddr);
            if (!entry)
            {
                ok = false;
                break;
            }

            Ps2SifDmaTransfer xfer{};
            std::memcpy(&xfer, entry, sizeof(xfer));
            if (xfer.size <= 0)
            {
                continue;
            }

            const uint32_t sizeBytes = static_cast<uint32_t>(xfer.size);
            if (sizeBytes > PS2_RAM_SIZE)
            {
                ok = false;
                break;
            }

            if (!canReadGuestByteRange(rdram, xfer.src, sizeBytes))
            {
                ok = false;
                break;
            }

            uint32_t iopOffset = 0u;
            if (resolveIopRamRange(xfer.dest, sizeBytes, iopOffset))
            {
                if (!iopRam)
                {
                    ok = false;
                    break;
                }
                pending[pendingCount++] = {
                    xfer,
                    SifDmaDestination::IopRam,
                    iopOffset,
                };
                continue;
            }

            // The high-level IOP compatibility layer historically returns
            // synthetic pointers backed by unused EE RAM. Keep that extension
            // only for ranges explicitly allocated by the SIF heap; arbitrary
            // IOP destinations must never alias and overwrite EE memory.
            if (!state ||
                !isAllocatedSyntheticIopRange(
                    *state, xfer.dest, sizeBytes) ||
                !canCopyGuestByteRange(rdram, xfer.dest, xfer.src, sizeBytes))
            {
                ok = false;
                break;
            }
            pending[pendingCount++] = {
                xfer,
                SifDmaDestination::SyntheticGuest,
                0u,
            };
        }

        PS2Runtime::HleSifDmaSubmission submission{};
        if (ok)
        {
            for (uint32_t i = 0; i < pendingCount; ++i)
            {
                const PendingSifDmaTransfer &pendingTransfer = pending[i];
                const Ps2SifDmaTransfer &xfer = pendingTransfer.transfer;
                PS2Runtime::HleSifDmaTransfer &scheduled =
                    submission.transfers[
                        submission.transferCount++];
                scheduled.src = xfer.src;
                scheduled.dest = xfer.dest;
                scheduled.size =
                    static_cast<uint32_t>(xfer.size);
                scheduled.iopOffset =
                    pendingTransfer.iopOffset;
                scheduled.destination =
                    pendingTransfer.destination ==
                            SifDmaDestination::IopRam
                        ? PS2Runtime::
                              HleSifDmaDestination::IopRam
                        : PS2Runtime::
                              HleSifDmaDestination::
                                  SyntheticGuest;
            }
        }

        if (!ok)
        {
            if (state &&
                claimSifWarning(
                    state->failedDmaWarnings))
            {
                PS2_IF_AGRESSIVE_LOGS({
                    std::cerr << "sceSifSetDma failed dmat=0x" << std::hex << dmatAddr
                              << " count=0x" << count
                              << std::dec << std::endl;
                });
            }
            setReturnS32(ctx, 0);
            return;
        }

        if (!runtime || !state)
        {
            setReturnS32(ctx, 0);
            return;
        }

        submission.transferId =
            allocateSifDmaTransferId(*state);
        if (!runtime->submitHleSifDma(
                rdram, ctx, submission))
        {
            setReturnS32(ctx, 0);
            return;
        }

        setReturnS32(
            ctx,
            static_cast<int32_t>(
                submission.transferId));
    }

    void sceSifSetIopAddr(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnU32(ctx, getRegU32(ctx, 5));
    }

    void sceSifSetReg(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SifRuntimeState *const state =
            getSifRuntimeState(runtime);
        const uint32_t reg = getRegU32(ctx, 4);
        const uint32_t value = getRegU32(ctx, 5);
        uint32_t prev = 0u;
        bool shouldLog = false;
        if (state)
        {
            std::lock_guard<std::mutex> lock(
                state->commandMutex);
            auto it = state->regs.find(reg);
            if (it != state->regs.end())
            {
                prev = it->second;
            }
            state->regs[reg] = value;
            shouldLog =
                shouldTraceSifReg(reg) &&
                state->setRegLogCount < 128u;
            if (shouldLog)
            {
                ++state->setRegLogCount;
            }
        }
        if (shouldLog)
        {
            PS2_IF_AGRESSIVE_LOGS({
                auto flags = std::cerr.flags();
                std::cerr << "[sceSifSetReg] reg=0x" << std::hex << reg
                          << " prev=0x" << prev
                          << " value=0x" << value
                          << " pc=0x" << (ctx ? ctx->pc : 0u)
                          << " ra=0x" << (ctx ? getRegU32(ctx, 31) : 0u)
                          << std::dec << std::endl;
                std::cerr.flags(flags);
            });
        }
        setReturnU32(ctx, prev);
    }

    void sceSifSetRpcQueue(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ps2_syscalls::SifSetRpcQueue(rdram, ctx, runtime);
    }

    void sceSifSetSreg(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SifRuntimeState *const state =
            getSifRuntimeState(runtime);
        const uint32_t reg = getRegU32(ctx, 4);
        const uint32_t value = getRegU32(ctx, 5);
        uint32_t prev = 0u;
        if (state)
        {
            std::lock_guard<std::mutex> lock(
                state->commandMutex);
            auto it = state->sregs.find(reg);
            if (it != state->sregs.end())
            {
                prev = it->second;
            }
            state->sregs[reg] = value;
        }
        setReturnU32(ctx, prev);
    }

    void sceSifSetSysCmdBuffer(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        SifRuntimeState *const state =
            getSifRuntimeState(runtime);
        const uint32_t newBuffer = getRegU32(ctx, 4);
        uint32_t prev = 0u;
        if (state)
        {
            std::lock_guard<std::mutex> lock(
                state->commandMutex);
            prev = state->sysCmdBuffer;
            state->sysCmdBuffer = newBuffer;
        }
        setReturnU32(ctx, prev);
    }

    void sceSifStopDma(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void sceSifSyncIop(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 1);
    }

    void sceSifWriteBackDCache(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }
}
