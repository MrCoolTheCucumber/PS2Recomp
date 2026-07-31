#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>

namespace ps2_stubs
{
    inline constexpr uint32_t kSifIopHeapBase =
        0x01A00000u;
    inline constexpr uint32_t kSifIopHeapLimit =
        0x01F00000u;
    inline constexpr uint32_t kSifIopHeapAlign = 64u;

    inline constexpr uint32_t kSifRegBootStatus = 0x4u;
    inline constexpr uint32_t kSifRegMainAddr =
        0x80000000u;
    inline constexpr uint32_t kSifRegSubAddr =
        0x80000001u;
    inline constexpr uint32_t kSifRegMsCom =
        0x80000002u;
    inline constexpr uint32_t kSifBootReadyMask =
        0x00020000u;

    // Command registers, synthetic IOP heap allocations, transfer IDs, and
    // diagnostic throttles describe one emulated SIF device. They must not
    // be shared by independent PS2Runtime instances in the host process.
    struct SifRuntimeState
    {
        SifRuntimeState()
        {
            resetCommandStateLocked();
        }

        void resetCommandStateLocked()
        {
            regs.clear();
            sregs.clear();
            cmdHandlers.clear();
            cmdBuffer = 0u;
            sysCmdBuffer = 0u;
            cmdInitialized = false;
            getRegLogCount = 0u;
            setRegLogCount = 0u;

            regs[kSifRegBootStatus] =
                kSifBootReadyMask;
            regs[kSifRegMainAddr] = 0u;
            regs[kSifRegSubAddr] = 0u;
            regs[kSifRegMsCom] = 0u;
        }

        std::mutex dmaTransferMutex;
        uint32_t nextDmaTransferId = 1u;

        std::mutex commandMutex;
        std::unordered_map<uint32_t, uint32_t> regs;
        std::unordered_map<uint32_t, uint32_t> sregs;
        std::unordered_map<uint32_t, uint32_t>
            cmdHandlers;
        uint32_t cmdBuffer = 0u;
        uint32_t sysCmdBuffer = 0u;
        bool cmdInitialized = false;
        uint32_t getRegLogCount = 0u;
        uint32_t setRegLogCount = 0u;

        std::mutex heapMutex;
        std::map<uint32_t, uint32_t> heapAllocations;
        uint32_t iopHeapNext = kSifIopHeapBase;

        std::atomic<uint32_t> oversizedTransferWarnings{0u};
        std::atomic<uint32_t> failedCopyWarnings{0u};
        std::atomic<uint32_t> failedDmaWarnings{0u};
    };
}
