#include "MiniTest.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_gs_psmct32.h"
#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"
#include "ps2_syscalls.h"
#include "Stubs/DMA.h"
#include "Stubs/GS.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace
{
    uint32_t makeVifCmd(uint8_t opcode, uint8_t num, uint16_t imm)
    {
        return (static_cast<uint32_t>(opcode) << 24) |
               (static_cast<uint32_t>(num) << 16) |
               static_cast<uint32_t>(imm);
    }

    void setRegU32(R5900Context &ctx, int reg, uint32_t value)
    {
        ctx.r[reg] = _mm_set_epi64x(0, static_cast<int64_t>(value));
    }

    void appendU32(std::vector<uint8_t> &dst, uint32_t value)
    {
        const size_t pos = dst.size();
        dst.resize(pos + sizeof(uint32_t));
        std::memcpy(dst.data() + pos, &value, sizeof(uint32_t));
    }

    void appendU64(std::vector<uint8_t> &dst, uint64_t value)
    {
        const size_t pos = dst.size();
        dst.resize(pos + sizeof(uint64_t));
        std::memcpy(dst.data() + pos, &value, sizeof(uint64_t));
    }

    size_t publishDmacCompletions(
        PS2Memory &memory,
        uint64_t eventSequence = 1u)
    {
        return memory.publishReadyDmacCompletions(
            eventSequence);
    }

    uint64_t makeDmaTag(uint16_t qwc, uint8_t id, uint32_t addr, bool irq = false)
    {
        return static_cast<uint64_t>(qwc) |
               (static_cast<uint64_t>(id & 0x7u) << 28) |
               (irq ? (1ull << 31) : 0ull) |
               (static_cast<uint64_t>(addr & 0x7FFFFFFFu) << 32);
    }

    void writeDmaTag(uint8_t *rdram, uint32_t tagAddr, uint64_t tagLo)
    {
        std::memset(rdram + tagAddr, 0, 16);
        std::memcpy(rdram + tagAddr, &tagLo, sizeof(tagLo));
    }

    void writeU64(uint8_t *rdram, uint32_t addr, uint64_t value)
    {
        std::memcpy(rdram + addr, &value, sizeof(value));
    }

    uint64_t makeGifTag(uint16_t nloop, uint8_t flg, uint8_t nreg, bool eop);

    uint64_t makeBitbltbuf(uint32_t dbp, uint32_t dbw, uint32_t dpsm)
    {
        return (static_cast<uint64_t>(dbp & 0x3FFFu) << 32) |
               (static_cast<uint64_t>(dbw & 0x3Fu) << 48) |
               (static_cast<uint64_t>(dpsm & 0x3Fu) << 56);
    }

    uint32_t writeGifAd(uint8_t *rdram, uint32_t addr, uint64_t value, uint64_t reg)
    {
        writeU64(rdram, addr + 0u, value);
        writeU64(rdram, addr + 8u, reg);
        return addr + 16u;
    }

    uint32_t writeTextureUploadSetup(uint8_t *rdram, uint32_t addr, uint32_t dbp, uint32_t dpsm)
    {
        writeDmaTag(rdram, addr, makeDmaTag(5u, 1u, 0u, false)); // CNT: setup tag + four A+D writes.
        addr += 16u;
        writeU64(rdram, addr + 0u, makeGifTag(4u, GIF_FMT_PACKED, 1u, false));
        writeU64(rdram, addr + 8u, 0x0Eull); // GIF PACKED A+D descriptor.
        addr += 16u;
        addr = writeGifAd(rdram, addr, makeBitbltbuf(dbp, 1u, dpsm), GS_REG_BITBLTBUF);
        addr = writeGifAd(rdram, addr, 0ull, GS_REG_TRXPOS);
        addr = writeGifAd(rdram, addr, (16ull << 0) | (16ull << 32), GS_REG_TRXREG);
        addr = writeGifAd(rdram, addr, 0ull, GS_REG_TRXDIR);
        return addr;
    }

    uint32_t writeTextureImageRef(uint8_t *rdram, uint32_t addr, uint32_t qwc, uint32_t dataAddr)
    {
        writeDmaTag(rdram, addr, makeDmaTag(1u, 1u, 0u, false)); // CNT: GIF IMAGE tag.
        addr += 16u;
        writeU64(rdram, addr + 0u, makeGifTag(static_cast<uint16_t>(qwc), GIF_FMT_IMAGE, 0u, false));
        writeU64(rdram, addr + 8u, 0ull);
        addr += 16u;
        writeDmaTag(rdram, addr, makeDmaTag(static_cast<uint16_t>(qwc), 3u, dataAddr, false)); // REF: image payload.
        return addr + 16u;
    }

    uint64_t makeGifTag(uint16_t nloop, uint8_t flg, uint8_t nreg, bool eop = true)
    {
        uint64_t tag = static_cast<uint64_t>(nloop & 0x7FFFu);
        if (eop)
            tag |= (1ull << 15);
        tag |= (static_cast<uint64_t>(flg & 0x3u) << 58);
        tag |= (static_cast<uint64_t>(nreg & 0xFu) << 60);
        return tag;
    }

    uint64_t makeGifTagPrim(uint16_t nloop, uint16_t prim, uint8_t flg, uint8_t nreg, bool eop = true, bool pre = true)
    {
        uint64_t tag = makeGifTag(nloop, flg, nreg, eop);
        if (pre)
            tag |= (1ull << 46);
        tag |= (static_cast<uint64_t>(prim & 0x7FFu) << 47);
        return tag;
    }

    uint64_t makeGsFrame(uint32_t fbp, uint32_t fbw, uint32_t psm, uint32_t mask = 0u)
    {
        return static_cast<uint64_t>(fbp & 0x1FFu) |
               (static_cast<uint64_t>(fbw & 0x3Fu) << 16u) |
               (static_cast<uint64_t>(psm & 0x3Fu) << 24u) |
               (static_cast<uint64_t>(mask) << 32u);
    }

    uint64_t makeGsScissor(uint32_t x0, uint32_t x1, uint32_t y0, uint32_t y1)
    {
        return static_cast<uint64_t>(x0 & 0x7FFu) |
               (static_cast<uint64_t>(x1 & 0x7FFu) << 16u) |
               (static_cast<uint64_t>(y0 & 0x7FFu) << 32u) |
               (static_cast<uint64_t>(y1 & 0x7FFu) << 48u);
    }

    void appendPackedRgbaq(std::vector<uint8_t> &packet, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        appendU64(packet, static_cast<uint64_t>(r) | (static_cast<uint64_t>(g) << 32u));
        appendU64(packet, static_cast<uint64_t>(b) | (static_cast<uint64_t>(a) << 32u));
    }

    void appendPackedXyzf2(std::vector<uint8_t> &packet, uint32_t x, uint32_t y, uint32_t z)
    {
        appendU64(packet, static_cast<uint64_t>(x & 0xFFFFu) | (static_cast<uint64_t>(y & 0xFFFFu) << 32u));
        appendU64(packet, static_cast<uint64_t>(z & 0xFFFFFFu) << 4u);
    }

    void appendPackedUv(std::vector<uint8_t> &packet, uint32_t u, uint32_t v)
    {
        appendU64(packet, static_cast<uint64_t>(u & 0x3FFFu) | (static_cast<uint64_t>(v & 0x3FFFu) << 32u));
        appendU64(packet, 0u);
    }

}

void register_ps2_memory_tests()
{
    MiniTest::Case("PS2Memory", [](TestCase &tc)
    {
        tc.Run("uncached aliases map to same RDRAM bytes", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            mem.write32(0x00001000u, 0xDEADBEEFu);
            t.Equals(mem.read32(0x00001000u), 0xDEADBEEFu, "base readback should match");
            t.Equals(mem.read32(0x20001000u), 0xDEADBEEFu, "0x2000_0000 alias should map to RDRAM");

            // 0x3010_0000 maps to physical 0x0010_0000 (AboutPS2 memory map).
            mem.write32(0x00101000u, 0xDEADBEEFu);
            t.Equals(mem.read32(0x30101000u), 0xDEADBEEFu, "0x3010_0000 accelerated alias should map to RDRAM");

            mem.write32(0x20002000u, 0x13579BDFu);
            t.Equals(mem.read32(0x00002000u), 0x13579BDFu, "writes through 0x2000 alias should land in base RDRAM");

            mem.write32(0x30103000u, 0x2468ACE0u);
            t.Equals(mem.read32(0x00103000u), 0x2468ACE0u, "writes through 0x3010 alias should land in base RDRAM");
        });

        tc.Run("translateAddress handles kseg and uncached aliases", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            t.Equals(mem.translateAddress(0x80001234u), 0x00001234u, "KSEG0 should map directly to physical");
            t.Equals(mem.translateAddress(0xA0005678u), 0x00005678u, "KSEG1 should map directly to physical");
            t.Equals(mem.translateAddress(0x20001234u), 0x00001234u, "0x2000 uncached alias should map to RAM");
            t.Equals(mem.translateAddress(0x30105678u), 0x00105678u, "0x3010 accelerated alias should map to RAM");
            t.Equals(mem.translateAddress(PS2_SCRATCHPAD_BASE + 0x123u), 0x123u, "scratchpad base should translate to local offset");
            t.Equals(mem.translateAddress(PS2_SCRATCHPAD_ALIAS_BASE + 0x123u), 0x123u, "0xF000 scratchpad alias should translate to local offset");
        });

        tc.Run("guest pointers reject non-RDRAM address spaces", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");
            uint8_t *const rdram = mem.getRDRAM();

            t.IsTrue(getMemPtr(rdram, 0x00000100u) == rdram + 0x100u,
                     "physical RDRAM pointer should resolve directly");
            t.IsTrue(getMemPtr(rdram, 0x20000100u) == rdram + 0x100u,
                     "uncached RDRAM pointer should resolve to its physical offset");
            t.IsTrue(getMemPtr(rdram, 0x30100100u) == rdram + 0x00100100u,
                     "accelerated RDRAM pointer should resolve to its physical offset");
            t.IsTrue(getMemPtr(rdram, 0x80000100u) == rdram + 0x100u,
                     "KSEG0 RDRAM pointer should resolve to its physical offset");
            t.IsTrue(getMemPtr(rdram, 0xA0000100u) == rdram + 0x100u,
                     "KSEG1 RDRAM pointer should resolve to its physical offset");
            uint8_t *const scratchpad = ps2GetScratchpadHostPtr();
            t.IsNotNull(scratchpad, "scratchpad host storage should be initialized");
            t.IsTrue(getMemPtr(rdram, PS2_SCRATCHPAD_ALIAS_BASE + 0x100u) ==
                         scratchpad + 0x100u,
                     "scratchpad alias should still resolve to scratchpad storage");

            t.IsNull(getMemPtr(rdram, 0x02000000u),
                     "address immediately beyond RDRAM must not wrap to offset zero");
            t.IsNull(getMemPtr(rdram, 0x10000000u),
                     "MMIO must not be exposed as an RDRAM host pointer");
            t.IsNull(getMemPtr(rdram, 0x22000000u),
                     "address beyond the uncached mirror must not wrap to RDRAM");
            t.IsNull(getMemPtr(rdram, 0x30000000u),
                     "unmapped first megabyte of the accelerated segment must be rejected");
            t.IsNull(getMemPtr(rdram, 0xC0000000u),
                     "TLB-mapped segments must not be treated as direct host pointers");
        });

        tc.Run("EE counters use emulated cycles and never host time", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kTimer0Count = 0x10000000u;
            constexpr uint32_t kTimer0Mode = 0x10000010u;
            constexpr uint32_t kTimer0Compare = 0x10000020u;
            uint64_t eeCycle = 0u;
            mem.setEeCounterCycleCallback(
                [&eeCycle]()
                {
                    return eeCycle;
                });

            t.IsTrue(mem.writeIORegister(kTimer0Count, 0u), "timer count reset write should succeed");
            t.IsTrue(mem.writeIORegister(kTimer0Compare, 3u), "timer compare write should succeed");
            t.IsTrue(mem.writeIORegister(kTimer0Mode, 0x82u), "timer mode write should be retained");
            t.Equals(mem.readIORegister(kTimer0Mode), 0x82u, "timer mode should be readable");

            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            t.Equals(
                mem.readIORegister(kTimer0Count),
                0u,
                "host sleep must not advance an EE hardware counter");

            eeCycle = 512u;
            const uint32_t firstCount =
                mem.readIORegister(kTimer0Count);
            t.Equals(
                firstCount, 1u,
                "clock source two should advance once per 512 emulated EE cycles");

            t.IsTrue(mem.writeIORegister(kTimer0Count, 0u), "timer count second reset should succeed");
            t.Equals(
                mem.readIORegister(kTimer0Count),
                0u,
                "timer reset should be visible at the same emulated cycle");
            eeCycle = 1024u;
            t.Equals(
                mem.readIORegister(kTimer0Count),
                1u,
                "timer reset should restart from the canonical source phase");
        });

        tc.Run("EE counter deadlines publish INTC state in device order", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kTimer0Count = 0x10000000u;
            constexpr uint32_t kTimer0Mode = 0x10000010u;
            constexpr uint32_t kTimer0Compare = 0x10000020u;
            constexpr uint32_t kIntcStat = 0x1000F000u;
            constexpr uint32_t kIntcMask = 0x1000F010u;
            uint64_t eeCycle = 0u;
            std::optional<uint64_t> deadline;
            uint32_t callbackStatus = 0u;
            uint32_t newlyRaised = 0u;
            mem.setEeCounterCycleCallback(
                [&eeCycle]()
                {
                    return eeCycle;
                });
            mem.setEeCounterScheduleCallback(
                [&deadline](std::optional<uint64_t> value)
                {
                    deadline = value;
                });
            mem.setEeCounterInterruptStateCallback(
                [&callbackStatus, &newlyRaised](
                    uint32_t status,
                    uint32_t,
                    uint32_t raised)
                {
                    callbackStatus = status;
                    newlyRaised |= raised;
                });

            (void)mem.writeIORegister(kTimer0Compare, 3u);
            (void)mem.writeIORegister(kTimer0Count, 0u);
            (void)mem.writeIORegister(kTimer0Mode, 0x1c0u);
            t.IsTrue(
                deadline.has_value(),
                "target-enabled counter should publish a scheduler deadline");
            if (deadline.has_value())
            {
                t.Equals(
                    *deadline, 6u,
                    "counter deadline should use the selected BUSCLK divisor");
            }

            eeCycle = 6u;
            t.Equals(
                mem.readIORegister(kTimer0Count),
                0u,
                "zero return should be visible before interrupt publication");
            t.IsTrue(
                (mem.readIORegister(kTimer0Mode) &
                 0x400u) != 0u,
                "counter mode should latch compare status");
            t.IsTrue(
                (mem.readIORegister(kIntcStat) &
                 (1u << 9u)) != 0u,
                "counter0 should latch INTC timer cause 9");
            t.IsTrue(
                (callbackStatus & (1u << 9u)) != 0u &&
                    (newlyRaised & (1u << 9u)) != 0u,
                "interrupt callback should observe state after device publication");

            mem.write8(kTimer0Mode, 0u);
            t.IsTrue(
                (mem.readIORegister(kTimer0Mode) &
                 0x400u) != 0u,
                "an unselected timer W1C status bit must survive SB");

            (void)mem.writeIORegister(
                kIntcMask, 1u << 9u);
            t.IsTrue(
                (mem.readIORegister(kIntcMask) &
                 (1u << 9u)) != 0u,
                "INTC mask writes should toggle selected bits");
            mem.write8(kIntcMask, 0u);
            t.IsTrue(
                (mem.readIORegister(kIntcMask) &
                 (1u << 9u)) != 0u,
                "an unselected INTC mask bit must not be replayed by SB");
            (void)mem.writeIORegister(
                kIntcStat, 1u << 9u);
            t.IsTrue(
                (mem.readIORegister(kIntcStat) &
                 (1u << 9u)) == 0u,
                "INTC status should clear selected bits on write-one");
        });

        tc.Run("scratchpad alias accesses the same bytes as base", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kOffset = 0x140u;
            constexpr uint32_t kScratchAddr = PS2_SCRATCHPAD_BASE + kOffset;
            constexpr uint32_t kScratchAliasAddr = PS2_SCRATCHPAD_ALIAS_BASE + kOffset;

            mem.write32(kScratchAliasAddr, 0xCAFEBABEu);
            t.Equals(mem.read32(kScratchAddr), 0xCAFEBABEu, "writes through 0xF000 scratchpad alias should land in scratchpad");
            t.Equals(mem.read32(kScratchAliasAddr), 0xCAFEBABEu, "reads through 0xF000 scratchpad alias should see scratchpad bytes");
        });

        tc.Run("VU0 code and data windows map through EE addresses", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kCodeAddr = PS2_VU0_CODE_BASE + 0x20u;
            mem.write32(kCodeAddr, 0x11223344u);
            t.Equals(mem.read32(kCodeAddr), 0x11223344u, "VU0 code readback should match written word");

            uint32_t codeWord = 0u;
            std::memcpy(&codeWord, mem.getVU0Code() + 0x20u, sizeof(codeWord));
            t.Equals(codeWord, 0x11223344u, "VU0 code write should land in micro memory buffer");

            constexpr uint32_t kDataAddr = PS2_VU0_DATA_BASE + 0x30u;
            const __m128i value = _mm_set_epi32(0x44556677u, 0x01234567u, 0x89ABCDEFu, 0xCAFEBABEu);
            mem.write128(kDataAddr, value);

            alignas(16) uint32_t words[4]{};
            const __m128i readback = mem.read128(kDataAddr);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(words), readback);

            t.Equals(words[0], 0xCAFEBABEu, "VU0 data lane 0 should match");
            t.Equals(words[1], 0x89ABCDEFu, "VU0 data lane 1 should match");
            t.Equals(words[2], 0x01234567u, "VU0 data lane 2 should match");
            t.Equals(words[3], 0x44556677u, "VU0 data lane 3 should match");
        });

        tc.Run("VU code generations track code writes and ignore data writes", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            uint64_t vu0Generation = mem.getVU0CodeGeneration();
            uint64_t vu1Generation = mem.getVU1CodeGeneration();
            t.IsTrue(vu0Generation != 0u, "VU0 initialization should publish a code generation");
            t.IsTrue(vu1Generation != 0u, "VU1 initialization should publish a code generation");

            mem.write8(PS2_VU0_CODE_BASE, 0x11u);
            t.Equals(mem.getVU0CodeGeneration(), ++vu0Generation,
                     "an 8-bit VU0 code write should invalidate decoded code");

            mem.write16(PS2_VU0_CODE_BASE + 2u, 0x2233u);
            t.Equals(mem.getVU0CodeGeneration(), ++vu0Generation,
                     "a 16-bit VU0 code write should invalidate decoded code");

            mem.write32(PS2_VU0_CODE_BASE + 4u, 0x44556677u);
            t.Equals(mem.getVU0CodeGeneration(), ++vu0Generation,
                     "a 32-bit VU0 code write should invalidate decoded code");

            mem.write64(PS2_VU0_CODE_BASE + 8u, 0x8899AABBCCDDEEFFull);
            t.Equals(mem.getVU0CodeGeneration(), ++vu0Generation,
                     "a 64-bit VU0 code write should invalidate decoded code");

            mem.write128(
                PS2_VU0_CODE_BASE + 16u,
                _mm_set_epi32(4, 3, 2, 1));
            t.Equals(mem.getVU0CodeGeneration(), ++vu0Generation,
                     "a 128-bit VU0 code write should invalidate decoded code");

            mem.write32(PS2_VU0_DATA_BASE, 0x12345678u);
            t.Equals(mem.getVU0CodeGeneration(), vu0Generation,
                     "VU0 data writes should retain the decoded-code generation");
            t.Equals(mem.getVU1CodeGeneration(), vu1Generation,
                     "VU0 writes should not invalidate VU1 decoded code");

            mem.write32(PS2_VU1_CODE_BASE, 0x89ABCDEFu);
            t.Equals(mem.getVU0CodeGeneration(), vu0Generation,
                     "VU1 code writes should not invalidate VU0 decoded code");
            t.Equals(mem.getVU1CodeGeneration(), ++vu1Generation,
                     "VU1 code writes should retain their existing invalidation behavior");
        });

        tc.Run("fast memory helpers wrap safely at RAM boundary", [](TestCase &t)
        {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            const uint32_t tail = PS2_RAM_SIZE - 4u;

            // Build a wrapped 64-bit pattern: [tail..tail+3] + [0..3]
            rdram[tail + 0u] = 0xA1u;
            rdram[tail + 1u] = 0xB2u;
            rdram[tail + 2u] = 0xC3u;
            rdram[tail + 3u] = 0xD4u;
            rdram[0u] = 0x11u;
            rdram[1u] = 0x22u;
            rdram[2u] = 0x33u;
            rdram[3u] = 0x44u;

            const uint64_t wrappedRead = Ps2FastRead64(rdram.data(), tail);
            t.Equals(wrappedRead, 0x44332211D4C3B2A1ull,
                     "Ps2FastRead64 should wrap across the 32MB boundary");

            Ps2FastWrite64(rdram.data(), tail, 0x8877665544332211ull);
            t.Equals(static_cast<uint32_t>(rdram[tail + 0u]), 0x11u, "write byte 0 should land at tail+0");
            t.Equals(static_cast<uint32_t>(rdram[tail + 1u]), 0x22u, "write byte 1 should land at tail+1");
            t.Equals(static_cast<uint32_t>(rdram[tail + 2u]), 0x33u, "write byte 2 should land at tail+2");
            t.Equals(static_cast<uint32_t>(rdram[tail + 3u]), 0x44u, "write byte 3 should land at tail+3");
            t.Equals(static_cast<uint32_t>(rdram[0u]), 0x55u, "write byte 4 should wrap to address 0");
            t.Equals(static_cast<uint32_t>(rdram[1u]), 0x66u, "write byte 5 should wrap to address 1");
            t.Equals(static_cast<uint32_t>(rdram[2u]), 0x77u, "write byte 6 should wrap to address 2");
            t.Equals(static_cast<uint32_t>(rdram[3u]), 0x88u, "write byte 7 should wrap to address 3");
        });

        tc.Run("hybrid memory macros validate aliases and alignment", [](TestCase &t)
        {
            PS2Runtime runtimeStorage;
            t.IsTrue(runtimeStorage.memory().initialize(), "PS2Memory initialize should succeed");
            PS2Runtime *runtime = &runtimeStorage;
            uint8_t *rdram = runtime->memory().getRDRAM();
            R5900Context loadContext{};
            // ERL makes this fixture's physical-style low probes direct;
            // BEM retains the original masked bus-error behavior.
            loadContext.cop0_status = 0x00001004u;
            R5900Context *ctx = &loadContext;

            const uint32_t baseValue = 0x11223344u;
            std::memcpy(rdram, &baseValue, sizeof(baseValue));

            t.Equals(READ32(0x22000000u), 0u,
                     "masked bus-error address should use the checked path");
            t.Equals(READ32(0x30000000u), 0u,
                     "masked accelerated bus-error address should not wrap to RDRAM");
            WRITE32(0x22000000u, 0xAABBCCDDu);

            uint32_t unchangedValue = 0u;
            std::memcpy(&unchangedValue, rdram, sizeof(unchangedValue));
            t.Equals(unchangedValue, baseValue,
                     "invalid direct write should not modify wrapped RDRAM");

            const uint32_t acceleratedValue = 0x55667788u;
            std::memcpy(rdram + 0x00100000u, &acceleratedValue, sizeof(acceleratedValue));
            t.Equals(READ32(0x30100000u), acceleratedValue,
                     "valid accelerated alias should retain the fast RDRAM path");

            bool loadRaised = false;
            loadContext.pc = 0x1000u;
            try
            {
                (void)READ32(0x00000002u);
            }
            catch (const PS2GuestException &)
            {
                loadRaised = true;
            }
            t.IsTrue(loadRaised, "misaligned fast-path load should raise an EE exception");
            t.Equals(loadContext.cop0_badvaddr, 0x00000002u,
                     "misaligned load should retain the original virtual address");

            R5900Context storeContext{};
            storeContext.cop0_status = 0x00000004u;
            ctx = &storeContext;
            storeContext.pc = 0x2000u;
            bool storeRaised = false;
            try
            {
                WRITE32(0x00000002u, 0xDEADBEEFu);
            }
            catch (const PS2GuestException &)
            {
                storeRaised = true;
            }
            t.IsTrue(storeRaised, "misaligned fast-path store should raise an EE exception");
            t.Equals(storeContext.cop0_badvaddr, 0x00000002u,
                     "misaligned store should retain the original virtual address");

            const uint32_t maskedBase = 0xA1B2C3D4u;
            std::memcpy(rdram + 0x20u, &maskedBase, sizeof(maskedBase));
            WRITE_MASKED32(0x20u, 0x00EEDD00u, 0x6u);
            uint32_t maskedResult = 0u;
            std::memcpy(&maskedResult, rdram + 0x20u, sizeof(maskedResult));
            t.Equals(maskedResult, 0xA1EEDDD4u,
                     "fast masked word writes should preserve disabled byte lanes");

            const uint64_t maskedDoubleBase = 0x8877665544332211ull;
            std::memcpy(rdram + 0x28u, &maskedDoubleBase, sizeof(maskedDoubleBase));
            WRITE_MASKED64(0x28u, 0x0000DDEECCBB0000ull, 0x3Cu);
            uint64_t maskedDoubleResult = 0u;
            std::memcpy(&maskedDoubleResult, rdram + 0x28u, sizeof(maskedDoubleResult));
            t.Equals(maskedDoubleResult, 0x8877DDEECCBB2211ull,
                     "fast masked doubleword writes should preserve disabled byte lanes");
        });

        tc.Run("VIF MPG num zero uploads 256 instructions", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            std::vector<uint8_t> packet;
            packet.reserve(4u + 2048u);
            appendU32(packet, makeVifCmd(0x4Au, 0u, 0u)); // MPG, num=0 -> 256 instructions (2048 bytes)

            for (uint32_t i = 0; i < 2048u; ++i)
            {
                packet.push_back(static_cast<uint8_t>(i & 0xFFu));
            }

            std::memset(mem.getVU1Code(), 0, PS2_VU1_CODE_SIZE);
            mem.processVIF1Data(packet.data(), static_cast<uint32_t>(packet.size()));

            const uint8_t *vu1Code = mem.getVU1Code();
            bool matches = true;
            for (uint32_t i = 0; i < 2048u; ++i)
            {
                if (vu1Code[i] != static_cast<uint8_t>(i & 0xFFu))
                {
                    matches = false;
                    break;
                }
            }
            t.IsTrue(matches, "MPG num=0 should copy 2048 bytes into VU1 code memory");
        });

        tc.Run("VIF MPG wraps microprogram uploads at each MicroMem boundary", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            const auto checkWrappedUpload =
                [&](bool vu1, uint16_t immediate,
                    uint8_t *code, uint32_t codeSize)
            {
                std::array<uint8_t, 16u> payload{};
                for (uint32_t index = 0u;
                     index < payload.size(); ++index)
                {
                    payload[index] = static_cast<uint8_t>(
                        0x40u + index);
                }

                std::vector<uint8_t> packet;
                appendU32(
                    packet,
                    makeVifCmd(0x4au, 2u, immediate));
                packet.insert(
                    packet.end(), payload.begin(), payload.end());
                std::memset(code, 0xcc, codeSize);

                const uint64_t generationBefore =
                    vu1
                        ? mem.getVU1CodeGeneration()
                        : mem.getVU0CodeGeneration();
                if (vu1)
                {
                    mem.processVIF1Data(
                        packet.data(),
                        static_cast<uint32_t>(packet.size()));
                }
                else
                {
                    mem.processVIF0Data(
                        packet.data(),
                        static_cast<uint32_t>(packet.size()));
                }

                t.IsTrue(
                    std::memcmp(
                        code + codeSize - 8u,
                        payload.data(), 8u) == 0,
                    "MPG should write the tail before wrapping");
                t.IsTrue(
                    std::memcmp(
                        code, payload.data() + 8u, 8u) == 0,
                    "MPG should continue at MicroMem offset zero");
                t.Equals(
                    vu1
                        ? mem.getVU1CodeGeneration()
                        : mem.getVU0CodeGeneration(),
                    generationBefore + 1u,
                    "one wrapped MPG command should publish one generation");
            };

            checkWrappedUpload(
                false, 0x01ffu,
                mem.getVU0Code(), PS2_VU0_CODE_SIZE);
            checkWrappedUpload(
                true, 0x07ffu,
                mem.getVU1Code(), PS2_VU1_CODE_SIZE);
        });

        tc.Run("VIF UNPACK num zero uploads 256 vectors", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            // UNPACK V4_32: opcode 0x6C (vn=3, vl=0), num=0 => 256 vectors, 16 bytes each.
            std::vector<uint8_t> packet;
            packet.reserve(4u + 4096u);
            appendU32(packet, makeVifCmd(0x6Cu, 0u, 0u));
            for (uint32_t i = 0; i < 4096u; ++i)
            {
                packet.push_back(static_cast<uint8_t>((i * 3u) & 0xFFu));
            }

            std::memset(mem.getVU1Data(), 0, PS2_VU1_DATA_SIZE);
            mem.processVIF1Data(packet.data(), static_cast<uint32_t>(packet.size()));

            const uint8_t *vu1Data = mem.getVU1Data();
            bool matches = true;
            for (uint32_t i = 0; i < 4096u; ++i)
            {
                if (vu1Data[i] != static_cast<uint8_t>((i * 3u) & 0xFFu))
                {
                    matches = false;
                    break;
                }
            }
            t.IsTrue(matches, "UNPACK num=0 should copy 256 V4_32 vectors (4096 bytes)");
        });

        tc.Run("VIF control commands update MARK MASK ROW and COL registers", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            std::vector<uint8_t> packet;
            appendU32(packet, makeVifCmd(0x07u, 0u, 0x1234u)); // MARK

            appendU32(packet, makeVifCmd(0x20u, 0u, 0u));      // STMASK
            appendU32(packet, 0x89ABCDEFu);

            appendU32(packet, makeVifCmd(0x30u, 0u, 0u));      // STROW
            appendU32(packet, 0x11111111u);
            appendU32(packet, 0x22222222u);
            appendU32(packet, 0x33333333u);
            appendU32(packet, 0x44444444u);

            appendU32(packet, makeVifCmd(0x31u, 0u, 0u));      // STCOL
            appendU32(packet, 0xAAAA0001u);
            appendU32(packet, 0xAAAA0002u);
            appendU32(packet, 0xAAAA0003u);
            appendU32(packet, 0xAAAA0004u);

            mem.processVIF1Data(packet.data(), static_cast<uint32_t>(packet.size()));

            t.Equals(mem.vif1_regs.mark, 0x1234u, "MARK should set VIF1 MARK register");
            t.Equals(mem.vif1_regs.mask, 0x89ABCDEFu, "STMASK should set VIF1 MASK register");

            t.Equals(mem.vif1_regs.row[0], 0x11111111u, "STROW should set row[0]");
            t.Equals(mem.vif1_regs.row[1], 0x22222222u, "STROW should set row[1]");
            t.Equals(mem.vif1_regs.row[2], 0x33333333u, "STROW should set row[2]");
            t.Equals(mem.vif1_regs.row[3], 0x44444444u, "STROW should set row[3]");

            t.Equals(mem.vif1_regs.col[0], 0xAAAA0001u, "STCOL should set col[0]");
            t.Equals(mem.vif1_regs.col[1], 0xAAAA0002u, "STCOL should set col[1]");
            t.Equals(mem.vif1_regs.col[2], 0xAAAA0003u, "STCOL should set col[2]");
            t.Equals(mem.vif1_regs.col[3], 0xAAAA0004u, "STCOL should set col[3]");
        });

        tc.Run("VIF UNPACK V4-16 sign and zero extension follow immediate bit14", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");
            std::memset(mem.getVU1Data(), 0, PS2_VU1_DATA_SIZE);

            // UNPACK V4-16 (opcode 0x6D), num=1, addr=0.
            // Payload components: x=0xFF80, y=0x0001, z=0x7FFF, w=0x8001.
            const uint16_t comps[4] = {0xFF80u, 0x0001u, 0x7FFFu, 0x8001u};

            std::vector<uint8_t> signPacket;
            appendU32(signPacket, makeVifCmd(0x6Du, 1u, 0x0000u)); // sign-extend
            for (uint16_t c : comps)
            {
                const size_t pos = signPacket.size();
                signPacket.resize(pos + sizeof(uint16_t));
                std::memcpy(signPacket.data() + pos, &c, sizeof(uint16_t));
            }
            mem.processVIF1Data(signPacket.data(), static_cast<uint32_t>(signPacket.size()));

            const uint8_t *vu1 = mem.getVU1Data();
            uint32_t sx = 0, sy = 0, sz = 0, sw = 0;
            std::memcpy(&sx, vu1 + 0, 4);
            std::memcpy(&sy, vu1 + 4, 4);
            std::memcpy(&sz, vu1 + 8, 4);
            std::memcpy(&sw, vu1 + 12, 4);
            t.Equals(sx, 0xFFFFFF80u, "sign-extend x");
            t.Equals(sy, 0x00000001u, "sign-extend y");
            t.Equals(sz, 0x00007FFFu, "sign-extend z");
            t.Equals(sw, 0xFFFF8001u, "sign-extend w");

            // Same UNPACK with imm bit14 set => zero-extend.
            std::vector<uint8_t> zeroPacket;
            appendU32(zeroPacket, makeVifCmd(0x6Du, 1u, 0x4000u)); // zero-extend
            for (uint16_t c : comps)
            {
                const size_t pos = zeroPacket.size();
                zeroPacket.resize(pos + sizeof(uint16_t));
                std::memcpy(zeroPacket.data() + pos, &c, sizeof(uint16_t));
            }
            mem.processVIF1Data(zeroPacket.data(), static_cast<uint32_t>(zeroPacket.size()));

            std::memcpy(&sx, vu1 + 0, 4);
            std::memcpy(&sy, vu1 + 4, 4);
            std::memcpy(&sz, vu1 + 8, 4);
            std::memcpy(&sw, vu1 + 12, 4);
            t.Equals(sx, 0x0000FF80u, "zero-extend x");
            t.Equals(sy, 0x00000001u, "zero-extend y");
            t.Equals(sz, 0x00007FFFu, "zero-extend z");
            t.Equals(sw, 0x00008001u, "zero-extend w");
        });

        tc.Run("VIF UNPACK bit15 adds TOPS to destination address", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");
            std::memset(mem.getVU1Data(), 0, PS2_VU1_DATA_SIZE);

            mem.vif1_regs.tops = 4u;

            // UNPACK V4-32, num=1, addr=2, bit15 set => effective addr = 6.
            std::vector<uint8_t> packet;
            appendU32(packet, makeVifCmd(0x6Cu, 1u, static_cast<uint16_t>(0x8000u | 0x0002u)));
            appendU32(packet, 0x11111111u);
            appendU32(packet, 0x22222222u);
            appendU32(packet, 0x33333333u);
            appendU32(packet, 0x44444444u);

            mem.processVIF1Data(packet.data(), static_cast<uint32_t>(packet.size()));

            const uint8_t *vu1 = mem.getVU1Data();

            uint32_t untouched = 0xDEADBEEFu;
            std::memcpy(&untouched, vu1 + (2u * 16u), 4);
            t.Equals(untouched, 0u, "base addr without TOPS should remain untouched");

            uint32_t x = 0, y = 0, z = 0, w = 0;
            const uint32_t dest = 6u * 16u;
            std::memcpy(&x, vu1 + dest + 0u, 4);
            std::memcpy(&y, vu1 + dest + 4u, 4);
            std::memcpy(&z, vu1 + dest + 8u, 4);
            std::memcpy(&w, vu1 + dest + 12u, 4);
            t.Equals(x, 0x11111111u, "TOPS-adjusted x");
            t.Equals(y, 0x22222222u, "TOPS-adjusted y");
            t.Equals(z, 0x33333333u, "TOPS-adjusted z");
            t.Equals(w, 0x44444444u, "TOPS-adjusted w");
        });

        tc.Run("VIF STCYCL skip mode advances destination by CL when CL>=WL", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");
            std::memset(mem.getVU1Data(), 0, PS2_VU1_DATA_SIZE);

            std::vector<uint8_t> packet;
            appendU32(packet, makeVifCmd(0x01u, 0u, static_cast<uint16_t>((1u << 8) | 3u))); // STCYCL: WL=1, CL=3
            appendU32(packet, makeVifCmd(0x6Cu, 2u, 0u)); // UNPACK V4-32, NUM=2, ADDR=0

            appendU32(packet, 0x11111111u);
            appendU32(packet, 0x22222222u);
            appendU32(packet, 0x33333333u);
            appendU32(packet, 0x44444444u);

            appendU32(packet, 0xAAAAAAAAu);
            appendU32(packet, 0xBBBBBBBBu);
            appendU32(packet, 0xCCCCCCCCu);
            appendU32(packet, 0xDDDDDDDDu);

            mem.processVIF1Data(packet.data(), static_cast<uint32_t>(packet.size()));

            const uint8_t *vu = mem.getVU1Data();

            uint32_t v0x = 0, v1x = 0, v2x = 0, v3x = 0;
            std::memcpy(&v0x, vu + 0u * 16u + 0u, 4);
            std::memcpy(&v1x, vu + 1u * 16u + 0u, 4);
            std::memcpy(&v2x, vu + 2u * 16u + 0u, 4);
            std::memcpy(&v3x, vu + 3u * 16u + 0u, 4);

            t.Equals(v0x, 0x11111111u, "first vector should write at addr 0");
            t.Equals(v1x, 0u, "skip mode should leave addr 1 untouched when WL=1 CL=3");
            t.Equals(v2x, 0u, "skip mode should leave addr 2 untouched when WL=1 CL=3");
            t.Equals(v3x, 0xAAAAAAAAu, "second vector should write at addr CL (addr 3)");
        });

        tc.Run("VIF masked UNPACK uses data row col and protect selectors", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");
            std::memset(mem.getVU1Data(), 0, PS2_VU1_DATA_SIZE);

            // Pre-fill destination W lane for write-protect verification.
            uint32_t preservedW = 0xDEADBEEFu;
            std::memcpy(mem.getVU1Data() + 12u, &preservedW, 4u);

            std::vector<uint8_t> packet;
            appendU32(packet, makeVifCmd(0x20u, 0u, 0u)); // STMASK
            appendU32(packet, 0x000000E4u); // m0=0(data), m1=1(row), m2=2(col), m3=3(protect)

            appendU32(packet, makeVifCmd(0x30u, 0u, 0u)); // STROW
            appendU32(packet, 0xAAAAB001u);
            appendU32(packet, 0xAAAAB002u);
            appendU32(packet, 0xAAAAB003u);
            appendU32(packet, 0xAAAAB004u);

            appendU32(packet, makeVifCmd(0x31u, 0u, 0u)); // STCOL
            appendU32(packet, 0x11110001u);
            appendU32(packet, 0x11110002u);
            appendU32(packet, 0x11110003u);
            appendU32(packet, 0x11110004u);

            appendU32(packet, makeVifCmd(0x7Cu, 1u, 0u)); // UNPACK V4-32 with CMD bit4 (mask enable)
            appendU32(packet, 0x01020304u);
            appendU32(packet, 0x11121314u);
            appendU32(packet, 0x21222324u);
            appendU32(packet, 0x31323334u);

            mem.processVIF1Data(packet.data(), static_cast<uint32_t>(packet.size()));

            const uint8_t *vu = mem.getVU1Data();
            uint32_t x = 0, y = 0, z = 0, w = 0;
            std::memcpy(&x, vu + 0u, 4u);
            std::memcpy(&y, vu + 4u, 4u);
            std::memcpy(&z, vu + 8u, 4u);
            std::memcpy(&w, vu + 12u, 4u);

            t.Equals(x, 0x01020304u, "mask=0 should write decompressed data");
            t.Equals(y, 0xAAAAB002u, "mask=1 should write row register for Y field");
            t.Equals(z, 0x11110001u, "mask=2 should write C0 on first write cycle");
            t.Equals(w, preservedW, "mask=3 should write-protect destination field");
        });

        tc.Run("VIF STMOD offset and difference modes apply to UNPACK data", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");
            std::memset(mem.getVU1Data(), 0, PS2_VU1_DATA_SIZE);

            std::vector<uint8_t> packet;
            appendU32(packet, makeVifCmd(0x30u, 0u, 0u)); // STROW
            appendU32(packet, 10u);
            appendU32(packet, 20u);
            appendU32(packet, 30u);
            appendU32(packet, 40u);

            appendU32(packet, makeVifCmd(0x05u, 0u, 1u)); // STMOD offset mode
            appendU32(packet, makeVifCmd(0x6Cu, 1u, 0u)); // UNPACK V4-32 -> addr 0
            appendU32(packet, 1u);
            appendU32(packet, 2u);
            appendU32(packet, 3u);
            appendU32(packet, 4u);

            appendU32(packet, makeVifCmd(0x30u, 0u, 0u)); // reset STROW for difference mode
            appendU32(packet, 100u);
            appendU32(packet, 100u);
            appendU32(packet, 100u);
            appendU32(packet, 100u);

            appendU32(packet, makeVifCmd(0x05u, 0u, 2u)); // STMOD difference mode
            appendU32(packet, makeVifCmd(0x6Cu, 2u, 1u)); // UNPACK V4-32 -> addr 1 and 2
            appendU32(packet, 1u);
            appendU32(packet, 1u);
            appendU32(packet, 1u);
            appendU32(packet, 1u);
            appendU32(packet, 2u);
            appendU32(packet, 2u);
            appendU32(packet, 2u);
            appendU32(packet, 2u);

            mem.processVIF1Data(packet.data(), static_cast<uint32_t>(packet.size()));

            const uint8_t *vu = mem.getVU1Data();
            uint32_t x0 = 0, y0 = 0, z0 = 0, w0 = 0;
            std::memcpy(&x0, vu + 0u * 16u + 0u, 4u);
            std::memcpy(&y0, vu + 0u * 16u + 4u, 4u);
            std::memcpy(&z0, vu + 0u * 16u + 8u, 4u);
            std::memcpy(&w0, vu + 0u * 16u + 12u, 4u);
            t.Equals(x0, 11u, "offset mode X");
            t.Equals(y0, 22u, "offset mode Y");
            t.Equals(z0, 33u, "offset mode Z");
            t.Equals(w0, 44u, "offset mode W");

            uint32_t x1 = 0, x2 = 0;
            std::memcpy(&x1, vu + 1u * 16u + 0u, 4u);
            std::memcpy(&x2, vu + 2u * 16u + 0u, 4u);
            t.Equals(x1, 101u, "difference mode first write should add initial row");
            t.Equals(x2, 103u, "difference mode second write should accumulate updated row");
            t.Equals(mem.vif1_regs.row[0], 103u, "difference mode should update row register");
            t.Equals(mem.vif1_regs.row[1], 103u, "difference mode should update row register for Y");
            t.Equals(mem.vif1_regs.row[2], 103u, "difference mode should update row register for Z");
            t.Equals(mem.vif1_regs.row[3], 103u, "difference mode should update row register for W");
        });

        tc.Run("VIF fill write uses STMASK and STROW when WL>CL", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");
            std::memset(mem.getVU1Data(), 0, PS2_VU1_DATA_SIZE);

            std::vector<uint8_t> packet;
            appendU32(packet, makeVifCmd(0x01u, 0u, static_cast<uint16_t>((3u << 8) | 1u))); // STCYCL: WL=3, CL=1

            appendU32(packet, makeVifCmd(0x20u, 0u, 0u)); // STMASK
            appendU32(packet, 0x55555555u); // all fields all cycles use row register

            appendU32(packet, makeVifCmd(0x30u, 0u, 0u)); // STROW
            appendU32(packet, 0x11111111u);
            appendU32(packet, 0x22222222u);
            appendU32(packet, 0x33333333u);
            appendU32(packet, 0x44444444u);

            appendU32(packet, makeVifCmd(0x7Cu, 3u, 0u)); // masked UNPACK V4-32, NUM=3 writes
            // Only one input vector should be consumed for CL=1, WL=3.
            appendU32(packet, 0xAAAABBBB);
            appendU32(packet, 0xCCCCDDDD);
            appendU32(packet, 0xEEEEFFFF);
            appendU32(packet, 0x12345678);

            mem.processVIF1Data(packet.data(), static_cast<uint32_t>(packet.size()));

            const uint8_t *vu = mem.getVU1Data();
            for (uint32_t i = 0; i < 3u; ++i)
            {
                uint32_t x = 0, y = 0, z = 0, w = 0;
                std::memcpy(&x, vu + i * 16u + 0u, 4u);
                std::memcpy(&y, vu + i * 16u + 4u, 4u);
                std::memcpy(&z, vu + i * 16u + 8u, 4u);
                std::memcpy(&w, vu + i * 16u + 12u, 4u);
                t.Equals(x, 0x11111111u, "fill write X should use row[0]");
                t.Equals(y, 0x22222222u, "fill write Y should use row[1]");
                t.Equals(z, 0x33333333u, "fill write Z should use row[2]");
                t.Equals(w, 0x44444444u, "fill write W should use row[3]");
            }
        });

        tc.Run("VIF irq command sets STAT.INT and CODE until FBRST.STC clears it", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            const uint32_t irqMarkCmd = 0x80000000u | makeVifCmd(0x07u, 0x12u, 0x3456u);
            mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&irqMarkCmd), sizeof(irqMarkCmd));

            t.Equals(mem.vif1_regs.code, irqMarkCmd, "VIF CODE should capture the last processed command");
            t.IsTrue((mem.vif1_regs.stat & (1u << 11)) != 0u, "irq bit should raise VIF1 STAT.INT");
            t.Equals(mem.vif1_regs.mark, 0x3456u, "MARK command should still update MARK register");

            t.IsTrue(mem.writeIORegister(0x10003C10u, 0x8u), "FBRST STC write should succeed");
            t.IsTrue((mem.vif1_regs.stat & (1u << 11)) == 0u, "FBRST.STC should clear VIF1 STAT.INT");
        });

        tc.Run("VIF FBRST RST clears VIF1 command state", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            mem.vif1_regs.mark = 0x1234u;
            mem.vif1_regs.cycle = 0x0102u;
            mem.vif1_regs.mode = 2u;
            mem.vif1_regs.num = 7u;
            mem.vif1_regs.mask = 0x89ABCDEFu;
            mem.vif1_regs.code = 0xCAFEBABEu;
            mem.vif1_regs.stat = 0x3F00u;

            t.IsTrue(mem.writeIORegister(0x10003C10u, 0x1u), "FBRST RST write should succeed");

            t.Equals(mem.vif1_regs.mark, 0u, "RST should clear MARK");
            t.Equals(mem.vif1_regs.cycle, 0u, "RST should clear CYCLE");
            t.Equals(mem.vif1_regs.mode, 0u, "RST should clear MODE");
            t.Equals(mem.vif1_regs.num, 0u, "RST should clear NUM");
            t.Equals(mem.vif1_regs.mask, 0u, "RST should clear MASK");
            t.Equals(mem.vif1_regs.code, 0u, "RST should clear CODE");
            t.Equals(mem.vif1_regs.stat, 0u, "RST should clear STAT");
        });

        tc.Run("VIF double-buffer OFFSET BASE and MSCAL update TOPS and ITOPS", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            mem.vif1_regs.base = 0x120u;
            mem.vif1_regs.tops = 0x120u;
            mem.vif1_regs.stat = (1u << 7); // DBF=1 before OFFSET

            struct MscalCall
            {
                uint32_t startPC;
                uint32_t top;
                uint32_t itop;
            };
            std::vector<MscalCall> mscalCalls;
            mem.setVu1MscalCallback([&](uint32_t startPC, uint32_t top, uint32_t itop)
            {
                mscalCalls.push_back({startPC, top, itop});
            });

            const uint32_t offsetCmd = makeVifCmd(0x02u, 0u, 0x0022u);
            mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&offsetCmd), sizeof(offsetCmd));
            t.Equals(mem.vif1_regs.ofst, 0x22u, "OFFSET should update OFST");
            t.Equals(mem.vif1_regs.base, 0x120u, "OFFSET should copy old TOPS into BASE");
            t.IsTrue((mem.vif1_regs.stat & (1u << 7)) == 0u, "OFFSET should clear DBF");
            t.Equals(mem.vif1_regs.tops, 0x120u, "DBF=0 should keep TOPS at BASE");

            const uint32_t baseCmd = makeVifCmd(0x03u, 0u, 0x0030u);
            mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&baseCmd), sizeof(baseCmd));
            t.Equals(mem.vif1_regs.base, 0x30u, "BASE should update BASE register");
            t.Equals(mem.vif1_regs.tops, 0x120u, "BASE should not rewrite current TOPS");

            const uint32_t itopCmd = makeVifCmd(0x04u, 0u, 0x0044u);
            mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&itopCmd), sizeof(itopCmd));
            t.Equals(mem.vif1_regs.itops, 0x44u, "ITOP VIFcode should update pending ITOPS register");

            const uint32_t mscalCmd = makeVifCmd(0x14u, 0u, 0x0003u);
            mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&mscalCmd), sizeof(mscalCmd));
            t.Equals(mscalCalls.size(), static_cast<size_t>(1u), "MSCAL should invoke callback once");
            t.Equals(mscalCalls[0].startPC, 0x18u, "MSCAL callback startPC should be IMMEDIATE*8");
            t.Equals(mscalCalls[0].top, 0x120u, "MSCAL callback should receive current TOPS");
            t.Equals(mscalCalls[0].itop, 0x44u, "MSCAL callback should receive pending ITOPS");
            t.Equals(mem.vif1_regs.top, 0x120u, "MSCAL should latch TOP from TOPS");
            t.Equals(mem.vif1_regs.itop, 0x44u, "MSCAL should latch ITOP from ITOPS");
            t.IsTrue((mem.vif1_regs.stat & (1u << 7)) != 0u, "MSCAL should toggle DBF");
            t.Equals(mem.vif1_regs.tops, 0x52u, "DBF=1 should set TOPS to BASE+OFST");

            const uint32_t mscntCmd = makeVifCmd(0x17u, 0u, 0u);
            mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&mscntCmd), sizeof(mscntCmd));
            t.IsTrue((mem.vif1_regs.stat & (1u << 7)) == 0u, "MSCNT should toggle DBF again");
            t.Equals(mem.vif1_regs.tops, 0x30u, "DBF=0 should restore TOPS to BASE");
        });

        tc.Run("VIF MSKPATH3 uses immediate bit15", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            const uint32_t setMask = makeVifCmd(0x06u, 0u, 0x8000u);
            mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&setMask), sizeof(setMask));
            t.IsTrue(mem.isPath3Masked(), "MSKPATH3 with imm bit15 set should enable PATH3 mask");

            const uint32_t clearMask = makeVifCmd(0x06u, 0u, 0x0000u);
            mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&clearMask), sizeof(clearMask));
            t.IsFalse(mem.isPath3Masked(), "MSKPATH3 with imm bit15 clear should disable PATH3 mask");
        });

        tc.Run("PATH3 mask queues packets until unmask", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            std::vector<std::vector<uint8_t>> captured;
            mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            std::vector<uint8_t> packetA(16u);
            std::vector<uint8_t> packetB(16u);
            for (uint32_t i = 0; i < 16u; ++i)
            {
                packetA[i] = static_cast<uint8_t>(0x10u + i);
                packetB[i] = static_cast<uint8_t>(0x40u + i);
            }

            const uint32_t setMask = makeVifCmd(0x06u, 0u, 0x8000u);
            mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&setMask), sizeof(setMask));
            t.IsTrue(mem.isPath3Masked(), "PATH3 mask should be enabled");

            mem.submitGifPacket(GifPathId::Path3, packetA.data(), static_cast<uint32_t>(packetA.size()));
            mem.submitGifPacket(GifPathId::Path3, packetB.data(), static_cast<uint32_t>(packetB.size()));
            t.Equals(captured.size(), static_cast<size_t>(0u), "masked PATH3 packets should be queued, not dropped/emitted");

            const uint32_t clearMask = makeVifCmd(0x06u, 0u, 0x0000u);
            mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&clearMask), sizeof(clearMask));

            t.Equals(captured.size(), static_cast<size_t>(2u), "unmask should flush queued PATH3 packets");
            bool firstOk = true;
            bool secondOk = true;
            for (uint32_t i = 0; i < 16u; ++i)
            {
                if (captured[0][i] != static_cast<uint8_t>(0x10u + i))
                    firstOk = false;
                if (captured[1][i] != static_cast<uint8_t>(0x40u + i))
                    secondOk = false;
            }
            t.IsTrue(firstOk, "first queued PATH3 packet should flush in-order");
            t.IsTrue(secondOk, "second queued PATH3 packet should flush in-order");
        });

        tc.Run("GIF arbiter prioritizes PATH1 then PATH2 then PATH3", [](TestCase &t)
        {
            std::vector<uint8_t> order;
            GifArbiter arbiter([&](const uint8_t *data, uint32_t sizeBytes)
            {
                if (data && sizeBytes >= 16u)
                    order.push_back(data[8]);
            });

            auto makeEmptyPacket = [](uint8_t marker)
            {
                std::vector<uint8_t> packet;
                appendU64(packet, makeGifTag(0u, GIF_FMT_PACKED, 1u, true));
                appendU64(packet, marker);
                return packet;
            };
            const std::vector<uint8_t> p1 = makeEmptyPacket(0x11u);
            const std::vector<uint8_t> p2 = makeEmptyPacket(0x22u);
            const std::vector<uint8_t> p3 = makeEmptyPacket(0x33u);

            arbiter.submit(GifPathId::Path3, p3.data(), static_cast<uint32_t>(p3.size()));
            arbiter.submit(GifPathId::Path2, p2.data(), static_cast<uint32_t>(p2.size()));
            arbiter.submit(GifPathId::Path1, p1.data(), static_cast<uint32_t>(p1.size()));
            arbiter.drain();

            t.Equals(order.size(), static_cast<size_t>(3u), "all queued packets should be drained");
            t.Equals(order[0], static_cast<uint8_t>(0x11u), "PATH1 should be drained first");
            t.Equals(order[1], static_cast<uint8_t>(0x22u), "PATH2 should be drained second");
            t.Equals(order[2], static_cast<uint8_t>(0x33u), "PATH3 should be drained third");
        });

        tc.Run("GIF arbiter brackets a non-empty drain once", [](TestCase &t)
        {
            std::vector<uint8_t> events;
            GifArbiter arbiter([&](const uint8_t *data, uint32_t sizeBytes)
            {
                if (data && sizeBytes >= 16u)
                    events.push_back(data[8]);
            });
            arbiter.setDrainCallbacks([&]
            {
                events.push_back(0xB0u);
            }, [&]
            {
                events.push_back(0xE0u);
            });

            auto makeEmptyPacket = [](uint8_t marker)
            {
                std::vector<uint8_t> packet;
                appendU64(packet, makeGifTag(0u, GIF_FMT_PACKED, 1u, true));
                appendU64(packet, marker);
                return packet;
            };
            const std::vector<uint8_t> p1 = makeEmptyPacket(0x11u);
            const std::vector<uint8_t> p3 = makeEmptyPacket(0x33u);
            arbiter.submit(GifPathId::Path3, p3.data(), static_cast<uint32_t>(p3.size()));
            arbiter.submit(GifPathId::Path1, p1.data(), static_cast<uint32_t>(p1.size()));
            arbiter.drain();
            arbiter.drain();

            t.Equals(events.size(), static_cast<size_t>(4u),
                     "only a non-empty drain should invoke its boundary callbacks");
            if (events.size() == 4u)
            {
                t.Equals(events[0], static_cast<uint8_t>(0xB0u),
                         "drain begin callback should run before packets");
                t.Equals(events[1], static_cast<uint8_t>(0x11u),
                         "PATH1 packet should remain first inside the batch");
                t.Equals(events[2], static_cast<uint8_t>(0x33u),
                         "PATH3 packet should remain last inside the batch");
                t.Equals(events[3], static_cast<uint8_t>(0xE0u),
                         "drain end callback should run after packets");
            }
        });

        tc.Run("GIF arbiter reassembles IMAGE payload split across submissions", [](TestCase &t)
        {
            std::vector<std::vector<uint8_t>> captured;
            GifArbiter arbiter([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(2u, GIF_FMT_IMAGE, 0u, true));
            appendU64(packet, 0u);
            for (uint32_t i = 0u; i < 32u; ++i)
                packet.push_back(static_cast<uint8_t>(0x40u + i));

            arbiter.submit(GifPathId::Path2, packet.data(), 32u);
            arbiter.drain();
            t.Equals(captured.size(), static_cast<size_t>(0u), "partial IMAGE must stay buffered");
            t.IsFalse(arbiter.empty(), "partial IMAGE should make the path stream non-empty");

            arbiter.submit(GifPathId::Path2, packet.data() + 32u, 16u);
            arbiter.drain();
            t.Equals(captured.size(), static_cast<size_t>(1u), "completed IMAGE should drain once");
            t.Equals(captured[0].size(), packet.size(), "reassembled IMAGE size should match the stream");
            t.IsTrue(std::memcmp(captured[0].data(), packet.data(), packet.size()) == 0,
                     "reassembled IMAGE bytes should preserve tag and payload order");
            t.IsTrue(arbiter.empty(), "completed IMAGE should leave no buffered path data");
        });

        tc.Run("GIF arbiter reassembles a tag split across submissions", [](TestCase &t)
        {
            std::vector<std::vector<uint8_t>> captured;
            GifArbiter arbiter([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(0u, GIF_FMT_PACKED, 1u, true));
            appendU64(packet, 0x8877665544332211ull);

            arbiter.submit(GifPathId::Path1, packet.data(), 8u);
            arbiter.drain();
            t.Equals(captured.size(), static_cast<size_t>(0u),
                     "a partial GIFtag should remain buffered");
            t.IsFalse(arbiter.empty(),
                      "a partial GIFtag should keep its path non-empty");

            arbiter.submit(GifPathId::Path1, packet.data() + 8u, 8u);
            arbiter.drain();
            t.Equals(captured.size(), static_cast<size_t>(1u),
                     "the completed GIFtag should drain once");
            t.IsTrue(
                captured.size() == 1u &&
                    captured[0] == packet,
                "the split GIFtag should retain exact bytes");
            t.IsTrue(arbiter.empty(),
                     "the completed split tag should release its stream");
        });

        tc.Run("GIF arbiter drains multiple EOP packets from one backing stream", [](TestCase &t)
        {
            std::vector<std::vector<uint8_t>> captured;
            GifArbiter arbiter([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            auto makePacket = [](uint64_t marker)
            {
                std::vector<uint8_t> packet;
                appendU64(packet, makeGifTag(0u, GIF_FMT_PACKED, 1u, true));
                appendU64(packet, marker);
                return packet;
            };
            const std::vector<uint8_t> first =
                makePacket(0x1111111111111111ull);
            const std::vector<uint8_t> second =
                makePacket(0x2222222222222222ull);
            std::vector<uint8_t> stream = first;
            stream.insert(stream.end(), second.begin(), second.end());

            arbiter.submit(
                GifPathId::Path3,
                stream.data(),
                static_cast<uint32_t>(stream.size()));
            arbiter.drain();

            t.Equals(captured.size(), static_cast<size_t>(2u),
                     "both EOP-delimited packets should drain");
            t.IsTrue(
                captured.size() == 2u &&
                    captured[0] == first &&
                    captured[1] == second,
                "packet spans should retain byte-exact boundaries");
            t.IsTrue(arbiter.empty(),
                     "draining complete spans should reuse an empty backing stream");
        });

        tc.Run("GIF arbiter compacts only the incomplete suffix after drain", [](TestCase &t)
        {
            std::vector<std::vector<uint8_t>> captured;
            GifArbiter arbiter([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            auto makePacket = [](uint64_t marker)
            {
                std::vector<uint8_t> packet;
                appendU64(packet, makeGifTag(0u, GIF_FMT_PACKED, 1u, true));
                appendU64(packet, marker);
                return packet;
            };
            const std::vector<uint8_t> first =
                makePacket(0x3333333333333333ull);
            const std::vector<uint8_t> second =
                makePacket(0x4444444444444444ull);
            std::vector<uint8_t> prefix = first;
            prefix.insert(prefix.end(), second.begin(), second.begin() + 8);

            arbiter.submit(
                GifPathId::Path2,
                prefix.data(),
                static_cast<uint32_t>(prefix.size()));
            arbiter.drain();
            t.Equals(captured.size(), static_cast<size_t>(1u),
                     "the complete prefix packet should drain immediately");
            t.IsTrue(
                captured.size() == 1u &&
                    captured[0] == first,
                "draining should not include the following partial tag");
            t.IsFalse(arbiter.empty(),
                      "the compacted partial suffix should remain buffered");

            arbiter.submit(
                GifPathId::Path2,
                second.data() + 8u,
                8u);
            arbiter.drain();
            t.Equals(captured.size(), static_cast<size_t>(2u),
                     "the completed suffix should drain once");
            t.IsTrue(
                captured.size() == 2u &&
                    captured[1] == second,
                "suffix compaction should preserve the next packet exactly");
            t.IsTrue(arbiter.empty(),
                     "the completed suffix should leave no buffered bytes");
        });

        tc.Run("GIF arbiter keeps non-EOP tags in one packet span", [](TestCase &t)
        {
            std::vector<std::vector<uint8_t>> captured;
            GifArbiter arbiter([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(0u, GIF_FMT_PACKED, 1u, false));
            appendU64(packet, 0x5555555555555555ull);
            appendU64(packet, makeGifTag(0u, GIF_FMT_PACKED, 1u, true));
            appendU64(packet, 0x6666666666666666ull);

            arbiter.submit(GifPathId::Path1, packet.data(), 16u);
            arbiter.drain();
            t.Equals(captured.size(), static_cast<size_t>(0u),
                     "a non-EOP tag should not complete its packet");

            arbiter.submit(
                GifPathId::Path1,
                packet.data() + 16u,
                16u);
            arbiter.drain();
            t.Equals(captured.size(), static_cast<size_t>(1u),
                     "the following EOP tag should complete one packet");
            t.IsTrue(
                captured.size() == 1u &&
                    captured[0] == packet,
                "the completed packet should include its non-EOP prefix");
        });

        tc.Run("GIF arbiter preserves queued packets when an oversized suffix is rejected", [](TestCase &t)
        {
            std::vector<std::vector<uint8_t>> captured;
            GifArbiter arbiter([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            std::vector<uint8_t> complete;
            appendU64(complete, makeGifTag(0u, GIF_FMT_PACKED, 1u, true));
            appendU64(complete, 0x7777777777777777ull);

            std::vector<uint8_t> next;
            appendU64(next, makeGifTag(0u, GIF_FMT_PACKED, 1u, true));
            appendU64(next, 0x8888888888888888ull);

            std::vector<uint8_t> prefix = complete;
            prefix.insert(prefix.end(), next.begin(), next.begin() + 8u);
            arbiter.submit(
                GifPathId::Path1,
                prefix.data(),
                static_cast<uint32_t>(prefix.size()));

            const uint8_t unused = 0u;
            arbiter.submit(
                GifPathId::Path1,
                &unused,
                (64u * 1024u * 1024u) + 1u);
            arbiter.drain();

            t.Equals(captured.size(), static_cast<size_t>(1u),
                     "the valid queued packet should survive suffix rejection");
            t.IsTrue(
                captured.size() == 1u &&
                    captured[0] == complete,
                "suffix rejection should preserve the queued packet bytes");
            t.IsTrue(
                arbiter.empty(),
                "suffix rejection should discard only the incomplete packet");
        });

        tc.Run("GIF arbiter frames IMAGE2 like IMAGE", [](TestCase &t)
        {
            std::vector<std::vector<uint8_t>> captured;
            GifArbiter arbiter([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(1u, GIF_FMT_IMAGE2, 0u, true));
            appendU64(packet, 0u);
            packet.insert(packet.end(), 16u, 0x5Au);

            arbiter.submit(GifPathId::Path3, packet.data(), static_cast<uint32_t>(packet.size()));
            arbiter.drain();

            t.Equals(captured.size(), static_cast<size_t>(1u), "IMAGE2 should drain as one complete packet");
            t.Equals(captured[0].size(), packet.size(), "IMAGE2 payload should be included in packet size");
        });

        tc.Run("VIF DIRECTHL stalls behind queued PATH3 IMAGE packets", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            std::vector<uint8_t> firstBytes;
            GifArbiter arbiter([&](const uint8_t *data, uint32_t sizeBytes)
            {
                if (data && sizeBytes >= 16u)
                    firstBytes.push_back(data[8]);
            });
            mem.setGifArbiter(&arbiter);

            std::vector<uint8_t> path3Image;
            appendU64(path3Image, makeGifTag(1u, GIF_FMT_IMAGE, 0u, true));
            appendU64(path3Image, 0xA3ull);
            path3Image.insert(path3Image.end(), 16u, 0xAAu);
            mem.submitGifPacket(GifPathId::Path3, path3Image.data(), static_cast<uint32_t>(path3Image.size()), false);

            std::vector<uint8_t> vifPacket;
            appendU32(vifPacket, makeVifCmd(0x51u, 0u, 1u)); // DIRECTHL 1 QW
            appendU64(vifPacket, makeGifTag(0u, GIF_FMT_PACKED, 1u, true));
            appendU64(vifPacket, 0xD2ull);
            mem.processVIF1Data(vifPacket.data(), static_cast<uint32_t>(vifPacket.size()));

            t.Equals(firstBytes.size(), static_cast<size_t>(2u), "PATH3 and DIRECTHL packets should both drain");
            t.Equals(firstBytes[0], static_cast<uint8_t>(0xA3u), "DIRECTHL should not preempt queued PATH3 IMAGE packet");
            t.Equals(firstBytes[1], static_cast<uint8_t>(0xD2u), "DIRECTHL packet should drain after PATH3 IMAGE packet");
        });

        tc.Run("GIF DMA mode0 copies RDRAM packet and clears channel", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kGifCh = 0x1000A000u;
            constexpr uint32_t kSrc = 0x00022000u;
            constexpr uint32_t kQwc = 2u; // 32 bytes

            uint8_t *rdram = mem.getRDRAM();
            for (uint32_t i = 0; i < kQwc * 16u; ++i)
            {
                rdram[kSrc + i] = static_cast<uint8_t>((0x40u + i) & 0xFFu);
            }

            std::vector<std::vector<uint8_t>> captured;
            mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            t.IsTrue(mem.writeIORegister(kGifCh + 0x10u, kSrc), "write MADR should succeed");
            t.IsTrue(mem.writeIORegister(kGifCh + 0x20u, kQwc), "write QWC should succeed");
            t.IsTrue(mem.writeIORegister(kGifCh + 0x00u, 0x100u), "write CHCR STR should succeed");

            t.Equals(mem.dmaStartCount(), 1ull, "starting GIF DMA should increment dmaStartCount");

            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            t.Equals(captured.size(), static_cast<size_t>(1u), "GIF DMA should emit one packet");
            t.Equals(captured[0].size(), static_cast<size_t>(kQwc * 16u), "GIF packet size should match QWC");

            bool contentOk = true;
            for (uint32_t i = 0; i < kQwc * 16u; ++i)
            {
                if (captured[0][i] != static_cast<uint8_t>((0x40u + i) & 0xFFu))
                {
                    contentOk = false;
                    break;
                }
            }
            t.IsTrue(contentOk, "GIF DMA packet bytes should match source RDRAM");
            t.IsTrue(mem.hasSeenGifCopy(), "GIF DMA should mark seen GIF copy");
            t.Equals(mem.gifCopyCount(), 1ull, "GIF DMA should increment gifCopyCount");
            t.IsTrue((mem.readIORegister(kGifCh + 0x00u) & 0x100u) == 0u, "GIF CHCR STR bit should be cleared after drain");
            t.Equals(mem.readIORegister(kGifCh + 0x20u), 0u, "GIF QWC should be cleared after drain");
        });

        tc.Run("GIF DMA scheduled normal mode retains PCSX2 slice boundaries", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(),
                     "PS2Memory initialize should succeed");

            constexpr uint32_t kGifCh = 0x1000A000u;
            constexpr uint32_t kDstat = 0x1000E010u;
            constexpr uint32_t kSource = 0x00022400u;
            constexpr uint32_t kQwc = 12u;
            std::memset(mem.getRDRAM() + kSource,
                        0x31, kQwc * 16u);

            std::vector<uint32_t> scheduledDelays;
            mem.setGifDmaScheduleCallback(
                [&](uint32_t delayEeCycles)
                {
                    scheduledDelays.push_back(
                        delayEeCycles);
                    return true;
                });
            std::vector<std::vector<uint8_t>> captured;
            mem.setGifPacketCallback(
                [&](const uint8_t *data,
                    uint32_t sizeBytes)
                {
                    captured.emplace_back(
                        data, data + sizeBytes);
                });

            t.IsTrue(mem.writeIORegister(
                         kGifCh + 0x10u, kSource),
                     "GIF MADR write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kGifCh + 0x20u, kQwc),
                     "GIF QWC write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kGifCh, 0x100u),
                     "GIF normal transfer should start");

            GifDmaSnapshot state =
                mem.gifDmaSnapshot();
            t.Equals(scheduledDelays.size(),
                     static_cast<size_t>(1u),
                     "submission should publish one deadline");
            if (!scheduledDelays.empty())
            {
                t.Equals(scheduledDelays[0], 8u,
                         "initial four QW should cost eight cycles");
            }
            t.IsTrue(state.active &&
                         state.eventManaged,
                     "normal transfer should retain event ownership");
            t.IsTrue(state.phase ==
                         GifDmaPhase::TransferPayload,
                     "eight QW should remain after the initial slice");
            t.Equals(state.qwc, 8u,
                     "initial slice should retain eight QW");
            t.Equals(state.madr, kSource + 4u * 16u,
                     "initial slice should advance MADR by four QW");

            const GifDmaAdvanceResult payload =
                mem.advanceGifDma();
            t.IsTrue(payload.progressed &&
                         payload.active &&
                         !payload.completed,
                     "first service should consume the remaining payload");
            t.Equals(payload.transferredQwc, 8u,
                     "first service should accept eight QW");
            t.Equals(payload.delayEeCycles, 16u,
                     "remaining eight QW should cost sixteen cycles");
            state = mem.gifDmaSnapshot();
            t.IsTrue(state.phase ==
                         GifDmaPhase::Finalize,
                     "payload service should retain a completion boundary");
            t.IsTrue(
                (mem.readIORegister(kGifCh) & 0x100u) != 0u,
                "STR must remain visible before final service");
            t.IsTrue(
                (mem.readIORegister(kDstat) & (1u << 2u)) == 0u,
                "D_STAT must remain clear before final service");

            const GifDmaAdvanceResult completion =
                mem.advanceGifDma();
            t.IsTrue(completion.completed &&
                         !completion.active,
                     "second service should request completion");
            publishDmacCompletions(mem);
            t.IsTrue(
                (mem.readIORegister(kGifCh) & 0x100u) == 0u,
                "typed publication should clear STR");
            t.IsTrue(
                (mem.readIORegister(kDstat) & (1u << 2u)) != 0u,
                "typed publication should latch GIF D_STAT");
            t.Equals(captured.size(),
                     static_cast<size_t>(1u),
                     "compatibility callback should receive one DMA stream");
            if (!captured.empty())
            {
                t.Equals(captured[0].size(),
                         static_cast<size_t>(kQwc * 16u),
                         "callback should receive all twelve QW");
            }
        });

        tc.Run("GIF DMA scheduled END chain combines tag and initial slice cost", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(),
                     "PS2Memory initialize should succeed");

            constexpr uint32_t kGifCh = 0x1000A000u;
            constexpr uint32_t kTag = 0x00022800u;
            constexpr uint32_t kQwc = 12u;
            const uint64_t endTag =
                makeDmaTag(kQwc, 7u, 0u, false);
            std::memcpy(mem.getRDRAM() + kTag,
                        &endTag, sizeof(endTag));
            std::memset(mem.getRDRAM() + kTag + 16u,
                        0x42, kQwc * 16u);

            std::vector<uint32_t> scheduledDelays;
            mem.setGifDmaScheduleCallback(
                [&](uint32_t delayEeCycles)
                {
                    scheduledDelays.push_back(
                        delayEeCycles);
                    return true;
                });
            std::vector<std::vector<uint8_t>> captured;
            mem.setGifPacketCallback(
                [&](const uint8_t *data,
                    uint32_t sizeBytes)
                {
                    captured.emplace_back(
                        data, data + sizeBytes);
                });

            t.IsTrue(mem.writeIORegister(
                         kGifCh + 0x30u, kTag),
                     "GIF TADR write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kGifCh, 0x104u),
                     "GIF END chain should start");

            GifDmaSnapshot state =
                mem.gifDmaSnapshot();
            t.Equals(scheduledDelays.size(),
                     static_cast<size_t>(1u),
                     "chain submission should publish one deadline");
            if (!scheduledDelays.empty())
            {
                t.Equals(scheduledDelays[0], 10u,
                         "tag plus four QW should cost ten cycles");
            }
            t.Equals(state.tagsProcessed, 1u,
                     "initial progress should consume the END tag");
            t.Equals(state.qwc, 8u,
                     "initial progress should retain eight QW");
            t.Equals(state.madr,
                     kTag + 16u + 4u * 16u,
                     "initial progress should expose payload MADR");
            t.Equals(state.tadr, kTag,
                     "END should retain its tag address in TADR");

            const GifDmaAdvanceResult payload =
                mem.advanceGifDma();
            t.Equals(payload.delayEeCycles, 16u,
                     "remaining chain payload should cost sixteen cycles");
            t.IsTrue(
                mem.advanceGifDma().completed,
                "the following service should complete the chain");
            publishDmacCompletions(mem);

            t.Equals(mem.readIORegister(kGifCh),
                     0x70000004u,
                     "completion should retain END tag and chain-mode bits while clearing STR");
            t.Equals(mem.readIORegister(kGifCh + 0x10u),
                     kTag + 16u + kQwc * 16u,
                     "completion should retain final MADR");
            t.Equals(mem.readIORegister(kGifCh + 0x20u),
                     0u,
                     "completion should retain QWC zero");
            t.Equals(mem.readIORegister(kGifCh + 0x30u),
                     kTag,
                     "completion should retain END TADR");
            t.Equals(captured.size(),
                     static_cast<size_t>(1u),
                     "chain should emit one contiguous stream");
        });

        tc.Run("GIF MODE unmask owns the full FIFO wakeup", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(),
                     "PS2Memory initialize should succeed");

            constexpr uint32_t kGifMode = 0x10003010u;
            constexpr uint32_t kGifStat = 0x10003020u;
            constexpr uint32_t kGifCh = 0x1000A000u;
            constexpr uint32_t kSource = 0x00022C00u;
            constexpr uint32_t kQwc = 16u;
            std::memset(mem.getRDRAM() + kSource,
                        0x53, kQwc * 16u);

            std::vector<uint32_t> scheduledDelays;
            mem.setGifDmaScheduleCallback(
                [&](uint32_t delayEeCycles)
                {
                    scheduledDelays.push_back(
                        delayEeCycles);
                    return true;
                });
            std::vector<std::vector<uint8_t>> captured;
            mem.setGifPacketCallback(
                [&](const uint8_t *data,
                    uint32_t sizeBytes)
                {
                    captured.emplace_back(
                        data, data + sizeBytes);
                });

            t.IsTrue(mem.writeIORegister(kGifMode, 1u),
                     "M3R should mask PATH3");
            t.IsTrue(mem.writeIORegister(
                         kGifCh + 0x10u, kSource),
                     "GIF MADR write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kGifCh + 0x20u, kQwc),
                     "GIF QWC write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kGifCh, 0x100u),
                     "masked GIF transfer should start");

            GifDmaSnapshot state =
                mem.gifDmaSnapshot();
            t.IsTrue(scheduledDelays.empty(),
                     "a full masked FIFO must have no deadline");
            t.IsTrue(state.active &&
                         state.stall ==
                             GifDmaStallReason::
                                 Path3Masked,
                     "masked transfer should retain its stall");
            t.Equals(state.fifoQwc, 16u,
                     "masked transfer should fill all sixteen FIFO QW");
            t.Equals(state.qwc, 0u,
                     "FIFO fill should consume channel QWC");
            t.Equals(mem.readIORegister(kGifStat),
                     0x10000001u,
                     "GIF_STAT should expose FQC=16 and M3R");
            t.IsTrue(
                (mem.readIORegister(kGifCh) & 0x100u) != 0u,
                "masked FIFO should retain STR");

            t.IsTrue(mem.writeIORegister(kGifMode, 0u),
                     "clearing M3R should succeed");
            t.Equals(scheduledDelays.size(),
                     static_cast<size_t>(1u),
                     "unmask should publish one wakeup");
            if (!scheduledDelays.empty())
            {
                t.Equals(scheduledDelays[0], 8u,
                         "M3R removal should own the +8 wake");
            }

            const GifDmaAdvanceResult drain =
                mem.advanceGifDma();
            t.IsTrue(drain.progressed &&
                         drain.active,
                     "wake service should drain the FIFO");
            t.Equals(drain.transferredQwc, 16u,
                     "wake service should drain sixteen QW");
            t.Equals(drain.delayEeCycles, 32u,
                     "sixteen FIFO QW should cost thirty-two cycles");
            t.Equals(mem.readIORegister(kGifStat), 0u,
                     "FIFO drain should clear FQC and M3R");
            t.IsTrue(
                mem.advanceGifDma().completed,
                "the following service should request completion");
            publishDmacCompletions(mem);
            t.Equals(mem.readIORegister(kGifCh), 0u,
                     "completion should clear channel STR");
            t.Equals(captured.size(),
                     static_cast<size_t>(1u),
                     "FIFO contents should emit as one stream");
        });

        tc.Run("SPR_TO scheduled progress limits payload slices to 0x400 QWC", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(
                mem.initialize(),
                "PS2Memory initialize should succeed");

            constexpr uint32_t kChannel = 0x1000D400u;
            constexpr uint32_t kSource = 0x0001C000u;
            constexpr uint32_t kQwc = 0x401u;
            uint32_t scheduleCount = 0u;
            uint32_t scheduledDelay = 0u;
            mem.setScratchpadDmaScheduleCallback(
                [&](DmacChannel channel,
                    uint32_t delayEeCycles)
                {
                    t.IsTrue(
                        channel ==
                            DmacChannel::ToScratchpad,
                        "SPR_TO should retain typed scheduling ownership");
                    ++scheduleCount;
                    scheduledDelay = delayEeCycles;
                    return true;
                });

            for (uint32_t qword = 0u;
                 qword < kQwc; ++qword)
            {
                std::memset(
                    mem.getRDRAM() + kSource +
                        qword * 16u,
                    static_cast<int>(
                        (qword + 0x5Au) & 0xFFu),
                    16u);
            }

            t.IsTrue(
                mem.writeIORegister(
                    kChannel + 0x10u, kSource),
                "scheduled SPR_TO MADR write should succeed");
            t.IsTrue(
                mem.writeIORegister(
                    kChannel + 0x20u, kQwc),
                "scheduled SPR_TO QWC write should succeed");
            t.IsTrue(
                mem.writeIORegister(
                    kChannel, 0x100u),
                "scheduled SPR_TO start should succeed");

            ScratchpadDmaSnapshot dma =
                mem.scratchpadDmaSnapshot(
                    DmacChannel::ToScratchpad);
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.phase ==
                        ScratchpadDmaPhase::TransferPayload,
                "the first bounded slice should retain one QWC");
            t.Equals(
                dma.qwc, 1u,
                "the initial callback should consume exactly 0x400 QWC");
            t.Equals(
                dma.madr, kSource + 0x4000u,
                "the initial callback should expose bounded MADR progress");
            t.Equals(
                dma.sadr, 0u,
                "a full scratchpad slice should wrap SADR");
            t.Equals(
                scheduleCount, 1u,
                "submission should request one device event");
            t.Equals(
                scheduledDelay, 0x800u,
                "0x400 QWC should cost 0x800 EE cycles");

            const ScratchpadDmaAdvanceResult tail =
                mem.advanceScratchpadDma(
                    DmacChannel::ToScratchpad);
            t.IsTrue(
                tail.active && tail.progressed &&
                    tail.phase ==
                        ScratchpadDmaPhase::Finalize,
                "the next event should consume the remaining QWC");
            t.Equals(
                tail.delayEeCycles, 2u,
                "the final QWC should retain its two-cycle bus cost");
            t.Equals(
                mem.getScratchpad()[0],
                static_cast<uint8_t>(0x5Au),
                "the wrapping tail should copy QWC 0x400 to scratch offset zero");

            const ScratchpadDmaAdvanceResult final =
                mem.advanceScratchpadDma(
                    DmacChannel::ToScratchpad);
            t.IsTrue(
                final.completed && !final.active,
                "a later transition should request completion");
            publishDmacCompletions(mem);
            t.IsTrue(
                (mem.readIORegister(kChannel) &
                 0x100u) == 0u,
                "completion publication should clear SPR_TO STR");
        });

        tc.Run("SPR_TO normal DMA copies RDRAM into wrapping scratchpad", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kChannel = 0x1000D400u;
            constexpr uint32_t kSource = 0x00022000u;
            constexpr uint32_t kScratchAddress = 0x3FF0u;
            constexpr uint32_t kQwc = 2u;
            for (uint32_t byte = 0u; byte < kQwc * 16u; ++byte)
            {
                mem.getRDRAM()[kSource + byte] = static_cast<uint8_t>(0x40u + byte);
            }

            t.IsTrue(mem.writeIORegister(kChannel + 0x10u, kSource),
                     "SPR_TO MADR write should succeed");
            t.IsTrue(mem.writeIORegister(kChannel + 0x20u, kQwc),
                     "SPR_TO QWC write should succeed");
            t.IsTrue(mem.writeIORegister(kChannel + 0x80u, kScratchAddress),
                     "SPR_TO SADR write should succeed");
            t.IsTrue(mem.writeIORegister(kChannel + 0x00u, 0x100u),
                     "SPR_TO normal transfer should start");
            publishDmacCompletions(mem);

            bool copied = true;
            for (uint32_t byte = 0u; byte < 16u; ++byte)
            {
                copied &= mem.getScratchpad()[0x3FF0u + byte] ==
                          static_cast<uint8_t>(0x40u + byte);
                copied &= mem.getScratchpad()[byte] ==
                          static_cast<uint8_t>(0x50u + byte);
            }
            t.IsTrue(copied, "SPR_TO should copy both qwords across the scratchpad wrap");
            t.Equals(mem.readIORegister(kChannel + 0x10u), kSource + 32u,
                     "SPR_TO should advance MADR");
            t.Equals(mem.readIORegister(kChannel + 0x80u), 0x10u,
                     "SPR_TO should wrap and advance SADR");
            t.Equals(mem.readIORegister(kChannel + 0x20u), 0u,
                     "SPR_TO should consume QWC");
            t.IsTrue((mem.readIORegister(0x1000E010u) & (1u << 9u)) != 0u,
                     "SPR_TO should raise its D_STAT completion bit");

            const std::vector<DmacPendingInterrupt> interrupts =
                mem.consumePendingDmacInterrupts();
            t.Equals(interrupts.size(), static_cast<size_t>(1u),
                     "SPR_TO should queue one completion");
            t.Equals(interrupts[0].cause, 9u,
                     "SPR_TO should queue DMAC cause nine");
            t.IsTrue(
                interrupts[0].source ==
                    DmacChannel::ToScratchpad,
                "SPR_TO completion should retain its typed source");
            t.Equals(
                interrupts[0].eventSequence, 1ull,
                "SPR_TO completion should retain its event sequence");
        });

        tc.Run("SPR_TO source chain follows REF and REFE tags", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kChannel = 0x1000D400u;
            constexpr uint32_t kTagAddress = 0x00024000u;
            constexpr uint32_t kFirstSource = 0x00025000u;
            constexpr uint32_t kSecondSource = 0x00025100u;
            constexpr uint32_t kScratchAddress = 0x3FF0u;

            writeDmaTag(mem.getRDRAM(), kTagAddress + 0x00u,
                        makeDmaTag(2u, 3u, kFirstSource)); // REF
            writeDmaTag(mem.getRDRAM(), kTagAddress + 0x10u,
                        makeDmaTag(1u, 0u, kSecondSource)); // REFE

            for (uint32_t byte = 0u; byte < 32u; ++byte)
            {
                mem.getRDRAM()[kFirstSource + byte] =
                    static_cast<uint8_t>(0x40u + byte);
            }
            for (uint32_t byte = 0u; byte < 16u; ++byte)
            {
                mem.getRDRAM()[kSecondSource + byte] =
                    static_cast<uint8_t>(0x80u + byte);
            }

            t.IsTrue(mem.writeIORegister(kChannel + 0x30u, kTagAddress),
                     "SPR_TO TADR write should succeed");
            t.IsTrue(mem.writeIORegister(kChannel + 0x80u, kScratchAddress),
                     "SPR_TO SADR write should succeed");
            t.IsTrue(mem.writeIORegister(kChannel + 0x00u, 0x104u),
                     "SPR_TO source-chain transfer should start");
            publishDmacCompletions(mem);

            bool copied = true;
            for (uint32_t byte = 0u; byte < 16u; ++byte)
            {
                copied &= mem.getScratchpad()[0x3FF0u + byte] ==
                          static_cast<uint8_t>(0x40u + byte);
                copied &= mem.getScratchpad()[byte] ==
                          static_cast<uint8_t>(0x50u + byte);
                copied &= mem.getScratchpad()[0x10u + byte] ==
                          static_cast<uint8_t>(0x80u + byte);
            }
            t.IsTrue(copied,
                     "SPR_TO should concatenate referenced payloads across scratchpad wrap");
            t.Equals(mem.readIORegister(kChannel + 0x10u), kSecondSource + 16u,
                     "SPR_TO should leave MADR after the terminal payload");
            t.Equals(mem.readIORegister(kChannel + 0x30u), kTagAddress + 32u,
                     "SPR_TO REFE should advance TADR past the terminal tag");
            t.Equals(mem.readIORegister(kChannel + 0x80u), 0x20u,
                     "SPR_TO should advance and wrap SADR for all chain payloads");
            t.Equals(mem.readIORegister(kChannel + 0x20u), 0u,
                     "SPR_TO source chain should consume QWC");
            t.IsTrue((mem.readIORegister(kChannel + 0x00u) & 0x100u) == 0u,
                     "SPR_TO source chain should clear CHCR.STR");

            const std::vector<DmacPendingInterrupt> interrupts =
                mem.consumePendingDmacInterrupts();
            t.Equals(interrupts.size(), static_cast<size_t>(1u),
                     "SPR_TO source chain should queue one completion");
            t.Equals(interrupts[0].cause, 9u,
                     "SPR_TO source chain should queue DMAC cause nine");
        });

        tc.Run("SPR_TO source chain follows inline CNT and END payloads", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kChannel = 0x1000D400u;
            constexpr uint32_t kTagAddress = 0x00026000u;
            constexpr uint32_t kScratchAddress = 0x0200u;

            writeDmaTag(mem.getRDRAM(), kTagAddress + 0x00u,
                        makeDmaTag(1u, 1u, 0u)); // CNT
            writeDmaTag(mem.getRDRAM(), kTagAddress + 0x20u,
                        makeDmaTag(1u, 7u, 0u)); // END
            for (uint32_t byte = 0u; byte < 16u; ++byte)
            {
                mem.getRDRAM()[kTagAddress + 0x10u + byte] =
                    static_cast<uint8_t>(0x20u + byte);
                mem.getRDRAM()[kTagAddress + 0x30u + byte] =
                    static_cast<uint8_t>(0xA0u + byte);
            }

            t.IsTrue(mem.writeIORegister(kChannel + 0x30u, kTagAddress),
                     "SPR_TO TADR write should succeed");
            t.IsTrue(mem.writeIORegister(kChannel + 0x80u, kScratchAddress),
                     "SPR_TO SADR write should succeed");
            t.IsTrue(mem.writeIORegister(kChannel + 0x00u, 0x104u),
                     "SPR_TO source-chain transfer should start");
            publishDmacCompletions(mem);

            bool copied = true;
            for (uint32_t byte = 0u; byte < 16u; ++byte)
            {
                copied &= mem.getScratchpad()[kScratchAddress + byte] ==
                          static_cast<uint8_t>(0x20u + byte);
                copied &= mem.getScratchpad()[kScratchAddress + 0x10u + byte] ==
                          static_cast<uint8_t>(0xA0u + byte);
            }
            t.IsTrue(copied, "SPR_TO should copy both inline chain payloads");
            t.Equals(mem.readIORegister(kChannel + 0x10u), kTagAddress + 0x40u,
                     "SPR_TO should leave MADR after the END payload");
            t.Equals(mem.readIORegister(kChannel + 0x30u), kTagAddress + 0x20u,
                     "SPR_TO END should retain the terminal tag address");
            t.Equals(mem.readIORegister(kChannel + 0x80u), kScratchAddress + 0x20u,
                     "SPR_TO should advance SADR for both inline payloads");
        });

        tc.Run("SPR_FROM normal DMA copies wrapping scratchpad into RDRAM", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kChannel = 0x1000D000u;
            constexpr uint32_t kDestination = 0x00023000u;
            constexpr uint32_t kScratchAddress = 0x3FF0u;
            constexpr uint32_t kQwc = 2u;
            for (uint32_t byte = 0u; byte < 16u; ++byte)
            {
                mem.getScratchpad()[0x3FF0u + byte] = static_cast<uint8_t>(0x80u + byte);
                mem.getScratchpad()[byte] = static_cast<uint8_t>(0x90u + byte);
            }

            t.IsTrue(mem.writeIORegister(kChannel + 0x10u, kDestination),
                     "SPR_FROM MADR write should succeed");
            t.IsTrue(mem.writeIORegister(kChannel + 0x20u, kQwc),
                     "SPR_FROM QWC write should succeed");
            t.IsTrue(mem.writeIORegister(kChannel + 0x80u, kScratchAddress),
                     "SPR_FROM SADR write should succeed");
            t.IsTrue(mem.writeIORegister(kChannel + 0x00u, 0x100u),
                     "SPR_FROM normal transfer should start");
            publishDmacCompletions(mem);

            bool copied = true;
            for (uint32_t byte = 0u; byte < 32u; ++byte)
            {
                copied &= mem.getRDRAM()[kDestination + byte] ==
                          static_cast<uint8_t>(0x80u + byte);
            }
            t.IsTrue(copied, "SPR_FROM should copy both qwords across the scratchpad wrap");
            t.Equals(mem.readIORegister(kChannel + 0x10u), kDestination + 32u,
                     "SPR_FROM should advance MADR");
            t.Equals(mem.readIORegister(kChannel + 0x80u), 0x10u,
                     "SPR_FROM should wrap and advance SADR");
            t.Equals(mem.readIORegister(kChannel + 0x20u), 0u,
                     "SPR_FROM should consume QWC");
            t.IsTrue((mem.readIORegister(0x1000E010u) & (1u << 8u)) != 0u,
                     "SPR_FROM should raise its D_STAT completion bit");

            const std::vector<DmacPendingInterrupt> interrupts =
                mem.consumePendingDmacInterrupts();
            t.Equals(interrupts.size(), static_cast<size_t>(1u),
                     "SPR_FROM should queue one completion");
            t.Equals(interrupts[0].cause, 8u,
                     "SPR_FROM should queue DMAC cause eight");
        });

        tc.Run("GIF DMA can source from scratchpad", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kGifCh = 0x1000A000u;
            constexpr uint32_t kSrcScratch = PS2_SCRATCHPAD_BASE + 0x80u;
            constexpr uint32_t kQwc = 1u; // 16 bytes

            uint8_t *scratch = mem.getScratchpad();
            for (uint32_t i = 0; i < 16u; ++i)
            {
                scratch[0x80u + i] = static_cast<uint8_t>((0xA0u + i) & 0xFFu);
            }

            std::vector<std::vector<uint8_t>> captured;
            mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            t.IsTrue(mem.writeIORegister(kGifCh + 0x10u, kSrcScratch), "write MADR scratchpad should succeed");
            t.IsTrue(mem.writeIORegister(kGifCh + 0x20u, kQwc), "write QWC should succeed");
            t.IsTrue(mem.writeIORegister(kGifCh + 0x00u, 0x100u), "write CHCR STR should succeed");

            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            t.Equals(captured.size(), static_cast<size_t>(1u), "scratchpad GIF DMA should emit one packet");
            t.Equals(captured[0].size(), static_cast<size_t>(16u), "scratchpad GIF DMA packet should be 16 bytes");
            bool contentOk = true;
            for (uint32_t i = 0; i < 16u; ++i)
            {
                if (captured[0][i] != static_cast<uint8_t>((0xA0u + i) & 0xFFu))
                {
                    contentOk = false;
                    break;
                }
            }
            t.IsTrue(contentOk, "scratchpad GIF DMA packet bytes should match scratchpad source");
        });

        tc.Run("GIF DMA chain can source tags and payload from 0xF000 scratchpad alias", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kGifCh = 0x1000A000u;
            constexpr uint32_t kTagAlias = PS2_SCRATCHPAD_ALIAS_BASE + 0x100u;

            uint8_t *scratch = mem.getScratchpad();
            std::memset(scratch + 0x100u, 0, 32u);

            const uint64_t endTag = makeDmaTag(1u, 7u, 0u, false);
            std::memcpy(scratch + 0x100u, &endTag, sizeof(endTag));
            for (uint32_t i = 0; i < 16u; ++i)
            {
                scratch[0x110u + i] = static_cast<uint8_t>(0xC0u + i);
            }

            std::vector<std::vector<uint8_t>> captured;
            mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            t.IsTrue(mem.writeIORegister(kGifCh + 0x30u, kTagAlias), "write TADR scratchpad alias should succeed");
            t.IsTrue(mem.writeIORegister(kGifCh + 0x00u, 0x104u), "write CHCR STR|CHAIN should succeed");

            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            t.Equals(captured.size(), static_cast<size_t>(1u), "scratchpad alias chain should emit one packet");
            t.Equals(captured[0].size(), static_cast<size_t>(16u), "scratchpad alias chain should emit one qword");

            bool contentOk = true;
            for (uint32_t i = 0; i < 16u; ++i)
            {
                if (captured[0][i] != static_cast<uint8_t>(0xC0u + i))
                {
                    contentOk = false;
                    break;
                }
            }
            t.IsTrue(contentOk, "scratchpad alias chain payload should match scratchpad bytes");
        });

        tc.Run("native GIF image upload recognizes canonical load-image chain", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            GS gs;
            gs.init(mem.getGSVRAM(), static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &mem.gs());

            constexpr uint32_t kGifCh = 0x1000A000u;
            constexpr uint32_t kDStat = 0x1000E010u;
            constexpr uint32_t kChain = 0x00028000u;
            constexpr uint32_t kPixels = 0x00029000u;
            constexpr uint32_t kQwc = 1u;

            uint8_t *rdram = mem.getRDRAM();
            for (uint32_t i = 0; i < kQwc * 16u; ++i)
            {
                rdram[kPixels + i] = static_cast<uint8_t>(0x40u + i);
            }

            uint32_t chain = kChain;
            chain = writeTextureUploadSetup(rdram, chain, 0u, GS_PSM_CT32);
            chain = writeTextureImageRef(rdram, chain, kQwc, kPixels);
            writeDmaTag(rdram, chain, makeDmaTag(0u, 7u, 0u, false)); // END.

            t.IsTrue(mem.writeIORegister(kGifCh + 0x30u, kChain), "write GIF TADR should succeed");
            t.IsTrue(mem.tryProcessNativeGifImageUploadChain(gs, kChain, 0x105u),
                     "canonical load-image chain should use the native upload path");
            publishDmacCompletions(mem);

            t.Equals(gs.nativeImageUploadCount(), 1ull, "native GIF DMA chain should upload through GS fast path");
            t.Equals(mem.gifCopyCount(), 1ull, "native GIF DMA chain should still count as a GIF DMA copy");
            t.IsTrue((mem.readIORegister(kDStat) & (1u << 2u)) != 0u,
                     "native GIF DMA chain should raise D_STAT GIF completion");
            t.Equals(mem.readIORegister(kGifCh + 0x20u), 0u, "native GIF DMA chain should clear GIF QWC");
            t.Equals(mem.readIORegister(kGifCh + 0x00u) & 0x100u, 0u,
                     "native GIF DMA chain should clear GIF STR");
            t.Equals(mem.readIORegister(kGifCh + 0x00u) & 0x70000000u, 0x70000000u,
                     "native GIF DMA chain should latch the terminal END tag id");

            bool pixelsOk = true;
            for (uint32_t x = 0; x < 4u && pixelsOk; ++x)
            {
                const uint32_t dstOff = GSPSMCT32::addrPSMCT32(0u, 1u, x, 0u);
                const uint32_t srcOff = kPixels + x * 4u;
                for (uint32_t c = 0; c < 4u; ++c)
                {
                    if (mem.getGSVRAM()[dstOff + c] != rdram[srcOff + c])
                    {
                        pixelsOk = false;
                        break;
                    }
                }
            }
            t.IsTrue(pixelsOk, "native GIF DMA chain should upload image payload into GS VRAM");
        });

        tc.Run("native GIF packed chain matches generic packed primitive packet", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            GS nativeGs;
            nativeGs.init(mem.getGSVRAM(), static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &mem.gs());

            GSRegisters genericRegs{};
            std::vector<uint8_t> genericVram(PS2_GS_VRAM_SIZE, 0u);
            GS genericGs;
            genericGs.init(genericVram.data(), static_cast<uint32_t>(genericVram.size()), &genericRegs);

            std::vector<uint8_t> packet;
            appendU64(packet, makeGifTag(4u, GIF_FMT_PACKED, 1u, false));
            appendU64(packet, 0x0Eull);
            appendU64(packet, makeGsFrame(0u, 1u, GS_PSM_CT32));
            appendU64(packet, GS_REG_FRAME_1);
            appendU64(packet, makeGsScissor(0u, 7u, 0u, 7u));
            appendU64(packet, GS_REG_SCISSOR_1);
            appendU64(packet, 1ull << 17u); // ZTST always.
            appendU64(packet, GS_REG_TEST_1);
            appendU64(packet, 1ull << 32u); // Mask Z writes so the test framebuffer remains visible.
            appendU64(packet, GS_REG_ZBUF_1);

            constexpr uint16_t kSpritePrim = static_cast<uint16_t>(GS_PRIM_SPRITE);
            appendU64(packet, makeGifTagPrim(2u, kSpritePrim, GIF_FMT_PACKED, 3u, true, true));
            appendU64(packet, static_cast<uint64_t>(GS_REG_UV) |
                                  (static_cast<uint64_t>(GS_REG_RGBAQ) << 4u) |
                                  (static_cast<uint64_t>(GS_REG_XYZF2) << 8u));
            appendPackedUv(packet, 0u, 0u);
            appendPackedRgbaq(packet, 0x20u, 0x40u, 0x80u, 0x80u);
            appendPackedXyzf2(packet, 0u, 0u, 0u);
            appendPackedUv(packet, 0u, 0u);
            appendPackedRgbaq(packet, 0xE0u, 0x30u, 0x10u, 0x80u);
            appendPackedXyzf2(packet, 64u, 64u, 0u);

            genericGs.processGIFPacket(packet.data(), static_cast<uint32_t>(packet.size()));

            constexpr uint32_t kGifCh = 0x1000A000u;
            constexpr uint32_t kDStat = 0x1000E010u;
            constexpr uint32_t kScratchTag = 0xF0000000u;
            uint8_t *scratch = mem.getScratchpad();
            writeDmaTag(scratch, 0u, makeDmaTag(static_cast<uint16_t>(packet.size() / 16u), 7u, 0u, false));
            std::memcpy(scratch + 16u, packet.data(), packet.size());

            t.IsTrue(mem.writeIORegister(kGifCh + 0x30u, kScratchTag), "write GIF TADR scratchpad alias should succeed");
            t.IsTrue(mem.tryProcessNativeGifPackedChain(nativeGs, kScratchTag, 0x105u),
                     "packed primitive chain should use the native packed GIF path");
            publishDmacCompletions(mem);

            t.Equals(nativeGs.nativePackedGIFPacketCount(), 1ull, "native packed GIF packet counter should increment");
            t.Equals(mem.gifCopyCount(), 1ull, "native packed GIF chain should still count as a GIF DMA copy");
            t.IsTrue((mem.readIORegister(kDStat) & (1u << 2u)) != 0u,
                     "native packed GIF chain should raise D_STAT GIF completion");
            t.Equals(mem.readIORegister(kGifCh + 0x20u), 0u, "native packed GIF chain should clear GIF QWC");
            t.Equals(mem.readIORegister(kGifCh + 0x00u) & 0x100u, 0u,
                     "native packed GIF chain should clear GIF STR");

            const uint32_t nativePixel = nativeGs.ReadVram(GS_PSM_CT32, 0u, 1u, 1u, 1u);
            const uint32_t genericPixel = genericGs.ReadVram(GS_PSM_CT32, 0u, 1u, 1u, 1u);
            t.IsTrue(genericPixel != 0u, "generic packed primitive packet should draw a test pixel");
            t.Equals(nativePixel, genericPixel, "native packed GIF chain should match generic GS packet output");
        });

        tc.Run("GIF DMA chain REF keeps CT32 image data after paletted upload", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            GS gs;
            gs.init(mem.getGSVRAM(), static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &mem.gs());
            std::vector<uint8_t> capturedGifPacket;
            mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
            {
                capturedGifPacket.assign(data, data + sizeBytes);
                gs.processGIFPacket(data, sizeBytes);
            });

            constexpr uint32_t kGifCh = 0x1000A000u;
            constexpr uint32_t kChain = 0x00028000u;
            constexpr uint32_t kT4Data = 0x00029000u;
            constexpr uint32_t kCt32Data = 0x0002A000u;
            constexpr uint32_t kT4Dbp = 0u;
            constexpr uint32_t kCt32Dbp = 32u;
            constexpr uint32_t kWidth = 16u;
            constexpr uint32_t kHeight = 16u;
            constexpr uint32_t kT4Bytes = kWidth * kHeight / 2u;
            constexpr uint32_t kCt32Bytes = kWidth * kHeight * 4u;

            uint8_t *rdram = mem.getRDRAM();
            uint32_t pixel = 0u;
            for (uint32_t i = 0; i < kT4Bytes; ++i)
            {
                const uint8_t lo = static_cast<uint8_t>(pixel & 0xFu);
                const uint8_t hi = static_cast<uint8_t>((pixel + 1u) & 0xFu);
                rdram[kT4Data + i] = static_cast<uint8_t>(lo | (hi << 4));
                pixel += 2u;
                if (pixel > 0xEu)
                {
                    pixel -= 0xEu;
                }
            }

            uint32_t color = 0u;
            for (uint32_t i = 0; i < kCt32Bytes; i += 4u)
            {
                rdram[kCt32Data + i + 0u] = static_cast<uint8_t>((color >> 0) & 0xFFu);
                rdram[kCt32Data + i + 1u] = static_cast<uint8_t>((color >> 8) & 0xFFu);
                rdram[kCt32Data + i + 2u] = static_cast<uint8_t>((color >> 16) & 0xFFu);
                rdram[kCt32Data + i + 3u] = 0x80u;
                color += 0xF1u;
                if (color >= 0xFFFFFFu)
                {
                    color = 0u;
                }
            }

            uint32_t chain = kChain;
            chain = writeTextureUploadSetup(rdram, chain, kT4Dbp, GS_PSM_T4HL);
            chain = writeTextureImageRef(rdram, chain, kT4Bytes / 16u, kT4Data);
            chain = writeTextureUploadSetup(rdram, chain, kCt32Dbp, GS_PSM_CT32);
            chain = writeTextureImageRef(rdram, chain, kCt32Bytes / 16u, kCt32Data);
            writeDmaTag(rdram, chain, makeDmaTag(0u, 7u, 0u, false)); // END.

            t.IsTrue(mem.writeIORegister(kGifCh + 0x30u, kChain), "write GIF TADR should succeed");
            t.IsTrue(mem.writeIORegister(kGifCh + 0x00u, 0x104u), "write GIF CHCR STR|CHAIN should succeed");

            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            constexpr uint32_t kCt32PayloadOffset = 5u * 16u + 16u + kT4Bytes + 5u * 16u + 16u;
            t.IsTrue(capturedGifPacket.size() >= kCt32PayloadOffset + kCt32Bytes,
                     "flattened GIF chain should contain the full CT32 REF payload");
            bool ct32PayloadOk = capturedGifPacket.size() >= kCt32PayloadOffset + kCt32Bytes;
            for (uint32_t i = 0; i < kCt32Bytes && ct32PayloadOk; ++i)
            {
                if (capturedGifPacket[kCt32PayloadOffset + i] != rdram[kCt32Data + i])
                {
                    ct32PayloadOk = false;
                    break;
                }
            }
            t.IsTrue(ct32PayloadOk, "flattened GIF chain should preserve CT32 REF bytes after the T4 REF payload");

            const uint32_t row1Off = GSPSMCT32::addrPSMCT32(kCt32Dbp, 1u, 0u, 1u);
            uint32_t actualRow1X0 = 0u;
            uint32_t expectedRow1X0 = 0u;
            std::memcpy(&actualRow1X0, mem.getGSVRAM() + row1Off, sizeof(actualRow1X0));
            std::memcpy(&expectedRow1X0, rdram + kCt32Data + kWidth * 4u, sizeof(expectedRow1X0));
            t.Equals(actualRow1X0, expectedRow1X0, "CT32 row 1 must come from CT32 data, not the previous T4 REF payload");

            bool ct32Ok = true;
            for (uint32_t y = 0; y < kHeight && ct32Ok; ++y)
            {
                for (uint32_t x = 0; x < kWidth && ct32Ok; ++x)
                {
                    const uint32_t dstOff = GSPSMCT32::addrPSMCT32(kCt32Dbp, 1u, x, y);
                    const uint32_t srcOff = kCt32Data + ((y * kWidth + x) * 4u);
                    for (uint32_t c = 0; c < 4u; ++c)
                    {
                        if (mem.getGSVRAM()[dstOff + c] != rdram[srcOff + c])
                        {
                            ct32Ok = false;
                            break;
                        }
                    }
                }
            }
            t.IsTrue(ct32Ok, "all CT32 pixels should survive a preceding paletted upload in the same GIF DMA chain");
        });

        tc.Run("VIF1 DMA DIRECT forwards payload to GIF callback and clears channel", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1Ch = 0x10009000u;
            constexpr uint32_t kSrc = 0x00024000u;
            constexpr uint32_t kQwc = 2u; // 32 bytes total transport

            uint8_t *rdram = mem.getRDRAM();
            std::memset(rdram + kSrc, 0, kQwc * 16u);

            // DIRECT 1 QW.
            const uint32_t cmd = makeVifCmd(0x50u, 0u, 1u);
            std::memcpy(rdram + kSrc, &cmd, sizeof(cmd));
            for (uint32_t i = 0; i < 16u; ++i)
            {
                rdram[kSrc + 4u + i] = static_cast<uint8_t>((0x11u + i) & 0xFFu);
            }

            std::vector<std::vector<uint8_t>> captured;
            mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x10u, kSrc), "write VIF1 MADR should succeed");
            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x20u, kQwc), "write VIF1 QWC should succeed");
            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x00u, 0x100u), "write VIF1 CHCR STR should succeed");

            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            t.Equals(captured.size(), static_cast<size_t>(1u), "VIF1 DIRECT should emit one GIF packet");
            t.Equals(captured[0].size(), static_cast<size_t>(16u), "VIF1 DIRECT packet should be 1 QW");
            bool contentOk = true;
            for (uint32_t i = 0; i < 16u; ++i)
            {
                if (captured[0][i] != static_cast<uint8_t>((0x11u + i) & 0xFFu))
                {
                    contentOk = false;
                    break;
                }
            }
            t.IsTrue(contentOk, "VIF1 DIRECT packet bytes should match payload");
            t.IsTrue((mem.readIORegister(kVif1Ch + 0x00u) & 0x100u) == 0u, "VIF1 CHCR STR bit should be cleared after drain");
            t.Equals(mem.readIORegister(kVif1Ch + 0x20u), 0u, "VIF1 QWC should be cleared after drain");
        });

        tc.Run("VIF1 DMA completion remains busy while VIF waits for VU1", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(),
                     "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1Ch = 0x10009000u;
            constexpr uint32_t kSrc = 0x00024800u;
            constexpr uint32_t kQwc = 1u;
            uint8_t *const rdram = mem.getRDRAM();
            std::memset(rdram + kSrc, 0, 16u);
            const uint32_t flush = makeVifCmd(0x11u, 0u, 0u);
            const uint32_t stmod = makeVifCmd(0x05u, 0u, 3u);
            std::memcpy(rdram + kSrc, &flush, sizeof(flush));
            std::memcpy(
                rdram + kSrc + sizeof(flush),
                &stmod, sizeof(stmod));

            bool vuBusy = true;
            mem.setVu1BusyCallback(
                [&]()
                {
                    return vuBusy;
                });
            t.IsTrue(
                mem.writeIORegister(kVif1Ch + 0x10u, kSrc),
                "write VIF1 MADR should succeed");
            t.IsTrue(
                mem.writeIORegister(kVif1Ch + 0x20u, kQwc),
                "write VIF1 QWC should succeed");
            t.IsTrue(
                mem.writeIORegister(kVif1Ch + 0x00u, 0x100u),
                "write VIF1 CHCR STR should succeed");

            mem.processPendingTransfers();
            t.IsTrue(mem.vif1WaitingForVu(),
                     "FLUSH should leave VIF1 waiting for active VU1");
            t.IsTrue(
                (mem.vif1_regs.stat & (1u << 2u)) != 0u,
                "the wait should set VIF1_STAT.VEW");
            t.Equals(mem.vif1_regs.mode, 0u,
                     "post-FLUSH commands should remain deferred");
            t.IsFalse(mem.hasReadyDmacCompletions(),
                      "VIF1 DMAC completion must wait with the parser");
            t.IsTrue(
                (mem.readIORegister(kVif1Ch + 0x00u) & 0x100u) != 0u,
                "VIF1 CHCR.STR should remain busy during the wait");

            vuBusy = false;
            t.IsTrue(mem.resumeVIF1AfterVu(),
                     "VU completion should wake the retained VIF work");
            t.IsFalse(mem.vif1WaitingForVu(),
                      "wake should clear explicit VIF wait state");
            t.Equals(mem.vif1_regs.mode, 3u,
                     "wake should resume deferred VIF commands");
            t.IsTrue(mem.hasReadyDmacCompletions(),
                     "wake should make the retained DMAC completion ready");
            t.IsTrue(
                (mem.readIORegister(kVif1Ch + 0x00u) & 0x100u) != 0u,
                "completion publication should still own CHCR.STR");

            t.Equals(publishDmacCompletions(mem), static_cast<size_t>(1u),
                     "retained VIF1 completion should publish once");
            t.IsTrue(
                (mem.readIORegister(kVif1Ch + 0x00u) & 0x100u) == 0u,
                "publication should clear VIF1 CHCR.STR");
        });

        tc.Run("VIF1 scheduled parser retains a split command across chain events", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(),
                     "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1 = 0x10009000u;
            constexpr uint32_t kTag = 0x00024900u;
            uint8_t *const rdram = mem.getRDRAM();
            std::memset(rdram + kTag, 0, 48u);

            const uint64_t cnt =
                makeDmaTag(0u, 1u, 0u, false);
            const uint64_t end =
                makeDmaTag(1u, 7u, 0u, false);
            std::memcpy(rdram + kTag, &cnt, sizeof(cnt));
            std::memcpy(
                rdram + kTag + 16u, &end, sizeof(end));

            const uint32_t strow =
                makeVifCmd(0x30u, 0u, 0u);
            const std::array<uint32_t, 4> row = {
                0x11111111u,
                0x22222222u,
                0x33333333u,
                0x44444444u,
            };
            std::memcpy(
                rdram + kTag + 8u,
                &strow, sizeof(strow));
            std::memcpy(
                rdram + kTag + 12u,
                &row[0], sizeof(row[0]));
            std::memcpy(
                rdram + kTag + 24u,
                &row[1], sizeof(row[1]) * 2u);
            std::memcpy(
                rdram + kTag + 32u,
                &row[3], sizeof(row[3]));

            uint32_t scheduleRequests = 0u;
            mem.setVif1DmaScheduleCallback(
                [&](uint32_t)
                {
                    ++scheduleRequests;
                    return true;
                });
            t.IsTrue(mem.writeIORegister(
                         kVif1 + 0x30u, kTag),
                     "split command TADR write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif1, 0x145u),
                     "split command chain start should succeed");
            t.Equals(scheduleRequests, 1u,
                     "submission should request one initial event");

            Vif1DmaAdvanceResult advance =
                mem.advanceVif1Dma();
            t.IsTrue(advance.progressed,
                     "first event should fetch the CNT tag");
            t.Equals(
                mem.vif1DmaSnapshot().parserBytes, 8u,
                "STROW header and first word should remain buffered");
            t.Equals(mem.vif1_regs.row[0], 0u,
                     "an incomplete STROW must not mutate row state");

            advance = mem.advanceVif1Dma();
            t.IsTrue(advance.progressed,
                     "second event should fetch the END tag");
            t.Equals(
                mem.vif1DmaSnapshot().parserBytes, 16u,
                "second tag high bytes should extend the same command");
            t.Equals(mem.vif1_regs.row[0], 0u,
                     "STROW should remain atomic until its full payload");

            advance = mem.advanceVif1Dma();
            t.IsTrue(advance.progressed,
                     "third event should transfer the END payload");
            for (size_t index = 0u;
                 index < row.size(); ++index)
            {
                t.Equals(
                    mem.vif1_regs.row[index], row[index],
                    "split STROW should commit the retained row");
            }
            t.Equals(
                mem.vif1DmaSnapshot().parserBytes, 0u,
                "complete command and trailing NOPs should drain");

            advance = mem.advanceVif1Dma();
            t.IsTrue(advance.completed,
                     "a separate final event should request completion");
            t.IsTrue(mem.hasReadyDmacCompletions(),
                     "final state should use typed DMAC completion");
        });

        tc.Run("VIF1 scheduled normal DMA normalizes QWC and retires event ownership", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(),
                     "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1 = 0x10009000u;
            constexpr uint32_t kSource = 0x00024980u;
            std::memset(
                mem.getRDRAM() + kSource, 0, 16u);

            uint32_t initialDelay = 0u;
            mem.setVif1DmaScheduleCallback(
                [&](uint32_t delayEeCycles)
                {
                    initialDelay = delayEeCycles;
                    return true;
                });
            t.IsTrue(mem.writeIORegister(
                         kVif1 + 0x10u, kSource),
                     "QWC fixture MADR write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif1 + 0x20u, 0x00010001u),
                     "QWC fixture write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif1, 0x100u),
                     "QWC fixture start should succeed");

            Vif1DmaSnapshot snapshot =
                mem.vif1DmaSnapshot();
            t.IsTrue(
                snapshot.active &&
                    snapshot.eventManaged,
                     "submission should retain event ownership");
            t.Equals(snapshot.qwc, 1u,
                     "VIF1 QWC should expose only its hardware bits");
            t.Equals(
                mem.readIORegister(kVif1 + 0x20u), 1u,
                "the visible QWC register should be normalized");
            t.Equals(initialDelay, 4u,
                     "normal submission should retain the reference startup delay");

            const Vif1DmaAdvanceResult payload =
                mem.advanceVif1Dma();
            t.IsTrue(payload.progressed,
                     "normalized QWC should transfer one qword");
            t.Equals(payload.delayEeCycles, 2u,
                     "one qword should cost two EE cycles");
            t.Equals(
                mem.readIORegister(kVif1 + 0x10u),
                kSource + 16u,
                "normal payload should advance MADR by one qword");

            const Vif1DmaAdvanceResult completed =
                mem.advanceVif1Dma();
            t.IsTrue(completed.completed,
                     "the later final state should request completion");
            snapshot = mem.vif1DmaSnapshot();
            t.IsFalse(snapshot.active,
                      "finalization should retire the descriptor");
            t.IsFalse(snapshot.eventManaged,
                      "finalization should retire event ownership");
        });

        tc.Run("VIF1 chain TIE and tag IRQ stop after the tagged payload", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(),
                     "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1 = 0x10009000u;
            constexpr uint32_t kTag = 0x000249C0u;
            uint8_t *const rdram = mem.getRDRAM();
            std::memset(rdram + kTag, 0, 48u);

            const uint64_t irqCnt =
                makeDmaTag(1u, 1u, 0u, true);
            const uint64_t ignoredEnd =
                makeDmaTag(0u, 7u, 0u, false);
            std::memcpy(
                rdram + kTag, &irqCnt, sizeof(irqCnt));
            std::memcpy(
                rdram + kTag + 32u,
                &ignoredEnd, sizeof(ignoredEnd));

            mem.setVif1DmaScheduleCallback(
                [](uint32_t)
                {
                    return true;
                });
            t.IsTrue(mem.writeIORegister(
                         kVif1 + 0x30u, kTag),
                     "TIE fixture TADR write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif1, 0x185u),
                     "TIE fixture chain start should succeed");

            Vif1DmaAdvanceResult advance =
                mem.advanceVif1Dma();
            Vif1DmaSnapshot snapshot =
                mem.vif1DmaSnapshot();
            t.IsTrue(advance.progressed &&
                         snapshot.tagIrq &&
                         snapshot.endAfterPayload,
                     "TIE should make the IRQ tag terminal");
            t.IsTrue(
                (mem.readIORegister(kVif1) &
                 0x80000000u) != 0u,
                "tag setup should expose the IRQ bit in CHCR");

            advance = mem.advanceVif1Dma();
            snapshot = mem.vif1DmaSnapshot();
            t.IsTrue(
                snapshot.phase == Vif1DmaPhase::Finalize,
                "the IRQ-tag payload should lead to finalization");
            t.Equals(snapshot.tagsProcessed, 1u,
                     "the following tag must remain unfetched");

            advance = mem.advanceVif1Dma();
            t.IsTrue(advance.completed,
                     "a separate event should request the final IRQ");
            t.Equals(
                publishDmacCompletions(mem),
                static_cast<size_t>(1u),
                "the terminal tag should publish one completion");
            t.IsTrue(
                (mem.readIORegister(0x1000E010u) &
                 0x2u) != 0u,
                "VIF1 completion should latch its D_STAT cause");
        });

        tc.Run("VIF1 STOP ForceBreak and DMA disable retain and wake scheduled work", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(),
                     "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1 = 0x10009000u;
            constexpr uint32_t kDctrl = 0x1000E000u;
            constexpr uint32_t kSource = 0x00024A00u;
            std::memset(
                mem.getRDRAM() + kSource, 0, 16u);

            uint32_t schedules = 0u;
            uint32_t cancels = 0u;
            mem.setVif1DmaScheduleCallback(
                [&](uint32_t)
                {
                    ++schedules;
                    return true;
                });
            mem.setVif1DmaCancelCallback(
                [&]()
                {
                    ++cancels;
                });

            t.IsTrue(mem.writeIORegister(
                         kVif1 + 0x10u, kSource),
                     "stall fixture MADR write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif1 + 0x20u, 1u),
                     "stall fixture QWC write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif1, 0x100u),
                     "stall fixture start should succeed");

            t.IsTrue(mem.writeIORegister(
                         0x10003C10u, 0x4u),
                     "FBRST.STP should succeed");
            t.IsTrue(
                (mem.vif1_regs.stat & (1u << 8u)) != 0u,
                "STOP should set VSS");
            Vif1DmaAdvanceResult stalled =
                mem.advanceVif1Dma();
            t.IsTrue(
                stalled.stall ==
                    Vif1DmaStallReason::VifStop,
                "STOP should prevent payload progress");
            t.Equals(
                mem.readIORegister(kVif1 + 0x20u), 1u,
                "STOP should retain QWC");

            t.IsTrue(mem.writeIORegister(
                         0x10003C10u, 0x8u),
                     "FBRST.STC should wake STOP");
            t.IsTrue(
                (mem.vif1_regs.stat & (1u << 8u)) == 0u,
                "STC should clear VSS");
            t.IsTrue(schedules >= 2u,
                     "STC should request a current-time retry");

            t.IsTrue(mem.writeIORegister(
                         0x10003C10u, 0x2u),
                     "FBRST.FBK should succeed");
            t.IsTrue(
                (mem.vif1_regs.stat & (1u << 9u)) != 0u,
                "ForceBreak should set VFS");
            stalled = mem.advanceVif1Dma();
            t.IsTrue(
                stalled.stall ==
                    Vif1DmaStallReason::ForceBreak,
                "ForceBreak should retain the descriptor");
            t.IsTrue(mem.writeIORegister(
                         0x10003C10u, 0x8u),
                     "STC should clear ForceBreak");

            t.IsTrue(mem.writeIORegister(kDctrl, 0u),
                     "clearing DMAE should succeed");
            stalled = mem.advanceVif1Dma();
            t.IsTrue(
                stalled.stall ==
                    Vif1DmaStallReason::DmacDisabled,
                "DMAE=0 should prevent progress");
            t.Equals(
                mem.readIORegister(kVif1 + 0x20u), 1u,
                "DMA disable should retain QWC");

            t.IsTrue(mem.writeIORegister(kDctrl, 1u),
                     "setting DMAE should wake the channel");
            Vif1DmaAdvanceResult payload =
                mem.advanceVif1Dma();
            t.IsTrue(payload.progressed,
                     "re-enabled DMA should transfer payload");
            t.Equals(
                mem.readIORegister(kVif1 + 0x20u), 0u,
                "payload progress should consume QWC");
            t.IsTrue(cancels >= 3u,
                     "each external stall should cancel its pending event");

            const Vif1DmaAdvanceResult completed =
                mem.advanceVif1Dma();
            t.IsTrue(completed.completed,
                     "post-enable finalization should complete");
        });

        tc.Run("VIF1 command IRQ stalls retained DMA until STC", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(),
                     "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1 = 0x10009000u;
            constexpr uint32_t kSource = 0x00024B00u;
            const std::array<uint32_t, 4> packet = {
                makeVifCmd(0x00u, 0u, 0u) |
                    0x80000000u,
                makeVifCmd(0x05u, 0u, 3u),
                0u,
                0u,
            };
            std::memcpy(
                mem.getRDRAM() + kSource,
                packet.data(), sizeof(packet));

            t.IsTrue(mem.writeIORegister(
                         kVif1 + 0x10u, kSource),
                     "IRQ fixture MADR write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif1 + 0x20u, 1u),
                     "IRQ fixture QWC write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif1, 0x100u),
                     "IRQ fixture start should succeed");

            const Vif1DmaSnapshot stalled =
                mem.vif1DmaSnapshot();
            t.IsTrue(stalled.active,
                     "command IRQ should retain active DMA");
            t.IsTrue(
                stalled.stall ==
                    Vif1DmaStallReason::VifInterrupt,
                "descriptor should report interrupt stall");
            t.IsTrue(
                (mem.vif1_regs.stat & (1u << 10u)) != 0u,
                "command IRQ should set VIS");
            t.IsTrue(
                (mem.vif1_regs.stat & (1u << 11u)) != 0u,
                "command IRQ should set INT");
            t.Equals(mem.vif1_regs.mode, 0u,
                     "post-IRQ STMOD should remain retained");
            t.IsFalse(mem.hasReadyDmacCompletions(),
                      "interrupt stall should not complete DMA");

            t.IsTrue(mem.writeIORegister(
                         0x10003C10u, 0x8u),
                     "STC should cancel the command stall");
            t.Equals(mem.vif1_regs.mode, 3u,
                     "STC wake should resume retained commands");
            t.IsTrue(mem.hasReadyDmacCompletions(),
                     "resumed final state should request completion");
        });

        tc.Run("VIF1 split IRQ command stalls only after its payload commits", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(),
                     "PS2Memory initialize should succeed");

            const uint32_t strow =
                makeVifCmd(0x30u, 0u, 0u) |
                0x80000000u;
            const std::array<uint32_t, 4> row = {
                0x11223344u,
                0x55667788u,
                0x99AABBCCu,
                0xDDEEFF00u,
            };
            std::array<uint8_t, 20> packet{};
            std::memcpy(
                packet.data(), &strow, sizeof(strow));
            std::memcpy(
                packet.data() + sizeof(strow),
                row.data(), sizeof(row));

            mem.processVIF1Data(packet.data(), 8u);
            t.IsTrue(
                (mem.vif1_regs.stat & (1u << 11u)) != 0u,
                "the parser should retain the observed IRQ bit");
            t.IsTrue(
                (mem.vif1_regs.stat & (1u << 10u)) == 0u,
                "an incomplete command must not raise VIS");
            t.Equals(mem.vif1_regs.row[0], 0u,
                     "an incomplete STROW must remain atomic");

            mem.processVIF1Data(
                packet.data() + 8u, 12u);
            for (size_t index = 0u;
                 index < row.size(); ++index)
            {
                t.Equals(
                    mem.vif1_regs.row[index], row[index],
                    "the completed STROW should commit every lane");
            }
            t.IsTrue(
                (mem.vif1_regs.stat & (1u << 10u)) != 0u,
                "VIS should assert after the command commits");
            t.Equals(
                mem.vif1DeferredByteCount(),
                static_cast<size_t>(0u),
                "the completed command should leave no retained bytes");
        });

        tc.Run("VIF1 DIRECT waits for GIF path availability and retries", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(),
                     "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1 = 0x10009000u;
            constexpr uint32_t kSource = 0x00024C00u;
            std::memset(
                mem.getRDRAM() + kSource, 0, 32u);
            const uint32_t direct =
                makeVifCmd(0x51u, 0u, 1u);
            std::memcpy(
                mem.getRDRAM() + kSource,
                &direct, sizeof(direct));

            bool pathAvailable = false;
            mem.setVif1GifPathAvailableCallback(
                [&](bool directHl)
                {
                    t.IsTrue(directHl,
                             "fixture should query DIRECTHL availability");
                    return pathAvailable;
                });
            std::vector<uint8_t> captured;
            mem.setGifPacketCallback(
                [&](const uint8_t *data, uint32_t size)
                {
                    captured.assign(data, data + size);
                });

            t.IsTrue(mem.writeIORegister(
                         kVif1 + 0x10u, kSource),
                     "GIF-wait MADR write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif1 + 0x20u, 2u),
                     "GIF-wait QWC write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif1, 0x100u),
                     "GIF-wait start should succeed");

            Vif1DmaSnapshot stalled =
                mem.vif1DmaSnapshot();
            t.IsTrue(stalled.active,
                     "unavailable PATH2 should retain DMA");
            t.IsTrue(
                stalled.stall ==
                    Vif1DmaStallReason::GifPath,
                "descriptor should report GIF path stall");
            t.IsTrue(
                (mem.vif1_regs.stat & (1u << 3u)) != 0u,
                "GIF path wait should set VGW");
            t.Equals(captured.size(), static_cast<size_t>(0u),
                     "stalled DIRECTHL should emit no packet");

            pathAvailable = true;
            t.IsTrue(mem.wakeVif1Dma(),
                     "GIF path availability should wake VIF1");
            t.IsTrue(
                (mem.vif1_regs.stat & (1u << 3u)) == 0u,
                "successful retry should clear VGW");
            t.Equals(captured.size(), static_cast<size_t>(16u),
                     "successful retry should emit one qword");
            t.IsTrue(mem.hasReadyDmacCompletions(),
                     "successful retry should reach final completion");
        });

        tc.Run("VIF1 DMA chain transfers tag high bytes for DIRECT packets when TTE is enabled", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1Ch = 0x10009000u;
            constexpr uint32_t kTag = 0x00025000u;

            uint8_t *rdram = mem.getRDRAM();
            std::memset(rdram + kTag, 0, 32u);

            const uint64_t endTag = makeDmaTag(1u, 7u, 0u, false);
            std::memcpy(rdram + kTag, &endTag, sizeof(endTag));

            // Compact VIF1 packet helpers place the DIRECT command in the tag's upper 64 bits.
            const uint32_t directCmd = makeVifCmd(0x50u, 0u, 1u);
            std::memcpy(rdram + kTag + 12u, &directCmd, sizeof(directCmd));
            for (uint32_t i = 0; i < 16u; ++i)
            {
                rdram[kTag + 16u + i] = static_cast<uint8_t>(0x70u + i);
            }

            std::vector<std::vector<uint8_t>> captured;
            mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x30u, kTag), "write VIF1 TADR should succeed");
            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x00u, 0x145u), "write VIF1 CHCR DIR|TTE|STR|CHAIN should succeed");

            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            t.Equals(captured.size(), static_cast<size_t>(1u), "compact VIF1 chain should emit one GIF packet");
            t.Equals(captured[0].size(), static_cast<size_t>(16u), "compact VIF1 DIRECT packet should be 1 QW");

            bool payloadOk = true;
            for (uint32_t i = 0; i < 16u; ++i)
            {
                if (captured[0][i] != static_cast<uint8_t>(0x70u + i))
                {
                    payloadOk = false;
                    break;
                }
            }
            t.IsTrue(payloadOk, "compact VIF1 chain payload should reach the GIF callback");
            t.IsTrue((mem.readIORegister(kVif1Ch + 0x00u) & 0x100u) == 0u,
                     "compact VIF1 chain should clear the STR bit after drain");
        });

        tc.Run("VIF1 DMA chain transfers tag high bytes when qwc is zero and TTE is enabled", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1Ch = 0x10009000u;
            constexpr uint32_t kTag = 0x00025100u;

            uint8_t *rdram = mem.getRDRAM();
            std::memset(rdram + kTag, 0, 16u);

            const uint64_t endTag = makeDmaTag(0u, 7u, 0u, false);
            std::memcpy(rdram + kTag, &endTag, sizeof(endTag));

            const uint32_t itopCmd = makeVifCmd(0x04u, 0u, 0x44u);
            std::memcpy(rdram + kTag + 12u, &itopCmd, sizeof(itopCmd));

            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x30u, kTag), "write VIF1 TADR should succeed");
            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x00u, 0x145u), "write VIF1 CHCR DIR|TTE|STR|CHAIN should succeed");

            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            t.Equals(mem.vif1_regs.itops, 0x44u,
                     "qwc-zero compact VIF1 chain should still process high-half VIFcodes");
            t.IsTrue((mem.readIORegister(kVif1Ch + 0x00u) & 0x100u) == 0u,
                     "qwc-zero compact VIF1 chain should clear the STR bit after drain");
        });

        tc.Run("VIF1 DMA chain ignores tag high bytes when TTE is disabled", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1Ch = 0x10009000u;
            constexpr uint32_t kTag = 0x00025200u;

            uint8_t *rdram = mem.getRDRAM();
            std::memset(rdram + kTag, 0, 16u);
            const uint64_t endTag = makeDmaTag(0u, 7u, 0u, false);
            std::memcpy(rdram + kTag, &endTag, sizeof(endTag));

            const uint32_t itopCmd = makeVifCmd(0x04u, 0u, 0x55u);
            std::memcpy(rdram + kTag + 12u, &itopCmd, sizeof(itopCmd));

            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x30u, kTag), "write VIF1 TADR should succeed");
            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x00u, 0x105u), "write VIF1 CHCR DIR|STR|CHAIN should succeed");

            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            t.Equals(mem.vif1_regs.itops, 0u, "tag high VIFcodes must not transfer while TTE is clear");
        });

        tc.Run("VIF1 DMA chain transfers REF tag high bytes before referenced payload", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1Ch = 0x10009000u;
            constexpr uint32_t kTag = 0x00025300u;
            constexpr uint32_t kPayload = 0x00025400u;

            uint8_t *rdram = mem.getRDRAM();
            std::memset(rdram + kTag, 0, 32u);
            std::memset(rdram + kPayload, 0, 16u);

            const uint64_t refTag = makeDmaTag(1u, 3u, kPayload, false);
            std::memcpy(rdram + kTag, &refTag, sizeof(refTag));
            const uint32_t directCmd = makeVifCmd(0x50u, 0u, 1u);
            std::memcpy(rdram + kTag + 12u, &directCmd, sizeof(directCmd));

            const uint64_t endTag = makeDmaTag(0u, 7u, 0u, false);
            std::memcpy(rdram + kTag + 16u, &endTag, sizeof(endTag));
            for (uint32_t i = 0; i < 16u; ++i)
                rdram[kPayload + i] = static_cast<uint8_t>(0x90u + i);

            std::vector<std::vector<uint8_t>> captured;
            mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x30u, kTag), "write VIF1 TADR should succeed");
            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x00u, 0x145u), "write VIF1 CHCR DIR|TTE|STR|CHAIN should succeed");

            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            t.Equals(captured.size(), static_cast<size_t>(1u), "REF tag DIRECT should emit one GIF packet");
            t.Equals(captured[0].size(), static_cast<size_t>(16u), "REF tag DIRECT packet should be one qword");
            t.IsTrue(std::memcmp(captured[0].data(), rdram + kPayload, 16u) == 0,
                     "REF tag high VIFcode must execute before referenced payload");
        });

        tc.Run("VIF1 packet builders keep chain qwc live before terminate", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1Ch = 0x10009000u;
            constexpr uint32_t kStateAddr = 0x00027000u;
            constexpr uint32_t kBaseAddr = 0x00027100u;

            uint8_t *rdram = mem.getRDRAM();
            std::memset(rdram + kStateAddr, 0, 0x200u);

            R5900Context ctx{};
            setRegU32(ctx, 4, kStateAddr);
            setRegU32(ctx, 5, kBaseAddr);
            ps2_stubs::sceVif1PkInit(rdram, &ctx, nullptr);

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kStateAddr);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::sceVif1PkCnt(rdram, &ctx, nullptr);

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kStateAddr);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::sceVif1PkOpenDirectCode(rdram, &ctx, nullptr);

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kStateAddr);
            setRegU32(ctx, 5, 4u); // reserve one qword of DIRECT payload
            ps2_stubs::sceVif1PkReserve(rdram, &ctx, nullptr);
            const uint32_t payloadAddr = ::getRegU32(&ctx, 2);
            for (uint32_t i = 0; i < 16u; ++i)
            {
                rdram[payloadAddr + i] = static_cast<uint8_t>(0x30u + i);
            }

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kStateAddr);
            ps2_stubs::sceVif1PkCloseDirectCode(rdram, &ctx, nullptr);

            uint32_t dmaTagWord = 0u;
            std::memcpy(&dmaTagWord, rdram + kBaseAddr, sizeof(dmaTagWord));
            t.Equals(dmaTagWord & 0xFFFFu, 1u, "live packet head qwc should reflect one qword before terminate");

            std::vector<std::vector<uint8_t>> captured;
            mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x30u, kBaseAddr), "write VIF1 TADR should succeed");
            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x00u, 0x145u), "write VIF1 CHCR DIR|TTE|STR|CHAIN should succeed");

            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            t.Equals(captured.size(), static_cast<size_t>(1u), "live VIF1 packet should emit one GIF packet");
            t.Equals(captured[0].size(), static_cast<size_t>(16u), "live VIF1 packet should emit one qword");

            bool payloadOk = true;
            for (uint32_t i = 0; i < 16u; ++i)
            {
                if (captured[0][i] != static_cast<uint8_t>(0x30u + i))
                {
                    payloadOk = false;
                    break;
                }
            }
            t.IsTrue(payloadOk, "live VIF1 packet payload should reach the GIF callback");
        });

        tc.Run("VIF1 DMA chain latches terminal tag bits in CHCR", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1Ch = 0x10009000u;
            constexpr uint32_t kTag0 = 0x00027400u;
            constexpr uint32_t kTag1 = kTag0 + 0x20u;

            uint8_t *rdram = mem.getRDRAM();
            writeDmaTag(rdram, kTag0, makeDmaTag(1u, 1u, 0u, false)); // CNT
            std::memset(rdram + kTag0 + 0x10u, 0, 0x10u);
            writeDmaTag(rdram, kTag1, makeDmaTag(0u, 7u, 0u, false)); // END

            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x30u, kTag0), "write VIF1 TADR should succeed");
            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x00u, 0x185u), "write VIF1 CHCR chain start should succeed");

            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            const uint32_t chcr = mem.readIORegister(kVif1Ch + 0x00u);
            t.Equals(chcr & 0x100u, 0u, "VIF1 STR should clear after DMA chain drain");
            t.Equals(chcr & 0x70000000u, 0x70000000u, "VIF1 CHCR should expose the terminal END tag id");
            t.IsTrue((mem.readIORegister(0x1000E010u) & 0x2u) != 0u, "VIF1 DMA completion should raise D_STAT channel bit");
        });

        tc.Run("VIF1 DMA source chain supports more than 4096 tags", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1Ch = 0x10009000u;
            constexpr uint32_t kTagBase = 0x00030000u;
            constexpr uint32_t kCntTagCount = 5000u;
            constexpr uint32_t kEndTag = kTagBase + kCntTagCount * 16u;

            uint8_t *rdram = mem.getRDRAM();
            for (uint32_t index = 0u; index < kCntTagCount; ++index)
            {
                writeDmaTag(rdram, kTagBase + index * 16u,
                            makeDmaTag(0u, 1u, 0u, false)); // CNT
            }
            writeDmaTag(rdram, kEndTag, makeDmaTag(1u, 7u, 0u, false)); // END
            const uint32_t markCmd = makeVifCmd(0x07u, 0u, 0x5A5Au);
            std::memcpy(rdram + kEndTag + 16u, &markCmd, sizeof(markCmd));
            std::memset(rdram + kEndTag + 20u, 0, 12u);

            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x30u, kTagBase),
                     "write long-chain VIF1 TADR should succeed");
            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x00u, 0x105u),
                     "write long-chain VIF1 CHCR should succeed");
            publishDmacCompletions(mem);

            t.Equals(mem.vif1_regs.mark, 0x5A5Au,
                     "VIF1 should consume the payload after tag 4096");
            const uint32_t chcr = mem.readIORegister(kVif1Ch + 0x00u);
            t.Equals(chcr & 0x100u, 0u,
                     "a valid long VIF1 source chain should complete");
            t.Equals(chcr & 0x70000000u, 0x70000000u,
                     "the long chain should latch its terminal END tag");
            t.IsTrue((mem.readIORegister(0x1000E010u) & 0x2u) != 0u,
                     "the long chain should raise VIF1's completion status");
        });

        tc.Run("VIF1 DMA cyclic source chain is not partially completed", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kVif1Ch = 0x10009000u;
            constexpr uint32_t kTag = 0x0002F000u;
            writeDmaTag(mem.getRDRAM(), kTag,
                        makeDmaTag(0u, 2u, kTag, false)); // NEXT to itself

            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x30u, kTag),
                     "write cyclic VIF1 TADR should succeed");
            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x00u, 0x105u),
                     "write cyclic VIF1 CHCR should be accepted");

            t.IsTrue((mem.m_ioRegisters[kVif1Ch + 0x00u] & 0x100u) != 0u,
                     "a cyclic chain must keep STR set instead of reporting false completion");
            t.IsTrue((mem.readIORegister(0x1000E010u) & 0x2u) == 0u,
                     "a cyclic chain must not raise VIF1 completion status");
            t.Equals(mem.consumePendingDmacInterrupts().size(), static_cast<size_t>(0u),
                     "a cyclic chain must not queue a completion interrupt");
        });

        tc.Run("GIF DMA chain CALL sources payload from TADR+16", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kGifCh = 0x1000A000u;
            constexpr uint32_t kTag0 = 0x00026000u;
            constexpr uint32_t kTag1 = 0x00026100u;

            uint8_t *rdram = mem.getRDRAM();

            // CALL qwc=1 addr=kTag1
            writeDmaTag(rdram, kTag0, makeDmaTag(1u, 5u, kTag1, false));
            // END qwc=1
            writeDmaTag(rdram, kTag1, makeDmaTag(1u, 7u, 0u, false));

            for (uint32_t i = 0; i < 16u; ++i)
            {
                rdram[kTag0 + 16u + i] = static_cast<uint8_t>(0x40u + i); // CALL payload
                rdram[kTag1 + 16u + i] = static_cast<uint8_t>(0x80u + i); // END payload
            }

            std::vector<std::vector<uint8_t>> captured;
            mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            t.IsTrue(mem.writeIORegister(kGifCh + 0x30u, kTag0), "write TADR should succeed");
            // STR + CHAIN mode (MOD=1)
            t.IsTrue(mem.writeIORegister(kGifCh + 0x00u, 0x104u), "write CHCR should succeed");

            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            t.Equals(captured.size(), static_cast<size_t>(1u), "chain CALL should emit one packet");
            t.Equals(captured[0].size(), static_cast<size_t>(32u), "CALL+END should emit two qwords");

            bool firstQwOk = true;
            bool secondQwOk = true;
            for (uint32_t i = 0; i < 16u; ++i)
            {
                if (captured[0][i] != static_cast<uint8_t>(0x40u + i))
                    firstQwOk = false;
                if (captured[0][16u + i] != static_cast<uint8_t>(0x80u + i))
                    secondQwOk = false;
            }
            t.IsTrue(firstQwOk, "CALL must transfer from TADR+16, not DMAtag ADDR");
            t.IsTrue(secondQwOk, "END payload should follow CALL payload");
        });

        tc.Run("GIF DMA chain RET transfers payload and resumes after CALL", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kGifCh = 0x1000A000u;
            constexpr uint32_t kTagCall = 0x00026200u;
            constexpr uint32_t kTagRet = 0x00026300u;
            constexpr uint32_t kTagEnd = 0x00026220u;

            uint8_t *rdram = mem.getRDRAM();

            // CALL qwc=1 -> jumps to RET tag
            writeDmaTag(rdram, kTagCall, makeDmaTag(1u, 5u, kTagRet, false));
            // RET qwc=1 -> should return to kTagEnd
            writeDmaTag(rdram, kTagRet, makeDmaTag(1u, 6u, 0u, false));
            // END qwc=1 after CALL payload
            writeDmaTag(rdram, kTagEnd, makeDmaTag(1u, 7u, 0u, false));

            for (uint32_t i = 0; i < 16u; ++i)
            {
                rdram[kTagCall + 16u + i] = static_cast<uint8_t>(0x11u + i); // CALL payload
                rdram[kTagRet + 16u + i] = static_cast<uint8_t>(0x22u + i);  // RET payload
                rdram[kTagEnd + 16u + i] = static_cast<uint8_t>(0x33u + i);  // END payload
            }

            std::vector<std::vector<uint8_t>> captured;
            mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            t.IsTrue(mem.writeIORegister(kGifCh + 0x30u, kTagCall), "write TADR should succeed");
            t.IsTrue(mem.writeIORegister(kGifCh + 0x00u, 0x104u), "write CHCR should succeed");

            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            t.Equals(captured.size(), static_cast<size_t>(1u), "CALL/RET chain should emit one packet");
            t.Equals(captured[0].size(), static_cast<size_t>(48u), "CALL+RET+END should emit three qwords");

            bool q0 = true;
            bool q1 = true;
            bool q2 = true;
            for (uint32_t i = 0; i < 16u; ++i)
            {
                if (captured[0][i] != static_cast<uint8_t>(0x11u + i))
                    q0 = false;
                if (captured[0][16u + i] != static_cast<uint8_t>(0x22u + i))
                    q1 = false;
                if (captured[0][32u + i] != static_cast<uint8_t>(0x33u + i))
                    q2 = false;
            }
            t.IsTrue(q0, "CALL payload should be first");
            t.IsTrue(q1, "RET must still transfer its own payload");
            t.IsTrue(q2, "RET must resume after CALL payload and continue chain");
        });

        tc.Run("GIF DMA chain IRQ stops only when TIE is set", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kGifCh = 0x1000A000u;
            constexpr uint32_t kTag0 = 0x00026400u;
            constexpr uint32_t kTag1 = 0x00026410u;
            constexpr uint32_t kRefData = 0x00026500u;

            auto runChain = [&](uint32_t chcrValue, std::vector<uint8_t> &packetOut) -> bool
            {
                uint8_t *rdram = mem.getRDRAM();
                writeDmaTag(rdram, kTag0, makeDmaTag(1u, 3u, kRefData, true)); // REF + IRQ
                writeDmaTag(rdram, kTag1, makeDmaTag(1u, 7u, 0u, false));       // END
                for (uint32_t i = 0; i < 16u; ++i)
                {
                    rdram[kRefData + i] = static_cast<uint8_t>(0x55u + i);
                    rdram[kTag1 + 16u + i] = static_cast<uint8_t>(0x77u + i);
                }

                std::vector<std::vector<uint8_t>> captured;
                mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
                {
                    captured.emplace_back(data, data + sizeBytes);
                });

                if (!mem.writeIORegister(kGifCh + 0x30u, kTag0))
                    return false;
                if (!mem.writeIORegister(kGifCh + 0x00u, chcrValue))
                    return false;
                mem.processPendingTransfers();
                publishDmacCompletions(mem);
                if (captured.empty())
                    return false;
                packetOut = captured[0];
                return true;
            };

            std::vector<uint8_t> packetNoTie;
            t.IsTrue(runChain(0x104u, packetNoTie), "chain run without TIE should succeed");
            t.Equals(packetNoTie.size(), static_cast<size_t>(32u), "IRQ tag should not stop chain when TIE is clear");

            std::vector<uint8_t> packetTie;
            // STR + CHAIN + TIE(bit7)
            t.IsTrue(runChain(0x184u, packetTie), "chain run with TIE should succeed");
            t.Equals(packetTie.size(), static_cast<size_t>(16u), "IRQ tag should stop chain when TIE is set");
        });

        tc.Run("DMAC completion service owns busy and interrupt visibility", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kGifCh = 0x1000A000u;
            constexpr uint32_t kDStat = 0x1000E010u;
            t.IsTrue(
                mem.writeIORegister(kGifCh + 0x20u, 7u),
                "GIF QWC setup should succeed");

            const DmacTransferToken transfer =
                mem.beginDmacTransfer(DmacChannel::Gif);
            t.IsTrue(
                transfer.generation != 0u,
                "DMAC start should assign a nonzero generation");
            t.IsTrue(
                mem.requestDmacCompletion(transfer),
                "finished work should queue completion");

            const DmacChannelSnapshot ready =
                mem.dmacChannelSnapshot(DmacChannel::Gif);
            t.IsTrue(
                ready.phase == DmacChannelPhase::CompletionReady,
                "finished work should remain completion-ready before service");
            t.IsTrue(
                (mem.readIORegister(kGifCh + 0x00u) & 0x100u) != 0u,
                "CHCR.STR should remain busy before completion service");
            t.Equals(
                mem.readIORegister(kGifCh + 0x20u), 7u,
                "QWC should remain visible before completion service");
            t.IsTrue(
                (mem.readIORegister(kDStat) & (1u << 2u)) == 0u,
                "D_STAT should remain clear before completion service");

            std::vector<DmacCompletionRequest> completions =
                mem.consumeReadyDmacCompletions();
            t.Equals(
                completions.size(), static_cast<size_t>(1u),
                "one finished generation should queue one completion");
            t.IsTrue(
                completions[0].transfer == transfer,
                "the completion should retain its transfer token");
            t.IsTrue(
                mem.publishDmacCompletion(completions[0], 55u),
                "the live completion should publish");
            t.IsFalse(
                mem.publishDmacCompletion(completions[0], 56u),
                "the same completion must not publish twice");

            t.IsTrue(
                (mem.readIORegister(kGifCh + 0x00u) & 0x100u) == 0u,
                "completion service should clear CHCR.STR");
            t.Equals(
                mem.readIORegister(kGifCh + 0x20u), 0u,
                "completion service should clear QWC");
            t.IsTrue(
                (mem.readIORegister(kDStat) & (1u << 2u)) != 0u,
                "completion service should latch D_STAT");

            const std::vector<DmacPendingInterrupt> interrupts =
                mem.consumePendingDmacInterrupts();
            t.Equals(
                interrupts.size(), static_cast<size_t>(1u),
                "exactly one typed interrupt should publish");
            t.IsTrue(
                interrupts[0].source == DmacChannel::Gif,
                "the interrupt should retain the GIF source");
            t.Equals(
                interrupts[0].channelGeneration,
                transfer.generation,
                "the interrupt should retain channel generation");
            t.Equals(
                interrupts[0].eventSequence, 55ull,
                "the interrupt should retain scheduler event sequence");
        });

        tc.Run("DMAC cancellation and restart reject stale completion", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            const DmacTransferToken stale =
                mem.beginDmacTransfer(DmacChannel::Vif1);
            t.IsTrue(
                mem.requestDmacCompletion(stale),
                "the first generation should become completion-ready");
            std::vector<DmacCompletionRequest> staleRequests =
                mem.consumeReadyDmacCompletions();
            t.Equals(
                staleRequests.size(), static_cast<size_t>(1u),
                "the first generation should expose one request");

            t.IsTrue(
                mem.cancelDmacTransfer(DmacChannel::Vif1),
                "stopping completion-ready work should cancel it");
            const DmacTransferToken current =
                mem.beginDmacTransfer(DmacChannel::Vif1);
            t.IsTrue(
                current.generation != stale.generation,
                "restart should assign a different generation");
            t.IsTrue(
                mem.requestDmacCompletion(current),
                "the restarted generation should become completion-ready");

            t.IsFalse(
                mem.publishDmacCompletion(
                    staleRequests[0], 70u),
                "a stopped generation must not publish after restart");
            t.IsTrue(
                (mem.readIORegister(
                     dmacChannelBase(DmacChannel::Vif1)) &
                 0x100u) != 0u,
                "stale service must not clear the restarted busy bit");

            std::vector<DmacCompletionRequest> currentRequests =
                mem.consumeReadyDmacCompletions();
            t.Equals(
                currentRequests.size(), static_cast<size_t>(1u),
                "the restarted generation should retain its request");
            t.IsTrue(
                mem.publishDmacCompletion(
                    currentRequests[0], 71u),
                "the restarted completion should publish");
            const std::vector<DmacPendingInterrupt> interrupts =
                mem.consumePendingDmacInterrupts();
            t.Equals(
                interrupts.size(), static_cast<size_t>(1u),
                "only the restarted generation should publish an interrupt");
            t.Equals(
                interrupts[0].channelGeneration,
                current.generation,
                "the published interrupt should identify the new generation");
        });

        tc.Run("equal-tick DMAC completions preserve ready order", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            const DmacTransferToken vif1 =
                mem.beginDmacTransfer(DmacChannel::Vif1);
            const DmacTransferToken gif =
                mem.beginDmacTransfer(DmacChannel::Gif);
            t.IsTrue(
                mem.requestDmacCompletion(vif1),
                "VIF1 completion should queue");
            t.IsTrue(
                mem.requestDmacCompletion(gif),
                "GIF completion should queue");
            t.Equals(
                mem.publishReadyDmacCompletions(99u),
                static_cast<size_t>(2u),
                "both equal-tick completions should publish");

            const std::vector<DmacPendingInterrupt> interrupts =
                mem.consumePendingDmacInterrupts();
            t.Equals(
                interrupts.size(), static_cast<size_t>(2u),
                "both typed interrupts should be visible");
            t.IsTrue(
                interrupts[0].source == DmacChannel::Vif1 &&
                    interrupts[1].source == DmacChannel::Gif,
                "publication should preserve deterministic ready order");
            t.IsTrue(
                interrupts[0].publicationSequence <
                    interrupts[1].publicationSequence,
                "publication sequence should encode that order");
            t.Equals(
                interrupts[0].eventSequence, 99ull,
                "equal-tick completions should retain the shared event");
            t.Equals(
                interrupts[1].eventSequence, 99ull,
                "equal-tick completions should retain the shared event");
        });

        tc.Run("DMAC D_STAT toggles masks and clears channel status on write-one", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kDStat = 0x1000E010u;
            constexpr uint32_t kGifMaskBit = (1u << 18); // channel 2 mask
            constexpr uint32_t kGifStatusBit = (1u << 2); // channel 2 status
            constexpr uint32_t kSummaryBit = (1u << 31);

            t.IsTrue(mem.writeIORegister(kDStat, kGifMaskBit), "D_STAT mask toggle write should succeed");
            t.IsTrue((mem.readIORegister(kDStat) & kGifMaskBit) != 0u, "first mask write should enable GIF mask bit");
            t.IsTrue(mem.writeIORegister(kDStat, kGifMaskBit), "D_STAT mask toggle write should succeed");
            t.IsTrue((mem.readIORegister(kDStat) & kGifMaskBit) == 0u, "second mask write should disable GIF mask bit");

            t.IsTrue(mem.writeIORegister(kDStat, kGifMaskBit), "re-enable GIF mask for summary test");

            constexpr uint32_t kGifCh = 0x1000A000u;
            constexpr uint32_t kSrc = 0x00027000u;
            uint8_t *rdram = mem.getRDRAM();
            for (uint32_t i = 0; i < 16u; ++i)
            {
                rdram[kSrc + i] = static_cast<uint8_t>(0x90u + i);
            }

            t.IsTrue(mem.writeIORegister(kGifCh + 0x10u, kSrc), "write MADR should succeed");
            t.IsTrue(mem.writeIORegister(kGifCh + 0x20u, 1u), "write QWC should succeed");
            t.IsTrue(mem.writeIORegister(kGifCh + 0x00u, 0x100u), "write CHCR STR should succeed");

            t.IsTrue((mem.readIORegister(kDStat) & kGifStatusBit) == 0u, "D_STAT status should not set before transfer drain");

            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            const uint32_t dstatAfter = mem.readIORegister(kDStat);
            t.IsTrue((dstatAfter & kGifStatusBit) != 0u, "GIF transfer completion should set D_STAT channel status bit");
            t.IsTrue((dstatAfter & kSummaryBit) != 0u, "status&mask should raise D_STAT summary bit");

            t.IsTrue(mem.writeIORegister(kDStat, kGifStatusBit), "D_STAT status clear write should succeed");
            const uint32_t dstatCleared = mem.readIORegister(kDStat);
            t.IsTrue((dstatCleared & kGifStatusBit) == 0u, "write-one should clear GIF channel status bit");
            t.IsTrue((dstatCleared & kSummaryBit) == 0u, "summary bit should clear after status clear");
        });

        tc.Run("partial D_STAT stores do not replay untouched toggle bits", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kDStat = 0x1000E010u;
            constexpr uint32_t kByteSentinel = 1u << 24u;
            constexpr uint32_t kHalfSentinel = 1u << 25u;

            t.IsTrue(mem.writeIORegister(kDStat, kByteSentinel),
                     "D_STAT byte sentinel should enable");
            mem.write8(kDStat, 0u);
            t.IsTrue((mem.readIORegister(kDStat) & kByteSentinel) != 0u,
                     "SB must not replay and toggle an unselected D_STAT mask bit");

            t.IsTrue(mem.writeIORegister(kDStat, kByteSentinel | kHalfSentinel),
                     "D_STAT sentinels should switch for the halfword check");
            mem.write16(kDStat, 0u);
            t.IsTrue((mem.readIORegister(kDStat) & kHalfSentinel) != 0u,
                     "SH must not replay and toggle an unselected D_STAT mask bit");
        });

        tc.Run("masked stores preserve disabled RAM and MMIO byte lanes", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kWordAddress = 0x00012000u;
            mem.write32(kWordAddress, 0x44332211u);
            mem.writeMasked32(kWordAddress, 0x00BBAA00u, 0x6u);
            t.Equals(mem.read32(kWordAddress), 0x44BBAA11u,
                     "masked word store should replace only enabled RAM lanes");

            constexpr uint32_t kDoubleAddress = 0x00012008u;
            mem.write64(kDoubleAddress, 0x8877665544332211ull);
            mem.writeMasked64(
                kDoubleAddress,
                0x0000DDEECCBB0000ull,
                0x3Cu);
            t.Equals(mem.read64(kDoubleAddress), 0x8877DDEECCBB2211ull,
                     "masked doubleword store should replace only enabled RAM lanes");

            constexpr uint32_t kGifMadr = 0x1000A010u;
            t.IsTrue(mem.writeIORegister(kGifMadr, 0x44332211u),
                     "GIF MADR setup should succeed");
            mem.write8(kGifMadr + 1u, 0xAAu);
            mem.write16(kGifMadr + 2u, 0xBBCCu);
            t.Equals(mem.readIORegister(kGifMadr), 0xBBCCAA11u,
                     "ordinary MMIO fields should merge only selected lanes without a guest read");

            constexpr uint32_t kDStat = 0x1000E010u;
            constexpr uint32_t kMaskSentinels =
                (1u << 16u) | (1u << 24u);
            t.IsTrue(mem.writeIORegister(kDStat, kMaskSentinels),
                     "D_STAT masked-store sentinels should enable");
            mem.writeMasked32(kDStat, 0u, 0x1u);
            mem.writeMasked32(kDStat, 0u, 0x8u);
            mem.writeMasked64(kDStat, 0u, 0x01u);
            mem.writeMasked64(kDStat, 0u, 0x80u);
            t.Equals(
                mem.readIORegister(kDStat) & kMaskSentinels,
                kMaskSentinels,
                "SWL/SWR/SDL/SDR-style byte enables must not replay D_STAT sentinels");
        });

        tc.Run("DMAC D_CTRL DMAE gates GIF DMA start", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kDctrl = 0x1000E000u;
            constexpr uint32_t kGifCh = 0x1000A000u;
            constexpr uint32_t kSrc = 0x00027800u;

            uint8_t *rdram = mem.getRDRAM();
            for (uint32_t i = 0; i < 16u; ++i)
            {
                rdram[kSrc + i] = static_cast<uint8_t>(0xE0u + i);
            }

            std::vector<std::vector<uint8_t>> captured;
            mem.setGifPacketCallback([&](const uint8_t *data, uint32_t sizeBytes)
            {
                captured.emplace_back(data, data + sizeBytes);
            });

            t.IsTrue(mem.writeIORegister(kDctrl, 0u), "clearing D_CTRL.DMAE should succeed");
            t.IsTrue(mem.writeIORegister(kGifCh + 0x10u, kSrc), "write MADR should succeed");
            t.IsTrue(mem.writeIORegister(kGifCh + 0x20u, 1u), "write QWC should succeed");
            t.IsTrue(mem.writeIORegister(kGifCh + 0x00u, 0x100u), "write CHCR STR should succeed");
            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            t.Equals(captured.size(), static_cast<size_t>(0u), "DMAE=0 should prevent GIF DMA transfer");
            t.Equals(mem.dmaStartCount(), 0ull, "DMAE=0 should not increment dmaStartCount");

            t.IsTrue(mem.writeIORegister(kDctrl, 1u), "setting D_CTRL.DMAE should succeed");
            t.IsTrue(mem.writeIORegister(kGifCh + 0x00u, 0x100u), "restarting GIF DMA should succeed");
            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            t.Equals(captured.size(), static_cast<size_t>(1u), "DMAE=1 should allow GIF DMA transfer");
            if (!captured.empty())
            {
                t.Equals(captured[0].size(), static_cast<size_t>(16u), "GIF DMA transfer should emit one qword");
            }
        });

        tc.Run("sceDmaReset re-enables DMAC DMAE", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "PS2Memory initialize should succeed");

            constexpr uint32_t kDctrl = 0x1000E000u;
            constexpr uint32_t kDpcr = 0x1000E020u;
            constexpr uint32_t kDsqwc = 0x1000E030u;
            constexpr uint32_t kDrbor = 0x1000E050u;
            constexpr uint32_t kDrbsr = 0x1000E040u;
            constexpr uint32_t kDstadr = 0x1000E060u;

            PS2Memory &mem = runtime.memory();
            t.IsTrue(mem.writeIORegister(kDctrl, 0u), "clearing D_CTRL should succeed");
            t.IsTrue(mem.writeIORegister(kDpcr, 0x12345678u), "writing D_PCR should succeed");
            t.IsTrue(mem.writeIORegister(kDsqwc, 0x11220044u), "writing D_SQWC should succeed");
            t.IsTrue(mem.writeIORegister(kDrbor, 0x2000u), "writing D_RBOR should succeed");
            t.IsTrue(mem.writeIORegister(kDrbsr, 0x3FFFu), "writing D_RBSR should succeed");
            t.IsTrue(mem.writeIORegister(kDstadr, 0x4567u), "writing D_STADR should succeed");

            R5900Context ctx{};
            ps2_stubs::sceDmaReset(mem.getRDRAM(), &ctx, &runtime);

            t.Equals(static_cast<int32_t>(::getRegU32(&ctx, 2)), 0, "sceDmaReset should return 0");
            t.Equals(mem.readIORegister(kDctrl), 1u, "sceDmaReset should leave D_CTRL DMAE enabled");
            t.Equals(mem.readIORegister(kDpcr), 0u, "sceDmaReset should clear D_PCR");
            t.Equals(mem.readIORegister(kDsqwc), 0u, "sceDmaReset should clear D_SQWC");
            t.Equals(mem.readIORegister(kDrbor), 0u, "sceDmaReset should clear D_RBOR");
            t.Equals(mem.readIORegister(kDrbsr), 0u, "sceDmaReset should clear D_RBSR");
            t.Equals(mem.readIORegister(kDstadr), 0u, "sceDmaReset should clear D_STADR");
        });

        tc.Run("sceDma state is isolated per runtime", [](TestCase &t)
        {
            PS2Runtime first;
            PS2Runtime second;
            t.IsTrue(
                first.memory().initialize(),
                "first PS2Memory initialize should succeed");
            t.IsTrue(
                second.memory().initialize(),
                "second PS2Memory initialize should succeed");

            constexpr uint32_t kFirstEnvAddr = 0x1000u;
            constexpr uint32_t kSecondEnvAddr = 0x1100u;
            constexpr uint32_t kReadbackAddr = 0x1200u;
            constexpr uint32_t kPayloadAddr = 0x2000u;
            constexpr uint32_t kGifChannel = 0x1000A000u;
            constexpr uint32_t kDctrl = 0x1000E000u;

            const std::array<uint8_t, 0x14u> firstEnv = {
                1u, 2u, 3u, 4u,
                0x11u, 0x22u, 0x33u, 0x44u,
                0x55u, 0x66u, 0x77u, 0x88u,
                0x99u, 0xAAu, 0xBBu, 0xCCu,
                0xDDu, 0xEEu, 0xF0u, 0x0Fu};
            const std::array<uint8_t, 0x14u> secondEnv = {
                2u, 1u, 2u, 3u,
                0x10u, 0x20u, 0x30u, 0x40u,
                0x50u, 0x60u, 0x70u, 0x80u,
                0x90u, 0xA0u, 0xB0u, 0xC0u,
                0xD0u, 0xE0u, 0xF0u, 0x00u};
            const std::array<uint8_t, 0x14u> resetEnv{};

            std::memcpy(
                first.memory().getRDRAM() + kFirstEnvAddr,
                firstEnv.data(),
                firstEnv.size());
            std::memcpy(
                second.memory().getRDRAM() + kSecondEnvAddr,
                secondEnv.data(),
                secondEnv.size());

            R5900Context firstCtx{};
            R5900Context secondCtx{};
            setRegU32(firstCtx, 4, kFirstEnvAddr);
            ps2_stubs::sceDmaPutEnv(
                first.memory().getRDRAM(),
                &firstCtx,
                &first);

            setRegU32(secondCtx, 4, kReadbackAddr);
            ps2_stubs::sceDmaGetEnv(
                second.memory().getRDRAM(),
                &secondCtx,
                &second);
            std::array<uint8_t, 0x14u> readback{};
            std::memcpy(
                readback.data(),
                second.memory().getRDRAM() + kReadbackAddr,
                readback.size());
            t.Equals(
                readback,
                resetEnv,
                "a fresh runtime should retain the default DMA environment");

            setRegU32(secondCtx, 4, kSecondEnvAddr);
            ps2_stubs::sceDmaPutEnv(
                second.memory().getRDRAM(),
                &secondCtx,
                &second);
            setRegU32(firstCtx, 4, kReadbackAddr);
            ps2_stubs::sceDmaGetEnv(
                first.memory().getRDRAM(),
                &firstCtx,
                &first);
            std::memcpy(
                readback.data(),
                first.memory().getRDRAM() + kReadbackAddr,
                readback.size());
            t.Equals(
                readback,
                firstEnv,
                "one runtime's DMA environment must not replace another runtime's environment");

            ps2_syscalls::notifyRuntimeStop(&second);
            setRegU32(secondCtx, 4, kReadbackAddr);
            ps2_stubs::sceDmaGetEnv(
                second.memory().getRDRAM(),
                &secondCtx,
                &second);
            std::memcpy(
                readback.data(),
                second.memory().getRDRAM() + kReadbackAddr,
                readback.size());
            t.Equals(
                readback,
                resetEnv,
                "stopping a runtime should reset its DMA environment");

            setRegU32(secondCtx, 4, kSecondEnvAddr);
            ps2_stubs::sceDmaPutEnv(
                second.memory().getRDRAM(),
                &secondCtx,
                &second);
            ps2_stubs::sceDmaReset(
                second.memory().getRDRAM(),
                &secondCtx,
                &second);
            ps2_stubs::sceDmaGetEnv(
                first.memory().getRDRAM(),
                &firstCtx,
                &first);
            std::memcpy(
                readback.data(),
                first.memory().getRDRAM() + kReadbackAddr,
                readback.size());
            t.Equals(
                readback,
                firstEnv,
                "resetting one runtime must not reset another runtime's DMA environment");

            first.memory().writeIORegister(kDctrl, 1u);
            std::memset(
                first.memory().getRDRAM() + kPayloadAddr,
                0,
                16u);
            setRegU32(firstCtx, 4, kGifChannel);
            setRegU32(firstCtx, 5, kPayloadAddr);
            setRegU32(firstCtx, 6, 1u);
            ps2_stubs::sceDmaSendN(
                first.memory().getRDRAM(),
                &firstCtx,
                &first);

            setRegU32(secondCtx, 4, kGifChannel);
            setRegU32(secondCtx, 5, 1u);
            ps2_stubs::sceDmaSync(
                second.memory().getRDRAM(),
                &secondCtx,
                &second);
            t.Equals(
                static_cast<int32_t>(
                    ::getRegU32(&secondCtx, 2)),
                0,
                "one runtime's modeled DMA completion poll must not make another runtime busy");

            ps2_stubs::sceDmaReset(
                first.memory().getRDRAM(),
                &firstCtx,
                &first);
            first.memory().writeIORegister(
                kGifChannel, 0u);
            setRegU32(firstCtx, 4, kGifChannel);
            setRegU32(firstCtx, 5, 1u);
            ps2_stubs::sceDmaSync(
                first.memory().getRDRAM(),
                &firstCtx,
                &first);
            t.Equals(
                static_cast<int32_t>(
                    ::getRegU32(&firstCtx, 2)),
                0,
                "resetting a runtime should clear its modeled DMA completion polls");
        });

        tc.Run("VIF1 DMA DIRECT image packet reaches GS through arbiter", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            GS gs;
            gs.init(mem.getGSVRAM(), static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &mem.gs());
            GifArbiter arbiter([&](const uint8_t *data, uint32_t sizeBytes)
            {
                gs.processGIFPacket(data, sizeBytes);
            });
            mem.setGifArbiter(&arbiter);

            const uint64_t bitblt =
                (static_cast<uint64_t>(0u) << 0) |
                (static_cast<uint64_t>(1u) << 16) |
                (static_cast<uint64_t>(0u) << 24) |
                (static_cast<uint64_t>(0u) << 32) |
                (static_cast<uint64_t>(1u) << 48) |
                (static_cast<uint64_t>(0u) << 56);
            gs.writeRegister(GS_REG_BITBLTBUF, bitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, (4ull << 0) | (1ull << 32));
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            constexpr uint32_t kVif1Ch = 0x10009000u;
            constexpr uint32_t kSrc = 0x00027C00u;
            constexpr uint32_t kQwc = 3u;

            uint8_t *rdram = mem.getRDRAM();
            std::memset(rdram + kSrc, 0, kQwc * 16u);

            const uint32_t directCmd = makeVifCmd(0x50u, 0u, 2u); // DIRECT 2 QW payload.
            std::memcpy(rdram + kSrc, &directCmd, sizeof(directCmd));

            uint8_t *gifPayload = rdram + kSrc + 4u;
            const uint64_t gifTag = makeGifTag(1u, GIF_FMT_IMAGE, 0u, true);
            std::memcpy(gifPayload + 0u, &gifTag, sizeof(gifTag));
            const uint64_t tagHi = 0u;
            std::memcpy(gifPayload + 8u, &tagHi, sizeof(tagHi));
            for (uint32_t i = 0; i < 16u; ++i)
            {
                gifPayload[16u + i] = static_cast<uint8_t>(0x70u + i);
            }

            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x10u, kSrc), "write VIF1 MADR should succeed");
            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x20u, kQwc), "write VIF1 QWC should succeed");
            t.IsTrue(mem.writeIORegister(kVif1Ch + 0x00u, 0x100u), "write VIF1 CHCR STR should succeed");

            mem.processPendingTransfers();
            publishDmacCompletions(mem);

            const uint8_t *vramOut = mem.getGSVRAM();
            bool imageOk = true;
            for (uint32_t x = 0; x < 4u && imageOk; ++x)
            {
                const uint32_t off = GSPSMCT32::addrPSMCT32(0u, 1u, x, 0u);
                for (uint32_t c = 0; c < 4u; ++c)
                {
                    if (vramOut[off + c] != static_cast<uint8_t>(0x70u + x * 4u + c))
                    {
                        imageOk = false;
                        break;
                    }
                }
            }
            t.IsTrue(imageOk, "VIF1 DIRECT image should update GS VRAM through GIF path2");
        });

        tc.Run("VIF1 DIRECT payload can continue across interpreter calls", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            GS gs;
            gs.init(mem.getGSVRAM(), static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &mem.gs());
            GifArbiter arbiter([&](const uint8_t *data, uint32_t sizeBytes)
            {
                gs.processGIFPacket(data, sizeBytes);
            });
            mem.setGifArbiter(&arbiter);

            const uint64_t bitblt =
                (static_cast<uint64_t>(0u) << 0) |
                (static_cast<uint64_t>(1u) << 16) |
                (static_cast<uint64_t>(0u) << 24) |
                (static_cast<uint64_t>(0u) << 32) |
                (static_cast<uint64_t>(1u) << 48) |
                (static_cast<uint64_t>(0u) << 56);
            gs.writeRegister(GS_REG_BITBLTBUF, bitblt);
            gs.writeRegister(GS_REG_TRXPOS, 0ull);
            gs.writeRegister(GS_REG_TRXREG, (4ull << 0) | (1ull << 32));
            gs.writeRegister(GS_REG_TRXDIR, 0ull);

            std::vector<uint8_t> packet;
            appendU32(packet, makeVifCmd(0x50u, 0u, 2u)); // DIRECT 2 QW payload split across calls.
            appendU64(packet, makeGifTag(1u, GIF_FMT_IMAGE, 0u, true));
            appendU64(packet, 0ull);
            for (uint32_t i = 0; i < 16u; ++i)
            {
                packet.push_back(static_cast<uint8_t>(0xA0u + i));
            }

            mem.processVIF1Data(packet.data(), 20u); // command plus GIF IMAGE tag
            mem.processVIF1Data(packet.data() + 20u, 16u); // remaining image qword

            const uint8_t *vramOut = mem.getGSVRAM();
            bool imageOk = true;
            for (uint32_t x = 0; x < 4u && imageOk; ++x)
            {
                const uint32_t off = GSPSMCT32::addrPSMCT32(0u, 1u, x, 0u);
                for (uint32_t c = 0; c < 4u; ++c)
                {
                    if (vramOut[off + c] != static_cast<uint8_t>(0xA0u + x * 4u + c))
                    {
                        imageOk = false;
                        break;
                    }
                }
            }
            t.IsTrue(imageOk, "continued DIRECT payload should complete the PATH2 image upload");
        });

        tc.Run("VIF0 retained DMA accounts the PCSX2 VU-wait prefix and tail", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(),
                     "VIF0 wait fixture memory should initialize");

            constexpr uint32_t kVif0 = 0x10008000u;
            constexpr uint32_t kVif0Stat = 0x10003800u;
            constexpr uint32_t kVif0Code = 0x10003880u;
            constexpr uint32_t kDstat = 0x1000E010u;
            constexpr uint32_t kTag = 0x00036000u;
            constexpr uint32_t kPayload = kTag + 16u;
            constexpr uint32_t kMpg =
                (0x4Au << 24u) | (42u << 16u);
            constexpr uint32_t kMscal = 0x14000000u;
            constexpr uint32_t kFlushe = 0x10000000u;

            uint8_t *const rdram = mem.getRDRAM();
            std::memset(rdram + kTag, 0, 16u + 22u * 16u);
            const uint64_t end =
                makeDmaTag(22u, 7u, 0u, false);
            std::memcpy(rdram + kTag, &end, sizeof(end));

            std::vector<uint8_t> payload;
            appendU32(payload, kMpg);
            payload.resize(payload.size() + 42u * 8u, 0u);
            appendU32(payload, kMscal);
            appendU32(payload, kFlushe);
            appendU32(payload, 0u);
            t.Equals(payload.size(), static_cast<size_t>(22u * 16u),
                     "reference VIF0 payload should be 22 QWC");
            std::memcpy(rdram + kPayload,
                        payload.data(), payload.size());

            bool vuBusy = false;
            uint32_t mscalCalls = 0u;
            uint32_t mscalPc = UINT32_MAX;
            uint32_t mscalItop = UINT32_MAX;
            uint32_t initialDelay = 0u;
            mem.setVu0BusyCallback([&]()
                                   { return vuBusy; });
            mem.setVu0MscalCallback(
                [&](uint32_t startPc, uint32_t itop)
                {
                    ++mscalCalls;
                    mscalPc = startPc;
                    mscalItop = itop;
                    vuBusy = true;
                });
            mem.setVif0DmaScheduleCallback(
                [&](uint32_t delayEeCycles)
                {
                    initialDelay = delayEeCycles;
                    return true;
                });

            t.IsTrue(mem.writeIORegister(
                         kVif0 + 0x30u, kTag),
                     "VIF0 reference TADR write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif0, 0x145u),
                     "VIF0 reference chain start should succeed");
            t.Equals(initialDelay, 4u,
                     "VIF0 submission should schedule after four EE cycles");

            Vif0DmaAdvanceResult advance =
                mem.advanceVif0Dma();
            t.IsTrue(advance.progressed,
                     "first event should fetch the END tag");
            t.Equals(advance.delayEeCycles, 2u,
                     "tag plus two TTE words should cost two cycles");
            Vif0DmaSnapshot snapshot =
                mem.vif0DmaSnapshot();
            t.IsTrue(
                snapshot.phase ==
                    Vif0DmaPhase::TransferPayload,
                "tag setup should expose the payload phase");
            t.Equals(snapshot.qwc, 22u,
                     "tag setup should publish payload QWC");
            t.Equals(
                mem.readIORegister(kVif0 + 0x10u),
                kPayload,
                "END setup should publish payload MADR");

            advance = mem.advanceVif0Dma();
            snapshot = mem.vif0DmaSnapshot();
            t.IsTrue(
                advance.progressed &&
                    advance.stall ==
                        Vif0DmaStallReason::WaitingForVu,
                "payload prefix should stop at FLUSHE");
            t.Equals(advance.acceptedBytes, 344u,
                     "MPG plus MSCAL should accept 86 words");
            t.Equals(advance.delayEeCycles, 43u,
                     "86 accepted words should cost 43 EE cycles");
            t.Equals(snapshot.qwc, 1u,
                     "partial payload should retain one QWC");
            t.Equals(snapshot.payloadByteOffset, 8u,
                     "partial payload should retain a two-word QW offset");
            t.Equals(snapshot.bufferedPayloadBytes, 8u,
                     "FLUSHE and trailing NOP should remain buffered");
            t.Equals(mscalCalls, 1u,
                     "MSCAL should start VU0 exactly once");
            t.Equals(mscalPc, 0u,
                     "MSCAL immediate zero should start at PC zero");
            t.Equals(mscalItop, 0u,
                     "MSCAL should latch the pending VIF0 ITOP");
            t.IsTrue(mem.vif0WaitingForVu(),
                     "VIF0 should expose the VU-owned wait");
            t.IsTrue(
                (mem.readIORegister(kVif0Stat) &
                 (1u << 2u)) != 0u,
                "guest-visible VIF0 STAT should expose VEW");
            t.Equals(mem.readIORegister(kVif0Code), kFlushe,
                     "guest-visible VIF0 CODE should retain FLUSHE");

            advance = mem.advanceVif0Dma();
            t.IsFalse(advance.progressed,
                      "the accounted payload callback should only hand off");
            t.IsTrue(
                advance.stall ==
                    Vif0DmaStallReason::WaitingForVu,
                "the later callback should retain the VU stall");

            vuBusy = false;
            t.IsTrue(mem.resumeVIF0AfterVu(),
                     "VU completion should clear the VIF wait");
            advance = mem.advanceVif0Dma();
            snapshot = mem.vif0DmaSnapshot();
            t.Equals(advance.acceptedBytes, 8u,
                     "finish wake should accept the retained tail");
            t.Equals(advance.delayEeCycles, 2u,
                     "the retained QW tail should cost two cycles");
            t.IsTrue(
                snapshot.phase == Vif0DmaPhase::Finalize,
                "tail completion should leave finalization pending");
            t.Equals(snapshot.qwc, 0u,
                     "tail completion should consume the final QWC");
            t.Equals(snapshot.payloadByteOffset, 0u,
                     "tail completion should clear the QW offset");
            t.Equals(snapshot.bufferedPayloadBytes, 0u,
                     "tail completion should drain copied payload bytes");
            t.IsTrue(
                (mem.readIORegister(kVif0Stat) &
                 (1u << 2u)) == 0u,
                "VU finish should clear guest-visible VEW");

            advance = mem.advanceVif0Dma();
            t.IsTrue(advance.completed,
                     "a distinct final callback should request completion");
            t.IsTrue(
                (mem.readIORegister(kVif0) & 0x100u) != 0u,
                "requesting completion should keep STR visible");
            t.IsTrue(
                (mem.readIORegister(kDstat) & 0x1u) == 0u,
                "requesting completion should keep D_STAT clear");
            t.Equals(publishDmacCompletions(mem),
                     static_cast<size_t>(1u),
                     "typed publication should publish one VIF0 cause");
            t.IsTrue(
                (mem.readIORegister(kVif0) & 0x100u) == 0u,
                "typed publication should clear VIF0 STR");
            t.IsTrue(
                (mem.readIORegister(kDstat) & 0x1u) != 0u,
                "typed publication should latch channel-zero D_STAT");
        });

        tc.Run("VIF0 retained parser completes a fixed command across TTE and payload", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(),
                     "VIF0 split fixture memory should initialize");

            constexpr uint32_t kVif0 = 0x10008000u;
            constexpr uint32_t kTag = 0x00036400u;
            uint8_t *const rdram = mem.getRDRAM();
            std::memset(rdram + kTag, 0, 48u);

            const uint64_t cnt =
                makeDmaTag(0u, 1u, 0u, false);
            const uint64_t end =
                makeDmaTag(1u, 7u, 0u, false);
            std::memcpy(rdram + kTag, &cnt, sizeof(cnt));
            std::memcpy(rdram + kTag + 16u,
                        &end, sizeof(end));

            const uint32_t strow =
                makeVifCmd(0x30u, 0u, 0u);
            const std::array<uint32_t, 4> row = {
                0x11111111u,
                0x22222222u,
                0x33333333u,
                0x44444444u,
            };
            std::memcpy(rdram + kTag + 8u,
                        &strow, sizeof(strow));
            std::memcpy(rdram + kTag + 12u,
                        &row[0], sizeof(row[0]));
            std::memcpy(rdram + kTag + 24u,
                        &row[1], sizeof(row[1]) * 2u);
            std::memcpy(rdram + kTag + 32u,
                        &row[3], sizeof(row[3]));

            mem.setVif0DmaScheduleCallback(
                [](uint32_t)
                {
                    return true;
                });
            t.IsTrue(mem.writeIORegister(
                         kVif0 + 0x30u, kTag),
                     "split VIF0 TADR write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif0, 0x145u),
                     "split VIF0 chain start should succeed");

            Vif0DmaAdvanceResult advance =
                mem.advanceVif0Dma();
            t.IsTrue(advance.progressed,
                     "first callback should fetch CNT");
            t.Equals(mem.vif0DmaSnapshot().parserBytes, 8u,
                     "STROW header and first word should remain buffered");
            t.Equals(mem.vif0_regs.row[0], 0u,
                     "incomplete STROW must remain atomic");

            advance = mem.advanceVif0Dma();
            t.IsTrue(advance.progressed,
                     "second callback should fetch END");
            t.Equals(mem.vif0DmaSnapshot().parserBytes, 16u,
                     "second TTE should extend the retained STROW");
            t.Equals(mem.vif0_regs.row[0], 0u,
                     "STROW should still wait for its final word");

            advance = mem.advanceVif0Dma();
            t.IsTrue(advance.progressed,
                     "third callback should transfer END payload");
            for (size_t index = 0u; index < row.size(); ++index)
            {
                t.Equals(mem.vif0_regs.row[index], row[index],
                         "split VIF0 STROW should commit atomically");
            }
            t.Equals(mem.vif0DmaSnapshot().parserBytes, 0u,
                     "complete STROW and NOP padding should drain");
        });

        tc.Run("VIF0 wait at the first payload word hands off without a phantom DMA cycle", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(),
                     "VIF0 zero-prefix fixture memory should initialize");

            constexpr uint32_t kVif0 = 0x10008000u;
            constexpr uint32_t kPayload = 0x00036700u;
            constexpr uint32_t kFlushe = 0x10000000u;
            std::memset(mem.getRDRAM() + kPayload, 0, 16u);
            std::memcpy(mem.getRDRAM() + kPayload,
                        &kFlushe, sizeof(kFlushe));

            bool vuBusy = true;
            mem.setVu0BusyCallback([&]()
                                   { return vuBusy; });
            mem.setVif0DmaScheduleCallback(
                [](uint32_t)
                {
                    return true;
                });
            t.IsTrue(mem.writeIORegister(
                         kVif0 + 0x10u, kPayload),
                     "zero-prefix MADR write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif0 + 0x20u, 1u),
                     "zero-prefix QWC write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif0, 0x100u),
                     "zero-prefix VIF0 DMA should start");

            Vif0DmaAdvanceResult advance =
                mem.advanceVif0Dma();
            t.IsFalse(advance.progressed,
                      "an immediate FLUSHE wait must not invent transfer work");
            t.Equals(advance.acceptedBytes, 0u,
                     "an immediate wait should accept no payload bytes");
            t.Equals(advance.delayEeCycles, 0u,
                     "a no-progress result should carry no DMA delay");
            t.IsTrue(
                advance.stall ==
                    Vif0DmaStallReason::WaitingForVu,
                "an immediate wait should hand ownership to VU completion");
            t.Equals(mem.vif0DmaSnapshot().qwc, 1u,
                     "an immediate wait should retain the complete QW");

            vuBusy = false;
            t.IsTrue(mem.resumeVIF0AfterVu(),
                     "VU completion should release the immediate wait");
            advance = mem.advanceVif0Dma();
            t.IsTrue(advance.progressed,
                     "the released QW should then make progress");
            t.Equals(advance.acceptedBytes, 16u,
                     "the released QW should transfer exactly once");
            t.Equals(advance.delayEeCycles, 2u,
                     "four released words should cost two cycles");
        });

        tc.Run("VIF0 reset cancels a generation and restart completes only its replacement", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(),
                     "VIF0 reset fixture memory should initialize");

            constexpr uint32_t kVif0 = 0x10008000u;
            constexpr uint32_t kFbrst = 0x10003810u;
            constexpr uint32_t kFirst = 0x00036800u;
            constexpr uint32_t kSecond = 0x00036900u;
            std::memset(mem.getRDRAM() + kFirst, 0, 16u);
            std::memset(mem.getRDRAM() + kSecond, 0, 16u);

            uint32_t schedules = 0u;
            uint32_t cancels = 0u;
            uint32_t resets = 0u;
            mem.setVif0DmaScheduleCallback(
                [&](uint32_t)
                {
                    ++schedules;
                    return true;
                });
            mem.setVif0DmaCancelCallback(
                [&]()
                {
                    ++cancels;
                });
            mem.setVif0ResetCallback(
                [&]()
                {
                    ++resets;
                });

            t.IsTrue(mem.writeIORegister(
                         kVif0 + 0x10u, kFirst),
                     "first VIF0 MADR write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif0 + 0x20u, 1u),
                     "first VIF0 QWC write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif0, 0x100u),
                     "first VIF0 generation should start");
            const uint64_t staleGeneration =
                mem.vif0DmaSnapshot().transfer.generation;

            t.IsTrue(mem.writeIORegister(kFbrst, 1u),
                     "VIF0 FBRST.RST should succeed");
            t.IsFalse(mem.vif0DmaSnapshot().active,
                      "reset should retire the descriptor");
            t.IsTrue(
                (mem.readIORegister(kVif0) & 0x100u) == 0u,
                "reset should clear channel STR");
            t.Equals(mem.readIORegister(0x10003800u), 0u,
                     "reset should clear guest-visible VIF0 state");
            t.Equals(cancels, 1u,
                     "reset should cancel event ownership once");
            t.Equals(resets, 1u,
                     "reset should invalidate the VU-finish owner once");

            t.IsTrue(mem.writeIORegister(
                         kVif0 + 0x10u, kSecond),
                     "replacement VIF0 MADR write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif0 + 0x20u, 1u),
                     "replacement VIF0 QWC write should succeed");
            t.IsTrue(mem.writeIORegister(
                         kVif0, 0x100u),
                     "replacement VIF0 generation should start");
            const Vif0DmaSnapshot replacement =
                mem.vif0DmaSnapshot();
            t.IsTrue(
                replacement.transfer.generation !=
                    staleGeneration,
                "restart should own a fresh DMAC generation");
            t.Equals(schedules, 2u,
                     "both generations should request startup events");

            Vif0DmaAdvanceResult advance =
                mem.advanceVif0Dma();
            t.IsTrue(
                advance.progressed &&
                    advance.phase ==
                        Vif0DmaPhase::Finalize,
                "replacement payload should reach finalization");
            advance = mem.advanceVif0Dma();
            t.IsTrue(advance.completed,
                     "replacement finalization should request completion");
            t.Equals(publishDmacCompletions(mem),
                     static_cast<size_t>(1u),
                     "only the replacement should publish");
        });

        tc.Run("unaligned accesses throw", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            bool threwRead32 = false;
            bool threwWrite64 = false;
            try
            {
                (void)mem.read32(0x00000002u);
            }
            catch (const std::exception &)
            {
                threwRead32 = true;
            }

            try
            {
                mem.write64(0x00000004u + 2u, 0x1122334455667788ull);
            }
            catch (const std::exception &)
            {
                threwWrite64 = true;
            }

            t.IsTrue(threwRead32, "unaligned read32 should throw");
            t.IsTrue(threwWrite64, "unaligned write64 should throw");
        });

        tc.Run("code invalidations aggregate by physical range and writer", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");
            mem.registerCodeRegion(0x1000u, 0x1100u);

            constexpr uint32_t firstWriter = 0x00123450u;
            constexpr uint32_t secondWriter = 0x006789a0u;
            mem.write8(0x1001u, 0x12u, firstWriter);
            mem.write16(0x1002u, 0x3456u, firstWriter);
            mem.write32(0x80001004u, 0x789abcdeu, firstWriter);
            mem.write64(0x1008u, 0x0123456789abcdefull, firstWriter);
            mem.write128(0x1010u, _mm_setzero_si128(), firstWriter);
            mem.write32(0x1020u, 0xfedcba98u, secondWriter);

            const auto events = mem.takeCodeInvalidationEvents();
            t.Equals(events.size(), static_cast<size_t>(2),
                     "contiguous writes from one guest PC should form one event");
            if (events.size() == 2u)
            {
                t.Equals(events[0].start, 0x1000u,
                         "first invalidation should use the physical code address");
                t.Equals(events[0].end, 0x1020u,
                         "first invalidation should end after the last contiguous word");
                t.Equals(events[0].words, 8u,
                         "overlapping byte and halfword writes should count a code word once");
                t.Equals(events[0].writerPc, firstWriter,
                         "first invalidation should retain its writer PC");
                t.Equals(events[1].start, 0x1020u,
                         "a different writer should begin a separate event");
                t.Equals(events[1].end, 0x1024u,
                         "the second event should cover one code word");
                t.Equals(events[1].words, 1u,
                         "the second event should count one code word");
                t.Equals(events[1].writerPc, secondWriter,
                         "second invalidation should retain its writer PC");
            }
            t.IsTrue(mem.takeCodeInvalidationEvents().empty(),
                     "consuming invalidations should leave no duplicate events");
            t.IsTrue(mem.isCodeModified(0x1000u, 0x24u),
                     "all invalidated code words should remain marked modified");
        });
    });
}
