#include "MiniTest.h"

#include "ps2_runtime.h"
#include "runtime/ee_cache.h"
#include "runtime/ps2_memory.h"

#include <array>
#include <cstdint>

namespace
{
    constexpr uint32_t kCop0CauseExceptionCodeMask =
        0x0000007Cu;

    bool execute(
        EeCache &cache,
        PS2Memory &memory,
        uint32_t operation,
        uint32_t address,
        uint32_t &tagLo,
        uint32_t &tagHi)
    {
        return cache.execute(
            memory,
            operation,
            address,
            tagLo,
            tagHi,
            0x00100000u);
    }
}

void register_ee_cache_tests()
{
    MiniTest::Case("EeCache", [](TestCase &tc)
    {
        tc.Run("data tag store and load reproduce the PCSX2 oracle", [](TestCase &t)
        {
            PS2Memory memory;
            EeCache cache;
            uint32_t tagLo = 0x00000038u;
            uint32_t tagHi = 0xA5A5A5A5u;
            constexpr uint32_t address = 0x80000400u;

            t.IsTrue(
                execute(
                    cache, memory, 0x12u, address,
                    tagLo, tagHi),
                "DXSTG should be implemented");
            tagLo = 0u;
            t.IsTrue(
                execute(
                    cache, memory, 0x10u, address,
                    tagLo, tagHi),
                "DXLTG should be implemented");
            t.Equals(
                tagLo, 0x00000038u,
                "DXLTG should restore the tag fields observed in strict PCSX2");
            t.Equals(
                tagHi, 0xA5A5A5A5u,
                "data-cache tag operations should not modify TagHi");

            tagLo = 0x00000040u;
            (void)execute(
                cache, memory, 0x12u, address,
                tagLo, tagHi);
            tagLo = 0u;
            (void)execute(
                cache, memory, 0x10u, address,
                tagLo, tagHi);
            t.Equals(
                tagLo, 0x00000040u,
                "DXSTG should retain diagnostic Dirty independently of Valid");
        });

        tc.Run("data index writeback publishes line data and invalidates", [](TestCase &t)
        {
            PS2Memory memory;
            t.IsTrue(
                memory.initialize(),
                "the writeback fixture should allocate RDRAM");
            EeCache cache;
            constexpr uint32_t address = 0x80000400u;
            constexpr uint32_t physicalBase = 0x00012400u;
            uint32_t tagLo = 0x00012070u;
            uint32_t tagHi = 0u;

            t.IsTrue(
                execute(
                    cache, memory, 0x12u, address,
                    tagLo, tagHi),
                "DXSTG should install a dirty valid tag");
            for (uint32_t word = 0u; word < 16u; ++word)
            {
                tagLo = 0xA5000000u | word;
                t.IsTrue(
                    execute(
                        cache, memory, 0x13u,
                        address + word * sizeof(uint32_t),
                        tagLo, tagHi),
                    "DXSDT should store every diagnostic word");
            }

            tagLo = 0u;
            t.IsTrue(
                execute(
                    cache, memory, 0x11u,
                    address + 7u * sizeof(uint32_t),
                    tagLo, tagHi),
                "DXLDT should load diagnostic data");
            t.Equals(
                tagLo, 0xA5000007u,
                "DXLDT should address the selected word within the line");
            t.Equals(
                memory.read32(physicalBase), 0u,
                "dirty diagnostic data should remain private before writeback");

            t.IsTrue(
                execute(
                    cache, memory, 0x14u, address,
                    tagLo, tagHi),
                "DXWBIN should write back and invalidate");
            for (uint32_t word = 0u; word < 16u; ++word)
            {
                t.Equals(
                    memory.read32(
                        physicalBase + word * sizeof(uint32_t)),
                    0xA5000000u | word,
                    "DXWBIN should publish the complete cache line");
            }

            tagLo = 0u;
            (void)execute(
                cache, memory, 0x10u, address,
                tagLo, tagHi);
            t.Equals(
                tagLo, 0x00012010u,
                "writeback-invalidate should preserve PFN/LRF and clear L/V/D");
        });

        tc.Run("data hit operations compare physical tags", [](TestCase &t)
        {
            PS2Memory memory;
            t.IsTrue(
                memory.initialize(),
                "the hit fixture should allocate RDRAM");
            EeCache cache;
            constexpr uint32_t indexAddress = 0x80000400u;
            constexpr uint32_t hitAddress = 0x80012400u;
            uint32_t tagLo = 0x00012060u;
            uint32_t tagHi = 0u;

            (void)execute(
                cache, memory, 0x12u, indexAddress,
                tagLo, tagHi);
            tagLo = 0xC001CAFEu;
            (void)execute(
                cache, memory, 0x13u, indexAddress,
                tagLo, tagHi);
            t.IsTrue(
                execute(
                    cache, memory, 0x1Cu, hitAddress,
                    tagLo, tagHi),
                "DHWOIN should be implemented");
            t.Equals(
                memory.read32(0x00012400u),
                0xC001CAFEu,
                "a matching hit should write dirty data to its physical line");

            tagLo = 0u;
            (void)execute(
                cache, memory, 0x10u, indexAddress,
                tagLo, tagHi);
            t.Equals(
                tagLo, 0x00012020u,
                "DHWOIN should retain Valid while clearing Dirty");

            t.IsTrue(
                execute(
                    cache, memory, 0x1Au, hitAddress,
                    tagLo, tagHi),
                "DHIN should be implemented");
            tagLo = 0u;
            (void)execute(
                cache, memory, 0x10u, indexAddress,
                tagLo, tagHi);
            t.Equals(
                tagLo, 0x00012000u,
                "DHIN should invalidate a physically matching line");
        });

        tc.Run("instruction diagnostics and BTAC retain visible state", [](TestCase &t)
        {
            PS2Memory memory;
            EeCache cache;
            constexpr uint32_t instructionAddress = 0x8000080Du;
            uint32_t tagLo = 0x00123030u;
            uint32_t tagHi = 0u;

            (void)execute(
                cache, memory, 0x04u,
                instructionAddress, tagLo, tagHi);
            tagLo = 0u;
            (void)execute(
                cache, memory, 0x00u,
                instructionAddress, tagLo, tagHi);
            t.Equals(
                tagLo, 0x00123030u,
                "IXLTG should restore PFN, LRF, and Valid");

            tagLo = 0xDEADBEEFu;
            tagHi = 0x0000002Du;
            (void)execute(
                cache, memory, 0x05u,
                instructionAddress, tagLo, tagHi);
            tagLo = 0u;
            tagHi = 0u;
            (void)execute(
                cache, memory, 0x01u,
                instructionAddress, tagLo, tagHi);
            t.Equals(
                tagLo, 0xDEADBEEFu,
                "IXLDT should restore the selected instruction word");
            t.Equals(
                tagHi, 0x0000002Du,
                "IXLDT should restore steering and BHT metadata");

            constexpr uint32_t btacIndex = 7u;
            tagLo = 0x12345679u;
            tagHi = 0x89ABCDECu;
            (void)execute(
                cache, memory, 0x06u,
                btacIndex, tagLo, tagHi);
            tagLo = 0u;
            tagHi = 0u;
            (void)execute(
                cache, memory, 0x02u,
                btacIndex, tagLo, tagHi);
            t.Equals(
                tagLo, 0x12345679u,
                "BXLBT should restore fetch address and Valid");
            t.Equals(
                tagHi, 0x89ABCDECu,
                "BXLBT should restore the target address");

            (void)execute(
                cache, memory, 0x0Au,
                0x12345678u, tagLo, tagHi);
            (void)execute(
                cache, memory, 0x02u,
                btacIndex, tagLo, tagHi);
            t.Equals(
                tagLo & 1u, 0u,
                "BHINBT should invalidate a matching fetch address");

            (void)execute(
                cache, memory, 0x07u,
                instructionAddress, tagLo, tagHi);
            tagLo = 0u;
            (void)execute(
                cache, memory, 0x00u,
                instructionAddress, tagLo, tagHi);
            t.Equals(
                tagLo, 0x00123010u,
                "IXIN should preserve PFN/LRF while clearing Valid");
        });

        tc.Run("instruction fill and hit invalidation use translated memory", [](TestCase &t)
        {
            PS2Memory memory;
            t.IsTrue(
                memory.initialize(),
                "the instruction-fill fixture should allocate RDRAM");
            EeCache cache;
            constexpr uint32_t address = 0x80034080u;
            constexpr uint32_t physicalBase = 0x00034080u;
            for (uint32_t word = 0u; word < 16u; ++word)
            {
                memory.write32(
                    physicalBase + word * sizeof(uint32_t),
                    0x0F000000u | word);
            }

            uint32_t tagLo = 0u;
            uint32_t tagHi = 0u;
            t.IsTrue(
                execute(
                    cache, memory, 0x0Eu,
                    address, tagLo, tagHi),
                "IFL should be implemented");
            (void)execute(
                cache, memory, 0x01u,
                address + 9u * sizeof(uint32_t),
                tagLo, tagHi);
            t.Equals(
                tagLo, 0x0F000009u,
                "IFL should copy the complete physical line");

            tagLo = 0u;
            (void)execute(
                cache, memory, 0x00u,
                address, tagLo, tagHi);
            t.Equals(
                tagLo, 0x00034030u,
                "IFL should publish PFN, Valid, and the flipped LRF bit");

            t.IsTrue(
                execute(
                    cache, memory, 0x0Bu,
                    address, tagLo, tagHi),
                "IHIN should be implemented");
            tagLo = 0u;
            (void)execute(
                cache, memory, 0x00u,
                address, tagLo, tagHi);
            t.Equals(
                tagLo, 0x00034010u,
                "IHIN should invalidate a physically matching line");
        });

        tc.Run("all documented operations are explicit and reserved encodings reject", [](TestCase &t)
        {
            PS2Memory memory;
            t.IsTrue(
                memory.initialize(),
                "the operation census should allocate RDRAM");
            EeCache cache;
            uint32_t tagLo = 0u;
            uint32_t tagHi = 0u;
            constexpr std::array<uint32_t, 20u> documented{
                0x00u, 0x01u, 0x02u, 0x04u, 0x05u,
                0x06u, 0x07u, 0x0Au, 0x0Bu, 0x0Cu,
                0x0Eu, 0x10u, 0x11u, 0x12u, 0x13u,
                0x14u, 0x16u, 0x18u, 0x1Au, 0x1Cu,
            };
            for (uint32_t operation : documented)
            {
                t.IsTrue(
                    execute(
                        cache, memory, operation,
                        0x80000000u, tagLo, tagHi),
                    "every documented CACHE operation should have an explicit path");
            }

            constexpr std::array<uint32_t, 12u> reserved{
                0x03u, 0x08u, 0x09u, 0x0Du,
                0x0Fu, 0x15u, 0x17u, 0x19u,
                0x1Bu, 0x1Du, 0x1Eu, 0x1Fu,
            };
            for (uint32_t operation : reserved)
            {
                t.IsFalse(
                    execute(
                        cache, memory, operation,
                        0x80000000u, tagLo, tagHi),
                    "reserved CACHE encodings should not silently execute");
            }
        });

        tc.Run("runtime reports reserved operations and translation faults", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context reservedContext{};
            reservedContext.pc = 0x00100000u;
            bool reservedRaised = false;
            try
            {
                runtime.handleEeCacheOperation(
                    nullptr, &reservedContext,
                    0x03u, 0x80000000u);
            }
            catch (const PS2GuestException &)
            {
                reservedRaised = true;
            }

            t.IsTrue(
                reservedRaised,
                "a reserved CACHE operation should unwind guest execution");
            t.Equals(
                reservedContext.cop0_cause &
                    kCop0CauseExceptionCodeMask,
                (static_cast<uint32_t>(
                     EXCEPTION_RESERVED_INSTRUCTION) << 2u) &
                    kCop0CauseExceptionCodeMask,
                "a reserved CACHE operation should raise Reserved Instruction");

            R5900Context tlbContext{};
            tlbContext.pc = 0x00100004u;
            bool tlbRaised = false;
            try
            {
                runtime.handleEeCacheOperation(
                    nullptr, &tlbContext,
                    0x0Eu, 0xC0000000u);
            }
            catch (const PS2GuestException &)
            {
                tlbRaised = true;
            }

            t.IsTrue(
                tlbRaised,
                "a CACHE address translation fault should unwind guest execution");
            t.Equals(
                tlbContext.cop0_badvaddr,
                0xC0000000u,
                "CACHE should publish the faulting virtual address");
            t.Equals(
                tlbContext.cop0_cause &
                    kCop0CauseExceptionCodeMask,
                (static_cast<uint32_t>(
                     EXCEPTION_TLB_REFILL_LOAD) << 2u) &
                    kCop0CauseExceptionCodeMask,
                "CACHE translation faults should use the load-refill exception");
        });
    });
}
