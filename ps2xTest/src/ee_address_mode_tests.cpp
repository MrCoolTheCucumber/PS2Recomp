#include "MiniTest.h"

#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

namespace
{
    constexpr uint32_t kStatusSupervisor = 0x00000008u;
    constexpr uint32_t kStatusUser = 0x00000010u;
    constexpr uint32_t kStatusExl = 0x00000002u;
    constexpr uint32_t kStatusErl = 0x00000004u;
    constexpr uint32_t kCauseExcCodeMask = 0x0000007Cu;
    constexpr uint32_t kCauseBd = 0x80000000u;
    constexpr uint32_t kCommonVector = 0x80000180u;

    constexpr uint32_t makeEntryLo(
        uint32_t pfn,
        bool dirty,
        bool valid,
        bool global)
    {
        return (pfn << 6u) |
               (2u << 3u) |
               (dirty ? 0x4u : 0u) |
               (valid ? 0x2u : 0u) |
               (global ? 0x1u : 0u);
    }

    bool raisesGuestException(const auto &operation)
    {
        try
        {
            operation();
        }
        catch (const PS2GuestException &)
        {
            return true;
        }
        return false;
    }

    void expectAddressError(
        TestCase &t,
        const R5900Context &ctx,
        bool raised,
        PS2Exception exception,
        uint32_t badVirtualAddress,
        uint32_t expectedEpc,
        bool expectedBd,
        const std::string &prefix)
    {
        t.IsTrue(
            raised,
            prefix + "access should unwind guest execution");
        t.Equals(
            ctx.cop0_cause & kCauseExcCodeMask,
            (static_cast<uint32_t>(exception) << 2u) &
                kCauseExcCodeMask,
            prefix + "Cause.ExcCode should identify an address error");
        t.Equals(
            ctx.cop0_badvaddr,
            badVirtualAddress,
            prefix + "BadVAddr should retain the rejected virtual address");
        t.Equals(
            ctx.cop0_epc,
            expectedEpc,
            prefix + "EPC should retain the precise restart address");
        t.Equals(
            ctx.pc,
            kCommonVector,
            prefix + "address errors should enter V_COMMON");
        t.Equals(
            (ctx.cop0_cause & kCauseBd) != 0u,
            expectedBd,
            prefix + "Cause.BD should describe delay-slot ownership");
    }

    void generatedFastLoad(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        SET_GPR_U32(ctx, 2u, READ32(0x80001000u));
    }

    void generatedFastStore(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        WRITE32(0xA0001000u, 0xA5A55A5Au);
    }

    uint32_t generatedModeMappedLoad(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime,
        uint32_t address)
    {
        return READ32(address);
    }

    void privilegedDispatchTarget(
        uint8_t *,
        R5900Context *ctx,
        PS2Runtime *)
    {
        SET_GPR_U32(ctx, 2u, 0xBAD0C0DEu);
        ctx->pc = 0u;
    }
}

void register_ee_address_mode_tests()
{
    MiniTest::Case("EE address operating modes", [](TestCase &tc)
    {
        tc.Run("translation context applies every segment boundary", [](TestCase &t)
        {
            PS2Memory memory;
            struct Segment
            {
                uint32_t first;
                uint32_t last;
                bool user;
                bool supervisor;
            };
            constexpr std::array<Segment, 5u> segments{
                Segment{0x00000000u, 0x7FFFFFFFu, true, true},
                Segment{0x80000000u, 0x9FFFFFFFu, false, false},
                Segment{0xA0000000u, 0xBFFFFFFFu, false, false},
                Segment{0xC0000000u, 0xDFFFFFFFu, false, true},
                Segment{0xE0000000u, 0xFFFFFFFFu, false, false},
            };
            struct Mode
            {
                const char *name;
                uint32_t status;
                bool kernel;
                bool supervisor;
            };
            constexpr std::array<Mode, 5u> modes{
                Mode{"user", kStatusUser, false, false},
                Mode{"supervisor", kStatusSupervisor, false, true},
                Mode{"kernel", 0u, true, false},
                Mode{"EXL", kStatusUser | kStatusExl, true, false},
                Mode{"ERL", kStatusUser | kStatusErl, true, false},
            };

            for (const Mode &mode : modes)
            {
                const EeAddressTranslationContext translation =
                    EeAddressTranslationContext::fromCop0Status(
                        mode.status);
                for (const Segment &segment : segments)
                {
                    const bool permitted =
                        mode.kernel ||
                        (mode.supervisor
                             ? segment.supervisor
                             : segment.user);
                    for (const uint32_t address :
                         std::array<uint32_t, 2u>{
                             segment.first,
                             segment.last})
                    {
                        bool addressError = false;
                        try
                        {
                            (void)memory.translateAddress(
                                address, translation);
                        }
                        catch (const PS2AddressErrorException &fault)
                        {
                            addressError = true;
                            t.Equals(
                                fault.virtualAddress(),
                                address,
                                std::string(mode.name) +
                                    " rejection should retain its virtual address");
                        }
                        catch (const PS2TlbMissException &)
                        {
                            // A mapped segment passed the operating-mode gate.
                        }
                        t.Equals(
                            addressError,
                            !permitted,
                            std::string(mode.name) +
                                " should apply the segment permission table");
                    }
                }
            }

            const EeAddressTranslationContext erl =
                EeAddressTranslationContext::fromCop0Status(
                    kStatusUser | kStatusErl);
            t.IsTrue(
                erl.errorLevel,
                "the translation context should retain ERL for kuseg translation");
            t.Equals(
                memory.translateAddress(0x00123456u, erl),
                0x00123456u,
                "ERL kuseg should retain the direct unmapped path");
        });

        tc.Run("user mode rejects every privileged segment boundary", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "the user-mode fixture should allocate RDRAM");

            constexpr std::array<uint32_t, 8u> privilegedBoundaries{
                0x80000000u,
                0x9FFFFFFCu,
                0xA0000000u,
                0xBFFFFFFCu,
                0xC0000000u,
                0xDFFFFFFCu,
                0xE0000000u,
                0xFFFFFFFCu,
            };
            for (const uint32_t address : privilegedBoundaries)
            {
                R5900Context ctx{};
                ctx.pc = 0x00110000u;
                ctx.cop0_status = kStatusUser;
                const bool raised = raisesGuestException([&]()
                {
                    (void)runtime.Load32(
                        runtime.memory().getRDRAM(),
                        &ctx,
                        address);
                });
                expectAddressError(
                    t,
                    ctx,
                    raised,
                    EXCEPTION_ADDRESS_ERROR_LOAD,
                    address,
                    0x00110000u,
                    false,
                    "user boundary: ");
            }
        });

        tc.Run("supervisor mode rejects kernel-only segments", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "the supervisor-mode fixture should allocate RDRAM");

            constexpr std::array<uint32_t, 6u> kernelBoundaries{
                0x80000000u,
                0x9FFFFFFCu,
                0xA0000000u,
                0xBFFFFFFCu,
                0xE0000000u,
                0xFFFFFFFCu,
            };
            for (const uint32_t address : kernelBoundaries)
            {
                R5900Context ctx{};
                ctx.pc = 0x00111000u;
                ctx.cop0_status = kStatusSupervisor;
                const bool raised = raisesGuestException([&]()
                {
                    (void)runtime.Load32(
                        runtime.memory().getRDRAM(),
                        &ctx,
                        address);
                });
                expectAddressError(
                    t,
                    ctx,
                    raised,
                    EXCEPTION_ADDRESS_ERROR_LOAD,
                    address,
                    0x00111000u,
                    false,
                    "supervisor boundary: ");
            }
        });

        tc.Run("loads and stores publish precise address-error state", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "the precision fixture should allocate RDRAM");

            R5900Context loadCtx{};
            loadCtx.pc = 0x00112000u;
            loadCtx.cop0_status = kStatusUser;
            const bool loadRaised = raisesGuestException([&]()
            {
                (void)runtime.Load32(
                    runtime.memory().getRDRAM(),
                    &loadCtx,
                    0x80001000u);
            });
            expectAddressError(
                t,
                loadCtx,
                loadRaised,
                EXCEPTION_ADDRESS_ERROR_LOAD,
                0x80001000u,
                0x00112000u,
                false,
                "load: ");

            R5900Context storeCtx{};
            storeCtx.pc = 0x00113004u;
            storeCtx.branch_pc = 0x00113000u;
            storeCtx.in_delay_slot = true;
            storeCtx.cop0_status = kStatusUser;
            const bool storeRaised = raisesGuestException([&]()
            {
                runtime.Store32(
                    runtime.memory().getRDRAM(),
                    &storeCtx,
                    0xA0001000u,
                    0x11223344u);
            });
            expectAddressError(
                t,
                storeCtx,
                storeRaised,
                EXCEPTION_ADDRESS_ERROR_STORE,
                0xA0001000u,
                0x00113000u,
                true,
                "delay-slot store: ");
        });

        tc.Run("EXL and ERL override a user KSU value", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "the exception-mode fixture should allocate RDRAM");
            runtime.memory().write32(0x00001000u, 0x89ABCDEFu);

            for (const uint32_t exceptionLevel :
                 std::array<uint32_t, 2u>{kStatusExl, kStatusErl})
            {
                R5900Context ctx{};
                ctx.pc = 0x00114000u;
                ctx.cop0_status = kStatusUser | exceptionLevel;
                uint32_t value = 0u;
                bool raised = false;
                try
                {
                    value = runtime.Load32(
                        runtime.memory().getRDRAM(),
                        &ctx,
                        0x80001000u);
                }
                catch (const PS2GuestException &)
                {
                    raised = true;
                }

                t.IsFalse(
                    raised,
                    "active exception levels should select kernel permissions");
                t.Equals(
                    value,
                    0x89ABCDEFu,
                    "active exception levels should retain direct KSEG0 translation");
            }
        });

        tc.Run("mapped fast hits are keyed by KSU EXL and ERL", [](TestCase &t)
        {
            constexpr uint32_t virtualBase = 0x01000000u;
            constexpr uint32_t virtualAddress =
                virtualBase + 0x100u;
            constexpr uint32_t mappedPfn = 0x00400u;
            constexpr uint32_t asid = 0x42u;
            constexpr uint32_t mappedValue = 0x13579bdfu;
            constexpr uint32_t erlDirectValue = 0x2468ace0u;

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "the mapped mode fixture should allocate RDRAM");
            for (uint32_t index = 0u;
                 index < runtime.memory().tlbEntryCount();
                 ++index)
            {
                t.IsTrue(
                    runtime.memory().tlbWrite(
                        index, EeTlbEntry{}),
                    "the mapped mode fixture should clear the TLB");
            }
            t.IsTrue(
                runtime.memory().tlbWrite(
                    9u,
                    EeTlbEntry{
                        0u,
                        virtualBase | asid,
                        makeEntryLo(
                            mappedPfn, true, true, false),
                        makeEntryLo(
                            mappedPfn + 1u, true, true, false),
                    }),
                "the mapped mode fixture should install a non-identity entry");
            std::memcpy(
                runtime.memory().getRDRAM() +
                    (mappedPfn << 12u) + 0x100u,
                &mappedValue,
                sizeof(mappedValue));
            std::memcpy(
                runtime.memory().getRDRAM() + virtualAddress,
                &erlDirectValue,
                sizeof(erlDirectValue));

            R5900Context ctx{};
            ctx.cop0_status = kStatusUser;
            ctx.cop0_entryhi = asid;
            uint8_t *const rdram =
                runtime.memory().getRDRAM();
            t.Equals(
                generatedModeMappedLoad(
                    rdram, &ctx, &runtime, virtualAddress),
                mappedValue,
                "user mode should fill the mapped PFN");
            t.Equals(
                generatedModeMappedLoad(
                    rdram, &ctx, &runtime, virtualAddress),
                mappedValue,
                "user mode should reuse its mapped cache entry");

            ctx.cop0_status = kStatusSupervisor;
            t.Equals(
                generatedModeMappedLoad(
                    rdram, &ctx, &runtime, virtualAddress),
                mappedValue,
                "supervisor mode should refill under its distinct cache key");

            ctx.cop0_status = kStatusUser | kStatusExl;
            t.Equals(
                generatedModeMappedLoad(
                    rdram, &ctx, &runtime, virtualAddress),
                mappedValue,
                "EXL should use the kernel-mode TLB key for ordinary kuseg");

            runtime.memory().setTlbTranslationCacheDiagnosticsEnabled(
                true);
            runtime.memory().resetTlbTranslationCacheStats();
            ctx.cop0_status = kStatusUser | kStatusErl;
            t.Equals(
                generatedModeMappedLoad(
                    rdram, &ctx, &runtime, virtualAddress),
                erlDirectValue,
                "ERL should bypass the stale mapped hit and use direct kuseg");
            t.Equals(
                runtime.memory().tlbTranslationCacheStats().hits,
                0ull,
                "ERL direct kuseg should not probe the mapped cache");

            ctx.cop0_status = kStatusUser;
            t.Equals(
                generatedModeMappedLoad(
                    rdram, &ctx, &runtime, virtualAddress),
                mappedValue,
                "leaving ERL should restore the original user mapping");
            t.Equals(
                runtime.memory().tlbTranslationCacheStats().hits,
                1ull,
                "the unchanged user-mode entry should remain safely reusable");
        });

        tc.Run("generated fast RDRAM paths cannot bypass protection", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "the fast-path fixture should allocate RDRAM");
            runtime.memory().write32(0x00001000u, 0x01020304u);

            R5900Context loadCtx{};
            loadCtx.pc = 0x00115000u;
            loadCtx.cop0_status = kStatusUser;
            const bool loadRaised = raisesGuestException([&]()
            {
                generatedFastLoad(
                    runtime.memory().getRDRAM(),
                    &loadCtx,
                    &runtime);
            });
            expectAddressError(
                t,
                loadCtx,
                loadRaised,
                EXCEPTION_ADDRESS_ERROR_LOAD,
                0x80001000u,
                0x00115000u,
                false,
                "fast load: ");

            R5900Context storeCtx{};
            storeCtx.pc = 0x00115004u;
            storeCtx.cop0_status = kStatusUser;
            const bool storeRaised = raisesGuestException([&]()
            {
                generatedFastStore(
                    runtime.memory().getRDRAM(),
                    &storeCtx,
                    &runtime);
            });
            expectAddressError(
                t,
                storeCtx,
                storeRaised,
                EXCEPTION_ADDRESS_ERROR_STORE,
                0xA0001000u,
                0x00115004u,
                false,
                "fast store: ");
            t.Equals(
                runtime.memory().read32(0x00001000u),
                0x01020304u,
                "a rejected fast store must not mutate physical RDRAM");
        });

        tc.Run("instruction dispatch faults before executing a privileged alias", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.registerFunction(
                    0x00001000u,
                    &privilegedDispatchTarget),
                "the fetch fixture should register its physical target");

            R5900Context ctx{};
            ctx.pc = 0x00116000u;
            ctx.cop0_status = kStatusUser;
            const bool raised = raisesGuestException([&]()
            {
                (void)runtime.dispatchGuestBranch(
                    nullptr,
                    &ctx,
                    0x80001000u,
                    0x00116000u,
                    0u,
                    PS2Runtime::GuestBranchKind::DirectJump,
                    "mode-fixture");
            });
            expectAddressError(
                t,
                ctx,
                raised,
                EXCEPTION_ADDRESS_ERROR_LOAD,
                0x80001000u,
                0x80001000u,
                false,
                "instruction fetch: ");
            t.Equals(
                static_cast<uint32_t>(
                    _mm_cvtsi128_si32(ctx.r[2u])),
                0u,
                "a rejected instruction fetch must not execute the alias target");
        });

        tc.Run("instruction fetch page cache rechecks mode and alignment", [](TestCase &t)
        {
            constexpr uint32_t alignedAddress =
                0x80001000u;
            constexpr uint32_t unalignedAddress =
                alignedAddress + 2u;

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "the fetch-cache fixture should allocate RDRAM");

            R5900Context modeCtx{};
            runtime.validateEeInstructionFetchPolicy<
                EeArchitecturalObservationMode::Fast>(
                &modeCtx, alignedAddress);
            modeCtx.cop0_status = kStatusUser;
            const bool modeRaised = raisesGuestException([&]()
            {
                runtime.validateEeInstructionFetchPolicy<
                    EeArchitecturalObservationMode::Fast>(
                    &modeCtx, alignedAddress);
            });
            expectAddressError(
                t,
                modeCtx,
                modeRaised,
                EXCEPTION_ADDRESS_ERROR_LOAD,
                alignedAddress,
                alignedAddress,
                false,
                "cached mode change: ");

            R5900Context alignmentCtx{};
            runtime.validateEeInstructionFetchPolicy<
                EeArchitecturalObservationMode::Fast>(
                &alignmentCtx, alignedAddress);
            const bool alignmentRaised = raisesGuestException([&]()
            {
                runtime.validateEeInstructionFetchPolicy<
                    EeArchitecturalObservationMode::Fast>(
                    &alignmentCtx, unalignedAddress);
            });
            expectAddressError(
                t,
                alignmentCtx,
                alignmentRaised,
                EXCEPTION_ADDRESS_ERROR_LOAD,
                unalignedAddress,
                unalignedAddress,
                false,
                "cached unaligned fetch: ");
        });

        tc.Run("CACHE address translation observes the current mode", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "the CACHE fixture should allocate RDRAM");

            R5900Context ctx{};
            ctx.pc = 0x00117000u;
            ctx.cop0_status = kStatusUser;
            const bool raised = raisesGuestException([&]()
            {
                runtime.handleEeCacheOperation(
                    runtime.memory().getRDRAM(),
                    &ctx,
                    0x0Eu,
                    0x80000000u);
            });
            expectAddressError(
                t,
                ctx,
                raised,
                EXCEPTION_ADDRESS_ERROR_LOAD,
                0x80000000u,
                0x00117000u,
                false,
                "CACHE fill: ");
        });
    });
}
