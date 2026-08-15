#include "MiniTest.h"
#include "ps2_runtime.h"

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>

namespace
{
    uint32_t makeVuIaddiu(uint8_t it, uint8_t is, uint16_t immediate)
    {
        return (0x08u << 25u) |
               (static_cast<uint32_t>((immediate >> 11u) & 0xFu) << 21u) |
               (static_cast<uint32_t>(it & 0xFu) << 16u) |
               (static_cast<uint32_t>(is & 0xFu) << 11u) |
               (static_cast<uint32_t>(immediate) & 0x7FFu);
    }

    uint32_t makeVuBranch(int16_t immediate)
    {
        return (0x20u << 25u) |
               (static_cast<uint32_t>(immediate) & 0x7FFu);
    }

    void writeVuInstructionPair(
        uint8_t *code,
        uint32_t pc,
        uint32_t lower,
        uint32_t upper)
    {
        std::memcpy(code + pc, &lower, sizeof(lower));
        std::memcpy(
            code + pc + sizeof(lower),
            &upper,
            sizeof(upper));
    }
}

void register_ps2_runtime_trace_arm_tests()
{
    MiniTest::Case("PS2RuntimeTraceArms", [](TestCase &tc)
    {
        tc.Run("PC trace arms survive cross-thread transitions", [](TestCase &t)
        {
            PS2RuntimeConfiguration configuration{};
            configuration.vu0Backend = VuBackendKind::Interpreter;
            configuration.vu1Backend = VuBackendKind::Interpreter;
            configuration.useVuBackendEnvironment = false;
            PS2Runtime runtime(configuration);
            t.IsTrue(
                runtime.memory().initialize(),
                "trace-arm fixture memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "trace-arm fixture subsystems should bind");

            uint8_t *const code = runtime.memory().getVU0Code();
            std::memset(code, 0, PS2_VU0_CODE_SIZE);
            constexpr uint32_t kVuNop = 0x0000003Fu;
            writeVuInstructionPair(
                code, 0u, makeVuIaddiu(1u, 1u, 1u), kVuNop);
            writeVuInstructionPair(
                code, 8u, makeVuBranch(-1), kVuNop);
            writeVuInstructionPair(code, 16u, 0u, kVuNop);

            constexpr uint32_t kSyncPc = 0x00122000u;
            constexpr uint32_t kInstructionPc = 0x00122010u;
            constexpr uint32_t kEventPc = 0x00122020u;
            constexpr uint32_t kRestartedSyncPc = 0x00122030u;
            constexpr uint32_t kStoppedEventPc = 0x00122040u;
            constexpr uint32_t kOtherPc = 0x00122050u;

            std::mutex phaseMutex;
            std::condition_variable phaseCv;
            uint32_t phase = 0u;
            const auto publishPhase =
                [&](uint32_t next)
                {
                    {
                        std::lock_guard<std::mutex> lock(phaseMutex);
                        phase = next;
                    }
                    phaseCv.notify_all();
                };
            const auto waitForPhase =
                [&](uint32_t expected)
                {
                    std::unique_lock<std::mutex> lock(phaseMutex);
                    phaseCv.wait(
                        lock,
                        [&]() { return phase >= expected; });
                };

            std::thread debugger(
                [&]()
                {
                    runtime.debugStartEeEventTrace(
                        1u, kEventPc, true);
                    runtime.debugStartVu0SyncTrace(
                        1u, kSyncPc, true);
                    publishPhase(1u);

                    waitForPhase(2u);
                    runtime.debugStartVu0InstructionTrace(
                        1u, kInstructionPc, true);
                    runtime.debugStartVu0SyncTrace(
                        2u, kRestartedSyncPc, true);
                    publishPhase(3u);

                    waitForPhase(4u);
                    runtime.debugEeEventTraceSnapshot(true);
                    runtime.debugStartEeEventTrace(
                        1u, kStoppedEventPc, true);
                    runtime.debugEeEventTraceSnapshot(true);
                    publishPhase(5u);

                    waitForPhase(6u);
                    runtime.debugVu0SyncTraceSnapshot(true);
                });

            R5900Context ctx{};
            const auto generatedBoundary =
                [&](uint32_t pc)
                {
                    ctx.pc = pc;
                    runtime.serviceEeEventsAtGeneratedBlockBoundary(
                        runtime.memory().getRDRAM(), &ctx);
                };

            waitForPhase(1u);
            generatedBoundary(kSyncPc);
            PS2Runtime::DebugVu0SyncTrace syncTrace =
                runtime.debugVu0SyncTraceSnapshot(false);
            PS2Runtime::DebugEeEventTrace eventTrace =
                runtime.debugEeEventTraceSnapshot(false);
            t.IsTrue(
                syncTrace.triggered,
                "the first matching generated boundary should trigger the cross-thread sync arm");
            t.IsFalse(
                eventTrace.triggered,
                "triggering one trace must retain an independent event arm");

            ctx.pc = kOtherPc;
            ctx.advanceEeCycleTicks(800u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &ctx);
            runtime.vu0StartMicroProgram(
                runtime.memory().getRDRAM(), &ctx, 0u);
            ctx.advanceEeCycleTicks(128u);
            runtime.synchronizeVU0Microprogram(
                runtime.memory().getRDRAM(), &ctx, false);
            syncTrace = runtime.debugVu0SyncTraceSnapshot(false);
            t.IsFalse(
                syncTrace.enabled,
                "sync stop-on-full should disable only its completed trace");
            t.Equals(
                syncTrace.totalEntries, 1ull,
                "sync stop-on-full should retain exactly its first entry");
            t.IsFalse(
                runtime.debugEeEventTraceSnapshot(false).triggered,
                "sync stop-on-full must not clear the independent event arm");
            publishPhase(2u);

            waitForPhase(3u);
            generatedBoundary(kInstructionPc);
            PS2Runtime::DebugVu0InstructionTrace instructionTrace =
                runtime.debugVu0InstructionTraceSnapshot(false);
            t.IsTrue(
                instructionTrace.triggered,
                "the first boundary after cross-thread publication should trigger the instruction arm");
            ctx.pc = kOtherPc;
            ctx.advanceEeCycleTicks(128u);
            runtime.synchronizeVU0Microprogram(
                runtime.memory().getRDRAM(), &ctx, false);
            instructionTrace =
                runtime.debugVu0InstructionTraceSnapshot(false);
            t.IsFalse(
                instructionTrace.enabled,
                "instruction stop-on-full should disable only its completed trace");
            t.Equals(
                instructionTrace.totalEntries, 1ull,
                "instruction stop-on-full should retain exactly its first instruction");

            generatedBoundary(kEventPc);
            eventTrace = runtime.debugEeEventTraceSnapshot(false);
            t.IsTrue(
                eventTrace.triggered,
                "the independent event arm should survive both VU trace completions");

            constexpr uint32_t kVif1 = 0x10009000u;
            constexpr uint32_t kSource = 0x00030300u;
            std::memset(
                runtime.memory().getRDRAM() + kSource,
                0, 16u);
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kVif1 + 0x10u, kSource),
                "event stop-on-full fixture MADR write should succeed");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kVif1 + 0x20u, 1u),
                "event stop-on-full fixture QWC write should succeed");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kVif1, 0x100u),
                "event stop-on-full fixture start should succeed");
            ctx.pc = kOtherPc;
            ctx.advanceEeCycleTicks(64u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &ctx);
            eventTrace = runtime.debugEeEventTraceSnapshot(false);
            t.IsFalse(
                eventTrace.enabled,
                "event stop-on-full should disable only its completed trace");
            t.Equals(
                eventTrace.totalEntries, 1ull,
                "event stop-on-full should retain exactly its first service");

            generatedBoundary(kRestartedSyncPc);
            syncTrace = runtime.debugVu0SyncTraceSnapshot(false);
            t.IsTrue(
                syncTrace.triggered,
                "restarting one trace must publish a fresh arm while another trace stops");
            publishPhase(4u);

            waitForPhase(5u);
            generatedBoundary(kStoppedEventPc);
            eventTrace = runtime.debugEeEventTraceSnapshot(false);
            t.IsFalse(
                eventTrace.enabled || eventTrace.triggered,
                "stopping a freshly restarted event trace must retire its aggregate arm");
            t.IsTrue(
                runtime.debugVu0SyncTraceSnapshot(false).triggered,
                "stopping an event arm must not disturb an already triggered sync trace");
            publishPhase(6u);
            debugger.join();
        });
    });
}
