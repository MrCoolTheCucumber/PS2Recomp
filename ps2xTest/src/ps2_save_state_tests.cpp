#include "MiniTest.h"

#include "runtime/ps2_save_state.h"
#include "runtime/cop0_timing.h"
#include "runtime/ee_counters.h"
#include "runtime/ee_event_scheduler.h"
#include "runtime/ee_thread_scheduler.h"
#include "runtime/ps2_memory.h"
#include "ps2_runtime.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using namespace ps2x::savestate;

namespace
{
    constexpr uint32_t kSaveStateLoopPc = 0x00104000u;
    constexpr uint32_t kSaveStateCounterAddress = 0x00018000u;
    std::atomic<uint64_t> g_saveStateLoopEntries{0u};

    void saveStateLoop(
        uint8_t *rdram,
        R5900Context *ctx,
        PS2Runtime *runtime)
    {
        uint32_t value = 0u;
        std::memcpy(
            &value, rdram + kSaveStateCounterAddress,
            sizeof(value));
        ++value;
        std::memcpy(
            rdram + kSaveStateCounterAddress,
            &value, sizeof(value));
        ctx->r[16] = _mm_set_epi64x(
            0, static_cast<int64_t>(value));
        ctx->pc = kSaveStateLoopPc;
        g_saveStateLoopEntries.fetch_add(
            1u, std::memory_order_release);
        (void)runtime->checkpointGuestExecution(ctx);
    }

    bool waitForLoopEntries(
        uint64_t minimum,
        std::chrono::milliseconds timeout)
    {
        const auto deadline =
            std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (g_saveStateLoopEntries.load(
                    std::memory_order_acquire) >= minimum)
            {
                return true;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(1));
        }
        return false;
    }

    PS2RuntimeConfiguration saveStateTestConfiguration()
    {
        PS2RuntimeConfiguration configuration{};
        configuration.eeExecutionBackend =
            EeExecutionBackendKind::LegacyCppFiber;
        configuration.useEeExecutionBackendEnvironment = false;
        return configuration;
    }
}

void register_ps2_save_state_tests()
{
    MiniTest::Case("PS2 save state container", [](TestCase &tc)
    {
        tc.Run("round trips chunks and primitive payloads", [](TestCase &t)
        {
            Writer payload;
            payload.u8(0xabu);
            payload.boolean(true);
            payload.u16(0x1234u);
            payload.u32(0x89abcdefu);
            payload.i32(-42);
            payload.u64(0x0123456789abcdefull);
            payload.i64(-1234567890ll);
            payload.f32(1.25f);
            payload.f64(-3.5);
            payload.string("Ratchet & Clank");

            Document source;
            source.chunks.push_back(Chunk{
                .id = makeChunkId('M', 'E', 'T', 'A'),
                .version = 3u,
                .flags = ChunkFlags::Required,
                .payload = payload.take(),
            });
            source.chunks.push_back(Chunk{
                .id = makeChunkId('F', 'U', 'T', 'R'),
                .version = 99u,
                .flags = ChunkFlags::None,
                .payload = {1u, 2u, 3u},
            });

            std::vector<uint8_t> encoded;
            std::string error;
            t.IsTrue(
                encode(source, encoded, &error),
                "container should encode: " + error);

            Document restored;
            t.IsTrue(
                decode(encoded, restored, &error),
                "container should decode: " + error);
            t.Equals(restored.chunks.size(), static_cast<size_t>(2u),
                     "both chunks should survive");

            const Chunk *meta = restored.find(makeChunkId('M', 'E', 'T', 'A'));
            t.IsNotNull(meta, "required metadata chunk should survive");
            if (!meta)
                return;
            t.Equals(meta->version, 3u, "chunk version should survive");
            t.Equals(
                static_cast<uint32_t>(meta->flags),
                static_cast<uint32_t>(ChunkFlags::Required),
                "chunk flags should survive");

            Reader reader(meta->payload);
            uint8_t u8 = 0u;
            bool boolean = false;
            uint16_t u16 = 0u;
            uint32_t u32 = 0u;
            int32_t i32 = 0;
            uint64_t u64 = 0u;
            int64_t i64 = 0;
            float f32 = 0.0f;
            double f64 = 0.0;
            std::string text;
            t.IsTrue(
                reader.u8(u8) && reader.boolean(boolean) &&
                    reader.u16(u16) && reader.u32(u32) &&
                    reader.i32(i32) && reader.u64(u64) &&
                    reader.i64(i64) && reader.f32(f32) &&
                    reader.f64(f64) && reader.string(text) &&
                    reader.atEnd(),
                "all primitive payload fields should decode");
            t.Equals(u8, static_cast<uint8_t>(0xabu), "u8 should survive");
            t.IsTrue(boolean, "boolean should survive");
            t.Equals(u16, static_cast<uint16_t>(0x1234u), "u16 should survive");
            t.Equals(u32, 0x89abcdefu, "u32 should survive");
            t.Equals(i32, -42, "i32 should survive");
            t.Equals(u64, 0x0123456789abcdefull, "u64 should survive");
            t.Equals(i64, -1234567890ll, "i64 should survive");
            t.Equals(f32, 1.25f, "f32 should survive");
            t.Equals(f64, -3.5, "f64 should survive");
            t.Equals(text, std::string("Ratchet & Clank"),
                     "string should survive");
        });

        tc.Run("rejects corruption without replacing the destination", [](TestCase &t)
        {
            Document source;
            source.chunks.push_back(Chunk{
                .id = makeChunkId('R', 'A', 'M', '0'),
                .version = 1u,
                .flags = ChunkFlags::Required,
                .payload = {10u, 20u, 30u, 40u},
            });
            std::vector<uint8_t> encoded;
            std::string error;
            t.IsTrue(encode(source, encoded, &error),
                     "fixture should encode");
            encoded.back() ^= 0xffu;

            Document destination;
            destination.chunks.push_back(Chunk{
                .id = makeChunkId('K', 'E', 'E', 'P'),
                .version = 7u,
            });
            t.IsFalse(decode(encoded, destination, &error),
                      "checksum mismatch should fail");
            t.IsTrue(error.find("checksum") != std::string::npos,
                     "error should identify checksum failure");
            t.IsNotNull(destination.find(makeChunkId('K', 'E', 'E', 'P')),
                        "failed decode must not replace destination");
        });

        tc.Run("rejects duplicate chunk identifiers", [](TestCase &t)
        {
            const uint32_t id = makeChunkId('D', 'U', 'P', 'E');
            Document document;
            document.chunks.push_back(Chunk{.id = id, .version = 1u});
            document.chunks.push_back(Chunk{.id = id, .version = 2u});
            std::vector<uint8_t> encoded;
            std::string error;
            t.IsFalse(encode(document, encoded, &error),
                      "duplicate chunks should fail encoding");
            t.IsTrue(error.find("duplicate") != std::string::npos,
                     "error should identify duplicate chunks");
        });

        tc.Run("bounds length-prefixed payloads", [](TestCase &t)
        {
            Writer writer;
            writer.u64(1024u);
            writer.u8(1u);
            Reader reader(writer.data());
            std::span<const uint8_t> value;
            t.IsFalse(reader.sizedBytes(value, 16u),
                      "oversized length should not be consumed");
        });

        tc.Run("restores timing and event scheduler state", [](TestCase &t)
        {
            using namespace ps2x::timing;

            Cop0Timing cop0;
            cop0.reset(100u, 7u, 90u, 0x80000000u, 3u, 4u);
            (void)cop0.synchronizeCount(180u);
            const Cop0TimingSnapshot cop0State = cop0.snapshot();
            Cop0Timing restoredCop0;
            restoredCop0.restore(cop0State);
            const Cop0TimingSnapshot restoredCop0State =
                restoredCop0.snapshot();
            t.Equals(restoredCop0State.count, cop0State.count,
                     "COP0 Count should restore");
            t.Equals(restoredCop0State.lastCountCycle,
                     cop0State.lastCountCycle,
                     "COP0 time anchor should restore");
            t.Equals(restoredCop0State.timerArmed, cop0State.timerArmed,
                     "COP0 timer arm should restore");

            EeCounterBank counters;
            (void)counters.configureVideoTiming(
                EeCounterVideoTiming{
                    .renderCycles = 1000u,
                    .blankCycles = 100u,
                    .hRenderCycles = 80u,
                    .hBlankCycles = 20u,
                    .hSyncErrorCycles = 2u,
                },
                50u);
            (void)counters.writeRegister(
                kEeCounterBaseAddress +
                    static_cast<uint32_t>(EeCounterRegister::Mode),
                0x0180u, 50u);
            (void)counters.advanceTo(375u);
            const EeCounterBankSnapshot counterState = counters.snapshot();
            EeCounterBank restoredCounters;
            t.IsTrue(restoredCounters.restore(counterState),
                     "counter snapshot should restore");
            const EeCounterBankSnapshot restoredCounterState =
                restoredCounters.snapshot();
            t.Equals(restoredCounterState.currentCycle,
                     counterState.currentCycle,
                     "counter time anchor should restore");
            t.Equals(restoredCounterState.counters[0].count,
                     counterState.counters[0].count,
                     "counter value should restore");
            t.Equals(restoredCounterState.hSyncStartCycle,
                     counterState.hSyncStartCycle,
                     "HSync phase anchor should restore");

            EeEventScheduler events;
            const EeEventToken vif = events.scheduleAbsolute(
                EeEventSource::DmacVif1, eeTickFromRaw(400u));
            (void)events.scheduleAbsolute(
                EeEventSource::VSync, eeTickFromRaw(300u));
            const EeEventSchedulerSnapshot eventState = events.snapshot();
            EeEventScheduler restoredEvents;
            t.IsTrue(restoredEvents.restore(eventState),
                     "event scheduler snapshot should restore");
            t.IsTrue(restoredEvents.pending(EeEventSource::DmacVif1),
                     "pending VIF event should restore");
            t.IsTrue(restoredEvents.cancel(vif),
                     "restored event token generation should remain valid");
            const std::optional<EeTick> deadline =
                restoredEvents.nextDeadline();
            t.IsTrue(deadline.has_value() && deadline->raw() == 300u,
                     "restored next deadline should be recomputed");
        });

        tc.Run("restores EE scheduler queues without native continuations", [](TestCase &t)
        {
            using namespace ps2x::ee;

            EeThreadScheduler scheduler;
            t.IsTrue(scheduler.addRunningThread(1, 11u, 20),
                     "main thread should register");
            t.IsTrue(scheduler.addDormantThread(2, 12u, 10),
                     "worker should register");
            t.IsTrue(scheduler.addDormantThread(3, 13u, 10),
                     "second worker should register");
            t.IsTrue(scheduler.startThread(2), "worker should start");
            t.IsTrue(scheduler.startThread(3), "second worker should start");
            const EeSchedulerWaitKey wait{
                EeSchedulerWaitKind::Semaphore, 7};
            (void)scheduler.blockCurrentThread(wait);
            t.IsTrue(scheduler.validate(),
                     "source scheduler should be valid");

            const EeThreadSchedulerSnapshot state = scheduler.snapshot();
            EeThreadScheduler restored;
            t.IsTrue(restored.restore(state),
                     "scheduler snapshot should restore");
            t.IsTrue(restored.validate(),
                     "restored scheduler should be valid");
            t.Equals(restored.currentThreadId().value_or(0), 2,
                     "running owner should restore");
            const std::vector<int> ready = restored.readyOrder(10);
            t.Equals(ready.size(), static_cast<size_t>(1u),
                     "one peer should remain ready");
            if (!ready.empty())
                t.Equals(ready[0], 3, "ready FIFO order should restore");
            const std::vector<int> waiting = restored.waitOrder(wait);
            t.Equals(waiting.size(), static_cast<size_t>(1u),
                     "one thread should remain waiting");
            if (!waiting.empty())
                t.Equals(waiting[0], 1, "wait FIFO order should restore");
        });

        tc.Run("cold load rebuilds a running EE continuation", [](TestCase &t)
        {
            if (!eeExecutionBackendBuildInfo()
                     .boostContextFcontextAvailable)
            {
                // Sanitizer/no-fcontext builds still cover the portable
                // chunk, memory, timing, and scheduler codecs above.
                return;
            }

            const std::filesystem::path path =
                std::filesystem::temp_directory_path() /
                ("ps2x-savestate-test-" +
                 std::to_string(
                     std::chrono::steady_clock::now()
                         .time_since_epoch().count()) +
                 ".p2s");
            std::error_code cleanupError;
            std::filesystem::remove(path, cleanupError);

            {
                PS2Runtime source(saveStateTestConfiguration());
                t.IsTrue(source.memory().initialize(),
                         "source memory should initialize");
                t.IsTrue(source.syncCoreSubsystems(),
                         "source core should bind");
                t.IsTrue(source.registerFunction(
                             kSaveStateLoopPc, &saveStateLoop),
                         "source loop should register");
                source.cpu().pc = kSaveStateLoopPc;
                source.cpu().r[29] = _mm_set_epi64x(0, 0x00300000u);
                g_saveStateLoopEntries.store(
                    0u, std::memory_order_release);
                source.startDedicatedEeExecutionForTesting();
                t.IsTrue(
                    waitForLoopEntries(
                        32u, std::chrono::seconds(2)),
                    "source continuation should execute");
                std::string error;
                t.IsTrue(source.saveState(path, &error),
                         "running state should save: " + error);
                source.stopDedicatedEeExecutionForTesting();
            }

            {
                PS2Runtime restored(saveStateTestConfiguration());
                t.IsTrue(restored.memory().initialize(),
                         "target memory should initialize");
                t.IsTrue(restored.syncCoreSubsystems(),
                         "target core should bind");
                t.IsTrue(restored.registerFunction(
                             kSaveStateLoopPc, &saveStateLoop),
                         "target loop should register");
                std::string error;
                t.IsTrue(restored.loadState(path, &error),
                         "cold state should load: " + error);
                uint32_t loadedCounter = 0u;
                std::memcpy(
                    &loadedCounter,
                    restored.memory().getRDRAM() +
                        kSaveStateCounterAddress,
                    sizeof(loadedCounter));
                t.IsTrue(loadedCounter >= 32u,
                         "RDRAM should contain the captured running state");
                t.Equals(
                    static_cast<uint32_t>(
                        _mm_extract_epi32(restored.cpu().r[16], 0)),
                    loadedCounter,
                    "EE registers and RDRAM should describe one boundary");

                g_saveStateLoopEntries.store(
                    0u, std::memory_order_release);
                restored.startDedicatedEeExecutionForTesting();
                t.IsTrue(
                    waitForLoopEntries(
                        8u, std::chrono::seconds(2)),
                    "rebuilt continuation should resume guest execution");
                restored.stopDedicatedEeExecutionForTesting();
            }
            std::filesystem::remove(path, cleanupError);
        });

        tc.Run("round trips architectural memory and device registers", [](TestCase &t)
        {
            PS2Memory source;
            PS2Memory restored;
            t.IsTrue(source.initialize(), "source memory should initialize");
            t.IsTrue(restored.initialize(), "target memory should initialize");

            source.getRDRAM()[0x1234u] = 0x5au;
            source.getRDRAM()[PS2_RAM_SIZE - 1u] = 0xa5u;
            source.getScratchpad()[0x321u] = 0x77u;
            source.getIOPRAM()[0x4567u] = 0x33u;
            source.getGSVRAM()[0xabcdu] = 0x91u;
            source.getVU0Code()[17u] = 0x18u;
            source.getVU0Data()[23u] = 0x24u;
            source.getVU1Code()[31u] = 0x32u;
            source.getVU1Data()[47u] = 0x48u;
            source.gs().pmode = 0x0123456789abcdefull;
            source.gs().csr.store(
                0xfedcba9876543210ull,
                std::memory_order_relaxed);

            // VIF1 stream positions are byte offsets within a 16-byte
            // qword, not within a 32-bit command. Leave a partial command
            // after offset four so the codec covers offsets above three.
            const std::array<uint8_t, 4> vifNop{};
            const std::array<uint8_t, 2> vifPartial{0x11u, 0x22u};
            source.processVIF1Data(
                vifNop.data(), static_cast<uint32_t>(vifNop.size()));
            source.processVIF1Data(
                vifPartial.data(),
                static_cast<uint32_t>(vifPartial.size()));

            std::vector<uint8_t> payload;
            std::string error;
            t.IsTrue(source.encodeSaveState(payload, &error),
                     "memory snapshot should encode: " + error);
            t.IsTrue(restored.decodeSaveState(payload, &error),
                     "memory snapshot should decode: " + error);

            t.Equals(restored.getRDRAM()[0x1234u],
                     static_cast<uint8_t>(0x5au),
                     "RDRAM byte should restore");
            t.Equals(restored.getRDRAM()[PS2_RAM_SIZE - 1u],
                     static_cast<uint8_t>(0xa5u),
                     "last RDRAM byte should restore");
            t.Equals(restored.getScratchpad()[0x321u],
                     static_cast<uint8_t>(0x77u),
                     "scratchpad byte should restore");
            t.Equals(restored.getIOPRAM()[0x4567u],
                     static_cast<uint8_t>(0x33u),
                     "IOP RAM byte should restore");
            t.Equals(restored.getGSVRAM()[0xabcdu],
                     static_cast<uint8_t>(0x91u),
                     "GS VRAM byte should restore");
            t.Equals(restored.getVU0Code()[17u],
                     static_cast<uint8_t>(0x18u),
                     "VU0 code byte should restore");
            t.Equals(restored.getVU1Data()[47u],
                     static_cast<uint8_t>(0x48u),
                     "VU1 data byte should restore");
            t.Equals(restored.gs().pmode, 0x0123456789abcdefull,
                     "GS privileged register should restore");
            t.Equals(
                restored.gs().csr.load(std::memory_order_relaxed),
                0xfedcba9876543210ull,
                "atomic GS CSR mirror should restore");
        });
    });
}
