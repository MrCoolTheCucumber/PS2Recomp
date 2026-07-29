#include "MiniTest.h"
#include "ps2_runtime.h"
#include "runtime/ps2_debug_server.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu_recompiler.h"
#include "runtime/ps2_watchdog.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

namespace
{
    void firstDispatch(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        ctx->pc = 0x00002000u;
    }

    void secondDispatch(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        ctx->pc = 0u;
    }

    void loopDispatch(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        ctx->pc = 0x00001000u;
    }
}

void register_ps2_debug_control_tests()
{
    MiniTest::Case("PS2DebugControl", [](TestCase &tc)
    {
        tc.Run("pause and resume are idempotent without an active guest", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.debugPause(std::chrono::milliseconds(50)),
                     "pause should quiesce an idle runtime");
            t.IsTrue(runtime.debugIsPaused(),
                     "pause should update debugger-visible state");
            t.IsTrue(runtime.debugPause(std::chrono::milliseconds(50)),
                     "a repeated pause should remain successful");
            runtime.debugResume();
            t.IsFalse(runtime.debugIsPaused(),
                      "resume should clear debugger-visible pause state");
            runtime.debugResume();
            t.IsFalse(runtime.debugIsPaused(),
                      "a repeated resume should remain harmless");
        });

        tc.Run("paused debugger changes only an inactive VU backend", [](TestCase &t)
        {
            PS2RuntimeConfiguration configuration{};
            configuration.useVuBackendEnvironment = false;
            PS2Runtime runtime(configuration);
            std::string diagnostic;
            t.IsFalse(
                runtime.debugSetVuBackend(
                    VuUnitId::Vu0,
                    VuBackendKind::Interpreter,
                    &diagnostic,
                    std::chrono::milliseconds(50)),
                "a running debugger request should be rejected");
            t.IsTrue(
                diagnostic.find("paused") != std::string::npos,
                "the running rejection should explain the safe-point requirement");

            t.IsTrue(
                runtime.debugPause(std::chrono::milliseconds(50)),
                "the idle runtime should pause for backend control");
            t.IsTrue(
                runtime.debugSetVuBackend(
                    VuUnitId::Vu0,
                    VuBackendKind::Interpreter,
                    &diagnostic,
                    std::chrono::milliseconds(50)),
                "paused VU0 selection should succeed");
            t.IsTrue(
                runtime.debugSetVuBackend(
                    VuUnitId::Vu1,
                    VuBackendKind::Interpreter,
                    &diagnostic,
                    std::chrono::milliseconds(50)),
                "paused VU1 selection should succeed independently");
            t.IsTrue(
                runtime.vu0().resolvedBackend() ==
                        VuBackendKind::Interpreter &&
                    runtime.vu1().resolvedBackend() ==
                        VuBackendKind::Interpreter,
                "both units should expose the selected debug backend");

            runtime.vu0().start();
            const VuBackendKind activeRequest =
                VuRecompilerBackend::supported()
                    ? VuBackendKind::Recompiler
                    : VuBackendKind::Auto;
            t.IsFalse(
                runtime.debugSetVuBackend(
                    VuUnitId::Vu0, activeRequest,
                    &diagnostic,
                    std::chrono::milliseconds(50)),
                "an active VU should reject a different backend");
            t.IsTrue(
                diagnostic.find("active") != std::string::npos,
                "the active rejection should preserve a useful diagnostic");
            t.IsTrue(
                runtime.vu0().requestedBackend() ==
                    VuBackendKind::Interpreter,
                "a rejected active switch should preserve VU0 selection");
            runtime.vu0().reset();

            if (VuRecompilerBackend::supported())
            {
                t.IsTrue(
                    runtime.debugSetVuBackend(
                        VuUnitId::Vu1,
                        VuBackendKind::Verify,
                        &diagnostic,
                        std::chrono::milliseconds(50)),
                    "paused inactive VU1 should accept verify");
                t.IsTrue(
                    runtime.vu0().resolvedBackend() ==
                            VuBackendKind::Interpreter &&
                        runtime.vu1().resolvedBackend() ==
                            VuBackendKind::Verify,
                    "debugger selection should remain per-unit");
            }
            runtime.debugResume();
        });

        tc.Run("pause publishes VU diagnostics on the cache owner thread", [](TestCase &t)
        {
            if (!VuRecompilerBackend::supported())
                return;

            PS2RuntimeConfiguration configuration{};
            configuration.vu1Backend =
                VuBackendKind::Verify;
            configuration.useVuBackendEnvironment = false;
            setenv(
                "PS2X_VU_BLOCK_PROFILE", "1", 1);
            PS2Runtime runtime(configuration);
            unsetenv("PS2X_VU_BLOCK_PROFILE");
            t.IsTrue(runtime.memory().initialize(),
                     "diagnostic snapshot memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "diagnostic snapshot subsystems should bind");
            runtime.registerFunction(0x00001000u, loopDispatch);

            uint8_t *const code =
                runtime.memory().getVU1Code();
            std::memset(code, 0, PS2_VU1_CODE_SIZE);
            constexpr uint32_t kVuUpperNop = 0x000002FFu;
            constexpr uint32_t kVuUpperEnd = 0x400002FFu;
            std::memcpy(code + 4u, &kVuUpperEnd, sizeof(kVuUpperEnd));
            std::memcpy(code + 12u, &kVuUpperNop, sizeof(kVuUpperNop));
            runtime.memory().markVU1CodeModified();

            R5900Context context{};
            context.pc = 0x00001000u;
            std::atomic<bool> vuExecuted{false};
            std::thread worker([&]()
            {
                runtime.vu1().start(0u, 0u, 0u, &runtime.memory());
                (void)runtime.vu1().advance(
                    code, PS2_VU1_CODE_SIZE,
                    runtime.memory().getVU1Data(),
                    PS2_VU1_DATA_SIZE,
                    runtime.gs(), &runtime.memory(), 2u);
                vuExecuted.store(true, std::memory_order_release);
                runtime.dispatchLoop(nullptr, &context);
            });

            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::seconds(1);
            while (!vuExecuted.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < deadline)
            {
                std::this_thread::yield();
            }

            t.IsTrue(
                vuExecuted.load(std::memory_order_acquire),
                "the cache owner should execute the native fixture");
            t.IsTrue(
                runtime.debugPause(std::chrono::seconds(1)),
                "a live guest pause should reach an owner-thread boundary");
            std::array<
                PS2Runtime::DebugVuBackendDiagnostics, 2u>
                snapshots{};
            const auto snapshotDeadline =
                std::chrono::steady_clock::now() +
                std::chrono::seconds(1);
            do
            {
                snapshots =
                    runtime.debugVuBackendDiagnosticsSnapshot();
                if (snapshots[1u].captured)
                    break;
                std::this_thread::yield();
            } while (
                std::chrono::steady_clock::now() <
                snapshotDeadline);
            t.IsTrue(snapshots[1u].captured,
                     "the pause boundary should publish a VU1 snapshot");
            t.IsTrue(snapshots[1u].cacheCreated,
                     "native execution should publish cache counters");
            t.IsTrue(snapshots[1u].recompilerCreated,
                     "native execution should publish backend counters");
            t.IsTrue(snapshots[1u].cache.compilations >= 1u,
                     "the paused snapshot should include compilation");
            t.Equals(
                snapshots[1u].recompiler.nativePairs,
                uint64_t{2u},
                "the paused snapshot should report exact native work");
            t.IsTrue(
                snapshots[1u].lastExitReason ==
                    VuExitReason::ProgramEnded,
                "the paused snapshot should retain the last backend exit");
            t.Equals(
                snapshots[1u].recompiler.instrumentedNativePairs,
                uint64_t{0u},
                "ordinary native work should not count as instrumented");
            t.Equals(
                snapshots[1u].recompiler.faultExits,
                uint64_t{0u},
                "the native diagnostic snapshot should remain fault-free");
            t.IsTrue(
                snapshots[1u].recompiler
                    .blockProfileEnabled,
                "the pause boundary should retain opt-in block profiling");
            t.Equals(
                snapshots[1u].recompiler
                    .blockProfileDroppedRecords,
                uint64_t{0u},
                "the focused debugger profile should not drop blocks");
            uint64_t profiledPairs = 0u;
            for (const auto &block :
                 snapshots[1u].recompiler
                     .blockProfiles)
            {
                profiledPairs += block.guestPairs;
            }
            t.Equals(
                profiledPairs, uint64_t{2u},
                "quiescent debugger blocks should report exact native work");
            t.Equals(
                snapshots[1u].verify.runs,
                uint64_t{1u},
                "the snapshot should retain one verify run");
            t.Equals(
                snapshots[1u].verify.comparedPairs,
                uint64_t{2u},
                "the snapshot should retain per-pair comparisons");
            t.Equals(
                snapshots[1u].verify.publishedPairs,
                uint64_t{2u},
                "the snapshot should count published work once");
            t.Equals(
                snapshots[1u].verify.mismatches,
                uint64_t{0u},
                "the paused verify snapshot should remain matched");

            runtime.requestStop();
            runtime.debugResume();
            worker.join();
        });

        tc.Run("breakpoint and watchpoint collections are deterministic", [](TestCase &t)
        {
            PS2Runtime runtime;
            runtime.debugAddBreakpoint(0x80002000u);
            runtime.debugAddBreakpoint(0x00001000u);
            runtime.debugAddBreakpoint(0x00002000u);
            const std::vector<uint32_t> breakpoints = runtime.debugBreakpoints();
            t.Equals(breakpoints.size(), static_cast<size_t>(2),
                     "direct-mapped aliases should not create duplicate breakpoints");
            t.Equals(breakpoints[0], 0x00001000u,
                     "breakpoints should be sorted by canonical address");
            t.Equals(breakpoints[1], 0x00002000u,
                     "breakpoints should retain the second canonical address");

            const uint64_t first = runtime.debugAddWatchpoint(
                0x80003000u, 4u, PS2Runtime::DebugMemoryAccess::Write);
            const uint64_t second = runtime.debugAddWatchpoint(
                0x00004000u, 8u, PS2Runtime::DebugMemoryAccess::ReadWrite);
            t.IsTrue(first != 0u && second > first,
                     "watchpoint IDs should be stable and monotonically increasing");
            const auto watchpoints = runtime.debugWatchpoints();
            t.Equals(watchpoints.size(), static_cast<size_t>(2),
                     "both watchpoints should be listed");
            t.Equals(watchpoints[0].start, 0x00003000u,
                     "watchpoint aliases should be canonicalized");
            t.IsTrue(runtime.debugWatchpointsEnabled(),
                     "the fast-path flag should be enabled while watchpoints exist");
            t.IsTrue(runtime.debugRemoveWatchpoint(first),
                     "an existing watchpoint should be removable");
            t.IsFalse(runtime.debugRemoveWatchpoint(first),
                      "removing an unknown watchpoint should report failure");
            t.IsTrue(runtime.debugRemoveWatchpoint(second),
                     "the remaining watchpoint should be removable");
            t.IsFalse(runtime.debugWatchpointsEnabled(),
                      "the fast-path flag should clear with the final watchpoint");
        });

        tc.Run("debug memory reads include scratchpad aliases", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(),
                     "the runtime memory should initialize");
            runtime.memory().write32(0x70000010u, 0x78563412u);

            std::vector<uint8_t> bytes;
            t.IsTrue(
                runtime.debugReadMemory(0xF0000010u, 4u, bytes),
                "the scratchpad alias should be readable");
            t.Equals(
                bytes.size(), static_cast<size_t>(4u),
                "the debug read should return the requested byte count");
            if (bytes.size() == 4u)
            {
                t.Equals(bytes[0], static_cast<uint8_t>(0x12u),
                         "the scratchpad read should retain little-endian byte zero");
                t.Equals(bytes[3], static_cast<uint8_t>(0x78u),
                         "the scratchpad read should retain little-endian byte three");
            }
            t.IsFalse(
                runtime.debugReadMemory(
                    0x70000000u + PS2_SCRATCHPAD_SIZE - 1u,
                    2u, bytes),
                "a debug read must not cross the scratchpad boundary");
        });

        tc.Run("run-until and dispatch stepping stop before and after a target", [](TestCase &t)
        {
            PS2Runtime runtime;
            runtime.registerFunction(0x00001000u, firstDispatch);
            runtime.registerFunction(0x00002000u, secondDispatch);

            R5900Context context{};
            context.pc = 0x00001000u;
            std::atomic<bool> finished{false};

            t.IsTrue(runtime.debugPause(std::chrono::milliseconds(50)),
                     "the runtime should pause before the worker starts");
            std::thread worker([&]()
                               {
                                   runtime.dispatchLoop(nullptr, &context);
                                   finished.store(true, std::memory_order_release);
                               });

            const PS2Runtime::DebugStopInfo target =
                runtime.debugRunUntilPc(0x00002000u, std::chrono::seconds(1));
            t.IsTrue(target.completed,
                     "run-until should stop at the requested dispatch");
            t.Equals(target.reason, std::string("predicate"),
                     "run-until should identify a predicate hit");
            t.Equals(target.pc, 0x00002000u,
                     "the target function must not execute before the stop");
            t.IsTrue(runtime.debugIsPaused(),
                     "run-until should leave the guest paused");

            const PS2Runtime::DebugStopInfo stepped =
                runtime.debugStepDispatches(1u, std::chrono::seconds(1));
            t.IsTrue(stepped.completed,
                     "one dispatch should execute after stepping");
            t.Equals(stepped.reason, std::string("step"),
                     "the step completion should identify its stop reason");
            t.Equals(stepped.pc, 0u,
                     "the stepped dispatch should publish its resulting PC");

            runtime.debugResume();
            worker.join();
            t.IsTrue(finished.load(std::memory_order_acquire),
                     "the guest worker should exit after its zero return PC");
        });

        tc.Run("branch history and missing-target faults are bounded and structured", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context context{};
            context.pc = 0x00003180u;
            context.r[31] = _mm_set_epi64x(0, 0x00002000u);
            context.r[29] = _mm_set_epi64x(0, 0x001ff000u);
            context.r[28] = _mm_set_epi64x(0, 0x00110000u);

            (void)runtime.lookupFunction(0x80001000u);
            (void)runtime.lookupFunction(0xa0002000u);
            const auto history = runtime.debugBranchHistory();
            t.Equals(history.size(), static_cast<size_t>(2),
                     "lookup targets should enter the shared branch ring");
            if (history.size() == 2u)
            {
                t.Equals(history[0].pc, 0x00001000u,
                         "branch history should canonicalize KSEG0 aliases");
                t.Equals(history[1].pc, 0x00002000u,
                         "branch history should canonicalize KSEG1 aliases");
                t.IsTrue(history[1].sequence > history[0].sequence,
                         "branch history sequences should be monotonic");
            }

            runtime.reportMissingFunction(
                nullptr, &context, 0x00003180u, 0x00002000u,
                PS2Runtime::GuestBranchKind::IndirectJump, "test-dispatch");
            const auto fault = runtime.debugFaultSnapshot();
            t.IsTrue(fault.active,
                     "the first missing target should publish a fault");
            t.Equals(fault.type, std::string("missing-recompiled-target"),
                     "the fault should have a machine-readable type");
            t.Equals(fault.operation, std::string("test-dispatch"),
                     "the fault should retain its operation");
            t.Equals(fault.targetPc, 0x00003180u,
                     "the fault should retain its target PC");
            t.Equals(fault.ra, 0x00002000u,
                     "the fault should retain the return address");

            runtime.resetMissingFunctionReportOnce();
            t.IsFalse(runtime.debugFaultSnapshot().active,
                      "resetting one-shot reporting should clear the active fault");
        });

        tc.Run("watchdog classification identifies active GS VU and waits", [](TestCase &t)
        {
            PS2WatchdogSample previous{};
            PS2WatchdogSample current{};

            current.gsActiveDraws = 1u;
            current.gsDrawsStarted = 1u;
            current.gsCandidatePixels = 4096u;
            auto assessment = ps2ClassifyWatchdogSample(previous, current);
            t.Equals(std::string(assessment.name), std::string("compute-bound-gs"),
                     "active raster progress should classify as compute-bound GS");
            t.Equals(std::string(assessment.hotOperation),
                     std::string("GSRasterizer::writePixel"),
                     "GS classification should name the hot runtime operation");

            previous = {};
            current = {};
            current.vu1Active = true;
            current.vu1Cycles = 512u;
            assessment = ps2ClassifyWatchdogSample(previous, current);
            t.Equals(std::string(assessment.name), std::string("compute-bound-vu"),
                     "advancing VU1 cycles should classify as compute-bound VU");

            previous = {};
            current = {};
            current.guestThreadCount = 3u;
            current.waitingThreadCount = 3u;
            assessment = ps2ClassifyWatchdogSample(previous, current);
            t.Equals(std::string(assessment.name), std::string("waiting-or-deadlock"),
                     "all known guest threads waiting should be distinguished from compute");

            previous = {};
            current = {};
            previous.pc = 0x1000u;
            current.pc = 0x1000u;
            current.dispatches = 32u;
            assessment = ps2ClassifyWatchdogSample(previous, current);
            t.Equals(std::string(assessment.name), std::string("repeated-guest-pc"),
                     "dispatch progress at one PC should identify a repeated guest loop");
        });

        tc.Run("GS debug capture retains bounded replayable GIF packets", [](TestCase &t)
        {
            PS2Memory memory;
            t.IsTrue(memory.initialize(),
                     "the test should initialize PS2 memory");
            GS gs;
            gs.init(memory.getGSVRAM(),
                    static_cast<uint32_t>(PS2_GS_VRAM_SIZE),
                    &memory.gs());
            gs.setDebugHistoryPaused(false);

            std::array<uint8_t, 16> packet{};
            packet[1] = 0x80u; // GIF EOP with NLOOP=0.
            gs.processGIFPacket(
                packet.data(), static_cast<uint32_t>(packet.size()));

            std::vector<uint8_t> stream;
            std::vector<uint32_t> sizes;
            std::vector<uint8_t> initialVram;
            GSDebugSnapshot initialState{};
            gs.copyRecentGifPackets(
                8u, stream, sizes, initialVram, &initialState);
            t.Equals(sizes.size(), static_cast<size_t>(1u),
                     "one submitted packet should be retained");
            if (!sizes.empty())
            {
                t.Equals(sizes[0], static_cast<uint32_t>(packet.size()),
                         "the packet boundary should be retained");
            }
            t.Equals(stream.size(), packet.size(),
                     "the flattened replay stream should retain every byte");
            t.IsTrue(stream == std::vector<uint8_t>(
                                   packet.begin(), packet.end()),
                     "the replay stream should preserve packet bytes");

            gs.clearDebugHistory();
            gs.copyRecentGifPackets(
                8u, stream, sizes, initialVram, &initialState);
            t.IsTrue(stream.empty() && sizes.empty(),
                     "clearing debug history should clear replay packets");
            t.IsTrue(initialVram.empty(),
                     "clearing debug history should clear initial VRAM");
        });

#if defined(PS2X_ENABLE_DEBUG_SERVER) && PS2X_ENABLE_DEBUG_SERVER
        tc.Run("debug server can pause before guest startup", [](TestCase &t)
        {
            const auto unique = std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count());
            const std::filesystem::path root =
                std::filesystem::temp_directory_path() /
                ("ps2dbg-start-paused-test-" + unique);
            const std::filesystem::path socket = root / "debug.sock";
            std::filesystem::create_directories(root);

            setenv("PS2DBG_ENABLE", "1", 1);
            setenv("PS2DBG_SOCKET", socket.c_str(), 1);
            setenv("PS2DBG_START_PAUSED", "1", 1);

            {
                PS2Runtime runtime;
                PS2DebugServer server(runtime);
                t.IsTrue(server.start(),
                         "the isolated debug server should start");
                t.IsTrue(runtime.debugIsPaused(),
                         "startup-paused mode should stop guest execution");
                runtime.debugResume();
                server.stop();
            }

            unsetenv("PS2DBG_START_PAUSED");
            unsetenv("PS2DBG_SOCKET");
            unsetenv("PS2DBG_ENABLE");
            std::filesystem::remove_all(root);
        });

        tc.Run("debug server can leave GS history paused", [](TestCase &t)
        {
            const auto unique = std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count());
            const std::filesystem::path root =
                std::filesystem::temp_directory_path() /
                ("ps2dbg-paused-gs-history-test-" + unique);
            const std::filesystem::path socket = root / "debug.sock";
            std::filesystem::create_directories(root);

            setenv("PS2DBG_ENABLE", "1", 1);
            setenv("PS2DBG_SOCKET", socket.c_str(), 1);
            setenv("PS2DBG_PAUSE_GS_HISTORY", "1", 1);

            {
                PS2Runtime runtime;
                PS2DebugServer server(runtime);
                t.IsTrue(server.start(),
                         "the isolated debug server should start");
                t.IsTrue(
                    runtime.gs().isDebugHistoryPaused(),
                    "the opt-in trace mode should preserve parallel GS batches");
                server.stop();
            }

            unsetenv("PS2DBG_PAUSE_GS_HISTORY");
            unsetenv("PS2DBG_SOCKET");
            unsetenv("PS2DBG_ENABLE");
            std::filesystem::remove_all(root);
        });

        tc.Run("missing targets create one automatic diagnostic bundle", [](TestCase &t)
        {
            const auto unique = std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count());
            const std::filesystem::path root =
                std::filesystem::temp_directory_path() /
                ("ps2dbg-fault-test-" + unique);
            const std::filesystem::path socket = root / "debug.sock";
            const std::filesystem::path crashes = root / "crashes";
            std::filesystem::create_directories(root);

            setenv("PS2DBG_ENABLE", "1", 1);
            setenv("PS2DBG_SOCKET", socket.c_str(), 1);
            setenv("PS2DBG_CRASH_DIR", crashes.c_str(), 1);
            setenv("PS2DBG_WATCHDOG_INTERVAL_MS", "100", 1);
            setenv("PS2DBG_WATCHDOG_THRESHOLD_MS", "600000", 1);

            {
                PS2Runtime runtime;
                PS2DebugServer server(runtime);
                t.IsTrue(server.start(),
                         "the isolated debug server should start");

                R5900Context context{};
                context.pc = 0x00003180u;
                context.r[31] = _mm_set_epi64x(0, 0x00002000u);
                runtime.reportMissingFunction(
                    nullptr, &context, 0x00003180u, 0x00002000u,
                    PS2Runtime::GuestBranchKind::IndirectJump,
                    "automatic-bundle-test");

                std::filesystem::path bundle;
                for (uint32_t attempt = 0u; attempt < 100u; ++attempt)
                {
                    if (std::filesystem::exists(crashes))
                    {
                        for (const auto &entry :
                             std::filesystem::directory_iterator(crashes))
                        {
                            // The writer publishes by renaming its partial
                            // directory. Retaining that transient path after
                            // manifest creation races the rename.
                            if (entry.is_directory() &&
                                entry.path().extension() != ".partial" &&
                                std::filesystem::exists(
                                    entry.path() / "manifest.json"))
                            {
                                bundle = entry.path();
                                break;
                            }
                        }
                    }
                    if (!bundle.empty())
                    {
                        break;
                    }
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(20));
                }

                t.IsFalse(bundle.empty(),
                          "the watchdog should emit a completed fault bundle");
                if (!bundle.empty())
                {
                    t.IsTrue(std::filesystem::exists(
                                 bundle / "runtime-diagnostics.json"),
                             "the bundle should include structured runtime diagnostics");
                    t.IsTrue(std::filesystem::exists(
                                 bundle / "events.jsonl"),
                             "the bundle should include bounded event history");
                    std::ifstream input(bundle / "manifest.json");
                    std::ostringstream text;
                    text << input.rdbuf();
                    t.IsTrue(
                        text.str().find("\"reason\":\"fault\"") !=
                            std::string::npos,
                        "the manifest should identify the automatic fault reason");
                }
                server.stop();
            }

            unsetenv("PS2DBG_WATCHDOG_THRESHOLD_MS");
            unsetenv("PS2DBG_WATCHDOG_INTERVAL_MS");
            unsetenv("PS2DBG_CRASH_DIR");
            unsetenv("PS2DBG_SOCKET");
            unsetenv("PS2DBG_ENABLE");
            std::filesystem::remove_all(root);
        });
#endif
    });
}
