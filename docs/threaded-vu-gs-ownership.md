# Threaded VU1 and GS ownership contract

Status: Milestone 1 contract, audited against PS2Recomp revision
`588d9a68d69b8a87a8cd755618502278a10b4ec3` on 2026-08-13 and reconciled
with the implemented GS owner through the Milestone 5 ownership hardening on
2026-08-14.

This document is the source inventory and publication contract for moving GS
and VU1 work off the EE executor. It describes current behavior first and the
required conversion at the ownership seam second. It does not authorize a
worker to publish guest state according to host completion time.

## Access classes

The inventory uses these classes:

| Code | Meaning | Required treatment |
|---|---|---|
| `FF` | Fire-and-forget mutation with no immediate guest answer | Ordered owned command; it may run ahead within the lead bound |
| `OM` | Mutation whose order relative to other work is architectural | Ordered owned command with sequence and generation |
| `GO` | Guest-visible observation or result | Barrier/result published by EE at the architectural boundary |
| `DO` | Control, diagnostic, or presentation observation | Immutable snapshot/result; never inspect live worker state |
| `LC` | Initialize, reset, restore, reconfigure, or shutdown | Explicit command plus barrier or global quiesce |

`FF` does not mean unordered. It means the producer does not need a reply
before continuing. Every command in a subsystem stream remains sequenced.

## Ownership map

| State | Current effective owner | Required owner after threading |
|---|---|---|
| EE tick, event scheduler, DMAC state, interrupt delivery, VIF1 parser and VIF1 registers | EE/runtime | EE/runtime |
| GIF arbitration, Path3 masking, DIRECTHL admission, pending path streams | EE/runtime through `PS2Memory` and `GifArbiter` | EE/runtime |
| GS GIF-visible registers, decode state, vertices, transfers, renderer coordination, debug capture | Calling EE thread, with subordinate raster/Vulkan workers | GS owner |
| Guest-readable privileged GS MMIO mirror (`GSRegisters`) | Shared between EE MMIO/kernel code and synchronous GS SIGNAL/FINISH/LABEL | EE/runtime publication owner; GS returns ordered side effects |
| GS VRAM and CPU/GPU coherency | GS call path; allocation is owned by `PS2Memory` | GS owner for mutation and coherency; EE observes only through a barrier/result |
| VU1 registers, pipeline state, caches, micro memory, data memory | Calling EE thread and VIF1/EE memory writes | VU1 owner |
| VU1 busy, VIF wait/resume, scheduled completion | EE/runtime, synchronously derived from `VuUnit` | EE-published shadow/result state |
| Window and raylib texture state | Main/UI path | Main/UI path |
| Vulkan objects | Existing `GsVulkanService` worker | Existing Vulkan worker, subordinate to GS owner unless later profiling justifies a redesign |
| Software raster tasks | Existing raster pool | Subordinate to GS owner; tasks receive immutable draw inputs |

The mutable `PS2Runtime::gs()`, `gifArbiter()`, and `vu1()` accessors and the
raw `PS2Memory::getGSVRAM()`, `getVU1Code()`, and `getVU1Data()` accessors are
temporary escape hatches. Production callers listed below must be converted;
tests may retain fixture-local direct access where no runtime worker exists.

## GS and GIF access inventory

### Runtime construction and hot path

| Location | Current access | Class | Conversion |
|---|---|---:|---|
| `PS2Runtime::syncCoreSubsystems` | Initializes `GS` with raw VRAM and privileged-register pointers; resets and installs `GifArbiter` callbacks | `LC` | Construct the GS processor/executor, give the processor exclusive GS/VRAM access, and replace callbacks with command/result flow |
| `GifArbiter::submit`, `drain`, `reset`, `empty`, `canAcceptPath2` | Owns copied Path1/2/3 streams and stable path-priority drain | `OM`, `GO`, `LC` | Keep on EE. A drain moves one fully owned `GsDrainBatch` into the GS command processor; `canAcceptPath2` remains an EE-local synchronous query |
| `PS2Memory::submitGifPacket` and `flushMaskedPath3Packets` | Submits Path1/2/3 and optionally drains immediately | `OM` | Continue to arbitrate on EE, then issue an ordered GS batch rather than calling `GS` |
| `PS2Memory::processGIFPacket` and GIF DMA service | Produces Path3 packets | `OM` | Same EE-owned arbiter path; packet bytes must be owned by the batch |
| VIF1 `DIRECT`/`DIRECTHL` | Queries Path2 admission and submits Path2 | `GO`, `OM` | Admission stays EE-local; accepted payload becomes owned Path2 arbiter input |
| VU1 `RuntimeVuSideEffectSink::submitPath1Packet` | Calls `PS2Memory::submitGifPacket(Path1)` synchronously | `OM` | VU slice result owns PATH1 bytes; EE publishes them into the arbiter at the scheduled VU event |
| `GifArbiter` process/drain-boundary callbacks | Directly call `GS::beginRenderBatch`, `processGIFPacket`, and `endRenderSubmissionBatch` | `OM` | Remove callbacks. `GifArbiter::drain` returns or builds a typed owned batch |

`GifArbiter` itself has no independent guest clock and must not be moved to the
GS worker. Path2 admission and Path3 masking feed back into VIF/DMA progress,
so their answers cannot depend on unscheduled host completion.

### Complete public GS surface

All public `GS` operations in `runtime/ps2_gs_gpu.h` fall into the following
owner-facing groups. Private methods are internal GS-owner implementation and
do not create cross-owner entry points.

| Public operations | Current consumers | Class | Required representation |
|---|---|---:|---|
| `init`, destructor | Runtime initialization/destruction and unit fixtures | `LC` | Owner startup/shutdown; Vulkan and raster subordinates terminate before backing memory |
| `reset` | Runtime/core reset and kernel GS reset stub | `LC` | Reset command with generation bump and acknowledgement |
| `processGIFPacket`, `beginRenderBatch`, `endRenderSubmissionBatch`, `flushRenderBatch`, `endRenderBatch` | GIF arbiter and replay/tests | `OM` | One ordered `GsDrainBatch`; flush/end are processor boundaries, not producer calls |
| `processNativePackedGIFPacket`, `uploadImageNative` | Native-chain helpers and tests; the normal packet decoder also has an internal native image fast path | `OM` | Typed packet/upload command with owned bytes, or reject the external shortcut |
| `writeRegister` | Kernel clear/setup helpers and tests | `OM` | Ordered GS-register command |
| `clearFramebufferContext`, `clearActiveFramebuffer` | Kernel double-buffer clear helpers and tests | `OM`, `GO` | Ordered clear command; synchronous boolean acknowledgement only where caller depends on it |
| `consumeLocalToHostBytes` | Kernel GIF FIFO read and tests | `GO` | Drain through requested sequence, materialize VRAM, return owned FIFO bytes |
| `WriteVram`, `ReadVram`, `ReadVramFrom` | GS/raster internals and direct fixtures | `OM`, `GO` | Keep owner-internal. Any external use requires a VRAM mutation/observation command |
| `lockDisplaySnapshot`, `unlockDisplaySnapshot`, `refreshDisplaySnapshot`, `getLastDisplayBaseBytes`, `getContextFrame`, `getPreferredDisplaySource` | Presentation/diagnostic paths and tests | `DO` | Replace pointer/lock protocol and reference returns with immutable snapshot values |
| `latchHostPresentationFrame`, `copyLatchedHostPresentationFrame` | Runtime presentation and debug server | `DO` | Ordered present/field marker produces immutable completed-frame identity and pixels |
| `getDebugSnapshot`, `getDebugHistory`, `copyRecentGifPackets`, `captureReplayState` | Debug UI/server, replay capture, runtime diagnostics | `DO` | Snapshot command/barrier returning owned records |
| `restoreReplayState` | Replay/tests | `LC` | Restore command after generation bump and global GS barrier |
| `clearDebugHistory`, `isDebugHistoryPaused`, `setDebugHistoryPaused` | Debug UI/server and runtime trace control | `DO`, `OM` | Low-frequency control command/snapshot; not a second hot producer |
| `getProgressSnapshot`, `setProgressTrackingEnabled` | Runtime/debug watchdog and server | `DO`, `OM` | Owner-local counters returned as a copied snapshot; setting marshalled through control path |
| `nativeImageUploadCount`, `nativePackedGIFPacketCount`, draw-limit getters | Tests/replay diagnostics | `DO` | Copied diagnostic snapshot |
| `setDrawCommandLimit`, `clearDrawCommandLimit` | Replay control/tests | `OM` | Ordered replay-control command |
| `configureVulkanRenderer`, `setRendererMode`, backend counter controls | Startup, replay/tools, tests | `LC`, `OM` | Configuration command plus barrier; choose ordinary runtime mode before hot execution |
| Renderer mode/diagnostic/capability/statistics/backend-statistics getters | Startup logging, tools, tests | `DO` | Immutable status result; never expose backend objects |

### Direct privileged-register and VRAM access

| Location | Current access | Class | Conversion |
|---|---|---:|---|
| `PS2Memory` initialization/destruction | Allocates, clears, and frees `m_gsVRAM`; initializes `gs_regs` | `LC` | Backing allocation lifetime encloses the GS owner; reset/restore occurs only under global quiesce |
| `PS2Memory` EE MMIO read/write paths | Reads/writes PMODE through SIGLBLID and atomically acknowledges CSR bits | `GO`, `OM` | Keep the guest-readable privileged mirror on EE. Writes that also affect GS-owned rendering become ordered commands |
| GS SIGNAL/FINISH/LABEL decode | Mutates `GSRegisters::csr` and `siglblid` through the pointer passed to `GS::init` | `OM`, `GO` | GS returns ordered `GsPrivilegedSideEffect` entries; EE applies them at the permitted publication point |
| `PS2Runtime` VSync/field paths | Writes `CSR.FIELD` and reads privileged registers for diagnostics | `OM`, `DO` | EE-local privileged mirror update; field marker is also sent to GS in stream order |
| Kernel `System.cpp`, `Interrupt.cpp`, `Stubs/GS.cpp`, and `Stubs/Helpers/Support.h` | Direct PMODE/SMODE2/IMR/CSR/SIGLBLID access, register writes, reset, clear, and FIFO reads | all | Privileged mirror remains EE-local; rendering mutations become commands; FIFO/reset/clear use the barriers above |
| `PS2Runtime::debugCopyGsVram` and raw VRAM diagnostics | Copies `m_boundGSVram` | `DO` | GS VRAM snapshot command that first completes raster/Vulkan coherency |
| Debug UI/server | Reads privileged mirror, GS snapshots/history/frame, and changes diagnostic flags | `DO`, `OM` | Immutable snapshots and the synchronized control path |
| `main.cpp` renderer selection | Calls mutable `GS` before execution | `LC`, `DO` | Configure executor before start or issue a barriered configuration command |

No hot production caller may retain the raw GS VRAM pointer after the command
seam. `GS::getContextFrame()` also returns a live reference today and must not
cross the seam in that form.

### Native and fallback bypasses

- `PS2Memory::tryProcessNativeGifImageUploadChain` calls
  `GS::uploadImageNative` directly and
  `PS2Memory::tryProcessNativeGifPackedChain` calls
  `GS::processNativePackedGIFPacket` directly. Source search finds only test
  callers at the audited revision, but both are public bypasses. Milestone 2
  must route them through the processor or make them fixture-only.
- `GS::processGIFPacket` may internally select
  `tryProcessNativeImageUploadPacket`. This is not an ownership bypass because
  it occurs inside the GS semantic entry; it remains owner-local.
- `RuntimeVuSideEffectSink` falls back to `GS::processGIFPacket` when its
  `PS2Memory*` is null. Runtime passes memory, so this is a fixture/standalone
  compatibility path. It must not exist in the threaded runtime executor.
- Kernel register, clear, FIFO, and reset calls bypass `GifArbiter` by design;
  they still must pass through the GS command processor because arbitration
  and subsystem ownership are separate concerns.

## VIF1 and VU1 access inventory

### Current VIF1 ownership

`PS2Memory` owns VIF1 DMA traversal, deferred input segments, pending DIRECT
payload state, VIF registers, Path3 mask state, and parser progress. These are
all EE-owned in the first threaded design. The parser directly mutates VU1
micro/data memory today; those mutations become VU commands after decoding.

| Location/operation | Current access | Class | Conversion |
|---|---|---:|---|
| EE MMIO reads/writes of VIF1 registers and FBRST | Direct `vif1_regs`, wait/DMA state, reset callback | `GO`, `OM`, `LC` | Retain EE mirror/parser state; reset emits VU generation-bump/reset and waits when required |
| `advanceVif1Dma`, wake/cancel/snapshot helpers | Mutate/query VIF1 DMA and deferred stream | `OM`, `GO` | Stay EE-owned; VU-targeting decoded commands carry owned payloads |
| `processVIF1Data`/parser scalar commands | Mutate CYCLE/OFFSET/BASE/ITOPS/MODE/MASK/ROW/COL and flags | `OM` | Parser state stays EE; copy state required by an UNPACK/start into commands |
| VIF1 `MPG` | Writes raw VU1 micro memory and increments code generation | `OM` | Owned micro-write command, ordered before any following start/slice |
| VIF1 `UNPACK` | Reads VIF row/column/mask/mode and writes raw VU1 data memory | `OM` | Prefer a decoded owned mutation command; result must exactly preserve masking/mode/update semantics |
| EE mapped VU1 micro/data memory reads and writes in all widths | `PS2Memory` maps directly to `m_vu1Code`/`m_vu1Data`; code writes bump generation | `OM`, `GO` | Writes become ordered VU commands. Reads that can observe prior work require a VU barrier/snapshot |
| VIF1 `MSCAL`/`MSCALF`/`MSCNT` callbacks | Calls runtime start/resume synchronously | `OM` | Typed start/resume command plus existing scheduled publication token |
| VU busy callback and VIF wait/resume | Reads `VuUnit::isActive`; completion clears VEW and resumes deferred parser/DMA | `GO`, `OM` | Query EE-published busy shadow; apply `resumeVif1` only from a matching VU result at its event |
| VIF trace/workload profiling | Reads VIF state, VU memories/state, and receives per-instruction/XGKICK callbacks | `DO` | Owner-local trace records returned in slice result or barriered snapshots; diagnostics may not add shared hot mutation |

### Busy-command behavior

The parser checks VU busy only for these six opcodes:

- `FLUSHE`
- `FLUSH`
- `FLUSHA`
- `MSCAL`
- `MSCALF`
- `MSCNT`

If one of them is decoded while VU1 is busy, the parser rewinds to the start
of that command, sets `VIF1_STAT.VEW`, and retains that command and all later
bytes. A completion result clears the wait and resumes parsing in original
byte order.

Before a wait opcode is reached, the current implementation permits these
commands to execute while VU1 is busy: `NOP`, `STCYCL`, `OFFSET`, `BASE`,
`ITOP`, `STMOD`, `MSKPATH3`, `MARK`, `STMASK`, `STROW`, `STCOL`, `MPG`,
`DIRECT`, `DIRECTHL`, and `UNPACK`. Unknown/ignored opcodes also do not acquire
a VU wait. DIRECT/DIRECTHL may independently stall on GIF-path availability,
and an IRQ-marked command may stall after its own effects. The command seam
must preserve these separate reasons rather than treating all input during
VU activity as blocked.

### Complete `VuUnit` surface and raw memory consumers

| Public operations | Current consumers | Class | Conversion |
|---|---|---:|---|
| Constructor/destructor, `reset` | Runtime lifecycle and tests | `LC` | VU owner startup/reset/shutdown with generation acknowledgement |
| `start`, `resumeState`, `advance` | Runtime VIF callbacks and VU scheduled event | `OM` | Typed start/resume/slice commands; slice returns a typed result |
| `execute`, `resume`, `continueExecution` | VU0/runtime helpers and tests | `OM` | VU1 runtime must use the same processor; standalone fixture use may remain inline |
| Mutable/const `state`, `isActive` | Scheduler, debug server, trace/watchdog, tests | `GO`, `DO` | Mutable state is owner-only; EE uses published busy/state results and debug uses immutable snapshots |
| Progress and instruction-observer controls | Debug server, trace and tests | `DO`, `OM` | Owner-local accounting and ordered control commands |
| Backend/instrumentation setters | Runtime initialization/tests | `LC`, `OM` | Configure before worker start or through a barriered control command |
| Backend, cache, recompiler, verify, exit-reason diagnostics | Runtime/debug/replay/tests | `DO` | Immutable diagnostic snapshot after barrier if exact state is requested |
| `PS2Memory::getVU1Code/Data` and code generation | Runtime slice, VIF, recompiler/cache, profiler, debug server, tests | all | Raw mutable pointers become VU-owner internals. Cache keys use owner-local code generation; debug copies through snapshot |

The program cache already asserts thread identity in relevant paths. After
the seam, construction and every cache lookup/invalidation for VU1 must occur
on the VU owner. Pointer identity must not be used as a cross-thread command.

## Callback removal and typed results

### Current callback edges

- `GS -> PS2Runtime`: `m_vsyncTickProvider`, used only to timestamp debug
  history. A GS command already carries `guestTick`; the worker must not call
  back for time.
- `GS -> PS2Memory`: direct mutation of the privileged `GSRegisters` pointer
  for SIGNAL/FINISH/LABEL. Replace it with result records.
- `VU1 -> PS2Memory`: PATH1 submission, VIF trace events, code-generation
  lookup, and workload-profile events. PATH1 and diagnostic events become
  owned slice results; code/data/generation become owner-local execution
  inputs.
- `VU1 -> GS`: the null-memory compatibility fallback. It is forbidden in the
  threaded runtime.
- `PS2Memory/VIF1 -> PS2Runtime`: start/resume/busy/reset and DMA
  schedule/cancel callbacks. These remain EE-local orchestration until the VU
  command seam replaces the VU-specific callbacks with explicit commands and
  published results.
- `GifArbiter -> GS`: packet and drain-boundary callbacks. Replace them with
  `GsDrainBatch` creation and processor submission.

The following shapes are contract definitions; exact C++ names may change in
the implementation. Variable payloads are owning containers.

```cpp
struct WorkIdentity {
    uint64_t sequence;
    uint64_t generation;
    uint64_t guestTick;
    uint64_t publicationToken; // zero if no scheduled publication
};

struct VuPath1PacketResult {
    uint32_t ordinal;
    uint32_t cycleOffset;
    std::vector<uint8_t> bytes;
};

struct Vu1SliceResult {
    WorkIdentity id;
    uint32_t requestedCycles;
    uint32_t executedCycles;
    VuExitReason reason;
    bool activeBefore;
    bool activeAfter;
    bool resumeVif1;
    VuExecutionState state;
    std::vector<VuPath1PacketResult> path1Packets;
    std::optional<VuFault> fault;
    uint64_t stateDigest;
};

using GsPrivilegedSideEffect = std::variant<
    GsSignalEffect, GsFinishEffect, GsLabelEffect>;

struct GsDrainResult {
    WorkIdentity id;
    uint64_t completedSequence;
    std::vector<GsPrivilegedSideEffect> privilegedEffects;
    std::optional<GsReadback> readback;
    std::optional<GsSnapshot> snapshot;
    std::optional<GsCompletedFrame> frame;
    std::optional<GsBackendFault> fault;
    bool barrierAcknowledged;
};
```

`VuExecutionState` may initially be copied in full for clarity. A delta format
is allowed only after exactness and copy cost are measured. SIGNAL and LABEL
records carry both value and mask, and EE applies result entries in vector
order. No result contains `GS*`, `PS2Memory*`, a span into mutable memory, or a
callback.

## Same-tick and publication order

`EeEventSource` numeric order is fixed priority. For the relevant sources it
is:

1. `DmacCompletion`
2. `DmacVif1`
3. `DmacGif`
4. `VifVu1Finish`

`EeEventScheduler::serviceDue` repeatedly calls `takeNextDue`. If a handler
schedules a higher-priority event at the same service tick, the scheduler
recomputes and services that new event on the next iteration. Therefore a
VIF1 finalization that makes DMAC completion ready has this observed order at
one boundary:

```text
DmacVif1 service
  -> schedule DmacCompletion at the same tick
DmacCompletion service
  -> latch D_STAT/pending interrupt state and request guest preemption
DmacGif service, if one was already due
VifVu1Finish service
```

In inline and threaded-synchronous GS policy, the current
`VifVu1Finish` handler has this host execution order:

```text
VuUnit::advance
  -> XGKICK reaches EOP
  -> copy packet bytes into the VU packet buffer
  -> PS2Memory::submitGifPacket(Path1)
  -> GifArbiter::submit + drain
  -> GS packet decode/raster submission
  -> SIGNAL/FINISH/LABEL mutate the privileged mirror, if present
VuUnit::advance returns
  -> publish inactive/busy state
  -> clear VIF1 VEW and resume deferred VIF1 parsing/DMA
```

The asynchronous GS policy preserves the same EE arbitration and submission
order but does not wait for ordinary drain completion. XGKICK bytes still
enter the EE-owned arbiter inside `serviceVU1AtEvent` before VIF1 wake; any
SIGNAL/FINISH/LABEL effect stays private in the ordered EE completion journal
until the first guest observation of the control-register bank, a lifecycle
barrier, or journal backpressure retires it. The only direct-GS XGKICK path is
the null-memory fixture fallback. The focused regression
`event VU1 publishes XGKICK before waking stalled VIF1` proves the normal
ordering, while `equal-tick VIF1 completion publishes before VU1 service`
proves the scheduler priority and same-tick rescan.

Threaded execution must preserve this as publication order, not as worker
wall-clock order. A VU worker may compute early; EE consumes its matching
result only in `VifVu1Finish`, submits PATH1 packets in result order, applies
busy/completion, and only then resumes VIF1. If the result is late, the host
waits at that event without advancing guest time.

## Barrier and publication matrix

There are no unknown production hot-path accesses at this audit point.

| Operation | Producer -> owner | May compute ahead? | EE publication/barrier |
|---|---|---:|---|
| Path1/2/3 GIF packet | EE arbiter -> GS | Yes, after arbitration | Preserve drain sequence and owned bytes |
| DIRECTHL admission | EE -> EE arbiter | No speculative query | Immediate EE-local answer |
| Path3 mask/unmask and buffered release | EE memory -> EE arbiter | No reorder | Release creates ordered GS batch |
| GS decode/register/draw/transfer mutation | EE -> GS | Yes | Later observer names a required sequence |
| SIGNAL/FINISH/LABEL | GS -> EE result | Limited | EE cannot pass a CSR/interrupt observation point without matching result |
| GS CSR/FIFO read | EE -> GS | No at observation | Drain sequence, apply side effects, return FIFO/snapshot |
| CPU/debug VRAM observation | EE/control -> GS | No at observation | Complete raster/Vulkan work and return immutable bytes |
| VSync/present | EE -> GS -> main/UI | Bounded | Ordered field marker; immutable completed-frame identity |
| Renderer change | control -> GS | No across change | Barrier, change, acknowledgement |
| GS/VU debug snapshot | control -> owner | No at snapshot | Barriered immutable copy; low-overhead progress counters may be relaxed snapshots |
| VU micro/data write or UNPACK | EE -> VU | Yes | Ordered before a later start/slice or explicit read |
| MSCAL/MSCALF/MSCNT | EE -> VU | Yes until service | Existing start tick and publication token retained |
| VU busy/status read | guest -> EE | Yes | Read EE-published shadow, never live worker state |
| VU completion/VIF resume | VU -> EE result | No early publication | Apply only at matching VU scheduled event |
| XGKICK | VU -> EE result -> arbiter | No early publication | Publish owned packets at matching VU event before VIF resume |
| Reset/load | EE/control -> all owners | No stale result | Stop publication, bump generation, quiesce, replace state, acknowledge |
| Save state | EE/control -> all owners | No | Fixed global quiesce and immutable snapshots |
| Shutdown/fault | control/worker -> runtime | No | Wake waiters, fixed teardown, deterministic error result |

The public accessors are not exemptions from this matrix. Milestones 2 and 6
must reduce their production visibility or add diagnostic owner assertions so
a new bypass fails close to its call site.

## Thread-affinity constraints

- raylib window and texture operations (`InitWindow`, `BeginDrawing`,
  `EndDrawing`, `UpdateTexture`, `SetWindowTitle`, `WindowShouldClose`,
  `UnloadTexture`, and `CloseWindow`) remain on the main/UI path. The GS owner
  produces an immutable completed frame; it never calls raylib.
- `GsVulkanService::Impl::threadMainImpl` constructs, initializes, uses, and
  destroys `VulkanExecutionContext` on its own worker. Preserve that affinity.
  Current posting calls may borrow spans only because the caller blocks under
  the service protocol; GS commands themselves must still own their inputs.
- The software raster pool is subordinate to the GS owner. A GS barrier joins
  relevant raster tasks and Vulkan/coherency work before reporting completion.
- Debug-server/watchdog threads are observers and control producers. They use
  the synchronized control path and cannot become additional producers of the
  hot SPSC command stream.

## Global quiesce order

Save, load, reset, renderer replacement, fatal shutdown, and ordinary
shutdown use this order:

1. At a safe EE boundary, stop accepting new guest publication and new hot
   submissions for the affected generation.
2. Submit a VU barrier and collect the current slice/result. During reset or
   load, bump the destination generation before accepting later commands.
3. Publish every accepted old-generation XGKICK at its permitted VU event, or
   explicitly discard it only when reset/load semantics invalidate that
   generation.
4. Drain the EE-owned GIF arbiter into final ordered GS batches.
5. Submit a GS barrier and complete software raster work, Vulkan work, and
   CPU-visible VRAM materialization.
6. Snapshot state, or install reset/restored state. Clear old queues,
   mailboxes, pending results, presentation frames, and diagnostics that carry
   the old generation.
7. Resume with the new generation, or shut down the Vulkan service and raster
   pool before destroying GS and its backing memory.

A stale result is silently rejected as guest work but remains visible in
diagnostic counters. Sequence values never move backward within a generation.
Generation zero is reserved as invalid.

## Focused regression ledger

| Required scenario | Evidence at this audit |
|---|---|
| Same-tick VIF1/VU1 event ordering | `equal-tick VIF1 completion publishes before VU1 service` |
| Two XGKICKs and second-XGKICK stall | Existing `a second XGKICK drains the first packet in order` |
| Path1 versus Path2/Path3 ordering | Existing GIF-arbiter path-priority tests |
| DIRECTHL versus Path3 IMAGE exclusion | Existing `GIF arbiter blocks DIRECTHL behind queued Path3 IMAGE data` |
| GS SIGNAL/FINISH/LABEL ordering | `GS SIGNAL FINISH and LABEL retain GIF order and CSR acknowledgement` |
| FIFO/VRAM observation after queued work | `GS local-to-host observation drains queued raster work` |
| VU1 micro/data write followed by MSCAL | `EE VU1 micro and data writes are ordered before MSCAL` |
| VIF1 UNPACK followed by VU1 execution | `VIF1 UNPACK data is visible to the following VU1 execution` |
| VU completion resumes stalled VIF1 DMA | Existing `event VIF1 VU wait removes DMAC_VIF1 until VU wake` |
| Reset/load versus in-flight generation | `threaded GS cancellation rejects stale queued generations and admits reset`, plus `asynchronous GS save load reset renderer and producer ownership remain ordered` |

The equivalent VU generation test remains required before asynchronous VU
ownership.

## Milestone 4 implemented GS publication contract

`threaded-async` uses the same `GsCommandProcessor` and bounded
`ThreadedGsExecutor` as the inline oracle and correctness-first synchronous
mode. The only policy difference is completion retention:

```text
EE-owned GifArbiter drain
  -> move one owned GsDrainBatch into the bounded owner queue
  -> retain its move-only completion in FIFO order
  -> continue EE execution while within queue/journal/field bounds

GS owner
  -> process commands in sequence/generation order
  -> return drained storage plus ordered privileged side effects

EE publication boundary
  -> retire journal entries only from the front
  -> apply SIGNAL/FINISH/LABEL in result-vector order
  -> recycle the moved GifArbiter storage
```

The EE journal is bounded to the configured command capacity plus two entries.
The owner ring and payload budgets remain the primary admission bounds; a full
bound blocks the producer and never drops work. The public host-control path
may submit only synchronous observation/control commands in asynchronous
mode. Hot drain/register/upload/clear/field commands are rejected there so the
runtime keeps exactly one hot producer.

The earliest guest-visible effect boundary is the privileged GS control bank
(`CSR`, `IMR`, `BUSDIR`, and `SIGLBLID`). `PS2Memory` invokes the runtime
observation hook before each such read or write, and the runtime retires all
older journal entries before the access proceeds. Display-register accesses
remain EE-owned and do not drain the journal; their values are copied by value
into the next ordered field marker. FIFO reads, CPU/debug VRAM copies,
snapshots, renderer changes, restore/reset, and shutdown are synchronous owner
commands or lifecycle quiesce points ordered behind all older owner work.

At GS blank, EE first updates its `CSR.FIELD` mirror and then enqueues a field
marker containing the field sequence and an immutable display-register
snapshot. The owner latches presentation from fully completed GS state and
publishes that identity only after the latch is complete. Host presentation
uses the owner's completed identity and copied pixels; it never reads live GS
state. Reset/restore starts a new processor generation and invalidates the
owner's completed presentation identity until a new marker completes.

The runtime accepts a maximum field lead of zero or one. Lead zero waits for
the marker at the same field boundary; lead one permits one incomplete marker
and blocks before a second. Counts are based on ordered EE publication, while
presentation intentionally uses the owner-completed identity, which may be one
journal entry newer without exposing partially mutated state.

Owner metrics distinguish explicit `GsBarrierCommand` retirements from
synchronous wait operations. `barriers_completed` counts only typed barriers;
`barrier_wait_count` and `barrier_wait_ns` count every `submit()` rendezvous
used for an observation or control operation. Queue/payload high-water,
producer blocking, owner active/idle time, field-marker completion, and the EE
journal/lead counters are exposed through `system.status.gs_async`. Those
statistics are observational and do not participate in guest scheduling.

## Milestone 5 hardened GS ownership

### SPSC signaling and lifecycle closure

Successful producer/consumer traffic no longer enters the executor lifecycle
mutex. One serialized hot producer reserves payload bytes and publishes to the
bounded SPSC ring. It then increments a counted atomic work signal; the owner
acquires exactly one signal before popping exactly one command. After a pop
and after payload release, the owner increments a monotonic atomic space epoch
and wakes capacity waiters. A producer loads that epoch before rechecking the
slot and byte bounds, so a release between the capacity check and
`atomic::wait` cannot be lost.

`m_submitMutex` remains the one-producer and lifecycle serialization point; it
does not synchronize the producer with the consumer. Drain, cancel, and fatal
closure follow this order:

1. Publish `accepting=false` and increment the space epoch.
2. Acquire and release the submit mutex, proving every producer that passed
   admission either published its ring entry plus work signal or rejected
   closure.
3. Publish drain or cancel and add one control work signal.
4. Join the owner. Drain consumes all command signals before the control
   signal; cancel resolves the active and queued completions exceptionally.

Fatal closure records its exception before waking a payload-capacity waiter.
This makes failure, rather than newly available payload space, the first state
observed by that producer. Slot and payload waits have separate counts and
wall-time totals; the combined block metric remains for compatibility.

### Exclusive frontend state and immutable subordinates

`GS` no longer contains a recursive state mutex. Architectural frontend state
is mutated and observed only by `GsCommandProcessor` on its selected inline or
owner lane. Runtime snapshots, controls, replay capture/restore, FIFO reads,
and renderer status are typed commands on that lane. The legacy completed
display-frame handoff retains its independent snapshot mutex because the main
thread copies already-latched pixels without accessing frontend state.

The two non-command semantic contexts are explicit and do not overlap a
runtime GS owner: lifecycle construction calls `GS::init`, and a standalone
`VuUnit` fixture with no `PS2Memory` may submit PATH1 directly to its fixture
`GS`. Normal runtime VU execution always has memory and enters the EE-owned GIF
arbiter instead.

Software raster work has this immutable handoff contract:

- each queued raster command owns a copied `GsDrawCommand`, including
  primitive/context/global state, vertices, fixed coordinates, and bounds;
- decoded palettes are copied into the parallel state;
- each raster worker owns a separate `GS` clone and mutates only that clone;
- workers share the canonical VRAM allocation only through the existing
  scanline/hazard partition and expose progress through atomics;
- the primary owner waits for `workersRemaining == 0` before clearing command
  or palette storage or advancing past the flush boundary.

Worker zero is the GS owner itself. The default is five total raster
participants, leaving host capacity for EE, Vulkan, and VU ownership while
retaining almost all measured software throughput on the audited eight-core
host. `PS2X_GS_RASTER_THREADS` remains an explicit 1..16 override because
portable C++ does not expose physical-core topology reliably. Re-audit this
budget after enabling a VU1 owner rather than silently growing the pool.

Debug GIF packets, history entries, replay state, and VRAM observations are
likewise copied into owning containers at an ordered owner boundary. No
subordinate or observer retains a reference to mutable command/frontend state
after the owner advances.

### Vulkan interaction observation

The Vulkan service remains the Vulkan-object owner. Its status now separates
posting-thread request/response wait time from fence wait time measured on the
Vulkan thread. Ordered renderer status also includes service submissions,
transfers, and the backend's resident-command batches and drain reasons.
`system.status` first submits the typed renderer-status command and then
refreshes executor counters, so owner and Vulkan interval snapshots describe
the same ordered boundary. On the audited RAC1 interval, 89-92% of request wait
is device fence time. The remaining 55.7-57.2 microseconds per request is an
upper bound that also includes necessary Vulkan-thread submission work, while
resident requests already average 2,248 commands and reach the 6,144-command
bound. Therefore the service topology is retained for M5. Sampled Vulkan CPU
share alone would not have established this because the posting owner can be
sleeping; reprofile fence and transfer behavior after integrated VU overlap.

## Audit maintenance

The inventory was produced with source searches over runtime headers/sources,
kernel stubs, debug UI/server, main, and tests for `GS`, `GifArbiter`,
`VuUnit`, privileged GS registers, raw GS VRAM, VIF1 state, and raw VU1
micro/data access. When any of these public surfaces changes, update this
document and add an ownership assertion or focused regression with the new
call site. Do not infer completeness from Git tracking alone; generated and
ignored project paths may still contain relevant consumers.
