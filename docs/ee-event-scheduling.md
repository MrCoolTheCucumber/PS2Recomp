# EE event scheduling

PS2Recomp uses one emulated Emotion Engine timeline and one deterministic
fixed-source scheduler for timed runtime work. Host time is not a timing
authority: VSync, DMA progress, COP0 timers, EE counters, SIF HLE work, and VU
progress all advance from committed guest EE work.

## Timeline and block boundary

`EeTick` is the scheduler unit. One EE cycle is eight ticks, which preserves
the recompiler's three-bit fixed-point dual-issue model inside a basic block.
Generated instructions accumulate local ticks in `R5900Context`. At a generated
basic-block exit:

1. `finishEeBasicBlock()` commits the block once, with a minimum of one EE
   cycle for a non-empty block, and clears its fractional remainder.
2. `EeTimeline` advances monotonically by that committed amount.
3. The runtime compares the new tick with the scheduler's cached next
   deadline.
4. Only a due boundary enters event dispatch.

Context switches flush local elapsed time before changing the bound guest
context. Runtime timing reset invalidates every source generation, resets the
canonical timeline, and re-arms sources whose architectural configuration
survives reset.

The ordinary no-event block path performs one local commit and one cached
deadline comparison. It does not allocate, take a scheduler lock, or scan the
source table. Guest execution ownership serializes timeline and scheduler
mutation.

## Fixed-source scheduler

`EeEventScheduler` has one slot per `EeEventSource`. A slot contains its
deadline, generation, insertion sequence, and pending bit. Scheduling an
already-pending source replaces that slot and advances its generation.
Cancellation also advances the generation, so a stale token cannot cancel
replacement work.

The scheduler caches the earliest pending deadline and source. Equal deadlines
use the order of `EeEventSource`, which is an explicit device-priority
contract. Adding a source therefore requires both an ordering decision and a
focused equal-deadline test.

Callbacks may schedule follow-up work. A callback that schedules its own source
at the current service tick remains in the same device batch, but service is
bounded to 1,024 callbacks per boundary. Exceeding that bound reports the
responsible source and requests runtime stop instead of spinning indefinitely.

Current fixed sources cover:

- COP0 performance overflow, VSync, EE counters, and COP0 Count/Compare;
- typed DMAC completion publication;
- VIF1, GIF, HLE SIF1, VIF0, From-IPU, To-IPU, and both scratchpad DMA
  directions;
- VIF-owned VU0 and VU1 completion.

## Timed-device contract

A timed device retains its architectural operation state independently of the
scheduler slot. That state includes the active phase, channel registers,
progress or stall reason, FIFO state where applicable, and an operation
generation.

A device implementation follows this sequence:

1. Publish guest-visible start/busy state.
2. Retain a resumable descriptor and schedule its next meaningful boundary.
3. At service, validate both the scheduler generation and device generation.
4. Apply only the progress that is due at that boundary.
5. Either schedule a follow-up deadline or publish final device state.
6. Schedule typed completion publication when an interrupt cause becomes
   ready.
7. Deliver guest handlers later at a guest-safe boundary.

Separating completion publication from handler delivery prevents reentrant
guest callbacks from observing half-published device state. Masking delays
delivery, not device completion. Reset, cancellation, and restart invalidate
old generations rather than relying on a synchronous compatibility sweep.
Because each DMAC channel exposes one level-latched D_STAT cause, repeated
completions before acknowledgement share one deferred delivery record. The
runtime retains the newest completion metadata instead of growing a duplicate
queue while that cause is masked.

Idle waits may advance the canonical timeline directly to the cached next
deadline only when no runnable guest owns execution. They do not start a
detached host-time clock.

The interactive runner applies a one-way real-time VSync rate limit after the
scheduler has selected an emulated deadline. It can delay an early VSync but
cannot create an event, advance the EE timeline, or alter a device deadline.
Directly constructed runtimes keep this presentation policy disabled, so
tests and deterministic tools can fast-forward idle waits without host delay.
A bounded four-field catch-up window prevents debugger pauses or slow frames
from turning later guest VSync waits into an unbounded wall-time burst.

## VU execution contract

The scheduler gives a VU an elapsed guest-cycle budget; the VU executes a
batch and reports the exact work consumed. It must not create one scheduler
event per instruction pair.

- VU1 progress and completion are owned by the scheduled VIF/VU1 source.
- VU0 catches up once after a shared due-device batch and at architectural
  COP2 interlocks. It has no private repeating cadence.
- A future interpreter or native backend must preserve the same cycle budget,
  resumable pair boundary, canonical state, observer behavior, and exact
  consumed-cycle result.

This contract allows a faster VU backend without weakening device timing or
reintroducing a second EE clock.

## Observability

The runtime exposes through `ps2dbg`:

- canonical tick, VSync tick, and cached next deadline;
- every fixed slot, generation, pending deadline, and typed device snapshot;
- scheduled and service ticks, lateness, state before/after, and follow-up
  deadline;
- scheduler counts for schedule, replacement, cancellation, service, late
  service, same-tick reschedule, and service-limit hits;
- VU cycle budgets, executed instruction pairs, and invocation correlation;
- DMAC interrupt publication sequence and later handler-delivery sequence.

Scheduler, VU synchronization, and VU instruction traces are dormant without
an atomic enabled check. Enabling a trace reserves a caller-bounded ring once;
recording never grows it past that bound.

## Validation and performance gate

The scheduler unit tests include equal-deadline priority, generation/reset,
same-tick storm, saturation, and four fixed-seed 4,096-operation model
comparisons. Timed-device fixtures assert intermediate register visibility,
stall/resume behavior, completion ordering, masking, cancellation, and reset.

Run the complete runtime suite from the PS2Recomp source directory:

```sh
cmake -S . -B out/test -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPS2X_BUILD_TEST=ON
cmake --build out/test --target ps2x_tests ee_timing_benchmark
out/test/ps2xTest/ps2x_tests
```

The equal-work performance gate is:

```sh
out/test/ps2xTest/ee_timing_benchmark \
  --blocks 1000000 --warmup-blocks 20000 \
  --block-ticks 64 --samples 5
```

`event_scheduler_equal_work` and `pre_scheduler_equal_work` must execute the
same VU instruction-pair count and produce the same state hash. Timed-path
allocation counts must remain zero. The median
`event_to_pre_scheduler_ratio` threshold is 1.05. Menu presentation rate is a
secondary end-to-end metric because it also measures game logic, graphics,
and interpreted VU throughput.

Use `--vu-workload fmac` to run the same equal-work comparison with an FMAC
instruction issued every pair. The JSON output identifies the workload and
reports guest pairs per host second so interpreter pipeline changes can be
measured independently of presentation rate.

## Adding a source

Before adding another timed source:

1. Capture or document its reference start, progress/stall, completion, and
   cancellation behavior.
2. Add its fixed slot and equal-deadline ordering expectation.
3. Keep device state outside the scheduler.
4. Add intermediate-state, generation, reset, and interrupt-order tests.
5. Verify the no-event benchmark still allocates nothing and the equal-work
   ratio remains within the gate.
