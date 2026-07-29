# VU1 workload profiling

PS2Recomp can write a bounded, deterministic description of the VU1 work
executed by a title. The profiler is disabled by default and is intended for
workload selection and cache design, not performance timing.

Configure it before starting the runtime:

```sh
PS2X_VU1_PROFILE=/tmp/vu1-workload.json \
PS2X_VU1_PROFILE_WARMUP_PAIRS=2752316 \
PS2X_VU1_PROFILE_PAIRS=100000000 \
ps2EntryRunner GAME.ELF GAME.iso
```

`PS2X_VU1_PROFILE_WARMUP_PAIRS` defaults to zero. A zero
`PS2X_VU1_PROFILE_PAIRS` leaves the capture unbounded; bounded captures write
their result immediately after the exact final measured instruction pair.
Immediate finalization is important for runners which terminate with
`std::_Exit`.

The JSON report groups invocations by reset epoch, complete VU code-memory
FNV-1a hash, and entry PC. For each group it records:

- invocation and instruction-pair counts, including bounded-window counts;
- average, minimum, and maximum invocation lengths;
- primary and special upper/lower opcode counts;
- branch, conditional-branch, and nonsequential-transition counts;
- observed and measured code-generation counts.

The report also records VIF `MPG` upload payload hashes, destinations, byte
counts, and whether the uploaded bytes were already identical at the
destination. Global counts distinguish the full observation interval from the
post-warm-up measurement window.

`system.status` exposes the capture configuration and current pair/upload
counters under `vu1_workload_profile`. A completed bounded capture reports
`measurement_complete: true` and `enabled: false`.

The profiler hashes the whole VU code space once per code generation and
updates maps and opcode counters for every measured pair. Do not enable it
during CPU sampling or throughput measurements. Use the ordinary progress
counters or a deterministic replay/benchmark for timed comparisons.

## Deterministic fixture benchmark

`vu_backend_benchmark` compares the VU1 interpreter and x86-64 recompiler on
one captured fixture. A fixture contains full 16 KiB data and microcode images
plus a replay-state file. Build and run it from a Release build:

```sh
cmake --build BUILD --target vu_backend_benchmark
taskset -c 6 BUILD/ps2xTest/vu_backend_benchmark FIXTURE 4096 \
    --iterations 1000 --warmup 10 --samples 15 \
    --backend both --scope all
```

The benchmark emits one JSON object per line. It first emits a separately
timed `cold-recompiler` result. `warm-sample` and `summary` records then report
the requested timing scopes:

- `vu-core` executes with a pre-sized buffered PATH1 sink. Packet construction
  and required packet copies remain timed, but FNV-1a correctness hashing is
  outside the timed window.
- `vu-plus-path1` submits PATH1 through `PS2Memory` and `GifArbiter`.
  Production-equivalent parsing, queueing, and copies are timed.
- `final_drain_nanoseconds` and `validation_nanoseconds` are outside
  `host_nanoseconds`. `timed_plus_final_drain_nanoseconds` is available when a
  combined wall duration is useful.

Every sample reports guest pairs and cycles, exit reason, allocation counts,
PATH1 packet and byte counts, architectural-state and VU-data hashes, a
per-invocation PATH1 hash, and a full-sample PATH1 hash. Recompiler records
also distinguish C++ native entries, executed native blocks, linked edges,
slow-link exits, native exit/fallback counters, budget-guard classes, and the
complete program-cache and link delta. The guard counters reconcile every
block and pair:

```text
full_block_guards + precise_tail_entries == native_blocks
full_guard_pairs + precise_tail_pairs == native_pairs
```

`budget_guard_comparisons` is the dynamic count of selector, precise-pair, and
required terminal-boundary comparisons. `native_budget_exits` counts native
calls which reach an exact scheduler boundary. A sample is valid only when
every invocation exactly matches the interpreter reference and a warm
recompiler sample performs no compilation.

Cold compilation, warm execution, and forced cache churn are distinct events.
Add cache churn without changing the warm sample:

```sh
taskset -c 6 BUILD/ps2xTest/vu_backend_benchmark FIXTURE 4096 \
    --iterations 1000 --warmup 10 --samples 15 \
    --cache-churn-iterations 10 --backend both --scope all
```

Each `cache-churn-sample` flushes the VU program cache before every timed
invocation. It requires a recompiler-enabled backend selection and verifies
the exact same output as the warm and cold paths.

### Exact budget sweep

`--budget-sweep on` is a correctness mode, not a timing mode. It resets the
fixture state, VU data, and PATH1 sink for every budget from zero through one
past the requested completion length, then compares interpreter and
recompiler result, architectural state, data, and side effects:

```sh
BUILD/ps2xTest/vu_backend_benchmark FIXTURE 4096 \
    --iterations 1 --warmup 1 --samples 1 \
    --backend recompiler --scope core --budget-sweep on
```

The `budget-sweep` record reports the number of tested budgets and retired
pairs plus the aggregate full/precise guard accounting. A nonzero native
fault, an accounting mismatch, or the first semantic divergence fails the
command. Ordinary benchmark records are still emitted afterward; do not use
timings from a sweep process as performance evidence.

### Static block-analysis validation

`--analysis-check on` is a validation mode, not a timing mode. Before any
benchmark warm-up it constructs the reusable basic-block CFG from verified
pair IR, then steps the IR interpreter across each selected block. The
`analysis-check` JSON record reports analyzed and executed blocks, exact
pairs/cycles, static and dynamic edges, XGKICK boundaries, and the terminal
exit. The command fails at the first mismatched PC, fixed cost, successor, or
exit.

```sh
BUILD/ps2xTest/vu_backend_benchmark FIXTURE 4096 \
    --iterations 1 --warmup 1 --samples 1 \
    --backend interpreter --scope core --analysis-check on
```

The analysis models branch and E-bit delay state across block boundaries,
code wrap, unsupported and native fault exits, conditional XGKICK
advancement, helper and unknown-memory clobbers, instrumentation and
transactional-verification barriers (including their phase relative to pair
execution), backwards liveness, VF0/VI0 constants, and the Q/P/VI/FMAC
pipeline facts already proven by the native emitter. Unsupported diagnostic
pairs retire no work and declare no architectural or pipeline resource
effects.
`serializeVuAnalysis` provides the stable `vu-analysis-v1` diagnostic used by
small synthetic golden tests.

Do not compare timings from an analysis-check process with ordinary benchmark
samples: CFG construction and the interpreter trace intentionally warm host
caches before the measured benchmark starts.

### Native pipeline specialization

The native emitter carries exact delayed-pipeline facts forward within a
compiled block. VU integer load/increment, store/increment, and MTIR
instructions create a two-pair VI backup used by immediately following
integer branches. Once the unknown entry countdown has drained, the emitter
also knows the backed-up register identity. It can then:

- refresh a still-active backup of the same register without testing its
  countdown and register at runtime;
- replace an inactive or different-register backup directly; and
- select the delayed or live value for a branch without runtime state tests.

Pipeline advancement still happens before pair execution, and the countdown,
register, and value remain canonical in `VuExecutionState` at every point.
Unknown entry state retains the dynamic checks. Exact budget-cut and split-run
tests cover same-register refresh, different-register replacement, expiry,
and backup-visible branch behavior.

### Host floating-point scope

VU execution sets round-toward-zero for both the x87 control word and MXCSR,
and enables MXCSR denormals-are-zero and flush-to-zero. The caller's exact
state is restored when the backend invocation returns. GNU-family x86 builds
save and update those two control fields directly; other supported hosts use
the standard floating-environment interface plus the available SIMD control
interface.

The internal native fast entry relies on that backend scope. The stable raw
native entry still establishes and restores its own MXCSR mode because callers
can invoke it without the backend adapter. Tests begin from a non-default x87
rounding mode and an MXCSR with DAZ/FTZ clear, verify VU truncation, and require
both caller states to be restored exactly.

## Generated-block attribution

Two independent, opt-in facilities attribute native VU work. Both are
disabled by default:

```sh
PS2X_PERF_JITDUMP=1 \
PS2X_PERF_JITDUMP_DIR="$PWD/profile" \
PS2X_VU_BLOCK_PROFILE=1 \
PS2X_VU_BLOCK_PROFILE_LIMIT=16384 \
ps2EntryRunner GAME.ELF GAME.iso
```

`PS2X_PERF_JITDUMP` creates `jit-PID.dump` on supported Linux x86-64 and
AArch64 hosts. The runtime opens it when the VU backend is created and keeps
the executable marker mapping alive until process shutdown, so `perf record`
can discover the dump even when it attaches to an already-paused process.
Each successfully cached executable block is registered and flushed once.
After recording, inject the symbols before reporting:

```sh
perf inject --jit -i perf.data -o perf.jit.data
perf report -i perf.jit.data
```

Generated names encode the VU unit, whole-code content identity and extent,
the generation at which that executable was compiled, entry PC, inclusive
block PC range, normal or instrumented mode, basic or linear block form, and a
process-unique compilation identity. A cache hit under a later guest
generation does not duplicate the code-load record: the executable and its
compilation identity are unchanged. New or replacement executable storage
always receives a new compilation identity, preventing an old name from being
confused with new code at a reused host address.

`PS2X_VU_BLOCK_PROFILE` adds cold-created records and dynamic counters for:

- executions and guest pairs;
- full-budget and precise bounded entries;
- linked edges and helper barriers;
- resident/dirty VF counts, maximum live pressure, and allocated-access
  coverage;
- canonical loads/stores, helper spill/reload traffic, and materialization
  counts split by canonical exit, link boundary, pair helper, and XGKICK
  helper;
- every native exit reason;
- static IR opcode counts, native range and byte size; and
- JIT code indices and whether the cache allocation is still resident.

The default record limit is 16,384 per VU backend. A positive
`PS2X_VU_BLOCK_PROFILE_LIMIT` changes it; compilations beyond the limit remain
correct and increment `dropped_records`. Enabled blocks update their record
inside generated code at every entry, helper pair, link, and exit, so linked
successors remain exact without returning through C++. Use this mode for
attribution rather than production timing. When the environment variable is
absent, emitted blocks contain none of those profile updates or branches.

With block profiling enabled, `vu_backend_benchmark` appends one
`block-profile-summary` and one `block-profile` JSON record per retained
compilation. Each record includes `block_form` so the adaptive basic-block and
linear-trace populations can be attributed separately. For a live runtime,
pause the guest before requesting a coherent snapshot. `system.status`
includes the first 16 records and reports whether they were truncated; the
`vu.block-profile` debugger method returns pages of up to 256 records. Static
opcode mixes identify the instruction families present in a sampled block,
but they are not per-opcode sample attribution.

### Hardware counters

On Linux, `ps2xTest/tools/vu_backend_perf_stat.sh` uses perf's control FIFO so
counters are enabled only around validated warm timing windows:

```sh
ps2xTest/tools/vu_backend_perf_stat.sh \
    --benchmark BUILD/ps2xTest/vu_backend_benchmark \
    --fixture FIXTURE --pairs 4096 --output OUTPUT \
    --cpu 6 --iterations 1000 --warmup 10 --samples 5 \
    --backend both --scope all
```

The wrapper requires `perf`, `jq`, and `sha256sum`, refuses to replace an
existing output directory, and preserves raw benchmark and perf JSON. It uses
two identical benchmark passes to avoid PMU multiplexing: cycles,
instructions, branches, branch misses, and context switches are collected in
the primary pass; L1 data misses, `LLC-load-misses`, and the portable
`cache-misses` fallback are collected in the cache pass. Unsupported perf
aliases remain explicit in the output. Check `pcnt-running` before comparing
counters and repeat short runs when process placement or host power state
causes outliers.
