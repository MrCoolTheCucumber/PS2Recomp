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
