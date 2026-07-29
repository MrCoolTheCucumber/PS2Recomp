# VU execution backends

`PS2Runtime` owns independent `VuUnit` instances for VU0 and VU1. A unit owns
the canonical `VuExecutionState`, decoded/compiled caches, progress counters,
observer configuration, and backend selection. An execution backend receives a
`VuExecutionContext` and returns an exact `VuRunResult`; VIF, the EE scheduler,
and debugger code do not depend on a concrete backend class.

`VuExecutionState` is a deep-copyable verification snapshot. It contains the
architectural registers, branch and active state, issued-cycle count, delayed Q
and P results, fixed FMAC flag pipeline, and retained XGKICK packet/progress.
Caches, observers, profiling state, and host progress atomics remain unit-owned
and are not cloned.

Device effects use the explicit `IVuSideEffectSink` in the execution context.
The normal runtime sink submits PATH1 packets to PS2 memory/GIF arbitration.
Differential backends can instead record effects transactionally without
publishing either candidate's output twice.

The compact instruction-pair representation and its development-only
differential interpreter are documented in
[`vu-ir.md`](vu-ir.md). They are not selectable runtime backends; runtime
selection can explicitly route either unit through the production x86-64
backend. Its ABI, native lowering, and helper-routed side exits are documented in
[`vu-x64-recompiler.md`](vu-x64-recompiler.md). Each unit lazily owns the
generation-scoped cache and W^X executable-memory layer documented in
[`vu-program-cache.md`](vu-program-cache.md).

## Selection

Code embedding the runtime can select each unit independently:

```cpp
PS2RuntimeConfiguration configuration;
configuration.vu0Backend = VuBackendKind::Interpreter;
configuration.vu1Backend = VuBackendKind::Auto;
configuration.useVuBackendEnvironment = false;
PS2Runtime runtime(configuration);
```

The temporary launch-time front end is:

```text
PS2X_VU0_BACKEND=auto|interpreter|recompiler|verify
PS2X_VU1_BACKEND=auto|interpreter|recompiler|verify
PS2X_VU0_NATIVE_INSTRUMENTATION=0|1
PS2X_VU1_NATIVE_INSTRUMENTATION=0|1
```

Programmatic values are applied independently; when
`useVuBackendEnvironment` is true, a corresponding environment variable
overrides that unit's value. Invalid names fail runtime construction with the
variable name and accepted values.

On a supported x86-64 host, an explicit VU0 or VU1 `recompiler` request resolves
to `x86-64-recompiler`; if native executable memory or the required host
features are unavailable, the request fails early with a diagnostic. `verify`
has the same host requirement and resolves to
`interpreter+x86-64-recompiler`. VU0 and VU1 `auto` resolve to
`x86-64-recompiler` when that backend is built and supported, and fall back to
the interpreter on other hosts. VU0 was enabled only after its EE
synchronization, instrumented trace, and transactional verification gates
passed.

`verify` clones the canonical execution state and complete unit data memory for
both engines. It advances one pair at a time, compares the public result,
architectural and pipeline state, data bytes, and transactional PATH1 packets,
then publishes the agreed state, data, and reference packet exactly once.
Unsupported native pairs retain the normal one-pair interpreter fallback. A
mismatch publishes neither candidate for the failing pair, returns a structured
fault, makes the unit idle to prevent a repeated divergence, and retains a
compact first-divergence diagnostic containing the unit, whole-microcode hash,
entry/failing PC, words and IR operations, budget/pair index, first
state/data/effect difference, pipeline snapshots, exit reasons, cache key, and
code generation. Verification is intentionally slow and is for bounded
fixtures rather than full-speed play.

The unit adapter consumes one scheduler budget across internal native exits,
including VU1 XGKICK helper boundaries. If a compiled block reaches an
unsupported pair, it records the native side exit, executes exactly that pair
through the permanent interpreter at the unchanged PC, and resumes native
execution with the remaining budget. This keeps VU0 EE synchronization and VIF
completion timing independent of native block boundaries. Changing a request
while a unit is active is rejected and leaves the previous selection intact.
The debugger's `vu.backend.set` request additionally requires paused guest
execution; it acquires the guest-execution safe point and changes only the
named inactive unit.

An armed instruction observer uses the permanent interpreter by default, which
preserves the normal instruction-level debugger path. CI and differential trace
runs can set the corresponding `PS2X_VU*_NATIVE_INSTRUMENTATION` value to `1`
(`true` and `on` are also accepted). A selected recompiler then uses
helper-routed native blocks from the distinct `Instrumented` cache namespace
and calls the observer before each supported pair mutates state. This mode is
deliberately slower and does not change normal unobserved blocks. VIF1 DMA
tracing and workload profiling continue to select the permanent interpreter.

Verification disables observers, VIF tracing, and workload profiling inside
both private candidates. Select the interpreter for instruction-level
debugging; this prevents either candidate from publishing trace/device state
before agreement.

Aggregate debugger progress wraps the complete scheduler-facing unit call, so
internal native entries and one-pair interpreter fallbacks do not double-count
cycles or invocations. Progress tracking alone does not request per-pair
instrumentation and therefore does not move supported native work back to the
interpreter.

`system.status` exposes `vu_backends.vu0` and `vu_backends.vu1`, including each
unit's requested mode, resolved mode, backend name, active state, and native
instrumentation setting. At a debugger pause boundary, the GameThread or the
quiescent guest-execution lock also publishes the current PC, last exit reason,
issued cycles, cache/compile counters, normal/instrumented native
pair/exit/fallback counters, block-linking and budget-guard policy,
block-local-VF and inline-XGKICK policy, and verify
run/comparison/publication/mismatch counters. The first mismatch text is
included when present. These snapshots keep owner-only cache state out of the
debugger thread while the guest is running.
