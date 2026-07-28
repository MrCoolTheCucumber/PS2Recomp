# VU execution backends

`PS2Runtime` owns independent `VuUnit` instances for VU0 and VU1. A unit owns
the canonical `VuExecutionState`, decoded/compiled caches, progress counters,
observer configuration, and backend selection. An execution backend receives a
`VuExecutionContext` and returns an exact `VuRunResult`; VIF, the EE scheduler,
and debugger code do not depend on a concrete backend class.

`VuExecutionState` is a deep-copyable verification snapshot. It contains the
architectural registers, branch and active state, issued-cycle count, delayed Q
result, fixed FMAC flag pipeline, and retained XGKICK packet/progress. Caches,
observers, profiling state, and host progress atomics remain unit-owned and are
not cloned.

Device effects use the explicit `IVuSideEffectSink` in the execution context.
The normal runtime sink submits PATH1 packets to PS2 memory/GIF arbitration.
Differential backends can instead record effects transactionally without
publishing either candidate's output twice.

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
```

Programmatic values are applied independently; when
`useVuBackendEnvironment` is true, a corresponding environment variable
overrides that unit's value. Invalid names fail runtime construction with the
variable name and accepted values.

During the backend-neutral architecture phase, every requested mode resolves
to the permanent interpreter. Requested and resolved modes remain distinct so
configuration can be tested before the native and verification backends are
enabled. Changing a request while a unit is active is rejected and leaves the
previous selection intact.

`system.status` exposes `vu_backends.vu0` and `vu_backends.vu1`, including each
unit's requested mode, resolved mode, backend name, and active state.

