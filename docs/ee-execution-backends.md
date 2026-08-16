# EE execution backends

PS2Recomp keeps `legacy-host-thread` as its default. A project can instead
make a default-constructed runtime select the fiber backend when configuring
its runner:

```sh
cmake -S . -B build \
  -DPS2X_DEFAULT_EE_EXECUTION_BACKEND=legacy-cpp-fiber
```

This configuration fails when the validated Boost.Context `fcontext` backend
is unavailable. The launch environment can override the compiled default for
testing or differential runs:

```sh
PS2X_EE_EXECUTION_BACKEND=legacy-cpp-fiber ps2EntryRunner game.elf disc.iso
```

Accepted values are:

- `legacy-host-thread`: one compatibility host thread per guest EE thread;
- `legacy-cpp-fiber`: one EE executor using stackful Boost.Context
  continuations for generated C++ calls.

An unknown value fails runtime construction. Selecting `legacy-cpp-fiber` on
a build without the validated Boost.Context `fcontext` implementation also
fails instead of silently selecting host threads. Runtime diagnostics report
the context build mode, selected backend, Boost version, target architecture,
ABI, and context implementation.

## Owner-local scheduler transitions

The dedicated fiber executor can carry one typed scheduler transition in an
`EeSchedulerRunResult`.  This path is only valid for a synchronous mutation
created by the currently running executor-owned guest fiber and consumed at
the boundary which immediately follows its yield.  It is a value-semantic
protocol: it does not allocate completion state, carry an arbitrary callback,
lock the host-publication queue, or notify its condition variable.

`RotateThreadReadyQueue` currently uses this path when all ownership checks
hold.  The fiber records the originating scheduler handle and generation,
literal requested priority, and declared reschedule policy.  The executor
then commits the exiting continuation, applies the transition exactly once,
reschedules exactly once, publishes the selected context, and only afterward
processes queued host publications, modeled timeouts, hardware events,
interrupts, and debugger disposition.  Stale identities, invalid payload
combinations, duplicate transitions, and reentrant application fail through
the executor's normal deterministic failure path.

The protocol does not replace asynchronous publication.  External host
threads, delayed work, control requests, wakeups, alarms, and operations that
need a completion future continue to use the synchronized publication queue.
The legacy host-thread backend, non-dedicated execution, interrupt-context
rotation, and executor-consequence code retain their established paths.
Runtime diagnostics expose the cumulative
`owner_local_transitions_applied` count separately from publications queued
and applied.

## Context build modes

`PS2X_EE_CONTEXT_BUILD_MODE` makes the continuation contract explicit:

| Mode | Supported target | Context contract |
| --- | --- | --- |
| `production-fcontext` | Linux x86-64 or Windows x64 MSVC | Requires `PS2X_ENABLE_EE_CPP_FIBER_BACKEND=ON` and exact Boost 1.91.0 `fcontext`. This is the production, performance, and platform-CI mode. |
| `host-fallback` | Any target | Uses `legacy-host-thread`. On a supported `fcontext` target, the fiber backend must be explicitly disabled. |
| `sanitizer-no-fcontext` | Linux only | Requires the fiber backend to be explicitly disabled. It covers host-thread, scheduler, publication, reset, and leak behavior, but not the production context switch. |

Contradictory combinations fail configuration. The build never changes from
`fcontext` to `ucontext`, WinFiber, or host threads silently.

Linux sanitizer runs currently use `sanitizer-no-fcontext` and must retain
their compiler sanitizer flags in the recorded configure command. For example:

```sh
cmake -S . -B build-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPS2X_EE_CONTEXT_BUILD_MODE=sanitizer-no-fcontext \
  -DPS2X_ENABLE_EE_CPP_FIBER_BACKEND=OFF \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
```

Boost documents sanitizer integration through `ucontext`. A future
Linux-only, sanitizer-only `ucontext` mode may supplement these checks, but
it must have a distinct build-mode diagnostic and may never count as
production `fcontext` correctness or performance coverage. No such mode is
currently provided.

Windows x64 production CI uses `production-fcontext`. Every Windows x64
configuration that includes the fiber backend uses `/EHs` and disables
IPO/LTCG across suspended frames. Windows MSVC ASan with the production Boost
`fcontext` switch is not a supported project configuration, and the MSVC
toolchain does not provide TSan. The Linux-only sanitizer mode therefore
fails configuration on Windows instead of changing the context implementation.
