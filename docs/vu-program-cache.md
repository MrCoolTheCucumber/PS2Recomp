# VU compiled-program cache

`VuProgramCache` is the generation-scoped owner for verified VU IR and its
immutable native code. Every `VuUnit` lazily owns a separate cache, so VU0 and
VU1 can never share entries even when their entry PCs and code bytes happen to
match.

The cache is present before the production native backend so invalidation,
ownership, bounds, diagnostics, and executable-memory policy can be tested
without mixing them with instruction lowering.

## Identity and invalidation

A compiled program key contains only inputs which can change compilation:

| Field | Purpose |
| --- | --- |
| VU unit | Separates VU0 and VU1 semantics and memory |
| Memory identity | Separates runtime/machine instances |
| Code identity | Separates host mappings of VU MicroMem |
| Code size and address mask | Defines PC wrapping and valid entry extent |
| Entry PC | Selects the compiled block |
| Code generation | Rejects every pre-write block |
| Native backend and host features | Separates incompatible emitted code |
| Compilation mode | Separates normal and instrumented blocks |

Architectural registers, pipeline values, and VU data memory are deliberately
absent. Native code must load those values dynamically through its execution
context.

`PS2Memory` increments an independent generation for VU0 or VU1 on every
MicroMem write. This includes direct EE stores and VIF `MPG`, including wrapped
or byte-identical uploads. Data-memory writes do not change the generation.
The first implementation does not compare content or reuse code across
generations.

A cache has one active memory/code scope and generation. Looking up or inserting
a key from a new scope or generation performs a full invalidation before doing
anything else. Architectural `VuUnit::reset()` retains the cache because it
does not modify MicroMem; a subsequent code write or memory rebind changes the
key and invalidates it.

The native backend must construct a key from the current generation and look it
up immediately before native entry. It must repeat that validation after any
helper which can write MicroMem. Under the current single-GameThread contract,
this avoids a per-pair generation check while ensuring no code write can be
crossed by a stale native entry.

## Handles, bounds, and lifetime

Insertion returns an opaque `{epoch, index}` handle rather than a durable raw
entry pointer. Every full invalidation, eviction, or manual flush advances the
nonzero epoch. Consequently, an old handle cannot resolve even if its vector
index is reused.

Resolved program pointers are valid only until the next non-const cache
operation on the owner thread. Native callers must not retain an entry pointer
across cache lookup, insertion, invalidation, or a helper boundary.

The default bounds are 256 programs and 16 MiB of executable mappings per VU.
Crossing either bound performs a deterministic full flush, then inserts the new
program. A program larger than the byte limit is rejected without disturbing
resident entries. Full flushes keep ownership simple and make dangling native
entries impossible; finer eviction is a later measured optimization.

## Executable memory

`VuExecutableMemory` owns one native allocation:

1. allocate page-rounded read/write memory;
2. copy emitted bytes while the mapping is non-executable;
3. transition the entire mapping once to read/execute;
4. flush the host instruction cache where the platform requires it;
5. keep the allocation immutable until destruction.

There is no API to return an executable mapping to writable state. Recompiling
allocates a fresh object. Windows uses `VirtualAlloc`, `VirtualProtect`, and
`FlushInstructionCache`; Unix-like hosts use `mmap`, `mprotect`, and the
compiler instruction-cache primitive. Unsupported hosts report allocation
failure so backend selection can retain the interpreter.

## Thread contract and diagnostics

The first cache call binds it to the current host thread. Every later call must
come from that same thread; cross-thread access throws `std::logic_error`.
Destruction is exempt. This matches the current GameThread-owned VU execution
model and makes accidental MTVU-style sharing explicit.

The debugger has one read-only exception:
`diagnosticsWhileExecutionQuiescent()` copies counters only while the runtime's
guest-execution mutex proves that the owner cannot access the cache. It neither
binds nor changes cache ownership and must not be used without equivalent
external synchronization.

`VuProgramCacheDiagnostics` exposes cumulative:

- hits, misses, and accepted compilations;
- generation/scope invalidations and discarded programs;
- eviction flushes, evicted programs, and manual flushes;
- rejected programs;
- emitted bytes and compilation nanoseconds;
- current and high-water program/allocated-byte residency.

Focused tests cover VU0/VU1 separation, every key dimension, direct and wrapped
`MPG` writes, byte-identical uploads, data-only writes, reset, generation
rollover, deterministic bounds, stale handles, W^X permissions, instruction
cache visibility, move ownership, malformed programs, and the thread contract.
