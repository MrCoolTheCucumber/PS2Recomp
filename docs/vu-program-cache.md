# VU compiled-program cache

`VuProgramCache` is the generation-safe owner for verified VU IR and its
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
| Code generation | Stales every pre-write direct handle |
| Exact code-content identity | Reuses a recurring MicroMem image safely |
| Native backend and host features | Separates incompatible emitted code |
| Compilation mode | Separates normal and instrumented blocks |

Architectural registers, pipeline values, and VU data memory are deliberately
absent. Native code must load those values dynamically through its execution
context.

`PS2Memory` increments an independent generation for VU0 or VU1 on every
MicroMem write. This includes direct EE stores and VIF `MPG`, including wrapped
or byte-identical uploads. Data-memory writes do not change the generation.

The native backend assigns content identities from a bounded catalog of exact
MicroMem snapshots. FNV-1a is only a candidate prefilter; a byte-for-byte
comparison is required before an identity is reused, so hash collisions cannot
alias native code. The catalog remembers at most 64 images and discards its
oldest comparison snapshot when full without reusing the identity number.

A cache has one active memory/code scope and generation. A new generation
always advances the handle epoch before lookup. Programs with exact content
identities remain resident and can be rebound through a fresh handle when the
whole code image recurs; changed content misses and compiles independently.
Keys without an exact identity retain the conservative generation-only full
invalidation behavior. A memory/code scope change still discards every
resident. Architectural `VuUnit::reset()` retains the cache because it does
not modify MicroMem.

The native backend must construct a key from the current generation and look it
up immediately before native entry. It must repeat that validation after any
helper which can write MicroMem. Under the current single-GameThread contract,
this avoids a per-pair generation check while ensuring no code write can be
crossed by a stale native entry.

## Handles, bounds, and lifetime

Insertion returns an opaque `{epoch, index}` handle rather than a durable raw
entry pointer. Every generation transition, full invalidation, eviction, or
manual flush advances the nonzero epoch. Consequently, an old handle cannot
resolve even when exact content recurs or its vector index is reused.

Resolved program pointers are valid only until the next non-const cache
operation on the owner thread. Native callers must not retain an entry pointer
across cache lookup, insertion, invalidation, or a helper boundary.

The default bounds are 3,072 programs and 96 MiB of executable mappings per VU.
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
- generation/scope invalidations, retained residents, and programs actually
  discarded;
- exact cross-generation reuse hits;
- eviction flushes, evicted programs, and manual flushes;
- rejected programs;
- emitted bytes and compilation nanoseconds;
- current and high-water program/allocated-byte residency.

Focused tests cover VU0/VU1 separation, every key dimension, direct and wrapped
`MPG` writes, byte-identical uploads, data-only writes, reset, generation
rollover, deterministic bounds, stale handles, W^X permissions, instruction
cache visibility, move ownership, malformed programs, and the thread contract.
