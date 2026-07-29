# VU x86-64 recompiler

The production VU recompiler lowers verified instruction-pair IR into immutable
x86-64 blocks. It is built on supported desktop x86-64 targets when
`PS2X_ENABLE_VU_X64_RECOMPILER=ON`. Unsupported or disabled builds retain the
same public class, report the backend unavailable, and continue to build the
permanent interpreter.

Xbyak 7.37 is fetched from its upstream repository at exact commit
`431abd865e70a46d56f5aa0e1f87572decb60169`. The selected headers are
byte-identical to the snapshot in the accepted PCSX2 reference
`897a2d5516787787e8d4236931b8e2804606bf3c`. Xbyak is used only as an
instruction encoder; its BSD 3-clause license is reproduced in
[`third-party/xbyak.txt`](third-party/xbyak.txt).

## Native entry contract

Every emitted entry has the stable signature:

```cpp
uint64_t block(VuExecutionContext *context, uint32_t maximumCycles);
```

The cache also records a separately addressed internal fast entry. The
recompiler backend builds a `NativeContextView` once for the complete
backend run, establishes VU floating-point mode around that run, and supplies
the view to this internal entry. This avoids repeating the out-of-line context
unpack and MXCSR save/setup/restore at every compiled-block boundary. The
stable two-argument entry remains available for ABI tests and external native
callers; it performs the complete unpack and floating-point-mode transition
itself.

The low 32 result bits are the exact number of retired instruction pairs. The
high 32 bits contain a `VuNativeBlockExit`. Generated code:

- checks the remaining budget before every pair;
- preserves the platform's nonvolatile registers and call-stack requirements;
- saves MXCSR, selects round-toward-zero with DAZ and FTZ for VU work, and
  restores the caller's exact value on every exit;
- writes all architectural and delayed-pipeline state through the supplied
  context;
- returns only at an instruction-pair boundary.

Every block also has a canonical linked entry. It assumes the original native
frame, context view, cumulative cycle count, and backend registers are already
live, but makes no assumptions about guest registers: the predecessor has
written complete architectural and pipeline state to `VuExecutionState`.
Consequently, this first linked form isolates C++ dispatch savings from later
register-state retention.

At a normal block-complete boundary, generated code checks the outgoing
writable slots for `state.pc` and the active MicroMem generation. A compatible
hit records the source completion and tail-jumps to the target linked entry.
A miss records the source and PC, returns canonical state, and lets the slow
C++ loop resolve or compile only after native code has left the stack. Exact
budget exhaustion returns without probing a successor. XGKICK, program end,
unsupported, fault, and invalidation exits continue to return normally.

The linked target reuses the original stack frame, maximum-cycle register,
cumulative retired-pair register, context, and floating-point scope. A source
which used AVX executes `vzeroupper` before the transfer so a later non-AVX
terminal block cannot leak dirty upper lanes through the ABI.
Set `PS2X_VU_BLOCK_LINKING=0` before backend creation to retain the canonical
C++ dispatch path for controlled A/B diagnosis; linking is enabled by default
and its state is exposed in debugger and benchmark diagnostics.

The backend can keep delicate pairs on permanent semantic helpers while
lowering the measured FMAC/conversion, integer, memory, DIV, and branch
families directly. Backend-neutral linear traces continue across sequential
fallthrough, with canonical-PC side exits for taken control flow. Unsupported
IR exits at its original PC without retiring or mutating the pair.

The opcode-disposition audit covers all 122 current IR opcodes. On the
baseline x86-64 feature mask, 71 opcodes are inline-capable, 50 use native
semantic helpers, and only `Unsupported` takes an interpreter side exit.
AVX/FMA promotes 16 FMAC opcodes from helpers to inline lowering, for an
87-inline/34-helper/1-side-exit split. The exhaustive test intentionally pins
these counts so adding or reclassifying an IR opcode requires an explicit
review; the unsupported-pair test separately proves that the side exit retires
and mutates nothing.

Entry pipeline values remain architectural data, not cache-key inputs. A block
therefore advances entry Q, P, and VI-backup state dynamically. Once an
instruction in that block establishes a delayed-scalar latency or VI-backup
countdown, later fallthrough pairs use the statically proven countdown and
omit inactive-pipeline tests. A VU VI backup is architecturally bounded to two
pairs; an impossible larger value faults before the block retires work rather
than entering code compiled under a false invariant.

The helper callback catches all exceptions. No C++ exception may unwind through
generated code.

## Code ownership and invalidation

Xbyak emits into ordinary read/write vector storage. Final bytes are copied
into `VuExecutableMemory`, which performs the sole RW-to-RX transition and host
instruction-cache maintenance. Xbyak never owns executable memory and no RWE
mapping is used.

Blocks embed their owning `VuRecompilerBackend` address. They therefore remain
valid only while that backend and its unit-owned cache live. The cache key
includes unit, memory/code identity, extent, entry PC, exact whole-code content
identity, native feature mask, and compilation mode.

The backend resolves a generation-scoped cache handle immediately before each
native call. Every MicroMem write advances the cache epoch and makes
outstanding handles and link generations stale in O(1). A bounded catalog
hashes the whole code image as a prefilter and then compares all bytes,
allowing compiled blocks for an exactly recurring image to be rebound safely
to the new generation. Physical slots are cleared before any scope
invalidation, eviction, flush, or destruction can reclaim a target mapping.
The backend rechecks the unit's code generation after every call because a
helper may publish an effect which modifies MicroMem. A change returns
`CodeInvalidated`. The current single-GameThread ownership contract avoids an
unmeasured generation load at each pair; every linked edge still compares its
slot generation.

`nativeEntries` counts literal C++-to-native calls. `nativeBlocks` counts every
executed block, including linked successors, and therefore equals
`nativeEntries + linkedEdges`. Slow-link exits, successful resolutions,
abandoned resolutions after cache churn, incompatible edges, block
completions, and all ten native exit reasons have separate cumulative
counters. The native-exit array sums to `nativeBlocks`.
Opt-in block profiles update executions, pairs, budget classes, helper
barriers, linked edges, and exits inside generated code, so their totals remain
exact when the C++ accounting loop is bypassed.

## Instrumentation and tests

An armed instruction observer, VIF trace, or workload profile uses the
permanent interpreter by default. Aggregate progress tracking wraps the
complete scheduler-facing unit call and does not add a per-pair native
callback.

Tests may explicitly enable native instrumentation for a selected unit. In that
mode every supported pair in the block uses the semantic helper, which calls
the instruction observer before pipeline aging or architectural mutation.
Instrumented blocks carry `VuCompilationMode::Instrumented` in their cache key
and cannot alias optimized normal blocks. Unsupported pairs still side-exit at
their original PC and use the permanent interpreter. VIF1 DMA tracing and
workload profiling retain their interpreter path.

An explicit supported VU0 or VU1 `recompiler` selection executes through
`VuUnit`. The unit adapter continues within one scheduler budget after an
internal exit, including a VU1 XGKICK boundary. On `UnsupportedInstruction`, it
executes exactly one pair through the interpreter and then resumes native code;
diagnostics count both the unsupported exit and interpreter fallback pair.
VU0 and VU1 `auto` select this backend on supported x86-64 hosts and retain
interpreter fallback elsewhere. VU0 automatic selection was enabled only after
its synchronization, instrumented trace, and transactional verification gates
passed.

Focused tests cover:

- every possible cycle-budget cut through a synthetic FMAC, Q, memory, branch,
  and E-bit program on both VU0 and VU1;
- VU0 MicroMem branch wrapping and EE-visible synchronization boundaries;
- cold compilation, warm hits, generation replacement, and stale handles;
- static and dynamic self-links, warm host-entry reduction, exact linked
  profiling, mode incompatibility, and forced pre-eviction unlinking;
- unsupported-pair no-mutation behavior;
- instrumentation fallback and exact observer/progress counts;
- opt-in instrumented-native observer ordering and cache separation;
- transactional runtime verification, unsupported-pair fallback, one-time
  PATH1 publication, and compact first-mismatch diagnostics;
- retained XGKICK progress across a native side exit;
- a code-generation change made by a helper;
- rejection of an impossible VI-backup countdown before pipeline
  specialization;
- SysV RBX, RBP, and R12-R15 preservation in a separate assembly caller;
- exact MXCSR restoration around a raw native entry.

Opcode families are inlined only after a measured helper-only baseline and
pair-by-pair differential coverage. Four retained RAC1 VU1 programs also run
through interpreter- and recompiler-selected `VuUnit` paths with identical
final state, VU memory, PATH1 output, and guest-pair counts.
