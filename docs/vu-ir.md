# VU instruction-pair IR

The VU recompiler uses a compact, backend-neutral representation of one
upper/lower instruction pair. `VuIrInstructionPair` retains the original PC
and words, semantic upper and lower opcodes, register read/write masks,
pipeline and side-effect flags, pair ordering, and the architectural
one-cycle issue cost. It is intentionally not a general compiler IR.

The pair decoder makes behavior which a native backend must preserve explicit:

- the upper/lower dependency order and LOI's upper-then-immediate order;
- branch, E-bit, and delay-pair boundaries;
- Q/P barriers and delayed Q, P, and FMAC production latency;
- ACC, Q, P, I, status, MAC, and clip accesses;
- VU data-memory reads and writes;
- XGKICK initiation and the per-pair retained-pipeline advance;
- unsupported encodings and recognized interpreter placeholders.

`decodeVuIrBlock` includes branch and E-bit delay pairs. Unsupported or
XGKICK operations take priority when they occur in a delay pair so a backend
can leave at the exact unexecuted operation. `verifyVuIrInstructionPair` and
`verifyVuIrBlock` canonicalize the original words and reject changed
operations, metadata, ordering, cycle costs, control-flow layout, or exit
reasons. Diagnostics retain the original PC and words.

## Development oracle

`VuIrInterpreterBackend` is a development oracle, not a selectable runtime
backend. Its control loop consumes the IR for ordering, barriers, termination,
and side exits while calling the permanent interpreter's proven arithmetic and
memory helpers. This independently checks the representation without creating
a second source of floating-point compatibility behavior. It stops before an
unsupported pair with `VuExitReason::UnsupportedInstruction` and leaves the PC
unchanged.

`destinationMask` uses the VU's XYZW bit order only for operations that
actually mask a destination or memory lane. `selector` is zero for
non-broadcast operations, a scalar component for one-source operations, and
packs DIV/RSQRT's source selector in bits 1:0 and target selector in bits 3:2.
This prevents immediate bits and scalar fields from being mistaken for write
masks by later lowering passes.

Differential runs use independent `VuExecutionState` and VU data-memory clones.
`VuTransactionalSideEffectSink` records PATH1 packets without publishing
either candidate's device effects. `vuExecutionStatesEqual` performs a
bit-exact comparison of all architectural and retained-pipeline fields.

The replay utility can apply this comparison after every pair:

```text
vu1_replay FIXTURE_DIRECTORY PAIRS --ir-differential
```

Existing optional output paths and register-write schedules remain supported.
The JSON result adds the compared-pair count and semantic opcode counts.
The command exits at the first state, VU-memory, PATH1, result, or unsupported
operation divergence and identifies the PC, original words, and IR opcodes.
