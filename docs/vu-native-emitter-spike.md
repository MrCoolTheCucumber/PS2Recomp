# VU x86-64 native-emitter spike

Status: Phase 0C accepted  
Recorded: 2026-07-28  
Spike commit: `82b2fc8`

## Decision

Use Xbyak as the first production x86-64 VU emitter.

Xbyak is header-only, has a BSD 3-clause license, adds no runtime shared
library, and produced a repeatable 1.20x warm-path speedup after only one
measured FMAC family was inlined. Its cold compile cost was tens of
microseconds for the selected 44-pair block. LLVM ORC is not justified for the
first backend, and PCSX2's custom emitter is too coupled to PCSX2 to import as
a dependency.

The production integration must pin Xbyak independently. The spike deliberately
accepts `PS2X_XBYAK_INCLUDE_DIR` and uses the copy bundled with the local PCSX2
reference; ordinary PS2Recomp builds remain dependency-free because
`PS2X_BUILD_VU_NATIVE_EMITTER_SPIKE` defaults to `OFF`.

## Scope

The spike uses the retained RAC1 fixture
`683eafb3192f70c2-entry-0000`. It replays the fixture to capture architectural
state at trace pair 116 and compiles the sequential 44-pair upper-lane block
from `PC 0x0578` through `0x06d0`.

The selected block contains:

- 10 `MULq` pairs (22.7%);
- 10 upper NOP pairs (22.7%);
- 24 other upper operations routed through the existing interpreter helper.

Three equal-work paths advance the fixed FMAC flag pipeline at the same point:

1. a direct C++ loop through `advanceFmacFlagPipeline()` and `execUpper()`;
2. an emitted block that calls those helpers for every pair;
3. an emitted block that removes upper NOP calls and inlines `MULq` with packed
   SSE, while retaining the interpreter's FMAC flag scheduler.

All paths start from the captured RAC state, execute the same pair count, flush
the FMAC pipeline, and compare a hash of every public `VU1State` field. The
spike also repeats one-pair native entries and verifies that `nextPair` resumes
at the correct position.

This is not a complete VU backend. It intentionally excludes the lower lane,
Q/XGKICK advancement, branches, cycle accounting, code invalidation, and
device side effects. The result establishes emitter feasibility and the value
of specialization; it does not predict an end-to-end RAC1 speedup.

## Emitter and ABI prototype

The spike used the Xbyak snapshot bundled in PCSX2 commit
`897a2d5516787787e8d4236931b8e2804606bf3c`. The header reports version
`0x7370`. Because the bundled directory has no independent Git metadata, the
source snapshot is identified by the PCSX2 revision and these SHA-256 hashes:

| File | SHA-256 |
| --- | --- |
| `COPYRIGHT` | `084ff0bd80b35911ee4fa702d41330038212e7cc50611bfaa868091d69301b6c` |
| `xbyak.h` | `8533bf74a3bb07a3301e1d7c40cec891b3ffaed68da31730d513f641384fdfa1` |
| `xbyak_mnemonic.h` | `3faece1ed593c6ac819c4d50a91b097533b6308442b51c38f3a64201338f222d` |
| `xbyak_util.h` | `e81d691c2cc9b6f84cb657362927483c600da15f68947b4d3faacfdb7a62a61f` |
| `xbyak_bin2hex.h` | `f1f2adbcf4fe752ee72e1150371f1a74f43dd9af2afc27ec954c45936ebef7a3` |

The prototype covers the required host boundary:

- Xbyak `AutoGrow` storage is left read/write while emitting and
  `readyRE()` changes the final mapping directly to read/execute. The observed
  Linux mappings were `rw-p` then `r-xp`; no writable/executable interval was
  used.
- The native signature is `uint32_t (*)(SpikeContext*)`. Generated code
  preserves its nonvolatile registers and keeps the SysV stack aligned across
  helper calls. A separate assembly caller seeds and verifies RBX, RBP, and
  R12-R15.
- Entry saves MXCSR, loads VU round-toward-zero plus DAZ/FTZ mode, and every
  exit restores the caller's value. The measured transition was host
  `0x00001fa0` to VU `0x0000ffe0` and back.
- Finalization calls `__builtin___clear_cache` after resolving labels and
  changing protection. This is a no-op on coherent x86-64 hosts but preserves
  the required portable cache-maintenance boundary.
- `nextPair` and an explicit pair budget provide exact side exits and resumed
  entry at pair boundaries.

Xbyak's default allocation interfaces can permit RWE mappings. Production code
must retain the explicit RW-to-RX wrapper used here rather than relying on
Xbyak defaults.

## Alternatives

| Candidate | Local evidence | Decision |
| --- | --- | --- |
| Xbyak | Five headers, 480 KiB total; BSD 3-clause; no shared-library dependency | Selected |
| PCSX2 x86 emitter | 34 source/header files, 324 KiB and 7,588 lines; GPL-compatible but tied to PCSX2 common types, assertions, register classes, and global emission state | Reuse design ideas, not the emitter code |
| LLVM ORC 22.1.8 | `/usr/lib/libLLVM.so.22.1` is 164 MiB; LLVM headers are 39 MiB across 2,281 files; the installed package reports 130.79 MiB and PS2Recomp currently has no LLVM dependency | Rejected for the first backend on build and deployment cost |
| Hand-written encoder | Small initial footprint, but would duplicate relocation, label, instruction-form, and feature-validation work | Rejected |

No local AsmJit, DynASM, or SLJIT checkout or system headers were present.
LLVM ORC was therefore not prototyped: its deployment cost already fails the
first-backend constraint, while Xbyak demonstrated the required latency and
warm benefit.

## PCSX2 microVU evidence

No PCSX2 emitter or microVU implementation code was copied. The following
mechanisms in PCSX2 revision `897a2d5516787787e8d4236931b8e2804606bf3c`
inform the production design:

- `pcsx2/x86/microVU_Execute.inl` loads VU MXCSR on native entry, restores EE
  MXCSR on exit, and has a distinct XGKICK resume/exit dispatcher.
- `pcsx2/x86/microVU.cpp` clears quick block references after microcode writes,
  retains microprogram copies or compiled ranges, compares current code before
  reuse, and keeps a quick entry-PC lookup.
- `pcsx2/x86/microVU.h` separates microprogram identity, per-entry block
  managers, retained pipeline state, code-cache pointers, and dispatcher
  addresses.
- `pcsx2/x86/microVU_Compile.inl` performs an analysis pass before emission and
  terminates blocks at control-flow boundaries.

PS2Recomp should adopt the entry/exit discipline, analysis-before-emission,
generation-aware lookup, explicit XGKICK boundary, and bounded code cache. It
should initially keep pipeline state dynamic and key blocks by code generation
and entry PC, as already specified in the VU plan. PCSX2's threading, global
emitter state, and full microVU cache representation are not being imported.

## Measurements

Hardware and toolchain:

- AMD Ryzen 7 9800X3D;
- Linux x86-64, CPU 6 pinned with `taskset`;
- `amd-pstate-epp`, `powersave`, 603,379-5,271,622 kHz configured range;
- GCC/G++ 16.1.1, CMake 4.4.0, Ninja 1.13.2;
- Release with PS2Recomp's IPO configuration.

Each process measured 44,000,000 upper-lane pairs per path, 15 warm samples,
101 compile samples, and 200,000 one-pair side exits. Rates are per-process
medians.

| Run | Direct helpers | Helper native | `MULq`/NOP native | Native/direct | Inline compile median | Inline side exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 137.042 Mpair/s | 134.909 Mpair/s | 164.732 Mpair/s | 1.202x | 20.91 us | 9.916 ns |
| 2 | 137.074 Mpair/s | 129.710 Mpair/s | 164.644 Mpair/s | 1.201x | 20.99 us | 9.225 ns |
| 3 | 136.431 Mpair/s | 134.816 Mpair/s | 164.839 Mpair/s | 1.208x | 20.78 us | 9.369 ns |

The helper-only native path ranged from 0.946x to 0.988x of the direct helper
loop, showing that emission alone is not useful. Inlining the measured family
and eliminating decoded NOP calls raised performance by 20.1%-20.8% over the
post-FMAC helper baseline and by 22.1%-26.9% over helper-only emission.

Other measured costs:

- first helper compilation: 37.43-37.88 us;
- first inline compilation: 29.95-37.76 us;
- steady helper compilation median: 16.71-16.83 us;
- steady inline compilation median: 20.78-20.99 us;
- helper block: 3,808 emitted bytes;
- inline block: 4,138 emitted bytes;
- inline one-pair entry/exit increment over a steady native pair:
  3.15-3.85 ns.

Every full run produced state hash `0xa3b335b28006df3e`; every 200,000-exit run
produced `0xa0e0711819005184` and resumed at pair 20.

Retained results:

| Run | Result SHA-256 |
| --- | --- |
| `scratch/runs/20260728-vu-perf-native-emitter-spike-1/result.json` | `418d4dbde67d7445ac6e21660686dce071eaaa5ba41dcdc169d8e938c32bbb7e` |
| `scratch/runs/20260728-vu-perf-native-emitter-spike-2/result.json` | `867134fef7e81257795ff545fedf13766cf8d986fbc1af5eac1d060ea537c7ac` |
| `scratch/runs/20260728-vu-perf-native-emitter-spike-3/result.json` | `c51bba77e3ae322c566e75da819501dbe3f714896b0bc2e90e818c2f16dbd337` |

The Release executable SHA-256 was
`6ee332416db4827059c310aa37472bad318fa3fcb8e5ecad7834c243c80712c2`.
Against the runtime-linked `vu1_replay` utility, the spike added 100,776
unstripped bytes, 86,064 stripped bytes, and 81,944 text bytes. Its main
translation-unit object was 847,712 bytes. `ldd` showed no Xbyak or LLVM
runtime dependency.

## Validation

The self-checking spike passed:

- strict RW-to-RX mapping checks;
- SysV nonvolatile-register preservation;
- MXCSR restoration;
- full-run state equality;
- one-pair side-exit/resume equality;
- ASan, UBSan, and leak checking with an empty diagnostic log.

The sanitizer spike result is
`scratch/runs/20260728-vu-perf-native-emitter-spike-sanitized-2/result.json`
(SHA-256
`6f5f4da5396b888e055a4b63f569c687577e4f9ec5413a60fbf50b14e990952f`).
The complete regression suite also remained 580/580 in Release and under
ASan/UBSan/leak checking:

- Release log SHA-256:
  `cdf33ae7c0d9f76d630403ddb9b5bc289bd987d573ddfe831fd5415a04160be9`;
- sanitizer log SHA-256:
  `cea6774664a1f16cc7ee4a0de67aaaf2499999922dcd49377a3032e559b6ffb1`.

## Reproduction

Configure and build:

```sh
cmake -S PS2Recomp -B scratch/ps2recomp-vu-native-spike \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPS2X_BUILD_STUDIO=OFF \
  -DPS2X_BUILD_VU_NATIVE_EMITTER_SPIKE=ON \
  -DPS2X_XBYAK_INCLUDE_DIR=/absolute/path/to/pcsx2/3rdparty/xbyak
cmake --build scratch/ps2recomp-vu-native-spike \
  --target vu_native_emitter_spike -j 8
```

Run the retained workload from the repository root:

```sh
taskset -c 6 \
  scratch/ps2recomp-vu-native-spike/ps2xTest/vu_native_emitter_spike \
  scratch/vu-perf/phase0a-fixtures/683eafb3192f70c2-entry-0000 \
  44000000 15 101 200000
```

## Production constraints carried into Phase 1

- Keep the interpreter available and make unsupported hosts fall back to it.
- Pin Xbyak and retain its copyright/license in distributed source and binary
  documentation.
- Put allocation/protection/cache maintenance behind a host-code-cache API;
  never expose an RWE default.
- Define separate SysV and Windows x64 entry stubs and test every ABI's
  nonvolatile registers.
- Restore host FP state on every structured exit, including faults and helper
  boundaries.
- Detect host SIMD features. The spike emits SSE4.1 `BLENDPS`; production must
  either require the feature explicitly or provide an SSE2-compatible masked
  write.
- Split production blocks at control-flow and mandatory helper boundaries
  instead of compiling whole invocation traces.
- Replace the spike-only friend access with the canonical cloneable execution
  state and backend-neutral context from Phase 1.
