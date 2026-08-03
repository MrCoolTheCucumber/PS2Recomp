# EE generated-code fixed-work benchmark

`ee_generated_code_benchmark` isolates the host cost of architectural
observation and EE address translation around a deterministic generated-style
instruction stream. It is a performance fixture, not an alternate EE
interpreter.

The fixture executes ten guest-style instructions per block:

- four scalar arithmetic instructions;
- one 32-bit load and one 32-bit store;
- one COP1 and one COP2 operation;
- one conditional branch; and
- one predicted loop branch.

Every instruction advances the current EE cycle and retires through
`R5900Context::beginEeInstruction`, so `insn_count` and COP0 Random are part of
the fixed result. The precise entry mirrors the breakpoint and PCCR call
shapes emitted by the C++ generator. The benchmark-only fast entry is a
compile-time specialization of the same body in which those calls do not
exist. It is the control for work that a production fast specialization should
eventually approach; it does not disable Random retirement or memory
translation.

## Memory paths

The mapped case installs a valid 4 KiB TLB mapping from virtual
`0x00400000` to physical `0x00800000`, using ASID `0x46`. Loads and stores use
the normal generated `READ32` and `WRITE32` expressions. The fixture poisons
and hashes physical `0x00400000` separately, so an accidental low-address
identity shortcut fails the run.

The direct reference performs the same work through the KSEG0 alias
`0x80800000`. Both paths modify and hash the same physical working set.

## Observation cases

The executable runs these cases in an interleaved, rotating order:

- compiled-out fast entry, mapped and direct memory;
- precise entry with BPC and PCCR inactive, mapped and direct memory;
- matching instruction, data-address, and data-value breakpoints with BED
  suppression;
- PCCR cycle, issue, branch/misprediction, low-order branch, completion,
  coprocessor, and memory occurrence families; and
- a completion counter initialized so the final measured instruction sets
  overflow, followed by exact level-2 delivery outside the timed interval.

Active observation cases are attribution and correctness results. They are not
expected to have the same host cost as inactive execution.

Each JSON-lines measurement includes the final register, physical-memory,
identity-memory, combined-work, and architectural-counter hashes. The program
rejects a run if:

- mapped and direct entries do not produce identical guest work;
- any hash changes between samples;
- `insn_count` or Random differs from the retired instruction count;
- the identity page changes;
- breakpoint sticky bits or PCCR counts differ from the fixed expectation; or
- the final-completion overflow does not publish the performance exception
  vector, ERL state, and `Cause.EXC2` value.

## Build and run

Use a Release build and pin the process to one host CPU when recording a
baseline:

```sh
cmake -S . -B out/benchmark -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPS2X_BUILD_TEST=ON
cmake --build out/benchmark \
  --target ee_generated_code_benchmark --parallel
taskset -c 6 \
  out/benchmark/ps2xTest/ee_generated_code_benchmark \
  --blocks 1000000 --warmup-blocks 20000 --samples 5
```

`--blocks`, `--warmup-blocks`, and `--samples` are optional. Output consists
of one configuration record, one record per case and sample, and a median/min/
max summary for each case.

The two exported entry symbols make the compiled-out property auditable in
the final executable:

```sh
objdump -d -C \
  --disassemble=EeGeneratedBenchmarkFastEntry \
  out/benchmark/ps2xTest/ee_generated_code_benchmark
objdump -d -C \
  --disassemble=EeGeneratedBenchmarkPreciseEntry \
  out/benchmark/ps2xTest/ee_generated_code_benchmark
```

The fast symbol must not call `CheckEeInstructionBreakpoint`,
`CheckEeDataAddressBreakpoint`, `CheckEeDataValueBreakpoint`, any
`recordEeInstruction*` function, or a branch-observation function. The precise
symbol should retain those paths.
