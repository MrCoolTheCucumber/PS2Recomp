# EE Random-retirement benchmark

`ee_random_retirement_benchmark` measures the host cost of counted COP0
Random retirement around a deterministic 64-instruction arithmetic block.
It runs three compile-time variants of the same guest effects:

- `scalar_reference` enters the legacy retirement boundary before every
  instruction;
- `generated_batch` uses the production `retireEeInstructions(64)` operation
  and materializes Random once at the outer block boundary; and
- `direct_block_total` is a benchmark-only control which adds 64 directly to
  `insn_count` and calls the counted Random primitive once.

Every block is an external observation boundary. All variants must therefore
have identical registers, `insn_count`, Random, Wired, PC, and empty pending
state after every measured run. The program rejects any mismatch and emits
JSON-lines measurements plus median/min/max summaries. Cases are interleaved
in rotating order so drift does not consistently favor one implementation.

## Build and run

Use a Release build and pin the process to one host CPU for retained results:

```sh
cmake -S . -B out/random-benchmark -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPS2X_BUILD_TEST=ON
cmake --build out/random-benchmark \
  --target ee_random_retirement_benchmark --parallel
taskset -c 6 \
  out/random-benchmark/ps2xTest/ee_random_retirement_benchmark \
  --blocks 2000000 --warmup-blocks 20000 --samples 5
```

`--case NAME` restricts execution for external host-counter collection. The
production batching gate compares the `generated_batch` median with
`direct_block_total`; it must be within 2% while preserving exact state.

The exported entries also make the native code shape auditable:

```sh
objdump -d -C \
  --disassemble=EeRandomRetirementGeneratedBatch \
  out/random-benchmark/ps2xTest/ee_random_retirement_benchmark
objdump -d -C \
  --disassemble=EeRandomRetirementDirectBlockTotal \
  out/random-benchmark/ps2xTest/ee_random_retirement_benchmark
```

The generated-batch entry may have one counted update and Random
materialization per block. It must not contain the scalar reference's
per-instruction Random update or branch pattern.
