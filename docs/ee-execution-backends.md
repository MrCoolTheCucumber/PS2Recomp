# EE execution backends

The native runtime currently keeps `legacy-host-thread` as its default while
the one-executor fiber mode is stabilized. Select the fiber backend explicitly
for testing or differential runs:

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
the selected backend, Boost version, target architecture, ABI, and context
implementation.

Production Linux x86-64 and Windows x64 builds use `fcontext`. Sanitizer
results must identify their context mode explicitly: the no-fcontext fallback
can validate scheduler, executor, and publication code, but it is not
coverage of a production context switch. A sanitizer-only `ucontext` build
may supplement production testing in the future; it must not replace
`fcontext` correctness or performance gates.
