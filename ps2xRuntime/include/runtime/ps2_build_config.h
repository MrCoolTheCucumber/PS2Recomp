#ifndef PS2X_RUNTIME_BUILD_CONFIG_H
#define PS2X_RUNTIME_BUILD_CONFIG_H

// CMake publishes this definition to every runtime consumer. Keep diagnostics
// enabled for non-CMake embedders so including the public headers preserves
// the historical API unless they explicitly select a production build.
#ifndef PS2X_ENABLE_RUNTIME_DIAGNOSTICS
#define PS2X_ENABLE_RUNTIME_DIAGNOSTICS 1
#endif

#endif // PS2X_RUNTIME_BUILD_CONFIG_H
