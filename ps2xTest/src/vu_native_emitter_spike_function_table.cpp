// SPDX-License-Identifier: GPL-3.0-or-later

#include "ps2_runtime.h"

// Standalone runtime tools do not link generated EE code. Keep the generated
// table contract present without the large test table or a game-specific map.
extern const uint32_t g_ps2RecompiledFunctionTableBase = 0u;
extern const uint32_t g_ps2RecompiledFunctionTableEnd = 0u;
extern const uint32_t g_ps2RecompiledFunctionTableSlotCount = 0u;
PS2Runtime::RecompiledFunction g_ps2RecompiledFunctionTable[1] = {};

extern const uint32_t g_ps2GuestFunctionSymbolCount = 0u;
extern const PS2GuestFunctionSymbol g_ps2GuestFunctionSymbols[1] = {};
