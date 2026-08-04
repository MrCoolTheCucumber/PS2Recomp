#include "ps2_runtime.h"
#include "runtime/ps2_memory.h"

extern const uint32_t g_ps2RecompiledFunctionPairAbiVersion = PS2_RECOMPILED_FUNCTION_PAIR_ABI_VERSION;
extern const uint32_t g_ps2RecompiledFunctionTableBase = 0x00000000u;
extern const uint32_t g_ps2RecompiledFunctionTableEnd = 0x01000000u;
extern const uint32_t g_ps2RecompiledFunctionTableSlotCount = (g_ps2RecompiledFunctionTableEnd - g_ps2RecompiledFunctionTableBase) >> 2;
PS2Runtime::RecompiledFunctionPair g_ps2RecompiledFunctionTable[g_ps2RecompiledFunctionTableSlotCount] = {};

extern const uint32_t g_ps2GuestFunctionSymbolCount = 0u;
extern const PS2GuestFunctionSymbol g_ps2GuestFunctionSymbols[1] = {};
