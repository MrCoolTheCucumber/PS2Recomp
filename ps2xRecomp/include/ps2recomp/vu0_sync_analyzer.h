#ifndef PS2RECOMP_VU0_SYNC_ANALYZER_H
#define PS2RECOMP_VU0_SYNC_ANALYZER_H

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace ps2recomp
{
    struct Instruction;

    // Marks the COP2 operations which must catch an active VU0 microprogram
    // up to EE time. Non-interlocked transfer chains may retain VU registers
    // between those boundaries; interlocked blocks remain fully synchronized.
    void applyVu0SyncPlan(
        std::vector<Instruction> &instructions,
        const std::unordered_set<uint32_t> &entryPoints);
}

#endif // PS2RECOMP_VU0_SYNC_ANALYZER_H
