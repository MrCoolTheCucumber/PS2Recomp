#include "ps2recomp/vu0_sync_analyzer.h"

#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"

#include <algorithm>
#include <unordered_set>

namespace ps2recomp
{
    namespace
    {
        bool isCop2Transfer(const Instruction &inst)
        {
            if (inst.opcode != OPCODE_COP2)
            {
                return false;
            }

            return inst.rs == COP2_QMFC2 ||
                   inst.rs == COP2_CFC2 ||
                   inst.rs == COP2_QMTC2 ||
                   inst.rs == COP2_CTC2;
        }

        bool isInterlockedTransfer(const Instruction &inst)
        {
            return isCop2Transfer(inst) && (inst.raw & 1u) != 0u;
        }

        bool isVu0LoadStore(const Instruction &inst)
        {
            return inst.opcode == OPCODE_LQC2 ||
                   inst.opcode == OPCODE_SQC2;
        }

        bool isVu0MacroOperation(const Instruction &inst)
        {
            return inst.opcode == OPCODE_COP2 &&
                   inst.rs >= COP2_CO;
        }

        bool isVu0Call(const Instruction &inst)
        {
            if (!isVu0MacroOperation(inst))
            {
                return false;
            }

            return inst.function == VU0_S1_VCALLMS ||
                   inst.function == VU0_S1_VCALLMSR;
        }

        bool isNonInterlockedMove(const Instruction &inst)
        {
            return isCop2Transfer(inst) &&
                   !isInterlockedTransfer(inst);
        }

        bool isLikelyHandshakeClear(const Instruction &inst)
        {
            return isNonInterlockedMove(inst) &&
                   inst.rs > COP2_CFC2 &&
                   inst.rt == 0u;
        }

        bool followingMacroNeedsFinish(
            const std::vector<Instruction> &instructions,
            size_t begin, size_t end)
        {
            for (size_t index = begin; index < end; ++index)
            {
                const Instruction &inst = instructions[index];
                if (isVu0Call(inst))
                {
                    return false;
                }
                if (isVu0MacroOperation(inst))
                {
                    return true;
                }
            }
            return false;
        }

        std::vector<size_t> findBlockStarts(
            const std::vector<Instruction> &instructions,
            const std::unordered_set<uint32_t> &entryPoints)
        {
            if (instructions.empty())
            {
                return {};
            }

            std::unordered_set<uint32_t> starts = entryPoints;
            starts.insert(instructions.front().address);
            for (size_t index = 0u; index < instructions.size(); ++index)
            {
                const Instruction &inst = instructions[index];
                if (inst.hasDelaySlot)
                {
                    starts.insert(inst.address + 8u);
                }
                if (index != 0u &&
                    inst.address != instructions[index - 1u].address + 4u)
                {
                    starts.insert(inst.address);
                }
            }

            std::vector<size_t> result;
            result.reserve(starts.size());
            for (size_t index = 0u; index < instructions.size(); ++index)
            {
                if (starts.contains(instructions[index].address))
                {
                    result.push_back(index);
                }
            }
            if (result.empty() || result.front() != 0u)
            {
                result.insert(result.begin(), 0u);
            }
            return result;
        }
    }

    void applyVu0SyncPlan(
        std::vector<Instruction> &instructions,
        const std::unordered_set<uint32_t> &entryPoints)
    {
        for (Instruction &inst : instructions)
        {
            inst.vu0SyncMode = Vu0SyncMode::Synchronize;
        }

        const std::vector<size_t> blockStarts =
            findBlockStarts(instructions, entryPoints);
        for (size_t block = 0u; block < blockStarts.size(); ++block)
        {
            const size_t begin = blockStarts[block];
            const size_t end =
                block + 1u < blockStarts.size()
                    ? blockStarts[block + 1u]
                    : instructions.size();
            const bool blockInterlocked = std::any_of(
                instructions.begin() + begin,
                instructions.begin() + end,
                isInterlockedTransfer);

            bool needsVu0Sync = true;
            bool needsVu0Finish = true;
            for (size_t index = begin; index < end; ++index)
            {
                Instruction &inst = instructions[index];

                // A normal EE store may start VIF0/VU0 through DMA. Make the
                // next COP2 access establish a fresh scheduling boundary.
                if (inst.isStore &&
                    inst.opcode != OPCODE_SQC2)
                {
                    needsVu0Sync = true;
                    needsVu0Finish = true;
                    continue;
                }

                if (isVu0Call(inst))
                {
                    inst.vu0SyncMode = Vu0SyncMode::Finish;
                    needsVu0Sync = true;
                    needsVu0Finish = true;
                    continue;
                }

                if (isInterlockedTransfer(inst))
                {
                    inst.vu0SyncMode = Vu0SyncMode::Finish;
                    needsVu0Sync = true;
                    needsVu0Finish = true;
                    continue;
                }

                const bool loadStore = isVu0LoadStore(inst);
                const bool move = isNonInterlockedMove(inst);
                const bool likelyClear = isLikelyHandshakeClear(inst);
                if (loadStore || move)
                {
                    if ((needsVu0Sync && (loadStore || move)) ||
                        likelyClear)
                    {
                        if (!blockInterlocked &&
                            followingMacroNeedsFinish(
                                instructions, index + 1u, end))
                        {
                            inst.vu0SyncMode = Vu0SyncMode::Finish;
                            needsVu0Sync = false;
                            needsVu0Finish = false;
                        }
                        else
                        {
                            inst.vu0SyncMode =
                                Vu0SyncMode::Synchronize;
                            needsVu0Sync =
                                blockInterlocked ||
                                (move && likelyClear);
                            needsVu0Finish = true;
                        }
                    }
                    else
                    {
                        inst.vu0SyncMode = Vu0SyncMode::Skip;
                    }
                    continue;
                }

                if (isVu0MacroOperation(inst))
                {
                    inst.vu0SyncMode = Vu0SyncMode::Finish;
                    needsVu0Sync = false;
                    needsVu0Finish = false;
                    continue;
                }

                if (inst.opcode == OPCODE_COP2)
                {
                    if (needsVu0Finish && inst.rs >= COP2_CO)
                    {
                        inst.vu0SyncMode = Vu0SyncMode::Finish;
                        needsVu0Sync = false;
                        needsVu0Finish = false;
                    }
                    else if (needsVu0Sync)
                    {
                        inst.vu0SyncMode =
                            Vu0SyncMode::Synchronize;
                        needsVu0Sync = blockInterlocked;
                    }
                    else
                    {
                        inst.vu0SyncMode = Vu0SyncMode::Skip;
                    }
                }
            }
        }
    }
}
