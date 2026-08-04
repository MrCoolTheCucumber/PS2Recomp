#include "ps2recomp/control_flow_analyzer.h"
#include "ps2recomp/ee_observation_mode.h"
#include "ps2recomp/gif_dma_kick_analyzer.h"
#include "ps2recomp/types.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/control_flow_utils.h"
#include "ps2recomp/recompiler_reporter.h"

#include <algorithm>
#include <cstring>

namespace ps2recomp
{
    ControlFlowAnalyzer::ControlFlowAnalyzer(
        const std::vector<Section> &sections,
        const std::unordered_map<uint32_t, std::vector<uint32_t>> &configuredJumpTableTargetsByAddress,
        const std::unordered_map<uint32_t, std::vector<uint32_t>> &configuredIndirectBranchTargetsByInstructionAddress,
        RecompilerReporter *reporter)
        : m_sections(sections),
          m_configJumpTableTargetsByAddress(configuredJumpTableTargetsByAddress),
          m_configIndirectBranchTargetsByInstructionAddress(
              configuredIndirectBranchTargetsByInstructionAddress),
          m_reporter(reporter)
    {
    }

    ControlFlowAnalysisResult ControlFlowAnalyzer::analyze(
        const Function &function,
        const std::vector<Instruction> &instructions,
        const std::vector<Function> *allFunctions) const
    {
        ControlFlowAnalysisResult result;
        std::unordered_set<uint32_t> instructionAddresses;
        instructionAddresses.reserve(instructions.size());
        std::unordered_map<uint32_t, const Instruction *>
            instructionsByAddress;
        instructionsByAddress.reserve(instructions.size());
        bool hasIndirectRegisterJump = false;
        std::vector<const Instruction *> indirectJumps;

        auto isExecutableAddress = [&](uint32_t address) -> bool
        {
            for (const auto &section : m_sections)
            {
                if (!section.isCode)
                {
                    continue;
                }
                if (address >= section.address && address < (section.address + section.size))
                {
                    return true;
                }
            }
            return false;
        };

        auto findContainingExternalFunction = [&](uint32_t address) -> const Function *
        {
            if (!allFunctions || !isExecutableAddress(address))
            {
                return nullptr;
            }

            const Function *best = nullptr;
            for (const auto &candidateFn : *allFunctions)
            {
                if (!candidateFn.isRecompiled || candidateFn.isStub || candidateFn.isSkipped)
                {
                    continue;
                }

                if (candidateFn.name.rfind("entry_", 0) == 0)
                {
                    continue;
                }

                if (address < candidateFn.start || address >= candidateFn.end)
                {
                    continue;
                }

                if (!best || candidateFn.start > best->start)
                {
                    best = &candidateFn;
                }
            }

            return best;
        };

        auto findContainingCallableFunction = [&](uint32_t address) -> const Function *
        {
            if (!allFunctions || !isExecutableAddress(address))
            {
                return nullptr;
            }

            const Function *best = nullptr;
            for (const auto &candidateFn : *allFunctions)
            {
                if (candidateFn.isSkipped ||
                    (!candidateFn.isRecompiled && !candidateFn.isStub))
                {
                    continue;
                }
                if (candidateFn.isStub && address != candidateFn.start)
                {
                    continue;
                }
                if (address < candidateFn.start || address >= candidateFn.end)
                {
                    continue;
                }
                if (!best || candidateFn.start > best->start)
                {
                    best = &candidateFn;
                }
            }
            return best;
        };

        auto queueExternalEntryTarget = [&](uint32_t target)
        {
            const Function *containingFn = findContainingExternalFunction(target);
            if (!containingFn)
            {
                return;
            }

            if (containingFn->start == function.start)
            {
                return;
            }

            if (target == containingFn->start)
            {
                return;
            }

            result.externalEntryPoints.insert(target);
        };

        auto queueResumeEntryTarget = [&](uint32_t resumeAddr)
        {
            if (resumeAddr >= function.start && resumeAddr < function.end &&
                instructionAddresses.contains(resumeAddr))
            {
                result.entryPoints.insert(resumeAddr);
                result.resumeEntryPoints.insert(resumeAddr);
            }
        };

        auto queueLoopResumeEntryTarget = [&](uint32_t target, uint32_t sourcePc)
        {
            if (target > sourcePc || target == function.start)
            {
                return;
            }

            queueResumeEntryTarget(target);
        };

        auto isRegimmBranchAndLink = [](const Instruction &inst)
        {
            if (inst.opcode != OPCODE_REGIMM)
            {
                return false;
            }

            return inst.rt == REGIMM_BLTZAL || inst.rt == REGIMM_BGEZAL ||
                   inst.rt == REGIMM_BLTZALL || inst.rt == REGIMM_BGEZALL;
        };

        const auto writesRegister = [](const Instruction &inst, uint32_t reg)
        {
            return reg != 0u &&
                   (inst.gprWriteMask & (1u << reg)) != 0u;
        };

        const auto isExactReturnAddressCopy = [](const Instruction &inst,
                                                 uint32_t destinationReg)
        {
            if (inst.rt == destinationReg && inst.rs == 31u &&
                inst.simmediate == 0u &&
                (inst.opcode == OPCODE_ADDI ||
                 inst.opcode == OPCODE_ADDIU ||
                 inst.opcode == OPCODE_DADDI ||
                 inst.opcode == OPCODE_DADDIU))
            {
                return true;
            }

            if (inst.opcode != OPCODE_SPECIAL ||
                inst.rd != destinationReg ||
                (inst.function != SPECIAL_OR &&
                 inst.function != SPECIAL_ADDU &&
                 inst.function != SPECIAL_DADDU))
            {
                return false;
            }

            return (inst.rs == 31u && inst.rt == 0u) ||
                   (inst.rs == 0u && inst.rt == 31u);
        };

        for (const auto &inst : instructions)
        {
            instructionAddresses.insert(inst.address);
            instructionsByAddress.emplace(inst.address, &inst);
            if (inst.opcode == OPCODE_SPECIAL &&
                ((inst.function == SPECIAL_JR && inst.rs != 31) ||
                 inst.function == SPECIAL_JALR))
            {
                hasIndirectRegisterJump = true;
                indirectJumps.push_back(&inst);
            }
        }

        auto hasObservationModeChangingDelaySlot =
            [&](const Instruction &inst)
        {
            if (!inst.hasDelaySlot)
            {
                return false;
            }
            const auto delay =
                instructionsByAddress.find(inst.address + 4u);
            return delay != instructionsByAddress.end() &&
                   mayChangeEeArchitecturalObservationMode(
                       *delay->second);
        };

        // A specialization transition after an ordinary MTC0 resumes at the
        // following instruction. Delay-slot writes also use this fallthrough
        // for the not-taken path of likely branches; resolved taken targets
        // are added with their owning control-flow instruction below.
        for (const Instruction &inst : instructions)
        {
            if (mayChangeEeArchitecturalObservationMode(inst))
            {
                queueResumeEntryTarget(inst.address + 4u);
            }
        }

        for (const auto &inst : instructions)
        {
            bool isStaticJump = (inst.opcode == OPCODE_J || inst.opcode == OPCODE_JAL);
            if (inst.isBranch && inst.opcode != OPCODE_J && inst.opcode != OPCODE_JAL)
            {
                const bool transitionDelay =
                    hasObservationModeChangingDelaySlot(inst);
                const int32_t offsetBytes = (static_cast<int32_t>(static_cast<int16_t>(inst.simmediate)) << 2);
                const uint32_t target = static_cast<uint32_t>(
                    static_cast<int64_t>(inst.address + 4u) + static_cast<int64_t>(offsetBytes));

                if (target >= function.start && target < function.end &&
                    instructionAddresses.contains(target))
                {
                    result.entryPoints.insert(target);
                    queueLoopResumeEntryTarget(target, inst.address);
                    if (transitionDelay)
                    {
                        queueResumeEntryTarget(target);
                    }
                }
                else
                {
                    queueExternalEntryTarget(target);
                }

                if (isRegimmBranchAndLink(inst))
                {
                    queueResumeEntryTarget(inst.address + 8u);
                }
                else if (transitionDelay)
                {
                    queueResumeEntryTarget(inst.address + 8u);
                }
            }
            else if (isStaticJump)
            {
                const bool transitionDelay =
                    hasObservationModeChangingDelaySlot(inst);
                uint32_t target = buildAbsoluteJumpTarget(inst.address, inst.target);
                if (target >= function.start && target < function.end &&
                    instructionAddresses.contains(target))
                {
                    result.entryPoints.insert(target);
                    queueLoopResumeEntryTarget(target, inst.address);
                    if (transitionDelay)
                    {
                        queueResumeEntryTarget(target);
                    }

                    if (inst.opcode == OPCODE_JAL)
                    {
                        queueResumeEntryTarget(inst.address + 8u);
                    }
                }
                else
                {
                    queueExternalEntryTarget(target);

                    if (inst.opcode == OPCODE_JAL)
                    {
                        queueResumeEntryTarget(inst.address + 8u);
                    }
                }
            }
        }

        // SetupThread installs a guest function pointer which the kernel later
        // copies into every started thread's $ra. That target is control flow
        // even though the ELF contains no J/JAL reference to it. Track only
        // constants proven within one basic block so alternate paths or calls
        // cannot manufacture a false entry point.
        {
            constexpr uint32_t kSyscallRegister = 3u;
            constexpr uint32_t kThreadRootRegister = 8u;
            constexpr uint32_t kSetupThreadSyscall = 0x3Cu;

            ConstantRegisterState constants;
            bool clearAfterDelaySlot = false;
            for (const Instruction &inst : instructions)
            {
                const bool isDelaySlot =
                    clearAfterDelaySlot;
                if (inst.address != function.start &&
                    result.entryPoints.contains(
                        inst.address))
                {
                    constants.clear();
                }

                if (inst.opcode == OPCODE_SPECIAL &&
                    inst.function == SPECIAL_SYSCALL)
                {
                    const uint32_t encodedSyscall =
                        (inst.raw >> 6) & 0xFFFFFu;
                    uint32_t syscallNumber =
                        encodedSyscall;
                    const bool syscallKnown =
                        encodedSyscall != 0u ||
                        constants.read(
                            kSyscallRegister,
                            syscallNumber);
                    uint32_t root = 0u;
                    if (syscallKnown &&
                        syscallNumber ==
                            kSetupThreadSyscall &&
                        constants.read(
                            kThreadRootRegister,
                            root) &&
                        (root & 3u) == 0u &&
                        isExecutableAddress(root))
                    {
                        result
                            .architecturalEntryPoints
                            .insert(root);
                    }
                }

                updateConstantRegisters(
                    inst,
                    constants);
                if (isDelaySlot)
                {
                    constants.clear();
                }
                clearAfterDelaySlot =
                    inst.hasDelaySlot;
            }
        }

        if (hasIndirectRegisterJump)
        {
            std::vector<uint32_t> unresolvedJumpAddresses;
            unresolvedJumpAddresses.reserve(indirectJumps.size());
            for (const Instruction *jrInst : indirectJumps)
            {
                if (jrInst->function == SPECIAL_JALR)
                {
                    queueResumeEntryTarget(jrInst->address + 8u);
                }

                bool foundTable = false;

                const auto configuredBranchIt =
                    m_configIndirectBranchTargetsByInstructionAddress.find(
                        jrInst->address);
                if (configuredBranchIt !=
                    m_configIndirectBranchTargetsByInstructionAddress.end())
                {
                    bool validTargets = !configuredBranchIt->second.empty();
                    std::vector<uint32_t> localTargets;
                    std::vector<uint32_t> externalTargets;
                    for (uint32_t target : configuredBranchIt->second)
                    {
                        if ((target & 3u) != 0u ||
                            !isExecutableAddress(target))
                        {
                            validTargets = false;
                            break;
                        }

                        const bool isLocalTarget =
                            target >= function.start &&
                            target < function.end;
                        if (isLocalTarget)
                        {
                            if (!instructionAddresses.contains(target))
                            {
                                validTargets = false;
                                break;
                            }
                            localTargets.push_back(target);
                        }
                        else
                        {
                            externalTargets.push_back(target);
                        }
                    }

                    if (validTargets)
                    {
                        for (uint32_t target : externalTargets)
                        {
                            queueExternalEntryTarget(target);
                        }
                        if (!localTargets.empty())
                        {
                            result.jumpTableTargets[jrInst->address] =
                                localTargets;
                            for (uint32_t target : localTargets)
                            {
                                result.entryPoints.insert(target);
                                if (hasObservationModeChangingDelaySlot(
                                        *jrInst))
                                {
                                    queueResumeEntryTarget(target);
                                }
                            }
                        }
                        foundTable = true;
                    }
                }

                uint32_t jrReg = jrInst->rs;

                int lwIndex = -1;
                uint32_t baseReg = 0;
                int32_t lwOffset = 0;

                auto it = std::find_if(instructions.begin(), instructions.end(), [&](const Instruction &inst)
                                       { return inst.address == jrInst->address; });
                if (it != instructions.end())
                {
                    int jrIndex = std::distance(instructions.begin(), it);

                    // Hand-written EE code sometimes preserves the incoming
                    // return address in a temporary register before making
                    // nested calls, then returns with JR through that alias.
                    // Accept only an exact copy in the straight-line entry
                    // block, before any instruction which could change $ra
                    // or make the copy path-dependent, which is never
                    // overwritten locally.
                    if (!foundTable &&
                        jrInst->function == SPECIAL_JR &&
                        jrReg != 0u && jrReg != 31u)
                    {
                        int aliasDefinitionIndex = -1;
                        for (int i = 0; i < jrIndex; ++i)
                        {
                            const auto &inst = instructions[i];
                            if (inst.hasDelaySlot || inst.isBranch ||
                                inst.isJump || inst.isCall || inst.isReturn ||
                                writesRegister(inst, 31u))
                            {
                                break;
                            }
                            if (isExactReturnAddressCopy(inst, jrReg))
                            {
                                aliasDefinitionIndex = i;
                                break;
                            }
                        }

                        if (aliasDefinitionIndex >= 0)
                        {
                            bool aliasWasOverwritten = false;
                            for (int i = aliasDefinitionIndex + 1;
                                 i < jrIndex; ++i)
                            {
                                if (writesRegister(instructions[i], jrReg))
                                {
                                    aliasWasOverwritten = true;
                                    break;
                                }
                            }
                            if (!aliasWasOverwritten)
                            {
                                result.returnAddressAliasJumps.insert(
                                    jrInst->address);
                                foundTable = true;
                            }
                        }
                    }

                    if (foundTable)
                    {
                        continue;
                    }

                    for (int i = jrIndex - 1; i >= 0 && i >= jrIndex - 20; --i)
                    {
                        const auto &inst = instructions[i];
                        if ((inst.opcode == OPCODE_LW || inst.opcode == OPCODE_LWU) && inst.rt == jrReg)
                        {
                            lwIndex = i;
                            baseReg = inst.rs;
                            lwOffset = inst.simmediate;
                            break;
                        }
                    }

                    if (lwIndex != -1)
                    {
                        int adduIndex = -1;
                        uint32_t tableBaseReg = 0;
                        uint32_t indexReg = 0;
                        for (int i = lwIndex - 1; i >= 0 && i >= lwIndex - 10; --i)
                        {
                            const auto &inst = instructions[i];
                            if (inst.opcode == OPCODE_SPECIAL && inst.function == SPECIAL_ADDU && inst.rd == baseReg)
                            {
                                adduIndex = i;
                                tableBaseReg = inst.rs;
                                indexReg = inst.rt;
                                break;
                            }
                        }

                        uint32_t tableAddress = 0;
                        bool foundTableAddress = false;

                        if (adduIndex != -1)
                        {
                            for (int i = adduIndex - 1; i >= 0 && i >= adduIndex - 20; --i)
                            {
                                const auto &inst = instructions[i];
                                if (inst.opcode == OPCODE_LUI)
                                {
                                    if (inst.rt == tableBaseReg || inst.rt == indexReg)
                                    {
                                        uint32_t high = inst.immediate << 16;
                                        uint32_t low = 0;
                                        for (int j = i + 1; j < adduIndex; ++j)
                                        {
                                            const auto &lowInst = instructions[j];
                                            if (lowInst.rs == inst.rt && lowInst.rt == inst.rt)
                                            {
                                                if (lowInst.opcode == OPCODE_ADDIU)
                                                {
                                                    low = (uint32_t)lowInst.simmediate;
                                                }
                                                else if (lowInst.opcode == OPCODE_ORI)
                                                {
                                                    low = lowInst.immediate;
                                                }
                                            }
                                        }
                                        tableAddress = high + low;
                                        foundTableAddress = true;
                                        break;
                                    }
                                }
                            }
                        }

                        if (foundTableAddress)
                        {
                            tableAddress += lwOffset;

                            const auto configuredTableIt = m_configJumpTableTargetsByAddress.find(tableAddress);
                            if (configuredTableIt != m_configJumpTableTargetsByAddress.end())
                            {
                                std::vector<uint32_t> jrTargets;
                                jrTargets.reserve(configuredTableIt->second.size());
                                for (uint32_t target : configuredTableIt->second)
                                {
                                    if (target >= function.start && target < function.end &&
                                        instructionAddresses.contains(target))
                                    {
                                        jrTargets.push_back(target);
                                    }
                                    else
                                    {
                                        queueExternalEntryTarget(target);
                                    }
                                }

                                if (!jrTargets.empty())
                                {
                                    std::sort(jrTargets.begin(), jrTargets.end());
                                    jrTargets.erase(std::unique(jrTargets.begin(), jrTargets.end()), jrTargets.end());
                                    result.jumpTableTargets[jrInst->address] = jrTargets;
                                    for (uint32_t target : jrTargets)
                                    {
                                        result.entryPoints.insert(target);
                                        if (hasObservationModeChangingDelaySlot(
                                                *jrInst))
                                        {
                                            queueResumeEntryTarget(target);
                                        }
                                    }
                                    foundTable = true;
                                }
                            }

                            struct ScaledIndexDefinition
                            {
                                uint32_t destinationRegister = 0;
                                uint32_t sourceRegister = 0;
                                uint32_t strideBytes = 0;
                                int instructionIndex = -1;
                            };

                            constexpr int kMaxJumpTableDataFlowInstructions = 64;
                            constexpr uint32_t kMaxInferredTableEntries = 1000u;
                            struct RegisterConstant
                            {
                                bool found = false;
                                uint32_t value = 0;
                            };
                            const auto findRegisterConstantBefore =
                                [&](uint32_t reg, int beforeIndex)
                                {
                                    RegisterConstant constant;
                                    if (reg == 0u)
                                    {
                                        constant.found = true;
                                        return constant;
                                    }

                                    for (int i = beforeIndex - 1;
                                         i >= 0 &&
                                         i >= beforeIndex - kMaxJumpTableDataFlowInstructions;
                                         --i)
                                    {
                                        const auto &inst = instructions[i];
                                        if (!writesRegister(inst, reg))
                                        {
                                            continue;
                                        }

                                        if (inst.rt == reg && inst.rs == 0u)
                                        {
                                            if (inst.opcode == OPCODE_ADDIU ||
                                                inst.opcode == OPCODE_DADDIU)
                                            {
                                                constant.found = true;
                                                constant.value = inst.simmediate;
                                            }
                                            else if (inst.opcode == OPCODE_ORI)
                                            {
                                                constant.found = true;
                                                constant.value = inst.immediate;
                                            }
                                        }
                                        return constant;
                                    }
                                    return constant;
                                };
                            const auto isUsableTableStride = [](RegisterConstant constant)
                            {
                                return constant.found &&
                                       constant.value >= sizeof(uint32_t) &&
                                       constant.value <= 0x10000u &&
                                       (constant.value & 3u) == 0u;
                            };
                            const auto findScaledIndexDefinition =
                                [&](uint32_t shiftedRegister)
                                {
                                    ScaledIndexDefinition definition;
                                    if (shiftedRegister == 0u)
                                    {
                                        return definition;
                                    }

                                    for (int i = adduIndex - 1;
                                         i >= 0 &&
                                         i >= adduIndex - kMaxJumpTableDataFlowInstructions;
                                         --i)
                                    {
                                        const auto &inst = instructions[i];
                                        if (!writesRegister(inst, shiftedRegister))
                                        {
                                            continue;
                                        }

                                        if (inst.opcode == OPCODE_SPECIAL &&
                                            inst.function == SPECIAL_SLL &&
                                            inst.rd == shiftedRegister &&
                                            inst.rt != 0u &&
                                            inst.sa >= 2u &&
                                            inst.sa < 31u)
                                        {
                                            definition.destinationRegister =
                                                shiftedRegister;
                                            definition.sourceRegister = inst.rt;
                                            definition.strideBytes = 1u << inst.sa;
                                            definition.instructionIndex = i;
                                        }
                                        else if (inst.opcode == OPCODE_SPECIAL &&
                                                 (inst.function == SPECIAL_MULT ||
                                                  inst.function == SPECIAL_MULTU) &&
                                                 inst.rd == shiftedRegister)
                                        {
                                            const RegisterConstant rsConstant =
                                                findRegisterConstantBefore(inst.rs, i);
                                            const RegisterConstant rtConstant =
                                                findRegisterConstantBefore(inst.rt, i);
                                            const bool rsIsStride =
                                                isUsableTableStride(rsConstant);
                                            const bool rtIsStride =
                                                isUsableTableStride(rtConstant);
                                            if (rsIsStride != rtIsStride)
                                            {
                                                definition.destinationRegister =
                                                    shiftedRegister;
                                                definition.sourceRegister =
                                                    rsIsStride ? inst.rt : inst.rs;
                                                definition.strideBytes =
                                                    rsIsStride
                                                        ? rsConstant.value
                                                        : rtConstant.value;
                                                definition.instructionIndex = i;
                                            }
                                        }
                                        return definition;
                                    }
                                    return definition;
                                };

                            ScaledIndexDefinition scaledIndex =
                                findScaledIndexDefinition(tableBaseReg);
                            if (scaledIndex.instructionIndex < 0)
                            {
                                scaledIndex =
                                    findScaledIndexDefinition(indexReg);
                            }

                            uint32_t numCases = 0;
                            if (scaledIndex.instructionIndex >= 0)
                            {
                                for (int i = adduIndex - 1;
                                     i >= 0 &&
                                     i >= adduIndex - kMaxJumpTableDataFlowInstructions;
                                     --i)
                                {
                                    const auto &inst = instructions[i];
                                    if ((inst.opcode != OPCODE_SLTIU &&
                                         inst.opcode != OPCODE_SLTI) ||
                                        inst.rs != scaledIndex.sourceRegister)
                                    {
                                        continue;
                                    }

                                    if (i > scaledIndex.instructionIndex &&
                                        scaledIndex.sourceRegister ==
                                            scaledIndex.destinationRegister)
                                    {
                                        continue;
                                    }

                                    // The bound and SLL must observe the same
                                    // source value. A guard may follow the SLL
                                    // (as long as the source was not changed),
                                    // but a guard which overwrites the source
                                    // before the SLL cannot bound that index.
                                    bool sourceWasClobbered = false;
                                    const int firstCheckedIndex =
                                        i < scaledIndex.instructionIndex
                                            ? i
                                            : scaledIndex.instructionIndex + 1;
                                    const int lastCheckedIndex =
                                        i < scaledIndex.instructionIndex
                                            ? scaledIndex.instructionIndex
                                            : i;
                                    for (int j = firstCheckedIndex;
                                         j < lastCheckedIndex; ++j)
                                    {
                                        if (writesRegister(
                                                instructions[j],
                                                scaledIndex.sourceRegister))
                                        {
                                            sourceWasClobbered = true;
                                            break;
                                        }
                                    }
                                    if (!sourceWasClobbered)
                                    {
                                        numCases = inst.immediate;
                                        break;
                                    }
                                }
                            }

                            if (!foundTable && numCases > 0 &&
                                numCases <= kMaxInferredTableEntries)
                            {
                                const Section *rodata = nullptr;
                                for (const auto &sec : m_sections)
                                {
                                    if (tableAddress >= sec.address && tableAddress < sec.address + sec.size)
                                    {
                                        rodata = &sec;
                                        break;
                                    }
                                }

                                if (rodata && rodata->data)
                                {
                                    std::vector<uint32_t> jrTargets;
                                    bool validJumpTable = true;
                                    std::unordered_set<uint32_t> uniqueTargets;
                                    for (uint32_t i = 0; i < numCases; ++i)
                                    {
                                        const uint64_t wideAddress =
                                            static_cast<uint64_t>(tableAddress) +
                                            static_cast<uint64_t>(i) *
                                                scaledIndex.strideBytes;
                                        if (wideAddress > UINT32_MAX)
                                        {
                                            validJumpTable = false;
                                            break;
                                        }
                                        const uint32_t addr =
                                            static_cast<uint32_t>(wideAddress);
                                        if (addr >= rodata->address && addr + 4 <= rodata->address + rodata->size)
                                        {
                                            uint32_t target = 0;
                                            std::memcpy(&target, rodata->data + (addr - rodata->address), 4);
                                            if (target >= function.start && target < function.end && instructionAddresses.contains(target))
                                            {
                                                if (!uniqueTargets.contains(target))
                                                {
                                                    jrTargets.push_back(target);
                                                    uniqueTargets.insert(target);
                                                }
                                            }
                                            else
                                            {
                                                queueExternalEntryTarget(target);
                                            }
                                        }
                                        else
                                        {
                                            validJumpTable = false;
                                            break;
                                        }
                                    }
                                    if (validJumpTable && !jrTargets.empty())
                                    {
                                        result.jumpTableTargets[jrInst->address] = jrTargets;
                                        for (uint32_t t : jrTargets)
                                        {
                                            result.entryPoints.insert(t);
                                            if (hasObservationModeChangingDelaySlot(
                                                    *jrInst))
                                            {
                                                queueResumeEntryTarget(t);
                                            }
                                        }
                                        foundTable = true;
                                    }
                                }
                            }

                            // Read-only sections frequently encode vtables as
                            // fixed-size records rather than dense switch
                            // tables. When a JALR loads one field from each
                            // record, the section boundary supplies the table
                            // extent. Accept the inference only when every
                            // non-null field is a known executable target.
                            if (!foundTable &&
                                jrInst->function == SPECIAL_JALR &&
                                scaledIndex.instructionIndex >= 0)
                            {
                                const Section *tableSection = nullptr;
                                for (const auto &sec : m_sections)
                                {
                                    if (sec.isReadOnly && sec.data &&
                                        tableAddress >= sec.address &&
                                        tableAddress < sec.address + sec.size)
                                    {
                                        tableSection = &sec;
                                        break;
                                    }
                                }

                                if (tableSection &&
                                    scaledIndex.strideBytes >= sizeof(uint32_t))
                                {
                                    const uint32_t fieldOffset =
                                        tableAddress - tableSection->address;
                                    const uint64_t sectionEnd =
                                        static_cast<uint64_t>(tableSection->address) +
                                        tableSection->size;
                                    const bool firstFieldFits =
                                        static_cast<uint64_t>(tableAddress) +
                                            sizeof(uint32_t) <=
                                        sectionEnd;
                                    const uint32_t recordCount =
                                        firstFieldFits
                                            ? 1u + static_cast<uint32_t>(
                                                       (sectionEnd - tableAddress -
                                                        sizeof(uint32_t)) /
                                                       scaledIndex.strideBytes)
                                            : 0u;
                                    if (fieldOffset < scaledIndex.strideBytes &&
                                        recordCount > 0u &&
                                        recordCount <= kMaxInferredTableEntries)
                                    {
                                        bool validFunctionTable = true;
                                        bool foundCallableTarget = false;
                                        std::vector<uint32_t> localTargets;
                                        std::unordered_set<uint32_t> uniqueTargets;
                                        for (uint32_t i = 0; i < recordCount; ++i)
                                        {
                                            const uint64_t wideAddress =
                                                static_cast<uint64_t>(tableAddress) +
                                                static_cast<uint64_t>(i) *
                                                    scaledIndex.strideBytes;
                                            if (wideAddress + sizeof(uint32_t) >
                                                sectionEnd)
                                            {
                                                validFunctionTable = false;
                                                break;
                                            }
                                            const uint32_t addr =
                                                static_cast<uint32_t>(wideAddress);
                                            uint32_t target = 0;
                                            std::memcpy(
                                                &target,
                                                tableSection->data +
                                                    (addr - tableSection->address),
                                                sizeof(target));
                                            if (target == 0u)
                                            {
                                                continue;
                                            }
                                            if ((target & 3u) != 0u ||
                                                !isExecutableAddress(target))
                                            {
                                                validFunctionTable = false;
                                                break;
                                            }

                                            const Function *targetOwner =
                                                findContainingCallableFunction(target);
                                            if (allFunctions && !targetOwner)
                                            {
                                                validFunctionTable = false;
                                                break;
                                            }

                                            foundCallableTarget = true;
                                            const bool isLocalTarget =
                                                target >= function.start &&
                                                target < function.end &&
                                                instructionAddresses.contains(target);
                                            if (targetOwner &&
                                                targetOwner->start == function.start &&
                                                !isLocalTarget)
                                            {
                                                validFunctionTable = false;
                                                break;
                                            }
                                            if (isLocalTarget)
                                            {
                                                if (uniqueTargets.insert(target).second)
                                                {
                                                    localTargets.push_back(target);
                                                }
                                            }
                                            else
                                            {
                                                queueExternalEntryTarget(target);
                                            }
                                        }

                                        if (validFunctionTable &&
                                            foundCallableTarget)
                                        {
                                            if (!localTargets.empty())
                                            {
                                                result.jumpTableTargets[
                                                    jrInst->address] = localTargets;
                                                for (uint32_t target : localTargets)
                                                {
                                                    result.entryPoints.insert(target);
                                                    if (hasObservationModeChangingDelaySlot(
                                                            *jrInst))
                                                    {
                                                        queueResumeEntryTarget(target);
                                                    }
                                                }
                                            }
                                            foundTable = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (!foundTable)
                {
                    unresolvedJumpAddresses.push_back(jrInst->address);
                }
            }

            if (!unresolvedJumpAddresses.empty())
            {
                if (m_reporter)
                {
                    m_reporter->recordIndirectFallbackPromotion(
                        function.name,
                        unresolvedJumpAddresses,
                        instructionAddresses.size());
                }

                for (uint32_t addr : instructionAddresses)
                {
                    if (addr >= function.start && addr < function.end)
                    {
                        result.entryPoints.insert(addr);
                        // Keep labels and runtime registration for unresolved JR/JALR targets
                        // without emitting a local switch over every possible target.
                        result.indirectFallbackEntryPoints.insert(addr);
                    }
                }
            }
        }

        return result;
    }

}
