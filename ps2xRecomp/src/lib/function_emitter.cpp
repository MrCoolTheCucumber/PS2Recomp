#include "ps2recomp/Emitters/function_emitter.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/ee_cycle_model.h"
#include "ps2recomp/ee_observation_mode.h"
#include "ps2recomp/ee_performance_model.h"
#include "ps2recomp/gif_dma_kick_analyzer.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/r5900_decoder.h"
#include "ps2recomp/recompiler_reporter.h"
#include "ps2recomp/types.h"
#include "ps2recomp/vu0_sync_analyzer.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace ps2recomp
{
    namespace
    {
        Instruction makeSyntheticDelaySlot(uint32_t address)
        {
            Instruction inst{};
            inst.address = address;
            inst.raw = 0;
            inst.opcode = OPCODE_SPECIAL;
            inst.function = SPECIAL_SLL;
            return inst;
        }

        bool changesEeAddressMode(const Instruction &instruction)
        {
            return instruction.opcode == OPCODE_COP0 &&
                   instruction.rs == COP0_MT &&
                   instruction.rd == COP0_REG_STATUS;
        }
    }

    FunctionEmitter::FunctionEmitter(CodeGenerator &codeGenerator)
        : m_codeGenerator(codeGenerator)
    {
    }

    std::string FunctionEmitter::emit(
        const Function &function,
        const std::vector<Instruction> &instructions,
        bool useHeaders)
    {
        CodeGenerator &cg = m_codeGenerator;
        std::stringstream ss;
        cg.m_currentFunctionName = function.name;

        if (useHeaders)
        {
            ss << "#include <stdexcept>\n";
            ss << "#include \"ps2_runtime_macros.h\"\n";
            ss << "#include \"ps2_runtime.h\"\n";
            ss << "#include \"ps2_recompiled_functions.h\"\n";
            ss << "#include \"ps2_recompiled_stubs.h\"\n\n";
            ss << "#include \"ps2_syscalls.h\"\n";
            ss << "#include \"ps2_stubs.h\"\n\n";
            ss << "#ifdef PS2_FUNCTION_LOG_TRACKER\n";
            ss << "#include \"ps2_log.h\"\n";
            ss << "#endif\n\n";
        }

        CodeGenerator::AnalysisResult analysisResult = cg.collectInternalBranchTargets(function, instructions);
        std::vector<uint32_t> resumeTargets(analysisResult.resumeEntryPoints.begin(),
                                            analysisResult.resumeEntryPoints.end());
        auto resumeIt = cg.m_resumeEntryTargetsByOwner.find(function.start);
        if (resumeIt != cg.m_resumeEntryTargetsByOwner.end())
        {
            resumeTargets.insert(resumeTargets.end(), resumeIt->second.begin(), resumeIt->second.end());
        }
        std::sort(resumeTargets.begin(), resumeTargets.end());
        resumeTargets.erase(std::unique(resumeTargets.begin(), resumeTargets.end()), resumeTargets.end());

        std::unordered_set<uint32_t> instructionAddresses;
        instructionAddresses.reserve(instructions.size());
        for (const Instruction &instruction : instructions)
        {
            instructionAddresses.insert(instruction.address);
        }

        for (uint32_t target : resumeTargets)
        {
            if (!instructionAddresses.contains(target))
            {
                std::ostringstream msg;
                msg << "Resume entry 0x" << std::hex << target
                    << " has no decoded instruction in owner "
                    << function.name << " at 0x" << function.start;
                if (cg.m_reporter)
                {
                    cg.m_reporter->errorAt(
                        "resume-entry", function.name, target, msg.str());
                }
                throw std::runtime_error(msg.str());
            }
            analysisResult.entryPoints.insert(target);
        }
        if (cg.m_reporter && resumeIt != cg.m_resumeEntryTargetsByOwner.end())
        {
            cg.m_reporter->recordResumeEntryTargetsAudited(
                resumeIt->second.size());
        }

        std::vector<Instruction> scheduledInstructions = instructions;
        applyVu0SyncPlan(
            scheduledInstructions, analysisResult.entryPoints);

        const std::unordered_set<uint32_t> &internalTargets = analysisResult.entryPoints;
        ConstantRegisterState constantRegisters;
        GifDmaKickPlan gifDmaKickPlan{};
        ss << "// Function: " << function.name << "\n";
        ss << "// Address: 0x" << std::hex << function.start << " - 0x" << function.end << std::dec << "\n";

        std::string sanitizedName = cg.getFunctionName(function.start);
        if (sanitizedName.empty())
        {
            std::stringstream nameBuilder;
            nameBuilder << "Errorfunc_" << std::hex << function.start;
            sanitizedName = nameBuilder.str();
        }

        std::vector<Instruction> observationDescriptors =
            scheduledInstructions;
        std::unordered_set<uint32_t> descriptorAddresses;
        descriptorAddresses.reserve(
            observationDescriptors.size() * 2u);
        for (const Instruction &instruction : observationDescriptors)
        {
            descriptorAddresses.insert(instruction.address);
        }
        for (const Instruction &instruction : scheduledInstructions)
        {
            const uint32_t delayAddress = instruction.address + 4u;
            if (instruction.hasDelaySlot &&
                !descriptorAddresses.contains(delayAddress))
            {
                observationDescriptors.push_back(
                    makeSyntheticDelaySlot(delayAddress));
                descriptorAddresses.insert(delayAddress);
            }
        }
        std::sort(
            observationDescriptors.begin(),
            observationDescriptors.end(),
            [](const Instruction &left, const Instruction &right)
            {
                return left.address < right.address;
            });

        const std::string descriptorName =
            eeObservationDescriptorName(sanitizedName);
        std::unordered_map<uint32_t, size_t> descriptorIndices;
        descriptorIndices.reserve(observationDescriptors.size());
        ss << "namespace {\n";
        ss << "PS2X_EE_OBSERVATION_COLD_DATA static const "
              "EeObservationInstructionDescriptor "
           << descriptorName << "[] = {\n";
        for (size_t index = 0u;
             index < observationDescriptors.size(); ++index)
        {
            descriptorIndices.emplace(
                observationDescriptors[index].address, index);
            ss << eeObservationDescriptorInitializer(
                      observationDescriptors[index])
               << ",\n";
        }
        ss << "};\n";
        ss << "}\n\n";
        cg.setCurrentEeObservationDescriptors(
            "eeo", std::move(descriptorIndices));

        ss << "template <EeArchitecturalObservationMode Mode>\n";
        ss << "static void " << eeTemplateBodyName(sanitizedName)
           << "(uint8_t* rdram, R5900Context* ctx, PS2Runtime *runtime) {\n";
        ss << "const auto*eeo=" << descriptorName << ";\n";
        ss << "#ifdef PS2_FUNCTION_LOG_TRACKER\n";
        ss << "    PS_LOG_ENTRY(\"" << sanitizedName << "\");\n";
        ss << "#endif\n";
        ss << "\n";
        if (!resumeTargets.empty())
        {
            ss << "    switch (ctx->pc) {\n";
            for (uint32_t target : resumeTargets)
            {
                ss << "        case 0x" << std::hex << target << "u: goto label_" << target << ";\n"
                   << std::dec;
            }
            ss << "        default: break;\n";
            ss << "    }\n\n";
        }
        ss << "    ctx->pc = 0x" << std::hex << function.start << "u;\n"
           << std::dec;
        ss << "\n";

        bool lastInstructionWasControlFlow = false;
        bool validateSequentialFetch = false;

        for (size_t i = 0; i < scheduledInstructions.size(); ++i)
        {
            const Instruction &inst = scheduledInstructions[i];
            lastInstructionWasControlFlow = inst.hasDelaySlot;

            if (internalTargets.contains(inst.address))
            {
                constantRegisters.clear();
                ss << "label_" << std::hex << inst.address << std::dec << ":\n";
            }

            if (cg.m_emitInstructionComments)
            {
                ss << "    // 0x" << std::hex << inst.address << ": 0x" << inst.raw << std::dec;
                std::string disassembly = R5900Decoder::disassembleInstruction(inst);
                if (!disassembly.empty())
                {
                    ss << "  " << disassembly;
                }
                ss << "\n";
            }

            try
            {
                const std::string observationDescriptor =
                    cg.eeObservationDescriptorExpression(
                        inst.address);
                if (observationDescriptor.empty())
                {
                    throw std::runtime_error(
                        "Missing EE observation descriptor");
                }
                ss << "    ctx->pc = 0x" << std::hex
                   << inst.address << "u;\n"
                   << std::dec;
                const bool breakpointAlreadyChecked =
                    validateSequentialFetch;
                if (validateSequentialFetch)
                {
                    ss << "    runtime->validateEeInstructionFetchPolicy<Mode>(ctx, ctx->pc);\n";
                    validateSequentialFetch = false;
                }
                ss << "    "
                   << eeObservationBeginCall(
                          observationDescriptor,
                          breakpointAlreadyChecked)
                   << "\n";

                if (inst.hasDelaySlot)
                {
                    ss << "    ctx->beginEeInstruction();\n";
                    ss << "    ctx->advanceEeCycleTicks("
                       << eeInstructionCycleTicks(inst) << "u);\n";
                    const bool hasDecodedDelaySlot =
                        i + 1 < scheduledInstructions.size() &&
                        scheduledInstructions[i + 1].address == inst.address + 4u;

                    Instruction syntheticDelaySlot{};
                    const Instruction *delaySlot = nullptr;
                    if (hasDecodedDelaySlot)
                    {
                        delaySlot = &scheduledInstructions[i + 1];
                    }
                    else
                    {
                        syntheticDelaySlot = makeSyntheticDelaySlot(inst.address + 4u);
                        delaySlot = &syntheticDelaySlot;
                    }

                    if (hasDecodedDelaySlot && internalTargets.contains(delaySlot->address))
                    {
                        ss << "label_" << std::hex << delaySlot->address << std::dec << ":\n";
                    }

                    if (gifDmaKickPlan.valid &&
                        gifDmaKickPlan.completesInDelaySlot &&
                        gifDmaKickPlan.branchIndex == i &&
                        hasDecodedDelaySlot)
                    {
                        const MemoryAccessHint delayMemoryHint =
                            resolveMemoryAccessHint(
                                *delaySlot, constantRegisters);
                        ss << cg.handleBranchDelaySlots(
                            inst,
                            *delaySlot,
                            function,
                            analysisResult,
                            gifDmaDelaySlotOverride(
                                *delaySlot,
                                gifDmaKickPlan,
                                cg.m_emitInstructionComments,
                                cg.translateInstruction(
                                    *delaySlot,
                                    delayMemoryHint)));
                        gifDmaKickPlan = {};
                    }
                    else
                    {
                        ss << cg.handleBranchDelaySlots(inst, *delaySlot, function, analysisResult);
                    }

                    if (hasDecodedDelaySlot)
                    {
                        ++i; // Skip delay slot instruction (handled inside branch logic)
                    }
                    constantRegisters.clear();
                }
                else
                {
                    const EeInstructionPerformanceInfo
                        performanceInfo =
                            eeInstructionPerformanceInfo(inst);
                    ss << "    ctx->beginEeInstruction();\n";
                    ss << "    ctx->advanceEeCycleTicks("
                       << eeInstructionCycleTicks(inst) << "u);\n";
                    if (!gifDmaKickPlan.valid)
                    {
                        gifDmaKickPlan = tryBuildGifDmaKickPlan(scheduledInstructions, i, constantRegisters, internalTargets);
                    }

                    if (gifDmaKickPlan.suppresses(i))
                    {
                        if (performanceInfo
                                .completionBeforeEffect)
                        {
                            ss << "    "
                               << eeObservationCompletionCall(
                                      observationDescriptor)
                               << "\n";
                        }
                        const size_t slot = gifDmaKickPlan.slotFor(i);
                        emitGifDmaCapture(ss, gifDmaKickPlan, slot, "    ");

                        const MemoryAccessHint memoryHint =
                            resolveMemoryAccessHint(
                                inst, constantRegisters);
                        ss << "    if (runtime->eeDataBreakpointEnabledPolicy<Mode>(ctx, true)) {\n";
                        ss << "        "
                           << cg.translateInstruction(inst, memoryHint)
                           << "\n";

                        if (gifDmaKickPlan.completesAt(i))
                        {
                            ss << "    } else {\n";
                            ss << "        " << gifDmaKickCall(gifDmaKickPlan) << "\n";
                            gifDmaKickPlan = {};
                        }
                        ss << "    }\n";

                        if (!performanceInfo
                                 .completionBeforeEffect)
                        {
                            ss << "    "
                               << eeObservationCompletionCall(
                                      observationDescriptor)
                               << "\n";
                        }

                        updateConstantRegisters(inst, constantRegisters);
                        continue;
                    }

                    ss << "    ctx->pc = 0x" << std::hex << inst.address << "u;\n"
                       << std::dec;
                    const MemoryAccessHint memoryHint = resolveMemoryAccessHint(inst, constantRegisters);
                    if (performanceInfo
                            .completionBeforeEffect)
                    {
                        ss << "    "
                           << eeObservationCompletionCall(
                                  observationDescriptor)
                           << "\n";
                    }
                    if (mayChangeEeArchitecturalObservationMode(inst))
                    {
                        ss << "    const EeArchitecturalObservationMode "
                              "eeObservationModeBefore_"
                           << std::hex << inst.address
                           << " = PS2Runtime::architecturalObservationMode(ctx);\n"
                           << std::dec;
                    }
                    ss << "    " << cg.translateInstruction(inst, memoryHint);
                    if (inst.isMmio)
                    {
                        ss << " // MMIO: 0x" << std::hex << inst.mmioAddress << std::dec;
                    }
                    ss << "\n";
                    if (!performanceInfo
                             .completionBeforeEffect)
                    {
                        ss << "    "
                           << eeObservationCompletionCall(
                                  observationDescriptor)
                           << "\n";
                    }

                    if (mayChangeEeArchitecturalObservationMode(inst))
                    {
                        ss << "    ctx->pc = 0x" << std::hex
                           << (inst.address + 4u) << "u;\n"
                           << std::dec;
                        ss << "    if (runtime->completeEeObservationModeTransition("
                              "rdram, ctx, eeObservationModeBefore_"
                           << std::hex << inst.address
                           << ")) { return; }\n"
                           << std::dec;
                    }

                    updateConstantRegisters(inst, constantRegisters);
                    validateSequentialFetch =
                        changesEeAddressMode(inst);
                }
            }
            catch (const std::exception &e)
            {
                if (cg.m_reporter)
                {
                    std::ostringstream msg;
                    msg << "translation failed: " << e.what() << " raw=0x" << std::hex << inst.raw;
                    cg.m_reporter->errorAt("codegen", function.name, inst.address, msg.str());
                }

                throw;
            }
        }

        // Fallthrough with no terminating branch: advance ctx->pc past the function so dispatchLoop doesn't re-call it forever.
        if (!instructions.empty() && !lastInstructionWasControlFlow)
        {
            ss << "    ctx->pc = 0x" << std::hex << function.end << "u;\n"
               << std::dec;
        }

        ss << "}\n\n";
        ss << "template PS2X_EE_OBSERVATION_PRECISE_BODY void "
           << eeTemplateBodyName(sanitizedName)
           << "<EeArchitecturalObservationMode::Precise>("
              "uint8_t*, R5900Context*, PS2Runtime*);\n\n";
        ss << "void " << eeFastFunctionName(sanitizedName)
           << "(uint8_t* rdram, R5900Context* ctx, PS2Runtime *runtime) {\n";
        ss << "    " << eeTemplateBodyName(sanitizedName)
           << "<EeArchitecturalObservationMode::Fast>(rdram, ctx, runtime);\n";
        ss << "}\n\n";
        ss << "void " << eePreciseFunctionName(sanitizedName)
           << "(uint8_t* rdram, R5900Context* ctx, PS2Runtime *runtime) {\n";
        ss << "    " << eeTemplateBodyName(sanitizedName)
           << "<EeArchitecturalObservationMode::Precise>(rdram, ctx, runtime);\n";
        ss << "}\n";
        return ss.str();
    }
}
