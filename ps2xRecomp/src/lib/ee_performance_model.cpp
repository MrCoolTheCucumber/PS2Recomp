#include "ps2recomp/ee_performance_model.h"

#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"
#include "runtime/ee_performance.h"

#include <fmt/format.h>

namespace ps2recomp
{
    namespace
    {
        using namespace ps2x::performance;

        bool isSpecialTrap(const Instruction &inst)
        {
            if (inst.opcode != OPCODE_SPECIAL)
            {
                return false;
            }
            switch (inst.function)
            {
            case SPECIAL_SYSCALL:
            case SPECIAL_BREAK:
            case SPECIAL_TGE:
            case SPECIAL_TGEU:
            case SPECIAL_TLT:
            case SPECIAL_TLTU:
            case SPECIAL_TEQ:
            case SPECIAL_TNE:
                return true;
            default:
                return false;
            }
        }

        bool isRegimmTrap(const Instruction &inst)
        {
            if (inst.opcode != OPCODE_REGIMM)
            {
                return false;
            }
            switch (inst.rt)
            {
            case REGIMM_TGEI:
            case REGIMM_TGEIU:
            case REGIMM_TLTI:
            case REGIMM_TLTIU:
            case REGIMM_TEQI:
            case REGIMM_TNEI:
                return true;
            default:
                return false;
            }
        }

        bool isCop1Instruction(const Instruction &inst)
        {
            return inst.opcode == OPCODE_COP1 ||
                   inst.opcode == OPCODE_LWC1 ||
                   inst.opcode == OPCODE_LDC1 ||
                   inst.opcode == OPCODE_SWC1 ||
                   inst.opcode == OPCODE_SDC1;
        }

        bool isCop2Instruction(const Instruction &inst)
        {
            return inst.opcode == OPCODE_COP2 ||
                   inst.opcode == OPCODE_LWC2 ||
                   inst.opcode == OPCODE_LDC2 ||
                   inst.opcode == OPCODE_SWC2 ||
                   inst.opcode == OPCODE_SDC2;
        }

        uint32_t issuePipeMask(const Instruction &inst)
        {
            if (inst.hasDelaySlot)
            {
                return kIssuePipe0 | kIssuePipe1;
            }
            if (inst.isLoad || inst.isStore ||
                inst.opcode == OPCODE_CACHE ||
                inst.opcode == OPCODE_PREF)
            {
                return kIssuePipe1;
            }
            if (inst.opcode == OPCODE_COP0)
            {
                return kIssuePipe1;
            }
            if (inst.opcode == OPCODE_COP1)
            {
                return inst.rs == COP1_S || inst.rs == COP1_W
                           ? kIssuePipe0
                           : kIssuePipe1;
            }
            if (inst.opcode == OPCODE_COP2)
            {
                return inst.rs >= COP2_CO
                           ? kIssuePipe0
                           : kIssuePipe1;
            }
            if (inst.opcode == OPCODE_MMI)
            {
                switch (inst.function)
                {
                case MMI_PLZCW:
                case MMI_MFHI1:
                case MMI_MTHI1:
                case MMI_MFLO1:
                case MMI_MTLO1:
                case MMI_MULT1:
                case MMI_MULTU1:
                case MMI_DIV1:
                case MMI_DIVU1:
                case MMI_MADD1:
                case MMI_MADDU1:
                    return kIssuePipe1;
                default:
                    break;
                }
                return kIssuePipe0;
            }
            if (inst.opcode == OPCODE_REGIMM &&
                (inst.rt == REGIMM_MTSAB ||
                 inst.rt == REGIMM_MTSAH))
            {
                return kIssuePipe0;
            }
            if (inst.opcode == OPCODE_SPECIAL)
            {
                switch (inst.function)
                {
                case SPECIAL_SYNC:
                    return kIssuePipe1;
                case SPECIAL_MFSA:
                case SPECIAL_MTSA:
                case SPECIAL_MFHI:
                case SPECIAL_MFLO:
                case SPECIAL_MTHI:
                case SPECIAL_MTLO:
                case SPECIAL_MULT:
                case SPECIAL_MULTU:
                case SPECIAL_DIV:
                case SPECIAL_DIVU:
                    return kIssuePipe0;
                default:
                    break;
                }
            }
            return kIssuePipe0 | kIssuePipe1;
        }
    }

    EeInstructionPerformanceInfo
    eeInstructionPerformanceInfo(const Instruction &inst)
    {
        using namespace ps2x::performance;

        EeInstructionPerformanceInfo info{};
        info.issuePipeMask = issuePipeMask(inst);
        if (inst.hasDelaySlot)
        {
            info.issueFlags |= kIssueBranch;
        }
        if (inst.opcode == OPCODE_COP0 &&
            inst.rs == COP0_CO &&
            inst.function == COP0_CO_ERET)
        {
            info.issueFlags |= kIssueEret;
        }

        info.completionEvents = kInstructionCompleted;
        if (!inst.hasDelaySlot)
        {
            info.completionEvents |=
                kNonBdsInstructionCompleted;
        }
        if (isCop2Instruction(inst))
        {
            info.completionEvents |=
                kCop2InstructionCompleted;
        }
        if (isCop1Instruction(inst))
        {
            info.completionEvents |=
                kCop1InstructionCompleted;
        }
        if (inst.isLoad)
        {
            info.completionEvents |= kLoadCompleted;
        }
        if (inst.isStore)
        {
            info.completionEvents |= kStoreCompleted;
        }
        const bool isEret =
            inst.opcode == OPCODE_COP0 &&
            inst.rs == COP0_CO &&
            inst.function == COP0_CO_ERET;
        info.completionBeforeEffect =
            isSpecialTrap(inst) ||
            isRegimmTrap(inst) ||
            isEret;
        return info;
    }

    std::string eeInstructionIssueCall(
        const Instruction &inst)
    {
        const EeInstructionPerformanceInfo info =
            eeInstructionPerformanceInfo(inst);
        return fmt::format(
            "if ((ctx->cop0_perf & 0x80000000u) != 0u) {{ "
            "runtime->recordEeInstructionIssue(ctx, {{{}u, 0x{:08X}u, "
            "0x{:08X}u, {}u}}); }}",
            info.issuePipeMask,
            inst.gprReadMask,
            inst.gprWriteMask,
            info.issueFlags);
    }

    std::string eeInstructionCompletionCall(
        const Instruction &inst)
    {
        const EeInstructionPerformanceInfo info =
            eeInstructionPerformanceInfo(inst);
        return fmt::format(
            "if ((ctx->cop0_perf & 0x80000000u) != 0u) {{ "
            "runtime->recordEeInstructionCompletion(ctx, {}u); }}",
            info.completionEvents);
    }
}
