#include "ps2recomp/ee_cycle_model.h"

#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"

namespace ps2recomp
{
    namespace
    {
        constexpr uint32_t kDefaultTicks = 9u;
        constexpr uint32_t kBranchTicks = 11u;
        constexpr uint32_t kCoprocessorTicks = 7u;
        constexpr uint32_t kLoadStoreTicks = 14u;
        constexpr uint32_t kMultiplyTicks = 16u;
        constexpr uint32_t kDivideTicks = 112u;
        constexpr uint32_t kMmiDefaultTicks = 14u;
        constexpr uint32_t kMmiMultiplyTicks = 24u;
        constexpr uint32_t kMmiDivideTicks = 176u;
        constexpr uint32_t kFpuMultiplyTicks = 32u;
        constexpr uint32_t kFpuDivideTicks = 48u;
        constexpr uint32_t kFpuReciprocalSqrtTicks = 64u;

        bool isLoadOrStoreOpcode(uint32_t opcode)
        {
            switch (opcode)
            {
            case OPCODE_LDL:
            case OPCODE_LDR:
            case OPCODE_LQ:
            case OPCODE_SQ:
            case OPCODE_LB:
            case OPCODE_LH:
            case OPCODE_LWL:
            case OPCODE_LW:
            case OPCODE_LBU:
            case OPCODE_LHU:
            case OPCODE_LWR:
            case OPCODE_LWU:
            case OPCODE_SB:
            case OPCODE_SH:
            case OPCODE_SWL:
            case OPCODE_SW:
            case OPCODE_SDL:
            case OPCODE_SDR:
            case OPCODE_SWR:
            case OPCODE_LL:
            case OPCODE_LWC1:
            case OPCODE_LWC2:
            case OPCODE_LLD:
            case OPCODE_LDC1:
            case OPCODE_LDC2:
            case OPCODE_LD:
            case OPCODE_SC:
            case OPCODE_SWC1:
            case OPCODE_SWC2:
            case OPCODE_SCD:
            case OPCODE_SDC1:
            case OPCODE_SDC2:
            case OPCODE_SD:
                return true;
            default:
                return false;
            }
        }

        uint32_t specialCycleTicks(const Instruction &inst)
        {
            switch (inst.function)
            {
            case SPECIAL_MULT:
            case SPECIAL_MULTU:
                return kMultiplyTicks;
            case SPECIAL_DIV:
            case SPECIAL_DIVU:
                return kDivideTicks;
            case SPECIAL_TGE:
            case SPECIAL_TGEU:
            case SPECIAL_TLT:
            case SPECIAL_TLTU:
            case SPECIAL_TEQ:
            case SPECIAL_TNE:
                return kBranchTicks;
            default:
                return kDefaultTicks;
            }
        }

        uint32_t regimmCycleTicks(const Instruction &inst)
        {
            switch (inst.rt)
            {
            case REGIMM_BLTZ:
            case REGIMM_BGEZ:
            case REGIMM_BLTZL:
            case REGIMM_BGEZL:
            case REGIMM_TGEI:
            case REGIMM_TGEIU:
            case REGIMM_TLTI:
            case REGIMM_TLTIU:
            case REGIMM_TEQI:
            case REGIMM_TNEI:
            case REGIMM_BLTZAL:
            case REGIMM_BGEZAL:
            case REGIMM_BLTZALL:
            case REGIMM_BGEZALL:
                return kBranchTicks;
            default:
                return kDefaultTicks;
            }
        }

        uint32_t mmiCycleTicks(const Instruction &inst)
        {
            switch (inst.function)
            {
            case MMI_MADD:
            case MMI_MADDU:
            case MMI_MSUB:
            case MMI_MSUBU:
            case MMI_MULT1:
            case MMI_MULTU1:
            case MMI_MADD1:
            case MMI_MADDU1:
                return kMultiplyTicks;
            case MMI_DIV1:
            case MMI_DIVU1:
                return kDivideTicks;
            case MMI_MMI2:
                switch (inst.mmiFunction)
                {
                case MMI2_PMADDW:
                case MMI2_PMSUBW:
                case MMI2_PMFHI:
                case MMI2_PMFLO:
                case MMI2_PMULTW:
                case MMI2_PMADDH:
                case MMI2_PHMADH:
                case MMI2_PMSUBH:
                case MMI2_PHMSBH:
                case MMI2_PMULTH:
                    return kMmiMultiplyTicks;
                case MMI2_PDIVW:
                case MMI2_PDIVBW:
                    return kMmiDivideTicks;
                default:
                    return kMmiDefaultTicks;
                }
            case MMI_MMI3:
                switch (inst.mmiFunction)
                {
                case MMI3_PMADDUW:
                case MMI3_PMULTUW:
                    return kMmiMultiplyTicks;
                case MMI3_PDIVUW:
                    return kMmiDivideTicks;
                default:
                    return kMmiDefaultTicks;
                }
            default:
                return kMmiDefaultTicks;
            }
        }

        uint32_t cop1CycleTicks(const Instruction &inst)
        {
            if (inst.rs == COP1_BC)
            {
                return kBranchTicks;
            }
            if (inst.rs != COP1_MF && inst.rs != COP1_CF &&
                inst.rs != COP1_MT && inst.rs != COP1_CT &&
                inst.rs != COP1_S && inst.rs != COP1_W)
            {
                return kDefaultTicks;
            }
            if (inst.rs != COP1_S)
            {
                return kCoprocessorTicks;
            }

            switch (inst.function)
            {
            case COP1_S_MUL:
            case COP1_S_MULA:
            case COP1_S_MADD:
            case COP1_S_MSUB:
            case COP1_S_MADDA:
            case COP1_S_MSUBA:
                return kFpuMultiplyTicks;
            case COP1_S_DIV:
            case COP1_S_SQRT:
                return kFpuDivideTicks;
            case COP1_S_RSQRT:
                return kFpuReciprocalSqrtTicks;
            default:
                return kCoprocessorTicks;
            }
        }
    }

    uint32_t eeInstructionCycleTicks(const Instruction &inst)
    {
        if (isLoadOrStoreOpcode(inst.opcode))
        {
            return kLoadStoreTicks;
        }

        switch (inst.opcode)
        {
        case OPCODE_SPECIAL:
            return specialCycleTicks(inst);
        case OPCODE_REGIMM:
            return regimmCycleTicks(inst);
        case OPCODE_BEQ:
        case OPCODE_BNE:
        case OPCODE_BLEZ:
        case OPCODE_BGTZ:
        case OPCODE_BEQL:
        case OPCODE_BNEL:
        case OPCODE_BLEZL:
        case OPCODE_BGTZL:
            return kBranchTicks;
        case OPCODE_COP0:
            return inst.rs == COP0_BC ? kBranchTicks : kCoprocessorTicks;
        case OPCODE_COP1:
            return cop1CycleTicks(inst);
        case OPCODE_MMI:
            return mmiCycleTicks(inst);
        default:
            return kDefaultTicks;
        }
    }
}
