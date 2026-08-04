#include "ps2recomp/ee_retirement_model.h"

#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"

namespace ps2recomp
{
    bool eeInstructionCanBatchRetirement(
        const Instruction &instruction)
    {
        if (instruction.hasDelaySlot || instruction.isBranch ||
            instruction.isJump || instruction.isCall ||
            instruction.isReturn || instruction.isLoad ||
            instruction.isStore || instruction.isMMI ||
            instruction.isVU || instruction.isMmio)
        {
            return false;
        }

        switch (instruction.opcode)
        {
        case OPCODE_ADDIU:
        case OPCODE_SLTI:
        case OPCODE_SLTIU:
        case OPCODE_ANDI:
        case OPCODE_ORI:
        case OPCODE_XORI:
        case OPCODE_LUI:
        case OPCODE_DADDIU:
        case OPCODE_PREF:
            return true;
        case OPCODE_REGIMM:
            return instruction.rt == REGIMM_MTSAB ||
                   instruction.rt == REGIMM_MTSAH;
        case OPCODE_SPECIAL:
            switch (instruction.function)
            {
            case SPECIAL_SLL:
            case SPECIAL_SRL:
            case SPECIAL_SRA:
            case SPECIAL_SLLV:
            case SPECIAL_SRLV:
            case SPECIAL_SRAV:
            case SPECIAL_SYNC:
            case SPECIAL_MFHI:
            case SPECIAL_MTHI:
            case SPECIAL_MFLO:
            case SPECIAL_MTLO:
            case SPECIAL_MULT:
            case SPECIAL_MULTU:
            case SPECIAL_DIV:
            case SPECIAL_DIVU:
            case SPECIAL_ADDU:
            case SPECIAL_SUBU:
            case SPECIAL_AND:
            case SPECIAL_OR:
            case SPECIAL_XOR:
            case SPECIAL_NOR:
            case SPECIAL_SLT:
            case SPECIAL_SLTU:
            case SPECIAL_MOVZ:
            case SPECIAL_MOVN:
            case SPECIAL_MFSA:
            case SPECIAL_MTSA:
            case SPECIAL_DADDU:
            case SPECIAL_DSUBU:
            case SPECIAL_DSLL:
            case SPECIAL_DSRL:
            case SPECIAL_DSRA:
            case SPECIAL_DSLLV:
            case SPECIAL_DSRLV:
            case SPECIAL_DSRAV:
            case SPECIAL_DSLL32:
            case SPECIAL_DSRL32:
            case SPECIAL_DSRA32:
                return true;
            default:
                return false;
            }
        default:
            return false;
        }
    }
}
