#include "runtime/ps2_vu1.h"
#include "runtime/ps2_memory.h"
#include "ps2_vu1_detail.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

// ============================================================================
// Lower instructions
// ============================================================================
void VuInterpreterBackend::execLower(
    uint32_t instr, uint8_t *vuData, uint32_t dataSize,
    IVuSideEffectSink &sideEffects, PS2Memory *memory,
    uint32_t upperInstr)
{
    VuExecutionState &state = *m_state;
    (void)upperInstr;
    if (instr == 0x00000000 || instr == 0x8000033C) // NOP
        return;

    uint8_t opHi = (instr >> 25) & 0x7F;

    // The lower instruction encoding uses bits 31:25 for the primary opcode
    switch (opHi)
    {
    case 0x00: // LQ (Load Quadword from VU data memory)
    {
        uint8_t it = FT(instr);      // VF destination
        uint8_t is = VIS(instr);    // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(state.vi[is] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            float tmp[4];
            std::memcpy(tmp, vuData + addr, 16);
            applyDest(state.vf[it], tmp, dest);
        }
        return;
    }
    case 0x01: // SQ (Store Quadword to VU data memory)
    {
        uint8_t is = FS(instr);      // VF source
        uint8_t it = VIT(instr);     // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(state.vi[it] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            float tmp[4];
            std::memcpy(tmp, vuData + addr, 16);
            if (dest & 0x8)
                tmp[0] = state.vf[is][0];
            if (dest & 0x4)
                tmp[1] = state.vf[is][1];
            if (dest & 0x2)
                tmp[2] = state.vf[is][2];
            if (dest & 0x1)
                tmp[3] = state.vf[is][3];
            std::memcpy(vuData + addr, tmp, 16);
        }
        return;
    }
    case 0x04: // ILW (Integer Load Word from VU data memory)
    {
        uint8_t it = VIT(instr);     // VI destination
        uint8_t is = VIS(instr);     // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(state.vi[is] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            int comp = 0;
            if (dest & 0x8)
                comp = 0;
            else if (dest & 0x4)
                comp = 1;
            else if (dest & 0x2)
                comp = 2;
            else
                comp = 3;
            uint32_t v;
            std::memcpy(&v, vuData + addr + comp * 4, 4);
            if (it != 0)
                state.vi[it] = (int32_t)(int16_t)(v & 0xFFFF);
        }
        return;
    }
    case 0x05: // ISW (Integer Store Word to VU data memory)
    {
        uint8_t it = VIT(instr);     // VI source
        uint8_t is = VIS(instr);     // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(state.vi[is] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            uint32_t val = (uint32_t)(uint16_t)(state.vi[it] & 0xFFFF);
            if (dest & 0x8)
                std::memcpy(vuData + addr + 0, &val, 4);
            if (dest & 0x4)
                std::memcpy(vuData + addr + 4, &val, 4);
            if (dest & 0x2)
                std::memcpy(vuData + addr + 8, &val, 4);
            if (dest & 0x1)
                std::memcpy(vuData + addr + 12, &val, 4);
        }
        return;
    }
    case 0x08: // IADDIU
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = (int16_t)(instr & 0x7FF) | ((instr >> 10) & 0x7800);
        if (it != 0)
            state.vi[it] = (int16_t)(state.vi[is] + imm);
        return;
    }
    case 0x09: // ISUBIU
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = (int16_t)(instr & 0x7FF) | ((instr >> 10) & 0x7800);
        if (it != 0)
            state.vi[it] = (int16_t)(state.vi[is] - imm);
        return;
    }
    case 0x10: // FCEQ
    {
        uint32_t imm24 = instr & 0xFFFFFF;
        if (1 != 0)
            state.vi[1] = ((state.clip & 0xFFFFFF) == imm24) ? 1 : 0;
        return;
    }
    case 0x11: // FCSET
    {
        state.clip = instr & 0xFFFFFF;
        return;
    }
    case 0x12: // FCAND
    {
        uint32_t imm24 = instr & 0xFFFFFF;
        if (1 != 0)
            state.vi[1] = ((state.clip & imm24) != 0) ? 1 : 0;
        return;
    }
    case 0x13: // FCOR
    {
        uint32_t imm24 = instr & 0xFFFFFF;
        if (1 != 0)
            state.vi[1] = ((state.clip | imm24) == 0xFFFFFF) ? 1 : 0;
        return;
    }
    case 0x14: // FSEQ
    {
        uint16_t imm12 = instr & 0xFFF;
        if (1 != 0)
            state.vi[1] = ((state.status & 0xFFF) == imm12) ? 1 : 0;
        return;
    }
    case 0x15: // FSSET
    {
        state.status = (instr >> 6) & 0xFC0;
        return;
    }
    case 0x16: // FSAND
    {
        uint16_t imm12 = instr & 0xFFF;
        if (1 != 0)
            state.vi[1] = (int32_t)(state.status & imm12);
        return;
    }
    case 0x17: // FSOR
    {
        uint16_t imm12 = instr & 0xFFF;
        if (1 != 0)
            state.vi[1] = ((state.status | imm12) == 0xFFF) ? 1 : 0;
        return;
    }
    case 0x18: // FMAND
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        if (it != 0)
            state.vi[it] = (int32_t)(state.mac & (uint32_t)(uint16_t)state.vi[is]);
        return;
    }
    case 0x1A: // FMEQ
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        if (it != 0)
            state.vi[it] = ((state.mac & 0xFFFF) == (uint32_t)(uint16_t)state.vi[is]) ? 1 : 0;
        return;
    }
    case 0x1C: // FMOR
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        if (it != 0)
            state.vi[it] = (int32_t)(state.mac | (uint32_t)(uint16_t)state.vi[is]);
        return;
    }
    case 0x20: // B (unconditional branch)
    {
        int16_t imm = IMM11(instr);
        uint32_t target = (state.pc + 8 + imm * 8) & 0x3FFF;
        state.branchPending = true;
        state.branchTarget = target;
        state.branchDelay = 1;
        return;
    }
    case 0x21: // BAL (Branch and link)
    {
        uint8_t it = VIT(instr);
        int16_t imm = IMM11(instr);
        uint32_t target = (state.pc + 8 + imm * 8) & 0x3FFF;
        if (it != 0)
            state.vi[it] = (int32_t)((state.pc + 16) / 8);
        state.branchPending = true;
        state.branchTarget = target;
        state.branchDelay = 1;
        return;
    }
    case 0x24: // JR
    {
        uint8_t is = VIS(instr);
        uint32_t target = ((uint32_t)(uint16_t)state.vi[is] * 8u) & 0x3FFF;
        state.branchPending = true;
        state.branchTarget = target;
        state.branchDelay = 1;
        return;
    }
    case 0x25: // JALR
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        uint32_t target = ((uint32_t)(uint16_t)state.vi[is] * 8u) & 0x3FFF;
        if (it != 0)
            state.vi[it] = (int32_t)((state.pc + 16) / 8);
        state.branchPending = true;
        state.branchTarget = target;
        state.branchDelay = 1;
        return;
    }
    case 0x28: // IBEQ
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if (readViForBranch(is) == readViForBranch(it))
        {
            uint32_t target = (state.pc + 8 + imm * 8) & 0x3FFF;
            state.branchPending = true;
            state.branchTarget = target;
            state.branchDelay = 1;
        }
        return;
    }
    case 0x29: // IBNE
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if (readViForBranch(is) != readViForBranch(it))
        {
            uint32_t target = (state.pc + 8 + imm * 8) & 0x3FFF;
            state.branchPending = true;
            state.branchTarget = target;
            state.branchDelay = 1;
        }
        return;
    }
    case 0x2C: // IBLTZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if (readViForBranch(is) < 0)
        {
            uint32_t target = (state.pc + 8 + imm * 8) & 0x3FFF;
            state.branchPending = true;
            state.branchTarget = target;
            state.branchDelay = 1;
        }
        return;
    }
    case 0x2D: // IBGTZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if (readViForBranch(is) > 0)
        {
            uint32_t target = (state.pc + 8 + imm * 8) & 0x3FFF;
            state.branchPending = true;
            state.branchTarget = target;
            state.branchDelay = 1;
        }
        return;
    }
    case 0x2E: // IBLEZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if (readViForBranch(is) <= 0)
        {
            uint32_t target = (state.pc + 8 + imm * 8) & 0x3FFF;
            state.branchPending = true;
            state.branchTarget = target;
            state.branchDelay = 1;
        }
        return;
    }
    case 0x2F: // IBGEZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if (readViForBranch(is) >= 0)
        {
            uint32_t target = (state.pc + 8 + imm * 8) & 0x3FFF;
            state.branchPending = true;
            state.branchTarget = target;
            state.branchDelay = 1;
        }
        return;
    }

    case 0x40: // Lower1 / lower special. Bit31 set; low 6 bits select integer or special op.
    {
        const uint8_t funct = instr & 0x3Fu;
        const uint8_t vfT = FT(instr);
        const uint8_t vfS = FS(instr);
        const uint8_t viT = VIT(instr);
        const uint8_t viS = VIS(instr);
        const uint8_t viD = VID(instr);
        const uint8_t dest = (instr >> 21) & 0xF;

        switch (funct)
        {
        case 0x30: // IADD
            if (viD != 0)
                state.vi[viD] = (int16_t)(state.vi[viS] + state.vi[viT]);
            return;
        case 0x31: // ISUB
            if (viD != 0)
                state.vi[viD] = (int16_t)(state.vi[viS] - state.vi[viT]);
            return;
        case 0x32: // IADDI
        {
            int16_t imm5 = (int16_t)((int32_t)((instr >> 6) & 0x1F) << 27 >> 27);
            if (viT != 0)
                state.vi[viT] = (int16_t)(state.vi[viS] + imm5);
            return;
        }
        case 0x34: // IAND
            if (viD != 0)
                state.vi[viD] = state.vi[viS] & state.vi[viT];
            return;
        case 0x35: // IOR
            if (viD != 0)
                state.vi[viD] = state.vi[viS] | state.vi[viT];
            return;

        case 0x3C:
        case 0x3D:
        case 0x3E:
        case 0x3F: // Lower1 special. Dobie decodes this as (instr & 3) | ((instr >> 4) & 0x7C).
        {
            const uint8_t funct2 = (uint8_t)((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
            switch (funct2)
            {
            case 0x30: // MOVE
            {
                float tmp[4];
                std::memcpy(tmp, state.vf[vfS], 16);
                applyDest(state.vf[vfT], tmp, dest);
                return;
            }
            case 0x31: // MR32 (rotate right by 32 bits = shift xyzw -> yzwx)
            {
                float tmp[4] = {state.vf[vfS][1], state.vf[vfS][2], state.vf[vfS][3], state.vf[vfS][0]};
                applyDest(state.vf[vfT], tmp, dest);
                return;
            }
            case 0x34: // LQI (Load Quadword, post-increment)
            {
                backupVi(viS);
                uint32_t addr = ((uint32_t)(uint16_t)state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    applyDest(state.vf[vfT], tmp, dest);
                }
                if (viS != 0)
                    state.vi[viS] = (int16_t)(state.vi[viS] + 1);
                return;
            }
            case 0x35: // SQI (Store Quadword, post-increment)
            {
                backupVi(viT);
                uint32_t addr = ((uint32_t)(uint16_t)state.vi[viT]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    if (dest & 0x8)
                        tmp[0] = state.vf[vfS][0];
                    if (dest & 0x4)
                        tmp[1] = state.vf[vfS][1];
                    if (dest & 0x2)
                        tmp[2] = state.vf[vfS][2];
                    if (dest & 0x1)
                        tmp[3] = state.vf[vfS][3];
                    std::memcpy(vuData + addr, tmp, 16);
                }
                if (viT != 0)
                    state.vi[viT] = (int16_t)(state.vi[viT] + 1);
                return;
            }
            case 0x36: // LQD (Load Quadword, pre-decrement)
            {
                backupVi(viS);
                if (viS != 0)
                    state.vi[viS] = (int16_t)(state.vi[viS] - 1);
                uint32_t addr = ((uint32_t)(uint16_t)state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    applyDest(state.vf[vfT], tmp, dest);
                }
                return;
            }
            case 0x37: // SQD (Store Quadword, pre-decrement)
            {
                backupVi(viT);
                if (viT != 0)
                    state.vi[viT] = (int16_t)(state.vi[viT] - 1);
                uint32_t addr = ((uint32_t)(uint16_t)state.vi[viT]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    if (dest & 0x8)
                        tmp[0] = state.vf[vfS][0];
                    if (dest & 0x4)
                        tmp[1] = state.vf[vfS][1];
                    if (dest & 0x2)
                        tmp[2] = state.vf[vfS][2];
                    if (dest & 0x1)
                        tmp[3] = state.vf[vfS][3];
                    std::memcpy(vuData + addr, tmp, 16);
                }
                return;
            }
            case 0x38: // DIV
            {
                int fsf = (instr >> 21) & 0x3;
                int ftf = (instr >> 23) & 0x3;
                float num = state.vf[vfS][fsf];
                float den = state.vf[vfT][ftf];
                if (den != 0.0f)
                    scheduleQ(num / den, 7u);
                else
                    scheduleQ(
                        (num >= 0.0f)
                            ? std::numeric_limits<float>::max()
                            : -std::numeric_limits<float>::max(),
                        7u);
                return;
            }
            case 0x39: // SQRT
            {
                int ftf = (instr >> 23) & 0x3;
                float val = state.vf[vfT][ftf];
                scheduleQ(std::sqrt(std::fabs(val)), 7u);
                return;
            }
            case 0x3A: // RSQRT
            {
                int fsf = (instr >> 21) & 0x3;
                int ftf = (instr >> 23) & 0x3;
                float num = state.vf[vfS][fsf];
                float den = std::sqrt(std::fabs(state.vf[vfT][ftf]));
                if (den != 0.0f)
                    scheduleQ(num / den, 13u);
                else
                    scheduleQ(
                        std::numeric_limits<float>::max(), 13u);
                return;
            }
            case 0x3B: // WAITQ
                return;
            case 0x3C: // MTIR (Move To Integer Register)
            {
                // MTIR encodes a two-bit scalar selector in bits 22:21. It is
                // not an XYZW destination mask like the neighbouring ops.
                const uint8_t comp = static_cast<uint8_t>((instr >> 21u) & 0x3u);
                uint32_t fval;
                std::memcpy(&fval, &state.vf[vfS][comp], 4);
                if (viT != 0)
                {
                    backupVi(viT);
                    state.vi[viT] = (int32_t)(int16_t)(fval & 0xFFFF);
                }
                return;
            }
            case 0x3D: // MFIR (Move From Integer Register)
            {
                float result[4];
                int32_t val = (int32_t)(int16_t)(state.vi[viS] & 0xFFFF);
                std::memcpy(&result[0], &val, 4);
                result[1] = result[0];
                result[2] = result[0];
                result[3] = result[0];
                applyDest(state.vf[vfT], result, dest);
                return;
            }
            case 0x3E: // ILWR - integer load word from address in VI[is]
            {
                uint32_t addr = ((uint32_t)(uint16_t)state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    int comp = 0;
                    if (dest & 0x8)
                        comp = 0;
                    else if (dest & 0x4)
                        comp = 1;
                    else if (dest & 0x2)
                        comp = 2;
                    else
                        comp = 3;
                    uint32_t v;
                    std::memcpy(&v, vuData + addr + comp * 4, 4);
                    if (viT != 0)
                        state.vi[viT] = (int32_t)(int16_t)(v & 0xFFFF);
                }
                return;
            }
            case 0x3F: // ISWR - integer store word to address in VI[is]
            {
                uint32_t addr = ((uint32_t)(uint16_t)state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    uint32_t val = (uint32_t)(uint16_t)(state.vi[viT] & 0xFFFF);
                    if (dest & 0x8)
                        std::memcpy(vuData + addr + 0, &val, 4);
                    if (dest & 0x4)
                        std::memcpy(vuData + addr + 4, &val, 4);
                    if (dest & 0x2)
                        std::memcpy(vuData + addr + 8, &val, 4);
                    if (dest & 0x1)
                        std::memcpy(vuData + addr + 12, &val, 4);
                }
                return;
            }
            case 0x40: // RNEXT
                return;
            case 0x41: // RGET
                return;
            case 0x42: // RINIT
                return;
            case 0x43: // RXOR
                return;
            case 0x64: // MFP (Move From P register)
            {
                float result[4] = {state.p, state.p, state.p, state.p};
                applyDest(state.vf[vfT], result, dest);
                return;
            }
            case 0x68: // XTOP - move current VIF1 TOP into VI register
            {
                if (viT != 0)
                    state.vi[viT] = (int32_t)(state.top & 0x3FFu);
                return;
            }
            case 0x69: // XITOP - move current VIF1 ITOP into VI register
            {
                if (viT != 0)
                    state.vi[viT] = (int32_t)(state.itop & 0x3FFu);
                return;
            }
            case 0x6C: // XGKICK - send GIF packet from VU1 data memory
                beginXgkick(static_cast<uint32_t>(static_cast<uint16_t>(state.vi[viS])),
                            vuData, dataSize, sideEffects, memory);
                return;
            case 0x70: // ESADD
                return;
            case 0x71: // ERSADD
                return;
            case 0x72: // ELENG
            {
                float s = state.vf[vfS][0] * state.vf[vfS][0] + state.vf[vfS][1] * state.vf[vfS][1] + state.vf[vfS][2] * state.vf[vfS][2];
                scheduleP(std::sqrt(s), 18u);
                return;
            }
            case 0x73: // ERLENG
            {
                float s = state.vf[vfS][0] * state.vf[vfS][0] + state.vf[vfS][1] * state.vf[vfS][1] + state.vf[vfS][2] * state.vf[vfS][2];
                float len = std::sqrt(s);
                scheduleP(
                    (len != 0.0f)
                        ? (1.0f / len)
                        : std::numeric_limits<float>::max(),
                    24u);
                return;
            }
            case 0x7A: // ERCPR
            {
                int fsf = (instr >> 21) & 0x3;
                float val = state.vf[vfS][fsf];
                scheduleP(
                    (val != 0.0f)
                        ? (1.0f / val)
                        : std::numeric_limits<float>::max(),
                    12u);
                return;
            }
            case 0x7B: // WAITP
                return;
            case 0x7D: // EATAN / EATANxy / EATANxz placeholder
                return;
            default:
                return;
            }
        }
        default:
            return;
        }
    }
    default:
        break;
    }
}
