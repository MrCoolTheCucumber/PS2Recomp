#include "runtime/ps2_vu1.h"
#include "runtime/ps2_vu_clip.h"
#include "ps2_vu1_detail.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace
{
    float vuMadd(float accumulator, float multiplicand, float multiplier)
    {
        // The VU FMAC keeps the product through the following add and
        // truncates the product-sum once.  A separated host multiply and add
        // loses that intermediate precision and can differ by one VU ULP.
        return std::fma(multiplicand, multiplier, accumulator);
    }

    float vuMsub(float accumulator, float multiplicand, float multiplier)
    {
        return std::fma(-multiplicand, multiplier, accumulator);
    }

    int32_t vuFtoi(float value, float scale)
    {
        const float scaled = value * scale;
        if (std::isnan(scaled))
            return std::numeric_limits<int32_t>::min();

        // 0x4effffff is the largest positive float that converts without
        // crossing the signed 32-bit boundary. The VUs saturate positive
        // overflow instead of returning the host CVTT indefinite value.
        constexpr float kLargestInRange = 2147483520.0f;
        if (scaled > kLargestInRange)
            return std::numeric_limits<int32_t>::max();
        if (scaled <=
            static_cast<float>(
                std::numeric_limits<int32_t>::min()))
        {
            return std::numeric_limits<int32_t>::min();
        }
        return static_cast<int32_t>(scaled);
    }
}

// ============================================================================
// Upper instructions (FMAC pipeline)
// ============================================================================
void VuInterpreterBackend::execUpper(uint32_t instr)
{
    VuExecutionState &state = *m_state;
    uint8_t dest = DEST(instr);
    uint8_t ft = FT(instr);
    uint8_t fs = FS(instr);
    uint8_t fd = FD(instr);
    uint8_t op = instr & 0x3F;

    float *vd = state.vf[fd];
    const float *vs = state.vf[fs];
    const float *vt = state.vf[ft];
    float result[4];
    const uint8_t specialOp = static_cast<uint8_t>(
        (instr & 0x3u) | ((instr >> 4u) & 0x7cu));
    const bool primaryMacOp =
        op <= 0x0fu ||
        (op >= 0x18u && op <= 0x1cu) ||
        op == 0x1eu ||
        (op >= 0x20u && op <= 0x2au) ||
        (op >= 0x2cu && op <= 0x2eu);
    const bool specialMacOp =
        op >= 0x3cu &&
        (specialOp <= 0x0fu ||
         (specialOp >= 0x18u && specialOp <= 0x1cu) ||
         specialOp == 0x1eu ||
         (specialOp >= 0x20u && specialOp <= 0x2au) ||
         (specialOp >= 0x2cu && specialOp <= 0x2eu));
    const bool writesMacFlags = primaryMacOp || specialMacOp;
    const bool preserveUnselectedFlags =
        op == 0x2eu ||
        (op >= 0x3cu && specialOp == 0x2eu);
    const auto applyFmacDest = [&](float *destination)
    {
        applyDest(destination, result, dest);
        if (writesMacFlags)
        {
            scheduleFmacFlags(
                result, dest, preserveUnselectedFlags);
        }
    };
    const auto applyFmacAcc = [&]()
    {
        applyDestAcc(result, dest);
        if (writesMacFlags)
        {
            scheduleFmacFlags(
                result, dest, preserveUnselectedFlags);
        }
    };

    // Upper opcode decoding (bits 5:0 of upper word)
    switch (op)
    {
    case 0x00:
    case 0x01:
    case 0x02:
    case 0x03: // ADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + bc;
        applyFmacDest(vd);
        return;
    }
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x07: // SUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - bc;
        applyFmacDest(vd);
        return;
    }
    case 0x08:
    case 0x09:
    case 0x0A:
    case 0x0B: // MADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vuMadd(state.acc[c], vs[c], bc);
        applyFmacDest(vd);
        return;
    }
    case 0x0C:
    case 0x0D:
    case 0x0E:
    case 0x0F: // MSUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vuMsub(state.acc[c], vs[c], bc);
        applyFmacDest(vd);
        return;
    }
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13: // MAXbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > bc) ? vs[c] : bc;
        applyFmacDest(vd);
        return;
    }
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x17: // MINIbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < bc) ? vs[c] : bc;
        applyFmacDest(vd);
        return;
    }
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B: // MULbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * bc;
        applyFmacDest(vd);
        return;
    }
    case 0x1C: // MULq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * state.q;
        applyFmacDest(vd);
        return;
    case 0x1D: // MAXi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > state.i) ? vs[c] : state.i;
        applyFmacDest(vd);
        return;
    case 0x1E: // MULi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * state.i;
        applyFmacDest(vd);
        return;
    case 0x1F: // MINIi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < state.i) ? vs[c] : state.i;
        applyFmacDest(vd);
        return;
    case 0x20: // ADDq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + state.q;
        applyFmacDest(vd);
        return;
    case 0x21: // MADDq
        for (int c = 0; c < 4; c++)
            result[c] = vuMadd(state.acc[c], vs[c], state.q);
        applyFmacDest(vd);
        return;
    case 0x22: // ADDi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + state.i;
        applyFmacDest(vd);
        return;
    case 0x23: // MADDi
        for (int c = 0; c < 4; c++)
            result[c] = vuMadd(state.acc[c], vs[c], state.i);
        applyFmacDest(vd);
        return;
    case 0x24: // SUBq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - state.q;
        applyFmacDest(vd);
        return;
    case 0x25: // MSUBq
        for (int c = 0; c < 4; c++)
            result[c] = vuMsub(state.acc[c], vs[c], state.q);
        applyFmacDest(vd);
        return;
    case 0x26: // SUBi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - state.i;
        applyFmacDest(vd);
        return;
    case 0x27: // MSUBi
        for (int c = 0; c < 4; c++)
            result[c] = vuMsub(state.acc[c], vs[c], state.i);
        applyFmacDest(vd);
        return;
    case 0x28: // ADD
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + vt[c];
        applyFmacDest(vd);
        return;
    case 0x29: // MADD
        for (int c = 0; c < 4; c++)
            result[c] = vuMadd(state.acc[c], vs[c], vt[c]);
        applyFmacDest(vd);
        return;
    case 0x2A: // MUL
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * vt[c];
        applyFmacDest(vd);
        return;
    case 0x2B: // MAX
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > vt[c]) ? vs[c] : vt[c];
        applyFmacDest(vd);
        return;
    case 0x2C: // SUB
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - vt[c];
        applyFmacDest(vd);
        return;
    case 0x2D: // MSUB
        for (int c = 0; c < 4; c++)
            result[c] = vuMsub(state.acc[c], vs[c], vt[c]);
        applyFmacDest(vd);
        return;
    case 0x2E: // OPMSUB
        result[0] = state.acc[0] - vs[1] * vt[2];
        result[1] = state.acc[1] - vs[2] * vt[0];
        result[2] = state.acc[2] - vs[0] * vt[1];
        result[3] = 0.0f;
        applyFmacDest(vd);
        return;
    case 0x2F: // MINI
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < vt[c]) ? vs[c] : vt[c];
        applyDest(vd, result, dest);
        return;

    // Upper special group (low op 0x3C..0x3F).
    // Like lower1 special, the real selector is not just bits 5:0.  Dobie decodes:
    //   op = (instr & 0x3) | ((instr >> 4) & 0x7C)
    // Several instructions in this group also use FT as the destination, not FD.
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    {
        float *vtDest = state.vf[ft];

        switch (specialOp)
        {
        case 0x00:
        case 0x01:
        case 0x02:
        case 0x03: // ADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + bc;
            applyFmacAcc();
            return;
        }
        case 0x04:
        case 0x05:
        case 0x06:
        case 0x07: // SUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - bc;
            applyFmacAcc();
            return;
        }
        case 0x08:
        case 0x09:
        case 0x0A:
        case 0x0B: // MADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vuMadd(state.acc[c], vs[c], bc);
            applyFmacAcc();
            return;
        }
        case 0x0C:
        case 0x0D:
        case 0x0E:
        case 0x0F: // MSUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vuMsub(state.acc[c], vs[c], bc);
            applyFmacAcc();
            return;
        }
        case 0x10: // ITOF0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &vs[c], 4);
                result[c] = static_cast<float>(iv);
            }
            applyFmacDest(vtDest);
            return;
        case 0x11: // ITOF4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &vs[c], 4);
                result[c] = static_cast<float>(iv) / 16.0f;
            }
            applyFmacDest(vtDest);
            return;
        case 0x12: // ITOF12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &vs[c], 4);
                result[c] = static_cast<float>(iv) / 4096.0f;
            }
            applyFmacDest(vtDest);
            return;
        case 0x13: // ITOF15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &vs[c], 4);
                result[c] = static_cast<float>(iv) / 32768.0f;
            }
            applyFmacDest(vtDest);
            return;
        case 0x14: // FTOI0
            for (int c = 0; c < 4; c++)
            {
                const int32_t iv = vuFtoi(vs[c], 1.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyFmacDest(vtDest);
            return;
        case 0x15: // FTOI4
            for (int c = 0; c < 4; c++)
            {
                const int32_t iv = vuFtoi(vs[c], 16.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyFmacDest(vtDest);
            return;
        case 0x16: // FTOI12
            for (int c = 0; c < 4; c++)
            {
                const int32_t iv = vuFtoi(vs[c], 4096.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyFmacDest(vtDest);
            return;
        case 0x17: // FTOI15
            for (int c = 0; c < 4; c++)
            {
                const int32_t iv = vuFtoi(vs[c], 32768.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyFmacDest(vtDest);
            return;
        case 0x18:
        case 0x19:
        case 0x1A:
        case 0x1B: // MULAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * bc;
            applyFmacAcc();
            return;
        }
        case 0x1C: // MULAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * state.q;
            applyFmacAcc();
            return;
        case 0x1D: // ABS
            for (int c = 0; c < 4; c++)
                result[c] = std::fabs(vs[c]);
            applyFmacDest(vtDest);
            return;
        case 0x1E: // MULAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * state.i;
            applyFmacAcc();
            return;
        case 0x1F: // CLIP
        {
            uint32_t bits[4] = {};
            uint32_t wBits = 0u;
            std::memcpy(bits, vs, sizeof(bits));
            std::memcpy(&wBits, &vt[3], sizeof(wBits));
            scheduleClipFlags(Ps2VuUpdateClipFlags(
                state.pipeline.workingClip,
                bits[0], bits[1], bits[2], wBits));
            return;
        }
        case 0x20: // ADDAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + state.q;
            applyFmacAcc();
            return;
        case 0x21: // MADDAq
            for (int c = 0; c < 4; c++)
                result[c] = vuMadd(state.acc[c], vs[c], state.q);
            applyFmacAcc();
            return;
        case 0x22: // ADDAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + state.i;
            applyFmacAcc();
            return;
        case 0x23: // MADDAi
            for (int c = 0; c < 4; c++)
                result[c] = vuMadd(state.acc[c], vs[c], state.i);
            applyFmacAcc();
            return;
        case 0x24: // SUBAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - state.q;
            applyFmacAcc();
            return;
        case 0x25: // MSUBAq
            for (int c = 0; c < 4; c++)
                result[c] = vuMsub(state.acc[c], vs[c], state.q);
            applyFmacAcc();
            return;
        case 0x26: // SUBAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - state.i;
            applyFmacAcc();
            return;
        case 0x27: // MSUBAi
            for (int c = 0; c < 4; c++)
                result[c] = vuMsub(state.acc[c], vs[c], state.i);
            applyFmacAcc();
            return;
        case 0x28: // ADDA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + vt[c];
            applyFmacAcc();
            return;
        case 0x29: // MADDA
            for (int c = 0; c < 4; c++)
                result[c] = vuMadd(state.acc[c], vs[c], vt[c]);
            applyFmacAcc();
            return;
        case 0x2A: // MULA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * vt[c];
            applyFmacAcc();
            return;
        case 0x2C: // SUBA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - vt[c];
            applyFmacAcc();
            return;
        case 0x2D: // MSUBA
            for (int c = 0; c < 4; c++)
                result[c] = vuMsub(state.acc[c], vs[c], vt[c]);
            applyFmacAcc();
            return;
        case 0x2E: // OPMULA
            result[0] = vs[1] * vt[2];
            result[1] = vs[2] * vt[0];
            result[2] = vs[0] * vt[1];
            result[3] = 0.0f;
            applyFmacAcc();
            return;
        case 0x2F:
        case 0x30: // NOP
            return;
        default:
            return;
        }
    }

    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    default:
        return;
    }
}
