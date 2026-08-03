#include "ps2recomp/Translators/cop0_translator.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/codegen_helpers.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"

#include <fmt/format.h>
#include <sstream>
#include <cmath>


namespace ps2recomp
{
    Cop0Translator::Cop0Translator(CodeGenerator &codeGenerator)
        : m_codeGenerator(codeGenerator)
    {
    }

    std::string Cop0Translator::translate(const Instruction &inst)
    {
        uint32_t format = inst.rs; // Format field
        uint32_t rt = inst.rt;     // GPR register
        uint32_t rd = inst.rd;     // COP0 register

        switch (format)
        {
        case COP0_MF:
            switch (rd)
            {
            case COP0_REG_INDEX:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_index);", rt);
            case COP0_REG_RANDOM:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_random);", rt);
            case COP0_REG_ENTRYLO0:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_entrylo0);", rt);
            case COP0_REG_ENTRYLO1:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_entrylo1);", rt);
            case COP0_REG_CONTEXT:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_context);", rt);
            case COP0_REG_PAGEMASK:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_pagemask);", rt);
            case COP0_REG_WIRED:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_wired);", rt);
            case COP0_REG_BADVADDR:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_badvaddr);", rt);
            case COP0_REG_COUNT:
                return fmt::format(
                    "{{ const uint32_t value = runtime->readCop0Count(rdram, ctx); "
                    "SET_GPR_S32(ctx, {}, (int32_t)value); }}",
                    rt);
            case COP0_REG_ENTRYHI:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_entryhi);", rt);
            case COP0_REG_COMPARE:
                return fmt::format(
                    "SET_GPR_S32(ctx, {}, (int32_t)runtime->readCop0Compare(ctx));",
                    rt);
            case COP0_REG_STATUS:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_status);", rt);
            case COP0_REG_CAUSE:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_cause);", rt);
            case COP0_REG_EPC:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_epc);", rt);
            case COP0_REG_PRID:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_prid);", rt);
            case COP0_REG_CONFIG:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_config);", rt);
            case COP0_REG_BADPADDR:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_badpaddr);", rt);
            case COP0_REG_DEBUG:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_debug);", rt);
            case COP0_REG_PERF:
                return fmt::format(
                    "SET_GPR_S32(ctx, {}, (int32_t)runtime->readCop0Performance("
                    "rdram, ctx, {}u));",
                    rt,
                    inst.raw & 0x3fu);
            case COP0_REG_TAGLO:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_taglo);", rt);
            case COP0_REG_TAGHI:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_taghi);", rt);
            case COP0_REG_ERROREPC:
                return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ctx->cop0_errorepc);", rt);
            default:
                return fmt::format("SET_GPR_S32(ctx, {}, 0);  // Unimplemented COP0 register {}", rt, rd);
            }
        case COP0_MT:
            switch (rd)
            {
            case COP0_REG_INDEX:
                return fmt::format("ctx->cop0_index = GPR_U32(ctx, {}) & 0x3F;", rt);
            case COP0_REG_RANDOM:
                return "// MTC0 to RANDOM register ignored (read-only)";
            case COP0_REG_ENTRYLO0:
                return fmt::format("ctx->cop0_entrylo0 = GPR_U32(ctx, {}) & 0x83FFFFFF;", rt);
            case COP0_REG_ENTRYLO1:
                return fmt::format("ctx->cop0_entrylo1 = GPR_U32(ctx, {}) & 0x03FFFFFF;", rt);
            case COP0_REG_CONTEXT:
                return fmt::format("ctx->cop0_context = (ctx->cop0_context & 0x007FFFF0) | (GPR_U32(ctx, {}) & 0xFF800000);", rt);
            case COP0_REG_PAGEMASK:
                return fmt::format("ctx->cop0_pagemask = GPR_U32(ctx, {}) & 0x01FFE000;", rt);
            case COP0_REG_WIRED:
                return fmt::format("ctx->cop0_wired = GPR_U32(ctx, {}) & 0x3F; ctx->cop0_random = 47;", rt);
            case COP0_REG_BADVADDR:
                return "// MTC0 to BADVADDR register ignored (read-only)";
            case COP0_REG_COUNT:
                return fmt::format(
                    "runtime->writeCop0Count(rdram, ctx, GPR_U32(ctx, {}));",
                    rt);
            case COP0_REG_ENTRYHI:
                return fmt::format("ctx->cop0_entryhi = GPR_U32(ctx, {}) & 0xFFFFE0FF;", rt);
            case COP0_REG_COMPARE:
                return fmt::format(
                    "runtime->writeCop0Compare(rdram, ctx, GPR_U32(ctx, {}));",
                    rt);
            case COP0_REG_STATUS:
                return fmt::format(
                    "runtime->writeCop0Status(rdram, ctx, GPR_U32(ctx, {}));",
                    rt);
            case COP0_REG_CAUSE:
                return fmt::format("ctx->cop0_cause = (ctx->cop0_cause & ~0x00000300) | (GPR_U32(ctx, {}) & 0x00000300);", rt);
            case COP0_REG_EPC:
                return fmt::format("ctx->cop0_epc = GPR_U32(ctx, {});", rt);
            case COP0_REG_PRID:
                return "// MTC0 to PRID register ignored (read-only)";
            case COP0_REG_CONFIG:
                return fmt::format("ctx->cop0_config = (ctx->cop0_config & 0x70000FC0u) | (GPR_U32(ctx, {}) & 0x00073007u);", rt);
            case COP0_REG_BADPADDR:
                return "// MTC0 to BADPADDR register ignored (read-only)";
            case COP0_REG_DEBUG:
                return fmt::format("ctx->cop0_debug = GPR_U32(ctx, {});", rt);
            case COP0_REG_PERF:
                return fmt::format(
                    "runtime->writeCop0Performance(rdram, ctx, {}u, GPR_U32(ctx, {}));",
                    inst.raw & 0x3fu, rt);
            case COP0_REG_TAGLO:
                return fmt::format("ctx->cop0_taglo = GPR_U32(ctx, {});", rt);
            case COP0_REG_TAGHI:
                return fmt::format("ctx->cop0_taghi = GPR_U32(ctx, {});", rt);
            case COP0_REG_ERROREPC:
                return fmt::format("ctx->cop0_errorepc = GPR_U32(ctx, {});", rt);
            default:
                return fmt::format("// Unimplemented MTC0 to COP0 {}", rd);
            }
        case COP0_BC:
            return fmt::format("// BC0 (Condition: 0x{:X}) - Handled by branch logic", rt);
        case COP0_CO:
        {
            uint8_t function = FUNCTION(inst.raw);
            switch (function)
            {
            case COP0_CO_TLBR:
                return fmt::format("runtime->handleTLBR(rdram, ctx);");
            case COP0_CO_TLBWI:
                return fmt::format("runtime->handleTLBWI(rdram, ctx);");
            case COP0_CO_TLBWR:
                return fmt::format("runtime->handleTLBWR(rdram, ctx);");
            case COP0_CO_TLBP:
                return fmt::format("runtime->handleTLBP(rdram, ctx);");
            case COP0_CO_ERET:
                return fmt::format(
                    "runtime->handleCop0Eret(rdram, ctx); \n"
                    "return;"
                );
            case COP0_CO_EI:
                return fmt::format(
                    "runtime->handleCop0Ei(rdram, ctx);");
            case COP0_CO_DI:
                return fmt::format(
                    "runtime->handleCop0Di(rdram, ctx);");
            default:
                return m_codeGenerator.emitUnhandledInstruction(inst, fmt::format("Unhandled COP0 CO-OP: 0x{:X}", function));
            }
        }
        default:
            return m_codeGenerator.emitUnhandledInstruction(inst, fmt::format("Unhandled COP0 instruction format: 0x{:X}", format));
        }
    }

}
