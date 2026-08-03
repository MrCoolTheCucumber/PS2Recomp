#include "ps2recomp/Translators/instruction_translator.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/codegen_helpers.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"
#include "ps2recomp/control_flow_utils.h"
#include "runtime/ps2_address.h"

#include <fmt/format.h>

namespace ps2recomp
{
    namespace
    {
        std::string addressLiteral(uint32_t address)
        {
            return fmt::format("0x{:X}u", address);
        }

        uint32_t memoryAccessSize(int width)
        {
            return static_cast<uint32_t>(width / 8);
        }
    }

    InstructionTranslator::InstructionTranslator(CodeGenerator &codeGenerator)
        : m_codeGenerator(codeGenerator)
    {
    }

    MemoryAccessHint InstructionTranslator::effectiveMemoryHintFor(const Instruction &inst, const MemoryAccessHint &memoryHint) const
    {
        MemoryAccessHint effectiveMemoryHint = memoryHint;
        if (inst.isMmio)
        {
            effectiveMemoryHint.hasAddress = true;
            effectiveMemoryHint.address = inst.mmioAddress;
        }

        return effectiveMemoryHint;
    }

    std::string InstructionTranslator::translateMemoryRead(const Instruction &inst,
                                                           const MemoryAccessHint &memoryHint,
                                                           int width,
                                                           const std::string &addr) const
    {
        if (memoryHint.hasAddress)
        {
            const uint32_t resolvedAddress = memoryHint.address;
            const std::string resolvedAddressExpr = addressLiteral(resolvedAddress);
            if (inst.isMmio ||
                !Ps2CanUseFastRdramAccess(resolvedAddress, memoryAccessSize(width)))
            {
                return fmt::format("runtime->Load{}(rdram, ctx, {})", width, resolvedAddressExpr);
            }
            return fmt::format("READ{}({})", width, resolvedAddressExpr);
        }

        if (inst.isMmio)
        {
            return fmt::format("runtime->Load{}(rdram, ctx, {})", width, addr);
        }
        return fmt::format("READ{}({})", width, addr);
    }

    std::string InstructionTranslator::translateMemoryWrite(const Instruction &inst,
                                                            const MemoryAccessHint &memoryHint,
                                                            int width,
                                                            const std::string &addr,
                                                            const std::string &value) const
    {
        if (memoryHint.hasAddress)
        {
            const uint32_t resolvedAddress = memoryHint.address;
            const std::string resolvedAddressExpr = addressLiteral(resolvedAddress);
            if (inst.isMmio ||
                !Ps2CanUseFastRdramAccess(resolvedAddress, memoryAccessSize(width)))
            {
                return fmt::format("runtime->Store{}(rdram, ctx, {}, {})", width, resolvedAddressExpr, value);
            }
            return fmt::format(
                "WRITE{}({}, {})",
                width,
                resolvedAddressExpr,
                value);
        }

        if (inst.isMmio)
        {
            return fmt::format("runtime->Store{}(rdram, ctx, {}, {})", width, addr, value);
        }
        return fmt::format("WRITE{}({}, {})", width, addr, value);
    }

    std::string InstructionTranslator::translate(
        const Instruction &inst,
        const MemoryAccessHint &memoryHint)
    {
        const std::string operation =
            translateUnchecked(inst, memoryHint);
        const int coprocessor =
            eeCoprocessorForOpcode(inst.opcode);
        if (coprocessor < 0)
        {
            return operation;
        }

        return fmt::format(
            "{{ runtime->RequireCoprocessorUsable(ctx, {}u);\n{}\n}}",
            coprocessor,
            operation);
    }

    std::string InstructionTranslator::translateUnchecked(
        const Instruction &inst,
        const MemoryAccessHint &memoryHint)
    {
        if (inst.isMMI)
        {
            return m_codeGenerator.translateMMIInstruction(inst);
        }

        const MemoryAccessHint effectiveMemoryHint = effectiveMemoryHintFor(inst, memoryHint);

        auto genRead = [&](int width, const std::string &addr)
        {
            return translateMemoryRead(inst, effectiveMemoryHint, width, addr);
        };

        auto genWrite = [&](int width, const std::string &addr, const std::string &val)
        {
            return translateMemoryWrite(inst, effectiveMemoryHint, width, addr, val);
        };

        auto genAlignedQuadwordRead = [&](const std::string &addr)
        {
            MemoryAccessHint alignedHint = effectiveMemoryHint;
            if (alignedHint.hasAddress)
            {
                alignedHint.address &= ~0xFu;
            }
            return translateMemoryRead(inst, alignedHint, 128, fmt::format("({} & ~0xFu)", addr));
        };

        auto genAlignedQuadwordWrite = [&](const std::string &addr, const std::string &val)
        {
            MemoryAccessHint alignedHint = effectiveMemoryHint;
            if (alignedHint.hasAddress)
            {
                alignedHint.address &= ~0xFu;
            }
            return translateMemoryWrite(inst, alignedHint, 128, fmt::format("({} & ~0xFu)", addr), val);
        };

        switch (inst.opcode)
        {
        case OPCODE_SPECIAL:
            return m_codeGenerator.translateSpecialInstruction(inst);
        case OPCODE_REGIMM:
            return m_codeGenerator.translateRegimmInstruction(inst);
        case OPCODE_COP0:
            return m_codeGenerator.translateCOP0Instruction(inst);
        case OPCODE_COP1:
            return m_codeGenerator.translateFPUInstruction(inst);
        case OPCODE_COP2:
        {
            const bool transfer =
                inst.rs == COP2_QMFC2 ||
                inst.rs == COP2_CFC2 ||
                inst.rs == COP2_QMTC2 ||
                inst.rs == COP2_CTC2;
            const bool interlocked =
                inst.rs >= COP2_CO ||
                (transfer && (inst.raw & 1u) != 0u);
            const bool finish =
                interlocked ||
                inst.vu0SyncMode == Vu0SyncMode::Finish;
            const std::string operation =
                m_codeGenerator.translateVUInstruction(inst);
            if (!finish &&
                inst.vu0SyncMode == Vu0SyncMode::Skip)
            {
                return fmt::format(
                    "{{ {}\nctx->enforceVu0RegisterInvariants(); }}",
                    operation);
            }
            return fmt::format(
                "{{ runtime->synchronizeVU0Microprogram(rdram, ctx, {}); "
                "{}\nctx->enforceVu0RegisterInvariants(); }}",
                finish ? "true" : "false",
                operation);
        }
        case OPCODE_ADDI:
            return fmt::format(
                "{{ uint32_t tmp; bool ov; "
                "ADD32_OV(GPR_U32(ctx, {}), (int32_t){}, tmp, ov); "
                "if (ov) runtime->SignalException(ctx, EXCEPTION_INTEGER_OVERFLOW); "
                "else SET_GPR_S32(ctx, {}, (int32_t)tmp); }}",
                inst.rs, inst.simmediate, inst.rt);

        case OPCODE_ADDIU:
            if (inst.rt == 0)
                return "// NOP (addiu $zero, ...)";
            return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)ADD32(GPR_U32(ctx, {}), {}));", inst.rt, inst.rs, inst.simmediate);
        case OPCODE_SLTI:
            return fmt::format("SET_GPR_U64(ctx, {}, ((int64_t)GPR_S64(ctx, {}) < (int64_t)(int32_t){}) ? 1 : 0);", inst.rt, inst.rs, inst.simmediate);
        case OPCODE_SLTIU:
            return fmt::format("SET_GPR_U64(ctx, {}, ((uint64_t)GPR_U64(ctx, {}) < (uint64_t)(int64_t)(int32_t){}) ? 1 : 0);", inst.rt, inst.rs, inst.simmediate);
        case OPCODE_ANDI:
            return fmt::format("SET_GPR_U64(ctx, {}, GPR_U64(ctx, {}) & (uint64_t)(uint16_t){});", inst.rt, inst.rs, inst.immediate);
        case OPCODE_ORI:
            return fmt::format("SET_GPR_U64(ctx, {}, GPR_U64(ctx, {}) | (uint64_t)(uint16_t){});", inst.rt, inst.rs, inst.immediate);
        case OPCODE_XORI:
            return fmt::format("SET_GPR_U64(ctx, {}, GPR_U64(ctx, {}) ^ (uint64_t)(uint16_t){});", inst.rt, inst.rs, inst.immediate);
        case OPCODE_LUI:
            return fmt::format("SET_GPR_S32(ctx, {}, (int32_t)((uint32_t){} << 16));", inst.rt, inst.immediate);
        case OPCODE_LB:
            return fmt::format("SET_GPR_S32(ctx, {}, (int8_t){});", inst.rt, genRead(8, fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate)));
        case OPCODE_LH:
            return fmt::format("SET_GPR_S32(ctx, {}, (int16_t){});", inst.rt, genRead(16, fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate)));
        case OPCODE_LW:
            return fmt::format("SET_GPR_S32(ctx, {}, (int32_t){});", inst.rt, genRead(32, fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate)));
        case OPCODE_LBU:
            return fmt::format("SET_GPR_U32(ctx, {}, (uint8_t){});", inst.rt, genRead(8, fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate)));
        case OPCODE_LHU:
            return fmt::format("SET_GPR_U32(ctx, {}, (uint16_t){});", inst.rt, genRead(16, fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate)));
        case OPCODE_LWU:
            return fmt::format("SET_GPR_U64(ctx, {}, (uint64_t)(uint32_t){});", inst.rt, genRead(32, fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate)));
        case OPCODE_SB:
            return genWrite(8, fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate), fmt::format("(uint8_t)GPR_U32(ctx, {})", inst.rt)) + ";";
        case OPCODE_SH:
            return genWrite(16, fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate), fmt::format("(uint16_t)GPR_U32(ctx, {})", inst.rt)) + ";";
        case OPCODE_SW:
            return genWrite(32, fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate), fmt::format("GPR_U32(ctx, {})", inst.rt)) + ";";
        case OPCODE_LQ:
            return fmt::format("SET_GPR_VEC(ctx, {}, {});", inst.rt, genAlignedQuadwordRead(fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate)));
        case OPCODE_SQ:
            return genAlignedQuadwordWrite(fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate), fmt::format("GPR_VEC(ctx, {})", inst.rt)) + ";";
        case OPCODE_LD:
            return fmt::format("SET_GPR_U64(ctx, {}, {});", inst.rt, genRead(64, fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate)));
        case OPCODE_SD:
            return genWrite(64, fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate), fmt::format("GPR_U64(ctx, {})", inst.rt)) + ";";
        case OPCODE_LWC1:
            return fmt::format("{{ uint32_t bits = {}; float f; std::memcpy(&f, &bits, sizeof(f)); ctx->f[{}] = f; }}", genRead(32, fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate)), inst.rt);
        case OPCODE_SWC1:
            return fmt::format(
                "{{ float f = ctx->f[{}]; uint32_t bits; std::memcpy(&bits, &f, sizeof(bits)); {}; }}",
                inst.rt,
                genWrite(32, fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate), "bits"));
        case OPCODE_LDC2:
            if (inst.vu0SyncMode == Vu0SyncMode::Skip)
            {
                return fmt::format(
                    "{{ ctx->vu0_vf[{}] = _mm_castsi128_ps({}); "
                    "ctx->enforceVu0RegisterInvariants(); }}",
                    inst.rt,
                    genRead(128, fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate)));
            }
            return fmt::format(
                "{{ runtime->synchronizeVU0Microprogram(rdram, ctx, {}); "
                "ctx->vu0_vf[{}] = _mm_castsi128_ps({}); "
                "ctx->enforceVu0RegisterInvariants(); }}",
                inst.vu0SyncMode == Vu0SyncMode::Finish
                    ? "true"
                    : "false",
                inst.rt,
                genRead(128, fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate)));
        case OPCODE_SDC2:
            if (inst.vu0SyncMode == Vu0SyncMode::Skip)
            {
                return genWrite(
                           128,
                           fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate),
                           fmt::format("_mm_castps_si128(ctx->vu0_vf[{}])", inst.rt)) +
                       ";";
            }
            return fmt::format(
                "{{ runtime->synchronizeVU0Microprogram(rdram, ctx, {}); {}; }}",
                inst.vu0SyncMode == Vu0SyncMode::Finish
                    ? "true"
                    : "false",
                genWrite(
                    128,
                    fmt::format("ADD32(GPR_U32(ctx, {}), {})", inst.rs, inst.simmediate),
                    fmt::format("_mm_castps_si128(ctx->vu0_vf[{}])", inst.rt)));
        case OPCODE_DADDI:
            return fmt::format(
                "{{ uint64_t result; bool overflow; "
                "ADD64_OV(GPR_U64(ctx, {}), (uint64_t)(int64_t)(int32_t){}, result, overflow); "
                "if (overflow) runtime->SignalException(ctx, EXCEPTION_INTEGER_OVERFLOW); "
                "else SET_GPR_U64(ctx, {}, result); }}",
                inst.rs, inst.simmediate, inst.rt);
        case OPCODE_DADDIU:
            return fmt::format(
                "SET_GPR_U64(ctx, {}, ADD64(GPR_U64(ctx, {}), (uint64_t)(int64_t)(int32_t){}));",
                inst.rt, inst.rs, inst.simmediate);
        case OPCODE_J:
            return fmt::format("// J 0x{:X} - Handled by branch logic", buildAbsoluteJumpTarget(inst.address, inst.target));
        case OPCODE_JAL:
            return fmt::format("// JAL 0x{:X} - Handled by branch logic", buildAbsoluteJumpTarget(inst.address, inst.target));
        case OPCODE_BEQ:
        case OPCODE_BNE:
        case OPCODE_BLEZ:
        case OPCODE_BGTZ:
        case OPCODE_BEQL:
        case OPCODE_BNEL:
        case OPCODE_BLEZL:
        case OPCODE_BGTZL:
            return fmt::format("// Likely branch instruction at 0x{:X} - Handled by branch logic", inst.address);

        case OPCODE_LDL:
            return fmt::format("{{ uint32_t addr = ADD32(GPR_U32(ctx, {}), {}); "
                               "uint32_t aligned_addr = addr & ~7u; "
                               "uint32_t offset = addr & 7u; "
                               "uint64_t mem = {}; "
                               "uint32_t shift = (7u - offset) << 3; "
                               "uint64_t keepMask = (shift == 0) ? 0ull : ((1ull << shift) - 1ull); "
                               "SET_GPR_U64(ctx, {}, (GPR_U64(ctx, {}) & keepMask) | (mem << shift)); }}",
                               inst.rs, inst.simmediate, genRead(64, "aligned_addr"), inst.rt, inst.rt);

        case OPCODE_LDR:
            return fmt::format("{{ uint32_t addr = ADD32(GPR_U32(ctx, {}), {}); "
                               "uint32_t aligned_addr = addr & ~7u; "
                               "uint32_t offset = addr & 7u; "
                               "uint64_t mem = {}; "
                               "uint32_t shift = offset << 3; "
                               "uint64_t keepMask = (offset == 0) ? 0ull : (0xFFFFFFFFFFFFFFFFull << ((8u - offset) << 3)); "
                               "SET_GPR_U64(ctx, {}, (GPR_U64(ctx, {}) & keepMask) | (mem >> shift)); }}",
                               inst.rs, inst.simmediate, genRead(64, "aligned_addr"), inst.rt, inst.rt);

        case OPCODE_LWL:
            return fmt::format("{{ uint32_t addr = ADD32(GPR_U32(ctx, {}), {}); "
                               "uint32_t aligned_addr = addr & ~3u; "
                               "uint32_t offset = addr & 3u; "
                               "uint32_t mem = {}; "
                               "uint32_t shift = (3u - offset) << 3; "
                               "uint32_t keepMask = (shift == 0) ? 0u : ((1u << shift) - 1u); "
                               "uint32_t merged = (GPR_U32(ctx, {}) & keepMask) | (mem << shift); "
                               "SET_GPR_S32(ctx, {}, (int32_t)merged); }}",
                               inst.rs, inst.simmediate, genRead(32, "aligned_addr"), inst.rt, inst.rt);

        case OPCODE_LWR:
            return fmt::format("{{ uint32_t addr = ADD32(GPR_U32(ctx, {}), {}); "
                               "uint32_t aligned_addr = addr & ~3u; "
                               "uint32_t offset = addr & 3u; "
                               "uint32_t mem = {}; "
                               "uint32_t shift = offset << 3; "
                               "uint32_t keepMask = (offset == 0) ? 0u : (0xFFFFFFFFu << ((4u - offset) << 3)); "
                               "uint32_t merged32 = (GPR_U32(ctx, {}) & keepMask) | (mem >> shift); "
                               "uint64_t merged64 = (GPR_U64(ctx, {}) & 0xFFFFFFFF00000000ull) | (uint64_t)merged32; "
                               "if (offset == 0) merged64 = (uint64_t)(int64_t)(int32_t)merged32; "
                               "SET_GPR_U64(ctx, {}, merged64); }}",
                               inst.rs, inst.simmediate, genRead(32, "aligned_addr"),
                               inst.rt, inst.rt, inst.rt);

        case OPCODE_SWL:
            return fmt::format("{{ uint32_t addr = ADD32(GPR_U32(ctx, {}), {}); "
                               "uint32_t aligned_addr = addr & ~3u; "
                               "uint32_t offset = addr & 3u; "
                               "uint32_t shift = (3u - offset) << 3; "
                               "uint8_t byte_enable = (uint8_t)((1u << (offset + 1u)) - 1u); "
                               "uint32_t value = GPR_U32(ctx, {}) >> shift; "
                               "WRITE_MASKED32(aligned_addr, value, byte_enable); }}",
                               inst.rs, inst.simmediate, inst.rt);

        case OPCODE_SWR:
            return fmt::format("{{ uint32_t addr = ADD32(GPR_U32(ctx, {}), {}); "
                               "uint32_t aligned_addr = addr & ~3u; "
                               "uint32_t offset = addr & 3u; "
                               "uint32_t shift = offset << 3; "
                               "uint8_t byte_enable = (uint8_t)((0xFu << offset) & 0xFu); "
                               "uint32_t value = GPR_U32(ctx, {}) << shift; "
                               "WRITE_MASKED32(aligned_addr, value, byte_enable); }}",
                               inst.rs, inst.simmediate, inst.rt);

        case OPCODE_SDL:
            return fmt::format("{{ uint32_t addr = ADD32(GPR_U32(ctx, {}), {}); "
                               "uint32_t aligned_addr = addr & ~7u; "
                               "uint32_t offset = addr & 7u; "
                               "uint32_t shift = (7u - offset) << 3; "
                               "uint8_t byte_enable = (uint8_t)((1u << (offset + 1u)) - 1u); "
                               "uint64_t value = GPR_U64(ctx, {}) >> shift; "
                               "WRITE_MASKED64(aligned_addr, value, byte_enable); }}",
                               inst.rs, inst.simmediate, inst.rt);

        case OPCODE_SDR:
            return fmt::format("{{ uint32_t addr = ADD32(GPR_U32(ctx, {}), {}); "
                               "uint32_t aligned_addr = addr & ~7u; "
                               "uint32_t offset = addr & 7u; "
                               "uint32_t shift = offset << 3; "
                               "uint8_t byte_enable = (uint8_t)((0xFFu << offset) & 0xFFu); "
                               "uint64_t value = GPR_U64(ctx, {}) << shift; "
                               "WRITE_MASKED64(aligned_addr, value, byte_enable); }}",
                               inst.rs, inst.simmediate, inst.rt);
        case OPCODE_CACHE:
            return fmt::format(
                "runtime->handleEeCacheOperation(rdram, ctx, 0x{:02X}u, "
                "ADD32(GPR_U32(ctx, {}), 0x{:08X}u));",
                inst.rt,
                inst.rs,
                inst.simmediate);
        case OPCODE_PREF:
            return "// PREF instruction (ignored)";
        case OPCODE_LL:
        case OPCODE_LLD:
        case OPCODE_SC:
        case OPCODE_SCD:
            // The EE Core omits the MIPS III load-linked/store-conditional
            // instructions. All four primary encodings are precise RI faults.
            return "runtime->SignalException(ctx, EXCEPTION_RESERVED_INSTRUCTION);";
        default:
            return m_codeGenerator.emitUnhandledInstruction(inst, fmt::format("Unhandled opcode: 0x{:X}", inst.opcode));
        }
    }

}
