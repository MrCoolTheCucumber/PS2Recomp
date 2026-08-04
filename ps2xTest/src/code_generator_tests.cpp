#include "MiniTest.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/ee_cycle_model.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/ps2_recompiler.h"
#include "ps2recomp/r5900_decoder.h"
#include "ps2recomp/recompiler_reporter.h"
#include "ps2recomp/types.h"
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

using namespace ps2recomp;

static Instruction makeBranch(uint32_t address, uint32_t targetOffsetWords)
{
    Instruction inst;
    inst.address = address;
    inst.raw = 0x10000000 | (address & 0xFFFF); // arbitrary debug value
    inst.opcode = OPCODE_BEQ;
    inst.rs = 1;
    inst.rt = 1; // always equal
    inst.simmediate = static_cast<uint32_t>(targetOffsetWords);
    inst.isBranch = true;
    inst.hasDelaySlot = true;
    return inst;
}

static Instruction makeCop0Branch(
    uint32_t address,
    uint8_t condition,
    int16_t targetOffsetWords)
{
    const uint32_t raw =
        (OPCODE_COP0 << 26) |
        (COP0_BC << 21) |
        (static_cast<uint32_t>(condition) << 16) |
        static_cast<uint16_t>(targetOffsetWords);
    return R5900Decoder{}.decodeInstruction(address, raw, false);
}

static Instruction makeCop2Branch(
    uint32_t address,
    uint8_t condition,
    int16_t targetOffsetWords)
{
    const uint32_t raw =
        (OPCODE_COP2 << 26) |
        (COP2_BC << 21) |
        (static_cast<uint32_t>(condition) << 16) |
        static_cast<uint16_t>(targetOffsetWords);
    return R5900Decoder{}.decodeInstruction(address, raw, false);
}

static Instruction makeNop(uint32_t address)
{
    Instruction inst;
    inst.address = address;
    inst.raw = 0;
    inst.opcode = OPCODE_ADDIU;
    inst.rt = 0; // encode as nop in translator
    inst.hasDelaySlot = false;
    return inst;
}

static uint32_t signExtend16(uint16_t value)
{
    return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(value)));
}

static size_t countOccurrences(
    const std::string &value, const std::string &needle)
{
    size_t count = 0u;
    size_t offset = 0u;
    while ((offset = value.find(needle, offset)) != std::string::npos)
    {
        ++count;
        offset += needle.size();
    }
    return count;
}

static Instruction makeIType(uint32_t address, uint32_t opcode, uint8_t rs, uint8_t rt, uint16_t immediate)
{
    Instruction inst{};
    inst.address = address;
    inst.opcode = opcode;
    inst.rs = rs;
    inst.rt = rt;
    inst.immediate = immediate;
    inst.simmediate = signExtend16(immediate);
    inst.raw = (opcode << 26) | (static_cast<uint32_t>(rs) << 21) |
               (static_cast<uint32_t>(rt) << 16) | immediate;
    return inst;
}

static Instruction makeLui(uint32_t address, uint8_t rt, uint16_t immediate)
{
    return makeIType(address, OPCODE_LUI, 0, rt, immediate);
}

static Instruction makeOri(uint32_t address, uint8_t rt, uint8_t rs, uint16_t immediate)
{
    return makeIType(address, OPCODE_ORI, rs, rt, immediate);
}

static Instruction makeAddiu(uint32_t address, uint8_t rt, uint8_t rs, uint16_t immediate)
{
    return makeIType(address, OPCODE_ADDIU, rs, rt, immediate);
}

static Instruction makeLw(uint32_t address, uint8_t rt, uint8_t rs, uint16_t immediate)
{
    return makeIType(address, OPCODE_LW, rs, rt, immediate);
}

static Instruction makeSw(uint32_t address, uint8_t rt, uint8_t rs, uint16_t immediate)
{
    return makeIType(address, OPCODE_SW, rs, rt, immediate);
}

static std::string readFileFromCandidates(const std::vector<std::string> &candidates)
{
    for (const auto &path : candidates)
    {
        std::ifstream file(path);
        if (file)
        {
            std::ostringstream ss;
            ss << file.rdbuf();
            return ss.str();
        }
    }
    return {};
}

static std::vector<uint32_t> parseEnumValues(const std::string &text, const std::string &prefix)
{
    std::vector<uint32_t> values;
    std::regex re("\\b(" + prefix + "[A-Za-z0-9_]+)\\b\\s*=\\s*0x([0-9A-Fa-f]+)");
    for (auto it = std::sregex_iterator(text.begin(), text.end(), re); it != std::sregex_iterator(); ++it)
    {
        const auto &match = *it;
        uint32_t value = static_cast<uint32_t>(std::stoul(match[2].str(), nullptr, 16));
        values.push_back(value);
    }
    return values;
}

static Instruction makeJal(uint32_t address, uint32_t target)
{
    Instruction inst{};
    inst.address = address;
    inst.opcode = OPCODE_JAL;
    inst.target = (target >> 2) & 0x3FFFFFF;
    inst.hasDelaySlot = true;
    inst.raw = (OPCODE_JAL << 26) | inst.target;
    return inst;
}

static Instruction makeJalr(uint32_t address, uint8_t rs, uint8_t rd)
{
    Instruction inst{};
    inst.address = address;
    inst.opcode = OPCODE_SPECIAL;
    inst.function = SPECIAL_JALR;
    inst.rs = rs;
    inst.rd = rd; // Destination for link address (default 31)
    inst.hasDelaySlot = true;
    inst.raw = (OPCODE_SPECIAL << 26) | (rs << 21) | (0 << 16) | (rd << 11) | (0 << 6) | SPECIAL_JALR;
    return inst;
}

static Instruction makeJr(uint32_t address, uint8_t rs)
{
    Instruction inst{};
    inst.address = address;
    inst.opcode = OPCODE_SPECIAL;
    inst.function = SPECIAL_JR;
    inst.rs = rs;
    inst.hasDelaySlot = true;
    inst.raw = (OPCODE_SPECIAL << 26) | (rs << 21) | SPECIAL_JR;
    return inst;
}

static Instruction decodeRawInstruction(uint32_t address, uint32_t raw)
{
    return R5900Decoder{}.decodeInstruction(address, raw, false);
}

static void printGeneratedCode(const std::string& name, const std::string& code)
{
#ifdef PRINT_GENERATED_CODE
    std::cout << "=== Generated Code for " << name << " ===" << std::endl;
    std::cout << code << std::endl;
    std::cout << "========================================" << std::endl;
#endif
}

void register_code_generator_tests()
{
    MiniTest::Case("CodeGenerator", [](TestCase &tc)
                   {
    tc.Run("recompiler metadata defaults are deterministic", [](TestCase &t) {
        Function function;
        Symbol symbol;
        Section section;
        Relocation relocation;
        JumpTableEntry entry;
        JumpTable jumpTable;
        CFGNode node;
        FunctionCall call;

        const Symbol copiedSymbol = symbol;
        t.Equals(function.start, 0u, "default function start should be zero");
        t.Equals(function.end, 0u, "default function end should be zero");
        t.Equals(copiedSymbol.address, 0u, "default symbol address should be zero");
        t.Equals(copiedSymbol.size, 0u, "default symbol size should be zero");
        t.IsFalse(copiedSymbol.isFunction, "default symbol should not be a function");
        t.IsFalse(copiedSymbol.isImported, "default symbol should not be imported");
        t.IsFalse(copiedSymbol.isExported, "default symbol should not be exported");
        t.Equals(section.address, 0u, "default section address should be zero");
        t.IsNull(section.data, "default section data should be null");
        t.Equals(relocation.offset, 0u, "default relocation offset should be zero");
        t.Equals(entry.target, 0u, "default jump-table target should be zero");
        t.Equals(jumpTable.address, 0u, "default jump-table address should be zero");
        t.IsFalse(node.isJumpTarget, "default CFG node should not be a jump target");
        t.IsFalse(node.hasJumpTable, "default CFG node should not have a jump table");
        t.Equals(call.callerAddress, 0u, "default function-call source should be zero");
        t.Equals(call.calleeAddress, 0u, "default function-call target should be zero");
    });

    tc.Run("R5900 MULT writes rd when rd is non-zero", [](TestCase &t) {
        CodeGenerator gen({}, {});

        Instruction mult{};
        mult.opcode = OPCODE_SPECIAL;
        mult.function = SPECIAL_MULT;
        mult.rs = 4;
        mult.rt = 5;
        mult.rd = 3;

        std::string generated = gen.translateInstruction(mult);
        printGeneratedCode("R5900 MULT writes rd when rd is non-zero", generated);

        t.IsTrue(generated.find("SET_GPR_S32(ctx, 3, (int32_t)result);") != std::string::npos,
                 "MULT should write low product to rd on R5900");

        mult.rd = 0;
        generated = gen.translateInstruction(mult);
        t.IsTrue(generated.find("SET_GPR_S32(") == std::string::npos,
                 "MULT should not write rd when rd is zero");
    });

    tc.Run("R5900 MMI MULT1 writes rd when rd is non-zero", [](TestCase &t) {
        CodeGenerator gen({}, {});

        Instruction mult1{};
        mult1.opcode = OPCODE_MMI;
        mult1.isMMI = true;
        mult1.function = MMI_MULT1;
        mult1.rs = 8;
        mult1.rt = 9;
        mult1.rd = 10;

        std::string generated = gen.translateInstruction(mult1);
        printGeneratedCode("R5900 MMI MULT1 writes rd when rd is non-zero", generated);

        t.IsTrue(generated.find("SET_GPR_S32(ctx, 10, (int32_t)result);") != std::string::npos,
                 "MULT1 should write low product to rd on R5900");
    });

    tc.Run("64-bit scalar arithmetic avoids host signed overflow", [](TestCase &t) {
        CodeGenerator gen({}, {});

        Instruction dadd{};
        dadd.opcode = OPCODE_SPECIAL;
        dadd.function = SPECIAL_DADD;
        dadd.rs = 2;
        dadd.rt = 3;
        dadd.rd = 4;
        const std::string daddCode = gen.translateInstruction(dadd);
        t.IsTrue(daddCode.find("ADD64_OV(GPR_U64(ctx, 2), GPR_U64(ctx, 3)") != std::string::npos,
                 "DADD should use defined unsigned arithmetic with overflow detection");

        Instruction dsub = dadd;
        dsub.function = SPECIAL_DSUB;
        const std::string dsubCode = gen.translateInstruction(dsub);
        t.IsTrue(dsubCode.find("SUB64_OV(GPR_U64(ctx, 2), GPR_U64(ctx, 3)") != std::string::npos,
                 "DSUB should use defined unsigned arithmetic with overflow detection");

        const Instruction daddi = makeIType(0x1000, OPCODE_DADDI, 5, 6, 0xFFFFu);
        const std::string daddiCode = gen.translateInstruction(daddi);
        t.IsTrue(daddiCode.find("ADD64_OV(GPR_U64(ctx, 5)") != std::string::npos,
                 "DADDI should use defined unsigned arithmetic with overflow detection");

        const Instruction daddiu = makeIType(0x1004, OPCODE_DADDIU, 7, 8, 0xFFFFu);
        const std::string daddiuCode = gen.translateInstruction(daddiu);
        t.IsTrue(daddiuCode.find("ADD64(GPR_U64(ctx, 7)") != std::string::npos,
                 "DADDIU should use defined wrapping unsigned arithmetic");

        t.IsTrue(daddCode.find("int64_t r =") == std::string::npos,
                 "DADD should not compute a potentially overflowing signed sum");
        t.IsTrue(dsubCode.find("int64_t r =") == std::string::npos,
                 "DSUB should not compute a potentially overflowing signed difference");
        t.IsTrue(daddiCode.find("int64_t src =") == std::string::npos,
                 "DADDI should not compute a potentially overflowing signed sum");
    });

    tc.Run("LWU zero-extends bit 31 into the low scalar lane", [](TestCase &t) {
        CodeGenerator gen({}, {});

        const Instruction lwu = makeIType(0x1000, OPCODE_LWU, 5, 6, 0x20u);
        const std::string generated = gen.translateInstruction(lwu);

        t.IsTrue(
            generated.find("SET_GPR_U64(ctx, 6, (uint64_t)(uint32_t)([&]()") != std::string::npos &&
                generated.find("PS2X_EE_READ32(Mode, eeBreakpointAddress)") != std::string::npos,
            "LWU should zero-extend the loaded word to 64 bits");
        t.IsTrue(
            generated.find("SET_GPR_U32(ctx, 6") == std::string::npos,
            "LWU must not use the sign-extending 32-bit setter");
    });

    tc.Run("VU FTOI uses saturating conversion", [](TestCase &t) {
        CodeGenerator gen({}, {});
        Instruction ftoi{};
        ftoi.rd = 17;
        ftoi.rt = 18;
        ftoi.vectorInfo.vectorField = 0xFu;

        const std::string ftoi0 = gen.translateVU_VFTOI(ftoi, 0);
        const std::string ftoi4 = gen.translateVU_VFTOI(ftoi, 4);

        t.IsTrue(ftoi0.find("Ps2VuFtoi(ctx->vu0_vf[17], 1.0f)") != std::string::npos,
                 "VFTOI0 should use the VU saturation helper");
        t.IsTrue(ftoi4.find("Ps2VuFtoi(ctx->vu0_vf[17], 16.0f)") != std::string::npos,
                 "VFTOI4 should scale through the VU saturation helper");
        t.IsTrue(ftoi0.find("_mm_cvttps_epi32") == std::string::npos,
                 "generated VFTOI must not use an unsaturated host conversion directly");
    });

    tc.Run("MMI signed multiply-accumulate uses defined wrapping arithmetic", [](TestCase &t) {
        CodeGenerator gen({}, {});

        Instruction madd{};
        madd.opcode = OPCODE_MMI;
        madd.isMMI = true;
        madd.function = MMI_MADD;
        madd.rs = 2;
        madd.rt = 3;
        madd.rd = 4;

        const std::string maddCode = gen.translateInstruction(madd);
        t.IsTrue(maddCode.find("uint64_t result = Ps2MaddSigned32(acc, GPR_S32(ctx, 2), GPR_S32(ctx, 3));") != std::string::npos,
                 "MADD should accumulate through the wrapping signed-product helper");
        t.IsTrue(maddCode.find(" int64_t result") == std::string::npos,
                 "MADD should not store an arbitrary accumulator bit pattern in a signed integer");

        madd.function = MMI_MSUB;
        const std::string msubCode = gen.translateInstruction(madd);
        t.IsTrue(msubCode.find("uint64_t result = Ps2MsubSigned32(acc, GPR_S32(ctx, 2), GPR_S32(ctx, 3));") != std::string::npos,
                 "MSUB should accumulate through the wrapping signed-product helper");
        t.IsTrue(msubCode.find(" int64_t result") == std::string::npos,
                 "MSUB should not store an arbitrary accumulator bit pattern in a signed integer");

        madd.function = MMI_MADD1;
        const std::string madd1Code = gen.translateInstruction(madd);
        t.IsTrue(madd1Code.find("Ps2MaddSigned32(acc, GPR_S32(ctx, 2), GPR_S32(ctx, 3))") != std::string::npos,
                 "MADD1 should use the same defined wrapping behavior for HI1/LO1");
    });

    tc.Run("packed HI and LO moves preserve all 128 bits", [](TestCase &t) {
        CodeGenerator gen({}, {});

        Instruction packed{};
        packed.opcode = OPCODE_MMI;
        packed.isMMI = true;
        packed.function = MMI_MMI2;
        packed.sa = MMI2_PMFHI;
        packed.rd = 4;
        t.IsTrue(gen.translateInstruction(packed).find("SET_GPR_VEC(ctx, 4, Ps2GetHi128(ctx));") != std::string::npos,
                 "PMFHI should copy both 64-bit halves of HI");

        packed.sa = MMI2_PMFLO;
        t.IsTrue(gen.translateInstruction(packed).find("SET_GPR_VEC(ctx, 4, Ps2GetLo128(ctx));") != std::string::npos,
                 "PMFLO should copy both 64-bit halves of LO");

        packed.function = MMI_MMI3;
        packed.sa = MMI3_PMTHI;
        packed.rs = 5;
        t.IsTrue(gen.translateInstruction(packed).find("Ps2SetHi128(ctx, GPR_VEC(ctx, 5));") != std::string::npos,
                 "PMTHI should replace both 64-bit halves of HI");

        packed.sa = MMI3_PMTLO;
        t.IsTrue(gen.translateInstruction(packed).find("Ps2SetLo128(ctx, GPR_VEC(ctx, 5));") != std::string::npos,
                 "PMTLO should replace both 64-bit halves of LO");

        packed.function = MMI_PMFHL;
        packed.sa = PMFHL_SH;
        packed.rd = 6;
        t.IsTrue(gen.translateInstruction(packed).find("SET_GPR_VEC(ctx, 6, Ps2PmfhlSh(ctx));") != std::string::npos,
                 "PMFHL.SH should consume all eight HI/LO words");

        packed.function = MMI_PMTHL;
        packed.sa = PMFHL_LW;
        packed.rs = 7;
        t.IsTrue(gen.translateInstruction(packed).find("Ps2PmthlLw(ctx, GPR_VEC(ctx, 7));") != std::string::npos,
                 "PMTHL.LW should update the lower word in all four HI/LO lanes");
    });

    tc.Run("PMULTH emits independent packed products", [](TestCase &t) {
        CodeGenerator gen({}, {});

        Instruction pmulth{};
        pmulth.opcode = OPCODE_MMI;
        pmulth.isMMI = true;
        pmulth.function = MMI_MMI2;
        pmulth.sa = MMI2_PMULTH;
        pmulth.rs = 2;
        pmulth.rt = 3;
        pmulth.rd = 4;

        const std::string generated = gen.translateInstruction(pmulth);
        t.IsTrue(generated.find("__m128i result = Ps2Pmulth(ctx, GPR_VEC(ctx, 2), GPR_VEC(ctx, 3));") != std::string::npos,
                 "PMULTH should use the packed HI/LO lane helper");
        t.IsTrue(generated.find("SET_GPR_VEC(ctx, 4, result);") != std::string::npos,
                 "PMULTH should return the four even products to rd");
        t.IsTrue(generated.find("_mm_madd_epi16") == std::string::npos,
                 "PMULTH must not horizontally add adjacent products");
    });

    tc.Run("PMULTUW emits two independent unsigned products", [](TestCase &t) {
        CodeGenerator gen({}, {});

        Instruction pmultuw{};
        pmultuw.opcode = OPCODE_MMI;
        pmultuw.isMMI = true;
        pmultuw.function = MMI_MMI3;
        pmultuw.sa = MMI3_PMULTUW;
        pmultuw.rs = 2;
        pmultuw.rt = 3;
        pmultuw.rd = 4;

        const std::string generated = gen.translateInstruction(pmultuw);
        t.IsTrue(generated.find("__m128i result = Ps2Pmultuw(ctx, GPR_VEC(ctx, 2), GPR_VEC(ctx, 3));") != std::string::npos,
                 "PMULTUW should use the dual-product HI/LO helper");
        t.IsTrue(generated.find("SET_GPR_VEC(ctx, 4, result);") != std::string::npos,
                 "PMULTUW should return both 64-bit products to rd");
        t.IsTrue(generated.find("_mm_mul_epu32") == std::string::npos,
                 "PMULTUW should not duplicate an even lane through overlapping SIMD products");
    });

    tc.Run("PMULTW emits two independent signed products", [](TestCase &t) {
        CodeGenerator gen({}, {});

        Instruction pmultw{};
        pmultw.opcode = OPCODE_MMI;
        pmultw.isMMI = true;
        pmultw.function = MMI_MMI2;
        pmultw.sa = MMI2_PMULTW;
        pmultw.rs = 2;
        pmultw.rt = 3;
        pmultw.rd = 4;

        const std::string generated = gen.translateInstruction(pmultw);
        t.IsTrue(generated.find("__m128i result = Ps2Pmultw(ctx, GPR_VEC(ctx, 2), GPR_VEC(ctx, 3));") != std::string::npos,
                 "PMULTW should use the dual signed-product HI/LO helper");
        t.IsTrue(generated.find("SET_GPR_VEC(ctx, 4, result);") != std::string::npos,
                 "PMULTW should return both 64-bit products to rd");
        t.IsTrue(generated.find("_mm_mul_epu32") == std::string::npos,
                 "PMULTW must not use unsigned multiplication or collapse product lanes");
    });

    tc.Run("PMADDW emits a packed signed accumulator operation", [](TestCase &t) {
        constexpr uint32_t rawPmaddw = 0x71f08809u;
        const Instruction pmaddw = R5900Decoder{}.decodeInstruction(
            0x00100000u, rawPmaddw);
        const std::string generated =
            CodeGenerator({}, {}).translateInstruction(pmaddw);

        t.Equals(pmaddw.rs, static_cast<uint8_t>(15u),
                 "PMADDW should decode rs from the literal encoding");
        t.Equals(pmaddw.rt, static_cast<uint8_t>(16u),
                 "PMADDW should decode rt from the literal encoding");
        t.Equals(pmaddw.rd, static_cast<uint8_t>(17u),
                 "PMADDW should decode rd from the literal encoding");
        t.IsTrue(
            generated.find(
                "__m128i result = Ps2Pmaddw(ctx, GPR_VEC(ctx, 15), GPR_VEC(ctx, 16));") !=
                std::string::npos,
            "PMADDW should use the packed signed-accumulator helper");
        t.IsTrue(
            generated.find("SET_GPR_VEC(ctx, 17, result);") !=
                std::string::npos,
            "PMADDW should return both packed accumulator results");
        t.IsTrue(
            generated.find("_mm_mul_epu32") == std::string::npos &&
                generated.find("Ps2HiLoToU64") == std::string::npos,
            "PMADDW must not use unsigned products or collapse packed HI/LO state");
    });

    tc.Run("constant MMIO store emits direct runtime store", [](TestCase &t) {
        Function func;
        func.name = "mmio_store";
        func.start = 0x1000;
        func.end = 0x1010;
        func.isRecompiled = true;

        std::vector<Instruction> instructions;
        instructions.push_back(makeLui(0x1000, 1, 0x1000));
        instructions.push_back(makeOri(0x1004, 1, 1, 0xE020));
        instructions.push_back(makeSw(0x1008, 2, 1, 0));

        CodeGenerator gen({}, {});
        std::string generated = gen.generateFunction(func, instructions, false);
        printGeneratedCode("constant MMIO store emits direct runtime store", generated);

        t.IsTrue(generated.find("runtime->Store32(rdram, ctx, 0x1000E020u, eeBreakpointStoreValue);") != std::string::npos,
                 "constant MMIO SW should emit a direct runtime Store32");
        t.IsTrue(generated.find("WRITE32(ADD32(GPR_U32(ctx, 1)") == std::string::npos,
                 "constant MMIO SW should not go through WRITE32 address classification");
    });

    tc.Run("constant RDRAM load and store emit fast memory access", [](TestCase &t) {
        Function func;
        func.name = "rdram_access";
        func.start = 0x2000;
        func.end = 0x2014;
        func.isRecompiled = true;

        std::vector<Instruction> instructions;
        instructions.push_back(makeLui(0x2000, 1, 0x0012));
        instructions.push_back(makeOri(0x2004, 1, 1, 0x3450));
        instructions.push_back(makeLw(0x2008, 3, 1, 0x0010));
        instructions.push_back(makeSw(0x200C, 4, 1, 0x0014));

        CodeGenerator gen({}, {});
        std::string generated = gen.generateFunction(func, instructions, false);
        printGeneratedCode("constant RDRAM load and store emit fast memory access", generated);

        t.IsTrue(generated.find("PS2X_EE_READ32(Mode, 0x123460u)") != std::string::npos,
                 "constant RDRAM LW should retain guest-mode-aware fast-path selection");
        t.IsTrue(generated.find("PS2X_EE_WRITE32(Mode, 0x123464u, eeBreakpointStoreValue);") != std::string::npos,
                 "constant RDRAM SW should retain guest-mode-aware fast-path selection");
        t.IsTrue(generated.find("PS2X_EE_READ32(Mode, ADD32(GPR_U32(ctx, 1)") == std::string::npos,
                 "constant RDRAM LW should not go through READ32 address classification");
        t.IsTrue(generated.find("PS2X_EE_WRITE32(Mode, ADD32(GPR_U32(ctx, 1)") == std::string::npos,
                 "constant RDRAM SW should not go through WRITE32 address classification");
    });

    tc.Run("constant privileged RDRAM aliases retain the guest permission gate", [](TestCase &t) {
        CodeGenerator gen({}, {});
        const Instruction load = makeLw(0x2380u, 3u, 1u, 0u);
        const Instruction store = makeSw(0x2384u, 4u, 1u, 0u);

        const std::string generatedLoad = gen.translateInstruction(
            load, MemoryAccessHint{true, 0x80001000u});
        const std::string generatedStore = gen.translateInstruction(
            store, MemoryAccessHint{true, 0xA0001000u});

        t.IsTrue(
            generatedLoad.find("PS2X_EE_READ32(Mode, 0x80001000u)") != std::string::npos,
            "a resolved KSEG0 load should retain runtime mode selection");
        t.IsTrue(
            generatedStore.find("PS2X_EE_WRITE32(Mode, 0xA0001000u, eeBreakpointStoreValue)") !=
                std::string::npos,
            "a resolved KSEG1 store should retain runtime mode selection");
        t.IsTrue(
            generatedLoad.find("DEBUG_FAST_READ32") == std::string::npos,
            "a resolved privileged load must not bypass guest permissions");
        t.IsTrue(
            generatedStore.find("DEBUG_FAST_WRITE32") == std::string::npos,
            "a resolved privileged store must not bypass guest permissions");
    });

    tc.Run("constant invalid and misaligned memory accesses use checked runtime", [](TestCase &t) {
        Function func;
        func.name = "checked_memory_access";
        func.start = 0x2400;
        func.end = 0x2420;
        func.isRecompiled = true;

        std::vector<Instruction> instructions;
        instructions.push_back(makeLui(0x2400, 1, 0x2200));
        instructions.push_back(makeLw(0x2404, 3, 1, 0));
        instructions.push_back(makeLui(0x2408, 2, 0x0012));
        instructions.push_back(makeOri(0x240C, 2, 2, 0x3450));
        instructions.push_back(makeLw(0x2410, 4, 2, 1));
        instructions.push_back(makeSw(0x2414, 5, 2, 2));

        CodeGenerator gen({}, {});
        std::string generated = gen.generateFunction(func, instructions, false);
        printGeneratedCode("constant invalid and misaligned memory accesses use checked runtime", generated);

        t.IsTrue(generated.find("runtime->Load32(rdram, ctx, 0x22000000u)") != std::string::npos,
                 "constant address beyond RDRAM aliases should use the checked load");
        t.IsTrue(generated.find("runtime->Load32(rdram, ctx, 0x123451u)") != std::string::npos,
                 "constant misaligned LW should use the checked load");
        t.IsTrue(generated.find("runtime->Store32(rdram, ctx, 0x123452u, eeBreakpointStoreValue)") != std::string::npos,
                 "constant misaligned SW should use the checked store");
        t.IsTrue(generated.find("FAST_READ32(0x22000000u)") == std::string::npos,
                 "invalid constant load must not wrap through the fast helper");
        t.IsTrue(generated.find("FAST_READ32(0x123451u)") == std::string::npos,
                 "misaligned constant load must not use the fast helper");
        t.IsTrue(generated.find("FAST_WRITE32(0x123452u") == std::string::npos,
                 "misaligned constant store must not use the fast helper");
    });

    tc.Run("EE LQ and SQ silently align addresses", [](TestCase &t) {
        CodeGenerator gen({}, {});
        const Instruction lq = makeIType(0x2800, OPCODE_LQ, 1, 2, 8);
        const Instruction sq = makeIType(0x2804, OPCODE_SQ, 3, 4, 8);

        const std::string dynamicLq = gen.translateInstruction(lq);
        const std::string dynamicSq = gen.translateInstruction(sq);
        printGeneratedCode("EE LQ silently aligns dynamic address", dynamicLq);
        printGeneratedCode("EE SQ silently aligns dynamic address", dynamicSq);

        t.IsTrue(dynamicLq.find("PS2X_EE_READ128(Mode, (eeBreakpointAddress & ~0xFu))") != std::string::npos,
                 "LQ should clear the low four effective-address bits");
        t.IsTrue(dynamicSq.find("PS2X_EE_WRITE128(Mode, (eeBreakpointAddress & ~0xFu), eeBreakpointStoreValue)") != std::string::npos,
                 "SQ should clear the low four effective-address bits");

        const MemoryAccessHint unalignedHint{true, 0x00123458u};
        const std::string constantLq = gen.translateInstruction(lq, unalignedHint);
        const std::string constantSq = gen.translateInstruction(sq, unalignedHint);

        t.IsTrue(constantLq.find("PS2X_EE_READ128(Mode, 0x123450u)") != std::string::npos,
                 "constant LQ hint should be aligned before selecting the fast path");
        t.IsTrue(constantSq.find("PS2X_EE_WRITE128(Mode, 0x123450u, eeBreakpointStoreValue)") != std::string::npos,
                 "constant SQ hint should be aligned before selecting the fast path");
        t.IsTrue(constantLq.find("runtime->Load128") == std::string::npos,
                 "silently aligned LQ should not be treated as an address error");
        t.IsTrue(constantSq.find("runtime->Store128") == std::string::npos,
                 "silently aligned SQ should not be treated as an address error");
    });

    tc.Run("EE partial stores preserve byte enables without memory loads", [](TestCase &t) {
        CodeGenerator gen({}, {});
        const Instruction swl = makeIType(0x2900, OPCODE_SWL, 1, 2, 3);
        const Instruction swr = makeIType(0x2904, OPCODE_SWR, 3, 4, 5);
        const Instruction sdl = makeIType(0x2908, OPCODE_SDL, 5, 6, 7);
        const Instruction sdr = makeIType(0x290C, OPCODE_SDR, 7, 8, 9);

        const std::string swlCode = gen.translateInstruction(swl);
        const std::string swrCode = gen.translateInstruction(swr);
        const std::string sdlCode = gen.translateInstruction(sdl);
        const std::string sdrCode = gen.translateInstruction(sdr);

        t.IsTrue(swlCode.find("PS2X_EE_WRITE_MASKED32(Mode, aligned_addr") != std::string::npos,
                 "SWL should issue one masked 32-bit store");
        t.IsTrue(swrCode.find("PS2X_EE_WRITE_MASKED32(Mode, aligned_addr") != std::string::npos,
                 "SWR should issue one masked 32-bit store");
        t.IsTrue(sdlCode.find("PS2X_EE_WRITE_MASKED64(Mode, aligned_addr") != std::string::npos,
                 "SDL should issue one masked 64-bit store");
        t.IsTrue(sdrCode.find("PS2X_EE_WRITE_MASKED64(Mode, aligned_addr") != std::string::npos,
                 "SDR should issue one masked 64-bit store");
        t.IsTrue(
            swlCode.find("byte_enable = (uint8_t)((1u << (offset + 1u)) - 1u)") !=
                    std::string::npos &&
                swlCode.find("GPR_U32(ctx, 2) >> shift") != std::string::npos,
            "SWL should enable low lanes through the effective byte and shift the source right");
        t.IsTrue(
            swrCode.find("byte_enable = (uint8_t)((0xFu << offset) & 0xFu)") !=
                    std::string::npos &&
                swrCode.find("GPR_U32(ctx, 4) << shift") != std::string::npos,
            "SWR should enable lanes from the effective byte upward and shift the source left");
        t.IsTrue(
            sdlCode.find("byte_enable = (uint8_t)((1u << (offset + 1u)) - 1u)") !=
                    std::string::npos &&
                sdlCode.find("GPR_U64(ctx, 6) >> shift") != std::string::npos,
            "SDL should enable low lanes through the effective byte and shift the source right");
        t.IsTrue(
            sdrCode.find("byte_enable = (uint8_t)((0xFFu << offset) & 0xFFu)") !=
                    std::string::npos &&
                sdrCode.find("GPR_U64(ctx, 8) << shift") != std::string::npos,
            "SDR should enable lanes from the effective byte upward and shift the source left");
        t.IsTrue(swlCode.find("READ32(") == std::string::npos &&
                     swrCode.find("READ32(") == std::string::npos,
                 "SWL and SWR must not issue an architectural memory read");
        t.IsTrue(sdlCode.find("READ64(") == std::string::npos &&
                     sdrCode.find("READ64(") == std::string::npos,
                 "SDL and SDR must not issue an architectural memory read");
    });

    tc.Run("known GIF DMA MMIO sequence emits native kick helper", [](TestCase &t) {
        Function func;
        func.name = "gif_dma_kick";
        func.start = 0x3000;
        func.end = 0x3030;
        func.isRecompiled = true;

        std::vector<Instruction> instructions;
        instructions.push_back(makeAddiu(0x3000, 2, 0, 4));
        instructions.push_back(makeLui(0x3004, 1, 0x1000));
        instructions.push_back(makeOri(0x3008, 1, 1, 0xE020));
        instructions.push_back(makeSw(0x300C, 2, 1, 0));
        instructions.push_back(makeLui(0x3010, 1, 0x1000));
        instructions.push_back(makeOri(0x3014, 1, 1, 0xE010));
        instructions.push_back(makeSw(0x3018, 2, 1, 0));
        instructions.push_back(makeLui(0x301C, 1, 0x1000));
        instructions.push_back(makeOri(0x3020, 1, 1, 0xA030));
        instructions.push_back(makeSw(0x3024, 4, 1, 0));
        instructions.push_back(makeAddiu(0x3028, 5, 0, 0x0105));
        instructions.push_back(makeAddiu(0x302C, 1, 1, 0xFFD0));
        instructions.push_back(makeSw(0x3030, 5, 1, 0));

        CodeGenerator gen({}, {});
        std::string generated = gen.generateFunction(func, instructions, false);
        printGeneratedCode("known GIF DMA MMIO sequence emits native kick helper", generated);

        t.IsTrue(generated.find("uint32_t gifDmaKickValue_3024_2 = GPR_U32(ctx, 4);") != std::string::npos,
                 "dynamic GIF TADR source should be captured when the store is coalesced");
        t.IsTrue(generated.find("runtime->kickGifDmaChainFromMMIO(rdram, ctx, 0x4u, 0x4u, gifDmaKickValue_3024_2, 0x105u);") != std::string::npos,
                 "known GIF DMA MMIO stores should coalesce into the native kick helper");
        t.IsTrue(generated.find("runtime->eeDataBreakpointEnabledPolicy<Mode>(ctx, true)") != std::string::npos,
                 "coalescing should retain a guarded architectural-breakpoint fallback");
        t.IsTrue(generated.find("runtime->Store32(rdram, ctx, 0x1000E020u") != std::string::npos,
                 "the breakpoint fallback should preserve the individual D_PCR store");
        t.IsTrue(generated.find("runtime->Store32(rdram, ctx, 0x1000A000u") != std::string::npos,
                 "the breakpoint fallback should preserve the individual GIF CHCR store");
    });

    tc.Run("GIF DMA kick coalesces when CHCR store is a return delay slot", [](TestCase &t) {
        Function func;
        func.name = "loadImage_like";
        func.start = 0x2E7C90;
        func.end = 0x2E7CC8;
        func.isRecompiled = true;

        std::vector<Instruction> instructions;
        instructions.push_back(makeBranch(0x2E7C90, 2));
        instructions.push_back(makeLui(0x2E7C94, 5, 0x1000));
        instructions.push_back(makeLui(0x2E7C98, 5, 0x1000));
        instructions.push_back(makeAddiu(0x2E7C9C, 6, 0, 4));
        instructions.push_back(makeOri(0x2E7CA0, 3, 5, 0xE020));
        instructions.push_back(makeSw(0x2E7CA4, 6, 3, 0));
        instructions.push_back(makeOri(0x2E7CA8, 3, 5, 0xE010));
        instructions.push_back(makeSw(0x2E7CAC, 6, 3, 0));
        instructions.push_back(makeOri(0x2E7CB0, 3, 5, 0xA030));
        instructions.push_back(makeSw(0x2E7CB4, 4, 3, 0));
        instructions.push_back(makeAddiu(0x2E7CB8, 4, 0, 0x0105));
        instructions.push_back(makeOri(0x2E7CBC, 3, 5, 0xA000));
        instructions.push_back(makeJr(0x2E7CC0, 31));
        instructions.push_back(makeSw(0x2E7CC4, 4, 3, 0));

        CodeGenerator gen({}, {});
        std::string generated = gen.generateFunction(func, instructions, false);
        printGeneratedCode("GIF DMA kick coalesces when CHCR store is a return delay slot", generated);

        t.IsTrue(generated.find("label_2e7c9c:") != std::string::npos,
                 "test should cover a branch target inside the GIF DMA setup");
        t.IsTrue(generated.find("uint32_t gifDmaKickValue_2e7cb4_2 = GPR_U32(ctx, 4);") != std::string::npos,
                 "TADR value should be captured before a0 is reused for CHCR");
        t.IsTrue(generated.find("ctx->in_delay_slot = true;") != std::string::npos,
                 "coalesced helper should still run as the return delay slot");
        t.IsTrue(generated.find("runtime->kickGifDmaChainFromMMIO(rdram, ctx, 0x4u, 0x4u, gifDmaKickValue_2e7cb4_2, 0x105u);") != std::string::npos,
                 "loadImage-like GIF DMA stores should coalesce into the native kick helper");
        t.IsTrue(generated.find("runtime->eeDataBreakpointEnabledPolicy<Mode>(ctx, true)") != std::string::npos &&
                     generated.find("observeEeDataAddressPolicy<Mode>") != std::string::npos,
                 "the coalesced delay-slot store should retain a guarded breakpoint fallback");
    });

        tc.Run("emits labels and gotos for internal branches", [](TestCase &t) {
            Function func;
            func.name = "test_func";
            func.start = 0x1000;
        func.end = 0x1020;
        func.isRecompiled = true;
        func.isStub = false;

        // Build a small function:
        // 0x1000: nop
        // 0x1004: beq $1,$1, target (0x100c) with delay slot at 0x1008
        // 0x1008: nop (delay slot)
        // 0x100c: nop (branch target)
        // 0x1010: nop (fallthrough)
        std::vector<Instruction> instructions;
        instructions.push_back(makeNop(0x1000));
        instructions.push_back(makeBranch(0x1004, 1)); // target = 0x1004 + 4 + (1<<2) = 0x100c
        instructions.push_back(makeNop(0x1008));       // delay slot
        instructions.push_back(makeNop(0x100c));       // branch target
        instructions.push_back(makeNop(0x1010));       // extra

        CodeGenerator gen({}, {});
        std::string generated = gen.generateFunction(func, instructions, false);
        printGeneratedCode("emits labels and gotos for internal branches", generated);

        t.IsTrue(generated.find("label_100c:") != std::string::npos, "branch target should emit a label");
        t.IsTrue(generated.find("goto label_100c;") != std::string::npos, "internal branch should jump via goto");
        t.IsTrue(generated.find("label_1008:") == std::string::npos, "delay slot without incoming branch should not get a label");
    });

    tc.Run("labels delay slot when it is a branch target", [](TestCase &t) {
        Function func;
        func.name = "delay_slot_label";
        func.start = 0x2000;
        func.end = 0x2020;
        func.isRecompiled = true;
        func.isStub = false;

        // Branch at 0x2000 targets 0x2004 (its own delay slot)
        std::vector<Instruction> instructions;
        instructions.push_back(makeBranch(0x2000, 0)); // target = 0x2004
        instructions.push_back(makeNop(0x2004));       // delay slot and target
        instructions.push_back(makeNop(0x2008));       // extra

        CodeGenerator gen({}, {});
        std::string generated = gen.generateFunction(func, instructions, false);
        printGeneratedCode("labels delay slot when it is a branch target", generated);

        t.IsTrue(generated.find("label_2004:") != std::string::npos, "delay slot that is a target should emit a label");
        t.IsTrue(generated.find("goto label_2004;") != std::string::npos, "branch to delay slot should use goto");
    });

    tc.Run("direct entry at a delay-slot address is not branch-delay execution", [](TestCase &t) {
        Function func;
        func.name = "direct_delay_slot_entry";
        func.start = 0x3000u;
        func.end = 0x3018u;
        func.isRecompiled = true;

        // The first branch enters 0x300c directly. The second branch owns the
        // same instruction as its delay slot, so codegen must emit distinct
        // ordinary-entry and branch-delay paths for the trapping load.
        const std::vector<Instruction> instructions = {
            makeBranch(0x3000u, 2u),
            makeNop(0x3004u),
            makeBranch(0x3008u, 2u),
            makeLw(0x300Cu, 2u, 0u, 1u),
            makeNop(0x3010u),
            makeNop(0x3014u),
        };

        CodeGenerator gen({}, {});
        const std::string generated =
            gen.generateFunction(func, instructions, false);

        const size_t directStart = generated.find(
            "if (ctx->pc == 0x300Cu) {");
        const size_t owningBranchStart = generated.find(
            "ctx->pc = 0x3008u;", directStart);
        t.IsTrue(
            directStart != std::string::npos &&
                owningBranchStart != std::string::npos,
            "fixture should contain separate direct and owning-branch paths");
        if (directStart == std::string::npos ||
            owningBranchStart == std::string::npos)
        {
            return;
        }

        const std::string directPath = generated.substr(
            directStart, owningBranchStart - directStart);
        const size_t directState = directPath.find(
            "ctx->in_delay_slot = false;");
        const size_t directTrap = directPath.find("READ32(");
        t.IsTrue(
            directState != std::string::npos &&
                directTrap != std::string::npos &&
                directState < directTrap,
            "direct entry should establish ordinary state before the trapping load");
        t.IsFalse(
            directPath.find("ctx->in_delay_slot = true;") !=
                std::string::npos,
            "direct entry must leave CAUSE.BD clear on an exception");
        t.IsFalse(
            directPath.find("ctx->branch_pc = 0x3008u;") !=
                std::string::npos,
            "direct entry must preserve the target instruction as EPC");

        const std::string owningBranchPath = generated.substr(
            owningBranchStart);
        const size_t owningState = owningBranchPath.find(
            "ctx->in_delay_slot = true;");
        const size_t owningBranchPc = owningBranchPath.find(
            "ctx->branch_pc = 0x3008u;");
        const size_t owningTrap = owningBranchPath.find("READ32(");
        t.IsTrue(
            owningState != std::string::npos &&
                owningBranchPc != std::string::npos &&
                owningTrap != std::string::npos &&
                owningState < owningTrap &&
                owningBranchPc < owningTrap,
            "the owning branch should establish delay state before the same trapping load");
        t.IsTrue(
            owningState != std::string::npos,
            "genuine delay-slot execution must retain CAUSE.BD behavior");
        t.IsTrue(
            owningBranchPc != std::string::npos,
            "genuine delay-slot execution must retain the branch EPC");
    });

    tc.Run("control-flow analysis keeps same-function JAL target internal and promotes only the return pc", [](TestCase &t) {
        Function func;
        func.name = "same_function_call";
        func.start = 0x1000;
        func.end = 0x101C;
        func.isRecompiled = true;
        func.isStub = false;

        std::vector<Instruction> instructions;
        instructions.push_back(makeJal(0x1000, 0x100C));
        instructions.push_back(makeNop(0x1004));
        instructions.push_back(makeNop(0x1008));
        instructions.push_back(makeNop(0x100C));
        instructions.push_back(makeNop(0x1010));
        instructions.push_back(makeJr(0x1014, 31));
        instructions.push_back(makeNop(0x1018));

        std::vector<Function> functions{func};
        std::vector<Section> sections = {
            {".text", 0x1000u, 0x100u, 0u, true, false, false, true, nullptr}
        };

        CodeGenerator gen({}, sections);
        CodeGenerator::AnalysisResult analysis =
            gen.collectInternalBranchTargets(func, instructions, &functions);

        t.IsTrue(analysis.entryPoints.contains(0x100Cu),
                 "same-function JAL target should stay as an internal label target");
        t.IsTrue(analysis.resumeEntryPoints.contains(0x1008u),
                 "same-function JAL should mark the fallthrough pc as resumable");
        t.IsFalse(analysis.externalEntryPoints.contains(0x100Cu),
                  "same-function JAL target should not become an external entry candidate");
    });

    tc.Run("control-flow analysis promotes REGIMM link branch fallthroughs as resumable entries", [](TestCase &t) {
        Function func;
        func.name = "regimm_link_resume";
        func.start = 0x1400;
        func.end = 0x1418;
        func.isRecompiled = true;
        func.isStub = false;

        const std::vector<uint32_t> linkTypes = {
            REGIMM_BLTZAL,
            REGIMM_BGEZAL,
            REGIMM_BLTZALL,
            REGIMM_BGEZALL,
        };

        for (uint32_t linkType : linkTypes)
        {
            Instruction branch{};
            branch.address = 0x1400;
            branch.opcode = OPCODE_REGIMM;
            branch.rs = 8;
            branch.rt = linkType;
            branch.simmediate = 3u; // target = 0x1410
            branch.isBranch = true;
            branch.isCall = true;
            branch.hasDelaySlot = true;
            branch.raw = (OPCODE_REGIMM << 26) | (8u << 21) | (linkType << 16) | 3u;

            std::vector<Instruction> instructions{
                branch,
                makeNop(0x1404),
                makeNop(0x1408),
                makeNop(0x140C),
                makeNop(0x1410),
                makeNop(0x1414),
            };

            CodeGenerator gen({}, {});
            CodeGenerator::AnalysisResult analysis =
                gen.collectInternalBranchTargets(func, instructions);

            t.IsTrue(analysis.resumeEntryPoints.contains(0x1408u),
                     "REGIMM branch-and-link should mark PC + 8 as resumable");
            t.IsTrue(analysis.entryPoints.contains(0x1408u),
                     "REGIMM branch-and-link return pc should emit an internal label");

            std::string generated = gen.generateFunction(func, instructions, false);
            t.IsTrue(generated.find("case 0x1408u: goto label_1408;") != std::string::npos,
                     "REGIMM branch-and-link return pc should be re-enterable through the owner wrapper");
        }

        Instruction nonLink{};
        nonLink.address = 0x1400;
        nonLink.opcode = OPCODE_REGIMM;
        nonLink.rs = 8;
        nonLink.rt = REGIMM_BGEZ;
        nonLink.simmediate = 3u;
        nonLink.isBranch = true;
        nonLink.hasDelaySlot = true;

        CodeGenerator gen({}, {});
        CodeGenerator::AnalysisResult nonLinkAnalysis = gen.collectInternalBranchTargets(
            func,
            {nonLink, makeNop(0x1404), makeNop(0x1408), makeNop(0x140C),
             makeNop(0x1410), makeNop(0x1414)});
        t.IsFalse(nonLinkAnalysis.resumeEntryPoints.contains(0x1408u),
                  "ordinary REGIMM branches must not create link return entries");
    });

    tc.Run("signed EE branches test the low 64-bit doubleword", [](TestCase &t) {
        struct BranchCase
        {
            uint32_t opcode;
            uint32_t regimmType;
            const char *condition;
        };

        const std::vector<BranchCase> cases = {
            {OPCODE_BLEZ, 0u, "GPR_S64(ctx, 8) <= 0"},
            {OPCODE_BGTZ, 0u, "GPR_S64(ctx, 8) > 0"},
            {OPCODE_BLEZL, 0u, "GPR_S64(ctx, 8) <= 0"},
            {OPCODE_BGTZL, 0u, "GPR_S64(ctx, 8) > 0"},
            {OPCODE_REGIMM, REGIMM_BLTZ, "GPR_S64(ctx, 8) < 0"},
            {OPCODE_REGIMM, REGIMM_BGEZ, "GPR_S64(ctx, 8) >= 0"},
            {OPCODE_REGIMM, REGIMM_BLTZL, "GPR_S64(ctx, 8) < 0"},
            {OPCODE_REGIMM, REGIMM_BGEZL, "GPR_S64(ctx, 8) >= 0"},
            {OPCODE_REGIMM, REGIMM_BLTZAL, "GPR_S64(ctx, 8) < 0"},
            {OPCODE_REGIMM, REGIMM_BGEZAL, "GPR_S64(ctx, 8) >= 0"},
            {OPCODE_REGIMM, REGIMM_BLTZALL, "GPR_S64(ctx, 8) < 0"},
            {OPCODE_REGIMM, REGIMM_BGEZALL, "GPR_S64(ctx, 8) >= 0"},
        };

        for (size_t i = 0; i < cases.size(); ++i)
        {
            const uint32_t start = 0x1800u + static_cast<uint32_t>(i) * 0x10u;

            Function func;
            func.name = "signed_branch_width";
            func.start = start;
            func.end = start + 0xCu;
            func.isRecompiled = true;
            func.isStub = false;

            Instruction branch{};
            branch.address = start;
            branch.opcode = cases[i].opcode;
            branch.rs = 8;
            branch.rt = cases[i].regimmType;
            branch.simmediate = 1u;
            branch.isBranch = true;
            branch.isCall = cases[i].regimmType == REGIMM_BLTZAL ||
                            cases[i].regimmType == REGIMM_BGEZAL ||
                            cases[i].regimmType == REGIMM_BLTZALL ||
                            cases[i].regimmType == REGIMM_BGEZALL;
            branch.hasDelaySlot = true;

            CodeGenerator gen({}, {});
            const std::string generated = gen.generateFunction(
                func,
                {branch, makeNop(start + 4u), makeNop(start + 8u)},
                false);

            t.IsTrue(generated.find(cases[i].condition) != std::string::npos,
                     "signed branch should compare the complete EE low doubleword");
            t.IsFalse(generated.find("GPR_S32(ctx, 8)") != std::string::npos,
                      "signed branch must not discard bits 63:32");
        }
    });

    tc.Run("control-flow analysis reports cross-function mid-function jumps as external entry candidates", [](TestCase &t) {
        Function caller;
        caller.name = "caller";
        caller.start = 0x4000;
        caller.end = 0x4008;
        caller.isRecompiled = true;
        caller.isStub = false;

        Function target;
        target.name = "target";
        target.start = 0x5000;
        target.end = 0x5010;
        target.isRecompiled = true;
        target.isStub = false;

        Instruction j{};
        j.address = 0x4000;
        j.opcode = OPCODE_J;
        j.target = (0x5004u >> 2) & 0x3FFFFFFu;
        j.hasDelaySlot = true;
        j.raw = (OPCODE_J << 26) | (j.target & 0x3FFFFFFu);

        std::vector<Instruction> instructions{j, makeNop(0x4004)};
        std::vector<Function> functions{caller, target};
        std::vector<Section> sections = {
            {".text", 0x4000u, 0x2000u, 0u, true, false, false, true, nullptr}
        };

        CodeGenerator gen({}, sections);
        CodeGenerator::AnalysisResult analysis =
            gen.collectInternalBranchTargets(caller, instructions, &functions);

        t.IsTrue(analysis.externalEntryPoints.contains(0x5004u),
                 "cross-function jump into the middle of a function should become an external entry candidate");
    });

    tc.Run("control-flow analysis promotes JALR fallthrough as a resumable entry", [](TestCase &t) {
        Function func;
        func.name = "jalr_resume";
        func.start = 0x1200;
        func.end = 0x1218;
        func.isRecompiled = true;
        func.isStub = false;

        std::vector<Instruction> instructions;
        instructions.push_back(makeNop(0x1200));
        instructions.push_back(makeJalr(0x1204, 2, 31));
        instructions.push_back(makeNop(0x1208));
        instructions.push_back(makeNop(0x120C));
        instructions.push_back(makeJr(0x1210, 31));
        instructions.push_back(makeNop(0x1214));

        std::vector<Function> functions{func};
        std::vector<Section> sections = {
            {".text", 0x1200u, 0x100u, 0u, true, false, false, true, nullptr}
        };

        CodeGenerator gen({}, sections);
        CodeGenerator::AnalysisResult analysis =
            gen.collectInternalBranchTargets(func, instructions, &functions);

        t.IsTrue(analysis.resumeEntryPoints.contains(0x120Cu),
                 "JALR should mark its return/fallthrough pc as resumable");
        t.IsTrue(analysis.entryPoints.contains(0x120Cu),
                 "JALR resume pc should also be emitted as an internal label");
    });

    tc.Run("unresolved JR marks internal labels as indirect fallback resume entries", [](TestCase &t) {
        Function func;
        func.name = "unresolved_jr_fallback";
        func.start = 0x3100;
        func.end = 0x3120;
        func.isRecompiled = true;
        func.isStub = false;

        std::vector<Instruction> instructions{
            makeNop(0x3100),
            makeJr(0x3104, 8),
            makeNop(0x3108),
            makeNop(0x310C),
            makeNop(0x3110),
        };

        CodeGenerator gen({}, {});
        CodeGenerator::AnalysisResult analysis = gen.collectInternalBranchTargets(func, instructions);

        t.IsTrue(analysis.indirectFallbackEntryPoints.contains(0x310Cu),
                 "unresolved JR should register internal labels as resumable entries for the owning function");
        t.IsTrue(analysis.entryPoints.contains(0x310Cu),
                 "unresolved JR fallback targets should still emit labels in the owner");
        t.IsFalse(analysis.jumpTableTargets.contains(0x3104u),
                  "unresolved JR should not pretend it has a resolved local jump table");
    });

    tc.Run("JR through an entry-block return-address alias is classified as a return", [](TestCase &t) {
        struct AliasCase
        {
            uint32_t start;
            uint8_t aliasReg;
            uint32_t copyRaw;
            uint32_t leadingNops;
        };

        const AliasCase cases[] = {
            {
                0x3140u,
                15u,
                (OPCODE_ADDI << 26) | (31u << 21) | (15u << 16),
                0u,
            },
            {
                0x3180u,
                25u,
                (31u << 21) | (25u << 11) | SPECIAL_DADDU,
                1u,
            },
            {
                0x31A0u,
                14u,
                (OPCODE_ADDIU << 26) | (31u << 21) | (14u << 16),
                12u,
            },
        };

        for (const AliasCase &aliasCase : cases)
        {
            Function func;
            func.name = "saved_ra_return";
            func.start = aliasCase.start;
            const uint32_t copyAddress = aliasCase.start +
                aliasCase.leadingNops * sizeof(uint32_t);
            const uint32_t jrAddress = copyAddress + 8u;
            func.end = jrAddress + 8u;
            func.isRecompiled = true;
            func.isStub = false;

            std::vector<Instruction> instructions;
            for (uint32_t i = 0u; i < aliasCase.leadingNops; ++i)
            {
                instructions.push_back(makeNop(
                    aliasCase.start + i * sizeof(uint32_t)));
            }
            instructions.push_back(decodeRawInstruction(
                copyAddress, aliasCase.copyRaw));
            instructions.push_back(makeNop(copyAddress + 4u));
            instructions.push_back(makeJr(jrAddress, aliasCase.aliasReg));
            instructions.push_back(makeNop(jrAddress + 4u));

            CodeGenerator gen({}, {});
            const CodeGenerator::AnalysisResult analysis =
                gen.collectInternalBranchTargets(func, instructions);

            t.IsTrue(analysis.returnAddressAliasJumps.contains(jrAddress),
                     "an exact entry-block copy of $ra should classify its JR as a return");
            t.IsTrue(analysis.indirectFallbackEntryPoints.empty(),
                     "a proven saved-$ra return should not promote broad fallback entries");
        }
    });

    tc.Run("copying $ra after a call does not prove an incoming return alias", [](TestCase &t) {
        Function func;
        func.name = "post_call_ra_copy";
        func.start = 0x3200u;
        func.end = 0x3214u;
        func.isRecompiled = true;
        func.isStub = false;

        const Instruction saveChangedRa = decodeRawInstruction(
            0x3208u,
            (OPCODE_ADDIU << 26) | (31u << 21) | (14u << 16));
        const std::vector<Instruction> instructions{
            makeJal(0x3200u, 0x4000u),
            makeNop(0x3204u),
            saveChangedRa,
            makeJr(0x320Cu, 14u),
            makeNop(0x3210u),
        };

        CodeGenerator gen({}, {});
        const CodeGenerator::AnalysisResult analysis =
            gen.collectInternalBranchTargets(func, instructions);

        t.IsFalse(analysis.returnAddressAliasJumps.contains(0x320Cu),
                  "$ra copied after a call is not the incoming return address");
        t.IsTrue(analysis.indirectFallbackEntryPoints.contains(0x3200u),
                 "a post-call $ra copy should retain conservative fallback coverage");
    });

    tc.Run("copying an overwritten $ra does not prove an incoming return alias", [](TestCase &t) {
        Function func;
        func.name = "overwritten_ra_copy";
        func.start = 0x3240u;
        func.end = 0x3250u;
        func.isRecompiled = true;
        func.isStub = false;

        const Instruction overwriteRa = decodeRawInstruction(
            0x3240u,
            (OPCODE_ADDIU << 26) | (31u << 16) | 0x1234u);
        const Instruction saveChangedRa = decodeRawInstruction(
            0x3244u,
            (OPCODE_ADDIU << 26) | (31u << 21) | (14u << 16));
        const std::vector<Instruction> instructions{
            overwriteRa,
            saveChangedRa,
            makeJr(0x3248u, 14u),
            makeNop(0x324Cu),
        };

        CodeGenerator gen({}, {});
        const CodeGenerator::AnalysisResult analysis =
            gen.collectInternalBranchTargets(func, instructions);

        t.IsFalse(analysis.returnAddressAliasJumps.contains(0x3248u),
                  "a copied locally-written $ra is not the incoming return address");
        t.IsTrue(analysis.indirectFallbackEntryPoints.contains(0x3240u),
                 "an overwritten-$ra copy should retain conservative fallback coverage");
    });

    tc.Run("overwriting a saved return-address alias keeps JR conservative", [](TestCase &t) {
        Function func;
        func.name = "clobbered_saved_ra";
        func.start = 0x31C0u;
        func.end = 0x31D0u;
        func.isRecompiled = true;
        func.isStub = false;

        const Instruction saveRa = decodeRawInstruction(
            0x31C0u,
            (OPCODE_ADDIU << 26) | (31u << 21) | (15u << 16));
        const Instruction overwriteAlias = decodeRawInstruction(
            0x31C4u,
            (OPCODE_ADDIU << 26) | (15u << 16) | 1u);
        const std::vector<Instruction> instructions{
            saveRa,
            overwriteAlias,
            makeJr(0x31C8u, 15u),
            makeNop(0x31CCu),
        };

        CodeGenerator gen({}, {});
        const CodeGenerator::AnalysisResult analysis =
            gen.collectInternalBranchTargets(func, instructions);

        t.IsFalse(analysis.returnAddressAliasJumps.contains(0x31C8u),
                  "a locally overwritten alias must not be classified as the incoming return address");
        t.IsTrue(analysis.indirectFallbackEntryPoints.contains(0x31C0u),
                 "a clobbered alias should retain conservative indirect fallback coverage");
    });

    tc.Run("unresolved JALR marks internal labels as indirect fallback resume entries", [](TestCase &t) {
        Function func;
        func.name = "unresolved_jalr_fallback";
        func.start = 0x3200;
        func.end = 0x3220;
        func.isRecompiled = true;
        func.isStub = false;

        std::vector<Instruction> instructions{
            makeNop(0x3200),
            makeJalr(0x3204, 25, 31),
            makeNop(0x3208),
            makeNop(0x320C),
            makeNop(0x3210),
        };

        CodeGenerator gen({}, {});
        CodeGenerator::AnalysisResult analysis = gen.collectInternalBranchTargets(func, instructions);

        t.IsTrue(analysis.indirectFallbackEntryPoints.contains(0x320Cu),
                 "unresolved JALR should register internal labels as resumable entries for the owning function");
        t.IsTrue(analysis.entryPoints.contains(0x320Cu),
                 "unresolved JALR fallback targets should still emit labels in the owner");
        t.IsFalse(analysis.jumpTableTargets.contains(0x3204u),
                  "unresolved JALR should not pretend it has a resolved local jump table");
    });

    tc.Run("control-flow diagnostics report once and list only unresolved indirect branches", [](TestCase &t) {
        Function func;
        func.name = "mixed_indirect_branches";
        func.start = 0x3400;
        func.end = 0x3460;
        func.isRecompiled = true;
        func.isStub = false;

        constexpr uint32_t tableAddress = 0x00200000u;

        Instruction sll{};
        sll.address = 0x3408;
        sll.opcode = OPCODE_SPECIAL;
        sll.function = SPECIAL_SLL;
        sll.rd = 8;
        sll.rt = 4;
        sll.sa = 2;

        Instruction addu{};
        addu.address = 0x340C;
        addu.opcode = OPCODE_SPECIAL;
        addu.function = SPECIAL_ADDU;
        addu.rs = 9;
        addu.rt = 8;
        addu.rd = 9;

        JumpTable configured{};
        configured.address = tableAddress;
        configured.entries.push_back({0u, 0x3440u});
        configured.entries.push_back({1u, 0x3450u});

        const std::vector<Instruction> instructions{
            makeLui(0x3400, 9, 0x0020),
            makeAddiu(0x3404, 9, 9, 0),
            sll,
            addu,
            makeLw(0x3410, 10, 9, 0),
            makeJr(0x3414, 10),
            makeNop(0x3418),
            makeJalr(0x3420, 25, 31),
            makeNop(0x3424),
            makeNop(0x3440),
            makeNop(0x3450),
        };

        RecompilerReporter reporter;
        CodeGenerator gen({}, {});
        gen.setConfiguredJumpTables({configured});
        gen.setReporter(&reporter);

        CodeGenerator::AnalysisResult quietAnalysis =
            gen.collectInternalBranchTargets(func, instructions);
        t.IsTrue(quietAnalysis.jumpTableTargets.contains(0x3414u),
                 "the configured JR should be resolved independently of diagnostics");
        t.Equals(reporter.counters().indirectFallbackPromotions,
                 static_cast<size_t>(0),
                 "ordinary analysis passes must not record duplicate fallback diagnostics");

        CodeGenerator::AnalysisResult reportedAnalysis =
            gen.collectInternalBranchTargets(func, instructions, nullptr, true);
        t.IsTrue(reportedAnalysis.indirectFallbackEntryPoints.contains(0x3440u),
                 "the unresolved JALR should still request conservative fallback entries");
        t.Equals(reporter.counters().indirectFallbackPromotions,
                 static_cast<size_t>(1),
                 "the designated diagnostic pass should record one fallback promotion");

        std::ostringstream summary;
        reporter.printSummary(summary);
        const std::string report = summary.str();
        t.IsTrue(report.find("unresolved JR/JALR at 0x3420;") != std::string::npos,
                 "the warning should name the unresolved JALR");
        t.IsFalse(report.find("unresolved JR/JALR at 0x3414") != std::string::npos,
                  "the warning must not include the resolved jump-table JR");
        t.Equals(countOccurrences(report, "unresolved JR/JALR at"),
                 static_cast<size_t>(1),
                 "the reporter should contain one control-flow fallback event");
    });

    tc.Run("jump-table inference reuses a bounded scaled index until it is overwritten", [](TestCase &t) {
        Function func;
        func.name = "reused_jump_table_index";
        func.start = 0x4000u;
        func.end = 0x4180u;
        func.isRecompiled = true;
        func.isStub = false;

        constexpr uint32_t tableAddress = 0x00200000u;
        std::vector<uint8_t> tableData(24u, 0u);
        const auto writeTableTarget = [&](size_t offset, uint32_t target)
        {
            tableData[offset] = static_cast<uint8_t>(target);
            tableData[offset + 1u] = static_cast<uint8_t>(target >> 8u);
            tableData[offset + 2u] = static_cast<uint8_t>(target >> 16u);
            tableData[offset + 3u] = static_cast<uint8_t>(target >> 24u);
        };
        writeTableTarget(0u, 0x4024u);
        writeTableTarget(4u, 0x4028u);
        writeTableTarget(16u, 0x409Cu);
        writeTableTarget(20u, 0x40A0u);

        Section tableSection{};
        tableSection.name = ".rodata";
        tableSection.address = tableAddress;
        tableSection.size = static_cast<uint32_t>(tableData.size());
        tableSection.isData = true;
        tableSection.isReadOnly = true;
        tableSection.data = tableData.data();

        const R5900Decoder decoder;
        const auto decodeI = [&](uint32_t address, uint32_t opcode,
                                 uint8_t rs, uint8_t rt, uint16_t immediate)
        {
            const uint32_t raw =
                (opcode << 26u) |
                (static_cast<uint32_t>(rs) << 21u) |
                (static_cast<uint32_t>(rt) << 16u) |
                immediate;
            return decoder.decodeInstruction(address, raw, false);
        };
        const auto decodeR = [&](uint32_t address, uint8_t rs, uint8_t rt,
                                 uint8_t rd, uint8_t sa, uint8_t function)
        {
            const uint32_t raw =
                (static_cast<uint32_t>(rs) << 21u) |
                (static_cast<uint32_t>(rt) << 16u) |
                (static_cast<uint32_t>(rd) << 11u) |
                (static_cast<uint32_t>(sa) << 6u) |
                function;
            return decoder.decodeInstruction(address, raw, false);
        };

        std::vector<Instruction> instructions{
            decodeI(0x4000u, OPCODE_SLTIU, 4u, 2u, 2u),
            makeNop(0x4004u),
            decodeR(0x4008u, 0u, 4u, 8u, 2u, SPECIAL_SLL),
            decodeI(0x400Cu, OPCODE_LUI, 0u, 9u, 0x0020u),
            decodeI(0x4010u, OPCODE_ADDIU, 9u, 9u, 0u),
            decodeR(0x4014u, 8u, 9u, 9u, 0u, SPECIAL_ADDU),
            decodeI(0x4018u, OPCODE_LW, 9u, 10u, 0u),
            makeJr(0x401Cu, 10u),
            makeNop(0x4020u),
            makeNop(0x4024u),
            makeNop(0x4028u),
        };
        for (uint32_t address = 0x402Cu; address <= 0x4080u; address += 4u)
        {
            instructions.push_back(makeNop(address));
        }
        instructions.push_back(
            decodeI(0x4084u, OPCODE_LUI, 0u, 9u, 0x0020u));
        instructions.push_back(
            decodeI(0x4088u, OPCODE_ADDIU, 9u, 9u, 0x0010u));
        instructions.push_back(
            decodeR(0x408Cu, 8u, 9u, 9u, 0u, SPECIAL_ADDU));
        instructions.push_back(
            decodeI(0x4090u, OPCODE_LW, 9u, 10u, 0u));
        instructions.push_back(makeJr(0x4094u, 10u));
        instructions.push_back(makeNop(0x4098u));
        instructions.push_back(makeNop(0x409Cu));
        instructions.push_back(makeNop(0x40A0u));

        const std::vector<Symbol> symbols;
        const std::vector<Section> sections{tableSection};
        CodeGenerator gen(symbols, sections);
        CodeGenerator::AnalysisResult analysis =
            gen.collectInternalBranchTargets(func, instructions);
        t.IsTrue(analysis.jumpTableTargets.contains(0x401Cu),
                 "the nearby first use of the scaled index should resolve");
        t.IsTrue(analysis.jumpTableTargets.contains(0x4094u),
                 "the bounded unmodified scaled index should resolve after more than ten instructions");
        t.IsTrue(analysis.indirectFallbackEntryPoints.empty(),
                 "resolved reused-index tables should not request broad fallback entries");

        std::vector<Instruction> inPlaceInstructions = instructions;
        inPlaceInstructions[0] =
            decodeI(0x4000u, OPCODE_SLTIU, 8u, 2u, 2u);
        inPlaceInstructions[2] =
            decodeR(0x4008u, 0u, 8u, 8u, 2u, SPECIAL_SLL);
        analysis = gen.collectInternalBranchTargets(func, inPlaceInstructions);
        t.IsTrue(analysis.jumpTableTargets.contains(0x4094u),
                 "an in-place scaled index guarded before the shift should remain reusable");

        std::vector<Instruction> postShiftInPlaceInstructions =
            inPlaceInstructions;
        postShiftInPlaceInstructions[0] = makeNop(0x4000u);
        postShiftInPlaceInstructions[11] =
            decodeI(0x402Cu, OPCODE_SLTIU, 8u, 2u, 2u);
        analysis = gen.collectInternalBranchTargets(
            func, postShiftInPlaceInstructions);
        t.IsFalse(analysis.jumpTableTargets.contains(0x4094u),
                  "a guard after an in-place shift must not bound the original index");

        std::vector<Instruction> sourceClobberInstructions = instructions;
        sourceClobberInstructions[1] =
            decodeI(0x4004u, OPCODE_ADDIU, 0u, 4u, 0u);
        analysis = gen.collectInternalBranchTargets(
            func, sourceClobberInstructions);
        t.IsFalse(analysis.jumpTableTargets.contains(0x401Cu),
                  "a source overwrite between the guard and shift must reject the stale bound");

        writeTableTarget(0u, 0x4030u);
        writeTableTarget(4u, 0x4034u);
        std::vector<Instruction> copiedIndexInstructions{
            decodeI(0x4000u, OPCODE_LW, 16u, 2u, 0x50u),
            decodeR(0x4004u, 2u, 0u, 5u, 0u, SPECIAL_DADDU),
            makeNop(0x4008u),
            decodeI(0x400Cu, OPCODE_SLTIU, 2u, 2u, 2u),
            decodeI(0x4010u, OPCODE_BEQ, 2u, 0u, 8u),
            decodeI(0x4014u, OPCODE_LUI, 0u, 2u, 0x0020u),
            decodeR(0x4018u, 0u, 5u, 8u, 2u, SPECIAL_SLL),
            decodeI(0x401Cu, OPCODE_ADDIU, 2u, 2u, 0u),
            decodeR(0x4020u, 8u, 2u, 9u, 0u, SPECIAL_ADDU),
            decodeI(0x4024u, OPCODE_LW, 9u, 10u, 0u),
            makeJr(0x4028u, 10u),
            makeNop(0x402Cu),
            makeNop(0x4030u),
            makeNop(0x4034u),
            makeNop(0x4038u),
        };
        analysis = gen.collectInternalBranchTargets(
            func, copiedIndexInstructions);
        t.IsTrue(analysis.jumpTableTargets.contains(0x4028u),
                 "a bound on the source of an exact index copy should resolve the table");

        copiedIndexInstructions[2] =
            decodeI(0x4008u, OPCODE_ADDIU, 0u, 2u, 0u);
        analysis = gen.collectInternalBranchTargets(
            func, copiedIndexInstructions);
        t.IsFalse(analysis.jumpTableTargets.contains(0x4028u),
                  "a copied-index source overwrite before the guard must reject the stale bound");

        std::vector<Instruction> copiedScaledIndexInstructions{
            decodeR(0x4000u, 0u, 23u, 16u, 2u, SPECIAL_SLL),
            makeNop(0x4004u),
            makeNop(0x4008u),
            makeNop(0x400Cu),
            decodeR(0x4010u, 16u, 0u, 30u, 0u, SPECIAL_DADDU),
            decodeI(0x4014u, OPCODE_ADDIU, 0u, 16u, 0u),
        };
        for (uint32_t address = 0x4018u; address <= 0x414Cu;
             address += 4u)
        {
            copiedScaledIndexInstructions.push_back(makeNop(address));
        }
        copiedScaledIndexInstructions.push_back(
            decodeI(0x4150u, OPCODE_SLTIU, 23u, 3u, 2u));
        copiedScaledIndexInstructions.push_back(
            decodeI(0x4154u, OPCODE_BEQ, 3u, 0u, 6u));
        copiedScaledIndexInstructions.push_back(
            decodeI(0x4158u, OPCODE_LUI, 0u, 2u, 0x0020u));
        copiedScaledIndexInstructions.push_back(
            decodeI(0x415Cu, OPCODE_ADDIU, 2u, 2u, 0u));
        copiedScaledIndexInstructions.push_back(
            decodeR(0x4160u, 30u, 2u, 2u, 0u, SPECIAL_ADDU));
        copiedScaledIndexInstructions.push_back(
            decodeI(0x4164u, OPCODE_LW, 2u, 3u, 0u));
        copiedScaledIndexInstructions.push_back(makeJr(0x4168u, 3u));
        copiedScaledIndexInstructions.push_back(makeNop(0x416Cu));
        copiedScaledIndexInstructions.push_back(makeNop(0x4170u));
        analysis = gen.collectInternalBranchTargets(
            func, copiedScaledIndexInstructions);
        t.IsTrue(analysis.jumpTableTargets.contains(0x4168u),
                 "an exact post-scale copy should preserve a long-lived bounded table offset");
        t.IsTrue(analysis.indirectFallbackEntryPoints.empty(),
                 "a resolved post-scale copy should not request broad fallback entries");

        std::vector<Instruction> copiedScaleSourceClobberInstructions =
            copiedScaledIndexInstructions;
        copiedScaleSourceClobberInstructions[10] =
            decodeI(0x4028u, OPCODE_ADDIU, 0u, 23u, 0u);
        analysis = gen.collectInternalBranchTargets(
            func, copiedScaleSourceClobberInstructions);
        t.IsFalse(analysis.jumpTableTargets.contains(0x4168u),
                  "a source overwrite must invalidate a saved scaled offset");

        copiedScaledIndexInstructions[3] =
            decodeI(0x400Cu, OPCODE_ADDIU, 0u, 16u, 0u);
        analysis = gen.collectInternalBranchTargets(
            func, copiedScaledIndexInstructions);
        t.IsFalse(analysis.jumpTableTargets.contains(0x4168u),
                  "a scaled-value overwrite before its copy must reject the stale offset");

        copiedScaledIndexInstructions[3] = makeNop(0x400Cu);
        copiedScaledIndexInstructions[5] =
            decodeI(0x4014u, OPCODE_ADDIU, 0u, 30u, 0u);
        analysis = gen.collectInternalBranchTargets(
            func, copiedScaledIndexInstructions);
        t.IsFalse(analysis.jumpTableTargets.contains(0x4168u),
                  "a copied scaled-value overwrite before table use must reject the stale offset");

        instructions[15] =
            decodeI(instructions[15].address, OPCODE_ADDIU, 0u, 8u, 0u);
        analysis = gen.collectInternalBranchTargets(func, instructions);
        t.IsTrue(analysis.jumpTableTargets.contains(0x401Cu),
                 "an overwrite after the first dispatch must not invalidate the first table");
        t.IsFalse(analysis.jumpTableTargets.contains(0x4094u),
                  "an intervening scaled-index overwrite must prevent stale table inference");
        t.IsFalse(analysis.indirectFallbackEntryPoints.empty(),
                  "the rejected stale index should retain conservative fallback entries");
    });

    tc.Run("JALR inference validates read-only strided function tables", [](TestCase &t) {
        Function caller;
        caller.name = "strided_function_table_caller";
        caller.start = 0x5000u;
        caller.end = 0x5030u;
        caller.isRecompiled = true;

        Function targetA;
        targetA.name = "strided_target_a";
        targetA.start = 0x6000u;
        targetA.end = 0x6010u;
        targetA.isRecompiled = true;

        Function targetB;
        targetB.name = "strided_target_b";
        targetB.start = 0x6100u;
        targetB.end = 0x6110u;
        targetB.isRecompiled = true;

        constexpr uint32_t tableAddress = 0x00200000u;
        constexpr uint32_t recordStride = 12u;
        constexpr uint32_t callableFieldOffset = 4u;
        std::vector<uint8_t> tableData(recordStride * 3u, 0u);
        const auto writeTableWord = [&](size_t offset, uint32_t value)
        {
            tableData[offset] = static_cast<uint8_t>(value);
            tableData[offset + 1u] = static_cast<uint8_t>(value >> 8u);
            tableData[offset + 2u] = static_cast<uint8_t>(value >> 16u);
            tableData[offset + 3u] = static_cast<uint8_t>(value >> 24u);
        };
        writeTableWord(callableFieldOffset, 0x6004u);
        writeTableWord(recordStride + callableFieldOffset, 0x6100u);

        Section tableSection{};
        tableSection.name = ".vtable";
        tableSection.address = tableAddress;
        tableSection.size = static_cast<uint32_t>(tableData.size());
        tableSection.isData = true;
        tableSection.isReadOnly = true;
        tableSection.data = tableData.data();

        Section codeSection{};
        codeSection.name = ".text";
        codeSection.address = 0x5000u;
        codeSection.size = 0x1200u;
        codeSection.isCode = true;
        codeSection.isReadOnly = true;

        const R5900Decoder decoder;
        const auto decodeI = [&](uint32_t address, uint32_t opcode,
                                 uint8_t rs, uint8_t rt, uint16_t immediate)
        {
            const uint32_t raw =
                (opcode << 26u) |
                (static_cast<uint32_t>(rs) << 21u) |
                (static_cast<uint32_t>(rt) << 16u) |
                immediate;
            return decoder.decodeInstruction(address, raw, false);
        };
        const auto decodeR = [&](uint32_t address, uint8_t rs, uint8_t rt,
                                 uint8_t rd, uint8_t sa, uint8_t function)
        {
            const uint32_t raw =
                (static_cast<uint32_t>(rs) << 21u) |
                (static_cast<uint32_t>(rt) << 16u) |
                (static_cast<uint32_t>(rd) << 11u) |
                (static_cast<uint32_t>(sa) << 6u) |
                function;
            return decoder.decodeInstruction(address, raw, false);
        };

        const std::vector<Instruction> instructions{
            decodeI(0x5000u, OPCODE_ADDIU, 0u, 4u, recordStride),
            decodeR(0x5004u, 5u, 4u, 8u, 0u, SPECIAL_MULT),
            decodeI(0x5008u, OPCODE_LUI, 0u, 9u, 0x0020u),
            decodeI(0x500Cu, OPCODE_ADDIU, 9u, 9u, 0u),
            decodeR(0x5010u, 8u, 9u, 9u, 0u, SPECIAL_ADDU),
            decodeI(0x5014u, OPCODE_LW, 9u, 10u, callableFieldOffset),
            makeJalr(0x5018u, 10u, 31u),
            makeNop(0x501Cu),
            makeNop(0x5020u),
        };
        const std::vector<Function> functions{caller, targetA, targetB};
        const std::vector<Symbol> symbols;
        const std::vector<Section> sections{tableSection, codeSection};
        CodeGenerator gen(symbols, sections);

        CodeGenerator::AnalysisResult analysis =
            gen.collectInternalBranchTargets(
                caller, instructions, &functions);
        t.IsTrue(analysis.indirectFallbackEntryPoints.empty(),
                 "a fully validated read-only function table should not request fallback entries");
        t.IsTrue(analysis.externalEntryPoints.contains(0x6004u),
                 "a validated mid-function table target should be promoted in its owner");
        t.IsFalse(analysis.jumpTableTargets.contains(0x5018u),
                  "external callable targets should continue through runtime JALR dispatch");

        analysis = gen.collectInternalBranchTargets(caller, instructions);
        t.IsTrue(analysis.indirectFallbackEntryPoints.empty(),
                 "emission analysis should validate executable table targets without function metadata");

        std::vector<Section> writableSections{tableSection, codeSection};
        writableSections[0].isReadOnly = false;
        CodeGenerator writableGen(symbols, writableSections);
        analysis = writableGen.collectInternalBranchTargets(
            caller, instructions, &functions);
        t.IsFalse(analysis.indirectFallbackEntryPoints.empty(),
                  "a writable function table must retain conservative fallback entries");

        writableGen.setConfiguredFunctionTableRanges({
            FunctionTableRange{
                tableAddress,
                static_cast<uint32_t>(tableData.size())}});
        analysis = writableGen.collectInternalBranchTargets(
            caller, instructions, &functions);
        t.IsTrue(analysis.indirectFallbackEntryPoints.empty(),
                 "an exact configured function-table range should permit validation inside writable ELF data");
        t.IsTrue(analysis.externalEntryPoints.contains(0x6004u),
                 "configured writable-section tables should promote validated mid-function targets");

        bool rejectedUnbackedRange = false;
        try
        {
            writableGen.setConfiguredFunctionTableRanges({
                FunctionTableRange{0x00300000u, 0x20u}});
        }
        catch (const std::runtime_error &)
        {
            rejectedUnbackedRange = true;
        }
        t.IsTrue(rejectedUnbackedRange,
                 "configured function-table ranges must be fully backed by ELF data");

        Function longBaseCaller;
        longBaseCaller.name = "long_lived_function_table_base_caller";
        longBaseCaller.start = 0x5200u;
        longBaseCaller.end = 0x529Cu;
        longBaseCaller.isRecompiled = true;

        std::vector<Instruction> longBaseInstructions{
            decodeI(0x5200u, OPCODE_LUI, 0u, 22u, 0x0020u),
            decodeI(0x5204u, OPCODE_ADDIU, 22u, 22u, 0u),
            decodeI(0x5208u, OPCODE_BEQ, 4u, 0u, 3u),
            makeNop(0x520Cu),
            makeNop(0x5210u),
            makeNop(0x5214u),
        };
        for (uint32_t address = 0x5218u; address <= 0x5280u;
             address += 4u)
        {
            longBaseInstructions.push_back(makeNop(address));
        }
        longBaseInstructions.push_back(
            decodeI(0x5284u, OPCODE_ADDIU, 0u, 4u, recordStride));
        longBaseInstructions.push_back(
            decodeR(0x5288u, 5u, 4u, 8u, 0u, SPECIAL_MULT));
        longBaseInstructions.push_back(
            decodeR(0x528Cu, 8u, 22u, 9u, 0u, SPECIAL_ADDU));
        longBaseInstructions.push_back(
            decodeI(0x5290u, OPCODE_LW, 9u, 10u, callableFieldOffset));
        longBaseInstructions.push_back(makeJalr(0x5294u, 10u, 31u));
        longBaseInstructions.push_back(makeNop(0x5298u));

        const std::vector<Function> longBaseFunctions{
            longBaseCaller, targetA, targetB};
        analysis = writableGen.collectInternalBranchTargets(
            longBaseCaller, longBaseInstructions, &longBaseFunctions);
        t.IsTrue(analysis.indirectFallbackEntryPoints.empty(),
                 "a path-invariant table base should survive beyond the old backward-scan window");

        std::vector<Instruction> pathClobberInstructions =
            longBaseInstructions;
        pathClobberInstructions[5] =
            decodeI(0x5214u, OPCODE_ADDIU, 0u, 22u, 0u);
        analysis = writableGen.collectInternalBranchTargets(
            longBaseCaller, pathClobberInstructions, &longBaseFunctions);
        t.IsFalse(analysis.indirectFallbackEntryPoints.empty(),
                  "a path-dependent table-base assignment must merge to unknown");

        writeTableWord(recordStride + callableFieldOffset, 0x12345678u);
        analysis = gen.collectInternalBranchTargets(
            caller, instructions, &functions);
        t.IsFalse(analysis.indirectFallbackEntryPoints.empty(),
                  "a non-executable non-null table field must reject the whole inference");
    });

    tc.Run("resume entry targets emit a top-level pc switch in the owner wrapper", [](TestCase &t) {
        Function func;
        func.name = "resume_owner";
        func.start = 0x6000;
        func.end = 0x6010;
        func.isRecompiled = true;
        func.isStub = false;

        std::vector<Instruction> instructions{
            makeNop(0x6000),
            makeNop(0x6004),
            makeNop(0x6008),
            makeNop(0x600C)
        };

        CodeGenerator gen({}, {});
        gen.setResumeEntryTargets({{0x6000u, {0x6008u}}});

        std::string generated = gen.generateFunction(func, instructions, false);
        printGeneratedCode("resume entry targets emit a top-level pc switch in the owner wrapper", generated);

        t.IsTrue(generated.find("switch (ctx->pc)") != std::string::npos,
                 "owner wrapper should dispatch resumable pcs with a switch");
        t.IsTrue(generated.find("case 0x6008u: goto label_6008;") != std::string::npos,
                 "resume pc should jump directly to the internal label");
        t.IsTrue(generated.find("label_6008:") != std::string::npos,
                 "resume pc should force label emission for that instruction");
    });

    tc.Run("resume entry audit rejects missing labels owners and collisions", [](TestCase &t) {
        Function func;
        func.name = "resume_owner";
        func.start = 0x6000u;
        func.end = 0x6010u;
        func.isRecompiled = true;
        func.isStub = false;

        const std::vector<Instruction> instructions{
            makeNop(0x6000u),
            makeNop(0x6004u),
            makeNop(0x6008u),
            makeNop(0x600Cu),
        };

        CodeGenerator missingLabel({}, {});
        missingLabel.setResumeEntryTargets(
            {{0x6000u, {0x6010u}}});
        bool rejectedMissingLabel = false;
        try
        {
            (void)missingLabel.generateFunction(
                func, instructions, false);
        }
        catch (const std::runtime_error &)
        {
            rejectedMissingLabel = true;
        }
        t.IsTrue(
            rejectedMissingLabel,
            "a resume entry outside the owner's decoded instructions should fail generation");

        CodeGenerator missingOwner({}, {});
        missingOwner.setResumeEntryTargets(
            {{0x7000u, {0x7008u}}});
        bool rejectedMissingOwner = false;
        try
        {
            (void)missingOwner.generateFunctionRegistration(
                {func}, {});
        }
        catch (const std::runtime_error &)
        {
            rejectedMissingOwner = true;
        }
        t.IsTrue(
            rejectedMissingOwner,
            "a resume-entry table mapping without a generated owner should fail generation");

        Function conflictingOwner = func;
        conflictingOwner.name = "conflicting_owner";
        conflictingOwner.start = 0x7000u;
        conflictingOwner.end = 0x7010u;

        CodeGenerator conflictingMapping({}, {});
        conflictingMapping.setRenamedFunctions(
            {{0x6000u, "resume_owner_0x6000"},
             {0x7000u, "conflicting_owner_0x7000"}});
        conflictingMapping.setResumeEntryTargets(
            {{0x7000u, {0x6000u}}});
        bool rejectedOwnerCollision = false;
        try
        {
            (void)conflictingMapping.generateFunctionRegistration(
                {func, conflictingOwner}, {});
        }
        catch (const std::runtime_error &)
        {
            rejectedOwnerCollision = true;
        }
        t.IsTrue(
            rejectedOwnerCollision,
            "a resume entry already mapped to a different wrapper should fail generation");
    });

    tc.Run("resume entry targets register to the owner wrapper", [](TestCase &t) {
        Function func;
        func.name = "resume_owner";
        func.start = 0x7000;
        func.end = 0x7010;
        func.isRecompiled = true;
        func.isStub = false;

        CodeGenerator gen({}, {});
        RecompilerReporter reporter;
        gen.setRenamedFunctions({{0x7000u, "resume_owner_0x7000"}});
        gen.setResumeEntryTargets(
            {{0x7000u, {0x7000u, 0x7008u, 0x700Cu}}});
        gen.setReporter(&reporter);

        const std::vector<Instruction> instructions{
            makeNop(0x7000u),
            makeNop(0x7004u),
            makeNop(0x7008u),
            makeNop(0x700Cu),
        };
        (void)gen.generateFunction(func, instructions, false);

        std::string registration = gen.generateFunctionRegistration({func}, {});
        printGeneratedCode("resume entry targets register to the owner wrapper", registration);

        t.IsTrue(
            registration.find(
                "g_ps2RecompiledFunctionPairAbiVersion = PS2_RECOMPILED_FUNCTION_PAIR_ABI_VERSION") !=
                std::string::npos,
            "primary registration should publish the function-pair ABI version");
        t.IsTrue(
            registration.find(
                "PS2Runtime::RecompiledFunctionPair g_ps2RecompiledFunctionTable") !=
                std::string::npos,
            "the dense table should retain O(1) slots containing function pairs");
        t.IsTrue(registration.find("g_ps2RecompiledFunctionTable[2] = {resume_owner_0x7000__ee_fast, resume_owner_0x7000__ee_precise}; // 0x7008") != std::string::npos,
                 "resume entry pc should register to the owner wrapper");
        t.IsTrue(registration.find("g_ps2RecompiledFunctionTable[3] = {resume_owner_0x7000__ee_fast, resume_owner_0x7000__ee_precise}; // 0x700c") != std::string::npos,
                 "multiple resume pcs should register to the same owner wrapper");
        const RecompilerReporter::Counters counters = reporter.counters();
        t.Equals(counters.resumeEntryTargetsAudited, size_t{3},
                 "the owner emitter should audit every mapped resume target");
        t.Equals(counters.resumeEntryTargetsRegistered, size_t{3},
                 "the function table should register every audited resume target");
    });

    tc.Run("function registration emits guest symbol ranges", [](TestCase &t) {
        Function mapped;
        mapped.name = "mapped \"function\"";
        mapped.start = 0x8000u;
        mapped.end = 0x8020u;

        CodeGenerator gen({}, {});
        std::string registration = gen.generateFunctionRegistration({mapped}, {});
        printGeneratedCode("function registration emits guest symbol ranges", registration);

        t.IsTrue(registration.find("g_ps2GuestFunctionSymbolCount = 1u") != std::string::npos,
                 "registration should publish the number of mapped guest symbols");
        t.IsTrue(registration.find(
                     "{0x8000u, 0x8020u, \"mapped \\\"function\\\"\"}") != std::string::npos,
                 "registration should retain map names and ranges as escaped string metadata");
    });

    tc.Run("module registration emits signature-activated descriptors", [](TestCase &t) {
        Function mapped;
        mapped.name = "overlay_entry";
        mapped.start = 0x9000u;
        mapped.end = 0x9040u;
        mapped.isRecompiled = true;

        CodeGenerator gen({}, {});
        gen.setRenamedFunctions(
            {{mapped.start, "ps2_module_test_overlay_entry_0x9000"}});
        CodeGenerator::ModuleInfo module{};
        module.valid = true;
        module.name = "test-overlay";
        module.matchAddress = mapped.start;
        module.signature = {0x70u, 0xFFu, 0xBDu, 0x27u};
        gen.setModuleInfo(module);

        std::string registration =
            gen.generateFunctionRegistration({mapped}, {});
        printGeneratedCode(
            "module registration emits signature-activated descriptors",
            registration);

        t.IsTrue(
            registration.find(
                "{0x9000u, {ps2_module_test_overlay_entry_0x9000__ee_fast, ps2_module_test_overlay_entry_0x9000__ee_precise}}") !=
                std::string::npos,
            "module registration should retain the prefixed native entry");
        t.IsTrue(
            registration.find(
                "PS2_RECOMPILED_FUNCTION_PAIR_ABI_VERSION") !=
                std::string::npos,
            "module registration should declare its function-pair ABI version");
        t.IsTrue(
            registration.find("\"test-overlay\"") != std::string::npos &&
                registration.find("0x9000u") != std::string::npos &&
                registration.find("g_moduleSignature") != std::string::npos,
            "module registration should embed its name, match address, and bytes");
        t.IsTrue(
            registration.find("ps2RegisterRecompiledModule") !=
                std::string::npos,
            "module registration should publish its descriptor at startup");
        t.IsTrue(
            registration.find("g_ps2RecompiledFunctionTableBase") ==
                std::string::npos,
            "a secondary module must not define the primary dense table");
    });

    tc.Run("external mid-function entry can register to the owner wrapper", [](TestCase &t) {
        Function caller;
        caller.name = "caller";
        caller.start = 0x4000;
        caller.end = 0x4008;
        caller.isRecompiled = true;
        caller.isStub = false;

        Function owner;
        owner.name = "owner";
        owner.start = 0x5000;
        owner.end = 0x5010;
        owner.isRecompiled = true;
        owner.isStub = false;

        Instruction j{};
        j.address = 0x4000;
        j.opcode = OPCODE_J;
        j.target = (0x5004u >> 2) & 0x3FFFFFFu;
        j.hasDelaySlot = true;
        j.raw = (OPCODE_J << 26) | (j.target & 0x3FFFFFFu);

        std::vector<Instruction> callerInstructions{j, makeNop(0x4004)};
        std::vector<Function> functions{caller, owner};
        std::vector<Section> sections = {
            {".text", 0x4000u, 0x2000u, 0u, true, false, false, true, nullptr}
        };

        CodeGenerator gen({}, sections);
        CodeGenerator::AnalysisResult analysis =
            gen.collectInternalBranchTargets(caller, callerInstructions, &functions);

        t.IsTrue(analysis.externalEntryPoints.contains(0x5004u),
                 "cross-function jump should identify the mid-function target as externally reachable");

        gen.setRenamedFunctions({{0x5000u, "owner_0x5000"}});
        gen.setResumeEntryTargets({{0x5000u, {0x5004u}}});

        std::string registration = gen.generateFunctionRegistration({owner}, {});
        printGeneratedCode("external mid-function entry can register to the owner wrapper", registration);

        t.IsTrue(registration.find("g_ps2RecompiledFunctionTable[1] = {owner_0x5000__ee_fast, owner_0x5000__ee_precise}; // 0x5004") != std::string::npos,
                 "mid-function external entry should register back to the owner wrapper");
    });

    tc.Run("branches outside function still set pc", [](TestCase &t) {
        Function func;
        func.name = "external_branch";
        func.start = 0x3000;
        func.end = 0x3020;
        func.isRecompiled = true;
        func.isStub = false;

        // Branch targets outside the function range
        std::vector<Instruction> instructions;
        instructions.push_back(makeBranch(0x3000, 4)); // target = 0x3014 (inside) -> make it outside by adjusting end? easier: set end smaller? Instead use large offset
        instructions.clear();
        Instruction br = makeBranch(0x3000, 0x100); // target far outside
        instructions.push_back(br);
        instructions.push_back(makeNop(0x3004)); // delay slot

        CodeGenerator gen({}, {});
        std::string generated = gen.generateFunction(func, instructions, false);
        printGeneratedCode("branches outside function still set pc", generated);

        t.IsTrue(generated.find("ctx->pc = 0x") != std::string::npos, "external branch should set ctx->pc");
        t.IsTrue(generated.find("goto label_") == std::string::npos, "external branch should not use goto");
    });

    tc.Run("jumps to known symbols call by name", [](TestCase &t) {
        Function func;
        func.name = "call_symbol";
        func.start = 0x4000;
        func.end = 0x4018;
        func.isRecompiled = true;
        func.isStub = false;

        Symbol targetSym;
        targetSym.name = "target_func";
        targetSym.address = 0x5000;
        targetSym.isFunction = true;

        Instruction j{};
        j.address = 0x4000;
        j.opcode = OPCODE_J;
        j.target = (targetSym.address >> 2) & 0x3FFFFFF;
        j.hasDelaySlot = true;
        j.raw = 0x08000000 | (j.target & 0x3FFFFFF);

        Instruction delay = makeNop(0x4004);

        std::vector<Instruction> instructions{j, delay, makeNop(0x4008)};

        CodeGenerator gen({targetSym}, {});
        std::string generated = gen.generateFunction(func, instructions, false);
        printGeneratedCode("jumps to known symbols call by name", generated);

        t.IsTrue(generated.find("target_func__ee_fast(rdram, ctx, runtime);") != std::string::npos &&
                     generated.find("target_func__ee_precise(rdram, ctx, runtime);") != std::string::npos,
                 "jump to known function should emit direct call");
    });

    tc.Run("jump to unknown target sets pc", [](TestCase &t) {
            Function func;
            func.name = "jump_unknown";
            func.start = 0x6000;
            func.end = 0x6010;
            func.isRecompiled = true;
            func.isStub = false;

            Instruction j{};
            j.address = 0x6000;
            j.opcode = OPCODE_J;
            j.target = 0x001234; // target = 0x00048d0
            j.hasDelaySlot = true;
            j.raw = (OPCODE_J << 26) | (j.target & 0x3FFFFFF);
            Instruction delay = makeNop(0x6004);

            std::vector<Instruction> instructions{j, delay};

            CodeGenerator gen({}, {});
            std::string generated = gen.generateFunction(func, instructions, false);
            printGeneratedCode("jump to unknown target sets pc", generated);

            t.IsTrue(generated.find("ctx->pc = 0x") != std::string::npos, "unknown jump target should set ctx->pc");
            t.IsTrue(generated.find("goto label_") == std::string::npos, "external jump should not use goto");
        });

        tc.Run("renamed function used in jump table", [](TestCase &t) {
            Function func;
            func.name = "jt_func";
            func.start = 0x7000;
            func.end = 0x7010;
            func.isRecompiled = true;
            func.isStub = false;

            JumpTableEntry entry;
            entry.index = 0;
            entry.target = 0x8000;
            std::vector<JumpTableEntry> entries{entry};

            Instruction inst{};
            inst.opcode = OPCODE_REGIMM;

            CodeGenerator gen({}, {});
            gen.setRenamedFunctions({{0x8000, "renamed_target"}});

            std::string sw = gen.generateJumpTableSwitch(inst, 0x0, entries);
            printGeneratedCode("renamed function used in jump table", sw);

        t.IsTrue(sw.find("renamed_target__ee_fast(rdram, ctx, runtime);") != std::string::npos &&
                     sw.find("renamed_target__ee_precise(rdram, ctx, runtime);") != std::string::npos,
                 "jump table should use renamed function name");
        });

        tc.Run("reserved identifiers are sanitized and used in calls", [](TestCase &t) {
            Function func;
            func.name = "__is_pointer";
            func.start = 0x9000;
            func.end = 0x9010;
            func.isRecompiled = true;
            func.isStub = false;

            Symbol targetSym;
            targetSym.name = "__is_pointer";
            targetSym.address = func.start;
            targetSym.isFunction = true;

            Instruction j{};
            j.address = 0x8000;
            j.opcode = OPCODE_J;
            j.target = (targetSym.address >> 2) & 0x3FFFFFF;
            j.hasDelaySlot = true;
            j.raw = (OPCODE_J << 26) | (j.target & 0x3FFFFFF);
            Instruction delay = makeNop(0x8004);

            std::vector<Instruction> instructions{j, delay};

            CodeGenerator gen({targetSym}, {});
            gen.setRenamedFunctions({{targetSym.address, "ps2___is_pointer"}});

            std::string generated = gen.generateFunction(func, instructions, false);
            printGeneratedCode("reserved identifiers are sanitized and used in calls", generated);

            t.IsTrue(generated.find("void ps2___is_pointer__ee_fast(") != std::string::npos &&
                         generated.find("void ps2___is_pointer__ee_precise(") != std::string::npos,
                     "definition should use sanitized name");
            t.IsTrue(generated.find("ps2___is_pointer__ee_fast(rdram, ctx, runtime);") != std::string::npos &&
                         generated.find("ps2___is_pointer__ee_precise(rdram, ctx, runtime);") != std::string::npos,
                "call should use sanitized name but got: " + generated);
        });

        tc.Run("COP0 MFC0/MTC0 translate to COP0 register access", [](TestCase &t) {
            CodeGenerator gen({}, {});

            Instruction mfc0{};
            mfc0.opcode = OPCODE_COP0;
            mfc0.rs = COP0_MF;
            mfc0.rt = 5;
            mfc0.rd = COP0_REG_STATUS;

            std::string mfc0Code = gen.translateInstruction(mfc0);
            printGeneratedCode("COP0 MFC0/MTC0 translate to COP0 register access (MFC0)", mfc0Code);
            t.IsTrue(mfc0Code.find("SET_GPR_S32(ctx, 5") != std::string::npos, "MFC0 should write to rt");
            t.IsTrue(mfc0Code.find("ctx->cop0_status") != std::string::npos, "MFC0 STATUS should read cop0_status");
            t.IsTrue(mfc0Code.find("Unimplemented COP0 register") == std::string::npos, "MFC0 should not hit unimplemented COP0 register path");
            t.IsTrue(mfc0Code.find("Unhandled COP0") == std::string::npos, "MFC0 should not hit unhandled COP0 path");

            Instruction mtc0{};
            mtc0.opcode = OPCODE_COP0;
            mtc0.rs = COP0_MT;
            mtc0.rt = 7;
            mtc0.rd = COP0_REG_STATUS;

            std::string mtc0Code = gen.translateInstruction(mtc0);
            printGeneratedCode("COP0 MFC0/MTC0 translate to COP0 register access (MTC0)", mtc0Code);
            t.IsTrue(
                mtc0Code.find("runtime->writeCop0Status") != std::string::npos,
                "MTC0 STATUS should route timing-sensitive state through the runtime");
            t.IsTrue(mtc0Code.find("GPR_U32(ctx, 7)") != std::string::npos, "MTC0 should read from rt");
            t.IsTrue(mtc0Code.find("Unimplemented MTC0") == std::string::npos, "MTC0 should not hit unimplemented path");
            t.IsTrue(mtc0Code.find("Unhandled COP0") == std::string::npos, "MTC0 should not hit unhandled COP0 path");

            mtc0.rd = COP0_REG_CONFIG;
            mtc0Code = gen.translateInstruction(mtc0);
            t.IsTrue(
                mtc0Code.find("ctx->cop0_config & 0x70000FC0u") != std::string::npos,
                "MTC0 CONFIG should preserve hardware configuration fields");
            t.IsTrue(
                mtc0Code.find("GPR_U32(ctx, 7) & 0x00073007u") != std::string::npos,
                "MTC0 CONFIG should accept only software-owned fields");

            mfc0.rt = 0;
            mfc0.rd = COP0_REG_COUNT;
            mfc0Code = gen.translateInstruction(mfc0);
            t.IsTrue(
                mfc0Code.find("runtime->readCop0Count") != std::string::npos,
                "MFC0 Count should update runtime-owned time even when rt is zero");

            mfc0.rt = 5;
            mfc0.rd = COP0_REG_PERF;
            mfc0.raw = 3u;
            mfc0Code = gen.translateInstruction(mfc0);
            t.IsTrue(
                mfc0Code.find("readCop0Performance(rdram, ctx, 3u)") != std::string::npos,
                "MFC0 register 25 should preserve the PCR selector bits");

            mfc0.rt = 0;
            mfc0Code = gen.translateInstruction(mfc0);
            t.IsTrue(
                mfc0Code.find("readCop0Performance") != std::string::npos,
                "MFC0 performance code should retain the access behind the zero-register guard");
            t.IsFalse(
                mfc0Code.find("const uint32_t value") != std::string::npos,
                "MFC0 performance should not force an access when rt is zero");

            mtc0.rd = COP0_REG_PERF;
            mtc0.raw = 1u;
            mtc0Code = gen.translateInstruction(mtc0);
            t.IsTrue(
                mtc0Code.find("writeCop0Performance(rdram, ctx, 1u") != std::string::npos,
                "MTC0 register 25 should preserve the PCR selector bits");
        });

        tc.Run("MTC0 Status validates the next sequential instruction fetch", [](TestCase &t) {
            Function function;
            function.name = "status_mode_transition";
            function.start = 0x80001000u;
            function.end = 0x80001008u;
            function.isRecompiled = true;

            Instruction mtc0{};
            mtc0.address = 0x80001000u;
            mtc0.opcode = OPCODE_COP0;
            mtc0.rs = COP0_MT;
            mtc0.rt = 7u;
            mtc0.rd = COP0_REG_STATUS;
            mtc0.raw =
                (OPCODE_COP0 << 26u) |
                (COP0_MT << 21u) |
                (7u << 16u) |
                (COP0_REG_STATUS << 11u);

            const std::string generated =
                CodeGenerator({}, {}).generateFunction(
                    function,
                    {mtc0, makeAddiu(0x80001004u, 2u, 2u, 1u)},
                    false);
            const size_t statusWrite = generated.find(
                "runtime->writeCop0Status");
            const size_t nextPc = generated.find(
                "ctx->pc = 0x80001004u;", statusWrite);
            const size_t validation = generated.find(
                "runtime->validateEeInstructionFetchPolicy<Mode>(ctx, ctx->pc);",
                nextPc);
            const size_t nextEffect = generated.find(
                "SET_GPR_S32(ctx, 2", validation);

            t.IsTrue(
                statusWrite != std::string::npos &&
                    nextPc != std::string::npos &&
                    validation != std::string::npos &&
                    nextEffect != std::string::npos &&
                    statusWrite < nextPc &&
                    nextPc < validation &&
                    validation < nextEffect,
                "a Status mode change should gate the following fetch before its effect");
        });

        tc.Run("CACHE dispatches its operation and effective address", [](TestCase &t) {
            constexpr uint32_t raw =
                (OPCODE_CACHE << 26u) |
                (5u << 21u) |
                (0x12u << 16u) |
                0xFFFCu;
            const Instruction cache =
                R5900Decoder{}.decodeInstruction(0x1000u, raw, false);
            const std::string code =
                CodeGenerator({}, {}).translateInstruction(cache);

            printGeneratedCode(
                "CACHE dispatches its operation and effective address",
                code);
            t.IsTrue(
                code.find(
                    "runtime->handleEeCacheOperation(rdram, ctx, 0x12u, "
                    "ADD32(GPR_U32(ctx, 5), 0xFFFFFFFCu))") !=
                    std::string::npos,
                "CACHE should preserve op and use a wrapping base-plus-offset address");
            t.IsFalse(
                code.find("ignored") != std::string::npos,
                "architecturally visible CACHE operations must not be discarded");
        });

        tc.Run("FCR access uses CFC1/CTC1", [](TestCase &t) {
            CodeGenerator gen({}, {});

            Instruction cfc1{};
            cfc1.opcode = OPCODE_COP1;
            cfc1.rs = COP1_CF;
            cfc1.rt = 4;
            cfc1.rd = 31;

            std::string cfc1Code = gen.translateInstruction(cfc1);
            printGeneratedCode("FCR access uses CFC1/CTC1 (CFC1)", cfc1Code);
            t.IsTrue(cfc1Code.find("SET_GPR_U32(ctx, 4") != std::string::npos, "CFC1 should write to rt");
            t.IsTrue(cfc1Code.find("ctx->fcr31") != std::string::npos, "CFC1 FCR31 should read fcr31");
            t.IsTrue(cfc1Code.find("Unimplemented FCR") == std::string::npos, "CFC1 should not hit unimplemented FCR path");

            Instruction ctc1{};
            ctc1.opcode = OPCODE_COP1;
            ctc1.rs = COP1_CT;
            ctc1.rt = 4;
            ctc1.rd = 31;

            std::string ctc1Code = gen.translateInstruction(ctc1);
            printGeneratedCode("FCR access uses CFC1/CTC1 (CTC1)", ctc1Code);
            t.IsTrue(ctc1Code.find("ctx->fcr31 = GPR_U32(ctx, 4) & 0x0183FFFF") != std::string::npos,
                     "CTC1 FCR31 should mask and write fcr31");
            t.IsTrue(ctc1Code.find("ignored") == std::string::npos, "CTC1 FCR31 should not be ignored");
        });

        tc.Run("VU CReg access uses architectural CFC2/CTC2 indices", [](TestCase &t) {
            CodeGenerator gen({}, {});

            Instruction cfc2{};
            cfc2.opcode = OPCODE_COP2;
            cfc2.rs = COP2_CFC2;
            cfc2.rt = 2;
            cfc2.rd = 18;

            std::string cfc2Code = gen.translateInstruction(cfc2);
            printGeneratedCode("VU CReg access uses architectural CFC2/CTC2 indices (CFC2 CLIP)", cfc2Code);
            t.IsTrue(cfc2Code.find("SET_GPR_U32(ctx, 2") != std::string::npos, "CFC2 should write to rt");
            t.IsTrue(cfc2Code.find("ctx->vu0_clip_flags") != std::string::npos, "CFC2 VI18 should read the CLIP register");
            t.IsTrue(cfc2Code.find("vu0_fbrst") == std::string::npos, "CFC2 VI18 must not alias FBRST");

            Instruction ctc2{};
            ctc2.opcode = OPCODE_COP2;
            ctc2.rs = COP2_CTC2;
            ctc2.rt = 3;
            ctc2.rd = 18;

            std::string ctc2Code = gen.translateInstruction(ctc2);
            printGeneratedCode("VU CReg access uses architectural CFC2/CTC2 indices (CTC2 CLIP)", ctc2Code);
            t.IsTrue(ctc2Code.find("ctx->vu0_clip_flags") != std::string::npos, "CTC2 VI18 should write the CLIP register");
            t.IsTrue(ctc2Code.find("& 0xFFFFFF") != std::string::npos, "CTC2 CLIP should keep the architectural 24 bits");

            Instruction cfc2Vi{};
            cfc2Vi.opcode = OPCODE_COP2;
            cfc2Vi.rs = COP2_CFC2;
            cfc2Vi.rt = 4;
            cfc2Vi.rd = 7;
            const std::string cfc2ViCode = gen.translateInstruction(cfc2Vi);
            t.IsTrue(cfc2ViCode.find("ctx->vi[7]") != std::string::npos, "CFC2 VI07 should read the integer register bank");

            Instruction cfc2Q = cfc2;
            cfc2Q.rd = 22;
            const std::string cfc2QCode = gen.translateInstruction(cfc2Q);
            t.IsTrue(cfc2QCode.find("ctx->vu0_q") != std::string::npos, "CFC2 VI22 should read Q");
        });

        tc.Run("VU0 calls and transfers preserve micro-mode interlocks", [](TestCase &t) {
            CodeGenerator gen({}, {});

            Instruction vcallms{};
            vcallms.opcode = OPCODE_COP2;
            vcallms.rs = COP2_CO;
            vcallms.function = VU0_S1_VCALLMS;
            vcallms.raw = 0x4A000038u | (0x12u << 6u);

            const std::string vcallmsCode =
                gen.translateInstruction(vcallms);
            t.IsTrue(
                vcallmsCode.find(
                    "synchronizeVU0Microprogram(rdram, ctx, true)") !=
                    std::string::npos,
                "VCALLMS should finish an earlier microprogram before starting");
            t.IsTrue(
                vcallmsCode.find(
                    "vu0StartMicroProgram(rdram, ctx, 0x90)") !=
                    std::string::npos,
                "VCALLMS should start its target without synchronously completing it");
            t.IsTrue(
                vcallmsCode.find("executeVU0Microprogram") ==
                    std::string::npos,
                "generated VCALLMS should not serialize EE and VU0 execution");

            Instruction ctc2{};
            ctc2.opcode = OPCODE_COP2;
            ctc2.rs = COP2_CTC2;
            ctc2.rt = 3u;
            ctc2.rd = 1u;
            ctc2.raw = 0x48C30800u;

            const std::string nonInterlocked =
                gen.translateInstruction(ctc2);
            t.IsTrue(
                nonInterlocked.find(
                    "synchronizeVU0Microprogram(rdram, ctx, false)") !=
                    std::string::npos,
                "CTC2.ni should advance VU0 without waiting for completion");

            ctc2.raw |= 1u;
            const std::string interlocked =
                gen.translateInstruction(ctc2);
            t.IsTrue(
                interlocked.find(
                    "synchronizeVU0Microprogram(rdram, ctx, true)") !=
                    std::string::npos,
                "CTC2.i should wait for the active microprogram");

            Instruction lqc2{};
            lqc2.opcode = OPCODE_LQC2;
            lqc2.rs = 4u;
            lqc2.rt = 5u;
            const std::string lqc2Code =
                gen.translateInstruction(lqc2);
            t.IsTrue(
                lqc2Code.find(
                    "synchronizeVU0Microprogram(rdram, ctx, false)") !=
                    std::string::npos,
                "LQC2 should remain non-interlocked while advancing concurrent VU0 work");
        });

        tc.Run("VU0 transfer chains synchronize only at scheduling boundaries", [](TestCase &t) {
            Function function;
            function.name = "vu0_transfer_chain";
            function.start = 0x3000u;
            function.end = 0x302Cu;
            function.isRecompiled = true;

            Instruction vcall{};
            vcall.address = 0x3000u;
            vcall.opcode = OPCODE_COP2;
            vcall.rs = COP2_CO;
            vcall.function = VU0_S1_VCALLMS;
            vcall.raw = 0x4A000038u;

            auto makeQmtc2 = [](uint32_t address, uint32_t rt, uint32_t rd)
            {
                Instruction inst{};
                inst.address = address;
                inst.opcode = OPCODE_COP2;
                inst.rs = COP2_QMTC2;
                inst.rt = rt;
                inst.rd = rd;
                inst.raw =
                    (OPCODE_COP2 << 26u) |
                    (COP2_QMTC2 << 21u) |
                    (rt << 16u) |
                    (rd << 11u);
                return inst;
            };

            Instruction lqc2{};
            lqc2.address = 0x3010u;
            lqc2.opcode = OPCODE_LQC2;
            lqc2.rs = 4u;
            lqc2.rt = 5u;

            Instruction clear{};
            clear.address = 0x3014u;
            clear.opcode = OPCODE_COP2;
            clear.rs = COP2_CTC2;
            clear.rt = 0u;
            clear.rd = 1u;
            clear.raw =
                (OPCODE_COP2 << 26u) |
                (COP2_CTC2 << 21u) |
                (1u << 11u);

            Instruction qmfc2{};
            qmfc2.address = 0x301Cu;
            qmfc2.opcode = OPCODE_COP2;
            qmfc2.rs = COP2_QMFC2;
            qmfc2.rt = 6u;
            qmfc2.rd = 7u;
            qmfc2.raw =
                (OPCODE_COP2 << 26u) |
                (COP2_QMFC2 << 21u) |
                (6u << 16u) |
                (7u << 11u);

            Instruction branch = makeBranch(0x3020u, 1u);
            const std::string generated = CodeGenerator({}, {}).generateFunction(
                function,
                {
                    vcall,
                    makeNop(0x3004u),
                    makeQmtc2(0x3008u, 8u, 1u),
                    makeQmtc2(0x300Cu, 9u, 2u),
                    lqc2,
                    clear,
                    makeNop(0x3018u),
                    qmfc2,
                    branch,
                    makeNop(0x3024u),
                    makeQmtc2(0x3028u, 10u, 3u),
                },
                false);

            t.Equals(
                countOccurrences(
                    generated,
                    "synchronizeVU0Microprogram(rdram, ctx, false)"),
                static_cast<size_t>(4u),
                "the four COP2 scheduling points should retain minimum-batch synchronization");
            t.Equals(
                countOccurrences(
                    generated,
                    "serviceEeEventsAtGeneratedBlockBoundary(rdram, ctx)"),
                static_cast<size_t>(2u),
                "taken and not-taken paths should each service the completed EE block");
            t.Equals(
                countOccurrences(
                    generated,
                    "synchronizeVU0Microprogram(rdram, ctx, true)"),
                static_cast<size_t>(1u),
                "VCALLMS should be the only finishing operation in this chain");
            t.Equals(
                countOccurrences(generated, "ctx->enforceVu0RegisterInvariants();"),
                static_cast<size_t>(7u),
                "skipped catch-ups must still execute every COP2 transfer");
        });

        tc.Run("an interlocked VU0 transfer keeps its whole EE block synchronized", [](TestCase &t) {
            Function function;
            function.name = "vu0_interlocked_block";
            function.start = 0x3100u;
            function.end = 0x310Cu;
            function.isRecompiled = true;

            auto makeQmtc2 = [](uint32_t address, uint32_t rd, bool interlocked)
            {
                Instruction inst{};
                inst.address = address;
                inst.opcode = OPCODE_COP2;
                inst.rs = COP2_QMTC2;
                inst.rt = rd + 1u;
                inst.rd = rd;
                inst.raw =
                    (OPCODE_COP2 << 26u) |
                    (COP2_QMTC2 << 21u) |
                    (inst.rt << 16u) |
                    (rd << 11u) |
                    (interlocked ? 1u : 0u);
                return inst;
            };

            CodeGenerator gen({}, {});
            const std::string generated = gen.generateFunction(
                function,
                {
                    makeQmtc2(0x3100u, 1u, false),
                    makeQmtc2(0x3104u, 2u, false),
                    makeQmtc2(0x3108u, 3u, true),
                },
                false);

            t.Equals(
                countOccurrences(
                    generated,
                    "synchronizeVU0Microprogram(rdram, ctx, false)"),
                static_cast<size_t>(2u),
                "non-interlocked transfers should remain tightly synchronized in an interlocked block");
            t.Equals(
                countOccurrences(
                    generated,
                    "synchronizeVU0Microprogram(rdram, ctx, true)"),
                static_cast<size_t>(1u),
                "the interlocked transfer should wait for VU0 completion");
        });

        tc.Run("generated EE instructions advance the scheduling counter", [](TestCase &t) {
            Function function;
            function.name = "instruction_count";
            function.start = 0x2000u;
            function.end = 0x200Cu;
            function.isRecompiled = true;

            CodeGenerator gen({}, {});
            const std::string generated = gen.generateFunction(
                function,
                {
                    makeNop(0x2000u),
                    makeNop(0x2004u),
                    makeNop(0x2008u),
                },
                false);

            t.Equals(
                countOccurrences(
                    generated,
                    "ctx->beginEeInstruction();"),
                static_cast<size_t>(3u),
                "each emitted non-branch EE instruction should retire");
            t.Equals(
                countOccurrences(generated, "ctx->advanceEeCycleTicks(9u);"),
                static_cast<size_t>(3u),
                "ordinary EE instructions should advance fixed-point issue time");
        });

        tc.Run("EE scheduling weights distinguish issue classes", [](TestCase &t) {
            Instruction ordinary{};
            ordinary.opcode = OPCODE_ADDIU;
            t.Equals(eeInstructionCycleTicks(ordinary), 9u,
                     "ordinary integer operation should use the default weight");

            Instruction branch{};
            branch.opcode = OPCODE_BEQ;
            t.Equals(eeInstructionCycleTicks(branch), 11u,
                     "conditional branch should include branch issue cost");

            Instruction load{};
            load.opcode = OPCODE_LW;
            t.Equals(eeInstructionCycleTicks(load), 14u,
                     "load should use the memory-operation weight");

            Instruction mmi{};
            mmi.opcode = OPCODE_MMI;
            mmi.function = MMI_MMI2;
            mmi.mmiFunction = MMI2_PMULTW;
            t.Equals(eeInstructionCycleTicks(mmi), 24u,
                     "packed multiply should use the MMI multiply weight");

            Instruction divide{};
            divide.opcode = OPCODE_SPECIAL;
            divide.function = SPECIAL_DIV;
            t.Equals(eeInstructionCycleTicks(divide), 112u,
                     "integer divide should retain its long issue latency");

            Instruction rsqrt{};
            rsqrt.opcode = OPCODE_COP1;
            rsqrt.rs = COP1_S;
            rsqrt.function = COP1_S_RSQRT;
            t.Equals(eeInstructionCycleTicks(rsqrt), 64u,
                     "FPU reciprocal square root should retain its issue latency");
        });

        tc.Run("branch delay slots retire and accrue time on their executed path", [](TestCase &t) {
            Function function;
            function.name = "branch_timing";
            function.start = 0x2100u;
            function.end = 0x2108u;
            function.isRecompiled = true;

            Instruction branch = makeBranch(0x2100u, 1u);
            Instruction delay = makeLw(0x2104u, 2u, 3u, 0u);

            CodeGenerator gen({}, {});
            const std::string generated =
                gen.generateFunction(function, {branch, delay}, false);

            t.IsTrue(
                generated.find("ctx->advanceEeCycleTicks(11u);") !=
                    std::string::npos,
                "branch instruction should use the branch weight");
            t.IsTrue(
                generated.find("ctx->advanceEeCycleTicks(14u);") !=
                    std::string::npos,
                "executed load delay slot should use the load weight");
            t.Equals(
                countOccurrences(
                    generated,
                    "ctx->beginEeInstruction();"),
                static_cast<size_t>(2u),
                "branch and delay slot should retire independently");
            t.Equals(
                countOccurrences(
                    generated, "ctx->finishEeBasicBlock();"),
                static_cast<size_t>(0u),
                "the context should not publish time independently of the runtime");
            t.Equals(
                countOccurrences(
                    generated,
                    "runtime->serviceEeEventsAtGeneratedBlockBoundary(rdram, ctx);"),
                static_cast<size_t>(2u),
                "each mutually exclusive branch path should commit time and service events");
        });

        tc.Run("post-delay boundaries validate the resolved branch PC", [](TestCase &t) {
            constexpr std::string_view boundary =
                "runtime->serviceEeEventsAtGeneratedBlockBoundary(rdram, ctx);";
            constexpr std::string_view validation =
                "runtime->validateEeInstructionFetchPolicy<Mode>(ctx, ctx->pc);";

            Function staticJumpFunction;
            staticJumpFunction.name = "post_delay_static_jump";
            staticJumpFunction.start = 0x2200u;
            staticJumpFunction.end = 0x2208u;
            staticJumpFunction.isRecompiled = true;

            const std::string staticJump = CodeGenerator({}, {}).generateFunction(
                staticJumpFunction,
                {makeJal(0x2200u, 0x3000u), makeNop(0x2204u)},
                false);
            const size_t staticTarget = staticJump.find(
                "ctx->pc = 0x3000u;");
            const size_t staticValidation = staticJump.find(
                validation, staticTarget);
            const size_t staticBoundary = staticJump.find(
                boundary, staticTarget);
            const size_t staticDispatch = staticJump.find(
                "dispatchValidatedGuestBranchPolicy<Mode>", staticTarget);
            t.IsTrue(
                staticTarget != std::string::npos &&
                    staticValidation != std::string::npos &&
                    staticBoundary != std::string::npos &&
                    staticDispatch != std::string::npos &&
                    staticTarget < staticValidation &&
                    staticValidation < staticBoundary &&
                    staticBoundary < staticDispatch,
                "a taken jump with a NOP slot should validate its target before servicing events");

            Function conditionalFunction;
            conditionalFunction.name = "post_delay_conditional";
            conditionalFunction.start = 0x2300u;
            conditionalFunction.end = 0x230Cu;
            conditionalFunction.isRecompiled = true;

            const std::string conditional = CodeGenerator({}, {}).generateFunction(
                conditionalFunction,
                {
                    makeBranch(0x2300u, 3u),
                    makeAddiu(0x2304u, 7u, 7u, 1u),
                    makeNop(0x2308u),
                },
                false);
            const size_t conditionalTarget = conditional.find(
                "ctx->pc = 0x2310u;");
            const size_t conditionalTargetValidation = conditional.find(
                validation, conditionalTarget);
            const size_t conditionalTargetBoundary = conditional.find(
                boundary, conditionalTarget);
            const size_t conditionalFallthrough = conditional.find(
                "ctx->pc = 0x2308u;", conditionalTarget);
            const size_t conditionalFallthroughValidation = conditional.find(
                validation, conditionalFallthrough);
            const size_t conditionalFallthroughBoundary = conditional.find(
                boundary, conditionalFallthrough);
            t.Equals(
                countOccurrences(conditional, std::string(boundary)),
                static_cast<size_t>(2u),
                "taken and not-taken paths should each service exactly one resolved boundary");
            t.Equals(
                countOccurrences(conditional, std::string(validation)),
                static_cast<size_t>(2u),
                "taken and not-taken paths should each validate the resolved fetch address");
            t.IsTrue(
                conditionalTarget != std::string::npos &&
                    conditionalTargetValidation != std::string::npos &&
                    conditionalTargetBoundary != std::string::npos &&
                    conditionalTarget < conditionalTargetValidation &&
                    conditionalTargetValidation < conditionalTargetBoundary,
                "a taken conditional with a real slot should validate its target before servicing events");
            t.IsTrue(
                conditionalFallthrough != std::string::npos &&
                    conditionalFallthroughValidation != std::string::npos &&
                    conditionalFallthroughBoundary != std::string::npos &&
                    conditionalFallthrough < conditionalFallthroughValidation &&
                    conditionalFallthroughValidation < conditionalFallthroughBoundary,
                "a not-taken conditional should validate its post-slot fallthrough before servicing events");

            Function likelyFunction;
            likelyFunction.name = "post_delay_likely";
            likelyFunction.start = 0x2400u;
            likelyFunction.end = 0x240Cu;
            likelyFunction.isRecompiled = true;

            Instruction likely{};
            likely.address = 0x2400u;
            likely.opcode = OPCODE_BEQL;
            likely.rs = 1u;
            likely.rt = 2u;
            likely.simmediate = 3u;
            likely.isBranch = true;
            likely.hasDelaySlot = true;
            const std::string likelyCode = CodeGenerator({}, {}).generateFunction(
                likelyFunction,
                {
                    likely,
                    makeAddiu(0x2404u, 8u, 8u, 1u),
                    makeNop(0x2408u),
                },
                false);
            const size_t likelyTakenStart = likelyCode.find(
                "if (branch_taken_0x2400)");
            const size_t likelyNotTakenStart = likelyCode.find(
                "if (!branch_taken_0x2400)", likelyTakenStart);
            const std::string likelyTakenPath =
                likelyTakenStart == std::string::npos ||
                        likelyNotTakenStart == std::string::npos
                    ? std::string{}
                    : likelyCode.substr(
                          likelyTakenStart,
                          likelyNotTakenStart - likelyTakenStart);
            const std::string likelyNotTakenPath =
                likelyNotTakenStart == std::string::npos
                    ? std::string{}
                    : likelyCode.substr(likelyNotTakenStart);
            const size_t likelyTarget = likelyTakenPath.find(
                "ctx->pc = 0x2410u;");
            const size_t likelyTargetValidation = likelyTakenPath.find(
                validation, likelyTarget);
            const size_t likelyTargetBoundary = likelyTakenPath.find(
                boundary, likelyTarget);
            const size_t likelyFallthrough = likelyNotTakenPath.find(
                "ctx->pc = 0x2408u;");
            const size_t likelyFallthroughValidation = likelyNotTakenPath.find(
                validation, likelyFallthrough);
            const size_t likelyFallthroughBoundary = likelyNotTakenPath.find(
                boundary, likelyFallthrough);
            t.IsTrue(
                likelyTarget != std::string::npos &&
                    likelyTargetValidation != std::string::npos &&
                    likelyTargetBoundary != std::string::npos &&
                    likelyTarget < likelyTargetValidation &&
                    likelyTargetValidation < likelyTargetBoundary,
                "a taken likely branch should validate its target after executing the slot");
            t.IsTrue(
                likelyFallthrough != std::string::npos &&
                    likelyFallthroughValidation != std::string::npos &&
                    likelyFallthroughBoundary != std::string::npos &&
                    likelyFallthrough < likelyFallthroughValidation &&
                    likelyFallthroughValidation < likelyFallthroughBoundary,
                "an annulled likely branch should validate its fallthrough before servicing events");

            Function registerJumpFunction;
            registerJumpFunction.name = "post_delay_register_jump";
            registerJumpFunction.start = 0x2500u;
            registerJumpFunction.end = 0x250Cu;
            registerJumpFunction.isRecompiled = true;

            const std::string registerJump = CodeGenerator({}, {}).generateFunction(
                registerJumpFunction,
                {
                    makeJr(0x2500u, 16u),
                    makeNop(0x2504u),
                    makeNop(0x2508u),
                },
                false);
            const size_t dynamicTarget = registerJump.find(
                "ctx->pc = jumpTarget;");
            const size_t dynamicValidation = registerJump.find(
                validation, dynamicTarget);
            const size_t dynamicBoundary = registerJump.find(
                boundary, dynamicTarget);
            t.IsTrue(
                dynamicTarget != std::string::npos &&
                    dynamicValidation != std::string::npos &&
                    dynamicBoundary != std::string::npos &&
                    dynamicTarget < dynamicValidation &&
                    dynamicValidation < dynamicBoundary,
                "a register jump should validate its captured target before servicing events");
        });

        tc.Run("VU0 macro writes preserve the hardwired zero registers", [](TestCase &t) {
            CodeGenerator gen({}, {});

            Instruction qmtc2{};
            qmtc2.opcode = OPCODE_COP2;
            qmtc2.rs = COP2_QMTC2;
            qmtc2.rt = 0;
            qmtc2.rd = 0;

            const std::string qmtc2Code = gen.translateInstruction(qmtc2);
            t.IsTrue(qmtc2Code.find("ctx->vu0_vf[0] = _mm_castsi128_ps(GPR_VEC(ctx, 0));") != std::string::npos,
                     "QMTC2 should still execute its architecturally discarded write");
            t.IsTrue(qmtc2Code.find("ctx->enforceVu0RegisterInvariants();") != std::string::npos,
                     "QMTC2 must restore VF0 and VI0 after a macro-mode write");

            Instruction vadd{};
            vadd.opcode = OPCODE_COP2;
            vadd.rs = COP2_CO;
            vadd.rt = 2;
            vadd.rd = 1;
            vadd.sa = 0;
            vadd.function = VU0_S1_VADD;
            vadd.vectorInfo.vectorField = 0xFu;

            const std::string vaddCode = gen.translateInstruction(vadd);
            t.IsTrue(vaddCode.find("ctx->vu0_vf[0]") != std::string::npos,
                     "the synthetic VADD should target VF0");
            t.IsTrue(vaddCode.find("ctx->enforceVu0RegisterInvariants();") != std::string::npos,
                     "arithmetic writes to VF0 must be discarded");

            Instruction lqc2{};
            lqc2.opcode = OPCODE_LQC2;
            lqc2.rs = 4;
            lqc2.rt = 0;
            lqc2.simmediate = 16;

            const std::string lqc2Code = gen.translateInstruction(lqc2);
            t.IsTrue(lqc2Code.find("READ128") != std::string::npos,
                     "LQC2 targeting VF0 must retain its memory access");
            t.IsTrue(lqc2Code.find("ctx->enforceVu0RegisterInvariants();") != std::string::npos,
                     "LQC2 targeting VF0 must discard the loaded value");
        });

        tc.Run("scalar logical immediates emit low64 operations", [](TestCase &t) {
            CodeGenerator gen({}, {});

            Instruction andi{};
            andi.opcode = OPCODE_ANDI;
            andi.rs = 4;
            andi.rt = 5;
            andi.immediate = 0xABCD;

            std::string andiCode = gen.translateInstruction(andi);
            t.IsTrue(andiCode.find("SET_GPR_U64(ctx, 5, GPR_U64(ctx, 4) & (uint64_t)(uint16_t)43981);") != std::string::npos,
                     "ANDI should use low64 scalar emission");
            t.IsTrue(andiCode.find("SET_GPR_VEC") == std::string::npos,
                     "ANDI should not use vector emission");

            Instruction ori{};
            ori.opcode = OPCODE_ORI;
            ori.rs = 6;
            ori.rt = 7;
            ori.immediate = 0x1234;

            std::string oriCode = gen.translateInstruction(ori);
            t.IsTrue(oriCode.find("SET_GPR_U64(ctx, 7, GPR_U64(ctx, 6) | (uint64_t)(uint16_t)4660);") != std::string::npos,
                     "ORI should use low64 scalar emission");
            t.IsTrue(oriCode.find("SET_GPR_VEC") == std::string::npos,
                     "ORI should not use vector emission");

            Instruction xori{};
            xori.opcode = OPCODE_XORI;
            xori.rs = 8;
            xori.rt = 9;
            xori.immediate = 0x00FF;

            std::string xoriCode = gen.translateInstruction(xori);
            t.IsTrue(xoriCode.find("SET_GPR_U64(ctx, 9, GPR_U64(ctx, 8) ^ (uint64_t)(uint16_t)255);") != std::string::npos,
                     "XORI should use low64 scalar emission");
            t.IsTrue(xoriCode.find("SET_GPR_VEC") == std::string::npos,
                     "XORI should not use vector emission");
        });

        tc.Run("scalar logical register ops emit low64 operations", [](TestCase &t) {
            CodeGenerator gen({}, {});

            Instruction andInst{};
            andInst.opcode = OPCODE_SPECIAL;
            andInst.function = SPECIAL_AND;
            andInst.rs = 2;
            andInst.rt = 3;
            andInst.rd = 1;

            std::string andCode = gen.translateInstruction(andInst);
            t.IsTrue(andCode.find("SET_GPR_U64(ctx, 1, GPR_U64(ctx, 2) & GPR_U64(ctx, 3));") != std::string::npos,
                     "AND should use low64 scalar emission");

            Instruction orInst{};
            orInst.opcode = OPCODE_SPECIAL;
            orInst.function = SPECIAL_OR;
            orInst.rs = 4;
            orInst.rt = 5;
            orInst.rd = 6;

            std::string orCode = gen.translateInstruction(orInst);
            t.IsTrue(orCode.find("SET_GPR_U64(ctx, 6, GPR_U64(ctx, 4) | GPR_U64(ctx, 5));") != std::string::npos,
                     "OR should use low64 scalar emission");

            Instruction xorInst{};
            xorInst.opcode = OPCODE_SPECIAL;
            xorInst.function = SPECIAL_XOR;
            xorInst.rs = 7;
            xorInst.rt = 8;
            xorInst.rd = 9;

            std::string xorCode = gen.translateInstruction(xorInst);
            t.IsTrue(xorCode.find("SET_GPR_U64(ctx, 9, GPR_U64(ctx, 7) ^ GPR_U64(ctx, 8));") != std::string::npos,
                     "XOR should use low64 scalar emission");

            Instruction norInst{};
            norInst.opcode = OPCODE_SPECIAL;
            norInst.function = SPECIAL_NOR;
            norInst.rs = 10;
            norInst.rt = 11;
            norInst.rd = 12;

            std::string norCode = gen.translateInstruction(norInst);
            t.IsTrue(norCode.find("SET_GPR_U64(ctx, 12, ~(GPR_U64(ctx, 10) | GPR_U64(ctx, 11)));") != std::string::npos,
                     "NOR should use low64 scalar emission");
            t.IsTrue(norCode.find("SET_GPR_VEC") == std::string::npos,
                     "SPECIAL logical ops should not use vector emission");
        });

        tc.Run("MOVZ and MOVN preserve the destination upper 64 bits", [](TestCase &t) {
            CodeGenerator gen({}, {});

            Instruction movz{};
            movz.opcode = OPCODE_SPECIAL;
            movz.function = SPECIAL_MOVZ;
            movz.rs = 4;
            movz.rt = 5;
            movz.rd = 6;

            const std::string movzCode = gen.translateInstruction(movz);
            t.Equals(movzCode,
                     std::string("if (GPR_U64(ctx, 5) == 0) SET_GPR_U64(ctx, 6, GPR_U64(ctx, 4));"),
                     "MOVZ should conditionally copy only the low 64 bits");
            t.IsTrue(movzCode.find("SET_GPR_VEC") == std::string::npos,
                     "MOVZ must not overwrite the destination upper 64 bits");

            Instruction movn{};
            movn.opcode = OPCODE_SPECIAL;
            movn.function = SPECIAL_MOVN;
            movn.rs = 7;
            movn.rt = 8;
            movn.rd = 9;

            const std::string movnCode = gen.translateInstruction(movn);
            t.Equals(movnCode,
                     std::string("if (GPR_U64(ctx, 8) != 0) SET_GPR_U64(ctx, 9, GPR_U64(ctx, 7));"),
                     "MOVN should conditionally copy only the low 64 bits");
            t.IsTrue(movnCode.find("SET_GPR_VEC") == std::string::npos,
                     "MOVN must not overwrite the destination upper 64 bits");
        });

        tc.Run("QFSRV translation uses runtime helper macro", [](TestCase &t) {
            CodeGenerator gen({}, {});

            Instruction qfsrv{};
            qfsrv.isMMI = true;
            qfsrv.opcode = OPCODE_MMI;
            qfsrv.function = MMI_MMI1;
            qfsrv.sa = MMI1_QFSRV;
            qfsrv.rd = 3;
            qfsrv.rs = 4;
            qfsrv.rt = 5;

            std::string out = gen.translateInstruction(qfsrv);
            t.IsTrue(out.find("PS2_QFSRV(GPR_VEC(ctx, 4), GPR_VEC(ctx, 5), ctx->sa & 0x7F)") != std::string::npos,
                     "QFSRV should map to PS2_QFSRV with rs/rt ordering");
        });

        tc.Run("PCPYLD translation uses runtime helper macro", [](TestCase &t) {
            CodeGenerator gen({}, {});

            Instruction pcpyld{};
            pcpyld.isMMI = true;
            pcpyld.opcode = OPCODE_MMI;
            pcpyld.function = MMI_MMI2;
            pcpyld.sa = MMI2_PCPYLD;
            pcpyld.rd = 6;
            pcpyld.rs = 7;
            pcpyld.rt = 8;

            std::string pcpyldOut = gen.translateInstruction(pcpyld);
            t.IsTrue(pcpyldOut.find("PS2_PCPYLD(GPR_VEC(ctx, 7), GPR_VEC(ctx, 8))") != std::string::npos,
                     "PCPYLD should use PS2_PCPYLD helper");
        });

        tc.Run("unary MMI shuffles read rt and use runtime lane helpers", [](TestCase &t) {
            CodeGenerator gen({}, {});

            const auto verify = [&](uint32_t group, uint32_t subfunction,
                                    const std::string &name) {
                Instruction inst{};
                inst.isMMI = true;
                inst.opcode = OPCODE_MMI;
                inst.function = group;
                inst.sa = subfunction;
                inst.rd = 9;
                inst.rs = 0;
                inst.rt = 10;

                const std::string out = gen.translateInstruction(inst);
                const std::string expected =
                    "PS2_" + name + "(GPR_VEC(ctx, 10))";
                t.IsTrue(out.find(expected) != std::string::npos,
                         name + " should read rt and use its runtime helper");
                t.IsTrue(out.find("GPR_VEC(ctx, 0)") == std::string::npos,
                         name + " must not read the unused rs field");
            };

            verify(MMI_MMI2, MMI2_PEXEH, "PEXEH");
            verify(MMI_MMI2, MMI2_PREVH, "PREVH");
            verify(MMI_MMI2, MMI2_PEXEW, "PEXEW");
            verify(MMI_MMI2, MMI2_PROT3W, "PROT3W");
            verify(MMI_MMI3, MMI3_PEXCH, "PEXCH");
            verify(MMI_MMI3, MMI3_PCPYH, "PCPYH");
            verify(MMI_MMI3, MMI3_PEXCW, "PEXCW");
        });

        tc.Run("VU0 macro mappings cover all S1/S2 enums", [](TestCase &t) {
            const std::vector<std::string> candidates = {
                (std::filesystem::path(__FILE__).parent_path() /
                 "../../ps2xRecomp/include/ps2recomp/instructions.h")
                    .lexically_normal()
                    .string(),
                "ps2xRecomp/include/ps2recomp/instructions.h",
                "../ps2xRecomp/include/ps2recomp/instructions.h",
                "../../ps2xRecomp/include/ps2recomp/instructions.h"
            };

            std::string text = readFileFromCandidates(candidates);
            t.IsTrue(!text.empty(), "instructions.h should be readable from the test working directory");

            std::vector<uint32_t> s1 = parseEnumValues(text, "VU0_S1_");
            std::vector<uint32_t> s2 = parseEnumValues(text, "VU0_S2_");
            t.IsTrue(!s1.empty(), "VU0_S1 enum list should not be empty");
            t.IsTrue(!s2.empty(), "VU0_S2 enum list should not be empty");

            CodeGenerator gen({}, {});

            for (uint32_t value : s1)
            {
                Instruction inst;
                inst.opcode = OPCODE_COP2;
                inst.rs = COP2_CO; // format
                inst.rt = 2;
                inst.rd = 3;
                inst.function = value;
                inst.vectorInfo.vectorField = 0xF;

                std::string out = gen.translateInstruction(inst);
                std::ostringstream msg;
                msg << "VU0 S1 0x" << std::hex << value << " should be mapped";
                t.IsTrue(out.find("Unhandled VU0 Special1") == std::string::npos, msg.str().c_str());
            }

            for (uint32_t value : s2)
            {
                Instruction inst;
                inst.opcode = OPCODE_COP2;
                inst.rs = COP2_CO; // format
                inst.rt = 2;
                inst.rd = 3;
                inst.function = 0x3C; // force Special2 path
                inst.vectorInfo.vectorField = 0xF;

                uint32_t upper = (value >> 2) & 0x1F;
                uint32_t lower = value & 0x3;
                inst.raw = (upper << 6) | lower;

                std::string out = gen.translateInstruction(inst);
                std::ostringstream msg;
                msg << "VU0 S2 0x" << std::hex << value << " should be mapped";
                t.IsTrue(out.find("Unhandled VU0 Special2") == std::string::npos, msg.str().c_str());
            }
        });

        tc.Run("VU0 S1 uses fd/fs/ft fields (sa/rd/rt)", [](TestCase &t) {
            Instruction inst{};
            inst.opcode = OPCODE_COP2;
            inst.rs = COP2_CO | 0xB; // format + destination mask bits, not a VF register index
            inst.rt = 7;
            inst.rd = 11;
            inst.sa = 3;
            inst.function = VU0_S1_VADD;
            inst.vectorInfo.vectorField = 0xF;

            CodeGenerator gen({}, {});
            std::string out = gen.translateInstruction(inst);

            t.IsTrue(out.find("ctx->vu0_vf[11]") != std::string::npos, "S1 fs should come from rd");
            t.IsTrue(out.find("ctx->vu0_vf[7]") != std::string::npos, "S1 ft should come from rt");
            t.IsTrue(out.find("ctx->vu0_vf[3]") != std::string::npos, "S1 fd should come from sa");
            t.IsTrue(out.find("ctx->vu0_vf[27]") == std::string::npos, "S1 must not use rs(format) as register index");
        });

        tc.Run("VU0 S1 q/i forms keep mask and use sa as destination", [](TestCase &t) {
            Instruction inst{};
            inst.opcode = OPCODE_COP2;
            inst.rs = COP2_CO | 0x9; // format + destination mask bits
            inst.rt = 5;
            inst.rd = 13;
            inst.sa = 4;
            inst.function = VU0_S1_VADDq;
            inst.vectorInfo.vectorField = 0x9;

            CodeGenerator gen({}, {});
            std::string out = gen.translateInstruction(inst);

            t.IsTrue(out.find("_mm_blendv_ps") != std::string::npos, "S1 q/i form should honor destination mask");
            t.IsTrue(out.find("ctx->vu0_vf[13]") != std::string::npos, "S1 q/i source should come from rd");
            t.IsTrue(out.find("ctx->vu0_vf[4]") != std::string::npos, "S1 q/i destination should come from sa");
            t.IsTrue(out.find("ctx->vu0_vf[25]") == std::string::npos, "S1 q/i must not use rs(format) as register index");
        });

        tc.Run("VU0 S2 vector ops use rd as source and rt as destination", [](TestCase &t) {
            Instruction inst{};
            inst.opcode = OPCODE_COP2;
            inst.rs = COP2_CO | 0x6; // format + destination mask bits
            inst.rt = 8;
            inst.rd = 12;
            inst.function = 0x3C; // force Special2 path
            inst.vectorInfo.vectorField = 0xF;

            uint32_t upper = (VU0_S2_VABS >> 2) & 0x1F;
            uint32_t lower = VU0_S2_VABS & 0x3;
            inst.raw = (upper << 6) | lower;

            CodeGenerator gen({}, {});
            std::string out = gen.translateInstruction(inst);

            t.IsTrue(out.find("ctx->vu0_vf[12]") != std::string::npos, "S2 source VF should come from rd");
            t.IsTrue(out.find("ctx->vu0_vf[8]") != std::string::npos, "S2 destination VF should come from rt");
            t.IsTrue(out.find("ctx->vu0_vf[22]") == std::string::npos, "S2 must not use rs(format) as register index");
        });

        tc.Run("VU0 VCLIP uses architectural sign-magnitude comparisons", [](TestCase &t) {
            Instruction inst{};
            inst.opcode = OPCODE_COP2;
            inst.rs = COP2_CO | 0xEu;
            inst.rt = 8;
            inst.rd = 12;
            inst.function = 0x3C; // force Special2 path

            const uint32_t upper = (VU0_S2_VCLIPw >> 2) & 0x1Fu;
            const uint32_t lower = VU0_S2_VCLIPw & 0x3u;
            inst.raw = (upper << 6) | lower;

            CodeGenerator gen({}, {});
            const std::string out = gen.translateInstruction(inst);

            t.IsTrue(out.find("Ps2VuUpdateClipFlags") != std::string::npos,
                     "VCLIP should use the shared bit-exact helper");
            t.IsTrue(out.find("ctx->vu0_vf[12]") != std::string::npos,
                     "VCLIP fs should come from rd");
            t.IsTrue(out.find("ctx->vu0_vf[8]") != std::string::npos,
                     "VCLIP ft should come from rt");
            t.IsTrue(out.find("_mm_cmpgt_ps") == std::string::npos,
                     "VCLIP must not use host IEEE floating-point comparisons");
        });

        tc.Run("VU0 S2 vector loads use rd as VI base register", [](TestCase &t) {
            Instruction inst{};
            inst.opcode = OPCODE_COP2;
            inst.rs = COP2_CO | 0x4; // format + destination mask bits
            inst.rt = 6;
            inst.rd = 14;
            inst.function = 0x3C; // force Special2 path
            inst.vectorInfo.vectorField = 0xF;

            uint32_t upper = (VU0_S2_VLQI >> 2) & 0x1F;
            uint32_t lower = VU0_S2_VLQI & 0x3;
            inst.raw = (upper << 6) | lower;

            CodeGenerator gen({}, {});
            std::string out = gen.translateInstruction(inst);

            t.IsTrue(out.find("ctx->vi[14]") != std::string::npos, "S2 VLQI base VI should come from rd");
            t.IsTrue(out.find("ctx->vu0_vf[6]") != std::string::npos, "S2 VLQI destination VF should come from rt");
            t.IsTrue(out.find("ctx->vi[20]") == std::string::npos, "S2 VLQI must not use rs(format) as VI index");
            t.IsTrue(out.find("PS2_VU0_DATA_BASE") != std::string::npos,
                     "S2 VLQI should access VU0 data memory rather than low EE RDRAM");
            t.IsTrue(out.find("PS2_VU0_DATA_SIZE - 1u") != std::string::npos,
                     "S2 VLQI should wrap addresses to the 4 KiB VU0 data memory");
            t.IsTrue(out.find("& 0x3FF") == std::string::npos,
                     "S2 VLQI should retain the architectural 16-bit VI value after increment");
        });

        tc.Run("VU0 S2 vector stores use rt as VI base and rd as VF source", [](TestCase &t) {
            Instruction inst{};
            inst.opcode = OPCODE_COP2;
            inst.rs = COP2_CO | 0x4; // format + destination mask bits
            inst.rt = 6;
            inst.rd = 14;
            inst.function = 0x3C; // force Special2 path
            inst.vectorInfo.vectorField = 0xF;

            CodeGenerator gen({}, {});

            uint32_t upper = (VU0_S2_VSQI >> 2) & 0x1F;
            uint32_t lower = VU0_S2_VSQI & 0x3;
            inst.raw = (upper << 6) | lower;
            const std::string sqi = gen.translateInstruction(inst);

            t.IsTrue(sqi.find("ctx->vi[6]") != std::string::npos,
                     "S2 VSQI base VI should come from rt");
            t.IsTrue(sqi.find("ctx->vu0_vf[14]") != std::string::npos,
                     "S2 VSQI source VF should come from rd");
            t.IsTrue(sqi.find("ctx->vi[14]") == std::string::npos,
                     "S2 VSQI must not use the VF source as a VI index");
            t.IsTrue(sqi.find("ctx->vu0_vf[6]") == std::string::npos,
                     "S2 VSQI must not use the VI base as a VF source");
            t.IsTrue(sqi.find("PS2_VU0_DATA_BASE") != std::string::npos,
                     "S2 VSQI should access VU0 data memory rather than low EE RDRAM");
            t.IsTrue(sqi.find("PS2_VU0_DATA_SIZE - 1u") != std::string::npos,
                     "S2 VSQI should wrap addresses to the 4 KiB VU0 data memory");
            t.IsTrue(sqi.find("& 0x3FF") == std::string::npos,
                     "S2 VSQI should retain the architectural 16-bit VI value after increment");

            upper = (VU0_S2_VSQD >> 2) & 0x1F;
            lower = VU0_S2_VSQD & 0x3;
            inst.raw = (upper << 6) | lower;
            const std::string sqd = gen.translateInstruction(inst);

            t.IsTrue(sqd.find("ctx->vi[6]") != std::string::npos,
                     "S2 VSQD base VI should come from rt");
            t.IsTrue(sqd.find("ctx->vu0_vf[14]") != std::string::npos,
                     "S2 VSQD source VF should come from rd");
            t.IsTrue(sqd.find("ctx->vi[14]") == std::string::npos,
                     "S2 VSQD must not use the VF source as a VI index");
            t.IsTrue(sqd.find("ctx->vu0_vf[6]") == std::string::npos,
                     "S2 VSQD must not use the VI base as a VF source");
            t.IsTrue(sqd.find("PS2_VU0_DATA_BASE") != std::string::npos,
                     "S2 VSQD should access VU0 data memory rather than low EE RDRAM");
            t.IsTrue(sqd.find("PS2_VU0_DATA_SIZE - 1u") != std::string::npos,
                     "S2 VSQD should wrap addresses to the 4 KiB VU0 data memory");
            t.IsTrue(sqd.find("& 0x3FF") == std::string::npos,
                     "S2 VSQD should retain the architectural 16-bit VI value after decrement");
        });

        tc.Run("JAL to known function emits call and check", [](TestCase &t) {
            Function func;
            func.name = "jal_test";
            func.start = 0xA000;
            func.end = 0xA020;
            func.isRecompiled = true;
            func.isStub = false;

            Symbol targetSym;
            targetSym.name = "some_func";
            targetSym.address = 0xB000;
            targetSym.isFunction = true;

            // 0xA000: JAL 0xB000
            // 0xA004: NOP (delay slot)
            Instruction jal = makeJal(0xA000, 0xB000);
            Instruction delay = makeNop(0xA004);
            
            CodeGenerator gen({targetSym}, {});
            std::string generated = gen.generateFunction(func, {jal, delay}, false);
            printGeneratedCode("JAL to known function emits call and check", generated);

            // Expect:
            // SET_GPR_U32(ctx, 31, 0xA008u);
            // ctx->pc = 0xA004u;
            // ... delay slot ...
            // runtime->dispatchValidatedGuestBranchPolicy<Mode>(..., DirectCall, "JAL")

            t.IsTrue(generated.find("SET_GPR_U32(ctx, 31, 0xA008u);") != std::string::npos, "JAL should set RA");
            t.IsTrue(generated.find("runtime->dispatchValidatedGuestBranchPolicy<Mode>(rdram, ctx, 0xB000u") != std::string::npos,
                     "JAL should dispatch through the runtime branch helper");
            t.IsTrue(generated.find("PS2Runtime::GuestBranchKind::DirectCall") != std::string::npos,
                     "JAL should identify itself as a direct call");
            t.IsTrue(generated.find("0xA000u, 0xA008u") != std::string::npos,
                     "JAL should pass call-site and fallthrough PCs to the runtime helper");
        });

        tc.Run("trailing JAL without decoded delay slot still emits call flow", [](TestCase &t) {
            Function func;
            func.name = "jal_truncated";
            func.start = 0xA100;
            func.end = 0xA108;
            func.isRecompiled = true;
            func.isStub = false;

            Symbol targetSym;
            targetSym.name = "some_func";
            targetSym.address = 0xB000;
            targetSym.isFunction = true;

            Instruction jal = makeJal(0xA100, 0xB000);

            CodeGenerator gen({targetSym}, {});
            std::string generated = gen.generateFunction(func, {jal}, false);
            printGeneratedCode("trailing JAL without decoded delay slot still emits call flow", generated);

            t.IsTrue(generated.find("SET_GPR_U32(ctx, 31, 0xA108u);") != std::string::npos,
                     "truncated trailing JAL should still set RA");
            t.IsTrue(generated.find("runtime->dispatchValidatedGuestBranchPolicy<Mode>(rdram, ctx, 0xB000u") != std::string::npos,
                     "truncated trailing JAL should still emit the call dispatch");
            t.IsTrue(generated.find("0xA100u, 0xA108u") != std::string::npos,
                     "truncated trailing JAL should still pass the fallthrough to runtime dispatch");
            t.IsTrue(generated.find("// JAL 0xB000 - Handled by branch logic") == std::string::npos,
                     "truncated trailing JAL must not degrade to comment-only output");
        });

        tc.Run("JAL to internal target becomes goto", [](TestCase &t) {
            Function func;
            func.name = "jal_internal";
            func.start = 0xC000;
            func.end = 0xC020;
            func.isRecompiled = true;
            func.isStub = false;

            // 0xC000: JAL 0xC010
            // 0xC004: NOP
            // ...
            // 0xC010: NOP
            Instruction jal = makeJal(0xC000, 0xC010);
            Instruction delay = makeNop(0xC004);
            Instruction targetInst = makeNop(0xC010);

            CodeGenerator gen({}, {});
            std::string generated = gen.generateFunction(func, {jal, delay, targetInst}, false);
            printGeneratedCode("JAL to internal target becomes goto", generated);

            t.IsTrue(generated.find("SET_GPR_U32(ctx, 31, 0xC008u);") != std::string::npos, "Internal JAL should set RA");
            t.IsTrue(generated.find("goto label_c010;") != std::string::npos, "Internal JAL should use goto");
        });

        tc.Run("backward internal JAL stays inline and does not return to dispatcher", [](TestCase &t) {
            Function func;
            func.name = "jal_internal_backward";
            func.start = 0xC100;
            func.end = 0xC120;
            func.isRecompiled = true;
            func.isStub = false;

            Instruction loopHead = makeNop(0xC100);
            Instruction backwardJal = makeJal(0xC108, 0xC100);
            Instruction delay = makeNop(0xC10C);

            CodeGenerator gen({}, {});
            std::string generated = gen.generateFunction(func, {loopHead, backwardJal, delay}, false);
            printGeneratedCode("backward internal JAL stays inline and does not return to dispatcher", generated);

            t.IsTrue(generated.find("SET_GPR_U32(ctx, 31, 0xC110u);") != std::string::npos,
                     "backward internal JAL should still set RA");
            t.IsTrue(generated.find("goto label_c100;") != std::string::npos,
                     "backward internal JAL should still re-enter the internal target directly");
            t.IsTrue(generated.find("runtime->checkpointGuestExecution(ctx)") == std::string::npos,
                     "backward internal JAL should not expose a mid-call dispatcher return");
        });

        tc.Run("JALR emits indirect call", [](TestCase &t) {
            Function func;
            func.name = "jalr_test";
            func.start = 0xD000;
            func.end = 0xD020;
            func.isRecompiled = true;
            func.isStub = false;

            // 0xD000: JALR $4, $31 (call addr in $4, link to $31)
            // 0xD004: NOP
            Instruction jalr = makeJalr(0xD000, 4, 31);
            Instruction delay = makeNop(0xD004);

            CodeGenerator gen({}, {});
            std::string generated = gen.generateFunction(func, {jalr, delay}, false);
            printGeneratedCode("JALR emits indirect call", generated);

            t.IsTrue(generated.find("const uint32_t jumpTarget = GPR_U32(ctx, 4);") != std::string::npos, "JALR should read target from RS");
            t.IsTrue(generated.find("SET_GPR_U32(ctx, 31, 0xD008u);") != std::string::npos, "JALR should set link register");
            t.IsTrue(generated.find("runtime->dispatchValidatedGuestBranchPolicy<Mode>(rdram, ctx, jumpTarget") != std::string::npos, "JALR should dispatch through runtime branch helper");
            t.IsTrue(generated.find("PS2Runtime::GuestBranchKind::IndirectCall") != std::string::npos,
                     "JALR should identify itself as an indirect call");
            t.IsTrue(generated.find("0xD000u, 0xD008u") != std::string::npos,
                     "JALR should pass call-site and fallthrough PCs to the runtime helper");
            t.IsFalse(generated.find("jumpTarget == 0u") != std::string::npos,
                      "JALR must not silently turn target 0 into a successful call return");
        }); 

        tc.Run("backward BEQ returns to the dispatcher through a resumable loop head", [](TestCase &t) {
            Function func;
            func.name = "backward_branch";
            func.start = 0x1100;
            func.end = 0x1120;
            func.isRecompiled = true;
            func.isStub = false;

            // 0x1100: nop
            // 0x1104: nop (loop head)
            // 0x1108: beq $1,$1, target 0x1104 (offset = -2 words)
            // 0x110c: nop (delay)
            std::vector<Instruction> instructions;
            instructions.push_back(makeNop(0x1100));
            instructions.push_back(makeNop(0x1104));

            Instruction br = makeBranch(0x1108, 0);
            br.simmediate = static_cast<uint32_t>(static_cast<int16_t>(-2));
            instructions.push_back(br);

            instructions.push_back(makeNop(0x110c));
            instructions.push_back(makeNop(0x1110));

            CodeGenerator gen({}, {});
            CodeGenerator::AnalysisResult analysis = gen.collectInternalBranchTargets(func, instructions);
            t.IsTrue(analysis.resumeEntryPoints.contains(0x1104u),
                     "mid-function backward loop head should be resumable so preemption can return to the dispatcher");

            std::string generated = gen.generateFunction(func, instructions, false);
            printGeneratedCode("backward BEQ returns to the dispatcher through a resumable loop head", generated);

            t.IsTrue(generated.find("label_1104:") != std::string::npos, "target should emit a label");
            t.IsTrue(generated.find("switch (ctx->pc)") != std::string::npos,
                     "resumable backward loop heads should emit a wrapper resume switch");
            t.IsTrue(generated.find("case 0x1104u: goto label_1104;") != std::string::npos,
                     "mid-function backward loop head should be re-enterable after returning to the dispatcher");
            t.IsTrue(generated.find("ctx->pc = 0x1104u;") != std::string::npos,
                     "backward internal branch should preserve the loop target in ctx->pc");
            t.IsTrue(
                generated.find(
                    "runtime->checkpointGuestExecution(ctx) == "
                    "PS2GuestCheckpointResult::ExitToDispatcher") !=
                    std::string::npos,
                "backward internal branch should reach the backend-neutral checkpoint before re-entering the loop");
            t.IsTrue(generated.find("return;") != std::string::npos,
                     "backward internal branch should return to the dispatcher when the runtime preemption policy requests it");
            t.IsTrue(generated.find("runtime->cooperativeGuestYield();") == std::string::npos,
                     "backward internal branch should no longer emit an unconditional cooperative-yield call");
            t.IsTrue(generated.find("goto label_1104;") != std::string::npos,
                     "backward internal branch should still re-enter the in-function label when it keeps the current slice");
        });

        tc.Run("branch-likely places delay slot only in taken path", [](TestCase &t) {
            Function func;
            func.name = "branch_likely";
            func.start = 0x1200;
            func.end = 0x1220;
            func.isRecompiled = true;
            func.isStub = false;

            Instruction br{};
            br.address = 0x1200;
            br.opcode = OPCODE_BEQL;   // likely
            br.rs = 1;
            br.rt = 2;
            br.simmediate = 1;         // target = 0x1208
            br.isBranch = true;
            br.hasDelaySlot = true;
            br.raw = 0;  

            Instruction delay{};
            delay.address = 0x1204;
            delay.opcode = OPCODE_ADDIU;
            delay.rs = 0;
            delay.rt = 7;              // make it non-nop so translation is distinctive
            delay.simmediate = 123;
            delay.raw = 0;

            Instruction target = makeNop(0x1208);

            CodeGenerator gen({}, {});
            std::string generated = gen.generateFunction(func, { br, delay, target }, false);
            printGeneratedCode("branch-likely places delay slot only in taken path", generated);
 
            t.IsTrue(generated.find("SET_GPR_S32(ctx, 7,") != std::string::npos, "delay slot should be translated");
            t.IsTrue(generated.find("if (branch_taken_0x1200)") != std::string::npos, "should generate branch_taken variable and if for likely branch");
        });

        tc.Run("BC0 branches use CPCOND0 and annul likely delay slots", [](TestCase &t) {
            struct BranchCase
            {
                uint8_t condition;
                bool branchOnTrue;
                bool likely;
            };
            const BranchCase cases[] = {
                {COP0_BC_BCF, false, false},
                {COP0_BC_BCT, true, false},
                {COP0_BC_BCFL, false, true},
                {COP0_BC_BCTL, true, true},
            };

            for (const BranchCase &branchCase : cases)
            {
                Function function;
                function.name = "bc0_condition";
                function.start = 0x1240u;
                function.end = 0x124Cu;
                function.isRecompiled = true;

                const Instruction branch = makeCop0Branch(
                    0x1240u, branchCase.condition, 3);
                const Instruction delay = makeAddiu(
                    0x1244u, 7u, 0u, 123u);
                const Instruction fallthrough = makeAddiu(
                    0x1248u, 8u, 0u, 77u);

                CodeGenerator generator({}, {});
                const std::string generated = generator.generateFunction(
                    function, {branch, delay, fallthrough}, false);
                const std::string expectedCondition =
                    branchCase.branchOnTrue
                        ? "const bool branch_taken_0x1240 = (runtime->readCop0Condition0());"
                        : "const bool branch_taken_0x1240 = (!runtime->readCop0Condition0());";
                t.IsTrue(
                    generated.find(expectedCondition) != std::string::npos,
                    "BC0 polarity should be derived from the live CPCOND0 signal");
                t.IsTrue(
                    generated.find("ctx->pc = 0x1250u;") !=
                        std::string::npos,
                    "a taken BC0 should dispatch to its external target");
                t.IsTrue(
                    generated.find("SET_GPR_S32(ctx, 8,") !=
                        std::string::npos,
                    "a not-taken BC0 should retain its decoded fallthrough");

                const size_t branchBody = generated.find(
                    "if (branch_taken_0x1240)");
                const size_t delaySlot = generated.find(
                    "SET_GPR_S32(ctx, 7,");
                t.IsTrue(
                    branchBody != std::string::npos &&
                        delaySlot != std::string::npos,
                    "BC0 fixture should contain a branch body and translated delay slot");
                if (branchBody != std::string::npos &&
                    delaySlot != std::string::npos)
                {
                    t.Equals(
                        branchBody < delaySlot,
                        branchCase.likely,
                        "only BC0 likely forms should place the delay slot inside the taken path");
                }
            }
        });

        tc.Run("BC2 branches use VU1 running state instead of VU0 flags", [](TestCase &t) {
            struct BranchCase
            {
                uint8_t condition;
                bool branchOnTrue;
                bool likely;
            };
            const BranchCase cases[] = {
                {COP2_BC_BCF, false, false},
                {COP2_BC_BCT, true, false},
                {COP2_BC_BCFL, false, true},
                {COP2_BC_BCTL, true, true},
            };

            for (const BranchCase &branchCase : cases)
            {
                Function function;
                function.name = "bc2_condition";
                function.start = 0x1280u;
                function.end = 0x128Cu;
                function.isRecompiled = true;

                const Instruction branch = makeCop2Branch(
                    0x1280u, branchCase.condition, 3);
                const Instruction delay = makeAddiu(
                    0x1284u, 9u, 0u, 123u);
                const Instruction fallthrough = makeAddiu(
                    0x1288u, 10u, 0u, 77u);

                CodeGenerator generator({}, {});
                const std::string generated = generator.generateFunction(
                    function, {branch, delay, fallthrough}, false);
                const std::string expectedCondition =
                    branchCase.branchOnTrue
                        ? "const bool branch_taken_0x1280 = (runtime->readCop2Condition(ctx));"
                        : "const bool branch_taken_0x1280 = (!runtime->readCop2Condition(ctx));";
                t.IsTrue(
                    generated.find(expectedCondition) != std::string::npos,
                    "BC2 polarity should be derived from VU1 running state");
                t.IsFalse(
                    generated.find("vu0_status") != std::string::npos,
                    "BC2 must not depend on VU0 arithmetic flags");
                t.IsTrue(
                    generated.find("ctx->pc = 0x1290u;") !=
                        std::string::npos,
                    "a taken BC2 should dispatch to its external target");
                t.IsTrue(
                    generated.find("SET_GPR_S32(ctx, 10,") !=
                        std::string::npos,
                    "a not-taken BC2 should retain its decoded fallthrough");

                const size_t branchBody = generated.find(
                    "if (branch_taken_0x1280)");
                const size_t delaySlot = generated.find(
                    "SET_GPR_S32(ctx, 9,");
                t.IsTrue(
                    branchBody != std::string::npos &&
                        delaySlot != std::string::npos,
                    "BC2 fixture should contain a branch body and translated delay slot");
                if (branchBody != std::string::npos &&
                    delaySlot != std::string::npos)
                {
                    t.Equals(
                        branchBody < delaySlot,
                        branchCase.likely,
                        "only BC2 likely forms should place the delay slot inside the taken path");
                }
            }
        });

        tc.Run("JR $31 returns through dynamic target without broad local switch", [](TestCase &t) {
            Function func;
            func.name = "jr_ra_return";
            func.start = 0x1300;
            func.end = 0x1340;
            func.isRecompiled = true;
            func.isStub = false;

            // Create an internal JAL with an explicit instruction at returnAddr (0x1308).
            Instruction jal = makeJal(0x1300, 0x1310);
            Instruction jalDelay = makeNop(0x1304);
            Instruction atReturn = makeNop(0x1308);
            Instruction atTarget = makeNop(0x1310);

            // JR $31 at 0x1314 with delay slot at 0x1318
            Instruction jr = makeJr(0x1314, 31);
            Instruction jrDelay = makeNop(0x1318);

            CodeGenerator gen({}, {});
            std::string generated = gen.generateFunction(func, { jal, jalDelay, atReturn, atTarget, jr, jrDelay }, false);
            printGeneratedCode("JR $31 returns through dynamic target without broad local switch", generated);

            t.IsTrue(generated.find("const uint32_t jumpTarget = GPR_U32(ctx, 31);") != std::string::npos,
                     "JR $31 should still read the dynamic return target");
            t.IsFalse(generated.find("switch (jumpTarget)") != std::string::npos,
                      "JR $31 should not emit a broad local switch over internal labels");
            t.IsTrue(generated.find("label_1308:") != std::string::npos,
                     "internal JAL return address should still be emitted as a label");
            const size_t targetValidation = generated.find(
                "runtime->validateEeGuestBranchTargetPolicy<Mode>("
                "ctx, jumpTarget, PS2Runtime::GuestBranchKind::Return)");
            const size_t callbackBoundary = generated.find(
                "runtime->serviceEeEventsAtGeneratedBlockBoundary(rdram, ctx);",
                targetValidation);
            const size_t callbackUnwind = generated.find(
                "if (!continueGuestExecution) { return; }",
                callbackBoundary);
            t.IsTrue(
                targetValidation != std::string::npos &&
                    callbackBoundary != std::string::npos &&
                    callbackUnwind != std::string::npos &&
                    targetValidation < callbackBoundary &&
                    callbackBoundary < callbackUnwind,
                "JR $31 should recognize the host callback boundary before fetch validation while still servicing retired work");
            t.IsTrue(generated.find("        return;") != std::string::npos,
                     "JR $31 should return to the dispatcher/runtime after setting ctx->pc");
            t.IsTrue(generated.find("PS2Runtime::GuestBranchKind::Return") != std::string::npos,
                     "JR $31 should use the Return branch kind for precise diagnostics");
            t.IsTrue(generated.find("\"JR $ra\"") != std::string::npos,
                     "JR $31 should pass a return-specific debug name");
        });

        tc.Run("trailing JR $31 without decoded delay slot still emits return flow", [](TestCase &t) {
            Function func;
            func.name = "jr_ra_truncated";
            func.start = 0x1500;
            func.end = 0x1540;
            func.isRecompiled = true;
            func.isStub = false;

            Instruction jal = makeJal(0x1500, 0x1510);
            Instruction jalDelay = makeNop(0x1504);
            Instruction atReturn = makeNop(0x1508);
            Instruction atTarget = makeNop(0x1510);
            Instruction jr = makeJr(0x1514, 31);

            CodeGenerator gen({}, {});
            std::string generated = gen.generateFunction(func, {jal, jalDelay, atReturn, atTarget, jr}, false);
            printGeneratedCode("trailing JR $31 without decoded delay slot still emits return flow", generated);

            t.IsTrue(generated.find("const uint32_t jumpTarget = GPR_U32(ctx, 31);") != std::string::npos,
                     "truncated trailing JR should still read the return target");
            t.IsFalse(generated.find("switch (jumpTarget)") != std::string::npos,
                      "truncated trailing JR should not emit a broad local return-target switch");
            t.IsTrue(generated.find("label_1508:") != std::string::npos,
                     "truncated trailing JR should still include the internal return label");
            t.IsTrue(generated.find("// JR $31 - Handled by branch logic") == std::string::npos,
                     "truncated trailing JR must not degrade to comment-only output");
            t.IsTrue(generated.find("PS2Runtime::GuestBranchKind::Return") != std::string::npos,
                     "truncated JR $31 should still use return diagnostics");
        });

        tc.Run("JR through a saved return-address alias emits return flow", [](TestCase &t) {
            Function func;
            func.name = "jr_saved_ra_return";
            func.start = 0x1580u;
            func.end = 0x15A0u;
            func.isRecompiled = true;
            func.isStub = false;

            const Instruction saveRa = decodeRawInstruction(
                0x1580u,
                (OPCODE_ADDI << 26) | (31u << 21) | (15u << 16));
            const Instruction jal = makeJal(0x1584u, 0x2000u);
            const Instruction jalDelay = makeNop(0x1588u);
            const Instruction jr = makeJr(0x158Cu, 15u);
            const Instruction jrDelay = makeNop(0x1590u);

            CodeGenerator gen({}, {});
            const std::string generated = gen.generateFunction(
                func, {saveRa, jal, jalDelay, jr, jrDelay}, false);
            printGeneratedCode(
                "JR through a saved return-address alias emits return flow",
                generated);

            t.IsTrue(generated.find(
                         "const uint32_t jumpTarget = GPR_U32(ctx, 15);") !=
                         std::string::npos,
                     "saved-$ra return should read the actual alias register");
            t.IsTrue(generated.find(
                         "PS2Runtime::GuestBranchKind::Return") !=
                         std::string::npos,
                     "saved-$ra JR should use return validation and dispatch semantics");
            t.IsFalse(generated.find(
                          "PS2Runtime::GuestBranchKind::IndirectJump") !=
                          std::string::npos,
                      "saved-$ra JR should not be emitted as an arbitrary computed jump");
            t.IsTrue(generated.find("\"JR saved $ra\"") !=
                         std::string::npos,
                     "saved-$ra returns should retain a precise diagnostic name");
        });

        tc.Run("unresolved JR non-RA uses dispatcher resume entries without broad local switch", [](TestCase &t) {
            Function func;
            func.name = "jr_non_ra_dispatcher_resume";
            func.start = 0x1400;
            func.end = 0x1420;
            func.isRecompiled = true;
            func.isStub = false;

            // 0x1400: nop
            // 0x1404: jr $16 (register jump)
            // 0x1408: nop (delay slot)
            // 0x140c: nop
            Instruction i0 = makeNop(0x1400);
            Instruction jr = makeJr(0x1404, 16);
            Instruction delay = makeNop(0x1408);
            Instruction i3 = makeNop(0x140c);

            CodeGenerator gen({}, {});
            CodeGenerator::AnalysisResult analysis = gen.collectInternalBranchTargets(func, {i0, jr, delay, i3});
            gen.setResumeEntryTargets({{func.start, std::vector<uint32_t>(
                                                        analysis.indirectFallbackEntryPoints.begin(),
                                                        analysis.indirectFallbackEntryPoints.end())}});
            std::string generated = gen.generateFunction(func, {i0, jr, delay, i3}, false);
            printGeneratedCode("unresolved JR non-RA uses dispatcher resume entries without broad local switch", generated);

            t.IsFalse(generated.find("switch (jumpTarget)") != std::string::npos,
                      "unresolved JR via non-RA register should not emit a broad local switch over internal labels");
            t.IsTrue(generated.find("switch (ctx->pc)") != std::string::npos,
                     "JR fallback labels should be promoted to dispatcher resume sites");
            t.IsTrue(generated.find("case 0x140cu: goto label_140c;") != std::string::npos,
                     "owner resume switch should include internal fallback labels");
            t.IsTrue(generated.find("ctx->pc = jumpTarget;") != std::string::npos,
                     "unresolved JR should hand the dynamic target back through ctx->pc");
            t.IsTrue(generated.find("PS2Runtime::GuestBranchKind::IndirectJump") != std::string::npos,
                     "unresolved non-RA JR should use indirect-jump diagnostics");
        });

        tc.Run("configured jump table addresses drive JR dispatch targets", [](TestCase &t) {
            Function func;
            func.name = "jr_configured_jump_table";
            func.start = 0x1600;
            func.end = 0x1640;
            func.isRecompiled = true;
            func.isStub = false;

            constexpr uint32_t tableAddress = 0x00200000u;

            Instruction lui{};
            lui.address = 0x1600;
            lui.opcode = OPCODE_LUI;
            lui.rt = 9;
            lui.immediate = static_cast<uint16_t>((tableAddress >> 16) & 0xFFFFu);

            Instruction addiu{};
            addiu.address = 0x1604;
            addiu.opcode = OPCODE_ADDIU;
            addiu.rs = 9;
            addiu.rt = 9;
            addiu.immediate = static_cast<uint16_t>(tableAddress & 0xFFFFu);
            addiu.simmediate = addiu.immediate;

            Instruction sll{};
            sll.address = 0x1608;
            sll.opcode = OPCODE_SPECIAL;
            sll.function = SPECIAL_SLL;
            sll.rd = 8;
            sll.rt = 4;
            sll.sa = 2;

            Instruction addu{};
            addu.address = 0x160C;
            addu.opcode = OPCODE_SPECIAL;
            addu.function = SPECIAL_ADDU;
            addu.rs = 9;
            addu.rt = 8;
            addu.rd = 9;

            Instruction lw{};
            lw.address = 0x1610;
            lw.opcode = OPCODE_LW;
            lw.rs = 9;
            lw.rt = 10;
            lw.immediate = 0;
            lw.simmediate = 0;

            Instruction jr = makeJr(0x1614, 10);
            Instruction jrDelay = makeNop(0x1618);
            Instruction target0 = makeNop(0x1620);
            Instruction target1 = makeNop(0x1630);

            JumpTable configured{};
            configured.address = tableAddress;
            configured.entries.push_back({0u, 0x1620u});
            configured.entries.push_back({1u, 0x1630u});

            CodeGenerator gen({}, {});
            gen.setConfiguredJumpTables({configured});
            CodeGenerator::AnalysisResult analysis = gen.collectInternalBranchTargets(
                func,
                {lui, addiu, sll, addu, lw, jr, jrDelay, target0, target1});

            t.IsTrue(analysis.jumpTableTargets.contains(0x1614u),
                     "configured JR table should be tracked as a resolved local jump table");
            t.IsFalse(analysis.resumeEntryPoints.contains(0x1620u),
                      "configured JR table targets should stay in-function dispatch labels");
            t.IsFalse(analysis.resumeEntryPoints.contains(0x1630u),
                      "configured JR table targets should not become dispatcher resume sites");

            std::string generated = gen.generateFunction(
                func,
                {lui, addiu, sll, addu, lw, jr, jrDelay, target0, target1},
                false);
            printGeneratedCode("configured jump table addresses drive JR dispatch targets", generated);

            t.IsTrue(generated.find("switch (jumpTarget)") != std::string::npos,
                     "JR should emit a switch");
            t.IsTrue(generated.find("case 0x1620u: goto label_1620;") != std::string::npos,
                     "configured table target 0x1620 should be emitted");
            t.IsTrue(generated.find("case 0x1630u: goto label_1630;") != std::string::npos,
                     "configured table target 0x1630 should be emitted");
            t.IsTrue(generated.find("switch (ctx->pc)") == std::string::npos,
                     "configured JR table labels should not emit a top-level dispatcher resume switch");
            t.IsTrue(generated.find("case 0x1600u: goto label_1600;") == std::string::npos,
                     "configured table should avoid broad JR fallback labels");
        });

        tc.Run("configured indirect branch sites drive validated JR targets", [](TestCase &t) {
            Function func;
            func.name = "jr_configured_targets";
            func.start = 0x1680u;
            func.end = 0x16A0u;
            func.isRecompiled = true;
            func.isStub = false;

            const std::vector<Instruction> instructions{
                makeJr(0x1680u, 8u),
                makeNop(0x1684u),
                makeNop(0x1690u),
                makeNop(0x1694u),
            };
            const std::vector<Section> sections{
                {".text", 0x1680u, 0x20u, 0u, true, false, false, true, nullptr},
            };

            CodeGenerator gen({}, sections);
            gen.setConfiguredIndirectBranchTargets({
                {0x1680u, {0x1694u, 0x1690u, 0x1694u}},
            });
            const CodeGenerator::AnalysisResult analysis =
                gen.collectInternalBranchTargets(func, instructions);

            const auto targets = analysis.jumpTableTargets.find(0x1680u);
            t.IsTrue(targets != analysis.jumpTableTargets.end(),
                     "configured JR site should have resolved local targets");
            if (targets != analysis.jumpTableTargets.end())
            {
                t.Equals(targets->second.size(), static_cast<size_t>(2),
                         "configured JR targets should be deduplicated");
                t.Equals(targets->second[0], 0x1690u,
                         "configured JR targets should be sorted");
                t.Equals(targets->second[1], 0x1694u,
                         "configured JR should retain every validated target");
            }
            t.IsTrue(analysis.indirectFallbackEntryPoints.empty(),
                     "validated configured targets should avoid broad fallback promotion");

            const std::string generated =
                gen.generateFunction(func, instructions, false);
            t.IsTrue(generated.find(
                         "case 0x1690u: goto label_1690;") !=
                         std::string::npos,
                     "first configured target should appear in the emitted switch");
            t.IsTrue(generated.find(
                         "case 0x1694u: goto label_1694;") !=
                         std::string::npos,
                     "second configured target should appear in the emitted switch");

            gen.setConfiguredIndirectBranchTargets({
                {0x1680u, {0x3000u}},
            });
            const CodeGenerator::AnalysisResult invalidAnalysis =
                gen.collectInternalBranchTargets(func, instructions);
            t.IsTrue(invalidAnalysis.indirectFallbackEntryPoints.contains(
                         0x1680u),
                     "a configured non-executable target should retain conservative fallback coverage");
        });

        tc.Run("unresolved JALR uses runtime dispatch without broad local switch", [](TestCase &t) {
            Function func;
            func.name = "jalr_runtime_fallback";
            func.start = 0x1500;
            func.end = 0x1530;
            func.isRecompiled = true;
            func.isStub = false;

            // A call-like setup so there are multiple in-function labels available.
            Instruction jal = makeJal(0x1500, 0x1510);
            Instruction jalDelay = makeNop(0x1504);
            Instruction atReturn = makeNop(0x1508);
            Instruction atTarget = makeNop(0x1510);
            Instruction jalr = makeJalr(0x1514, 4, 31);
            Instruction jalrDelay = makeNop(0x1518);

            CodeGenerator gen({}, {});
            std::string generated = gen.generateFunction(func, {jal, jalDelay, atReturn, atTarget, jalr, jalrDelay}, false);
            printGeneratedCode("unresolved JALR uses runtime dispatch without broad local switch", generated);

            t.IsFalse(generated.find("switch (jumpTarget)") != std::string::npos,
                      "unresolved JALR should not emit a broad local switch over every internal label");
            t.IsTrue(generated.find("runtime->dispatchValidatedGuestBranchPolicy<Mode>(rdram, ctx, jumpTarget") != std::string::npos,
                     "unresolved JALR should dispatch through the runtime branch helper");
            t.IsTrue(generated.find("PS2Runtime::GuestBranchKind::IndirectCall") != std::string::npos,
                     "JALR should retain indirect-call dispatch kind");
            t.IsTrue(generated.find("0x1514u, 0x151Cu") != std::string::npos,
                     "JALR should pass call-site and fallthrough PCs to runtime dispatch");
        });

        tc.Run("JALR fallback should not expose epilogue tail-jump labels", [](TestCase &t) {
            Function func;
            func.name = "jalr_epilogue_guard";
            func.start = 0x2000;
            func.end = 0x2030;
            func.isRecompiled = true;
            func.isStub = false;

            Instruction prolog{};
            prolog.address = 0x2000;
            prolog.opcode = OPCODE_ADDIU;
            prolog.rs = 29;
            prolog.rt = 29;
            prolog.simmediate = static_cast<uint32_t>(static_cast<int32_t>(-0x20));
            prolog.raw = 0;

            Instruction saveRa{};
            saveRa.address = 0x2004;
            saveRa.opcode = OPCODE_SD;
            saveRa.rs = 29;
            saveRa.rt = 31;
            saveRa.simmediate = 0x10;
            saveRa.raw = 0;

            // Dynamic callback entry point.
            Instruction jalr = makeJalr(0x2008, 2, 31);
            Instruction jalrDelay = makeNop(0x200C);

            Instruction restoreRa{};
            restoreRa.address = 0x2010;
            restoreRa.opcode = OPCODE_LD;
            restoreRa.rs = 29;
            restoreRa.rt = 31;
            restoreRa.simmediate = 0x10;
            restoreRa.raw = 0;

            // Tail jump sequence that must not be reachable from jalr fallback dispatch.
            Instruction tailJump{};
            tailJump.address = 0x2014;
            tailJump.opcode = OPCODE_J;
            tailJump.target = (0x3000u >> 2) & 0x3FFFFFFu;
            tailJump.hasDelaySlot = true;
            tailJump.raw = 0;

            Instruction tailDelay{};
            tailDelay.address = 0x2018;
            tailDelay.opcode = OPCODE_ADDIU;
            tailDelay.rs = 29;
            tailDelay.rt = 29;
            tailDelay.simmediate = 0x20;
            tailDelay.raw = 0;

            CodeGenerator gen({}, {});
            std::string generated = gen.generateFunction(
                func,
                {prolog, saveRa, jalr, jalrDelay, restoreRa, tailJump, tailDelay},
                false);
            printGeneratedCode("JALR fallback should not expose epilogue tail-jump labels", generated);

            t.IsTrue(generated.find("case 0x2014u: goto label_2014;") == std::string::npos,
                     "jalr fallback should not dispatch directly to epilogue tail-jump block");
            t.IsTrue(generated.find("case 0x2018u: goto label_2018;") == std::string::npos,
                     "jalr fallback should not dispatch directly to tail-jump delay slot");
        });

        tc.Run("VU random helpers emit line comments on separate lines", [](TestCase &t) {
            CodeGenerator gen({}, {});

            Instruction inst{};
            inst.rd = 7;
            inst.vectorInfo.fsf = 2;

            std::string vrnext = gen.translateVU_VRNEXT(inst);
            printGeneratedCode("VU random helpers emit line comments on separate lines - VRNEXT", vrnext);
            t.IsTrue(vrnext.find("// Simple LFSR-based random number generation (PS2-like behavior)\n"
                                 "    uint32_t feedback") != std::string::npos,
                     "VRNEXT should place the generated line comment on its own line");

            std::string vrinit = gen.translateVU_VRINIT(inst);
            printGeneratedCode("VU random helpers emit line comments on separate lines - VRINIT", vrinit);
            t.IsTrue(vrinit.find("// PS2 uses a specific LFSR initialization pattern\n"
                                 "    if (seed == 0) seed = 1;") != std::string::npos,
                     "VRINIT should place the generated line comment on its own line");

            std::string vrxor = gen.translateVU_VRXOR(inst);
            printGeneratedCode("VU random helpers emit line comments on separate lines - VRXOR", vrxor);
            t.IsTrue(vrxor.find("// XOR the current random value with the data from the VU vector register\n"
                                "    __m128i xored") != std::string::npos,
                     "VRXOR should keep the XOR comment on its own line");
            t.IsTrue(vrxor.find("// Apply a simple mixing function similar to PS2's LFSR\n"
                                "    __m128i mixed") != std::string::npos,
                     "VRXOR should keep the LFSR comment on its own line");
        });

        tc.Run("resolveStubTarget allows leading underscore alias", [](TestCase &t) {
            t.Equals(PS2Recompiler::resolveStubTarget("_rand"), StubTarget::Stub,
                     "_rand should resolve via rand stub alias");
            t.Equals(PS2Recompiler::resolveStubTarget("_GetThreadId"), StubTarget::Syscall,
                     "_GetThreadId should resolve via GetThreadId syscall alias");
            t.Equals(PS2Recompiler::resolveStubTarget("_DefinitelyNotARealCall"), StubTarget::Unknown,
                     "unknown names must still stay unknown");
        });
    
    });
}
