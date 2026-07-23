#ifndef PS2RECOMP_TYPES_H
#define PS2RECOMP_TYPES_H

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <map>

namespace ps2recomp
{

    // Instruction representation
    struct Instruction
    {
        uint32_t address;
        uint32_t opcode;
        uint32_t rs;         // Source register
        uint32_t rt;         // Target register
        uint32_t rd;         // Destination register
        uint32_t sa;         // Shift amount
        uint32_t function;   // Function code for R-type instructions
        uint32_t immediate;  // Immediate value for I-type instructions
        uint32_t simmediate; // Sign-extended immediate value (extended to 32 bits)
        uint32_t target;     // Jump target for J-type instructions
        uint32_t raw;        // Raw instruction value

        // Instruction type flags
        bool isMMI;        // Is MMI instruction (PS2 specific)
        bool isVU;         // Is VU instruction (PS2 specific)
        bool isBranch;     // Is branch instruction
        bool isJump;       // Is jump instruction
        bool isCall;       // Is function call
        bool isReturn;     // Is return instruction
        bool hasDelaySlot; // Has delay slot
        bool isMultimedia; // PS2-specific multimedia operations
        bool isStore;      // Is store instruction
        bool isLoad;       // Is load instruction

        // Additional PS2-specific fields
        uint8_t mmiType;        // 0=MMI0, 1=MMI1, 2=MMI2, 3=MMI3
        uint8_t mmiFunction;    // Function within MMI type
        uint8_t pmfhlVariation; // For PMFHL instructions
        uint8_t vuFunction;     // For VU instructions

        bool isMmio = false;
        uint32_t mmioAddress = 0;
        std::string disassembly;

        struct
        {
            bool isVector;       // Uses vector operations
            bool usesQReg;       // Uses Q register
            bool usesPReg;       // Uses P register
            bool modifiesMAC;    // Modifies MAC flags
            uint8_t vectorField; // xyzw field mask
            uint8_t fsf;         // Field select for FS reg (bits 10-11)
            uint8_t ftf;         // Source Field select for FT reg (bits 8-9)
        } vectorInfo;

        struct
        {
            bool modifiesGPR;     // Modifies general purpose register
            bool modifiesFPR;     // Modifies floating point register
            bool modifiesVFR;     // Modifies vector float register
            bool modifiesVIR;     // Modifies vector integer register
            bool modifiesVIC;     // Modifies vector integer control register
            bool modifiesMemory;  // Modifies memory
            bool modifiesControl; // Modifies control register
        } modificationInfo;

        Instruction() : address(0), opcode(0), rs(0), rt(0), rd(0), sa(0), function(0),
                        immediate(0), simmediate(0), target(0), raw(0),
                        isMMI(false), isVU(false), isBranch(false), isJump(false), isCall(false),
                        isReturn(false), hasDelaySlot(false), isMultimedia(false), isStore(false), isLoad(false),
                        mmiType(0), mmiFunction(0), pmfhlVariation(0), vuFunction(0), isMmio(false), mmioAddress(0)
        {
            vectorInfo = {};
            modificationInfo = {};
        }
    };

    struct MemoryAccessHint
    {
        bool hasAddress = false;
        uint32_t address = 0;
    };

    // Function information
    struct Function
    {
        std::string name;
        uint32_t start = 0;
        uint32_t end = 0;
        std::vector<Instruction> instructions;
        std::vector<uint32_t> callers;
        std::vector<uint32_t> callees;
        bool isRecompiled = false;
        bool isStub = false;
        bool isSkipped = false;
    };

    // Symbol information
    struct Symbol
    {
        std::string name;
        uint32_t address = 0;
        uint32_t size = 0;
        bool isFunction = false;
        bool isImported = false;
        bool isExported = false;
    };

    // Section information
    struct Section
    {
        std::string name;
        uint32_t address = 0;
        uint32_t size = 0;
        uint32_t offset = 0;
        bool isCode = false;
        bool isData = false;
        bool isBSS = false;
        bool isReadOnly = false;
        uint8_t *data = nullptr;
    };

    // Relocation information
    struct Relocation
    {
        uint32_t offset = 0;
        uint32_t info = 0;
        uint32_t symbol = 0;
        std::string symbolName;
        uint32_t type = 0;
        int32_t addend = 0;
    };

    // Jump table entry
    struct JumpTableEntry
    {
        uint32_t index = 0;
        uint32_t target = 0;
    };

    // Jump table
    struct JumpTable
    {
        uint32_t address = 0;
        uint32_t baseRegister = 0;
        std::vector<JumpTableEntry> entries;
    };

    // Control flow graph
    struct CFGNode
    {
        uint32_t startAddress = 0;
        uint32_t endAddress = 0;
        std::vector<Instruction> instructions;
        std::vector<uint32_t> predecessors;
        std::vector<uint32_t> successors;
        bool isJumpTarget = false;
        bool hasJumpTable = false;
        JumpTable jumpTable;
    };

    // Function call
    struct FunctionCall
    {
        uint32_t callerAddress = 0;
        uint32_t calleeAddress = 0;
        std::string calleeName;
    };

    // Recompiler configuration
    struct RecompilerConfig
    {
        std::string inputPath;
        std::string outputPath;
        std::string ghidraMapPath;
        bool singleFileOutput = false;
        bool lowMemoryMode = false;
        uint32_t outputWorkerThreads = 0;
        bool patchSyscalls = false;
        bool patchCop0 = true;
        bool patchCache = true;
        bool recoverLeafFunctions = false;
        std::vector<std::string> skipFunctions;
        std::unordered_map<uint32_t, std::string> patches;
        std::vector<std::string> stubImplementations;
        std::unordered_map<uint32_t, uint32_t> mmioByInstructionAddress;
        std::vector<JumpTable> jumpTables;
        std::vector<Function> functionBoundaries;
    };

} // namespace ps2recomp

#endif // PS2RECOMP_TYPES_H
