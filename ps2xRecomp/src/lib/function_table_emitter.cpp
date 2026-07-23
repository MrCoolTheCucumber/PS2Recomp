#include "ps2recomp/Emitters/function_table_emitter.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/ps2_recompiler.h"
#include "ps2recomp/recompiler_reporter.h"
#include "ps2recomp/types.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ps2recomp
{
    namespace
    {
        std::string escapeCStringLiteral(const std::string &value)
        {
            std::string escaped;
            escaped.reserve(value.size());
            for (unsigned char c : value)
            {
                switch (c)
                {
                case '\\':
                    escaped += "\\\\";
                    break;
                case '"':
                    escaped += "\\\"";
                    break;
                case '\n':
                    escaped += "\\n";
                    break;
                case '\r':
                    escaped += "\\r";
                    break;
                case '\t':
                    escaped += "\\t";
                    break;
                default:
                    if (c < 0x20u || c == 0x7Fu)
                    {
                        escaped += '\\';
                        escaped += static_cast<char>('0' + ((c >> 6u) & 0x07u));
                        escaped += static_cast<char>('0' + ((c >> 3u) & 0x07u));
                        escaped += static_cast<char>('0' + (c & 0x07u));
                    }
                    else
                    {
                        escaped += static_cast<char>(c);
                    }
                    break;
                }
            }
            return escaped;
        }
    }

    FunctionTableEmitter::FunctionTableEmitter(CodeGenerator &codeGenerator)
        : m_codeGenerator(codeGenerator)
    {
    }

    std::string FunctionTableEmitter::emit(const std::vector<Function> &functions, const std::map<uint32_t, std::string> &stubs)
    {
        (void)stubs;

        CodeGenerator &cg = m_codeGenerator;
        std::vector<std::pair<uint32_t, std::string>> entries;
        std::unordered_set<uint32_t> registeredAddresses;

        auto addEntry = [&](uint32_t address, const std::string &name)
        {
            if (name.empty())
            {
                return;
            }
            if ((address & 3u) != 0u)
            {
                std::ostringstream oss;
                oss << "Unaligned function table entry for " << name << " at 0x" << std::hex << address;

                if (cg.m_reporter)
                {
                    cg.m_reporter->errorAt("function-table", name, address, oss.str());
                }

                throw std::runtime_error(oss.str());
            }
            if (!registeredAddresses.insert(address).second)
            {
                return;
            }
            entries.emplace_back(address, name);
        };

        std::vector<std::pair<uint32_t, std::string>> normalFunctions;
        std::vector<std::pair<uint32_t, std::string>> stubFunctions;
        std::vector<std::pair<uint32_t, std::string>> systemCallFunctions;
        std::vector<std::pair<uint32_t, std::string>> libraryFunctions;

        for (const auto &function : functions)
        {
            if (!function.isRecompiled && !function.isStub && !function.isSkipped)
                continue;

            std::string generatedName = cg.getFunctionName(function.start);

            if (function.isSkipped)
            {
                libraryFunctions.emplace_back(function.start, generatedName);
            }
            else if (function.isStub)
            {
                const auto target = PS2Recompiler::resolveStubTarget(function.name);
                if (target == StubTarget::Syscall)
                {
                    systemCallFunctions.emplace_back(function.start, generatedName);
                }
                else
                {
                    stubFunctions.emplace_back(function.start, generatedName);
                }
            }
            else
            {
                normalFunctions.emplace_back(function.start, generatedName);
            }
        }

        if (cg.m_bootstrapInfo.valid)
        {
            std::string entryTarget = cg.m_bootstrapInfo.entryName;
            if (entryTarget.empty())
            {
                entryTarget = cg.getFunctionName(cg.m_bootstrapInfo.entry);
            }
            if (entryTarget.empty())
            {
                throw std::runtime_error("No entry function name available for registration.");
            }
            addEntry(cg.m_bootstrapInfo.entry, entryTarget);
        }

        for (const auto &[address, name] : normalFunctions)
        {
            addEntry(address, name);
        }

        for (const auto &[ownerStart, targets] : cg.m_resumeEntryTargetsByOwner)
        {
            const std::string ownerName = cg.getFunctionName(ownerStart);
            if (ownerName.empty())
            {
                continue;
            }

            for (uint32_t target : targets)
            {
                addEntry(target, ownerName);
            }
        }

        for (const auto &[address, name] : stubFunctions)
        {
            addEntry(address, name);
        }
        for (const auto &[address, name] : systemCallFunctions)
        {
            addEntry(address, name);
        }
        for (const auto &[address, name] : libraryFunctions)
        {
            addEntry(address, name);
        }

        std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b)
                  { return a.first < b.first; });

        struct GuestFunctionSymbolRecord
        {
            uint32_t end;
            std::string name;
        };
        std::map<uint32_t, GuestFunctionSymbolRecord> symbols;
        for (const auto &function : functions)
        {
            if (function.name.empty() || function.end <= function.start)
            {
                continue;
            }

            symbols.try_emplace(function.start,
                                GuestFunctionSymbolRecord{function.end, function.name});
        }

        uint32_t tableBase = 0u;
        uint32_t tableEnd = 0u;
        uint32_t slotCount = 0u;
        if (!entries.empty())
        {
            tableBase = entries.front().first & ~3u;
            tableEnd = (entries.back().first + 4u + 3u) & ~3u;
            slotCount = (tableEnd - tableBase) >> 2;
        }

        std::stringstream ss;
        ss << "#include \"ps2_runtime.h\"\n";
        ss << "#include \"ps2_recompiled_functions.h\"\n";
        ss << "#include \"ps2_stubs.h\"\n";
        ss << "#include \"ps2_recompiled_stubs.h\"//this will give duplicated erros because runtime maybe has it define already, just delete the TODOS ones\n";
        ss << "#include \"ps2_syscalls.h\"\n\n";

        ss << "extern const uint32_t g_ps2RecompiledFunctionTableBase = 0x" << std::hex << tableBase << "u;\n";
        ss << "extern const uint32_t g_ps2RecompiledFunctionTableEnd = 0x" << std::hex << tableEnd << "u;\n";
        ss << "extern const uint32_t g_ps2RecompiledFunctionTableSlotCount = " << std::dec << slotCount << "u;\n";
        ss << "PS2Runtime::RecompiledFunction g_ps2RecompiledFunctionTable[" << std::dec << (slotCount == 0u ? 1u : slotCount) << "u] = {};\n\n";

        ss << "extern const uint32_t g_ps2GuestFunctionSymbolCount = " << std::dec << symbols.size() << "u;\n";
        ss << "extern const PS2GuestFunctionSymbol g_ps2GuestFunctionSymbols["
           << std::dec << (symbols.empty() ? 1u : symbols.size()) << "u] = {\n";
        if (symbols.empty())
        {
            ss << "    {0u, 0u, nullptr},\n";
        }
        else
        {
            for (const auto &[start, symbol] : symbols)
            {
                ss << "    {0x" << std::hex << start << "u, 0x" << symbol.end
                   << "u, \"" << escapeCStringLiteral(symbol.name) << "\"},\n";
            }
        }
        ss << "};\n\n";

        ss << "namespace {\n";
        ss << "struct GeneratedFunctionTableInitializer {\n";
        ss << "    GeneratedFunctionTableInitializer() {\n";
        for (const auto &[address, name] : entries)
        {
            const uint32_t slot = (address - tableBase) >> 2;
            ss << "        g_ps2RecompiledFunctionTable[" << std::dec << slot << "] = " << name
               << "; // 0x" << std::hex << address << std::dec << "\n";
        }
        ss << "    }\n";
        ss << "};\n";
        ss << "static const GeneratedFunctionTableInitializer g_generatedFunctionTableInitializer;\n";
        ss << "}\n";

        return ss.str();
    }
}
