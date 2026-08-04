#include "ps2recomp/code_generator.h"
#include "ps2recomp/codegen_helpers.h"
#include "ps2recomp/ee_observation_mode.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/types.h"
#include <fmt/format.h>
#include <sstream>
#include <cmath>

namespace ps2recomp
{
    std::string CodeGenerator::generateJumpTableSwitch(const Instruction &inst, uint32_t tableAddress,
                                                       const std::vector<JumpTableEntry> &entries)
    {
        std::stringstream ss;

        uint32_t indexReg = inst.rs;

        ss << "switch (GPR_U32(ctx, " << indexReg << ")) {\n";

        for (const auto &[index, target] : entries)
        {
            ss << "    case " << index << ": {\n";

            std::string funcName = getFunctionName(target);
            if (!funcName.empty())
            {
                ss << "        if constexpr (Mode == "
                      "EeArchitecturalObservationMode::Fast) {\n";
                ss << "            " << eeFastFunctionName(funcName)
                   << "(rdram, ctx, runtime);\n";
                ss << "        } else {\n";
                ss << "            " << eePreciseFunctionName(funcName)
                   << "(rdram, ctx, runtime);\n";
                ss << "        }\n";
            }
            else
            {
                const std::string fallbackName =
                    fmt::format("func_{:x}", target);
                ss << "        if constexpr (Mode == "
                      "EeArchitecturalObservationMode::Fast) {\n";
                ss << "            " << eeFastFunctionName(fallbackName)
                   << "(rdram, ctx, runtime);\n";
                ss << "        } else {\n";
                ss << "            " << eePreciseFunctionName(fallbackName)
                   << "(rdram, ctx, runtime);\n";
                ss << "        }\n";
            }

            ss << "        return;\n";
            ss << "    }\n";
        }

        ss << "    default:\n";
        ss << "        // Unknown jump table target\n";
        ss << "        return;\n";
        ss << "}\n";

        return ss.str();
    }

}
