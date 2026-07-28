#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu_ir.h"
#include "runtime/ps2_vu_recompiler.h"
#include "runtime/ps2_vu1.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
    struct FixturePaths
    {
        std::filesystem::path data;
        std::filesystem::path micro;
        std::filesystem::path state;
    };

    enum class ScheduledWriteKind
    {
        Vi,
        Vf,
    };

    struct ScheduledRegisterWrite
    {
        uint64_t instructionIndex = 0;
        ScheduledWriteKind kind = ScheduledWriteKind::Vi;
        uint8_t registerIndex = 0;
        uint8_t lane = 0;
        uint32_t value = 0;
    };

    bool readFile(const std::string &path, void *destination, size_t size)
    {
        std::ifstream input(path, std::ios::binary);
        input.read(static_cast<char *>(destination), static_cast<std::streamsize>(size));
        return input.good() || static_cast<size_t>(input.gcount()) == size;
    }

    uint32_t floatBits(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    bool parseUnsigned(const std::string &text, uint64_t &value)
    {
        if (text.empty() || text.front() == '-')
            return false;

        size_t consumed = 0u;
        try
        {
            value = std::stoull(text, &consumed, 0);
        }
        catch (const std::exception &)
        {
            return false;
        }
        return consumed == text.size();
    }

    bool resolveFixture(const std::filesystem::path &directory,
                        FixturePaths &paths)
    {
        const FixturePaths standard{
            directory / "vu1-data.bin",
            directory / "vu1-code.bin",
            directory / "vu1-replay-state.bin",
        };
        if (std::filesystem::is_regular_file(standard.data) &&
            std::filesystem::is_regular_file(standard.micro) &&
            std::filesystem::is_regular_file(standard.state))
        {
            paths = standard;
            return true;
        }

        std::vector<FixturePaths> candidates;
        std::error_code error;
        for (const auto &entry :
             std::filesystem::directory_iterator(directory, error))
        {
            if (error || !entry.is_regular_file())
                continue;
            const std::string filename =
                entry.path().filename().string();
            constexpr std::string_view suffix = "-state.bin";
            if (!filename.ends_with(suffix))
                continue;
            const std::string prefix =
                filename.substr(0u, filename.size() - suffix.size());
            FixturePaths candidate{
                directory / (prefix + "-data.bin"),
                directory / (prefix + "-micro.bin"),
                entry.path(),
            };
            if (!std::filesystem::is_regular_file(candidate.micro))
                candidate.micro = directory / (prefix + "-code.bin");
            if (std::filesystem::is_regular_file(candidate.data) &&
                std::filesystem::is_regular_file(candidate.micro))
            {
                candidates.push_back(std::move(candidate));
            }
        }
        if (candidates.size() != 1u)
            return false;
        paths = std::move(candidates.front());
        return true;
    }

    bool readRegisterWriteSchedule(
        const std::filesystem::path &path,
        std::vector<ScheduledRegisterWrite> &writes)
    {
        std::ifstream input(path);
        if (!input)
            return false;

        std::string line;
        uint64_t previousIndex = 0;
        bool havePreviousIndex = false;
        while (std::getline(input, line))
        {
            const size_t comment = line.find('#');
            if (comment != std::string::npos)
                line.resize(comment);

            std::istringstream fields(line);
            std::vector<std::string> tokens;
            for (std::string token; fields >> token;)
                tokens.push_back(std::move(token));
            if (tokens.empty())
                continue;

            uint64_t instructionIndex = 0u;
            if (!parseUnsigned(tokens[0], instructionIndex))
                return false;
            ScheduledRegisterWrite write;
            write.instructionIndex = instructionIndex;

            // Preserve the original "INDEX REGISTER VALUE" VI-only format.
            if (tokens.size() == 3u)
            {
                uint64_t registerIndex = 0u;
                uint64_t value = 0u;
                if (!parseUnsigned(tokens[1], registerIndex) ||
                    !parseUnsigned(tokens[2], value) ||
                    registerIndex >= 16u || value > UINT32_MAX)
                {
                    return false;
                }
                write.kind = ScheduledWriteKind::Vi;
                write.registerIndex = static_cast<uint8_t>(
                    registerIndex);
                write.value = static_cast<uint32_t>(value);
            }
            else if (tokens.size() == 4u && tokens[1] == "vi")
            {
                uint64_t registerIndex = 0u;
                uint64_t value = 0u;
                if (!parseUnsigned(tokens[2], registerIndex) ||
                    !parseUnsigned(tokens[3], value) ||
                    registerIndex >= 16u || value > UINT32_MAX)
                {
                    return false;
                }
                write.kind = ScheduledWriteKind::Vi;
                write.registerIndex = static_cast<uint8_t>(
                    registerIndex);
                write.value = static_cast<uint32_t>(value);
            }
            else if (tokens.size() == 5u && tokens[1] == "vf")
            {
                uint64_t registerIndex = 0u;
                uint64_t lane = 0u;
                uint64_t value = 0u;
                if (!parseUnsigned(tokens[2], registerIndex) ||
                    !parseUnsigned(tokens[3], lane) ||
                    !parseUnsigned(tokens[4], value) ||
                    registerIndex >= 32u || lane >= 4u ||
                    value > UINT32_MAX)
                {
                    return false;
                }
                write.kind = ScheduledWriteKind::Vf;
                write.registerIndex = static_cast<uint8_t>(
                    registerIndex);
                write.lane = static_cast<uint8_t>(lane);
                write.value = static_cast<uint32_t>(value);
            }
            else
            {
                return false;
            }

            if (havePreviousIndex && instructionIndex < previousIndex)
            {
                return false;
            }

            writes.push_back(write);
            previousIndex = instructionIndex;
            havePreviousIndex = true;
        }

        return true;
    }

    bool sameRunResult(
        const VuRunResult &left, const VuRunResult &right)
    {
        return
            left.requestedCycles == right.requestedCycles &&
            left.executedCycles == right.executedCycles &&
            left.reason == right.reason &&
            left.activeBefore == right.activeBefore &&
            left.activeAfter == right.activeAfter &&
            left.completed == right.completed;
    }

    bool sameDifferentialRunResult(
        const VuRunResult &reference,
        const VuRunResult &candidate,
        bool nativeCandidate)
    {
        if (sameRunResult(reference, candidate))
            return true;
        return
            nativeCandidate &&
            reference.requestedCycles ==
                candidate.requestedCycles &&
            reference.executedCycles ==
                candidate.executedCycles &&
            reference.reason == VuExitReason::CycleBudget &&
            candidate.reason ==
                VuExitReason::XgkickBoundary &&
            reference.activeBefore ==
                candidate.activeBefore &&
            reference.activeAfter ==
                candidate.activeAfter &&
            reference.completed == candidate.completed;
    }
}

int main(int argc, char **argv)
{
    bool irDifferential = false;
    bool recompilerDifferential = false;
    bool unitBackendSpecified = false;
    VuBackendKind unitBackend = VuBackendKind::Interpreter;
    std::vector<std::string> arguments;
    for (int index = 1; index < argc; ++index)
    {
        const std::string_view argument(argv[index]);
        if (argument == "--ir-differential")
            irDifferential = true;
        else if (argument == "--recompiler-differential")
        {
            recompilerDifferential = true;
        }
        else if (argument.starts_with("--backend="))
        {
            const std::string_view value =
                argument.substr(
                    std::string_view("--backend=").size());
            if (!parseVuBackendKind(value, unitBackend) ||
                (unitBackend != VuBackendKind::Interpreter &&
                 unitBackend != VuBackendKind::Recompiler))
            {
                std::cerr
                    << "--backend must be interpreter or recompiler\n";
                return 2;
            }
            unitBackendSpecified = true;
        }
        else
            arguments.emplace_back(argv[index]);
    }
    if ((irDifferential && recompilerDifferential) ||
        (unitBackendSpecified &&
         (irDifferential || recompilerDifferential)))
    {
        std::cerr
            << "select either one differential mode or --backend\n";
        return 2;
    }
    if (arguments.size() < 2u || arguments.size() > 6u)
    {
        std::cerr << "usage: vu1_replay FIXTURE_DIRECTORY INSTRUCTION_PAIRS "
                     "[GIF_OUTPUT] [FINAL_VU_DATA] [INSTRUCTION_TRACE] "
                     "[REGISTER_WRITE_SCHEDULE] "
                     "[--ir-differential|--recompiler-differential|"
                     "--backend=interpreter|recompiler]\n"
                     "schedule lines: INDEX REGISTER VALUE (legacy VI), "
                     "INDEX vi REGISTER VALUE, or "
                     "INDEX vf REGISTER LANE VALUE\n";
        return 2;
    }

    const std::filesystem::path directory = arguments[0];
    const uint32_t maxCycles =
        static_cast<uint32_t>(std::stoul(arguments[1]));
    FixturePaths fixture;
    if (!resolveFixture(directory, fixture))
    {
        std::cerr
            << "fixture directory must contain vu1-data.bin, "
               "vu1-code.bin, and vu1-replay-state.bin, or exactly one "
               "matching *-data/*-micro/*-state set\n";
        return 2;
    }
    std::array<uint8_t, PS2_VU1_DATA_SIZE> initialData{};
    std::array<uint8_t, PS2_VU1_CODE_SIZE> micro{};
    std::array<uint32_t, 159> stateWords{};
    std::error_code sizeError;
    const uintmax_t dataFileSize =
        std::filesystem::file_size(fixture.data, sizeError);
    const uintmax_t microFileSize =
        sizeError ? 0u : std::filesystem::file_size(fixture.micro, sizeError);
    if (sizeError ||
        dataFileSize != microFileSize ||
        (dataFileSize != PS2_VU0_DATA_SIZE &&
         dataFileSize != PS2_VU1_DATA_SIZE))
    {
        std::cerr
            << "VU data and micro files must have matching VU0 or VU1 sizes\n";
        return 2;
    }
    const uint32_t vuSize = static_cast<uint32_t>(dataFileSize);
    if (!readFile(fixture.data.string(), initialData.data(), vuSize) ||
        !readFile(fixture.micro.string(), micro.data(), vuSize) ||
        !readFile(fixture.state.string(), stateWords.data(), stateWords.size() * sizeof(uint32_t)))
    {
        std::cerr << "failed to read replay fixture\n";
        return 2;
    }
    if (stateWords[0] != 0x31555652u || stateWords[1] != 1u)
    {
        std::cerr << "unsupported replay fixture\n";
        return 2;
    }

    PS2Memory memory;
    if (!memory.initialize())
    {
        std::cerr << "failed to initialize memory\n";
        return 2;
    }
    GS gs;
    gs.init(memory.getGSVRAM(), static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &memory.gs());
    std::vector<uint8_t> gifOutput;
    memory.setGifPacketCallback([&](const uint8_t *data, uint32_t size)
    {
        gifOutput.insert(gifOutput.end(), data, data + size);
    });
    std::memcpy(memory.getVU1Code(), micro.data(), vuSize);
    std::memcpy(memory.getVU1Data(), initialData.data(), vuSize);

    VuUnit unit;
    if (unitBackendSpecified)
    {
        std::string diagnostic;
        if (!unit.setBackend(unitBackend, &diagnostic))
        {
            std::cerr << diagnostic << "\n";
            return 2;
        }
    }
    VuExecutionState &state = unit.state();
    size_t cursor = 5;
    std::memcpy(state.vf, stateWords.data() + cursor, sizeof(state.vf));
    cursor += 32 * 4;
    for (size_t index = 0; index < 16; ++index)
        state.vi[index] = static_cast<int32_t>(stateWords[cursor++]);
    std::memcpy(state.acc, stateWords.data() + cursor, sizeof(state.acc));
    cursor += 4;
    std::memcpy(&state.q, &stateWords[cursor++], sizeof(state.q));
    std::memcpy(&state.p, &stateWords[cursor++], sizeof(state.p));
    std::memcpy(&state.i, &stateWords[cursor++], sizeof(state.i));
    state.status = stateWords[cursor++];
    state.mac = stateWords[cursor++];
    state.clip = stateWords[cursor++];

    std::vector<ScheduledRegisterWrite> scheduledWrites;
    if (arguments.size() == 6u &&
        !readRegisterWriteSchedule(
            arguments[5], scheduledWrites))
    {
        std::cerr << "failed to read register write schedule\n";
        return 2;
    }

    std::ofstream instructionTrace;
    uint64_t observedInstructionIndex = 0;
    if (arguments.size() >= 5u)
    {
        instructionTrace.open(arguments[4]);
        if (!instructionTrace)
        {
            std::cerr << "failed to open instruction trace output\n";
            return 2;
        }
        instructionTrace << std::hex << std::setfill('0');
        unit.setInstructionObserver(
            [&](uint64_t index, uint32_t pc, uint32_t lower, uint32_t upper,
                const VuExecutionState &stepState)
            {
                (void)index;
                const uint64_t traceIndex = observedInstructionIndex++;
                instructionTrace
                    << "{\"schema_version\":1,\"event\":\"vu-step\","
                    << "\"index\":" << std::dec << traceIndex << std::hex
                    << ",\"pc\":\"0x" << std::setw(4) << (pc & 0x3fffu)
                    << "\",\"lower\":\"0x" << std::setw(8) << lower
                    << "\",\"upper\":\"0x" << std::setw(8) << upper
                    << "\",\"vi\":[";
                for (size_t registerIndex = 0; registerIndex < 16;
                     ++registerIndex)
                {
                    instructionTrace
                        << (registerIndex == 0 ? "" : ",")
                        << "\"0x" << std::setw(4)
                        << (static_cast<uint32_t>(
                                stepState.vi[registerIndex]) &
                            0xffffu)
                        << "\"";
                }
                instructionTrace << "],\"vf\":[";
                for (size_t componentIndex = 0;
                     componentIndex < 32u * 4u; ++componentIndex)
                {
                    instructionTrace
                        << (componentIndex == 0 ? "" : ",")
                        << "\"0x" << std::setw(8)
                        << floatBits((&stepState.vf[0][0])[componentIndex])
                        << "\"";
                }
                instructionTrace << "],\"acc\":[";
                for (size_t componentIndex = 0; componentIndex < 4u;
                     ++componentIndex)
                {
                    instructionTrace
                        << (componentIndex == 0 ? "" : ",")
                        << "\"0x" << std::setw(8)
                        << floatBits(stepState.acc[componentIndex])
                        << "\"";
                }
                instructionTrace
                    << "],\"q\":\"0x" << std::setw(8)
                    << floatBits(stepState.q)
                    << "\",\"p\":\"0x" << std::setw(8)
                    << floatBits(stepState.p)
                    << "\",\"i\":\"0x" << std::setw(8)
                    << floatBits(stepState.i)
                    << "\",\"status\":\"0x" << std::setw(8)
                    << stepState.status
                    << "\",\"mac\":\"0x" << std::setw(8)
                    << stepState.mac
                    << "\",\"clip\":\"0x" << std::setw(8)
                    << stepState.clip
                    << "\",\"branch_pending\":"
                    << (stepState.branchPending ? "true" : "false")
                    << ",\"branch_target\":\"0x" << std::setw(4)
                    << (stepState.branchTarget & 0x3fffu)
                    << "\",\"branch_delay\":" << std::dec
                    << stepState.branchDelay
                    << ",\"vi_backup_cycles\":"
                    << static_cast<uint32_t>(stepState.viBackupCycles)
                    << ",\"vi_backup_register\":"
                    << static_cast<uint32_t>(stepState.viBackupRegister)
                    << ",\"vi_backup_value\":\"0x" << std::hex
                    << std::setw(4)
                    << (static_cast<uint32_t>(stepState.viBackupValue) &
                        0xffffu)
                    << "\"}\n";
            });
    }

    constexpr size_t kVuIrOpcodeCount =
        static_cast<size_t>(VuIrOpcode::Unsupported) + 1u;
    std::array<uint64_t, kVuIrOpcodeCount> irOpcodeCounts{};
    uint64_t differentialComparedPairs = 0u;
    VuRecompilerDiagnostics recompilerDiagnostics{};
    if (irDifferential || recompilerDifferential)
    {
        if (recompilerDifferential &&
            vuSize != PS2_VU1_CODE_SIZE)
        {
            std::cerr
                << "the VU1 recompiler requires a full VU1 fixture\n";
            return 2;
        }
        if (recompilerDifferential &&
            !VuRecompilerBackend::supported())
        {
            std::cerr
                << "the x86-64 VU recompiler is unavailable\n";
            return 2;
        }

        unit.start(
            stateWords[2], stateWords[3], stateWords[4],
            &memory);
        VuExecutionState referenceState = state;
        VuExecutionState candidateState = state;
        std::vector<uint8_t> referenceData(
            initialData.begin(), initialData.begin() + vuSize);
        std::vector<uint8_t> candidateData = referenceData;
        VuTransactionalSideEffectSink referenceEffects;
        VuTransactionalSideEffectSink candidateEffects;
        VuUnit candidateUnit(VuUnitId::Vu1);
        VuInterpreterBackend reference(unit);
        VuIrInterpreterBackend oracle(candidateUnit);
        VuRecompilerBackend recompiler(candidateUnit);
        IVuExecutionBackend *const candidate =
            recompilerDifferential
                ? static_cast<IVuExecutionBackend *>(
                      &recompiler)
                : static_cast<IVuExecutionBackend *>(
                      &oracle);
        VuExecutionContext referenceContext{
            .state = referenceState,
            .code = memory.getVU1Code(),
            .codeSize = vuSize,
            .data = referenceData.data(),
            .dataSize = vuSize,
            .sideEffects = referenceEffects,
            .memory = &memory,
            .enableInstrumentation = true,
        };
        VuExecutionContext candidateContext{
            .state = candidateState,
            .code = memory.getVU1Code(),
            .codeSize = vuSize,
            .data = candidateData.data(),
            .dataSize = vuSize,
            .sideEffects = candidateEffects,
            .memory = &memory,
            .enableInstrumentation = false,
        };

        size_t nextWrite = 0u;
        for (uint64_t instructionIndex = 0u;
             instructionIndex < maxCycles &&
             referenceState.active;
             ++instructionIndex)
        {
            while (nextWrite < scheduledWrites.size() &&
                   scheduledWrites[nextWrite].instructionIndex ==
                       instructionIndex)
            {
                const ScheduledRegisterWrite &write =
                    scheduledWrites[nextWrite++];
                if (write.kind == ScheduledWriteKind::Vi)
                {
                    referenceState.vi[write.registerIndex] =
                        static_cast<int32_t>(write.value);
                    candidateState.vi[write.registerIndex] =
                        static_cast<int32_t>(write.value);
                }
                else
                {
                    std::memcpy(
                        &referenceState.vf[write.registerIndex][write.lane],
                        &write.value, sizeof(write.value));
                    std::memcpy(
                        &candidateState.vf[
                            write.registerIndex][write.lane],
                        &write.value, sizeof(write.value));
                }
            }

            const uint32_t issuedPc = referenceState.pc;
            uint32_t lowerWord = 0u;
            uint32_t upperWord = 0u;
            VuIrInstructionPair pair;
            pair.pc = issuedPc;
            if (issuedPc + 8u <= vuSize)
            {
                std::memcpy(
                    &lowerWord, memory.getVU1Code() + issuedPc,
                    sizeof(lowerWord));
                std::memcpy(
                    &upperWord,
                    memory.getVU1Code() + issuedPc + 4u,
                    sizeof(upperWord));
                pair = decodeVuIrInstructionPair(
                    issuedPc, lowerWord, upperWord);
                ++irOpcodeCounts[
                    static_cast<size_t>(pair.upper.opcode)];
                ++irOpcodeCounts[
                    static_cast<size_t>(pair.lower.opcode)];
            }

            const VuRunResult referenceResult =
                reference.run(referenceContext, 1u);
            const VuRunResult candidateResult =
                candidate->run(candidateContext, 1u);
            differentialComparedPairs +=
                referenceResult.executedCycles;
            std::string stateDifference;
            const bool statesMatch = vuExecutionStatesEqual(
                referenceState, candidateState,
                &stateDifference);
            if (!sameDifferentialRunResult(
                    referenceResult, candidateResult,
                    recompilerDifferential) ||
                !statesMatch ||
                referenceData != candidateData ||
                referenceEffects.path1Packets() !=
                    candidateEffects.path1Packets())
            {
                std::cerr
                    << (recompilerDifferential
                            ? "recompiler"
                            : "IR")
                    << " differential mismatch at pair "
                    << instructionIndex
                    << " pc=0x" << std::hex << issuedPc
                    << " lower=0x" << lowerWord
                    << " (" << vuIrOpcodeName(pair.lower.opcode)
                    << ") upper=0x" << upperWord
                    << " (" << vuIrOpcodeName(pair.upper.opcode)
                    << ") reference_exit="
                    << vuExitReasonName(referenceResult.reason)
                    << " candidate_exit="
                    << vuExitReasonName(
                           candidateResult.reason);
                if (!statesMatch)
                    std::cerr << " state=" << stateDifference;
                if (referenceData != candidateData)
                    std::cerr << " data=VU";
                if (referenceEffects.path1Packets() !=
                    candidateEffects.path1Packets())
                {
                    std::cerr << " effects=PATH1";
                }
                std::cerr << "\n";
                return 1;
            }
        }

        if (recompilerDifferential)
            recompilerDiagnostics = recompiler.diagnostics();
        state = std::move(referenceState);
        std::memcpy(
            memory.getVU1Data(), referenceData.data(), vuSize);
        for (const std::vector<uint8_t> &packet :
             referenceEffects.path1Packets())
        {
            gifOutput.insert(
                gifOutput.end(), packet.begin(), packet.end());
        }
    }
    else if (scheduledWrites.empty())
    {
        unit.execute(
            memory.getVU1Code(), vuSize,
            memory.getVU1Data(), vuSize,
            gs, &memory, stateWords[2], stateWords[3], stateWords[4],
            maxCycles);
    }
    else
    {
        unit.execute(
            memory.getVU1Code(), vuSize,
            memory.getVU1Data(), vuSize,
            gs, &memory, stateWords[2], stateWords[3], stateWords[4], 0u);

        size_t nextWrite = 0u;
        for (uint64_t instructionIndex = 0u;
             instructionIndex < maxCycles && unit.isActive();
             ++instructionIndex)
        {
            while (nextWrite < scheduledWrites.size() &&
                   scheduledWrites[nextWrite].instructionIndex ==
                       instructionIndex)
            {
                const ScheduledRegisterWrite &write =
                    scheduledWrites[nextWrite++];
                if (write.kind == ScheduledWriteKind::Vi)
                {
                    state.vi[write.registerIndex] =
                        static_cast<int32_t>(write.value);
                }
                else
                {
                    std::memcpy(
                        &state.vf[write.registerIndex][write.lane],
                        &write.value, sizeof(write.value));
                }
            }
            unit.continueExecution(
                memory.getVU1Code(), vuSize,
                memory.getVU1Data(), vuSize,
                gs, &memory, 1u);
        }
    }

    if (unitBackendSpecified)
    {
        if (const VuRecompilerDiagnostics *const diagnostics =
                unit.recompilerDiagnosticsIfCreated())
        {
            recompilerDiagnostics = *diagnostics;
        }
    }

    if (arguments.size() >= 3u)
    {
        std::ofstream output(arguments[2], std::ios::binary);
        output.write(reinterpret_cast<const char *>(gifOutput.data()),
                     static_cast<std::streamsize>(gifOutput.size()));
    }
    if (arguments.size() >= 4u)
    {
        std::ofstream output(arguments[3], std::ios::binary);
        output.write(reinterpret_cast<const char *>(memory.getVU1Data()),
                     static_cast<std::streamsize>(vuSize));
    }

    std::cout << std::hex << std::setfill('0');
    std::cout << "{\"schema_version\":1";
    if (unitBackendSpecified)
    {
        std::cout
            << ",\"unit_backend_requested\":\""
            << vuBackendKindName(unit.requestedBackend())
            << "\",\"unit_backend_resolved\":\""
            << vuBackendKindName(unit.resolvedBackend())
            << "\"";
    }
    if (irDifferential)
    {
        std::cout
            << ",\"ir_differential\":true"
            << ",\"ir_compared_pairs\":" << std::dec
            << differentialComparedPairs
            << ",\"ir_opcodes\":{";
        bool firstOpcode = true;
        for (size_t index = 0u;
             index < irOpcodeCounts.size(); ++index)
        {
            if (irOpcodeCounts[index] == 0u)
                continue;
            std::cout
                << (firstOpcode ? "" : ",")
                << "\"" << vuIrOpcodeName(
                       static_cast<VuIrOpcode>(index))
                << "\":" << irOpcodeCounts[index];
            firstOpcode = false;
        }
        std::cout << "}" << std::hex;
    }
    if (recompilerDifferential ||
        (unitBackendSpecified &&
         unit.resolvedBackend() ==
             VuBackendKind::Recompiler))
    {
        if (recompilerDifferential)
        {
            std::cout
                << ",\"recompiler_differential\":true"
                << ",\"recompiler_compared_pairs\":"
                << std::dec << differentialComparedPairs;
        }
        std::cout
            << ",\"recompiler_diagnostics\":{"
            << std::dec
            << "\"native_entries\":"
            << recompilerDiagnostics.nativeEntries
            << ",\"native_pairs\":"
            << recompilerDiagnostics.nativePairs
            << ",\"inline_pairs\":"
            << recompilerDiagnostics.inlinePairs
            << ",\"helper_pairs\":"
            << recompilerDiagnostics.helperPairs
            << ",\"block_completes\":"
            << recompilerDiagnostics.blockCompletes
            << ",\"cycle_budget_exits\":"
            << recompilerDiagnostics.cycleBudgetExits
            << ",\"xgkick_exits\":"
            << recompilerDiagnostics.xgkickExits
            << ",\"xgkick_advance_helper_calls\":"
            << recompilerDiagnostics.xgkickAdvanceHelperCalls
            << ",\"unsupported_exits\":"
            << recompilerDiagnostics.unsupportedExits
            << ",\"code_invalidation_exits\":"
            << recompilerDiagnostics.codeInvalidationExits
            << ",\"fault_exits\":"
            << recompilerDiagnostics.faultExits
            << ",\"interpreter_fallback_pairs\":"
            << recompilerDiagnostics.interpreterFallbackPairs
            << ",\"code_image_identities\":"
            << recompilerDiagnostics.codeImageIdentities
            << ",\"code_image_reuses\":"
            << recompilerDiagnostics.codeImageReuses
            << ",\"code_image_catalog_evictions\":"
            << recompilerDiagnostics.codeImageCatalogEvictions
            << "}" << std::hex;
    }
    std::cout << ",\"pc\":\"0x"
              << std::setw(4) << state.pc << "\",\"gif_bytes\":"
              << std::dec << gifOutput.size() << std::hex << ",\"vi\":[";
    for (size_t index = 0; index < 16; ++index)
        std::cout << (index == 0 ? "" : ",") << "\"0x" << std::setw(4)
                  << (static_cast<uint32_t>(state.vi[index]) & 0xffffu) << "\"";
    std::cout << "],\"vf\":[";
    for (size_t index = 0; index < 32 * 4; ++index)
        std::cout << (index == 0 ? "" : ",") << "\"0x" << std::setw(8)
                  << floatBits((&state.vf[0][0])[index]) << "\"";
    std::cout << "],\"acc\":[";
    for (size_t index = 0; index < 4; ++index)
        std::cout << (index == 0 ? "" : ",") << "\"0x" << std::setw(8)
                  << floatBits(state.acc[index]) << "\"";
    std::cout << "],\"q\":\"0x" << std::setw(8) << floatBits(state.q)
              << "\",\"p\":\"0x" << std::setw(8) << floatBits(state.p)
              << "\",\"i\":\"0x" << std::setw(8) << floatBits(state.i)
              << "\",\"status\":\"0x" << std::setw(8) << state.status
              << "\",\"mac\":\"0x" << std::setw(8) << state.mac
              << "\",\"clip\":\"0x" << std::setw(8) << state.clip << "\"}\n";
    return 0;
}
