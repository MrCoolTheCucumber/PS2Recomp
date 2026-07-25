#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu1.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace
{
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
}

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 5)
    {
        std::cerr << "usage: vu1_replay FIXTURE_DIRECTORY INSTRUCTION_PAIRS "
                     "[GIF_OUTPUT] [FINAL_VU_DATA]\n";
        return 2;
    }

    const std::string directory = argv[1];
    const uint32_t maxCycles = static_cast<uint32_t>(std::stoul(argv[2]));
    std::array<uint8_t, PS2_VU1_DATA_SIZE> initialData{};
    std::array<uint8_t, PS2_VU1_CODE_SIZE> micro{};
    std::array<uint32_t, 159> stateWords{};
    if (!readFile(directory + "/ordinal-2256-data.bin", initialData.data(), initialData.size()) ||
        !readFile(directory + "/ordinal-2256-micro.bin", micro.data(), micro.size()) ||
        !readFile(directory + "/ordinal-2256-state.bin", stateWords.data(), stateWords.size() * sizeof(uint32_t)))
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
    std::memcpy(memory.getVU1Code(), micro.data(), micro.size());
    std::memcpy(memory.getVU1Data(), initialData.data(), initialData.size());

    VU1Interpreter interpreter;
    VU1State &state = interpreter.state();
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

    interpreter.execute(memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                        memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                        gs, &memory, stateWords[2], stateWords[3], stateWords[4],
                        maxCycles);

    if (argc >= 4)
    {
        std::ofstream output(argv[3], std::ios::binary);
        output.write(reinterpret_cast<const char *>(gifOutput.data()),
                     static_cast<std::streamsize>(gifOutput.size()));
    }
    if (argc == 5)
    {
        std::ofstream output(argv[4], std::ios::binary);
        output.write(reinterpret_cast<const char *>(memory.getVU1Data()),
                     static_cast<std::streamsize>(PS2_VU1_DATA_SIZE));
    }

    std::cout << std::hex << std::setfill('0');
    std::cout << "{\"pc\":\"0x" << std::setw(4) << state.pc << "\",\"vi\":[";
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
