#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_memory.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace
{
    bool readFile(const std::string &path, std::vector<uint8_t> &data)
    {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input)
            return false;

        const std::streampos end = input.tellg();
        if (end < 0)
            return false;

        data.resize(static_cast<size_t>(end));
        input.seekg(0, std::ios::beg);
        if (!data.empty())
            input.read(reinterpret_cast<char *>(data.data()),
                       static_cast<std::streamsize>(data.size()));
        return input.good() ||
               static_cast<size_t>(input.gcount()) == data.size();
    }

    bool writeFile(const std::string &path,
                   const uint8_t *data,
                   size_t size)
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output)
            return false;

        output.write(reinterpret_cast<const char *>(data),
                     static_cast<std::streamsize>(size));
        return output.good();
    }

    void printUsage()
    {
        std::cerr
            << "usage: gs_replay [--vram-in FILE] [--vram-out FILE] "
               "[--register ADDRESS=VALUE] [--register-file FILE] "
               "[--packet-sizes FILE] [--hash-trace FILE] "
               "[--batch-stream] "
               "GIF_PACKET [GIF_PACKET ...]\n";
    }

    bool readPacketSizes(const std::string &path,
                         std::vector<size_t> &packetSizes)
    {
        std::ifstream input(path);
        if (!input)
            return false;

        std::string line;
        while (std::getline(input, line))
        {
            if (const size_t comment = line.find('#');
                comment != std::string::npos)
            {
                line.erase(comment);
            }

            const size_t first = line.find_first_not_of(" \t\r");
            if (first == std::string::npos)
                continue;
            const size_t last = line.find_last_not_of(" \t\r");
            const std::string valueText =
                line.substr(first, last - first + 1u);

            errno = 0;
            char *end = nullptr;
            const unsigned long long parsed =
                std::strtoull(valueText.c_str(), &end, 0);
            if (errno != 0 || end == valueText.c_str() ||
                !end || *end != '\0' ||
                parsed > std::numeric_limits<size_t>::max())
            {
                return false;
            }
            packetSizes.push_back(static_cast<size_t>(parsed));
        }
        return input.eof() && !packetSizes.empty();
    }

    uint64_t fnv1a64(const uint8_t *data, size_t size)
    {
        uint64_t hash = 14695981039346656037ull;
        for (size_t index = 0; index < size; ++index)
        {
            hash ^= data[index];
            hash *= 1099511628211ull;
        }
        return hash;
    }

    bool parseRegisterAssignment(
        const std::string &text,
        std::pair<uint8_t, uint64_t> &assignment)
    {
        const size_t equals = text.find('=');
        if (equals == std::string::npos ||
            equals == 0u ||
            equals + 1u == text.size())
        {
            return false;
        }

        auto parseInteger = [](const std::string &part,
                               uint64_t &value) -> bool
        {
            errno = 0;
            char *end = nullptr;
            const unsigned long long parsed =
                std::strtoull(part.c_str(), &end, 0);
            if (errno != 0 || end == part.c_str() || !end || *end != '\0')
                return false;

            value = static_cast<uint64_t>(parsed);
            return true;
        };

        uint64_t address = 0u;
        uint64_t value = 0u;
        if (!parseInteger(text.substr(0u, equals), address) ||
            !parseInteger(text.substr(equals + 1u), value) ||
            address > 0xFFu)
        {
            return false;
        }

        assignment = {
            static_cast<uint8_t>(address),
            value,
        };
        return true;
    }

    bool readRegisterFile(
        const std::string &path,
        std::vector<std::pair<uint8_t, uint64_t>> &assignments)
    {
        std::ifstream input(path);
        if (!input)
            return false;

        std::string line;
        while (std::getline(input, line))
        {
            if (const size_t comment = line.find('#');
                comment != std::string::npos)
            {
                line.erase(comment);
            }

            const size_t first = line.find_first_not_of(" \t\r");
            if (first == std::string::npos)
                continue;
            const size_t last = line.find_last_not_of(" \t\r");

            std::pair<uint8_t, uint64_t> assignment{};
            if (!parseRegisterAssignment(
                    line.substr(first, last - first + 1u),
                    assignment))
            {
                return false;
            }
            assignments.push_back(assignment);
        }
        return input.eof();
    }
}

int main(int argc, char **argv)
{
    std::string vramInputPath;
    std::string vramOutputPath;
    std::string packetSizesPath;
    std::string hashTracePath;
    bool batchStream = false;
    std::vector<std::pair<uint8_t, uint64_t>> initialRegisters;
    std::vector<std::string> packetPaths;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--batch-stream")
        {
            batchStream = true;
        }
        else if (argument == "--vram-in" ||
            argument == "--vram-out" ||
            argument == "--register" ||
            argument == "--register-file" ||
            argument == "--packet-sizes" ||
            argument == "--hash-trace")
        {
            if (++index >= argc)
            {
                printUsage();
                return 2;
            }

            if (argument == "--vram-in")
                vramInputPath = argv[index];
            else if (argument == "--vram-out")
                vramOutputPath = argv[index];
            else if (argument == "--packet-sizes")
                packetSizesPath = argv[index];
            else if (argument == "--hash-trace")
                hashTracePath = argv[index];
            else if (argument == "--register-file")
            {
                if (!readRegisterFile(argv[index], initialRegisters))
                {
                    std::cerr << "failed to read register file: "
                              << argv[index] << '\n';
                    return 2;
                }
            }
            else
            {
                std::pair<uint8_t, uint64_t> assignment{};
                if (!parseRegisterAssignment(argv[index], assignment))
                {
                    std::cerr << "invalid register assignment: "
                              << argv[index] << '\n';
                    return 2;
                }
                initialRegisters.push_back(assignment);
            }
        }
        else if (argument == "--help" || argument == "-h")
        {
            printUsage();
            return 0;
        }
        else if (!argument.empty() && argument[0] == '-')
        {
            std::cerr << "unknown option: " << argument << '\n';
            printUsage();
            return 2;
        }
        else
        {
            packetPaths.push_back(argument);
        }
    }

    if (packetPaths.empty())
    {
        printUsage();
        return 2;
    }

    PS2Memory memory;
    if (!memory.initialize())
    {
        std::cerr << "failed to initialize memory\n";
        return 2;
    }

    GS gs;
    gs.init(memory.getGSVRAM(), static_cast<uint32_t>(PS2_GS_VRAM_SIZE),
            &memory.gs());

    if (!vramInputPath.empty())
    {
        std::vector<uint8_t> initialVram;
        if (!readFile(vramInputPath, initialVram))
        {
            std::cerr << "failed to read initial GS memory: "
                      << vramInputPath << '\n';
            return 2;
        }
        if (initialVram.size() != PS2_GS_VRAM_SIZE)
        {
            std::cerr << "initial GS memory must be exactly "
                      << PS2_GS_VRAM_SIZE << " bytes\n";
            return 2;
        }

        std::memcpy(memory.getGSVRAM(), initialVram.data(),
                    initialVram.size());
    }

    for (const auto &[address, value] : initialRegisters)
        gs.writeRegister(address, value);

    if (batchStream)
        gs.beginRenderBatch();
    struct RenderBatchScope
    {
        GS *gs = nullptr;
        ~RenderBatchScope()
        {
            if (gs)
                gs->endRenderBatch();
        }
    } renderBatchScope{batchStream ? &gs : nullptr};

    std::ofstream hashTrace;
    if (!hashTracePath.empty())
    {
        hashTrace.open(hashTracePath, std::ios::out | std::ios::trunc);
        if (!hashTrace)
        {
            std::cerr << "failed to open hash trace: "
                      << hashTracePath << '\n';
            return 2;
        }
        hashTrace << "index,fnv1a64\n";
    }

    std::vector<size_t> packetSizes;
    if (!packetSizesPath.empty() &&
        !readPacketSizes(packetSizesPath, packetSizes))
    {
        std::cerr << "failed to read packet sizes: "
                  << packetSizesPath << '\n';
        return 2;
    }
    if (!packetSizes.empty() && packetPaths.size() != 1u)
    {
        std::cerr << "--packet-sizes requires exactly one GIF_PACKET\n";
        return 2;
    }

    uint64_t totalBytes = 0u;
    uint64_t packetIndex = 0u;
    auto processPacket = [&](const uint8_t *data, size_t size) -> bool
    {
        if (size > std::numeric_limits<uint32_t>::max())
        {
            std::cerr << "GIF packet is too large\n";
            return false;
        }

        gs.processGIFPacket(data, static_cast<uint32_t>(size));
        totalBytes += size;
        if (hashTrace)
        {
            gs.flushRenderBatch();
            hashTrace << packetIndex << ','
                      << std::hex << std::setw(16) << std::setfill('0')
                      << fnv1a64(memory.getGSVRAM(), PS2_GS_VRAM_SIZE)
                      << std::dec << '\n';
        }
        ++packetIndex;
        return true;
    };

    for (const std::string &packetPath : packetPaths)
    {
        std::vector<uint8_t> packet;
        if (!readFile(packetPath, packet))
        {
            std::cerr << "failed to read GIF packet: " << packetPath << '\n';
            return 2;
        }

        if (packetSizes.empty())
        {
            if (!processPacket(packet.data(), packet.size()))
                return 2;
            continue;
        }

        size_t offset = 0u;
        for (size_t size : packetSizes)
        {
            if (offset > packet.size() || size > packet.size() - offset)
            {
                std::cerr << "packet sizes exceed GIF stream length\n";
                return 2;
            }
            if (!processPacket(packet.data() + offset, size))
                return 2;
            offset += size;
        }
        if (offset != packet.size())
        {
            std::cerr << "packet sizes do not consume GIF stream\n";
            return 2;
        }
    }

    if (batchStream)
        gs.flushRenderBatch();

    if (!vramOutputPath.empty() &&
        !writeFile(vramOutputPath, memory.getGSVRAM(), PS2_GS_VRAM_SIZE))
    {
        std::cerr << "failed to write final GS memory: "
                  << vramOutputPath << '\n';
        return 2;
    }

    std::cout << "{\"packets\":" << packetPaths.size()
              << ",\"bytes\":" << totalBytes << "}\n";
    return 0;
}
