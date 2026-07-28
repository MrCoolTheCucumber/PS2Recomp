#include "MiniTest.h"
#include "runtime/ps2_perf_jitdump.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace
{
#pragma pack(push, 1)
    struct TestJitDumpHeader
    {
        uint32_t magic;
        uint32_t version;
        uint32_t totalSize;
        uint32_t elfMachine;
        uint32_t padding;
        uint32_t processId;
        uint64_t timestamp;
        uint64_t flags;
    };

    struct TestRecordHeader
    {
        uint32_t id;
        uint32_t totalSize;
        uint64_t timestamp;
    };

    struct TestCodeLoad
    {
        TestRecordHeader header;
        uint32_t processId;
        uint32_t threadId;
        uint64_t virtualAddress;
        uint64_t codeAddress;
        uint64_t codeSize;
        uint64_t codeIndex;
    };
#pragma pack(pop)

    static_assert(sizeof(TestJitDumpHeader) == 40u);
    static_assert(sizeof(TestRecordHeader) == 16u);
    static_assert(sizeof(TestCodeLoad) == 56u);

    class TemporaryDirectory
    {
    public:
        TemporaryDirectory()
        {
            const uint64_t nonce =
                static_cast<uint64_t>(
                    std::chrono::steady_clock::now()
                        .time_since_epoch()
                        .count()) ^
                static_cast<uint64_t>(
                    reinterpret_cast<uintptr_t>(this));
            path =
                std::filesystem::temp_directory_path() /
                ("ps2x-jitdump-test-" +
                 std::to_string(nonce));
            std::filesystem::create_directories(path);
        }

        ~TemporaryDirectory()
        {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }

        std::filesystem::path path;
    };

    template <typename Value>
    bool readValue(
        const std::vector<uint8_t> &bytes,
        size_t &offset, Value &value)
    {
        if (offset > bytes.size() ||
            sizeof(value) > bytes.size() - offset)
        {
            return false;
        }
        std::memcpy(
            &value, bytes.data() + offset,
            sizeof(value));
        offset += sizeof(value);
        return true;
    }

    struct ParsedLoad
    {
        TestCodeLoad record{};
        std::string name;
        std::vector<uint8_t> code;
    };

    bool processMapsContain(
        const std::filesystem::path &path)
    {
#if defined(__linux__)
        std::ifstream maps("/proc/self/maps");
        std::string line;
        const std::string expected = path.string();
        while (std::getline(maps, line))
        {
            if (line.find(expected) != std::string::npos)
                return true;
        }
#else
        (void)path;
#endif
        return false;
    }

    bool readLoad(
        const std::vector<uint8_t> &bytes,
        size_t &offset, ParsedLoad &load)
    {
        const size_t recordOffset = offset;
        if (!readValue(bytes, offset, load.record) ||
            load.record.header.id != 0u ||
            load.record.header.totalSize <
                sizeof(TestCodeLoad) + 2u)
        {
            return false;
        }
        const size_t recordEnd =
            recordOffset +
            load.record.header.totalSize;
        if (recordEnd < recordOffset ||
            recordEnd > bytes.size())
        {
            return false;
        }
        const auto nameEnd = std::find(
            bytes.begin() +
                static_cast<std::ptrdiff_t>(offset),
            bytes.begin() +
                static_cast<std::ptrdiff_t>(recordEnd),
            uint8_t{0u});
        if (nameEnd ==
            bytes.begin() +
                static_cast<std::ptrdiff_t>(recordEnd))
        {
            return false;
        }
        load.name.assign(
            reinterpret_cast<const char *>(
                bytes.data() + offset),
            static_cast<size_t>(
                nameEnd -
                (bytes.begin() +
                 static_cast<std::ptrdiff_t>(
                     offset))));
        offset =
            static_cast<size_t>(
                nameEnd - bytes.begin()) +
            1u;
        if (load.record.codeSize >
                std::numeric_limits<size_t>::max() ||
            static_cast<size_t>(
                load.record.codeSize) !=
                recordEnd - offset)
        {
            return false;
        }
        load.code.assign(
            bytes.begin() +
                static_cast<std::ptrdiff_t>(offset),
            bytes.begin() +
                static_cast<std::ptrdiff_t>(recordEnd));
        offset = recordEnd;
        return true;
    }
}

void register_ps2_perf_jitdump_tests()
{
    MiniTest::Case("PS2PerfJitDump", [](TestCase &tc)
    {
        tc.Run(
            "writer emits complete monotonic code-load records",
            [](TestCase &t)
            {
                if (!PS2PerfJitDumpWriter::supported())
                    return;

                TemporaryDirectory temporary;
                const std::filesystem::path output =
                    temporary.path / "jit-test.dump";
                PS2PerfJitDumpWriter writer;
                std::string diagnostic;
                t.IsTrue(
                    writer.open(output, &diagnostic),
                    "jitdump should open: " + diagnostic);
                if (!writer.isOpen())
                    return;
                t.IsTrue(
                    processMapsContain(output),
                    "jitdump executable marker should remain "
                    "mapped while the writer is open");

                const std::array<uint8_t, 3u> firstCode{
                    0x90u, 0x90u, 0xc3u};
                const std::array<uint8_t, 2u> secondCode{
                    0x90u, 0xc3u};
                uint64_t firstIndex = 0u;
                uint64_t secondIndex = 0u;
                t.IsTrue(
                    writer.registerCode(
                        firstCode.data(),
                        firstCode.size(),
                        "vu1-first", &firstIndex,
                        &diagnostic),
                    "first code load should write: " +
                        diagnostic);
                t.IsTrue(
                    writer.registerCode(
                        secondCode.data(),
                        secondCode.size(),
                        "vu1-second", &secondIndex,
                        &diagnostic),
                    "second code load should write: " +
                        diagnostic);
                t.Equals(
                    firstIndex, uint64_t{1u},
                    "first code index");
                t.Equals(
                    secondIndex, uint64_t{2u},
                    "second code index");
                t.IsTrue(
                    writer.close(&diagnostic),
                    "jitdump should close: " + diagnostic);
                t.IsFalse(
                    processMapsContain(output),
                    "jitdump executable marker should be "
                    "unmapped when the writer closes");

                std::ifstream input(
                    output, std::ios::binary);
                const std::vector<uint8_t> bytes{
                    std::istreambuf_iterator<char>(input),
                    std::istreambuf_iterator<char>()};
                size_t offset = 0u;
                TestJitDumpHeader header{};
                ParsedLoad first;
                ParsedLoad second;
                TestRecordHeader close{};
                t.IsTrue(
                    readValue(bytes, offset, header),
                    "jitdump header should be complete");
                t.Equals(
                    header.magic, uint32_t{0x4a695444u},
                    "jitdump magic");
                t.Equals(
                    header.version, uint32_t{1u},
                    "jitdump version");
                t.Equals(
                    header.totalSize,
                    uint32_t{
                        sizeof(TestJitDumpHeader)},
                    "jitdump header size");
                t.IsTrue(
                    header.elfMachine != 0u,
                    "jitdump ELF machine should be known");
                t.IsTrue(
                    readLoad(bytes, offset, first),
                    "first code load should parse");
                t.IsTrue(
                    readLoad(bytes, offset, second),
                    "second code load should parse");
                t.IsTrue(
                    readValue(bytes, offset, close),
                    "close record should parse");
                t.Equals(
                    close.id, uint32_t{3u},
                    "close record type");
                t.Equals(
                    close.totalSize,
                    uint32_t{
                        sizeof(TestRecordHeader)},
                    "close record size");
                t.Equals(
                    offset, bytes.size(),
                    "jitdump should contain no trailing bytes");
                t.IsTrue(
                    header.timestamp <
                        first.record.header.timestamp &&
                    first.record.header.timestamp <
                        second.record.header.timestamp &&
                    second.record.header.timestamp <
                        close.timestamp,
                    "jitdump timestamps should be strictly monotonic");
                t.Equals(
                    first.name, std::string("vu1-first"),
                    "first symbol name");
                t.Equals(
                    second.name,
                    std::string("vu1-second"),
                    "second symbol name");
                t.IsTrue(
                    first.code ==
                        std::vector<uint8_t>(
                            firstCode.begin(),
                            firstCode.end()),
                    "first code bytes should match");
                t.IsTrue(
                    second.code ==
                        std::vector<uint8_t>(
                            secondCode.begin(),
                            secondCode.end()),
                    "second code bytes should match");
                t.Equals(
                    first.record.codeAddress,
                    static_cast<uint64_t>(
                        reinterpret_cast<uintptr_t>(
                            firstCode.data())),
                    "first runtime address");
                t.Equals(
                    first.record.virtualAddress,
                    first.record.codeAddress,
                    "first VMA should match runtime address");
            });

        tc.Run(
            "writer rejects ambiguous symbol strings",
            [](TestCase &t)
            {
                if (!PS2PerfJitDumpWriter::supported())
                    return;

                TemporaryDirectory temporary;
                PS2PerfJitDumpWriter writer;
                std::string diagnostic;
                t.IsTrue(
                    writer.open(
                        temporary.path / "jit-test.dump",
                        &diagnostic),
                    "jitdump should open: " + diagnostic);
                const std::array<uint8_t, 1u> code{
                    0xc3u};
                const std::array<char, 3u> name{
                    'a', '\0', 'b'};
                uint64_t index = 7u;
                t.IsFalse(
                    writer.registerCode(
                        code.data(), code.size(),
                        std::string_view(
                            name.data(), name.size()),
                        &index, &diagnostic),
                    "embedded NUL should be rejected");
                t.Equals(
                    index, uint64_t{0u},
                    "rejected loads should not return an index");
            });
    });
}
