#include "MiniTest.h"
#include "runtime/ps2_vu_executable_memory.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#if defined(__linux__)
#include <fstream>
#include <sstream>
#endif

namespace
{
#if defined(__linux__)
    std::string mappingPermissions(const void *address)
    {
        std::ifstream maps("/proc/self/maps");
        std::string line;
        const uintptr_t target =
            reinterpret_cast<uintptr_t>(address);
        while (std::getline(maps, line))
        {
            std::istringstream fields(line);
            std::string range;
            std::string permissions;
            fields >> range >> permissions;
            const size_t separator = range.find('-');
            if (separator == std::string::npos)
                continue;

            const uintptr_t begin = static_cast<uintptr_t>(
                std::stoull(
                    range.substr(0u, separator), nullptr, 16));
            const uintptr_t end = static_cast<uintptr_t>(
                std::stoull(
                    range.substr(separator + 1u), nullptr, 16));
            if (target >= begin && target < end)
                return permissions;
        }
        return {};
    }
#endif

    std::array<uint8_t, 8u> returnFortyTwoCode(
        size_t &usedBytes)
    {
        std::array<uint8_t, 8u> code{};
#if defined(__x86_64__) || defined(_M_X64)
        // mov eax, 42; ret
        code = {0xb8u, 0x2au, 0x00u, 0x00u,
                0x00u, 0xc3u, 0x00u, 0x00u};
        usedBytes = 6u;
#elif defined(__aarch64__) || defined(_M_ARM64)
        // mov w0, #42; ret
        code = {0x40u, 0x05u, 0x80u, 0x52u,
                0xc0u, 0x03u, 0x5fu, 0xd6u};
        usedBytes = 8u;
#else
        code[0] = 0u;
        usedBytes = 1u;
#endif
        return code;
    }
}

void register_ps2_vu_executable_memory_tests()
{
    MiniTest::Case("PS2VUExecutableMemory", [](TestCase &tc)
    {
        tc.Run(
            "allocation rejects invalid sizes without creating a mapping",
            [](TestCase &t)
            {
                std::string diagnostic;
                VuExecutableMemory empty =
                    VuExecutableMemory::allocate(0u, &diagnostic);
                t.IsTrue(empty.empty(),
                         "zero-byte allocation should remain empty");
                t.IsFalse(
                    diagnostic.empty(),
                    "zero-byte allocation should explain its failure");

                VuExecutableMemory overflow =
                    VuExecutableMemory::allocate(
                        std::numeric_limits<size_t>::max(),
                        &diagnostic);
                t.IsTrue(
                    overflow.empty(),
                    "overflowing allocation should remain empty");
                t.IsFalse(
                    diagnostic.empty(),
                    "overflowing allocation should explain its failure");
            });

        tc.Run(
            "native code transitions once from writable to executable",
            [](TestCase &t)
            {
                t.IsTrue(
                    VuExecutableMemory::supported(),
                    "the test host should support executable mappings");
                t.IsTrue(
                    VuExecutableMemory::pageSize() != 0u,
                    "the executable allocator should discover a page size");

                std::string diagnostic;
                VuExecutableMemory code =
                    VuExecutableMemory::allocate(8u, &diagnostic);
                t.IsFalse(code.empty(),
                          "a small executable allocation should succeed");
                t.Equals(
                    code.state(),
                    VuExecutableMemoryState::Writable,
                    "a new mapping should be writable");
                t.IsNotNull(
                    code.writableData(),
                    "a writable mapping should expose its emission pointer");
                t.IsNull(
                    code.executableData(),
                    "a writable mapping must not expose an executable pointer");
                t.IsTrue(
                    code.capacity() >= 8u &&
                        code.capacity() %
                                VuExecutableMemory::pageSize() ==
                            0u,
                    "the mapping should be page-sized");

#if defined(__linux__)
                const std::string writablePermissions =
                    mappingPermissions(code.writableData());
                t.IsTrue(
                    writablePermissions.find('w') !=
                        std::string::npos,
                    "the emission mapping should be writable");
                t.IsTrue(
                    writablePermissions.find('x') ==
                        std::string::npos,
                    "the emission mapping must not be executable");
#endif

                const uint8_t byte = 0u;
                t.IsFalse(
                    code.write(
                        code.capacity(), &byte, 1u, &diagnostic),
                    "a write beyond the mapping should fail");
                t.IsFalse(
                    code.write(0u, nullptr, 1u, &diagnostic),
                    "a nonempty null-source write should fail");
                t.IsFalse(
                    code.finalize(0u, &diagnostic),
                    "an empty native program should not finalize");
                t.Equals(
                    code.state(),
                    VuExecutableMemoryState::Writable,
                    "failed validation should leave the mapping writable");

                size_t usedBytes = 0u;
                const auto bytes = returnFortyTwoCode(usedBytes);
                t.IsTrue(
                    code.write(
                        0u, bytes.data(), usedBytes, &diagnostic),
                    "native bytes should copy into the writable mapping");
                t.IsTrue(
                    code.finalize(usedBytes, &diagnostic),
                    "the completed mapping should transition to RX");
                t.Equals(
                    code.state(),
                    VuExecutableMemoryState::Executable,
                    "a finalized mapping should be executable");
                t.Equals(
                    code.usedSize(), usedBytes,
                    "the allocation should retain its emitted size");
                t.IsNull(
                    code.writableData(),
                    "an executable mapping must not expose a writable pointer");
                t.IsNotNull(
                    code.executableData(),
                    "an executable mapping should expose its entry bytes");
                t.IsNull(
                    code.executableAddress(usedBytes),
                    "an entry cannot point past emitted native code");
                t.IsFalse(
                    code.write(0u, &byte, 1u, &diagnostic),
                    "finalized native code must be immutable");
                t.IsFalse(
                    code.finalize(usedBytes, &diagnostic),
                    "an RX mapping must not transition a second time");

#if defined(__linux__)
                const std::string executablePermissions =
                    mappingPermissions(code.executableData());
                t.IsTrue(
                    executablePermissions.find('x') !=
                        std::string::npos,
                    "the finalized mapping should be executable");
                t.IsTrue(
                    executablePermissions.find('w') ==
                        std::string::npos,
                    "the finalized mapping must not remain writable");
#endif

#if defined(__x86_64__) || defined(_M_X64) || \
    defined(__aarch64__) || defined(_M_ARM64)
                using NativeFunction = int (*)();
                const void *entry = code.executableAddress();
                NativeFunction function = nullptr;
                static_assert(sizeof(function) == sizeof(entry));
                std::memcpy(&function, &entry, sizeof(function));
                t.Equals(
                    function(), 42,
                    "instruction-cache maintenance should expose emitted code");
#endif
            });

        tc.Run(
            "move operations transfer mapping ownership",
            [](TestCase &t)
            {
                std::string diagnostic;
                VuExecutableMemory source =
                    VuExecutableMemory::allocate(1u, &diagnostic);
                const uint8_t byte = 0xc3u;
                t.IsTrue(
                    source.write(0u, &byte, 1u, &diagnostic),
                    "source code byte should be writable");
                t.IsTrue(
                    source.finalize(1u, &diagnostic),
                    "source mapping should finalize");
                const void *const original =
                    source.executableAddress();

                VuExecutableMemory moved(std::move(source));
                t.IsTrue(
                    source.empty(),
                    "a moved-from allocation should become empty");
                t.Equals(
                    moved.executableAddress(), original,
                    "move construction should retain the mapping");

                VuExecutableMemory destination =
                    VuExecutableMemory::allocate(1u, &diagnostic);
                destination = std::move(moved);
                t.IsTrue(
                    moved.empty(),
                    "move assignment should empty its source");
                t.Equals(
                    destination.executableAddress(), original,
                    "move assignment should transfer mapping ownership");
            });
    });
}
