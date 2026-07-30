#include "MiniTest.h"
#include "ps2_runtime.h"
#include "ps2_syscalls.h"
#include "ps2_stubs.h"
#include "Kernel/Stubs/CD.h"
#include "Kernel/Stubs/MemoryCard.h"

#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>
#include <chrono>

using namespace ps2_syscalls;

namespace
{
    // Guest memory address ranges for test data
    constexpr uint32_t GUEST_STRING_AREA_START = 0x1000;
    constexpr uint32_t GUEST_BUFFER_AREA_START = 0x2000;
    constexpr uint32_t GUEST_STACK_AREA_START = 0x6000;
    constexpr uint32_t GUEST_MC_SYNC_CMD_ADDR = GUEST_BUFFER_AREA_START + 0x1C00;
    constexpr uint32_t GUEST_MC_SYNC_RESULT_ADDR = GUEST_BUFFER_AREA_START + 0x1C04;
    constexpr uint32_t GUEST_MC_TABLE_ADDR = GUEST_BUFFER_AREA_START + 0x2000;
    
    // Common file I/O flag combinations
    constexpr uint32_t PS2_FIO_WRITE_CREATE_TRUNC = 
        PS2_FIO_O_WRONLY | PS2_FIO_O_CREAT | PS2_FIO_O_TRUNC;

    struct SceMcStDateTime
    {
        uint8_t resv2;
        uint8_t sec;
        uint8_t min;
        uint8_t hour;
        uint8_t day;
        uint8_t month;
        uint16_t year;
    };

    struct SceMcTblGetDir
    {
        SceMcStDateTime create;
        SceMcStDateTime modify;
        uint32_t fileSizeByte;
        uint16_t attrFile;
        uint16_t reserve1;
        uint32_t reserve2;
        uint32_t pdaAplNo;
        char entryName[32];
    };

    static_assert(sizeof(SceMcTblGetDir) == 64, "sceMcTblGetDir size mismatch");

    void setRegU32(R5900Context &ctx, int reg, uint32_t value)
    {
        ctx.r[reg] = _mm_set_epi64x(0, static_cast<int64_t>(value));
    }

    int32_t getRegS32(const R5900Context *ctx, int reg)
    {
        return static_cast<int32_t>(::getRegU32(ctx, reg));
    }

    void writeGuestString(uint8_t *rdram, uint32_t addr, const std::string &value)
    {
        std::memcpy(rdram + addr, value.c_str(), value.size() + 1);
    }

    void writeGuestU32(uint8_t *rdram, uint32_t addr, uint32_t value)
    {
        std::memcpy(rdram + addr, &value, sizeof(value));
    }

    int32_t readGuestS32(const uint8_t *rdram, uint32_t addr)
    {
        int32_t value = 0;
        std::memcpy(&value, rdram + addr, sizeof(value));
        return value;
    }

    uint32_t readGuestU32(const uint8_t *rdram, uint32_t addr)
    {
        uint32_t value = 0;
        std::memcpy(&value, rdram + addr, sizeof(value));
        return value;
    }

    void clearContext(R5900Context &ctx)
    {
        std::memset(&ctx, 0, sizeof(ctx));
    }

    int32_t syncMc(
        std::vector<uint8_t> &rdram,
        PS2Runtime *runtime,
        int32_t *cmdOut = nullptr)
    {
        R5900Context syncCtx{};
        setRegU32(syncCtx, 4, 0u);
        setRegU32(syncCtx, 5, GUEST_MC_SYNC_CMD_ADDR);
        setRegU32(syncCtx, 6, GUEST_MC_SYNC_RESULT_ADDR);
        ps2_stubs::sceMcSync(
            rdram.data(), &syncCtx, runtime);

        if (cmdOut)
        {
            *cmdOut = readGuestS32(rdram.data(), GUEST_MC_SYNC_CMD_ADDR);
        }
        return readGuestS32(rdram.data(), GUEST_MC_SYNC_RESULT_ADDR);
    }

    struct TempPaths
    {
        std::filesystem::path base;
        std::filesystem::path mcRoot;
        std::filesystem::path cdRoot;

        ~TempPaths()
        {
            std::error_code ec;
            std::filesystem::remove_all(base, ec);
        }
    };

    TempPaths makeTempPaths()
    {
        TempPaths paths;
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        paths.base = std::filesystem::temp_directory_path()
                   / ("ps2recomp-mc0-" + std::to_string(now));
        paths.mcRoot = paths.base / "mcroot";
        paths.cdRoot = paths.base / "cdroot";
        std::filesystem::create_directories(paths.mcRoot);
        std::filesystem::create_directories(paths.cdRoot);
        return paths;
    }

    struct HostWorkingDirectoryGuard
    {
        HostWorkingDirectoryGuard()
            : original(
                  std::filesystem::current_path())
        {
        }

        ~HostWorkingDirectoryGuard()
        {
            std::error_code ec;
            std::filesystem::current_path(
                original, ec);
        }

        std::filesystem::path original;
    };

    struct TestContext
    {
        TempPaths paths;
        std::vector<uint8_t> rdram;
        R5900Context ctx;
        PS2Runtime runtime;

        TestContext() : paths(makeTempPaths()), rdram(PS2_RAM_SIZE, 0)
        {
            PS2Runtime::IoPaths ioPaths;
            ioPaths.elfDirectory = paths.cdRoot;
            ioPaths.hostRoot = paths.cdRoot;
            ioPaths.cdRoot = paths.cdRoot;
            ioPaths.mcRoot = paths.mcRoot;
            runtime.configureIoPaths(ioPaths);
        }
    };
}

void register_ps2_runtime_io_tests()
{
    MiniTest::Case("PS2RuntimeIO", [](TestCase &tc)
    {
        tc.Run("file descriptors and CD state are isolated per runtime", [](TestCase &t)
        {
            TestContext test;
            PS2Runtime first;
            PS2Runtime second;
            const PS2Runtime::IoPaths ioPaths =
                test.runtime.ioPaths();
            first.configureIoPaths(ioPaths);
            second.configureIoPaths(ioPaths);
            std::vector<uint8_t> secondRdram(
                PS2_RAM_SIZE, 0u);
            R5900Context secondCtx{};

            const std::filesystem::path hostFile =
                test.paths.cdRoot / "shared.bin";
            {
                std::ofstream out(
                    hostFile, std::ios::binary);
                out << "runtime-io";
            }

            constexpr uint32_t kPathAddr =
                GUEST_STRING_AREA_START + 0xC00u;
            writeGuestString(
                test.rdram.data(),
                kPathAddr,
                "host0:/shared.bin");
            writeGuestString(
                secondRdram.data(),
                kPathAddr,
                "host0:/shared.bin");

            setRegU32(test.ctx, 4, kPathAddr);
            setRegU32(test.ctx, 5, PS2_FIO_O_RDONLY);
            fioOpen(
                test.rdram.data(),
                &test.ctx,
                &first);
            const int32_t firstFd =
                getRegS32(&test.ctx, 2);

            setRegU32(secondCtx, 4, kPathAddr);
            setRegU32(secondCtx, 5, PS2_FIO_O_RDONLY);
            fioOpen(
                secondRdram.data(),
                &secondCtx,
                &second);
            const int32_t secondFd =
                getRegS32(&secondCtx, 2);

            t.IsTrue(
                firstFd >= 3 && secondFd >= 3,
                "both runtimes should open the host file");
            t.Equals(
                secondFd,
                firstFd,
                "each runtime should independently allocate its first file descriptor");

            ps2_syscalls::notifyRuntimeStop(&first);

            constexpr uint32_t kReadAddr =
                GUEST_BUFFER_AREA_START + 0x1E00u;
            setRegU32(
                test.ctx,
                4,
                static_cast<uint32_t>(firstFd));
            setRegU32(test.ctx, 5, kReadAddr);
            setRegU32(test.ctx, 6, 1u);
            fioRead(
                test.rdram.data(),
                &test.ctx,
                &first);
            t.Equals(
                getRegS32(&test.ctx, 2),
                -1,
                "stopping a runtime should close its file descriptors");

            setRegU32(
                secondCtx,
                4,
                static_cast<uint32_t>(secondFd));
            setRegU32(secondCtx, 5, kReadAddr);
            setRegU32(secondCtx, 6, 1u);
            fioRead(
                secondRdram.data(),
                &secondCtx,
                &second);
            t.Equals(
                getRegS32(&secondCtx, 2),
                1,
                "stopping one runtime must not close another runtime's file descriptors");

            setRegU32(
                secondCtx,
                4,
                static_cast<uint32_t>(secondFd));
            fioClose(
                secondRdram.data(),
                &secondCtx,
                &second);

            clearContext(test.ctx);
            clearContext(secondCtx);
            ps2_stubs::sceCdInit(
                test.rdram.data(),
                &test.ctx,
                &first);
            ps2_stubs::sceCdStatus(
                secondRdram.data(),
                &secondCtx,
                &second);
            t.Equals(
                getRegS32(&secondCtx, 2),
                0,
                "initializing one runtime must not initialize another runtime's CD device");
            ps2_stubs::sceCdInit(
                secondRdram.data(),
                &secondCtx,
                &second);

            constexpr uint32_t kFirstLbn = 0x1111u;
            constexpr uint32_t kSecondLbn = 0x2222u;
            setRegU32(test.ctx, 4, kFirstLbn);
            ps2_stubs::sceCdSeek(
                test.rdram.data(),
                &test.ctx,
                &first);
            setRegU32(secondCtx, 4, kSecondLbn);
            ps2_stubs::sceCdSeek(
                secondRdram.data(),
                &secondCtx,
                &second);

            ps2_stubs::sceCdGetReadPos(
                test.rdram.data(),
                &test.ctx,
                &first);
            t.Equals(
                ::getRegU32(&test.ctx, 2),
                kFirstLbn,
                "the first runtime should retain its CD stream position");
            ps2_stubs::sceCdGetReadPos(
                secondRdram.data(),
                &secondCtx,
                &second);
            t.Equals(
                ::getRegU32(&secondCtx, 2),
                kSecondLbn,
                "the second runtime should retain its CD stream position");

            ps2_syscalls::notifyRuntimeStop(&first);
            ps2_stubs::sceCdGetReadPos(
                test.rdram.data(),
                &test.ctx,
                &first);
            t.Equals(
                ::getRegU32(&test.ctx, 2),
                0u,
                "stopping a runtime should reset its CD stream position");
            ps2_stubs::sceCdGetReadPos(
                secondRdram.data(),
                &secondCtx,
                &second);
            t.Equals(
                ::getRegU32(&secondCtx, 2),
                kSecondLbn,
                "stopping one runtime must not reset another runtime's CD stream position");

            ps2_stubs::sceCdStatus(
                test.rdram.data(),
                &test.ctx,
                &first);
            t.Equals(
                getRegS32(&test.ctx, 2),
                0,
                "stopping a runtime should reset its CD initialization state");
            ps2_stubs::sceCdStatus(
                secondRdram.data(),
                &secondCtx,
                &second);
            t.Equals(
                getRegS32(&secondCtx, 2),
                6,
                "stopping one runtime must not reset another runtime's CD initialization state");
        });

        tc.Run("IO path configuration is isolated per runtime", [](TestCase &t)
        {
            TempPaths firstPaths = makeTempPaths();
            TempPaths secondPaths = makeTempPaths();
            PS2Runtime first;
            PS2Runtime second;
            std::vector<uint8_t> firstRdram(
                PS2_RAM_SIZE, 0u);
            std::vector<uint8_t> secondRdram(
                PS2_RAM_SIZE, 0u);
            R5900Context firstCtx{};
            R5900Context secondCtx{};

            const std::filesystem::path firstFile =
                firstPaths.cdRoot / "identity.bin";
            const std::filesystem::path secondFile =
                secondPaths.cdRoot / "identity.bin";
            {
                std::ofstream out(
                    firstFile, std::ios::binary);
                out.put('1');
            }
            {
                std::ofstream out(
                    secondFile, std::ios::binary);
                out.put('2');
            }

            PS2Runtime::IoPaths firstIoPaths;
            firstIoPaths.elfDirectory =
                firstPaths.cdRoot;
            firstIoPaths.hostRoot =
                firstPaths.cdRoot;
            firstIoPaths.cdRoot =
                firstPaths.cdRoot;
            firstIoPaths.mcRoot =
                firstPaths.mcRoot;
            first.configureIoPaths(firstIoPaths);

            PS2Runtime::IoPaths secondIoPaths;
            secondIoPaths.elfDirectory =
                secondPaths.cdRoot;
            secondIoPaths.hostRoot =
                secondPaths.cdRoot;
            secondIoPaths.cdRoot =
                secondPaths.cdRoot;
            secondIoPaths.mcRoot =
                secondPaths.mcRoot;
            second.configureIoPaths(secondIoPaths);

            t.Equals(
                first.ioPaths().hostRoot.string(),
                firstPaths.cdRoot.string(),
                "configuring a peer runtime must not replace the first runtime's host root");
            t.Equals(
                second.ioPaths().hostRoot.string(),
                secondPaths.cdRoot.string(),
                "the second runtime should retain its own host root");

            const ps2_stubs::CdDebugSnapshot firstCd =
                ps2_stubs::getCdDebugSnapshot(&first);
            const ps2_stubs::CdDebugSnapshot secondCd =
                ps2_stubs::getCdDebugSnapshot(&second);
            t.Equals(
                firstCd.cdRoot.string(),
                firstPaths.cdRoot.string(),
                "CD inspection should resolve the first runtime's CD root");
            t.Equals(
                secondCd.cdRoot.string(),
                secondPaths.cdRoot.string(),
                "CD inspection should resolve the second runtime's CD root");

            const ps2_stubs::MemoryCardDebugSnapshot firstMc =
                ps2_stubs::getMemoryCardDebugSnapshot(
                    &first);
            const ps2_stubs::MemoryCardDebugSnapshot secondMc =
                ps2_stubs::getMemoryCardDebugSnapshot(
                    &second);
            t.Equals(
                firstMc.ports[0].rootPath.string(),
                firstPaths.mcRoot.string(),
                "memory-card inspection should resolve the first runtime's card root");
            t.Equals(
                secondMc.ports[0].rootPath.string(),
                secondPaths.mcRoot.string(),
                "memory-card inspection should resolve the second runtime's card root");

            constexpr uint32_t kPathAddr =
                GUEST_STRING_AREA_START + 0xC80u;
            constexpr uint32_t kReadAddr =
                GUEST_BUFFER_AREA_START + 0x1E80u;
            writeGuestString(
                firstRdram.data(),
                kPathAddr,
                "host0:/identity.bin");
            writeGuestString(
                secondRdram.data(),
                kPathAddr,
                "host0:/identity.bin");

            const auto readIdentity =
                [&](PS2Runtime &runtime,
                    std::vector<uint8_t> &rdram,
                    R5900Context &ctx) -> char
            {
                setRegU32(ctx, 4, kPathAddr);
                setRegU32(ctx, 5, PS2_FIO_O_RDONLY);
                fioOpen(
                    rdram.data(),
                    &ctx,
                    &runtime);
                const int32_t fd =
                    getRegS32(&ctx, 2);
                if (fd < 0)
                {
                    return '\0';
                }

                setRegU32(
                    ctx,
                    4,
                    static_cast<uint32_t>(fd));
                setRegU32(ctx, 5, kReadAddr);
                setRegU32(ctx, 6, 1u);
                fioRead(
                    rdram.data(),
                    &ctx,
                    &runtime);
                const int32_t bytesRead =
                    getRegS32(&ctx, 2);

                setRegU32(
                    ctx,
                    4,
                    static_cast<uint32_t>(fd));
                fioClose(
                    rdram.data(),
                    &ctx,
                    &runtime);
                return bytesRead == 1
                           ? static_cast<char>(
                                 rdram[kReadAddr])
                           : '\0';
            };

            t.Equals(
                readIdentity(
                    first,
                    firstRdram,
                    firstCtx),
                '1',
                "the first runtime should resolve host0 through its own root");
            t.Equals(
                readIdentity(
                    second,
                    secondRdram,
                    secondCtx),
                '2',
                "the second runtime should resolve host0 through its own root");
        });

        tc.Run("guest working directories are isolated per runtime", [](TestCase &t)
        {
            HostWorkingDirectoryGuard hostCwd;
            TempPaths firstPaths = makeTempPaths();
            TempPaths secondPaths = makeTempPaths();
            PS2Runtime first;
            PS2Runtime second;
            std::vector<uint8_t> firstRdram(
                PS2_RAM_SIZE, 0u);
            std::vector<uint8_t> secondRdram(
                PS2_RAM_SIZE, 0u);
            R5900Context firstCtx{};
            R5900Context secondCtx{};

            std::filesystem::create_directories(
                firstPaths.cdRoot / "subdir");
            std::filesystem::create_directories(
                secondPaths.cdRoot / "subdir");
            {
                std::ofstream out(
                    firstPaths.cdRoot /
                        "relative.bin",
                    std::ios::binary);
                out.put('r');
            }
            {
                std::ofstream out(
                    firstPaths.cdRoot / "subdir" /
                        "relative.bin",
                    std::ios::binary);
                out.put('1');
            }
            {
                std::ofstream out(
                    secondPaths.cdRoot /
                        "relative.bin",
                    std::ios::binary);
                out.put('2');
            }
            {
                std::ofstream out(
                    secondPaths.cdRoot / "subdir" /
                        "relative.bin",
                    std::ios::binary);
                out.put('3');
            }

            PS2Runtime::IoPaths firstIoPaths;
            firstIoPaths.elfDirectory =
                firstPaths.cdRoot;
            firstIoPaths.hostRoot =
                firstPaths.cdRoot;
            firstIoPaths.cdRoot =
                firstPaths.cdRoot;
            firstIoPaths.mcRoot =
                firstPaths.mcRoot;
            first.configureIoPaths(firstIoPaths);

            PS2Runtime::IoPaths secondIoPaths;
            secondIoPaths.elfDirectory =
                secondPaths.cdRoot;
            secondIoPaths.hostRoot =
                secondPaths.cdRoot;
            secondIoPaths.cdRoot =
                secondPaths.cdRoot;
            secondIoPaths.mcRoot =
                secondPaths.mcRoot;
            second.configureIoPaths(secondIoPaths);

            constexpr uint32_t kDirectoryAddr =
                GUEST_STRING_AREA_START + 0xD80u;
            constexpr uint32_t kPathAddr =
                GUEST_STRING_AREA_START + 0xE80u;
            constexpr uint32_t kReadAddr =
                GUEST_BUFFER_AREA_START + 0x1F80u;
            writeGuestString(
                firstRdram.data(),
                kDirectoryAddr,
                "host0:/subdir");
            writeGuestString(
                secondRdram.data(),
                kDirectoryAddr,
                "host0:/subdir");
            writeGuestString(
                firstRdram.data(),
                kPathAddr,
                "host0:relative.bin");
            writeGuestString(
                secondRdram.data(),
                kPathAddr,
                "host0:relative.bin");

            const auto changeDirectory =
                [&](PS2Runtime &runtime,
                    std::vector<uint8_t> &rdram,
                    R5900Context &ctx)
            {
                clearContext(ctx);
                setRegU32(
                    ctx, 4, kDirectoryAddr);
                fioChdir(
                    rdram.data(),
                    &ctx,
                    &runtime);
                return getRegS32(&ctx, 2);
            };
            const auto readRelative =
                [&](PS2Runtime &runtime,
                    std::vector<uint8_t> &rdram,
                    R5900Context &ctx) -> char
            {
                clearContext(ctx);
                setRegU32(ctx, 4, kPathAddr);
                setRegU32(
                    ctx, 5, PS2_FIO_O_RDONLY);
                fioOpen(
                    rdram.data(),
                    &ctx,
                    &runtime);
                const int32_t fd =
                    getRegS32(&ctx, 2);
                if (fd < 0)
                {
                    return '\0';
                }

                setRegU32(
                    ctx,
                    4,
                    static_cast<uint32_t>(fd));
                setRegU32(ctx, 5, kReadAddr);
                setRegU32(ctx, 6, 1u);
                fioRead(
                    rdram.data(),
                    &ctx,
                    &runtime);
                const int32_t bytesRead =
                    getRegS32(&ctx, 2);

                setRegU32(
                    ctx,
                    4,
                    static_cast<uint32_t>(fd));
                fioClose(
                    rdram.data(),
                    &ctx,
                    &runtime);
                return bytesRead == 1
                           ? static_cast<char>(
                                 rdram[kReadAddr])
                           : '\0';
            };

            t.Equals(
                changeDirectory(
                    first,
                    firstRdram,
                    firstCtx),
                0,
                "the first runtime should accept its guest directory");
            t.Equals(
                std::filesystem::current_path().string(),
                hostCwd.original.string(),
                "fioChdir must not change the host process working directory");
            t.Equals(
                readRelative(
                    first,
                    firstRdram,
                    firstCtx),
                '1',
                "the first runtime should resolve relative host paths through its guest directory");
            t.Equals(
                readRelative(
                    second,
                    secondRdram,
                    secondCtx),
                '2',
                "changing one runtime's guest directory must not affect its peer");

            t.Equals(
                changeDirectory(
                    second,
                    secondRdram,
                    secondCtx),
                0,
                "the second runtime should accept its own guest directory");
            ps2_syscalls::notifyRuntimeStop(&first);
            t.Equals(
                readRelative(
                    first,
                    firstRdram,
                    firstCtx),
                'r',
                "stopping a runtime should reset its guest directory");
            t.Equals(
                readRelative(
                    second,
                    secondRdram,
                    secondCtx),
                '3',
                "stopping one runtime must preserve its peer's guest directory");

            std::error_code restoreError;
            std::filesystem::current_path(
                hostCwd.original,
                restoreError);
        });

        tc.Run("libc and memory-card handles are isolated per runtime", [](TestCase &t)
        {
            TestContext test;
            PS2Runtime first;
            PS2Runtime second;
            const PS2Runtime::IoPaths ioPaths =
                test.runtime.ioPaths();
            first.configureIoPaths(ioPaths);
            second.configureIoPaths(ioPaths);
            std::vector<uint8_t> secondRdram(
                PS2_RAM_SIZE, 0u);
            R5900Context secondCtx{};

            constexpr uint32_t kLibcPathAddr =
                GUEST_STRING_AREA_START + 0xD00u;
            constexpr uint32_t kLibcModeAddr =
                GUEST_STRING_AREA_START + 0xE00u;
            constexpr uint32_t kLibcReadAddr =
                GUEST_BUFFER_AREA_START + 0x1F00u;
            const std::filesystem::path libcPath =
                test.paths.base / "libc-shared.bin";
            {
                std::ofstream out(libcPath, std::ios::binary);
                out << "runtime-libc";
            }

            writeGuestString(
                test.rdram.data(),
                kLibcPathAddr,
                libcPath.string());
            writeGuestString(
                secondRdram.data(),
                kLibcPathAddr,
                libcPath.string());
            writeGuestString(
                test.rdram.data(),
                kLibcModeAddr,
                "rb");
            writeGuestString(
                secondRdram.data(),
                kLibcModeAddr,
                "rb");

            setRegU32(test.ctx, 4, kLibcPathAddr);
            setRegU32(test.ctx, 5, kLibcModeAddr);
            ps2_stubs::fopen(
                test.rdram.data(),
                &test.ctx,
                &first);
            const uint32_t firstLibcHandle =
                ::getRegU32(&test.ctx, 2);

            setRegU32(secondCtx, 4, kLibcPathAddr);
            setRegU32(secondCtx, 5, kLibcModeAddr);
            ps2_stubs::fopen(
                secondRdram.data(),
                &secondCtx,
                &second);
            const uint32_t secondLibcHandle =
                ::getRegU32(&secondCtx, 2);

            t.IsTrue(
                firstLibcHandle != 0u &&
                    secondLibcHandle != 0u,
                "both runtimes should open a libc stream");
            t.Equals(
                secondLibcHandle,
                firstLibcHandle,
                "each runtime should independently allocate its first libc stream handle");

            clearContext(test.ctx);
            setRegU32(test.ctx, 4, 1u);
            ps2_stubs::srand(
                test.rdram.data(),
                &test.ctx,
                &first);
            clearContext(secondCtx);
            setRegU32(secondCtx, 4, 1u);
            ps2_stubs::srand(
                secondRdram.data(),
                &secondCtx,
                &second);

            clearContext(test.ctx);
            ps2_stubs::rand(
                test.rdram.data(),
                &test.ctx,
                &first);
            const uint32_t firstRandom =
                ::getRegU32(&test.ctx, 2);
            clearContext(test.ctx);
            ps2_stubs::rand(
                test.rdram.data(),
                &test.ctx,
                &first);
            const uint32_t firstNextRandom =
                ::getRegU32(&test.ctx, 2);
            clearContext(secondCtx);
            ps2_stubs::rand(
                secondRdram.data(),
                &secondCtx,
                &second);
            const uint32_t secondRandom =
                ::getRegU32(&secondCtx, 2);

            t.Equals(
                firstRandom,
                0x41C67EA6u,
                "libc rand should match the guest LCG from seed one");
            t.Equals(
                secondRandom,
                firstRandom,
                "each runtime should independently produce its first libc random value");
            t.IsTrue(
                firstNextRandom != firstRandom,
                "advancing one runtime should advance its own libc random sequence");

            clearContext(test.ctx);
            setRegU32(test.ctx, 5, 7u);
            ps2_stubs::mcSetFileSelectWindowCursol(
                test.rdram.data(),
                &test.ctx,
                &first);
            clearContext(secondCtx);
            setRegU32(secondCtx, 5, 4u);
            ps2_stubs::mcSetFileSelectWindowCursol(
                secondRdram.data(),
                &secondCtx,
                &second);

            clearContext(test.ctx);
            ps2_stubs::mcGetFileSelectWindowCursol(
                test.rdram.data(),
                &test.ctx,
                &first);
            t.Equals(
                getRegS32(&test.ctx, 2),
                7,
                "the first runtime should retain its memory-card UI cursor");
            clearContext(secondCtx);
            ps2_stubs::mcGetFileSelectWindowCursol(
                secondRdram.data(),
                &secondCtx,
                &second);
            t.Equals(
                getRegS32(&secondCtx, 2),
                4,
                "the second runtime should retain its memory-card UI cursor");

            constexpr uint32_t kFirstMcPathAddr =
                GUEST_STRING_AREA_START + 0xF00u;
            constexpr uint32_t kSecondMcPathAddr =
                GUEST_STRING_AREA_START + 0x1000u;
            writeGuestString(
                test.rdram.data(),
                kFirstMcPathAddr,
                "/first-runtime.bin");
            writeGuestString(
                secondRdram.data(),
                kSecondMcPathAddr,
                "/second-runtime.bin");

            clearContext(test.ctx);
            setRegU32(test.ctx, 4, 0u);
            setRegU32(test.ctx, 5, 0u);
            setRegU32(test.ctx, 6, kFirstMcPathAddr);
            setRegU32(
                test.ctx,
                7,
                PS2_FIO_O_RDWR |
                    PS2_FIO_O_CREAT |
                    PS2_FIO_O_TRUNC);
            ps2_stubs::sceMcOpen(
                test.rdram.data(),
                &test.ctx,
                &first);
            R5900Context peerSyncCtx{};
            setRegU32(peerSyncCtx, 4, 0u);
            setRegU32(
                peerSyncCtx,
                5,
                GUEST_MC_SYNC_CMD_ADDR);
            setRegU32(
                peerSyncCtx,
                6,
                GUEST_MC_SYNC_RESULT_ADDR);
            ps2_stubs::sceMcSync(
                secondRdram.data(),
                &peerSyncCtx,
                &second);
            t.Equals(
                getRegS32(&peerSyncCtx, 2),
                -1,
                "one runtime must not consume another runtime's pending memory-card command");
            const int32_t firstMcFd =
                syncMc(test.rdram, &first);

            clearContext(secondCtx);
            setRegU32(secondCtx, 4, 0u);
            setRegU32(secondCtx, 5, 0u);
            setRegU32(secondCtx, 6, kSecondMcPathAddr);
            setRegU32(
                secondCtx,
                7,
                PS2_FIO_O_RDWR |
                    PS2_FIO_O_CREAT |
                    PS2_FIO_O_TRUNC);
            ps2_stubs::sceMcOpen(
                secondRdram.data(),
                &secondCtx,
                &second);
            const int32_t secondMcFd =
                syncMc(secondRdram, &second);

            t.IsTrue(
                firstMcFd > 0 && secondMcFd > 0,
                "both runtimes should open a memory-card file");
            t.Equals(
                secondMcFd,
                firstMcFd,
                "each runtime should independently allocate its first memory-card descriptor");

            clearContext(test.ctx);
            ps2_stubs::sceMcEnd(
                test.rdram.data(),
                &test.ctx,
                &first);

            clearContext(secondCtx);
            setRegU32(
                secondCtx,
                4,
                static_cast<uint32_t>(secondMcFd));
            setRegU32(secondCtx, 5, 0u);
            setRegU32(secondCtx, 6, PS2_FIO_SEEK_SET);
            ps2_stubs::sceMcSeek(
                secondRdram.data(),
                &secondCtx,
                &second);
            t.Equals(
                syncMc(secondRdram, &second),
                0,
                "ending one runtime's memory-card session must not close another runtime's files");

            ps2_syscalls::notifyRuntimeStop(&first);

            clearContext(test.ctx);
            setRegU32(test.ctx, 4, kLibcReadAddr);
            setRegU32(test.ctx, 5, 1u);
            setRegU32(test.ctx, 6, 1u);
            setRegU32(test.ctx, 7, firstLibcHandle);
            ps2_stubs::fread(
                test.rdram.data(),
                &test.ctx,
                &first);
            const uint32_t firstItemsRead =
                ::getRegU32(&test.ctx, 2);
            t.Equals(
                firstItemsRead,
                0u,
                "stopping a runtime should close its libc streams");

            clearContext(secondCtx);
            setRegU32(secondCtx, 4, kLibcReadAddr);
            setRegU32(secondCtx, 5, 1u);
            setRegU32(secondCtx, 6, 1u);
            setRegU32(secondCtx, 7, secondLibcHandle);
            ps2_stubs::fread(
                secondRdram.data(),
                &secondCtx,
                &second);
            t.Equals(
                ::getRegU32(&secondCtx, 2),
                1u,
                "stopping one runtime must not close another runtime's libc streams");

            clearContext(test.ctx);
            ps2_stubs::rand(
                test.rdram.data(),
                &test.ctx,
                &first);
            t.Equals(
                ::getRegU32(&test.ctx, 2),
                firstRandom,
                "stopping a runtime should reset its libc random sequence");
            clearContext(secondCtx);
            ps2_stubs::rand(
                secondRdram.data(),
                &secondCtx,
                &second);
            t.Equals(
                ::getRegU32(&secondCtx, 2),
                firstNextRandom,
                "stopping one runtime must not reset another runtime's libc random sequence");

            clearContext(test.ctx);
            ps2_stubs::mcGetFileSelectWindowCursol(
                test.rdram.data(),
                &test.ctx,
                &first);
            t.Equals(
                getRegS32(&test.ctx, 2),
                0,
                "stopping a runtime should reset its memory-card UI cursor");
            clearContext(secondCtx);
            ps2_stubs::mcGetFileSelectWindowCursol(
                secondRdram.data(),
                &secondCtx,
                &second);
            t.Equals(
                getRegS32(&secondCtx, 2),
                4,
                "stopping one runtime must not reset another runtime's memory-card UI cursor");

            if (firstItemsRead != 0u)
            {
                clearContext(test.ctx);
                setRegU32(
                    test.ctx,
                    4,
                    firstLibcHandle);
                ps2_stubs::fclose(
                    test.rdram.data(),
                    &test.ctx,
                    &first);
            }

            clearContext(secondCtx);
            setRegU32(
                secondCtx,
                4,
                secondLibcHandle);
            ps2_stubs::fclose(
                secondRdram.data(),
                &secondCtx,
                &second);
            clearContext(secondCtx);
            ps2_stubs::sceMcEnd(
                secondRdram.data(),
                &secondCtx,
                &second);
        });

        tc.Run("mc0 directory creation", [](TestCase &t)
        {
            TestContext test;

            const std::string dirPath = "mc0:/SAVEDATA";
            const uint32_t dirAddr = GUEST_STRING_AREA_START;
            writeGuestString(test.rdram.data(), dirAddr, dirPath);

            setRegU32(test.ctx, 4, dirAddr);
            fioMkdir(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            
            const int32_t result = getRegS32(&test.ctx, 2);
            t.IsTrue(result >= 0, "fioMkdir should succeed for mc0: directory");

            const std::filesystem::path expected = test.paths.mcRoot / "SAVEDATA";
            t.IsTrue(std::filesystem::exists(expected), 
                "Directory should exist under mcRoot");
            t.IsTrue(std::filesystem::is_directory(expected), 
                "Created path should be a directory");
        });

        tc.Run("mc0 file write operations", [](TestCase &t)
        {
            TestContext test;

            // Setup: create directory first
            const std::string dirPath = "mc0:/SAVEDATA";
            const uint32_t dirAddr = GUEST_STRING_AREA_START;
            writeGuestString(test.rdram.data(), dirAddr, dirPath);
            setRegU32(test.ctx, 4, dirAddr);
            fioMkdir(
                test.rdram.data(), &test.ctx,
                &test.runtime);

            // Test: open file for writing
            const std::string filePath = "mc0:/SAVEDATA/test.txt";
            const uint32_t fileAddr = GUEST_STRING_AREA_START + 0x100;
            writeGuestString(test.rdram.data(), fileAddr, filePath);

            setRegU32(test.ctx, 4, fileAddr);
            setRegU32(test.ctx, 5, PS2_FIO_WRITE_CREATE_TRUNC);
            fioOpen(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            
            const int32_t fd = getRegS32(&test.ctx, 2);
            t.IsTrue(fd >= 0, "fioOpen should return valid file descriptor");

            // Write payload
            const std::string payload = "hello mc0";
            const uint32_t bufAddr = GUEST_BUFFER_AREA_START;
            std::memcpy(test.rdram.data() + bufAddr, payload.data(), payload.size());

            setRegU32(test.ctx, 4, static_cast<uint32_t>(fd));
            setRegU32(test.ctx, 5, bufAddr);
            setRegU32(test.ctx, 6, static_cast<uint32_t>(payload.size()));
            fioWrite(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            
            const int32_t bytesWritten = getRegS32(&test.ctx, 2);
            t.Equals(bytesWritten, static_cast<int32_t>(payload.size()), 
                "fioWrite should write all bytes");

            // Close file
            setRegU32(test.ctx, 4, static_cast<uint32_t>(fd));
            fioClose(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            
            const int32_t closeResult = getRegS32(&test.ctx, 2);
            t.IsTrue(closeResult >= 0, "fioClose should succeed");

            // Verify on host filesystem
            const std::filesystem::path expectedPath = 
                test.paths.mcRoot / "SAVEDATA" / "test.txt";
            t.IsTrue(std::filesystem::exists(expectedPath), 
                "File should exist under mcRoot");

            std::ifstream in(expectedPath, std::ios::binary);
            std::string readback(
                (std::istreambuf_iterator<char>(in)), 
                std::istreambuf_iterator<char>());
            t.Equals(readback, payload, "File content should match written payload");
        });

        tc.Run("mc0 file read operations", [](TestCase &t)
        {
            TestContext test;

            // Setup: create directory and write file
            const std::string dirPath = "mc0:/SAVEDATA";
            const uint32_t dirAddr = GUEST_STRING_AREA_START;
            writeGuestString(test.rdram.data(), dirAddr, dirPath);
            setRegU32(test.ctx, 4, dirAddr);
            fioMkdir(
                test.rdram.data(), &test.ctx,
                &test.runtime);

            const std::string filePath = "mc0:/SAVEDATA/test.txt";
            const uint32_t fileAddr = GUEST_STRING_AREA_START + 0x100;
            writeGuestString(test.rdram.data(), fileAddr, filePath);

            // Write data
            const std::string payload = "hello mc0 read test";
            const uint32_t writeBufAddr = GUEST_BUFFER_AREA_START;
            std::memcpy(test.rdram.data() + writeBufAddr, payload.data(), payload.size());

            setRegU32(test.ctx, 4, fileAddr);
            setRegU32(test.ctx, 5, PS2_FIO_WRITE_CREATE_TRUNC);
            fioOpen(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            int32_t fd = getRegS32(&test.ctx, 2);

            setRegU32(test.ctx, 4, static_cast<uint32_t>(fd));
            setRegU32(test.ctx, 5, writeBufAddr);
            setRegU32(test.ctx, 6, static_cast<uint32_t>(payload.size()));
            fioWrite(
                test.rdram.data(), &test.ctx,
                &test.runtime);

            setRegU32(test.ctx, 4, static_cast<uint32_t>(fd));
            fioClose(
                test.rdram.data(), &test.ctx,
                &test.runtime);

            // Test: read back via fioRead
            setRegU32(test.ctx, 4, fileAddr);
            setRegU32(test.ctx, 5, PS2_FIO_O_RDONLY);
            fioOpen(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            fd = getRegS32(&test.ctx, 2);
            t.IsTrue(fd >= 0, "fioOpen for reading should succeed");

            // Read into different buffer area
            const uint32_t readBufAddr = GUEST_BUFFER_AREA_START + 0x1000;
            std::memset(test.rdram.data() + readBufAddr, 0, payload.size());

            setRegU32(test.ctx, 4, static_cast<uint32_t>(fd));
            setRegU32(test.ctx, 5, readBufAddr);
            setRegU32(test.ctx, 6, static_cast<uint32_t>(payload.size()));
            fioRead(
                test.rdram.data(), &test.ctx,
                &test.runtime);

            const int32_t bytesRead = getRegS32(&test.ctx, 2);
            t.Equals(bytesRead, static_cast<int32_t>(payload.size()), 
                "fioRead should read all bytes");

            std::string readback(
                reinterpret_cast<const char*>(test.rdram.data() + readBufAddr),
                payload.size()
            );
            t.Equals(readback, payload, "fioRead content should match original");

            setRegU32(test.ctx, 4, static_cast<uint32_t>(fd));
            fioClose(
                test.rdram.data(), &test.ctx,
                &test.runtime);
        });

        tc.Run("mc0 paths isolated from cdRoot", [](TestCase &t)
        {
            TestContext test;

            const std::string dirPath = "mc0:/ISOLATED";
            const std::string filePath = "mc0:/ISOLATED/test.txt";
            const uint32_t dirAddr = GUEST_STRING_AREA_START;
            const uint32_t fileAddr = GUEST_STRING_AREA_START + 0x100;
            
            writeGuestString(test.rdram.data(), dirAddr, dirPath);
            writeGuestString(test.rdram.data(), fileAddr, filePath);

            // Create directory and file on mc0:
            setRegU32(test.ctx, 4, dirAddr);
            fioMkdir(
                test.rdram.data(), &test.ctx,
                &test.runtime);

            setRegU32(test.ctx, 4, fileAddr);
            setRegU32(test.ctx, 5, PS2_FIO_WRITE_CREATE_TRUNC);
            fioOpen(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            const int32_t fd = getRegS32(&test.ctx, 2);

            const std::string payload = "isolation test";
            const uint32_t bufAddr = GUEST_BUFFER_AREA_START;
            std::memcpy(test.rdram.data() + bufAddr, payload.data(), payload.size());

            setRegU32(test.ctx, 4, static_cast<uint32_t>(fd));
            setRegU32(test.ctx, 5, bufAddr);
            setRegU32(test.ctx, 6, static_cast<uint32_t>(payload.size()));
            fioWrite(
                test.rdram.data(), &test.ctx,
                &test.runtime);

            setRegU32(test.ctx, 4, static_cast<uint32_t>(fd));
            fioClose(
                test.rdram.data(), &test.ctx,
                &test.runtime);

            // Verify isolation
            const std::filesystem::path expectedMc = 
                test.paths.mcRoot / "ISOLATED" / "test.txt";
            const std::filesystem::path unexpectedCd = 
                test.paths.cdRoot / "ISOLATED" / "test.txt";

            t.IsTrue(std::filesystem::exists(expectedMc), 
                "mc0: file should exist under mcRoot");
            t.IsFalse(std::filesystem::exists(unexpectedCd), 
                "mc0: file should NOT exist under cdRoot");

            // Verify mcRoot directory structure
            t.IsTrue(std::filesystem::exists(test.paths.mcRoot / "ISOLATED"), 
                "mc0: directory should exist under mcRoot");
            t.IsFalse(std::filesystem::exists(test.paths.cdRoot / "ISOLATED"), 
                "mc0: directory should NOT exist under cdRoot");
        });

        tc.Run("sceMc open write read and close roundtrip through sync", [](TestCase &t)
        {
            TestContext test;

            const uint32_t dirAddr = GUEST_STRING_AREA_START + 0x400;
            const uint32_t fileAddr = GUEST_STRING_AREA_START + 0x500;
            const uint32_t writeBufAddr = GUEST_BUFFER_AREA_START + 0x300;
            const uint32_t readBufAddr = GUEST_BUFFER_AREA_START + 0x500;
            const std::string payload = "libmc roundtrip";

            writeGuestString(test.rdram.data(), dirAddr, "/SAVEDATA");
            writeGuestString(test.rdram.data(), fileAddr, "/SAVEDATA/test.bin");
            std::memcpy(test.rdram.data() + writeBufAddr, payload.data(), payload.size());

            clearContext(test.ctx);
            setRegU32(test.ctx, 4, 0u);
            setRegU32(test.ctx, 5, 0u);
            setRegU32(test.ctx, 6, dirAddr);
            ps2_stubs::sceMcMkdir(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            t.Equals(getRegS32(&test.ctx, 2), 0, "sceMcMkdir should dispatch successfully");

            int32_t cmd = 0;
            t.Equals(syncMc(test.rdram, &test.runtime, &cmd), 0, "sceMcMkdir should finish successfully");
            t.Equals(cmd, 0x0B, "sceMcSync should report MKDIR as the last command");

            clearContext(test.ctx);
            setRegU32(test.ctx, 4, 0u);
            setRegU32(test.ctx, 5, 0u);
            setRegU32(test.ctx, 6, fileAddr);
            setRegU32(test.ctx, 7, PS2_FIO_O_RDWR | PS2_FIO_O_CREAT | PS2_FIO_O_TRUNC);
            ps2_stubs::sceMcOpen(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            t.Equals(getRegS32(&test.ctx, 2), 0, "sceMcOpen should dispatch successfully");

            const int32_t fd =
                syncMc(test.rdram, &test.runtime, &cmd);
            t.IsTrue(fd > 0, "sceMcOpen should produce a positive descriptor in sceMcSync");
            t.Equals(cmd, 0x02, "sceMcSync should report OPEN as the last command");

            clearContext(test.ctx);
            setRegU32(test.ctx, 4, static_cast<uint32_t>(fd));
            setRegU32(test.ctx, 5, writeBufAddr);
            setRegU32(test.ctx, 6, static_cast<uint32_t>(payload.size()));
            ps2_stubs::sceMcWrite(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            t.Equals(syncMc(test.rdram, &test.runtime, &cmd), static_cast<int32_t>(payload.size()),
                     "sceMcWrite should report the full byte count via sceMcSync");
            t.Equals(cmd, 0x06, "sceMcSync should report WRITE as the last command");

            clearContext(test.ctx);
            setRegU32(test.ctx, 4, static_cast<uint32_t>(fd));
            setRegU32(test.ctx, 5, 0u);
            setRegU32(test.ctx, 6, PS2_FIO_SEEK_SET);
            ps2_stubs::sceMcSeek(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            t.Equals(syncMc(test.rdram, &test.runtime, &cmd), 0, "sceMcSeek should rewind to offset zero");
            t.Equals(cmd, 0x04, "sceMcSync should report SEEK as the last command");

            std::memset(test.rdram.data() + readBufAddr, 0, payload.size());
            clearContext(test.ctx);
            setRegU32(test.ctx, 4, static_cast<uint32_t>(fd));
            setRegU32(test.ctx, 5, readBufAddr);
            setRegU32(test.ctx, 6, static_cast<uint32_t>(payload.size()));
            ps2_stubs::sceMcRead(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            t.Equals(syncMc(test.rdram, &test.runtime, &cmd), static_cast<int32_t>(payload.size()),
                     "sceMcRead should report the full byte count via sceMcSync");
            t.Equals(cmd, 0x05, "sceMcSync should report READ as the last command");

            std::string readback(reinterpret_cast<const char *>(test.rdram.data() + readBufAddr), payload.size());
            t.Equals(readback, payload, "sceMcRead should fill the guest buffer with the written payload");

            clearContext(test.ctx);
            setRegU32(test.ctx, 4, static_cast<uint32_t>(fd));
            ps2_stubs::sceMcClose(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            t.Equals(syncMc(test.rdram, &test.runtime, &cmd), 0, "sceMcClose should finish successfully");
            t.Equals(cmd, 0x03, "sceMcSync should report CLOSE as the last command");

            const std::filesystem::path hostPath = test.paths.mcRoot / "SAVEDATA" / "test.bin";
            t.IsTrue(std::filesystem::exists(hostPath), "sceMcOpen/sceMcWrite should create the host file under mcRoot");
        });

        tc.Run("sceMcGetDir includes dot entries and file metadata", [](TestCase &t)
        {
            TestContext test;

            std::filesystem::create_directories(test.paths.mcRoot / "SAVEDATA");
            const std::string hostPayload = "abc123";
            {
                std::ofstream out(test.paths.mcRoot / "SAVEDATA" / "game.dat", std::ios::binary);
                out.write(hostPayload.data(), static_cast<std::streamsize>(hostPayload.size()));
            }

            const uint32_t patternAddr = GUEST_STRING_AREA_START + 0x700;
            writeGuestString(test.rdram.data(), patternAddr, "/SAVEDATA/*");

            clearContext(test.ctx);
            setRegU32(test.ctx, 4, 0u);
            setRegU32(test.ctx, 5, 0u);
            setRegU32(test.ctx, 6, patternAddr);
            setRegU32(test.ctx, 7, 0u);
            // EE n32 ABI: arguments 5 and 6 travel in $t0/$t1
            setRegU32(test.ctx, 8, 8u);
            setRegU32(test.ctx, 9, GUEST_MC_TABLE_ADDR);

            ps2_stubs::sceMcGetDir(
                test.rdram.data(), &test.ctx,
                &test.runtime);

            int32_t cmd = 0;
            const int32_t entryCount =
                syncMc(test.rdram, &test.runtime, &cmd);
            t.Equals(cmd, 0x0D, "sceMcSync should report GETDIR as the last command");
            t.Equals(entryCount, 3, "sceMcGetDir should return '.', '..', and the matching file");

            const auto *entries = reinterpret_cast<const SceMcTblGetDir *>(test.rdram.data() + GUEST_MC_TABLE_ADDR);
            t.Equals(std::string(entries[0].entryName), std::string("."), "sceMcGetDir should return '.' first");
            t.Equals(std::string(entries[1].entryName), std::string(".."), "sceMcGetDir should return '..' second");
            t.Equals(std::string(entries[2].entryName), std::string("game.dat"), "sceMcGetDir should include the matching file entry");
            t.Equals(entries[2].fileSizeByte, static_cast<uint32_t>(hostPayload.size()),
                     "sceMcGetDir should report the host file size");
            t.IsTrue((entries[2].attrFile & 0x0080u) != 0u,
                     "sceMcGetDir file entries should carry the closed-file attribute");
        });

        tc.Run("sceMcGetInfo reports formatted and unformatted states", [](TestCase &t)
        {
            TestContext test;

            constexpr uint32_t typeAddr = GUEST_BUFFER_AREA_START + 0x900;
            constexpr uint32_t freeAddr = GUEST_BUFFER_AREA_START + 0x904;
            constexpr uint32_t formatAddr = GUEST_BUFFER_AREA_START + 0x908;

            clearContext(test.ctx);
            setRegU32(test.ctx, 4, 0u);
            setRegU32(test.ctx, 5, 0u);
            setRegU32(test.ctx, 6, typeAddr);
            setRegU32(test.ctx, 7, freeAddr);
            // EE n32 ABI: the fifth argument travels in $t0
            setRegU32(test.ctx, 8, formatAddr);
            ps2_stubs::sceMcGetInfo(
                test.rdram.data(), &test.ctx,
                &test.runtime);

            int32_t cmd = 0;
            t.Equals(syncMc(test.rdram, &test.runtime, &cmd), 0, "formatted cards should report success through sceMcSync");
            t.Equals(cmd, 0x01, "sceMcSync should report GETINFO as the last command");
            t.Equals(readGuestS32(test.rdram.data(), typeAddr), 2, "sceMcGetInfo should report a PS2 memory card");
            t.Equals(readGuestS32(test.rdram.data(), formatAddr), 1, "sceMcGetInfo should report a formatted card");

            clearContext(test.ctx);
            setRegU32(test.ctx, 4, 0u);
            setRegU32(test.ctx, 5, 0u);
            ps2_stubs::sceMcUnformat(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            t.Equals(syncMc(test.rdram, &test.runtime, &cmd), 0, "sceMcUnformat should complete successfully");

            clearContext(test.ctx);
            setRegU32(test.ctx, 4, 0u);
            setRegU32(test.ctx, 5, 0u);
            setRegU32(test.ctx, 6, typeAddr);
            setRegU32(test.ctx, 7, freeAddr);
            setRegU32(test.ctx, 8, formatAddr);
            ps2_stubs::sceMcGetInfo(
                test.rdram.data(), &test.ctx,
                &test.runtime);

            t.Equals(syncMc(test.rdram, &test.runtime, &cmd), -2, "unformatted cards should report sceMcResNoFormat through sceMcSync");
            t.Equals(readGuestS32(test.rdram.data(), formatAddr), 0, "sceMcGetInfo should report an unformatted card after sceMcUnformat");
        });

        tc.Run("sceMcEnd resets libmc state so sync reports no active command", [](TestCase &t)
        {
            TestContext test;

            constexpr uint32_t dirAddr = GUEST_STRING_AREA_START + 0xB00;

            clearContext(test.ctx);
            ps2_stubs::sceMcInit(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            t.Equals(getRegS32(&test.ctx, 2), 0, "sceMcInit should succeed");

            writeGuestString(test.rdram.data(), dirAddr, "/SAVEDATA");
            clearContext(test.ctx);
            setRegU32(test.ctx, 4, 0u);
            setRegU32(test.ctx, 5, 0u);
            setRegU32(test.ctx, 6, dirAddr);
            ps2_stubs::sceMcMkdir(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            int32_t cmd = 0;
            t.Equals(syncMc(test.rdram, &test.runtime, &cmd), 0, "sceMcMkdir should complete before teardown");
            t.Equals(cmd, 0x0B, "sceMcSync should report MKDIR before teardown");

            clearContext(test.ctx);
            ps2_stubs::sceMcEnd(
                test.rdram.data(), &test.ctx,
                &test.runtime);
            t.Equals(getRegS32(&test.ctx, 2), 0, "sceMcEnd should succeed");

            // libmc semantics: with no async command pending, sceMcSync returns -1
            // and leaves the cmd/result out-parameters untouched.
            R5900Context syncCtx{};
            setRegU32(syncCtx, 4, 0u);
            setRegU32(syncCtx, 5, GUEST_MC_SYNC_CMD_ADDR);
            setRegU32(syncCtx, 6, GUEST_MC_SYNC_RESULT_ADDR);
            ps2_stubs::sceMcSync(
                test.rdram.data(), &syncCtx,
                &test.runtime);
            t.Equals(getRegS32(&syncCtx, 2), -1,
                     "sceMcSync after sceMcEnd should report that no command is active");
        });

        tc.Run("sceIoctl cmd1 updates wait flag state", [](TestCase &t)
        {
            TestContext test;

            constexpr uint32_t statusAddr = GUEST_BUFFER_AREA_START + 0x1800;
            const uint32_t busy = 1u;
            std::memcpy(test.rdram.data() + statusAddr, &busy, sizeof(busy));

            setRegU32(test.ctx, 4, 3u);          // fd
            setRegU32(test.ctx, 5, 1u);          // cmd
            setRegU32(test.ctx, 6, statusAddr);  // arg

            ps2_stubs::sceIoctl(
                test.rdram.data(), &test.ctx,
                &test.runtime);

            t.Equals(getRegS32(&test.ctx, 2), 0, "sceIoctl cmd1 should return success");

            uint32_t state = 0xFFFFFFFFu;
            std::memcpy(&state, test.rdram.data() + statusAddr, sizeof(state));
            t.Equals(state, 0u, "sceIoctl cmd1 should clear wait state from busy to ready");
        });

        tc.Run("sceCdSearchFile resolves movie filenames with zero-padded host leaf", [](TestCase &t)
        {
            TestContext test;

            std::filesystem::create_directories(test.paths.cdRoot / "movie");
            {
                std::ofstream out(test.paths.cdRoot / "movie" / "mv_016.pss", std::ios::binary);
                const std::string payload = "pss";
                out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
            }

            constexpr uint32_t fileAddr = GUEST_BUFFER_AREA_START + 0x1A00;
            constexpr uint32_t pathAddr = GUEST_STRING_AREA_START + 0xA00;
            writeGuestString(test.rdram.data(), pathAddr, "\\MOVIE\\MV_16.PSS;1");

            clearContext(test.ctx);
            setRegU32(test.ctx, 4, fileAddr);
            setRegU32(test.ctx, 5, pathAddr);
            ps2_stubs::sceCdSearchFile(
                test.rdram.data(), &test.ctx,
                &test.runtime);

            t.Equals(getRegS32(&test.ctx, 2), 1, "sceCdSearchFile should resolve the extracted movie file");
            t.Equals(readGuestU32(test.rdram.data(), fileAddr + 4), 3u,
                     "sceCdSearchFile should report the host file size");
            t.IsTrue(readGuestU32(test.rdram.data(), fileAddr + 0) >= 0x00100000u,
                     "sceCdSearchFile should assign a pseudo LSN for the resolved host file");
        });

        tc.Run("sceCdRead reads from explicit cdImage path", [](TestCase &t)
        {
            TestContext test;

            constexpr uint32_t kSectorSize = 2048u;
            constexpr uint32_t bufAddr = GUEST_BUFFER_AREA_START + 0x1C80;
            const std::filesystem::path imagePath = test.paths.base / "disc.iso";
            {
                std::vector<uint8_t> sector(kSectorSize, 0);
                const char payload[] = "cd-image";
                std::memcpy(sector.data(), payload, sizeof(payload) - 1);

                std::ofstream out(imagePath, std::ios::binary);
                out.write(reinterpret_cast<const char *>(sector.data()),
                          static_cast<std::streamsize>(sector.size()));
            }

            PS2Runtime::IoPaths ioPaths;
            ioPaths.elfDirectory = test.paths.cdRoot;
            ioPaths.hostRoot = test.paths.cdRoot;
            ioPaths.cdRoot = test.paths.cdRoot;
            ioPaths.mcRoot = test.paths.mcRoot;
            ioPaths.cdImage = imagePath;
            test.runtime.configureIoPaths(ioPaths);

            clearContext(test.ctx);
            setRegU32(test.ctx, 4, 0u);
            setRegU32(test.ctx, 5, 1u);
            setRegU32(test.ctx, 6, bufAddr);
            ps2_stubs::sceCdRead(
                test.rdram.data(), &test.ctx,
                &test.runtime);

            t.Equals(getRegS32(&test.ctx, 2), 1, "sceCdRead should succeed when cdImage is configured");
            t.Equals(std::memcmp(test.rdram.data() + bufAddr, "cd-image", 8), 0,
                     "sceCdRead should copy sector data from the configured image");
        });
    });
}
