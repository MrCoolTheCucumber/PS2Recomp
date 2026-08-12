#include "MiniTest.h"
#include "ps2x/iop/iop_subsystem.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#endif

namespace
{
    using namespace ps2x::iop;

    constexpr uint32_t kSyntheticSid = 0xF00DCAFEu;
    constexpr uint32_t kCoreCollisionSid = 0x80001300u;
    constexpr uint32_t kSyntheticFunction = 0x42u;
    constexpr uint32_t kCoreCollisionFunction = 0x99u;
    constexpr uint32_t kSyntheticEntryPoint = 0x00123456u;
    constexpr uint32_t kSpecificRecvXEntryPoint = kSyntheticEntryPoint + 0x100u;
    constexpr uint32_t kSyntheticCrc32 = 0xA1B2C3D4u;
    constexpr uint32_t kResponseXor = 0xA5A55A5Au;
    constexpr uint32_t kCoreCollisionResponse = 0xC0DEF00Du;

    class FakeIopHost final : public IopHost
    {
    public:
        explicit FakeIopHost(size_t memorySize = 0x10000u)
            : memory(memorySize, 0u),
              iopMemory(2u * 1024u * 1024u, 0u)
        {
        }

        bool readGuest(uint32_t address, void *destination, size_t size) const override
        {
            if ((!destination && size != 0u) || !contains(address, size))
            {
                return false;
            }
            if (size != 0u)
            {
                std::memcpy(destination, memory.data() + address, size);
            }
            return true;
        }

        bool writeGuest(uint32_t address, const void *source, size_t size) override
        {
            if ((!source && size != 0u) || !contains(address, size))
            {
                return false;
            }
            if (size != 0u)
            {
                std::memcpy(memory.data() + address, source, size);
            }
            return true;
        }

        bool zeroGuest(uint32_t address, size_t size) override
        {
            if (!contains(address, size))
            {
                return false;
            }
            std::fill(memory.begin() + address, memory.begin() + address + size, 0u);
            return true;
        }

        bool normalizeGuestAddress(uint32_t address, uint32_t &normalized) const override
        {
            normalized = address & 0x1FFFFFFFu;
            return normalized < memory.size();
        }

        uint32_t allocateIopHandle(IopHandleKind kind) override
        {
            const uint32_t value = nextHandle;
            switch (kind)
            {
            case IopHandleKind::RpcPacket:
                nextHandle += 0x40u;
                break;
            case IopHandleKind::RpcServer:
                nextHandle += 0x80u;
                break;
            case IopHandleKind::ServiceResource:
                nextHandle += 0x10u;
                break;
            }
            return value;
        }

        uint32_t allocateGuest(uint32_t size, uint32_t alignment) override
        {
            if (size == 0u)
            {
                return 0u;
            }
            const uint64_t effectiveAlignment = alignment == 0u ? 1u : alignment;
            const uint64_t aligned = ((static_cast<uint64_t>(nextGuestAddress) + effectiveAlignment - 1u) /
                                      effectiveAlignment) *
                                     effectiveAlignment;
            if (aligned + size > memory.size())
            {
                return 0u;
            }
            nextGuestAddress = static_cast<uint32_t>(aligned + size);
            guestAllocations.push_back(static_cast<uint32_t>(aligned));
            return static_cast<uint32_t>(aligned);
        }

        void freeGuest(uint32_t address) override
        {
            freedGuestAddresses.push_back(address);
        }

        void audioCommand(uint32_t sid,
                          uint32_t function,
                          GuestBuffer send,
                          GuestBuffer receive) override
        {
            lastAudioSid = sid;
            lastAudioFunction = function;
            lastAudioSend = send;
            lastAudioReceive = receive;
            lastAudioPayload.clear();
            const std::vector<uint8_t> &source =
                receive.domain == GuestMemoryDomain::IopRam ? iopMemory : memory;
            if (receive.address != 0u &&
                contains(source, receive.address, receive.size))
            {
                lastAudioPayload.assign(source.begin() + receive.address,
                                        source.begin() + receive.address + receive.size);
            }
            ++audioCalls;
        }

        void audioBankLoaded(uint32_t sid,
                             uint32_t bankHandle,
                             const void *container,
                             size_t containerSize) override
        {
            lastAudioBankSid = sid;
            lastLoadedAudioBankHandle = bankHandle;
            lastAudioBankContainer.clear();
            if (container && containerSize != 0u)
            {
                const auto *bytes = static_cast<const uint8_t *>(container);
                lastAudioBankContainer.assign(bytes, bytes + containerSize);
            }
            ++audioBankLoads;
        }

        void audioBankUnloaded(uint32_t sid, uint32_t bankHandle) override
        {
            lastAudioBankSid = sid;
            lastUnloadedAudioBankHandle = bankHandle;
            ++audioBankUnloads;
        }

        uint64_t virtualTimeNanoseconds() const override
        {
            return virtualTime;
        }

        std::string hostPath(HostPathKind kind) const override
        {
            switch (kind)
            {
            case HostPathKind::CdRoot:
                return "fake/cd";
            case HostPathKind::CdImage:
                return "fake/disc.iso";
            case HostPathKind::HostRoot:
                return "fake/host";
            case HostPathKind::MemoryCardRoot:
                return "fake/mc0";
            default:
                return "fake/elf";
            }
        }

        std::string translateGuestPath(std::string_view path) const override
        {
            return "translated/" + std::string(path);
        }

        uint64_t openHostFile(std::string_view path) override
        {
            const auto file = hostFileContents.find(std::string(path));
            if (file == hostFileContents.end())
            {
                return 0u;
            }
            const uint64_t handle = nextHostFileHandle++;
            openHostFiles.emplace(handle, file->first);
            return handle;
        }

        bool hostFileSize(uint64_t handle, uint64_t &size) const override
        {
            size = 0u;
            const auto open = openHostFiles.find(handle);
            if (open == openHostFiles.end())
            {
                return false;
            }
            const auto file = hostFileContents.find(open->second);
            if (file == hostFileContents.end())
            {
                return false;
            }
            size = file->second.size();
            return true;
        }

        bool readHostFile(uint64_t handle,
                          uint64_t offset,
                          void *destination,
                          size_t size,
                          size_t &bytesRead) override
        {
            bytesRead = 0u;
            if (!destination && size != 0u)
            {
                return false;
            }
            const auto open = openHostFiles.find(handle);
            if (open == openHostFiles.end())
            {
                return false;
            }
            const auto file = hostFileContents.find(open->second);
            if (file == hostFileContents.end() || offset > file->second.size())
            {
                return false;
            }
            bytesRead = std::min<size_t>(size, file->second.size() - static_cast<size_t>(offset));
            if (bytesRead != 0u)
            {
                std::memcpy(destination,
                            file->second.data() + static_cast<size_t>(offset),
                            bytesRead);
            }
            return true;
        }

        void closeHostFile(uint64_t handle) override
        {
            if (openHostFiles.erase(handle) != 0u)
            {
                closedHostFileHandles.push_back(handle);
            }
        }

        int32_t memoryCard(const MemoryCardRequest &request) override
        {
            lastMemoryCardRequest = request;
            ++memoryCardCalls;
            return 0;
        }

        bool hasGuestFunction(uint32_t address) const override
        {
            return address == guestFunctionAddress;
        }

        bool invokeGuestFunction(uint64_t callToken,
                                 uint32_t address,
                                 uint32_t a0,
                                 uint32_t a1,
                                 uint32_t a2,
                                 uint32_t a3,
                                 uint32_t *resultAddress) override
        {
            if (!hasGuestFunction(address))
            {
                return false;
            }
            lastCallToken = callToken;
            lastGuestArguments = {a0, a1, a2, a3};
            if (resultAddress)
            {
                *resultAddress = guestFunctionResult;
            }
            return true;
        }

        void log(LogLevel level, std::string_view message) override
        {
            logs.emplace_back(level, std::string(message));
        }

        bool writeWord(uint32_t address, uint32_t value)
        {
            return writeGuest(address, &value, sizeof(value));
        }

        uint32_t readWord(uint32_t address) const
        {
            uint32_t value = 0u;
            (void)readGuest(address, &value, sizeof(value));
            return value;
        }

        bool hasLog(std::string_view expected) const
        {
            return std::any_of(logs.begin(), logs.end(), [&](const auto &entry)
                               { return entry.second == expected; });
        }

        std::vector<uint8_t> memory;
        std::vector<uint8_t> iopMemory;
        uint32_t nextHandle = 0x8000u;
        uint32_t nextGuestAddress = 0x4000u;
        std::vector<uint32_t> guestAllocations;
        std::vector<uint32_t> freedGuestAddresses;
        uint32_t audioCalls = 0u;
        uint32_t lastAudioSid = 0u;
        uint32_t lastAudioFunction = 0u;
        GuestBuffer lastAudioSend{};
        GuestBuffer lastAudioReceive{};
        std::vector<uint8_t> lastAudioPayload;
        uint32_t audioBankLoads = 0u;
        uint32_t audioBankUnloads = 0u;
        uint32_t lastAudioBankSid = 0u;
        uint32_t lastLoadedAudioBankHandle = 0u;
        uint32_t lastUnloadedAudioBankHandle = 0u;
        std::vector<uint8_t> lastAudioBankContainer;
        uint64_t virtualTime = 0u;
        uint32_t memoryCardCalls = 0u;
        MemoryCardRequest lastMemoryCardRequest{};
        uint32_t guestFunctionAddress = 0x2000u;
        uint32_t guestFunctionResult = 0x3000u;
        uint64_t lastCallToken = 0u;
        std::vector<uint32_t> lastGuestArguments;
        std::vector<std::pair<LogLevel, std::string>> logs;
        std::unordered_map<std::string, std::vector<uint8_t>> hostFileContents;
        std::unordered_map<uint64_t, std::string> openHostFiles;
        std::vector<uint64_t> closedHostFileHandles;
        uint64_t nextHostFileHandle = 1u;

    private:
        bool contains(uint32_t address, size_t size) const
        {
            return contains(memory, address, size);
        }

        static bool contains(const std::vector<uint8_t> &source,
                             uint32_t address,
                             size_t size)
        {
            const uint64_t end =
                static_cast<uint64_t>(address) + static_cast<uint64_t>(size);
            return end <= source.size();
        }
    };

    bool containsDiagnostic(const DebugSnapshot &snapshot, std::string_view text)
    {
        return std::any_of(snapshot.diagnostics.begin(), snapshot.diagnostics.end(), [&](const std::string &diagnostic)
                           { return diagnostic.find(text) != std::string::npos; });
    }

    const DebugService *findService(const DebugSnapshot &snapshot, std::string_view name)
    {
        const auto it = std::find_if(snapshot.services.begin(), snapshot.services.end(), [&](const DebugService &service)
                                     { return service.name == name; });
        return it == snapshot.services.end() ? nullptr : &*it;
    }

    uint64_t metricValue(const DebugService &service, std::string_view name)
    {
        const auto it = std::find_if(service.metrics.begin(), service.metrics.end(), [&](const DebugMetric &metric)
                                     { return metric.name == name; });
        return it == service.metrics.end() ? std::numeric_limits<uint64_t>::max() : it->value;
    }

    bool pluginModuleIsLoaded(const std::filesystem::path &path)
    {
#if defined(_WIN32)
        return GetModuleHandleW(path.c_str()) != nullptr;
#elif defined(__linux__)
        void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD);
        if (!handle)
        {
            return false;
        }
        dlclose(handle);
        return true;
#else
        (void)path;
        return false;
#endif
    }
}

void register_ps2_iop_tests()
{
    MiniTest::Case("PS2IopSubsystem", [](TestCase &tc)
    {
        tc.Run("unknown SID remains unhandled without a matching profile", [](TestCase &t)
        {
            FakeIopHost host;
            ps2x::iop::IopSubsystem subsystem(host);

            std::string error;
            const bool configured = subsystem.configure({"unmatched.elf", 0x100000u, 0x12345678u}, &error);
            t.IsTrue(configured, "configuring an unmatched game should keep core-only IOP services available");

            ps2x::iop::RpcRequest request{};
            request.sid = 0xDEADC0DEu;
            request.function = 0x99u;
            const ps2x::iop::RpcResult result = subsystem.handleRpc(request);
            t.IsFalse(result.handled, "an unknown SID should not be claimed by the IOP subsystem");
            t.Equals(result.resultAddress, 0u, "an unknown SID should not return a guest result address");
            t.IsFalse(result.signalNowaitCompletion, "an unknown SID should not signal nowait completion");
            t.Equals(result.callbackPolicy, ps2x::iop::CallbackPolicy::RuntimeDefault,
                     "an unknown SID should preserve runtime callback handling");

            const ps2x::iop::DebugSnapshot snapshot = subsystem.debugSnapshot();
            t.IsTrue(snapshot.activeProfile.empty(), "an unmatched game should not activate a profile");
            t.IsTrue(snapshot.activeProvider.empty(), "an unmatched game should not report a profile provider");
        });

        tc.Run("built-in profiles select by ELF basename and keep core services active", [](TestCase &t)
        {
            FakeIopHost host;
            ps2x::iop::IopSubsystem subsystem(host);
            std::string error;

            t.IsTrue(subsystem.configure({"SLUS_201.84", 0u, 0u}, &error),
                     "RECVX profile should match case-insensitively by basename");
            ps2x::iop::DebugSnapshot snapshot = subsystem.debugSnapshot();
            t.Equals(snapshot.activeProfile, std::string("recvx-us"),
                     "RECVX ELF should select its built-in profile");
            t.IsNotNull(findService(snapshot, "TSNDDRV"),
                        "RECVX profile should register TSNDDRV");
            t.IsNotNull(findService(snapshot, "CRI DTX"),
                        "RECVX profile should register CRI DTX");
            t.IsNotNull(findService(snapshot, "dbcman"),
                        "core DBCMAN should remain active with a game profile");
            t.IsNotNull(findService(snapshot, "libsd"),
                        "core LIBSD should remain active with a game profile");
            t.IsNotNull(findService(snapshot, "MCSERV"),
                        "core MCSERV should remain active with a game profile");

            error.clear();
            t.IsTrue(subsystem.configure({"slus_203.88", 0u, 0u}, &error),
                     "Fatal Frame profile should configure after a different game");
            snapshot = subsystem.debugSnapshot();
            t.Equals(snapshot.activeProfile, std::string("fatal-frame-us"),
                     "reload should replace the active profile");
            t.IsNull(findService(snapshot, "CRI DTX"),
                     "reload should destroy services from the previous profile");
            t.IsNotNull(findService(snapshot, "SDRDRV"),
                        "Fatal Frame profile should expose SDRDRV");
        });

        tc.Run("RAC1 stash manager round-trips EE DMA payloads through its IOP arena", [](TestCase &t)
        {
            FakeIopHost host;
            ps2x::iop::IopSubsystem subsystem(host);
            std::string error;

            constexpr uint32_t kSid = 0x11u;
            constexpr uint32_t kArenaBase = 0x000DE4D0u;
            constexpr uint32_t kArenaSize = 0x00094000u;
            constexpr uint32_t kInitializeReceive = 0x1000u;
            constexpr uint32_t kUploadSource = 0x2000u;
            constexpr uint32_t kReadRequest = 0x2100u;
            constexpr uint32_t kReadReceive = 0x2200u;

            t.IsTrue(subsystem.configure({"scus_971.99", 0u, 0u}, &error),
                     "RAC1 profile should match its ELF basename case-insensitively");
            ps2x::iop::DebugSnapshot snapshot = subsystem.debugSnapshot();
            t.Equals(snapshot.activeProfile, std::string("rac1-us"),
                     "RAC1 should select the stash-manager profile");
            const ps2x::iop::DebugService *service = findService(snapshot, "IOP_stashmgr");
            if (!service)
            {
                t.Fail("RAC1 profile should expose IOP_stashmgr");
                return;
            }
            t.Equals(metricValue(*service, "arena_base"), uint64_t{kArenaBase},
                     "stash-manager diagnostics should publish the reference IOP base");
            t.Equals(metricValue(*service, "arena_size"), uint64_t{kArenaSize},
                     "stash-manager diagnostics should publish the reference arena size");

            std::array<uint32_t, 4> initializeSentinel{
                0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
            (void)host.writeGuest(kInitializeReceive,
                                  initializeSentinel.data(),
                                  sizeof(initializeSentinel));
            ps2x::iop::RpcRequest initialize{};
            initialize.sid = kSid;
            initialize.function = 2u;
            initialize.receive = {kInitializeReceive, sizeof(initializeSentinel)};
            const ps2x::iop::RpcResult initializeResult = subsystem.handleRpc(initialize);
            t.IsTrue(initializeResult.handled,
                     "stash-manager initialization RPC should be handled");
            t.Equals(initializeResult.resultAddress, kInitializeReceive,
                     "initialization should return its receive buffer");
            t.Equals(host.readWord(kInitializeReceive), kArenaBase,
                     "initialization should return the IOP arena base");
            t.Equals(host.readWord(kInitializeReceive + 4u), kArenaSize,
                     "initialization should return the IOP arena size");
            t.Equals(host.readWord(kInitializeReceive + 8u), 0u,
                     "initialization should clear the third response word");
            t.Equals(host.readWord(kInitializeReceive + 12u), 0u,
                     "initialization should clear the fourth response word");

            std::array<uint8_t, 64> payload{};
            for (uint32_t index = 0u; index < payload.size(); ++index)
            {
                payload[index] = static_cast<uint8_t>(0x40u + index);
            }
            (void)host.writeGuest(kUploadSource, payload.data(), payload.size());
            subsystem.onSifTransfer({ps2x::iop::SifTransferKind::SetDma,
                                     ps2x::iop::SifTransferPhase::AfterCopy,
                                     kUploadSource,
                                     kArenaBase + 0x20u,
                                     static_cast<uint32_t>(payload.size())});

            constexpr uint32_t kPayloadReadOffset = 8u;
            constexpr uint32_t kPayloadReadBytes = 32u;
            const uint32_t readSource = kArenaBase + 0x20u + kPayloadReadOffset;
            (void)host.writeWord(kReadRequest, readSource);
            std::array<uint8_t, kPayloadReadBytes> receiveSentinel{};
            receiveSentinel.fill(0xCCu);
            (void)host.writeGuest(kReadReceive,
                                  receiveSentinel.data(),
                                  receiveSentinel.size());

            ps2x::iop::RpcRequest read{};
            read.sid = kSid;
            read.function = 1u;
            read.send = {kReadRequest, sizeof(readSource)};
            read.receive = {kReadReceive, kPayloadReadBytes};
            const ps2x::iop::RpcResult readResult = subsystem.handleRpc(read);
            t.IsTrue(readResult.handled, "stash-manager read RPC should be handled");
            t.Equals(readResult.resultAddress, kReadReceive,
                     "stash-manager read should return its receive buffer");
            t.IsTrue(std::equal(payload.begin() + kPayloadReadOffset,
                                payload.begin() + kPayloadReadOffset + kPayloadReadBytes,
                                host.memory.begin() + kReadReceive),
                     "read RPC should return bytes uploaded to the IOP arena");

            snapshot = subsystem.debugSnapshot();
            service = findService(snapshot, "IOP_stashmgr");
            if (!service)
            {
                t.Fail("stash-manager service should remain visible after transfers");
                return;
            }
            t.Equals(metricValue(*service, "uploads"), uint64_t{1},
                     "one matching SIF DMA should be retained");
            t.Equals(metricValue(*service, "upload_bytes"), uint64_t{payload.size()},
                     "upload metrics should count transferred bytes");
            t.Equals(metricValue(*service, "reads"), uint64_t{1},
                     "one valid read RPC should be counted");
            t.Equals(metricValue(*service, "read_bytes"), uint64_t{kPayloadReadBytes},
                     "read metrics should count returned bytes");

            subsystem.reset();
            receiveSentinel.fill(0xCCu);
            (void)host.writeGuest(kReadReceive,
                                  receiveSentinel.data(),
                                  receiveSentinel.size());
            (void)subsystem.handleRpc(read);
            t.IsTrue(std::all_of(host.memory.begin() + kReadReceive,
                                 host.memory.begin() + kReadReceive + kPayloadReadBytes,
                                 [](uint8_t value) { return value == 0u; }),
                     "reset should clear retained IOP stash bytes");

            const uint32_t outOfBoundsSource = kArenaBase + kArenaSize - 8u;
            (void)host.writeWord(kReadRequest, outOfBoundsSource);
            receiveSentinel.fill(0xCCu);
            (void)host.writeGuest(kReadReceive,
                                  receiveSentinel.data(),
                                  receiveSentinel.size());
            (void)subsystem.handleRpc(read);
            t.IsTrue(std::all_of(host.memory.begin() + kReadReceive,
                                 host.memory.begin() + kReadReceive + kPayloadReadBytes,
                                 [](uint8_t value) { return value == 0xCCu; }),
                     "an out-of-bounds read should leave the receive buffer untouched");
            snapshot = subsystem.debugSnapshot();
            service = findService(snapshot, "IOP_stashmgr");
            if (!service)
            {
                t.Fail("stash-manager service should remain visible after a rejected read");
                return;
            }
            t.Equals(metricValue(*service, "rejected_reads"), uint64_t{1},
                     "out-of-bounds reads should be visible in diagnostics");
        });

        tc.Run("Sony 989snd loads and unloads a validated CD sound bank", [](TestCase &t)
        {
            FakeIopHost host;
            ps2x::iop::IopSubsystem subsystem(host);

            constexpr uint32_t kMainSid = 0x00123456u;
            constexpr uint32_t kLoadingSid = 0x00123457u;
            constexpr uint32_t kSendAddress = 0x1000u;
            constexpr uint32_t kReceiveAddress = 0x1100u;
            constexpr uint32_t kSector = 1u;
            constexpr uint32_t kByteOffset = 0x20u;
            constexpr uint32_t kSectorSize = 2048u;
            constexpr uint32_t kBankOffset = kSector * kSectorSize + kByteOffset;

            std::vector<uint8_t> image(0x3000u, 0u);
            auto writeImageWord = [&](uint32_t offset, uint32_t value) {
                std::memcpy(image.data() + offset, &value, sizeof(value));
            };
            writeImageWord(kBankOffset + 0x00u, 3u);
            writeImageWord(kBankOffset + 0x04u, 2u);
            writeImageWord(kBankOffset + 0x08u, 0x18u);
            writeImageWord(kBankOffset + 0x0Cu, 0x40u);
            writeImageWord(kBankOffset + 0x10u, 0x58u);
            writeImageWord(kBankOffset + 0x14u, 0x80u);
            writeImageWord(kBankOffset + 0x18u, 0x6B6C4253u); // "SBlk"
            const std::vector<uint8_t> expectedContainer(
                image.begin() + kBankOffset,
                image.begin() + kBankOffset + 0xD8u);
            host.hostFileContents.emplace("fake/disc.iso", std::move(image));

            const std::array<uint32_t, 2> source{kSector, kByteOffset};
            (void)host.writeGuest(kSendAddress, source.data(), sizeof(source));
            (void)host.writeWord(kReceiveAddress, 0xFFFFFFFFu);
            (void)host.writeWord(kReceiveAddress + 4u, 0xA5A5A5A5u);

            ps2x::iop::RpcRequest load{};
            load.sid = kLoadingSid;
            load.function = 3u;
            load.send = {kSendAddress, sizeof(source)};
            load.receive = {kReceiveAddress, sizeof(uint32_t)};
            const ps2x::iop::RpcResult loadResult = subsystem.handleRpc(load);
            t.IsTrue(loadResult.handled,
                     "the dedicated 989snd loading server should handle bank load RPC 3");
            t.Equals(loadResult.resultAddress, kReceiveAddress,
                     "the loading server should return its four-byte response buffer");
            t.Equals(host.readWord(kReceiveAddress), 0x8000u,
                     "a structurally valid CD sound bank should receive a resource handle");
            t.Equals(host.readWord(kReceiveAddress + 4u), 0xA5A5A5A5u,
                     "the four-byte loading response must not overwrite adjacent memory");
            t.Equals(host.closedHostFileHandles.size(), size_t{1},
                     "the sound-bank loader should close the CD image after validation");
            t.Equals(host.audioBankLoads, 1u,
                     "a validated sound bank should be delivered to the audio backend once");
            t.Equals(host.lastAudioBankSid, kMainSid,
                     "the bank notification should identify the owning audio service");
            t.Equals(host.lastLoadedAudioBankHandle, 0x8000u,
                     "the audio backend should receive the published bank handle");
            t.Equals(host.lastAudioBankContainer.size(), expectedContainer.size(),
                     "the backend should receive the complete bounded bank container");
            t.Equals(std::memcmp(host.lastAudioBankContainer.data(),
                                 expectedContainer.data(),
                                 expectedContainer.size()),
                     0,
                     "the delivered bank container should match the validated CD bytes");

            constexpr uint32_t kWait = 1u;
            (void)host.writeGuest(kSendAddress, &kWait, sizeof(kWait));
            ps2x::iop::RpcRequest wait{};
            wait.sid = kMainSid;
            wait.function = 0x36u;
            wait.send = {kSendAddress, sizeof(kWait)};
            wait.receive = {kReceiveAddress, 12u};
            t.IsTrue(subsystem.handleRpc(wait).handled,
                     "the main 989snd server should handle wait-for-loads RPC 0x36");
            t.Equals(host.readWord(kReceiveAddress), 0xFFFFFFFFu,
                     "the main response should retain its leading sentinel");
            t.Equals(host.readWord(kReceiveAddress + 4u), 1u,
                     "a completed synchronous bank load should report idle");
            t.Equals(host.readWord(kReceiveAddress + 8u), 0xFFFFFFFFu,
                     "the main response should retain its trailing sentinel");

            constexpr std::array<uint32_t, 3> kUnloadBank{
                1u,
                (sizeof(uint32_t) << 16u) | 0x06u,
                0x8000u,
            };
            (void)host.writeGuest(kSendAddress,
                                  kUnloadBank.data(),
                                  sizeof(kUnloadBank));
            ps2x::iop::RpcRequest unload{};
            unload.sid = kMainSid;
            unload.function = 0x4Du;
            unload.send = {kSendAddress, sizeof(kUnloadBank)};
            unload.receive = {kReceiveAddress, 12u};
            const ps2x::iop::RpcResult unloadResult = subsystem.handleRpc(unload);
            t.IsTrue(unloadResult.handled,
                     "989snd command 0x06 should unload a sound bank");
            t.Equals(host.readWord(kReceiveAddress), 0xFFFFFFFFu,
                     "the unload response should retain its leading sentinel");
            t.Equals(host.readWord(kReceiveAddress + 4u), 0u,
                     "the unload command should return zero");
            t.Equals(host.readWord(kReceiveAddress + 8u), 0xFFFFFFFFu,
                     "the unload response should retain its trailing sentinel");
            t.Equals(host.audioCalls, 1u,
                     "the audio backend should observe the bank unload");
            t.Equals(host.lastAudioFunction, 0x06u,
                     "the backend notification should identify command 0x06");
            t.Equals(host.lastAudioSend.address, kSendAddress + 8u,
                     "the backend should receive the bank-handle payload address");
            t.Equals(host.lastAudioSend.size, sizeof(uint32_t),
                     "the backend should receive exactly one bank handle");
            t.Equals(host.audioBankUnloads, 1u,
                     "an unloaded bank should retire its backend copy once");
            t.Equals(host.lastUnloadedAudioBankHandle, 0x8000u,
                     "the unload notification should identify the released bank");

            constexpr std::array<uint32_t, 4> kMalformedUnload{
                1u,
                (2u * sizeof(uint32_t) << 16u) | 0x06u,
                0x8000u,
                0u,
            };
            (void)host.writeGuest(kSendAddress,
                                  kMalformedUnload.data(),
                                  sizeof(kMalformedUnload));
            unload.send.size = sizeof(kMalformedUnload);
            t.IsFalse(subsystem.handleRpc(unload).handled,
                      "989snd command 0x06 should reject a two-word payload");
            t.Equals(host.audioCalls, 1u,
                     "a malformed unload must not reach the audio backend");

            load.send.size = sizeof(uint32_t);
            t.IsFalse(subsystem.handleRpc(load).handled,
                      "the loading server should reject a malformed request size");

            const std::array<uint32_t, 2> outOfRangeSource{0x100u, 0u};
            (void)host.writeGuest(kSendAddress,
                                  outOfRangeSource.data(),
                                  sizeof(outOfRangeSource));
            load.send.size = sizeof(outOfRangeSource);
            t.IsTrue(subsystem.handleRpc(load).handled,
                     "a valid loading RPC should complete even when its source is invalid");
            t.Equals(host.readWord(kReceiveAddress), 0u,
                     "an out-of-range CD source should return a null bank handle");
            t.IsTrue(host.hasLog(
                         "[sony-989snd] container header is outside the CD image lbn=0x100 offset=0x0"),
                     "an out-of-range source should produce a precise diagnostic");
        });

        tc.Run("Sony 989snd parses and completes multi-command batches", [](TestCase &t)
        {
            FakeIopHost host;
            ps2x::iop::IopSubsystem subsystem(host);

            constexpr uint32_t kSid = 0x00123456u;
            constexpr uint32_t kSendAddress = 0x1000u;
            constexpr uint32_t kReceiveAddress = 0x1100u;
            constexpr uint32_t kCompletionAddress = 0x1200u;
            constexpr uint32_t kDestinationAddress = 0x2000u;
            constexpr uint32_t kSectorSize = 2048u;

            std::vector<uint8_t> image(3u * kSectorSize, 0u);
            for (uint32_t index = 0u; index < kSectorSize; ++index)
            {
                image[kSectorSize + index] =
                    static_cast<uint8_t>((index * 13u + 7u) & 0xFFu);
            }
            host.hostFileContents.emplace("fake/disc.iso", image);

            (void)host.writeWord(kSendAddress, kCompletionAddress);
            (void)host.writeWord(kCompletionAddress, 1u);
            ps2x::iop::RpcRequest start{};
            start.sid = kSid;
            start.function = 0u;
            start.send = {kSendAddress, sizeof(uint32_t)};
            start.receive = {kReceiveAddress, 12u};
            t.IsTrue(subsystem.handleRpc(start).handled,
                     "989snd start should configure the data-read completion word");

            constexpr std::array<uint32_t, 10> kBatch{
                4u,
                (8u << 16u) | 0x09u,
                5u,
                0x400u,
                0x08u,
                0x08u,
                (12u << 16u) | 0x38u,
                1u,
                1u,
                kDestinationAddress,
            };
            (void)host.writeGuest(kSendAddress, kBatch.data(), sizeof(kBatch));
            (void)host.writeWord(kReceiveAddress + 24u, 0xA5A5A5A5u);

            ps2x::iop::RpcRequest batch{};
            batch.sid = kSid;
            batch.function = 0x4Du;
            batch.send = {kSendAddress, sizeof(kBatch)};
            batch.receive = {kReceiveAddress, 24u};
            const ps2x::iop::RpcResult batchResult = subsystem.handleRpc(batch);
            t.IsTrue(batchResult.handled,
                     "a sequential four-command 989snd batch should be handled");
            t.Equals(batchResult.resultAddress, kReceiveAddress,
                     "a command batch should return its variable-size response buffer");
            t.Equals(host.readWord(kReceiveAddress), 0xFFFFFFFFu,
                     "a multi-command response should begin with a sentinel");
            for (uint32_t index = 1u; index <= 4u; ++index)
            {
                t.Equals(host.readWord(kReceiveAddress + index * sizeof(uint32_t)),
                         1u,
                         "each recognized command should retain its success word");
            }
            t.Equals(host.readWord(kReceiveAddress + 20u), 0xFFFFFFFFu,
                     "a multi-command response should end after one slot per command");
            t.Equals(host.readWord(kReceiveAddress + 24u), 0xA5A5A5A5u,
                     "a variable-size response should preserve adjacent guest memory");
            t.Equals(host.readWord(kCompletionAddress), 0u,
                     "the data-read command should signal completion");
            t.Equals(std::memcmp(host.memory.data() + kDestinationAddress,
                                 image.data() + kSectorSize,
                                 kSectorSize),
                     0,
                     "the data-read command should copy its requested CD sector");
            t.Equals(host.audioCalls, 4u,
                     "the native audio backend should observe every command in order");
            t.Equals(host.lastAudioFunction, 0x38u,
                     "the final backend notification should identify the data-read command");

            batch.send.size -= 1u;
            t.IsFalse(subsystem.handleRpc(batch).handled,
                      "a batch truncated inside its last payload should be rejected");
        });

        tc.Run("Sony 989snd pauses and continues sound groups", [](TestCase &t)
        {
            FakeIopHost host;
            ps2x::iop::IopSubsystem subsystem(host);

            constexpr uint32_t kSid = 0x00123456u;
            constexpr uint32_t kSendAddress = 0x1000u;
            constexpr uint32_t kReceiveAddress = 0x1100u;
            constexpr uint32_t kGroupMask = 0x1Du;
            constexpr std::array<uint32_t, 5> kBatch{
                2u,
                (sizeof(uint32_t) << 16u) | 0x16u,
                kGroupMask,
                (sizeof(uint32_t) << 16u) | 0x17u,
                kGroupMask,
            };
            (void)host.writeGuest(kSendAddress, kBatch.data(), sizeof(kBatch));

            ps2x::iop::RpcRequest batch{};
            batch.sid = kSid;
            batch.function = 0x4Du;
            batch.send = {kSendAddress, sizeof(kBatch)};
            batch.receive = {kReceiveAddress, 4u * sizeof(uint32_t)};
            const ps2x::iop::RpcResult result = subsystem.handleRpc(batch);

            t.IsTrue(result.handled,
                     "the retail pause/continue group-mask command shapes should be handled");
            t.Equals(host.readWord(kReceiveAddress), 0xFFFFFFFFu,
                     "the group-control response should begin with a sentinel");
            t.Equals(host.readWord(kReceiveAddress + sizeof(uint32_t)), 1u,
                     "pause-groups should retain the queue success word");
            t.Equals(host.readWord(kReceiveAddress + 2u * sizeof(uint32_t)), 1u,
                     "continue-groups should retain the queue success word");
            t.Equals(host.readWord(kReceiveAddress + 3u * sizeof(uint32_t)),
                     0xFFFFFFFFu,
                     "the group-control response should end with a sentinel");
            t.Equals(host.audioCalls, 2u,
                     "the audio backend should observe both group-control commands");
            t.Equals(host.lastAudioFunction, 0x17u,
                     "the final backend notification should identify continue-groups");
            t.Equals(host.lastAudioSend.address,
                     kSendAddress + 4u * sizeof(uint32_t),
                     "the backend should receive the continue group mask");
            t.Equals(host.lastAudioSend.size, sizeof(uint32_t),
                     "the backend group mask should occupy one word");

            std::array<uint32_t, 4> malformed{
                1u,
                (2u * sizeof(uint32_t) << 16u) | 0x16u,
                kGroupMask,
                0u,
            };
            (void)host.writeGuest(kSendAddress,
                                  malformed.data(),
                                  sizeof(malformed));
            batch.send.size = sizeof(malformed);
            t.IsFalse(subsystem.handleRpc(batch).handled,
                      "pause-groups should reject a two-word payload");
            t.Equals(host.audioCalls, 2u,
                     "a malformed group-control command must not reach the backend");
        });

        tc.Run("Sony 989snd play commands return distinct sound handles", [](TestCase &t)
        {
            FakeIopHost host;
            ps2x::iop::IopSubsystem subsystem(host);

            constexpr uint32_t kSid = 0x00123456u;
            constexpr uint32_t kSendAddress = 0x1000u;
            constexpr uint32_t kReceiveAddress = 0x1100u;
            constexpr std::array<uint32_t, 15> kBatch{
                2u,
                (24u << 16u) | 0x11u,
                0x100000u,
                3u,
                0x400u,
                0u,
                0u,
                0u,
                (24u << 16u) | 0x11u,
                0x100000u,
                7u,
                0x300u,
                180u,
                0u,
                0u,
            };
            (void)host.writeGuest(kSendAddress, kBatch.data(), sizeof(kBatch));

            ps2x::iop::RpcRequest batch{};
            batch.sid = kSid;
            batch.function = 0x4Du;
            batch.send = {kSendAddress, sizeof(kBatch)};
            batch.receive = {kReceiveAddress, 16u};
            const ps2x::iop::RpcResult result = subsystem.handleRpc(batch);

            const uint32_t firstHandle = host.readWord(kReceiveAddress + 4u);
            const uint32_t secondHandle = host.readWord(kReceiveAddress + 8u);
            t.IsTrue(result.handled,
                     "a batch containing synchronous play commands should be handled");
            t.Equals(host.readWord(kReceiveAddress), 0xFFFFFFFFu,
                     "a play-command response should begin with a sentinel");
            t.IsTrue(firstHandle != 0u,
                     "a successful synchronous play command should return a sound handle");
            t.IsTrue(secondHandle != 0u,
                     "each successful synchronous play command should return a sound handle");
            t.IsTrue(firstHandle != secondHandle,
                     "simultaneous sounds should receive distinct opaque handles");
            t.Equals(host.readWord(kReceiveAddress + 12u), 0xFFFFFFFFu,
                     "a play-command response should end after one slot per command");
            t.Equals(host.audioCalls, 2u,
                     "the native audio backend should observe both play commands");
            t.Equals(host.lastAudioFunction, 0x11u,
                     "the backend notification should preserve the play command ID");
            t.Equals(host.lastAudioReceive.address, kReceiveAddress + 8u,
                     "the backend should receive the matching per-command result slot");

            std::array<uint32_t, 8> parameterBatch{
                1u,
                (24u << 16u) | 0x21u,
                firstHandle,
                0x1Fu,
                0x400u,
                180u,
                0u,
                0u,
            };
            (void)host.writeGuest(kSendAddress,
                                  parameterBatch.data(),
                                  sizeof(parameterBatch));
            batch.send.size = sizeof(parameterBatch);
            batch.receive.size = 12u;
            t.IsTrue(subsystem.handleRpc(batch).handled,
                     "sound-parameter updates should be handled");
            t.Equals(host.readWord(kReceiveAddress + 4u), firstHandle,
                     "a sound-parameter update should return its active handle");
            t.Equals(host.lastAudioFunction, 0x21u,
                     "the backend should observe the sound-parameter update");

            parameterBatch[2] = 0xDEADBEEFu;
            (void)host.writeGuest(kSendAddress,
                                  parameterBatch.data(),
                                  sizeof(parameterBatch));
            t.IsTrue(subsystem.handleRpc(batch).handled,
                     "parameters for an unknown sound should still complete");
            t.Equals(host.readWord(kReceiveAddress + 4u), 0u,
                     "parameters for an unknown sound should return zero");

            parameterBatch[1] = (20u << 16u) | 0x21u;
            batch.send.size -= sizeof(uint32_t);
            (void)host.writeGuest(kSendAddress,
                                  parameterBatch.data(),
                                  batch.send.size);
            t.IsFalse(subsystem.handleRpc(batch).handled,
                      "a malformed sound-parameter payload should be rejected");

            std::array<uint32_t, 3> statusBatch{
                1u,
                (4u << 16u) | 0x19u,
                firstHandle,
            };
            (void)host.writeGuest(kSendAddress,
                                  statusBatch.data(),
                                  sizeof(statusBatch));
            batch.send.size = sizeof(statusBatch);
            batch.receive.size = 12u;
            t.IsTrue(subsystem.handleRpc(batch).handled,
                     "a sound-status query should be handled");
            t.Equals(host.readWord(kReceiveAddress + 4u), firstHandle,
                     "an active sound-status query should echo its sound handle");

            std::array<uint32_t, 3> stopBatch{
                1u,
                (4u << 16u) | 0x15u,
                firstHandle,
            };
            (void)host.writeGuest(kSendAddress,
                                  stopBatch.data(),
                                  sizeof(stopBatch));
            batch.send.size = sizeof(stopBatch);
            t.IsTrue(subsystem.handleRpc(batch).handled,
                     "a stop-sound command should be handled");
            t.Equals(host.readWord(kReceiveAddress), 0xFFFFFFFFu,
                     "a stop-sound response should begin with a sentinel");
            t.Equals(host.readWord(kReceiveAddress + 4u), 1u,
                     "a stop-sound command should retain its success word");
            t.Equals(host.readWord(kReceiveAddress + 8u), 0xFFFFFFFFu,
                     "a stop-sound response should end with a sentinel");
            t.Equals(host.lastAudioFunction, 0x15u,
                     "the backend should observe the stop-sound command");

            statusBatch[2] = firstHandle;
            (void)host.writeGuest(kSendAddress,
                                  statusBatch.data(),
                                  sizeof(statusBatch));
            batch.send.size = sizeof(statusBatch);
            t.IsTrue(subsystem.handleRpc(batch).handled,
                     "a stopped sound should remain queryable");
            t.Equals(host.readWord(kReceiveAddress + 4u), 0u,
                     "a stopped sound should no longer report active");

            statusBatch[2] = secondHandle;
            (void)host.writeGuest(kSendAddress,
                                  statusBatch.data(),
                                  sizeof(statusBatch));
            t.IsTrue(subsystem.handleRpc(batch).handled,
                     "stopping one sound should leave other sounds queryable");
            t.Equals(host.readWord(kReceiveAddress + 4u), secondHandle,
                     "stopping one sound should not retire another handle");

            stopBatch[1] = (8u << 16u) | 0x15u;
            stopBatch[2] = secondHandle;
            (void)host.writeGuest(kSendAddress,
                                  stopBatch.data(),
                                  sizeof(stopBatch));
            batch.send.size = sizeof(stopBatch);
            t.IsFalse(subsystem.handleRpc(batch).handled,
                      "a malformed stop-sound payload should be rejected");

            statusBatch[2] = 0xDEADBEEFu;
            (void)host.writeGuest(kSendAddress,
                                  statusBatch.data(),
                                  sizeof(statusBatch));
            batch.send.size = sizeof(statusBatch);
            t.IsTrue(subsystem.handleRpc(batch).handled,
                     "a status query for an unknown sound should still complete");
            t.Equals(host.readWord(kReceiveAddress + 4u), 0u,
                     "an unknown sound handle should report that playback has stopped");

            statusBatch[2] = 0xFFFFFFFFu;
            (void)host.writeGuest(kSendAddress,
                                  statusBatch.data(),
                                  sizeof(statusBatch));
            t.IsTrue(subsystem.handleRpc(batch).handled,
                     "the all-sounds status sentinel should be handled");
            t.Equals(host.readWord(kReceiveAddress + 4u), 0xFFFFFFFFu,
                     "the all-sounds status sentinel should remain active");

            subsystem.reset();
            statusBatch[2] = firstHandle;
            (void)host.writeGuest(kSendAddress,
                                  statusBatch.data(),
                                  sizeof(statusBatch));
            t.IsTrue(subsystem.handleRpc(batch).handled,
                     "sound status should remain queryable after reset");
            t.Equals(host.readWord(kReceiveAddress + 4u), 0u,
                     "reset should clear every tracked sound handle");
        });

        tc.Run("Sony 989snd VAG stream play returns a tracked handle", [](TestCase &t)
        {
            FakeIopHost host;
            ps2x::iop::IopSubsystem subsystem(host);

            constexpr uint32_t kSid = 0x00123456u;
            constexpr uint32_t kSendAddress = 0x1000u;
            constexpr uint32_t kReceiveAddress = 0x1100u;
            std::array<uint32_t, 9> playBatch{
                1u,
                (28u << 16u) | 0x2Cu,
                0x001790C6u,
                0u,
                0x00040000u,
                0x01000000u,
                0u,
                0x20000000u,
                0u,
            };
            (void)host.writeGuest(kSendAddress,
                                  playBatch.data(),
                                  sizeof(playBatch));

            ps2x::iop::RpcRequest request{};
            request.sid = kSid;
            request.function = 0x4Du;
            request.send = {kSendAddress, sizeof(playBatch)};
            request.receive = {kReceiveAddress, 12u};
            const ps2x::iop::RpcResult result = subsystem.handleRpc(request);

            const uint32_t soundHandle = host.readWord(kReceiveAddress + 4u);
            t.IsTrue(result.handled,
                     "a 28-byte VAG stream-by-location command should be handled");
            t.Equals(host.readWord(kReceiveAddress), 0xFFFFFFFFu,
                     "a VAG stream response should begin with a sentinel");
            t.IsTrue(soundHandle != 0u,
                     "VAG stream play should return an opaque sound handle");
            t.Equals(host.readWord(kReceiveAddress + 8u), 0xFFFFFFFFu,
                     "a VAG stream response should end with a sentinel");
            t.Equals(host.audioCalls, 1u,
                     "the native audio backend should observe VAG stream play");
            t.Equals(host.lastAudioFunction, 0x2Cu,
                     "the backend notification should preserve the VAG stream command ID");
            t.Equals(host.lastAudioSend.address, kSendAddress + 8u,
                     "the backend should receive the VAG stream payload");
            t.Equals(host.lastAudioSend.size, 28u,
                     "the backend should receive the complete VAG stream payload");
            t.Equals(host.lastAudioReceive.address, kReceiveAddress + 4u,
                     "the backend should receive the matching stream-handle slot");

            const std::array<uint32_t, 3> pauseBatch{
                1u,
                (4u << 16u) | 0x2Du,
                soundHandle,
            };
            (void)host.writeGuest(kSendAddress,
                                  pauseBatch.data(),
                                  sizeof(pauseBatch));
            request.send.size = sizeof(pauseBatch);
            t.IsTrue(subsystem.handleRpc(request).handled,
                     "a VAG stream pause command should be handled");
            t.Equals(host.readWord(kReceiveAddress), 0xFFFFFFFFu,
                     "a VAG pause response should begin with a sentinel");
            t.Equals(host.readWord(kReceiveAddress + 4u), 1u,
                     "VAG pause should retain the command success word");
            t.Equals(host.readWord(kReceiveAddress + 8u), 0xFFFFFFFFu,
                     "a VAG pause response should end with a sentinel");
            t.Equals(host.lastAudioFunction, 0x2Du,
                     "the backend should observe the VAG pause command");

            const std::array<uint32_t, 3> continueBatch{
                1u,
                (4u << 16u) | 0x2Eu,
                soundHandle,
            };
            (void)host.writeGuest(kSendAddress,
                                  continueBatch.data(),
                                  sizeof(continueBatch));
            request.send.size = sizeof(continueBatch);
            t.IsTrue(subsystem.handleRpc(request).handled,
                     "a VAG stream continue command should be handled");
            t.Equals(host.readWord(kReceiveAddress), 0xFFFFFFFFu,
                     "a VAG continue response should begin with a sentinel");
            t.Equals(host.readWord(kReceiveAddress + 4u), 1u,
                     "VAG continue should retain the command success word");
            t.Equals(host.readWord(kReceiveAddress + 8u), 0xFFFFFFFFu,
                     "a VAG continue response should end with a sentinel");
            t.Equals(host.lastAudioFunction, 0x2Eu,
                     "the backend should observe the VAG continue command");

            std::array<uint32_t, 3> timeRemainingBatch{
                1u,
                (4u << 16u) | 0x32u,
                soundHandle,
            };
            (void)host.writeGuest(kSendAddress,
                                  timeRemainingBatch.data(),
                                  sizeof(timeRemainingBatch));
            request.send.size = sizeof(timeRemainingBatch);
            t.IsTrue(subsystem.handleRpc(request).handled,
                     "a VAG time-remaining query should be handled");
            t.Equals(host.readWord(kReceiveAddress), 0xFFFFFFFFu,
                     "a VAG time response should begin with a sentinel");
            t.Equals(host.readWord(kReceiveAddress + 4u), 0u,
                     "a host-backed VAG stream without finite duration should return zero");
            t.Equals(host.readWord(kReceiveAddress + 8u), 0xFFFFFFFFu,
                     "a VAG time response should end with a sentinel");
            t.Equals(host.lastAudioFunction, 0x32u,
                     "the backend should observe the VAG time query");

            timeRemainingBatch[2] = 0xDEADBEEFu;
            (void)host.writeGuest(kSendAddress,
                                  timeRemainingBatch.data(),
                                  sizeof(timeRemainingBatch));
            t.IsTrue(subsystem.handleRpc(request).handled,
                     "a VAG time query for an invalid handle should complete");
            t.Equals(host.readWord(kReceiveAddress + 4u), 0u,
                     "an invalid VAG handle should reproduce the retail zero result");

            std::array<uint32_t, 3> statusBatch{
                1u,
                (4u << 16u) | 0x19u,
                soundHandle,
            };
            (void)host.writeGuest(kSendAddress,
                                  statusBatch.data(),
                                  sizeof(statusBatch));
            request.send.size = sizeof(statusBatch);
            t.IsTrue(subsystem.handleRpc(request).handled,
                     "a VAG stream handle should be queryable");
            t.Equals(host.readWord(kReceiveAddress + 4u), soundHandle,
                     "a VAG stream handle should remain active");

            std::array<uint32_t, 21> bufferedBatch{
                7u,
                (8u << 16u) | 0x09u,
                1u,
                0u,
                (8u << 16u) | 0x09u,
                0u,
                0x199u,
                (8u << 16u) | 0x09u,
                3u,
                0x166u,
                (8u << 16u) | 0x09u,
                2u,
                0x333u,
                (8u << 16u) | 0x09u,
                4u,
                0x2CCu,
                (8u << 16u) | 0x09u,
                5u,
                0x400u,
                (4u << 16u) | 0x4Fu,
                soundHandle,
            };
            (void)host.writeGuest(kSendAddress,
                                  bufferedBatch.data(),
                                  sizeof(bufferedBatch));
            request.send.size = sizeof(bufferedBatch);
            request.receive.size = 9u * sizeof(uint32_t);
            t.IsTrue(subsystem.handleRpc(request).handled,
                     "the live seven-command stream-buffer query should be handled");
            t.Equals(host.readWord(kReceiveAddress), 0xFFFFFFFFu,
                     "a stream-buffer query response should begin with a sentinel");
            for (uint32_t index = 0u; index < 6u; ++index)
            {
                t.Equals(host.readWord(kReceiveAddress +
                                       (index + 1u) * sizeof(uint32_t)),
                         1u,
                         "the preceding parameter commands should retain success");
            }
            t.Equals(host.readWord(kReceiveAddress + 7u * sizeof(uint32_t)),
                     1u,
                     "an active HLE sound should report buffered");
            t.Equals(host.readWord(kReceiveAddress + 8u * sizeof(uint32_t)),
                     0xFFFFFFFFu,
                     "a stream-buffer query response should end with a sentinel");
            t.Equals(host.lastAudioFunction, 0x4Fu,
                     "the backend should observe the stream-buffer query");

            std::array<uint32_t, 3> unknownBufferedBatch{
                1u,
                (4u << 16u) | 0x4Fu,
                0xDEADBEEFu,
            };
            (void)host.writeGuest(kSendAddress,
                                  unknownBufferedBatch.data(),
                                  sizeof(unknownBufferedBatch));
            request.send.size = sizeof(unknownBufferedBatch);
            request.receive.size = 3u * sizeof(uint32_t);
            t.IsTrue(subsystem.handleRpc(request).handled,
                     "a stream-buffer query for an invalid handle should complete");
            t.Equals(host.readWord(kReceiveAddress + sizeof(uint32_t)),
                     0xFFFFFFFFu,
                     "an invalid stream handle should reproduce the retail -1 result");

            playBatch[1] = (24u << 16u) | 0x2Cu;
            (void)host.writeGuest(kSendAddress,
                                  playBatch.data(),
                                  8u * sizeof(uint32_t));
            request.send.size = 8u * sizeof(uint32_t);
            t.IsFalse(subsystem.handleRpc(request).handled,
                      "VAG stream-by-location should reject a truncated payload");
            t.Equals(host.audioCalls, 14u,
                     "a malformed VAG stream command must not reach the audio backend");
        });

        tc.Run("Sony 989snd stop-all RPC retires sounds and preserves the shared result", [](TestCase &t)
        {
            FakeIopHost host;
            ps2x::iop::IopSubsystem subsystem(host);

            constexpr uint32_t kSid = 0x00123456u;
            constexpr uint32_t kSendAddress = 0x1000u;
            constexpr uint32_t kReceiveAddress = 0x1100u;
            constexpr std::array<uint32_t, 8> kPlayBatch{
                1u,
                (24u << 16u) | 0x11u,
                0x100000u,
                3u,
                0x400u,
                0u,
                0u,
                0u,
            };
            (void)host.writeGuest(kSendAddress,
                                  kPlayBatch.data(),
                                  sizeof(kPlayBatch));

            ps2x::iop::RpcRequest request{};
            request.sid = kSid;
            request.function = 0x4Du;
            request.send = {kSendAddress, sizeof(kPlayBatch)};
            request.receive = {kReceiveAddress, 12u};
            t.IsTrue(subsystem.handleRpc(request).handled,
                     "a play command should establish an active sound");
            const uint32_t soundHandle = host.readWord(kReceiveAddress + 4u);
            t.IsTrue(soundHandle != 0u,
                     "a play command should return a sound handle");

            constexpr std::array<uint32_t, 6> kOpenStream{
                0x400u,
                0x1000u,
                0x400u,
                0u,
                5u,
                3u,
            };
            (void)host.writeGuest(kSendAddress,
                                  kOpenStream.data(),
                                  sizeof(kOpenStream));
            request.function = 0x3Bu;
            request.send.size = sizeof(kOpenStream);
            t.IsTrue(subsystem.handleRpc(request).handled,
                     "streaming open should seed a non-default shared result");
            const uint32_t previousResult = host.readWord(kReceiveAddress + 4u);
            t.IsTrue(previousResult != 0u && previousResult != 1u,
                     "streaming open should return a distinct resource handle");

            request.function = 0x18u;
            request.send = {};
            const ps2x::iop::RpcResult stopAllResult = subsystem.handleRpc(request);
            t.IsTrue(stopAllResult.handled,
                     "the stop-all-sounds RPC should be handled");
            t.Equals(stopAllResult.resultAddress, kReceiveAddress,
                     "the stop-all-sounds RPC should return its completion record");
            t.Equals(host.readWord(kReceiveAddress), 0xFFFFFFFFu,
                     "the stop-all-sounds response should begin with a sentinel");
            t.Equals(host.readWord(kReceiveAddress + 4u), previousResult,
                     "the stop-all-sounds RPC should preserve the shared result word");
            t.Equals(host.readWord(kReceiveAddress + 8u), 0xFFFFFFFFu,
                     "the stop-all-sounds response should end with a sentinel");
            t.Equals(host.lastAudioSid, kSid,
                     "the backend should receive the 989snd server ID");
            t.Equals(host.lastAudioFunction, 0x18u,
                     "the backend should observe the stop-all-sounds RPC");
            t.Equals(host.lastAudioSend.size, 0u,
                     "the backend stop-all notification should have no payload");

            const std::array<uint32_t, 3> statusBatch{
                1u,
                (4u << 16u) | 0x19u,
                soundHandle,
            };
            (void)host.writeGuest(kSendAddress,
                                  statusBatch.data(),
                                  sizeof(statusBatch));
            request.function = 0x4Du;
            request.send = {kSendAddress, sizeof(statusBatch)};
            t.IsTrue(subsystem.handleRpc(request).handled,
                     "a stopped sound should remain queryable");
            t.Equals(host.readWord(kReceiveAddress + 4u), 0u,
                     "stop all should retire every active sound handle");

            request.function = 0x18u;
            request.send = {kSendAddress, sizeof(uint32_t)};
            t.IsFalse(subsystem.handleRpc(request).handled,
                      "the stop-all-sounds RPC should reject a payload");
        });

        tc.Run("Sony 989snd streaming position follows virtual SPU time", [](TestCase &t)
        {
            FakeIopHost host;
            ps2x::iop::IopSubsystem subsystem(host);

            constexpr uint32_t kSid = 0x00123456u;
            constexpr uint32_t kSendAddress = 0x1000u;
            constexpr uint32_t kReceiveAddress = 0x1100u;
            auto call = [&](uint32_t function, uint32_t sendSize) {
                ps2x::iop::RpcRequest request{};
                request.sid = kSid;
                request.function = function;
                request.send = {sendSize == 0u ? 0u : kSendAddress, sendSize};
                request.receive = {kReceiveAddress, 12u};
                return subsystem.handleRpc(request);
            };

            constexpr std::array<uint32_t, 6> kOpen{
                0x400u, 0x1000u, 0x400u, 0u, 5u, 3u,
            };
            t.IsTrue(host.writeGuest(kSendAddress, kOpen.data(), sizeof(kOpen)),
                     "streaming-open request should fit in fake guest memory");
            t.IsTrue(call(0x3Bu, sizeof(kOpen)).handled,
                     "streaming-open RPC should be handled");
            const uint32_t workArea = host.readWord(kReceiveAddress + 4u);
            t.IsTrue(workArea != 0u, "streaming-open RPC should allocate a work area");

            constexpr std::array<uint32_t, 2> kTransfer{0x400u, 0u};
            (void)host.writeGuest(kSendAddress, kTransfer.data(), sizeof(kTransfer));
            t.IsTrue(call(0x5Au, sizeof(kTransfer)).handled,
                     "streaming transfer should prime playback");

            const std::array<uint32_t, 5> playback{
                workArea, 0x400u, 0u, 44100u, 2u,
            };
            (void)host.writeGuest(kSendAddress, playback.data(), sizeof(playback));
            t.IsTrue(call(0x3Eu, sizeof(playback)).handled,
                     "streaming playback should start at 44.1 kHz");

            auto expectPosition = [&](uint64_t timeNanoseconds,
                                      uint32_t expected,
                                      const std::string &reason) {
                host.virtualTime = timeNanoseconds;
                t.IsTrue(call(0x5Bu, 0u).handled,
                         "streaming-position RPC should be handled");
                t.Equals(host.readWord(kReceiveAddress + 4u), expected, reason);
            };

            expectPosition(0u, 0u,
                           "polling at the playback start must not advance the cursor");
            expectPosition(90'703u, 0x4u,
                           "four samples should advance NAX by one 16-bit data word");
            expectPosition(634'921u, 0x12u,
                           "crossing an ADPCM block should skip its header word");
            expectPosition(16'666'667u, 0x1A4u,
                           "one video tick should expose the exact SPU NAX word");
            expectPosition(16'666'667u, 0x1A4u,
                           "repeated polling at one virtual instant must be stable");
            expectPosition(162'539'683u, 0x2u,
                           "the NAX cursor should wrap to the first data word");
            expectPosition(163'174'604u, 0x12u,
                           "the cursor should skip the next block header after wrapping");
            expectPosition(1'000'000'000u, 0x272u,
                           "one second should retain the exact wrapped NAX word");
        });

        tc.Run("Sony 989snd forwards DMA staging data to the host audio backend", [](TestCase &t)
        {
            FakeIopHost host;
            ps2x::iop::IopSubsystem subsystem(host);

            constexpr uint32_t kSid = 0x00123456u;
            constexpr uint32_t kSendAddress = 0x1000u;
            constexpr uint32_t kReceiveAddress = 0x1100u;
            auto call = [&](uint32_t function, uint32_t sendSize) {
                ps2x::iop::RpcRequest request{};
                request.sid = kSid;
                request.function = function;
                request.send = {sendSize == 0u ? 0u : kSendAddress, sendSize};
                request.receive = {kReceiveAddress, 12u};
                return subsystem.handleRpc(request);
            };

            constexpr std::array<uint32_t, 6> kOpen{
                0x400u, 0x1000u, 0x400u, 0u, 5u, 3u,
            };
            (void)host.writeGuest(kSendAddress, kOpen.data(), sizeof(kOpen));
            t.IsTrue(call(0x3Bu, sizeof(kOpen)).handled,
                     "streaming open should complete");
            const uint32_t workArea = host.readWord(kReceiveAddress + 4u);
            t.Equals(host.audioCalls, 1u,
                     "streaming open should notify the host audio backend");
            t.Equals(host.lastAudioFunction, 0x3Bu,
                     "the host should observe the streaming-open command");
            t.Equals(host.lastAudioReceive.address, workArea,
                     "streaming open should identify the allocated DMA staging area");
            t.Equals(host.lastAudioReceive.size, 0x400u,
                     "streaming open should report the staging-area size");
            t.IsTrue(host.lastAudioReceive.domain == GuestMemoryDomain::IopRam,
                     "streaming open should identify the staging area as IOP RAM");

            std::array<uint8_t, 0x400> encoded{};
            for (size_t index = 0u; index < encoded.size(); ++index)
                encoded[index] = static_cast<uint8_t>(index);
            std::memcpy(host.iopMemory.data() + workArea,
                        encoded.data(),
                        encoded.size());

            constexpr std::array<uint32_t, 2> kTransfer{0x400u, 0x1000u};
            (void)host.writeGuest(kSendAddress, kTransfer.data(), sizeof(kTransfer));
            t.IsTrue(call(0x5Au, sizeof(kTransfer)).handled,
                     "streaming transfer should complete");
            t.Equals(host.audioCalls, 2u,
                     "streaming transfer should notify the host audio backend");
            t.Equals(host.lastAudioFunction, 0x5Au,
                     "the host should observe the streaming-submit command");
            t.Equals(host.lastAudioReceive.address, workArea,
                     "streaming submit should expose the DMA staging area");
            t.IsTrue(host.lastAudioReceive.domain == GuestMemoryDomain::IopRam,
                     "streaming submit should preserve the IOP RAM domain");
            t.Equals(host.lastAudioPayload.size(), encoded.size(),
                     "the host should receive the complete encoded chunk");
            t.Equals(host.lastAudioPayload.front(), encoded.front(),
                     "the first encoded byte should reach the host");
            t.Equals(host.lastAudioPayload.back(), encoded.back(),
                     "the last encoded byte should reach the host");

            const std::array<uint32_t, 5> playback{
                workArea, 0x400u, 0u, 44100u, 2u,
            };
            (void)host.writeGuest(kSendAddress, playback.data(), sizeof(playback));
            t.IsTrue(call(0x3Eu, sizeof(playback)).handled,
                     "streaming start should complete");
            t.Equals(host.audioCalls, 3u,
                     "streaming start should notify the host audio backend");
            t.Equals(host.lastAudioFunction, 0x3Eu,
                     "the host should observe the streaming-start command");

            t.IsTrue(call(0x3Du, 0u).handled,
                     "streaming stop should complete");
            t.Equals(host.lastAudioFunction, 0x3Du,
                     "the host should observe the streaming-stop command");
            t.IsTrue(call(0x3Cu, 0u).handled,
                     "streaming close should complete");
            t.Equals(host.lastAudioFunction, 0x3Cu,
                     "the host should observe the streaming-close command");
        });

        tc.Run("Sony 989snd stop freezes an open stream and close releases it", [](TestCase &t)
        {
            FakeIopHost host;
            ps2x::iop::IopSubsystem subsystem(host);

            constexpr uint32_t kSid = 0x00123456u;
            constexpr uint32_t kSendAddress = 0x1000u;
            constexpr uint32_t kReceiveAddress = 0x1100u;
            auto call = [&](uint32_t function, uint32_t sendSize) {
                ps2x::iop::RpcRequest request{};
                request.sid = kSid;
                request.function = function;
                request.send = {sendSize == 0u ? 0u : kSendAddress, sendSize};
                request.receive = {kReceiveAddress, 12u};
                return subsystem.handleRpc(request);
            };

            constexpr std::array<uint32_t, 6> kOpen{
                0x400u, 0x1000u, 0x400u, 0u, 5u, 3u,
            };
            (void)host.writeGuest(kSendAddress, kOpen.data(), sizeof(kOpen));
            t.IsTrue(call(0x3Bu, sizeof(kOpen)).handled,
                     "streaming-open RPC should be handled");
            const uint32_t workArea = host.readWord(kReceiveAddress + 4u);
            t.IsTrue(workArea != 0u,
                     "streaming-open RPC should allocate a work area");

            constexpr std::array<uint32_t, 2> kTransfer{0x400u, 0u};
            (void)host.writeGuest(kSendAddress, kTransfer.data(), sizeof(kTransfer));
            t.IsTrue(call(0x5Au, sizeof(kTransfer)).handled,
                     "streaming transfer should prime playback");

            const std::array<uint32_t, 5> playback{
                workArea, 0x400u, 0u, 44100u, 2u,
            };
            (void)host.writeGuest(kSendAddress, playback.data(), sizeof(playback));
            t.IsTrue(call(0x3Eu, sizeof(playback)).handled,
                     "streaming playback should start");

            host.virtualTime = 16'666'667u;
            t.IsTrue(call(0x3Du, 0u).handled,
                     "streaming-stop RPC should be handled");
            t.Equals(host.readWord(kReceiveAddress + 4u), workArea,
                     "streaming stop should preserve the shared response result");

            host.virtualTime = 1'000'000'000u;
            t.IsTrue(call(0x5Bu, 0u).handled,
                     "a stopped stream should remain open");
            t.Equals(host.readWord(kReceiveAddress + 4u), 0x1A4u,
                     "a stopped stream should retain its frozen encoded-byte cursor");

            t.IsTrue(call(0x3Cu, 0u).handled,
                     "streaming-close RPC should be handled");
            t.Equals(host.readWord(kReceiveAddress + 4u), 0x1A4u,
                     "streaming close should preserve the shared response result");
            t.IsFalse(call(0x5Bu, 0u).handled,
                      "streaming position should be unavailable after close");
            t.IsTrue(call(0x3Cu, 0u).handled,
                     "streaming close should be idempotent");

            t.IsFalse(call(0x3Du, sizeof(uint32_t)).handled,
                      "streaming stop should reject an unexpected request body");
        });

        tc.Run("Sony 989snd stop-all-streams freezes playback and preserves the shared result", [](TestCase &t)
        {
            FakeIopHost host;
            ps2x::iop::IopSubsystem subsystem(host);

            constexpr uint32_t kSid = 0x00123456u;
            constexpr uint32_t kSendAddress = 0x1000u;
            constexpr uint32_t kReceiveAddress = 0x1100u;
            auto call = [&](uint32_t function, uint32_t sendSize) {
                ps2x::iop::RpcRequest request{};
                request.sid = kSid;
                request.function = function;
                request.send = {sendSize == 0u ? 0u : kSendAddress, sendSize};
                request.receive = {kReceiveAddress, 12u};
                return subsystem.handleRpc(request);
            };

            constexpr std::array<uint32_t, 6> kOpen{
                0x400u, 0x1000u, 0x400u, 0u, 5u, 3u,
            };
            (void)host.writeGuest(kSendAddress, kOpen.data(), sizeof(kOpen));
            t.IsTrue(call(0x3Bu, sizeof(kOpen)).handled,
                     "streaming-open RPC should be handled");
            const uint32_t workArea = host.readWord(kReceiveAddress + 4u);
            t.IsTrue(workArea != 0u,
                     "streaming-open RPC should allocate a work area");

            constexpr std::array<uint32_t, 2> kTransfer{0x400u, 0u};
            (void)host.writeGuest(kSendAddress, kTransfer.data(), sizeof(kTransfer));
            t.IsTrue(call(0x5Au, sizeof(kTransfer)).handled,
                     "streaming transfer should prime playback");

            const std::array<uint32_t, 5> playback{
                workArea, 0x400u, 0u, 44100u, 2u,
            };
            (void)host.writeGuest(kSendAddress, playback.data(), sizeof(playback));
            t.IsTrue(call(0x3Eu, sizeof(playback)).handled,
                     "streaming playback should start");

            host.virtualTime = 16'666'667u;
            t.IsTrue(call(0x5Bu, 0u).handled,
                     "streaming position should expose a non-default shared result");
            constexpr uint32_t kFrozenPosition = 0x1A4u;
            t.Equals(host.readWord(kReceiveAddress + 4u), kFrozenPosition,
                     "one video tick should advance the streaming cursor");

            const ps2x::iop::RpcResult stopAllResult = call(0x34u, 0u);
            t.IsTrue(stopAllResult.handled,
                     "the stop-all-streams RPC should be handled");
            t.Equals(stopAllResult.resultAddress, kReceiveAddress,
                     "stop all streams should return its completion record");
            t.Equals(host.readWord(kReceiveAddress), 0xFFFFFFFFu,
                     "the stop-all-streams response should begin with a sentinel");
            t.Equals(host.readWord(kReceiveAddress + 4u), kFrozenPosition,
                     "stop all streams should preserve the shared result word");
            t.Equals(host.readWord(kReceiveAddress + 8u), 0xFFFFFFFFu,
                     "the stop-all-streams response should end with a sentinel");
            t.Equals(host.lastAudioFunction, 0x34u,
                     "the backend should observe the stop-all-streams RPC");

            host.virtualTime = 1'000'000'000u;
            t.IsTrue(call(0x5Bu, 0u).handled,
                     "a stopped stream should remain open");
            t.Equals(host.readWord(kReceiveAddress + 4u), kFrozenPosition,
                     "stop all streams should freeze the encoded-byte cursor");

            t.IsFalse(call(0x34u, sizeof(uint32_t)).handled,
                      "stop all streams should reject an unexpected request body");
        });

        tc.Run("two subsystem instances isolate profile state and reset deterministically", [](TestCase &t)
        {
            FakeIopHost hostA;
            FakeIopHost hostB;
            ps2x::iop::IopSubsystem subsystemA(hostA);
            ps2x::iop::IopSubsystem subsystemB(hostB);
            std::string error;
            t.IsTrue(subsystemA.configure({"SLUS_205.78", 0u, 0u}, &error),
                     "first LotR instance should configure");
            t.IsTrue(subsystemB.configure({"SLUS_205.78", 0u, 0u}, &error),
                     "second LotR instance should configure");

            ps2x::iop::RpcRequest request{};
            request.sid = 0x00012345u;
            request.receive = {0x1000u, 8u};

            t.IsTrue(subsystemA.handleRpc(request).handled,
                     "first instance should handle LotR sound RPC");
            t.Equals(hostA.readWord(0x1004u), 1u,
                     "first instance should start its counter at one");
            (void)subsystemA.handleRpc(request);
            t.Equals(hostA.readWord(0x1004u), 2u,
                     "first instance should advance independently");

            t.IsTrue(subsystemB.handleRpc(request).handled,
                     "second instance should handle LotR sound RPC");
            t.Equals(hostB.readWord(0x1004u), 1u,
                     "second instance must not inherit the first counter");

            subsystemA.reset();
            (void)subsystemA.handleRpc(request);
            t.Equals(hostA.readWord(0x1004u), 1u,
                     "reset should restore per-instance service state");
        });

        tc.Run("TSNDDRV uses profile checksum bindings without writing invalid ports", [](TestCase &t)
        {
            FakeIopHost host(0x02000000u);
            ps2x::iop::IopSubsystem subsystem(host);
            std::string error;
            t.IsTrue(subsystem.configure({"slus_201.84", 0u, 0u}, &error),
                     "RECVX profile should configure for TSNDDRV command testing");

            constexpr uint32_t kResponseAddress = 0x1000u;
            ps2x::iop::RpcRequest stateRequest{};
            stateRequest.sid = 1u;
            stateRequest.function = 0x12u;
            stateRequest.receive = {kResponseAddress, sizeof(uint32_t)};
            t.IsTrue(subsystem.handleRpc(stateRequest).handled,
                     "TSNDDRV should return its configured status buffer");
            const uint32_t statusAddress = host.readWord(kResponseAddress);
            t.IsTrue(statusAddress != 0u, "TSNDDRV status buffer should be allocated");

            constexpr int16_t kChecksum = 0x1234;
            t.IsTrue(host.writeGuest(0x01E0EF10u, &kChecksum, sizeof(kChecksum)),
                     "RECVX primary checksum binding should be writable in the fake guest");

            constexpr uint32_t kCommandAddress = 0x2000u;
            std::array<uint8_t, 8> command{};
            command[0] = 0x29u;
            command[1] = 0u;
            t.IsTrue(host.writeGuest(kCommandAddress, command.data(), command.size()),
                     "valid TSNDDRV command should be writable");

            ps2x::iop::RpcRequest commandRequest{};
            commandRequest.sid = 0u;
            commandRequest.function = 0u;
            commandRequest.send = {kCommandAddress, static_cast<uint32_t>(command.size())};
            t.IsTrue(subsystem.handleRpc(commandRequest).handled,
                     "TSNDDRV should handle the characterized command queue");

            int16_t writtenChecksum = 0;
            t.IsTrue(host.readGuest(statusAddress + 0x26u,
                                    &writtenChecksum,
                                    sizeof(writtenChecksum)),
                     "TSNDDRV SE checksum slot should be readable");
            t.Equals(writtenChecksum, kChecksum,
                     "valid port should mirror the profile-bound checksum table");

            constexpr uint32_t kPastStatusAddress = 0x44u;
            constexpr uint16_t kSentinel = 0xBEEFu;
            t.IsTrue(host.writeGuest(statusAddress + kPastStatusAddress,
                                     &kSentinel,
                                     sizeof(kSentinel)),
                     "sentinel after the status structure should be writable");
            command[1] = 0x0Fu;
            (void)host.writeGuest(kCommandAddress, command.data(), command.size());
            (void)subsystem.handleRpc(commandRequest);

            uint16_t sentinelAfter = 0u;
            (void)host.readGuest(statusAddress + kPastStatusAddress,
                                 &sentinelAfter,
                                 sizeof(sentinelAfter));
            t.Equals(sentinelAfter, kSentinel,
                     "invalid port must not overwrite memory past the 0x42-byte status structure");
        });

        tc.Run("RECVX reset clears CRI object maps without global state", [](TestCase &t)
        {
            FakeIopHost host(0x02000000u);
            ps2x::iop::IopSubsystem subsystem(host);
            std::string error;
            t.IsTrue(subsystem.configure({"slus_201.84", 0u, 0u}, &error),
                     "RECVX profile should configure");

            constexpr uint32_t kSendAddress = 0x2000u;
            constexpr uint32_t kReceiveAddress = 0x2100u;
            host.writeWord(kSendAddress + 0u, 0u);
            host.writeWord(kSendAddress + 4u, 0x4000u);
            host.writeWord(kSendAddress + 8u, 0x100u);

            ps2x::iop::RpcRequest request{};
            request.sid = 0x7D000000u;
            request.function = 0x422u;
            request.send = {kSendAddress, 12u};
            request.receive = {kReceiveAddress, 4u};
            t.IsTrue(subsystem.handleRpc(request).handled,
                     "SJRMT create should be emulated by the RECVX profile");

            ps2x::iop::DebugSnapshot snapshot = subsystem.debugSnapshot();
            const ps2x::iop::DebugService *service =
                findService(snapshot, "CRI DTX");
            if (!service)
            {
                t.Fail("CRI DTX service should be visible in the debug snapshot");
                return;
            }
            t.Equals(metricValue(*service, "sjrmt_objects"), uint64_t{1},
                     "created CRI object should be tracked by this instance");

            subsystem.reset();
            snapshot = subsystem.debugSnapshot();
            service = findService(snapshot, "CRI DTX");
            if (!service)
            {
                t.Fail("CRI DTX service should survive reset");
                return;
            }
            t.Equals(metricValue(*service, "sjrmt_objects"), uint64_t{0},
                     "reset should clear CRI object maps");
        });

        tc.Run("reset closes profile-owned host file handles", [](TestCase &t)
        {
            FakeIopHost host;
            host.hostFileContents["translated/test.bin"] = {0x10u, 0x20u, 0x30u};

            ps2x::iop::IopSubsystem subsystem(host);
            std::string error;
            t.IsTrue(subsystem.configure({"SLUS_205.78", 0u, 0u}, &error),
                     "LotR profile should configure for file lifecycle testing");

            constexpr uint32_t kPathAddress = 0x1000u;
            constexpr uint32_t kReceiveAddress = 0x1100u;
            constexpr char kPath[] = "test.bin";
            t.IsTrue(host.writeGuest(kPathAddress, kPath, sizeof(kPath)),
                     "fake guest path should be writable");

            ps2x::iop::RpcRequest request{};
            request.sid = 0x0000FF01u;
            request.function = 0x08u;
            request.send = {kPathAddress, sizeof(kPath)};
            request.receive = {kReceiveAddress, 8u};
            t.IsTrue(subsystem.handleRpc(request).handled,
                     "LotR CLFILE open should be handled");
            t.Equals(host.openHostFiles.size(), size_t{1},
                     "open RPC should retain one opaque host file handle");

            ps2x::iop::DebugSnapshot snapshot = subsystem.debugSnapshot();
            const ps2x::iop::DebugService *service =
                findService(snapshot, "CLFILE");
            if (!service)
            {
                t.Fail("LotR CLFILE service should be visible before reset");
                return;
            }
            t.Equals(metricValue(*service, "open_files"), uint64_t{1},
                     "debug state should report the open file");

            subsystem.reset();
            t.IsTrue(host.openHostFiles.empty(),
                     "reset should release every retained host file handle");
            t.Equals(host.closedHostFileHandles.size(), size_t{1},
                     "host close callback should run exactly once");
            snapshot = subsystem.debugSnapshot();
            service = findService(snapshot, "CLFILE");
            if (!service)
            {
                t.Fail("LotR CLFILE service should survive reset");
                return;
            }
            t.Equals(metricValue(*service, "open_files"), uint64_t{0},
                     "reset should clear the CLFILE handle registry");
        });

#if defined(PS2X_TEST_IOP_PLUGIN_DIR)
        tc.Run("plugin module remains loaded through instances and unloads after subsystem destruction", [](TestCase &t)
        {
            const std::filesystem::path pluginDirectory(PS2X_TEST_IOP_PLUGIN_DIR);
#if defined(_WIN32)
            const std::filesystem::path pluginPath =
                pluginDirectory / "ps2_iop_fake_plugin.dll";
#else
            const std::filesystem::path pluginPath =
                pluginDirectory / "ps2_iop_fake_plugin.so";
#endif
            t.IsFalse(pluginModuleIsLoaded(pluginPath),
                      "synthetic plugin should not be loaded before discovery");
            {
                FakeIopHost host;
                ps2x::iop::IopSubsystem subsystem(host);
                subsystem.setPluginSearchPaths({pluginDirectory});
                std::string error;
                t.IsTrue(subsystem.loadPlugins(&error),
                         "synthetic plugins should load for lifetime testing");
                t.IsTrue(subsystem.configure({"synthetic_iop_test.elf",
                                              kSyntheticEntryPoint,
                                              kSyntheticCrc32},
                                             &error),
                         "synthetic plugin instance should be created");
                t.IsTrue(pluginModuleIsLoaded(pluginPath),
                         "module must stay loaded while a profile instance exists");
            }
            t.IsFalse(pluginModuleIsLoaded(pluginPath),
                      "module should unload after profile destruction and catalog teardown");
        });

        tc.Run("plugin discovery matches all identity fields and dispatches through the host bridge", [](TestCase &t)
        {
            FakeIopHost host;
            ps2x::iop::IopSubsystem subsystem(host);
            const std::filesystem::path pluginDirectory(PS2X_TEST_IOP_PLUGIN_DIR);

            t.IsTrue(std::filesystem::is_directory(pluginDirectory),
                     "the synthetic IOP plugin directory should be staged by the test build");
            subsystem.setPluginSearchPaths({pluginDirectory});

            std::string error;
            t.IsTrue(subsystem.loadPlugins(&error), "synthetic IOP plugin discovery should succeed");
            ps2x::iop::DebugSnapshot discoverySnapshot = subsystem.debugSnapshot();
            t.IsTrue(containsDiagnostic(discoverySnapshot, "loaded 4 profile(s)"),
                     "plugin discovery diagnostics should report all accepted synthetic profiles");
            t.IsTrue(containsDiagnostic(discoverySnapshot, "too many SIDs"),
                     "an invalid profile descriptor should be ignored with a diagnostic");
            t.IsTrue(containsDiagnostic(discoverySnapshot, "bad_abi"),
                     "an ABI-incompatible plugin should be ignored with a diagnostic");
            t.IsTrue(containsDiagnostic(discoverySnapshot, "incompatible ABI"),
                     "the incompatible-plugin diagnostic should explain the ABI failure");
            t.IsTrue(containsDiagnostic(discoverySnapshot, "missing_symbol"),
                     "a plugin without the query symbol should be ignored with a diagnostic");
            t.IsTrue(containsDiagnostic(discoverySnapshot, "missing ps2x_iop_query_v1"),
                     "the missing-symbol diagnostic should name the required entry point");

            auto expectNoProfile = [&](const ps2x::iop::GameIdentity &identity, const std::string &reason) {
                error.clear();
                t.IsTrue(subsystem.configure(identity, &error), "mismatching plugin identity should configure core-only services");
                const ps2x::iop::DebugSnapshot snapshot = subsystem.debugSnapshot();
                t.IsTrue(snapshot.activeProfile.empty(), reason);

                ps2x::iop::RpcRequest request{};
                request.sid = kSyntheticSid;
                request.function = kSyntheticFunction;
                t.IsFalse(subsystem.handleRpc(request).handled,
                          "a mismatching profile must not expose its synthetic SID");
            };

            expectNoProfile({"different.elf", kSyntheticEntryPoint, kSyntheticCrc32},
                            "a different ELF basename should not match the plugin profile");
            expectNoProfile({"synthetic_iop_test.elf", kSyntheticEntryPoint + 4u, kSyntheticCrc32},
                            "a different entry point should not match the plugin profile");
            expectNoProfile({"synthetic_iop_test.elf", kSyntheticEntryPoint, kSyntheticCrc32 ^ 1u},
                            "a different CRC32 should not match the plugin profile");

            error.clear();
            t.IsTrue(subsystem.configure({"synthetic_iop_test.elf", kSyntheticEntryPoint, kSyntheticCrc32}, &error),
                     "the synthetic ELF identity should activate the plugin profile");

            ps2x::iop::DebugSnapshot snapshot = subsystem.debugSnapshot();
            t.Equals(snapshot.activeProfile, std::string("synthetic-test-profile"),
                     "debug snapshot should expose the active plugin profile id");
            t.Equals(snapshot.activeProvider, std::string("ps2x-test-plugin"),
                     "debug snapshot should expose the plugin provider name");
            const ps2x::iop::DebugService *service = findService(snapshot, "synthetic-test-profile");
            if (!service)
            {
                t.Fail("debug snapshot should include the synthetic profile service");
                return;
            }
            t.IsTrue(service->profileSpecific, "plugin service should be marked profile-specific");
            t.IsTrue(std::find(service->sids.begin(), service->sids.end(), kSyntheticSid) != service->sids.end(),
                     "plugin service should advertise its synthetic SID");
            t.Equals(metricValue(*service, "reset_generation"), uint64_t{1},
                     "profile configuration should reset a new plugin instance once");

            ps2x::iop::RpcAbiRequest abiRequest{};
            abiRequest.boundSid = kSyntheticSid;
            abiRequest.function = kSyntheticFunction;
            abiRequest.registers.plausible = true;
            abiRequest.stack.plausible = true;
            t.Equals(subsystem.selectRpcAbi(abiRequest), ps2x::iop::RpcAbi::Stack,
                     "plugin should be able to select the stack RPC ABI");
            abiRequest.function = kSyntheticFunction + 1u;
            t.Equals(subsystem.selectRpcAbi(abiRequest), ps2x::iop::RpcAbi::RuntimeDefault,
                     "plugin ABI selection should fall back for unrelated functions");

            constexpr uint32_t kSendAddress = 0x1000u;
            constexpr uint32_t kReceiveAddress = 0x1100u;
            constexpr uint32_t kInput = 0x1234ABCDu;
            t.IsTrue(host.writeWord(kSendAddress, kInput), "fake host should seed the plugin send buffer");
            t.IsTrue(host.writeWord(kReceiveAddress, 0u), "fake host should clear the plugin receive buffer");

            ps2x::iop::RpcRequest request{};
            request.callToken = 0x1122334455667788ull;
            request.sid = kSyntheticSid;
            request.function = kSyntheticFunction;
            request.send = {kSendAddress, sizeof(uint32_t)};
            request.receive = {kReceiveAddress, sizeof(uint32_t)};
            const ps2x::iop::RpcResult result = subsystem.handleRpc(request);

            t.IsTrue(result.handled, "matching synthetic SID/function should dispatch to the plugin");
            t.Equals(result.resultAddress, kReceiveAddress, "plugin should return its receive-buffer address");
            t.IsTrue(result.signalNowaitCompletion, "plugin should request nowait completion signaling");
            t.Equals(result.callbackPolicy, ps2x::iop::CallbackPolicy::Suppress,
                     "plugin should be able to suppress the runtime callback");
            t.Equals(host.readWord(kReceiveAddress), kInput ^ kResponseXor,
                     "plugin should read and write guest memory through the IopHost bridge");

            ps2x::iop::RpcRequest unknownRequest{};
            unknownRequest.sid = 0xDEADC0DEu;
            unknownRequest.function = kSyntheticFunction;
            t.IsFalse(subsystem.handleRpc(unknownRequest).handled,
                      "unknown SID should remain unhandled while a plugin profile is active");

            constexpr uint32_t kCoreCollisionReceiveAddress = 0x1200u;
            ps2x::iop::RpcRequest collisionRequest{};
            collisionRequest.sid = kCoreCollisionSid;
            collisionRequest.function = kCoreCollisionFunction;
            collisionRequest.receive = {kCoreCollisionReceiveAddress, sizeof(uint32_t)};
            const ps2x::iop::RpcResult collisionResult = subsystem.handleRpc(collisionRequest);
            t.IsTrue(collisionResult.handled,
                     "a profile service should take precedence over a core service for the same SID");
            t.Equals(host.readWord(kCoreCollisionReceiveAddress), kCoreCollisionResponse,
                     "the profile collision route should reach the plugin implementation");

            subsystem.onSifTransfer({ps2x::iop::SifTransferKind::SetDma,
                                     ps2x::iop::SifTransferPhase::AfterCopy,
                                     kSendAddress,
                                     kReceiveAddress,
                                     sizeof(uint32_t)});
            snapshot = subsystem.debugSnapshot();
            service = findService(snapshot, "synthetic-test-profile");
            if (!service)
            {
                t.Fail("synthetic profile service should remain visible after dispatch");
                return;
            }
            t.Equals(metricValue(*service, "rpc_calls"), uint64_t{2},
                     "plugin debug metrics should count dispatched RPCs");
            t.Equals(metricValue(*service, "sif_transfers"), uint64_t{1},
                     "plugin debug metrics should count SIF transfer hooks");

            subsystem.reset();
            snapshot = subsystem.debugSnapshot();
            service = findService(snapshot, "synthetic-test-profile");
            if (!service)
            {
                t.Fail("synthetic profile service should remain visible after reset");
                return;
            }
            t.Equals(metricValue(*service, "reset_generation"), uint64_t{2},
                     "explicit subsystem reset should reach the plugin instance");
            t.Equals(metricValue(*service, "rpc_calls"), uint64_t{0},
                     "plugin reset should clear per-instance RPC state");
            t.Equals(metricValue(*service, "sif_transfers"), uint64_t{0},
                     "plugin reset should clear per-instance transfer state");

            error.clear();
            t.IsFalse(subsystem.configure({"synthetic_duplicate.elf", kSyntheticEntryPoint, kSyntheticCrc32}, &error),
                      "duplicate SIDs inside one profile layer should reject configuration");
            t.IsTrue(error.find("duplicate IOP SID") != std::string::npos,
                     "duplicate-SID failure should clearly identify the registry conflict");

            error.clear();
            t.IsFalse(subsystem.configure({"slus_201.84", kSyntheticEntryPoint, kSyntheticCrc32}, &error),
                      "equally specific built-in and plugin matchers should be ambiguous");
            t.IsTrue(error.find("ambiguous IOP profiles") != std::string::npos,
                     "ambiguous profile selection should fail clearly");

            error.clear();
            t.IsTrue(subsystem.configure({"slus_201.84",
                                          kSpecificRecvXEntryPoint,
                                          kSyntheticCrc32},
                                         &error),
                     "a more-specific matcher should win over a lower-specificity tie");
            t.Equals(subsystem.debugSnapshot().activeProfile,
                     std::string("synthetic-specific-recvx-profile"),
                     "the most specific plugin profile should be selected");

            error.clear();
            t.IsTrue(subsystem.configure({"different.elf", kSyntheticEntryPoint, kSyntheticCrc32}, &error),
                     "switching to an unmatched ELF should destroy the active plugin profile");
            t.IsTrue(host.hasLog("fake-plugin-destroy"),
                     "plugin profile destroy callback should run when the active profile is replaced");
            t.IsTrue(subsystem.debugSnapshot().activeProfile.empty(),
                     "switching to an unmatched ELF should leave no active profile");
        });
#endif
    });
}
