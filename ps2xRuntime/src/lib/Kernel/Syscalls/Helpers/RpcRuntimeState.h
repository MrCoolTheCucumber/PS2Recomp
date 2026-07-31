#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

struct RpcServerState
{
    uint32_t sid = 0;
    uint32_t sd_ptr = 0; // PS2 address
};

struct RpcClientState
{
    bool busy = false;
    uint32_t last_rpc = 0;
    uint32_t sid = 0;
};

struct SifRpcDebugEvent
{
    uint64_t seq = 0;
    const char *op = "";
    uint32_t pc = 0;
    uint32_t ra = 0;
    uint32_t threadId = 0;
    uint32_t sid = 0;
    uint32_t rpcNum = 0;
    uint32_t clientPtr = 0;
    uint32_t serverPtr = 0;
    uint32_t sendBuf = 0;
    uint32_t sendSize = 0;
    uint32_t recvBuf = 0;
    uint32_t recvSize = 0;
    uint32_t resultPtr = 0;
    uint32_t mode = 0;
    uint32_t endFunc = 0;
    uint32_t endParam = 0;
    uint32_t semaId = 0;
    uint32_t flags = 0;
    uint32_t sendPreviewSize = 0;
    uint32_t recvPreviewSize = 0;
    uint8_t sendPreview[64]{};
    uint8_t recvPreview[64]{};
    int32_t result = 0;
};

inline constexpr size_t kSifRpcDebugHistoryCount = 256u;
inline constexpr size_t kSifRpcDebugPreviewBytes = 64u;
inline constexpr uint32_t kSifRpcDebugFlagNowait = 1u << 0;
inline constexpr uint32_t kSifRpcDebugFlagHandledByHle = 1u << 1;
inline constexpr uint32_t kSifRpcDebugFlagCallback = 1u << 2;
inline constexpr uint32_t kSifRpcDebugFlagMissingClient = 1u << 3;
inline constexpr uint32_t kSifRpcDebugFlagServerDispatch = 1u << 4;
inline constexpr uint32_t kSifRpcDebugFlagUnhandled = 1u << 6;
inline constexpr uint32_t kSifRpcDebugFlagFallbackCopy = 1u << 7;
inline constexpr uint32_t kSifRpcDebugFlagFallbackZero = 1u << 8;

struct SifModuleRecord
{
    int32_t id = 0;
    std::string path;
    std::string pathKey;
    uint32_t refCount = 0;
    bool loaded = false;
};

// SIF RPC registries, guest-visible allocation cursors, diagnostic history,
// and IOP module bookkeeping belong to one emulated machine. Keeping the
// call mutex in the same state also permits independent runtimes to execute
// RPC callbacks concurrently while retaining recursive calls within one
// runtime.
struct EeRpcRuntimeState
{
    std::mutex rpcMutex;
    std::recursive_mutex callMutex;
    bool initialized = false;
    std::unordered_map<uint32_t, RpcServerState> servers;
    std::unordered_map<uint32_t, RpcClientState> clients;
    std::array<SifRpcDebugEvent, kSifRpcDebugHistoryCount>
        debugHistory{};
    uint64_t nextDebugSequence = 0;
    uint32_t nextRequestId = 1;
    uint32_t packetIndex = 0;
    uint32_t serverIndex = 0;
    uint32_t activeQueue = 0;

    std::mutex moduleMutex;
    std::unordered_map<int32_t, SifModuleRecord> modulesById;
    std::unordered_map<std::string, int32_t> moduleIdByPath;
    int32_t nextModuleId = 1;
    uint32_t moduleLogCount = 0;

    std::mutex traceMutex;
    std::unordered_set<uint64_t> loggedTraceSignatures;
};
