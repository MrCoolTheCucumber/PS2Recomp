#ifndef PS2_VU_PROGRAM_CACHE_H
#define PS2_VU_PROGRAM_CACHE_H

#include "runtime/ps2_vu_executable_memory.h"
#include "runtime/ps2_vu_ir.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

enum class VuUnitId : uint8_t
{
    Vu0,
    Vu1,
};

enum class VuNativeBackendMode : uint8_t
{
    X86_64,
    AArch64,
};

enum class VuCompilationMode : uint8_t
{
    Normal,
    Instrumented,
};

struct VuBlockProfileRecord;

struct VuProgramKey
{
    VuUnitId unit = VuUnitId::Vu1;
    uintptr_t memoryIdentity = 0u;
    uintptr_t codeIdentity = 0u;
    uint32_t codeSize = 0u;
    uint32_t addressMask = 0u;
    uint32_t entryPc = 0u;
    uint64_t codeGeneration = 0u;
    uint64_t codeContentIdentity = 0u;
    VuNativeBackendMode backend =
        VuNativeBackendMode::X86_64;
    uint64_t hostFeatures = 0u;
    VuCompilationMode compilationMode =
        VuCompilationMode::Normal;

    bool operator==(const VuProgramKey &) const = default;
};

struct VuCompiledProgram
{
    VuProgramKey key;
    VuIrBlock block;
    VuExecutableMemory nativeCode;
    size_t entryOffset = 0u;
    size_t fastEntryOffset = 0u;
    uint64_t compilationNanoseconds = 0u;
    uint64_t compilationIdentity = 0u;
    mutable uint64_t jitCodeIndex = 0u;
    std::shared_ptr<VuBlockProfileRecord> blockProfile;
    std::shared_ptr<void> blockProfileLifetime;

    [[nodiscard]] const void *nativeEntry() const
    {
        return nativeCode.executableAddress(entryOffset);
    }

    [[nodiscard]] const void *nativeFastEntry() const
    {
        return nativeCode.executableAddress(fastEntryOffset);
    }
};

struct VuProgramHandle
{
    static constexpr uint32_t kInvalidIndex = UINT32_MAX;

    uint64_t epoch = 0u;
    uint32_t index = kInvalidIndex;

    [[nodiscard]] bool valid() const
    {
        return epoch != 0u && index != kInvalidIndex;
    }

    bool operator==(const VuProgramHandle &) const = default;
};

struct VuProgramCacheLimits
{
    size_t maximumPrograms = 3072u;
    size_t maximumExecutableBytes = 96u * 1024u * 1024u;
};

struct VuProgramCacheDiagnostics
{
    uint64_t hits = 0u;
    uint64_t misses = 0u;
    uint64_t compilations = 0u;
    uint64_t invalidations = 0u;
    uint64_t invalidatedPrograms = 0u;
    uint64_t generationRetentions = 0u;
    uint64_t retainedPrograms = 0u;
    uint64_t crossGenerationHits = 0u;
    uint64_t evictionFlushes = 0u;
    uint64_t evictedPrograms = 0u;
    uint64_t manualFlushes = 0u;
    uint64_t rejectedPrograms = 0u;
    uint64_t generatedBytes = 0u;
    uint64_t compilationNanoseconds = 0u;
    size_t residentPrograms = 0u;
    size_t residentExecutableBytes = 0u;
    size_t highWaterPrograms = 0u;
    size_t highWaterExecutableBytes = 0u;
};

// One cache belongs to one VU. All non-destructor calls must come from one
// host thread; the first call binds ownership and later cross-thread access is
// rejected. Handles carry a cache epoch, so a flush can never resolve an old
// native entry. A resolved pointer is valid only until the next non-const cache
// operation on the owning thread.
class VuProgramCache
{
public:
    explicit VuProgramCache(
        VuUnitId unit,
        VuProgramCacheLimits limits = {});

    VuProgramCache(const VuProgramCache &) = delete;
    VuProgramCache &operator=(const VuProgramCache &) = delete;
    VuProgramCache(VuProgramCache &&) = delete;
    VuProgramCache &operator=(VuProgramCache &&) = delete;

    [[nodiscard]] VuProgramHandle lookup(
        const VuProgramKey &key);
    [[nodiscard]] VuProgramHandle insert(
        VuCompiledProgram program,
        std::string *diagnostic = nullptr);
    [[nodiscard]] const VuCompiledProgram *resolveCurrent(
        VuProgramHandle handle,
        const VuProgramKey &expectedKey);
    [[nodiscard]] const VuCompiledProgram *resolve(
        VuProgramHandle handle) const;

    void flush();

    [[nodiscard]] VuUnitId unit() const { return m_unit; }
    [[nodiscard]] VuProgramCacheLimits limits() const
    {
        return m_limits;
    }
    [[nodiscard]] VuProgramCacheDiagnostics diagnostics() const;
    // Read-only debugger snapshot. The caller must externally prove that the
    // owner thread cannot access the cache for the duration of this copy.
    [[nodiscard]] VuProgramCacheDiagnostics
    diagnosticsWhileExecutionQuiescent() const;

    [[nodiscard]] static bool validKey(
        const VuProgramKey &key,
        std::string *diagnostic = nullptr);

private:
    struct KeyHash
    {
        size_t operator()(const VuProgramKey &key) const;
    };

    struct KeyEqual
    {
        bool operator()(
            const VuProgramKey &left,
            const VuProgramKey &right) const;
    };

    struct Scope
    {
        uintptr_t memoryIdentity = 0u;
        uintptr_t codeIdentity = 0u;
        uint32_t codeSize = 0u;
        uint32_t addressMask = 0u;

        bool operator==(const Scope &) const = default;
    };

    enum class FlushReason : uint8_t
    {
        Invalidation,
        Eviction,
        Manual,
    };

    void requireOwnerThread() const;
    bool activateScope(
        const VuProgramKey &key,
        std::string *diagnostic = nullptr);
    void retainProgramsForGeneration();
    void discardPrograms(FlushReason reason);
    [[nodiscard]] static bool sameProgramIdentity(
        const VuProgramKey &left,
        const VuProgramKey &right);
    static uint64_t nextEpoch(uint64_t epoch);
    static bool fail(
        std::string message, std::string *diagnostic);

    VuUnitId m_unit;
    VuProgramCacheLimits m_limits;
    bool m_scopeActive = false;
    Scope m_scope{};
    uint64_t m_codeGeneration = 0u;
    uint64_t m_epoch = 1u;
    std::vector<VuCompiledProgram> m_programs;
    std::unordered_map<
        VuProgramKey, uint32_t, KeyHash, KeyEqual> m_lookup;
    VuProgramCacheDiagnostics m_diagnostics{};
    mutable bool m_ownerBound = false;
    mutable std::thread::id m_ownerThread;
};

#endif
