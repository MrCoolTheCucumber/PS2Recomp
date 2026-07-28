#include "runtime/ps2_vu_program_cache.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace
{
    void hashCombine(size_t &seed, size_t value)
    {
        seed ^= value +
                static_cast<size_t>(0x9e3779b97f4a7c15ull) +
                (seed << 6u) + (seed >> 2u);
    }
}

VuProgramCache::VuProgramCache(
    VuUnitId unit, VuProgramCacheLimits limits)
    : m_unit(unit), m_limits(limits)
{
}

VuProgramHandle VuProgramCache::lookup(
    const VuProgramKey &key)
{
    requireOwnerThread();
    if (!activateScope(key))
    {
        ++m_diagnostics.misses;
        return {};
    }

    const auto found = m_lookup.find(key);
    if (found == m_lookup.end())
    {
        ++m_diagnostics.misses;
        return {};
    }

    ++m_diagnostics.hits;
    if (m_programs[found->second].key.codeGeneration !=
        key.codeGeneration)
    {
        ++m_diagnostics.crossGenerationHits;
    }
    return {
        .epoch = m_epoch,
        .index = found->second,
    };
}

VuProgramHandle VuProgramCache::insert(
    VuCompiledProgram program,
    std::string *diagnostic)
{
    requireOwnerThread();
    if (!activateScope(program.key, diagnostic))
    {
        ++m_diagnostics.rejectedPrograms;
        return {};
    }

    VuIrVerificationError verificationError;
    if (!verifyVuIrBlock(program.block, &verificationError))
    {
        ++m_diagnostics.rejectedPrograms;
        fail(
            "refusing malformed VU IR block at PC " +
                std::to_string(verificationError.pc) +
                ": " + verificationError.message,
            diagnostic);
        return {};
    }
    if (program.block.entryPc != program.key.entryPc)
    {
        ++m_diagnostics.rejectedPrograms;
        fail(
            "compiled block entry does not match its cache key",
            diagnostic);
        return {};
    }
    if (program.nativeCode.state() !=
            VuExecutableMemoryState::Executable ||
        !program.nativeEntry())
    {
        ++m_diagnostics.rejectedPrograms;
        fail(
            "compiled program has no executable entry",
            diagnostic);
        return {};
    }

    const auto existing = m_lookup.find(program.key);
    if (existing != m_lookup.end())
    {
        if (diagnostic)
            diagnostic->clear();
        return {
            .epoch = m_epoch,
            .index = existing->second,
        };
    }

    const size_t residentBytes =
        program.nativeCode.capacity();
    if (m_limits.maximumPrograms == 0u ||
        m_limits.maximumExecutableBytes == 0u ||
        residentBytes > m_limits.maximumExecutableBytes)
    {
        ++m_diagnostics.rejectedPrograms;
        fail(
            "compiled program exceeds the cache limits",
            diagnostic);
        return {};
    }

    if (m_programs.size() >= m_limits.maximumPrograms ||
        residentBytes >
            m_limits.maximumExecutableBytes -
                m_diagnostics.residentExecutableBytes)
    {
        discardPrograms(FlushReason::Eviction);
    }

    if (m_programs.size() >=
        static_cast<size_t>(
            std::numeric_limits<uint32_t>::max()))
    {
        ++m_diagnostics.rejectedPrograms;
        fail(
            "VU program cache handle space is exhausted",
            diagnostic);
        return {};
    }

    const uint32_t index =
        static_cast<uint32_t>(m_programs.size());
    const size_t generatedBytes =
        program.nativeCode.usedSize();
    const uint64_t compilationNanoseconds =
        program.compilationNanoseconds;
    m_programs.push_back(std::move(program));
    m_lookup.emplace(m_programs.back().key, index);

    ++m_diagnostics.compilations;
    m_diagnostics.generatedBytes += generatedBytes;
    m_diagnostics.compilationNanoseconds +=
        compilationNanoseconds;
    m_diagnostics.residentPrograms = m_programs.size();
    m_diagnostics.residentExecutableBytes += residentBytes;
    m_diagnostics.highWaterPrograms = std::max(
        m_diagnostics.highWaterPrograms,
        m_diagnostics.residentPrograms);
    m_diagnostics.highWaterExecutableBytes = std::max(
        m_diagnostics.highWaterExecutableBytes,
        m_diagnostics.residentExecutableBytes);
    if (diagnostic)
        diagnostic->clear();
    return {
        .epoch = m_epoch,
        .index = index,
    };
}

const VuCompiledProgram *VuProgramCache::resolve(
    VuProgramHandle handle) const
{
    requireOwnerThread();
    if (!handle.valid() || handle.epoch != m_epoch ||
        handle.index >= m_programs.size())
    {
        return nullptr;
    }
    return &m_programs[handle.index];
}

const VuCompiledProgram *VuProgramCache::resolveCurrent(
    VuProgramHandle handle,
    const VuProgramKey &expectedKey)
{
    requireOwnerThread();
    const Scope expectedScope{
        .memoryIdentity = expectedKey.memoryIdentity,
        .codeIdentity = expectedKey.codeIdentity,
        .codeSize = expectedKey.codeSize,
        .addressMask = expectedKey.addressMask,
    };
    if (!m_scopeActive ||
        m_scope != expectedScope ||
        m_codeGeneration != expectedKey.codeGeneration ||
        !handle.valid() || handle.epoch != m_epoch ||
        handle.index >= m_programs.size() ||
        !sameProgramIdentity(
            m_programs[handle.index].key,
            expectedKey))
    {
        return nullptr;
    }
    ++m_diagnostics.hits;
    return &m_programs[handle.index];
}

void VuProgramCache::flush()
{
    requireOwnerThread();
    discardPrograms(FlushReason::Manual);
    m_scopeActive = false;
    m_scope = {};
    m_codeGeneration = 0u;
}

VuProgramCacheDiagnostics VuProgramCache::diagnostics() const
{
    requireOwnerThread();
    return m_diagnostics;
}

VuProgramCacheDiagnostics
VuProgramCache::diagnosticsWhileExecutionQuiescent() const
{
    return m_diagnostics;
}

bool VuProgramCache::validKey(
    const VuProgramKey &key,
    std::string *diagnostic)
{
    if (key.memoryIdentity == 0u || key.codeIdentity == 0u)
    {
        return fail(
            "VU cache identity pointers must be nonzero",
            diagnostic);
    }
    if (key.codeSize < 8u ||
        (key.codeSize & 7u) != 0u ||
        key.addressMask != key.codeSize - 1u ||
        (key.codeSize & key.addressMask) != 0u)
    {
        return fail(
            "VU cache code extent must be a pair-aligned power of two",
            diagnostic);
    }
    if ((key.entryPc & 7u) != 0u ||
        key.entryPc > key.codeSize - 8u)
    {
        return fail(
            "VU cache entry PC is outside its code extent",
            diagnostic);
    }
    if (diagnostic)
        diagnostic->clear();
    return true;
}

size_t VuProgramCache::KeyHash::operator()(
    const VuProgramKey &key) const
{
    size_t value = 0u;
    hashCombine(
        value, std::hash<uint8_t>{}(
                   static_cast<uint8_t>(key.unit)));
    hashCombine(
        value, std::hash<uintptr_t>{}(key.memoryIdentity));
    hashCombine(
        value, std::hash<uintptr_t>{}(key.codeIdentity));
    hashCombine(value, std::hash<uint32_t>{}(key.codeSize));
    hashCombine(value, std::hash<uint32_t>{}(key.addressMask));
    hashCombine(value, std::hash<uint32_t>{}(key.entryPc));
    if (key.codeContentIdentity != 0u)
    {
        hashCombine(value, 1u);
        hashCombine(
            value,
            std::hash<uint64_t>{}(
                key.codeContentIdentity));
    }
    else
    {
        hashCombine(value, 0u);
        hashCombine(
            value,
            std::hash<uint64_t>{}(
                key.codeGeneration));
    }
    hashCombine(
        value, std::hash<uint8_t>{}(
                   static_cast<uint8_t>(key.backend)));
    hashCombine(
        value, std::hash<uint64_t>{}(key.hostFeatures));
    hashCombine(
        value, std::hash<uint8_t>{}(
                   static_cast<uint8_t>(
                       key.compilationMode)));
    return value;
}

bool VuProgramCache::KeyEqual::operator()(
    const VuProgramKey &left,
    const VuProgramKey &right) const
{
    return sameProgramIdentity(left, right);
}

bool VuProgramCache::sameProgramIdentity(
    const VuProgramKey &left,
    const VuProgramKey &right)
{
    if (left.unit != right.unit ||
        left.memoryIdentity != right.memoryIdentity ||
        left.codeIdentity != right.codeIdentity ||
        left.codeSize != right.codeSize ||
        left.addressMask != right.addressMask ||
        left.entryPc != right.entryPc ||
        left.backend != right.backend ||
        left.hostFeatures != right.hostFeatures ||
        left.compilationMode != right.compilationMode)
    {
        return false;
    }
    if (left.codeContentIdentity != 0u ||
        right.codeContentIdentity != 0u)
    {
        return left.codeContentIdentity != 0u &&
               left.codeContentIdentity ==
                   right.codeContentIdentity;
    }
    return left.codeGeneration == right.codeGeneration;
}

void VuProgramCache::requireOwnerThread() const
{
    const std::thread::id current = std::this_thread::get_id();
    if (!m_ownerBound)
    {
        m_ownerThread = current;
        m_ownerBound = true;
        return;
    }
    if (m_ownerThread != current)
    {
        throw std::logic_error(
            "VU program cache accessed from a non-owner thread");
    }
}

bool VuProgramCache::activateScope(
    const VuProgramKey &key, std::string *diagnostic)
{
    if (!validKey(key, diagnostic))
        return false;
    if (key.unit != m_unit)
    {
        return fail(
            "VU cache key belongs to a different unit",
            diagnostic);
    }

    const Scope scope{
        .memoryIdentity = key.memoryIdentity,
        .codeIdentity = key.codeIdentity,
        .codeSize = key.codeSize,
        .addressMask = key.addressMask,
    };
    if (!m_scopeActive)
    {
        m_scopeActive = true;
        m_scope = scope;
        m_codeGeneration = key.codeGeneration;
        if (diagnostic)
            diagnostic->clear();
        return true;
    }
    if (m_scope != scope)
    {
        discardPrograms(FlushReason::Invalidation);
        m_scope = scope;
        m_codeGeneration = key.codeGeneration;
    }
    else if (m_codeGeneration != key.codeGeneration)
    {
        if (key.codeContentIdentity != 0u)
            retainProgramsForGeneration();
        else
            discardPrograms(FlushReason::Invalidation);
        m_codeGeneration = key.codeGeneration;
    }
    if (diagnostic)
        diagnostic->clear();
    return true;
}

void VuProgramCache::retainProgramsForGeneration()
{
    ++m_diagnostics.invalidations;
    ++m_diagnostics.generationRetentions;
    m_diagnostics.retainedPrograms +=
        static_cast<uint64_t>(m_programs.size());
    m_epoch = nextEpoch(m_epoch);
}

void VuProgramCache::discardPrograms(FlushReason reason)
{
    const uint64_t programCount =
        static_cast<uint64_t>(m_programs.size());
    switch (reason)
    {
    case FlushReason::Invalidation:
        ++m_diagnostics.invalidations;
        m_diagnostics.invalidatedPrograms += programCount;
        break;
    case FlushReason::Eviction:
        ++m_diagnostics.evictionFlushes;
        m_diagnostics.evictedPrograms += programCount;
        break;
    case FlushReason::Manual:
        ++m_diagnostics.manualFlushes;
        break;
    }

    m_lookup.clear();
    m_programs.clear();
    m_diagnostics.residentPrograms = 0u;
    m_diagnostics.residentExecutableBytes = 0u;
    m_epoch = nextEpoch(m_epoch);
}

uint64_t VuProgramCache::nextEpoch(uint64_t epoch)
{
    ++epoch;
    if (epoch == 0u)
        ++epoch;
    return epoch;
}

bool VuProgramCache::fail(
    std::string message, std::string *diagnostic)
{
    if (diagnostic)
        *diagnostic = std::move(message);
    return false;
}
