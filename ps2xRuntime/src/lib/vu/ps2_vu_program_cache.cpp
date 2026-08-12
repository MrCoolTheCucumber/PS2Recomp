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

uint64_t VuNativeLinkState::invalidateTargets() noexcept
{
    uint64_t invalidated = 0u;
    const auto invalidate =
        [&](VuNativeLinkSlot &slot, bool dynamic)
        {
            if (slot.targetEntry != 0u)
                ++invalidated;
            // Publish the unusable entry before changing its metadata. Cache
            // mutation and execution share one owner thread, but this order
            // also makes quiescent diagnostics unambiguous.
            slot.targetEntry = 0u;
            slot.codeGeneration = 0u;
            if (dynamic)
                slot.targetPc = UINT32_MAX;
        };
    for (VuNativeLinkSlot &slot : staticTargets)
        invalidate(slot, false);
    for (VuNativeLinkSlot &slot : dynamicTargets)
        invalidate(slot, true);
    nextDynamicSlot = 0u;
    return invalidated;
}

uint64_t VuNativeLinkState::invalidateTarget(
    uintptr_t targetEntry) noexcept
{
    if (targetEntry == 0u)
        return 0u;

    uint64_t invalidated = 0u;
    const auto invalidate =
        [&](VuNativeLinkSlot &slot, bool dynamic)
        {
            if (slot.targetEntry != targetEntry)
                return;
            slot.targetEntry = 0u;
            slot.codeGeneration = 0u;
            if (dynamic)
                slot.targetPc = UINT32_MAX;
            ++invalidated;
        };
    for (VuNativeLinkSlot &slot : staticTargets)
        invalidate(slot, false);
    for (VuNativeLinkSlot &slot : dynamicTargets)
        invalidate(slot, true);
    return invalidated;
}

VuProgramCache::VuProgramCache(
    VuUnitId unit, VuProgramCacheLimits limits)
    : m_unit(unit), m_limits(limits)
{
    m_diagnostics.maximumPrograms = limits.maximumPrograms;
    m_diagnostics.maximumExecutableBytes =
        limits.maximumExecutableBytes;
}

VuProgramCache::~VuProgramCache()
{
    invalidateAllLinks();
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
    ProgramSlot &slot = m_programs[found->second];
    if (!slot.program)
    {
        throw std::logic_error(
            "VU program cache lookup references an empty slot");
    }
    if (slot.program->key.codeGeneration != key.codeGeneration)
    {
        ++m_diagnostics.crossGenerationHits;
    }
    touchProgram(found->second);
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
        !program.nativeEntry() ||
        !program.nativeFastEntry() ||
        (program.outgoingLinks &&
         !program.nativeLinkedEntry()))
    {
        ++m_diagnostics.rejectedPrograms;
        fail(
            "compiled program has no executable entry points",
            diagnostic);
        return {};
    }

    const auto existing = m_lookup.find(program.key);
    if (existing != m_lookup.end())
    {
        touchProgram(existing->second);
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

    while (m_diagnostics.residentPrograms >=
               m_limits.maximumPrograms ||
           residentBytes >
               m_limits.maximumExecutableBytes -
                   m_diagnostics.residentExecutableBytes)
    {
        if (!evictLeastRecentlyUsed())
        {
            ++m_diagnostics.rejectedPrograms;
            fail(
                "VU program cache could not reclaim a resident program",
                diagnostic);
            return {};
        }
    }

    if (m_freeProgramIndices.empty() &&
        m_programs.size() >=
            static_cast<size_t>(
                std::numeric_limits<uint32_t>::max()))
    {
        ++m_diagnostics.rejectedPrograms;
        fail(
            "VU program cache handle space is exhausted",
            diagnostic);
        return {};
    }

    uint32_t index = 0u;
    if (m_freeProgramIndices.empty())
    {
        index = static_cast<uint32_t>(m_programs.size());
        m_programs.emplace_back();
    }
    else
    {
        index = m_freeProgramIndices.back();
        m_freeProgramIndices.pop_back();
    }
    const size_t generatedBytes =
        program.nativeCode.usedSize();
    const uint64_t compilationNanoseconds =
        program.compilationNanoseconds;
    if (program.outgoingLinks)
    {
        program.outgoingLinks->ownerCacheIdentity =
            reinterpret_cast<uintptr_t>(this);
        program.outgoingLinks->ownerProgramIndex = index;
    }
    ProgramSlot &slot = m_programs[index];
    slot.program = std::make_unique<VuCompiledProgram>(
        std::move(program));
    m_lookup.emplace(slot.program->key, index);
    touchProgram(index);

    ++m_diagnostics.compilations;
    m_diagnostics.generatedBytes += generatedBytes;
    m_diagnostics.compilationNanoseconds +=
        compilationNanoseconds;
    ++m_diagnostics.residentPrograms;
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
        handle.index >= m_programs.size() ||
        !m_programs[handle.index].program)
    {
        return nullptr;
    }
    return m_programs[handle.index].program.get();
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
        !m_programs[handle.index].program ||
        !sameProgramIdentity(
            m_programs[handle.index].program->key,
            expectedKey))
    {
        return nullptr;
    }
    ++m_diagnostics.hits;
    touchProgram(handle.index);
    return m_programs[handle.index].program.get();
}

std::shared_ptr<VuNativeLinkState>
VuProgramCache::pinLinkSource(
    const VuNativeLinkState *source)
{
    requireOwnerThread();
    if (!source)
        return {};

    const uint32_t sourceIndex = source->ownerProgramIndex;
    if (source->ownerCacheIdentity !=
            reinterpret_cast<uintptr_t>(this) ||
        sourceIndex >= m_programs.size() ||
        !m_programs[sourceIndex].program ||
        m_programs[sourceIndex].program->
                outgoingLinks.get() != source)
    {
        return {};
    }
    touchProgram(sourceIndex);
    return m_programs[sourceIndex].program->outgoingLinks;
}

VuProgramLinkResult VuProgramCache::populateLink(
    const VuNativeLinkState *source,
    VuProgramHandle targetHandle,
    const VuProgramKey &currentTargetKey)
{
    requireOwnerThread();
    const auto reject =
        [&](VuProgramLinkResult result)
        {
            ++m_diagnostics.linkResolutionFailures;
            return result;
        };
    if (!source)
        return reject(VuProgramLinkResult::SourceUnavailable);

    const uint32_t sourceIndex =
        source->ownerProgramIndex;
    if (source->ownerCacheIdentity !=
            reinterpret_cast<uintptr_t>(this) ||
        sourceIndex >= m_programs.size() ||
        !m_programs[sourceIndex].program ||
        m_programs[sourceIndex].program->
                outgoingLinks.get() != source)
    {
        return reject(VuProgramLinkResult::SourceUnavailable);
    }
    VuCompiledProgram &sourceProgram =
        *m_programs[sourceIndex].program;

    if (!m_scopeActive ||
        m_codeGeneration !=
            currentTargetKey.codeGeneration ||
        !targetHandle.valid() ||
        targetHandle.epoch != m_epoch ||
        targetHandle.index >= m_programs.size() ||
        !m_programs[targetHandle.index].program)
    {
        return reject(VuProgramLinkResult::TargetUnavailable);
    }

    const VuCompiledProgram &target =
        *m_programs[targetHandle.index].program;
    if (!sameProgramIdentity(
            target.key, currentTargetKey) ||
        !target.outgoingLinks ||
        !target.nativeLinkedEntry())
    {
        return reject(VuProgramLinkResult::TargetUnavailable);
    }

    VuNativeLinkState &sourceLinks =
        *sourceProgram.outgoingLinks;
    if (!sameLinkIdentity(
            sourceProgram.key, target.key) ||
        sourceLinks.backendIdentity == 0u ||
        sourceLinks.backendIdentity !=
            target.outgoingLinks->backendIdentity)
    {
        return reject(VuProgramLinkResult::Incompatible);
    }

    VuNativeLinkSlot *slot = nullptr;
    for (VuNativeLinkSlot &candidate :
         sourceLinks.staticTargets)
    {
        if (candidate.targetPc ==
            currentTargetKey.entryPc)
        {
            slot = &candidate;
            break;
        }
    }
    if (!slot)
    {
        for (VuNativeLinkSlot &candidate :
             sourceLinks.dynamicTargets)
        {
            if (candidate.targetPc ==
                currentTargetKey.entryPc)
            {
                slot = &candidate;
                break;
            }
        }
    }
    if (!slot)
    {
        for (VuNativeLinkSlot &candidate :
             sourceLinks.dynamicTargets)
        {
            if (candidate.targetEntry == 0u)
            {
                slot = &candidate;
                break;
            }
        }
    }
    if (!slot)
    {
        const size_t index =
            sourceLinks.nextDynamicSlot %
            sourceLinks.dynamicTargets.size();
        slot = &sourceLinks.dynamicTargets[index];
        sourceLinks.nextDynamicSlot =
            static_cast<uint32_t>(
                (index + 1u) %
                sourceLinks.dynamicTargets.size());
    }

    slot->targetEntry = 0u;
    slot->codeGeneration =
        currentTargetKey.codeGeneration;
    slot->targetPc = currentTargetKey.entryPc;
    slot->targetEntry =
        reinterpret_cast<uintptr_t>(
            target.nativeLinkedEntry());
    touchProgram(sourceIndex);
    touchProgram(targetHandle.index);
    ++m_diagnostics.linkResolutions;
    return VuProgramLinkResult::Linked;
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
    if (key.blockLocalVfRegistersAutomatic &&
        !key.blockLocalVfRegisters)
    {
        return fail(
            "automatic VU register allocation requires register-resident code",
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
    hashCombine(
        value, std::hash<uint8_t>{}(
                   static_cast<uint8_t>(
                       key.blockForm)));
    hashCombine(
        value, std::hash<bool>{}(
                   key.blockLocalVfRegisters));
    hashCombine(
        value, std::hash<bool>{}(
                   key.blockLocalVfRegistersAutomatic));
    hashCombine(
        value, std::hash<bool>{}(
                   key.inlineXgkick));
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
        left.compilationMode != right.compilationMode ||
        left.blockForm != right.blockForm ||
        left.blockLocalVfRegisters !=
            right.blockLocalVfRegisters ||
        left.blockLocalVfRegistersAutomatic !=
            right.blockLocalVfRegistersAutomatic ||
        left.inlineXgkick !=
            right.inlineXgkick)
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

bool VuProgramCache::sameLinkIdentity(
    const VuProgramKey &left,
    const VuProgramKey &right)
{
    VuProgramKey leftEntry = left;
    VuProgramKey rightEntry = right;
    leftEntry.entryPc = 0u;
    rightEntry.entryPc = 0u;
    return sameProgramIdentity(leftEntry, rightEntry);
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
        static_cast<uint64_t>(
            m_diagnostics.residentPrograms);
    m_epoch = nextEpoch(m_epoch);
}

void VuProgramCache::discardPrograms(FlushReason reason)
{
    invalidateAllLinks();
    const uint64_t programCount =
        static_cast<uint64_t>(
            m_diagnostics.residentPrograms);
    switch (reason)
    {
    case FlushReason::Invalidation:
        ++m_diagnostics.invalidations;
        m_diagnostics.invalidatedPrograms += programCount;
        break;
    case FlushReason::Manual:
        ++m_diagnostics.manualFlushes;
        break;
    }

    m_lookup.clear();
    m_programs.clear();
    m_freeProgramIndices.clear();
    m_diagnostics.residentPrograms = 0u;
    m_diagnostics.residentExecutableBytes = 0u;
    m_epoch = nextEpoch(m_epoch);
}

void VuProgramCache::invalidateAllLinks()
{
    for (ProgramSlot &slot : m_programs)
    {
        if (slot.program && slot.program->outgoingLinks)
        {
            m_diagnostics.linkInvalidations +=
                slot.program->outgoingLinks->
                    invalidateTargets();
            slot.program->outgoingLinks->
                ownerCacheIdentity = 0u;
            slot.program->outgoingLinks->
                ownerProgramIndex =
                    VuNativeLinkState::
                        kUnboundProgramIndex;
        }
    }
}

void VuProgramCache::touchProgram(uint32_t index)
{
    m_accessClock = nextEpoch(m_accessClock);
    m_programs[index].lastUse = m_accessClock;
}

bool VuProgramCache::evictLeastRecentlyUsed()
{
    uint32_t victimIndex = VuProgramHandle::kInvalidIndex;
    uint64_t oldestUse = std::numeric_limits<uint64_t>::max();
    for (size_t index = 0u; index < m_programs.size(); ++index)
    {
        const ProgramSlot &slot = m_programs[index];
        if (!slot.program ||
            (slot.lastUse > oldestUse) ||
            (slot.lastUse == oldestUse &&
             victimIndex != VuProgramHandle::kInvalidIndex &&
             index >= victimIndex))
        {
            continue;
        }
        victimIndex = static_cast<uint32_t>(index);
        oldestUse = slot.lastUse;
    }
    if (victimIndex == VuProgramHandle::kInvalidIndex)
        return false;

    ProgramSlot &victimSlot = m_programs[victimIndex];
    VuCompiledProgram &victim = *victimSlot.program;
    const size_t residentBytes = victim.nativeCode.capacity();
    const uintptr_t targetEntry = victim.outgoingLinks
        ? reinterpret_cast<uintptr_t>(victim.nativeLinkedEntry())
        : 0u;

    // Clear every incoming reference before releasing the executable image.
    // The victim's outgoing links are then cleared and unbound as a separate
    // ownership operation; self-links already cleared above count only once.
    if (targetEntry != 0u)
    {
        for (ProgramSlot &slot : m_programs)
        {
            if (slot.program && slot.program->outgoingLinks)
            {
                m_diagnostics.linkInvalidations +=
                    slot.program->outgoingLinks->
                        invalidateTarget(targetEntry);
            }
        }
    }
    if (victim.outgoingLinks)
    {
        m_diagnostics.linkInvalidations +=
            victim.outgoingLinks->invalidateTargets();
        victim.outgoingLinks->ownerCacheIdentity = 0u;
        victim.outgoingLinks->ownerProgramIndex =
            VuNativeLinkState::kUnboundProgramIndex;
    }

    m_lookup.erase(victim.key);
    victimSlot.program.reset();
    victimSlot.lastUse = 0u;
    m_freeProgramIndices.push_back(victimIndex);
    --m_diagnostics.residentPrograms;
    m_diagnostics.residentExecutableBytes -= residentBytes;
    ++m_diagnostics.selectiveEvictions;
    ++m_diagnostics.evictedPrograms;
    m_epoch = nextEpoch(m_epoch);
    return true;
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
