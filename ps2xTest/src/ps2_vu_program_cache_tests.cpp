#include "MiniTest.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu_program_cache.h"
#include "runtime/ps2_vu1.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{
    constexpr uint32_t kVuUpperNop = 0x000002ffu;

    VuProgramKey syntheticKey(
        VuUnitId unit, const void *memoryIdentity,
        const void *codeIdentity)
    {
        return {
            .unit = unit,
            .memoryIdentity =
                reinterpret_cast<uintptr_t>(memoryIdentity),
            .codeIdentity =
                reinterpret_cast<uintptr_t>(codeIdentity),
            .codeSize = 16u,
            .addressMask = 15u,
            .entryPc = 0u,
            .codeGeneration = 1u,
            .backend = VuNativeBackendMode::X86_64,
            .hostFeatures = 0x1234u,
            .compilationMode = VuCompilationMode::Normal,
        };
    }

    VuProgramKey memoryKey(
        PS2Memory &memory, VuUnitId unit, uint32_t entryPc = 0u)
    {
        const bool vu0 = unit == VuUnitId::Vu0;
        const uint8_t *const code =
            vu0 ? memory.getVU0Code() : memory.getVU1Code();
        const uint32_t codeSize =
            vu0 ? PS2_VU0_CODE_SIZE : PS2_VU1_CODE_SIZE;
        return {
            .unit = unit,
            .memoryIdentity =
                reinterpret_cast<uintptr_t>(&memory),
            .codeIdentity = reinterpret_cast<uintptr_t>(code),
            .codeSize = codeSize,
            .addressMask = codeSize - 1u,
            .entryPc = entryPc,
            .codeGeneration =
                vu0
                    ? memory.getVU0CodeGeneration()
                    : memory.getVU1CodeGeneration(),
            .backend = VuNativeBackendMode::X86_64,
            .hostFeatures = 0u,
            .compilationMode = VuCompilationMode::Normal,
        };
    }

    VuExecutableMemory makeNativeCode(
        size_t minimumBytes = 8u)
    {
        std::string diagnostic;
        VuExecutableMemory code =
            VuExecutableMemory::allocate(
                minimumBytes, &diagnostic);
        if (code.empty())
        {
            throw std::runtime_error(
                "native allocation failed: " + diagnostic);
        }

#if defined(__aarch64__) || defined(_M_ARM64)
        const std::array<uint8_t, 4u> bytes{
            0xc0u, 0x03u, 0x5fu, 0xd6u};
#else
        const std::array<uint8_t, 1u> bytes{0xc3u};
#endif
        if (!code.write(
                0u, bytes.data(), bytes.size(), &diagnostic) ||
            !code.finalize(bytes.size(), &diagnostic))
        {
            throw std::runtime_error(
                "native finalization failed: " + diagnostic);
        }
        return code;
    }

    VuCompiledProgram makeProgram(
        const VuProgramKey &key,
        size_t minimumNativeBytes = 8u,
        uint64_t compilationNanoseconds = 37u)
    {
        std::vector<uint8_t> code(key.codeSize, 0u);
        std::memcpy(
            code.data() + key.entryPc + 4u,
            &kVuUpperNop, sizeof(kVuUpperNop));
        VuIrBlock block = decodeVuIrBlock(
            code.data(), key.codeSize, key.entryPc, 1u);
        return {
            .key = key,
            .block = std::move(block),
            .nativeCode = makeNativeCode(minimumNativeBytes),
            .entryOffset = 0u,
            .compilationNanoseconds = compilationNanoseconds,
        };
    }

    std::shared_ptr<VuNativeLinkState> makeLinks(
        const VuProgramKey &key,
        uintptr_t backendIdentity,
        std::initializer_list<uint32_t> staticTargets = {})
    {
        auto links =
            std::make_shared<VuNativeLinkState>();
        links->backendIdentity = backendIdentity;
        links->sourceEntryPc = key.entryPc;
        for (const uint32_t target : staticTargets)
        {
            links->staticTargets.push_back({
                .targetPc = target,
            });
        }
        return links;
    }

    VuCompiledProgram makeLinkedProgram(
        const VuProgramKey &key,
        uintptr_t backendIdentity,
        std::initializer_list<uint32_t> staticTargets = {})
    {
        VuCompiledProgram program = makeProgram(key);
        program.outgoingLinks =
            makeLinks(
                key, backendIdentity,
                staticTargets);
        return program;
    }

    void appendU32(
        std::vector<uint8_t> &bytes, uint32_t value)
    {
        const size_t offset = bytes.size();
        bytes.resize(offset + sizeof(value));
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    uint32_t makeVifCommand(
        uint8_t opcode, uint8_t count, uint16_t immediate)
    {
        return
            (static_cast<uint32_t>(opcode) << 24u) |
            (static_cast<uint32_t>(count) << 16u) |
            immediate;
    }

    std::vector<uint8_t> makeMpgPacket(
        uint16_t destination, const uint8_t *payload,
        uint8_t instructionCount)
    {
        const size_t payloadBytes =
            static_cast<size_t>(instructionCount) * 8u;
        std::vector<uint8_t> packet;
        appendU32(
            packet,
            makeVifCommand(
                0x4au, instructionCount, destination));
        packet.insert(
            packet.end(), payload, payload + payloadBytes);
        return packet;
    }
}

void register_ps2_vu_program_cache_tests()
{
    MiniTest::Case("PS2VUProgramCache", [](TestCase &tc)
    {
        tc.Run(
            "keys distinguish entry backend features and compilation mode",
            [](TestCase &t)
            {
                int memoryIdentity = 0;
                int codeIdentity = 0;
                const VuProgramKey firstKey =
                    syntheticKey(
                        VuUnitId::Vu1,
                        &memoryIdentity, &codeIdentity);
                VuProgramCache cache(VuUnitId::Vu1);

                t.IsFalse(
                    cache.lookup(firstKey).valid(),
                    "a cold cache lookup should miss");
                std::string diagnostic;
                const VuProgramHandle first =
                    cache.insert(
                        makeProgram(firstKey), &diagnostic);
                t.IsTrue(
                    first.valid(),
                    "a verified executable block should enter the cache");
                t.IsNotNull(
                    cache.resolve(first),
                    "a current handle should resolve its program");
                t.Equals(
                    cache.lookup(firstKey), first,
                    "an identical key should reuse compiled code");

                const VuProgramHandle duplicate =
                    cache.insert(
                        makeProgram(firstKey), &diagnostic);
                t.Equals(
                    duplicate, first,
                    "duplicate insertion should reuse the resident program");

                VuProgramKey entryKey = firstKey;
                entryKey.entryPc = 8u;
                VuProgramKey backendKey = firstKey;
                backendKey.backend =
                    VuNativeBackendMode::AArch64;
                VuProgramKey featuresKey = firstKey;
                featuresKey.hostFeatures ^= 0x80u;
                VuProgramKey modeKey = firstKey;
                modeKey.compilationMode =
                    VuCompilationMode::Instrumented;
                VuProgramKey formKey = firstKey;
                formKey.blockForm =
                    VuIrBlockForm::Basic;

                t.IsFalse(
                    cache.lookup(entryKey).valid(),
                    "a distinct entry PC should miss");
                t.IsFalse(
                    cache.lookup(backendKey).valid(),
                    "a distinct native backend should miss");
                t.IsFalse(
                    cache.lookup(featuresKey).valid(),
                    "distinct host features should miss");
                t.IsFalse(
                    cache.lookup(modeKey).valid(),
                    "instrumented and normal blocks should not alias");
                t.IsFalse(
                    cache.lookup(formKey).valid(),
                    "basic blocks and linear traces should not alias");
                t.IsNotNull(
                    cache.resolve(first),
                    "same-generation key variants must not invalidate residents");

                const VuProgramCacheDiagnostics stats =
                    cache.diagnostics();
                t.Equals(
                    stats.hits, uint64_t{1u},
                    "one warm lookup should be recorded");
                t.Equals(
                    stats.misses, uint64_t{6u},
                    "cold and five distinct keys should be misses");
                t.Equals(
                    stats.compilations, uint64_t{1u},
                    "duplicate insertion should not count as compilation");
                t.Equals(
                    stats.generatedBytes, uint64_t{1u},
                    "diagnostics should retain emitted native bytes");
                t.Equals(
                    stats.compilationNanoseconds, uint64_t{37u},
                    "diagnostics should retain compile time");
            });

        tc.Run(
            "direct handles require the complete current key and epoch",
            [](TestCase &t)
            {
                int memoryIdentity = 0;
                int codeIdentity = 0;
                const VuProgramKey key =
                    syntheticKey(
                        VuUnitId::Vu1,
                        &memoryIdentity, &codeIdentity);
                VuProgramCache cache(VuUnitId::Vu1);
                const VuProgramHandle handle =
                    cache.insert(makeProgram(key));

                t.IsNotNull(
                    cache.resolveCurrent(handle, key),
                    "the matching current handle should bypass hashing");

                VuProgramKey differentEntry = key;
                differentEntry.entryPc = 8u;
                t.IsNull(
                    cache.resolveCurrent(
                        handle, differentEntry),
                    "a direct handle must not alias another entry");

                VuProgramKey newerGeneration = key;
                ++newerGeneration.codeGeneration;
                t.IsNull(
                    cache.resolveCurrent(
                        handle, newerGeneration),
                    "a direct handle must not cross generations");
                t.IsNotNull(
                    cache.resolve(handle),
                    "a rejected direct probe must not mutate cache scope");

                cache.flush();
                t.IsNull(
                    cache.resolveCurrent(handle, key),
                    "a stale epoch must reject a direct handle");
                t.Equals(
                    cache.diagnostics().hits,
                    uint64_t{1u},
                    "only the exact direct resolution should count as a hit");
            });

        tc.Run(
            "link slots accept only compatible compilation namespaces",
            [](TestCase &t)
            {
                int memoryIdentity = 0;
                int codeIdentity = 0;
                constexpr uintptr_t backendIdentity =
                    0x12345678u;
                VuProgramKey normalSource =
                    syntheticKey(
                        VuUnitId::Vu1,
                        &memoryIdentity, &codeIdentity);
                normalSource.codeContentIdentity = 7u;
                VuProgramKey normalTarget = normalSource;
                normalTarget.entryPc = 8u;
                VuProgramKey instrumentedSource =
                    normalSource;
                instrumentedSource.compilationMode =
                    VuCompilationMode::Instrumented;
                VuProgramKey instrumentedTarget =
                    instrumentedSource;
                instrumentedTarget.entryPc = 8u;

                VuProgramCache cache(VuUnitId::Vu1);
                const VuProgramHandle normalSourceHandle =
                    cache.insert(
                        makeLinkedProgram(
                            normalSource,
                            backendIdentity, {8u}));
                const VuProgramHandle normalTargetHandle =
                    cache.insert(
                        makeLinkedProgram(
                            normalTarget,
                            backendIdentity));
                const VuProgramHandle
                    instrumentedSourceHandle =
                        cache.insert(
                            makeLinkedProgram(
                                instrumentedSource,
                                backendIdentity, {8u}));
                const VuProgramHandle
                    instrumentedTargetHandle =
                        cache.insert(
                            makeLinkedProgram(
                                instrumentedTarget,
                                backendIdentity));

                const VuCompiledProgram *const
                    normalSourceProgram =
                        cache.resolve(
                            normalSourceHandle);
                const VuCompiledProgram *const
                    instrumentedSourceProgram =
                        cache.resolve(
                            instrumentedSourceHandle);
                t.IsNotNull(
                    normalSourceProgram,
                    "the normal source should resolve");
                t.IsNotNull(
                    instrumentedSourceProgram,
                    "the instrumented source should resolve");
                if (!normalSourceProgram ||
                    !instrumentedSourceProgram)
                {
                    return;
                }
                VuNativeLinkState *const normalLinks =
                    normalSourceProgram->
                        outgoingLinks.get();
                VuNativeLinkState *const
                    instrumentedLinks =
                        instrumentedSourceProgram->
                            outgoingLinks.get();

                t.IsTrue(
                    cache.populateLink(
                        normalLinks,
                        normalTargetHandle,
                        normalTarget) ==
                        VuProgramLinkResult::Linked,
                    "normal blocks should link");
                t.IsTrue(
                    cache.populateLink(
                        instrumentedLinks,
                        instrumentedTargetHandle,
                        instrumentedTarget) ==
                        VuProgramLinkResult::Linked,
                    "instrumented blocks should link separately");
                t.IsTrue(
                    cache.populateLink(
                        normalLinks,
                        instrumentedTargetHandle,
                        instrumentedTarget) ==
                        VuProgramLinkResult::Incompatible,
                    "normal code must not link to instrumented code");
                t.IsTrue(
                    cache.populateLink(
                        instrumentedLinks,
                        normalTargetHandle,
                        normalTarget) ==
                        VuProgramLinkResult::Incompatible,
                    "instrumented code must not link to normal code");
                VuProgramKey basicTarget = normalTarget;
                basicTarget.blockForm =
                    VuIrBlockForm::Basic;
                const VuProgramHandle basicTargetHandle =
                    cache.insert(
                        makeLinkedProgram(
                            basicTarget,
                            backendIdentity));
                t.IsTrue(
                    cache.populateLink(
                        normalLinks,
                        basicTargetHandle,
                        basicTarget) ==
                        VuProgramLinkResult::Incompatible,
                    "linear traces must not link to basic blocks");

                const VuNativeLinkSlot &normalSlot =
                    normalLinks->
                        staticTargets.front();
                t.Equals(
                    normalSlot.targetPc, uint32_t{8u},
                    "the known target should use its static slot");
                t.Equals(
                    normalSlot.codeGeneration,
                    normalTarget.codeGeneration,
                    "the slot should publish its generation");
                t.IsTrue(
                    normalSlot.targetEntry != 0u,
                    "the slot should publish an executable target");

                VuProgramKey foreignBackend =
                    normalTarget;
                foreignBackend.hostFeatures ^= 1u;
                const VuProgramHandle foreignHandle =
                    cache.insert(
                        makeLinkedProgram(
                            foreignBackend,
                            backendIdentity));
                t.IsTrue(
                    cache.populateLink(
                        normalLinks,
                        foreignHandle,
                        foreignBackend) ==
                        VuProgramLinkResult::Incompatible,
                    "host-feature namespaces must not link");
                t.Equals(
                    cache.diagnostics().linkResolutions,
                    uint64_t{2u},
                    "only compatible links should resolve");
                t.Equals(
                    cache.diagnostics().
                        linkResolutionFailures,
                    uint64_t{4u},
                    "every incompatible attempt should be counted");
            });

        tc.Run(
            "generation and eviction clear links before native storage",
            [](TestCase &t)
            {
                int memoryIdentity = 0;
                int codeIdentity = 0;
                constexpr uintptr_t backendIdentity =
                    0x87654321u;
                VuProgramKey sourceKey =
                    syntheticKey(
                        VuUnitId::Vu1,
                        &memoryIdentity, &codeIdentity);
                sourceKey.codeContentIdentity = 0x55u;
                VuProgramKey targetKey = sourceKey;
                targetKey.entryPc = 8u;
                VuProgramCache cache(
                    VuUnitId::Vu1,
                    {
                        .maximumPrograms = 2u,
                        .maximumExecutableBytes =
                            VuExecutableMemory::pageSize() *
                            2u,
                    });
                const VuProgramHandle sourceHandle =
                    cache.insert(
                        makeLinkedProgram(
                            sourceKey,
                            backendIdentity));
                const VuProgramHandle targetHandle =
                    cache.insert(
                        makeLinkedProgram(
                            targetKey,
                            backendIdentity));
                const VuCompiledProgram *const source =
                    cache.resolve(sourceHandle);
                t.IsNotNull(
                    source,
                    "the source should be resident");
                if (!source)
                    return;
                const std::shared_ptr<VuNativeLinkState>
                    retainedLinks = source->outgoingLinks;
                t.IsTrue(
                    cache.populateLink(
                        retainedLinks.get(),
                        targetHandle, targetKey) ==
                        VuProgramLinkResult::Linked,
                    "an unknown target should use a dynamic slot");
                t.IsTrue(
                    retainedLinks->
                        dynamicTargets.front().
                        targetEntry != 0u,
                    "the dynamic target should be live");
                const uintptr_t originalTargetEntry =
                    retainedLinks->
                        dynamicTargets.front().
                        targetEntry;

                VuProgramKey nextSource = sourceKey;
                ++nextSource.codeGeneration;
                t.IsTrue(
                    cache.lookup(nextSource).valid(),
                    "identical content should survive generation activation");
                t.Equals(
                    retainedLinks->
                        dynamicTargets.front().
                        targetEntry,
                    originalTargetEntry,
                    "generation activation should retain storage");
                t.Equals(
                    retainedLinks->
                        dynamicTargets.front().
                        codeGeneration,
                    sourceKey.codeGeneration,
                    "the retained slot must remain generation-stale");
                t.Equals(
                    retainedLinks->
                        dynamicTargets.front().targetPc,
                    targetKey.entryPc,
                    "logical invalidation may retain target metadata");

                VuProgramKey nextTarget = targetKey;
                nextTarget.codeGeneration =
                    nextSource.codeGeneration;
                const VuProgramHandle reboundSource =
                    cache.lookup(nextSource);
                const VuProgramHandle reboundTarget =
                    cache.lookup(nextTarget);
                const VuCompiledProgram *const
                    reboundSourceProgram =
                        cache.resolve(reboundSource);
                t.IsNotNull(
                    reboundSourceProgram,
                    "the source should rebind in the new epoch");
                t.IsTrue(
                    cache.populateLink(
                        reboundSourceProgram
                            ? reboundSourceProgram->
                                  outgoingLinks.get()
                            : nullptr,
                        reboundTarget, nextTarget) ==
                        VuProgramLinkResult::Linked,
                    "the current generation should relink");

                VuProgramKey evictionKey = nextSource;
                evictionKey.hostFeatures ^= 1u;
                const VuProgramHandle afterEviction =
                    cache.insert(
                        makeLinkedProgram(
                            evictionKey,
                            backendIdentity));
                t.IsTrue(
                    afterEviction.valid(),
                    "the post-eviction program should be resident");
                t.Equals(
                    retainedLinks->
                        dynamicTargets.front().
                        targetEntry,
                    uintptr_t{0u},
                    "eviction must unlink before releasing target code");
                const VuProgramCacheDiagnostics diagnostics =
                    cache.diagnostics();
                t.Equals(
                    diagnostics.linkInvalidations,
                    uint64_t{1u},
                    "only reclamation should physically clear a live link");
                t.Equals(
                    diagnostics.evictionFlushes,
                    uint64_t{1u},
                    "the forced cache churn should flush once");
            });

        tc.Run(
            "dynamic link targets use bounded deterministic replacement",
            [](TestCase &t)
            {
                int memoryIdentity = 0;
                int codeIdentity = 0;
                constexpr uintptr_t backendIdentity =
                    0x31415926u;
                VuProgramKey sourceKey =
                    syntheticKey(
                        VuUnitId::Vu1,
                        &memoryIdentity, &codeIdentity);
                sourceKey.codeSize = 64u;
                sourceKey.addressMask = 63u;
                sourceKey.codeContentIdentity = 0x77u;

                VuProgramCache cache(VuUnitId::Vu1);
                const VuProgramHandle sourceHandle =
                    cache.insert(
                        makeLinkedProgram(
                            sourceKey,
                            backendIdentity));
                std::array<VuProgramKey, 6u> targetKeys{};
                std::array<VuProgramHandle, 6u>
                    targetHandles{};
                for (size_t index = 0u;
                     index < targetKeys.size();
                     ++index)
                {
                    targetKeys[index] = sourceKey;
                    targetKeys[index].entryPc =
                        static_cast<uint32_t>(
                            (index + 1u) * 8u);
                    targetHandles[index] =
                        cache.insert(
                            makeLinkedProgram(
                                targetKeys[index],
                                backendIdentity));
                    t.IsTrue(
                        targetHandles[index].valid(),
                        "every dynamic target should be resident");
                }

                const VuCompiledProgram *const source =
                    cache.resolve(sourceHandle);
                t.IsNotNull(
                    source,
                    "the dynamic source should remain resident");
                if (!source || !source->outgoingLinks)
                    return;
                VuNativeLinkState &links =
                    *source->outgoingLinks;
                for (size_t index = 0u;
                     index < targetKeys.size();
                     ++index)
                {
                    t.IsTrue(
                        cache.populateLink(
                            &links,
                            targetHandles[index],
                            targetKeys[index]) ==
                            VuProgramLinkResult::Linked,
                        "every dynamic edge should resolve");
                }

                const auto containsTarget =
                    [&](uint32_t targetPc)
                    {
                        return std::any_of(
                            links.dynamicTargets.begin(),
                            links.dynamicTargets.end(),
                            [&](const VuNativeLinkSlot &slot)
                            {
                                return
                                    slot.targetPc ==
                                        targetPc &&
                                    slot.targetEntry != 0u;
                            });
                    };
                t.IsFalse(
                    containsTarget(8u),
                    "the oldest dynamic target should be replaced");
                t.IsFalse(
                    containsTarget(16u),
                    "the second-oldest target should be replaced");
                for (const uint32_t targetPc :
                     {24u, 32u, 40u, 48u})
                {
                    t.IsTrue(
                        containsTarget(targetPc),
                        "the four newest dynamic targets should remain");
                }
                t.Equals(
                    links.nextDynamicSlot, uint32_t{2u},
                    "replacement should advance deterministically");
                t.Equals(
                    cache.diagnostics().linkResolutions,
                    uint64_t{6u},
                    "every successful replacement should be counted");
            });

        tc.Run(
            "invalid keys and malformed programs are rejected",
            [](TestCase &t)
            {
                int memoryIdentity = 0;
                int codeIdentity = 0;
                VuProgramKey key =
                    syntheticKey(
                        VuUnitId::Vu1,
                        &memoryIdentity, &codeIdentity);
                std::string diagnostic;
                t.IsTrue(
                    VuProgramCache::validKey(key, &diagnostic),
                    "the synthetic key should be valid");

                VuProgramKey invalid = key;
                invalid.memoryIdentity = 0u;
                t.IsFalse(
                    VuProgramCache::validKey(
                        invalid, &diagnostic),
                    "zero identity should be rejected");
                invalid = key;
                invalid.codeSize = 24u;
                invalid.addressMask = 23u;
                t.IsFalse(
                    VuProgramCache::validKey(
                        invalid, &diagnostic),
                    "non-power-of-two code extent should be rejected");
                invalid = key;
                invalid.entryPc = 4u;
                t.IsFalse(
                    VuProgramCache::validKey(
                        invalid, &diagnostic),
                    "unaligned entry PC should be rejected");

                VuProgramCache cache(VuUnitId::Vu1);
                VuProgramKey wrongUnit = key;
                wrongUnit.unit = VuUnitId::Vu0;
                t.IsFalse(
                    cache.lookup(wrongUnit).valid(),
                    "a key for another VU should be rejected");

                VuCompiledProgram malformed = makeProgram(key);
                malformed.block.pairs.front().cycles = 2u;
                t.IsFalse(
                    cache.insert(
                             std::move(malformed), &diagnostic)
                        .valid(),
                    "malformed IR should be rejected");
                t.IsTrue(
                    diagnostic.find("malformed") !=
                        std::string::npos,
                    "malformed IR rejection should be diagnostic");

                VuCompiledProgram writable = makeProgram(key);
                writable.nativeCode =
                    VuExecutableMemory::allocate(8u, &diagnostic);
                t.IsFalse(
                    cache.insert(
                             std::move(writable), &diagnostic)
                        .valid(),
                    "unfinalized native code should be rejected");
                t.Equals(
                    cache.diagnostics().residentPrograms,
                    size_t{0u},
                    "rejected programs must not become resident");
            });

        tc.Run(
            "generation changes invalidate stale handles including rollover",
            [](TestCase &t)
            {
                int memoryIdentity = 0;
                int codeIdentity = 0;
                VuProgramKey key =
                    syntheticKey(
                        VuUnitId::Vu1,
                        &memoryIdentity, &codeIdentity);
                key.codeGeneration =
                    std::numeric_limits<uint64_t>::max();
                VuProgramCache cache(VuUnitId::Vu1);
                const VuProgramHandle beforeRollover =
                    cache.insert(makeProgram(key));
                t.IsTrue(
                    beforeRollover.valid(),
                    "the maximum generation should be cacheable");

                key.codeGeneration = 0u;
                t.IsFalse(
                    cache.lookup(key).valid(),
                    "wrapped generation should not reuse old code");
                t.IsNull(
                    cache.resolve(beforeRollover),
                    "generation invalidation should stale old handles");
                const VuProgramHandle afterRollover =
                    cache.insert(makeProgram(key));
                t.IsTrue(
                    afterRollover.valid(),
                    "the wrapped generation should compile normally");
                t.IsFalse(
                    afterRollover == beforeRollover,
                    "a new epoch should distinguish reused indices");

                const VuProgramCacheDiagnostics stats =
                    cache.diagnostics();
                t.Equals(
                    stats.invalidations, uint64_t{1u},
                    "generation rollover should publish one invalidation");
                t.Equals(
                    stats.invalidatedPrograms, uint64_t{1u},
                    "rollover should discard the prior program");
            });

        tc.Run(
            "exact content identity reuses programs across stale generations",
            [](TestCase &t)
            {
                int memoryIdentity = 0;
                int codeIdentity = 0;
                VuProgramKey firstKey =
                    syntheticKey(
                        VuUnitId::Vu1,
                        &memoryIdentity, &codeIdentity);
                firstKey.codeContentIdentity = 0x1234u;
                VuProgramCache cache(VuUnitId::Vu1);
                const VuProgramHandle first =
                    cache.insert(makeProgram(firstKey));

                VuProgramKey sameContent = firstKey;
                ++sameContent.codeGeneration;
                t.IsNull(
                    cache.resolveCurrent(first, sameContent),
                    "a direct handle must not cross a write generation");
                const VuProgramHandle reused =
                    cache.lookup(sameContent);
                t.IsTrue(
                    reused.valid(),
                    "an exact recurring code image should reuse native code");
                t.IsFalse(
                    reused == first,
                    "generation activation should advance the handle epoch");
                t.IsNull(
                    cache.resolve(first),
                    "the pre-write handle should remain stale");
                t.IsNotNull(
                    cache.resolveCurrent(reused, sameContent),
                    "the rebound handle should resolve current content");

                VuProgramKey differentContent = sameContent;
                ++differentContent.codeGeneration;
                ++differentContent.codeContentIdentity;
                t.IsFalse(
                    cache.lookup(differentContent).valid(),
                    "different exact content must not alias resident code");
                const VuProgramHandle second =
                    cache.insert(makeProgram(differentContent));
                t.IsTrue(
                    second.valid(),
                    "different content should compile independently");

                VuProgramKey recurringContent = firstKey;
                recurringContent.codeGeneration =
                    differentContent.codeGeneration + 1u;
                const VuProgramHandle recurring =
                    cache.lookup(recurringContent);
                t.IsTrue(
                    recurring.valid(),
                    "a later return to exact content should reuse the first program");
                t.IsNotNull(
                    cache.resolveCurrent(
                        recurring, recurringContent),
                    "the recurring content handle should be current");

                const VuProgramCacheDiagnostics stats =
                    cache.diagnostics();
                t.Equals(
                    stats.compilations, uint64_t{2u},
                    "only distinct content should compile");
                t.Equals(
                    stats.invalidations, uint64_t{3u},
                    "every generation should stale prior handles");
                t.Equals(
                    stats.residentPrograms, size_t{2u},
                    "both exact content images should remain bounded residents");
            });

        tc.Run(
            "code-space identity and extent changes replace the active scope",
            [](TestCase &t)
            {
                int firstMemory = 0;
                int secondMemory = 0;
                int firstCode = 0;
                int secondCode = 0;
                VuProgramKey key =
                    syntheticKey(
                        VuUnitId::Vu1,
                        &firstMemory, &firstCode);
                VuProgramCache cache(VuUnitId::Vu1);
                const VuProgramHandle first =
                    cache.insert(makeProgram(key));

                VuProgramKey memoryKeyValue = key;
                memoryKeyValue.memoryIdentity =
                    reinterpret_cast<uintptr_t>(&secondMemory);
                t.IsFalse(
                    cache.lookup(memoryKeyValue).valid(),
                    "a different memory owner should miss");
                t.IsNull(
                    cache.resolve(first),
                    "memory identity change should stale old handles");
                const VuProgramHandle second =
                    cache.insert(makeProgram(memoryKeyValue));

                VuProgramKey codeKey = memoryKeyValue;
                codeKey.codeIdentity =
                    reinterpret_cast<uintptr_t>(&secondCode);
                t.IsFalse(
                    cache.lookup(codeKey).valid(),
                    "a different code mapping should miss");
                t.IsNull(
                    cache.resolve(second),
                    "code identity change should stale old handles");
                const VuProgramHandle third =
                    cache.insert(makeProgram(codeKey));

                VuProgramKey extentKey = codeKey;
                extentKey.codeSize = 32u;
                extentKey.addressMask = 31u;
                t.IsFalse(
                    cache.lookup(extentKey).valid(),
                    "a different code extent should miss");
                t.IsNull(
                    cache.resolve(third),
                    "code extent change should stale old handles");
                t.Equals(
                    cache.diagnostics().invalidations,
                    uint64_t{3u},
                    "each scope replacement should invalidate once");
            });

        tc.Run(
            "direct and MPG writes invalidate while data writes and reset do not",
            [](TestCase &t)
            {
                PS2Memory memory;
                t.IsTrue(
                    memory.initialize(),
                    "PS2 memory should initialize");
                VuUnit vu1(VuUnitId::Vu1);
                VuProgramCache &cache = vu1.programCache();

                VuProgramKey key =
                    memoryKey(memory, VuUnitId::Vu1);
                const VuProgramHandle initial =
                    cache.insert(makeProgram(key));
                t.IsTrue(
                    initial.valid(),
                    "initial VU1 program should be cached");

                const uint64_t beforeData =
                    memory.getVU1CodeGeneration();
                memory.write32(
                    PS2_VU1_DATA_BASE, 0x11223344u);
                t.Equals(
                    memory.getVU1CodeGeneration(), beforeData,
                    "VU data writes must not change code generation");
                t.Equals(
                    cache.lookup(key), initial,
                    "data-only writes should retain compiled code");

                vu1.reset();
                t.Equals(
                    cache.lookup(key), initial,
                    "architectural reset should retain unchanged code");

                memory.write32(
                    PS2_VU1_CODE_BASE, 0x55667788u);
                VuProgramKey directKey =
                    memoryKey(memory, VuUnitId::Vu1);
                t.Equals(
                    directKey.codeGeneration,
                    key.codeGeneration + 1u,
                    "a direct micro-memory write should publish generation");
                t.IsFalse(
                    cache.lookup(directKey).valid(),
                    "a direct code write should invalidate native code");
                t.IsNull(
                    cache.resolve(initial),
                    "a direct write should stale the old handle");

                const VuProgramHandle afterDirect =
                    cache.insert(makeProgram(directKey));
                std::array<uint8_t, 8u> payload{};
                std::memcpy(
                    payload.data(),
                    memory.getVU1Code(), payload.size());
                const std::vector<uint8_t> identicalMpg =
                    makeMpgPacket(
                        0u, payload.data(), 1u);
                memory.processVIF1Data(
                    identicalMpg.data(),
                    static_cast<uint32_t>(
                        identicalMpg.size()));
                VuProgramKey identicalKey =
                    memoryKey(memory, VuUnitId::Vu1);
                t.Equals(
                    identicalKey.codeGeneration,
                    directKey.codeGeneration + 1u,
                    "even identical MPG uploads should publish generation");
                t.IsFalse(
                    cache.lookup(identicalKey).valid(),
                    "identical reupload must not cross generations");
                t.IsNull(
                    cache.resolve(afterDirect),
                    "identical MPG should stale previous native code");

                const VuProgramHandle beforeWrap =
                    cache.insert(makeProgram(identicalKey));
                std::array<uint8_t, 16u> wrapPayload{};
                for (size_t index = 0u;
                     index < wrapPayload.size(); ++index)
                {
                    wrapPayload[index] =
                        static_cast<uint8_t>(0x80u + index);
                }
                const std::vector<uint8_t> wrappedMpg =
                    makeMpgPacket(
                        0x07ffu, wrapPayload.data(), 2u);
                memory.processVIF1Data(
                    wrappedMpg.data(),
                    static_cast<uint32_t>(
                        wrappedMpg.size()));
                t.IsTrue(
                    std::memcmp(
                        memory.getVU1Code() +
                            PS2_VU1_CODE_SIZE - 8u,
                        wrapPayload.data(), 8u) == 0,
                    "wrapped MPG should write the MicroMem tail");
                t.IsTrue(
                    std::memcmp(
                        memory.getVU1Code(),
                        wrapPayload.data() + 8u, 8u) == 0,
                    "wrapped MPG should continue at MicroMem zero");
                const VuProgramKey wrappedKey =
                    memoryKey(memory, VuUnitId::Vu1);
                t.IsFalse(
                    cache.lookup(wrappedKey).valid(),
                    "wrapped MPG should invalidate the unit cache");
                t.IsNull(
                    cache.resolve(beforeWrap),
                    "wrapped MPG should stale previous handles");
            });

        tc.Run(
            "VU0 and VU1 caches remain independent",
            [](TestCase &t)
            {
                PS2Memory memory;
                t.IsTrue(
                    memory.initialize(),
                    "PS2 memory should initialize");
                VuUnit vu0(VuUnitId::Vu0);
                VuUnit vu1(VuUnitId::Vu1);
                t.Equals(
                    vu0.unitId(), VuUnitId::Vu0,
                    "VU0 should retain its cache identity");
                t.Equals(
                    vu1.unitId(), VuUnitId::Vu1,
                    "VU1 should retain its cache identity");
                t.IsNull(
                    vu0.programCacheIfCreated(),
                    "the VU0 cache should be lazy");
                t.IsNull(
                    vu1.programCacheIfCreated(),
                    "the VU1 cache should be lazy");

                VuProgramCache &vu0Cache = vu0.programCache();
                VuProgramCache &vu1Cache = vu1.programCache();
                const VuProgramKey vu0Key =
                    memoryKey(memory, VuUnitId::Vu0);
                const VuProgramKey vu1Key =
                    memoryKey(memory, VuUnitId::Vu1);
                const VuProgramHandle vu0Handle =
                    vu0Cache.insert(makeProgram(vu0Key));
                const VuProgramHandle vu1Handle =
                    vu1Cache.insert(makeProgram(vu1Key));

                memory.write32(
                    PS2_VU1_CODE_BASE, 0xabcdef01u);
                t.Equals(
                    vu0Cache.lookup(vu0Key), vu0Handle,
                    "VU1 code writes must not invalidate VU0");
                t.IsNotNull(
                    vu0Cache.resolve(vu0Handle),
                    "VU0 handle should remain current");
                const VuProgramKey changedVu1 =
                    memoryKey(memory, VuUnitId::Vu1);
                t.IsFalse(
                    vu1Cache.lookup(changedVu1).valid(),
                    "VU1 should observe its own code generation");
                t.IsNull(
                    vu1Cache.resolve(vu1Handle),
                    "VU1 handle should be stale");

                const VuProgramHandle changedVu1Handle =
                    vu1Cache.insert(makeProgram(changedVu1));
                std::array<uint8_t, 16u> payload{};
                for (size_t index = 0u;
                     index < payload.size(); ++index)
                {
                    payload[index] =
                        static_cast<uint8_t>(0x20u + index);
                }
                const std::vector<uint8_t> vu0WrappedMpg =
                    makeMpgPacket(
                        0x01ffu, payload.data(), 2u);
                memory.processVIF0Data(
                    vu0WrappedMpg.data(),
                    static_cast<uint32_t>(
                        vu0WrappedMpg.size()));
                const VuProgramKey changedVu0 =
                    memoryKey(memory, VuUnitId::Vu0);
                t.IsFalse(
                    vu0Cache.lookup(changedVu0).valid(),
                    "VU0 MPG should invalidate VU0");
                t.IsNull(
                    vu0Cache.resolve(vu0Handle),
                    "VU0 MPG should stale the VU0 handle");
                t.Equals(
                    vu1Cache.lookup(changedVu1),
                    changedVu1Handle,
                    "VU0 MPG must not invalidate VU1");
            });

        tc.Run(
            "bounded eviction flushes deterministically without dangling handles",
            [](TestCase &t)
            {
                int memoryIdentity = 0;
                int codeIdentity = 0;
                VuProgramKey firstKey =
                    syntheticKey(
                        VuUnitId::Vu1,
                        &memoryIdentity, &codeIdentity);
                VuProgramKey secondKey = firstKey;
                secondKey.entryPc = 8u;
                VuProgramKey thirdKey = firstKey;
                thirdKey.hostFeatures ^= 1u;
                const size_t page =
                    VuExecutableMemory::pageSize();
                VuProgramCache cache(
                    VuUnitId::Vu1,
                    {
                        .maximumPrograms = 2u,
                        .maximumExecutableBytes = page * 2u,
                    });

                const VuProgramHandle first =
                    cache.insert(makeProgram(firstKey));
                const VuProgramHandle second =
                    cache.insert(makeProgram(secondKey));
                t.IsNotNull(
                    cache.resolve(first),
                    "first resident should resolve before eviction");
                t.IsNotNull(
                    cache.resolve(second),
                    "second resident should resolve before eviction");

                const VuProgramHandle third =
                    cache.insert(makeProgram(thirdKey));
                t.IsTrue(
                    third.valid(),
                    "the post-flush program should enter the cache");
                t.IsNull(
                    cache.resolve(first),
                    "eviction should stale the first handle");
                t.IsNull(
                    cache.resolve(second),
                    "eviction should stale the second handle");
                t.IsNotNull(
                    cache.resolve(third),
                    "the new epoch's program should resolve");

                std::string diagnostic;
                VuProgramKey oversizedKey = thirdKey;
                oversizedKey.hostFeatures ^= 2u;
                const VuProgramHandle oversized =
                    cache.insert(
                        makeProgram(
                            oversizedKey, page * 2u + 1u),
                        &diagnostic);
                t.IsFalse(
                    oversized.valid(),
                    "a program larger than the byte limit should fail");
                t.IsNotNull(
                    cache.resolve(third),
                    "oversized rejection must not evict current code");

                const VuProgramCacheDiagnostics stats =
                    cache.diagnostics();
                t.Equals(
                    stats.compilations, uint64_t{3u},
                    "three accepted programs should be counted");
                t.Equals(
                    stats.evictionFlushes, uint64_t{1u},
                    "the third program should cause one full flush");
                t.Equals(
                    stats.evictedPrograms, uint64_t{2u},
                    "the deterministic flush should evict two programs");
                t.Equals(
                    stats.residentPrograms, size_t{1u},
                    "only the post-flush program should remain");
                t.Equals(
                    stats.highWaterPrograms, size_t{2u},
                    "the resident high-water mark should be bounded");
                t.Equals(
                    stats.rejectedPrograms, uint64_t{1u},
                    "the oversized program should be diagnosed");

                cache.flush();
                t.IsNull(
                    cache.resolve(third),
                    "manual flush should stale the last handle");
                t.Equals(
                    cache.diagnostics().manualFlushes,
                    uint64_t{1u},
                    "manual flush should be observable");
            });

        tc.Run(
            "cache ownership rejects access from another host thread",
            [](TestCase &t)
            {
                VuProgramCache cache(VuUnitId::Vu1);
                (void)cache.diagnostics();
                std::atomic<bool> rejected{false};
                std::thread other([&]()
                {
                    try
                    {
                        (void)cache.diagnostics();
                    }
                    catch (const std::logic_error &)
                    {
                        rejected.store(
                            true, std::memory_order_relaxed);
                    }
                });
                other.join();
                t.IsTrue(
                    rejected.load(std::memory_order_relaxed),
                    "cross-thread cache access should fail explicitly");
                t.Equals(
                    cache.diagnostics().residentPrograms,
                    size_t{0u},
                    "the owner thread should remain usable");
            });
    });
}
