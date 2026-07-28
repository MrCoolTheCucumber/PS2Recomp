// SPDX-License-Identifier: GPL-3.0-or-later
//
// Disposable Phase 0 feasibility spike for the VU dynamic-recompiler plan.
// This deliberately exercises only the upper lane of one RAC-derived block.
// It is evidence for emitter/ABI/W^X costs, not a production VU backend.

#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu1.h"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <xmmintrin.h>

struct VU1NativeEmitterSpikeAccess
{
    static void prepare(VU1Interpreter &interpreter,
                        const VU1State &initialState)
    {
        interpreter.reset();
        interpreter.m_state = initialState;
        interpreter.m_fmacFlagPipeline = {};
        interpreter.m_fmacFlagPipelineIndex = 0u;
        interpreter.m_workingMac = initialState.mac;
    }

    static void advanceFmac(VU1Interpreter &interpreter)
    {
        interpreter.advanceFmacFlagPipeline();
    }

    static void executeUpper(VU1Interpreter &interpreter, uint32_t upper)
    {
        interpreter.execUpper(upper);
    }

    static void scheduleFmac(VU1Interpreter &interpreter,
                             const float result[4], uint8_t destination)
    {
        interpreter.scheduleFmacFlags(result, destination, false);
    }

    static void flushFmac(VU1Interpreter &interpreter)
    {
        interpreter.flushFmacFlagPipeline();
    }
};

namespace
{
    using Clock = std::chrono::steady_clock;

    constexpr uint64_t kFnvOffset = 1469598103934665603ull;
    constexpr uint64_t kFnvPrime = 1099511628211ull;
    constexpr uint32_t kBlockTraceStart = 116u;
    constexpr uint32_t kBlockPairCount = 44u;
    constexpr uint32_t kDefaultTimedPairs = 4'400'000u;
    constexpr uint32_t kDefaultSamples = 15u;
    constexpr uint32_t kDefaultCompileSamples = 101u;
    constexpr uint32_t kDefaultSideExits = 200'000u;
    constexpr uint32_t kMxcsrRoundingMask = 3u << 13u;
    constexpr uint32_t kMxcsrRoundTowardZero = 3u << 13u;
    constexpr uint32_t kMxcsrDenormalsAreZero = 1u << 6u;
    constexpr uint32_t kMxcsrFlushToZero = 1u << 15u;

    struct FixturePaths
    {
        std::filesystem::path data;
        std::filesystem::path micro;
        std::filesystem::path state;
    };

    struct CapturedBlock
    {
        VU1State initialState{};
        std::vector<uint32_t> pcs;
        std::vector<uint32_t> upper;
    };

    struct alignas(16) SpikeContext
    {
        VU1Interpreter *interpreter = nullptr;
        VU1State *state = nullptr;
        uint32_t nextPair = 0u;
        uint32_t remainingPairs = 0u;
        uint32_t savedMxcsr = 0u;
        uint32_t vuMxcsr = 0u;
        uint32_t invalidEntry = 0u;
        uint32_t reserved = 0u;
        alignas(16) float fmacResult[4]{};
    };

    using NativeFunction = uint32_t (*)(SpikeContext *);

    extern "C" uint32_t vuNativeSpikeProbeCalleeSaved(
        NativeFunction function, SpikeContext *context,
        uint64_t output[6]);

#if defined(__GNUC__) || defined(__clang__)
#define PS2X_SPIKE_NOINLINE __attribute__((noinline))
#else
#define PS2X_SPIKE_NOINLINE
#endif

    extern "C" PS2X_SPIKE_NOINLINE void
    vuNativeSpikeAdvanceFmac(SpikeContext *context)
    {
        VU1NativeEmitterSpikeAccess::advanceFmac(*context->interpreter);
    }

    extern "C" PS2X_SPIKE_NOINLINE void
    vuNativeSpikeExecuteUpper(SpikeContext *context, uint32_t upper)
    {
        VU1NativeEmitterSpikeAccess::executeUpper(
            *context->interpreter, upper);
    }

    extern "C" PS2X_SPIKE_NOINLINE void
    vuNativeSpikeScheduleFmac(SpikeContext *context, uint32_t destination)
    {
        VU1NativeEmitterSpikeAccess::scheduleFmac(
            *context->interpreter, context->fmacResult,
            static_cast<uint8_t>(destination));
    }

    uint8_t destinationMask(uint32_t upper)
    {
        return static_cast<uint8_t>((upper >> 21u) & 0x0fu);
    }

    uint8_t sourceRegister(uint32_t upper)
    {
        return static_cast<uint8_t>((upper >> 11u) & 0x1fu);
    }

    uint8_t destinationRegister(uint32_t upper)
    {
        return static_cast<uint8_t>((upper >> 6u) & 0x1fu);
    }

    bool isMulq(uint32_t upper)
    {
        return (upper & 0x3fu) == 0x1cu;
    }

    bool isUpperNop(uint32_t upper)
    {
        const uint8_t op = static_cast<uint8_t>(upper & 0x3fu);
        if (op < 0x3cu)
            return false;
        const uint8_t special = static_cast<uint8_t>(
            (upper & 0x3u) | ((upper >> 4u) & 0x7cu));
        return special == 0x2fu || special == 0x30u;
    }

    class NativeBlock final : public Xbyak::CodeGenerator
    {
    public:
        NativeBlock(const std::vector<uint32_t> &upper,
                    bool inlineMulqAndNop)
            : Xbyak::CodeGenerator(4096u, Xbyak::AutoGrow),
              m_inlineMulqAndNop(inlineMulqAndNop)
        {
            if (upper.empty())
                throw std::runtime_error("native block cannot be empty");
            emitBlock(upper);
        }

        void finalize()
        {
            if (m_finalized)
                return;

            // AutoGrow mappings begin RW. readyRE resolves forward labels and
            // transitions the final mapping directly to RX; no RWX window is
            // needed. Keep an explicit cache-maintenance call even though
            // x86-64 has coherent instruction/data caches.
            readyRE();
            auto *begin = getCode<uint8_t *>();
            __builtin___clear_cache(
                reinterpret_cast<char *>(begin),
                reinterpret_cast<char *>(begin + getSize()));
            m_finalized = true;
        }

        NativeFunction function() const
        {
            if (!m_finalized)
                throw std::runtime_error("native block is not executable");
            return getCode<NativeFunction>();
        }

    private:
        bool m_inlineMulqAndNop = false;
        bool m_finalized = false;

        static constexpr int contextOffset(size_t value)
        {
            return static_cast<int>(value);
        }

        void emitAbsoluteCall(const void *functionAddress)
        {
            mov(rax, reinterpret_cast<uint64_t>(functionAddress));
            call(rax);
        }

        void emitMulq(uint32_t upper)
        {
            const uint8_t destination = destinationMask(upper);
            const uint8_t source = sourceRegister(upper);
            const uint8_t output = destinationRegister(upper);
            const int stateOffset =
                contextOffset(offsetof(SpikeContext, state));
            const int resultOffset =
                contextOffset(offsetof(SpikeContext, fmacResult));
            const int sourceOffset =
                contextOffset(offsetof(VU1State, vf) +
                              static_cast<size_t>(source) * 4u *
                                  sizeof(float));
            const int outputOffset =
                contextOffset(offsetof(VU1State, vf) +
                              static_cast<size_t>(output) * 4u *
                                  sizeof(float));
            const int qOffset = contextOffset(offsetof(VU1State, q));

            mov(rdx, ptr[r12 + stateOffset]);
            movups(xmm0, ptr[rdx + sourceOffset]);
            movss(xmm1, ptr[rdx + qOffset]);
            shufps(xmm1, xmm1, 0u);
            mulps(xmm0, xmm1);
            movups(ptr[r12 + resultOffset], xmm0);

            if (destination != 0u)
            {
                const uint8_t blendMask = static_cast<uint8_t>(
                    ((destination & 0x8u) ? 0x1u : 0u) |
                    ((destination & 0x4u) ? 0x2u : 0u) |
                    ((destination & 0x2u) ? 0x4u : 0u) |
                    ((destination & 0x1u) ? 0x8u : 0u));
                if (blendMask == 0x0fu)
                {
                    movups(ptr[rdx + outputOffset], xmm0);
                }
                else
                {
                    movups(xmm2, ptr[rdx + outputOffset]);
                    blendps(xmm2, xmm0, blendMask);
                    movups(ptr[rdx + outputOffset], xmm2);
                }
            }

            mov(rdi, r12);
            mov(esi, static_cast<uint32_t>(destination));
            emitAbsoluteCall(
                reinterpret_cast<const void *>(&vuNativeSpikeScheduleFmac));
        }

        void emitBlock(const std::vector<uint32_t> &upper)
        {
            std::vector<Xbyak::Label> entries(upper.size());
            Xbyak::Label exit;
            const int nextPairOffset =
                contextOffset(offsetof(SpikeContext, nextPair));
            const int remainingOffset =
                contextOffset(offsetof(SpikeContext, remainingPairs));
            const int savedMxcsrOffset =
                contextOffset(offsetof(SpikeContext, savedMxcsr));
            const int vuMxcsrOffset =
                contextOffset(offsetof(SpikeContext, vuMxcsr));
            const int invalidEntryOffset =
                contextOffset(offsetof(SpikeContext, invalidEntry));

            // SysV entry has RSP % 16 == 8. Preserve the two nonvolatile
            // registers used by the block and realign before helper calls.
            push(r12);
            push(r13);
            sub(rsp, 8);
            mov(r12, rdi);
            xor_(r13d, r13d);
            stmxcsr(ptr[r12 + savedMxcsrOffset]);
            ldmxcsr(ptr[r12 + vuMxcsrOffset]);

            mov(eax, dword[r12 + nextPairOffset]);
            for (size_t index = 0; index < entries.size(); ++index)
            {
                cmp(eax, static_cast<uint32_t>(index));
                je(entries[index], T_NEAR);
            }
            mov(dword[r12 + invalidEntryOffset], 1u);
            jmp(exit, T_NEAR);

            for (size_t index = 0; index < upper.size(); ++index)
            {
                L(entries[index]);
                cmp(dword[r12 + remainingOffset], 0u);
                je(exit, T_NEAR);

                mov(rdi, r12);
                emitAbsoluteCall(
                    reinterpret_cast<const void *>(&vuNativeSpikeAdvanceFmac));

                const uint32_t instruction = upper[index];
                if (m_inlineMulqAndNop && isMulq(instruction))
                {
                    emitMulq(instruction);
                }
                else if (!(m_inlineMulqAndNop && isUpperNop(instruction)))
                {
                    mov(rdi, r12);
                    mov(esi, instruction);
                    emitAbsoluteCall(reinterpret_cast<const void *>(
                        &vuNativeSpikeExecuteUpper));
                }

                inc(r13d);
                dec(dword[r12 + remainingOffset]);
                mov(dword[r12 + nextPairOffset],
                    static_cast<uint32_t>((index + 1u) % upper.size()));
                cmp(dword[r12 + remainingOffset], 0u);
                je(exit, T_NEAR);
                if (index + 1u == upper.size())
                    jmp(entries.front());
            }

            L(exit);
            ldmxcsr(ptr[r12 + savedMxcsrOffset]);
            mov(eax, r13d);
            add(rsp, 8);
            pop(r13);
            pop(r12);
            ret();
        }
    };

    class ScopedVuMxcsr
    {
    public:
        ScopedVuMxcsr()
            : m_previous(_mm_getcsr())
        {
            _mm_setcsr(vuMxcsr(m_previous));
        }

        ~ScopedVuMxcsr()
        {
            _mm_setcsr(m_previous);
        }

        ScopedVuMxcsr(const ScopedVuMxcsr &) = delete;
        ScopedVuMxcsr &operator=(const ScopedVuMxcsr &) = delete;

        static uint32_t vuMxcsr(uint32_t host)
        {
            return (host & ~kMxcsrRoundingMask) |
                   kMxcsrRoundTowardZero |
                   kMxcsrDenormalsAreZero |
                   kMxcsrFlushToZero;
        }

    private:
        uint32_t m_previous = 0u;
    };

    struct Distribution
    {
        double minimum = 0.0;
        double median = 0.0;
        double mean = 0.0;
        double p95 = 0.0;
        double maximum = 0.0;
    };

    struct TimedPath
    {
        Distribution pairsPerSecond;
        uint64_t stateHash = 0u;
    };

    struct SideExitPath
    {
        Distribution nanosecondsPerExit;
        uint64_t stateHash = 0u;
        uint32_t nextPair = 0u;
    };

    struct MappingPermissions
    {
        std::string permissions;
        uintptr_t begin = 0u;
        uintptr_t end = 0u;
    };

    bool readFile(const std::filesystem::path &path,
                  void *destination, size_t size)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            return false;
        input.read(static_cast<char *>(destination),
                   static_cast<std::streamsize>(size));
        return static_cast<size_t>(input.gcount()) == size;
    }

    bool resolveFixture(const std::filesystem::path &directory,
                        FixturePaths &paths)
    {
        const FixturePaths standard{
            directory / "vu1-data.bin",
            directory / "vu1-code.bin",
            directory / "vu1-replay-state.bin",
        };
        if (std::filesystem::is_regular_file(standard.data) &&
            std::filesystem::is_regular_file(standard.micro) &&
            std::filesystem::is_regular_file(standard.state))
        {
            paths = standard;
            return true;
        }

        std::vector<FixturePaths> candidates;
        std::error_code error;
        for (const auto &entry :
             std::filesystem::directory_iterator(directory, error))
        {
            if (error || !entry.is_regular_file())
                continue;
            const std::string filename = entry.path().filename().string();
            constexpr std::string_view suffix = "-state.bin";
            if (!filename.ends_with(suffix))
                continue;
            const std::string prefix =
                filename.substr(0u, filename.size() - suffix.size());
            FixturePaths candidate{
                directory / (prefix + "-data.bin"),
                directory / (prefix + "-micro.bin"),
                entry.path(),
            };
            if (std::filesystem::is_regular_file(candidate.data) &&
                std::filesystem::is_regular_file(candidate.micro))
            {
                candidates.push_back(std::move(candidate));
            }
        }
        if (candidates.size() != 1u)
            return false;
        paths = std::move(candidates.front());
        return true;
    }

    VU1State decodeFixtureState(const std::array<uint32_t, 159> &words)
    {
        if (words[0] != 0x31555652u || words[1] != 1u)
            throw std::runtime_error("unsupported VU replay state");

        VU1State state{};
        size_t cursor = 5u;
        std::memcpy(state.vf, words.data() + cursor, sizeof(state.vf));
        cursor += 32u * 4u;
        for (size_t index = 0u; index < 16u; ++index)
            state.vi[index] = static_cast<int32_t>(words[cursor++]);
        std::memcpy(state.acc, words.data() + cursor, sizeof(state.acc));
        cursor += 4u;
        std::memcpy(&state.q, &words[cursor++], sizeof(state.q));
        std::memcpy(&state.p, &words[cursor++], sizeof(state.p));
        std::memcpy(&state.i, &words[cursor++], sizeof(state.i));
        state.status = words[cursor++];
        state.mac = words[cursor++];
        state.clip = words[cursor++];
        return state;
    }

    CapturedBlock captureBlock(const std::filesystem::path &fixtureDirectory)
    {
        FixturePaths fixture;
        if (!resolveFixture(fixtureDirectory, fixture))
            throw std::runtime_error("could not resolve replay fixture");

        std::array<uint8_t, PS2_VU1_DATA_SIZE> initialData{};
        std::array<uint8_t, PS2_VU1_CODE_SIZE> micro{};
        std::array<uint32_t, 159> stateWords{};
        if (!readFile(fixture.data, initialData.data(), initialData.size()) ||
            !readFile(fixture.micro, micro.data(), micro.size()) ||
            !readFile(fixture.state, stateWords.data(),
                      stateWords.size() * sizeof(uint32_t)))
        {
            throw std::runtime_error("failed to read replay fixture");
        }

        PS2Memory memory;
        if (!memory.initialize())
            throw std::runtime_error("failed to initialize PS2 memory");
        GS gs;
        gs.init(memory.getGSVRAM(),
                static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &memory.gs());
        std::memcpy(memory.getVU1Code(), micro.data(), micro.size());
        std::memcpy(memory.getVU1Data(), initialData.data(),
                    initialData.size());

        CapturedBlock block;
        bool capturedInitialState = false;
        VU1Interpreter interpreter;
        interpreter.state() = decodeFixtureState(stateWords);
        interpreter.setInstructionObserver(
            [&](uint64_t index, uint32_t pc, uint32_t, uint32_t upper,
                const VU1State &state)
            {
                if (index == kBlockTraceStart)
                {
                    block.initialState = state;
                    capturedInitialState = true;
                }
                if (index >= kBlockTraceStart &&
                    index < kBlockTraceStart + kBlockPairCount)
                {
                    block.pcs.push_back(pc);
                    block.upper.push_back(upper);
                }
            });
        interpreter.execute(
            memory.getVU1Code(), PS2_VU1_CODE_SIZE,
            memory.getVU1Data(), PS2_VU1_DATA_SIZE,
            gs, &memory, stateWords[2], stateWords[3], stateWords[4],
            kBlockTraceStart + kBlockPairCount);

        if (!capturedInitialState ||
            block.upper.size() != kBlockPairCount)
        {
            throw std::runtime_error("fixture did not reach selected block");
        }
        for (size_t index = 1u; index < block.pcs.size(); ++index)
        {
            if (block.pcs[index] !=
                ((block.pcs[index - 1u] + 8u) & 0x3fffu))
            {
                throw std::runtime_error(
                    "selected fixture block is not sequential");
            }
        }
        return block;
    }

    template <typename Value>
    void hashValue(uint64_t &hash, const Value &value)
    {
        const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
        for (size_t index = 0u; index < sizeof(Value); ++index)
        {
            hash ^= bytes[index];
            hash *= kFnvPrime;
        }
    }

    uint64_t hashState(const VU1State &state)
    {
        uint64_t hash = kFnvOffset;
        hashValue(hash, state.vf);
        hashValue(hash, state.vi);
        hashValue(hash, state.acc);
        hashValue(hash, state.q);
        hashValue(hash, state.p);
        hashValue(hash, state.i);
        hashValue(hash, state.pc);
        hashValue(hash, state.mac);
        hashValue(hash, state.clip);
        hashValue(hash, state.status);
        hashValue(hash, state.ebit);
        hashValue(hash, state.top);
        hashValue(hash, state.itop);
        hashValue(hash, state.branchPending);
        hashValue(hash, state.branchTarget);
        hashValue(hash, state.branchDelay);
        hashValue(hash, state.viBackupCycles);
        hashValue(hash, state.viBackupRegister);
        hashValue(hash, state.viBackupValue);
        return hash;
    }

    Distribution summarize(std::vector<double> values)
    {
        if (values.empty())
            throw std::runtime_error("cannot summarize an empty sample set");
        std::sort(values.begin(), values.end());
        double sum = 0.0;
        for (const double value : values)
            sum += value;
        const size_t p95Index = static_cast<size_t>(
            std::ceil(0.95 * static_cast<double>(values.size())) - 1.0);
        return {
            .minimum = values.front(),
            .median = values[values.size() / 2u],
            .mean = sum / static_cast<double>(values.size()),
            .p95 = values[std::min(p95Index, values.size() - 1u)],
            .maximum = values.back(),
        };
    }

    SpikeContext makeContext(VU1Interpreter &interpreter)
    {
        SpikeContext context;
        context.interpreter = &interpreter;
        context.state = &interpreter.state();
        context.vuMxcsr = ScopedVuMxcsr::vuMxcsr(_mm_getcsr());
        return context;
    }

    void runDirect(VU1Interpreter &interpreter,
                   const std::vector<uint32_t> &upper,
                   uint32_t pairCount)
    {
        ScopedVuMxcsr floatMode;
        size_t nextPair = 0u;
        for (uint32_t pair = 0u; pair < pairCount; ++pair)
        {
            VU1NativeEmitterSpikeAccess::advanceFmac(interpreter);
            VU1NativeEmitterSpikeAccess::executeUpper(
                interpreter, upper[nextPair]);
            if (++nextPair == upper.size())
                nextPair = 0u;
        }
    }

    TimedPath measureDirect(const CapturedBlock &block, uint32_t pairs,
                            uint32_t samples)
    {
        std::vector<double> rates;
        rates.reserve(samples);
        uint64_t expectedHash = 0u;
        for (uint32_t sample = 0u; sample < samples; ++sample)
        {
            VU1Interpreter interpreter;
            VU1NativeEmitterSpikeAccess::prepare(
                interpreter, block.initialState);
            const auto begin = Clock::now();
            runDirect(interpreter, block.upper, pairs);
            const auto end = Clock::now();
            VU1NativeEmitterSpikeAccess::flushFmac(interpreter);
            const uint64_t stateHash = hashState(interpreter.state());
            if (sample == 0u)
                expectedHash = stateHash;
            else if (stateHash != expectedHash)
                throw std::runtime_error("direct path state was unstable");
            const double seconds =
                std::chrono::duration<double>(end - begin).count();
            rates.push_back(static_cast<double>(pairs) / seconds);
        }
        return {summarize(std::move(rates)), expectedHash};
    }

    TimedPath measureNative(const CapturedBlock &block,
                            NativeFunction function, uint32_t pairs,
                            uint32_t samples)
    {
        std::vector<double> rates;
        rates.reserve(samples);
        uint64_t expectedHash = 0u;
        for (uint32_t sample = 0u; sample < samples; ++sample)
        {
            VU1Interpreter interpreter;
            VU1NativeEmitterSpikeAccess::prepare(
                interpreter, block.initialState);
            SpikeContext context = makeContext(interpreter);
            context.remainingPairs = pairs;
            const auto begin = Clock::now();
            const uint32_t executed = function(&context);
            const auto end = Clock::now();
            if (executed != pairs || context.remainingPairs != 0u ||
                context.invalidEntry != 0u)
            {
                throw std::runtime_error(
                    "native block did not execute the requested work");
            }
            VU1NativeEmitterSpikeAccess::flushFmac(interpreter);
            const uint64_t stateHash = hashState(interpreter.state());
            if (sample == 0u)
                expectedHash = stateHash;
            else if (stateHash != expectedHash)
                throw std::runtime_error("native path state was unstable");
            const double seconds =
                std::chrono::duration<double>(end - begin).count();
            rates.push_back(static_cast<double>(pairs) / seconds);
        }
        return {summarize(std::move(rates)), expectedHash};
    }

    SideExitPath measureSideExits(const CapturedBlock &block,
                                  NativeFunction function,
                                  uint32_t exits, uint32_t samples)
    {
        std::vector<double> costs;
        costs.reserve(samples);
        uint64_t expectedHash = 0u;
        uint32_t expectedNextPair = 0u;
        for (uint32_t sample = 0u; sample < samples; ++sample)
        {
            VU1Interpreter interpreter;
            VU1NativeEmitterSpikeAccess::prepare(
                interpreter, block.initialState);
            SpikeContext context = makeContext(interpreter);
            const auto begin = Clock::now();
            for (uint32_t exitIndex = 0u; exitIndex < exits; ++exitIndex)
            {
                context.remainingPairs = 1u;
                if (function(&context) != 1u)
                    throw std::runtime_error("one-pair side exit failed");
            }
            const auto end = Clock::now();
            VU1NativeEmitterSpikeAccess::flushFmac(interpreter);
            const uint64_t stateHash = hashState(interpreter.state());
            if (sample == 0u)
            {
                expectedHash = stateHash;
                expectedNextPair = context.nextPair;
            }
            else if (stateHash != expectedHash ||
                     context.nextPair != expectedNextPair)
            {
                throw std::runtime_error("side-exit path was unstable");
            }
            const double nanoseconds =
                std::chrono::duration<double, std::nano>(
                    end - begin).count();
            costs.push_back(nanoseconds / static_cast<double>(exits));
        }
        return {
            summarize(std::move(costs)),
            expectedHash,
            expectedNextPair,
        };
    }

    std::vector<double> measureCompilation(
        const std::vector<uint32_t> &upper, bool inlineMulqAndNop,
        uint32_t samples)
    {
        std::vector<double> microseconds;
        microseconds.reserve(samples);
        for (uint32_t sample = 0u; sample < samples; ++sample)
        {
            const auto begin = Clock::now();
            auto block = std::make_unique<NativeBlock>(
                upper, inlineMulqAndNop);
            block->finalize();
            const auto end = Clock::now();
            microseconds.push_back(
                std::chrono::duration<double, std::micro>(
                    end - begin).count());
        }
        return microseconds;
    }

    MappingPermissions mappingForAddress(const void *address)
    {
        const uintptr_t target = reinterpret_cast<uintptr_t>(address);
        std::ifstream maps("/proc/self/maps");
        std::string line;
        while (std::getline(maps, line))
        {
            std::istringstream fields(line);
            std::string range;
            std::string permissions;
            if (!(fields >> range >> permissions))
                continue;
            const size_t separator = range.find('-');
            if (separator == std::string::npos)
                continue;
            const uintptr_t begin = std::stoull(
                range.substr(0u, separator), nullptr, 16);
            const uintptr_t end = std::stoull(
                range.substr(separator + 1u), nullptr, 16);
            if (target >= begin && target < end)
                return {permissions, begin, end};
        }
        throw std::runtime_error(
            "could not find native block in /proc/self/maps");
    }

    bool permissionsAreRwNotExecutable(const std::string &permissions)
    {
        return permissions.size() >= 3u &&
               permissions[0] == 'r' &&
               permissions[1] == 'w' &&
               permissions[2] == '-';
    }

    bool permissionsAreRxNotWritable(const std::string &permissions)
    {
        return permissions.size() >= 3u &&
               permissions[0] == 'r' &&
               permissions[1] == '-' &&
               permissions[2] == 'x';
    }

    uint32_t parseBounded(const char *text, uint32_t minimum,
                          const char *name)
    {
        size_t consumed = 0u;
        const unsigned long long parsed =
            std::stoull(text, &consumed, 0);
        if (consumed != std::strlen(text) ||
            parsed < minimum ||
            parsed > std::numeric_limits<uint32_t>::max())
        {
            throw std::runtime_error(std::string("invalid ") + name);
        }
        return static_cast<uint32_t>(parsed);
    }

    std::string jsonEscape(std::string_view value)
    {
        std::ostringstream output;
        for (const char character : value)
        {
            switch (character)
            {
            case '\\':
                output << "\\\\";
                break;
            case '"':
                output << "\\\"";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                output << character;
                break;
            }
        }
        return output.str();
    }

    std::string hexValue(uint64_t value, unsigned width)
    {
        std::ostringstream output;
        output << "0x" << std::hex << std::setfill('0')
               << std::setw(static_cast<int>(width)) << value;
        return output.str();
    }

    void printDistribution(std::string_view indent,
                           const Distribution &distribution)
    {
        std::cout << indent
                  << "\"min\":" << distribution.minimum << ","
                  << "\"median\":" << distribution.median << ","
                  << "\"mean\":" << distribution.mean << ","
                  << "\"p95\":" << distribution.p95 << ","
                  << "\"max\":" << distribution.maximum;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 6)
    {
        std::cerr
            << "usage: vu_native_emitter_spike FIXTURE_DIRECTORY "
               "[TIMED_PAIRS] [SAMPLES] [COMPILE_SAMPLES] [SIDE_EXITS]\n";
        return 2;
    }

    try
    {
        const std::filesystem::path fixtureDirectory = argv[1];
        const uint32_t timedPairs =
            argc >= 3 ? parseBounded(argv[2], 1u, "timed pair count")
                      : kDefaultTimedPairs;
        const uint32_t samples =
            argc >= 4 ? parseBounded(argv[3], 1u, "sample count")
                      : kDefaultSamples;
        const uint32_t compileSamples =
            argc >= 5 ? parseBounded(argv[4], 1u,
                                     "compile sample count")
                      : kDefaultCompileSamples;
        const uint32_t sideExits =
            argc >= 6 ? parseBounded(argv[5], 1u, "side-exit count")
                      : kDefaultSideExits;

        const CapturedBlock block = captureBlock(fixtureDirectory);
        const uint32_t mulqCount = static_cast<uint32_t>(
            std::count_if(block.upper.begin(), block.upper.end(), isMulq));
        const uint32_t nopCount = static_cast<uint32_t>(
            std::count_if(
                block.upper.begin(), block.upper.end(), isUpperNop));

        const auto helperColdBegin = Clock::now();
        auto helperBlock =
            std::make_unique<NativeBlock>(block.upper, false);
        helperBlock->finalize();
        const auto helperColdEnd = Clock::now();
        const double helperColdMicroseconds =
            std::chrono::duration<double, std::micro>(
                helperColdEnd - helperColdBegin).count();

        const auto inlineColdBegin = Clock::now();
        auto inlineBlock =
            std::make_unique<NativeBlock>(block.upper, true);
        inlineBlock->finalize();
        const auto inlineColdEnd = Clock::now();
        const double inlineColdMicroseconds =
            std::chrono::duration<double, std::micro>(
                inlineColdEnd - inlineColdBegin).count();

        auto protectionProbe =
            std::make_unique<NativeBlock>(block.upper, true);
        const MappingPermissions writableMapping =
            mappingForAddress(protectionProbe->getCode<const uint8_t *>());
        protectionProbe->finalize();
        const MappingPermissions executableMapping =
            mappingForAddress(protectionProbe->getCode<const uint8_t *>());
        const bool wxOk =
            permissionsAreRwNotExecutable(writableMapping.permissions) &&
            permissionsAreRxNotWritable(executableMapping.permissions);
        if (!wxOk)
            throw std::runtime_error("native mapping violated W^X");

        const Distribution helperCompile = summarize(
            measureCompilation(block.upper, false, compileSamples));
        const Distribution inlineCompile = summarize(
            measureCompilation(block.upper, true, compileSamples));

        VU1Interpreter abiInterpreter;
        VU1NativeEmitterSpikeAccess::prepare(
            abiInterpreter, block.initialState);
        SpikeContext abiContext = makeContext(abiInterpreter);
        std::array<uint64_t, 6> preserved{};
        const std::array<uint64_t, 6> expectedPreserved{
            0x1122334455667788ull,
            0x2233445566778899ull,
            0x33445566778899aaull,
            0x445566778899aabbull,
            0x5566778899aabbccull,
            0x66778899aabbccddull,
        };
        abiContext.remainingPairs = 0u;
        const uint32_t abiResult = vuNativeSpikeProbeCalleeSaved(
            inlineBlock->function(), &abiContext, preserved.data());
        const bool calleeSavedPreserved =
            abiResult == 0u && preserved == expectedPreserved;
        if (!calleeSavedPreserved)
            throw std::runtime_error("native block corrupted callee-saved registers");

        const uint32_t hostMxcsrBefore = _mm_getcsr();
        abiContext.remainingPairs = 1u;
        if (inlineBlock->function()(&abiContext) != 1u)
            throw std::runtime_error("MXCSR probe failed to execute");
        const uint32_t hostMxcsrAfter = _mm_getcsr();
        const bool mxcsrRestored = hostMxcsrBefore == hostMxcsrAfter;
        if (!mxcsrRestored)
            throw std::runtime_error("native block did not restore MXCSR");

        // Touch each path before collecting samples.
        {
            VU1Interpreter warmupInterpreter;
            VU1NativeEmitterSpikeAccess::prepare(
                warmupInterpreter, block.initialState);
            runDirect(warmupInterpreter, block.upper,
                      static_cast<uint32_t>(block.upper.size() * 100u));
        }
        {
            VU1Interpreter warmupInterpreter;
            VU1NativeEmitterSpikeAccess::prepare(
                warmupInterpreter, block.initialState);
            SpikeContext context = makeContext(warmupInterpreter);
            context.remainingPairs =
                static_cast<uint32_t>(block.upper.size() * 100u);
            helperBlock->function()(&context);
        }
        {
            VU1Interpreter warmupInterpreter;
            VU1NativeEmitterSpikeAccess::prepare(
                warmupInterpreter, block.initialState);
            SpikeContext context = makeContext(warmupInterpreter);
            context.remainingPairs =
                static_cast<uint32_t>(block.upper.size() * 100u);
            inlineBlock->function()(&context);
        }

        const TimedPath direct =
            measureDirect(block, timedPairs, samples);
        const TimedPath helper =
            measureNative(block, helperBlock->function(),
                          timedPairs, samples);
        const TimedPath inlined =
            measureNative(block, inlineBlock->function(),
                          timedPairs, samples);
        if (direct.stateHash != helper.stateHash ||
            direct.stateHash != inlined.stateHash)
        {
            throw std::runtime_error(
                "helper or inlined native state differs from direct helpers");
        }

        const uint32_t sideSamples = std::min<uint32_t>(samples, 7u);
        const SideExitPath helperSide = measureSideExits(
            block, helperBlock->function(), sideExits, sideSamples);
        const SideExitPath inlineSide = measureSideExits(
            block, inlineBlock->function(), sideExits, sideSamples);
        VU1Interpreter sideReference;
        VU1NativeEmitterSpikeAccess::prepare(
            sideReference, block.initialState);
        runDirect(sideReference, block.upper, sideExits);
        VU1NativeEmitterSpikeAccess::flushFmac(sideReference);
        const uint64_t sideReferenceHash = hashState(sideReference.state());
        if (helperSide.stateHash != sideReferenceHash ||
            inlineSide.stateHash != sideReferenceHash ||
            helperSide.nextPair != sideExits % block.upper.size() ||
            inlineSide.nextPair != sideExits % block.upper.size())
        {
            throw std::runtime_error(
                "side-exit/resume state differs from direct helpers");
        }

        const double helperSpeedup =
            helper.pairsPerSecond.median /
            direct.pairsPerSecond.median;
        const double inlineSpeedup =
            inlined.pairsPerSecond.median /
            direct.pairsPerSecond.median;
        const double inlineVsHelper =
            inlined.pairsPerSecond.median /
            helper.pairsPerSecond.median;
        const double inlineWarmNanosecondsPerPair =
            1.0e9 / inlined.pairsPerSecond.median;
        const double inlineSideExitIncrement =
            inlineSide.nanosecondsPerExit.median -
            inlineWarmNanosecondsPerPair;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "{\n"
                  << "  \"schema_version\":1,\n"
                  << "  \"fixture\":\""
                  << jsonEscape(fixtureDirectory.string()) << "\",\n"
                  << "  \"block\":{\"trace_start\":"
                  << kBlockTraceStart
                  << ",\"pairs\":" << block.upper.size()
                  << ",\"start_pc\":\""
                  << hexValue(block.pcs.front(), 4u)
                  << "\",\"end_pc\":\""
                  << hexValue(block.pcs.back(), 4u)
                  << "\",\"mulq_pairs\":" << mulqCount
                  << ",\"nop_pairs\":" << nopCount << "},\n"
                  << "  \"emitter\":{\"name\":\"Xbyak\","
                  << "\"header_version\":\""
                  << hexValue(Xbyak::VERSION, 4u)
                  << "\",\"helper_bytes\":" << helperBlock->getSize()
                  << ",\"inline_bytes\":" << inlineBlock->getSize()
                  << "},\n"
                  << "  \"security\":{\"rw_permissions\":\""
                  << writableMapping.permissions
                  << "\",\"rx_permissions\":\""
                  << executableMapping.permissions
                  << "\",\"wx_ok\":" << (wxOk ? "true" : "false")
                  << ",\"instruction_cache_clear\":true},\n"
                  << "  \"abi\":{\"sysv_callee_saved_preserved\":"
                  << (calleeSavedPreserved ? "true" : "false")
                  << ",\"mxcsr_host\":\""
                  << hexValue(hostMxcsrBefore, 8u)
                  << "\",\"mxcsr_vu\":\""
                  << hexValue(abiContext.vuMxcsr, 8u)
                  << "\",\"mxcsr_restored\":"
                  << (mxcsrRestored ? "true" : "false") << "},\n"
                  << "  \"compile_us\":{\"helper_cold\":"
                  << helperColdMicroseconds
                  << ",\"inline_cold\":" << inlineColdMicroseconds
                  << ",\"helper\":{";
        printDistribution("", helperCompile);
        std::cout << "},\"inline\":{";
        printDistribution("", inlineCompile);
        std::cout << "}},\n"
                  << "  \"warm_pairs_per_second\":{\"timed_pairs\":"
                  << timedPairs << ",\"samples\":" << samples
                  << ",\"direct\":{";
        printDistribution("", direct.pairsPerSecond);
        std::cout << "},\"helper_native\":{";
        printDistribution("", helper.pairsPerSecond);
        std::cout << "},\"inline_native\":{";
        printDistribution("", inlined.pairsPerSecond);
        std::cout << "},\"helper_over_direct\":"
                  << helperSpeedup
                  << ",\"inline_over_direct\":" << inlineSpeedup
                  << ",\"inline_over_helper\":" << inlineVsHelper
                  << "},\n"
                  << "  \"side_exit\":{\"exits\":" << sideExits
                  << ",\"samples\":" << sideSamples
                  << ",\"helper_ns_per_exit\":{";
        printDistribution("", helperSide.nanosecondsPerExit);
        std::cout << "},\"inline_ns_per_exit\":{";
        printDistribution("", inlineSide.nanosecondsPerExit);
        std::cout << "},\"inline_increment_over_warm_pair_ns\":"
                  << inlineSideExitIncrement
                  << ",\"resume_next_pair\":" << inlineSide.nextPair
                  << "},\n"
                  << "  \"correctness\":{\"state_hash\":\""
                  << hexValue(direct.stateHash, 16u)
                  << "\",\"full_equal\":true,\"side_exit_hash\":\""
                  << hexValue(sideReferenceHash, 16u)
                  << "\",\"side_exit_equal\":true}\n"
                  << "}\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << "vu_native_emitter_spike: "
                  << error.what() << '\n';
        return 1;
    }

    return 0;
}
