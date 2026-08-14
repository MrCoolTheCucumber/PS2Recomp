#include "MiniTest.h"

#include "ps2_runtime.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu1_command_stream.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{
    using namespace std::chrono_literals;

    template <typename Predicate>
    bool waitFor(
        std::condition_variable &cv,
        std::mutex &mutex,
        Predicate predicate)
    {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(
            lock, 2s, std::move(predicate));
    }

    template <typename Predicate>
    bool waitUntil(Predicate predicate)
    {
        const auto deadline =
            std::chrono::steady_clock::now() + 2s;
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (predicate())
                return true;
            std::this_thread::sleep_for(1ms);
        }
        return predicate();
    }

    template <typename Operation>
    std::string captureException(Operation operation)
    {
        try
        {
            operation();
        }
        catch (const std::exception &error)
        {
            return error.what();
        }
        catch (...)
        {
            return "non-standard exception";
        }
        return {};
    }

    class Vu1WorkerGate
    {
    public:
        explicit Vu1WorkerGate(
            uint64_t blockedTicket = 1u,
            bool throwAfterRelease = false)
            : m_blockedTicket(blockedTicket),
              m_throwAfterRelease(throwAfterRelease)
        {
        }

        void beforeProcess(
            const Vu1Command &,
            uint64_t ticket)
        {
            if (ticket != m_blockedTicket)
                return;
            std::unique_lock<std::mutex> lock(m_mutex);
            m_entered = true;
            m_cv.notify_all();
            m_cv.wait(
                lock,
                [this]()
                {
                    return m_released;
                });
            if (m_throwAfterRelease)
            {
                throw std::runtime_error(
                    "injected VU1 owner failure");
            }
        }

        [[nodiscard]] bool waitUntilEntered()
        {
            return waitFor(
                m_cv,
                m_mutex,
                [this]()
                {
                    return m_entered;
                });
        }

        void release()
        {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_released = true;
            }
            m_cv.notify_all();
        }

    private:
        std::mutex m_mutex;
        std::condition_variable m_cv;
        uint64_t m_blockedTicket = 1u;
        bool m_throwAfterRelease = false;
        bool m_entered = false;
        bool m_released = false;
    };

    constexpr uint32_t kVuUpperNop = 0x000002ffu;
    constexpr uint32_t kVuUpperEnd = 1u << 30u;

    uint32_t makeVuIaddiu(
        uint8_t target, uint8_t source, int16_t immediate)
    {
        return (0x08u << 25u) |
               (static_cast<uint32_t>(target & 0x0fu) << 16u) |
               (static_cast<uint32_t>(source & 0x0fu) << 11u) |
               (static_cast<uint32_t>(immediate) & 0x07ffu);
    }

    uint32_t makeVuSq(
        uint8_t destinationMask,
        uint8_t sourceVf,
        uint8_t baseVi,
        int16_t immediate)
    {
        return (0x01u << 25u) |
               (static_cast<uint32_t>(destinationMask & 0x0fu) << 21u) |
               (static_cast<uint32_t>(baseVi & 0x0fu) << 16u) |
               (static_cast<uint32_t>(sourceVf & 0x1fu) << 11u) |
               (static_cast<uint32_t>(immediate) & 0x07ffu);
    }

    uint32_t makeVuLowerSpecial(
        uint8_t specialOperation, uint8_t sourceVi)
    {
        return 0x80000000u |
               (static_cast<uint32_t>(sourceVi & 0x0fu) << 11u) |
               (static_cast<uint32_t>(specialOperation & 0x7cu) << 4u) |
               static_cast<uint32_t>(specialOperation & 0x03u) |
               0x3cu;
    }

    std::array<uint8_t, 8u> instructionPair(
        uint32_t lower, uint32_t upper)
    {
        std::array<uint8_t, 8u> bytes{};
        std::memcpy(bytes.data(), &lower, sizeof(lower));
        std::memcpy(
            bytes.data() + sizeof(lower),
            &upper, sizeof(upper));
        return bytes;
    }

    void writeRuntimeVu1InstructionPair(
        PS2Runtime &runtime,
        uint32_t pc,
        uint32_t lower,
        uint32_t upper)
    {
        runtime.memory().write32(
            PS2_VU1_CODE_BASE + pc, lower);
        runtime.memory().write32(
            PS2_VU1_CODE_BASE + pc + 4u, upper);
    }

    void writeRuntimeVu1Qword(
        PS2Runtime &runtime,
        uint32_t offset,
        const std::array<uint8_t, 16u> &bytes)
    {
        for (uint32_t word = 0u; word < 4u; ++word)
        {
            uint32_t value = 0u;
            std::memcpy(
                &value,
                bytes.data() + word * sizeof(value),
                sizeof(value));
            runtime.memory().write32(
                PS2_VU1_DATA_BASE + offset +
                    word * sizeof(value),
                value);
        }
    }

    uint32_t makeVifCommand(
        uint8_t opcode, uint8_t count, uint16_t immediate)
    {
        return (static_cast<uint32_t>(opcode) << 24u) |
               (static_cast<uint32_t>(count) << 16u) |
               immediate;
    }

    struct Fixture
    {
        VuUnit unit{VuUnitId::Vu1};
        std::vector<uint8_t> code =
            std::vector<uint8_t>(PS2_VU1_CODE_SIZE, 0u);
        std::vector<uint8_t> data =
            std::vector<uint8_t>(PS2_VU1_DATA_SIZE, 0u);
        Vu1CommandProcessor processor{
            unit,
            {
                .captureCommandDigests = true,
                .captureArchitecturalStateHashes = true,
            }};
        InlineVu1Executor executor{processor};

        Fixture()
        {
            processor.bindMemory(
                code.data(), static_cast<uint32_t>(code.size()),
                data.data(), static_cast<uint32_t>(data.size()),
                1u);
        }

        Vu1CommandResult submit(
            Vu1CommandPayload payload,
            uint64_t tick = 0u)
        {
            return executor.submit(std::move(payload), tick);
        }

        Vu1CommandResult advance(
            Vu1AdvanceSliceCommand command,
            uint64_t tick = 0u)
        {
            return executor.submitAdvanceSlice(command, tick);
        }

        Vu1CommandResult unpack(
            Vu1DecodedUnpackCommand command,
            uint64_t tick = 0u)
        {
            return executor.submitDecodedUnpack(
                std::move(command), tick);
        }

        Vu1CommandResult updateVif(
            Vu1VifStateUpdateCommand command,
            uint64_t tick = 0u)
        {
            return executor.submitVifStateUpdate(command, tick);
        }
    };

    const Vu1SliceResult *sliceResult(
        const Vu1CommandResult &result)
    {
        return std::get_if<Vu1SliceResult>(&result.payload);
    }
}

void register_ps2_vu1_command_stream_tests()
{
    MiniTest::Case("PS2 VU1 command stream", [](TestCase &tc)
    {
        tc.Run("inline slices match the direct semantic entry after every budget", [](TestCase &t)
        {
            Fixture inlineFixture;
            const auto firstPair = instructionPair(
                makeVuIaddiu(1u, 0u, 7), kVuUpperNop);
            const auto secondPair = instructionPair(
                makeVuIaddiu(2u, 1u, 5), kVuUpperEnd);
            std::vector<uint8_t> program;
            program.insert(
                program.end(), firstPair.begin(), firstPair.end());
            program.insert(
                program.end(), secondPair.begin(), secondPair.end());
            t.IsTrue(
                inlineFixture.submit(Vu1MicroMemoryWriteCommand{
                    .offset = 0u,
                    .bytes = program,
                }).disposition == Vu1CommandDisposition::Completed,
                "the inline owner should accept an owned program upload");

            VuUnit directUnit(VuUnitId::Vu1);
            std::vector<uint8_t> directCode = inlineFixture.code;
            std::vector<uint8_t> directData = inlineFixture.data;
            directUnit.start(0u, 0x123u, 0x2abu);
            t.IsTrue(
                inlineFixture.submit(Vu1MscalCommand{
                    .startPc = 0u,
                    .top = 0x123u,
                    .itop = 0x2abu,
                }).disposition == Vu1CommandDisposition::Completed,
                "MSCAL should enter through the processor");

            for (uint32_t budget : {1u, 1u, 1u})
            {
                VuTransactionalSideEffectSink directEffects;
                VuExecutionContext directContext{
                    .state = directUnit.state(),
                    .code = directCode.data(),
                    .codeSize = static_cast<uint32_t>(directCode.size()),
                    .data = directData.data(),
                    .dataSize = static_cast<uint32_t>(directData.size()),
                    .sideEffects = directEffects,
                    .codeUnit = VuUnitId::Vu1,
                    .memoryIdentity = reinterpret_cast<uintptr_t>(&directUnit),
                    .codeIdentity = reinterpret_cast<uintptr_t>(directCode.data()),
                    .codeGeneration = 2u,
                    .trackedCode = true,
                    .enableInstrumentation = false,
                };
                const VuRunResult direct =
                    directUnit.advance(directContext, budget);
                const Vu1CommandResult inlineResult =
                    inlineFixture.advance(Vu1AdvanceSliceCommand{
                        .maximumCycles = budget,
                        .captureState = true,
                    });
                Vu1Command expectedCommand{
                    .identity = inlineResult.identity,
                    .type = Vu1CommandType::AdvanceSlice,
                    .payload = Vu1AdvanceSliceCommand{
                        .maximumCycles = budget,
                        .captureState = true,
                    },
                };
                expectedCommand.payloadSize =
                    vu1CommandPayloadSize(expectedCommand.payload);
                t.IsTrue(
                    inlineResult.digest ==
                        vu1CommandDigest(expectedCommand),
                    "the typed slice route should retain the canonical digest");
                const Vu1SliceResult *const slice =
                    sliceResult(inlineResult);
                t.IsTrue(slice != nullptr,
                         "a slice command should return a typed result");
                if (!slice)
                    continue;
                t.Equals(slice->run.executedCycles, direct.executedCycles,
                         "each seam slice should execute the direct cycle count");
                t.IsTrue(slice->run.reason == direct.reason,
                         "each seam slice should retain the direct exit reason");
                std::string difference;
                t.IsTrue(
                    static_cast<bool>(slice->state) &&
                        vuExecutionStatesEqual(
                            *slice->state, directUnit.state(), &difference),
                    "each seam slice should retain exact architectural state");
                t.Equals(inlineFixture.data, directData,
                         "each seam slice should retain exact data memory");
            }
        });

        tc.Run("XGKICK is returned as owned ordered slice output", [](TestCase &t)
        {
            Fixture fixture;
            const auto pair = instructionPair(
                makeVuLowerSpecial(0x6cu, 0u), kVuUpperEnd);
            fixture.submit(Vu1MicroMemoryWriteCommand{
                .offset = 0u,
                .bytes = std::vector<uint8_t>(
                    pair.begin(), pair.end()),
            });

            std::vector<uint8_t> packet(32u, 0u);
            const uint64_t tag =
                1u | (1ull << 15u) | (2ull << 58u);
            std::memcpy(packet.data(), &tag, sizeof(tag));
            for (uint32_t index = 0u; index < 16u; ++index)
                packet[16u + index] = static_cast<uint8_t>(0xa0u + index);
            fixture.submit(Vu1DataMemoryWriteCommand{
                .offset = 0u,
                .bytes = packet,
            });
            fixture.submit(Vu1MscalCommand{});
            const Vu1CommandResult result =
                fixture.advance(Vu1AdvanceSliceCommand{
                    .maximumCycles = 2u,
                    .captureState = true,
                });
            const Vu1SliceResult *const slice = sliceResult(result);
            t.IsTrue(slice != nullptr,
                     "XGKICK execution should return a slice result");
            if (!slice)
                return;
            t.Equals(slice->path1Packets.size(), size_t{1u},
                     "one completed XGKICK should return one packet");
            if (!slice->path1Packets.empty())
            {
                t.Equals(slice->path1Packets[0].bytes, packet,
                         "the packet result should own the live VU bytes");
                fixture.data[16u] ^= 0xffu;
                t.Equals(slice->path1Packets[0].bytes, packet,
                         "later VU mutation must not change the result");
            }
            t.IsTrue(slice->vifCanResume,
                     "an ended slice should publish the VIF resume condition");
        });

        tc.Run("decoded UNPACK owns source bytes and applies VIF state", [](TestCase &t)
        {
            Fixture fixture;
            Vu1VifState vif{
                .cycle = 0x0202u,
                .mode = 1u,
                .row = {10u, 20u, 30u, 40u},
            };
            const Vu1CommandResult vifResult =
                fixture.updateVif(Vu1VifStateUpdateCommand{vif});
            Vu1Command expectedVifCommand{
                .identity = vifResult.identity,
                .type = Vu1CommandType::VifStateUpdate,
                .payload = Vu1VifStateUpdateCommand{vif},
            };
            expectedVifCommand.payloadSize =
                vu1CommandPayloadSize(expectedVifCommand.payload);
            t.IsTrue(
                vifResult.digest ==
                    vu1CommandDigest(expectedVifCommand),
                "the typed VIF state route should retain the canonical digest");
            std::vector<uint8_t> source(32u);
            const std::array<uint32_t, 8u> words{
                1u, 2u, 3u, 4u,
                5u, 6u, 7u, 8u,
            };
            std::memcpy(source.data(), words.data(), source.size());
            const std::vector<uint8_t> owned = source;
            Vu1DecodedUnpackCommand unpack{
                .immediate = 3u,
                .vectorLength = 0u,
                .componentCount = 4u,
                .writeVectorCount = 2u,
                .sourceVectorCount = 2u,
                .sourceWordAlignment = 4u,
                .bytes = std::move(source),
            };
            const Vu1DecodedUnpackCommand expectedUnpack = unpack;
            const Vu1CommandResult result =
                fixture.unpack(std::move(unpack));
            Vu1Command expectedUnpackCommand{
                .identity = result.identity,
                .type = Vu1CommandType::DecodedUnpack,
                .payload = expectedUnpack,
            };
            expectedUnpackCommand.payloadSize =
                vu1CommandPayloadSize(expectedUnpackCommand.payload);
            t.IsTrue(
                result.digest ==
                    vu1CommandDigest(expectedUnpackCommand),
                "the typed UNPACK route should retain the canonical digest");
            t.IsTrue(
                result.disposition == Vu1CommandDisposition::Completed,
                "a complete decoded UNPACK should be accepted");
            std::array<uint32_t, 8u> actual{};
            std::memcpy(
                actual.data(), fixture.data.data() + 3u * 16u,
                sizeof(actual));
            const std::array<uint32_t, 8u> expected{
                11u, 22u, 33u, 44u,
                15u, 26u, 37u, 48u,
            };
            t.Equals(actual, expected,
                     "UNPACK should apply row-add mode in VIF order");
            t.IsTrue(owned != std::vector<uint8_t>{},
                     "the fixture should retain an independent source oracle");
        });

        tc.Run("digests identities and generation rejection are deterministic", [](TestCase &t)
        {
            t.IsTrue(
                sizeof(Vu1Command) <= 128u &&
                    sizeof(Vu1CommandResult) <= 256u,
                "hot command and result envelopes should remain bounded");
            VuUnit unit(VuUnitId::Vu1);
            std::vector<uint8_t> code(PS2_VU1_CODE_SIZE);
            std::vector<uint8_t> data(PS2_VU1_DATA_SIZE);
            Vu1CommandProcessor processor(
                unit, {.captureCommandDigests = true});
            processor.bindMemory(
                code.data(), static_cast<uint32_t>(code.size()),
                data.data(), static_cast<uint32_t>(data.size()), 7u);

            Vu1Command command{
                .identity = {
                    .sequence = 1u,
                    .generation = 1u,
                    .guestTick = 99u,
                },
                .type = Vu1CommandType::DataMemoryWrite,
                .payload = Vu1DataMemoryWriteCommand{
                    .offset = 4u,
                    .bytes = {1u, 2u, 3u, 4u},
                },
            };
            command.payloadSize =
                vu1CommandPayloadSize(command.payload);
            const Vu1CommandDigest expected =
                vu1CommandDigest(command);
            const Vu1CommandResult completed =
                processor.process(command);
            t.IsTrue(
                completed.disposition ==
                    Vu1CommandDisposition::Completed,
                "the first ordered command should complete");
            t.IsTrue(completed.digest == expected,
                     "the processor should publish the canonical digest");
            t.Equals(
                vu1CommandDigestHash(completed.digest),
                vu1CommandDigestHash(expected),
                "equal command digests should have equal stream hashes");

            command.identity.sequence = 2u;
            command.identity.generation = 0u;
            t.IsTrue(
                processor.process(command).disposition ==
                    Vu1CommandDisposition::StaleGeneration,
                "an old generation should be rejected");
            command.identity.generation = 2u;
            t.IsTrue(
                processor.process(command).disposition ==
                    Vu1CommandDisposition::FutureGeneration,
                "an unpublished generation should be rejected");
            command.identity.generation = 1u;
            command.identity.sequence = 3u;
            t.IsTrue(
                processor.process(command).disposition ==
                    Vu1CommandDisposition::OutOfOrder,
                "a sequence gap should be rejected");
        });

        tc.Run("reset remains ordered before owner memory is bound", [](TestCase &t)
        {
            VuUnit unit(VuUnitId::Vu1);
            Vu1CommandProcessor processor(unit);
            InlineVu1Executor executor(processor);
            const Vu1CommandResult reset =
                executor.submit(Vu1ResetCommand{});
            t.IsTrue(
                reset.disposition ==
                    Vu1CommandDisposition::Completed,
                "runtime timing reset should not require initialized VU memory");
            t.Equals(processor.generation(), uint64_t{2u},
                     "an unbound reset should still advance owner generation");
            t.Equals(processor.nextSequence(), uint64_t{2u},
                     "an unbound reset should consume exactly one sequence");
            const Vu1CommandResult snapshotResult =
                executor.submit(Vu1SnapshotCommand{});
            const auto *const snapshot =
                std::get_if<Vu1SnapshotResult>(
                    &snapshotResult.payload);
            t.IsTrue(
                snapshot && snapshot->snapshot &&
                    snapshot->snapshot->microMemory.empty() &&
                    snapshot->snapshot->dataMemory.empty(),
                "pre-initialization diagnostics should snapshot unbound state");
        });

        tc.Run("mapped writes MPG and UNPACK enter one ordered owner stream", [](TestCase &t)
        {
            PS2Memory memory;
            t.IsTrue(memory.initialize(),
                     "owner-routing memory should initialize");
            VuUnit unit(VuUnitId::Vu1);
            Vu1CommandProcessor processor(
                unit, {.captureCommandDigests = true});
            processor.bindMemory(
                memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                memory.getVU1CodeGeneration());
            InlineVu1Executor executor(processor);
            memory.setVu1CommandCallback(
                [&](Vu1CommandPayload payload)
                {
                    return executor.submit(std::move(payload), 17u);
                });
            memory.setVu1DecodedUnpackCallback(
                [&](Vu1DecodedUnpackCommand command)
                {
                    return executor.submitDecodedUnpack(
                        std::move(command), 17u);
                });
            memory.setVu1VifStateUpdateCallback(
                [&](Vu1VifStateUpdateCommand command)
                {
                    return executor.submitVifStateUpdate(
                        command, 17u);
                });

            const uint64_t initialCodeGeneration =
                processor.codeGeneration();
            memory.write32(
                PS2_VU1_CODE_BASE + 4u, 0x44332211u);
            memory.writeMasked32(
                PS2_VU1_DATA_BASE + 8u,
                0xaabbccddu, 0x6u);
            t.Equals(processor.nextSequence(), uint64_t{3u},
                     "mapped code and data writes should consume two commands");
            t.Equals(processor.codeGeneration(),
                     initialCodeGeneration + 1u,
                     "only the mapped micro write should advance code generation");
            t.Equals(memory.getVU1CodeGeneration(),
                     processor.codeGeneration(),
                     "the EE generation mirror should publish the owner result");
            t.Equals(memory.getVU1Code()[4u], uint8_t{0x11u},
                     "mapped micro bytes should be applied by the owner");
            t.Equals(memory.getVU1Data()[9u], uint8_t{0xccu},
                     "masked data writes should retain their byte offset");
            t.Equals(memory.getVU1Data()[10u], uint8_t{0xbbu},
                     "masked data writes should retain contiguous enabled bytes");

            std::vector<uint8_t> stream;
            const auto append =
                [&](const void *source, size_t size)
                {
                    const auto *const bytes =
                        static_cast<const uint8_t *>(source);
                    stream.insert(stream.end(), bytes, bytes + size);
                };
            const uint32_t stcycl =
                makeVifCommand(0x01u, 0u, 0x0101u);
            const uint32_t mpg =
                makeVifCommand(0x4au, 1u, 2u);
            const auto uploadedPair =
                instructionPair(0x10203040u, kVuUpperNop);
            const uint32_t unpack =
                makeVifCommand(0x6cu, 1u, 3u);
            const std::array<uint32_t, 4u> unpackWords{
                1u, 2u, 3u, 4u};
            append(&stcycl, sizeof(stcycl));
            append(&mpg, sizeof(mpg));
            append(uploadedPair.data(), uploadedPair.size());
            append(&unpack, sizeof(unpack));
            append(unpackWords.data(), sizeof(unpackWords));
            memory.processVIF1Data(
                stream.data(), static_cast<uint32_t>(stream.size()));

            t.Equals(processor.nextSequence(), uint64_t{6u},
                     "STCYCL MPG and decoded UNPACK should add three commands");
            t.Equals(processor.codeGeneration(),
                     initialCodeGeneration + 2u,
                     "MPG should publish one additional code generation");
            t.Equals(processor.vifState().cycle, uint32_t{0x0101u},
                     "VIF state should be ordered before decoded UNPACK");
            t.IsTrue(
                std::memcmp(
                    memory.getVU1Code() + 16u,
                    uploadedPair.data(), uploadedPair.size()) == 0,
                "MPG bytes should be owned and applied at their wrapped address");
            std::array<uint32_t, 4u> actual{};
            std::memcpy(
                actual.data(), memory.getVU1Data() + 48u,
                sizeof(actual));
            t.Equals(actual, unpackWords,
                     "decoded UNPACK should mutate owner data memory");
        });

        tc.Run("decoded owner UNPACK matches the legacy parser matrix", [](TestCase &t)
        {
            PS2Memory legacy;
            PS2Memory owned;
            t.IsTrue(legacy.initialize() && owned.initialize(),
                     "UNPACK differential memories should initialize");
            VuUnit unit(VuUnitId::Vu1);
            Vu1CommandProcessor processor(unit);
            processor.bindMemory(
                owned.getVU1Code(), PS2_VU1_CODE_SIZE,
                owned.getVU1Data(), PS2_VU1_DATA_SIZE,
                owned.getVU1CodeGeneration());
            InlineVu1Executor executor(processor);
            owned.setVu1CommandCallback(
                [&](Vu1CommandPayload payload)
                {
                    return executor.submit(std::move(payload));
                });
            owned.setVu1DecodedUnpackCallback(
                [&](Vu1DecodedUnpackCommand command)
                {
                    return executor.submitDecodedUnpack(
                        std::move(command));
                });
            owned.setVu1VifStateUpdateCallback(
                [&](Vu1VifStateUpdateCommand command)
                {
                    return executor.submitVifStateUpdate(command);
                });

            constexpr std::array<std::pair<uint8_t, uint8_t>, 6u>
                cycles{{
                    {1u, 1u}, {2u, 2u}, {1u, 3u},
                    {3u, 1u}, {4u, 4u}, {2u, 4u},
                }};
            constexpr std::array<uint32_t, 4u> row{
                0x10u, 0x20u, 0x30u, 0x40u};
            constexpr std::array<uint32_t, 4u> column{
                0x100u, 0x200u, 0x300u, 0x400u};
            constexpr uint32_t mask = 0xe4e4e4e4u;
            const auto append = [](
                std::vector<uint8_t> &output,
                const void *source, size_t size)
            {
                const auto *const bytes =
                    static_cast<const uint8_t *>(source);
                output.insert(output.end(), bytes, bytes + size);
            };

            for (uint8_t vn = 0u; vn < 4u; ++vn)
            {
                for (uint8_t vl = 0u; vl < 4u; ++vl)
                {
                    const uint32_t components = vn + 1u;
                    const uint32_t bitsPerComponent =
                        vl == 0u ? 32u :
                        vl == 1u ? 16u :
                        vl == 2u ? 8u :
                        vn == 3u ? 4u : 16u;
                    const uint32_t bytesPerVector =
                        ((vl == 3u && vn == 3u)
                             ? 16u
                             : components * bitsPerComponent) /
                        8u;
                    for (bool maskEnabled : {false, true})
                    {
                        for (uint8_t mode = 0u; mode < 4u; ++mode)
                        {
                            for (const auto [cl, wl] : cycles)
                            {
                                constexpr uint32_t writeCount = 5u;
                                uint32_t sourceCount = writeCount;
                                if (cl < wl)
                                {
                                    sourceCount =
                                        (writeCount / wl) * cl +
                                        std::min<uint32_t>(
                                            writeCount % wl, cl);
                                }
                                const uint32_t payloadBytes =
                                    (sourceCount * bytesPerVector + 3u) &
                                    ~3u;
                                for (uint8_t alignment = 0u;
                                     alignment < 4u; ++alignment)
                                {
                                    for (uint16_t flags :
                                         {uint16_t{0u},
                                          uint16_t{0x4000u},
                                          uint16_t{0x8000u},
                                          uint16_t{0xc000u}})
                                    {
                                        for (uint32_t index = 0u;
                                             index < PS2_VU1_DATA_SIZE;
                                             ++index)
                                        {
                                            const uint8_t value =
                                                static_cast<uint8_t>(
                                                    index * 29u + 7u);
                                            legacy.getVU1Data()[index] = value;
                                            owned.getVU1Data()[index] = value;
                                        }

                                        std::vector<uint8_t> stream;
                                        const std::array<uint32_t, 7u>
                                            commands{
                                                makeVifCommand(
                                                    0x03u, 0u, 7u),
                                                makeVifCommand(
                                                    0x02u, 0u, 5u),
                                                makeVifCommand(
                                                    0x01u, 0u,
                                                    static_cast<uint16_t>(
                                                        (wl << 8u) | cl)),
                                                makeVifCommand(
                                                    0x05u, 0u, mode),
                                                makeVifCommand(
                                                    0x20u, 0u, 0u),
                                                makeVifCommand(
                                                    0x30u, 0u, 0u),
                                                makeVifCommand(
                                                    0x31u, 0u, 0u),
                                            };
                                        append(stream, &commands[0], 4u);
                                        append(stream, &commands[1], 4u);
                                        append(stream, &commands[2], 4u);
                                        append(stream, &commands[3], 4u);
                                        append(stream, &commands[4], 4u);
                                        append(stream, &mask, sizeof(mask));
                                        append(stream, &commands[5], 4u);
                                        append(stream, row.data(), sizeof(row));
                                        append(stream, &commands[6], 4u);
                                        append(
                                            stream, column.data(),
                                            sizeof(column));
                                        const uint32_t nop = 0u;
                                        for (uint8_t index = 0u;
                                             index < alignment; ++index)
                                        {
                                            append(stream, &nop, sizeof(nop));
                                        }
                                        const uint8_t opcode =
                                            static_cast<uint8_t>(
                                                0x60u |
                                                (maskEnabled ? 0x10u : 0u) |
                                                (vn << 2u) | vl);
                                        const uint32_t unpack =
                                            makeVifCommand(
                                                opcode,
                                                static_cast<uint8_t>(
                                                    writeCount),
                                                static_cast<uint16_t>(
                                                    flags | 11u));
                                        append(stream, &unpack, sizeof(unpack));
                                        for (uint32_t index = 0u;
                                             index < payloadBytes; ++index)
                                        {
                                            stream.push_back(
                                                static_cast<uint8_t>(
                                                    index * 17u +
                                                    vn * 11u + vl * 3u));
                                        }

                                        legacy.processVIF1Data(
                                            stream.data(),
                                            static_cast<uint32_t>(
                                                stream.size()));
                                        owned.processVIF1Data(
                                            stream.data(),
                                            static_cast<uint32_t>(
                                                stream.size()));
                                        const bool memoryMatches =
                                            std::memcmp(
                                                legacy.getVU1Data(),
                                                owned.getVU1Data(),
                                                PS2_VU1_DATA_SIZE) == 0;
                                        const bool stateMatches =
                                            std::memcmp(
                                                legacy.vif1_regs.row,
                                                owned.vif1_regs.row,
                                                sizeof(row)) == 0 &&
                                            processor.vifState().cycle ==
                                                owned.vif1_regs.cycle &&
                                            processor.vifState().mode ==
                                                owned.vif1_regs.mode &&
                                            processor.vifState().mask ==
                                                owned.vif1_regs.mask &&
                                            processor.vifState().tops ==
                                                owned.vif1_regs.tops &&
                                            processor.vifState().column ==
                                                std::array<uint32_t, 4u>{
                                                    owned.vif1_regs.col[0],
                                                    owned.vif1_regs.col[1],
                                                    owned.vif1_regs.col[2],
                                                    owned.vif1_regs.col[3]} &&
                                            processor.vifState().row ==
                                                std::array<uint32_t, 4u>{
                                                    owned.vif1_regs.row[0],
                                                    owned.vif1_regs.row[1],
                                                    owned.vif1_regs.row[2],
                                                    owned.vif1_regs.row[3]};
                                        if (!memoryMatches || !stateMatches)
                                        {
                                            t.Fail(
                                                "first UNPACK divergence: vn=" +
                                                std::to_string(vn) +
                                                " vl=" + std::to_string(vl) +
                                                " mask=" +
                                                std::to_string(maskEnabled) +
                                                " mode=" +
                                                std::to_string(mode) +
                                                " cl=" + std::to_string(cl) +
                                                " wl=" + std::to_string(wl) +
                                                " alignment=" +
                                                std::to_string(alignment) +
                                                " flags=" +
                                                std::to_string(flags));
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        tc.Run("snapshot restore barrier and hashes cover complete owner state", [](TestCase &t)
        {
            Fixture fixture;
            fixture.submit(Vu1DataMemoryWriteCommand{
                .offset = 0x40u,
                .bytes = {0x11u, 0x22u, 0x33u, 0x44u},
            });
            fixture.submit(Vu1RegisterWriteCommand{
                .kind = Vu1RegisterKind::VectorInteger,
                .index = 3u,
                .words = {0x1234u, 0u, 0u, 0u},
            });
            const Vu1CommandResult snapshotResult =
                fixture.submit(Vu1SnapshotCommand{});
            const auto *const snapshot =
                std::get_if<Vu1SnapshotResult>(&snapshotResult.payload);
            t.IsTrue(snapshot != nullptr && snapshot->snapshot,
                     "snapshot should return an owned typed result");
            if (!snapshot || !snapshot->snapshot)
                return;
            const uint64_t originalHash =
                snapshot->architecturalStateHash;
            fixture.submit(Vu1DataMemoryWriteCommand{
                .offset = 0x40u,
                .bytes = {0xffu, 0xeeu, 0xddu, 0xccu},
            });
            const Vu1CommandResult changedBarrierResult =
                fixture.submit(Vu1BarrierCommand{});
            const auto *changedBarrier =
                std::get_if<Vu1BarrierResult>(
                    &changedBarrierResult.payload);
            t.IsTrue(
                changedBarrier &&
                    changedBarrier->architecturalStateHash != originalHash,
                "architectural hash should change with data memory");

            t.IsTrue(
                fixture.submit(Vu1RestoreCommand{
                    .snapshot = snapshot->snapshot,
                }).disposition == Vu1CommandDisposition::Completed,
                "restore should accept the complete owner snapshot");
            const Vu1CommandResult restoredResult =
                fixture.submit(Vu1BarrierCommand{});
            const auto *restored =
                std::get_if<Vu1BarrierResult>(&restoredResult.payload);
            t.IsTrue(
                restored &&
                    restored->architecturalStateHash == originalHash,
                "restore should reproduce the complete architectural hash");
            t.Equals(fixture.unit.state().vi[3], int32_t{0x1234},
                     "restore should reproduce register state");
        });

        tc.Run("threaded VU1 queue applies bounded slot and payload backpressure", [](TestCase &t)
        {
            {
                VuUnit unit(VuUnitId::Vu1);
                Vu1CommandProcessor processor(unit);
                const auto gate = std::make_shared<Vu1WorkerGate>();
                ThreadedVu1Executor executor(
                    processor,
                    ThreadedVu1ExecutorOptions{
                        .queueCapacity = 2u,
                        .payloadCapacityBytes = 4096u,
                        .beforeProcess =
                            [gate](const Vu1Command &command, uint64_t ticket)
                            {
                                gate->beforeProcess(command, ticket);
                            },
                    });

                Vu1CommandSubmission first =
                    executor.submitAsync(Vu1BarrierCommand{});
                t.IsTrue(gate->waitUntilEntered(),
                         "the first command should remain owner-active");
                Vu1CommandSubmission second =
                    executor.submitAsync(Vu1BarrierCommand{});
                Vu1CommandSubmission third =
                    executor.submitAsync(Vu1BarrierCommand{});
                t.Equals(executor.statistics().queueDepth, size_t{2u},
                         "the ring should expose exactly its configured capacity");

                std::atomic<bool> fourthSubmitted{false};
                std::optional<Vu1CommandSubmission> fourth;
                std::string fourthError;
                std::thread producer([&]()
                {
                    try
                    {
                        fourth.emplace(
                            executor.submitAsync(Vu1BarrierCommand{}));
                        fourthSubmitted.store(
                            true, std::memory_order_release);
                    }
                    catch (const std::exception &error)
                    {
                        fourthError = error.what();
                    }
                });
                t.IsTrue(
                    waitUntil([&]()
                    {
                        return executor.statistics().producerBlockCount >= 1u;
                    }),
                    "a producer should block at the bounded slot limit");
                t.IsFalse(
                    fourthSubmitted.load(std::memory_order_acquire),
                    "the full ring must not admit another command");

                gate->release();
                producer.join();
                t.IsTrue(fourthError.empty() && fourth.has_value(),
                         "freeing a slot should wake one serialized producer");
                t.IsTrue(
                    first.wait().disposition == Vu1CommandDisposition::Completed &&
                        second.wait().disposition == Vu1CommandDisposition::Completed &&
                        third.wait().disposition == Vu1CommandDisposition::Completed &&
                        fourth->wait().disposition == Vu1CommandDisposition::Completed,
                    "slot pressure must preserve every FIFO completion");
                const ThreadedVu1ExecutorStatistics statistics =
                    executor.statistics();
                t.Equals(statistics.queueHighWater, size_t{2u},
                         "slot high-water should not exceed the ring bound");
                t.IsTrue(statistics.producerSlotWaitCount >= 1u &&
                             statistics.producerSlotWaitNanoseconds > 0u,
                         "slot pressure should retain bounded wait evidence");

                for (uint64_t index = 0u; index < 100u; ++index)
                {
                    (void)executor.submit(
                        Vu1BarrierCommand{}, index);
                }
                t.Equals(executor.lastCompletedTicket(), 104ull,
                         "repeated ring wrap and idle wakeups should lose no work");

                const std::string ownerError = captureException([&]()
                {
                    (void)processor.process(Vu1Command{});
                });
                t.IsTrue(
                    ownerError.find("outside its owner thread") !=
                        std::string::npos,
                    "the processor should reject direct non-owner mutation");
                t.IsTrue(executor.ownerThreadId() != std::this_thread::get_id(),
                         "the threaded processor should publish a distinct owner");
            }

            {
                VuUnit unit(VuUnitId::Vu1);
                std::vector<uint8_t> code(PS2_VU1_CODE_SIZE);
                std::vector<uint8_t> data(PS2_VU1_DATA_SIZE);
                Vu1CommandProcessor processor(unit);
                processor.bindMemory(
                    code.data(), static_cast<uint32_t>(code.size()),
                    data.data(), static_cast<uint32_t>(data.size()), 1u);
                const auto gate = std::make_shared<Vu1WorkerGate>();
                const Vu1DataMemoryWriteCommand write{
                    .offset = 0u,
                    .bytes = std::vector<uint8_t>(256u, 0x5au),
                };
                const uint64_t payloadBytes =
                    vu1CommandPayloadSize(Vu1CommandPayload{write});
                ThreadedVu1Executor executor(
                    processor,
                    ThreadedVu1ExecutorOptions{
                        .queueCapacity = 8u,
                        .payloadCapacityBytes = payloadBytes * 2u,
                        .beforeProcess =
                            [gate](const Vu1Command &command, uint64_t ticket)
                            {
                                gate->beforeProcess(command, ticket);
                            },
                    });
                Vu1CommandSubmission first = executor.submitAsync(write);
                t.IsTrue(gate->waitUntilEntered(),
                         "the first payload should remain owner-active");
                Vu1CommandSubmission second = executor.submitAsync(write);
                std::atomic<bool> thirdSubmitted{false};
                std::optional<Vu1CommandSubmission> third;
                std::thread producer([&]()
                {
                    third.emplace(executor.submitAsync(write));
                    thirdSubmitted.store(true, std::memory_order_release);
                });
                t.IsTrue(
                    waitUntil([&]()
                    {
                        return executor.statistics().producerBlockCount >= 1u;
                    }),
                    "the independent payload limit should block the producer");
                t.IsFalse(
                    thirdSubmitted.load(std::memory_order_acquire),
                    "free command slots must not bypass the payload budget");
                t.Equals(executor.statistics().payloadHighWaterBytes,
                         payloadBytes * 2u,
                         "payload high-water should stop at the byte bound");
                gate->release();
                producer.join();
                (void)first.wait();
                (void)second.wait();
                (void)third->wait();
                t.IsTrue(
                    executor.statistics().producerPayloadWaitCount >= 1u,
                    "the blocked wake should be classified as payload pressure");
                t.Equals(executor.statistics().queuedPayloadBytes, 0ull,
                         "retired results should release the full payload budget");
            }
        });

        tc.Run("threaded VU1 cancellation opens reset and restore generations safely", [](TestCase &t)
        {
            VuUnit unit(VuUnitId::Vu1);
            std::vector<uint8_t> code(PS2_VU1_CODE_SIZE);
            std::vector<uint8_t> data(PS2_VU1_DATA_SIZE);
            Vu1CommandProcessor processor(
                unit, {.captureArchitecturalStateHashes = true});
            processor.bindMemory(
                code.data(), static_cast<uint32_t>(code.size()),
                data.data(), static_cast<uint32_t>(data.size()), 1u);
            const auto gate = std::make_shared<Vu1WorkerGate>();
            ThreadedVu1Executor executor(
                processor,
                ThreadedVu1ExecutorOptions{
                    .queueCapacity = 4u,
                    .payloadCapacityBytes = 1024u * 1024u,
                    .beforeProcess =
                        [gate](const Vu1Command &command, uint64_t ticket)
                        {
                            gate->beforeProcess(command, ticket);
                        },
                });

            Vu1CommandSubmission oldFirst =
                executor.submitAsync(Vu1DataMemoryWriteCommand{
                    .offset = 0u,
                    .bytes = {0x11u},
                });
            t.IsTrue(gate->waitUntilEntered(),
                     "the old generation should be held before mutation");
            Vu1CommandSubmission oldSecond =
                executor.submitAsync(Vu1BarrierCommand{});
            Vu1CommandSubmission reset =
                executor.submitAsync(Vu1ResetCommand{});
            t.Equals(reset.identity().generation, 2ull,
                     "reset should reserve the next owner generation");
            t.Equals(reset.identity().sequence, 1ull,
                     "reset should restart sequencing at one");
            executor.cancelPendingBeforeGeneration(2u);
            gate->release();

            t.IsTrue(
                oldFirst.wait().disposition ==
                    Vu1CommandDisposition::StaleGeneration &&
                    oldSecond.wait().disposition ==
                        Vu1CommandDisposition::StaleGeneration,
                "every skipped old-generation command should resolve stale");
            const Vu1CommandResult resetResult = reset.wait();
            t.IsTrue(
                resetResult.disposition == Vu1CommandDisposition::Completed &&
                    resetResult.ownerGeneration == 2u,
                "the reset should become the canonical admitted generation");
            t.Equals(data[0], uint8_t{0u},
                     "cancelled owner work must not mutate VU data memory");

            const Vu1CommandResult snapshotResult =
                executor.submit(Vu1SnapshotCommand{});
            const auto *snapshot =
                std::get_if<Vu1SnapshotResult>(&snapshotResult.payload);
            t.IsTrue(snapshot && snapshot->snapshot,
                     "the quiesced owner should return a complete snapshot");
            if (!snapshot || !snapshot->snapshot)
                return;
            (void)executor.submit(Vu1DataMemoryWriteCommand{
                .offset = 0u,
                .bytes = {0x77u},
            });
            const Vu1CommandResult restoreResult =
                executor.submit(Vu1RestoreCommand{
                    .snapshot = snapshot->snapshot,
                });
            t.Equals(restoreResult.identity.generation, 3ull,
                     "restore should open another explicit generation");
            t.Equals(restoreResult.identity.sequence, 1ull,
                     "restore should restart its generation sequence");
            t.Equals(data[0], uint8_t{0u},
                     "restore should reproduce owner memory exactly");
            const Vu1CommandResult barrier =
                executor.submit(Vu1BarrierCommand{});
            t.Equals(barrier.identity.sequence, 2ull,
                     "post-restore work should continue within the new generation");
        });

        tc.Run("threaded VU1 fatal failure reaches active queued and blocked work", [](TestCase &t)
        {
            VuUnit unit(VuUnitId::Vu1);
            Vu1CommandProcessor processor(unit);
            const auto gate =
                std::make_shared<Vu1WorkerGate>(1u, true);
            ThreadedVu1Executor executor(
                processor,
                ThreadedVu1ExecutorOptions{
                    .queueCapacity = 1u,
                    .payloadCapacityBytes = 4096u,
                    .beforeProcess =
                        [gate](const Vu1Command &command, uint64_t ticket)
                        {
                            gate->beforeProcess(command, ticket);
                        },
                });
            Vu1CommandSubmission first =
                executor.submitAsync(Vu1BarrierCommand{});
            t.IsTrue(gate->waitUntilEntered(),
                     "the injected failure should be held deterministically");
            Vu1CommandSubmission second =
                executor.submitAsync(Vu1BarrierCommand{});

            std::string thirdError;
            std::thread producer([&]()
            {
                thirdError = captureException([&]()
                {
                    (void)executor.submitAsync(Vu1BarrierCommand{});
                });
            });
            t.IsTrue(
                waitUntil([&]()
                {
                    return executor.statistics().producerBlockCount >= 1u;
                }),
                "the third command should wait on the full queue");
            gate->release();
            producer.join();

            const std::string firstError = captureException([&]()
            {
                (void)first.wait();
            });
            const std::string secondError = captureException([&]()
            {
                (void)second.wait();
            });
            t.IsTrue(
                firstError.find("injected VU1 owner failure") !=
                        std::string::npos &&
                    secondError.find("injected VU1 owner failure") !=
                        std::string::npos &&
                    thirdError.find("injected VU1 owner failure") !=
                        std::string::npos,
                "one owner failure should resolve every admitted or waiting command");
            t.IsTrue(executor.statistics().failed,
                     "fatal owner state should remain observable");
            t.IsTrue(
                captureException([&]()
                {
                    executor.rethrowFailure();
                }).find("injected VU1 owner failure") != std::string::npos,
                "later observers should receive the original owner exception");
        });

        tc.Run("threaded VU1 shutdown drains or cancels every accepted result", [](TestCase &t)
        {
            {
                VuUnit unit(VuUnitId::Vu1);
                Vu1CommandProcessor processor(unit);
                ThreadedVu1Executor executor(processor);
                executor.shutdown(Vu1ExecutorShutdownMode::Drain);
                const ThreadedVu1ExecutorStatistics statistics =
                    executor.statistics();
                t.IsFalse(statistics.started || statistics.running ||
                              statistics.accepting,
                          "empty shutdown should not create an owner");
            }

            {
                VuUnit unit(VuUnitId::Vu1);
                Vu1CommandProcessor processor(unit);
                ThreadedVu1Executor executor(
                    processor,
                    ThreadedVu1ExecutorOptions{
                        .queueCapacity = 16u,
                        .payloadCapacityBytes = 4096u,
                    });
                std::vector<Vu1CommandSubmission> submissions;
                for (uint64_t index = 0u; index < 10u; ++index)
                {
                    submissions.push_back(
                        executor.submitAsync(
                            Vu1BarrierCommand{}, index));
                }
                executor.shutdown(Vu1ExecutorShutdownMode::Drain);
                bool allCompleted = true;
                for (Vu1CommandSubmission &submission : submissions)
                {
                    allCompleted = allCompleted &&
                        submission.wait().disposition ==
                            Vu1CommandDisposition::Completed;
                }
                t.IsTrue(allCompleted,
                         "drain shutdown should retire every accepted command");
            }

            {
                VuUnit unit(VuUnitId::Vu1);
                Vu1CommandProcessor processor(unit);
                const auto gate = std::make_shared<Vu1WorkerGate>();
                ThreadedVu1Executor executor(
                    processor,
                    ThreadedVu1ExecutorOptions{
                        .queueCapacity = 2u,
                        .payloadCapacityBytes = 4096u,
                        .beforeProcess =
                            [gate](const Vu1Command &command, uint64_t ticket)
                            {
                                gate->beforeProcess(command, ticket);
                            },
                    });
                Vu1CommandSubmission active =
                    executor.submitAsync(Vu1BarrierCommand{});
                t.IsTrue(gate->waitUntilEntered(),
                         "cancel should cover an owner-active command");
                Vu1CommandSubmission queued =
                    executor.submitAsync(Vu1BarrierCommand{});
                std::atomic<bool> shutdownReturned{false};
                std::thread shutdownThread([&]()
                {
                    executor.shutdown(Vu1ExecutorShutdownMode::Cancel);
                    shutdownReturned.store(true, std::memory_order_release);
                });
                t.IsTrue(
                    waitUntil([&]()
                    {
                        return executor.statistics().cancelRequested;
                    }),
                    "cancel should publish its lifecycle state before join");
                t.IsFalse(
                    shutdownReturned.load(std::memory_order_acquire),
                    "shutdown should wait for the active owner boundary");
                gate->release();
                shutdownThread.join();
                t.IsFalse(captureException([&]()
                {
                    (void)active.wait();
                }).empty(),
                          "cancel should resolve active work exceptionally");
                t.IsFalse(captureException([&]()
                {
                    (void)queued.wait();
                }).empty(),
                          "cancel should resolve queued work exceptionally");
                const ThreadedVu1ExecutorStatistics statistics =
                    executor.statistics();
                t.Equals(statistics.queueDepth, size_t{0u},
                         "cancel should empty the bounded queue");
                t.Equals(statistics.queuedPayloadBytes, 0ull,
                         "cancel should release all payload reservations");
            }
        });

        tc.Run("threaded synchronous VU1 matches inline slices under worker jitter", [](TestCase &t)
        {
            VuUnit inlineUnit(VuUnitId::Vu1);
            VuUnit threadedUnit(VuUnitId::Vu1);
            std::vector<uint8_t> inlineCode(PS2_VU1_CODE_SIZE);
            std::vector<uint8_t> threadedCode(PS2_VU1_CODE_SIZE);
            std::vector<uint8_t> inlineData(PS2_VU1_DATA_SIZE);
            std::vector<uint8_t> threadedData(PS2_VU1_DATA_SIZE);
            constexpr Vu1CommandProcessorConfiguration processorConfig{
                .captureCommandDigests = true,
                .captureArchitecturalStateHashes = true,
            };
            Vu1CommandProcessor inlineProcessor(
                inlineUnit, processorConfig);
            Vu1CommandProcessor threadedProcessor(
                threadedUnit, processorConfig);
            InlineVu1Executor inlineExecutor(inlineProcessor);
            ThreadedVu1Executor threadedExecutor(
                threadedProcessor,
                ThreadedVu1ExecutorOptions{
                    .queueCapacity = 4u,
                    .payloadCapacityBytes = 1024u * 1024u,
                    .beforeProcess =
                        [](const Vu1Command &, uint64_t ticket)
                        {
                            const uint64_t delay =
                                ((ticket * 1103515245ull + 12345ull) >> 8u) %
                                5u;
                            std::this_thread::sleep_for(
                                std::chrono::microseconds(delay * 20u));
                        },
                });

            const auto compareResults = [&](
                const Vu1CommandResult &inlineResult,
                const Vu1CommandResult &threadedResult,
                const char *message)
            {
                t.IsTrue(
                    inlineResult.identity == threadedResult.identity &&
                        inlineResult.digest == threadedResult.digest &&
                        inlineResult.disposition == threadedResult.disposition &&
                        inlineResult.ownerGeneration ==
                            threadedResult.ownerGeneration &&
                        inlineResult.codeGeneration ==
                            threadedResult.codeGeneration &&
                        inlineResult.programCounter ==
                            threadedResult.programCounter &&
                        inlineResult.active == threadedResult.active,
                    message);
            };
            const auto submitBoth = [&](
                Vu1CommandPayload inlinePayload,
                Vu1CommandPayload threadedPayload,
                uint64_t tick)
            {
                Vu1CommandResult inlineResult = inlineExecutor.submit(
                    std::move(inlinePayload), tick, tick + 1000u);
                Vu1CommandResult threadedResult = threadedExecutor.submit(
                    std::move(threadedPayload), tick, tick + 1000u);
                compareResults(
                    inlineResult, threadedResult,
                    "jitter must not alter command identity or digest");
                return std::pair<Vu1CommandResult, Vu1CommandResult>{
                    std::move(inlineResult),
                    std::move(threadedResult),
                };
            };

            (void)submitBoth(
                Vu1BindMemoryCommand{
                    .microMemory = inlineCode,
                    .dataMemory = inlineData,
                    .codeGeneration = 1u,
                },
                Vu1BindMemoryCommand{
                    .microMemory = threadedCode,
                    .dataMemory = threadedData,
                    .codeGeneration = 1u,
                },
                0u);

            std::vector<uint8_t> program;
            for (uint32_t index = 0u; index < 300u; ++index)
            {
                const auto pair = instructionPair(
                    makeVuIaddiu(1u, 1u, 1),
                    index == 299u ? kVuUpperEnd : kVuUpperNop);
                program.insert(program.end(), pair.begin(), pair.end());
            }
            (void)submitBoth(
                Vu1MicroMemoryWriteCommand{
                    .offset = 0u,
                    .bytes = program,
                },
                Vu1MicroMemoryWriteCommand{
                    .offset = 0u,
                    .bytes = program,
                },
                16u);
            (void)submitBoth(
                Vu1MscalCommand{}, Vu1MscalCommand{}, 32u);

            bool slicesMatch = true;
            uint64_t tick = 48u;
            uint32_t sliceIndex = 0u;
            for (;;)
            {
                const uint32_t budget =
                    sliceIndex == 0u ? 16u : 128u;
                auto [inlineResult, threadedResult] = submitBoth(
                    Vu1AdvanceSliceCommand{
                        .maximumCycles = budget,
                        .captureState = true,
                    },
                    Vu1AdvanceSliceCommand{
                        .maximumCycles = budget,
                        .captureState = true,
                    },
                    tick);
                const Vu1SliceResult *inlineSlice =
                    sliceResult(inlineResult);
                const Vu1SliceResult *threadedSlice =
                    sliceResult(threadedResult);
                std::string difference;
                slicesMatch = slicesMatch && inlineSlice && threadedSlice &&
                    inlineSlice->run.executedCycles ==
                        threadedSlice->run.executedCycles &&
                    inlineSlice->run.reason == threadedSlice->run.reason &&
                    inlineSlice->architecturalStateHash ==
                        threadedSlice->architecturalStateHash &&
                    inlineSlice->vifCanResume ==
                        threadedSlice->vifCanResume &&
                    inlineSlice->path1Packets.size() ==
                        threadedSlice->path1Packets.size() &&
                    inlineSlice->state && threadedSlice->state &&
                    vuExecutionStatesEqual(
                        *inlineSlice->state,
                        *threadedSlice->state,
                        &difference);
                if (!inlineResult.active)
                    break;
                tick += 128u;
                ++sliceIndex;
                if (sliceIndex > 8u)
                {
                    slicesMatch = false;
                    break;
                }
            }
            t.IsTrue(slicesMatch,
                     "each synchronous owner slice should match inline state exactly");
            t.Equals(inlineUnit.state().vi[1], int32_t{300},
                     "the differential program should execute all instruction pairs");

            auto [inlineSnapshotResult, threadedSnapshotResult] = submitBoth(
                Vu1SnapshotCommand{}, Vu1SnapshotCommand{}, tick + 128u);
            const auto *inlineSnapshot =
                std::get_if<Vu1SnapshotResult>(&inlineSnapshotResult.payload);
            const auto *threadedSnapshot =
                std::get_if<Vu1SnapshotResult>(&threadedSnapshotResult.payload);
            t.IsTrue(
                inlineSnapshot && threadedSnapshot &&
                    inlineSnapshot->snapshot && threadedSnapshot->snapshot &&
                    inlineSnapshot->architecturalStateHash ==
                        threadedSnapshot->architecturalStateHash &&
                    threadedSnapshot->snapshot->state.vi[1] == 300,
                "quiesced snapshots should have identical architectural hashes");
            if (!inlineSnapshot || !threadedSnapshot ||
                !inlineSnapshot->snapshot || !threadedSnapshot->snapshot)
            {
                return;
            }

            (void)submitBoth(
                Vu1DataMemoryWriteCommand{
                    .offset = 0x40u,
                    .bytes = {1u, 2u, 3u, 4u},
                },
                Vu1DataMemoryWriteCommand{
                    .offset = 0x40u,
                    .bytes = {1u, 2u, 3u, 4u},
                },
                tick + 256u);
            (void)submitBoth(
                Vu1RestoreCommand{
                    .snapshot = inlineSnapshot->snapshot,
                },
                Vu1RestoreCommand{
                    .snapshot = threadedSnapshot->snapshot,
                },
                tick + 384u);
            auto [inlineBarrier, threadedBarrier] = submitBoth(
                Vu1BarrierCommand{}, Vu1BarrierCommand{}, tick + 512u);
            const auto *inlineHash =
                std::get_if<Vu1BarrierResult>(&inlineBarrier.payload);
            const auto *threadedHash =
                std::get_if<Vu1BarrierResult>(&threadedBarrier.payload);
            t.IsTrue(
                inlineHash && threadedHash &&
                    inlineHash->architecturalStateHash ==
                        threadedHash->architecturalStateHash,
                "restore and barrier should retain byte-exact owner state");
            t.IsTrue(threadedExecutor.statistics().resultWaitCount > 0u,
                     "synchronous mode should expose its conservative rendezvous cost");
        });

        tc.Run("threaded VU1 commits a completed speculative slice in order", [](TestCase &t)
        {
            VuUnit unit(VuUnitId::Vu1);
            unit.setProgressTrackingEnabled(true);
            Vu1CommandProcessor processor(
                unit,
                Vu1CommandProcessorConfiguration{
                    .captureCommandDigests = true,
                    .captureArchitecturalStateHashes = true,
                });
            ThreadedVu1Executor executor(
                processor,
                ThreadedVu1ExecutorOptions{
                    .queueCapacity = 1u,
                    .payloadCapacityBytes = 1024u * 1024u,
                });
            (void)executor.submit(Vu1BindMemoryCommand{
                .microMemory =
                    std::vector<uint8_t>(PS2_VU1_CODE_SIZE),
                .dataMemory =
                    std::vector<uint8_t>(PS2_VU1_DATA_SIZE),
                .codeGeneration = 1u,
                .deferredDiagnostics = true,
            });
            const auto pair = instructionPair(
                makeVuIaddiu(1u, 0u, 7), kVuUpperEnd);
            (void)executor.submit(Vu1MicroMemoryWriteCommand{
                .offset = 0u,
                .bytes = std::vector<uint8_t>(
                    pair.begin(), pair.end()),
            });
            (void)executor.submit(Vu1MscalCommand{}, 16u, 100u);

            Vu1CommandSubmission pending =
                executor.submitSpeculativeAdvance(
                    Vu1AdvanceSliceCommand{
                        .maximumCycles = 16u,
                        .captureState = true,
                    },
                    Vu1SpeculationResolution::None,
                    32u, 101u);
            const Vu1CommandResult privateResult =
                executor.wait(std::move(pending));
            const Vu1SliceResult *const privateSlice =
                sliceResult(privateResult);
            t.IsTrue(
                privateSlice && privateSlice->state &&
                    privateSlice->state->vi[1] == 7,
                "a ready speculative result should own its post-slice state");
            t.IsTrue(
                captureException(
                    [&executor]()
                    {
                        (void)executor.submit(Vu1SnapshotCommand{});
                    }) ==
                    "threaded VU1 speculation requires an explicit resolution",
                "ordinary owner work must not observe unresolved speculative state");

            const Vu1CommandResult committed =
                executor.submitResolvingSpeculation(
                    Vu1SnapshotCommand{},
                    Vu1SpeculationResolution::Commit,
                    48u, 102u);
            const auto *const snapshot =
                std::get_if<Vu1SnapshotResult>(&committed.payload);
            t.IsTrue(
                snapshot && snapshot->snapshot &&
                    snapshot->snapshot->state.vi[1] == 7,
                "committing should make the private post-slice state canonical");
            t.Equals(
                committed.identity.sequence,
                privateResult.identity.sequence + 1u,
                "a committed speculative slice should consume one semantic sequence");
            const ThreadedVu1ExecutorStatistics statistics =
                executor.statistics();
            t.Equals(statistics.speculation.capturedSlices, 1ull,
                     "one speculative checkpoint should be captured");
            t.Equals(statistics.speculation.committedSlices, 1ull,
                     "the next ordered command should commit the checkpoint");
            t.Equals(statistics.speculation.rolledBackSlices, 0ull,
                     "the ordinary publication path should not roll back");
        });

        tc.Run("threaded VU1 rolls back an unpublished slice before a hazard", [](TestCase &t)
        {
            VuUnit unit(VuUnitId::Vu1);
            unit.setProgressTrackingEnabled(true);
            Vu1CommandProcessor processor(
                unit,
                Vu1CommandProcessorConfiguration{
                    .captureCommandDigests = true,
                    .captureArchitecturalStateHashes = true,
                });
            ThreadedVu1Executor executor(
                processor,
                ThreadedVu1ExecutorOptions{
                    .queueCapacity = 2u,
                    .payloadCapacityBytes = 1024u * 1024u,
                });
            (void)executor.submit(Vu1BindMemoryCommand{
                .microMemory =
                    std::vector<uint8_t>(PS2_VU1_CODE_SIZE),
                .dataMemory =
                    std::vector<uint8_t>(PS2_VU1_DATA_SIZE),
                .codeGeneration = 1u,
                .deferredDiagnostics = true,
            });
            std::vector<uint8_t> program(16u, 0u);
            const auto storePair = instructionPair(
                makeVuSq(0x0fu, 1u, 0u, 0),
                kVuUpperEnd);
            const auto delayPair = instructionPair(
                0u, kVuUpperNop);
            std::copy(
                storePair.begin(), storePair.end(),
                program.begin());
            std::copy(
                delayPair.begin(), delayPair.end(),
                program.begin() + 8u);
            (void)executor.submit(Vu1MicroMemoryWriteCommand{
                .offset = 0u,
                .bytes = program,
            });
            const std::array<uint32_t, 4u> storedWords = {
                0x3f800000u, 0x40000000u,
                0x40400000u, 0x40800000u,
            };
            (void)executor.submit(Vu1RegisterWriteCommand{
                .kind = Vu1RegisterKind::VectorFloat,
                .index = 1u,
                .laneMask = 0x0fu,
                .words = storedWords,
            });
            (void)executor.submit(Vu1MscalCommand{}, 20u, 200u);

            Vu1CommandSubmission first =
                executor.submitSpeculativeAdvance(
                    Vu1AdvanceSliceCommand{
                        .maximumCycles = 16u,
                    },
                    Vu1SpeculationResolution::None,
                    100u, 201u);
            const Vu1CommandResult discarded =
                executor.wait(std::move(first));

            const Vu1CommandResult hazard =
                executor.submitResolvingSpeculation(
                    Vu1DataMemoryWriteCommand{
                        .offset = 0u,
                        .bytes = {0x99u},
                    },
                    Vu1SpeculationResolution::Rollback,
                    30u, 202u);
            t.Equals(
                hazard.identity.sequence,
                discarded.identity.sequence,
                "rollback should reuse the unpublished semantic sequence");
            t.Equals(hazard.identity.guestTick, 30ull,
                     "rollback should restore the pre-slice guest-tick frontier");

            Vu1CommandSubmission second =
                executor.submitSpeculativeAdvance(
                    Vu1AdvanceSliceCommand{
                        .maximumCycles = 16u,
                    },
                    Vu1SpeculationResolution::None,
                    100u, 201u);
            const Vu1CommandResult published =
                executor.wait(std::move(second));
            t.Equals(
                published.identity.sequence,
                hazard.identity.sequence + 1u,
                "the recomputed slice should follow the intervening hazard");
            const Vu1CommandResult committed =
                executor.submitResolvingSpeculation(
                    Vu1SnapshotCommand{},
                    Vu1SpeculationResolution::Commit,
                    120u, 203u);
            const auto *const snapshot =
                std::get_if<Vu1SnapshotResult>(&committed.payload);
            bool memoryMatches = snapshot && snapshot->snapshot;
            if (memoryMatches)
            {
                std::array<uint32_t, 4u> observed{};
                std::memcpy(
                    observed.data(),
                    snapshot->snapshot->dataMemory.data(),
                    sizeof(observed));
                memoryMatches = observed == storedWords;
            }
            t.IsTrue(memoryMatches,
                     "the recomputed slice should apply after the ordered memory write");
            t.IsTrue(
                snapshot && snapshot->snapshot &&
                    snapshot->snapshot->progress.invocations == 1u,
                "rolled-back speculative work must not double-count progress");

            const ThreadedVu1ExecutorStatistics statistics =
                executor.statistics();
            t.Equals(statistics.speculation.capturedSlices, 2ull,
                     "both attempted slices should capture checkpoints");
            t.Equals(statistics.speculation.rolledBackSlices, 1ull,
                     "the intervening write should roll back one slice");
            t.Equals(statistics.speculation.committedSlices, 1ull,
                     "the recomputed result should commit once");
            t.Equals(
                statistics.speculation.checkpointBytesCopied,
                uint64_t{2u * PS2_VU1_DATA_SIZE},
                "each speculative slice should copy one bounded data image");
        });

        tc.Run("threaded VU1 restores a save snapshot with a slice pending", [](TestCase &t)
        {
            std::mutex gateMutex;
            std::condition_variable gateCv;
            bool entered = false;
            bool release = false;
            VuUnit unit(VuUnitId::Vu1);
            unit.setProgressTrackingEnabled(true);
            Vu1CommandProcessor processor(
                unit,
                Vu1CommandProcessorConfiguration{
                    .captureCommandDigests = true,
                    .captureArchitecturalStateHashes = true,
                });
            ThreadedVu1Executor executor(
                processor,
                ThreadedVu1ExecutorOptions{
                    .queueCapacity = 2u,
                    .payloadCapacityBytes = 1024u * 1024u,
                    .beforeProcess =
                        [&](const Vu1Command &command, uint64_t)
                        {
                            if (command.type !=
                                Vu1CommandType::AdvanceSlice)
                            {
                                return;
                            }
                            std::unique_lock<std::mutex> lock(gateMutex);
                            if (entered)
                                return;
                            entered = true;
                            gateCv.notify_all();
                            gateCv.wait(
                                lock,
                                [&release]()
                                {
                                    return release;
                                });
                        },
                });
            (void)executor.submit(Vu1BindMemoryCommand{
                .microMemory =
                    std::vector<uint8_t>(PS2_VU1_CODE_SIZE),
                .dataMemory =
                    std::vector<uint8_t>(PS2_VU1_DATA_SIZE),
                .codeGeneration = 1u,
                .deferredDiagnostics = true,
            });
            std::vector<uint8_t> program(16u, 0u);
            const auto workPair = instructionPair(
                makeVuIaddiu(1u, 0u, 7), kVuUpperEnd);
            const auto delayPair = instructionPair(
                0u, kVuUpperNop);
            std::copy(
                workPair.begin(), workPair.end(),
                program.begin());
            std::copy(
                delayPair.begin(), delayPair.end(),
                program.begin() + 8u);
            (void)executor.submit(Vu1MicroMemoryWriteCommand{
                .offset = 0u,
                .bytes = program,
            });
            (void)executor.submit(Vu1MscalCommand{}, 16u, 300u);
            const Vu1CommandResult savedResult =
                executor.submit(Vu1SnapshotCommand{}, 20u, 301u);
            const auto *const saved =
                std::get_if<Vu1SnapshotResult>(
                    &savedResult.payload);
            t.IsTrue(saved && saved->snapshot,
                     "save should capture the active pre-slice state");
            if (!saved || !saved->snapshot)
                return;

            Vu1CommandSubmission pending =
                executor.submitSpeculativeAdvance(
                    Vu1AdvanceSliceCommand{
                        .maximumCycles = 16u,
                        .captureState = true,
                    },
                    Vu1SpeculationResolution::None,
                    128u, 302u);
            t.IsTrue(
                waitFor(
                    gateCv, gateMutex,
                    [&entered]()
                    {
                        return entered;
                    }),
                "the saved generation should have one pending slice");

            std::atomic<bool> restoreStarted{false};
            std::atomic<bool> restoreDone{false};
            std::optional<Vu1CommandResult> restored;
            std::string restoreError;
            std::thread restorer(
                [&]()
                {
                    restoreStarted.store(
                        true, std::memory_order_release);
                    try
                    {
                        restored.emplace(
                            executor.submitResolvingSpeculation(
                                Vu1RestoreCommand{
                                    .snapshot = saved->snapshot,
                                },
                                Vu1SpeculationResolution::Rollback,
                                24u, 303u));
                    }
                    catch (const std::exception &error)
                    {
                        restoreError = error.what();
                    }
                    restoreDone.store(
                        true, std::memory_order_release);
                });
            t.IsTrue(
                waitUntil(
                    [&restoreStarted]()
                    {
                        return restoreStarted.load(
                            std::memory_order_acquire);
                    }),
                "the load producer should start");
            std::this_thread::sleep_for(5ms);
            t.IsFalse(
                restoreDone.load(std::memory_order_acquire),
                "load should wait for the speculative rollback boundary");
            {
                std::lock_guard<std::mutex> lock(gateMutex);
                release = true;
            }
            gateCv.notify_all();
            restorer.join();
            const Vu1CommandResult privateResult =
                executor.wait(std::move(pending));
            const Vu1SliceResult *const privateSlice =
                sliceResult(privateResult);
            t.IsTrue(
                privateSlice && privateSlice->state &&
                    privateSlice->state->vi[1] == 7,
                "the discarded future should retain its private result");
            t.IsTrue(restoreError.empty() && restored.has_value(),
                     "the pending load should complete without transport failure");
            if (!restored)
                return;
            t.Equals(restored->identity.sequence, 1ull,
                     "restore should begin a fresh semantic generation");

            const Vu1CommandResult canonicalResult =
                executor.submit(Vu1SnapshotCommand{}, 25u, 304u);
            const auto *const canonical =
                std::get_if<Vu1SnapshotResult>(
                    &canonicalResult.payload);
            t.IsTrue(
                canonical && canonical->snapshot &&
                    canonical->architecturalStateHash ==
                        saved->architecturalStateHash &&
                    canonical->snapshot->state.vi[1] == 0 &&
                    canonical->snapshot->state.active,
                "load should restore the exact pre-slice architecture");

            VuUnit inlineUnit(VuUnitId::Vu1);
            inlineUnit.setProgressTrackingEnabled(true);
            Vu1CommandProcessor inlineProcessor(
                inlineUnit,
                Vu1CommandProcessorConfiguration{
                    .captureArchitecturalStateHashes = true,
                });
            InlineVu1Executor inlineExecutor(inlineProcessor);
            (void)inlineExecutor.submit(Vu1BindMemoryCommand{
                .microMemory =
                    std::vector<uint8_t>(PS2_VU1_CODE_SIZE),
                .dataMemory =
                    std::vector<uint8_t>(PS2_VU1_DATA_SIZE),
                .codeGeneration = 1u,
            });
            (void)inlineExecutor.submit(Vu1RestoreCommand{
                .snapshot = saved->snapshot,
            });
            const Vu1CommandResult threadedAdvance =
                executor.submit(Vu1AdvanceSliceCommand{
                    .maximumCycles = 16u,
                    .captureState = true,
                });
            const Vu1CommandResult inlineAdvance =
                inlineExecutor.submit(Vu1AdvanceSliceCommand{
                    .maximumCycles = 16u,
                    .captureState = true,
                });
            const Vu1SliceResult *const threadedSlice =
                sliceResult(threadedAdvance);
            const Vu1SliceResult *const inlineSlice =
                sliceResult(inlineAdvance);
            t.IsTrue(threadedSlice != nullptr,
                     "threaded post-load advance should return a slice");
            t.IsTrue(inlineSlice != nullptr,
                     "inline post-load advance should return a slice");
            t.IsTrue(threadedSlice && threadedSlice->state,
                     "threaded post-load advance should capture state");
            t.IsTrue(inlineSlice && inlineSlice->state,
                     "inline post-load advance should capture state");
            std::string difference;
            const bool stateMatches =
                threadedSlice && inlineSlice &&
                threadedSlice->state && inlineSlice->state &&
                vuExecutionStatesEqual(
                    *threadedSlice->state,
                    *inlineSlice->state,
                    &difference);
            if (!stateMatches)
            {
                t.Fail(
                    "post-load VU state should match the inline oracle: " +
                    difference);
            }
            t.IsTrue(
                threadedSlice && inlineSlice &&
                    threadedSlice->architecturalStateHash ==
                        inlineSlice->architecturalStateHash,
                "post-load architecture hash should match the inline oracle");
            t.Equals(
                executor.statistics().speculation.rolledBackSlices,
                1ull,
                "pending load should roll back exactly one private slice");
        });

        tc.Run("threaded VU1 defers diagnostics for EE publication", [](TestCase &t)
        {
            class Observer final : public IVuExecutionObserver
            {
            public:
                bool vuTraceEnabled() const override
                {
                    return true;
                }

                void traceVuInvocation(
                    uint32_t, uint32_t, uint32_t, bool,
                    const VuExecutionState &) override
                {
                    ++invocations;
                    publicationThread = std::this_thread::get_id();
                }

                void traceVuInstruction(
                    uint32_t, uint32_t, uint32_t,
                    const VuExecutionState &) override
                {
                    ++instructions;
                    publicationThread = std::this_thread::get_id();
                }

                uint64_t invocations = 0u;
                uint64_t instructions = 0u;
                std::thread::id publicationThread{};
            } observer;

            VuUnit unit(VuUnitId::Vu1);
            Vu1CommandProcessor processor(unit);
            ThreadedVu1Executor executor(processor);
            (void)executor.submit(Vu1BindMemoryCommand{
                .microMemory =
                    std::vector<uint8_t>(PS2_VU1_CODE_SIZE),
                .dataMemory =
                    std::vector<uint8_t>(PS2_VU1_DATA_SIZE),
                .codeGeneration = 1u,
                .deferredDiagnostics = true,
            });
            (void)executor.submit(Vu1SetDiagnosticsCommand{
                .traceEnabled = true,
            });
            const auto pair = instructionPair(
                makeVuIaddiu(1u, 0u, 7), kVuUpperEnd);
            (void)executor.submit(Vu1MicroMemoryWriteCommand{
                .offset = 0u,
                .bytes = std::vector<uint8_t>(
                    pair.begin(), pair.end()),
            });
            const Vu1CommandResult start =
                executor.submit(Vu1MscalCommand{});
            t.Equals(observer.invocations, 0ull,
                     "the owner must not call the EE observer directly");
            t.Equals(start.diagnostics.size(), size_t{1u},
                     "start should return one owned invocation record");
            replayVu1Diagnostics(start.diagnostics, observer);
            t.Equals(observer.invocations, 1ull,
                     "EE replay should publish the deferred invocation");
            t.IsTrue(observer.publicationThread ==
                         std::this_thread::get_id(),
                     "diagnostics should publish on the waiting EE thread");

            const Vu1CommandResult slice = executor.submit(
                Vu1AdvanceSliceCommand{
                    .maximumCycles = 2u,
                });
            t.Equals(observer.instructions, 0ull,
                     "slice tracing should remain private on completion");
            t.IsTrue(!slice.diagnostics.empty(),
                     "the slice should own its trace records");
            replayVu1Diagnostics(slice.diagnostics, observer);
            t.IsTrue(observer.instructions >= 1u,
                     "EE replay should publish traced instruction pairs");
        });

        tc.Run("threaded VU1 snapshots backend diagnostics on the owner", [](TestCase &t)
        {
            VuUnit unit(VuUnitId::Vu1);
            Vu1CommandProcessor processor(unit);
            ThreadedVu1Executor executor(processor);
            (void)executor.submit(Vu1BindMemoryCommand{
                .microMemory =
                    std::vector<uint8_t>(PS2_VU1_CODE_SIZE),
                .dataMemory =
                    std::vector<uint8_t>(PS2_VU1_DATA_SIZE),
                .codeGeneration = 1u,
                .deferredDiagnostics = true,
            });

            if (VuRecompilerBackend::supported())
            {
                const Vu1CommandResult backendResult =
                    executor.submit(Vu1SetBackendCommand{
                        .backend = VuBackendKind::Recompiler,
                    });
                const auto *const backendStatus =
                    std::get_if<Vu1BackendStatusResult>(
                        &backendResult.payload);
                t.IsTrue(
                    backendStatus && backendStatus->accepted,
                    "the owner should accept the supported native backend");
                const auto pair = instructionPair(
                    makeVuIaddiu(1u, 0u, 7), kVuUpperEnd);
                (void)executor.submit(Vu1MicroMemoryWriteCommand{
                    .offset = 0u,
                    .bytes = std::vector<uint8_t>(
                        pair.begin(), pair.end()),
                });
                (void)executor.submit(Vu1MscalCommand{});
                (void)executor.submit(Vu1AdvanceSliceCommand{
                    .maximumCycles = 16u,
                });
            }

            const Vu1CommandResult snapshotResult =
                executor.submit(Vu1SnapshotCommand{
                    .includeBackendDiagnostics = true,
                });
            const auto *const snapshot =
                std::get_if<Vu1SnapshotResult>(
                    &snapshotResult.payload);
            t.IsTrue(
                snapshot && snapshot->snapshot &&
                    snapshot->snapshot->backendDiagnostics,
                "the typed owner snapshot should include requested diagnostics");
            if (!snapshot || !snapshot->snapshot ||
                !snapshot->snapshot->backendDiagnostics)
            {
                return;
            }
            if (VuRecompilerBackend::supported())
            {
                const Vu1BackendDiagnosticsSnapshot &diagnostics =
                    *snapshot->snapshot->backendDiagnostics;
                t.IsTrue(
                    diagnostics.programCache.has_value(),
                    "native execution should publish cache diagnostics");
                t.IsTrue(
                    diagnostics.recompiler.has_value(),
                    "native execution should publish recompiler diagnostics");
                if (diagnostics.recompiler)
                {
                    t.IsTrue(
                        diagnostics.recompiler->diagnostics.nativeEntries !=
                            0u,
                        "the owner should snapshot native activity");
                }
            }
        });

        tc.Run("runtime selects synchronous VU1 ownership and mirrors observations", [](TestCase &t)
        {
            PS2RuntimeConfiguration configuration =
                defaultPs2RuntimeConfiguration();
            configuration.vu1ExecutionMode =
                Vu1ExecutionMode::ThreadedSynchronous;
            configuration.vu1CommandQueueCapacity = 2u;
            configuration.vu1CommandPayloadCapacityBytes =
                1024u * 1024u;
            configuration.captureVu1ArchitecturalStateHashes = true;
            PS2Runtime runtime(configuration);
            t.IsTrue(runtime.memory().initialize(),
                     "threaded runtime memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "threaded runtime should bind owned VU1 images");
            t.IsTrue(
                runtime.vu1ExecutionMode() ==
                    Vu1ExecutionMode::ThreadedSynchronous,
                "the runtime should retain its selected VU1 mode");
            t.IsTrue(runtime.vu1OwnerStatistics().started,
                     "binding should lazily start the VU1 owner");
            t.IsTrue(
                captureException(
                    [&runtime]()
                    {
                        (void)runtime.vu1().isActive();
                    }) ==
                    "direct VU1 access is unavailable in threaded mode",
                "threaded runtime should reject direct unit access");
            t.IsTrue(
                captureException(
                    [&runtime]()
                    {
                        (void)runtime.vu1CommandProcessor().generation();
                    }) ==
                    "direct VU1 processor access is unavailable in threaded mode",
                "threaded runtime should reject direct processor access");

            runtime.memory().write32(
                PS2_VU1_CODE_BASE, 0x44332211u);
            runtime.memory().write32(
                PS2_VU1_DATA_BASE + 4u, 0xaabbccddu);
            const std::shared_ptr<const Vu1Snapshot> snapshot =
                runtime.snapshotVu1Owner();
            uint32_t codeWord = 0u;
            uint32_t dataWord = 0u;
            std::memcpy(
                &codeWord, snapshot->microMemory.data(),
                sizeof(codeWord));
            std::memcpy(
                &dataWord, snapshot->dataMemory.data() + 4u,
                sizeof(dataWord));
            t.Equals(codeWord, uint32_t{0x44332211u},
                     "mapped MPG-style writes should reach owner micro memory");
            t.Equals(dataWord, uint32_t{0xaabbccddu},
                     "mapped writes should reach owner data memory");
            t.Equals(
                runtime.memory().read32(PS2_VU1_DATA_BASE + 4u),
                uint32_t{0xaabbccddu},
                "EE mapped reads should observe an ordered owner snapshot");

            uint8_t *const mirror = runtime.memory().getVU1Data();
            mirror[4u] ^= 0xffu;
            const std::shared_ptr<const Vu1Snapshot> ownerAfterMirrorEdit =
                runtime.snapshotVu1Owner();
            t.Equals(ownerAfterMirrorEdit->dataMemory[4u], uint8_t{0xddu},
                     "the EE observation mirror must not be canonical owner storage");
        });

        tc.Run("runtime publishes an early VU1 slice only at its scheduler event", [](TestCase &t)
        {
            PS2RuntimeConfiguration configuration =
                defaultPs2RuntimeConfiguration();
            configuration.vu1ExecutionMode =
                Vu1ExecutionMode::ThreadedAsync;
            configuration.vu1CommandQueueCapacity = 1u;
            configuration.vu1CommandPayloadCapacityBytes =
                1024u * 1024u;
            configuration.captureVu1ArchitecturalStateHashes = true;
            PS2Runtime runtime(configuration);
            t.IsTrue(runtime.memory().initialize(),
                     "asynchronous runtime memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "asynchronous runtime subsystems should bind");

            runtime.memory().write32(
                PS2_VU1_CODE_BASE,
                makeVuIaddiu(1u, 0u, 7));
            runtime.memory().write32(
                PS2_VU1_CODE_BASE + 4u,
                kVuUpperEnd);
            runtime.memory().write32(
                PS2_VU1_CODE_BASE + 8u, 0u);
            runtime.memory().write32(
                PS2_VU1_CODE_BASE + 12u,
                kVuUpperNop);
            const uint32_t mscal =
                makeVifCommand(0x14u, 0u, 0u);
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(&mscal),
                sizeof(mscal));

            t.IsTrue(
                waitUntil(
                    [&runtime]()
                    {
                        const Vu1AsyncRuntimeStatistics statistics =
                            runtime.vu1AsyncStatistics();
                        return statistics.pendingSlice &&
                               statistics.owner.completedTickets ==
                                   statistics.owner.submittedTickets;
                    }),
                "the startup slice should become ready before its event");
            const PS2Runtime::DebugVu1Timing beforeTiming =
                runtime.debugVu1TimingSnapshot();
            t.IsTrue(beforeTiming.active,
                     "worker completion must not clear published VU busy");
            t.Equals(beforeTiming.totalAdvancedCycles, 0ull,
                     "worker completion must not advance published guest time");

            const std::shared_ptr<const Vu1Snapshot> before =
                runtime.snapshotVu1Owner();
            t.Equals(before->state.vi[1], int32_t{0},
                     "a pre-event observation should see the rolled-back state");
            t.IsTrue(runtime.vu1AsyncStatistics().pendingSlice,
                     "the observation barrier should recompute the same slice");

            t.IsTrue(
                waitUntil(
                    [&runtime]()
                    {
                        const Vu1AsyncRuntimeStatistics statistics =
                            runtime.vu1AsyncStatistics();
                        return statistics.owner.completedTickets ==
                               statistics.owner.submittedTickets;
                    }),
                "the recomputed slice should finish before publication");
            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);

            const std::shared_ptr<const Vu1Snapshot> after =
                runtime.snapshotVu1Owner();
            t.Equals(after->state.vi[1], int32_t{7},
                     "the scheduled event should publish the completed slice");
            const PS2Runtime::DebugVu1Timing afterTiming =
                runtime.debugVu1TimingSnapshot();
            t.IsFalse(afterTiming.active,
                      "the E-bit program should complete at publication");
            t.IsTrue(afterTiming.totalAdvancedCycles != 0u,
                     "publication should account the executed VU cycles");

            const Vu1AsyncRuntimeStatistics statistics =
                runtime.vu1AsyncStatistics();
            t.Equals(statistics.slicesSubmitted, 2ull,
                     "one observation rollback should cause one recomputation");
            t.Equals(statistics.slicesPublished, 1ull,
                     "exactly one guest slice should publish");
            t.Equals(statistics.resultsReadyAtEvent, 1ull,
                     "the forced-ready result should be classified at the event");
            t.Equals(statistics.resultsLateAtEvent, 0ull,
                     "the forced-ready result should not be classified late");
            t.Equals(statistics.hazardBarrierCount, 1ull,
                     "the pre-event snapshot should be one explicit hazard");
            t.Equals(statistics.hazardRequeueCount, 1ull,
                     "the hazard should requeue the unchanged event slice");
            t.Equals(statistics.owner.speculation.rolledBackSlices, 1ull,
                     "the owner should roll back the first private result");
            t.Equals(statistics.owner.speculation.committedSlices, 1ull,
                     "the post-event snapshot should commit the published result");
        });

        tc.Run("runtime coalesces one VIF1 parser batch into one speculation hazard", [](TestCase &t)
        {
            struct RunResult
            {
                uint64_t architecturalHash = 0u;
                Vu1AsyncRuntimeStatistics afterPayload{};
                Vu1AsyncRuntimeStatistics afterEmptyService{};
                Vu1AsyncRuntimeStatistics afterPublication{};
            };

            const auto run =
                [&](Vu1ExecutionMode mode) -> RunResult
                {
                    PS2RuntimeConfiguration configuration =
                        defaultPs2RuntimeConfiguration();
                    configuration.vu1ExecutionMode = mode;
                    configuration.vu1CommandQueueCapacity = 1u;
                    configuration.vu1CommandPayloadCapacityBytes =
                        1024u * 1024u;
                    configuration.captureVu1ArchitecturalStateHashes =
                        true;
                    PS2Runtime runtime(configuration);
                    if (!runtime.memory().initialize() ||
                        !runtime.syncCoreSubsystems())
                    {
                        t.Fail("VIF1 hazard-batch runtime should initialize");
                        return {};
                    }

                    writeRuntimeVu1InstructionPair(
                        runtime, 0u,
                        makeVuIaddiu(1u, 0u, 7),
                        kVuUpperNop);
                    writeRuntimeVu1InstructionPair(
                        runtime, 8u, 0u, kVuUpperEnd);
                    writeRuntimeVu1InstructionPair(
                        runtime, 16u, 0u, kVuUpperNop);
                    const uint32_t mscal =
                        makeVifCommand(0x14u, 0u, 0u);
                    runtime.memory().processVIF1Data(
                        reinterpret_cast<const uint8_t *>(&mscal),
                        sizeof(mscal));

                    if (mode == Vu1ExecutionMode::ThreadedAsync &&
                        !waitUntil(
                            [&runtime]()
                            {
                                const auto statistics =
                                    runtime.vu1AsyncStatistics();
                                return statistics.pendingSlice &&
                                       statistics.owner.completedTickets ==
                                           statistics.owner.submittedTickets;
                            }))
                    {
                        t.Fail("the initial private slice should finish early");
                        return {};
                    }

                    constexpr uint32_t kVif1 = 0x10009000u;
                    constexpr uint32_t kSource = 0x00030000u;
                    const std::array<uint32_t, 4u> vifCommands = {
                        makeVifCommand(0x01u, 0u, 0x0101u),
                        makeVifCommand(0x02u, 0u, 3u),
                        makeVifCommand(0x05u, 0u, 1u),
                        makeVifCommand(0x01u, 0u, 0x0202u),
                    };
                    std::memcpy(
                        runtime.memory().getRDRAM() + kSource,
                        vifCommands.data(), sizeof(vifCommands));
                    if (!runtime.memory().writeIORegister(
                            kVif1 + 0x10u, kSource) ||
                        !runtime.memory().writeIORegister(
                            kVif1 + 0x20u, 1u) ||
                        !runtime.memory().writeIORegister(
                            kVif1, 0x100u))
                    {
                        t.Fail("the VIF1 hazard-batch DMA should start");
                        return {};
                    }

                    R5900Context &context = runtime.cpu();
                    context.advanceEeCycleTicks(32u);
                    runtime.serviceEeEventsAtBlockBoundary(
                        runtime.memory().getRDRAM(), &context);
                    const Vu1AsyncRuntimeStatistics afterPayload =
                        runtime.vu1AsyncStatistics();

                    if (mode == Vu1ExecutionMode::ThreadedAsync &&
                        !waitUntil(
                            [&runtime]()
                            {
                                const auto statistics =
                                    runtime.vu1AsyncStatistics();
                                return statistics.pendingSlice &&
                                       statistics.owner.completedTickets ==
                                           statistics.owner.submittedTickets;
                            }))
                    {
                        t.Fail("the once-requeued slice should finish early");
                        return {};
                    }

                    // The payload schedules a finalization-only VIF1 service
                    // two EE cycles later. It must not disturb speculation.
                    context.advanceEeCycleTicks(16u);
                    runtime.serviceEeEventsAtBlockBoundary(
                        runtime.memory().getRDRAM(), &context);
                    const Vu1AsyncRuntimeStatistics afterEmptyService =
                        runtime.vu1AsyncStatistics();

                    context.advanceEeCycleTicks(80u);
                    runtime.serviceEeEventsAtBlockBoundary(
                        runtime.memory().getRDRAM(), &context);
                    const Vu1AsyncRuntimeStatistics afterPublication =
                        runtime.vu1AsyncStatistics();
                    const std::shared_ptr<const Vu1Snapshot> snapshot =
                        runtime.snapshotVu1Owner();
                    return RunResult{
                        .architecturalHash =
                            vu1ArchitecturalStateHash(
                                snapshot->state,
                                snapshot->microMemory,
                                snapshot->dataMemory,
                                snapshot->vif,
                                snapshot->codeGeneration),
                        .afterPayload = afterPayload,
                        .afterEmptyService = afterEmptyService,
                        .afterPublication = afterPublication,
                    };
                };

            const RunResult reference =
                run(Vu1ExecutionMode::Inline);
            const RunResult asynchronous =
                run(Vu1ExecutionMode::ThreadedAsync);
            t.Equals(
                asynchronous.architecturalHash,
                reference.architecturalHash,
                "batched VIF1 hazards should retain inline architecture");
            t.Equals(
                asynchronous.afterPayload.hazardBarrierCount, 1ull,
                "one parser batch should resolve speculation once");
            t.Equals(
                asynchronous.afterPayload.hazardRequeueCount, 1ull,
                "one parser batch should requeue speculation once");
            t.Equals(
                asynchronous.afterPayload.slicesSubmitted, 2ull,
                "four owner commands should produce only one recomputed slice");
            t.Equals(
                asynchronous.afterPayload.owner.speculation.rolledBackSlices,
                1ull,
                "the parser batch should roll back one private checkpoint");
            t.Equals(
                asynchronous.afterEmptyService.hazardBarrierCount,
                asynchronous.afterPayload.hazardBarrierCount,
                "a VIF1 service without owner commands must not add a hazard");
            t.Equals(
                asynchronous.afterEmptyService.slicesSubmitted,
                asynchronous.afterPayload.slicesSubmitted,
                "a finalization-only VIF1 service must keep the pending slice");
            t.Equals(
                asynchronous.afterPublication.slicesPublished, 1ull,
                "the recomputed startup slice should publish exactly once");
            t.Equals(
                asynchronous.afterPublication.budgetFallbackCount, 0ull,
                "the on-time batch should not require budget fallback");
        });

        tc.Run("runtime abandons a VIF1 batch requeue after owner failure", [](TestCase &t)
        {
            std::atomic<bool> injectFailure{false};
            std::atomic<uint32_t> stateUpdates{0u};
            PS2RuntimeConfiguration configuration =
                defaultPs2RuntimeConfiguration();
            configuration.vu1ExecutionMode =
                Vu1ExecutionMode::ThreadedAsync;
            configuration.vu1CommandQueueCapacity = 1u;
            configuration.vu1CommandPayloadCapacityBytes =
                1024u * 1024u;
            configuration.vu1BeforeProcess =
                [&](const Vu1Command &command, uint64_t)
                {
                    if (!injectFailure.load(
                            std::memory_order_acquire) ||
                        command.type !=
                            Vu1CommandType::VifStateUpdate)
                    {
                        return;
                    }
                    if (stateUpdates.fetch_add(
                            1u, std::memory_order_relaxed) == 1u)
                    {
                        throw std::runtime_error(
                            "injected VIF1 batch owner failure");
                    }
                };
            PS2Runtime runtime(configuration);
            t.IsTrue(runtime.memory().initialize(),
                     "failure-batch runtime memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "failure-batch runtime subsystems should bind");

            writeRuntimeVu1InstructionPair(
                runtime, 0u,
                makeVuIaddiu(1u, 0u, 7),
                kVuUpperNop);
            writeRuntimeVu1InstructionPair(
                runtime, 8u, 0u, kVuUpperEnd);
            writeRuntimeVu1InstructionPair(
                runtime, 16u, 0u, kVuUpperNop);
            const uint32_t mscal =
                makeVifCommand(0x14u, 0u, 0u);
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(&mscal),
                sizeof(mscal));
            t.IsTrue(
                waitUntil(
                    [&runtime]()
                    {
                        const auto statistics =
                            runtime.vu1AsyncStatistics();
                        return statistics.pendingSlice &&
                               statistics.owner.completedTickets ==
                                   statistics.owner.submittedTickets;
                    }),
                "the failure fixture should begin with ready private work");

            constexpr uint32_t kVif1 = 0x10009000u;
            constexpr uint32_t kSource = 0x00030000u;
            const std::array<uint32_t, 4u> vifCommands = {
                makeVifCommand(0x01u, 0u, 0x0101u),
                makeVifCommand(0x02u, 0u, 3u),
                makeVifCommand(0x05u, 0u, 1u),
                makeVifCommand(0x01u, 0u, 0x0202u),
            };
            std::memcpy(
                runtime.memory().getRDRAM() + kSource,
                vifCommands.data(), sizeof(vifCommands));
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1 + 0x10u, kSource) &&
                         runtime.memory().writeIORegister(
                             kVif1 + 0x20u, 1u) &&
                         runtime.memory().writeIORegister(
                             kVif1, 0x100u),
                     "the failure-batch DMA should start");

            injectFailure.store(true, std::memory_order_release);
            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(32u);
            const std::string serviceError =
                captureException(
                    [&]()
                    {
                        runtime.serviceEeEventsAtBlockBoundary(
                            runtime.memory().getRDRAM(), &context);
                    });
            t.IsTrue(
                serviceError.find(
                    "injected VIF1 batch owner failure") !=
                    std::string::npos,
                "the parser service should surface the owner failure");

            t.IsTrue(
                waitUntil(
                    [&runtime]()
                    {
                        return runtime.vu1AsyncStatistics().owner.failed;
                    }),
                "the owner transport should publish fatal closure");
            const Vu1AsyncRuntimeStatistics statistics =
                runtime.vu1AsyncStatistics();
            t.IsFalse(statistics.pendingSlice,
                      "failed batch cleanup must retain no private slice");
            t.Equals(statistics.hazardBarrierCount, 1ull,
                     "the failed parser batch should resolve one hazard");
            t.Equals(statistics.hazardRequeueCount, 0ull,
                     "the failed parser batch must not requeue work");
            t.IsTrue(statistics.owner.failed,
                     "the owner transport should retain fatal closure");

            const std::string laterError =
                captureException(
                    [&]()
                    {
                        (void)runtime.snapshotVu1Owner();
                    });
            t.IsTrue(
                laterError.find(
                    "injected VIF1 batch owner failure") !=
                    std::string::npos,
                "later work should observe the retained owner failure");
            t.IsTrue(
                laterError.find("synchronous command batch") ==
                    std::string::npos,
                "failed cleanup must not leak an active batch state");
        });

        tc.Run("runtime waits for a late VU1 result without moving guest time", [](TestCase &t)
        {
            std::mutex gateMutex;
            std::condition_variable gateCv;
            bool entered = false;
            bool release = false;
            PS2RuntimeConfiguration configuration =
                defaultPs2RuntimeConfiguration();
            configuration.vu1ExecutionMode =
                Vu1ExecutionMode::ThreadedAsync;
            configuration.vu1CommandQueueCapacity = 2u;
            configuration.vu1CommandPayloadCapacityBytes =
                1024u * 1024u;
            configuration.vu1BeforeProcess =
                [&](const Vu1Command &command, uint64_t)
                {
                    if (command.type !=
                        Vu1CommandType::AdvanceSlice)
                    {
                        return;
                    }
                    std::unique_lock<std::mutex> lock(gateMutex);
                    if (entered)
                        return;
                    entered = true;
                    gateCv.notify_all();
                    gateCv.wait(
                        lock,
                        [&release]()
                        {
                            return release;
                        });
                };
            PS2Runtime runtime(configuration);
            t.IsTrue(runtime.memory().initialize(),
                     "late-result runtime memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "late-result runtime subsystems should bind");
            runtime.memory().write32(
                PS2_VU1_CODE_BASE,
                makeVuIaddiu(1u, 0u, 9));
            runtime.memory().write32(
                PS2_VU1_CODE_BASE + 4u,
                kVuUpperEnd);
            runtime.memory().write32(
                PS2_VU1_CODE_BASE + 8u, 0u);
            runtime.memory().write32(
                PS2_VU1_CODE_BASE + 12u,
                kVuUpperNop);
            const uint32_t mscal =
                makeVifCommand(0x14u, 0u, 0u);
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(&mscal),
                sizeof(mscal));
            t.IsTrue(
                waitFor(
                    gateCv, gateMutex,
                    [&entered]()
                    {
                        return entered;
                    }),
                "the owner should enter the delayed slice");

            std::thread releaser(
                [&]()
                {
                    std::this_thread::sleep_for(20ms);
                    {
                        std::lock_guard<std::mutex> lock(gateMutex);
                        release = true;
                    }
                    gateCv.notify_all();
                });
            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            releaser.join();

            const PS2Runtime::DebugVu1Timing timing =
                runtime.debugVu1TimingSnapshot();
            const std::shared_ptr<const Vu1Snapshot> snapshot =
                runtime.snapshotVu1Owner();
            t.Equals(snapshot->state.vi[1], int32_t{9},
                     "the late result should publish exact state");
            t.Equals(timing.currentTick, 128ull,
                     "host waiting must not add guest ticks");
            const Vu1AsyncRuntimeStatistics statistics =
                runtime.vu1AsyncStatistics();
            t.Equals(statistics.resultsReadyAtEvent, 0ull,
                     "the blocked owner should not be ready at the event");
            t.Equals(statistics.resultsLateAtEvent, 1ull,
                     "the blocked owner should be classified late");
            t.IsTrue(statistics.eventWaitNanoseconds >= 10'000'000ull,
                     "the event should report its forced host wait");
        });

        tc.Run("runtime recomputes a speculative slice at a late EE service boundary", [](TestCase &t)
        {
            struct RunResult
            {
                uint64_t architecturalHash = 0u;
                PS2Runtime::DebugVu1Timing timing{};
                Vu1AsyncRuntimeStatistics statistics{};
            };
            const auto run =
                [&](Vu1ExecutionMode mode,
                    uint64_t serviceTicks) -> RunResult
                {
                    PS2RuntimeConfiguration configuration =
                        defaultPs2RuntimeConfiguration();
                    configuration.vu1ExecutionMode = mode;
                    configuration.vu1CommandQueueCapacity = 1u;
                    configuration.vu1CommandPayloadCapacityBytes =
                        1024u * 1024u;
                    configuration.captureVu1ArchitecturalStateHashes =
                        true;
                    PS2Runtime runtime(configuration);
                    if (!runtime.memory().initialize() ||
                        !runtime.syncCoreSubsystems())
                    {
                        t.Fail("late-service runtime should initialize");
                        return {};
                    }

                    constexpr uint32_t kWorkPairs = 64u;
                    for (uint32_t pair = 0u;
                         pair < kWorkPairs; ++pair)
                    {
                        writeRuntimeVu1InstructionPair(
                            runtime, pair * 8u,
                            makeVuIaddiu(1u, 1u, 1),
                            pair + 1u == kWorkPairs
                                ? kVuUpperEnd
                                : kVuUpperNop);
                    }
                    writeRuntimeVu1InstructionPair(
                        runtime, kWorkPairs * 8u,
                        0u, kVuUpperNop);
                    const uint32_t mscal =
                        makeVifCommand(0x14u, 0u, 0u);
                    runtime.memory().processVIF1Data(
                        reinterpret_cast<const uint8_t *>(&mscal),
                        sizeof(mscal));

                    if (mode == Vu1ExecutionMode::ThreadedAsync &&
                        !waitUntil(
                            [&runtime]()
                            {
                                const auto statistics =
                                    runtime.vu1AsyncStatistics();
                                return statistics.pendingSlice &&
                                       statistics.owner.completedTickets ==
                                           statistics.owner.submittedTickets;
                            }))
                    {
                        t.Fail("the scheduled-budget slice should finish early");
                        return {};
                    }

                    R5900Context &context = runtime.cpu();
                    context.advanceEeCycleTicks(serviceTicks);
                    const std::shared_ptr<const Vu1Snapshot>
                        delayedSnapshot = runtime.snapshotVu1Owner();
                    if (!delayedSnapshot)
                    {
                        t.Fail("the overdue hazard should return a snapshot");
                        return {};
                    }
                    if (mode == Vu1ExecutionMode::ThreadedAsync &&
                        !waitUntil(
                            [&runtime]()
                            {
                                const auto statistics =
                                    runtime.vu1AsyncStatistics();
                                return statistics.pendingSlice &&
                                       statistics.owner.completedTickets ==
                                           statistics.owner.submittedTickets;
                            }))
                    {
                        t.Fail("the overdue requeued slice should finish early");
                        return {};
                    }
                    runtime.serviceEeEventsAtBlockBoundary(
                        runtime.memory().getRDRAM(), &context);
                    const std::shared_ptr<const Vu1Snapshot> snapshot =
                        runtime.snapshotVu1Owner();
                    return RunResult{
                        .architecturalHash =
                            vu1ArchitecturalStateHash(
                                snapshot->state,
                                snapshot->microMemory,
                                snapshot->dataMemory,
                                snapshot->vif,
                                snapshot->codeGeneration),
                        .timing = runtime.debugVu1TimingSnapshot(),
                        .statistics = runtime.vu1AsyncStatistics(),
                    };
                };

            const RunResult sameBudgetReference =
                run(Vu1ExecutionMode::Inline, 132u);
            const RunResult sameBudgetAsynchronous =
                run(Vu1ExecutionMode::ThreadedAsync, 132u);
            t.Equals(
                sameBudgetAsynchronous.architecturalHash,
                sameBudgetReference.architecturalHash,
                "a sub-cycle-late requeue should retain inline architecture");
            t.Equals(
                sameBudgetAsynchronous.timing.currentTick,
                sameBudgetReference.timing.currentTick,
                "a sub-cycle-late requeue should retain the canonical guest tick");
            t.Equals(
                sameBudgetAsynchronous.timing.totalAdvancedCycles,
                sameBudgetReference.timing.totalAdvancedCycles,
                "a sub-cycle-late requeue should retain the canonical VU cycles");
            t.Equals(
                sameBudgetAsynchronous.statistics.slicesPublished, 1ull,
                "the monotonic-identity slice should publish once");
            t.Equals(
                sameBudgetAsynchronous.statistics.budgetFallbackCount, 0ull,
                "sub-cycle lateness should not require recomputation");

            const RunResult reference =
                run(Vu1ExecutionMode::Inline, 136u);
            const RunResult asynchronous =
                run(Vu1ExecutionMode::ThreadedAsync, 136u);
            t.Equals(
                asynchronous.architecturalHash,
                reference.architecturalHash,
                "late service should recompute the exact inline architecture");
            t.Equals(
                asynchronous.timing.currentTick,
                reference.timing.currentTick,
                "late service should retain the canonical guest tick");
            t.Equals(
                asynchronous.timing.totalAdvancedCycles,
                reference.timing.totalAdvancedCycles,
                "late service should retain the canonical VU cycle count");
            t.Equals(
                asynchronous.statistics.slicesPublished, 1ull,
                "the fallback should still publish one guest event slice");
            t.Equals(
                asynchronous.statistics.budgetFallbackCount, 1ull,
                "the delayed boundary should use one budget fallback");
            t.Equals(
                asynchronous.statistics.budgetFallbackCycleDelta, 1ull,
                "eight late EE ticks should add one VU cycle");
            t.Equals(
                asynchronous.statistics.resultsReadyAtEvent, 1ull,
                "the discarded scheduled-budget result should be ready early");
            t.Equals(
                asynchronous.statistics.hazardBarrierCount, 2ull,
                "the overdue and result snapshots should resolve private slices");
            t.Equals(
                asynchronous.statistics.hazardRequeueCount, 2ull,
                "both snapshots should requeue their pending events");
            t.IsTrue(
                asynchronous.statistics.owner.speculation.rolledBackSlices >=
                    1u,
                "the incorrect scheduled-budget future must be rolled back");
        });

        tc.Run("runtime reset cancels an in-flight speculative VU1 slice", [](TestCase &t)
        {
            std::mutex gateMutex;
            std::condition_variable gateCv;
            bool entered = false;
            bool release = false;
            PS2RuntimeConfiguration configuration =
                defaultPs2RuntimeConfiguration();
            configuration.vu1ExecutionMode =
                Vu1ExecutionMode::ThreadedAsync;
            configuration.vu1CommandQueueCapacity = 1u;
            configuration.vu1CommandPayloadCapacityBytes =
                1024u * 1024u;
            configuration.vu1BeforeProcess =
                [&](const Vu1Command &command, uint64_t)
                {
                    if (command.type !=
                        Vu1CommandType::AdvanceSlice)
                    {
                        return;
                    }
                    std::unique_lock<std::mutex> lock(gateMutex);
                    if (entered)
                        return;
                    entered = true;
                    gateCv.notify_all();
                    gateCv.wait(
                        lock,
                        [&release]()
                        {
                            return release;
                        });
                };
            PS2Runtime runtime(configuration);
            t.IsTrue(runtime.memory().initialize(),
                     "pending-reset runtime memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "pending-reset runtime subsystems should bind");
            for (uint32_t pair = 0u; pair < 64u; ++pair)
            {
                writeRuntimeVu1InstructionPair(
                    runtime, pair * 8u,
                    makeVuIaddiu(1u, 1u, 1),
                    kVuUpperNop);
            }
            const uint32_t mscal =
                makeVifCommand(0x14u, 0u, 0u);
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(&mscal),
                sizeof(mscal));
            const PS2Runtime::DebugVu1Timing started =
                runtime.debugVu1TimingSnapshot();
            t.IsTrue(
                waitFor(
                    gateCv, gateMutex,
                    [&entered]()
                    {
                        return entered;
                    }),
                "the speculative slice should be owner-active");

            std::atomic<bool> resetDone{false};
            std::atomic<bool> resetStarted{false};
            bool resetAccepted = false;
            std::thread resetter(
                [&]()
                {
                    resetStarted.store(
                        true, std::memory_order_release);
                    resetAccepted = runtime.memory().writeIORegister(
                        0x10003c10u, 1u);
                    resetDone.store(
                        true, std::memory_order_release);
                });
            t.IsTrue(
                waitUntil(
                    [&resetStarted]()
                    {
                        return resetStarted.load(
                            std::memory_order_acquire);
                    }),
                "the reset producer should start");
            std::this_thread::sleep_for(5ms);
            t.IsFalse(
                resetDone.load(std::memory_order_acquire),
                "reset must wait for the owner to reach a rollback boundary");
            {
                std::lock_guard<std::mutex> lock(gateMutex);
                release = true;
            }
            gateCv.notify_all();
            resetter.join();

            t.IsTrue(resetAccepted,
                     "VIF1 FBRST should accept the reset");
            const PS2Runtime::DebugVu1Timing reset =
                runtime.debugVu1TimingSnapshot();
            t.IsFalse(reset.active,
                      "reset should clear the published VU1 busy state");
            t.IsFalse(reset.eventPending,
                      "reset should cancel the old scheduler token");
            t.IsTrue(reset.generation > started.generation,
                     "reset should invalidate the old execution generation");
            const Vu1AsyncRuntimeStatistics statistics =
                runtime.vu1AsyncStatistics();
            t.IsFalse(statistics.pendingSlice,
                      "reset should retain no speculative work");
            t.Equals(statistics.slicesPublished, 0ull,
                     "reset must not publish the discarded slice");
            t.Equals(statistics.hazardBarrierCount, 1ull,
                     "reset should cross one pending-slice barrier");
            t.Equals(
                statistics.owner.speculation.rolledBackSlices,
                1ull,
                "the owner should roll back the in-flight slice once");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            const std::shared_ptr<const Vu1Snapshot> snapshot =
                runtime.snapshotVu1Owner();
            t.IsFalse(snapshot->state.active,
                      "the stale event must not reactivate VU1");
            t.Equals(snapshot->state.vi[1], int32_t{0},
                     "discarded speculative arithmetic must remain private");
        });

        tc.Run("runtime reset resolves ready published and follow-up speculation", [](TestCase &t)
        {
            enum class ResetPhase : uint8_t
            {
                ReadyUnpublished,
                PublishedComplete,
                FollowupPending,
            };
            for (const ResetPhase phase : {
                     ResetPhase::ReadyUnpublished,
                     ResetPhase::PublishedComplete,
                     ResetPhase::FollowupPending})
            {
                PS2RuntimeConfiguration configuration =
                    defaultPs2RuntimeConfiguration();
                configuration.vu1ExecutionMode =
                    Vu1ExecutionMode::ThreadedAsync;
                configuration.vu1CommandQueueCapacity =
                    phase == ResetPhase::FollowupPending
                        ? 2u
                        : 1u;
                configuration.vu1CommandPayloadCapacityBytes =
                    1024u * 1024u;
                PS2Runtime runtime(configuration);
                t.IsTrue(runtime.memory().initialize(),
                         "reset-phase runtime memory should initialize");
                t.IsTrue(runtime.syncCoreSubsystems(),
                         "reset-phase runtime subsystems should bind");

                const bool completesAtStartup =
                    phase == ResetPhase::PublishedComplete;
                const uint32_t pairCount =
                    completesAtStartup ? 2u : 64u;
                for (uint32_t pair = 0u;
                     pair < pairCount; ++pair)
                {
                    writeRuntimeVu1InstructionPair(
                        runtime, pair * 8u,
                        makeVuIaddiu(1u, 1u, 1),
                        completesAtStartup && pair == 0u
                            ? kVuUpperEnd
                            : kVuUpperNop);
                }
                const uint32_t mscal =
                    makeVifCommand(0x14u, 0u, 0u);
                runtime.memory().processVIF1Data(
                    reinterpret_cast<const uint8_t *>(&mscal),
                    sizeof(mscal));

                if (phase == ResetPhase::ReadyUnpublished)
                {
                    t.IsTrue(
                        waitUntil(
                            [&runtime]()
                            {
                                const auto statistics =
                                    runtime.vu1AsyncStatistics();
                                return statistics.pendingSlice &&
                                       statistics.owner.completedTickets ==
                                           statistics.owner.submittedTickets;
                            }),
                        "the pre-event result should become ready");
                }
                else
                {
                    R5900Context &context = runtime.cpu();
                    context.advanceEeCycleTicks(128u);
                    runtime.serviceEeEventsAtBlockBoundary(
                        runtime.memory().getRDRAM(), &context);
                    if (phase == ResetPhase::FollowupPending)
                    {
                        t.IsTrue(runtime.vu1AsyncStatistics().pendingSlice,
                                 "the first event should enqueue its follow-up");
                    }
                }

                t.IsTrue(runtime.memory().writeIORegister(
                             0x10003c10u, 1u),
                         "VIF1 FBRST should accept each reset phase");
                const auto timing =
                    runtime.debugVu1TimingSnapshot();
                const auto statistics =
                    runtime.vu1AsyncStatistics();
                t.IsFalse(timing.active,
                          "each reset phase should clear VU1 busy");
                t.IsFalse(timing.eventPending,
                          "each reset phase should cancel its event");
                t.IsFalse(statistics.pendingSlice,
                          "each reset phase should retire pending transport");
                const uint64_t expectedPublished =
                    phase == ResetPhase::ReadyUnpublished ? 0u : 1u;
                const uint64_t expectedCommitted =
                    phase == ResetPhase::ReadyUnpublished ? 0u : 1u;
                const uint64_t expectedRolledBack =
                    phase == ResetPhase::PublishedComplete ? 0u : 1u;
                t.Equals(statistics.slicesPublished, expectedPublished,
                         "reset should preserve only prior event publication");
                t.Equals(
                    statistics.owner.speculation.committedSlices,
                    expectedCommitted,
                    "reset should commit only an already-published slice");
                t.Equals(
                    statistics.owner.speculation.rolledBackSlices,
                    expectedRolledBack,
                    "reset should discard only an unpublished slice");
                const std::shared_ptr<const Vu1Snapshot> snapshot =
                    runtime.snapshotVu1Owner();
                t.IsFalse(snapshot->state.active,
                          "reset snapshot should remain inactive");
                t.Equals(snapshot->state.vi[1], int32_t{0},
                         "reset should clear all published or private work");
            }
        });

        tc.Run("runtime shutdown resolves a pending speculative VU1 slice", [](TestCase &t)
        {
            auto advanceAttempts =
                std::make_shared<std::atomic<uint64_t>>(0u);
            const auto startedAt =
                std::chrono::steady_clock::now();
            {
                PS2RuntimeConfiguration configuration =
                    defaultPs2RuntimeConfiguration();
                configuration.vu1ExecutionMode =
                    Vu1ExecutionMode::ThreadedAsync;
                configuration.vu1CommandQueueCapacity = 1u;
                configuration.vu1CommandPayloadCapacityBytes =
                    1024u * 1024u;
                configuration.vu1BeforeProcess =
                    [advanceAttempts](
                        const Vu1Command &command, uint64_t)
                    {
                        if (command.type !=
                            Vu1CommandType::AdvanceSlice)
                        {
                            return;
                        }
                        advanceAttempts->fetch_add(
                            1u, std::memory_order_relaxed);
                        std::this_thread::sleep_for(20ms);
                    };
                PS2Runtime runtime(configuration);
                t.IsTrue(runtime.memory().initialize(),
                         "shutdown runtime memory should initialize");
                t.IsTrue(runtime.syncCoreSubsystems(),
                         "shutdown runtime subsystems should bind");
                for (uint32_t pair = 0u; pair < 32u; ++pair)
                {
                    writeRuntimeVu1InstructionPair(
                        runtime, pair * 8u, 0u,
                        kVuUpperNop);
                }
                const uint32_t mscal =
                    makeVifCommand(0x14u, 0u, 0u);
                runtime.memory().processVIF1Data(
                    reinterpret_cast<const uint8_t *>(&mscal),
                    sizeof(mscal));
                t.IsTrue(
                    waitUntil(
                        [&advanceAttempts]()
                        {
                            return advanceAttempts->load(
                                       std::memory_order_acquire) != 0u;
                        }),
                    "destruction should begin with owner-active speculation");
            }
            const auto elapsed =
                std::chrono::steady_clock::now() - startedAt;
            t.IsTrue(
                elapsed < 2s,
                "pending shutdown should resolve without a lifecycle hang");
            t.Equals(advanceAttempts->load(std::memory_order_acquire), 1ull,
                     "shutdown should not requeue discarded future work");
        });

        tc.Run("runtime chains MSCAL MSCNT and VIF waits through async events", [](TestCase &t)
        {
            PS2RuntimeConfiguration configuration =
                defaultPs2RuntimeConfiguration();
            configuration.vu1ExecutionMode =
                Vu1ExecutionMode::ThreadedAsync;
            configuration.vu1CommandQueueCapacity = 1u;
            configuration.vu1CommandPayloadCapacityBytes =
                1024u * 1024u;
            PS2Runtime runtime(configuration);
            t.IsTrue(runtime.memory().initialize(),
                     "chained-VIF runtime memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "chained-VIF runtime subsystems should bind");

            const auto makeVuBranch = [](int16_t immediate)
            {
                return (0x20u << 25u) |
                       (static_cast<uint32_t>(immediate) & 0x07ffu);
            };
            writeRuntimeVu1InstructionPair(
                runtime, 0u, makeVuBranch(2), kVuUpperEnd);
            writeRuntimeVu1InstructionPair(
                runtime, 8u,
                makeVuIaddiu(1u, 0u, 1),
                kVuUpperNop);
            writeRuntimeVu1InstructionPair(
                runtime, 24u,
                makeVuIaddiu(2u, 0u, 7),
                kVuUpperEnd);
            writeRuntimeVu1InstructionPair(
                runtime, 32u, 0u, kVuUpperNop);

            const std::array<uint32_t, 5u> commands = {
                makeVifCommand(0x14u, 0u, 0u),
                makeVifCommand(0x11u, 0u, 0u),
                makeVifCommand(0x17u, 0u, 0u),
                makeVifCommand(0x11u, 0u, 0u),
                makeVifCommand(0x05u, 0u, 3u),
            };
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(commands.data()),
                static_cast<uint32_t>(sizeof(commands)));
            t.IsTrue(runtime.memory().vif1WaitingForVu(),
                     "the first FLUSH should defer MSCNT and STMOD");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            const PS2Runtime::DebugVu1Timing resumed =
                runtime.debugVu1TimingSnapshot();
            const std::shared_ptr<const Vu1Snapshot> first =
                runtime.snapshotVu1Owner();
            t.IsTrue(resumed.active && resumed.eventPending,
                     "MSCNT should schedule a fresh async startup slice");
            t.IsTrue(runtime.memory().vif1WaitingForVu(),
                     "the second FLUSH should retain the wait");
            t.Equals(first->state.vi[1], int32_t{1},
                     "the first segment should publish its delay slot");
            t.Equals(first->state.vi[2], int32_t{0},
                     "the resumed segment must remain private until its event");
            t.Equals(runtime.memory().vif1_regs.mode, 0u,
                     "STMOD should remain deferred behind MSCNT");

            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            const std::shared_ptr<const Vu1Snapshot> second =
                runtime.snapshotVu1Owner();
            t.IsFalse(second->state.active,
                      "the resumed E-bit segment should finish");
            t.Equals(second->state.vi[2], int32_t{7},
                     "MSCNT should resume at the canonical branch target");
            t.IsFalse(runtime.memory().vif1WaitingForVu(),
                      "second completion should wake VIF1");
            t.Equals(runtime.memory().vif1_regs.mode, 3u,
                     "the deferred STMOD should execute last");
            t.Equals(runtime.vu1AsyncStatistics().slicesPublished, 2ull,
                     "both VU segments should publish at their own events");
        });

        tc.Run("runtime publishes multiple XGKICKs in slice and event order", [](TestCase &t)
        {
            const auto run =
                [&](uint32_t secondKickPair,
                    size_t packetsAfterStartup)
                {
                    PS2RuntimeConfiguration configuration =
                        defaultPs2RuntimeConfiguration();
                    configuration.vu1ExecutionMode =
                        Vu1ExecutionMode::ThreadedAsync;
                    configuration.vu1CommandQueueCapacity = 2u;
                    configuration.vu1CommandPayloadCapacityBytes =
                        1024u * 1024u;
                    PS2Runtime runtime(configuration);
                    if (!runtime.memory().initialize() ||
                        !runtime.syncCoreSubsystems())
                    {
                        t.Fail("multi-XGKICK runtime should initialize");
                        return;
                    }

                    std::vector<std::vector<uint8_t>> captured;
                    runtime.gifArbiter().setProcessPacketFn(
                        [&](const uint8_t *data, uint32_t sizeBytes)
                        {
                            captured.emplace_back(
                                data, data + sizeBytes);
                        });
                    std::array<uint8_t, 16u> firstPacket{};
                    std::array<uint8_t, 16u> secondPacket{};
                    const uint64_t eopImageTag =
                        (1ull << 15u) | (2ull << 58u);
                    const uint64_t firstMarker = 0x1111u;
                    const uint64_t secondMarker = 0x2222u;
                    std::memcpy(
                        firstPacket.data(), &eopImageTag,
                        sizeof(eopImageTag));
                    std::memcpy(
                        firstPacket.data() + 8u, &firstMarker,
                        sizeof(firstMarker));
                    std::memcpy(
                        secondPacket.data(), &eopImageTag,
                        sizeof(eopImageTag));
                    std::memcpy(
                        secondPacket.data() + 8u, &secondMarker,
                        sizeof(secondMarker));
                    writeRuntimeVu1Qword(runtime, 0u, firstPacket);
                    writeRuntimeVu1Qword(runtime, 4u * 16u, secondPacket);

                    for (uint32_t pair = 0u;
                         pair <= secondKickPair + 2u; ++pair)
                    {
                        writeRuntimeVu1InstructionPair(
                            runtime, pair * 8u, 0u,
                            kVuUpperNop);
                    }
                    writeRuntimeVu1InstructionPair(
                        runtime, 0u,
                        makeVuIaddiu(1u, 0u, 0),
                        kVuUpperNop);
                    writeRuntimeVu1InstructionPair(
                        runtime, 1u * 8u,
                        makeVuLowerSpecial(0x6cu, 1u),
                        kVuUpperNop);
                    writeRuntimeVu1InstructionPair(
                        runtime, (secondKickPair - 1u) * 8u,
                        makeVuIaddiu(1u, 0u, 4),
                        kVuUpperNop);
                    writeRuntimeVu1InstructionPair(
                        runtime, secondKickPair * 8u,
                        makeVuLowerSpecial(0x6cu, 1u),
                        kVuUpperEnd);
                    writeRuntimeVu1InstructionPair(
                        runtime, (secondKickPair + 1u) * 8u,
                        0u, kVuUpperNop);

                    const uint32_t mscal =
                        makeVifCommand(0x14u, 0u, 0u);
                    runtime.memory().processVIF1Data(
                        reinterpret_cast<const uint8_t *>(&mscal),
                        sizeof(mscal));
                    if (!waitUntil(
                            [&runtime]()
                            {
                                const auto statistics =
                                    runtime.vu1AsyncStatistics();
                                return statistics.owner.completedTickets ==
                                       statistics.owner.submittedTickets;
                            }))
                    {
                        t.Fail("startup XGKICK slice should become ready");
                        return;
                    }
                    t.Equals(captured.size(), size_t{0u},
                             "ready XGKICK packets must remain private");

                    R5900Context &context = runtime.cpu();
                    context.advanceEeCycleTicks(128u);
                    runtime.serviceEeEventsAtBlockBoundary(
                        runtime.memory().getRDRAM(), &context);
                    t.Equals(
                        captured.size(), packetsAfterStartup,
                        "startup event should publish only its earned packets");
                    if (captured.size() < 2u)
                    {
                        const auto timing =
                            runtime.debugVu1TimingSnapshot();
                        const uint64_t delta =
                            timing.eventDeadlineTick -
                            timing.currentTick;
                        context.advanceEeCycleTicks(delta);
                        runtime.serviceEeEventsAtBlockBoundary(
                            runtime.memory().getRDRAM(), &context);
                    }
                    t.Equals(captured.size(), size_t{2u},
                             "the complete program should publish both packets");
                    if (captured.size() == 2u)
                    {
                        t.Equals(
                            captured[0],
                            std::vector<uint8_t>(
                                firstPacket.begin(),
                                firstPacket.end()),
                            "the first kick should retain its source bytes");
                        t.Equals(
                            captured[1],
                            std::vector<uint8_t>(
                                secondPacket.begin(),
                                secondPacket.end()),
                            "the second kick should follow in VU order");
                    }
                };

            run(3u, 2u);
            run(18u, 1u);
        });

        tc.Run("runtime async checkpoints match inline under ready late and jitter schedules", [](TestCase &t)
        {
            struct Checkpoint
            {
                uint64_t architecturalHash = 0u;
                uint64_t eventTick = 0u;
                uint64_t advancedCycles = 0u;
                uint32_t pc = 0u;
                bool active = false;

                bool operator==(
                    const Checkpoint &) const = default;
            };
            struct RunResult
            {
                std::vector<Checkpoint> checkpoints;
                Vu1AsyncRuntimeStatistics statistics{};
            };
            const auto run =
                [&](Vu1ExecutionMode mode,
                    size_t queueCapacity,
                    std::vector<std::chrono::microseconds> delays,
                    bool waitForReady) -> RunResult
                {
                    std::atomic<size_t> sliceAttempt{0u};
                    PS2RuntimeConfiguration configuration =
                        defaultPs2RuntimeConfiguration();
                    configuration.vu1ExecutionMode = mode;
                    configuration.vu1CommandQueueCapacity =
                        queueCapacity;
                    configuration.vu1CommandPayloadCapacityBytes =
                        1024u * 1024u;
                    configuration.captureVu1ArchitecturalStateHashes =
                        true;
                    configuration.vu1BeforeProcess =
                        [&](const Vu1Command &command, uint64_t)
                        {
                            if (command.type !=
                                    Vu1CommandType::AdvanceSlice ||
                                delays.empty())
                            {
                                return;
                            }
                            const size_t index =
                                sliceAttempt.fetch_add(
                                    1u,
                                    std::memory_order_relaxed);
                            std::this_thread::sleep_for(
                                delays[index % delays.size()]);
                        };
                    PS2Runtime runtime(configuration);
                    if (!runtime.memory().initialize() ||
                        !runtime.syncCoreSubsystems())
                    {
                        t.Fail("delay-matrix runtime should initialize");
                        return {};
                    }

                    constexpr uint32_t kWorkPairs = 170u;
                    for (uint32_t pair = 0u;
                         pair < kWorkPairs; ++pair)
                    {
                        writeRuntimeVu1InstructionPair(
                            runtime, pair * 8u,
                            makeVuIaddiu(1u, 1u, 1),
                            pair + 1u == kWorkPairs
                                ? kVuUpperEnd
                                : kVuUpperNop);
                    }
                    writeRuntimeVu1InstructionPair(
                        runtime, kWorkPairs * 8u,
                        0u, kVuUpperNop);
                    const uint32_t mscal =
                        makeVifCommand(0x14u, 0u, 0u);
                    runtime.memory().processVIF1Data(
                        reinterpret_cast<const uint8_t *>(&mscal),
                        sizeof(mscal));

                    RunResult result;
                    R5900Context &context = runtime.cpu();
                    for (uint32_t event = 0u;
                         event < 8u; ++event)
                    {
                        PS2Runtime::DebugVu1Timing before =
                            runtime.debugVu1TimingSnapshot();
                        if (!before.eventPending)
                            break;
                        if (waitForReady &&
                            mode == Vu1ExecutionMode::ThreadedAsync)
                        {
                            if (!waitUntil(
                                    [&runtime]()
                                    {
                                        const auto statistics =
                                            runtime.vu1AsyncStatistics();
                                        return statistics.pendingSlice &&
                                               statistics.owner.completedTickets ==
                                                   statistics.owner.submittedTickets;
                                    }))
                            {
                                t.Fail("forced-early slice should become ready");
                                return result;
                            }
                        }
                        const uint64_t delta =
                            before.eventDeadlineTick -
                            before.currentTick;
                        context.advanceEeCycleTicks(delta);
                        runtime.serviceEeEventsAtBlockBoundary(
                            runtime.memory().getRDRAM(), &context);
                        const PS2Runtime::DebugVu1Timing after =
                            runtime.debugVu1TimingSnapshot();
                        const std::shared_ptr<const Vu1Snapshot> snapshot =
                            runtime.snapshotVu1Owner();
                        result.checkpoints.push_back(
                            Checkpoint{
                                .architecturalHash =
                                    vu1ArchitecturalStateHash(
                                        snapshot->state,
                                        snapshot->microMemory,
                                        snapshot->dataMemory,
                                        snapshot->vif,
                                        snapshot->codeGeneration),
                                .eventTick = after.currentTick,
                                .advancedCycles =
                                    after.totalAdvancedCycles,
                                .pc = snapshot->state.pc,
                                .active = snapshot->state.active,
                            });
                        if (!after.active)
                            break;
                    }
                    result.statistics =
                        runtime.vu1AsyncStatistics();
                    return result;
                };

            const RunResult reference = run(
                Vu1ExecutionMode::Inline, 1u, {}, false);
            t.Equals(reference.checkpoints.size(), size_t{3u},
                     "the oracle should cross three scheduler events");
            const auto compare =
                [&](const RunResult &candidate,
                    const char *description)
                {
                    t.Equals(
                        candidate.checkpoints.size(),
                        reference.checkpoints.size(),
                        std::string(description) +
                            " should retain every event boundary");
                    const size_t count = std::min(
                        candidate.checkpoints.size(),
                        reference.checkpoints.size());
                    for (size_t index = 0u;
                         index < count; ++index)
                    {
                        t.IsTrue(
                            candidate.checkpoints[index] ==
                                reference.checkpoints[index],
                            std::string(description) +
                                " should match checkpoint " +
                                std::to_string(index));
                    }
                };

            const RunResult early = run(
                Vu1ExecutionMode::ThreadedAsync,
                1u, {0us}, true);
            compare(early, "forced-early capacity-one execution");
            t.Equals(
                early.statistics.resultsReadyAtEvent,
                early.statistics.slicesPublished,
                "every forced-early result should be ready at publication");
            t.Equals(early.statistics.resultsLateAtEvent, 0ull,
                     "forced-early execution should have no late result");
            t.Equals(early.statistics.owner.queueCapacity, size_t{1u},
                     "the early arm should exercise capacity one");

            const RunResult late = run(
                Vu1ExecutionMode::ThreadedAsync,
                2u, {10ms}, false);
            compare(late, "forced-late capacity-two execution");
            t.Equals(
                late.statistics.resultsLateAtEvent,
                late.statistics.slicesPublished,
                "every delayed result should be late at publication");
            t.Equals(late.statistics.resultsReadyAtEvent, 0ull,
                     "forced-late execution should have no ready result");
            t.Equals(late.statistics.owner.queueCapacity, size_t{2u},
                     "the late arm should exercise capacity two");

            const RunResult jitter = run(
                Vu1ExecutionMode::ThreadedAsync,
                1u,
                {0us, 250us, 20ms, 1ms, 0us, 5ms},
                false);
            compare(jitter, "fixed-seed jitter execution");
            t.Equals(jitter.statistics.slicesPublished, 3ull,
                     "jitter should publish the unchanged event count");
        });
    });
}
