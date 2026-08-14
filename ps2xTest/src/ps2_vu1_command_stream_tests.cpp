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
    });
}
