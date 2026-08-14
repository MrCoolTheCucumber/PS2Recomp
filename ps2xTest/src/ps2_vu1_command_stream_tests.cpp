#include "MiniTest.h"

#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu1_command_stream.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
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
    });
}
