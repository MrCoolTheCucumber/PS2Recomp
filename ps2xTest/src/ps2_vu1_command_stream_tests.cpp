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
                    inlineFixture.submit(Vu1AdvanceSliceCommand{
                        .maximumCycles = budget,
                        .captureState = true,
                    });
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
                    slice->state.has_value() &&
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
                fixture.submit(Vu1AdvanceSliceCommand{
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
            fixture.submit(Vu1VifStateUpdateCommand{vif});
            std::vector<uint8_t> source(32u);
            const std::array<uint32_t, 8u> words{
                1u, 2u, 3u, 4u,
                5u, 6u, 7u, 8u,
            };
            std::memcpy(source.data(), words.data(), source.size());
            const std::vector<uint8_t> owned = source;
            const Vu1CommandResult result = fixture.submit(
                Vu1DecodedUnpackCommand{
                    .immediate = 3u,
                    .vectorLength = 0u,
                    .componentCount = 4u,
                    .writeVectorCount = 2u,
                    .sourceVectorCount = 2u,
                    .sourceWordAlignment = 4u,
                    .bytes = std::move(source),
                });
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
            t.IsTrue(snapshot != nullptr,
                     "snapshot should return an owned typed result");
            if (!snapshot)
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
