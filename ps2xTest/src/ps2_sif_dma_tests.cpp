#include "MiniTest.h"
#include "ps2_runtime.h"
#include "ps2_iop_transport.h"
#include "ps2_syscalls.h"
#include "ps2_stubs.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    constexpr int KE_OK = 0;

    struct TestEnv
    {
        std::vector<uint8_t> rdram;
        R5900Context ctx{};
        PS2Runtime runtime;

        TestEnv() : rdram(PS2_RAM_SIZE, 0u)
        {
            ps2_stubs::resetSifState(&runtime);
            std::memset(&ctx, 0, sizeof(ctx));
            if (!runtime.memory().initialize())
            {
                throw std::runtime_error("failed to initialize runtime memory");
            }
        }
    };

    void configureProfile(TestEnv &env, std::string_view elfName)
    {
        std::string error;
        const bool configured = PS2IopTransport::configureForTesting(
            &env.runtime, {std::string(elfName), 0u, 0u}, &error);
        if (!configured)
        {
            throw std::runtime_error("failed to configure test IOP profile: " + error);
        }
    }

    #pragma pack(push, 1)
    struct Ps2SifDmaTransfer
    {
        uint32_t src;
        uint32_t dest;
        int32_t size;
        int32_t attr;
    };

    struct SifRpcHeader
    {
        uint32_t pkt_addr;
        uint32_t rpc_id;
        int32_t sema_id;
        uint32_t mode;
    };

    struct SifRpcReceiveData
    {
        SifRpcHeader hdr;
        uint32_t src;
        uint32_t dest;
        int32_t size;
    };
    #pragma pack(pop)

    static_assert(sizeof(Ps2SifDmaTransfer) == 16u, "Unexpected Ps2SifDmaTransfer size.");
    static_assert(sizeof(SifRpcReceiveData) == 28u, "Unexpected SifRpcReceiveData size.");

    void setRegU32(R5900Context &ctx, int reg, uint32_t value)
    {
        ctx.r[reg] = _mm_set_epi64x(0, static_cast<int64_t>(value));
    }

    int32_t getRegS32(const R5900Context &ctx, int reg)
    {
        return static_cast<int32_t>(::getRegU32(&ctx, reg));
    }

    void writeGuestU32(uint8_t *rdram, uint32_t addr, uint32_t value)
    {
        std::memcpy(rdram + addr, &value, sizeof(value));
    }

    uint32_t readGuestU32(const uint8_t *rdram, uint32_t addr)
    {
        uint32_t value = 0;
        std::memcpy(&value, rdram + addr, sizeof(value));
        return value;
    }

    int32_t submitSifDma(
        TestEnv &env,
        uint32_t descriptorAddress,
        uint32_t descriptorCount)
    {
        setRegU32(env.ctx, 4, descriptorAddress);
        setRegU32(env.ctx, 5, descriptorCount);
        ps2_stubs::sceSifSetDma(
            env.rdram.data(), &env.ctx, &env.runtime);
        return getRegS32(env.ctx, 2);
    }

    int32_t sifDmaStatus(
        TestEnv &env, uint32_t transferId)
    {
        setRegU32(env.ctx, 4, transferId);
        ps2_stubs::sceSifDmaStat(
            env.rdram.data(), &env.ctx, &env.runtime);
        return getRegS32(env.ctx, 2);
    }

    void advanceEventTime(
        TestEnv &env, uint32_t rawEeTicks)
    {
        env.ctx.cop0_config |= 1u << 18u;
        env.ctx.advanceEeCycleTicks(rawEeTicks);
        env.runtime.serviceEeEventsAtBlockBoundary(
            env.rdram.data(), &env.ctx);
    }

    bool serviceAllScheduledBoundaries(
        TestEnv &env,
        size_t maximumBoundaries = 64u)
    {
        for (size_t boundary = 0u;
             boundary < maximumBoundaries;
             ++boundary)
        {
            const PS2Runtime::DebugEeScheduler scheduler =
                env.runtime.debugEeSchedulerSnapshot();
            if (!scheduler.hasNextDeadline)
            {
                return true;
            }

            const uint64_t now =
                env.runtime.currentEeTick().raw();
            const uint64_t elapsed =
                scheduler.nextDeadlineTick > now
                    ? scheduler.nextDeadlineTick - now
                    : 0u;
            if (elapsed >
                std::numeric_limits<uint32_t>::max())
            {
                return false;
            }

            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                env.ctx.cop0_config |= 1u << 18u;
                env.ctx.advanceEeCycleTicks(
                    static_cast<uint32_t>(elapsed));
                env.runtime.serviceEeEventsAtBlockBoundary(
                    env.rdram.data(), &env.ctx);
            }
        }
        return !env.runtime
                    .debugEeSchedulerSnapshot()
                    .hasNextDeadline;
    }

    void writeGuestS16(uint8_t *rdram, uint32_t addr, int16_t value)
    {
        std::memcpy(rdram + addr, &value, sizeof(value));
    }

    int16_t readGuestS16(const uint8_t *rdram, uint32_t addr)
    {
        int16_t value = 0;
        std::memcpy(&value, rdram + addr, sizeof(value));
        return value;
    }

    uint64_t fnv1a64(const uint8_t *data, size_t size)
    {
        constexpr uint64_t kOffsetBasis = 14695981039346656037ull;
        constexpr uint64_t kPrime = 1099511628211ull;
        uint64_t hash = kOffsetBasis;
        for (size_t index = 0; index < size; ++index)
        {
            hash ^= data[index];
            hash *= kPrime;
        }
        return hash;
    }

    uint32_t g_dmacHandlerWriteAddr = 0u;
    uint32_t g_dmacHandlerValue = 0u;
    uint32_t g_dmacHandlerLastCause = 0u;
    uint32_t g_dmacHandlerLastArg = 0u;

    void testDmacHandler(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        g_dmacHandlerLastCause = ::getRegU32(ctx, 4);
        g_dmacHandlerLastArg = ::getRegU32(ctx, 5);
        if (g_dmacHandlerWriteAddr != 0u)
        {
            writeGuestU32(rdram, g_dmacHandlerWriteAddr, g_dmacHandlerValue);
        }
        ctx->pc = 0u;
    }
}

void register_ps2_sif_dma_tests()
{
    MiniTest::Case("PS2SifDma", [](TestCase &tc)
    {
        tc.Run("SIF command heap and DMA state is isolated per runtime", [](TestCase &t)
        {
            TestEnv first;
            TestEnv second;

            constexpr uint32_t kTransientReg = 0x80000002u;
            constexpr uint32_t kFirstRegValue = 0x11112222u;
            constexpr uint32_t kSecondRegValue = 0x33334444u;
            constexpr uint32_t kFirstCmdBuffer = 0x00024000u;
            constexpr uint32_t kSecondCmdBuffer = 0x00028000u;

            const auto setSifReg =
                [](TestEnv &env, uint32_t reg, uint32_t value)
            {
                setRegU32(env.ctx, 4, reg);
                setRegU32(env.ctx, 5, value);
                ps2_stubs::sceSifSetReg(
                    env.rdram.data(),
                    &env.ctx,
                    &env.runtime);
            };
            const auto getSifReg =
                [](TestEnv &env, uint32_t reg)
            {
                setRegU32(env.ctx, 4, reg);
                ps2_stubs::sceSifGetReg(
                    env.rdram.data(),
                    &env.ctx,
                    &env.runtime);
                return ::getRegU32(&env.ctx, 2);
            };
            const auto setCmdBuffer =
                [](TestEnv &env, uint32_t address)
            {
                setRegU32(env.ctx, 4, address);
                ps2_stubs::sceSifSetCmdBuffer(
                    env.rdram.data(),
                    &env.ctx,
                    &env.runtime);
            };
            const auto getCmdBuffer =
                [](TestEnv &env)
            {
                ps2_stubs::sceSifGetDataTable(
                    env.rdram.data(),
                    &env.ctx,
                    &env.runtime);
                return ::getRegU32(&env.ctx, 2);
            };
            const auto allocateHeap =
                [](TestEnv &env, uint32_t size)
            {
                setRegU32(env.ctx, 4, size);
                ps2_stubs::sceSifAllocIopHeap(
                    env.rdram.data(),
                    &env.ctx,
                    &env.runtime);
                return ::getRegU32(&env.ctx, 2);
            };

            setSifReg(
                first, kTransientReg, kFirstRegValue);
            setSifReg(
                second, kTransientReg, kSecondRegValue);
            t.Equals(
                getSifReg(first, kTransientReg),
                kFirstRegValue,
                "the second runtime must not replace the first SIF register");
            t.Equals(
                getSifReg(second, kTransientReg),
                kSecondRegValue,
                "the second runtime should retain its own SIF register");

            setCmdBuffer(first, kFirstCmdBuffer);
            setCmdBuffer(second, kSecondCmdBuffer);
            t.Equals(
                getCmdBuffer(first),
                kFirstCmdBuffer,
                "the first runtime must retain its SIF command buffer");
            t.Equals(
                getCmdBuffer(second),
                kSecondCmdBuffer,
                "the second runtime must retain its SIF command buffer");

            const uint32_t firstHeap =
                allocateHeap(first, 0x40u);
            const uint32_t secondHeap =
                allocateHeap(second, 0x40u);
            t.IsTrue(
                firstHeap != 0u && secondHeap != 0u,
                "both runtimes should allocate SIF heap storage");
            t.Equals(
                secondHeap,
                firstHeap,
                "each runtime should independently allocate the first SIF heap block");

            constexpr uint32_t kDescAddr = 0x00020000u;
            constexpr uint32_t kSrcAddr = 0x00020100u;
            constexpr uint32_t kIopDstAddr = 0x00001000u;
            const Ps2SifDmaTransfer descriptor{
                kSrcAddr,
                kIopDstAddr,
                4,
                0};
            std::memcpy(
                first.rdram.data() + kDescAddr,
                &descriptor,
                sizeof(descriptor));
            std::memcpy(
                second.rdram.data() + kDescAddr,
                &descriptor,
                sizeof(descriptor));
            writeGuestU32(
                first.rdram.data(), kSrcAddr, 0xA1A2A3A4u);
            writeGuestU32(
                second.rdram.data(), kSrcAddr, 0xB1B2B3B4u);

            const int32_t firstDma =
                submitSifDma(first, kDescAddr, 1u);
            const int32_t secondDma =
                submitSifDma(second, kDescAddr, 1u);
            t.IsTrue(
                firstDma > 0 && secondDma > 0,
                "both runtimes should submit SIF DMA work");
            t.Equals(
                secondDma,
                firstDma,
                "each runtime should independently allocate the first SIF DMA ID");
        });

        tc.Run("scheduled sceSifSetDma copies into IOP RAM without aliasing EE memory", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kDescAddr = 0x00020000u;
            constexpr uint32_t kSrcAddr = 0x00020100u;
            constexpr uint32_t kDstAddr = 0x00020200u;

            std::array<uint8_t, 16> payload{};
            for (size_t i = 0; i < payload.size(); ++i)
            {
                payload[i] = static_cast<uint8_t>(0x30u + i);
            }
            std::memcpy(env.rdram.data() + kSrcAddr, payload.data(), payload.size());
            std::memset(env.rdram.data() + kDstAddr, 0xA5, payload.size());
            std::memset(env.runtime.memory().getIOPRAM() + kDstAddr, 0, payload.size());

            const Ps2SifDmaTransfer desc{
                kSrcAddr,
                kDstAddr,
                static_cast<int32_t>(payload.size()),
                0};
            std::memcpy(env.rdram.data() + kDescAddr, &desc, sizeof(desc));

            setRegU32(env.ctx, 4, kDescAddr);
            setRegU32(env.ctx, 5, 1u);
            ps2_stubs::sceSifSetDma(env.rdram.data(), &env.ctx, &env.runtime);
            const int32_t dmaId = getRegS32(env.ctx, 2);
            t.IsTrue(dmaId > 0, "sceSifSetDma should return a positive transfer id on success");
            t.IsTrue(
                serviceAllScheduledBoundaries(env),
                "scheduled SIF DMA should reach completion");

            t.IsTrue(std::memcmp(env.runtime.memory().getIOPRAM() + kDstAddr,
                                 payload.data(),
                                 payload.size()) == 0,
                     "sceSifSetDma should copy transfer payload into IOP RAM");
            const std::array<uint8_t, 16> expectedEeBytes{
                0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5,
                0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5};
            t.IsTrue(std::memcmp(env.rdram.data() + kDstAddr,
                                 expectedEeBytes.data(),
                                 expectedEeBytes.size()) == 0,
                     "sceSifSetDma must not treat the IOP destination as an EE address");

            setRegU32(env.ctx, 4, static_cast<uint32_t>(dmaId));
            ps2_stubs::sceSifDmaStat(env.rdram.data(), &env.ctx, &env.runtime);
            t.IsTrue(getRegS32(env.ctx, 2) < 0, "sceSifDmaStat should be negative when transfer is complete");
        });

        tc.Run("event sceSifSetDma defers HLE IOP visibility to the IOP-cycle deadline", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kDescAddr = 0x00020600u;
            constexpr uint32_t kSrcAddr = 0x00020700u;
            constexpr uint32_t kDstAddr = 0x00020800u;
            constexpr uint32_t kDstat = 0x1000E010u;

            std::array<uint8_t, 16> submitted{};
            std::array<uint8_t, 16> serviced{};
            for (size_t index = 0u;
                 index < submitted.size(); ++index)
            {
                submitted[index] =
                    static_cast<uint8_t>(0x20u + index);
                serviced[index] =
                    static_cast<uint8_t>(0x80u + index);
            }
            std::memcpy(
                env.rdram.data() + kSrcAddr,
                submitted.data(), submitted.size());
            std::memset(
                env.runtime.memory().getIOPRAM() + kDstAddr,
                0, submitted.size());
            const Ps2SifDmaTransfer descriptor{
                kSrcAddr, kDstAddr,
                static_cast<int32_t>(submitted.size()), 0};
            std::memcpy(
                env.rdram.data() + kDescAddr,
                &descriptor, sizeof(descriptor));

            env.runtime.debugStartEeEventTrace(4u);
            const int32_t transferId =
                submitSifDma(env, kDescAddr, 1u);
            t.IsTrue(
                transferId > 0,
                "event submission should return a transfer id");
            t.Equals(
                sifDmaStatus(
                    env,
                    static_cast<uint32_t>(transferId)),
                0,
                "the front event transfer should report in-progress");
            t.IsTrue(
                std::memcmp(
                    env.runtime.memory().getIOPRAM() +
                        kDstAddr,
                    std::array<uint8_t, 16>{}.data(),
                    submitted.size()) == 0,
                "submission must not expose payload before service");
            t.IsTrue(
                (env.runtime.memory().readIORegister(
                     kDstat) &
                 (1u << 5u)) == 0u,
                "submission must not publish the compatibility cause");

            const PS2Runtime::DebugEeScheduler scheduler =
                env.runtime.debugEeSchedulerSnapshot();
            const auto slot =
                scheduler.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            HleSif1)];
            t.IsTrue(
                slot.pending &&
                    slot.device.kind ==
                        PS2Runtime::
                            DebugEeEventDeviceKind::
                                HleSifDma,
                "HLE SIF1 should retain visible event ownership");
            t.Equals(
                slot.deadlineTick, 64ull,
                "one QW should cost one IOP cycle or eight EE cycles");

            std::memcpy(
                env.rdram.data() + kSrcAddr,
                serviced.data(), serviced.size());
            advanceEventTime(env, 56u);
            t.IsTrue(
                std::memcmp(
                    env.runtime.memory().getIOPRAM() +
                        kDstAddr,
                    std::array<uint8_t, 16>{}.data(),
                    serviced.size()) == 0,
                "payload should remain hidden before tick 64");

            advanceEventTime(env, 8u);
            t.IsTrue(
                std::memcmp(
                    env.runtime.memory().getIOPRAM() +
                        kDstAddr,
                    serviced.data(), serviced.size()) == 0,
                "service should read the payload at service time");
            t.IsTrue(
                sifDmaStatus(
                    env,
                    static_cast<uint32_t>(transferId)) < 0,
                "serviced transfer should report complete");
            t.IsTrue(
                (env.runtime.memory().readIORegister(
                     kDstat) &
                 (1u << 5u)) != 0u,
                "typed publication should latch SIF0 cause 5");

            const PS2Runtime::DebugEeEventTrace trace =
                env.runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(2u),
                "service and typed publication should be distinct events");
            if (trace.entries.size() == 2u)
            {
                t.IsTrue(
                    trace.entries[0].source ==
                            ps2x::timing::EeEventSource::
                                HleSif1 &&
                        trace.entries[1].source ==
                            ps2x::timing::EeEventSource::
                                DmacCompletion,
                    "copy and HLE notification must precede publication");
                t.Equals(
                    trace.entries[0].serviceTick,
                    64ull,
                    "HLE SIF service should retain the exact deadline");
                t.Equals(
                    trace.entries[1].serviceTick,
                    64ull,
                    "typed completion should publish in the same boundary");
            }
        });

        tc.Run("event sceSifSetDma queues calls and sums rounded descriptor work", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kDescA = 0x00020900u;
            constexpr uint32_t kDescB = 0x00020940u;
            constexpr uint32_t kSrcA0 = 0x00020A00u;
            constexpr uint32_t kSrcA1 = 0x00020A40u;
            constexpr uint32_t kSrcB = 0x00020A80u;
            constexpr uint32_t kDstA0 = 0x00020B00u;
            constexpr uint32_t kDstA1 = 0x00020B40u;
            constexpr uint32_t kDstB = 0x00020B80u;

            std::array<uint8_t, 1> payloadA0{0x31u};
            std::array<uint8_t, 17> payloadA1{};
            std::array<uint8_t, 16> payloadB{};
            for (size_t index = 0u;
                 index < payloadA1.size(); ++index)
            {
                payloadA1[index] =
                    static_cast<uint8_t>(0x50u + index);
            }
            for (size_t index = 0u;
                 index < payloadB.size(); ++index)
            {
                payloadB[index] =
                    static_cast<uint8_t>(0xA0u + index);
            }
            std::memcpy(
                env.rdram.data() + kSrcA0,
                payloadA0.data(), payloadA0.size());
            std::memcpy(
                env.rdram.data() + kSrcA1,
                payloadA1.data(), payloadA1.size());
            std::memcpy(
                env.rdram.data() + kSrcB,
                payloadB.data(), payloadB.size());
            std::memset(
                env.runtime.memory().getIOPRAM() + kDstA0,
                0, payloadA0.size());
            std::memset(
                env.runtime.memory().getIOPRAM() + kDstA1,
                0, payloadA1.size());
            std::memset(
                env.runtime.memory().getIOPRAM() + kDstB,
                0, payloadB.size());

            const std::array<Ps2SifDmaTransfer, 2>
                descriptorsA{{
                    {kSrcA0, kDstA0, 1, 0},
                    {kSrcA1, kDstA1, 17, 0},
                }};
            const Ps2SifDmaTransfer descriptorB{
                kSrcB, kDstB, 16, 0};
            std::memcpy(
                env.rdram.data() + kDescA,
                descriptorsA.data(), sizeof(descriptorsA));
            std::memcpy(
                env.rdram.data() + kDescB,
                &descriptorB, sizeof(descriptorB));

            env.runtime.debugStartEeEventTrace(8u);
            const int32_t firstId =
                submitSifDma(env, kDescA, 2u);
            const int32_t secondId =
                submitSifDma(env, kDescB, 1u);
            t.IsTrue(
                firstId > 0 && secondId > firstId,
                "queued submissions should receive ordered ids");
            t.Equals(
                sifDmaStatus(
                    env, static_cast<uint32_t>(firstId)),
                0,
                "front operation should report in-progress");
            t.IsTrue(
                sifDmaStatus(
                    env, static_cast<uint32_t>(secondId)) > 0,
                "later operation should report queued");

            auto scheduler =
                env.runtime.debugEeSchedulerSnapshot();
            auto slot =
                scheduler.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            HleSif1)];
            t.Equals(
                slot.deadlineTick, 192ull,
                "1-byte and 17-byte descriptors should cost three rounded QWs");

            advanceEventTime(env, 192u);
            t.IsTrue(
                std::memcmp(
                    env.runtime.memory().getIOPRAM() +
                        kDstA0,
                    payloadA0.data(), payloadA0.size()) == 0 &&
                    std::memcmp(
                        env.runtime.memory().getIOPRAM() +
                            kDstA1,
                        payloadA1.data(),
                        payloadA1.size()) == 0,
                "first service should apply all descriptors in order");
            t.IsTrue(
                std::memcmp(
                    env.runtime.memory().getIOPRAM() +
                        kDstB,
                    std::array<uint8_t, 16>{}.data(),
                    payloadB.size()) == 0,
                "queued operation must remain hidden");
            t.IsTrue(
                sifDmaStatus(
                    env, static_cast<uint32_t>(firstId)) < 0,
                "first operation should retire");
            t.Equals(
                sifDmaStatus(
                    env, static_cast<uint32_t>(secondId)),
                0,
                "second operation should become active");

            scheduler =
                env.runtime.debugEeSchedulerSnapshot();
            slot =
                scheduler.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            HleSif1)];
            t.Equals(
                slot.deadlineTick, 256ull,
                "queued one-QW work should rebase from service tick");

            advanceEventTime(env, 64u);
            t.IsTrue(
                std::memcmp(
                    env.runtime.memory().getIOPRAM() +
                        kDstB,
                    payloadB.data(), payloadB.size()) == 0,
                "second operation should apply at tick 256");
            t.IsTrue(
                sifDmaStatus(
                    env, static_cast<uint32_t>(secondId)) < 0,
                "second operation should retire");

            const PS2Runtime::DebugEeEventTrace trace =
                env.runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(4u),
                "two operations should each own service and publication");
            if (trace.entries.size() == 4u)
            {
                const std::array<
                    ps2x::timing::EeEventSource, 4u>
                    expected{
                        ps2x::timing::EeEventSource::
                            HleSif1,
                        ps2x::timing::EeEventSource::
                            DmacCompletion,
                        ps2x::timing::EeEventSource::
                            HleSif1,
                        ps2x::timing::EeEventSource::
                            DmacCompletion,
                    };
                for (size_t index = 0u;
                     index < expected.size(); ++index)
                {
                    t.IsTrue(
                        trace.entries[index].source ==
                            expected[index],
                        "queued SIF events should retain source order");
                }
            }
        });

        tc.Run("event sceSifSetDma bounds its queue and rejects stale deadlines", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kDescAddr = 0x00020F00u;
            constexpr uint32_t kSrcAddr = 0x00021000u;
            constexpr uint32_t kOldDstAddr = 0x00021100u;
            constexpr uint32_t kNewDstAddr = 0x00021200u;
            std::array<uint8_t, 32> payload{};
            for (size_t index = 0u;
                 index < payload.size(); ++index)
            {
                payload[index] =
                    static_cast<uint8_t>(0xC0u + index);
            }
            std::memcpy(
                env.rdram.data() + kSrcAddr,
                payload.data(), payload.size());
            std::memset(
                env.runtime.memory().getIOPRAM() +
                    kOldDstAddr,
                0, 16u);
            std::memset(
                env.runtime.memory().getIOPRAM() +
                    kNewDstAddr,
                0, payload.size());

            Ps2SifDmaTransfer descriptor{
                kSrcAddr, kOldDstAddr, 16, 0};
            std::memcpy(
                env.rdram.data() + kDescAddr,
                &descriptor, sizeof(descriptor));
            for (size_t index = 0u;
                 index <
                     PS2Runtime::
                         kHleSifDmaQueueCapacity;
                 ++index)
            {
                t.IsTrue(
                    submitSifDma(
                        env, kDescAddr, 1u) > 0,
                    "each fixed queue slot should accept one operation");
            }
            t.Equals(
                submitSifDma(env, kDescAddr, 1u),
                0,
                "queue overflow should fail without side effects");
            t.IsTrue(
                std::memcmp(
                    env.runtime.memory().getIOPRAM() +
                        kOldDstAddr,
                    std::array<uint8_t, 16>{}.data(),
                    16u) == 0,
                "queued and rejected work should remain hidden");

            const auto oldScheduler =
                env.runtime.debugEeSchedulerSnapshot();
            const auto oldSlot =
                oldScheduler.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            HleSif1)];
            t.IsTrue(
                oldSlot.pending,
                "the original generation should own a deadline");

            env.runtime.resetEeTiming(&env.ctx);
            descriptor.dest = kNewDstAddr;
            descriptor.size = 32;
            std::memcpy(
                env.rdram.data() + kDescAddr,
                &descriptor, sizeof(descriptor));
            const int32_t replacementId =
                submitSifDma(env, kDescAddr, 1u);
            t.IsTrue(
                replacementId > 0,
                "replacement generation should submit");
            const auto replacementScheduler =
                env.runtime.debugEeSchedulerSnapshot();
            const auto replacementSlot =
                replacementScheduler.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            HleSif1)];
            t.IsTrue(
                replacementSlot.pending &&
                    replacementSlot.generation >
                        oldSlot.generation,
                "replacement should own a newer scheduler generation");
            t.Equals(
                replacementSlot.deadlineTick, 128ull,
                "two replacement QWs should cost two IOP cycles");

            env.runtime.debugStartEeEventTrace(4u);
            advanceEventTime(env, 64u);
            t.IsTrue(
                std::memcmp(
                    env.runtime.memory().getIOPRAM() +
                        kOldDstAddr,
                    std::array<uint8_t, 16>{}.data(),
                    16u) == 0 &&
                    std::memcmp(
                        env.runtime.memory().getIOPRAM() +
                            kNewDstAddr,
                        std::array<uint8_t, 32>{}.data(),
                        payload.size()) == 0,
                "the cancelled tick-64 generation must not copy");
            t.Equals(
                env.runtime
                    .debugEeEventTraceSnapshot(false)
                    .entries.size(),
                static_cast<size_t>(0u),
                "the stale deadline must not dispatch");

            advanceEventTime(env, 64u);
            t.IsTrue(
                std::memcmp(
                    env.runtime.memory().getIOPRAM() +
                        kNewDstAddr,
                    payload.data(), payload.size()) == 0,
                "replacement should copy at tick 128");
            t.IsTrue(
                sifDmaStatus(
                    env,
                    static_cast<uint32_t>(
                        replacementId)) < 0,
                "replacement should retire");
            t.IsTrue(
                std::memcmp(
                    env.runtime.memory().getIOPRAM() +
                        kOldDstAddr,
                    std::array<uint8_t, 16>{}.data(),
                    16u) == 0,
                "cancelled queued work must never become visible");

            descriptor.dest = kOldDstAddr;
            descriptor.size = 16;
            std::memcpy(
                env.rdram.data() + kDescAddr,
                &descriptor, sizeof(descriptor));
            const int32_t resetCancelledId =
                submitSifDma(env, kDescAddr, 1u);
            t.IsTrue(
                resetCancelledId > 0,
                "pre-reset work should submit");
            env.runtime.resetEeTiming(&env.ctx);
            t.IsFalse(
                env.runtime.debugEeSchedulerSnapshot()
                    .slots[
                        ps2x::timing::eeEventSourceIndex(
                            ps2x::timing::EeEventSource::
                                HleSif1)]
                    .pending,
                "timing reset should cancel the retained source");
            advanceEventTime(env, 192u);
            t.IsTrue(
                std::memcmp(
                    env.runtime.memory().getIOPRAM() +
                        kOldDstAddr,
                    std::array<uint8_t, 16>{}.data(),
                    16u) == 0,
                "timing-reset work must never become visible");
            t.IsTrue(
                sifDmaStatus(
                    env,
                    static_cast<uint32_t>(
                        resetCancelledId)) < 0,
                "timing-reset work should report retired");
        });

        tc.Run("Sony 989snd decodes streaming DMA from IOP RAM rather than the aliased EE address", [](TestCase &t)
        {
            TestEnv env;
            configureProfile(env, "slus_201.84");

            constexpr uint32_t kSid = 0x00123456u;
            constexpr uint32_t kClientAddr = 0x00024000u;
            constexpr uint32_t kSendAddr = 0x00024100u;
            constexpr uint32_t kRecvAddr = 0x00024200u;
            constexpr uint32_t kDescAddr = 0x00024300u;
            constexpr uint32_t kDmaSourceAddr = 0x00024400u;
            constexpr uint32_t kTransferBytes = 0x400u;

            ps2_syscalls::SifInitRpc(env.rdram.data(), &env.ctx, &env.runtime);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, kSid);
            setRegU32(env.ctx, 6, 0u);
            ps2_syscalls::SifBindRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK,
                     "SifBindRpc should bind the Sony 989snd server");

            const auto callRpc = [&](uint32_t function, uint32_t sendSize)
            {
                setRegU32(env.ctx, 4, kClientAddr);
                setRegU32(env.ctx, 5, function);
                setRegU32(env.ctx, 6, 0u);
                setRegU32(env.ctx, 7, sendSize == 0u ? 0u : kSendAddr);
                setRegU32(env.ctx, 8, sendSize);
                setRegU32(env.ctx, 9, kRecvAddr);
                setRegU32(env.ctx, 10, 12u);
                setRegU32(env.ctx, 11, 0u);
                ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            };

            constexpr std::array<uint32_t, 6> kOpen{
                kTransferBytes,
                0x1000u,
                kTransferBytes,
                0u,
                5u,
                3u,
            };
            std::memcpy(env.rdram.data() + kSendAddr, kOpen.data(), sizeof(kOpen));
            callRpc(0x3Bu, sizeof(kOpen));
            t.Equals(getRegS32(env.ctx, 2), KE_OK,
                     "streaming-open RPC should succeed");
            const uint32_t workArea = readGuestU32(env.rdram.data(), kRecvAddr + 4u);
            t.IsTrue(workArea != 0u && workArea + kTransferBytes <= PS2_IOP_RAM_SIZE,
                     "streaming-open RPC should allocate an address inside IOP RAM");
            t.IsTrue(env.runtime.audioBackend().streamDebugSnapshot().opened,
                     "the host stream should receive a valid IOP work-area pointer");

            std::array<uint8_t, kTransferBytes> encoded{};
            for (size_t block = 0u; block < encoded.size(); block += 16u)
            {
                encoded[block] = 0x0Cu;
                encoded[block + 1u] = 0u;
                for (size_t byte = 2u; byte < 16u; ++byte)
                    encoded[block + byte] = static_cast<uint8_t>(0x10u + (block / 16u + byte) % 0x70u);
            }

            std::memset(env.rdram.data() + workArea, 0xE7, encoded.size());
            std::memcpy(env.rdram.data() + kDmaSourceAddr,
                        encoded.data(),
                        encoded.size());
            std::memset(env.runtime.memory().getIOPRAM() + workArea,
                        0,
                        encoded.size());

            const Ps2SifDmaTransfer descriptor{
                kDmaSourceAddr,
                workArea,
                static_cast<int32_t>(encoded.size()),
                0,
            };
            std::memcpy(env.rdram.data() + kDescAddr,
                        &descriptor,
                        sizeof(descriptor));
            setRegU32(env.ctx, 4, kDescAddr);
            setRegU32(env.ctx, 5, 1u);
            ps2_stubs::sceSifSetDma(env.rdram.data(), &env.ctx, &env.runtime);
            t.IsTrue(getRegS32(env.ctx, 2) > 0,
                     "streaming payload DMA should succeed");
            t.IsTrue(
                serviceAllScheduledBoundaries(env),
                "scheduled streaming payload DMA should complete");
            t.IsTrue(std::memcmp(env.runtime.memory().getIOPRAM() + workArea,
                                 encoded.data(),
                                 encoded.size()) == 0,
                     "streaming payload should reside in IOP RAM");
            t.Equals(env.rdram[workArea], static_cast<uint8_t>(0xE7),
                     "the same EE virtual address should remain unrelated");

            constexpr std::array<uint32_t, 2> kSubmit{
                kTransferBytes,
                0u,
            };
            std::memcpy(env.rdram.data() + kSendAddr,
                        kSubmit.data(),
                        sizeof(kSubmit));
            callRpc(0x5Au, sizeof(kSubmit));
            t.Equals(getRegS32(env.ctx, 2), KE_OK,
                     "streaming-submit RPC should succeed");

            const PS2AudioStreamDebugSnapshot snapshot =
                env.runtime.audioBackend().streamDebugSnapshot();
            t.Equals(snapshot.submissionCount, uint64_t{1u},
                     "the audio backend should decode one submitted chunk");
            t.Equals(snapshot.submittedBytes, uint64_t{kTransferBytes},
                     "the audio backend should decode the complete DMA chunk");
            t.Equals(snapshot.lastSubmissionHash,
                     fnv1a64(encoded.data(), encoded.size()),
                     "the decoded source must be the IOP DMA payload, not aliased EE bytes");
        });

        tc.Run("isceSifSetDma and isceSifSetDChain alias the SIF DMA helpers", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kDescAddr = 0x00020240u;
            constexpr uint32_t kSrcAddr = 0x00020340u;
            constexpr uint32_t kIopDstAddr = 0x00020440u;
            constexpr uint32_t kTaggedDstAddr = 0xA0020440u;

            std::array<uint8_t, 12> payload{};
            for (size_t i = 0; i < payload.size(); ++i)
            {
                payload[i] = static_cast<uint8_t>(0x50u + i);
            }
            std::memcpy(env.rdram.data() + kSrcAddr, payload.data(), payload.size());
            std::memset(env.rdram.data() + kIopDstAddr, 0xCC, payload.size());
            std::memset(env.runtime.memory().getIOPRAM() + kIopDstAddr, 0, payload.size());

            const Ps2SifDmaTransfer desc{
                kSrcAddr,
                kTaggedDstAddr,
                static_cast<int32_t>(payload.size()),
                0};
            std::memcpy(env.rdram.data() + kDescAddr, &desc, sizeof(desc));

            setRegU32(env.ctx, 4, kDescAddr);
            setRegU32(env.ctx, 5, 1u);
            ps2_stubs::isceSifSetDma(env.rdram.data(), &env.ctx, &env.runtime);
            t.IsTrue(getRegS32(env.ctx, 2) > 0, "isceSifSetDma should report a successful transfer id");
            t.IsTrue(
                serviceAllScheduledBoundaries(env),
                "scheduled isceSifSetDma transfer should complete");
            t.IsTrue(std::memcmp(env.runtime.memory().getIOPRAM() + kIopDstAddr,
                                 payload.data(),
                                 payload.size()) == 0,
                     "isceSifSetDma should use the tag's 24-bit IOP destination");

            ps2_stubs::isceSifSetDChain(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), 0, "isceSifSetDChain should mirror sceSifSetDChain");
        });

        tc.Run("sceSifSetDma preserves explicitly allocated synthetic IOP heap ranges", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kDescAddr = 0x00020500u;
            constexpr uint32_t kSrcAddr = 0x00020600u;
            constexpr uint32_t kAllocationSize = 64u;
            std::array<uint8_t, 16> payload{};
            for (size_t i = 0; i < payload.size(); ++i)
            {
                payload[i] = static_cast<uint8_t>(0xB0u + i);
            }
            std::memcpy(env.rdram.data() + kSrcAddr, payload.data(), payload.size());

            setRegU32(env.ctx, 4, kAllocationSize);
            ps2_stubs::sceSifAllocIopHeap(env.rdram.data(), &env.ctx, &env.runtime);
            const uint32_t allocation = ::getRegU32(&env.ctx, 2);
            t.IsTrue(allocation >= 0x00200000u,
                     "test requires the compatibility heap to use a synthetic address");

            const uint32_t destination = allocation + 8u;
            std::memset(env.rdram.data() + destination, 0, payload.size());
            const Ps2SifDmaTransfer desc{
                kSrcAddr,
                destination,
                static_cast<int32_t>(payload.size()),
                0};
            std::memcpy(env.rdram.data() + kDescAddr, &desc, sizeof(desc));

            setRegU32(env.ctx, 4, kDescAddr);
            setRegU32(env.ctx, 5, 1u);
            ps2_stubs::sceSifSetDma(env.rdram.data(), &env.ctx, &env.runtime);
            t.IsTrue(getRegS32(env.ctx, 2) > 0,
                     "sceSifSetDma should accept an allocated synthetic IOP range");
            t.IsTrue(
                serviceAllScheduledBoundaries(env),
                "scheduled synthetic IOP transfer should complete");
            t.IsTrue(std::memcmp(env.rdram.data() + destination,
                                 payload.data(),
                                 payload.size()) == 0,
                     "synthetic IOP allocations should retain their EE backing");
        });

        tc.Run("sceSifSetDma dispatches enabled DMAC handlers for cause 5", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kDescAddr = 0x00020300u;
            constexpr uint32_t kSrcAddr = 0x00020400u;
            constexpr uint32_t kDstAddr = 0x00020500u;
            constexpr uint32_t kHandlerAddr = 0x00100000u;
            constexpr uint32_t kHandlerWriteAddr = 0x00020600u;
            constexpr uint32_t kHandlerArg = 0x12345678u;

            g_dmacHandlerWriteAddr = kHandlerWriteAddr;
            g_dmacHandlerValue = 0xCAFEBABEu;
            g_dmacHandlerLastCause = 0u;
            g_dmacHandlerLastArg = 0u;
            env.runtime.registerFunction(kHandlerAddr, &testDmacHandler);

            setRegU32(env.ctx, 4, 5u);
            setRegU32(env.ctx, 5, kHandlerAddr);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kHandlerArg);
            ps2_syscalls::AddDmacHandler(env.rdram.data(), &env.ctx, &env.runtime);
            const int32_t handlerId = getRegS32(env.ctx, 2);
            t.IsTrue(handlerId > 0, "AddDmacHandler should register a handler");

            setRegU32(env.ctx, 4, 5u);
            ps2_syscalls::EnableDmac(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), 0, "EnableDmac should succeed");

            std::array<uint8_t, 16> payload{};
            for (size_t i = 0; i < payload.size(); ++i)
            {
                payload[i] = static_cast<uint8_t>(0x40u + i);
            }
            std::memcpy(env.rdram.data() + kSrcAddr, payload.data(), payload.size());

            const Ps2SifDmaTransfer desc{
                kSrcAddr,
                kDstAddr,
                static_cast<int32_t>(payload.size()),
                0};
            std::memcpy(env.rdram.data() + kDescAddr, &desc, sizeof(desc));

            setRegU32(env.ctx, 4, kDescAddr);
            setRegU32(env.ctx, 5, 1u);
            {
                PS2Runtime::GuestExecutionScope guestExecution(
                    &env.runtime, &env.ctx);
                ps2_stubs::sceSifSetDma(
                    env.rdram.data(), &env.ctx, &env.runtime);
                t.IsTrue(
                    serviceAllScheduledBoundaries(env),
                    "scheduled SIF DMA should publish its completion");
                t.Equals(
                    readGuestU32(
                        env.rdram.data(),
                        kHandlerWriteAddr),
                    0u,
                    "sceSifSetDma must not recursively enter its DMAC handler");
            }

            t.IsTrue(getRegS32(env.ctx, 2) > 0, "sceSifSetDma should still report success");
            t.Equals(readGuestU32(
                         env.runtime.memory().getRDRAM(),
                         kHandlerWriteAddr),
                     g_dmacHandlerValue,
                     "sceSifSetDma should invoke registered DMAC handlers");
            t.Equals(g_dmacHandlerLastCause, 5u, "DMAC handler should observe cause 5");
            t.Equals(g_dmacHandlerLastArg, kHandlerArg, "DMAC handler should receive registered argument");
        });

        tc.Run("sceSifSetDma acknowledges DTX work-buffer transfers by advancing the EE footer ticket", [](TestCase &t)
        {
            TestEnv env;
            configureProfile(env, "slus_201.84");

            constexpr uint32_t kClientAddr = 0x0002D000u;
            constexpr uint32_t kDtxSid = 0x7D000000u;
            constexpr uint32_t kSendAddr = 0x0002D100u;
            constexpr uint32_t kRecvAddr = 0x0002D200u;
            constexpr uint32_t kDescAddr = 0x0002D300u;
            constexpr uint32_t kEeWorkAddr = 0x0002D400u;
            constexpr uint32_t kIopWorkAddr = 0x0002D800u;
            constexpr uint32_t kDtxId = 3u;
            constexpr uint32_t kWorkLen = 0x100u;
            constexpr uint32_t kFooterTicketAddr = kEeWorkAddr + kWorkLen - sizeof(uint32_t);

            ps2_syscalls::SifInitRpc(env.rdram.data(), &env.ctx, &env.runtime);

            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, kDtxSid);
            setRegU32(env.ctx, 6, 0u);
            ps2_syscalls::SifBindRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "SifBindRpc should succeed for the DTX sid");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, kDtxId);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, kEeWorkAddr);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x08u, kIopWorkAddr);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x0Cu, kWorkLen);
            writeGuestU32(env.rdram.data(), kRecvAddr + 0x00u, 0u);

            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 2u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 16u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "SifCallRpc should create the DTX transport");
            t.IsTrue(readGuestU32(env.rdram.data(), kRecvAddr) != 0u, "DTX create should return a remote handle");

            std::memset(env.rdram.data() + kEeWorkAddr, 0x44, kWorkLen);
            std::memset(env.rdram.data() + kIopWorkAddr, 0x00, kWorkLen);
            writeGuestU32(env.rdram.data(), kFooterTicketAddr, 1u);

            const Ps2SifDmaTransfer desc{
                kEeWorkAddr,
                kIopWorkAddr,
                static_cast<int32_t>(kWorkLen),
                0};
            std::memcpy(env.rdram.data() + kDescAddr, &desc, sizeof(desc));

            setRegU32(env.ctx, 4, kDescAddr);
            setRegU32(env.ctx, 5, 1u);
            ps2_stubs::sceSifSetDma(env.rdram.data(), &env.ctx, &env.runtime);
            t.IsTrue(getRegS32(env.ctx, 2) > 0, "sceSifSetDma should succeed for the DTX transfer");
            t.IsTrue(
                serviceAllScheduledBoundaries(env),
                "scheduled DTX transfer should complete");

            t.Equals(readGuestU32(env.rdram.data(), kFooterTicketAddr), 2u,
                     "sceSifSetDma should advance the EE footer ticket so DTX clears wait_flag");
        });

        tc.Run("event sceSifSetDma orders DTX copy notification and completion", [](TestCase &t)
        {
            TestEnv env;
            configureProfile(env, "slus_201.84");

            constexpr uint32_t kClientAddr = 0x0002D000u;
            constexpr uint32_t kDtxSid = 0x7D000000u;
            constexpr uint32_t kSendAddr = 0x0002D100u;
            constexpr uint32_t kRecvAddr = 0x0002D200u;
            constexpr uint32_t kDescAddr = 0x0002D300u;
            constexpr uint32_t kEeWorkAddr = 0x0002D400u;
            constexpr uint32_t kIopWorkAddr = 0x0002D800u;
            constexpr uint32_t kDtxId = 3u;
            constexpr uint32_t kWorkLen = 0x100u;
            constexpr uint32_t kFooterTicketAddr =
                kEeWorkAddr + kWorkLen - sizeof(uint32_t);
            constexpr uint32_t kIopFooterTicketOffset =
                kIopWorkAddr + kWorkLen - sizeof(uint32_t);
            constexpr uint32_t kDstat = 0x1000E010u;

            ps2_syscalls::SifInitRpc(
                env.rdram.data(), &env.ctx, &env.runtime);

            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, kDtxSid);
            setRegU32(env.ctx, 6, 0u);
            ps2_syscalls::SifBindRpc(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2), KE_OK,
                "SifBindRpc should succeed for the DTX sid");

            writeGuestU32(
                env.rdram.data(), kSendAddr + 0x00u, kDtxId);
            writeGuestU32(
                env.rdram.data(), kSendAddr + 0x04u,
                kEeWorkAddr);
            writeGuestU32(
                env.rdram.data(), kSendAddr + 0x08u,
                kIopWorkAddr);
            writeGuestU32(
                env.rdram.data(), kSendAddr + 0x0Cu, kWorkLen);
            writeGuestU32(
                env.rdram.data(), kRecvAddr + 0x00u, 0u);

            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 2u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 16u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(
                env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(
                getRegS32(env.ctx, 2), KE_OK,
                "SifCallRpc should create the DTX transport");

            std::memset(
                env.rdram.data() + kEeWorkAddr,
                0x44, kWorkLen);
            std::memset(
                env.runtime.memory().getIOPRAM() +
                    kIopWorkAddr,
                0, kWorkLen);
            writeGuestU32(
                env.rdram.data(), kFooterTicketAddr, 1u);

            const Ps2SifDmaTransfer descriptor{
                kEeWorkAddr,
                kIopWorkAddr,
                static_cast<int32_t>(kWorkLen),
                0};
            std::memcpy(
                env.rdram.data() + kDescAddr,
                &descriptor, sizeof(descriptor));

            env.runtime.debugStartEeEventTrace(4u);
            const int32_t transferId =
                submitSifDma(env, kDescAddr, 1u);
            t.IsTrue(
                transferId > 0,
                "event DTX transfer should submit");
            t.Equals(
                readGuestU32(
                    env.rdram.data(), kFooterTicketAddr),
                1u,
                "DTX notification must remain deferred");
            t.Equals(
                readGuestU32(
                    env.runtime.memory().getIOPRAM(),
                    kIopFooterTicketOffset),
                0u,
                "DTX payload must remain hidden before service");

            advanceEventTime(env, 1016u);
            t.Equals(
                readGuestU32(
                    env.rdram.data(), kFooterTicketAddr),
                1u,
                "DTX notification must remain hidden before tick 1024");
            t.Equals(
                readGuestU32(
                    env.runtime.memory().getIOPRAM(),
                    kIopFooterTicketOffset),
                0u,
                "DTX copy must remain hidden before tick 1024");

            advanceEventTime(env, 8u);
            t.Equals(
                readGuestU32(
                    env.runtime.memory().getIOPRAM(),
                    kIopFooterTicketOffset),
                1u,
                "copy should capture the pre-notification footer");
            t.Equals(
                readGuestU32(
                    env.rdram.data(), kFooterTicketAddr),
                2u,
                "AfterCopy notification should advance the EE footer");
            t.IsTrue(
                (env.runtime.memory().readIORegister(
                     kDstat) &
                 (1u << 5u)) != 0u,
                "typed completion should follow DTX notification");

            const PS2Runtime::DebugEeEventTrace trace =
                env.runtime.debugEeEventTraceSnapshot(true);
            t.IsTrue(
                trace.entries.size() == 2u &&
                    trace.entries[0].source ==
                        ps2x::timing::EeEventSource::
                            HleSif1 &&
                    trace.entries[1].source ==
                        ps2x::timing::EeEventSource::
                            DmacCompletion,
                "DTX service should precede typed publication");
        });

        tc.Run("sceSifSetDma applies SJX DTX payloads into the emulated SJRMT data ring", [](TestCase &t)
        {
            TestEnv env;
            configureProfile(env, "slus_201.84");

            constexpr uint32_t kClientAddr = 0x0002E000u;
            constexpr uint32_t kDtxSid = 0x7D000000u;
            constexpr uint32_t kRecvAddr = 0x0002E100u;
            constexpr uint32_t kSendAddr = 0x0002E200u;
            constexpr uint32_t kDescAddr = 0x0002E300u;
            constexpr uint32_t kEeWorkAddr = 0x0002E400u;
            constexpr uint32_t kIopWorkAddr = 0x0002E800u;
            constexpr uint32_t kRingAddr = 0x0002EC00u;
            constexpr uint32_t kChunkDataAddr = 0x0002ED00u;
            constexpr uint32_t kWorkLen = 0x100u;
            constexpr uint32_t kChunkLen = 8u;

            ps2_syscalls::SifInitRpc(env.rdram.data(), &env.ctx, &env.runtime);

            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, kDtxSid);
            setRegU32(env.ctx, 6, 0u);
            ps2_syscalls::SifBindRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "SifBindRpc should bind the DTX sid");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, 1u);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, kRingAddr);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x08u, kWorkLen);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 0x422u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 12u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            const uint32_t sjrmtHandle = readGuestU32(env.rdram.data(), kRecvAddr);
            t.IsTrue(sjrmtHandle != 0u, "SJRMT_UNI_CREATE should return a handle");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, 0u);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, sjrmtHandle);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x08u, 1u);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x0Cu, 0x12345678u);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 0x400u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 16u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            const uint32_t sjxHandle = readGuestU32(env.rdram.data(), kRecvAddr);
            t.IsTrue(sjxHandle != 0u, "SJX_CREATE should return a handle");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, 0u);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, kEeWorkAddr);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x08u, kIopWorkAddr);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x0Cu, kWorkLen);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 2u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 16u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "DTX create should succeed");

            std::memset(env.rdram.data() + kEeWorkAddr, 0, kWorkLen);
            std::memset(env.rdram.data() + kIopWorkAddr, 0, kWorkLen);
            std::memset(env.rdram.data() + kRingAddr, 0, kWorkLen);
            for (uint32_t i = 0; i < kChunkLen; ++i)
            {
                env.rdram[kChunkDataAddr + i] = static_cast<uint8_t>(0xA0u + i);
            }

            writeGuestU32(env.rdram.data(), kEeWorkAddr + 0x00u, 1u);
            env.rdram[kEeWorkAddr + 0x10u] = 0u;
            env.rdram[kEeWorkAddr + 0x11u] = 1u;
            std::memcpy(env.rdram.data() + kEeWorkAddr + 0x12u, "\0\0", 2u);
            writeGuestU32(env.rdram.data(), kEeWorkAddr + 0x14u, sjxHandle);
            writeGuestU32(env.rdram.data(), kEeWorkAddr + 0x18u, kChunkDataAddr);
            writeGuestU32(env.rdram.data(), kEeWorkAddr + 0x1Cu, kChunkLen);
            writeGuestU32(env.rdram.data(), kEeWorkAddr + kWorkLen - sizeof(uint32_t), 1u);

            const Ps2SifDmaTransfer desc{
                kEeWorkAddr,
                kIopWorkAddr,
                static_cast<int32_t>(kWorkLen),
                0};
            std::memcpy(env.rdram.data() + kDescAddr, &desc, sizeof(desc));

            setRegU32(env.ctx, 4, kDescAddr);
            setRegU32(env.ctx, 5, 1u);
            ps2_stubs::sceSifSetDma(env.rdram.data(), &env.ctx, &env.runtime);
            t.IsTrue(getRegS32(env.ctx, 2) > 0, "sceSifSetDma should succeed for the SJX transport");
            t.IsTrue(
                serviceAllScheduledBoundaries(env),
                "scheduled SJX transfer should complete");
            t.Equals(env.rdram[kEeWorkAddr + 0x11u], static_cast<uint8_t>(0u),
                     "SJX DMA ack should rewrite the response line to room so EE recycles the chunk");
            t.Equals(readGuestU32(env.rdram.data(), kEeWorkAddr + kWorkLen - sizeof(uint32_t)), 2u,
                     "SJX DMA ack should still advance the EE footer ticket");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, sjrmtHandle);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, 1u);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 0x429u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 8u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(readGuestU32(env.rdram.data(), kRecvAddr), kChunkLen,
                     "SJX DMA should make SJRMT report available data");
            t.IsTrue(std::memcmp(env.rdram.data() + kRingAddr, env.rdram.data() + kChunkDataAddr, kChunkLen) == 0,
                     "SJX DMA should copy the chunk payload into the emulated SJRMT ring");
        });

        tc.Run("sceSifSetDma recognizes SJX DTX payloads from rotated EE work buffers", [](TestCase &t)
        {
            TestEnv env;
            configureProfile(env, "slus_201.84");

            constexpr uint32_t kClientAddr = 0x00031000u;
            constexpr uint32_t kDtxSid = 0x7D000000u;
            constexpr uint32_t kRecvAddr = 0x00031100u;
            constexpr uint32_t kSendAddr = 0x00031200u;
            constexpr uint32_t kDescAddr = 0x00031300u;
            constexpr uint32_t kRegisteredEeWorkAddr = 0x00031400u;
            constexpr uint32_t kRegisteredIopWorkAddr = 0x00031800u;
            constexpr uint32_t kAltEeWorkAddr = 0x00031C00u;
            constexpr uint32_t kAltIopWorkAddr = 0x00032000u;
            constexpr uint32_t kRingAddr = 0x00032400u;
            constexpr uint32_t kChunkDataAddr = 0x00032500u;
            constexpr uint32_t kRegisteredWorkLen = 0x100u;
            constexpr uint32_t kAltWorkLen = 0x180u;
            constexpr uint32_t kChunkLen = 12u;

            ps2_syscalls::SifInitRpc(env.rdram.data(), &env.ctx, &env.runtime);

            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, kDtxSid);
            setRegU32(env.ctx, 6, 0u);
            ps2_syscalls::SifBindRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "SifBindRpc should bind the DTX sid");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, 1u);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, kRingAddr);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x08u, kRegisteredWorkLen);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 0x422u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 12u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            const uint32_t sjrmtHandle = readGuestU32(env.rdram.data(), kRecvAddr);
            t.IsTrue(sjrmtHandle != 0u, "SJRMT_UNI_CREATE should return a handle");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, 0u);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, sjrmtHandle);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x08u, 1u);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x0Cu, 0x87654321u);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 0x400u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 16u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            const uint32_t sjxHandle = readGuestU32(env.rdram.data(), kRecvAddr);
            t.IsTrue(sjxHandle != 0u, "SJX_CREATE should return a handle");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, 0u);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, kRegisteredEeWorkAddr);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x08u, kRegisteredIopWorkAddr);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x0Cu, kRegisteredWorkLen);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 2u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 16u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "DTX create should succeed");

            std::memset(env.rdram.data() + kRegisteredEeWorkAddr, 0, kRegisteredWorkLen);
            std::memset(env.rdram.data() + kRegisteredIopWorkAddr, 0, kRegisteredWorkLen);
            std::memset(env.rdram.data() + kAltEeWorkAddr, 0, kAltWorkLen);
            std::memset(env.rdram.data() + kAltIopWorkAddr, 0, kAltWorkLen);
            std::memset(env.rdram.data() + kRingAddr, 0, kRegisteredWorkLen);
            for (uint32_t i = 0; i < kChunkLen; ++i)
            {
                env.rdram[kChunkDataAddr + i] = static_cast<uint8_t>(0xC0u + i);
            }

            writeGuestU32(env.rdram.data(), kAltEeWorkAddr + 0x00u, 1u);
            env.rdram[kAltEeWorkAddr + 0x10u] = 0u;
            env.rdram[kAltEeWorkAddr + 0x11u] = 1u;
            std::memcpy(env.rdram.data() + kAltEeWorkAddr + 0x12u, "\0\0", 2u);
            writeGuestU32(env.rdram.data(), kAltEeWorkAddr + 0x14u, sjxHandle);
            writeGuestU32(env.rdram.data(), kAltEeWorkAddr + 0x18u, kChunkDataAddr);
            writeGuestU32(env.rdram.data(), kAltEeWorkAddr + 0x1Cu, kChunkLen);
            writeGuestU32(env.rdram.data(), kAltEeWorkAddr + kAltWorkLen - sizeof(uint32_t), 9u);

            const Ps2SifDmaTransfer desc{
                kAltEeWorkAddr,
                kAltIopWorkAddr,
                static_cast<int32_t>(kAltWorkLen),
                0};
            std::memcpy(env.rdram.data() + kDescAddr, &desc, sizeof(desc));

            setRegU32(env.ctx, 4, kDescAddr);
            setRegU32(env.ctx, 5, 1u);
            ps2_stubs::sceSifSetDma(env.rdram.data(), &env.ctx, &env.runtime);
            t.IsTrue(getRegS32(env.ctx, 2) > 0, "sceSifSetDma should succeed for the rotated SJX transport");
            t.IsTrue(
                serviceAllScheduledBoundaries(env),
                "scheduled rotated SJX transfer should complete");
            t.Equals(env.rdram[kAltEeWorkAddr + 0x11u], static_cast<uint8_t>(0u),
                     "rotated SJX DMA ack should rewrite the response line to room");
            t.Equals(readGuestU32(env.rdram.data(), kAltEeWorkAddr + kAltWorkLen - sizeof(uint32_t)), 10u,
                     "rotated SJX DMA ack should advance the alternate EE footer ticket");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, sjrmtHandle);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, 1u);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 0x429u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 8u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(readGuestU32(env.rdram.data(), kRecvAddr), kChunkLen,
                     "rotated SJX DMA should make SJRMT report available data");
            t.IsTrue(std::memcmp(env.rdram.data() + kRingAddr, env.rdram.data() + kChunkDataAddr, kChunkLen) == 0,
                     "rotated SJX DMA should copy the chunk payload into the emulated SJRMT ring");
        });

        tc.Run("sceSifSetDma lets active PS2RNA playback drain emulated SJRMT data", [](TestCase &t)
        {
            TestEnv env;
            configureProfile(env, "slus_201.84");

            constexpr uint32_t kClientAddr = 0x0002F000u;
            constexpr uint32_t kDtxSid = 0x7D000000u;
            constexpr uint32_t kRecvAddr = 0x0002F100u;
            constexpr uint32_t kSendAddr = 0x0002F200u;
            constexpr uint32_t kDesc0Addr = 0x0002F300u;
            constexpr uint32_t kDesc1Addr = 0x0002F320u;
            constexpr uint32_t kEeWork0Addr = 0x0002F400u;
            constexpr uint32_t kIopWork0Addr = 0x0002F800u;
            constexpr uint32_t kEeWork1Addr = 0x0002FC00u;
            constexpr uint32_t kIopWork1Addr = 0x00030000u;
            constexpr uint32_t kRingAddr = 0x00030400u;
            constexpr uint32_t kChunkDataAddr = 0x00030500u;
            constexpr uint32_t kWorkLen = 0x100u;
            constexpr uint32_t kChunkLen = 8u;

            ps2_syscalls::SifInitRpc(env.rdram.data(), &env.ctx, &env.runtime);

            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, kDtxSid);
            setRegU32(env.ctx, 6, 0u);
            ps2_syscalls::SifBindRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "SifBindRpc should bind the DTX sid");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, 1u);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, kRingAddr);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x08u, kWorkLen);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 0x422u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 12u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            const uint32_t sjrmtHandle = readGuestU32(env.rdram.data(), kRecvAddr);
            t.IsTrue(sjrmtHandle != 0u, "SJRMT_UNI_CREATE should return a handle");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, 0u);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, sjrmtHandle);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x08u, 1u);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x0Cu, 0xCAFEBABEu);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 0x400u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 16u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            const uint32_t sjxHandle = readGuestU32(env.rdram.data(), kRecvAddr);
            t.IsTrue(sjxHandle != 0u, "SJX_CREATE should return a handle");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, 1u);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, 0u);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x08u, sjrmtHandle);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x0Cu, 0u);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 0x408u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 16u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            const uint32_t ps2RnaHandle = readGuestU32(env.rdram.data(), kRecvAddr);
            t.IsTrue(ps2RnaHandle != 0u, "PS2RNA_CREATE should return a handle");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, 0u);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, kEeWork0Addr);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x08u, kIopWork0Addr);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x0Cu, kWorkLen);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 2u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 16u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "DTX create should succeed for SJX transport");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, 1u);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, kEeWork1Addr);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x08u, kIopWork1Addr);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x0Cu, kWorkLen);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 2u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 16u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "DTX create should succeed for PS2RNA transport");

            std::memset(env.rdram.data() + kEeWork0Addr, 0, kWorkLen);
            std::memset(env.rdram.data() + kIopWork0Addr, 0, kWorkLen);
            std::memset(env.rdram.data() + kEeWork1Addr, 0, kWorkLen);
            std::memset(env.rdram.data() + kIopWork1Addr, 0, kWorkLen);
            std::memset(env.rdram.data() + kRingAddr, 0, kWorkLen);
            for (uint32_t i = 0; i < kChunkLen; ++i)
            {
                env.rdram[kChunkDataAddr + i] = static_cast<uint8_t>(0xB0u + i);
            }

            writeGuestU32(env.rdram.data(), kEeWork1Addr + 0x00u, 1u);
            writeGuestU32(env.rdram.data(), kEeWork1Addr + 0x10u, 2u);
            writeGuestU32(env.rdram.data(), kEeWork1Addr + 0x14u, ps2RnaHandle);
            writeGuestU32(env.rdram.data(), kEeWork1Addr + 0x18u, 1u);
            writeGuestU32(env.rdram.data(), kEeWork1Addr + 0x1Cu, 0u);
            writeGuestU32(env.rdram.data(), kEeWork1Addr + kWorkLen - sizeof(uint32_t), 1u);

            const Ps2SifDmaTransfer desc1{
                kEeWork1Addr,
                kIopWork1Addr,
                static_cast<int32_t>(kWorkLen),
                0};
            std::memcpy(env.rdram.data() + kDesc1Addr, &desc1, sizeof(desc1));

            setRegU32(env.ctx, 4, kDesc1Addr);
            setRegU32(env.ctx, 5, 1u);
            ps2_stubs::sceSifSetDma(env.rdram.data(), &env.ctx, &env.runtime);
            t.IsTrue(getRegS32(env.ctx, 2) > 0, "sceSifSetDma should succeed for the PS2RNA control transport");
            t.IsTrue(
                serviceAllScheduledBoundaries(env),
                "scheduled PS2RNA control transfer should complete");
            t.Equals(readGuestU32(env.rdram.data(), kEeWork1Addr + kWorkLen - sizeof(uint32_t)), 2u,
                     "PS2RNA control DMA should advance the EE footer ticket");

            writeGuestU32(env.rdram.data(), kEeWork0Addr + 0x00u, 1u);
            env.rdram[kEeWork0Addr + 0x10u] = 0u;
            env.rdram[kEeWork0Addr + 0x11u] = 1u;
            std::memcpy(env.rdram.data() + kEeWork0Addr + 0x12u, "\0\0", 2u);
            writeGuestU32(env.rdram.data(), kEeWork0Addr + 0x14u, sjxHandle);
            writeGuestU32(env.rdram.data(), kEeWork0Addr + 0x18u, kChunkDataAddr);
            writeGuestU32(env.rdram.data(), kEeWork0Addr + 0x1Cu, kChunkLen);
            writeGuestU32(env.rdram.data(), kEeWork0Addr + kWorkLen - sizeof(uint32_t), 1u);

            const Ps2SifDmaTransfer desc0{
                kEeWork0Addr,
                kIopWork0Addr,
                static_cast<int32_t>(kWorkLen),
                0};
            std::memcpy(env.rdram.data() + kDesc0Addr, &desc0, sizeof(desc0));

            setRegU32(env.ctx, 4, kDesc0Addr);
            setRegU32(env.ctx, 5, 1u);
            ps2_stubs::sceSifSetDma(env.rdram.data(), &env.ctx, &env.runtime);
            t.IsTrue(getRegS32(env.ctx, 2) > 0, "sceSifSetDma should succeed for the SJX transport");
            t.IsTrue(
                serviceAllScheduledBoundaries(env),
                "scheduled SJX payload transfer should complete");
            t.Equals(env.rdram[kEeWork0Addr + 0x11u], static_cast<uint8_t>(0u),
                     "SJX DMA ack should still rewrite the response line to room");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, sjrmtHandle);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, 1u);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 0x429u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 8u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(readGuestU32(env.rdram.data(), kRecvAddr), 0u,
                     "active PS2RNA playback should drain remote SJRMT data instead of leaving it queued forever");

            writeGuestU32(env.rdram.data(), kSendAddr + 0x00u, sjrmtHandle);
            writeGuestU32(env.rdram.data(), kSendAddr + 0x04u, 0u);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 0x429u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, kSendAddr);
            setRegU32(env.ctx, 8, 8u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(readGuestU32(env.rdram.data(), kRecvAddr), kWorkLen,
                     "drained PS2RNA playback should return remote SJRMT room to full capacity");
        });

        tc.Run("resetSifState seeds boot-ready SIF registers", [](TestCase &t)
        {
            TestEnv env;

            auto getReg = [&](uint32_t reg) -> uint32_t
            {
                setRegU32(env.ctx, 4, reg);
                ps2_stubs::sceSifGetReg(env.rdram.data(), &env.ctx, &env.runtime);
                return ::getRegU32(&env.ctx, 2);
            };

            t.Equals(getReg(0x4u), 0x00020000u, "SIF boot status register should expose ready bit by default");
            t.Equals(getReg(0x80000000u), 0u, "SIF main-address register should default to zero");
            t.Equals(getReg(0x80000001u), 0u, "SIF sub-address register should default to zero");
            t.Equals(getReg(0x80000002u), 0u, "SIF mscom register should default to zero");
        });

        tc.Run("sceSifExitCmd restores default boot-ready SIF registers", [](TestCase &t)
        {
            TestEnv env;

            setRegU32(env.ctx, 4, 0x4u);
            setRegU32(env.ctx, 5, 0x12340000u);
            ps2_stubs::sceSifSetReg(env.rdram.data(), &env.ctx, &env.runtime);

            setRegU32(env.ctx, 4, 0x80000002u);
            setRegU32(env.ctx, 5, 0x89ABCDEFu);
            ps2_stubs::sceSifSetReg(env.rdram.data(), &env.ctx, &env.runtime);

            ps2_stubs::sceSifExitCmd(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), 0, "sceSifExitCmd should succeed");

            auto getReg = [&](uint32_t reg) -> uint32_t
            {
                setRegU32(env.ctx, 4, reg);
                ps2_stubs::sceSifGetReg(env.rdram.data(), &env.ctx, &env.runtime);
                return ::getRegU32(&env.ctx, 2);
            };

            t.Equals(getReg(0x4u), 0x00020000u, "sceSifExitCmd should restore the boot-ready status bit");
            t.Equals(getReg(0x80000002u), 0u, "sceSifExitCmd should clear transient mscom state");
        });

        tc.Run("sceSifSetDma rejects invalid descriptors without partial writes", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kDescAddr = 0x00021000u;
            constexpr uint32_t kSrcA = 0x00021100u;
            constexpr uint32_t kDstA = 0x00021200u;
            constexpr uint32_t kSrcB = 0x00021300u;
            constexpr uint32_t kInvalidDstB = PS2_IOP_RAM_SIZE;

            std::array<uint8_t, 8> payloadA{};
            for (size_t i = 0; i < payloadA.size(); ++i)
            {
                payloadA[i] = static_cast<uint8_t>(0x70u + i);
            }
            std::array<uint8_t, 8> payloadB{};
            for (size_t i = 0; i < payloadB.size(); ++i)
            {
                payloadB[i] = static_cast<uint8_t>(0x90u + i);
            }

            std::memcpy(env.rdram.data() + kSrcA, payloadA.data(), payloadA.size());
            std::memcpy(env.rdram.data() + kSrcB, payloadB.data(), payloadB.size());
            std::memset(env.runtime.memory().getIOPRAM() + kDstA, 0x5Au, payloadA.size());

            const Ps2SifDmaTransfer descs[2] = {
                {kSrcA, kDstA, static_cast<int32_t>(payloadA.size()), 0},
                {kSrcB, kInvalidDstB, static_cast<int32_t>(payloadB.size()), 0}};
            std::memcpy(env.rdram.data() + kDescAddr, descs, sizeof(descs));

            setRegU32(env.ctx, 4, kDescAddr);
            setRegU32(env.ctx, 5, 2u);
            ps2_stubs::sceSifSetDma(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), 0, "sceSifSetDma should fail when any descriptor is invalid");

            const std::array<uint8_t, 8> expectedUnchanged{
                0x5A, 0x5A, 0x5A, 0x5A, 0x5A, 0x5A, 0x5A, 0x5A};
            t.IsTrue(std::memcmp(env.runtime.memory().getIOPRAM() + kDstA,
                                 expectedUnchanged.data(),
                                 expectedUnchanged.size()) == 0,
                     "failed multi-descriptor sceSifSetDma should not partially write earlier descriptors");
        });

        tc.Run("sceSifSetDma enforces descriptor count limit", [](TestCase &t)
        {
            TestEnv env;
            constexpr uint32_t kDescAddr = 0x00022000u;

            setRegU32(env.ctx, 4, kDescAddr);
            setRegU32(env.ctx, 5, 33u);
            ps2_stubs::sceSifSetDma(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), 0, "sceSifSetDma should reject count > 32");
        });

        tc.Run("sceSifGetOtherData copies payload and writes receive metadata", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kRdAddr = 0x00023000u;
            constexpr uint32_t kSrcAddr = 0x00023100u;
            constexpr uint32_t kDstAddr = 0x00023200u;
            constexpr uint32_t kSize = 20u;

            std::array<uint8_t, kSize> payload{};
            for (size_t i = 0; i < payload.size(); ++i)
            {
                payload[i] = static_cast<uint8_t>((i * 7u) & 0xFFu);
            }
            std::memcpy(env.rdram.data() + kSrcAddr, payload.data(), payload.size());
            std::memset(env.rdram.data() + kDstAddr, 0, payload.size());
            std::memset(env.rdram.data() + kRdAddr, 0, sizeof(SifRpcReceiveData));

            setRegU32(env.ctx, 4, kRdAddr);
            setRegU32(env.ctx, 5, kSrcAddr);
            setRegU32(env.ctx, 6, kDstAddr);
            setRegU32(env.ctx, 7, kSize);
            ps2_stubs::sceSifGetOtherData(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), 0, "sceSifGetOtherData should succeed for valid transfer");

            t.IsTrue(std::memcmp(env.rdram.data() + kDstAddr, payload.data(), payload.size()) == 0,
                     "sceSifGetOtherData should copy payload");

            const SifRpcReceiveData rd = *reinterpret_cast<const SifRpcReceiveData *>(env.rdram.data() + kRdAddr);
            t.Equals(rd.src, kSrcAddr, "receive metadata src should be populated");
            t.Equals(rd.dest, kDstAddr, "receive metadata dest should be populated");
            t.Equals(static_cast<uint32_t>(rd.size), kSize, "receive metadata size should be populated");
        });

        tc.Run("sceSifGetOtherData preserves live sound-status sums when compat backfill is enabled", [](TestCase &t)
        {
            TestEnv env;
            configureProfile(env, "slus_201.84");

            constexpr uint32_t kRdAddr = 0x00023300u;
            constexpr uint32_t kDstAddr = 0x00023400u;
            constexpr uint32_t kSize = 0x42u;
            constexpr uint32_t kPrimarySeCheckAddr = 0x01E0EF10u;
            constexpr uint32_t kPrimaryMidiCheckAddr = 0x01E0EF20u;
            constexpr uint32_t kMidiSumOffset = 0x1Eu;
            constexpr uint32_t kSeSumOffset = 0x26u;
            constexpr uint32_t kBank = 1u;

            constexpr uint32_t kClientAddr = 0x00023500u;
            constexpr uint32_t kRecvAddr = 0x00023600u;
            constexpr uint32_t kSid = 1u;

            ps2_syscalls::SifInitRpc(env.rdram.data(), &env.ctx, &env.runtime);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, kSid);
            setRegU32(env.ctx, 6, 0u);
            ps2_syscalls::SifBindRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "SifBindRpc should succeed for sound-driver sid");

            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 0x12u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, 0u);
            setRegU32(env.ctx, 8, 0u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            const uint32_t kSrcAddr = readGuestU32(env.rdram.data(), kRecvAddr);

            std::memset(env.rdram.data() + kDstAddr, 0, kSize);
            std::memset(env.rdram.data() + kRdAddr, 0, sizeof(SifRpcReceiveData));

            writeGuestS16(env.rdram.data(), kSrcAddr + kSeSumOffset + (kBank * 2u), static_cast<int16_t>(0x1357));
            writeGuestS16(env.rdram.data(), kSrcAddr + kMidiSumOffset + (kBank * 2u), static_cast<int16_t>(0x2468));

            writeGuestS16(env.rdram.data(), kPrimarySeCheckAddr + (kBank * 2u), static_cast<int16_t>(0x7B7B));
            writeGuestS16(env.rdram.data(), kPrimaryMidiCheckAddr + (kBank * 2u), static_cast<int16_t>(0x6A6A));

            setRegU32(env.ctx, 4, kRdAddr);
            setRegU32(env.ctx, 5, kSrcAddr);
            setRegU32(env.ctx, 6, kDstAddr);
            setRegU32(env.ctx, 7, kSize);
            ps2_stubs::sceSifGetOtherData(env.rdram.data(), &env.ctx, &env.runtime);

            t.Equals(getRegS32(env.ctx, 2), 0,
                     "sceSifGetOtherData should succeed for sound-status transfer");
            t.Equals(readGuestS16(env.rdram.data(), kDstAddr + kSeSumOffset + (kBank * 2u)),
                     static_cast<int16_t>(0x1357),
                     "live se_sum for the active bank should not be clobbered by compat check arrays");
            t.Equals(readGuestS16(env.rdram.data(), kDstAddr + kMidiSumOffset + (kBank * 2u)),
                     static_cast<int16_t>(0x2468),
                     "live midi_sum for the active bank should not be clobbered by compat check arrays");
        });

        tc.Run("sceSifGetOtherData backfills zero sound-status sums for later banks", [](TestCase &t)
        {
            TestEnv env;
            configureProfile(env, "slus_201.84");

            constexpr uint32_t kRdAddr = 0x00023700u;
            constexpr uint32_t kDstAddr = 0x00023800u;
            constexpr uint32_t kSize = 0x42u;
            constexpr uint32_t kPrimarySeCheckAddr = 0x01E0EF10u;
            constexpr uint32_t kPrimaryMidiCheckAddr = 0x01E0EF20u;
            constexpr uint32_t kMidiSumOffset = 0x1Eu;
            constexpr uint32_t kSeSumOffset = 0x26u;
            constexpr uint32_t kLiveBank = 0u;
            constexpr uint32_t kPendingBank = 1u;

            constexpr uint32_t kClientAddr = 0x00023900u;
            constexpr uint32_t kRecvAddr = 0x00023A00u;

            ps2_syscalls::SifInitRpc(env.rdram.data(), &env.ctx, &env.runtime);
            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 1u);
            setRegU32(env.ctx, 6, 0u);
            ps2_syscalls::SifBindRpc(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), KE_OK, "SifBindRpc should succeed for sound-driver sid");

            setRegU32(env.ctx, 4, kClientAddr);
            setRegU32(env.ctx, 5, 0x12u);
            setRegU32(env.ctx, 6, 0u);
            setRegU32(env.ctx, 7, 0u);
            setRegU32(env.ctx, 8, 0u);
            setRegU32(env.ctx, 9, kRecvAddr);
            setRegU32(env.ctx, 10, 4u);
            setRegU32(env.ctx, 11, 0u);
            ps2_syscalls::SifCallRpc(env.rdram.data(), &env.ctx, &env.runtime);
            const uint32_t kSrcAddr = readGuestU32(env.rdram.data(), kRecvAddr);

            std::memset(env.rdram.data() + kDstAddr, 0, kSize);
            std::memset(env.rdram.data() + kRdAddr, 0, sizeof(SifRpcReceiveData));

            writeGuestS16(env.rdram.data(), kSrcAddr + kSeSumOffset + (kLiveBank * 2u), static_cast<int16_t>(0x1111));
            writeGuestS16(env.rdram.data(), kSrcAddr + kMidiSumOffset + (kLiveBank * 2u), static_cast<int16_t>(0x2222));

            writeGuestS16(env.rdram.data(), kPrimarySeCheckAddr + (kPendingBank * 2u), static_cast<int16_t>(0x3333));
            writeGuestS16(env.rdram.data(), kPrimaryMidiCheckAddr + (kPendingBank * 2u), static_cast<int16_t>(0x4444));

            setRegU32(env.ctx, 4, kRdAddr);
            setRegU32(env.ctx, 5, kSrcAddr);
            setRegU32(env.ctx, 6, kDstAddr);
            setRegU32(env.ctx, 7, kSize);
            ps2_stubs::sceSifGetOtherData(env.rdram.data(), &env.ctx, &env.runtime);

            t.Equals(getRegS32(env.ctx, 2), 0,
                     "sceSifGetOtherData should succeed for later-bank sound-status transfer");
            t.Equals(readGuestS16(env.rdram.data(), kDstAddr + kSeSumOffset + (kLiveBank * 2u)),
                     static_cast<int16_t>(0x1111),
                     "existing live se_sum values should remain intact");
            t.Equals(readGuestS16(env.rdram.data(), kDstAddr + kMidiSumOffset + (kLiveBank * 2u)),
                     static_cast<int16_t>(0x2222),
                     "existing live midi_sum values should remain intact");
            t.Equals(readGuestS16(env.rdram.data(), kDstAddr + kSeSumOffset + (kPendingBank * 2u)),
                     static_cast<int16_t>(0x3333),
                     "zero se_sum slots should backfill from compat tables for later banks");
            t.Equals(readGuestS16(env.rdram.data(), kDstAddr + kMidiSumOffset + (kPendingBank * 2u)),
                     static_cast<int16_t>(0x4444),
                     "zero midi_sum slots should backfill from compat tables for later banks");
        });

        tc.Run("sceSifGetOtherData rejects unsupported guest segments", [](TestCase &t)
        {
            TestEnv env;

            constexpr uint32_t kRdAddr = 0x00024000u;
            constexpr uint32_t kDstAddr = 0x00024100u;
            constexpr uint32_t kInvalidSrcAddr = 0xE0000200u;
            constexpr uint32_t kSize = 16u;

            std::memset(env.rdram.data() + kDstAddr, 0xA5, kSize);
            writeGuestU32(env.rdram.data(), kRdAddr + 0x10u, 0x11111111u);
            writeGuestU32(env.rdram.data(), kRdAddr + 0x14u, 0x22222222u);
            writeGuestU32(env.rdram.data(), kRdAddr + 0x18u, 0x33333333u);

            setRegU32(env.ctx, 4, kRdAddr);
            setRegU32(env.ctx, 5, kInvalidSrcAddr);
            setRegU32(env.ctx, 6, kDstAddr);
            setRegU32(env.ctx, 7, kSize);
            ps2_stubs::sceSifGetOtherData(env.rdram.data(), &env.ctx, &env.runtime);
            t.Equals(getRegS32(env.ctx, 2), -1, "sceSifGetOtherData should fail for unsupported source segment");

            std::array<uint8_t, kSize> expected{};
            expected.fill(0xA5u);
            t.IsTrue(std::memcmp(env.rdram.data() + kDstAddr, expected.data(), expected.size()) == 0,
                     "failed sceSifGetOtherData should not modify destination");
            t.Equals(readGuestU32(env.rdram.data(), kRdAddr + 0x10u), 0x11111111u,
                     "failed sceSifGetOtherData should not overwrite rd metadata");
        });
    });
}
