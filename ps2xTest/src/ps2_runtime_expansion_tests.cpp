#include "MiniTest.h"
#include "ps2recomp/code_generator.h"
#include "ps2recomp/instructions.h"
#include "ps2recomp/r5900_decoder.h"
#include "ps2recomp/types.h"
#include "ps2_runtime.h"
#include "runtime/ps2_memory.h"
#include "runtime/ps2_vu_recompiler.h"
#include "ps2_syscalls.h"
#include "ps2_stubs.h"
#include "runtime/ps2_gs_gpu.h"
#include "runtime/ps2_gs_psmct32.h"
#include "ps2_runtime_macros.h"
#include "Stubs/MPEG.h"
#include "Stubs/CD.h"
#include "Stubs/Audio.h"
#include "Stubs/GS.h"
#include "Stubs/VU.h"

#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if PS2X_HAS_FFMPEG
extern "C"
{
#include <libavcodec/avcodec.h>
}
#endif

using namespace ps2recomp;
using namespace ps2_syscalls;

namespace
{
    constexpr uint32_t COP0_CAUSE_BD = 0x80000000u;
    constexpr uint32_t COP0_CAUSE_EXCCODE_MASK = 0x0000007Cu;
    constexpr uint32_t COP0_STATUS_EXL = 0x00000002u;
    constexpr uint32_t COP0_STATUS_BEV = 0x00400000u;
    constexpr uint32_t EXCEPTION_VECTOR_GENERAL = 0x80000180u;
    constexpr uint32_t EXCEPTION_VECTOR_BOOT = 0xBFC00380u;
    constexpr uint32_t EXCEPTION_VECTOR_TLB_REFILL = 0x80000000u;
    constexpr uint32_t EXCEPTION_VECTOR_TLB_REFILL_BOOT = 0xBFC00200u;

    constexpr int KE_OK = 0;

    void testPrimaryModuleFunction(
        uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        SET_GPR_U32(ctx, 2, 0x11111111u);
    }

    void testOverlayModuleFunction(
        uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        SET_GPR_U32(ctx, 2, 0x22222222u);
    }

    void setRegU32(R5900Context &ctx, int reg, uint32_t value)
    {
        ctx.r[reg] = _mm_set_epi64x(0, static_cast<int64_t>(value));
    }

    int32_t getRegS32(const R5900Context &ctx, int reg)
    {
        return static_cast<int32_t>(::getRegU32(&ctx, reg));
    }

    uint32_t makeVifCmd(uint8_t opcode, uint8_t num, uint16_t imm)
    {
        return (static_cast<uint32_t>(opcode) << 24) |
               (static_cast<uint32_t>(num) << 16) |
               static_cast<uint32_t>(imm);
    }

    uint64_t makeDmaTag(
        uint16_t qwc, uint8_t id, uint32_t addr,
        bool irq = false)
    {
        uint64_t tag = static_cast<uint64_t>(qwc);
        tag |= static_cast<uint64_t>(id & 0x7u) << 28u;
        if (irq)
            tag |= 1ull << 31u;
        tag |= static_cast<uint64_t>(
                   addr & 0x7FFFFFFFu)
               << 32u;
        return tag;
    }

    uint32_t makeVuLq(uint8_t dest, uint8_t targetVf, uint8_t baseVi, int16_t imm)
    {
        return (static_cast<uint32_t>(dest & 0xFu) << 21) |
               (static_cast<uint32_t>(targetVf & 0x1Fu) << 16) |
               (static_cast<uint32_t>(baseVi & 0x1Fu) << 11) |
               (static_cast<uint32_t>(imm) & 0x7FFu);
    }

    uint32_t makeVuSq(uint8_t dest, uint8_t sourceVf, uint8_t baseVi, int16_t imm)
    {
        return (0x01u << 25) |
               (static_cast<uint32_t>(dest & 0xFu) << 21) |
               (static_cast<uint32_t>(baseVi & 0x1Fu) << 16) |
               (static_cast<uint32_t>(sourceVf & 0x1Fu) << 11) |
               (static_cast<uint32_t>(imm) & 0x7FFu);
    }

    uint32_t makeVuAdd(uint8_t dest, uint8_t fd, uint8_t fs, uint8_t ft)
    {
        return (static_cast<uint32_t>(dest & 0xFu) << 21) |
               (static_cast<uint32_t>(ft & 0x1Fu) << 16) |
               (static_cast<uint32_t>(fs & 0x1Fu) << 11) |
               (static_cast<uint32_t>(fd & 0x1Fu) << 6) |
               0x28u;
    }

    uint32_t makeVuIsubiu(uint8_t it, uint8_t is, uint16_t imm)
    {
        return (0x09u << 25) |
               (static_cast<uint32_t>((imm >> 11) & 0xFu) << 21) |
               (static_cast<uint32_t>(it & 0xFu) << 16) |
               (static_cast<uint32_t>(is & 0xFu) << 11) |
               (static_cast<uint32_t>(imm) & 0x7FFu);
    }

    uint32_t makeVuIaddiu(uint8_t it, uint8_t is, uint16_t imm)
    {
        return (0x08u << 25) |
               (static_cast<uint32_t>((imm >> 11) & 0xFu) << 21) |
               (static_cast<uint32_t>(it & 0xFu) << 16) |
               (static_cast<uint32_t>(is & 0xFu) << 11) |
               (static_cast<uint32_t>(imm) & 0x7FFu);
    }

    uint32_t makeVuIbeq(uint8_t it, uint8_t is, int16_t imm)
    {
        return (0x28u << 25) |
               (static_cast<uint32_t>(it & 0xFu) << 16) |
               (static_cast<uint32_t>(is & 0xFu) << 11) |
               (static_cast<uint32_t>(imm) & 0x7FFu);
    }

    uint32_t makeVuBranch(int16_t imm)
    {
        return (0x20u << 25) |
               (static_cast<uint32_t>(imm) & 0x7FFu);
    }

    uint32_t makeVuIbgtz(uint8_t is, int16_t imm)
    {
        return (0x2Du << 25) |
               (static_cast<uint32_t>(is & 0xFu) << 11) |
               (static_cast<uint32_t>(imm) & 0x7FFu);
    }

    uint32_t makeVuLowerSpecial(
        uint8_t specialOp, uint8_t is, uint8_t it = 0u,
        uint8_t id = 0u, uint8_t dest = 0u)
    {
        return (0x40u << 25) |
               (static_cast<uint32_t>(dest & 0xFu) << 21) |
               (static_cast<uint32_t>(it & 0x1Fu) << 16) |
               (static_cast<uint32_t>(is & 0x1Fu) << 11) |
               (static_cast<uint32_t>(id & 0x1Fu) << 6) |
               (static_cast<uint32_t>(specialOp & 0x7Cu) << 4) |
               static_cast<uint32_t>(specialOp & 0x3u) |
               0x3Cu;
    }

    uint64_t makeGifTag(
        uint16_t nloop, uint8_t flg, uint8_t nreg,
        bool eop = true)
    {
        uint64_t tag = static_cast<uint64_t>(nloop & 0x7FFFu);
        if (eop)
            tag |= 1ull << 15u;
        tag |= static_cast<uint64_t>(flg & 0x3u) << 58u;
        tag |= static_cast<uint64_t>(nreg & 0xFu) << 60u;
        return tag;
    }

    void writeGifImagePacket(
        uint8_t *destination, uint32_t qwc,
        uint8_t payloadByte)
    {
        if (!destination || qwc == 0u)
            return;
        std::memset(destination, 0, qwc * 16u);
        const uint64_t tag = makeGifTag(
            static_cast<uint16_t>(qwc - 1u),
            GIF_FMT_IMAGE, 0u);
        std::memcpy(destination, &tag, sizeof(tag));
        if (qwc > 1u)
        {
            std::memset(
                destination + 16u, payloadByte,
                (qwc - 1u) * 16u);
        }
    }

    void writeVuInstructionPair(uint8_t *code, uint32_t pc, uint32_t lower, uint32_t upper)
    {
        std::memcpy(code + pc, &lower, sizeof(lower));
        std::memcpy(code + pc + sizeof(lower), &upper, sizeof(upper));
    }

    bool hasSignedRdWrite(const std::string &generated, uint8_t rd)
    {
        if (rd == 0u)
        {
            return false;
        }

        const std::string needle = "SET_GPR_S32(ctx, " + std::to_string(rd) + ",";
        return generated.find(needle) != std::string::npos;
    }

    template <typename Function>
    bool raisesGuestException(Function &&function)
    {
        try
        {
            function();
        }
        catch (const PS2GuestException &)
        {
            return true;
        }
        return false;
    }

    template <typename Predicate>
    bool waitUntil(Predicate pred, std::chrono::milliseconds timeout)
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (pred())
            {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return pred();
    }

    uint32_t frameOffsetBytes(uint32_t x, uint32_t y, uint32_t fbw)
    {
        return GSPSMCT32::addrPSMCT32(0u, (fbw != 0u) ? fbw : 1u, x, y);
    }

    void testRuntimeWorkerLoop(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!ctx || !runtime)
        {
            return;
        }

        // Keep touching guest memory so teardown races are easier to catch.
        (void)Ps2FastRead64(rdram, static_cast<uint32_t>(0x01FFFFF8u + (ctx->insn_count & 0x7u)));
        ++ctx->insn_count;

        if (runtime->isStopRequested())
        {
            ctx->pc = 0u;
            return;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::atomic<int32_t> gSerializedGuestActive{0};
    std::atomic<int32_t> gSerializedGuestMaxActive{0};
    std::atomic<int32_t> gPreemptionPolicyEntryCount{0};
    std::atomic<bool> gPreemptionPolicyAllowFirstProbe{false};
    std::atomic<bool> gPreemptionPolicyPeerRan{false};

    void testSerializedGuestStep(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        const int32_t active = gSerializedGuestActive.fetch_add(1, std::memory_order_acq_rel) + 1;
        int32_t observedMax = gSerializedGuestMaxActive.load(std::memory_order_relaxed);
        while (observedMax < active &&
               !gSerializedGuestMaxActive.compare_exchange_weak(
                   observedMax,
                   active,
                   std::memory_order_release,
                   std::memory_order_relaxed))
        {
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        gSerializedGuestActive.fetch_sub(1, std::memory_order_acq_rel);
        if (ctx)
        {
            ctx->pc = 0u;
        }
    }

    void testPreemptionPolicyStep(uint8_t *, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!ctx || !runtime)
        {
            return;
        }

        const int32_t entryIndex = gPreemptionPolicyEntryCount.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (entryIndex == 1)
        {
            while (!gPreemptionPolicyAllowFirstProbe.load(std::memory_order_acquire))
            {
                std::this_thread::yield();
            }

            bool shouldPreempt = false;
            for (int attempt = 0; attempt < 256 &&
                                  !shouldPreempt;
                 ++attempt)
            {
                shouldPreempt =
                    runtime->checkpointGuestExecution(
                        ctx) ==
                    PS2GuestCheckpointResult::
                        ExitToDispatcher;
            }
            setRegU32(*ctx, 2, shouldPreempt ? 1u : 0u);
        }
        else
        {
            gPreemptionPolicyPeerRan.store(true, std::memory_order_release);
            setRegU32(*ctx, 2, 2u);
        }

        ctx->pc = 0u;
    }

    void testResumeOwnerFallbackHandler(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        if (ctx)
        {
            setRegU32(*ctx, 2, 0x00ABC123u);
            ctx->pc = 0u;
        }
    }

    void testResumeNextFunctionHandler(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        if (ctx)
        {
            setRegU32(*ctx, 2, 0x00555555u);
            ctx->pc = 0u;
        }
    }

    void testGuestBranchImplicitReturnHandler(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        if (ctx)
        {
            setRegU32(*ctx, 2, 0x00FACE42u);
            // Leave ctx->pc at the entry point. dispatchGuestBranch should convert
            // unchanged call PC into the supplied fallthrough PC for call-like edges.
        }
    }

    void testGuestBranchTransferHandler(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        if (ctx)
        {
            setRegU32(*ctx, 2, 0x00BEEFu);
            ctx->pc = 0x33330000u;
        }
    }

    void testGuestBranchPreemptionHandler(uint8_t *,
                                          R5900Context *ctx,
                                          PS2Runtime *runtime)
    {
        if (!ctx || !runtime)
        {
            return;
        }

        uint32_t probes = 0u;
        while (probes < 256u)
        {
            ++probes;
            if (runtime->checkpointGuestExecution(
                    ctx) ==
                PS2GuestCheckpointResult::
                    ExitToDispatcher)
            {
                setRegU32(*ctx, 2, probes);
                return;
            }
        }
    }

    constexpr uint32_t kExceptionUnwindEntry = 0x160000u;
    constexpr uint32_t kExceptionUnwindNested = 0x160100u;
    constexpr uint32_t kExceptionUnwindVector = 0x00000180u;

    void testExceptionUnwindNested(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ctx->pc = kExceptionUnwindNested;
        const uint32_t value = runtime->Load32(rdram, ctx, 0x00000002u);
        setRegU32(*ctx, 2, value);
        setRegU32(*ctx, 3, 0x33333333u);
        ctx->pc = 0u;
    }

    void testExceptionUnwindEntryFunction(uint8_t *rdram,
                                          R5900Context *ctx,
                                          PS2Runtime *runtime)
    {
        ctx->pc = kExceptionUnwindNested;
        PS2Runtime::RecompiledFunction nested = runtime->lookupFunction(ctx->pc);
        nested(rdram, ctx, runtime);
        setRegU32(*ctx, 4, 0x44444444u);
        ctx->pc = 0u;
    }

    void testExceptionUnwindVectorHandler(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        setRegU32(*ctx, 5, 0x55555555u);
        ctx->pc = 0u;
    }

    constexpr uint32_t kAsyncCounterAddr = 0x2400u;

    void testWaitForAsyncCounter(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (!rdram || !ctx)
        {
            return;
        }

        uint32_t counter = 0u;
        do
        {
            std::memcpy(&counter, rdram + kAsyncCounterAddr, sizeof(counter));
            if (counter == 0u)
            {
                if (runtime != nullptr)
                {
                    runtime->yieldGuestExecutionAfterWake();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (runtime &&
                    runtime->checkpointGuestExecution(
                        ctx) ==
                        PS2GuestCheckpointResult::
                            ExitToDispatcher)
                {
                    // Generated back edges return to the dispatcher at this
                    // point, allowing a pending interrupt to acquire the
                    // serialized guest execution scope.
                    return;
                }
            }
        } while (counter == 0u);

        ctx->pc = 0u;
    }

    void testSignalAsyncCounter(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        if (rdram)
        {
            const uint32_t counter = 1u;
            std::memcpy(rdram + kAsyncCounterAddr, &counter, sizeof(counter));
        }

        if (ctx)
        {
            ctx->pc = 0u;
        }
    }

    std::atomic<uint32_t> gAsyncCallbackObservedSp{0u};
    std::atomic<uint32_t> gAsyncCallbackObservedGp{0u};

    void testRecordAsyncCallbackStack(uint8_t *, R5900Context *ctx, PS2Runtime *)
    {
        if (!ctx)
        {
            return;
        }

        gAsyncCallbackObservedSp.store(::getRegU32(ctx, 29), std::memory_order_release);
        gAsyncCallbackObservedGp.store(::getRegU32(ctx, 28), std::memory_order_release);
        ctx->pc = 0u;
    }

    std::atomic<uint32_t> gMpegStreamCallbackCount{0u};
    std::atomic<uint32_t> gMpegStreamCallbackMpeg{0u};
    std::atomic<uint32_t> gMpegStreamCallbackType{0u};
    std::atomic<uint32_t> gMpegStreamCallbackDataAddr{0u};
    std::atomic<uint32_t> gMpegStreamCallbackLen{0u};
    std::atomic<uint32_t> gMpegStreamCallbackUserData{0u};
    std::atomic<uint32_t> gMpegStreamCallbackSp{0u};
    std::atomic<uint32_t> gMpegStreamCallbackReturn{1u};
    std::mutex gMpegStreamCallbackPayloadMutex;
    std::vector<uint8_t> gMpegStreamCallbackPayload;

    void testRecordMpegStreamCallback(uint8_t *rdram, R5900Context *ctx, PS2Runtime *)
    {
        if (!rdram || !ctx)
        {
            return;
        }

        const uint32_t cbData = ::getRegU32(ctx, 5);
        uint32_t type = 0u;
        uint32_t dataAddr = 0u;
        uint32_t len = 0u;
        std::memcpy(&type, rdram + cbData + 0x00u, sizeof(type));
        std::memcpy(&dataAddr, rdram + cbData + 0x08u, sizeof(dataAddr));
        std::memcpy(&len, rdram + cbData + 0x0Cu, sizeof(len));

        {
            std::lock_guard<std::mutex> lock(gMpegStreamCallbackPayloadMutex);
            gMpegStreamCallbackPayload.clear();
            if (len != 0u)
            {
                const uint8_t *const payload = getConstMemPtr(rdram, dataAddr);
                const uint8_t *const payloadLast =
                    getConstMemPtr(rdram, dataAddr + len - 1u);
                if (payload && payloadLast && payloadLast >= payload &&
                    static_cast<size_t>(payloadLast - payload) == len - 1u)
                {
                    gMpegStreamCallbackPayload.assign(payload, payload + len);
                }
            }
        }

        gMpegStreamCallbackMpeg.store(::getRegU32(ctx, 4), std::memory_order_release);
        gMpegStreamCallbackType.store(type, std::memory_order_release);
        gMpegStreamCallbackDataAddr.store(dataAddr, std::memory_order_release);
        gMpegStreamCallbackLen.store(len, std::memory_order_release);
        gMpegStreamCallbackUserData.store(::getRegU32(ctx, 6), std::memory_order_release);
        gMpegStreamCallbackSp.store(::getRegU32(ctx, 29), std::memory_order_release);
        gMpegStreamCallbackCount.fetch_add(1u, std::memory_order_acq_rel);
        setRegU32(*ctx, 2, gMpegStreamCallbackReturn.load(std::memory_order_acquire));
        ctx->pc = 0u;
    }

    void testRejectSecondMpegStreamCallback(uint8_t *rdram,
                                            R5900Context *ctx,
                                            PS2Runtime *runtime)
    {
        testRecordMpegStreamCallback(rdram, ctx, runtime);
        if (ctx &&
            gMpegStreamCallbackCount.load(std::memory_order_acquire) == 2u)
        {
            setRegU32(*ctx, 2, 0u);
        }
    }

#if PS2X_HAS_FFMPEG
    struct AvCodecContextDeleter
    {
        void operator()(AVCodecContext *context) const
        {
            avcodec_free_context(&context);
        }
    };

    struct AvFrameDeleter
    {
        void operator()(AVFrame *frame) const
        {
            av_frame_free(&frame);
        }
    };

    struct AvPacketDeleter
    {
        void operator()(AVPacket *packet) const
        {
            av_packet_free(&packet);
        }
    };

    std::vector<uint8_t> makeSyntheticMpeg2StartupStream()
    {
        constexpr int kWidth = 32;
        constexpr int kHeight = 16;
        constexpr int kFrameCount = 4;

        const AVCodec *encoder = avcodec_find_encoder(AV_CODEC_ID_MPEG2VIDEO);
        std::unique_ptr<AVCodecContext, AvCodecContextDeleter> codec(
            encoder ? avcodec_alloc_context3(encoder) : nullptr);
        std::unique_ptr<AVFrame, AvFrameDeleter> frame(av_frame_alloc());
        std::unique_ptr<AVPacket, AvPacketDeleter> packet(av_packet_alloc());
        if (!codec || !frame || !packet)
        {
            return {};
        }

        codec->bit_rate = 200000;
        codec->width = kWidth;
        codec->height = kHeight;
        codec->time_base = AVRational{1, 30};
        codec->framerate = AVRational{30, 1};
        codec->gop_size = 12;
        codec->max_b_frames = 0;
        codec->pix_fmt = AV_PIX_FMT_YUV420P;
        codec->thread_count = 1;
        if (avcodec_open2(codec.get(), encoder, nullptr) < 0)
        {
            return {};
        }

        frame->format = codec->pix_fmt;
        frame->width = kWidth;
        frame->height = kHeight;
        if (av_frame_get_buffer(frame.get(), 32) < 0)
        {
            return {};
        }

        std::vector<uint8_t> elementaryStream;
        const auto receivePackets = [&]()
        {
            while (true)
            {
                const int result = avcodec_receive_packet(codec.get(), packet.get());
                if (result == AVERROR(EAGAIN) || result == AVERROR_EOF)
                {
                    return true;
                }
                if (result < 0)
                {
                    return false;
                }
                elementaryStream.insert(
                    elementaryStream.end(), packet->data, packet->data + packet->size);
                av_packet_unref(packet.get());
            }
        };

        for (int frameIndex = 0; frameIndex < kFrameCount; ++frameIndex)
        {
            if (av_frame_make_writable(frame.get()) < 0)
            {
                return {};
            }
            for (int row = 0; row < kHeight; ++row)
            {
                std::memset(
                    frame->data[0] + row * frame->linesize[0],
                    64,
                    kWidth);
            }
            for (int row = 0; row < kHeight / 2; ++row)
            {
                std::memset(frame->data[1] + row * frame->linesize[1], 96, kWidth / 2);
                std::memset(frame->data[2] + row * frame->linesize[2], 160, kWidth / 2);
            }
            frame->pts = frameIndex;
            if (avcodec_send_frame(codec.get(), frame.get()) < 0 || !receivePackets())
            {
                return {};
            }
        }
        if (avcodec_send_frame(codec.get(), nullptr) < 0 || !receivePackets())
        {
            return {};
        }

        std::vector<size_t> pictureStarts;
        for (size_t offset = 0u; offset + 4u <= elementaryStream.size(); ++offset)
        {
            if (elementaryStream[offset + 0u] == 0x00u &&
                elementaryStream[offset + 1u] == 0x00u &&
                elementaryStream[offset + 2u] == 0x01u &&
                elementaryStream[offset + 3u] == 0x00u)
            {
                pictureStarts.push_back(offset);
            }
        }
        if (pictureStarts.size() < kFrameCount)
        {
            return {};
        }

        // Keep three complete pictures but omit both the next GOP and the
        // sequence-end marker. A decoder must emit the initial I picture once
        // the following P reference arrives; waiting for another I picture can
        // deadlock a demuxer whose audio stream is already backpressured.
        elementaryStream.resize(pictureStarts[3]);
        return elementaryStream;
    }
#endif

}

void register_ps2_runtime_expansion_tests()
{
    MiniTest::Case("PS2RuntimeExpansion", [](TestCase &tc)
    {
        tc.Run("guest PCs include mapped function names and offsets", [](TestCase &t)
        {
            t.Equals(PS2Runtime::formatGuestPc(0x00001000u),
                     std::string("0x1000<test_function>"),
                     "an exact mapped PC should include its function name");
            t.Equals(PS2Runtime::formatGuestPc(0x0000101Cu),
                     std::string("0x101c<test_function+0x1c>"),
                     "an interior mapped PC should include its function-relative offset");
            t.Equals(PS2Runtime::formatGuestPc(0x80001004u),
                     std::string("0x80001004<test_function+0x4>"),
                     "a direct-mapped RDRAM alias should resolve against the physical function range");
            t.Equals(PS2Runtime::formatGuestPc(0x00000180u),
                     std::string("0x180"),
                     "an unknown PC should remain an unambiguous raw address");
        });

        tc.Run("fixed-address AOT modules activate from guest-memory signatures", [](TestCase &t)
        {
            constexpr uint32_t kMatchAddress = 0x00301000u;
            constexpr uint32_t kOverlapEntry = 0x00302000u;
            constexpr uint32_t kModuleOnlyEntry = 0x00302100u;
            static constexpr uint8_t kMatchBytes[] = {
                0x70u, 0xFFu, 0xBDu, 0x27u,
                0x16u, 0x00u, 0x02u, 0x3Cu,
            };
            static const PS2RecompiledModuleFunction kFunctions[] = {
                {kOverlapEntry, &testOverlayModuleFunction},
                {kModuleOnlyEntry, &testOverlayModuleFunction},
            };
            static const PS2GuestFunctionSymbol kSymbols[] = {
                {kOverlapEntry, kOverlapEntry + 0x20u, "overlay_overlap"},
                {kModuleOnlyEntry, kModuleOnlyEntry + 0x20u, "overlay_only"},
            };
            static const PS2RecompiledModuleDescriptor kModule = {
                "runtime-test-overlay",
                kMatchAddress,
                kMatchBytes,
                sizeof(kMatchBytes),
                kFunctions,
                std::size(kFunctions),
                kSymbols,
                std::size(kSymbols),
            };

            t.IsTrue(
                ps2RegisterRecompiledModule(&kModule),
                "a valid generated module descriptor should register");

            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "runtime memory initialize should succeed");
            t.IsTrue(
                runtime.registerFunction(
                    kOverlapEntry, &testPrimaryModuleFunction),
                "the primary overlapping function should register");

            t.IsFalse(
                runtime.hasFunction(kModuleOnlyEntry),
                "module-only code should remain unavailable before its bytes load");
            R5900Context ctx{};
            ctx.pc = kOverlapEntry;
            runtime.lookupFunction(kOverlapEntry)(
                runtime.memory().getRDRAM(), &ctx, &runtime);
            t.Equals(
                ::getRegU32(&ctx, 2), 0x11111111u,
                "the primary implementation should win while the module is inactive");

            std::memcpy(
                runtime.memory().getRDRAM() + kMatchAddress,
                kMatchBytes,
                sizeof(kMatchBytes));
            t.IsTrue(
                runtime.hasFunction(kModuleOnlyEntry),
                "module-only code should become available after the exact signature loads");
            t.IsTrue(
                runtime.hasFunction(0x80000000u | kModuleOnlyEntry),
                "module dispatch should normalize a direct-mapped EE alias");
            std::memset(&ctx, 0, sizeof(ctx));
            ctx.pc = kOverlapEntry;
            runtime.lookupFunction(kOverlapEntry)(
                runtime.memory().getRDRAM(), &ctx, &runtime);
            t.Equals(
                ::getRegU32(&ctx, 2), 0x22222222u,
                "the active overlay implementation should replace an overlapping primary entry");

            runtime.memory().getRDRAM()[kMatchAddress] ^= 0x01u;
            t.IsFalse(
                runtime.hasFunction(kModuleOnlyEntry),
                "a changed signature should deactivate the stale module immediately");
            std::memset(&ctx, 0, sizeof(ctx));
            ctx.pc = kOverlapEntry;
            runtime.lookupFunction(kOverlapEntry)(
                runtime.memory().getRDRAM(), &ctx, &runtime);
            t.Equals(
                ::getRegU32(&ctx, 2), 0x11111111u,
                "dispatch should fall back to the primary implementation after replacement");
        });

        tc.Run("scalar overflow helpers use defined wrapping arithmetic", [](TestCase &t)
        {
            bool overflow = false;
            uint32_t result32 = 0u;
            uint64_t result64 = 0ull;

            ADD32_OV(0x7FFFFFFFu, 1u, result32, overflow);
            t.Equals(result32, 0x80000000u, "32-bit addition should retain wrapped result bits");
            t.IsTrue(overflow, "positive 32-bit addition overflow should be detected");

            ADD32_OV(0x80000000u, 0xFFFFFFFFu, result32, overflow);
            t.Equals(result32, 0x7FFFFFFFu, "negative 32-bit addition should wrap predictably");
            t.IsTrue(overflow, "negative 32-bit addition overflow should be detected");

            SUB32_OV(0x80000000u, 1u, result32, overflow);
            t.Equals(result32, 0x7FFFFFFFu, "32-bit subtraction should retain wrapped result bits");
            t.IsTrue(overflow, "negative 32-bit subtraction overflow should be detected");

            SUB32_OV(9u, 4u, result32, overflow);
            t.Equals(result32, 5u, "ordinary 32-bit subtraction should retain its result");
            t.IsFalse(overflow, "ordinary 32-bit subtraction should not report overflow");

            ADD64_OV(0x7FFFFFFFFFFFFFFFull, 1ull, result64, overflow);
            t.Equals(result64, 0x8000000000000000ull,
                     "64-bit addition should retain wrapped result bits");
            t.IsTrue(overflow, "positive 64-bit addition overflow should be detected");

            SUB64_OV(0x8000000000000000ull, 1ull, result64, overflow);
            t.Equals(result64, 0x7FFFFFFFFFFFFFFFull,
                     "64-bit subtraction should retain wrapped result bits");
            t.IsTrue(overflow, "negative 64-bit subtraction overflow should be detected");

            t.Equals(ADD64(0xFFFFFFFFFFFFFFFFull, 1ull), 0ull,
                     "non-trapping 64-bit addition should wrap");
            t.Equals(SUB64(0ull, 1ull), 0xFFFFFFFFFFFFFFFFull,
                     "non-trapping 64-bit subtraction should wrap");

            t.Equals(Ps2MaddSigned32(0x7FFFFFFFFFFFFFFFull, 1, 1),
                     0x8000000000000000ull,
                     "signed MADD should wrap across the host signed boundary");
            t.Equals(Ps2MaddSigned32(0ull, -1, 1),
                     0xFFFFFFFFFFFFFFFFull,
                     "signed MADD should preserve a negative product's two's-complement bits");
            t.Equals(Ps2MsubSigned32(0xFFFFFFFFFFFFFFFFull, -1, 1),
                     0ull,
                     "signed MSUB should wrap while subtracting a negative product");
            t.Equals(Ps2MaddSigned32(0ull, INT32_MIN, INT32_MIN),
                     0x4000000000000000ull,
                     "signed MADD should handle the largest 32-bit product exactly");
        });

        tc.Run("VU0 zero registers retain their architectural constants", [](TestCase &t)
        {
            R5900Context ctx{};
            float vf0[4]{};
            _mm_storeu_ps(vf0, ctx.vu0_vf[0]);

            t.Equals(vf0[0], 0.0f, "VF0.x should initialize to zero");
            t.Equals(vf0[1], 0.0f, "VF0.y should initialize to zero");
            t.Equals(vf0[2], 0.0f, "VF0.z should initialize to zero");
            t.Equals(vf0[3], 1.0f, "VF0.w should initialize to one");
            t.Equals(ctx.vi[0], static_cast<uint16_t>(0), "VI0 should initialize to zero");

            ctx.vu0_vf[0] = _mm_set1_ps(7.0f);
            ctx.vi[0] = 0xFFFFu;
            ctx.enforceVu0RegisterInvariants();
            _mm_storeu_ps(vf0, ctx.vu0_vf[0]);

            t.Equals(vf0[0], 0.0f, "VF0.x writes should be discarded");
            t.Equals(vf0[1], 0.0f, "VF0.y writes should be discarded");
            t.Equals(vf0[2], 0.0f, "VF0.z writes should be discarded");
            t.Equals(vf0[3], 1.0f, "VF0.w writes should be discarded");
            t.Equals(ctx.vi[0], static_cast<uint16_t>(0), "VI0 writes should be discarded");
        });

        tc.Run("VU FTOI saturates unrepresentable values", [](TestCase &t)
        {
            uint32_t words[4]{};
            const __m128 ordinaryAndOverflow = _mm_set_ps(
                -2147483904.0f, 2147483648.0f, -123.75f, 123.75f);
            __m128i result = Ps2VuFtoi(ordinaryAndOverflow, 1.0f);
            std::memcpy(words, &result, sizeof(words));

            t.Equals(words[0], 123u, "positive finite VFTOI should truncate toward zero");
            t.Equals(words[1], 0xFFFFFF85u, "negative finite VFTOI should truncate toward zero");
            t.Equals(words[2], 0x7FFFFFFFu, "positive VFTOI overflow should saturate to INT_MAX");
            t.Equals(words[3], 0x80000000u, "negative VFTOI overflow should saturate to INT_MIN");

            const __m128 specialValues = _mm_set_ps(
                std::numeric_limits<float>::quiet_NaN(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                2147483520.0f);
            result = Ps2VuFtoi(specialValues, 1.0f);
            std::memcpy(words, &result, sizeof(words));

            t.Equals(words[0], 0x7FFFFF80u, "largest in-range float should convert exactly");
            t.Equals(words[1], 0x7FFFFFFFu, "positive infinity should saturate to INT_MAX");
            t.Equals(words[2], 0x80000000u, "negative infinity should saturate to INT_MIN");
            t.Equals(words[3], 0x80000000u, "NaN should retain the VU indefinite value");

            result = Ps2VuFtoi(
                _mm_set_ps(0.0f, 0.0f, -134217728.0f, 134217728.0f),
                16.0f);
            std::memcpy(words, &result, sizeof(words));
            t.Equals(words[0], 0x7FFFFFFFu, "scaled positive overflow should saturate");
            t.Equals(words[1], 0x80000000u, "scaled negative boundary should convert to INT_MIN");
        });

        tc.Run("packed HI and LO helpers preserve all lanes", [](TestCase &t)
        {
            R5900Context ctx{};
            ctx.lo = 0x2222222211111111ull;
            ctx.hi = 0x4444444433333333ull;
            ctx.lo1 = 0x6666666655555555ull;
            ctx.hi1 = 0x8888888877777777ull;

            uint64_t doublewords[2]{};
            __m128i value = Ps2GetLo128(&ctx);
            std::memcpy(doublewords, &value, sizeof(value));
            t.Equals(doublewords[0], ctx.lo, "PMFLO low lane should come from LO");
            t.Equals(doublewords[1], ctx.lo1, "PMFLO high lane should come from LO1");

            value = Ps2GetHi128(&ctx);
            std::memcpy(doublewords, &value, sizeof(value));
            t.Equals(doublewords[0], ctx.hi, "PMFHI low lane should come from HI");
            t.Equals(doublewords[1], ctx.hi1, "PMFHI high lane should come from HI1");

            uint32_t words[4]{};
            value = Ps2PmfhlLw(&ctx);
            std::memcpy(words, &value, sizeof(value));
            const uint32_t expectedLw[4] = {
                0x11111111u, 0x33333333u, 0x55555555u, 0x77777777u};
            for (size_t i = 0; i < 4; ++i)
            {
                t.Equals(words[i], expectedLw[i], "PMFHL.LW lane mismatch");
            }

            value = Ps2PmfhlUw(&ctx);
            std::memcpy(words, &value, sizeof(value));
            const uint32_t expectedUw[4] = {
                0x22222222u, 0x44444444u, 0x66666666u, 0x88888888u};
            for (size_t i = 0; i < 4; ++i)
            {
                t.Equals(words[i], expectedUw[i], "PMFHL.UW lane mismatch");
            }

            uint16_t halves[8]{};
            value = Ps2PmfhlLh(&ctx);
            std::memcpy(halves, &value, sizeof(value));
            const uint16_t expectedLh[8] = {
                0x1111u, 0x2222u, 0x3333u, 0x4444u,
                0x5555u, 0x6666u, 0x7777u, 0x8888u};
            for (size_t i = 0; i < 8; ++i)
            {
                t.Equals(halves[i], expectedLh[i], "PMFHL.LH lane mismatch");
            }

            ctx.lo = 0x0000800000007fffull;
            ctx.hi = 0xffff7fffffff8000ull;
            ctx.lo1 = 0xffffffff00000001ull;
            ctx.hi1 = 0x800000007fffffffull;
            value = Ps2PmfhlSh(&ctx);
            std::memcpy(halves, &value, sizeof(value));
            const uint16_t expectedSh[8] = {
                0x7fffu, 0x7fffu, 0x8000u, 0x8000u,
                0x0001u, 0xffffu, 0x7fffu, 0x8000u};
            for (size_t i = 0; i < 8; ++i)
            {
                t.Equals(halves[i], expectedSh[i], "PMFHL.SH saturation mismatch");
            }

            ctx.lo = 0xAAAAAAAA80000000ull;
            ctx.hi = 0xBBBBBBBB00000000ull;
            ctx.lo1 = 0xCCCCCCCC7fffffffull;
            ctx.hi1 = 0xDDDDDDDDffffffffull;
            value = Ps2PmfhlSlw(&ctx);
            std::memcpy(doublewords, &value, sizeof(value));
            t.Equals(doublewords[0], 0x000000007fffffffull,
                     "PMFHL.SLW should clamp positive overflow");
            t.Equals(doublewords[1], 0xffffffff80000000ull,
                     "PMFHL.SLW should clamp negative overflow");

            const __m128i replacement = Ps2MakeU32Vector(1u, 2u, 3u, 4u);
            Ps2PmthlLw(&ctx, replacement);
            t.Equals(ctx.lo, 0xAAAAAAAA00000001ull,
                     "PMTHL.LW should preserve LO's upper word");
            t.Equals(ctx.hi, 0xBBBBBBBB00000002ull,
                     "PMTHL.LW should preserve HI's upper word");
            t.Equals(ctx.lo1, 0xCCCCCCCC00000003ull,
                     "PMTHL.LW should preserve LO1's upper word");
            t.Equals(ctx.hi1, 0xDDDDDDDD00000004ull,
                     "PMTHL.LW should preserve HI1's upper word");

            Ps2SetLo128(&ctx, Ps2MakeU64Vector(0x0123456789ABCDEFull,
                                               0xFEDCBA9876543210ull));
            Ps2SetHi128(&ctx, Ps2MakeU64Vector(0x13579BDF2468ACE0ull,
                                               0x02468ACE13579BDFull));
            t.Equals(ctx.lo, 0x0123456789ABCDEFull, "PMTLO low lane mismatch");
            t.Equals(ctx.lo1, 0xFEDCBA9876543210ull, "PMTLO high lane mismatch");
            t.Equals(ctx.hi, 0x13579BDF2468ACE0ull, "PMTHI low lane mismatch");
            t.Equals(ctx.hi1, 0x02468ACE13579BDFull, "PMTHI high lane mismatch");
        });

        tc.Run("PMULTH writes eight independent signed products", [](TestCase &t)
        {
            R5900Context ctx{};
            const __m128i lhs = Ps2MakeU16Vector(
                0x8000u, static_cast<uint16_t>(-2), 3u, static_cast<uint16_t>(-4),
                5u, static_cast<uint16_t>(-6), 0x7fffu, static_cast<uint16_t>(-8));
            const __m128i rhs = Ps2MakeU16Vector(
                0x8000u, 20u, static_cast<uint16_t>(-30), static_cast<uint16_t>(-40),
                50u, 60u, 0x7fffu, static_cast<uint16_t>(-80));

            const __m128i result = Ps2Pmulth(&ctx, lhs, rhs);
            const uint32_t products[8] = {
                0x40000000u,
                static_cast<uint32_t>(-40),
                static_cast<uint32_t>(-90),
                160u,
                250u,
                static_cast<uint32_t>(-360),
                0x3fff0001u,
                640u,
            };

            t.Equals(ctx.lo, (static_cast<uint64_t>(products[1]) << 32) | products[0],
                     "PMULTH should store products 0 and 1 in LO");
            t.Equals(ctx.hi, (static_cast<uint64_t>(products[3]) << 32) | products[2],
                     "PMULTH should store products 2 and 3 in HI");
            t.Equals(ctx.lo1, (static_cast<uint64_t>(products[5]) << 32) | products[4],
                     "PMULTH should store products 4 and 5 in LO1");
            t.Equals(ctx.hi1, (static_cast<uint64_t>(products[7]) << 32) | products[6],
                     "PMULTH should store products 6 and 7 in HI1");

            uint32_t resultWords[4]{};
            std::memcpy(resultWords, &result, sizeof(result));
            const uint32_t expectedResult[4] = {
                products[0], products[2], products[4], products[6]};
            for (size_t i = 0; i < 4; ++i)
            {
                t.Equals(resultWords[i], expectedResult[i],
                         "PMULTH rd should contain the even products");
            }
        });

        tc.Run("PMULTUW writes two independent unsigned products", [](TestCase &t)
        {
            R5900Context ctx{};
            const __m128i lhs =
                Ps2MakeU32Vector(0xffffffffu, 0x11111111u, 0x80000000u, 0x22222222u);
            const __m128i rhs =
                Ps2MakeU32Vector(2u, 0x33333333u, 0xffffffffu, 0x44444444u);

            const __m128i result = Ps2Pmultuw(&ctx, lhs, rhs);
            constexpr uint64_t product0 = 0x00000001fffffffeull;
            constexpr uint64_t product1 = 0x7fffffff80000000ull;

            t.Equals(ctx.lo, 0xfffffffffffffffeull,
                     "PMULTUW should sign-extend product 0's lower word into LO");
            t.Equals(ctx.hi, 0x0000000000000001ull,
                     "PMULTUW should sign-extend product 0's upper word into HI");
            t.Equals(ctx.lo1, 0xffffffff80000000ull,
                     "PMULTUW should sign-extend product 1's lower word into LO1");
            t.Equals(ctx.hi1, 0x000000007fffffffull,
                     "PMULTUW should sign-extend product 1's upper word into HI1");

            uint64_t resultLanes[2]{};
            std::memcpy(resultLanes, &result, sizeof(result));
            t.Equals(resultLanes[0], product0,
                     "PMULTUW rd low lane should contain source word 0's full product");
            t.Equals(resultLanes[1], product1,
                     "PMULTUW rd high lane should contain source word 2's full product");
        });

        tc.Run("PMULTW writes two independent signed products", [](TestCase &t)
        {
            R5900Context ctx{};
            const __m128i lhs =
                Ps2MakeU32Vector(0x80000000u, 0x11111111u, 0xffffffffu, 0x22222222u);
            const __m128i rhs =
                Ps2MakeU32Vector(2u, 0x33333333u, 0x80000000u, 0x44444444u);

            const __m128i result = Ps2Pmultw(&ctx, lhs, rhs);
            constexpr uint64_t product0 = 0xffffffff00000000ull;
            constexpr uint64_t product1 = 0x0000000080000000ull;

            t.Equals(ctx.lo, 0ull,
                     "PMULTW should sign-extend product 0's lower word into LO");
            t.Equals(ctx.hi, 0xffffffffffffffffull,
                     "PMULTW should sign-extend product 0's upper word into HI");
            t.Equals(ctx.lo1, 0xffffffff80000000ull,
                     "PMULTW should sign-extend product 1's lower word into LO1");
            t.Equals(ctx.hi1, 0ull,
                     "PMULTW should sign-extend product 1's upper word into HI1");

            uint64_t resultLanes[2]{};
            std::memcpy(resultLanes, &result, sizeof(result));
            t.Equals(resultLanes[0], product0,
                     "PMULTW rd low lane should contain source word 0's full product");
            t.Equals(resultLanes[1], product1,
                     "PMULTW rd high lane should contain source word 2's full product");
        });

        tc.Run("differential decoder/codegen gpr-write contract for MULT and DIV families", [](TestCase &t)
        {
            R5900Decoder decoder;
            CodeGenerator generator({}, {});

            const struct
            {
                const char *name;
                uint32_t raw;
            } cases[] = {
                {"MULT rd!=0", (OPCODE_SPECIAL << 26) | (4u << 21) | (5u << 16) | (3u << 11) | SPECIAL_MULT},
                {"MULT rd==0", (OPCODE_SPECIAL << 26) | (4u << 21) | (5u << 16) | (0u << 11) | SPECIAL_MULT},
                {"DIV rd!=0", (OPCODE_SPECIAL << 26) | (6u << 21) | (7u << 16) | (9u << 11) | SPECIAL_DIV},
                {"MMI MULT1 rd!=0", (OPCODE_MMI << 26) | (8u << 21) | (9u << 16) | (10u << 11) | MMI_MULT1},
                {"MMI DIV1 rd!=0", (OPCODE_MMI << 26) | (8u << 21) | (9u << 16) | (10u << 11) | MMI_DIV1},
            };

            for (size_t i = 0; i < std::size(cases); ++i)
            {
                const Instruction inst = decoder.decodeInstruction(0x1000u + static_cast<uint32_t>(i * 4u), cases[i].raw);
                const std::string generated = generator.translateInstruction(inst);
                const bool emittedRdWrite = hasSignedRdWrite(generated, inst.rd);

                t.Equals(emittedRdWrite, inst.modificationInfo.modifiesGPR,
                         std::string("decoder/codegen mismatch for ") + cases[i].name);
                t.IsTrue(inst.modificationInfo.modifiesControl,
                         std::string("HI/LO control side-effect missing for ") + cases[i].name);
            }
        });

        tc.Run("guest execution is serialized per runtime", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            gSerializedGuestActive.store(0, std::memory_order_release);
            gSerializedGuestMaxActive.store(0, std::memory_order_release);

            constexpr uint32_t kEntries[] = {
                0x120000u,
                0x130000u,
                0x140000u,
                0x150000u,
            };
            constexpr size_t kEntryCount = sizeof(kEntries) / sizeof(kEntries[0]);
            R5900Context contexts[kEntryCount]{};
            std::vector<std::thread> workers;
            workers.reserve(kEntryCount);

            for (size_t i = 0; i < kEntryCount; ++i)
            {
                runtime.registerFunction(kEntries[i], &testSerializedGuestStep);
                contexts[i].pc = kEntries[i];
            }

            for (size_t i = 0; i < kEntryCount; ++i)
            {
                workers.emplace_back([&, i]()
                {
                    runtime.dispatchLoop(rdram.data(), &contexts[i]);
                });
            }

            for (std::thread &worker : workers)
            {
                if (worker.joinable())
                {
                    worker.join();
                }
            }

            t.Equals(gSerializedGuestActive.load(std::memory_order_acquire), 0,
                     "serialized guest dispatch should leave no active workers");
            t.Equals(gSerializedGuestMaxActive.load(std::memory_order_acquire), 1,
                     "dispatchLoop should not execute guest code concurrently on one runtime");
        });

        tc.Run("EE thread diagnostics are disabled by default and classify rotations when enabled", [](TestCase &t)
        {
            {
                PS2RuntimeConfiguration configuration{};
                configuration.useEeThreadDiagnosticsEnvironment =
                    false;
                PS2Runtime runtime(configuration);
                R5900Context context{};
                runtime.recordEeThreadQueueRotation(
                    2, 1, 1, true);
                {
                    PS2Runtime::GuestExecutionScope scope(
                        &runtime, &context);
                }
                const auto disabled =
                    runtime.debugEeThreadDiagnosticsSnapshot();
                t.IsFalse(
                    disabled.enabled,
                    "scheduler diagnostics must be opt-in");
                t.Equals(
                    disabled.rotationRequests,
                    uint64_t{0u},
                    "disabled scheduler diagnostics must not update rotation counters");
                t.Equals(
                    disabled.guestLockAcquisitions,
                    uint64_t{0u},
                    "disabled scheduler diagnostics must not update lock counters");
            }

            PS2RuntimeConfiguration configuration{};
            configuration.eeThreadDiagnostics = true;
            configuration.useEeThreadDiagnosticsEnvironment = false;
            PS2Runtime runtime(configuration);
            R5900Context first{};
            R5900Context second{};
            {
                PS2Runtime::GuestExecutionScope scope(
                    &runtime, &first);
            }
            {
                PS2Runtime::GuestExecutionScope scope(
                    &runtime, &second);
            }
            runtime.recordEeThreadQueueRotation(
                2, 0, 1, true);
            runtime.recordEeThreadQueueRotation(
                2, 1, 1, true);
            runtime.recordEeThreadQueueRotation(
                3, 128, 128, false);

            const auto diagnostics =
                runtime.debugEeThreadDiagnosticsSnapshot();
            t.IsTrue(
                diagnostics.enabled,
                "the explicit runtime configuration should enable scheduler diagnostics");
            t.Equals(
                diagnostics.guestLockRequests,
                uint64_t{2u},
                "two outer scopes should request the guest lock twice");
            t.Equals(
                diagnostics.guestLockAcquisitions,
                uint64_t{2u},
                "two uncontended outer scopes should acquire the guest lock twice");
            t.Equals(
                diagnostics.guestLockContentions,
                uint64_t{0u},
                "sequential scopes should not report lock contention");
            t.Equals(
                diagnostics.outerGuestExecutionAcquisitions,
                uint64_t{2u},
                "both outer scopes should be classified");
            t.Equals(
                diagnostics.guestContextChanges,
                uint64_t{1u},
                "switching from the first context to the second should count once");
            t.Equals(
                diagnostics.handoffNotifications,
                uint64_t{2u},
                "each outer acquisition should publish one handoff epoch");
            t.Equals(
                diagnostics.rotationRequests,
                uint64_t{3u},
                "accepted and rejected rotations should all be counted");
            t.Equals(
                diagnostics.acceptedRotationRequests,
                uint64_t{2u},
                "accepted rotations should be counted separately");
            t.Equals(
                diagnostics.rejectedRotationRequests,
                uint64_t{1u},
                "invalid-priority rotations should be retained");
            t.Equals(
                diagnostics.priorityZeroRotationRequests,
                uint64_t{1u},
                "priority-zero requests should remain visible after resolution");
            t.Equals(
                diagnostics.acceptedRotationsByPriority[1u],
                uint64_t{2u},
                "resolved priority buckets should include both accepted requests");
            t.Equals(
                diagnostics.acceptedRotationsByThread[2u],
                uint64_t{2u},
                "thread buckets should include both accepted requests");

            runtime.recordMpegPictureServed(false);
            runtime.recordMpegPictureServed(true);
            const auto progress = runtime.debugRuntimeProgress();
            t.Equals(
                progress.mpegPicturesServed,
                uint64_t{2u},
                "all successfully served MPEG pictures should be counted");
            t.Equals(
                progress.mpegUniquePicturesServed,
                uint64_t{1u},
                "decoded MPEG pictures should be distinguished from repeats");
            t.Equals(
                progress.mpegRepeatedPicturesServed,
                uint64_t{1u},
                "repeated MPEG pictures should be reported separately");
        });

        tc.Run("scheduler boundary skips an empty handoff and serves a queued guest", [](TestCase &t)
        {
            PS2RuntimeConfiguration configuration{};
            configuration.eeThreadDiagnostics = true;
            configuration.useEeThreadDiagnosticsEnvironment = false;
            PS2Runtime runtime(configuration);
            std::atomic<bool> peerRan{false};
            std::thread peer;
            bool peerWaiting = false;

            {
                PS2Runtime::GuestExecutionScope mainScope(&runtime);
                runtime.yieldGuestExecutionAtBoundary();
                const auto emptyBoundary =
                    runtime.debugEeThreadDiagnosticsSnapshot();
                t.Equals(
                    emptyBoundary.requestedGuestSwitches,
                    uint64_t{0u},
                    "an empty scheduler boundary should not request an impossible switch");
                t.Equals(
                    emptyBoundary.guestSwitchTimeouts,
                    uint64_t{0u},
                    "an empty scheduler boundary should not manufacture a timeout");

                peer = std::thread([&]()
                {
                    PS2Runtime::GuestExecutionScope peerScope(
                        &runtime);
                    peerRan.store(true, std::memory_order_release);
                });
                peerWaiting = waitUntil(
                    [&]()
                    {
                        return runtime
                                   .guestExecutionWaiterCountForTesting() >
                               0u;
                    },
                    std::chrono::milliseconds(100));

                runtime.yieldGuestExecutionAtBoundary();
            }

            if (peer.joinable())
            {
                peer.join();
            }

            t.IsTrue(
                peerWaiting,
                "a queued peer should be visible at the scheduler boundary");
            t.IsTrue(
                peerRan.load(std::memory_order_acquire),
                "a queued peer should acquire guest execution before the boundary returns");
            const auto diagnostics =
                runtime.debugEeThreadDiagnosticsSnapshot();
            t.Equals(
                diagnostics.requestedGuestSwitches,
                uint64_t{1u},
                "only the boundary with a queued peer should request a switch");
            t.Equals(
                diagnostics.completedGuestSwitches,
                uint64_t{1u},
                "the queued boundary handoff should complete");
            t.Equals(
                diagnostics.guestSwitchTimeouts,
                uint64_t{0u},
                "the scheduler boundary should not time out");
        });

        tc.Run("wake handoff lets a contending guest thread acquire before returning", [](TestCase &t)
        {
            PS2RuntimeConfiguration configuration{};
            configuration.eeThreadDiagnostics = true;
            configuration.useEeThreadDiagnosticsEnvironment = false;
            PS2Runtime runtime(configuration);
            std::atomic<bool> peerRan{false};
            std::thread peer;
            bool peerWaiting = false;
            bool peerRanWhileMainHeld = false;
            bool peerRanAfterHandoff = false;

            {
                PS2Runtime::GuestExecutionScope mainScope(&runtime);
                peer = std::thread([&]()
                {
                    PS2Runtime::GuestExecutionScope peerScope(&runtime);
                    peerRan.store(true, std::memory_order_release);
                });

                peerWaiting = waitUntil([&]()
                {
                    return runtime.guestExecutionWaiterCountForTesting() > 0u;
                }, std::chrono::milliseconds(100));

                peerRanWhileMainHeld = peerRan.load(std::memory_order_acquire);
                runtime.yieldGuestExecutionAfterWake();
                peerRanAfterHandoff = peerRan.load(std::memory_order_acquire);
            }

            if (peer.joinable())
            {
                peer.join();
            }

            t.IsTrue(peerWaiting, "peer guest thread should contend while the waker owns guest execution");
            t.IsFalse(peerRanWhileMainHeld, "peer guest thread should not run before the waker yields execution");
            t.IsTrue(peerRanAfterHandoff, "wake handoff should let the peer acquire guest execution before returning");
            const auto diagnostics =
                runtime.debugEeThreadDiagnosticsSnapshot();
            t.Equals(
                diagnostics.requestedGuestSwitches,
                uint64_t{1u},
                "one in-scope yield should request one guest switch");
            t.Equals(
                diagnostics.completedGuestSwitches,
                uint64_t{1u},
                "the contending peer should complete the requested guest switch");
            t.Equals(
                diagnostics.guestSwitchCvWaits,
                uint64_t{1u},
                "the requested switch should perform one bounded CV wait");
            t.Equals(
                diagnostics.guestSwitchTimeouts,
                uint64_t{0u},
                "the completed handoff should not be classified as a timeout");
            t.IsTrue(
                diagnostics.guestLockContentions >= 1u,
                "the waiting peer should be observed as a contended guest lock");
        });

        tc.Run("recursive guest execution acquisition does not advance the handoff epoch", [](TestCase &t)
        {
            PS2Runtime runtime;
            const uint64_t initial = runtime.guestExecutionHandoffEpochSnapshot();

            PS2Runtime::GuestExecutionScope outer(&runtime);
            const uint64_t afterOuter = runtime.guestExecutionHandoffEpochSnapshot();
            t.Equals(afterOuter, initial + 1u, "outer acquisition should advance the epoch exactly once");

            {
                PS2Runtime::GuestExecutionScope inner(&runtime);
                t.Equals(runtime.guestExecutionHandoffEpochSnapshot(), afterOuter,
                         "recursive acquisition must not advance the epoch");

                PS2Runtime::GuestExecutionScope innermost(&runtime);
                t.Equals(runtime.guestExecutionHandoffEpochSnapshot(), afterOuter,
                         "deeper recursive acquisitions must not advance the epoch either");
            }

            t.Equals(runtime.guestExecutionHandoffEpochSnapshot(), afterOuter,
                     "releasing recursive acquisitions must not advance the epoch");
        });

        tc.Run("reacquiring a depth-4 release advances the handoff epoch exactly once", [](TestCase &t)
        {
            PS2Runtime runtime;

            PS2Runtime::GuestExecutionScope s1(&runtime);
            PS2Runtime::GuestExecutionScope s2(&runtime);
            PS2Runtime::GuestExecutionScope s3(&runtime);
            PS2Runtime::GuestExecutionScope s4(&runtime);

            const uint64_t before = runtime.guestExecutionHandoffEpochSnapshot();
            {
                PS2Runtime::GuestExecutionReleaseScope release(&runtime);
                t.Equals(runtime.guestExecutionHandoffEpochSnapshot(), before,
                         "releasing guest execution must not advance the epoch");
            }

            t.Equals(runtime.guestExecutionHandoffEpochSnapshot(), before + 1u,
                     "reacquiring a depth-4 release should advance the epoch exactly once");
        });

        tc.Run("handoff completed before the wait does not count as a timeout", [](TestCase &t)
        {
            PS2Runtime runtime;
            const uint64_t timeoutsBefore = runtime.guestExecutionHandoffTimeouts();

            std::atomic<bool> holderAcquired{false};
            std::atomic<bool> releaseHolder{false};
            uint64_t baseline = 0u;
            std::thread holder;

            {
                PS2Runtime::GuestExecutionScope mainScope(&runtime);

                holder = std::thread([&]()
                {
                    PS2Runtime::GuestExecutionScope scope(&runtime);
                    holderAcquired.store(true, std::memory_order_release);
                    while (!releaseHolder.load(std::memory_order_acquire))
                    {
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                    }
                });

                const bool holderContending = waitUntil([&]()
                {
                    return runtime.guestExecutionWaiterCountForTesting() > 0u;
                }, std::chrono::milliseconds(250));
                t.IsTrue(holderContending, "holder thread should be queued before the release");

                // Token captured BEFORE releasing guest execution, like the dispatchers do
                baseline = runtime.guestExecutionHandoffEpochSnapshot();
            }

            const bool acquired = waitUntil([&]()
            {
                return holderAcquired.load(std::memory_order_acquire);
            }, std::chrono::milliseconds(250));
            t.IsTrue(acquired, "holder should acquire guest execution after the release");

            // Second waiter keeps waiters > 0 so the wait below cannot take the
            // no-waiters fast path: it must recognize the epoch advance instead.
            std::thread secondWaiter([&]()
            {
                PS2Runtime::GuestExecutionScope scope(&runtime);
            });
            const bool secondContending = waitUntil([&]()
            {
                return runtime.guestExecutionWaiterCountForTesting() > 0u;
            }, std::chrono::milliseconds(250));
            t.IsTrue(secondContending, "second waiter should be queued while the holder owns guest execution");

            runtime.waitForGuestExecutionHandoff(baseline);

            t.Equals(runtime.guestExecutionHandoffTimeouts(), timeoutsBefore,
                     "a handoff that completed before the wait must not count as a timeout");

            releaseHolder.store(true, std::memory_order_release);
            if (holder.joinable())
            {
                holder.join();
            }
            if (secondWaiter.joinable())
            {
                secondWaiter.join();
            }
        });

        tc.Run("nested DeferredGuestYieldScope delivers pending only to the outermost scope", [](TestCase &t)
        {
            PS2Runtime runtime;
            bool outerPending = false;
            bool innerPending = false;

            {
                PS2Runtime::DeferredGuestYieldScope outer(outerPending);
                {
                    PS2Runtime::DeferredGuestYieldScope inner(innerPending);
                    runtime.yieldGuestExecutionAfterWake(); // must defer instead of yielding
                }
                t.IsFalse(innerPending, "inner scope must not consume the deferred yield");
                t.IsFalse(outerPending, "pending must only be delivered when the outermost scope closes");
            }

            t.IsTrue(outerPending, "outermost scope should deliver the deferred yield");
            t.IsFalse(innerPending, "inner scope must stay untouched");
        });

        tc.Run(
            "guest checkpoint materializes a deferred callback exit",
            [](TestCase &t)
            {
                PS2Runtime runtime;
                R5900Context ctx{};
                ctx.pc = 0x190000u;
                bool pending = false;
                PS2GuestCheckpointResult result =
                    PS2GuestCheckpointResult::Continue;

                runtime.requestGuestPreemption();
                {
                    PS2Runtime::DeferredGuestYieldScope
                        defer(pending);
                    result =
                        runtime
                            .checkpointGuestExecution(
                                &ctx);
                    t.IsFalse(
                        pending,
                        "the defer scope should retain its pending exit until destruction");
                }

                t.IsTrue(
                    result ==
                            PS2GuestCheckpointResult::
                                ExitToDispatcher &&
                        pending,
                    "a checkpoint inside a callback-safe region should return generated code and defer scheduling");
            });

        tc.Run("guest preemption policy requests a dispatcher handoff when another guest thread contends", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            constexpr uint32_t kFirstEntry = 0x190000u;
            constexpr uint32_t kSecondEntry = 0x1A0000u;

            gPreemptionPolicyEntryCount.store(0, std::memory_order_release);
            gPreemptionPolicyAllowFirstProbe.store(false, std::memory_order_release);
            gPreemptionPolicyPeerRan.store(false, std::memory_order_release);

            runtime.registerFunction(kFirstEntry, &testPreemptionPolicyStep);
            runtime.registerFunction(kSecondEntry, &testPreemptionPolicyStep);

            R5900Context firstCtx{};
            R5900Context secondCtx{};
            firstCtx.pc = kFirstEntry;
            secondCtx.pc = kSecondEntry;

            std::thread firstWorker([&]()
            {
                runtime.dispatchLoop(rdram.data(), &firstCtx);
            });

            const bool firstEntered = waitUntil([&]()
            {
                return gPreemptionPolicyEntryCount.load(std::memory_order_acquire) >= 1;
            }, std::chrono::milliseconds(100));

            std::thread secondWorker([&]()
            {
                runtime.dispatchLoop(rdram.data(), &secondCtx);
            });

            const bool secondContending = waitUntil([&]()
            {
                return runtime.guestExecutionWaiterCountForTesting() > 0u;
            }, std::chrono::milliseconds(100));

            gPreemptionPolicyAllowFirstProbe.store(true, std::memory_order_release);

            if (firstWorker.joinable())
            {
                firstWorker.join();
            }
            if (secondWorker.joinable())
            {
                secondWorker.join();
            }

            t.IsTrue(firstEntered, "first guest worker should enter before probing for preemption");
            t.IsTrue(secondContending, "second guest worker should contend for guest execution before the first returns");
            t.IsTrue(gPreemptionPolicyPeerRan.load(std::memory_order_acquire),
                     "second guest worker should run after the first returns to the dispatcher");
            t.Equals(getRegU32(&firstCtx, 2), 1u,
                     "first guest worker should observe that the runtime requested preemption under contention");
        });

        tc.Run("lookupFunction rejects internal resume PCs without exact registration", [](TestCase &t)
        {
            PS2Runtime runtime;
            runtime.setMissingFunctionPolicy(PS2Runtime::MissingFunctionPolicy::Stop);
            runtime.registerFunction(0x1000u, &testResumeOwnerFallbackHandler);
            runtime.registerFunction(0x1100u, &testResumeNextFunctionHandler);

            R5900Context ctx{};
            ctx.pc = 0x1010u;
            auto fn = runtime.lookupFunction(ctx.pc);
            fn(nullptr, &ctx, &runtime);

            t.Equals(::getRegU32(&ctx, 2), 0u,
                     "unregistered resume PC should not alias to the nearest owner");
            t.IsTrue(runtime.isStopRequested(),
                     "missing exact dispatch target should request runtime stop");
        });

        tc.Run("lookupFunction rejects final-function PCs inside code regions without exact registration", [](TestCase &t)
        {
            PS2Runtime runtime;
            runtime.setMissingFunctionPolicy(PS2Runtime::MissingFunctionPolicy::Stop);
            runtime.memory().registerCodeRegion(0x2000u, 0x2100u);
            runtime.registerFunction(0x2000u, &testResumeOwnerFallbackHandler);

            R5900Context ctx{};
            ctx.pc = 0x2010u;
            auto fn = runtime.lookupFunction(ctx.pc);
            fn(nullptr, &ctx, &runtime);

            t.Equals(::getRegU32(&ctx, 2), 0u,
                     "code-region membership alone should not alias to the previous function");
            t.IsTrue(runtime.isStopRequested(),
                     "missing exact final-function target should request runtime stop");
        });

        tc.Run("function dispatch resolves direct-mapped EE code aliases", [](TestCase &t)
        {
            PS2Runtime runtime;
            runtime.registerFunction(0x3000u, &testGuestBranchImplicitReturnHandler);

            t.IsTrue(runtime.hasFunction(0x80003000u),
                     "KSEG0 aliases should resolve registered physical functions");
            t.IsTrue(runtime.hasFunction(0xA0003000u),
                     "KSEG1 aliases should resolve registered physical functions");
            t.IsTrue(runtime.hasFunction(0x20003000u),
                     "the EE uncached RAM mirror should resolve registered physical functions");

            R5900Context lookupCtx{};
            lookupCtx.pc = 0xA0003000u;
            auto fn = runtime.lookupFunction(lookupCtx.pc);
            fn(nullptr, &lookupCtx, &runtime);
            t.Equals(lookupCtx.pc, 0x3000u,
                     "alias lookup should present the physical PC to generated code");

            R5900Context branchCtx{};
            branchCtx.pc = 0x2000u;
            const bool returnedToFallthrough = runtime.dispatchGuestBranch(
                nullptr,
                &branchCtx,
                0x80003000u,
                0x2000u,
                0x2008u,
                PS2Runtime::GuestBranchKind::IndirectCall,
                "KSEG0 test call");
            t.IsTrue(returnedToFallthrough,
                     "an aliased call target should retain ordinary call-return semantics");
            t.Equals(branchCtx.pc, 0x2008u,
                     "an unchanged aliased callee PC should normalize to the call fallthrough");
        });

        tc.Run("dispatchGuestBranch call normalizes unchanged callee PC to fallthrough", [](TestCase &t)
        {
            PS2Runtime runtime;
            runtime.registerFunction(0x3000u, &testGuestBranchImplicitReturnHandler);

            R5900Context ctx{};
            ctx.pc = 0x2000u;

            const bool returnedToFallthrough = runtime.dispatchGuestBranch(
                nullptr,
                &ctx,
                0x3000u,
                0x2000u,
                0x2008u,
                PS2Runtime::GuestBranchKind::IndirectCall,
                "test-jalr");

            t.IsTrue(returnedToFallthrough,
                     "call-like dispatch should report true when it resumes at fallthrough");
            t.Equals(ctx.pc, 0x2008u,
                     "unchanged callee PC should be converted to call fallthrough");
            t.Equals(::getRegU32(&ctx, 2), 0x00FACE42u,
                     "callee should still execute normally");
        });

        tc.Run("dispatchGuestBranch call returns false when callee transfers elsewhere", [](TestCase &t)
        {
            PS2Runtime runtime;
            runtime.registerFunction(0x3100u, &testGuestBranchTransferHandler);

            R5900Context ctx{};
            ctx.pc = 0x2000u;

            const bool returnedToFallthrough = runtime.dispatchGuestBranch(
                nullptr,
                &ctx,
                0x3100u,
                0x2000u,
                0x2008u,
                PS2Runtime::GuestBranchKind::IndirectCall,
                "test-jalr-transfer");

            t.IsFalse(returnedToFallthrough,
                      "call-like dispatch should stop caller flow when callee transfers elsewhere");
            t.Equals(ctx.pc, 0x33330000u,
                     "callee transfer PC should be preserved");
        });

        tc.Run("dispatchGuestBranch preserves a preempted callee resume PC", [](TestCase &t)
        {
            PS2Runtime runtime;
            runtime.registerFunction(0x3150u, &testGuestBranchPreemptionHandler);

            R5900Context ctx{};
            ctx.pc = 0x2000u;

            runtime.requestGuestPreemption();
            const bool returnedToFallthrough = runtime.dispatchGuestBranch(
                nullptr,
                &ctx,
                0x3150u,
                0x2000u,
                0x2008u,
                PS2Runtime::GuestBranchKind::DirectCall,
                "test-preempted-call");

            t.IsFalse(returnedToFallthrough,
                      "a nested preemption should unwind the generated caller");
            t.Equals(ctx.pc, 0x3150u,
                     "the preempted callee entry must remain the dispatcher resume PC");
            t.IsTrue(::getRegU32(&ctx, 2) != 0u,
                     "the callee should have observed a cooperative preemption request");
        });

        tc.Run("dispatchGuestBranch rejects missing exact targets", [](TestCase &t)
        {
            PS2Runtime runtime;
            runtime.setMissingFunctionPolicy(PS2Runtime::MissingFunctionPolicy::Stop);
            runtime.registerFunction(0x3200u, &testGuestBranchImplicitReturnHandler);

            R5900Context ctx{};
            ctx.pc = 0x2000u;

            const bool returnedToFallthrough = runtime.dispatchGuestBranch(
                nullptr,
                &ctx,
                0x3210u,
                0x2000u,
                0x2008u,
                PS2Runtime::GuestBranchKind::IndirectCall,
                "test-missing");

            t.IsFalse(returnedToFallthrough,
                      "missing target should not resume caller flow");
            t.IsTrue(runtime.isStopRequested(),
                     "missing exact target should request runtime stop");
            t.Equals(ctx.pc, 0x3210u,
                     "missing target should remain visible in ctx->pc for diagnostics");
        });


        tc.Run("GS async callbacks keep a dedicated stack when guest heap is exhausted", [](TestCase &t)
        {
            notifyRuntimeStop();

            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            constexpr uint32_t kCallbackEntry = 0x180000u;
            constexpr uint32_t kCallerGp = 0x0036A7F0u;
            constexpr uint32_t kCallerSp = 0x00123450u;
            constexpr uint32_t kAsyncStackFloor = 0x01F00000u;

            runtime.configureGuestHeap(kAsyncStackFloor, kAsyncStackFloor);
            runtime.registerFunction(kCallbackEntry, &testRecordAsyncCallbackStack);
            ps2_stubs::resetGsSyncVCallbackState(
                &runtime);
            gAsyncCallbackObservedSp.store(0u, std::memory_order_release);
            gAsyncCallbackObservedGp.store(0u, std::memory_order_release);

            R5900Context registerCtx{};
            registerCtx.pc = 0x00101900u;
            setRegU32(registerCtx, 4, kCallbackEntry);
            setRegU32(registerCtx, 28, kCallerGp);
            setRegU32(registerCtx, 29, kCallerSp);
            ps2_stubs::sceGsSyncVCallback(rdram.data(), &registerCtx, &runtime);

            ps2_stubs::dispatchGsSyncVCallback(rdram.data(), &runtime, 1u);

            const uint32_t observedSp = gAsyncCallbackObservedSp.load(std::memory_order_acquire);
            const uint32_t observedGp = gAsyncCallbackObservedGp.load(std::memory_order_acquire);

            t.IsTrue(observedSp != 0u, "callback should execute");
            t.Equals(observedGp, kCallerGp, "callback should preserve the registered GP");
            t.IsTrue(observedSp != kCallerSp, "callback should not reuse the registering thread stack");
            t.IsTrue(observedSp >= kAsyncStackFloor,
                     "callback should switch to the reserved async stack pool");

            runtime.requestStop();
            notifyRuntimeStop();
        });

        tc.Run("MPEG init and callback stubs return success instead of TODO errors", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            ps2_stubs::resetMpegStubState(&runtime);

            R5900Context initCtx{};
            ps2_stubs::sceMpegInit(rdram.data(), &initCtx, &runtime);
            t.Equals(getRegS32(initCtx, 2), 0,
                     "sceMpegInit should succeed so games can continue through movie setup");

            R5900Context addCtx0{};
            setRegU32(addCtx0, 4, 0x00123000u);
            setRegU32(addCtx0, 5, 1u);
            setRegU32(addCtx0, 6, 0x00124000u);
            setRegU32(addCtx0, 7, 0u);
            ps2_stubs::sceMpegAddCallback(rdram.data(), &addCtx0, &runtime);
            t.Equals(getRegS32(addCtx0, 2), 1,
                     "first sceMpegAddCallback should hand back a non-error callback handle");

            R5900Context addCtx1{};
            setRegU32(addCtx1, 4, 0x00123000u);
            setRegU32(addCtx1, 5, 2u);
            setRegU32(addCtx1, 6, 0x00124010u);
            setRegU32(addCtx1, 7, 0u);
            ps2_stubs::sceMpegAddCallback(rdram.data(), &addCtx1, &runtime);
            t.Equals(getRegS32(addCtx1, 2), 2,
                     "subsequent sceMpegAddCallback calls should keep succeeding");

            R5900Context reinitCtx{};
            ps2_stubs::sceMpegInit(rdram.data(), &reinitCtx, &runtime);

            R5900Context addAfterReinit{};
            setRegU32(addAfterReinit, 4, 0x00123000u);
            setRegU32(addAfterReinit, 5, 3u);
            setRegU32(addAfterReinit, 6, 0x00124020u);
            setRegU32(addAfterReinit, 7, 0u);
            ps2_stubs::sceMpegAddCallback(rdram.data(), &addAfterReinit, &runtime);
            t.Equals(getRegS32(addAfterReinit, 2), 1,
                     "sceMpegInit should reset MPEG callback bookkeeping between runs");
        });

        tc.Run("MPEG state and callback stacks are isolated per runtime", [](TestCase &t)
        {
            PS2Runtime first;
            PS2Runtime second;
            std::vector<uint8_t> firstRdram(
                PS2_RAM_SIZE, 0u);
            std::vector<uint8_t> secondRdram(
                PS2_RAM_SIZE, 0u);
            ps2_stubs::resetMpegStubState(&first);
            ps2_stubs::resetMpegStubState(&second);

            constexpr uint32_t kMpegAddr = 0x00123000u;
            constexpr uint32_t kCallbackEntry = 0x00124000u;
            constexpr uint32_t kPacketAddr = 0x00128000u;

            auto addCallback =
                [&](PS2Runtime &runtime,
                    std::vector<uint8_t> &rdram,
                    uint32_t callbackType)
            {
                R5900Context context{};
                setRegU32(context, 4, kMpegAddr);
                setRegU32(context, 5, callbackType);
                setRegU32(context, 6, kCallbackEntry);
                ps2_stubs::sceMpegAddCallback(
                    rdram.data(), &context, &runtime);
                return ::getRegU32(&context, 2);
            };

            t.Equals(
                addCallback(first, firstRdram, 1u),
                1u,
                "the first runtime should allocate MPEG callback handle 1");
            t.Equals(
                addCallback(second, secondRdram, 2u),
                1u,
                "the second runtime should independently allocate MPEG callback handle 1");

            auto setDecodeMode =
                [&](PS2Runtime &runtime,
                    std::vector<uint8_t> &rdram,
                    uint32_t mode)
            {
                R5900Context context{};
                setRegU32(context, 4, kMpegAddr);
                setRegU32(context, 5, mode);
                ps2_stubs::sceMpegSetDecodeMode(
                    rdram.data(), &context, &runtime);
            };
            auto getDecodeMode =
                [&](PS2Runtime &runtime,
                    std::vector<uint8_t> &rdram)
            {
                R5900Context context{};
                setRegU32(context, 4, kMpegAddr);
                ps2_stubs::sceMpegGetDecodeMode(
                    rdram.data(), &context, &runtime);
                return ::getRegU32(&context, 2);
            };

            setDecodeMode(first, firstRdram, 7u);
            setDecodeMode(second, secondRdram, 9u);
            t.Equals(
                getDecodeMode(first, firstRdram),
                7u,
                "the second runtime must not replace the first runtime's MPEG playback state");

            R5900Context deleteContext{};
            setRegU32(deleteContext, 4, kMpegAddr);
            ps2_stubs::sceMpegDelete(
                secondRdram.data(),
                &deleteContext,
                &second);
            t.Equals(
                getDecodeMode(first, firstRdram),
                7u,
                "deleting the second runtime's MPEG object must not erase the first runtime's object");

            ps2_stubs::resetMpegStubState(&first);
            first.registerFunction(
                kCallbackEntry,
                &testRecordMpegStreamCallback);

            R5900Context addStreamContext{};
            setRegU32(addStreamContext, 4, kMpegAddr);
            setRegU32(addStreamContext, 5, 3u);
            setRegU32(addStreamContext, 7, kCallbackEntry);
            setRegU32(addStreamContext, 8, 0x55667788u);
            ps2_stubs::sceMpegAddStrCallback(
                firstRdram.data(),
                &addStreamContext,
                &first);

            const std::vector<uint8_t> payload = {
                0x11u, 0x22u, 0x33u, 0x44u};
            const uint16_t packetLength =
                static_cast<uint16_t>(
                    payload.size() + 3u);
            std::vector<uint8_t> packet = {
                0x00u, 0x00u, 0x01u, 0xBDu,
                static_cast<uint8_t>(
                    packetLength >> 8u),
                static_cast<uint8_t>(
                    packetLength & 0xFFu),
                0x80u, 0x00u, 0x00u};
            packet.insert(
                packet.end(),
                payload.begin(),
                payload.end());
            std::memcpy(
                firstRdram.data() + kPacketAddr,
                packet.data(),
                packet.size());

            auto dispatchPacket = [&]()
            {
                R5900Context context{};
                setRegU32(context, 4, kMpegAddr);
                setRegU32(context, 5, kPacketAddr);
                setRegU32(
                    context,
                    6,
                    static_cast<uint32_t>(
                        packet.size()));
                setRegU32(context, 7, kPacketAddr);
                setRegU32(
                    context,
                    8,
                    static_cast<uint32_t>(
                        packet.size()));
                ps2_stubs::sceMpegDemuxPssRing(
                    firstRdram.data(),
                    &context,
                    &first);
            };

            gMpegStreamCallbackCount.store(
                0u, std::memory_order_release);
            gMpegStreamCallbackSp.store(
                0u, std::memory_order_release);
            std::thread firstPublisher(dispatchPacket);
            firstPublisher.join();
            const uint32_t firstCallbackSp =
                gMpegStreamCallbackSp.load(
                    std::memory_order_acquire);

            R5900Context resetContext{};
            setRegU32(resetContext, 4, kMpegAddr);
            ps2_stubs::sceMpegReset(
                firstRdram.data(),
                &resetContext,
                &first);
            std::thread secondPublisher(dispatchPacket);
            secondPublisher.join();
            const uint32_t secondCallbackSp =
                gMpegStreamCallbackSp.load(
                    std::memory_order_acquire);

            t.IsTrue(
                firstCallbackSp != 0u,
                "the first host publisher should dispatch the MPEG callback");
            t.Equals(
                secondCallbackSp,
                firstCallbackSp,
                "the MPEG callback stack should belong to the runtime rather than the publishing host thread");

            first.requestStop();
            second.requestStop();
        });

        tc.Run("sceMpegDemuxPssRing dispatches registered video and audio stream callbacks", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            ps2_stubs::resetMpegStubState(&runtime);

            constexpr uint32_t kMpegAddr = 0x00123000u;
            constexpr uint32_t kCallbackEntry = 0x00124000u;
            constexpr uint32_t kVideoUserData = 0x11223344u;
            constexpr uint32_t kAudioUserData = 0x55667788u;
            constexpr uint32_t kVideoPacketAddr = 0x00128000u;
            constexpr uint32_t kAudioPacketAddr = 0x00129000u;

            runtime.registerFunction(kCallbackEntry, &testRecordMpegStreamCallback);

            auto registerGenericCallback = [&](uint32_t callbackType, uint32_t userData)
            {
                R5900Context addCtx{};
                setRegU32(addCtx, 4, kMpegAddr);
                setRegU32(addCtx, 5, callbackType);
                setRegU32(addCtx, 6, kCallbackEntry);
                setRegU32(addCtx, 7, userData);
                ps2_stubs::sceMpegAddCallback(rdram.data(), &addCtx, &runtime);
            };

            auto registerStreamCallback = [&](uint32_t streamType, uint32_t userData)
            {
                R5900Context addCtx{};
                setRegU32(addCtx, 4, kMpegAddr);
                setRegU32(addCtx, 5, streamType);
                setRegU32(addCtx, 6, 0u);
                setRegU32(addCtx, 7, kCallbackEntry);
                setRegU32(addCtx, 8, userData);
                ps2_stubs::sceMpegAddStrCallback(rdram.data(), &addCtx, &runtime);
            };

            auto writePesPacket = [&](uint32_t addr, uint8_t streamId, const std::vector<uint8_t> &payload)
            {
                const uint16_t packetLen = static_cast<uint16_t>(payload.size() + 3u);
                std::vector<uint8_t> packet = {
                    0x00u, 0x00u, 0x01u, streamId,
                    static_cast<uint8_t>(packetLen >> 8u),
                    static_cast<uint8_t>(packetLen & 0xFFu),
                    0x80u, 0x00u, 0x00u};
                packet.insert(packet.end(), payload.begin(), payload.end());
                std::memcpy(rdram.data() + addr, packet.data(), packet.size());
                return static_cast<uint32_t>(packet.size());
            };

            registerGenericCallback(0u, 0xDEAD0000u);
            registerGenericCallback(2u, 0xDEAD0002u);
            registerStreamCallback(0u, kVideoUserData);
            registerStreamCallback(2u, kAudioUserData);

            const std::vector<uint8_t> videoPayload = {
                0x00u, 0x00u, 0x01u, 0xB3u, 0x14u, 0x00u, 0xF0u, 0x13u};
            const uint32_t videoPacketSize = writePesPacket(kVideoPacketAddr, 0xE0u, videoPayload);

            gMpegStreamCallbackCount.store(0u, std::memory_order_release);
            R5900Context videoDemuxCtx{};
            setRegU32(videoDemuxCtx, 4, kMpegAddr);
            setRegU32(videoDemuxCtx, 5, kVideoPacketAddr);
            setRegU32(videoDemuxCtx, 6, videoPacketSize);
            setRegU32(videoDemuxCtx, 7, kVideoPacketAddr);
            setRegU32(videoDemuxCtx, 8, videoPacketSize);
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &videoDemuxCtx, &runtime);

            t.Equals(getRegS32(videoDemuxCtx, 2), static_cast<int32_t>(videoPacketSize),
                     "sceMpegDemuxPssRing should consume the video PES packet");
            t.Equals(gMpegStreamCallbackCount.load(std::memory_order_acquire), 1u,
                     "registered video stream callback should be invoked");
            t.Equals(gMpegStreamCallbackMpeg.load(std::memory_order_acquire), kMpegAddr,
                     "video callback should receive the MPEG handle");
            t.Equals(gMpegStreamCallbackType.load(std::memory_order_acquire), 0u,
                     "video callback data should report M2V stream type");
            t.Equals(gMpegStreamCallbackDataAddr.load(std::memory_order_acquire), kVideoPacketAddr + 9u,
                     "video callback data should point at PES payload");
            t.Equals(gMpegStreamCallbackLen.load(std::memory_order_acquire),
                     static_cast<uint32_t>(videoPayload.size()),
                     "video callback data should report PES payload length");
            t.Equals(gMpegStreamCallbackUserData.load(std::memory_order_acquire), kVideoUserData,
                     "video callback should receive registered user data");

            const std::vector<uint8_t> audioPayload = {0x80u, 0x01u, 0x02u, 0x03u, 0x04u, 0x05u};
            const uint32_t audioPacketSize = writePesPacket(kAudioPacketAddr, 0xBDu, audioPayload);

            gMpegStreamCallbackCount.store(0u, std::memory_order_release);
            R5900Context audioDemuxCtx{};
            setRegU32(audioDemuxCtx, 4, kMpegAddr);
            setRegU32(audioDemuxCtx, 5, kAudioPacketAddr);
            setRegU32(audioDemuxCtx, 6, audioPacketSize);
            setRegU32(audioDemuxCtx, 7, kAudioPacketAddr);
            setRegU32(audioDemuxCtx, 8, audioPacketSize);
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &audioDemuxCtx, &runtime);

            t.Equals(getRegS32(audioDemuxCtx, 2), static_cast<int32_t>(audioPacketSize),
                     "sceMpegDemuxPssRing should consume the audio PES packet");
            t.Equals(gMpegStreamCallbackCount.load(std::memory_order_acquire), 1u,
                     "registered audio stream callback should be invoked");
            t.Equals(gMpegStreamCallbackType.load(std::memory_order_acquire), 2u,
                     "audio callback data should report PCM stream type");
            t.Equals(gMpegStreamCallbackDataAddr.load(std::memory_order_acquire), kAudioPacketAddr + 9u,
                     "audio callback data should point at PES payload");
            t.Equals(gMpegStreamCallbackLen.load(std::memory_order_acquire),
                     static_cast<uint32_t>(audioPayload.size()),
                     "audio callback data should report PES payload length");
            t.Equals(gMpegStreamCallbackUserData.load(std::memory_order_acquire), kAudioUserData,
                     "audio callback should receive registered user data");

            ps2_stubs::notifyMpegCdStreamEof(&runtime);

            gMpegStreamCallbackCount.store(0u, std::memory_order_release);
            R5900Context afterEofDemuxCtx{};
            setRegU32(afterEofDemuxCtx, 4, kMpegAddr);
            setRegU32(afterEofDemuxCtx, 5, kVideoPacketAddr);
            setRegU32(afterEofDemuxCtx, 6, videoPacketSize);
            setRegU32(afterEofDemuxCtx, 7, kVideoPacketAddr);
            setRegU32(afterEofDemuxCtx, 8, videoPacketSize);
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &afterEofDemuxCtx, &runtime);

            t.Equals(getRegS32(afterEofDemuxCtx, 2), static_cast<int32_t>(videoPacketSize),
                     "post-EOF demux should continue consuming caller data");
            t.Equals(gMpegStreamCallbackCount.load(std::memory_order_acquire), 0u,
                     "post-EOF demux should not feed callbacks again");

            R5900Context resetCtx{};
            setRegU32(resetCtx, 4, kMpegAddr);
            ps2_stubs::sceMpegReset(rdram.data(), &resetCtx, &runtime);

            gMpegStreamCallbackCount.store(0u, std::memory_order_release);
            R5900Context afterResetDemuxCtx{};
            setRegU32(afterResetDemuxCtx, 4, kMpegAddr);
            setRegU32(afterResetDemuxCtx, 5, kVideoPacketAddr);
            setRegU32(afterResetDemuxCtx, 6, videoPacketSize);
            setRegU32(afterResetDemuxCtx, 7, kVideoPacketAddr);
            setRegU32(afterResetDemuxCtx, 8, videoPacketSize);
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &afterResetDemuxCtx, &runtime);

            t.Equals(getRegS32(afterResetDemuxCtx, 2), static_cast<int32_t>(videoPacketSize),
                     "post-EOF reset demux should still drain caller data");
            t.Equals(gMpegStreamCallbackCount.load(std::memory_order_acquire), 0u,
                     "post-EOF reset demux should not restart callbacks on stale data");

            ps2_stubs::notifyMpegCdStreamStart(&runtime);

            gMpegStreamCallbackCount.store(0u, std::memory_order_release);
            R5900Context afterNewStreamDemuxCtx{};
            setRegU32(afterNewStreamDemuxCtx, 4, kMpegAddr);
            setRegU32(afterNewStreamDemuxCtx, 5, kVideoPacketAddr);
            setRegU32(afterNewStreamDemuxCtx, 6, videoPacketSize);
            setRegU32(afterNewStreamDemuxCtx, 7, kVideoPacketAddr);
            setRegU32(afterNewStreamDemuxCtx, 8, videoPacketSize);
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &afterNewStreamDemuxCtx, &runtime);

            t.Equals(getRegS32(afterNewStreamDemuxCtx, 2), static_cast<int32_t>(videoPacketSize),
                     "new CD stream demux should reopen an ended MPEG handle");
            t.Equals(gMpegStreamCallbackCount.load(std::memory_order_acquire), 1u,
                     "new CD stream demux should allow callbacks on a reused MPEG handle");

            constexpr uint32_t kMpegWorkAddr = 0x00130000u;
            R5900Context createCtx{};
            setRegU32(createCtx, 4, kMpegAddr);
            setRegU32(createCtx, 5, kMpegWorkAddr);
            setRegU32(createCtx, 6, 0x2000u);
            ps2_stubs::sceMpegCreate(rdram.data(), &createCtx, &runtime);
            t.IsTrue(::getRegU32(&createCtx, 2) != 0u,
                     "sceMpegCreate should reopen the MPEG handle after an ended reset");

            gMpegStreamCallbackCount.store(0u, std::memory_order_release);
            R5900Context afterCreateDemuxCtx{};
            setRegU32(afterCreateDemuxCtx, 4, kMpegAddr);
            setRegU32(afterCreateDemuxCtx, 5, kVideoPacketAddr);
            setRegU32(afterCreateDemuxCtx, 6, videoPacketSize);
            setRegU32(afterCreateDemuxCtx, 7, kVideoPacketAddr);
            setRegU32(afterCreateDemuxCtx, 8, videoPacketSize);
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &afterCreateDemuxCtx, &runtime);

            t.Equals(gMpegStreamCallbackCount.load(std::memory_order_acquire), 1u,
                     "new MPEG create should allow callbacks for the next stream");

            runtime.requestStop();
        });

#if PS2X_HAS_FFMPEG
        tc.Run("MPEG decoder emits the first reference frame without waiting for another GOP", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            ps2_stubs::resetMpegStubState(&runtime);

            constexpr uint32_t kMpegAddr = 0x00123000u;
            constexpr uint32_t kMpegWorkAddr = 0x00130000u;
            constexpr uint32_t kPacketAddr = 0x00140000u;
            constexpr uint32_t kImageAddr = 0x00180000u;
            const std::vector<uint8_t> elementaryStream =
                makeSyntheticMpeg2StartupStream();
            t.IsFalse(elementaryStream.empty(),
                      "the test should generate a synthetic MPEG-2 stream");
            t.IsTrue(elementaryStream.size() <= 0xFFFCu,
                     "the synthetic MPEG-2 stream should fit in one PES packet");
            if (elementaryStream.empty() || elementaryStream.size() > 0xFFFCu)
            {
                return;
            }

            const uint16_t packetLen =
                static_cast<uint16_t>(elementaryStream.size() + 3u);
            std::vector<uint8_t> packet = {
                0x00u, 0x00u, 0x01u, 0xE0u,
                static_cast<uint8_t>(packetLen >> 8u),
                static_cast<uint8_t>(packetLen & 0xFFu),
                0x80u, 0x00u, 0x00u};
            packet.insert(packet.end(), elementaryStream.begin(), elementaryStream.end());
            std::memcpy(rdram.data() + kPacketAddr, packet.data(), packet.size());

            R5900Context createCtx{};
            setRegU32(createCtx, 4, kMpegAddr);
            setRegU32(createCtx, 5, kMpegWorkAddr);
            setRegU32(createCtx, 6, 0x2000u);
            ps2_stubs::sceMpegCreate(rdram.data(), &createCtx, &runtime);

            R5900Context demuxCtx{};
            setRegU32(demuxCtx, 4, kMpegAddr);
            setRegU32(demuxCtx, 5, kPacketAddr);
            setRegU32(demuxCtx, 6, static_cast<uint32_t>(packet.size()));
            setRegU32(demuxCtx, 7, kPacketAddr);
            setRegU32(demuxCtx, 8, static_cast<uint32_t>(packet.size()));
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &demuxCtx, &runtime);
            t.Equals(getRegS32(demuxCtx, 2), static_cast<int32_t>(packet.size()),
                     "the synthetic video PES packet should be consumed");

            R5900Context pictureCtx{};
            setRegU32(pictureCtx, 4, kMpegAddr);
            setRegU32(pictureCtx, 5, kImageAddr);
            ps2_stubs::sceMpegGetPicture(rdram.data(), &pictureCtx, &runtime);
            t.Equals(getRegS32(pictureCtx, 2), 1,
                     "the first I picture should be available after the following P reference");
            t.Equals(Ps2FastRead32(rdram.data(), kMpegAddr + 0x00u), 32u,
                     "the decoded frame should publish its synthetic width");
            t.Equals(Ps2FastRead32(rdram.data(), kMpegAddr + 0x04u), 16u,
                     "the decoded frame should publish its synthetic height");

            ps2_stubs::resetMpegStubState(&runtime);
        });

        tc.Run("host-decoded MPEG video is not stalled by a bypassed IPU callback", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            ps2_stubs::resetMpegStubState(&runtime);

            constexpr uint32_t kMpegAddr = 0x00123000u;
            constexpr uint32_t kCallbackEntry = 0x00124000u;
            constexpr uint32_t kPacketAddr = 0x00128000u;
            const std::vector<uint8_t> payload = {
                0x00u, 0x00u, 0x01u, 0xB3u, 0x14u, 0x00u, 0xF0u, 0x13u};
            const uint16_t packetLen = static_cast<uint16_t>(payload.size() + 3u);
            std::vector<uint8_t> packet = {
                0x00u, 0x00u, 0x01u, 0xE0u,
                static_cast<uint8_t>(packetLen >> 8u),
                static_cast<uint8_t>(packetLen & 0xFFu),
                0x80u, 0x00u, 0x00u};
            packet.insert(packet.end(), payload.begin(), payload.end());
            std::memcpy(rdram.data() + kPacketAddr, packet.data(), packet.size());

            runtime.registerFunction(kCallbackEntry, &testRecordMpegStreamCallback);

            R5900Context addCtx{};
            setRegU32(addCtx, 4, kMpegAddr);
            setRegU32(addCtx, 5, 0u);
            setRegU32(addCtx, 6, 0u);
            setRegU32(addCtx, 7, kCallbackEntry);
            setRegU32(addCtx, 8, 0x55667788u);
            ps2_stubs::sceMpegAddStrCallback(rdram.data(), &addCtx, &runtime);

            gMpegStreamCallbackCount.store(0u, std::memory_order_release);
            gMpegStreamCallbackReturn.store(0u, std::memory_order_release);

            R5900Context demuxCtx{};
            setRegU32(demuxCtx, 4, kMpegAddr);
            setRegU32(demuxCtx, 5, kPacketAddr);
            setRegU32(demuxCtx, 6, static_cast<uint32_t>(packet.size()));
            setRegU32(demuxCtx, 7, kPacketAddr);
            setRegU32(demuxCtx, 8, static_cast<uint32_t>(packet.size()));
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &demuxCtx, &runtime);

            t.Equals(gMpegStreamCallbackCount.load(std::memory_order_acquire), 1u,
                     "the native IPU feed callback should still be observable");
            t.Equals(getRegS32(demuxCtx, 2), static_cast<int32_t>(packet.size()),
                     "the host-owned video stream should acknowledge bytes already accepted by FFmpeg");

            gMpegStreamCallbackReturn.store(1u, std::memory_order_release);
            runtime.requestStop();
        });
#endif

        tc.Run("sceMpegDemuxPssRing dispatches private-stream data callbacks", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            ps2_stubs::resetMpegStubState(&runtime);

            constexpr uint32_t kMpegAddr = 0x00123000u;
            constexpr uint32_t kCallbackEntry = 0x00124000u;
            constexpr uint32_t kCallbackUserData = 0x55667788u;
            constexpr uint32_t kPacketAddr = 0x00129000u;
            constexpr uint32_t kDataStreamType = 3u;
            const std::vector<uint8_t> payload = {
                0x21u, 0x43u, 0x65u, 0x87u, 0xA9u, 0xCBu};

            runtime.configureGuestHeap(0x01F00000u, 0x01F00000u);
            t.Equals(runtime.guestMalloc(0x20u, 16u), 0u,
                     "test setup should leave no runtime guest heap for callback scratch data");
            runtime.registerFunction(kCallbackEntry, &testRecordMpegStreamCallback);

            R5900Context addCtx{};
            setRegU32(addCtx, 4, kMpegAddr);
            setRegU32(addCtx, 5, kDataStreamType);
            setRegU32(addCtx, 6, 0u);
            setRegU32(addCtx, 7, kCallbackEntry);
            setRegU32(addCtx, 8, kCallbackUserData);
            ps2_stubs::sceMpegAddStrCallback(rdram.data(), &addCtx, &runtime);

            const uint16_t packetLen = static_cast<uint16_t>(payload.size() + 3u);
            std::vector<uint8_t> packet = {
                0x00u, 0x00u, 0x01u, 0xBDu,
                static_cast<uint8_t>(packetLen >> 8u),
                static_cast<uint8_t>(packetLen & 0xFFu),
                0x80u, 0x00u, 0x00u};
            packet.insert(packet.end(), payload.begin(), payload.end());
            std::memcpy(rdram.data() + kPacketAddr, packet.data(), packet.size());

            gMpegStreamCallbackCount.store(0u, std::memory_order_release);
            R5900Context demuxCtx{};
            setRegU32(demuxCtx, 4, kMpegAddr);
            setRegU32(demuxCtx, 5, kPacketAddr);
            setRegU32(demuxCtx, 6, static_cast<uint32_t>(packet.size()));
            setRegU32(demuxCtx, 7, kPacketAddr);
            setRegU32(demuxCtx, 8, static_cast<uint32_t>(packet.size()));
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &demuxCtx, &runtime);

            t.Equals(getRegS32(demuxCtx, 2), static_cast<int32_t>(packet.size()),
                     "sceMpegDemuxPssRing should consume a private-stream PES packet");
            t.Equals(gMpegStreamCallbackCount.load(std::memory_order_acquire), 1u,
                     "registered data stream callback should be invoked once");
            t.Equals(gMpegStreamCallbackMpeg.load(std::memory_order_acquire), kMpegAddr,
                     "data callback should receive the MPEG handle");
            t.Equals(gMpegStreamCallbackType.load(std::memory_order_acquire), kDataStreamType,
                     "data callback should report the registered stream type");
            t.Equals(gMpegStreamCallbackDataAddr.load(std::memory_order_acquire), kPacketAddr + 9u,
                     "data callback should point at the raw PES payload");
            t.Equals(gMpegStreamCallbackLen.load(std::memory_order_acquire),
                     static_cast<uint32_t>(payload.size()),
                     "data callback should report the raw PES payload length");
            t.Equals(gMpegStreamCallbackUserData.load(std::memory_order_acquire), kCallbackUserData,
                     "data callback should receive registered user data");

            runtime.requestStop();
        });

        tc.Run("sceMpegDemuxPssRing preserves wrapped callback guest addresses", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            ps2_stubs::resetMpegStubState(&runtime);

            constexpr uint32_t kMpegAddr = 0x00123000u;
            constexpr uint32_t kCallbackEntry = 0x00124000u;
            constexpr uint32_t kRingBase = 0x00129000u;
            constexpr uint32_t kRingSize = 0x40u;
            constexpr uint32_t kPacketOffset = 0x30u;
            constexpr uint32_t kDataStreamType = 3u;
            const std::vector<uint8_t> payload = {
                0x10u, 0x21u, 0x32u, 0x43u, 0x54u, 0x65u, 0x76u, 0x87u,
                0x98u, 0xA9u, 0xBAu, 0xCBu, 0xDCu, 0xEDu, 0xFEu, 0x0Fu};

            runtime.registerFunction(kCallbackEntry, &testRecordMpegStreamCallback);

            R5900Context addCtx{};
            setRegU32(addCtx, 4, kMpegAddr);
            setRegU32(addCtx, 5, kDataStreamType);
            setRegU32(addCtx, 6, 0u);
            setRegU32(addCtx, 7, kCallbackEntry);
            setRegU32(addCtx, 8, 0x55667788u);
            ps2_stubs::sceMpegAddStrCallback(rdram.data(), &addCtx, &runtime);

            const uint16_t packetLen = static_cast<uint16_t>(payload.size() + 3u);
            std::vector<uint8_t> packet = {
                0x00u, 0x00u, 0x01u, 0xBDu,
                static_cast<uint8_t>(packetLen >> 8u),
                static_cast<uint8_t>(packetLen & 0xFFu),
                0x80u, 0x00u, 0x00u};
            packet.insert(packet.end(), payload.begin(), payload.end());
            for (size_t index = 0u; index < packet.size(); ++index)
            {
                const uint32_t ringOffset =
                    (kPacketOffset + static_cast<uint32_t>(index)) % kRingSize;
                rdram[kRingBase + ringOffset] = packet[index];
            }

            {
                std::lock_guard<std::mutex> lock(gMpegStreamCallbackPayloadMutex);
                gMpegStreamCallbackPayload.clear();
            }
            gMpegStreamCallbackCount.store(0u, std::memory_order_release);

            R5900Context demuxCtx{};
            setRegU32(demuxCtx, 4, kMpegAddr);
            setRegU32(demuxCtx, 5, kRingBase + kPacketOffset);
            setRegU32(demuxCtx, 6, static_cast<uint32_t>(packet.size()));
            setRegU32(demuxCtx, 7, kRingBase);
            setRegU32(demuxCtx, 8, kRingSize);
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &demuxCtx, &runtime);

            const uint32_t wrappedPayloadStart =
                kRingBase + ((kPacketOffset + 9u) % kRingSize);
            t.Equals(getRegS32(demuxCtx, 2), static_cast<int32_t>(packet.size()),
                     "sceMpegDemuxPssRing should consume the wrapped PES packet");
            t.Equals(gMpegStreamCallbackCount.load(std::memory_order_acquire), 1u,
                     "wrapped private-stream payload should dispatch one callback");
            t.Equals(gMpegStreamCallbackLen.load(std::memory_order_acquire),
                     static_cast<uint32_t>(payload.size()),
                     "wrapped callback should preserve the payload length");
            t.Equals(gMpegStreamCallbackDataAddr.load(std::memory_order_acquire),
                     wrappedPayloadStart,
                     "ring-aware callbacks should receive the original wrapped guest address");

            runtime.requestStop();
        });

        tc.Run("sceMpegDemuxPssRing retries a backpressured callback with stable data", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            ps2_stubs::resetMpegStubState(&runtime);

            constexpr uint32_t kMpegAddr = 0x00123000u;
            constexpr uint32_t kCallbackEntry = 0x00124000u;
            constexpr uint32_t kPacketAddr = 0x00129000u;
            constexpr uint32_t kDataStreamType = 3u;
            const std::vector<uint8_t> payload = {
                0x12u, 0x34u, 0x56u, 0x78u, 0x9Au, 0xBCu, 0xDEu, 0xF0u};

            runtime.registerFunction(kCallbackEntry, &testRecordMpegStreamCallback);

            R5900Context addCtx{};
            setRegU32(addCtx, 4, kMpegAddr);
            setRegU32(addCtx, 5, kDataStreamType);
            setRegU32(addCtx, 6, 0u);
            setRegU32(addCtx, 7, kCallbackEntry);
            setRegU32(addCtx, 8, 0x55667788u);
            ps2_stubs::sceMpegAddStrCallback(rdram.data(), &addCtx, &runtime);

            const uint16_t packetLen = static_cast<uint16_t>(payload.size() + 3u);
            std::vector<uint8_t> packet = {
                0x00u, 0x00u, 0x01u, 0xBDu,
                static_cast<uint8_t>(packetLen >> 8u),
                static_cast<uint8_t>(packetLen & 0xFFu),
                0x80u, 0x00u, 0x00u};
            packet.insert(packet.end(), payload.begin(), payload.end());
            std::memcpy(rdram.data() + kPacketAddr, packet.data(), packet.size());

            gMpegStreamCallbackCount.store(0u, std::memory_order_release);
            gMpegStreamCallbackReturn.store(0u, std::memory_order_release);

            R5900Context blockedCtx{};
            setRegU32(blockedCtx, 4, kMpegAddr);
            setRegU32(blockedCtx, 5, kPacketAddr);
            setRegU32(blockedCtx, 6, static_cast<uint32_t>(packet.size()));
            setRegU32(blockedCtx, 7, kPacketAddr);
            setRegU32(blockedCtx, 8, static_cast<uint32_t>(packet.size()));
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &blockedCtx, &runtime);

            t.Equals(gMpegStreamCallbackCount.load(std::memory_order_acquire), 1u,
                     "a callback that reports backpressure should be invoked once");
            t.Equals(getRegS32(blockedCtx, 2), 0,
                     "backpressure should retain ownership of the offered input batch");

            std::memset(rdram.data() + kPacketAddr, 0xEE, packet.size());
            gMpegStreamCallbackReturn.store(1u, std::memory_order_release);

            R5900Context overwrittenRetryCtx{};
            setRegU32(overwrittenRetryCtx, 4, kMpegAddr);
            setRegU32(overwrittenRetryCtx, 5, kPacketAddr);
            setRegU32(overwrittenRetryCtx, 6, static_cast<uint32_t>(packet.size()));
            setRegU32(overwrittenRetryCtx, 7, kPacketAddr);
            setRegU32(overwrittenRetryCtx, 8, static_cast<uint32_t>(packet.size()));
            ps2_stubs::sceMpegDemuxPssRing(
                rdram.data(), &overwrittenRetryCtx, &runtime);

            t.Equals(gMpegStreamCallbackCount.load(std::memory_order_acquire), 1u,
                     "changed retained bytes should not be delivered through stale guest pointers");
            t.Equals(getRegS32(overwrittenRetryCtx, 2), 0,
                     "changed retained bytes should remain unacknowledged");

            std::memcpy(rdram.data() + kPacketAddr, packet.data(), packet.size());
            R5900Context retryCtx{};
            setRegU32(retryCtx, 4, kMpegAddr);
            setRegU32(retryCtx, 5, kPacketAddr);
            setRegU32(retryCtx, 6, static_cast<uint32_t>(packet.size()));
            setRegU32(retryCtx, 7, kPacketAddr);
            setRegU32(retryCtx, 8, static_cast<uint32_t>(packet.size()));
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &retryCtx, &runtime);

            std::vector<uint8_t> observedPayload;
            {
                std::lock_guard<std::mutex> lock(gMpegStreamCallbackPayloadMutex);
                observedPayload = gMpegStreamCallbackPayload;
            }

            t.Equals(gMpegStreamCallbackCount.load(std::memory_order_acquire), 2u,
                     "the pending callback should retry when the same input is presented");
            t.Equals(getRegS32(retryCtx, 2), static_cast<int32_t>(packet.size()),
                     "a successful retry should acknowledge the retained input batch");
            t.IsTrue(observedPayload == payload,
                     "the retried callback should observe the unchanged guest payload");
            t.Equals(gMpegStreamCallbackDataAddr.load(std::memory_order_acquire),
                     kPacketAddr + 9u,
                     "a retry should preserve the original guest payload address");

            gMpegStreamCallbackReturn.store(1u, std::memory_order_release);
            runtime.requestStop();
        });

        tc.Run("sceMpegDemuxPssRing acknowledges packets before a backpressured callback", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            ps2_stubs::resetMpegStubState(&runtime);

            constexpr uint32_t kMpegAddr = 0x00123000u;
            constexpr uint32_t kCallbackEntry = 0x00124000u;
            constexpr uint32_t kRingBase = 0x00129000u;
            constexpr uint32_t kRingSize = 0x40u;
            constexpr uint32_t kInputOffset = 0x28u;
            constexpr uint32_t kDataStreamType = 3u;
            const std::vector<uint8_t> firstPayload = {
                0x10u, 0x21u, 0x32u, 0x43u, 0x54u, 0x65u, 0x76u, 0x87u};
            const std::vector<uint8_t> secondPayload = {
                0x98u, 0xA9u, 0xBAu, 0xCBu, 0xDCu, 0xEDu, 0xFEu, 0x0Fu};
            const std::vector<uint8_t> thirdPayload = {
                0x01u, 0x23u, 0x45u, 0x67u, 0x89u, 0xABu, 0xCDu, 0xEFu};
            const auto makePrivatePacket = [](const std::vector<uint8_t> &payload)
            {
                const uint16_t packetLen =
                    static_cast<uint16_t>(payload.size() + 3u);
                std::vector<uint8_t> packet = {
                    0x00u, 0x00u, 0x01u, 0xBDu,
                    static_cast<uint8_t>(packetLen >> 8u),
                    static_cast<uint8_t>(packetLen & 0xFFu),
                    0x80u, 0x00u, 0x00u};
                packet.insert(packet.end(), payload.begin(), payload.end());
                return packet;
            };
            const std::vector<uint8_t> firstPacket =
                makePrivatePacket(firstPayload);
            const std::vector<uint8_t> secondPacket =
                makePrivatePacket(secondPayload);
            const std::vector<uint8_t> thirdPacket =
                makePrivatePacket(thirdPayload);
            std::vector<uint8_t> input = firstPacket;
            input.insert(input.end(), secondPacket.begin(), secondPacket.end());
            input.insert(input.end(), thirdPacket.begin(), thirdPacket.end());

            runtime.registerFunction(
                kCallbackEntry, &testRejectSecondMpegStreamCallback);

            R5900Context addCtx{};
            setRegU32(addCtx, 4, kMpegAddr);
            setRegU32(addCtx, 5, kDataStreamType);
            setRegU32(addCtx, 6, 0u);
            setRegU32(addCtx, 7, kCallbackEntry);
            setRegU32(addCtx, 8, 0x55667788u);
            ps2_stubs::sceMpegAddStrCallback(rdram.data(), &addCtx, &runtime);

            for (size_t index = 0u; index < input.size(); ++index)
            {
                const uint32_t ringOffset =
                    (kInputOffset + static_cast<uint32_t>(index)) % kRingSize;
                rdram[kRingBase + ringOffset] = input[index];
            }

            gMpegStreamCallbackCount.store(0u, std::memory_order_release);
            gMpegStreamCallbackReturn.store(1u, std::memory_order_release);

            R5900Context firstDemuxCtx{};
            setRegU32(firstDemuxCtx, 4, kMpegAddr);
            setRegU32(firstDemuxCtx, 5, kRingBase + kInputOffset);
            setRegU32(firstDemuxCtx, 6, static_cast<uint32_t>(input.size()));
            setRegU32(firstDemuxCtx, 7, kRingBase);
            setRegU32(firstDemuxCtx, 8, kRingSize);
            ps2_stubs::sceMpegDemuxPssRing(
                rdram.data(), &firstDemuxCtx, &runtime);

            t.Equals(gMpegStreamCallbackCount.load(std::memory_order_acquire), 2u,
                     "demux should stop when the second packet callback rejects input");
            t.Equals(getRegS32(firstDemuxCtx, 2),
                     static_cast<int32_t>(firstPacket.size()),
                     "demux should acknowledge the complete packet before backpressure");

            const uint32_t retryOffset =
                (kInputOffset + static_cast<uint32_t>(firstPacket.size())) %
                kRingSize;
            R5900Context retryCtx{};
            setRegU32(retryCtx, 4, kMpegAddr);
            setRegU32(retryCtx, 5, kRingBase + retryOffset);
            setRegU32(
                retryCtx,
                6,
                static_cast<uint32_t>(
                    secondPacket.size() + thirdPacket.size()));
            setRegU32(retryCtx, 7, kRingBase);
            setRegU32(retryCtx, 8, kRingSize);
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &retryCtx, &runtime);

            std::vector<uint8_t> observedPayload;
            {
                std::lock_guard<std::mutex> lock(gMpegStreamCallbackPayloadMutex);
                observedPayload = gMpegStreamCallbackPayload;
            }
            const uint32_t expectedPayloadAddr =
                kRingBase + ((retryOffset + 9u) % kRingSize);
            t.Equals(gMpegStreamCallbackCount.load(std::memory_order_acquire), 3u,
                     "the rejected packet callback should retry once");
            t.Equals(getRegS32(retryCtx, 2),
                     static_cast<int32_t>(secondPacket.size()),
                     "a successful retry should acknowledge only the blocked packet");
            t.Equals(gMpegStreamCallbackDataAddr.load(std::memory_order_acquire),
                     expectedPayloadAddr,
                     "suffix retry should preserve the wrapped guest payload address");
            t.IsTrue(observedPayload == secondPayload,
                     "suffix retry should deliver the unchanged second payload");

            const uint32_t thirdOffset =
                (retryOffset + static_cast<uint32_t>(secondPacket.size())) %
                kRingSize;
            R5900Context thirdCtx{};
            setRegU32(thirdCtx, 4, kMpegAddr);
            setRegU32(thirdCtx, 5, kRingBase + thirdOffset);
            setRegU32(
                thirdCtx, 6, static_cast<uint32_t>(thirdPacket.size()));
            setRegU32(thirdCtx, 7, kRingBase);
            setRegU32(thirdCtx, 8, kRingSize);
            ps2_stubs::sceMpegDemuxPssRing(
                rdram.data(), &thirdCtx, &runtime);

            {
                std::lock_guard<std::mutex> lock(
                    gMpegStreamCallbackPayloadMutex);
                observedPayload = gMpegStreamCallbackPayload;
            }
            t.Equals(
                gMpegStreamCallbackCount.load(std::memory_order_acquire),
                4u,
                "input after the blocked packet should remain caller-owned");
            t.Equals(
                getRegS32(thirdCtx, 2),
                static_cast<int32_t>(thirdPacket.size()),
                "the caller should be able to submit the unretained tail");
            t.IsTrue(
                observedPayload == thirdPayload,
                "the later call should deliver the caller-owned tail payload");

            runtime.requestStop();
        });

        tc.Run("MPEG playback stays active during a temporary demux pause before CD EOF", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            ps2_stubs::resetMpegStubState(&runtime);

            constexpr uint32_t kMpegAddr = 0x00123000u;
            constexpr uint32_t kPacketAddr = 0x00128000u;
            constexpr uint32_t kImageAddr = 0x00130000u;
            const std::vector<uint8_t> es = {
                0x00u, 0x00u, 0x01u, 0xB3u, 0x01u, 0x00u, 0x10u, 0x12u, 0xFFu, 0xFFu, 0xE0u, 0x18u,
                0x00u, 0x00u, 0x01u, 0xB5u, 0x14u, 0x8Au, 0x00u, 0x01u, 0x00u, 0x17u, 0x00u, 0x00u,
                0x01u, 0xB8u, 0x00u, 0x08u, 0x00u, 0x40u, 0x00u, 0x00u, 0x01u, 0x00u, 0x00u, 0x0Fu,
                0xFFu, 0xF8u, 0x00u, 0x00u, 0x01u, 0xB5u, 0x8Fu, 0xFFu, 0xF3u, 0x41u, 0x80u, 0x00u,
                0x00u, 0x01u, 0x01u, 0x13u, 0xF8u, 0x7Du, 0x29u, 0x48u, 0x88u, 0x00u, 0x00u, 0x01u,
                0xB3u, 0x01u, 0x00u, 0x10u, 0x12u, 0xFFu, 0xFFu, 0xE0u, 0x18u, 0x00u, 0x00u, 0x01u,
                0xB5u, 0x14u, 0x8Au, 0x00u, 0x01u, 0x00u, 0x17u, 0x00u, 0x00u, 0x01u, 0xB8u, 0x00u,
                0x08u, 0x00u, 0xC0u, 0x00u, 0x00u, 0x01u, 0x00u, 0x00u, 0x0Fu, 0xFFu, 0xF8u, 0x00u,
                0x00u, 0x01u, 0xB5u, 0x8Fu, 0xFFu, 0xF3u, 0x41u, 0x80u, 0x00u, 0x00u, 0x01u, 0x01u,
                0x13u, 0xF8u, 0x7Du, 0x29u, 0x48u, 0x88u, 0x00u, 0x00u, 0x01u, 0xB3u, 0x01u, 0x00u,
                0x10u, 0x12u, 0xFFu, 0xFFu, 0xE0u, 0x18u, 0x00u, 0x00u, 0x01u, 0xB5u, 0x14u, 0x8Au,
                0x00u, 0x01u, 0x00u, 0x17u, 0x00u, 0x00u, 0x01u, 0xB8u, 0x00u, 0x08u, 0x01u, 0x40u,
                0x00u, 0x00u, 0x01u, 0x00u, 0x00u, 0x0Fu, 0xFFu, 0xF8u, 0x00u, 0x00u, 0x01u, 0xB5u,
                0x8Fu, 0xFFu, 0xF3u, 0x41u, 0x80u, 0x00u, 0x00u, 0x01u, 0x01u, 0x13u, 0xF8u, 0x7Du,
                0x29u, 0x48u, 0x88u};

            std::vector<uint8_t> packet = {
                0x00u, 0x00u, 0x01u, 0xE0u,
                0x00u, static_cast<uint8_t>(es.size() + 3u),
                0x80u, 0x00u, 0x00u};
            packet.insert(packet.end(), es.begin(), es.end());
            std::memcpy(rdram.data() + kPacketAddr, packet.data(), packet.size());

            R5900Context demuxCtx{};
            setRegU32(demuxCtx, 4, kMpegAddr);
            setRegU32(demuxCtx, 5, kPacketAddr);
            setRegU32(demuxCtx, 6, static_cast<uint32_t>(packet.size()));
            setRegU32(demuxCtx, 7, kPacketAddr);
            setRegU32(demuxCtx, 8, static_cast<uint32_t>(packet.size()));
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &demuxCtx, &runtime);

            // This fixture polls decoder state synchronously. A stopped
            // runtime preserves the old non-blocking direct-stub behavior;
            // live-runtime starvation is covered by the waiter tests below.
            runtime.requestStop();

            R5900Context pictureCtx{};
            setRegU32(pictureCtx, 4, kMpegAddr);
            setRegU32(pictureCtx, 5, kImageAddr);
            ps2_stubs::sceMpegGetPicture(rdram.data(), &pictureCtx, &runtime);
            t.Equals(Ps2FastRead32(rdram.data(), kMpegAddr + 0x00u), 16u,
                     "test stream should decode one frame before the pause");
            t.Equals(Ps2FastRead32(rdram.data(), kMpegAddr + 0x08u), 0u,
                     "first decoded frame should report frame index zero");

            std::this_thread::sleep_for(std::chrono::milliseconds(650));

            R5900Context isEndCtx{};
            setRegU32(isEndCtx, 4, kMpegAddr);
            ps2_stubs::sceMpegIsEnd(rdram.data(), &isEndCtx, &runtime);
            t.Equals(getRegS32(isEndCtx, 2), 0,
                     "a temporary demux pause must not end an active stream before CD EOF");

            ps2_stubs::sceMpegGetPicture(rdram.data(), &pictureCtx, &runtime);
            t.Equals(Ps2FastRead32(rdram.data(), kMpegAddr + 0x08u), 1u,
                     "temporary non-EOF starvation should keep movie frame progress moving");

            ps2_stubs::sceMpegGetPicture(rdram.data(), &pictureCtx, &runtime);
            t.Equals(Ps2FastRead32(rdram.data(), kMpegAddr + 0x08u), 2u,
                     "repeated temporary starvation should continue advancing from the held frame");
        });

        tc.Run("sceMpegGetPicture releases an old waiter when the CD stream restarts", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            ps2_stubs::resetMpegStubState(&runtime);
            ps2_stubs::notifyMpegCdStreamStart(&runtime);

            constexpr uint32_t kMpegAddr = 0x00123000u;
            constexpr uint32_t kImageAddr = 0x00130000u;
            R5900Context pictureCtx{};
            setRegU32(pictureCtx, 4, kMpegAddr);
            setRegU32(pictureCtx, 5, kImageAddr);

            std::atomic<bool> returned{false};
            std::thread waiter([&]()
            {
                ps2_stubs::sceMpegGetPicture(rdram.data(), &pictureCtx, &runtime);
                returned.store(true, std::memory_order_release);
            });

            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            t.IsFalse(returned.load(std::memory_order_acquire),
                      "sceMpegGetPicture should wait while the current stream still has no frame");

            ps2_stubs::notifyMpegCdStreamStart(&runtime);
            const bool released = waitUntil(
                [&]() { return returned.load(std::memory_order_acquire); },
                std::chrono::milliseconds(30));

            runtime.requestStop();
            waiter.join();
            t.IsTrue(released,
                     "a new sceCdStStart generation should release a waiter owned by the previous movie");
        });

        tc.Run("sceMpegGetPicture blocks during active stream starvation before CD EOF", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            ps2_stubs::resetMpegStubState(&runtime);
            ps2_stubs::notifyMpegCdStreamStart(&runtime);

            constexpr uint32_t kMpegAddr = 0x00123000u;
            constexpr uint32_t kPssAddr = 0x0012C000u;
            constexpr uint32_t kImageAddr = 0x00130000u;
            const uint8_t incompletePssStart[] = {0x00u, 0x00u, 0x01u};
            std::memcpy(rdram.data() + kPssAddr, incompletePssStart, sizeof(incompletePssStart));

            R5900Context demuxCtx{};
            setRegU32(demuxCtx, 4, kMpegAddr);
            setRegU32(demuxCtx, 5, kPssAddr);
            setRegU32(demuxCtx, 6, sizeof(incompletePssStart));
            setRegU32(demuxCtx, 7, kPssAddr);
            setRegU32(demuxCtx, 8, sizeof(incompletePssStart));
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &demuxCtx, &runtime);

            R5900Context pictureCtx{};
            setRegU32(pictureCtx, 4, kMpegAddr);
            setRegU32(pictureCtx, 5, kImageAddr);

            std::atomic<bool> returned{false};
            std::thread waiter([&]()
            {
                ps2_stubs::sceMpegGetPicture(rdram.data(), &pictureCtx, &runtime);
                returned.store(true, std::memory_order_release);
            });

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            t.IsFalse(returned.load(std::memory_order_acquire),
                      "active non-EOF streams must keep waiting when no decoded frame is currently available");

            R5900Context isEndCtx{};
            setRegU32(isEndCtx, 4, kMpegAddr);
            ps2_stubs::sceMpegIsEnd(rdram.data(), &isEndCtx, &runtime);
            t.Equals(getRegS32(isEndCtx, 2), 0,
                     "waiting without a frame must not mark the active stream ended");

            runtime.requestStop();
            waiter.join();
            t.Equals(getRegS32(pictureCtx, 2), -1,
                     "runtime teardown should interrupt a pending picture wait with an error");
        });

        tc.Run("movie startup MPEG and audio stubs return safe progress values", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            ps2_stubs::resetMpegStubState(&runtime);
            ps2_stubs::resetAudioStubState(&runtime);

            R5900Context firstIsEndCtx{};
            setRegU32(firstIsEndCtx, 4, 0x00123000u);
            ps2_stubs::sceMpegIsEnd(rdram.data(), &firstIsEndCtx, &runtime);
            t.Equals(getRegS32(firstIsEndCtx, 2), 0,
                     "sceMpegIsEnd should allow one synthetic frame before reporting end");

            R5900Context demuxCtx{};
            setRegU32(demuxCtx, 4, 0x00123000u);
            setRegU32(demuxCtx, 5, 0x00400000u);
            setRegU32(demuxCtx, 6, 0x00004000u);
            setRegU32(demuxCtx, 7, 0x00410000u);
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &demuxCtx, &runtime);
            t.Equals(getRegS32(demuxCtx, 2), 0x4000,
                     "sceMpegDemuxPssRing should consume the provided input instead of trapping");

            // This compatibility fixture inspects synthetic output without a
            // producer. Use stopped-runtime semantics instead of the removed
            // process-global null-runtime shortcut.
            runtime.requestStop();

            R5900Context getPictureCtx{};
            setRegU32(getPictureCtx, 4, 0x00123000u);
            setRegU32(getPictureCtx, 5, 0x00124000u);
            setRegU32(getPictureCtx, 6, 440u);
            ps2_stubs::sceMpegGetPicture(rdram.data(), &getPictureCtx, &runtime);
            t.Equals(Ps2FastRead32(rdram.data(), 0x00123000u + 0x00u), 320u,
                     "sceMpegGetPicture should seed a safe movie width");
            t.Equals(Ps2FastRead32(rdram.data(), 0x00123000u + 0x04u), 240u,
                     "sceMpegGetPicture should seed a safe movie height");
            t.Equals(Ps2FastRead32(rdram.data(), 0x00123000u + 0x08u), 0u,
                     "first synthetic picture should preserve frameCount==0 for guest setup");

            R5900Context secondIsEndCtx{};
            setRegU32(secondIsEndCtx, 4, 0x00123000u);
            ps2_stubs::sceMpegIsEnd(rdram.data(), &secondIsEndCtx, &runtime);
            t.Equals(getRegS32(secondIsEndCtx, 2), 0,
                     "sceMpegIsEnd should keep the decode thread alive and let the guest stop playback");

            constexpr uint32_t pssEndAddr = 0x00128000u;
            constexpr uint32_t stackAddr = 0x00129000u;
            const uint8_t programEnd[] = {0x00u, 0x00u, 0x01u, 0xB9u};
            std::memcpy(rdram.data() + pssEndAddr, programEnd, sizeof(programEnd));
            std::memcpy(rdram.data() + stackAddr + 0x10u, "\x04\x00\x00\x00", 4u);

            R5900Context endDemuxCtx{};
            setRegU32(endDemuxCtx, 29, stackAddr);
            setRegU32(endDemuxCtx, 4, 0x00123000u);
            setRegU32(endDemuxCtx, 5, pssEndAddr);
            setRegU32(endDemuxCtx, 6, sizeof(programEnd));
            setRegU32(endDemuxCtx, 7, pssEndAddr);
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &endDemuxCtx, &runtime);

            R5900Context endIsEndCtx{};
            setRegU32(endIsEndCtx, 4, 0x00123000u);
            ps2_stubs::sceMpegIsEnd(rdram.data(), &endIsEndCtx, &runtime);
            t.Equals(getRegS32(endIsEndCtx, 2), 1,
                     "sceMpegIsEnd should report end after a demuxed MPEG program end code");

            ps2_stubs::resetMpegStubState(&runtime);
            constexpr uint32_t wrappedEndBase = 0x0012A000u;
            rdram[wrappedEndBase + 0u] = 0x00u;
            rdram[wrappedEndBase + 1u] = 0x01u;
            rdram[wrappedEndBase + 2u] = 0xB9u;
            rdram[wrappedEndBase + 3u] = 0x00u;

            R5900Context wrappedEndDemuxCtx{};
            setRegU32(wrappedEndDemuxCtx, 4, 0x00123000u);
            setRegU32(wrappedEndDemuxCtx, 5, wrappedEndBase + 3u);
            setRegU32(wrappedEndDemuxCtx, 6, 4u);
            setRegU32(wrappedEndDemuxCtx, 7, wrappedEndBase);
            setRegU32(wrappedEndDemuxCtx, 8, 4u);
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &wrappedEndDemuxCtx, &runtime);

            R5900Context wrappedEndIsEndCtx{};
            setRegU32(wrappedEndIsEndCtx, 4, 0x00123000u);
            ps2_stubs::sceMpegIsEnd(rdram.data(), &wrappedEndIsEndCtx, &runtime);
            t.Equals(getRegS32(wrappedEndIsEndCtx, 2), 1,
                     "sceMpegDemuxPssRing should use the ABI fifth argument in t0 for wrapped rings");

            ps2_stubs::resetMpegStubState(&runtime);
            constexpr uint32_t eofMpegAddr = 0x0012B000u;
            constexpr uint32_t eofPssAddr = 0x0012C000u;
            const uint8_t incompletePssStart[] = {0x00u, 0x00u, 0x01u};
            std::memcpy(rdram.data() + eofPssAddr, incompletePssStart, sizeof(incompletePssStart));

            R5900Context eofDemuxCtx{};
            setRegU32(eofDemuxCtx, 4, eofMpegAddr);
            setRegU32(eofDemuxCtx, 5, eofPssAddr);
            setRegU32(eofDemuxCtx, 6, sizeof(incompletePssStart));
            setRegU32(eofDemuxCtx, 7, eofPssAddr);
            setRegU32(eofDemuxCtx, 8, sizeof(incompletePssStart));
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &eofDemuxCtx, &runtime);
            t.Equals(getRegS32(eofDemuxCtx, 2), static_cast<int32_t>(sizeof(incompletePssStart)),
                     "sceMpegDemuxPssRing should accept partial trailing stream data");

            R5900Context eofBeforeStopCtx{};
            setRegU32(eofBeforeStopCtx, 4, eofMpegAddr);
            ps2_stubs::sceMpegIsEnd(rdram.data(), &eofBeforeStopCtx, &runtime);
            t.Equals(getRegS32(eofBeforeStopCtx, 2), 0,
                     "sceMpegIsEnd should not report end until the CD stream terminates");

            R5900Context cdStopCtx{};
            ps2_stubs::sceCdStStop(rdram.data(), &cdStopCtx, &runtime);

            R5900Context eofAfterStopCtx{};
            setRegU32(eofAfterStopCtx, 4, eofMpegAddr);
            ps2_stubs::sceMpegIsEnd(rdram.data(), &eofAfterStopCtx, &runtime);
            t.Equals(getRegS32(eofAfterStopCtx, 2), 1,
                     "sceCdStStop should finalize active MPEG playback so movie threads can advance");

            R5900Context remoteInitCtx{};
            ps2_stubs::sceSdRemoteInit(rdram.data(), &remoteInitCtx, &runtime);
            t.Equals(getRegS32(remoteInitCtx, 2), 0,
                     "sceSdRemoteInit should succeed so Veronica can set up movie audio");

            R5900Context blockTransCtx{};
            const uint32_t blockTransSp = 0x00100000u;
            setRegU32(blockTransCtx, 29, blockTransSp);
            setRegU32(blockTransCtx, 4, 1u);
            setRegU32(blockTransCtx, 5, 0x80E0u);
            setRegU32(blockTransCtx, 6, 1u);
            setRegU32(blockTransCtx, 7, 0x13u);
            std::memcpy(rdram.data() + blockTransSp + 0x10u, "\x40\x23\x01\x00", 4u);
            std::memcpy(rdram.data() + blockTransSp + 0x14u, "\x00\x30\x00\x00", 4u);
            std::memcpy(rdram.data() + blockTransSp + 0x18u, "\x40\x27\x01\x00", 4u);
            ps2_stubs::sceSdRemote(rdram.data(), &blockTransCtx, &runtime);
            t.Equals(getRegU32(&blockTransCtx, 2), 0u,
                     "sceSdRemote block-transfer start should report libsd success");

            R5900Context statusCtx{};
            setRegU32(statusCtx, 29, blockTransSp);
            setRegU32(statusCtx, 4, 1u);
            setRegU32(statusCtx, 5, 0x8100u);
            setRegU32(statusCtx, 6, 1u);
            setRegU32(statusCtx, 7, 0u);
            std::memset(rdram.data() + blockTransSp + 0x10u, 0, 12u);
            ps2_stubs::sceSdRemote(rdram.data(), &statusCtx, &runtime);
            t.Equals(getRegU32(&statusCtx, 2), 0x00012B40u,
                     "sceSdRemote status polling should advance the emulated SPU transfer head");

            for (uint32_t i = 0u; i < 11u; ++i)
            {
                ps2_stubs::sceSdRemote(rdram.data(), &statusCtx, &runtime);
            }
            t.Equals(getRegU32(&statusCtx, 2), 0x00012740u,
                     "sceSdRemote status polling should wrap inside the configured IOP ring");

            R5900Context setParamCtx{};
            setRegU32(setParamCtx, 29, blockTransSp);
            setRegU32(setParamCtx, 4, 1u);
            setRegU32(setParamCtx, 5, 0x8010u);
            setRegU32(setParamCtx, 6, 0x0F81u);
            setRegU32(setParamCtx, 7, 0u);
            ps2_stubs::sceSdRemote(rdram.data(), &setParamCtx, &runtime);
            t.Equals(getRegU32(&setParamCtx, 2), 0u,
                     "sceSdRemote set-param calls should not trap or disturb the movie audio state");
        });

        tc.Run("sceMpegGetPicture publishes libmpeg end-of-sequence state", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            ps2_stubs::resetMpegStubState(&runtime);

            constexpr uint32_t kMpegAddr = 0x00123000u;
            constexpr uint32_t kWorkAddr = 0x00140000u;
            constexpr uint32_t kEndCodeAddr = 0x00150000u;
            constexpr uint32_t kImageAddr = 0x00160000u;

            R5900Context createCtx{};
            setRegU32(createCtx, 4, kMpegAddr);
            setRegU32(createCtx, 5, kWorkAddr);
            setRegU32(createCtx, 6, 0x3000u);
            ps2_stubs::sceMpegCreate(rdram.data(), &createCtx, &runtime);

            const uint32_t innerAddr = Ps2FastRead32(rdram.data(), kMpegAddr + 0x40u);
            t.IsTrue(innerAddr != 0u, "sceMpegCreate should publish its internal state address");
            t.Equals(Ps2FastRead32(rdram.data(), innerAddr), 0u,
                     "fresh libmpeg state should not report end");

            const uint8_t programEnd[] = {0x00u, 0x00u, 0x01u, 0xB9u};
            std::memcpy(rdram.data() + kEndCodeAddr, programEnd, sizeof(programEnd));

            R5900Context demuxCtx{};
            setRegU32(demuxCtx, 4, kMpegAddr);
            setRegU32(demuxCtx, 5, kEndCodeAddr);
            setRegU32(demuxCtx, 6, sizeof(programEnd));
            setRegU32(demuxCtx, 7, kEndCodeAddr);
            setRegU32(demuxCtx, 8, sizeof(programEnd));
            ps2_stubs::sceMpegDemuxPssRing(rdram.data(), &demuxCtx, &runtime);

            R5900Context pictureCtx{};
            setRegU32(pictureCtx, 4, kMpegAddr);
            setRegU32(pictureCtx, 5, kImageAddr);
            setRegU32(pictureCtx, 6, 832u);
            ps2_stubs::sceMpegGetPicture(rdram.data(), &pictureCtx, &runtime);

            t.Equals(Ps2FastRead32(rdram.data(), innerAddr), 1u,
                     "sceMpegGetPicture should publish sequence end through libmpeg state");
            t.Equals(getRegS32(pictureCtx, 2), 1,
                     "sceMpegGetPicture should report completed end-of-sequence processing");
        });

        tc.Run("sceSdRemote isolates voice transfers from block streaming state", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            constexpr uint32_t kStackAddr = 0x00100000u;
            constexpr uint32_t kBlockBase = 0x00012340u;
            constexpr uint32_t kBlockSize = 0x00003000u;
            constexpr uint32_t kBlockPause = 0x00012740u;

            R5900Context initCtx{};
            ps2_stubs::sceSdRemoteInit(rdram.data(), &initCtx, &runtime);

            R5900Context blockCtx{};
            setRegU32(blockCtx, 29, kStackAddr);
            setRegU32(blockCtx, 4, 1u);
            setRegU32(blockCtx, 5, 0x80E0u);
            setRegU32(blockCtx, 6, 1u);
            setRegU32(blockCtx, 7, 0x13u);
            setRegU32(blockCtx, 8, kBlockBase);
            setRegU32(blockCtx, 9, kBlockSize);
            setRegU32(blockCtx, 10, kBlockPause);
            ps2_stubs::sceSdRemote(rdram.data(), &blockCtx, &runtime);

            R5900Context blockStatusCtx{};
            setRegU32(blockStatusCtx, 29, kStackAddr);
            setRegU32(blockStatusCtx, 4, 1u);
            setRegU32(blockStatusCtx, 5, 0x8100u);
            setRegU32(blockStatusCtx, 6, 1u);
            setRegU32(blockStatusCtx, 7, 0u);
            ps2_stubs::sceSdRemote(rdram.data(), &blockStatusCtx, &runtime);
            t.Equals(getRegU32(&blockStatusCtx, 2), 0x00012B40u,
                     "initial block-status poll should advance the streaming ring");

            R5900Context voiceCtx{};
            setRegU32(voiceCtx, 29, kStackAddr);
            setRegU32(voiceCtx, 4, 1u);
            setRegU32(voiceCtx, 5, 0x80D0u);
            setRegU32(voiceCtx, 6, 0u);
            setRegU32(voiceCtx, 7, 0u);
            setRegU32(voiceCtx, 8, 0x00022000u);
            setRegU32(voiceCtx, 9, 0x00004000u);
            setRegU32(voiceCtx, 10, 0x00000800u);
            ps2_stubs::sceSdRemote(rdram.data(), &voiceCtx, &runtime);
            t.Equals(getRegU32(&voiceCtx, 2), 0x00000800u,
                     "DMA voice transfer should report its transferred byte count");

            R5900Context voiceStatusCtx{};
            setRegU32(voiceStatusCtx, 29, kStackAddr);
            setRegU32(voiceStatusCtx, 4, 1u);
            setRegU32(voiceStatusCtx, 5, 0x80F0u);
            setRegU32(voiceStatusCtx, 6, 0u);
            setRegU32(voiceStatusCtx, 7, 1u);
            ps2_stubs::sceSdRemote(rdram.data(), &voiceStatusCtx, &runtime);
            t.Equals(getRegU32(&voiceStatusCtx, 2), 1u,
                     "voice-transfer status should complete independently from block position");

            ps2_stubs::sceSdRemote(rdram.data(), &blockStatusCtx, &runtime);
            t.Equals(getRegU32(&blockStatusCtx, 2), 0x00012F40u,
                     "voice transfer should not replace or advance the block-streaming ring");
        });

        tc.Run("sceSdRemote keeps block cursors and loop banks isolated per core", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            constexpr uint32_t kStackAddr = 0x00100000u;

            R5900Context initCtx{};
            ps2_stubs::sceSdRemoteInit(rdram.data(), &initCtx, &runtime);

            auto remote = [&](uint32_t command,
                              uint32_t core,
                              uint32_t mode,
                              uint32_t arg4 = 0u,
                              uint32_t arg5 = 0u,
                              uint32_t arg6 = 0u)
            {
                R5900Context ctx{};
                setRegU32(ctx, 29, kStackAddr);
                setRegU32(ctx, 4, 1u);
                setRegU32(ctx, 5, command);
                setRegU32(ctx, 6, core);
                setRegU32(ctx, 7, mode);
                setRegU32(ctx, 8, arg4);
                setRegU32(ctx, 9, arg5);
                setRegU32(ctx, 10, arg6);
                ps2_stubs::sceSdRemote(rdram.data(), &ctx, &runtime);
                return getRegU32(&ctx, 2);
            };

            t.Equals(remote(0x80E0u, 0u, 0x10u, 0x00010000u, 0x00001000u, 0x00010000u), 0u,
                     "core 0 block stream should start successfully");
            t.Equals(remote(0x80E0u, 1u, 0x13u, 0x00020000u, 0x00002000u, 0x00020800u), 0u,
                     "core 1 block stream should start independently");

            t.Equals(remote(0x8100u, 0u, 0u), 0x00010400u,
                     "core 0 status should advance only the core 0 cursor");
            t.Equals(remote(0x8100u, 1u, 0u), 0x00020C00u,
                     "core 1 status should retain its independent pause position");
            t.Equals(remote(0x8100u, 0u, 0u), 0x01010800u,
                     "loop status should expose the second buffer in the high byte");

            t.Equals(remote(0x80E0u, 0u, 0x02u), 0x01010800u,
                     "block STOP should return the final core 0 cursor");
            t.Equals(remote(0x8100u, 0u, 0u), 0u,
                     "stopped block status should no longer expose a live cursor");
            t.Equals(remote(0x8100u, 1u, 0u), 0x01021000u,
                     "stopping core 0 should not stop or advance core 1");

            ps2_stubs::sceSdRemoteInit(rdram.data(), &initCtx, &runtime);
            t.Equals(remote(0x8100u, 1u, 0u), 0u,
                     "sceSdRemoteInit should reset block state for both cores");
            t.Equals(remote(0x80F0u, 1u, 0u), 1u,
                     "sceSdRemoteInit should restore idle voice status to complete");
        });
         
        tc.Run("IPU init skips missing optional helper instead of dispatching the default trap", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};
            ctx.pc = 0x0010B470u;

            ps2_stubs::sceIpuInit(rdram.data(), &ctx, &runtime);

            t.IsFalse(runtime.isStopRequested(),
                      "sceIpuInit should tolerate the missing optional SetD4 helper");
            t.Equals(runtime.memory().read32(0x10002010u), 0x40000000u,
                     "sceIpuInit should still program IPU_CTRL");
            t.Equals(runtime.memory().read32(0x10002000u), 0u,
                     "sceIpuInit should leave IPU_CMD reset after initialization");
        });

        tc.Run("sprintf consumes EE varargs from a2 a3 t0 and preserves width formatting", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};

            constexpr uint32_t kDestAddr = 0x00002000u;
            constexpr uint32_t kFormatAddr = 0x00002100u;
            constexpr char kFormat[] = "rm_%1d%02d%1d.rdx";

            std::memcpy(rdram.data() + kFormatAddr, kFormat, sizeof(kFormat));
            setRegU32(ctx, 4, kDestAddr);
            setRegU32(ctx, 5, kFormatAddr);
            setRegU32(ctx, 6, 0u); // a2
            setRegU32(ctx, 7, 3u); // a3
            setRegU32(ctx, 8, 1u); // t0

            ps2_stubs::sprintf(rdram.data(), &ctx, &runtime);

            const std::string rendered(reinterpret_cast<const char *>(rdram.data() + kDestAddr));
            t.Equals(rendered, std::string("rm_0031.rdx"),
                     "sprintf should read the third variadic integer from t0 and honor %02d");
            t.Equals(getRegS32(ctx, 2), static_cast<int32_t>(rendered.size()),
                     "sprintf should return the rendered length");
        });

        tc.Run("multiply-add matrix writes rd only when R5900 requires it", [](TestCase &t)
        {
            R5900Decoder decoder;
            CodeGenerator generator({}, {});

            const struct
            {
                const char *name;
                uint32_t raw;
                bool expectedRdWrite;
            } cases[] = {
                {"MULTU rd!=0", (OPCODE_SPECIAL << 26) | (2u << 21) | (3u << 16) | (11u << 11) | SPECIAL_MULTU, true},
                {"MMI MADD rd!=0", (OPCODE_MMI << 26) | (2u << 21) | (3u << 16) | (12u << 11) | MMI_MADD, true},
                {"MMI MADDU rd!=0", (OPCODE_MMI << 26) | (2u << 21) | (3u << 16) | (13u << 11) | MMI_MADDU, true},
                {"MMI MADD1 rd!=0", (OPCODE_MMI << 26) | (2u << 21) | (3u << 16) | (14u << 11) | MMI_MADD1, true},
                {"MMI MADDU1 rd!=0", (OPCODE_MMI << 26) | (2u << 21) | (3u << 16) | (15u << 11) | MMI_MADDU1, true},
                {"MMI DIVU1 rd!=0", (OPCODE_MMI << 26) | (2u << 21) | (3u << 16) | (16u << 11) | MMI_DIVU1, false},
            };

            for (size_t i = 0; i < std::size(cases); ++i)
            {
                const Instruction inst = decoder.decodeInstruction(0x2000u + static_cast<uint32_t>(i * 4u), cases[i].raw);
                const std::string generated = generator.translateInstruction(inst);
                const bool emittedRdWrite = hasSignedRdWrite(generated, inst.rd);

                t.Equals(inst.modificationInfo.modifiesGPR, cases[i].expectedRdWrite,
                         std::string("decoder rd-write metadata mismatch for ") + cases[i].name);
                t.Equals(emittedRdWrite, cases[i].expectedRdWrite,
                         std::string("codegen rd-write mismatch for ") + cases[i].name);
            }
        });

        tc.Run("SignalException marks EPC and BD for delay-slot exceptions", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};

            ctx.pc = 0x2000u;
            ctx.branch_pc = 0x1FFCu;
            ctx.in_delay_slot = true;
            ctx.cop0_status = 0u;
            ctx.cop0_cause = 0u;

            const bool raised = raisesGuestException([&]()
            {
                runtime.SignalException(&ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
            });

            t.IsTrue(raised, "exception delivery should unwind generated execution");
            t.Equals(ctx.cop0_epc, 0x1FFCu, "delay-slot exception should capture branch_pc in EPC");
            t.IsTrue((ctx.cop0_cause & COP0_CAUSE_BD) != 0u, "delay-slot exception should set CAUSE.BD");
            t.Equals(ctx.cop0_cause & COP0_CAUSE_EXCCODE_MASK,
                     (static_cast<uint32_t>(EXCEPTION_ADDRESS_ERROR_LOAD) << 2) & COP0_CAUSE_EXCCODE_MASK,
                     "CAUSE.EXCCODE should match exception");
            t.IsTrue((ctx.cop0_status & COP0_STATUS_EXL) != 0u, "exception should set STATUS.EXL");
            t.Equals(ctx.pc, EXCEPTION_VECTOR_GENERAL, "exception should jump to general vector when BEV=0");
            t.IsFalse(ctx.in_delay_slot, "exception delivery should clear delay-slot state");
        });

        tc.Run("SignalException uses current pc without BD and honors BEV vector", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};

            ctx.pc = 0x3000u;
            ctx.in_delay_slot = false;
            ctx.cop0_status = COP0_STATUS_BEV;
            ctx.cop0_cause = COP0_CAUSE_BD;

            const bool raised = raisesGuestException([&]()
            {
                runtime.SignalException(&ctx, EXCEPTION_ADDRESS_ERROR_STORE);
            });

            t.IsTrue(raised, "exception delivery should unwind generated execution");
            t.Equals(ctx.cop0_epc, 0x3000u, "non-delay exception should capture current pc in EPC");
            t.IsTrue((ctx.cop0_cause & COP0_CAUSE_BD) == 0u, "non-delay exception should clear CAUSE.BD");
            t.Equals(ctx.pc, EXCEPTION_VECTOR_BOOT, "BEV=1 should route exception to boot vector");
        });

        tc.Run("SignalException selects TLB vectors for load and store refills", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context loadCtx{};
            R5900Context storeCtx{};

            loadCtx.pc = 0x4000u;
            const bool loadRaised = raisesGuestException([&]()
            {
                runtime.SignalException(&loadCtx, EXCEPTION_TLB_REFILL_LOAD);
            });

            storeCtx.pc = 0x5000u;
            storeCtx.cop0_status = COP0_STATUS_BEV;
            const bool storeRaised = raisesGuestException([&]()
            {
                runtime.SignalException(&storeCtx, EXCEPTION_TLB_REFILL_STORE);
            });

            t.IsTrue(loadRaised, "TLB load refill should unwind generated execution");
            t.IsTrue(storeRaised, "TLB store refill should unwind generated execution");
            t.Equals(loadCtx.pc, EXCEPTION_VECTOR_TLB_REFILL,
                     "TLB load refill should use the normal refill vector");
            t.Equals(loadCtx.cop0_cause & COP0_CAUSE_EXCCODE_MASK,
                     (static_cast<uint32_t>(EXCEPTION_TLB_REFILL_LOAD) << 2) & COP0_CAUSE_EXCCODE_MASK,
                     "TLB load refill should report ExcCode 2");
            t.Equals(storeCtx.pc, EXCEPTION_VECTOR_TLB_REFILL_BOOT,
                     "TLB store refill with BEV=1 should use the bootstrap refill vector");
            t.Equals(storeCtx.cop0_cause & COP0_CAUSE_EXCCODE_MASK,
                     (static_cast<uint32_t>(EXCEPTION_TLB_REFILL_STORE) << 2) & COP0_CAUSE_EXCCODE_MASK,
                     "TLB store refill should report ExcCode 3");
        });

        tc.Run("SignalException preserves EPC and BD while EXL is already set", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context ctx{};

            ctx.pc = 0x6000u;
            ctx.branch_pc = 0x5FFCu;
            ctx.in_delay_slot = true;
            ctx.cop0_status = COP0_STATUS_EXL;
            ctx.cop0_epc = 0x12345678u;
            ctx.cop0_cause = COP0_CAUSE_BD | 0x00000100u;

            const bool raised = raisesGuestException([&]()
            {
                runtime.SignalException(&ctx, EXCEPTION_TLB_REFILL_LOAD);
            });

            t.IsTrue(raised, "nested exception should unwind generated execution");
            t.Equals(ctx.cop0_epc, 0x12345678u,
                     "nested level-1 exception should preserve the original EPC");
            t.IsTrue((ctx.cop0_cause & COP0_CAUSE_BD) != 0u,
                     "nested level-1 exception should preserve the original BD bit");
            t.Equals(ctx.cop0_cause & COP0_CAUSE_EXCCODE_MASK,
                     (static_cast<uint32_t>(EXCEPTION_TLB_REFILL_LOAD) << 2) & COP0_CAUSE_EXCCODE_MASK,
                     "nested exception should still update CAUSE.EXCCODE");
            t.Equals(ctx.pc, EXCEPTION_VECTOR_GENERAL,
                     "TLB refill while EXL=1 should use the common vector");
            t.IsFalse(ctx.in_delay_slot, "exception delivery should clear runtime delay-slot state");
        });

        tc.Run("runtime memory faults populate BadVAddr and TLB metadata", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "PS2Memory initialize should succeed");
            uint8_t *const rdram = runtime.memory().getRDRAM();

            R5900Context addressErrorCtx{};
            addressErrorCtx.pc = 0x7000u;
            const bool addressErrorRaised = raisesGuestException([&]()
            {
                (void)runtime.Load32(rdram, &addressErrorCtx, 0x00000002u);
            });

            t.IsTrue(addressErrorRaised, "faulting load should unwind generated execution");
            t.Equals(addressErrorCtx.cop0_badvaddr, 0x00000002u,
                     "address error should record the faulting virtual address");
            t.Equals(addressErrorCtx.cop0_cause & COP0_CAUSE_EXCCODE_MASK,
                     (static_cast<uint32_t>(EXCEPTION_ADDRESS_ERROR_LOAD) << 2) & COP0_CAUSE_EXCCODE_MASK,
                     "unaligned load should report an address error");
            t.Equals(addressErrorCtx.pc, EXCEPTION_VECTOR_GENERAL,
                     "address error should use the common vector");

            constexpr uint32_t loadMissAddress = 0xC1234000u;
            R5900Context loadMissCtx{};
            loadMissCtx.pc = 0x8000u;
            loadMissCtx.cop0_context = 0xAB80000Fu;
            loadMissCtx.cop0_entryhi = 0x000005A5u;
            const bool loadMissRaised = raisesGuestException([&]()
            {
                (void)runtime.Load32(rdram, &loadMissCtx, loadMissAddress);
            });

            t.IsTrue(loadMissRaised, "TLB-missing load should unwind generated execution");
            t.Equals(loadMissCtx.cop0_badvaddr, loadMissAddress,
                     "TLB load refill should record BadVAddr");
            t.Equals(loadMissCtx.cop0_context,
                     (0xAB80000Fu & 0xFF80000Fu) | ((loadMissAddress >> 9u) & 0x007FFFF0u),
                     "TLB load refill should update Context.BadVPN2");
            t.Equals(loadMissCtx.cop0_entryhi,
                     (loadMissAddress & 0xFFFFE000u) | (0x000005A5u & 0x00001FFFu),
                     "TLB load refill should update EntryHi.VPN2 and preserve ASID");
            t.Equals(loadMissCtx.cop0_cause & COP0_CAUSE_EXCCODE_MASK,
                     (static_cast<uint32_t>(EXCEPTION_TLB_REFILL_LOAD) << 2) & COP0_CAUSE_EXCCODE_MASK,
                     "TLB-missing load should report ExcCode 2");
            t.Equals(loadMissCtx.pc, EXCEPTION_VECTOR_TLB_REFILL,
                     "TLB-missing load should use the refill vector");

            constexpr uint32_t storeMissAddress = 0xC5678000u;
            R5900Context storeMissCtx{};
            storeMissCtx.pc = 0x9000u;
            storeMissCtx.cop0_status = COP0_STATUS_BEV;
            const bool storeMissRaised = raisesGuestException([&]()
            {
                runtime.Store32(rdram, &storeMissCtx, storeMissAddress, 0x11223344u);
            });

            t.IsTrue(storeMissRaised, "TLB-missing store should unwind generated execution");
            t.Equals(storeMissCtx.cop0_badvaddr, storeMissAddress,
                     "TLB store refill should record BadVAddr");
            t.Equals(storeMissCtx.cop0_cause & COP0_CAUSE_EXCCODE_MASK,
                     (static_cast<uint32_t>(EXCEPTION_TLB_REFILL_STORE) << 2) & COP0_CAUSE_EXCCODE_MASK,
                     "TLB-missing store should report ExcCode 3");
            t.Equals(storeMissCtx.pc, EXCEPTION_VECTOR_TLB_REFILL_BOOT,
                     "TLB-missing store with BEV=1 should use the bootstrap refill vector");
        });

        tc.Run("guest exceptions unwind nested generated calls and resume at the vector", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};

            t.IsTrue(runtime.registerFunction(kExceptionUnwindEntry,
                                              &testExceptionUnwindEntryFunction),
                     "entry function registration should succeed");
            t.IsTrue(runtime.registerFunction(kExceptionUnwindNested,
                                              &testExceptionUnwindNested),
                     "nested function registration should succeed");
            t.IsTrue(runtime.registerFunction(kExceptionUnwindVector,
                                              &testExceptionUnwindVectorHandler),
                     "exception vector registration should succeed");

            setRegU32(ctx, 2, 0x22222222u);
            ctx.pc = kExceptionUnwindEntry;
            runtime.dispatchLoop(rdram.data(), &ctx);

            t.Equals(::getRegU32(&ctx, 2), 0x22222222u,
                     "faulting load must not overwrite its destination");
            t.Equals(::getRegU32(&ctx, 3), 0u,
                     "nested generated code after the fault must not execute");
            t.Equals(::getRegU32(&ctx, 4), 0u,
                     "calling generated code after the fault must not execute");
            t.Equals(::getRegU32(&ctx, 5), 0x55555555u,
                     "dispatcher should execute the exception vector");
            t.Equals(ctx.cop0_epc, kExceptionUnwindNested,
                     "EPC should identify the faulting generated function");
            t.Equals(ctx.cop0_badvaddr, 0x00000002u,
                     "BadVAddr should identify the faulting load address");
            t.Equals(ctx.cop0_cause & COP0_CAUSE_EXCCODE_MASK,
                     (static_cast<uint32_t>(EXCEPTION_ADDRESS_ERROR_LOAD) << 2) &
                         COP0_CAUSE_EXCCODE_MASK,
                     "fault should retain the address-error cause at the handler");
        });

        tc.Run("handleSyscall rejects invocation in delay slot", [](TestCase &t)
        {
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};
            ctx.in_delay_slot = true;

            bool threw = false;
            try
            {
                runtime.handleSyscall(rdram.data(), &ctx, 0x3Cu);
            }
            catch (const std::runtime_error &)
            {
                threw = true;
            }

            t.IsTrue(threw, "syscall from delay slot should throw to preserve block atomicity");
        });

        tc.Run("VIF MSCAL and MSCNT toggle DBF and keep TOPS/ITOPS coherent", [](TestCase &t)
        {
            PS2Memory mem;
            t.IsTrue(mem.initialize(), "PS2Memory initialize should succeed");

            mem.vif1_regs.base = 4u;
            mem.vif1_regs.ofst = 2u;
            mem.vif1_regs.tops = 4u;
            mem.vif1_regs.itops = 0x21u;
            mem.vif1_regs.stat &= ~(1u << 7); // DBF = 0

            uint32_t callbackPc = 0xFFFFFFFFu;
            uint32_t callbackTop = 0xFFFFFFFFu;
            uint32_t callbackItop = 0xFFFFFFFFu;
            uint32_t callbackCount = 0u;
            mem.setVu1MscalCallback([&](uint32_t startPC, uint32_t top, uint32_t itop)
            {
                callbackPc = startPC;
                callbackTop = top;
                callbackItop = itop;
                ++callbackCount;
            });

            const uint32_t mscal = makeVifCmd(0x14u, 0u, 3u); // start PC = 3 * 8
            mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&mscal), sizeof(mscal));

            t.Equals(callbackCount, 1u, "MSCAL should invoke VU1 callback exactly once");
            t.Equals(callbackPc, 24u, "MSCAL should pass startPC=imm*8");
            t.Equals(callbackTop, 4u, "MSCAL callback should receive current TOPS");
            t.Equals(callbackItop, 0x21u, "MSCAL callback should receive pending ITOPS");
            t.Equals(mem.vif1_regs.top, 4u, "MSCAL should latch TOP from TOPS");
            t.Equals(mem.vif1_regs.itop, 0x21u, "MSCAL should latch ITOP from ITOPS");
            t.IsTrue((mem.vif1_regs.stat & (1u << 7)) != 0u, "MSCAL should toggle DBF on");
            t.Equals(mem.vif1_regs.tops, 6u, "DBF=1 should make TOPS=BASE+OFST");

            const uint32_t mscnt = makeVifCmd(0x17u, 0u, 0u);
            mem.processVIF1Data(reinterpret_cast<const uint8_t *>(&mscnt), sizeof(mscnt));

            t.Equals(callbackCount, 1u, "MSCNT should not invoke MSCAL callback");
            t.IsTrue((mem.vif1_regs.stat & (1u << 7)) == 0u, "MSCNT should toggle DBF back off");
            t.Equals(mem.vif1_regs.tops, 4u, "DBF=0 should make TOPS=BASE");
            t.Equals(mem.vif1_regs.top, 6u, "MSCNT should latch TOP from current TOPS before toggling");
            t.Equals(mem.vif1_regs.itop, 0x21u, "MSCNT should keep latching ITOP from ITOPS");
        });

        tc.Run("event VU1 MSCAL stays busy until scheduled finish wakes VIF1", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "VU1 timing fixture memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "VU1 timing fixture subsystems should bind");

            constexpr uint32_t kVuUpperNop = 0x000002FFu;
            constexpr uint32_t kVuUpperEnd = 0x400002FFu;
            uint8_t *const code = runtime.memory().getVU1Code();
            std::memset(code, 0, PS2_VU1_CODE_SIZE);
            writeVuInstructionPair(code, 0u, 0u, kVuUpperEnd);
            writeVuInstructionPair(code, 8u, 0u, kVuUpperNop);
            runtime.memory().markVU1CodeModified();

            const std::array<uint32_t, 3> commands = {
                makeVifCmd(0x14u, 0u, 0u), // MSCAL
                makeVifCmd(0x11u, 0u, 0u), // FLUSH
                makeVifCmd(0x05u, 0u, 3u), // STMOD
            };
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(commands.data()),
                static_cast<uint32_t>(sizeof(commands)));

            PS2Runtime::DebugVu1Timing timing =
                runtime.debugVu1TimingSnapshot();
            t.IsTrue(timing.active,
                     "MSCAL should make VU1 active without completing it");
            t.Equals(timing.totalAdvancedCycles, 0u,
                     "MSCAL submission should execute zero VU cycles");
            t.IsTrue(timing.eventPending,
                     "MSCAL should schedule VIF_VU1_FINISH");
            t.Equals(timing.eventDeadlineTick, 128u,
                     "the first progress boundary should be sixteen VU cycles");
            t.IsTrue(timing.vifWaitingForVu,
                     "a following FLUSH should retain explicit VIF wait state");
            t.IsTrue(
                (runtime.memory().vif1_regs.stat & (1u << 2u)) != 0u,
                "VIF wait should expose VIF1_STAT.VEW");
            t.Equals(runtime.memory().vif1_regs.mode, 0u,
                     "commands after the wait must remain deferred");
            t.IsTrue(
                (runtime.cpu().vu0_vpu_stat & (1u << 8u)) != 0u,
                "VPU_STAT should expose VU1 busy before completion");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(120u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            timing = runtime.debugVu1TimingSnapshot();
            t.IsTrue(timing.active,
                     "VU1 should remain busy immediately before its deadline");
            t.Equals(timing.totalAdvancedCycles, 0u,
                     "pre-deadline service should not advance VU1");

            context.advanceEeCycleTicks(8u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            timing = runtime.debugVu1TimingSnapshot();
            t.IsFalse(timing.active,
                      "the E-bit program should finish at the scheduled boundary");
            t.Equals(timing.totalAdvancedCycles, 2u,
                     "finish service should report the exact executed pairs");
            t.IsFalse(timing.eventPending,
                      "completed VU1 work should leave no progress event");
            t.IsFalse(timing.vifWaitingForVu,
                      "completion should clear retained VIF wait state");
            t.IsTrue(
                (runtime.memory().vif1_regs.stat & (1u << 2u)) == 0u,
                "completion should clear VIF1_STAT.VEW");
            t.Equals(runtime.memory().vif1_regs.mode, 3u,
                     "completion should resume the deferred VIF command");
            t.IsTrue(
                (runtime.cpu().vu0_vpu_stat & (1u << 8u)) == 0u,
                "completion should clear VU1 busy");

            const uint32_t mscalf = makeVifCmd(0x15u, 0u, 0u);
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(&mscalf),
                sizeof(mscalf));
            timing = runtime.debugVu1TimingSnapshot();
            t.IsTrue(timing.active,
                     "MSCALF should use the same cooperative start contract");
            t.Equals(timing.totalAdvancedCycles, 0u,
                     "MSCALF should not execute synchronously");
            (void)runtime.memory().writeIORegister(
                0x10003C10u, 1u);
        });

        tc.Run("event SPR_TO normal DMA retains busy state for its reference bus cost", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "SPR_TO timing fixture memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "SPR_TO timing fixture subsystems should bind");

            constexpr uint32_t kSprTo = 0x1000D400u;
            constexpr uint32_t kDstat = 0x1000E010u;
            constexpr uint32_t kSource = 0x0002A000u;
            constexpr uint32_t kScratch = 0x2800u;
            constexpr uint32_t kQwc = 118u;
            constexpr uint64_t kCompletionTick =
                static_cast<uint64_t>(kQwc) * 2u *
                ps2x::timing::kEeTicksPerCycle;

            for (uint32_t byte = 0u;
                 byte < kQwc * 16u; ++byte)
            {
                runtime.memory().getRDRAM()[kSource + byte] =
                    static_cast<uint8_t>(
                        (0x31u + byte * 7u) & 0xFFu);
            }

            runtime.debugStartEeEventTrace(4u);
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo + 0x10u, kSource),
                "SPR_TO MADR write should succeed");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo + 0x20u, kQwc),
                "SPR_TO QWC write should succeed");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo + 0x80u, kScratch),
                "SPR_TO SADR write should succeed");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo, 0x100u),
                "SPR_TO normal start should succeed");

            const ScratchpadDmaSnapshot submitted =
                runtime.memory().scratchpadDmaSnapshot(
                    DmacChannel::ToScratchpad);
            t.IsTrue(
                submitted.active && submitted.eventManaged,
                "submission should retain an event-owned SPR_TO operation");
            t.IsTrue(
                submitted.phase ==
                    ScratchpadDmaPhase::Finalize,
                "the bounded payload slice should await timed finalization");
            t.Equals(
                submitted.madr, kSource + kQwc * 16u,
                "submission should expose transferred MADR");
            t.Equals(
                submitted.qwc, 0u,
                "submission should expose the consumed payload QWC");
            t.IsTrue(
                (runtime.memory().readIORegister(kSprTo) &
                 0x100u) != 0u,
                "SPR_TO should expose CHCR.STR while timing remains");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 (1u << 9u)) == 0u,
                "SPR_TO submission must not publish completion");

            bool copied = true;
            for (uint32_t byte = 0u;
                 byte < kQwc * 16u; ++byte)
            {
                copied &=
                    runtime.memory().getScratchpad()[
                        kScratch + byte] ==
                    static_cast<uint8_t>(
                        (0x31u + byte * 7u) & 0xFFu);
            }
            t.IsTrue(
                copied,
                "SPR_TO should copy its first bounded slice at submission");

            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            const auto &submittedSlot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacToScratchpad)];
            t.IsTrue(
                submittedSlot.pending,
                "SPR_TO should retain a timed device event");
            t.Equals(
                submittedSlot.deadlineTick,
                kCompletionTick,
                "SPR_TO should cost two EE cycles per QWC");
            t.IsTrue(
                submittedSlot.device.kind ==
                    PS2Runtime::DebugEeEventDeviceKind::
                        ScratchpadDma,
                "scheduler status should identify the event-owned device");
            t.IsTrue(
                submittedSlot.device.active,
                "scheduler status should retain active device state");
            t.Equals(
                submittedSlot.device.operationGeneration,
                submitted.transfer.generation,
                "scheduler status should correlate slot and operation generations");
            t.Equals(
                submittedSlot.device.madr,
                submitted.madr,
                "scheduler status should expose current device registers");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(
                static_cast<uint32_t>(
                    kCompletionTick -
                    ps2x::timing::kEeTicksPerCycle));
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsTrue(
                runtime.memory()
                    .scratchpadDmaSnapshot(
                        DmacChannel::ToScratchpad)
                    .active,
                "SPR_TO should remain active immediately before its deadline");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 (1u << 9u)) == 0u,
                "pre-deadline service should keep D_STAT clear");

            context.advanceEeCycleTicks(
                ps2x::timing::kEeTicksPerCycle);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(
                runtime.memory()
                    .scratchpadDmaSnapshot(
                        DmacChannel::ToScratchpad)
                    .active,
                "SPR_TO should retire at the reference deadline");
            t.IsTrue(
                (runtime.memory().readIORegister(kSprTo) &
                 0x100u) == 0u,
                "completion publication should clear SPR_TO STR");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 (1u << 9u)) != 0u,
                "completion publication should latch SPR_TO D_STAT");

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(2u),
                "SPR_TO should trace device finalization and publication");
            if (trace.entries.size() == 2u)
            {
                t.IsTrue(
                    trace.entries[0].source ==
                            ps2x::timing::EeEventSource::
                                DmacToScratchpad &&
                        trace.entries[1].source ==
                            ps2x::timing::EeEventSource::
                                DmacCompletion,
                    "SPR_TO state must finalize before completion publication");
                t.Equals(
                    trace.entries[0].scheduledTick,
                    kCompletionTick,
                    "SPR_TO trace should retain its exact deadline");
            }
        });

        tc.Run("equal-tick SPR events service from-memory before to-memory", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "equal-tick SPR fixture memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "equal-tick SPR fixture subsystems should bind");

            constexpr uint32_t kSprFrom = 0x1000D000u;
            constexpr uint32_t kSprTo = 0x1000D400u;
            constexpr uint32_t kDstat = 0x1000E010u;
            constexpr uint32_t kFromDestination =
                0x0002B000u;
            constexpr uint32_t kToSource = 0x0002C000u;
            constexpr uint32_t kFromScratch = 0x1000u;
            constexpr uint32_t kToScratch = 0x2000u;
            constexpr uint32_t kQwc = 2u;

            for (uint32_t byte = 0u;
                 byte < kQwc * 16u; ++byte)
            {
                runtime.memory().getScratchpad()[
                    kFromScratch + byte] =
                    static_cast<uint8_t>(0x80u + byte);
                runtime.memory().getRDRAM()[
                    kToSource + byte] =
                    static_cast<uint8_t>(0x40u + byte);
            }

            runtime.debugStartEeEventTrace(8u);
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo + 0x10u, kToSource),
                "equal-tick SPR_TO MADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo + 0x20u, kQwc),
                "equal-tick SPR_TO QWC should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo + 0x80u, kToScratch),
                "equal-tick SPR_TO SADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo, 0x100u),
                "equal-tick SPR_TO should start");

            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprFrom + 0x10u,
                    kFromDestination),
                "equal-tick SPR_FROM MADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprFrom + 0x20u, kQwc),
                "equal-tick SPR_FROM QWC should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprFrom + 0x80u, kFromScratch),
                "equal-tick SPR_FROM SADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprFrom, 0x100u),
                "equal-tick SPR_FROM should start");

            const PS2Runtime::DebugEeScheduler submitted =
                runtime.debugEeSchedulerSnapshot();
            const auto &fromSlot =
                submitted.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacFromScratchpad)];
            const auto &toSlot =
                submitted.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacToScratchpad)];
            t.IsTrue(
                fromSlot.pending && toSlot.pending,
                "both SPR directions should retain events");
            t.Equals(
                fromSlot.deadlineTick, 32ull,
                "SPR_FROM should use the two-cycle bus cost");
            t.Equals(
                toSlot.deadlineTick, 32ull,
                "SPR_TO should use the two-cycle bus cost");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(32u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);

            bool copied = true;
            for (uint32_t byte = 0u;
                 byte < kQwc * 16u; ++byte)
            {
                copied &=
                    runtime.memory().getRDRAM()[
                        kFromDestination + byte] ==
                    static_cast<uint8_t>(0x80u + byte);
                copied &=
                    runtime.memory().getScratchpad()[
                        kToScratch + byte] ==
                    static_cast<uint8_t>(0x40u + byte);
            }
            t.IsTrue(
                copied,
                "both equal-tick SPR payloads should be visible");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 ((1u << 8u) | (1u << 9u))) ==
                    ((1u << 8u) | (1u << 9u)),
                "both SPR causes should publish in the same batch");

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(4u),
                "each SPR device event should precede its publication");
            if (trace.entries.size() == 4u)
            {
                t.IsTrue(
                    trace.entries[0].source ==
                            ps2x::timing::EeEventSource::
                                DmacFromScratchpad &&
                        trace.entries[1].source ==
                            ps2x::timing::EeEventSource::
                                DmacCompletion &&
                        trace.entries[2].source ==
                            ps2x::timing::EeEventSource::
                                DmacToScratchpad &&
                        trace.entries[3].source ==
                            ps2x::timing::EeEventSource::
                                DmacCompletion,
                    "fixed priority should service SPR_FROM before SPR_TO");
            }
        });

        tc.Run("event SPR_TO chain retains tag progress between deadlines", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "SPR_TO chain fixture memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "SPR_TO chain fixture subsystems should bind");

            constexpr uint32_t kSprTo = 0x1000D400u;
            constexpr uint32_t kDstat = 0x1000E010u;
            constexpr uint32_t kTag = 0x0002D000u;
            constexpr uint32_t kScratch = 0x0300u;
            uint8_t *const rdram =
                runtime.memory().getRDRAM();
            const uint64_t cnt =
                makeDmaTag(1u, 1u, 0u);
            const uint64_t end =
                makeDmaTag(1u, 7u, 0u);
            std::memcpy(
                rdram + kTag, &cnt, sizeof(cnt));
            std::memcpy(
                rdram + kTag + 32u,
                &end, sizeof(end));
            for (uint32_t byte = 0u; byte < 16u;
                 ++byte)
            {
                rdram[kTag + 16u + byte] =
                    static_cast<uint8_t>(0x20u + byte);
                rdram[kTag + 48u + byte] =
                    static_cast<uint8_t>(0xA0u + byte);
            }

            runtime.debugStartEeEventTrace(8u);
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo + 0x30u, kTag),
                "SPR_TO chain TADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo + 0x80u, kScratch),
                "SPR_TO chain SADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo, 0x104u),
                "SPR_TO chain should start");

            ScratchpadDmaSnapshot dma =
                runtime.memory().scratchpadDmaSnapshot(
                    DmacChannel::ToScratchpad);
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.phase ==
                        ScratchpadDmaPhase::FetchTag,
                "submission should retain the next tag after CNT payload");
            t.Equals(
                dma.tagsProcessed, 1u,
                "submission should process exactly one source-chain tag");
            t.Equals(
                dma.tadr, kTag + 32u,
                "CNT progress should expose the next tag address");
            t.Equals(
                dma.sadr, kScratch + 16u,
                "CNT progress should expose scratchpad advancement");

            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            t.Equals(
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacToScratchpad)]
                    .deadlineTick,
                16ull,
                "one CNT payload QWC should cost two EE cycles");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(16u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            dma = runtime.memory().scratchpadDmaSnapshot(
                DmacChannel::ToScratchpad);
            t.IsTrue(
                dma.active &&
                    dma.phase ==
                        ScratchpadDmaPhase::Finalize,
                "the END payload should leave finalization pending");
            t.Equals(
                dma.tagsProcessed, 2u,
                "the first deadline should process only the END tag");
            t.Equals(
                dma.tadr, kTag + 32u,
                "END should retain its terminal tag address");
            t.Equals(
                dma.madr, kTag + 64u,
                "END payload should advance MADR");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 (1u << 9u)) == 0u,
                "terminal payload progress should not yet publish completion");

            scheduler = runtime.debugEeSchedulerSnapshot();
            t.Equals(
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacToScratchpad)]
                    .deadlineTick,
                32ull,
                "the END payload should retain a later finalization event");

            context.advanceEeCycleTicks(16u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            t.IsFalse(
                runtime.memory()
                    .scratchpadDmaSnapshot(
                        DmacChannel::ToScratchpad)
                    .active,
                "the final event should retire the chain");
            t.IsTrue(
                (runtime.memory().readIORegister(kSprTo) &
                 0x100u) == 0u,
                "chain publication should clear SPR_TO STR");

            bool copied = true;
            for (uint32_t byte = 0u; byte < 16u;
                 ++byte)
            {
                copied &=
                    runtime.memory().getScratchpad()[
                        kScratch + byte] ==
                    static_cast<uint8_t>(0x20u + byte);
                copied &=
                    runtime.memory().getScratchpad()[
                        kScratch + 16u + byte] ==
                    static_cast<uint8_t>(0xA0u + byte);
            }
            t.IsTrue(
                copied,
                "scheduled chain progress should preserve both payloads");

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(3u),
                "chain should trace END progress, finalization, and publication");
            if (trace.entries.size() == 3u)
            {
                t.IsTrue(
                    trace.entries[0].source ==
                            ps2x::timing::EeEventSource::
                                DmacToScratchpad &&
                        trace.entries[1].source ==
                            ps2x::timing::EeEventSource::
                                DmacToScratchpad &&
                        trace.entries[2].source ==
                            ps2x::timing::EeEventSource::
                                DmacCompletion,
                    "chain finalization should retain typed event ownership");
            }
        });

        tc.Run("event SPR_TO cancel and restart reject the stale generation", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "SPR_TO restart fixture memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "SPR_TO restart fixture subsystems should bind");

            constexpr uint32_t kSprTo = 0x1000D400u;
            constexpr uint32_t kDstat = 0x1000E010u;
            constexpr uint32_t kFirstSource =
                0x0002E000u;
            constexpr uint32_t kSecondSource =
                0x0002F000u;
            constexpr uint32_t kFirstQwc = 4u;
            constexpr uint32_t kSecondQwc = 8u;
            std::memset(
                runtime.memory().getRDRAM() + kFirstSource,
                0x31, kFirstQwc * 16u);
            std::memset(
                runtime.memory().getRDRAM() + kSecondSource,
                0x72, kSecondQwc * 16u);

            runtime.debugStartEeEventTrace(4u);
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo + 0x10u, kFirstSource),
                "first SPR_TO MADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo + 0x20u, kFirstQwc),
                "first SPR_TO QWC should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo + 0x80u, 0x0400u),
                "first SPR_TO SADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo, 0x100u),
                "first SPR_TO should start");

            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            const auto first =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacToScratchpad)];
            t.IsTrue(
                first.pending,
                "the first SPR_TO generation should be pending");
            t.Equals(
                first.deadlineTick, 64ull,
                "four QWC should target tick 64");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(8u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo, 0u),
                "clearing SPR_TO STR should cancel the first generation");
            t.IsFalse(
                runtime.memory()
                    .scratchpadDmaSnapshot(
                        DmacChannel::ToScratchpad)
                    .active,
                "cancellation should retire the first operation");

            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo + 0x10u, kSecondSource),
                "replacement SPR_TO MADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo + 0x20u, kSecondQwc),
                "replacement SPR_TO QWC should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo + 0x80u, 0x0800u),
                "replacement SPR_TO SADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kSprTo, 0x100u),
                "replacement SPR_TO should start");

            scheduler = runtime.debugEeSchedulerSnapshot();
            const auto second =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacToScratchpad)];
            t.IsTrue(
                second.pending &&
                    second.generation > first.generation,
                "replacement should own a newer event generation");
            t.Equals(
                second.deadlineTick, 136ull,
                "replacement should schedule from committed tick 8");

            context.advanceEeCycleTicks(56u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            const ScratchpadDmaSnapshot beforeReplacement =
                runtime.memory().scratchpadDmaSnapshot(
                    DmacChannel::ToScratchpad);
            t.IsTrue(
                beforeReplacement.active &&
                    beforeReplacement.phase ==
                        ScratchpadDmaPhase::Finalize,
                "the stale tick-64 deadline must not finalize replacement work");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 (1u << 9u)) == 0u,
                "the stale generation must not publish D_STAT");
            t.Equals(
                runtime.debugEeEventTraceSnapshot(false)
                    .entries.size(),
                static_cast<size_t>(0u),
                "the cancelled deadline must not dispatch");

            context.advanceEeCycleTicks(72u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(
                runtime.memory()
                    .scratchpadDmaSnapshot(
                        DmacChannel::ToScratchpad)
                    .active,
                "the replacement should finalize at tick 136");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 (1u << 9u)) != 0u,
                "only the replacement should publish completion");

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(2u),
                "only replacement finalization and publication should dispatch");
            if (!trace.entries.empty())
            {
                t.Equals(
                    trace.entries.front().serviceTick,
                    136ull,
                    "the replacement should own the first live service");
            }
        });

        tc.Run("event GIF normal DMA reproduces PCSX2 slice deadlines", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "GIF normal fixture memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "GIF normal fixture subsystems should bind");

            constexpr uint32_t kGif = 0x1000A000u;
            constexpr uint32_t kDstat = 0x1000E010u;
            constexpr uint32_t kSource = 0x00034000u;
            constexpr uint32_t kQwc = 12u;
            writeGifImagePacket(
                runtime.memory().getRDRAM() + kSource,
                kQwc, 0x31u);

            std::vector<std::vector<uint8_t>> captured;
            runtime.gifArbiter().setProcessPacketFn(
                [&](const uint8_t *data,
                    uint32_t sizeBytes)
                {
                    captured.emplace_back(
                        data, data + sizeBytes);
                });

            runtime.debugStartEeEventTrace(8u);
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGif + 0x10u, kSource),
                "GIF normal MADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGif + 0x20u, kQwc),
                "GIF normal QWC should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGif, 0x100u),
                "GIF normal DMA should start");

            GifDmaSnapshot dma =
                runtime.memory().gifDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.phase ==
                        GifDmaPhase::TransferPayload,
                "submission should retain scheduled payload work");
            t.Equals(
                dma.qwc, 8u,
                "submission should consume the initial four QW");
            t.Equals(
                dma.madr, kSource + 4u * 16u,
                "submission should expose the initial MADR advance");
            t.Equals(
                captured.size(), static_cast<size_t>(0u),
                "the partial GIF packet should remain buffered");

            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            auto slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacGif)];
            t.IsTrue(
                slot.pending,
                "submission should retain DMAC_GIF ownership");
            t.Equals(
                slot.deadlineTick, 64ull,
                "four initial QW should cost eight EE cycles");
            t.IsTrue(
                slot.device.kind ==
                    PS2Runtime::DebugEeEventDeviceKind::
                        GifDma,
                "scheduler status should identify GIF DMA");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(64u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            dma = runtime.memory().gifDmaSnapshot();
            t.IsTrue(
                dma.active &&
                    dma.phase == GifDmaPhase::Finalize,
                "the first service should retain finalization");
            t.Equals(
                dma.qwc, 0u,
                "the first service should consume the remaining QW");
            t.Equals(
                captured.size(), static_cast<size_t>(1u),
                "the remaining slice should complete one GIF packet");
            scheduler = runtime.debugEeSchedulerSnapshot();
            slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacGif)];
            t.IsTrue(
                slot.pending,
                "payload service should retain finalization ownership");
            t.Equals(
                slot.deadlineTick, 192ull,
                "eight remaining QW should cost sixteen cycles from service");
            t.IsTrue(
                (runtime.memory().readIORegister(kGif) &
                 0x100u) != 0u,
                "STR should remain visible before finalization");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 (1u << 2u)) == 0u,
                "D_STAT should remain clear before finalization");

            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(
                runtime.memory().gifDmaSnapshot().active,
                "the second service should retire GIF DMA");
            t.IsTrue(
                (runtime.memory().readIORegister(kGif) &
                 0x100u) == 0u,
                "same-boundary publication should clear STR");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 (1u << 2u)) != 0u,
                "same-boundary publication should latch GIF D_STAT");

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(3u),
                "normal DMA should trace two services and publication");
            if (trace.entries.size() == 3u)
            {
                const std::array<
                    ps2x::timing::EeEventSource, 3u>
                    expectedSources = {
                        ps2x::timing::EeEventSource::DmacGif,
                        ps2x::timing::EeEventSource::DmacGif,
                        ps2x::timing::EeEventSource::
                            DmacCompletion,
                    };
                const std::array<uint64_t, 3u>
                    expectedTicks = {
                        64ull, 192ull, 192ull};
                for (size_t index = 0u;
                     index < trace.entries.size(); ++index)
                {
                    t.IsTrue(
                        trace.entries[index].source ==
                            expectedSources[index],
                        "normal GIF services should retain source order");
                    t.Equals(
                        trace.entries[index].serviceTick,
                        expectedTicks[index],
                        "normal GIF services should retain exact ticks");
                }
            }
        });

        tc.Run("event GIF END chain combines tag and initial slice cost", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "GIF chain fixture memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "GIF chain fixture subsystems should bind");

            constexpr uint32_t kGif = 0x1000A000u;
            constexpr uint32_t kTag = 0x00034400u;
            constexpr uint32_t kQwc = 12u;
            uint8_t *const rdram =
                runtime.memory().getRDRAM();
            std::memset(rdram + kTag, 0, 16u);
            const uint64_t end =
                makeDmaTag(kQwc, 7u, 0u);
            std::memcpy(rdram + kTag, &end, sizeof(end));
            writeGifImagePacket(
                rdram + kTag + 16u, kQwc, 0x42u);

            std::vector<std::vector<uint8_t>> captured;
            runtime.gifArbiter().setProcessPacketFn(
                [&](const uint8_t *data,
                    uint32_t sizeBytes)
                {
                    captured.emplace_back(
                        data, data + sizeBytes);
                });

            runtime.debugStartEeEventTrace(8u);
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGif + 0x30u, kTag),
                "GIF chain TADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGif, 0x104u),
                "GIF END chain should start");

            GifDmaSnapshot dma =
                runtime.memory().gifDmaSnapshot();
            t.Equals(
                dma.tagsProcessed, 1u,
                "submission should fetch the END tag");
            t.Equals(
                dma.qwc, 8u,
                "submission should retain eight payload QW");
            t.Equals(
                dma.madr, kTag + 16u + 4u * 16u,
                "submission should expose payload MADR progress");
            t.Equals(
                dma.tadr, kTag,
                "END should retain the terminal tag address");

            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            auto slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacGif)];
            t.IsTrue(
                slot.pending,
                "END submission should retain DMAC_GIF");
            t.Equals(
                slot.deadlineTick, 80ull,
                "tag plus four QW should cost ten EE cycles");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(80u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            dma = runtime.memory().gifDmaSnapshot();
            t.IsTrue(
                dma.active &&
                    dma.phase == GifDmaPhase::Finalize,
                "payload service should retain finalization");
            scheduler = runtime.debugEeSchedulerSnapshot();
            slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacGif)];
            t.Equals(
                slot.deadlineTick, 208ull,
                "remaining payload should cost sixteen cycles");

            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            t.Equals(
                runtime.memory().readIORegister(kGif),
                0x70000004u,
                "completion should retain END and chain-mode bits");
            t.Equals(
                runtime.memory().readIORegister(
                    kGif + 0x10u),
                kTag + 16u + kQwc * 16u,
                "completion should retain final MADR");
            t.Equals(
                runtime.memory().readIORegister(
                    kGif + 0x30u),
                kTag,
                "completion should retain terminal TADR");
            t.Equals(
                captured.size(), static_cast<size_t>(1u),
                "chain progress should submit one complete GIF packet");

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(3u),
                "END chain should trace two services and publication");
            if (trace.entries.size() == 3u)
            {
                const std::array<uint64_t, 3u>
                    expectedTicks = {
                        80ull, 208ull, 208ull};
                t.IsTrue(
                    trace.entries[0].source ==
                            ps2x::timing::EeEventSource::
                                DmacGif &&
                        trace.entries[1].source ==
                            ps2x::timing::EeEventSource::
                                DmacGif &&
                        trace.entries[2].source ==
                            ps2x::timing::EeEventSource::
                                DmacCompletion,
                    "END completion should follow GIF progress");
                for (size_t index = 0u;
                     index < trace.entries.size(); ++index)
                {
                    t.Equals(
                        trace.entries[index].serviceTick,
                        expectedTicks[index],
                        "END services should retain exact ticks");
                }
            }
        });

        tc.Run("event GIF MODE unmask owns the full FIFO wakeup", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "GIF mask fixture memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "GIF mask fixture subsystems should bind");

            constexpr uint32_t kGifCtrl = 0x10003000u;
            constexpr uint32_t kGifMode = 0x10003010u;
            constexpr uint32_t kGifStat = 0x10003020u;
            constexpr uint32_t kGif = 0x1000A000u;
            constexpr uint32_t kDstat = 0x1000E010u;
            constexpr uint32_t kSource = 0x00034800u;
            constexpr uint32_t kQwc = 16u;
            writeGifImagePacket(
                runtime.memory().getRDRAM() + kSource,
                kQwc, 0x53u);

            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGifCtrl, 1u),
                "GIF reset should establish an empty parser");
            std::vector<std::vector<uint8_t>> captured;
            runtime.gifArbiter().setProcessPacketFn(
                [&](const uint8_t *data,
                    uint32_t sizeBytes)
                {
                    captured.emplace_back(
                        data, data + sizeBytes);
                });
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGifMode, 1u),
                "M3R should mask PATH3");

            runtime.debugStartEeEventTrace(8u);
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGif + 0x10u, kSource),
                "masked GIF MADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGif + 0x20u, kQwc),
                "masked GIF QWC should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGif, 0x100u),
                "masked GIF DMA should start");

            GifDmaSnapshot dma =
                runtime.memory().gifDmaSnapshot();
            t.IsTrue(
                dma.active &&
                    dma.stall ==
                        GifDmaStallReason::Path3Masked,
                "masked submission should retain a path stall");
            t.Equals(
                dma.fifoQwc, 16u,
                "masked submission should fill the GIF FIFO");
            t.Equals(
                dma.qwc, 0u,
                "FIFO fill should consume channel QWC");
            t.Equals(
                runtime.memory().readIORegister(kGifStat),
                0x10000001u,
                "GIF_STAT should expose FQC=16 and M3R");
            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacGif)]
                    .pending,
                "a full masked FIFO should have no deadline");

            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGifMode, 0u),
                "clearing M3R should succeed");
            scheduler = runtime.debugEeSchedulerSnapshot();
            auto slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacGif)];
            t.IsTrue(
                slot.pending,
                "unmask should schedule DMAC_GIF");
            t.Equals(
                slot.deadlineTick, 64ull,
                "M3R removal should own the eight-cycle wakeup");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(64u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            dma = runtime.memory().gifDmaSnapshot();
            t.IsTrue(
                dma.active &&
                    dma.phase == GifDmaPhase::Finalize,
                "wake service should drain into finalization");
            t.Equals(
                dma.fifoQwc, 0u,
                "wake service should empty the FIFO");
            t.Equals(
                runtime.memory().readIORegister(kGifStat),
                0u,
                "FIFO drain should clear FQC and M3R");
            t.Equals(
                captured.size(), static_cast<size_t>(1u),
                "FIFO drain should submit one complete GIF packet");
            scheduler = runtime.debugEeSchedulerSnapshot();
            slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacGif)];
            t.Equals(
                slot.deadlineTick, 320ull,
                "sixteen drained QW should cost thirty-two cycles");

            context.advanceEeCycleTicks(256u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(
                runtime.memory().gifDmaSnapshot().active,
                "final service should retire masked GIF DMA");
            t.IsTrue(
                (runtime.memory().readIORegister(kGif) &
                 0x100u) == 0u,
                "publication should clear masked GIF STR");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 (1u << 2u)) != 0u,
                "publication should latch masked GIF D_STAT");

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(3u),
                "masked DMA should trace wake, finalization, and publication");
            if (trace.entries.size() == 3u)
            {
                const std::array<uint64_t, 3u>
                    expectedTicks = {
                        64ull, 320ull, 320ull};
                t.IsTrue(
                    trace.entries[0].source ==
                            ps2x::timing::EeEventSource::
                                DmacGif &&
                        trace.entries[1].source ==
                            ps2x::timing::EeEventSource::
                                DmacGif &&
                        trace.entries[2].source ==
                            ps2x::timing::EeEventSource::
                                DmacCompletion,
                    "masked wake should precede typed publication");
                for (size_t index = 0u;
                     index < trace.entries.size(); ++index)
                {
                    t.Equals(
                        trace.entries[index].serviceTick,
                        expectedTicks[index],
                        "masked GIF services should retain exact ticks");
                }
            }
        });

        tc.Run("event GIF cancel and restart reject the stale generation", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "GIF restart fixture memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "GIF restart fixture subsystems should bind");

            constexpr uint32_t kGifCtrl = 0x10003000u;
            constexpr uint32_t kGif = 0x1000A000u;
            constexpr uint32_t kDstat = 0x1000E010u;
            constexpr uint32_t kFirstSource =
                0x00034C00u;
            constexpr uint32_t kSecondSource =
                0x00035000u;
            constexpr uint32_t kFirstQwc = 12u;
            constexpr uint32_t kSecondQwc = 16u;
            writeGifImagePacket(
                runtime.memory().getRDRAM() +
                    kFirstSource,
                kFirstQwc, 0x61u);
            writeGifImagePacket(
                runtime.memory().getRDRAM() +
                    kSecondSource,
                kSecondQwc, 0x72u);

            std::vector<std::vector<uint8_t>> captured;
            runtime.gifArbiter().setProcessPacketFn(
                [&](const uint8_t *data,
                    uint32_t sizeBytes)
                {
                    captured.emplace_back(
                        data, data + sizeBytes);
                });

            runtime.debugStartEeEventTrace(8u);
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGif + 0x10u, kFirstSource),
                "first GIF MADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGif + 0x20u, kFirstQwc),
                "first GIF QWC should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGif, 0x100u),
                "first GIF DMA should start");

            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            const auto first =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacGif)];
            const GifDmaSnapshot firstDma =
                runtime.memory().gifDmaSnapshot();
            t.IsTrue(
                first.pending,
                "first GIF generation should be pending");
            t.Equals(
                first.deadlineTick, 64ull,
                "first GIF generation should target tick 64");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(8u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsTrue(
                runtime.memory().cancelDmacTransfer(
                    DmacChannel::Gif),
                "explicit cancellation should retire first GIF DMA");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGifCtrl, 1u),
                "GIF reset should discard the partial first packet");

            scheduler = runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacGif)]
                    .pending,
                "cancellation should remove the first deadline");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGif + 0x10u, kSecondSource),
                "replacement GIF MADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGif + 0x20u, kSecondQwc),
                "replacement GIF QWC should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kGif, 0x100u),
                "replacement GIF DMA should start");

            scheduler = runtime.debugEeSchedulerSnapshot();
            const auto second =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacGif)];
            const GifDmaSnapshot secondDma =
                runtime.memory().gifDmaSnapshot();
            t.IsTrue(
                second.pending &&
                    second.generation > first.generation,
                "replacement should own a newer event generation");
            t.IsTrue(
                secondDma.transfer.generation >
                    firstDma.transfer.generation,
                "replacement should own a newer operation generation");
            t.Equals(
                second.deadlineTick, 136ull,
                "replacement should schedule from committed tick 8");

            context.advanceEeCycleTicks(56u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            GifDmaSnapshot dma =
                runtime.memory().gifDmaSnapshot();
            t.IsTrue(
                dma.active &&
                    dma.phase ==
                        GifDmaPhase::TransferPayload &&
                    dma.qwc == 8u,
                "the stale tick-64 deadline must not advance replacement work");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 (1u << 2u)) == 0u,
                "the stale generation must not publish D_STAT");
            t.Equals(
                runtime.debugEeEventTraceSnapshot(false)
                    .entries.size(),
                static_cast<size_t>(0u),
                "the cancelled deadline must not dispatch");

            context.advanceEeCycleTicks(72u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            dma = runtime.memory().gifDmaSnapshot();
            t.IsTrue(
                dma.active &&
                    dma.phase == GifDmaPhase::Finalize,
                "replacement should reach finalization at tick 136");
            scheduler = runtime.debugEeSchedulerSnapshot();
            t.Equals(
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacGif)]
                    .deadlineTick,
                264ull,
                "replacement should retain its finalization deadline");

            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(
                runtime.memory().gifDmaSnapshot().active,
                "only the replacement should complete");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 (1u << 2u)) != 0u,
                "only the replacement should publish D_STAT");
            t.Equals(
                captured.size(), static_cast<size_t>(1u),
                "reset should discard the cancelled partial packet");

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(3u),
                "only replacement services and publication should dispatch");
            if (trace.entries.size() == 3u)
            {
                const std::array<uint64_t, 3u>
                    expectedTicks = {
                        136ull, 264ull, 264ull};
                for (size_t index = 0u;
                     index < trace.entries.size(); ++index)
                {
                    t.Equals(
                        trace.entries[index].serviceTick,
                        expectedTicks[index],
                        "replacement services should retain exact ticks");
                }
            }
        });

        tc.Run("event VIF0 source chain reproduces the reference VU-finish handoff", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(),
                     "VIF0 reference memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "VIF0 reference subsystems should bind");

            constexpr uint32_t kVif0 = 0x10008000u;
            constexpr uint32_t kDstat = 0x1000E010u;
            constexpr uint32_t kTag = 0x00035C00u;
            constexpr uint32_t kVuUpperNop = 0x000002FFu;
            constexpr uint32_t kVuUpperEnd = 0x400002FFu;
            uint8_t *const rdram =
                runtime.memory().getRDRAM();
            std::memset(rdram + kTag, 0, 16u + 22u * 16u);
            const uint64_t end =
                makeDmaTag(22u, 7u, 0u);
            std::memcpy(rdram + kTag, &end, sizeof(end));

            std::array<uint32_t, 88u> payload{};
            payload[0] = makeVifCmd(0x4Au, 42u, 0u);
            for (uint32_t pair = 0u; pair < 42u; ++pair)
            {
                payload[1u + pair * 2u] = 0u;
                payload[2u + pair * 2u] =
                    pair == 40u
                        ? kVuUpperEnd
                        : kVuUpperNop;
            }
            payload[85] = makeVifCmd(0x14u, 0u, 0u);
            payload[86] = makeVifCmd(0x10u, 0u, 0u);
            payload[87] = 0u;
            std::memcpy(rdram + kTag + 16u,
                        payload.data(), sizeof(payload));

            runtime.debugStartEeEventTrace(12u);
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif0 + 0x30u, kTag),
                     "VIF0 reference TADR write should succeed");
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif0, 0x145u),
                     "VIF0 reference chain should start");

            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            auto dmacSlot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacVif0)];
            t.IsTrue(dmacSlot.pending,
                     "submission should schedule DMAC_VIF0");
            t.Equals(dmacSlot.deadlineTick, 32ull,
                     "submission should retain the four-cycle delay");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(32u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);

            Vif0DmaSnapshot dma =
                runtime.memory().vif0DmaSnapshot();
            t.IsTrue(
                dma.active &&
                    dma.stall ==
                        Vif0DmaStallReason::WaitingForVu,
                "same-tick tag and payload services should reach FLUSHE");
            t.Equals(dma.qwc, 1u,
                     "payload prefix should retain one QWC");
            t.Equals(dma.payloadByteOffset, 8u,
                     "payload prefix should retain two words");
            t.Equals(dma.bufferedPayloadBytes, 8u,
                     "FLUSHE and NOP should remain copied");
            t.IsTrue(runtime.vu0().isActive(),
                     "MSCAL should leave the 42-pair VU0 program active");
            scheduler = runtime.debugEeSchedulerSnapshot();
            dmacSlot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacVif0)];
            t.IsTrue(dmacSlot.pending,
                     "the consumed prefix should retain DMAC ownership");
            t.Equals(dmacSlot.deadlineTick, 376ull,
                     "43 payload cycles should end at tick 376");
            t.IsFalse(
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VifVu0Finish)]
                    .pending,
                "VU finish must wait for the accounted payload event");

            context.advanceEeCycleTicks(344u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            scheduler = runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacVif0)]
                    .pending,
                "the handoff callback should remove DMAC_VIF0");
            const auto &finishSlot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VifVu0Finish)];
            t.IsTrue(finishSlot.pending,
                     "the handoff should schedule VIF_VU0_FINISH");
            t.Equals(finishSlot.deadlineTick, 504ull,
                     "VU finish should be sixteen cycles after handoff");
            t.IsFalse(runtime.vu0().isActive(),
                      "ordinary shared-event catch-up should finish 42 pairs");

            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            dma = runtime.memory().vif0DmaSnapshot();
            t.IsFalse(runtime.memory().vif0WaitingForVu(),
                      "VU finish should clear VEW");
            t.IsTrue(
                dma.phase == Vif0DmaPhase::Finalize,
                "finish wake should consume the retained tail");
            t.Equals(dma.qwc, 0u,
                     "finish wake should consume the final QWC");
            scheduler = runtime.debugEeSchedulerSnapshot();
            dmacSlot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacVif0)];
            t.IsTrue(dmacSlot.pending,
                     "finish wake should schedule final DMAC service");
            t.Equals(dmacSlot.deadlineTick, 520ull,
                     "the resumed two-cycle tail must remain a later event");
            t.IsTrue(
                (runtime.memory().readIORegister(kVif0) &
                 0x100u) != 0u,
                "finish wake should not publish completion");

            context.advanceEeCycleTicks(16u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            t.IsFalse(runtime.memory().vif0DmaSnapshot().active,
                      "final DMAC service should retire the descriptor");
            t.IsTrue(
                (runtime.memory().readIORegister(kVif0) &
                 0x100u) == 0u,
                "same-boundary typed publication should clear STR");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 0x1u) != 0u,
                "typed publication should latch channel-zero D_STAT");

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(trace.entries.size(),
                     static_cast<size_t>(6u),
                     "reference chain should retain six scheduled services");
            if (trace.entries.size() == 6u)
            {
                const std::array<
                    ps2x::timing::EeEventSource, 6u>
                    expectedSources = {
                        ps2x::timing::EeEventSource::DmacVif0,
                        ps2x::timing::EeEventSource::DmacVif0,
                        ps2x::timing::EeEventSource::DmacVif0,
                        ps2x::timing::EeEventSource::VifVu0Finish,
                        ps2x::timing::EeEventSource::DmacVif0,
                        ps2x::timing::EeEventSource::DmacCompletion,
                    };
                const std::array<uint64_t, 6u> expectedTicks = {
                    32ull, 32ull, 376ull,
                    504ull, 520ull, 520ull,
                };
                for (size_t index = 0u;
                     index < trace.entries.size(); ++index)
                {
                    t.IsTrue(
                        trace.entries[index].source ==
                            expectedSources[index],
                        "VIF0 reference services should retain source order");
                    t.Equals(
                        trace.entries[index].serviceTick,
                        expectedTicks[index],
                        "VIF0 reference services should retain exact ticks");
                }
                t.IsTrue(
                    trace.entries[0].deviceBefore.kind ==
                        PS2Runtime::DebugEeEventDeviceKind::
                            Vif0Dma,
                    "DMAC_VIF0 trace should expose typed channel state");
                t.IsTrue(
                    trace.entries[3].deviceBefore.kind ==
                        PS2Runtime::DebugEeEventDeviceKind::
                            Vu0,
                    "VIF_VU0_FINISH trace should expose typed VU0 state");
            }
        });

        tc.Run("event VIF1 normal DMA exposes payload and finalization boundaries", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(),
                     "VIF1 normal fixture memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "VIF1 normal fixture subsystems should bind");

            constexpr uint32_t kVif1 = 0x10009000u;
            constexpr uint32_t kDstat = 0x1000E010u;
            constexpr uint32_t kSource = 0x00030000u;
            constexpr uint32_t kQwc = 2u;
            std::memset(
                runtime.memory().getRDRAM() + kSource,
                0, kQwc * 16u);

            runtime.debugStartEeEventTrace(8u);
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1 + 0x10u, kSource),
                     "VIF1 MADR write should succeed");
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1 + 0x20u, kQwc),
                     "VIF1 QWC write should succeed");
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1, 0x100u),
                     "VIF1 normal start should succeed");

            Vif1DmaSnapshot dma =
                runtime.memory().vif1DmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged,
                     "submission should retain an event-managed descriptor");
            t.IsTrue(
                dma.phase == Vif1DmaPhase::TransferPayload,
                "normal DMA should begin at the payload phase");
            t.Equals(dma.madr, kSource,
                     "submission should retain MADR");
            t.Equals(dma.qwc, kQwc,
                     "submission should retain QWC");
            t.IsTrue(
                (runtime.memory().readIORegister(kVif1) &
                 0x100u) != 0u,
                "submission should expose CHCR.STR immediately");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 0x2u) == 0u,
                "submission must not publish completion");

            const PS2Runtime::DebugEeScheduler submitted =
                runtime.debugEeSchedulerSnapshot();
            const auto &submittedSlot =
                submitted.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacVif1)];
            t.IsTrue(submittedSlot.pending,
                     "submission should schedule DMAC_VIF1");
            t.Equals(submittedSlot.deadlineTick, 32ull,
                     "initial VIF1 event should be four EE cycles away");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(32u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);

            dma = runtime.memory().vif1DmaSnapshot();
            t.IsTrue(dma.active,
                     "payload service should leave finalization pending");
            t.IsTrue(
                dma.phase == Vif1DmaPhase::Finalize,
                "payload service should advance to finalization");
            t.Equals(
                runtime.memory().readIORegister(
                    kVif1 + 0x10u),
                kSource + kQwc * 16u,
                "payload service should advance MADR");
            t.Equals(
                runtime.memory().readIORegister(
                    kVif1 + 0x20u),
                0u,
                "payload service should consume QWC");
            t.IsTrue(
                (runtime.memory().readIORegister(kVif1) &
                 0x100u) != 0u,
                "payload service should keep CHCR.STR busy");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 0x2u) == 0u,
                "payload service should keep D_STAT clear");

            const PS2Runtime::DebugEeScheduler transferred =
                runtime.debugEeSchedulerSnapshot();
            const auto &transferredSlot =
                transferred.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacVif1)];
            t.IsTrue(transferredSlot.pending,
                     "payload work should schedule finalization");
            t.Equals(transferredSlot.deadlineTick, 64ull,
                     "two QWC should cost four EE cycles");

            context.advanceEeCycleTicks(32u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            dma = runtime.memory().vif1DmaSnapshot();
            t.IsFalse(dma.active,
                      "the final event should retire the descriptor");
            t.IsTrue(
                (runtime.memory().readIORegister(kVif1) &
                 0x100u) == 0u,
                "completion publication should clear CHCR.STR");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 0x2u) != 0u,
                "completion publication should latch D_STAT");

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(trace.entries.size(),
                     static_cast<size_t>(3u),
                     "normal DMA should trace payload, finalization, and publication");
            if (trace.entries.size() == 3u)
            {
                t.IsTrue(
                    trace.entries[0].source ==
                        ps2x::timing::EeEventSource::DmacVif1 &&
                    trace.entries[1].source ==
                        ps2x::timing::EeEventSource::DmacVif1 &&
                    trace.entries[2].source ==
                        ps2x::timing::EeEventSource::DmacCompletion,
                    "final state must precede completion publication");
                t.IsTrue(
                    trace.entries[0].deviceBefore.kind ==
                            PS2Runtime::DebugEeEventDeviceKind::
                                Vif1Dma &&
                        trace.entries[0].deviceAfter.kind ==
                            PS2Runtime::DebugEeEventDeviceKind::
                                Vif1Dma,
                    "VIF1 events should retain typed device state");
                t.IsTrue(
                    trace.entries[0].deviceBefore.active &&
                        trace.entries[0].deviceAfter.active,
                    "payload state should remain active across its event");
                t.Equals(
                    trace.entries[0].deviceBefore.phase,
                    static_cast<uint32_t>(
                        Vif1DmaPhase::TransferPayload),
                    "payload trace should expose its starting phase");
                t.Equals(
                    trace.entries[0].deviceBefore.madr,
                    kSource,
                    "payload trace should expose its starting MADR");
                t.Equals(
                    trace.entries[0].deviceBefore.qwc,
                    kQwc,
                    "payload trace should expose its starting QWC");
                t.Equals(
                    trace.entries[0].deviceAfter.phase,
                    static_cast<uint32_t>(
                        Vif1DmaPhase::Finalize),
                    "payload trace should expose its follow-up phase");
                t.Equals(
                    trace.entries[0].deviceAfter.madr,
                    kSource + kQwc * 16u,
                    "payload trace should expose its ending MADR");
                t.Equals(
                    trace.entries[0].deviceAfter.qwc,
                    0u,
                    "payload trace should expose consumed QWC");
                t.IsTrue(
                    trace.entries[1].deviceBefore.active &&
                        !trace.entries[1].deviceAfter.active,
                    "finalization trace should expose retirement");
                t.Equals(
                    trace.entries[0]
                        .deviceBefore.operationGeneration,
                    trace.entries[1]
                        .deviceBefore.operationGeneration,
                    "one VIF1 operation generation should span both events");
            }
        });

        tc.Run("EE event trace trigger retains the first matching service window", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(),
                     "trigger fixture memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "trigger fixture subsystems should bind");

            constexpr uint32_t kVif1 = 0x10009000u;
            constexpr uint32_t kSource = 0x00030300u;
            constexpr uint32_t kTriggerPc = 0x001EE97Cu;
            constexpr uint32_t kBeforePc = 0x001EE970u;
            constexpr uint32_t kAfterPc = 0x001EE980u;
            std::memset(
                runtime.memory().getRDRAM() + kSource,
                0, 16u);

            runtime.debugStartEeEventTrace(
                4u, kTriggerPc, false);
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1 + 0x10u, kSource),
                     "trigger fixture MADR write should succeed");
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1 + 0x20u, 1u),
                     "trigger fixture QWC write should succeed");
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1, 0x100u),
                     "trigger fixture VIF1 start should succeed");

            R5900Context &context = runtime.cpu();
            context.pc = kBeforePc;
            context.advanceEeCycleTicks(32u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);

            const PS2Runtime::DebugEeEventTrace dormant =
                runtime.debugEeEventTraceSnapshot(false);
            t.IsFalse(
                dormant.triggered,
                "a nonmatching service PC must leave the trace dormant");
            t.Equals(
                dormant.totalEntries, 0ull,
                "pre-trigger events must not consume trace capacity");
            t.IsTrue(
                dormant.entries.empty(),
                "pre-trigger events must not enter the retained window");
            t.IsTrue(
                dormant.triggerEePc.has_value() &&
                    *dormant.triggerEePc == kTriggerPc,
                "the trace snapshot should retain its EE PC trigger");

            context.pc = kTriggerPc;
            context.advanceEeCycleTicks(8u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            const PS2Runtime::DebugEeEventTrace armed =
                runtime.debugEeEventTraceSnapshot(false);
            t.IsTrue(
                armed.triggered && armed.entries.empty(),
                "the matching boundary should arm before a later event");

            context.pc = kAfterPc;
            context.advanceEeCycleTicks(8u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.IsTrue(
                trace.triggered,
                "the matching service PC should arm the trace");
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(2u),
                "finalization and completion should form the first retained window");
            if (trace.entries.size() == 2u)
            {
                t.Equals(
                    trace.entries[0].eePc, kAfterPc,
                    "the first post-trigger event should retain its service PC");
                t.Equals(
                    trace.entries[0].sequence, 1ull,
                    "pre-trigger dispatches must not advance trace sequence");
                t.Equals(
                    trace.entries[1].sequence, 2ull,
                    "same-tick follow-up should remain in the triggered window");
            }
        });

        tc.Run("event VIF1 cancel and restart reject the stale DMA generation", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(),
                     "VIF1 restart fixture memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "VIF1 restart fixture subsystems should bind");

            constexpr uint32_t kVif1 = 0x10009000u;
            constexpr uint32_t kFirstSource = 0x00030400u;
            constexpr uint32_t kSecondSource = 0x00030500u;
            std::memset(
                runtime.memory().getRDRAM() + kFirstSource,
                0, 16u);
            std::memset(
                runtime.memory().getRDRAM() + kSecondSource,
                0, 16u);

            runtime.debugStartEeEventTrace(8u);
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1 + 0x10u, kFirstSource),
                     "first MADR write should succeed");
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1 + 0x20u, 1u),
                     "first QWC write should succeed");
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1, 0x100u),
                     "first VIF1 start should succeed");

            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            const auto first =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacVif1)];
            t.IsTrue(first.pending,
                     "the first generation should be scheduled");
            t.Equals(first.deadlineTick, 32ull,
                     "the first generation should target tick 32");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(8u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1, 0u),
                     "clearing STR should cancel the first generation");
            t.IsFalse(
                runtime.memory().vif1DmaSnapshot().active,
                "cancellation should retire the first descriptor");
            scheduler = runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacVif1)]
                    .pending,
                "cancellation should remove the active DMAC_VIF1 slot");

            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1 + 0x10u, kSecondSource),
                     "second MADR write should succeed");
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1 + 0x20u, 1u),
                     "second QWC write should succeed");
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1, 0x100u),
                     "second VIF1 start should succeed");

            scheduler = runtime.debugEeSchedulerSnapshot();
            const auto second =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacVif1)];
            t.IsTrue(second.pending,
                     "the replacement generation should be scheduled");
            t.IsTrue(second.generation > first.generation,
                     "restart should own a newer event generation");
            t.Equals(second.deadlineTick, 40ull,
                     "restart should schedule from committed tick 8");

            context.advanceEeCycleTicks(24u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            Vif1DmaSnapshot dma =
                runtime.memory().vif1DmaSnapshot();
            t.IsTrue(
                dma.active &&
                    dma.phase ==
                        Vif1DmaPhase::TransferPayload,
                "the stale tick-32 event must not advance the replacement");
            t.Equals(dma.madr, kSecondSource,
                     "the replacement descriptor should retain its source");
            t.Equals(dma.qwc, 1u,
                     "the replacement descriptor should retain its QWC");

            context.advanceEeCycleTicks(8u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            dma = runtime.memory().vif1DmaSnapshot();
            t.IsTrue(
                dma.phase == Vif1DmaPhase::Finalize,
                "the replacement should first progress at tick 40");
            t.Equals(dma.madr, kSecondSource + 16u,
                     "only the replacement payload should transfer");

            context.advanceEeCycleTicks(16u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            dma = runtime.memory().vif1DmaSnapshot();
            t.IsFalse(
                dma.active || dma.eventManaged,
                      "replacement finalization should retire all ownership");

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(trace.entries.size(),
                     static_cast<size_t>(3u),
                     "only replacement payload, finalization, and publication should dispatch");
            if (!trace.entries.empty())
            {
                t.Equals(trace.entries.front().serviceTick, 40ull,
                         "the stale tick-32 queue entry must not dispatch");
            }
        });

        tc.Run("event VIF1 reset invalidates a pending DMA event", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(),
                     "VIF1 reset fixture memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "VIF1 reset fixture subsystems should bind");

            constexpr uint32_t kVif1 = 0x10009000u;
            constexpr uint32_t kVif1Fbrst = 0x10003C10u;
            constexpr uint32_t kDstat = 0x1000E010u;
            constexpr uint32_t kSource = 0x00030600u;
            std::memset(
                runtime.memory().getRDRAM() + kSource,
                0, 16u);

            runtime.debugStartEeEventTrace(4u);
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1 + 0x10u, kSource),
                     "reset fixture MADR write should succeed");
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1 + 0x20u, 1u),
                     "reset fixture QWC write should succeed");
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1, 0x100u),
                     "reset fixture VIF1 start should succeed");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(8u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1Fbrst, 1u),
                     "VIF1 reset should succeed");

            t.IsFalse(
                runtime.memory().vif1DmaSnapshot().active,
                "reset should retire the DMA descriptor");
            t.IsTrue(
                (runtime.memory().readIORegister(kVif1) &
                 0x100u) == 0u,
                "reset should clear CHCR.STR");
            const PS2Runtime::DebugEeScheduler reset =
                runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                reset.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacVif1)]
                    .pending,
                "reset should cancel the pending DMAC_VIF1 slot");

            context.advanceEeCycleTicks(24u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.Equals(
                runtime.memory().readIORegister(
                    kVif1 + 0x10u),
                kSource,
                "the stale deadline must not advance MADR");
            t.IsTrue(
                (runtime.memory().readIORegister(kDstat) &
                 0x2u) == 0u,
                "the stale deadline must not publish a completion");
            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(trace.entries.size(),
                     static_cast<size_t>(0u),
                     "the stale reset generation must not dispatch");
        });

        tc.Run("event VIF1 chain DMA advances one tag or payload per event", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(),
                     "VIF1 chain fixture memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "VIF1 chain fixture subsystems should bind");

            constexpr uint32_t kVif1 = 0x10009000u;
            constexpr uint32_t kTag = 0x00031000u;
            uint8_t *const rdram =
                runtime.memory().getRDRAM();
            const uint64_t cnt =
                makeDmaTag(1u, 1u, 0u);
            const uint64_t end =
                makeDmaTag(0u, 7u, 0u);
            std::memcpy(rdram + kTag, &cnt, sizeof(cnt));
            std::memset(rdram + kTag + 16u, 0, 16u);
            std::memcpy(
                rdram + kTag + 32u, &end, sizeof(end));

            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1 + 0x30u, kTag),
                     "VIF1 TADR write should succeed");
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1, 0x105u),
                     "VIF1 chain start should succeed");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(32u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            Vif1DmaSnapshot dma =
                runtime.memory().vif1DmaSnapshot();
            t.IsTrue(
                dma.phase == Vif1DmaPhase::TransferPayload,
                "first event should fetch only the CNT tag");
            t.Equals(dma.tagsProcessed, 1u,
                     "first event should process one tag");
            t.Equals(
                runtime.memory().readIORegister(
                    kVif1 + 0x10u),
                kTag + 16u,
                "CNT setup should expose its payload MADR");
            t.Equals(
                runtime.memory().readIORegister(
                    kVif1 + 0x20u),
                1u,
                "CNT setup should expose its payload QWC");
            t.Equals(
                runtime.memory().readIORegister(
                    kVif1 + 0x30u),
                kTag + 16u,
                "CNT setup should expose the post-tag TADR");

            context.advanceEeCycleTicks(8u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            dma = runtime.memory().vif1DmaSnapshot();
            t.IsTrue(
                dma.phase == Vif1DmaPhase::FetchTag,
                "second event should transfer only the CNT payload");
            t.Equals(dma.tagsProcessed, 1u,
                     "payload service must not fetch another tag");
            t.Equals(
                runtime.memory().readIORegister(
                    kVif1 + 0x10u),
                kTag + 32u,
                "payload service should advance MADR");
            t.Equals(
                runtime.memory().readIORegister(
                    kVif1 + 0x20u),
                0u,
                "payload service should consume QWC");
            t.Equals(
                runtime.memory().readIORegister(
                    kVif1 + 0x30u),
                kTag + 32u,
                "CNT payload completion should advance TADR");

            context.advanceEeCycleTicks(16u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            dma = runtime.memory().vif1DmaSnapshot();
            t.IsTrue(
                dma.phase == Vif1DmaPhase::Finalize,
                "third event should fetch the terminal END tag");
            t.Equals(dma.tagsProcessed, 2u,
                     "END fetch should be the second tag");
            t.Equals(
                runtime.memory().readIORegister(kVif1) &
                    0x70000000u,
                0x70000000u,
                "END fetch should latch tag ID in CHCR");
            t.IsTrue(
                (runtime.memory().readIORegister(kVif1) &
                 0x100u) != 0u,
                "terminal tag setup should remain busy until finalization");

            context.advanceEeCycleTicks(8u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            t.IsTrue(
                (runtime.memory().readIORegister(kVif1) &
                 0x100u) == 0u,
                "the later finalization event should clear STR");
        });

        tc.Run("event VIF1 VU wait removes DMAC_VIF1 until VU wake", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(),
                     "VIF1 VU-wait fixture memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "VIF1 VU-wait fixture subsystems should bind");

            constexpr uint32_t kVuUpperNop = 0x000002FFu;
            constexpr uint32_t kVuUpperEnd = 0x400002FFu;
            uint8_t *const code =
                runtime.memory().getVU1Code();
            std::memset(code, 0, PS2_VU1_CODE_SIZE);
            writeVuInstructionPair(
                code, 0u, 0u, kVuUpperEnd);
            writeVuInstructionPair(
                code, 8u, 0u, kVuUpperNop);
            runtime.memory().markVU1CodeModified();

            const uint32_t mscal =
                makeVifCmd(0x14u, 0u, 0u);
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(&mscal),
                sizeof(mscal));

            constexpr uint32_t kVif1 = 0x10009000u;
            constexpr uint32_t kSource = 0x00032000u;
            const std::array<uint32_t, 4> packet = {
                makeVifCmd(0x11u, 0u, 0u),
                makeVifCmd(0x05u, 0u, 3u),
                0u,
                0u,
            };
            std::memcpy(
                runtime.memory().getRDRAM() + kSource,
                packet.data(), sizeof(packet));
            runtime.debugStartEeEventTrace(12u);
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1 + 0x10u, kSource),
                     "VIF1 wait MADR write should succeed");
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1 + 0x20u, 1u),
                     "VIF1 wait QWC write should succeed");
            t.IsTrue(runtime.memory().writeIORegister(
                         kVif1, 0x100u),
                     "VIF1 wait start should succeed");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(32u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);

            const Vif1DmaSnapshot waiting =
                runtime.memory().vif1DmaSnapshot();
            t.IsTrue(waiting.active,
                     "VU wait should retain the DMA descriptor");
            t.IsTrue(
                waiting.stall ==
                    Vif1DmaStallReason::WaitingForVu,
                "the descriptor should report its VU stall");
            t.IsTrue(runtime.memory().vif1WaitingForVu(),
                     "VIF should expose its VU wait");
            t.IsTrue(
                (runtime.memory().vif1_regs.stat &
                 (1u << 2u)) != 0u,
                "VU wait should set VEW");
            t.Equals(runtime.memory().vif1_regs.mode, 0u,
                     "post-FLUSH commands should remain retained");

            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacVif1)]
                    .pending,
                "DMAC_VIF1 should be absent during the VU-owned wait");
            t.IsTrue(
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VifVu1Finish)]
                    .pending,
                "VIF_VU1_FINISH should own the wait");

            context.advanceEeCycleTicks(96u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(runtime.memory().vif1WaitingForVu(),
                      "VU completion should wake VIF");
            t.Equals(runtime.memory().vif1_regs.mode, 3u,
                     "same-tick DMA wake should resume retained commands");
            t.IsFalse(
                runtime.memory().vif1DmaSnapshot().active,
                "same-tick wake should finalize the completed payload");
            scheduler = runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacVif1)]
                    .pending,
                "the boundary-34-style state should retain no DMAC_VIF1 event");
            t.IsFalse(
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            VifVu1Finish)]
                    .pending,
                "completed VU work should retain no finish event");
        });

        tc.Run("event VU1 deadline includes committed in-block EE time", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(),
                     "VU1 timestamp fixture memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "VU1 timestamp fixture subsystems should bind");

            constexpr uint32_t kVuUpperNop = 0x000002FFu;
            uint8_t *const code = runtime.memory().getVU1Code();
            writeVuInstructionPair(code, 0u, 0u, kVuUpperNop);
            runtime.memory().markVU1CodeModified();

            R5900Context context{};
            {
                PS2Runtime::GuestExecutionScope guest(
                    &runtime, &context);
                context.advanceEeCycleTicks(80u);
                const uint32_t mscal =
                    makeVifCmd(0x14u, 0u, 0u);
                runtime.memory().processVIF1Data(
                    reinterpret_cast<const uint8_t *>(&mscal),
                    sizeof(mscal));

                const PS2Runtime::DebugVu1Timing timing =
                    runtime.debugVu1TimingSnapshot();
                t.Equals(timing.currentTick, 80u,
                         "MSCAL should publish elapsed in-block EE time");
                t.Equals(timing.lastAdvancedTick, 80u,
                         "VU1 should start from the submission tick");
                t.Equals(timing.eventDeadlineTick, 208u,
                         "the startup deadline should be relative to submission");
            }
            (void)runtime.memory().writeIORegister(
                0x10003C10u, 1u);
        });

        tc.Run("event VU1 MSCNT resumes canonical branch state", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(),
                     "VU1 resume fixture memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "VU1 resume fixture subsystems should bind");

            constexpr uint32_t kVuUpperNop = 0x000002FFu;
            constexpr uint32_t kVuUpperEnd = 0x400002FFu;
            uint8_t *const code = runtime.memory().getVU1Code();
            std::memset(code, 0, PS2_VU1_CODE_SIZE);
            writeVuInstructionPair(
                code, 0u, makeVuBranch(2), kVuUpperEnd);
            writeVuInstructionPair(
                code, 8u, makeVuIaddiu(1u, 0u, 1u),
                kVuUpperNop);
            writeVuInstructionPair(
                code, 24u, makeVuIaddiu(2u, 0u, 7u),
                kVuUpperEnd);
            writeVuInstructionPair(
                code, 32u, 0u, kVuUpperNop);
            runtime.memory().markVU1CodeModified();

            const uint32_t mscal = makeVifCmd(0x14u, 0u, 0u);
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(&mscal),
                sizeof(mscal));
            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);

            PS2Runtime::DebugVu1Timing first =
                runtime.debugVu1TimingSnapshot();
            t.IsFalse(first.active,
                      "the first E-bit segment should finish");
            t.Equals(runtime.vu1().state().pc, 24u,
                     "branch target should remain the canonical resume PC");
            t.Equals(runtime.vu1().state().vi[1], 1,
                     "the branch delay slot should execute before completion");

            runtime.memory().vif1_regs.tops = 0x123u;
            runtime.memory().vif1_regs.itops = 0x2ABu;
            const uint32_t mscnt = makeVifCmd(0x17u, 0u, 0u);
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(&mscnt),
                sizeof(mscnt));

            PS2Runtime::DebugVu1Timing resumed =
                runtime.debugVu1TimingSnapshot();
            t.IsTrue(resumed.active,
                     "MSCNT should reactivate the canonical VU1 state");
            t.Equals(resumed.pc, 24u,
                     "MSCNT submission must preserve the resume PC");
            t.Equals(resumed.totalAdvancedCycles, 0u,
                     "MSCNT submission should execute no VU pairs");
            t.IsTrue(resumed.generation > first.generation,
                     "MSCNT should own a new scheduled generation");
            t.Equals(runtime.vu1().state().top, 0x123u,
                     "MSCNT should latch the new TOP value");
            t.Equals(runtime.vu1().state().itop, 0x2ABu,
                     "MSCNT should latch the new ITOP value");

            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(runtime.vu1().isActive(),
                      "the resumed E-bit segment should finish");
            t.Equals(runtime.vu1().state().vi[2], 7,
                     "MSCNT should execute from the retained branch target");
        });

        tc.Run("event VU1 reset invalidates stale finish generation", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(),
                     "VU1 reset fixture memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "VU1 reset fixture subsystems should bind");

            constexpr uint32_t kVuUpperNop = 0x000002FFu;
            constexpr uint32_t kVuUpperEnd = 0x400002FFu;
            uint8_t *const code = runtime.memory().getVU1Code();
            std::memset(code, 0, PS2_VU1_CODE_SIZE);
            for (uint32_t pc = 0u; pc < 256u; pc += 8u)
                writeVuInstructionPair(code, pc, 0u, kVuUpperNop);
            runtime.memory().markVU1CodeModified();

            const uint32_t mscal = makeVifCmd(0x14u, 0u, 0u);
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(&mscal),
                sizeof(mscal));
            const PS2Runtime::DebugVu1Timing started =
                runtime.debugVu1TimingSnapshot();
            t.IsTrue(started.active && started.eventPending,
                     "the old generation should begin active");

            (void)runtime.memory().writeIORegister(
                0x10003C10u, 1u);
            PS2Runtime::DebugVu1Timing reset =
                runtime.debugVu1TimingSnapshot();
            t.IsFalse(reset.active,
                      "VIF1 reset should cancel VU1 execution");
            t.IsFalse(reset.eventPending,
                      "VIF1 reset should cancel the finish event");
            t.IsTrue(reset.generation > started.generation,
                     "VIF1 reset should invalidate the old generation");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            reset = runtime.debugVu1TimingSnapshot();
            t.Equals(reset.totalAdvancedCycles, 0u,
                     "the stale deadline must execute no VU work");

            writeVuInstructionPair(code, 0u, 0u, kVuUpperEnd);
            writeVuInstructionPair(code, 8u, 0u, kVuUpperNop);
            runtime.memory().markVU1CodeModified();
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(&mscal),
                sizeof(mscal));
            const PS2Runtime::DebugVu1Timing restarted =
                runtime.debugVu1TimingSnapshot();
            t.IsTrue(restarted.active && restarted.eventPending,
                     "a post-reset MSCAL should schedule fresh work");
            t.IsTrue(restarted.generation > reset.generation,
                     "restart should use a later VU1 generation");
            t.IsTrue(
                restarted.eventGeneration != started.eventGeneration,
                "restart should use a different scheduler token");

            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(runtime.vu1().isActive(),
                      "only the fresh generation should complete");
        });

        tc.Run("event VU1 XGKICK output follows scheduled cycle budgets", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(),
                     "VU1 XGKICK fixture memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "VU1 XGKICK fixture subsystems should bind");

            std::vector<std::vector<uint8_t>> captured;
            runtime.gifArbiter().setProcessPacketFn(
                [&](const uint8_t *data, uint32_t sizeBytes)
                {
                    captured.emplace_back(data, data + sizeBytes);
                });

            constexpr uint32_t kVuUpperNop = 0x000002FFu;
            uint8_t *const code = runtime.memory().getVU1Code();
            uint8_t *const data = runtime.memory().getVU1Data();
            std::memset(code, 0, PS2_VU1_CODE_SIZE);
            std::memset(data, 0, PS2_VU1_DATA_SIZE);
            for (uint32_t pc = 0u; pc < 2048u; pc += 8u)
                writeVuInstructionPair(code, pc, 0u, kVuUpperNop);
            writeVuInstructionPair(
                code, 0u,
                makeVuLowerSpecial(0x6Cu, 0u),
                kVuUpperNop);
            runtime.memory().markVU1CodeModified();

            constexpr uint16_t kPayloadQwords = 12u;
            const uint64_t tag =
                makeGifTag(kPayloadQwords, GIF_FMT_IMAGE, 0u);
            std::memcpy(data, &tag, sizeof(tag));
            for (uint32_t index = 0u;
                 index < kPayloadQwords * 16u; ++index)
            {
                data[16u + index] =
                    static_cast<uint8_t>(index);
            }

            const uint32_t mscal = makeVifCmd(0x14u, 0u, 0u);
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(&mscal),
                sizeof(mscal));
            t.Equals(captured.size(), static_cast<size_t>(0u),
                     "XGKICK must not submit output with MSCAL");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.Equals(captured.size(), static_cast<size_t>(0u),
                     "the first sixteen cycles should leave the long PATH1 packet incomplete");
            t.Equals(
                runtime.debugVu1TimingSnapshot().totalAdvancedCycles,
                16u,
                "the first event should consume its exact startup budget");

            context.advanceEeCycleTicks(1024u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.Equals(captured.size(), static_cast<size_t>(1u),
                     "the follow-up budget should reach PATH1 EOP exactly once");
            if (!captured.empty())
            {
                t.Equals(
                    captured[0].size(),
                    static_cast<size_t>(
                        (kPayloadQwords + 1u) * 16u),
                    "scheduled XGKICK should retain the complete GIF packet");
            }
            t.Equals(
                runtime.debugVu1TimingSnapshot().totalAdvancedCycles,
                144u,
                "follow-up service should add the reference 128-cycle budget");

            (void)runtime.memory().writeIORegister(
                0x10003C10u, 1u);
        });

        tc.Run("selected VU1 recompiler consumes one scheduled budget across XGKICK exits", [](TestCase &t)
        {
            if (!VuRecompilerBackend::supported())
                return;

            PS2RuntimeConfiguration configuration{};
            configuration.vu1Backend =
                VuBackendKind::Recompiler;
            configuration.useVuBackendEnvironment = false;
            PS2Runtime runtime(configuration);
            t.IsTrue(runtime.memory().initialize(),
                     "native VU1 timing memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "native VU1 timing subsystems should bind");

            std::vector<std::vector<uint8_t>> captured;
            runtime.gifArbiter().setProcessPacketFn(
                [&](const uint8_t *data, uint32_t sizeBytes)
                {
                    captured.emplace_back(
                        data, data + sizeBytes);
                });

            constexpr uint32_t kVuUpperNop = 0x000002FFu;
            uint8_t *const code =
                runtime.memory().getVU1Code();
            uint8_t *const data =
                runtime.memory().getVU1Data();
            std::memset(code, 0, PS2_VU1_CODE_SIZE);
            std::memset(data, 0, PS2_VU1_DATA_SIZE);
            for (uint32_t pc = 0u; pc < 2048u; pc += 8u)
            {
                writeVuInstructionPair(
                    code, pc, 0u, kVuUpperNop);
            }
            writeVuInstructionPair(
                code, 0u,
                makeVuLowerSpecial(0x6Cu, 0u),
                kVuUpperNop);
            runtime.memory().markVU1CodeModified();

            constexpr uint16_t kPayloadQwords = 12u;
            const uint64_t tag =
                makeGifTag(
                    kPayloadQwords, GIF_FMT_IMAGE, 0u);
            std::memcpy(data, &tag, sizeof(tag));
            for (uint32_t index = 0u;
                 index < kPayloadQwords * 16u; ++index)
            {
                data[16u + index] =
                    static_cast<uint8_t>(index);
            }

            const uint32_t mscal =
                makeVifCmd(0x14u, 0u, 0u);
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(&mscal),
                sizeof(mscal));
            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);

            const PS2Runtime::DebugVu1Timing timing =
                runtime.debugVu1TimingSnapshot();
            t.Equals(
                timing.totalAdvancedCycles, uint64_t{16u},
                "native helper exits must not discard the startup budget");
            t.Equals(
                timing.pc, uint32_t{16u * 8u},
                "native execution should reach the same scheduled boundary");
            t.Equals(
                captured.size(), static_cast<size_t>(0u),
                "the first scheduled budget should retain the partial packet");
            t.IsNotNull(
                runtime.vu1().programCacheIfCreated(),
                "the scheduled path should execute compiled VU1 blocks");

            (void)runtime.memory().writeIORegister(
                0x10003C10u, 1u);
        });

        tc.Run("selected VU1 recompiler preserves VIF start resume completion and reset", [](TestCase &t)
        {
            if (!VuRecompilerBackend::supported())
                return;

            PS2RuntimeConfiguration configuration{};
            configuration.vu1Backend =
                VuBackendKind::Recompiler;
            configuration.useVuBackendEnvironment = false;
            PS2Runtime runtime(configuration);
            t.IsTrue(runtime.memory().initialize(),
                     "native VIF/VU1 memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "native VIF/VU1 subsystems should bind");

            constexpr uint32_t kVuUpperNop = 0x000002FFu;
            constexpr uint32_t kVuUpperEnd = 0x400002FFu;
            uint8_t *const code =
                runtime.memory().getVU1Code();
            std::memset(code, 0, PS2_VU1_CODE_SIZE);
            writeVuInstructionPair(
                code, 0u, makeVuBranch(2), kVuUpperEnd);
            writeVuInstructionPair(
                code, 8u,
                makeVuIaddiu(1u, 0u, 1u),
                kVuUpperNop);
            writeVuInstructionPair(
                code, 24u,
                makeVuIaddiu(2u, 0u, 7u),
                kVuUpperEnd);
            writeVuInstructionPair(
                code, 32u, 0u, kVuUpperNop);
            runtime.memory().markVU1CodeModified();

            runtime.memory().vif1_regs.tops = 0x123u;
            runtime.memory().vif1_regs.itops = 0x2ABu;
            const std::array<uint32_t, 3u> startCommands{
                makeVifCmd(0x15u, 0u, 0u), // MSCALF
                makeVifCmd(0x11u, 0u, 0u), // FLUSH
                makeVifCmd(0x05u, 0u, 3u), // STMOD
            };
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(
                    startCommands.data()),
                static_cast<uint32_t>(
                    sizeof(startCommands)));
            t.IsTrue(runtime.vu1().isActive(),
                     "MSCALF should start selected native VU1");
            t.Equals(runtime.vu1().state().top, uint32_t{0x123u},
                     "MSCALF should latch TOPS into TOP");
            t.Equals(runtime.vu1().state().itop, uint32_t{0x2ABu},
                     "MSCALF should latch ITOPS into ITOP");
            t.IsTrue(runtime.memory().vif1WaitingForVu(),
                     "FLUSH should wait for selected native VU1");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(runtime.vu1().isActive(),
                      "native E-bit branch segment should complete");
            t.Equals(runtime.vu1().state().pc, uint32_t{24u},
                     "native branch target should remain the resume PC");
            t.Equals(runtime.vu1().state().vi[1], int32_t{1},
                     "native branch delay slot should execute");
            t.IsFalse(runtime.memory().vif1WaitingForVu(),
                      "native completion should wake VIF1");
            t.Equals(runtime.memory().vif1_regs.mode, uint32_t{3u},
                     "native completion should resume deferred VIF1 commands");

            runtime.memory().vif1_regs.tops = 0x155u;
            runtime.memory().vif1_regs.itops = 0x2CCu;
            const uint32_t mscnt =
                makeVifCmd(0x17u, 0u, 0u);
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(&mscnt),
                sizeof(mscnt));
            t.IsTrue(runtime.vu1().isActive(),
                     "MSCNT should resume selected native VU1");
            t.Equals(runtime.vu1().state().pc, uint32_t{24u},
                     "MSCNT should preserve the native resume PC");
            t.Equals(runtime.vu1().state().top, uint32_t{0x155u},
                     "MSCNT should latch the new TOP");
            t.Equals(runtime.vu1().state().itop, uint32_t{0x2CCu},
                     "MSCNT should latch the new ITOP");

            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(runtime.vu1().isActive(),
                      "resumed native E-bit segment should complete");
            t.Equals(runtime.vu1().state().vi[2], int32_t{7},
                     "MSCNT should execute the retained native target");

            for (uint32_t pc = 0u; pc < 256u; pc += 8u)
            {
                writeVuInstructionPair(
                    code, pc, 0u, kVuUpperNop);
            }
            runtime.memory().markVU1CodeModified();
            const uint32_t mscal =
                makeVifCmd(0x14u, 0u, 0u);
            runtime.memory().processVIF1Data(
                reinterpret_cast<const uint8_t *>(&mscal),
                sizeof(mscal));
            t.IsTrue(runtime.vu1().isActive(),
                     "MSCAL should start a new native generation");
            (void)runtime.memory().writeIORegister(
                0x10003C10u, 1u);
            const PS2Runtime::DebugVu1Timing reset =
                runtime.debugVu1TimingSnapshot();
            t.IsFalse(reset.active,
                      "VIF1 reset should cancel selected native VU1");
            t.IsFalse(reset.eventPending,
                      "VIF1 reset should cancel its native finish event");
        });

        tc.Run("EE cycle ticks honor the post-BIOS dual-issue configuration", [](TestCase &t)
        {
            R5900Context context{};
            t.Equals(
                context.cop0_config, 0x00073443u,
                "standalone recompiled ELF context should start in the normal post-BIOS mode");

            context.advanceEeCycleTicks(9u);
            t.Equals(
                context.ee_block_cycle_ticks, 9u,
                "dual-issue mode should accumulate the base fixed-point weight locally");

            context.cop0_config &= ~(1u << 18u);
            context.advanceEeCycleTicks(9u);
            t.Equals(
                context.ee_block_cycle_ticks, 27u,
                "single-issue mode should double subsequent local issue time");

            t.Equals(
                context.finishEeBasicBlock().raw(), 24u,
                "an EE block boundary should publish only whole cycles");
            t.Equals(
                context.ee_block_cycle_ticks, 0u,
                "a finished block should discard its fractional remainder");

            R5900Context splitBlock{};
            splitBlock.advanceEeCycleTicks(9u);
            t.Equals(
                splitBlock.commitEeBlockCycles().raw(), 8u,
                "an in-block synchronization should expose whole EE cycles");
            t.Equals(
                splitBlock.ee_block_cycle_ticks, 1u,
                "an in-block synchronization should retain fractional issue time");
            t.Equals(
                splitBlock.finishEeBasicBlock().raw(), 8u,
                "the final block commit should preserve PCSX2's one-cycle minimum");

            PS2Runtime runtime;
            t.Equals(
                runtime.debugCpuSnapshot().cop0_config, 0x00073443u,
                "runtime initialization should preserve the post-BIOS configuration");
        });

        tc.Run("EE timing conversions are explicit and saturating", [](TestCase &t)
        {
            using namespace ps2x::timing;

            t.Equals(
                eeCyclesToTicks(3u).raw(), 24u,
                "EE cycles should use eight fixed-point ticks");
            t.Equals(
                eeTicksToCyclesFloor(
                    eeTickDeltaFromRaw(23u)), 2u,
                "EE cycle conversion should round down explicitly");
            t.Equals(
                eeTicksToCyclesCeil(
                    eeTickDeltaFromRaw(17u)), 3u,
                "EE cycle conversion should round up explicitly");
            t.Equals(
                vuCyclesToEeTicks(5u).raw(), 40u,
                "VU cycles should convert through the named EE helper");
            t.Equals(
                iopCyclesToEeTicks(2u).raw(), 128u,
                "one IOP cycle should span eight EE cycles");
            t.Equals(
                eeTicksToIopCyclesFloor(
                    eeTickDeltaFromRaw(191u)), 2u,
                "IOP conversion should state its floor rounding");

            EeTimeline timeline;
            (void)timeline.advance(eeTickDeltaFromRaw(
                std::numeric_limits<uint64_t>::max()));
            (void)timeline.advance(eeCyclesToTicks(1u));
            t.Equals(
                timeline.now().raw(),
                std::numeric_limits<uint64_t>::max(),
                "the canonical timeline should saturate instead of wrapping");
        });

        tc.Run("fresh EE contexts share one monotonic runtime timeline", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context mainContext{};
            R5900Context interruptContext{};

            mainContext.advanceEeCycleTicks(16u);
            runtime.serviceEeEventsAtBlockBoundary(nullptr, &mainContext);
            t.Equals(
                runtime.currentEeTick().raw(), 16u,
                "the first context should publish its elapsed block");

            interruptContext.advanceEeCycleTicks(8u);
            runtime.serviceEeEventsAtBlockBoundary(
                nullptr, &interruptContext);
            t.Equals(
                runtime.currentEeTick().raw(), 24u,
                "a fresh interrupt context must extend rather than replace time");

            mainContext.advanceEeCycleTicks(24u);
            runtime.serviceEeEventsAtBlockBoundary(nullptr, &mainContext);
            t.Equals(
                runtime.currentEeTick().raw(), 48u,
                "returning to the original context must not rewind time");
        });

        tc.Run("nested guest contexts flush local time exactly once", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context mainContext{};

            {
                PS2Runtime::GuestExecutionScope mainScope(
                    &runtime, &mainContext);
                mainContext.advanceEeCycleTicks(9u);

                R5900Context interruptContext = mainContext;
                {
                    PS2Runtime::GuestExecutionScope interruptScope(
                        &runtime, &interruptContext);
                    t.Equals(
                        runtime.currentEeTick().raw(), 8u,
                        "switching contexts should finish the outgoing block");
                    t.Equals(
                        interruptContext.ee_block_cycle_ticks, 0u,
                        "a copied context must not duplicate the caller's local time");

                    interruptContext.advanceEeCycleTicks(9u);
                    {
                        PS2Runtime::GuestExecutionScope callbackScope(
                            &runtime, &interruptContext);
                        interruptContext.advanceEeCycleTicks(7u);
                    }
                    t.Equals(
                        runtime.currentEeTick().raw(), 8u,
                        "nesting the same context should not commit it early");
                }

                t.Equals(
                    runtime.currentEeTick().raw(), 24u,
                    "the nested context should publish one sixteen-tick block");
                mainContext.advanceEeCycleTicks(7u);
            }

            t.Equals(
                runtime.currentEeTick().raw(), 32u,
                "the restored context should publish only its new local block");
        });

        tc.Run("EE timing reset clears canonical and bound local state", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context context{};
            R5900Context staleContext{};
            staleContext.advanceEeCycleTicks(32u);

            {
                PS2Runtime::GuestExecutionScope scope(
                    &runtime, &context);
                context.advanceEeCycleTicks(9u);
                runtime.synchronizeVU0Microprogram(
                    nullptr, &context, false);

                const PS2Runtime::DebugEeTiming beforeReset =
                    runtime.debugEeTimingSnapshot();
                t.Equals(
                    beforeReset.currentTick, 8u,
                    "an in-block synchronization should publish whole ticks");
                t.Equals(
                    beforeReset.localBlockTicks, 1u,
                    "debug timing should expose the fractional local remainder");
                t.IsTrue(
                    beforeReset.contextBound,
                    "debug timing should identify the serialized context");

                runtime.resetEeTiming(&staleContext);
                t.Equals(
                    runtime.currentEeTick().raw(), 0u,
                    "reset should clear canonical runtime time");
                t.Equals(
                    context.ee_block_cycle_ticks, 0u,
                    "reset should clear the currently bound context");
                t.Equals(
                    staleContext.ee_block_cycle_ticks, 0u,
                    "reset should clear an explicitly restored context");

                context.advanceEeCycleTicks(16u);
            }

            t.Equals(
                runtime.currentEeTick().raw(), 16u,
                "post-reset work should begin from the defined zero tick");
        });

        tc.Run("VU0 microprogram executes against VU0 code and data memory", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "PS2Memory initialize should succeed");
            t.IsTrue(runtime.syncCoreSubsystems(), "runtime core subsystems should bind");

            uint8_t *const code = runtime.memory().getVU0Code();
            uint8_t *const data = runtime.memory().getVU0Data();
            std::memset(code, 0, PS2_VU0_CODE_SIZE);
            std::memset(data, 0, PS2_VU0_DATA_SIZE);

            const float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
            std::memcpy(data, input, sizeof(input));

            constexpr uint32_t kVuNop = 0x0000003Fu;
            constexpr uint32_t kVuEndNop = 0x4000003Fu;
            writeVuInstructionPair(code, 0u, makeVuLq(0xFu, 1u, 0u, 0), kVuNop);
            writeVuInstructionPair(code, 8u, 0u, makeVuAdd(0xFu, 2u, 1u, 1u));
            writeVuInstructionPair(code, 16u, makeVuSq(0xFu, 2u, 0u, 1), kVuEndNop);

            R5900Context ctx;
            runtime.executeVU0Microprogram(runtime.memory().getRDRAM(), &ctx, 0u);

            float output[4]{};
            std::memcpy(output, data + 16u, sizeof(output));
            t.Equals(output[0], 2.0f, "VU0 output x should be doubled");
            t.Equals(output[1], 4.0f, "VU0 output y should be doubled");
            t.Equals(output[2], 6.0f, "VU0 output z should be doubled");
            t.Equals(output[3], 8.0f, "VU0 output w should be doubled");

            alignas(16) float vf2[4]{};
            _mm_storeu_ps(vf2, ctx.vu0_vf[2]);
            t.Equals(vf2[0], 2.0f, "VU0 VF2.x should copy back to CPU context");
            t.Equals(static_cast<uint32_t>(ctx.vi[0]), 0u, "VU0 VI0 should remain zero");
        });

        tc.Run("direct VU0 code writes invalidate cached decode", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "PS2Memory initialize should succeed");
            t.IsTrue(runtime.syncCoreSubsystems(), "runtime core subsystems should bind");

            constexpr uint32_t kVuUpperNop = 0x000002FFu;
            constexpr uint32_t kVuUpperEnd = 0x400002FFu;
            const auto pair = [](uint32_t lower, uint32_t upper)
            {
                return static_cast<uint64_t>(lower) |
                       (static_cast<uint64_t>(upper) << 32u);
            };
            const auto writeProgram = [&](uint16_t value)
            {
                runtime.memory().write64(
                    PS2_VU0_CODE_BASE,
                    pair(makeVuIaddiu(1u, 0u, value), kVuUpperNop));
                runtime.memory().write64(
                    PS2_VU0_CODE_BASE + 8u,
                    pair(0u, kVuUpperEnd));
                runtime.memory().write64(
                    PS2_VU0_CODE_BASE + 16u,
                    pair(0u, kVuUpperNop));
            };

            writeProgram(1u);
            R5900Context first{};
            runtime.executeVU0Microprogram(
                runtime.memory().getRDRAM(), &first, 0u);
            t.Equals(static_cast<uint32_t>(first.vi[1]), 1u,
                     "first execution should use the initial VU0 program");

            writeProgram(2u);
            R5900Context second{};
            runtime.executeVU0Microprogram(
                runtime.memory().getRDRAM(), &second, 0u);
            t.Equals(static_cast<uint32_t>(second.vi[1]), 2u,
                     "second execution should rebuild after an EE code write");
        });

        tc.Run("VIF0 MPG invalidates cached VU0 decode", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "PS2Memory initialize should succeed");
            t.IsTrue(runtime.syncCoreSubsystems(), "runtime core subsystems should bind");

            constexpr uint32_t kVuUpperNop = 0x000002FFu;
            constexpr uint32_t kVuUpperEnd = 0x400002FFu;
            const auto pair = [](uint32_t lower, uint32_t upper)
            {
                return static_cast<uint64_t>(lower) |
                       (static_cast<uint64_t>(upper) << 32u);
            };
            const auto uploadProgram = [&](uint16_t value)
            {
                std::vector<uint8_t> packet(sizeof(uint32_t) + 3u * sizeof(uint64_t));
                const uint32_t command = makeVifCmd(0x4Au, 3u, 0u);
                const uint64_t instructions[3] = {
                    pair(makeVuIaddiu(1u, 0u, value), kVuUpperNop),
                    pair(0u, kVuUpperEnd),
                    pair(0u, kVuUpperNop),
                };
                std::memcpy(packet.data(), &command, sizeof(command));
                std::memcpy(
                    packet.data() + sizeof(command),
                    instructions, sizeof(instructions));
                runtime.memory().processVIF0Data(
                    packet.data(), static_cast<uint32_t>(packet.size()));
            };

            const uint64_t initialGeneration =
                runtime.memory().getVU0CodeGeneration();
            uploadProgram(3u);
            t.Equals(
                runtime.memory().getVU0CodeGeneration(),
                initialGeneration + 1u,
                "one successful MPG command should publish one generation");

            R5900Context first{};
            runtime.executeVU0Microprogram(
                runtime.memory().getRDRAM(), &first, 0u);
            t.Equals(static_cast<uint32_t>(first.vi[1]), 3u,
                     "first execution should use the initial MPG upload");

            uploadProgram(4u);
            R5900Context second{};
            runtime.executeVU0Microprogram(
                runtime.memory().getRDRAM(), &second, 0u);
            t.Equals(static_cast<uint32_t>(second.vi[1]), 4u,
                     "second execution should rebuild after an MPG upload");
        });

        tc.Run("VU0 microprogram may complete after more than 4096 instruction pairs", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "PS2Memory initialize should succeed");
            t.IsTrue(runtime.syncCoreSubsystems(), "runtime core subsystems should bind");

            uint8_t *const code = runtime.memory().getVU0Code();
            std::memset(code, 0, PS2_VU0_CODE_SIZE);

            constexpr uint32_t kVuNop = 0x0000003Fu;
            constexpr uint32_t kVuEndNop = 0x4000003Fu;
            writeVuInstructionPair(
                code, 0u, makeVuIsubiu(1u, 1u, 1u), kVuNop);
            writeVuInstructionPair(
                code, 8u, makeVuIbgtz(1u, -2), kVuNop);
            writeVuInstructionPair(code, 16u, 0u, kVuNop);
            writeVuInstructionPair(code, 24u, 0u, kVuEndNop);
            writeVuInstructionPair(code, 32u, 0u, kVuNop);

            R5900Context ctx{};
            ctx.vi[1] = 1370u;
            runtime.executeVU0Microprogram(
                runtime.memory().getRDRAM(), &ctx, 0u);

            t.Equals(
                static_cast<uint32_t>(ctx.vi[1]), 0u,
                "VU0 should reach the E-bit instead of stopping at an internal cycle guard");
            t.Equals(
                ctx.vu0_pc, 40u,
                "VU0 should execute the instruction pair after the E-bit before ending");
        });

        tc.Run("VU0 microprogram observes a non-interlocked EE register write", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(runtime.memory().initialize(), "PS2Memory initialize should succeed");
            t.IsTrue(runtime.syncCoreSubsystems(), "runtime core subsystems should bind");

            uint8_t *const code = runtime.memory().getVU0Code();
            std::memset(code, 0, PS2_VU0_CODE_SIZE);

            constexpr uint32_t kVuNop = 0x0000003Fu;
            constexpr uint32_t kVuEndNop = 0x4000003Fu;
            writeVuInstructionPair(
                code, 0u, makeVuIaddiu(1u, 0u, 1u), kVuNop);
            writeVuInstructionPair(
                code, 8u, makeVuIbeq(1u, 0u, 3), kVuNop);
            writeVuInstructionPair(code, 16u, 0u, kVuNop);
            writeVuInstructionPair(
                code, 24u, makeVuBranch(-3), kVuNop);
            writeVuInstructionPair(code, 32u, 0u, kVuNop);
            writeVuInstructionPair(
                code, 40u, makeVuIaddiu(2u, 0u, 7u), kVuEndNop);
            writeVuInstructionPair(code, 48u, 0u, kVuNop);

            R5900Context ctx{};
            ctx.insn_count = 100u;
            ctx.advanceEeCycleTicks(800u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &ctx);
            runtime.vu0StartMicroProgram(
                runtime.memory().getRDRAM(), &ctx, 0u);

            t.IsTrue(
                (ctx.vu0_vpu_stat & 1u) != 0u,
                "VCALLMS should leave VU0 active while EE execution continues");

            const uint32_t startupPc = ctx.vu0_pc;
            ctx.insn_count = 99u;
            runtime.synchronizeVU0Microprogram(
                runtime.memory().getRDRAM(), &ctx, false);
            t.Equals(
                ctx.vu0_pc, startupPc,
                "an EE instruction-counter reset should not become a huge VU0 time jump");

            ctx.advanceEeCycleTicks(7u);
            runtime.synchronizeVU0Microprogram(
                runtime.memory().getRDRAM(), &ctx, false);
            t.Equals(
                ctx.vu0_pc, startupPc,
                "a fractional EE cycle should not prematurely advance VU0");

            ctx.insn_count += 8u;
            ctx.advanceEeCycleTicks(57u);
            runtime.synchronizeVU0Microprogram(
                runtime.memory().getRDRAM(), &ctx, false);
            t.Equals(
                static_cast<uint32_t>(ctx.vi[1]), 1u,
                "VU0 should reach its EE handshake loop");
            t.Equals(
                static_cast<uint32_t>(ctx.vi[2]), 0u,
                "VU0 should not pass the handshake before the EE write");

            ctx.vi[1] = 0u;
            ctx.insn_count += 5u;
            ctx.advanceEeCycleTicks(104u);
            runtime.synchronizeVU0Microprogram(
                runtime.memory().getRDRAM(), &ctx, false);

            t.Equals(
                static_cast<uint32_t>(ctx.vi[2]), 7u,
                "the running microprogram should observe the non-interlocked VI write");
            t.IsTrue(
                (ctx.vu0_vpu_stat & 1u) == 0u,
                "VU0 should become idle after its E-bit delay slot");
            t.Equals(
                ctx.vu0_pc, 56u,
                "VU0 should stop after the E-bit delay-slot instruction pair");
        });

        tc.Run("selected VU0 recompiler stops at the same EE synchronization "
               "boundaries",
            [](TestCase &t)
            {
                if (!VuRecompilerBackend::supported())
                    return;

                PS2RuntimeConfiguration referenceConfiguration{};
                referenceConfiguration.vu0Backend = VuBackendKind::Interpreter;
                referenceConfiguration.vu1Backend = VuBackendKind::Interpreter;
                referenceConfiguration.useVuBackendEnvironment = false;
                PS2RuntimeConfiguration nativeConfiguration =
                    referenceConfiguration;
                nativeConfiguration.vu0Backend = VuBackendKind::Recompiler;
                PS2RuntimeConfiguration verifyConfiguration =
                    referenceConfiguration;
                verifyConfiguration.vu0Backend = VuBackendKind::Verify;

                PS2Runtime reference(referenceConfiguration);
                PS2Runtime native(nativeConfiguration);
                PS2Runtime verified(verifyConfiguration);
                for (PS2Runtime *runtime :
                     {&reference, &native, &verified})
                {
                    t.IsTrue(runtime->memory().initialize(),
                        "VU0 differential memory should initialize");
                    t.IsTrue(runtime->syncCoreSubsystems(),
                        "VU0 differential subsystems should bind");

                    uint8_t *const code = runtime->memory().getVU0Code();
                    std::memset(code, 0, PS2_VU0_CODE_SIZE);
                    constexpr uint32_t kVuNop = 0x0000003Fu;
                    constexpr uint32_t kVuEndNop = 0x4000003Fu;
                    writeVuInstructionPair(
                        code, 0u, makeVuIaddiu(1u, 0u, 1u), kVuNop);
                    writeVuInstructionPair(
                        code, 8u, makeVuIbeq(1u, 0u, 3), kVuNop);
                    writeVuInstructionPair(code, 16u, 0u, kVuNop);
                    writeVuInstructionPair(code, 24u, makeVuBranch(-3), kVuNop);
                    writeVuInstructionPair(code, 32u, 0u, kVuNop);
                    writeVuInstructionPair(
                        code, 40u, makeVuIaddiu(2u, 0u, 7u), kVuEndNop);
                    writeVuInstructionPair(code, 48u, 0u, kVuNop);
                    runtime->memory().markVU0CodeModified();
                    runtime->vu0().setProgressTrackingEnabled(true);
                }

                R5900Context referenceContext{};
                R5900Context nativeContext{};
                R5900Context verifiedContext{};
                const auto compareCandidate =
                    [&](PS2Runtime &candidate,
                        R5900Context &candidateContext,
                        const std::string &boundary,
                        const std::string &name)
                {
                    std::string difference;
                    t.IsTrue(
                        vuExecutionStatesEqual(
                            reference.vu0().state(),
                            candidate.vu0().state(),
                            &difference),
                        boundary + " " + name +
                            " canonical VU0 state differs at " +
                            difference);
                    t.IsTrue(
                        std::memcmp(
                            reference.memory().getVU0Data(),
                            candidate.memory().getVU0Data(),
                            PS2_VU0_DATA_SIZE) == 0,
                        boundary + " " + name +
                            " VU0 data differs");
                    t.Equals(
                        referenceContext.vu0_pc,
                        candidateContext.vu0_pc,
                        boundary + " " + name +
                            " copied PC differs");
                    t.Equals(
                        referenceContext.vu0_vpu_stat,
                        candidateContext.vu0_vpu_stat,
                        boundary + " " + name +
                            " copied busy state differs");
                    t.Equals(
                        reference.vu0().
                            getProgressSnapshot().cycles,
                        candidate.vu0().
                            getProgressSnapshot().cycles,
                        boundary + " " + name +
                            " retired cycles differ");
                };
                const auto compare = [&](const std::string &boundary)
                {
                    compareCandidate(
                        native, nativeContext,
                        boundary, "native");
                    compareCandidate(
                        verified, verifiedContext,
                        boundary, "verify");
                };

                for (auto pair : {
                         std::pair{&reference, &referenceContext},
                         std::pair{&native, &nativeContext},
                         std::pair{&verified, &verifiedContext},
                     })
                {
                    pair.second->advanceEeCycleTicks(800u);
                    pair.first->serviceEeEventsAtBlockBoundary(
                        pair.first->memory().getRDRAM(), pair.second);
                    pair.first->vu0StartMicroProgram(
                        pair.first->memory().getRDRAM(), pair.second, 0u);
                }
                compare("startup");

                constexpr std::array<uint32_t, 3u> kElapsedTicks{
                    64u, 64u, 128u};
                for (size_t boundary = 0u; boundary < kElapsedTicks.size();
                    ++boundary)
                {
                    if (boundary == 2u)
                    {
                        referenceContext.vi[1] = 0u;
                        nativeContext.vi[1] = 0u;
                        verifiedContext.vi[1] = 0u;
                    }
                    referenceContext.advanceEeCycleTicks(
                        kElapsedTicks[boundary]);
                    nativeContext.advanceEeCycleTicks(kElapsedTicks[boundary]);
                    verifiedContext.advanceEeCycleTicks(
                        kElapsedTicks[boundary]);
                    reference.synchronizeVU0Microprogram(
                        reference.memory().getRDRAM(),
                        &referenceContext,
                        false);
                    native.synchronizeVU0Microprogram(
                        native.memory().getRDRAM(), &nativeContext, false);
                    verified.synchronizeVU0Microprogram(
                        verified.memory().getRDRAM(),
                        &verifiedContext, false);
                    compare("synchronization " + std::to_string(boundary));
                }

                const VuRecompilerDiagnostics *const diagnostics =
                    native.vu0().recompilerDiagnosticsIfCreated();
                t.IsNotNull(diagnostics,
                    "selected VU0 native execution should expose diagnostics");
                if (diagnostics)
                {
                    t.IsTrue(diagnostics->nativePairs > 0u,
                        "the selected VU0 path should retire native pairs");
                    t.Equals(diagnostics->interpreterInstrumentationFallbacks,
                        uint64_t{0u},
                        "ordinary VU0 synchronization should remain native");
                    t.Equals(diagnostics->faultExits,
                        uint64_t{0u},
                        "the selected VU0 path should not fault");
                }
                const VuVerifyDiagnostics &verify =
                    verified.vu0().verifyDiagnostics();
                t.IsTrue(
                    verify.runs > 0u &&
                        verify.comparedPairs > 0u,
                    "VU0 verify should compare scheduler-visible work");
                t.Equals(
                    verify.comparedPairs,
                    verify.publishedPairs,
                    "every compared VU0 pair should publish once");
                t.Equals(
                    verify.mismatches, uint64_t{0u},
                    "VU0 synchronization should remain divergence-free");
            });

        tc.Run("event VU0 progress depends on due shared device work", [](TestCase &t)
        {
            PS2Runtime withDma;
            PS2Runtime withoutDma;
            for (PS2Runtime *runtime :
                 {&withDma, &withoutDma})
            {
                t.IsTrue(
                    runtime->memory().initialize(),
                    "shared-event VU0 memory should initialize");
                t.IsTrue(
                    runtime->syncCoreSubsystems(),
                    "shared-event VU0 subsystems should bind");

                uint8_t *const code =
                    runtime->memory().getVU0Code();
                std::memset(code, 0, PS2_VU0_CODE_SIZE);
                constexpr uint32_t kVuNop = 0x0000003Fu;
                writeVuInstructionPair(
                    code, 0u,
                    makeVuIaddiu(1u, 1u, 1u), kVuNop);
                writeVuInstructionPair(
                    code, 8u, makeVuBranch(-1), kVuNop);
                writeVuInstructionPair(
                    code, 16u, 0u, kVuNop);
                runtime->vu0().setProgressTrackingEnabled(true);
                runtime->vu0StartMicroProgram(
                    runtime->memory().getRDRAM(),
                    &runtime->cpu(), 0u);
            }

            const VuProgressSnapshot startupWithDma =
                withDma.vu0().getProgressSnapshot();
            const VuProgressSnapshot startupWithoutDma =
                withoutDma.vu0().getProgressSnapshot();
            t.Equals(
                startupWithDma.cycles,
                startupWithoutDma.cycles,
                "identical VCALLs should execute the same startup batch");

            const VuProgressSnapshot commonProgress =
                startupWithDma;
            t.Equals(
                commonProgress.cycles,
                withoutDma.vu0().getProgressSnapshot().cycles,
                "event mode should begin with identical VU state");

            constexpr uint32_t kVif1 = 0x10009000u;
            constexpr uint32_t kDstat = 0x1000E010u;
            constexpr uint32_t kSource = 0x00036000u;
            constexpr uint32_t kQwc = 8u;
            std::memset(
                withDma.memory().getRDRAM() + kSource,
                0, kQwc * 16u);
            t.IsTrue(withDma.memory().writeIORegister(
                         kVif1 + 0x10u, kSource),
                     "shared-event VIF1 MADR write should succeed");
            t.IsTrue(withDma.memory().writeIORegister(
                         kVif1 + 0x20u, kQwc),
                     "shared-event VIF1 QWC write should succeed");
            t.IsTrue(withDma.memory().writeIORegister(
                         kVif1, 0x100u),
                     "shared-event VIF1 start should succeed");

            // The four-cycle DMA event is earlier than VU0's new sixteen-
            // cycle lead, so the shared test must not duplicate work.
            withDma.cpu().advanceEeCycleTicks(32u);
            withDma.serviceEeEventsAtBlockBoundary(
                withDma.memory().getRDRAM(),
                &withDma.cpu());
            withoutDma.cpu().advanceEeCycleTicks(32u);
            withoutDma.serviceEeEventsAtBlockBoundary(
                withoutDma.memory().getRDRAM(),
                &withoutDma.cpu());
            t.Equals(
                withDma.vu0().getProgressSnapshot().cycles,
                commonProgress.cycles,
                "an event before the current VU lead should execute no work");
            t.Equals(
                withoutDma.vu0().getProgressSnapshot().cycles,
                commonProgress.cycles,
                "an ordinary block boundary should execute no VU work");

            // Eight QWC cost sixteen EE cycles. Reach finalization through a
            // COP2 synchronization rather than a generated block boundary:
            // the architectural access must first service the due batch.
            withDma.cpu().advanceEeCycleTicks(128u);
            withDma.synchronizeVU0Microprogram(
                withDma.memory().getRDRAM(),
                &withDma.cpu(), false);
            withoutDma.cpu().advanceEeCycleTicks(128u);
            withoutDma.serviceEeEventsAtBlockBoundary(
                withoutDma.memory().getRDRAM(),
                &withoutDma.cpu());

            const VuProgressSnapshot progressed =
                withDma.vu0().getProgressSnapshot();
            const VuProgressSnapshot idleBoundary =
                withoutDma.vu0().getProgressSnapshot();
            t.Equals(
                progressed.cycles,
                commonProgress.cycles + 16u,
                "a COP2 boundary should service due work before catching up VU0");
            t.Equals(
                idleBoundary.cycles,
                commonProgress.cycles,
                "the same EE time without a due source must not advance VU0");
            t.IsTrue(
                (withDma.memory().readIORegister(kDstat) &
                 (1u << 1u)) != 0u,
                "the COP2 boundary should publish the due VIF1 completion");

            withoutDma.synchronizeVU0Microprogram(
                withoutDma.memory().getRDRAM(),
                &withoutDma.cpu(), false);
            const VuProgressSnapshot caughtUp =
                withoutDma.vu0().getProgressSnapshot();
            t.Equals(
                caughtUp.cycles, progressed.cycles,
                "an architectural non-interlocked access should still catch up elapsed work");
            t.Equals(
                withoutDma.vu0().state().pc,
                withDma.vu0().state().pc,
                "shared-event and explicit catch-up should reach the same VU boundary");
        });

        tc.Run("active VU0 does not schedule a private periodic event", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "VU0 scheduler memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "VU0 scheduler subsystems should bind");

            uint8_t *const code = runtime.memory().getVU0Code();
            std::memset(code, 0, PS2_VU0_CODE_SIZE);
            constexpr uint32_t kVuNop = 0x0000003Fu;
            writeVuInstructionPair(
                code, 0u, makeVuIaddiu(1u, 1u, 1u), kVuNop);
            writeVuInstructionPair(
                code, 8u, makeVuBranch(-1), kVuNop);
            writeVuInstructionPair(code, 16u, 0u, kVuNop);
            runtime.vu0().setProgressTrackingEnabled(true);

            R5900Context ctx{};
            ctx.advanceEeCycleTicks(800u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &ctx);
            runtime.vu0StartMicroProgram(
                runtime.memory().getRDRAM(), &ctx, 0u);
            const VuProgressSnapshot startup =
                runtime.vu0().getProgressSnapshot();
            const uint32_t startupPc = runtime.vu0().state().pc;

            constexpr std::array<uint32_t, 5u> kBlockTicks = {
                128u, 40u, 40u, 384u, 40u};
            for (const uint32_t ticks : kBlockTicks)
            {
                ctx.advanceEeCycleTicks(ticks);
                runtime.serviceEeEventsAtBlockBoundary(
                    runtime.memory().getRDRAM(), &ctx);
            }

            const VuProgressSnapshot after =
                runtime.vu0().getProgressSnapshot();
            const PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            t.Equals(
                after.cycles, startup.cycles,
                "ordinary EE blocks should not advance VU0 without shared device work");
            t.Equals(
                runtime.vu0().state().pc, startupPc,
                "an active VU alone should retain its architectural state");
            t.Equals(
                scheduler.statistics.serviced, 0u,
                "an active VU alone should not fabricate scheduler services");
            t.IsFalse(
                scheduler.hasNextDeadline,
                "an active VU alone should not own an event deadline");
        });

        tc.Run("VU0 sync trace may wait for an EE PC trigger", [](TestCase &t)
        {
            PS2RuntimeConfiguration configuration{};
            configuration.vu0Backend = VuBackendKind::Interpreter;
            configuration.vu1Backend = VuBackendKind::Interpreter;
            configuration.useVuBackendEnvironment = false;
            PS2Runtime runtime(configuration);
            t.IsTrue(runtime.memory().initialize(), "PS2Memory initialize should succeed");
            t.IsTrue(runtime.syncCoreSubsystems(), "runtime core subsystems should bind");

            uint8_t *const code = runtime.memory().getVU0Code();
            std::memset(code, 0, PS2_VU0_CODE_SIZE);

            constexpr uint32_t kVuNop = 0x0000003Fu;
            writeVuInstructionPair(
                code, 0u, makeVuIaddiu(1u, 0u, 1u), kVuNop);
            writeVuInstructionPair(
                code, 8u, makeVuIbeq(1u, 0u, 3), kVuNop);
            writeVuInstructionPair(code, 16u, 0u, kVuNop);
            writeVuInstructionPair(
                code, 24u, makeVuBranch(-3), kVuNop);
            writeVuInstructionPair(code, 32u, 0u, kVuNop);

            R5900Context ctx{};
            constexpr uint32_t kTriggerPc = 0x00123456u;
            runtime.debugStartVu0SyncTrace(2u, kTriggerPc, true);

            ctx.pc = 0x00120000u;
            runtime.synchronizeVU0Microprogram(
                runtime.memory().getRDRAM(), &ctx, false);
            PS2Runtime::DebugVu0SyncTrace trace =
                runtime.debugVu0SyncTraceSnapshot(false);
            t.IsFalse(
                trace.triggered,
                "an unrelated inactive synchronization should not arm the trace");

            ctx.pc = kTriggerPc;
            runtime.synchronizeVU0Microprogram(
                runtime.memory().getRDRAM(), &ctx, true);
            trace = runtime.debugVu0SyncTraceSnapshot(false);
            t.IsTrue(
                trace.triggered,
                "an inactive VCALL-style synchronization should arm the trace");
            t.IsTrue(
                trace.hasTriggerSchedulerSnapshot,
                "arming should retain the complete scheduler state at the VCALL boundary");
            t.Equals(
                trace.triggerScheduler.currentTick,
                runtime.currentEeTick().raw(),
                "the trigger snapshot should use the canonical EE timeline");
            t.Equals(
                trace.triggerVsyncTick,
                ps2_syscalls::GetCurrentVSyncTick(
                    &runtime),
                "the trigger snapshot should correlate the host VSync count");
            t.Equals(
                trace.entries.size(), static_cast<size_t>(0u),
                "arming on inactive VU0 should not fabricate a trace entry");

            ctx.advanceEeCycleTicks(800u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &ctx);
            runtime.vu0StartMicroProgram(
                runtime.memory().getRDRAM(), &ctx, 0u);
            t.IsTrue(
                (ctx.vu0_vpu_stat & 1u) != 0u,
                "the handshake fixture should leave VU0 active");

            ctx.pc = 0x00120000u;
            ctx.advanceEeCycleTicks(128u);
            runtime.synchronizeVU0Microprogram(
                runtime.memory().getRDRAM(), &ctx, false);
            trace = runtime.debugVu0SyncTraceSnapshot(false);
            t.Equals(
                trace.entries.size(), static_cast<size_t>(1u),
                "the first active synchronization after the trigger should be retained");
            if (!trace.entries.empty())
            {
                t.Equals(
                    trace.entries.front().eePc, 0x00120000u,
                    "the first retained entry should follow the inactive trigger");
                t.Equals(
                    trace.entries.front().invocationInstruction,
                    static_cast<uint64_t>(16u),
                    "sync tracing should count VU instruction pairs without requiring instruction-state tracing");
            }

            ctx.advanceEeCycleTicks(40u);
            runtime.synchronizeVU0Microprogram(
                runtime.memory().getRDRAM(), &ctx, false);
            trace = runtime.debugVu0SyncTraceSnapshot(false);
            t.IsFalse(
                trace.enabled,
                "a stop-on-full sync trace should disable itself at capacity");
            t.IsTrue(
                trace.stopOnFull,
                "the sync trace snapshot should report stop-on-full mode");
            t.Equals(
                trace.droppedEntries, static_cast<uint64_t>(0u),
                "a stop-on-full trace should retain its first window without drops");
            t.Equals(
                trace.entries.size(), static_cast<size_t>(2u),
                "a later active synchronization should append one trace entry");
            if (trace.entries.size() == 2u)
            {
                t.Equals(
                    trace.entries.back().cycleBudget, 16u,
                    "non-interlocked VU0 catch-up should retain the reference minimum batch");
            }
        });

        tc.Run("VU0 instruction trace captures raw state after an EE PC trigger", [](TestCase &t)
        {
            PS2RuntimeConfiguration configuration{};
            configuration.vu0Backend = VuBackendKind::Interpreter;
            configuration.vu1Backend = VuBackendKind::Interpreter;
            configuration.useVuBackendEnvironment = false;
            PS2Runtime runtime(configuration);
            t.IsTrue(runtime.memory().initialize(), "PS2Memory initialize should succeed");
            t.IsTrue(runtime.syncCoreSubsystems(), "runtime core subsystems should bind");

            uint8_t *const code = runtime.memory().getVU0Code();
            std::memset(code, 0, PS2_VU0_CODE_SIZE);

            constexpr uint32_t kVuNop = 0x0000003Fu;
            writeVuInstructionPair(
                code, 0u, makeVuIaddiu(1u, 0u, 1u), kVuNop);
            writeVuInstructionPair(
                code, 8u, makeVuIbeq(1u, 0u, 3), kVuNop);
            writeVuInstructionPair(code, 16u, 0u, kVuNop);
            writeVuInstructionPair(
                code, 24u, makeVuBranch(-3), kVuNop);
            writeVuInstructionPair(code, 32u, 0u, kVuNop);

            R5900Context ctx{};
            ctx.vi[3] = 0x1234u;
            ctx.vu0_vf[1] = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);
            constexpr uint32_t kTriggerPc = 0x00123456u;
            runtime.debugStartVu0InstructionTrace(
                4u, kTriggerPc, true);

            ctx.pc = 0x00120000u;
            runtime.synchronizeVU0Microprogram(
                runtime.memory().getRDRAM(), &ctx, false);
            PS2Runtime::DebugVu0InstructionTrace trace =
                runtime.debugVu0InstructionTraceSnapshot(false);
            t.IsFalse(
                trace.triggered,
                "an unrelated inactive synchronization should not arm the instruction trace");

            ctx.pc = kTriggerPc;
            runtime.synchronizeVU0Microprogram(
                runtime.memory().getRDRAM(), &ctx, true);
            trace = runtime.debugVu0InstructionTraceSnapshot(false);
            t.IsTrue(
                trace.triggered,
                "an inactive VCALL-style synchronization should arm the instruction trace");
            t.Equals(
                trace.entries.size(), static_cast<size_t>(0u),
                "arming on inactive VU0 should not fabricate an instruction entry");

            ctx.advanceEeCycleTicks(800u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &ctx);
            runtime.vu0StartMicroProgram(
                runtime.memory().getRDRAM(), &ctx, 0u);
            trace = runtime.debugVu0InstructionTraceSnapshot(false);
            t.IsFalse(
                trace.enabled,
                "a stop-on-full instruction trace should disable itself");
            t.IsTrue(
                trace.stopOnFull,
                "the instruction trace snapshot should report stop-on-full mode");
            t.Equals(
                trace.totalEntries, static_cast<uint64_t>(4u),
                "the instruction trace should stop exactly at capacity");
            t.Equals(
                trace.droppedEntries, static_cast<uint64_t>(0u),
                "the instruction trace should retain its first window without drops");
            t.IsTrue(
                !trace.entries.empty(),
                "the triggered VCALL should retain VU0 instructions");
            if (!trace.entries.empty())
            {
                const PS2Runtime::DebugVu0InstructionEntry &entry =
                    trace.entries.front();
                t.Equals(
                    entry.invocationInstruction, 0u,
                    "the first instruction should start a fresh invocation");
                t.Equals(
                    entry.pc, 0u,
                    "the first retained instruction should use the VCALL start PC");
                t.Equals(
                    static_cast<uint32_t>(entry.vi[3]), 0x1234u,
                    "raw VI state should precede the first instruction");
                t.Equals(
                    entry.vf[4], 0x3F800000u,
                    "raw VF state should preserve exact IEEE-754 bits");
            }
        });

        tc.Run("GS sprite draw applies XYOFFSET and fully-outside scissor should not render", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            const uint64_t frame1 =
                (0ull << 0) |   // FBP
                (1ull << 16) |  // FBW
                (0ull << 24) |  // PSM CT32
                (0ull << 32);   // FBMSK

            const uint64_t zbuf1 = (1ull << 32);

            gs.writeRegister(GS_REG_FRAME_1, frame1);
            gs.writeRegister(GS_REG_ZBUF_1, zbuf1);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);

            // XYOFFSET=1,1 pixels (16.4 fixed point).
            const uint64_t xyoffset = (16ull) | (16ull << 32);
            gs.writeRegister(GS_REG_XYOFFSET_1, xyoffset);

            // Scissor initially includes pixel (1,1).
            const uint64_t scissorInside = (0ull) | (3ull << 16) | (0ull << 32) | (3ull << 48);
            gs.writeRegister(GS_REG_SCISSOR_1, scissorInside);

            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_SPRITE));
            gs.writeRegister(GS_REG_RGBAQ, 0xFF3214C8ull); // RGBA=(200,20,50,255)

            // With XYOFFSET=(1,1), a one-pixel sprite from (2,2) to
            // (3,3) draws to pixel (1,1).
            const uint64_t xyz0 = (32ull) | (32ull << 16) | (0ull << 32);
            const uint64_t xyz1 = (48ull) | (48ull << 16) | (0ull << 32);
            gs.writeRegister(GS_REG_XYZ2, xyz0);
            gs.writeRegister(GS_REG_XYZ2, xyz1);

            const uint32_t insideOff = frameOffsetBytes(1u, 1u, 1u);
            t.Equals(vram[insideOff + 0u], static_cast<uint8_t>(200u), "inside draw should write R");
            t.Equals(vram[insideOff + 1u], static_cast<uint8_t>(20u), "inside draw should write G");
            t.Equals(vram[insideOff + 2u], static_cast<uint8_t>(50u), "inside draw should write B");
            t.Equals(vram[insideOff + 3u], static_cast<uint8_t>(255u), "inside draw should write A");

            std::memset(vram.data(), 0, 1024u);

            // Move scissor so target pixel is fully outside.
            const uint64_t scissorOutside = (3ull) | (4ull << 16) | (3ull << 32) | (4ull << 48);
            gs.writeRegister(GS_REG_SCISSOR_1, scissorOutside);
            gs.writeRegister(GS_REG_XYZ2, xyz0);
            gs.writeRegister(GS_REG_XYZ2, xyz1);

            bool anyWrite = false;
            for (size_t i = 0; i < 1024u; ++i)
            {
                if (vram[i] != 0u)
                {
                    anyWrite = true;
                    break;
                }
            }
            t.IsFalse(anyWrite, "fully-outside sprite should not render any pixel");
        });

        tc.Run("GS alpha blend uses ALPHA register FIX factor", [](TestCase &t)
        {
            std::vector<uint8_t> vram(PS2_GS_VRAM_SIZE, 0u);
            GS gs;
            gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);

            const uint64_t frame1 =
                (0ull << 0) |   // FBP
                (1ull << 16) |  // FBW
                (0ull << 24) |  // PSM CT32
                (0ull << 32);   // FBMSK
            const uint64_t zbuf1 = (1ull << 32);
            gs.writeRegister(GS_REG_FRAME_1, frame1);
            gs.writeRegister(GS_REG_ZBUF_1, zbuf1);
            gs.writeRegister(GS_REG_SCISSOR_1, (0ull) | (4ull << 16) | (0ull << 32) | (4ull << 48));
            gs.writeRegister(GS_REG_XYOFFSET_1, 0ull);
            gs.writeRegister(GS_REG_TEST_1, 0x30000ull);

            const uint32_t pxOff = frameOffsetBytes(1u, 1u, 1u);
            vram[pxOff + 0u] = 40u;
            vram[pxOff + 1u] = 40u;
            vram[pxOff + 2u] = 40u;
            vram[pxOff + 3u] = 255u;

            // ABE on sprite prim.
            gs.writeRegister(GS_REG_PRIM, static_cast<uint64_t>(GS_PRIM_SPRITE) | (1ull << 6));

            // ALPHA: (A-B)*FIX/128 + D
            // A=Cs(0), B=Cd(1), C=FIX(2), D=Cd(1), FIX=64.
            const uint64_t alpha = (0ull << 0) | (1ull << 2) | (2ull << 4) | (1ull << 6) | (64ull << 32);
            gs.writeRegister(GS_REG_ALPHA_1, alpha);
            gs.writeRegister(GS_REG_RGBAQ, 0xFFC8C8C8ull); // src RGB = 200

            const uint64_t xyz0 = (16ull) | (16ull << 16) | (0ull << 32);
            const uint64_t xyz1 = (32ull) | (32ull << 16) | (0ull << 32);
            gs.writeRegister(GS_REG_XYZ2, xyz0);
            gs.writeRegister(GS_REG_XYZ2, xyz1);

            // ((200 - 40) * 64 >> 7) + 40 = 120
            t.Equals(vram[pxOff + 0u], static_cast<uint8_t>(120u), "alpha blend should update R with FIX factor");
            t.Equals(vram[pxOff + 1u], static_cast<uint8_t>(120u), "alpha blend should update G with FIX factor");
            t.Equals(vram[pxOff + 2u], static_cast<uint8_t>(120u), "alpha blend should update B with FIX factor");
        });

        tc.Run("notifyRuntimeStop joins guest worker threads before teardown", [](TestCase &t)
        {
            notifyRuntimeStop();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            constexpr uint32_t kEntry = 0x250000u;
            constexpr uint32_t kThreadParamAddr = 0x2600u;
            const uint32_t threadParam[7] = {
                0u,          // attr
                kEntry,      // entry
                0x00100000u, // stack
                0x00000400u, // stack size
                0x00110000u, // gp
                8u,          // priority
                0u           // option
            };

            runtime.registerFunction(kEntry, &testRuntimeWorkerLoop);
            std::memcpy(rdram.data() + kThreadParamAddr, threadParam, sizeof(threadParam));

            R5900Context createCtx{};
            setRegU32(createCtx, 4, kThreadParamAddr);
            CreateThread(rdram.data(), &createCtx, &runtime);
            const int32_t tid = getRegS32(createCtx, 2);
            t.IsTrue(tid > 0, "CreateThread should succeed for teardown-join test");

            R5900Context startCtx{};
            setRegU32(startCtx, 4, static_cast<uint32_t>(tid));
            setRegU32(startCtx, 5, 0u);
            StartThread(rdram.data(), &startCtx, &runtime);
            t.Equals(getRegS32(startCtx, 2), tid, "StartThread should return the launched worker id");

            const bool started = waitUntil([&]()
            {
                return runtime.activeEeHostThreadCount() > 0;
            }, std::chrono::milliseconds(500));
            t.IsTrue(started, "worker thread should become active");
            t.Equals(
                runtime.managedEeExecutionThreadCountForTesting(),
                size_t{1u},
                "the selected EE backend should own the live continuation");

            runtime.requestStop();
            const bool drained = waitUntil([&]()
            {
                return runtime.activeEeHostThreadCount() == 0;
            }, std::chrono::milliseconds(2000));
            t.IsTrue(drained, "requestStop should drain all guest worker threads");

            notifyRuntimeStop(&runtime);
            joinAllGuestHostThreads(&runtime);
            t.Equals(
                runtime.managedEeExecutionThreadCountForTesting(),
                size_t{0u},
                "teardown should release the backend-owned continuation");
        });

        tc.Run("Semaphore poll/signal remains stable under host-thread contention", [](TestCase &t)
        {
            notifyRuntimeStop();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            constexpr uint32_t kParamAddr = 0x2000u;
            // init=1 < max=2 headroom makes both first calls succeed
            // regardless of scheduling: poller is the sole decrementer
            // (count>=1 at first poll), signaler the sole incrementer
            // (count<max at first signal). Keep init<max.
            const uint32_t semaParam[6] = {
                0u, // count
                2u, // max_count
                1u, // init_count
                0u, // wait_threads
                0u, // attr
                0u  // option
            };
            std::memcpy(rdram.data() + kParamAddr, semaParam, sizeof(semaParam));

            R5900Context createCtx{};
            setRegU32(createCtx, 4, kParamAddr);
            CreateSema(rdram.data(), &createCtx, &runtime);
            const int32_t sid = getRegS32(createCtx, 2);
            t.IsTrue(sid >= 0, "CreateSema should return a valid nonnegative sid");

            std::atomic<int32_t> pollOkCount{0};
            std::atomic<int32_t> signalOkCount{0};
            std::atomic<bool> pollerThrew{false};
            std::atomic<bool> signalerThrew{false};
            std::atomic<int32_t> readyCount{0};

            // Release both workers together so their 64-iteration loops start at
            // the same instant, maximizing the opportunity to interleave instead
            // of one thread running to completion before the other is scheduled.
            const auto waitForStart = [&]()
            {
                readyCount.fetch_add(1, std::memory_order_acq_rel);
                while (readyCount.load(std::memory_order_acquire) < 2)
                {
                    std::this_thread::yield();
                }
            };

            std::thread poller([&]()
            {
                try
                {
                    waitForStart();
                    for (int i = 0; i < 64; ++i)
                    {
                        R5900Context pollCtx{};
                        setRegU32(pollCtx, 4, static_cast<uint32_t>(sid));
                        PollSema(rdram.data(), &pollCtx, &runtime);
                        if (getRegS32(pollCtx, 2) == sid)
                        {
                            pollOkCount.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
                catch (...)
                {
                    pollerThrew.store(true, std::memory_order_release);
                }
            });

            std::thread signaler([&]()
            {
                try
                {
                    waitForStart();
                    for (int i = 0; i < 64; ++i)
                    {
                        R5900Context signalCtx{};
                        setRegU32(signalCtx, 4, static_cast<uint32_t>(sid));
                        SignalSema(rdram.data(), &signalCtx, &runtime);
                        if (getRegS32(signalCtx, 2) == sid)
                        {
                            signalOkCount.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
                catch (...)
                {
                    signalerThrew.store(true, std::memory_order_release);
                }
            });

            if (poller.joinable())
            {
                poller.join();
            }
            if (signaler.joinable())
            {
                signaler.join();
            }

            t.IsFalse(pollerThrew.load(std::memory_order_acquire),
                      "PollSema worker thread should not throw");
            t.IsFalse(signalerThrew.load(std::memory_order_acquire),
                      "SignalSema worker thread should not throw");
            t.IsTrue(pollOkCount.load(std::memory_order_relaxed) > 0,
                     "contended PollSema should observe at least one successful acquire");
            t.IsTrue(signalOkCount.load(std::memory_order_relaxed) > 0,
                     "contended SignalSema should observe successful releases");

            constexpr uint32_t kStatusAddr = 0x2100u;
            R5900Context referCtx{};
            setRegU32(referCtx, 4, static_cast<uint32_t>(sid));
            setRegU32(referCtx, 5, kStatusAddr);
            ReferSemaStatus(rdram.data(), &referCtx, &runtime);
            t.Equals(getRegS32(referCtx, 2), KE_OK, "ReferSemaStatus should succeed after contention");

            int32_t finalCount = 0;
            std::memcpy(&finalCount, rdram.data() + kStatusAddr + 0u, sizeof(finalCount));
            t.IsTrue(finalCount >= 0 && finalCount <= 2, "semaphore count should remain within [0, max_count]");

            runtime.requestStop();
            notifyRuntimeStop();
        });

        tc.Run("WaitEventFlag AND-mode is stable under concurrent setters", [](TestCase &t)
        {
            notifyRuntimeStop();
            PS2Runtime runtime;
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);

            constexpr uint32_t kEventParamAddr = 0x2400u;
            constexpr uint32_t kResBitsAddr = 0x2410u;
            const uint32_t eventParam[3] = {0u, 0u, 0u};
            std::memcpy(rdram.data() + kEventParamAddr, eventParam, sizeof(eventParam));

            R5900Context createCtx{};
            setRegU32(createCtx, 4, kEventParamAddr);
            CreateEventFlag(rdram.data(), &createCtx, &runtime);
            const int32_t eid = getRegS32(createCtx, 2);
            t.IsTrue(eid > 0, "CreateEventFlag should return a valid id");

            std::atomic<bool> waiterDone{false};
            std::atomic<int32_t> waiterRet{-9999};
            std::atomic<uint32_t> waiterBits{0u};
            std::atomic<bool> waiterThrew{false};
            std::atomic<bool> setterAThrew{false};
            std::atomic<bool> setterBThrew{false};

            std::thread waiter([&]()
            {
                try
                {
                    R5900Context waitCtx{};
                    setRegU32(waitCtx, 4, static_cast<uint32_t>(eid));
                    setRegU32(waitCtx, 5, 0x3u); // wait for bit0 and bit1 (AND mode)
                    setRegU32(waitCtx, 6, 0u);   // AND, no clear
                    setRegU32(waitCtx, 7, kResBitsAddr);
                    WaitEventFlag(rdram.data(), &waitCtx, &runtime);
                    waiterRet.store(getRegS32(waitCtx, 2), std::memory_order_relaxed);
                    uint32_t bits = 0u;
                    std::memcpy(&bits, rdram.data() + kResBitsAddr, sizeof(bits));
                    waiterBits.store(bits, std::memory_order_relaxed);
                }
                catch (...)
                {
                    waiterThrew.store(true, std::memory_order_release);
                }
                waiterDone.store(true, std::memory_order_release);
            });

            std::thread setterA([&]()
            {
                try
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    R5900Context setCtx{};
                    setRegU32(setCtx, 4, static_cast<uint32_t>(eid));
                    setRegU32(setCtx, 5, 0x1u);
                    SetEventFlag(rdram.data(), &setCtx, &runtime);
                }
                catch (...)
                {
                    setterAThrew.store(true, std::memory_order_release);
                }
            });

            std::thread setterB([&]()
            {
                try
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(15));
                    R5900Context setCtx{};
                    setRegU32(setCtx, 4, static_cast<uint32_t>(eid));
                    setRegU32(setCtx, 5, 0x2u);
                    SetEventFlag(rdram.data(), &setCtx, &runtime);
                }
                catch (...)
                {
                    setterBThrew.store(true, std::memory_order_release);
                }
            });

            const bool woke = waitUntil([&]()
            {
                return waiterDone.load(std::memory_order_acquire);
            }, std::chrono::milliseconds(500));

            if (setterA.joinable())
            {
                setterA.join();
            }
            if (setterB.joinable())
            {
                setterB.join();
            }
            if (waiter.joinable())
            {
                waiter.join();
            }

            t.IsFalse(waiterThrew.load(std::memory_order_acquire),
                      "WaitEventFlag waiter thread should not throw");
            t.IsFalse(setterAThrew.load(std::memory_order_acquire),
                      "SetEventFlag setterA thread should not throw");
            t.IsFalse(setterBThrew.load(std::memory_order_acquire),
                      "SetEventFlag setterB thread should not throw");
            t.IsTrue(woke, "WaitEventFlag AND waiter should wake after both bits are published");
            t.Equals(waiterRet.load(std::memory_order_relaxed), KE_OK, "WaitEventFlag should return KE_OK");
            t.IsTrue((waiterBits.load(std::memory_order_relaxed) & 0x3u) == 0x3u,
                     "WaitEventFlag result bits should include both concurrently-set bits");

            R5900Context deleteCtx{};
            setRegU32(deleteCtx, 4, static_cast<uint32_t>(eid));
            DeleteEventFlag(rdram.data(), &deleteCtx, &runtime);
            runtime.requestStop();
            notifyRuntimeStop();
        });

        tc.Run("sceVu0ApplyMatrix uses libvux matrix math with the imported EE ABI", [](TestCase &t)
        {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};

            constexpr uint32_t kOutAddr = 0x00100000u;
            constexpr uint32_t kMatrixAddr = 0x00100040u;
            constexpr uint32_t kSrcAddr = 0x00100080u;

            const float matrix[16] = {
                1.0f, 2.0f, 3.0f, 4.0f,
                5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f,
                13.0f, 14.0f, 15.0f, 16.0f,
            };
            const float src[4] = {1.0f, 2.0f, 3.0f, 1.0f};
            std::memcpy(rdram.data() + kMatrixAddr, matrix, sizeof(matrix));
            std::memcpy(rdram.data() + kSrcAddr, src, sizeof(src));

            setRegU32(ctx, 4, kOutAddr);
            setRegU32(ctx, 5, kMatrixAddr);
            setRegU32(ctx, 6, kSrcAddr);

            ps2_stubs::sceVu0ApplyMatrix(rdram.data(), &ctx, nullptr);

            float out[4]{};
            std::memcpy(out, rdram.data() + kOutAddr, sizeof(out));
            t.Equals(out[0], 51.0f, "sceVu0ApplyMatrix should compute X with libvux layout");
            t.Equals(out[1], 58.0f, "sceVu0ApplyMatrix should compute Y with libvux layout");
            t.Equals(out[2], 65.0f, "sceVu0ApplyMatrix should compute Z with libvux layout");
            t.Equals(out[3], 72.0f, "sceVu0ApplyMatrix should compute W with libvux layout");
            t.Equals(getRegS32(ctx, 2), 0, "sceVu0ApplyMatrix should report success");
        });

        tc.Run("sceVu0TransposeMatrix transposes a 4x4 matrix with dst/src ABI", [](TestCase &t)
        {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};

            constexpr uint32_t kDstAddr = 0x00100100u;
            constexpr uint32_t kSrcAddr = 0x00100140u;
            const float src[16] = {
                1.0f, 2.0f, 3.0f, 4.0f,
                5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f,
                13.0f, 14.0f, 15.0f, 16.0f,
            };
            std::memcpy(rdram.data() + kSrcAddr, src, sizeof(src));

            setRegU32(ctx, 4, kDstAddr);
            setRegU32(ctx, 5, kSrcAddr);

            ps2_stubs::sceVu0TransposeMatrix(rdram.data(), &ctx, nullptr);

            float out[16]{};
            std::memcpy(out, rdram.data() + kDstAddr, sizeof(out));
            t.Equals(out[0], 1.0f, "transpose should preserve [0][0]");
            t.Equals(out[1], 5.0f, "transpose should swap row 0 col 1");
            t.Equals(out[2], 9.0f, "transpose should swap row 0 col 2");
            t.Equals(out[3], 13.0f, "transpose should swap row 0 col 3");
            t.Equals(out[4], 2.0f, "transpose should swap row 1 col 0");
            t.Equals(out[5], 6.0f, "transpose should preserve [1][1]");
            t.Equals(out[10], 11.0f, "transpose should preserve [2][2]");
            t.Equals(out[12], 4.0f, "transpose should swap row 3 col 0");
            t.Equals(out[15], 16.0f, "transpose should preserve [3][3]");
            t.Equals(getRegS32(ctx, 2), 0, "sceVu0TransposeMatrix should report success");
        });

        tc.Run("event EE counters publish state and INTC only at their canonical deadline", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "counter scheduler fixture memory should initialize");

            constexpr uint32_t kCount = 0x10000000u;
            constexpr uint32_t kMode = 0x10000010u;
            constexpr uint32_t kTarget = 0x10000020u;
            constexpr uint32_t kIntcStat = 0x1000F000u;
            R5900Context &ctx = runtime.cpu();
            uint8_t *const rdram =
                runtime.memory().getRDRAM();

            runtime.Store32(rdram, &ctx, kTarget, 3u);
            runtime.Store32(rdram, &ctx, kCount, 0u);
            runtime.Store32(rdram, &ctx, kMode, 0x1c0u);

            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            const auto &counterSlot =
                scheduler.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            EeCounters)];
            t.IsTrue(
                counterSlot.pending,
                "counter target should occupy its fixed scheduler slot");
            t.Equals(
                counterSlot.deadlineTick, 48u,
                "target three on BUSCLK should schedule six EE cycles");

            ctx.advanceEeCycleTicks(40u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &ctx);
            t.Equals(
                runtime.Load32(rdram, &ctx, kCount),
                2u,
                "counter should expose pre-target progress");
            t.IsTrue(
                (runtime.Load32(
                     rdram, &ctx, kIntcStat) &
                 (1u << 9u)) == 0u,
                "INTC cause must remain hidden before the deadline");

            ctx.advanceEeCycleTicks(8u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &ctx);
            t.Equals(
                runtime.Load32(rdram, &ctx, kCount),
                0u,
                "zero return should publish at target service");
            t.IsTrue(
                (runtime.Load32(rdram, &ctx, kMode) &
                 0x400u) != 0u,
                "target service should latch EQUF");
            t.IsTrue(
                (runtime.Load32(
                     rdram, &ctx, kIntcStat) &
                 (1u << 9u)) != 0u,
                "target service should publish INTC cause 9 after device state");

            runtime.resetEeTiming(&ctx);
            t.Equals(
                runtime.Load32(rdram, &ctx, kMode),
                0u,
                "timing reset should clear counter state");
            scheduler = runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                scheduler.slots[
                    ps2x::timing::eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            EeCounters)]
                    .pending,
                "timing reset should invalidate the counter deadline");
        });

        tc.Run("scratchpad accesses do not alias EE counter MMIO timing", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "scratchpad counter-alias fixture memory should initialize");

            R5900Context &ctx = runtime.cpu();
            uint8_t *const rdram =
                runtime.memory().getRDRAM();
            const __m128i value =
                _mm_set_epi32(4, 3, 2, 1);

            ctx.advanceEeCycleTicks(6u);
            (void)runtime.Load128(
                rdram, &ctx, 0x70000000u);
            PS2Runtime::DebugEeTiming timing =
                runtime.debugEeTimingSnapshot();
            t.Equals(
                timing.currentTick, 0u,
                "a direct scratchpad load must not publish EE time");
            t.Equals(
                timing.localBlockTicks, 6u,
                "a direct scratchpad load must retain local issue time");

            runtime.Store128(
                rdram, &ctx, 0x70000010u, value);
            timing = runtime.debugEeTimingSnapshot();
            t.Equals(
                timing.currentTick, 0u,
                "a direct scratchpad store must not publish EE time");
            t.Equals(
                timing.localBlockTicks, 6u,
                "a direct scratchpad store must retain local issue time");

            (void)runtime.Load128(
                rdram, &ctx, 0xf0000020u);
            timing = runtime.debugEeTimingSnapshot();
            t.Equals(
                timing.currentTick, 0u,
                "an aliased scratchpad load must not publish EE time");
            t.Equals(
                timing.localBlockTicks, 6u,
                "an aliased scratchpad load must retain local issue time");

            runtime.resetEeTiming(&ctx);
            ctx.advanceEeCycleTicks(9u);
            (void)runtime.Load32(
                rdram, &ctx, 0x90000000u);
            timing = runtime.debugEeTimingSnapshot();
            t.Equals(
                timing.currentTick, 8u,
                "a KSEG0 EE-counter alias must still publish EE time");
            t.Equals(
                timing.localBlockTicks, 1u,
                "a real EE-counter access must retain its fractional issue time");
        });

        tc.Run("event From-IPU normal DMA reproduces PCSX2 producer wakes and final delay", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "From-IPU fixture memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "From-IPU fixture subsystems should bind");

            constexpr uint32_t kFromIpu =
                0x1000B000u;
            constexpr uint32_t kIpuCtrl =
                0x10002010u;
            constexpr uint32_t kIpuOutFifo =
                0x10007000u;
            constexpr uint32_t kDctrl =
                0x1000E000u;
            constexpr uint32_t kDstat =
                0x1000E010u;
            constexpr uint32_t kStadr =
                0x1000E060u;
            constexpr uint32_t kDestination =
                0x00035200u;
            std::array<uint8_t, 16u * 16u>
                payload{};
            for (size_t index = 0u;
                 index < payload.size(); ++index)
            {
                payload[index] =
                    static_cast<uint8_t>(
                        (index * 37u + 11u) & 0xFFu);
            }

            t.IsTrue(
                runtime.memory().writeIORegister(
                    kDctrl, 0x31u),
                "From-IPU stall-control mode should enable");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kFromIpu + 0x10u,
                    kDestination),
                "From-IPU MADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kFromIpu + 0x20u, 16u),
                "From-IPU QWC should write");
            runtime.debugStartEeEventTrace(8u);
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kFromIpu, 0x100u),
                "From-IPU normal DMA should start");

            FromIpuDmaSnapshot dma =
                runtime.memory().fromIpuDmaSnapshot();
            t.IsTrue(
                dma.active && !dma.eventManaged &&
                    dma.phase ==
                        FromIpuDmaPhase::
                            TransferPayload &&
                    dma.stall ==
                        FromIpuDmaStallReason::
                            OutputFifoEmpty,
                "an empty output FIFO should leave From-IPU dormant");
            t.Equals(
                dma.qwc, 16u,
                "the empty start should retain all QWC");
            t.Equals(
                runtime.memory().readIORegister(
                    kStadr),
                kDestination,
                "STS=From-IPU should publish initial D_STADR");
            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            auto slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacFromIpu)];
            t.IsFalse(
                slot.pending,
                "an empty output FIFO should own no deadline");

            t.Equals(
                runtime.memory().writeIpuOutputFifo(
                    payload.data(), 8u),
                8u,
                "the first producer slice should fill eight QW");
            dma =
                runtime.memory().fromIpuDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.stall ==
                        FromIpuDmaStallReason::None,
                "output production should wake From-IPU");
            t.Equals(
                (runtime.memory().readIORegister(
                     kIpuCtrl) >>
                 4u) &
                    0xFu,
                8u,
                "IPU_CTRL.OFC should expose output occupancy");
            scheduler =
                runtime.debugEeSchedulerSnapshot();
            slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacFromIpu)];
            t.IsTrue(
                slot.pending,
                "the producer should retain From-IPU ownership");
            t.Equals(
                slot.deadlineTick, 8ull,
                "the producer wake should be one EE cycle later");
            t.IsTrue(
                slot.device.kind ==
                    PS2Runtime::DebugEeEventDeviceKind::
                        FromIpuDma,
                "scheduler status should identify From-IPU DMA");
            t.Equals(
                std::string(
                    PS2Runtime::
                        debugEeEventDeviceKindName(
                            slot.device.kind)),
                std::string("from_ipu_dma"),
                "debugger JSON should use the typed From-IPU label");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(8u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            dma =
                runtime.memory().fromIpuDmaSnapshot();
            t.IsTrue(
                dma.active && !dma.eventManaged &&
                    dma.stall ==
                        FromIpuDmaStallReason::
                            OutputFifoEmpty,
                "the first service should return dormant");
            t.Equals(
                dma.madr,
                kDestination + 8u * 16u,
                "the first service should advance MADR");
            t.Equals(
                dma.qwc, 8u,
                "the first service should retain eight QW");
            t.Equals(
                dma.fifoQwc, 0u,
                "the first service should drain the output FIFO");
            t.Equals(
                runtime.memory().readIORegister(
                    kStadr),
                kDestination + 8u * 16u,
                "the first service should advance D_STADR");
            t.IsTrue(
                std::memcmp(
                    runtime.memory().getRDRAM() +
                        kDestination,
                    payload.data(), 8u * 16u) == 0,
                "the first service should copy exact FIFO bytes");

            t.Equals(
                runtime.memory().writeIpuOutputFifo(
                    payload.data() + 8u * 16u, 8u),
                8u,
                "the second producer slice should fill eight QW");
            scheduler =
                runtime.debugEeSchedulerSnapshot();
            slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacFromIpu)];
            t.Equals(
                slot.deadlineTick, 16ull,
                "the second producer should wake one cycle later");

            context.advanceEeCycleTicks(8u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            dma =
                runtime.memory().fromIpuDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.phase ==
                        FromIpuDmaPhase::Finalize &&
                    dma.stall ==
                        FromIpuDmaStallReason::None,
                "the second service should retain finalization");
            t.Equals(
                dma.madr,
                kDestination + 16u * 16u,
                "the second service should advance final MADR");
            t.Equals(
                dma.qwc, 0u,
                "the second service should consume all QWC");
            t.Equals(
                runtime.memory().readIORegister(
                    kStadr),
                kDestination + 16u * 16u,
                "the second service should advance final D_STADR");
            t.IsTrue(
                std::memcmp(
                    runtime.memory().getRDRAM() +
                        kDestination,
                    payload.data(), payload.size()) == 0,
                "both services should reproduce the FIFO payload");
            scheduler =
                runtime.debugEeSchedulerSnapshot();
            slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacFromIpu)];
            t.Equals(
                slot.deadlineTick, 144ull,
                "eight final QW should delay completion sixteen cycles");

            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(
                runtime.memory()
                    .fromIpuDmaSnapshot()
                    .active,
                "the final callback should retire From-IPU DMA");
            t.IsTrue(
                (runtime.memory().readIORegister(
                     kFromIpu) &
                 0x100u) == 0u,
                "same-boundary publication should clear STR");
            t.IsTrue(
                (runtime.memory().readIORegister(
                     kDstat) &
                 (1u << 3u)) != 0u,
                "same-boundary publication should latch From-IPU D_STAT");

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(4u),
                "the oracle should trace three device services and completion");
            if (trace.entries.size() == 4u)
            {
                const std::array<
                    ps2x::timing::EeEventSource, 4u>
                    expectedSources = {
                        ps2x::timing::EeEventSource::
                            DmacFromIpu,
                        ps2x::timing::EeEventSource::
                            DmacFromIpu,
                        ps2x::timing::EeEventSource::
                            DmacFromIpu,
                        ps2x::timing::EeEventSource::
                            DmacCompletion,
                    };
                const std::array<uint64_t, 4u>
                    expectedTicks = {
                        8ull, 16ull, 144ull, 144ull};
                for (size_t index = 0u;
                     index < trace.entries.size();
                     ++index)
                {
                    t.IsTrue(
                        trace.entries[index].source ==
                            expectedSources[index],
                        "From-IPU services should retain source order");
                    t.Equals(
                        trace.entries[index].serviceTick,
                        expectedTicks[index],
                        "From-IPU services should retain exact ticks");
                }
            }

            std::array<uint8_t, 16u> directPayload{};
            for (size_t index = 0u;
                 index < directPayload.size();
                 ++index)
            {
                directPayload[index] =
                    static_cast<uint8_t>(0xD0u + index);
            }
            t.Equals(
                runtime.memory().writeIpuOutputFifo(
                    directPayload.data(), 1u),
                1u,
                "a direct-read payload should enter the output FIFO");
            const __m128i direct =
                runtime.memory().read128(kIpuOutFifo);
            alignas(16) std::array<uint8_t, 16u>
                directRead{};
            _mm_storeu_si128(
                reinterpret_cast<__m128i *>(
                    directRead.data()),
                direct);
            t.IsTrue(
                directRead == directPayload,
                "direct IPU output reads should consume exact FIFO bytes");
            t.Equals(
                (runtime.memory().readIORegister(
                     kIpuCtrl) >>
                 4u) &
                    0xFu,
                0u,
                "direct output reads should decrement OFC");
        });

        tc.Run("From-IPU zero QWC disable reset and cancel preserve retained ownership", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "From-IPU lifecycle memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "From-IPU lifecycle subsystems should bind");

            constexpr uint32_t kFromIpu =
                0x1000B000u;
            constexpr uint32_t kIpuCtrl =
                0x10002010u;
            constexpr uint32_t kDctrl =
                0x1000E000u;
            constexpr uint32_t kDstat =
                0x1000E010u;
            constexpr uint32_t kDestination =
                0x00035400u;
            std::array<uint8_t, 16u> payload{};
            payload.fill(0xA5u);

            (void)runtime.memory().writeIORegister(
                kFromIpu + 0x10u, kDestination);
            (void)runtime.memory().writeIORegister(
                kFromIpu + 0x20u, 0u);
            (void)runtime.memory().writeIORegister(
                kFromIpu, 0x100u);
            t.IsFalse(
                runtime.memory()
                    .fromIpuDmaSnapshot()
                    .active,
                "an empty zero-QWC start should terminate without underflow");
            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                scheduler
                    .slots[ps2x::timing::
                        eeEventSourceIndex(
                            ps2x::timing::
                                EeEventSource::
                                    DmacFromIpu)]
                    .pending,
                "zero-QWC termination should not invent a device callback");
            t.IsTrue(
                scheduler
                    .slots[ps2x::timing::
                        eeEventSourceIndex(
                            ps2x::timing::
                                EeEventSource::
                                    DmacCompletion)]
                    .pending,
                "zero-QWC termination should use typed completion");
            R5900Context &context = runtime.cpu();
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsTrue(
                (runtime.memory().readIORegister(
                     kDstat) &
                 (1u << 3u)) != 0u,
                "zero-QWC termination should latch From-IPU D_STAT");
            (void)runtime.memory().writeIORegister(
                kDstat, 1u << 3u);

            t.Equals(
                runtime.memory().writeIpuOutputFifo(
                    payload.data(), 1u),
                1u,
                "a zero-QWC transfer should see prefilled output");
            (void)runtime.memory().writeIORegister(
                kFromIpu + 0x10u, kDestination);
            (void)runtime.memory().writeIORegister(
                kFromIpu + 0x20u, 0u);
            (void)runtime.memory().writeIORegister(
                kFromIpu, 0x100u);
            FromIpuDmaSnapshot dma =
                runtime.memory().fromIpuDmaSnapshot();
            t.IsTrue(
                dma.active && !dma.eventManaged &&
                    dma.stall ==
                        FromIpuDmaStallReason::
                            OutputFifoEmpty,
                "prefilled zero-QWC should transfer before testing completion");
            t.Equals(
                dma.qwc, 0xFFFFu,
                "prefilled zero-QWC should apply normal DMAC underflow");
            t.Equals(
                dma.fifoQwc, 0u,
                "the underflowing transfer should consume its output");
            t.IsTrue(
                std::memcmp(
                    runtime.memory().getRDRAM() +
                        kDestination,
                    payload.data(), payload.size()) == 0,
                "the underflowing transfer should copy exact output");
            (void)runtime.memory().writeIORegister(
                kFromIpu, 0u);

            (void)runtime.memory().writeIORegister(
                kDctrl, 0u);
            (void)runtime.memory().writeIORegister(
                kFromIpu + 0x10u, kDestination);
            (void)runtime.memory().writeIORegister(
                kFromIpu + 0x20u, 1u);
            (void)runtime.memory().writeIORegister(
                kFromIpu, 0x100u);
            dma =
                runtime.memory().fromIpuDmaSnapshot();
            t.IsTrue(
                dma.active && !dma.eventManaged &&
                    dma.stall ==
                        FromIpuDmaStallReason::
                            DmacDisabled,
                "disabled DMAC should retain From-IPU without a deadline");
            t.Equals(
                runtime.memory().writeIpuOutputFifo(
                    payload.data(), 1u),
                1u,
                "a disabled channel should retain produced output");
            scheduler =
                runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                scheduler
                    .slots[ps2x::timing::
                        eeEventSourceIndex(
                            ps2x::timing::
                                EeEventSource::
                                    DmacFromIpu)]
                    .pending,
                "disabled From-IPU should not acquire a deadline");

            (void)runtime.memory().writeIORegister(
                kIpuCtrl, 1u << 30u);
            dma =
                runtime.memory().fromIpuDmaSnapshot();
            t.Equals(
                dma.fifoQwc, 0u,
                "IPU reset should clear the output FIFO");
            t.IsTrue(
                dma.active &&
                    dma.stall ==
                        FromIpuDmaStallReason::
                            DmacDisabled,
                "IPU reset should not fabricate channel completion");
            (void)runtime.memory().writeIORegister(
                kDctrl, 1u);
            dma =
                runtime.memory().fromIpuDmaSnapshot();
            t.IsTrue(
                dma.active && !dma.eventManaged &&
                    dma.stall ==
                        FromIpuDmaStallReason::
                            OutputFifoEmpty,
                "reenabling an empty channel should leave it dormant");

            t.Equals(
                runtime.memory().writeIpuOutputFifo(
                    payload.data(), 1u),
                1u,
                "post-reset output should wake the retained channel");
            scheduler =
                runtime.debugEeSchedulerSnapshot();
            t.IsTrue(
                scheduler
                    .slots[ps2x::timing::
                        eeEventSourceIndex(
                            ps2x::timing::
                                EeEventSource::
                                    DmacFromIpu)]
                    .pending,
                "post-reset output should own a deadline");
            (void)runtime.memory().writeIORegister(
                kFromIpu, 0u);
            t.IsFalse(
                runtime.memory()
                    .fromIpuDmaSnapshot()
                    .active,
                "clearing STR should cancel retained From-IPU");
            scheduler =
                runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                scheduler
                    .slots[ps2x::timing::
                        eeEventSourceIndex(
                            ps2x::timing::
                                EeEventSource::
                                    DmacFromIpu)]
                    .pending,
                "cancellation should invalidate the scheduled generation");
            context.advanceEeCycleTicks(8u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsTrue(
                (runtime.memory().readIORegister(
                     kDstat) &
                 (1u << 3u)) == 0u,
                "a stale callback must not publish completion");

            (void)runtime.memory().writeIORegister(
                kIpuCtrl, 1u << 30u);
            (void)runtime.memory().writeIORegister(
                kFromIpu + 0x20u, 1u);
            (void)runtime.memory().writeIORegister(
                kFromIpu, 0x104u);
            dma =
                runtime.memory().fromIpuDmaSnapshot();
            t.IsTrue(
                dma.active && !dma.eventManaged &&
                    dma.phase ==
                        FromIpuDmaPhase::Fault &&
                    dma.stall ==
                        FromIpuDmaStallReason::
                            UnsupportedMode,
                "From-IPU should retain unsupported modes as an explicit fault");
            (void)runtime.memory().writeIORegister(
                kFromIpu, 0u);
        });

        tc.Run("From-IPU destinations follow DMAC scratchpad and zero-write mappings", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "From-IPU mapping memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "From-IPU mapping subsystems should bind");

            constexpr uint32_t kFromIpu =
                0x1000B000u;
            constexpr uint32_t kScratchDestination =
                0x80000100u;
            constexpr uint32_t kZeroWriteDestination =
                0x02000000u;
            std::array<uint8_t, 3u * 16u> payload{};
            for (size_t index = 0u;
                 index < payload.size(); ++index)
            {
                payload[index] =
                    static_cast<uint8_t>(
                        0x40u + index);
            }

            t.Equals(
                runtime.memory().writeIpuOutputFifo(
                    payload.data(), 3u),
                3u,
                "the mapping fixture should prefill three output QW");
            (void)runtime.memory().writeIORegister(
                kFromIpu + 0x10u,
                kScratchDestination);
            (void)runtime.memory().writeIORegister(
                kFromIpu + 0x20u, 2u);
            (void)runtime.memory().writeIORegister(
                kFromIpu, 0x100u);
            FromIpuDmaSnapshot dma =
                runtime.memory().fromIpuDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.phase ==
                        FromIpuDmaPhase::Finalize,
                "prefilled scratchpad output should retain finalization");
            t.IsTrue(
                std::memcmp(
                    runtime.memory().getScratchpad() +
                        0x100u,
                    payload.data(), 2u * 16u) == 0,
                "flagged DMAC addresses should target scratchpad");
            t.Equals(
                dma.fifoQwc, 1u,
                "the scratch transfer should retain the third output QW");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(32u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(
                runtime.memory()
                    .fromIpuDmaSnapshot()
                    .active,
                "the scratch transfer should complete");

            (void)runtime.memory().writeIORegister(
                kFromIpu + 0x10u,
                kZeroWriteDestination);
            (void)runtime.memory().writeIORegister(
                kFromIpu + 0x20u, 1u);
            (void)runtime.memory().writeIORegister(
                kFromIpu, 0x100u);
            dma =
                runtime.memory().fromIpuDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.phase ==
                        FromIpuDmaPhase::Finalize &&
                    dma.stall ==
                        FromIpuDmaStallReason::None,
                "unpopulated physical writes should use the DMAC zero-write page");
            t.Equals(
                dma.fifoQwc, 0u,
                "the zero-write page should still consume output");
            context.advanceEeCycleTicks(16u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(
                runtime.memory()
                    .fromIpuDmaSnapshot()
                    .active,
                "the zero-write transfer should complete normally");
        });


        tc.Run("event To-IPU normal DMA reproduces PCSX2 FIFO stall and reset wake", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "To-IPU fixture memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "To-IPU fixture subsystems should bind");

            constexpr uint32_t kToIpu =
                0x1000B400u;
            constexpr uint32_t kIpuCtrl =
                0x10002010u;
            constexpr uint32_t kIpuInFifo =
                0x10007010u;
            constexpr uint32_t kDstat =
                0x1000E010u;
            constexpr uint32_t kFirstSource =
                0x00035000u;
            constexpr uint32_t kSecondSource =
                0x00035100u;
            std::memset(
                runtime.memory().getRDRAM() +
                    kFirstSource,
                0x31, 4u * 16u);
            std::memset(
                runtime.memory().getRDRAM() +
                    kSecondSource,
                0x72, 12u * 16u);

            runtime.debugStartEeEventTrace(8u);
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kToIpu + 0x10u,
                    kFirstSource),
                "first To-IPU MADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kToIpu + 0x20u, 4u),
                "first To-IPU QWC should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kToIpu, 0x100u),
                "first To-IPU normal DMA should start");

            ToIpuDmaSnapshot dma =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.phase ==
                        ToIpuDmaPhase::Finalize,
                "four QW should enter the FIFO and retain finalization");
            t.Equals(
                dma.fifoQwc, 4u,
                "the first payload should occupy four FIFO QW");
            t.Equals(
                dma.qwc, 0u,
                "the first payload should consume its QWC");
            t.Equals(
                dma.madr,
                kFirstSource + 4u * 16u,
                "the first payload should advance MADR");
            t.Equals(
                runtime.memory().readIORegister(
                    kIpuCtrl) &
                    0xFu,
                4u,
                "IPU_CTRL.IFC should expose FIFO occupancy");

            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            auto slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacToIpu)];
            t.IsTrue(
                slot.pending,
                "the first transfer should retain To-IPU ownership");
            t.Equals(
                slot.deadlineTick, 64ull,
                "four QW should finalize after eight EE cycles");
            t.IsTrue(
                slot.device.kind ==
                    PS2Runtime::DebugEeEventDeviceKind::
                        ToIpuDma,
                "scheduler status should identify To-IPU DMA");
            t.Equals(
                std::string(
                    PS2Runtime::
                        debugEeEventDeviceKindName(
                            slot.device.kind)),
                std::string("to_ipu_dma"),
                "debugger JSON should use the typed To-IPU device label");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(64u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(
                runtime.memory()
                    .toIpuDmaSnapshot()
                    .active,
                "the first finalization should retire To-IPU DMA");
            t.IsTrue(
                (runtime.memory().readIORegister(
                     kToIpu) &
                 0x100u) == 0u,
                "same-boundary publication should clear first STR");
            t.IsTrue(
                (runtime.memory().readIORegister(
                     kDstat) &
                 (1u << 4u)) != 0u,
                "same-boundary publication should latch To-IPU D_STAT");

            t.IsTrue(
                runtime.memory().writeIORegister(
                    kDstat, 1u << 4u),
                "To-IPU D_STAT should clear");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kToIpu + 0x10u,
                    kSecondSource),
                "second To-IPU MADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kToIpu + 0x20u, 12u),
                "second To-IPU QWC should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kToIpu, 0x100u),
                "second To-IPU normal DMA should start");

            dma = runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active && !dma.eventManaged &&
                    dma.phase ==
                        ToIpuDmaPhase::
                            TransferPayload &&
                    dma.stall ==
                        ToIpuDmaStallReason::
                            InputFifoFull,
                "the second transfer should stall without a deadline");
            t.Equals(
                dma.fifoQwc, 8u,
                "the second transfer should fill the FIFO");
            t.Equals(
                dma.qwc, 8u,
                "eight QW should remain while the FIFO is full");
            t.Equals(
                dma.madr,
                kSecondSource + 4u * 16u,
                "the partial second transfer should expose MADR progress");
            scheduler = runtime.debugEeSchedulerSnapshot();
            slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacToIpu)];
            t.IsFalse(
                slot.pending,
                "a full input FIFO should leave To-IPU dormant");
            t.IsTrue(
                (runtime.memory().readIORegister(
                     kToIpu) &
                 0x100u) != 0u,
                "STR should remain visible during the FIFO stall");

            t.IsTrue(
                runtime.memory().writeIORegister(
                    kIpuCtrl, 1u << 30u),
                "IPU reset should clear the input FIFO");
            dma = runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.stall ==
                        ToIpuDmaStallReason::None,
                "IPU reset should wake the stalled transfer");
            t.Equals(
                dma.fifoQwc, 0u,
                "IPU reset should empty the input FIFO");
            scheduler = runtime.debugEeSchedulerSnapshot();
            slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacToIpu)];
            t.IsTrue(
                slot.pending,
                "IPU reset should schedule To-IPU work");
            t.Equals(
                slot.deadlineTick, 96ull,
                "the reset wake should occur four EE cycles later");

            context.advanceEeCycleTicks(32u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            dma = runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active &&
                    dma.phase ==
                        ToIpuDmaPhase::Finalize &&
                    dma.stall ==
                        ToIpuDmaStallReason::None,
                "the reset wake should transfer the remaining payload");
            t.Equals(
                dma.qwc, 0u,
                "the reset wake should consume the remaining QWC");
            t.Equals(
                dma.madr,
                kSecondSource + 12u * 16u,
                "the reset wake should advance MADR to the end");
            t.Equals(
                dma.fifoQwc, 8u,
                "the remaining eight QW should refill the FIFO");
            scheduler = runtime.debugEeSchedulerSnapshot();
            slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacToIpu)];
            t.IsTrue(
                slot.pending,
                "payload service should retain finalization ownership");
            t.Equals(
                slot.deadlineTick, 224ull,
                "eight QW should cost sixteen cycles from reset service");

            context.advanceEeCycleTicks(128u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(
                runtime.memory()
                    .toIpuDmaSnapshot()
                    .active,
                "the second finalization should retire To-IPU DMA");
            t.IsTrue(
                (runtime.memory().readIORegister(
                     kToIpu) &
                 0x100u) == 0u,
                "same-boundary publication should clear second STR");
            t.IsTrue(
                (runtime.memory().readIORegister(
                     kDstat) &
                 (1u << 4u)) != 0u,
                "same-boundary publication should latch second D_STAT");

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(5u),
                "the oracle should trace two completions and three device services");
            if (trace.entries.size() == 5u)
            {
                const std::array<
                    ps2x::timing::EeEventSource, 5u>
                    expectedSources = {
                        ps2x::timing::EeEventSource::
                            DmacToIpu,
                        ps2x::timing::EeEventSource::
                            DmacCompletion,
                        ps2x::timing::EeEventSource::
                            DmacToIpu,
                        ps2x::timing::EeEventSource::
                            DmacToIpu,
                        ps2x::timing::EeEventSource::
                            DmacCompletion,
                    };
                const std::array<uint64_t, 5u>
                    expectedTicks = {
                        64ull, 64ull, 96ull,
                        224ull, 224ull};
                for (size_t index = 0u;
                     index < trace.entries.size();
                     ++index)
                {
                    t.IsTrue(
                        trace.entries[index].source ==
                            expectedSources[index],
                        "To-IPU services should retain source order");
                    t.Equals(
                        trace.entries[index].serviceTick,
                        expectedTicks[index],
                        "To-IPU services should retain exact ticks");
                }
            }

            t.IsTrue(
                runtime.memory().writeIORegister(
                    kIpuCtrl, 1u << 30u),
                "a final IPU reset should clear FIFO state");
            runtime.memory().write128(
                kIpuInFifo,
                _mm_set_epi32(4, 3, 2, 1));
            t.Equals(
                runtime.memory()
                    .toIpuDmaSnapshot()
                    .fifoQwc,
                1u,
                "a direct 128-bit IPU input write should enqueue one QW");
            t.Equals(
                runtime.memory().readIORegister(
                    kIpuCtrl) &
                    0xFu,
                1u,
                "direct FIFO writes should update IPU_CTRL.IFC");

            t.IsTrue(
                runtime.memory().writeIORegister(
                    kIpuCtrl, 1u << 30u),
                "the zero-QWC check should begin with an empty FIFO");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kToIpu + 0x10u,
                    kSecondSource),
                "zero-QWC To-IPU MADR should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kToIpu + 0x20u, 0u),
                "zero-QWC To-IPU QWC should write");
            t.IsTrue(
                runtime.memory().writeIORegister(
                    kToIpu, 0x100u),
                "zero-QWC normal To-IPU DMA should start");
            dma = runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active &&
                    dma.phase ==
                        ToIpuDmaPhase::
                            TransferPayload &&
                    dma.stall ==
                        ToIpuDmaStallReason::
                            InputFifoFull,
                "zero QWC should expand to a FIFO-limited normal transfer");
            t.Equals(
                dma.qwc, 0xFFF8u,
                "zero QWC should represent 0x10000 QW before the first eight-QW slice");
            t.Equals(
                dma.madr,
                kSecondSource + 8u * 16u,
                "the first zero-QWC slice should advance MADR");
            t.Equals(
                runtime.memory().readIORegister(
                    kToIpu + 0x20u),
                0xFFF8u,
                "the QWC register should expose the low 16 bits after underflow");
            (void)runtime.memory().writeIORegister(
                kToIpu, 0u);

            constexpr uint32_t kScratchSource =
                0x00000300u;
            std::memset(
                runtime.memory().getScratchpad() +
                    kScratchSource,
                0x9c, 16u);
            (void)runtime.memory().writeIORegister(
                kIpuCtrl, 1u << 30u);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x10u,
                0x80000000u | kScratchSource);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x20u, 1u);
            (void)runtime.memory().writeIORegister(
                kToIpu, 0x100u);
            dma = runtime.memory().toIpuDmaSnapshot();
            t.Equals(
                dma.madr,
                0x80000000u + kScratchSource +
                    16u,
                "To-IPU MADR should preserve the DMAC scratchpad flag");
            t.Equals(
                dma.fifoQwc, 1u,
                "a scratchpad-sourced QW should enter the input FIFO");
            (void)runtime.memory().writeIORegister(
                kToIpu, 0u);
        });

        tc.Run("event To-IPU source chain reproduces tag yields and reset wake", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "To-IPU chain fixture memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "To-IPU chain fixture subsystems should bind");

            constexpr uint32_t kToIpu =
                0x1000B400u;
            constexpr uint32_t kIpuCtrl =
                0x10002010u;
            constexpr uint32_t kDstat =
                0x1000E010u;
            constexpr uint32_t kEndChain =
                0x00036000u;
            constexpr uint32_t kMultiChain =
                0x00036100u;
            constexpr uint32_t kReferencePayload =
                0x00036200u;
            uint8_t *const rdram =
                runtime.memory().getRDRAM();
            const auto storeWord =
                [rdram](uint32_t address,
                        uint32_t value)
                {
                    std::memcpy(
                        rdram + address,
                        &value,
                        sizeof(value));
                };

            storeWord(kEndChain, 0x70000004u);
            std::memset(
                rdram + kEndChain + 16u,
                0x11, 4u * 16u);

            runtime.debugStartEeEventTrace(12u);
            (void)runtime.memory().writeIORegister(
                kIpuCtrl, 1u << 30u);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x30u, kEndChain);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x20u, 0u);
            (void)runtime.memory().writeIORegister(
                kToIpu, 0x104u);

            ToIpuDmaSnapshot dma =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.chainMode &&
                    !dma.normalMode &&
                    dma.phase ==
                        ToIpuDmaPhase::Finalize,
                "END/QWC4 should retain chain finalization");
            t.Equals(
                dma.tagsProcessed, 1u,
                "END should consume one DMA tag");
            t.Equals(
                static_cast<uint32_t>(dma.tagId),
                7u,
                "END should remain the visible tag ID");
            t.Equals(
                dma.fifoQwc, 4u,
                "END/QWC4 should enqueue four input QW");
            t.Equals(
                dma.madr,
                kEndChain + 5u * 16u,
                "END should expose the address after its inline payload");
            t.Equals(
                dma.tadr, kEndChain,
                "END should retain TADR on the terminal tag");
            t.Equals(
                runtime.memory().readIORegister(
                    kToIpu),
                0x70000104u,
                "END should publish tag bits, mode, and STR");

            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            auto slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacToIpu)];
            t.IsTrue(
                slot.pending,
                "END should retain a finalization deadline");
            t.Equals(
                slot.deadlineTick, 80ull,
                "END/QWC4 plus one tag should cost ten EE cycles");
            t.Equals(
                slot.device.tadr, kEndChain,
                "debug state should expose To-IPU TADR");
            t.Equals(
                slot.device.tagsProcessed, 1u,
                "debug state should expose consumed tags");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(80u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            t.IsFalse(
                runtime.memory()
                    .toIpuDmaSnapshot()
                    .active,
                "END finalization should retire the channel");
            t.Equals(
                runtime.memory().readIORegister(
                    kToIpu),
                0x70000004u,
                "completion should clear only END STR");
            t.IsTrue(
                (runtime.memory().readIORegister(
                     kDstat) &
                 (1u << 4u)) != 0u,
                "END completion should publish To-IPU D_STAT");

            (void)runtime.memory().writeIORegister(
                kDstat, 1u << 4u);
            (void)runtime.memory().writeIORegister(
                kIpuCtrl, 1u << 30u);

            storeWord(
                kMultiChain + 0u,
                0x10000000u);
            storeWord(
                kMultiChain + 16u,
                0x30000002u);
            storeWord(
                kMultiChain + 20u,
                kReferencePayload);
            storeWord(
                kMultiChain + 32u,
                0x70000003u);
            std::memset(
                rdram + kMultiChain + 48u,
                0x33, 3u * 16u);
            std::memset(
                rdram + kReferencePayload,
                0x22, 2u * 16u);

            (void)runtime.memory().writeIORegister(
                kToIpu + 0x30u, kMultiChain);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x20u, 0u);
            (void)runtime.memory().writeIORegister(
                kToIpu, 0x104u);

            dma =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.phase ==
                        ToIpuDmaPhase::FetchTag,
                "CNT/QWC0 should retain a tag rescan");
            t.Equals(
                dma.tagsProcessed, 1u,
                "the initial transition should read only CNT");
            t.Equals(
                dma.tadr, kMultiChain + 16u,
                "CNT/QWC0 should advance TADR to the next tag");
            t.Equals(
                dma.madr, kMultiChain + 16u,
                "CNT/QWC0 should publish its inline data address");
            scheduler =
                runtime.debugEeSchedulerSnapshot();
            slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacToIpu)];
            t.Equals(
                slot.deadlineTick, 160ull,
                "CNT/QWC0 should schedule a ten-cycle rescan");

            context.advanceEeCycleTicks(80u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            dma =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active && !dma.eventManaged &&
                    dma.phase ==
                        ToIpuDmaPhase::FetchTag &&
                    dma.stall ==
                        ToIpuDmaStallReason::
                            WaitingForInputRequest,
                "REF/QWC2 should become dormant after its nonterminal payload");
            t.Equals(
                dma.tagsProcessed, 2u,
                "the rescan should consume REF");
            t.Equals(
                static_cast<uint32_t>(dma.tagId),
                3u,
                "REF should remain the visible tag ID");
            t.Equals(
                dma.fifoQwc, 2u,
                "REF/QWC2 should enqueue two input QW");
            t.Equals(
                dma.madr,
                kReferencePayload + 2u * 16u,
                "REF should expose referenced-payload progress");
            t.Equals(
                dma.tadr, kMultiChain + 32u,
                "REF should advance TADR to END");
            t.Equals(
                runtime.memory().readIORegister(
                    kToIpu),
                0x30000104u,
                "dormant REF should retain tag bits and STR");
            scheduler =
                runtime.debugEeSchedulerSnapshot();
            slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacToIpu)];
            t.IsFalse(
                slot.pending,
                "a nonterminal positive payload should have no deadline");

            (void)runtime.memory().writeIORegister(
                kIpuCtrl, 1u << 30u);
            dma =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.stall ==
                        ToIpuDmaStallReason::None,
                "IPU reset should wake dormant chain progress");
            t.Equals(
                dma.fifoQwc, 0u,
                "chain reset wake should begin with an empty FIFO");
            scheduler =
                runtime.debugEeSchedulerSnapshot();
            slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacToIpu)];
            t.Equals(
                slot.deadlineTick, 192ull,
                "chain reset wake should cost four EE cycles");

            context.advanceEeCycleTicks(32u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            dma =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.phase ==
                        ToIpuDmaPhase::Finalize,
                "END/QWC3 wake should retain finalization");
            t.Equals(
                dma.tagsProcessed, 3u,
                "the wake should consume END");
            t.Equals(
                dma.fifoQwc, 3u,
                "END/QWC3 should enqueue three input QW");
            t.Equals(
                dma.madr,
                kMultiChain + 6u * 16u,
                "END should expose its completed inline payload");
            t.Equals(
                dma.tadr, kMultiChain + 32u,
                "terminal END should retain its tag address");
            scheduler =
                runtime.debugEeSchedulerSnapshot();
            slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::EeEventSource::
                            DmacToIpu)];
            t.Equals(
                slot.deadlineTick, 272ull,
                "END/QWC3 plus one tag should cost ten EE cycles");

            context.advanceEeCycleTicks(80u);
            runtime.serviceEeEventsAtBlockBoundary(
                rdram, &context);
            t.IsFalse(
                runtime.memory()
                    .toIpuDmaSnapshot()
                    .active,
                "the multi-tag chain should retire");
            t.Equals(
                runtime.memory().readIORegister(
                    kToIpu),
                0x70000004u,
                "multi-tag completion should preserve END and mode");
            t.IsTrue(
                (runtime.memory().readIORegister(
                     kDstat) &
                 (1u << 4u)) != 0u,
                "multi-tag completion should publish D_STAT");

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(6u),
                "the chain oracle should expose four channel services and two publications");
            if (trace.entries.size() == 6u)
            {
                const std::array<
                    ps2x::timing::EeEventSource, 6u>
                    expectedSources = {
                        ps2x::timing::EeEventSource::
                            DmacToIpu,
                        ps2x::timing::EeEventSource::
                            DmacCompletion,
                        ps2x::timing::EeEventSource::
                            DmacToIpu,
                        ps2x::timing::EeEventSource::
                            DmacToIpu,
                        ps2x::timing::EeEventSource::
                            DmacToIpu,
                        ps2x::timing::EeEventSource::
                            DmacCompletion,
                    };
                const std::array<uint64_t, 6u>
                    expectedTicks = {
                        80ull, 80ull, 160ull,
                        192ull, 272ull, 272ull};
                for (size_t index = 0u;
                     index < trace.entries.size();
                     ++index)
                {
                    t.IsTrue(
                        trace.entries[index].source ==
                            expectedSources[index],
                        "chain services should retain source order");
                    t.Equals(
                        trace.entries[index].serviceTick,
                        expectedTicks[index],
                        "chain services should retain exact ticks");
                }
            }
        });

        tc.Run("To-IPU source-chain tags retain IPU-specific address rules", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "To-IPU tag fixture memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "To-IPU tag fixture subsystems should bind");

            constexpr uint32_t kToIpu =
                0x1000B400u;
            constexpr uint32_t kIpuCtrl =
                0x10002010u;
            constexpr uint32_t kTagBase =
                0x00036400u;
            constexpr uint32_t kPayloadBase =
                0x00036800u;
            uint8_t *const rdram =
                runtime.memory().getRDRAM();
            const auto storeWord =
                [rdram](uint32_t address,
                        uint32_t value)
                {
                    std::memcpy(
                        rdram + address,
                        &value,
                        sizeof(value));
                };

            struct TagCase
            {
                uint8_t id;
                bool terminal;
            };
            constexpr std::array<TagCase, 8u>
                cases = {{
                    {0u, true},  // REFE
                    {1u, false}, // CNT
                    {2u, false}, // NEXT
                    {3u, false}, // REF
                    {4u, false}, // REFS
                    {5u, true},  // CALL
                    {6u, true},  // RET
                    {7u, true},  // END
                }};

            for (const TagCase &tagCase : cases)
            {
                const uint32_t tagAddress =
                    kTagBase +
                    static_cast<uint32_t>(
                        tagCase.id) *
                        0x40u;
                const uint32_t payloadAddress =
                    kPayloadBase +
                    static_cast<uint32_t>(
                        tagCase.id) *
                        0x20u;
                storeWord(
                    tagAddress,
                    (static_cast<uint32_t>(
                         tagCase.id)
                     << 28u) |
                        1u);
                storeWord(
                    tagAddress + 4u,
                    payloadAddress);
                std::memset(
                    rdram + tagAddress + 16u,
                    0x40 + tagCase.id, 16u);
                std::memset(
                    rdram + payloadAddress,
                    0x60 + tagCase.id, 16u);

                (void)runtime.memory()
                    .writeIORegister(
                        kIpuCtrl, 1u << 30u);
                (void)runtime.memory()
                    .writeIORegister(
                        kToIpu + 0x30u,
                        tagAddress);
                (void)runtime.memory()
                    .writeIORegister(
                        kToIpu + 0x20u, 0u);
                (void)runtime.memory()
                    .writeIORegister(
                        kToIpu, 0x104u);

                const ToIpuDmaSnapshot dma =
                    runtime.memory()
                        .toIpuDmaSnapshot();
                t.IsTrue(
                    dma.active && dma.chainMode &&
                        dma.tagsProcessed == 1u &&
                        dma.tagId == tagCase.id,
                    "each source-chain ID should consume exactly one tag");
                t.Equals(
                    dma.fifoQwc, 1u,
                    "each one-QW tag should enqueue only its payload");
                t.Equals(
                    dma.qwc, 0u,
                    "each one-QW tag should consume its payload");
                t.IsTrue(
                    dma.phase ==
                        (tagCase.terminal
                             ? ToIpuDmaPhase::
                                   Finalize
                             : ToIpuDmaPhase::
                                   FetchTag),
                    "IPU tag IDs should retain their terminal classification");
                t.IsTrue(
                    dma.endAfterPayload ==
                        tagCase.terminal,
                    "IPU tag terminal state should remain observable");
                t.IsTrue(
                    dma.eventManaged ==
                        tagCase.terminal,
                    "only a terminal positive payload should retain a deadline");
                t.IsTrue(
                    dma.stall ==
                        (tagCase.terminal
                             ? ToIpuDmaStallReason::
                                   None
                             : ToIpuDmaStallReason::
                                   WaitingForInputRequest),
                    "nonterminal positive payloads should sleep for another IPU request");

                uint32_t expectedTadr =
                    tagAddress;
                uint32_t expectedMadr =
                    payloadAddress + 16u;
                switch (tagCase.id)
                {
                case 0u: // REFE
                case 3u: // REF
                case 4u: // REFS
                    expectedTadr =
                        tagAddress + 16u;
                    break;
                case 1u: // CNT
                    expectedTadr =
                        tagAddress + 32u;
                    expectedMadr =
                        tagAddress + 32u;
                    break;
                case 2u: // NEXT
                    expectedTadr =
                        payloadAddress;
                    expectedMadr =
                        tagAddress + 32u;
                    break;
                case 5u: // CALL
                case 6u: // RET
                    // IPU chains terminate instead of applying the
                    // source-chain address stack.
                    break;
                case 7u: // END
                    expectedMadr =
                        tagAddress + 32u;
                    break;
                default:
                    break;
                }
                t.Equals(
                    dma.tadr, expectedTadr,
                    "the tag should publish its IPU-specific TADR transition");
                t.Equals(
                    dma.madr, expectedMadr,
                    "the tag should publish its IPU-specific MADR transition");

                const PS2Runtime::DebugEeScheduler
                    scheduler =
                        runtime
                            .debugEeSchedulerSnapshot();
                const auto slot =
                    scheduler.slots[ps2x::timing::
                        eeEventSourceIndex(
                            ps2x::timing::
                                EeEventSource::
                                    DmacToIpu)];
                t.IsTrue(
                    slot.pending ==
                        tagCase.terminal,
                    "terminal tag classification should own scheduler visibility");
                if (tagCase.terminal)
                {
                    t.Equals(
                        slot.deadlineTick, 80ull,
                        "one tag and one QW should use the ten-cycle minimum");
                }
                (void)runtime.memory()
                    .writeIORegister(kToIpu, 0u);
            }

            // dmaGetAddr() honors the source-chain SPR bit for both the tag
            // fetch and a referenced payload. Preserve the flag in published
            // addresses instead of normalizing it to an RDRAM offset.
            constexpr uint32_t kScratchTagOffset =
                0x00000400u;
            constexpr uint32_t kScratchTag =
                0x80000000u | kScratchTagOffset;
            uint8_t *const scratchpad =
                runtime.memory().getScratchpad();
            const auto storeScratchWord =
                [scratchpad](uint32_t offset,
                             uint32_t value)
                {
                    std::memcpy(
                        scratchpad + offset,
                        &value,
                        sizeof(value));
                };
            storeScratchWord(
                kScratchTagOffset, 0x70000001u);
            std::memset(
                scratchpad + kScratchTagOffset + 16u,
                0x83, 16u);
            (void)runtime.memory().writeIORegister(
                kIpuCtrl, 1u << 30u);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x30u, kScratchTag);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x20u, 0u);
            (void)runtime.memory().writeIORegister(
                kToIpu, 0x104u);
            ToIpuDmaSnapshot dma =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.phase ==
                        ToIpuDmaPhase::Finalize &&
                    dma.tagsProcessed == 1u,
                "a scratchpad-resident END tag should transfer normally");
            t.Equals(
                dma.tadr, kScratchTag,
                "a terminal scratchpad tag should retain its flagged TADR");
            t.Equals(
                dma.madr, kScratchTag + 32u,
                "an inline scratchpad payload should preserve flagged MADR progress");
            t.Equals(
                dma.fifoQwc, 1u,
                "a scratchpad tag should enqueue its inline payload");
            (void)runtime.memory().writeIORegister(
                kToIpu, 0u);

            constexpr uint32_t
                kScratchReferenceOffset = 0x00000300u;
            constexpr uint32_t
                kScratchReference =
                    0x80000000u |
                    kScratchReferenceOffset;
            constexpr uint32_t kScratchReferenceTag =
                0x00036600u;
            storeWord(
                kScratchReferenceTag,
                0x00000001u);
            storeWord(
                kScratchReferenceTag + 4u,
                kScratchReference);
            std::memset(
                scratchpad + kScratchReferenceOffset,
                0xa7, 16u);
            (void)runtime.memory().writeIORegister(
                kIpuCtrl, 1u << 30u);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x30u,
                kScratchReferenceTag);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x20u, 0u);
            (void)runtime.memory().writeIORegister(
                kToIpu, 0x104u);
            dma =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.phase ==
                        ToIpuDmaPhase::Finalize,
                "a REFE tag should accept a flagged scratchpad payload");
            t.Equals(
                dma.tadr,
                kScratchReferenceTag + 16u,
                "REFE should advance its RDRAM TADR independently");
            t.Equals(
                dma.madr,
                kScratchReference + 16u,
                "the scratchpad payload should preserve flagged MADR progress");
            t.Equals(
                dma.fifoQwc, 1u,
                "the referenced scratchpad QW should enter the input FIFO");
            (void)runtime.memory().writeIORegister(
                kToIpu, 0u);

            // Physical DMAC reads above exposed RDRAM and below the MMIO
            // window use the zero-read page. A zero tag is REFE/QWC0 and
            // therefore reaches a delayed terminal boundary without a bus
            // fault.
            constexpr uint32_t kZeroReadTag =
                0x03000000u;
            (void)runtime.memory().writeIORegister(
                kIpuCtrl, 1u << 30u);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x30u, kZeroReadTag);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x20u, 0u);
            (void)runtime.memory().writeIORegister(
                kToIpu, 0x104u);
            dma =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.phase ==
                        ToIpuDmaPhase::Finalize &&
                    dma.tagsProcessed == 1u &&
                    dma.tagId == 0u,
                "an unpopulated physical tag should decode from the DMAC zero-read page");
            t.Equals(
                dma.tadr, kZeroReadTag + 16u,
                "zero-read REFE should advance TADR");
            t.Equals(
                dma.madr, 0u,
                "the zero-read tag should publish its zero address field");
            t.Equals(
                dma.fifoQwc, 0u,
                "zero-read REFE/QWC0 should not enqueue payload");
            (void)runtime.memory().writeIORegister(
                kToIpu, 0u);

            // TTE is ignored by the physical IPU channel. TIE plus IRQ makes
            // an otherwise nonterminal REF payload terminal.
            constexpr uint32_t kIrqTag =
                0x00036700u;
            constexpr uint32_t kIrqPayload =
                0x00036A00u;
            storeWord(kIrqTag, 0xB0000001u);
            storeWord(kIrqTag + 4u, kIrqPayload);
            std::memset(
                rdram + kIrqPayload, 0x9a, 16u);
            (void)runtime.memory().writeIORegister(
                kIpuCtrl, 1u << 30u);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x30u, kIrqTag);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x20u, 0u);
            (void)runtime.memory().writeIORegister(
                kToIpu,
                0x104u | (1u << 6u) |
                    (1u << 7u));
            dma =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.tagIrq && dma.endAfterPayload &&
                    dma.phase ==
                        ToIpuDmaPhase::Finalize,
                "TIE plus an IRQ REF should become terminal");
            t.Equals(
                dma.fifoQwc, 1u,
                "TTE should not enqueue the DMA tag itself");
            t.Equals(
                runtime.memory().readIORegister(
                    kToIpu),
                0xB00001C4u,
                "IRQ tag bits and TTE/TIE should remain visible");
            (void)runtime.memory().writeIORegister(
                kToIpu, 0u);

            // A nonzero chain QWC resumes the tag already latched in CHCR and
            // does not fetch or charge another tag.
            constexpr uint32_t kResumePayload =
                0x00036B00u;
            constexpr uint32_t kResumeTadr =
                0x00036C00u;
            std::memset(
                rdram + kResumePayload,
                0xbc, 2u * 16u);
            (void)runtime.memory().writeIORegister(
                kIpuCtrl, 1u << 30u);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x10u,
                kResumePayload);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x20u, 2u);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x30u, kResumeTadr);
            (void)runtime.memory().writeIORegister(
                kToIpu, 0x70000104u);
            dma =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.phase ==
                        ToIpuDmaPhase::Finalize &&
                    dma.endAfterPayload,
                "a resumed END payload should retain finalization");
            t.Equals(
                dma.tagsProcessed, 0u,
                "resumed payload must not fetch another tag");
            t.Equals(
                dma.tadr, kResumeTadr,
                "resumed END should retain the latched TADR");
            t.Equals(
                dma.madr,
                kResumePayload + 2u * 16u,
                "resumed END should advance only its payload MADR");
            const PS2Runtime::DebugEeScheduler
                scheduler =
                    runtime.debugEeSchedulerSnapshot();
            const auto slot =
                scheduler.slots[ps2x::timing::
                    eeEventSourceIndex(
                        ps2x::timing::
                            EeEventSource::
                                DmacToIpu)];
            t.Equals(
                slot.deadlineTick, 64ull,
                "resumed payload should omit the tag-cycle charge");
            (void)runtime.memory().writeIORegister(
                kToIpu, 0u);

            // A full FIFO withdraws the data request before a new chain may
            // expose its tag. Reset re-arms the untouched descriptor.
            constexpr uint32_t kIpuInFifo =
                0x10007010u;
            constexpr uint32_t kBlockedTag =
                0x00036D00u;
            storeWord(kBlockedTag, 0x70000001u);
            std::memset(
                rdram + kBlockedTag + 16u,
                0xdd, 16u);
            (void)runtime.memory().writeIORegister(
                kIpuCtrl, 1u << 30u);
            for (uint32_t index = 0u;
                 index < 8u; ++index)
            {
                runtime.memory().write128(
                    kIpuInFifo,
                    _mm_set1_epi32(
                        static_cast<int>(index)));
            }
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x30u, kBlockedTag);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x20u, 0u);
            (void)runtime.memory().writeIORegister(
                kToIpu, 0x104u);
            dma =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active && !dma.eventManaged &&
                    dma.phase ==
                        ToIpuDmaPhase::FetchTag &&
                    dma.stall ==
                        ToIpuDmaStallReason::
                            InputFifoFull,
                "a full FIFO should block chain tag visibility");
            t.Equals(
                dma.tagsProcessed, 0u,
                "a full FIFO must not consume the blocked tag");
            PS2Runtime::DebugEeScheduler
                blockedScheduler =
                    runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                blockedScheduler
                    .slots[ps2x::timing::
                        eeEventSourceIndex(
                            ps2x::timing::
                                EeEventSource::
                                    DmacToIpu)]
                    .pending,
                "a full FIFO should leave the untouched chain dormant");

            (void)runtime.memory().writeIORegister(
                kIpuCtrl, 1u << 30u);
            dma =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                dma.active && dma.eventManaged &&
                    dma.tagsProcessed == 0u,
                "reset should re-arm the blocked tag without consuming it early");
            blockedScheduler =
                runtime.debugEeSchedulerSnapshot();
            t.Equals(
                blockedScheduler
                    .slots[ps2x::timing::
                        eeEventSourceIndex(
                            ps2x::timing::
                                EeEventSource::
                                    DmacToIpu)]
                    .deadlineTick,
                32ull,
                "reset should wake the blocked chain after four EE cycles");
            (void)runtime.memory().writeIORegister(
                kToIpu, 0u);
        });

        tc.Run("To-IPU cancellation invalidates scheduled generations", [](TestCase &t)
        {
            PS2Runtime runtime;
            t.IsTrue(
                runtime.memory().initialize(),
                "To-IPU cancellation memory should initialize");
            t.IsTrue(
                runtime.syncCoreSubsystems(),
                "To-IPU cancellation subsystems should bind");

            constexpr uint32_t kToIpu =
                0x1000B400u;
            constexpr uint32_t kSource =
                0x00035400u;
            constexpr uint32_t kDstat =
                0x1000E010u;
            std::memset(
                runtime.memory().getRDRAM() +
                    kSource,
                0x5a, 16u);

            runtime.debugStartEeEventTrace(4u);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x10u, kSource);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x20u, 1u);
            (void)runtime.memory().writeIORegister(
                kToIpu, 0x100u);
            const uint64_t cancelledGeneration =
                runtime.memory()
                    .toIpuDmaSnapshot()
                    .transfer.generation;
            (void)runtime.memory().writeIORegister(
                kToIpu, 0u);

            t.IsFalse(
                runtime.memory()
                    .toIpuDmaSnapshot()
                    .active,
                "clearing STR should cancel retained To-IPU work");
            PS2Runtime::DebugEeScheduler scheduler =
                runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                scheduler
                    .slots[ps2x::timing::
                        eeEventSourceIndex(
                            ps2x::timing::
                                EeEventSource::
                                    DmacToIpu)]
                    .pending,
                "clearing STR should cancel the To-IPU deadline");

            (void)runtime.memory().writeIORegister(
                0x10002010u, 1u << 30u);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x10u, kSource);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x20u, 1u);
            (void)runtime.memory().writeIORegister(
                kToIpu, 0x100u);
            const ToIpuDmaSnapshot replacement =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                replacement.transfer.generation !=
                    cancelledGeneration,
                "replacement work should use a new generation");

            R5900Context &context = runtime.cpu();
            context.advanceEeCycleTicks(64u);
            runtime.serviceEeEventsAtBlockBoundary(
                runtime.memory().getRDRAM(), &context);
            t.IsFalse(
                runtime.memory()
                    .toIpuDmaSnapshot()
                    .active,
                "only the replacement should finalize");
            t.IsTrue(
                (runtime.memory().readIORegister(
                     kDstat) &
                 (1u << 4u)) != 0u,
                "only live replacement work should publish completion");

            const PS2Runtime::DebugEeEventTrace trace =
                runtime.debugEeEventTraceSnapshot(true);
            t.Equals(
                trace.entries.size(),
                static_cast<size_t>(2u),
                "only replacement finalization and publication should dispatch");
            if (trace.entries.size() == 2u)
            {
                t.IsTrue(
                    trace.entries[0].source ==
                        ps2x::timing::EeEventSource::
                            DmacToIpu,
                    "replacement device service should dispatch first");
                t.IsTrue(
                    trace.entries[1].source ==
                        ps2x::timing::EeEventSource::
                            DmacCompletion,
                    "replacement completion should publish second");
            }

            (void)runtime.memory().writeIORegister(
                0x10002010u, 1u << 30u);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x10u, kSource);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x20u, 1u);
            (void)runtime.memory().writeIORegister(
                kToIpu, 0x100u);
            runtime.resetEeTiming(&context);
            t.IsFalse(
                runtime.memory()
                    .toIpuDmaSnapshot()
                    .active,
                "a timing reset should discard retained To-IPU work");
            scheduler = runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                scheduler
                    .slots[ps2x::timing::
                        eeEventSourceIndex(
                            ps2x::timing::
                                EeEventSource::
                                    DmacToIpu)]
                    .pending,
                "a timing reset should clear the To-IPU slot");

            (void)runtime.memory().writeIORegister(
                0x10002010u, 1u << 30u);
            (void)runtime.memory().writeIORegister(
                0x1000E000u, 0u);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x10u, kSource);
            (void)runtime.memory().writeIORegister(
                kToIpu + 0x20u, 1u);
            (void)runtime.memory().writeIORegister(
                kToIpu, 0x100u);
            const ToIpuDmaSnapshot disabled =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                disabled.active &&
                    !disabled.eventManaged &&
                    disabled.stall ==
                        ToIpuDmaStallReason::
                            DmacDisabled,
                "a start while DMAC is disabled should remain queued without a deadline");
            scheduler = runtime.debugEeSchedulerSnapshot();
            t.IsFalse(
                scheduler
                    .slots[ps2x::timing::
                        eeEventSourceIndex(
                            ps2x::timing::
                                EeEventSource::
                                    DmacToIpu)]
                    .pending,
                "disabled To-IPU work should remain dormant");

            (void)runtime.memory().writeIORegister(
                0x1000E000u, 1u);
            const ToIpuDmaSnapshot reenabled =
                runtime.memory().toIpuDmaSnapshot();
            t.IsTrue(
                reenabled.active &&
                    reenabled.eventManaged &&
                    reenabled.stall ==
                        ToIpuDmaStallReason::None,
                "reenabling DMAC should wake queued To-IPU work");
            scheduler = runtime.debugEeSchedulerSnapshot();
            t.IsTrue(
                scheduler
                    .slots[ps2x::timing::
                        eeEventSourceIndex(
                            ps2x::timing::
                                EeEventSource::
                                    DmacToIpu)]
                    .pending,
                "reenabled To-IPU work should acquire a deadline");
            t.Equals(
                scheduler
                    .slots[ps2x::timing::
                        eeEventSourceIndex(
                            ps2x::timing::
                                EeEventSource::
                                    DmacToIpu)]
                    .deadlineTick,
                scheduler.currentTick,
                "reenabling DMAC should make queued To-IPU work immediately eligible");
            (void)runtime.memory().writeIORegister(
                kToIpu, 0u);
        });

        tc.Run("sceVif1PkReset preserves the packet base pointer and clears open tag state", [](TestCase &t)
        {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};

            constexpr uint32_t kStateAddr = 0x00100200u;
            constexpr uint32_t kBaseAddr = 0x00101000u;

            setRegU32(ctx, 4, kStateAddr);
            setRegU32(ctx, 5, kBaseAddr);
            ps2_stubs::sceVif1PkInit(rdram.data(), &ctx, nullptr);

            const uint32_t dirtyCurrent = kBaseAddr + 0x40u;
            const uint32_t dirtyPending = 0x12345678u;
            const uint32_t dirtyDirectOpen = 0x00ABCDEFu;
            const uint32_t dirtyGifOpen = 0x00112233u;
            std::memcpy(rdram.data() + kStateAddr + 0u, &dirtyCurrent, sizeof(dirtyCurrent));
            std::memcpy(rdram.data() + kStateAddr + 8u, &dirtyPending, sizeof(dirtyPending));
            std::memcpy(rdram.data() + kStateAddr + 12u, &dirtyDirectOpen, sizeof(dirtyDirectOpen));
            std::memcpy(rdram.data() + kStateAddr + 20u, &dirtyGifOpen, sizeof(dirtyGifOpen));

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kStateAddr);
            ps2_stubs::sceVif1PkReset(rdram.data(), &ctx, nullptr);

            uint32_t current = 0u;
            uint32_t base = 0u;
            uint32_t pending = 0u;
            uint32_t directOpen = 0u;
            uint32_t gifOpen = 0u;
            std::memcpy(&current, rdram.data() + kStateAddr + 0u, sizeof(current));
            std::memcpy(&base, rdram.data() + kStateAddr + 4u, sizeof(base));
            std::memcpy(&pending, rdram.data() + kStateAddr + 8u, sizeof(pending));
            std::memcpy(&directOpen, rdram.data() + kStateAddr + 12u, sizeof(directOpen));
            std::memcpy(&gifOpen, rdram.data() + kStateAddr + 20u, sizeof(gifOpen));

            t.Equals(current, kBaseAddr, "sceVif1PkReset should restore current pointer to the packet base");
            t.Equals(base, kBaseAddr, "sceVif1PkReset should preserve the packet base pointer");
            t.Equals(pending, 0u, "sceVif1PkReset should clear pending count tracking");
            t.Equals(directOpen, 0u, "sceVif1PkReset should clear direct-code open state");
            t.Equals(gifOpen, 0u, "sceVif1PkReset should clear GIF-tag open state");
            t.Equals(::getRegU32(&ctx, 2), kBaseAddr, "sceVif1PkReset should return the packet base pointer");
        });

        tc.Run("sceVif1PkCloseDirectCode encodes DIRECT length in qwords", [](TestCase &t)
        {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0u);
            R5900Context ctx{};

            constexpr uint32_t kStateAddr = 0x00100400u;
            constexpr uint32_t kBaseAddr = 0x00102000u;

            setRegU32(ctx, 4, kStateAddr);
            setRegU32(ctx, 5, kBaseAddr);
            ps2_stubs::sceVif1PkInit(rdram.data(), &ctx, nullptr);

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kStateAddr);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::sceVif1PkCnt(rdram.data(), &ctx, nullptr);

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kStateAddr);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::sceVif1PkOpenDirectCode(rdram.data(), &ctx, nullptr);

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kStateAddr);
            setRegU32(ctx, 5, 4u); // reserve one qword worth of GIF payload
            ps2_stubs::sceVif1PkReserve(rdram.data(), &ctx, nullptr);

            std::memset(&ctx, 0, sizeof(ctx));
            setRegU32(ctx, 4, kStateAddr);
            ps2_stubs::sceVif1PkCloseDirectCode(rdram.data(), &ctx, nullptr);

            uint32_t directCmd = 0u;
            std::memcpy(&directCmd, rdram.data() + kBaseAddr + 12u, sizeof(directCmd));
            t.Equals(directCmd, 0x50000001u, "sceVif1PkCloseDirectCode should store a 1-QW DIRECT length");
        });
    });
}
