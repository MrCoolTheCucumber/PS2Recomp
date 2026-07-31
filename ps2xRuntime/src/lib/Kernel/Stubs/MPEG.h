#pragma once

#include "ps2_stubs.h"

#include <memory>

namespace ps2_stubs
{
    class MpegRuntimeState
    {
    public:
        struct Impl;

        MpegRuntimeState();
        ~MpegRuntimeState();

        Impl &implementation() noexcept;
        const Impl &implementation() const noexcept;

        MpegRuntimeState(
            const MpegRuntimeState &) = delete;
        MpegRuntimeState &operator=(
            const MpegRuntimeState &) = delete;

    private:
        std::unique_ptr<Impl> m_impl;
    };

    void resetMpegStubState(PS2Runtime *runtime);
    void notifyMpegCdStreamStart(PS2Runtime *runtime);
    void notifyMpegCdStreamEof(PS2Runtime *runtime);
    void sceMpegFlush(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegAddBs(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegAddCallback(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegAddStrCallback(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegClearRefBuff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegCreate(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegDelete(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegDemuxPss(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegDemuxPssRing(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegDispCenterOffX(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegDispCenterOffY(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegDispHeight(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegDispWidth(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegGetDecodeMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegGetPicture(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegGetPictureRAW8(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegGetPictureRAW8xy(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegInit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegIsEnd(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegIsRefBuffEmpty(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegReset(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegResetDefaultPtsGap(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegSetDecodeMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegSetDefaultPtsGap(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
    void sceMpegSetImageBuff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime);
}
