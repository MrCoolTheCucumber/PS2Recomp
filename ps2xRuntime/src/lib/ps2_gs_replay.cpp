#include "runtime/ps2_gs_replay.h"

#include <algorithm>
#include <bit>
#include <string_view>
#include <utility>

namespace
{
    constexpr std::array<uint8_t, 8> kReplayStateMagic{
        'P', 'S', '2', 'G', 'S', 'R', 'P', 'L'};

    void setError(std::string *error, std::string_view message)
    {
        if (error)
            *error = message;
    }

    bool validateReplayState(
        const GsReplayState &state,
        std::string *error)
    {
        if (static_cast<uint32_t>(state.prim.type) >
            static_cast<uint32_t>(GS_PRIM_SPRITE))
        {
            setError(error, "invalid GS replay primitive type");
            return false;
        }
        if (state.scanMask > 3u)
        {
            setError(error, "invalid GS replay scan mask");
            return false;
        }
        if (state.trxdir > 3u)
        {
            setError(error, "invalid GS replay transfer direction");
            return false;
        }
        if (state.vertexCount < 0 ||
            state.vertexCount >
                static_cast<int32_t>(GS_REPLAY_VERTEX_QUEUE_CAPACITY))
        {
            setError(error, "invalid GS replay vertex count");
            return false;
        }
        if (state.rasterizer.feedbackSnapshotValid &&
            state.rasterizer.feedbackVram.size() != GS_REPLAY_VRAM_SIZE)
        {
            setError(error, "valid GS feedback snapshot is not exactly 4 MiB");
            return false;
        }
        if (!state.rasterizer.feedbackSnapshotValid &&
            !state.rasterizer.feedbackVram.empty())
        {
            setError(error, "invalid GS feedback snapshot contains VRAM bytes");
            return false;
        }
        for (uint8_t valid : state.clutCacheValid)
        {
            if (valid > 1u)
            {
                setError(error, "invalid GS CLUT cache-valid flag");
                return false;
            }
        }
        return true;
    }

    class StateWriter
    {
    public:
        explicit StateWriter(std::vector<uint8_t> &output)
            : m_output(output)
        {
        }

        void u8(uint8_t value)
        {
            m_output.push_back(value);
        }

        void boolean(bool value)
        {
            u8(value ? 1u : 0u);
        }

        void u16(uint16_t value)
        {
            u8(static_cast<uint8_t>(value));
            u8(static_cast<uint8_t>(value >> 8u));
        }

        void u32(uint32_t value)
        {
            for (uint32_t shift = 0u; shift < 32u; shift += 8u)
                u8(static_cast<uint8_t>(value >> shift));
        }

        void i32(int32_t value)
        {
            u32(std::bit_cast<uint32_t>(value));
        }

        void u64(uint64_t value)
        {
            for (uint32_t shift = 0u; shift < 64u; shift += 8u)
                u8(static_cast<uint8_t>(value >> shift));
        }

        void f32(float value)
        {
            u32(std::bit_cast<uint32_t>(value));
        }

        void f64(double value)
        {
            u64(std::bit_cast<uint64_t>(value));
        }

        void bytes(std::span<const uint8_t> bytes)
        {
            m_output.insert(m_output.end(), bytes.begin(), bytes.end());
        }

    private:
        std::vector<uint8_t> &m_output;
    };

    class StateReader
    {
    public:
        explicit StateReader(std::span<const uint8_t> input)
            : m_input(input)
        {
        }

        bool u8(uint8_t &value)
        {
            if (m_offset >= m_input.size())
                return false;
            value = m_input[m_offset++];
            return true;
        }

        bool boolean(bool &value)
        {
            uint8_t encoded = 0u;
            if (!u8(encoded) || encoded > 1u)
                return false;
            value = encoded != 0u;
            return true;
        }

        bool u16(uint16_t &value)
        {
            uint8_t bytes[2]{};
            if (!u8(bytes[0]) || !u8(bytes[1]))
                return false;
            value = static_cast<uint16_t>(bytes[0]) |
                    (static_cast<uint16_t>(bytes[1]) << 8u);
            return true;
        }

        bool u32(uint32_t &value)
        {
            value = 0u;
            for (uint32_t shift = 0u; shift < 32u; shift += 8u)
            {
                uint8_t byte = 0u;
                if (!u8(byte))
                    return false;
                value |= static_cast<uint32_t>(byte) << shift;
            }
            return true;
        }

        bool i32(int32_t &value)
        {
            uint32_t encoded = 0u;
            if (!u32(encoded))
                return false;
            value = std::bit_cast<int32_t>(encoded);
            return true;
        }

        bool u64(uint64_t &value)
        {
            value = 0u;
            for (uint32_t shift = 0u; shift < 64u; shift += 8u)
            {
                uint8_t byte = 0u;
                if (!u8(byte))
                    return false;
                value |= static_cast<uint64_t>(byte) << shift;
            }
            return true;
        }

        bool f32(float &value)
        {
            uint32_t encoded = 0u;
            if (!u32(encoded))
                return false;
            value = std::bit_cast<float>(encoded);
            return true;
        }

        bool f64(double &value)
        {
            uint64_t encoded = 0u;
            if (!u64(encoded))
                return false;
            value = std::bit_cast<double>(encoded);
            return true;
        }

        bool bytes(std::span<uint8_t> output)
        {
            if (output.size() > m_input.size() - m_offset)
                return false;
            std::copy_n(
                m_input.begin() + static_cast<ptrdiff_t>(m_offset),
                output.size(),
                output.begin());
            m_offset += output.size();
            return true;
        }

        [[nodiscard]] bool atEnd() const noexcept
        {
            return m_offset == m_input.size();
        }

    private:
        std::span<const uint8_t> m_input;
        size_t m_offset = 0u;
    };

    void writeFrame(StateWriter &writer, const GSFrameReg &frame)
    {
        writer.u32(frame.fbp);
        writer.u32(frame.fbw);
        writer.u8(frame.psm);
        writer.u32(frame.fbmsk);
    }

    bool readFrame(StateReader &reader, GSFrameReg &frame)
    {
        return reader.u32(frame.fbp) &&
               reader.u32(frame.fbw) &&
               reader.u8(frame.psm) &&
               reader.u32(frame.fbmsk);
    }

    void writeContext(StateWriter &writer, const GSContext &context)
    {
        writeFrame(writer, context.frame);
        writer.u16(context.scissor.x0);
        writer.u16(context.scissor.x1);
        writer.u16(context.scissor.y0);
        writer.u16(context.scissor.y1);
        writer.u32(context.tex0.tbp0);
        writer.u8(context.tex0.tbw);
        writer.u8(context.tex0.psm);
        writer.u8(context.tex0.tw);
        writer.u8(context.tex0.th);
        writer.u8(context.tex0.tcc);
        writer.u8(context.tex0.tfx);
        writer.u32(context.tex0.cbp);
        writer.u8(context.tex0.cpsm);
        writer.u8(context.tex0.csm);
        writer.u8(context.tex0.csa);
        writer.u8(context.tex0.cld);
        writer.u16(context.xyoffset.ofx);
        writer.u16(context.xyoffset.ofy);
        writer.u32(context.zbuf.zbp);
        writer.u8(context.zbuf.psm);
        writer.boolean(context.zbuf.zmask);
        writer.u64(context.tex1);
        writer.u64(context.miptbp1);
        writer.u64(context.miptbp2);
        writer.u64(context.clamp);
        writer.u64(context.alpha);
        writer.u64(context.test);
        writer.u64(context.fba);
    }

    bool readContext(StateReader &reader, GSContext &context)
    {
        return readFrame(reader, context.frame) &&
               reader.u16(context.scissor.x0) &&
               reader.u16(context.scissor.x1) &&
               reader.u16(context.scissor.y0) &&
               reader.u16(context.scissor.y1) &&
               reader.u32(context.tex0.tbp0) &&
               reader.u8(context.tex0.tbw) &&
               reader.u8(context.tex0.psm) &&
               reader.u8(context.tex0.tw) &&
               reader.u8(context.tex0.th) &&
               reader.u8(context.tex0.tcc) &&
               reader.u8(context.tex0.tfx) &&
               reader.u32(context.tex0.cbp) &&
               reader.u8(context.tex0.cpsm) &&
               reader.u8(context.tex0.csm) &&
               reader.u8(context.tex0.csa) &&
               reader.u8(context.tex0.cld) &&
               reader.u16(context.xyoffset.ofx) &&
               reader.u16(context.xyoffset.ofy) &&
               reader.u32(context.zbuf.zbp) &&
               reader.u8(context.zbuf.psm) &&
               reader.boolean(context.zbuf.zmask) &&
               reader.u64(context.tex1) &&
               reader.u64(context.miptbp1) &&
               reader.u64(context.miptbp2) &&
               reader.u64(context.clamp) &&
               reader.u64(context.alpha) &&
               reader.u64(context.test) &&
               reader.u64(context.fba);
    }

    void writePrimitive(StateWriter &writer, const GSPrimReg &primitive)
    {
        writer.u8(static_cast<uint8_t>(primitive.type));
        writer.boolean(primitive.iip);
        writer.boolean(primitive.tme);
        writer.boolean(primitive.fge);
        writer.boolean(primitive.abe);
        writer.boolean(primitive.aa1);
        writer.boolean(primitive.fst);
        writer.boolean(primitive.ctxt);
        writer.boolean(primitive.fix);
    }

    bool readPrimitive(StateReader &reader, GSPrimReg &primitive)
    {
        uint8_t type = 0u;
        if (!reader.u8(type) || type > GS_PRIM_SPRITE)
            return false;
        primitive.type = static_cast<GSPrimType>(type);
        return reader.boolean(primitive.iip) &&
               reader.boolean(primitive.tme) &&
               reader.boolean(primitive.fge) &&
               reader.boolean(primitive.abe) &&
               reader.boolean(primitive.aa1) &&
               reader.boolean(primitive.fst) &&
               reader.boolean(primitive.ctxt) &&
               reader.boolean(primitive.fix);
    }

    void writeVertex(StateWriter &writer, const GSVertex &vertex)
    {
        writer.f32(vertex.x);
        writer.f32(vertex.y);
        writer.f64(vertex.z);
        writer.u8(vertex.r);
        writer.u8(vertex.g);
        writer.u8(vertex.b);
        writer.u8(vertex.a);
        writer.f32(vertex.q);
        writer.f32(vertex.s);
        writer.f32(vertex.t);
        writer.u16(vertex.u);
        writer.u16(vertex.v);
        writer.u8(vertex.fog);
        writer.u16(vertex.x12_4);
        writer.u16(vertex.y12_4);
        writer.u32(vertex.zInteger);
    }

    bool readVertex(StateReader &reader, GSVertex &vertex)
    {
        return reader.f32(vertex.x) &&
               reader.f32(vertex.y) &&
               reader.f64(vertex.z) &&
               reader.u8(vertex.r) &&
               reader.u8(vertex.g) &&
               reader.u8(vertex.b) &&
               reader.u8(vertex.a) &&
               reader.f32(vertex.q) &&
               reader.f32(vertex.s) &&
               reader.f32(vertex.t) &&
               reader.u16(vertex.u) &&
               reader.u16(vertex.v) &&
               reader.u8(vertex.fog) &&
               reader.u16(vertex.x12_4) &&
               reader.u16(vertex.y12_4) &&
               reader.u32(vertex.zInteger);
    }
}

bool encodeGsReplayState(
    const GsReplayState &state,
    std::vector<uint8_t> &output,
    std::string *error)
{
    output.clear();
    if (error)
        error->clear();
    if (!validateReplayState(state, error))
        return false;

    output.reserve(4096u + state.rasterizer.feedbackVram.size());
    StateWriter writer(output);
    writer.bytes(kReplayStateMagic);
    writer.u32(GS_REPLAY_STATE_VERSION);
    for (const GSContext &context : state.ctx)
        writeContext(writer, context);
    writePrimitive(writer, state.prim);

    writer.u8(state.currentR);
    writer.u8(state.currentG);
    writer.u8(state.currentB);
    writer.u8(state.currentA);
    writer.f32(state.currentQ);
    writer.f32(state.currentS);
    writer.f32(state.currentT);
    writer.u16(state.currentU);
    writer.u16(state.currentV);
    writer.u8(state.currentFog);

    writer.u32(state.fogColor);
    writer.boolean(state.prmodecont);
    writer.boolean(state.pabe);
    writer.u8(state.scanMask);
    writer.u64(state.dimx);
    writer.boolean(state.dither);
    writer.boolean(state.colorClamp);
    writer.u8(state.texa.ta0);
    writer.boolean(state.texa.aem);
    writer.u8(state.texa.ta1);
    writer.u8(state.texclut.cbw);
    writer.u8(state.texclut.cou);
    writer.u16(state.texclut.cov);

    for (uint32_t value : state.clutCache)
        writer.u32(value);
    writer.bytes(state.clutCacheFormat);
    writer.bytes(state.clutCacheValid);
    for (uint32_t value : state.clutCbp)
        writer.u32(value);
    writer.u64(state.clutCacheGeneration);

    writer.u32(state.bitbltbuf.sbp);
    writer.u8(state.bitbltbuf.sbw);
    writer.u8(state.bitbltbuf.spsm);
    writer.u32(state.bitbltbuf.dbp);
    writer.u8(state.bitbltbuf.dbw);
    writer.u8(state.bitbltbuf.dpsm);
    writer.u16(state.trxpos.ssax);
    writer.u16(state.trxpos.ssay);
    writer.u16(state.trxpos.dsax);
    writer.u16(state.trxpos.dsay);
    writer.u8(state.trxpos.dir);
    writer.u16(state.trxreg.rrw);
    writer.u16(state.trxreg.rrh);
    writer.u32(state.trxdir);
    writer.u32(state.transfer.x);
    writer.u32(state.transfer.y);
    writer.u32(state.transfer.totalPixels);
    writer.u32(state.transfer.copiedPixels);

    writer.i32(state.vertexCount);
    for (const GSVertex &vertex : state.vertexQueue)
        writeVertex(writer, vertex);

    const GsReplayRasterizerState &rasterizer = state.rasterizer;
    writer.u32(rasterizer.feedbackTextureBase);
    writer.u32(rasterizer.feedbackFrameBase);
    writer.u8(rasterizer.feedbackTexturePsm);
    writer.u8(rasterizer.feedbackFramePsm);
    writer.u8(rasterizer.feedbackTextureWidth);
    writer.u8(rasterizer.feedbackFrameWidth);
    writer.boolean(rasterizer.feedbackSnapshotValid);
    writer.u32(static_cast<uint32_t>(rasterizer.feedbackVram.size()));
    writer.bytes(rasterizer.feedbackVram);
    for (uint32_t value : rasterizer.decodedClut)
        writer.u32(value);
    writer.u64(rasterizer.decodedClutGeneration);
    writer.u16(rasterizer.decodedClutTexa);
    writer.u8(rasterizer.decodedClutSourcePsm);
    writer.u8(rasterizer.decodedClutCsm);
    writer.u8(rasterizer.decodedClutCsa);
    writer.boolean(rasterizer.decodedClutActive);
    return true;
}

bool decodeGsReplayState(
    std::span<const uint8_t> input,
    GsReplayState &state,
    std::string *error)
{
    StateReader reader(input);
    std::array<uint8_t, kReplayStateMagic.size()> magic{};
    if (!reader.bytes(magic) || magic != kReplayStateMagic)
    {
        setError(error, "invalid GS replay state magic");
        return false;
    }
    uint32_t version = 0u;
    if (!reader.u32(version) || version != GS_REPLAY_STATE_VERSION)
    {
        setError(error, "unsupported GS replay state version");
        return false;
    }

    GsReplayState decoded{};
    for (GSContext &context : decoded.ctx)
    {
        if (!readContext(reader, context))
        {
            setError(error, "truncated GS replay context state");
            return false;
        }
    }
    if (!readPrimitive(reader, decoded.prim) ||
        !reader.u8(decoded.currentR) ||
        !reader.u8(decoded.currentG) ||
        !reader.u8(decoded.currentB) ||
        !reader.u8(decoded.currentA) ||
        !reader.f32(decoded.currentQ) ||
        !reader.f32(decoded.currentS) ||
        !reader.f32(decoded.currentT) ||
        !reader.u16(decoded.currentU) ||
        !reader.u16(decoded.currentV) ||
        !reader.u8(decoded.currentFog) ||
        !reader.u32(decoded.fogColor) ||
        !reader.boolean(decoded.prmodecont) ||
        !reader.boolean(decoded.pabe) ||
        !reader.u8(decoded.scanMask) ||
        !reader.u64(decoded.dimx) ||
        !reader.boolean(decoded.dither) ||
        !reader.boolean(decoded.colorClamp) ||
        !reader.u8(decoded.texa.ta0) ||
        !reader.boolean(decoded.texa.aem) ||
        !reader.u8(decoded.texa.ta1) ||
        !reader.u8(decoded.texclut.cbw) ||
        !reader.u8(decoded.texclut.cou) ||
        !reader.u16(decoded.texclut.cov))
    {
        setError(error, "truncated or invalid GS replay frontend state");
        return false;
    }

    for (uint32_t &value : decoded.clutCache)
    {
        if (!reader.u32(value))
        {
            setError(error, "truncated GS replay CLUT cache");
            return false;
        }
    }
    if (!reader.bytes(decoded.clutCacheFormat) ||
        !reader.bytes(decoded.clutCacheValid))
    {
        setError(error, "truncated GS replay CLUT metadata");
        return false;
    }
    for (uint8_t valid : decoded.clutCacheValid)
    {
        if (valid > 1u)
        {
            setError(error, "invalid GS replay CLUT cache-valid flag");
            return false;
        }
    }
    for (uint32_t &value : decoded.clutCbp)
    {
        if (!reader.u32(value))
        {
            setError(error, "truncated GS replay CLUT base state");
            return false;
        }
    }

    if (!reader.u64(decoded.clutCacheGeneration) ||
        !reader.u32(decoded.bitbltbuf.sbp) ||
        !reader.u8(decoded.bitbltbuf.sbw) ||
        !reader.u8(decoded.bitbltbuf.spsm) ||
        !reader.u32(decoded.bitbltbuf.dbp) ||
        !reader.u8(decoded.bitbltbuf.dbw) ||
        !reader.u8(decoded.bitbltbuf.dpsm) ||
        !reader.u16(decoded.trxpos.ssax) ||
        !reader.u16(decoded.trxpos.ssay) ||
        !reader.u16(decoded.trxpos.dsax) ||
        !reader.u16(decoded.trxpos.dsay) ||
        !reader.u8(decoded.trxpos.dir) ||
        !reader.u16(decoded.trxreg.rrw) ||
        !reader.u16(decoded.trxreg.rrh) ||
        !reader.u32(decoded.trxdir) ||
        !reader.u32(decoded.transfer.x) ||
        !reader.u32(decoded.transfer.y) ||
        !reader.u32(decoded.transfer.totalPixels) ||
        !reader.u32(decoded.transfer.copiedPixels) ||
        !reader.i32(decoded.vertexCount))
    {
        setError(error, "truncated GS replay transfer or vertex state");
        return false;
    }
    for (GSVertex &vertex : decoded.vertexQueue)
    {
        if (!readVertex(reader, vertex))
        {
            setError(error, "truncated GS replay vertex queue");
            return false;
        }
    }

    GsReplayRasterizerState &rasterizer = decoded.rasterizer;
    uint32_t feedbackBytes = 0u;
    if (!reader.u32(rasterizer.feedbackTextureBase) ||
        !reader.u32(rasterizer.feedbackFrameBase) ||
        !reader.u8(rasterizer.feedbackTexturePsm) ||
        !reader.u8(rasterizer.feedbackFramePsm) ||
        !reader.u8(rasterizer.feedbackTextureWidth) ||
        !reader.u8(rasterizer.feedbackFrameWidth) ||
        !reader.boolean(rasterizer.feedbackSnapshotValid) ||
        !reader.u32(feedbackBytes) ||
        feedbackBytes > GS_REPLAY_VRAM_SIZE)
    {
        setError(error, "invalid GS replay feedback state");
        return false;
    }
    if (rasterizer.feedbackSnapshotValid &&
        feedbackBytes != GS_REPLAY_VRAM_SIZE)
    {
        setError(error, "valid GS feedback snapshot is not exactly 4 MiB");
        return false;
    }
    rasterizer.feedbackVram.resize(feedbackBytes);
    if (!reader.bytes(rasterizer.feedbackVram))
    {
        setError(error, "truncated GS replay feedback snapshot");
        return false;
    }
    for (uint32_t &value : rasterizer.decodedClut)
    {
        if (!reader.u32(value))
        {
            setError(error, "truncated GS replay decoded CLUT");
            return false;
        }
    }
    if (!reader.u64(rasterizer.decodedClutGeneration) ||
        !reader.u16(rasterizer.decodedClutTexa) ||
        !reader.u8(rasterizer.decodedClutSourcePsm) ||
        !reader.u8(rasterizer.decodedClutCsm) ||
        !reader.u8(rasterizer.decodedClutCsa) ||
        !reader.boolean(rasterizer.decodedClutActive) ||
        !reader.atEnd())
    {
        setError(error, "trailing or truncated GS replay rasterizer state");
        return false;
    }

    if (!validateReplayState(decoded, error))
        return false;

    state = std::move(decoded);
    if (error)
        error->clear();
    return true;
}
