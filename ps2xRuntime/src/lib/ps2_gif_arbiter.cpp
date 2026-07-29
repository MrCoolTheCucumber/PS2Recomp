#include "runtime/ps2_gif_arbiter.h"
#include "ps2_log.h"
#include <algorithm>
#include <cstring>

namespace
{
    constexpr size_t kMaxBufferedGifPathBytes = 64u * 1024u * 1024u;

    bool gifTagSize(
        const uint8_t *data,
        size_t availableBytes,
        size_t &outSize,
        bool &outEop,
        bool &outImage)
    {
        if (!data || availableBytes < 16u)
            return false;

        uint64_t tagLo = 0u;
        std::memcpy(&tagLo, data, sizeof(tagLo));
        const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
        const uint8_t flg = static_cast<uint8_t>((tagLo >> 58u) & 0x3u);
        uint32_t nreg = static_cast<uint32_t>((tagLo >> 60u) & 0xFu);
        if (nreg == 0u)
            nreg = 16u;

        uint64_t payloadBytes = 0u;
        if (flg == 0u)
            payloadBytes = static_cast<uint64_t>(nloop) * nreg * 16u;
        else if (flg == 1u)
            payloadBytes = (static_cast<uint64_t>(nloop) * nreg * 8u + 15u) & ~15ull;
        else
            payloadBytes = static_cast<uint64_t>(nloop) * 16u; // IMAGE and IMAGE2.

        const uint64_t totalBytes = 16u + payloadBytes;
        if (totalBytes > kMaxBufferedGifPathBytes)
            return false;

        outSize = static_cast<size_t>(totalBytes);
        outEop = ((tagLo >> 15u) & 1u) != 0u;
        outImage = flg == 2u || flg == 3u;
        return true;
    }
}

GifArbiter::GifArbiter(ProcessPacketFn processFn)
    : m_processFn(std::move(processFn))
{
}

void GifArbiter::submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl)
{
    if (!data || sizeBytes == 0u || !m_processFn)
        return;

    const size_t pathIndex = static_cast<size_t>(pathId);
    if (pathIndex >= m_pathStreams.size())
        return;

    PathStream &stream = m_pathStreams[pathIndex];
    auto dropIncompletePacket = [&stream]()
    {
        stream.data.resize(stream.packetStart);
        stream.parseOffset = stream.packetStart;
        stream.path2DirectHl = false;
        stream.packetContainsImage = false;
    };

    if (stream.packetStart == stream.data.size())
        stream.path2DirectHl = path2DirectHl;
    else
        stream.path2DirectHl = stream.path2DirectHl || path2DirectHl;

    if (sizeBytes > kMaxBufferedGifPathBytes || stream.data.size() > kMaxBufferedGifPathBytes - sizeBytes)
    {
        RUNTIME_LOG("[gif] dropping malformed path " << pathIndex
                                                       << " stream larger than " << kMaxBufferedGifPathBytes
                                                       << " bytes" << std::endl);
        dropIncompletePacket();
        return;
    }
    stream.data.insert(stream.data.end(), data, data + sizeBytes);

    size_t offset = stream.parseOffset;
    while (offset + 16u <= stream.data.size())
    {
        size_t tagBytes = 0u;
        bool eop = false;
        bool image = false;
        if (!gifTagSize(
                stream.data.data() + offset,
                stream.data.size() - offset,
                tagBytes,
                eop,
                image))
        {
            RUNTIME_LOG("[gif] dropping malformed tag on path " << pathIndex << std::endl);
            dropIncompletePacket();
            return;
        }
        if (tagBytes > stream.data.size() - offset)
            break;

        stream.packetContainsImage =
            stream.packetContainsImage || image;
        offset += tagBytes;
        stream.parseOffset = offset;
        if (!eop)
            continue;

        GifArbiterPacket pkt;
        pkt.pathId = pathId;
        pkt.path2DirectHl = (pathId == GifPathId::Path2) && stream.path2DirectHl;
        pkt.path3Image =
            (pathId == GifPathId::Path3) &&
            stream.packetContainsImage;
        pkt.offset = stream.packetStart;
        pkt.size = offset - stream.packetStart;
        m_queue.push_back(pkt);

        stream.packetStart = offset;
        stream.packetContainsImage = false;
        stream.path2DirectHl =
            offset < stream.data.size()
                ? path2DirectHl
                : false;
    }
}

void GifArbiter::reset()
{
    m_queue.clear();
    for (PathStream &stream : m_pathStreams)
        stream = {};
}

bool GifArbiter::empty() const
{
    if (!m_queue.empty())
        return false;
    for (const PathStream &stream : m_pathStreams)
    {
        if (!stream.data.empty())
            return false;
    }
    return true;
}

bool GifArbiter::canAcceptPath2(bool directHl) const
{
    if (!directHl)
        return true;

    return std::none_of(
        m_queue.begin(), m_queue.end(),
        [](const GifArbiterPacket &packet)
        {
            return packet.pathId == GifPathId::Path3 &&
                   packet.path3Image;
        });
}

void GifArbiter::drain()
{
    if (!m_processFn || m_queue.empty())
        return;

    std::stable_sort(m_queue.begin(), m_queue.end(),
                     [](const GifArbiterPacket &a, const GifArbiterPacket &b)
                     {
                         // DIRECTHL cannot preempt PATH3 IMAGE transfers.
                         if (a.path2DirectHl != b.path2DirectHl || a.path3Image != b.path3Image)
                         {
                             if (a.path3Image && b.path2DirectHl)
                                 return true;
                             if (a.path2DirectHl && b.path3Image)
                                 return false;
                         }
                         return pathPriority(a.pathId) < pathPriority(b.pathId);
                     });

    if (m_beginDrainFn)
        m_beginDrainFn();
    struct DrainScope
    {
        DrainBoundaryFn &endFn;
        ~DrainScope()
        {
            if (endFn)
                endFn();
        }
    } drainScope{m_endDrainFn};

    for (size_t i = 0; i < m_queue.size(); ++i)
    {
        const GifArbiterPacket &pkt = m_queue[i];
        const size_t pathIndex =
            static_cast<size_t>(pkt.pathId);
        if (pathIndex >= m_pathStreams.size())
            continue;

        const PathStream &stream =
            m_pathStreams[pathIndex];
        if (pkt.size != 0u &&
            pkt.offset <= stream.data.size() &&
            pkt.size <= stream.data.size() - pkt.offset)
        {
            m_processFn(
                stream.data.data() + pkt.offset,
                static_cast<uint32_t>(pkt.size));
        }
    }
    m_queue.clear();

    for (PathStream &stream : m_pathStreams)
    {
        const size_t consumed = stream.packetStart;
        if (consumed == 0u)
            continue;

        const size_t remaining =
            stream.data.size() - consumed;
        if (remaining != 0u)
        {
            std::memmove(
                stream.data.data(),
                stream.data.data() + consumed,
                remaining);
        }
        stream.data.resize(remaining);
        stream.packetStart = 0u;
        stream.parseOffset -= consumed;
        if (remaining == 0u)
        {
            stream.path2DirectHl = false;
            stream.packetContainsImage = false;
        }
    }
}

uint8_t GifArbiter::pathPriority(GifPathId id)
{
    return static_cast<uint8_t>(id);
}
