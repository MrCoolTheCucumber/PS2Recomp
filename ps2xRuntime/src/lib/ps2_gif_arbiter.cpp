#include "runtime/ps2_gif_arbiter.h"
#include "ps2_log.h"
#include <algorithm>
#include <cstring>

namespace
{
    constexpr size_t kMaxBufferedGifPathBytes = 64u * 1024u * 1024u;

    bool gifTagSize(const uint8_t *data, size_t availableBytes, size_t &outSize, bool &outEop)
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
        return true;
    }
}

GifArbiter::GifArbiter(ProcessPacketFn processFn)
    : m_processFn(std::move(processFn))
{
}

bool GifArbiter::containsImagePacket(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes < 16u)
        return false;

    size_t offset = 0u;
    while (offset + 16u <= sizeBytes)
    {
        uint64_t tagLo = 0u;
        std::memcpy(&tagLo, data + offset, sizeof(tagLo));
        const uint8_t flg = static_cast<uint8_t>((tagLo >> 58u) & 0x3u);
        if (flg == 2u || flg == 3u)
            return true;

        size_t tagBytes = 0u;
        bool eop = false;
        if (!gifTagSize(data + offset, sizeBytes - offset, tagBytes, eop) || tagBytes > sizeBytes - offset)
            return false;
        offset += tagBytes;
    }
    return false;
}

void GifArbiter::submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl)
{
    if (!data || sizeBytes == 0u || !m_processFn)
        return;

    const size_t pathIndex = static_cast<size_t>(pathId);
    if (pathIndex >= m_pathStreams.size())
        return;

    PathStream &stream = m_pathStreams[pathIndex];
    if (stream.data.empty())
        stream.path2DirectHl = path2DirectHl;
    else
        stream.path2DirectHl = stream.path2DirectHl || path2DirectHl;

    if (sizeBytes > kMaxBufferedGifPathBytes || stream.data.size() > kMaxBufferedGifPathBytes - sizeBytes)
    {
        RUNTIME_LOG("[gif] dropping malformed path " << pathIndex
                                                       << " stream larger than " << kMaxBufferedGifPathBytes
                                                       << " bytes" << std::endl);
        stream = {};
        return;
    }
    stream.data.insert(stream.data.end(), data, data + sizeBytes);

    size_t packetStart = 0u;
    size_t offset = 0u;
    while (offset + 16u <= stream.data.size())
    {
        size_t tagBytes = 0u;
        bool eop = false;
        if (!gifTagSize(stream.data.data() + offset, stream.data.size() - offset, tagBytes, eop))
        {
            RUNTIME_LOG("[gif] dropping malformed tag on path " << pathIndex << std::endl);
            stream = {};
            return;
        }
        if (tagBytes > stream.data.size() - offset)
            break;

        offset += tagBytes;
        if (!eop)
            continue;

        GifArbiterPacket pkt;
        pkt.pathId = pathId;
        pkt.path2DirectHl = (pathId == GifPathId::Path2) && stream.path2DirectHl;
        pkt.data.assign(stream.data.begin() + static_cast<std::ptrdiff_t>(packetStart),
                        stream.data.begin() + static_cast<std::ptrdiff_t>(offset));
        pkt.path3Image = (pathId == GifPathId::Path3) &&
                         containsImagePacket(pkt.data.data(), static_cast<uint32_t>(pkt.data.size()));
        m_queue.push_back(std::move(pkt));

        packetStart = offset;
        stream.path2DirectHl = path2DirectHl;
    }

    if (packetStart != 0u)
    {
        stream.data.erase(stream.data.begin(),
                          stream.data.begin() + static_cast<std::ptrdiff_t>(packetStart));
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

void GifArbiter::drain()
{
    if (!m_processFn)
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

    for (size_t i = 0; i < m_queue.size(); ++i)
    {
        auto &pkt = m_queue[i];
        if (!pkt.data.empty())
        {
            m_processFn(pkt.data.data(), static_cast<uint32_t>(pkt.data.size()));
        }
    }
    m_queue.clear();
}

uint8_t GifArbiter::pathPriority(GifPathId id)
{
    return static_cast<uint8_t>(id);
}
