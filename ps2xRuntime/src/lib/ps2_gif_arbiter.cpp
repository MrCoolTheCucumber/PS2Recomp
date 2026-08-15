#include "runtime/ps2_gif_arbiter.h"
#include "ps2_log.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

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

GifArbiter::Path1ReservationToken GifArbiter::reservePath1()
{
    if (m_nextPath1ReservationToken == 0u ||
        m_nextPath1ReservationToken ==
            std::numeric_limits<Path1ReservationToken>::max())
    {
        throw std::overflow_error(
            "GIF PATH1 reservation token exhausted");
    }
    const Path1ReservationToken token =
        m_nextPath1ReservationToken++;
    m_path1Reservations.push_back(token);
    return token;
}

bool GifArbiter::releasePath1(
    Path1ReservationToken token) noexcept
{
    if (token == 0u)
        return false;
    const auto found = std::find(
        m_path1Reservations.begin(),
        m_path1Reservations.end(), token);
    if (found == m_path1Reservations.end())
        return false;
    m_path1Reservations.erase(found);
    return true;
}

void GifArbiter::submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl)
{
    if (!data || sizeBytes == 0u)
        return;

    const size_t pathIndex = static_cast<size_t>(pathId);
    if (pathIndex >= m_pathStreams.size())
        return;

    PathStream &stream = m_pathStreams[pathIndex];
    if (stream.data.empty() &&
        m_recycledStorage[pathIndex].capacity() >
            stream.data.capacity())
    {
        stream.data.swap(m_recycledStorage[pathIndex]);
        stream.data.clear();
    }
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
    m_recycledStorage = {};
    m_recycledPackets = {};
    m_path1Reservations.clear();
}

bool GifArbiter::empty() const
{
    if (!m_path1Reservations.empty())
        return false;
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
    if (!m_path1Reservations.empty())
        return false;
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
    if ((!m_processFn && !m_processBatchFn) ||
        m_queue.empty() ||
        !m_path1Reservations.empty())
        return;

    GifArbiterDrainBatch batch = takeDrainBatch();
    if (batch.empty())
        return;

    if (m_processBatchFn)
    {
        m_processBatchFn(std::move(batch));
        return;
    }

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

    for (const GifArbiterDrainBatch::Packet &packet : batch.packets)
    {
        const std::span<const uint8_t> bytes =
            batch.packetBytes(packet);
        if (!bytes.empty())
        {
            m_processFn(
                bytes.data(),
                static_cast<uint32_t>(bytes.size()));
        }
    }
    recycleDrainBatch(std::move(batch));
}

GifArbiterDrainBatch GifArbiter::takeDrainBatch()
{
    GifArbiterDrainBatch batch{};
    if (m_queue.empty() ||
        !m_path1Reservations.empty())
        return batch;

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

    std::array<size_t, 4> consumedByPath{};
    for (const GifArbiterPacket &packet : m_queue)
    {
        const size_t pathIndex =
            static_cast<size_t>(packet.pathId);
        if (pathIndex < consumedByPath.size())
        {
            consumedByPath[pathIndex] = std::max(
                consumedByPath[pathIndex],
                packet.offset + packet.size);
        }
    }
    for (size_t pathIndex = 0u;
         pathIndex < m_pathStreams.size(); ++pathIndex)
    {
        PathStream &stream = m_pathStreams[pathIndex];
        const size_t consumed = consumedByPath[pathIndex];
        if (consumed == 0u || consumed > stream.data.size())
            continue;

        if (consumed == stream.data.size())
        {
            batch.storage[pathIndex] = std::move(stream.data);
            stream.data.clear();
        }
        else
        {
            // A complete prefix plus an incomplete suffix cannot share one
            // vector after ownership moves to the command. Copy only that
            // rare suffix, then move the complete prefix without copying it.
            std::vector<uint8_t> suffix(
                stream.data.begin() +
                    static_cast<std::ptrdiff_t>(consumed),
                stream.data.end());
            stream.data.resize(consumed);
            batch.storage[pathIndex] = std::move(stream.data);
            stream.data = std::move(suffix);
        }

        stream.packetStart = 0u;
        stream.parseOffset -= consumed;
        if (stream.data.empty())
        {
            stream.path2DirectHl = false;
            stream.packetContainsImage = false;
        }
    }

    batch.packets.swap(m_recycledPackets);
    batch.packets.clear();
    batch.packets.reserve(m_queue.size());
    for (size_t i = 0; i < m_queue.size(); ++i)
    {
        const GifArbiterPacket &pkt = m_queue[i];
        const size_t pathIndex =
            static_cast<size_t>(pkt.pathId);
        if (pathIndex >= m_pathStreams.size())
            continue;

        const std::vector<uint8_t> &storage =
            batch.storage[pathIndex];
        if (pkt.size != 0u &&
            pkt.offset <= storage.size() &&
            pkt.size <= storage.size() - pkt.offset)
        {
            GifArbiterDrainBatch::Packet owned{};
            owned.pathId = pkt.pathId;
            owned.path2DirectHl = pkt.path2DirectHl;
            owned.path3Image = pkt.path3Image;
            owned.storageIndex =
                static_cast<uint8_t>(pathIndex);
            owned.offset = pkt.offset;
            owned.size = pkt.size;
            batch.packets.push_back(std::move(owned));
        }
    }
    m_queue.clear();
    return batch;
}

void GifArbiter::recycleDrainBatch(
    GifArbiterDrainBatch batch)
{
    for (size_t pathIndex = 0u;
         pathIndex < batch.storage.size(); ++pathIndex)
    {
        std::vector<uint8_t> &returned = batch.storage[pathIndex];
        returned.clear();
        PathStream &stream = m_pathStreams[pathIndex];
        if (stream.data.empty() &&
            stream.packetStart == 0u &&
            stream.parseOffset == 0u &&
            returned.capacity() > stream.data.capacity())
        {
            stream.data = std::move(returned);
            continue;
        }
        if (returned.capacity() >
            m_recycledStorage[pathIndex].capacity())
        {
            m_recycledStorage[pathIndex] =
                std::move(returned);
        }
    }
    batch.packets.clear();
    if (batch.packets.capacity() >
        m_recycledPackets.capacity())
    {
        m_recycledPackets =
            std::move(batch.packets);
    }
}

uint8_t GifArbiter::pathPriority(GifPathId id)
{
    return static_cast<uint8_t>(id);
}
