#ifndef PS2_GIF_ARBITER_H
#define PS2_GIF_ARBITER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

enum class GifPathId : uint8_t
{
    Path1 = 1,
    Path2 = 2,
    Path3 = 3,
};

struct GifArbiterPacket
{
    GifPathId pathId;
    bool path2DirectHl = false;
    bool path3Image = false;
    size_t offset = 0u;
    size_t size = 0u;
};

struct GifArbiterDrainBatch
{
    using Storage = std::array<std::vector<uint8_t>, 4>;

    struct Packet
    {
        GifPathId pathId = GifPathId::Path3;
        bool path2DirectHl = false;
        bool path3Image = false;
        uint8_t storageIndex = 0u;
        size_t offset = 0u;
        size_t size = 0u;
    };

    Storage storage;
    std::vector<Packet> packets;

    [[nodiscard]] bool empty() const noexcept
    {
        return packets.empty();
    }

    [[nodiscard]] std::span<const uint8_t> packetBytes(
        const Packet &packet) const noexcept
    {
        const size_t storageIndex = packet.storageIndex;
        if (storageIndex >= storage.size())
            return {};
        const std::vector<uint8_t> &bytes = storage[storageIndex];
        if (packet.offset > bytes.size() ||
            packet.size > bytes.size() - packet.offset)
        {
            return {};
        }
        return std::span<const uint8_t>(
            bytes.data() + packet.offset, packet.size);
    }
};

class GifArbiter
{
public:
    using Path1ReservationToken = uint64_t;

    using ProcessPacketFn = std::function<void(const uint8_t *, uint32_t)>;
    using DrainBoundaryFn = std::function<void()>;
    using ProcessBatchFn =
        std::function<void(GifArbiterDrainBatch)>;

    GifArbiter() = default;
    explicit GifArbiter(ProcessPacketFn processFn);

    void setProcessPacketFn(ProcessPacketFn fn)
    {
        m_processFn = std::move(fn);
        if (m_processFn)
            m_processBatchFn = {};
    }
    void setProcessBatchFn(ProcessBatchFn fn)
    {
        m_processBatchFn = std::move(fn);
        if (m_processBatchFn)
            m_processFn = {};
    }
    void setDrainCallbacks(DrainBoundaryFn beginFn, DrainBoundaryFn endFn)
    {
        m_beginDrainFn = std::move(beginFn);
        m_endDrainFn = std::move(endFn);
    }

    void submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl = false);

    // A threaded VU invocation can contain PATH1 output which is not known
    // until the owner completes. Reserve its place before allowing later GIF
    // paths to drain, mirroring the future-PATH1 marker used by MTVU designs.
    // Reset invalidates outstanding tokens, so lifecycle code can release a
    // stale reservation without confusing a new GIF generation.
    [[nodiscard]] Path1ReservationToken reservePath1();
    [[nodiscard]] bool releasePath1(
        Path1ReservationToken token) noexcept;
    [[nodiscard]] size_t pendingPath1Reservations() const noexcept
    {
        return m_path1Reservations.size();
    }

    void drain();
    [[nodiscard]] GifArbiterDrainBatch takeDrainBatch();
    void recycleDrainBatch(GifArbiterDrainBatch batch);
    void reset();
    bool empty() const;
    bool canAcceptPath2(bool directHl) const;

private:
    struct PathStream
    {
        std::vector<uint8_t> data;
        size_t packetStart = 0u;
        size_t parseOffset = 0u;
        bool path2DirectHl = false;
        bool packetContainsImage = false;
    };

    ProcessPacketFn m_processFn;
    ProcessBatchFn m_processBatchFn;
    DrainBoundaryFn m_beginDrainFn;
    DrainBoundaryFn m_endDrainFn;
    std::vector<GifArbiterPacket> m_queue;
    std::array<PathStream, 4> m_pathStreams;
    GifArbiterDrainBatch::Storage m_recycledStorage;
    std::vector<GifArbiterDrainBatch::Packet> m_recycledPackets;
    std::vector<Path1ReservationToken> m_path1Reservations;
    Path1ReservationToken m_nextPath1ReservationToken = 1u;

    static uint8_t pathPriority(GifPathId id);
};

#endif
