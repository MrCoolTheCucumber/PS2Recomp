#ifndef PS2_GIF_ARBITER_H
#define PS2_GIF_ARBITER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
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

class GifArbiter
{
public:
    using ProcessPacketFn = std::function<void(const uint8_t *, uint32_t)>;
    using DrainBoundaryFn = std::function<void()>;

    GifArbiter() = default;
    explicit GifArbiter(ProcessPacketFn processFn);

    void setProcessPacketFn(ProcessPacketFn fn) { m_processFn = std::move(fn); }
    void setDrainCallbacks(DrainBoundaryFn beginFn, DrainBoundaryFn endFn)
    {
        m_beginDrainFn = std::move(beginFn);
        m_endDrainFn = std::move(endFn);
    }

    void submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl = false);

    void drain();
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
    DrainBoundaryFn m_beginDrainFn;
    DrainBoundaryFn m_endDrainFn;
    std::vector<GifArbiterPacket> m_queue;
    std::array<PathStream, 4> m_pathStreams;

    static uint8_t pathPriority(GifPathId id);
};

#endif
