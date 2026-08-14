// Based on Blackline Interactive implementation
#include "runtime/ps2_memory.h"

#include <algorithm>
#include <cstring>

enum VIFCmd : uint8_t
{
    VIF_NOP = 0x00,
    VIF_STCYCL = 0x01,
    VIF_OFFSET = 0x02,
    VIF_BASE = 0x03,
    VIF_ITOP = 0x04,
    VIF_STMOD = 0x05,
    VIF_MSKPATH3 = 0x06,
    VIF_MARK = 0x07,
    VIF_FLUSHE = 0x10,
    VIF_FLUSH = 0x11,
    VIF_FLUSHA = 0x13,
    VIF_MSCAL = 0x14,
    VIF_MSCALF = 0x15,
    VIF_MSCNT = 0x17,
    VIF_STMASK = 0x20,
    VIF_STROW = 0x30,
    VIF_STCOL = 0x31,
    VIF_MPG = 0x4A,
    VIF_DIRECT = 0x50,
    VIF_DIRECTHL = 0x51,
};

namespace
{
    void copyWrappedMicroMemory(
        uint8_t *destination, uint32_t destinationSize,
        uint32_t destinationOffset, const uint8_t *source,
        uint32_t sizeBytes)
    {
        while (sizeBytes != 0u)
        {
            const uint32_t chunk = std::min(
                sizeBytes,
                destinationSize - destinationOffset);
            std::memcpy(
                destination + destinationOffset,
                source, chunk);
            source += chunk;
            sizeBytes -= chunk;
            destinationOffset = 0u;
        }
    }

    bool equalsWrappedMicroMemory(
        const uint8_t *destination, uint32_t destinationSize,
        uint32_t destinationOffset, const uint8_t *source,
        uint32_t sizeBytes)
    {
        while (sizeBytes != 0u)
        {
            const uint32_t chunk = std::min(
                sizeBytes,
                destinationSize - destinationOffset);
            if (std::memcmp(
                    destination + destinationOffset,
                    source, chunk) != 0)
            {
                return false;
            }
            source += chunk;
            sizeBytes -= chunk;
            destinationOffset = 0u;
        }
        return true;
    }

    bool tryUnpackContiguousV4(
        uint8_t *destination, uint32_t destinationSize,
        uint32_t destinationVector, const uint8_t *source,
        uint32_t vectorCount, uint8_t componentWidth,
        bool zeroExtend, uint32_t mode, const uint32_t *row)
    {
        if (!destination || !source || !row ||
            destinationSize < 16u ||
            (destinationSize & 15u) != 0u ||
            componentWidth > 2u || mode > 1u)
        {
            return false;
        }

        const uint32_t destinationVectors = destinationSize / 16u;
        destinationVector %= destinationVectors;
        if (componentWidth == 0u && mode != 1u)
        {
            copyWrappedMicroMemory(
                destination, destinationSize,
                destinationVector * 16u, source,
                vectorCount * 16u);
            return true;
        }

        const uint32_t sourceStride =
            componentWidth == 0u ? 16u :
            componentWidth == 1u ? 8u : 4u;
        const __m128i rowValues = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(row));
        uint32_t remaining = vectorCount;
        while (remaining != 0u)
        {
            const uint32_t chunk = std::min(
                remaining,
                destinationVectors - destinationVector);
            uint8_t *output =
                destination + destinationVector * 16u;
            for (uint32_t index = 0u; index < chunk; ++index)
            {
                __m128i value{};
                if (componentWidth == 0u)
                {
                    value = _mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(source));
                }
                else if (componentWidth == 1u)
                {
                    const __m128i packed = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i *>(source));
                    value = zeroExtend
                        ? _mm_cvtepu16_epi32(packed)
                        : _mm_cvtepi16_epi32(packed);
                }
                else
                {
                    uint32_t packed = 0u;
                    std::memcpy(&packed, source, sizeof(packed));
                    const __m128i bytes = _mm_cvtsi32_si128(
                        static_cast<int32_t>(packed));
                    value = zeroExtend
                        ? _mm_cvtepu8_epi32(bytes)
                        : _mm_cvtepi8_epi32(bytes);
                }

                if (mode == 1u)
                    value = _mm_add_epi32(value, rowValues);
                _mm_storeu_si128(
                    reinterpret_cast<__m128i *>(output), value);
                source += sourceStride;
                output += 16u;
            }
            remaining -= chunk;
            destinationVector = 0u;
        }
        return true;
    }
}

void PS2Memory::processVIF0Data(uint32_t srcPhys, uint32_t sizeBytes)
{
    if (sizeBytes == 0u || srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    processVIF0Data(m_rdram + srcPhys, sizeBytes);
}

void PS2Memory::processVIF0Data(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes == 0u)
        return;

    appendVif0Stream(data, sizeBytes);
    if (!m_vif0WaitingForVu)
        (void)processVif0Stream();
}

void PS2Memory::appendVif0Stream(
    const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes == 0u)
        return;
    m_vif0DeferredData.insert(
        m_vif0DeferredData.end(), data, data + sizeBytes);
}

bool PS2Memory::resumeVIF0AfterVu()
{
    if (!m_vif0WaitingForVu)
        return false;

    m_vif0WaitingForVu = false;
    vif0_regs.stat &= ~(1u << 2); // VEW
    m_vif0Dma.stall = Vif0DmaStallReason::None;
    return true;
}

void PS2Memory::cancelVIF0VuWait()
{
    m_vif0WaitingForVu = false;
    if (m_rdram)
        vif0_regs.stat &= ~(1u << 2); // VEW
    m_vif0DeferredData.clear();
    m_vif0Dma.bufferedPayloadBytes = 0u;
}

PS2Memory::Vif0ParserResult
PS2Memory::processVif0Stream()
{
    if (m_vif0WaitingForVu)
    {
        return {
            Vif0ParserDisposition::WaitingForVu, 0u};
    }
    if ((vif0_regs.stat & (1u << 10)) != 0u)
    {
        return {
            Vif0ParserDisposition::VifInterrupt, 0u};
    }
    if (m_vif0DeferredData.empty())
    {
        vif0_regs.stat &= ~0x3u;
        return {Vif0ParserDisposition::Drained, 0u};
    }

    const uint8_t *const data = m_vif0DeferredData.data();
    const uint32_t sizeBytes =
        static_cast<uint32_t>(
            std::min<size_t>(
                m_vif0DeferredData.size(),
                static_cast<size_t>(UINT32_MAX)));
    uint32_t pos = 0u;
    bool stallAfterCommand = false;

    const auto finish =
        [&](Vif0ParserDisposition disposition)
        {
            const uint32_t consumed = pos;
            if (consumed != 0u)
            {
                m_vif0DeferredData.erase(
                    m_vif0DeferredData.begin(),
                    m_vif0DeferredData.begin() + consumed);
            }

            vif0_regs.stat &= ~0x3u;
            if (disposition ==
                Vif0ParserDisposition::NeedMoreData)
            {
                vif0_regs.stat |= 0x1u;
            }
            return Vif0ParserResult{
                disposition, consumed};
        };

    while (pos + 4 <= sizeBytes)
    {
        if (stallAfterCommand)
        {
            vif0_regs.stat |= (1u << 10); // VIS
            return finish(
                Vif0ParserDisposition::VifInterrupt);
        }

        const uint32_t commandStart = pos;
        uint32_t cmd = 0u;
        std::memcpy(&cmd, data + pos, sizeof(cmd));
        pos += 4u;

        const uint8_t opcode = static_cast<uint8_t>((cmd >> 24) & 0x7Fu);
        const uint16_t imm = static_cast<uint16_t>(cmd & 0xFFFFu);
        const uint8_t num = static_cast<uint8_t>((cmd >> 16) & 0xFFu);
        const bool irq = (cmd & 0x80000000u) != 0u;

        vif0_regs.code = cmd;
        vif0_regs.num = num;
        if (irq)
            vif0_regs.stat |= (1u << 11);

        const bool waitsForVu =
            opcode == VIF_FLUSHE ||
            opcode == VIF_FLUSH ||
            opcode == VIF_FLUSHA ||
            opcode == VIF_MSCAL ||
            opcode == VIF_MSCALF ||
            opcode == VIF_MSCNT;
        if (waitsForVu &&
            m_vu0BusyCallback &&
            m_vu0BusyCallback())
        {
            m_vif0WaitingForVu = true;
            vif0_regs.stat |= (1u << 2); // VEW
            pos = commandStart;
            return finish(
                Vif0ParserDisposition::WaitingForVu);
        }

        stallAfterCommand = irq;

        if (opcode == VIF_NOP)
        {
            continue;
        }
        else if (opcode == VIF_STCYCL)
        {
            vif0_regs.cycle = imm;
            continue;
        }
        else if (opcode == VIF_ITOP)
        {
            vif0_regs.itops = imm & 0xFFu;
            continue;
        }
        else if (opcode == VIF_STMOD)
        {
            vif0_regs.mode = imm & 3u;
            continue;
        }
        else if (opcode == VIF_MARK)
        {
            vif0_regs.mark = imm;
            vif0_regs.stat |= (1u << 6);
            continue;
        }
        else if (opcode == VIF_FLUSHE || opcode == VIF_FLUSH || opcode == VIF_FLUSHA)
        {
            continue;
        }
        else if (opcode == VIF_MSCAL ||
                 opcode == VIF_MSCALF)
        {
            const uint32_t runItop =
                vif0_regs.itops & 0xFFu;
            vif0_regs.itop = runItop;
            if (m_vu0MscalCallback)
            {
                m_vu0MscalCallback(
                    static_cast<uint32_t>(
                        imm & 0x1FFu) *
                        8u,
                    runItop);
            }
            continue;
        }
        else if (opcode == VIF_MSCNT)
        {
            const uint32_t runItop =
                vif0_regs.itops & 0xFFu;
            vif0_regs.itop = runItop;
            if (m_vu0MscntCallback)
                m_vu0MscntCallback(runItop);
            continue;
        }
        else if (opcode == VIF_STMASK)
        {
            if (pos + 4u > sizeBytes)
            {
                pos = commandStart;
                stallAfterCommand = false;
                break;
            }
            std::memcpy(&vif0_regs.mask, data + pos, sizeof(vif0_regs.mask));
            pos += 4u;
            continue;
        }
        else if (opcode == VIF_STROW)
        {
            if (pos + 16u > sizeBytes)
            {
                pos = commandStart;
                stallAfterCommand = false;
                break;
            }
            std::memcpy(vif0_regs.row, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == VIF_STCOL)
        {
            if (pos + 16u > sizeBytes)
            {
                pos = commandStart;
                stallAfterCommand = false;
                break;
            }
            std::memcpy(vif0_regs.col, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == VIF_MPG)
        {
            const uint32_t destAddr = static_cast<uint32_t>(imm & 0x1FFu) * 8u;
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            if (pos + mpgBytes > sizeBytes)
            {
                pos = commandStart;
                stallAfterCommand = false;
                break;
            }
            if (m_vu0Code && mpgBytes > 0u)
            {
                copyWrappedMicroMemory(
                    m_vu0Code, PS2_VU0_CODE_SIZE,
                    destAddr, data + pos, mpgBytes);
                markVU0CodeModified();
            }

            pos += mpgBytes;
            continue;
        }
        else if ((opcode & 0x60u) == 0x60u)
        {
            const uint8_t vn = static_cast<uint8_t>((opcode >> 2) & 0x3u);
            const uint8_t vl = static_cast<uint8_t>(opcode & 0x3u);
            const int components = static_cast<int>(vn) + 1;
            int bitsPerComponent = 32;
            switch (vl)
            {
            case 0:
                bitsPerComponent = 32;
                break;
            case 1:
                bitsPerComponent = 16;
                break;
            case 2:
                bitsPerComponent = 8;
                break;
            case 3:
                bitsPerComponent = (vn == 3u) ? 4 : 16;
                break;
            default:
                break;
            }
            const int bitsPerVector = (vl == 3u && vn == 3u) ? 16 : (components * bitsPerComponent);
            uint32_t bytesPerVector = static_cast<uint32_t>((bitsPerVector + 7) / 8);
            const uint32_t writeVectorCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            uint32_t cl = vif0_regs.cycle & 0xFFu;
            uint32_t wl = (vif0_regs.cycle >> 8) & 0xFFu;
            if (cl == 0u)
                cl = 1u;
            if (wl == 0u)
                wl = 1u;
            uint32_t sourceVectorCount = writeVectorCount;
            if (cl < wl)
            {
                const uint32_t fullBlocks = writeVectorCount / wl;
                uint32_t remainder = writeVectorCount % wl;
                if (remainder > cl)
                    remainder = cl;
                sourceVectorCount = fullBlocks * cl + remainder;
            }
            uint32_t totalBytes = sourceVectorCount * bytesPerVector;
            totalBytes = (totalBytes + 3u) & ~3u;

            if (pos + totalBytes > sizeBytes)
            {
                pos = commandStart;
                stallAfterCommand = false;
                break;
            }

            if (m_vu0Data && vl == 0u)
            {
                uint32_t vuAddr =
                    static_cast<uint32_t>(imm & 0xFFu);
                if ((imm & 0x8000u) != 0u)
                    vuAddr =
                        (vuAddr +
                         (vif0_regs.tops & 0xFFu)) &
                        0xFFu;
                const uint8_t *srcBase = data + pos;
                uint32_t srcIndex = 0u;
                for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
                {
                    const uint32_t cyclePos = writeIndex % wl;
                    const bool sourceAvailable = (cl >= wl) || (cyclePos < cl);
                    uint32_t destVec =
                        (cl >= wl)
                            ? ((vuAddr +
                                (writeIndex / wl) * cl +
                                cyclePos) &
                               0xFFu)
                            : ((vuAddr + writeIndex) &
                               0xFFu);
                    const uint32_t destOff = destVec * 16u;
                    if (destOff + 16u > PS2_VU0_DATA_SIZE)
                    {
                        if (sourceAvailable && srcIndex < sourceVectorCount)
                            ++srcIndex;
                        continue;
                    }
                    if (!sourceAvailable || srcIndex >= sourceVectorCount)
                        continue;
                    const uint8_t *srcVec = srcBase + srcIndex * bytesPerVector;
                    ++srcIndex;
                    uint32_t lanes[4] = {0u, 0u, 0u, 0u};
                    std::memcpy(lanes, m_vu0Data + destOff, sizeof(lanes));
                    const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                    for (uint32_t c = 0; c < limit; ++c)
                    {
                        uint32_t scalar = 0u;
                        std::memcpy(&scalar, srcVec + c * 4u, sizeof(scalar));
                        lanes[c] = scalar;
                    }
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(m_vu0Data + destOff), _mm_loadu_si128(reinterpret_cast<const __m128i *>(lanes)));
                }
            }
            pos += totalBytes;
            continue;
        }
        else
        {
            continue;
        }
    }

    if (stallAfterCommand)
    {
        vif0_regs.stat |= (1u << 10); // VIS
        return finish(
            Vif0ParserDisposition::VifInterrupt);
    }
    if (pos == sizeBytes)
        return finish(Vif0ParserDisposition::Drained);
    return finish(Vif0ParserDisposition::NeedMoreData);
}

void PS2Memory::processVIF1Data(uint32_t srcPhys, uint32_t sizeBytes)
{
    if (sizeBytes == 0u || srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    processVIF1Data(m_rdram + srcPhys, sizeBytes);
}

bool PS2Memory::resumeVIF1AfterVu()
{
    if (!m_vif1WaitingForVu)
        return false;

    m_vif1WaitingForVu = false;
    vif1_regs.stat &= ~(1u << 2); // VEW

    if (m_vif1Dma.active)
    {
        m_vif1Dma.stall = Vif1DmaStallReason::None;
        if (m_vif1Dma.eventManaged)
            return wakeVif1Dma();
        drainVif1DmaCompatibility();
        return true;
    }

    (void)processVif1Stream();
    return true;
}

void PS2Memory::cancelVIF1VuWait()
{
    m_vif1WaitingForVu = false;
    if (m_rdram)
        vif1_regs.stat &= ~(1u << 2); // VEW
    clearVif1Stream();
}

void PS2Memory::processVIF1Data(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes == 0u)
        return;

    appendVif1Stream(data, sizeBytes);
    if (m_vif1WaitingForVu)
        return;

    (void)processVif1Stream();
}

void PS2Memory::appendVif1Stream(
    const uint8_t *data, uint32_t sizeBytes,
    std::optional<uint8_t> firstByteOffset)
{
    if (!data || sizeBytes == 0u)
        return;

    const uint8_t startOffset = static_cast<uint8_t>(
        firstByteOffset.value_or(
            m_vif1NextByteOffset) &
        0x0Fu);
    m_vif1DeferredData.insert(
        m_vif1DeferredData.end(), data, data + sizeBytes);

    if (!m_vif1DeferredSegments.empty())
    {
        Vif1StreamSegment &tail =
            m_vif1DeferredSegments.back();
        const uint8_t expectedOffset =
            static_cast<uint8_t>(
                (tail.firstByteOffset +
                 tail.byteCount) &
                0x0Fu);
        if (expectedOffset == startOffset)
        {
            tail.byteCount += sizeBytes;
            m_vif1NextByteOffset =
                static_cast<uint8_t>(
                    (startOffset + sizeBytes) &
                    0x0Fu);
            return;
        }
    }

    m_vif1DeferredSegments.push_back(
        {sizeBytes, startOffset});
    m_vif1NextByteOffset =
        static_cast<uint8_t>(
            (startOffset + sizeBytes) & 0x0Fu);
}

void PS2Memory::consumeVif1StreamSegments(
    size_t sizeBytes)
{
    while (sizeBytes != 0u &&
           !m_vif1DeferredSegments.empty())
    {
        Vif1StreamSegment &segment =
            m_vif1DeferredSegments.front();
        if (sizeBytes < segment.byteCount)
        {
            segment.firstByteOffset =
                static_cast<uint8_t>(
                    (segment.firstByteOffset +
                     sizeBytes) &
                    0x0Fu);
            segment.byteCount -= sizeBytes;
            return;
        }

        sizeBytes -= segment.byteCount;
        m_vif1DeferredSegments.pop_front();
    }
}

uint8_t PS2Memory::vif1StreamByteOffset(
    size_t position) const
{
    for (const Vif1StreamSegment &segment :
         m_vif1DeferredSegments)
    {
        if (position < segment.byteCount)
        {
            return static_cast<uint8_t>(
                (segment.firstByteOffset +
                 position) &
                0x0Fu);
        }
        position -= segment.byteCount;
    }
    return m_vif1NextByteOffset;
}

void PS2Memory::clearVif1Stream()
{
    m_vif1DeferredData.clear();
    m_vif1DeferredSegments.clear();
    m_vif1NextByteOffset = 0u;
}

PS2Memory::Vif1ParserDisposition
PS2Memory::processVif1Stream()
{
    if (m_vif1WaitingForVu)
        return Vif1ParserDisposition::WaitingForVu;
    if ((vif1_regs.stat & (1u << 10)) != 0u)
        return Vif1ParserDisposition::VifInterrupt;
    if (m_vif1DeferredData.empty())
    {
        vif1_regs.stat &= ~0x3u;
        return Vif1ParserDisposition::Drained;
    }

    const uint8_t *const data = m_vif1DeferredData.data();
    const uint32_t sizeBytes =
        static_cast<uint32_t>(
            std::min<size_t>(
                m_vif1DeferredData.size(),
                static_cast<size_t>(UINT32_MAX)));
    uint32_t pos = 0;
    bool stallAfterCommand = false;

    const auto finish =
        [&](Vif1ParserDisposition disposition)
        {
            if (pos != 0u)
            {
                consumeVif1StreamSegments(pos);
                m_vif1DeferredData.erase(
                    m_vif1DeferredData.begin(),
                    m_vif1DeferredData.begin() + pos);
            }

            vif1_regs.stat &= ~0x3u;
            switch (disposition)
            {
            case Vif1ParserDisposition::NeedMoreData:
                vif1_regs.stat |= 0x1u;
                break;
            case Vif1ParserDisposition::Drained:
            case Vif1ParserDisposition::WaitingForVu:
            case Vif1ParserDisposition::VifInterrupt:
            case Vif1ParserDisposition::GifPath:
                break;
            }
            return disposition;
        };

    while (pos + 4 <= sizeBytes)
    {
        if (stallAfterCommand)
        {
            vif1_regs.stat |= (1u << 10); // VIS
            return finish(
                Vif1ParserDisposition::VifInterrupt);
        }

        if (m_vif1PendingPath2DirectQwc != 0u)
        {
            if (!isVif1GifPathAvailable(
                    m_vif1PendingPath2DirectHl))
            {
                vif1_regs.stat |= (1u << 3); // VGW
                return finish(
                    Vif1ParserDisposition::GifPath);
            }
            vif1_regs.stat &= ~(1u << 3); // VGW

            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            if (availableQw == 0u)
            {
                break;
            }

            const uint32_t chunkQw = std::min<uint32_t>(m_vif1PendingPath2DirectQwc, availableQw);
            submitGifPacket(GifPathId::Path2,
                            data + pos,
                            chunkQw * 16u,
                            true,
                            m_vif1PendingPath2DirectHl);

            pos += chunkQw * 16u;
            m_vif1PendingPath2DirectQwc -= chunkQw;
            if (m_vif1PendingPath2DirectQwc == 0u)
            {
                m_vif1PendingPath2DirectHl = false;
                if (m_vif1PendingIrqAfterCommand)
                {
                    m_vif1PendingIrqAfterCommand = false;
                    vif1_regs.stat |= (1u << 10); // VIS
                    return finish(
                        Vif1ParserDisposition::
                            VifInterrupt);
                }
            }
            continue;
        }

        const uint32_t commandStart = pos;
        uint32_t cmd;
        memcpy(&cmd, data + pos, 4);
        pos += 4;

        uint8_t opcode = (cmd >> 24) & 0x7F;
        uint16_t imm = cmd & 0xFFFF;
        uint8_t num = (cmd >> 16) & 0xFF;
        const bool irq = (cmd & 0x80000000u) != 0u;

        // Track most-recent command for VIFn_CODE emulation.
        vif1_regs.code = cmd;
        vif1_regs.num = num;
        traceVif1Command(cmd, data + pos, sizeBytes - pos);
        if (irq)
            vif1_regs.stat |= (1u << 11); // INT

        const bool waitsForVu =
            opcode == VIF_FLUSHE ||
            opcode == VIF_FLUSH ||
            opcode == VIF_FLUSHA ||
            opcode == VIF_MSCAL ||
            opcode == VIF_MSCALF ||
            opcode == VIF_MSCNT;
        if (waitsForVu &&
            m_vu1BusyCallback &&
            m_vu1BusyCallback())
        {
            m_vif1WaitingForVu = true;
            vif1_regs.stat |= (1u << 2); // VEW
            pos = commandStart;
            return finish(
                Vif1ParserDisposition::WaitingForVu);
        }

        stallAfterCommand = irq;

        if (opcode == VIF_NOP)
        {
            continue;
        }
        else if (opcode == VIF_STCYCL)
        {
            vif1_regs.cycle = imm;
            submitVu1VifStateUpdate();
            continue;
        }
        else if (opcode == VIF_OFFSET)
        {
            // VIF double-buffer setup. OFFSET clears DBF and resets TOPS to BASE.
            // Do not rewrite BASE from the previous TOPS value.
            vif1_regs.ofst = imm & 0x3FFu;
            vif1_regs.tops = vif1_regs.base & 0x3FFu;
            vif1_regs.stat &= ~(1u << 7); // clear DBF
            submitVu1VifStateUpdate();
            continue;
        }
        else if (opcode == VIF_BASE)
        {
            // BASE only updates the base register. TOPS changes on OFFSET/MSCAL.
            vif1_regs.base = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_ITOP)
        {
            // ITOP VIFcode writes pending ITOPS; VU XITOP observes it after MSCAL/MSCNT.
            vif1_regs.itops = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_STMOD)
        {
            vif1_regs.mode = imm & 3u;
            submitVu1VifStateUpdate();
            continue;
        }
        else if (opcode == VIF_MSKPATH3)
        {
            // VIF command docs: MSKPATH3 uses IMMEDIATE bit 15.
            const bool wasMasked = isPath3Masked();
            m_path3Masked = (imm & 0x8000u) != 0u;
            updateGifStat();
            if (wasMasked && !isPath3Masked())
            {
                flushMaskedPath3Packets();
                (void)wakeGifDma(0u);
            }
            continue;
        }
        else if (opcode == VIF_MARK)
        {
            vif1_regs.mark = imm;
            vif1_regs.stat |= (1u << 6); // MRK
            continue;
        }
        else if (opcode == VIF_FLUSHE || opcode == VIF_FLUSH || opcode == VIF_FLUSHA)
        {
            continue;
        }
        else if (opcode == VIF_MSCAL || opcode == VIF_MSCALF)
        {
            uint32_t startPC = (uint32_t)imm * 8u;

            // Values visible to the VU program for this MSCAL.
            // DobieStation semantics: ITOP = ITOPS; TOP = current TOPS;
            // then TOPS/DBF are prepared for the next buffer.
            const uint32_t runTop = vif1_regs.tops & 0x3FFu;
            const uint32_t runItop = vif1_regs.itops & 0x3FFu;
            vif1_regs.top = runTop;
            vif1_regs.itop = runItop;

            const bool dbf = (vif1_regs.stat & (1u << 7)) != 0u;
            if (dbf)
                vif1_regs.tops = vif1_regs.base & 0x3FFu;
            else
                vif1_regs.tops = (vif1_regs.base + vif1_regs.ofst) & 0x3FFu;
            vif1_regs.stat ^= (1u << 7); // toggle DBF
            submitVu1VifStateUpdate();

            if (m_vu1MscalCallback)
            {
                m_vu1MscalCallback(
                    startPC, runTop, runItop,
                    opcode == VIF_MSCALF);
            }
            continue;
        }
        else if (opcode == VIF_MSCNT)
        {
            const uint32_t runTop = vif1_regs.tops & 0x3FFu;
            const uint32_t runItop = vif1_regs.itops & 0x3FFu;
            vif1_regs.top = runTop;
            vif1_regs.itop = runItop;

            const bool dbf = (vif1_regs.stat & (1u << 7)) != 0u;
            if (dbf)
                vif1_regs.tops = vif1_regs.base & 0x3FFu;
            else
                vif1_regs.tops = (vif1_regs.base + vif1_regs.ofst) & 0x3FFu;
            vif1_regs.stat ^= (1u << 7); // toggle DBF
            submitVu1VifStateUpdate();

            if (m_vu1MscntCallback)
                m_vu1MscntCallback(runTop, runItop);
            continue;
        }
        else if (opcode == VIF_STMASK)
        {
            if (pos + 4 > sizeBytes)
            {
                pos = commandStart;
                stallAfterCommand = false;
                break;
            }
            uint32_t maskValue = 0;
            std::memcpy(&maskValue, data + pos, sizeof(maskValue));
            vif1_regs.mask = maskValue;
            pos += 4;
            submitVu1VifStateUpdate();
            continue;
        }
        else if (opcode == VIF_STROW)
        {
            if (pos + 16 > sizeBytes)
            {
                pos = commandStart;
                stallAfterCommand = false;
                break;
            }
            std::memcpy(vif1_regs.row, data + pos, 16);
            pos += 16;
            submitVu1VifStateUpdate();
            continue;
        }
        else if (opcode == VIF_STCOL)
        {
            if (pos + 16 > sizeBytes)
            {
                pos = commandStart;
                stallAfterCommand = false;
                break;
            }
            std::memcpy(vif1_regs.col, data + pos, 16);
            pos += 16;
            submitVu1VifStateUpdate();
            continue;
        }
        else if (opcode == VIF_MPG)
        {
            const uint32_t destAddr =
                (static_cast<uint32_t>(imm) * 8u) &
                (PS2_VU1_CODE_SIZE - 1u);
            // VIF MPG semantics: NUM==0 means 256 instructions (2048 bytes).
            // MPG payload is instruction-packed and should not be QW-aligned.
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            if (pos + mpgBytes > sizeBytes)
            {
                pos = commandStart;
                stallAfterCommand = false;
                break;
            }
            if (m_vu1Code && mpgBytes > 0u)
            {
                const bool profileUpload =
                    isVu1WorkloadProfileEnabled();
                const bool identical =
                    profileUpload &&
                    equalsWrappedMicroMemory(
                        m_vu1Code, PS2_VU1_CODE_SIZE,
                        destAddr, data + pos, mpgBytes);
                const uint64_t generationBefore =
                    profileUpload ? getVU1CodeGeneration() : 0u;
                if (!writeVu1OwnerMemory(
                        m_vu1Code, destAddr,
                        data + pos, mpgBytes, true))
                {
                    copyWrappedMicroMemory(
                        m_vu1Code, PS2_VU1_CODE_SIZE,
                        destAddr, data + pos, mpgBytes);
                    markVU1CodeModified();
                }
                if (profileUpload)
                {
                    recordVu1WorkloadProfileCodeUpload(
                        destAddr, data + pos, mpgBytes, identical,
                        generationBefore, getVU1CodeGeneration());
                }
            }
            pos += mpgBytes;
            continue;
        }
        else if (opcode == VIF_DIRECT || opcode == VIF_DIRECTHL)
        {
            const bool directHl = (opcode == VIF_DIRECTHL);
            if (!isVif1GifPathAvailable(directHl))
            {
                vif1_regs.stat |= (1u << 3); // VGW
                pos = commandStart;
                return finish(
                    Vif1ParserDisposition::GifPath);
            }
            vif1_regs.stat &= ~(1u << 3); // VGW

            uint32_t qwCount = imm;
            if (qwCount == 0)
                qwCount = 65536;
            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            const bool truncated = qwCount > availableQw;
            if (qwCount > availableQw)
                qwCount = availableQw;

            if (qwCount > 0)
            {
                submitGifPacket(GifPathId::Path2, data + pos, qwCount * 16, true, directHl);

            }

            pos += qwCount * 16;
            if (truncated)
            {
                const uint32_t requestedQw = (imm == 0u) ? 65536u : static_cast<uint32_t>(imm);
                m_vif1PendingPath2DirectQwc = requestedQw - qwCount;
                m_vif1PendingPath2DirectHl = directHl;
                m_vif1PendingIrqAfterCommand = irq;
                stallAfterCommand = false;
                break;
            }
            continue;
        }
        else if ((opcode & 0x60) == 0x60)
        {
            uint8_t vn = (opcode >> 2) & 0x3;
            uint8_t vl = opcode & 0x3;
            const bool maskEnable = (opcode & 0x10u) != 0u;
            int components = vn + 1;
            int bitsPerComponent = 32;
            switch (vl)
            {
            case 0:
                bitsPerComponent = 32;
                break;
            case 1:
                bitsPerComponent = 16;
                break;
            case 2:
                bitsPerComponent = 8;
                break;
            case 3:
                bitsPerComponent = (vn == 3) ? 4 : 16;
                break;
            default:
                break;
            }
            int bitsPerVector = (vl == 3 && vn == 3) ? 16 : (components * bitsPerComponent);
            uint32_t bytesPerVector = (bitsPerVector + 7) / 8;
            // UNPACK semantics: NUM is 8-bit and NUM==0 means 256 vectors (writes).
            const uint32_t writeVectorCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);

            // STCYCL controls write cycles for UNPACK.
            uint32_t cl = vif1_regs.cycle & 0xFFu;
            uint32_t wl = (vif1_regs.cycle >> 8) & 0xFFu;
            if (cl == 0u)
                cl = 1u;
            if (wl == 0u)
                wl = 1u;

            uint32_t sourceVectorCount = writeVectorCount;
            if (cl < wl)
            {
                const uint32_t fullBlocks = writeVectorCount / wl;
                uint32_t remainder = writeVectorCount % wl;
                if (remainder > cl)
                    remainder = cl;
                sourceVectorCount = fullBlocks * cl + remainder;
            }

            uint32_t totalBytes = sourceVectorCount * bytesPerVector;
            totalBytes = (totalBytes + 3) & ~3u;
            if (pos + totalBytes > sizeBytes)
            {
                pos = commandStart;
                stallAfterCommand = false;
                break;
            }

            // PCSX2's VIF implementation records this as start_aligned.
            // Values 1..3 identify the source word within its input
            // quadword; 4 represents an exactly aligned source.
            const uint32_t sourceByteOffset =
                vif1StreamByteOffset(pos);
            const uint32_t sourceWordAlignment =
                sourceByteOffset == 0u
                    ? 4u
                    : sourceByteOffset / 4u;

            uint32_t vuAddr = (uint32_t)imm & 0x3FFu;
            if ((imm & 0x8000u) != 0u)
                vuAddr = (vuAddr + (vif1_regs.tops & 0x3FFu)) & 0x3FFu;

            const bool zeroExtend = (imm & 0x4000u) != 0u;

            if (m_vu1Data && totalBytes > 0u)
            {
                Vu1DecodedUnpackCommand unpack{
                    .immediate = imm,
                    .vectorLength = vl,
                    .componentCount =
                        static_cast<uint8_t>(components),
                    .writeVectorCount = static_cast<uint16_t>(
                        writeVectorCount),
                    .sourceVectorCount = static_cast<uint16_t>(
                        sourceVectorCount),
                    .sourceWordAlignment = static_cast<uint8_t>(
                        sourceWordAlignment),
                    .maskEnabled = maskEnable,
                    .zeroExtend = zeroExtend,
                    .bytes = std::vector<uint8_t>(
                        data + pos, data + pos + totalBytes),
                };
                if (submitVu1OwnerCommand(std::move(unpack)))
                {
                    pos += totalBytes;
                    continue;
                }
            }

            if (m_vu1Data && totalBytes > 0 && pos + totalBytes <= sizeBytes)
            {
                const uint8_t *srcBase = data + pos;
                const uint32_t mode = vif1_regs.mode & 3u;
                if (!maskEnable && vn == 3u && vl <= 2u &&
                    cl == wl && mode <= 1u &&
                    tryUnpackContiguousV4(
                        m_vu1Data, PS2_VU1_DATA_SIZE, vuAddr,
                        srcBase, writeVectorCount, vl,
                        zeroExtend, mode, vif1_regs.row))
                {
                    pos += totalBytes;
                    continue;
                }
                uint32_t srcIndex = 0u;
                for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
                {
                    const uint32_t cyclePos = writeIndex % wl;
                    const bool sourceAvailable = (cl >= wl) || (cyclePos < cl);

                    uint32_t destVec = 0;
                    if (cl >= wl)
                    {
                        destVec = (vuAddr + (writeIndex / wl) * cl + cyclePos) & 0x3FFu;
                    }
                    else
                    {
                        destVec = (vuAddr + writeIndex) & 0x3FFu;
                    }

                    uint32_t destOff = destVec * 16u;
                    if (destOff + 16u > PS2_VU1_DATA_SIZE)
                    {
                        if (sourceAvailable && srcIndex < sourceVectorCount)
                            ++srcIndex;
                        continue;
                    }

                    uint32_t lanes[4] = {0u, 0u, 0u, 0u};
                    std::memcpy(lanes, m_vu1Data + destOff, sizeof(lanes));
                    uint32_t decompressed[4] = {lanes[0], lanes[1], lanes[2], lanes[3]};
                    bool decoded = false;

                    const uint8_t *srcVec = nullptr;
                    if (sourceAvailable && srcIndex < sourceVectorCount)
                    {
                        srcVec = srcBase + srcIndex * bytesPerVector;
                        ++srcIndex;
                        decoded = true;
                    }

                    const auto sourceHasBytes =
                        [&](uint32_t offset, uint32_t count) -> bool
                        {
                            if (!srcVec)
                                return false;
                            const size_t sourceOffset =
                                static_cast<size_t>(srcVec - srcBase);
                            return sourceOffset + offset + count <=
                                   static_cast<size_t>(totalBytes);
                        };

                    auto extend16 = [&](uint16_t raw) -> uint32_t
                    {
                        if (zeroExtend)
                            return static_cast<uint32_t>(raw);
                        return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(raw)));
                    };

                    auto extend8 = [&](uint8_t raw) -> uint32_t
                    {
                        if (zeroExtend)
                            return static_cast<uint32_t>(raw);
                        return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(raw)));
                    };

                    bool handledFormat = true;
                    if (!decoded)
                    {
                        handledFormat = false;
                    }
                    else if (vl == 0u)
                    {
                        if (components == 1)
                        {
                            uint32_t scalar = 0;
                            std::memcpy(&scalar, srcVec, sizeof(scalar));
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            // V3 uses the V4 decoder on real hardware. Its W
                            // lane therefore reads the adjacent packed source
                            // component while the source still advances by
                            // only three components. V2 repeats XY into ZW.
                            const uint32_t limit =
                                components == 3
                                    ? 4u
                                    : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                if (!sourceHasBytes(c * 4u, 4u))
                                    break;
                                uint32_t scalar = 0;
                                std::memcpy(&scalar, srcVec + c * 4u, sizeof(scalar));
                                decompressed[c] = scalar;
                            }
                            if (components == 2)
                            {
                                decompressed[2] = decompressed[0];
                                decompressed[3] = decompressed[1];
                            }
                        }
                    }
                    else if (vl == 1u)
                    {
                        if (components == 1)
                        {
                            uint16_t raw = 0;
                            std::memcpy(&raw, srcVec, sizeof(raw));
                            const uint32_t scalar = extend16(raw);
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit =
                                components == 3
                                    ? 4u
                                    : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                if (!sourceHasBytes(c * 2u, 2u))
                                    break;
                                uint16_t raw = 0;
                                std::memcpy(&raw, srcVec + c * 2u, sizeof(raw));
                                decompressed[c] = extend16(raw);
                            }
                            if (components == 2)
                            {
                                decompressed[2] = decompressed[0];
                                decompressed[3] = decompressed[1];
                            }
                        }
                    }
                    else if (vl == 2u)
                    {
                        if (components == 1)
                        {
                            const uint32_t scalar = extend8(srcVec[0]);
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit =
                                components == 3
                                    ? 4u
                                    : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                if (!sourceHasBytes(c, 1u))
                                    break;
                                decompressed[c] = extend8(srcVec[c]);
                            }
                            if (components == 2)
                            {
                                decompressed[2] = decompressed[0];
                                decompressed[3] = decompressed[1];
                            }
                        }
                    }
                    else if (vl == 3u && vn == 3u)
                    {
                        // V4-5: packed color-like format in a single 16-bit value.
                        uint16_t packed = 0;
                        std::memcpy(&packed, srcVec, sizeof(packed));
                        decompressed[0] = packed & 0x1Fu;
                        decompressed[1] = (packed >> 5) & 0x1Fu;
                        decompressed[2] = (packed >> 10) & 0x1Fu;
                        decompressed[3] = (packed >> 15) & 0x01u;
                    }
                    else
                    {
                        handledFormat = false;
                    }

                    // The hardware V3 decoder fetches a fourth component,
                    // but selected source-quadword boundaries force W to
                    // zero. These alignment rules are deterministic despite
                    // the manual describing the lane as indeterminate.
                    if (decoded && components == 3)
                    {
                        bool zeroW = false;
                        if (vl == 0u)
                        {
                            const uint32_t iteration =
                                (srcIndex - 1u) & 1u;
                            zeroW =
                                iteration !=
                                sourceWordAlignment;
                        }
                        else if (vl == 1u)
                        {
                            const uint32_t iteration =
                                srcIndex;
                            const uint32_t boundary =
                                ((iteration / 4u) + 1u +
                                 (4u -
                                  sourceWordAlignment)) &
                                3u;
                            zeroW =
                                (iteration & 1u) == 0u &&
                                boundary == 0u;
                        }
                        else if (vl == 2u)
                        {
                            zeroW =
                                srcIndex !=
                                sourceWordAlignment;
                        }

                        if (zeroW)
                            decompressed[3] = 0u;
                    }

                    // V2-32 also exposes a hardware-tested zero W lane. The
                    // narrower V2 formats retain their XYXY duplication.
                    if (decoded && components == 2 &&
                        vl == 0u)
                    {
                        decompressed[3] = 0u;
                    }

                    // Unknown compressed format fallback: preserve legacy raw-copy behavior.
                    if (!handledFormat && decoded && !maskEnable && (vif1_regs.mode == 0u || vif1_regs.mode == 3u))
                    {
                        uint32_t copyBytes = (bytesPerVector < 16u) ? bytesPerVector : 16u;
                        std::memcpy(m_vu1Data + destOff, srcVec, copyBytes);
                        continue;
                    }

                    const bool canAdd = (vl != 3u || vn != 3u);
                    const uint32_t colIdx = (cyclePos > 3u) ? 3u : cyclePos;
                    const uint32_t maskCycle = (cyclePos > 3u) ? 3u : cyclePos;

                    for (uint32_t field = 0u; field < 4u; ++field)
                    {
                        uint32_t maskSpec = 0u;
                        if (maskEnable)
                        {
                            const uint32_t shift = ((maskCycle * 4u) + field) * 2u;
                            maskSpec = (vif1_regs.mask >> shift) & 0x3u;
                        }

                        // In fill-write cycles with suspended source reads, treat raw-data selections as row-fill.
                        if (!decoded && maskSpec == 0u)
                            maskSpec = 1u;

                        uint32_t writeVal = lanes[field];
                        if (maskSpec == 0u)
                        {
                            if (handledFormat)
                            {
                                writeVal = decompressed[field];
                                if (canAdd && (mode == 1u || mode == 2u))
                                {
                                    writeVal = writeVal + vif1_regs.row[field];
                                    if (mode == 2u)
                                        vif1_regs.row[field] = writeVal;
                                }
                            }
                        }
                        else if (maskSpec == 1u)
                        {
                            writeVal = vif1_regs.row[field];
                        }
                        else if (maskSpec == 2u)
                        {
                            writeVal = vif1_regs.col[colIdx];
                        }
                        else
                        {
                            continue; // write-protect
                        }

                        lanes[field] = writeVal;
                    }

                    std::memcpy(m_vu1Data + destOff, lanes, sizeof(lanes));
                }
            }
            pos += totalBytes;
            continue;
        }
        else
        {
            continue;
        }
    }

    if (stallAfterCommand)
    {
        vif1_regs.stat |= (1u << 10); // VIS
        return finish(
            Vif1ParserDisposition::VifInterrupt);
    }
    if (pos == sizeBytes &&
        m_vif1PendingPath2DirectQwc == 0u)
    {
        return finish(Vif1ParserDisposition::Drained);
    }
    return finish(Vif1ParserDisposition::NeedMoreData);
}
