#include "module_factories.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kSony989sndSid = 0x00123456u;
        constexpr uint32_t kSony989sndLoadingSid = 0x00123457u;
        constexpr uint32_t kRpcStartSoundSystem = 0u;
        constexpr uint32_t kRpcConfigureSoundSystem = 0x2Au;
        constexpr uint32_t kRpcWaitForLoads = 0x36u;
        constexpr uint32_t kRpcOpenVagStreaming = 0x3Bu;
        constexpr uint32_t kRpcCloseVagStreaming = 0x3Cu;
        constexpr uint32_t kRpcStopVagStreaming = 0x3Du;
        constexpr uint32_t kRpcStartVagStreaming = 0x3Eu;
        constexpr uint32_t kRpcCommandBatch = 0x4Du;
        constexpr uint32_t kRpcSubmitVagStreaming = 0x5Au;
        constexpr uint32_t kRpcGetVagStreamingPosition = 0x5Bu;
        constexpr uint32_t kLoadingRpcLoadBankBySector = 3u;
        constexpr uint32_t kStartRequestSize = sizeof(uint32_t);
        constexpr uint32_t kConfigureRequestSize = 4u * sizeof(uint32_t);
        constexpr uint32_t kWaitForLoadsRequestSize = sizeof(uint32_t);
        constexpr uint32_t kOpenVagStreamingRequestSize = 6u * sizeof(uint32_t);
        constexpr uint32_t kStartVagStreamingRequestSize = 5u * sizeof(uint32_t);
        constexpr uint32_t kSubmitVagStreamingRequestSize = 2u * sizeof(uint32_t);
        constexpr uint32_t kStreamingAllocationAlignment = alignof(uint32_t);
        constexpr uint32_t kCommandCountSize = sizeof(uint32_t);
        constexpr uint32_t kCommandRecordHeaderSize = sizeof(uint32_t);
        constexpr uint32_t kMaximumCommandBatchCount = 1024u;
        constexpr uint32_t kCommandIdMask = 0xFFFFu;
        constexpr uint32_t kCommandPayloadSizeShift = 16u;
        constexpr uint32_t kCommandId08 = 0x08u;
        constexpr uint32_t kCommandId09 = 0x09u;
        constexpr uint32_t kCommandId0B = 0x0Bu;
        constexpr uint32_t kCommandId0D = 0x0Du;
        constexpr uint32_t kCommandIdPlaySound = 0x11u;
        constexpr uint32_t kCommandIdSoundIsStillPlaying = 0x19u;
        constexpr uint32_t kCommandIdSetSoundParams = 0x21u;
        constexpr uint32_t kCommandIdDataReadCancel = 0x37u;
        constexpr uint32_t kCommandIdDataRead = 0x38u;
        constexpr uint32_t kCommandId4E = 0x4Eu;
        constexpr uint32_t kCommandId51 = 0x51u;
        constexpr uint32_t kCdSectorSize = 2048u;
        constexpr uint32_t kSoundBankLoadRequestSize = 2u * sizeof(uint32_t);
        constexpr uint32_t kSoundBankContainerHeaderSize = 2u * sizeof(uint32_t);
        constexpr uint32_t kSoundBankChunkDescriptorSize = 2u * sizeof(uint32_t);
        constexpr uint32_t kMinimumSoundBankChunkCount = 2u;
        constexpr uint32_t kMaximumSoundBankChunkCount = 3u;
        constexpr uint32_t kSoundBankFileType1 = 1u;
        constexpr uint32_t kSoundBankFileType3 = 3u;
        constexpr uint32_t kSfxBlockDataId = 0x6B6C4253u; // "SBlk"
        constexpr uint32_t kMusicBankDataId = 0x32764253u; // "SBv2"
        constexpr uint32_t kEeRamSize = 32u * 1024u * 1024u;
        constexpr uint32_t kSpuAdpcmBlockBytes = 16u;
        constexpr uint32_t kSpuAdpcmSamplesPerBlock = 28u;
        constexpr uint32_t kMaximumSpuSampleRate = 192000u;
        constexpr uint64_t kNanosecondsPerSecond = 1'000'000'000u;

        struct CompletionResponse
        {
            uint32_t leadingSentinel;
            uint32_t result;
            uint32_t trailingSentinel;
        };

        static_assert(sizeof(CompletionResponse) == 12u);

        struct SoundBankChunk
        {
            uint32_t offset = 0u;
            uint32_t size = 0u;
        };

        struct LoadedSoundBank
        {
            uint32_t handle = 0u;
            uint32_t sector = 0u;
            uint32_t byteOffset = 0u;
            uint32_t dataId = 0u;
            uint64_t size = 0u;
            bool ready = false;
            std::vector<SoundBankChunk> chunks;
        };

        struct ParsedSoundCommand
        {
            uint32_t id = 0u;
            uint32_t payloadAddress = 0u;
            uint32_t payloadSize = 0u;
        };

        class Sony989sndService final : public IopService
        {
        public:
            explicit Sony989sndService(IopHost &host)
                : m_host(host)
            {
            }

            [[nodiscard]] std::string_view name() const override
            {
                return "sony-989snd";
            }

            [[nodiscard]] std::span<const uint32_t> sids() const override
            {
                return kSids;
            }

            void reset() override
            {
                releaseStreamingResource();
                m_loadedSoundBanks.clear();
                m_activeSoundHandles.clear();
                m_loadInProgress = false;
                m_dataReadCompletionAddress = 0u;
                m_lastResponseResult = 1u;
            }

            [[nodiscard]] RpcResult handleRpc(const RpcRequest &request) override
            {
                if (request.sid == kSony989sndLoadingSid)
                {
                    return handleLoadingRpc(request);
                }

                RpcResult result;
                if (request.sid != kSony989sndSid ||
                    (request.send.size != 0u && request.send.address == 0u) ||
                    request.receive.address == 0u ||
                    request.receive.size < sizeof(CompletionResponse))
                {
                    return result;
                }

                if (request.function == kRpcCommandBatch)
                {
                    return handleCommandBatch(request);
                }

                uint32_t responseResult = m_lastResponseResult;
                bool supported = false;
                if (request.function == kRpcStartSoundSystem &&
                    request.send.size == kStartRequestSize)
                {
                    uint32_t completionAddress = 0u;
                    supported = m_host.readGuest(request.send.address,
                                                 &completionAddress,
                                                 sizeof(completionAddress)) &&
                                configureDataReadCompletion(completionAddress);
                    if (supported)
                    {
                        responseResult = 1u;
                    }
                }
                if (request.function == kRpcConfigureSoundSystem &&
                    request.send.size == kConfigureRequestSize)
                {
                    supported = true;
                    responseResult = 1u;
                }
                if (request.function == kRpcWaitForLoads &&
                    request.send.size == kWaitForLoadsRequestSize)
                {
                    uint32_t wait = 0u;
                    if (!m_host.readGuest(request.send.address, &wait, sizeof(wait)))
                    {
                        return result;
                    }
                    (void)wait;

                    // The IRX returns zero to a non-blocking poll while work is
                    // active and one once all bank-loading work is idle. Host
                    // bank loads are synchronous, so a successful loading RPC
                    // has completed before this main-server RPC can run.
                    supported = true;
                    responseResult = m_loadInProgress ? 0u : 1u;
                }
                if (request.function == kRpcOpenVagStreaming &&
                    request.send.size == kOpenVagStreamingRequestSize)
                {
                    std::array<uint32_t, 6> streamingConfiguration{};
                    if (!m_host.readGuest(request.send.address,
                                          streamingConfiguration.data(),
                                          sizeof(streamingConfiguration)))
                    {
                        return result;
                    }

                    supported = true;
                    responseResult = openVagStreaming(streamingConfiguration);
                }
                if (request.function == kRpcCloseVagStreaming &&
                    request.send.size == 0u)
                {
                    closeVagStreaming();
                    supported = true;
                    // The original RPC leaves the shared result word untouched.
                }
                if (request.function == kRpcStopVagStreaming &&
                    request.send.size == 0u)
                {
                    stopVagStreaming();
                    supported = true;
                    // The original RPC leaves the shared result word untouched.
                }
                if (request.function == kRpcStartVagStreaming &&
                    request.send.size == kStartVagStreamingRequestSize)
                {
                    std::array<uint32_t, 5> playback{};
                    if (!m_host.readGuest(request.send.address,
                                          playback.data(),
                                          sizeof(playback)))
                    {
                        return result;
                    }

                    supported = startVagStreaming(playback);
                    // Like transfer submission, the original RPC leaves the
                    // shared result word untouched.
                }
                if (request.function == kRpcSubmitVagStreaming &&
                    request.send.size == kSubmitVagStreamingRequestSize)
                {
                    std::array<uint32_t, 2> transfer{};
                    if (!m_host.readGuest(request.send.address,
                                          transfer.data(),
                                          sizeof(transfer)))
                    {
                        return result;
                    }

                    supported = submitVagStreaming(transfer[0], transfer[1]);
                    // The original server does not write a result for this RPC.
                    // Its shared response record therefore retains the result of
                    // the preceding result-producing call.
                }
                if (request.function == kRpcGetVagStreamingPosition &&
                    request.send.size == 0u &&
                    m_streamingWorkArea != 0u)
                {
                    supported = true;
                    responseResult = advanceVagStreamingPosition();
                }
                if (!supported)
                {
                    return result;
                }

                const CompletionResponse response{
                    0xFFFFFFFFu,
                    responseResult,
                    0xFFFFFFFFu,
                };
                if (!m_host.writeGuest(request.receive.address, &response, sizeof(response)))
                {
                    return result;
                }

                if (request.function != kRpcCloseVagStreaming &&
                    request.function != kRpcStopVagStreaming &&
                    request.function != kRpcStartVagStreaming &&
                    request.function != kRpcSubmitVagStreaming)
                {
                    m_lastResponseResult = responseResult;
                }

                result.handled = true;
                result.resultAddress = request.receive.address;
                return result;
            }

        private:
            [[nodiscard]] RpcResult handleCommandBatch(const RpcRequest &request)
            {
                RpcResult result;
                if (request.send.address == 0u ||
                    request.send.size < kCommandCountSize + kCommandRecordHeaderSize)
                {
                    return result;
                }

                uint32_t commandCount = 0u;
                if (!m_host.readGuest(request.send.address,
                                      &commandCount,
                                      sizeof(commandCount)) ||
                    commandCount == 0u ||
                    commandCount > kMaximumCommandBatchCount ||
                    commandCount >
                        (request.send.size - kCommandCountSize) /
                            kCommandRecordHeaderSize)
                {
                    return result;
                }

                const uint64_t responseSize =
                    (static_cast<uint64_t>(commandCount) + 2u) * sizeof(uint32_t);
                if (responseSize > request.receive.size ||
                    responseSize > std::numeric_limits<size_t>::max() ||
                    static_cast<uint64_t>(request.receive.address) + responseSize >
                        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1u)
                {
                    return result;
                }

                std::vector<ParsedSoundCommand> commands;
                commands.reserve(commandCount);
                uint64_t cursor = kCommandCountSize;
                for (uint32_t index = 0u; index < commandCount; ++index)
                {
                    if (cursor + kCommandRecordHeaderSize > request.send.size)
                    {
                        return result;
                    }

                    const uint64_t recordAddress =
                        static_cast<uint64_t>(request.send.address) + cursor;
                    if (recordAddress > std::numeric_limits<uint32_t>::max())
                    {
                        return result;
                    }

                    uint32_t encodedHeader = 0u;
                    if (!m_host.readGuest(static_cast<uint32_t>(recordAddress),
                                          &encodedHeader,
                                          sizeof(encodedHeader)))
                    {
                        return result;
                    }

                    const uint32_t commandId = encodedHeader & kCommandIdMask;
                    const uint32_t payloadSize =
                        encodedHeader >> kCommandPayloadSizeShift;
                    const uint32_t paddedPayloadSize =
                        (payloadSize + (alignof(uint32_t) - 1u)) &
                        ~(alignof(uint32_t) - 1u);
                    const uint64_t payloadOffset =
                        cursor + kCommandRecordHeaderSize;
                    const uint64_t nextCursor =
                        payloadOffset + paddedPayloadSize;
                    const uint64_t payloadAddress =
                        static_cast<uint64_t>(request.send.address) + payloadOffset;
                    if (nextCursor > request.send.size)
                    {
                        warnCommandRecord(index,
                                          commandId,
                                          payloadSize,
                                          cursor,
                                          request.send.size,
                                          "record exceeds request buffer");
                        return result;
                    }
                    if (payloadAddress > std::numeric_limits<uint32_t>::max())
                    {
                        warnCommandRecord(index,
                                          commandId,
                                          payloadSize,
                                          cursor,
                                          request.send.size,
                                          "payload address overflows guest address space");
                        return result;
                    }
                    if (!isSupportedCommand(commandId, payloadSize))
                    {
                        warnCommandRecord(index,
                                          commandId,
                                          payloadSize,
                                          cursor,
                                          request.send.size,
                                          "unsupported command shape");
                        return result;
                    }

                    commands.push_back({
                        commandId,
                        static_cast<uint32_t>(payloadAddress),
                        payloadSize,
                    });
                    cursor = nextCursor;
                }
                if (cursor != request.send.size)
                {
                    warnCommandRecord(commandCount,
                                      0u,
                                      0u,
                                      cursor,
                                      request.send.size,
                                      "trailing bytes after command records");
                    return result;
                }

                // The 989snd server keeps one four-byte result slot per
                // command, bracketed by queue sentinels. Commands which do not
                // explicitly return a value retain the server's success word.
                std::vector<uint32_t> response(commandCount + 2u, 1u);
                response.front() = 0xFFFFFFFFu;
                response.back() = 0xFFFFFFFFu;

                for (uint32_t index = 0u; index < commands.size(); ++index)
                {
                    const ParsedSoundCommand &command = commands[index];
                    if (command.id == kCommandId08)
                    {
                        for (LoadedSoundBank &bank : m_loadedSoundBanks)
                        {
                            bank.ready = true;
                        }
                    }
                    if (command.id == kCommandIdDataRead)
                    {
                        std::array<uint32_t, 3> dataRead{};
                        if (!m_host.readGuest(command.payloadAddress,
                                              dataRead.data(),
                                              sizeof(dataRead)))
                        {
                            warnCommandBatch("cannot read data-read payload");
                            return result;
                        }
                        if (m_dataReadCompletionAddress == 0u)
                        {
                            warnCommandBatch("sound work area is not configured");
                            return result;
                        }
                        if (!readCdSectors(dataRead[0], dataRead[1], dataRead[2]))
                        {
                            return result;
                        }
                        if (!signalDataReadCompletion())
                        {
                            warnCommandBatch("cannot publish data-read completion");
                            return result;
                        }
                    }
                    if (command.id == kCommandIdPlaySound)
                    {
                        const uint32_t soundHandle =
                            m_host.allocateIopHandle(IopHandleKind::ServiceResource);
                        if (soundHandle == 0u)
                        {
                            warnSoundHandleAllocation(index);
                            return result;
                        }
                        response[index + 1u] = soundHandle;
                        m_activeSoundHandles.push_back(soundHandle);
                    }
                    if (command.id == kCommandIdSoundIsStillPlaying)
                    {
                        uint32_t soundHandle = 0u;
                        if (!m_host.readGuest(command.payloadAddress,
                                              &soundHandle,
                                              sizeof(soundHandle)))
                        {
                            warnCommandBatch("cannot read sound-status payload");
                            return result;
                        }
                        response[index + 1u] = soundIsStillPlaying(soundHandle);
                    }
                    if (command.id == kCommandIdSetSoundParams)
                    {
                        std::array<uint32_t, 6> parameters{};
                        if (!m_host.readGuest(command.payloadAddress,
                                              parameters.data(),
                                              sizeof(parameters)))
                        {
                            warnCommandBatch("cannot read sound-parameter payload");
                            return result;
                        }
                        response[index + 1u] =
                            activeSoundHandle(parameters[0]);
                    }
                }

                if (!m_host.writeGuest(request.receive.address,
                                       response.data(),
                                       static_cast<size_t>(responseSize)))
                {
                    return result;
                }

                for (uint32_t index = 0u; index < commands.size(); ++index)
                {
                    const ParsedSoundCommand &command = commands[index];
                    m_host.audioCommand(kSony989sndSid,
                                        command.id,
                                        {command.payloadAddress, command.payloadSize},
                                        {static_cast<uint32_t>(
                                             static_cast<uint64_t>(request.receive.address) +
                                             (index + 1u) * sizeof(uint32_t)),
                                         sizeof(uint32_t)});
                }

                m_lastResponseResult = 1u;
                result.handled = true;
                result.resultAddress = request.receive.address;
                return result;
            }

            [[nodiscard]] static bool isSupportedCommand(uint32_t commandId,
                                                          uint32_t payloadSize)
            {
                return (commandId == kCommandId08 && payloadSize == 0u) ||
                       (commandId == kCommandId09 &&
                        payloadSize == 2u * sizeof(uint32_t)) ||
                       (commandId == kCommandId0B &&
                        payloadSize == sizeof(uint32_t)) ||
                       (commandId == kCommandId0D &&
                        payloadSize == 2u * sizeof(uint32_t)) ||
                       (commandId == kCommandIdPlaySound &&
                        payloadSize == 6u * sizeof(uint32_t)) ||
                       (commandId == kCommandIdSoundIsStillPlaying &&
                        payloadSize == sizeof(uint32_t)) ||
                       (commandId == kCommandIdSetSoundParams &&
                        payloadSize == 6u * sizeof(uint32_t)) ||
                       (commandId == kCommandIdDataReadCancel &&
                        payloadSize == 0u) ||
                       (commandId == kCommandIdDataRead &&
                        payloadSize == 3u * sizeof(uint32_t)) ||
                       (commandId == kCommandId4E &&
                        payloadSize == 3u * sizeof(uint32_t)) ||
                       (commandId == kCommandId51 &&
                        payloadSize == 2u * sizeof(uint32_t));
            }

            [[nodiscard]] RpcResult handleLoadingRpc(const RpcRequest &request)
            {
                RpcResult result;
                if (request.function != kLoadingRpcLoadBankBySector ||
                    request.send.address == 0u ||
                    request.send.size != kSoundBankLoadRequestSize ||
                    request.receive.address == 0u ||
                    request.receive.size < sizeof(uint32_t))
                {
                    return result;
                }

                std::array<uint32_t, 2> source{};
                if (!m_host.readGuest(request.send.address,
                                      source.data(),
                                      sizeof(source)))
                {
                    return result;
                }

                m_loadInProgress = true;
                const uint32_t bankHandle = loadSoundBankFromCd(source[0], source[1]);
                m_loadInProgress = false;
                if (!m_host.writeGuest(request.receive.address,
                                       &bankHandle,
                                       sizeof(bankHandle)))
                {
                    return result;
                }

                result.handled = true;
                result.resultAddress = request.receive.address;
                return result;
            }

            [[nodiscard]] uint32_t soundIsStillPlaying(uint32_t soundHandle) const
            {
                if (soundHandle == 0u || soundHandle == 0xFFFFFFFFu)
                {
                    return soundHandle;
                }
                return std::find(m_activeSoundHandles.begin(),
                                 m_activeSoundHandles.end(),
                                 soundHandle) != m_activeSoundHandles.end()
                           ? soundHandle
                           : 0u;
            }

            [[nodiscard]] uint32_t activeSoundHandle(uint32_t soundHandle) const
            {
                return std::find(m_activeSoundHandles.begin(),
                                 m_activeSoundHandles.end(),
                                 soundHandle) != m_activeSoundHandles.end()
                           ? soundHandle
                           : 0u;
            }

            [[nodiscard]] uint32_t loadSoundBankFromCd(uint32_t sector,
                                                       uint32_t byteOffset)
            {
                const uint64_t sourceOffset =
                    static_cast<uint64_t>(sector) * kCdSectorSize + byteOffset;
                const uint64_t fileHandle =
                    m_host.openHostFile(m_host.hostPath(HostPathKind::CdImage));
                if (fileHandle == 0u)
                {
                    warnSoundBankLoad(sector, byteOffset, "CD image unavailable");
                    return 0u;
                }

                auto fail = [&](const char *reason) -> uint32_t {
                    m_host.closeHostFile(fileHandle);
                    warnSoundBankLoad(sector, byteOffset, reason);
                    return 0u;
                };
                auto readExact = [&](uint64_t offset, void *destination, size_t size) {
                    size_t bytesRead = 0u;
                    return m_host.readHostFile(fileHandle,
                                               offset,
                                               destination,
                                               size,
                                               bytesRead) &&
                           bytesRead == size;
                };

                uint64_t fileSize = 0u;
                std::array<uint32_t, 2> prefix{};
                if (!m_host.hostFileSize(fileHandle, fileSize) ||
                    sourceOffset > fileSize ||
                    kSoundBankContainerHeaderSize > fileSize - sourceOffset ||
                    !readExact(sourceOffset, prefix.data(), sizeof(prefix)))
                {
                    return fail("container header is outside the CD image");
                }

                const uint32_t fileType = prefix[0];
                const uint32_t chunkCount = prefix[1];
                if ((fileType != kSoundBankFileType1 &&
                     fileType != kSoundBankFileType3) ||
                    chunkCount < kMinimumSoundBankChunkCount ||
                    chunkCount > kMaximumSoundBankChunkCount)
                {
                    return fail("unsupported sound-bank container");
                }

                const uint64_t descriptorBytes =
                    static_cast<uint64_t>(chunkCount) * kSoundBankChunkDescriptorSize;
                const uint64_t headerSize =
                    kSoundBankContainerHeaderSize + descriptorBytes;
                if (headerSize > fileSize - sourceOffset)
                {
                    return fail("truncated sound-bank chunk table");
                }

                std::vector<uint32_t> descriptorWords(chunkCount * 2u);
                if (!readExact(sourceOffset + kSoundBankContainerHeaderSize,
                               descriptorWords.data(),
                               static_cast<size_t>(descriptorBytes)))
                {
                    return fail("failed to read sound-bank chunk table");
                }

                std::vector<SoundBankChunk> chunks;
                chunks.reserve(chunkCount);
                uint64_t previousEnd = headerSize;
                uint64_t containerSize = headerSize;
                for (uint32_t index = 0u; index < chunkCount; ++index)
                {
                    const SoundBankChunk chunk{
                        descriptorWords[index * 2u],
                        descriptorWords[index * 2u + 1u],
                    };
                    const uint64_t chunkEnd =
                        static_cast<uint64_t>(chunk.offset) + chunk.size;
                    if (chunk.size == 0u ||
                        chunk.offset < previousEnd ||
                        chunkEnd < chunk.offset ||
                        chunkEnd > fileSize - sourceOffset)
                    {
                        return fail("invalid sound-bank chunk range");
                    }
                    chunks.push_back(chunk);
                    previousEnd = chunkEnd;
                    containerSize = chunkEnd;
                }

                uint32_t dataId = 0u;
                if (chunks.front().size < sizeof(dataId) ||
                    !readExact(sourceOffset + chunks.front().offset,
                               &dataId,
                               sizeof(dataId)) ||
                    (dataId != kSfxBlockDataId && dataId != kMusicBankDataId) ||
                    (dataId == kMusicBankDataId && chunkCount != 3u))
                {
                    return fail("unsupported sound-bank data block");
                }

                const uint32_t bankHandle =
                    m_host.allocateIopHandle(IopHandleKind::ServiceResource);
                if (bankHandle == 0u)
                {
                    return fail("resource allocation failed");
                }

                m_host.closeHostFile(fileHandle);
                m_loadedSoundBanks.push_back({
                    bankHandle,
                    sector,
                    byteOffset,
                    dataId,
                    containerSize,
                    false,
                    std::move(chunks),
                });
                return bankHandle;
            }

            [[nodiscard]] uint32_t openVagStreaming(
                const std::array<uint32_t, 6> &configuration)
            {
                const uint32_t workAreaSize = configuration[0];
                if (workAreaSize == 0u ||
                    (workAreaSize & (kStreamingAllocationAlignment - 1u)) != 0u)
                {
                    warnStreamingAllocation(workAreaSize, "invalid work-area size");
                    return 0u;
                }

                releaseStreamingResource();
                const uint32_t workArea =
                    m_host.allocateIopHandle(IopHandleKind::ServiceResource);
                if (workArea == 0u)
                {
                    warnStreamingAllocation(workAreaSize, "allocation failed");
                    return 0u;
                }

                m_streamingConfiguration = configuration;
                m_streamingWorkArea = workArea;
                m_streamingPosition = 0u;
                m_streamingDataSubmitted = false;
                m_streamingPlaybackStarted = false;
                m_streamingSampleRate = 0u;
                m_streamingPlaybackStartNanoseconds = 0u;
                return workArea;
            }

            [[nodiscard]] bool startVagStreaming(
                const std::array<uint32_t, 5> &playback)
            {
                const uint32_t workArea = playback[0];
                const uint32_t workAreaBytes = playback[1];
                const uint32_t sampleRate = playback[3];
                const uint32_t channelCount = playback[4];
                if (m_streamingWorkArea == 0u ||
                    workArea != m_streamingWorkArea ||
                    workAreaBytes == 0u ||
                    workAreaBytes > m_streamingConfiguration[0] ||
                    sampleRate == 0u ||
                    sampleRate > kMaximumSpuSampleRate ||
                    channelCount == 0u ||
                    channelCount > 2u)
                {
                    warnStreamingPlayback(workArea,
                                          workAreaBytes,
                                          sampleRate,
                                          channelCount,
                                          "invalid playback configuration");
                    return false;
                }

                m_streamingPosition = 0u;
                m_streamingSampleRate = sampleRate;
                m_streamingPlaybackStartNanoseconds =
                    m_host.virtualTimeNanoseconds();
                m_streamingPlaybackStarted = true;
                return true;
            }

            [[nodiscard]] bool submitVagStreaming(uint32_t byteCount,
                                                  uint32_t destinationOffset)
            {
                if (m_streamingWorkArea == 0u ||
                    byteCount == 0u ||
                    byteCount > m_streamingConfiguration[0])
                {
                    warnStreamingTransfer(byteCount, destinationOffset,
                                          "invalid transfer size or unopened stream");
                    return false;
                }

                const uint64_t channelBufferSize = m_streamingConfiguration[1];
                const uint64_t destinationSize = channelBufferSize * 2u;
                if (channelBufferSize == 0u ||
                    destinationOffset > destinationSize ||
                    byteCount > destinationSize - destinationOffset)
                {
                    warnStreamingTransfer(byteCount, destinationOffset,
                                          "destination range is outside the stream buffers");
                    return false;
                }

                m_streamingDataSubmitted = true;
                return true;
            }

            [[nodiscard]] uint32_t advanceVagStreamingPosition()
            {
                const uint32_t channelBufferSize = m_streamingConfiguration[1];
                if (!m_streamingDataSubmitted ||
                    !m_streamingPlaybackStarted ||
                    channelBufferSize == 0u)
                {
                    return m_streamingPosition;
                }

                const uint64_t now = m_host.virtualTimeNanoseconds();
                if (now <= m_streamingPlaybackStartNanoseconds)
                {
                    return m_streamingPosition;
                }

                const uint64_t elapsed = now - m_streamingPlaybackStartNanoseconds;
                const uint64_t wholeSeconds = elapsed / kNanosecondsPerSecond;
                const uint64_t remainingNanoseconds = elapsed % kNanosecondsPerSecond;

                // One SPU ADPCM block consumes 16 encoded bytes and produces 28
                // samples. Keep only one ring cycle of the sample count so an
                // arbitrarily old virtual clock cannot overflow this service.
                const uint64_t blockCycles =
                    channelBufferSize / std::gcd(channelBufferSize, kSpuAdpcmBlockBytes);
                const uint64_t samplesPerCycle =
                    blockCycles * kSpuAdpcmSamplesPerBlock;
                const uint64_t wholeSamples =
                    ((wholeSeconds % samplesPerCycle) * m_streamingSampleRate) %
                    samplesPerCycle;
                const uint64_t partialSamples =
                    (remainingNanoseconds * m_streamingSampleRate) /
                    kNanosecondsPerSecond;
                const uint64_t samplePosition =
                    (wholeSamples + partialSamples) % samplesPerCycle;
                const uint64_t blockPosition =
                    samplePosition / kSpuAdpcmSamplesPerBlock;
                m_streamingPosition = static_cast<uint32_t>(
                    (blockPosition * kSpuAdpcmBlockBytes) % channelBufferSize);
                return m_streamingPosition;
            }

            void stopVagStreaming()
            {
                if (m_streamingPlaybackStarted)
                {
                    (void)advanceVagStreamingPosition();
                }
                m_streamingPlaybackStarted = false;
                m_streamingPlaybackStartNanoseconds = 0u;
            }

            void closeVagStreaming()
            {
                stopVagStreaming();
                releaseStreamingResource();
            }

            void releaseStreamingResource()
            {
                m_streamingWorkArea = 0u;
                m_streamingConfiguration.fill(0u);
                m_streamingPosition = 0u;
                m_streamingDataSubmitted = false;
                m_streamingPlaybackStarted = false;
                m_streamingSampleRate = 0u;
                m_streamingPlaybackStartNanoseconds = 0u;
            }

            void warnStreamingAllocation(uint32_t size, const char *reason)
            {
                std::ostringstream message;
                message << "[sony-989snd] " << reason
                        << " size=0x" << std::hex << size;
                m_host.log(LogLevel::Warning, message.str());
            }

            void warnStreamingTransfer(uint32_t size,
                                       uint32_t destinationOffset,
                                       const char *reason)
            {
                std::ostringstream message;
                message << "[sony-989snd] " << reason
                        << " size=0x" << std::hex << size
                        << " destination=0x" << destinationOffset;
                m_host.log(LogLevel::Warning, message.str());
            }

            void warnStreamingPlayback(uint32_t workArea,
                                       uint32_t workAreaBytes,
                                       uint32_t sampleRate,
                                       uint32_t channelCount,
                                       const char *reason)
            {
                std::ostringstream message;
                message << "[sony-989snd] " << reason
                        << " work_area=0x" << std::hex << workArea
                        << " bytes=0x" << workAreaBytes
                        << " sample_rate=" << std::dec << sampleRate
                        << " channels=" << channelCount;
                m_host.log(LogLevel::Warning, message.str());
            }

            [[nodiscard]] bool configureDataReadCompletion(uint32_t address)
            {
                uint32_t normalizedAddress = 0u;
                if ((address & (alignof(uint32_t) - 1u)) != 0u ||
                    !m_host.normalizeGuestAddress(address, normalizedAddress) ||
                    normalizedAddress > kEeRamSize - sizeof(uint32_t))
                {
                    m_dataReadCompletionAddress = 0u;
                    return false;
                }

                m_dataReadCompletionAddress = normalizedAddress;
                return true;
            }

            [[nodiscard]] bool signalDataReadCompletion()
            {
                constexpr uint32_t kComplete = 0u;
                return m_host.writeGuest(m_dataReadCompletionAddress,
                                         &kComplete,
                                         sizeof(kComplete));
            }

            [[nodiscard]] bool readCdSectors(uint32_t sector,
                                             uint32_t sectorCount,
                                             uint32_t destination)
            {
                const uint64_t offset = static_cast<uint64_t>(sector) * kCdSectorSize;
                const uint64_t byteCount = static_cast<uint64_t>(sectorCount) * kCdSectorSize;
                uint32_t normalizedDestination = 0u;
                if (sectorCount == 0u ||
                    offset > std::numeric_limits<uint64_t>::max() - byteCount ||
                    !m_host.normalizeGuestAddress(destination, normalizedDestination) ||
                    normalizedDestination >= kEeRamSize ||
                    byteCount > static_cast<uint64_t>(kEeRamSize - normalizedDestination))
                {
                    warnDataRead(sector, sectorCount, destination, "invalid range");
                    return false;
                }

                const uint64_t handle = m_host.openHostFile(m_host.hostPath(HostPathKind::CdImage));
                if (handle == 0u)
                {
                    warnDataRead(sector, sectorCount, destination, "CD image unavailable");
                    return false;
                }

                uint64_t fileSize = 0u;
                bool copied = m_host.hostFileSize(handle, fileSize) &&
                              offset <= fileSize &&
                              byteCount <= fileSize - offset;
                std::array<uint8_t, 16u * 1024u> chunk{};
                uint64_t copiedBytes = 0u;
                while (copied && copiedBytes < byteCount)
                {
                    const size_t wanted = static_cast<size_t>(
                        std::min<uint64_t>(chunk.size(), byteCount - copiedBytes));
                    size_t bytesRead = 0u;
                    copied = m_host.readHostFile(handle,
                                                 offset + copiedBytes,
                                                 chunk.data(),
                                                 wanted,
                                                 bytesRead) &&
                             bytesRead == wanted &&
                             m_host.writeGuest(normalizedDestination +
                                                   static_cast<uint32_t>(copiedBytes),
                                               chunk.data(),
                                               wanted);
                    copiedBytes += copied ? wanted : 0u;
                }
                m_host.closeHostFile(handle);

                if (!copied)
                {
                    warnDataRead(sector, sectorCount, destination, "host read failed");
                }
                return copied;
            }

            void warnDataRead(uint32_t sector,
                              uint32_t sectorCount,
                              uint32_t destination,
                              const char *reason)
            {
                std::ostringstream message;
                message << "[sony-989snd] " << reason
                        << " lbn=0x" << std::hex << sector
                        << " sectors=0x" << sectorCount
                        << " dst=0x" << destination;
                m_host.log(LogLevel::Warning, message.str());
            }

            void warnCommandBatch(const char *reason)
            {
                std::ostringstream message;
                message << "[sony-989snd] " << reason
                        << " work_area=0x" << std::hex
                        << m_dataReadCompletionAddress;
                m_host.log(LogLevel::Warning, message.str());
            }

            void warnCommandRecord(uint32_t index,
                                   uint32_t commandId,
                                   uint32_t payloadSize,
                                   uint64_t cursor,
                                   uint32_t requestSize,
                                   const char *reason)
            {
                std::ostringstream message;
                message << "[sony-989snd] " << reason
                        << " index=" << std::dec << index
                        << " command=0x" << std::hex << commandId
                        << " payload=0x" << payloadSize
                        << " cursor=0x" << cursor
                        << " request_size=0x" << requestSize;
                m_host.log(LogLevel::Warning, message.str());
            }

            void warnSoundHandleAllocation(uint32_t index)
            {
                std::ostringstream message;
                message << "[sony-989snd] sound handle allocation failed"
                        << " index=" << index;
                m_host.log(LogLevel::Warning, message.str());
            }

            void warnSoundBankLoad(uint32_t sector,
                                   uint32_t byteOffset,
                                   const char *reason)
            {
                std::ostringstream message;
                message << "[sony-989snd] " << reason
                        << " lbn=0x" << std::hex << sector
                        << " offset=0x" << byteOffset;
                m_host.log(LogLevel::Warning, message.str());
            }

            inline static constexpr std::array<uint32_t, 2> kSids{
                kSony989sndSid,
                kSony989sndLoadingSid,
            };

            IopHost &m_host;
            std::vector<LoadedSoundBank> m_loadedSoundBanks;
            std::vector<uint32_t> m_activeSoundHandles;
            std::array<uint32_t, 6> m_streamingConfiguration{};
            uint32_t m_streamingWorkArea = 0u;
            uint32_t m_streamingPosition = 0u;
            uint32_t m_dataReadCompletionAddress = 0u;
            uint32_t m_lastResponseResult = 1u;
            bool m_streamingDataSubmitted = false;
            bool m_streamingPlaybackStarted = false;
            bool m_loadInProgress = false;
            uint32_t m_streamingSampleRate = 0u;
            uint64_t m_streamingPlaybackStartNanoseconds = 0u;
        };
    }

    std::unique_ptr<IopService> createSony989sndService(IopHost &host)
    {
        return std::make_unique<Sony989sndService>(host);
    }
}
