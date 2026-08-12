#include "../module_factories.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kIopRamSize = 2u * 1024u * 1024u;

        class StashManagerService final : public IopService
        {
        public:
            StashManagerService(IopHost &host, StashManagerBindings bindings)
                : m_host(host),
                  m_bindings(std::move(bindings)),
                  m_sids{m_bindings.sid},
                  m_storage(m_bindings.arenaSize, 0u)
            {
            }

            [[nodiscard]] std::string_view name() const override
            {
                return m_bindings.serviceName;
            }

            [[nodiscard]] std::span<const uint32_t> sids() const override
            {
                return m_sids;
            }

            void reset() override
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                std::fill(m_storage.begin(), m_storage.end(), 0u);
                m_uploadCount = 0u;
                m_uploadBytes = 0u;
                m_readCount = 0u;
                m_readBytes = 0u;
                m_rejectedTransfers = 0u;
                m_rejectedReads = 0u;
            }

            [[nodiscard]] RpcResult handleRpc(const RpcRequest &request) override
            {
                if (request.sid != m_bindings.sid)
                {
                    return {};
                }

                RpcResult result;
                result.handled = true;
                result.resultAddress = request.receive.address;

                if (request.function == m_bindings.initializeFunction)
                {
                    initialize(request.receive);
                    return result;
                }

                if (request.function == m_bindings.readFunction)
                {
                    read(request.send, request.receive);
                    return result;
                }

                return result;
            }

            void onSifTransfer(const SifTransfer &transfer) override
            {
                if (transfer.kind != SifTransferKind::SetDma ||
                    transfer.phase != SifTransferPhase::AfterCopy ||
                    transfer.size == 0u)
                {
                    return;
                }

                uint32_t offset = 0u;
                if (!arenaRange(transfer.destinationAddress, transfer.size, offset))
                {
                    return;
                }

                std::vector<uint8_t> payload(transfer.size);
                if (!m_host.readGuest(transfer.sourceAddress, payload.data(), payload.size()))
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    ++m_rejectedTransfers;
                    return;
                }

                std::lock_guard<std::mutex> lock(m_mutex);
                std::memcpy(m_storage.data() + offset, payload.data(), payload.size());
                ++m_uploadCount;
                m_uploadBytes += payload.size();
            }

            void appendDebugMetrics(std::vector<DebugMetric> &metrics) const override
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                metrics.push_back({"sid", m_bindings.sid, true});
                metrics.push_back({"arena_base", m_bindings.arenaBase, true});
                metrics.push_back({"arena_size", m_bindings.arenaSize, true});
                metrics.push_back({"uploads", m_uploadCount, false});
                metrics.push_back({"upload_bytes", m_uploadBytes, false});
                metrics.push_back({"reads", m_readCount, false});
                metrics.push_back({"read_bytes", m_readBytes, false});
                metrics.push_back({"rejected_transfers", m_rejectedTransfers, false});
                metrics.push_back({"rejected_reads", m_rejectedReads, false});
            }

        private:
            void initialize(GuestBuffer receive)
            {
                if (receive.address == 0u || receive.size == 0u)
                {
                    return;
                }

                const uint32_t clearBytes = std::min(receive.size,
                                                     m_bindings.initializeResponseBytes);
                (void)m_host.zeroGuest(receive.address, clearBytes);
                if (receive.size >= sizeof(uint32_t))
                {
                    (void)m_host.writeGuest(receive.address,
                                            &m_bindings.arenaBase,
                                            sizeof(m_bindings.arenaBase));
                }
                if (receive.size >= 2u * sizeof(uint32_t))
                {
                    (void)m_host.writeGuest(receive.address + sizeof(uint32_t),
                                            &m_bindings.arenaSize,
                                            sizeof(m_bindings.arenaSize));
                }
            }

            void read(GuestBuffer send, GuestBuffer receive)
            {
                uint32_t sourceAddress = 0u;
                if (send.address == 0u || send.size < sizeof(sourceAddress) ||
                    !m_host.readGuest(send.address, &sourceAddress, sizeof(sourceAddress)))
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    ++m_rejectedReads;
                    return;
                }

                uint32_t offset = 0u;
                if (receive.address == 0u ||
                    !arenaRange(sourceAddress, receive.size, offset))
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    ++m_rejectedReads;
                    return;
                }

                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_host.writeGuest(receive.address,
                                      m_storage.data() + offset,
                                      receive.size))
                {
                    ++m_readCount;
                    m_readBytes += receive.size;
                }
                else
                {
                    ++m_rejectedReads;
                }
            }

            [[nodiscard]] bool arenaRange(uint32_t address,
                                          uint32_t size,
                                          uint32_t &offset) const
            {
                const uint32_t physicalAddress = address & 0x00FFFFFFu;
                if (physicalAddress < m_bindings.arenaBase)
                {
                    return false;
                }

                offset = physicalAddress - m_bindings.arenaBase;
                return offset <= m_bindings.arenaSize &&
                       size <= m_bindings.arenaSize - offset;
            }

            IopHost &m_host;
            StashManagerBindings m_bindings;
            std::array<uint32_t, 1> m_sids;
            std::vector<uint8_t> m_storage;
            mutable std::mutex m_mutex;
            uint64_t m_uploadCount = 0u;
            uint64_t m_uploadBytes = 0u;
            uint64_t m_readCount = 0u;
            uint64_t m_readBytes = 0u;
            uint64_t m_rejectedTransfers = 0u;
            uint64_t m_rejectedReads = 0u;
        };
    }

    std::unique_ptr<IopService> createStashManagerService(IopHost &host,
                                                          StashManagerBindings bindings)
    {
        const uint64_t arenaEnd = static_cast<uint64_t>(bindings.arenaBase) +
                                  static_cast<uint64_t>(bindings.arenaSize);
        if (bindings.serviceName.empty() || bindings.sid == 0u ||
            bindings.arenaBase == 0u || bindings.arenaSize == 0u ||
            arenaEnd > kIopRamSize ||
            bindings.readFunction == bindings.initializeFunction ||
            bindings.initializeResponseBytes < 2u * sizeof(uint32_t))
        {
            throw std::invalid_argument("invalid stash-manager bindings");
        }
        return std::make_unique<StashManagerService>(host, std::move(bindings));
    }
}
