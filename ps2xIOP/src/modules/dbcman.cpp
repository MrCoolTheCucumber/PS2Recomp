#include "module_factories.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kDbcManV2Sid = 0x80000900u;
        constexpr uint32_t kDbcManV3Sid = 0x80001300u;
        constexpr uint32_t kRpcV2Connect = 0x80000901u;
        constexpr uint32_t kRpcV2SetDmaAddress = 0x80000904u;
        constexpr uint32_t kRpcV2CheckVersion = 0x80000963u;
        constexpr uint32_t kRpcV3CheckVersion = 0x80001363u;
        constexpr uint32_t kDbcManV2Version = 0x0202u;
        constexpr uint32_t kDbcManV3Version = 0x0320u;
        constexpr uint32_t kDbcManSuccess = 1u;
        constexpr uint32_t kV2ConnectResultOffset = 0x24u;
        constexpr uint32_t kMaxUnknownRpcLogs = 32u;

        class DbcmanService final : public IopService
        {
        public:
            explicit DbcmanService(IopHost &host)
                : m_host(host)
            {
            }

            [[nodiscard]] std::string_view name() const override
            {
                return "dbcman";
            }

            [[nodiscard]] std::span<const uint32_t> sids() const override
            {
                return kSids;
            }

            void reset() override
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_unknownRpcLogCount = 0u;
                m_nextSocket = 0u;
            }

            [[nodiscard]] RpcResult handleRpc(const RpcRequest &request) override
            {
                RpcResult result;
                if (request.sid != kDbcManV2Sid && request.sid != kDbcManV3Sid)
                {
                    return result;
                }

                result.handled = true;
                result.resultAddress = request.receive.address;
                if (request.receive.address == 0u || request.receive.size == 0u)
                {
                    return result;
                }

                if (request.sid == kDbcManV3Sid && request.function == kRpcV3CheckVersion)
                {
                    const uint32_t wordCount = request.receive.size / sizeof(uint32_t);
                    const uint32_t count = wordCount < 4u ? wordCount : 4u;
                    for (uint32_t index = 0u; index < count; ++index)
                    {
                        const uint32_t address = request.receive.address + index * sizeof(uint32_t);
                        (void)m_host.writeGuest(address, &kDbcManV3Version, sizeof(kDbcManV3Version));
                    }
                    return result;
                }

                if (request.sid == kDbcManV2Sid)
                {
                    if (request.function == kRpcV2CheckVersion)
                    {
                        (void)m_host.writeGuest(request.receive.address,
                                                &kDbcManV2Version,
                                                sizeof(kDbcManV2Version));
                        return result;
                    }
                    if (request.function == kRpcV2SetDmaAddress)
                    {
                        (void)m_host.writeGuest(request.receive.address,
                                                &kDbcManSuccess,
                                                sizeof(kDbcManSuccess));
                        return result;
                    }
                    if (request.function == kRpcV2Connect &&
                        request.receive.size >= kV2ConnectResultOffset + sizeof(uint32_t))
                    {
                        uint32_t socket = 0u;
                        {
                            std::lock_guard<std::mutex> lock(m_mutex);
                            socket = m_nextSocket++;
                        }
                        (void)m_host.writeGuest(request.receive.address + kV2ConnectResultOffset,
                                                &socket,
                                                sizeof(socket));
                        return result;
                    }
                }

                bool shouldLog = false;
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    if (m_unknownRpcLogCount < kMaxUnknownRpcLogs)
                    {
                        ++m_unknownRpcLogCount;
                        shouldLog = true;
                    }
                }

                if (shouldLog)
                {
                    std::ostringstream message;
                    message << "[DBCMAN:stub]"
                            << " sid=0x" << std::hex << request.sid
                            << " rpc=0x" << request.function
                            << " send=0x" << request.send.address
                            << " sendSize=0x" << request.send.size
                            << " recv=0x" << request.receive.address
                            << " recvSize=0x" << request.receive.size;
                    m_host.log(LogLevel::Info, message.str());
                }
                return result;
            }

            void appendDebugMetrics(std::vector<DebugMetric> &metrics) const override
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                metrics.push_back({"unknown_rpc_logs", m_unknownRpcLogCount, false});
            }

        private:
            inline static constexpr std::array<uint32_t, 2> kSids{kDbcManV2Sid, kDbcManV3Sid};

            IopHost &m_host;
            mutable std::mutex m_mutex;
            uint32_t m_unknownRpcLogCount = 0u;
            uint32_t m_nextSocket = 0u;
        };
    }

    std::unique_ptr<IopService> createDbcmanService(IopHost &host)
    {
        return std::make_unique<DbcmanService>(host);
    }
}
