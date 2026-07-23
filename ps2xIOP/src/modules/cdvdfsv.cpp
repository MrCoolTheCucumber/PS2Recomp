#include "module_factories.h"

#include <array>
#include <cstdint>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kCdDiskReadySid = 0x8000059Au;
        constexpr uint32_t kCdDiskReadyRpc = 0u;
        constexpr uint32_t kCdComplete = 2u;

        class CdvdfsvService final : public IopService
        {
        public:
            explicit CdvdfsvService(IopHost &host)
                : m_host(host)
            {
            }

            [[nodiscard]] std::string_view name() const override
            {
                return "cdvdfsv";
            }

            [[nodiscard]] std::span<const uint32_t> sids() const override
            {
                return kSids;
            }

            void reset() override
            {
            }

            [[nodiscard]] RpcResult handleRpc(const RpcRequest &request) override
            {
                RpcResult result;
                if (request.sid != kCdDiskReadySid || request.function != kCdDiskReadyRpc)
                {
                    return result;
                }

                result.handled = true;
                result.resultAddress = request.receive.address;
                if (request.receive.address != 0u && request.receive.size >= sizeof(kCdComplete))
                {
                    (void)m_host.writeGuest(request.receive.address, &kCdComplete, sizeof(kCdComplete));
                }
                return result;
            }

        private:
            inline static constexpr std::array<uint32_t, 1> kSids{kCdDiskReadySid};

            IopHost &m_host;
        };
    }

    std::unique_ptr<IopService> createCdvdfsvService(IopHost &host)
    {
        return std::make_unique<CdvdfsvService>(host);
    }
}
