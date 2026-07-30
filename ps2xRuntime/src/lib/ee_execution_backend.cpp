#include "runtime/ee_execution_backend.h"

#ifndef PS2X_HAS_EE_CPP_FIBER_BACKEND
#define PS2X_HAS_EE_CPP_FIBER_BACKEND 0
#endif

#if PS2X_HAS_EE_CPP_FIBER_BACKEND
#include <boost/version.hpp>

#if BOOST_VERSION != 109100
#error "The EE fiber backend requires exact Boost 1.91.0"
#endif
#if defined(BOOST_USE_UCONTEXT)
#error "Production EE fibers must not select Boost.Context ucontext"
#endif
#if defined(BOOST_USE_WINFIB)
#error "Production EE fibers must not select Boost.Context WinFiber"
#endif
#endif

#include <atomic>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
    class LegacyHostThreadEeExecutionBackend final
        : public IEeExecutionBackend
    {
    public:
        ~LegacyHostThreadEeExecutionBackend() override
        {
            detachAll();
        }

        [[nodiscard]] EeExecutionBackendKind
        kind() const noexcept override
        {
            return EeExecutionBackendKind::LegacyHostThread;
        }

        [[nodiscard]] std::string_view
        name() const noexcept override
        {
            return "legacy-host-thread";
        }

        void create(
            int threadId,
            ThreadEntry entry) override
        {
            auto finished =
                std::make_shared<std::atomic<bool>>(false);
            std::thread worker(
                [entry = std::move(entry), finished]() mutable
                {
                    struct Completion
                    {
                        std::atomic<bool> &finished;

                        ~Completion()
                        {
                            finished.store(
                                true,
                                std::memory_order_release);
                        }
                    } completion{*finished};
                    entry();
                });
            ThreadRecord stale;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = m_threads.find(threadId);
                if (it != m_threads.end())
                {
                    stale = std::move(it->second);
                    m_threads.erase(it);
                }
                m_threads.emplace(
                    threadId,
                    ThreadRecord{
                        std::move(worker),
                        std::move(finished)});
            }

            joinOrDetachSelf(stale.worker);
        }

        [[nodiscard]] bool
        isFinished(int threadId) const override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            const auto it = m_threads.find(threadId);
            return it == m_threads.end() ||
                   it->second.finished->load(
                       std::memory_order_acquire);
        }

        void destroy(int threadId) override
        {
            ThreadRecord record;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = m_threads.find(threadId);
                if (it != m_threads.end())
                {
                    record = std::move(it->second);
                    m_threads.erase(it);
                }
            }

            joinOrDetachSelf(record.worker);
        }

        void detach(int threadId) override
        {
            ThreadRecord record;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = m_threads.find(threadId);
                if (it != m_threads.end())
                {
                    record = std::move(it->second);
                    m_threads.erase(it);
                }
            }
            if (record.worker.joinable())
            {
                record.worker.detach();
            }
        }

        void joinAll() override
        {
            std::vector<std::thread> workers;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                workers.reserve(m_threads.size());
                const std::thread::id selfId =
                    std::this_thread::get_id();
                for (auto it = m_threads.begin();
                     it != m_threads.end();)
                {
                    std::thread &worker =
                        it->second.worker;
                    if (worker.joinable() &&
                        worker.get_id() == selfId)
                    {
                        ++it;
                        continue;
                    }

                    workers.push_back(
                        std::move(worker));
                    it = m_threads.erase(it);
                }
            }

            for (std::thread &worker : workers)
            {
                if (worker.joinable())
                {
                    worker.join();
                }
            }
        }

        void detachAll() override
        {
            std::vector<std::thread> workers;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                workers.reserve(m_threads.size());
                for (auto &entry : m_threads)
                {
                    workers.push_back(
                        std::move(entry.second.worker));
                }
                m_threads.clear();
            }

            for (std::thread &worker : workers)
            {
                if (worker.joinable())
                {
                    worker.detach();
                }
            }
        }

        [[nodiscard]] size_t
        managedThreadCount() const override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_threads.size();
        }

    private:
        static void joinOrDetachSelf(std::thread &worker)
        {
            if (!worker.joinable())
            {
                return;
            }
            if (worker.get_id() ==
                std::this_thread::get_id())
            {
                worker.detach();
            }
            else
            {
                worker.join();
            }
        }

        struct ThreadRecord
        {
            std::thread worker;
            std::shared_ptr<std::atomic<bool>> finished;
        };

        mutable std::mutex m_mutex;
        std::unordered_map<int, ThreadRecord> m_threads;
    };
}

std::unique_ptr<IEeExecutionBackend>
createEeExecutionBackend(EeExecutionBackendKind kind)
{
    switch (kind)
    {
    case EeExecutionBackendKind::LegacyHostThread:
        return std::make_unique<
            LegacyHostThreadEeExecutionBackend>();
    }

    throw std::invalid_argument(
        "unsupported EE execution backend");
}

EeExecutionBackendBuildInfo
eeExecutionBackendBuildInfo() noexcept
{
#if PS2X_HAS_EE_CPP_FIBER_BACKEND
    return {
        true,
        PS2X_EE_BOOST_VERSION,
        PS2X_EE_BOOST_CONTEXT_ARCHITECTURE,
        PS2X_EE_BOOST_CONTEXT_BINARY_FORMAT,
        PS2X_EE_BOOST_CONTEXT_ABI,
        PS2X_EE_BOOST_CONTEXT_IMPLEMENTATION};
#else
    return {};
#endif
}

std::string eeExecutionBackendDiagnostics(
    EeExecutionBackendKind selected)
{
    const std::string_view selectedName =
        selected ==
                EeExecutionBackendKind::LegacyHostThread
            ? std::string_view{"legacy-host-thread"}
            : std::string_view{"unknown"};
    const EeExecutionBackendBuildInfo build =
        eeExecutionBackendBuildInfo();

    std::ostringstream out;
    out << "selected=" << selectedName
        << ", Boost.Context="
        << (build.boostContextFcontextAvailable
                ? "available"
                : "unavailable")
        << ", Boost=" << build.boostVersion
        << ", architecture=" << build.architecture
        << ", binary-format=" << build.binaryFormat
        << ", ABI=" << build.abi
        << ", context-implementation="
        << build.contextImplementation;
    return out.str();
}
