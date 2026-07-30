#include "runtime/ee_execution_backend.h"

#include <mutex>
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
            std::thread worker(std::move(entry));
            std::thread stale;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = m_threads.find(threadId);
                if (it != m_threads.end())
                {
                    stale = std::move(it->second);
                    m_threads.erase(it);
                }
                m_threads.emplace(
                    threadId, std::move(worker));
            }

            joinOrDetachSelf(stale);
        }

        void destroy(int threadId) override
        {
            std::thread worker;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = m_threads.find(threadId);
                if (it != m_threads.end())
                {
                    worker = std::move(it->second);
                    m_threads.erase(it);
                }
            }

            joinOrDetachSelf(worker);
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
                    std::thread &worker = it->second;
                    if (worker.joinable() &&
                        worker.get_id() == selfId)
                    {
                        ++it;
                        continue;
                    }

                    workers.push_back(std::move(worker));
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
                        std::move(entry.second));
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

        mutable std::mutex m_mutex;
        std::unordered_map<int, std::thread> m_threads;
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
