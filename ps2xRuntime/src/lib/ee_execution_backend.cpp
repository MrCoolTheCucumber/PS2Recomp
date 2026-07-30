#include "runtime/ee_execution_backend.h"
#include "runtime/boost_ee_fiber.h"

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
#include <exception>
#include <mutex>
#include <optional>
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

        [[nodiscard]] bool
        executorResumable() const noexcept override
        {
            return false;
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

        ps2x::ee::EeSchedulerRunResult resume(
            int threadId,
            uint64_t tickBudget) override
        {
            static_cast<void>(threadId);
            static_cast<void>(tickBudget);
            throw std::logic_error(
                "legacy-host-thread continuations are "
                "not executor-resumable");
        }

        void yieldCurrent(
            ps2x::ee::EeSchedulerRunResult result) override
        {
            static_cast<void>(result);
            throw std::logic_error(
                "legacy-host-thread continuations cannot "
                "yield through the executor backend");
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

    class LegacyCppFiberEeExecutionBackend final
        : public IEeExecutionBackend
    {
    public:
        ~LegacyCppFiberEeExecutionBackend() override
        {
            if (m_owner.has_value() &&
                *m_owner != std::this_thread::get_id())
            {
                if (managedThreadCount() != 0u)
                {
                    std::terminate();
                }
                return;
            }
            try
            {
                destroyAll();
            }
            catch (...)
            {
                std::terminate();
            }
        }

        [[nodiscard]] EeExecutionBackendKind
        kind() const noexcept override
        {
            return EeExecutionBackendKind::LegacyCppFiber;
        }

        [[nodiscard]] std::string_view
        name() const noexcept override
        {
            return "legacy-cpp-fiber";
        }

        [[nodiscard]] bool
        executorResumable() const noexcept override
        {
            return true;
        }

        void create(
            int threadId,
            ThreadEntry entry) override
        {
            requireOwner("create");
            if (threadId <= 0)
            {
                throw std::invalid_argument(
                    "EE fiber thread ID must be positive");
            }

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_threads.contains(threadId))
                {
                    throw std::logic_error(
                        "EE fiber thread ID is already "
                        "managed");
                }
            }

            auto record = std::make_shared<ThreadRecord>();
            record->fiber =
                std::make_unique<BoostEeFiber>(
                    std::move(entry));
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                const bool inserted =
                    m_threads.emplace(
                        threadId, record)
                        .second;
                if (!inserted)
                {
                    throw std::logic_error(
                        "EE fiber thread ID raced with "
                        "another creation");
                }
            }
        }

        ps2x::ee::EeSchedulerRunResult resume(
            int threadId,
            uint64_t tickBudget) override
        {
            static_cast<void>(tickBudget);
            requireOwner("resume");
            if (m_runningRecord)
            {
                throw std::logic_error(
                    "EE fiber backend forbids direct "
                    "guest-to-guest resume");
            }

            const std::shared_ptr<ThreadRecord> record =
                lookup(threadId);
            if (!record)
            {
                throw std::out_of_range(
                    "EE fiber thread ID is not managed");
            }
            if (record->terminal.load(
                    std::memory_order_acquire) ||
                !record->fiber->resumable())
            {
                throw std::logic_error(
                    "EE fiber thread is not resumable");
            }

            record->yielded.reset();
            m_runningRecord = record;
            try
            {
                record->fiber->resume();
            }
            catch (...)
            {
                clearRunning();
                throw;
            }
            clearRunning();

            switch (record->fiber->state())
            {
            case BoostEeFiber::State::Suspended:
            {
                if (!record->yielded.has_value())
                {
                    throw std::logic_error(
                        "EE fiber suspended without an "
                        "executor exit result");
                }
                ps2x::ee::EeSchedulerRunResult result =
                    std::move(*record->yielded);
                record->yielded.reset();
                return result;
            }
            case BoostEeFiber::State::Finished:
                record->terminal.store(
                    true, std::memory_order_release);
                return {
                    ps2x::ee::EeSchedulerExitReason::
                        Finished,
                    0u,
                    {},
                    {}};
            case BoostEeFiber::State::Failed:
            {
                record->terminal.store(
                    true, std::memory_order_release);
                return {
                    ps2x::ee::EeSchedulerExitReason::
                        Exception,
                    0u,
                    {},
                    record->fiber->failure()};
            }
            case BoostEeFiber::State::Created:
            case BoostEeFiber::State::Running:
            case BoostEeFiber::State::Destroyed:
                break;
            }

            throw std::logic_error(
                "EE fiber returned in an invalid state");
        }

        void yieldCurrent(
            ps2x::ee::EeSchedulerRunResult result) override
        {
            requireOwner("yield");
            const std::shared_ptr<ThreadRecord> record =
                m_runningRecord;
            if (!record)
            {
                throw std::logic_error(
                    "EE fiber yield requires a running "
                    "continuation");
            }
            validateYieldResult(result);

            record->yielded = std::move(result);
            try
            {
                BoostEeFiber::yieldCurrent();
            }
            catch (...)
            {
                record->yielded.reset();
                throw;
            }
        }

        [[nodiscard]] bool
        isFinished(int threadId) const override
        {
            const std::shared_ptr<ThreadRecord> record =
                lookup(threadId);
            return !record ||
                   record->terminal.load(
                       std::memory_order_acquire);
        }

        void destroy(int threadId) override
        {
            requireOwner("destroy");
            requireExecutorStack("destroy");

            std::shared_ptr<ThreadRecord> record;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                const auto it = m_threads.find(threadId);
                if (it == m_threads.end())
                {
                    return;
                }
                record = std::move(it->second);
                m_threads.erase(it);
            }
            record->fiber->destroy();
            record->terminal.store(
                true, std::memory_order_release);
        }

        void detach(int threadId) override
        {
            // A stackful continuation cannot outlive its executor. Preserve
            // the lifecycle API for the transitional host backend, but make
            // fiber "detach" deterministic destruction.
            destroy(threadId);
        }

        void joinAll() override
        {
            requireOwner("join all");
            requireExecutorStack("join all");
            destroyAll();
            m_owner.reset();
        }

        void detachAll() override
        {
            requireOwner("detach all");
            requireExecutorStack("detach all");
            destroyAll();
            m_owner.reset();
        }

        [[nodiscard]] size_t
        managedThreadCount() const override
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_threads.size();
        }

    private:
        struct ThreadRecord
        {
            std::unique_ptr<BoostEeFiber> fiber;
            std::optional<
                ps2x::ee::EeSchedulerRunResult>
                yielded;
            std::atomic<bool> terminal{false};
        };

        void requireOwner(const char *operation)
        {
            const std::thread::id current =
                std::this_thread::get_id();
            if (!m_owner.has_value())
            {
                m_owner = current;
                return;
            }
            if (*m_owner != current)
            {
                throw std::logic_error(
                    std::string("EE fiber backend ") +
                    operation +
                    " must run on its executor thread");
            }
        }

        void requireExecutorStack(
            const char *operation) const
        {
            if (m_runningRecord)
            {
                throw std::logic_error(
                    std::string("EE fiber backend ") +
                    operation +
                    " requires the executor stack");
            }
        }

        [[nodiscard]] std::shared_ptr<ThreadRecord>
        lookup(int threadId) const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            const auto it = m_threads.find(threadId);
            return it != m_threads.end()
                       ? it->second
                       : std::shared_ptr<ThreadRecord>{};
        }

        static void validateYieldResult(
            const ps2x::ee::EeSchedulerRunResult &result)
        {
            using ps2x::ee::EeSchedulerExitReason;
            switch (result.reason)
            {
            case EeSchedulerExitReason::Blocked:
                if (!result.wait.valid())
                {
                    throw std::invalid_argument(
                        "blocked EE fiber exit requires "
                        "a wait key");
                }
                break;
            case EeSchedulerExitReason::Yielded:
            case EeSchedulerExitReason::Preempted:
            case EeSchedulerExitReason::StopRequested:
                if (result.wait.valid())
                {
                    throw std::invalid_argument(
                        "non-blocking EE fiber exit must "
                        "not carry a wait key");
                }
                break;
            case EeSchedulerExitReason::Finished:
            case EeSchedulerExitReason::Exception:
                throw std::invalid_argument(
                    "terminal EE fiber exits are owned by "
                    "the entry boundary");
            }
            if (result.failure)
            {
                throw std::invalid_argument(
                    "yielded EE fiber exits must not carry "
                    "an exception");
            }
        }

        void clearRunning() noexcept
        {
            m_runningRecord.reset();
        }

        void destroyAll()
        {
            requireExecutorStack("destroy all");
            std::vector<
                std::shared_ptr<ThreadRecord>>
                records;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                records.reserve(m_threads.size());
                for (auto &entry : m_threads)
                {
                    records.push_back(
                        std::move(entry.second));
                }
                m_threads.clear();
            }
            for (const auto &record : records)
            {
                record->fiber->destroy();
                record->terminal.store(
                    true, std::memory_order_release);
            }
        }

        mutable std::mutex m_mutex;
        std::unordered_map<
            int,
            std::shared_ptr<ThreadRecord>>
            m_threads;
        std::optional<std::thread::id> m_owner;
        std::shared_ptr<ThreadRecord> m_runningRecord;
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
    case EeExecutionBackendKind::LegacyCppFiber:
        if (!BoostEeFiber::available())
        {
            throw std::runtime_error(
                "legacy-cpp-fiber requires the pinned "
                "Boost.Context fcontext build");
        }
        return std::make_unique<
            LegacyCppFiberEeExecutionBackend>();
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
    std::string_view selectedName = "unknown";
    switch (selected)
    {
    case EeExecutionBackendKind::LegacyHostThread:
        selectedName = "legacy-host-thread";
        break;
    case EeExecutionBackendKind::LegacyCppFiber:
        selectedName = "legacy-cpp-fiber";
        break;
    }
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
