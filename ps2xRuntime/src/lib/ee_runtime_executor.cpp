#include "runtime/ee_runtime_executor.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace ps2x::ee
{
    namespace
    {
        class NullEeSchedulerExecutorHooks final
            : public IEeSchedulerExecutorHooks
        {
        public:
            void commitPriorContext(
                std::optional<int>,
                ps2x::timing::EeTickDelta,
                ps2x::timing::EeTick) override
            {
            }

            void publishSelectedContext(
                std::optional<int>,
                ps2x::timing::EeTick) override
            {
            }

            void publishWaitCompletion(
                int,
                EeSchedulerCompletedWait,
                ps2x::timing::EeTick) override
            {
            }

            [[nodiscard]] bool
            hasImmediateConsequence(
                EeSchedulerConsequenceStage,
                ps2x::timing::EeTick) const override
            {
                return false;
            }

            [[nodiscard]]
            EeSchedulerReschedulePolicy
            applyNextConsequence(
                EeSchedulerConsequenceStage,
                ps2x::timing::EeTick,
                EeThreadScheduler &) override
            {
                throw std::logic_error(
                    "null EE executor hooks have no "
                    "consequences");
            }
        };
    }

    class EeRuntimeExecutor::Impl final
        : public IEeSchedulerExecutorHooks
    {
    public:
        Impl(
            IEeExecutionBackend &backend,
            IEeSchedulerExecutorHooks *hooks,
            Options options)
            : m_backend(backend),
              m_hooks(hooks ? hooks : &m_nullHooks),
              m_options(options)
        {
            if (m_options.tickBudget == 0u)
            {
                throw std::invalid_argument(
                    "EE runtime executor tick budget must "
                    "be nonzero");
            }
            if (!m_backend.executorResumable())
            {
                throw std::invalid_argument(
                    std::string(m_backend.name()) +
                    " is not an executor-resumable EE "
                    "backend");
            }
        }

        ~Impl() noexcept
        {
            requestStop();
            try
            {
                join();
            }
            catch (...)
            {
                std::terminate();
            }
        }

        void start(Command setup)
        {
            if (!setup)
            {
                throw std::invalid_argument(
                    "EE runtime executor requires setup");
            }

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_thread.joinable() ||
                    m_starting ||
                    m_running)
                {
                    throw std::logic_error(
                        "EE runtime executor is already "
                        "started or has not been joined");
                }

                m_starting = true;
                m_startComplete = false;
                m_acceptingPublications = false;
                m_stopRequested = false;
                m_failure = {};
                m_startFailure = {};
                m_executorThreadId = {};
                m_pausedAtBoundary = false;
                m_publications.clear();
                m_delayedPublications.clear();
                m_nextPublicationSequence = 0u;
                ++m_statistics.starts;
                m_statistics.modeledTick = {};
                m_statistics.boundary = {};
                ++m_wakeGeneration;
            }

            try
            {
                m_thread = std::thread(
                    [this, setup = std::move(setup)]() mutable
                    {
                        run(std::move(setup));
                    });
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_starting = false;
                m_startComplete = true;
                throw;
            }

            std::exception_ptr startFailure;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv.wait(
                    lock,
                    [this]()
                    {
                        return m_startComplete;
                    });
                startFailure = m_startFailure;
            }
            if (startFailure)
            {
                join();
                std::rethrow_exception(startFailure);
            }
        }

        [[nodiscard]] bool publish(
            Command command,
            EeSchedulerReschedulePolicy policy)
        {
            if (!command)
            {
                return false;
            }

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!m_acceptingPublications ||
                    m_stopRequested ||
                    m_failure)
                {
                    return false;
                }

                promoteReadyDelayedLocked(
                    std::chrono::steady_clock::now());
                m_publications.push_back(
                    Publication{
                        std::move(command),
                        {},
                        policy});
                ++m_statistics.publicationsQueued;
                m_statistics.peakQueuedPublications =
                    std::max(
                        m_statistics
                            .peakQueuedPublications,
                        queuedPublicationCountLocked());
                ++m_wakeGeneration;
            }
            m_cv.notify_all();
            return true;
        }

        [[nodiscard]] bool publishAt(
            std::chrono::steady_clock::time_point deadline,
            Command command,
            EeSchedulerReschedulePolicy policy)
        {
            if (!command)
            {
                return false;
            }

            const auto now =
                std::chrono::steady_clock::now();
            if (deadline <= now)
            {
                return publish(
                    std::move(command), policy);
            }

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!m_acceptingPublications ||
                    m_stopRequested ||
                    m_failure)
                {
                    return false;
                }

                DelayedPublication delayed{
                    deadline,
                    m_nextPublicationSequence++,
                    Publication{
                        std::move(command),
                        {},
                        policy,
                    },
                };
                const auto position =
                    std::upper_bound(
                        m_delayedPublications.begin(),
                        m_delayedPublications.end(),
                        delayed,
                        [](const DelayedPublication &left,
                           const DelayedPublication &right)
                        {
                            if (left.deadline !=
                                right.deadline)
                            {
                                return left.deadline <
                                       right.deadline;
                            }
                            return left.sequence <
                                   right.sequence;
                        });
                m_delayedPublications.insert(
                    position, std::move(delayed));
                ++m_statistics.publicationsQueued;
                m_statistics.peakQueuedPublications =
                    std::max(
                        m_statistics
                            .peakQueuedPublications,
                        queuedPublicationCountLocked());
                ++m_wakeGeneration;
            }
            m_cv.notify_all();
            return true;
        }

        void invokeAtBoundary(
            Command command,
            EeSchedulerReschedulePolicy policy)
        {
            if (!command)
            {
                throw std::invalid_argument(
                    "EE executor boundary command is "
                    "empty");
            }
            if (isExecutorThread())
            {
                throw std::logic_error(
                    "an EE guest/executor caller cannot "
                    "wait on its own boundary");
            }

            auto completion =
                std::make_shared<std::promise<void>>();
            std::future<void> completed =
                completion->get_future();
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!m_acceptingPublications ||
                    m_stopRequested ||
                    m_failure)
                {
                    throw std::runtime_error(
                        "EE runtime executor is not "
                        "accepting boundary commands");
                }

                promoteReadyDelayedLocked(
                    std::chrono::steady_clock::now());
                m_publications.push_back(
                    Publication{
                        std::move(command),
                        completion,
                        policy});
                ++m_statistics.publicationsQueued;
                m_statistics.peakQueuedPublications =
                    std::max(
                        m_statistics
                            .peakQueuedPublications,
                        queuedPublicationCountLocked());
                ++m_wakeGeneration;
            }
            m_cv.notify_all();
            completed.get();
        }

        void notify() noexcept
        {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                ++m_wakeGeneration;
            }
            m_cv.notify_all();
        }

        void debugRequestPause() noexcept
        {
            m_boundaryExecutor.debugRequestPause();
            notify();
        }

        [[nodiscard]] bool debugWaitUntilPaused(
            std::chrono::milliseconds timeout)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            (void)m_cv.wait_for(
                lock,
                timeout,
                [this]()
                {
                    return m_pausedAtBoundary ||
                           m_stopRequested ||
                           m_failure ||
                           (!m_running &&
                            !m_starting);
                });
            return m_pausedAtBoundary;
        }

        void debugResume() noexcept
        {
            m_boundaryExecutor.debugResume();
            notify();
        }

        [[nodiscard]] bool debugStepBoundaries(
            uint64_t count) noexcept
        {
            const bool accepted =
                m_boundaryExecutor.debugStepBoundaries(
                    count);
            if (accepted)
            {
                notify();
            }
            return accepted;
        }

        [[nodiscard]] bool debugPaused() const noexcept
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_pausedAtBoundary;
        }

        void requestStop() noexcept
        {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_stopRequested = true;
                m_acceptingPublications = false;
                ++m_wakeGeneration;
            }
            m_boundaryExecutor.debugRequestStop();
            m_cv.notify_all();
        }

        void join()
        {
            std::thread thread;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!m_thread.joinable())
                {
                    return;
                }
                if (m_thread.get_id() ==
                    std::this_thread::get_id())
                {
                    throw std::logic_error(
                        "EE runtime executor cannot join "
                        "itself");
                }
                thread = std::move(m_thread);
            }
            thread.join();
        }

        [[nodiscard]] bool running() const noexcept
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_running;
        }

        [[nodiscard]] bool
        acceptingPublications() const noexcept
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_acceptingPublications;
        }

        [[nodiscard]] std::thread::id
        executorThreadId() const noexcept
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_executorThreadId;
        }

        [[nodiscard]] EeRuntimeExecutorStatistics
        statistics() const noexcept
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_statistics;
        }

        [[nodiscard]] std::exception_ptr
        failure() const noexcept
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_failure;
        }

        void rethrowFailure() const
        {
            const std::exception_ptr current = failure();
            if (current)
            {
                std::rethrow_exception(current);
            }
        }

        void commitPriorContext(
            std::optional<int> priorThreadId,
            ps2x::timing::EeTickDelta elapsed,
            ps2x::timing::EeTick now) override
        {
            m_hooks->commitPriorContext(
                priorThreadId, elapsed, now);
        }

        void publishSelectedContext(
            std::optional<int> selectedThreadId,
            ps2x::timing::EeTick now) override
        {
            m_hooks->publishSelectedContext(
                selectedThreadId, now);
        }

        void publishWaitCompletion(
            int threadId,
            EeSchedulerCompletedWait completion,
            ps2x::timing::EeTick now) override
        {
            m_hooks->publishWaitCompletion(
                threadId, completion, now);
        }

        [[nodiscard]] bool
        hasImmediateConsequence(
            EeSchedulerConsequenceStage stage,
            ps2x::timing::EeTick now) const override
        {
            if (stage ==
                    EeSchedulerConsequenceStage::
                        AsynchronousWake &&
                hasQueuedPublication())
            {
                return true;
            }
            return m_hooks->hasImmediateConsequence(
                stage, now);
        }

        [[nodiscard]]
        EeSchedulerReschedulePolicy
        applyNextConsequence(
            EeSchedulerConsequenceStage stage,
            ps2x::timing::EeTick now,
            EeThreadScheduler &scheduler) override
        {
            if (stage ==
                    EeSchedulerConsequenceStage::
                        AsynchronousWake)
            {
                const auto publicationPolicy =
                    applyQueuedPublication();
                if (publicationPolicy.has_value())
                {
                    return *publicationPolicy;
                }
            }
            if (stage ==
                    EeSchedulerConsequenceStage::
                        AsynchronousWake &&
                stopRequested())
            {
                return EeSchedulerReschedulePolicy::None;
            }
            return m_hooks->applyNextConsequence(
                stage, now, scheduler);
        }

    private:
        struct Publication
        {
            Command command;
            std::shared_ptr<std::promise<void>>
                completion;
            EeSchedulerReschedulePolicy policy =
                EeSchedulerReschedulePolicy::
                    HigherPriorityOnly;
        };

        struct DelayedPublication
        {
            std::chrono::steady_clock::time_point deadline;
            uint64_t sequence = 0u;
            Publication publication;
        };

        [[nodiscard]] size_t
        queuedPublicationCountLocked() const noexcept
        {
            return m_publications.size() +
                   m_delayedPublications.size();
        }

        void promoteReadyDelayedLocked(
            std::chrono::steady_clock::time_point now)
        {
            while (!m_delayedPublications.empty() &&
                   m_delayedPublications.front().deadline <=
                       now)
            {
                m_publications.push_back(
                    std::move(
                        m_delayedPublications.front()
                            .publication));
                m_delayedPublications.pop_front();
            }
        }

        [[nodiscard]] bool isExecutorThread() const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_executorThreadId !=
                       std::thread::id{} &&
                   m_executorThreadId ==
                       std::this_thread::get_id();
        }

        [[nodiscard]] bool
        hasQueuedPublication() const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return !m_stopRequested &&
                   (!m_publications.empty() ||
                    (!m_delayedPublications.empty() &&
                     m_delayedPublications.front()
                             .deadline <=
                         std::chrono::steady_clock::now()));
        }

        [[nodiscard]]
        EeSchedulerReschedulePolicy
        initialBoundaryPolicy() const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_stopRequested &&
                (!m_publications.empty() ||
                 (!m_delayedPublications.empty() &&
                  m_delayedPublications.front().deadline <=
                      std::chrono::steady_clock::now())))
            {
                // A queued syscall consequence is part of the operation
                // which reached this boundary. Apply it before considering
                // any alternate READY thread, then use the publication's
                // policy exactly once after the command. This preserves raw
                // atomicity and prevents an equal-priority ordinary rotation
                // from selecting a peer both before and after rotating the
                // named queue.
                return EeSchedulerReschedulePolicy::None;
            }
            return EeSchedulerReschedulePolicy::
                HigherPriorityOnly;
        }

        [[nodiscard]] std::optional<
            EeSchedulerReschedulePolicy>
        applyQueuedPublication()
        {
            Publication publication;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                promoteReadyDelayedLocked(
                    std::chrono::steady_clock::now());
                if (m_stopRequested ||
                    m_publications.empty())
                {
                    return std::nullopt;
                }
                publication =
                    std::move(m_publications.front());
                m_publications.pop_front();
            }

            try
            {
                publication.command(
                    m_scheduler, m_backend);
                {
                    std::lock_guard<std::mutex> lock(
                        m_mutex);
                    ++m_statistics.publicationsApplied;
                }
                if (publication.completion)
                {
                    publication.completion->set_value();
                }
            }
            catch (...)
            {
                if (publication.completion)
                {
                    publication.completion
                        ->set_exception(
                            std::current_exception());
                }
                throw;
            }
            return publication.policy;
        }

        void cacheBoundary(
            const EeSchedulerBoundaryResult &result)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_statistics.modeledTick = result.tick;
            m_statistics.boundary =
                m_boundaryExecutor.statistics();
        }

        void recordDispatch(
            bool continuationDestroyed)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            ++m_statistics.dispatches;
            if (continuationDestroyed)
            {
                ++m_statistics
                      .continuationsDestroyed;
            }
        }

        [[nodiscard]] bool stopRequested() const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_stopRequested;
        }

        void waitForWork(uint64_t observedGeneration)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            ++m_statistics.idleWaits;
            const auto wakePredicate =
                [this, observedGeneration]()
                {
                    return m_stopRequested ||
                           m_wakeGeneration !=
                               observedGeneration;
                };
            if (m_delayedPublications.empty())
            {
                m_cv.wait(lock, wakePredicate);
            }
            else
            {
                (void)m_cv.wait_until(
                    lock,
                    m_delayedPublications.front().deadline,
                    wakePredicate);
            }
        }

        [[nodiscard]] uint64_t wakeGeneration() const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_wakeGeneration;
        }

        void publishStartComplete()
        {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_starting = false;
                m_running = true;
                m_acceptingPublications =
                    !m_stopRequested;
                m_startComplete = true;
            }
            m_cv.notify_all();
        }

        void run(Command setup) noexcept
        {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_executorThreadId =
                    std::this_thread::get_id();
            }

            bool setupCompleted = false;
            try
            {
                m_scheduler = EeThreadScheduler{};
                m_boundaryExecutor.reset();
                setup(m_scheduler, m_backend);
                if (!m_scheduler.validate())
                {
                    throw std::logic_error(
                        "EE runtime executor setup "
                        "produced invalid scheduler state");
                }

                setupCompleted = true;
                publishStartComplete();

                std::optional<int> priorThreadId;
                ps2x::timing::EeTickDelta elapsed{};
                std::exception_ptr pendingGuestFailure;
                for (;;)
                {
                    const uint64_t observedWakeGeneration =
                        wakeGeneration();
                    if (stopRequested() &&
                        !priorThreadId.has_value())
                    {
                        break;
                    }

                    const EeSchedulerBoundaryResult boundary =
                        m_boundaryExecutor.processBoundary(
                            m_scheduler,
                            *this,
                            priorThreadId,
                            elapsed,
                            initialBoundaryPolicy());
                    priorThreadId.reset();
                    elapsed = {};
                    cacheBoundary(boundary);
                    {
                        std::lock_guard<std::mutex> lock(
                            m_mutex);
                        m_pausedAtBoundary =
                            boundary.disposition ==
                            EeSchedulerExecutorDisposition::
                                Paused;
                    }
                    m_cv.notify_all();

                    if (boundary.limitExceeded)
                    {
                        throw std::runtime_error(
                            std::string(
                                "EE executor consequence "
                                "limit exceeded at ") +
                            std::string(
                                eeSchedulerConsequenceStageName(
                                    *boundary
                                         .offendingStage)));
                    }
                    if (boundary.reentrantBoundaryRejected)
                    {
                        throw std::logic_error(
                            "EE executor boundary was "
                            "entered recursively");
                    }
                    if (pendingGuestFailure)
                    {
                        std::rethrow_exception(
                            pendingGuestFailure);
                    }
                    if (boundary.disposition ==
                        EeSchedulerExecutorDisposition::
                            StopRequested)
                    {
                        break;
                    }
                    if (boundary.disposition ==
                            EeSchedulerExecutorDisposition::
                                Paused ||
                        !boundary.selectedThreadId
                             .has_value())
                    {
                        waitForWork(
                            observedWakeGeneration);
                        continue;
                    }
                    if (stopRequested())
                    {
                        break;
                    }

                    const auto selectedHandle =
                        m_scheduler.threadHandle(
                            *boundary.selectedThreadId);
                    if (!selectedHandle.has_value())
                    {
                        throw std::logic_error(
                            "EE executor selected a thread "
                            "without a valid handle");
                    }
                    const auto waitCompletion =
                        m_scheduler
                            .consumeWaitCompletion(
                                *selectedHandle);
                    if (waitCompletion.has_value())
                    {
                        publishWaitCompletion(
                            selectedHandle->id,
                            *waitCompletion,
                            boundary.tick);
                    }

                    const auto dispatch =
                        m_scheduler.dispatchOne(
                            m_backend,
                            m_options.tickBudget);
                    if (!dispatch.has_value())
                    {
                        throw std::logic_error(
                            "EE executor selected a thread "
                            "but could not dispatch it");
                    }

                    priorThreadId =
                        dispatch->resumedThreadId;
                    elapsed =
                        ps2x::timing::EeTickDelta::fromRaw(
                            dispatch->result.elapsedTicks);

                    const bool terminal =
                        dispatch->result.reason ==
                            EeSchedulerExitReason::
                                Finished ||
                        dispatch->result.reason ==
                            EeSchedulerExitReason::
                                Exception;
                    if (terminal)
                    {
                        m_backend.destroy(
                            dispatch->resumedThreadId);
                    }
                    recordDispatch(terminal);

                    if (dispatch->result.reason ==
                        EeSchedulerExitReason::Exception)
                    {
                        pendingGuestFailure =
                            dispatch->result.failure
                                ? dispatch->result.failure
                                : std::make_exception_ptr(
                                      std::runtime_error(
                                          "EE guest exception "
                                          "exit had no "
                                          "exception payload"));
                    }
                    else if (
                        dispatch->result.reason ==
                        EeSchedulerExitReason::
                            StopRequested)
                    {
                        requestStop();
                    }
                }
            }
            catch (...)
            {
                const std::exception_ptr current =
                    std::current_exception();
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!setupCompleted)
                {
                    m_startFailure = current;
                }
                if (!m_failure)
                {
                    m_failure = current;
                    ++m_statistics.failures;
                }
                m_stopRequested = true;
                m_acceptingPublications = false;
            }

            try
            {
                m_backend.joinAll();
            }
            catch (...)
            {
                const std::exception_ptr current =
                    std::current_exception();
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!setupCompleted &&
                    !m_startFailure)
                {
                    m_startFailure = current;
                }
                if (!m_failure)
                {
                    m_failure = current;
                    ++m_statistics.failures;
                }
            }

            m_scheduler = EeThreadScheduler{};
            failPendingPublications();
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_starting = false;
                m_running = false;
                m_acceptingPublications = false;
                m_startComplete = true;
                m_executorThreadId = {};
                m_pausedAtBoundary = false;
            }
            m_cv.notify_all();
        }

        void failPendingPublications() noexcept
        {
            std::deque<Publication> pending;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                pending.swap(m_publications);
                for (DelayedPublication &delayed :
                     m_delayedPublications)
                {
                    pending.push_back(
                        std::move(delayed.publication));
                }
                m_delayedPublications.clear();
            }

            const auto failure = std::make_exception_ptr(
                std::runtime_error(
                    "EE runtime executor stopped before "
                    "applying a boundary command"));
            for (Publication &publication : pending)
            {
                if (!publication.completion)
                {
                    continue;
                }
                try
                {
                    publication.completion->set_exception(
                        failure);
                }
                catch (...)
                {
                }
            }
        }

        IEeExecutionBackend &m_backend;
        NullEeSchedulerExecutorHooks m_nullHooks;
        IEeSchedulerExecutorHooks *m_hooks = nullptr;
        Options m_options{};

        mutable std::mutex m_mutex;
        std::condition_variable m_cv;
        std::thread m_thread;
        bool m_starting = false;
        bool m_startComplete = false;
        bool m_running = false;
        bool m_acceptingPublications = false;
        bool m_stopRequested = false;
        bool m_pausedAtBoundary = false;
        uint64_t m_wakeGeneration = 0u;
        std::thread::id m_executorThreadId;
        std::deque<Publication> m_publications;
        std::deque<DelayedPublication>
            m_delayedPublications;
        uint64_t m_nextPublicationSequence = 0u;
        std::exception_ptr m_failure;
        std::exception_ptr m_startFailure;
        EeRuntimeExecutorStatistics m_statistics{};

        EeThreadScheduler m_scheduler;
        EeSchedulerExecutor m_boundaryExecutor;
    };

    EeRuntimeExecutor::EeRuntimeExecutor(
        IEeExecutionBackend &backend,
        IEeSchedulerExecutorHooks *hooks)
        : EeRuntimeExecutor(
              backend, hooks, Options{})
    {
    }

    EeRuntimeExecutor::EeRuntimeExecutor(
        IEeExecutionBackend &backend,
        IEeSchedulerExecutorHooks *hooks,
        Options options)
        : m_impl(std::make_unique<Impl>(
              backend, hooks, options))
    {
    }

    EeRuntimeExecutor::~EeRuntimeExecutor() noexcept =
        default;

    void EeRuntimeExecutor::start(Command setup)
    {
        m_impl->start(std::move(setup));
    }

    bool EeRuntimeExecutor::publish(
        Command command,
        EeSchedulerReschedulePolicy policy)
    {
        return m_impl->publish(
            std::move(command), policy);
    }

    bool EeRuntimeExecutor::publishAt(
        std::chrono::steady_clock::time_point deadline,
        Command command,
        EeSchedulerReschedulePolicy policy)
    {
        return m_impl->publishAt(
            deadline, std::move(command), policy);
    }

    void EeRuntimeExecutor::invokeAtBoundary(
        Command command,
        EeSchedulerReschedulePolicy policy)
    {
        m_impl->invokeAtBoundary(
            std::move(command), policy);
    }

    void EeRuntimeExecutor::notify() noexcept
    {
        m_impl->notify();
    }

    void EeRuntimeExecutor::debugRequestPause() noexcept
    {
        m_impl->debugRequestPause();
    }

    bool EeRuntimeExecutor::debugWaitUntilPaused(
        std::chrono::milliseconds timeout)
    {
        return m_impl->debugWaitUntilPaused(timeout);
    }

    void EeRuntimeExecutor::debugResume() noexcept
    {
        m_impl->debugResume();
    }

    bool EeRuntimeExecutor::debugStepBoundaries(
        uint64_t count) noexcept
    {
        return m_impl->debugStepBoundaries(count);
    }

    bool EeRuntimeExecutor::debugPaused() const noexcept
    {
        return m_impl->debugPaused();
    }

    void EeRuntimeExecutor::requestStop() noexcept
    {
        m_impl->requestStop();
    }

    void EeRuntimeExecutor::join()
    {
        m_impl->join();
    }

    bool EeRuntimeExecutor::running() const noexcept
    {
        return m_impl->running();
    }

    bool EeRuntimeExecutor::
        acceptingPublications() const noexcept
    {
        return m_impl->acceptingPublications();
    }

    std::thread::id
    EeRuntimeExecutor::executorThreadId() const noexcept
    {
        return m_impl->executorThreadId();
    }

    EeRuntimeExecutorStatistics
    EeRuntimeExecutor::statistics() const noexcept
    {
        return m_impl->statistics();
    }

    std::exception_ptr
    EeRuntimeExecutor::failure() const noexcept
    {
        return m_impl->failure();
    }

    void EeRuntimeExecutor::rethrowFailure() const
    {
        m_impl->rethrowFailure();
    }
}
