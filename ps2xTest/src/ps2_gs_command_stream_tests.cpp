#include "MiniTest.h"

#include "ps2_runtime.h"
#include "runtime/ps2_gs_command_stream.h"
#include "runtime/ps2_spsc_queue.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{
    using namespace std::chrono_literals;

    template <typename Predicate>
    bool waitFor(
        std::condition_variable &cv,
        std::mutex &mutex,
        Predicate predicate)
    {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(
            lock, 2s, std::move(predicate));
    }

    template <typename Predicate>
    bool waitUntil(Predicate predicate)
    {
        const auto deadline =
            std::chrono::steady_clock::now() + 2s;
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (predicate())
                return true;
            std::this_thread::sleep_for(1ms);
        }
        return predicate();
    }

    template <typename Operation>
    std::string captureException(Operation operation)
    {
        try
        {
            operation();
        }
        catch (const std::exception &error)
        {
            return error.what();
        }
        catch (...)
        {
            return "non-standard exception";
        }
        return {};
    }

    class WorkerGate
    {
    public:
        explicit WorkerGate(
            uint64_t blockedTicket = 1u,
            bool throwAfterRelease = false)
            : m_blockedTicket(blockedTicket),
              m_throwAfterRelease(throwAfterRelease)
        {
        }

        void beforeProcess(
            const GsCommand &,
            uint64_t ticket)
        {
            if (ticket != m_blockedTicket)
                return;
            std::unique_lock<std::mutex> lock(m_mutex);
            m_entered = true;
            m_cv.notify_all();
            m_cv.wait(
                lock,
                [this]()
                {
                    return m_released;
                });
            if (m_throwAfterRelease)
            {
                throw std::runtime_error(
                    "injected GS owner failure");
            }
        }

        [[nodiscard]] bool waitUntilEntered()
        {
            return waitFor(
                m_cv,
                m_mutex,
                [this]()
                {
                    return m_entered;
                });
        }

        void release()
        {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_released = true;
            }
            m_cv.notify_all();
        }

    private:
        std::mutex m_mutex;
        std::condition_variable m_cv;
        uint64_t m_blockedTicket = 1u;
        bool m_throwAfterRelease = false;
        bool m_entered = false;
        bool m_released = false;
    };

    void appendU64(
        std::vector<uint8_t> &bytes,
        uint64_t value)
    {
        const size_t offset = bytes.size();
        bytes.resize(offset + sizeof(value));
        std::memcpy(
            bytes.data() + offset,
            &value,
            sizeof(value));
    }

    GsDrainBatchCommand makeScanMaskBatch(
        uint8_t mask)
    {
        constexpr uint64_t packedAdTag =
            1ull | (1ull << 15u) |
            (1ull << 60u);
        std::vector<uint8_t> bytes;
        appendU64(bytes, packedAdTag);
        appendU64(bytes, 0x0Eull);
        appendU64(bytes, mask);
        appendU64(bytes, GS_REG_SCANMSK);

        GsDrainBatchCommand command{};
        command.batch.storage[2] = std::move(bytes);
        command.batch.packets.push_back(
            GifArbiterDrainBatch::Packet{
                .pathId = GifPathId::Path2,
                .storageIndex = 2u,
                .size = command.batch.storage[2].size(),
            });
        return command;
    }

    struct GsFixture
    {
        explicit GsFixture(bool digests = true)
            : vram(PS2_GS_VRAM_SIZE, 0u),
              processor(gs, digests)
        {
            gs.init(
                vram.data(),
                static_cast<uint32_t>(vram.size()),
                nullptr);
        }

        std::vector<uint8_t> vram;
        GS gs;
        GsCommandProcessor processor;
    };
}

void register_ps2_gs_command_stream_tests()
{
    MiniTest::Case("PS2 GS command transport", [](TestCase &tc)
    {
        tc.Run("bounded SPSC queue preserves empty full and FIFO transitions", [](TestCase &t)
        {
            BoundedSpscQueue<uint32_t> queue(3u);
            t.IsTrue(queue.empty(),
                     "a new queue should be empty");
            t.IsFalse(queue.full(),
                      "a new queue should not be full");
            t.Equals(queue.capacity(), static_cast<size_t>(3u),
                     "the queue should expose all requested slots");

            t.IsTrue(queue.tryPush(10u),
                     "the first entry should fit");
            t.IsTrue(queue.tryPush(20u),
                     "the second entry should fit");
            t.IsTrue(queue.tryPush(30u),
                     "the final bounded entry should fit");
            t.IsTrue(queue.full(),
                     "all usable slots should report full");
            t.IsFalse(queue.tryPush(40u),
                      "a full queue must retain producer backpressure");
            t.Equals(queue.size(), static_cast<size_t>(3u),
                     "a rejected push must not alter depth");

            const std::optional<uint32_t> first = queue.tryPop();
            const std::optional<uint32_t> second = queue.tryPop();
            const std::optional<uint32_t> third = queue.tryPop();
            t.IsTrue(
                first == 10u && second == 20u && third == 30u,
                "the consumer should observe exact FIFO order");
            t.IsFalse(queue.tryPop().has_value(),
                      "an empty queue must not fabricate an entry");
            t.IsTrue(queue.empty(),
                     "popping every entry should restore empty state");

            const std::string zeroCapacity = captureException([]()
            {
                BoundedSpscQueue<uint8_t> invalid(0u);
                (void)invalid;
            });
            t.IsFalse(zeroCapacity.empty(),
                      "zero queue capacity should fail closed");
        });

        tc.Run("bounded SPSC queue wraps every cursor and retains terminal variable payload", [](TestCase &t)
        {
            for (size_t capacity = 1u; capacity <= 8u; ++capacity)
            {
                BoundedSpscQueue<uint64_t> queue(capacity);
                std::vector<bool> producerPositions(
                    capacity + 1u, false);
                std::vector<bool> consumerPositions(
                    capacity + 1u, false);
                for (uint64_t value = 0u;
                     value < (capacity + 1u) * 4u;
                     ++value)
                {
                    t.IsTrue(queue.tryPush(value),
                             "one-at-a-time producer traffic should fit");
                    producerPositions[queue.producerPosition()] = true;
                    const std::optional<uint64_t> observed =
                        queue.tryPop();
                    t.IsTrue(
                        observed.has_value() && *observed == value,
                        "one-at-a-time wrap traffic should remain exact");
                    consumerPositions[queue.consumerPosition()] = true;
                }
                bool visitedEveryPosition = true;
                for (size_t position = 0u;
                     position <= capacity; ++position)
                {
                    visitedEveryPosition =
                        visitedEveryPosition &&
                        producerPositions[position] &&
                        consumerPositions[position];
                }
                t.IsTrue(visitedEveryPosition,
                         "producer and consumer should wrap through every physical cursor boundary");
            }

            BoundedSpscQueue<std::vector<uint8_t>> queue(2u);
            for (uint8_t value = 0u; value < 2u; ++value)
            {
                t.IsTrue(queue.tryPush(std::vector<uint8_t>{value}),
                         "cursor setup payload should fit");
                (void)queue.tryPop();
            }
            t.Equals(queue.producerPosition(), static_cast<size_t>(2u),
                     "the next payload should occupy the terminal physical slot");
            std::vector<uint8_t> terminalPayload(4097u);
            for (size_t index = 0u;
                 index < terminalPayload.size(); ++index)
            {
                terminalPayload[index] =
                    static_cast<uint8_t>((index * 37u) & 0xFFu);
            }
            const std::vector<uint8_t> expected = terminalPayload;
            t.IsTrue(queue.tryPush(std::move(terminalPayload)),
                     "a variable payload at the terminal slot should fit");
            t.Equals(queue.producerPosition(), static_cast<size_t>(0u),
                     "publishing the terminal variable payload should wrap the producer");
            const std::optional<std::vector<uint8_t>> observed =
                queue.tryPop();
            t.IsTrue(observed.has_value() && *observed == expected,
                     "variable payload ownership must survive the ring end exactly");
        });

        tc.Run("threaded GS owner starts lazily and processes on its own thread", [](TestCase &t)
        {
            GsFixture fixture;
            const std::thread::id producer =
                std::this_thread::get_id();
            std::mutex observedMutex;
            std::thread::id observedOwner{};
            ThreadedGsExecutor executor(
                fixture.processor,
                ThreadedGsExecutorOptions{
                    .queueCapacity = 4u,
                    .payloadCapacityBytes = 4096u,
                    .beforeProcess =
                        [&](const GsCommand &, uint64_t)
                        {
                            std::lock_guard<std::mutex> lock(
                                observedMutex);
                            observedOwner =
                                std::this_thread::get_id();
                        },
                });
            t.IsFalse(executor.statistics().started,
                      "constructing an executor should not race GS initialization by starting early");

            const GsCommandResult barrier =
                executor.submit(GsBarrierCommand{});
            std::thread::id callbackOwner{};
            {
                std::lock_guard<std::mutex> lock(observedMutex);
                callbackOwner = observedOwner;
            }
            t.IsTrue(barrier.completed(),
                     "the first owner command should complete");
            t.IsTrue(callbackOwner != std::thread::id{} &&
                         callbackOwner != producer,
                     "the command processor should execute on the GS owner thread");
            t.IsTrue(executor.ownerThreadId() == callbackOwner,
                     "the executor should publish its stable owner identity");
            const ThreadedGsExecutorStatistics stats =
                executor.statistics();
            t.IsTrue(stats.started && stats.running,
                     "the lazy worker should be live after first submission");
            t.Equals(stats.queueCapacity, static_cast<size_t>(4u),
                     "statistics should retain the bounded slot capacity");
            t.Equals(stats.queueHighWater, static_cast<size_t>(1u),
                     "synchronous owner traffic should require one queued slot");
            t.Equals(stats.completedTickets, 1ull,
                     "the first transport ticket should retire monotonically");
            t.Equals(stats.barriersCompleted, 1ull,
                     "barrier retirement should be owner-local and ordered");
        });

        tc.Run("threaded GS VRAM observation returns an owned owner-side snapshot", [](TestCase &t)
        {
            GsFixture fixture;
            fixture.vram.front() = 0x5Au;
            fixture.vram.back() = 0xA5u;
            ThreadedGsExecutor executor(
                fixture.processor,
                ThreadedGsExecutorOptions{
                    .queueCapacity = 2u,
                    .payloadCapacityBytes = 4096u,
                });

            GsVramSnapshotResult snapshot =
                takeGsCommandResult<GsVramSnapshotResult>(
                    executor.submit(GsCopyVramCommand{}));
            t.IsTrue(snapshot.available,
                     "initialized owner VRAM should be observable");
            t.Equals(snapshot.bytes.size(), fixture.vram.size(),
                     "the owner should copy the complete bounded VRAM image");
            t.IsTrue(snapshot.bytes == fixture.vram,
                     "the command result should match owner VRAM byte-for-byte");

            fixture.vram.front() = 0u;
            t.Equals(snapshot.bytes.front(), static_cast<uint8_t>(0x5Au),
                     "the result must own its bytes after the command completes");
        });

        tc.Run("threaded GS queue applies slot backpressure without loss or lost wakeup", [](TestCase &t)
        {
            GsFixture fixture;
            const auto gate = std::make_shared<WorkerGate>();
            ThreadedGsExecutor executor(
                fixture.processor,
                ThreadedGsExecutorOptions{
                    .queueCapacity = 2u,
                    .payloadCapacityBytes = 4096u,
                    .beforeProcess =
                        [gate](const GsCommand &command, uint64_t ticket)
                        {
                            gate->beforeProcess(command, ticket);
                        },
                });

            GsCommandSubmission first =
                executor.submitAsync(GsBarrierCommand{});
            t.IsTrue(gate->waitUntilEntered(),
                     "the fixture should hold the first in-flight command");
            GsCommandSubmission second =
                executor.submitAsync(GsBarrierCommand{});
            GsCommandSubmission third =
                executor.submitAsync(GsBarrierCommand{});
            t.Equals(executor.statistics().queueDepth,
                     static_cast<size_t>(2u),
                     "two queued commands should fill the bounded ring while one is active");

            std::atomic<bool> fourthStarted{false};
            std::atomic<bool> fourthSubmitted{false};
            std::optional<GsCommandSubmission> fourth;
            std::string fourthError;
            std::thread producer([&]()
            {
                fourthStarted.store(true, std::memory_order_release);
                try
                {
                    fourth.emplace(
                        executor.submitAsync(GsBarrierCommand{}));
                    fourthSubmitted.store(true, std::memory_order_release);
                }
                catch (const std::exception &error)
                {
                    fourthError = error.what();
                }
            });
            t.IsTrue(
                waitUntil([&]()
                {
                    return fourthStarted.load(
                        std::memory_order_acquire);
                }),
                "the fourth producer should start within the bounded test interval");
            const bool fourthBlocked = waitUntil([&]()
            {
                return executor.statistics().producerBlockCount >= 1u;
            });
            t.IsTrue(fourthBlocked,
                     "the fourth producer should reach bounded backpressure");
            t.IsFalse(
                fourthSubmitted.load(std::memory_order_acquire),
                "a full ring should block its serialized producer");

            gate->release();
            producer.join();
            t.IsTrue(fourthError.empty() && fourth.has_value(),
                     "freeing one slot should wake and admit the blocked producer exactly once");
            const GsCommandResult firstResult = first.wait();
            const GsCommandResult secondResult = second.wait();
            const GsCommandResult thirdResult = third.wait();
            const GsCommandResult fourthResult = fourth->wait();
            t.IsTrue(firstResult.completed() && secondResult.completed() &&
                         thirdResult.completed() && fourthResult.completed(),
                     "backpressure must not drop or reject guest work");
            t.Equals(firstResult.identity.sequence, 1ull,
                     "the first command should retain sequence one");
            t.Equals(fourthResult.identity.sequence, 4ull,
                     "the woken producer should retain FIFO identity");
            const ThreadedGsExecutorStatistics stats =
                executor.statistics();
            t.Equals(stats.queueHighWater, static_cast<size_t>(2u),
                     "queue high-water should reach but not exceed capacity");
            t.IsTrue(stats.producerBlockCount >= 1u &&
                         stats.producerBlockedNanoseconds > 0u,
                     "backpressure should retain bounded wait evidence");

            for (uint64_t index = 0u; index < 100u; ++index)
            {
                t.IsTrue(
                    executor.submit(GsBarrierCommand{}).completed(),
                    "repeated idle-to-active wakeups must not lose a notification");
            }
        });

        tc.Run("threaded GS payload budget blocks independently of free slots", [](TestCase &t)
        {
            GsFixture fixture;
            const auto gate = std::make_shared<WorkerGate>();
            const GsRecentGifPacketsCommand probe{.limit = 3u};
            const uint64_t bytes = gsCommandPayloadSize(probe);
            t.IsTrue(bytes > 0u,
                     "the variable observation command should consume logical payload budget");
            ThreadedGsExecutor executor(
                fixture.processor,
                ThreadedGsExecutorOptions{
                    .queueCapacity = 8u,
                    .payloadCapacityBytes = bytes * 2u,
                    .beforeProcess =
                        [gate](const GsCommand &command, uint64_t ticket)
                        {
                            gate->beforeProcess(command, ticket);
                        },
                });

            GsCommandSubmission first =
                executor.submitAsync(probe);
            t.IsTrue(gate->waitUntilEntered(),
                     "the first payload should remain owner-active");
            GsCommandSubmission second =
                executor.submitAsync(probe);

            std::atomic<bool> thirdSubmitted{false};
            std::atomic<bool> thirdStarted{false};
            std::optional<GsCommandSubmission> third;
            std::thread producer([&]()
            {
                thirdStarted.store(true, std::memory_order_release);
                third.emplace(executor.submitAsync(probe));
                thirdSubmitted.store(true, std::memory_order_release);
            });
            t.IsTrue(
                waitUntil([&]()
                {
                    return thirdStarted.load(
                        std::memory_order_acquire);
                }),
                "the third payload producer should start within the bounded test interval");
            const bool thirdBlocked = waitUntil([&]()
            {
                return executor.statistics().producerBlockCount >= 1u;
            });
            t.IsTrue(thirdBlocked,
                     "the third payload should reach bounded byte backpressure");
            t.IsFalse(
                thirdSubmitted.load(std::memory_order_acquire),
                "the payload bound should block even with unused command slots");
            t.Equals(executor.statistics().payloadHighWaterBytes,
                     bytes * 2u,
                     "payload high-water should stop exactly at its configured bound");

            gate->release();
            producer.join();
            t.IsTrue(first.wait().completed() &&
                         second.wait().completed() &&
                         third->wait().completed(),
                     "payload backpressure should preserve every completion");
            t.IsTrue(executor.statistics().queueHighWater < 8u,
                     "the payload gate should distinguish itself from slot exhaustion");
        });

        tc.Run("threaded GS barriers retire in FIFO order after prior mutations", [](TestCase &t)
        {
            GsFixture fixture;
            ThreadedGsExecutor executor(
                fixture.processor,
                ThreadedGsExecutorOptions{
                    .queueCapacity = 8u,
                    .payloadCapacityBytes = 4096u,
                });
            GsCommandSubmission first = executor.submitAsync(
                GsWriteRegisterCommand{
                    .address = GS_REG_SCANMSK,
                    .value = 1u,
                });
            GsCommandSubmission barrier =
                executor.submitAsync(GsBarrierCommand{});
            GsCommandSubmission third = executor.submitAsync(
                GsWriteRegisterCommand{
                    .address = GS_REG_SCANMSK,
                    .value = 3u,
                });

            const GsCommandResult firstResult = first.wait();
            const GsCommandResult barrierResult = barrier.wait();
            const GsCommandResult thirdResult = third.wait();
            t.IsTrue(firstResult.completed() &&
                         barrierResult.completed() &&
                         thirdResult.completed(),
                     "all commands around the barrier should complete");
            t.Equals(firstResult.identity.sequence, 1ull,
                     "the first mutation should lead the stream");
            t.Equals(barrierResult.identity.sequence, 2ull,
                     "the barrier should retain its exact stream position");
            t.Equals(thirdResult.identity.sequence, 3ull,
                     "the later mutation must not overtake the barrier");
            t.Equals(executor.lastCompletedTicket(), 3ull,
                     "transport completion should advance monotonically through the barrier");

            const GsReplayState state =
                takeGsCommandResult<GsReplayStateResult>(
                    executor.submit(GsCaptureReplayStateCommand{}))
                    .state;
            t.Equals(state.scanMask, static_cast<uint8_t>(3u),
                     "observing after the barrier should include the later ordered mutation");
            t.Equals(executor.statistics().barriersCompleted, 1ull,
                     "the owner should count the explicit barrier exactly once");
        });

        tc.Run("threaded GS cancellation rejects stale queued generations and admits reset", [](TestCase &t)
        {
            GsFixture fixture;
            const auto gate = std::make_shared<WorkerGate>();
            ThreadedGsExecutor executor(
                fixture.processor,
                ThreadedGsExecutorOptions{
                    .queueCapacity = 4u,
                    .payloadCapacityBytes = 4096u,
                    .beforeProcess =
                        [gate](const GsCommand &command, uint64_t ticket)
                        {
                            gate->beforeProcess(command, ticket);
                        },
                });

            GsDrainBatchCommand staleBatch{};
            staleBatch.batch.storage[1] = {1u, 2u, 3u, 4u};
            staleBatch.batch.packets.push_back(
                GifArbiterDrainBatch::Packet{
                    .pathId = GifPathId::Path1,
                    .storageIndex = 1u,
                    .size = 4u,
                });
            GsCommandSubmission oldFirst =
                executor.submitAsync(std::move(staleBatch));
            t.IsTrue(gate->waitUntilEntered(),
                     "the first old-generation command should be held before mutation");
            GsCommandSubmission oldSecond =
                executor.submitAsync(GsBarrierCommand{});
            GsCommandSubmission reset =
                executor.submitAsync(GsResetCommand{});
            t.Equals(reset.identity().generation, 2ull,
                     "the queued reset should reserve the next generation");
            t.Equals(reset.identity().sequence, 1ull,
                     "the queued reset should begin at sequence one");

            executor.cancelPendingBeforeGeneration(2u);
            gate->release();
            GsCommandResult oldFirstResult = oldFirst.wait();
            const GsCommandResult oldSecondResult = oldSecond.wait();
            const GsCommandResult resetResult = reset.wait();
            t.Equals(
                static_cast<uint32_t>(oldFirstResult.disposition),
                static_cast<uint32_t>(
                    GsCommandDisposition::StaleGeneration),
                "the held old generation should be cancelled before GS mutation");
            t.Equals(
                static_cast<uint32_t>(oldSecondResult.disposition),
                static_cast<uint32_t>(
                    GsCommandDisposition::StaleGeneration),
                "all remaining old-generation work should be rejected in FIFO order");
            const auto *returnedBatch =
                std::get_if<GsDrainBatchResult>(
                    &oldFirstResult.payload);
            t.IsTrue(returnedBatch != nullptr &&
                         returnedBatch->batch.storage[1].size() == 4u,
                     "stale drain cancellation should return owned backing storage for recycling");
            t.IsTrue(resetResult.completed(),
                     "the matching reset should begin the admitted generation");

            (void)executor.submit(
                GsWriteRegisterCommand{
                    .address = GS_REG_SCANMSK,
                    .value = 2u,
                });
            const GsReplayState state =
                takeGsCommandResult<GsReplayStateResult>(
                    executor.submit(GsCaptureReplayStateCommand{}))
                    .state;
            t.Equals(state.scanMask, static_cast<uint8_t>(2u),
                     "current-generation work should proceed after cancellation and reset");
        });

        tc.Run("threaded GS worker failure reaches current queued and waiting producers", [](TestCase &t)
        {
            GsFixture fixture;
            const auto gate =
                std::make_shared<WorkerGate>(1u, true);
            ThreadedGsExecutor executor(
                fixture.processor,
                ThreadedGsExecutorOptions{
                    .queueCapacity = 1u,
                    .payloadCapacityBytes = 4096u,
                    .beforeProcess =
                        [gate](const GsCommand &command, uint64_t ticket)
                        {
                            gate->beforeProcess(command, ticket);
                        },
                });
            GsCommandSubmission first =
                executor.submitAsync(GsBarrierCommand{});
            t.IsTrue(gate->waitUntilEntered(),
                     "the failing owner hook should be held deterministically");
            GsCommandSubmission second =
                executor.submitAsync(GsBarrierCommand{});

            std::atomic<bool> thirdStarted{false};
            std::string thirdError;
            std::thread producer([&]()
            {
                thirdStarted.store(true, std::memory_order_release);
                thirdError = captureException([&]()
                {
                    (void)executor.submitAsync(
                        GsBarrierCommand{});
                });
            });
            t.IsTrue(
                waitUntil([&]()
                {
                    return thirdStarted.load(
                        std::memory_order_acquire);
                }),
                "the waiting producer should start within the bounded test interval");
            const bool thirdBlocked = waitUntil([&]()
            {
                return executor.statistics().producerBlockCount >= 1u;
            });
            t.IsTrue(thirdBlocked,
                     "the third producer should be waiting when failure is injected");
            gate->release();
            producer.join();

            const std::string firstError = captureException([&]()
            {
                (void)first.wait();
            });
            const std::string secondError = captureException([&]()
            {
                (void)second.wait();
            });
            t.IsTrue(
                firstError.find("injected GS owner failure") !=
                    std::string::npos,
                "the active command should receive the owner exception");
            t.IsTrue(
                secondError.find("injected GS owner failure") !=
                    std::string::npos,
                "already queued work should receive the same fatal exception");
            t.IsTrue(
                thirdError.find("injected GS owner failure") !=
                    std::string::npos,
                "a producer blocked on capacity should wake and receive the fatal exception");
            t.IsTrue(executor.statistics().failed,
                     "fatal owner state should remain observable");
            const std::string laterError = captureException([&]()
            {
                (void)executor.submit(GsBarrierCommand{});
            });
            t.IsTrue(
                laterError.find("injected GS owner failure") !=
                    std::string::npos,
                "future submissions should deterministically rethrow the owner failure");
        });

        tc.Run("threaded GS shutdown handles empty partial full cancel and drain", [](TestCase &t)
        {
            {
                GsFixture fixture;
                ThreadedGsExecutor executor(fixture.processor);
                executor.shutdown(GsExecutorShutdownMode::Drain);
                const ThreadedGsExecutorStatistics stats =
                    executor.statistics();
                t.IsFalse(stats.started || stats.running,
                          "shutdown before first submission should not create a worker");
                t.IsFalse(stats.accepting,
                          "empty shutdown should close future admission");
            }

            for (size_t queued = 1u; queued <= 2u; ++queued)
            {
                GsFixture fixture;
                const auto gate = std::make_shared<WorkerGate>();
                ThreadedGsExecutor executor(
                    fixture.processor,
                    ThreadedGsExecutorOptions{
                        .queueCapacity = 2u,
                        .payloadCapacityBytes = 4096u,
                        .beforeProcess =
                            [gate](const GsCommand &command, uint64_t ticket)
                            {
                                gate->beforeProcess(command, ticket);
                            },
                    });
                std::vector<GsCommandSubmission> submissions;
                submissions.push_back(
                    executor.submitAsync(GsBarrierCommand{}));
                t.IsTrue(gate->waitUntilEntered(),
                         "shutdown fixture should hold one owner-active command");
                for (size_t index = 0u; index < queued; ++index)
                {
                    submissions.push_back(
                        executor.submitAsync(GsBarrierCommand{}));
                }
                t.Equals(executor.statistics().queueDepth, queued,
                         "shutdown fixture should reach its requested queue occupancy");

                std::atomic<bool> shutdownStarted{false};
                std::atomic<bool> shutdownReturned{false};
                std::thread shutdownThread([&]()
                {
                    shutdownStarted.store(
                        true, std::memory_order_release);
                    executor.shutdown(
                        GsExecutorShutdownMode::Cancel);
                    shutdownReturned.store(
                        true, std::memory_order_release);
                });
                t.IsTrue(
                    waitUntil([&]()
                    {
                        return shutdownStarted.load(
                            std::memory_order_acquire);
                    }),
                    "the shutdown thread should start within the bounded test interval");
                t.IsFalse(
                    shutdownReturned.load(
                        std::memory_order_acquire),
                    "shutdown should join an owner still inside a command boundary");
                gate->release();
                shutdownThread.join();
                bool everyCancelled = true;
                for (GsCommandSubmission &submission : submissions)
                {
                    everyCancelled =
                        everyCancelled &&
                        !captureException([&]()
                        {
                            (void)submission.wait();
                        }).empty();
                }
                t.IsTrue(everyCancelled,
                         "cancel shutdown should resolve active and queued completions");
                const ThreadedGsExecutorStatistics stats =
                    executor.statistics();
                t.IsFalse(stats.running || stats.accepting,
                          "cancel shutdown should leave no live owner or admission");
                t.Equals(stats.queueDepth, static_cast<size_t>(0u),
                         "cancel shutdown should empty the slot ring");
                t.Equals(stats.queuedPayloadBytes, 0ull,
                         "cancel shutdown should release the full payload budget");
            }

            {
                GsFixture fixture;
                ThreadedGsExecutor executor(
                    fixture.processor,
                    ThreadedGsExecutorOptions{
                        .queueCapacity = 16u,
                        .payloadCapacityBytes = 4096u,
                    });
                std::vector<GsCommandSubmission> submissions;
                for (uint64_t value = 0u; value < 10u; ++value)
                {
                    submissions.push_back(
                        executor.submitAsync(
                            GsWriteRegisterCommand{
                                .address = GS_REG_SCANMSK,
                                .value = value & 3u,
                            }));
                }
                executor.shutdown(GsExecutorShutdownMode::Drain);
                bool everyCompleted = true;
                for (GsCommandSubmission &submission : submissions)
                {
                    everyCompleted =
                        everyCompleted &&
                        submission.wait().completed();
                }
                t.IsTrue(everyCompleted,
                         "drain shutdown should retire all accepted commands");
                t.Equals(fixture.gs.captureReplayState().scanMask,
                         static_cast<uint8_t>(1u),
                         "drain shutdown should preserve the final ordered mutation");
            }
        });

        tc.Run("threaded synchronous and inline streams match under deterministic worker jitter", [](TestCase &t)
        {
            GsFixture inlineFixture;
            GsFixture threadedFixture;
            InlineGsExecutor inlineExecutor(
                inlineFixture.processor);
            ThreadedGsExecutor threadedExecutor(
                threadedFixture.processor,
                ThreadedGsExecutorOptions{
                    .queueCapacity = 4u,
                    .payloadCapacityBytes = 1024u * 1024u,
                    .beforeProcess =
                        [](const GsCommand &, uint64_t ticket)
                        {
                            const uint64_t delay =
                                ((ticket * 1103515245ull + 12345ull) >> 8u) %
                                5u;
                            std::this_thread::sleep_for(
                                std::chrono::microseconds(
                                    delay * 25u));
                        },
                });

            bool allDigestsMatch = true;
            for (uint64_t index = 0u; index < 256u; ++index)
            {
                GsCommandResult inlineResult{};
                GsCommandResult threadedResult{};
                if ((index % 13u) == 0u)
                {
                    inlineResult = inlineExecutor.submit(
                        makeScanMaskBatch(
                            static_cast<uint8_t>(index & 3u)),
                        index * 16u, index + 1000u, index / 2u);
                    threadedResult = threadedExecutor.submit(
                        makeScanMaskBatch(
                            static_cast<uint8_t>(index & 3u)),
                        index * 16u, index + 1000u, index / 2u);
                }
                else
                {
                    const GsWriteRegisterCommand command{
                        .address = GS_REG_SCANMSK,
                        .value = index & 3u,
                    };
                    inlineResult = inlineExecutor.submit(
                        command,
                        index * 16u, index + 1000u, index / 2u);
                    threadedResult = threadedExecutor.submit(
                        command,
                        index * 16u, index + 1000u, index / 2u);
                }
                allDigestsMatch =
                    allDigestsMatch &&
                    inlineResult.digest == threadedResult.digest &&
                    inlineResult.identity == threadedResult.identity;
            }
            t.IsTrue(allDigestsMatch,
                     "owner scheduling jitter must not alter command identity or digest");

            const GsReplayState inlineState =
                takeGsCommandResult<GsReplayStateResult>(
                    inlineExecutor.submit(
                        GsCaptureReplayStateCommand{}))
                    .state;
            const GsReplayState threadedState =
                takeGsCommandResult<GsReplayStateResult>(
                    threadedExecutor.submit(
                        GsCaptureReplayStateCommand{}))
                    .state;
            std::vector<uint8_t> inlineEncoded;
            std::vector<uint8_t> threadedEncoded;
            t.IsTrue(
                encodeGsReplayState(inlineState, inlineEncoded) &&
                    encodeGsReplayState(
                        threadedState, threadedEncoded),
                "both differential states should serialize canonically");
            t.IsTrue(inlineEncoded == threadedEncoded,
                     "inline and threaded owner frontend state should be byte-identical");
            t.IsTrue(inlineFixture.vram == threadedFixture.vram,
                     "inline and threaded owner VRAM should remain byte-identical");
            t.Equals(threadedExecutor.lastSubmittedTicket(), 257ull,
                     "every jittered command plus final observation should receive one transport ticket");
        });

        tc.Run("runtime can select threaded synchronous GS execution once", [](TestCase &t)
        {
            PS2RuntimeConfiguration configuration =
                defaultPs2RuntimeConfiguration();
            configuration.gsExecutionMode =
                GsExecutionMode::ThreadedSynchronous;
            configuration.gsCommandQueueCapacity = 2u;
            configuration.gsCommandPayloadCapacityBytes =
                1024u * 1024u;
            PS2Runtime runtime(configuration);
            t.IsTrue(runtime.memory().initialize(),
                     "threaded runtime memory should initialize");
            t.IsTrue(runtime.syncCoreSubsystems(),
                     "threaded runtime should bind GS only before its lazy owner starts");
            (void)runtime.submitEeGsCommand(
                GsWriteRegisterCommand{
                    .address = GS_REG_SCANMSK,
                    .value = 2u,
                });
            const GsReplayState state =
                takeGsCommandResult<GsReplayStateResult>(
                    runtime.submitGsCommand(
                        GsCaptureReplayStateCommand{}))
                    .state;
            t.Equals(state.scanMask, static_cast<uint8_t>(2u),
                     "runtime-selected threaded mode should consume the same semantic command stream");
        });

#if PS2X_ENABLE_RUNTIME_DIAGNOSTICS
        tc.Run("threaded GS diagnostics retain one hot producer", [](TestCase &t)
        {
            GsFixture fixture;
            ThreadedGsExecutor executor(fixture.processor);
            (void)executor.submit(
                GsWriteRegisterCommand{
                    .address = GS_REG_SCANMSK,
                    .value = 1u,
                });
            std::string secondProducerError;
            std::thread secondProducer([&]()
            {
                secondProducerError = captureException([&]()
                {
                    (void)executor.submit(
                        GsWriteRegisterCommand{
                            .address = GS_REG_SCANMSK,
                            .value = 3u,
                        });
                });
            });
            secondProducer.join();
            t.IsFalse(secondProducerError.empty(),
                      "thread ownership diagnostics should reject a second hot producer before enqueue");
            const GsReplayState state =
                takeGsCommandResult<GsReplayStateResult>(
                    executor.submit(
                        GsCaptureReplayStateCommand{}))
                    .state;
            t.Equals(state.scanMask, static_cast<uint8_t>(1u),
                     "the rejected producer must not reach the GS owner");
        });
#endif
    });
}
