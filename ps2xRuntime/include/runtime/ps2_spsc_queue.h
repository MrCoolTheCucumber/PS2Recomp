#ifndef PS2_SPSC_QUEUE_H
#define PS2_SPSC_QUEUE_H

#include <atomic>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

// A bounded single-producer/single-consumer ring. The producer owns the write
// cursor and the consumer owns the read cursor. The extra physical slot makes
// the empty and full states unambiguous while exposing exactly `capacity`
// usable entries.
//
// Publication is release/acquire: a consumer cannot observe an entry before
// its construction completes, and a producer cannot reuse a slot before the
// consumer destroys the prior entry. Blocking and wake policy deliberately
// live in the subsystem executor rather than this reusable transport.
template <typename T>
class BoundedSpscQueue
{
public:
    explicit BoundedSpscQueue(size_t capacity)
        : m_slots(storageCapacity(capacity)),
          m_capacity(capacity)
    {
    }

    BoundedSpscQueue(const BoundedSpscQueue &) = delete;
    BoundedSpscQueue &operator=(const BoundedSpscQueue &) = delete;

    [[nodiscard]] size_t capacity() const noexcept
    {
        return m_capacity;
    }

    [[nodiscard]] bool empty() const noexcept
    {
        const size_t consumer =
            m_consumer.value.load(std::memory_order_relaxed);
        return consumer ==
               m_producer.value.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool full() const noexcept
    {
        const size_t producer =
            m_producer.value.load(std::memory_order_relaxed);
        return advance(producer) ==
               m_consumer.value.load(std::memory_order_acquire);
    }

    [[nodiscard]] size_t size() const noexcept
    {
        const size_t consumer =
            m_consumer.value.load(std::memory_order_acquire);
        const size_t producer =
            m_producer.value.load(std::memory_order_acquire);
        if (producer >= consumer)
            return producer - consumer;
        return m_slots.size() - consumer + producer;
    }

    [[nodiscard]] size_t producerPosition() const noexcept
    {
        return m_producer.value.load(std::memory_order_acquire);
    }

    [[nodiscard]] size_t consumerPosition() const noexcept
    {
        return m_consumer.value.load(std::memory_order_acquire);
    }

    template <typename... Args>
    bool tryEmplace(Args &&...args)
    {
        const size_t producer =
            m_producer.value.load(std::memory_order_relaxed);
        const size_t next = advance(producer);
        if (next ==
            m_consumer.value.load(std::memory_order_acquire))
        {
            return false;
        }

        std::optional<T> &slot = m_slots[producer];
        if (slot.has_value())
        {
            throw std::logic_error(
                "SPSC producer reached an occupied slot");
        }
        slot.emplace(std::forward<Args>(args)...);
        m_producer.value.store(next, std::memory_order_release);
        return true;
    }

    bool tryPush(T &&value)
    {
        return tryEmplace(std::move(value));
    }

    bool tryPush(const T &value)
    {
        return tryEmplace(value);
    }

    [[nodiscard]] std::optional<T> tryPop()
    {
        const size_t consumer =
            m_consumer.value.load(std::memory_order_relaxed);
        if (consumer ==
            m_producer.value.load(std::memory_order_acquire))
        {
            return std::nullopt;
        }

        std::optional<T> &slot = m_slots[consumer];
        if (!slot.has_value())
        {
            throw std::logic_error(
                "SPSC consumer reached an empty published slot");
        }
        std::optional<T> value(std::move(slot));
        slot.reset();
        m_consumer.value.store(
            advance(consumer), std::memory_order_release);
        return value;
    }

private:
    struct alignas(64) Cursor
    {
        std::atomic<size_t> value{0u};
    };

    [[nodiscard]] static size_t storageCapacity(size_t capacity)
    {
        if (capacity == 0u)
        {
            throw std::invalid_argument(
                "SPSC queue capacity must be non-zero");
        }
        if (capacity == std::numeric_limits<size_t>::max())
        {
            throw std::length_error(
                "SPSC queue capacity cannot be represented");
        }
        return capacity + 1u;
    }

    [[nodiscard]] size_t advance(size_t position) const noexcept
    {
        ++position;
        return position == m_slots.size() ? 0u : position;
    }

    std::vector<std::optional<T>> m_slots;
    size_t m_capacity = 0u;
    Cursor m_producer;
    Cursor m_consumer;
};

#endif
