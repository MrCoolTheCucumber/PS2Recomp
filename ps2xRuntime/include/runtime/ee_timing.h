#pragma once

#include <cstdint>
#include <limits>

namespace ps2x::timing
{
    inline constexpr uint64_t kEeTicksPerCycle = 8u;
    inline constexpr uint64_t kEeCyclesPerIopCycle = 8u;

    class EeTickDelta
    {
    public:
        constexpr EeTickDelta() noexcept = default;

        [[nodiscard]] static constexpr EeTickDelta fromRaw(
            uint64_t value) noexcept
        {
            return EeTickDelta(value);
        }

        [[nodiscard]] constexpr uint64_t raw() const noexcept
        {
            return m_value;
        }

        friend constexpr bool operator==(
            EeTickDelta, EeTickDelta) noexcept = default;

        friend constexpr bool operator<(
            EeTickDelta lhs, EeTickDelta rhs) noexcept
        {
            return lhs.m_value < rhs.m_value;
        }

    private:
        explicit constexpr EeTickDelta(uint64_t value) noexcept
            : m_value(value)
        {
        }

        uint64_t m_value = 0u;
    };

    class EeTick
    {
    public:
        constexpr EeTick() noexcept = default;

        [[nodiscard]] static constexpr EeTick fromRaw(
            uint64_t value) noexcept
        {
            return EeTick(value);
        }

        [[nodiscard]] constexpr uint64_t raw() const noexcept
        {
            return m_value;
        }

        friend constexpr bool operator==(EeTick, EeTick) noexcept = default;

        friend constexpr bool operator<(EeTick lhs, EeTick rhs) noexcept
        {
            return lhs.m_value < rhs.m_value;
        }

        friend constexpr bool operator<=(EeTick lhs, EeTick rhs) noexcept
        {
            return lhs.m_value <= rhs.m_value;
        }

        friend constexpr bool operator>(EeTick lhs, EeTick rhs) noexcept
        {
            return lhs.m_value > rhs.m_value;
        }

        friend constexpr bool operator>=(EeTick lhs, EeTick rhs) noexcept
        {
            return lhs.m_value >= rhs.m_value;
        }

    private:
        explicit constexpr EeTick(uint64_t value) noexcept
            : m_value(value)
        {
        }

        uint64_t m_value = 0u;
    };

    [[nodiscard]] constexpr EeTickDelta eeTickDeltaFromRaw(
        uint64_t ticks) noexcept
    {
        return EeTickDelta::fromRaw(ticks);
    }

    [[nodiscard]] constexpr EeTick eeTickFromRaw(uint64_t tick) noexcept
    {
        return EeTick::fromRaw(tick);
    }

    [[nodiscard]] constexpr EeTickDelta cyclesToTickDelta(
        uint64_t cycles, uint64_t ticksPerCycle) noexcept
    {
        const uint64_t maximum = std::numeric_limits<uint64_t>::max();
        if (ticksPerCycle != 0u && cycles > maximum / ticksPerCycle)
        {
            return EeTickDelta::fromRaw(maximum);
        }
        return EeTickDelta::fromRaw(cycles * ticksPerCycle);
    }

    [[nodiscard]] constexpr EeTickDelta eeCyclesToTicks(
        uint64_t cycles) noexcept
    {
        return cyclesToTickDelta(cycles, kEeTicksPerCycle);
    }

    [[nodiscard]] constexpr uint64_t eeTicksToCyclesFloor(
        EeTickDelta ticks) noexcept
    {
        return ticks.raw() / kEeTicksPerCycle;
    }

    [[nodiscard]] constexpr uint64_t eeTickToCyclesFloor(
        EeTick tick) noexcept
    {
        return tick.raw() / kEeTicksPerCycle;
    }

    [[nodiscard]] constexpr uint64_t eeTicksToCyclesCeil(
        EeTickDelta ticks) noexcept
    {
        return ticks.raw() / kEeTicksPerCycle +
               ((ticks.raw() % kEeTicksPerCycle) != 0u ? 1u : 0u);
    }

    [[nodiscard]] constexpr EeTickDelta vuCyclesToEeTicks(
        uint64_t cycles) noexcept
    {
        return eeCyclesToTicks(cycles);
    }

    [[nodiscard]] constexpr uint64_t eeTicksToVuCyclesFloor(
        EeTickDelta ticks) noexcept
    {
        return eeTicksToCyclesFloor(ticks);
    }

    [[nodiscard]] constexpr EeTickDelta iopCyclesToEeTicks(
        uint64_t cycles) noexcept
    {
        return cyclesToTickDelta(
            cycles, kEeCyclesPerIopCycle * kEeTicksPerCycle);
    }

    [[nodiscard]] constexpr uint64_t eeTicksToIopCyclesFloor(
        EeTickDelta ticks) noexcept
    {
        return ticks.raw() /
               (kEeCyclesPerIopCycle * kEeTicksPerCycle);
    }

    [[nodiscard]] constexpr EeTick saturatingAdd(
        EeTick tick, EeTickDelta delta) noexcept
    {
        const uint64_t maximum = std::numeric_limits<uint64_t>::max();
        if (delta.raw() > maximum - tick.raw())
        {
            return EeTick::fromRaw(maximum);
        }
        return EeTick::fromRaw(tick.raw() + delta.raw());
    }

    [[nodiscard]] constexpr EeTickDelta elapsedEeTicks(
        EeTick earlier, EeTick later) noexcept
    {
        return EeTickDelta::fromRaw(
            later >= earlier ? later.raw() - earlier.raw() : 0u);
    }

    class EeTimeline
    {
    public:
        [[nodiscard]] constexpr EeTick now() const noexcept
        {
            return m_now;
        }

        [[nodiscard]] constexpr EeTick advance(
            EeTickDelta elapsed) noexcept
        {
            m_now = saturatingAdd(m_now, elapsed);
            return m_now;
        }

        constexpr void reset() noexcept
        {
            m_now = EeTick{};
        }

        constexpr void restore(EeTick now) noexcept
        {
            m_now = now;
        }

    private:
        EeTick m_now{};
    };
}
