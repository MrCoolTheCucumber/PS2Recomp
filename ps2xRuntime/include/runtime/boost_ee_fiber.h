#pragma once

#include <cstddef>
#include <exception>
#include <functional>
#include <memory>

// Stackful continuation used only by the legacy generated-C++ execution
// backend. Boost.Context types remain confined to the implementation.
//
// Windows x64 deliberately uses Boost fcontext as well. It may not preserve
// undocumented TIB fields, so code suspended here must not rely on those
// fields; this is not backed by a second native-Win32-Fiber implementation.
class BoostEeFiber final
{
public:
    static constexpr size_t DefaultUsableStackBytes =
        1024u * 1024u;

    enum class State
    {
        Created,
        Running,
        Suspended,
        Finished,
        Failed,
        Destroyed,
    };

    enum class StackFailureForTesting
    {
        None,
        Allocation,
        GuardSetup,
    };

    struct Options
    {
        size_t usableStackBytes =
            DefaultUsableStackBytes;
        StackFailureForTesting failureForTesting =
            StackFailureForTesting::None;
    };

    struct StackUsage
    {
        size_t allocationCount = 0;
        size_t deallocationCount = 0;
        size_t activeAllocationCount = 0;
        size_t requestedUsableBytes = 0;
        size_t usableBytes = 0;
        size_t guardBytes = 0;
        size_t committedBytes = 0;
        size_t peakCommittedBytes = 0;
        size_t residentBytes = 0;
        size_t peakResidentBytes = 0;
        size_t highWaterBytes = 0;
    };

    using Entry = std::function<void()>;

    explicit BoostEeFiber(Entry entry);
    BoostEeFiber(Entry entry, Options options);
    ~BoostEeFiber() noexcept;

    BoostEeFiber(const BoostEeFiber &) = delete;
    BoostEeFiber &operator=(
        const BoostEeFiber &) = delete;
    BoostEeFiber(BoostEeFiber &&) = delete;
    BoostEeFiber &operator=(BoostEeFiber &&) = delete;

    [[nodiscard]] static bool available() noexcept;

    // Resume is legal only on the constructing executor thread and only from
    // the executor stack. It returns after one yield, normal completion, or
    // a caught guest exception.
    void resume();

    // Yield the currently running BoostEeFiber back to its executor. Calling
    // this outside a running guest fiber is an error.
    static void yieldCurrent();

    // Destroy a finished or suspended continuation on its executor thread.
    // Destroying a suspended continuation performs Boost's forced unwind.
    void destroy();

    [[nodiscard]] State state() const;
    [[nodiscard]] bool resumable() const;
    [[nodiscard]] std::exception_ptr failure() const;
    void rethrowFailure() const;

    // High-water and resident sampling is explicit so a production yield does
    // not scan the full fixed-size stack.
    void sampleStackUsage();
    [[nodiscard]] StackUsage stackUsage() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
