#include "runtime/boost_ee_fiber.h"

#ifndef PS2X_HAS_EE_CPP_FIBER_BACKEND
#define PS2X_HAS_EE_CPP_FIBER_BACKEND 0
#endif

#include <cstdint>
#include <stdexcept>
#include <utility>

#if PS2X_HAS_EE_CPP_FIBER_BACKEND

// protected_fixedsize_stack ignores a failed guard setup after BOOST_ASSERT.
// This wrapper performs and checks the platform operation itself, so disable
// only those header assertions in this isolated translation unit.
#ifndef BOOST_DISABLE_ASSERTS
#define BOOST_DISABLE_ASSERTS
#endif

#include <boost/context/detail/exception.hpp>
#include <boost/context/fiber.hpp>
#include <boost/context/protected_fixedsize_stack.hpp>
#include <boost/context/stack_traits.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <system_error>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace
{
    constexpr unsigned char StackWatermark = 0xa5u;

    struct BoostEeFiberStackState
    {
        std::thread::id owner;
        BoostEeFiber::StackFailureForTesting
            failureForTesting =
                BoostEeFiber::StackFailureForTesting::None;
        BoostEeFiber::StackUsage usage;
        unsigned char *mappingBase = nullptr;
        unsigned char *usableBase = nullptr;
        unsigned char *stackTop = nullptr;

        [[nodiscard]] bool containsAddress(
            const void *address) const noexcept
        {
            if (address == nullptr ||
                usableBase == nullptr ||
                stackTop == nullptr)
            {
                return false;
            }

            const std::uintptr_t probe =
                reinterpret_cast<std::uintptr_t>(address);
            const std::uintptr_t begin =
                reinterpret_cast<std::uintptr_t>(
                    usableBase);
            const std::uintptr_t end =
                reinterpret_cast<std::uintptr_t>(
                    stackTop);
            return probe >= begin && probe < end;
        }

        void requireOwner(const char *operation) const
        {
            if (std::this_thread::get_id() != owner)
            {
                throw std::logic_error(
                    std::string("BoostEeFiber ") +
                    operation +
                    " must run on its executor thread");
            }
        }

        void requireOwnerNoexcept() const noexcept
        {
            if (std::this_thread::get_id() != owner)
            {
                std::terminate();
            }
        }

        void sampleHighWater() noexcept
        {
            if (usableBase == nullptr ||
                stackTop == nullptr)
            {
                return;
            }

            const unsigned char *cursor = usableBase;
            while (cursor != stackTop &&
                   *cursor == StackWatermark)
            {
                ++cursor;
            }
            usage.highWaterBytes = std::max(
                usage.highWaterBytes,
                static_cast<size_t>(stackTop - cursor));
        }

        void sampleResident() noexcept
        {
            if (mappingBase == nullptr ||
                usage.committedBytes == 0u ||
                usage.guardBytes == 0u)
            {
                usage.residentBytes = 0u;
                return;
            }

            size_t resident = 0u;
#if defined(_WIN32)
            for (size_t offset = 0u;
                 offset < usage.committedBytes;
                 offset += usage.guardBytes)
            {
                PSAPI_WORKING_SET_EX_INFORMATION info{};
                info.VirtualAddress =
                    mappingBase + offset;
                if (::QueryWorkingSetEx(
                        ::GetCurrentProcess(),
                        &info,
                        sizeof(info)) != FALSE &&
                    info.VirtualAttributes.Valid != 0u)
                {
                    resident += usage.guardBytes;
                }
            }
#else
            for (size_t offset = 0u;
                 offset < usage.committedBytes;
                 offset += usage.guardBytes)
            {
                unsigned char status = 0u;
                if (::mincore(
                        mappingBase + offset,
                        usage.guardBytes,
                        &status) == 0 &&
                    (status & 1u) != 0u)
                {
                    resident += usage.guardBytes;
                }
            }
#endif
            usage.residentBytes = resident;
            usage.peakResidentBytes = std::max(
                usage.peakResidentBytes,
                resident);
        }
    };

    class AccountingProtectedStack
    {
    public:
        explicit AccountingProtectedStack(
            std::shared_ptr<BoostEeFiberStackState> state)
            : m_state(std::move(state)),
              m_allocator(
                  m_state->usage.requestedUsableBytes)
        {
        }

        boost::context::stack_context allocate()
        {
            m_state->requireOwner("stack allocation");
            if (m_state->failureForTesting ==
                BoostEeFiber::StackFailureForTesting::
                    Allocation)
            {
                throw std::bad_alloc();
            }

            boost::context::stack_context context =
                m_allocator.allocate();
            const size_t pageSize =
                boost::context::stack_traits::page_size();
            unsigned char *const base =
                static_cast<unsigned char *>(context.sp) -
                context.size;

            bool guardReady = false;
            int platformError = 0;
            if (m_state->failureForTesting !=
                BoostEeFiber::StackFailureForTesting::
                    GuardSetup)
            {
#if defined(_WIN32)
                DWORD oldProtection = 0u;
                guardReady =
                    ::VirtualProtect(
                        base,
                        pageSize,
                        PAGE_READWRITE | PAGE_GUARD,
                        &oldProtection) != FALSE;
                if (!guardReady)
                {
                    platformError =
                        static_cast<int>(::GetLastError());
                }
#else
                guardReady =
                    ::mprotect(
                        base,
                        pageSize,
                        PROT_NONE) == 0;
                if (!guardReady)
                {
                    platformError = errno;
                }
#endif
            }

            if (!guardReady)
            {
                m_allocator.deallocate(context);
                if (m_state->failureForTesting ==
                    BoostEeFiber::StackFailureForTesting::
                        GuardSetup)
                {
                    throw std::runtime_error(
                        "BoostEeFiber guard setup failed "
                        "(injected)");
                }
                const std::error_code error(
                    platformError,
#if defined(_WIN32)
                    std::system_category()
#else
                    std::generic_category()
#endif
                );
                throw std::system_error(
                    error,
                    "BoostEeFiber guard setup failed");
            }

            m_state->mappingBase = base;
            m_state->usableBase = base + pageSize;
            m_state->stackTop =
                static_cast<unsigned char *>(context.sp);
            m_state->usage.guardBytes = pageSize;
            m_state->usage.usableBytes =
                context.size - pageSize;
            m_state->usage.committedBytes =
                context.size;
            m_state->usage.peakCommittedBytes =
                std::max(
                    m_state->usage.peakCommittedBytes,
                    context.size);
            ++m_state->usage.allocationCount;
            ++m_state->usage.activeAllocationCount;

            std::memset(
                m_state->usableBase,
                StackWatermark,
                m_state->usage.usableBytes);
            m_state->sampleResident();
            return context;
        }

        void deallocate(
            boost::context::stack_context &context) noexcept
        {
            m_state->requireOwnerNoexcept();
            m_state->sampleHighWater();
            m_state->sampleResident();
            m_allocator.deallocate(context);
            m_state->mappingBase = nullptr;
            m_state->usableBase = nullptr;
            m_state->stackTop = nullptr;
            m_state->usage.committedBytes = 0u;
            m_state->usage.residentBytes = 0u;
            ++m_state->usage.deallocationCount;
            if (m_state->usage.activeAllocationCount == 0u)
            {
                std::terminate();
            }
            --m_state->usage.activeAllocationCount;
        }

    private:
        std::shared_ptr<BoostEeFiberStackState> m_state;
        boost::context::protected_fixedsize_stack
            m_allocator;
    };
}

class BoostEeFiber::Impl
{
public:
    Impl(Entry entry, Options options)
        : m_owner(std::this_thread::get_id()),
          m_entry(std::move(entry)),
          m_stack(std::make_shared<
                  BoostEeFiberStackState>())
    {
        if (!m_entry)
        {
            throw std::invalid_argument(
                "BoostEeFiber requires an entry function");
        }

        const size_t minimum =
            boost::context::stack_traits::minimum_size();
        if (options.usableStackBytes < minimum)
        {
            throw std::invalid_argument(
                "BoostEeFiber stack is smaller than "
                "Boost.Context's minimum");
        }

        m_stack->owner = m_owner;
        m_stack->failureForTesting =
            options.failureForTesting;
        m_stack->usage.requestedUsableBytes =
            options.usableStackBytes;

        m_guest = boost::context::fiber(
            std::allocator_arg,
            AccountingProtectedStack(m_stack),
            [this](
                boost::context::fiber &&executor)
                -> boost::context::fiber
            {
                requireOwner("entry");
                if (s_current != nullptr)
                {
                    throw std::logic_error(
                        "BoostEeFiber entry is recursive");
                }

                m_executor = std::move(executor);
                m_state = State::Running;
                s_current = this;
                try
                {
                    m_entry();
                    s_current = nullptr;
                    m_state = State::Finished;
                }
                catch (const boost::context::detail::
                           forced_unwind &)
                {
                    s_current = nullptr;
                    throw;
                }
                catch (...)
                {
                    m_failure =
                        std::current_exception();
                    s_current = nullptr;
                    m_state = State::Failed;
                }
                return std::move(m_executor);
            });
    }

    ~Impl() noexcept
    {
        if (std::this_thread::get_id() != m_owner)
        {
            std::terminate();
        }
        try
        {
            destroy();
        }
        catch (...)
        {
            std::terminate();
        }
    }

    void requireOwner(const char *operation) const
    {
        if (std::this_thread::get_id() != m_owner)
        {
            throw std::logic_error(
                std::string("BoostEeFiber ") +
                operation +
                " must run on its executor thread");
        }
    }

    void resume()
    {
        requireOwner("resume");
        if (s_current != nullptr)
        {
            throw std::logic_error(
                "BoostEeFiber forbids guest-to-guest "
                "switches");
        }
        if ((m_state != State::Created &&
             m_state != State::Suspended) ||
            !m_guest)
        {
            throw std::logic_error(
                "BoostEeFiber is not resumable");
        }

        m_guest = std::move(m_guest).resume();
        if (m_state != State::Suspended &&
            m_state != State::Finished &&
            m_state != State::Failed)
        {
            throw std::logic_error(
                "BoostEeFiber returned without a terminal "
                "or suspended state");
        }
        if (m_state == State::Finished ||
            m_state == State::Failed)
        {
            m_entry = {};
        }
    }

    static void yieldCurrent()
    {
        Impl *const current = s_current;
        if (current == nullptr)
        {
            throw std::logic_error(
                "BoostEeFiber yield requires a running "
                "guest fiber");
        }
        current->requireOwner("yield");
        if (current->m_state != State::Running ||
            !current->m_executor)
        {
            throw std::logic_error(
                "BoostEeFiber has no executor continuation");
        }

        current->m_state = State::Suspended;
        s_current = nullptr;
        current->m_executor =
            std::move(current->m_executor).resume();
        if (s_current != nullptr)
        {
            throw std::logic_error(
                "BoostEeFiber resumed with an active "
                "continuation adapter");
        }
        s_current = current;
        current->m_state = State::Running;
    }

    [[nodiscard]] static bool
    currentStackContainsAddressForTesting(
        const void *address) noexcept
    {
        return s_current != nullptr &&
               s_current->m_stack->containsAddress(
                   address);
    }

    void destroy()
    {
        requireOwner("destroy");
        if (s_current != nullptr)
        {
            throw std::logic_error(
                "BoostEeFiber cannot destroy a running "
                "guest fiber");
        }
        if (m_state == State::Destroyed)
        {
            return;
        }

        // Resetting a suspended public fiber invokes Boost's forced-unwind
        // path on this executor stack.
        m_guest = boost::context::fiber{};
        m_executor = boost::context::fiber{};
        m_entry = {};
        m_state = State::Destroyed;
    }

    State state() const
    {
        requireOwner("state query");
        return m_state;
    }

    bool resumable() const
    {
        requireOwner("resumable query");
        return (m_state == State::Created ||
                m_state == State::Suspended) &&
               static_cast<bool>(m_guest);
    }

    std::exception_ptr failure() const
    {
        requireOwner("failure query");
        return m_failure;
    }

    void sampleStackUsage()
    {
        requireOwner("stack sampling");
        if (s_current != nullptr)
        {
            throw std::logic_error(
                "BoostEeFiber stack sampling requires the "
                "executor");
        }
        m_stack->sampleHighWater();
        m_stack->sampleResident();
    }

    StackUsage stackUsage() const
    {
        requireOwner("stack usage query");
        return m_stack->usage;
    }

private:
    std::thread::id m_owner;
    Entry m_entry;
    State m_state = State::Created;
    std::exception_ptr m_failure;
    std::shared_ptr<BoostEeFiberStackState> m_stack;
    boost::context::fiber m_guest;
    boost::context::fiber m_executor;

    static thread_local Impl *s_current;
};

thread_local BoostEeFiber::Impl *
    BoostEeFiber::Impl::s_current = nullptr;

#else

class BoostEeFiber::Impl
{
};

#endif

BoostEeFiber::BoostEeFiber(Entry entry)
    : BoostEeFiber(std::move(entry), Options{})
{
}

BoostEeFiber::BoostEeFiber(
    Entry entry,
    Options options)
{
#if PS2X_HAS_EE_CPP_FIBER_BACKEND
    m_impl = std::make_unique<Impl>(
        std::move(entry),
        options);
#else
    static_cast<void>(entry);
    static_cast<void>(options);
    throw std::logic_error(
        "Boost.Context fcontext is unavailable");
#endif
}

BoostEeFiber::~BoostEeFiber() noexcept = default;

bool BoostEeFiber::available() noexcept
{
    return PS2X_HAS_EE_CPP_FIBER_BACKEND != 0;
}

bool BoostEeFiber::
    currentStackContainsAddressForTesting(
        const void *address) noexcept
{
#if PS2X_HAS_EE_CPP_FIBER_BACKEND
    return Impl::
        currentStackContainsAddressForTesting(
            address);
#else
    static_cast<void>(address);
    return false;
#endif
}

void BoostEeFiber::resume()
{
#if PS2X_HAS_EE_CPP_FIBER_BACKEND
    m_impl->resume();
#else
    throw std::logic_error(
        "Boost.Context fcontext is unavailable");
#endif
}

void BoostEeFiber::yieldCurrent()
{
#if PS2X_HAS_EE_CPP_FIBER_BACKEND
    Impl::yieldCurrent();
#else
    throw std::logic_error(
        "Boost.Context fcontext is unavailable");
#endif
}

void BoostEeFiber::destroy()
{
#if PS2X_HAS_EE_CPP_FIBER_BACKEND
    m_impl->destroy();
#else
    throw std::logic_error(
        "Boost.Context fcontext is unavailable");
#endif
}

BoostEeFiber::State BoostEeFiber::state() const
{
#if PS2X_HAS_EE_CPP_FIBER_BACKEND
    return m_impl->state();
#else
    throw std::logic_error(
        "Boost.Context fcontext is unavailable");
#endif
}

bool BoostEeFiber::resumable() const
{
#if PS2X_HAS_EE_CPP_FIBER_BACKEND
    return m_impl->resumable();
#else
    return false;
#endif
}

std::exception_ptr BoostEeFiber::failure() const
{
#if PS2X_HAS_EE_CPP_FIBER_BACKEND
    return m_impl->failure();
#else
    return {};
#endif
}

void BoostEeFiber::rethrowFailure() const
{
    const std::exception_ptr stored = failure();
    if (stored)
    {
        std::rethrow_exception(stored);
    }
}

void BoostEeFiber::sampleStackUsage()
{
#if PS2X_HAS_EE_CPP_FIBER_BACKEND
    m_impl->sampleStackUsage();
#else
    throw std::logic_error(
        "Boost.Context fcontext is unavailable");
#endif
}

BoostEeFiber::StackUsage
BoostEeFiber::stackUsage() const
{
#if PS2X_HAS_EE_CPP_FIBER_BACKEND
    return m_impl->stackUsage();
#else
    return {};
#endif
}
