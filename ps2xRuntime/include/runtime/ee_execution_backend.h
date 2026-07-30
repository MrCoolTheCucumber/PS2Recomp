#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string_view>

enum class EeExecutionBackendKind
{
    LegacyHostThread,
};

// Owns the native continuation mechanism used to execute EE guest threads.
// Phase 1 retains the existing one-host-thread-per-guest continuation; guest
// state and scheduling decisions remain owned by the runtime/kernel model.
class IEeExecutionBackend
{
public:
    using ThreadEntry = std::function<void()>;

    virtual ~IEeExecutionBackend() = default;

    [[nodiscard]] virtual EeExecutionBackendKind kind() const noexcept = 0;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    // Legacy continuations begin immediately. A later single-executor
    // backend may split creation from resume without changing guest state.
    virtual void create(int threadId, ThreadEntry entry) = 0;
    virtual void destroy(int threadId) = 0;
    virtual void joinAll() = 0;
    virtual void detachAll() = 0;

    [[nodiscard]] virtual size_t managedThreadCount() const = 0;
};

std::unique_ptr<IEeExecutionBackend>
createEeExecutionBackend(EeExecutionBackendKind kind);
