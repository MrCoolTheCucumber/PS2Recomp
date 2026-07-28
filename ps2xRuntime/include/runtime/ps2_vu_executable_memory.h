#ifndef PS2_VU_EXECUTABLE_MEMORY_H
#define PS2_VU_EXECUTABLE_MEMORY_H

#include <cstddef>
#include <cstdint>
#include <string>

enum class VuExecutableMemoryState : uint8_t
{
    Empty,
    Writable,
    Executable,
};

// One immutable native-code allocation. A mapping is writable while bytes are
// emitted, then transitions directly to read/execute and can never become
// writable again. Recompilation allocates a fresh object.
class VuExecutableMemory
{
public:
    VuExecutableMemory() = default;
    ~VuExecutableMemory();

    VuExecutableMemory(VuExecutableMemory &&other) noexcept;
    VuExecutableMemory &operator=(
        VuExecutableMemory &&other) noexcept;

    VuExecutableMemory(const VuExecutableMemory &) = delete;
    VuExecutableMemory &operator=(
        const VuExecutableMemory &) = delete;

    [[nodiscard]] static bool supported();
    [[nodiscard]] static size_t pageSize();

    [[nodiscard]] static VuExecutableMemory allocate(
        size_t minimumBytes, std::string *diagnostic = nullptr);

    [[nodiscard]] bool write(
        size_t offset, const void *source, size_t sizeBytes,
        std::string *diagnostic = nullptr);
    [[nodiscard]] bool finalize(
        size_t usedBytes, std::string *diagnostic = nullptr);

    [[nodiscard]] uint8_t *writableData();
    [[nodiscard]] const uint8_t *executableData() const;
    [[nodiscard]] const void *executableAddress(
        size_t offset = 0u) const;

    [[nodiscard]] VuExecutableMemoryState state() const
    {
        return m_state;
    }
    [[nodiscard]] size_t capacity() const { return m_capacity; }
    [[nodiscard]] size_t usedSize() const { return m_usedSize; }
    [[nodiscard]] bool empty() const
    {
        return m_state == VuExecutableMemoryState::Empty;
    }

private:
    void release() noexcept;
    static bool fail(
        std::string message, std::string *diagnostic);

    void *m_mapping = nullptr;
    size_t m_capacity = 0u;
    size_t m_usedSize = 0u;
    VuExecutableMemoryState m_state =
        VuExecutableMemoryState::Empty;
};

#endif
