#include "runtime/ps2_vu_executable_memory.h"

#include <cstring>
#include <limits>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace
{
    size_t queryPageSize()
    {
#if defined(_WIN32)
        SYSTEM_INFO information{};
        GetSystemInfo(&information);
        return static_cast<size_t>(information.dwPageSize);
#elif defined(__unix__) || defined(__APPLE__)
        const long value = sysconf(_SC_PAGESIZE);
        return value > 0 ? static_cast<size_t>(value) : 0u;
#else
        return 0u;
#endif
    }
}

VuExecutableMemory::~VuExecutableMemory()
{
    release();
}

VuExecutableMemory::VuExecutableMemory(
    VuExecutableMemory &&other) noexcept
    : m_mapping(std::exchange(other.m_mapping, nullptr)),
      m_capacity(std::exchange(other.m_capacity, 0u)),
      m_usedSize(std::exchange(other.m_usedSize, 0u)),
      m_state(std::exchange(
          other.m_state, VuExecutableMemoryState::Empty))
{
}

VuExecutableMemory &VuExecutableMemory::operator=(
    VuExecutableMemory &&other) noexcept
{
    if (this == &other)
        return *this;

    release();
    m_mapping = std::exchange(other.m_mapping, nullptr);
    m_capacity = std::exchange(other.m_capacity, 0u);
    m_usedSize = std::exchange(other.m_usedSize, 0u);
    m_state = std::exchange(
        other.m_state, VuExecutableMemoryState::Empty);
    return *this;
}

bool VuExecutableMemory::supported()
{
#if defined(_WIN32) || defined(__unix__) || defined(__APPLE__)
    return pageSize() != 0u;
#else
    return false;
#endif
}

size_t VuExecutableMemory::pageSize()
{
    static const size_t value = queryPageSize();
    return value;
}

VuExecutableMemory VuExecutableMemory::allocate(
    size_t minimumBytes, std::string *diagnostic)
{
    VuExecutableMemory allocation;
    const size_t page = pageSize();
    if (minimumBytes == 0u)
    {
        fail(
            "executable allocation size must be nonzero",
            diagnostic);
        return allocation;
    }
    if (page == 0u)
    {
        fail(
            "executable memory is unsupported on this host",
            diagnostic);
        return allocation;
    }
    if (minimumBytes >
        std::numeric_limits<size_t>::max() - (page - 1u))
    {
        fail("executable allocation size overflow", diagnostic);
        return allocation;
    }

    const size_t capacity =
        ((minimumBytes + page - 1u) / page) * page;
#if defined(_WIN32)
    void *const mapping = VirtualAlloc(
        nullptr, capacity, MEM_COMMIT | MEM_RESERVE,
        PAGE_READWRITE);
    if (!mapping)
    {
        fail("VirtualAlloc(PAGE_READWRITE) failed", diagnostic);
        return allocation;
    }
#elif defined(__unix__) || defined(__APPLE__)
    void *const mapping = mmap(
        nullptr, capacity, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED)
    {
        fail("mmap(PROT_READ|PROT_WRITE) failed", diagnostic);
        return allocation;
    }
#else
    (void)capacity;
    fail(
        "executable memory is unsupported on this host",
        diagnostic);
    return allocation;
#endif

    allocation.m_mapping = mapping;
    allocation.m_capacity = capacity;
    allocation.m_state = VuExecutableMemoryState::Writable;
    if (diagnostic)
        diagnostic->clear();
    return allocation;
}

bool VuExecutableMemory::write(
    size_t offset, const void *source, size_t sizeBytes,
    std::string *diagnostic)
{
    if (m_state != VuExecutableMemoryState::Writable)
    {
        return fail(
            "executable memory is not writable", diagnostic);
    }
    if (sizeBytes != 0u && !source)
    {
        return fail(
            "native-code source is null", diagnostic);
    }
    if (offset > m_capacity ||
        sizeBytes > m_capacity - offset)
    {
        return fail(
            "native-code write exceeds its allocation",
            diagnostic);
    }
    if (sizeBytes != 0u)
    {
        std::memcpy(
            static_cast<uint8_t *>(m_mapping) + offset,
            source, sizeBytes);
    }
    if (diagnostic)
        diagnostic->clear();
    return true;
}

bool VuExecutableMemory::finalize(
    size_t usedBytes, std::string *diagnostic)
{
    if (m_state != VuExecutableMemoryState::Writable)
    {
        return fail(
            "executable memory is not in its writable state",
            diagnostic);
    }
    if (usedBytes == 0u || usedBytes > m_capacity)
    {
        return fail(
            "native-code size is outside its allocation",
            diagnostic);
    }

#if defined(_WIN32)
    DWORD previousProtection = 0u;
    if (!VirtualProtect(
            m_mapping, m_capacity, PAGE_EXECUTE_READ,
            &previousProtection))
    {
        return fail(
            "VirtualProtect(PAGE_EXECUTE_READ) failed",
            diagnostic);
    }
    if (!FlushInstructionCache(
            GetCurrentProcess(), m_mapping, usedBytes))
    {
        // The mapping is already RX at this point. Discard it rather than
        // leaving an allocation whose logical state says it is writable.
        release();
        return fail("FlushInstructionCache failed", diagnostic);
    }
#elif defined(__unix__) || defined(__APPLE__)
    if (mprotect(
            m_mapping, m_capacity,
            PROT_READ | PROT_EXEC) != 0)
    {
        return fail(
            "mprotect(PROT_READ|PROT_EXEC) failed",
            diagnostic);
    }
#if defined(__GNUC__) || defined(__clang__)
    auto *const begin = static_cast<char *>(m_mapping);
    __builtin___clear_cache(begin, begin + usedBytes);
#endif
#else
    (void)usedBytes;
    return fail(
        "executable memory is unsupported on this host",
        diagnostic);
#endif

    m_usedSize = usedBytes;
    m_state = VuExecutableMemoryState::Executable;
    if (diagnostic)
        diagnostic->clear();
    return true;
}

uint8_t *VuExecutableMemory::writableData()
{
    if (m_state != VuExecutableMemoryState::Writable)
        return nullptr;
    return static_cast<uint8_t *>(m_mapping);
}

const uint8_t *VuExecutableMemory::executableData() const
{
    if (m_state != VuExecutableMemoryState::Executable)
        return nullptr;
    return static_cast<const uint8_t *>(m_mapping);
}

const void *VuExecutableMemory::executableAddress(
    size_t offset) const
{
    if (m_state != VuExecutableMemoryState::Executable ||
        offset >= m_usedSize)
    {
        return nullptr;
    }
    return static_cast<const uint8_t *>(m_mapping) + offset;
}

void VuExecutableMemory::release() noexcept
{
    if (!m_mapping)
        return;

#if defined(_WIN32)
    VirtualFree(m_mapping, 0u, MEM_RELEASE);
#elif defined(__unix__) || defined(__APPLE__)
    munmap(m_mapping, m_capacity);
#endif
    m_mapping = nullptr;
    m_capacity = 0u;
    m_usedSize = 0u;
    m_state = VuExecutableMemoryState::Empty;
}

bool VuExecutableMemory::fail(
    std::string message, std::string *diagnostic)
{
    if (diagnostic)
        *diagnostic = std::move(message);
    return false;
}
