#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ps2x::savestate
{
    inline constexpr uint32_t kContainerVersion = 1u;
    inline constexpr uint64_t kMaximumFileSize = 512ull * 1024ull * 1024ull;
    inline constexpr uint32_t kMaximumChunkCount = 256u;

    enum class ChunkFlags : uint32_t
    {
        None = 0u,
        Required = 1u << 0u,
    };

    [[nodiscard]] constexpr uint32_t makeChunkId(
        char a, char b, char c, char d) noexcept
    {
        return static_cast<uint32_t>(static_cast<uint8_t>(a)) |
               (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8u) |
               (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16u) |
               (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24u);
    }

    [[nodiscard]] std::string chunkIdString(uint32_t id);

    class Writer
    {
    public:
        void u8(uint8_t value);
        void boolean(bool value);
        void u16(uint16_t value);
        void u32(uint32_t value);
        void i32(int32_t value);
        void u64(uint64_t value);
        void i64(int64_t value);
        void f32(float value);
        void f64(double value);
        void bytes(std::span<const uint8_t> value);
        void sizedBytes(std::span<const uint8_t> value);
        void string(std::string_view value);

        [[nodiscard]] const std::vector<uint8_t> &data() const noexcept;
        [[nodiscard]] std::vector<uint8_t> take();

    private:
        std::vector<uint8_t> m_data;
    };

    class Reader
    {
    public:
        explicit Reader(std::span<const uint8_t> input) noexcept;

        [[nodiscard]] bool u8(uint8_t &value) noexcept;
        [[nodiscard]] bool boolean(bool &value) noexcept;
        [[nodiscard]] bool u16(uint16_t &value) noexcept;
        [[nodiscard]] bool u32(uint32_t &value) noexcept;
        [[nodiscard]] bool i32(int32_t &value) noexcept;
        [[nodiscard]] bool u64(uint64_t &value) noexcept;
        [[nodiscard]] bool i64(int64_t &value) noexcept;
        [[nodiscard]] bool f32(float &value) noexcept;
        [[nodiscard]] bool f64(double &value) noexcept;
        [[nodiscard]] bool bytes(
            size_t size, std::span<const uint8_t> &value) noexcept;
        [[nodiscard]] bool sizedBytes(
            std::span<const uint8_t> &value,
            uint64_t maximumSize = kMaximumFileSize) noexcept;
        [[nodiscard]] bool string(
            std::string &value,
            uint64_t maximumSize = 1024u * 1024u);

        [[nodiscard]] size_t offset() const noexcept;
        [[nodiscard]] size_t remaining() const noexcept;
        [[nodiscard]] bool atEnd() const noexcept;

    private:
        std::span<const uint8_t> m_input;
        size_t m_offset = 0u;
    };

    struct Chunk
    {
        uint32_t id = 0u;
        uint32_t version = 0u;
        ChunkFlags flags = ChunkFlags::None;
        std::vector<uint8_t> payload;
    };

    struct Document
    {
        std::vector<Chunk> chunks;

        [[nodiscard]] const Chunk *find(uint32_t id) const noexcept;
        [[nodiscard]] Chunk *find(uint32_t id) noexcept;
    };

    [[nodiscard]] bool encode(
        const Document &document,
        std::vector<uint8_t> &output,
        std::string *error = nullptr);

    [[nodiscard]] bool decode(
        std::span<const uint8_t> input,
        Document &document,
        std::string *error = nullptr);

    [[nodiscard]] bool writeFileAtomically(
        const std::filesystem::path &path,
        const Document &document,
        std::string *error = nullptr);

    [[nodiscard]] bool readFile(
        const std::filesystem::path &path,
        Document &document,
        std::string *error = nullptr);
}
