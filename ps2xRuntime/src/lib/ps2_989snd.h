#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ps2_989snd
{
    struct DebugSnapshot
    {
        size_t bankCount = 0u;
        size_t soundCount = 0u;
        size_t decodedSampleCount = 0u;
        size_t activeHandlerCount = 0u;
        size_t activeVoiceCount = 0u;
        uint64_t renderedFrames = 0u;
        uint64_t nonzeroFrames = 0u;
        uint64_t rejectedBanks = 0u;
        uint64_t rejectedCommands = 0u;
        uint64_t unsupportedGrains = 0u;
    };

    class Engine
    {
    public:
        Engine();
        ~Engine();

        Engine(const Engine &) = delete;
        Engine &operator=(const Engine &) = delete;

        bool loadBank(uint32_t bankHandle,
                      const uint8_t *container,
                      size_t containerSize);
        void unloadBank(uint32_t bankHandle);

        bool playSound(uint32_t bankHandle,
                       uint32_t soundId,
                       uint32_t soundHandle,
                       int32_t volume,
                       int32_t pan,
                       int32_t pitchModifier,
                       int32_t pitchBend);
        void stopSound(uint32_t soundHandle);
        void stopAllSounds();
        void setMasterVolume(uint32_t group, int32_t volume);
        void setSoundParameters(uint32_t soundHandle,
                                uint32_t mask,
                                int32_t volume,
                                int32_t pan,
                                int32_t pitchModifier,
                                int32_t pitchBend);

        [[nodiscard]] bool soundActive(uint32_t soundHandle) const;
        void render(int16_t *stereoOutput, size_t frameCount);
        [[nodiscard]] DebugSnapshot debugSnapshot() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };
}
