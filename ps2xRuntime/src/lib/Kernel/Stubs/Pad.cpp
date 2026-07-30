#include "Common.h"
#include "Pad.h"
#include "Helpers/PadRuntimeState.h"

namespace ps2_stubs
{
    namespace
    {
        constexpr uint8_t kPadModeDigital = 0x41;
        constexpr uint8_t kPadModeDualShock = 0x73;
        constexpr int32_t kPadTypeDigital = 4;
        constexpr int32_t kPadTypeDualShock = 7;
        constexpr int32_t kPadStateDisconnected = 0;
        constexpr int32_t kPadStateExecCmd = 5;
        constexpr int32_t kPadStateStable = 6;

        constexpr uint16_t kPadBtnSelect = 1u << 0;
        constexpr uint16_t kPadBtnL3 = 1u << 1;
        constexpr uint16_t kPadBtnR3 = 1u << 2;
        constexpr uint16_t kPadBtnStart = 1u << 3;
        constexpr uint16_t kPadBtnUp = 1u << 4;
        constexpr uint16_t kPadBtnRight = 1u << 5;
        constexpr uint16_t kPadBtnDown = 1u << 6;
        constexpr uint16_t kPadBtnLeft = 1u << 7;
        constexpr uint16_t kPadBtnL2 = 1u << 8;
        constexpr uint16_t kPadBtnR2 = 1u << 9;
        constexpr uint16_t kPadBtnL1 = 1u << 10;
        constexpr uint16_t kPadBtnR1 = 1u << 11;
        constexpr uint16_t kPadBtnTriangle = 1u << 12;
        constexpr uint16_t kPadBtnCircle = 1u << 13;
        constexpr uint16_t kPadBtnCross = 1u << 14;
        constexpr uint16_t kPadBtnSquare = 1u << 15;

        PadRuntimeState *getPadRuntimeState(PS2Runtime *runtime)
        {
            return runtime ? &runtime->padRuntimeState() : nullptr;
        }

        uint8_t axisToByte(float axis)
        {
            axis = std::clamp(axis, -1.0f, 1.0f);
            const float mapped = (axis + 1.0f) * 127.5f;
            return static_cast<uint8_t>(std::lround(mapped));
        }

        void setButton(PadInputState &state, uint16_t mask, bool pressed)
        {
            if (pressed)
            {
                state.buttons = static_cast<uint16_t>(state.buttons & ~mask);
            }
        }

        int findFirstGamepad()
        {
            for (int i = 0; i < 4; ++i)
            {
                if (IsGamepadAvailable(i))
                {
                    return i;
                }
            }
            return -1;
        }

        void applyGamepadState(PadInputState &state)
        {
            if (!IsWindowReady())
            {
                return;
            }

            const int gamepad = findFirstGamepad();
            if (gamepad < 0)
            {
                return;
            }

            // Raylib mapping (PS2 -> raylib buttons/axes):
            // D-Pad -> LEFT_FACE_*, Cross/Circle/Square/Triangle -> RIGHT_FACE_*
            // L1/R1 -> TRIGGER_1, L2/R2 -> TRIGGER_2, L3/R3 -> THUMB
            // Select/Start -> MIDDLE_LEFT/MIDDLE_RIGHT
            state.lx = axisToByte(GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_LEFT_X));
            state.ly = axisToByte(GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_LEFT_Y));
            state.rx = axisToByte(GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_RIGHT_X));
            state.ry = axisToByte(GetGamepadAxisMovement(gamepad, GAMEPAD_AXIS_RIGHT_Y));

            setButton(state, kPadBtnUp, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_LEFT_FACE_UP));
            setButton(state, kPadBtnDown, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_LEFT_FACE_DOWN));
            setButton(state, kPadBtnLeft, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_LEFT_FACE_LEFT));
            setButton(state, kPadBtnRight, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_LEFT_FACE_RIGHT));

            setButton(state, kPadBtnCross, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_RIGHT_FACE_DOWN));
            setButton(state, kPadBtnCircle, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT));
            setButton(state, kPadBtnSquare, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_RIGHT_FACE_LEFT));
            setButton(state, kPadBtnTriangle, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_RIGHT_FACE_UP));

            setButton(state, kPadBtnL1, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_1));
            setButton(state, kPadBtnR1, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_1));
            setButton(state, kPadBtnL2, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_2));
            setButton(state, kPadBtnR2, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_2));

            setButton(state, kPadBtnL3, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_LEFT_THUMB));
            setButton(state, kPadBtnR3, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_RIGHT_THUMB));

            setButton(state, kPadBtnSelect, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_MIDDLE_LEFT));
            setButton(state, kPadBtnStart, IsGamepadButtonDown(gamepad, GAMEPAD_BUTTON_MIDDLE_RIGHT));
        }

        void applyKeyboardState(PadInputState &state, bool allowAnalog)
        {
            if (!IsWindowReady())
            {
                return;
            }

            // Keyboard mapping (PS2 -> keys):
            // D-Pad: arrows, Square/Cross/Circle/Triangle: Z/X/C/V
            // L1/R1: Q/E, L2/R2: 1/3, Start/Select: Enter/RightShift
            // L3/R3: LeftCtrl/RightCtrl, Analog left: WASD
            setButton(state, kPadBtnUp, IsKeyDown(KEY_UP));
            setButton(state, kPadBtnDown, IsKeyDown(KEY_DOWN));
            setButton(state, kPadBtnLeft, IsKeyDown(KEY_LEFT));
            setButton(state, kPadBtnRight, IsKeyDown(KEY_RIGHT));

            setButton(state, kPadBtnSquare, IsKeyDown(KEY_Z));
            setButton(state, kPadBtnCross, IsKeyDown(KEY_X));
            setButton(state, kPadBtnCircle, IsKeyDown(KEY_C));
            setButton(state, kPadBtnTriangle, IsKeyDown(KEY_V));

            setButton(state, kPadBtnL1, IsKeyDown(KEY_Q));
            setButton(state, kPadBtnR1, IsKeyDown(KEY_E));
            setButton(state, kPadBtnL2, IsKeyDown(KEY_ONE));
            setButton(state, kPadBtnR2, IsKeyDown(KEY_THREE));

            setButton(state, kPadBtnStart, IsKeyDown(KEY_ENTER));
            setButton(state, kPadBtnSelect, IsKeyDown(KEY_RIGHT_SHIFT));
            setButton(state, kPadBtnL3, IsKeyDown(KEY_LEFT_CONTROL));
            setButton(state, kPadBtnR3, IsKeyDown(KEY_RIGHT_CONTROL));

            if (!allowAnalog)
            {
                return;
            }

            float ax = 0.0f;
            float ay = 0.0f;
            if (IsKeyDown(KEY_D))
                ax += 1.0f;
            if (IsKeyDown(KEY_A))
                ax -= 1.0f;
            if (IsKeyDown(KEY_S))
                ay += 1.0f;
            if (IsKeyDown(KEY_W))
                ay -= 1.0f;

            if (ax != 0.0f || ay != 0.0f)
            {
                state.lx = axisToByte(ax);
                state.ly = axisToByte(ay);
            }
        }

        PadPortState *lookupPadPortStateLocked(
            PadRuntimeState &state, int port, int slot)
        {
            if (port < 0 ||
                port >= static_cast<int>(kPadRuntimePortCount))
            {
                return nullptr;
            }
            if (slot < 0 ||
                slot >= static_cast<int>(kPadRuntimeSlotCount))
            {
                return nullptr;
            }
            return &state.ports[port];
        }

        void queueExecCmdStateLocked(PadPortState &portState)
        {
            portState.transientState = static_cast<uint32_t>(kPadStateExecCmd);
        }

        uint8_t pressureValue(const PadInputState &state, const PadPortState &portState, uint16_t mask)
        {
            if (!portState.pressureEnabled)
            {
                return 0u;
            }
            if ((portState.buttonMask & mask) == 0u)
            {
                return 0u;
            }
            return ((state.buttons & mask) == 0u) ? 0xFFu : 0u;
        }

        void fillPadStatus(uint8_t *data, const PadInputState &state, const PadPortState &portState)
        {
            std::memset(data, 0, 32);
            data[1] = portState.analogMode ? kPadModeDualShock : kPadModeDigital;
            data[2] = static_cast<uint8_t>(state.buttons & 0xFFu);
            data[3] = static_cast<uint8_t>((state.buttons >> 8) & 0xFFu);
            data[4] = state.rx;
            data[5] = state.ry;
            data[6] = state.lx;
            data[7] = state.ly;
            data[8] = pressureValue(state, portState, kPadBtnRight);
            data[9] = pressureValue(state, portState, kPadBtnLeft);
            data[10] = pressureValue(state, portState, kPadBtnUp);
            data[11] = pressureValue(state, portState, kPadBtnDown);
            data[12] = pressureValue(state, portState, kPadBtnTriangle);
            data[13] = pressureValue(state, portState, kPadBtnCircle);
            data[14] = pressureValue(state, portState, kPadBtnCross);
            data[15] = pressureValue(state, portState, kPadBtnSquare);
            data[16] = pressureValue(state, portState, kPadBtnL1);
            data[17] = pressureValue(state, portState, kPadBtnL2);
            data[18] = pressureValue(state, portState, kPadBtnR1);
            data[19] = pressureValue(state, portState, kPadBtnR2);
        }

        void fillPad2Status(uint8_t *data, const PadInputState &state)
        {
            constexpr size_t kPad2StatusSize = 18u;
            std::memset(data, 0, kPad2StatusSize);
            data[0] = static_cast<uint8_t>(state.buttons & 0xFFu);
            data[1] = static_cast<uint8_t>((state.buttons >> 8) & 0xFFu);
            data[2] = state.rx;
            data[3] = state.ry;
            data[4] = state.lx;
            data[5] = state.ly;

            const auto pressed = [&state](uint16_t mask)
            {
                return static_cast<uint8_t>((state.buttons & mask) == 0u ? 0xFFu : 0u);
            };
            data[6] = pressed(kPadBtnRight);
            data[7] = pressed(kPadBtnLeft);
            data[8] = pressed(kPadBtnUp);
            data[9] = pressed(kPadBtnDown);
            data[10] = pressed(kPadBtnTriangle);
            data[11] = pressed(kPadBtnCircle);
            data[12] = pressed(kPadBtnCross);
            data[13] = pressed(kPadBtnSquare);
            data[14] = pressed(kPadBtnL1);
            data[15] = pressed(kPadBtnL2);
            data[16] = pressed(kPadBtnR1);
            data[17] = pressed(kPadBtnR2);
        }

        void samplePadInput(
            PadRuntimeState *runtimeState,
            int port,
            int slot,
            PS2Runtime *runtime,
            bool allowAnalog,
            PadInputState &state,
            bool &usedOverride,
            bool &usedBackend)
        {
            usedOverride = false;
            usedBackend = false;
            if (runtimeState)
            {
                std::lock_guard<std::mutex> lock(
                    runtimeState->mutex);
                if (runtimeState->overrideEnabled)
                {
                    state = runtimeState->overrideInput;
                    usedOverride = true;
                }
            }

            if (usedOverride)
            {
                return;
            }

            uint8_t backendData[32]{};
            if (runtime && runtime->padBackend().readState(port, slot, backendData, sizeof(backendData)))
            {
                state.buttons = static_cast<uint16_t>(backendData[2] | (backendData[3] << 8));
                state.rx = backendData[4];
                state.ry = backendData[5];
                state.lx = backendData[6];
                state.ly = backendData[7];
                usedBackend = true;
                return;
            }

            applyGamepadState(state);
            applyKeyboardState(state, allowAnalog);
        }

        bool readPadPortData(
            PadRuntimeState &runtimeState,
            int port,
            int slot,
            PS2Runtime *runtime,
            uint8_t *outData,
            uint32_t dataAddr)
        {
            if (!outData)
            {
                return false;
            }

            PadPortState portState;
            {
                std::lock_guard<std::mutex> lock(
                    runtimeState.mutex);
                const PadPortState *sharedPortState =
                    lookupPadPortStateLocked(
                        runtimeState, port, slot);
                if (!sharedPortState || !sharedPortState->open)
                {
                    return false;
                }
                portState = *sharedPortState;
            }

            PadInputState state;
            bool usedOverride = false;
            bool usedBackend = false;
            samplePadInput(
                &runtimeState,
                port,
                slot,
                runtime,
                portState.analogMode,
                state,
                usedOverride,
                usedBackend);

            fillPadStatus(outData, state, portState);

            {
                std::lock_guard<std::mutex> lock(
                    runtimeState.mutex);
                if (PadPortState *sharedPortState =
                        lookupPadPortStateLocked(
                            runtimeState, port, slot))
                {
                    sharedPortState->lastInput = state;
                    std::memcpy(
                        sharedPortState->lastData.data(),
                        outData,
                        sharedPortState->lastData.size());
                    sharedPortState->lastUsedOverride = usedOverride;
                    sharedPortState->lastUsedBackend = usedBackend;
                    sharedPortState->lastReadOk = true;
                    sharedPortState->lastReadDataAddr = dataAddr;
                    ++sharedPortState->readCount;
                }
            }

            return true;
        }
    }

    void resetPadState(PS2Runtime *runtime)
    {
        if (PadRuntimeState *state =
                getPadRuntimeState(runtime))
        {
            state->reset();
        }
    }

    void PadSyncCallback(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void scePadEnd(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, 0);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->resetGuestStateLocked();
        }
        setReturnS32(ctx, 1);
    }

    void scePadEnterPressMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, 0);
            return;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        PadPortState *portState = lookupPadPortStateLocked(
            *state,
            static_cast<int>(getRegU32(ctx, 4)),
            static_cast<int>(getRegU32(ctx, 5)));
        if (!portState || !portState->open)
        {
            setReturnS32(ctx, 0);
            return;
        }

        portState->pressureEnabled = true;
        portState->reqState = 0u;
        queueExecCmdStateLocked(*portState);
        setReturnS32(ctx, 1);
    }

    void scePadExitPressMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, 0);
            return;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        PadPortState *portState = lookupPadPortStateLocked(
            *state,
            static_cast<int>(getRegU32(ctx, 4)),
            static_cast<int>(getRegU32(ctx, 5)));
        if (!portState || !portState->open)
        {
            setReturnS32(ctx, 0);
            return;
        }

        portState->pressureEnabled = false;
        portState->reqState = 0u;
        queueExecCmdStateLocked(*portState);
        setReturnS32(ctx, 1);
    }

    void scePadGetButtonMask(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, 0xFFFF);
            return;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        const PadPortState *portState = lookupPadPortStateLocked(
            *state,
            static_cast<int>(getRegU32(ctx, 4)),
            static_cast<int>(getRegU32(ctx, 5)));
        const uint16_t mask = portState ? portState->buttonMask : 0xFFFFu;
        setReturnS32(ctx, static_cast<int32_t>(mask));
    }

    void scePadGetDmaStr(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            setReturnU32(ctx, getRegU32(ctx, 6));
            return;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        const PadPortState *portState = lookupPadPortStateLocked(
            *state,
            static_cast<int>(getRegU32(ctx, 4)),
            static_cast<int>(getRegU32(ctx, 5)));
        const uint32_t dmaAddr = portState ? portState->dmaAddr : getRegU32(ctx, 6);
        setReturnU32(ctx, dmaAddr);
    }

    void scePadGetFrameCount(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        PadRuntimeState *state = getPadRuntimeState(runtime);
        setReturnU32(
            ctx,
            state ? state->frameCount.fetch_add(
                        1u, std::memory_order_relaxed)
                  : 0u);
    }

    void scePadGetModVersion(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        // Arbitrary non-zero module version.
        setReturnS32(ctx, 0x0200);
    }

    void scePadGetPortMax(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        setReturnS32(ctx, 2);
    }

    void scePadGetReqState(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, 0);
            return;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        const PadPortState *portState = lookupPadPortStateLocked(
            *state,
            static_cast<int>(getRegU32(ctx, 4)),
            static_cast<int>(getRegU32(ctx, 5)));
        setReturnS32(ctx, static_cast<int32_t>(portState ? portState->reqState : 0u));
    }

    void scePadGetSlotMax(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        // Most games use one slot unless multitap is active.
        setReturnS32(ctx, 1);
    }

    void scePadGetState(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        PadRuntimeState *runtimeState =
            getPadRuntimeState(runtime);
        if (!runtimeState)
        {
            setReturnS32(ctx, kPadStateDisconnected);
            return;
        }
        std::lock_guard<std::mutex> lock(
            runtimeState->mutex);
        PadPortState *portState = lookupPadPortStateLocked(
            *runtimeState,
            static_cast<int>(getRegU32(ctx, 4)),
            static_cast<int>(getRegU32(ctx, 5)));
        int32_t state = kPadStateDisconnected;
        if (portState && portState->open)
        {
            if (portState->transientState != 0u)
            {
                state = static_cast<int32_t>(portState->transientState);
                portState->transientState = 0u;
            }
            else
            {
                state = kPadStateStable;
            }
        }
        setReturnS32(ctx, state);
    }

    void scePadInfoAct(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        const int32_t act = static_cast<int32_t>(getRegU32(ctx, 6));
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, 0);
            return;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        const PadPortState *portState = lookupPadPortStateLocked(
            *state,
            static_cast<int>(getRegU32(ctx, 4)),
            static_cast<int>(getRegU32(ctx, 5)));
        if (!portState || !portState->open)
        {
            setReturnS32(ctx, 0);
            return;
        }

        if (act < 0)
        {
            setReturnS32(ctx, 2); // small + large motors
            return;
        }
        setReturnS32(ctx, (act < 2) ? 1 : 0);
    }

    void scePadInfoComb(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        // No combined modes reported.
        setReturnS32(ctx, 0);
    }

    void scePadInfoMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        const int32_t infoMode = static_cast<int32_t>(getRegU32(ctx, 6)); // a2
        const int32_t index = static_cast<int32_t>(getRegU32(ctx, 7));    // a3
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, 0);
            return;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        const PadPortState *portState = lookupPadPortStateLocked(
            *state,
            static_cast<int>(getRegU32(ctx, 4)),
            static_cast<int>(getRegU32(ctx, 5)));
        if (!portState || !portState->open)
        {
            setReturnS32(ctx, 0);
            return;
        }

        const int32_t currentId = portState->analogMode ? kPadTypeDualShock : kPadTypeDigital;
        switch (infoMode)
        {
        case 1: // PAD_MODECURID
            setReturnS32(ctx, currentId);
            return;
        case 2: // PAD_MODECUREXID
            setReturnS32(ctx, currentId);
            return;
        case 3: // PAD_MODECUROFFS
            setReturnS32(ctx, 0);
            return;
        case 4: // PAD_MODETABLE
            if (index == -1)
            {
                setReturnS32(ctx, 1); // one available mode
            }
            else if (index == 0)
            {
                setReturnS32(ctx, currentId);
            }
            else
            {
                setReturnS32(ctx, 0);
            }
            return;
        default:
            setReturnS32(ctx, 0);
            return;
        }
    }

    void scePadInfoPressMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, 0);
            return;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        const PadPortState *portState = lookupPadPortStateLocked(
            *state,
            static_cast<int>(getRegU32(ctx, 4)),
            static_cast<int>(getRegU32(ctx, 5)));
        setReturnS32(ctx, (portState && portState->open) ? 1 : 0);
    }

    void scePadInit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, 0);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->resetGuestStateLocked();
        }
        setReturnS32(ctx, 1);
    }

    void scePadInit2(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        scePadInit(rdram, ctx, runtime);
    }

    void scePad2GetButtonProfile(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        constexpr uint32_t kStandardButtonProfile = 0xFFFFFFFFu;
        const int socket = static_cast<int>(getRegU32(ctx, 4));
        const uint32_t profileAddr = getRegU32(ctx, 5);
        uint8_t *profile = (profileAddr != 0u) ? getMemPtr(rdram, profileAddr) : nullptr;
        if (socket < 0 || !profile)
        {
            setReturnS32(ctx, -1);
            return;
        }

        (void)runtime;
        ps2TraceGuestRangeWrite(rdram, profileAddr, sizeof(kStandardButtonProfile),
                                "scePad2GetButtonProfile", ctx);
        std::memcpy(profile, &kStandardButtonProfile, sizeof(kStandardButtonProfile));
        setReturnS32(ctx, static_cast<int32_t>(sizeof(kStandardButtonProfile)));
    }

    void scePad2GetState(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        const int socket = static_cast<int>(getRegU32(ctx, 4));
        setReturnS32(ctx, socket >= 0 ? 1 : 0);
    }

    void scePad2Read(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        constexpr uint32_t kPad2StatusSize = 18u;
        const int socket = static_cast<int>(getRegU32(ctx, 4));
        const uint32_t dataAddr = getRegU32(ctx, 5);
        uint8_t *data = (dataAddr != 0u) ? getMemPtr(rdram, dataAddr) : nullptr;
        if (socket < 0 || !data)
        {
            setReturnS32(ctx, -1);
            return;
        }

        PadInputState state;
        bool usedOverride = false;
        bool usedBackend = false;
        samplePadInput(
            getPadRuntimeState(runtime),
            socket,
            0,
            runtime,
            true,
            state,
            usedOverride,
            usedBackend);
        (void)usedOverride;
        (void)usedBackend;

        ps2TraceGuestRangeWrite(rdram, dataAddr, kPad2StatusSize, "scePad2Read", ctx);
        fillPad2Status(data, state);
        setReturnS32(ctx, static_cast<int32_t>(kPad2StatusSize));
    }

    void sceVibGetProfile(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        constexpr uint32_t kDualShockActuatorProfile = 3u;
        const int socket = static_cast<int>(getRegU32(ctx, 4));
        const uint32_t profileAddr = getRegU32(ctx, 5);
        uint8_t *profile = (profileAddr != 0u) ? getMemPtr(rdram, profileAddr) : nullptr;
        if (socket < 0 || !profile)
        {
            setReturnS32(ctx, -1);
            return;
        }

        (void)runtime;
        ps2TraceGuestRangeWrite(rdram, profileAddr, sizeof(kDualShockActuatorProfile),
                                "sceVibGetProfile", ctx);
        std::memcpy(profile, &kDualShockActuatorProfile, sizeof(kDualShockActuatorProfile));
        setReturnS32(ctx, static_cast<int32_t>(sizeof(kDualShockActuatorProfile)));
    }

    void scePadPortClose(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, 0);
            return;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        PadPortState *portState = lookupPadPortStateLocked(
            *state,
            static_cast<int>(getRegU32(ctx, 4)),
            static_cast<int>(getRegU32(ctx, 5)));
        if (!portState)
        {
            setReturnS32(ctx, 0);
            return;
        }

        portState->open = false;
        portState->pressureEnabled = false;
        portState->reqState = 0u;
        portState->transientState = 0u;
        setReturnS32(ctx, 1);
    }

    void scePadPortOpen(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t dmaAddr = getRegU32(ctx, 6);
        uint8_t *dmaStr = getMemPtr(rdram, dmaAddr);
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, 0);
            return;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        PadPortState *portState = lookupPadPortStateLocked(
            *state,
            static_cast<int>(getRegU32(ctx, 4)),
            static_cast<int>(getRegU32(ctx, 5)));
        if (!portState || (dmaAddr != 0u && !dmaStr))
        {
            setReturnS32(ctx, 0);
            return;
        }

        portState->open = true;
        portState->analogMode = false;     // real pads open DIGITAL
        portState->pressureEnabled = false;
        portState->buttonMask = 0xFFFFu;
        portState->dmaAddr = dmaAddr;
        portState->reqState = 0u;
        portState->transientState = 0u;
        if (dmaStr)
        {
            ps2TraceGuestRangeWrite(rdram, dmaAddr, 32u, "scePadPortOpen", ctx);
            std::memset(dmaStr, 0, 32);
        }
        setReturnS32(ctx, 1);
    }

    void scePadRead(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const int port = static_cast<int>(getRegU32(ctx, 4));
        const int slot = static_cast<int>(getRegU32(ctx, 5));
        const uint32_t dataAddr = getRegU32(ctx, 6);
        uint8_t *data = getMemPtr(rdram, dataAddr);
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!data || !state)
        {
            setReturnS32(ctx, 0);
            return;
        }

        ps2TraceGuestRangeWrite(rdram, dataAddr, 32u, "scePadRead", ctx);
        if (!readPadPortData(
                *state,
                port,
                slot,
                runtime,
                data,
                dataAddr))
        {
            setReturnS32(ctx, 0);
            return;
        }

        PS2_IF_AGRESSIVE_LOGS({
            const int gamepad = findFirstGamepad();
            const bool gamepadStartPressed =
                (gamepad >= 0) &&
                IsGamepadButtonDown(
                    gamepad,
                    GAMEPAD_BUTTON_MIDDLE_RIGHT);
            const bool startPressed =
                (data[2] != 0xFFu ||
                 data[3] != 0xFFu ||
                 IsKeyDown(KEY_ENTER) ||
                 gamepadStartPressed);
            if (startPressed)
            {
                bool shouldLog = false;
                {
                    std::lock_guard<std::mutex> lock(
                        state->mutex);
                    if (state->readLogCount < 48)
                    {
                        ++state->readLogCount;
                        shouldLog = true;
                    }
                }
                if (shouldLog)
                {
                    const uint32_t guestButtons =
                        (static_cast<uint32_t>(static_cast<uint8_t>(data[2] ^ 0xFFu)) << 8) |
                        static_cast<uint32_t>(static_cast<uint8_t>(data[3] ^ 0xFFu));
                    std::printf("[padread] port=%d slot=%d data2=0x%02x data3=0x%02x guestButtons=0x%04x enter=%d gamepadStart=%d\n",
                                port, slot, data[2], data[3], guestButtons,
                                IsKeyDown(KEY_ENTER) ? 1 : 0, gamepadStartPressed ? 1 : 0);
                }
            }
        });

        setReturnS32(ctx, 1);
    }

    void scePadReqIntToStr(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t state = getRegU32(ctx, 4);
        const uint32_t strAddr = getRegU32(ctx, 5);
        char *buf = reinterpret_cast<char *>(getMemPtr(rdram, strAddr));
        if (!buf)
        {
            setReturnS32(ctx, -1);
            return;
        }

        const char *text = (state == 0) ? "COMPLETE" : "BUSY";
        std::strncpy(buf, text, 31);
        buf[31] = '\0';
        setReturnS32(ctx, 0);
    }

    void scePadSetActAlign(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        setReturnS32(ctx, 1);
    }

    void scePadSetActDirect(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        setReturnS32(ctx, 1);
    }

    void scePadSetButtonInfo(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, 0);
            return;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        PadPortState *portState = lookupPadPortStateLocked(
            *state,
            static_cast<int>(getRegU32(ctx, 4)),
            static_cast<int>(getRegU32(ctx, 5)));
        if (portState && portState->open)
        {
            portState->buttonMask = static_cast<uint16_t>(getRegU32(ctx, 6));
            portState->reqState = 0u;
            queueExecCmdStateLocked(*portState);
        }
        setReturnS32(ctx, 1);
    }

    void scePadSetMainMode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, 0);
            return;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        PadPortState *portState = lookupPadPortStateLocked(
            *state,
            static_cast<int>(getRegU32(ctx, 4)),
            static_cast<int>(getRegU32(ctx, 5)));
        if (!portState || !portState->open)
        {
            setReturnS32(ctx, 0);
            return;
        }

        portState->analogMode = (getRegU32(ctx, 6) != 0u);
        portState->reqState = 0u;
        queueExecCmdStateLocked(*portState);
        setReturnS32(ctx, 1);
    }

    void scePadSetReqState(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, 0);
            return;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        PadPortState *portState = lookupPadPortStateLocked(
            *state,
            static_cast<int>(getRegU32(ctx, 4)),
            static_cast<int>(getRegU32(ctx, 5)));
        if (portState && portState->open)
        {
            portState->reqState = static_cast<uint32_t>(getRegU32(ctx, 6) ? 1u : 0u);
            queueExecCmdStateLocked(*portState);
        }
        setReturnS32(ctx, 1);
    }

    void scePadSetVrefParam(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        setReturnS32(ctx, 1);
    }

    void scePadSetWarningLevel(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        setReturnS32(ctx, 0);
    }

    void scePadStateIntToStr(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t state = getRegU32(ctx, 4);
        const uint32_t strAddr = getRegU32(ctx, 5);
        char *buf = reinterpret_cast<char *>(getMemPtr(rdram, strAddr));
        if (!buf)
        {
            setReturnS32(ctx, -1);
            return;
        }

        const char *text = "UNKNOWN";
        if (state == 6)
        {
            text = "STABLE";
        }
        else if (state == 1)
        {
            text = "FINDPAD";
        }
        else if (state == 5)
        {
            text = "EXECCMD";
        }
        else if (state == 0)
        {
            text = "DISCONNECTED";
        }

        std::strncpy(buf, text, 31);
        buf[31] = '\0';
        setReturnS32(ctx, 0);
    }

    PadDebugSnapshot getPadDebugSnapshot(
        PS2Runtime *runtime)
    {
        PadDebugSnapshot snapshot{};
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            return snapshot;
        }

        std::lock_guard<std::mutex> lock(state->mutex);
        snapshot.overrideEnabled = state->overrideEnabled;
        snapshot.overrideButtons =
            state->overrideInput.buttons;
        snapshot.overrideRx = state->overrideInput.rx;
        snapshot.overrideRy = state->overrideInput.ry;
        snapshot.overrideLx = state->overrideInput.lx;
        snapshot.overrideLy = state->overrideInput.ly;
        snapshot.readLogCount = state->readLogCount;
        for (size_t port = 0;
             port < kPadDebugPortCount;
             ++port)
        {
            for (size_t slot = 0;
                 slot < kPadDebugSlotCount;
                 ++slot)
            {
                const PadPortState &src =
                    state->ports[port];
                PadDebugPortSnapshot &dst =
                    snapshot.ports[port][slot];
                dst.open = src.open;
                dst.analogMode = src.analogMode;
                dst.pressureEnabled =
                    src.pressureEnabled;
                dst.lastUsedOverride =
                    src.lastUsedOverride;
                dst.lastUsedBackend =
                    src.lastUsedBackend;
                dst.lastReadOk = src.lastReadOk;
                dst.buttonMask = src.buttonMask;
                dst.lastButtons =
                    src.lastInput.buttons;
                dst.dmaAddr = src.dmaAddr;
                dst.reqState = src.reqState;
                dst.readCount = src.readCount;
                dst.lastReadDataAddr =
                    src.lastReadDataAddr;
                dst.rx = src.lastInput.rx;
                dst.ry = src.lastInput.ry;
                dst.lx = src.lastInput.lx;
                dst.ly = src.lastInput.ly;
                std::memcpy(
                    dst.lastData,
                    src.lastData.data(),
                    sizeof(dst.lastData));
            }
        }
        return snapshot;
    }

    void setPadOverrideState(
        PS2Runtime *runtime,
        uint16_t buttons,
        uint8_t lx,
        uint8_t ly,
        uint8_t rx,
        uint8_t ry)
    {
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(state->mutex);
        state->overrideEnabled = true;
        state->overrideInput.buttons = buttons;
        state->overrideInput.lx = lx;
        state->overrideInput.ly = ly;
        state->overrideInput.rx = rx;
        state->overrideInput.ry = ry;
    }

    void clearPadOverrideState(PS2Runtime *runtime)
    {
        PadRuntimeState *state = getPadRuntimeState(runtime);
        if (!state)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(state->mutex);
        state->overrideEnabled = false;
        state->overrideInput = {};
    }
}
