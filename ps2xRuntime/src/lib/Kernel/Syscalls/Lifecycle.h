#pragma once

#include "ps2_syscalls.h"

namespace ps2_syscalls
{
    void notifyRuntimeStop(PS2Runtime *runtime);
    void joinAllGuestHostThreads(PS2Runtime *runtime);
    void detachAllGuestHostThreads(PS2Runtime *runtime);
}
