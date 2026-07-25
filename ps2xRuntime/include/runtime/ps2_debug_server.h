#ifndef PS2_DEBUG_SERVER_H
#define PS2_DEBUG_SERVER_H

#include <memory>

class PS2Runtime;

// Local, opt-in JSON-RPC debugger transport used by tools/ps2dbg. The server is
// intentionally independent of any game and is only built on supported desktop
// hosts.
class PS2DebugServer
{
public:
    explicit PS2DebugServer(PS2Runtime &runtime);
    ~PS2DebugServer();

    PS2DebugServer(const PS2DebugServer &) = delete;
    PS2DebugServer &operator=(const PS2DebugServer &) = delete;

    bool start();
    void stop();
    bool isRunning() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif // PS2_DEBUG_SERVER_H
