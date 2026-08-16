#pragma once

#include <memory>

class PS2Runtime;
class PS2PerformanceHudImpl;

class PS2PerformanceHud {
public:
  explicit PS2PerformanceHud(PS2Runtime &runtime);
  ~PS2PerformanceHud();

  PS2PerformanceHud(const PS2PerformanceHud &) = delete;
  PS2PerformanceHud &operator=(const PS2PerformanceHud &) = delete;

  void toggle();
  void update();
  void stop();

private:
  std::unique_ptr<PS2PerformanceHudImpl> m_impl;
};
