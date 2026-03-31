#pragma once

#include "../core/frame.hpp"

#include <functional>

namespace phc {

class ICamera {
 public:
  virtual ~ICamera() = default;
  using FrameCallback = std::function<void(const Frame&)>;
  virtual bool Start(FrameCallback cb) = 0;
  virtual void Stop() = 0;
};

class IDisplay {
 public:
  virtual ~IDisplay() = default;
  virtual bool Init(int width, int height) = 0;
  virtual bool Present(const Frame& rgb888) = 0;
};

}  // namespace phc

