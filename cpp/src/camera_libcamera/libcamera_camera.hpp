#pragma once

#include "../app_runtime/interfaces.hpp"

namespace libcamera {
class Request;
}  // namespace libcamera

namespace phc {

struct LibcameraConfig {
  int width = 640;
  int height = 480;
  int fps = 30;
};

// In-process libcamera capture. Build-gated behind ENABLE_LIBCAMERA.
class LibcameraCamera final : public ICamera {
 public:
  explicit LibcameraCamera(LibcameraConfig cfg = {});
  ~LibcameraCamera() override;

  bool Start(FrameCallback cb) override;
  void Stop() override;

 private:
  struct Impl;
  void OnRequestCompleted(libcamera::Request* request);
  LibcameraConfig cfg_;
  Impl* impl_ = nullptr;
};

}  // namespace phc
