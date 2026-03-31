#pragma once

#include "../app_runtime/interfaces.hpp"

namespace phc {

struct DrmConfig {
  // If empty, pick the first connected connector and its preferred mode.
  // If set, try to open and use this node (e.g. "/dev/dri/card0").
  const char* device_path = "";
};

// DRM/KMS display with dumb buffers. Build-gated behind ENABLE_DRMKMS.
class DrmKmsDisplay final : public IDisplay {
 public:
  explicit DrmKmsDisplay(DrmConfig cfg = {});
  ~DrmKmsDisplay() override;

  bool Init(int width, int height) override;
  bool Present(const Frame& rgb888) override;

 private:
  struct Impl;
  DrmConfig cfg_;
  Impl* impl_ = nullptr;
};

}  // namespace phc

