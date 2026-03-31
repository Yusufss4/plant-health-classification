#pragma once

#include "../core/frame.hpp"
#include "tensor.hpp"

#include <string>

namespace phc {

// MobileNet-family preprocessing used by this repo today:
// - Input: RGB888
// - Resize: bilinear to 224x224
// - Normalize: ImageNet mean/std
// - Layout: NCHW float32
class MobilenetPreprocessor {
 public:
  static constexpr int kInputSize = 224;

  // Returns false if frame format unsupported.
  bool Run(const Frame& frame_rgb888, TensorF32& out) const;

  // Convenience for existing CLI code paths (disk image -> tensor).
  bool ImageFileToTensorNchw(const std::string& path, TensorF32& out) const;

  // Loads raw float32 tensor from file (1x3x224x224 NCHW).
  bool LoadTensorBin(const std::string& path, TensorF32& out) const;
};

}  // namespace phc

