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
//
// The hot path is fused into a single pass that resizes, optionally swaps R/B,
// and applies ImageNet normalization, writing straight into the destination
// NCHW float buffer (no intermediate uint8 image, no second pass). On aarch64
// the inner arithmetic uses NEON; other targets use the scalar fallback.
class MobilenetPreprocessor {
 public:
  static constexpr int kInputSize = 224;

  bool RunInto(const Frame& frame_rgb888, float* dst_nchw,
               bool swap_rb = false) const;

  bool Run(const Frame& frame_rgb888, TensorF32& out) const;

  bool ImageFileToTensorNchw(const std::string& path, TensorF32& out) const;

  bool ImageFileToTensorInto(const std::string& path, float* dst_nchw) const;

  bool LoadTensorBin(const std::string& path, TensorF32& out) const;

  bool LoadTensorBinInto(const std::string& path, float* dst_nchw) const;
};

}  // namespace phc
