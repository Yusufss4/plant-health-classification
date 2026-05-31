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

  // Fused fast path: resize + normalize directly into a caller-owned NCHW float
  // buffer of exactly 1*3*kInputSize*kInputSize floats (e.g. engine.input_data()).
  // If swap_rb is true, source channel 0 and 2 are swapped before normalization.
  // Returns false if the frame format/shape is unsupported.
  bool RunInto(const Frame& frame_rgb888, float* dst_nchw,
               bool swap_rb = false) const;

  // Returns false if frame format unsupported.
  bool Run(const Frame& frame_rgb888, TensorF32& out) const;

  // Convenience for existing CLI code paths (disk image -> tensor).
  bool ImageFileToTensorNchw(const std::string& path, TensorF32& out) const;

  // Disk image -> caller-owned NCHW float buffer (fused fast path).
  bool ImageFileToTensorInto(const std::string& path, float* dst_nchw) const;

  // Loads raw float32 tensor from file (1x3x224x224 NCHW).
  bool LoadTensorBin(const std::string& path, TensorF32& out) const;

  // Loads raw float32 tensor (1x3x224x224 NCHW) into a caller-owned buffer.
  bool LoadTensorBinInto(const std::string& path, float* dst_nchw) const;
};

}  // namespace phc
