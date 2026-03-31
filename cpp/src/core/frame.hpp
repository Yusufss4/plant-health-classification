#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace phc {

enum class PixelFormat {
  Rgb888,
  Nv12,
};

struct Frame {
  PixelFormat format = PixelFormat::Rgb888;
  int width = 0;
  int height = 0;
  int stride_bytes = 0;
  uint64_t timestamp_ns = 0;

  // For RGB888: size = height * stride_bytes, 3 bytes per pixel.
  // For NV12: size = height * stride_bytes * 3/2 (Y then interleaved UV).
  std::vector<uint8_t> data;

  bool empty() const { return data.empty() || width <= 0 || height <= 0 || stride_bytes <= 0; }
};

}  // namespace phc

