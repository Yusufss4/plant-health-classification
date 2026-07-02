#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace phc {

enum class PackedRgbOrder {
  Rgb,
  Bgr,
};

inline bool NormalizePackedToRgb888(const uint8_t* src, int width, int height,
                                    int src_stride_bytes, int src_bpp,
                                    PackedRgbOrder src_order,
                                    std::vector<uint8_t>& out_rgb888,
                                    int& out_stride_bytes) {
  if (!src || width <= 0 || height <= 0) {
    return false;
  }
  if (src_bpp != 3 && src_bpp != 4) {
    return false;
  }
  const int min_src_stride = width * src_bpp;
  if (src_stride_bytes < min_src_stride) {
    return false;
  }

  out_stride_bytes = width * 3;
  const size_t out_size =
      static_cast<size_t>(height) * static_cast<size_t>(out_stride_bytes);
  if (out_rgb888.size() != out_size) {
    out_rgb888.resize(out_size);
  }

  if (src_order == PackedRgbOrder::Rgb && src_bpp == 3 &&
      src_stride_bytes == out_stride_bytes) {
    std::memcpy(out_rgb888.data(), src, out_size);
    return true;
  }

  for (int y = 0; y < height; ++y) {
    const uint8_t* src_row =
        src + static_cast<size_t>(y) * static_cast<size_t>(src_stride_bytes);
    uint8_t* dst_row = out_rgb888.data() +
                       static_cast<size_t>(y) * static_cast<size_t>(out_stride_bytes);
    for (int x = 0; x < width; ++x) {
      const uint8_t* in = src_row + static_cast<size_t>(x) * src_bpp;
      uint8_t* out = dst_row + static_cast<size_t>(x) * 3;
      if (src_order == PackedRgbOrder::Rgb) {
        out[0] = in[0];
        out[1] = in[1];
        out[2] = in[2];
      } else {
        out[0] = in[2];
        out[1] = in[1];
        out[2] = in[0];
      }
    }
  }
  return true;
}

}  // namespace phc
