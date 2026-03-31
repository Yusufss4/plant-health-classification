#include "mobilenet_preprocess.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

namespace phc {
namespace {

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

inline float Lerp(float a, float b, float t) { return a + (b - a) * t; }

void ResizeBilinearRgb(const uint8_t* src, int sw, int sh, int src_stride_bytes, uint8_t* dst, int dw, int dh) {
  // src assumed tightly packed RGB888 per row (stride may include padding).
  for (int y = 0; y < dh; ++y) {
    const float sy = (static_cast<float>(y) + 0.5f) * static_cast<float>(sh) / static_cast<float>(dh) - 0.5f;
    int y0 = static_cast<int>(std::floor(sy));
    y0 = std::clamp(y0, 0, sh - 1);
    const int y1 = std::min(y0 + 1, sh - 1);
    const float fy = sy - static_cast<float>(y0);

    const uint8_t* row0 = src + y0 * src_stride_bytes;
    const uint8_t* row1 = src + y1 * src_stride_bytes;
    for (int x = 0; x < dw; ++x) {
      const float sx = (static_cast<float>(x) + 0.5f) * static_cast<float>(sw) / static_cast<float>(dw) - 0.5f;
      int x0 = static_cast<int>(std::floor(sx));
      x0 = std::clamp(x0, 0, sw - 1);
      const int x1 = std::min(x0 + 1, sw - 1);
      const float fx = sx - static_cast<float>(x0);

      const uint8_t* p00 = row0 + x0 * 3;
      const uint8_t* p01 = row0 + x1 * 3;
      const uint8_t* p10 = row1 + x0 * 3;
      const uint8_t* p11 = row1 + x1 * 3;
      uint8_t* out = dst + (y * dw + x) * 3;
      for (int c = 0; c < 3; ++c) {
        const float v00 = static_cast<float>(p00[c]);
        const float v01 = static_cast<float>(p01[c]);
        const float v10 = static_cast<float>(p10[c]);
        const float v11 = static_cast<float>(p11[c]);
        const float v0 = Lerp(v00, v01, fx);
        const float v1 = Lerp(v10, v11, fx);
        const float v = Lerp(v0, v1, fy);
        out[c] = static_cast<uint8_t>(std::round(v));
      }
    }
  }
}

void PreprocessRgbToNchw(const uint8_t* rgb224, std::vector<float>& nchw) {
  constexpr size_t n = 1 * 3 * MobilenetPreprocessor::kInputSize * MobilenetPreprocessor::kInputSize;
  nchw.resize(n);
  const int s = MobilenetPreprocessor::kInputSize;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < s; ++y) {
      for (int x = 0; x < s; ++x) {
        float px = static_cast<float>(rgb224[(y * s + x) * 3 + c]) / 255.0f;
        px = (px - kMean[c]) / kStd[c];
        nchw[static_cast<size_t>(c * s * s + y * s + x)] = px;
      }
    }
  }
}

}  // namespace

bool MobilenetPreprocessor::Run(const Frame& frame_rgb888, TensorF32& out) const {
  if (frame_rgb888.format != PixelFormat::Rgb888) {
    return false;
  }
  if (frame_rgb888.empty()) {
    return false;
  }
  if (frame_rgb888.stride_bytes < frame_rgb888.width * 3) {
    return false;
  }

  std::vector<uint8_t> resized(static_cast<size_t>(3 * kInputSize * kInputSize));
  ResizeBilinearRgb(frame_rgb888.data.data(),
                    frame_rgb888.width,
                    frame_rgb888.height,
                    frame_rgb888.stride_bytes,
                    resized.data(),
                    kInputSize,
                    kInputSize);

  out.n = 1;
  out.c = 3;
  out.h = kInputSize;
  out.w = kInputSize;
  PreprocessRgbToNchw(resized.data(), out.data);
  return true;
}

bool MobilenetPreprocessor::ImageFileToTensorNchw(const std::string& path, TensorF32& out) const {
  int w = 0, h = 0, comp = 0;
  unsigned char* data = stbi_load(path.c_str(), &w, &h, &comp, 3);
  if (!data) {
    std::cerr << "stbi_load failed: " << path << " — " << stbi_failure_reason() << "\n";
    return false;
  }
  Frame f;
  f.format = PixelFormat::Rgb888;
  f.width = w;
  f.height = h;
  f.stride_bytes = w * 3;
  f.data.assign(data, data + static_cast<size_t>(w * h * 3));
  stbi_image_free(data);
  return Run(f, out);
}

bool MobilenetPreprocessor::LoadTensorBin(const std::string& path, TensorF32& out) const {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "Cannot open " << path << "\n";
    return false;
  }
  const auto bytes = static_cast<size_t>(f.tellg());
  f.seekg(0);
  constexpr size_t expected = 1 * 3 * kInputSize * kInputSize * sizeof(float);
  if (bytes != expected) {
    std::cerr << "Expected " << expected << " bytes, got " << bytes << "\n";
    return false;
  }
  out.n = 1;
  out.c = 3;
  out.h = kInputSize;
  out.w = kInputSize;
  out.data.resize(1 * 3 * kInputSize * kInputSize);
  f.read(reinterpret_cast<char*>(out.data.data()), static_cast<std::streamsize>(bytes));
  return static_cast<bool>(f);
}

}  // namespace phc

