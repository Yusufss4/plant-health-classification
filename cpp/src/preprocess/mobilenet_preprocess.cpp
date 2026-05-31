#include "mobilenet_preprocess.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace phc {
namespace {

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

// Per-channel constants folding /255 and ImageNet normalize into one FMA:
//   normalized = raw * inv255_over_std + neg_mean_over_std   (raw in [0,255]).
struct NormConsts {
  float inv[4];  // lane 3 unused (padding for NEON)
  float neg[4];
};

inline NormConsts MakeNormConsts() {
  NormConsts n{};
  for (int c = 0; c < 3; ++c) {
    n.inv[c] = 1.0f / (255.0f * kStd[c]);
    n.neg[c] = -kMean[c] / kStd[c];
  }
  n.inv[3] = 0.0f;
  n.neg[3] = 0.0f;
  return n;
}

// Bilinear-sample tap: integer neighbor indices + fractional weight per axis.
struct Tap {
  int i0;
  int i1;
  float frac;
};

// Precompute taps for one axis. Matches the original floor()+clamp behavior:
// for the sampling formula sx > -0.5 always holds, so a truncating cast plus
// clamp reproduces floor() exactly while skipping the std::floor call.
void BuildTaps(int dst_len, int src_len, std::vector<Tap>& taps) {
  taps.resize(static_cast<size_t>(dst_len));
  const float scale = static_cast<float>(src_len) / static_cast<float>(dst_len);
  for (int i = 0; i < dst_len; ++i) {
    const float s = (static_cast<float>(i) + 0.5f) * scale - 0.5f;
    int i0 = static_cast<int>(s);
    i0 = std::clamp(i0, 0, src_len - 1);
    const int i1 = std::min(i0 + 1, src_len - 1);
    taps[static_cast<size_t>(i)] = {i0, i1, s - static_cast<float>(i0)};
  }
}

// Normalize one interpolated RGB triplet (dst channel order) and scatter into
// the three planar destinations.
inline void NormalizeStore(const float p00[3], const float p01[3],
                           const float p10[3], const float p11[3], float fx,
                           float fy, const NormConsts& nc, float* d0, float* d1,
                           float* d2) {
#if defined(__aarch64__)
  const float a00[4] = {p00[0], p00[1], p00[2], 0.0f};
  const float a01[4] = {p01[0], p01[1], p01[2], 0.0f};
  const float a10[4] = {p10[0], p10[1], p10[2], 0.0f};
  const float a11[4] = {p11[0], p11[1], p11[2], 0.0f};
  const float32x4_t v00 = vld1q_f32(a00);
  const float32x4_t v01 = vld1q_f32(a01);
  const float32x4_t v10 = vld1q_f32(a10);
  const float32x4_t v11 = vld1q_f32(a11);
  const float32x4_t r0 = vfmaq_n_f32(v00, vsubq_f32(v01, v00), fx);
  const float32x4_t r1 = vfmaq_n_f32(v10, vsubq_f32(v11, v10), fx);
  const float32x4_t v = vfmaq_n_f32(r0, vsubq_f32(r1, r0), fy);
  const float32x4_t out =
      vfmaq_f32(vld1q_f32(nc.neg), v, vld1q_f32(nc.inv));
  float o[4];
  vst1q_f32(o, out);
  *d0 = o[0];
  *d1 = o[1];
  *d2 = o[2];
#else
  float* dst[3] = {d0, d1, d2};
  for (int c = 0; c < 3; ++c) {
    const float v0 = p00[c] + (p01[c] - p00[c]) * fx;
    const float v1 = p10[c] + (p11[c] - p10[c]) * fx;
    const float v = v0 + (v1 - v0) * fy;
    *dst[c] = v * nc.inv[c] + nc.neg[c];
  }
#endif
}

}  // namespace

bool MobilenetPreprocessor::RunInto(const Frame& frame_rgb888, float* dst_nchw,
                                    bool swap_rb) const {
  if (frame_rgb888.format != PixelFormat::Rgb888) {
    return false;
  }
  if (frame_rgb888.empty()) {
    return false;
  }
  if (frame_rgb888.stride_bytes < frame_rgb888.width * 3) {
    return false;
  }
  if (dst_nchw == nullptr) {
    return false;
  }

  const int s = kInputSize;
  const int sw = frame_rgb888.width;
  const int sh = frame_rgb888.height;
  const int src_stride = frame_rgb888.stride_bytes;
  const uint8_t* src = frame_rgb888.data.data();

  const NormConsts nc = MakeNormConsts();

  std::vector<Tap> tx, ty;
  BuildTaps(s, sw, tx);
  BuildTaps(s, sh, ty);

  // Map dst channel -> source channel (handles optional R/B swap).
  const int ch[3] = {swap_rb ? 2 : 0, 1, swap_rb ? 0 : 2};

  const size_t plane = static_cast<size_t>(s) * static_cast<size_t>(s);
  float* d0 = dst_nchw;
  float* d1 = dst_nchw + plane;
  float* d2 = dst_nchw + 2 * plane;

  for (int y = 0; y < s; ++y) {
    const Tap& ty_t = ty[static_cast<size_t>(y)];
    const uint8_t* row0 = src + static_cast<size_t>(ty_t.i0) * src_stride;
    const uint8_t* row1 = src + static_cast<size_t>(ty_t.i1) * src_stride;
    const float fy = ty_t.frac;
    const size_t row_off = static_cast<size_t>(y) * static_cast<size_t>(s);

    for (int x = 0; x < s; ++x) {
      const Tap& tx_t = tx[static_cast<size_t>(x)];
      const int x0 = tx_t.i0 * 3;
      const int x1 = tx_t.i1 * 3;
      const float fx = tx_t.frac;

      float p00[3], p01[3], p10[3], p11[3];
      for (int c = 0; c < 3; ++c) {
        const int sc = ch[c];
        p00[c] = static_cast<float>(row0[x0 + sc]);
        p01[c] = static_cast<float>(row0[x1 + sc]);
        p10[c] = static_cast<float>(row1[x0 + sc]);
        p11[c] = static_cast<float>(row1[x1 + sc]);
      }

      const size_t idx = row_off + static_cast<size_t>(x);
      NormalizeStore(p00, p01, p10, p11, fx, fy, nc, d0 + idx, d1 + idx,
                     d2 + idx);
    }
  }
  return true;
}

bool MobilenetPreprocessor::Run(const Frame& frame_rgb888,
                                TensorF32& out) const {
  out.n = 1;
  out.c = 3;
  out.h = kInputSize;
  out.w = kInputSize;
  out.data.resize(static_cast<size_t>(3 * kInputSize * kInputSize));
  return RunInto(frame_rgb888, out.data.data(), /*swap_rb=*/false);
}

bool MobilenetPreprocessor::ImageFileToTensorInto(const std::string& path,
                                                  float* dst_nchw) const {
  int w = 0, h = 0, comp = 0;
  unsigned char* data = stbi_load(path.c_str(), &w, &h, &comp, 3);
  if (!data) {
    std::cerr << "stbi_load failed: " << path << " — " << stbi_failure_reason()
              << "\n";
    return false;
  }
  Frame f;
  f.format = PixelFormat::Rgb888;
  f.width = w;
  f.height = h;
  f.stride_bytes = w * 3;
  f.data.assign(data, data + static_cast<size_t>(w * h * 3));
  stbi_image_free(data);
  return RunInto(f, dst_nchw, /*swap_rb=*/false);
}

bool MobilenetPreprocessor::ImageFileToTensorNchw(const std::string& path,
                                                  TensorF32& out) const {
  out.n = 1;
  out.c = 3;
  out.h = kInputSize;
  out.w = kInputSize;
  out.data.resize(static_cast<size_t>(3 * kInputSize * kInputSize));
  return ImageFileToTensorInto(path, out.data.data());
}

bool MobilenetPreprocessor::LoadTensorBinInto(const std::string& path,
                                              float* dst_nchw) const {
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
  f.read(reinterpret_cast<char*>(dst_nchw),
         static_cast<std::streamsize>(bytes));
  return static_cast<bool>(f);
}

bool MobilenetPreprocessor::LoadTensorBin(const std::string& path,
                                          TensorF32& out) const {
  out.n = 1;
  out.c = 3;
  out.h = kInputSize;
  out.w = kInputSize;
  out.data.resize(static_cast<size_t>(1 * 3 * kInputSize * kInputSize));
  return LoadTensorBinInto(path, out.data.data());
}

}  // namespace phc
