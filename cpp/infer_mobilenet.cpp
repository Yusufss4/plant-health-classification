/**
 * MobileNet-v3 plant health inference using ONNX Runtime (C++).
 *
 * Usage:
 *   infer_mobilenet <model.onnx> <image.jpg>
 *   infer_mobilenet <model.onnx> --tensor-bin <float32_nchw_1x3x224x224.bin>
 *
 * Preprocessing matches training: RGB, resize 224x224 (bilinear), ImageNet mean/std, NCHW.
 * Class 0 = healthy, 1 = diseased.
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kSize = 224;
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

inline float Lerp(float a, float b, float t) { return a + (b - a) * t; }

void ResizeBilinearRgb(const unsigned char* src, int sw, int sh, unsigned char* dst, int dw, int dh) {
  for (int y = 0; y < dh; ++y) {
    float sy = (static_cast<float>(y) + 0.5f) * static_cast<float>(sh) / static_cast<float>(dh) - 0.5f;
    int y0 = static_cast<int>(std::floor(sy));
    y0 = std::clamp(y0, 0, sh - 1);
    int y1 = std::min(y0 + 1, sh - 1);
    float fy = sy - static_cast<float>(y0);
    for (int x = 0; x < dw; ++x) {
      float sx = (static_cast<float>(x) + 0.5f) * static_cast<float>(sw) / static_cast<float>(dw) - 0.5f;
      int x0 = static_cast<int>(std::floor(sx));
      x0 = std::clamp(x0, 0, sw - 1);
      int x1 = std::min(x0 + 1, sw - 1);
      float fx = sx - static_cast<float>(x0);
      for (int c = 0; c < 3; ++c) {
        float v00 = static_cast<float>(src[(y0 * sw + x0) * 3 + c]);
        float v01 = static_cast<float>(src[(y0 * sw + x1) * 3 + c]);
        float v10 = static_cast<float>(src[(y1 * sw + x0) * 3 + c]);
        float v11 = static_cast<float>(src[(y1 * sw + x1) * 3 + c]);
        float v0 = Lerp(v00, v01, fx);
        float v1 = Lerp(v10, v11, fx);
        float v = Lerp(v0, v1, fy);
        dst[(y * dw + x) * 3 + c] = static_cast<unsigned char>(std::round(v));
      }
    }
  }
}

void PreprocessRgbToNchw(const unsigned char* rgb224, std::vector<float>& nchw) {
  constexpr size_t n = 1 * 3 * kSize * kSize;
  nchw.resize(n);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < kSize; ++y) {
      for (int x = 0; x < kSize; ++x) {
        float px = static_cast<float>(rgb224[(y * kSize + x) * 3 + c]) / 255.0f;
        px = (px - kMean[c]) / kStd[c];
        nchw[c * kSize * kSize + y * kSize + x] = px;
      }
    }
  }
}

bool LoadTensorBin(const std::string& path, std::vector<float>& out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "Cannot open " << path << "\n";
    return false;
  }
  const auto bytes = static_cast<size_t>(f.tellg());
  f.seekg(0);
  constexpr size_t expected = 1 * 3 * kSize * kSize * sizeof(float);
  if (bytes != expected) {
    std::cerr << "Expected " << expected << " bytes, got " << bytes << "\n";
    return false;
  }
  out.resize(1 * 3 * kSize * kSize);
  f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
  return static_cast<bool>(f);
}

bool ImageToNchw(const std::string& path, std::vector<float>& nchw) {
  int w = 0, h = 0, comp = 0;
  unsigned char* data = stbi_load(path.c_str(), &w, &h, &comp, 3);
  if (!data) {
    std::cerr << "stbi_load failed: " << path << " — " << stbi_failure_reason() << "\n";
    return false;
  }
  std::vector<unsigned char> resized(3 * kSize * kSize);
  ResizeBilinearRgb(data, w, h, resized.data(), kSize, kSize);
  stbi_image_free(data);
  PreprocessRgbToNchw(resized.data(), nchw);
  return true;
}

void PrintLogits(const float* logits, size_t n) {
  float maxv = logits[0];
  int argmax = 0;
  for (size_t i = 1; i < n; ++i) {
    if (logits[i] > maxv) {
      maxv = logits[i];
      argmax = static_cast<int>(i);
    }
  }
  float sum = 0.f;
  std::vector<float> prob(n);
  for (size_t i = 0; i < n; ++i) {
    prob[i] = std::exp(logits[i] - maxv);
    sum += prob[i];
  }
  for (size_t i = 0; i < n; ++i) {
    prob[i] /= sum;
  }
  static const char* kNames[] = {"healthy", "diseased"};
  std::cout << "logits: [" << logits[0] << ", " << logits[1] << "]\n";
  std::cout << "prob:   [" << prob[0] << ", " << prob[1] << "]\n";
  std::cout << "class:  " << argmax << " (" << kNames[argmax] << ")\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::vector<float> input_nchw;

  if (argc == 4 && std::strcmp(argv[2], "--tensor-bin") == 0) {
    model_path = argv[1];
    if (!LoadTensorBin(argv[3], input_nchw)) {
      return 1;
    }
  } else if (argc == 3) {
    model_path = argv[1];
    if (!ImageToNchw(argv[2], input_nchw)) {
      return 1;
    }
  } else {
    std::cerr << "Usage:\n  " << argv[0]
              << " <model.onnx> <image.jpg>\n  " << argv[0]
              << " <model.onnx> --tensor-bin <preprocessed_float32_nchw.bin>\n";
    return 1;
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "infer");
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);
  opts.SetInterOpNumThreads(1);

#ifdef _WIN32
  std::wstring wmodel(model_path.begin(), model_path.end());
  Ort::Session session(env, wmodel.c_str(), opts);
#else
  Ort::Session session(env, model_path.c_str(), opts);
#endif

  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name_holder = session.GetInputNameAllocated(0, allocator);
  auto output_name_holder = session.GetOutputNameAllocated(0, allocator);

  std::array<int64_t, 4> shape = {1, 3, kSize, kSize};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem, input_nchw.data(), input_nchw.size(), shape.data(), shape.size());

  const char* in_names[] = {input_name_holder.get()};
  const char* out_names[] = {output_name_holder.get()};
  auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 1);

  float* out_data = outputs[0].GetTensorMutableData<float>();
  auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  size_t out_count = 1;
  for (int64_t d : out_shape) {
    out_count *= static_cast<size_t>(d);
  }
  PrintLogits(out_data, out_count);
  return 0;
}
