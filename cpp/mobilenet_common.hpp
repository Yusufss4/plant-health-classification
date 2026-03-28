#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace mobilenet {

constexpr int kInputSize = 224;

/** Load image from disk, bilinear resize 224×224, RGB, ImageNet NCHW float32 (1×3×224×224 flattened). */
bool ImageToNchw(const std::string& path, std::vector<float>& out);

/** Load raw float32 tensor from file (1×3×224×224 NCHW). */
bool LoadTensorBin(const std::string& path, std::vector<float>& out);

/** Run ONNX session; input must be 1×3×224×224. Returns logits (length = num classes). */
std::vector<float> RunInference(Ort::Session& session, const std::vector<float>& nchw);

void PrintLogitsLine(const float* logits, size_t n);

}  // namespace mobilenet
