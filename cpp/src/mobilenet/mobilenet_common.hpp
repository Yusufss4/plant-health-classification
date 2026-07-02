#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace mobilenet {

constexpr int kInputSize = 224;

bool ImageToNchw(const std::string& path, std::vector<float>& out);

bool LoadTensorBin(const std::string& path, std::vector<float>& out);

std::vector<float> RunInference(Ort::Session& session,
                                const std::vector<float>& nchw);

void PrintLogitsLine(const float* logits, size_t n);

}  // namespace mobilenet
