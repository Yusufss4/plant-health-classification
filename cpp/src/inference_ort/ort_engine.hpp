#pragma once

#include "../core/inference_result.hpp"
#include "../preprocess/tensor.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <string>
#include <vector>

namespace phc {

class OrtInferenceEngine {
 public:
  struct Options {
    int intra_op_threads = 1;
    int inter_op_threads = 1;
    std::vector<std::string> class_names = {"healthy", "diseased"};
  };

  explicit OrtInferenceEngine(const std::string& model_path);
  OrtInferenceEngine(const std::string& model_path, const Options& options);

  InferenceResult Run(const TensorF32& input_nchw, uint64_t timestamp_ns = 0);

 private:
  Options options_;
  Ort::Env env_;
  Ort::Session session_;
};

}  // namespace phc
