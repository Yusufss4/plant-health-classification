#pragma once

#include "../core/inference_result.hpp"
#include "../preprocess/tensor.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace phc {

// MobileNet ORT inference engine.
//
// Hot-path allocations are eliminated (C1) and I/O buffers are bound once via
// Ort::IoBinding (C13): the engine owns the input and output buffers, the model
// input/output names are resolved a single time at construction, and each Run()
// reuses the same InferenceResult/scratch vectors. Preprocessing writes directly
// into input_data(); call Run() with no tensor argument for the fast path.
class OrtInferenceEngine {
 public:
  struct Options {
    int intra_op_threads = 2;
    int inter_op_threads = 2;
    // Ort::Env severity; ORT_LOGGING_LEVEL_ERROR suppresses GPU discovery noise on Pi.
    int ort_log_level = ORT_LOGGING_LEVEL_WARNING;
    // Must match utils.data_loader.DEFAULT_CLASSES on the Python side.
    std::vector<std::string> class_names = {"healthy", "diseased", "background"};
  };

  explicit OrtInferenceEngine(const std::string& model_path);
  OrtInferenceEngine(const std::string& model_path, const Options& options);

  // Fast path: fill input_data() with the NCHW float tensor, then call Run().
  // The returned reference points at reused internal storage; copy it if you
  // need to retain it past the next Run().
  const InferenceResult& Run(uint64_t timestamp_ns = 0);

  // Compatibility path: copies the tensor into the owned input buffer, then runs.
  const InferenceResult& Run(const TensorF32& input_nchw,
                             uint64_t timestamp_ns = 0);

  // Direct access to the owned, pre-bound NCHW input buffer.
  float* input_data() { return input_buf_.data(); }
  const float* input_data() const { return input_buf_.data(); }
  std::size_t input_count() const { return input_buf_.size(); }
  int input_height() const { return static_cast<int>(in_h_); }
  int input_width() const { return static_cast<int>(in_w_); }

 private:
  void Setup(const std::string& model_path);
  void FillResult(uint64_t timestamp_ns);

  Options options_;
  Ort::Env env_;
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::MemoryInfo mem_info_;

  std::string input_name_str_;
  std::string output_name_str_;
  std::array<const char*, 1> in_names_{};
  std::array<const char*, 1> out_names_{};

  int64_t in_n_ = 1, in_c_ = 3, in_h_ = 224, in_w_ = 224;
  std::array<int64_t, 4> in_shape_{};
  std::vector<int64_t> out_shape_;

  std::vector<float> input_buf_;
  std::vector<float> output_buf_;
  Ort::Value input_value_{nullptr};
  Ort::Value output_value_{nullptr};
  Ort::IoBinding binding_;

  InferenceResult result_;
};

}  // namespace phc
