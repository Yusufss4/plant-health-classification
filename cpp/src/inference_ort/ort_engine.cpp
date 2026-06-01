#include "ort_engine.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace phc {
namespace {

bool EndsWith(const std::string& s, const std::string& suffix) {
  if (s.size() < suffix.size()) {
    return false;
  }
  return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin(),
                    [](char a, char b) {
                      return std::tolower(static_cast<unsigned char>(a)) ==
                             std::tolower(static_cast<unsigned char>(b));
                    });
}

Ort::Session MakeSession(Ort::Env& env, const std::string& model_path,
                         const OrtInferenceEngine::Options& options) {
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(options.intra_op_threads);
  opts.SetInterOpNumThreads(options.inter_op_threads);
  opts.SetExecutionMode(ORT_SEQUENTIAL);
  // A pre-optimized .ort already has graph optimizations baked in; re-running
  // them is wasted startup time (and unsupported for some ORT-format models).
  if (EndsWith(model_path, ".ort")) {
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  } else {
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  }
  return Ort::Session(env, model_path.c_str(), opts);
}

}  // namespace

OrtInferenceEngine::OrtInferenceEngine(const std::string& model_path,
                                       const Options& options)
    : options_(options),
      env_(ORT_LOGGING_LEVEL_WARNING, "phc"),
      session_(MakeSession(env_, model_path, options_)),
      allocator_(),
      mem_info_(
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      binding_(session_) {
  Setup(model_path);
}

OrtInferenceEngine::OrtInferenceEngine(const std::string& model_path)
    : OrtInferenceEngine(model_path, Options{}) {}

void OrtInferenceEngine::Setup(const std::string& /*model_path*/) {
  {
    Ort::AllocatedStringPtr in_holder =
        session_.GetInputNameAllocated(0, allocator_);
    Ort::AllocatedStringPtr out_holder =
        session_.GetOutputNameAllocated(0, allocator_);
    input_name_str_ = in_holder.get();
    output_name_str_ = out_holder.get();
  }
  in_names_[0] = input_name_str_.c_str();
  out_names_[0] = output_name_str_.c_str();

  // Resolve the (static) input shape, falling back to the project contract
  // [1, 3, 224, 224] for any dimension the model leaves dynamic (<= 0).
  const auto in_dims =
      session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  const int64_t defaults[4] = {1, 3, 224, 224};
  int64_t resolved[4] = {1, 3, 224, 224};
  for (size_t i = 0; i < 4 && i < in_dims.size(); ++i) {
    resolved[i] = (in_dims[i] > 0) ? in_dims[i] : defaults[i];
  }
  in_n_ = resolved[0];
  in_c_ = resolved[1];
  in_h_ = resolved[2];
  in_w_ = resolved[3];
  in_shape_ = {in_n_, in_c_, in_h_, in_w_};

  const size_t in_count = static_cast<size_t>(in_n_) *
                          static_cast<size_t>(in_c_) *
                          static_cast<size_t>(in_h_) *
                          static_cast<size_t>(in_w_);
  input_buf_.assign(in_count, 0.0f);

  // Resolve the (static) output shape; any dynamic dim collapses to 1.
  const auto out_dims =
      session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  out_shape_.clear();
  size_t out_count = 1;
  if (out_dims.empty()) {
    out_shape_.push_back(1);
  } else {
    for (int64_t d : out_dims) {
      const int64_t dim = (d > 0) ? d : 1;
      out_shape_.push_back(dim);
      out_count *= static_cast<size_t>(dim);
    }
  }
  output_buf_.assign(out_count, 0.0f);

  input_value_ = Ort::Value::CreateTensor<float>(
      mem_info_, input_buf_.data(), input_buf_.size(), in_shape_.data(),
      in_shape_.size());
  output_value_ = Ort::Value::CreateTensor<float>(
      mem_info_, output_buf_.data(), output_buf_.size(), out_shape_.data(),
      out_shape_.size());

  binding_.BindInput(in_names_[0], input_value_);
  binding_.BindOutput(out_names_[0], output_value_);

  result_.logits.assign(out_count, 0.0f);
  result_.probabilities.assign(out_count, 0.0f);
}

void OrtInferenceEngine::FillResult(uint64_t timestamp_ns) {
  const size_t n = output_buf_.size();
  result_.timestamp_ns = timestamp_ns;

  if (result_.logits.size() != n) {
    result_.logits.resize(n);
    result_.probabilities.resize(n);
  }
  std::copy(output_buf_.begin(), output_buf_.end(), result_.logits.begin());

  if (n == 0) {
    result_.label = -1;
    result_.confidence = 0.0f;
    result_.label_name.clear();
    return;
  }

  // Softmax + argmax over the reused probability buffer (no allocation).
  float maxv = output_buf_[0];
  for (size_t i = 1; i < n; ++i) {
    maxv = std::max(maxv, output_buf_[i]);
  }
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    const float e = std::exp(output_buf_[i] - maxv);
    result_.probabilities[i] = e;
    sum += e;
  }
  const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
  int best = 0;
  float best_v = -1.0f;
  for (size_t i = 0; i < n; ++i) {
    result_.probabilities[i] *= inv_sum;
    if (result_.probabilities[i] > best_v) {
      best_v = result_.probabilities[i];
      best = static_cast<int>(i);
    }
  }

  result_.label = best;
  result_.confidence = result_.probabilities[static_cast<size_t>(best)];
  if (static_cast<size_t>(best) < options_.class_names.size()) {
    result_.label_name = options_.class_names[static_cast<size_t>(best)];
  } else {
    result_.label_name.clear();
  }
}

const InferenceResult& OrtInferenceEngine::Run(uint64_t timestamp_ns) {
  session_.Run(Ort::RunOptions{nullptr}, binding_);
  FillResult(timestamp_ns);
  return result_;
}

const InferenceResult& OrtInferenceEngine::Run(const TensorF32& input_nchw,
                                               uint64_t timestamp_ns) {
  if (input_nchw.n != 1 || input_nchw.c <= 0 || input_nchw.h <= 0 ||
      input_nchw.w <= 0) {
    throw std::runtime_error("Bad input tensor shape");
  }
  if (input_nchw.data.size() != input_buf_.size()) {
    throw std::runtime_error("Input tensor size mismatch with model input");
  }
  std::memcpy(input_buf_.data(), input_nchw.data.data(),
              input_buf_.size() * sizeof(float));
  return Run(timestamp_ns);
}

}  // namespace phc
