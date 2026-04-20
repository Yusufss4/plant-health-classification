#include "ort_engine.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace phc {
namespace {

std::vector<float> Softmax(const std::vector<float>& logits) {
  if (logits.empty()) {
    return {};
  }
  const float maxv = *std::max_element(logits.begin(), logits.end());
  std::vector<float> out(logits.size());
  float sum = 0.0f;
  for (size_t i = 0; i < logits.size(); ++i) {
    out[i] = std::exp(logits[i] - maxv);
    sum += out[i];
  }
  if (sum > 0.0f) {
    for (float& v : out) {
      v /= sum;
    }
  }
  return out;
}

int Argmax(const std::vector<float>& v) {
  if (v.empty()) {
    return -1;
  }
  return static_cast<int>(
      std::distance(v.begin(), std::max_element(v.begin(), v.end())));
}

}  // namespace

OrtInferenceEngine::OrtInferenceEngine(const std::string& model_path,
                                       const Options& options)
    : options_(options),
      env_(ORT_LOGGING_LEVEL_WARNING, "phc"),
      session_([&]() {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(options_.intra_op_threads);
        opts.SetInterOpNumThreads(options_.inter_op_threads);
        return Ort::Session(env_, model_path.c_str(), opts);
      }()) {}

OrtInferenceEngine::OrtInferenceEngine(const std::string& model_path)
    : OrtInferenceEngine(model_path, Options{}) {}

InferenceResult OrtInferenceEngine::Run(const TensorF32& input_nchw,
                                        uint64_t timestamp_ns) {
  if (input_nchw.n != 1 || input_nchw.c <= 0 || input_nchw.h <= 0 ||
      input_nchw.w <= 0) {
    throw std::runtime_error("Bad input tensor shape");
  }
  const int64_t n = input_nchw.n;
  const int64_t c = input_nchw.c;
  const int64_t h = input_nchw.h;
  const int64_t w = input_nchw.w;
  const std::array<int64_t, 4> shape = {n, c, h, w};

  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name_holder = session_.GetInputNameAllocated(0, allocator);
  auto output_name_holder = session_.GetOutputNameAllocated(0, allocator);

  Ort::MemoryInfo mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem, const_cast<float*>(input_nchw.data.data()), input_nchw.data.size(),
      shape.data(), shape.size());

  const char* in_names[] = {input_name_holder.get()};
  const char* out_names[] = {output_name_holder.get()};
  auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &input_tensor,
                              1, out_names, 1);

  float* out_data = outputs[0].GetTensorMutableData<float>();
  auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  size_t out_count = 1;
  for (int64_t d : out_shape) {
    out_count *= static_cast<size_t>(d);
  }
  std::vector<float> logits(out_count);
  std::copy(out_data, out_data + out_count, logits.begin());

  InferenceResult r;
  r.timestamp_ns = timestamp_ns;
  r.logits = logits;
  r.probabilities = Softmax(r.logits);
  r.label = Argmax(r.probabilities);
  r.confidence =
      (r.label >= 0 && static_cast<size_t>(r.label) < r.probabilities.size())
          ? r.probabilities[static_cast<size_t>(r.label)]
          : 0.0f;
  if (r.label >= 0 &&
      static_cast<size_t>(r.label) < options_.class_names.size()) {
    r.label_name = options_.class_names[static_cast<size_t>(r.label)];
  }
  return r;
}

}  // namespace phc
