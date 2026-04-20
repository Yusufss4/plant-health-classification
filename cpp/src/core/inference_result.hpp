#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace phc {

struct InferenceResult {
  uint64_t timestamp_ns = 0;
  int label = -1;
  float confidence = 0.0f;
  std::vector<float> logits;
  std::vector<float> probabilities;
  std::string label_name;
};

}  // namespace phc
