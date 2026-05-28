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
  // engine.Run wall time, set by LivePipeline. 0 means not measured.
  float inference_ms = 0.0f;
  // JPEG encode wall time, set by HttpStreamDisplay. 0 means not measured.
  float encode_ms = 0.0f;
};

}  // namespace phc
