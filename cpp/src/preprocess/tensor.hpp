#pragma once

#include <cstdint>
#include <vector>

namespace phc {

struct TensorF32 {
  // Flattened in NCHW order for now.
  std::vector<float> data;
  int n = 1;
  int c = 0;
  int h = 0;
  int w = 0;
};

}  // namespace phc
