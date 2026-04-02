/**
 * MobileNet-v3 plant health inference using ONNX Runtime (C++).
 *
 * Usage:
 *   infer_mobilenet <model.onnx> <image.jpg>
 *   infer_mobilenet <model.onnx> --tensor-bin <float32_nchw_1x3x224x224.bin>
 */

#include "inference_ort/ort_engine.hpp"
#include "preprocess/mobilenet_preprocess.hpp"

#include <cstring>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  std::string model_path;
  phc::TensorF32 input;
  phc::MobilenetPreprocessor pp;

  if (argc == 4 && std::strcmp(argv[2], "--tensor-bin") == 0) {
    model_path = argv[1];
    if (!pp.LoadTensorBin(argv[3], input)) {
      return 1;
    }
  } else if (argc == 3) {
    model_path = argv[1];
    if (!pp.ImageFileToTensorNchw(argv[2], input)) {
      return 1;
    }
  } else {
    std::cerr << "Usage:\n  " << argv[0]
              << " <model.onnx> <image.jpg>\n  " << argv[0]
              << " <model.onnx> --tensor-bin <preprocessed_float32_nchw.bin>\n";
    return 1;
  }

  phc::OrtInferenceEngine engine(model_path);
  const phc::InferenceResult r = engine.Run(input);
  if (r.logits.size() >= 2 && r.probabilities.size() >= 2) {
    std::cout << "logits: [" << r.logits[0] << ", " << r.logits[1] << "]\n";
    std::cout << "prob:   [" << r.probabilities[0] << ", " << r.probabilities[1] << "]\n";
    std::cout << "class:  " << r.label << " (" << (r.label_name.empty() ? "?" : r.label_name) << ")\n";
  } else {
    std::cout << "class:  " << r.label << " (" << (r.label_name.empty() ? "?" : r.label_name) << ")\n";
  }
  return 0;
}

