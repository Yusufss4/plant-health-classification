// Single-image ONNX inference. See cpp/README.md for usage.

#include "inference_ort/ort_engine.hpp"
#include "preprocess/mobilenet_preprocess.hpp"

#include <cstring>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  std::string model_path;
  std::string tensor_bin_path;
  std::string image_path;
  phc::MobilenetPreprocessor pp;

  if (argc == 4 && std::strcmp(argv[2], "--tensor-bin") == 0) {
    model_path = argv[1];
    tensor_bin_path = argv[3];
  } else if (argc == 3) {
    model_path = argv[1];
    image_path = argv[2];
  } else {
    std::cerr << "Usage:\n  " << argv[0] << " <model.onnx> <image.jpg>\n  "
              << argv[0]
              << " <model.onnx> --tensor-bin <preprocessed_float32_nchw.bin>\n";
    return 1;
  }

  phc::OrtInferenceEngine engine(model_path);
  if (!tensor_bin_path.empty()) {
    if (!pp.LoadTensorBinInto(tensor_bin_path, engine.input_data())) {
      return 1;
    }
  } else {
    if (!pp.ImageFileToTensorInto(image_path, engine.input_data())) {
      return 1;
    }
  }
  const phc::InferenceResult r = engine.Run();
  if (!r.logits.empty()) {
    std::cout << "logits: [";
    for (size_t i = 0; i < r.logits.size(); ++i) {
      std::cout << r.logits[i] << (i + 1 < r.logits.size() ? ", " : "");
    }
    std::cout << "]\n";
  }
  if (!r.probabilities.empty()) {
    std::cout << "prob:   [";
    for (size_t i = 0; i < r.probabilities.size(); ++i) {
      std::cout << r.probabilities[i]
                << (i + 1 < r.probabilities.size() ? ", " : "");
    }
    std::cout << "]\n";
  }
  std::cout << "class:  " << r.label << " ("
            << (r.label_name.empty() ? "?" : r.label_name) << ")\n";
  return 0;
}
