/**
 * MobileNet-v3 plant health inference using ONNX Runtime (C++).
 *
 * Usage:
 *   infer_mobilenet <model.onnx> <image.jpg>
 *   infer_mobilenet <model.onnx> --tensor-bin <float32_nchw_1x3x224x224.bin>
 */

#include "mobilenet_common.hpp"

#include <cstring>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  std::string model_path;
  std::vector<float> input_nchw;

  if (argc == 4 && std::strcmp(argv[2], "--tensor-bin") == 0) {
    model_path = argv[1];
    if (!mobilenet::LoadTensorBin(argv[3], input_nchw)) {
      return 1;
    }
  } else if (argc == 3) {
    model_path = argv[1];
    if (!mobilenet::ImageToNchw(argv[2], input_nchw)) {
      return 1;
    }
  } else {
    std::cerr << "Usage:\n  " << argv[0]
              << " <model.onnx> <image.jpg>\n  " << argv[0]
              << " <model.onnx> --tensor-bin <preprocessed_float32_nchw.bin>\n";
    return 1;
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "infer");
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);
  opts.SetInterOpNumThreads(1);

#ifdef _WIN32
  std::wstring wmodel(model_path.begin(), model_path.end());
  Ort::Session session(env, wmodel.c_str(), opts);
#else
  Ort::Session session(env, model_path.c_str(), opts);
#endif

  std::vector<float> logits = mobilenet::RunInference(session, input_nchw);
  mobilenet::PrintLogitsLine(logits.data(), logits.size());
  return 0;
}
