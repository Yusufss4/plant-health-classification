/**
 * Evaluate MobileNet ONNX on a folder layout matching evaluate.py / PlantHealthDataset:
 *   test_dir/healthy/(.jpg|.png|.jpeg)
 *   test_dir/diseased/(.jpg|.png|.jpeg)
 *
 * Usage:
 *   evaluate_mobilenet <model.onnx> [test_dir]
 *
 * Prints confusion matrix, accuracy, precision, recall, F1, balanced accuracy, specificity,
 * and timing (load + preprocess + ORT per image).
 */

#include "src/inference_ort/ort_engine.hpp"
#include "src/preprocess/mobilenet_preprocess.hpp"

#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct LabeledPath {
  std::string path;
  int label;  // 0 = healthy, 1 = diseased
};

bool IsImageExt(const fs::path& p) {
  auto e = p.extension().string();
  for (char& c : e) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return e == ".jpg" || e == ".jpeg" || e == ".png";
}

std::vector<LabeledPath> CollectImages(const fs::path& test_root) {
  std::vector<LabeledPath> out;
  const fs::path healthy = test_root / "healthy";
  const fs::path diseased = test_root / "diseased";
  if (!fs::is_directory(healthy) || !fs::is_directory(diseased)) {
    std::cerr << "Expected " << healthy << " and " << diseased << " directories.\n";
    return out;
  }
  for (const auto& entry : fs::directory_iterator(healthy)) {
    if (!entry.is_regular_file() || !IsImageExt(entry.path())) {
      continue;
    }
    out.push_back({entry.path().string(), 0});
  }
  for (const auto& entry : fs::directory_iterator(diseased)) {
    if (!entry.is_regular_file() || !IsImageExt(entry.path())) {
      continue;
    }
    out.push_back({entry.path().string(), 1});
  }
  return out;
}

void PrintMetrics(int64_t tn, int64_t fp, int64_t fn, int64_t tp, double total_sec, size_t n) {
  const double n_d = static_cast<double>(n);
  const double acc = static_cast<double>(tp + tn) / n_d;
  const double prec = (tp + fp) > 0 ? static_cast<double>(tp) / static_cast<double>(tp + fp) : 0.0;
  const double rec = (tp + fn) > 0 ? static_cast<double>(tp) / static_cast<double>(tp + fn) : 0.0;
  const double f1 = (prec + rec) > 1e-12 ? 2.0 * prec * rec / (prec + rec) : 0.0;
  const double tnr = (tn + fp) > 0 ? static_cast<double>(tn) / static_cast<double>(tn + fp) : 0.0;  // specificity
  const double tpr = rec;                                                                         // sensitivity
  const double bal_acc = 0.5 * (tpr + tnr);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "\n============================================================\n";
  std::cout << "EVALUATION RESULTS (ONNX / C++)\n";
  std::cout << "============================================================\n";
  std::cout << "\nOverall metrics:\n";
  std::cout << "  Accuracy:          " << acc << " (" << acc * 100.0 << "%)\n";
  std::cout << "  Balanced accuracy: " << bal_acc << " (" << bal_acc * 100.0 << "%)\n";
  std::cout << "  Precision (diseased): " << prec << " (" << prec * 100.0 << "%)\n";
  std::cout << "  Recall (diseased):    " << rec << " (" << rec * 100.0 << "%)\n";
  std::cout << "  F1-score (diseased):  " << f1 << " (" << f1 * 100.0 << "%)\n";
  std::cout << "  Specificity (healthy): " << tnr << " (" << tnr * 100.0 << "%)\n";

  std::cout << "\nConfusion matrix (rows=actual, cols=predicted):\n";
  std::cout << "                  healthy  diseased\n";
  std::cout << "  healthy    " << std::setw(8) << tn << std::setw(10) << fp << "\n";
  std::cout << "  diseased   " << std::setw(8) << fn << std::setw(10) << tp << "\n";
  std::cout << "\n  TN: " << tn << "  FP: " << fp << "  FN: " << fn << "  TP: " << tp << "\n";

  std::cout << "\nInference timing (load + preprocess + ORT, all " << n << " images):\n";
  std::cout << "  Total:     " << std::setprecision(3) << total_sec << " s\n";
  std::cout << "  Avg/image: " << (total_sec * 1000.0 / n_d) << " ms\n";
  std::cout << "  Throughput:" << (n_d / total_sec) << " img/s\n";
  std::cout << "============================================================\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> [test_dir]\n"
              << "  Default test_dir: data/test (relative to current directory)\n";
    return 1;
  }

  const std::string model_path = argv[1];
  fs::path test_root = (argc == 3) ? fs::path(argv[2]) : fs::path("data/test");
  if (!fs::is_directory(test_root)) {
    std::cerr << "Not a directory: " << test_root << "\n";
    return 1;
  }

  std::vector<LabeledPath> items = CollectImages(test_root);
  if (items.empty()) {
    std::cerr << "No images found under healthy/ and diseased/.\n";
    return 1;
  }

  phc::MobilenetPreprocessor pp;
  phc::OrtInferenceEngine engine(model_path);

  // Warm-up (ORT thread pools, caches)
  {
    phc::TensorF32 t;
    if (pp.ImageFileToTensorNchw(items[0].path, t)) {
      (void)engine.Run(t);
    }
  }

  int64_t tp = 0, tn = 0, fp = 0, fn = 0;
  const auto t0 = std::chrono::high_resolution_clock::now();

  for (const auto& item : items) {
    phc::TensorF32 t;
    if (!pp.ImageFileToTensorNchw(item.path, t)) {
      std::cerr << "Skip: " << item.path << "\n";
      continue;
    }
    phc::InferenceResult r = engine.Run(t);
    if (r.logits.size() < 2) {
      std::cerr << "Bad output size\n";
      continue;
    }
    const int pred = (r.logits[0] >= r.logits[1]) ? 0 : 1;
    const int y = item.label;
    if (y == 0 && pred == 0) {
      ++tn;
    } else if (y == 0 && pred == 1) {
      ++fp;
    } else if (y == 1 && pred == 0) {
      ++fn;
    } else {
      ++tp;
    }
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double total_sec =
      std::chrono::duration<double>(t1 - t0).count();

  const size_t n = static_cast<size_t>(tp + tn + fp + fn);
  if (n == 0) {
    std::cerr << "No successful inferences.\n";
    return 1;
  }

  std::cout << "Evaluated " << n << " images from " << fs::absolute(test_root) << "\n";
  PrintMetrics(tn, fp, fn, tp, total_sec, n);
  return 0;
}
