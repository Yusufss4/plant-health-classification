/**
 * Evaluate MobileNet ONNX on a folder layout matching evaluate.py / PlantHealthDataset:
 *   test_dir/healthy/    (.jpg|.png|.jpeg)   label 0
 *   test_dir/diseased/   (.jpg|.png|.jpeg)   label 1
 *   test_dir/background/ (.jpg|.png|.jpeg)   label 2   (optional - legacy 2-class
 *                                                       layouts still work)
 *
 * Usage:
 *   evaluate_mobilenet <model.onnx> [test_dir]
 *
 * Prints an NxN confusion matrix, accuracy, balanced accuracy, per-class
 * precision/recall/F1, macro-F1, and timing (load + preprocess + ORT per image).
 *
 * Class indices and names must match utils.data_loader.DEFAULT_CLASSES.
 */

#include "inference_ort/ort_engine.hpp"
#include "preprocess/mobilenet_preprocess.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Canonical class ordering. The runtime test set may omit some of these
// directories (e.g. legacy 2-class layouts); those labels get zero support but
// the confusion matrix and printed metrics still use the full k x k layout.
const std::vector<std::string>& ClassNames() {
  static const std::vector<std::string> kNames = {
      "healthy", "diseased", "background"};
  return kNames;
}

struct LabeledPath {
  std::string path;
  int label;
};

bool IsImageExt(const fs::path& p) {
  auto e = p.extension().string();
  for (char& c : e) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return e == ".jpg" || e == ".jpeg" || e == ".png";
}

std::vector<LabeledPath> CollectImages(const fs::path& test_root,
                                       std::vector<int>& present_labels_out) {
  std::vector<LabeledPath> out;
  present_labels_out.clear();
  const auto& names = ClassNames();
  for (size_t label = 0; label < names.size(); ++label) {
    const fs::path class_dir = test_root / names[label];
    if (!fs::is_directory(class_dir)) {
      continue;
    }
    bool added_any = false;
    for (const auto& entry : fs::directory_iterator(class_dir)) {
      if (!entry.is_regular_file() || !IsImageExt(entry.path())) {
        continue;
      }
      out.push_back({entry.path().string(), static_cast<int>(label)});
      added_any = true;
    }
    if (added_any) {
      present_labels_out.push_back(static_cast<int>(label));
    }
  }
  return out;
}

int Argmax(const std::vector<float>& v) {
  int best = 0;
  float best_v = v[0];
  for (size_t i = 1; i < v.size(); ++i) {
    if (v[i] > best_v) {
      best_v = v[i];
      best = static_cast<int>(i);
    }
  }
  return best;
}

void PrintMetrics(const std::vector<std::vector<int64_t>>& cm,
                  double total_sec, size_t n) {
  const auto& names = ClassNames();
  const size_t k = cm.size();
  const double n_d = static_cast<double>(n);

  int64_t correct = 0;
  for (size_t i = 0; i < k; ++i) {
    correct += cm[i][i];
  }
  const double acc = n_d > 0 ? static_cast<double>(correct) / n_d : 0.0;

  // Per-class precision, recall, F1, support.
  std::vector<double> prec(k, 0.0), rec(k, 0.0), f1(k, 0.0);
  std::vector<int64_t> support(k, 0);
  for (size_t c = 0; c < k; ++c) {
    int64_t tp = cm[c][c];
    int64_t row_sum = 0;  // actual count for class c (support)
    int64_t col_sum = 0;  // predicted count for class c
    for (size_t j = 0; j < k; ++j) {
      row_sum += cm[c][j];
      col_sum += cm[j][c];
    }
    support[c] = row_sum;
    prec[c] = col_sum > 0 ? static_cast<double>(tp) / static_cast<double>(col_sum) : 0.0;
    rec[c] = row_sum > 0 ? static_cast<double>(tp) / static_cast<double>(row_sum) : 0.0;
    f1[c] = (prec[c] + rec[c]) > 1e-12
                ? 2.0 * prec[c] * rec[c] / (prec[c] + rec[c])
                : 0.0;
  }

  // Macro averages (unweighted mean over classes with non-zero support so a
  // missing class directory doesn't drag the mean to zero).
  double macro_p = 0.0, macro_r = 0.0, macro_f = 0.0;
  size_t active = 0;
  for (size_t c = 0; c < k; ++c) {
    if (support[c] == 0) continue;
    macro_p += prec[c];
    macro_r += rec[c];
    macro_f += f1[c];
    ++active;
  }
  if (active > 0) {
    macro_p /= active;
    macro_r /= active;
    macro_f /= active;
  }

  // Balanced accuracy = mean per-class recall (over classes with support).
  double bal_acc = active > 0 ? macro_r : 0.0;

  std::cout << std::fixed << std::setprecision(4);
  std::cout
      << "\n============================================================\n";
  std::cout << "EVALUATION RESULTS (ONNX / C++)\n";
  std::cout << "============================================================\n";
  std::cout << "\nOverall metrics:\n";
  std::cout << "  Accuracy:          " << acc << " (" << acc * 100.0 << "%)\n";
  std::cout << "  Balanced accuracy: " << bal_acc << " (" << bal_acc * 100.0
            << "%)\n";
  std::cout << "  Macro precision:   " << macro_p << "\n";
  std::cout << "  Macro recall:      " << macro_r << "\n";
  std::cout << "  Macro F1-score:    " << macro_f << "\n";

  std::cout << "\nConfusion matrix (rows=actual, cols=predicted):\n";
  size_t col_w = 12;
  std::cout << std::string(col_w + 2, ' ');
  for (size_t j = 0; j < k; ++j) {
    std::cout << std::setw(static_cast<int>(col_w)) << names[j];
  }
  std::cout << "\n";
  for (size_t i = 0; i < k; ++i) {
    std::cout << std::setw(static_cast<int>(col_w)) << names[i] << "  ";
    for (size_t j = 0; j < k; ++j) {
      std::cout << std::setw(static_cast<int>(col_w)) << cm[i][j];
    }
    std::cout << "\n";
  }

  std::cout << "\nPer-class metrics:\n";
  std::cout << "  " << std::setw(12) << "class"
            << std::setw(12) << "precision"
            << std::setw(12) << "recall"
            << std::setw(12) << "f1"
            << std::setw(12) << "support" << "\n";
  for (size_t c = 0; c < k; ++c) {
    std::cout << "  " << std::setw(12) << names[c]
              << std::setw(12) << prec[c]
              << std::setw(12) << rec[c]
              << std::setw(12) << f1[c]
              << std::setw(12) << support[c] << "\n";
  }

  std::cout << "\nInference timing (load + preprocess + ORT, all " << n
            << " images):\n";
  std::cout << "  Total:     " << std::setprecision(3) << total_sec << " s\n";
  std::cout << "  Avg/image: " << (total_sec * 1000.0 / n_d) << " ms\n";
  std::cout << "  Throughput:" << (n_d / total_sec) << " img/s\n";
  std::cout << "============================================================\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr
        << "Usage: " << argv[0] << " <model.onnx> [test_dir]\n"
        << "  Default test_dir: data/test (relative to current directory)\n";
    return 1;
  }

  const std::string model_path = argv[1];
  fs::path test_root = (argc == 3) ? fs::path(argv[2]) : fs::path("data/test");
  if (!fs::is_directory(test_root)) {
    std::cerr << "Not a directory: " << test_root << "\n";
    return 1;
  }

  std::vector<int> present_labels;
  std::vector<LabeledPath> items = CollectImages(test_root, present_labels);
  if (items.empty()) {
    std::cerr << "No images found under any of: ";
    for (const auto& n : ClassNames()) std::cerr << n << "/ ";
    std::cerr << "\n";
    return 1;
  }
  std::cout << "Found classes on disk: ";
  for (int l : present_labels) std::cout << ClassNames()[l] << " ";
  std::cout << "\n";

  phc::MobilenetPreprocessor pp;
  phc::OrtInferenceEngine engine(model_path);

  // Warm-up (ORT thread pools, caches)
  if (pp.ImageFileToTensorInto(items[0].path, engine.input_data())) {
    (void)engine.Run();
  }

  const size_t k = ClassNames().size();
  std::vector<std::vector<int64_t>> cm(k, std::vector<int64_t>(k, 0));
  size_t evaluated = 0;
  bool warned_logit_mismatch = false;
  const auto t0 = std::chrono::high_resolution_clock::now();

  for (const auto& item : items) {
    if (!pp.ImageFileToTensorInto(item.path, engine.input_data())) {
      std::cerr << "Skip: " << item.path << "\n";
      continue;
    }
    const phc::InferenceResult& r = engine.Run();
    if (r.logits.size() < 2) {
      std::cerr << "Bad output size: " << r.logits.size() << "\n";
      continue;
    }
    if (!warned_logit_mismatch && r.logits.size() != k) {
      std::cerr << "Warning: model outputs " << r.logits.size()
                << " logits but " << k << " classes expected. Argmax uses all "
                << "outputs; predictions are clamped to [0, " << (k - 1)
                << "] for the confusion matrix.\n";
      warned_logit_mismatch = true;
    }
    const int pred = Argmax(r.logits);
    const int y = item.label;
    // pred may exceed k-1 if the model has more outputs than expected; clamp
    // into the confusion-matrix range so we don't write out of bounds.
    const int pred_clamped = std::max(0, std::min(pred, static_cast<int>(k) - 1));
    cm[y][pred_clamped] += 1;
    ++evaluated;
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double total_sec = std::chrono::duration<double>(t1 - t0).count();

  if (evaluated == 0) {
    std::cerr << "No successful inferences.\n";
    return 1;
  }

  std::cout << "Evaluated " << evaluated << " images from "
            << fs::absolute(test_root) << "\n";
  PrintMetrics(cm, total_sec, evaluated);
  return 0;
}
