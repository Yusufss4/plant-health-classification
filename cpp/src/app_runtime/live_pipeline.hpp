#pragma once

#include "../core/inference_result.hpp"
#include "../preprocess/mobilenet_preprocess.hpp"
#include "../inference_ort/ort_engine.hpp"
#include "interfaces.hpp"

#include <atomic>
#include <mutex>
#include <optional>
#include <thread>

namespace phc {

struct LivePipelineConfig {
  int display_width = 640;
  int display_height = 480;
};

class LivePipeline {
 public:
  LivePipeline(ICamera& camera, IDisplay& display, MobilenetPreprocessor preprocess, OrtInferenceEngine engine, LivePipelineConfig cfg);
  ~LivePipeline();

  bool Start();
  void Stop();

 private:
  void OnFrame(const Frame& frame);
  void WorkerLoop();

  ICamera& camera_;
  IDisplay& display_;
  MobilenetPreprocessor preprocess_;
  OrtInferenceEngine engine_;
  LivePipelineConfig cfg_;

  std::atomic<bool> running_{false};
  std::thread worker_;

  std::mutex mu_;
  std::optional<Frame> latest_frame_;
  InferenceResult latest_result_;
};

}  // namespace phc

