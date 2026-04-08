#include "live_pipeline.hpp"

#include <chrono>

namespace phc {

LivePipeline::LivePipeline(ICamera& camera, IDisplay& display,
                           MobilenetPreprocessor preprocess,
                           OrtInferenceEngine& engine, LivePipelineConfig cfg)
    : camera_(camera),
      display_(display),
      preprocess_(std::move(preprocess)),
      engine_(engine),
      cfg_(cfg) {}

LivePipeline::~LivePipeline() {
  Stop();
}

bool LivePipeline::Start() {
  if (running_.exchange(true)) {
    return false;
  }
  if (!display_.Init(cfg_.display_width, cfg_.display_height)) {
    running_ = false;
    return false;
  }

  worker_ = std::thread([this]() { WorkerLoop(); });

  const bool ok = camera_.Start([this](const Frame& f) { OnFrame(f); });
  if (!ok) {
    running_ = false;
    if (worker_.joinable()) {
      worker_.join();
    }
    return false;
  }
  return true;
}

void LivePipeline::Stop() {
  if (!running_.exchange(false)) {
    return;
  }
  camera_.Stop();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void LivePipeline::OnFrame(const Frame& frame) {
  std::lock_guard<std::mutex> lk(mu_);
  latest_frame_ = frame;  // copy; drop older frames to keep preview responsive
}

void LivePipeline::WorkerLoop() {
  while (running_) {
    std::optional<Frame> frame;
    {
      std::lock_guard<std::mutex> lk(mu_);
      frame.swap(latest_frame_);
    }
    if (!frame.has_value()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    if (frame->format == PixelFormat::Rgb888) {
      TensorF32 t;
      if (preprocess_.Run(*frame, t)) {
        try {
          latest_result_ = engine_.Run(t, frame->timestamp_ns);
        } catch (...) {
          // Keep last result.
        }
      }
      (void)display_.Present(*frame, latest_result_);
    }
  }
}

}  // namespace phc
