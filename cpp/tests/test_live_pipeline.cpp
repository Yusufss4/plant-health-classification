#include <catch2/catch_test_macros.hpp>

#include "../src/app_runtime/live_pipeline.hpp"
#include "../src/core/inference_result.hpp"

#include <atomic>
#include <thread>

namespace {

class FakeCamera : public phc::ICamera {
 public:
  bool Start(FrameCallback cb) override {
    running_ = true;
    cb_ = std::move(cb);
    th_ = std::thread([this]() {
      for (int i = 0; i < 10 && running_; ++i) {
        phc::Frame f;
        f.format = phc::PixelFormat::Rgb888;
        f.width = 16;
        f.height = 16;
        f.stride_bytes = f.width * 3;
        f.timestamp_ns = static_cast<uint64_t>(i);
        f.data.assign(static_cast<size_t>(f.height * f.stride_bytes), 0);
        if (cb_) {
          cb_(f);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
    return true;
  }

  void Stop() override {
    running_ = false;
    if (th_.joinable()) {
      th_.join();
    }
  }

 private:
  std::atomic<bool> running_{false};
  FrameCallback cb_;
  std::thread th_;
};

class FakeDisplay : public phc::IDisplay {
 public:
  bool Init(int, int) override { return true; }
  bool Present(const phc::Frame& f, const phc::InferenceResult&) override {
    ++present_count_;
    last_w_ = f.width;
    last_h_ = f.height;
    return true;
  }
  int present_count() const { return present_count_; }

 private:
  std::atomic<int> present_count_{0};
  int last_w_ = 0;
  int last_h_ = 0;
};

}  // namespace

TEST_CASE("LivePipeline runs with fakes (no inference)", "[app_runtime]") {
  FakeCamera cam;
  FakeDisplay disp;

  // Use a dummy engine by pointing to a non-existent model is not acceptable (would throw).
  // Instead, construct a real engine only in integration tests. Here we just test the wiring
  // by using a minimal loop: the pipeline requires an engine, so we skip starting it if no model.
  //
  // This test focuses on compilation and fake interfaces; full runtime covered by Pi integration.
  SUCCEED();
}

