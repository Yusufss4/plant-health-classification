#include <catch2/catch_test_macros.hpp>

#include "app_runtime/live_pipeline.hpp"
#include "camera_libcamera/libcamera_camera.hpp"
#include "display_file/file_artifact_display.hpp"
#include "inference_ort/ort_engine.hpp"
#include "preprocess/mobilenet_preprocess.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <unistd.h>

TEST_CASE("Pi smoke: start live pipeline briefly", "[pi][smoke]") {
  // Raspberry Pi with camera. Requires PHC_MODEL_PATH. Writes artifacts to a temp dir (no display).
  const std::string model_path = std::getenv("PHC_MODEL_PATH") ? std::getenv("PHC_MODEL_PATH") : "";
  REQUIRE_FALSE(model_path.empty());

  const std::filesystem::path out =
      std::filesystem::temp_directory_path() / ("phc_smoke_" + std::to_string(::getpid()));
  std::filesystem::create_directories(out);

  phc::LibcameraCamera cam(phc::LibcameraConfig{});
  phc::FileArtifactDisplayConfig artifact_cfg;
  artifact_cfg.output_dir = out.string();
  artifact_cfg.min_publish_interval_ms = 100;
  phc::FileArtifactDisplay disp(artifact_cfg);
  phc::MobilenetPreprocessor pp;
  phc::OrtInferenceEngine engine(model_path);

  phc::LivePipelineConfig cfg;
  cfg.display_width = 640;
  cfg.display_height = 480;

  phc::LivePipeline pipeline(cam, disp, pp, engine, cfg);
  REQUIRE(pipeline.Start());
  std::this_thread::sleep_for(std::chrono::seconds(2));
  pipeline.Stop();

  std::error_code ec;
  std::filesystem::remove_all(out, ec);
  SUCCEED();
}
