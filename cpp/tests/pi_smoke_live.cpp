#include <catch2/catch_test_macros.hpp>

#if defined(PHC_HAVE_LIBCAMERA) && defined(PHC_HAVE_DRM)

#include "../src/app_runtime/live_pipeline.hpp"
#include "../src/camera_libcamera/libcamera_camera.hpp"
#include "../src/display_drm/drm_display.hpp"

#include <chrono>
#include <thread>

TEST_CASE("Pi smoke: start live pipeline briefly", "[pi][smoke]") {
  // This test is intended to run on the Raspberry Pi with camera+DRM available.
  // It simply checks we can start the pipeline and run briefly without crashing.
  const std::string model_path = std::getenv("PHC_MODEL_PATH") ? std::getenv("PHC_MODEL_PATH") : "";
  REQUIRE_FALSE(model_path.empty());

  phc::LibcameraCamera cam(phc::LibcameraConfig{});
  phc::DrmKmsDisplay disp(phc::DrmConfig{});
  phc::MobilenetPreprocessor pp;
  phc::OrtInferenceEngine engine(model_path);

  phc::LivePipelineConfig cfg;
  cfg.display_width = 640;
  cfg.display_height = 480;
  cfg.overlay_enabled = true;

  phc::LivePipeline pipeline(cam, disp, pp, engine, cfg);
  REQUIRE(pipeline.Start());
  std::this_thread::sleep_for(std::chrono::seconds(2));
  pipeline.Stop();
  SUCCEED();
}

#else

TEST_CASE("Pi smoke: skipped (no libcamera/DRM)", "[pi][smoke]") { SUCCEED(); }

#endif

