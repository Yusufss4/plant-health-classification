#include <catch2/catch_test_macros.hpp>

#include "app_runtime/live_pipeline.hpp"
#include "camera_libcamera/libcamera_camera.hpp"
#include "inference_ort/ort_engine.hpp"
#include "preprocess/mobilenet_preprocess.hpp"
#include "server_http/http_stream_display.hpp"

#include <chrono>
#include <cstdlib>
#include <thread>

TEST_CASE("Pi smoke: start live pipeline briefly", "[pi][smoke]") {
  // Raspberry Pi with camera. Requires PHC_MODEL_PATH. Binds the HTTP server
  // to an ephemeral port so the test never collides with a running daemon.
  const std::string model_path =
      std::getenv("PHC_MODEL_PATH") ? std::getenv("PHC_MODEL_PATH") : "";
  REQUIRE_FALSE(model_path.empty());

  phc::LibcameraCamera cam(phc::LibcameraConfig{});
  phc::HttpStreamDisplayConfig disp_cfg;
  disp_cfg.bind_host = "127.0.0.1";
  disp_cfg.port = 0;  // ephemeral
  phc::HttpStreamDisplay disp(disp_cfg);
  phc::MobilenetPreprocessor pp;
  phc::OrtInferenceEngine engine(model_path);

  phc::LivePipelineConfig cfg;
  cfg.display_width = 640;
  cfg.display_height = 480;

  phc::LivePipeline pipeline(cam, disp, pp, engine, cfg);
  REQUIRE(pipeline.Start());
  REQUIRE(disp.bound_port() > 0);
  std::this_thread::sleep_for(std::chrono::seconds(2));
  pipeline.Stop();
  SUCCEED();
}
