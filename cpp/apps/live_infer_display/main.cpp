// Live camera -> preprocess -> inference -> overlay -> DRM/KMS display.
//
// Build with:
//   cmake -DENABLE_LIBCAMERA=ON -DENABLE_DRMKMS=ON ...
//
// Run on Pi (needs /dev/dri/card0 and camera access):
//   ./live_infer_display /path/to/model.onnx

#include "../../src/app_runtime/live_pipeline.hpp"
#include "../../src/camera_libcamera/libcamera_camera.hpp"
#include "../../src/display_drm/drm_display.hpp"

#include <iostream>
#include <thread>
#include <chrono>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx>\n";
    return 1;
  }
  const std::string model_path = argv[1];

  phc::LibcameraCamera cam(phc::LibcameraConfig{});
  phc::DrmKmsDisplay disp(phc::DrmConfig{});
  phc::MobilenetPreprocessor pp;
  phc::OrtInferenceEngine engine(model_path);

  phc::LivePipelineConfig cfg;
  cfg.display_width = 640;
  cfg.display_height = 480;

  phc::LivePipeline pipeline(cam, disp, pp, engine, cfg);
  if (!pipeline.Start()) {
    std::cerr << "Failed to start pipeline\n";
    return 1;
  }

  std::cout << "Running. Press Ctrl+C to exit.\n";
  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  return 0;
}

