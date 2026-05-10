// Live camera -> preprocess -> inference -> JPEG + JSON artifacts for static web UI.
//
// Build with:
//   cmake -DENABLE_LIBCAMERA=ON ...
//
// Serve the artifact directory (and copy web/live/index.html there) with e.g.:
//   python3 -m http.server 8080 --bind 0.0.0.0 --directory /path/to/artifacts

#include "../../src/app_runtime/live_pipeline.hpp"
#include "../../src/camera_libcamera/libcamera_camera.hpp"
#include "../../src/display_file/file_artifact_display.hpp"
#include "../../src/inference_ort/ort_engine.hpp"
#include "../../src/preprocess/mobilenet_preprocess.hpp"

#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> [artifact_dir]\n";
    return 1;
  }
  const std::string model_path = argv[1];

  phc::FileArtifactDisplayConfig artifact_cfg;
  if (argc >= 3) {
    artifact_cfg.output_dir = argv[2];
  }

  phc::LibcameraCamera cam(phc::LibcameraConfig{});
  phc::FileArtifactDisplay disp(artifact_cfg);
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

  std::cout << "Writing preview + JSON to: " << artifact_cfg.output_dir << "\n";
  std::cout << "Running. Press Ctrl+C to exit.\n";
  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  return 0;
}
