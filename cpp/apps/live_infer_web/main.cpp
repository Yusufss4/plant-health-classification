// Live camera -> preprocess -> inference -> in-process HTTP server that
// streams the preview as MJPEG and inference results over SSE.
//
// Build with:
//   cmake -DENABLE_LIBCAMERA=ON ...
//
// Run:
//   ./live_infer_web <model.onnx> [--port N] [--bind HOST] [--jpeg-quality Q]
//
// Then open http://<host>:<port>/ in a browser.

#include "../../src/app_runtime/live_pipeline.hpp"
#include "../../src/camera_libcamera/libcamera_camera.hpp"
#include "../../src/core/phc_log.hpp"
#include "../../src/inference_ort/ort_engine.hpp"
#include <onnxruntime_c_api.h>
#include "../../src/preprocess/mobilenet_preprocess.hpp"
#include "../../src/server_http/http_stream_display.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

namespace {

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " <model.onnx> [--port N] [--bind HOST] [--jpeg-quality Q]\n"
      << "       " << argv0 << " <model.onnx> <port>   (positional port, "
      << "kept for backward compat)\n";
}

bool ParseInt(const std::string& s, int& out) {
  try {
    size_t pos = 0;
    int v = std::stoi(s, &pos);
    if (pos != s.size()) {
      return false;
    }
    out = v;
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace

int main(int argc, char** argv) {
  phc::log::ConfigureThirdPartyLogLevels();

  if (argc < 2) {
    PrintUsage(argv[0]);
    return 1;
  }
  const std::string model_path = argv[1];

  phc::HttpStreamDisplayConfig disp_cfg;

  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto need_value = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        return nullptr;
      }
      return argv[++i];
    };
    if (a == "--port") {
      const char* v = need_value("--port");
      if (!v) return 1;
      if (!ParseInt(v, disp_cfg.port)) {
        std::cerr << "Invalid --port value: " << v << "\n";
        return 1;
      }
    } else if (a == "--bind") {
      const char* v = need_value("--bind");
      if (!v) return 1;
      disp_cfg.bind_host = v;
    } else if (a == "--jpeg-quality") {
      const char* v = need_value("--jpeg-quality");
      if (!v) return 1;
      if (!ParseInt(v, disp_cfg.jpeg_quality)) {
        std::cerr << "Invalid --jpeg-quality value: " << v << "\n";
        return 1;
      }
    } else if (a == "-h" || a == "--help") {
      PrintUsage(argv[0]);
      return 0;
    } else if (i == 2 && !a.empty() && a[0] != '-') {
      // Backward-compat: second positional arg is the port (the old
      // signature took an artifact directory here, but that's gone).
      int p = 0;
      if (!ParseInt(a, p)) {
        std::cerr << "Unknown argument: " << a << "\n";
        PrintUsage(argv[0]);
        return 1;
      }
      disp_cfg.port = p;
    } else {
      std::cerr << "Unknown argument: " << a << "\n";
      PrintUsage(argv[0]);
      return 1;
    }
  }

  phc::LibcameraCamera cam(phc::LibcameraConfig{});
  phc::HttpStreamDisplay disp(disp_cfg);
  phc::MobilenetPreprocessor pp;
  phc::OrtInferenceEngine::Options engine_opts;
  engine_opts.ort_log_level = ORT_LOGGING_LEVEL_ERROR;
  phc::OrtInferenceEngine engine(model_path, engine_opts);

  phc::LivePipelineConfig cfg;
  cfg.display_width = 640;
  cfg.display_height = 480;

  phc::LivePipeline pipeline(cam, disp, pp, engine, cfg);
  if (!pipeline.Start()) {
    phc::log::Error() << "Failed to start pipeline";
    return 1;
  }

  phc::log::Info() << "Live preview ready: http://" << disp_cfg.bind_host << ":"
                   << disp.bound_port() << "/";
  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  return 0;
}
