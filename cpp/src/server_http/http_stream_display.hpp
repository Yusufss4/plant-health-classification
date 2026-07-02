#pragma once

#include "../app_runtime/interfaces.hpp"
#include "http_stream_server.hpp"

#include <memory>
#include <string>

namespace phc {

struct HttpStreamDisplayConfig {
  std::string bind_host = "0.0.0.0";
  int port = 8080;
  // JPEG quality (1..100). Lower is faster on the Pi Zero 2W and produces
  // smaller frames over the wire. 50 favors encode/FPS on device.
  int jpeg_quality = 50;
  // Minimum gap between published frames (across the network). 0 means
  // "publish as fast as the pipeline produces them".
  int min_publish_interval_ms = 0;
  int worker_threads = 4;
  int sse_keepalive_seconds = 15;
};

// IDisplay that JPEG-encodes each frame and pushes it (plus the inference
// result JSON) to an in-process HTTP server. Replaces FileArtifactDisplay:
// no on-disk artifacts, no external HTTP server, no per-frame fsync.
class HttpStreamDisplay final : public IDisplay {
 public:
  explicit HttpStreamDisplay(HttpStreamDisplayConfig cfg);
  ~HttpStreamDisplay() override;

  HttpStreamDisplay(const HttpStreamDisplay&) = delete;
  HttpStreamDisplay& operator=(const HttpStreamDisplay&) = delete;

  bool Init(int width, int height) override;
  bool Present(const Frame& rgb888, const InferenceResult& result) override;

  int bound_port() const;

 private:
  HttpStreamDisplayConfig cfg_;
  std::unique_ptr<HttpStreamServer> server_;
  int64_t last_publish_ns_ = 0;
};

}  // namespace phc
