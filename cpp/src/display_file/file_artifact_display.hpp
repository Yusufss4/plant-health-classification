#pragma once

#include "../app_runtime/interfaces.hpp"

#include <string>

namespace phc {

struct FileArtifactDisplayConfig {
  // Directory where preview.jpg and result.json are written (created if missing).
  std::string output_dir = ".";
  std::string image_filename = "preview.jpg";
  std::string json_filename = "result.json";
  int jpeg_quality = 85;
  // Minimum time between published artifacts; inference still runs every frame.
  int min_publish_interval_ms = 250;
};

// Writes RGB preview as JPEG plus JSON metadata for a static web UI polling the same directory.
class FileArtifactDisplay final : public IDisplay {
 public:
  explicit FileArtifactDisplay(FileArtifactDisplayConfig cfg);
  ~FileArtifactDisplay() override = default;

  bool Init(int width, int height) override;
  bool Present(const Frame& rgb888, const InferenceResult& result) override;

 private:
  FileArtifactDisplayConfig cfg_;
  int64_t last_publish_ns_ = 0;
};

}  // namespace phc
