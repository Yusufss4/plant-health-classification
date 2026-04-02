#include "file_artifact_display.hpp"

#include "../core/inference_result.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace phc {

namespace {

std::string JsonEscape(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
    }
  }
  return out;
}

bool FsyncFile(const fs::path& p) {
  const int fd = ::open(p.c_str(), O_RDONLY);
  if (fd < 0) {
    return false;
  }
  (void)::fsync(fd);
  (void)::close(fd);
  return true;
}

bool WriteJsonFile(const fs::path& tmp, const fs::path& final, const InferenceResult& r) {
  std::ostringstream oss;
  oss << '{';
  oss << "\"timestamp_ns\":" << r.timestamp_ns << ',';
  oss << "\"label\":" << r.label << ',';
  oss << "\"confidence\":" << std::setprecision(9) << r.confidence << ',';
  oss << "\"label_name\":\"" << JsonEscape(r.label_name) << "\",";
  oss << "\"logits\":[";
  for (size_t i = 0; i < r.logits.size(); ++i) {
    if (i) {
      oss << ',';
    }
    oss << r.logits[i];
  }
  oss << "],\"probabilities\":[";
  for (size_t i = 0; i < r.probabilities.size(); ++i) {
    if (i) {
      oss << ',';
    }
    oss << r.probabilities[i];
  }
  oss << "]}";

  const std::string content = oss.str();
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
      return false;
    }
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!out.flush()) {
      return false;
    }
  }
  if (!FsyncFile(tmp)) {
    return false;
  }
  std::error_code ec;
  fs::rename(tmp, final, ec);
  return !ec;
}

}  // namespace

FileArtifactDisplay::FileArtifactDisplay(FileArtifactDisplayConfig cfg) : cfg_(std::move(cfg)) {}

bool FileArtifactDisplay::Init(int width, int height) {
  (void)width;
  (void)height;
  last_publish_ns_ = 0;
  std::error_code ec;
  fs::create_directories(cfg_.output_dir, ec);
  if (ec) {
    std::cerr << "FileArtifactDisplay: cannot create output_dir: " << cfg_.output_dir << " — " << ec.message() << "\n";
    return false;
  }
  return true;
}

bool FileArtifactDisplay::Present(const Frame& rgb888, const InferenceResult& result) {
  if (rgb888.format != PixelFormat::Rgb888 || rgb888.empty()) {
    return false;
  }

  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
  if (cfg_.min_publish_interval_ms > 0 && last_publish_ns_ != 0) {
    const int64_t min_ns = static_cast<int64_t>(cfg_.min_publish_interval_ms) * 1000000LL;
    if (now_ns - last_publish_ns_ < min_ns) {
      return true;
    }
  }

  const fs::path dir(cfg_.output_dir);
  const fs::path image_final = dir / cfg_.image_filename;
  const fs::path json_final = dir / cfg_.json_filename;
  const fs::path image_tmp = dir / (cfg_.image_filename + ".tmp");
  const fs::path json_tmp = dir / (cfg_.json_filename + ".tmp");

  const int q = std::max(1, std::min(100, cfg_.jpeg_quality));
  const int w = rgb888.width;
  const int h = rgb888.height;
  const int packed_stride = w * 3;
  const uint8_t* pixels = rgb888.data.data();
  std::vector<uint8_t> packed;
  if (rgb888.stride_bytes != packed_stride) {
    packed.assign(static_cast<size_t>(packed_stride * h), 0);
    for (int y = 0; y < h; ++y) {
      std::memcpy(packed.data() + static_cast<size_t>(y) * static_cast<size_t>(packed_stride),
                  rgb888.data.data() + static_cast<size_t>(y) * static_cast<size_t>(rgb888.stride_bytes),
                  static_cast<size_t>(packed_stride));
    }
    pixels = packed.data();
  }
  if (!stbi_write_jpg(image_tmp.string().c_str(), w, h, 3, static_cast<const void*>(pixels), q)) {
    return false;
  }
  if (!FsyncFile(image_tmp)) {
    return false;
  }
  std::error_code ec;
  fs::rename(image_tmp, image_final, ec);
  if (ec) {
    return false;
  }

  if (!WriteJsonFile(json_tmp, json_final, result)) {
    return false;
  }

  last_publish_ns_ = now_ns;
  return true;
}

}  // namespace phc
