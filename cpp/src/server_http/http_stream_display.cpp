#include "http_stream_display.hpp"

#include "../core/inference_result.hpp"
#include "../core/phc_log.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

namespace phc {

namespace {

void StbWriteToVector(void* ctx, void* data, int size) {
  if (size <= 0) {
    return;
  }
  auto* out = static_cast<std::vector<uint8_t>*>(ctx);
  const auto* p = static_cast<const uint8_t*>(data);
  out->insert(out->end(), p, p + static_cast<size_t>(size));
}

}  // namespace

HttpStreamDisplay::HttpStreamDisplay(HttpStreamDisplayConfig cfg)
    : cfg_(std::move(cfg)) {
  HttpStreamServerConfig sc;
  sc.bind_host = cfg_.bind_host;
  sc.port = cfg_.port;
  sc.worker_threads = cfg_.worker_threads;
  sc.sse_keepalive_seconds = cfg_.sse_keepalive_seconds;
  server_ = std::make_unique<HttpStreamServer>(std::move(sc));
}

HttpStreamDisplay::~HttpStreamDisplay() {
  if (server_) {
    server_->Stop();
  }
}

bool HttpStreamDisplay::Init(int width, int height) {
  (void)width;
  (void)height;
  last_publish_ns_ = 0;
  if (!server_->Start()) {
    log::Error() << "HttpStreamDisplay: failed to bind HTTP server on "
                 << cfg_.bind_host << ":" << cfg_.port;
    return false;
  }
  log::Debug() << "HttpStreamDisplay: listening on http://" << cfg_.bind_host
               << ":" << server_->bound_port() << "/";
  return true;
}

int HttpStreamDisplay::bound_port() const {
  return server_ ? server_->bound_port() : 0;
}

bool HttpStreamDisplay::Present(const Frame& rgb888,
                                const InferenceResult& result) {
  if (rgb888.format != PixelFormat::Rgb888 || rgb888.empty()) {
    return false;
  }

  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  const int64_t now_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
  if (cfg_.min_publish_interval_ms > 0 && last_publish_ns_ != 0) {
    const int64_t min_ns =
        static_cast<int64_t>(cfg_.min_publish_interval_ms) * 1000000LL;
    if (now_ns - last_publish_ns_ < min_ns) {
      return true;
    }
  }

  const int q = std::max(1, std::min(100, cfg_.jpeg_quality));
  const int w = rgb888.width;
  const int h = rgb888.height;
  const int packed_stride = w * 3;
  const uint8_t* pixels = rgb888.data.data();
  std::vector<uint8_t> packed;
  if (rgb888.stride_bytes != packed_stride) {
    packed.assign(static_cast<size_t>(packed_stride) * static_cast<size_t>(h),
                  0);
    for (int y = 0; y < h; ++y) {
      std::memcpy(
          packed.data() +
              static_cast<size_t>(y) * static_cast<size_t>(packed_stride),
          rgb888.data.data() +
              static_cast<size_t>(y) * static_cast<size_t>(rgb888.stride_bytes),
          static_cast<size_t>(packed_stride));
    }
    pixels = packed.data();
  }

  std::vector<uint8_t> jpeg;
  // Reasonable upper-bound reservation: ~5% of raw size is a typical
  // headroom for quality 55 on natural images; expanding on demand is fine.
  jpeg.reserve(static_cast<size_t>(packed_stride) *
               static_cast<size_t>(h) / 16);
  const auto enc_t0 = std::chrono::steady_clock::now();
  if (!stbi_write_jpg_to_func(StbWriteToVector, &jpeg, w, h, 3,
                              static_cast<const void*>(pixels), q)) {
    return false;
  }
  const auto enc_t1 = std::chrono::steady_clock::now();

  const uint64_t ts =
      rgb888.timestamp_ns != 0 ? rgb888.timestamp_ns
                               : static_cast<uint64_t>(now_ns);
  server_->PublishFrame(std::move(jpeg), ts);

  // Stamp encode_ms on a local copy so the const& parameter contract is kept
  // and the rest of the result (which the LivePipeline owns) is unchanged.
  InferenceResult annotated = result;
  annotated.encode_ms =
      std::chrono::duration<float, std::milli>(enc_t1 - enc_t0).count();
  server_->PublishResult(annotated);

  last_publish_ns_ = now_ns;
  return true;
}

}  // namespace phc
