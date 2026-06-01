#pragma once

#include "../core/inference_result.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace phc {

struct HttpStreamServerConfig {
  std::string bind_host = "0.0.0.0";
  int port = 8080;
  // Cap concurrent worker threads. The Pi Zero 2W has four cores, but most of
  // them are busy with capture + inference; keep this small.
  int worker_threads = 4;
  // SSE keep-alive comment interval, sent when no new result arrives. Prevents
  // intermediaries (and some browsers) from closing an idle stream.
  int sse_keepalive_seconds = 15;
};

// In-process HTTP server that fans out the latest JPEG frame as MJPEG and the
// latest InferenceResult as Server-Sent Events. Producer threads call
// PublishFrame / PublishResult after each pipeline tick; the server holds only
// the most recent value, so slow subscribers naturally drop intermediate
// frames rather than backing up.
class HttpStreamServer {
 public:
  explicit HttpStreamServer(HttpStreamServerConfig cfg);
  ~HttpStreamServer();

  HttpStreamServer(const HttpStreamServer&) = delete;
  HttpStreamServer& operator=(const HttpStreamServer&) = delete;

  bool Start();
  void Stop();
  int bound_port() const { return bound_port_.load(); }

  void PublishFrame(std::vector<uint8_t> jpeg, uint64_t timestamp_ns);
  void PublishResult(const InferenceResult& result);

 private:
  // Forward-declare the impl to keep cpp-httplib out of the public header
  // (it's a 20k-line single header; including it transitively into every
  // translation unit that touches the server would balloon build times).
  struct Impl;
  std::unique_ptr<Impl> impl_;

  HttpStreamServerConfig cfg_;
  std::atomic<int> bound_port_{0};

  // State shared with content providers. Stored here (not in Impl) so we can
  // wake subscribers from Stop() without dragging httplib into the header.
  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<bool> stop_{false};

  struct LatestJpeg {
    std::shared_ptr<const std::vector<uint8_t>> bytes;
    uint64_t seq = 0;
    uint64_t timestamp_ns = 0;
  };
  struct LatestResult {
    std::shared_ptr<const std::string> json;
    uint64_t seq = 0;
  };
  LatestJpeg latest_jpeg_;
  LatestResult latest_result_;

  // Previous /proc/stat snapshot, used to compute cpu_percent across two
  // successive /metrics requests. Guarded by cpu_mu_ (separate from mu_ so
  // metrics polling cannot block the MJPEG/SSE producers).
  struct CpuStatSnapshot {
    uint64_t idle = 0;
    uint64_t total = 0;
    bool valid = false;
  };
  std::mutex cpu_mu_;
  CpuStatSnapshot cpu_prev_;

  std::thread listen_thread_;
  std::atomic<bool> running_{false};

  // /proc and /sys readers; cpu_percent needs two successive /proc/stat samples.
  std::string BuildMetricsJson();

  friend struct Impl;
};

}  // namespace phc
