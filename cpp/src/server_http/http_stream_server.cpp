#include "http_stream_server.hpp"

#include "embedded_index_html.hpp"

// cpp-httplib is a 20k-line single header. Including it from this single
// translation unit (instead of the .hpp) keeps build times tolerable.
#include "httplib.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <utility>

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

std::string ResultToJson(const InferenceResult& r) {
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
  return oss.str();
}

}  // namespace

struct HttpStreamServer::Impl {
  httplib::Server svr;
};

HttpStreamServer::HttpStreamServer(HttpStreamServerConfig cfg)
    : impl_(std::make_unique<Impl>()), cfg_(std::move(cfg)) {}

HttpStreamServer::~HttpStreamServer() {
  Stop();
}

bool HttpStreamServer::Start() {
  if (running_.exchange(true)) {
    return false;
  }
  stop_.store(false);

  auto& svr = impl_->svr;

  // Trim the worker pool: capture + inference already consume most cores
  // on the Pi Zero 2W, and each subscriber holds one worker thread for the
  // life of the connection.
  if (cfg_.worker_threads > 0) {
    const int n = cfg_.worker_threads;
    svr.new_task_queue = [n]() {
      return new httplib::ThreadPool(static_cast<size_t>(n));
    };
  }

  // GET / and /index.html: embedded HTML page.
  auto serve_index = [](const httplib::Request&, httplib::Response& res) {
    res.set_header("Cache-Control", "no-store");
    res.set_content(std::string(kEmbeddedIndexHtml.data(),
                                 kEmbeddedIndexHtml.size()),
                    "text/html; charset=utf-8");
  };
  svr.Get("/", serve_index);
  svr.Get("/index.html", serve_index);

  // GET /healthz: trivial liveness probe.
  svr.Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
    res.set_content("ok\n", "text/plain; charset=utf-8");
  });

  // GET /stream.mjpg: multipart MJPEG stream. Each invocation of the
  // provider blocks until a newer frame arrives, then writes one part.
  svr.Get("/stream.mjpg", [this](const httplib::Request&,
                                  httplib::Response& res) {
    res.set_header("Cache-Control", "no-store");
    res.set_header("Pragma", "no-cache");
    res.set_header("Connection", "close");

    auto last_seq = std::make_shared<uint64_t>(0);
    res.set_chunked_content_provider(
        "multipart/x-mixed-replace; boundary=phcframe",
        [this, last_seq](size_t /*offset*/,
                         httplib::DataSink& sink) -> bool {
          std::shared_ptr<const std::vector<uint8_t>> jpeg;
          {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] {
              return stop_.load() || latest_jpeg_.seq > *last_seq;
            });
            if (stop_.load()) {
              return false;
            }
            jpeg = latest_jpeg_.bytes;
            *last_seq = latest_jpeg_.seq;
          }
          if (!jpeg || jpeg->empty()) {
            return true;
          }

          std::string hdr;
          hdr.reserve(96);
          hdr.append("--phcframe\r\n");
          hdr.append("Content-Type: image/jpeg\r\n");
          hdr.append("Content-Length: ");
          hdr.append(std::to_string(jpeg->size()));
          hdr.append("\r\n\r\n");

          if (!sink.write(hdr.data(), hdr.size())) {
            return false;
          }
          if (!sink.write(reinterpret_cast<const char*>(jpeg->data()),
                          jpeg->size())) {
            return false;
          }
          if (!sink.write("\r\n", 2)) {
            return false;
          }
          return true;
        });
  });

  // GET /events: Server-Sent Events stream of inference results.
  svr.Get("/events", [this](const httplib::Request&,
                             httplib::Response& res) {
    res.set_header("Cache-Control", "no-store");
    res.set_header("Connection", "keep-alive");
    // Hint to nginx/Caddy not to buffer; harmless if absent.
    res.set_header("X-Accel-Buffering", "no");

    auto last_seq = std::make_shared<uint64_t>(0);
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, last_seq](size_t /*offset*/,
                         httplib::DataSink& sink) -> bool {
          std::shared_ptr<const std::string> payload;
          {
            std::unique_lock<std::mutex> lk(mu_);
            const auto timeout =
                std::chrono::seconds(cfg_.sse_keepalive_seconds > 0
                                         ? cfg_.sse_keepalive_seconds
                                         : 15);
            const bool got = cv_.wait_for(lk, timeout, [&] {
              return stop_.load() || latest_result_.seq > *last_seq;
            });
            if (stop_.load()) {
              return false;
            }
            if (!got) {
              // Idle keepalive: a comment line is ignored by EventSource
              // but keeps the TCP/HTTP connection alive through proxies.
              static constexpr std::string_view kPing = ": ping\n\n";
              return sink.write(kPing.data(), kPing.size());
            }
            payload = latest_result_.json;
            *last_seq = latest_result_.seq;
          }
          if (!payload) {
            return true;
          }
          std::string msg;
          msg.reserve(payload->size() + 8);
          msg.append("data: ");
          msg.append(*payload);
          msg.append("\n\n");
          return sink.write(msg.data(), msg.size());
        });
  });

  // Bind synchronously so callers see bind failures and learn the real port.
  int port = -1;
  if (cfg_.port == 0) {
    port = svr.bind_to_any_port(cfg_.bind_host);
    if (port <= 0) {
      running_.store(false);
      return false;
    }
  } else {
    if (!svr.bind_to_port(cfg_.bind_host, cfg_.port)) {
      running_.store(false);
      return false;
    }
    port = cfg_.port;
  }
  bound_port_.store(port);

  listen_thread_ = std::thread([this]() {
    if (!impl_->svr.listen_after_bind()) {
      // listen() returns false on shutdown; nothing actionable here.
    }
  });
  return true;
}

void HttpStreamServer::Stop() {
  if (!running_.exchange(false)) {
    return;
  }
  stop_.store(true);
  cv_.notify_all();
  impl_->svr.stop();
  if (listen_thread_.joinable()) {
    listen_thread_.join();
  }
  bound_port_.store(0);
}

void HttpStreamServer::PublishFrame(std::vector<uint8_t> jpeg,
                                    uint64_t timestamp_ns) {
  auto shared = std::make_shared<const std::vector<uint8_t>>(std::move(jpeg));
  {
    std::lock_guard<std::mutex> lk(mu_);
    latest_jpeg_.bytes = std::move(shared);
    latest_jpeg_.timestamp_ns = timestamp_ns;
    latest_jpeg_.seq += 1;
  }
  cv_.notify_all();
}

void HttpStreamServer::PublishResult(const InferenceResult& result) {
  auto shared = std::make_shared<const std::string>(ResultToJson(result));
  {
    std::lock_guard<std::mutex> lk(mu_);
    latest_result_.json = std::move(shared);
    latest_result_.seq += 1;
  }
  cv_.notify_all();
}

}  // namespace phc
