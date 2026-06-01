#include "http_stream_server.hpp"

#include "embedded_index_html.hpp"

// cpp-httplib is a 20k-line single header. Including it from this single
// translation unit (instead of the .hpp) keeps build times tolerable.
#include "httplib.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
  oss << ']';
  oss << ",\"inference_ms\":" << r.inference_ms;
  oss << ",\"encode_ms\":" << r.encode_ms;
  oss << '}';
  return oss.str();
}

// Reads an entire small file (e.g. /proc/* or /sys/*) into a string. Returns
// nullopt if the file is missing or cannot be read.
std::optional<std::string> SlurpFile(const char* path) {
  std::ifstream in(path);
  if (!in) {
    return std::nullopt;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  if (!in && !in.eof()) {
    return std::nullopt;
  }
  return ss.str();
}

// Parses /proc/loadavg ("0.50 0.60 0.55 1/123 4567") and appends load_1m /
// load_5m / load_15m to the JSON object being built. `first` tracks whether
// a comma should be emitted before the next field.
void AppendLoadavg(std::ostringstream& oss, bool& first) {
  auto txt = SlurpFile("/proc/loadavg");
  if (!txt) return;
  std::istringstream is(*txt);
  double l1 = 0, l5 = 0, l15 = 0;
  if (!(is >> l1 >> l5 >> l15)) return;
  for (auto [name, value] : std::initializer_list<std::pair<const char*, double>>{
           {"load_1m", l1}, {"load_5m", l5}, {"load_15m", l15}}) {
    if (!first) oss << ',';
    oss << '"' << name << "\":" << value;
    first = false;
  }
}

// Parses MemTotal: and MemAvailable: lines out of /proc/meminfo and appends
// mem_total_kb, mem_available_kb, mem_used_pct.
void AppendMeminfo(std::ostringstream& oss, bool& first) {
  auto txt = SlurpFile("/proc/meminfo");
  if (!txt) return;
  uint64_t total_kb = 0;
  uint64_t avail_kb = 0;
  bool have_total = false;
  bool have_avail = false;
  std::istringstream is(*txt);
  std::string key;
  while (is >> key) {
    uint64_t value = 0;
    std::string unit;
    if (!(is >> value)) break;
    is >> unit;  // "kB" or empty; ignored
    if (key == "MemTotal:") {
      total_kb = value;
      have_total = true;
    } else if (key == "MemAvailable:") {
      avail_kb = value;
      have_avail = true;
    }
    if (have_total && have_avail) break;
  }
  if (have_total) {
    if (!first) oss << ',';
    oss << "\"mem_total_kb\":" << total_kb;
    first = false;
  }
  if (have_avail) {
    if (!first) oss << ',';
    oss << "\"mem_available_kb\":" << avail_kb;
    first = false;
  }
  if (have_total && have_avail && total_kb > 0) {
    const double used_pct =
        100.0 * (1.0 - static_cast<double>(avail_kb) /
                            static_cast<double>(total_kb));
    if (!first) oss << ',';
    oss << "\"mem_used_pct\":" << used_pct;
    first = false;
  }
}

// Parses /sys/class/thermal/thermal_zone0/temp (millidegrees C) and appends
// cpu_temp_c as a float in degrees Celsius.
void AppendThermal(std::ostringstream& oss, bool& first) {
  auto txt = SlurpFile("/sys/class/thermal/thermal_zone0/temp");
  if (!txt) return;
  std::istringstream is(*txt);
  int64_t milli_c = 0;
  if (!(is >> milli_c)) return;
  const double c = milli_c / 1000.0;
  if (!first) oss << ',';
  oss << "\"cpu_temp_c\":" << c;
  first = false;
}

// Parses the aggregate "cpu" line of /proc/stat into (idle_jiffies, total_jiffies).
// Returns false if the file is missing or malformed.
bool ReadProcStatAggregate(uint64_t& out_idle, uint64_t& out_total) {
  auto txt = SlurpFile("/proc/stat");
  if (!txt) return false;
  std::istringstream is(*txt);
  std::string label;
  if (!(is >> label) || label != "cpu") return false;
  // user nice system idle iowait irq softirq steal guest guest_nice
  std::vector<uint64_t> fields;
  uint64_t v = 0;
  while (is >> v) fields.push_back(v);
  if (fields.size() < 4) return false;
  // idle = idle + iowait (when present), per the kernel convention used by `top`.
  const uint64_t idle =
      fields[3] + (fields.size() > 4 ? fields[4] : 0);
  uint64_t total = 0;
  for (uint64_t f : fields) total += f;
  out_idle = idle;
  out_total = total;
  return true;
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

  auto serve_index = [](const httplib::Request&, httplib::Response& res) {
    res.set_header("Cache-Control", "no-store");
    res.set_content(std::string(kEmbeddedIndexHtml.data(),
                                 kEmbeddedIndexHtml.size()),
                    "text/html; charset=utf-8");
  };
  svr.Get("/", serve_index);
  svr.Get("/index.html", serve_index);

  svr.Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
    res.set_content("ok\n", "text/plain; charset=utf-8");
  });

  svr.Get("/metrics", [this](const httplib::Request&,
                              httplib::Response& res) {
    res.set_header("Cache-Control", "no-store");
    res.set_content(BuildMetricsJson(), "application/json");
  });

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

std::string HttpStreamServer::BuildMetricsJson() {
  std::ostringstream oss;
  oss << '{';
  bool first = true;

  AppendLoadavg(oss, first);
  AppendMeminfo(oss, first);
  AppendThermal(oss, first);

  // cpu_percent requires two snapshots, so we compute it across requests.
  uint64_t idle_now = 0;
  uint64_t total_now = 0;
  if (ReadProcStatAggregate(idle_now, total_now)) {
    std::lock_guard<std::mutex> lk(cpu_mu_);
    if (cpu_prev_.valid && total_now > cpu_prev_.total) {
      const uint64_t didle = idle_now - cpu_prev_.idle;
      const uint64_t dtotal = total_now - cpu_prev_.total;
      const double pct =
          100.0 * (1.0 - static_cast<double>(didle) /
                              static_cast<double>(dtotal));
      if (!first) oss << ',';
      oss << "\"cpu_percent\":" << pct;
      first = false;
    }
    cpu_prev_.idle = idle_now;
    cpu_prev_.total = total_now;
    cpu_prev_.valid = true;
  }

  oss << '}';
  return oss.str();
}

}  // namespace phc
