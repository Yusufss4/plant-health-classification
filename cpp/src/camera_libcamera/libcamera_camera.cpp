#include "libcamera_camera.hpp"

#include "../core/rgb_normalize.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>

#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/control_ids.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>

#include <sys/mman.h>
#include <unistd.h>

namespace phc {

namespace {

constexpr bool kForceLiveRbSwap = true;

void ApplyRequestedFocusProfile(libcamera::Request* request) {
  if (!request) {
    return;
  }
  auto& controls = request->controls();
  controls.set(libcamera::controls::AfMode,
               libcamera::controls::AfModeContinuous);
  controls.set(libcamera::controls::AfRange,
               libcamera::controls::AfRangeMacro);
  controls.set(libcamera::controls::AfSpeed,
               libcamera::controls::AfSpeedFast);
}

const char* PackedOrderName(PackedRgbOrder order) {
  return (order == PackedRgbOrder::Rgb) ? "RGB" : "BGR";
}

bool ResolvePackedFormat(const libcamera::PixelFormat& pf, int& bpp,
                         PackedRgbOrder& order) {
  if (pf == libcamera::formats::RGB888) {
    bpp = 3;
    order = PackedRgbOrder::Rgb;
    return true;
  }
  if (pf == libcamera::formats::BGR888) {
    bpp = 3;
    order = PackedRgbOrder::Bgr;
    return true;
  }
  return false;
}

void LogStreamConfig(const char* stage, const libcamera::StreamConfiguration& sc) {
  std::cerr << "libcamera " << stage << ": format=" << sc.pixelFormat.toString()
            << " size=" << sc.size.width << "x" << sc.size.height
            << " stride=" << sc.stride << "\n";
}

}  // namespace

struct LibcameraCamera::Impl {
  FrameCallback cb;
  std::atomic<bool> running{false};

  std::unique_ptr<libcamera::CameraManager> cm;
  std::shared_ptr<libcamera::Camera> camera;
  std::unique_ptr<libcamera::CameraConfiguration> config;
  std::unique_ptr<libcamera::FrameBufferAllocator> allocator;
  libcamera::Stream* stream = nullptr;
  std::vector<std::unique_ptr<libcamera::Request>> requests;
  int src_bpp = 3;
  PackedRgbOrder src_order = PackedRgbOrder::Rgb;
  std::mutex mu;

  uint64_t NowNs() const {
    using Clock = std::chrono::steady_clock;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now().time_since_epoch())
                        .count();
    return static_cast<uint64_t>(ns);
  }
};

void LibcameraCamera::OnRequestCompleted(libcamera::Request* request) {
  if (!impl_ || !impl_->running) {
    return;
  }
  if (request->status() == libcamera::Request::RequestCancelled) {
    return;
  }

  const auto& bufs = request->buffers();
  auto it = bufs.find(impl_->stream);
  if (it == bufs.end()) {
    ApplyRequestedFocusProfile(request);
    request->reuse(libcamera::Request::ReuseBuffers);
    impl_->camera->queueRequest(request);
    return;
  }
  const libcamera::FrameBuffer* fb = it->second;
  if (fb->planes().empty()) {
    ApplyRequestedFocusProfile(request);
    request->reuse(libcamera::Request::ReuseBuffers);
    impl_->camera->queueRequest(request);
    return;
  }

  // Plane 0 for packed RGB-family output.
  const int fd = fb->planes()[0].fd.get();
  const size_t length = fb->planes()[0].length;
  void* map = ::mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0);
  if (map == MAP_FAILED) {
    ApplyRequestedFocusProfile(request);
    request->reuse(libcamera::Request::ReuseBuffers);
    impl_->camera->queueRequest(request);
    return;
  }

  Frame f;
  f.format = PixelFormat::Rgb888;
  f.width = static_cast<int>(impl_->config->at(0).size.width);
  f.height = static_cast<int>(impl_->config->at(0).size.height);
  // libcamera's stride is accessible via StreamConfiguration::stride (bytes per line)
  f.stride_bytes = f.width * 3;
  f.timestamp_ns = impl_->NowNs();

  const int src_stride_bytes = static_cast<int>(impl_->config->at(0).stride);
  const bool ok = NormalizePackedToRgb888(
      static_cast<const uint8_t*>(map), f.width, f.height, src_stride_bytes,
      impl_->src_bpp, impl_->src_order, f.data, f.stride_bytes);
  ::munmap(map, length);
  if (!ok) {
    std::cerr << "libcamera: failed to normalize frame to RGB888"
              << " src_stride=" << src_stride_bytes
              << " src_bpp=" << impl_->src_bpp
              << " src_order=" << PackedOrderName(impl_->src_order)
              << " dst=" << f.width << "x" << f.height << "\n";
    ApplyRequestedFocusProfile(request);
    request->reuse(libcamera::Request::ReuseBuffers);
    impl_->camera->queueRequest(request);
    return;
  }

  if (impl_->cb) {
    impl_->cb(std::move(f));
  }

  ApplyRequestedFocusProfile(request);
  request->reuse(libcamera::Request::ReuseBuffers);
  impl_->camera->queueRequest(request);
}

LibcameraCamera::LibcameraCamera(LibcameraConfig cfg)
    : cfg_(cfg), impl_(new Impl()) {}

LibcameraCamera::~LibcameraCamera() {
  Stop();
  delete impl_;
  impl_ = nullptr;
}

bool LibcameraCamera::Start(FrameCallback cb) {
  if (!impl_) {
    return false;
  }
  if (impl_->running.exchange(true)) {
    return false;
  }
  impl_->cb = std::move(cb);

  impl_->cm = std::make_unique<libcamera::CameraManager>();
  if (impl_->cm->start()) {
    std::cerr << "libcamera CameraManager start failed\n";
    impl_->running = false;
    return false;
  }
  if (impl_->cm->cameras().empty()) {
    std::cerr << "No libcamera cameras found\n";
    impl_->cm->stop();
    impl_->running = false;
    return false;
  }
  impl_->camera = impl_->cm->cameras().front();
  if (impl_->camera->acquire()) {
    std::cerr << "Camera acquire failed\n";
    impl_->cm->stop();
    impl_->running = false;
    return false;
  }

  // Configure single stream (viewfinder).
  impl_->config =
      impl_->camera->generateConfiguration({libcamera::StreamRole::Viewfinder});
  if (!impl_->config || impl_->config->empty()) {
    std::cerr << "generateConfiguration failed\n";
    impl_->camera->release();
    impl_->cm->stop();
    impl_->running = false;
    return false;
  }

  auto& sc = impl_->config->at(0);
  sc.size.width = static_cast<unsigned int>(cfg_.width);
  sc.size.height = static_cast<unsigned int>(cfg_.height);
  sc.pixelFormat = libcamera::formats::RGB888;
  sc.bufferCount = 4;

  const libcamera::CameraConfiguration::Status v = impl_->config->validate();
  LogStreamConfig("post-validate", sc);
  if (v == libcamera::CameraConfiguration::Invalid) {
    std::cerr << "Camera configuration invalid\n";
    impl_->camera->release();
    impl_->cm->stop();
    impl_->running = false;
    return false;
  }

  if (impl_->camera->configure(impl_->config.get())) {
    std::cerr << "Camera configure failed\n";
    impl_->camera->release();
    impl_->cm->stop();
    impl_->running = false;
    return false;
  }

  LogStreamConfig("post-configure", impl_->config->at(0));
  int src_bpp = 0;
  PackedRgbOrder src_order = PackedRgbOrder::Rgb;
  if (!ResolvePackedFormat(impl_->config->at(0).pixelFormat, src_bpp, src_order)) {
    std::cerr << "Unsupported libcamera pixel format negotiated: "
              << impl_->config->at(0).pixelFormat.toString()
              << ". Expected RGB888 or BGR888. "
              << "If your pipeline negotiates a 4-byte format, add explicit "
              << "normalization before emitting Frame::Rgb888.\n";
    impl_->camera->release();
    impl_->cm->stop();
    impl_->running = false;
    return false;
  }
  impl_->src_bpp = src_bpp;
  impl_->src_order = kForceLiveRbSwap ? PackedRgbOrder::Bgr : src_order;
  if (kForceLiveRbSwap) {
    std::cerr << "libcamera live color fix active: forcing source channel order to "
              << PackedOrderName(impl_->src_order)
              << " before RGB888 normalization"
              << " (negotiated order was " << PackedOrderName(src_order) << ")\n";
  }

  impl_->stream = sc.stream();
  impl_->allocator =
      std::make_unique<libcamera::FrameBufferAllocator>(impl_->camera);
  if (impl_->allocator->allocate(impl_->stream) < 0) {
    std::cerr << "FrameBufferAllocator allocate failed\n";
    impl_->camera->release();
    impl_->cm->stop();
    impl_->running = false;
    return false;
  }

  impl_->camera->requestCompleted.connect(this,
                                          &LibcameraCamera::OnRequestCompleted);

  // Create requests for each allocated buffer.
  impl_->requests.clear();
  for (const auto& fb : impl_->allocator->buffers(impl_->stream)) {
    std::unique_ptr<libcamera::Request> req = impl_->camera->createRequest();
    if (!req) {
      continue;
    }
    if (req->addBuffer(impl_->stream, fb.get()) < 0) {
      continue;
    }
    impl_->requests.push_back(std::move(req));
  }
  if (impl_->requests.empty()) {
    std::cerr << "No libcamera requests created\n";
    Stop();
    return false;
  }

  if (impl_->camera->start()) {
    std::cerr << "Camera start failed\n";
    Stop();
    return false;
  }

  std::cerr << "libcamera focus profile: AfModeContinuous, AfRangeMacro, "
               "AfSpeedFast\n";
  for (auto& req : impl_->requests) {
    ApplyRequestedFocusProfile(req.get());
    impl_->camera->queueRequest(req.get());
  }

  return true;
}

void LibcameraCamera::Stop() {
  if (!impl_) {
    return;
  }
  if (!impl_->running.exchange(false)) {
    return;
  }

  if (impl_->camera) {
    impl_->camera->stop();
  }
  impl_->requests.clear();
  if (impl_->allocator && impl_->stream) {
    (void)impl_->allocator->free(impl_->stream);
  }
  impl_->allocator.reset();
  impl_->config.reset();
  if (impl_->camera) {
    impl_->camera->release();
    impl_->camera.reset();
  }
  if (impl_->cm) {
    impl_->cm->stop();
    impl_->cm.reset();
  }
}

}  // namespace phc
