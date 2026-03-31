#include "libcamera_camera.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>

#if defined(PHC_HAVE_LIBCAMERA)
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/control_ids.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>

#include <sys/mman.h>
#include <unistd.h>
#endif

namespace phc {

struct LibcameraCamera::Impl {
  FrameCallback cb;
  std::atomic<bool> running{false};

#if defined(PHC_HAVE_LIBCAMERA)
  std::unique_ptr<libcamera::CameraManager> cm;
  std::shared_ptr<libcamera::Camera> camera;
  std::unique_ptr<libcamera::CameraConfiguration> config;
  std::unique_ptr<libcamera::FrameBufferAllocator> allocator;
  libcamera::Stream* stream = nullptr;
  std::vector<std::unique_ptr<libcamera::Request>> requests;
  std::mutex mu;
#else
  std::thread stub_thread;
#endif

  uint64_t NowNs() const {
    using Clock = std::chrono::steady_clock;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
    return static_cast<uint64_t>(ns);
  }
};

LibcameraCamera::LibcameraCamera(LibcameraConfig cfg) : cfg_(cfg), impl_(new Impl()) {}

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

#if !defined(PHC_HAVE_LIBCAMERA)
  std::cerr << "LibcameraCamera: libcamera not available in this build (ENABLE_LIBCAMERA requires libcamera dev headers).\n";
  impl_->running = false;
  return false;
#else
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
  impl_->config = impl_->camera->generateConfiguration({libcamera::StreamRole::Viewfinder});
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

  impl_->stream = sc.stream();
  impl_->allocator = std::make_unique<libcamera::FrameBufferAllocator>(impl_->camera);
  if (impl_->allocator->allocate(impl_->stream) < 0) {
    std::cerr << "FrameBufferAllocator allocate failed\n";
    impl_->camera->release();
    impl_->cm->stop();
    impl_->running = false;
    return false;
  }

  impl_->camera->requestCompleted.connect([this](libcamera::Request* request) {
    if (!impl_ || !impl_->running) {
      return;
    }
    if (request->status() == libcamera::Request::RequestCancelled) {
      return;
    }

    const auto& bufs = request->buffers();
    auto it = bufs.find(impl_->stream);
    if (it == bufs.end()) {
      request->reuse(libcamera::Request::ReuseBuffers);
      impl_->camera->queueRequest(request);
      return;
    }
    const libcamera::FrameBuffer* fb = it->second;
    if (fb->planes().empty()) {
      request->reuse(libcamera::Request::ReuseBuffers);
      impl_->camera->queueRequest(request);
      return;
    }

    // Plane 0 for RGB888.
    const int fd = fb->planes()[0].fd.get();
    const size_t length = fb->planes()[0].length;
    void* map = ::mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
      request->reuse(libcamera::Request::ReuseBuffers);
      impl_->camera->queueRequest(request);
      return;
    }

    Frame f;
    f.format = PixelFormat::Rgb888;
    f.width = static_cast<int>(impl_->config->at(0).size.width);
    f.height = static_cast<int>(impl_->config->at(0).size.height);
    // libcamera's stride is accessible via StreamConfiguration::stride (bytes per line)
    f.stride_bytes = static_cast<int>(impl_->config->at(0).stride);
    if (f.stride_bytes <= 0) {
      f.stride_bytes = f.width * 3;
    }
    f.timestamp_ns = impl_->NowNs();

    const size_t needed = static_cast<size_t>(f.height) * static_cast<size_t>(f.stride_bytes);
    f.data.resize(needed);
    std::memcpy(f.data.data(), map, std::min(needed, length));
    ::munmap(map, length);

    if (impl_->cb) {
      impl_->cb(f);
    }

    request->reuse(libcamera::Request::ReuseBuffers);
    impl_->camera->queueRequest(request);
  });

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

  for (auto& req : impl_->requests) {
    impl_->camera->queueRequest(req.get());
  }

  return true;
#endif
}

void LibcameraCamera::Stop() {
  if (!impl_) {
    return;
  }
  if (!impl_->running.exchange(false)) {
    return;
  }

#if defined(PHC_HAVE_LIBCAMERA)
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
#else
  if (impl_->stub_thread.joinable()) {
    impl_->stub_thread.join();
  }
#endif
}

}  // namespace phc

