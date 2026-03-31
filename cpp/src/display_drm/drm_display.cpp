#include "drm_display.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#if defined(PHC_HAVE_DRM)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <drm.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#endif

namespace phc {

struct DrmKmsDisplay::Impl {
  int fd = -1;
  uint32_t crtc_id = 0;
  uint32_t connector_id = 0;
  drmModeModeInfo mode{};
  uint32_t mode_width = 0;
  uint32_t mode_height = 0;

  struct Buffer {
    uint32_t fb_id = 0;
    uint32_t handle = 0;
    uint32_t pitch = 0;
    uint64_t size = 0;
    void* map = nullptr;
  };

  Buffer bufs[2]{};
  int cur = 0;
  bool modeset_done = false;
};

static void CloseFd(int& fd) {
  if (fd >= 0) {
    ::close(fd);
    fd = -1;
  }
}

DrmKmsDisplay::DrmKmsDisplay(DrmConfig cfg) : cfg_(cfg), impl_(new Impl()) {}

DrmKmsDisplay::~DrmKmsDisplay() {
  if (!impl_) {
    return;
  }
#if defined(PHC_HAVE_DRM)
  for (auto& b : impl_->bufs) {
    if (b.map && b.size) {
      ::munmap(b.map, static_cast<size_t>(b.size));
      b.map = nullptr;
    }
    if (b.fb_id) {
      drmModeRmFB(impl_->fd, b.fb_id);
      b.fb_id = 0;
    }
    if (b.handle) {
      drm_mode_destroy_dumb d{};
      d.handle = b.handle;
      (void)drmIoctl(impl_->fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
      b.handle = 0;
    }
  }
  CloseFd(impl_->fd);
#else
  (void)cfg_;
#endif
  delete impl_;
  impl_ = nullptr;
}

#if defined(PHC_HAVE_DRM)
static bool CreateDumbBuffer(int fd, uint32_t width, uint32_t height, DrmKmsDisplay::Impl::Buffer& out) {
  drm_mode_create_dumb creq{};
  creq.width = width;
  creq.height = height;
  creq.bpp = 32;  // XRGB8888
  if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &creq) < 0) {
    return false;
  }
  out.handle = creq.handle;
  out.pitch = creq.pitch;
  out.size = creq.size;

  if (drmModeAddFB(fd, width, height, 24, 32, out.pitch, out.handle, &out.fb_id) != 0) {
    return false;
  }

  drm_mode_map_dumb mreq{};
  mreq.handle = out.handle;
  if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &mreq) < 0) {
    return false;
  }

  out.map = ::mmap(nullptr, static_cast<size_t>(out.size), PROT_READ | PROT_WRITE, MAP_SHARED, fd, mreq.offset);
  if (out.map == MAP_FAILED) {
    out.map = nullptr;
    return false;
  }
  std::memset(out.map, 0, static_cast<size_t>(out.size));
  return true;
}

static bool PickConnectorCrtc(int fd, uint32_t& out_connector_id, uint32_t& out_crtc_id, drmModeModeInfo& out_mode) {
  drmModeRes* res = drmModeGetResources(fd);
  if (!res) {
    return false;
  }

  bool ok = false;
  for (int i = 0; i < res->count_connectors && !ok; ++i) {
    drmModeConnector* conn = drmModeGetConnector(fd, res->connectors[i]);
    if (!conn) {
      continue;
    }
    const bool connected = (conn->connection == DRM_MODE_CONNECTED) && (conn->count_modes > 0);
    if (!connected) {
      drmModeFreeConnector(conn);
      continue;
    }

    // Choose preferred mode if present, otherwise mode[0].
    int best = 0;
    for (int m = 0; m < conn->count_modes; ++m) {
      if (conn->modes[m].type & DRM_MODE_TYPE_PREFERRED) {
        best = m;
        break;
      }
    }
    out_mode = conn->modes[best];
    out_connector_id = conn->connector_id;

    // Find an encoder+crtc.
    drmModeEncoder* enc = nullptr;
    if (conn->encoder_id) {
      enc = drmModeGetEncoder(fd, conn->encoder_id);
    }
    if (enc && enc->crtc_id) {
      out_crtc_id = enc->crtc_id;
      ok = true;
    } else {
      // Fallback: pick first CRTC.
      if (res->count_crtcs > 0) {
        out_crtc_id = res->crtcs[0];
        ok = true;
      }
    }
    if (enc) {
      drmModeFreeEncoder(enc);
    }
    drmModeFreeConnector(conn);
  }

  drmModeFreeResources(res);
  return ok;
}

static inline uint32_t PackXrgb(uint8_t r, uint8_t g, uint8_t b) {
  return (0xFFu << 24) | (static_cast<uint32_t>(r) << 16) | (static_cast<uint32_t>(g) << 8) | static_cast<uint32_t>(b);
}
#endif

bool DrmKmsDisplay::Init(int width, int height) {
  if (!impl_) {
    return false;
  }
#if !defined(PHC_HAVE_DRM)
  std::cerr << "DrmKmsDisplay: libdrm not available in this build (ENABLE_DRMKMS requires libdrm dev headers).\n";
  (void)width;
  (void)height;
  return false;
#else
  const char* path = (cfg_.device_path && cfg_.device_path[0] != '\0') ? cfg_.device_path : "/dev/dri/card0";
  impl_->fd = ::open(path, O_RDWR | O_CLOEXEC);
  if (impl_->fd < 0) {
    std::cerr << "Failed to open DRM device: " << path << "\n";
    return false;
  }

  if (!PickConnectorCrtc(impl_->fd, impl_->connector_id, impl_->crtc_id, impl_->mode)) {
    std::cerr << "No connected DRM connector found\n";
    CloseFd(impl_->fd);
    return false;
  }

  impl_->mode_width = static_cast<uint32_t>(impl_->mode.hdisplay);
  impl_->mode_height = static_cast<uint32_t>(impl_->mode.vdisplay);
  (void)width;
  (void)height;

  if (!CreateDumbBuffer(impl_->fd, impl_->mode_width, impl_->mode_height, impl_->bufs[0]) ||
      !CreateDumbBuffer(impl_->fd, impl_->mode_width, impl_->mode_height, impl_->bufs[1])) {
    std::cerr << "Failed to create dumb buffers\n";
    return false;
  }

  // Initial modeset to buffer 0.
  if (drmModeSetCrtc(impl_->fd, impl_->crtc_id, impl_->bufs[0].fb_id, 0, 0, &impl_->connector_id, 1, &impl_->mode) != 0) {
    std::cerr << "drmModeSetCrtc failed\n";
    return false;
  }
  impl_->modeset_done = true;
  impl_->cur = 0;
  return true;
#endif
}

bool DrmKmsDisplay::Present(const Frame& rgb888) {
  if (!impl_) {
    return false;
  }
#if !defined(PHC_HAVE_DRM)
  (void)rgb888;
  return false;
#else
  if (!impl_->modeset_done || impl_->fd < 0) {
    return false;
  }
  if (rgb888.format != PixelFormat::Rgb888 || rgb888.empty()) {
    return false;
  }

  const int next = 1 - impl_->cur;
  auto& b = impl_->bufs[next];
  if (!b.map) {
    return false;
  }

  const uint32_t dw = impl_->mode_width;
  const uint32_t dh = impl_->mode_height;
  const uint32_t sw = static_cast<uint32_t>(rgb888.width);
  const uint32_t sh = static_cast<uint32_t>(rgb888.height);

  // Simple top-left blit with cropping. (Later: scale or letterbox.)
  const uint32_t copy_w = std::min(dw, sw);
  const uint32_t copy_h = std::min(dh, sh);

  uint8_t* dst = static_cast<uint8_t*>(b.map);
  const uint8_t* src = rgb888.data.data();

  for (uint32_t y = 0; y < copy_h; ++y) {
    const uint8_t* srow = src + static_cast<size_t>(y) * static_cast<size_t>(rgb888.stride_bytes);
    uint32_t* drow = reinterpret_cast<uint32_t*>(dst + static_cast<size_t>(y) * static_cast<size_t>(b.pitch));
    for (uint32_t x = 0; x < copy_w; ++x) {
      const uint8_t* p = srow + static_cast<size_t>(x) * 3;
      drow[x] = PackXrgb(p[0], p[1], p[2]);
    }
  }

  // Page flip by setting CRTC to the next framebuffer.
  if (drmModeSetCrtc(impl_->fd, impl_->crtc_id, b.fb_id, 0, 0, &impl_->connector_id, 1, &impl_->mode) != 0) {
    return false;
  }
  impl_->cur = next;
  return true;
#endif
}

}  // namespace phc

