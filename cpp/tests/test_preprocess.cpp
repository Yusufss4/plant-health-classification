#include <catch2/catch_test_macros.hpp>

#include "../src/core/rgb_normalize.hpp"
#include "../src/preprocess/mobilenet_preprocess.hpp"

TEST_CASE("MobilenetPreprocessor rejects non-RGB frames", "[preprocess]") {
  phc::MobilenetPreprocessor pp;
  phc::Frame f;
  f.format = phc::PixelFormat::Nv12;
  f.width = 4;
  f.height = 4;
  f.stride_bytes = 4;
  f.data.resize(16);

  phc::TensorF32 t;
  REQUIRE_FALSE(pp.Run(f, t));
}

TEST_CASE("MobilenetPreprocessor produces 1x3x224x224 tensor", "[preprocess]") {
  phc::MobilenetPreprocessor pp;
  phc::Frame f;
  f.format = phc::PixelFormat::Rgb888;
  f.width = 2;
  f.height = 2;
  f.stride_bytes = 2 * 3;
  f.data = {
      255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255,
  };

  phc::TensorF32 t;
  REQUIRE(pp.Run(f, t));
  REQUIRE(t.n == 1);
  REQUIRE(t.c == 3);
  REQUIRE(t.h == 224);
  REQUIRE(t.w == 224);
  REQUIRE(t.data.size() == static_cast<size_t>(1 * 3 * 224 * 224));
}

TEST_CASE("NormalizePackedToRgb888 swaps BGR channels correctly", "[camera][color]") {
  // Source pixel order per pixel: B, G, R
  const uint8_t bgr_data[] = {
      200, 20, 10,  // expected RGB: 10,20,200
      3, 2, 1,      // expected RGB: 1,2,3
  };
  std::vector<uint8_t> out;
  int out_stride = 0;
  REQUIRE(phc::NormalizePackedToRgb888(bgr_data, 2, 1, 2 * 3, 3,
                                       phc::PackedRgbOrder::Bgr, out,
                                       out_stride));
  REQUIRE(out_stride == 2 * 3);
  REQUIRE(out.size() == static_cast<size_t>(out_stride));
  REQUIRE(out[0] == 10);
  REQUIRE(out[1] == 20);
  REQUIRE(out[2] == 200);
  REQUIRE(out[3] == 1);
  REQUIRE(out[4] == 2);
  REQUIRE(out[5] == 3);
}

TEST_CASE("NormalizePackedToRgb888 ignores alpha bytes in 4-byte input",
          "[camera][color]") {
  // Source pixel order per pixel: B, G, R, A (A should be ignored).
  const uint8_t bgra_data[] = {
      30, 40, 50, 99,  // expected RGB: 50,40,30
      1, 2, 3, 4,      // expected RGB: 3,2,1
  };
  std::vector<uint8_t> out;
  int out_stride = 0;
  REQUIRE(phc::NormalizePackedToRgb888(bgra_data, 2, 1, 2 * 4, 4,
                                       phc::PackedRgbOrder::Bgr, out,
                                       out_stride));
  REQUIRE(out_stride == 2 * 3);
  REQUIRE(out.size() == static_cast<size_t>(out_stride));
  REQUIRE(out[0] == 50);
  REQUIRE(out[1] == 40);
  REQUIRE(out[2] == 30);
  REQUIRE(out[3] == 3);
  REQUIRE(out[4] == 2);
  REQUIRE(out[5] == 1);
}

TEST_CASE("NormalizePackedToRgb888 can force R/B swap for RGB888 input",
          "[camera][color]") {
  // Source pixel order per pixel: R, G, B.
  const uint8_t rgb_data[] = {
      10, 20, 200,  // expected (forced swap): 200,20,10
      1, 2, 3,      // expected (forced swap): 3,2,1
  };
  std::vector<uint8_t> out;
  int out_stride = 0;
  REQUIRE(phc::NormalizePackedToRgb888(rgb_data, 2, 1, 2 * 3, 3,
                                       phc::PackedRgbOrder::Bgr, out,
                                       out_stride));
  REQUIRE(out_stride == 2 * 3);
  REQUIRE(out.size() == static_cast<size_t>(out_stride));
  REQUIRE(out[0] == 200);
  REQUIRE(out[1] == 20);
  REQUIRE(out[2] == 10);
  REQUIRE(out[3] == 3);
  REQUIRE(out[4] == 2);
  REQUIRE(out[5] == 1);
}
