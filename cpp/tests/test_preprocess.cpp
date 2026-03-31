#include <catch2/catch_test_macros.hpp>

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
      255, 0, 0,   0, 255, 0,
      0, 0, 255,   255, 255, 255,
  };

  phc::TensorF32 t;
  REQUIRE(pp.Run(f, t));
  REQUIRE(t.n == 1);
  REQUIRE(t.c == 3);
  REQUIRE(t.h == 224);
  REQUIRE(t.w == 224);
  REQUIRE(t.data.size() == static_cast<size_t>(1 * 3 * 224 * 224));
}

