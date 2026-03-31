cmake_minimum_required(VERSION 3.16)
project(mobilenet_plant_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_LIBCAMERA "Build libcamera-based capture module" OFF)
option(ENABLE_DRMKMS "Build DRM/KMS display module" OFF)
option(ENABLE_TESTS "Build unit tests" ON)

if(NOT DEFINED ENV{ONNXRUNTIME_ROOT})
  message(FATAL_ERROR
    "ONNXRUNTIME_ROOT is not set. Extract the ONNX Runtime release tarball and point to it, e.g.:\n"
    "  export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.17.3\n"
    "Or run: bash ../scripts/download_onnxruntime.sh"
  )
endif()

set(ORT_ROOT "$ENV{ONNXRUNTIME_ROOT}")
if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNXRUNTIME_ROOT=${ORT_ROOT} does not contain include/onnxruntime_cxx_api.h")
endif()

# stb_image (MIT) — single-header loader
set(STB_IMAGE_H "${CMAKE_CURRENT_BINARY_DIR}/stb_image.h")
if(NOT EXISTS "${STB_IMAGE_H}")
  file(DOWNLOAD
    "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
    "${STB_IMAGE_H}"
    SHOW_PROGRESS
  )
endif()

add_library(phc_preprocess STATIC
  src/preprocess/mobilenet_preprocess.cpp
)
target_include_directories(phc_preprocess PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_BINARY_DIR}"
)

add_library(phc_inference_ort STATIC
  src/inference_ort/ort_engine.cpp
)
target_include_directories(phc_inference_ort PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${ORT_ROOT}/include"
)
target_link_directories(phc_inference_ort PUBLIC "${ORT_ROOT}/lib")
target_link_libraries(phc_inference_ort PUBLIC onnxruntime)

add_library(phc_app_runtime STATIC
  src/app_runtime/live_pipeline.cpp
)
target_include_directories(phc_app_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(phc_app_runtime PUBLIC phc_preprocess phc_inference_ort)

if(ENABLE_LIBCAMERA)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBCAMERA REQUIRED IMPORTED_TARGET libcamera)
  add_library(phc_camera_libcamera STATIC
    src/camera_libcamera/libcamera_camera.cpp
  )
  target_include_directories(phc_camera_libcamera PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(phc_camera_libcamera PUBLIC PkgConfig::LIBCAMERA)
  target_compile_definitions(phc_camera_libcamera PUBLIC PHC_HAVE_LIBCAMERA=1)
endif()

if(ENABLE_DRMKMS)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBDRM REQUIRED IMPORTED_TARGET libdrm)
  add_library(phc_display_drm STATIC
    src/display_drm/drm_display.cpp
  )
  target_include_directories(phc_display_drm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(phc_display_drm PUBLIC PkgConfig::LIBDRM)
  target_compile_definitions(phc_display_drm PUBLIC PHC_HAVE_DRM=1)
endif()

function(link_ort_rpath target_name)
  if(UNIX AND NOT APPLE)
    set_target_properties(${target_name} PROPERTIES
      BUILD_RPATH "${ORT_ROOT}/lib"
      INSTALL_RPATH "\$ORIGIN/../lib"
    )
  endif()
endfunction()

add_executable(infer_mobilenet infer_mobilenet.cpp)
target_include_directories(infer_mobilenet PRIVATE "${ORT_ROOT}/include")
target_link_directories(infer_mobilenet PRIVATE "${ORT_ROOT}/lib")
target_link_libraries(infer_mobilenet PRIVATE phc_preprocess phc_inference_ort onnxruntime)
link_ort_rpath(infer_mobilenet)

add_executable(evaluate_mobilenet evaluate_mobilenet.cpp)
target_include_directories(evaluate_mobilenet PRIVATE "${ORT_ROOT}/include")
target_link_directories(evaluate_mobilenet PRIVATE "${ORT_ROOT}/lib")
target_link_libraries(evaluate_mobilenet PRIVATE phc_preprocess phc_inference_ort onnxruntime)
link_ort_rpath(evaluate_mobilenet)

if(ENABLE_LIBCAMERA AND ENABLE_DRMKMS)
  add_executable(live_infer_display
    apps/live_infer_display/main.cpp
  )
  target_include_directories(live_infer_display PRIVATE "${ORT_ROOT}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(live_infer_display PRIVATE
    phc_app_runtime
    phc_camera_libcamera
    phc_display_drm
    phc_inference_ort
    phc_preprocess
    onnxruntime
  )
  link_ort_rpath(live_infer_display)
endif()

if(ENABLE_TESTS)
  include(CTest)
  enable_testing()

  include(FetchContent)
  FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.6.0
  )
  FetchContent_MakeAvailable(catch2)

  add_executable(phc_tests
    tests/test_preprocess.cpp
    tests/test_live_pipeline.cpp
    tests/pi_smoke_live.cpp
  )
  target_include_directories(phc_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(phc_tests PRIVATE Catch2::Catch2WithMain phc_preprocess phc_app_runtime)
  if(ENABLE_LIBCAMERA)
    target_link_libraries(phc_tests PRIVATE phc_camera_libcamera)
  endif()
  if(ENABLE_DRMKMS)
    target_link_libraries(phc_tests PRIVATE phc_display_drm)
  endif()
  add_test(NAME phc_tests COMMAND phc_tests)
endif()
