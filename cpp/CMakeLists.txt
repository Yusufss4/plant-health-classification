cmake_minimum_required(VERSION 3.16)
project(mobilenet_plant_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_LIBCAMERA "Build libcamera-based capture module" OFF)
option(ENABLE_TESTS "Build unit tests" ON)

find_package(Threads REQUIRED)

if(NOT DEFINED ENV{ONNXRUNTIME_ROOT})
  message(FATAL_ERROR
    "ONNXRUNTIME_ROOT is not set. Extract the ONNX Runtime release tarball and point to it, e.g.:\n"
    "  export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.24.4\n"
    "Or run: bash ../scripts/download_onnxruntime.sh"
  )
endif()

set(ORT_ROOT "$ENV{ONNXRUNTIME_ROOT}")
if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNXRUNTIME_ROOT=${ORT_ROOT} does not contain include/onnxruntime_cxx_api.h")
endif()

# Vendored stb headers (MIT) for offline/reproducible builds.
set(STB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb")
set(STB_IMAGE_H "${STB_INCLUDE_DIR}/stb_image.h")
set(STB_IMAGE_WRITE_H "${STB_INCLUDE_DIR}/stb_image_write.h")

add_library(phc_preprocess STATIC
  src/preprocess/mobilenet_preprocess.cpp
)
target_include_directories(phc_preprocess PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${STB_INCLUDE_DIR}"
)

add_library(phc_inference_ort STATIC
  src/inference_ort/ort_engine.cpp
)
target_include_directories(phc_inference_ort PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SYSTEM "${ORT_ROOT}/include"
)
target_link_directories(phc_inference_ort PUBLIC "${ORT_ROOT}/lib")
target_link_libraries(phc_inference_ort PUBLIC onnxruntime)

add_library(phc_app_runtime STATIC
  src/app_runtime/live_pipeline.cpp
)
target_include_directories(phc_app_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(phc_app_runtime PUBLIC phc_preprocess phc_inference_ort)

# Generate embedded_index_html.cpp from web/live/index.html. Re-runs whenever
# the HTML source changes; deleted on `cmake --build --target clean` via the
# generated-file dependency tracking.
set(PHC_EMBEDDED_HTML_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../web/live/index.html")
set(PHC_EMBEDDED_HTML_OUT
  "${CMAKE_CURRENT_BINARY_DIR}/generated/embedded_index_html.cpp")
add_custom_command(
  OUTPUT "${PHC_EMBEDDED_HTML_OUT}"
  COMMAND "${CMAKE_COMMAND}"
          -DINPUT=${PHC_EMBEDDED_HTML_SRC}
          -DOUTPUT=${PHC_EMBEDDED_HTML_OUT}
          -DSYMBOL=kEmbeddedIndexHtml
          -DHEADER=embedded_index_html.hpp
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_html.cmake"
  DEPENDS "${PHC_EMBEDDED_HTML_SRC}"
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_html.cmake"
  VERBATIM
  COMMENT "Embedding ${PHC_EMBEDDED_HTML_SRC} into ${PHC_EMBEDDED_HTML_OUT}"
)

add_library(phc_server_http STATIC
  src/server_http/http_stream_server.cpp
  src/server_http/http_stream_display.cpp
  "${PHC_EMBEDDED_HTML_OUT}"
)
target_include_directories(phc_server_http PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${STB_INCLUDE_DIR}"
)
target_include_directories(phc_server_http PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cpp-httplib"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/server_http"
)
target_link_libraries(phc_server_http PUBLIC Threads::Threads)

if(ENABLE_LIBCAMERA)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBCAMERA REQUIRED IMPORTED_TARGET libcamera)
  add_library(phc_camera_libcamera STATIC
    src/camera_libcamera/libcamera_camera.cpp
  )
  target_include_directories(phc_camera_libcamera PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(phc_camera_libcamera PUBLIC PkgConfig::LIBCAMERA)
endif()

function(link_ort_rpath target_name)
  if(UNIX AND NOT APPLE)
    set_target_properties(${target_name} PROPERTIES
      BUILD_RPATH "${ORT_ROOT}/lib"
      INSTALL_RPATH "\$ORIGIN/../lib"
    )
  endif()
endfunction()

add_executable(phc_infer_mobilenet tools/infer_mobilenet.cpp)
target_include_directories(phc_infer_mobilenet PRIVATE SYSTEM "${ORT_ROOT}/include")
target_link_directories(phc_infer_mobilenet PRIVATE "${ORT_ROOT}/lib")
target_link_libraries(phc_infer_mobilenet PRIVATE phc_preprocess phc_inference_ort onnxruntime)
link_ort_rpath(phc_infer_mobilenet)

add_executable(phc_evaluate_mobilenet tools/evaluate_mobilenet.cpp)
target_include_directories(phc_evaluate_mobilenet PRIVATE SYSTEM "${ORT_ROOT}/include")
target_link_directories(phc_evaluate_mobilenet PRIVATE "${ORT_ROOT}/lib")
target_link_libraries(phc_evaluate_mobilenet PRIVATE phc_preprocess phc_inference_ort onnxruntime)
link_ort_rpath(phc_evaluate_mobilenet)

if(ENABLE_LIBCAMERA)
  add_executable(live_infer_web
    apps/live_infer_web/main.cpp
  )
  target_include_directories(live_infer_web PRIVATE
    SYSTEM "${ORT_ROOT}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(live_infer_web PRIVATE
    phc_app_runtime
    phc_camera_libcamera
    phc_server_http
    phc_inference_ort
    phc_preprocess
    onnxruntime
  )
  link_ort_rpath(live_infer_web)
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

  # Third-party: silence warnings in Catch2 sources (we cannot fix upstream).
  foreach(_phc_catch_tgt Catch2 Catch2WithMain)
    if(TARGET ${_phc_catch_tgt})
      if(MSVC)
        target_compile_options(${_phc_catch_tgt} PRIVATE /W0)
      else()
        target_compile_options(${_phc_catch_tgt} PRIVATE -w)
      endif()
    endif()
  endforeach()

  add_executable(phc_tests tests/test_preprocess.cpp)
  target_include_directories(phc_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(phc_tests PRIVATE Catch2::Catch2WithMain phc_preprocess)
  add_test(NAME phc_tests COMMAND phc_tests)
endif()
