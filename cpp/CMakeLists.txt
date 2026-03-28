cmake_minimum_required(VERSION 3.16)
project(mobilenet_plant_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_executable(infer_mobilenet infer_mobilenet.cpp)
target_include_directories(infer_mobilenet PRIVATE
  "${ORT_ROOT}/include"
  "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_directories(infer_mobilenet PRIVATE "${ORT_ROOT}/lib")
target_link_libraries(infer_mobilenet PRIVATE onnxruntime)

if(UNIX AND NOT APPLE)
  set_target_properties(infer_mobilenet PROPERTIES
    BUILD_RPATH "${ORT_ROOT}/lib"
    INSTALL_RPATH "\$ORIGIN/../lib"
  )
endif()
