# Cross-compile toolchain for Raspberry Pi Zero 2 W (aarch64) from Linux x86_64.
# Expects a standard GNU triplet toolchain (e.g. Debian/Ubuntu: gcc-aarch64-linux-gnu).
#
# Override prefix if your install uses a different prefix:
#   cmake --preset rpi-zero2w-release -D RPI_CROSS_PREFIX=my-aarch64-linux-gnu

set(RPI_CROSS_PREFIX "aarch64-linux-gnu" CACHE STRING "Cross compiler prefix (e.g. aarch64-linux-gnu)")

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER "${RPI_CROSS_PREFIX}-gcc")
set(CMAKE_CXX_COMPILER "${RPI_CROSS_PREFIX}-g++")

# Optional: point at a Pi/sysroot tree for future deps (libcamera, etc.)
set(RPI_SYSROOT "" CACHE PATH "Sysroot for target (leave empty to use the cross compiler default)")
if(RPI_SYSROOT)
  set(CMAKE_SYSROOT "${RPI_SYSROOT}")
  set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

# Common flags (applied to all configs)
set(_COMMON_WARNINGS
  -Wall
  -Wextra
  -Wpedantic
  -Wshadow
  -Wconversion
  -Wsign-conversion
  -Wformat=2
  -Wnull-dereference
)

string(JOIN " " _C_FLAGS_COMMON ${_COMMON_WARNINGS})
set(CMAKE_C_FLAGS_INIT "${_C_FLAGS_COMMON}")

string(JOIN " " _CXX_FLAGS_COMMON ${_COMMON_WARNINGS})
set(CMAKE_CXX_FLAGS_INIT "${_CXX_FLAGS_COMMON}")

# Debug flags
set(CMAKE_C_FLAGS_DEBUG_INIT "-O0 -g3")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-O0 -g3")

# Release flags
set(CMAKE_C_FLAGS_RELEASE_INIT "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 -DNDEBUG")

