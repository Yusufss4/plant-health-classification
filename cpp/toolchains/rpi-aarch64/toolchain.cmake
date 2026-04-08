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

# Sysroot is required for this toolchain (target headers/libs, pkg-config, etc.).
# Prefer:
#   export RPI_SYSROOT=/abs/path/to/sysroot
# Or:
#   cmake ... -D RPI_SYSROOT=/abs/path/to/sysroot
set(RPI_SYSROOT "" CACHE PATH "Sysroot for target (required)")
if(NOT RPI_SYSROOT AND DEFINED ENV{RPI_SYSROOT})
  set(RPI_SYSROOT "$ENV{RPI_SYSROOT}")
endif()

if(NOT RPI_SYSROOT)
  message(FATAL_ERROR
    "RPI_SYSROOT is required for the rpi-aarch64 toolchain.\n"
    "Set it via environment or cache:\n"
    "  export RPI_SYSROOT=/abs/path/to/sysroot\n"
    "  # or\n"
    "  cmake --preset rpi-zero2w-release -D RPI_SYSROOT=/abs/path/to/sysroot\n"
  )
endif()

# Normalize (helps avoid subtle mismatches in find logic).
get_filename_component(RPI_SYSROOT "${RPI_SYSROOT}" ABSOLUTE)

set(CMAKE_SYSROOT "${RPI_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")

# Ensure host pkg-config resolves *target* .pc files under the sysroot.
# (Most projects call find_package(PkgConfig) then pkg_check_modules().)
set(ENV{PKG_CONFIG_SYSROOT_DIR} "${CMAKE_SYSROOT}")
set(ENV{PKG_CONFIG_DIR} "")
set(ENV{PKG_CONFIG_PATH} "")
# CMake's set(ENV{...}) takes a single value; build a colon-separated list.
string(JOIN ":" _RPI_PKG_CONFIG_LIBDIR
  "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig"
  "${CMAKE_SYSROOT}/usr/lib/pkgconfig"
  "${CMAKE_SYSROOT}/usr/share/pkgconfig"
)
set(ENV{PKG_CONFIG_LIBDIR} "${_RPI_PKG_CONFIG_LIBDIR}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# When cross compiling with a sysroot, do not accidentally pick up host headers/libs.
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Common flags (applied to all configs)
set(_COMMON_WARNINGS
  -Wall
  -Wextra
  -Wpedantic
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

