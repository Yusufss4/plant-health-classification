# Native GCC toolchain + shared compiler flags.
#
# This file is intended to be used via CMake Presets:
#   "toolchainFile": "${sourceDir}/toolchains/gcc/toolchain.cmake"
#
# It sets common warnings and per-config Debug/Release flags in one place.

set(CMAKE_C_COMPILER gcc CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER g++ CACHE FILEPATH "C++ compiler" FORCE)

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

# Reasonable default for single-config generators (Ninja/Unix Makefiles).
# Multi-config generators (Visual Studio/Xcode) ignore this.
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

