# CMake toolchain file for cross-compiling Windows x64 binaries on Linux
# using the MinGW-w64 toolchain.
#
# Usage (from the repo root):
#   cmake -S onnx -B onnx/build-mingw \
#         -DCMAKE_TOOLCHAIN_FILE=onnx/scripts/mingw-w64-x86_64.toolchain.cmake
#   cmake --build onnx/build-mingw -j$(nproc)
#   cmake --build onnx/build-mingw --target dcvc_package
#
# Or use the convenience wrapper:  onnx/scripts/build_windows_cross.sh
#
# Requires:  apt-get install mingw-w64     (or your distro's equivalent)

set(CMAKE_SYSTEM_NAME      Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Use the posix threads variant so C++ std::thread / std::shared_ptr work
# the same way as on Linux (rANS uses std::thread).
set(_triplet x86_64-w64-mingw32)
set(CMAKE_C_COMPILER   ${_triplet}-gcc-posix)
set(CMAKE_CXX_COMPILER ${_triplet}-g++-posix)
set(CMAKE_RC_COMPILER  ${_triplet}-windres)

# Tell CMake/FindXXX to look for target (Windows) libraries under the
# mingw sysroot, but never try to run host-built tools as target programs.
set(CMAKE_FIND_ROOT_PATH      /usr/${_triplet})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM  NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY  ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE  ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE  ONLY)
