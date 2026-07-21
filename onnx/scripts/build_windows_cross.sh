#!/usr/bin/env bash
# Cross-compile the DCVC CPU ONNX codec for Windows x64 on a Linux host,
# using the MinGW-w64 toolchain, then assemble a runnable package.
#
# Output: onnx/build-mingw/*.exe  +  onnx/dist/dcvc_onnx_codec/*.exe
#
# Prerequisites (Debian/Ubuntu):
#   sudo apt-get install mingw-w64
# (the toolchain file uses the posix-threads variant of gcc/g++.)
#
# The MinGW-built executables statically link the C/C++/pthread runtimes
# (-static), so the package only needs onnxruntime.dll + the models.
#
# Environment variables (ONNX Runtime download):
#   DCVC_ORT_ARCHIVE   - use a local .zip you already downloaded (fastest)
#                        e.g. fetched via a mirror:
#   curl -L -o ort.zip \
#     https://gh-proxy.com/https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-win-x64-1.19.0.zip
#   DCVC_ORT_ARCHIVE=ort.zip bash onnx/scripts/build_windows_cross.sh
#
#   DCVC_ORT_URL_BASE  - mirror host prefix (github.com path is appended)
#                        e.g. export DCVC_ORT_URL_BASE=https://gh-proxy.com/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLCHAIN="${SCRIPT_DIR}/mingw-w64-x86_64.toolchain.cmake"

if ! command -v x86_64-w64-mingw32-gcc-posix >/dev/null 2>&1; then
    echo "error: MinGW-w64 not found. Install it, e.g.:" >&2
    echo "  sudo apt-get install mingw-w64" >&2
    exit 1
fi

BUILD_DIR="${ONNX_DIR}/build-mingw"

CMAKE_ARGS=(
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}"
    -DCMAKE_BUILD_TYPE=Release
)
[ -n "${DCVC_ORT_ARCHIVE:-}" ]  && CMAKE_ARGS+=(-DDCVC_ORT_ARCHIVE="$(readlink -f "${DCVC_ORT_ARCHIVE}")")
[ -n "${DCVC_ORT_URL_BASE:-}" ] && CMAKE_ARGS+=(-DDCVC_ORT_URL_BASE="${DCVC_ORT_URL_BASE}")

echo "== Configuring (MinGW cross, Windows x64) =="
cmake -S "${ONNX_DIR}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"

echo "== Cross-building =="
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "== Packaging into onnx/dist/dcvc_onnx_codec/ =="
cmake --build "${BUILD_DIR}" --target dcvc_package

echo
echo "Done. Windows executables:"
echo "  ${BUILD_DIR}/*.exe"
echo "  ${ONNX_DIR}/dist/dcvc_onnx_codec/*.exe  (runnable folder)"
echo
echo "Copy onnx/dist/dcvc_onnx_codec/ to a Windows x64 machine and run:"
echo "  test_cpu_end2end.exe . 256 256 32"
