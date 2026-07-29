#!/usr/bin/env bash
# Cross-compile the DCVC CPU ONNX codec for Windows x64 on a Linux host,
# using the MinGW-w64 toolchain, then assemble a runnable package.
#
# Output: onnx/out/build/windows-x64-mingw/*.exe
#         onnx/out/runnable/windows-x64-mingw/*.exe
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
#     https://gh-proxy.com/https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-win-x64-1.27.0.zip
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

OUTPUT_ROOT="${DCVC_OUTPUT_ROOT:-${ONNX_DIR}/out}"
ARTIFACT_TAG="windows-x64-mingw"
BUILD_DIR="${OUTPUT_ROOT}/build/${ARTIFACT_TAG}"
RUNNABLE_DIR="${OUTPUT_ROOT}/runnable/${ARTIFACT_TAG}"

CMAKE_ARGS=(
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}"
    -DCMAKE_BUILD_TYPE=Release
    -DDCVC_FXP_CUDA=OFF
    -DDCVC_OUTPUT_ROOT="${OUTPUT_ROOT}"
    -DDCVC_ARTIFACT_TAG="${ARTIFACT_TAG}"
)
[ -n "${DCVC_ORT_ARCHIVE:-}" ]  && CMAKE_ARGS+=(-DDCVC_ORT_ARCHIVE="$(readlink -f "${DCVC_ORT_ARCHIVE}")")
[ -n "${DCVC_ORT_URL_BASE:-}" ] && CMAKE_ARGS+=(-DDCVC_ORT_URL_BASE="${DCVC_ORT_URL_BASE}")

echo "== Configuring (MinGW cross, Windows x64) =="
cmake -S "${ONNX_DIR}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"

echo "== Cross-building =="
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "== Packaging into ${RUNNABLE_DIR} =="
cmake --build "${BUILD_DIR}" --target dcvc_package

echo
echo "Done. Windows executables:"
echo "  ${BUILD_DIR}/*.exe"
echo "  ${RUNNABLE_DIR}/*.exe  (runnable folder)"
echo
echo "Copy ${RUNNABLE_DIR}/ to a Windows x64 machine."
