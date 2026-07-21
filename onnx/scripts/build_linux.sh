#!/usr/bin/env bash
# Build the pure-CPU DCVC-RT I-frame codec on Linux.
# Produces test executables under build/ and a ready-to-run package under dist/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${ONNX_DIR}/build"
DIST_DIR="${ONNX_DIR}/dist"

echo "== Configuring (Linux, ${BUILD_TYPE}) =="
cmake -S "${ONNX_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

echo "== Building =="
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "== Packaging into ${DIST_DIR} =="
cmake --build "${BUILD_DIR}" --target dcvc_package

echo
echo "Done. Package: ${DIST_DIR}/dcvc_onnx_codec/"
echo "Quick test:    ${BUILD_DIR}/test_intra_analysis ../intra_analysis_standard.onnx"
