#!/usr/bin/env bash
# Build the pure-CPU DCVC-RT I-frame codec on Linux.
# Produces build intermediates and a runnable folder under out/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_TYPE="${BUILD_TYPE:-Release}"
OUTPUT_ROOT="${DCVC_OUTPUT_ROOT:-${ONNX_DIR}/out}"
ARTIFACT_TAG="linux-x64"
BUILD_DIR="${OUTPUT_ROOT}/build/${ARTIFACT_TAG}"
RUNNABLE_DIR="${OUTPUT_ROOT}/runnable/${ARTIFACT_TAG}"

echo "== Configuring (Linux, ${BUILD_TYPE}) =="
cmake -S "${ONNX_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DDCVC_FXP_CUDA=OFF \
    -DDCVC_OUTPUT_ROOT="${OUTPUT_ROOT}" \
    -DDCVC_ARTIFACT_TAG="${ARTIFACT_TAG}"

echo "== Building =="
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "== Packaging into ${RUNNABLE_DIR} =="
cmake --build "${BUILD_DIR}" --target dcvc_package

echo
echo "Done. Runnable folder: ${RUNNABLE_DIR}/"
echo "Build directory:       ${BUILD_DIR}/"
