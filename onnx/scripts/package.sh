#!/usr/bin/env bash
# ===========================================================================
# DCVC-SDK portable package builder.
#
# Produces a self-contained, relocatable SDK folder (headers + shared library
# + ONNX Runtime + CMake config + example app + docs + discovered model packs),
# then archives it as .tar.gz (Linux) or .zip (Windows).
#
#   Linux native build:
#     bash scripts/package.sh
#     bash scripts/package.sh --model-pack 720p=/path/to/720p \
#         --model-pack 1080p=/path/to/1080p --out /path/to/packages
#
#   Windows x64 via MinGW cross-compile (from a Linux host):
#     bash scripts/package.sh --target mingw
#
# Output:
#   out/packages/dcvc-sdk-${VERSION}-${PLATFORM}.tar.gz  (linux)
#   out/packages/dcvc-sdk-${VERSION}-${PLATFORM}.zip     (mingw)
#
# Environment overrides:
#   DCVC_ORT_ARCHIVE   local ONNX Runtime archive (skips the download)
#   DCVC_ORT_URL_BASE  mirror prefix for the ORT download
#   DCVC_ORT_GPU=1     fetch the CUDA-enabled ORT package
#   DCVC_FXP_CUDA=1    build the CUDA FXP backends
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- defaults -------------------------------------------------------------
TARGET=""            # "" => autodetect host; "mingw" => Windows cross-compile
MODEL_PACK_SPECS=()
OUTPUT_ROOT="${DCVC_OUTPUT_ROOT:-${ONNX_DIR}/out}"
OUT_DIR=""
BUILD_DIR=""
NO_MODELS=0
KEEP_STAGING=0

usage() {
    sed -n '2,26p' "${BASH_SOURCE[0]}"
    echo
    echo "Options:"
    echo "  --target mingw     cross-compile for Windows x64 (MinGW)"
    echo "  --model-pack <name=dir> add a model pack (repeatable; overrides discovery)"
    echo "  --no-models             do not bundle model packs"
    echo "  --keep-staging          keep the expanded SDK tree after archiving"
    echo "  --out <dir>        archive directory (default: out/packages/)"
    echo "  --build <dir>      CMake build directory (default: out/build/<platform>/)"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)         TARGET="$2"; shift 2;;
        --model-pack)     MODEL_PACK_SPECS+=("$2"); shift 2;;
        --no-models)      NO_MODELS=1; shift;;
        --keep-staging)   KEEP_STAGING=1; shift;;
        --out)            OUT_DIR="$2"; shift 2;;
        --build)          BUILD_DIR="$2"; shift 2;;
        -h|--help)        usage 0;;
        *) echo "unknown option: $1" >&2; usage 1;;
    esac
done

# With no explicit list, discover resolution packs such as models_720p,
# models_1080p and models_2160p. Experimental suffixes (for example *_fp32)
# are intentionally excluded.
if [[ "${NO_MODELS}" -eq 0 && ${#MODEL_PACK_SPECS[@]} -eq 0 ]]; then
    for model_dir in "${ONNX_DIR}"/models_*; do
        [[ -d "${model_dir}" ]] || continue
        model_name="$(basename "${model_dir}")"
        model_name="${model_name#models_}"
        [[ "${model_name}" =~ ^[0-9]+p$ ]] || continue
        MODEL_PACK_SPECS+=("${model_name}=${model_dir}")
    done
    mapfile -t MODEL_PACK_SPECS < <(printf '%s\n' "${MODEL_PACK_SPECS[@]}" | sort -V)
fi

if [[ "${NO_MODELS}" -eq 0 && ${#MODEL_PACK_SPECS[@]} -eq 0 ]]; then
    echo "error: no model packs found (expected models_<resolution>, or use --model-pack name=dir)" >&2
    exit 1
fi

declare -A MODEL_PACK_PATHS=()
MODEL_PACK_NAMES=()
for spec in "${MODEL_PACK_SPECS[@]}"; do
    if [[ "${spec}" != *=* ]]; then
        echo "error: invalid --model-pack '${spec}' (expected name=dir)" >&2
        exit 1
    fi
    model_name="${spec%%=*}"
    model_dir="${spec#*=}"
    if [[ ! "${model_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "error: invalid model pack name '${model_name}'" >&2
        exit 1
    fi
    if [[ -n "${MODEL_PACK_PATHS[${model_name}]+x}" ]]; then
        echo "error: duplicate model pack name '${model_name}'" >&2
        exit 1
    fi
    MODEL_PACK_NAMES+=("${model_name}")
    MODEL_PACK_PATHS["${model_name}"]="${model_dir}"
    if [[ "${NO_MODELS}" -eq 0 && ! -d "${model_dir}" ]]; then
        echo "error: model dir '${model_dir}' not found" >&2
        exit 1
    fi
    if [[ "${NO_MODELS}" -eq 0 && ! -f "${model_dir}/intra_analysis_standard.onnx" ]]; then
        echo "error: model pack '${model_name}' is missing intra_analysis_standard.onnx" >&2
        exit 1
    fi
done

# ---- resolve target + platform tag ---------------------------------------
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) HOST_OS="windows";;
    Linux*)               HOST_OS="linux";;
    *)                    HOST_OS="linux";;
esac

if [[ -z "${TARGET}" ]]; then TARGET="${HOST_OS}"; fi

case "${TARGET}" in
    linux)
        PLATFORM="linux-x64"
        BUILD_TAG="linux-x64"
        ORT_PLATFORM="linux-x64"
        ARCHIVE_EXT="tar.gz"
        EXE=""
        ;;
    mingw|windows)
        TARGET="mingw"
        PLATFORM="windows-x64-mingw"
        BUILD_TAG="windows-x64-mingw"
        ORT_PLATFORM="win-x64"
        ARCHIVE_EXT="zip"
        EXE=".exe"
        if [[ "${HOST_OS}" != "windows" ]]; then
            if ! command -v x86_64-w64-mingw32-gcc-posix >/dev/null 2>&1; then
                echo "error: MinGW-w64 not found. Install it:" >&2
                echo "  sudo apt-get install mingw-w64" >&2
                exit 1
            fi
        fi
        ;;
    *) echo "error: --target must be 'linux' or 'mingw'" >&2; exit 1;;
esac
if [[ -n "${DCVC_ORT_GPU:-}" ]]; then
    PLATFORM="${PLATFORM}-gpu"
    BUILD_TAG="${BUILD_TAG}-gpu"
fi

# ---- version (kept in sync with include/dcvc/dcvc_version.h) --------------
VERSION="1.0.0"

PKG_NAME="dcvc-sdk-${VERSION}-${PLATFORM}"
[[ -z "${BUILD_DIR}" ]] && BUILD_DIR="${OUTPUT_ROOT}/build/${BUILD_TAG}"
[[ -z "${OUT_DIR}" ]] && OUT_DIR="${OUTPUT_ROOT}/packages"
STAGING="${OUTPUT_ROOT}/staging/${PKG_NAME}"

echo "==========================================================="
echo " DCVC-SDK package builder"
echo "   target   : ${TARGET} (${PLATFORM})"
echo "   version  : ${VERSION}"
echo "   build    : ${BUILD_DIR}"
echo "   staging  : ${STAGING}"
if [[ "${NO_MODELS}" -eq 1 ]]; then
    MODEL_SUMMARY="<disabled>"
else
    MODEL_SUMMARY="${MODEL_PACK_SPECS[*]}"
fi
echo "   models   : ${MODEL_SUMMARY}"
echo "   out      : ${OUT_DIR}/${PKG_NAME}.${ARCHIVE_EXT}"
echo "==========================================================="

# ---- configure ------------------------------------------------------------
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DDCVC_BUILD_SDK=ON
    -DDCVC_FXP_CUDA="${DCVC_FXP_CUDA:-OFF}"
    -DDCVC_OUTPUT_ROOT="${OUTPUT_ROOT}"
    -DDCVC_ARTIFACT_TAG="${BUILD_TAG}"
)
[[ -n "${DCVC_ORT_GPU:-}" ]]    && CMAKE_ARGS+=(-DDCVC_ORT_GPU=ON)
[[ -n "${DCVC_ORT_ARCHIVE:-}" ]] && CMAKE_ARGS+=(-DDCVC_ORT_ARCHIVE="$(readlink -f "${DCVC_ORT_ARCHIVE}")")
[[ -n "${DCVC_ORT_URL_BASE:-}" ]] && CMAKE_ARGS+=(-DDCVC_ORT_URL_BASE="${DCVC_ORT_URL_BASE}")

if [[ "${TARGET}" == "mingw" && "${HOST_OS}" != "windows" ]]; then
    CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="${SCRIPT_DIR}/mingw-w64-x86_64.toolchain.cmake")
fi

echo "== Configuring =="
cmake -S "${ONNX_DIR}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"

# ---- build ----------------------------------------------------------------
echo "== Building (dcvc + dcvc_demo) =="
cmake --build "${BUILD_DIR}" --target dcvc dcvc_demo --config Release -j"$(nproc)"

# ---- stage via CMake install ---------------------------------------------
# Install into a relocatable prefix (everything lands under STAGING, which
# becomes the package root; the CMake config uses relative paths so it works
# after the folder is moved/copied).
echo "== Staging SDK into ${STAGING} =="
rm -rf "${STAGING}"
mkdir -p "${STAGING}"
cmake --install "${BUILD_DIR}" --prefix "${STAGING}" --config Release >/dev/null

# ---- locate the ORT runtime to bundle alongside the library --------------
ORT_ROOT_DIR="${BUILD_DIR}/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION:-1.27.0}"
# ORT's folder name differs slightly per platform; fall back to a glob.
if [[ ! -d "${ORT_ROOT_DIR}/lib" ]]; then
    ORT_ROOT_DIR="$(find "${BUILD_DIR}" -maxdepth 1 -type d -name 'onnxruntime-*' | head -n1)"
fi
ORT_LIB_SRC="${ORT_ROOT_DIR}/lib"
if [[ ! -d "${ORT_LIB_SRC}" ]]; then
    echo "error: ONNX Runtime lib not found under ${BUILD_DIR}" >&2; exit 1
fi

# ---- post-install copies: ORT runtime, demo, examples, docs --------------
echo "== Copying runtime + demo + examples + docs =="
STAGE_LIB="${STAGING}/lib"
STAGE_BIN="${STAGING}/bin"
mkdir -p "${STAGE_BIN}"

# Demo binary into bin/.
cp -f "${BUILD_DIR}/dcvc_demo${EXE}" "${STAGE_BIN}/"

# ORT shared library next to libdcvc (the lib uses $ORIGIN rpath, so this is
# all that is needed for the runtime to find onnxruntime).
if [[ "${TARGET}" == "mingw" ]]; then
    # Windows: dcvc.dll lands in bin/ (runtime artifact), so co-locate the
    # ORT DLLs there too -- the demo loads all DLLs from its own directory.
    cp -f "${ORT_LIB_SRC}"/onnxruntime.dll "${STAGE_BIN}/" 2>/dev/null || true
    cp -f "${ORT_LIB_SRC}"/onnxruntime_providers_*.dll "${STAGE_BIN}/" 2>/dev/null || true
else
    cp -f "${ORT_LIB_SRC}"/libonnxruntime.so.* "${STAGE_LIB}/" 2>/dev/null || true
    cp -f "${ORT_LIB_SRC}"/libonnxruntime_providers_*.so "${STAGE_LIB}/" 2>/dev/null || true
fi

# Example source + quickstart docs.
mkdir -p "${STAGING}/examples"
cp -f "${ONNX_DIR}/examples/dcvc_demo.c" "${STAGING}/examples/"
cp -f "${ONNX_DIR}/docs/sdk_quickstart.md" "${STAGING}/" 2>/dev/null || true

# ---- models ---------------------------------------------------------------
if [[ "${NO_MODELS}" -eq 0 ]]; then
    echo "== Bundling ${#MODEL_PACK_NAMES[@]} model pack(s) =="
    for model_name in "${MODEL_PACK_NAMES[@]}"; do
        model_dir="${MODEL_PACK_PATHS[${model_name}]}"
        mkdir -p "${STAGING}/models/${model_name}"
        cp -rf "${model_dir}/." "${STAGING}/models/${model_name}/"
    done
fi

MODEL_CONTENTS=""
MODEL_QUICK_TESTS=""
if [[ "${NO_MODELS}" -eq 0 ]]; then
    MODEL_CONTENTS="  models/<name>/     model packs: ${MODEL_PACK_NAMES[*]}"
    for model_name in "${MODEL_PACK_NAMES[@]}"; do
        case "${model_name}" in
            720p)  dims="1280 768";;
            1080p) dims="1920 1088";;
            *)     continue;;
        esac
        if [[ "${PLATFORM}" == *windows* ]]; then
            MODEL_QUICK_TESTS+="  bin\\dcvc_demo.exe models\\${model_name} ${dims} 32 3"$'\n'
        else
            MODEL_QUICK_TESTS+="  LD_LIBRARY_PATH=lib bin/dcvc_demo models/${model_name} ${dims} 32 3"$'\n'
        fi
    done
fi

# ---- top-level README for the package ------------------------------------
cat > "${STAGING}/README.txt" <<EOF
DCVC-SDK ${VERSION} (${PLATFORM})
============================

Portable neural video codec. Cross-platform bit-exact by construction
(FXP fixed-point nets + integer scale->CDF index + rANS).

Contents:
  include/dcvc/      public C headers
  bin/               executables + Windows DLLs (dcvc.dll, onnxruntime.dll)
  lib/               shared/import library + ONNX Runtime (.so on Linux)
  lib/cmake/dcvc/    find_package(dcvc) support
  examples/          dcvc_demo.c source
${MODEL_CONTENTS}

Quick test:
${MODEL_QUICK_TESTS%$'\n'}

Integrate (CMake):
  find_package(dcvc ${VERSION} CONFIG REQUIRED)
  target_link_libraries(myapp PRIVATE dcvc::dcvc)

See sdk_quickstart.md for the full walkthrough.
EOF

# ---- verify nothing critical is missing ----------------------------------
echo "== Sanity check =="
MISSING=0
for f in \
    "include/dcvc/dcvc.h" \
    "lib/dcvc.dll" \
    "lib/dcvc.lib" ; do :; done  # platform-specific checks below

if [[ "${TARGET}" == "mingw" ]]; then
    # Windows runtime layout: DLLs + exe in bin/, import lib in lib/.
    for f in include/dcvc/dcvc.h bin/dcvc_demo.exe \
             lib/cmake/dcvc/dcvcTargets.cmake; do
        [[ -f "${STAGING}/${f}" ]] || { echo "  MISSING: ${f}"; MISSING=1; }
    done
    # The codec DLL is libdcvc.dll (MinGW) or dcvc.dll (MSVC) -- accept either.
    if [[ ! -f "${STAGING}/bin/libdcvc.dll" && ! -f "${STAGING}/bin/dcvc.dll" ]]; then
        echo "  MISSING: bin/libdcvc.dll (or dcvc.dll)"; MISSING=1
    fi
else
    for f in include/dcvc/dcvc.h lib/libdcvc.so.1 lib/cmake/dcvc/dcvcTargets.cmake bin/dcvc_demo; do
        [[ -e "${STAGING}/${f}" ]] || { echo "  MISSING: ${f}"; MISSING=1; }
    done
    if [[ -n "${DCVC_ORT_GPU:-}" ]]; then
        for f in lib/libonnxruntime_providers_shared.so lib/libonnxruntime_providers_cuda.so; do
            [[ -f "${STAGING}/${f}" ]] || { echo "  MISSING: ${f}"; MISSING=1; }
        done
    fi
fi
if [[ "${NO_MODELS}" -eq 0 ]]; then
    for model_name in "${MODEL_PACK_NAMES[@]}"; do
        f="models/${model_name}/intra_analysis_standard.onnx"
        [[ -f "${STAGING}/${f}" ]] || { echo "  MISSING: ${f}"; MISSING=1; }
    done
fi
[[ ${MISSING} -ne 0 ]] && { echo "package incomplete" >&2; exit 1; }
echo "  all expected files present"

# ---- archive --------------------------------------------------------------
mkdir -p "${OUT_DIR}"
ARCHIVE="${OUT_DIR}/${PKG_NAME}.${ARCHIVE_EXT}"
echo "== Archiving -> ${ARCHIVE} =="
if [[ "${ARCHIVE_EXT}" == "zip" ]]; then
    ( cd "${OUTPUT_ROOT}/staging" && zip -qr "${ARCHIVE}" "${PKG_NAME}" )
else
    tar -czf "${ARCHIVE}" -C "${OUTPUT_ROOT}/staging" "${PKG_NAME}"
fi

echo
echo "Done."
echo "  package : ${ARCHIVE}"
echo "  size    : $(du -h "${ARCHIVE}" | cut -f1)"
echo "  contents: $(find "${STAGING}" -type f | wc -l) files"
if [[ "${KEEP_STAGING}" -eq 0 ]]; then
    cmake -E remove_directory "${STAGING}"
    echo "  staging : removed (use --keep-staging to retain it)"
else
    echo "  staging : ${STAGING}"
fi
