#!/usr/bin/env bash
# Run a DCVC ONNX GPU binary with cuDNN visible to ORT's CUDA EP.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
# Prefer repo venv; fall back to hard path under site-packages.
CANDIDATES=(
  "${REPO}/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
  "${REPO}/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib"
  "${REPO}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
)
CUDNN_LIB=""
for d in "${CANDIDATES[@]}"; do
  if [[ -f "${d}/libcudnn.so.9" ]]; then CUDNN_LIB="$d"; break; fi
done
if [[ -z "${CUDNN_LIB}" ]]; then
  CUDNN_LIB="$(python3 - <<'PY'
import glob, os, sys
for sp in sys.path:
    hits = glob.glob(os.path.join(sp, "nvidia", "cudnn", "lib", "libcudnn.so.9"))
    if hits:
        print(os.path.dirname(hits[0])); break
PY
)"
fi
if [[ -n "${CUDNN_LIB}" && -d "${CUDNN_LIB}" ]]; then
  export LD_LIBRARY_PATH="${CUDNN_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  echo "warning: libcudnn.so.9 not found; CUDA EP may fail to load" >&2
fi
exec "$@"
