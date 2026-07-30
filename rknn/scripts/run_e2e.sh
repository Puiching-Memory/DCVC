#!/usr/bin/env bash
# Power on RK3588 NPU and run e2e. Must run with real root / no sandbox.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$ROOT/models/1080p_i8}"
H="${2:-1088}"; W="${3:-1920}"; QP="${4:-32}"; N="${5:-3}"

if [[ -w /sys/kernel/debug/rknpu/power ]]; then
  echo on > /sys/kernel/debug/rknpu/power
  echo "NPU power=$(cat /sys/kernel/debug/rknpu/power)"
else
  echo "WARN: cannot write rknpu power (sandbox or permissions)" >&2
  cat /sys/kernel/debug/rknpu/power 2>/dev/null || true
fi

pgrep -x rknn_server >/dev/null || (rknn_server >/tmp/rknn_server.log 2>&1 & sleep 1)

BIN="$ROOT/out/build/linux-aarch64/test_rknn_e2e"
if [[ ! -x "$BIN" ]]; then
  bash "$ROOT/scripts/build_linux.sh"
fi

exec "$BIN" "$MODEL" "$H" "$W" "$QP" "$N"
