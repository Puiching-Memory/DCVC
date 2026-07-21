#!/usr/bin/env python3
"""
Sweep prior (params_fusion) mismatch noise magnitudes and report the rANS
tolerance threshold.  Runs tensorRT/build/test_prior_mismatch with varying
noise_mag and seed, parses SUMMARY lines, and writes a CSV.

Example:
    python tensorRT/tools/sweep_prior_tolerance.py

Output:
    tensorRT/prior_tolerance_results.csv
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

# Repo root (DCVC/). This script lives in <root>/tensorRT/tools/.
ROOT = Path(__file__).resolve().parents[2]
EXEC = ROOT / "tensorRT" / "build" / "test_prior_mismatch"
ASSET_DIR = ROOT / "tensorRT" / "assets"
PLUGIN_DIR = ROOT / "tensorRT" / "build" / "plugin_demo"

NOISE_MAGS = [
    1e-9,
    5e-9,
    1e-8,
    2e-8,
    3e-8,
    4e-8,
    5e-8,
    6e-8,
    8e-8,
    1e-7,
    5e-7,
    1e-6,
]

SEEDS = [0, 1, 2, 3, 4]

SUMMARY_RE = re.compile(
    r"SUMMARY\s+noise_mag=([\d.eE+-]+)\s+seed=(\d+)\s+all_match=(\d+)\s+"
    r"min_match_pct=([\d.eE+-]+)\s+avg_match_pct=([\d.eE+-]+)\s+"
    r"exact_y=(\d+)/(\d+)\s+max_y_err=([\d.eE+-]+)"
)


def run_one(noise_mag: float, seed: int) -> dict:
    """Run the test executable for one (noise, seed) and parse the summary."""
    cmd = [
        str(EXEC),
        str(ASSET_DIR),
        str(PLUGIN_DIR),
        f"{noise_mag:.6e}",
        str(seed),
    ]
    print(f"[running] noise_mag={noise_mag:.6e} seed={seed}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = proc.stdout

    result = {
        "noise_mag": noise_mag,
        "seed": seed,
        "returncode": proc.returncode,
        "all_match": None,
        "min_match_pct": None,
        "avg_match_pct": None,
        "exact_y": None,
        "total_y": None,
        "max_y_err": None,
    }

    if proc.returncode != 0:
        print(f"  CRASH (returncode={proc.returncode})")
        print(out[-800:])
        return result

    m = SUMMARY_RE.search(out)
    if not m:
        print("  no SUMMARY line found")
        print(out[-800:])
        return result

    result.update(
        {
            "all_match": int(m.group(3)),
            "min_match_pct": float(m.group(4)),
            "avg_match_pct": float(m.group(5)),
            "exact_y": int(m.group(6)),
            "total_y": int(m.group(7)),
            "max_y_err": float(m.group(8)),
        }
    )
    print(
        f"  all_match={result['all_match']}  "
        f"min_match_pct={result['min_match_pct']:.2f}%  "
        f"avg_match_pct={result['avg_match_pct']:.2f}%  "
        f"max_y_err={result['max_y_err']:.3e}"
    )
    return result


def summarize(results: list) -> None:
    """Print aggregate statistics and threshold estimates."""
    # Find the largest noise_mag where every seed had all_match==1 and no crash.
    safe = {}
    unsafe = {}
    for r in results:
        if r["returncode"] != 0 or r["all_match"] is None:
            continue
        mag = r["noise_mag"]
        safe.setdefault(mag, {"all_match": True, "min_pct": 100.0})
        unsafe.setdefault(mag, False)
        if not r["all_match"]:
            safe[mag]["all_match"] = False
            unsafe[mag] = True
        safe[mag]["min_pct"] = min(safe[mag]["min_pct"], r["min_match_pct"])

    all_safe_mags = [m for m, v in safe.items() if v["all_match"]]
    all_unsafe_mags = [m for m, v in safe.items() if not v["all_match"]]

    print("\n=== Aggregate ===")
    if all_safe_mags:
        print(f"largest noise_mag with all seeds exact: {max(all_safe_mags):.6e}")
    if all_unsafe_mags:
        print(f"smallest noise_mag with any mismatch: {min(all_unsafe_mags):.6e}")
    if all_safe_mags and all_unsafe_mags:
        lo = max(all_safe_mags)
        hi = min(all_unsafe_mags)
        print(f"tolerance bracket: [{lo:.6e}, {hi:.6e})")

    print("\nper-noise summary:")
    for mag in sorted(safe.keys()):
        v = safe[mag]
        status = "SAFE" if v["all_match"] else "UNSAFE"
        print(f"  {mag:.6e} -> {status} (worst min_match_pct={v['min_pct']:.2f}%)")


def main() -> int:
    if not EXEC.exists():
        print(f"missing executable: {EXEC}", file=sys.stderr)
        return 1

    results = []
    for noise_mag in NOISE_MAGS:
        for seed in SEEDS:
            results.append(run_one(noise_mag, seed))

    out_path = ROOT / "tensorRT" / "prior_tolerance_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "noise_mag",
                "seed",
                "returncode",
                "all_match",
                "min_match_pct",
                "avg_match_pct",
                "exact_y",
                "total_y",
                "max_y_err",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote CSV: {out_path}")

    summarize(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
