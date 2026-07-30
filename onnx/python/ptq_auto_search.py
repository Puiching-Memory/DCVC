#!/usr/bin/env python3
"""Automated INT8 quantization parameter search for DCVC-UF entropy nets.

Three search strategies, all injecting per-tensor activation ranges via ORT's
calibration_cache mechanism (bypassing ORT's built-in calibration):

  sweep      — Fine-grained global percentile sweep (20-30 values)
  per-net    — Optuna Bayesian (TPE) search over per-network percentile (7 dims)
  per-tensor — Per-tensor MSE-optimal clipping (analytical, no search needed)
  aciq       — Analytical Clipping for Integer Quantization (kurtosis-based)

Pipeline per config:
  1. Compute per-tensor (min,max) from pre-collected activation histograms
  2. Save as calibration_cache.json
  3. ORT quantize_static reads the cache → quantized model
  4. Assemble all 7 nets + copy → model dir for C binary
  5. C end-to-end RD test (bitrate + PSNR vs FP32)

Usage:
  python ptq_auto_search.py --strategy per-net --trials 60
  python ptq_auto_search.py --strategy per-tensor
  python ptq_auto_search.py --strategy sweep
  python ptq_auto_search.py --strategy aciq
"""
import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys

import numpy as np
import onnx
from onnxruntime.quantization import QuantType, QuantFormat, quantize_static
from onnxruntime.quantization.calibrate import (
    CalibrationMethod, TensorsData, create_calibrator, save_tensors_data,
)

REPO = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.normpath(os.path.join(REPO, ".."))
FP32_DIR = os.path.normpath(os.path.join(REPO, "..", "models"))
C_BIN = os.path.normpath(os.path.join(REPO, "..", "build_127", "test_cpu_end2end"))
HIST_CACHE = os.path.join(REPO, ".hist_cache")
MODEL_CACHE = os.path.join(REPO, "..", "models_autocache")
RESULTS_DIR = os.path.join(REPO, "auto_search_results")

sys.path.insert(0, REPO)
from ptq_quantize import (
    TARGET_NETS, SOURCE_IR_VERSION, SPLIT_OUTPUT_GROUPS,
    split_output_conv, NpyCalibrationReader,
)
from ptq_dump_calib import DEFAULT_FRAMES_DIR, load_frame


# ===== Phase 1: Calibration histogram collection (one-time) =====

def get_split_model(net_name):
    fp32 = os.path.join(FP32_DIR, net_name + ".onnx")
    if net_name not in SPLIT_OUTPUT_GROUPS:
        return fp32
    os.makedirs(HIST_CACHE, exist_ok=True)
    split = os.path.join(HIST_CACHE, f"{net_name}_split.onnx")
    if not os.path.exists(split):
        split_output_conv(fp32, SPLIT_OUTPUT_GROUPS[net_name], split)
    return split


def collect_histograms(net_name, force=False):
    cache = os.path.join(HIST_CACHE, f"{net_name}.pkl")
    if not force and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)

    os.makedirs(HIST_CACHE, exist_ok=True)
    model_path = get_split_model(net_name)
    calib_dir = os.path.join(REPO, "calib_data", net_name)
    aug = os.path.join(HIST_CACHE, f"{net_name}_aug.onnx")

    calibrator = create_calibrator(
        model_path,
        op_types_to_calibrate=["Conv", "MatMul", "Gemm"],
        calibrate_method=CalibrationMethod.Percentile,
        augmented_model_path=aug,
        extra_options={"symmetric": False},
    )
    reader = NpyCalibrationReader(calib_dir)
    calibrator.collect_data(reader)

    hists = {}
    for t, hd in calibrator.collector.histogram_dict.items():
        hists[t] = (np.asarray(hd[0]), np.asarray(hd[1]),
                     float(hd[2]), float(hd[3]))
    with open(cache, "wb") as f:
        pickle.dump(hists, f)
    if os.path.exists(aug):
        os.remove(aug)
    print(f"  {net_name}: {len(hists)} tensors")
    return hists


def collect_all_histograms():
    print("Collecting calibration histograms (one-time)...")
    all_hists = {}
    for net in TARGET_NETS:
        all_hists[net] = collect_histograms(net)
    total = sum(len(h) for h in all_hists.values())
    print(f"  Total: {total} tensors across {len(TARGET_NETS)} networks")
    return all_hists


# ===== Phase 2: Range computation =====

def percentile_range(hist, edges, mn, mx, pct, symmetric=False):
    """Compute (low, high) at percentile pct from a SIGNED histogram.

    symmetric=False (default): asymmetric range [low, high].
    symmetric=True: symmetric range [-t, t] where t = pct-th percentile of |values|.
    """
    total = max(hist.sum(), 1)
    cdf = np.cumsum(hist / total)
    if symmetric:
        # Build absolute-value CDF from signed histogram
        centers = (edges[:-1] + edges[1:]) / 2.0
        abs_centers = np.abs(centers)
        order = np.argsort(abs_centers)
        sorted_counts = hist[order]
        sorted_abs = abs_centers[order]
        abs_cdf = np.cumsum(sorted_counts / total)
        idx = int(np.searchsorted(abs_cdf, pct / 100.0))
        t = float(sorted_abs[min(idx, len(sorted_abs) - 1)])
        return -t, t
    else:
        cut = (100.0 - pct) / 200.0
        i_lo = int(np.searchsorted(cdf, cut))
        i_hi = int(np.searchsorted(cdf, 1.0 - cut))
        lo = float(edges[min(i_lo, len(edges) - 1)])
        hi = float(edges[min(i_hi, len(edges) - 1)])
        if mn is not None and lo < mn:
            lo = mn
        if mx is not None and hi > mx:
            hi = mx
        return lo, hi


def ranges_uniform(hists, pct, symmetric=False):
    return {t: percentile_range(h, e, mn, mx, pct, symmetric=symmetric)
            for t, (h, e, mn, mx) in hists.items()}


def ranges_per_net(all_hists, net_percentiles):
    out = {}
    for net, pct in net_percentiles.items():
        out.update(ranges_uniform(all_hists[net], pct))
    return out


def ranges_per_tensor_mse(hists, pcts=None):
    """Per-tensor MSE-optimal clipping using symmetric percentile.

    For each tensor, finds the symmetric threshold that minimizes expected
    u8 quantization MSE on the calibration histogram.
    """
    if pcts is None:
        pcts = np.concatenate([
            np.arange(99.0, 99.9, 0.05),
            np.arange(99.9, 99.99, 0.005),
            np.arange(99.99, 99.999, 0.001),
        ])
    out = {}
    for t, (h, e, mn, mx) in hists.items():
        best_pct, best_mse = 99.999, float("inf")
        centers = (e[:-1] + e[1:]) / 2.0
        counts = h.astype(np.float64)
        total = max(counts.sum(), 1)
        for pct in pcts:
            lo, hi = percentile_range(h, e, mn, mx, pct, symmetric=True)
            if hi <= lo:
                continue
            step = (hi - lo) / 255.0
            # u8 symmetric: zero_point=128, values mapped to [-128*s, 127*s]
            q = np.clip(np.round(centers / step) * step, lo, hi)
            mse = float(np.sum(counts * (q - centers) ** 2) / total)
            if mse < best_mse:
                best_mse = mse
                best_pct = pct
        out[t] = percentile_range(h, e, mn, mx, best_pct, symmetric=True)
    return out


def ranges_aciq(hists):
    """Analytical Clipping for Integer Quantization (MSE-optimal, u8 asymmetric).

    For each tensor, estimate distribution from kurtosis and apply the known
    optimal clipping factor from the ACIQ paper (Table 1, 8-bit, MSE criterion):

      Gaussian (kurtosis≈3): clip at 4.0 * sigma
      Laplacian (kurtosis≈6): clip at 4.5 * b  (b = sigma / sqrt(2))

    Linear interpolation in kurtosis space between the two regimes.
    """
    out = {}
    for t, (h, e, mn, mx) in hists.items():
        centers = (e[:-1] + e[1:]) / 2.0
        counts = h.astype(np.float64)
        total = max(counts.sum(), 1)
        w = counts / total
        mean = float(np.sum(w * centers))
        var = float(np.sum(w * (centers - mean) ** 2))
        std = max(var ** 0.5, 1e-8)
        fourth = float(np.sum(w * (centers - mean) ** 4))
        kurt = fourth / max(var ** 2, 1e-30)  # 3=Gaussian, 6=Laplacian
        # Interpolate alpha between Gaussian (kurt=3 → alpha=4.0*sigma)
        # and Laplacian (kurt=6 → alpha=4.5*b=4.5*sigma/sqrt(2))
        t_k = np.clip((kurt - 3.0) / 3.0, 0, 1)  # 0=gauss, 1=laplace
        sigma_clip = 4.0
        laplace_b_clip = 4.5 / (2 ** 0.5)  # ≈ 3.18 sigma
        alpha = sigma_clip * (1 - t_k) + laplace_b_clip * t_k
        lo = mean - alpha * std
        hi = mean + alpha * std
        out[t] = (float(lo), float(hi))
    return out



def quantize_net_percentile(net_name, percentile, symmetric=True):
    """Quantize using ORT's built-in Percentile calibration (matches ablation study).

    This is the reference method — uses ORT's full calibration pipeline.
    With symmetric=True it exactly replicates the ablation study behavior.
    """
    tag = f"ort_pct{percentile:.4f}_sym{int(symmetric)}"
    out_dir = os.path.join(MODEL_CACHE, f"{net_name}_{tag}")
    out_path = os.path.join(out_dir, net_name + ".onnx")
    if os.path.exists(out_path):
        return out_path

    os.makedirs(out_dir, exist_ok=True)
    model_path = get_split_model(net_name)
    calib_dir = os.path.join(REPO, "calib_data", net_name)
    reader = NpyCalibrationReader(calib_dir)

    quantize_static(
        model_input=model_path,
        model_output=out_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.Percentile,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        extra_options={"CalibPercentile": percentile, "symmetric": symmetric},
    )
    model = onnx.load(out_path, load_external_data=False)
    if model.ir_version != SOURCE_IR_VERSION:
        model.ir_version = SOURCE_IR_VERSION
        onnx.save(model, out_path)
    return out_path


def assemble_from_percentiles(net_percentiles, symmetric=True):
    """Assemble model dir using ORT-native per-network percentile quantization."""
    net_paths = {}
    for net, pct in net_percentiles.items():
        net_paths[net] = quantize_net_percentile(net, pct, symmetric=symmetric)
    return assemble_model_dir(net_paths)




# ===== Parallel helpers =====

def _quantize_one_net_task(args):
    """Module-level worker: quantize a single network (for ProcessPoolExecutor)."""
    net, pct, symmetric = args
    return net, quantize_net_percentile(net, pct, symmetric=symmetric)


def quantize_nets_parallel(net_percentiles, symmetric=True, workers=None):
    """Quantize all 7 networks in parallel using ProcessPoolExecutor."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    tasks = [(net, pct, symmetric) for net, pct in net_percentiles.items()]
    results = {}
    if workers is None:
        workers = min(len(tasks), mp.cpu_count() // 4 or 1)
    if workers <= 1:
        for task in tasks:
            net, path = _quantize_one_net_task(task)
            results[net] = path
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for net, path in pool.map(_quantize_one_net_task, tasks):
                results[net] = path
    return results


def assemble_from_percentiles_parallel(net_percentiles, symmetric=True, workers=7):
    """Assemble model dir with parallel network quantization."""
    net_paths = quantize_nets_parallel(net_percentiles, symmetric=symmetric, workers=workers)
    return assemble_model_dir(net_paths)


# ===== Parallel trial evaluation (module-level for pickling) =====

PARALLEL_FRAMES = None
PARALLEL_QPS = None


def _evaluate_trial_task(trial_params):
    """Evaluate a single trial: {net: pct, ...} → {ratio, dPSNR, ...}.

    Module-level so it can be pickled for multiprocessing.
    """
    pcts = {k: v for k, v in trial_params.items() if k in TARGET_NETS}
    pct_min = min(trial_params.get("pct_min", 99.9))
    pct_max = max(trial_params.get("pct_max", 99.999))
    workers_q = trial_params.get("workers_q", 7)
    frames = PARALLEL_FRAMES or ["f00050"]
    qps = PARALLEL_QPS or [32]

    try:
        mdir = assemble_from_percentiles_parallel(pcts, symmetric=True, workers=workers_q)
        rd = evaluate(mdir, frames, qps)
        return {"ok": True, **rd}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ===== Phase 3: Quantization with custom ranges =====

def quantize_with_ranges(net_name, ranges):
    rh = hashlib.md5(json.dumps(
        {k: [round(v[0], 8), round(v[1], 8)]
         for k, v in sorted(ranges.items())},
        sort_keys=True,
    ).encode()).hexdigest()[:12]

    out_dir = os.path.join(MODEL_CACHE, f"{net_name}_{rh}")
    out_path = os.path.join(out_dir, net_name + ".onnx")
    if os.path.exists(out_path):
        return out_path

    os.makedirs(out_dir, exist_ok=True)
    model_path = get_split_model(net_name)

    cache_path = os.path.join(out_dir, "calib.json")
    td = TensorsData(CalibrationMethod.MinMax, {
        t: (np.float32(lo), np.float32(hi)) for t, (lo, hi) in ranges.items()
    })
    save_tensors_data(td, cache_path)

    quantize_static(
        model_input=model_path,
        model_output=out_path,
        calibration_data_reader=None,
        calibration_cache_path=cache_path,
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )

    model = onnx.load(out_path, load_external_data=False)
    if model.ir_version != SOURCE_IR_VERSION:
        model.ir_version = SOURCE_IR_VERSION
        onnx.save(model, out_path)
    return out_path


def assemble_model_dir(net_paths):
    dh = hashlib.md5(
        json.dumps({k: os.path.basename(os.path.dirname(v)) for k, v in sorted(net_paths.items())},
                   sort_keys=True).encode()
    ).hexdigest()[:8]
    mdir = os.path.join(MODEL_CACHE, f"asm_{dh}")
    marker = os.path.join(mdir, "bitest_cdf.npy")
    if os.path.exists(marker):
        return mdir
    os.makedirs(mdir, exist_ok=True)
    for f in os.listdir(FP32_DIR):
        s = os.path.join(FP32_DIR, f)
        if os.path.isfile(s):
            d = os.path.join(mdir, f)
            if not os.path.exists(d):
                shutil.copy2(s, d)
    for net, path in net_paths.items():
        shutil.copy2(path, os.path.join(mdir, net + ".onnx"))
    return mdir


# ===== Phase 4: RD evaluation =====

def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def _roundtrip(model_dir, x_path, rec_path, s, qp):
    proc = subprocess.run(
        [C_BIN, model_dir, str(s), str(s), str(qp), x_path, rec_path],
        capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0 or "PASS" not in proc.stdout:
        raise RuntimeError(f"roundtrip fail:\n{proc.stdout}\n{proc.stderr}")
    return int(re.search(r"stream_size=(\d+) bytes", proc.stdout).group(1))


def evaluate(model_dir, frames, qps, crop=512):
    rd = os.path.join(RESULTS_DIR, "rd_tmp")
    os.makedirs(rd, exist_ok=True)
    res = []
    for frame in frames:
        x = load_frame(DEFAULT_FRAMES_DIR, frame, (crop, crop))
        xp = os.path.join(rd, f"{frame}.npy")
        np.save(xp, x)
        for qp in qps:
            r32 = os.path.join(rd, f"{frame}_{qp}_f32.npy")
            r8 = os.path.join(rd, f"{frame}_{qp}_i8.npy")
            b32 = _roundtrip(FP32_DIR, xp, r32, crop, qp)
            b8 = _roundtrip(model_dir, xp, r8, crop, qp)
            p32 = _psnr(x, np.load(r32))
            p8 = _psnr(x, np.load(r8))
            res.append((b32, b8, p32, p8))
    n = len(res)
    avg_db = sum(100 * (b8 - b32) / b32 for b32, b8, _, _ in res) / n
    avg_dp = sum(p8 - p32 for _, _, p32, p8 in res) / n
    ratio = sum(b8 for _, b8, _, _ in res) / max(1, sum(b32 for b32, _, _, _ in res))
    return {"dBytes_pct": avg_db, "ratio": ratio, "dPSNR": avg_dp, "n": n}


# ===== Phase 5: Search strategies =====

def run_sweep(all_hists, frames, qps, pmin=99.0, pmax=99.999, npts=25):
    pcts = np.linspace(pmin, pmax, npts)
    results = []
    for pct in pcts:
        try:
            mdir = assemble_from_percentiles(
                {net: float(pct) for net in TARGET_NETS}, symmetric=True)
            rd = evaluate(mdir, frames, qps)
        except Exception as e:
            print(f"  pct={pct:.3f}: FAIL {e}")
            continue
        results.append({"pct": float(pct), **rd})
        print(f"  pct={pct:.4f}: dBytes={rd['dBytes_pct']:+.1f}% "
              f"ratio={rd['ratio']:.3f} dPSNR={rd['dPSNR']:+.3f} dB")
    return results


def run_per_tensor_mse(all_hists, frames, qps):
    net_paths = {}
    for net in TARGET_NETS:
        r = ranges_per_tensor_mse(all_hists[net])
        net_paths[net] = quantize_with_ranges(net, r)
        print(f"  {net} quantized")
    mdir = assemble_model_dir(net_paths)
    rd = evaluate(mdir, frames, qps)
    print(f"  per-tensor MSE: dBytes={rd['dBytes_pct']:+.1f}% "
          f"ratio={rd['ratio']:.3f} dPSNR={rd['dPSNR']:+.3f} dB")
    return {"strategy": "per-tensor-mse", **rd}


def run_aciq(all_hists, frames, qps):
    net_paths = {}
    for net in TARGET_NETS:
        r = ranges_aciq(all_hists[net])
        net_paths[net] = quantize_with_ranges(net, r)
        print(f"  {net} quantized (ACIQ)")
    mdir = assemble_model_dir(net_paths)
    rd = evaluate(mdir, frames, qps)
    print(f"  ACIQ: dBytes={rd['dBytes_pct']:+.1f}% "
          f"ratio={rd['ratio']:.3f} dPSNR={rd['dPSNR']:+.3f} dB")
    return {"strategy": "aciq", **rd}


def run_per_net_optuna(all_hists, frames, qps, n_trials=60, alpha=0.5):
    """Optuna TPE search: each of 7 networks gets its own percentile.

    Uses ORT-native Percentile calibration (symmetric=True) for reliability.
    Quantized models are cached by (network, percentile) so only changed
    networks are re-quantized per trial.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pareto = []

    def objective(trial):
        pcts = {net: trial.suggest_float(net, 99.0, 99.999) for net in TARGET_NETS}
        try:
            mdir = assemble_from_percentiles(pcts, symmetric=True)
            rd = evaluate(mdir, frames, qps)
        except Exception as exc:
            print(f"  trial {trial.number}: FAIL {exc}")
            return 1e6
        obj = rd["ratio"] + alpha * (-rd["dPSNR"])
        pareto.append({"trial": trial.number, **pcts, **rd, "obj": obj})
        dominated = any(
            p["ratio"] <= rd["ratio"] and p["dPSNR"] >= rd["dPSNR"]
            and (p["ratio"] < rd["ratio"] or p["dPSNR"] > rd["dPSNR"])
            for p in pareto if p["trial"] != trial.number
        )
        tag = "" if dominated else " *PARETO*"
        pct_str = " ".join(f"{n[:4]}={v:.3f}" for n, v in pcts.items())
        print(f"  trial {trial.number}: obj={obj:.4f} ratio={rd['ratio']:.3f} "
              f"dPSNR={rd['dPSNR']:+.3f} | {pct_str}{tag}")
        return obj

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = min(pareto, key=lambda p: p["obj"])
    print(f"\nBest trial #{best['trial']}: obj={best['obj']:.4f}")
    for net in TARGET_NETS:
        print(f"  {net}: pct={best[net]:.4f}")
    print(f"  ratio={best['ratio']:.3f} dPSNR={best['dPSNR']:+.3f} dB")

    return {"strategy": "per-net-optuna", "best": best, "pareto": pareto,
            "all_trials": pareto}




# ===== Parallel Optuna search =====

def run_per_net_optuna_parallel(all_hists, frames, qps, n_trials=60,
                                alpha=0.5, n_workers=8, pct_min=99.9, pct_max=99.999):
    """Parallel Optuna TPE search using multiprocessing.

    Runs n_workers processes simultaneously, each executing trials
    against a shared JournalStorage-backed study. Within each trial,
    the 7-network quantization is also parallelized.

    n_trials is divided across workers.
    """
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileStorage
    import multiprocessing as mp
    import tempfile
    from concurrent.futures import ProcessPoolExecutor, as_completed

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    journal_path = os.path.join(RESULTS_DIR, f"optuna_journal_{os.getpid()}.log")

    # Create shared study
    storage = JournalStorage(JournalFileStorage(journal_path))
    study = optuna.create_study(
        direction="minimize",
        storage=storage,
        study_name="per_net_pct",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    # Set global frames/qps for worker processes
    global PARALLEL_FRAMES, PARALLEL_QPS
    PARALLEL_FRAMES = frames
    PARALLEL_QPS = qps

    trials_per_worker = max(1, n_trials // n_workers)

    def worker_optimize(seed_offset):
        """Run trials in a subprocess connected to the shared study."""
        import optuna as opt
        from optuna.storages import JournalStorage as JS
        from optuna.storages.journal import JournalFileStorage as JFS
        opt.logging.set_verbosity(opt.logging.WARNING)

        storage = JS(JFS(journal_path))
        study = opt.load_study(study_name="per_net_pct", storage=storage)

        def objective(trial):
            pcts = {net: trial.suggest_float(net, pct_min, pct_max) for net in TARGET_NETS}
            mdir = assemble_from_percentiles_parallel(pcts, symmetric=True, workers=7)
            rd = evaluate(mdir, frames, qps)
            obj = rd["ratio"] + alpha * (-rd["dPSNR"])
            return obj

        study.optimize(objective, n_trials=trials_per_worker, show_progress_bar=False)

    print(f"Launching {n_workers} workers × {trials_per_worker} trials/worker "
          f"= {n_workers * trials_per_worker} total trials")
    print(f"  Search range: [{pct_min}, {pct_max}], alpha={alpha}")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(worker_optimize, i) for i in range(n_workers)]
        for i, f in enumerate(as_completed(futures)):
            print(f"  worker {i} done")

    # Collect results from the shared study
    all_trials = []
    study = optuna.load_study(
        study_name="per_net_pct",
        storage=JournalStorage(JournalFileStorage(journal_path)),
    )
    for t in study.trials:
        if t.state.name != "COMPLETE":
            continue
        pcts = {net: t.params[net] for net in TARGET_NETS}
        obj = t.value
        # Re-evaluate best trial for detailed RD metrics
        all_trials.append({"trial": t.number, **pcts, "obj": obj})

    # Find best by objective
    best_trial = min(all_trials, key=lambda p: p["obj"])
    # Re-evaluate best config for full RD stats
    mdir_best = assemble_from_percentiles_parallel(
        {n: best_trial[n] for n in TARGET_NETS}, symmetric=True, workers=7)
    rd_best = evaluate(mdir_best, frames, qps)
    best_trial.update(rd_best)

    # Pareto front
    pareto = []
    for t in all_trials:
        if t.get("ratio") is None:
            # Need to evaluate for Pareto - skip (only evaluate top candidates)
            continue
        dominated = any(
            p["ratio"] <= t["ratio"] and p["dPSNR"] >= t["dPSNR"]
            and (p["ratio"] < t["ratio"] or p["dPSNR"] > t["dPSNR"])
            for p in pareto
        )
        if not dominated:
            pareto.append(t)

    print(f"\nBest trial #{best_trial['trial']}: obj={best_trial['obj']:.4f}")
    for net in TARGET_NETS:
        print(f"  {net}: pct={best_trial[net]:.4f}")
    print(f"  ratio={best_trial['ratio']:.3f} dPSNR={best_trial['dPSNR']:+.3f} dB")
    print(f"  Total trials completed: {len(all_trials)}")

    # Clean up journal
    if os.path.exists(journal_path):
        os.remove(journal_path)

    return {"strategy": "per-net-optuna-parallel", "best": best_trial,
            "pareto": pareto, "n_trials": len(all_trials),
            "n_workers": n_workers}

# ===== Main =====

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="per-net",
                    choices=["sweep", "per-net", "per-net-parallel", "per-tensor", "aciq", "all"])
    ap.add_argument("--trials", type=int, default=60,
                    help="Optuna trials for per-net strategy")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Objective weight: minimize ratio + alpha * (-dPSNR)")
    ap.add_argument("--frames", nargs="+", default=["f00050", "f00200", "f00400"])
    ap.add_argument("--qps", nargs="+", type=int, default=[22, 32, 42])
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--quick", action="store_true",
                    help="1 frame x 1 QP for fast search")
    ap.add_argument("--force-calib", action="store_true",
                    help="Re-collect calibration histograms")
    ap.add_argument("--workers", type=int, default=8,
                    help="Number of parallel Optuna worker processes")
    ap.add_argument("--pct-min", type=float, default=99.9,
                    help="Lower bound for percentile search")
    ap.add_argument("--pct-max", type=float, default=99.999,
                    help="Upper bound for percentile search")
    args = ap.parse_args()

    frames = ["f00050"] if args.quick else args.frames
    qps = [32] if args.quick else args.qps
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_hists = collect_all_histograms()

    results = {}
    strat = args.strategy
    if strat in ("sweep", "all"):
        print("\n=== Strategy: sweep ===")
        results["sweep"] = run_sweep(all_hists, frames, qps)
    if strat in ("per-tensor", "all"):
        print("\n=== Strategy: per-tensor MSE ===")
        results["per-tensor-mse"] = run_per_tensor_mse(all_hists, frames, qps)
    if strat in ("aciq", "all"):
        print("\n=== Strategy: ACIQ ===")
        results["aciq"] = run_aciq(all_hists, frames, qps)
    if strat in ("per-net", "per-net-parallel", "all"):
        n_workers = args.workers if args.workers > 0 else os.cpu_count() // 8
        print(f"\n=== Strategy: per-net Optuna ({args.trials} trials, "
              f"{n_workers} workers, range=[{args.pct_min},{args.pct_max}]) ===")
        results["per-net"] = run_per_net_optuna_parallel(
            all_hists, frames, qps, n_trials=args.trials, alpha=args.alpha,
            n_workers=n_workers, pct_min=args.pct_min, pct_max=args.pct_max)

    out_path = os.path.join(RESULTS_DIR, "auto_search_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'strategy':20s} {'dBytes%':>10s} {'ratio':>7s} {'dPSNR':>10s}")
    print("-" * 50)
    if "sweep" in results:
        best_sweep = min(results["sweep"], key=lambda r: r["ratio"])
        print(f"{'sweep(best)':20s} {best_sweep['dBytes_pct']:+9.1f}% "
              f"{best_sweep['ratio']:7.3f} {best_sweep['dPSNR']:+9.3f} dB "
              f"(pct={best_sweep['pct']:.3f})")
    for key, label in [("per-tensor-mse", "per-tensor-mse"), ("aciq", "aciq")]:
        if key in results:
            r = results[key]
            print(f"{label:20s} {r['dBytes_pct']:+9.1f}% "
                  f"{r['ratio']:7.3f} {r['dPSNR']:+9.3f} dB")
    if "per-net" in results:
        b = results["per-net"]["best"]
        print(f"{'per-net-optuna':20s} {b['dBytes_pct']:+9.1f}% "
              f"{b['ratio']:7.3f} {b['dPSNR']:+9.3f} dB")
    print()
    print("Baseline (from ablation):")
    print(f"{'pct9999_split':20s} {'+73.7':>9s}% {1.814:7.3f} {'-0.747':>9s} dB")
    print(f"{'minmax_split':20s} {'+245.9':>9s}% {3.206:7.3f} {'-1.890':>9s} dB")


if __name__ == "__main__":
    raise SystemExit(main())
