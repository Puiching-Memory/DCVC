#!/usr/bin/env python3
"""QAT fine-tune of INT8 backbone (analysis / hyper_enc / synthesis).

Uses the DCVC-UF DMCI checkpoint (384-ch, matches onnx/models). Entropy nets
stay FP32 (frozen); only Conv2d inside enc / hyper_enc / dec get fake-quant
(STE, per-tensor act u8 + per-channel weight s8), matching ORT QLinearConv.

Pipeline:
  1) Load UF checkpoint, wrap backbone Convs with FakeQuant
  2) Distill + recon loss on random crops (YCbCr, matches C codec)
  3) Export FP32-but-QAT-aware backbone ONNX
  4) Native ORT INT8 PTQ → hybrid pack (FXP entropy + QAT-int8 backbone)
  5) RD compare vs baseline and PTQ-only hybrid

Usage:
  uv run python python/qat_backbone_i8.py --steps 400 --crop 256
"""
from __future__ import annotations

import argparse
import copy
import glob
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.normpath(os.path.join(REPO, ".."))
ROOT = os.path.normpath(os.path.join(ONNX_DIR, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, REPO)

os.environ.setdefault("SUPPRESS_CUSTOM_KERNEL_WARNING", "1")

from src.models.image_model import DMCI, g_ch_enc_dec  # noqa: E402
from ptq_dump_calib import rgb_to_ycbcr_c  # noqa: E402
from ptq_hybrid_backbone_i8 import (  # noqa: E402
    BACKBONE_NETS,
    assemble_hybrid,
    eval_rd,
    quantize_backbone,
)

C_BIN = os.path.join(ONNX_DIR, "build", "test_cpu_end2end")


# ---------------- FakeQuant ----------------

def _ste_round(x):
    return (x.round() - x).detach() + x


class FakeQuantConv2d(nn.Module):
    """Drop-in Conv2d with STE fake-quant (u8 act / s8 per-channel weight)."""

    def __init__(self, conv: nn.Conv2d, momentum=0.01):
        super().__init__()
        self.conv = conv
        self.momentum = momentum
        cout = conv.weight.shape[0]
        self.register_buffer("w_scale", torch.ones(cout))
        self.register_buffer("x_scale", torch.ones(1))
        self.register_buffer("x_absmax", torch.ones(1))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))

    def _update_x(self, x):
        am = x.detach().abs().amax()
        am = torch.clamp(am, min=1e-8)
        if not bool(self.initialized):
            self.x_absmax.copy_(am)
            self.initialized.fill_(True)
        else:
            self.x_absmax.mul_(1 - self.momentum).add_(am * self.momentum)
        self.x_scale.copy_(self.x_absmax / 127.0)

    def _fake_x(self, x):
        self._update_x(x)
        s = self.x_scale.clamp(min=1e-8)
        q = _ste_round(x / s).clamp(-128, 127)
        return q * s

    def _fake_w(self):
        w = self.conv.weight
        flat = w.detach().abs().reshape(w.shape[0], -1).amax(dim=1).clamp(min=1e-8)
        self.w_scale.copy_(flat / 127.0)
        s = self.w_scale.view(-1, *([1] * (w.ndim - 1))).clamp(min=1e-8)
        q = _ste_round(w / s).clamp(-127, 127)
        return q * s

    def forward(self, x):
        x = self._fake_x(x)
        w = self._fake_w()
        return F.conv2d(
            x, w, self.conv.bias,
            stride=self.conv.stride, padding=self.conv.padding,
            dilation=self.conv.dilation, groups=self.conv.groups,
        )


def _replace_modules(root: nn.Module, pred, factory):
    """Replace modules matching pred(name, mod) under root; returns count."""
    to_fix = []
    for name, child in root.named_modules():
        if name == "" or not pred(name, child):
            continue
        to_fix.append(name)
    for name in to_fix:
        parts = name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old = getattr(parent, parts[-1])
        setattr(parent, parts[-1], factory(old))
    return len(to_fix)


def wrap_backbone_convs(model: DMCI):
    """Replace Conv2d under enc / hyper_enc / dec with FakeQuantConv2d."""
    n = 0
    for root in (model.enc, model.hyper_enc, model.dec):
        n += _replace_modules(
            root,
            lambda _n, m: isinstance(m, nn.Conv2d) and not isinstance(m, FakeQuantConv2d),
            FakeQuantConv2d,
        )
    return n


def unwrap_backbone_convs(model: DMCI):
    """Restore plain Conv2d (keep learned float weights) for ONNX export."""
    n = 0
    for root in (model.enc, model.hyper_enc, model.dec):
        n += _replace_modules(
            root,
            lambda _n, m: isinstance(m, FakeQuantConv2d),
            lambda m: m.conv,
        )
    return n


# ---------------- Data / forward ----------------

def load_png_rgb(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)  # CHW


def random_crop_ycbcr(imgs, crop, rng):
    """imgs: list of CHW float RGB. Return [1,3,crop,crop] YCbCr torch."""
    im = imgs[int(rng.integers(0, len(imgs)))]
    _, H, W = im.shape
    assert H >= crop and W >= crop
    top = int(rng.integers(0, H - crop + 1))
    left = int(rng.integers(0, W - crop + 1))
    patch = im[:, top : top + crop, left : left + crop]
    if rng.random() < 0.5:
        patch = patch[:, :, ::-1].copy()
    ycbcr = rgb_to_ycbcr_c(patch)
    return torch.from_numpy(np.ascontiguousarray(ycbcr[None]))


@torch.no_grad()
def teacher_latents(model, x, qp):
    q_enc = model.q_scale_enc[qp : qp + 1]
    q_dec = model.q_scale_dec[qp : qp + 1]
    y = model.enc(x, q_enc)
    y_pad = model.pad_for_y(y)
    z = model.hyper_enc(y_pad)
    z_hat = torch.round(z)
    params = model.hyper_dec(z_hat)
    params = model.y_prior_fusion(params)
    _, _, yH, yW = y.shape
    params = params[:, :, :yH, :yW].contiguous()
    *_, y_hat = model.compress_prior_4x(
        y, params, model.y_spatial_prior_reduction,
        model.y_spatial_prior_adaptor_1, model.y_spatial_prior_adaptor_2,
        model.y_spatial_prior_adaptor_3, model.y_spatial_prior,
    )
    x_hat = model.dec(y_hat, q_dec).clamp(0, 1)
    return y, z, y_hat, x_hat


def student_loss(model, x, qp, y_t, z_t, y_hat_t, x_t):
    q_enc = model.q_scale_enc[qp : qp + 1]
    q_dec = model.q_scale_dec[qp : qp + 1]
    y = model.enc(x, q_enc)
    z = model.hyper_enc(model.pad_for_y(y))
    # Train decoder on teacher y_hat (stable); also recon to input.
    x_hat = model.dec(y_hat_t, q_dec)
    mse = nn.MSELoss()
    loss_y = mse(y, y_t)
    loss_z = mse(z, z_t)
    loss_rec = mse(x_hat, x)
    loss_distill = mse(x_hat, x_t)
    # Weight: recon dominates; latent match keeps analysis/hyper aligned for PTQ.
    loss = loss_rec + 0.5 * loss_distill + 0.1 * loss_y + 0.05 * loss_z
    stats = {
        "loss": float(loss.detach()),
        "rec": float(loss_rec.detach()),
        "y": float(loss_y.detach()),
        "z": float(loss_z.detach()),
    }
    return loss, stats


def freeze_entropy(model):
    for name, p in model.named_parameters():
        if name.startswith(("enc.", "hyper_enc.", "dec.", "q_scale_")):
            p.requires_grad = True
        else:
            p.requires_grad = False


def force_torch_path():
    """CUSTOMIZED_CUDA_INFERENCE is imported by value in several modules — patch all."""
    import src.layers.cuda_inference as ci
    import src.layers.layers as layers
    import src.models.image_model as imod

    ci.CUSTOMIZED_CUDA_INFERENCE = False
    layers.CUSTOMIZED_CUDA_INFERENCE = False
    imod.CUSTOMIZED_CUDA_INFERENCE = False


def load_model(ckpt_path, device):
    force_torch_path()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model = DMCI()
    model.load_state_dict(sd, strict=True)
    model.to(device)
    return model


def export_backbone(model, out_dir, opset=26):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    H = W = 256
    N = 256

    def export_torch(net, args, path, in_names):
        if not isinstance(args, (tuple, list)):
            args = (args,)
        h = torch.export.Dim("h", min=4)
        w = torch.export.Dim("w", min=4)
        dynamic_shapes = tuple(
            {2: h, 3: w} if i == 0 else None for i, a in enumerate(args) if a.dim() == 4
        )
        torch.onnx.export(
            net, args, path,
            input_names=in_names, output_names=["out0"],
            dynamic_shapes=dynamic_shapes, opset_version=opset, dynamo=True,
            external_data=False,
        )
        print("exported", path)

    device = next(model.parameters()).device
    # export on CPU for dynamo stability
    model_cpu = copy.deepcopy(model).cpu().eval()
    export_torch(
        model_cpu.enc,
        (torch.randn(1, 3, H, W), torch.ones(1, g_ch_enc_dec, 1, 1)),
        os.path.join(out_dir, "intra_analysis_standard.onnx"),
        ["in0", "in1"],
    )
    export_torch(
        model_cpu.hyper_enc,
        torch.randn(1, N, H // 16, W // 16),
        os.path.join(out_dir, "intra_hyper_enc.onnx"),
        ["in0"],
    )
    export_torch(
        model_cpu.dec,
        (torch.randn(1, N, H // 16, W // 16), torch.ones(1, g_ch_enc_dec, 1, 1)),
        os.path.join(out_dir, "intra_synthesis.onnx"),
        ["in0", "in1"],
    )


def dump_calib_from_qat_onnx(fp32_backbone_dir, frames_dir, frames, qps, crop, out_dir, model_q_scales_dir):
    """Reuse hybrid dump path: need full FP32 pack for Encoder — copy scales from models_fp32."""
    # Build a temporary full-fp32 dir: entropy from models_fp32 + QAT backbone
    tmp = os.path.join(out_dir, "_fp32_pack")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(model_q_scales_dir, tmp)
    for name in BACKBONE_NETS:
        shutil.copy2(os.path.join(fp32_backbone_dir, name + ".onnx"), os.path.join(tmp, name + ".onnx"))
    from ptq_hybrid_backbone_i8 import dump_backbone_calib
    dump_backbone_calib(tmp, frames_dir, frames, qps, crop, out_dir)
    shutil.rmtree(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "cvpr2026_image.pth.tar"))
    ap.add_argument("--frames-dir", default=os.path.join(ONNX_DIR, "rd_frames"))
    ap.add_argument("--frames", nargs="+", default=["beauty", "bosphorus", "jockey"])
    ap.add_argument("--holdout", default="jockey",
                    help="frame held out of QAT training (still used in final RD)")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--qps", nargs="+", type=int, default=[12, 22, 32, 42, 52])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-ckpt", default=os.path.join(ONNX_DIR, "python", "qat_backbone_ckpt.pth"))
    ap.add_argument("--export-dir", default=os.path.join(ONNX_DIR, "models_backbone_qat_fp32"))
    ap.add_argument("--i8-dir", default=os.path.join(ONNX_DIR, "models_backbone_qat_i8"))
    ap.add_argument("--hybrid-dir", default=os.path.join(ONNX_DIR, "models_hybrid_qat_i8bb"))
    ap.add_argument("--baseline-dir", default=os.path.join(ONNX_DIR, "models"))
    ap.add_argument("--fp32-dir", default=os.path.join(ONNX_DIR, "models_fp32"))
    ap.add_argument("--calib-dir", default=os.path.join(REPO, "calib_backbone_qat"))
    ap.add_argument("--rd-out", default=os.path.join(REPO, "rd_hybrid_qat_i8bb"))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    train_frames = [f for f in args.frames if f != args.holdout]
    if not train_frames:
        train_frames = list(args.frames)
        print("warning: holdout empties train set; using all frames")
    imgs = []
    for f in train_frames:
        p = os.path.join(args.frames_dir, f if f.endswith(".png") else f + ".png")
        imgs.append(load_png_rgb(p))
        print(f"train image {f}: {imgs[-1].shape}")
    print(f"holdout for train: {args.holdout}; device={device}")

    if not args.skip_train:
        model = load_model(args.checkpoint, device)
        # Teacher copy without FQ
        teacher = load_model(args.checkpoint, device).eval()
        for p in teacher.parameters():
            p.requires_grad = False

        force_torch_path()
        n = wrap_backbone_convs(model)
        model.to(device)  # move FakeQuant buffers onto device
        print(f"wrapped {n} Conv2d with FakeQuant")
        freeze_entropy(model)
        model.train()

        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr
        )

        t0 = time.time()
        for step in range(1, args.steps + 1):
            xs = []
            qps = []
            for _ in range(args.batch):
                xs.append(random_crop_ycbcr(imgs, args.crop, rng))
                qps.append(int(rng.choice(args.qps)))
            x = torch.cat(xs, dim=0).to(device)
            # per-sample qp: run one-by-one if qps differ (q_scale index)
            opt.zero_grad()
            losses = []
            stats_acc = {"rec": 0.0, "y": 0.0, "z": 0.0}
            for b in range(args.batch):
                xb = x[b : b + 1]
                qp = qps[b]
                y_t, z_t, y_hat_t, x_t = teacher_latents(teacher, xb, qp)
                loss_b, st = student_loss(model, xb, qp, y_t, z_t, y_hat_t, x_t)
                losses.append(loss_b)
                for k in stats_acc:
                    stats_acc[k] += st[k]
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 0.1
            )
            opt.step()
            if step % 50 == 0 or step == 1:
                print(
                    f"[{step:4d}/{args.steps}] loss={float(loss):.5f} "
                    f"rec={stats_acc['rec']/args.batch:.5f} "
                    f"y={stats_acc['y']/args.batch:.5f} "
                    f"z={stats_acc['z']/args.batch:.5f} "
                    f"({time.time()-t0:.0f}s)"
                )

        unwrap_backbone_convs(model)
        os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, args.out_ckpt)
        print(f"saved QAT weights → {args.out_ckpt}")
    else:
        model = load_model(args.checkpoint, device)
        sd = torch.load(args.out_ckpt, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(sd, strict=True)
        model.to(device)
        print(f"loaded QAT weights from {args.out_ckpt}")

    if not args.skip_export:
        force_torch_path()
        print("=== export QAT backbone ONNX ===")
        export_backbone(model, args.export_dir)
        print("=== dump calib + PTQ ===")
        dump_calib_from_qat_onnx(
            args.export_dir, args.frames_dir, args.frames, args.qps, args.crop,
            args.calib_dir, args.fp32_dir,
        )
        quantize_backbone(
            args.export_dir, args.calib_dir, args.i8_dir,
            calibrate_method="percentile", percentile=99.99,
        )
        assemble_hybrid(args.baseline_dir, args.i8_dir, args.hybrid_dir)

    if not args.skip_eval:
        print("=== RD: baseline vs QAT-hybrid ===")
        eval_rd(
            args.baseline_dir, args.hybrid_dir, args.frames_dir, args.frames,
            args.qps, args.crop, args.rd_out,
        )
        # Also print vs PTQ-only hybrid if present
        ptq_json = os.path.join(REPO, "rd_hybrid_i8bb", "rd_results.json")
        qat_json = os.path.join(args.rd_out, "rd_results.json")
        if os.path.isfile(ptq_json) and os.path.isfile(qat_json):
            import json
            ptq = json.load(open(ptq_json))
            qat = json.load(open(qat_json))
            print("\n=== QAT vs PTQ (both vs same baseline) ===")
            print(
                f"  PTQ  mean ratio={np.mean([r['byte_ratio'] for r in ptq]):.4f}  "
                f"dPSNR={np.mean([r['dpsnr'] for r in ptq]):+.4f}"
            )
            print(
                f"  QAT  mean ratio={np.mean([r['byte_ratio'] for r in qat]):.4f}  "
                f"dPSNR={np.mean([r['dpsnr'] for r in qat]):+.4f}"
            )
            # holdout-only
            if args.holdout:
                qh = [r for r in qat if r["frame"] == args.holdout]
                ph = [r for r in ptq if r["frame"] == args.holdout]
                if qh and ph:
                    print(
                        f"  holdout ({args.holdout}) PTQ ratio="
                        f"{np.mean([r['byte_ratio'] for r in ph]):.4f} "
                        f"dPSNR={np.mean([r['dpsnr'] for r in ph]):+.4f}"
                    )
                    print(
                        f"  holdout ({args.holdout}) QAT ratio="
                        f"{np.mean([r['byte_ratio'] for r in qh]):.4f} "
                        f"dPSNR={np.mean([r['dpsnr'] for r in qh]):+.4f}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
