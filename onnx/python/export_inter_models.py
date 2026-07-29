#!/usr/bin/env python3
"""Export DCVC P-frame (inter) subnets to ONNX and verify each against ONNX Runtime.

Produces 11 inter_*/recon_generation ONNX models + 4 q-bank .npy files in
<root>/onnx/models/, mirroring the TensorRT layout in
<root>/tensorRT/tools/build_inter_engines.py but in FP32 for the CPU ONNX pipeline.

Each exported model is verified against ONNX Runtime on identical random FP32
inputs (torch.manual_seed(42)): features/latents use torch.randn, frames use
torch.rand, and q-vectors use torch.ones. Max abs diff must be < 1e-4.
"""
import argparse
import os
import sys
import types

# Silence the "customized cuda kernel is not used" warning and pick the torch path.
os.environ.setdefault("SUPPRESS_CUSTOM_KERNEL_WARNING", "1")

# Make src/ importable (same trick as export_all_models.py).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The top-level src/ is the CVPR-2026 code without video_model.py; the shipped
# inter models come from DCVC-family/DCVC-RT. --src-root selects which tree
# provides the src/ package. Pre-parsed here because the src.* imports below
# happen at module load time.
SRC_ROOT = ROOT
for _i, _a in enumerate(sys.argv):
    if _a == '--src-root' and _i + 1 < len(sys.argv):
        SRC_ROOT = os.path.abspath(sys.argv[_i + 1])
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

src = types.ModuleType('src')
src.__path__ = [os.path.join(SRC_ROOT, 'src')]
sys.modules['src'] = src
models = types.ModuleType('models')
models.__path__ = [os.path.join(SRC_ROOT, 'src', 'models')]
sys.modules['src.models'] = models
layers = types.ModuleType('layers')
layers.__path__ = [os.path.join(SRC_ROOT, 'src', 'layers')]
sys.modules['src.layers'] = layers
utils = types.ModuleType('utils')
utils.__path__ = [os.path.join(SRC_ROOT, 'src', 'utils')]
sys.modules['src.utils'] = utils

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from src.models.video_model import DMC, g_ch_src_d, g_ch_recon, g_ch_y, g_ch_z, g_ch_d
from src.utils.common import get_state_dict

# Spatial size constants at the standard H=W=256 export resolution.
H = W = 256
fH, fW = H // 8, W // 8    # feature / context resolution
yH, yW = H // 16, W // 16  # y latent resolution
zH, zW = H // 64, W // 64  # z (hyper) latent resolution

THRESH = 1e-4

# Latest opset officially supported by the deployed ONNX Runtime 1.27
# (opset 27 is still "under development" and rejected by ORT at load time).
# The IR version is whatever the torch dynamo exporter emits natively.
OPSET_VERSION = 26

# Axes map with DISTINCT symbolic dim names per tensor kind: the dynamo
# exporter bakes the value_info of every intermediate with the symbolic names
# we give, and reusing the same name ('h'/'w') for tensors that differ in
# resolution (feature-plane vs y-plane) makes ORT's shape inference collapse
# them and fail buffer reuse checks. Keeping the names distinct matches what
# the old dynamo=False exporter inferred.
AXES_FEATURE = {2: 'h', 3: 'w'}          # H/8  plane
AXES_LATENT = {2: 'h_latent', 3: 'w_latent'}  # H/16 plane


def axes_for(name, spatial_names, feature_names):
    if name not in spatial_names:
        return None
    return AXES_FEATURE if name in feature_names else AXES_LATENT


def export_module(path, module, args, input_names, output_names, spatial_names,
                  feature_names, fixed_size=False):
    """Export one wrapper/module to ONNX with dynamic spatial axes (dims 2,3).

    Only names in `spatial_names` (the feature/latent tensors) get dynamic
    axes; the q-vectors (1x1) stay fixed. `feature_names` selects which of
    the spatial tensors live on the H/8 feature plane (the rest are on the
    H/16 y plane).

    Uses the dynamo exporter at the latest opset supported by ORT 1.27.
    With fixed_size=True no dynamic axes are declared: all dims are baked
    to the example-input sizes (fixed-resolution model pack).
    """
    dynamic_axes = None
    if not fixed_size:
        dynamic_axes = {}
        for n in input_names + output_names:
            ax = axes_for(n, spatial_names, feature_names)
            if ax is not None:
                dynamic_axes[n] = ax
    torch.onnx.export(
        module, args, path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=OPSET_VERSION, dynamo=True,
        external_data=False,  # keep each model a single self-contained file
    )
    print('exported', os.path.relpath(path, ROOT))


# ---------------------------------------------------------------------------
# Small wrappers so the exported ONNX graphs have clean named I/O and exclude
# the ops that the C++ runtime performs outside (pixel_un/shuffle, clamp).
# ---------------------------------------------------------------------------
class FEWrap(nn.Module):
    """FeatureExtractor: returns BOTH ctx and ctx_t (part1 then part2)."""
    def __init__(self, fe):
        super().__init__()
        self.fe = fe

    def forward(self, feature, q_feature):
        x1, ctx_t = self.fe.forward_part1(feature, q_feature)
        ctx = self.fe.forward_part2(x1)
        return ctx, ctx_t


class EncWrap(nn.Module):
    """Encoder on already pixel-unshuffled input (pixel_unshuffle done in C++)."""
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, x_unshuf, ctx, q_enc):
        return self.enc.forward_torch(x_unshuf, ctx, q_enc)


class DecWrap(nn.Module):
    """Decoder using the torch path."""
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, y_hat, ctx, q_dec):
        return self.dec.forward_torch(y_hat, ctx, q_dec)


class RGWrap(nn.Module):
    """ReconGeneration: conv -> *q_recon -> head (NO pixel_shuffle / clamp)."""
    def __init__(self, rg):
        super().__init__()
        self.conv = rg.conv
        self.head = rg.head

    def forward(self, feature, q_recon):
        out = self.conv(feature)
        out = out * q_recon
        return self.head(out)


# ---------------------------------------------------------------------------
# Export helper.
# ---------------------------------------------------------------------------
# Model specifications: name, wrapper, input specs, output names, spatial names.
# Each input spec is (name, (C, h, w), kind) where kind in {feat, frame, q}.
# ---------------------------------------------------------------------------
def build_specs(net):
    """Return list of dicts describing each of the 11 models to export."""
    specs = []

    # 1. feature_adaptor_i: DepthConvBlock(192 -> 256)
    specs.append(dict(
        name="inter_feature_adaptor_i", module=net.feature_adaptor_i,
        inputs=[("in0", (g_ch_src_d, fH, fW), "frame")],
        outputs=["out0"], spatial=["in0", "out0"], feature=["in0", "out0"]))

    # 2. feature_adaptor_p: Conv2d(256 -> 256)
    specs.append(dict(
        name="inter_feature_adaptor_p", module=net.feature_adaptor_p,
        inputs=[("in0", (g_ch_d, fH, fW), "feat")],
        outputs=["out0"], spatial=["in0", "out0"], feature=["in0", "out0"]))

    # 3. feature_extractor: returns ctx, ctx_t (TWO outputs)
    specs.append(dict(
        name="inter_feature_extractor", module=FEWrap(net.feature_extractor),
        inputs=[("in0", (g_ch_d, fH, fW), "feat"), ("q_feat", (g_ch_d, 1, 1), "q")],
        outputs=["ctx", "ctx_t"],
        spatial=["in0", "ctx", "ctx_t"], feature=["in0", "ctx", "ctx_t"]))

    # 4. inter_encoder: already-unshuffled x + ctx + q_enc -> y latent
    specs.append(dict(
        name="inter_encoder", module=EncWrap(net.encoder),
        inputs=[("in0", (g_ch_src_d, fH, fW), "frame"),
                ("ctx", (g_ch_d, fH, fW), "feat"),
                ("q_enc", (g_ch_d, 1, 1), "q")],
        outputs=["out0"], spatial=["in0", "ctx", "out0"],
        feature=["in0", "ctx"]))

    # 5. hyper_encoder: y -> z
    specs.append(dict(
        name="inter_hyper_enc", module=net.hyper_encoder,
        inputs=[("in0", (g_ch_y, yH, yW), "feat")],
        outputs=["out0"], spatial=["in0", "out0"], feature=[]))

    # 6. hyper_decoder: z -> y
    specs.append(dict(
        name="inter_hyper_dec", module=net.hyper_decoder,
        inputs=[("in0", (g_ch_z, zH, zW), "feat")],
        outputs=["out0"], spatial=["in0", "out0"], feature=[]))

    # 7. temporal_prior_encoder: ResidualBlockWithStride2(256 -> 256) fH -> yH
    specs.append(dict(
        name="inter_temporal_prior", module=net.temporal_prior_encoder,
        inputs=[("in0", (g_ch_d, fH, fW), "feat")],
        outputs=["out0"], spatial=["in0", "out0"], feature=["in0"]))

    # 8. y_prior_fusion: cat(hier[128], temporal[256]) = 384 -> 384
    specs.append(dict(
        name="inter_prior_fusion", module=net.y_prior_fusion,
        inputs=[("in0", (g_ch_y * 3, yH, yW), "feat")],
        outputs=["out0"], spatial=["in0", "out0"], feature=[]))

    # 9. y_spatial_prior: cat(y_hat[128], params[384]) = 512 -> 256
    specs.append(dict(
        name="inter_spatial_prior", module=net.y_spatial_prior,
        inputs=[("in0", (g_ch_y * 4, yH, yW), "feat")],
        outputs=["out0"], spatial=["in0", "out0"], feature=[]))

    # 10. inter_decoder: y_hat + ctx + q_dec -> feature (fH)
    specs.append(dict(
        name="inter_decoder", module=DecWrap(net.decoder),
        inputs=[("in0", (g_ch_y, yH, yW), "feat"),
                ("ctx", (g_ch_d, fH, fW), "feat"),
                ("q_dec", (g_ch_d, 1, 1), "q")],
        outputs=["out0"], spatial=["in0", "ctx", "out0"],
        feature=["ctx", "out0"]))

    # 11. recon_generation: feature + q_recon -> src_d (NO pixel_shuffle/clamp)
    specs.append(dict(
        name="recon_generation", module=RGWrap(net.recon_generation_net),
        inputs=[("in0", (g_ch_d, fH, fW), "feat"),
                ("q_recon", (g_ch_recon, 1, 1), "q")],
        outputs=["out0"], spatial=["in0", "out0"], feature=["in0", "out0"]))

    return specs


def make_input(kind, shape):
    """Build a random FP32 numpy input of the given kind."""
    t = torch.empty((1,) + shape, dtype=torch.float32)
    if kind == "q":
        t.fill_(1.0)
    elif kind == "frame":
        t.uniform_(0.0, 1.0)
    else:  # feat / latent
        t.normal_(0.0, 1.0)
    return t.numpy()


def verify_model(path, module, spec):
    """Run the torch wrapper and ORT on identical inputs; return max abs diff."""
    torch.manual_seed(42)
    feed = {}
    torch_inputs = []
    for (iname, shape, kind) in spec["inputs"]:
        arr = make_input(kind, shape)
        feed[iname] = arr
        torch_inputs.append(torch.from_numpy(arr))

    with torch.no_grad():
        torch_out = module(*torch_inputs)
    if isinstance(torch_out, (tuple, list)):
        torch_out = [o.detach().cpu().numpy() for o in torch_out]
    else:
        torch_out = [torch_out.detach().cpu().numpy()]

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, feed)

    out_names = spec["outputs"]
    max_diff = 0.0
    for name, to, oo in zip(out_names, torch_out, ort_out):
        d = float(np.max(np.abs(to - oo)))
        max_diff = max(max_diff, d)
    return max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default=os.path.join(ROOT, 'onnx', 'models'))
    parser.add_argument('--checkpoint', default=os.path.join(ROOT, 'checkpoints', 'cvpr2025_video.pth.tar'))
    parser.add_argument('--src-root', default=SRC_ROOT,
                        help='tree providing the src/ package to export from '
                             '(default: repo top-level; use DCVC-family/DCVC-RT '
                             'for the shipped cvpr2025 models)')
    parser.add_argument('--height', type=int, default=None,
                        help='fixed picture height (padded up to a multiple of 64); '
                             'omit for dynamic H/W')
    parser.add_argument('--width', type=int, default=None,
                        help='fixed picture width (padded up to a multiple of 64); '
                             'omit for dynamic H/W')
    args = parser.parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    fixed_size = args.height is not None or args.width is not None
    if fixed_size:
        # Export at the PADDED network input size (multiple of 64), matching
        # what the C pipeline feeds the nets for that picture size.
        if args.height is None or args.width is None:
            parser.error('--height and --width must be given together')
        global H, W, fH, fW, yH, yW, zH, zW
        H = (args.height + 63) // 64 * 64
        W = (args.width + 63) // 64 * 64
        fH, fW = H // 8, W // 8
        yH, yW = H // 16, W // 16
        zH, zW = H // 64, W // 64
        print(f'fixed-size export: picture {args.width}x{args.height} -> network input {W}x{H}')

    # Load model in FP32 on CPU (DepthConvBlock uses the torch path).
    net = DMC()
    net.load_state_dict(get_state_dict(args.checkpoint))
    net.eval()
    net.update()

    qp_num = net.get_qp_num()
    print(f"=== Inter export (H={H} W={W}, fH={fH}, yH={yH}, zH={zH}, qp_num={qp_num}) ===")
    print(f"  g_ch_src_d={g_ch_src_d} g_ch_recon={g_ch_recon} g_ch_y={g_ch_y} "
          f"g_ch_z={g_ch_z} g_ch_d={g_ch_d}")

    specs = build_specs(net)

    # ---- Export all 11 models ----
    for spec in specs:
        path = os.path.join(out_dir, spec["name"] + ".onnx")
        with torch.no_grad():
            # Example inputs for tracing (shapes only; content irrelevant).
            example = tuple(torch.randn(1, *shape, dtype=torch.float32)
                            for (_, shape, _) in spec["inputs"])
        export_module(path, spec["module"], example,
                      [n for (n, _, _) in spec["inputs"]],
                      spec["outputs"], spec["spatial"], spec["feature"],
                      fixed_size=fixed_size)

    # ---- Export 4 q-bank .npy files ----
    qbanks = [
        ("q_encoder", net.q_encoder),
        ("q_decoder", net.q_decoder),
        ("q_feature", net.q_feature),
        ("q_recon", net.q_recon),
    ]
    for name, param in qbanks:
        arr = param.detach().cpu().numpy()
        np.save(os.path.join(out_dir, name + ".npy"), arr)
        print(f"saved {name}.npy shape={arr.shape}")

    # ---- Verify all 11 models ----
    print("\n=== Verification (torch vs ONNX Runtime, FP32) ===")
    results = []
    all_ok = True
    for spec in specs:
        path = os.path.join(out_dir, spec["name"] + ".onnx")
        diff = verify_model(path, spec["module"], spec)
        ok = diff < THRESH
        all_ok = all_ok and ok
        results.append((spec["name"], diff, ok))
        print(f"  {spec['name']:30s} max_abs_diff={diff:.3e}  "
              f"{'PASS' if ok else 'FAIL'}")

    # ---- Final summary ----
    print("\n=== Final summary ===")
    print(f"qp_num = {qp_num}  (q-bank rows = qp_num + extra_qp = {qp_num + max([0,8,4])})")
    print(f"{'file':36s} {'size':>12s}")
    saved = []
    for spec in specs:
        p = os.path.join(out_dir, spec["name"] + ".onnx")
        saved.append((os.path.basename(p), os.path.getsize(p)))
    for name, param in qbanks:
        p = os.path.join(out_dir, name + ".npy")
        saved.append((os.path.basename(p), os.path.getsize(p)))
    for fn, sz in saved:
        print(f"  {fn:36s} {sz:>12d}")
    print("\nPer-model max abs diff (threshold %.0e):" % THRESH)
    for name, diff, ok in results:
        print(f"  {name:30s} {diff:.3e}  {'PASS' if ok else 'FAIL'}")

    if not all_ok:
        print("\n!!! One or more models FAILED parity verification. !!!")
        return 1
    print("\nAll 11 inter models exported and verified within %.0e." % THRESH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
