#!/usr/bin/env python3
"""Export all ONNX models and auxiliary data needed for the pure-CPU pipeline.

All 9 intra nets (including intra_analysis_standard) are exported directly from
DMCI submodules via torch dynamo. DepthConvBlock uses its pure-PyTorch path on
CPU, so no custom-op ONNX or hand conversion is required.
"""
import argparse
import os
import shutil
import sys
import types

# Silence the "customized cuda kernel is not used" warning and pick the torch path.
os.environ.setdefault("SUPPRESS_CUSTOM_KERNEL_WARNING", "1")

# Make src/ importable
# Repo root (DCVC/). This script lives in <root>/onnx/python/.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The top-level src/ is the CVPR-2026 (384-ch) code; the shipped models were
# built from DCVC-family/DCVC-RT (368-ch, cvpr2025 checkpoints). --src-root
# selects which tree provides the src/ package for this export. It is
# pre-parsed here because the src.* imports below happen at module load time.
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

from src.models.image_model import DMCI, g_ch_enc_dec
import torch
import numpy as np

N = 256
ZC = 128

# Latest opset officially supported by the deployed ONNX Runtime 1.27
# (opset 27 is still "under development" and rejected by ORT at load time).
# The IR version is whatever the torch dynamo exporter / the source model
# carries natively; we do not re-stamp it.
OPSET_VERSION = 26


def export_torch(net, args, path, in_names, out_name, fixed_size=False):
    """Export with dynamic H/W via the dynamo exporter's dynamic_shapes API.

    dynamic_axes is legacy-only under dynamo=True and its derived-dim
    constraints (e.g. pixel_shuffle 16x in intra_synthesis) conflict; Dim
    objects avoid that. Outputs reuse the same Dim symbols, so the exporter
    derives the output size symbolically from the input.

    With fixed_size=True no dynamic shapes are declared: all dims are baked
    to the example-input sizes (fixed-resolution model pack).
    """
    h = torch.export.Dim('h', min=4)
    w = torch.export.Dim('w', min=4)
    # Only dims 2/3 of the FIRST 4D input are symbolic; trailing q-vectors
    # (1x1 broadcast) are left fully static, otherwise torch.export derives a
    # bogus constraint q.size(2)==x.size(2) from the broadcast mul. min=4 keeps
    # the Dim symbolic even for the tiny z-plane (4x4 at 256x256).
    if not isinstance(args, (tuple, list)):
        args = (args,)
    if fixed_size:
        dynamic_shapes = None
    else:
        dynamic_shapes = tuple({2: h, 3: w} if i == 0 else None
                               for i, a in enumerate(args) if a.dim() == 4)
    torch.onnx.export(
        net, args, path,
        input_names=in_names, output_names=[out_name],
        dynamic_shapes=dynamic_shapes, opset_version=OPSET_VERSION, dynamo=True,
        external_data=False,  # keep each model a single self-contained file
    )
    print('exported', path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default=os.path.join(ROOT, 'onnx', 'models'))
    parser.add_argument('--checkpoint', default=os.path.join(ROOT, 'checkpoints', 'cvpr2025_image.pth.tar'))
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

    fixed_size = args.height is not None or args.width is not None

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load checkpoint
    model = DMCI()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    if all(k.startswith('module.') for k in sd.keys()):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    # All 9 intra nets: dynamo export from DMCI submodules (DepthConvBlock torch path).
    # intra_analysis = IntraEncoder(x, quant_step): pixel_unshuffle(8) + DepthConv stack.
    # Fixed-size mode exports at the PADDED network input size (multiple of 64),
    # matching what the C pipeline feeds the nets for that picture size.
    if fixed_size:
        if args.height is None or args.width is None:
            parser.error('--height and --width must be given together')
        H = (args.height + 63) // 64 * 64
        W = (args.width + 63) // 64 * 64
        print(f'fixed-size export: picture {args.width}x{args.height} -> network input {W}x{H}')
    else:
        H, W = 256, 256
    export_torch(model.enc, (torch.randn(1, 3, H, W), torch.ones(1, g_ch_enc_dec, 1, 1)),
                 os.path.join(out_dir, 'intra_analysis_standard.onnx'), ['in0', 'in1'], 'out0',
                 fixed_size=fixed_size)
    export_torch(model.hyper_enc, torch.randn(1, N, H // 16, W // 16),
                 os.path.join(out_dir, 'intra_hyper_enc.onnx'), ['in0'], 'out0',
                 fixed_size=fixed_size)
    export_torch(model.hyper_dec, torch.randn(1, ZC, H // 64, W // 64),
                 os.path.join(out_dir, 'hyper_dec.onnx'), ['in0'], 'out0',
                 fixed_size=fixed_size)
    export_torch(model.y_prior_fusion, torch.randn(1, N, H // 16, W // 16),
                 os.path.join(out_dir, 'y_prior_fusion.onnx'), ['in0'], 'out0',
                 fixed_size=fixed_size)
    export_torch(model.y_spatial_prior_reduction, torch.randn(1, 2 * N + 2, H // 16, W // 16),
                 os.path.join(out_dir, 'y_spatial_prior_reduction.onnx'), ['in0'], 'out0',
                 fixed_size=fixed_size)
    for i in [1, 2, 3]:
        export_torch(getattr(model, f'y_spatial_prior_adaptor_{i}'), torch.randn(1, 2 * N, H // 16, W // 16),
                     os.path.join(out_dir, f'y_spatial_prior_adaptor_{i}.onnx'), ['in0'], 'out0',
                     fixed_size=fixed_size)
    export_torch(model.y_spatial_prior, torch.randn(1, 2 * N, H // 16, W // 16),
                 os.path.join(out_dir, 'y_spatial_prior.onnx'), ['in0'], 'out0',
                 fixed_size=fixed_size)
    export_torch(model.dec, (torch.randn(1, N, H // 16, W // 16), torch.ones(1, g_ch_enc_dec, 1, 1)),
                 os.path.join(out_dir, 'intra_synthesis.onnx'), ['in0', 'in1'], 'out0',
                 fixed_size=fixed_size)

    # QP scales
    np.save(os.path.join(out_dir, 'q_scale_enc.npy'), model.q_scale_enc.detach().cpu().numpy())
    np.save(os.path.join(out_dir, 'q_scale_dec.npy'), model.q_scale_dec.detach().cpu().numpy())
    print('saved q_scale_enc.npy / q_scale_dec.npy')

    # CDF tables from tensorRT assets
    cdf_src = os.path.join(ROOT, 'tensorRT', 'assets', 'decode')
    for name in ['gaussian_cdf.npy', 'gaussian_cdf_length.npy', 'gaussian_offset.npy',
                 'bitest_cdf.npy', 'bitest_cdf_length.npy', 'bitest_offset.npy']:
        src_path = os.path.join(cdf_src, name)
        dst_path = os.path.join(out_dir, name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print('copied', name)
        else:
            print('warning: missing', src_path)

    print('All models and data exported to', out_dir)


if __name__ == '__main__':
    main()
