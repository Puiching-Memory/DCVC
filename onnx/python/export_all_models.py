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

src = types.ModuleType('src')
src.__path__ = [os.path.join(ROOT, 'src')]
sys.modules['src'] = src
models = types.ModuleType('models')
models.__path__ = [os.path.join(ROOT, 'src', 'models')]
sys.modules['src.models'] = models
layers = types.ModuleType('layers')
layers.__path__ = [os.path.join(ROOT, 'src', 'layers')]
sys.modules['src.layers'] = layers
utils = types.ModuleType('utils')
utils.__path__ = [os.path.join(ROOT, 'src', 'utils')]
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


def export_torch(net, args, path, in_names, out_name):
    """Export with dynamic H/W via the dynamo exporter's dynamic_shapes API.

    dynamic_axes is legacy-only under dynamo=True and its derived-dim
    constraints (e.g. pixel_shuffle 16x in intra_synthesis) conflict; Dim
    objects avoid that. Outputs reuse the same Dim symbols, so the exporter
    derives the output size symbolically from the input.
    """
    h = torch.export.Dim('h', min=4)
    w = torch.export.Dim('w', min=4)
    # Only dims 2/3 of the FIRST 4D input are symbolic; trailing q-vectors
    # (1x1 broadcast) are left fully static, otherwise torch.export derives a
    # bogus constraint q.size(2)==x.size(2) from the broadcast mul. min=4 keeps
    # the Dim symbolic even for the tiny z-plane (4x4 at 256x256).
    if not isinstance(args, (tuple, list)):
        args = (args,)
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
    args = parser.parse_args()

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
    H, W = 256, 256
    export_torch(model.enc, (torch.randn(1, 3, H, W), torch.ones(1, g_ch_enc_dec, 1, 1)),
                 os.path.join(out_dir, 'intra_analysis_standard.onnx'), ['in0', 'in1'], 'out0')
    export_torch(model.hyper_enc, torch.randn(1, N, H // 16, W // 16),
                 os.path.join(out_dir, 'intra_hyper_enc.onnx'), ['in0'], 'out0')
    export_torch(model.hyper_dec, torch.randn(1, ZC, H // 64, W // 64),
                 os.path.join(out_dir, 'hyper_dec.onnx'), ['in0'], 'out0')
    export_torch(model.y_prior_fusion, torch.randn(1, N, H // 16, W // 16),
                 os.path.join(out_dir, 'y_prior_fusion.onnx'), ['in0'], 'out0')
    export_torch(model.y_spatial_prior_reduction, torch.randn(1, 2 * N + 2, H // 16, W // 16),
                 os.path.join(out_dir, 'y_spatial_prior_reduction.onnx'), ['in0'], 'out0')
    for i in [1, 2, 3]:
        export_torch(getattr(model, f'y_spatial_prior_adaptor_{i}'), torch.randn(1, 2 * N, H // 16, W // 16),
                     os.path.join(out_dir, f'y_spatial_prior_adaptor_{i}.onnx'), ['in0'], 'out0')
    export_torch(model.y_spatial_prior, torch.randn(1, 2 * N, H // 16, W // 16),
                 os.path.join(out_dir, 'y_spatial_prior.onnx'), ['in0'], 'out0')
    export_torch(model.dec, (torch.randn(1, N, H // 16, W // 16), torch.ones(1, g_ch_enc_dec, 1, 1)),
                 os.path.join(out_dir, 'intra_synthesis.onnx'), ['in0', 'in1'], 'out0')

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
