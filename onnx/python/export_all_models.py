#!/usr/bin/env python3
"""Export all ONNX models and auxiliary data needed for the pure-CPU pipeline."""
import argparse
import os
import shutil
import sys
import types

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

from src.models.image_model import DMCI
from src.layers.cuda_inference import round_and_to_int8
import torch
import numpy as np

from convert_to_standard_ops import convert_dcvc_depthconv_to_standard_ops
from make_intra_analysis_dynamic import patch_model as make_intra_analysis_dynamic

N = 256
ZC = 128


def export_torch(net, args, path, in_names, out_name):
    dynamic = {n: {0: 'batch', 2: 'h', 3: 'w'} for n in in_names}
    dynamic[out_name] = {0: 'batch', 2: 'h', 3: 'w'}
    torch.onnx.export(
        net, args, path,
        input_names=in_names, output_names=[out_name],
        dynamic_axes=dynamic, opset_version=17, dynamo=False
    )
    print('exported', path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default=os.path.join(ROOT, 'onnx', 'models'))
    parser.add_argument('--checkpoint', default=os.path.join(ROOT, 'checkpoints', 'cvpr2025_image.pth.tar'))
    parser.add_argument('--original-intra-analysis', default=os.path.join(ROOT, 'native', 'assets', 'onnx', 'intra_analysis.onnx'))
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

    # 1. intra_analysis_standard.onnx (from original custom-op ONNX + FP32 weights)
    standard_path = os.path.join(out_dir, 'intra_analysis_standard.onnx')
    convert_dcvc_depthconv_to_standard_ops(args.original_intra_analysis, standard_path, args.checkpoint)
    # intra_analysis_standard.onnx ships as a dynamic-resolution model (any H,W
    # multiple of 8): rewrite the static pixel_unshuffle Reshape constants
    # so input 'in0' is [1,3,'h','w']. Conv weights are untouched.
    make_intra_analysis_dynamic(standard_path, standard_path)
    print('made dynamic:', standard_path)

    # 2. Other standard-op models, exported from PyTorch on CPU (DepthConvBlock uses torch path)
    H, W = 256, 256
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
    export_torch(model.dec, (torch.randn(1, N, H // 16, W // 16), torch.ones(1, 368, 1, 1)),
                 os.path.join(out_dir, 'intra_synthesis.onnx'), ['in0', 'in1'], 'out0')

    # 3. QP scales
    np.save(os.path.join(out_dir, 'q_scale_enc.npy'), model.q_scale_enc.detach().cpu().numpy())
    np.save(os.path.join(out_dir, 'q_scale_dec.npy'), model.q_scale_dec.detach().cpu().numpy())
    print('saved q_scale_enc.npy / q_scale_dec.npy')

    # 4. CDF tables from native assets
    cdf_src = os.path.join(ROOT, 'native', 'assets', 'decode')
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
