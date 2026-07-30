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

# The top-level src/ is the DCVC-UF (CVPR-2026, 384-ch) code and is the
# default. --src-root selects which tree provides the src/ package for this
# export. It is pre-parsed here because the src.* imports below happen at
# module load time.
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

from src.models.image_model import DMCI, g_ch_enc_dec, g_ch_y, g_ch_z
from src.utils.common import get_state_dict
import torch
import numpy as np


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


def _remap_cdf_assets(out_dir, cdf_src):
    """Remap the gaussian / bit-estimator CDFs from DCVC-RT offset-centred
    order to DCVC-UF zigzag order (value = |sym|*2-(sym>0), symbol 0 at CDF
    index 0). Writes zigzag-ordered *_cdf.npy / *_cdf_length.npy and drops the
    *_offset.npy files (zigzag has no offset; the C runtime ignores offset)."""
    import numpy as np
    SCALE = 1 << 16  # CDF precision used by the rANS codec
    pairs = [('gaussian', 'gaussian_offset'), ('bitest', 'bitest_offset')]
    for stem, off_stem in pairs:
        cdf_path = os.path.join(cdf_src, stem + '_cdf.npy')
        len_path = os.path.join(cdf_src, stem + '_cdf_length.npy')
        if not os.path.exists(cdf_path):
            print('warning: missing', cdf_path); continue
        cdf = np.load(cdf_path)        # [qp_num, max_len]
        clen = np.load(len_path)       # [qp_num]
        qp_num, max_len = cdf.shape
        zig_cdf = np.zeros_like(cdf)
        for q in range(qp_num):
            L = int(clen[q])           # number of CDF entries (incl. escape)
            max_value = L - 2          # escape triggers when value >= max_value
            # RT offset order: CDF entry i is symbol (offset? no) — RT builds the
            # gaussian CDF symmetric around its centre. The escape (largest
            # magnitude) sits at the LAST real interval. We rebuild the zigzag
            # pmf by walking RT symbols [-sym_range .. +sym_range] (+escape).
            sym_range = (L - 3) // 2    # RT stores 2*sym_range+1 symbols + escape + cdf terminal
            # pmf of each CDF interval (length L-1 intervals over the L entries)
            pmf = np.diff(cdf[q, :L]).astype(np.int64)
            # RT interval i -> symbol = i - sym_range; the final interval is escape
            zig = np.zeros(L, dtype=np.int64)
            for i in range(len(pmf)):
                s = i - sym_range
                if i == len(pmf) - 1:
                    # escape interval: belongs at max_value (the UF escape slot)
                    z = max_value
                else:
                    z = abs(s) * 2 - (1 if s > 0 else 0)
                    if z >= max_value:
                        z = max_value
                zig[z] += pmf[i]
            zig_cdf[q, 0] = 0
            for i in range(1, L):
                zig_cdf[q, i] = zig_cdf[q, i - 1] + zig[i - 1]
            zig_cdf[q, L:] = SCALE      # pad tail to full precision
        np.save(os.path.join(out_dir, stem + '_cdf.npy'), zig_cdf)
        np.save(os.path.join(out_dir, stem + '_cdf_length.npy'), clen)
        # write a zero offset array for backward compat (the C runtime still
        # accepts an offset argument but ignores it under zigzag)
        np.save(os.path.join(out_dir, stem + '_offset.npy'),
                np.zeros(qp_num, dtype=cdf.dtype))
        print('remapped', stem, 'CDF -> zigzag order')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default=os.path.join(ROOT, 'onnx', 'models'))
    parser.add_argument('--checkpoint', default=os.path.join(ROOT, 'checkpoints', 'cvpr2026_image.pth.tar'))
    parser.add_argument('--src-root', default=SRC_ROOT,
                        help='tree providing the src/ package to export from '
                             '(default: repo top-level = DCVC-UF)')
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
    model.load_state_dict(get_state_dict(args.checkpoint))
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
    export_torch(model.hyper_enc, torch.randn(1, g_ch_y, H // 16, W // 16),
                 os.path.join(out_dir, 'intra_hyper_enc.onnx'), ['in0'], 'out0',
                 fixed_size=fixed_size)
    export_torch(model.hyper_dec, torch.randn(1, g_ch_z, H // 64, W // 64),
                 os.path.join(out_dir, 'hyper_dec.onnx'), ['in0'], 'out0',
                 fixed_size=fixed_size)
    export_torch(model.y_prior_fusion, torch.randn(1, g_ch_y, H // 16, W // 16),
                 os.path.join(out_dir, 'y_prior_fusion.onnx'), ['in0'], 'out0',
                 fixed_size=fixed_size)
    export_torch(model.y_spatial_prior_reduction, torch.randn(1, g_ch_y * 2, H // 16, W // 16),
                 os.path.join(out_dir, 'y_spatial_prior_reduction.onnx'), ['in0'], 'out0',
                 fixed_size=fixed_size)
    for i in [1, 2, 3]:
        export_torch(getattr(model, f'y_spatial_prior_adaptor_{i}'), torch.randn(1, g_ch_y * 2, H // 16, W // 16),
                     os.path.join(out_dir, f'y_spatial_prior_adaptor_{i}.onnx'), ['in0'], 'out0',
                     fixed_size=fixed_size)
    export_torch(model.y_spatial_prior, torch.randn(1, g_ch_y * 2, H // 16, W // 16),
                 os.path.join(out_dir, 'y_spatial_prior.onnx'), ['in0'], 'out0',
                 fixed_size=fixed_size)
    export_torch(model.dec, (torch.randn(1, g_ch_y, H // 16, W // 16), torch.ones(1, g_ch_enc_dec, 1, 1)),
                 os.path.join(out_dir, 'intra_synthesis.onnx'), ['in0', 'in1'], 'out0',
                 fixed_size=fixed_size)

    # QP scales (encoder/synthesis use g_ch_enc_dec channels; the 4x prior
    # quantization steps for y use the per-channel q_scale_y_enc/dec banks).
    np.save(os.path.join(out_dir, 'q_scale_enc.npy'), model.q_scale_enc.detach().cpu().numpy())
    np.save(os.path.join(out_dir, 'q_scale_dec.npy'), model.q_scale_dec.detach().cpu().numpy())
    np.save(os.path.join(out_dir, 'q_scale_y_enc.npy'), model.q_scale_y_enc.detach().cpu().numpy())
    np.save(os.path.join(out_dir, 'q_scale_y_dec.npy'), model.q_scale_y_dec.detach().cpu().numpy())
    print('saved q_scale_enc/dec + q_scale_y_enc/dec .npy')

    # CDF tables in DCVC-UF zigzag order. The legacy RT assets store the CDFs in
    # offset-centred order (peak in the middle, offset = -sym_range); the UF
    # rANS uses value = |sym|*2-(sym>0), so symbol 0 must sit at CDF index 0.
    # We remap the per-QP gaussian / bit-estimator CDFs from offset order to
    # zigzag order here and drop the *_offset arrays (zigzag has no offset).
    _remap_cdf_assets(out_dir, os.path.join(ROOT, 'tensorRT', 'assets', 'decode'))

    print('All models and data exported to', out_dir)


if __name__ == '__main__':
    main()
