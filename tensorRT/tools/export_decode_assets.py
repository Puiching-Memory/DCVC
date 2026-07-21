#!/usr/bin/env python3
"""Export all assets needed for C I-frame decode: CDF tables, masks, QP scales."""
import os, sys, json, struct
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SUPPRESS_CUSTOM_KERNEL_WARNING", "1")

from src.models.image_model import DMCI
from src.utils.common import get_state_dict

def main():
    out = ROOT / "tensorRT/assets/decode"
    out.mkdir(parents=True, exist_ok=True)
    
    dev = "cuda:0"
    net = DMCI().to(dev).half().eval()
    net.load_state_dict(get_state_dict(str(ROOT / "checkpoints/cvpr2025_image.pth.tar")))
    net.update()

    # 1. Gaussian CDF (128 CDFs for y decode)
    ge = net.gaussian_encoder
    g_cdf, g_len, g_off = ge.get_cdf_info()
    np.save(out / "gaussian_cdf.npy", g_cdf.astype(np.int32))
    np.save(out / "gaussian_cdf_length.npy", g_len.astype(np.int32))
    np.save(out / "gaussian_offset.npy", g_off.astype(np.int32))
    print(f"Gaussian CDF: shape={g_cdf.shape} length_max={g_len.max()}")

    # 2. BitEstimator CDF (z decode) — already in cdf/intra/bit_estimator_z/
    be = net.bit_estimator_z
    z_cdf, z_len, z_off = be.get_cdf_info()
    np.save(out / "bitest_cdf.npy", z_cdf.astype(np.int32))
    np.save(out / "bitest_cdf_length.npy", z_len.astype(np.int32))
    np.save(out / "bitest_offset.npy", z_off.astype(np.int32))
    print(f"BitEstimator CDF: shape={z_cdf.shape} length_max={z_len.max()}")

    # 3. Masks for 4x decode (pre-compute for 256-ch, 16x16 spatial)
    N = 256
    H, W = 16, 16
    masks = net.get_mask_4x(1, N, H, W, torch.float16, dev)
    for i, m in enumerate(masks):
        np.save(out / f"mask_{i}.npy", m.cpu().numpy())
    print(f"Masks: 4 × {tuple(masks[0].shape)}")

    # 4. Golden decode test data: full decode from bitstream
    golden_dir = ROOT / "tensorRT/assets/golden"
    meta = json.loads((golden_dir / "i_256x256_qp20_meta.json").read_text())
    with open(golden_dir / "i_256x256_qp20_bitstream.bin", 'rb') as f:
        bitstream = f.read()
    
    # Do full decode with PyTorch to get intermediate tensors
    qp = meta['qp']
    net.entropy_coder.set_use_two_entropy_coders(meta['ec_part'] == 1)
    net.entropy_coder.set_stream(bitstream)
    z_h, z_w = net.get_downsampled_shape(meta['height'], meta['width'], 64)
    net.bit_estimator_z.decode_z((z_h, z_w), qp)
    z_hat = net.bit_estimator_z.get_z((z_h, z_w), dev, torch.float16)
    np.save(out / "golden_z_hat.npy", z_hat.cpu().numpy())
    
    params = net.hyper_dec(z_hat).detach()
    np.save(out / "golden_params.npy", params.cpu().numpy())
    
    params_full = net.y_prior_fusion(params).detach()
    y_h, y_w = net.get_downsampled_shape(meta['height'], meta['width'], 16)
    params_crop = params_full[:, :, :y_h, :y_w].contiguous().detach()
    np.save(out / "golden_params_fusion.npy", params_crop.cpu().numpy())
    
    y_hat = net.decompress_prior_4x(params_crop,
        net.y_spatial_prior_reduction,
        net.y_spatial_prior_adaptor_1,
        net.y_spatial_prior_adaptor_2,
        net.y_spatial_prior_adaptor_3,
        net.y_spatial_prior)
    np.save(out / "golden_y_hat.npy", y_hat.detach().cpu().numpy())
    
    # Also save intermediate spatial prior outputs for round-by-round verification
    # We need to instrument decompress_prior_4x to capture intermediates
    print(f"Golden z_hat: {tuple(z_hat.shape)}")
    print(f"Golden params: {tuple(params.shape)}")
    print(f"Golden params_fusion: {tuple(params_crop.shape)}")
    print(f"Golden y_hat: {tuple(y_hat.shape)}")
    
    # Save QP scales
    q_dec = net.q_scale_dec[qp:qp+1, :, :, :].half()
    np.save(out / "q_dec.npy", q_dec.detach().cpu().numpy())
    print(f"q_dec: {tuple(q_dec.shape)}")
    
    print(f"\nDecode assets saved to {out}")

if __name__ == "__main__":
    main()
