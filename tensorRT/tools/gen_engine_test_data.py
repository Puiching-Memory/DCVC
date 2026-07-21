#!/usr/bin/env python3
"""Generate test tensors for C-level engine parity test.
Saves FP16 input + reference output for each I-frame engine."""
import os, sys, json
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SUPPRESS_CUSTOM_KERNEL_WARNING", "1")
from src.models.image_model import DMCI, g_ch_enc_dec
from src.utils.common import get_state_dict

dev = "cuda:0"
net = DMCI().to(dev).half().eval()
net.load_state_dict(get_state_dict(str(ROOT / "checkpoints/cvpr2025_image.pth.tar")))

out_dir = ROOT / "tensorRT/assets/test_tensors"
out_dir.mkdir(parents=True, exist_ok=True)

g = torch.Generator(dev).manual_seed(7)
H, W, N = 256, 256, 256

# intra_analysis: IntraEncoder(x, quant_step) -> y
x = (torch.rand((1, 3, H, W), device=dev, dtype=torch.float16, generator=g) * 0.8 + 0.1)
q_enc = torch.ones((1, g_ch_enc_dec, 1, 1), device=dev, dtype=torch.float16)
with torch.no_grad():
    y_ref = net.enc(x, q_enc)
np.save(out_dir / "analysis_input.npy", x.cpu().numpy())
np.save(out_dir / "analysis_quant.npy", q_enc.cpu().numpy())
np.save(out_dir / "analysis_ref.npy", y_ref.cpu().numpy())
print(f"intra_analysis: input {tuple(x.shape)} -> output {tuple(y_ref.shape)}")

# intra_hyper_enc: hyper_enc(y_pad) -> z
y_pad = torch.randn((1, N, H//16, W//16), device=dev, dtype=torch.float16, generator=g)
with torch.no_grad():
    z_ref = net.hyper_enc(y_pad)
np.save(out_dir / "hyper_input.npy", y_pad.cpu().numpy())
np.save(out_dir / "hyper_ref.npy", z_ref.cpu().numpy())
print(f"intra_hyper_enc: input {tuple(y_pad.shape)} -> output {tuple(z_ref.shape)}")

# intra_synthesis: IntraDecoder(y_hat, quant_step) -> x_hat
y_hat = torch.randn((1, N, H//16, W//16), device=dev, dtype=torch.float16, generator=g)
q_dec = torch.ones((1, g_ch_enc_dec, 1, 1), device=dev, dtype=torch.float16)
with torch.no_grad():
    xhat_ref = net.dec(y_hat, q_dec)
np.save(out_dir / "synthesis_input.npy", y_hat.cpu().numpy())
np.save(out_dir / "synthesis_quant.npy", q_dec.cpu().numpy())
np.save(out_dir / "synthesis_ref.npy", xhat_ref.cpu().numpy())
print(f"intra_synthesis: input {tuple(y_hat.shape)} -> output {tuple(xhat_ref.shape)}")
print(f"\nTest tensors saved to {out_dir}")
