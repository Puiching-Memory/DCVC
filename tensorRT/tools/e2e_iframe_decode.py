#!/usr/bin/env python3
"""End-to-end I-frame decode using TRT engines + Python rANS.

Validates the full pipeline: bitstream → z_hat → y_hat → x_hat
compared against PyTorch golden data.
"""
import ctypes, os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"src/cpp"))
os.environ.setdefault("SUPPRESS_CUSTOM_KERNEL_WARNING", "1")

for so in ["libdcvc_depthconv.so", "libdcvc_subpel.so"]:
    ctypes.CDLL(str(ROOT/"tensorRT/build/plugin_demo"/so), mode=ctypes.RTLD_GLOBAL)

import tensorrt as trt
from src.models.image_model import DMCI, g_ch_enc_dec
from src.layers.cuda_inference import combine_for_reading_2x, restore_y_2x, \
    restore_y_4x, add_and_multiply, replicate_pad
from src.utils.common import get_state_dict

def wn(t): return t.data.cpu().numpy().astype(np.float32).flatten()
def bn(t): return t.data.cpu().numpy().astype(np.float32).flatten()

_creators = {}
def get_creator(name):
    if name not in _creators:
        for c in trt.get_plugin_registry().all_creators:
            if c.name == name and c.plugin_version == "1":
                _creators[name] = c; break
    return _creators[name]

# --- Engine management ---
_engines = {}
def get_engine(runner, name):
    if name in _engines: return _engines[name]
    e = runner._load_engine(name)
    _engines[name] = e
    return e

class EngineRunner:
    def __init__(self, asset_dir, dev):
        self.asset_dir = Path(asset_dir)
        self.dev = dev
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self._engines = {}
        self._contexts = {}

    def _load_engine(self, name):
        path = self.asset_dir / "engines" / f"{name}.engine"
        with open(path, 'rb') as f:
            data = f.read()
        eng = self.runtime.deserialize_cuda_engine(data)
        ctx = eng.create_execution_context()
        self._engines[name] = eng
        self._contexts[name] = ctx
        return eng, ctx

    def run(self, name, inputs):
        eng, ctx = self._load_engine(name) if name not in self._contexts else (self._engines[name], self._contexts[name])

        tensor_names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
        input_names = [n for n in tensor_names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        output_names = [n for n in tensor_names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        for n, t in zip(input_names, inputs):
            ctx.set_input_shape(n, list(t.shape))
            ctx.set_tensor_address(n, t.data_ptr())
        # Allocate output
        outputs = []
        for n in output_names:
            shape = tuple(ctx.get_tensor_shape(n))
            out = torch.empty(shape, dtype=torch.float16, device=self.dev)
            ctx.set_tensor_address(n, out.data_ptr())
            outputs.append(out)

        stream = torch.cuda.Stream(self.dev)
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return outputs[0] if len(outputs) == 1 else outputs


def run_full_decode():
    dev = "cuda:0"
    runner = EngineRunner(ROOT / "tensorRT/assets", dev)

    # Load PyTorch model for reference and CDF tables
    net = DMCI().to(dev).half().eval()
    net.load_state_dict(get_state_dict(str(ROOT / "checkpoints/cvpr2025_image.pth.tar")))
    net.update()

    # Load golden bitstream
    golden_dir = ROOT / "tensorRT/assets/golden"
    meta = json.loads((golden_dir / "i_256x256_qp20_meta.json").read_text())
    with open(golden_dir / "i_256x256_qp20_bitstream.bin", 'rb') as f:
        bitstream = f.read()
    print(f"Bitstream: {len(bitstream)} bytes, meta: {meta}")

    H, W = meta['height'], meta['width']
    qp = meta['qp']
    ec_part = meta['ec_part']
    N = 256
    z_channel = 128

    # Step 1: rANS decode z
    z_h, z_w = net.get_downsampled_shape(H, W, 64)
    y_h, y_w = net.get_downsampled_shape(H, W, 16)
    print(f"z_size: ({z_h}, {z_w}), y_size: ({y_h}, {y_w})")

    net.entropy_coder.set_use_two_entropy_coders(ec_part == 1)
    net.entropy_coder.set_stream(bitstream)
    net.bit_estimator_z.decode_z((z_h, z_w), qp)
    z_hat = net.bit_estimator_z.get_z((z_h, z_w), dev, torch.float16)
    print(f"z_hat: {tuple(z_hat.shape)}")

    # Step 2: hyper_dec
    params = runner.run("hyper_dec", [z_hat])
    print(f"After hyper_dec: {tuple(params.shape)}")

    # Step 3: y_prior_fusion
    params_full = runner.run("y_prior_fusion", [params])
    params_cropped = params_full[:, :, :y_h, :y_w].contiguous()
    print(f"After y_prior_fusion: {tuple(params_cropped.shape)}")

    # Step 4: decompress_prior_4x using PyTorch rANS
    y_hat = net.decompress_prior_4x(
        params_cropped,
        net.y_spatial_prior_reduction,
        net.y_spatial_prior_adaptor_1,
        net.y_spatial_prior_adaptor_2,
        net.y_spatial_prior_adaptor_3,
        net.y_spatial_prior)
    print(f"y_hat: {tuple(y_hat.shape)}")

    # Step 5: synthesis (using TRT engine)
    q_dec = net.q_scale_dec[qp:qp+1, :, :, :].half()
    x_hat = runner.run("intra_synthesis", [y_hat, q_dec])
    x_hat = x_hat.clamp_(0, 1)
    print(f"x_hat: {tuple(x_hat.shape)}")

    # Compare to golden
    golden_xhat = np.load(golden_dir / "i_256x256_qp20_xhat_enc.npy")
    golden_t = torch.from_numpy(golden_xhat).to(dev).half()
    # Golden is HWC, x_hat is CHW — permute for comparison
    if golden_t.shape[1] == x_hat.shape[2]:  # HWC vs CHW
        golden_t = golden_t.permute(0, 3, 1, 2).contiguous()
    err = float((x_hat.float() - golden_t.float()).abs().max())
    psnr = 10 * torch.log10(1.0 / ((x_hat.float() - golden_t.float()) ** 2).mean())
    print(f"\n=== RESULT ===")
    print(f"max_abs_err = {err:.6e}")
    print(f"PSNR = {psnr.item():.2f} dB")
    print(f"Reference PSNR = {meta['psnr_db']:.2f} dB")
    return err


if __name__ == "__main__":
    run_full_decode()
