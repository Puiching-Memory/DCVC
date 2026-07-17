# PyTorch DCVC-RT benchmark: same setup as native test_perf (256x256, fBm noise)
# Follows test_video.py logic exactly for correct QP shifting and sps construction.
import time
import numpy as np
import torch

from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.utils.common import get_state_dict


def gen_natural_y(w, h, rng, motion_x, motion_y):
    field = np.zeros((h, w), dtype=np.float32)
    for oct in range(5):
        gw = 4 << oct
        grid = rng.uniform(-1, 1, (gw, gw)).astype(np.float32)
        amp = (0.5 ** oct) * 0.5
        for p in range(h):
            for q in range(w):
                gx = (q + motion_x) / (w - 1) * (gw - 1)
                gy = (p + motion_y) / (h - 1) * (gw - 1)
                x0, y0 = max(0, min(int(gx), gw-2)), max(0, min(int(gy), gw-2))
                fx, fy = gx - x0, gy - y0
                v00 = grid[y0, x0]; v10 = grid[y0, x0+1]
                v01 = grid[y0+1, x0]; v11 = grid[y0+1, x0+1]
                field[p, q] += (v00*(1-fx)*(1-fy) + v10*fx*(1-fy) +
                                v01*(1-fx)*fy + v11*fx*fy) * amp
    mn, mx = field.min(), field.max()
    field = 16 + 219.0 * (field - mn) / max(mx - mn, 1e-6)
    return np.clip(field, 0, 255).astype(np.uint8)


def gen_chroma(w, h, off):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.clip(128 + 30*np.sin(xx*0.03+off)*np.cos(yy*0.04+off*0.7), 0, 255).astype(np.uint8)


def main():
    import sys
    qp_i = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    W = H = 256
    n_warmup = 3
    device = torch.device("cuda:0")
    intra_period = -1  # only first frame is I
    reset_interval = 32
    use_two_entropy_coders = True

    # Build nets (official test_video.py setup)
    i_net = DMCI()
    i_net.load_state_dict(get_state_dict("checkpoints/cvpr2025_image.pth.tar"))
    i_net = i_net.to(device).eval()
    i_net.update(None)
    i_net.half()
    i_net.set_use_two_entropy_coders(use_two_entropy_coders)

    p_net = DMC()
    p_net.load_state_dict(get_state_dict("checkpoints/cvpr2025_video.pth.tar"))
    p_net = p_net.to(device).eval()
    p_net.update(None)
    p_net.half()

    index_map = [0, 1, 0, 2, 0, 2, 0, 2]
    padding_r, padding_b = DMCI.get_padding_size(H, W, 16)

    # Pre-generate frames as YUV420 -> pad -> YCbCr444 f16 [0,1] centered
    rng_seed = np.random.RandomState(42)
    frames = []
    for f in range(n_frames):
        y = gen_natural_y(W, H, np.random.RandomState(42+f), f*2.0, f*1.0)
        u = gen_chroma(W//2, H//2, f*0.3)
        v = gen_chroma(W//2, H//2, f*0.5+1.0)
        u444 = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
        v444 = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)
        ycbcr = np.stack([y, u444-128, v444-128], axis=0).astype(np.float32)
        x = torch.from_numpy(ycbcr).unsqueeze(0).to(device).half() / 255.0
        x_padded = torch.nn.functional.pad(x, (0, padding_r, 0, padding_b), mode="replicate")
        frames.append(x_padded)

    enc_ms = np.zeros(n_frames)
    dec_ms = np.zeros(n_frames)
    bytes_arr = np.zeros(n_frames)
    last_qp = qp_i

    for f in range(n_frames):
        x_padded = frames[f]
        is_i = (f == 0) or (intra_period > 0 and f % intra_period == 0)

        if is_i:
            curr_qp = qp_i
            sps = {'sps_id': -1, 'height': H, 'width': W,
                   'ec_part': 1 if use_two_entropy_coders else 0, 'use_ada_i': 0}
            torch.cuda.synchronize(device)
            t0 = time.time()
            encoded = i_net.compress(x_padded, curr_qp)
            torch.cuda.synchronize(device)
            enc_ms[f] = (time.time() - t0) * 1000
            p_net.clear_dpb()
            p_net.add_ref_frame(None, encoded['x_hat'])
            bs = encoded['bit_stream']
            bytes_arr[f] = len(bytes(bs))
            torch.cuda.synchronize(device)
            t1 = time.time()
            dres = i_net.decompress(bs, sps, curr_qp)
            torch.cuda.synchronize(device)
            dec_ms[f] = (time.time() - t1) * 1000
            last_qp = curr_qp
        else:
            fa_idx = index_map[f % 8]
            if reset_interval > 0 and f % reset_interval == 1:
                p_net.prepare_feature_adaptor_i(last_qp)
            curr_qp = p_net.shift_qp(qp_i, fa_idx)
            sps = {'sps_id': -1, 'height': H, 'width': W,
                   'ec_part': 1 if use_two_entropy_coders else 0, 'use_ada_i': 0}
            torch.cuda.synchronize(device)
            t0 = time.time()
            encoded = p_net.compress(x_padded, curr_qp)
            torch.cuda.synchronize(device)
            enc_ms[f] = (time.time() - t0) * 1000
            bs = encoded['bit_stream']
            bytes_arr[f] = len(bytes(bs))
            torch.cuda.synchronize(device)
            t1 = time.time()
            dres = p_net.decompress(bs, sps, curr_qp)
            torch.cuda.synchronize(device)
            dec_ms[f] = (time.time() - t1) * 1000
            last_qp = curr_qp

    idx = np.arange(n_warmup, n_frames)
    e = enc_ms[idx]; d = dec_ms[idx]
    print("=" * 60)
    print(" PyTorch DCVC-RT Benchmark (256x256, A30)")
    print("=" * 60)
    print(f" QP_i={qp_i}  frames={n_frames}  warmup={n_warmup}")
    print(f"                  ENCODE          DECODE")
    print(f"  Avg latency:  {e.mean():6.2f} ms       {d.mean():6.2f} ms")
    print(f"  Std dev:      {e.std():6.2f} ms       {d.std():6.2f} ms")
    print(f"  Throughput:   {1000/e.mean():6.1f} fps       {1000/d.mean():6.1f} fps")
    print(f"  End-to-end:   {(e.mean()+d.mean()):.2f} ms/frame")
    print(f"  Avg bitrate:  {bytes_arr[idx].mean()*8/(W*H):.4f} bpp")
    print("=" * 60)


if __name__ == "__main__":
    main()
