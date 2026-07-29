#!/usr/bin/env python3
"""Sequential least-squares weight tuning for INT8 QDQ models (AdaQuant-lite).

After bias correction, the residual INT8 error is accumulated activation-
rounding noise through the conv stack. For each conv (topological order) we:
  1. take the conv's ACTUAL quantized input X_q from the current int8 model
     (truncated ORT run on calib inputs),
  2. take the FP32 reference output Y_f of the same conv (fp32 model),
  3. solve ridge-LS for weights+bias minimizing ||conv(X_q,W)-Y_f|| with a
     Tikhonov anchor to the original (dequantized) weights,
  4. write W* back as the fp32 input of the weight QuantizeLinear (the Q node
     re-rounds it onto the int8 grid at runtime), set bias = b*.

Grouped (incl. depthwise) convs are solved per group. calib = npz/npy dir.

Usage:
  .venv/bin/python trt_i8_weight_tune.py --fp32 ../models_smooth/intra_synthesis.onnx \
      --int8 /tmp/synth_bc.onnx --calib calib_trt_i8/intra_synthesis --out /tmp/synth_wt.onnx
"""
import argparse
import glob
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import numpy_helper

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def truncate_by_output(m, out_name, path):
    g = m.graph
    idx = next(i for i, n in enumerate(g.node) if out_name in n.output)
    sub = onnx.helper.make_graph(
        list(g.node[: idx + 1]), g.name + "_sub", list(g.input),
        [onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, None)],
        list(g.initializer))
    mm = onnx.helper.make_model(sub, producer_name="wt",
                                opset_imports=list(m.opset_import))
    mm.ir_version = m.ir_version
    onnx.save(mm, path)
    return path


def load_calib(calib_dir):
    files = sorted(glob.glob(os.path.join(calib_dir, "sample_*.npz"))) or \
            sorted(glob.glob(os.path.join(calib_dir, "*.npy")))
    for f in files:
        if f.endswith(".npz"):
            z = np.load(f)
            yield {k: z[k].astype(np.float32) for k in z.files}
        else:
            yield {"in0": np.load(f).astype(np.float32)}


def solve_conv_ls(xq, yf, w0, b0, stride, pads, dil, groups, lam):
    """Ridge LS for conv weights. xq:[N,Ci,H,W] yf:[N,Co,H',W'] w0:[Co,Ci/g,kh,kw]."""
    Co, Cig, kh, kw = w0.shape
    xt = torch.from_numpy(xq).to(DEV, torch.float64)
    yt = torch.from_numpy(yf).to(DEV, torch.float64)
    cols = F.unfold(xt, (kh, kw), dilation=dil, padding=(pads[0], pads[2]),
                    stride=stride)                      # [N, Ci*kh*kw, L]
    L = cols.shape[2]
    w_new = np.zeros_like(w0, dtype=np.float64)
    b_new = np.zeros(Co, dtype=np.float64)
    Ci = xq.shape[1]
    for gi in range(groups):
        ch_in = slice(gi * Cig, (gi + 1) * Cig)
        a = cols[:, gi * Cig * kh * kw:(gi + 1) * Cig * kh * kw, :]  # [N, Cig*k*k, L]
        a = a.permute(0, 2, 1).reshape(-1, Cig * kh * kw)            # [N*L, P]
        a = torch.cat([a, torch.ones(a.shape[0], 1, dtype=a.dtype, device=a.device)], 1)
        co = range(gi * (Co // groups), (gi + 1) * (Co // groups))
        y = yt[:, list(co), :, :].permute(0, 2, 3, 1).reshape(-1, len(co))  # [N*L, Co_g]
        # target unfold already matches conv geometry via unfold/conv pairing
        ata = a.T @ a
        atb = a.T @ y
        p = ata.shape[0]
        w0g = torch.from_numpy(
            np.concatenate([w0[list(co)].reshape(len(co), -1).T,
                            b0[list(co)][None]], 0)).to(DEV, torch.float64)  # [P+1? no P, Co_g]
        # anchor: lam * (w - w0)
        sol = torch.linalg.solve(ata + lam * torch.eye(p, dtype=ata.dtype, device=ata.device),
                                 atb + lam * w0g)
        w_new[list(co)] = sol[:-1].T.reshape(len(co), Cig, kh, kw).cpu().numpy()
        b_new[list(co)] = sol[-1].cpu().numpy()
    return w_new, b_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=1e-2, help="anchor strength")
    ap.add_argument("--max-calib", type=int, default=6)
    args = ap.parse_args()

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    prov = ["CPUExecutionProvider"]

    mf = onnx.load(args.fp32)
    mq = onnx.load(args.int8)
    calib = list(load_calib(args.calib))[: args.max_calib]
    qi = {i.name: i for i in mq.graph.initializer}
    fi = {i.name: numpy_helper.to_array(i) for i in mf.graph.initializer}

    # map: int8 conv weight Q node input (fp32 weight init) per conv
    prod = {o: n for n in mq.graph.node for o in n.output}

    n_done = 0
    for node in mq.graph.node:
        if node.op_type != "Conv":
            continue
        # find fp32 twin conv by output name
        twin = next((n for n in mf.graph.node if n.output[0] == node.output[0]), None)
        if twin is None:
            continue
        pads = next((list(a.ints) for a in node.attribute if a.name == "pads"), [0, 0, 0, 0])
        stride = next((list(a.ints) for a in node.attribute if a.name == "strides"), [1, 1])
        dil = next((list(a.ints) for a in node.attribute if a.name == "dilations"), [1, 1])
        groups = next((a.i for a in node.attribute if a.name == "group"), 1)

        # X_q: truncate int8 model at the conv's input tensor
        truncate_by_output(mq, node.input[0], "/tmp/wt_q.onnx")
        sq = ort.InferenceSession("/tmp/wt_q.onnx", sess_options=opts, providers=prov)
        # Y_f: truncate fp32 model at the conv's output
        truncate_by_output(mf, twin.output[0], "/tmp/wt_f.onnx")
        sf = ort.InferenceSession("/tmp/wt_f.onnx", sess_options=opts, providers=prov)
        xqs, yfs = [], []
        for feed in calib:
            xqs.append(sq.run([node.input[0]], feed)[0])
            yfs.append(sf.run([twin.output[0]], feed)[0])
        xq = np.concatenate(xqs, 0)
        yf = np.concatenate(yfs, 0)

        # current dequantized weight: either a plain fp32 initializer, or the
        # fp32 input init of the weight QuantizeLinear (QDQ)
        if node.input[1] in qi:
            w_init_name = node.input[1]
        else:
            dqn = prod[node.input[1]]
            qn = prod[dqn.input[0]]
            w_init_name = qn.input[0]
        w0 = numpy_helper.to_array(qi[w_init_name]).astype(np.float64)
        b0 = numpy_helper.to_array(qi[node.input[2]]).astype(np.float64) \
            if len(node.input) >= 3 else np.zeros(w0.shape[0])

        try:
            w_new, b_new = solve_conv_ls(xq, yf, w0, b0, stride, pads, dil, groups, args.lam)
        except Exception as e:
            print(f"skip {node.name}: {type(e).__name__} {str(e)[:80]}")
            continue
        qi[w_init_name].CopyFrom(numpy_helper.from_array(w_new.astype(np.float32), w_init_name))
        if len(node.input) >= 3:
            qi[node.input[2]].CopyFrom(numpy_helper.from_array(b_new.astype(np.float32), node.input[2]))
        n_done += 1
        if n_done % 12 == 0:
            print(f"  tuned {n_done} convs...")

    onnx.save(mq, args.out)
    print(f"weight-tuned {n_done} convs -> {args.out}")


if __name__ == "__main__":
    main()
