# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# QAT fake-quant for DCVC-RK. Wraps every Conv2d with STE fake quantization
# (u8 per-tensor activation + s8 per-channel weight), matching RKNN's w8a8 /
# channel quantization so QAT-trained weights transfer to the int8 RKNN export.
#
# Match rationale: RKNN config(quantized_dtype='w8a8', quantized_method='channel')
# does per-channel s8 weight + per-tensor u8 activation. FakeQuantConv2d uses the
# exact same scheme (weight absmax/127 per-output-channel, act absmax/127), so the
# quantization noise QAT trains against is the noise RKNN will apply at deploy.
#
# Flow: wrap_qat(model) -> train -> freeze_qat(model) -> export normal fp ONNX
# -> build_rknn_i8.py (do_quantization=True) -> int8 .rknn. The fp ONNX already
# carries STE-baked weights, so RKNN's PTQ is near-lossless on QAT weights.
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ste_round(x):
    return (x.round() - x).detach() + x


class FakeQuantConv2d(nn.Module):
    """Drop-in Conv2d with STE fake quant (u8 act per-tensor / s8 weight per-channel)."""

    def __init__(self, conv: nn.Conv2d, momentum=0.01):
        super().__init__()
        self.conv = conv
        self.momentum = momentum
        cout = conv.weight.shape[0]
        self.register_buffer("w_scale", torch.ones(cout))
        self.register_buffer("x_scale", torch.ones(1))
        self.register_buffer("x_absmax", torch.ones(1))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("frozen", torch.zeros(1, dtype=torch.bool))

    def _update_x(self, x):
        if bool(self.frozen):
            return
        am = x.detach().abs().amax()
        am = torch.clamp(am, min=1e-8)
        if not bool(self.initialized):
            self.x_absmax.copy_(am)
            self.initialized.fill_(True)
        else:
            self.x_absmax.mul_(1 - self.momentum).add_(am * self.momentum)
        self.x_scale.copy_(self.x_absmax / 127.0)

    def forward(self, x):
        self._update_x(x)
        sx = self.x_scale.clamp(min=1e-8)
        xq = _ste_round(x / sx).clamp(-128, 127) * sx
        w = self.conv.weight
        flat = w.detach().abs().reshape(w.shape[0], -1).amax(dim=1).clamp(min=1e-8)
        if not bool(self.frozen):
            self.w_scale.copy_(flat / 127.0)
        sw = self.w_scale.view(-1, *([1] * (w.ndim - 1))).clamp(min=1e-8)
        wq = _ste_round(w / sw).clamp(-127, 127) * sw
        return F.conv2d(xq, wq, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)


def _resolve(model, parts):
    obj = model
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def wrap_qat(model: nn.Module, skip_depthwise=True):
    """Replace every Conv2d under model with a FakeQuantConv2d (keeps weights).
    skip_depthwise=True leaves depthwise (groups==C) convs in fp16, because the
    NPU depthwise path is cheap (0.6% of MACs) and quantizing it hurts accuracy
    for no throughput gain -- matches the hybrid split in qat_backbone_i8.py."""
    targets = [n for n, m in model.named_modules()
               if isinstance(m, nn.Conv2d)
               and not (skip_depthwise and m.groups == m.weight.shape[0]
                        and m.weight.shape[0] == m.in_channels)]
    for name in targets:
        parts = name.split(".")
        parent = _resolve(model, parts)
        last = parts[-1]
        old = parent[int(last)] if last.isdigit() else getattr(parent, last)
        fq = FakeQuantConv2d(old)
        if last.isdigit():
            parent[int(last)] = fq
        else:
            setattr(parent, last, fq)
    return len(targets)


def freeze_qat(model: nn.Module):
    """Freeze the learned act/weight scales so export uses fixed quantization."""
    n = 0
    for m in model.modules():
        if isinstance(m, FakeQuantConv2d):
            m.frozen.fill_(True)
            n += 1
    return n


def materialize_qat(model: nn.Module):
    """Replace each FakeQuantConv2d back with its plain Conv2d (float weights).

    STE only matters during training (gradient path). At export we want a clean
    Conv graph carrying the QAT-trained float weights, which RKNN's
    do_quantization then quantizes to int8 with near-lossless accuracy (because
    training made the weights robust to exactly that quantization). Leaving the
    fake-quant Round/Clip/Mul ops in the ONNX would confuse RKNN's quantizer."""
    targets = [n for n, m in model.named_modules() if isinstance(m, FakeQuantConv2d)]
    for name in targets:
        parts = name.split(".")
        parent = _resolve(model, parts)
        last = parts[-1]
        fq = parent[int(last)] if last.isdigit() else getattr(parent, last)
        conv = fq.conv
        if last.isdigit():
            parent[int(last)] = conv
        else:
            setattr(parent, last, conv)
    return len(targets)
