# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# DCVC-RK layers (NPU-first design). Base: DCVC-UF. Each change is backed by a
# RKNN-toolkit measurement (see micro-benchmarks ffn_variants2.py / fallback_sweep2.py).
#
# 1. WSiLU = sigmoid(4x)*x  -> RKNN folds to one exSwish op. KEPT from UF.
# 2. Upsample: SubpelConv2x (Conv+PixelShuffle) -> Conv + F.interpolate(nearest).
#    RKNN re-lowers PixelShuffle/DepthToSpace to a costly ConvTranspose (~29% of
#    intra_synthesis); Resize stays native NPU, ~free (16 us, 0 cycles).
# 3. FFN: WSiLUChunkAdd (UF: expand x4 + channel-stride add contraction) -> GLU.
#    RKNN fuses a*sigmoid(b) to a native exGlu op; GLU's 2C intermediate is half
#    of x4 (lower memory traffic), and the gate is learned (better capacity at
#    lower width). Measured: x4_chunkadd=Slice+Split+exSwish (41MB, fb=2);
#    x2=ConvExSwish (41MB, fb=1); GLU=exGlu (18MB, fb=0). GLU wins on every axis.
# 4. dc branch (pw->dw3x3->pw): KEPT identical to UF (NPU-native dw3x3, no
#    fallback in practice; weights transfer).
# 5. Downsample: UF already uses stride-2 Conv (NPU-only) -> unchanged.

import torch
import torch.nn.functional as F
from torch import nn


class WSiLU(nn.Module):
    """sigmoid(4x)*x. RKNN folds to exSwish. Identical to UF (weights transfer)."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(4.0 * x) * x


class ResizeUpsampleRK(nn.Module):
    """Conv -> nearest Resize(scale). THE RKNN upsample solution.

    SubpelConv2x (PixelShuffle) is lowered by RKNN to a ConvTranspose that costs
    ~29% of intra_synthesis on rk3588. F.interpolate(nearest) stays a native
    NPU 'Resize' op (~16 us, 0 compute cycles). The conv produces out_ch directly
    (not out_ch*scale^2), so it is also cheaper than pixel_shuffle's conv.
    Trade-off: no learned sub-pixel arrangement -> needs training.
    """

    def __init__(self, in_ch, out_ch, kernel_size=1, scale=2, mode="nearest"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = self.conv(x)
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode)


class GLUFFN(nn.Module):
    """Gated linear unit FFN: gate ea(x)*sigmoid(eb(x)) -> Conv(C->C), residual.

    Expand is SPLIT into two Conv(C->C) instead of one Conv(C->2C)+chunk. The math
    is identical (a merged (2C,C,1,1) weight splits row-wise into ea/eb), but each
    output tensor is width-C not 2C. This matters for RKNN 3-core tiling: 3-core
    replicates the layer's activation buffer across 3 cores, so a 2C buffer at
    f-resolution (136x240) exceeds SRAM and the compiler demotes the layer to
    single-core ("3Core fallback"). With two C-width convs the peak tensor halves,
    clearing the threshold -> 0 fallback, plus native ConvSigmoid/ConvMul fusion.
    Measured C=256 @136x240: merged Conv(256->512)=1 fallback; split =0 fallback.
    """

    def __init__(self, ch):
        super().__init__()
        self.ea = nn.Conv2d(ch, ch, 1)
        self.eb = nn.Conv2d(ch, ch, 1)
        self.c = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return self.c(self.ea(x) * torch.sigmoid(self.eb(x))) + x


class DepthConvBlockRK(nn.Module):
    """UF-compatible DepthConvBlock with GLU FFN.

    dc branch (pw->WSiLU->dw3x3->pw): identical to UF -> weights transfer.
    ffn branch: UF WSiLUChunkAdd -> GLUFFN -> needs training (different math).
    shortcut / dcb2 / adaptor semantics preserved from UF DepthConvBlock.

    split_adaptor=True: replaces a concat-fed Conv(2C->C) adaptor with two
    Conv(C->C) adaptors summed as ca(a)+cb(b). Same math & FLOPs as Conv(2C->C)
    on cat([a,b]) (W.[a;b]=W_a.a+W_b.b), but each half is C-wide so 3-core
    tiling clears the SRAM bank budget -- a11 TileChannel demotes Conv(2C->C)
    (ic=512, 8 weight banks) to single-core; split = 0 fallback. Weight transfer
    from the merged (C,2C,1,1): ca.weight=W[:, :C], ca.bias=bias, cb.weight=W[:, C:]
    (cb has bias=False). dc/ffn branches are unchanged.
    """

    def __init__(self, in_ch, out_ch, *, dcb2=False, shortcut=False,
                 force_adaptor=False, split_adaptor=False):
        super().__init__()
        self.split_adaptor = split_adaptor
        self.adaptor = None
        if split_adaptor:
            # forward must be called as block(a, b); each input is in_ch wide.
            self.ca = nn.Conv2d(in_ch, out_ch, 1)
            self.cb = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        elif in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)
        ch_ratio = 1
        if dcb2:
            assert not shortcut
            ch_ratio = 2
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // ch_ratio, 1),
            WSiLU(),
            nn.Conv2d(out_ch // ch_ratio, out_ch // ch_ratio, 3, padding=1,
                      groups=out_ch // ch_ratio),
            nn.Conv2d(out_ch // ch_ratio, out_ch, 1),
        )
        self.ffn = GLUFFN(out_ch)

    def forward(self, x, x_b=None):
        if self.split_adaptor:
            x = self.ca(x) + self.cb(x_b)
        elif self.adaptor is not None:
            x = self.adaptor(x)
        out = self.dc(x) + x
        out = self.ffn(out)
        if self.shortcut:
            out = out + x
        return out


class ResidualBlockUpsampleResize(nn.Module):
    """Upsample x2 (Resize) + DepthConvBlockRK. Replaces UF ResidualBlockUpsample."""

    def __init__(self, in_ch, out_ch, dcb2=False, shortcut=False, force_bias=False):
        super().__init__()
        self.up = ResizeUpsampleRK(in_ch, out_ch, kernel_size=1, scale=2, mode="nearest")
        self.conv = DepthConvBlockRK(out_ch, out_ch, dcb2=dcb2, shortcut=shortcut)
        self.shortcut = shortcut

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out
