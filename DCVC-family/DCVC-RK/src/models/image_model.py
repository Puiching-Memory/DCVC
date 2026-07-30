# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# DCVC-RK intra (I-frame) model. Mirrors DCVC-UF's image_model with RK-friendly
# ops: Resize upsampling (replaces SubpelConv2x/DepthToSpace->ConvTranspose),
# GLU FFN (replaces WSiLUChunkAdd). All changes need training.

import torch
import torch.nn.functional as F
from torch import nn

from ..layers.layers import (
    DepthConvBlockRK,
    ResidualBlockUpsampleResize,
    ResizeUpsampleRK,
)

# DCVC-UF intra channel plan (from cvpr2025_image.pth.tar).
g_ch_src = 3 * 8 * 8      # 192
g_ch_enc_dec = 256
g_ch_y = 256
g_ch_z = 128


class IntraDecoderRK(nn.Module):
    """intra_synthesis: y + quant -> dec stack + up(x8) -> image."""

    def __init__(self):
        super().__init__()
        self.dec_1 = nn.Sequential(
            ResidualBlockUpsampleResize(g_ch_y, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
        )
        self.dec_2 = DepthConvBlockRK(g_ch_enc_dec, g_ch_src)
        # x8 upsample to image res via NPU-native Resize.
        self.up_x8 = ResizeUpsampleRK(g_ch_src, 3, kernel_size=1, scale=8, mode="nearest")

    def forward(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step
        out = self.dec_2(out)
        return self.up_x8(out)


class IntraEncoderRK(nn.Module):
    """intra_analysis: image -> y latent. Strided stem (replaces pixel_unshuffle)."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, g_ch_src, 3, stride=8, padding=1)
        self.enc_1 = DepthConvBlockRK(g_ch_src, g_ch_enc_dec)
        self.enc_2 = nn.Sequential(
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlockRK(g_ch_enc_dec, g_ch_enc_dec),
            nn.Conv2d(g_ch_enc_dec, g_ch_y, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        out = self.stem(x)
        out = self.enc_1(out)
        out = out * quant_step
        return self.enc_2(out)


class IntraPriorFusionRK(nn.Module):
    """intra y_prior_fusion: hyper_dec prior (256ch) -> mask+means+scales (514ch).

    DCVC intra uses a single hyperprior (NOT the inter autoregressive
    y_spatial_prior_* chain): hyper_dec output -> this net -> 2*N+2 channels,
    where the C runtime (onnx/src/cpu_ar_codec.c) splits them into
    mask[2] + scales[256] + means[256] for a gaussian+rANS entropy model of y.
    Mirrors checkpoint module.y_prior_fusion: adaptor(256->512) + 3 DCB@512 +
    head Conv(512->514). The 512 internal width is required for the 3-way split
    and matches the UF checkpoint 1:1 (weights transfer); it is wider than the
    RK-friendly 256/320 widths but the net is single-call and small."""

    def __init__(self):
        super().__init__()
        N = g_ch_y                       # 256
        w = 512
        out = 2 * N + 2                  # 514: mask(2) + scales(256) + means(256)
        self.conv = nn.Sequential(
            DepthConvBlockRK(N, w),      # block 0 (adaptor 256->512 baked in)
            DepthConvBlockRK(w, w),      # block 1
            DepthConvBlockRK(w, w),      # block 2
            nn.Conv2d(w, out, 1),        # head 512->514
        )

    def forward(self, x):
        return self.conv(x)


# AR spatial-prior channel width = 2*g_ch_y = 512 (checkpoint module.y_spatial_prior*).
g_ch_w = 512


class YSpatialPriorReductionRK(nn.Module):
    """y_spatial_prior_reduction: prior_fusion 514ch -> 256ch. Single 1x1 Conv.
    Starts the 4-pass AR chain: prior_fusion output -> reduction -> (per-pass
    scales/means seed). Runtime: cpu_ar_codec.c eng_reduction, input (1,514,h,w)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2 * g_ch_y + 2, g_ch_y, 1)

    def forward(self, x):
        return self.conv(x)


class YSpatialPriorAdaptorRK(nn.Module):
    """y_spatial_prior_adaptor_{1,2,3}: cat([yhat_so_far, mask_or_context]) 512ch
    -> 512ch via a single DepthConvBlockRK (adaptor + dc + ffn, all 512). One per AR
    pass; runtime builds the per-pass input (2*N=512 = yhat half + ctx half) and
    feeds the matching adaptor. Runtime: cpu_ar_codec.c eng_adaptor[round]."""

    def __init__(self):
        super().__init__()
        # checkpoint has explicit adaptor.weight (512,512) + dc + ffn, all @512 -> force it
        self.block = DepthConvBlockRK(g_ch_w, g_ch_w, force_adaptor=True)

    def forward(self, x):
        return self.block(x)


class YSpatialPriorRK(nn.Module):
    """y_spatial_prior: 3 DCB@512 + head Conv(512->512). The shared per-pass
    parameter refiner: adaptor_out -> this -> refined scales/means half for the
    current pass. Runtime: cpu_ar_codec.c eng_spatial_prior, (1,512,h,w)->(1,512,h,w)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlockRK(g_ch_w, g_ch_w),
            DepthConvBlockRK(g_ch_w, g_ch_w),
            DepthConvBlockRK(g_ch_w, g_ch_w),
            nn.Conv2d(g_ch_w, g_ch_w, 1),
        )

    def forward(self, x):
        return self.conv(x)


class IntraHyperEncoderRK(nn.Module):
    """intra_hyper_enc: y -> z, down x2 twice via stride-2 conv (no pixel_unshuffle)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlockRK(g_ch_y, g_ch_z),
            nn.Conv2d(g_ch_z, g_ch_z, 2, stride=2),
            DepthConvBlockRK(g_ch_z, g_ch_z),
            nn.Conv2d(g_ch_z, g_ch_z, 2, stride=2),
        )

    def forward(self, x):
        return self.conv(x)


class IntraHyperDecoderRK(nn.Module):
    """hyper_dec: z -> y-space prior, up x2 twice via Resize."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsampleResize(g_ch_z, g_ch_z),
            ResidualBlockUpsampleResize(g_ch_z, g_ch_z),
            DepthConvBlockRK(g_ch_z, g_ch_y),
        )

    def forward(self, x):
        return self.conv(x)
