# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# DCVC-RK inter (P-frame) subnets. Based on DCVC-UF (checkpoint ground truth:
# checkpoints/cvpr2025_video.pth.tar), with RKNN-friendly op swaps.
# Each subnet mirrors UF's exact structure; only the RK-hostile ops change:
#   - SubpelConv2x upsampling -> Conv + Resize (hyper_decoder, decoder)
#   - WSiLUChunkAdd FFN       -> GLU FFN        (every DepthConvBlock)
# dc branch, WSiLU activation, stride-2 down convs, channel widths: all unchanged.

import torch
import torch.nn as nn

from ..layers.layers import (
    DepthConvBlockRK,
    ResizeUpsampleRK,
    ResidualBlockUpsampleResize,
)


# UF channel plan (from checkpoint shapes).
g_ch_src_d = 3 * 8 * 8      # 192
g_ch_y = 128
g_ch_z = 128
g_ch_d = 256
g_ch_m = 256
g_ch_recon = 320


class FeatureAdaptorIRK(nn.Module):
    """inter_feature_adaptor_i: single DCB(192->256) [adaptor+dc+ffn]."""

    def __init__(self):
        super().__init__()
        self.conv = DepthConvBlockRK(g_ch_src_d, g_ch_m)

    def forward(self, x):
        return self.conv(x)


class FeatureAdaptorPRK(nn.Module):
    """inter_feature_adaptor_p: single 1x1 Conv(256->256). Fully transferable."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(g_ch_d, g_ch_d, 1)

    def forward(self, x):
        return self.conv(x)


class FeatureExtractorRK(nn.Module):
    """inter_feature_extractor (UF checkpoint): conv1[2 DCB] + conv2[4 DCB] = 6 DCB,
    single output ctx. q_feature scaling is applied later in y_prior_fusion
    (cat(hyper, temporal*quant)), NOT here -- the checkpoint is a clean 6-DCB stack
    with no split head and no in-net quant param."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthConvBlockRK(g_ch_m, g_ch_m),
            DepthConvBlockRK(g_ch_m, g_ch_m),
        )
        self.conv2 = nn.Sequential(
            DepthConvBlockRK(g_ch_m, g_ch_m),
            DepthConvBlockRK(g_ch_m, g_ch_m),
            DepthConvBlockRK(g_ch_m, g_ch_m),
            DepthConvBlockRK(g_ch_m, g_ch_m),
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


class EncoderRK(nn.Module):
    """inter_encoder (UF checkpoint): conv1(192->256) -> conv2[DCB(512->256),
    DCB(256->256)] cat ctx -> conv3 DCB(256->256) -> *quant -> down(stride2 256->128).
    pixel_unshuffle is done in C; the subnet takes the 192-ch unshuffled input
    directly (checkpoint conv1 in=192, NOT the stale 448-cat source)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_src_d, g_ch_d, 1)
        # conv2[0] was DCB(512->256) fed cat([feature,ctx]); split_adaptor feeds the
        # two 256-ch halves separately (ca(feature)+cb(ctx)) -> 3-core on a11 (was fb=2).
        self.conv2_0 = DepthConvBlockRK(g_ch_d, g_ch_d, split_adaptor=True)
        self.conv2_1 = DepthConvBlockRK(g_ch_d, g_ch_d)
        self.conv3 = DepthConvBlockRK(g_ch_d, g_ch_d)
        self.down = nn.Conv2d(g_ch_d, g_ch_y, 3, stride=2, padding=1)

    def forward(self, x, ctx, quant_step):
        feature = self.conv1(x)
        feature = self.conv2_0(feature, ctx)
        feature = self.conv2_1(feature)
        feature = self.conv3(feature)
        feature = feature * quant_step
        return self.down(feature)


class HyperEncoderRK(nn.Module):
    """inter_hyper_enc (UF checkpoint): DCB(128->128) + 2x [stride-2 Conv(2x2) +
    DCB(128->128)]. UF down.weight is (128,128,2,2) -> plain stride-2 conv, NOT
    pixel_unshuffle (which would be (128,512,1,1)). Each down is followed by a DCB
    (UF ResidualBlockWithStride2 minus the now-removed unshuffle)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlockRK(g_ch_y, g_ch_z),
            nn.Conv2d(g_ch_z, g_ch_z, 2, stride=2),
            DepthConvBlockRK(g_ch_z, g_ch_z),
            nn.Conv2d(g_ch_z, g_ch_z, 2, stride=2),
            DepthConvBlockRK(g_ch_z, g_ch_z),
        )

    def forward(self, x):
        return self.conv(x)


class HyperDecoderRK(nn.Module):
    """inter_hyper_dec: 2x upsample (SubpelConv2x -> Resize) + DCB(128->128).
    UF: ResidualBlockUpsample(SubpelConv2x+DCB) x2, then DCB(128->128).
    RK: up via Resize. The 2 DCBs after each up stay (UF structure)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ResizeUpsampleRK(g_ch_z, g_ch_z, 1, scale=2, mode="nearest"),
            DepthConvBlockRK(g_ch_z, g_ch_z),
            ResizeUpsampleRK(g_ch_z, g_ch_z, 1, scale=2, mode="nearest"),
            DepthConvBlockRK(g_ch_z, g_ch_z),
            DepthConvBlockRK(g_ch_z, g_ch_y),
        )

    def forward(self, x):
        return self.conv(x)


class TemporalPriorEncoderRK(nn.Module):
    """inter_temporal_prior (UF checkpoint): stride-2 Conv(256->256, 2x2) +
    DCB(256->256). Single input (memory, 256, f-res) -> output (256, y-res). This
    output is the 'temporal' half of y_prior_fusion's cat(hyper(128) + temporal(256))."""

    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(g_ch_d, g_ch_d, 2, stride=2)
        self.conv = DepthConvBlockRK(g_ch_d, g_ch_d)

    def forward(self, x):
        return self.conv(self.down(x))


class PriorFusionRK(nn.Module):
    """inter_prior_fusion: 3 DCB @ 320 + 1x1 out(320->384). Input/output stay 384
    (= g_ch_y*3, the 3-latent concat the C engine feeds), but the DCB *internal* width
    drops to 320: the only 3-core-hostile op in this subnet was the depthwise (dc.2) at
    384ch; 320 is 3-core-friendly for depthwise while 1x1 convs never demote regardless
    of width. First DCB(384->320) adds an adaptor (3-core for 1x1), trailing Conv restores
    the 384 output width. Net: 3 depthwise layers move from single-core to 3-core."""

    def __init__(self):
        super().__init__()
        c = g_ch_y * 3
        w = 320
        self.conv = nn.Sequential(
            DepthConvBlockRK(c, w),
            DepthConvBlockRK(w, w),
            DepthConvBlockRK(w, w),
            nn.Conv2d(w, c, 1),
        )

    def forward(self, x):
        return self.conv(x)


class SpatialPriorRK(nn.Module):
    """inter_spatial_prior: adaptor(512->320) + 2 DCB @ 320 + 1x1(320->256).
    Internal width 320 (not 384) so depthwise stays 3-core; 1x1 convs never demote."""

    def __init__(self):
        super().__init__()
        w = 320
        self.conv = nn.Sequential(
            DepthConvBlockRK(g_ch_y * 4, w),
            DepthConvBlockRK(w, w),
            nn.Conv2d(w, g_ch_y * 2, 1),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderRK(nn.Module):
    """inter_decoder: up(SubpelConv2x->Resize) cat ctx -> 3 DCB(512->256) + 1x1.
    UF: SubpelConv2x(128->256,3x3) then 3 DCB. RK: up via Resize(128->256)."""

    def __init__(self):
        super().__init__()
        self.up = ResizeUpsampleRK(g_ch_y, g_ch_d, kernel_size=3, scale=2, mode="nearest")
        # conv1[0] was DCB(512->256) fed cat([up(x),ctx]); split_adaptor feeds the
        # two 256-ch halves separately (ca(feature)+cb(ctx)) -> 3-core on a11 (was fb=2).
        self.conv1_0 = DepthConvBlockRK(g_ch_d, g_ch_d, split_adaptor=True)
        self.conv1_1 = DepthConvBlockRK(g_ch_d, g_ch_d)
        self.conv1_2 = DepthConvBlockRK(g_ch_d, g_ch_d)
        self.conv2 = nn.Conv2d(g_ch_d, g_ch_d, 1)

    def forward(self, x, ctx, quant_step):
        feature = self.up(x)
        feature = self.conv1_0(feature, ctx)
        feature = self.conv1_1(feature)
        feature = self.conv1_2(feature)
        feature = self.conv2(feature)
        feature = feature * quant_step
        return feature


class ReconGenerationRK(nn.Module):
    """recon_generation: 4 DCB(256->320) + head(320->192). NO pixel_shuffle (done in C)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlockRK(g_ch_d, g_ch_recon),
            DepthConvBlockRK(g_ch_recon, g_ch_recon),
            DepthConvBlockRK(g_ch_recon, g_ch_recon),
            DepthConvBlockRK(g_ch_recon, g_ch_recon),
        )
        self.head = nn.Conv2d(g_ch_recon, g_ch_src_d, 1)

    def forward(self, x, quant_step):
        out = self.conv(x)
        out = out * quant_step
        return self.head(out)
