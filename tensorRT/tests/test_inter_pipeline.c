// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// End-to-end inter (P-frame) pipeline closed-loop test.
// Synthesizes an image [1,3,H,W] and a reference decoder feature [1,256,H/8,W/8],
// encodes a P-frame, then decodes and verifies x_hat and the decoder feature
// are reproduced bit-exactly (self-consistent closed loop).

#include "dcvc_inter_pipeline.h"
#include "dcvc_ar_codec.h"
#include "rans_c.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32h(float f)
{
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t s = (x >> 31) & 1;
    int e = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = (x >> 13) & 0x3ff;
    if (e <= 0) { m |= 0x400; while (e < 0) { m >>= 1; e++; } e = 0; }
    if (e >= 31) { e = 31; m = 0; }
    return (uint16_t)((s << 15) | (e << 10) | m);
}
static float h2f(uint16_t h)
{
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, f;
    if (e == 0) {
        if (m == 0) f = s << 31;
        else { int ex = -1; while (!(m & 0x400)) { m <<= 1; ex--; } m &= 0x3ff; e = 127 + ex - 14; f = (s << 31) | (e << 23) | (m << 13); }
    } else if (e == 31) {
        f = (s << 31) | (0xff << 23) | (m << 13);
    } else {
        f = (s << 31) | ((e + 127 - 15) << 23) | (m << 13);
    }
    float r; memcpy(&r, &f, 4); return r;
}

int main(int argc, char** argv)
{
    const char* asset_dir  = argc > 1 ? argv[1] : "tensorRT/assets";
    const char* plugin_dir = argc > 2 ? argv[2] : "tensorRT/build/plugin_demo";
    const int H = 256, W = 256;
    const int D = 256;
    int fH = H/8, fW = W/8, fHW = fH*fW;
    int qp = 32;

    DcvcRtStatus st;
    DcvcInterPipeline* pl = dcvc_inter_pipeline_create(asset_dir, plugin_dir, &st);
    if (!pl) { fprintf(stderr, "pipeline create failed: %s\n", dcvc_rt_status_string(st)); return 1; }
    DcvcArCodec* ar = dcvc_ar_codec_create(asset_dir, plugin_dir, 2, &st);
    if (!ar) { fprintf(stderr, "ar create failed: %s\n", dcvc_rt_status_string(st)); return 1; }
    DcvcRansEncoder* re = dcvc_rans_encoder_create();
    DcvcRansDecoder* rd = dcvc_rans_decoder_create();
    printf("inter pipeline + AR(2x) + rANS ready\n");

    /* synthesize image [1,3,H,W] in [0,1] */
    uint16_t* img_h = malloc(3 * H * W * 2);
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < H*W; i++)
            img_h[c*H*W+i] = f32h(0.5f + 0.4f*sinf(c*0.7f + i*0.0003f));
    void* d_img; cudaMalloc(&d_img, 3*H*W*2);
    cudaMemcpy(d_img, img_h, 3*H*W*2, cudaMemcpyHostToDevice);

    /* synthesize reference feature [1,256,fH,fW] */
    uint16_t* ref_h = malloc(D*fHW*2);
    for (int c = 0; c < D; c++)
        for (int i = 0; i < fHW; i++)
            ref_h[c*fHW+i] = f32h(sinf(c*0.11f + i*0.05f));
    void* d_ref; cudaMalloc(&d_ref, D*fHW*2);
    cudaMemcpy(d_ref, ref_h, D*fHW*2, cudaMemcpyHostToDevice);

    void* d_xhat_e; cudaMalloc(&d_xhat_e, 3*H*W*2);
    void* d_xhat_d; cudaMalloc(&d_xhat_d, 3*H*W*2);
    void* d_fdec_e; cudaMalloc(&d_fdec_e, D*fHW*2);
    void* d_fdec_d; cudaMalloc(&d_fdec_d, D*fHW*2);

    printf("Encoding P-frame...\n");
    uint8_t* stream = NULL; size_t stream_size = 0;
    st = dcvc_inter_encode(pl, ar, re, d_img, d_ref, NULL, H, W, qp,
                           &stream, &stream_size, d_xhat_e, d_fdec_e);
    if (st != DCVC_RT_OK) { fprintf(stderr, "encode failed: %s\n", dcvc_rt_status_string(st)); return 1; }
    printf("bitstream: %zu bytes\n", stream_size);

    printf("Decoding P-frame...\n");
    st = dcvc_inter_decode(pl, ar, rd, stream, stream_size, d_ref, NULL, H, W, qp, d_xhat_d, d_fdec_d);
    if (st != DCVC_RT_OK) { fprintf(stderr, "decode failed: %s\n", dcvc_rt_status_string(st)); return 1; }

    /* compare x_hat */
    uint16_t *xe = malloc(3*H*W*2), *xd = malloc(3*H*W*2);
    cudaMemcpy(xe, d_xhat_e, 3*H*W*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(xd, d_xhat_d, 3*H*W*2, cudaMemcpyDeviceToHost);
    int nimg = 3*H*W, exact_img = 0; float mx_img = 0;
    for (int i = 0; i < nimg; i++) {
        if (xe[i] == xd[i]) exact_img++;
        float d = fabsf(h2f(xe[i]) - h2f(xd[i]));
        if (d > mx_img) mx_img = d;
    }

    /* compare decoder feature (DPB) */
    uint16_t *fe = malloc(D*fHW*2), *fd = malloc(D*fHW*2);
    cudaMemcpy(fe, d_fdec_e, D*fHW*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(fd, d_fdec_d, D*fHW*2, cudaMemcpyDeviceToHost);
    int nfeat = D*fHW, exact_feat = 0; float mx_feat = 0;
    for (int i = 0; i < nfeat; i++) {
        if (fe[i] == fd[i]) exact_feat++;
        float d = fabsf(h2f(fe[i]) - h2f(fd[i]));
        if (d > mx_feat) mx_feat = d;
    }

    printf("\n=== RESULT ===\n");
    printf("x_hat    enc-vs-dec: BIT-EXACT=%d/%d (%.1f%%)  max_abs=%.3e\n",
           exact_img, nimg, 100.0*exact_img/nimg, mx_img);
    printf("dec_feat enc-vs-dec: BIT-EXACT=%d/%d (%.1f%%)  max_abs=%.3e\n",
           exact_feat, nfeat, 100.0*exact_feat/nfeat, mx_feat);

    int ok = (exact_img == nimg) && (exact_feat == nfeat);
    printf("\n*** INTER PIPELINE: %s ***\n", ok ? "PASS (closed loop)" : "FAIL");

    free(stream); free(img_h); free(ref_h); free(xe); free(xd); free(fe); free(fd);
    cudaFree(d_img); cudaFree(d_ref); cudaFree(d_xhat_e); cudaFree(d_xhat_d);
    cudaFree(d_fdec_e); cudaFree(d_fdec_d);
    dcvc_rans_encoder_destroy(re); dcvc_rans_decoder_destroy(rd);
    dcvc_ar_codec_destroy(ar); dcvc_inter_pipeline_destroy(pl);
    return ok ? 0 : 1;
}
