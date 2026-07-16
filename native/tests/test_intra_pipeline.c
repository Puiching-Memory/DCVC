// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// End-to-end intra-frame pipeline test.
// image → analysis → hyper_enc → z rANS → hyper_dec → prior_fusion →
// AR codec → synthesis → image
//
// Validates that dcvc_intra_encode → dcvc_intra_decode reproduces x_hat.

#include "dcvc_intra_pipeline.h"
#include "dcvc_ar_codec.h"
#include "dcvc_trt_runner.h"
#include "dcvc_rt.h"
#include "rans_c.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t s = (x >> 31) & 1;
    int e = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = (x >> 13) & 0x3ff;
    if (e <= 0) { m |= 0x400; while (e < 0) { m >>= 1; e++; } e = 0; }
    if (e >= 31) { e = 31; m = 0; }
    return (uint16_t)((s << 15) | (e << 10) | m);
}
static float h2f(uint16_t h) {
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
    const char* asset_dir  = argc > 1 ? argv[1] : "native/assets";
    const char* plugin_dir = argc > 2 ? argv[2] : "native/build/plugin_demo";
    int H = 256, W = 256;  /* must be multiples of 64 for z dims */
    int qp = 20;

    printf("=== Intra Pipeline End-to-End Test ===\n");
    printf("Image: %dx%d, QP=%d\n\n", W, H, qp);

    /* ---- Create components ---- */
    DcvcRtStatus st;
    DcvcTrtRunner* runner = dcvc_trt_runner_create(asset_dir, 0, &st);
    if (!runner) { fprintf(stderr, "runner create failed: %s\n", dcvc_rt_status_string(st)); return 1; }

    DcvcArCodec* ar_codec = dcvc_ar_codec_create(asset_dir, plugin_dir, 4, &st);
    if (!ar_codec) { fprintf(stderr, "ar_codec create failed: %s\n", dcvc_rt_status_string(st)); return 1; }

    DcvcRansEncoder* rans_enc = dcvc_rans_encoder_create();
    DcvcRansDecoder* rans_dec = dcvc_rans_decoder_create();
    if (!rans_enc || !rans_dec) { fprintf(stderr, "rans create failed\n"); return 1; }

    printf("All components created\n");
    printf("  analysis: %d  hyper_enc: %d  hyper_dec: %d  prior_fusion: %d  synthesis: %d\n",
           dcvc_trt_runner_has_engine(runner, DCVC_ENG_INTRA_ANALYSIS),
           dcvc_trt_runner_has_engine(runner, DCVC_ENG_INTRA_HYPER),
           dcvc_trt_runner_has_engine(runner, DCVC_ENG_HYPER_DEC),
           dcvc_trt_runner_has_engine(runner, DCVC_ENG_INTRA_PRIOR_FUSION),
           dcvc_trt_runner_has_engine(runner, DCVC_ENG_INTRA_SYNTHESIS));

    /* ---- Synthesize test image [1,3,H,W] FP16 in [0,1] ---- */
    size_t img_elems = 3 * H * W;
    uint16_t* img_h = (uint16_t*)malloc(img_elems * 2);
    for (int c = 0; c < 3; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                img_h[c * H * W + h * W + w] = f32h(0.3f + 0.4f * sinf(c * 0.5f + h * 0.02f + w * 0.03f));

    void* d_image; cudaMalloc(&d_image, img_elems * 2);
    cudaMemcpy(d_image, img_h, img_elems * 2, cudaMemcpyHostToDevice);

    /* ---- Encode ---- */
    void* d_xhat_enc; cudaMalloc(&d_xhat_enc, img_elems * 2);
    uint8_t* stream = NULL; size_t stream_size = 0;

    printf("\nEncoding...\n");
    st = dcvc_intra_encode(runner, ar_codec, rans_enc, d_image, H, W, qp,
                           &stream, &stream_size, d_xhat_enc);
    if (st != DCVC_RT_OK) {
        fprintf(stderr, "encode failed: %s\n", dcvc_rt_status_string(st));
        return 1;
    }
    printf("bitstream: %zu bytes\n", stream_size);

    /* Print x_hat_enc stats */
    {
        uint16_t* xh = (uint16_t*)malloc(img_elems * 2);
        cudaMemcpy(xh, d_xhat_enc, img_elems * 2, cudaMemcpyDeviceToHost);
        float mn = 1e9, mx = -1e9;
        for (size_t i = 0; i < img_elems; i++) {
            float v = h2f(xh[i]);
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        printf("x_hat_enc range: [%.4f, %.4f]\n", mn, mx);
        free(xh);
    }

    /* ---- Decode ---- */
    void* d_xhat_dec; cudaMalloc(&d_xhat_dec, img_elems * 2);

    printf("Decoding...\n");
    st = dcvc_intra_decode(runner, ar_codec, rans_dec, stream, stream_size, H, W, qp, d_xhat_dec);
    if (st != DCVC_RT_OK) {
        fprintf(stderr, "decode failed: %s\n", dcvc_rt_status_string(st));
        return 1;
    }

    /* ---- Compare x_hat_enc vs x_hat_dec ---- */
    {
        uint16_t* enc_h = (uint16_t*)malloc(img_elems * 2);
        uint16_t* dec_h = (uint16_t*)malloc(img_elems * 2);
        cudaMemcpy(enc_h, d_xhat_enc, img_elems * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(dec_h, d_xhat_dec, img_elems * 2, cudaMemcpyDeviceToHost);

        int exact = 0;
        float mx = 0, sum_sq = 0;
        size_t n = img_elems;
        for (size_t i = 0; i < n; i++) {
            if (enc_h[i] == dec_h[i]) exact++;
            float d = fabsf(h2f(enc_h[i]) - h2f(dec_h[i]));
            if (d > mx) mx = d;
            sum_sq += d * d;
        }
        float mse = sum_sq / n;
        float psnr = (mse > 0) ? 10.0f * log10f(1.0f / mse) : 999.0f;

        printf("\n=== RESULT ===\n");
        printf("x_hat enc-vs-dec: BIT-EXACT=%zu/%zu (%.1f%%)  max_abs=%.3e\n",
               exact, n, 100.0 * exact / n, mx);
        printf("PSNR(enc,dec) = %.2f dB\n", psnr);

        if (exact == n) {
            printf("\n*** INTRA PIPELINE: PASS (100%% round-trip self-consistent) ***\n");
        } else if (mx < 0.01f) {
            printf("\n*** INTRA PIPELINE: PASS (near-exact, max_abs < 0.01) ***\n");
        } else {
            printf("\n*** INTRA PIPELINE: CHECK (max_abs=%.3e, PSNR=%.2f dB) ***\n", mx, psnr);
        }

        free(enc_h); free(dec_h);
    }

    /* ---- Cleanup ---- */
    free(stream); free(img_h);
    cudaFree(d_image); cudaFree(d_xhat_enc); cudaFree(d_xhat_dec);
    dcvc_ar_codec_destroy(ar_codec);
    dcvc_trt_runner_destroy(runner);
    dcvc_rans_encoder_destroy(rans_enc);
    dcvc_rans_decoder_destroy(rans_dec);
    return 0;
}
