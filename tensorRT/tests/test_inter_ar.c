// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Round-trip test for the 2x (inter / P-frame) AR spatial-prior codec.
// Synthesizes a params tensor [1, 3*128, H, W] and a y latent [1, 128, H, W],
// then verifies dcvc_ar_codec_encode → dcvc_ar_codec_decode reproduces y_hat
// bit-exactly (self-consistent closed loop).

#include "dcvc_ar_codec.h"
#include "dcvc_rt.h"

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
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, mm = h & 0x3ff, f;
    if (e == 0) {
        if (mm == 0) f = s << 31;
        else { int ex = -1; while (!(mm & 0x400)) { mm <<= 1; ex--; } mm &= 0x3ff; e = 127 + ex - 14; f = (s << 31) | (e << 23) | (mm << 13); }
    } else if (e == 31) {
        f = (s << 31) | (0xff << 23) | (mm << 13);
    } else {
        f = (s << 31) | ((e + 127 - 15) << 23) | (mm << 13);
    }
    float r; memcpy(&r, &f, 4); return r;
}

int main(int argc, char** argv)
{
    const char* asset_dir  = argc > 1 ? argv[1] : "tensorRT/assets";
    const char* plugin_dir = argc > 2 ? argv[2] : "tensorRT/build/plugin_demo";

    const int NC = 128;            /* g_ch_y inter */
    const int H = 16, W = 16;      /* latent (y) resolution, even for 2x mask */
    const int hw = H * W;

    DcvcRtStatus st;
    DcvcArCodec* codec = dcvc_ar_codec_create(asset_dir, plugin_dir, 2, &st);
    if (!codec) {
        fprintf(stderr, "inter codec create failed: %s\n", dcvc_rt_status_string(st));
        return 1;
    }
    printf("AR codec created (2 passes, inter)\n");

    /* ---- Synthesize params [1, 3*NC, H, W] = [q_dec | scales | means] ---- */
    const int PCH = 3 * NC;
    uint16_t* pf_h = malloc((size_t)PCH * hw * 2);
    for (int c = 0; c < PCH; c++)
        for (int i = 0; i < hw; i++) {
            float v;
            if (c < NC)        v = 0.6f + 1.4f * (0.5f + 0.5f * sinf(c * 0.31f + i * 0.17f)); /* q_dec > 0.5 */
            else if (c < 2*NC) v = 0.3f + 2.0f * (0.5f + 0.5f * cosf(c * 0.23f + i * 0.11f)); /* scales */
            else               v = 2.5f * sinf(c * 0.19f + i * 0.05f);                        /* means */
            pf_h[c * hw + i] = f32h(v);
        }
    void* d_pf; cudaMalloc(&d_pf, (size_t)PCH * hw * 2);
    cudaMemcpy(d_pf, pf_h, (size_t)PCH * hw * 2, cudaMemcpyHostToDevice);

    /* ---- Synthesize y latent [1, NC, H, W] ---- */
    uint16_t* y_h = malloc((size_t)NC * hw * 2);
    for (int c = 0; c < NC; c++)
        for (int i = 0; i < hw; i++)
            y_h[c * hw + i] = f32h(1.8f * sinf(c * 0.137f + i * 0.071f));
    void* d_y; cudaMalloc(&d_y, (size_t)NC * hw * 2);
    cudaMemcpy(d_y, y_h, (size_t)NC * hw * 2, cudaMemcpyHostToDevice);

    /* ---- Encode ---- */
    void* d_yhat_enc; cudaMalloc(&d_yhat_enc, (size_t)NC * hw * 2);
    uint8_t* stream = NULL;
    size_t stream_size = 0;

    printf("\nEncoding...\n");
    st = dcvc_ar_codec_encode(codec, d_y, d_pf, H, W, &stream, &stream_size, d_yhat_enc);
    if (st != DCVC_RT_OK) {
        fprintf(stderr, "encode failed: %s\n", dcvc_rt_status_string(st));
        return 1;
    }
    printf("bitstream: %zu bytes\n", stream_size);

    /* ---- Decode ---- */
    void* d_yhat_dec; cudaMalloc(&d_yhat_dec, (size_t)NC * hw * 2);
    printf("Decoding...\n");
    st = dcvc_ar_codec_decode(codec, d_pf, H, W, stream, stream_size, d_yhat_dec);
    if (st != DCVC_RT_OK) {
        fprintf(stderr, "decode failed: %s\n", dcvc_rt_status_string(st));
        return 1;
    }

    /* ---- Compare ---- */
    uint16_t* enc_h = malloc((size_t)NC * hw * 2);
    uint16_t* dec_h = malloc((size_t)NC * hw * 2);
    cudaMemcpy(enc_h, d_yhat_enc, (size_t)NC * hw * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(dec_h, d_yhat_dec, (size_t)NC * hw * 2, cudaMemcpyDeviceToHost);

    int exact = 0, n = NC * hw;
    float mx = 0;
    for (int i = 0; i < n; i++) {
        if (enc_h[i] == dec_h[i]) exact++;
        float d = fabsf(h2f(enc_h[i]) - h2f(dec_h[i]));
        if (d > mx) mx = d;
    }
    printf("\n=== RESULT ===\n");
    printf("y_hat enc-vs-dec: BIT-EXACT=%d/%d (%.1f%%)  max_abs=%.3e\n",
           exact, n, 100.0 * exact / n, mx);

    if (exact == n)
        printf("\n*** INTER AR CODEC (2x): PASS (100%% round-trip) ***\n");
    else
        printf("\n*** INTER AR CODEC (2x): FAIL ***\n");

    free(stream); free(y_h); free(pf_h); free(enc_h); free(dec_h);
    cudaFree(d_pf); cudaFree(d_y); cudaFree(d_yhat_enc); cudaFree(d_yhat_dec);
    dcvc_ar_codec_destroy(codec);
    return exact == n ? 0 : 1;
}
