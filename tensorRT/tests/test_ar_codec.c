// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Integration test for the production AR codec module (dcvc_ar_codec).
// Verifies that dcvc_ar_codec_encode → dcvc_ar_codec_decode reproduces y_hat
// bit-exactly, exercising the full production API (not raw CUDA in test code).

#include "dcvc_ar_codec.h"
#include "dcvc_rt.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Minimal npy reader for test data ---- */
typedef struct { int dims[4]; size_t elems; uint16_t* fp16; float* fp32; } Npy;
static int read_npy(const char* path, Npy* o) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char m[6]; fread(m, 1, 6, f);
    if (memcmp(m, "\x93NUMPY", 6)) { fclose(f); return -1; }
    uint8_t mj, mn; fread(&mj, 1, 1, f); fread(&mn, 1, 1, f);
    uint32_t hl = 0;
    if (mj == 1) { uint16_t h16; fread(&h16, 2, 1, f); hl = h16; }
    else fread(&hl, 4, 1, f);
    char hdr[512];
    if (hl >= sizeof(hdr)) { fclose(f); return -1; }
    fread(hdr, 1, hl, f); hdr[hl] = 0;
    o->dims[0] = o->dims[1] = o->dims[2] = o->dims[3] = 1;
    int nd = 0;
    char* sp = strstr(hdr, "shape");
    if (sp) { sp = strchr(sp, '('); if (sp) { sp++;
        while (*sp && *sp != ')' && nd < 4) {
            if (*sp >= '0' && *sp <= '9') { o->dims[nd++] = atoi(sp);
                while (*sp >= '0' && *sp <= '9') sp++; }
            sp++;
        }
    } }
    while (nd < 4) o->dims[nd++] = 1;
    o->elems = 1; for (int i = 0; i < 4; i++) o->elems *= (size_t)o->dims[i];
    o->fp16 = o->fp32 = NULL;
    if (strstr(hdr, "f2")) { o->fp16 = malloc(o->elems * 2);
        size_t g = fread(o->fp16, 2, o->elems, f); fclose(f);
        return g == o->elems ? 0 : -1; }
    o->fp32 = malloc(o->elems * 4);
    size_t g = fread(o->fp32, 4, o->elems, f); fclose(f);
    return g == o->elems ? 0 : -1;
}

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
    const char* asset_dir  = argc > 1 ? argv[1] : "tensorRT/assets";
    const char* plugin_dir = argc > 2 ? argv[2] : "tensorRT/build/plugin_demo";
    char path[600];

    /* ---- Create codec ---- */
    DcvcRtStatus st;
    DcvcArCodec* codec = dcvc_ar_codec_create(asset_dir, plugin_dir, 4, &st);
    if (!codec) {
        fprintf(stderr, "codec create failed: %s\n", dcvc_rt_status_string(st));
        return 1;
    }
    printf("AR codec created (4 passes, intra)\n");

    /* ---- Load golden params_fusion ---- */
    Npy pf;
    snprintf(path, sizeof(path), "%s/decode/golden_params_fusion_ec0.npy", asset_dir);
    if (read_npy(path, &pf) != 0) { fprintf(stderr, "no params_fusion\n"); return 1; }
    int H = pf.dims[2], W = pf.dims[3];
    printf("params_fusion: [%d,%d,%d,%d]  H=%d W=%d\n",
           pf.dims[0], pf.dims[1], pf.dims[2], pf.dims[3], H, W);

    /* Upload params_fusion to device */
    void* d_pf; cudaMalloc(&d_pf, pf.elems * 2);
    if (pf.fp16) cudaMemcpy(d_pf, pf.fp16, pf.elems * 2, cudaMemcpyHostToDevice);
    else {
        uint16_t* tmp = malloc(pf.elems * 2);
        for (size_t i = 0; i < pf.elems; i++) tmp[i] = f32h(pf.fp32[i]);
        cudaMemcpy(d_pf, tmp, pf.elems * 2, cudaMemcpyHostToDevice);
        free(tmp);
    }

    /* ---- Synthesize deterministic y latent [1, 256, H, W] ---- */
    int N_CH = 256, hw = H * W;
    uint16_t* y_h = malloc(N_CH * hw * 2);
    for (int c = 0; c < N_CH; c++)
        for (int i = 0; i < hw; i++)
            y_h[c * hw + i] = f32h(sinf(c * 0.137f + i * 0.071f) * 1.5f);
    void* d_y; cudaMalloc(&d_y, N_CH * hw * 2);
    cudaMemcpy(d_y, y_h, N_CH * hw * 2, cudaMemcpyHostToDevice);

    /* ---- Encode ---- */
    void* d_yhat_enc; cudaMalloc(&d_yhat_enc, N_CH * hw * 2);
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
    void* d_yhat_dec; cudaMalloc(&d_yhat_dec, N_CH * hw * 2);
    printf("Decoding...\n");
    st = dcvc_ar_codec_decode(codec, d_pf, H, W, stream, stream_size, d_yhat_dec);
    if (st != DCVC_RT_OK) {
        fprintf(stderr, "decode failed: %s\n", dcvc_rt_status_string(st));
        return 1;
    }

    /* ---- Compare ---- */
    uint16_t* enc_h = malloc(N_CH * hw * 2);
    uint16_t* dec_h = malloc(N_CH * hw * 2);
    cudaMemcpy(enc_h, d_yhat_enc, N_CH * hw * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(dec_h, d_yhat_dec, N_CH * hw * 2, cudaMemcpyDeviceToHost);

    int exact = 0, n = N_CH * hw;
    float mx = 0;
    for (int i = 0; i < n; i++) {
        if (enc_h[i] == dec_h[i]) exact++;
        float d = fabsf(h2f(enc_h[i]) - h2f(dec_h[i]));
        if (d > mx) mx = d;
    }
    printf("\n=== RESULT ===\n");
    printf("y_hat enc-vs-dec: BIT-EXACT=%d/%d (%.1f%%)  max_abs=%.3e\n",
           exact, n, 100.0 * exact / n, mx);

    if (exact == n) {
        printf("\n*** AR CODEC PRODUCTION MODULE: PASS (100%% round-trip) ***\n");
    } else {
        printf("\n*** AR CODEC PRODUCTION MODULE: FAIL ***\n");
    }

    /* ---- Cleanup ---- */
    free(stream); free(y_h); free(enc_h); free(dec_h);
    if (pf.fp16) free(pf.fp16); if (pf.fp32) free(pf.fp32);
    cudaFree(d_pf); cudaFree(d_y); cudaFree(d_yhat_enc); cudaFree(d_yhat_dec);
    dcvc_ar_codec_destroy(codec);
    return exact == n ? 0 : 1;
}
