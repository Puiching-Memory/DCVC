// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Parity test for the 2× AR kernels (inter/P-frame path):
//   single_part_writing_2x, combine_for_reading_2x, restore_y_2x,
//   add_and_multiply, clamp_recip_quant
// Verifies the CUDA kernel math against a reference implementation.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t s = (x >> 31) & 1; int e = ((x >> 23) & 0xff) - 127 + 15; uint32_t m = (x >> 13) & 0x3ff;
    if (e <= 0) { m |= 0x400; while (e < 0) { m >>= 1; e++; } e = 0; }
    if (e >= 31) { e = 31; m = 0; }
    return (uint16_t)((s << 15) | (e << 10) | m);
}
static float h2f(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, f;
    if (e == 0) { if (m == 0) f = s << 31; else { int ex = -1; while (!(m & 0x400)) { m <<= 1; ex--; } m &= 0x3ff; e = 127 + ex - 14; f = (s << 31) | (e << 23) | (m << 13); } }
    else if (e == 31) f = (s << 31) | (0xff << 23) | (m << 13);
    else f = (s << 31) | ((e + 127 - 15) << 23) | (m << 13);
    float r; memcpy(&r, &f, 4); return r;
}

/* kernel fn types */
typedef void (*sp2x_fn)(const void*, void*, int, cudaStream_t);
typedef void (*cr2x_fn)(const void*, const void*, void*, int, cudaStream_t);
typedef void (*ry2x_fn)(const void*, const void*, const void*, void*, int, int, cudaStream_t);
typedef void (*am_fn)(void*, const void*, const void*, int, cudaStream_t);
typedef void (*crq_fn)(const void*, const void*, void*, void*, float, int, cudaStream_t);

static int fails = 0;
static void check(const char* name, const uint16_t* got, const uint16_t* ref, int n, float tol) {
    float mx = 0;
    for (int i = 0; i < n; i++) { float d = fabsf(h2f(got[i]) - h2f(ref[i])); if (d > mx) mx = d; }
    int ok = mx < tol;
    if (!ok) fails++;
    printf("  [%s] %-26s max_abs=%.3e\n", ok ? "PASS" : "FAIL", name, mx);
}

int main(int argc, char** argv) {
    const char* pd = argc > 1 ? argv[1] : "tensorRT/build/plugin_demo";
    char path[512];
    snprintf(path, sizeof(path), "%s/libdcvc_kernels.so", pd);
    void* kh = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!kh) { fprintf(stderr, "Cannot load kernels: %s\n", dlerror()); return 1; }

    sp2x_fn p_sp2x = (sp2x_fn)dlsym(kh, "dcvc_k_single_part_writing_2x");
    cr2x_fn p_cr2x = (cr2x_fn)dlsym(kh, "dcvc_k_combine_read_2x");
    ry2x_fn p_ry2x = (ry2x_fn)dlsym(kh, "dcvc_k_restore_y_2x");
    am_fn   p_am   = (am_fn)dlsym(kh, "dcvc_k_add_and_multiply");
    crq_fn  p_crq  = (crq_fn)dlsym(kh, "dcvc_k_clamp_recip_quant");
    printf("2x kernels: sp2x=%p cr2x=%p ry2x=%p am=%p crq=%p\n", p_sp2x, p_cr2x, p_ry2x, p_am, p_crq);

    /* deterministic test data */
    int C = 8, H = 4, W = 4, HW = H * W;
    int N = C * HW;      /* full channels */
    int halfN = N / 2;   /* C/2 channels */
    uint16_t *x_h = malloc(N * 2), *mask_h = malloc(N * 2), *means_h = malloc(N * 2);
    for (int i = 0; i < N; i++) {
        x_h[i]    = f32h(0.1f * (i % 13) - 0.6f);
        mask_h[i] = (i % 3 == 0) ? f32h(1.0f) : f32h(0.0f);
        means_h[i] = f32h(0.05f * (i % 7) - 0.15f);
    }

    /* ---- single_part_writing_2x: out[i] = x[i] + x[i+halfN] ---- */
    {
        void *d_x, *d_out;
        cudaMalloc(&d_x, N * 2); cudaMalloc(&d_out, halfN * 2);
        cudaMemcpy(d_x, x_h, N * 2, cudaMemcpyHostToDevice);
        p_sp2x(d_x, d_out, halfN, 0);
        uint16_t* got = malloc(halfN * 2); uint16_t* ref = malloc(halfN * 2);
        cudaMemcpy(got, d_out, halfN * 2, cudaMemcpyDeviceToHost);
        for (int i = 0; i < halfN; i++) ref[i] = f32h(h2f(x_h[i]) + h2f(x_h[i + halfN]));
        check("single_part_writing_2x", got, ref, halfN, 1e-3f);
        free(got); free(ref); cudaFree(d_x); cudaFree(d_out);
    }

    /* ---- combine_for_reading_2x: out[i] = x[i]*mask[i] + x[i+halfN]*mask[i+halfN] ---- */
    {
        void *d_x, *d_mask, *d_out;
        cudaMalloc(&d_x, N * 2); cudaMalloc(&d_mask, N * 2); cudaMalloc(&d_out, halfN * 2);
        cudaMemcpy(d_x, x_h, N * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mask, mask_h, N * 2, cudaMemcpyHostToDevice);
        p_cr2x(d_x, d_mask, d_out, halfN, 0);
        uint16_t* got = malloc(halfN * 2); uint16_t* ref = malloc(halfN * 2);
        cudaMemcpy(got, d_out, halfN * 2, cudaMemcpyDeviceToHost);
        for (int i = 0; i < halfN; i++)
            ref[i] = f32h(h2f(x_h[i]) * h2f(mask_h[i]) + h2f(x_h[i + halfN]) * h2f(mask_h[i + halfN]));
        check("combine_for_reading_2x", got, ref, halfN, 1e-3f);
        free(got); free(ref); cudaFree(d_x); cudaFree(d_mask); cudaFree(d_out);
    }

    /* ---- restore_y_2x: out[i] = (y[i % CyHW] + means[i]) * mask[i] ---- */
    {
        int CyHW = halfN;  /* y has C/2 channels */
        void *d_y, *d_means, *d_mask, *d_out;
        cudaMalloc(&d_y, halfN * 2); cudaMalloc(&d_means, N * 2);
        cudaMalloc(&d_mask, N * 2); cudaMalloc(&d_out, N * 2);
        /* y = first half of x */
        cudaMemcpy(d_y, x_h, halfN * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_means, means_h, N * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mask, mask_h, N * 2, cudaMemcpyHostToDevice);
        p_ry2x(d_y, d_means, d_mask, d_out, CyHW, N, 0);
        uint16_t* got = malloc(N * 2); uint16_t* ref = malloc(N * 2);
        cudaMemcpy(got, d_out, N * 2, cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++)
            ref[i] = f32h((h2f(x_h[i % CyHW]) + h2f(means_h[i])) * h2f(mask_h[i]));
        check("restore_y_2x", got, ref, N, 1e-3f);
        free(got); free(ref); cudaFree(d_y); cudaFree(d_means); cudaFree(d_mask); cudaFree(d_out);
    }

    /* ---- add_and_multiply: x0 = (x0 + x1) * q ---- */
    {
        void *d_x0, *d_x1, *d_q;
        cudaMalloc(&d_x0, N * 2); cudaMalloc(&d_x1, N * 2); cudaMalloc(&d_q, N * 2);
        cudaMemcpy(d_x0, x_h, N * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x1, means_h, N * 2, cudaMemcpyHostToDevice);
        uint16_t* q_h = malloc(N * 2);
        for (int i = 0; i < N; i++) q_h[i] = f32h(1.5f);
        cudaMemcpy(d_q, q_h, N * 2, cudaMemcpyHostToDevice);
        p_am(d_x0, d_x1, d_q, N, 0);
        uint16_t* got = malloc(N * 2); uint16_t* ref = malloc(N * 2);
        cudaMemcpy(got, d_x0, N * 2, cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++) ref[i] = f32h((h2f(x_h[i]) + h2f(means_h[i])) * 1.5f);
        check("add_and_multiply", got, ref, N, 1e-3f);
        free(got); free(ref); free(q_h); cudaFree(d_x0); cudaFree(d_x1); cudaFree(d_q);
    }

    /* ---- clamp_recip_quant: q_clamp=max(q,0.5), y_out=y/q_clamp ---- */
    {
        void *d_q, *d_y, *d_qc, *d_yo;
        cudaMalloc(&d_q, N * 2); cudaMalloc(&d_y, N * 2);
        cudaMalloc(&d_qc, N * 2); cudaMalloc(&d_yo, N * 2);
        uint16_t *q_h = malloc(N * 2), *y_h = malloc(N * 2);
        for (int i = 0; i < N; i++) {
            q_h[i] = f32h(0.3f + 0.01f * (i % 40));  /* some < 0.5 */
            y_h[i] = f32h(0.5f * (i % 5) - 1.0f);
        }
        cudaMemcpy(d_q, q_h, N * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y_h, N * 2, cudaMemcpyHostToDevice);
        p_crq(d_q, d_y, d_qc, d_yo, 0.5f, N, 0);
        uint16_t *got_yo = malloc(N * 2), *got_qc = malloc(N * 2), *ref_yo = malloc(N * 2), *ref_qc = malloc(N * 2);
        cudaMemcpy(got_yo, d_yo, N * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(got_qc, d_qc, N * 2, cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++) {
            float qd = h2f(q_h[i]); if (qd < 0.5f) qd = 0.5f;
            ref_qc[i] = f32h(qd);
            ref_yo[i] = f32h(h2f(y_h[i]) * (1.0f / qd));
        }
        check("clamp_recip_quant (y_out)", got_yo, ref_yo, N, 1e-3f);
        check("clamp_recip_quant (q_clamp)", got_qc, ref_qc, N, 1e-3f);
        free(got_yo); free(got_qc); free(ref_yo); free(ref_qc);
        free(q_h); free(y_h); cudaFree(d_q); cudaFree(d_y); cudaFree(d_qc); cudaFree(d_yo);
    }

    free(x_h); free(mask_h); free(means_h);
    if (fails == 0) {
        printf("\n*** 2x KERNELS: PASS (all parity checks) ***\n");
        return 0;
    }
    printf("\n*** 2x KERNELS: FAIL (%d checks failed) ***\n", fails);
    return 1;
}
