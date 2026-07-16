// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
//
// End-to-end C runtime test: load a TRT engine, run inference, compare to
// a reference .npy tensor. Verifies the C trt_runner → trt_engine.cpp path.
//
// Usage: test_trt_engine <engine_id> <input.npy> [quant_step.npy] <ref_output.npy>
//   engine_id: 0=intra_analysis 1=intra_hyper_enc 4=intra_synthesis

#include "dcvc_trt_runner.h"
#include "dcvc_rt.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Minimal NPY reader for FP16 [N,C,H,W] tensors */
typedef struct { int n, c, h, w; size_t elems; uint16_t* data; } NpyArray;

static int read_npy(const char* path, NpyArray* out)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }

    char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) {
        fclose(f); return -1;
    }
    uint8_t major, minor;
    fread(&major, 1, 1, f); fread(&minor, 1, 1, f);

    /* header length: v1=uint16, v2=uint32 */
    uint32_t hlen = 0;
    if (major == 1) { uint16_t h16; fread(&h16, 2, 1, f); hlen = h16; }
    else { fread(&hlen, 4, 1, f); }

    char header[256];
    if (hlen >= sizeof(header)) { fclose(f); return -1; }
    fread(header, 1, hlen, f);
    header[hlen] = 0;

    /* Parse shape from header: "'descr': '<f2', 'fortran_order': False, 'shape': (1, 3, 256, 256)" */
    /* Very simple parser: find shape tuple */
    int dims[4] = {1,1,1,1}; int nd = 0;
    char* sp = strstr(header, "shape");
    if (sp) {
        sp = strchr(sp, '(');
        if (sp) {
            sp++;
            while (*sp && *sp != ')' && nd < 4) {
                if (*sp >= '0' && *sp <= '9') {
                    dims[nd++] = atoi(sp);
                    while (*sp >= '0' && *sp <= '9') sp++;
                }
                sp++;
            }
        }
    }
    if (nd == 0) { fclose(f); return -1; }
    /* Expand to 4D */
    while (nd < 4) dims[nd++] = 1;
    /* Realign */
    out->n = dims[0]; out->c = dims[1]; out->h = dims[2]; out->w = dims[3];
    out->elems = (size_t)out->n * out->c * out->h * out->w;

    out->data = (uint16_t*)malloc(out->elems * 2);
    if (!out->data) { fclose(f); return -1; }
    size_t got = fread(out->data, 2, out->elems, f);
    fclose(f);
    return got == out->elems ? 0 : -1;
}

static float fp16_to_f32(uint16_t h)
{
    uint32_t sign = (h >> 15) & 1;
    uint32_t expo = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (expo == 0) {
        if (mant == 0) { f = sign << 31; }
        else { /* subnormal */ int e = -1; while (!(mant & 0x400)) { mant <<= 1; e--; } mant &= 0x3ff; expo = 127 + e - 14; f = (sign << 31) | (expo << 23) | (mant << 13); }
    } else if (expo == 31) {
        f = (sign << 31) | (0xff << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((expo + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <engine_id> <input.npy> [quant.npy] <ref.npy>\n", argv[0]);
        return 1;
    }

    int engine_id = atoi(argv[1]);
    const char* input_path = argv[2];
    const char* ref_path = (argc == 4) ? argv[3] : argv[4];
    const char* quant_path = (argc == 4) ? NULL : argv[3];

    DcvcRtStatus st;
    DcvcTrtRunner* runner = dcvc_trt_runner_create("native/assets", 0, &st);
    if (!runner || st != DCVC_RT_OK) {
        fprintf(stderr, "Failed to create runner: %s\n", dcvc_rt_status_string(st));
        return 1;
    }

    printf("Engine %d present: %s\n", engine_id,
           dcvc_trt_runner_has_engine(runner, engine_id) ? "YES" : "NO");
    if (!dcvc_trt_runner_has_engine(runner, engine_id)) {
        fprintf(stderr, "Engine not found\n");
        return 1;
    }

    /* Load inputs */
    NpyArray in_npy;
    if (read_npy(input_path, &in_npy) != 0) { fprintf(stderr, "Failed to read input\n"); return 1; }
    printf("Input: [%d,%d,%d,%d] = %zu FP16 elems\n", in_npy.n, in_npy.c, in_npy.h, in_npy.w, in_npy.elems);

    /* Allocate CUDA device memory for inputs */
    DcvcTensorView inputs[2];
    int n_in = 1;
    void* d_in;
    cudaMalloc(&d_in, in_npy.elems * 2);
    cudaMemcpy(d_in, in_npy.data, in_npy.elems * 2, cudaMemcpyHostToDevice);
    inputs[0].device_ptr = d_in;
    inputs[0].host_ptr = in_npy.data;
    inputs[0].n = in_npy.n; inputs[0].c = in_npy.c;
    inputs[0].h = in_npy.h; inputs[0].w = in_npy.w;
    inputs[0].is_fp16 = 1;
    inputs[0].bytes = in_npy.elems * 2;

    if (quant_path) {
        NpyArray q_npy;
        if (read_npy(quant_path, &q_npy) != 0) { fprintf(stderr, "Failed to read quant\n"); return 1; }
        void* d_q;
        cudaMalloc(&d_q, q_npy.elems * 2);
        cudaMemcpy(d_q, q_npy.data, q_npy.elems * 2, cudaMemcpyHostToDevice);
        inputs[1].device_ptr = d_q;
        inputs[1].host_ptr = q_npy.data;
        inputs[1].n = q_npy.n; inputs[1].c = q_npy.c;
        inputs[1].h = q_npy.h; inputs[1].w = q_npy.w;
        inputs[1].is_fp16 = 1;
        inputs[1].bytes = q_npy.elems * 2;
        n_in = 2;
    }

    /* Output: allocate large buffer (max possible output) */
    /* For analysis: [1,256,16,16]; for synthesis: [1,3,256,256]; for hyper: [1,128,4,4] */
    size_t max_out = 3 * 256 * 256; /* worst case */
    void* d_out;
    cudaMalloc(&d_out, max_out * 2);

    DcvcTensorView output;
    memset(&output, 0, sizeof(output));
    output.device_ptr = d_out;
    output.n = 1; output.c = 0; output.h = 0; output.w = 0;
    output.is_fp16 = 1;
    output.bytes = max_out * 2;

    /* Execute */
    printf("Running engine...\n");
    st = dcvc_trt_runner_execute(runner, engine_id, inputs, n_in, &output, 1, NULL);
    cudaDeviceSynchronize();

    if (st != DCVC_RT_OK) {
        fprintf(stderr, "Execute failed: %s\n", dcvc_rt_status_string(st));
        return 1;
    }

    printf("Output: [%d,%d,%d,%d]\n", output.n, output.c, output.h, output.w);
    size_t out_elems = (size_t)output.n * output.c * output.h * output.w;
    uint16_t* out_host = (uint16_t*)malloc(out_elems * 2);
    cudaMemcpy(out_host, d_out, out_elems * 2, cudaMemcpyDeviceToHost);

    /* Compare to reference */
    NpyArray ref_npy;
    if (read_npy(ref_path, &ref_npy) != 0) {
        fprintf(stderr, "Failed to read ref, printing output stats only\n");
        float mn = 1e9f, mx = -1e9f;
        for (size_t i = 0; i < out_elems; i++) {
            float v = fp16_to_f32(out_host[i]);
            if (v < mn) mn = v; if (v > mx) mx = v;
        }
        printf("Output range: [%.4f, %.4f]\n", mn, mx);
    } else {
        size_t cmp = out_elems < ref_npy.elems ? out_elems : ref_npy.elems;
        float max_abs = 0, max_ref = 0;
        for (size_t i = 0; i < cmp; i++) {
            float diff = fabsf(fp16_to_f32(out_host[i]) - fp16_to_f32(ref_npy.data[i]));
            if (diff > max_abs) max_abs = diff;
            float r = fabsf(fp16_to_f32(ref_npy.data[i]));
            if (r > max_ref) max_ref = r;
        }
        float rel = max_abs / (max_ref + 1e-6f);
        printf("[RESULT] max_abs=%.4e rel_err=%.4f ref_max=%.4f\n", max_abs, rel, max_ref);
        free(ref_npy.data);
    }

    /* Cleanup */
    free(out_host); free(in_npy.data);
    cudaFree(d_in); cudaFree(d_out);
    dcvc_trt_runner_destroy(runner);
    return 0;
}
