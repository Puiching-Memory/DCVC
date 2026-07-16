// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Production AR spatial-prior codec — productized from test_closed_loop.c.
// Manages TRT engines, CDF tables, CUDA kernels, and rANS encoder/decoder.

#include "dcvc_ar_codec.h"
#include "dcvc_rt_internal.h"
#include "rans_c.h"
#include "trt_engine.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- FP16 conversion helpers (host-side) ---- */
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

/* ---- Minimal .npy reader (int32 and fp16 only) ---- */
typedef struct { int dims[4]; size_t elems; int32_t* i32; uint16_t* fp16; float* fp32; } ArNpy;
static int ar_read_npy(const char* path, ArNpy* o)
{
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char magic[6]; fread(magic, 1, 6, f);
    if (memcmp(magic, "\x93NUMPY", 6)) { fclose(f); return -1; }
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
    if (sp) { sp = strchr(sp, '('); if (sp) { sp++; while (*sp && *sp != ')' && nd < 4) {
        if (*sp >= '0' && *sp <= '9') { o->dims[nd++] = atoi(sp); while (*sp >= '0' && *sp <= '9') sp++; }
        sp++; } } }
    while (nd < 4) o->dims[nd++] = 1;
    o->elems = 1; for (int i = 0; i < 4; i++) o->elems *= (size_t)o->dims[i];
    o->i32 = o->fp16 = o->fp32 = NULL;
    if (strstr(hdr, "i4")) { o->i32 = malloc(o->elems * 4); size_t g = fread(o->i32, 4, o->elems, f); fclose(f); return g == o->elems ? 0 : -1; }
    if (strstr(hdr, "f2")) { o->fp16 = malloc(o->elems * 2); size_t g = fread(o->fp16, 2, o->elems, f); fclose(f); return g == o->elems ? 0 : -1; }
    o->fp32 = malloc(o->elems * 4); size_t g = fread(o->fp32, 4, o->elems, f); fclose(f); return g == o->elems ? 0 : -1;
}
static void ar_npy_free(ArNpy* o) { free(o->i32); free(o->fp16); free(o->fp32); o->i32 = o->fp16 = o->fp32 = NULL; }

/* ---- CUDA kernel function pointer types ---- */
typedef void (*fn_build_idx_dec)(const void*, uint8_t*, float, float, float, float, int, cudaStream_t);
typedef void (*fn_build_idx_enc)(const void*, const void*, int16_t*, float, float, float, float, int, cudaStream_t);
typedef void (*fn_restore_y4x)(const void*, const void*, const void*, void*, int, int, cudaStream_t);
typedef void (*fn_sp4x)(const void*, void*, int, cudaStream_t);
typedef void (*fn_mul)(const void*, const void*, void*, int, cudaStream_t);
typedef void (*fn_pmask_yq)(const void*, const void*, const void*, const void*, void*, float, int, cudaStream_t);
/* 2x-specific kernels (inter/P-frame) */
typedef void (*fn_sp2x)(const void*, void*, int, cudaStream_t);
typedef void (*fn_restore_y2x)(const void*, const void*, const void*, void*, int, int, cudaStream_t);
typedef void (*fn_add_mul)(void*, const void*, const void*, int, cudaStream_t);
typedef void (*fn_crq)(const void*, const void*, void*, void*, float, int, cudaStream_t);

/* ---- Persistent workspace ---- */
typedef struct {
    int hw;      /* H*W */
    int n_ch;    /* 256 or 128 */
    int r_ch;    /* n_ch / 4 for 4x, n_ch / 2 for 2x */
    int p_ch;    /* params channels: 514 for 4x, 384 for 2x */
    void *d_qenc, *d_qdec, *d_scales, *d_means, *d_common;
    void *d_smask, *d_sr, *d_yq_full, *d_yq_w, *d_packed, *d_indexes;
    void *d_yhat_step, *d_cat, *d_sp_out, *d_yhat;
} ArWorkspace;

struct DcvcArCodec {
    char asset_dir[512];
    char plugin_dir[512];
    int passes;        /* 4 intra, 2 inter */
    int n_ch;          /* 256 intra, 128 inter */

    /* CUDA kernels (dynamically loaded) */
    void* kernel_so;
    fn_build_idx_dec  k_build_idx_dec;
    fn_build_idx_enc  k_build_idx_enc;
    fn_restore_y4x    k_restore_y4x;
    fn_sp4x           k_sp4x;
    fn_mul            k_mul;
    fn_pmask_yq       k_pmask_yq;
    /* 2x-specific */
    fn_sp2x           k_sp2x;
    fn_restore_y2x    k_restore_y2x;
    fn_add_mul        k_add_mul;
    fn_crq            k_crq;

    /* TRT engines */
    DcvcTrtEngine* eng_reduction;
    DcvcTrtEngine* eng_adaptor[4]; /* [0] unused, [1..3] for 4-pass; [1] for 2-pass */
    DcvcTrtEngine* eng_spatial_prior;

    /* CDF data (owned) */
    ArNpy gcdf, glen, goff;   /* gaussian */
    int g_cdf_idx;            /* CDF group index returned by rANS add_cdf */

    /* rANS instances (owned, one for encode, one for decode) */
    DcvcRansEncoder* rans_enc;
    DcvcRansDecoder* rans_dec;

    /* Workspace (reallocated when dimensions change) */
    ArWorkspace ws;

    /* Entropy constants */
    float scale_min, scale_max, log_scale_min, log_step_recip;
};

/* ---- Workspace management ---- */
static void ws_free(ArWorkspace* ws)
{
    if (!ws->hw) return;
    cudaFree(ws->d_qenc);     cudaFree(ws->d_qdec);
    cudaFree(ws->d_scales);   cudaFree(ws->d_means);
    cudaFree(ws->d_common);   cudaFree(ws->d_smask);
    cudaFree(ws->d_sr);       cudaFree(ws->d_yq_full);
    cudaFree(ws->d_yq_w);     cudaFree(ws->d_packed);
    cudaFree(ws->d_indexes);  cudaFree(ws->d_yhat_step);
    cudaFree(ws->d_cat);      cudaFree(ws->d_sp_out);
    cudaFree(ws->d_yhat);
    memset(ws, 0, sizeof(*ws));
}

static DcvcRtStatus ws_ensure(DcvcArCodec* c, int H, int W)
{
    int hw = H * W;
    if (c->ws.hw == hw && c->ws.n_ch == c->n_ch) return DCVC_RT_OK;
    ws_free(&c->ws);

    int nc = c->n_ch, r;
    if (c->passes == 4) r = nc / 4;
    else                r = nc / 2;
    int pch = (c->passes == 4) ? (2 * nc + 2) : (3 * nc);
    ArWorkspace* ws = &c->ws;
    ws->hw = hw; ws->n_ch = nc; ws->r_ch = r; ws->p_ch = pch;

    cudaMalloc(&ws->d_qenc,     hw * 2);
    /* q_dec is per-pixel [H*W] for 4x intra but per-channel-per-pixel [n_ch,H*W] for 2x inter */
    cudaMalloc(&ws->d_qdec,     nc * hw * 2);
    cudaMalloc(&ws->d_scales,   nc * hw * 2);
    cudaMalloc(&ws->d_means,    nc * hw * 2);
    cudaMalloc(&ws->d_common,   nc * hw * 2);
    cudaMalloc(&ws->d_smask,    nc * hw * 2);
    cudaMalloc(&ws->d_sr,       r * hw * 2);
    cudaMalloc(&ws->d_yq_full,  nc * hw * 2);
    cudaMalloc(&ws->d_yq_w,     r * hw * 2);
    cudaMalloc(&ws->d_packed,   r * hw * 2);
    cudaMalloc(&ws->d_indexes,  r * hw);
    cudaMalloc(&ws->d_yhat_step, nc * hw * 2);
    /* d_cat: 4x uses [y_hat | common] = 2*nc; 2x uses [y_hat | params] = 4*nc */
    cudaMalloc(&ws->d_cat,      4 * nc * hw * 2);
    cudaMalloc(&ws->d_sp_out,   2 * nc * hw * 2);
    cudaMalloc(&ws->d_yhat,     nc * hw * 2);

    /* Check all allocations succeeded */
    void* ptrs[] = {ws->d_qenc, ws->d_qdec, ws->d_scales, ws->d_means, ws->d_common,
                    ws->d_smask, ws->d_sr, ws->d_yq_full, ws->d_yq_w, ws->d_packed,
                    ws->d_indexes, ws->d_yhat_step, ws->d_cat, ws->d_sp_out, ws->d_yhat,
                    ws->d_yhat};
    for (int i = 0; i < 15; i++) {
        if (!ptrs[i]) { ws_free(ws); return DCVC_RT_ERR_OOM; }
    }
    return DCVC_RT_OK;
}

/* ---- Engine binding helper ---- */
static DcvcRtStatus bind_and_run(DcvcTrtEngine* eng, const char* in_name,
                                 const int32_t* in_dims, void* in_ptr,
                                 void* out_ptr)
{
    DcvcRtStatus st = dcvc_trt_engine_set_shape(eng, in_name, in_dims, 4);
    if (st != DCVC_RT_OK) return st;
    st = dcvc_trt_engine_set_addr(eng, in_name, in_ptr);
    if (st != DCVC_RT_OK) return st;
    int nio = dcvc_trt_engine_num_io(eng);
    const char* out_name = dcvc_trt_engine_tensor_name(eng, nio - 1);
    st = dcvc_trt_engine_set_addr(eng, out_name, out_ptr);
    if (st != DCVC_RT_OK) return st;
    return dcvc_trt_engine_execute(eng, NULL);
}

/* ---- separate_prior: extract q_enc, q_dec, scales, means from params_fusion ---- */
static void separate_prior(DcvcArCodec* c, const void* d_pf, int H, int W)
{
    int hw = H * W, nc = c->n_ch;
    size_t pf_bytes = 514 * hw * 2;
    uint16_t* pf = malloc(pf_bytes);
    cudaMemcpy((void*)pf, d_pf, pf_bytes, cudaMemcpyDeviceToHost);

    uint16_t* qe = malloc(hw * 2);
    uint16_t* qd = malloc(hw * 2);
    for (int i = 0; i < hw; i++) {
        float v0 = 1.0f / (1.0f + expf(-h2f(pf[0 * hw + i]))) * 1.5f + 0.5f;
        float v1 = 1.0f / (1.0f + expf(-h2f(pf[1 * hw + i]))) * 1.5f + 0.5f;
        qe[i] = f32h(v0);
        qd[i] = f32h(v1);
    }
    cudaMemcpy(c->ws.d_qenc, qe, hw * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(c->ws.d_qdec, qd, hw * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(c->ws.d_scales, pf + 2 * hw, nc * hw * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(c->ws.d_means, pf + (2 + nc) * hw, nc * hw * 2, cudaMemcpyHostToDevice);
    free(pf); free(qe); free(qd);
}

/* ---- Core AR loop: shared by encode and decode ---- */
/* Runs adaptor + spatial_prior engines for rounds 1..passes-1, updating scales/means */
static DcvcRtStatus run_spatial_prior(DcvcArCodec* c, void* d_yhat_sf,
                                       int H, int W, int round)
{
    int nc = c->n_ch, hw = H * W;
    ArWorkspace* ws = &c->ws;

    /* cat(y_hat_so_far, common) → [1, 2*N_CH, H, W] */
    cudaMemcpy(ws->d_cat, d_yhat_sf, nc * hw * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpy((char*)ws->d_cat + nc * hw * 2, ws->d_common, nc * hw * 2, cudaMemcpyDeviceToDevice);

    /* adaptor_{round} */
    int32_t cat_dims[4] = {1, 2 * nc, H, W};
    DcvcTrtEngine* eng_ad = c->eng_adaptor[round];
    dcvc_trt_engine_set_shape(eng_ad, "in0", cat_dims, 4);
    dcvc_trt_engine_set_addr(eng_ad, "in0", ws->d_cat);
    int nio_ad = dcvc_trt_engine_num_io(eng_ad);
    const char* ad_out = dcvc_trt_engine_tensor_name(eng_ad, nio_ad - 1);
    int32_t ad_dims[8]; int ad_nd;
    dcvc_trt_engine_get_shape(eng_ad, ad_out, ad_dims, &ad_nd, 8);
    dcvc_trt_engine_set_addr(eng_ad, ad_out, ws->d_sp_out);
    DcvcRtStatus st = dcvc_trt_engine_execute(eng_ad, NULL);
    if (st != DCVC_RT_OK) return st;

    /* y_spatial_prior on adaptor output */
    dcvc_trt_engine_set_shape(c->eng_spatial_prior, "in0", ad_dims, 4);
    dcvc_trt_engine_set_addr(c->eng_spatial_prior, "in0", ws->d_sp_out);
    int nio_sp = dcvc_trt_engine_num_io(c->eng_spatial_prior);
    const char* sp_out = dcvc_trt_engine_tensor_name(c->eng_spatial_prior, nio_sp - 1);
    dcvc_trt_engine_set_addr(c->eng_spatial_prior, sp_out, ws->d_cat);
    st = dcvc_trt_engine_execute(c->eng_spatial_prior, NULL);
    cudaDeviceSynchronize();
    return st;
}

/* Helper: element-wise add on host (sf += step) */
static void host_add_inplace(void* d_sf, void* d_step, int n)
{
    uint16_t* sf = malloc(n * 2);
    uint16_t* st = malloc(n * 2);
    cudaMemcpy(sf, d_sf, n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(st, d_step, n * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) sf[i] = f32h(h2f(sf[i]) + h2f(st[i]));
    cudaMemcpy(d_sf, sf, n * 2, cudaMemcpyHostToDevice);
    free(sf); free(st);
}

/* Helper: multiply by q_dec broadcast across channels, in-place on host buffer */
static void host_mul_qdec(void* d_buf, const uint16_t* qdec_h, int nc, int hw)
{
    uint16_t* buf = malloc(nc * hw * 2);
    cudaMemcpy(buf, d_buf, nc * hw * 2, cudaMemcpyDeviceToHost);
    for (int c = 0; c < nc; c++)
        for (int i = 0; i < hw; i++)
            buf[c * hw + i] = f32h(h2f(buf[c * hw + i]) * h2f(qdec_h[i]));
    cudaMemcpy(d_buf, buf, nc * hw * 2, cudaMemcpyHostToDevice);
    free(buf);
}

DcvcArCodec* dcvc_ar_codec_create(const char* asset_dir, const char* plugin_dir,
                                  int passes, DcvcRtStatus* st_out)
{
    if (!asset_dir || !plugin_dir || (passes != 4 && passes != 2)) {
        if (st_out) *st_out = DCVC_RT_ERR_INVALID_ARG;
        return NULL;
    }
    DcvcArCodec* c = (DcvcArCodec*)calloc(1, sizeof(*c));
    if (!c) { if (st_out) *st_out = DCVC_RT_ERR_OOM; return NULL; }
    strncpy(c->asset_dir, asset_dir, sizeof(c->asset_dir) - 1);
    strncpy(c->plugin_dir, plugin_dir, sizeof(c->plugin_dir) - 1);
    c->passes = passes;
    c->n_ch = (passes == 4) ? DCVC_RT_CH_Y_INTRA : DCVC_RT_CH_Y_INTER;

    /* entropy constants */
    c->scale_min = 0.11f;
    c->scale_max = 16.0f;
    c->log_scale_min = logf(c->scale_min);
    c->log_step_recip = 1.0f / ((logf(c->scale_max) - logf(c->scale_min)) / 127.0f);

    char path[640], decode_dir[600], eng_dir[600];
    snprintf(decode_dir, sizeof(decode_dir), "%s/decode", asset_dir);
    snprintf(eng_dir, sizeof(eng_dir), "%s/engines", asset_dir);
    DcvcRtStatus st = DCVC_RT_OK;

    /* Load CUDA kernels */
    snprintf(path, sizeof(path), "%s/libdcvc_kernels.so", plugin_dir);
    c->kernel_so = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!c->kernel_so) { st = DCVC_RT_ERR_IO; goto fail; }
    c->k_build_idx_dec = (fn_build_idx_dec)dlsym(c->kernel_so, "dcvc_k_build_index_dec");
    c->k_build_idx_enc = (fn_build_idx_enc)dlsym(c->kernel_so, "dcvc_k_build_index_enc");
    c->k_restore_y4x   = (fn_restore_y4x)dlsym(c->kernel_so, "dcvc_k_restore_y_4x");
    c->k_sp4x          = (fn_sp4x)dlsym(c->kernel_so, "dcvc_k_single_part_writing_4x");
    c->k_mul           = (fn_mul)dlsym(c->kernel_so, "dcvc_k_elem_mul");
    c->k_pmask_yq      = (fn_pmask_yq)dlsym(c->kernel_so, "dcvc_k_process_mask_yq");
    if (!c->k_build_idx_dec || !c->k_build_idx_enc || !c->k_restore_y4x ||
        !c->k_sp4x || !c->k_mul || !c->k_pmask_yq) { st = DCVC_RT_ERR_IO; goto fail; }

    /* Load 2x kernels (needed for inter) */
    c->k_sp2x        = (fn_sp2x)dlsym(c->kernel_so, "dcvc_k_single_part_writing_2x");
    c->k_restore_y2x = (fn_restore_y2x)dlsym(c->kernel_so, "dcvc_k_restore_y_2x");
    c->k_add_mul     = (fn_add_mul)dlsym(c->kernel_so, "dcvc_k_add_and_multiply");
    c->k_crq         = (fn_crq)dlsym(c->kernel_so, "dcvc_k_clamp_recip_quant");

    /* Load CDFs */
    snprintf(path, sizeof(path), "%s/gaussian_cdf.npy", decode_dir);
    if (ar_read_npy(path, &c->gcdf) != 0) { st = DCVC_RT_ERR_IO; goto fail; }
    snprintf(path, sizeof(path), "%s/gaussian_cdf_length.npy", decode_dir);
    if (ar_read_npy(path, &c->glen) != 0) { st = DCVC_RT_ERR_IO; goto fail; }
    snprintf(path, sizeof(path), "%s/gaussian_offset.npy", decode_dir);
    if (ar_read_npy(path, &c->goff) != 0) { st = DCVC_RT_ERR_IO; goto fail; }

    /* Load engines — different sets for 4x (intra) vs 2x (inter) */
    if (c->passes == 4) {
        snprintf(path, sizeof(path), "%s/y_spatial_prior_reduction.engine", eng_dir);
        c->eng_reduction = dcvc_trt_engine_load(path, plugin_dir, &st);
        if (!c->eng_reduction) goto fail;
        for (int i = 1; i <= 3; i++) {
            snprintf(path, sizeof(path), "%s/y_spatial_prior_adaptor_%d.engine", eng_dir, i);
            c->eng_adaptor[i] = dcvc_trt_engine_load(path, plugin_dir, &st);
            if (!c->eng_adaptor[i]) goto fail;
        }
        snprintf(path, sizeof(path), "%s/y_spatial_prior.engine", eng_dir);
        c->eng_spatial_prior = dcvc_trt_engine_load(path, plugin_dir, &st);
        if (!c->eng_spatial_prior) goto fail;
    } else {
        /* 2x (inter): only inter_spatial_prior engine, no reduction/adaptors */
        snprintf(path, sizeof(path), "%s/inter_spatial_prior.engine", eng_dir);
        c->eng_spatial_prior = dcvc_trt_engine_load(path, plugin_dir, &st);
        if (!c->eng_spatial_prior) goto fail;
    }

    /* rANS instances */
    c->rans_enc = dcvc_rans_encoder_create();
    c->rans_dec = dcvc_rans_decoder_create();
    if (!c->rans_enc || !c->rans_dec) { st = DCVC_RT_ERR_OOM; goto fail; }

    if (st_out) *st_out = DCVC_RT_OK;
    return c;

fail:
    dcvc_ar_codec_destroy(c);
    if (st_out) *st_out = st;
    return NULL;
}

void dcvc_ar_codec_destroy(DcvcArCodec* c)
{
    if (!c) return;
    ws_free(&c->ws);
    if (c->rans_enc) dcvc_rans_encoder_destroy(c->rans_enc);
    if (c->rans_dec) dcvc_rans_decoder_destroy(c->rans_dec);
    if (c->eng_reduction) dcvc_trt_engine_destroy(c->eng_reduction);
    for (int i = 1; i <= 3; i++)
        if (c->eng_adaptor[i]) dcvc_trt_engine_destroy(c->eng_adaptor[i]);
    if (c->eng_spatial_prior) dcvc_trt_engine_destroy(c->eng_spatial_prior);
    ar_npy_free(&c->gcdf); ar_npy_free(&c->glen); ar_npy_free(&c->goff);
    if (c->kernel_so) dlclose(c->kernel_so);
    free(c);
}

/* ---- Mask generation (deterministic 4× channel-spatial checkerboard) ---- */
/* Pattern from common_model.py get_mask_4x:
 *   round 0: quarters [m0, m1, m2, m3] → spatial_pos [0,1,2,3]
 *   round 1: quarters [m3, m2, m1, m0] → spatial_pos [3,2,1,0]
 *   round 2: quarters [m2, m3, m0, m1] → spatial_pos [2,3,0,1]
 *   round 3: quarters [m1, m0, m3, m2] → spatial_pos [1,0,3,2]
 * spatial_pos = (h % 2) * 2 + (w % 2)
 * mask[c, h, w] = 1 if spatial_pos == pattern[round][c / (C/4)] */
static const int k_mask_pattern[4][4] = {
    {0, 1, 2, 3},
    {3, 2, 1, 0},
    {2, 3, 0, 1},
    {1, 0, 3, 2},
};

static void* generate_mask(int round, int nc, int H, int W)
{
    int hw = H * W, q = nc / 4;
    uint16_t* mask_h = (uint16_t*)malloc(nc * hw * 2);
    for (int c = 0; c < nc; c++) {
        int quarter = c / q;
        if (quarter > 3) quarter = 3;
        int target_pos = k_mask_pattern[round][quarter];
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int sp = (h % 2) * 2 + (w % 2);
                mask_h[c * hw + h * W + w] = (sp == target_pos) ? f32h(1.0f) : f32h(0.0f);
            }
        }
    }
    void* d_mask;
    cudaMalloc(&d_mask, nc * hw * 2);
    cudaMemcpy(d_mask, mask_h, nc * hw * 2, cudaMemcpyHostToDevice);
    free(mask_h);
    return d_mask;
}



/* ---- 2x separate_prior for video (inter) ---- */
/* Encode: params.chunk(3) → q_dec, scales, means; y = y / max(q_dec, 0.5) */
static void separate_prior_video_enc(DcvcArCodec* c, const void* d_params, int H, int W)
{
    int hw = H * W, nc = c->n_ch;
    size_t pbytes = (size_t)3 * nc * hw * 2;
    uint16_t* ph = (uint16_t*)malloc(pbytes);
    cudaMemcpy(ph, d_params, pbytes, cudaMemcpyDeviceToHost);
    /* q_dec = first nc channels, scales = next nc, means = last nc */
    cudaMemcpy(c->ws.d_qdec, ph, nc * hw * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(c->ws.d_scales, ph + nc * hw, nc * hw * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(c->ws.d_means, ph + 2 * nc * hw, nc * hw * 2, cudaMemcpyHostToDevice);
    free(ph);
}

/* Decode: params.chunk(3) → quant_step(clamp≥0.5), scales, means */
static void separate_prior_video_dec(DcvcArCodec* c, const void* d_params, int H, int W)
{
    int hw = H * W, nc = c->n_ch;
    size_t pbytes = (size_t)3 * nc * hw * 2;
    uint16_t* ph = (uint16_t*)malloc(pbytes);
    cudaMemcpy(ph, d_params, pbytes, cudaMemcpyDeviceToHost);
    /* quant_step = first nc channels, clamp to ≥0.5 on host */
    uint16_t* qd = (uint16_t*)malloc(nc * hw * 2);
    for (int i = 0; i < nc * hw; i++) {
        float v = h2f(ph[i]); if (v < 0.5f) v = 0.5f;
        qd[i] = f32h(v);
    }
    cudaMemcpy(c->ws.d_qdec, qd, nc * hw * 2, cudaMemcpyHostToDevice);
    free(qd);
    cudaMemcpy(c->ws.d_scales, ph + nc * hw, nc * hw * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(c->ws.d_means, ph + 2 * nc * hw, nc * hw * 2, cudaMemcpyHostToDevice);
    free(ph);
}

/* ---- 2x mask generation (diagonal / anti-diagonal checkerboard) ---- */
/* m0 = ((1,0),(0,1)): spatial pos where h%2 == w%2 → diag */
/* m1 = ((0,1),(1,0)): spatial pos where h%2 != w%2 → anti-diag */
static void* generate_mask_2x(int round, int nc, int H, int W)
{
    int hw = H * W, half = nc / 2;
    uint16_t* mask_h = (uint16_t*)malloc(nc * hw * 2);
    for (int c = 0; c < nc; c++) {
        int half_idx = c / half;  /* 0 or 1 */
        /* round 0: first half gets m0(diag), second half gets m1(anti)
         * round 1: first half gets m1(anti), second half gets m0(diag) */
        int use_m0 = (round == 0) ? (half_idx == 0) : (half_idx == 1);
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int is_diag = ((h % 2) == (w % 2));
                int val = use_m0 ? is_diag : !is_diag;
                mask_h[c * hw + h * W + w] = val ? f32h(1.0f) : f32h(0.0f);
            }
        }
    }
    void* d_mask;
    cudaMalloc(&d_mask, nc * hw * 2);
    cudaMemcpy(d_mask, mask_h, nc * hw * 2, cudaMemcpyHostToDevice);
    free(mask_h);
    return d_mask;
}

/* ---- 2x spatial_prior: cat(y_hat_step, params) → engine → scales, means ---- */
static DcvcRtStatus run_spatial_prior_2x(DcvcArCodec* c, void* d_yhat_step,
                                          const void* d_params, int H, int W)
{
    int nc = c->n_ch, hw = H * W;
    ArWorkspace* ws = &c->ws;
    /* cat(y_hat_step, params) → [1, nc + 3*nc, H, W] = [1, 4*nc, H, W] */
    cudaMemcpy(ws->d_cat, d_yhat_step, nc * hw * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpy((char*)ws->d_cat + (size_t)nc * hw * 2, d_params,
               (size_t)3 * nc * hw * 2, cudaMemcpyDeviceToDevice);

    int32_t in_dims[4] = {1, 4 * nc, H, W};
    dcvc_trt_engine_set_shape(c->eng_spatial_prior, "in0", in_dims, 4);
    dcvc_trt_engine_set_addr(c->eng_spatial_prior, "in0", ws->d_cat);
    int nio = dcvc_trt_engine_num_io(c->eng_spatial_prior);
    const char* out_name = dcvc_trt_engine_tensor_name(c->eng_spatial_prior, nio - 1);
    dcvc_trt_engine_set_addr(c->eng_spatial_prior, out_name, ws->d_sp_out);
    DcvcRtStatus st = dcvc_trt_engine_execute(c->eng_spatial_prior, NULL);
    cudaDeviceSynchronize();
    return st;
}

/* ---- 2x (inter) encode ---- */
static DcvcRtStatus dcvc_ar_codec_encode_2x(DcvcArCodec* c,
                                            const void* d_y, const void* d_params,
                                            int H, int W,
                                            uint8_t** out_stream, size_t* out_size,
                                            void* d_y_hat_out)
{
    *out_stream = NULL; *out_size = 0;
    DcvcRtStatus st = ws_ensure(c, H, W);
    if (st != DCVC_RT_OK) return st;

    int nc = c->n_ch, hw = H * W, r = nc / 2;
    ArWorkspace* ws = &c->ws;

    /* separate_prior_video_enc: extract q_dec, scales, means from params */
    separate_prior_video_enc(c, d_params, H, W);

    /* clamp_recip_quant (in-place): q_dec = max(q_dec,0.5) [n_ch,H*W];
     * y_scaled = y / q_dec.  k_crq reads q_dec[i] before writing q_dec_clamp[i],
     * so aliasing the input and clamp output is safe per-thread. */
    c->k_crq(ws->d_qdec, d_y, ws->d_qdec, ws->d_yq_full, 0.5f, nc * hw, 0);
    /* ws->d_qdec = clamped q_dec [n_ch,H*W]; ws->d_yq_full = y_scaled [n_ch,H*W] */

    /* rANS encoder setup */
    dcvc_rans_encoder_reset(c->rans_enc);
    c->g_cdf_idx = dcvc_rans_encoder_add_cdf(c->rans_enc, c->gcdf.i32,
                        c->gcdf.dims[0], c->gcdf.dims[1], c->glen.i32, c->goff.i32);

    /* Round 0 */
    void* d_mask0 = generate_mask_2x(0, nc, H, W);
    void *curr_scales = ws->d_scales, *curr_means = ws->d_means;

    /* y_q = process_mask_yq(y_scaled, scales, means, mask_0) */
    c->k_pmask_yq(ws->d_yq_full, curr_scales, curr_means, d_mask0,
                  ws->d_yq_full, -1.0f, nc * hw, 0);
    /* y_q_w = sp2x(y_q) */
    c->k_sp2x(ws->d_yq_full, ws->d_yq_w, r * hw, 0);
    /* scales_r = sp2x(scales * mask_0) */
    c->k_mul(curr_scales, d_mask0, ws->d_smask, nc * hw, 0);
    c->k_sp2x(ws->d_smask, ws->d_sr, r * hw, 0);
    /* packed = build_index_enc(y_q_w, scales_r) */
    c->k_build_idx_enc(ws->d_yq_w, ws->d_sr, (int16_t*)ws->d_packed,
                       c->scale_min, c->scale_max,
                       c->log_scale_min, c->log_step_recip, r * hw, 0);
    {
        int16_t* packed_h = (int16_t*)malloc(r * hw * 2);
        cudaMemcpy(packed_h, ws->d_packed, r * hw * 2, cudaMemcpyDeviceToHost);
        dcvc_rans_encoder_encode_y(c->rans_enc, packed_h, r * hw, c->g_cdf_idx);
        free(packed_h);
    }
    /* y_hat_0 = restore_y_2x(y_q_w, means, mask_0) */
    c->k_restore_y2x(ws->d_yq_w, curr_means, d_mask0, ws->d_yhat_step,
                     r * hw, nc * hw, 0);
    cudaFree(d_mask0);

    /* spatial_prior: cat(y_hat_0, params) → scales_1, means_1 */
    st = run_spatial_prior_2x(c, ws->d_yhat_step, d_params, H, W);
    if (st != DCVC_RT_OK) return st;
    curr_scales = ws->d_sp_out;
    curr_means = (void*)((char*)ws->d_sp_out + (size_t)nc * hw * 2);

    /* Round 1 */
    void* d_mask1 = generate_mask_2x(1, nc, H, W);
    c->k_pmask_yq(ws->d_yq_full, curr_scales, curr_means, d_mask1,
                  ws->d_yq_full, -1.0f, nc * hw, 0);
    c->k_sp2x(ws->d_yq_full, ws->d_yq_w, r * hw, 0);
    c->k_mul(curr_scales, d_mask1, ws->d_smask, nc * hw, 0);
    c->k_sp2x(ws->d_smask, ws->d_sr, r * hw, 0);
    c->k_build_idx_enc(ws->d_yq_w, ws->d_sr, (int16_t*)ws->d_packed,
                       c->scale_min, c->scale_max,
                       c->log_scale_min, c->log_step_recip, r * hw, 0);
    {
        int16_t* packed_h = (int16_t*)malloc(r * hw * 2);
        cudaMemcpy(packed_h, ws->d_packed, r * hw * 2, cudaMemcpyDeviceToHost);
        dcvc_rans_encoder_encode_y(c->rans_enc, packed_h, r * hw, c->g_cdf_idx);
        free(packed_h);
    }
    /* y_hat_1 = restore_y_2x(y_q_w, means_1, mask_1) */
    c->k_restore_y2x(ws->d_yq_w, curr_means, d_mask1, ws->d_cat,
                     r * hw, nc * hw, 0);
    cudaFree(d_mask1);

    /* y_hat = add_and_multiply(y_hat_0, y_hat_1, q_dec) = (y_hat_0 + y_hat_1) * q_dec */
    /* ws->d_yhat_step has y_hat_0, ws->d_cat has y_hat_1 */
    /* add_and_multiply: x0 = (x0 + x1) * q, in-place on x0 */
    c->k_add_mul(ws->d_yhat_step, ws->d_cat, ws->d_qdec, nc * hw, 0);

    if (d_y_hat_out)
        cudaMemcpy(d_y_hat_out, ws->d_yhat_step, (size_t)nc * hw * 2, cudaMemcpyDeviceToDevice);

    dcvc_rans_encoder_flush(c->rans_enc);
    int rc = dcvc_rans_encoder_get_stream(c->rans_enc, out_stream, out_size);
    return rc == 0 ? DCVC_RT_OK : DCVC_RT_ERR_ENTROPY;
}

/* ---- 2x (inter) decode ---- */
static DcvcRtStatus dcvc_ar_codec_decode_2x(DcvcArCodec* c,
                                            const void* d_params,
                                            int H, int W,
                                            const uint8_t* stream, size_t stream_size,
                                            void* d_y_hat_out)
{
    DcvcRtStatus st = ws_ensure(c, H, W);
    if (st != DCVC_RT_OK) return st;

    int nc = c->n_ch, hw = H * W, r = nc / 2;
    ArWorkspace* ws = &c->ws;

    /* separate_prior_video_dec */
    separate_prior_video_dec(c, d_params, H, W);
    void *curr_scales = ws->d_scales, *curr_means = ws->d_means;

    /* rANS decoder setup */
    dcvc_rans_decoder_reset_cdf(c->rans_dec);
    c->g_cdf_idx = dcvc_rans_decoder_add_cdf(c->rans_dec, c->gcdf.i32,
                        c->gcdf.dims[0], c->gcdf.dims[1], c->glen.i32, c->goff.i32);
    dcvc_rans_decoder_set_stream(c->rans_dec, stream, stream_size);

    /* Round 0 */
    void* d_mask0 = generate_mask_2x(0, nc, H, W);
    /* scales_r = sp2x(scales * mask_0) */
    c->k_mul(curr_scales, d_mask0, ws->d_smask, nc * hw, 0);
    c->k_sp2x(ws->d_smask, ws->d_sr, r * hw, 0);
    /* build_index_dec → indexes */
    c->k_build_idx_dec(ws->d_sr, (uint8_t*)ws->d_indexes, c->scale_min, c->scale_max,
                       c->log_scale_min, c->log_step_recip, r * hw, 0);
    {
        uint8_t* idx_h = (uint8_t*)malloc(r * hw);
        cudaMemcpy(idx_h, ws->d_indexes, r * hw, cudaMemcpyDeviceToHost);
        dcvc_rans_decoder_decode_y(c->rans_dec, idx_h, r * hw, c->g_cdf_idx);
        free(idx_h);
    }
    int8_t* syms = NULL; size_t sym_n = 0;
    dcvc_rans_decoder_get_symbols(c->rans_dec, (int8_t**)&syms, &sym_n);
    /* y_q_r as fp16 on device */
    {
        uint16_t* yqh = (uint16_t*)malloc(r * hw * 2);
        for (int i = 0; i < r * hw; i++) yqh[i] = f32h((float)syms[i]);
        cudaMemcpy(ws->d_yq_w, yqh, r * hw * 2, cudaMemcpyHostToDevice);
        free(yqh);
    }
    /* y_hat_0 = restore_y_2x(y_q_r, means, mask_0) */
    c->k_restore_y2x(ws->d_yq_w, curr_means, d_mask0, ws->d_yhat_step,
                     r * hw, nc * hw, 0);
    cudaFree(d_mask0);

    /* spatial_prior: cat(y_hat_0, params) → scales_1, means_1 */
    st = run_spatial_prior_2x(c, ws->d_yhat_step, d_params, H, W);
    if (st != DCVC_RT_OK) return st;
    curr_scales = ws->d_sp_out;
    curr_means = (void*)((char*)ws->d_sp_out + (size_t)nc * hw * 2);

    /* Round 1 */
    void* d_mask1 = generate_mask_2x(1, nc, H, W);
    c->k_mul(curr_scales, d_mask1, ws->d_smask, nc * hw, 0);
    c->k_sp2x(ws->d_smask, ws->d_sr, r * hw, 0);
    c->k_build_idx_dec(ws->d_sr, (uint8_t*)ws->d_indexes, c->scale_min, c->scale_max,
                       c->log_scale_min, c->log_step_recip, r * hw, 0);
    {
        uint8_t* idx_h = (uint8_t*)malloc(r * hw);
        cudaMemcpy(idx_h, ws->d_indexes, r * hw, cudaMemcpyDeviceToHost);
        dcvc_rans_decoder_decode_y(c->rans_dec, idx_h, r * hw, c->g_cdf_idx);
        free(idx_h);
    }
    dcvc_rans_decoder_get_symbols(c->rans_dec, (int8_t**)&syms, &sym_n);
    {
        uint16_t* yqh = (uint16_t*)malloc(r * hw * 2);
        for (int i = 0; i < r * hw; i++) yqh[i] = f32h((float)syms[i]);
        cudaMemcpy(ws->d_yq_w, yqh, r * hw * 2, cudaMemcpyHostToDevice);
        free(yqh);
    }
    /* y_hat_1 = restore_y_2x(y_q_r, means_1, mask_1) */
    c->k_restore_y2x(ws->d_yq_w, curr_means, d_mask1, ws->d_cat,
                     r * hw, nc * hw, 0);
    cudaFree(d_mask1);

    /* y_hat = (y_hat_0 + y_hat_1) * q_dec */
    c->k_add_mul(ws->d_yhat_step, ws->d_cat, ws->d_qdec, nc * hw, 0);

    cudaMemcpy(d_y_hat_out, ws->d_yhat_step, (size_t)nc * hw * 2, cudaMemcpyDeviceToDevice);
    return DCVC_RT_OK;
}

DcvcRtStatus dcvc_ar_codec_encode(DcvcArCodec* c,
                                  const void* d_y, const void* d_params_fusion,
                                  int H, int W,
                                  uint8_t** out_stream, size_t* out_size,
                                  void* d_y_hat_out)
{
    if (!c || !d_y || !d_params_fusion || !out_stream || !out_size)
        return DCVC_RT_ERR_INVALID_ARG;
    *out_stream = NULL; *out_size = 0;

    /* Dispatch: 2x (inter) has a completely separate code path */
    if (c->passes == 2)
        return dcvc_ar_codec_encode_2x(c, d_y, d_params_fusion, H, W,
                                       out_stream, out_size, d_y_hat_out);

    DcvcRtStatus st = ws_ensure(c, H, W);
    if (st != DCVC_RT_OK) return st;

    int nc = c->n_ch, hw = H * W, r = nc / 4;
    ArWorkspace* ws = &c->ws;

    /* separate_prior: q_enc, q_dec, scales, means from params_fusion */
    separate_prior(c, d_params_fusion, H, W);

    /* y_spatial_prior_reduction: params_fusion → common_params [1, nc, H, W] */
    int32_t pf_dims[4] = {1, 514, H, W};
    st = bind_and_run(c->eng_reduction, "in0", pf_dims, (void*)d_params_fusion, ws->d_common);
    if (st != DCVC_RT_OK) return st;
    cudaDeviceSynchronize();

    /* y_scaled = y * q_enc (broadcast q_enc across all channels) */
    {
        uint16_t* qe = (uint16_t*)malloc(hw * 2);
        cudaMemcpy(qe, ws->d_qenc, hw * 2, cudaMemcpyDeviceToHost);
        uint16_t* yh = (uint16_t*)malloc(nc * hw * 2);
        cudaMemcpy(yh, d_y, nc * hw * 2, cudaMemcpyDeviceToHost);
        for (int ch = 0; ch < nc; ch++)
            for (int i = 0; i < hw; i++)
                yh[ch * hw + i] = f32h(h2f(yh[ch * hw + i]) * h2f(qe[i]));
        cudaMemcpy(ws->d_yq_full, yh, nc * hw * 2, cudaMemcpyHostToDevice);
        free(qe); free(yh);
    }

    /* rANS encoder: reset and add gaussian CDF */
    dcvc_rans_encoder_reset(c->rans_enc);
    c->g_cdf_idx = dcvc_rans_encoder_add_cdf(c->rans_enc, c->gcdf.i32,
                        c->gcdf.dims[0], c->gcdf.dims[1], c->glen.i32, c->goff.i32);

    /* y_hat_so_far = 0 */
    cudaMemset(ws->d_yhat, 0, nc * hw * 2);

    for (int round = 0; round < c->passes; round++) {
        void *curr_scales = ws->d_scales, *curr_means = ws->d_means;
        if (round > 0) {
            st = run_spatial_prior(c, ws->d_yhat, H, W, round);
            if (st != DCVC_RT_OK) return st;
            curr_scales = ws->d_cat;
            curr_means = (void*)((char*)ws->d_cat + (size_t)nc * hw * 2);
        }

        /* Generate mask for this round */
        void* d_mask = generate_mask(round, nc, H, W);

        /* scales_r = sp4x(scales * mask) */
        c->k_mul(curr_scales, d_mask, ws->d_smask, nc * hw, 0);
        c->k_sp4x(ws->d_smask, ws->d_sr, r * hw, 0);

        /* y_q = process_mask_yq(y_scaled, scales, means, mask) → full [nc, hw] */
        c->k_pmask_yq(ws->d_yq_full, curr_scales, curr_means, d_mask,
                      ws->d_yq_full, -1.0f, nc * hw, 0);
        /* y_q_w = sp4x(y_q) → [r, hw] */
        c->k_sp4x(ws->d_yq_full, ws->d_yq_w, r * hw, 0);

        /* packed = build_index_enc(y_q_w, scales_r) */
        c->k_build_idx_enc(ws->d_yq_w, ws->d_sr, (int16_t*)ws->d_packed,
                           c->scale_min, c->scale_max,
                           c->log_scale_min, c->log_step_recip, r * hw, 0);
        int16_t* packed_h = (int16_t*)malloc(r * hw * 2);
        cudaMemcpy(packed_h, ws->d_packed, r * hw * 2, cudaMemcpyDeviceToHost);
        dcvc_rans_encoder_encode_y(c->rans_enc, packed_h, r * hw, c->g_cdf_idx);
        free(packed_h);

        /* restore_y_4x(y_q_w, means, mask) → y_hat_step ; accumulate */
        c->k_restore_y4x(ws->d_yq_w, curr_means, d_mask, ws->d_yhat_step,
                         r * hw, nc * hw, 0);
        host_add_inplace(ws->d_yhat, ws->d_yhat_step, nc * hw);

        cudaFree(d_mask);
    }

    /* y_hat_enc = y_hat_so_far * q_dec */
    {
        uint16_t* qd = (uint16_t*)malloc(hw * 2);
        cudaMemcpy(qd, ws->d_qdec, hw * 2, cudaMemcpyDeviceToHost);
        host_mul_qdec(ws->d_yhat, qd, nc, hw);
        free(qd);
    }
    if (d_y_hat_out)
        cudaMemcpy(d_y_hat_out, ws->d_yhat, (size_t)nc * hw * 2, cudaMemcpyDeviceToDevice);

    dcvc_rans_encoder_flush(c->rans_enc);
    int rc = dcvc_rans_encoder_get_stream(c->rans_enc, out_stream, out_size);
    return rc == 0 ? DCVC_RT_OK : DCVC_RT_ERR_ENTROPY;
}

DcvcRtStatus dcvc_ar_codec_decode(DcvcArCodec* c,
                                  const void* d_params_fusion,
                                  int H, int W,
                                  const uint8_t* stream, size_t stream_size,
                                  void* d_y_hat_out)
{
    if (!c || !d_params_fusion || !stream || !d_y_hat_out)
        return DCVC_RT_ERR_INVALID_ARG;

    /* Dispatch: 2x (inter) has a completely separate code path */
    if (c->passes == 2)
        return dcvc_ar_codec_decode_2x(c, d_params_fusion, H, W,
                                       stream, stream_size, d_y_hat_out);

    DcvcRtStatus st = ws_ensure(c, H, W);
    if (st != DCVC_RT_OK) return st;

    int nc = c->n_ch, hw = H * W, r = nc / 4;
    ArWorkspace* ws = &c->ws;

    /* separate_prior */
    separate_prior(c, d_params_fusion, H, W);

    /* y_spatial_prior_reduction */
    int32_t pf_dims[4] = {1, 514, H, W};
    st = bind_and_run(c->eng_reduction, "in0", pf_dims, (void*)d_params_fusion, ws->d_common);
    if (st != DCVC_RT_OK) return st;
    cudaDeviceSynchronize();

    /* rANS decoder setup */
    dcvc_rans_decoder_reset_cdf(c->rans_dec);
    c->g_cdf_idx = dcvc_rans_decoder_add_cdf(c->rans_dec, c->gcdf.i32,
                        c->gcdf.dims[0], c->gcdf.dims[1], c->glen.i32, c->goff.i32);
    dcvc_rans_decoder_set_stream(c->rans_dec, stream, stream_size);

    /* y_hat_so_far = 0 */
    cudaMemset(ws->d_yhat, 0, nc * hw * 2);

    for (int round = 0; round < c->passes; round++) {
        void *curr_scales = ws->d_scales, *curr_means = ws->d_means;
        if (round > 0) {
            st = run_spatial_prior(c, ws->d_yhat, H, W, round);
            if (st != DCVC_RT_OK) return st;
            curr_scales = ws->d_cat;
            curr_means = (void*)((char*)ws->d_cat + (size_t)nc * hw * 2);
        }

        void* d_mask = generate_mask(round, nc, H, W);

        /* scales_r = sp4x(scales * mask) */
        c->k_mul(curr_scales, d_mask, ws->d_smask, nc * hw, 0);
        c->k_sp4x(ws->d_smask, ws->d_sr, r * hw, 0);

        /* build_index_dec → indexes */
        c->k_build_idx_dec(ws->d_sr, (uint8_t*)ws->d_indexes, c->scale_min, c->scale_max,
                           c->log_scale_min, c->log_step_recip, r * hw, 0);
        uint8_t* indexes_h = (uint8_t*)malloc(r * hw);
        cudaMemcpy(indexes_h, ws->d_indexes, r * hw, cudaMemcpyDeviceToHost);

        /* rANS decode_y */
        dcvc_rans_decoder_decode_y(c->rans_dec, indexes_h, r * hw, c->g_cdf_idx);
        int8_t* syms = NULL; size_t sym_n = 0;
        dcvc_rans_decoder_get_symbols(c->rans_dec, (int8_t**)&syms, &sym_n);
        free(indexes_h);

        /* y_q_r as fp16 on device */
        uint16_t* yqh = (uint16_t*)malloc(r * hw * 2);
        for (int i = 0; i < r * hw; i++) yqh[i] = f32h((float)syms[i]);
        cudaMemcpy(ws->d_yq_w, yqh, r * hw * 2, cudaMemcpyHostToDevice);
        free(yqh);

        /* restore_y_4x and accumulate */
        c->k_restore_y4x(ws->d_yq_w, curr_means, d_mask, ws->d_yhat_step,
                         r * hw, nc * hw, 0);
        host_add_inplace(ws->d_yhat, ws->d_yhat_step, nc * hw);

        cudaFree(d_mask);
    }

    /* y_hat = y_hat_so_far * q_dec */
    {
        uint16_t* qd = (uint16_t*)malloc(hw * 2);
        cudaMemcpy(qd, ws->d_qdec, hw * 2, cudaMemcpyDeviceToHost);
        host_mul_qdec(ws->d_yhat, qd, nc, hw);
        free(qd);
    }

    /* Copy result to caller's buffer */
    cudaMemcpy(d_y_hat_out, ws->d_yhat, nc * hw * 2, cudaMemcpyDeviceToDevice);
    return DCVC_RT_OK;
}
