// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Full inter (P-frame) pipeline. Loads the inter engines directly and
// orchestrates: feature_adaptor_{p|i} → feature_extractor(p1/p2) → encoder →
// hyper_enc → z rANS → hyper_dec+temporal_prior → prior_fusion → AR(2x) →
// decoder → recon_generation. Self-consistent closed loop.

#include "dcvc_inter_pipeline.h"
#include "dcvc_rt_internal.h"
#include "trt_engine.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- FP16 helpers ---- */
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

typedef void (*fn_round_to_int8)(const void*, void*, int8_t*, int, cudaStream_t);
typedef void (*fn_pixel_unshuffle_8)(const void*, void*, int, int, cudaStream_t);
typedef void (*fn_mul_qfeat_broadcast)(const void*, const void*, void*, int, int, cudaStream_t);

/* ---- Minimal npy reader (int32 + fp16) ---- */
typedef struct { int dims[4]; size_t elems; int32_t* i32; uint16_t* fp16; } PnNpy;
static int pn_read_npy(const char* path, PnNpy* o)
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
    if (sp) { sp = strchr(sp, '('); if (sp) { sp++;
        while (*sp && *sp != ')' && nd < 4) {
            if (*sp >= '0' && *sp <= '9') { o->dims[nd++] = atoi(sp);
                while (*sp >= '0' && *sp <= '9') sp++; }
            sp++;
        }
    } }
    while (nd < 4) o->dims[nd++] = 1;
    o->elems = 1; for (int i = 0; i < 4; i++) o->elems *= (size_t)o->dims[i];
    o->i32 = o->fp16 = NULL;
    if (strstr(hdr, "i4")) { o->i32 = malloc(o->elems * 4);
        size_t g = fread(o->i32, 4, o->elems, f); fclose(f); return g == o->elems ? 0 : -1; }
    if (strstr(hdr, "f2")) { o->fp16 = malloc(o->elems * 2);
        size_t g = fread(o->fp16, 2, o->elems, f); fclose(f); return g == o->elems ? 0 : -1; }
    fclose(f);
    return -1;
}

/* ---- pixel_unshuffle(frame[1,3,H,W], 8) → [1,192,H/8,W/8] ---- */
/* GPU pixel_unshuffle: inline kernel call (struct must be defined first) */

/* ---- Engine binding helpers ---- */
static const char* eng_out_name(DcvcTrtEngine* eng)
{
    int nio = dcvc_trt_engine_num_io(eng);
    for (int j = 0; j < nio; j++)
        if (!dcvc_trt_engine_is_input(eng, j))
            return dcvc_trt_engine_tensor_name(eng, j);
    return dcvc_trt_engine_tensor_name(eng, nio - 1);
}
static DcvcRtStatus bind_in(DcvcTrtEngine* eng, const char* name,
                            const int32_t* dims, void* ptr)
{
    DcvcRtStatus st = dcvc_trt_engine_set_shape(eng, name, dims, 4);
    if (st != DCVC_RT_OK) return st;
    return dcvc_trt_engine_set_addr(eng, name, ptr);
}
static DcvcRtStatus run1(DcvcTrtEngine* eng, const char* n0, const int32_t* d0, void* p0, void* out)
{
    DcvcRtStatus st = bind_in(eng, n0, d0, p0);
    if (st != DCVC_RT_OK) return st;
    st = dcvc_trt_engine_set_addr(eng, eng_out_name(eng), out);
    if (st != DCVC_RT_OK) return st;
    return dcvc_trt_engine_execute(eng, NULL);
}
static DcvcRtStatus run2(DcvcTrtEngine* eng, const char* n0, const int32_t* d0, void* p0,
                         const char* n1, const int32_t* d1, void* p1, void* out)
{
    DcvcRtStatus st = bind_in(eng, n0, d0, p0);
    if (st != DCVC_RT_OK) return st;
    st = bind_in(eng, n1, d1, p1);
    if (st != DCVC_RT_OK) return st;
    st = dcvc_trt_engine_set_addr(eng, eng_out_name(eng), out);
    if (st != DCVC_RT_OK) return st;
    return dcvc_trt_engine_execute(eng, NULL);
}
static DcvcRtStatus run3(DcvcTrtEngine* eng, const char* n0, const int32_t* d0, void* p0,
                         const char* n1, const int32_t* d1, void* p1,
                         const char* n2, const int32_t* d2, void* p2, void* out)
{
    DcvcRtStatus st = bind_in(eng, n0, d0, p0);
    if (st != DCVC_RT_OK) return st;
    st = bind_in(eng, n1, d1, p1);
    if (st != DCVC_RT_OK) return st;
    st = bind_in(eng, n2, d2, p2);
    if (st != DCVC_RT_OK) return st;
    st = dcvc_trt_engine_set_addr(eng, eng_out_name(eng), out);
    if (st != DCVC_RT_OK) return st;
    return dcvc_trt_engine_execute(eng, NULL);
}

/* ---- Pipeline state ---- */
#define INTER_N_ENG 11
struct DcvcInterPipeline {
    char asset_dir[512];
    char plugin_dir[512];
    DcvcTrtEngine* eng[INTER_N_ENG];
    void* kernel_so;
    fn_round_to_int8 k_round_to_int8;
    fn_pixel_unshuffle_8 k_pixel_unshuffle_8;
    fn_mul_qfeat_broadcast k_mul_qfeat_broadcast;
    int H, W, fH, fW, yH, yW, zH, zW;
    void *d_feature, *d_pixunshuf, *d_x1, *d_ctx_t, *d_ctx, *d_y, *d_z, *d_zhat, *d_z_int8;
    void *d_hier, *d_temporal, *d_pf_in, *d_params, *d_yhat, *d_fdec;
    void *d_q_enc, *d_q_dec, *d_q_feat, *d_q_recon, *d_q_ones;
    int8_t* z_int8_h;
    uint16_t* z_hat_h;
};
enum { E_FADAP_P, E_FADAP_I, E_FE1, E_FE2, E_ENC, E_HENC, E_HDEC, E_TEMP, E_PFUS, E_DEC, E_RECON };
static const char* k_eng_files[INTER_N_ENG] = {
    "inter_feature_adaptor_p", "inter_feature_adaptor_i", "inter_feature_extractor_p1",
    "inter_feature_extractor_p2", "inter_encoder", "inter_hyper_enc", "inter_hyper_dec",
    "inter_temporal_prior", "inter_prior_fusion", "inter_decoder", "recon_generation",
};

static void ips_free_bufs(DcvcInterPipeline* p)
{
    if (!p->d_feature) return;
    cudaFree(p->d_feature); cudaFree(p->d_pixunshuf); cudaFree(p->d_x1); cudaFree(p->d_ctx_t);
    cudaFree(p->d_ctx); cudaFree(p->d_y); cudaFree(p->d_z); cudaFree(p->d_zhat); cudaFree(p->d_z_int8);
    cudaFree(p->d_hier); cudaFree(p->d_temporal); cudaFree(p->d_pf_in);
    cudaFree(p->d_params); cudaFree(p->d_yhat); cudaFree(p->d_fdec);
    cudaFree(p->d_q_enc); cudaFree(p->d_q_dec); cudaFree(p->d_q_feat);
    cudaFree(p->d_q_recon); cudaFree(p->d_q_ones);
    free(p->z_int8_h); free(p->z_hat_h);
    p->d_feature = NULL;
}

static DcvcRtStatus ips_ensure(DcvcInterPipeline* p, int H, int W)
{
    if (p->H == H && p->W == W && p->d_feature) return DCVC_RT_OK;
    ips_free_bufs(p);
    p->H = H; p->W = W;
    p->fH = H / 8;  p->fW = W / 8;
    p->yH = H / 16; p->yW = W / 16;
    p->zH = H / 64; p->zW = W / 64;
    int fHW = p->fH * p->fW, yHW = p->yH * p->yW, zHW = p->zH * p->zW;
    const int D = DCVC_RT_CH_D, SRC = DCVC_RT_CH_SRC_D, YC = DCVC_RT_CH_Y_INTER, ZC = DCVC_RT_CH_Z, RECON = 320;

    cudaMalloc(&p->d_feature,  D * fHW * 2);
    cudaMalloc(&p->d_pixunshuf, SRC * fHW * 2);
    cudaMalloc(&p->d_x1,       D * fHW * 2);
    cudaMalloc(&p->d_ctx_t,    D * fHW * 2);
    cudaMalloc(&p->d_ctx,      D * fHW * 2);
    cudaMalloc(&p->d_y,        YC * yHW * 2);
    cudaMalloc(&p->d_z,        ZC * zHW * 2);
    cudaMalloc(&p->d_zhat,     ZC * zHW * 2);
    cudaMalloc(&p->d_z_int8,   ZC * zHW);
    cudaMalloc(&p->d_hier,     YC * yHW * 2);
    cudaMalloc(&p->d_temporal, (YC * 2) * yHW * 2);
    cudaMalloc(&p->d_pf_in,    (YC * 3) * yHW * 2);
    cudaMalloc(&p->d_params,   (YC * 3) * yHW * 2);
    cudaMalloc(&p->d_yhat,     YC * yHW * 2);
    cudaMalloc(&p->d_fdec,     D * fHW * 2);
    cudaMalloc(&p->d_q_enc,    D * 2);
    cudaMalloc(&p->d_q_dec,    D * 2);
    cudaMalloc(&p->d_q_feat,   D * 2);
    cudaMalloc(&p->d_q_recon,  RECON * 2);
    cudaMalloc(&p->d_q_ones,   D * 2);
    p->z_int8_h = malloc(ZC * zHW);
    p->z_hat_h  = malloc(ZC * zHW * 2);

    void* ptrs[] = {p->d_feature,p->d_pixunshuf,p->d_x1,p->d_ctx_t,p->d_ctx,p->d_y,p->d_z,
        p->d_zhat,p->d_z_int8,p->d_hier,p->d_temporal,p->d_pf_in,p->d_params,p->d_yhat,p->d_fdec,
        p->d_q_enc,p->d_q_dec,p->d_q_feat,p->d_q_recon,p->d_q_ones};
    for (int i = 0; i < 20; i++)
        if (!ptrs[i]) { ips_free_bufs(p); return DCVC_RT_ERR_OOM; }

    uint16_t ones[D]; for (int i = 0; i < D; i++) ones[i] = f32h(1.0f);
    cudaMemcpy(p->d_q_ones, ones, D * 2, cudaMemcpyHostToDevice);
    return DCVC_RT_OK;
}

/* build reference feature: adaptor_p(feature) or pixel_unshuffle+adaptor_i(pixels) */
static DcvcRtStatus build_ref_feature(DcvcInterPipeline* p, const void* d_ref_feature,
                                      const void* d_ref_pixels, int H, int W)
{
    int fH = p->fH, fW = p->fW;
    const int D = DCVC_RT_CH_D, SRC = DCVC_RT_CH_SRC_D;
    if (d_ref_feature) {
        int32_t d[4] = {1, D, fH, fW};
        return run1(p->eng[E_FADAP_P], "in0", d, (void*)d_ref_feature, p->d_feature);
    }
    /* P-after-I: pixel_unshuffle then adaptor_i */
    p->k_pixel_unshuffle_8(d_ref_pixels, p->d_pixunshuf, H, W, 0);
    int32_t d[4] = {1, SRC, fH, fW};
    return run1(p->eng[E_FADAP_I], "in0", d, p->d_pixunshuf, p->d_feature);
}

static DcvcRtStatus load_qp(DcvcInterPipeline* p, const char* bank, int qp,
                            void* d_out, int channels)
{
    char path[640];
    snprintf(path, sizeof(path), "%s/qp/%s.bin", p->asset_dir, bank);
    FILE* f = fopen(path, "rb");
    if (!f) return DCVC_RT_ERR_IO;
    int32_t qp_num = 0, ch = 0;
    if (fread(&qp_num, 4, 1, f) != 1 || fread(&ch, 4, 1, f) != 1) { fclose(f); return DCVC_RT_ERR_IO; }
    if (ch != channels || qp < 0 || qp >= qp_num) { fclose(f); return DCVC_RT_ERR_INVALID_ARG; }
    if (fseek(f, (long)(8 + (size_t)qp * channels * 2), SEEK_SET) != 0) { fclose(f); return DCVC_RT_ERR_IO; }
    uint16_t* tmp = malloc(channels * 2);
    size_t got = fread(tmp, 2, channels, f);
    fclose(f);
    if (got != (size_t)channels) { free(tmp); return DCVC_RT_ERR_IO; }
    cudaMemcpy(d_out, tmp, channels * 2, cudaMemcpyHostToDevice);
    free(tmp);
    return DCVC_RT_OK;
}

static DcvcRtStatus load_z_cdfs(DcvcInterPipeline* p, int enc, void* r)
{
    static int lenc = 0, ldec = 0;
    if (enc ? lenc : ldec) return DCVC_RT_OK;
    char path[640]; PnNpy zcdf, zlen, zoff;
    snprintf(path, sizeof(path), "%s/decode/bitest_cdf.npy", p->asset_dir);
    if (pn_read_npy(path, &zcdf) != 0) return DCVC_RT_ERR_IO;
    snprintf(path, sizeof(path), "%s/decode/bitest_cdf_length.npy", p->asset_dir);
    if (pn_read_npy(path, &zlen) != 0) { free(zcdf.i32); free(zcdf.fp16); return DCVC_RT_ERR_IO; }
    snprintf(path, sizeof(path), "%s/decode/bitest_offset.npy", p->asset_dir);
    if (pn_read_npy(path, &zoff) != 0) { free(zcdf.i32); free(zcdf.fp16); free(zlen.i32); free(zlen.fp16); return DCVC_RT_ERR_IO; }
    if (enc) {
        dcvc_rans_encoder_reset((DcvcRansEncoder*)r);
        dcvc_rans_encoder_add_cdf((DcvcRansEncoder*)r, zcdf.i32, zcdf.dims[0], zcdf.dims[1], zlen.i32, zoff.i32);
        lenc = 1;
    } else {
        dcvc_rans_decoder_reset_cdf((DcvcRansDecoder*)r);
        dcvc_rans_decoder_add_cdf((DcvcRansDecoder*)r, zcdf.i32, zcdf.dims[0], zcdf.dims[1], zlen.i32, zoff.i32);
        ldec = 1;
    }
    free(zcdf.i32); free(zcdf.fp16); free(zlen.i32); free(zlen.fp16); free(zoff.i32); free(zoff.fp16);
    return DCVC_RT_OK;
}

static void mul_qfeat_broadcast(DcvcInterPipeline* p)
{
    int D = DCVC_RT_CH_D, fHW = p->fH * p->fW;
    /* GPU: ctx_t = x1 * q_feat (channel-broadcast multiply, no host round-trip) */
    p->k_mul_qfeat_broadcast(p->d_x1, p->d_q_feat, p->d_ctx_t, D, fHW, 0);
}

DcvcInterPipeline* dcvc_inter_pipeline_create(const char* asset_dir,
                                              const char* plugin_dir,
                                              DcvcRtStatus* st_out)
{
    DcvcInterPipeline* p = calloc(1, sizeof(*p));
    if (!p) { if (st_out) *st_out = DCVC_RT_ERR_OOM; return NULL; }
    strncpy(p->asset_dir, asset_dir ? asset_dir : "native/assets", sizeof(p->asset_dir) - 1);
    strncpy(p->plugin_dir, plugin_dir ? plugin_dir : "native/build/plugin_demo", sizeof(p->plugin_dir) - 1);
    DcvcRtStatus st = DCVC_RT_OK;
    char path[640];
    snprintf(path, sizeof(path), "%s/libdcvc_kernels.so", p->plugin_dir);
    p->kernel_so = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!p->kernel_so) { st = DCVC_RT_ERR_IO; goto fail; }
    p->k_round_to_int8 = (fn_round_to_int8)dlsym(p->kernel_so, "dcvc_k_round_to_int8");
    if (!p->k_round_to_int8) { st = DCVC_RT_ERR_IO; goto fail; }
    p->k_pixel_unshuffle_8 = (fn_pixel_unshuffle_8)dlsym(p->kernel_so, "dcvc_k_pixel_unshuffle_8");
    p->k_mul_qfeat_broadcast = (fn_mul_qfeat_broadcast)dlsym(p->kernel_so, "dcvc_k_mul_qfeat_broadcast");
    if (!p->k_pixel_unshuffle_8 || !p->k_mul_qfeat_broadcast) { st = DCVC_RT_ERR_IO; goto fail; }
    for (int i = 0; i < INTER_N_ENG; i++) {
        snprintf(path, sizeof(path), "%s/engines/%s.engine", p->asset_dir, k_eng_files[i]);
        p->eng[i] = dcvc_trt_engine_load(path, p->plugin_dir, &st);
        if (!p->eng[i]) goto fail;
    }
    if (st_out) *st_out = DCVC_RT_OK;
    return p;
fail:
    dcvc_inter_pipeline_destroy(p);
    if (st_out) *st_out = st;
    return NULL;
}

void dcvc_inter_pipeline_destroy(DcvcInterPipeline* p)
{
    if (!p) return;
    ips_free_bufs(p);
    for (int i = 0; i < INTER_N_ENG; i++)
        if (p->eng[i]) dcvc_trt_engine_destroy(p->eng[i]);
    if (p->kernel_so) dlclose(p->kernel_so);
    free(p);
}

/* ===================================================================== */
DcvcRtStatus dcvc_inter_encode(DcvcInterPipeline* p, DcvcArCodec* ar_codec,
                               DcvcRansEncoder* rans_enc,
                               const void* d_image, const void* d_ref_feature,
                               const void* d_ref_pixels, int H, int W, int qp,
                               uint8_t** out_stream, size_t* out_size,
                               void* d_xhat_out, void* d_ref_feature_out)
{
    if (!p || !ar_codec || !rans_enc || !d_image || (!d_ref_feature && !d_ref_pixels) || !out_stream || !out_size)
        return DCVC_RT_ERR_INVALID_ARG;
    *out_stream = NULL; *out_size = 0;
    DcvcRtStatus st = ips_ensure(p, H, W);
    if (st != DCVC_RT_OK) return st;

    const int D = DCVC_RT_CH_D, YC = DCVC_RT_CH_Y_INTER, ZC = DCVC_RT_CH_Z, RECON = 320;
    int fH = p->fH, fW = p->fW, yH = p->yH, yW = p->yW, zH = p->zH, zW = p->zW;
    int yHW = yH * yW, zHW = zH * zW;

    if ((st = load_qp(p, "inter_q_encoder", qp, p->d_q_enc, D)) != DCVC_RT_OK) return st;
    if ((st = load_qp(p, "inter_q_decoder", qp, p->d_q_dec, D)) != DCVC_RT_OK) return st;
    if ((st = load_qp(p, "inter_q_feature", qp, p->d_q_feat, D)) != DCVC_RT_OK) return st;
    if ((st = load_qp(p, "inter_q_recon", qp, p->d_q_recon, RECON)) != DCVC_RT_OK) return st;
    if ((st = load_z_cdfs(p, 1, rans_enc)) != DCVC_RT_OK) return st;

    /* 1. ref feature */
    st = build_ref_feature(p, d_ref_feature, d_ref_pixels, H, W);
    if (st != DCVC_RT_OK) return st

    /* 2. fe1(feature, ones) → x1 ; ctx_t = x1·q_feat */
    { int32_t fd[4]={1,D,fH,fW}, qd[4]={1,D,1,1};
      st = run2(p->eng[E_FE1], "in0", fd, p->d_feature, "q", qd, p->d_q_ones, p->d_x1);
      if (st != DCVC_RT_OK) return st }
    mul_qfeat_broadcast(p);

    /* 3. fe2(x1) → ctx */
    { int32_t fd[4]={1,D,fH,fW};
      st = run1(p->eng[E_FE2], "in0", fd, p->d_x1, p->d_ctx);
      if (st != DCVC_RT_OK) return st }

    /* 4. encoder(x, ctx, q_enc) → y */
    { int32_t xd[4]={1,3,H,W}, cd[4]={1,D,fH,fW}, qd[4]={1,D,1,1};
      st = run3(p->eng[E_ENC], "x", xd, (void*)d_image, "ctx", cd, p->d_ctx, "q", qd, p->d_q_enc, p->d_y);
      if (st != DCVC_RT_OK) return st }

    /* 5. hyper_enc(y) → z */
    { int32_t yd[4]={1,YC,yH,yW};
      st = run1(p->eng[E_HENC], "in0", yd, p->d_y, p->d_z);
      if (st != DCVC_RT_OK) return st }

    /* 6. round_to_int8 */
    { int zt = ZC * zHW;
      p->k_round_to_int8(p->d_z, p->d_zhat, (int8_t*)p->d_z_int8, zt, 0);
      cudaDeviceSynchronize();
      cudaMemcpy(p->z_int8_h, p->d_z_int8, zt, cudaMemcpyDeviceToHost); }

    /* 7. z rANS encode */
    { int zt = ZC * zHW;
      /* Reset pending symbol list each frame; CDFs persist in the encoder.
       * Without this, m_pendingEncodingList accumulates symbols across frames
       * and the z bitstream is corrupted from the 2nd P-frame onward. */
      dcvc_rans_encoder_reset(rans_enc);
      dcvc_rans_encoder_encode_z(rans_enc, p->z_int8_h, zt, 0, qp * ZC, zHW);
      dcvc_rans_encoder_flush(rans_enc); }

    /* 8. hyper_dec(z_hat) → hier */
    { int32_t zd[4]={1,ZC,zH,zW};
      st = run1(p->eng[E_HDEC], "in0", zd, p->d_zhat, p->d_hier);
      if (st != DCVC_RT_OK) return st }

    /* 9. temporal_prior(ctx_t) → temporal */
    { int32_t cd[4]={1,D,fH,fW};
      st = run1(p->eng[E_TEMP], "in0", cd, p->d_ctx_t, p->d_temporal);
      if (st != DCVC_RT_OK) return st }

    /* 10. cat(hier,temporal) → prior_fusion → params */
    { cudaMemcpy(p->d_pf_in, p->d_hier, YC*yHW*2, cudaMemcpyDeviceToDevice);
      cudaMemcpy((char*)p->d_pf_in+(size_t)YC*yHW*2, p->d_temporal, (YC*2)*yHW*2, cudaMemcpyDeviceToDevice);
      int32_t pd[4]={1,YC*3,yH,yW};
      st = run1(p->eng[E_PFUS], "in0", pd, p->d_pf_in, p->d_params);
      if (st != DCVC_RT_OK) return st }

    /* 11. AR(2x) encode */
    uint8_t* y_stream = NULL; size_t y_stream_size = 0;
    st = dcvc_ar_codec_encode(ar_codec, p->d_y, p->d_params, yH, yW, &y_stream, &y_stream_size, p->d_yhat);
    if (st != DCVC_RT_OK) return st;

    /* 12. decoder(y_hat, ctx, q_dec) → dec_feature */
    { int32_t yd[4]={1,YC,yH,yW}, cd[4]={1,D,fH,fW}, qd[4]={1,D,1,1};
      st = run3(p->eng[E_DEC], "in0", yd, p->d_yhat, "ctx", cd, p->d_ctx, "q", qd, p->d_q_dec, p->d_fdec);
      if (st != DCVC_RT_OK) { free(y_stream); return st; } }

    /* 13. optional recon → x_hat */
    if (d_xhat_out) {
        int32_t fd[4]={1,D,fH,fW}, qd[4]={1,RECON,1,1};
        st = run2(p->eng[E_RECON], "in0", fd, p->d_fdec, "q", qd, p->d_q_recon, d_xhat_out);
        if (st != DCVC_RT_OK) { free(y_stream); return st; }
    }
    if (d_ref_feature_out)
        cudaMemcpy(d_ref_feature_out, p->d_fdec, D*(fH*fW)*2, cudaMemcpyDeviceToDevice);

    /* 14. z stream + mux */
    uint8_t* z_stream = NULL; size_t z_stream_size = 0;
    if (dcvc_rans_encoder_get_stream(rans_enc, &z_stream, &z_stream_size) != 0) {
        free(y_stream); return DCVC_RT_ERR_ENTROPY; }
    { size_t total = 4 + z_stream_size + y_stream_size;
      uint8_t* buf = malloc(total);
      if (!buf) { free(z_stream); free(y_stream); return DCVC_RT_ERR_OOM; }
      uint32_t zlen = (uint32_t)z_stream_size;
      memcpy(buf, &zlen, 4);
      if (z_stream_size) memcpy(buf+4, z_stream, z_stream_size);
      if (y_stream_size) memcpy(buf+4+z_stream_size, y_stream, y_stream_size);
      free(z_stream); free(y_stream);
      *out_stream = buf; *out_size = total; }
    return DCVC_RT_OK;
}

/* ===================================================================== */
DcvcRtStatus dcvc_inter_decode(DcvcInterPipeline* p, DcvcArCodec* ar_codec,
                               DcvcRansDecoder* rans_dec,
                               const uint8_t* stream, size_t stream_size,
                               const void* d_ref_feature, const void* d_ref_pixels,
                               int H, int W, int qp,
                               void* d_xhat_out, void* d_ref_feature_out)
{
    if (!p || !ar_codec || !rans_dec || !stream || (!d_ref_feature && !d_ref_pixels) || !d_xhat_out)
        return DCVC_RT_ERR_INVALID_ARG;
    DcvcRtStatus st = ips_ensure(p, H, W);
    if (st != DCVC_RT_OK) return st;

    const int D = DCVC_RT_CH_D, YC = DCVC_RT_CH_Y_INTER, ZC = DCVC_RT_CH_Z, RECON = 320;
    int fH = p->fH, fW = p->fW, yH = p->yH, yW = p->yW, zH = p->zH, zW = p->zW;
    int yHW = yH * yW, zHW = zH * zW;

    uint32_t z_len = 0;
    if (stream_size < 4) return DCVC_RT_ERR_BITSTREAM;
    memcpy(&z_len, stream, 4);
    if (4 + (size_t)z_len > stream_size) return DCVC_RT_ERR_BITSTREAM;
    const uint8_t* z_payload = stream + 4;
    const uint8_t* y_payload = z_payload + z_len;
    size_t y_size = stream_size - 4 - z_len;

    if ((st = load_qp(p, "inter_q_decoder", qp, p->d_q_dec, D)) != DCVC_RT_OK) return st;
    if ((st = load_qp(p, "inter_q_feature", qp, p->d_q_feat, D)) != DCVC_RT_OK) return st;
    if ((st = load_qp(p, "inter_q_recon", qp, p->d_q_recon, RECON)) != DCVC_RT_OK) return st;
    if ((st = load_z_cdfs(p, 0, rans_dec)) != DCVC_RT_OK) return st;

    /* 1. z rANS decode → z_hat */
    { int zt = ZC * zHW;
      dcvc_rans_decoder_set_stream(rans_dec, z_payload, z_len);
      dcvc_rans_decoder_decode_z(rans_dec, zt, 0, qp * ZC, zHW);
      int8_t* syms = NULL; size_t sn = 0;
      dcvc_rans_decoder_get_symbols(rans_dec, (int8_t**)&syms, &sn);
      for (size_t i = 0; i < sn; i++) p->z_hat_h[i] = f32h((float)syms[i]);
      cudaMemcpy(p->d_zhat, p->z_hat_h, ZC*zHW*2, cudaMemcpyHostToDevice); }

    /* 2. ref feature */
    st = build_ref_feature(p, d_ref_feature, d_ref_pixels, H, W);
    if (st != DCVC_RT_OK) return st

    /* 3. fe1(feature, ones) → x1 ; ctx_t */
    { int32_t fd[4]={1,D,fH,fW}, qd[4]={1,D,1,1};
      st = run2(p->eng[E_FE1], "in0", fd, p->d_feature, "q", qd, p->d_q_ones, p->d_x1);
      if (st != DCVC_RT_OK) return st }
    mul_qfeat_broadcast(p);

    /* 4. hyper_dec(z_hat) → hier */
    { int32_t zd[4]={1,ZC,zH,zW};
      st = run1(p->eng[E_HDEC], "in0", zd, p->d_zhat, p->d_hier);
      if (st != DCVC_RT_OK) return st }

    /* 5. temporal_prior(ctx_t) → temporal */
    { int32_t cd[4]={1,D,fH,fW};
      st = run1(p->eng[E_TEMP], "in0", cd, p->d_ctx_t, p->d_temporal);
      if (st != DCVC_RT_OK) return st }

    /* 6. cat → prior_fusion → params */
    { cudaMemcpy(p->d_pf_in, p->d_hier, YC*yHW*2, cudaMemcpyDeviceToDevice);
      cudaMemcpy((char*)p->d_pf_in+(size_t)YC*yHW*2, p->d_temporal, (YC*2)*yHW*2, cudaMemcpyDeviceToDevice);
      int32_t pd[4]={1,YC*3,yH,yW};
      st = run1(p->eng[E_PFUS], "in0", pd, p->d_pf_in, p->d_params);
      if (st != DCVC_RT_OK) return st }

    /* 7. AR(2x) decode */
    st = dcvc_ar_codec_decode(ar_codec, p->d_params, yH, yW, y_payload, y_size, p->d_yhat);
    if (st != DCVC_RT_OK) return st;

    /* 8. fe2(x1) → ctx */
    { int32_t fd[4]={1,D,fH,fW};
      st = run1(p->eng[E_FE2], "in0", fd, p->d_x1, p->d_ctx);
      if (st != DCVC_RT_OK) return st }

    /* 9. decoder(y_hat, ctx, q_dec) → dec_feature */
    { int32_t yd[4]={1,YC,yH,yW}, cd[4]={1,D,fH,fW}, qd[4]={1,D,1,1};
      st = run3(p->eng[E_DEC], "in0", yd, p->d_yhat, "ctx", cd, p->d_ctx, "q", qd, p->d_q_dec, p->d_fdec);
      if (st != DCVC_RT_OK) return st }

    /* 10. recon → x_hat */
    { int32_t fd[4]={1,D,fH,fW}, qd[4]={1,RECON,1,1};
      st = run2(p->eng[E_RECON], "in0", fd, p->d_fdec, "q", qd, p->d_q_recon, d_xhat_out);
      if (st != DCVC_RT_OK) return st }

    if (d_ref_feature_out)
        cudaMemcpy(d_ref_feature_out, p->d_fdec, D*(fH*fW)*2, cudaMemcpyDeviceToDevice);
    return DCVC_RT_OK;
}
