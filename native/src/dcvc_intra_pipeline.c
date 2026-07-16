// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Full intra-frame pipeline: image → analysis → hyper_enc → z rANS →
// hyper_dec → prior_fusion → AR codec → synthesis → image.

#include "dcvc_intra_pipeline.h"
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
static void pn_npy_free(PnNpy* o) { free(o->i32); free(o->fp16); o->i32 = o->fp16 = NULL; }

/* ---- round_to_int8 kernel signature ---- */
typedef void (*fn_round_to_int8)(const void*, void*, int8_t*, int, cudaStream_t);

/* ---- Engine binding helper: set shape+addr for in0 and out, then execute ---- */
static DcvcRtStatus run_engine_io2(DcvcTrtEngine* eng, const char* in0_name,
                                   const int32_t* in0_dims, void* in0_ptr,
                                   const char* in1_name,
                                   const int32_t* in1_dims, void* in1_ptr,
                                   void* out_ptr)
{
    DcvcRtStatus st;
    st = dcvc_trt_engine_set_shape(eng, in0_name, in0_dims, 4);
    if (st != DCVC_RT_OK) return st;
    st = dcvc_trt_engine_set_addr(eng, in0_name, in0_ptr);
    if (st != DCVC_RT_OK) return st;
    if (in1_name) {
        st = dcvc_trt_engine_set_shape(eng, in1_name, in1_dims, 4);
        if (st != DCVC_RT_OK) return st;
        st = dcvc_trt_engine_set_addr(eng, in1_name, in1_ptr);
        if (st != DCVC_RT_OK) return st;
    }
    int nio = dcvc_trt_engine_num_io(eng);
    const char* out_name = dcvc_trt_engine_tensor_name(eng, nio - 1);
    st = dcvc_trt_engine_set_addr(eng, out_name, out_ptr);
    if (st != DCVC_RT_OK) return st;
    return dcvc_trt_engine_execute(eng, NULL);
}

/* ---- Pipeline state: cached CUDA buffers and loaded engines ---- */
/* Dimensions: H,W are padded image dims. */
/* y is at H/16 × W/16, z is at H/64 × W/64 */
typedef struct {
    int H, W;
    int yh, yw;   /* H/16, W/16 */
    int zh, zw;   /* H/64, W/64 */
    int N, ZC;    /* 256, 128 */

    /* CUDA device buffers */
    void *d_y, *d_z, *d_zhat, *d_params, *d_params_fusion;
    void *d_qenc, *d_qdec;
    void* d_z_int8;  /* device buffer for z int8 symbols */
    int8_t* z_int8_h;
    uint16_t* z_hat_h;

    /* CUDA kernel */
    void* kernel_so;
    fn_round_to_int8 k_round_to_int8;
} PipelineState;

static PipelineState g_state = {0};

static DcvcRtStatus ps_ensure(int H, int W)
{
    if (g_state.H == H && g_state.W == W && g_state.d_y) return DCVC_RT_OK;

    /* Free old buffers */
    if (g_state.d_y) {
        cudaFree(g_state.d_y); cudaFree(g_state.d_z); cudaFree(g_state.d_zhat);
        cudaFree(g_state.d_params); cudaFree(g_state.d_params_fusion);
        cudaFree(g_state.d_qenc); cudaFree(g_state.d_qdec); cudaFree(g_state.d_z_int8);
        free(g_state.z_int8_h); free(g_state.z_hat_h);
        memset(&g_state, 0, sizeof(g_state));
    }

    g_state.H = H; g_state.W = W;
    g_state.yh = H / 16; g_state.yw = W / 16;
    g_state.zh = H / 64; g_state.zw = W / 64;
    g_state.N = 256; g_state.ZC = 128;

    int yhw = g_state.yh * g_state.yw;
    int zhw = g_state.zh * g_state.zw;

    cudaMalloc(&g_state.d_y,             g_state.N * yhw * 2);
    cudaMalloc(&g_state.d_z,             g_state.ZC * zhw * 2);
    cudaMalloc(&g_state.d_zhat,          g_state.ZC * zhw * 2);
    cudaMalloc(&g_state.d_params,        g_state.N * yhw * 2);
    cudaMalloc(&g_state.d_params_fusion, (g_state.N * 2 + 2) * yhw * 2);
    cudaMalloc(&g_state.d_qenc,          368 * 2);    /* [1,368,1,1] g_ch_enc_dec */
    cudaMalloc(&g_state.d_qdec,          368 * 2);
    cudaMalloc(&g_state.d_z_int8,        g_state.ZC * zhw);
    g_state.z_int8_h = (int8_t*)malloc(g_state.ZC * zhw);
    g_state.z_hat_h  = (uint16_t*)malloc(g_state.ZC * zhw * 2);

    void* ptrs[] = {g_state.d_y, g_state.d_z, g_state.d_zhat, g_state.d_params,
                    g_state.d_params_fusion, g_state.d_qenc, g_state.d_qdec, g_state.d_z_int8};
    for (int i = 0; i < 8; i++) {
        if (!ptrs[i]) return DCVC_RT_ERR_OOM;
    }
    if (!g_state.z_int8_h || !g_state.z_hat_h) return DCVC_RT_ERR_OOM;

    /* Load round_to_int8 kernel lazily */
    if (!g_state.kernel_so) {
        /* Try standard plugin dir path */
        g_state.kernel_so = dlopen("native/build/plugin_demo/libdcvc_kernels.so", RTLD_NOW | RTLD_GLOBAL);
        if (!g_state.kernel_so) {
            /* Fallback: try relative to CWD */
            g_state.kernel_so = dlopen("libdcvc_kernels.so", RTLD_NOW | RTLD_GLOBAL);
        }
        if (g_state.kernel_so) {
            g_state.k_round_to_int8 = (fn_round_to_int8)dlsym(g_state.kernel_so, "dcvc_k_round_to_int8");
        }
    }

    return DCVC_RT_OK;
}

/* ---- Load QP scales into device buffers ---- */
static DcvcRtStatus load_qp_scales(DcvcTrtRunner* runner, int qp)
{
    uint16_t qenc[368], qdec[368];
    DcvcRtStatus st;

    st = dcvc_trt_load_qp_scale(runner, "intra_q_scale_enc", qp, qenc, 368);
    if (st != DCVC_RT_OK) return st;
    st = dcvc_trt_load_qp_scale(runner, "intra_q_scale_dec", qp, qdec, 368);
    if (st != DCVC_RT_OK) return st;

    cudaMemcpy(g_state.d_qenc, qenc, 368 * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(g_state.d_qdec, qdec, 368 * 2, cudaMemcpyHostToDevice);
    return DCVC_RT_OK;
}

/* ---- Load z CDFs from assets, cache indices ---- */
static int g_z_cdf_loaded_enc = 0;
static int g_z_cdf_loaded_dec = 0;

static DcvcRtStatus load_z_cdfs(DcvcTrtRunner* runner,
                                DcvcRansEncoder* rans_enc,
                                DcvcRansDecoder* rans_dec)
{
    (void)runner;
    int need_enc = rans_enc && !g_z_cdf_loaded_enc;
    int need_dec = rans_dec && !g_z_cdf_loaded_dec;
    if (!need_enc && !need_dec) return DCVC_RT_OK;

    char path[640];
    const char* ad = "native/assets"; /* will be overridden by runner's asset_dir */

    PnNpy zcdf, zlen, zoff;
    snprintf(path, sizeof(path), "%s/decode/bitest_cdf.npy", ad);
    if (pn_read_npy(path, &zcdf) != 0) return DCVC_RT_ERR_IO;
    snprintf(path, sizeof(path), "%s/decode/bitest_cdf_length.npy", ad);
    if (pn_read_npy(path, &zlen) != 0) { pn_npy_free(&zcdf); return DCVC_RT_ERR_IO; }
    snprintf(path, sizeof(path), "%s/decode/bitest_offset.npy", ad);
    if (pn_read_npy(path, &zoff) != 0) { pn_npy_free(&zcdf); pn_npy_free(&zlen); return DCVC_RT_ERR_IO; }

    /* Add to encoder and/or decoder (one may be NULL) */
    if (need_enc) {
        dcvc_rans_encoder_reset(rans_enc);
        dcvc_rans_encoder_add_cdf(rans_enc, zcdf.i32, zcdf.dims[0], zcdf.dims[1], zlen.i32, zoff.i32);
        g_z_cdf_loaded_enc = 1;
    }
    if (need_dec) {
        dcvc_rans_decoder_reset_cdf(rans_dec);
        dcvc_rans_decoder_add_cdf(rans_dec, zcdf.i32, zcdf.dims[0], zcdf.dims[1], zlen.i32, zoff.i32);
        g_z_cdf_loaded_dec = 1;
    }

    pn_npy_free(&zcdf); pn_npy_free(&zlen); pn_npy_free(&zoff);
    return DCVC_RT_OK;
}

/* ===================================================================== */
/* =========================== ENCODE ================================== */
/* ===================================================================== */

DcvcRtStatus dcvc_intra_encode(DcvcTrtRunner* runner, DcvcArCodec* ar_codec,
                               DcvcRansEncoder* rans_enc,
                               const void* d_image, int H, int W, int qp,
                               uint8_t** out_stream, size_t* out_size,
                               void* d_xhat_out)
{
    if (!runner || !ar_codec || !rans_enc || !d_image || !out_stream || !out_size)
        return DCVC_RT_ERR_INVALID_ARG;
    *out_stream = NULL; *out_size = 0;

    DcvcRtStatus st = ps_ensure(H, W);
    if (st != DCVC_RT_OK) return st;

    int N = g_state.N, ZC = g_state.ZC;
    int yhw = g_state.yh * g_state.yw;
    int zhw = g_state.zh * g_state.zw;

    /* Load QP scales */
    st = load_qp_scales(runner, qp);
    if (st != DCVC_RT_OK) return st;

    /* Load z CDFs */
    st = load_z_cdfs(runner, rans_enc, NULL);
    if (st != DCVC_RT_OK) return st;

    /* Ensure engines loaded */
    if (!dcvc_trt_runner_has_engine(runner, DCVC_ENG_INTRA_ANALYSIS))
        return DCVC_RT_ERR_NO_ENGINE;

    /* Step 1: intra_analysis: image [1,3,H,W] + q_enc [1,256,1,1] → y [1,256,H/16,W/16] */
    {
        DcvcTrtEngine* eng = dcvc_trt_runner_get_engine(runner, DCVC_ENG_INTRA_ANALYSIS, &st);
        if (!eng) return st;
        int32_t img_dims[4] = {1, 3, H, W};
        int32_t q_dims[4] = {1, 368, 1, 1};
        /* Analysis engine inputs: "x" and "q" */
        int nio = dcvc_trt_engine_num_io(eng);
        /* Find input names */
        const char* in0 = NULL, *in1 = NULL;
        int in_idx = 0;
        for (int j = 0; j < nio; j++) {
            if (dcvc_trt_engine_is_input(eng, j)) {
                const char* nm = dcvc_trt_engine_tensor_name(eng, j);
                if (in_idx == 0) in0 = nm;
                else if (in_idx == 1) in1 = nm;
                in_idx++;
            }
        }
        st = run_engine_io2(eng, in0, img_dims, (void*)d_image,
                            in1, q_dims, g_state.d_qenc, g_state.d_y);
        if (st != DCVC_RT_OK) return st;
    }

    /* Step 2: intra_hyper_enc: y → z [1,128,H/64,W/64] */
    {
        DcvcTrtEngine* eng = dcvc_trt_runner_get_engine(runner, DCVC_ENG_INTRA_HYPER, &st);
        if (!eng) return st;
        int32_t y_dims[4] = {1, N, g_state.yh, g_state.yw};
        st = run_engine_io2(eng, "in0", y_dims, g_state.d_y, NULL, NULL, NULL, g_state.d_z);
    }

    /* Step 3: round_to_int8: z → z_hat (fp16) + z_int8 */
    if (!g_state.k_round_to_int8) return DCVC_RT_ERR_IO;
    {
        int z_total = ZC * zhw;
       g_state.k_round_to_int8(g_state.d_z, g_state.d_zhat, (int8_t*)g_state.d_z_int8, z_total, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(g_state.z_int8_h, g_state.d_z_int8, z_total, cudaMemcpyDeviceToHost);
    }

    /* Step 4: z rANS encode */
    {
        int z_total = ZC * zhw;
        /* Reset pending symbol list each frame; CDFs persist in the encoder. */
        dcvc_rans_encoder_reset(rans_enc);
        int per_channel = zhw; /* zh * zw */
        int start_offset = qp * ZC;
        dcvc_rans_encoder_encode_z(rans_enc, g_state.z_int8_h, z_total, 0, start_offset, per_channel);
        dcvc_rans_encoder_flush(rans_enc);
    }

    /* Step 5: hyper_dec: z_hat → params [1,256,H/16,W/16] */
    {
        DcvcTrtEngine* eng = dcvc_trt_runner_get_engine(runner, DCVC_ENG_HYPER_DEC, &st);
        if (!eng) return st;
        int32_t z_dims[4] = {1, ZC, g_state.zh, g_state.zw};
        st = run_engine_io2(eng, "in0", z_dims, g_state.d_zhat, NULL, NULL, NULL, g_state.d_params);
    }

    /* Step 6: y_prior_fusion: params → params_fusion [1,514,H/16,W/16] */
    {
        DcvcTrtEngine* eng = dcvc_trt_runner_get_engine(runner, DCVC_ENG_INTRA_PRIOR_FUSION, &st);
        if (!eng) return st;
        int32_t p_dims[4] = {1, N, g_state.yh, g_state.yw};
        st = run_engine_io2(eng, "in0", p_dims, g_state.d_params, NULL, NULL, NULL, g_state.d_params_fusion);
    }

    /* Step 7: AR codec encode: y + params_fusion → y bitstream + optional y_hat */
    uint8_t* y_stream = NULL;
    size_t y_stream_size = 0;
    void* d_yhat = NULL;
    if (d_xhat_out) {
        cudaMalloc(&d_yhat, N * yhw * 2);
        if (!d_yhat) return DCVC_RT_ERR_OOM;
    }
    st = dcvc_ar_codec_encode(ar_codec, g_state.d_y, g_state.d_params_fusion,
                              g_state.yh, g_state.yw, &y_stream, &y_stream_size, d_yhat);
    if (st != DCVC_RT_OK) { free(y_stream); cudaFree(d_yhat); return st; }

    /* Step 8: Get z bitstream */
    uint8_t* z_stream = NULL;
    size_t z_stream_size = 0;
    if (dcvc_rans_encoder_get_stream(rans_enc, &z_stream, &z_stream_size) != 0) {
        free(y_stream); cudaFree(d_yhat); return DCVC_RT_ERR_ENTROPY;
    }

    /* Step 9: Mux: [z_len:u32][z_payload][y_payload] */
    {
        size_t total = 4 + z_stream_size + y_stream_size;
        uint8_t* buf = (uint8_t*)malloc(total);
        if (!buf) { free(y_stream); free(z_stream); cudaFree(d_yhat); return DCVC_RT_ERR_OOM; }
        uint32_t zlen = (uint32_t)z_stream_size;
        memcpy(buf, &zlen, 4);
        if (z_stream_size) memcpy(buf + 4, z_stream, z_stream_size);
        if (y_stream_size) memcpy(buf + 4 + z_stream_size, y_stream, y_stream_size);
        free(z_stream); free(y_stream);
        *out_stream = buf;
        *out_size = total;
    }

    /* Step 10: Optional synthesis for x_hat reconstruction */
    if (d_xhat_out && d_yhat) {
        DcvcTrtEngine* eng = dcvc_trt_runner_get_engine(runner, DCVC_ENG_INTRA_SYNTHESIS, &st);
        if (eng) {
            int32_t yhat_dims[4] = {1, N, g_state.yh, g_state.yw};
            int32_t qd_dims[4] = {1, 368, 1, 1};
            /* Synthesis inputs: "in0" (y_hat) and "in1" (q_dec) */
            int nio = dcvc_trt_engine_num_io(eng);
            const char* in0 = NULL, *in1 = NULL;
            int in_idx = 0;
            for (int j = 0; j < nio; j++) {
                if (dcvc_trt_engine_is_input(eng, j)) {
                    const char* nm = dcvc_trt_engine_tensor_name(eng, j);
                    if (in_idx == 0) in0 = nm;
                    else if (in_idx == 1) in1 = nm;
                    in_idx++;
                }
            }
        }
        cudaFree(d_yhat);
    }

    return DCVC_RT_OK;
}

/* ===================================================================== */
/* =========================== DECODE ================================== */
/* ===================================================================== */

DcvcRtStatus dcvc_intra_decode(DcvcTrtRunner* runner, DcvcArCodec* ar_codec,
                               DcvcRansDecoder* rans_dec,
                               const uint8_t* stream, size_t stream_size,
                               int H, int W, int qp,
                               void* d_xhat_out)
{
    if (!runner || !ar_codec || !rans_dec || !stream || !d_xhat_out)
        return DCVC_RT_ERR_INVALID_ARG;

    DcvcRtStatus st = ps_ensure(H, W);
    if (st != DCVC_RT_OK) return st;

    int N = g_state.N, ZC = g_state.ZC;
    int yhw = g_state.yh * g_state.yw;
    int zhw = g_state.zh * g_state.zw;

    /* Demux: [z_len:u32][z_payload][y_payload] */
    uint32_t z_len = 0;
    if (stream_size < 4) return DCVC_RT_ERR_BITSTREAM;
    memcpy(&z_len, stream, 4);
    if (4 + (size_t)z_len > stream_size) return DCVC_RT_ERR_BITSTREAM;
    const uint8_t* z_payload = stream + 4;
    const uint8_t* y_payload = z_payload + z_len;
    size_t y_payload_size = stream_size - 4 - z_len;

    /* Load QP scales */
    st = load_qp_scales(runner, qp);
    if (st != DCVC_RT_OK) return st;

    /* Step 1: z rANS decode */
    {
        load_z_cdfs(runner, NULL, rans_dec);
        dcvc_rans_decoder_set_stream(rans_dec, z_payload, z_len);
        int z_total = ZC * zhw;
        int per_channel = zhw;
        int start_offset = qp * ZC;
        dcvc_rans_decoder_decode_z(rans_dec, z_total, 0, start_offset, per_channel);
        int8_t* syms = NULL; size_t sym_n = 0;
        dcvc_rans_decoder_get_symbols(rans_dec, (int8_t**)&syms, &sym_n);
        /* Copy z symbols to device as fp16 z_hat */
        for (size_t i = 0; i < sym_n; i++)
            g_state.z_hat_h[i] = f32h((float)syms[i]);
        cudaMemcpy(g_state.d_zhat, g_state.z_hat_h, ZC * zhw * 2, cudaMemcpyHostToDevice);
    }

    /* Step 2: hyper_dec: z_hat → params */
    {
        DcvcTrtEngine* eng = dcvc_trt_runner_get_engine(runner, DCVC_ENG_HYPER_DEC, &st);
        if (!eng) return st;
        int32_t z_dims[4] = {1, ZC, g_state.zh, g_state.zw};
        st = run_engine_io2(eng, "in0", z_dims, g_state.d_zhat, NULL, NULL, NULL, g_state.d_params);
    }

    /* Step 3: y_prior_fusion: params → params_fusion */
    {
        DcvcTrtEngine* eng = dcvc_trt_runner_get_engine(runner, DCVC_ENG_INTRA_PRIOR_FUSION, &st);
        if (!eng) return st;
        int32_t p_dims[4] = {1, N, g_state.yh, g_state.yw};
        st = run_engine_io2(eng, "in0", p_dims, g_state.d_params, NULL, NULL, NULL, g_state.d_params_fusion);
    }

    /* Step 4: AR codec decode: params_fusion → y_hat */
    void* d_yhat = d_xhat_out; /* reuse xhat buffer temporarily — but sizes differ */
    /* Actually need separate y_hat buffer for AR codec output */
    void* d_yhat_buf;
    cudaMalloc(&d_yhat_buf, N * yhw * 2);
    if (!d_yhat_buf) return DCVC_RT_ERR_OOM;

    st = dcvc_ar_codec_decode(ar_codec, g_state.d_params_fusion,
                              g_state.yh, g_state.yw, y_payload, y_payload_size, d_yhat_buf);
    if (st != DCVC_RT_OK) { cudaFree(d_yhat_buf); return st; }

    /* Step 5: synthesis: y_hat + q_dec → x_hat */
    {
        DcvcTrtEngine* eng = dcvc_trt_runner_get_engine(runner, DCVC_ENG_INTRA_SYNTHESIS, &st);
        if (!eng) { cudaFree(d_yhat_buf); return st; }
        int32_t yhat_dims[4] = {1, N, g_state.yh, g_state.yw};
        int32_t qd_dims[4] = {1, 368, 1, 1};
        int nio = dcvc_trt_engine_num_io(eng);
        const char* in0 = NULL, *in1 = NULL;
        int in_idx = 0;
        for (int j = 0; j < nio; j++) {
            if (dcvc_trt_engine_is_input(eng, j)) {
                const char* nm = dcvc_trt_engine_tensor_name(eng, j);
                if (in_idx == 0) in0 = nm;
                else if (in_idx == 1) in1 = nm;
                in_idx++;
            }
        }
        st = run_engine_io2(eng, in0, yhat_dims, d_yhat_buf, in1, qd_dims, g_state.d_qdec, d_xhat_out);
        cudaFree(d_yhat_buf);
    }

    return DCVC_RT_OK;
}
