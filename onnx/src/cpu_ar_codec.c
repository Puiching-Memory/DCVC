/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Pure-CPU AR prior codec implementation (intra 4-pass).
 */
#include "cpu_ar_codec.h"
#include "npy_reader.h"
#include "onnx_engine.h"
#include "rans_c.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- Structures ---------- */

typedef struct {
    int hw;
    int n_ch;
    int r_ch;     /* n_ch / 4 */
    int p_ch;     /* 2*n_ch + 2 */
    float* qenc;     /* HW */
    float* qdec;     /* HW */
    float* scales;   /* n_ch*HW */
    float* means;    /* n_ch*HW */
    float* common;   /* n_ch*HW */
    float* sr;       /* r_ch*HW */
    float* smask;     /* n_ch*HW */
    float* yq_full;  /* n_ch*HW */
    float* yq;       /* n_ch*HW */
    float* yq_w;    /* r_ch*HW */
    int16_t* packed; /* r_ch*HW */
    uint8_t* indexes; /* r_ch*HW */
    float* yhat_step; /* n_ch*HW */
    float* cat;      /* 2*n_ch*HW */
    float* sp_out;   /* 2*n_ch*HW */
    float* yhat;     /* n_ch*HW */
    float* masks[4]; /* n_ch*HW each */
    int8_t* syms_i8; /* r_ch*HW */
} ArWorkspace;

struct DcvcCpuArCodec {
    int n_ch;
    int passes;
    DcvcCpuEngine* eng_reduction;
    DcvcCpuEngine* eng_adaptor[4];  /* 1..3 used */
    DcvcCpuEngine* eng_spatial_prior;
    DcvcRansEncoder* rans_enc;
    DcvcRansDecoder* rans_dec;
    ArWorkspace ws;

    DcvcNpy gcdf, glen, goff;
    int g_cdf_idx;

    float scale_min, scale_max, log_scale_min, log_step_recip;
};

static const int k_mask_pattern[4][4] = {
    {0, 1, 2, 3}, {3, 2, 1, 0}, {2, 3, 0, 1}, {1, 0, 3, 2},
};

/* ---------- Workspace ---------- */

static void ws_free(ArWorkspace* ws)
{
    if (!ws->hw) return;
    free(ws->qenc); free(ws->qdec); free(ws->scales); free(ws->means);
    free(ws->common); free(ws->smask); free(ws->sr); free(ws->yq_full); free(ws->yq); free(ws->yq_w);
    free(ws->packed); free(ws->indexes); free(ws->yhat_step); free(ws->cat);
    free(ws->sp_out); free(ws->yhat); free(ws->syms_i8);
    for (int i = 0; i < 4; i++) free(ws->masks[i]);
    memset(ws, 0, sizeof(*ws));
}

static DcvcCpuStatus ws_ensure(DcvcCpuArCodec* c, int H, int W)
{
    int hw = H * W;
    if (c->ws.hw == hw && c->ws.n_ch == c->n_ch) return DCVC_CPU_OK;
    ws_free(&c->ws);

    int nc = c->n_ch;
    int r = nc / 4;
    int pch = 2 * nc + 2;
    ArWorkspace* ws = &c->ws;
    ws->hw = hw; ws->n_ch = nc; ws->r_ch = r; ws->p_ch = pch;

    ws->qenc = (float*)malloc(hw * sizeof(float));
    ws->qdec = (float*)malloc(hw * sizeof(float));
    ws->scales = (float*)malloc(nc * hw * sizeof(float));
    ws->means = (float*)malloc(nc * hw * sizeof(float));
    ws->common = (float*)malloc(nc * hw * sizeof(float));
    ws->smask = (float*)malloc(nc * hw * sizeof(float));
    ws->sr = (float*)malloc(r * hw * sizeof(float));
    ws->yq_full = (float*)malloc(nc * hw * sizeof(float));
    ws->yq = (float*)malloc(nc * hw * sizeof(float));
    ws->yq_w = (float*)malloc(r * hw * sizeof(float));
    ws->packed = (int16_t*)malloc(r * hw * sizeof(int16_t));
    ws->indexes = (uint8_t*)malloc(r * hw);
    ws->yhat_step = (float*)malloc(nc * hw * sizeof(float));
    ws->cat = (float*)malloc(2 * nc * hw * sizeof(float));
    ws->sp_out = (float*)malloc(2 * nc * hw * sizeof(float));
    ws->yhat = (float*)malloc(nc * hw * sizeof(float));
    ws->syms_i8 = (int8_t*)malloc(r * hw);

    float** ptrs[] = {&ws->qenc, &ws->qdec, &ws->scales, &ws->means, &ws->common,
                      &ws->smask, &ws->sr, &ws->yq_full, &ws->yq, &ws->yq_w, (float**)&ws->packed,
                      (float**)&ws->indexes, &ws->yhat_step, &ws->cat, &ws->sp_out,
                      &ws->yhat, (float**)&ws->syms_i8};
    for (size_t i = 0; i < sizeof(ptrs)/sizeof(ptrs[0]); i++) {
        if (!*ptrs[i]) { ws_free(ws); return DCVC_CPU_ERR_OOM; }
    }

    for (int m = 0; m < 4; m++) {
        ws->masks[m] = (float*)malloc(nc * hw * sizeof(float));
        if (!ws->masks[m]) { ws_free(ws); return DCVC_CPU_ERR_OOM; }
        int q = nc / 4;
        for (int ch = 0; ch < nc; ch++) {
            int quarter = ch / q; if (quarter > 3) quarter = 3;
            int target = k_mask_pattern[m][quarter];
            for (int hh = 0; hh < H; hh++)
                for (int ww = 0; ww < W; ww++) {
                    int sp = (hh % 2) * 2 + (ww % 2);
                    ws->masks[m][ch * hw + hh * W + ww] = (sp == target) ? 1.0f : 0.0f;
                }
        }
    }

    return DCVC_CPU_OK;
}

/* ---------- Kernels ---------- */

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static void separate_prior_intra(const float* pf, float* qenc, float* qdec,
                                 float* scales, float* means, int nc, int hw)
{
    for (int i = 0; i < hw; i++) {
        qenc[i] = sigmoid(pf[i]) * 1.5f + 0.5f;
        qdec[i] = sigmoid(pf[hw + i]) * 1.5f + 0.5f;
    }
    for (int i = 0; i < nc * hw; i++) {
        scales[i] = pf[2 * hw + i];
        means[i] = pf[(2 + nc) * hw + i];
    }
}

static void broadcast_mul(const float* in, const float* q, float* out, int nc, int hw)
{
    for (int ch = 0; ch < nc; ch++)
        for (int i = 0; i < hw; i++)
            out[ch * hw + i] = in[ch * hw + i] * q[i];
}

static void sp4x(const float* x, float* out, int n)
{
    for (int i = 0; i < n; i++)
        out[i] = x[i] + x[i + n] + x[i + 2*n] + x[i + 3*n];
}

static void process_mask_yq(const float* y, const float* scales, const float* means,
                            const float* mask, float* yq, int n, float force_zero_thres)
{
    (void)scales;
    for (int i = 0; i < n; i++) {
        float fm = mask[i];
        float means_hat = means[i] * fm;
        float q = roundf((y[i] - means_hat) * fm);
        if (force_zero_thres > 0.0f && scales[i] * fm > force_zero_thres) q = 0.0f;
        if (q > 127.0f) q = 127.0f;
        if (q < -128.0f) q = -128.0f;
        yq[i] = q;
    }
}

static void restore_y_4x(const float* yq_r, const float* means, const float* mask,
                         float* out, int cyhw, int n)
{
    for (int i = 0; i < n; i++) {
        out[i] = (yq_r[i % cyhw] + means[i]) * mask[i];
    }
}

static void add_inplace(float* out, const float* step, int n)
{
    for (int i = 0; i < n; i++) out[i] += step[i];
}

static void build_index_dec(const float* scales, uint8_t* out,
                            float scale_min, float scale_max,
                            float log_scale_min, float log_step_recip, int n)
{
    for (int i = 0; i < n; i++) {
        float s = scales[i];
        if (s < scale_min) s = scale_min;
        if (s > scale_max) s = scale_max;
        float v = (logf(s) - log_scale_min) * log_step_recip;
        int idx = (int)floorf(v);
        if (idx < 0) idx = 0;
        if (idx > 127) idx = 127;
        out[i] = (uint8_t)idx;
    }
}

static void build_index_enc(const float* symbols, const float* scales, int16_t* out,
                            float scale_min, float scale_max,
                            float log_scale_min, float log_step_recip, int n)
{
    for (int i = 0; i < n; i++) {
        float s = scales[i];
        if (s < scale_min) s = scale_min;
        if (s > scale_max) s = scale_max;
        float v = (logf(s) - log_scale_min) * log_step_recip;
        int idx = (int)floorf(v);
        if (idx < 0) idx = 0;
        if (idx > 127) idx = 127;
        int sym = (int)roundf(symbols[i]);
        out[i] = (int16_t)((sym << 8) + idx);
    }
}

static void int8_to_fp32(const int8_t* in, float* out, int n)
{
    for (int i = 0; i < n; i++) out[i] = (float)in[i];
}

/* ---------- Engine helpers ---------- */

static DcvcCpuStatus run_engine(DcvcCpuEngine* eng,
                                 const float* in, int n, int c, int h, int w,
                                 float* out, int on, int oc, int oh, int ow)
{
    DcvcCpuTensorView inputs[1] = { { (void*)in, 0, n, c, h, w } };
    DcvcCpuTensorView outputs[1] = { { out, 0, on, oc, oh, ow } };
    return dcvc_cpu_engine_run(eng, inputs, 1, outputs, 1);
}

static DcvcCpuStatus run_spatial_prior(DcvcCpuArCodec* c, const float* yhat_so_far,
                                       int H, int W, int round)
{
    int nc = c->n_ch, hw = H * W;
    ArWorkspace* ws = &c->ws;

    /* cat(y_hat_so_far, common) */
    memcpy(ws->cat, yhat_so_far, nc * hw * sizeof(float));
    memcpy(ws->cat + nc * hw, ws->common, nc * hw * sizeof(float));

    DcvcCpuStatus st = run_engine(c->eng_adaptor[round], ws->cat, 1, 2 * nc, H, W,
                                  ws->sp_out, 1, 2 * nc, H, W);
    if (st != DCVC_CPU_OK) return st;

    st = run_engine(c->eng_spatial_prior, ws->sp_out, 1, 2 * nc, H, W,
                    ws->cat, 1, 2 * nc, H, W);
    return st;
}

/* ---------- Public API ---------- */

DcvcCpuArCodec* dcvc_cpu_ar_codec_create(const char* model_dir, int n_ch,
                                          DcvcCpuStatus* out_st)
{
    if (out_st) *out_st = DCVC_CPU_OK;
    if (n_ch != 256) { if (out_st) *out_st = DCVC_CPU_ERR_UNSUPPORTED; return NULL; }

    DcvcCpuArCodec* c = (DcvcCpuArCodec*)calloc(1, sizeof(*c));
    if (!c) { if (out_st) *out_st = DCVC_CPU_ERR_OOM; return NULL; }
    c->n_ch = n_ch;
    c->passes = 4;

    c->scale_min = 0.11f;
    c->scale_max = 16.0f;
    c->log_scale_min = logf(c->scale_min);
    c->log_step_recip = 1.0f / ((logf(c->scale_max) - c->log_scale_min) / 127.0f);

    char path[640];
    DcvcCpuStatus st = DCVC_CPU_OK;

    snprintf(path, sizeof(path), "%s/y_spatial_prior_reduction.onnx", model_dir);
    c->eng_reduction = dcvc_cpu_engine_create(path, 0, &st);
    if (!c->eng_reduction) goto fail;

    for (int i = 1; i <= 3; i++) {
        snprintf(path, sizeof(path), "%s/y_spatial_prior_adaptor_%d.onnx", model_dir, i);
        c->eng_adaptor[i] = dcvc_cpu_engine_create(path, 0, &st);
        if (!c->eng_adaptor[i]) goto fail;
    }

    snprintf(path, sizeof(path), "%s/y_spatial_prior.onnx", model_dir);
    c->eng_spatial_prior = dcvc_cpu_engine_create(path, 0, &st);
    if (!c->eng_spatial_prior) goto fail;

    snprintf(path, sizeof(path), "%s/gaussian_cdf.npy", model_dir);
    if (dcvc_npy_read(path, &c->gcdf) != 0) { st = DCVC_CPU_ERR_IO; goto fail; }
    snprintf(path, sizeof(path), "%s/gaussian_cdf_length.npy", model_dir);
    if (dcvc_npy_read(path, &c->glen) != 0) { st = DCVC_CPU_ERR_IO; goto fail; }
    snprintf(path, sizeof(path), "%s/gaussian_offset.npy", model_dir);
    if (dcvc_npy_read(path, &c->goff) != 0) { st = DCVC_CPU_ERR_IO; goto fail; }

    c->rans_enc = dcvc_rans_encoder_create();
    c->rans_dec = dcvc_rans_decoder_create();
    if (!c->rans_enc || !c->rans_dec) { st = DCVC_CPU_ERR_OOM; goto fail; }

    return c;

fail:
    dcvc_cpu_ar_codec_destroy(c);
    if (out_st) *out_st = st;
    return NULL;
}

void dcvc_cpu_ar_codec_destroy(DcvcCpuArCodec* c)
{
    if (!c) return;
    ws_free(&c->ws);
    if (c->rans_enc) dcvc_rans_encoder_destroy(c->rans_enc);
    if (c->rans_dec) dcvc_rans_decoder_destroy(c->rans_dec);
    if (c->eng_reduction) dcvc_cpu_engine_destroy(c->eng_reduction);
    for (int i = 1; i <= 3; i++)
        if (c->eng_adaptor[i]) dcvc_cpu_engine_destroy(c->eng_adaptor[i]);
    if (c->eng_spatial_prior) dcvc_cpu_engine_destroy(c->eng_spatial_prior);
    dcvc_npy_free(&c->gcdf); dcvc_npy_free(&c->glen); dcvc_npy_free(&c->goff);
    free(c);
}

DcvcCpuStatus dcvc_cpu_ar_codec_encode_y(DcvcCpuArCodec* c,
                                          const float* y, const float* params_fusion,
                                          int H, int W,
                                          uint8_t** out_stream, size_t* out_size,
                                          float* y_hat_out)
{
    if (!c || !y || !params_fusion || !out_stream || !out_size)
        return DCVC_CPU_ERR_INVALID_ARG;
    *out_stream = NULL; *out_size = 0;

    DcvcCpuStatus st = ws_ensure(c, H, W);
    if (st != DCVC_CPU_OK) return st;

    int nc = c->n_ch, hw = H * W, r = nc / 4;
    ArWorkspace* ws = &c->ws;

    separate_prior_intra(params_fusion, ws->qenc, ws->qdec, ws->scales, ws->means, nc, hw);

    st = run_engine(c->eng_reduction, params_fusion, 1, 2 * nc + 2, H, W,
                    ws->common, 1, nc, H, W);
    if (st != DCVC_CPU_OK) return st;

    broadcast_mul(y, ws->qenc, ws->yq_full, nc, hw);

    dcvc_rans_encoder_reset(c->rans_enc);
    c->g_cdf_idx = dcvc_rans_encoder_add_cdf(c->rans_enc, dcvc_npy_i32(&c->gcdf),
                        c->gcdf.dims[0], c->gcdf.dims[1],
                        dcvc_npy_i32(&c->glen), dcvc_npy_i32(&c->goff));

    memset(ws->yhat, 0, nc * hw * sizeof(float));

    for (int round = 0; round < c->passes; round++) {
        float* curr_scales = ws->scales;
        float* curr_means = ws->means;
        if (round > 0) {
            st = run_spatial_prior(c, ws->yhat, H, W, round);
            if (st != DCVC_CPU_OK) return st;
            curr_scales = ws->cat;
            curr_means = ws->cat + nc * hw;
        }

        float* mask = ws->masks[round];

        process_mask_yq(ws->yq_full, curr_scales, curr_means, mask, ws->yq, nc * hw, -1.0f);
        sp4x(ws->yq, ws->yq_w, r * hw);

        for (int i = 0; i < nc * hw; i++) ws->smask[i] = curr_scales[i] * mask[i];
        sp4x(ws->smask, ws->sr, r * hw);

        build_index_enc(ws->yq_w, ws->sr, ws->packed, c->scale_min, c->scale_max,
                        c->log_scale_min, c->log_step_recip, r * hw);

        dcvc_rans_encoder_encode_y(c->rans_enc, ws->packed, r * hw, c->g_cdf_idx);

        restore_y_4x(ws->yq_w, curr_means, mask, ws->yhat_step, r * hw, nc * hw);
        add_inplace(ws->yhat, ws->yhat_step, nc * hw);
    }

    broadcast_mul(ws->yhat, ws->qdec, ws->yhat, nc, hw);

    if (y_hat_out)
        memcpy(y_hat_out, ws->yhat, nc * hw * sizeof(float));

    dcvc_rans_encoder_flush(c->rans_enc);
    int rc = dcvc_rans_encoder_get_stream(c->rans_enc, out_stream, out_size);
    return rc == 0 ? DCVC_CPU_OK : DCVC_CPU_ERR_ONNX;
}

DcvcCpuStatus dcvc_cpu_ar_codec_decode_y(DcvcCpuArCodec* c,
                                          const float* params_fusion,
                                          int H, int W,
                                          const uint8_t* stream, size_t stream_size,
                                          float* y_hat_out)
{
    if (!c || !params_fusion || !stream || !y_hat_out)
        return DCVC_CPU_ERR_INVALID_ARG;

    DcvcCpuStatus st = ws_ensure(c, H, W);
    if (st != DCVC_CPU_OK) return st;

    int nc = c->n_ch, hw = H * W, r = nc / 4;
    ArWorkspace* ws = &c->ws;

    separate_prior_intra(params_fusion, ws->qenc, ws->qdec, ws->scales, ws->means, nc, hw);

    st = run_engine(c->eng_reduction, params_fusion, 1, 2 * nc + 2, H, W,
                    ws->common, 1, nc, H, W);
    if (st != DCVC_CPU_OK) return st;

    dcvc_rans_decoder_reset_cdf(c->rans_dec);
    c->g_cdf_idx = dcvc_rans_decoder_add_cdf(c->rans_dec, dcvc_npy_i32(&c->gcdf),
                        c->gcdf.dims[0], c->gcdf.dims[1],
                        dcvc_npy_i32(&c->glen), dcvc_npy_i32(&c->goff));
    dcvc_rans_decoder_set_stream(c->rans_dec, stream, stream_size);

    memset(ws->yhat, 0, nc * hw * sizeof(float));

    for (int round = 0; round < c->passes; round++) {
        float* curr_scales = ws->scales;
        float* curr_means = ws->means;
        if (round > 0) {
            st = run_spatial_prior(c, ws->yhat, H, W, round);
            if (st != DCVC_CPU_OK) return st;
            curr_scales = ws->cat;
            curr_means = ws->cat + nc * hw;
        }

        float* mask = ws->masks[round];

        for (int i = 0; i < nc * hw; i++) ws->smask[i] = curr_scales[i] * mask[i];
        sp4x(ws->smask, ws->sr, r * hw);

        build_index_dec(ws->sr, ws->indexes, c->scale_min, c->scale_max,
                        c->log_scale_min, c->log_step_recip, r * hw);

        dcvc_rans_decoder_decode_y(c->rans_dec, ws->indexes, r * hw, c->g_cdf_idx);
        int8_t* syms = NULL; size_t sym_n = 0;
        dcvc_rans_decoder_get_symbols(c->rans_dec, &syms, &sym_n);

        int8_to_fp32(syms, ws->yq_w, r * hw);
        restore_y_4x(ws->yq_w, curr_means, mask, ws->yhat_step, r * hw, nc * hw);
        add_inplace(ws->yhat, ws->yhat_step, nc * hw);
    }

    broadcast_mul(ws->yhat, ws->qdec, ws->yhat, nc, hw);
    memcpy(y_hat_out, ws->yhat, nc * hw * sizeof(float));
    return DCVC_CPU_OK;
}
