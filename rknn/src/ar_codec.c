#include "dcvc_rk/ar_codec.h"
#include "dcvc_rk/kernels.h"
#include "dcvc_rk/rknn_engine.h"
#include "fxp_scale_index.h"
#include "npy_reader.h"
#include "rans_c.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int hw, n_ch, r_ch, p_ch;
    float *qenc, *qdec, *scales, *means, *common;
    float *smask, *sr, *yq_full, *yq, *yq_w;
    float *yhat_step, *cat, *sp_out, *yhat;
    float *masks[4];
    int16_t* packed;
    uint8_t* indexes;
} ArWs;

struct DcvcRkArCodec {
    int n_ch, passes;
    DcvcRkEngine* eng_reduction;
    DcvcRkEngine* eng_adaptor[4];
    DcvcRkEngine* eng_spatial_prior;
    DcvcRansEncoder* rans_enc;
    DcvcRansDecoder* rans_dec;
    DcvcNpy gcdf, glen, goff;
    int g_cdf_idx;
    float skip_thres;
    ArWs ws;
    DcvcRkProfile prof;
    int64_t npu_us;
};

static void acc_eng(DcvcRkStageProf* s, DcvcRkEngine* e, int64_t* npu_acc)
{
    if (!s || !e) return;
    int64_t u = dcvc_rk_engine_last_run_us(e);
    int64_t su = dcvc_rk_engine_last_set_us(e);
    int64_t gu = dcvc_rk_engine_last_get_us(e);
    if (u > 0) { s->npu_ms += u / 1000.0; if (npu_acc) *npu_acc += u; }
    if (su > 0) s->set_ms += su / 1000.0;
    if (gu > 0) s->get_ms += gu / 1000.0;
    s->calls++;
}

static DcvcRkStatus run1(DcvcRkEngine* eng, const float* in, int c, int h, int w,
                         float* out, int oc, int oh, int ow)
{
    DcvcRkTensorView vin = { (void*)in, 1, c, h, w };
    DcvcRkTensorView vout = { out, 1, oc, oh, ow };
    return dcvc_rk_engine_run(eng, &vin, 1, &vout, 1);
}

static void ws_free(ArWs* ws)
{
    if (!ws->hw) return;
    free(ws->qenc); free(ws->qdec); free(ws->scales); free(ws->means);
    free(ws->common); free(ws->smask); free(ws->sr); free(ws->yq_full);
    free(ws->yq); free(ws->yq_w); free(ws->yhat_step); free(ws->cat);
    free(ws->sp_out); free(ws->yhat); free(ws->packed); free(ws->indexes);
    for (int i = 0; i < 4; i++) free(ws->masks[i]);
    memset(ws, 0, sizeof(*ws));
}

static DcvcRkStatus ws_ensure(DcvcRkArCodec* c, int H, int W)
{
    int hw = H * W;
    if (c->ws.hw == hw && c->ws.n_ch == c->n_ch) return DCVC_RK_OK;
    ws_free(&c->ws);
    int nc = c->n_ch;
    int r = (c->passes == 4) ? nc / 4 : nc / 2;
    int pch = (c->passes == 4) ? (2 * nc + 2) : (3 * nc);
    ArWs* ws = &c->ws;
    ws->hw = hw; ws->n_ch = nc; ws->r_ch = r; ws->p_ch = pch;
    #define A(p, n) do { p = (float*)calloc((size_t)(n), sizeof(float)); if (!p) return DCVC_RK_ERR_OOM; } while (0)
    A(ws->qenc, hw);
    A(ws->qdec, nc * hw);
    A(ws->scales, nc * hw);
    A(ws->means, nc * hw);
    A(ws->common, nc * hw);
    A(ws->smask, nc * hw);
    A(ws->sr, r * hw);
    A(ws->yq_full, nc * hw);
    A(ws->yq, nc * hw);
    A(ws->yq_w, r * hw);
    A(ws->yhat_step, nc * hw);
    A(ws->cat, 4 * nc * hw);
    A(ws->sp_out, 2 * nc * hw);
    A(ws->yhat, nc * hw);
    ws->packed = (int16_t*)calloc((size_t)r * hw, sizeof(int16_t));
    ws->indexes = (uint8_t*)calloc((size_t)r * hw, 1);
    if (!ws->packed || !ws->indexes) return DCVC_RK_ERR_OOM;
    int nm = (c->passes == 4) ? 4 : 2;
    for (int m = 0; m < nm; m++) A(ws->masks[m], nc * hw);
    #undef A
    if (c->passes == 4) dcvc_rk_fill_masks_4x(ws->masks, nc, H, W);
    else dcvc_rk_fill_masks_2x(ws->masks, nc, H, W);
    return DCVC_RK_OK;
}

static int path_rknn(char* buf, size_t n, const char* dir, const char* stem)
{
    return snprintf(buf, n, "%s/%s.rknn", dir, stem);
}

DcvcRkArCodec* dcvc_rk_ar_codec_create(const char* model_dir, int n_ch, int passes,
                                       DcvcRkStatus* out_st)
{
    if (out_st) *out_st = DCVC_RK_OK;
    if (!model_dir || (passes != 2 && passes != 4) ||
        (passes == 4 && n_ch != 256) || (passes == 2 && n_ch != 128)) {
        if (out_st) *out_st = DCVC_RK_ERR_INVALID_ARG;
        return NULL;
    }
    DcvcRkArCodec* c = (DcvcRkArCodec*)calloc(1, sizeof(*c));
    if (!c) { if (out_st) *out_st = DCVC_RK_ERR_OOM; return NULL; }
    c->n_ch = n_ch; c->passes = passes;
    { const char* e = getenv("DCVC_SKIP_THRES"); if (e) c->skip_thres = strtof(e, NULL); }

    char path[768];
    DcvcRkStatus st = DCVC_RK_OK;
    if (passes == 4) {
        path_rknn(path, sizeof(path), model_dir, "y_spatial_prior_reduction");
        c->eng_reduction = dcvc_rk_engine_create(path, &st);
        if (!c->eng_reduction) goto fail;
        for (int i = 1; i <= 3; i++) {
            char stem[64]; snprintf(stem, sizeof(stem), "y_spatial_prior_adaptor_%d", i);
            path_rknn(path, sizeof(path), model_dir, stem);
            c->eng_adaptor[i] = dcvc_rk_engine_create(path, &st);
            if (!c->eng_adaptor[i]) goto fail;
        }
        path_rknn(path, sizeof(path), model_dir, "y_spatial_prior");
        c->eng_spatial_prior = dcvc_rk_engine_create(path, &st);
        if (!c->eng_spatial_prior) goto fail;
    } else {
        path_rknn(path, sizeof(path), model_dir, "inter_spatial_prior");
        c->eng_spatial_prior = dcvc_rk_engine_create(path, &st);
        if (!c->eng_spatial_prior) goto fail;
    }

    snprintf(path, sizeof(path), "%s/gaussian_cdf.npy", model_dir);
    if (dcvc_npy_read(path, &c->gcdf) != 0) { st = DCVC_RK_ERR_IO; goto fail; }
    snprintf(path, sizeof(path), "%s/gaussian_cdf_length.npy", model_dir);
    if (dcvc_npy_read(path, &c->glen) != 0) { st = DCVC_RK_ERR_IO; goto fail; }
    snprintf(path, sizeof(path), "%s/gaussian_offset.npy", model_dir);
    if (dcvc_npy_read(path, &c->goff) != 0) { st = DCVC_RK_ERR_IO; goto fail; }

    c->rans_enc = dcvc_rans_encoder_create();
    c->rans_dec = dcvc_rans_decoder_create();
    if (!c->rans_enc || !c->rans_dec) { st = DCVC_RK_ERR_OOM; goto fail; }
    return c;
fail:
    dcvc_rk_ar_codec_destroy(c);
    if (out_st) *out_st = st;
    return NULL;
}

void dcvc_rk_ar_codec_destroy(DcvcRkArCodec* c)
{
    if (!c) return;
    ws_free(&c->ws);
    if (c->rans_enc) dcvc_rans_encoder_destroy(c->rans_enc);
    if (c->rans_dec) dcvc_rans_decoder_destroy(c->rans_dec);
    if (c->eng_reduction) dcvc_rk_engine_destroy(c->eng_reduction);
    for (int i = 1; i <= 3; i++)
        if (c->eng_adaptor[i]) dcvc_rk_engine_destroy(c->eng_adaptor[i]);
    if (c->eng_spatial_prior) dcvc_rk_engine_destroy(c->eng_spatial_prior);
    dcvc_npy_free(&c->gcdf); dcvc_npy_free(&c->glen); dcvc_npy_free(&c->goff);
    free(c);
}

const DcvcRkProfile* dcvc_rk_ar_last_profile(const DcvcRkArCodec* c)
{
    return c ? &c->prof : NULL;
}
int64_t dcvc_rk_ar_npu_us(const DcvcRkArCodec* c) { return c ? c->npu_us : 0; }

static DcvcRkStatus run_sp_4x(DcvcRkArCodec* c, const float* yhat, int H, int W, int round,
                              DcvcRkStageProf* s_adp, DcvcRkStageProf* s_sp)
{
    int nc = c->n_ch, hw = H * W;
    ArWs* ws = &c->ws;
    memcpy(ws->cat, yhat, (size_t)nc * hw * sizeof(float));
    memcpy(ws->cat + nc * hw, ws->common, (size_t)nc * hw * sizeof(float));
    DcvcRkStatus st = run1(c->eng_adaptor[round], ws->cat, 2 * nc, H, W, ws->sp_out, 2 * nc, H, W);
    if (st != DCVC_RK_OK) return st;
    acc_eng(s_adp, c->eng_adaptor[round], &c->npu_us);
    st = run1(c->eng_spatial_prior, ws->sp_out, 2 * nc, H, W, ws->cat, 2 * nc, H, W);
    if (st == DCVC_RK_OK) acc_eng(s_sp, c->eng_spatial_prior, &c->npu_us);
    return st;
}

static DcvcRkStatus run_sp_2x(DcvcRkArCodec* c, const float* yhat_step, const float* params,
                              int H, int W, DcvcRkStageProf* s_sp)
{
    int nc = c->n_ch, hw = H * W;
    ArWs* ws = &c->ws;
    memcpy(ws->cat, yhat_step, (size_t)nc * hw * sizeof(float));
    memcpy(ws->cat + nc * hw, params, (size_t)3 * nc * hw * sizeof(float));
    DcvcRkStatus st = run1(c->eng_spatial_prior, ws->cat, 4 * nc, H, W, ws->sp_out, 2 * nc, H, W);
    if (st == DCVC_RK_OK) acc_eng(s_sp, c->eng_spatial_prior, &c->npu_us);
    return st;
}

static DcvcRkStatus encode_4x(DcvcRkArCodec* c, const float* y, const float* pf,
                              int H, int W, uint8_t** out_stream, size_t* out_size,
                              float* y_hat_out)
{
    DcvcRkStatus st = ws_ensure(c, H, W);
    if (st != DCVC_RK_OK) return st;
    int nc = c->n_ch, hw = H * W, r = nc / 4;
    ArWs* ws = &c->ws;
    dcvc_rk_profile_reset(&c->prof);
    c->npu_us = 0;
    DcvcRkStageProf* s_red = dcvc_rk_profile_add(&c->prof, "ar.reduction");
    DcvcRkStageProf* s_adp = dcvc_rk_profile_add(&c->prof, "ar.adaptor");
    DcvcRkStageProf* s_sp  = dcvc_rk_profile_add(&c->prof, "ar.spatial_prior");
    DcvcRkStageProf* s_cpu = dcvc_rk_profile_add(&c->prof, "ar.cpu_kernels");
    DcvcRkStageProf* s_rans = dcvc_rk_profile_add(&c->prof, "ar.rans_y");

    double t_all = dcvc_rk_now_ms();
    double t = dcvc_rk_now_ms();
    dcvc_rk_separate_prior_intra(pf, ws->qenc, ws->qdec, ws->scales, ws->means, nc, hw);
    st = run1(c->eng_reduction, pf, 2 * nc + 2, H, W, ws->common, nc, H, W);
    if (st != DCVC_RK_OK) return st;
    acc_eng(s_red, c->eng_reduction, &c->npu_us);
    if (s_red) s_red->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    dcvc_rk_mul_q_pixel(y, ws->qenc, ws->yq_full, nc, hw);
    if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    dcvc_rans_encoder_reset(c->rans_enc);
    c->g_cdf_idx = dcvc_rans_encoder_add_cdf(c->rans_enc, dcvc_npy_i32(&c->gcdf),
                        c->gcdf.dims[0], c->gcdf.dims[1],
                        dcvc_npy_i32(&c->glen), dcvc_npy_i32(&c->goff));
    if (s_rans) s_rans->wall_ms += dcvc_rk_now_ms() - t;
    memset(ws->yhat, 0, (size_t)nc * hw * sizeof(float));

    for (int round = 0; round < 4; round++) {
        float *curr_s = ws->scales, *curr_m = ws->means;
        if (round > 0) {
            t = dcvc_rk_now_ms();
            st = run_sp_4x(c, ws->yhat, H, W, round, s_adp, s_sp);
            double dt = dcvc_rk_now_ms() - t;
            if (s_adp) s_adp->wall_ms += dt * 0.5;
            if (s_sp) s_sp->wall_ms += dt * 0.5;
            if (st != DCVC_RK_OK) return st;
            curr_s = ws->cat; curr_m = ws->cat + nc * hw;
        }
        float* mask = ws->masks[round];
        t = dcvc_rk_now_ms();
        dcvc_rk_elem_mul(curr_s, mask, ws->smask, nc * hw);
        dcvc_rk_sp4x(ws->smask, ws->sr, r * hw);
        dcvc_rk_process_mask_yq(ws->yq_full, curr_s, curr_m, mask, ws->yq, nc * hw, c->skip_thres);
        dcvc_rk_sp4x(ws->yq, ws->yq_w, r * hw);
        dcvc_build_index_enc_i(ws->yq_w, ws->sr, ws->packed, r * hw);
        if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;

        t = dcvc_rk_now_ms();
        dcvc_rans_encoder_encode_y(c->rans_enc, ws->packed, r * hw, c->g_cdf_idx);
        if (s_rans) s_rans->wall_ms += dcvc_rk_now_ms() - t;

        t = dcvc_rk_now_ms();
        dcvc_rk_restore_y_nx(ws->yq_w, curr_m, mask, ws->yhat_step, r * hw, nc * hw);
        dcvc_rk_add_inplace(ws->yhat, ws->yhat_step, nc * hw);
        if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;
    }
    t = dcvc_rk_now_ms();
    dcvc_rk_mul_q_pixel(ws->yhat, ws->qdec, ws->yhat, nc, hw);
    if (y_hat_out) memcpy(y_hat_out, ws->yhat, (size_t)nc * hw * sizeof(float));
    if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    dcvc_rans_encoder_flush(c->rans_enc);
    st = dcvc_rans_encoder_get_stream(c->rans_enc, out_stream, out_size) == 0
               ? DCVC_RK_OK : DCVC_RK_ERR_ENTROPY;
    if (s_rans) s_rans->wall_ms += dcvc_rk_now_ms() - t;
    (void)t_all;
    return st;
}

static DcvcRkStatus decode_4x(DcvcRkArCodec* c, const float* pf, int H, int W,
                              const uint8_t* stream, size_t stream_size, float* y_hat_out)
{
    DcvcRkStatus st = ws_ensure(c, H, W);
    if (st != DCVC_RK_OK) return st;
    int nc = c->n_ch, hw = H * W, r = nc / 4;
    ArWs* ws = &c->ws;
    dcvc_rk_profile_reset(&c->prof);
    c->npu_us = 0;
    DcvcRkStageProf* s_red = dcvc_rk_profile_add(&c->prof, "ar.reduction");
    DcvcRkStageProf* s_adp = dcvc_rk_profile_add(&c->prof, "ar.adaptor");
    DcvcRkStageProf* s_sp  = dcvc_rk_profile_add(&c->prof, "ar.spatial_prior");
    DcvcRkStageProf* s_cpu = dcvc_rk_profile_add(&c->prof, "ar.cpu_kernels");
    DcvcRkStageProf* s_rans = dcvc_rk_profile_add(&c->prof, "ar.rans_y");

    double t = dcvc_rk_now_ms();
    dcvc_rk_separate_prior_intra(pf, ws->qenc, ws->qdec, ws->scales, ws->means, nc, hw);
    st = run1(c->eng_reduction, pf, 2 * nc + 2, H, W, ws->common, nc, H, W);
    if (st != DCVC_RK_OK) return st;
    acc_eng(s_red, c->eng_reduction, &c->npu_us);
    if (s_red) s_red->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    dcvc_rans_decoder_reset_cdf(c->rans_dec);
    c->g_cdf_idx = dcvc_rans_decoder_add_cdf(c->rans_dec, dcvc_npy_i32(&c->gcdf),
                        c->gcdf.dims[0], c->gcdf.dims[1],
                        dcvc_npy_i32(&c->glen), dcvc_npy_i32(&c->goff));
    dcvc_rans_decoder_set_stream(c->rans_dec, stream, stream_size);
    if (s_rans) s_rans->wall_ms += dcvc_rk_now_ms() - t;
    memset(ws->yhat, 0, (size_t)nc * hw * sizeof(float));

    for (int round = 0; round < 4; round++) {
        float *curr_s = ws->scales, *curr_m = ws->means;
        if (round > 0) {
            t = dcvc_rk_now_ms();
            st = run_sp_4x(c, ws->yhat, H, W, round, s_adp, s_sp);
            double dt = dcvc_rk_now_ms() - t;
            if (s_adp) s_adp->wall_ms += dt * 0.5;
            if (s_sp) s_sp->wall_ms += dt * 0.5;
            if (st != DCVC_RK_OK) return st;
            curr_s = ws->cat; curr_m = ws->cat + nc * hw;
        }
        float* mask = ws->masks[round];
        t = dcvc_rk_now_ms();
        dcvc_rk_elem_mul(curr_s, mask, ws->smask, nc * hw);
        dcvc_rk_sp4x(ws->smask, ws->sr, r * hw);
        dcvc_build_index_dec_i(ws->sr, ws->indexes, r * hw);
        if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;

        t = dcvc_rk_now_ms();
        dcvc_rans_decoder_decode_y(c->rans_dec, ws->indexes, r * hw, c->g_cdf_idx);
        int8_t* syms = NULL; size_t sn = 0;
        dcvc_rans_decoder_get_symbols(c->rans_dec, &syms, &sn);
        for (int i = 0; i < r * hw; i++) ws->yq_w[i] = (float)syms[i];
        free(syms);
        if (s_rans) s_rans->wall_ms += dcvc_rk_now_ms() - t;

        t = dcvc_rk_now_ms();
        dcvc_rk_restore_y_nx(ws->yq_w, curr_m, mask, ws->yhat_step, r * hw, nc * hw);
        dcvc_rk_add_inplace(ws->yhat, ws->yhat_step, nc * hw);
        if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;
    }
    if (dcvc_rans_decoder_has_error(c->rans_dec)) return DCVC_RK_ERR_ENTROPY;
    t = dcvc_rk_now_ms();
    dcvc_rk_mul_q_pixel(ws->yhat, ws->qdec, y_hat_out, nc, hw);
    if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;
    return DCVC_RK_OK;
}

static DcvcRkStatus encode_2x(DcvcRkArCodec* c, const float* y, const float* params,
                              int H, int W, uint8_t** out_stream, size_t* out_size,
                              float* y_hat_out)
{
    DcvcRkStatus st = ws_ensure(c, H, W);
    if (st != DCVC_RK_OK) return st;
    int nc = c->n_ch, hw = H * W, r = nc / 2;
    ArWs* ws = &c->ws;
    dcvc_rk_profile_reset(&c->prof);
    c->npu_us = 0;
    DcvcRkStageProf* s_sp  = dcvc_rk_profile_add(&c->prof, "ar.spatial_prior");
    DcvcRkStageProf* s_cpu = dcvc_rk_profile_add(&c->prof, "ar.cpu_kernels");
    DcvcRkStageProf* s_rans = dcvc_rk_profile_add(&c->prof, "ar.rans_y");

    double t = dcvc_rk_now_ms();
    dcvc_rk_separate_prior_video_enc(params, ws->qdec, ws->scales, ws->means, nc, hw);
    for (int i = 0; i < nc * hw; i++) {
        float q = ws->qdec[i] < 0.5f ? 0.5f : ws->qdec[i];
        ws->qdec[i] = q;
        ws->yq_full[i] = y[i] / q;
    }
    if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    dcvc_rans_encoder_reset(c->rans_enc);
    c->g_cdf_idx = dcvc_rans_encoder_add_cdf(c->rans_enc, dcvc_npy_i32(&c->gcdf),
                        c->gcdf.dims[0], c->gcdf.dims[1],
                        dcvc_npy_i32(&c->glen), dcvc_npy_i32(&c->goff));
    if (s_rans) s_rans->wall_ms += dcvc_rk_now_ms() - t;

    float *curr_s = ws->scales, *curr_m = ws->means;
    float* mask0 = ws->masks[0];
    t = dcvc_rk_now_ms();
    dcvc_rk_process_mask_yq(ws->yq_full, curr_s, curr_m, mask0, ws->yq, nc * hw, c->skip_thres);
    dcvc_rk_sp2x(ws->yq, ws->yq_w, r * hw);
    dcvc_rk_elem_mul(curr_s, mask0, ws->smask, nc * hw);
    dcvc_rk_sp2x(ws->smask, ws->sr, r * hw);
    dcvc_build_index_enc_i(ws->yq_w, ws->sr, ws->packed, r * hw);
    if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    dcvc_rans_encoder_encode_y(c->rans_enc, ws->packed, r * hw, c->g_cdf_idx);
    if (s_rans) s_rans->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    dcvc_rk_restore_y_nx(ws->yq_w, curr_m, mask0, ws->yhat_step, r * hw, nc * hw);
    if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    st = run_sp_2x(c, ws->yhat_step, params, H, W, s_sp);
    if (s_sp) s_sp->wall_ms += dcvc_rk_now_ms() - t;
    if (st != DCVC_RK_OK) return st;
    curr_s = ws->sp_out; curr_m = ws->sp_out + nc * hw;

    float* mask1 = ws->masks[1];
    t = dcvc_rk_now_ms();
    dcvc_rk_process_mask_yq(ws->yq_full, curr_s, curr_m, mask1, ws->yq, nc * hw, c->skip_thres);
    dcvc_rk_sp2x(ws->yq, ws->yq_w, r * hw);
    dcvc_rk_elem_mul(curr_s, mask1, ws->smask, nc * hw);
    dcvc_rk_sp2x(ws->smask, ws->sr, r * hw);
    dcvc_build_index_enc_i(ws->yq_w, ws->sr, ws->packed, r * hw);
    if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    dcvc_rans_encoder_encode_y(c->rans_enc, ws->packed, r * hw, c->g_cdf_idx);
    if (s_rans) s_rans->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    dcvc_rk_restore_y_nx(ws->yq_w, curr_m, mask1, ws->cat, r * hw, nc * hw);
    for (int i = 0; i < nc * hw; i++)
        ws->yhat[i] = (ws->yhat_step[i] + ws->cat[i]) * ws->qdec[i];
    if (y_hat_out) memcpy(y_hat_out, ws->yhat, (size_t)nc * hw * sizeof(float));
    if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    dcvc_rans_encoder_flush(c->rans_enc);
    st = dcvc_rans_encoder_get_stream(c->rans_enc, out_stream, out_size) == 0
               ? DCVC_RK_OK : DCVC_RK_ERR_ENTROPY;
    if (s_rans) s_rans->wall_ms += dcvc_rk_now_ms() - t;
    return st;
}

static DcvcRkStatus decode_2x(DcvcRkArCodec* c, const float* params, int H, int W,
                              const uint8_t* stream, size_t stream_size, float* y_hat_out)
{
    DcvcRkStatus st = ws_ensure(c, H, W);
    if (st != DCVC_RK_OK) return st;
    int nc = c->n_ch, hw = H * W, r = nc / 2;
    ArWs* ws = &c->ws;
    dcvc_rk_profile_reset(&c->prof);
    c->npu_us = 0;
    DcvcRkStageProf* s_sp  = dcvc_rk_profile_add(&c->prof, "ar.spatial_prior");
    DcvcRkStageProf* s_cpu = dcvc_rk_profile_add(&c->prof, "ar.cpu_kernels");
    DcvcRkStageProf* s_rans = dcvc_rk_profile_add(&c->prof, "ar.rans_y");

    double t = dcvc_rk_now_ms();
    dcvc_rk_separate_prior_video_dec(params, ws->qdec, ws->scales, ws->means, nc, hw);
    if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    dcvc_rans_decoder_reset_cdf(c->rans_dec);
    c->g_cdf_idx = dcvc_rans_decoder_add_cdf(c->rans_dec, dcvc_npy_i32(&c->gcdf),
                        c->gcdf.dims[0], c->gcdf.dims[1],
                        dcvc_npy_i32(&c->glen), dcvc_npy_i32(&c->goff));
    dcvc_rans_decoder_set_stream(c->rans_dec, stream, stream_size);
    if (s_rans) s_rans->wall_ms += dcvc_rk_now_ms() - t;

    float *curr_s = ws->scales, *curr_m = ws->means;
    float* mask0 = ws->masks[0];
    t = dcvc_rk_now_ms();
    dcvc_rk_elem_mul(curr_s, mask0, ws->smask, nc * hw);
    dcvc_rk_sp2x(ws->smask, ws->sr, r * hw);
    dcvc_build_index_dec_i(ws->sr, ws->indexes, r * hw);
    if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    dcvc_rans_decoder_decode_y(c->rans_dec, ws->indexes, r * hw, c->g_cdf_idx);
    { int8_t* syms = NULL; size_t sn = 0;
      dcvc_rans_decoder_get_symbols(c->rans_dec, &syms, &sn);
      for (int i = 0; i < r * hw; i++) ws->yq_w[i] = (float)syms[i];
      free(syms); }
    if (s_rans) s_rans->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    dcvc_rk_restore_y_nx(ws->yq_w, curr_m, mask0, ws->yhat_step, r * hw, nc * hw);
    if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    st = run_sp_2x(c, ws->yhat_step, params, H, W, s_sp);
    if (s_sp) s_sp->wall_ms += dcvc_rk_now_ms() - t;
    if (st != DCVC_RK_OK) return st;
    curr_s = ws->sp_out; curr_m = ws->sp_out + nc * hw;

    float* mask1 = ws->masks[1];
    t = dcvc_rk_now_ms();
    dcvc_rk_elem_mul(curr_s, mask1, ws->smask, nc * hw);
    dcvc_rk_sp2x(ws->smask, ws->sr, r * hw);
    dcvc_build_index_dec_i(ws->sr, ws->indexes, r * hw);
    if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    dcvc_rans_decoder_decode_y(c->rans_dec, ws->indexes, r * hw, c->g_cdf_idx);
    { int8_t* syms = NULL; size_t sn = 0;
      dcvc_rans_decoder_get_symbols(c->rans_dec, &syms, &sn);
      for (int i = 0; i < r * hw; i++) ws->yq_w[i] = (float)syms[i];
      free(syms); }
    if (s_rans) s_rans->wall_ms += dcvc_rk_now_ms() - t;

    t = dcvc_rk_now_ms();
    dcvc_rk_restore_y_nx(ws->yq_w, curr_m, mask1, ws->cat, r * hw, nc * hw);
    if (dcvc_rans_decoder_has_error(c->rans_dec)) return DCVC_RK_ERR_ENTROPY;
    for (int i = 0; i < nc * hw; i++)
        y_hat_out[i] = (ws->yhat_step[i] + ws->cat[i]) * ws->qdec[i];
    if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t;
    return DCVC_RK_OK;
}

DcvcRkStatus dcvc_rk_ar_encode_y(DcvcRkArCodec* c, const float* y, const float* params,
                                 int H, int W, uint8_t** out_stream, size_t* out_size,
                                 float* y_hat_out)
{
    if (!c || !y || !params || !out_stream || !out_size) return DCVC_RK_ERR_INVALID_ARG;
    *out_stream = NULL; *out_size = 0;
    return c->passes == 4
        ? encode_4x(c, y, params, H, W, out_stream, out_size, y_hat_out)
        : encode_2x(c, y, params, H, W, out_stream, out_size, y_hat_out);
}

DcvcRkStatus dcvc_rk_ar_decode_y(DcvcRkArCodec* c, const float* params,
                                 int H, int W, const uint8_t* stream, size_t stream_size,
                                 float* y_hat_out)
{
    if (!c || !params || !stream || !y_hat_out) return DCVC_RK_ERR_INVALID_ARG;
    return c->passes == 4
        ? decode_4x(c, params, H, W, stream, stream_size, y_hat_out)
        : decode_2x(c, params, H, W, stream, stream_size, y_hat_out);
}
