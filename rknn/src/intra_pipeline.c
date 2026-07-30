#include "dcvc_rk/intra_pipeline.h"
#include "dcvc_rk/ar_codec.h"
#include "dcvc_rk/kernels.h"
#include "dcvc_rk/profile.h"
#include "dcvc_rk/quant.h"
#include "dcvc_rk/rknn_engine.h"
#include "async_overlap.h"
#include "npy_reader.h"
#include "rans_c.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_CH 256
#define Z_CH 128
#define Q_CH 256
#define PF_CH (2 * N_CH + 2) /* 514 */

struct DcvcRkIntraPipeline {
    int H, W, Hp, Wp, yH, yW, zH, zW, qp;
    int pipe_i8;
    DcvcRkEngine *eng_analysis_hyper;
    DcvcRkEngine *eng_prior_chain;
    DcvcRkEngine *eng_synthesis;
    DcvcRkArCodec* ar;
    DcvcRansEncoder* rans_enc;
    DcvcRansDecoder* rans_dec;
    DcvcNpy zcdf, zlen, zoff;
    float *qenc, *qdec;
    float *y, *z, *z_hat, *params_fusion, *y_hat;
    float *x_pad, *x_ycbcr, *x_hat;
    int8_t* z_int8;
    /* Native INT8 intermediates / I/O scratch */
    int8_t *qenc_i8, *qdec_i8;
    int8_t *x_i8_in, *y_i8, *z_i8, *z_hat_i8, *params_i8, *y_hat_i8, *x_i8_out;
    DcvcRkQuant q_ah_in0, q_ah_in1, q_ah_out0, q_ah_out1;
    DcvcRkQuant q_pc_in0, q_pc_out0;
    DcvcRkQuant q_syn_in0, q_syn_in1, q_syn_out0;
    int64_t npu_us;
    DcvcRkProfile prof;
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

static DcvcRkStatus run_io(DcvcRkEngine* e,
                           DcvcRkTensorView* vin, int n_in,
                           DcvcRkTensorView* vout, int n_out,
                           DcvcRkStageProf* s, int64_t* acc)
{
    double t0 = dcvc_rk_now_ms();
    DcvcRkStatus st = dcvc_rk_engine_run(e, vin, n_in, vout, n_out);
    if (s) s->wall_ms += dcvc_rk_now_ms() - t0;
    if (st == DCVC_RK_OK) acc_eng(s, e, acc);
    return st;
}

typedef struct {
    DcvcRkIntraPipeline* p;
    uint8_t** z_stream;
    size_t* z_sz;
} IntraZEncCtx;

static DcvcRkStatus intra_cpu_z_enc(void* ctx)
{
    IntraZEncCtx* c = (IntraZEncCtx*)ctx;
    DcvcRkIntraPipeline* p = c->p;
    int zhw = p->zH * p->zW;
    dcvc_rans_encoder_reset(p->rans_enc);
    int zidx = dcvc_rans_encoder_add_cdf(p->rans_enc, dcvc_npy_i32(&p->zcdf),
                    p->zcdf.dims[0], p->zcdf.dims[1],
                    dcvc_npy_i32(&p->zlen), dcvc_npy_i32(&p->zoff));
    dcvc_rans_encoder_encode_z(p->rans_enc, p->z_int8, Z_CH * zhw, zidx, p->qp * Z_CH, zhw);
    dcvc_rans_encoder_flush(p->rans_enc);
    if (dcvc_rans_encoder_get_stream(p->rans_enc, c->z_stream, c->z_sz) != 0)
        return DCVC_RK_ERR_ENTROPY;
    return DCVC_RK_OK;
}

typedef struct {
    uint8_t* z_stream;
    size_t z_sz;
    uint8_t* y_stream;
    size_t y_sz;
    uint8_t** out_stream;
    size_t* out_size;
} IntraPackCtx;

static DcvcRkStatus intra_cpu_pack(void* ctx)
{
    IntraPackCtx* c = (IntraPackCtx*)ctx;
    size_t total = 4 + c->z_sz + c->y_sz;
    uint8_t* buf = (uint8_t*)malloc(total);
    if (!buf) return DCVC_RK_ERR_OOM;
    uint32_t zl = (uint32_t)c->z_sz;
    memcpy(buf, &zl, 4);
    if (c->z_sz) memcpy(buf + 4, c->z_stream, c->z_sz);
    if (c->y_sz) memcpy(buf + 4 + c->z_sz, c->y_stream, c->y_sz);
    *c->out_stream = buf;
    *c->out_size = total;
    return DCVC_RK_OK;
}

static int load_qrow(const char* path, int qp, int expect_c, float** out)
{
    DcvcNpy q = {0};
    if (dcvc_npy_read(path, &q) != 0) return -1;
    if (qp < 0 || qp >= q.dims[0] || q.dims[1] < expect_c) {
        dcvc_npy_free(&q); return -1;
    }
    *out = (float*)malloc((size_t)expect_c * sizeof(float));
    if (!*out) { dcvc_npy_free(&q); return -1; }
    for (int i = 0; i < expect_c; i++)
        (*out)[i] = dcvc_npy_f32(&q)[qp * q.dims[1] + i];
    dcvc_npy_free(&q);
    return 0;
}

DcvcRkIntraPipeline* dcvc_rk_intra_create(const char* model_dir, int H, int W, int qp,
                                          DcvcRkStatus* out_st)
{
    if (out_st) *out_st = DCVC_RK_OK;
    if (!model_dir || H <= 0 || W <= 0) {
        if (out_st) *out_st = DCVC_RK_ERR_INVALID_ARG;
        return NULL;
    }
    DcvcRkIntraPipeline* p = (DcvcRkIntraPipeline*)calloc(1, sizeof(*p));
    if (!p) { if (out_st) *out_st = DCVC_RK_ERR_OOM; return NULL; }
    p->H = H; p->W = W; p->qp = qp;
    p->Hp = (H + 63) / 64 * 64;
    p->Wp = (W + 63) / 64 * 64;
    p->yH = p->Hp / 16; p->yW = p->Wp / 16;
    p->zH = p->Hp / 64; p->zW = p->Wp / 64;

    char path[768];
    DcvcRkStatus st = DCVC_RK_OK;
    #define LOAD(field, stem) do { \
        snprintf(path, sizeof(path), "%s/%s.rknn", model_dir, stem); \
        p->field = dcvc_rk_engine_create(path, &st); \
        if (!p->field) goto fail; \
    } while (0)
    LOAD(eng_analysis_hyper, "intra_analysis_hyper");
    LOAD(eng_prior_chain, "intra_prior_chain");
    LOAD(eng_synthesis, "intra_synthesis");
    #undef LOAD

    p->ar = dcvc_rk_ar_codec_create(model_dir, N_CH, 4, &st);
    if (!p->ar) goto fail;

    p->rans_enc = dcvc_rans_encoder_create();
    p->rans_dec = dcvc_rans_decoder_create();
    if (!p->rans_enc || !p->rans_dec) { st = DCVC_RK_ERR_OOM; goto fail; }

    snprintf(path, sizeof(path), "%s/bitest_cdf.npy", model_dir);
    if (dcvc_npy_read(path, &p->zcdf) != 0) { st = DCVC_RK_ERR_IO; goto fail; }
    snprintf(path, sizeof(path), "%s/bitest_cdf_length.npy", model_dir);
    if (dcvc_npy_read(path, &p->zlen) != 0) { st = DCVC_RK_ERR_IO; goto fail; }
    snprintf(path, sizeof(path), "%s/bitest_offset.npy", model_dir);
    if (dcvc_npy_read(path, &p->zoff) != 0) { st = DCVC_RK_ERR_IO; goto fail; }

    snprintf(path, sizeof(path), "%s/q_scale_enc.npy", model_dir);
    if (load_qrow(path, qp, Q_CH, &p->qenc) != 0) { st = DCVC_RK_ERR_IO; goto fail; }
    snprintf(path, sizeof(path), "%s/q_scale_dec.npy", model_dir);
    if (load_qrow(path, qp, Q_CH, &p->qdec) != 0) { st = DCVC_RK_ERR_IO; goto fail; }

    size_t yhw = (size_t)p->yH * p->yW, zhw = (size_t)p->zH * p->zW;
    size_t HpWp = (size_t)p->Hp * p->Wp;
    #define M(ptr, n) do { ptr = (float*)calloc((n), sizeof(float)); if (!ptr) { st = DCVC_RK_ERR_OOM; goto fail; } } while (0)
    M(p->y, N_CH * yhw); M(p->z, Z_CH * zhw); M(p->z_hat, Z_CH * zhw);
    M(p->params_fusion, PF_CH * yhw); M(p->y_hat, N_CH * yhw);
    M(p->x_pad, 3 * HpWp); M(p->x_ycbcr, 3 * HpWp); M(p->x_hat, 3 * HpWp);
    #undef M
    p->z_int8 = (int8_t*)calloc(Z_CH * zhw, 1);
    if (!p->z_int8) { st = DCVC_RK_ERR_OOM; goto fail; }

    p->pipe_i8 = dcvc_rk_pipe_i8_enabled();
    if (p->pipe_i8) {
        if (dcvc_rk_engine_in_quant(p->eng_analysis_hyper, 0, &p->q_ah_in0) != DCVC_RK_OK ||
            dcvc_rk_engine_in_quant(p->eng_analysis_hyper, 1, &p->q_ah_in1) != DCVC_RK_OK ||
            dcvc_rk_engine_out_quant(p->eng_analysis_hyper, 0, &p->q_ah_out0) != DCVC_RK_OK ||
            dcvc_rk_engine_out_quant(p->eng_analysis_hyper, 1, &p->q_ah_out1) != DCVC_RK_OK ||
            dcvc_rk_engine_in_quant(p->eng_prior_chain, 0, &p->q_pc_in0) != DCVC_RK_OK ||
            dcvc_rk_engine_out_quant(p->eng_prior_chain, 0, &p->q_pc_out0) != DCVC_RK_OK ||
            dcvc_rk_engine_in_quant(p->eng_synthesis, 0, &p->q_syn_in0) != DCVC_RK_OK ||
            dcvc_rk_engine_in_quant(p->eng_synthesis, 1, &p->q_syn_in1) != DCVC_RK_OK ||
            dcvc_rk_engine_out_quant(p->eng_synthesis, 0, &p->q_syn_out0) != DCVC_RK_OK) {
            st = DCVC_RK_ERR_UNSUPPORTED; goto fail;
        }
        #define MI8(ptr, n) do { ptr = (int8_t*)calloc((size_t)(n), 1); if (!ptr) { st = DCVC_RK_ERR_OOM; goto fail; } } while (0)
        MI8(p->qenc_i8, Q_CH);
        MI8(p->qdec_i8, Q_CH);
        MI8(p->x_i8_in, 3 * HpWp);
        MI8(p->y_i8, dcvc_rk_engine_out_buf_bytes(p->eng_analysis_hyper, 0));
        MI8(p->z_i8, dcvc_rk_engine_out_buf_bytes(p->eng_analysis_hyper, 1));
        MI8(p->z_hat_i8, Z_CH * zhw);
        MI8(p->params_i8, dcvc_rk_engine_out_buf_bytes(p->eng_prior_chain, 0));
        MI8(p->y_hat_i8, N_CH * yhw);
        MI8(p->x_i8_out, dcvc_rk_engine_out_buf_bytes(p->eng_synthesis, 0));
        #undef MI8
        dcvc_rk_quant_nchw_f32_to_i8(p->qenc, p->qenc_i8, 1, Q_CH, 1, 1,
                                     p->q_ah_in1.scale, p->q_ah_in1.zp);
        dcvc_rk_quant_nchw_f32_to_i8(p->qdec, p->qdec_i8, 1, Q_CH, 1, 1,
                                     p->q_syn_in1.scale, p->q_syn_in1.zp);
        {
            static int once;
            if (!once) {
                fprintf(stderr, "dcvc_rk: PIPE_I8=1 (native INT8 intermediates)\n");
                once = 1;
            }
        }
    }
    return p;
fail:
    dcvc_rk_intra_destroy(p);
    if (out_st) *out_st = st;
    return NULL;
}

void dcvc_rk_intra_destroy(DcvcRkIntraPipeline* p)
{
    if (!p) return;
    if (p->eng_analysis_hyper) dcvc_rk_engine_destroy(p->eng_analysis_hyper);
    if (p->eng_prior_chain) dcvc_rk_engine_destroy(p->eng_prior_chain);
    if (p->eng_synthesis) dcvc_rk_engine_destroy(p->eng_synthesis);
    if (p->ar) dcvc_rk_ar_codec_destroy(p->ar);
    if (p->rans_enc) dcvc_rans_encoder_destroy(p->rans_enc);
    if (p->rans_dec) dcvc_rans_decoder_destroy(p->rans_dec);
    dcvc_npy_free(&p->zcdf); dcvc_npy_free(&p->zlen); dcvc_npy_free(&p->zoff);
    free(p->qenc); free(p->qdec);
    free(p->y); free(p->z); free(p->z_hat); free(p->params_fusion);
    free(p->y_hat); free(p->x_pad); free(p->x_ycbcr); free(p->x_hat); free(p->z_int8);
    free(p->qenc_i8); free(p->qdec_i8);
    free(p->x_i8_in); free(p->y_i8); free(p->z_i8); free(p->z_hat_i8);
    free(p->params_i8); free(p->y_hat_i8); free(p->x_i8_out);
    free(p);
}

int64_t dcvc_rk_intra_npu_us(DcvcRkIntraPipeline* p) { return p ? p->npu_us : 0; }
void dcvc_rk_intra_reset_npu_us(DcvcRkIntraPipeline* p) { if (p) p->npu_us = 0; }
const DcvcRkProfile* dcvc_rk_intra_last_profile(const DcvcRkIntraPipeline* p)
{
    return p ? &p->prof : NULL;
}
const DcvcRkProfile* dcvc_rk_intra_last_ar_profile(const DcvcRkIntraPipeline* p)
{
    return (p && p->ar) ? dcvc_rk_ar_last_profile(p->ar) : NULL;
}

DcvcRkStatus dcvc_rk_intra_encode(DcvcRkIntraPipeline* p, const float* x,
                                  uint8_t** out_stream, size_t* out_size,
                                  float* x_hat_out)
{
    if (!p || !x || !out_stream || !out_size) return DCVC_RK_ERR_INVALID_ARG;
    *out_stream = NULL; *out_size = 0;
    int Hp = p->Hp, Wp = p->Wp, zhw = p->zH * p->zW;
    DcvcRkStatus st;
    dcvc_rk_profile_reset(&p->prof);
    DcvcRkStageProf* s_pre = dcvc_rk_profile_add(&p->prof, "preprocess");
    DcvcRkStageProf* s_ah  = dcvc_rk_profile_add(&p->prof, "analysis_hyper");
    DcvcRkStageProf* s_ze  = dcvc_rk_profile_add(&p->prof, "z_entropy");
    DcvcRkStageProf* s_pc  = dcvc_rk_profile_add(&p->prof, "prior_chain");
    DcvcRkStageProf* s_ar  = dcvc_rk_profile_add(&p->prof, "ar_y");
    DcvcRkStageProf* s_syn = dcvc_rk_profile_add(&p->prof, "synthesis");
    DcvcRkStageProf* s_post = dcvc_rk_profile_add(&p->prof, "postprocess");

    double t = dcvc_rk_now_ms();
    dcvc_rk_replicate_pad_3(x, p->H, p->W, p->x_pad, Hp, Wp);
    dcvc_rk_rgb_to_ycbcr(p->x_pad, p->x_ycbcr, Hp * Wp);
    if (s_pre) s_pre->wall_ms = dcvc_rk_now_ms() - t;

    if (p->pipe_i8) {
        t = dcvc_rk_now_ms();
        dcvc_rk_quant_nchw_f32_to_i8(p->x_ycbcr, p->x_i8_in, 1, 3, Hp, Wp,
                                     p->q_ah_in0.scale, p->q_ah_in0.zp);
        if (s_pre) s_pre->wall_ms += dcvc_rk_now_ms() - t;

        {
            DcvcRkI8View vin[2] = {
                { p->x_i8_in, 1, 3, Hp, Wp },
                { p->qenc_i8, 1, Q_CH, 1, 1 }
            };
            DcvcRkI8View vout[2] = {
                { p->y_i8, 1, N_CH, p->yH, p->yW },
                { p->z_i8, 1, Z_CH, p->zH, p->zW }
            };
            double t0 = dcvc_rk_now_ms();
            st = dcvc_rk_engine_run_i8(p->eng_analysis_hyper, vin, 2, vout, 2);
            if (s_ah) s_ah->wall_ms += dcvc_rk_now_ms() - t0;
            if (st == DCVC_RK_OK) acc_eng(s_ah, p->eng_analysis_hyper, &p->npu_us);
            if (st != DCVC_RK_OK) return st;
        }

        t = dcvc_rk_now_ms();
        dcvc_rk_dequant_nchw_i8_to_f32(p->z_i8, p->z, 1, Z_CH, p->zH, p->zW,
                                       p->q_ah_out1.w_stride, p->q_ah_out1.scale, p->q_ah_out1.zp);
        dcvc_rk_round_to_int8(p->z, p->z_hat, p->z_int8, Z_CH * zhw);
        dcvc_rk_quant_nchw_f32_to_i8(p->z_hat, p->z_hat_i8, 1, Z_CH, p->zH, p->zW,
                                     p->q_pc_in0.scale, p->q_pc_in0.zp);
        if (s_ze) s_ze->wall_ms += dcvc_rk_now_ms() - t;

        uint8_t* z_stream = NULL; size_t z_sz = 0;
        IntraZEncCtx zctx = { p, &z_stream, &z_sz };
        {
            DcvcRkI8View vin = { p->z_hat_i8, 1, Z_CH, p->zH, p->zW };
            DcvcRkI8View vout = { p->params_i8, 1, PF_CH, p->yH, p->yW };
            st = dcvc_rk_overlap_npu_i8_cpu(p->eng_prior_chain, &vin, 1, &vout, 1,
                                           intra_cpu_z_enc, &zctx, s_pc, s_ze, &p->npu_us);
            if (st != DCVC_RK_OK) { free(z_stream); return st; }
        }

        t = dcvc_rk_now_ms();
        dcvc_rk_dequant_nchw_i8_to_f32(p->y_i8, p->y, 1, N_CH, p->yH, p->yW,
                                       p->q_ah_out0.w_stride, p->q_ah_out0.scale, p->q_ah_out0.zp);
        dcvc_rk_dequant_nchw_i8_to_f32(p->params_i8, p->params_fusion, 1, PF_CH, p->yH, p->yW,
                                       p->q_pc_out0.w_stride, p->q_pc_out0.scale, p->q_pc_out0.zp);
        if (s_ar) s_ar->wall_ms += dcvc_rk_now_ms() - t;

        t = dcvc_rk_now_ms();
        uint8_t* y_stream = NULL; size_t y_sz = 0;
        st = dcvc_rk_ar_encode_y(p->ar, p->y, p->params_fusion, p->yH, p->yW,
                                 &y_stream, &y_sz, p->y_hat);
        if (s_ar) {
            s_ar->wall_ms += dcvc_rk_now_ms() - t;
            const DcvcRkProfile* ap = dcvc_rk_ar_last_profile(p->ar);
            if (ap) {
                for (int i = 0; i < ap->n; i++) {
                    s_ar->npu_ms += ap->stages[i].npu_ms;
                    s_ar->set_ms += ap->stages[i].set_ms;
                    s_ar->get_ms += ap->stages[i].get_ms;
                    s_ar->calls  += ap->stages[i].calls;
                }
            }
            p->npu_us += dcvc_rk_ar_npu_us(p->ar);
        }
        if (st != DCVC_RK_OK) { free(z_stream); return st; }

        IntraPackCtx pack = { z_stream, z_sz, y_stream, y_sz, out_stream, out_size };
        if (x_hat_out) {
            t = dcvc_rk_now_ms();
            dcvc_rk_quant_nchw_f32_to_i8(p->y_hat, p->y_hat_i8, 1, N_CH, p->yH, p->yW,
                                         p->q_syn_in0.scale, p->q_syn_in0.zp);
            if (s_syn) s_syn->wall_ms += dcvc_rk_now_ms() - t;
            DcvcRkI8View vin[2] = {
                { p->y_hat_i8, 1, N_CH, p->yH, p->yW },
                { p->qdec_i8, 1, Q_CH, 1, 1 }
            };
            DcvcRkI8View vout = { p->x_i8_out, 1, 3, Hp, Wp };
            st = dcvc_rk_overlap_npu_i8_cpu(p->eng_synthesis, vin, 2, &vout, 1,
                                           intra_cpu_pack, &pack, s_syn, NULL, &p->npu_us);
            free(z_stream); free(y_stream);
            if (st != DCVC_RK_OK) return st;
            t = dcvc_rk_now_ms();
            dcvc_rk_dequant_nchw_i8_to_f32(p->x_i8_out, p->x_ycbcr, 1, 3, Hp, Wp,
                                           p->q_syn_out0.w_stride, p->q_syn_out0.scale, p->q_syn_out0.zp);
            dcvc_rk_ycbcr_to_rgb(p->x_ycbcr, p->x_hat, Hp * Wp);
            dcvc_rk_crop_3(p->x_hat, Hp, Wp, x_hat_out, p->H, p->W);
            if (s_post) s_post->wall_ms = dcvc_rk_now_ms() - t;
        } else {
            st = intra_cpu_pack(&pack);
            free(z_stream); free(y_stream);
            if (st != DCVC_RK_OK) return st;
        }
        return DCVC_RK_OK;
    }

    /* ---- legacy float-hop path ---- */
    {
        DcvcRkTensorView vin[2] = {
            { p->x_ycbcr, 1, 3, Hp, Wp },
            { p->qenc, 1, Q_CH, 1, 1 }
        };
        DcvcRkTensorView vout[2] = {
            { p->y, 1, N_CH, p->yH, p->yW },
            { p->z, 1, Z_CH, p->zH, p->zW }
        };
        st = run_io(p->eng_analysis_hyper, vin, 2, vout, 2, s_ah, &p->npu_us);
        if (st != DCVC_RK_OK) return st;
    }

    t = dcvc_rk_now_ms();
    dcvc_rk_round_to_int8(p->z, p->z_hat, p->z_int8, Z_CH * zhw);
    if (s_ze) s_ze->wall_ms = dcvc_rk_now_ms() - t;

    uint8_t* z_stream = NULL; size_t z_sz = 0;
    IntraZEncCtx zctx = { p, &z_stream, &z_sz };
    {
        DcvcRkTensorView vin = { p->z_hat, 1, Z_CH, p->zH, p->zW };
        DcvcRkTensorView vout = { p->params_fusion, 1, PF_CH, p->yH, p->yW };
        st = dcvc_rk_overlap_npu_cpu(p->eng_prior_chain, &vin, 1, &vout, 1,
                                     intra_cpu_z_enc, &zctx, s_pc, s_ze, &p->npu_us);
        if (st != DCVC_RK_OK) { free(z_stream); return st; }
    }

    t = dcvc_rk_now_ms();
    uint8_t* y_stream = NULL; size_t y_sz = 0;
    st = dcvc_rk_ar_encode_y(p->ar, p->y, p->params_fusion, p->yH, p->yW,
                             &y_stream, &y_sz, p->y_hat);
    if (s_ar) {
        s_ar->wall_ms = dcvc_rk_now_ms() - t;
        const DcvcRkProfile* ap = dcvc_rk_ar_last_profile(p->ar);
        if (ap) {
            for (int i = 0; i < ap->n; i++) {
                s_ar->npu_ms += ap->stages[i].npu_ms;
                s_ar->set_ms += ap->stages[i].set_ms;
                s_ar->get_ms += ap->stages[i].get_ms;
                s_ar->calls  += ap->stages[i].calls;
            }
        }
        p->npu_us += dcvc_rk_ar_npu_us(p->ar);
    }
    if (st != DCVC_RK_OK) { free(z_stream); return st; }

    IntraPackCtx pack = { z_stream, z_sz, y_stream, y_sz, out_stream, out_size };
    if (x_hat_out) {
        DcvcRkTensorView vin[2] = {
            { p->y_hat, 1, N_CH, p->yH, p->yW },
            { p->qdec, 1, Q_CH, 1, 1 }
        };
        DcvcRkTensorView vout = { p->x_ycbcr, 1, 3, Hp, Wp };
        st = dcvc_rk_overlap_npu_cpu(p->eng_synthesis, vin, 2, &vout, 1,
                                     intra_cpu_pack, &pack, s_syn, NULL, &p->npu_us);
        free(z_stream); free(y_stream);
        if (st != DCVC_RK_OK) return st;
        t = dcvc_rk_now_ms();
        dcvc_rk_ycbcr_to_rgb(p->x_ycbcr, p->x_hat, Hp * Wp);
        dcvc_rk_crop_3(p->x_hat, Hp, Wp, x_hat_out, p->H, p->W);
        if (s_post) s_post->wall_ms = dcvc_rk_now_ms() - t;
    } else {
        st = intra_cpu_pack(&pack);
        free(z_stream); free(y_stream);
        if (st != DCVC_RK_OK) return st;
    }
    return DCVC_RK_OK;
}

DcvcRkStatus dcvc_rk_intra_decode(DcvcRkIntraPipeline* p,
                                  const uint8_t* stream, size_t stream_size,
                                  float* x_hat_out)
{
    if (!p || !stream || stream_size < 4 || !x_hat_out) return DCVC_RK_ERR_INVALID_ARG;
    int Hp = p->Hp, Wp = p->Wp, zhw = p->zH * p->zW;
    uint32_t z_sz = 0;
    memcpy(&z_sz, stream, 4);
    if (4 + z_sz > stream_size) return DCVC_RK_ERR_ENTROPY;
    const uint8_t* z_stream = stream + 4;
    const uint8_t* y_stream = stream + 4 + z_sz;
    size_t y_sz = stream_size - 4 - z_sz;

    dcvc_rk_profile_reset(&p->prof);
    DcvcRkStageProf* s_ze  = dcvc_rk_profile_add(&p->prof, "z_entropy");
    DcvcRkStageProf* s_pc  = dcvc_rk_profile_add(&p->prof, "prior_chain");
    DcvcRkStageProf* s_ar  = dcvc_rk_profile_add(&p->prof, "ar_y");
    DcvcRkStageProf* s_syn = dcvc_rk_profile_add(&p->prof, "synthesis");
    DcvcRkStageProf* s_post = dcvc_rk_profile_add(&p->prof, "postprocess");

    double t = dcvc_rk_now_ms();
    dcvc_rans_decoder_reset_cdf(p->rans_dec);
    int zidx = dcvc_rans_decoder_add_cdf(p->rans_dec, dcvc_npy_i32(&p->zcdf),
                    p->zcdf.dims[0], p->zcdf.dims[1],
                    dcvc_npy_i32(&p->zlen), dcvc_npy_i32(&p->zoff));
    dcvc_rans_decoder_set_stream(p->rans_dec, z_stream, z_sz);
    dcvc_rans_decoder_decode_z(p->rans_dec, Z_CH * zhw, zidx, p->qp * Z_CH, zhw);
    int8_t* zsyms = NULL; size_t zn = 0;
    dcvc_rans_decoder_get_symbols(p->rans_dec, &zsyms, &zn);
    if (dcvc_rans_decoder_has_error(p->rans_dec)) { free(zsyms); return DCVC_RK_ERR_ENTROPY; }
    dcvc_rk_int8_to_float(zsyms, p->z_hat, Z_CH * zhw);
    free(zsyms);
    if (s_ze) s_ze->wall_ms = dcvc_rk_now_ms() - t;

    if (p->pipe_i8) {
        t = dcvc_rk_now_ms();
        dcvc_rk_quant_nchw_f32_to_i8(p->z_hat, p->z_hat_i8, 1, Z_CH, p->zH, p->zW,
                                     p->q_pc_in0.scale, p->q_pc_in0.zp);
        if (s_pc) s_pc->wall_ms += dcvc_rk_now_ms() - t;
        {
            DcvcRkI8View vin = { p->z_hat_i8, 1, Z_CH, p->zH, p->zW };
            DcvcRkI8View vout = { p->params_i8, 1, PF_CH, p->yH, p->yW };
            double t0 = dcvc_rk_now_ms();
            DcvcRkStatus st = dcvc_rk_engine_run_i8(p->eng_prior_chain, &vin, 1, &vout, 1);
            if (s_pc) s_pc->wall_ms += dcvc_rk_now_ms() - t0;
            if (st == DCVC_RK_OK) acc_eng(s_pc, p->eng_prior_chain, &p->npu_us);
            if (st != DCVC_RK_OK) return st;
        }
        t = dcvc_rk_now_ms();
        dcvc_rk_dequant_nchw_i8_to_f32(p->params_i8, p->params_fusion, 1, PF_CH, p->yH, p->yW,
                                       p->q_pc_out0.w_stride, p->q_pc_out0.scale, p->q_pc_out0.zp);
        if (s_ar) s_ar->wall_ms += dcvc_rk_now_ms() - t;

        t = dcvc_rk_now_ms();
        DcvcRkStatus st = dcvc_rk_ar_decode_y(p->ar, p->params_fusion, p->yH, p->yW,
                                              y_stream, y_sz, p->y_hat);
        if (s_ar) {
            s_ar->wall_ms += dcvc_rk_now_ms() - t;
            const DcvcRkProfile* ap = dcvc_rk_ar_last_profile(p->ar);
            if (ap) {
                for (int i = 0; i < ap->n; i++) {
                    s_ar->npu_ms += ap->stages[i].npu_ms;
                    s_ar->set_ms += ap->stages[i].set_ms;
                    s_ar->get_ms += ap->stages[i].get_ms;
                    s_ar->calls  += ap->stages[i].calls;
                }
            }
            p->npu_us += dcvc_rk_ar_npu_us(p->ar);
        }
        if (st != DCVC_RK_OK) return st;

        t = dcvc_rk_now_ms();
        dcvc_rk_quant_nchw_f32_to_i8(p->y_hat, p->y_hat_i8, 1, N_CH, p->yH, p->yW,
                                     p->q_syn_in0.scale, p->q_syn_in0.zp);
        if (s_syn) s_syn->wall_ms += dcvc_rk_now_ms() - t;
        {
            DcvcRkI8View vin[2] = {
                { p->y_hat_i8, 1, N_CH, p->yH, p->yW },
                { p->qdec_i8, 1, Q_CH, 1, 1 }
            };
            DcvcRkI8View vout = { p->x_i8_out, 1, 3, Hp, Wp };
            double t0 = dcvc_rk_now_ms();
            st = dcvc_rk_engine_run_i8(p->eng_synthesis, vin, 2, &vout, 1);
            if (s_syn) s_syn->wall_ms += dcvc_rk_now_ms() - t0;
            if (st == DCVC_RK_OK) acc_eng(s_syn, p->eng_synthesis, &p->npu_us);
            if (st != DCVC_RK_OK) return st;
        }
        t = dcvc_rk_now_ms();
        dcvc_rk_dequant_nchw_i8_to_f32(p->x_i8_out, p->x_ycbcr, 1, 3, Hp, Wp,
                                       p->q_syn_out0.w_stride, p->q_syn_out0.scale, p->q_syn_out0.zp);
        dcvc_rk_ycbcr_to_rgb(p->x_ycbcr, p->x_hat, Hp * Wp);
        dcvc_rk_crop_3(p->x_hat, Hp, Wp, x_hat_out, p->H, p->W);
        if (s_post) s_post->wall_ms = dcvc_rk_now_ms() - t;
        return DCVC_RK_OK;
    }

    {
        DcvcRkTensorView vin = { p->z_hat, 1, Z_CH, p->zH, p->zW };
        DcvcRkTensorView vout = { p->params_fusion, 1, PF_CH, p->yH, p->yW };
        DcvcRkStatus st = run_io(p->eng_prior_chain, &vin, 1, &vout, 1, s_pc, &p->npu_us);
        if (st != DCVC_RK_OK) return st;
    }

    t = dcvc_rk_now_ms();
    DcvcRkStatus st = dcvc_rk_ar_decode_y(p->ar, p->params_fusion, p->yH, p->yW,
                                          y_stream, y_sz, p->y_hat);
    if (s_ar) {
        s_ar->wall_ms = dcvc_rk_now_ms() - t;
        const DcvcRkProfile* ap = dcvc_rk_ar_last_profile(p->ar);
        if (ap) {
            for (int i = 0; i < ap->n; i++) {
                s_ar->npu_ms += ap->stages[i].npu_ms;
                s_ar->set_ms += ap->stages[i].set_ms;
                s_ar->get_ms += ap->stages[i].get_ms;
                s_ar->calls  += ap->stages[i].calls;
            }
        }
        p->npu_us += dcvc_rk_ar_npu_us(p->ar);
    }
    if (st != DCVC_RK_OK) return st;

    {
        DcvcRkTensorView vin[2] = {
            { p->y_hat, 1, N_CH, p->yH, p->yW },
            { p->qdec, 1, Q_CH, 1, 1 }
        };
        DcvcRkTensorView vout = { p->x_ycbcr, 1, 3, Hp, Wp };
        st = run_io(p->eng_synthesis, vin, 2, &vout, 1, s_syn, &p->npu_us);
        if (st != DCVC_RK_OK) return st;
    }
    t = dcvc_rk_now_ms();
    dcvc_rk_ycbcr_to_rgb(p->x_ycbcr, p->x_hat, Hp * Wp);
    dcvc_rk_crop_3(p->x_hat, Hp, Wp, x_hat_out, p->H, p->W);
    if (s_post) s_post->wall_ms = dcvc_rk_now_ms() - t;
    return DCVC_RK_OK;
}
