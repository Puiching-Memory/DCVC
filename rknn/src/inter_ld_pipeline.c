#include "dcvc_rk/inter_ld_pipeline.h"
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

#define YCH 128
#define ZCH 128
#define DCH 256
#define SRCD 192
#define RECON_Q 320
#define PFCH (3 * YCH) /* 384 */

struct DcvcRkInterLdPipeline {
    int H, W, Hp, Wp, fH, fW, yH, yW, zH, zW, qp;
    int pipe_i8;
    DcvcRkEngine *eng_feat_i;
    DcvcRkEngine *eng_feat_p;
    DcvcRkEngine *eng_enc_hyper;
    DcvcRkEngine *eng_prior_chain;
    DcvcRkEngine *eng_dec_recon;
    DcvcRkArCodec* ar;
    DcvcRansEncoder* rans_enc;
    DcvcRansDecoder* rans_dec;
    DcvcNpy zcdf, zlen, zoff;
    float *q_enc, *q_dec, *q_feat, *q_recon;
    float *memory, *ref_feature, *ctx, *x_unshuf, *x_unshuf_next, *ref_unshuf;
    float *y, *z, *z_hat, *params, *y_hat, *feature_dec;
    float *recon_192, *frame_pad, *frame_ycbcr, *frame_rgb;
    int8_t* z_int8;
    int8_t *q_enc_i8, *q_dec_i8, *q_feat_i8, *q_recon_i8;
    int8_t *x_unshuf_i8, *ref_unshuf_i8;
    int8_t *memory_i8, *ctx_i8, *ref_feature_i8;
    int8_t *y_i8, *z_i8, *z_hat_i8, *params_i8, *y_hat_i8;
    int8_t *feature_i8, *recon_i8;
    int8_t *scratch_i8; /* requant temps sized to max feature map */
    DcvcRkQuant q_fi_in, q_fi_mem, q_fi_ctx;
    DcvcRkQuant q_fp_in, q_fp_mem, q_fp_ctx;
    DcvcRkQuant q_eh_in0, q_eh_in1, q_eh_in2, q_eh_y, q_eh_z;
    DcvcRkQuant q_pc_z, q_pc_mem, q_pc_qf, q_pc_out;
    DcvcRkQuant q_dr_y, q_dr_ctx, q_dr_qd, q_dr_qr, q_dr_feat, q_dr_recon;
    DcvcRkQuant q_mem_cur, q_ctx_cur; /* scale of current memory/ctx i8 buffers */
    int has_prefetch;
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

static void fold_ar(DcvcRkStageProf* s_ar, DcvcRkArCodec* ar, int64_t* npu_acc)
{
    if (!s_ar || !ar) return;
    const DcvcRkProfile* ap = dcvc_rk_ar_last_profile(ar);
    if (ap) {
        for (int i = 0; i < ap->n; i++) {
            s_ar->npu_ms += ap->stages[i].npu_ms;
            s_ar->set_ms += ap->stages[i].set_ms;
            s_ar->get_ms += ap->stages[i].get_ms;
            s_ar->calls  += ap->stages[i].calls;
        }
    }
    if (npu_acc) *npu_acc += dcvc_rk_ar_npu_us(ar);
}

typedef struct {
    DcvcRkInterLdPipeline* p;
    uint8_t** z_stream;
    size_t* z_sz;
} InterZEncCtx;

static DcvcRkStatus inter_cpu_z_enc(void* ctx)
{
    InterZEncCtx* c = (InterZEncCtx*)ctx;
    DcvcRkInterLdPipeline* p = c->p;
    int zHW = p->zH * p->zW;
    dcvc_rans_encoder_reset(p->rans_enc);
    int zidx = dcvc_rans_encoder_add_cdf(p->rans_enc, dcvc_npy_i32(&p->zcdf),
                    p->zcdf.dims[0], p->zcdf.dims[1],
                    dcvc_npy_i32(&p->zlen), dcvc_npy_i32(&p->zoff));
    dcvc_rans_encoder_encode_z(p->rans_enc, p->z_int8, ZCH * zHW, zidx, p->qp * ZCH, zHW);
    dcvc_rans_encoder_flush(p->rans_enc);
    if (dcvc_rans_encoder_get_stream(p->rans_enc, c->z_stream, c->z_sz) != 0)
        return DCVC_RK_ERR_ENTROPY;
    return DCVC_RK_OK;
}

typedef struct {
    DcvcRkInterLdPipeline* p;
    uint8_t* z_stream;
    size_t z_sz;
    uint8_t* y_stream;
    size_t y_sz;
    uint8_t** out_stream;
    size_t* out_size;
    const float* next_x;
    DcvcRkStageProf* s_pre;
} InterTailCtx;

static DcvcRkStatus inter_cpu_tail(void* ctx)
{
    InterTailCtx* c = (InterTailCtx*)ctx;
    size_t total = 4 + c->z_sz + c->y_sz;
    uint8_t* buf = (uint8_t*)malloc(total);
    if (!buf) return DCVC_RK_ERR_OOM;
    uint32_t zl = (uint32_t)c->z_sz;
    memcpy(buf, &zl, 4);
    if (c->z_sz) memcpy(buf + 4, c->z_stream, c->z_sz);
    if (c->y_sz) memcpy(buf + 4 + c->z_sz, c->y_stream, c->y_sz);
    *c->out_stream = buf;
    *c->out_size = total;

    /* Prefetch next frame while dec_recon NPU runs.
     * Use frame_rgb as temporary YCbCr scratch (postprocess has not started). */
    if (c->next_x) {
        DcvcRkInterLdPipeline* p = c->p;
        double t = dcvc_rk_now_ms();
        dcvc_rk_replicate_pad_3(c->next_x, p->H, p->W, p->frame_pad, p->Hp, p->Wp);
        dcvc_rk_rgb_to_ycbcr(p->frame_pad, p->frame_rgb, p->Hp * p->Wp);
        dcvc_rk_pixel_unshuffle_8(p->frame_rgb, p->x_unshuf_next, p->Hp, p->Wp);
        p->has_prefetch = 1;
        if (c->s_pre) c->s_pre->wall_ms += dcvc_rk_now_ms() - t;
    }
    return DCVC_RK_OK;
}

static DcvcRkStatus build_memory(DcvcRkInterLdPipeline* p, int reset, const float* x_ref,
                                 DcvcRkStageProf* s_pre, DcvcRkStageProf* s_feat)
{
    if (p->pipe_i8) {
        DcvcRkI8View vout[2] = {
            { p->memory_i8, 1, DCH, p->fH, p->fW },
            { p->ctx_i8, 1, DCH, p->fH, p->fW }
        };
        if (reset) {
            if (!x_ref) return DCVC_RK_ERR_INVALID_ARG;
            double t = dcvc_rk_now_ms();
            dcvc_rk_replicate_pad_3(x_ref, p->H, p->W, p->frame_pad, p->Hp, p->Wp);
            dcvc_rk_rgb_to_ycbcr(p->frame_pad, p->frame_ycbcr, p->Hp * p->Wp);
            dcvc_rk_pixel_unshuffle_8(p->frame_ycbcr, p->ref_unshuf, p->Hp, p->Wp);
            dcvc_rk_quant_nchw_f32_to_i8(p->ref_unshuf, p->ref_unshuf_i8, 1, SRCD, p->fH, p->fW,
                                         p->q_fi_in.scale, p->q_fi_in.zp);
            if (s_pre) s_pre->wall_ms += dcvc_rk_now_ms() - t;
            DcvcRkI8View vin = { p->ref_unshuf_i8, 1, SRCD, p->fH, p->fW };
            double t0 = dcvc_rk_now_ms();
            DcvcRkStatus st = dcvc_rk_engine_run_i8(p->eng_feat_i, &vin, 1, vout, 2);
            if (s_feat) s_feat->wall_ms += dcvc_rk_now_ms() - t0;
            if (st == DCVC_RK_OK) {
                acc_eng(s_feat, p->eng_feat_i, &p->npu_us);
                p->q_mem_cur = p->q_fi_mem;
                p->q_ctx_cur = p->q_fi_ctx;
            }
            return st;
        }
        /* requant stored feature → feat_p input domain */
        dcvc_rk_requant_nchw_i8(p->ref_feature_i8, 1, DCH, p->fH, p->fW, p->q_dr_feat.w_stride,
                                p->q_dr_feat.scale, p->q_dr_feat.zp,
                                p->scratch_i8, p->q_fp_in.scale, p->q_fp_in.zp);
        DcvcRkI8View vin = { p->scratch_i8, 1, DCH, p->fH, p->fW };
        double t0 = dcvc_rk_now_ms();
        DcvcRkStatus st = dcvc_rk_engine_run_i8(p->eng_feat_p, &vin, 1, vout, 2);
        if (s_feat) s_feat->wall_ms += dcvc_rk_now_ms() - t0;
        if (st == DCVC_RK_OK) {
            acc_eng(s_feat, p->eng_feat_p, &p->npu_us);
            p->q_mem_cur = p->q_fp_mem;
            p->q_ctx_cur = p->q_fp_ctx;
        }
        return st;
    }

    DcvcRkTensorView vout[2] = {
        { p->memory, 1, DCH, p->fH, p->fW },
        { p->ctx, 1, DCH, p->fH, p->fW }
    };
    if (reset) {
        if (!x_ref) return DCVC_RK_ERR_INVALID_ARG;
        double t = dcvc_rk_now_ms();
        dcvc_rk_replicate_pad_3(x_ref, p->H, p->W, p->frame_pad, p->Hp, p->Wp);
        dcvc_rk_rgb_to_ycbcr(p->frame_pad, p->frame_ycbcr, p->Hp * p->Wp);
        dcvc_rk_pixel_unshuffle_8(p->frame_ycbcr, p->ref_unshuf, p->Hp, p->Wp);
        if (s_pre) s_pre->wall_ms += dcvc_rk_now_ms() - t;
        DcvcRkTensorView vin = { p->ref_unshuf, 1, SRCD, p->fH, p->fW };
        return run_io(p->eng_feat_i, &vin, 1, vout, 2, s_feat, &p->npu_us);
    }
    DcvcRkTensorView vin = { p->ref_feature, 1, DCH, p->fH, p->fW };
    return run_io(p->eng_feat_p, &vin, 1, vout, 2, s_feat, &p->npu_us);
}

DcvcRkInterLdPipeline* dcvc_rk_inter_create(const char* model_dir, int H, int W, int qp,
                                            DcvcRkStatus* out_st)
{
    if (out_st) *out_st = DCVC_RK_OK;
    if (!model_dir || H <= 0 || W <= 0) {
        if (out_st) *out_st = DCVC_RK_ERR_INVALID_ARG;
        return NULL;
    }
    DcvcRkInterLdPipeline* p = (DcvcRkInterLdPipeline*)calloc(1, sizeof(*p));
    if (!p) { if (out_st) *out_st = DCVC_RK_ERR_OOM; return NULL; }
    p->H = H; p->W = W; p->qp = qp;
    p->Hp = (H + 63) / 64 * 64; p->Wp = (W + 63) / 64 * 64;
    p->fH = p->Hp / 8; p->fW = p->Wp / 8;
    p->yH = p->Hp / 16; p->yW = p->Wp / 16;
    p->zH = p->Hp / 64; p->zW = p->Wp / 64;

    char path[768];
    DcvcRkStatus st = DCVC_RK_OK;
    #define LOAD(field, stem) do { \
        snprintf(path, sizeof(path), "%s/%s.rknn", model_dir, stem); \
        p->field = dcvc_rk_engine_create(path, &st); \
        if (!p->field) goto fail; \
    } while (0)
    LOAD(eng_feat_i, "inter_feat_i");
    LOAD(eng_feat_p, "inter_feat_p");
    LOAD(eng_enc_hyper, "inter_enc_hyper");
    LOAD(eng_prior_chain, "inter_prior_chain");
    LOAD(eng_dec_recon, "inter_dec_recon");
    #undef LOAD

    p->ar = dcvc_rk_ar_codec_create(model_dir, YCH, 2, &st);
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

    snprintf(path, sizeof(path), "%s/q_encoder.npy", model_dir);
    if (load_qrow(path, qp, DCH, &p->q_enc) != 0) { st = DCVC_RK_ERR_IO; goto fail; }
    snprintf(path, sizeof(path), "%s/q_decoder.npy", model_dir);
    if (load_qrow(path, qp, DCH, &p->q_dec) != 0) { st = DCVC_RK_ERR_IO; goto fail; }
    snprintf(path, sizeof(path), "%s/q_feature.npy", model_dir);
    if (load_qrow(path, qp, DCH, &p->q_feat) != 0) { st = DCVC_RK_ERR_IO; goto fail; }
    snprintf(path, sizeof(path), "%s/q_recon.npy", model_dir);
    if (load_qrow(path, qp, RECON_Q, &p->q_recon) != 0) { st = DCVC_RK_ERR_IO; goto fail; }

    size_t fHW = (size_t)p->fH * p->fW, yHW = (size_t)p->yH * p->yW, zHW = (size_t)p->zH * p->zW;
    size_t HpWp = (size_t)p->Hp * p->Wp;
    #define M(ptr, n) do { ptr = (float*)calloc((n), sizeof(float)); if (!ptr) { st = DCVC_RK_ERR_OOM; goto fail; } } while (0)
    M(p->memory, DCH * fHW); M(p->ref_feature, DCH * fHW); M(p->ctx, DCH * fHW);
    M(p->x_unshuf, SRCD * fHW); M(p->x_unshuf_next, SRCD * fHW); M(p->ref_unshuf, SRCD * fHW);
    M(p->y, YCH * yHW); M(p->z, ZCH * zHW); M(p->z_hat, ZCH * zHW);
    M(p->params, PFCH * yHW); M(p->y_hat, YCH * yHW);
    M(p->feature_dec, DCH * fHW); M(p->recon_192, SRCD * fHW);
    M(p->frame_pad, 3 * HpWp); M(p->frame_ycbcr, 3 * HpWp); M(p->frame_rgb, 3 * HpWp);
    #undef M
    p->z_int8 = (int8_t*)calloc(ZCH * zHW, 1);
    if (!p->z_int8) { st = DCVC_RK_ERR_OOM; goto fail; }

    p->pipe_i8 = dcvc_rk_pipe_i8_enabled();
    if (p->pipe_i8) {
        if (dcvc_rk_engine_in_quant(p->eng_feat_i, 0, &p->q_fi_in) ||
            dcvc_rk_engine_out_quant(p->eng_feat_i, 0, &p->q_fi_mem) ||
            dcvc_rk_engine_out_quant(p->eng_feat_i, 1, &p->q_fi_ctx) ||
            dcvc_rk_engine_in_quant(p->eng_feat_p, 0, &p->q_fp_in) ||
            dcvc_rk_engine_out_quant(p->eng_feat_p, 0, &p->q_fp_mem) ||
            dcvc_rk_engine_out_quant(p->eng_feat_p, 1, &p->q_fp_ctx) ||
            dcvc_rk_engine_in_quant(p->eng_enc_hyper, 0, &p->q_eh_in0) ||
            dcvc_rk_engine_in_quant(p->eng_enc_hyper, 1, &p->q_eh_in1) ||
            dcvc_rk_engine_in_quant(p->eng_enc_hyper, 2, &p->q_eh_in2) ||
            dcvc_rk_engine_out_quant(p->eng_enc_hyper, 0, &p->q_eh_y) ||
            dcvc_rk_engine_out_quant(p->eng_enc_hyper, 1, &p->q_eh_z) ||
            dcvc_rk_engine_in_quant(p->eng_prior_chain, 0, &p->q_pc_z) ||
            dcvc_rk_engine_in_quant(p->eng_prior_chain, 1, &p->q_pc_mem) ||
            dcvc_rk_engine_in_quant(p->eng_prior_chain, 2, &p->q_pc_qf) ||
            dcvc_rk_engine_out_quant(p->eng_prior_chain, 0, &p->q_pc_out) ||
            dcvc_rk_engine_in_quant(p->eng_dec_recon, 0, &p->q_dr_y) ||
            dcvc_rk_engine_in_quant(p->eng_dec_recon, 1, &p->q_dr_ctx) ||
            dcvc_rk_engine_in_quant(p->eng_dec_recon, 2, &p->q_dr_qd) ||
            dcvc_rk_engine_in_quant(p->eng_dec_recon, 3, &p->q_dr_qr) ||
            dcvc_rk_engine_out_quant(p->eng_dec_recon, 0, &p->q_dr_feat) ||
            dcvc_rk_engine_out_quant(p->eng_dec_recon, 1, &p->q_dr_recon)) {
            st = DCVC_RK_ERR_UNSUPPORTED; goto fail;
        }
        size_t fbytes = (size_t)DCH * fHW;
        uint32_t mem_b = dcvc_rk_engine_out_buf_bytes(p->eng_feat_i, 0);
        uint32_t ctx_b = dcvc_rk_engine_out_buf_bytes(p->eng_feat_i, 1);
        uint32_t mem_bp = dcvc_rk_engine_out_buf_bytes(p->eng_feat_p, 0);
        uint32_t ctx_bp = dcvc_rk_engine_out_buf_bytes(p->eng_feat_p, 1);
        if (mem_bp > mem_b) mem_b = mem_bp;
        if (ctx_bp > ctx_b) ctx_b = ctx_bp;
        #define MI8(ptr, n) do { ptr = (int8_t*)calloc((size_t)(n), 1); if (!ptr) { st = DCVC_RK_ERR_OOM; goto fail; } } while (0)
        MI8(p->q_enc_i8, DCH); MI8(p->q_dec_i8, DCH); MI8(p->q_feat_i8, DCH); MI8(p->q_recon_i8, RECON_Q);
        MI8(p->x_unshuf_i8, SRCD * fHW); MI8(p->ref_unshuf_i8, SRCD * fHW);
        MI8(p->memory_i8, mem_b);
        MI8(p->ctx_i8, ctx_b);
        MI8(p->ref_feature_i8, dcvc_rk_engine_out_buf_bytes(p->eng_dec_recon, 0));
        MI8(p->y_i8, dcvc_rk_engine_out_buf_bytes(p->eng_enc_hyper, 0));
        MI8(p->z_i8, dcvc_rk_engine_out_buf_bytes(p->eng_enc_hyper, 1));
        MI8(p->z_hat_i8, ZCH * zHW);
        MI8(p->params_i8, dcvc_rk_engine_out_buf_bytes(p->eng_prior_chain, 0));
        MI8(p->y_hat_i8, YCH * yHW);
        MI8(p->feature_i8, dcvc_rk_engine_out_buf_bytes(p->eng_dec_recon, 0));
        MI8(p->recon_i8, dcvc_rk_engine_out_buf_bytes(p->eng_dec_recon, 1));
        MI8(p->scratch_i8, fbytes > (size_t)SRCD * fHW ? fbytes : (size_t)SRCD * fHW);
        #undef MI8
        dcvc_rk_quant_nchw_f32_to_i8(p->q_enc, p->q_enc_i8, 1, DCH, 1, 1, p->q_eh_in2.scale, p->q_eh_in2.zp);
        dcvc_rk_quant_nchw_f32_to_i8(p->q_dec, p->q_dec_i8, 1, DCH, 1, 1, p->q_dr_qd.scale, p->q_dr_qd.zp);
        dcvc_rk_quant_nchw_f32_to_i8(p->q_feat, p->q_feat_i8, 1, DCH, 1, 1, p->q_pc_qf.scale, p->q_pc_qf.zp);
        dcvc_rk_quant_nchw_f32_to_i8(p->q_recon, p->q_recon_i8, 1, RECON_Q, 1, 1, p->q_dr_qr.scale, p->q_dr_qr.zp);
    }
    return p;
fail:
    dcvc_rk_inter_destroy(p);
    if (out_st) *out_st = st;
    return NULL;
}

void dcvc_rk_inter_destroy(DcvcRkInterLdPipeline* p)
{
    if (!p) return;
    #define D(e) if (p->e) dcvc_rk_engine_destroy(p->e)
    D(eng_feat_i); D(eng_feat_p); D(eng_enc_hyper); D(eng_prior_chain); D(eng_dec_recon);
    #undef D
    if (p->ar) dcvc_rk_ar_codec_destroy(p->ar);
    if (p->rans_enc) dcvc_rans_encoder_destroy(p->rans_enc);
    if (p->rans_dec) dcvc_rans_decoder_destroy(p->rans_dec);
    dcvc_npy_free(&p->zcdf); dcvc_npy_free(&p->zlen); dcvc_npy_free(&p->zoff);
    free(p->q_enc); free(p->q_dec); free(p->q_feat); free(p->q_recon);
    free(p->memory); free(p->ref_feature); free(p->ctx);
    free(p->x_unshuf); free(p->x_unshuf_next); free(p->ref_unshuf);
    free(p->y); free(p->z); free(p->z_hat); free(p->params); free(p->y_hat);
    free(p->feature_dec); free(p->recon_192);
    free(p->frame_pad); free(p->frame_ycbcr); free(p->frame_rgb); free(p->z_int8);
    free(p->q_enc_i8); free(p->q_dec_i8); free(p->q_feat_i8); free(p->q_recon_i8);
    free(p->x_unshuf_i8); free(p->ref_unshuf_i8);
    free(p->memory_i8); free(p->ctx_i8); free(p->ref_feature_i8);
    free(p->y_i8); free(p->z_i8); free(p->z_hat_i8); free(p->params_i8); free(p->y_hat_i8);
    free(p->feature_i8); free(p->recon_i8); free(p->scratch_i8);
    free(p);
}

int64_t dcvc_rk_inter_npu_us(DcvcRkInterLdPipeline* p) { return p ? p->npu_us : 0; }
void dcvc_rk_inter_reset_npu_us(DcvcRkInterLdPipeline* p) { if (p) p->npu_us = 0; }
const DcvcRkProfile* dcvc_rk_inter_last_profile(const DcvcRkInterLdPipeline* p)
{
    return p ? &p->prof : NULL;
}
const DcvcRkProfile* dcvc_rk_inter_last_ar_profile(const DcvcRkInterLdPipeline* p)
{
    return (p && p->ar) ? dcvc_rk_ar_last_profile(p->ar) : NULL;
}

static DcvcRkStatus encode_core(DcvcRkInterLdPipeline* p, const float* x,
                                int reset, const float* x_ref,
                                uint8_t** out_stream, size_t* out_size,
                                float* x_hat_out, const float* next_x)
{
    int Hp = p->Hp, Wp = p->Wp, zHW = p->zH * p->zW;
    dcvc_rk_profile_reset(&p->prof);
    DcvcRkStageProf* s_pre = dcvc_rk_profile_add(&p->prof, "preprocess");
    DcvcRkStageProf* s_feat = dcvc_rk_profile_add(&p->prof, "feat");
    DcvcRkStageProf* s_eh  = dcvc_rk_profile_add(&p->prof, "enc_hyper");
    DcvcRkStageProf* s_ze  = dcvc_rk_profile_add(&p->prof, "z_entropy");
    DcvcRkStageProf* s_pc  = dcvc_rk_profile_add(&p->prof, "prior_chain");
    DcvcRkStageProf* s_ar  = dcvc_rk_profile_add(&p->prof, "ar_y");
    DcvcRkStageProf* s_dr  = dcvc_rk_profile_add(&p->prof, "dec_recon");
    DcvcRkStageProf* s_post = dcvc_rk_profile_add(&p->prof, "postprocess");

    DcvcRkStatus st = build_memory(p, reset, x_ref, s_pre, s_feat);
    if (st != DCVC_RK_OK) return st;

    if (p->has_prefetch) {
        float* tmp = p->x_unshuf;
        p->x_unshuf = p->x_unshuf_next;
        p->x_unshuf_next = tmp;
        p->has_prefetch = 0;
    } else {
        double t = dcvc_rk_now_ms();
        dcvc_rk_replicate_pad_3(x, p->H, p->W, p->frame_pad, Hp, Wp);
        dcvc_rk_rgb_to_ycbcr(p->frame_pad, p->frame_ycbcr, Hp * Wp);
        dcvc_rk_pixel_unshuffle_8(p->frame_ycbcr, p->x_unshuf, Hp, Wp);
        if (s_pre) s_pre->wall_ms += dcvc_rk_now_ms() - t;
    }

    if (p->pipe_i8) {
        double t = dcvc_rk_now_ms();
        dcvc_rk_quant_nchw_f32_to_i8(p->x_unshuf, p->x_unshuf_i8, 1, SRCD, p->fH, p->fW,
                                     p->q_eh_in0.scale, p->q_eh_in0.zp);
        dcvc_rk_requant_nchw_i8(p->ctx_i8, 1, DCH, p->fH, p->fW, p->q_ctx_cur.w_stride,
                                p->q_ctx_cur.scale, p->q_ctx_cur.zp,
                                p->scratch_i8, p->q_eh_in1.scale, p->q_eh_in1.zp);
        if (s_eh) s_eh->wall_ms += dcvc_rk_now_ms() - t;
        {
            DcvcRkI8View vin[3] = {
                { p->x_unshuf_i8, 1, SRCD, p->fH, p->fW },
                { p->scratch_i8, 1, DCH, p->fH, p->fW },
                { p->q_enc_i8, 1, DCH, 1, 1 }
            };
            DcvcRkI8View vout[2] = {
                { p->y_i8, 1, YCH, p->yH, p->yW },
                { p->z_i8, 1, ZCH, p->zH, p->zW }
            };
            double t0 = dcvc_rk_now_ms();
            st = dcvc_rk_engine_run_i8(p->eng_enc_hyper, vin, 3, vout, 2);
            if (s_eh) s_eh->wall_ms += dcvc_rk_now_ms() - t0;
            if (st == DCVC_RK_OK) acc_eng(s_eh, p->eng_enc_hyper, &p->npu_us);
            if (st != DCVC_RK_OK) return st;
        }

        t = dcvc_rk_now_ms();
        dcvc_rk_dequant_nchw_i8_to_f32(p->z_i8, p->z, 1, ZCH, p->zH, p->zW,
                                       p->q_eh_z.w_stride, p->q_eh_z.scale, p->q_eh_z.zp);
        dcvc_rk_round_to_int8(p->z, p->z_hat, p->z_int8, ZCH * zHW);
        dcvc_rk_quant_nchw_f32_to_i8(p->z_hat, p->z_hat_i8, 1, ZCH, p->zH, p->zW,
                                     p->q_pc_z.scale, p->q_pc_z.zp);
        if (s_ze) s_ze->wall_ms += dcvc_rk_now_ms() - t;

        dcvc_rk_requant_nchw_i8(p->memory_i8, 1, DCH, p->fH, p->fW, p->q_mem_cur.w_stride,
                                p->q_mem_cur.scale, p->q_mem_cur.zp,
                                p->scratch_i8, p->q_pc_mem.scale, p->q_pc_mem.zp);

        uint8_t* z_stream = NULL; size_t z_sz = 0;
        InterZEncCtx zctx = { p, &z_stream, &z_sz };
        {
            DcvcRkI8View vin[3] = {
                { p->z_hat_i8, 1, ZCH, p->zH, p->zW },
                { p->scratch_i8, 1, DCH, p->fH, p->fW },
                { p->q_feat_i8, 1, DCH, 1, 1 }
            };
            DcvcRkI8View vout = { p->params_i8, 1, PFCH, p->yH, p->yW };
            st = dcvc_rk_overlap_npu_i8_cpu(p->eng_prior_chain, vin, 3, &vout, 1,
                                           inter_cpu_z_enc, &zctx, s_pc, s_ze, &p->npu_us);
            if (st != DCVC_RK_OK) { free(z_stream); return st; }
        }

        t = dcvc_rk_now_ms();
        dcvc_rk_dequant_nchw_i8_to_f32(p->y_i8, p->y, 1, YCH, p->yH, p->yW,
                                       p->q_eh_y.w_stride, p->q_eh_y.scale, p->q_eh_y.zp);
        dcvc_rk_dequant_nchw_i8_to_f32(p->params_i8, p->params, 1, PFCH, p->yH, p->yW,
                                       p->q_pc_out.w_stride, p->q_pc_out.scale, p->q_pc_out.zp);
        if (s_ar) s_ar->wall_ms += dcvc_rk_now_ms() - t;

        t = dcvc_rk_now_ms();
        uint8_t* y_stream = NULL; size_t y_sz = 0;
        st = dcvc_rk_ar_encode_y(p->ar, p->y, p->params, p->yH, p->yW, &y_stream, &y_sz, p->y_hat);
        if (s_ar) s_ar->wall_ms += dcvc_rk_now_ms() - t;
        fold_ar(s_ar, p->ar, &p->npu_us);
        if (st != DCVC_RK_OK) { free(z_stream); return st; }

        t = dcvc_rk_now_ms();
        dcvc_rk_quant_nchw_f32_to_i8(p->y_hat, p->y_hat_i8, 1, YCH, p->yH, p->yW,
                                     p->q_dr_y.scale, p->q_dr_y.zp);
        dcvc_rk_requant_nchw_i8(p->ctx_i8, 1, DCH, p->fH, p->fW, p->q_ctx_cur.w_stride,
                                p->q_ctx_cur.scale, p->q_ctx_cur.zp,
                                p->scratch_i8, p->q_dr_ctx.scale, p->q_dr_ctx.zp);
        if (s_dr) s_dr->wall_ms += dcvc_rk_now_ms() - t;

        InterTailCtx tail = {
            p, z_stream, z_sz, y_stream, y_sz, out_stream, out_size, next_x, s_pre
        };
        {
            DcvcRkI8View vin[4] = {
                { p->y_hat_i8, 1, YCH, p->yH, p->yW },
                { p->scratch_i8, 1, DCH, p->fH, p->fW },
                { p->q_dec_i8, 1, DCH, 1, 1 },
                { p->q_recon_i8, 1, RECON_Q, 1, 1 }
            };
            DcvcRkI8View vout[2] = {
                { p->feature_i8, 1, DCH, p->fH, p->fW },
                { p->recon_i8, 1, SRCD, p->fH, p->fW }
            };
            st = dcvc_rk_overlap_npu_i8_cpu(p->eng_dec_recon, vin, 4, vout, 2,
                                           inter_cpu_tail, &tail, s_dr, NULL, &p->npu_us);
            free(z_stream); free(y_stream);
            if (st != DCVC_RK_OK) return st;
        }

        memcpy(p->ref_feature_i8, p->feature_i8,
               (size_t)dcvc_rk_engine_out_buf_bytes(p->eng_dec_recon, 0));

        if (x_hat_out) {
            t = dcvc_rk_now_ms();
            dcvc_rk_dequant_nchw_i8_to_f32(p->recon_i8, p->recon_192, 1, SRCD, p->fH, p->fW,
                                           p->q_dr_recon.w_stride, p->q_dr_recon.scale, p->q_dr_recon.zp);
            dcvc_rk_pixel_shuffle_8(p->recon_192, p->frame_ycbcr, Hp, Wp);
            dcvc_rk_ycbcr_to_rgb(p->frame_ycbcr, p->frame_rgb, Hp * Wp);
            dcvc_rk_crop_3(p->frame_rgb, Hp, Wp, x_hat_out, p->H, p->W);
            if (s_post) s_post->wall_ms = dcvc_rk_now_ms() - t;
        }
        return DCVC_RK_OK;
    }

    {
        DcvcRkTensorView vin[3] = {
            { p->x_unshuf, 1, SRCD, p->fH, p->fW },
            { p->ctx, 1, DCH, p->fH, p->fW },
            { p->q_enc, 1, DCH, 1, 1 }
        };
        DcvcRkTensorView vout[2] = {
            { p->y, 1, YCH, p->yH, p->yW },
            { p->z, 1, ZCH, p->zH, p->zW }
        };
        st = run_io(p->eng_enc_hyper, vin, 3, vout, 2, s_eh, &p->npu_us);
        if (st != DCVC_RK_OK) return st;
    }

    double t = dcvc_rk_now_ms();
    dcvc_rk_round_to_int8(p->z, p->z_hat, p->z_int8, ZCH * zHW);
    if (s_ze) s_ze->wall_ms = dcvc_rk_now_ms() - t;

    uint8_t* z_stream = NULL; size_t z_sz = 0;
    InterZEncCtx zctx = { p, &z_stream, &z_sz };
    {
        DcvcRkTensorView vin[3] = {
            { p->z_hat, 1, ZCH, p->zH, p->zW },
            { p->memory, 1, DCH, p->fH, p->fW },
            { p->q_feat, 1, DCH, 1, 1 }
        };
        DcvcRkTensorView vout = { p->params, 1, PFCH, p->yH, p->yW };
        st = dcvc_rk_overlap_npu_cpu(p->eng_prior_chain, vin, 3, &vout, 1,
                                     inter_cpu_z_enc, &zctx, s_pc, s_ze, &p->npu_us);
        if (st != DCVC_RK_OK) { free(z_stream); return st; }
    }

    t = dcvc_rk_now_ms();
    uint8_t* y_stream = NULL; size_t y_sz = 0;
    st = dcvc_rk_ar_encode_y(p->ar, p->y, p->params, p->yH, p->yW, &y_stream, &y_sz, p->y_hat);
    if (s_ar) s_ar->wall_ms = dcvc_rk_now_ms() - t;
    fold_ar(s_ar, p->ar, &p->npu_us);
    if (st != DCVC_RK_OK) { free(z_stream); return st; }

    InterTailCtx tail = {
        p, z_stream, z_sz, y_stream, y_sz, out_stream, out_size, next_x, s_pre
    };
    {
        DcvcRkTensorView vin[4] = {
            { p->y_hat, 1, YCH, p->yH, p->yW },
            { p->ctx, 1, DCH, p->fH, p->fW },
            { p->q_dec, 1, DCH, 1, 1 },
            { p->q_recon, 1, RECON_Q, 1, 1 }
        };
        DcvcRkTensorView vout[2] = {
            { p->feature_dec, 1, DCH, p->fH, p->fW },
            { p->recon_192, 1, SRCD, p->fH, p->fW }
        };
        /* dec_recon NPU ∥ pack (+ optional next-frame preprocess) */
        st = dcvc_rk_overlap_npu_cpu(p->eng_dec_recon, vin, 4, vout, 2,
                                     inter_cpu_tail, &tail, s_dr, NULL, &p->npu_us);
        free(z_stream); free(y_stream);
        if (st != DCVC_RK_OK) return st;
    }

    memcpy(p->ref_feature, p->feature_dec, (size_t)DCH * p->fH * p->fW * sizeof(float));

    if (x_hat_out) {
        t = dcvc_rk_now_ms();
        dcvc_rk_pixel_shuffle_8(p->recon_192, p->frame_ycbcr, Hp, Wp);
        dcvc_rk_ycbcr_to_rgb(p->frame_ycbcr, p->frame_rgb, Hp * Wp);
        dcvc_rk_crop_3(p->frame_rgb, Hp, Wp, x_hat_out, p->H, p->W);
        if (s_post) s_post->wall_ms = dcvc_rk_now_ms() - t;
    }
    return DCVC_RK_OK;
}

DcvcRkStatus dcvc_rk_inter_encode(DcvcRkInterLdPipeline* p, const float* x,
                                  int reset, const float* x_ref,
                                  uint8_t** out_stream, size_t* out_size,
                                  float* x_hat_out)
{
    return dcvc_rk_inter_encode_ex(p, x, reset, x_ref, out_stream, out_size,
                                   x_hat_out, NULL);
}

DcvcRkStatus dcvc_rk_inter_encode_ex(DcvcRkInterLdPipeline* p, const float* x,
                                     int reset, const float* x_ref,
                                     uint8_t** out_stream, size_t* out_size,
                                     float* x_hat_out, const float* next_x)
{
    if (!p || !x || !out_stream || !out_size) return DCVC_RK_ERR_INVALID_ARG;
    *out_stream = NULL; *out_size = 0;
    return encode_core(p, x, reset, x_ref, out_stream, out_size, x_hat_out, next_x);
}

DcvcRkStatus dcvc_rk_inter_decode(DcvcRkInterLdPipeline* p,
                                  const uint8_t* stream, size_t stream_size,
                                  int reset, const float* x_ref,
                                  float* x_hat_out)
{
    if (!p || !stream || stream_size < 4 || !x_hat_out) return DCVC_RK_ERR_INVALID_ARG;
    int Hp = p->Hp, Wp = p->Wp, zHW = p->zH * p->zW;
    uint32_t z_sz = 0;
    memcpy(&z_sz, stream, 4);
    if (4 + z_sz > stream_size) return DCVC_RK_ERR_ENTROPY;
    const uint8_t* z_stream = stream + 4;
    const uint8_t* y_stream = stream + 4 + z_sz;
    size_t y_sz = stream_size - 4 - z_sz;

    dcvc_rk_profile_reset(&p->prof);
    DcvcRkStageProf* s_pre = dcvc_rk_profile_add(&p->prof, "preprocess");
    DcvcRkStageProf* s_feat = dcvc_rk_profile_add(&p->prof, "feat");
    DcvcRkStageProf* s_ze  = dcvc_rk_profile_add(&p->prof, "z_entropy");
    DcvcRkStageProf* s_pc  = dcvc_rk_profile_add(&p->prof, "prior_chain");
    DcvcRkStageProf* s_ar  = dcvc_rk_profile_add(&p->prof, "ar_y");
    DcvcRkStageProf* s_dr  = dcvc_rk_profile_add(&p->prof, "dec_recon");
    DcvcRkStageProf* s_post = dcvc_rk_profile_add(&p->prof, "postprocess");

    DcvcRkStatus st = build_memory(p, reset, x_ref, s_pre, s_feat);
    if (st != DCVC_RK_OK) return st;

    double t = dcvc_rk_now_ms();
    dcvc_rans_decoder_reset_cdf(p->rans_dec);
    int zidx = dcvc_rans_decoder_add_cdf(p->rans_dec, dcvc_npy_i32(&p->zcdf),
                    p->zcdf.dims[0], p->zcdf.dims[1],
                    dcvc_npy_i32(&p->zlen), dcvc_npy_i32(&p->zoff));
    dcvc_rans_decoder_set_stream(p->rans_dec, z_stream, z_sz);
    dcvc_rans_decoder_decode_z(p->rans_dec, ZCH * zHW, zidx, p->qp * ZCH, zHW);
    int8_t* zsyms = NULL; size_t zn = 0;
    dcvc_rans_decoder_get_symbols(p->rans_dec, &zsyms, &zn);
    if (dcvc_rans_decoder_has_error(p->rans_dec)) { free(zsyms); return DCVC_RK_ERR_ENTROPY; }
    dcvc_rk_int8_to_float(zsyms, p->z_hat, ZCH * zHW);
    free(zsyms);
    if (s_ze) s_ze->wall_ms = dcvc_rk_now_ms() - t;

    if (p->pipe_i8) {
        t = dcvc_rk_now_ms();
        dcvc_rk_quant_nchw_f32_to_i8(p->z_hat, p->z_hat_i8, 1, ZCH, p->zH, p->zW,
                                     p->q_pc_z.scale, p->q_pc_z.zp);
        dcvc_rk_requant_nchw_i8(p->memory_i8, 1, DCH, p->fH, p->fW, p->q_mem_cur.w_stride,
                                p->q_mem_cur.scale, p->q_mem_cur.zp,
                                p->scratch_i8, p->q_pc_mem.scale, p->q_pc_mem.zp);
        if (s_pc) s_pc->wall_ms += dcvc_rk_now_ms() - t;
        {
            DcvcRkI8View vin[3] = {
                { p->z_hat_i8, 1, ZCH, p->zH, p->zW },
                { p->scratch_i8, 1, DCH, p->fH, p->fW },
                { p->q_feat_i8, 1, DCH, 1, 1 }
            };
            DcvcRkI8View vout = { p->params_i8, 1, PFCH, p->yH, p->yW };
            double t0 = dcvc_rk_now_ms();
            st = dcvc_rk_engine_run_i8(p->eng_prior_chain, vin, 3, &vout, 1);
            if (s_pc) s_pc->wall_ms += dcvc_rk_now_ms() - t0;
            if (st == DCVC_RK_OK) acc_eng(s_pc, p->eng_prior_chain, &p->npu_us);
            if (st != DCVC_RK_OK) return st;
        }

        t = dcvc_rk_now_ms();
        dcvc_rk_dequant_nchw_i8_to_f32(p->params_i8, p->params, 1, PFCH, p->yH, p->yW,
                                       p->q_pc_out.w_stride, p->q_pc_out.scale, p->q_pc_out.zp);
        if (s_ar) s_ar->wall_ms += dcvc_rk_now_ms() - t;

        t = dcvc_rk_now_ms();
        st = dcvc_rk_ar_decode_y(p->ar, p->params, p->yH, p->yW, y_stream, y_sz, p->y_hat);
        if (s_ar) s_ar->wall_ms += dcvc_rk_now_ms() - t;
        fold_ar(s_ar, p->ar, &p->npu_us);
        if (st != DCVC_RK_OK) return st;

        t = dcvc_rk_now_ms();
        dcvc_rk_quant_nchw_f32_to_i8(p->y_hat, p->y_hat_i8, 1, YCH, p->yH, p->yW,
                                     p->q_dr_y.scale, p->q_dr_y.zp);
        dcvc_rk_requant_nchw_i8(p->ctx_i8, 1, DCH, p->fH, p->fW, p->q_ctx_cur.w_stride,
                                p->q_ctx_cur.scale, p->q_ctx_cur.zp,
                                p->scratch_i8, p->q_dr_ctx.scale, p->q_dr_ctx.zp);
        if (s_dr) s_dr->wall_ms += dcvc_rk_now_ms() - t;
        {
            DcvcRkI8View vin[4] = {
                { p->y_hat_i8, 1, YCH, p->yH, p->yW },
                { p->scratch_i8, 1, DCH, p->fH, p->fW },
                { p->q_dec_i8, 1, DCH, 1, 1 },
                { p->q_recon_i8, 1, RECON_Q, 1, 1 }
            };
            DcvcRkI8View vout[2] = {
                { p->feature_i8, 1, DCH, p->fH, p->fW },
                { p->recon_i8, 1, SRCD, p->fH, p->fW }
            };
            double t0 = dcvc_rk_now_ms();
            st = dcvc_rk_engine_run_i8(p->eng_dec_recon, vin, 4, vout, 2);
            if (s_dr) s_dr->wall_ms += dcvc_rk_now_ms() - t0;
            if (st == DCVC_RK_OK) acc_eng(s_dr, p->eng_dec_recon, &p->npu_us);
            if (st != DCVC_RK_OK) return st;
        }

        memcpy(p->ref_feature_i8, p->feature_i8,
               (size_t)dcvc_rk_engine_out_buf_bytes(p->eng_dec_recon, 0));

        t = dcvc_rk_now_ms();
        dcvc_rk_dequant_nchw_i8_to_f32(p->recon_i8, p->recon_192, 1, SRCD, p->fH, p->fW,
                                       p->q_dr_recon.w_stride, p->q_dr_recon.scale, p->q_dr_recon.zp);
        dcvc_rk_pixel_shuffle_8(p->recon_192, p->frame_ycbcr, Hp, Wp);
        dcvc_rk_ycbcr_to_rgb(p->frame_ycbcr, p->frame_rgb, Hp * Wp);
        dcvc_rk_crop_3(p->frame_rgb, Hp, Wp, x_hat_out, p->H, p->W);
        if (s_post) s_post->wall_ms = dcvc_rk_now_ms() - t;
        return DCVC_RK_OK;
    }

    {
        DcvcRkTensorView vin[3] = {
            { p->z_hat, 1, ZCH, p->zH, p->zW },
            { p->memory, 1, DCH, p->fH, p->fW },
            { p->q_feat, 1, DCH, 1, 1 }
        };
        DcvcRkTensorView vout = { p->params, 1, PFCH, p->yH, p->yW };
        st = run_io(p->eng_prior_chain, vin, 3, &vout, 1, s_pc, &p->npu_us);
        if (st != DCVC_RK_OK) return st;
    }

    t = dcvc_rk_now_ms();
    st = dcvc_rk_ar_decode_y(p->ar, p->params, p->yH, p->yW, y_stream, y_sz, p->y_hat);
    if (s_ar) s_ar->wall_ms = dcvc_rk_now_ms() - t;
    fold_ar(s_ar, p->ar, &p->npu_us);
    if (st != DCVC_RK_OK) return st;

    {
        DcvcRkTensorView vin[4] = {
            { p->y_hat, 1, YCH, p->yH, p->yW },
            { p->ctx, 1, DCH, p->fH, p->fW },
            { p->q_dec, 1, DCH, 1, 1 },
            { p->q_recon, 1, RECON_Q, 1, 1 }
        };
        DcvcRkTensorView vout[2] = {
            { p->feature_dec, 1, DCH, p->fH, p->fW },
            { p->recon_192, 1, SRCD, p->fH, p->fW }
        };
        st = run_io(p->eng_dec_recon, vin, 4, vout, 2, s_dr, &p->npu_us);
        if (st != DCVC_RK_OK) return st;
    }

    memcpy(p->ref_feature, p->feature_dec, (size_t)DCH * p->fH * p->fW * sizeof(float));
    t = dcvc_rk_now_ms();
    dcvc_rk_pixel_shuffle_8(p->recon_192, p->frame_ycbcr, Hp, Wp);
    dcvc_rk_ycbcr_to_rgb(p->frame_ycbcr, p->frame_rgb, Hp * Wp);
    dcvc_rk_crop_3(p->frame_rgb, Hp, Wp, x_hat_out, p->H, p->W);
    if (s_post) s_post->wall_ms = dcvc_rk_now_ms() - t;
    return DCVC_RK_OK;
}
