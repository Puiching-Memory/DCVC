#include "dcvc_rk/inter_ld_pipeline.h"
#include "dcvc_rk/ar_codec.h"
#include "dcvc_rk/kernels.h"
#include "dcvc_rk/profile.h"
#include "dcvc_rk/rknn_engine.h"
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
    DcvcRkEngine *eng_fai, *eng_fap, *eng_fe, *eng_enc, *eng_henc, *eng_hdec;
    DcvcRkEngine *eng_temp, *eng_pfus, *eng_dec, *eng_recon;
    DcvcRkArCodec* ar;
    DcvcRansEncoder* rans_enc;
    DcvcRansDecoder* rans_dec;
    DcvcNpy zcdf, zlen, zoff;
    float *q_enc, *q_dec, *q_feat, *q_recon;
    float *memory, *ref_feature, *ctx, *x_unshuf, *ref_unshuf;
    float *y, *z, *z_hat, *hier, *temporal, *pf_in, *params, *y_hat, *feature_dec;
    float *recon_192, *frame_pad, *frame_ycbcr, *frame_rgb;
    int8_t* z_int8;
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

static DcvcRkStatus run1(DcvcRkEngine* e, const float* in, int c, int h, int w,
                         float* out, int oc, int oh, int ow,
                         DcvcRkStageProf* s, int64_t* acc)
{
    DcvcRkTensorView vin = { (void*)in, 1, c, h, w };
    DcvcRkTensorView vout = { out, 1, oc, oh, ow };
    double t0 = dcvc_rk_now_ms();
    DcvcRkStatus st = dcvc_rk_engine_run(e, &vin, 1, &vout, 1);
    if (s) s->wall_ms += dcvc_rk_now_ms() - t0;
    if (st == DCVC_RK_OK) acc_eng(s, e, acc);
    return st;
}

static DcvcRkStatus run2(DcvcRkEngine* e,
                         const float* a, int ca, int ha, int wa,
                         const float* b, int cb, int hb, int wb,
                         float* out, int oc, int oh, int ow,
                         DcvcRkStageProf* s, int64_t* acc)
{
    DcvcRkTensorView vin[2] = {
        { (void*)a, 1, ca, ha, wa }, { (void*)b, 1, cb, hb, wb }
    };
    DcvcRkTensorView vout = { out, 1, oc, oh, ow };
    double t0 = dcvc_rk_now_ms();
    DcvcRkStatus st = dcvc_rk_engine_run(e, vin, 2, &vout, 1);
    if (s) s->wall_ms += dcvc_rk_now_ms() - t0;
    if (st == DCVC_RK_OK) acc_eng(s, e, acc);
    return st;
}

static DcvcRkStatus run3(DcvcRkEngine* e,
                         const float* a, int ca, int ha, int wa,
                         const float* b, int cb, int hb, int wb,
                         const float* c, int cc, int hc, int wc,
                         float* out, int oc, int oh, int ow,
                         DcvcRkStageProf* s, int64_t* acc)
{
    DcvcRkTensorView vin[3] = {
        { (void*)a, 1, ca, ha, wa },
        { (void*)b, 1, cb, hb, wb },
        { (void*)c, 1, cc, hc, wc }
    };
    DcvcRkTensorView vout = { out, 1, oc, oh, ow };
    double t0 = dcvc_rk_now_ms();
    DcvcRkStatus st = dcvc_rk_engine_run(e, vin, 3, &vout, 1);
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

static DcvcRkStatus build_memory(DcvcRkInterLdPipeline* p, int reset, const float* x_ref,
                                 DcvcRkStageProf* s_pre, DcvcRkStageProf* s_fa,
                                 DcvcRkStageProf* s_fe)
{
    DcvcRkStatus st;
    if (reset) {
        if (!x_ref) return DCVC_RK_ERR_INVALID_ARG;
        double t = dcvc_rk_now_ms();
        dcvc_rk_replicate_pad_3(x_ref, p->H, p->W, p->frame_pad, p->Hp, p->Wp);
        dcvc_rk_rgb_to_ycbcr(p->frame_pad, p->frame_ycbcr, p->Hp * p->Wp);
        dcvc_rk_pixel_unshuffle_8(p->frame_ycbcr, p->ref_unshuf, p->Hp, p->Wp);
        if (s_pre) s_pre->wall_ms += dcvc_rk_now_ms() - t;
        st = run1(p->eng_fai, p->ref_unshuf, SRCD, p->fH, p->fW,
                  p->memory, DCH, p->fH, p->fW, s_fa, &p->npu_us);
    } else {
        st = run1(p->eng_fap, p->ref_feature, DCH, p->fH, p->fW,
                  p->memory, DCH, p->fH, p->fW, s_fa, &p->npu_us);
    }
    if (st != DCVC_RK_OK) return st;
    return run1(p->eng_fe, p->memory, DCH, p->fH, p->fW, p->ctx, DCH, p->fH, p->fW,
                s_fe, &p->npu_us);
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
    LOAD(eng_fai, "inter_feature_adaptor_i");
    LOAD(eng_fap, "inter_feature_adaptor_p");
    LOAD(eng_fe, "inter_feature_extractor");
    LOAD(eng_enc, "inter_encoder");
    LOAD(eng_henc, "inter_hyper_enc");
    LOAD(eng_hdec, "inter_hyper_dec");
    LOAD(eng_temp, "inter_temporal_prior");
    LOAD(eng_pfus, "inter_prior_fusion");
    LOAD(eng_dec, "inter_decoder");
    LOAD(eng_recon, "recon_generation");
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
    M(p->x_unshuf, SRCD * fHW); M(p->ref_unshuf, SRCD * fHW);
    M(p->y, YCH * yHW); M(p->z, ZCH * zHW); M(p->z_hat, ZCH * zHW);
    M(p->hier, YCH * yHW); M(p->temporal, DCH * yHW);
    M(p->pf_in, PFCH * yHW); M(p->params, PFCH * yHW); M(p->y_hat, YCH * yHW);
    M(p->feature_dec, DCH * fHW); M(p->recon_192, SRCD * fHW);
    M(p->frame_pad, 3 * HpWp); M(p->frame_ycbcr, 3 * HpWp); M(p->frame_rgb, 3 * HpWp);
    #undef M
    p->z_int8 = (int8_t*)calloc(ZCH * zHW, 1);
    if (!p->z_int8) { st = DCVC_RK_ERR_OOM; goto fail; }
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
    D(eng_fai); D(eng_fap); D(eng_fe); D(eng_enc); D(eng_henc); D(eng_hdec);
    D(eng_temp); D(eng_pfus); D(eng_dec); D(eng_recon);
    #undef D
    if (p->ar) dcvc_rk_ar_codec_destroy(p->ar);
    if (p->rans_enc) dcvc_rans_encoder_destroy(p->rans_enc);
    if (p->rans_dec) dcvc_rans_decoder_destroy(p->rans_dec);
    dcvc_npy_free(&p->zcdf); dcvc_npy_free(&p->zlen); dcvc_npy_free(&p->zoff);
    free(p->q_enc); free(p->q_dec); free(p->q_feat); free(p->q_recon);
    free(p->memory); free(p->ref_feature); free(p->ctx); free(p->x_unshuf); free(p->ref_unshuf);
    free(p->y); free(p->z); free(p->z_hat); free(p->hier); free(p->temporal);
    free(p->pf_in); free(p->params); free(p->y_hat); free(p->feature_dec); free(p->recon_192);
    free(p->frame_pad); free(p->frame_ycbcr); free(p->frame_rgb); free(p->z_int8);
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

static DcvcRkStatus encode_core(DcvcRkInterLdPipeline* p, const float* x,
                                int reset, const float* x_ref,
                                uint8_t** out_stream, size_t* out_size,
                                float* x_hat_out)
{
    int Hp = p->Hp, Wp = p->Wp, yHW = p->yH * p->yW, zHW = p->zH * p->zW;
    dcvc_rk_profile_reset(&p->prof);
    DcvcRkStageProf* s_pre = dcvc_rk_profile_add(&p->prof, "preprocess");
    DcvcRkStageProf* s_fa  = dcvc_rk_profile_add(&p->prof, "feat_adaptor");
    DcvcRkStageProf* s_fe  = dcvc_rk_profile_add(&p->prof, "feat_extract");
    DcvcRkStageProf* s_enc = dcvc_rk_profile_add(&p->prof, "inter_encoder");
    DcvcRkStageProf* s_he  = dcvc_rk_profile_add(&p->prof, "hyper_enc");
    DcvcRkStageProf* s_ze  = dcvc_rk_profile_add(&p->prof, "z_entropy");
    DcvcRkStageProf* s_hd  = dcvc_rk_profile_add(&p->prof, "hyper_dec");
    DcvcRkStageProf* s_tp  = dcvc_rk_profile_add(&p->prof, "temporal_prior");
    DcvcRkStageProf* s_pf  = dcvc_rk_profile_add(&p->prof, "prior_fusion");
    DcvcRkStageProf* s_ar  = dcvc_rk_profile_add(&p->prof, "ar_y");
    DcvcRkStageProf* s_dec = dcvc_rk_profile_add(&p->prof, "inter_decoder");
    DcvcRkStageProf* s_rec = dcvc_rk_profile_add(&p->prof, "recon_gen");
    DcvcRkStageProf* s_post = dcvc_rk_profile_add(&p->prof, "postprocess");

    DcvcRkStatus st = build_memory(p, reset, x_ref, s_pre, s_fa, s_fe);
    if (st != DCVC_RK_OK) return st;

    double t = dcvc_rk_now_ms();
    dcvc_rk_replicate_pad_3(x, p->H, p->W, p->frame_pad, Hp, Wp);
    dcvc_rk_rgb_to_ycbcr(p->frame_pad, p->frame_ycbcr, Hp * Wp);
    dcvc_rk_pixel_unshuffle_8(p->frame_ycbcr, p->x_unshuf, Hp, Wp);
    if (s_pre) s_pre->wall_ms += dcvc_rk_now_ms() - t;

    st = run3(p->eng_enc, p->x_unshuf, SRCD, p->fH, p->fW,
              p->ctx, DCH, p->fH, p->fW, p->q_enc, DCH, 1, 1,
              p->y, YCH, p->yH, p->yW, s_enc, &p->npu_us);
    if (st != DCVC_RK_OK) return st;
    st = run1(p->eng_henc, p->y, YCH, p->yH, p->yW, p->z, ZCH, p->zH, p->zW,
              s_he, &p->npu_us);
    if (st != DCVC_RK_OK) return st;

    t = dcvc_rk_now_ms();
    dcvc_rk_round_to_int8(p->z, p->z_hat, p->z_int8, ZCH * zHW);
    dcvc_rans_encoder_reset(p->rans_enc);
    int zidx = dcvc_rans_encoder_add_cdf(p->rans_enc, dcvc_npy_i32(&p->zcdf),
                    p->zcdf.dims[0], p->zcdf.dims[1],
                    dcvc_npy_i32(&p->zlen), dcvc_npy_i32(&p->zoff));
    dcvc_rans_encoder_encode_z(p->rans_enc, p->z_int8, ZCH * zHW, zidx, p->qp * ZCH, zHW);
    dcvc_rans_encoder_flush(p->rans_enc);
    uint8_t* z_stream = NULL; size_t z_sz = 0;
    if (dcvc_rans_encoder_get_stream(p->rans_enc, &z_stream, &z_sz) != 0)
        return DCVC_RK_ERR_ENTROPY;
    if (s_ze) s_ze->wall_ms = dcvc_rk_now_ms() - t;

    st = run1(p->eng_hdec, p->z_hat, ZCH, p->zH, p->zW, p->hier, YCH, p->yH, p->yW,
              s_hd, &p->npu_us);
    if (st != DCVC_RK_OK) { free(z_stream); return st; }
    st = run1(p->eng_temp, p->memory, DCH, p->fH, p->fW, p->temporal, DCH, p->yH, p->yW,
              s_tp, &p->npu_us);
    if (st != DCVC_RK_OK) { free(z_stream); return st; }

    t = dcvc_rk_now_ms();
    dcvc_rk_mul_q_broadcast(p->temporal, p->q_feat, p->temporal, DCH, yHW);
    memcpy(p->pf_in, p->hier, (size_t)YCH * yHW * sizeof(float));
    memcpy(p->pf_in + YCH * yHW, p->temporal, (size_t)DCH * yHW * sizeof(float));
    double t_cpu = dcvc_rk_now_ms() - t;
    st = run1(p->eng_pfus, p->pf_in, PFCH, p->yH, p->yW, p->params, PFCH, p->yH, p->yW,
              s_pf, &p->npu_us);
    if (s_pf) s_pf->wall_ms += t_cpu; /* include cat/mul in prior_fusion wall */
    if (st != DCVC_RK_OK) { free(z_stream); return st; }

    t = dcvc_rk_now_ms();
    uint8_t* y_stream = NULL; size_t y_sz = 0;
    st = dcvc_rk_ar_encode_y(p->ar, p->y, p->params, p->yH, p->yW, &y_stream, &y_sz, p->y_hat);
    if (s_ar) s_ar->wall_ms = dcvc_rk_now_ms() - t;
    fold_ar(s_ar, p->ar, &p->npu_us);
    if (st != DCVC_RK_OK) { free(z_stream); return st; }

    st = run3(p->eng_dec, p->y_hat, YCH, p->yH, p->yW,
              p->ctx, DCH, p->fH, p->fW, p->q_dec, DCH, 1, 1,
              p->feature_dec, DCH, p->fH, p->fW, s_dec, &p->npu_us);
    if (st != DCVC_RK_OK) { free(z_stream); free(y_stream); return st; }
    st = run2(p->eng_recon, p->feature_dec, DCH, p->fH, p->fW, p->q_recon, RECON_Q, 1, 1,
              p->recon_192, SRCD, p->fH, p->fW, s_rec, &p->npu_us);
    if (st != DCVC_RK_OK) { free(z_stream); free(y_stream); return st; }

    memcpy(p->ref_feature, p->feature_dec, (size_t)DCH * p->fH * p->fW * sizeof(float));

    size_t total = 4 + z_sz + y_sz;
    uint8_t* buf = (uint8_t*)malloc(total);
    if (!buf) { free(z_stream); free(y_stream); return DCVC_RK_ERR_OOM; }
    uint32_t zl = (uint32_t)z_sz;
    memcpy(buf, &zl, 4);
    if (z_sz) memcpy(buf + 4, z_stream, z_sz);
    if (y_sz) memcpy(buf + 4 + z_sz, y_stream, y_sz);
    free(z_stream); free(y_stream);
    *out_stream = buf; *out_size = total;

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
    if (!p || !x || !out_stream || !out_size) return DCVC_RK_ERR_INVALID_ARG;
    *out_stream = NULL; *out_size = 0;
    return encode_core(p, x, reset, x_ref, out_stream, out_size, x_hat_out);
}

DcvcRkStatus dcvc_rk_inter_decode(DcvcRkInterLdPipeline* p,
                                  const uint8_t* stream, size_t stream_size,
                                  int reset, const float* x_ref,
                                  float* x_hat_out)
{
    if (!p || !stream || stream_size < 4 || !x_hat_out) return DCVC_RK_ERR_INVALID_ARG;
    int Hp = p->Hp, Wp = p->Wp, yHW = p->yH * p->yW, zHW = p->zH * p->zW;
    uint32_t z_sz = 0;
    memcpy(&z_sz, stream, 4);
    if (4 + z_sz > stream_size) return DCVC_RK_ERR_ENTROPY;
    const uint8_t* z_stream = stream + 4;
    const uint8_t* y_stream = stream + 4 + z_sz;
    size_t y_sz = stream_size - 4 - z_sz;

    dcvc_rk_profile_reset(&p->prof);
    DcvcRkStageProf* s_pre = dcvc_rk_profile_add(&p->prof, "preprocess");
    DcvcRkStageProf* s_fa  = dcvc_rk_profile_add(&p->prof, "feat_adaptor");
    DcvcRkStageProf* s_fe  = dcvc_rk_profile_add(&p->prof, "feat_extract");
    DcvcRkStageProf* s_ze  = dcvc_rk_profile_add(&p->prof, "z_entropy");
    DcvcRkStageProf* s_hd  = dcvc_rk_profile_add(&p->prof, "hyper_dec");
    DcvcRkStageProf* s_tp  = dcvc_rk_profile_add(&p->prof, "temporal_prior");
    DcvcRkStageProf* s_pf  = dcvc_rk_profile_add(&p->prof, "prior_fusion");
    DcvcRkStageProf* s_ar  = dcvc_rk_profile_add(&p->prof, "ar_y");
    DcvcRkStageProf* s_dec = dcvc_rk_profile_add(&p->prof, "inter_decoder");
    DcvcRkStageProf* s_rec = dcvc_rk_profile_add(&p->prof, "recon_gen");
    DcvcRkStageProf* s_post = dcvc_rk_profile_add(&p->prof, "postprocess");

    DcvcRkStatus st = build_memory(p, reset, x_ref, s_pre, s_fa, s_fe);
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

    st = run1(p->eng_hdec, p->z_hat, ZCH, p->zH, p->zW, p->hier, YCH, p->yH, p->yW,
              s_hd, &p->npu_us);
    if (st != DCVC_RK_OK) return st;
    st = run1(p->eng_temp, p->memory, DCH, p->fH, p->fW, p->temporal, DCH, p->yH, p->yW,
              s_tp, &p->npu_us);
    if (st != DCVC_RK_OK) return st;

    t = dcvc_rk_now_ms();
    dcvc_rk_mul_q_broadcast(p->temporal, p->q_feat, p->temporal, DCH, yHW);
    memcpy(p->pf_in, p->hier, (size_t)YCH * yHW * sizeof(float));
    memcpy(p->pf_in + YCH * yHW, p->temporal, (size_t)DCH * yHW * sizeof(float));
    double t_cpu = dcvc_rk_now_ms() - t;
    st = run1(p->eng_pfus, p->pf_in, PFCH, p->yH, p->yW, p->params, PFCH, p->yH, p->yW,
              s_pf, &p->npu_us);
    if (s_pf) s_pf->wall_ms += t_cpu;
    if (st != DCVC_RK_OK) return st;

    t = dcvc_rk_now_ms();
    st = dcvc_rk_ar_decode_y(p->ar, p->params, p->yH, p->yW, y_stream, y_sz, p->y_hat);
    if (s_ar) s_ar->wall_ms = dcvc_rk_now_ms() - t;
    fold_ar(s_ar, p->ar, &p->npu_us);
    if (st != DCVC_RK_OK) return st;

    st = run3(p->eng_dec, p->y_hat, YCH, p->yH, p->yW,
              p->ctx, DCH, p->fH, p->fW, p->q_dec, DCH, 1, 1,
              p->feature_dec, DCH, p->fH, p->fW, s_dec, &p->npu_us);
    if (st != DCVC_RK_OK) return st;
    st = run2(p->eng_recon, p->feature_dec, DCH, p->fH, p->fW, p->q_recon, RECON_Q, 1, 1,
              p->recon_192, SRCD, p->fH, p->fW, s_rec, &p->npu_us);
    if (st != DCVC_RK_OK) return st;

    memcpy(p->ref_feature, p->feature_dec, (size_t)DCH * p->fH * p->fW * sizeof(float));
    t = dcvc_rk_now_ms();
    dcvc_rk_pixel_shuffle_8(p->recon_192, p->frame_ycbcr, Hp, Wp);
    dcvc_rk_ycbcr_to_rgb(p->frame_ycbcr, p->frame_rgb, Hp * Wp);
    dcvc_rk_crop_3(p->frame_rgb, Hp, Wp, x_hat_out, p->H, p->W);
    if (s_post) s_post->wall_ms = dcvc_rk_now_ms() - t;
    return DCVC_RK_OK;
}
