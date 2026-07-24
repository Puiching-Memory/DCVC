/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Pure-CPU inter-frame (P-frame) pipeline using ONNX Runtime + rANS.
 * Implements the DCVC-RT video model (DMC) 2x-prior path. Reference is the
 * previous reconstructed frame (pixel domain) -> feature_adaptor_i.
 *
 * Encode/decode use identical FP32 ONNX models, so the closed loop (encoder
 * reconstruction == decoder reconstruction, and the DPB reference feature)
 * is bit-exact across platforms, exactly like the intra codec.
 */
#include "cpu_inter_pipeline.h"
#include "npy_reader.h"
#include "rans_c.h"
#include "fxp/fxp_scale_index.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void inter_dump(const char* env, const float* d, int c, int h, int w)
{
    const char* path = getenv(env);
    if (!path || !path[0]) return;
    char frame_path[1024];
    const char* frame_idx = getenv("DCVC_DUMP_FRAME");
    if (frame_idx && frame_idx[0]) {
        snprintf(frame_path, sizeof(frame_path), "%s_f%s.npy", path, frame_idx);
    } else {
        snprintf(frame_path, sizeof(frame_path), "%s.npy", path);
    }
    int dims[4] = {1, c, h, w};
    if (dcvc_npy_write_f32(frame_path, d, dims, 4) == 0)
        fprintf(stderr, "[inter] dumped %s [%dx%dx%d] to %s\n", env, c, h, w, frame_path);
}

/* Channel constants (src/models/video_model.py) */
#define YCH   128   /* g_ch_y  */
#define ZCH   128   /* g_ch_z  */
#define DCH   256   /* g_ch_d  */
#define SRCD  192   /* g_ch_src_d = 3*8*8 */
#define RCH   320   /* g_ch_recon */

struct DcvcCpuInterPipeline {
    int H, W;            /* actual (cropped) frame size        */
    int Hp, Wp;          /* padded size, multiple of 64        */
    int fH, fW, yH, yW, zH, zW, qp;  /* derived from PADDED size */

    DcvcCpuEngine* eng_fai;     /* feature_adaptor_i */
    DcvcCpuEngine* eng_fe;      /* feature_extractor -> ctx, ctx_t */
    DcvcCpuEngine* eng_enc;     /* inter_encoder */
    DcvcCpuEngine* eng_henc;    /* hyper_encoder */
    DcvcCpuEngine* eng_hdec;    /* hyper_decoder */
    DcvcCpuEngine* eng_temp;    /* temporal_prior */
    DcvcCpuEngine* eng_pfus;    /* prior_fusion */
    DcvcCpuEngine* eng_sp;      /* spatial_prior */
    DcvcCpuEngine* eng_dec;     /* inter_decoder */
    DcvcCpuEngine* eng_recon;   /* recon_generation (pre pixel_shuffle) */

    DcvcRansEncoder* rans_enc;
    DcvcRansDecoder* rans_dec;
    DcvcNpy gcdf, glen, goff;   /* gaussian CDF (shared) */
    DcvcNpy zcdf, zlen, zoff;   /* bit-estimator CDF (shared) */
    int g_cdf_idx, z_cdf_idx;

    float scale_min, scale_max, log_scale_min, log_step_recip;
    int scale_level;
    float skip_thres;  /* adaptive entropy: force-zero symbols with scale <= this */

    float *q_enc, *q_dec, *q_feat, *q_recon;   /* [DCH], [DCH], [DCH], [RCH] */

    /* workspace */
    float *ref_unshuf, *feature, *ctx, *ctx_t, *x_unshuf;
    float *y, *z, *z_hat, *fdec;
    int8_t* z_int8;
    float *hier, *temporal, *pfus_in, *params;
    float *scales, *means, *q_dec_v, *yhat_acc;
    float *masks[2];
    float *scl_mask, *sr, *yq_full, *yq_w, *cat_sp, *sp_out;
    int16_t* packed;
    uint8_t* indexes;
    float *x_pad, *ref_pad;  /* replicate-padded RGB pixel inputs [3,Hp,Wp] */
    float *x_ycbcr, *ref_ycbcr; /* intermediate YCbCr for NN [3,Hp,Wp] */
    float *rg_out, *x_hat;   /* x_hat is RGB [3,Hp,Wp], cropped on output */
};

/* ---------- engine run helpers ---------- */
static DcvcCpuStatus run1(DcvcCpuEngine* e, const float* in, int c, int h, int w,
                          float* out, int oc, int oh, int ow)
{
    DcvcCpuTensorView vin = { (void*)in, 0, 1, c, h, w };
    DcvcCpuTensorView vout = { out, 0, 1, oc, oh, ow };
    return dcvc_cpu_engine_run(e, &vin, 1, &vout, 1);
}
static DcvcCpuStatus run2(DcvcCpuEngine* e,
                          const float* i0, int c0, int h0, int w0,
                          const float* i1, int c1, int h1, int w1,
                          float* out, int oc, int oh, int ow)
{
    DcvcCpuTensorView vin[2] = { { (void*)i0, 0, 1, c0, h0, w0 },
                                  { (void*)i1, 0, 1, c1, h1, w1 } };
    DcvcCpuTensorView vout = { out, 0, 1, oc, oh, ow };
    return dcvc_cpu_engine_run(e, vin, 2, &vout, 1);
}
static DcvcCpuStatus run3(DcvcCpuEngine* e,
                          const float* i0, int c0, int h0, int w0,
                          const float* i1, int c1, int h1, int w1,
                          const float* i2, int c2, int h2, int w2,
                          float* out, int oc, int oh, int ow)
{
    DcvcCpuTensorView vin[3] = { { (void*)i0, 0, 1, c0, h0, w0 },
                                  { (void*)i1, 0, 1, c1, h1, w1 },
                                  { (void*)i2, 0, 1, c2, h2, w2 } };
    DcvcCpuTensorView vout = { out, 0, 1, oc, oh, ow };
    return dcvc_cpu_engine_run(e, vin, 3, &vout, 1);
}
/* feature_extractor: 2 inputs (feature, q_feat) -> 2 outputs (ctx, ctx_t) */
static DcvcCpuStatus run_fe(DcvcCpuEngine* e,
                            const float* feature, const float* qfeat,
                            float* ctx, float* ctx_t, int fh, int fw)
{
    DcvcCpuTensorView vin[2] = { { (void*)feature, 0, 1, DCH, fh, fw },
                                  { (void*)qfeat, 0, 1, DCH, 1, 1 } };
    DcvcCpuTensorView vout[2] = { { ctx, 0, 1, DCH, fh, fw },
                                   { ctx_t, 0, 1, DCH, fh, fw } };
    return dcvc_cpu_engine_run(e, vin, 2, vout, 2);
}

/* ---------- math kernels ---------- */
static void round_to_int8(const float* in, float* outf, int8_t* out8, int n)
{
    for (int i = 0; i < n; i++) {
        float v = roundf(in[i]);
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        outf[i] = v;
        out8[i] = (int8_t)v;
    }
}
static void int8_to_float(const int8_t* in, float* out, int n)
{
    for (int i = 0; i < n; i++) out[i] = (float)in[i];
}
/* sp2x: sum two channel halves (n = half-count). out[i]=x[i]+x[i+n] */
static void sp2x(const float* x, float* out, int n)
{
    for (int i = 0; i < n; i++) out[i] = x[i] + x[i + n];
}
static void build_index_dec(const float* s, uint8_t* out,
                            float smin, float smax, float lmin, float lrecip, int n)
{
    (void)smin; (void)smax; (void)lmin; (void)lrecip;
    dcvc_build_index_dec_i(s, out, n);
}
static void build_index_enc(const float* sym, const float* s, int16_t* out,
                            float smin, float smax, float lmin, float lrecip, int n)
{
    (void)smin; (void)smax; (void)lmin; (void)lrecip;
    dcvc_build_index_enc_i(sym, s, out, n);
}
/* pixel_unshuffle(x[3,H,W], out[192,fH,fW], factor 8) */
static void pixel_unshuffle_8(const float* x, float* out, int H, int W, int fH, int fW)
{
    int fHW = fH * fW;
    for (int ci = 0; ci < 3; ci++)
        for (int dy = 0; dy < 8; dy++)
            for (int dx = 0; dx < 8; dx++) {
                int oc = ci * 64 + dy * 8 + dx;
                float* o = out + (size_t)oc * fHW;
                for (int i = 0; i < fH; i++) {
                    const float* row = x + (size_t)ci * H * W + (size_t)(i * 8 + dy) * W;
                    for (int j = 0; j < fW; j++)
                        o[i * fW + j] = row[j * 8 + dx];
                }
            }
}
/* pixel_shuffle(rg[192,fH,fW], out[3,H,W], factor 8) + clamp[0,1] */
static void pixel_shuffle_8(const float* rg, float* out, int H, int W, int fH, int fW)
{
    int fHW = fH * fW;
    for (int ci = 0; ci < 3; ci++)
        for (int dy = 0; dy < 8; dy++)
            for (int dx = 0; dx < 8; dx++) {
                int oc = ci * 64 + dy * 8 + dx;
                const float* in = rg + (size_t)oc * fHW;
                for (int i = 0; i < fH; i++) {
                    float* row = out + (size_t)ci * H * W + (size_t)(i * 8 + dy) * W;
                    for (int j = 0; j < fW; j++) {
                        float v = in[i * fW + j];
                        if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f;
                        row[j * 8 + dx] = v;
                    }
                }
            }
}

/* build the two 2x checkerboard masks [YCH, yH, yW].
 * mask_0: ch<64 -> pattern((1,0),(0,1)); ch>=64 -> pattern((0,1),(1,0))
 * mask_1: the complement. */
/* BT.709 coefficients for RGB <-> YCbCr conversion (matches src/utils/transforms.py). */
static const float Kr = 0.2126f;
static const float Kg = 0.7152f;
static const float Kb = 0.0722f;

static void rgb_to_ycbcr(const float* rgb, float* ycbcr, int n)
{
    for (int i = 0; i < n; i++) {
        float r = rgb[i + 0 * n];
        float g = rgb[i + 1 * n];
        float b = rgb[i + 2 * n];
        float y = Kr * r + Kg * g + Kb * b;
        float cb = 0.5f * (b - y) / (1.0f - Kb) + 0.5f;
        float cr = 0.5f * (r - y) / (1.0f - Kr) + 0.5f;
        y = (y < 0.0f) ? 0.0f : (y > 1.0f ? 1.0f : y);
        cb = (cb < 0.0f) ? 0.0f : (cb > 1.0f ? 1.0f : cb);
        cr = (cr < 0.0f) ? 0.0f : (cr > 1.0f ? 1.0f : cr);
        ycbcr[i + 0 * n] = y;
        ycbcr[i + 1 * n] = cb;
        ycbcr[i + 2 * n] = cr;
    }
}

static void ycbcr_to_rgb(const float* ycbcr, float* rgb, int n)
{
    for (int i = 0; i < n; i++) {
        float y = ycbcr[i + 0 * n];
        float cb = ycbcr[i + 1 * n];
        float cr = ycbcr[i + 2 * n];
        float r = y + (2.0f - 2.0f * Kr) * (cr - 0.5f);
        float b = y + (2.0f - 2.0f * Kb) * (cb - 0.5f);
        float g = (y - Kr * r - Kb * b) / Kg;
        r = (r < 0.0f) ? 0.0f : (r > 1.0f ? 1.0f : r);
        g = (g < 0.0f) ? 0.0f : (g > 1.0f ? 1.0f : g);
        b = (b < 0.0f) ? 0.0f : (b > 1.0f ? 1.0f : b);
        rgb[i + 0 * n] = r;
        rgb[i + 1 * n] = g;
        rgb[i + 2 * n] = b;
    }
}

static void replicate_pad_3(const float* x, int H, int W,
                            float* dst, int Hp, int Wp)
{
    for (int c = 0; c < 3; c++) {
        const float* xc = x + (size_t)c * H * W;
        float* dc = dst + (size_t)c * Hp * Wp;
        for (int i = 0; i < Hp; i++) {
            const float* srow = xc + (size_t)(i < H ? i : H - 1) * W;
            float* drow = dc + (size_t)i * Wp;
            for (int j = 0; j < Wp; j++)
                drow[j] = srow[j < W ? j : W - 1];
        }
    }
}

/* Crop src [3,Hp,Wp] -> dst [3,H,W]. */
static void crop_3(const float* src, int Hp, int Wp,
                   float* dst, int H, int W)
{
    for (int c = 0; c < 3; c++) {
        const float* sc = src + (size_t)c * Hp * Wp;
        float* dc = dst + (size_t)c * H * W;
        for (int i = 0; i < H; i++)
            memcpy(dc + (size_t)i * W, sc + (size_t)i * Wp, (size_t)W * sizeof(float));
    }
}

static void build_masks_2x(float* m0, float* m1, int yH, int yW)
{
    int half = YCH / 2;   /* 64 */
    int hw = yH * yW;
    /* m0 micro ((1,0),(0,1)): (0,0)=1,(0,1)=0,(1,0)=0,(1,1)=1
     * m1 micro ((0,1),(1,0)): (0,0)=0,(0,1)=1,(1,0)=1,(1,1)=0  (complement) */
    for (int ch = 0; ch < YCH; ch++) {
        int second = ch >= half;
        for (int i = 0; i < yH; i++)
            for (int j = 0; j < yW; j++) {
                int sp = (i & 1) * 2 + (j & 1);   /* 0,1,2,3 */
                int p0 = (sp == 0 || sp == 3) ? 1 : 0;
                int val = second ? (1 - p0) : p0;
                m0[ch * hw + i * yW + j] = (float)val;
                m1[ch * hw + i * yW + j] = (float)(1 - val);
            }
    }
}

/* ---------- scale table (gaussian) ---------- */
static void init_scale_table(DcvcCpuInterPipeline* p)
{
    p->scale_min = 0.11f; p->scale_max = 16.0f; p->scale_level = 128;
    p->log_scale_min = logf(p->scale_min);
    float log_max = logf(p->scale_max);
    p->log_step_recip = (float)(p->scale_level - 1) / (log_max - p->log_scale_min);
    p->skip_thres = 0.0f;  /* disabled by default; enable via DCVC_SKIP_THRES env */
}

/* ---------- create / destroy ---------- */
#define LOAD_ENG(field, fname) do { \
    snprintf(path, sizeof(path), "%s/" fname, model_dir); \
    p->field = dcvc_cpu_engine_create(path, dcvc_cpu_default_use_gpu(), &st); \
    if (!p->field) goto fail; } while (0)
#define LOAD_NPY(np, fname) do { \
    snprintf(path, sizeof(path), "%s/" fname, model_dir); \
    if (dcvc_npy_read(path, &p->np) != 0) { st = DCVC_CPU_ERR_IO; goto fail; } } while (0)
#define LOAD_QBANK(name, dst, ch) do { \
    snprintf(path, sizeof(path), "%s/" name ".npy", model_dir); \
    DcvcNpy _t = {0}; if (dcvc_npy_read(path, &_t) != 0) { st = DCVC_CPU_ERR_IO; goto fail; } \
    if (qp < 0 || qp >= _t.dims[0]) { dcvc_npy_free(&_t); st = DCVC_CPU_ERR_INVALID_ARG; goto fail; } \
    p->dst = (float*)malloc(ch * sizeof(float)); if (!p->dst) { dcvc_npy_free(&_t); st = DCVC_CPU_ERR_OOM; goto fail; } \
    for (int _i = 0; _i < ch; _i++) p->dst[_i] = dcvc_npy_f32(&_t)[qp * _t.dims[1] + _i]; \
    dcvc_npy_free(&_t); } while (0)

DcvcCpuInterPipeline* dcvc_cpu_inter_pipeline_create(const char* model_dir,
                                                     int H, int W, int qp,
                                                     DcvcCpuStatus* out_st)
{
    if (out_st) *out_st = DCVC_CPU_OK;
    if (!model_dir || H <= 0 || W <= 0) {
        if (out_st) *out_st = DCVC_CPU_ERR_INVALID_ARG; return NULL;
    }
    if (H % 64 != 0 || W % 64 != 0) {
        fprintf(stderr, "error: inter pipeline requires H and W to be multiples of 64 (got %dx%d)\n", H, W);
        if (out_st) *out_st = DCVC_CPU_ERR_INVALID_ARG; return NULL;
    }
    DcvcCpuInterPipeline* p = (DcvcCpuInterPipeline*)calloc(1, sizeof(*p));
    if (!p) { if (out_st) *out_st = DCVC_CPU_ERR_OOM; return NULL; }
    p->H = H; p->W = W; p->qp = qp;
    p->Hp = (H + 63) / 64 * 64;   /* replicate-pad up to a multiple of 64 */
    p->Wp = (W + 63) / 64 * 64;
    p->fH = p->Hp / 8; p->fW = p->Wp / 8; p->yH = p->Hp / 16; p->yW = p->Wp / 16;
    p->zH = p->Hp / 64; p->zW = p->Wp / 64;
    DcvcCpuStatus st = DCVC_CPU_OK;
    char path[640];

    LOAD_ENG(eng_fai,   "inter_feature_adaptor_i.onnx");
    LOAD_ENG(eng_fe,    "inter_feature_extractor.onnx");
    LOAD_ENG(eng_enc,   "inter_encoder.onnx");
    LOAD_ENG(eng_henc,  "inter_hyper_enc.onnx");
    LOAD_ENG(eng_hdec,  "inter_hyper_dec.onnx");
    LOAD_ENG(eng_temp,  "inter_temporal_prior.onnx");
    LOAD_ENG(eng_pfus,  "inter_prior_fusion.onnx");
    LOAD_ENG(eng_sp,    "inter_spatial_prior.onnx");
    LOAD_ENG(eng_dec,   "inter_decoder.onnx");
    LOAD_ENG(eng_recon, "recon_generation.onnx");

    LOAD_NPY(gcdf, "gaussian_cdf.npy");
    LOAD_NPY(glen, "gaussian_cdf_length.npy");
    LOAD_NPY(goff, "gaussian_offset.npy");
    LOAD_NPY(zcdf, "bitest_cdf.npy");
    LOAD_NPY(zlen, "bitest_cdf_length.npy");
    LOAD_NPY(zoff, "bitest_offset.npy");

    LOAD_QBANK("q_encoder", q_enc, DCH);
    LOAD_QBANK("q_decoder", q_dec, DCH);
    LOAD_QBANK("q_feature", q_feat, DCH);
    LOAD_QBANK("q_recon",   q_recon, RCH);

    init_scale_table(p);

    p->rans_enc = dcvc_rans_encoder_create();
    p->rans_dec = dcvc_rans_decoder_create();
    if (!p->rans_enc || !p->rans_dec) { st = DCVC_CPU_ERR_OOM; goto fail; }

    int fHW = p->fH * p->fW, yHW = p->yH * p->yW, zHW = p->zH * p->zW;
    int HW = p->Hp * p->Wp;
#define M(cnt, sz) (float*)malloc((size_t)(cnt) * (sz) * sizeof(float))
#define M8(cnt, sz) (int8_t*)malloc((size_t)(cnt) * (sz))
#define MU8(cnt, sz) (uint8_t*)malloc((size_t)(cnt) * (sz))
#define M16(cnt, sz) (int16_t*)malloc((size_t)(cnt) * (sz) * sizeof(int16_t))
    p->ref_unshuf = M(SRCD, fHW); p->feature = M(DCH, fHW);
    p->ctx = M(DCH, fHW); p->ctx_t = M(DCH, fHW); p->x_unshuf = M(SRCD, fHW);
    p->y = M(YCH, yHW); p->z = M(ZCH, zHW); p->z_hat = M(ZCH, zHW);
    p->z_int8 = M8(ZCH, zHW); p->fdec = M(DCH, fHW);
    p->hier = M(YCH, yHW); p->temporal = M(DCH, yHW); p->pfus_in = M(3 * YCH, yHW);
    p->params = M(3 * YCH, yHW);
    p->scales = M(YCH, yHW); p->means = M(YCH, yHW); p->q_dec_v = M(YCH, yHW);
    p->yhat_acc = M(YCH, yHW); p->masks[0] = M(YCH, yHW); p->masks[1] = M(YCH, yHW);
    p->scl_mask = M(YCH, yHW); p->sr = M(YCH / 2, yHW);
    p->yq_full = M(YCH, yHW); p->yq_w = M(YCH / 2, yHW);
    p->cat_sp = M(4 * YCH, yHW); p->sp_out = M(2 * YCH, yHW);
    p->packed = M16(YCH / 2, yHW); p->indexes = MU8(YCH / 2, yHW);
    p->x_pad = M(3, HW); p->ref_pad = M(3, HW);
    p->x_ycbcr = M(3, HW); p->ref_ycbcr = M(3, HW);
    p->rg_out = M(SRCD, fHW); p->x_hat = M(3, HW);
    float* all[] = { p->x_pad, p->ref_pad, p->x_ycbcr, p->ref_ycbcr, p->ref_unshuf, p->feature, p->ctx, p->ctx_t, p->x_unshuf,
        p->y, p->z, p->z_hat, p->fdec, p->hier, p->temporal, p->pfus_in, p->params,
        p->scales, p->means, p->q_dec_v, p->yhat_acc, p->masks[0], p->masks[1],
        p->scl_mask, p->sr, p->yq_full, p->yq_w, p->cat_sp, p->sp_out,
        (float*)p->z_int8, p->rg_out, p->x_hat, (float*)p->packed, (float*)p->indexes };
    for (size_t i = 0; i < sizeof(all)/sizeof(all[0]); i++)
        if (!all[i]) { st = DCVC_CPU_ERR_OOM; goto fail; }
    build_masks_2x(p->masks[0], p->masks[1], p->yH, p->yW);

    /* Optional skip_thres from environment (adaptive entropy coding). */
    {
        const char* st_env = getenv("DCVC_SKIP_THRES");
        if (st_env) { p->skip_thres = strtof(st_env, NULL); }
    }

    if (out_st) *out_st = DCVC_CPU_OK;
    return p;
fail:
    if (out_st) *out_st = st;
    dcvc_cpu_inter_pipeline_destroy(p);
    return NULL;
}
#undef LOAD_ENG
#undef LOAD_NPY
#undef LOAD_QBANK
#undef M
#undef M8
#undef MU8
#undef M16

void dcvc_cpu_inter_pipeline_destroy(DcvcCpuInterPipeline* p)
{
    if (!p) return;
#define E(x) dcvc_cpu_engine_destroy(p->x)
    E(eng_fai); E(eng_fe); E(eng_enc); E(eng_henc); E(eng_hdec);
    E(eng_temp); E(eng_pfus); E(eng_sp); E(eng_dec); E(eng_recon);
#undef E
    dcvc_rans_encoder_destroy(p->rans_enc); dcvc_rans_decoder_destroy(p->rans_dec);
    dcvc_npy_free(&p->gcdf); dcvc_npy_free(&p->glen); dcvc_npy_free(&p->goff);
    dcvc_npy_free(&p->zcdf); dcvc_npy_free(&p->zlen); dcvc_npy_free(&p->zoff);
    free(p->q_enc); free(p->q_dec); free(p->q_feat); free(p->q_recon);
    free(p->ref_unshuf); free(p->feature); free(p->ctx); free(p->ctx_t); free(p->x_unshuf);
    free(p->y); free(p->z); free(p->z_hat); free(p->z_int8); free(p->fdec);
    free(p->hier); free(p->temporal); free(p->pfus_in); free(p->params);
    free(p->scales); free(p->means); free(p->q_dec_v); free(p->yhat_acc);
    free(p->masks[0]); free(p->masks[1]); free(p->scl_mask); free(p->sr);
    free(p->yq_full); free(p->yq_w); free(p->cat_sp); free(p->sp_out);
    free(p->packed); free(p->indexes); free(p->rg_out); free(p->x_hat);
    free(p->x_pad); free(p->ref_pad); free(p->x_ycbcr); free(p->ref_ycbcr);
    free(p);
}

/* res_prior_param_decoder(z_hat, ctx_t):
 *   hier = hyper_decoder(z_hat)            [128, yH, yW]
 *   temporal = temporal_prior(ctx_t)       [256, yH, yW]  (fH->yH stride2)
 *   params = prior_fusion(cat(hier, temporal))  [384, yH, yW] */
static DcvcCpuStatus prior_params(DcvcCpuInterPipeline* p)
{
    DcvcCpuStatus st;
    st = run1(p->eng_hdec, p->z_hat, ZCH, p->zH, p->zW, p->hier, YCH, p->yH, p->yW);
    if (st != DCVC_CPU_OK) return st;
    st = run1(p->eng_temp, p->ctx_t, DCH, p->fH, p->fW, p->temporal, DCH, p->yH, p->yW);
    if (st != DCVC_CPU_OK) return st;
    /* pfus_in = cat(hier[128], temporal[256]) = 384 */
    int yHW = p->yH * p->yW;
    memcpy(p->pfus_in, p->hier, (size_t)YCH * yHW * sizeof(float));
    memcpy(p->pfus_in + (size_t)YCH * yHW, p->temporal, (size_t)DCH * yHW * sizeof(float));
    return run1(p->eng_pfus, p->pfus_in, 3 * YCH, p->yH, p->yW, p->params, 3 * YCH, p->yH, p->yW);
}

/* decode-side 2x prior: common_params -> y_hat (rANS symbols already decoded
 * order: pass0 then pass1). Returns y_hat in original (unscaled) domain. */
static DcvcCpuStatus prior_2x_decode(DcvcCpuInterPipeline* p)
{
    int yHW = p->yH * p->yW;
    int half = YCH / 2;            /* 64 channels after combine */
    int hw64 = half * yHW;
    /* separate_prior_for_video_decoding: q_dec=clamp_min(params[0:128],0.5) */
    for (int i = 0; i < YCH * yHW; i++) {
        float qd = p->params[i]; if (qd < 0.5f) qd = 0.5f; p->q_dec_v[i] = qd;
    }
    memcpy(p->scales, p->params + (size_t)YCH * yHW, (size_t)YCH * yHW * sizeof(float));
    memcpy(p->means,  p->params + (size_t)2 * YCH * yHW, (size_t)YCH * yHW * sizeof(float));

    /* ---- pass 0 ---- */
    float* mask0 = p->masks[0];
    for (int i = 0; i < YCH * yHW; i++) p->scl_mask[i] = p->scales[i] * mask0[i];
    sp2x(p->scl_mask, p->sr, hw64);
    build_index_dec(p->sr, p->indexes, p->scale_min, p->scale_max,
                    p->log_scale_min, p->log_step_recip, hw64);
    dcvc_rans_decoder_decode_y(p->rans_dec, p->indexes, hw64, p->g_cdf_idx);
    int8_t* syms = NULL; size_t symn = 0;
    dcvc_rans_decoder_get_symbols(p->rans_dec, &syms, &symn);
    if (dcvc_rans_decoder_has_error(p->rans_dec))
        return DCVC_CPU_ERR_ENTROPY;
    /* restore_y_2x_with_cat_after: y_hat_0=(cat(yq,yq)+means)*mask0 ; cat_params=cat(y_hat_0,params) */
    for (int i = 0; i < YCH * yHW; i++) {
        float yq = (float)syms[i % hw64];
        p->yhat_acc[i] = (yq + p->means[i]) * mask0[i];
    }
    /* cat_sp = cat(yhat_acc[128], params[384]) */
    memcpy(p->cat_sp, p->yhat_acc, (size_t)YCH * yHW * sizeof(float));
    memcpy(p->cat_sp + (size_t)YCH * yHW, p->params, (size_t)3 * YCH * yHW * sizeof(float));
    DcvcCpuStatus st = run1(p->eng_sp, p->cat_sp, 4 * YCH, p->yH, p->yW,
                            p->sp_out, 2 * YCH, p->yH, p->yW);
    if (st != DCVC_CPU_OK) return st;
    /* sp_out -> scales1[128], means1[128] */
    float* scales1 = p->sp_out;
    float* means1 = p->sp_out + (size_t)YCH * yHW;

    /* ---- pass 1 ---- */
    float* mask1 = p->masks[1];
    for (int i = 0; i < YCH * yHW; i++) p->scl_mask[i] = scales1[i] * mask1[i];
    sp2x(p->scl_mask, p->sr, hw64);
    build_index_dec(p->sr, p->indexes, p->scale_min, p->scale_max,
                    p->log_scale_min, p->log_step_recip, hw64);
    dcvc_rans_decoder_decode_y(p->rans_dec, p->indexes, hw64, p->g_cdf_idx);
    dcvc_rans_decoder_get_symbols(p->rans_dec, &syms, &symn);
    if (dcvc_rans_decoder_has_error(p->rans_dec))
        return DCVC_CPU_ERR_ENTROPY;
    for (int i = 0; i < YCH * yHW; i++) {
        float yq = (float)syms[i % hw64];
        p->yhat_acc[i] += (yq + means1[i]) * mask1[i];
    }
    /* y_hat = yhat_acc * q_dec */
    for (int i = 0; i < YCH * yHW; i++) p->y[i] = p->yhat_acc[i] * p->q_dec_v[i];
    inter_dump("DCVC_DUMP_YHAT", p->y, YCH, p->yH, p->yW);
    return DCVC_CPU_OK;
}

/* encode-side 2x prior: y_scaled already in p->y; produces rANS symbols and
 * writes final y_hat into p->y. */
static DcvcCpuStatus prior_2x_encode(DcvcCpuInterPipeline* p)
{
    int yHW = p->yH * p->yW;
    int half = YCH / 2;
    int hw64 = half * yHW;
    /* separate_prior_for_video_encoding: q_dec=clamp_min(params[0:128],0.5);
     * y_scaled = y / q_dec ; scales=params[128:256]; means=params[256:384] */
    for (int i = 0; i < YCH * yHW; i++) {
        float qd = p->params[i]; if (qd < 0.5f) qd = 0.5f; p->q_dec_v[i] = qd;
        p->y[i] = p->y[i] / qd;
    }
    memcpy(p->scales, p->params + (size_t)YCH * yHW, (size_t)YCH * yHW * sizeof(float));
    memcpy(p->means,  p->params + (size_t)2 * YCH * yHW, (size_t)YCH * yHW * sizeof(float));

    /* pass 0: process_with_mask(y_scaled, scales, means, mask0) */
    float* mask0 = p->masks[0];
    for (int i = 0; i < YCH * yHW; i++) {
        float fm = mask0[i];
        float mh = p->means[i] * fm;
        float q = roundf((p->y[i] - mh) * fm);
        if (q > 127.0f) q = 127.0f; if (q < -128.0f) q = -128.0f;
        float s_hat = p->scales[i] * fm;     /* s_hat_0 */
        if (s_hat <= p->skip_thres) q = 0.0f;   /* force-zero: skip rANS */
        p->yq_full[i] = q;
        p->yhat_acc[i] = q + mh;                /* y_hat_0 */
        p->scl_mask[i] = s_hat;
    }
    sp2x(p->yq_full, p->yq_w, hw64);
    sp2x(p->scl_mask, p->sr, hw64);
    build_index_enc(p->yq_w, p->sr, p->packed, p->scale_min, p->scale_max,
                    p->log_scale_min, p->log_step_recip, hw64);
    dcvc_rans_encoder_encode_y(p->rans_enc, p->packed, hw64, p->g_cdf_idx);

    /* spatial prior: cat(y_hat_0[128], params[384]) -> scales1, means1 */
    memcpy(p->cat_sp, p->yhat_acc, (size_t)YCH * yHW * sizeof(float));
    memcpy(p->cat_sp + (size_t)YCH * yHW, p->params, (size_t)3 * YCH * yHW * sizeof(float));
    DcvcCpuStatus st = run1(p->eng_sp, p->cat_sp, 4 * YCH, p->yH, p->yW,
                            p->sp_out, 2 * YCH, p->yH, p->yW);
    if (st != DCVC_CPU_OK) return st;
    float* scales1 = p->sp_out;
    float* means1 = p->sp_out + (size_t)YCH * yHW;

    /* pass 1 */
    float* mask1 = p->masks[1];
    for (int i = 0; i < YCH * yHW; i++) {
        float fm = mask1[i];
        float mh = means1[i] * fm;
        float q = roundf((p->y[i] - mh) * fm);
        if (q > 127.0f) q = 127.0f; if (q < -128.0f) q = -128.0f;
        float s_hat = scales1[i] * fm;
        if (s_hat <= p->skip_thres) q = 0.0f;   /* force-zero: skip rANS */
        p->yq_full[i] = q;
        p->yhat_acc[i] += q + mh;               /* accumulate y_hat_1 */
        p->scl_mask[i] = s_hat;
    }
    sp2x(p->yq_full, p->yq_w, hw64);
    sp2x(p->scl_mask, p->sr, hw64);
    build_index_enc(p->yq_w, p->sr, p->packed, p->scale_min, p->scale_max,
                    p->log_scale_min, p->log_step_recip, hw64);
    dcvc_rans_encoder_encode_y(p->rans_enc, p->packed, hw64, p->g_cdf_idx);

    /* y_hat = (y_hat_0 + y_hat_1) * q_dec  == yhat_acc * q_dec */
    for (int i = 0; i < YCH * yHW; i++) p->y[i] = p->yhat_acc[i] * p->q_dec_v[i];
    inter_dump("DCVC_DUMP_YHAT", p->y, YCH, p->yH, p->yW);
    return DCVC_CPU_OK;
}

static DcvcCpuStatus reconstruct(DcvcCpuInterPipeline* p, float* x_hat_out)
{
    /* feature = decoder(y_hat, ctx, q_dec) ; x_hat = recon(feature, q_recon) */
    DcvcCpuStatus st = run3(p->eng_dec, p->y, YCH, p->yH, p->yW,
                            p->ctx, DCH, p->fH, p->fW,
                            p->q_dec, DCH, 1, 1,
                            p->fdec, DCH, p->fH, p->fW);
    if (st != DCVC_CPU_OK) return st;
    st = run2(p->eng_recon, p->fdec, DCH, p->fH, p->fW,
              p->q_recon, RCH, 1, 1, p->rg_out, SRCD, p->fH, p->fW);
    if (st != DCVC_CPU_OK) return st;
    pixel_shuffle_8(p->rg_out, p->x_hat, p->Hp, p->Wp, p->fH, p->fW);
    inter_dump("DCVC_DUMP_FEAT", p->fdec, DCH, p->fH, p->fW);
    inter_dump("DCVC_DUMP_XHAT", p->x_hat, 3, p->Hp, p->Wp);
    if (x_hat_out) crop_3(p->x_hat, p->Hp, p->Wp, x_hat_out, p->H, p->W);
    return DCVC_CPU_OK;
}

/* ---------- encode ---------- */
DcvcCpuStatus dcvc_cpu_inter_pipeline_encode(DcvcCpuInterPipeline* p,
                                             const float* x,
                                             const float* x_hat_ref,
                                             uint8_t** out_stream, size_t* out_size,
                                             float* x_hat_out)
{
    if (!p || !x || !x_hat_ref || !out_stream || !out_size) return DCVC_CPU_ERR_INVALID_ARG;
    *out_stream = NULL; *out_size = 0;
    DcvcCpuStatus st;
    int zHW = p->zH * p->zW;

    /* Replicate-pad both RGB inputs and convert to YCbCr before feeding NN. */
    replicate_pad_3(x_hat_ref, p->H, p->W, p->ref_pad, p->Hp, p->Wp);
    rgb_to_ycbcr(p->ref_pad, p->ref_ycbcr, p->Hp * p->Wp);
    pixel_unshuffle_8(p->ref_ycbcr, p->ref_unshuf, p->Hp, p->Wp, p->fH, p->fW);
    st = run1(p->eng_fai, p->ref_unshuf, SRCD, p->fH, p->fW, p->feature, DCH, p->fH, p->fW);
    if (st != DCVC_CPU_OK) return st;
    /* ctx, ctx_t = feature_extractor(feature, q_feature) */
    st = run_fe(p->eng_fe, p->feature, p->q_feat, p->ctx, p->ctx_t, p->fH, p->fW);
    if (st != DCVC_CPU_OK) return st;
    /* y = encoder(pixel_unshuffle(x), ctx, q_encoder) */
    replicate_pad_3(x, p->H, p->W, p->x_pad, p->Hp, p->Wp);
    rgb_to_ycbcr(p->x_pad, p->x_ycbcr, p->Hp * p->Wp);
    pixel_unshuffle_8(p->x_ycbcr, p->x_unshuf, p->Hp, p->Wp, p->fH, p->fW);
    st = run3(p->eng_enc, p->x_unshuf, SRCD, p->fH, p->fW,
              p->ctx, DCH, p->fH, p->fW, p->q_enc, DCH, 1, 1,
              p->y, YCH, p->yH, p->yW);
    if (st != DCVC_CPU_OK) return st;
    /* z = hyper_encoder(y)  (yH,yW mult of 4 at 256 => no pad) */
    st = run1(p->eng_henc, p->y, YCH, p->yH, p->yW, p->z, ZCH, p->zH, p->zW);
    if (st != DCVC_CPU_OK) return st;
    round_to_int8(p->z, p->z_hat, p->z_int8, ZCH * zHW);

    /* params = prior_params(z_hat, ctx_t) */
    st = prior_params(p);
    if (st != DCVC_CPU_OK) return st;

    /* entropy: z then y(2 passes) */
    dcvc_rans_encoder_reset(p->rans_enc);
    p->z_cdf_idx = dcvc_rans_encoder_add_cdf(p->rans_enc, dcvc_npy_i32(&p->zcdf),
                        p->zcdf.dims[0], p->zcdf.dims[1],
                        dcvc_npy_i32(&p->zlen), dcvc_npy_i32(&p->zoff));
    dcvc_rans_encoder_encode_z(p->rans_enc, p->z_int8, ZCH * zHW,
                               p->z_cdf_idx, p->qp * ZCH, zHW);
    p->g_cdf_idx = dcvc_rans_encoder_add_cdf(p->rans_enc, dcvc_npy_i32(&p->gcdf),
                        p->gcdf.dims[0], p->gcdf.dims[1],
                        dcvc_npy_i32(&p->glen), dcvc_npy_i32(&p->goff));
    st = prior_2x_encode(p);
    if (st != DCVC_CPU_OK) return st;
    dcvc_rans_encoder_flush(p->rans_enc);
    uint8_t* stream = NULL; size_t ssize = 0;
    if (dcvc_rans_encoder_get_stream(p->rans_enc, &stream, &ssize) != 0)
        return DCVC_CPU_ERR_ONNX;

    /* Reconstruct (closed-loop), then convert NN output YCbCr back to RGB. */
    st = reconstruct(p, x_hat_out ? p->x_hat : NULL);
    if (st != DCVC_CPU_OK) { free(stream); return st; }
    if (x_hat_out) {
        ycbcr_to_rgb(p->x_hat, p->x_pad, p->Hp * p->Wp);  /* reuse x_pad as RGB buffer */
        crop_3(p->x_pad, p->Hp, p->Wp, x_hat_out, p->H, p->W);
    }

    *out_stream = stream; *out_size = ssize;
    return DCVC_CPU_OK;
}

/* ---------- decode ---------- */
DcvcCpuStatus dcvc_cpu_inter_pipeline_decode(DcvcCpuInterPipeline* p,
                                             const uint8_t* stream, size_t stream_size,
                                             const float* x_hat_ref,
                                             float* x_hat_out)
{
    if (!p || !stream || !x_hat_ref || !x_hat_out) return DCVC_CPU_ERR_INVALID_ARG;
    DcvcCpuStatus st;
    int zHW = p->zH * p->zW;

    /* feature = feature_adaptor_i(pixel_unshuffle(x_hat_ref)) ; ctx, ctx_t */
    replicate_pad_3(x_hat_ref, p->H, p->W, p->ref_pad, p->Hp, p->Wp);
    rgb_to_ycbcr(p->ref_pad, p->ref_ycbcr, p->Hp * p->Wp);
    pixel_unshuffle_8(p->ref_ycbcr, p->ref_unshuf, p->Hp, p->Wp, p->fH, p->fW);
    st = run1(p->eng_fai, p->ref_unshuf, SRCD, p->fH, p->fW, p->feature, DCH, p->fH, p->fW);
    if (st != DCVC_CPU_OK) return st;
    st = run_fe(p->eng_fe, p->feature, p->q_feat, p->ctx, p->ctx_t, p->fH, p->fW);
    if (st != DCVC_CPU_OK) return st;

    /* decode z */
    dcvc_rans_decoder_reset_cdf(p->rans_dec);
    p->z_cdf_idx = dcvc_rans_decoder_add_cdf(p->rans_dec, dcvc_npy_i32(&p->zcdf),
                        p->zcdf.dims[0], p->zcdf.dims[1],
                        dcvc_npy_i32(&p->zlen), dcvc_npy_i32(&p->zoff));
    dcvc_rans_decoder_set_stream(p->rans_dec, stream, stream_size);
    dcvc_rans_decoder_decode_z(p->rans_dec, ZCH * zHW, p->z_cdf_idx, p->qp * ZCH, zHW);
    int8_t* zsyms = NULL; size_t zn = 0;
    dcvc_rans_decoder_get_symbols(p->rans_dec, &zsyms, &zn);
    if (dcvc_rans_decoder_has_error(p->rans_dec))
        return DCVC_CPU_ERR_ENTROPY;
    int8_to_float(zsyms, p->z_hat, ZCH * zHW);

    /* params = prior_params(z_hat, ctx_t) */
    st = prior_params(p);
    if (st != DCVC_CPU_OK) return st;

    /* decode y (2 passes) */
    p->g_cdf_idx = dcvc_rans_decoder_add_cdf(p->rans_dec, dcvc_npy_i32(&p->gcdf),
                        p->gcdf.dims[0], p->gcdf.dims[1],
                        dcvc_npy_i32(&p->glen), dcvc_npy_i32(&p->goff));
    st = prior_2x_decode(p);
    if (st != DCVC_CPU_OK) return st;
    /* z and y share one stream; a synchronized decode consumes it exactly. */
    if (dcvc_rans_decoder_bytes_consumed(p->rans_dec) != stream_size)
        return DCVC_CPU_ERR_ENTROPY;

    /* Reconstruct and convert YCbCr output back to RGB before cropping. */
    st = reconstruct(p, p->x_hat);
    if (st != DCVC_CPU_OK) return st;
    ycbcr_to_rgb(p->x_hat, p->x_pad, p->Hp * p->Wp);  /* reuse x_pad as RGB buffer */
    crop_3(p->x_pad, p->Hp, p->Wp, x_hat_out, p->H, p->W);
    return DCVC_CPU_OK;
}
