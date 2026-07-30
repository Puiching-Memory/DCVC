/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Pure-CPU inter-frame (P-chunk) pipeline implementation for DCVC-UF (HT-S /
 * HT-L). A P-chunk encodes g_frame_delay (8) frames into a single latent y and
 * reconstructs them in parallel via a frame-wise recon head.
 *
 * Encode/decode use identical FP32 ONNX models, so the closed loop (encoder
 * reconstruction == decoder reconstruction, and the DPB feature memory) stays
 * bit-consistent. The feature-memory DPB is owned by the pipeline.
 */
#include "cpu_inter_pipeline.h"
#include "cpu_ar_codec.h"
#include "npy_reader.h"
#include "onnx_engine.h"
#include "rans_c.h"
#include "fxp/fxp_scale_index.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* HT channel constants (HT-S / HT-L share these). */
#define FRAMES DCVC_FRAME_DELAY  /* 8 frames per chunk */
#define SRCD_RAW   (3 * FRAMES)  /* raw chunk channels before pixel_unshuffle */
#define SRCD_INTRA (3 * 8 * 8)   /* 192: single frame pixel_unshuffled */
#define YCH 256                  /* g_ch_y */
#define ZCH 128                  /* g_ch_z */
#define DCH 512                  /* g_ch_d */
#define MCH 512                  /* g_ch_m */

struct DcvcCpuInterPipeline {
    int H, W;            /* actual (cropped) per-frame size */
    int Hp, Wp;          /* padded size, multiple of 64 */
    int fH, fW, yH, yW, zH, zW, qp;
    int is_hts;

    DcvcCpuEngine* eng_fai;     /* feature_adaptor_i  (192 -> 512) */
    DcvcCpuEngine* eng_fam;     /* feature_adaptor_m  (512+512 -> 512) */
    DcvcCpuEngine* eng_fe;      /* feature_extractor (512 -> 512) */
    DcvcCpuEngine* eng_enc;     /* inter_encoder (raw chunk -> y) */
    DcvcCpuEngine* eng_henc;    /* hyper_encoder (y -> z) */
    DcvcCpuEngine* eng_hdec;    /* hyper_decoder (z -> y) */
    DcvcCpuEngine* eng_temp;    /* temporal_prior (memory,q -> 512@yH) */
    DcvcCpuEngine* eng_pfus;    /* prior_fusion (cat(256,512)=768 -> 768) */
    DcvcCpuEngine* eng_red;     /* spatial_prior_reduction (768 -> 256) */
    DcvcCpuEngine* eng_adp[4];  /* spatial_prior_adaptor_{1,2,3} (idx 1..3) */
    DcvcCpuEngine* eng_sp;      /* spatial_prior (512 -> 256 hts / 512 htl) */
    DcvcCpuEngine* eng_dec;     /* inter_decoder (y,ctx,q -> 512 feature) */
    DcvcCpuEngine* eng_recon;   /* recon_head (512 -> 8 RGB frames) */

    DcvcRansEncoder* rans_enc;
    DcvcRansDecoder* rans_dec;
    DcvcNpy gcdf, glen, goff;   /* gaussian CDF (model-independent) */
    DcvcNpy zcdf, zlen, zoff;   /* bit-estimator CDF (per-checkpoint) */
    int g_cdf_idx, z_cdf_idx;

    float scale_min, scale_max, log_scale_min, log_step_recip;
    int scale_level;
    float skip_thres;

    float *q_enc, *q_dec, *q_feat;   /* [DCH] q-banks for current qp */

    /* feature-memory DPB */
    int has_memory;
    float *memory;            /* [MCH, fH, fW] */
    float *ref_feature;       /* [DCH, fH, fW] previous decoder feature */

    /* workspace */
    float *ref_unshuf;        /* [SRCD_INTRA, fH, fW] (intra handoff) */
    float *ctx;               /* [DCH, fH, fW] */
    float *x_chunk;           /* [SRCD_RAW, Hp, Wp] raw chunk (YCbCr) */
    float *y, *z, *z_hat;     /* [YCH,yH,yW], [ZCH,zH,zW] */
    int8_t* z_int8;
    float *temporal, *hier, *pfus_in, *params;  /* [512@yH],[256@yH],[768@yH],[768@yH] */
    float *common;            /* [YCH, yH, yW] reduced prior */
    float *scales, *means;    /* [YCH,yH,yW] */
    float *q_enc_v, *q_dec_v; /* [YCH,yH,yW] */
    float *y_scaled, *yhat;   /* [YCH,yH,yW] */
    float *yq, *yq_w, *scl_mask, *sr, *adp_in, *sp_out;
    float *sp_sm;            /* spatial-prior output [2*YCH] for HT-L (scales+means) */
    int16_t* packed;
    uint8_t* indexes;
    float *masks[4];          /* [YCH,yH,yW] each */
    float *feature_dec;       /* [DCH, fH, fW] current decoder feature */
    float *recon_out[FRAMES]; /* [3, Hp, Wp] each frame (YCbCr) */
    float *frame_pad, *frame_rgb; /* scratch [3,Hp,Wp] */
};

/* ---------- engine helpers ---------- */
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
/* recon_head: 1 input -> FRAMES outputs (each [3, Hp, Wp]). */
static DcvcCpuStatus run_recon(DcvcCpuInterPipeline* p, const float* in)
{
    DcvcCpuTensorView vin = { (void*)in, 0, 1, DCH, p->fH, p->fW };
    DcvcCpuTensorView vout[FRAMES];
    for (int i = 0; i < FRAMES; i++)
        vout[i] = (DcvcCpuTensorView){ p->recon_out[i], 0, 1, 3, p->Hp, p->Wp };
    return dcvc_cpu_engine_run(p->eng_recon, &vin, 1, vout, FRAMES);
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
/* sp4x: sum four channel quarters (n = quarter-count). out[i]=x[i]+x[i+n]+x[i+2n]+x[i+3n] */
static void sp4x(const float* x, float* out, int n)
{
    for (int i = 0; i < n; i++) out[i] = x[i] + x[i + n] + x[i + 2*n] + x[i + 3*n];
}
static void build_index_dec(const float* s, uint8_t* out, int n)
{
    dcvc_build_index_dec_i(s, out, n);
}
static void build_index_enc(const float* sym, const float* s, int16_t* out, int n)
{
    dcvc_build_index_enc_i(sym, s, out, n);
}

/* separate_prior_video(params[3*YCH]): q_enc=1/clamp(params[0:YCH],0.5),
 * q_dec=params[0:YCH], scales=params[YCH:2YCH], means=params[2YCH:3YCH]. */
static void separate_prior_video(const float* params, float* q_enc, float* q_dec,
                                 float* scales, float* means, int n)
{
    for (int i = 0; i < n; i++) {
        float qs = params[i];
        if (qs < 0.5f) qs = 0.5f;
        q_enc[i] = 1.0f / qs;
        q_dec[i] = qs;
    }
    memcpy(scales, params + n, n * sizeof(float));
    memcpy(means, params + 2 * n, n * sizeof(float));
}
/* element-wise multiply (UF video prior: q_enc_v/q_dec_v are per-channel,
 * per-spatial, same layout as the y latent — NOT a spatial broadcast). */
static void elem_mul(const float* in, const float* q, float* out, int n)
{
    for (int i = 0; i < n; i++) out[i] = in[i] * q[i];
}
static void process_mask_yq_enc(const float* y, const float* scales, const float* means,
                                const float* mask, float* yq, int n, float fz_thres)
{
    for (int i = 0; i < n; i++) {
        float fm = mask[i];
        float mh = means[i] * fm;
        float q = roundf((y[i] - mh) * fm);
        if (fz_thres > 0.0f && scales[i] * fm <= fz_thres) q = 0.0f;
        if (q > 127.0f) q = 127.0f;
        if (q < -128.0f) q = -128.0f;
        yq[i] = q;
    }
}
static void restore_y_4x(const float* yq_r, const float* means, const float* mask,
                         float* out, int cyhw, int n)
{
    for (int i = 0; i < n; i++)
        out[i] = (yq_r[i % cyhw] + means[i]) * mask[i];
}

/* pixel_unshuffle(x[3,H,W], out[192,fH,fW], factor 8) */
/* Optional debug dump: DCVC_DUMP_XHAT=<path> writes the reconstructed chunk
 * [FRAMES,3,H,W] after crop. Used for PyTorch-vs-C++ parity checks. */
static void dcvc_debug_dump_xhat(const char* env_var, const float* data,
                                 int n, int h, int w)
{
    const char* path = getenv(env_var);
    if (!path || !path[0]) return;
    int dims[4] = {n, 3, h, w};
    if (dcvc_npy_write_f32(path, data, dims, 4) == 0)
        fprintf(stderr, "[debug] dumped %s [%dx3x%dx%d] to %s\n", env_var, n, h, w, path);
}

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

static const int k_mask_pattern[4][4] = {
    {0, 1, 2, 3}, {3, 2, 1, 0}, {2, 3, 0, 1}, {1, 0, 3, 2},
};
static void build_masks_4x(DcvcCpuInterPipeline* p)
{
    int nc = YCH, hw = p->yH * p->yW, q = nc / 4;
    for (int m = 0; m < 4; m++)
        for (int ch = 0; ch < nc; ch++) {
            int quarter = ch / q; if (quarter > 3) quarter = 3;
            int target = k_mask_pattern[m][quarter];
            for (int hh = 0; hh < p->yH; hh++)
                for (int ww = 0; ww < p->yW; ww++) {
                    int sp = (hh % 2) * 2 + (ww % 2);
                    p->masks[m][ch * hw + hh * p->yW + ww] = (sp == target) ? 1.0f : 0.0f;
                }
        }
}

/* BT.709 RGB <-> YCbCr (matches src/utils/transforms.py). */
static const float Kr = 0.2126f, Kg = 0.7152f, Kb = 0.0722f;
static void rgb_to_ycbcr(const float* rgb, float* ycbcr, int n)
{
    for (int i = 0; i < n; i++) {
        float r = rgb[0*n+i], g = rgb[1*n+i], b = rgb[2*n+i];
        ycbcr[0*n+i] = Kr*r + Kg*g + Kb*b;
        ycbcr[1*n+i] = (b - ycbcr[0*n+i]) / (1.0f - Kb) * 0.5f + 0.5f;
        ycbcr[2*n+i] = (r - ycbcr[0*n+i]) / (1.0f - Kr) * 0.5f + 0.5f;
    }
}
static void ycbcr_to_rgb(const float* ycbcr, float* rgb, int n)
{
    for (int i = 0; i < n; i++) {
        float y = ycbcr[0*n+i], cb = ycbcr[1*n+i], cr = ycbcr[2*n+i];
        cb = (cb - 0.5f) * (1.0f - Kb) * 2.0f;
        cr = (cr - 0.5f) * (1.0f - Kr) * 2.0f;
        float r = y + cr, g = y - Kr/Kg*cr - Kb/Kg*cb, b = y + cb;
        /* clamp to [0,1], matching src/utils/transforms.py ycbcr2rgb (clamp=True,
         * the default used by the official test_video.py inference path). */
        r = (r < 0.0f) ? 0.0f : (r > 1.0f ? 1.0f : r);
        g = (g < 0.0f) ? 0.0f : (g > 1.0f ? 1.0f : g);
        b = (b < 0.0f) ? 0.0f : (b > 1.0f ? 1.0f : b);
        rgb[0*n+i] = r; rgb[1*n+i] = g; rgb[2*n+i] = b;
    }
}
static void replicate_pad_3(const float* x, int H, int W, float* dst, int Hp, int Wp)
{
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < Hp; i++) {
            int si = i < H ? i : H - 1;
            for (int j = 0; j < Wp; j++) {
                int sj = j < W ? j : W - 1;
                dst[c*Hp*Wp + i*Wp + j] = x[c*H*W + si*W + sj];
            }
        }
    }
}
static void crop_3(const float* src, int Hp, int Wp, float* dst, int H, int W)
{
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < H; i++)
            memcpy(dst + c*H*W + i*W, src + c*Hp*Wp + i*Wp, W * sizeof(float));
}
/* Build the raw chunk input [SRCD_RAW, Hp, Wp] from FRAMES RGB frames. */
static void build_chunk_input(DcvcCpuInterPipeline* p, const float* x_chunk)
{
    int per = 3 * p->H * p->W;
    for (int f = 0; f < FRAMES; f++) {
        replicate_pad_3(x_chunk + f * per, p->H, p->W, p->frame_pad, p->Hp, p->Wp);
        rgb_to_ycbcr(p->frame_pad, p->frame_rgb, p->Hp * p->Wp);
        memcpy(p->x_chunk + f * 3 * p->Hp * p->Wp, p->frame_rgb,
               3 * p->Hp * p->Wp * sizeof(float));
    }
}

/* ---------- create / destroy ---------- */
#define LOAD_ENG(field, fname) do { \
    snprintf(path, sizeof(path), "%s/" fname ".onnx", model_dir); \
    p->field = dcvc_cpu_engine_create(path, use_gpu, &st); \
    if (!p->field) goto fail; \
} while (0)
#define LOAD_NPY(np, fname) do { \
    snprintf(path, sizeof(path), "%s/" fname ".npy", model_dir); \
    if (dcvc_npy_read(path, &p->np) != 0) { st = DCVC_CPU_ERR_IO; goto fail; } \
} while (0)
#define LOAD_QBANK(name, dst) do { \
    snprintf(path, sizeof(path), "%s/" name ".npy", model_dir); \
    DcvcNpy qb; if (dcvc_npy_read(path, &qb) != 0) { st = DCVC_CPU_ERR_IO; goto fail; } \
    for (int i = 0; i < DCH; i++) p->dst[i] = dcvc_npy_f32(&qb)[qp * DCH + i]; \
} while (0)
#define M(cnt) ((float*)calloc(cnt, sizeof(float)))

DcvcCpuInterPipeline* dcvc_cpu_inter_pipeline_create(const char* model_dir,
                                                     int H, int W, int qp,
                                                     int is_hts,
                                                     DcvcCpuStatus* out_st)
{
    if (out_st) *out_st = DCVC_CPU_OK;
    if (!model_dir || H <= 0 || W <= 0) { if (out_st) *out_st = DCVC_CPU_ERR_INVALID_ARG; return NULL; }
    DcvcCpuStatus st = DCVC_CPU_OK;
    int use_gpu = 0;
    const char* gp = getenv("DCVC_USE_GPU"); if (gp && gp[0] == '1') use_gpu = 1;
    char path[1024];

    DcvcCpuInterPipeline* p = (DcvcCpuInterPipeline*)calloc(1, sizeof(*p));
    if (!p) { if (out_st) *out_st = DCVC_CPU_ERR_OOM; return NULL; }
    p->H = H; p->W = W; p->qp = qp; p->is_hts = is_hts;
    p->Hp = (H + 63) / 64 * 64; p->Wp = (W + 63) / 64 * 64;
    p->fH = p->Hp / 8; p->fW = p->Wp / 8;
    p->yH = p->Hp / 16; p->yW = p->Wp / 16;
    p->zH = p->Hp / 64; p->zW = p->Wp / 64;

    p->scale_level = 128; p->scale_min = 0.11f; p->scale_max = 16.0f;
    p->log_scale_min = logf(p->scale_min); p->log_step_recip = 1.0f / logf(2.0f);
    const char* sk = getenv("DCVC_SKIP_THRES"); if (sk) p->skip_thres = (float)atof(sk);

    LOAD_ENG(eng_fai,  "inter_feature_adaptor_i");
    LOAD_ENG(eng_fam,  "inter_feature_adaptor_m");
    LOAD_ENG(eng_fe,   "inter_feature_extractor");
    LOAD_ENG(eng_enc,  "inter_encoder");
    LOAD_ENG(eng_henc, "inter_hyper_enc");
    LOAD_ENG(eng_hdec, "inter_hyper_dec");
    LOAD_ENG(eng_temp, "inter_temporal_prior");
    LOAD_ENG(eng_pfus, "inter_prior_fusion");
    LOAD_ENG(eng_red,  "inter_spatial_prior_reduction");
    LOAD_ENG(eng_adp[1], "inter_spatial_prior_adaptor_1");
    LOAD_ENG(eng_adp[2], "inter_spatial_prior_adaptor_2");
    LOAD_ENG(eng_adp[3], "inter_spatial_prior_adaptor_3");
    LOAD_ENG(eng_sp,   "inter_spatial_prior");
    LOAD_ENG(eng_dec,  "inter_decoder");
    LOAD_ENG(eng_recon,"inter_recon_head");

    LOAD_NPY(gcdf, "gaussian_cdf"); LOAD_NPY(glen, "gaussian_cdf_length"); LOAD_NPY(goff, "gaussian_offset");
    LOAD_NPY(zcdf, "bitest_cdf");   LOAD_NPY(zlen, "bitest_cdf_length");   LOAD_NPY(zoff, "bitest_offset");

    p->q_enc = M(DCH); p->q_dec = M(DCH); p->q_feat = M(DCH);
    LOAD_QBANK("q_encoder", q_enc);
    LOAD_QBANK("q_decoder", q_dec);
    LOAD_QBANK("q_feature", q_feat);

    p->rans_enc = dcvc_rans_encoder_create();
    p->rans_dec = dcvc_rans_decoder_create();

    size_t fHW = (size_t)p->fH * p->fW, yHW = (size_t)p->yH * p->yW, zHW = (size_t)p->zH * p->zW;
    size_t HpWp = (size_t)p->Hp * p->Wp;
    p->memory = M(MCH * fHW); p->ref_feature = M(DCH * fHW);
    p->ref_unshuf = M(SRCD_INTRA * fHW); p->ctx = M(DCH * fHW);
    p->x_chunk = M(SRCD_RAW * HpWp);
    p->y = M(YCH * yHW); p->z = M(ZCH * zHW); p->z_hat = M(ZCH * zHW);
    p->z_int8 = (int8_t*)calloc(ZCH * zHW, 1);
    p->temporal = M(DCH * yHW); p->hier = M(YCH * yHW);
    p->pfus_in = M(3 * YCH * yHW); p->params = M(3 * YCH * yHW);
    p->common = M(YCH * yHW); p->scales = M(YCH * yHW); p->means = M(YCH * yHW);
    p->q_enc_v = M(YCH * yHW); p->q_dec_v = M(YCH * yHW);
    p->y_scaled = M(YCH * yHW); p->yhat = M(YCH * yHW);
    p->yq = M(YCH * yHW); p->yq_w = M((YCH/4) * yHW);
    p->scl_mask = M(YCH * yHW); p->sr = M((YCH/4) * yHW);
    p->adp_in = M(2 * YCH * yHW); p->sp_out = M(2 * YCH * yHW);
    p->sp_sm = M(2 * YCH * yHW);
    p->packed = (int16_t*)calloc((YCH/4) * yHW, sizeof(int16_t));
    p->indexes = (uint8_t*)calloc((YCH/4) * yHW, 1);
    for (int m = 0; m < 4; m++) p->masks[m] = M(YCH * yHW);
    p->feature_dec = M(DCH * fHW);
    for (int f = 0; f < FRAMES; f++) p->recon_out[f] = M(3 * HpWp);
    p->frame_pad = M(3 * HpWp); p->frame_rgb = M(3 * HpWp);

    float* allocs[] = { p->q_enc,p->q_dec,p->q_feat,p->memory,p->ref_feature,p->ref_unshuf,
        p->ctx,p->x_chunk,p->y,p->z,p->z_hat,p->temporal,p->hier,p->pfus_in,p->params,
        p->common,p->scales,p->means,p->q_enc_v,p->q_dec_v,p->y_scaled,p->yhat,p->yq,
        p->yq_w,p->scl_mask,p->sr,p->adp_in,p->sp_out,p->sp_sm,p->packed?(float*)1:NULL };
    (void)allocs;
    for (size_t i = 0; i < sizeof(allocs)/sizeof(allocs[0]); i++)
        if (!allocs[i]) { st = DCVC_CPU_ERR_OOM; goto fail; }

    build_masks_4x(p);
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

void dcvc_cpu_inter_pipeline_destroy(DcvcCpuInterPipeline* p)
{
    if (!p) return;
#define E(x) if (x) dcvc_cpu_engine_destroy(x)
    E(p->eng_fai);E(p->eng_fam);E(p->eng_fe);E(p->eng_enc);E(p->eng_henc);E(p->eng_hdec);
    E(p->eng_temp);E(p->eng_pfus);E(p->eng_red);
    E(p->eng_adp[1]);E(p->eng_adp[2]);E(p->eng_adp[3]);E(p->eng_sp);E(p->eng_dec);E(p->eng_recon);
#undef E
    if (p->rans_enc) dcvc_rans_encoder_destroy(p->rans_enc);
    if (p->rans_dec) dcvc_rans_decoder_destroy(p->rans_dec);
    free(p->q_enc);free(p->q_dec);free(p->q_feat);
    free(p->memory);free(p->ref_feature);free(p->ref_unshuf);free(p->ctx);free(p->x_chunk);
    free(p->y);free(p->z);free(p->z_hat);free(p->z_int8);
    free(p->temporal);free(p->hier);free(p->pfus_in);free(p->params);
    free(p->common);free(p->scales);free(p->means);free(p->q_enc_v);free(p->q_dec_v);
    free(p->y_scaled);free(p->yhat);free(p->yq);free(p->yq_w);free(p->scl_mask);free(p->sr);
    free(p->adp_in);free(p->sp_out);free(p->packed);free(p->indexes);
    free(p->sp_sm);
    for (int m = 0; m < 4; m++) free(p->masks[m]);
    free(p->feature_dec);
    for (int f = 0; f < FRAMES; f++) free(p->recon_out[f]);
    free(p->frame_pad);free(p->frame_rgb);
    free(p);
}

/* ---------- feature-memory DPB ---------- */
static DcvcCpuStatus apply_feature_adaptor(DcvcCpuInterPipeline* p, int reset,
                                           const float* x_ref)
{
    if (reset) {
        /* first chunk: ref = pixel_unshuffle(intra recon) -> feature_adaptor_i */
        replicate_pad_3(x_ref, p->H, p->W, p->frame_pad, p->Hp, p->Wp);
        rgb_to_ycbcr(p->frame_pad, p->frame_rgb, p->Hp * p->Wp);
        pixel_unshuffle_8(p->frame_rgb, p->ref_unshuf, p->Hp, p->Wp, p->fH, p->fW);
            DcvcCpuStatus st = run1(p->eng_fai, p->ref_unshuf, SRCD_INTRA, p->fH, p->fW,
                                p->memory, MCH, p->fH, p->fW);
        if (st != DCVC_CPU_OK) return st;
        p->has_memory = 1;
    } else {
        /* subsequent chunk: memory = feature_adaptor_m(memory, prev_feature) */
        DcvcCpuStatus st = run2(p->eng_fam, p->memory, MCH, p->fH, p->fW,
                                p->ref_feature, DCH, p->fH, p->fW,
                                p->memory, MCH, p->fH, p->fW);
        if (st != DCVC_CPU_OK) return st;
    }
    return run1(p->eng_fe, p->memory, MCH, p->fH, p->fW, p->ctx, DCH, p->fH, p->fW);
}

/* params = prior_fusion(cat(hyper_decoder(z_hat), temporal_prior(memory,q_feat)))
 *   hier      [256, yH, yW]
 *   temporal  [512, yH, yW]
 *   params    [768, yH, yW] */
static DcvcCpuStatus prior_params(DcvcCpuInterPipeline* p)
{
    DcvcCpuStatus st;
    size_t yHW = (size_t)p->yH * p->yW;
    st = run1(p->eng_hdec, p->z_hat, ZCH, p->zH, p->zW, p->hier, YCH, p->yH, p->yW);
    if (st != DCVC_CPU_OK) return st;
    st = run2(p->eng_temp, p->memory, MCH, p->fH, p->fW, p->q_feat, DCH, 1, 1,
              p->temporal, DCH, p->yH, p->yW);
    if (st != DCVC_CPU_OK) return st;
    memcpy(p->pfus_in, p->hier, YCH * yHW * sizeof(float));
    memcpy(p->pfus_in + YCH * yHW, p->temporal, 2 * YCH * yHW * sizeof(float));
    return run1(p->eng_pfus, p->pfus_in, 3 * YCH, p->yH, p->yW,
                p->params, 3 * YCH, p->yH, p->yW);
}

/* spatial prior: adaptor_r(cat(yhat, common)) -> sp -> means (HT-S, 256ch,
 * scales unchanged) or scales+means (HT-L, 512ch, both updated). */
static DcvcCpuStatus run_sp(DcvcCpuInterPipeline* p, const float* yhat, int r)
{
    size_t yHW = (size_t)p->yH * p->yW;
    memcpy(p->adp_in, yhat, YCH * yHW * sizeof(float));
    memcpy(p->adp_in + YCH * yHW, p->common, YCH * yHW * sizeof(float));
    DcvcCpuStatus st = run1(p->eng_adp[r], p->adp_in, 2 * YCH, p->yH, p->yW,
                            p->sp_out, 2 * YCH, p->yH, p->yW);
    if (st != DCVC_CPU_OK) return st;
    if (p->is_hts) {
        return run1(p->eng_sp, p->sp_out, 2 * YCH, p->yH, p->yW,
                    p->means, YCH, p->yH, p->yW);
    }
    st = run1(p->eng_sp, p->sp_out, 2 * YCH, p->yH, p->yW,
              p->sp_sm, 2 * YCH, p->yH, p->yW);
    if (st != DCVC_CPU_OK) return st;
    memcpy(p->scales, p->sp_sm, YCH * yHW * sizeof(float));
    memcpy(p->means, p->sp_sm + YCH * yHW, YCH * yHW * sizeof(float));
    return DCVC_CPU_OK;
}

/* 4x prior encode (HT-S): y_scaled -> rANS symbols, final y_hat into p->yhat.
 * y on entry is the raw encoder output; y_scaled = y * q_enc is computed here. */
static DcvcCpuStatus prior_4x_encode(DcvcCpuInterPipeline* p)
{
    size_t yHW = (size_t)p->yH * p->yW;
    int n = YCH * yHW, quarter = (YCH / 4) * yHW;
    separate_prior_video(p->params, p->q_enc_v, p->q_dec_v, p->scales, p->means, n);
    DcvcCpuStatus st = run1(p->eng_red, p->params, 3 * YCH, p->yH, p->yW,
                            p->common, YCH, p->yH, p->yW);
    if (st != DCVC_CPU_OK) return st;
    elem_mul(p->y, p->q_enc_v, p->y_scaled, YCH * yHW);
    memset(p->yhat, 0, n * sizeof(float));

    p->g_cdf_idx = dcvc_rans_encoder_add_cdf(p->rans_enc, dcvc_npy_i32(&p->gcdf),
                        p->gcdf.dims[0], p->gcdf.dims[1], dcvc_npy_i32(&p->glen), dcvc_npy_i32(&p->goff));

    for (int round = 0; round < 4; round++) {
        float* means = p->means;
        if (round > 0) {
            st = run_sp(p, p->yhat, round);
            if (st != DCVC_CPU_OK) return st;
            means = p->means;
        }
        float* mask = p->masks[round];
        process_mask_yq_enc(p->y_scaled, p->scales, means, mask, p->yq, n, p->skip_thres);
        sp4x(p->yq, p->yq_w, quarter);
        for (int i = 0; i < n; i++) p->scl_mask[i] = p->scales[i] * mask[i];
        sp4x(p->scl_mask, p->sr, quarter);
        build_index_enc(p->yq_w, p->sr, p->packed, quarter);
        dcvc_rans_encoder_encode_y(p->rans_enc, p->packed, quarter, p->g_cdf_idx);
        restore_y_4x(p->yq_w, means, mask, p->scl_mask, quarter, n);
        for (int i = 0; i < n; i++) p->yhat[i] += p->scl_mask[i];
    }
    for (int i = 0; i < n; i++) p->yhat[i] *= p->q_dec_v[i];
    return DCVC_CPU_OK;
}

/* 4x prior decode (HT-S): common_params already in p->params; reads rANS y. */
static DcvcCpuStatus prior_4x_decode(DcvcCpuInterPipeline* p)
{
    size_t yHW = (size_t)p->yH * p->yW;
    int n = YCH * yHW, quarter = (YCH / 4) * yHW;
    separate_prior_video(p->params, p->q_enc_v, p->q_dec_v, p->scales, p->means, n);
    DcvcCpuStatus st = run1(p->eng_red, p->params, 3 * YCH, p->yH, p->yW,
                            p->common, YCH, p->yH, p->yW);
    if (st != DCVC_CPU_OK) return st;
    memset(p->yhat, 0, n * sizeof(float));

    p->g_cdf_idx = dcvc_rans_decoder_add_cdf(p->rans_dec, dcvc_npy_i32(&p->gcdf),
                        p->gcdf.dims[0], p->gcdf.dims[1], dcvc_npy_i32(&p->glen), dcvc_npy_i32(&p->goff));

    for (int round = 0; round < 4; round++) {
        float* means = p->means;
        if (round > 0) {
            st = run_sp(p, p->yhat, round);
            if (st != DCVC_CPU_OK) return st;
            means = p->means;
        }
        float* mask = p->masks[round];
        for (int i = 0; i < n; i++) p->scl_mask[i] = p->scales[i] * mask[i];
        sp4x(p->scl_mask, p->sr, quarter);
        build_index_dec(p->sr, p->indexes, quarter);
        dcvc_rans_decoder_decode_y(p->rans_dec, p->indexes, quarter, p->g_cdf_idx);
        if (dcvc_rans_decoder_has_error(p->rans_dec)) return DCVC_CPU_ERR_ENTROPY;
        int8_t* syms = NULL; size_t symn = 0;
        dcvc_rans_decoder_get_symbols(p->rans_dec, &syms, &symn);
        for (int i = 0; i < n; i++) p->yq[i] = (float)syms[i % quarter];
        restore_y_4x(p->yq, means, mask, p->scl_mask, quarter, n);
        for (int i = 0; i < n; i++) p->yhat[i] += p->scl_mask[i];
    }
    for (int i = 0; i < n; i++) p->yhat[i] *= p->q_dec_v[i];
    return DCVC_CPU_OK;
}

/* decoder + recon_head -> 8 reconstructed YCbCr frames in p->recon_out[],
 * stores the shared feature plane in p->feature_dec for the DPB. */
static DcvcCpuStatus reconstruct(DcvcCpuInterPipeline* p)
{
    DcvcCpuStatus st = run3(p->eng_dec, p->yhat, YCH, p->yH, p->yW,
                            p->ctx, DCH, p->fH, p->fW, p->q_dec, DCH, 1, 1,
                            p->feature_dec, DCH, p->fH, p->fW);
    if (st != DCVC_CPU_OK) return st;
    return run_recon(p, p->feature_dec);
}

/* ---------- encode ---------- */
DcvcCpuStatus dcvc_cpu_inter_pipeline_encode(DcvcCpuInterPipeline* p,
                                             const float* x_chunk,
                                             int reset, const float* x_ref,
                                             uint8_t** out_stream, size_t* out_size,
                                             float* x_hat_out)
{
    if (!p || !x_chunk || !out_stream || !out_size || (reset && !x_ref))
        return DCVC_CPU_ERR_INVALID_ARG;
    *out_stream = NULL; *out_size = 0;
    DcvcCpuStatus st;
    size_t zHW = (size_t)p->zH * p->zW;

    st = apply_feature_adaptor(p, reset, x_ref);
    if (st != DCVC_CPU_OK) return st;

    build_chunk_input(p, x_chunk);
    st = run3(p->eng_enc, p->x_chunk, SRCD_RAW, p->Hp, p->Wp,
              p->ctx, DCH, p->fH, p->fW, p->q_enc, DCH, 1, 1,
              p->y, YCH, p->yH, p->yW);
    if (st != DCVC_CPU_OK) return st;
    { const char* dp=getenv("DCVC_DUMP_Y"); if(dp&&dp[0]){int d[4]={1,YCH,p->yH,p->yW};dcvc_npy_write_f32(dp,p->y,d,4);} }
    st = run1(p->eng_henc, p->y, YCH, p->yH, p->yW, p->z, ZCH, p->zH, p->zW);
    if (st != DCVC_CPU_OK) return st;
    round_to_int8(p->z, p->z_hat, p->z_int8, ZCH * zHW);

    st = prior_params(p);
    if (st != DCVC_CPU_OK) return st;

    dcvc_rans_encoder_reset(p->rans_enc);
    p->z_cdf_idx = dcvc_rans_encoder_add_cdf(p->rans_enc, dcvc_npy_i32(&p->zcdf),
                        p->zcdf.dims[0], p->zcdf.dims[1], dcvc_npy_i32(&p->zlen), dcvc_npy_i32(&p->zoff));
    dcvc_rans_encoder_encode_z(p->rans_enc, p->z_int8, ZCH * zHW, p->z_cdf_idx, p->qp * ZCH, zHW);
    st = prior_4x_encode(p);
    if (st != DCVC_CPU_OK) return st;
    dcvc_rans_encoder_flush(p->rans_enc);
    uint8_t* stream = NULL; size_t ssize = 0;
    if (dcvc_rans_encoder_get_stream(p->rans_enc, &stream, &ssize) != 0)
        return DCVC_CPU_ERR_ONNX;

    st = reconstruct(p);
    if (st != DCVC_CPU_OK) { free(stream); return st; }
    /* store feature plane for the next chunk's feature_adaptor_m */
    memcpy(p->ref_feature, p->feature_dec, (size_t)DCH * p->fH * p->fW * sizeof(float));

    if (x_hat_out) {
        size_t per = 3 * p->H * p->W;
        for (int f = 0; f < FRAMES; f++) {
            ycbcr_to_rgb(p->recon_out[f], p->frame_pad, p->Hp * p->Wp);
            crop_3(p->frame_pad, p->Hp, p->Wp, x_hat_out + f * per, p->H, p->W);
        }
        dcvc_debug_dump_xhat("DCVC_DUMP_XHAT", x_hat_out, FRAMES, p->H, p->W);
    }
    *out_stream = stream; *out_size = ssize;
    return DCVC_CPU_OK;
}

/* ---------- decode ---------- */
DcvcCpuStatus dcvc_cpu_inter_pipeline_decode(DcvcCpuInterPipeline* p,
                                             const uint8_t* stream, size_t stream_size,
                                             int reset, const float* x_ref,
                                             float* x_hat_out)
{
    if (!p || !stream || !x_hat_out || (reset && !x_ref))
        return DCVC_CPU_ERR_INVALID_ARG;
    DcvcCpuStatus st;
    size_t zHW = (size_t)p->zH * p->zW;

    st = apply_feature_adaptor(p, reset, x_ref);
    if (st != DCVC_CPU_OK) return st;

    dcvc_rans_decoder_reset_cdf(p->rans_dec);
    p->z_cdf_idx = dcvc_rans_decoder_add_cdf(p->rans_dec, dcvc_npy_i32(&p->zcdf),
                        p->zcdf.dims[0], p->zcdf.dims[1], dcvc_npy_i32(&p->zlen), dcvc_npy_i32(&p->zoff));
    dcvc_rans_decoder_set_stream(p->rans_dec, stream, stream_size);
    dcvc_rans_decoder_decode_z(p->rans_dec, ZCH * zHW, p->z_cdf_idx, p->qp * ZCH, zHW);
    if (dcvc_rans_decoder_has_error(p->rans_dec)) return DCVC_CPU_ERR_ENTROPY;
    int8_t* zsyms = NULL; size_t zn = 0;
    dcvc_rans_decoder_get_symbols(p->rans_dec, &zsyms, &zn);
    int8_to_float(zsyms, p->z_hat, ZCH * zHW);

    st = prior_params(p);
    if (st != DCVC_CPU_OK) return st;
    st = prior_4x_decode(p);
    if (st != DCVC_CPU_OK) return st;
    if (dcvc_rans_decoder_bytes_consumed(p->rans_dec) != stream_size)
        return DCVC_CPU_ERR_ENTROPY;

    st = reconstruct(p);
    if (st != DCVC_CPU_OK) return st;
    memcpy(p->ref_feature, p->feature_dec, (size_t)DCH * p->fH * p->fW * sizeof(float));

    size_t per = 3 * p->H * p->W;
    for (int f = 0; f < FRAMES; f++) {
        ycbcr_to_rgb(p->recon_out[f], p->frame_pad, p->Hp * p->Wp);
        crop_3(p->frame_pad, p->Hp, p->Wp, x_hat_out + f * per, p->H, p->W);
    }
    return DCVC_CPU_OK;
}
