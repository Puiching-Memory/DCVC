/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Pure-CPU intra-frame pipeline implementation.
 */
#include "cpu_intra_pipeline.h"
#include "cpu_ar_codec.h"
#include "npy_reader.h"
#include "onnx_engine.h"
#include "rans_c.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Optional debug dump, triggered by environment variable
 *   DCVC_DUMP_PARAMS=<path>   -> dump params_fusion (the prior) to <path>.npy
 *   DCVC_DUMP_Y=<path>        -> dump the analysis output y to <path>.npy
 * Used to measure cross-platform ONNX Runtime FP differences. */
static void dcvc_debug_dump(const char* env_var, const float* data,
                            int c, int h, int w)
{
    const char* path = getenv(env_var);
    if (!path || !path[0]) return;
    int dims[4] = {1, c, h, w};
    if (dcvc_npy_write_f32(path, data, dims, 4) == 0)
        fprintf(stderr, "[debug] dumped %s [%dx%dx%d] to %s\n", env_var, c, h, w, path);
}

struct DcvcCpuIntraPipeline {
    int H, W;            /* actual (cropped) frame size        */
    int Hp, Wp;          /* padded size, multiple of 64        */
    int yH, yW, zH, zW;  /* derived from the PADDED size       */
    int N, ZC;
    int qp;

    DcvcCpuEngine* eng_analysis;
    DcvcCpuEngine* eng_hyper_enc;
    DcvcCpuEngine* eng_hyper_dec;
    DcvcCpuEngine* eng_prior_fusion;
    DcvcCpuEngine* eng_synthesis;
    DcvcCpuArCodec* ar_codec;

    DcvcRansEncoder* rans_enc;
    DcvcRansDecoder* rans_dec;
    DcvcNpy zcdf, zlen, zoff;
    int z_cdf_idx;

    float* qenc; /* 368 floats */
    float* qdec; /* 368 floats */

    /* Workspace */
    float* y;            /* N * yH * yW */
    float* y_pad;        /* N * pad_yH * pad_yW */
    float* z;            /* ZC * zH * zW */
    float* z_hat;        /* ZC * zH * zW */
    int8_t* z_int8;      /* ZC * zH * zW */
    float* params;       /* N * yH * yW */
    float* params_fusion;/* (2N+2) * yH * yW */
    float* y_hat;        /* N * yH * yW */
    float* x_pad;        /* 3 * Hp * Wp (replicate-padded YCbCr input) */
    float* x_hat;        /* 3 * Hp * Wp (cropped RGB output) */
    float* x_ycbcr;      /* 3 * Hp * Wp (intermediate YCbCr for encode) */
};

static DcvcCpuStatus run_engine(DcvcCpuEngine* eng,
                                 const float* in, int n, int c, int h, int w,
                                 float* out, int on, int oc, int oh, int ow)
{
    DcvcCpuTensorView inputs[1] = { { (void*)in, 0, n, c, h, w } };
    DcvcCpuTensorView outputs[1] = { { out, 0, on, oc, oh, ow } };
    return dcvc_cpu_engine_run(eng, inputs, 1, outputs, 1);
}

static DcvcCpuStatus run_engine2(DcvcCpuEngine* eng,
                                  const float* in0, int n0, int c0, int h0, int w0,
                                  const float* in1, int n1, int c1, int h1, int w1,
                                  float* out, int on, int oc, int oh, int ow)
{
    DcvcCpuTensorView inputs[2] = {
        { (void*)in0, 0, n0, c0, h0, w0 },
        { (void*)in1, 0, n1, c1, h1, w1 }
    };
    DcvcCpuTensorView outputs[1] = { { out, 0, on, oc, oh, ow } };
    return dcvc_cpu_engine_run(eng, inputs, 2, outputs, 1);
}

static void round_to_int8(const float* in, float* out, int8_t* out_i8, int n)
{
    for (int i = 0; i < n; i++) {
        float v = roundf(in[i]);
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        out[i] = v;
        out_i8[i] = (int8_t)v;
    }
}

static void int8_to_float(const int8_t* in, float* out, int n)
{
    for (int i = 0; i < n; i++) out[i] = (float)in[i];
}

/* Replicate (edge) pad x [3,H,W] -> dst [3,Hp,Wp] (NCHW, C=3).
 * Matches PyTorch F.pad(..., mode="replicate") along bottom/right. */
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
        float y  = ycbcr[i + 0 * n];
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

DcvcCpuIntraPipeline* dcvc_cpu_intra_pipeline_create(const char* model_dir,
                                                      int H, int W, int qp,
                                                      DcvcCpuStatus* out_st)
{
    if (out_st) *out_st = DCVC_CPU_OK;
    if (H <= 0 || W <= 0) {
        if (out_st) *out_st = DCVC_CPU_ERR_INVALID_ARG;
        return NULL;
    }
    if (H % 64 != 0 || W % 64 != 0) {
        fprintf(stderr, "error: intra pipeline requires H and W to be multiples of 64 (got %dx%d)\n", H, W);
        if (out_st) *out_st = DCVC_CPU_ERR_INVALID_ARG;
        return NULL;
    }

    DcvcCpuIntraPipeline* p = (DcvcCpuIntraPipeline*)calloc(1, sizeof(*p));
    if (!p) { if (out_st) *out_st = DCVC_CPU_ERR_OOM; return NULL; }
    p->H = H; p->W = W; p->qp = qp;
    p->Hp = (H + 63) / 64 * 64;   /* replicate-pad up to a multiple of 64 */
    p->Wp = (W + 63) / 64 * 64;
    p->yH = p->Hp / 16; p->yW = p->Wp / 16;
    p->zH = p->Hp / 64; p->zW = p->Wp / 64;
    p->N = 256; p->ZC = 128;

    char path[640];
    DcvcCpuStatus st = DCVC_CPU_OK;

    snprintf(path, sizeof(path), "%s/intra_analysis_standard.onnx", model_dir);
    p->eng_analysis = dcvc_cpu_engine_create(path, 0, &st);
    if (!p->eng_analysis) goto fail;

    snprintf(path, sizeof(path), "%s/intra_hyper_enc.onnx", model_dir);
    p->eng_hyper_enc = dcvc_cpu_engine_create(path, 0, &st);
    if (!p->eng_hyper_enc) goto fail;

    snprintf(path, sizeof(path), "%s/hyper_dec.onnx", model_dir);
    p->eng_hyper_dec = dcvc_cpu_engine_create(path, 0, &st);
    if (!p->eng_hyper_dec) goto fail;

    snprintf(path, sizeof(path), "%s/y_prior_fusion.onnx", model_dir);
    p->eng_prior_fusion = dcvc_cpu_engine_create(path, 0, &st);
    if (!p->eng_prior_fusion) goto fail;

    snprintf(path, sizeof(path), "%s/intra_synthesis.onnx", model_dir);
    p->eng_synthesis = dcvc_cpu_engine_create(path, 0, &st);
    if (!p->eng_synthesis) goto fail;

    p->ar_codec = dcvc_cpu_ar_codec_create(model_dir, p->N, &st);
    if (!p->ar_codec) goto fail;

    p->rans_enc = dcvc_rans_encoder_create();
    p->rans_dec = dcvc_rans_decoder_create();
    if (!p->rans_enc || !p->rans_dec) { st = DCVC_CPU_ERR_OOM; goto fail; }

    snprintf(path, sizeof(path), "%s/bitest_cdf.npy", model_dir);
    if (dcvc_npy_read(path, &p->zcdf) != 0) { st = DCVC_CPU_ERR_IO; goto fail; }
    snprintf(path, sizeof(path), "%s/bitest_cdf_length.npy", model_dir);
    if (dcvc_npy_read(path, &p->zlen) != 0) { st = DCVC_CPU_ERR_IO; goto fail; }
    snprintf(path, sizeof(path), "%s/bitest_offset.npy", model_dir);
    if (dcvc_npy_read(path, &p->zoff) != 0) { st = DCVC_CPU_ERR_IO; goto fail; }

    snprintf(path, sizeof(path), "%s/q_scale_enc.npy", model_dir);
    DcvcNpy qe = {0};
    if (dcvc_npy_read(path, &qe) != 0) { st = DCVC_CPU_ERR_IO; goto fail; }
    if (qp < 0 || qp >= qe.dims[0]) { dcvc_npy_free(&qe); st = DCVC_CPU_ERR_INVALID_ARG; goto fail; }
    p->qenc = (float*)malloc(qe.dims[1] * sizeof(float));
    if (!p->qenc) { dcvc_npy_free(&qe); st = DCVC_CPU_ERR_OOM; goto fail; }
    for (int i = 0; i < qe.dims[1]; i++) p->qenc[i] = dcvc_npy_f32(&qe)[qp * qe.dims[1] + i];
    dcvc_npy_free(&qe);

    snprintf(path, sizeof(path), "%s/q_scale_dec.npy", model_dir);
    DcvcNpy qd = {0};
    if (dcvc_npy_read(path, &qd) != 0) { st = DCVC_CPU_ERR_IO; goto fail; }
    if (qp < 0 || qp >= qd.dims[0]) { dcvc_npy_free(&qd); st = DCVC_CPU_ERR_INVALID_ARG; goto fail; }
    p->qdec = (float*)malloc(qd.dims[1] * sizeof(float));
    if (!p->qdec) { dcvc_npy_free(&qd); st = DCVC_CPU_ERR_OOM; goto fail; }
    for (int i = 0; i < qd.dims[1]; i++) p->qdec[i] = dcvc_npy_f32(&qd)[qp * qd.dims[1] + i];
    dcvc_npy_free(&qd);

    int yhw = p->yH * p->yW;
    int zhw = p->zH * p->zW;
    p->y = (float*)malloc(p->N * yhw * sizeof(float));
    p->y_pad = (float*)malloc(p->N * yhw * sizeof(float));
    p->z = (float*)malloc(p->ZC * zhw * sizeof(float));
    p->z_hat = (float*)malloc(p->ZC * zhw * sizeof(float));
    p->z_int8 = (int8_t*)malloc(p->ZC * zhw);
    p->params = (float*)malloc(p->N * yhw * sizeof(float));
    p->params_fusion = (float*)malloc((2 * p->N + 2) * yhw * sizeof(float));
    p->y_hat = (float*)malloc(p->N * yhw * sizeof(float));
    p->x_pad = (float*)malloc(3 * p->Hp * p->Wp * sizeof(float));
    p->x_hat = (float*)malloc(3 * p->Hp * p->Wp * sizeof(float));
    p->x_ycbcr = (float*)malloc(3 * p->Hp * p->Wp * sizeof(float));
    if (!p->y || !p->y_pad || !p->z || !p->z_hat || !p->z_int8 ||
        !p->params || !p->params_fusion || !p->y_hat || !p->x_pad || !p->x_hat ||
        !p->x_ycbcr) {
        st = DCVC_CPU_ERR_OOM; goto fail;
    }

    return p;

fail:
    dcvc_cpu_intra_pipeline_destroy(p);
    if (out_st) *out_st = st;
    return NULL;
}

void dcvc_cpu_intra_pipeline_destroy(DcvcCpuIntraPipeline* p)
{
    if (!p) return;
    if (p->eng_analysis) dcvc_cpu_engine_destroy(p->eng_analysis);
    if (p->eng_hyper_enc) dcvc_cpu_engine_destroy(p->eng_hyper_enc);
    if (p->eng_hyper_dec) dcvc_cpu_engine_destroy(p->eng_hyper_dec);
    if (p->eng_prior_fusion) dcvc_cpu_engine_destroy(p->eng_prior_fusion);
    if (p->eng_synthesis) dcvc_cpu_engine_destroy(p->eng_synthesis);
    if (p->ar_codec) dcvc_cpu_ar_codec_destroy(p->ar_codec);
    if (p->rans_enc) dcvc_rans_encoder_destroy(p->rans_enc);
    if (p->rans_dec) dcvc_rans_decoder_destroy(p->rans_dec);
    dcvc_npy_free(&p->zcdf); dcvc_npy_free(&p->zlen); dcvc_npy_free(&p->zoff);
    free(p->qenc); free(p->qdec);
    free(p->y); free(p->y_pad); free(p->z); free(p->z_hat); free(p->z_int8);
    free(p->params); free(p->params_fusion); free(p->y_hat); free(p->x_hat);
    free(p->x_pad); free(p->x_ycbcr);
    free(p);
}

DcvcCpuStatus dcvc_cpu_intra_pipeline_encode(DcvcCpuIntraPipeline* p,
                                              const float* x,
                                              uint8_t** out_stream, size_t* out_size,
                                              float* x_hat_out)
{
    if (!p || !x || !out_stream || !out_size) return DCVC_CPU_ERR_INVALID_ARG;
    *out_stream = NULL; *out_size = 0;

    int H = p->H, W = p->W;
    int Hp = p->Hp, Wp = p->Wp;
    int yhw = p->yH * p->yW;
    int zhw = p->zH * p->zW;
    DcvcCpuStatus st;

    /* Step 1: replicate-pad x to a multiple of 64, convert RGB to YCbCr,
     * then analysis: x_pad [1,3,Hp,Wp] + qenc [1,368,1,1] -> y [1,256,yH,yW].
     * The public API now always accepts RGB and performs the color conversion
     * internally, matching the PyTorch test_video.py path. */
    replicate_pad_3(x, H, W, p->x_pad, Hp, Wp);
    rgb_to_ycbcr(p->x_pad, p->x_ycbcr, Hp * Wp);
    st = run_engine2(p->eng_analysis, p->x_ycbcr, 1, 3, Hp, Wp,
                     p->qenc, 1, 368, 1, 1,
                     p->y, 1, p->N, p->yH, p->yW);
    if (st != DCVC_CPU_OK) return st;

    /* Clamp y to [-128, 127] to match PyTorch path */
    for (int i = 0; i < p->N * yhw; i++) {
        float v = p->y[i];
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        p->y[i] = v;
    }

    dcvc_debug_dump("DCVC_DUMP_Y", p->y, p->N, p->yH, p->yW);

    /* Step 2: hyper_enc: y -> z. H,W multiple of 64 => yH,yW multiple of 4 => no padding. */
    memcpy(p->y_pad, p->y, p->N * yhw * sizeof(float));
    st = run_engine(p->eng_hyper_enc, p->y_pad, 1, p->N, p->yH, p->yW,
                    p->z, 1, p->ZC, p->zH, p->zW);
    if (st != DCVC_CPU_OK) return st;

    /* Step 3: round z to int8 */
    round_to_int8(p->z, p->z_hat, p->z_int8, p->ZC * zhw);

    /* Step 4: encode z with rANS */
    dcvc_rans_encoder_reset(p->rans_enc);
    p->z_cdf_idx = dcvc_rans_encoder_add_cdf(p->rans_enc, dcvc_npy_i32(&p->zcdf),
                        p->zcdf.dims[0], p->zcdf.dims[1],
                        dcvc_npy_i32(&p->zlen), dcvc_npy_i32(&p->zoff));
    dcvc_rans_encoder_encode_z(p->rans_enc, p->z_int8, p->ZC * zhw,
                                p->z_cdf_idx, p->qp * p->ZC, zhw);
    dcvc_rans_encoder_flush(p->rans_enc);
    uint8_t* z_stream = NULL; size_t z_stream_size = 0;
    int rc = dcvc_rans_encoder_get_stream(p->rans_enc, &z_stream, &z_stream_size);
    if (rc != 0) return DCVC_CPU_ERR_ONNX;

    /* Step 5: hyper_dec: z_hat -> params */
    st = run_engine(p->eng_hyper_dec, p->z_hat, 1, p->ZC, p->zH, p->zW,
                    p->params, 1, p->N, p->yH, p->yW);
    if (st != DCVC_CPU_OK) { free(z_stream); return st; }

    /* Step 6: y_prior_fusion: params -> params_fusion */
    st = run_engine(p->eng_prior_fusion, p->params, 1, p->N, p->yH, p->yW,
                    p->params_fusion, 1, 2 * p->N + 2, p->yH, p->yW);
    if (st != DCVC_CPU_OK) { free(z_stream); return st; }

    dcvc_debug_dump("DCVC_DUMP_PARAMS", p->params_fusion, 2 * p->N + 2, p->yH, p->yW);

    /* Step 7: AR codec encode */
    uint8_t* y_stream = NULL; size_t y_stream_size = 0;
    st = dcvc_cpu_ar_codec_encode_y(p->ar_codec, p->y, p->params_fusion,
                                     p->yH, p->yW, &y_stream, &y_stream_size, p->y_hat);
    if (st != DCVC_CPU_OK) { free(z_stream); return st; }

    /* Step 8: Mux [z_len:u32][z_stream][y_stream] */
    size_t total = 4 + z_stream_size + y_stream_size;
    uint8_t* buf = (uint8_t*)malloc(total);
    if (!buf) { free(z_stream); free(y_stream); return DCVC_CPU_ERR_OOM; }
    uint32_t zlen32 = (uint32_t)z_stream_size;
    memcpy(buf, &zlen32, 4);
    if (z_stream_size) memcpy(buf + 4, z_stream, z_stream_size);
    if (y_stream_size) memcpy(buf + 4 + z_stream_size, y_stream, y_stream_size);
    free(z_stream); free(y_stream);
    *out_stream = buf;
    *out_size = total;

    /* Step 9: optional synthesis, then YCbCr -> RGB conversion on output */
    if (x_hat_out) {
        st = run_engine2(p->eng_synthesis, p->y_hat, 1, p->N, p->yH, p->yW,
                         p->qdec, 1, 368, 1, 1,
                         p->x_ycbcr, 1, 3, Hp, Wp);
        if (st != DCVC_CPU_OK) return st;
        ycbcr_to_rgb(p->x_ycbcr, p->x_hat, Hp * Wp);
        crop_3(p->x_hat, Hp, Wp, x_hat_out, H, W);
    }

    return DCVC_CPU_OK;
}

DcvcCpuStatus dcvc_cpu_intra_pipeline_decode(DcvcCpuIntraPipeline* p,
                                              const uint8_t* stream, size_t stream_size,
                                              float* x_hat_out)
{
    if (!p || !stream || !x_hat_out || stream_size < 4) return DCVC_CPU_ERR_INVALID_ARG;

    int H = p->H, W = p->W;
    int Hp = p->Hp, Wp = p->Wp;
    int zhw = p->zH * p->zW;
    DcvcCpuStatus st;

    /* Demux */
    uint32_t z_len = 0;
    memcpy(&z_len, stream, 4);
    if (4 + (size_t)z_len > stream_size) return DCVC_CPU_ERR_UNSUPPORTED;
    const uint8_t* z_payload = stream + 4;
    const uint8_t* y_payload = z_payload + z_len;
    size_t y_payload_size = stream_size - 4 - z_len;

    /* Decode z */
    dcvc_rans_decoder_reset_cdf(p->rans_dec);
    p->z_cdf_idx = dcvc_rans_decoder_add_cdf(p->rans_dec, dcvc_npy_i32(&p->zcdf),
                        p->zcdf.dims[0], p->zcdf.dims[1],
                        dcvc_npy_i32(&p->zlen), dcvc_npy_i32(&p->zoff));
    dcvc_rans_decoder_set_stream(p->rans_dec, z_payload, z_len);
    dcvc_rans_decoder_decode_z(p->rans_dec, p->ZC * zhw, p->z_cdf_idx,
                                p->qp * p->ZC, zhw);
    int8_t* z_syms = NULL; size_t z_sym_n = 0;
    dcvc_rans_decoder_get_symbols(p->rans_dec, &z_syms, &z_sym_n);
    int8_to_float(z_syms, p->z_hat, p->ZC * zhw);

    /* hyper_dec -> params -> prior_fusion */
    st = run_engine(p->eng_hyper_dec, p->z_hat, 1, p->ZC, p->zH, p->zW,
                    p->params, 1, p->N, p->yH, p->yW);
    if (st != DCVC_CPU_OK) return st;
    st = run_engine(p->eng_prior_fusion, p->params, 1, p->N, p->yH, p->yW,
                    p->params_fusion, 1, 2 * p->N + 2, p->yH, p->yW);
    if (st != DCVC_CPU_OK) return st;

    /* AR codec decode */
    st = dcvc_cpu_ar_codec_decode_y(p->ar_codec, p->params_fusion,
                                     p->yH, p->yW, y_payload, y_payload_size, p->y_hat);
    if (st != DCVC_CPU_OK) return st;

    /* Synthesis + color conversion, then crop to original HxW RGB */
    st = run_engine2(p->eng_synthesis, p->y_hat, 1, p->N, p->yH, p->yW,
                     p->qdec, 1, 368, 1, 1,
                     p->x_ycbcr, 1, 3, Hp, Wp);
    if (st != DCVC_CPU_OK) return st;
    ycbcr_to_rgb(p->x_ycbcr, p->x_hat, Hp * Wp);
    crop_3(p->x_hat, Hp, Wp, x_hat_out, H, W);
    return DCVC_CPU_OK;
}
