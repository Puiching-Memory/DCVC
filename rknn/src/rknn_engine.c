#include "dcvc_rk/rknn_engine.h"
#include "dcvc_rk/profile.h"
#include "dcvc_rk/quant.h"

#include <rknn_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    DCVC_IO_FP32  = 0, /* FP32 → driver quant+layout; float out */
    DCVC_IO_I8    = 1, /* CPU quant INT8 NHWC + pass_through; float/INT8 out */
    DCVC_IO_NHWC  = 2, /* CPU transpose FP32 NHWC; driver quant */
    DCVC_IO_I8OUT = 3  /* FP32 in; INT8 out + CPU dequant */
} DcvcIoMode;

struct DcvcRkEngine {
    rknn_context ctx;
    uint32_t n_input;
    uint32_t n_output;
    rknn_tensor_attr* in_attr;   /* model-native attrs from query */
    rknn_tensor_attr* out_attr;
    rknn_tensor_attr* in_io;     /* attrs used with set_io_mem */
    rknn_tensor_attr* out_io;
    int64_t last_run_us;
    int64_t last_set_us;
    int64_t last_get_us;
    int64_t last_wall_us;
    DcvcIoMode io_mode;
    int use_dma;                 /* rknn_create_mem + set_io_mem */
    int dma_ready;
    rknn_tensor_mem** in_mem;
    rknn_tensor_mem** out_mem;
    /* Scratch when not using DMA (or NHWC mode). */
    int8_t** in_i8;
    int8_t** out_i8;
    float** in_nhwc;
    uint32_t* in_i8_bytes;
    uint32_t* out_i8_bytes;
    uint32_t* in_nhwc_elems;
};

static DcvcIoMode io_mode_from_env(void)
{
    const char* e = getenv("DCVC_RKNN_IO");
    if (!e) return DCVC_IO_I8; /* INT8 pass-through beats FP32 host↔NPU */
    if (!strcmp(e, "fp32") || !strcmp(e, "FP32")) return DCVC_IO_FP32;
    if (!strcmp(e, "i8") || !strcmp(e, "I8") || !strcmp(e, "int8")) return DCVC_IO_I8;
    if (!strcmp(e, "nhwc") || !strcmp(e, "NHWC")) return DCVC_IO_NHWC;
    if (!strcmp(e, "i8out") || !strcmp(e, "I8OUT")) return DCVC_IO_I8OUT;
    return DCVC_IO_I8;
}

static int dma_from_env(void)
{
    /* Default off: DMA float-out inflates PERF_RUN on many subnets; i8+inputs_set
     * is the measured sweet spot. Set DCVC_RKNN_DMA=1 to try set_io_mem. */
    const char* e = getenv("DCVC_RKNN_DMA");
    if (!e) return 0;
    if (!strcmp(e, "0") || !strcmp(e, "off") || !strcmp(e, "false")) return 0;
    return 1;
}

static const char* io_mode_name(DcvcIoMode m)
{
    switch (m) {
    case DCVC_IO_I8: return "i8";
    case DCVC_IO_NHWC: return "nhwc";
    case DCVC_IO_I8OUT: return "i8out";
    default: return "fp32";
    }
}

const char* dcvc_rk_status_string(DcvcRkStatus st)
{
    switch (st) {
    case DCVC_RK_OK: return "ok";
    case DCVC_RK_ERR_INVALID_ARG: return "invalid_arg";
    case DCVC_RK_ERR_IO: return "io";
    case DCVC_RK_ERR_RKNN: return "rknn";
    case DCVC_RK_ERR_OOM: return "oom";
    case DCVC_RK_ERR_UNSUPPORTED: return "unsupported";
    case DCVC_RK_ERR_ENTROPY: return "entropy";
    default: return "unknown";
    }
}

static void* load_file(const char* path, uint32_t* size)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (len <= 0) { fclose(fp); return NULL; }
    void* buf = malloc((size_t)len);
    if (!buf) { fclose(fp); return NULL; }
    if (fread(buf, 1, (size_t)len, fp) != (size_t)len) {
        free(buf); fclose(fp); return NULL;
    }
    fclose(fp);
    *size = (uint32_t)len;
    return buf;
}

/* FP32 NCHW → INT8 NHWC (affine). */
static void quant_nchw_f32_to_nhwc_i8(const float* src, int8_t* dst,
                                      int N, int C, int H, int W,
                                      float scale, int32_t zp)
{
    float inv = (scale > 1e-12f) ? (1.0f / scale) : 0.f;
    const size_t plane = (size_t)H * W;
    for (int n = 0; n < N; n++) {
        const float* sn = src + (size_t)n * C * plane;
        int8_t* dn = dst + (size_t)n * H * W * C;
#pragma omp parallel for collapse(2) schedule(static) if(H * W > 4096)
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int8_t* pix = dn + ((size_t)h * W + w) * C;
                size_t hw = (size_t)h * W + w;
                for (int c = 0; c < C; c++) {
                    float v = sn[c * plane + hw];
                    int q = (int)lrintf(v * inv) + (int)zp;
                    if (q > 127) q = 127;
                    if (q < -128) q = -128;
                    pix[c] = (int8_t)q;
                }
            }
        }
    }
}

/* INT8 NCHW (tight) → INT8 NHWC. H=W=1 is a no-op memcpy. */
static void nchw_i8_to_nhwc_i8(const int8_t* src, int8_t* dst, int N, int C, int H, int W)
{
    const size_t plane = (size_t)H * W;
    const size_t elems = (size_t)N * C * plane;
    if (H == 1 && W == 1) {
        memcpy(dst, src, elems);
        return;
    }
    for (int n = 0; n < N; n++) {
        const int8_t* sn = src + (size_t)n * C * plane;
        int8_t* dn = dst + (size_t)n * H * W * C;
#pragma omp parallel for collapse(2) schedule(static) if(H * W > 4096)
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int8_t* pix = dn + ((size_t)h * W + w) * C;
                size_t hw = (size_t)h * W + w;
                for (int c = 0; c < C; c++)
                    pix[c] = sn[c * plane + hw];
            }
        }
    }
}

/* Scratch for NHWC pass-through (needed when DMA stole alloc, or FP32 io_mode). */
static DcvcRkStatus ensure_in_i8_scratch(DcvcRkEngine* eng)
{
    if (eng->in_i8) return DCVC_RK_OK;
    eng->in_i8 = (int8_t**)calloc(eng->n_input, sizeof(int8_t*));
    eng->in_i8_bytes = (uint32_t*)calloc(eng->n_input, sizeof(uint32_t));
    if (!eng->in_i8 || !eng->in_i8_bytes) return DCVC_RK_ERR_OOM;
    for (uint32_t i = 0; i < eng->n_input; i++) {
        if (eng->in_attr[i].type != RKNN_TENSOR_INT8 || eng->in_attr[i].fmt != RKNN_TENSOR_NHWC)
            return DCVC_RK_ERR_UNSUPPORTED;
        uint32_t n = eng->in_attr[i].size_with_stride ? eng->in_attr[i].size_with_stride
                                                      : eng->in_attr[i].size;
        eng->in_i8_bytes[i] = n;
        eng->in_i8[i] = (int8_t*)malloc(n);
        if (!eng->in_i8[i]) return DCVC_RK_ERR_OOM;
    }
    return DCVC_RK_OK;
}

static void nchw_to_nhwc_f32(const float* src, float* dst, int N, int C, int H, int W)
{
    const size_t plane = (size_t)H * W;
    for (int n = 0; n < N; n++) {
        const float* sn = src + (size_t)n * C * plane;
        float* dn = dst + (size_t)n * H * W * C;
#pragma omp parallel for collapse(2) schedule(static) if(H * W > 4096)
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float* pix = dn + ((size_t)h * W + w) * C;
                size_t hw = (size_t)h * W + w;
                for (int c = 0; c < C; c++)
                    pix[c] = sn[c * plane + hw];
            }
        }
    }
}

static void dequant_nchw_i8_to_f32(const int8_t* src, float* dst,
                                   int N, int C, int H, int W, int w_stride,
                                   float scale, int32_t zp)
{
    if (w_stride <= 0) w_stride = W;
#pragma omp parallel for collapse(3) schedule(static) if((size_t)N * C * H * W > 16384)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                const int8_t* row = src + ((((size_t)n * C + c) * H + h) * (size_t)w_stride);
                float* drow = dst + ((((size_t)n * C + c) * H + h) * (size_t)W);
                for (int w = 0; w < W; w++)
                    drow[w] = ((float)row[w] - (float)zp) * scale;
            }
        }
    }
}

static int nhwc_dims(const rknn_tensor_attr* a, int* N, int* H, int* W, int* C)
{
    if (a->fmt != RKNN_TENSOR_NHWC || a->n_dims < 4) return -1;
    *N = (int)a->dims[0];
    *H = (int)a->dims[1];
    *W = (int)a->dims[2];
    *C = (int)a->dims[3];
    return 0;
}

static int nchw_dims(const rknn_tensor_attr* a, int* N, int* C, int* H, int* W)
{
    if (a->fmt != RKNN_TENSOR_NCHW || a->n_dims < 4) return -1;
    *N = (int)a->dims[0];
    *C = (int)a->dims[1];
    *H = (int)a->dims[2];
    *W = (int)a->dims[3];
    return 0;
}

static void free_scratch(DcvcRkEngine* eng)
{
    if (!eng) return;
    if (eng->in_i8) {
        for (uint32_t i = 0; i < eng->n_input; i++) free(eng->in_i8[i]);
        free(eng->in_i8);
    }
    if (eng->out_i8) {
        for (uint32_t i = 0; i < eng->n_output; i++) free(eng->out_i8[i]);
        free(eng->out_i8);
    }
    if (eng->in_nhwc) {
        for (uint32_t i = 0; i < eng->n_input; i++) free(eng->in_nhwc[i]);
        free(eng->in_nhwc);
    }
    free(eng->in_i8_bytes);
    free(eng->out_i8_bytes);
    free(eng->in_nhwc_elems);
    eng->in_i8 = NULL; eng->out_i8 = NULL; eng->in_nhwc = NULL;
    eng->in_i8_bytes = NULL; eng->out_i8_bytes = NULL; eng->in_nhwc_elems = NULL;
}

static void free_dma(DcvcRkEngine* eng)
{
    if (!eng) return;
    if (eng->in_mem) {
        for (uint32_t i = 0; i < eng->n_input; i++) {
            if (eng->in_mem[i]) rknn_destroy_mem(eng->ctx, eng->in_mem[i]);
        }
        free(eng->in_mem);
        eng->in_mem = NULL;
    }
    if (eng->out_mem) {
        for (uint32_t i = 0; i < eng->n_output; i++) {
            if (eng->out_mem[i]) rknn_destroy_mem(eng->ctx, eng->out_mem[i]);
        }
        free(eng->out_mem);
        eng->out_mem = NULL;
    }
    free(eng->in_io); eng->in_io = NULL;
    free(eng->out_io); eng->out_io = NULL;
    eng->dma_ready = 0;
}

/* DMA path for FP32: NHWC float in (NPU zero-copy wants NHWC) + float out. */
static DcvcRkStatus setup_dma_fp32(DcvcRkEngine* eng)
{
    eng->in_mem = (rknn_tensor_mem**)calloc(eng->n_input, sizeof(rknn_tensor_mem*));
    eng->out_mem = (rknn_tensor_mem**)calloc(eng->n_output, sizeof(rknn_tensor_mem*));
    eng->in_io = (rknn_tensor_attr*)calloc(eng->n_input, sizeof(rknn_tensor_attr));
    eng->out_io = (rknn_tensor_attr*)calloc(eng->n_output, sizeof(rknn_tensor_attr));
    if (!eng->in_mem || !eng->out_mem || !eng->in_io || !eng->out_io)
        return DCVC_RK_ERR_OOM;

    for (uint32_t i = 0; i < eng->n_input; i++) {
        eng->in_io[i] = eng->in_attr[i];
        eng->in_io[i].type = RKNN_TENSOR_FLOAT32;
        eng->in_io[i].fmt = RKNN_TENSOR_NHWC;
        eng->in_io[i].pass_through = 0;
        uint32_t bytes = eng->in_attr[i].n_elems * (uint32_t)sizeof(float);
        eng->in_mem[i] = rknn_create_mem(eng->ctx, bytes);
        if (!eng->in_mem[i]) return DCVC_RK_ERR_OOM;
        int ret = rknn_set_io_mem(eng->ctx, eng->in_mem[i], &eng->in_io[i]);
        if (ret != RKNN_SUCC) return DCVC_RK_ERR_RKNN;
    }
    for (uint32_t i = 0; i < eng->n_output; i++) {
        eng->out_io[i] = eng->out_attr[i];
        eng->out_io[i].type = RKNN_TENSOR_FLOAT32;
        uint32_t bytes = eng->out_attr[i].n_elems * (uint32_t)sizeof(float);
        eng->out_mem[i] = rknn_create_mem(eng->ctx, bytes);
        if (!eng->out_mem[i]) return DCVC_RK_ERR_OOM;
        int ret = rknn_set_io_mem(eng->ctx, eng->out_mem[i], &eng->out_io[i]);
        if (ret != RKNN_SUCC) return DCVC_RK_ERR_RKNN;
    }
    eng->dma_ready = 1;
    return DCVC_RK_OK;
}

/* DMA path for I8: INT8 NHWC pass-through in + float out (driver dequant). */
static DcvcRkStatus setup_dma_i8(DcvcRkEngine* eng)
{
    eng->in_mem = (rknn_tensor_mem**)calloc(eng->n_input, sizeof(rknn_tensor_mem*));
    eng->out_mem = (rknn_tensor_mem**)calloc(eng->n_output, sizeof(rknn_tensor_mem*));
    eng->in_io = (rknn_tensor_attr*)calloc(eng->n_input, sizeof(rknn_tensor_attr));
    eng->out_io = (rknn_tensor_attr*)calloc(eng->n_output, sizeof(rknn_tensor_attr));
    if (!eng->in_mem || !eng->out_mem || !eng->in_io || !eng->out_io)
        return DCVC_RK_ERR_OOM;

    for (uint32_t i = 0; i < eng->n_input; i++) {
        if (eng->in_attr[i].type != RKNN_TENSOR_INT8 || eng->in_attr[i].fmt != RKNN_TENSOR_NHWC)
            return DCVC_RK_ERR_UNSUPPORTED;
        eng->in_io[i] = eng->in_attr[i];
        eng->in_io[i].pass_through = 1;
        uint32_t bytes = eng->in_attr[i].size_with_stride
                             ? eng->in_attr[i].size_with_stride
                             : eng->in_attr[i].size;
        eng->in_mem[i] = rknn_create_mem(eng->ctx, bytes);
        if (!eng->in_mem[i]) return DCVC_RK_ERR_OOM;
        int ret = rknn_set_io_mem(eng->ctx, eng->in_mem[i], &eng->in_io[i]);
        if (ret != RKNN_SUCC) return DCVC_RK_ERR_RKNN;
    }
    for (uint32_t i = 0; i < eng->n_output; i++) {
        eng->out_io[i] = eng->out_attr[i];
        eng->out_io[i].type = RKNN_TENSOR_FLOAT32;
        uint32_t bytes = eng->out_attr[i].n_elems * (uint32_t)sizeof(float);
        eng->out_mem[i] = rknn_create_mem(eng->ctx, bytes);
        if (!eng->out_mem[i]) return DCVC_RK_ERR_OOM;
        int ret = rknn_set_io_mem(eng->ctx, eng->out_mem[i], &eng->out_io[i]);
        if (ret != RKNN_SUCC) return DCVC_RK_ERR_RKNN;
    }
    eng->dma_ready = 1;
    return DCVC_RK_OK;
}

static DcvcRkStatus alloc_i8_bufs(DcvcRkEngine* eng)
{
    eng->in_i8 = (int8_t**)calloc(eng->n_input, sizeof(int8_t*));
    eng->out_i8 = (int8_t**)calloc(eng->n_output, sizeof(int8_t*));
    eng->in_i8_bytes = (uint32_t*)calloc(eng->n_input, sizeof(uint32_t));
    eng->out_i8_bytes = (uint32_t*)calloc(eng->n_output, sizeof(uint32_t));
    if (!eng->in_i8 || !eng->out_i8 || !eng->in_i8_bytes || !eng->out_i8_bytes)
        return DCVC_RK_ERR_OOM;
    for (uint32_t i = 0; i < eng->n_input; i++) {
        if (eng->in_attr[i].type != RKNN_TENSOR_INT8 || eng->in_attr[i].fmt != RKNN_TENSOR_NHWC)
            return DCVC_RK_ERR_UNSUPPORTED;
        uint32_t n = eng->in_attr[i].size_with_stride ? eng->in_attr[i].size_with_stride
                                                      : eng->in_attr[i].size;
        eng->in_i8_bytes[i] = n;
        eng->in_i8[i] = (int8_t*)malloc(n);
        if (!eng->in_i8[i]) return DCVC_RK_ERR_OOM;
    }
    for (uint32_t i = 0; i < eng->n_output; i++) {
        if (eng->out_attr[i].type != RKNN_TENSOR_INT8 || eng->out_attr[i].fmt != RKNN_TENSOR_NCHW)
            return DCVC_RK_ERR_UNSUPPORTED;
        uint32_t n = eng->out_attr[i].size_with_stride ? eng->out_attr[i].size_with_stride
                                                       : eng->out_attr[i].size;
        eng->out_i8_bytes[i] = n;
        eng->out_i8[i] = (int8_t*)malloc(n);
        if (!eng->out_i8[i]) return DCVC_RK_ERR_OOM;
    }
    return DCVC_RK_OK;
}

static DcvcRkStatus alloc_nhwc_bufs(DcvcRkEngine* eng)
{
    eng->in_nhwc = (float**)calloc(eng->n_input, sizeof(float*));
    eng->in_nhwc_elems = (uint32_t*)calloc(eng->n_input, sizeof(uint32_t));
    eng->out_i8 = (int8_t**)calloc(eng->n_output, sizeof(int8_t*));
    eng->out_i8_bytes = (uint32_t*)calloc(eng->n_output, sizeof(uint32_t));
    if (!eng->in_nhwc || !eng->in_nhwc_elems || !eng->out_i8 || !eng->out_i8_bytes)
        return DCVC_RK_ERR_OOM;
    for (uint32_t i = 0; i < eng->n_input; i++) {
        if (eng->in_attr[i].fmt != RKNN_TENSOR_NHWC) return DCVC_RK_ERR_UNSUPPORTED;
        int N, H, W, C;
        if (nhwc_dims(&eng->in_attr[i], &N, &H, &W, &C) != 0) return DCVC_RK_ERR_UNSUPPORTED;
        eng->in_nhwc_elems[i] = (uint32_t)N * H * W * C;
        eng->in_nhwc[i] = (float*)malloc((size_t)eng->in_nhwc_elems[i] * sizeof(float));
        if (!eng->in_nhwc[i]) return DCVC_RK_ERR_OOM;
    }
    for (uint32_t i = 0; i < eng->n_output; i++) {
        if (eng->out_attr[i].type != RKNN_TENSOR_INT8 || eng->out_attr[i].fmt != RKNN_TENSOR_NCHW)
            return DCVC_RK_ERR_UNSUPPORTED;
        uint32_t n = eng->out_attr[i].size_with_stride ? eng->out_attr[i].size_with_stride
                                                       : eng->out_attr[i].size;
        eng->out_i8_bytes[i] = n;
        eng->out_i8[i] = (int8_t*)malloc(n);
        if (!eng->out_i8[i]) return DCVC_RK_ERR_OOM;
    }
    return DCVC_RK_OK;
}

DcvcRkEngine* dcvc_rk_engine_create(const char* rknn_path, DcvcRkStatus* out_st)
{
    if (out_st) *out_st = DCVC_RK_OK;
    if (!rknn_path) { if (out_st) *out_st = DCVC_RK_ERR_INVALID_ARG; return NULL; }

    uint32_t model_len = 0;
    void* model = load_file(rknn_path, &model_len);
    if (!model) {
        fprintf(stderr, "dcvc_rk: cannot read %s\n", rknn_path);
        if (out_st) *out_st = DCVC_RK_ERR_IO;
        return NULL;
    }

    DcvcRkEngine* eng = (DcvcRkEngine*)calloc(1, sizeof(*eng));
    if (!eng) { free(model); if (out_st) *out_st = DCVC_RK_ERR_OOM; return NULL; }
    eng->last_run_us = eng->last_set_us = eng->last_get_us = eng->last_wall_us = -1;
    eng->io_mode = io_mode_from_env();
    eng->use_dma = dma_from_env();
    if (eng->io_mode == DCVC_IO_I8 || eng->io_mode == DCVC_IO_NHWC || eng->use_dma) {
        if (!getenv("OMP_NUM_THREADS"))
            setenv("OMP_NUM_THREADS", "4", 0);
    }

    int ret = rknn_init(&eng->ctx, model, model_len, 0, NULL);
    free(model);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "dcvc_rk: rknn_init(%s) failed: %d\n", rknn_path, ret);
        free(eng);
        if (out_st) *out_st = DCVC_RK_ERR_RKNN;
        return NULL;
    }

    ret = rknn_set_core_mask(eng->ctx, RKNN_NPU_CORE_0_1_2);
    if (ret != RKNN_SUCC)
        fprintf(stderr, "dcvc_rk: set_core_mask warning: %d\n", ret);

    rknn_input_output_num io;
    memset(&io, 0, sizeof(io));
    ret = rknn_query(eng->ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    if (ret != RKNN_SUCC) {
        rknn_destroy(eng->ctx); free(eng);
        if (out_st) *out_st = DCVC_RK_ERR_RKNN;
        return NULL;
    }
    eng->n_input = io.n_input;
    eng->n_output = io.n_output;
    eng->in_attr = (rknn_tensor_attr*)calloc(eng->n_input, sizeof(rknn_tensor_attr));
    eng->out_attr = (rknn_tensor_attr*)calloc(eng->n_output, sizeof(rknn_tensor_attr));
    if (!eng->in_attr || !eng->out_attr) {
        free(eng->in_attr); free(eng->out_attr);
        rknn_destroy(eng->ctx); free(eng);
        if (out_st) *out_st = DCVC_RK_ERR_OOM;
        return NULL;
    }
    for (uint32_t i = 0; i < eng->n_input; i++) {
        eng->in_attr[i].index = i;
        rknn_query(eng->ctx, RKNN_QUERY_INPUT_ATTR, &eng->in_attr[i], sizeof(rknn_tensor_attr));
    }
    for (uint32_t i = 0; i < eng->n_output; i++) {
        eng->out_attr[i].index = i;
        rknn_query(eng->ctx, RKNN_QUERY_OUTPUT_ATTR, &eng->out_attr[i], sizeof(rknn_tensor_attr));
    }

    /* DMA zero-copy: only win for INT8 NHWC pass-through.
     * FP32+DMA (NHWC float) inflates PERF_RUN badly on RK3588 — skip unless
     * DCVC_RKNN_DMA=force. */
    {
        const char* dma_env = getenv("DCVC_RKNN_DMA");
        int force = dma_env && !strcmp(dma_env, "force");
        int want_dma = eng->use_dma &&
                       (eng->io_mode == DCVC_IO_I8 ||
                        (force && eng->io_mode == DCVC_IO_FP32));
        if (want_dma) {
            DcvcRkStatus st = (eng->io_mode == DCVC_IO_I8) ? setup_dma_i8(eng)
                                                           : setup_dma_fp32(eng);
            if (st != DCVC_RK_OK) {
                fprintf(stderr, "dcvc_rk: DMA setup failed for %s (%s), fallback inputs_set\n",
                        rknn_path, io_mode_name(eng->io_mode));
                free_dma(eng);
                eng->use_dma = 0;
            }
        } else {
            eng->use_dma = 0;
        }
    }

    if (!eng->dma_ready) {
        if (eng->io_mode == DCVC_IO_I8) {
            DcvcRkStatus st = alloc_i8_bufs(eng);
            if (st != DCVC_RK_OK) {
                fprintf(stderr, "dcvc_rk: I8 IO unsupported for %s, fallback FP32\n", rknn_path);
                free_scratch(eng);
                eng->io_mode = DCVC_IO_FP32;
            }
        } else if (eng->io_mode == DCVC_IO_NHWC || eng->io_mode == DCVC_IO_I8OUT) {
            DcvcRkStatus st;
            if (eng->io_mode == DCVC_IO_NHWC)
                st = alloc_nhwc_bufs(eng);
            else {
                eng->out_i8 = (int8_t**)calloc(eng->n_output, sizeof(int8_t*));
                eng->out_i8_bytes = (uint32_t*)calloc(eng->n_output, sizeof(uint32_t));
                st = (!eng->out_i8 || !eng->out_i8_bytes) ? DCVC_RK_ERR_OOM : DCVC_RK_OK;
                for (uint32_t i = 0; st == DCVC_RK_OK && i < eng->n_output; i++) {
                    if (eng->out_attr[i].type != RKNN_TENSOR_INT8 ||
                        eng->out_attr[i].fmt != RKNN_TENSOR_NCHW) {
                        st = DCVC_RK_ERR_UNSUPPORTED; break;
                    }
                    uint32_t n = eng->out_attr[i].size_with_stride
                                     ? eng->out_attr[i].size_with_stride
                                     : eng->out_attr[i].size;
                    eng->out_i8_bytes[i] = n;
                    eng->out_i8[i] = (int8_t*)malloc(n);
                    if (!eng->out_i8[i]) st = DCVC_RK_ERR_OOM;
                }
            }
            if (st != DCVC_RK_OK) {
                fprintf(stderr, "dcvc_rk: %s IO unsupported for %s, fallback FP32\n",
                        io_mode_name(eng->io_mode), rknn_path);
                free_scratch(eng);
                eng->io_mode = DCVC_IO_FP32;
            }
        }
    }

    {
        static int once;
        if (!once) {
            fprintf(stderr, "dcvc_rk: DCVC_RKNN_IO=%s DMA=%d\n",
                    io_mode_name(eng->io_mode), eng->dma_ready);
            once = 1;
        }
    }
    return eng;
}

void dcvc_rk_engine_destroy(DcvcRkEngine* eng)
{
    if (!eng) return;
    free_scratch(eng);
    free_dma(eng);
    if (eng->ctx) rknn_destroy(eng->ctx);
    free(eng->in_attr);
    free(eng->out_attr);
    free(eng);
}

int64_t dcvc_rk_engine_last_run_us(DcvcRkEngine* eng)
{
    return eng ? eng->last_run_us : -1;
}
int64_t dcvc_rk_engine_last_set_us(DcvcRkEngine* eng)
{
    return eng ? eng->last_set_us : -1;
}
int64_t dcvc_rk_engine_last_get_us(DcvcRkEngine* eng)
{
    return eng ? eng->last_get_us : -1;
}
int64_t dcvc_rk_engine_last_wall_us(DcvcRkEngine* eng)
{
    return eng ? eng->last_wall_us : -1;
}

static void query_perf(DcvcRkEngine* eng)
{
    rknn_perf_run perf;
    memset(&perf, 0, sizeof(perf));
    if (rknn_query(eng->ctx, RKNN_QUERY_PERF_RUN, &perf, sizeof(perf)) == RKNN_SUCC)
        eng->last_run_us = (int64_t)perf.run_duration;
    else
        eng->last_run_us = -1;
}

static DcvcRkStatus run_fp32_dma(DcvcRkEngine* eng,
                                 DcvcRkTensorView* inputs, int n_in,
                                 DcvcRkTensorView* outputs, int n_out)
{
    double t0 = dcvc_rk_now_ms();
    double t_set0 = dcvc_rk_now_ms();
    for (int i = 0; i < n_in; i++) {
        int N, H, W, C;
        /* Prefer dims from model NHWC attr; fall back to view. */
        if (eng->in_attr[i].fmt == RKNN_TENSOR_NHWC) {
            if (nhwc_dims(&eng->in_attr[i], &N, &H, &W, &C) != 0)
                return DCVC_RK_ERR_UNSUPPORTED;
        } else if (nchw_dims(&eng->in_attr[i], &N, &C, &H, &W) != 0) {
            N = inputs[i].n; C = inputs[i].c; H = inputs[i].h; W = inputs[i].w;
        }
        if (inputs[i].c != C || inputs[i].h != H || inputs[i].w != W)
            return DCVC_RK_ERR_INVALID_ARG;
        nchw_to_nhwc_f32((const float*)inputs[i].data,
                         (float*)eng->in_mem[i]->virt_addr, N, C, H, W);
    }
    eng->last_set_us = (int64_t)((dcvc_rk_now_ms() - t_set0) * 1000.0);

    int ret = rknn_run(eng->ctx, NULL);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "dcvc_rk: rknn_run(dma fp32) failed: %d\n", ret);
        return DCVC_RK_ERR_RKNN;
    }
    query_perf(eng);

    double t_get0 = dcvc_rk_now_ms();
    for (int i = 0; i < n_out; i++) {
        size_t elems = (size_t)outputs[i].n * outputs[i].c * outputs[i].h * outputs[i].w;
        if (elems != eng->out_attr[i].n_elems) {
            /* Allow view to match n_elems from attr. */
            elems = eng->out_attr[i].n_elems;
        }
        memcpy(outputs[i].data, eng->out_mem[i]->virt_addr, elems * sizeof(float));
    }
    eng->last_get_us = (int64_t)((dcvc_rk_now_ms() - t_get0) * 1000.0);
    eng->last_wall_us = (int64_t)((dcvc_rk_now_ms() - t0) * 1000.0);
    return DCVC_RK_OK;
}

static DcvcRkStatus run_i8_dma(DcvcRkEngine* eng,
                               DcvcRkTensorView* inputs, int n_in,
                               DcvcRkTensorView* outputs, int n_out)
{
    double t0 = dcvc_rk_now_ms();
    double t_set0 = dcvc_rk_now_ms();
    for (int i = 0; i < n_in; i++) {
        int N, H, W, C;
        if (nhwc_dims(&eng->in_attr[i], &N, &H, &W, &C) != 0)
            return DCVC_RK_ERR_UNSUPPORTED;
        if (inputs[i].c != C || inputs[i].h != H || inputs[i].w != W)
            return DCVC_RK_ERR_INVALID_ARG;
        quant_nchw_f32_to_nhwc_i8((const float*)inputs[i].data,
                                  (int8_t*)eng->in_mem[i]->virt_addr,
                                  N, C, H, W, eng->in_attr[i].scale, eng->in_attr[i].zp);
    }
    eng->last_set_us = (int64_t)((dcvc_rk_now_ms() - t_set0) * 1000.0);

    int ret = rknn_run(eng->ctx, NULL);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "dcvc_rk: rknn_run(dma i8) failed: %d\n", ret);
        return DCVC_RK_ERR_RKNN;
    }
    query_perf(eng);

    double t_get0 = dcvc_rk_now_ms();
    for (int i = 0; i < n_out; i++) {
        size_t elems = eng->out_attr[i].n_elems;
        memcpy(outputs[i].data, eng->out_mem[i]->virt_addr, elems * sizeof(float));
    }
    eng->last_get_us = (int64_t)((dcvc_rk_now_ms() - t_get0) * 1000.0);
    eng->last_wall_us = (int64_t)((dcvc_rk_now_ms() - t0) * 1000.0);
    (void)n_out;
    return DCVC_RK_OK;
}

static DcvcRkStatus run_fp32(DcvcRkEngine* eng,
                             DcvcRkTensorView* inputs, int n_in,
                             DcvcRkTensorView* outputs, int n_out)
{
    if (eng->dma_ready && eng->io_mode == DCVC_IO_FP32)
        return run_fp32_dma(eng, inputs, n_in, outputs, n_out);

    double t0 = dcvc_rk_now_ms();
    rknn_input* rin = (rknn_input*)calloc((size_t)n_in, sizeof(rknn_input));
    rknn_output* rout = (rknn_output*)calloc((size_t)n_out, sizeof(rknn_output));
    if (!rin || !rout) { free(rin); free(rout); return DCVC_RK_ERR_OOM; }

    for (int i = 0; i < n_in; i++) {
        size_t elems = (size_t)inputs[i].n * inputs[i].c * inputs[i].h * inputs[i].w;
        rin[i].index = i;
        rin[i].buf = inputs[i].data;
        rin[i].size = (uint32_t)(elems * sizeof(float));
        rin[i].pass_through = 0;
        rin[i].type = RKNN_TENSOR_FLOAT32;
        rin[i].fmt = RKNN_TENSOR_NCHW;
    }
    double t_set0 = dcvc_rk_now_ms();
    int ret = rknn_inputs_set(eng->ctx, (uint32_t)n_in, rin);
    eng->last_set_us = (int64_t)((dcvc_rk_now_ms() - t_set0) * 1000.0);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "dcvc_rk: inputs_set failed: %d\n", ret);
        free(rin); free(rout);
        return DCVC_RK_ERR_RKNN;
    }

    ret = rknn_run(eng->ctx, NULL);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "dcvc_rk: rknn_run failed: %d\n", ret);
        free(rin); free(rout);
        return DCVC_RK_ERR_RKNN;
    }
    query_perf(eng);

    for (int i = 0; i < n_out; i++) {
        size_t elems = (size_t)outputs[i].n * outputs[i].c * outputs[i].h * outputs[i].w;
        rout[i].want_float = 1;
        rout[i].is_prealloc = 1;
        rout[i].index = i;
        rout[i].buf = outputs[i].data;
        rout[i].size = (uint32_t)(elems * sizeof(float));
    }
    double t_get0 = dcvc_rk_now_ms();
    ret = rknn_outputs_get(eng->ctx, (uint32_t)n_out, rout, NULL);
    eng->last_get_us = (int64_t)((dcvc_rk_now_ms() - t_get0) * 1000.0);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "dcvc_rk: outputs_get failed: %d\n", ret);
        free(rin); free(rout);
        return DCVC_RK_ERR_RKNN;
    }
    rknn_outputs_release(eng->ctx, (uint32_t)n_out, rout);
    free(rin);
    free(rout);
    eng->last_wall_us = (int64_t)((dcvc_rk_now_ms() - t0) * 1000.0);
    return DCVC_RK_OK;
}

static DcvcRkStatus run_i8(DcvcRkEngine* eng,
                           DcvcRkTensorView* inputs, int n_in,
                           DcvcRkTensorView* outputs, int n_out)
{
    if (eng->dma_ready && eng->io_mode == DCVC_IO_I8)
        return run_i8_dma(eng, inputs, n_in, outputs, n_out);

    double t0 = dcvc_rk_now_ms();
    rknn_input* rin = (rknn_input*)calloc((size_t)n_in, sizeof(rknn_input));
    rknn_output* rout = (rknn_output*)calloc((size_t)n_out, sizeof(rknn_output));
    if (!rin || !rout) { free(rin); free(rout); return DCVC_RK_ERR_OOM; }

    double t_set0 = dcvc_rk_now_ms();
    for (int i = 0; i < n_in; i++) {
        int N, H, W, C;
        if (nhwc_dims(&eng->in_attr[i], &N, &H, &W, &C) != 0) {
            free(rin); free(rout); return DCVC_RK_ERR_UNSUPPORTED;
        }
        if (inputs[i].c != C || inputs[i].h != H || inputs[i].w != W) {
            free(rin); free(rout);
            return DCVC_RK_ERR_INVALID_ARG;
        }
        quant_nchw_f32_to_nhwc_i8((const float*)inputs[i].data, eng->in_i8[i],
                                  N, C, H, W, eng->in_attr[i].scale, eng->in_attr[i].zp);
        rin[i].index = i;
        rin[i].buf = eng->in_i8[i];
        rin[i].size = eng->in_attr[i].size;
        rin[i].pass_through = 1;
        rin[i].type = RKNN_TENSOR_INT8;
        rin[i].fmt = RKNN_TENSOR_NHWC;
    }
    int ret = rknn_inputs_set(eng->ctx, (uint32_t)n_in, rin);
    eng->last_set_us = (int64_t)((dcvc_rk_now_ms() - t_set0) * 1000.0);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "dcvc_rk: inputs_set(i8) failed: %d\n", ret);
        free(rin); free(rout);
        return DCVC_RK_ERR_RKNN;
    }

    ret = rknn_run(eng->ctx, NULL);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "dcvc_rk: rknn_run failed: %d\n", ret);
        free(rin); free(rout);
        return DCVC_RK_ERR_RKNN;
    }
    query_perf(eng);

    double t_get0 = dcvc_rk_now_ms();
    for (int i = 0; i < n_out; i++) {
        rout[i].want_float = 0;
        rout[i].is_prealloc = 1;
        rout[i].index = i;
        rout[i].buf = eng->out_i8[i];
        rout[i].size = eng->out_i8_bytes[i];
    }
    ret = rknn_outputs_get(eng->ctx, (uint32_t)n_out, rout, NULL);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "dcvc_rk: outputs_get(i8) failed: %d\n", ret);
        free(rin); free(rout);
        return DCVC_RK_ERR_RKNN;
    }
    for (int i = 0; i < n_out; i++) {
        int N, C, H, W;
        if (nchw_dims(&eng->out_attr[i], &N, &C, &H, &W) != 0) {
            rknn_outputs_release(eng->ctx, (uint32_t)n_out, rout);
            free(rin); free(rout);
            return DCVC_RK_ERR_UNSUPPORTED;
        }
        int w_stride = (int)eng->out_attr[i].w_stride;
        if (w_stride <= 0) w_stride = W;
        dequant_nchw_i8_to_f32(eng->out_i8[i], (float*)outputs[i].data,
                               N, C, H, W, w_stride,
                               eng->out_attr[i].scale, eng->out_attr[i].zp);
    }
    eng->last_get_us = (int64_t)((dcvc_rk_now_ms() - t_get0) * 1000.0);
    rknn_outputs_release(eng->ctx, (uint32_t)n_out, rout);
    free(rin);
    free(rout);
    eng->last_wall_us = (int64_t)((dcvc_rk_now_ms() - t0) * 1000.0);
    return DCVC_RK_OK;
}

static DcvcRkStatus get_i8_dequant(DcvcRkEngine* eng, DcvcRkTensorView* outputs, int n_out,
                                   rknn_output* rout)
{
    for (int i = 0; i < n_out; i++) {
        rout[i].want_float = 0;
        rout[i].is_prealloc = 1;
        rout[i].index = i;
        rout[i].buf = eng->out_i8[i];
        rout[i].size = eng->out_i8_bytes[i];
    }
    int ret = rknn_outputs_get(eng->ctx, (uint32_t)n_out, rout, NULL);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "dcvc_rk: outputs_get(i8) failed: %d\n", ret);
        return DCVC_RK_ERR_RKNN;
    }
    for (int i = 0; i < n_out; i++) {
        int N, C, H, W;
        if (nchw_dims(&eng->out_attr[i], &N, &C, &H, &W) != 0)
            return DCVC_RK_ERR_UNSUPPORTED;
        int w_stride = (int)eng->out_attr[i].w_stride;
        if (w_stride <= 0) w_stride = W;
        dequant_nchw_i8_to_f32(eng->out_i8[i], (float*)outputs[i].data,
                               N, C, H, W, w_stride,
                               eng->out_attr[i].scale, eng->out_attr[i].zp);
    }
    return DCVC_RK_OK;
}

static DcvcRkStatus run_nhwc(DcvcRkEngine* eng,
                             DcvcRkTensorView* inputs, int n_in,
                             DcvcRkTensorView* outputs, int n_out)
{
    double t0 = dcvc_rk_now_ms();
    rknn_input* rin = (rknn_input*)calloc((size_t)n_in, sizeof(rknn_input));
    rknn_output* rout = (rknn_output*)calloc((size_t)n_out, sizeof(rknn_output));
    if (!rin || !rout) { free(rin); free(rout); return DCVC_RK_ERR_OOM; }

    double t_set0 = dcvc_rk_now_ms();
    for (int i = 0; i < n_in; i++) {
        int N, H, W, C;
        if (nhwc_dims(&eng->in_attr[i], &N, &H, &W, &C) != 0) {
            free(rin); free(rout); return DCVC_RK_ERR_UNSUPPORTED;
        }
        nchw_to_nhwc_f32((const float*)inputs[i].data, eng->in_nhwc[i], N, C, H, W);
        rin[i].index = i;
        rin[i].buf = eng->in_nhwc[i];
        rin[i].size = eng->in_nhwc_elems[i] * (uint32_t)sizeof(float);
        rin[i].pass_through = 0;
        rin[i].type = RKNN_TENSOR_FLOAT32;
        rin[i].fmt = RKNN_TENSOR_NHWC;
    }
    int ret = rknn_inputs_set(eng->ctx, (uint32_t)n_in, rin);
    eng->last_set_us = (int64_t)((dcvc_rk_now_ms() - t_set0) * 1000.0);
    if (ret != RKNN_SUCC) {
        free(rin); free(rout);
        return DCVC_RK_ERR_RKNN;
    }

    ret = rknn_run(eng->ctx, NULL);
    if (ret != RKNN_SUCC) {
        free(rin); free(rout);
        return DCVC_RK_ERR_RKNN;
    }
    query_perf(eng);

    double t_get0 = dcvc_rk_now_ms();
    DcvcRkStatus st = get_i8_dequant(eng, outputs, n_out, rout);
    eng->last_get_us = (int64_t)((dcvc_rk_now_ms() - t_get0) * 1000.0);
    if (st == DCVC_RK_OK)
        rknn_outputs_release(eng->ctx, (uint32_t)n_out, rout);
    free(rin); free(rout);
    eng->last_wall_us = (int64_t)((dcvc_rk_now_ms() - t0) * 1000.0);
    return st;
}

static DcvcRkStatus run_i8out(DcvcRkEngine* eng,
                              DcvcRkTensorView* inputs, int n_in,
                              DcvcRkTensorView* outputs, int n_out)
{
    double t0 = dcvc_rk_now_ms();
    rknn_input* rin = (rknn_input*)calloc((size_t)n_in, sizeof(rknn_input));
    rknn_output* rout = (rknn_output*)calloc((size_t)n_out, sizeof(rknn_output));
    if (!rin || !rout) { free(rin); free(rout); return DCVC_RK_ERR_OOM; }

    for (int i = 0; i < n_in; i++) {
        size_t elems = (size_t)inputs[i].n * inputs[i].c * inputs[i].h * inputs[i].w;
        rin[i].index = i;
        rin[i].buf = inputs[i].data;
        rin[i].size = (uint32_t)(elems * sizeof(float));
        rin[i].pass_through = 0;
        rin[i].type = RKNN_TENSOR_FLOAT32;
        rin[i].fmt = RKNN_TENSOR_NCHW;
    }
    double t_set0 = dcvc_rk_now_ms();
    int ret = rknn_inputs_set(eng->ctx, (uint32_t)n_in, rin);
    eng->last_set_us = (int64_t)((dcvc_rk_now_ms() - t_set0) * 1000.0);
    if (ret != RKNN_SUCC) {
        free(rin); free(rout);
        return DCVC_RK_ERR_RKNN;
    }

    ret = rknn_run(eng->ctx, NULL);
    if (ret != RKNN_SUCC) {
        free(rin); free(rout);
        return DCVC_RK_ERR_RKNN;
    }
    query_perf(eng);

    double t_get0 = dcvc_rk_now_ms();
    DcvcRkStatus st = get_i8_dequant(eng, outputs, n_out, rout);
    eng->last_get_us = (int64_t)((dcvc_rk_now_ms() - t_get0) * 1000.0);
    if (st == DCVC_RK_OK)
        rknn_outputs_release(eng->ctx, (uint32_t)n_out, rout);
    free(rin); free(rout);
    eng->last_wall_us = (int64_t)((dcvc_rk_now_ms() - t0) * 1000.0);
    return st;
}

DcvcRkStatus dcvc_rk_engine_run(DcvcRkEngine* eng,
                                DcvcRkTensorView* inputs, int n_in,
                                DcvcRkTensorView* outputs, int n_out)
{
    if (!eng || !inputs || !outputs || n_in <= 0 || n_out <= 0)
        return DCVC_RK_ERR_INVALID_ARG;
    if ((uint32_t)n_in != eng->n_input || (uint32_t)n_out != eng->n_output)
        return DCVC_RK_ERR_INVALID_ARG;
    switch (eng->io_mode) {
    case DCVC_IO_I8: return run_i8(eng, inputs, n_in, outputs, n_out);
    case DCVC_IO_NHWC: return run_nhwc(eng, inputs, n_in, outputs, n_out);
    case DCVC_IO_I8OUT: return run_i8out(eng, inputs, n_in, outputs, n_out);
    default: return run_fp32(eng, inputs, n_in, outputs, n_out);
    }
}

DcvcRkStatus dcvc_rk_engine_in_quant(const DcvcRkEngine* eng, int idx, DcvcRkQuant* q)
{
    if (!eng || !q || idx < 0 || (uint32_t)idx >= eng->n_input)
        return DCVC_RK_ERR_INVALID_ARG;
    const rknn_tensor_attr* a = &eng->in_attr[idx];
    q->scale = a->scale;
    q->zp = a->zp;
    q->w_stride = 0;
    if (a->fmt == RKNN_TENSOR_NHWC && a->n_dims >= 4) {
        q->n = (int)a->dims[0];
        q->h = (int)a->dims[1];
        q->w = (int)a->dims[2];
        q->c = (int)a->dims[3];
    } else if (a->fmt == RKNN_TENSOR_NCHW && a->n_dims >= 4) {
        q->n = (int)a->dims[0];
        q->c = (int)a->dims[1];
        q->h = (int)a->dims[2];
        q->w = (int)a->dims[3];
    } else {
        return DCVC_RK_ERR_UNSUPPORTED;
    }
    return DCVC_RK_OK;
}

DcvcRkStatus dcvc_rk_engine_out_quant(const DcvcRkEngine* eng, int idx, DcvcRkQuant* q)
{
    if (!eng || !q || idx < 0 || (uint32_t)idx >= eng->n_output)
        return DCVC_RK_ERR_INVALID_ARG;
    const rknn_tensor_attr* a = &eng->out_attr[idx];
    q->scale = a->scale;
    q->zp = a->zp;
    q->w_stride = (int)a->w_stride;
    if (a->fmt != RKNN_TENSOR_NCHW || a->n_dims < 4)
        return DCVC_RK_ERR_UNSUPPORTED;
    q->n = (int)a->dims[0];
    q->c = (int)a->dims[1];
    q->h = (int)a->dims[2];
    q->w = (int)a->dims[3];
    return DCVC_RK_OK;
}

uint32_t dcvc_rk_engine_out_buf_bytes(const DcvcRkEngine* eng, int idx)
{
    if (!eng || idx < 0 || (uint32_t)idx >= eng->n_output) return 0;
    const rknn_tensor_attr* a = &eng->out_attr[idx];
    uint32_t n = a->size_with_stride ? a->size_with_stride : a->size;
    return n ? n : a->n_elems;
}

/* INT8 NCHW in → CPU NHWC + pass_through; INT8 NCHW out (no host float hop). */
DcvcRkStatus dcvc_rk_engine_run_i8(DcvcRkEngine* eng,
                                   DcvcRkI8View* inputs, int n_in,
                                   DcvcRkI8View* outputs, int n_out)
{
    if (!eng || !inputs || !outputs || n_in <= 0 || n_out <= 0)
        return DCVC_RK_ERR_INVALID_ARG;
    if ((uint32_t)n_in != eng->n_input || (uint32_t)n_out != eng->n_output)
        return DCVC_RK_ERR_INVALID_ARG;

    DcvcRkStatus st = ensure_in_i8_scratch(eng);
    if (st != DCVC_RK_OK) return st;

    double t0 = dcvc_rk_now_ms();
    rknn_input* rin = (rknn_input*)calloc((size_t)n_in, sizeof(rknn_input));
    rknn_output* rout = (rknn_output*)calloc((size_t)n_out, sizeof(rknn_output));
    if (!rin || !rout) { free(rin); free(rout); return DCVC_RK_ERR_OOM; }

    double t_set0 = dcvc_rk_now_ms();
    for (int i = 0; i < n_in; i++) {
        int N, H, W, C;
        if (nhwc_dims(&eng->in_attr[i], &N, &H, &W, &C) != 0) {
            free(rin); free(rout); return DCVC_RK_ERR_UNSUPPORTED;
        }
        if (inputs[i].c != C || inputs[i].h != H || inputs[i].w != W) {
            fprintf(stderr, "dcvc_rk: run_i8 in[%d] shape mismatch view=%dx%dx%d attr=%dx%dx%d\n",
                    i, inputs[i].c, inputs[i].h, inputs[i].w, C, H, W);
            free(rin); free(rout);
            return DCVC_RK_ERR_INVALID_ARG;
        }
        nchw_i8_to_nhwc_i8(inputs[i].data, eng->in_i8[i], N, C, H, W);
        rin[i].index = i;
        rin[i].buf = eng->in_i8[i];
        rin[i].size = eng->in_attr[i].size;
        rin[i].pass_through = 1;
        rin[i].type = RKNN_TENSOR_INT8;
        rin[i].fmt = RKNN_TENSOR_NHWC;
    }
    int ret = rknn_inputs_set(eng->ctx, (uint32_t)n_in, rin);
    eng->last_set_us = (int64_t)((dcvc_rk_now_ms() - t_set0) * 1000.0);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "dcvc_rk: inputs_set(run_i8) failed: %d\n", ret);
        free(rin); free(rout);
        return DCVC_RK_ERR_RKNN;
    }

    ret = rknn_run(eng->ctx, NULL);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "dcvc_rk: rknn_run(run_i8) failed: %d\n", ret);
        free(rin); free(rout);
        return DCVC_RK_ERR_RKNN;
    }
    query_perf(eng);

    double t_get0 = dcvc_rk_now_ms();
    for (int i = 0; i < n_out; i++) {
        uint32_t nbytes = dcvc_rk_engine_out_buf_bytes(eng, i);
        rout[i].want_float = 0;
        rout[i].is_prealloc = 1;
        rout[i].index = i;
        rout[i].buf = outputs[i].data;
        rout[i].size = nbytes;
    }
    ret = rknn_outputs_get(eng->ctx, (uint32_t)n_out, rout, NULL);
    eng->last_get_us = (int64_t)((dcvc_rk_now_ms() - t_get0) * 1000.0);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "dcvc_rk: outputs_get(run_i8) failed: %d\n", ret);
        free(rin); free(rout);
        return DCVC_RK_ERR_RKNN;
    }
    rknn_outputs_release(eng->ctx, (uint32_t)n_out, rout);
    free(rin);
    free(rout);
    eng->last_wall_us = (int64_t)((dcvc_rk_now_ms() - t0) * 1000.0);
    return DCVC_RK_OK;
}
