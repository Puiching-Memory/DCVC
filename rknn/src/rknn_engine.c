#include "dcvc_rk/rknn_engine.h"
#include "dcvc_rk/profile.h"

#include <rknn_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct DcvcRkEngine {
    rknn_context ctx;
    uint32_t n_input;
    uint32_t n_output;
    rknn_tensor_attr* in_attr;
    rknn_tensor_attr* out_attr;
    int64_t last_run_us;
    int64_t last_set_us;
    int64_t last_get_us;
    int64_t last_wall_us;
};

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

    /* flags=0: never COLLECT_PERF (segfault risk on large feature maps). */
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
    return eng;
}

void dcvc_rk_engine_destroy(DcvcRkEngine* eng)
{
    if (!eng) return;
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

DcvcRkStatus dcvc_rk_engine_run(DcvcRkEngine* eng,
                                DcvcRkTensorView* inputs, int n_in,
                                DcvcRkTensorView* outputs, int n_out)
{
    if (!eng || !inputs || !outputs || n_in <= 0 || n_out <= 0)
        return DCVC_RK_ERR_INVALID_ARG;
    if ((uint32_t)n_in != eng->n_input || (uint32_t)n_out != eng->n_output)
        return DCVC_RK_ERR_INVALID_ARG;

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

    rknn_perf_run perf;
    memset(&perf, 0, sizeof(perf));
    if (rknn_query(eng->ctx, RKNN_QUERY_PERF_RUN, &perf, sizeof(perf)) == RKNN_SUCC)
        eng->last_run_us = (int64_t)perf.run_duration;
    else
        eng->last_run_us = -1;

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
