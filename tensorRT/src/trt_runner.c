#include "dcvc_trt_runner.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dcvc_rt_internal.h"
#include "trt_engine.h"

struct DcvcTrtRunner {
    char asset_dir[512];
    char plugin_dir[512];
    int device_id;
    int engine_present[DCVC_ENG_COUNT];
    /* Lazily loaded engine contexts */
    DcvcTrtEngine* engines[DCVC_ENG_COUNT];
};

static const char* k_engine_names[DCVC_ENG_COUNT] = {
    "intra_analysis",
    "intra_hyper_enc",
    "y_prior_fusion",      /* was intra_prior_fusion — actual engine file name */
    "y_spatial_prior",     /* was intra_spatial_prior — actual engine file name */
    "intra_synthesis",
    "inter_feature",
    "inter_analysis",
    "inter_hyper",
    "y_prior_fusion",      /* inter uses same prior_fusion engine */
    "y_spatial_prior",     /* inter uses same spatial_prior engine */
    "inter_synthesis",
    "hyper_dec",           /* shared hyper_dec engine */
};

static int file_exists(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static void build_engine_path(const DcvcTrtRunner* r, int id, char* out, size_t outsz)
{
    snprintf(out, outsz, "%s/engines/%s.engine", r->asset_dir, k_engine_names[id]);
}

DcvcTrtRunner* dcvc_trt_runner_create(const char* asset_dir, int device_id, DcvcRtStatus* st)
{
    DcvcTrtRunner* r = (DcvcTrtRunner*)calloc(1, sizeof(DcvcTrtRunner));
    if (!r) {
        if (st) *st = DCVC_RT_ERR_OOM;
        return NULL;
    }
    if (asset_dir) {
        strncpy(r->asset_dir, asset_dir, sizeof(r->asset_dir) - 1);
    } else {
        strncpy(r->asset_dir, "tensorRT/assets", sizeof(r->asset_dir) - 1);
    }
    r->device_id = device_id;

    /* Plugin directory: look for .so next to engines or in build dir */
    snprintf(r->plugin_dir, sizeof(r->plugin_dir), "%s/../build/plugin_demo", r->asset_dir);

    for (int i = 0; i < DCVC_ENG_COUNT; i++) {
        char path[640];
        build_engine_path(r, i, path, sizeof(path));
        r->engine_present[i] = file_exists(path);
    }

    if (st) *st = DCVC_RT_OK;
    return r;
}

void dcvc_trt_runner_destroy(DcvcTrtRunner* r)
{
    if (r) {
        for (int i = 0; i < DCVC_ENG_COUNT; i++)
            if (r->engines[i]) dcvc_trt_engine_destroy(r->engines[i]);
        free(r);
    }
}

int dcvc_trt_runner_has_engine(const DcvcTrtRunner* r, DcvcEngineId id)
{
    if (!r || id < 0 || id >= DCVC_ENG_COUNT) return 0;
    return r->engine_present[id];
}

static DcvcTrtEngine* ensure_engine(DcvcTrtRunner* r, DcvcEngineId id, DcvcRtStatus* st)
{
    if (r->engines[id]) return r->engines[id];

    char path[640];
    build_engine_path(r, id, path, sizeof(path));
    r->engines[id] = dcvc_trt_engine_load(path, r->plugin_dir, st);
    return r->engines[id];
}

struct DcvcTrtEngine* dcvc_trt_runner_get_engine(DcvcTrtRunner* r, DcvcEngineId id,
                                                   DcvcRtStatus* st)
{
    if (!r || id < 0 || id >= DCVC_ENG_COUNT) { if (st) *st = DCVC_RT_ERR_INVALID_ARG; return NULL; }
    return ensure_engine(r, id, st);
}

DcvcRtStatus dcvc_trt_runner_execute(DcvcTrtRunner* r, DcvcEngineId id,
                                     DcvcTensorView* inputs, int n_in,
                                     DcvcTensorView* outputs, int n_out,
                                     void* cuda_stream)
{
    if (!r || !inputs || !outputs) return DCVC_RT_ERR_INVALID_ARG;
    if (id < 0 || id >= DCVC_ENG_COUNT) return DCVC_RT_ERR_INVALID_ARG;
    if (!dcvc_trt_runner_has_engine(r, id)) return DCVC_RT_ERR_NO_ENGINE;

    DcvcRtStatus st;
    DcvcTrtEngine* eng = ensure_engine(r, id, &st);
    if (!eng) return st;

    /* Set input shapes */
    for (int i = 0; i < n_in; i++) {
        int32_t dims[8] = {inputs[i].n, inputs[i].c, inputs[i].h, inputs[i].w};
        /* Find the input tensor name by matching mode */
        int nio = dcvc_trt_engine_num_io(eng);
        int input_idx = 0;
        for (int j = 0; j < nio; j++) {
            if (dcvc_trt_engine_is_input(eng, j)) {
                if (input_idx == i) {
                    const char* name = dcvc_trt_engine_tensor_name(eng, j);
                    st = dcvc_trt_engine_set_shape(eng, name, dims, 4);
                    if (st != DCVC_RT_OK) return st;
                    st = dcvc_trt_engine_set_addr(eng, name, inputs[i].device_ptr);
                    if (st != DCVC_RT_OK) return st;
                    break;
                }
                input_idx++;
            }
        }
    }

    /* Set output addresses (shapes inferred from input) */
    int nio = dcvc_trt_engine_num_io(eng);
    int out_idx = 0;
    for (int j = 0; j < nio; j++) {
        if (!dcvc_trt_engine_is_input(eng, j)) {
            if (out_idx < n_out) {
                const char* name = dcvc_trt_engine_tensor_name(eng, j);
                st = dcvc_trt_engine_set_addr(eng, name, outputs[out_idx].device_ptr);
                if (st != DCVC_RT_OK) return st;

                /* Propagate output shape back to the view */
                int32_t dims[8]; int ndims;
                if (dcvc_trt_engine_get_shape(eng, name, dims, &ndims, 8) == DCVC_RT_OK && ndims >= 4) {
                    outputs[out_idx].n = dims[0];
                    outputs[out_idx].c = dims[1];
                    outputs[out_idx].h = dims[2];
                    outputs[out_idx].w = dims[3];
                }
            }
            out_idx++;
        }
    }

    return dcvc_trt_engine_execute(eng, cuda_stream);
}

DcvcRtStatus dcvc_trt_load_qp_scale(DcvcTrtRunner* r, const char* bank_name, int qp,
                                    uint16_t* out_fp16, int channels)
{
    if (!r || !bank_name || !out_fp16 || channels <= 0) return DCVC_RT_ERR_INVALID_ARG;
    if (qp < 0 || qp >= DCVC_RT_QP_NUM + 16) return DCVC_RT_ERR_INVALID_ARG;

    char path[640];
    snprintf(path, sizeof(path), "%s/qp/%s.bin", r->asset_dir, bank_name);
    FILE* f = fopen(path, "rb");
    if (!f) return DCVC_RT_ERR_IO;

    /* Layout: int32 qp_num, int32 channels, then qp_num * channels FP16 */
    int32_t qp_num = 0, ch = 0;
    if (fread(&qp_num, 4, 1, f) != 1 || fread(&ch, 4, 1, f) != 1) {
        fclose(f);
        return DCVC_RT_ERR_IO;
    }
    if (ch != channels || qp < 0 || qp >= qp_num) {
        fclose(f);
        return DCVC_RT_ERR_INVALID_ARG;
    }
    if (fseek(f, (long)(8 + (size_t)qp * channels * 2), SEEK_SET) != 0) {
        fclose(f);
        return DCVC_RT_ERR_IO;
    }
    size_t got = fread(out_fp16, 2, (size_t)channels, f);
    fclose(f);
    return got == (size_t)channels ? DCVC_RT_OK : DCVC_RT_ERR_IO;
}
