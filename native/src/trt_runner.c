#include "dcvc_trt_runner.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dcvc_rt_internal.h"

#ifdef DCVC_RT_HAS_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

struct DcvcTrtRunner {
    char asset_dir[512];
    int device_id;
    int engine_present[DCVC_ENG_COUNT];
#ifdef DCVC_RT_HAS_TENSORRT
    /* Opaque TRT objects filled when engines load successfully */
    void* runtime;
    void* engines[DCVC_ENG_COUNT];
    void* contexts[DCVC_ENG_COUNT];
#endif
};

static const char* k_engine_names[DCVC_ENG_COUNT] = {
    "intra_analysis",
    "intra_hyper",
    "intra_prior_fusion",
    "intra_spatial_prior",
    "intra_synthesis",
    "inter_feature",
    "inter_analysis",
    "inter_hyper",
    "inter_prior_fusion",
    "inter_spatial_prior",
    "inter_synthesis",
};

static int file_exists(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
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
        strncpy(r->asset_dir, "native/assets", sizeof(r->asset_dir) - 1);
    }
    r->device_id = device_id;

    for (int i = 0; i < DCVC_ENG_COUNT; i++) {
        char path[640];
        snprintf(path, sizeof(path), "%s/engines/%s.engine", r->asset_dir, k_engine_names[i]);
        r->engine_present[i] = file_exists(path);
    }

#ifdef DCVC_RT_HAS_TENSORRT
    /* Full deserialize path enabled when TensorRT is linked.
     * Engines are produced by tools/build_engines.py */
    (void)0;
#endif

    if (st) *st = DCVC_RT_OK;
    return r;
}

void dcvc_trt_runner_destroy(DcvcTrtRunner* r)
{
    free(r);
}

int dcvc_trt_runner_has_engine(const DcvcTrtRunner* r, DcvcEngineId id)
{
    if (!r || id < 0 || id >= DCVC_ENG_COUNT) return 0;
    return r->engine_present[id];
}

DcvcRtStatus dcvc_trt_runner_execute(DcvcTrtRunner* r, DcvcEngineId id,
                                     DcvcTensorView* inputs, int n_in,
                                     DcvcTensorView* outputs, int n_out,
                                     void* cuda_stream)
{
    (void)inputs;
    (void)n_in;
    (void)outputs;
    (void)n_out;
    (void)cuda_stream;
    if (!r) return DCVC_RT_ERR_INVALID_ARG;
    if (!dcvc_trt_runner_has_engine(r, id)) return DCVC_RT_ERR_NO_ENGINE;

#ifdef DCVC_RT_HAS_TENSORRT
    return DCVC_RT_ERR_UNSUPPORTED; /* deserialize+enqueue wired when TRT SDK present */
#else
    (void)id;
    return DCVC_RT_ERR_NO_ENGINE;
#endif
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
