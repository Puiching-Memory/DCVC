#ifndef DCVC_TRT_RUNNER_H
#define DCVC_TRT_RUNNER_H

#include "dcvc_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DcvcEngineId {
    DCVC_ENG_INTRA_ANALYSIS = 0,
    DCVC_ENG_INTRA_HYPER = 1,
    DCVC_ENG_INTRA_PRIOR_FUSION = 2,
    DCVC_ENG_INTRA_SPATIAL_PRIOR = 3,
    DCVC_ENG_INTRA_SYNTHESIS = 4,
    DCVC_ENG_INTER_FEATURE = 5,
    DCVC_ENG_INTER_ANALYSIS = 6,
    DCVC_ENG_INTER_HYPER = 7,
    DCVC_ENG_INTER_PRIOR_FUSION = 8,
    DCVC_ENG_INTER_SPATIAL_PRIOR = 9,
    DCVC_ENG_INTER_SYNTHESIS = 10,
    DCVC_ENG_HYPER_DEC = 11,
    DCVC_ENG_COUNT = 12
} DcvcEngineId;

typedef struct DcvcTensorView {
    void* device_ptr; /* CUDA device pointer when available; host fallback otherwise */
    void* host_ptr;
    int n, c, h, w;
    int is_fp16; /* 1 = FP16, 0 = FP32 */
    size_t bytes;
} DcvcTensorView;

typedef struct DcvcTrtRunner DcvcTrtRunner;

DcvcTrtRunner* dcvc_trt_runner_create(const char* asset_dir, int device_id, DcvcRtStatus* st);
void dcvc_trt_runner_destroy(DcvcTrtRunner* r);

int dcvc_trt_runner_has_engine(const DcvcTrtRunner* r, DcvcEngineId id);

/* Execute named engine with NCHW bindings. Returns DCVC_RT_OK on success. */
DcvcRtStatus dcvc_trt_runner_execute(DcvcTrtRunner* r, DcvcEngineId id,
                                     DcvcTensorView* inputs, int n_in,
                                     DcvcTensorView* outputs, int n_out,
                                     void* cuda_stream /* may be NULL */);

/* QP bank gather: copy q_[qp] into out host FP16 tensor [1,C,1,1] from asset blob. */
DcvcRtStatus dcvc_trt_load_qp_scale(DcvcTrtRunner* r, const char* bank_name, int qp,
                                    uint16_t* out_fp16, int channels);

/* Get (lazy-load) a raw engine handle for direct binding. */
struct DcvcTrtEngine;
struct DcvcTrtEngine* dcvc_trt_runner_get_engine(DcvcTrtRunner* r, DcvcEngineId id,
                                                  DcvcRtStatus* st);

#ifdef __cplusplus
}
#endif

#endif
