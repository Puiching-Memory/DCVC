#ifndef DCVC_RK_ENGINE_H
#define DCVC_RK_ENGINE_H

#include "dcvc_rk/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DcvcRkEngine DcvcRkEngine;

/* Load a .rknn model. Uses all 3 NPU cores on RK3588. No PERF_DETAIL. */
DcvcRkEngine* dcvc_rk_engine_create(const char* rknn_path, DcvcRkStatus* out_st);
void dcvc_rk_engine_destroy(DcvcRkEngine* eng);

/* Run inference. Caller-owned FP32 NCHW buffers. */
DcvcRkStatus dcvc_rk_engine_run(DcvcRkEngine* eng,
                                DcvcRkTensorView* inputs, int n_in,
                                DcvcRkTensorView* outputs, int n_out);

/* Timing of last dcvc_rk_engine_run, microseconds (-1 if unavailable).
 * run  = RKNN_QUERY_PERF_RUN (NPU only)
 * set  = rknn_inputs_set (FP32 host→NPU)
 * get  = rknn_outputs_get (NPU→FP32 host)
 * wall = set + run + get (full call) */
int64_t dcvc_rk_engine_last_run_us(DcvcRkEngine* eng);
int64_t dcvc_rk_engine_last_set_us(DcvcRkEngine* eng);
int64_t dcvc_rk_engine_last_get_us(DcvcRkEngine* eng);
int64_t dcvc_rk_engine_last_wall_us(DcvcRkEngine* eng);

#ifdef __cplusplus
}
#endif

#endif
