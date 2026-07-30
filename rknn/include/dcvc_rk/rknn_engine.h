#ifndef DCVC_RK_ENGINE_H
#define DCVC_RK_ENGINE_H

#include "dcvc_rk/types.h"
#include "dcvc_rk/quant.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DcvcRkEngine DcvcRkEngine;

/* Load a .rknn model. Uses all 3 NPU cores on RK3588. No PERF_DETAIL. */
DcvcRkEngine* dcvc_rk_engine_create(const char* rknn_path, DcvcRkStatus* out_st);
void dcvc_rk_engine_destroy(DcvcRkEngine* eng);

/* Run inference. Caller-owned FP32 NCHW buffers (mode from DCVC_RKNN_IO). */
DcvcRkStatus dcvc_rk_engine_run(DcvcRkEngine* eng,
                                DcvcRkTensorView* inputs, int n_in,
                                DcvcRkTensorView* outputs, int n_out);

/* Pipeline-native INT8 path: INT8 NCHW in → CPU NHWC + pass_through; INT8 NCHW out.
 * Inputs are tightly packed n*c*h*w. Outputs need out_buf_bytes() space. */
DcvcRkStatus dcvc_rk_engine_run_i8(DcvcRkEngine* eng,
                                   DcvcRkI8View* inputs, int n_in,
                                   DcvcRkI8View* outputs, int n_out);

/* Affine quant params + logical NCHW shape (inputs converted from NHWC attr). */
DcvcRkStatus dcvc_rk_engine_in_quant(const DcvcRkEngine* eng, int idx, DcvcRkQuant* q);
DcvcRkStatus dcvc_rk_engine_out_quant(const DcvcRkEngine* eng, int idx, DcvcRkQuant* q);
uint32_t dcvc_rk_engine_out_buf_bytes(const DcvcRkEngine* eng, int idx);

/* Timing of last run, microseconds (-1 if unavailable). */
int64_t dcvc_rk_engine_last_run_us(DcvcRkEngine* eng);
int64_t dcvc_rk_engine_last_set_us(DcvcRkEngine* eng);
int64_t dcvc_rk_engine_last_get_us(DcvcRkEngine* eng);
int64_t dcvc_rk_engine_last_wall_us(DcvcRkEngine* eng);

#ifdef __cplusplus
}
#endif

#endif
