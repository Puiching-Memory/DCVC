#ifndef DCVC_RK_INTRA_PIPELINE_H
#define DCVC_RK_INTRA_PIPELINE_H

#include "dcvc_rk/types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DcvcRkIntraPipeline DcvcRkIntraPipeline;

/* H,W any positive; internally padded to multiple of 64. qp indexes q_scale banks. */
DcvcRkIntraPipeline* dcvc_rk_intra_create(const char* model_dir, int H, int W, int qp,
                                          DcvcRkStatus* out_st);
void dcvc_rk_intra_destroy(DcvcRkIntraPipeline* p);

/* x: RGB FP32 NCHW [0,1] [1,3,H,W]. Optional x_hat_out same shape. */
DcvcRkStatus dcvc_rk_intra_encode(DcvcRkIntraPipeline* p, const float* x,
                                  uint8_t** out_stream, size_t* out_size,
                                  float* x_hat_out);
DcvcRkStatus dcvc_rk_intra_decode(DcvcRkIntraPipeline* p,
                                  const uint8_t* stream, size_t stream_size,
                                  float* x_hat_out);

/* Cumulative NPU run time (us) across subnet engines since last reset. */
int64_t dcvc_rk_intra_npu_us(DcvcRkIntraPipeline* p);
void dcvc_rk_intra_reset_npu_us(DcvcRkIntraPipeline* p);

#ifdef __cplusplus
}
#endif

#endif
