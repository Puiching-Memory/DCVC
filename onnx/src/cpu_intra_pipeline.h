/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Pure-CPU intra-frame encode/decode pipeline using ONNX Runtime + rANS.
 */
#ifndef DCVC_CPU_INTRA_PIPELINE_H
#define DCVC_CPU_INTRA_PIPELINE_H

#include "cpu_ar_codec.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DcvcCpuIntraPipeline DcvcCpuIntraPipeline;

DcvcCpuIntraPipeline* dcvc_cpu_intra_pipeline_create(const char* model_dir,
                                                      int H, int W, int qp,
                                                      DcvcCpuStatus* out_st);
void dcvc_cpu_intra_pipeline_destroy(DcvcCpuIntraPipeline* p);

/* Encode one frame: x is FP32 NCHW RGB [0,1] of shape [1,3,H,W].
 * On success, *out_stream is malloc'd; caller frees with free().
 * If x_hat_out is non-NULL, writes the reconstructed [1,3,H,W] FP32 RGB. */
DcvcCpuStatus dcvc_cpu_intra_pipeline_encode(DcvcCpuIntraPipeline* p,
                                              const float* x,
                                              uint8_t** out_stream, size_t* out_size,
                                              float* x_hat_out);

/* Decode one frame from the bitstream. x_hat_out [1,3,H,W] must be allocated
 * and will be filled with RGB FP32 values in [0,1]. */
DcvcCpuStatus dcvc_cpu_intra_pipeline_decode(DcvcCpuIntraPipeline* p,
                                              const uint8_t* stream, size_t stream_size,
                                              float* x_hat_out);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_CPU_INTRA_PIPELINE_H */
