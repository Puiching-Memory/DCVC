/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Pure-CPU inter-frame (P-frame) encode/decode pipeline using ONNX Runtime
 * + rANS. Implements the DCVC-RT video model (DMC) 2x prior path.
 *
 * A P-frame references the previous reconstructed frame (pixel domain) via
 * feature_adaptor_i(pixel_unshuffle(x_hat_prev)). The caller maintains the
 * reference frame (the previous x_hat) and passes it to encode/decode.
 */
#ifndef DCVC_CPU_INTER_PIPELINE_H
#define DCVC_CPU_INTER_PIPELINE_H

#include "onnx_engine.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DcvcCpuInterPipeline DcvcCpuInterPipeline;

/* Create a P-frame pipeline for H,W,qp. Any positive H,W is accepted; the
 * pipeline replicate-pads to the next multiple of 64 internally and crops the
 * reconstruction back to H,W. model_dir must contain the inter_* ONNX models,
 * q_encoder/decoder/feature/recon .npy banks, and gaussian_cdf/bitest_cdf. */
DcvcCpuInterPipeline* dcvc_cpu_inter_pipeline_create(const char* model_dir,
                                                     int H, int W, int qp,
                                                     DcvcCpuStatus* out_st);
void dcvc_cpu_inter_pipeline_destroy(DcvcCpuInterPipeline* p);

/* Encode one P-frame.
 *   x        : current frame, FP32 NCHW RGB [1,3,H,W] in [0,1].
 *   x_hat_ref: previous reconstructed frame, FP32 RGB [1,3,H,W] in [0,1].
 * On success, *out_stream is malloc'd (caller frees); if x_hat_out != NULL it
 * receives the reconstructed current frame [1,3,H,W] RGB. */
DcvcCpuStatus dcvc_cpu_inter_pipeline_encode(DcvcCpuInterPipeline* p,
                                             const float* x,
                                             const float* x_hat_ref,
                                             uint8_t** out_stream, size_t* out_size,
                                             float* x_hat_out);

/* Decode one P-frame from the bitstream, given the previous RGB reconstruction. */
DcvcCpuStatus dcvc_cpu_inter_pipeline_decode(DcvcCpuInterPipeline* p,
                                             const uint8_t* stream, size_t stream_size,
                                             const float* x_hat_ref,
                                             float* x_hat_out);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_CPU_INTER_PIPELINE_H */
