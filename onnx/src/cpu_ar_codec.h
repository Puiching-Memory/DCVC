/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Pure-CPU AR prior codec for DCVC-UF using ONNX Runtime.
 * Runs y_spatial_prior_* networks on CPU and entropy coding via rANS.
 * params_fusion is 2*n_ch (scales+means); the y quant steps come from
 * the caller-supplied per-channel q_enc/q_dec (n_ch each).
 */
#ifndef DCVC_CPU_AR_CODEC_H
#define DCVC_CPU_AR_CODEC_H

#include "onnx_engine.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DcvcCpuArCodec DcvcCpuArCodec;

/* Create a CPU AR codec for a fixed number of channels:
 *   n_ch = 256 (intra / inter 4-pass prior; inter owns its path in
 *   cpu_inter_pipeline, which shares the same channel width).
 * model_dir: path containing the prior ONNX models and CDF files.
 * CDF files expected: gaussian_cdf.npy, gaussian_cdf_length.npy
 * (zigzag layout; *_offset.npy is unused).
 */
DcvcCpuArCodec* dcvc_cpu_ar_codec_create(const char* model_dir, int n_ch,
                                          DcvcCpuStatus* out_st);
void dcvc_cpu_ar_codec_destroy(DcvcCpuArCodec* c);

/* Encode y (FP32 NCHW) with the fused prior params (FP32 NCHW, 514 channels for 4-pass).
 * On success, *out_stream is malloc'd and *out_size is set; caller frees with free().
 * If y_hat_out is non-NULL, writes the decoded y_hat (same shape as y) into it. */
DcvcCpuStatus dcvc_cpu_ar_codec_encode_y(DcvcCpuArCodec* c,
                                          const float* y, const float* params_fusion,
                                          const float* q_enc, const float* q_dec,
                                          int H, int W,
                                          uint8_t** out_stream, size_t* out_size,
                                          float* y_hat_out);

/* Decode y_hat from the entropy stream using the fused prior params.
 * y_hat_out must be allocated by the caller [n_ch * H * W floats]. */
DcvcCpuStatus dcvc_cpu_ar_codec_decode_y(DcvcCpuArCodec* c,
                                          const float* params_fusion,
                                          const float* q_enc, const float* q_dec,
                                          int H, int W,
                                          const uint8_t* stream, size_t stream_size,
                                          float* y_hat_out);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_CPU_AR_CODEC_H */
