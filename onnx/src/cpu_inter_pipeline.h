/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Pure-CPU inter (P-chunk) encode/decode pipeline using ONNX Runtime + rANS.
 * Implements the DCVC-UF chunk-based video model (DMC, HT-S / HT-L) 4x prior
 * path: encodes a *chunk* of g_frame_delay (8) frames into one latent and
 * reconstructs all 8 frames in parallel from a single decoder feature plane.
 *
 * A P-chunk references the previous reference via a feature-memory DPB:
 *   - first P-chunk after an intra frame: feature_adaptor_i(pixel_unshuffle(x_hat_intra))
 *   - subsequent P-chunks:                feature_adaptor_m(memory, prev_decoder_feature)
 * The pipeline owns the `memory` buffer and the previous decoder feature; the
 * caller signals `reset` (== first chunk, supply the intra reconstruction).
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

#ifndef DCVC_FRAME_DELAY
#define DCVC_FRAME_DELAY 8   /* g_frame_delay: frames per chunk */
#endif

/* Create a P-chunk pipeline for H,W,qp. Any positive H,W is accepted; the
 * pipeline replicate-pads to the next multiple of 64 internally and crops the
 * reconstruction back to H,W. is_hts selects the HT-S (1) or HT-L (0) variant.
 * model_dir must contain the inter_* ONNX models, q_encoder/decoder/feature
 * .npy banks, and gaussian_cdf/bitest_cdf. */
DcvcCpuInterPipeline* dcvc_cpu_inter_pipeline_create(const char* model_dir,
                                                     int H, int W, int qp,
                                                     int is_hts,
                                                     DcvcCpuStatus* out_st);
void dcvc_cpu_inter_pipeline_destroy(DcvcCpuInterPipeline* p);

/* Encode one P-chunk.
 *   x_chunk : DCVC_FRAME_DELAY frames, FP32 NCHW RGB [N,3,H,W] in [0,1].
 *   reset   : nonzero for the first P-chunk (then x_ref is the intra recon).
 *   x_ref   : if reset, the intra reconstruction [3,H,W]; else ignored (the
 *             pipeline reuses its stored previous decoder feature).
 * On success, *out_stream is malloc'd (caller frees); if x_hat_out != NULL it
 * receives the reconstructed chunk [N,3,H,W] RGB. */
DcvcCpuStatus dcvc_cpu_inter_pipeline_encode(DcvcCpuInterPipeline* p,
                                             const float* x_chunk,
                                             int reset, const float* x_ref,
                                             uint8_t** out_stream, size_t* out_size,
                                             float* x_hat_out);

/* Decode one P-chunk from the bitstream. x_hat_out receives [N,3,H,W] RGB. */
DcvcCpuStatus dcvc_cpu_inter_pipeline_decode(DcvcCpuInterPipeline* p,
                                             const uint8_t* stream, size_t stream_size,
                                             int reset, const float* x_ref,
                                             float* x_hat_out);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_CPU_INTER_PIPELINE_H */
