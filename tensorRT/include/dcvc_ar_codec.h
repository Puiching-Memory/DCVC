/* Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 *
 * Production AR (autoregressive) spatial-prior codec for DCVC-RT.
 *
 * Encapsulates the 4× AR encode/decode loop (productized from
 * test_closed_loop.c).  Both encoder and decoder use the same TensorRT
 * plugin engines (cuBLAS DepthConv), guaranteeing bit-exact self-consistency.
 *
 * Encode: y latent + params_fusion → rANS bitstream
 * Decode: params_fusion + rANS bitstream → y_hat
 */
#ifndef DCVC_AR_CODEC_H
#define DCVC_AR_CODEC_H

#include "dcvc_rt.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DcvcArCodec DcvcArCodec;

/* Create the codec.
 *   asset_dir:  path to tensorRT/assets (engines/ and decode/ subdirs)
 *   plugin_dir: path to directory containing libdcvc_kernels.so
 *   passes:     4 for intra, 2 for inter
 * Caller frees with dcvc_ar_codec_destroy(). */
DcvcArCodec* dcvc_ar_codec_create(const char* asset_dir, const char* plugin_dir,
                                  int passes, DcvcRtStatus* st);
void dcvc_ar_codec_destroy(DcvcArCodec* codec);

/* Encode the y latent via 4× AR spatial-prior loop.
 *   d_y:            CUDA device pointer, FP16 [1, N_CH, H, W]
 *   d_params_fusion: CUDA device pointer, FP16 [1, 514, H, W]
 *   H, W:           spatial dimensions (e.g. 16×16 for 256×256 input)
 *   out_stream:     receives malloc'd bitstream (caller frees with free())
 *   out_size:       receives bitstream size in bytes
 *   d_y_hat_out:    optional CUDA device ptr for reconstructed y_hat [1, N_CH, H, W] FP16 (may be NULL)
 * N_CH is 256 for intra (4 passes) or 128 for inter (2 passes). */
DcvcRtStatus dcvc_ar_codec_encode(DcvcArCodec* codec,
                                  const void* d_y, const void* d_params_fusion,
                                  int H, int W,
                                  uint8_t** out_stream, size_t* out_size,
                                  void* d_y_hat_out /* may be NULL */);

/* Decode y_hat via 4× AR spatial-prior loop.
 *   d_params_fusion: CUDA device pointer, FP16 [1, 514, H, W]
 *   stream/size:     rANS bitstream produced by dcvc_ar_codec_encode
 *   d_y_hat_out:     CUDA device pointer, FP16 [1, N_CH, H, W] (caller-allocated)
 * Returns DCVC_RT_OK on success. */
DcvcRtStatus dcvc_ar_codec_decode(DcvcArCodec* codec,
                                  const void* d_params_fusion,
                                  int H, int W,
                                  const uint8_t* stream, size_t stream_size,
                                  void* d_y_hat_out);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_AR_CODEC_H */
