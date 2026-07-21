/* Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 *
 * Full intra-frame pipeline orchestration for DCVC-RT.
 *
 * Wires together: analysis → hyper_enc → z rANS → hyper_dec → prior_fusion
 *                 → AR codec → synthesis.
 *
 * Encode: image [1,3,H,W] → bitstream + optional x_hat reconstruction
 * Decode: bitstream → x_hat [1,3,H,W] reconstruction
 */
#ifndef DCVC_INTRA_PIPELINE_H
#define DCVC_INTRA_PIPELINE_H

#include "dcvc_rt.h"
#include "dcvc_trt_runner.h"
#include "dcvc_ar_codec.h"
#include "rans_c.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Encode a full intra I-frame.
 *   runner:     TRT runner with analysis/hyper/synthesis engines
 *   ar_codec:   AR spatial-prior codec (4 passes for intra)
 *   rans_enc:   rANS encoder for z symbols
 *   d_image:    CUDA device FP16 [1,3,H,W] input image in [0,1] YCbCr444
 *   H, W:       padded spatial dimensions (must be multiples of 16)
 *   qp:         quantization parameter 0..63
 *   out_stream: receives malloc'd bitstream (caller frees)
 *   out_size:   receives bitstream byte count
 *   d_xhat_out: optional CUDA FP16 [1,3,H,W] reconstruction (may be NULL)
 *
 * Bitstream format: [z_len:u32][z_payload][y_payload]
 */
DcvcRtStatus dcvc_intra_encode(DcvcTrtRunner* runner, DcvcArCodec* ar_codec,
                               DcvcRansEncoder* rans_enc,
                               const void* d_image, int H, int W, int qp,
                               uint8_t** out_stream, size_t* out_size,
                               void* d_xhat_out /* may be NULL */);

/* Decode a full intra I-frame.
 *   runner:     TRT runner with hyper/synthesis engines
 *   ar_codec:   AR spatial-prior codec (4 passes for intra)
 *   rans_dec:   rANS decoder for z symbols
 *   stream/size: bitstream from dcvc_intra_encode
 *   H, W:       padded spatial dimensions
 *   qp:         quantization parameter
 *   d_xhat_out: CUDA FP16 [1,3,H,W] reconstruction output (caller-allocated)
 */
DcvcRtStatus dcvc_intra_decode(DcvcTrtRunner* runner, DcvcArCodec* ar_codec,
                               DcvcRansDecoder* rans_dec,
                               const uint8_t* stream, size_t stream_size,
                               int H, int W, int qp,
                               void* d_xhat_out);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_INTRA_PIPELINE_H */
