/* Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 *
 * Full inter (P-frame) pipeline orchestration for DCVC-RT.
 *
 * Flow (encode):
 *   ref → feature_adaptor_{p|i} → x1(p1,q=1) → ctx_t=x1·q_feat,
 *                                  x1 → p2 → ctx
 *   image+ctx+q_enc → encoder → y → hyper_enc → z → round_int8 → z rANS
 *   z_hat → hyper_dec(hier) ; ctx_t → temporal_prior(temp) ; cat(hier,temp)→prior_fusion→params
 *   y+params → AR(2x) → y_hat ; y_hat+ctx+q_dec → decoder → dec_feature (DPB)
 *   dec_feature+q_recon → recon_generation → x_hat
 *
 * Reference: if d_ref_feature != NULL use feature_adaptor_p (P-after-P);
 * otherwise pixel_unshuffle(d_ref_pixels) + feature_adaptor_i (P-after-I).
 */
#ifndef DCVC_INTER_PIPELINE_H
#define DCVC_INTER_PIPELINE_H

#include "dcvc_rt.h"
#include "dcvc_ar_codec.h"
#include "rans_c.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DcvcInterPipeline DcvcInterPipeline;

DcvcInterPipeline* dcvc_inter_pipeline_create(const char* asset_dir,
                                              const char* plugin_dir,
                                              DcvcRtStatus* st);
void dcvc_inter_pipeline_destroy(DcvcInterPipeline* p);

DcvcRtStatus dcvc_inter_encode(DcvcInterPipeline* p, DcvcArCodec* ar_codec,
                               DcvcRansEncoder* rans_enc,
                               const void* d_image, const void* d_ref_feature,
                               const void* d_ref_pixels,
                               int H, int W, int qp,
                               uint8_t** out_stream, size_t* out_size,
                               void* d_xhat_out, void* d_ref_feature_out);

DcvcRtStatus dcvc_inter_decode(DcvcInterPipeline* p, DcvcArCodec* ar_codec,
                               DcvcRansDecoder* rans_dec,
                               const uint8_t* stream, size_t stream_size,
                               const void* d_ref_feature,
                               const void* d_ref_pixels,
                               int H, int W, int qp,
                               void* d_xhat_out, void* d_ref_feature_out);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_INTER_PIPELINE_H */
