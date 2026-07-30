#ifndef DCVC_RK_INTER_LD_PIPELINE_H
#define DCVC_RK_INTER_LD_PIPELINE_H

#include "dcvc_rk/types.h"
#include "dcvc_rk/profile.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DcvcRkInterLdPipeline DcvcRkInterLdPipeline;

DcvcRkInterLdPipeline* dcvc_rk_inter_create(const char* model_dir, int H, int W, int qp,
                                            DcvcRkStatus* out_st);
void dcvc_rk_inter_destroy(DcvcRkInterLdPipeline* p);

/* reset!=0: first P after I; x_ref is intra recon RGB [3,H,W].
 * Else uses stored previous decoder feature.
 * next_x: optional next-frame RGB; prefetched during dec_recon (overlap). */
DcvcRkStatus dcvc_rk_inter_encode(DcvcRkInterLdPipeline* p, const float* x,
                                  int reset, const float* x_ref,
                                  uint8_t** out_stream, size_t* out_size,
                                  float* x_hat_out);
DcvcRkStatus dcvc_rk_inter_encode_ex(DcvcRkInterLdPipeline* p, const float* x,
                                     int reset, const float* x_ref,
                                     uint8_t** out_stream, size_t* out_size,
                                     float* x_hat_out, const float* next_x);
DcvcRkStatus dcvc_rk_inter_decode(DcvcRkInterLdPipeline* p,
                                  const uint8_t* stream, size_t stream_size,
                                  int reset, const float* x_ref,
                                  float* x_hat_out);

int64_t dcvc_rk_inter_npu_us(DcvcRkInterLdPipeline* p);
void dcvc_rk_inter_reset_npu_us(DcvcRkInterLdPipeline* p);

const DcvcRkProfile* dcvc_rk_inter_last_profile(const DcvcRkInterLdPipeline* p);
const DcvcRkProfile* dcvc_rk_inter_last_ar_profile(const DcvcRkInterLdPipeline* p);

#ifdef __cplusplus
}
#endif

#endif
