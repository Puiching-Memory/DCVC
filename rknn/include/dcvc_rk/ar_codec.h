#ifndef DCVC_RK_AR_CODEC_H
#define DCVC_RK_AR_CODEC_H

#include "dcvc_rk/types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DcvcRkArCodec DcvcRkArCodec;

/* passes=4 (intra, n_ch=256) or passes=2 (inter, n_ch=128). */
DcvcRkArCodec* dcvc_rk_ar_codec_create(const char* model_dir, int n_ch, int passes,
                                       DcvcRkStatus* out_st);
void dcvc_rk_ar_codec_destroy(DcvcRkArCodec* c);

DcvcRkStatus dcvc_rk_ar_encode_y(DcvcRkArCodec* c,
                                 const float* y, const float* params,
                                 int H, int W,
                                 uint8_t** out_stream, size_t* out_size,
                                 float* y_hat_out);

DcvcRkStatus dcvc_rk_ar_decode_y(DcvcRkArCodec* c,
                                 const float* params,
                                 int H, int W,
                                 const uint8_t* stream, size_t stream_size,
                                 float* y_hat_out);

#ifdef __cplusplus
}
#endif

#endif
