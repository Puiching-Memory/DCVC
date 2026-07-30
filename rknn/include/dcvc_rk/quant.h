#ifndef DCVC_RK_QUANT_H
#define DCVC_RK_QUANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DcvcRkQuant {
    float scale;
    int32_t zp;
    int n, c, h, w;
    int w_stride; /* output may be strided; 0 or ==w means tight */
} DcvcRkQuant;

typedef struct DcvcRkI8View {
    int8_t* data; /* INT8 NCHW; outs may use w_stride from matching Quant */
    int n, c, h, w;
} DcvcRkI8View;

/* FP32 NCHW → INT8 NCHW (tight). */
void dcvc_rk_quant_nchw_f32_to_i8(const float* src, int8_t* dst,
                                  int N, int C, int H, int W,
                                  float scale, int32_t zp);

/* INT8 NCHW (optional w_stride) → FP32 NCHW tight. */
void dcvc_rk_dequant_nchw_i8_to_f32(const int8_t* src, float* dst,
                                    int N, int C, int H, int W, int w_stride,
                                    float scale, int32_t zp);

/* INT8 NCHW strided → INT8 NCHW tight, affine requant across models. */
void dcvc_rk_requant_nchw_i8(const int8_t* src, int N, int C, int H, int W, int w_stride,
                             float scale_src, int32_t zp_src,
                             int8_t* dst, float scale_dst, int32_t zp_dst);

#ifdef __cplusplus
}
#endif

#endif
