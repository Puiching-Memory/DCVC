#ifndef DCVC_FXP_CONV_H
#define DCVC_FXP_CONV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fixed-order int-act / int16-weight Conv (float I/O), NCHW.
 * Accumulation is int64 to avoid overflow on wide 1x1 layers.
 * w_int layout matches ONNX: [Cout, Cin/group, kH, kW].
 * pads are [top, left, bottom, right].
 *
 * act_bits: 16 (default, int16 activations) or 32 (int32, near-lossless).
 */
void fxp_conv_f32(const float* x, float* y,
                  int n, int cin, int cout, int h, int w,
                  const int16_t* w_int, const float* w_scale, const float* bias,
                  float x_scale,
                  int kh, int kw,
                  int pad_t, int pad_l, int pad_b, int pad_r,
                  int stride_h, int stride_w, int group, int act_bits);

void fxp_conv1x1_f32(const float* x, float* y,
                     int n, int cin, int cout, int h, int w,
                     const int16_t* w_int, const float* w_scale,
                     const float* bias, float x_scale, int act_bits);

/* Threaded variant: caller splits [oc_start, oc_end) across threads.
 * For 1x1, each call re-quantizes X — prefer fxp_conv1x1_pack + _oc_range. */
void fxp_conv_f32_range(const float* x, float* y,
                  int n, int cin, int cout, int h, int w,
                  const int16_t* w_int, const float* w_scale, const float* bias,
                  float x_scale,
                  int kh, int kw,
                  int pad_t, int pad_l, int pad_b, int pad_r,
                  int stride_h, int stride_w, int group, int act_bits,
                  int oc_start, int oc_end);

/* --- Pack-once API (bit-exact with fxp_conv1x1_f32; for multi-thread ORT) ---
 * 1) fxp_conv1x1_pack_i16: float NCHW → int16 NHWC  [hw][cin]
 * 2) fxp_conv1x1_oc_range_i16: MAC over oc in [oc_start, oc_end) into float y
 */
void fxp_conv1x1_pack_i16(const float* x_nchw, int16_t* xq_hw_cin,
                          int cin, int hw, float x_scale);

void fxp_conv1x1_oc_range_i16(const int16_t* xq_hw_cin, float* y_nchw,
                              int cin, int cout, int hw,
                              const int16_t* w_int, const float* w_scale,
                              const float* bias, float x_scale,
                              int oc_start, int oc_end);

/* Depthwise 3x3: pack NCHW int16 once, then oc-range. */
void fxp_dw3x3_pack_i16(const float* x_nchw, int16_t* xq_nchw,
                        int c, int hw, float x_scale);

void fxp_dw3x3_oc_range_i16(const int16_t* xq_nchw, float* y_nchw,
                            int c, int h, int w,
                            const int16_t* w_int, const float* w_scale,
                            const float* bias, float x_scale,
                            int oc_start, int oc_end);

/* After writing float y for [oc_start, oc_end), apply WsRelu in-place on those
 * channels (NCHW). Bit-exact with a separate FxpWsRelu over the same span. */
void fxp_wsrelu_nchw_oc_range(float* y_nchw, int hw,
                              int oc_start, int oc_end,
                              const float* lut65536, float wsrelu_x_scale);

#ifdef __cplusplus
}
#endif

#endif
