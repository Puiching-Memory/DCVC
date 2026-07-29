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

/* --- int8 (act_bits=8) API: uint8 act × int8 weight, VNNI-accelerated ---
 * w8_comp[oc] = 128 * sum(w8[oc]) compensates the uint8 activation offset.
 */
void fxp_conv1x1_pack_i8(const float* x_nchw, uint8_t* xq_u8_hw_cin,
                         int cin, int hw, float x_scale);
void fxp_conv1x1_oc_range_i8(const uint8_t* xq_u8, float* y_nchw,
                             int cin, int cout, int hw,
                             const int8_t* w8, const float* w8_scale,
                             const int32_t* w8_comp, const float* bias,
                             float x_scale, int oc_start, int oc_end);
void fxp_conv_im2col_oc_range_i8(const uint8_t* col, float* y_nchw,
                                 int cin, int cout, int oh, int ow,
                                 const int8_t* w8, const float* w8_scale,
                                 const int32_t* w8_comp, const float* bias,
                                 float x_scale, int kh, int kw,
                                 int oc_start, int oc_end);

/* Depthwise 3x3 (int8): pack NCHW float into a PADDED uint8 layout [c][h+2][w+2]
 * (row stride = w+2). CRITICAL: out-of-bounds cells MUST be 128 (NOT 0), because
 * the uint8 activation = int8_act + 128, so a true-0 activation offsets to 128.
 * The w8_comp subtraction (128*sum(weights)) assumes every one of the 9 taps
 * carries the +128 offset, which only holds if border cells are 128. */
void fxp_dw3x3_pack_i8(const float* x_nchw, uint8_t* xq_pad,
                       int c, int h, int w, float x_scale);
void fxp_dw3x3_oc_range_i8(const uint8_t* xq_pad, float* y_nchw,
                           int c, int h, int w,
                           const int8_t* w8, const float* w8_scale,
                           const int32_t* w8_comp, const float* bias,
                           float x_scale, int oc_start, int oc_end);

/* --- im2col API (general k×k group=1 conv; for multi-thread ORT) ---
 * 1) fxp_conv_im2col_build: quantize float input + build [oh*ow][cin*kh*kw] int16 col
 * 2) fxp_conv_im2col_oc_range: SIMD GEMM over oc in [oc_start, oc_end)
 * Bit-exact with fxp_conv_f32's general path.
 */
int16_t* fxp_conv_im2col_build(const float* x_nchw, int cin, int h, int w,
                               int kh, int kw, int pad_t, int pad_l,
                               int stride_h, int stride_w,
                               int oh, int ow, float x_scale);

void fxp_conv_im2col_oc_range(const int16_t* col, float* y_nchw,
                              int cin, int cout, int oh, int ow,
                              const int16_t* w_int, const float* w_scale,
                              const float* bias, float x_scale,
                              int kh, int kw,
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

/* Depthwise 3x3: pack NCHW float into a PADDED int16 layout [c][h+2][w+2]
 * (row stride = w+2). Row 0 and row h+1 are zero, and each data row has a
 * leading 0 at col 0 and a trailing 0 at col w+1. With this padding the 3x3
 * convolution needs no per-pixel bounds checks: a zero cell contributes 0*w=0
 * to the int64 accumulator, identical to the out-of-bounds->0 rule, so it is
 * bit-exact with the compact bordered version.  xq_pad must hold c*(h+2)*(w+2). */
void fxp_dw3x3_pack_i16(const float* x_nchw, int16_t* xq_pad,
                        int c, int h, int w, float x_scale);

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
