#ifndef DCVC_FXP_CUDA_H
#define DCVC_FXP_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Device-pointer FXP kernels (NCHW float I/O). stream may be NULL (= default).
 * Semantics match the CPU fxp_conv / fxp_wsrelu path (bit-exact on IEEE float).
 * Returns 0 on success, non-zero on CUDA / shape error.
 */
int fxp_conv1x1_f32_cuda(const float* x, float* y,
                         int n, int cin, int cout, int h, int w,
                         const int16_t* w_int, const float* w_scale,
                         const float* bias, float x_scale,
                         void* cuda_stream /* cudaStream_t */);

int fxp_dwconv3x3_f32_cuda(const float* x, float* y,
                           int n, int c, int h, int w,
                           const int16_t* w_int, const float* w_scale,
                           const float* bias, float x_scale,
                           void* cuda_stream);

/* Dispatches 1x1 / dw3x3; other shapes return -1 (caller must fallback). */
int fxp_conv_f32_cuda(const float* x, float* y,
                      int n, int cin, int cout, int h, int w,
                      const int16_t* w_int, const float* w_scale, const float* bias,
                      float x_scale,
                      int kh, int kw,
                      int pad_t, int pad_l, int pad_b, int pad_r,
                      int stride_h, int stride_w, int group,
                      void* cuda_stream);

int fxp_wsrelu_f32_cuda(const float* x, float* y, int n_elem,
                        const float* lut65536, float x_scale,
                        void* cuda_stream);

#ifdef __cplusplus
}
#endif

#endif
