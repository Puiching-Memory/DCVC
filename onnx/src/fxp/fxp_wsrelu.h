#ifndef DCVC_FXP_WSRELU_H
#define DCVC_FXP_WSRELU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* y = LUT[quantize(x, x_scale) + 32768]
 * LUT is length 65536, baked offline as
 *   lut[q+32768] = (q*s) * sigmoid(4*q*s)   (float32)
 * so the nonlinear is bit-exact by table lookup.
 */
void fxp_wsrelu_f32(const float* x, float* y, int n_elem,
                    const float* lut65536, float x_scale);

/* Half-open index range [i0, i1) — for Ort::KernelContext::ParallelFor. */
void fxp_wsrelu_f32_range(const float* x, float* y, int i0, int i1,
                          const float* lut65536, float x_scale);

#ifdef __cplusplus
}
#endif

#endif
