#ifndef DCVC_DET_CONV_H
#define DCVC_DET_CONV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Deterministic float32 Conv with fixed-order accumulation.
 * No quantization — weights and activations are float32.
 * -ffp-contract=off makes it cross-platform bit-exact.
 *
 * Threaded variant: caller splits [oc_start, oc_end) across threads.
 */
void det_conv_f32(const float* x, float* y,
                  int n, int cin, int cout, int h, int w,
                  const float* weight, const float* bias,
                  int kh, int kw,
                  int pad_t, int pad_l, int pad_b, int pad_r,
                  int stride_h, int stride_w, int group);

void det_conv_f32_range(const float* x, float* y,
    int n, int cin, int cout, int h, int w,
    const float* weight, const float* bias,
    int kh, int kw, int pad_t, int pad_l, int pad_b, int pad_r,
    int stride_h, int stride_w, int group, int oc_start, int oc_end);

#ifdef __cplusplus
}
#endif

#endif
