/* Frozen pre-optimization FXP kernels for A/B benchmarks only. */
#include "fxp/fxp_common.h"

#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>

void fxp_baseline_conv1x1_f32(const float* x, float* y,
                              int n, int cin, int cout, int h, int w,
                              const int16_t* w_int, const float* w_scale,
                              const float* bias, float x_scale, int act_bits)
{
    (void)act_bits;
    const float inv_x = 1.0f / x_scale;
    const int hw = h * w;
    int16_t* xq = (int16_t*)malloc((size_t)cin * hw * sizeof(int16_t));
    if (!xq)
        return;
    for (int ni = 0; ni < n; ni++) {
        const float* x_n = x + (size_t)ni * cin * hw;
        float* y_n = y + (size_t)ni * cout * hw;
        for (int ic = 0; ic < cin; ic++) {
            const float* xc = x_n + (size_t)ic * hw;
            int16_t* qc = xq + (size_t)ic * hw;
            for (int s = 0; s < hw; s++)
                qc[s] = fxp_quantize_act(xc[s], inv_x);
        }
        for (int oc = 0; oc < cout; oc++) {
            const int16_t* wk = w_int + (size_t)oc * cin;
            float* yo = y_n + (size_t)oc * hw;
            const float scale = x_scale * w_scale[oc];
            const float b = bias[oc];
            for (int s = 0; s < hw; s++) {
                int64_t acc = 0;
                for (int ic = 0; ic < cin; ic++)
                    acc += (int64_t)xq[(size_t)ic * hw + s] * (int64_t)wk[ic];
                yo[s] = (float)acc * scale + b;
            }
        }
    }
    free(xq);
}

void fxp_baseline_dw3x3_f32(const float* x, float* y,
                            int n, int c, int h, int w,
                            const int16_t* w_int, const float* w_scale,
                            const float* bias, float x_scale, int act_bits)
{
    (void)act_bits;
    const float inv_x = 1.0f / x_scale;
    const int hw = h * w;
    int16_t* xq = (int16_t*)malloc((size_t)c * hw * sizeof(int16_t));
    if (!xq)
        return;
    for (int ni = 0; ni < n; ni++) {
        const float* x_n = x + (size_t)ni * c * hw;
        float* y_n = y + (size_t)ni * c * hw;
        for (int ic = 0; ic < c; ic++) {
            const float* xc = x_n + (size_t)ic * hw;
            int16_t* qc = xq + (size_t)ic * hw;
            for (int s = 0; s < hw; s++)
                qc[s] = fxp_quantize_act(xc[s], inv_x);
        }
        for (int oc = 0; oc < c; oc++) {
            const int16_t* wk = w_int + (size_t)oc * 9;
            float* yo = y_n + (size_t)oc * hw;
            const int16_t* qi = xq + (size_t)oc * hw;
            const float scale = x_scale * w_scale[oc];
            const float b = bias[oc];
            for (int yh = 0; yh < h; yh++) {
                for (int xw = 0; xw < w; xw++) {
                    int64_t acc = 0;
                    for (int kh = 0; kh < 3; kh++) {
                        int ih = yh + kh - 1;
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = xw + kw - 1;
                            int16_t v = 0;
                            if ((unsigned)ih < (unsigned)h && (unsigned)iw < (unsigned)w)
                                v = qi[ih * w + iw];
                            acc += (int64_t)v * (int64_t)wk[kh * 3 + kw];
                        }
                    }
                    yo[yh * w + xw] = (float)acc * scale + b;
                }
            }
        }
    }
    free(xq);
}
