#include "fxp_wsrelu.h"
#include "fxp_common.h"

/* Linear-interpolated LUT lookup: recovers sub-LSB precision that the
 * original truncating direct-lookup discarded.  Still fully deterministic
 * (floorf + float32 mul/sub/add are all IEEE-754 correctly-rounded on x86-64,
 * and the same MinGW-compiled code runs on both Linux and Windows).
 *
 * Error drops from O(x_scale) to O(x_scale²) — ~65536× finer.
 */
void fxp_wsrelu_f32_range(const float* x, float* y, int i0, int i1,
                          const float* lut65536, float x_scale)
{
    const float inv = 1.0f / x_scale;
    for (int i = i0; i < i1; i++) {
        float scaled = x[i] * inv;
        if (scaled >= 32767.0f) {
            y[i] = lut65536[65535];
        } else if (scaled <= -32768.0f) {
            y[i] = lut65536[0];
        } else {
            float fl = floorf(scaled);
            int idx = (int)fl + 32768;
            float frac = scaled - fl;
            y[i] = lut65536[idx] + (lut65536[idx + 1] - lut65536[idx]) * frac;
        }
    }
}

void fxp_wsrelu_f32(const float* x, float* y, int n_elem,
                    const float* lut65536, float x_scale)
{
    fxp_wsrelu_f32_range(x, y, 0, n_elem, lut65536, x_scale);
}
