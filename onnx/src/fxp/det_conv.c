/* Deterministic float32 Conv — fixed-order accumulation, no quantization.
 *
 * Cross-platform bit-exact because:
 *   1. float32 multiply/add are IEEE-754 correctly-rounded individually
 *   2. Fixed summation order (loop ic=0..Cin-1, left to right)
 *   3. Compiled with -ffp-contract=off → no FMA contraction
 *   4. -fno-fast-math → no reassociation
 *
 * Threading: each output channel is independent, so parallelizing across
 * oc_range is safe and preserves determinism.
 */
#include "det_conv.h"
#include <stddef.h>
#include <stdlib.h>

#if defined(_MSC_VER)
#define DET_TLS __declspec(thread)
#else
#define DET_TLS _Thread_local
#endif

/* Thread-local NHWC transpose scratch (for cache-friendly 1x1 MAC). */
static DET_TLS float* det_tls_xhw = NULL;
static DET_TLS size_t det_tls_xhw_cap = 0;

static float* det_scratch_xhw(size_t n_elem)
{
    if (n_elem > det_tls_xhw_cap) {
        float* p = (float*)realloc(det_tls_xhw, n_elem * sizeof(float));
        if (!p) return NULL;
        det_tls_xhw = p;
        det_tls_xhw_cap = n_elem;
    }
    return det_tls_xhw;
}

/* ── 1×1 conv: transposes NCHW → [hw][cin] for contiguous inner-loop MAC. */
static void det_conv1x1(const float* x, float* y,
                        int n, int cin, int cout, int h, int w,
                        const float* weight, const float* bias,
                        int oc_start, int oc_end)
{
    const int hw = h * w;
    for (int ni = 0; ni < n; ni++) {
        const float* x_n = x + (size_t)ni * cin * hw;
        float* y_n = y + (size_t)ni * cout * hw;

        /* Transpose NCHW → [hw][cin] (done once, reused across output channels) */
        float* xhw = det_scratch_xhw((size_t)cin * (size_t)hw);
        if (!xhw) return;
        for (int ic = 0; ic < cin; ic++) {
            const float* xc = x_n + (size_t)ic * hw;
            for (int s = 0; s < hw; s++)
                xhw[(size_t)s * cin + ic] = xc[s];
        }

        for (int oc = oc_start; oc < oc_end; oc++) {
            const float* wk = weight + (size_t)oc * cin;
            const float bf = bias[oc];
            float* yo = y_n + (size_t)oc * hw;
            for (int s = 0; s < hw; s++) {
                const float* xs = xhw + (size_t)s * cin;
                float acc = bf;
                for (int ic = 0; ic < cin; ic++)
                    acc += xs[ic] * wk[ic];
                yo[s] = acc;
            }
        }
    }
}

/* ── 3×3 depthwise conv (pad=1, stride=1) */
static void det_dwconv3x3(const float* x, float* y,
                          int n, int c, int h, int w,
                          const float* weight, const float* bias,
                          int oc_start, int oc_end)
{
    const int hw = h * w;
    for (int ni = 0; ni < n; ni++) {
        const float* x_n = x + (size_t)ni * c * hw;
        float* y_n = y + (size_t)ni * c * hw;
        for (int oc = oc_start; oc < oc_end; oc++) {
            const float* wk = weight + (size_t)oc * 9;
            const float* qi = x_n + (size_t)oc * hw;
            float* yo = y_n + (size_t)oc * hw;
            const float bf = bias[oc];
            for (int yh = 0; yh < h; yh++) {
                for (int xw = 0; xw < w; xw++) {
                    float acc = bf;
                    for (int kh = 0; kh < 3; kh++) {
                        int ih = yh + kh - 1;
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = xw + kw - 1;
                            if ((unsigned)ih < (unsigned)h && (unsigned)iw < (unsigned)w)
                                acc += qi[ih * w + iw] * wk[kh * 3 + kw];
                        }
                    }
                    yo[yh * w + xw] = acc;
                }
            }
        }
    }
}

/* ── General conv (fallback) with oc range for threading */
static void det_conv_general(const float* x, float* y,
    int n, int cin, int cout, int h, int w,
    const float* weight, const float* bias,
    int kh, int kw, int pad_t, int pad_l, int pad_b, int pad_r,
    int stride_h, int stride_w, int group, int oc_start, int oc_end)
{
    const int cin_g = cin / group;
    const int cout_g = cout / group;
    const int oh = (h + pad_t + pad_b - kh) / stride_h + 1;
    const int ow = (w + pad_l + pad_r - kw) / stride_w + 1;
    const int x_hw = h * w;
    const int y_hw = oh * ow;
    const int k_area = kh * kw;
    const int w_inner = cin_g * k_area;

    for (int ni = 0; ni < n; ni++) {
        const float* x_n = x + (size_t)ni * cin * x_hw;
        float* y_n = y + (size_t)ni * cout * y_hw;
        for (int oh_i = 0; oh_i < oh; oh_i++) {
            for (int ow_i = 0; ow_i < ow; ow_i++) {
                const int y_spat = oh_i * ow + ow_i;
                for (int g = 0; g < group; g++) {
                    int oc_lo = g * cout_g, oc_hi = oc_lo + cout_g;
                    if (oc_end <= oc_lo || oc_start >= oc_hi) continue;
                    int s0 = oc_start > oc_lo ? oc_start : oc_lo;
                    int s1 = oc_end < oc_hi ? oc_end : oc_hi;
                    for (int oc = s0; oc < s1; oc++) {
                        int oc_g = oc - g * cout_g;
                        const float* wk = weight + (size_t)oc * w_inner;
                        float acc = bias[oc];
                        for (int ic_g = 0; ic_g < cin_g; ic_g++) {
                            const int ic = g * cin_g + ic_g;
                            const float* x_c = x_n + (size_t)ic * x_hw;
                            const float* w_ic = wk + (size_t)ic_g * k_area;
                            for (int kh_i = 0; kh_i < kh; kh_i++) {
                                const int ih = oh_i * stride_h + kh_i - pad_t;
                                for (int kw_i = 0; kw_i < kw; kw_i++) {
                                    const int iw = ow_i * stride_w + kw_i - pad_l;
                                    if ((unsigned)ih < (unsigned)h && (unsigned)iw < (unsigned)w)
                                        acc += x_c[ih * w + iw] * w_ic[kh_i * kw + kw_i];
                                }
                            }
                        }
                        y_n[(size_t)oc * y_hw + y_spat] = acc;
                    }
                }
            }
        }
    }
}

/* ── Threaded dispatch: caller splits oc range across threads ── */
void det_conv_f32_range(const float* x, float* y,
    int n, int cin, int cout, int h, int w,
    const float* weight, const float* bias,
    int kh, int kw, int pad_t, int pad_l, int pad_b, int pad_r,
    int stride_h, int stride_w, int group, int oc_start, int oc_end)
{
    if (kh == 1 && kw == 1 && group == 1
        && pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0
        && stride_h == 1 && stride_w == 1) {
        det_conv1x1(x, y, n, cin, cout, h, w, weight, bias, oc_start, oc_end);
        return;
    }
    if (kh == 3 && kw == 3 && group == cin && cin == cout
        && pad_t == 1 && pad_l == 1 && pad_b == 1 && pad_r == 1
        && stride_h == 1 && stride_w == 1) {
        det_dwconv3x3(x, y, n, cin, h, w, weight, bias, oc_start, oc_end);
        return;
    }
    det_conv_general(x, y, n, cin, cout, h, w, weight, bias,
                     kh, kw, pad_t, pad_l, pad_b, pad_r,
                     stride_h, stride_w, group, oc_start, oc_end);
}

/* Single-threaded entry point (backward compatible) */
void det_conv_f32(const float* x, float* y,
                  int n, int cin, int cout, int h, int w,
                  const float* weight, const float* bias,
                  int kh, int kw,
                  int pad_t, int pad_l, int pad_b, int pad_r,
                  int stride_h, int stride_w, int group)
{
    det_conv_f32_range(x, y, n, cin, cout, h, w, weight, bias,
                       kh, kw, pad_t, pad_l, pad_b, pad_r,
                       stride_h, stride_w, group, 0, cout);
}
