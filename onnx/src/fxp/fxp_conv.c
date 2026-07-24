#include "fxp_conv.h"
#include "fxp_common.h"
#include "fxp_wsrelu.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#define FXP_TLS __declspec(thread)
#else
#define FXP_TLS _Thread_local
#endif

/* TLS scratch for single-threaded entry points (tests / fallback). */
static FXP_TLS int16_t* fxp_tls_i16 = NULL;
static FXP_TLS size_t fxp_tls_i16_cap = 0;
static FXP_TLS int32_t* fxp_tls_i32 = NULL;
static FXP_TLS size_t fxp_tls_i32_cap = 0;

static int16_t* fxp_scratch_i16(size_t n_elem)
{
    if (n_elem > fxp_tls_i16_cap) {
        int16_t* p = (int16_t*)realloc(fxp_tls_i16, n_elem * sizeof(int16_t));
        if (!p) return NULL;
        fxp_tls_i16 = p;
        fxp_tls_i16_cap = n_elem;
    }
    return fxp_tls_i16;
}

static int32_t* fxp_scratch_i32(size_t n_elem)
{
    if (n_elem > fxp_tls_i32_cap) {
        int32_t* p = (int32_t*)realloc(fxp_tls_i32, n_elem * sizeof(int32_t));
        if (!p) return NULL;
        fxp_tls_i32 = p;
        fxp_tls_i32_cap = n_elem;
    }
    return fxp_tls_i32;
}

enum { FXP_OC_TILE = 4, FXP_S_TILE = 8 };

void fxp_conv1x1_pack_i16(const float* x_nchw, int16_t* xq_hw_cin,
                          int cin, int hw, float x_scale)
{
    const float inv_x = 1.0f / x_scale;
    /* Write NHWC with spatial-major stores (better than ic-outer strided stores). */
    for (int s = 0; s < hw; s++) {
        int16_t* row = xq_hw_cin + (size_t)s * cin;
        for (int ic = 0; ic < cin; ic++)
            row[ic] = fxp_quantize_act(x_nchw[(size_t)ic * hw + s], inv_x);
    }
}

void fxp_conv1x1_oc_range_i16(const int16_t* xq_hw_cin, float* y_nchw,
                              int cin, int cout, int hw,
                              const int16_t* w_int, const float* w_scale,
                              const float* bias, float x_scale,
                              int oc_start, int oc_end)
{
    (void)cout;
    for (int oc0 = oc_start; oc0 < oc_end; oc0 += FXP_OC_TILE) {
        const int noc = oc_end - oc0 < FXP_OC_TILE ? oc_end - oc0 : FXP_OC_TILE;
        for (int s0 = 0; s0 < hw; s0 += FXP_S_TILE) {
            const int ns = hw - s0 < FXP_S_TILE ? hw - s0 : FXP_S_TILE;
            for (int oi = 0; oi < noc; oi++) {
                const int oc = oc0 + oi;
                const int16_t* wk = w_int + (size_t)oc * cin;
                float* yo = y_nchw + (size_t)oc * hw;
                const float scale = x_scale * w_scale[oc];
                const float b = bias[oc];
                for (int si = 0; si < ns; si++) {
                    const int s = s0 + si;
                    const int16_t* xs = xq_hw_cin + (size_t)s * cin;
                    int64_t acc = 0;
                    for (int ic = 0; ic < cin; ic++)
                        acc += (int64_t)xs[ic] * (int64_t)wk[ic];
                    yo[s] = (float)acc * scale + b;
                }
            }
        }
    }
}

void fxp_dw3x3_pack_i16(const float* x_nchw, int16_t* xq_nchw,
                        int c, int hw, float x_scale)
{
    const float inv_x = 1.0f / x_scale;
    for (int ic = 0; ic < c; ic++) {
        const float* xc = x_nchw + (size_t)ic * hw;
        int16_t* qc = xq_nchw + (size_t)ic * hw;
        for (int s = 0; s < hw; s++)
            qc[s] = fxp_quantize_act(xc[s], inv_x);
    }
}

void fxp_dw3x3_oc_range_i16(const int16_t* xq_nchw, float* y_nchw,
                            int c, int h, int w,
                            const int16_t* w_int, const float* w_scale,
                            const float* bias, float x_scale,
                            int oc_start, int oc_end)
{
    (void)c;
    const int hw = h * w;
    for (int oc = oc_start; oc < oc_end; oc++) {
        const int16_t* wk = w_int + (size_t)oc * 9;
        float* yo = y_nchw + (size_t)oc * hw;
        const int16_t* qi = xq_nchw + (size_t)oc * hw;
        const float scale = x_scale * w_scale[oc];
        const float b = bias[oc];
        const int16_t w00 = wk[0], w01 = wk[1], w02 = wk[2];
        const int16_t w10 = wk[3], w11 = wk[4], w12 = wk[5];
        const int16_t w20 = wk[6], w21 = wk[7], w22 = wk[8];

        for (int yh = 0; yh < h; yh++) {
            const int is_border_row = (yh == 0 || yh == h - 1);
            if (!is_border_row && w >= 3) {
                for (int pass = 0; pass < 2; pass++) {
                    const int xw = pass == 0 ? 0 : w - 1;
                    int64_t acc = 0;
                    for (int kh = 0; kh < 3; kh++) {
                        int ih = yh + kh - 1;
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = xw + kw - 1;
                            int32_t v = 0;
                            if ((unsigned)ih < (unsigned)h && (unsigned)iw < (unsigned)w)
                                v = qi[ih * w + iw];
                            acc += (int64_t)v * (int64_t)wk[kh * 3 + kw];
                        }
                    }
                    yo[yh * w + xw] = (float)acc * scale + b;
                }
                for (int xw = 1; xw < w - 1; xw++) {
                    const int16_t* r0 = qi + (yh - 1) * w + (xw - 1);
                    const int16_t* r1 = qi + yh * w + (xw - 1);
                    const int16_t* r2 = qi + (yh + 1) * w + (xw - 1);
                    int64_t acc =
                        (int64_t)r0[0] * w00 + (int64_t)r0[1] * w01 + (int64_t)r0[2] * w02 +
                        (int64_t)r1[0] * w10 + (int64_t)r1[1] * w11 + (int64_t)r1[2] * w12 +
                        (int64_t)r2[0] * w20 + (int64_t)r2[1] * w21 + (int64_t)r2[2] * w22;
                    yo[yh * w + xw] = (float)acc * scale + b;
                }
            } else {
                for (int xw = 0; xw < w; xw++) {
                    int64_t acc = 0;
                    for (int kh = 0; kh < 3; kh++) {
                        int ih = yh + kh - 1;
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = xw + kw - 1;
                            int32_t v = 0;
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
}

void fxp_wsrelu_nchw_oc_range(float* y_nchw, int hw,
                              int oc_start, int oc_end,
                              const float* lut65536, float wsrelu_x_scale)
{
    for (int oc = oc_start; oc < oc_end; oc++) {
        float* yo = y_nchw + (size_t)oc * hw;
        fxp_wsrelu_f32_range(yo, yo, 0, hw, lut65536, wsrelu_x_scale);
    }
}

static void fxp_conv1x1_fast(const float* x, float* y,
                             int n, int cin, int cout, int h, int w,
                             const int16_t* w_int, const float* w_scale,
                             const float* bias, float x_scale, int act_bits)
{
    if (act_bits >= 24) {
        /* int32 activation path (rare): keep prior int32 pack loop. */
        const float inv_x = 1.0f / x_scale;
        const int hw = h * w;
        int32_t* xq = fxp_scratch_i32((size_t)cin * (size_t)hw);
        if (!xq) return;
        for (int ni = 0; ni < n; ni++) {
            const float* x_n = x + (size_t)ni * cin * hw;
            float* y_n = y + (size_t)ni * cout * hw;
            for (int s = 0; s < hw; s++) {
                int32_t* row = xq + (size_t)s * cin;
                for (int ic = 0; ic < cin; ic++)
                    row[ic] = fxp_quantize_act_i32(x_n[(size_t)ic * hw + s], inv_x);
            }
            for (int oc = 0; oc < cout; oc++) {
                const int16_t* wk = w_int + (size_t)oc * cin;
                float* yo = y_n + (size_t)oc * hw;
                const float scale = x_scale * w_scale[oc];
                const float b = bias[oc];
                for (int s = 0; s < hw; s++) {
                    const int32_t* xs = xq + (size_t)s * cin;
                    int64_t acc = 0;
                    for (int ic = 0; ic < cin; ic++)
                        acc += (int64_t)xs[ic] * (int64_t)wk[ic];
                    yo[s] = (float)acc * scale + b;
                }
            }
        }
        return;
    }

    const int hw = h * w;
    int16_t* xq = fxp_scratch_i16((size_t)cin * (size_t)hw);
    if (!xq) return;
    for (int ni = 0; ni < n; ni++) {
        const float* x_n = x + (size_t)ni * cin * hw;
        float* y_n = y + (size_t)ni * cout * hw;
        fxp_conv1x1_pack_i16(x_n, xq, cin, hw, x_scale);
        fxp_conv1x1_oc_range_i16(xq, y_n, cin, cout, hw, w_int, w_scale, bias,
                                 x_scale, 0, cout);
    }
}

static void fxp_dwconv3x3_fast(const float* x, float* y,
                               int n, int c, int h, int w,
                               const int16_t* w_int, const float* w_scale,
                               const float* bias, float x_scale, int act_bits)
{
    if (act_bits >= 24) {
        /* Fallback: quantize inline as int32 (uncommon). */
        const float inv_x = 1.0f / x_scale;
        const int hw = h * w;
        int32_t* xq = fxp_scratch_i32((size_t)c * (size_t)hw);
        if (!xq) return;
        for (int ni = 0; ni < n; ni++) {
            const float* x_n = x + (size_t)ni * c * hw;
            float* y_n = y + (size_t)ni * c * hw;
            for (int ic = 0; ic < c; ic++) {
                const float* xc = x_n + (size_t)ic * hw;
                int32_t* qc = xq + (size_t)ic * hw;
                for (int s = 0; s < hw; s++)
                    qc[s] = fxp_quantize_act_i32(xc[s], inv_x);
            }
            /* Reuse dw range by temporarily widening — call via float path copy. */
            for (int oc = 0; oc < c; oc++) {
                const int16_t* wk = w_int + (size_t)oc * 9;
                float* yo = y_n + (size_t)oc * hw;
                const int32_t* qi = xq + (size_t)oc * hw;
                const float scale = x_scale * w_scale[oc];
                const float b = bias[oc];
                for (int yh = 0; yh < h; yh++) {
                    for (int xw = 0; xw < w; xw++) {
                        int64_t acc = 0;
                        for (int kh = 0; kh < 3; kh++) {
                            int ih = yh + kh - 1;
                            for (int kw = 0; kw < 3; kw++) {
                                int iw = xw + kw - 1;
                                int32_t v = 0;
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
        return;
    }

    const int hw = h * w;
    int16_t* xq = fxp_scratch_i16((size_t)c * (size_t)hw);
    if (!xq) return;
    for (int ni = 0; ni < n; ni++) {
        const float* x_n = x + (size_t)ni * c * hw;
        float* y_n = y + (size_t)ni * c * hw;
        fxp_dw3x3_pack_i16(x_n, xq, c, hw, x_scale);
        fxp_dw3x3_oc_range_i16(xq, y_n, c, h, w, w_int, w_scale, bias, x_scale, 0, c);
    }
}

void fxp_conv_f32(const float* x, float* y,
                  int n, int cin, int cout, int h, int w,
                  const int16_t* w_int, const float* w_scale, const float* bias,
                  float x_scale,
                  int kh, int kw,
                  int pad_t, int pad_l, int pad_b, int pad_r,
                  int stride_h, int stride_w, int group, int act_bits)
{
    if (kh == 1 && kw == 1 && group == 1
        && pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0
        && stride_h == 1 && stride_w == 1) {
        fxp_conv1x1_fast(x, y, n, cin, cout, h, w, w_int, w_scale, bias, x_scale, act_bits);
        return;
    }
    if (kh == 3 && kw == 3 && group == cin && cin == cout
        && pad_t == 1 && pad_l == 1 && pad_b == 1 && pad_r == 1
        && stride_h == 1 && stride_w == 1) {
        fxp_dwconv3x3_fast(x, y, n, cin, h, w, w_int, w_scale, bias, x_scale, act_bits);
        return;
    }

    const float inv_x = 1.0f / x_scale;
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
                    for (int oc_g = 0; oc_g < cout_g; oc_g++) {
                        const int oc = g * cout_g + oc_g;
                        const int16_t* wk = w_int + (size_t)oc * w_inner;
                        int64_t acc = 0;
                        for (int ic_g = 0; ic_g < cin_g; ic_g++) {
                            const int ic = g * cin_g + ic_g;
                            const float* x_c = x_n + (size_t)ic * x_hw;
                            const int16_t* w_ic = wk + (size_t)ic_g * k_area;
                            for (int kh_i = 0; kh_i < kh; kh_i++) {
                                const int ih = oh_i * stride_h + kh_i - pad_t;
                                for (int kw_i = 0; kw_i < kw; kw_i++) {
                                    const int iw = ow_i * stride_w + kw_i - pad_l;
                                    int32_t xq = 0;
                                    if ((unsigned)ih < (unsigned)h && (unsigned)iw < (unsigned)w) {
                                        if (act_bits >= 24)
                                            xq = fxp_quantize_act_i32(x_c[ih * w + iw], inv_x);
                                        else
                                            xq = fxp_quantize_act(x_c[ih * w + iw], inv_x);
                                    }
                                    acc += (int64_t)xq * (int64_t)w_ic[kh_i * kw + kw_i];
                                }
                            }
                        }
                        float scale = x_scale * w_scale[oc];
                        y_n[(size_t)oc * y_hw + y_spat] = (float)acc * scale + bias[oc];
                    }
                }
            }
        }
    }
}

void fxp_conv1x1_f32(const float* x, float* y,
                     int n, int cin, int cout, int h, int w,
                     const int16_t* w_int, const float* w_scale,
                     const float* bias, float x_scale, int act_bits)
{
    fxp_conv1x1_fast(x, y, n, cin, cout, h, w, w_int, w_scale, bias, x_scale, act_bits);
}

void fxp_conv_f32_range(const float* x, float* y,
                  int n, int cin, int cout, int h, int w,
                  const int16_t* w_int, const float* w_scale, const float* bias,
                  float x_scale,
                  int kh, int kw,
                  int pad_t, int pad_l, int pad_b, int pad_r,
                  int stride_h, int stride_w, int group, int act_bits,
                  int oc_start, int oc_end)
{
    if (kh == 1 && kw == 1 && group == 1
        && pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0
        && stride_h == 1 && stride_w == 1 && act_bits < 24) {
        const int hw = h * w;
        int16_t* xq = fxp_scratch_i16((size_t)cin * (size_t)hw);
        if (!xq) return;
        for (int ni = 0; ni < n; ni++) {
            const float* x_n = x + (size_t)ni * cin * hw;
            float* y_n = y + (size_t)ni * cout * hw;
            fxp_conv1x1_pack_i16(x_n, xq, cin, hw, x_scale);
            fxp_conv1x1_oc_range_i16(xq, y_n, cin, cout, hw, w_int, w_scale, bias,
                                     x_scale, oc_start, oc_end);
        }
        return;
    }
    if (kh == 3 && kw == 3 && group == cin && cin == cout
        && pad_t == 1 && pad_l == 1 && pad_b == 1 && pad_r == 1
        && stride_h == 1 && stride_w == 1 && act_bits < 24) {
        const int hw = h * w;
        int16_t* xq = fxp_scratch_i16((size_t)cin * (size_t)hw);
        if (!xq) return;
        for (int ni = 0; ni < n; ni++) {
            const float* x_n = x + (size_t)ni * cin * hw;
            float* y_n = y + (size_t)ni * cout * hw;
            fxp_dw3x3_pack_i16(x_n, xq, cin, hw, x_scale);
            fxp_dw3x3_oc_range_i16(xq, y_n, cin, h, w, w_int, w_scale, bias,
                                   x_scale, oc_start, oc_end);
        }
        return;
    }
    /* Slow fallback: full conv (ignores range — caller must be single-threaded). */
    (void)oc_start; (void)oc_end;
    fxp_conv_f32(x, y, n, cin, cout, h, w, w_int, w_scale, bias, x_scale,
                 kh, kw, pad_t, pad_l, pad_b, pad_r,
                 stride_h, stride_w, group, act_bits);
}
