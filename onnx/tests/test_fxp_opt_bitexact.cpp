/* Bit-exactness: optimized fxp_conv vs frozen pre-optimization golden loops.
 *
 * Usage:  test_fxp_opt_bitexact
 */
#include "fxp/fxp_common.h"
#include "fxp/fxp_conv.h"
#include "fxp/fxp_wsrelu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_rand_f(float* p, int n, unsigned seed)
{
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = ((int)(s >> 8) % 2001 - 1000) / 1000.0f;
    }
}

static void fill_rand_i16(int16_t* p, int n, unsigned seed)
{
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (int16_t)((int)(s >> 8) % 20001 - 10000);
    }
}

/* ---- Golden 1x1 (pre-opt: NCHW quant, strided ic MAC) ---- */
static void golden_conv1x1(const float* x, float* y,
                           int n, int cin, int cout, int h, int w,
                           const int16_t* w_int, const float* w_scale,
                           const float* bias, float x_scale)
{
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

/* ---- Golden DW 3x3 ---- */
static void golden_dw3x3(const float* x, float* y,
                         int n, int c, int h, int w,
                         const int16_t* w_int, const float* w_scale,
                         const float* bias, float x_scale)
{
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

static int mem_identical(const float* a, const float* b, int n)
{
    return memcmp(a, b, (size_t)n * sizeof(float)) == 0;
}

static int test_one_1x1(int cin, int cout, int h, int w, unsigned seed)
{
    const int N = 1;
    const int nx = N * cin * h * w;
    const int ny = N * cout * h * w;
    float* x = (float*)malloc((size_t)nx * sizeof(float));
    float* y0 = (float*)malloc((size_t)ny * sizeof(float));
    float* y1 = (float*)malloc((size_t)ny * sizeof(float));
    int16_t* wt = (int16_t*)malloc((size_t)cout * cin * sizeof(int16_t));
    float* ws = (float*)malloc((size_t)cout * sizeof(float));
    float* b = (float*)malloc((size_t)cout * sizeof(float));
    if (!x || !y0 || !y1 || !wt || !ws || !b)
        return 1;

    fill_rand_f(x, nx, seed);
    fill_rand_i16(wt, cout * cin, seed + 1);
    for (int i = 0; i < cout; i++) {
        ws[i] = 0.0001f * (1 + (i % 7));
        b[i] = 0.001f * (float)i;
    }
    const float x_scale = 0.02f;

    golden_conv1x1(x, y0, N, cin, cout, h, w, wt, ws, b, x_scale);
    fxp_conv1x1_f32(x, y1, N, cin, cout, h, w, wt, ws, b, x_scale, 16);
    int ok = mem_identical(y0, y1, ny);
    printf("1x1 cin=%d cout=%d %dx%d: %s\n", cin, cout, h, w, ok ? "PASS" : "FAIL");

    free(x);
    free(y0);
    free(y1);
    free(wt);
    free(ws);
    free(b);
    return ok ? 0 : 1;
}

static int test_one_dw(int c, int h, int w, unsigned seed)
{
    const int N = 1;
    const int nx = N * c * h * w;
    float* x = (float*)malloc((size_t)nx * sizeof(float));
    float* y0 = (float*)malloc((size_t)nx * sizeof(float));
    float* y1 = (float*)malloc((size_t)nx * sizeof(float));
    int16_t* wt = (int16_t*)malloc((size_t)c * 9 * sizeof(int16_t));
    float* ws = (float*)malloc((size_t)c * sizeof(float));
    float* b = (float*)malloc((size_t)c * sizeof(float));
    if (!x || !y0 || !y1 || !wt || !ws || !b)
        return 1;

    fill_rand_f(x, nx, seed);
    fill_rand_i16(wt, c * 9, seed + 3);
    for (int i = 0; i < c; i++) {
        ws[i] = 0.0002f * (1 + (i % 5));
        b[i] = 0.0005f * (float)i;
    }
    const float x_scale = 0.015f;

    golden_dw3x3(x, y0, N, c, h, w, wt, ws, b, x_scale);
    fxp_conv_f32(x, y1, N, c, c, h, w, wt, ws, b, x_scale,
                 3, 3, 1, 1, 1, 1, 1, 1, c, 16);
    int ok = mem_identical(y0, y1, nx);
    printf("dw3x3 c=%d %dx%d: %s\n", c, h, w, ok ? "PASS" : "FAIL");

    free(x);
    free(y0);
    free(y1);
    free(wt);
    free(ws);
    free(b);
    return ok ? 0 : 1;
}

static int test_wsrelu(void)
{
    const int n = 4096;
    float* x = (float*)malloc((size_t)n * sizeof(float));
    float* y0 = (float*)malloc((size_t)n * sizeof(float));
    float* y1 = (float*)malloc((size_t)n * sizeof(float));
    float* lut = (float*)malloc(65536 * sizeof(float));
    if (!x || !y0 || !y1 || !lut)
        return 1;
    fill_rand_f(x, n, 99);
    for (int i = 0; i < 65536; i++)
        lut[i] = (float)(i - 32768) * 1e-4f;
    const float xs = 0.01f;
    const float inv = 1.0f / xs;
    /* Golden matches fxp_wsrelu_f32 linear-interpolated LUT lookup. */
    for (int i = 0; i < n; i++) {
        float scaled = x[i] * inv;
        if (scaled >= 32767.0f) {
            y0[i] = lut[65535];
        } else if (scaled <= -32768.0f) {
            y0[i] = lut[0];
        } else {
            float fl = floorf(scaled);
            int idx = (int)fl + 32768;
            float frac = scaled - fl;
            y0[i] = lut[idx] + (lut[idx + 1] - lut[idx]) * frac;
        }
    }
    fxp_wsrelu_f32(x, y1, n, lut, xs);
    int ok = mem_identical(y0, y1, n);
    printf("wsrelu: %s\n", ok ? "PASS" : "FAIL");
    free(x);
    free(y0);
    free(y1);
    free(lut);
    return ok ? 0 : 1;
}

int main(void)
{
    int fail = 0;
    fail |= test_one_1x1(8, 4, 3, 3, 42);
    fail |= test_one_1x1(256, 512, 16, 16, 7);
    fail |= test_one_1x1(512, 512, 16, 16, 11);
    fail |= test_one_1x1(1024, 512, 16, 16, 13);
    fail |= test_one_1x1(512, 2048, 16, 16, 17);
    fail |= test_one_dw(16, 8, 8, 21);
    fail |= test_one_dw(64, 16, 16, 23);
    fail |= test_one_dw(512, 16, 16, 29);
    fail |= test_wsrelu();
    printf(fail ? "OVERALL FAIL\n" : "OVERALL PASS\n");
    return fail ? 1 : 0;
}
