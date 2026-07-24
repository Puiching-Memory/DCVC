/* Micro-benchmark: optimized FXP kernels vs frozen pre-opt baseline.
 *
 * Usage:  bench_fxp_conv [iters]
 *
 * Baseline and opt are separate .c TUs compiled with the same -O3 -mavx2.
 */
#include "fxp/fxp_conv.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif
void fxp_baseline_conv1x1_f32(const float* x, float* y,
                              int n, int cin, int cout, int h, int w,
                              const int16_t* w_int, const float* w_scale,
                              const float* bias, float x_scale, int act_bits);
void fxp_baseline_dw3x3_f32(const float* x, float* y,
                            int n, int c, int h, int w,
                            const int16_t* w_int, const float* w_scale,
                            const float* bias, float x_scale, int act_bits);
#ifdef __cplusplus
}
#endif

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void fill(float* p, int n, unsigned seed)
{
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = ((int)(s >> 8) % 2001 - 1000) / 1000.0f;
    }
}

static void fill_i16(int16_t* p, int n, unsigned seed)
{
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (int16_t)((int)(s >> 8) % 20001 - 10000);
    }
}

typedef void (*conv1x1_fn)(const float*, float*, int, int, int, int, int,
                           const int16_t*, const float*, const float*, float, int);

static double time_1x1(conv1x1_fn fn, float* x, float* y,
                       int cin, int cout, int h, int w,
                       const int16_t* wt, const float* ws, const float* b,
                       float x_scale, int iters)
{
    fn(x, y, 1, cin, cout, h, w, wt, ws, b, x_scale, 16);
    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        fn(x, y, 1, cin, cout, h, w, wt, ws, b, x_scale, 16);
    double t1 = now_sec();
    return 1e3 * (t1 - t0) / (double)iters;
}

static void bench_1x1(int cin, int cout, int h, int w, int iters)
{
    const int hw = h * w;
    float* x = (float*)malloc((size_t)cin * hw * sizeof(float));
    float* y = (float*)malloc((size_t)cout * hw * sizeof(float));
    int16_t* wt = (int16_t*)malloc((size_t)cout * cin * sizeof(int16_t));
    float* ws = (float*)malloc((size_t)cout * sizeof(float));
    float* bias = (float*)malloc((size_t)cout * sizeof(float));
    fill(x, cin * hw, 1);
    fill_i16(wt, cout * cin, 2);
    for (int i = 0; i < cout; i++) {
        ws[i] = 1e-4f;
        bias[i] = 0.f;
    }
    const float xs = 0.02f;
    double ms_base = time_1x1(fxp_baseline_conv1x1_f32, x, y, cin, cout, h, w,
                              wt, ws, bias, xs, iters);
    double ms_opt = time_1x1(fxp_conv1x1_f32, x, y, cin, cout, h, w,
                             wt, ws, bias, xs, iters);
    double speedup = ms_base / ms_opt;
    double gops = (2.0 * (double)cin * (double)cout * (double)hw) / (ms_opt * 1e6);
    printf("1x1  cin=%4d cout=%4d %2dx%2d | baseline %7.3f ms | opt %7.3f ms | "
           "speedup %.2fx | opt ~%.2f GOPS\n",
           cin, cout, h, w, ms_base, ms_opt, speedup, gops);
    free(x);
    free(y);
    free(wt);
    free(ws);
    free(bias);
}

static void bench_dw(int c, int h, int w, int iters)
{
    const int hw = h * w;
    float* x = (float*)malloc((size_t)c * hw * sizeof(float));
    float* y = (float*)malloc((size_t)c * hw * sizeof(float));
    int16_t* wt = (int16_t*)malloc((size_t)c * 9 * sizeof(int16_t));
    float* ws = (float*)malloc((size_t)c * sizeof(float));
    float* bias = (float*)malloc((size_t)c * sizeof(float));
    fill(x, c * hw, 3);
    fill_i16(wt, c * 9, 4);
    for (int i = 0; i < c; i++) {
        ws[i] = 1e-4f;
        bias[i] = 0.f;
    }
    const float xs = 0.02f;

    fxp_baseline_dw3x3_f32(x, y, 1, c, h, w, wt, ws, bias, xs, 16);
    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        fxp_baseline_dw3x3_f32(x, y, 1, c, h, w, wt, ws, bias, xs, 16);
    double t1 = now_sec();
    double ms_base = 1e3 * (t1 - t0) / (double)iters;

    fxp_conv_f32(x, y, 1, c, c, h, w, wt, ws, bias, xs, 3, 3, 1, 1, 1, 1, 1, 1, c, 16);
    t0 = now_sec();
    for (int i = 0; i < iters; i++)
        fxp_conv_f32(x, y, 1, c, c, h, w, wt, ws, bias, xs, 3, 3, 1, 1, 1, 1, 1, 1, c, 16);
    t1 = now_sec();
    double ms_opt = 1e3 * (t1 - t0) / (double)iters;
    printf("dw3x3 c=%4d %2dx%2d | baseline %7.3f ms | opt %7.3f ms | speedup %.2fx\n",
           c, h, w, ms_base, ms_opt, ms_base / ms_opt);
    free(x);
    free(y);
    free(wt);
    free(ws);
    free(bias);
}

int main(int argc, char** argv)
{
    int iters = 80;
    if (argc >= 2)
        iters = atoi(argv[1]);
    if (iters < 1)
        iters = 1;
    printf("bench_fxp_conv iters=%d\n", iters);
    printf("baseline = pre-opt (NCHW/strided); opt = pack+tile+TLS\n");
    printf("Both .c TUs: -O3 -mavx2 (fair layout comparison)\n\n");
    bench_1x1(256, 512, 16, 16, iters);
    bench_1x1(512, 512, 16, 16, iters);
    bench_1x1(1024, 512, 16, 16, iters);
    bench_1x1(512, 2048, 16, 16, iters);
    bench_dw(512, 16, 16, iters);
    return 0;
}
