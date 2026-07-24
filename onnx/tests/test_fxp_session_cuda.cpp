/* Session-level FXP: CPU EP vs CUDA EP on the same FXP ONNX nets.
 *
 * Requires GPU ORT (-DDCVC_ORT_GPU=ON) and DCVC_FXP_CUDA.
 *
 * Usage:
 *   test_fxp_session_cuda [model_dir] [H] [W]
 * Defaults: auto-detect models_fxp, H=W=16
 */
#include "cpu_intra_pipeline.h"
#include "model_dir.h"
#include "onnx_engine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_rand(float* p, int n, unsigned seed)
{
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = ((int)(s >> 8) % 2001 - 1000) / 500.0f;
    }
}

static int run_net(const char* path, int use_gpu,
                   float* x, int n, int c, int h, int w,
                   float* y, int oc, int oh, int ow)
{
    DcvcCpuStatus st = DCVC_CPU_OK;
    DcvcCpuEngine* eng = dcvc_cpu_engine_create(path, use_gpu, &st);
    if (!eng) {
        fprintf(stderr, "failed to load %s (gpu=%d): %s\n",
                path, use_gpu, dcvc_cpu_status_string(st));
        return 1;
    }
    DcvcCpuTensorView in = { x, 0, n, c, h, w };
    DcvcCpuTensorView out = { y, 0, n, oc, oh, ow };
    st = dcvc_cpu_engine_run(eng, &in, 1, &out, 1);
    dcvc_cpu_engine_destroy(eng);
    if (st != DCVC_CPU_OK) {
        fprintf(stderr, "run failed %s: %s\n", path, dcvc_cpu_status_string(st));
        return 1;
    }
    return 0;
}

static void diff_stats(const float* a, const float* b, int n,
                       float* max_abs, int* n_exact)
{
    float m = 0.f;
    int exact = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
        if (a[i] == b[i]) exact++;
    }
    *max_abs = m;
    *n_exact = exact;
}

static int cuda_ep_available(void)
{
    return dcvc_cpu_cuda_ep_usable();
}

static int test_one_net(const char* model_dir, const char* name,
                        int cin, int cout, int ih, int iw, int oh, int ow,
                        int require_exact)
{
    char path[1100];
    snprintf(path, sizeof(path), "%s/%s.onnx", model_dir, name);
    const int n_in = cin * ih * iw;
    const int n_out = cout * oh * ow;
    float* x = (float*)malloc((size_t)n_in * sizeof(float));
    float* y0 = (float*)malloc((size_t)n_out * sizeof(float));
    float* y1 = (float*)malloc((size_t)n_out * sizeof(float));
    if (!x || !y0 || !y1) return 1;
    fill_rand(x, n_in, 12345u + (unsigned)cin);

    if (run_net(path, 0, x, 1, cin, ih, iw, y0, cout, oh, ow) ||
        run_net(path, 1, x, 1, cin, ih, iw, y1, cout, oh, ow)) {
        free(x); free(y0); free(y1);
        return 1;
    }

    float max_abs = 0.f;
    int n_exact = 0;
    diff_stats(y0, y1, n_out, &max_abs, &n_exact);
    int ok = require_exact ? (max_abs == 0.f) : (max_abs < 1e-3f);
    printf("%-32s in=%dx%d out=%dx%d  max|Δ|=%.6g  exact=%d/%d  %s\n",
           name, ih, iw, oh, ow, max_abs, n_exact, n_out, ok ? "PASS" : "FAIL");
    free(x); free(y0); free(y1);
    return ok ? 0 : 1;
}

static int test_e2e_cuda(const char* model_dir, int H, int W)
{
    setenv("DCVC_USE_GPU", "1", 1);

    float* x = (float*)malloc((size_t)3 * H * W * sizeof(float));
    float* x_hat_enc = (float*)malloc((size_t)3 * H * W * sizeof(float));
    float* x_hat_dec = (float*)malloc((size_t)3 * H * W * sizeof(float));
    if (!x || !x_hat_enc || !x_hat_dec) return 1;
    for (int i = 0; i < 3 * H * W; i++)
        x[i] = ((float)((i * 9301 + 49297) % 2048) / 2048.0f);

    DcvcCpuStatus st = DCVC_CPU_OK;
    DcvcCpuIntraPipeline* p = dcvc_cpu_intra_pipeline_create(model_dir, H, W, 32, &st);
    if (!p) {
        fprintf(stderr, "pipeline create failed: %s\n", dcvc_cpu_status_string(st));
        free(x); free(x_hat_enc); free(x_hat_dec);
        unsetenv("DCVC_USE_GPU");
        return 1;
    }

    uint8_t* stream = nullptr;
    size_t stream_size = 0;
    st = dcvc_cpu_intra_pipeline_encode(p, x, &stream, &stream_size, x_hat_enc);
    if (st != DCVC_CPU_OK) {
        fprintf(stderr, "encode failed: %s\n", dcvc_cpu_status_string(st));
        dcvc_cpu_intra_pipeline_destroy(p);
        free(x); free(x_hat_enc); free(x_hat_dec);
        unsetenv("DCVC_USE_GPU");
        return 1;
    }
    st = dcvc_cpu_intra_pipeline_decode(p, stream, stream_size, x_hat_dec);
    free(stream);
    if (st != DCVC_CPU_OK) {
        fprintf(stderr, "decode failed: %s\n", dcvc_cpu_status_string(st));
        dcvc_cpu_intra_pipeline_destroy(p);
        free(x); free(x_hat_enc); free(x_hat_dec);
        unsetenv("DCVC_USE_GPU");
        return 1;
    }

    double max_diff = 0;
    for (int i = 0; i < 3 * H * W; i++) {
        double d = fabs((double)x_hat_enc[i] - (double)x_hat_dec[i]);
        if (d > max_diff) max_diff = d;
    }
    int ok = max_diff == 0.0;
    printf("e2e CUDA enc↔dec %dx%d stream=%zu  x_hat max_diff=%.6g  %s\n",
           H, W, stream_size, max_diff, ok ? "PASS" : "FAIL");

    dcvc_cpu_intra_pipeline_destroy(p);
    free(x); free(x_hat_enc); free(x_hat_dec);
    unsetenv("DCVC_USE_GPU");
    return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    const char* model_dir = (argc >= 2) ? argv[1] : NULL;
    int H = 16, W = 16;
    if (argc >= 3) H = atoi(argv[2]);
    if (argc >= 4) W = atoi(argv[3]);
    if (!model_dir)
        model_dir = dcvc_resolve_model_dir(NULL, "y_prior_fusion.onnx");
    if (!model_dir) {
        fprintf(stderr, "model dir not found\n");
        return 2;
    }
    printf("model_dir=%s\n", model_dir);

    if (!cuda_ep_available()) {
        fprintf(stderr, "CUDAExecutionProvider not in this ORT build. "
                        "Configure with -DDCVC_ORT_GPU=ON.\n");
        return 3;
    }
    printf("CUDAExecutionProvider: available\n");

    int fail = 0;
    /* Single FxpConv — expect bit-exact CPU vs CUDA session. */
    fail |= test_one_net(model_dir, "y_spatial_prior_reduction",
                         514, 256, H, W, H, W, 1);
    /* Graphs with residual Add may show tiny FP EP noise. */
    fail |= test_one_net(model_dir, "y_prior_fusion",
                         256, 514, H, W, H, W, 0);
    int zh = H / 4, zw = W / 4;
    if (zh < 1) zh = 1;
    if (zw < 1) zw = 1;
    fail |= test_one_net(model_dir, "hyper_dec",
                         128, 256, zh, zw, H, W, 0);

    fail |= test_e2e_cuda(model_dir, 128, 128);

    printf(fail ? "OVERALL FAIL\n" : "OVERALL PASS\n");
    return fail ? 1 : 0;
}
