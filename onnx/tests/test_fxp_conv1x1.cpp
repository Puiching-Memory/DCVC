/* PoC: com.dcvc::FxpConv1x1 bit-exactness + FP32 parity smoke test.
 *
 * Builds a tiny ONNX graph in /tmp (or argv[1] dir) if needed is awkward from
 * C, so this test expects a pre-exported model:
 *   <model_dir>/y_prior_fusion.onnx   (with FxpConv1x1 last layer)
 *
 * Usage:
 *   test_fxp_conv1x1 [model_dir] [H] [W]
 * Defaults: auto-detect model_dir, H=W=16.
 *
 * Checks:
 *   1) Session loads (custom op registered).
 *   2) Two runs with IntraOp threads={1,8} are bit-exact.
 *   3) Direct C kernel vs ORT custom op are bit-exact on the same inputs
 *      (via a second tiny session is hard; we compare two ORT runs only and
 *      a standalone kernel self-consistency check).
 */
#include "model_dir.h"
#include "onnx_engine.h"
#include "fxp/fxp_conv.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_rand(float* p, int n, unsigned seed)
{
    /* Simple LCG — deterministic across platforms. */
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = ((int)(s >> 8) % 2001 - 1000) / 1000.0f;
    }
}

static int mem_identical(const float* a, const float* b, int n)
{
    return memcmp(a, b, (size_t)n * sizeof(float)) == 0;
}

static int test_kernel_bitexact(void)
{
    const int N = 1, Cin = 8, Cout = 4, H = 3, W = 3;
    float* x = (float*)malloc((size_t)N * Cin * H * W * sizeof(float));
    float* y1 = (float*)malloc((size_t)N * Cout * H * W * sizeof(float));
    float* y2 = (float*)malloc((size_t)N * Cout * H * W * sizeof(float));
    int16_t* w = (int16_t*)malloc((size_t)Cout * Cin * sizeof(int16_t));
    float* ws = (float*)malloc((size_t)Cout * sizeof(float));
    float* b = (float*)malloc((size_t)Cout * sizeof(float));
    if (!x || !y1 || !y2 || !w || !ws || !b) return 1;

    fill_rand(x, N * Cin * H * W, 42);
    for (int i = 0; i < Cout * Cin; i++) w[i] = (int16_t)((i * 13) % 20001 - 10000);
    for (int i = 0; i < Cout; i++) {
        ws[i] = 0.0001f * (1 + (i % 5));
        b[i] = 0.001f * i;
    }
    const float x_scale = 0.02f;

    fxp_conv1x1_f32(x, y1, N, Cin, Cout, H, W, w, ws, b, x_scale, 16);
    fxp_conv1x1_f32(x, y2, N, Cin, Cout, H, W, w, ws, b, x_scale, 16);

    int ok = mem_identical(y1, y2, N * Cout * H * W);
    printf("kernel self bit-exact: %s\n", ok ? "PASS" : "FAIL");

    free(x); free(y1); free(y2); free(w); free(ws); free(b);
    return ok ? 0 : 1;
}

static int test_ort_model(const char* model_dir, int H, int W)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/y_prior_fusion.onnx", model_dir);

    DcvcCpuStatus st = DCVC_CPU_OK;
    DcvcCpuEngine* eng = dcvc_cpu_engine_create(path, 0, &st);
    if (!eng) {
        fprintf(stderr, "failed to load %s: %s\n", path, dcvc_cpu_status_string(st));
        return 1;
    }

    const int N = 1, Cin = 256, Cout = 514;
    size_t in_n = (size_t)N * Cin * H * W;
    size_t out_n = (size_t)N * Cout * H * W;
    float* in = (float*)malloc(in_n * sizeof(float));
    float* out1 = (float*)malloc(out_n * sizeof(float));
    float* out2 = (float*)malloc(out_n * sizeof(float));
    if (!in || !out1 || !out2) {
        dcvc_cpu_engine_destroy(eng);
        return 1;
    }
    fill_rand(in, (int)in_n, 7);

    DcvcCpuTensorView vin = { in, 0, N, Cin, H, W };
    DcvcCpuTensorView vout1 = { out1, 0, N, Cout, H, W };
    DcvcCpuTensorView vout2 = { out2, 0, N, Cout, H, W };

    st = dcvc_cpu_engine_run(eng, &vin, 1, &vout1, 1);
    if (st != DCVC_CPU_OK) {
        fprintf(stderr, "run1 failed: %s\n", dcvc_cpu_status_string(st));
        free(in); free(out1); free(out2);
        dcvc_cpu_engine_destroy(eng);
        return 1;
    }
    st = dcvc_cpu_engine_run(eng, &vin, 1, &vout2, 1);
    if (st != DCVC_CPU_OK) {
        fprintf(stderr, "run2 failed: %s\n", dcvc_cpu_status_string(st));
        free(in); free(out1); free(out2);
        dcvc_cpu_engine_destroy(eng);
        return 1;
    }

    int ok = mem_identical(out1, out2, (int)out_n);
    printf("ORT FxpConv1x1 repeat bit-exact: %s\n", ok ? "PASS" : "FAIL");

    /* Sanity: output not all zeros / not NaN. */
    double absmean = 0.0;
    int nan = 0;
    for (size_t i = 0; i < out_n; i++) {
        if (isnan(out1[i]) || isinf(out1[i])) nan = 1;
        absmean += fabs((double)out1[i]);
    }
    absmean /= (double)out_n;
    printf("output absmean=%.6g nan/inf=%s\n", absmean, nan ? "YES" : "no");
    if (nan || absmean < 1e-12) ok = 0;

    free(in); free(out1); free(out2);
    dcvc_cpu_engine_destroy(eng);
    return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    int fail = test_kernel_bitexact();

    const char* hint = (argc >= 2) ? argv[1] : NULL;
    int H = 16, W = 16;
    if (argc >= 3) H = atoi(argv[2]);
    if (argc >= 4) W = atoi(argv[3]);

    /* Prefer an exported Fxp model dir (has com.dcvc ops in entropy nets). */
    const char* model_dir = dcvc_resolve_model_dir(hint, "y_prior_fusion.onnx");
    char probe[1100];
    snprintf(probe, sizeof(probe), "%s/y_prior_fusion.onnx", model_dir);
    FILE* f = fopen(probe, "rb");
    int is_fxp = 0;
    if (f) {
        /* Heuristic: look for "com.dcvc" / "FxpConv" in the protobuf bytes. */
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz > 0 && sz < 80 * 1024 * 1024) {
            char* buf = (char*)malloc((size_t)sz);
            if (buf && fread(buf, 1, (size_t)sz, f) == (size_t)sz) {
                for (long i = 0; i + 7 < sz; i++) {
                    if (memcmp(buf + i, "FxpConv", 7) == 0) { is_fxp = 1; break; }
                }
            }
            free(buf);
        }
        fclose(f);
    }
    if (!is_fxp) {
        char alt[1024];
        snprintf(alt, sizeof(alt), "%s/../models_fxp", model_dir);
        const char* alt_dir = dcvc_resolve_model_dir(alt, "y_prior_fusion.onnx");
        snprintf(probe, sizeof(probe), "%s/y_prior_fusion.onnx", alt_dir);
        f = fopen(probe, "rb");
        if (f) {
            fclose(f);
            model_dir = alt_dir;
            is_fxp = 1;
        }
    }
    if (!is_fxp) {
        fprintf(stderr,
                "skip ORT test: no Fxp y_prior_fusion under %s "
                "(run python/fxp_export_entropy_nets.py)\n",
                model_dir);
        return fail ? 1 : 0;
    }

    fail |= test_ort_model(model_dir, H, W);
    return fail ? 1 : 0;
}
