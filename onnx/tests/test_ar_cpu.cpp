/* Test pure-CPU AR encode/decode loop using ONNX prior models and C rANS. */
#include "cpu_ar_codec.h"
#include "npy_reader.h"
#include "model_dir.h"
#include "onnx_engine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "console_pause.h"

static int read_f32_npy_dims(const char* path, float** out_data, int dims[4])
{
    DcvcNpy n;
    if (dcvc_npy_read(path, &n) != 0) return -1;
    if (n.dtype != DCVC_NPY_F32) { dcvc_npy_free(&n); return -1; }
    for (int i = 0; i < 4; i++) dims[i] = (i < n.ndims) ? n.dims[i] : 1;
    *out_data = (float*)malloc(n.elems * sizeof(float));
    if (!*out_data) { dcvc_npy_free(&n); return -1; }
    memcpy(*out_data, dcvc_npy_f32(&n), n.elems * sizeof(float));
    dcvc_npy_free(&n);
    return 0;
}

/* Synthetic NCHW float tensor. Returns pointer to C*H*W floats; caller frees. */
static float* synth_tensor(int c, int h, int w)
{
    float* p = (float*)malloc(c * h * w * sizeof(float));
    if (!p) return NULL;
    for (int i = 0; i < c * h * w; i++) {
        float v = ((float)((i * 9301 + 49297) % 2048) / 2048.0f - 0.5f) * 2.0f;
        p[i] = v;
    }
    return p;
}

int main(int argc, char** argv)
{
    const char* model_dir = dcvc_resolve_model_dir(argc > 1 ? argv[1] : NULL,
                                               "y_spatial_prior_reduction.onnx");
    int H = argc > 2 ? atoi(argv[2]) : 8;
    int W = argc > 3 ? atoi(argv[3]) : 8;
    int n_ch = 256;

    /* Optional real data mode: argv[4]=y.npy argv[5]=params_fusion.npy argv[6]=ref_y_hat.npy */
    const char* y_npy = argc > 4 ? argv[4] : NULL;
    const char* pf_npy = argc > 5 ? argv[5] : NULL;
    const char* ref_npy = argc > 6 ? argv[6] : NULL;

    printf("CPU AR codec test: model_dir=%s H=%d W=%d n_ch=%d\n", model_dir, H, W, n_ch);

    DcvcCpuStatus st;
    DcvcCpuArCodec* c = dcvc_cpu_ar_codec_create(model_dir, n_ch, &st);
    if (!c) {
        fprintf(stderr, "create failed: %s\n", dcvc_cpu_status_string(st));
        { dcvc_pause_if_dblclick(); return 1; }
    }

    float *y = NULL, *pf = NULL;
    float* y_hat = NULL;
    float* y_hat_dec = NULL;
    float* y_ref = NULL;
    int y_dims[4] = {0}, pf_dims[4] = {0};

    if (y_npy && pf_npy) {
        if (read_f32_npy_dims(y_npy, &y, y_dims) != 0) { fprintf(stderr, "failed to read %s\n", y_npy); { dcvc_pause_if_dblclick(); return 1; } }
        if (read_f32_npy_dims(pf_npy, &pf, pf_dims) != 0) { fprintf(stderr, "failed to read %s\n", pf_npy); { dcvc_pause_if_dblclick(); return 1; } }
        n_ch = y_dims[1];
        H = y_dims[2];
        W = y_dims[3];
        printf("loaded real data: y %dx%dx%dx%d, pf %dx%dx%dx%d\n",
               y_dims[0], y_dims[1], y_dims[2], y_dims[3],
               pf_dims[0], pf_dims[1], pf_dims[2], pf_dims[3]);
    } else {
        y = synth_tensor(n_ch, H, W);
        pf = synth_tensor(2 * n_ch + 2, H, W);
    }

    y_hat = (float*)malloc(n_ch * H * W * sizeof(float));
    y_hat_dec = (float*)malloc(n_ch * H * W * sizeof(float));

    if (!y || !pf || !y_hat || !y_hat_dec) {
        fprintf(stderr, "oom\n");
        { dcvc_pause_if_dblclick(); return 1; }
    }

    uint8_t* stream = NULL;
    size_t stream_size = 0;

    st = dcvc_cpu_ar_codec_encode_y(c, y, pf, H, W, &stream, &stream_size, y_hat);
    if (st != DCVC_CPU_OK) {
        fprintf(stderr, "encode failed: %s\n", dcvc_cpu_status_string(st));
        { dcvc_pause_if_dblclick(); return 1; }
    }
    printf("encode OK: stream_size=%zu bytes\n", stream_size);

    st = dcvc_cpu_ar_codec_decode_y(c, pf, H, W, stream, stream_size, y_hat_dec);
    if (st != DCVC_CPU_OK) {
        fprintf(stderr, "decode failed: %s\n", dcvc_cpu_status_string(st));
        { dcvc_pause_if_dblclick(); return 1; }
    }
    printf("decode OK\n");

    double max_diff = 0, sum_diff = 0;
    for (int i = 0; i < n_ch * H * W; i++) {
        double d = fabs((double)y_hat[i] - y_hat_dec[i]);
        if (d > max_diff) max_diff = d;
        sum_diff += d;
    }
    printf("encode-decode y_hat max_diff=%.6f avg_diff=%.8f\n", max_diff, sum_diff / (n_ch * H * W));
    if (max_diff > 1e-4) {
        fprintf(stderr, "ERROR: y_hat mismatch too large\n");
        { dcvc_pause_if_dblclick(); return 1; }
    }

    if (ref_npy) {
        int ref_dims[4] = {0};
        if (read_f32_npy_dims(ref_npy, &y_ref, ref_dims) != 0) { fprintf(stderr, "failed to read %s\n", ref_npy); { dcvc_pause_if_dblclick(); return 1; } }
        double ref_max = 0, ref_sum = 0;
        for (int i = 0; i < n_ch * H * W; i++) {
            double d = fabs((double)y_hat[i] - y_ref[i]);
            if (d > ref_max) ref_max = d;
            ref_sum += d;
        }
        printf("vs Python reference y_hat max_diff=%.6f avg_diff=%.8f\n", ref_max, ref_sum / (n_ch * H * W));
    }

    free(y); free(pf); free(y_hat); free(y_hat_dec); free(stream); free(y_ref);
    dcvc_cpu_ar_codec_destroy(c);
    printf("PASS\n");
    { dcvc_pause_if_dblclick(); return 0; }
}
