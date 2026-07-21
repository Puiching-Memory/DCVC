/* Test the standard-op intra_analysis ONNX model against golden tensors. */
#include "model_dir.h"
#include "npy_reader.h"
#include "onnx_engine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Load a .npy and convert to FP32 regardless of whether it was f16 or f32.
 * Returns malloc'd buffer (caller frees) or NULL on failure. dims[0..3] out. */
static float* load_as_f32(const char* path, int dims[4])
{
    DcvcNpy n;
    if (dcvc_npy_read(path, &n) != 0) return NULL;
    for (int i = 0; i < 4; i++) dims[i] = (i < n.ndims) ? n.dims[i] : 1;
    float* out = (float*)malloc(n.elems * sizeof(float));
    if (!out) { dcvc_npy_free(&n); return NULL; }
    if (n.dtype == DCVC_NPY_F16) {
        const uint16_t* h = dcvc_npy_f16(&n);
        for (size_t i = 0; i < n.elems; i++) out[i] = dcvc_f16_to_f32(h[i]);
    } else if (n.dtype == DCVC_NPY_F32) {
        const float* f = dcvc_npy_f32(&n);
        memcpy(out, f, n.elems * sizeof(float));
    } else {
        free(out); dcvc_npy_free(&n); return NULL;
    }
    dcvc_npy_free(&n);
    return out;
}

int main(int argc, char** argv)
{
    char model_buf[1024];
    const char* model;
    if (argc > 1) {
        model = argv[1];  /* explicit model path */
    } else {
        /* auto-detect the model directory (in-tree build/ or packaged folder) */
        const char* dir = dcvc_resolve_model_dir(NULL, "intra_analysis_standard.onnx");
        snprintf(model_buf, sizeof(model_buf), "%s/intra_analysis_standard.onnx", dir);
        model = model_buf;
    }
    const char* input_npy = argc > 2 ? argv[2] : "../assets/golden/intra_analysis_input.npy";
    const char* output_npy = argc > 3 ? argv[3] : "../assets/golden/intra_analysis_output.npy";
    const char* quant_npy = argc > 4 ? argv[4] : "../assets/golden/intra_analysis_quant.npy";

    DcvcCpuStatus st;
    DcvcCpuEngine* eng = dcvc_cpu_engine_create(model, 0, &st);
    if (!eng) {
        fprintf(stderr, "create: %s\n", dcvc_cpu_status_string(st));
        return 1;
    }

    int in_dims[4] = {1, 3, 256, 256};
    int out_dims[4] = {0};
    int quant_dims[4] = {1, 368, 1, 1};

    float* in_fp32 = load_as_f32(input_npy, in_dims);
    if (!in_fp32) {
        fprintf(stderr, "failed to read %s, using zeros 1x3x256x256\n", input_npy);
        in_dims[0] = 1; in_dims[1] = 3; in_dims[2] = 256; in_dims[3] = 256;
        size_t n = (size_t)in_dims[0] * in_dims[1] * in_dims[2] * in_dims[3];
        in_fp32 = (float*)calloc(n, sizeof(float));
    }

    float* out_ref = load_as_f32(output_npy, out_dims);
    if (!out_ref) fprintf(stderr, "warning: no golden output %s, will just print stats\n", output_npy);

    float* quant_fp32 = load_as_f32(quant_npy, quant_dims);
    if (!quant_fp32) {
        fprintf(stderr, "warning: no quant file %s, using ones\n", quant_npy);
        quant_dims[0] = 1; quant_dims[1] = 368; quant_dims[2] = 1; quant_dims[3] = 1;
        quant_fp32 = (float*)malloc(368 * sizeof(float));
        for (int i = 0; i < 368; i++) quant_fp32[i] = 1.0f;
    }

    DcvcCpuTensorView inputs[2];
    inputs[0] = { in_fp32, 0, in_dims[0], in_dims[1], in_dims[2], in_dims[3] };
    inputs[1] = { quant_fp32, 0, quant_dims[0], quant_dims[1], quant_dims[2], quant_dims[3] };

    int out_c = 256, out_h = in_dims[2] / 16, out_w = in_dims[3] / 16;
    size_t out_elems = (size_t)1 * out_c * out_h * out_w;
    float* out_buf = (float*)malloc(out_elems * sizeof(float));
    DcvcCpuTensorView output = { out_buf, 0, 1, out_c, out_h, out_w };

    st = dcvc_cpu_engine_run(eng, inputs, 2, &output, 1);
    if (st != DCVC_CPU_OK) {
        fprintf(stderr, "run: %s\n", dcvc_cpu_status_string(st));
        return 1;
    }

    printf("intra_analysis OK: input %dx%dx%dx%d + quant %dx%dx%dx%d -> output %dx%dx%dx%d\n",
           inputs[0].n, inputs[0].c, inputs[0].h, inputs[0].w,
           inputs[1].n, inputs[1].c, inputs[1].h, inputs[1].w,
           output.n, output.c, output.h, output.w);

    if (out_ref) {
        double max_err = 0, sum_err = 0;
        for (size_t i = 0; i < out_elems; i++) {
            float d = fabsf(out_buf[i] - out_ref[i]);
            if (d > max_err) max_err = d;
            sum_err += d;
        }
        printf("golden max_abs_err=%.4f avg_abs_err=%.6f\n", max_err, sum_err / out_elems);
    }

    if (argc > 5) {
        int save_dims[4] = { output.n, output.c, output.h, output.w };
        if (dcvc_npy_write_f16(argv[5], out_buf, save_dims, 4) == 0) {
            printf("saved CPU ONNX y to %s\n", argv[5]);
        } else {
            fprintf(stderr, "failed to save %s\n", argv[5]);
        }
    }

    free(in_fp32); free(out_ref); free(quant_fp32); free(out_buf);
    dcvc_cpu_engine_destroy(eng);
    return 0;
}
