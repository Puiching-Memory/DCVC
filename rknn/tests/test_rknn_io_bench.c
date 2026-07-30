/* Microbench: compare FP32 vs INT8-passthrough I/O on one .rknn.
 *
 *   test_rknn_io_bench <model.rknn> [loops=20]
 *
 * Runs both modes in-process by re-init (env is read at create).
 * Prints avg wall / npu / set / get (ms).
 */
#include "dcvc_rk/rknn_engine.h"
#include "dcvc_rk/profile.h"

#include <rknn_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_env_io(const char* v)
{
    setenv("DCVC_RKNN_IO", v, 1);
}

static int query_shapes(const char* path, int* n_in, int* n_out,
                        int in_c[], int in_h[], int in_w[],
                        int out_c[], int out_h[], int out_w[])
{
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END); long len = ftell(fp); fseek(fp, 0, SEEK_SET);
    void* buf = malloc((size_t)len);
    if (!buf || fread(buf, 1, (size_t)len, fp) != (size_t)len) {
        free(buf); fclose(fp); return -1;
    }
    fclose(fp);
    rknn_context ctx = 0;
    if (rknn_init(&ctx, buf, (uint32_t)len, 0, NULL) != RKNN_SUCC) {
        free(buf); return -1;
    }
    free(buf);
    rknn_input_output_num io = {0};
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    *n_in = (int)io.n_input; *n_out = (int)io.n_output;
    for (uint32_t i = 0; i < io.n_input; i++) {
        rknn_tensor_attr a = {0}; a.index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &a, sizeof(a));
        /* NHWC */
        in_c[i] = (int)a.dims[3]; in_h[i] = (int)a.dims[1]; in_w[i] = (int)a.dims[2];
    }
    for (uint32_t i = 0; i < io.n_output; i++) {
        rknn_tensor_attr a = {0}; a.index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a));
        /* NCHW */
        out_c[i] = (int)a.dims[1]; out_h[i] = (int)a.dims[2]; out_w[i] = (int)a.dims[3];
    }
    rknn_destroy(ctx);
    return 0;
}

typedef struct {
    double wall, npu, set, get;
} Avg;

static int bench_mode(const char* path, const char* mode, int loops,
                      int n_in, int n_out,
                      int in_c[], int in_h[], int in_w[],
                      int out_c[], int out_h[], int out_w[],
                      float** in_bufs, float** out_bufs, Avg* out)
{
    set_env_io(mode);
    DcvcRkStatus st;
    DcvcRkEngine* eng = dcvc_rk_engine_create(path, &st);
    if (!eng) {
        fprintf(stderr, "create(%s) failed: %s\n", mode, dcvc_rk_status_string(st));
        return -1;
    }
    DcvcRkTensorView* vin = calloc((size_t)n_in, sizeof(*vin));
    DcvcRkTensorView* vout = calloc((size_t)n_out, sizeof(*vout));
    for (int i = 0; i < n_in; i++) {
        vin[i].data = in_bufs[i];
        vin[i].n = 1; vin[i].c = in_c[i]; vin[i].h = in_h[i]; vin[i].w = in_w[i];
    }
    for (int i = 0; i < n_out; i++) {
        vout[i].data = out_bufs[i];
        vout[i].n = 1; vout[i].c = out_c[i]; vout[i].h = out_h[i]; vout[i].w = out_w[i];
    }

    /* warmup */
    st = dcvc_rk_engine_run(eng, vin, n_in, vout, n_out);
    if (st != DCVC_RK_OK) {
        fprintf(stderr, "warmup(%s) failed: %s\n", mode, dcvc_rk_status_string(st));
        dcvc_rk_engine_destroy(eng); free(vin); free(vout);
        return -1;
    }

    double sw = 0, sn = 0, ss = 0, sg = 0;
    for (int i = 0; i < loops; i++) {
        st = dcvc_rk_engine_run(eng, vin, n_in, vout, n_out);
        if (st != DCVC_RK_OK) {
            fprintf(stderr, "run(%s) failed\n", mode);
            dcvc_rk_engine_destroy(eng); free(vin); free(vout);
            return -1;
        }
        sw += dcvc_rk_engine_last_wall_us(eng) / 1000.0;
        sn += dcvc_rk_engine_last_run_us(eng) / 1000.0;
        ss += dcvc_rk_engine_last_set_us(eng) / 1000.0;
        sg += dcvc_rk_engine_last_get_us(eng) / 1000.0;
    }
    out->wall = sw / loops; out->npu = sn / loops;
    out->set = ss / loops; out->get = sg / loops;
    dcvc_rk_engine_destroy(eng);
    free(vin); free(vout);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.rknn [loops=20]\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    int loops = argc > 2 ? atoi(argv[2]) : 20;
    if (loops < 1) loops = 1;

    int n_in = 0, n_out = 0;
    int in_c[8], in_h[8], in_w[8], out_c[8], out_h[8], out_w[8];
    if (query_shapes(path, &n_in, &n_out, in_c, in_h, in_w, out_c, out_h, out_w) != 0) {
        fprintf(stderr, "query failed\n");
        return 2;
    }
    printf("model=%s  in=%d out=%d  loops=%d\n", path, n_in, n_out, loops);
    for (int i = 0; i < n_in; i++)
        printf("  in[%d] NCHW 1x%dx%dx%d\n", i, in_c[i], in_h[i], in_w[i]);
    for (int i = 0; i < n_out; i++)
        printf("  out[%d] NCHW 1x%dx%dx%d\n", i, out_c[i], out_h[i], out_w[i]);

    float* in_bufs[8] = {0}; float* out_bufs[8] = {0};
    for (int i = 0; i < n_in; i++) {
        size_t n = (size_t)in_c[i] * in_h[i] * in_w[i];
        in_bufs[i] = (float*)malloc(n * sizeof(float));
        for (size_t k = 0; k < n; k++) in_bufs[i][k] = ((k * 17) % 1000) / 1000.0f - 0.5f;
    }
    for (int i = 0; i < n_out; i++) {
        size_t n = (size_t)out_c[i] * out_h[i] * out_w[i];
        out_bufs[i] = (float*)calloc(n, sizeof(float));
    }

    Avg a_fp = {0}, a_i8 = {0}, a_nhwc = {0}, a_i8o = {0};
    if (bench_mode(path, "fp32", loops, n_in, n_out, in_c, in_h, in_w, out_c, out_h, out_w,
                   in_bufs, out_bufs, &a_fp) != 0) return 3;
    if (bench_mode(path, "i8", loops, n_in, n_out, in_c, in_h, in_w, out_c, out_h, out_w,
                   in_bufs, out_bufs, &a_i8) != 0) return 4;
    if (bench_mode(path, "nhwc", loops, n_in, n_out, in_c, in_h, in_w, out_c, out_h, out_w,
                   in_bufs, out_bufs, &a_nhwc) != 0) return 5;
    if (bench_mode(path, "i8out", loops, n_in, n_out, in_c, in_h, in_w, out_c, out_h, out_w,
                   in_bufs, out_bufs, &a_i8o) != 0) return 6;

    printf("\n%-6s %8s %8s %8s %8s\n", "mode", "wall", "npu", "set", "get");
    printf("%-6s %7.2f %7.2f %7.2f %7.2f\n", "fp32", a_fp.wall, a_fp.npu, a_fp.set, a_fp.get);
    printf("%-6s %7.2f %7.2f %7.2f %7.2f\n", "i8", a_i8.wall, a_i8.npu, a_i8.set, a_i8.get);
    printf("%-6s %7.2f %7.2f %7.2f %7.2f\n", "nhwc", a_nhwc.wall, a_nhwc.npu, a_nhwc.set, a_nhwc.get);
    printf("%-6s %7.2f %7.2f %7.2f %7.2f\n", "i8out", a_i8o.wall, a_i8o.npu, a_i8o.set, a_i8o.get);
    printf("best wall vs fp32: i8 %+.0f%%  nhwc %+.0f%%  i8out %+.0f%%\n",
           100.0 * (a_i8.wall - a_fp.wall) / a_fp.wall,
           100.0 * (a_nhwc.wall - a_fp.wall) / a_fp.wall,
           100.0 * (a_i8o.wall - a_fp.wall) / a_fp.wall);

    for (int i = 0; i < n_in; i++) free(in_bufs[i]);
    for (int i = 0; i < n_out; i++) free(out_bufs[i]);
    return 0;
}
