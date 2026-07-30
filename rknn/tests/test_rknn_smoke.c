/* Smoke: load one .rknn and run once with zeros. */
#include "dcvc_rk/rknn_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.rknn [C] [H] [W]\n", argv[0]);
        return 1;
    }
    int C = argc > 2 ? atoi(argv[2]) : 3;
    int H = argc > 3 ? atoi(argv[3]) : 1088;
    int W = argc > 4 ? atoi(argv[4]) : 1920;
    DcvcRkStatus st;
    DcvcRkEngine* e = dcvc_rk_engine_create(argv[1], &st);
    if (!e) { fprintf(stderr, "create failed: %s\n", dcvc_rk_status_string(st)); return 1; }
    float* in = (float*)calloc((size_t)C * H * W, sizeof(float));
    /* Guess single output same spatial — query via a large buffer; for smoke just
     * use analysis-like: out y is 256x68x120. Caller should pass right dims. */
    int oc = argc > 5 ? atoi(argv[5]) : 256;
    int oh = argc > 6 ? atoi(argv[6]) : H / 16;
    int ow = argc > 7 ? atoi(argv[7]) : W / 16;
    float* out = (float*)calloc((size_t)oc * oh * ow, sizeof(float));
    if (!in || !out) return 1;
    DcvcRkTensorView vin = { in, 1, C, H, W };
    DcvcRkTensorView vout = { out, 1, oc, oh, ow };
    /* Many models have 2 inputs; smoke only works for 1-in if n_input==1.
     * For multi-input models this will fail — that's OK for smoke. */
    st = dcvc_rk_engine_run(e, &vin, 1, &vout, 1);
    printf("run=%s last_us=%lld\n", dcvc_rk_status_string(st),
           (long long)dcvc_rk_engine_last_run_us(e));
    dcvc_rk_engine_destroy(e);
    free(in); free(out);
    return st == DCVC_RK_OK ? 0 : 2;
}
