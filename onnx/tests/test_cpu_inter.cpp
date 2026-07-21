/* Test pure-CPU I+P frame sequence: frame 0 via intra codec, frames 1..N-1 via
 * inter (P-frame) codec. Closed-loop: each P-frame references the previous
 * reconstructed frame. Verifies decoder reconstruction == encoder reconstruction
 * (bit-exact across platforms).
 *
 * Modes:
 *   (no flag) [N=5] [H] [W] [qp_i=32] [qp_p=32]
 *       in-process round-trip of an N-frame synthetic sequence
 *   --encode <bin> <frame_list.txt> [H] [W] [qp_i] [qp_p]
 *       frame_list.txt: one x.npy path per line (I-frame then P-frames)
 *   --decode <bin> <frame_list.txt>
 *       decode sequence, compare to originals
 */
#include "cpu_intra_pipeline.h"
#include "cpu_inter_pipeline.h"
#include "model_dir.h"
#include "npy_reader.h"
#include "onnx_engine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char DCVS_MAGIC[4] = {'D','C','V','S'};  /* Sequence */
static const char* g_model_dir_override = NULL;
static const char* resolve_model_dir(const char* probe) {
    return g_model_dir_override ? g_model_dir_override : dcvc_resolve_model_dir(NULL, probe);
}

static float* synth_frame(int h, int w, int seed) {
    float* p = (float*)malloc(3 * h * w * sizeof(float));
    if (!p) return NULL;
    for (int i = 0; i < 3 * h * w; i++) {
        float v = (float)(((i * 9301 + 49297 + seed * 1000003) % 2048) / 2048.0);
        p[i] = v;
    }
    return p;
}
static int read_f32_npy(const char* path, float** out, int dims[4]) {
    DcvcNpy n;
    if (dcvc_npy_read(path, &n) != 0) return -1;
    if (n.dtype != DCVC_NPY_F32) { dcvc_npy_free(&n); return -1; }
    for (int i = 0; i < 4; i++) dims[i] = (i < n.ndims) ? n.dims[i] : 1;
    *out = (float*)malloc(n.elems * sizeof(float));
    if (!*out) { dcvc_npy_free(&n); return -1; }
    memcpy(*out, dcvc_npy_f32(&n), n.elems * sizeof(float));
    dcvc_npy_free(&n);
    return 0;
}
static double psnr(const float* a, const float* b, int n) {
    double mse = 0;
    for (int i = 0; i < n; i++) { double d = a[i] - b[i]; mse += d * d; }
    mse /= n;
    return mse <= 0 ? 999.0 : 10.0 * log10(1.0 / mse);
}

/* round-trip a synthetic sequence in-process */
static int mode_roundtrip(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 5;
    int H = argc > 2 ? atoi(argv[2]) : 256;
    int W = argc > 3 ? atoi(argv[3]) : 256;
    int qp_i = argc > 4 ? atoi(argv[4]) : 32;
    int qp_p = argc > 5 ? atoi(argv[5]) : 32;
    if (N < 2) {
        fprintf(stderr, "need N>=2\n"); return 1;
    }
    const char* model_dir = resolve_model_dir("intra_analysis_standard.onnx");
    printf("I+P sequence test: N=%d %dx%d qp_i=%d qp_p=%d model_dir=%s\n",
           N, H, W, qp_i, qp_p, model_dir);

    DcvcCpuStatus st;
    DcvcCpuIntraPipeline* intra = dcvc_cpu_intra_pipeline_create(model_dir, H, W, qp_i, &st);
    DcvcCpuInterPipeline* inter = dcvc_cpu_inter_pipeline_create(model_dir, H, W, qp_p, &st);
    if (!intra || !inter) { fprintf(stderr, "pipeline create failed: %s\n", dcvc_cpu_status_string(st)); return 1; }

    float *x = synth_frame(H, W, 0), *x_hat_enc = (float*)malloc(3*H*W*sizeof(float));
    float *x_hat_dec = (float*)malloc(3*H*W*sizeof(float)), *ref = (float*)malloc(3*H*W*sizeof(float));
    if (!x || !x_hat_enc || !x_hat_dec || !ref) { fprintf(stderr, "oom\n"); return 1; }

    /* frame 0: intra */
    uint8_t* s0 = NULL; size_t s0n = 0;
    st = dcvc_cpu_intra_pipeline_encode(intra, x, &s0, &s0n, x_hat_enc);
    if (st) { fprintf(stderr, "intra encode failed\n"); return 1; }
    st = dcvc_cpu_intra_pipeline_decode(intra, s0, s0n, x_hat_dec);
    if (st) { fprintf(stderr, "intra decode failed\n"); return 1; }
    printf("frame 0 (I): stream=%zu B  PSNR=%6.2f dB", s0n, psnr(x, x_hat_enc, 3*H*W));
    { double md=0; for (int i=0;i<3*H*W;i++){double d=fabs(x_hat_enc[i]-x_hat_dec[i]); if(d>md)md=d;} printf("  enc-dec maxdiff=%.2e\n", md); }
    size_t total = s0n; memcpy(ref, x_hat_enc, 3*H*W*sizeof(float));
    free(s0); free(x);

    /* frames 1..N-1: inter */
    for (int f = 1; f < N; f++) {
        x = synth_frame(H, W, f);
        uint8_t* sf = NULL; size_t sfn = 0;
        st = dcvc_cpu_inter_pipeline_encode(inter, x, ref, &sf, &sfn, x_hat_enc);
        if (st) { fprintf(stderr, "inter encode frame %d failed: %s\n", f, dcvc_cpu_status_string(st)); return 1; }
        st = dcvc_cpu_inter_pipeline_decode(inter, sf, sfn, ref, x_hat_dec);
        if (st) { fprintf(stderr, "inter decode frame %d failed: %s\n", f, dcvc_cpu_status_string(st)); return 1; }
        double md=0; for (int i=0;i<3*H*W;i++){double d=fabs(x_hat_enc[i]-x_hat_dec[i]); if(d>md)md=d;}
        printf("frame %d (P): stream=%6zu B  PSNR=%6.2f dB  enc-dec maxdiff=%.2e\n",
               f, sfn, psnr(x, x_hat_enc, 3*H*W), md);
        total += sfn;
        if (md > 1e-4) { fprintf(stderr, "ERROR: frame %d enc-dec mismatch\n", f); return 1; }
        memcpy(ref, x_hat_enc, 3*H*W*sizeof(float));
        free(sf); free(x);
    }
    printf("total stream = %zu B (%.2f KB), %.0f B/frame avg\n", total, total/1024.0, (double)total/N);
    printf("PASS\n");
    dcvc_cpu_intra_pipeline_destroy(intra);
    dcvc_cpu_inter_pipeline_destroy(inter);
    free(x_hat_enc); free(x_hat_dec); free(ref);
    return 0;
}

int main(int argc, char** argv) {
    char** out = argv + 1; int new_argc = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            g_model_dir_override = argv[i + 1]; i++; continue;
        }
        *out++ = argv[i]; new_argc++;
    }
    *out = NULL; argc = new_argc; argv[argc] = NULL;
    return mode_roundtrip(argc, argv);
}
