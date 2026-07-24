/* Test pure-CPU I+P frame sequence: frame 0 via intra codec, frames 1..N-1 via
 * inter (P-frame) codec. Closed-loop: each P-frame references the previous
 * reconstructed frame. Verifies decoder reconstruction == encoder reconstruction
 * (bit-exact across platforms).
 *
 * Modes:
 *   (no flag) [N=5] [H] [W] [qp_i=32] [qp_p=32]
 *       in-process round-trip of an N-frame synthetic sequence
 *   --encode <npy> <bin> [qp_i] [qp_p]
 *       Read a single (N,3,H,W) float32 npy, encode frame 0 as I-frame and
 *       frames 1..N-1 as P-frames, write a self-describing bitstream to <bin>.
 *   --decode <bin> <npy>
 *       Decode a self-describing bitstream and write the reconstructed
 *       (N,3,H,W) float32 npy.
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
#include "console_pause.h"

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
    if (N < 2) { fprintf(stderr, "need N>=2\n"); return 1; }
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
    uint8_t* s0 = NULL; size_t s0n = 0;
    st = dcvc_cpu_intra_pipeline_encode(intra, x, &s0, &s0n, x_hat_enc);
    if (st) { fprintf(stderr, "intra encode failed\n"); return 1; }
    st = dcvc_cpu_intra_pipeline_decode(intra, s0, s0n, x_hat_dec);
    if (st) { fprintf(stderr, "intra decode failed\n"); return 1; }
    printf("frame 0 (I): stream=%zu B  PSNR=%6.2f dB", s0n, psnr(x, x_hat_enc, 3*H*W));
    { double md=0; for (int i=0;i<3*H*W;i++){double d=fabs(x_hat_enc[i]-x_hat_dec[i]); if(d>md)md=d;} printf("  enc-dec maxdiff=%.2e\n", md); }
    size_t total = s0n; memcpy(ref, x_hat_enc, 3*H*W*sizeof(float));
    free(s0); free(x);
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

/* ------------------------------------------------------------------ */
/* Helpers for the single multi-frame .npy modes.                     */
/* ------------------------------------------------------------------ */
static void set_dump_frame(int f)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%d", f);
#ifdef _WIN32
    char env[80];
    snprintf(env, sizeof(env), "DCVC_DUMP_FRAME=%s", buf);
    _putenv(env);
#else
    setenv("DCVC_DUMP_FRAME", buf, 1);
#endif
}

static float* load_frame(const char* npy, int idx, int dims[3], int* H, int* W)
{
    int ndims = 0, shape[8] = {0};
    if (dcvc_npy_read_meta(npy, &ndims, shape, 8) != 0 || ndims != 4) return NULL;
    if (shape[1] != 3) { fprintf(stderr, "expected 3-channel RGB, got C=%d\n", shape[1]); return NULL; }
    *H = shape[2]; *W = shape[3];
    float* frame = (float*)malloc(3 * (*H) * (*W) * sizeof(float));
    if (!frame) return NULL;
    if (dcvc_npy_read_frame_f32(npy, idx, frame, dims) != 0) { free(frame); return NULL; }
    return frame;
}

static int write_raw_file(const char* path, const void* data, size_t size)
{
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    size_t got = fwrite(data, 1, size, f);
    fclose(f);
    return got == size ? 0 : -1;
}

static int read_raw_file(const char* path, void** out_data, size_t* out_size)
{
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return -1; }
    void* buf = malloc((size_t)sz);
    if (!buf) { fclose(f); return -1; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) { free(buf); return -1; }
    *out_data = buf;
    *out_size = (size_t)sz;
    return 0;
}

/* Encode a multi-frame .npy (N,3,H,W) to a self-describing bitstream:
 *   magic "DCVS" (4)
 *   N (uint32), H (uint32), W (uint32), qp_i (uint32), qp_p (uint32), gop (uint32) (24)
 *   for each frame:
 *       frame_size (uint32)
 *       frame_stream
 *
 * gop <= 0: only frame 0 is an I-frame (legacy single-I layout).
 * gop  > 0: frame f is an I-frame when (f % gop == 0), otherwise a P-frame.
 */
static int mode_encode(int argc, char** argv)
{
    const char* npy = argc > 2 ? argv[2] : NULL;
    const char* bin = argc > 3 ? argv[3] : "dcvc_sequence.bin";
    int qp_i = argc > 4 ? atoi(argv[4]) : 32;
    int qp_p = argc > 5 ? atoi(argv[5]) : 32;
    int gop  = argc > 6 ? atoi(argv[6]) : 0;
    if (!npy) { fprintf(stderr, "usage: --encode <npy> <bin> [qp_i] [qp_p] [gop]\n"); return 1; }

    int ndims = 0, shape[8] = {0};
    if (dcvc_npy_read_meta(npy, &ndims, shape, 8) != 0 || ndims != 4) {
        fprintf(stderr, "%s: expected 4-D npy (N,3,H,W)\n", npy); return 1;
    }
    int N = shape[0], H = shape[2], W = shape[3];
    if (shape[1] != 3) { fprintf(stderr, "expected C=3, got %d\n", shape[1]); return 1; }
    if (N < 1) { fprintf(stderr, "need at least one frame\n"); return 1; }
    if (gop < 0) { fprintf(stderr, "gop must be >= 0 (0 = single I-frame)\n"); return 1; }

    const char* model_dir = resolve_model_dir("intra_analysis_standard.onnx");
    printf("ENCODE sequence: %s N=%d %dx%d qp_i=%d qp_p=%d gop=%d -> %s\n",
           npy, N, H, W, qp_i, qp_p, gop, bin);

    DcvcCpuStatus st;
    DcvcCpuIntraPipeline* intra = dcvc_cpu_intra_pipeline_create(model_dir, H, W, qp_i, &st);
    DcvcCpuInterPipeline* inter = dcvc_cpu_inter_pipeline_create(model_dir, H, W, qp_p, &st);
    if (!intra || !inter) { fprintf(stderr, "pipeline create failed: %s\n", dcvc_cpu_status_string(st)); return 1; }

    FILE* out = fopen(bin, "wb");
    if (!out) { fprintf(stderr, "cannot write %s\n", bin); return 1; }
    fwrite(DCVS_MAGIC, 1, 4, out);
    uint32_t nu = (uint32_t)N, hu = (uint32_t)H, wu = (uint32_t)W;
    uint32_t qiu = (uint32_t)qp_i, qpu = (uint32_t)qp_p;
    uint32_t gopu = (uint32_t)gop;
    fwrite(&nu, 4, 1, out); fwrite(&hu, 4, 1, out); fwrite(&wu, 4, 1, out);
    fwrite(&qiu, 4, 1, out); fwrite(&qpu, 4, 1, out); fwrite(&gopu, 4, 1, out);

    float* ref = (float*)malloc(3 * H * W * sizeof(float));
    float* x_hat = (float*)malloc(3 * H * W * sizeof(float));
    if (!ref || !x_hat) { fprintf(stderr, "oom\n"); return 1; }

    size_t total = 0;
    for (int f = 0; f < N; f++) {
        set_dump_frame(f);
        int is_intra = (gop <= 0) ? (f == 0) : (f % gop == 0);
        int dims[3];
        float* x = load_frame(npy, f, dims, &H, &W);
        if (!x) { fprintf(stderr, "failed to load frame %d\n", f); return 1; }
        uint8_t* stream = NULL; size_t sfn = 0;
        if (is_intra) {
            st = dcvc_cpu_intra_pipeline_encode(intra, x, &stream, &sfn, x_hat);
        } else {
            st = dcvc_cpu_inter_pipeline_encode(inter, x, ref, &stream, &sfn, x_hat);
        }
        free(x);
        if (st) { fprintf(stderr, "encode frame %d failed: %s\n", f, dcvc_cpu_status_string(st)); return 1; }
        uint32_t len = (uint32_t)sfn;
        fwrite(&len, 4, 1, out); fwrite(stream, 1, sfn, out);
        total += sfn;
        printf("frame %d (%s): %zu bytes\n", f, is_intra ? "I" : "P", sfn);
        memcpy(ref, x_hat, 3 * H * W * sizeof(float));
        free(stream);
    }
    fclose(out);
    dcvc_cpu_intra_pipeline_destroy(intra);
    dcvc_cpu_inter_pipeline_destroy(inter);
    free(ref); free(x_hat);
    printf("wrote %s: header=28 bytes, streams=%zu bytes (%.2f KB)\n",
           bin, total, total / 1024.0);
    return 0;
}

/* Decode a self-describing sequence bitstream to a multi-frame .npy. */
static int mode_decode(int argc, char** argv)
{
    const char* bin = argc > 2 ? argv[2] : "dcvc_sequence.bin";
    const char* npy = argc > 3 ? argv[3] : "dcvc_sequence_dec.npy";
    void* raw = NULL; size_t raw_size = 0;
    if (read_raw_file(bin, &raw, &raw_size) != 0) { fprintf(stderr, "failed to read %s\n", bin); return 1; }
    if (raw_size < 28 || memcmp(raw, DCVS_MAGIC, 4) != 0) {
        fprintf(stderr, "%s: not a DCVS container\n", bin); return 1;
    }
    uint32_t N, H, W, qp_i, qp_p, gop;
    memcpy(&N, (char*)raw + 4, 4); memcpy(&H, (char*)raw + 8, 4); memcpy(&W, (char*)raw + 12, 4);
    memcpy(&qp_i, (char*)raw + 16, 4); memcpy(&qp_p, (char*)raw + 20, 4);
    memcpy(&gop, (char*)raw + 24, 4);
    printf("DECODE sequence: %s N=%u %dx%d qp_i=%u qp_p=%u gop=%u -> %s\n",
           bin, N, H, W, qp_i, qp_p, gop, npy);

    const char* model_dir = resolve_model_dir("intra_analysis_standard.onnx");
    DcvcCpuStatus st;
    DcvcCpuIntraPipeline* intra = dcvc_cpu_intra_pipeline_create(model_dir, (int)H, (int)W, (int)qp_i, &st);
    DcvcCpuInterPipeline* inter = dcvc_cpu_inter_pipeline_create(model_dir, (int)H, (int)W, (int)qp_p, &st);
    if (!intra || !inter) { fprintf(stderr, "pipeline create failed: %s\n", dcvc_cpu_status_string(st)); return 1; }

    float* ref = (float*)malloc(3 * H * W * sizeof(float));
    float* x_hat = (float*)malloc(3 * H * W * sizeof(float));
    float* all = (float*)malloc((size_t)N * 3 * H * W * sizeof(float));
    if (!ref || !x_hat || !all) { fprintf(stderr, "oom\n"); return 1; }

    const uint8_t* p = (const uint8_t*)raw + 28;
    size_t remain = raw_size - 28;
    for (uint32_t f = 0; f < N; f++) {
        set_dump_frame((int)f);
        int is_intra = (gop <= 0) ? (f == 0) : (f % gop == 0);
        if (remain < 4) { fprintf(stderr, "truncated: frame %u length missing\n", f); return 1; }
        uint32_t len; memcpy(&len, p, 4); p += 4; remain -= 4;
        if (remain < len) { fprintf(stderr, "truncated: frame %u stream missing\n", f); return 1; }
        if (is_intra) {
            st = dcvc_cpu_intra_pipeline_decode(intra, p, len, x_hat);
        } else {
            st = dcvc_cpu_inter_pipeline_decode(inter, p, len, ref, x_hat);
        }
        if (st) { fprintf(stderr, "decode frame %u failed: %s\n", f, dcvc_cpu_status_string(st)); return 1; }
        memcpy(all + (size_t)f * 3 * H * W, x_hat, 3 * H * W * sizeof(float));
        memcpy(ref, x_hat, 3 * H * W * sizeof(float));
        p += len; remain -= len;
        printf("frame %u (%s): %u bytes\n", f, is_intra ? "I" : "P", len);
    }
    free(raw);
    dcvc_cpu_intra_pipeline_destroy(intra);
    dcvc_cpu_inter_pipeline_destroy(inter);
    free(ref); free(x_hat);

    int dims[4] = { (int)N, 3, (int)H, (int)W };
    if (dcvc_npy_write_f32(npy, all, dims, 4) != 0) { fprintf(stderr, "failed to write %s\n", npy); return 1; }
    free(all);
    printf("DECODE PASS\n");
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

    if (argc > 1 && strcmp(argv[1], "--encode") == 0) { dcvc_pause_if_dblclick(); return mode_encode(argc, argv); }
    if (argc > 1 && strcmp(argv[1], "--decode") == 0) { dcvc_pause_if_dblclick(); return mode_decode(argc, argv); }
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        printf("Usage:\n");
        printf("  test_cpu_inter[.exe] [N] [H] [W] [qp_i] [qp_p]\n");
        printf("      In-process synthetic I+P round-trip (default 5 256 256 32 32).\n");
        printf("  test_cpu_inter[.exe] --encode <npy> <bin> [qp_i] [qp_p] [gop]\n");
        printf("      Encode a single (N,3,H,W) float32 npy to a self-describing bitstream.\n");
        printf("      gop<=0 (default): frame 0 is I-frame, rest are P-frames.\n");
        printf("      gop >0: frame f is I-frame when f%%gop==0 (periodic I refresh).\n");
        printf("  test_cpu_inter[.exe] --decode <bin> <npy>\n");
        printf("      Decode a self-describing bitstream to a (N,3,H,W) float32 npy.\n");
        printf("\n  --model-dir <dir> is optional in all modes (auto-detected otherwise).\n");
        { dcvc_pause_if_dblclick(); return 0; }
    }
    { dcvc_pause_if_dblclick(); return mode_roundtrip(argc, argv); }
}
