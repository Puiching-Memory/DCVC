/* Test pure-CPU intra-frame encode/decode with ONNX + rANS.
 *
 * Modes:
 *   (no flag)                              in-process encode->decode round-trip
 *   --encode <bin> [H] [W] [qp] [x.npy]    encode a frame, write bitstream to <bin>
 *                                          (self-describing: header + stream)
 *   --decode <bin> [ref.npy]               decode bitstream from <bin>, optional compare
 *
 * The --encode/--decode modes enable cross-platform interoperability testing:
 * encode on one OS, copy <bin> to the other, decode there. <bin> is
 * self-describing (magic + H + W + qp + stream), so decode needs no extra args.
 */
#include "cpu_intra_pipeline.h"
#include "model_dir.h"
#include "npy_reader.h"
#include "onnx_engine.h"
#include "rans_c.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "console_pause.h"

/* Self-describing container for the codec bitstream (little-endian).
 *   offset 0: magic "DCV1"
 *   offset 4: H   (uint32)
 *   offset 8: W   (uint32)
 *   offset 12: qp (uint32)
 *   offset 16: codec bitstream ([z_len:u32][z_stream][y_stream])
 *   trailer (optional, DCVC_CRC=1): CRC-32 over offset 16..end-4 (uint32, LE)
 *      Guards the entropy stream against in-transit corruption; the rANS
 *      state machine itself cannot detect a flipped bit in the bytestream.
 */
static const char DCV_MAGIC[4] = {'D', 'C', 'V', '1'};

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

/* Deterministic synthetic RGB image in [0,1]. Uses 32-bit int arithmetic only,
 * so it is byte-identical on Windows (MSVC/MinGW) and Linux. */
static float* synth_image(int h, int w)
{
    float* p = (float*)malloc(3 * h * w * sizeof(float));
    if (!p) return NULL;
    for (int i = 0; i < 3 * h * w; i++) {
        float v = ((float)((i * 9301 + 49297) % 2048) / 2048.0f);
        p[i] = v;
    }
    return p;
}

/* Compute RGB PSNR in dB between two FP32 NCHW [0,1] images. */
static double rgb_psnr(const float* a, const float* b, int n)
{
    double mse = 0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        mse += d * d;
    }
    mse /= n;
    if (mse <= 1e-12) return 99.0;
    return 10.0 * log10(1.0 / mse);
}

/* Optional explicit model directory, parsed from --model-dir in main(). When
 * non-NULL it takes priority over the auto-detected candidate paths. */
static const char* g_model_dir_override = NULL;

static const char* resolve_model_dir(const char* probe_filename)
{
    if (g_model_dir_override)
        return g_model_dir_override;
    return dcvc_resolve_model_dir(NULL, probe_filename);
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

/* ------------------------------------------------------------------ */
/* --encode mode: encode a frame and persist the bitstream.         */
/* ------------------------------------------------------------------ */
static int mode_encode(int argc, char** argv)
{
    const char* bin = argc > 2 ? argv[2] : "dcvc_frame.bin";
    int H = argc > 3 ? atoi(argv[3]) : 256;
    int W = argc > 4 ? atoi(argv[4]) : 256;
    int qp = argc > 5 ? atoi(argv[5]) : 32;
    const char* x_npy = argc > 6 ? argv[6] : NULL;

    const char* model_dir = resolve_model_dir("intra_analysis_standard.onnx");
    printf("ENCODE: model_dir=%s H=%d W=%d qp=%d -> %s\n", model_dir, H, W, qp, bin);

    DcvcCpuStatus st;
    DcvcCpuIntraPipeline* p = dcvc_cpu_intra_pipeline_create(model_dir, H, W, qp, &st);
    if (!p) {
        fprintf(stderr, "pipeline create failed: %s\n", dcvc_cpu_status_string(st));
        return 1;
    }

    float* x = NULL;
    int x_dims[4] = {0};
    if (x_npy) {
        if (read_f32_npy_dims(x_npy, &x, x_dims) != 0) {
            fprintf(stderr, "failed to read %s\n", x_npy);
            return 1;
        }
        if (x_dims[1] != 3 || x_dims[2] != H || x_dims[3] != W) {
            fprintf(stderr, "input shape mismatch: expected 1x3x%dx%d\n", H, W);
            return 1;
        }
    } else {
        x = synth_image(H, W);
    }

    float* x_hat_enc = (float*)malloc(3 * H * W * sizeof(float));
    if (!x || !x_hat_enc) { fprintf(stderr, "oom\n"); return 1; }

    uint8_t* stream = NULL;
    size_t stream_size = 0;
    st = dcvc_cpu_intra_pipeline_encode(p, x, &stream, &stream_size, x_hat_enc);
    if (st != DCVC_CPU_OK) {
        fprintf(stderr, "encode failed: %s\n", dcvc_cpu_status_string(st));
        return 1;
    }
    printf("encode OK: stream_size=%zu bytes\n", stream_size);

    /* Wrap the codec bitstream in the self-describing container. */
    uint32_t hu = (uint32_t)H, wu = (uint32_t)W, qu = (uint32_t)qp;
    int with_crc = getenv("DCVC_CRC") && atoi(getenv("DCVC_CRC")) != 0;
    size_t crc_bytes = with_crc ? 4 : 0;
    size_t total = 16 + stream_size + crc_bytes;
    uint8_t* buf = (uint8_t*)malloc(total);
    memcpy(buf + 0, DCV_MAGIC, 4);
    memcpy(buf + 4, &hu, 4);
    memcpy(buf + 8, &wu, 4);
    memcpy(buf + 12, &qu, 4);
    if (stream_size) memcpy(buf + 16, stream, stream_size);
    if (with_crc) {
        uint32_t crc = dcvc_crc32(buf + 16, stream_size);
        memcpy(buf + 16 + stream_size, &crc, 4);
    }

    if (write_raw_file(bin, buf, total) != 0) {
        fprintf(stderr, "failed to write %s\n", bin);
        return 1;
    }
    printf("wrote %s (%zu bytes, incl. 16-byte header)\n", bin, total);

    /* Save the encoder-side reconstruction so the decode side can compare. */
    char enc_npy[1024];
    snprintf(enc_npy, sizeof(enc_npy), "%s.enc.npy", bin);
    int dims[4] = {1, 3, H, W};
    if (dcvc_npy_write_f32(enc_npy, x_hat_enc, dims, 4) == 0)
        printf("wrote encoder reconstruction %s\n", enc_npy);
    else
        fprintf(stderr, "warning: failed to write %s\n", enc_npy);

    free(x); free(x_hat_enc); free(stream); free(buf);
    dcvc_cpu_intra_pipeline_destroy(p);
    printf("ENCODE PASS\n");
    return 0;
}

/* ------------------------------------------------------------------ */
/* --decode mode: decode a bitstream produced by --encode.           */
/* ------------------------------------------------------------------ */
static int mode_decode(int argc, char** argv)
{
    const char* bin = argc > 2 ? argv[2] : "dcvc_frame.bin";
    const char* ref_npy = argc > 3 ? argv[3] : NULL;

    void* raw = NULL;
    size_t raw_size = 0;
    if (read_raw_file(bin, &raw, &raw_size) != 0) {
        fprintf(stderr, "failed to read %s\n", bin);
        return 1;
    }
    if (raw_size < 16 || memcmp(raw, DCV_MAGIC, 4) != 0) {
        fprintf(stderr, "%s: not a DCV1 container (bad magic or too short)\n", bin);
        return 1;
    }
    uint32_t hu, wu, qu;
    memcpy(&hu, (char*)raw + 4, 4);
    memcpy(&wu, (char*)raw + 8, 4);
    memcpy(&qu, (char*)raw + 12, 4);
    int H = (int)hu, W = (int)wu, qp = (int)qu;
    const uint8_t* stream = (const uint8_t*)raw + 16;
    size_t stream_size = raw_size - 16;
    /* Optional CRC-32 trailer: if the container is large enough and the
     * decoder is asked (DCVC_CRC=1) to expect it, verify before decoding. */
    int with_crc = getenv("DCVC_CRC") && atoi(getenv("DCVC_CRC")) != 0;
    if (with_crc) {
        if (stream_size < 4) {
            fprintf(stderr, "%s: container too small for CRC trailer\n", bin);
            return 1;
        }
        uint32_t stored = 0;
        memcpy(&stored, (const uint8_t*)raw + raw_size - 4, 4);
        uint32_t calc = dcvc_crc32(stream, stream_size - 4);
        if (stored != calc) {
            fprintf(stderr, "%s: bitstream CRC mismatch (stored=%08x calc=%08x)\n",
                    bin, stored, calc);
            return 1;
        }
        stream_size -= 4;
    }

    const char* model_dir = resolve_model_dir("intra_analysis_standard.onnx");
    printf("DECODE: model_dir=%s H=%d W=%d qp=%d stream=%zu bytes <- %s\n",
           model_dir, H, W, qp, stream_size, bin);

    DcvcCpuStatus st;
    DcvcCpuIntraPipeline* p = dcvc_cpu_intra_pipeline_create(model_dir, H, W, qp, &st);
    if (!p) {
        fprintf(stderr, "pipeline create failed: %s\n", dcvc_cpu_status_string(st));
        return 1;
    }

    float* x_hat_dec = (float*)malloc(3 * H * W * sizeof(float));
    if (!x_hat_dec) { fprintf(stderr, "oom\n"); return 1; }

    st = dcvc_cpu_intra_pipeline_decode(p, stream, stream_size, x_hat_dec);
    if (st != DCVC_CPU_OK) {
        fprintf(stderr, "decode failed: %s\n", dcvc_cpu_status_string(st));
        return 1;
    }
    printf("decode OK\n");

    char dec_npy[1024];
    snprintf(dec_npy, sizeof(dec_npy), "%s.dec.npy", bin);
    int dims[4] = {1, 3, H, W};
    if (dcvc_npy_write_f32(dec_npy, x_hat_dec, dims, 4) == 0)
        printf("wrote decoder reconstruction %s\n", dec_npy);
    else
        fprintf(stderr, "warning: failed to write %s\n", dec_npy);

    if (ref_npy) {
        float* ref = NULL;
        int ref_dims[4] = {0};
        if (read_f32_npy_dims(ref_npy, &ref, ref_dims) != 0) {
            fprintf(stderr, "failed to read %s\n", ref_npy);
            return 1;
        }
        double max_diff = 0, sum_diff = 0;
        for (int i = 0; i < 3 * H * W; i++) {
            double d = fabs((double)x_hat_dec[i] - ref[i]);
            if (d > max_diff) max_diff = d;
            sum_diff += d;
        }
        printf("decode-vs-ref x_hat max_diff=%.6f avg_diff=%.8f\n",
               max_diff, sum_diff / (3 * H * W));
        if (max_diff > 1e-4) {
            fprintf(stderr, "ERROR: cross-platform reconstruction mismatch too large\n");
            free(ref); free(x_hat_dec); free(raw);
            dcvc_cpu_intra_pipeline_destroy(p);
            return 1;
        }
        free(ref);
    }

    free(x_hat_dec); free(raw);
    dcvc_cpu_intra_pipeline_destroy(p);
    printf("DECODE PASS\n");
    return 0;
}

/* ------------------------------------------------------------------ */
/* Default mode: in-process encode->decode round-trip.                */
/* ------------------------------------------------------------------ */
static int mode_roundtrip(int argc, char** argv)
{
    const char* model_dir = dcvc_resolve_model_dir(argc > 1 ? argv[1] : NULL,
                                               "intra_analysis_standard.onnx");
    int H = argc > 2 ? atoi(argv[2]) : 256;
    int W = argc > 3 ? atoi(argv[3]) : 256;
    int qp = argc > 4 ? atoi(argv[4]) : 32;
    const char* x_npy = argc > 5 ? argv[5] : NULL;
    const char* out_npy = argc > 6 ? argv[6] : NULL;
    const char* ref_npy = argc > 7 ? argv[7] : NULL;

    printf("CPU end-to-end test: model_dir=%s H=%d W=%d qp=%d\n", model_dir, H, W, qp);
    DcvcCpuStatus st;
    DcvcCpuIntraPipeline* p = dcvc_cpu_intra_pipeline_create(model_dir, H, W, qp, &st);
    if (!p) {
        fprintf(stderr, "pipeline create failed: %s\n", dcvc_cpu_status_string(st));
        return 1;
    }

    float* x = NULL;
    int x_dims[4] = {0};
    if (x_npy) {
        if (read_f32_npy_dims(x_npy, &x, x_dims) != 0) { fprintf(stderr, "failed to read %s\n", x_npy); return 1; }
        if (x_dims[1] != 3 || x_dims[2] != H || x_dims[3] != W) {
            fprintf(stderr, "input shape mismatch: expected 1x3x%dx%d\n", H, W);
            return 1;
        }
    } else {
        x = synth_image(H, W);
    }

    float* x_hat_enc = (float*)malloc(3 * H * W * sizeof(float));
    float* x_hat_dec = (float*)malloc(3 * H * W * sizeof(float));
    if (!x || !x_hat_enc || !x_hat_dec) { fprintf(stderr, "oom\n"); return 1; }

    uint8_t* stream = NULL;
    size_t stream_size = 0;

    st = dcvc_cpu_intra_pipeline_encode(p, x, &stream, &stream_size, x_hat_enc);
    if (st != DCVC_CPU_OK) {
        fprintf(stderr, "encode failed: %s\n", dcvc_cpu_status_string(st));
        return 1;
    }
    printf("encode OK: stream_size=%zu bytes\n", stream_size);

    st = dcvc_cpu_intra_pipeline_decode(p, stream, stream_size, x_hat_dec);
    if (st != DCVC_CPU_OK) {
        fprintf(stderr, "decode failed: %s\n", dcvc_cpu_status_string(st));
        return 1;
    }
    printf("decode OK\n");

    double max_diff = 0, sum_diff = 0;
    for (int i = 0; i < 3 * H * W; i++) {
        double d = fabs((double)x_hat_enc[i] - x_hat_dec[i]);
        if (d > max_diff) max_diff = d;
        sum_diff += d;
    }
    printf("encode-decode x_hat max_diff=%.6f avg_diff=%.8f\n", max_diff, sum_diff / (3 * H * W));
    if (max_diff > 1e-4) {
        fprintf(stderr, "ERROR: x_hat mismatch too large\n");
        return 1;
    }

    /* Report RGB PSNR when a real input image is provided, so users can
     * immediately see quality without post-processing. */
    if (x_npy) {
        printf("RGB PSNR vs input: %.3f dB\n", rgb_psnr(x, x_hat_dec, 3 * H * W));
    }

    if (out_npy) {
        int dims[4] = {1, 3, H, W};
        if (dcvc_npy_write_f32(out_npy, x_hat_dec, dims, 4) == 0)
            printf("saved decoded x_hat to %s\n", out_npy);
        else
            fprintf(stderr, "failed to save %s\n", out_npy);
    }

    if (ref_npy) {
        float* ref = NULL;
        int ref_dims[4] = {0};
        if (read_f32_npy_dims(ref_npy, &ref, ref_dims) != 0) { fprintf(stderr, "failed to read %s\n", ref_npy); return 1; }
        double ref_max = 0, ref_sum = 0;
        for (int i = 0; i < 3 * H * W; i++) {
            double d = fabs((double)x_hat_dec[i] - ref[i]);
            if (d > ref_max) ref_max = d;
            ref_sum += d;
        }
        printf("vs Python reference x_hat max_diff=%.6f avg_diff=%.8f\n", ref_max, ref_sum / (3 * H * W));
        free(ref);
    }

    free(x); free(x_hat_enc); free(x_hat_dec); free(stream);
    dcvc_cpu_intra_pipeline_destroy(p);
    printf("PASS\n");
    return 0;
}

int main(int argc, char** argv)
{
    /* Strip an optional --model-dir <dir> flag from anywhere in argv so it
     * works with every mode. The remaining args keep their original order. */
    char** out = argv + 1;      /* argv[0] (program name) is preserved */
    int new_argc = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            g_model_dir_override = argv[i + 1];
            i++;                 /* consume the value too */
            continue;
        }
        *out++ = argv[i];
        new_argc++;
    }
    *out = NULL;
    argc = new_argc;
    argv[argc] = NULL;

    if (argc > 1 && strcmp(argv[1], "--encode") == 0) { dcvc_pause_if_dblclick(); return mode_encode(argc, argv); }
    if (argc > 1 && strcmp(argv[1], "--decode") == 0) { dcvc_pause_if_dblclick(); return mode_decode(argc, argv); }
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        printf("Usage:\n");
        printf("  test_cpu_end2end [model_dir] [H] [W] [qp] [x.npy] [out.npy] [ref.npy]\n");
        printf("      In-process encode->decode round-trip (default).\n");
        printf("  test_cpu_end2end --model-dir <dir> --encode <bin> [H] [W] [qp] [x.npy]\n");
        printf("      Encode a frame; write self-describing bitstream to <bin> and\n");
        printf("      reconstruction to <bin>.enc.npy. (use for cross-platform testing)\n");
        printf("  test_cpu_end2end --model-dir <dir> --decode <bin> [ref.npy]\n");
        printf("      Decode <bin>; write <bin>.dec.npy and optionally compare to ref.npy.\n");
        printf("\n  --model-dir <dir> is optional in all modes (auto-detected otherwise).\n");
        { dcvc_pause_if_dblclick(); return 0; }
    }
    { dcvc_pause_if_dblclick(); return mode_roundtrip(argc, argv); }
}
