/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * DCVC-SDK demonstration / cross-platform interop tool.
 *
 *   dcvc_demo <model_dir> [width] [height] [qp] [n_frames]
 *       In-process encode -> decode round-trip on synthetic RGB input,
 *       reporting bitstream size and reconstruction PSNR per frame.
 *
 *   dcvc_demo --emit <model_dir> <out.dcv> [width] [height] [qp] [n]
 *       Encode n synthetic frames and write the packets to <out.dcv>
 *       (LE u32 length prefix per packet). Deterministic input + integer
 *       entropy path => byte-identical <out.dcv> across OS/CPU/GPU.
 *
 *   dcvc_demo --replay <model_dir> <in.dcv>
 *       Read packets back and decode each, reporting type + decode status.
 *       A packet produced on Windows decodes here (and vice versa).
 *
 * Build:  cc dcvc_demo.c -o dcvc_demo -ldcvc
 */
#include <dcvc/dcvc.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- deterministic synthetic frame [1,3,H,W] FP32 RGB in [0,1] ------------ *
 * Integer-only PRNG so the input is byte-identical on every platform (this is
 * what makes --emit a valid cross-OS bit-exactness probe). */
static float* synth_frame(int h, int w, unsigned seed) {
    float* p = (float*)malloc((size_t)3 * h * w * sizeof(float));
    if (!p) return NULL;
    unsigned s = seed;
    for (int i = 0; i < 3 * h * w; i++) {
        s = s * 1103515245u + 12345u;
        p[i] = (float)((s >> 16) & 1023) / 1023.0f;
    }
    return p;
}

static double rgb_psnr(const float* a, const float* b, long n) {
    double mse = 0.0;
    for (long i = 0; i < n; i++) { double d = (double)a[i] - (double)b[i]; mse += d * d; }
    mse /= (double)n;
    return mse <= 1e-12 ? 99.0 : 10.0 * log10(1.0 / mse);
}

static void on_log(dcvc_log_level_t lvl, const char* msg, void* user) {
    (void)user;
    const char* tag[] = {"ERROR", "WARN", "INFO", "DEBUG"};
    fprintf(stderr, "[dcvc %s] %s\n", tag[(int)lvl], msg);
}

/* little-endian helpers so the .dcv file is identical on any host */
static void put_u32le(uint8_t* p, unsigned int v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static unsigned int get_u32le(const uint8_t* p) {
    return ((unsigned)p[0]) | ((unsigned)p[1] << 8) |
           ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}

/* ========================================================================= *
 *  In-process round-trip                                                     *
 * ========================================================================= */
static int run_roundtrip(const char* model_dir, int W, int H, int qp, int nframes) {
    printf("DCVC-SDK %s  (%dx%d qp=%d, %d frames)\n",
           dcvc_version_string(), W, H, qp, nframes);

    dcvc_config_t* cfg = dcvc_config_create();
    if (dcvc_config_set_model_dir(cfg, model_dir) != DCVC_OK) {
        fprintf(stderr, "model dir not set\n"); return 1;
    }
    dcvc_session_t* sess = NULL;
    dcvc_status_t st = dcvc_session_create(cfg, &sess);
    dcvc_config_destroy(cfg);
    if (st != DCVC_OK) { fprintf(stderr, "session create: %s\n", dcvc_status_string(st)); return 1; }

    dcvc_encoder_t* enc = NULL;
    st = dcvc_encoder_create(sess, W, H, qp, &enc);
    if (st != DCVC_OK) {
        fprintf(stderr, "encoder create: %s\n", dcvc_session_last_error(sess));
        dcvc_session_destroy(sess); return 1;
    }
    dcvc_decoder_t* dec = NULL;
    dcvc_decoder_create(sess, &dec);

    long px = (long)3 * W * H;
    double total_bits = 0.0, total_psnr = 0.0;
    printf("frame  type  bytes   bpp    psnr(dB)\n");
    int dec_count = 0;
    for (int f = 0; f < nframes; f++) {
        float* frame = synth_frame(H, W, (unsigned)(f * 7919 + 1));
        uint8_t* pkt = NULL; size_t pkt_size = 0; dcvc_frame_type_t ftype;
        st = dcvc_encoder_encode(enc, frame, (f == 0), &pkt, &pkt_size, &ftype);
        if (st != DCVC_OK) { fprintf(stderr, "encode frame %d: %s\n", f, dcvc_status_string(st)); free(frame); break; }
        if (pkt_size == 0) { free(frame); continue; }  /* HT: buffered inter frame */

        int pw = 0, ph = 0; dcvc_frame_type_t pt;
        dcvc_packet_probe(pkt, pkt_size, &pw, &ph, NULL, &pt);
        /* HT: an inter packet decodes to DCVC_FRAME_DELAY frames. */
        int nframes_pkt = (pt == DCVC_FRAME_INTRA) ? 1 : DCVC_FRAME_DELAY;
        float* rec = (float*)malloc((size_t)3 * pw * ph * nframes_pkt * sizeof(float));
        dcvc_status_t dst = dcvc_decoder_decode(dec, pkt, pkt_size, rec, NULL, NULL, &pt);
        if (dst == DCVC_OK) {
            for (int j = 0; j < nframes_pkt; j++) {
                double psnr = rgb_psnr(frame, rec + (size_t)j*3*pw*ph, px);
                total_psnr += psnr; dec_count++;
                printf("%4d   %s  %6zu  %5.3f  %7.3f\n", f + j,
                       pt == DCVC_FRAME_INTRA ? "I" : "P", pkt_size,
                       8.0 * pkt_size / (double)(W * H * nframes_pkt), psnr);
            }
            total_bits += 8.0 * pkt_size;
        }
        dcvc_packet_free(pkt); free(rec); free(frame);
    }
    nframes = dec_count > 0 ? dec_count : nframes;
    printf("avg bpp=%.4f  avg psnr=%.3f dB\n",
           total_bits / ((double)nframes * W * H), total_psnr / nframes);

    dcvc_decoder_destroy(dec);
    dcvc_encoder_destroy(enc);
    dcvc_session_destroy(sess);
    return 0;
}

/* ========================================================================= *
 *  --emit / --replay (cross-OS interop)                                      *
 * ========================================================================= *
 *  .dcv format: a sequence of framed packets, each:
 *    [u32 LE payload_len][payload_len bytes = a DCV2 packet]
 *  followed by a trailing [u32 LE = crc32 of all payload bytes].
 */
static int run_emit(const char* model_dir, const char* path,
                    int W, int H, int qp, int nframes) {
    dcvc_config_t* cfg = dcvc_config_create();
    dcvc_config_set_model_dir(cfg, model_dir);
    dcvc_session_t* sess = NULL;
    if (dcvc_session_create(cfg, &sess) != DCVC_OK) { dcvc_config_destroy(cfg); return 1; }
    dcvc_config_destroy(cfg);

    dcvc_encoder_t* enc = NULL;
    if (dcvc_encoder_create(sess, W, H, qp, &enc) != DCVC_OK) {
        fprintf(stderr, "encoder create: %s\n", dcvc_session_last_error(sess));
        dcvc_session_destroy(sess); return 1;
    }

    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    unsigned long crc = 0xffffffffu; size_t total = 0;
    for (int fr = 0; fr < nframes; fr++) {
        float* frame = synth_frame(H, W, (unsigned)(fr * 7919 + 1));
        uint8_t* pkt = NULL; size_t n = 0; dcvc_frame_type_t t;
        dcvc_status_t st = dcvc_encoder_encode(enc, frame, (fr == 0), &pkt, &n, &t);
        if (st != DCVC_OK) { fprintf(stderr, "encode %d: %s\n", fr, dcvc_status_string(st)); free(frame); fclose(f); return 1; }
        if (n == 0) { dcvc_packet_free(pkt); free(frame); continue; }  /* HT: buffered inter frame */

        uint8_t hdr[4]; put_u32le(hdr, (unsigned)n);
        fwrite(hdr, 1, 4, f);
        fwrite(pkt, 1, n, f);
        for (size_t i = 0; i < n; i++)
            crc = (crc >> 8) ^ 0xedb88320u ^ ((crc ^ pkt[i]) & 0xff);  /* crc32 step */
        total += n;
        printf("emit %2d  %s  %zu bytes\n", fr, t == DCVC_FRAME_INTRA ? "I" : "P", n);
        dcvc_packet_free(pkt); free(frame);
    }
    unsigned c = (unsigned)(crc ^ 0xffffffffu);
    uint8_t cb[4]; put_u32le(cb, c);
    fwrite(cb, 1, 4, f);
    fclose(f);
    printf("wrote %s : %zu payload bytes, crc32=%08x\n", path, total, c);

    dcvc_encoder_destroy(enc);
    dcvc_session_destroy(sess);
    return 0;
}

static int run_replay(const char* model_dir, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);

    dcvc_config_t* cfg = dcvc_config_create();
    dcvc_config_set_model_dir(cfg, model_dir);
    dcvc_session_t* sess = NULL;
    if (dcvc_session_create(cfg, &sess) != DCVC_OK) { dcvc_config_destroy(cfg); fclose(f); return 1; }
    dcvc_config_destroy(cfg);
    dcvc_decoder_t* dec = NULL;
    dcvc_decoder_create(sess, &dec);

    int fr = 0, ok = 1;
    while (ftell(f) < fsz - 4) {        /* last 4 bytes = trailing crc */
        uint8_t hb[4];
        if (fread(hb, 1, 4, f) != 4) break;
        unsigned n = get_u32le(hb);
        if (n == 0 || n > 100000000u) { fprintf(stderr, "bad frame length %u\n", n); ok = 0; break; }
        uint8_t* pkt = (uint8_t*)malloc(n);
        if (fread(pkt, 1, n, f) != n) { fprintf(stderr, "truncated frame %d\n", fr); free(pkt); ok = 0; break; }

        int W = 0, H = 0; dcvc_frame_type_t t;
        dcvc_packet_probe(pkt, n, &W, &H, NULL, &t);
        int npkt = (t == DCVC_FRAME_INTRA) ? 1 : DCVC_FRAME_DELAY;
        float* rec = (float*)malloc((size_t)3 * W * H * npkt * sizeof(float));
        dcvc_status_t st = dcvc_decoder_decode(dec, pkt, n, rec, &W, &H, &t);
        const char* res = (st == DCVC_OK) ? "OK" : dcvc_status_string(st);
        printf("replay %2d  %s  %ux%u  %d frames  %s\n", fr,
               t == DCVC_FRAME_INTRA ? "I" : "P", (unsigned)W, (unsigned)H, npkt, res);
        if (st != DCVC_OK) ok = 0;
        free(rec); free(pkt); fr++;
    }
    fclose(f);
    dcvc_decoder_destroy(dec);
    dcvc_session_destroy(sess);
    printf("replayed %d packet(s): %s\n", fr, ok ? "all OK" : "errors above");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    dcvc_set_log_callback(on_log, NULL);

    if (argc >= 2 && strcmp(argv[1], "--emit") == 0) {
        if (argc < 4) { fprintf(stderr, "usage: --emit <model_dir> <out.dcv> [W] [H] [qp] [n]\n"); return 1; }
        return run_emit(argv[2], argv[3],
                        argc > 4 ? atoi(argv[4]) : 256,
                        argc > 5 ? atoi(argv[5]) : 256,
                        argc > 6 ? atoi(argv[6]) : 32,
                        argc > 7 ? atoi(argv[7]) : 3);
    }
    if (argc >= 2 && strcmp(argv[1], "--replay") == 0) {
        if (argc < 4) { fprintf(stderr, "usage: --replay <model_dir> <in.dcv>\n"); return 1; }
        return run_replay(argv[2], argv[3]);
    }

    const char* model_dir = argc > 1 ? argv[1] : "models";
    int W = argc > 2 ? atoi(argv[2]) : 256;
    int H = argc > 3 ? atoi(argv[3]) : 256;
    int qp = argc > 4 ? atoi(argv[4]) : 32;
    int nframes = argc > 5 ? atoi(argv[5]) : 3;
    return run_roundtrip(model_dir, W, H, qp, nframes);
}
