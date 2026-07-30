/* End-to-end I+P performance on DCVC-RK .rknn pack (UF-LD).
 *
 *   test_rknn_e2e <model_dir> [H] [W] [qp] [n_frames]
 *
 * Frame 0 = I; frames 1..n-1 = P.
 * Prints wall/NPU summary + stage breakdown (wall / npu / set / get / cpu).
 * Set DCVC_PROFILE=0 to suppress stage tables.
 */
#include "dcvc_rk/intra_pipeline.h"
#include "dcvc_rk/inter_ld_pipeline.h"
#include "dcvc_rk/profile.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_profile = 1;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static float* synth_rgb(int H, int W, int seed)
{
    float* p = (float*)malloc(3 * (size_t)H * W * sizeof(float));
    if (!p) return NULL;
    for (int i = 0; i < 3 * H * W; i++) {
        unsigned v = (unsigned)((i * 9301 + 49297 + seed * 17) % 2048);
        p[i] = (float)v / 2048.0f;
    }
    return p;
}

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

static void dump_prof(const DcvcRkProfile* p, const DcvcRkProfile* ar, const char* title)
{
    if (!g_profile || !p) return;
    dcvc_rk_profile_print(p, title);
    if (ar && ar->n > 0) {
        char buf[128];
        snprintf(buf, sizeof(buf), "%s / AR detail", title);
        dcvc_rk_profile_print(ar, buf);
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model_dir> [H=1088] [W=1920] [qp=32] [n_frames=3]\n",
                argv[0]);
        return 1;
    }
    const char* model_dir = argv[1];
    int H = argc > 2 ? atoi(argv[2]) : 1088;
    int W = argc > 3 ? atoi(argv[3]) : 1920;
    int qp = argc > 4 ? atoi(argv[4]) : 32;
    int n_frames = argc > 5 ? atoi(argv[5]) : 3;
    if (n_frames < 1) n_frames = 1;
    {
        const char* e = getenv("DCVC_PROFILE");
        if (e && e[0] == '0') g_profile = 0;
    }

    printf("DCVC-RK e2e  model=%s  %dx%d  qp=%d  frames=%d\n",
           model_dir, H, W, qp, n_frames);

    DcvcRkStatus st;
    double t0 = now_ms();
    DcvcRkIntraPipeline* intra = dcvc_rk_intra_create(model_dir, H, W, qp, &st);
    double t_load_i = now_ms() - t0;
    if (!intra) {
        fprintf(stderr, "intra create failed: %s\n", dcvc_rk_status_string(st));
        return 2;
    }
    printf("intra load: %.1f ms\n", t_load_i);

    float* x0 = synth_rgb(H, W, 0);
    float* xhat = (float*)malloc(3 * (size_t)H * W * sizeof(float));
    if (!x0 || !xhat) return 3;

    uint8_t* stream = NULL; size_t stream_sz = 0;
    dcvc_rk_intra_reset_npu_us(intra);
    t0 = now_ms();
    st = dcvc_rk_intra_encode(intra, x0, &stream, &stream_sz, xhat);
    double enc_ms = now_ms() - t0;
    if (st != DCVC_RK_OK) {
        fprintf(stderr, "I encode failed: %s\n", dcvc_rk_status_string(st));
        return 4;
    }
    double psnr_e = rgb_psnr(x0, xhat, 3 * H * W);
    double bpp = (stream_sz * 8.0) / ((double)H * W);
    double npu_e = dcvc_rk_intra_npu_us(intra) / 1000.0;
    printf("I encode: wall=%.1f ms  npu=%.1f ms  non_npu=%.1f ms  bytes=%zu  bpp=%.4f  recon_psnr=%.2f dB\n",
           enc_ms, npu_e, enc_ms - npu_e, stream_sz, bpp, psnr_e);
    dump_prof(dcvc_rk_intra_last_profile(intra),
              dcvc_rk_intra_last_ar_profile(intra), "I encode");

    memset(xhat, 0, 3 * (size_t)H * W * sizeof(float));
    dcvc_rk_intra_reset_npu_us(intra);
    t0 = now_ms();
    st = dcvc_rk_intra_decode(intra, stream, stream_sz, xhat);
    double dec_ms = now_ms() - t0;
    if (st != DCVC_RK_OK) {
        fprintf(stderr, "I decode failed: %s\n", dcvc_rk_status_string(st));
        return 5;
    }
    double psnr_d = rgb_psnr(x0, xhat, 3 * H * W);
    double npu_d = dcvc_rk_intra_npu_us(intra) / 1000.0;
    printf("I decode: wall=%.1f ms  npu=%.1f ms  non_npu=%.1f ms  psnr=%.2f dB\n",
           dec_ms, npu_d, dec_ms - npu_d, psnr_d);
    dump_prof(dcvc_rk_intra_last_profile(intra),
              dcvc_rk_intra_last_ar_profile(intra), "I decode");

    /* Keep I recon as P reference; free I engines before loading P to save RAM. */
    float* ref = (float*)malloc(3 * (size_t)H * W * sizeof(float));
    memcpy(ref, xhat, 3 * (size_t)H * W * sizeof(float));
    free(stream); stream = NULL;
    dcvc_rk_intra_destroy(intra); intra = NULL;

    if (n_frames <= 1) {
        free(x0); free(xhat); free(ref);
        printf("done (I-only)\n");
        return 0;
    }

    t0 = now_ms();
    DcvcRkInterLdPipeline* inter = dcvc_rk_inter_create(model_dir, H, W, qp, &st);
    double t_load_p = now_ms() - t0;
    if (!inter) {
        fprintf(stderr, "inter create failed: %s\n", dcvc_rk_status_string(st));
        free(x0); free(xhat); free(ref);
        return 6;
    }
    printf("inter load: %.1f ms\n", t_load_p);

    double sum_enc = 0, sum_npu = 0;
    int np = 0;
    for (int f = 1; f < n_frames; f++) {
        float* xf = synth_rgb(H, W, f);
        int reset = (f == 1);
        dcvc_rk_inter_reset_npu_us(inter);
        t0 = now_ms();
        st = dcvc_rk_inter_encode(inter, xf, reset, reset ? ref : NULL,
                                  &stream, &stream_sz, xhat);
        double ems = now_ms() - t0;
        if (st != DCVC_RK_OK) {
            fprintf(stderr, "P%d encode failed: %s\n", f, dcvc_rk_status_string(st));
            free(xf); break;
        }
        double psnr = rgb_psnr(xf, xhat, 3 * H * W);
        double pbpp = (stream_sz * 8.0) / ((double)H * W);
        double pnpu = dcvc_rk_inter_npu_us(inter) / 1000.0;
        printf("P%d encode: wall=%.1f ms  npu=%.1f ms  non_npu=%.1f ms  bytes=%zu  bpp=%.4f  psnr=%.2f dB\n",
               f, ems, pnpu, ems - pnpu, stream_sz, pbpp, psnr);
        {
            char title[32];
            snprintf(title, sizeof(title), "P%d encode", f);
            dump_prof(dcvc_rk_inter_last_profile(inter),
                      dcvc_rk_inter_last_ar_profile(inter), title);
        }
        sum_enc += ems;
        sum_npu += pnpu;
        np++;

        memcpy(ref, xhat, 3 * (size_t)H * W * sizeof(float));
        free(stream); stream = NULL;
        free(xf);
    }

    if (np > 0) {
        printf("P avg encode: wall=%.1f ms (%.2f fps)  npu=%.1f ms  non_npu=%.1f ms\n",
               sum_enc / np, 1000.0 / (sum_enc / np), sum_npu / np,
               (sum_enc - sum_npu) / np);
    }

    dcvc_rk_inter_destroy(inter);
    free(x0); free(xhat); free(ref);
    printf("done\n");
    return 0;
}
