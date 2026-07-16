// Performance benchmark: steady-state latency, throughput, GPU memory,
// and bandwidth for the full DCVC-RT encode+decode pipeline.

#include "dcvc_rt.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define W 256
#define H 256

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static size_t gpu_mem_used(void)
{
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    return total - free;
}

/* Replicate test_quality image generation (compact) */
static float lerp_f(float a, float b, float t) { return a + (b - a) * t; }
static float bilinear(const float* g, int gw, float x, float y)
{
    if (x < 0) x = 0; if (x > gw - 1.001f) x = gw - 1.001f;
    if (y < 0) y = 0; if (y > gw - 1.001f) y = gw - 1.001f;
    int x0 = (int)x, y0 = (int)y;
    float fx = x - x0, fy = y - y0;
    return lerp_f(lerp_f(g[y0*gw+x0], g[y0*gw+x0+1], fx),
                  lerp_f(g[(y0+1)*gw+x0], g[(y0+1)*gw+x0+1], fx), fy);
}
static void gen_natural_y(uint8_t* y, int w, int h, uint32_t* seed, float mx, float my)
{
    float* field = calloc(w * h, sizeof(float));
    for (int oct = 0; oct < 5; oct++) {
        int gw = 4 << oct; float* grid = malloc(gw*gw*sizeof(float));
        for (int i = 0; i < gw*gw; i++) grid[i] = (rand_r(seed)/(float)RAND_MAX)*2-1;
        float amp = powf(0.5f, oct) * 0.5f;
        for (int p = 0; p < h; p++)
            for (int q = 0; q < w; q++)
                field[p*w+q] += bilinear(grid, gw, (float)(q+mx)/(w-1)*(gw-1),
                                              (float)(p+my)/(h-1)*(gw-1)) * amp;
        free(grid);
    }
    float mn=1e9f, vmax=-1e9f;
    for (int i = 0; i < w*h; i++) { if(field[i]<mn)mn=field[i]; if(field[i]>vmax)vmax=field[i]; }
    float rg = vmax-mn; if (rg<1e-6f) rg=1.0f;
    for (int i = 0; i < w*h; i++) y[i] = (uint8_t)(16+219.0f*(field[i]-mn)/rg);
    free(field);
}

int main(int argc, char** argv)
{
    int qp = argc > 1 ? atoi(argv[1]) : 32;
    int n_frames = argc > 2 ? atoi(argv[2]) : 50;   /* more frames for stable timing */
    int n_warmup = 3;  /* skip first few for warmup */

    DcvcRtConfig cfg; dcvc_rt_config_init(&cfg);
    cfg.width = W; cfg.height = H; cfg.qp = qp;
    cfg.reset_interval = 64; cfg.asset_dir = "native/assets";
    DcvcRtStatus st;
    DcvcRtEncoder* enc = dcvc_rt_encoder_create(&cfg, &st);
    DcvcRtDecoder* dec = dcvc_rt_decoder_create(&cfg, &st);
    if (!enc || !dec) { fprintf(stderr, "create failed\n"); return 1; }

    size_t mem_baseline = gpu_mem_used();

    /* Pre-generate frames */
    uint8_t** fy = malloc(n_frames * sizeof(void*));
    uint8_t** fu = malloc(n_frames * sizeof(void*));
    uint8_t** fv = malloc(n_frames * sizeof(void*));
    uint32_t seed = 42;
    for (int f = 0; f < n_frames; f++) {
        fy[f] = malloc(W*H); fu[f] = malloc((W/2)*(H/2)); fv[f] = malloc((W/2)*(H/2));
        gen_natural_y(fy[f], W, H, &seed, f*2.0f, f*1.0f);
        memset(fu[f], 128, (W/2)*(H/2)); memset(fv[f], 128, (W/2)*(H/2));
    }

    double *enc_ms = calloc(n_frames, sizeof(double));
    double *dec_ms = calloc(n_frames, sizeof(double));
    double *bytes = calloc(n_frames, sizeof(double));
    int n_p = 0, n_i = 0;

    for (int f = 0; f < n_frames; f++) {
        DcvcRtFrame in = {0};
        in.width=W; in.height=H; in.format=DCVC_RT_FMT_YUV420P;
        in.data[0]=fy[f]; in.data[1]=fu[f]; in.data[2]=fv[f];
        in.stride[0]=W; in.stride[1]=W/2; in.stride[2]=W/2;

        DcvcRtPacket pkt = {0};
        double t0 = now_ms();
        st = dcvc_rt_encode_frame(enc, &in, &pkt);
        enc_ms[f] = now_ms() - t0;
        if (st != DCVC_RT_OK) { fprintf(stderr,"enc %d: %s\n",f,dcvc_rt_status_string(st)); return 1; }
        bytes[f] = (double)pkt.size;
        if (pkt.frame_type == DCVC_RT_FRAME_I) n_i++; else n_p++;

        DcvcRtFrame out = {0};
        double t1 = now_ms();
        st = dcvc_rt_decode_packet(dec, pkt.data, pkt.size, &out);
        dec_ms[f] = now_ms() - t1;
        if (st != DCVC_RT_OK) { fprintf(stderr,"dec %d: %s\n",f,dcvc_rt_status_string(st)); return 1; }

        dcvc_rt_frame_free_planes(&out);
        free(pkt.data);
    }

    size_t mem_peak = gpu_mem_used();

    /* Compute steady-state stats (skip warmup) */
    double e_sum=0, d_sum=0, e_sq=0, d_sq=0, b_sum=0;
    int cnt = 0;
    for (int f = n_warmup; f < n_frames; f++) {
        e_sum += enc_ms[f]; e_sq += enc_ms[f]*enc_ms[f];
        d_sum += dec_ms[f]; d_sq += dec_ms[f]*dec_ms[f];
        b_sum += bytes[f];
        cnt++;
    }
    double e_avg = e_sum/cnt, d_avg = d_sum/cnt;
    double e_std = sqrt(e_sq/cnt - e_avg*e_avg);
    double d_std = sqrt(d_sq/cnt - d_avg*d_avg);
    double bpp = (b_sum*8.0) / ((double)W*H*cnt);
    double eud = e_avg + d_avg;  /* end-to-end single-frame latency */
    double throughput = 1000.0 / e_avg;  /* encode fps (bottleneck) */

    printf("===========================================================\n");
    printf(" DCVC-RT Performance Benchmark\n");
    printf("===========================================================\n");
    printf(" GPU:          %s\n", "NVIDIA A30 (see nvidia-smi)");
    printf(" Resolution:   %dx%d\n", W, H);
    printf(" QP:           %d\n", qp);
    printf(" Frames:       %d total (%d I + %d P), %d warmup discarded\n",
           n_frames, n_i, n_p, n_warmup);
    printf("-----------------------------------------------------------\n");
    printf("                     ENCODE          DECODE\n");
    printf("  Avg latency:    %6.2f ms       %6.2f ms\n", e_avg, d_avg);
    printf("  Std dev:        %6.2f ms       %6.2f ms\n", e_std, d_std);
    printf("  Throughput:     %6.1f fps       %6.1f fps\n", 1000.0/e_avg, 1000.0/d_avg);
    printf("-----------------------------------------------------------\n");
    printf("  End-to-end (enc+dec):  %.2f ms/frame\n", eud);
    printf("  Encode-limited:        %.1f fps\n", throughput);
    printf("  Avg bitrate:           %.4f bpp  (%.0f bytes/P-frame avg)\n",
           bpp, b_sum/cnt);
    printf("  GPU memory (codec):    %.1f MB  (baseline %.1f MB)\n",
           (mem_peak - mem_baseline)/1048576.0, mem_baseline/1048576.0);
    printf("===========================================================\n");

    /* Per-frame detail (compact) */
    printf("\nPer-frame (first 10):\n");
    printf("  f   type  enc_ms  dec_ms   bytes\n");
    for (int f = 0; f < n_frames && f < 10; f++)
        printf("  %-3d %s    %6.1f  %6.1f  %6.0f\n",
               f, f==0?"I":"P", enc_ms[f], dec_ms[f], bytes[f]);

    for (int f = 0; f < n_frames; f++) { free(fy[f]); free(fu[f]); free(fv[f]); }
    free(fy); free(fu); free(fv); free(enc_ms); free(dec_ms); free(bytes);
    dcvc_rt_encoder_destroy(enc); dcvc_rt_decoder_destroy(dec);
    return 0;
}
