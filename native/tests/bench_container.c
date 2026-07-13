/* Micro-benchmark for NAL container encode/decode (no NN). */
#include "dcvc_rt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char** argv)
{
    int frames = 100;
    int w = 1920, h = 1080;
    if (argc > 1) frames = atoi(argv[1]);

    DcvcRtConfig cfg;
    dcvc_rt_config_init(&cfg);
    cfg.width = w;
    cfg.height = h;
    cfg.asset_dir = "native/assets";

    DcvcRtStatus st;
    DcvcRtEncoder* enc = dcvc_rt_encoder_create(&cfg, &st);
    DcvcRtDecoder* dec = dcvc_rt_decoder_create(&cfg, &st);
    if (!enc || !dec) return 1;

    size_t ysz = (size_t)w * h;
    size_t usz = (size_t)(w / 2) * (h / 2);
    uint8_t* y = (uint8_t*)malloc(ysz);
    uint8_t* u = (uint8_t*)malloc(usz);
    uint8_t* v = (uint8_t*)malloc(usz);
    memset(y, 16, ysz);
    memset(u, 128, usz);
    memset(v, 128, usz);

    DcvcRtFrame fr = {0};
    fr.width = w;
    fr.height = h;
    fr.format = DCVC_RT_FMT_YUV420P;
    fr.data[0] = y;
    fr.data[1] = u;
    fr.data[2] = v;
    fr.stride[0] = w;
    fr.stride[1] = w / 2;
    fr.stride[2] = w / 2;

    clock_t t0 = clock();
    for (int i = 0; i < frames; i++) {
        DcvcRtPacket pkt = {0};
        if (dcvc_rt_encode_frame(enc, &fr, &pkt) != DCVC_RT_OK) return 1;
        DcvcRtFrame out = {0};
        if (dcvc_rt_decode_packet(dec, pkt.data, pkt.size, &out) != DCVC_RT_OK) return 1;
        dcvc_rt_frame_free_planes(&out);
        dcvc_rt_packet_free(&pkt);
    }
    clock_t t1 = clock();
    double sec = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("bench_container: %d frames %dx%d in %.3fs (%.1f FPS) [container only, no NN]\n",
           frames, w, h, sec, frames / (sec > 0 ? sec : 1e-9));

    free(y);
    free(u);
    free(v);
    dcvc_rt_encoder_destroy(enc);
    dcvc_rt_decoder_destroy(dec);
    return 0;
}
