// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// SDK-level integration test: encode an I→P→P sequence via dcvc_rt_encoder
// and decode via dcvc_rt_decoder, exercising the inter pipeline wiring.
// Verifies frame types, successful round-trips, and valid output dimensions.

#include "dcvc_rt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_yuv(uint8_t* y, uint8_t* u, uint8_t* v, int W, int H, int frame)
{
    for (int j = 0; j < H; j++)
        for (int i = 0; i < W; i++)
            y[j*W+i] = (uint8_t)((i*2 + j + frame*17) & 0xff);
    for (int j = 0; j < H/2; j++)
        for (int i = 0; i < W/2; i++) {
            u[j*(W/2)+i] = (uint8_t)(128 + ((i+j+frame) & 0x3f));
            v[j*(W/2)+i] = (uint8_t)(128 - ((i-j-frame) & 0x3f));
        }
}

int main(void)
{
    const int W = 256, H = 256;
    DcvcRtConfig cfg;
    dcvc_rt_config_init(&cfg);
    cfg.width = W; cfg.height = H; cfg.qp = 32;
    cfg.reset_interval = 64;
    cfg.asset_dir = "native/assets";

    DcvcRtStatus st;
    DcvcRtEncoder* enc = dcvc_rt_encoder_create(&cfg, &st);
    if (!enc) { fprintf(stderr, "encoder create: %s\n", dcvc_rt_status_string(st)); return 1; }
    DcvcRtDecoder* dec = dcvc_rt_decoder_create(&cfg, &st);
    if (!dec) { fprintf(stderr, "decoder create: %s\n", dcvc_rt_status_string(st)); return 1; }
    printf("SDK encoder+decoder created (256x256)\n");

    uint8_t *y = malloc(W*H), *u = malloc(W/2*H/2), *v = malloc(W/2*H/2);
    DcvcRtFrame fr = {0};
    fr.width = W; fr.height = H; fr.format = DCVC_RT_FMT_YUV420P;
    fr.data[0] = y; fr.data[1] = u; fr.data[2] = v;
    fr.stride[0] = W; fr.stride[1] = W/2; fr.stride[2] = W/2;

    const char* expect_type[3] = { "I", "P", "P" };
    int expect_ft[3] = { DCVC_RT_FRAME_I, DCVC_RT_FRAME_P, DCVC_RT_FRAME_P };
    int ok = 1;

    for (int f = 0; f < 3; f++) {
        fill_yuv(y, u, v, W, H, f);

        DcvcRtPacket pkt = {0};
        st = dcvc_rt_encode_frame(enc, &fr, &pkt);
        if (st != DCVC_RT_OK) {
            fprintf(stderr, "frame %d encode failed: %s\n", f, dcvc_rt_status_string(st));
            ok = 0; break;
        }
        printf("frame %d: type=%s size=%zu bytes\n", f,
               pkt.frame_type == DCVC_RT_FRAME_I ? "I" : "P", pkt.size);
        if (pkt.frame_type != expect_ft[f]) {
            fprintf(stderr, "frame %d: expected %s\n", f, expect_type[f]);
            ok = 0;
        }

        DcvcRtFrame out = {0};
        st = dcvc_rt_decode_packet(dec, pkt.data, pkt.size, &out);
        if (st != DCVC_RT_OK) {
            fprintf(stderr, "frame %d decode failed: %s\n", f, dcvc_rt_status_string(st));
            ok = 0; free(pkt.data); break;
        }
        if (out.width != W || out.height != H || !out.data[0]) {
            fprintf(stderr, "frame %d: bad output %dx%d\n", f, out.width, out.height);
            ok = 0;
        }
        dcvc_rt_frame_free_planes(&out);
        free(pkt.data);
    }

    printf("\n*** SDK I→P→P: %s ***\n", ok ? "PASS" : "FAIL");
    free(y); free(u); free(v);
    dcvc_rt_encoder_destroy(enc);
    dcvc_rt_decoder_destroy(dec);
    return ok ? 0 : 1;
}
