#include "dcvc_rt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    DcvcRtConfig cfg;
    dcvc_rt_config_init(&cfg);
    cfg.width = 64;
    cfg.height = 64;
    cfg.qp = 32;
    cfg.asset_dir = "native/assets";

    DcvcRtStatus st = DCVC_RT_OK;
    DcvcRtEncoder* enc = dcvc_rt_encoder_create(&cfg, &st);
    if (!enc || st != DCVC_RT_OK) {
        fprintf(stderr, "encoder create: %s\n", dcvc_rt_status_string(st));
        return 1;
    }

    uint8_t* y = (uint8_t*)calloc(64 * 64, 1);
    uint8_t* u = (uint8_t*)calloc(32 * 32, 1);
    uint8_t* v = (uint8_t*)calloc(32 * 32, 1);
    memset(y, 128, 64 * 64);
    memset(u, 128, 32 * 32);
    memset(v, 128, 32 * 32);

    DcvcRtFrame fr = {0};
    fr.width = 64;
    fr.height = 64;
    fr.format = DCVC_RT_FMT_YUV420P;
    fr.data[0] = y;
    fr.data[1] = u;
    fr.data[2] = v;
    fr.stride[0] = 64;
    fr.stride[1] = 32;
    fr.stride[2] = 32;

    DcvcRtPacket pkt = {0};
    st = dcvc_rt_encode_frame(enc, &fr, &pkt);
    if (st != DCVC_RT_OK || pkt.size == 0) {
        fprintf(stderr, "encode failed: %s size=%zu\n", dcvc_rt_status_string(st), pkt.size);
        return 1;
    }
    if (!pkt.sps_written || pkt.frame_type != DCVC_RT_FRAME_I) {
        fprintf(stderr, "expected SPS+I\n");
        return 1;
    }

    DcvcRtDecoder* dec = dcvc_rt_decoder_create(&cfg, &st);
    if (!dec) {
        fprintf(stderr, "decoder create failed\n");
        return 1;
    }
    DcvcRtFrame out = {0};
    st = dcvc_rt_decode_packet(dec, pkt.data, pkt.size, &out);
    if (st != DCVC_RT_OK) {
        fprintf(stderr, "decode failed: %s\n", dcvc_rt_status_string(st));
        return 1;
    }
    if (out.width != 64 || out.height != 64 || !out.data[0]) {
        fprintf(stderr, "bad output frame\n");
        return 1;
    }

    /* Second frame as P */
    DcvcRtPacket pkt2 = {0};
    st = dcvc_rt_encode_frame(enc, &fr, &pkt2);
    if (st != DCVC_RT_OK || pkt2.frame_type != DCVC_RT_FRAME_P) {
        fprintf(stderr, "P frame failed\n");
        return 1;
    }

    dcvc_rt_frame_free_planes(&out);
    dcvc_rt_packet_free(&pkt);
    dcvc_rt_packet_free(&pkt2);
    dcvc_rt_encoder_destroy(enc);
    dcvc_rt_decoder_destroy(dec);
    free(y);
    free(u);
    free(v);
    printf("test_roundtrip OK (version %s)\n", dcvc_rt_version());
    return 0;
}
