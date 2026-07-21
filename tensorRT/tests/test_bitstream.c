#include "dcvc_bitstream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void expect(int cond, const char* msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        failures++;
    }
}

int main(void)
{
    DcvcBitWriter w;
    dcvc_bw_init(&w);

    DcvcSps sps;
    sps.sps_id = 0;
    sps.height = 1080;
    sps.width = 1920;
    sps.ec_part = 1;
    sps.use_ada_i = 1;
    expect(dcvc_bw_write_sps(&w, &sps) == 0, "write_sps");

    uint8_t payload[] = {1, 2, 3, 4, 5};
    expect(dcvc_bw_write_ip(&w, 1, 0, 32, payload, sizeof(payload)) == 0, "write_i");
    expect(dcvc_bw_write_ip(&w, 0, 0, 40, payload, sizeof(payload)) == 0, "write_p");

    DcvcBitReader r;
    dcvc_br_init(&r, w.data, w.size);

    int sps_id = -1;
    expect(dcvc_br_read_header(&r, &sps_id) == DCVC_NAL_SPS, "nal sps");
    DcvcSps sps2;
    expect(dcvc_br_read_sps_remaining(&r, sps_id, &sps2) == 0, "read sps");
    expect(sps2.width == 1920 && sps2.height == 1080, "sps dims");
    expect(sps2.ec_part == 1 && sps2.use_ada_i == 1, "sps flags");

    expect(dcvc_br_read_header(&r, &sps_id) == DCVC_NAL_I, "nal i");
    int qp = 0;
    uint8_t* pl = NULL;
    size_t plen = 0;
    expect(dcvc_br_read_ip_remaining(&r, &qp, &pl, &plen) == 0, "read i");
    expect(qp == 32 && plen == 5 && memcmp(pl, payload, 5) == 0, "i payload");
    free(pl);

    expect(dcvc_br_read_header(&r, &sps_id) == DCVC_NAL_P, "nal p");
    expect(dcvc_br_read_ip_remaining(&r, &qp, &pl, &plen) == 0, "read p");
    expect(qp == 40 && plen == 5, "p payload");
    free(pl);

    dcvc_bw_free(&w);
    if (failures) {
        fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    printf("test_bitstream OK\n");
    return 0;
}
