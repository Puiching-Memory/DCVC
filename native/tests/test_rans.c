#include "rans_c.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    /* Minimal CDF: single symbol distribution */
    const int cdf_num = 1;
    const int per = 4; /* cdf entries: 0, mid, end, pad */
    int32_t cdfs[4] = {0, 16384, 65536, 65536};
    int32_t sizes[1] = {3};
    int32_t offsets[1] = {0};

    DcvcRansEncoder* enc = dcvc_rans_encoder_create();
    DcvcRansDecoder* dec = dcvc_rans_decoder_create();
    if (!enc || !dec) {
        fprintf(stderr, "create failed\n");
        return 1;
    }

    int gi = dcvc_rans_encoder_add_cdf(enc, cdfs, cdf_num, per, sizes, offsets);
    int gj = dcvc_rans_decoder_add_cdf(dec, cdfs, cdf_num, per, sizes, offsets);
    if (gi < 0 || gj < 0 || gi != gj) {
        fprintf(stderr, "add_cdf failed\n");
        return 1;
    }

    /* Combined symbol: high byte = value, low byte = cdf index */
    int16_t symbols[8];
    for (int i = 0; i < 8; i++) {
        int8_t v = (int8_t)(i - 4);
        symbols[i] = (int16_t)(((int16_t)v << 8) | 0);
    }

    dcvc_rans_encoder_encode_y(enc, symbols, 8, gi);
    dcvc_rans_encoder_flush(enc);

    uint8_t* stream = NULL;
    size_t stream_n = 0;
    if (dcvc_rans_encoder_get_stream(enc, &stream, &stream_n) != 0 || stream_n == 0) {
        fprintf(stderr, "get_stream failed\n");
        return 1;
    }

    uint8_t indexes[8];
    memset(indexes, 0, sizeof(indexes));
    dcvc_rans_decoder_set_stream(dec, stream, stream_n);
    dcvc_rans_decoder_decode_y(dec, indexes, 8, gj);

    int8_t* out = NULL;
    size_t out_n = 0;
    if (dcvc_rans_decoder_get_symbols(dec, &out, &out_n) != 0) {
        fprintf(stderr, "get_symbols failed\n");
        return 1;
    }
    if (out_n != 8) {
        fprintf(stderr, "unexpected out_n=%zu\n", out_n);
        return 1;
    }
    for (int i = 0; i < 8; i++) {
        int8_t expect = (int8_t)(i - 4);
        if (out[i] != expect) {
            fprintf(stderr, "symbol[%d]=%d expect %d\n", i, out[i], expect);
            free(stream);
            free(out);
            return 1;
        }
    }

    free(stream);
    free(out);
    dcvc_rans_encoder_destroy(enc);
    dcvc_rans_decoder_destroy(dec);
    printf("test_rans OK\n");
    return 0;
}
