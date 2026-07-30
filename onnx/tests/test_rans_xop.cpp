/* rANS interoperability test: encode/decode a symbol stream against a CDF.
 *   test_rans_xop encode <cdf.npy> <len.npy> <symbols.i16bin> <n> <out.bin> [group]
 *   test_rans_xop decode <cdf.npy> <len.npy> <stream.bin> <n> <out.i8bin> [group]
 * symbols are int16 (symbol<<8)|cdf_idx; decode returns int8 symbols.
 */
#include "rans_c.h"
#include "npy_reader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static long read_file(const char* path, void** out) {
    FILE* f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    *out = malloc(sz ? sz : 1); long r = fread(*out, 1, sz, f); fclose(f);
    return r == sz ? sz : -1;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s encode|decode ...\n", argv[0]); return 1; }
    if (strcmp(argv[1], "encode") == 0) {
        if (argc < 6) { fprintf(stderr, "encode <cdf.npy> <len.npy> <symbols.i16bin> <n> <out.bin> [group]\n"); return 1; }
        DcvcNpy cdf, len; int group = argc > 7 ? atoi(argv[7]) : 0;
        if (dcvc_npy_read(argv[2], &cdf) || dcvc_npy_read(argv[3], &len)) { fprintf(stderr, "cdf/len read fail\n"); return 1; }
        int n = atoi(argv[5]);
        void* sbuf = nullptr; long got = read_file(argv[4], &sbuf);
        if (got < 0 || got < (long)(n * (long)sizeof(int16_t))) { fprintf(stderr, "symbols read fail\n"); return 1; }
        DcvcRansEncoder* e = dcvc_rans_encoder_create();
        int cdf_num = cdf.dims[0], pvs = cdf.dims[1];
        std::vector<int32_t> off(cdf_num, 0);
        dcvc_rans_encoder_add_cdf(e, dcvc_npy_i32(&cdf), cdf_num, pvs, dcvc_npy_i32(&len), off.data());
        dcvc_rans_encoder_encode_y(e, (const int16_t*)sbuf, n, group);
        dcvc_rans_encoder_flush(e);
        uint8_t* buf = nullptr; size_t sz = 0;
        dcvc_rans_encoder_get_stream(e, &buf, &sz);
        FILE* f = fopen(argv[6], "wb"); fwrite(buf, 1, sz, f); fclose(f);
        fprintf(stderr, "encoded %zu bytes\n", sz);
        free(buf); free(sbuf); dcvc_rans_encoder_destroy(e); return 0;
    }
    if (strcmp(argv[1], "decode") == 0) {
        if (argc < 6) { fprintf(stderr, "decode <cdf.npy> <len.npy> <stream.bin> <n> <out.i8bin> [group]\n"); return 1; }
        DcvcNpy cdf, len; int group = argc > 7 ? atoi(argv[7]) : 0;
        if (dcvc_npy_read(argv[2], &cdf) || dcvc_npy_read(argv[3], &len)) { fprintf(stderr, "cdf/len read fail\n"); return 1; }
        void* sbuf = nullptr; long sz = read_file(argv[4], &sbuf);
        if (sz < 0) { fprintf(stderr, "stream read fail\n"); return 1; }
        int n = atoi(argv[5]);
        DcvcRansDecoder* d = dcvc_rans_decoder_create();
        int cdf_num = cdf.dims[0], pvs = cdf.dims[1];
        std::vector<int32_t> off(cdf_num, 0);
        dcvc_rans_decoder_add_cdf(d, dcvc_npy_i32(&cdf), cdf_num, pvs, dcvc_npy_i32(&len), off.data());
        dcvc_rans_decoder_set_stream(d, (const uint8_t*)sbuf, sz);
        // single-cdf test: every symbol uses cdf index 0
        std::vector<uint8_t> idx(n, 0);
        dcvc_rans_decoder_decode_y(d, idx.data(), n, group);
        int8_t* out = nullptr; size_t outn = 0;
        dcvc_rans_decoder_get_symbols(d, &out, &outn);
        FILE* f = fopen(argv[6], "wb"); fwrite(out, 1, outn, f); fclose(f);
        fprintf(stderr, "decoded %zu symbols\n", outn);
        free(out); free(sbuf); dcvc_rans_decoder_destroy(d); return 0;
    }
    fprintf(stderr, "unknown mode %s\n", argv[1]); return 1;
}
