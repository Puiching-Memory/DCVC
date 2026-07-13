/* C API for DCVC-RT rANS entropy coding (wraps C++ RansEncoderLib). */
#ifndef DCVC_RANS_C_H
#define DCVC_RANS_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DcvcRansEncoder DcvcRansEncoder;
typedef struct DcvcRansDecoder DcvcRansDecoder;

DcvcRansEncoder* dcvc_rans_encoder_create(void);
void dcvc_rans_encoder_destroy(DcvcRansEncoder* enc);
void dcvc_rans_encoder_set_two(DcvcRansEncoder* enc, int use_two);
int dcvc_rans_encoder_add_cdf(DcvcRansEncoder* enc,
                              const int32_t* cdfs, int cdf_num, int per_vector_size,
                              const int32_t* cdf_sizes, const int32_t* offsets);
void dcvc_rans_encoder_encode_y(DcvcRansEncoder* enc, const int16_t* symbols, int n,
                                int cdf_group_index);
void dcvc_rans_encoder_encode_z(DcvcRansEncoder* enc, const int8_t* symbols, int n,
                                int cdf_group_index, int start_offset, int per_channel_size);
void dcvc_rans_encoder_flush(DcvcRansEncoder* enc);
/* Caller frees *out with free() */
int dcvc_rans_encoder_get_stream(DcvcRansEncoder* enc, uint8_t** out, size_t* out_size);
void dcvc_rans_encoder_reset(DcvcRansEncoder* enc);

DcvcRansDecoder* dcvc_rans_decoder_create(void);
void dcvc_rans_decoder_destroy(DcvcRansDecoder* dec);
void dcvc_rans_decoder_set_two(DcvcRansDecoder* dec, int use_two);
int dcvc_rans_decoder_add_cdf(DcvcRansDecoder* dec,
                              const int32_t* cdfs, int cdf_num, int per_vector_size,
                              const int32_t* cdf_sizes, const int32_t* offsets);
void dcvc_rans_decoder_set_stream(DcvcRansDecoder* dec, const uint8_t* data, size_t size);
void dcvc_rans_decoder_decode_y(DcvcRansDecoder* dec, const uint8_t* indexes, int n,
                                int cdf_group_index);
void dcvc_rans_decoder_decode_z(DcvcRansDecoder* dec, int total_size, int cdf_group_index,
                                int start_offset, int per_channel_size);
/* Caller frees *out with free() */
int dcvc_rans_decoder_get_symbols(DcvcRansDecoder* dec, int8_t** out, size_t* out_n);
void dcvc_rans_decoder_reset_cdf(DcvcRansDecoder* dec);

/* PMF → quantized CDF (precision typically 16). out must hold pmf_n+1 ints. */
void dcvc_pmf_to_quantized_cdf(const float* pmf, int pmf_n, int precision, uint32_t* out_cdf);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_RANS_C_H */
