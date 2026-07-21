#ifndef DCVC_BITSTREAM_H
#define DCVC_BITSTREAM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DcvcNalType {
    DCVC_NAL_SPS = 0,
    DCVC_NAL_I = 1,
    DCVC_NAL_P = 2
} DcvcNalType;

typedef struct DcvcSps {
    int sps_id;
    int height;
    int width;
    int ec_part;
    int use_ada_i;
} DcvcSps;

typedef struct DcvcBitWriter {
    uint8_t* data;
    size_t size;
    size_t cap;
} DcvcBitWriter;

typedef struct DcvcBitReader {
    const uint8_t* data;
    size_t size;
    size_t pos;
} DcvcBitReader;

void dcvc_bw_init(DcvcBitWriter* w);
void dcvc_bw_free(DcvcBitWriter* w);
int dcvc_bw_write_sps(DcvcBitWriter* w, const DcvcSps* sps);
int dcvc_bw_write_ip(DcvcBitWriter* w, int is_i, int sps_id, int qp,
                     const uint8_t* payload, size_t payload_len);

void dcvc_br_init(DcvcBitReader* r, const uint8_t* data, size_t size);
/* Returns nal type; fills sps_id for SPS/I/P. Returns -1 on error. */
int dcvc_br_read_header(DcvcBitReader* r, int* sps_id);
int dcvc_br_read_sps_remaining(DcvcBitReader* r, int sps_id, DcvcSps* out);
/* Reads QP and payload; payload allocated with malloc, caller frees. */
int dcvc_br_read_ip_remaining(DcvcBitReader* r, int* qp, uint8_t** payload, size_t* payload_len);

#ifdef __cplusplus
}
#endif

#endif
