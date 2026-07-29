/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Self-describing packet container "DCV2". This wraps the raw codec stream
 * (the [z_len][z_stream][y_stream] blob produced by the pipelines) with a
 * fixed header and an optional CRC-32 trailer, so a decoder needs no
 * out-of-band information.
 *
 *   offset 0   : magic      "DCV2"            (4 bytes)
 *   offset 4   : version    uint8  (= 2)
 *   offset 5   : flags      uint8  (bit0 = CRC present)
 *   offset 6   : frame_type uint8  (0 = intra, 1 = inter)
 *   offset 7   : qp         uint8
 *   offset 8   : width      uint32 LE
 *   offset 12  : height     uint32 LE
 *   offset 16  : payload_len uint32 LE
 *   offset 20  : payload[payload_len]
 *   trailer    : crc32      uint32 LE  (only if flags bit0; over [0,20+len))
 *
 * All multi-byte fields are little-endian regardless of host, so packets are
 * byte-identical across Windows/Linux/ARM. Internal / not part of the public
 * ABI surface.
 */
#ifndef DCVC_CONTAINER_H
#define DCVC_CONTAINER_H

#include "../include/dcvc/dcvc_types.h"
#include "../include/dcvc/dcvc_codec.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DCVC_CONTAINER_MAGIC0 'D'
#define DCVC_CONTAINER_MAGIC1 'C'
#define DCVC_CONTAINER_MAGIC2 'V'
#define DCVC_CONTAINER_MAGIC3 '2'
#define DCVC_CONTAINER_VERSION 2
#define DCVC_CONTAINER_HEADER_BYTES 20
#define DCVC_CONTAINER_CRC_BYTES 4
#define DCVC_CONTAINER_FLAG_CRC 0x01

/* Fixed header layout (20 bytes). */
typedef struct {
    uint8_t  version;
    uint8_t  flags;
    uint8_t  frame_type;     /* dcvc_frame_type_t */
    uint8_t  qp;
    uint32_t width;
    uint32_t height;
    uint32_t payload_len;
} dcvc_container_header;

/* Parse a packet buffer into a header. Returns DCVC_OK on a well-formed
 * packet (valid magic + version), DCVC_ERR_FORMAT otherwise. payload_ptr
 * receives a pointer into the buffer at the payload start. */
dcvc_status_t dcvc_container_parse(const uint8_t* buf, size_t size,
                                   dcvc_container_header* hdr,
                                   const uint8_t** payload_ptr);

/* Total wire size of a packet given a header (header + payload + optional crc). */
static inline size_t dcvc_container_total_bytes(const dcvc_container_header* hdr) {
    size_t n = (size_t)DCVC_CONTAINER_HEADER_BYTES + hdr->payload_len;
    if (hdr->flags & DCVC_CONTAINER_FLAG_CRC) n += DCVC_CONTAINER_CRC_BYTES;
    return n;
}

/* Build a packet (malloc'd, caller frees via dcvc_packet_free()). `payload` is
 * copied. When with_crc, a CRC-32 over the header+payload is appended. */
dcvc_status_t dcvc_container_build(const dcvc_container_header* hdr,
                                   const uint8_t* payload,
                                   int with_crc,
                                   uint8_t** out_buf, size_t* out_size);

/* Verify the CRC trailer of a parsed packet. Returns DCVC_OK if no CRC is
 * present or if it matches; DCVC_ERR_FORMAT on mismatch. */
dcvc_status_t dcvc_container_verify_crc(const uint8_t* buf, size_t size,
                                        const dcvc_container_header* hdr);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_CONTAINER_H */
