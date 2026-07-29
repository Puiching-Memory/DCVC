/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * DCV2 container read/write. Explicit little-endian byte ops so packets are
 * identical across any host endianness. */
#include "dcvc_container.h"
#include "../rans/rans_c.h"

#include <stdlib.h>
#include <string.h>

/* --- little-endian byte helpers (host-endian independent) ---------------- */
static void put_u32le(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}
static uint32_t get_u32le(const uint8_t* p) {
    return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

dcvc_status_t dcvc_container_parse(const uint8_t* buf, size_t size,
                                   dcvc_container_header* hdr,
                                   const uint8_t** payload_ptr)
{
    if (!buf || !hdr) return DCVC_ERR_INVALID_ARG;
    if (size < (size_t)DCVC_CONTAINER_HEADER_BYTES) return DCVC_ERR_FORMAT;
    if (buf[0] != DCVC_CONTAINER_MAGIC0 || buf[1] != DCVC_CONTAINER_MAGIC1 ||
        buf[2] != DCVC_CONTAINER_MAGIC2 || buf[3] != DCVC_CONTAINER_MAGIC3)
        return DCVC_ERR_FORMAT;

    hdr->version    = buf[4];
    hdr->flags      = buf[5];
    hdr->frame_type = buf[6];
    hdr->qp         = buf[7];
    hdr->width      = get_u32le(buf + 8);
    hdr->height     = get_u32le(buf + 12);
    hdr->payload_len = get_u32le(buf + 16);

    if (hdr->version != DCVC_CONTAINER_VERSION) return DCVC_ERR_FORMAT;
    if (hdr->frame_type != DCVC_FRAME_INTRA && hdr->frame_type != DCVC_FRAME_INTER)
        return DCVC_ERR_FORMAT;

    /* Bounds-check payload + optional crc against the actual buffer size. */
    size_t need = dcvc_container_total_bytes(hdr);
    if (need > size) return DCVC_ERR_FORMAT;

    if (payload_ptr) *payload_ptr = buf + DCVC_CONTAINER_HEADER_BYTES;
    return DCVC_OK;
}

dcvc_status_t dcvc_container_build(const dcvc_container_header* hdr,
                                   const uint8_t* payload,
                                   int with_crc,
                                   uint8_t** out_buf, size_t* out_size)
{
    if (!hdr || !payload || !out_buf || !out_size) return DCVC_ERR_INVALID_ARG;
    dcvc_container_header h = *hdr;
    h.version = DCVC_CONTAINER_VERSION;
    h.flags = (uint8_t)((h.flags & ~DCVC_CONTAINER_FLAG_CRC) |
                        (with_crc ? DCVC_CONTAINER_FLAG_CRC : 0));

    size_t total = dcvc_container_total_bytes(&h);
    uint8_t* buf = (uint8_t*)malloc(total);
    if (!buf) return DCVC_ERR_OOM;

    buf[0] = DCVC_CONTAINER_MAGIC0;
    buf[1] = DCVC_CONTAINER_MAGIC1;
    buf[2] = DCVC_CONTAINER_MAGIC2;
    buf[3] = DCVC_CONTAINER_MAGIC3;
    buf[4] = h.version;
    buf[5] = h.flags;
    buf[6] = h.frame_type;
    buf[7] = h.qp;
    put_u32le(buf + 8,  h.width);
    put_u32le(buf + 12, h.height);
    put_u32le(buf + 16, h.payload_len);
    memcpy(buf + DCVC_CONTAINER_HEADER_BYTES, payload, h.payload_len);

    if (with_crc) {
        uint32_t crc = dcvc_crc32(buf, DCVC_CONTAINER_HEADER_BYTES + h.payload_len);
        put_u32le(buf + DCVC_CONTAINER_HEADER_BYTES + h.payload_len, crc);
    }

    *out_buf = buf;
    *out_size = total;
    return DCVC_OK;
}

dcvc_status_t dcvc_container_verify_crc(const uint8_t* buf, size_t size,
                                        const dcvc_container_header* hdr)
{
    if (!hdr) return DCVC_ERR_INVALID_ARG;
    if (!(hdr->flags & DCVC_CONTAINER_FLAG_CRC)) return DCVC_OK;  /* no crc to check */
    size_t body = (size_t)DCVC_CONTAINER_HEADER_BYTES + hdr->payload_len;
    if (body + DCVC_CONTAINER_CRC_BYTES > size) return DCVC_ERR_FORMAT;
    uint32_t want = get_u32le(buf + body);
    uint32_t calc = dcvc_crc32(buf, body);
    return (want == calc) ? DCVC_OK : DCVC_ERR_FORMAT;
}
