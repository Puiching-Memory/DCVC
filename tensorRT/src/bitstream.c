#include "dcvc_bitstream.h"

#include <stdlib.h>
#include <string.h>

static int bw_ensure(DcvcBitWriter* w, size_t more)
{
    if (w->size + more <= w->cap) return 0;
    size_t ncap = w->cap ? w->cap : 256;
    while (ncap < w->size + more) ncap *= 2;
    uint8_t* p = (uint8_t*)realloc(w->data, ncap);
    if (!p) return -1;
    w->data = p;
    w->cap = ncap;
    return 0;
}

static int bw_u8(DcvcBitWriter* w, uint8_t v)
{
    if (bw_ensure(w, 1) != 0) return -1;
    w->data[w->size++] = v;
    return 0;
}

static int bw_bytes(DcvcBitWriter* w, const uint8_t* p, size_t n)
{
    if (n == 0) return 0;
    if (bw_ensure(w, n) != 0) return -1;
    memcpy(w->data + w->size, p, n);
    w->size += n;
    return 0;
}

/* Adaptive unsigned integer matching stream_helper.write_uint_adaptive */
static int bw_uint_adaptive(DcvcBitWriter* w, uint32_t a)
{
    if (a < (1u << 7)) {
        return bw_u8(w, (uint8_t)((a & 0xff) | (0x00u << 7)));
    }
    if (a < (1u << 14)) {
        uint8_t a0 = (uint8_t)((a >> 0) & 0xff);
        uint8_t a1 = (uint8_t)(((a >> 8) & 0xff) | (0x02u << 6));
        if (bw_u8(w, a1) != 0) return -1;
        return bw_u8(w, a0);
    }
    uint8_t a0 = (uint8_t)((a >> 0) & 0xff);
    uint8_t a1 = (uint8_t)((a >> 8) & 0xff);
    uint8_t a2 = (uint8_t)((a >> 16) & 0xff);
    uint8_t a3 = (uint8_t)(((a >> 24) & 0xff) | (0x03u << 6));
    if (bw_u8(w, a3) != 0) return -1;
    if (bw_u8(w, a2) != 0) return -1;
    if (bw_u8(w, a1) != 0) return -1;
    return bw_u8(w, a0);
}

static int br_u8(DcvcBitReader* r, uint8_t* v)
{
    if (r->pos >= r->size) return -1;
    *v = r->data[r->pos++];
    return 0;
}

static int br_bytes(DcvcBitReader* r, uint8_t* dst, size_t n)
{
    if (r->pos + n > r->size) return -1;
    memcpy(dst, r->data + r->pos, n);
    r->pos += n;
    return 0;
}

static int br_uint_adaptive(DcvcBitReader* r, uint32_t* out)
{
    uint8_t a3;
    if (br_u8(r, &a3) != 0) return -1;
    if ((a3 >> 7) == 0) {
        *out = a3;
        return 0;
    }
    uint8_t a2;
    if (br_u8(r, &a2) != 0) return -1;
    if ((a3 >> 6) == 0x02) {
        a3 = (uint8_t)(a3 & 0x3f);
        *out = ((uint32_t)a3 << 8) + a2;
        return 0;
    }
    a3 = (uint8_t)(a3 & 0x3f);
    uint8_t a1, a0;
    if (br_u8(r, &a1) != 0) return -1;
    if (br_u8(r, &a0) != 0) return -1;
    *out = ((uint32_t)a3 << 24) + ((uint32_t)a2 << 16) + ((uint32_t)a1 << 8) + a0;
    return 0;
}

void dcvc_bw_init(DcvcBitWriter* w)
{
    w->data = NULL;
    w->size = 0;
    w->cap = 0;
}

void dcvc_bw_free(DcvcBitWriter* w)
{
    free(w->data);
    w->data = NULL;
    w->size = 0;
    w->cap = 0;
}

int dcvc_bw_write_sps(DcvcBitWriter* w, const DcvcSps* sps)
{
    if (!w || !sps || sps->sps_id >= 16) return -1;
    uint8_t flag = (uint8_t)((DCVC_NAL_SPS << 4) + (sps->sps_id & 0x0f));
    if (bw_u8(w, flag) != 0) return -1;
    if (bw_uint_adaptive(w, (uint32_t)sps->height) != 0) return -1;
    if (bw_uint_adaptive(w, (uint32_t)sps->width) != 0) return -1;
    flag = (uint8_t)((sps->ec_part << 2) + (sps->use_ada_i & 1));
    return bw_u8(w, flag);
}

int dcvc_bw_write_ip(DcvcBitWriter* w, int is_i, int sps_id, int qp,
                     const uint8_t* payload, size_t payload_len)
{
    if (!w || sps_id < 0 || sps_id >= 16 || qp < 0 || qp >= 256) return -1;
    uint8_t nal = (uint8_t)(((is_i ? DCVC_NAL_I : DCVC_NAL_P) << 4) + (sps_id & 0x0f));
    if (bw_u8(w, nal) != 0) return -1;
    if (bw_u8(w, (uint8_t)qp) != 0) return -1;
    if (bw_uint_adaptive(w, (uint32_t)payload_len) != 0) return -1;
    return bw_bytes(w, payload, payload_len);
}

void dcvc_br_init(DcvcBitReader* r, const uint8_t* data, size_t size)
{
    r->data = data;
    r->size = size;
    r->pos = 0;
}

int dcvc_br_read_header(DcvcBitReader* r, int* sps_id)
{
    uint8_t flag;
    if (br_u8(r, &flag) != 0) return -1;
    int nal = flag >> 4;
    if (sps_id) *sps_id = flag & 0x0f;
    return nal;
}

int dcvc_br_read_sps_remaining(DcvcBitReader* r, int sps_id, DcvcSps* out)
{
    if (!out) return -1;
    uint32_t h = 0, ww = 0;
    if (br_uint_adaptive(r, &h) != 0) return -1;
    if (br_uint_adaptive(r, &ww) != 0) return -1;
    uint8_t flag;
    if (br_u8(r, &flag) != 0) return -1;
    out->sps_id = sps_id;
    out->height = (int)h;
    out->width = (int)ww;
    out->ec_part = (flag >> 2) & 0x01;
    out->use_ada_i = flag & 0x01;
    return 0;
}

int dcvc_br_read_ip_remaining(DcvcBitReader* r, int* qp, uint8_t** payload, size_t* payload_len)
{
    uint8_t q;
    if (br_u8(r, &q) != 0) return -1;
    uint32_t len = 0;
    if (br_uint_adaptive(r, &len) != 0) return -1;
    uint8_t* buf = (uint8_t*)malloc(len ? len : 1);
    if (!buf) return -1;
    if (len && br_bytes(r, buf, len) != 0) {
        free(buf);
        return -1;
    }
    if (qp) *qp = q;
    *payload = buf;
    *payload_len = len;
    return 0;
}
