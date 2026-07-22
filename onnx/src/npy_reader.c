/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Cross-platform .npy reader/writer implementation.
 */
#include "npy_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- float16 <-> float32 ---------- */

float dcvc_f16_to_f32(uint16_t h)
{
    uint32_t s = (uint32_t)(h >> 15) & 1u;
    uint32_t e = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t m = (uint32_t)h & 0x3ffu;
    uint32_t f;

    if (e == 0) {
        if (m == 0) {
            f = s << 31;
        } else {
            int ex = -1;
            while (!(m & 0x400u)) { m <<= 1; ex--; }
            m &= 0x3ffu;
            e = 127u + (uint32_t)ex - 14u;
            f = (s << 31) | (e << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = (s << 31) | (0xffu << 23) | (m << 13);
    } else {
        f = (s << 31) | ((e + 127u - 15u) << 23) | (m << 13);
    }
    float r;
    memcpy(&r, &f, sizeof(r));
    return r;
}

uint16_t dcvc_f32_to_f16(float fv)
{
    uint32_t x;
    memcpy(&x, &fv, sizeof(x));
    uint32_t s = (x >> 16) & 0x8000u;
    int e = (int)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t m = x & 0x7fffffu;

    if ((x & 0x7fffffffu) == 0u) return (uint16_t)s;
    if (e <= 0) {
        if (e < -10) return (uint16_t)s;
        m |= 0x800000u;
        uint32_t t = (m >> (14 - e)) + ((m >> (13 - e)) & 1u);
        return (uint16_t)(s | t);
    }
    if (e >= 31) return (uint16_t)(s | 0x7c00u);
    return (uint16_t)(s | ((uint32_t)e << 10) | (m >> 13));
}

/* ---------- Header parsing ---------- */

static int parse_header(const char* hdr, int* ndims, int* dims, int max_nd,
                        DcvcNpyDtype* dtype)
{
    /* expected: {'descr': '<f4', 'fortran_order': False, 'shape': (a, b,), }
     * We only need the dtype tag and the shape tuple. */
    *dtype = DCVC_NPY_NONE;
    *ndims = 0;

    if (strstr(hdr, "<f2") || strstr(hdr, "'f2'"))
        *dtype = DCVC_NPY_F16;
    else if (strstr(hdr, "<f4") || strstr(hdr, "'f4'"))
        *dtype = DCVC_NPY_F32;
    else if (strstr(hdr, "<i4") || strstr(hdr, "'i4'"))
        *dtype = DCVC_NPY_I32;
    else
        return -1;

    const char* sp = strstr(hdr, "shape");
    if (!sp) return -1;
    sp = strchr(sp, '(');
    if (!sp) return -1;
    sp++;
    while (*sp && *sp != ')' && *ndims < max_nd) {
        if (*sp >= '0' && *sp <= '9') {
            dims[(*ndims)++] = atoi(sp);
            while (*sp >= '0' && *sp <= '9') sp++;
        }
        sp++;
    }
    return 0;
}

int dcvc_npy_read(const char* path, DcvcNpy* o)
{
    memset(o, 0, sizeof(*o));

    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) {
        fclose(f);
        return -1;
    }

    uint8_t major = 0, minor = 0;
    if (fread(&major, 1, 1, f) != 1 || fread(&minor, 1, 1, f) != 1) {
        fclose(f);
        return -1;
    }

    size_t header_len = 0;
    if (major == 1) {
        uint16_t h16 = 0;
        if (fread(&h16, 2, 1, f) != 1) { fclose(f); return -1; }
        header_len = h16;
    } else if (major >= 2) {
        uint32_t h32 = 0;
        if (fread(&h32, 4, 1, f) != 1) { fclose(f); return -1; }
        header_len = h32;
    } else {
        fclose(f);
        return -1;
    }

    char hdr[1024];
    if (header_len >= sizeof(hdr)) { fclose(f); return -1; }
    if (fread(hdr, 1, header_len, f) != header_len) { fclose(f); return -1; }
    hdr[header_len] = '\0';

    DcvcNpyDtype dtype;
    if (parse_header(hdr, &o->ndims, o->dims, (int)(sizeof(o->dims) / sizeof(o->dims[0])), &dtype) != 0) {
        fclose(f);
        return -1;
    }
    o->dtype = dtype;

    o->elems = 1;
    for (int i = 0; i < o->ndims; i++) o->elems *= (size_t)o->dims[i];

    size_t elem_bytes = (dtype == DCVC_NPY_F16) ? 2 : 4;
    o->data = malloc(o->elems * elem_bytes);
    if (!o->data) { fclose(f); return -1; }

    size_t got = fread(o->data, elem_bytes, o->elems, f);
    fclose(f);
    if (got != o->elems) { dcvc_npy_free(o); return -1; }
    return 0;
}

/* Open the file and parse the header; return the file pointer positioned at
 * the beginning of the data payload. On success leaves metadata in the
 * supplied buffers and returns 0. */
static int open_npy_data(const char* path, FILE** out_f, int* out_ndims, int* out_dims,
                         int max_nd, DcvcNpyDtype* out_dtype)
{
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); return -1; }
    uint8_t major = 0, minor = 0;
    if (fread(&major, 1, 1, f) != 1 || fread(&minor, 1, 1, f) != 1) { fclose(f); return -1; }
    size_t header_len = 0;
    if (major == 1) {
        uint16_t h16 = 0;
        if (fread(&h16, 2, 1, f) != 1) { fclose(f); return -1; }
        header_len = h16;
    } else if (major >= 2) {
        uint32_t h32 = 0;
        if (fread(&h32, 4, 1, f) != 1) { fclose(f); return -1; }
        header_len = h32;
    } else { fclose(f); return -1; }
    char hdr[1024];
    if (header_len >= sizeof(hdr) || fread(hdr, 1, header_len, f) != header_len) { fclose(f); return -1; }
    hdr[header_len] = '\0';
    if (parse_header(hdr, out_ndims, out_dims, max_nd, out_dtype) != 0) { fclose(f); return -1; }
    *out_f = f;
    return 0;
}

int dcvc_npy_read_meta(const char* path, int* out_ndims, int* out_dims, int max_nd)
{
    FILE* f = NULL;
    DcvcNpyDtype dtype;
    if (open_npy_data(path, &f, out_ndims, out_dims, max_nd, &dtype) != 0) return -1;
    fclose(f);
    return 0;
}

int dcvc_npy_read_frame_f32(const char* path, int frame_idx, float* out_data,
                              int* out_dims /* 3 entries: C, H, W */)
{
    FILE* f = NULL;
    int ndims = 0;
    int dims[8] = {0};
    DcvcNpyDtype dtype;
    if (open_npy_data(path, &f, &ndims, dims, 8, &dtype) != 0) return -1;
    if (dtype != DCVC_NPY_F32 || ndims < 4) { fclose(f); return -1; }
    if (frame_idx < 0 || frame_idx >= dims[0]) { fclose(f); return -1; }
    size_t c = (size_t)dims[ndims - 3];
    size_t h = (size_t)dims[ndims - 2];
    size_t w = (size_t)dims[ndims - 1];
    size_t frame_floats = c * h * w;
    size_t offset = (size_t)frame_idx * frame_floats * sizeof(float);
    if (fseek(f, (long)offset, SEEK_CUR) != 0) { fclose(f); return -1; }
    if (fread(out_data, sizeof(float), frame_floats, f) != frame_floats) { fclose(f); return -1; }
    fclose(f);
    out_dims[0] = (int)c;
    out_dims[1] = (int)h;
    out_dims[2] = (int)w;
    return 0;
}

void dcvc_npy_free(DcvcNpy* o)
{
    if (!o) return;
    free(o->data);
    memset(o, 0, sizeof(*o));
}

const float*    dcvc_npy_f32(const DcvcNpy* o) { return (o && o->dtype == DCVC_NPY_F32) ? (const float*)o->data : NULL; }
const int32_t*  dcvc_npy_i32(const DcvcNpy* o) { return (o && o->dtype == DCVC_NPY_I32) ? (const int32_t*)o->data : NULL; }
const uint16_t* dcvc_npy_f16(const DcvcNpy* o) { return (o && o->dtype == DCVC_NPY_F16) ? (const uint16_t*)o->data : NULL; }

/* ---------- Writers ---------- */

static int write_npy(const char* path, const void* data, size_t elem_bytes,
                     const char* descr, const int* dims, int nd)
{
    if (nd <= 0) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    fwrite("\x93NUMPY", 1, 6, f);
    uint8_t major = 1, minor = 0;
    fwrite(&major, 1, 1, f);
    fwrite(&minor, 1, 1, f);

    char shape[128];
    int pos = 0;
    pos += snprintf(shape + pos, sizeof(shape) - pos, "(");
    for (int i = 0; i < nd; i++)
        pos += snprintf(shape + pos, sizeof(shape) - pos, "%d,", dims[i]);
    pos += snprintf(shape + pos, sizeof(shape) - pos, ")");

    char hdr[512];
    int len = snprintf(hdr, sizeof(hdr),
                       "{'descr': '%s', 'fortran_order': False, 'shape': %s, }",
                       descr, shape);
    if (len < 0 || len >= (int)sizeof(hdr)) { fclose(f); return -1; }
    size_t hl = (size_t)len;
    while ((hl + 10) % 64 != 0) hdr[hl++] = ' ';

    uint16_t h16 = (uint16_t)hl;
    fwrite(&h16, 2, 1, f);
    fwrite(hdr, 1, hl, f);

    size_t n = 1;
    for (int i = 0; i < nd; i++) n *= (size_t)dims[i];
    fwrite(data, elem_bytes, n, f);

    fclose(f);
    return 0;
}

int dcvc_npy_write_f32(const char* path, const float* data, const int* dims, int nd)
{
    return write_npy(path, data, 4, "<f4", dims, nd);
}

int dcvc_npy_write_f16(const char* path, const float* data, const int* dims, int nd)
{
    if (nd <= 0) return -1;
    size_t n = 1;
    for (int i = 0; i < nd; i++) n *= (size_t)dims[i];
    uint16_t* buf = (uint16_t*)malloc(n * 2);
    if (!buf) return -1;
    for (size_t i = 0; i < n; i++) buf[i] = dcvc_f32_to_f16(data[i]);
    int rc = write_npy(path, buf, 2, "<f2", dims, nd);
    free(buf);
    return rc;
}
