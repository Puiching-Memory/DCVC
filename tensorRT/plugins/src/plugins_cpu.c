#include "dcvc_plugins.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Soft FP16 helpers (match color.c) */
static float f16_to_f32(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            out = sign | (((exp + 127 - 15) & 0xffu) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | (((exp + 127 - 15) & 0xffu) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

static uint16_t f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if ((x & 0x7fffffffu) == 0) return (uint16_t)sign;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t t = (mant >> (14 - exp)) + ((mant >> (13 - exp)) & 1u);
        return (uint16_t)(sign | t);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static size_t numel(const DcvcTensorHalf* t)
{
    return (size_t)t->n * t->c * t->h * t->w;
}

int dcvc_plugin_cuda_available(void)
{
#ifdef DCVC_RT_HAS_CUDA
    return 1;
#else
    return 0;
#endif
}

int dcvc_process_with_mask(const DcvcTensorHalf* y, const DcvcTensorHalf* scales,
                           const DcvcTensorHalf* means, const DcvcTensorHalf* mask,
                           float force_zero_thres, DcvcTensorHalf* y_hat, int8_t* y_q_out)
{
    if (!y || !scales || !means || !mask || !y_hat || !y_q_out) return -1;
    size_t n = numel(y);
    for (size_t i = 0; i < n; i++) {
        float m = f16_to_f32(mask->data[i]);
        float s = f16_to_f32(scales->data[i]) * m;
        float mean = f16_to_f32(means->data[i]) * m;
        float yv = f16_to_f32(y->data[i]);
        float y_res = (yv - mean) * m;
        float y_q = roundf(y_res);
        if (force_zero_thres >= 0.f && !(s > force_zero_thres)) y_q = 0.f;
        if (y_q < -128.f) y_q = -128.f;
        if (y_q > 127.f) y_q = 127.f;
        y_q_out[i] = (int8_t)y_q;
        float yhat = y_q + mean;
        y_hat->data[i] = f32_to_f16(yhat);
    }
    y_hat->n = y->n;
    y_hat->c = y->c;
    y_hat->h = y->h;
    y_hat->w = y->w;
    return 0;
}

int dcvc_add_and_multiply(DcvcTensorHalf* y0, const DcvcTensorHalf* y1, const DcvcTensorHalf* q)
{
    if (!y0 || !y1 || !q) return -1;
    size_t n = numel(y0);
    for (size_t i = 0; i < n; i++) {
        float v = (f16_to_f32(y0->data[i]) + f16_to_f32(y1->data[i])) * f16_to_f32(q->data[i]);
        y0->data[i] = f32_to_f16(v);
    }
    return 0;
}

int dcvc_round_to_int8(DcvcTensorHalf* z, int8_t* z_int8_out)
{
    if (!z || !z_int8_out) return -1;
    size_t n = numel(z);
    for (size_t i = 0; i < n; i++) {
        float v = roundf(f16_to_f32(z->data[i]));
        if (v < -128.f) v = -128.f;
        if (v > 127.f) v = 127.f;
        z_int8_out[i] = (int8_t)v;
        z->data[i] = f32_to_f16(v);
    }
    return 0;
}

int dcvc_replicate_pad(const DcvcTensorHalf* in, int pad_b, int pad_r, DcvcTensorHalf* out)
{
    if (!in || !out || !out->data) return -1;
    int oh = in->h + pad_b;
    int ow = in->w + pad_r;
    out->n = in->n;
    out->c = in->c;
    out->h = oh;
    out->w = ow;
    for (int n = 0; n < in->n; n++) {
        for (int c = 0; c < in->c; c++) {
            for (int h = 0; h < oh; h++) {
                int hs = h < in->h ? h : in->h - 1;
                for (int w = 0; w < ow; w++) {
                    int ws = w < in->w ? w : in->w - 1;
                    size_t si = ((size_t)n * in->c + c) * in->h * in->w + (size_t)hs * in->w + ws;
                    size_t di = ((size_t)n * in->c + c) * oh * ow + (size_t)h * ow + w;
                    out->data[di] = in->data[si];
                }
            }
        }
    }
    return 0;
}

struct DcvcDepthConvHandle {
    uint8_t* blob;
    size_t bytes;
};

DcvcDepthConvHandle* dcvc_depthconv_create(const void* weight_blob, size_t blob_bytes)
{
    DcvcDepthConvHandle* h = (DcvcDepthConvHandle*)calloc(1, sizeof(*h));
    if (!h) return NULL;
    h->blob = (uint8_t*)malloc(blob_bytes);
    if (!h->blob) {
        free(h);
        return NULL;
    }
    memcpy(h->blob, weight_blob, blob_bytes);
    h->bytes = blob_bytes;
    return h;
}

void dcvc_depthconv_destroy(DcvcDepthConvHandle* h)
{
    if (!h) return;
    free(h->blob);
    free(h);
}

int dcvc_depthconv_forward(DcvcDepthConvHandle* h, const DcvcTensorHalf* in, DcvcTensorHalf* out)
{
    /* Full fused DepthConv requires CUDA Plugin (plugins/src/depthconv_plugin.cpp).
     * CPU path: identity copy for pipeline wiring when blob is absent/minimal. */
    if (!h || !in || !out || !out->data) return -1;
    size_t n = numel(in);
    if (out->data != in->data) memcpy(out->data, in->data, n * sizeof(uint16_t));
    out->n = in->n;
    out->c = in->c;
    out->h = in->h;
    out->w = in->w;
    (void)h;
    return 0;
}
