#include "dcvc_color.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define KR 0.2126f
#define KG 0.7152f
#define KB 0.0722f

static float clampf(float x, float lo, float hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

void dcvc_yuv420_to_ycbcr444_f32(const uint8_t* y, int y_stride,
                                 const uint8_t* u, int u_stride,
                                 const uint8_t* v, int v_stride,
                                 int width, int height, float* out_chw)
{
    int hw = width * height;
    float* yo = out_chw;
    float* uo = out_chw + hw;
    float* vo = out_chw + 2 * hw;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            yo[j * width + i] = y[j * y_stride + i] / 255.0f;
            int uj = j / 2;
            int ui = i / 2;
            uo[j * width + i] = u[uj * u_stride + ui] / 255.0f;
            vo[j * width + i] = v[uj * v_stride + ui] / 255.0f;
        }
    }
}

void dcvc_ycbcr444_f32_to_yuv420(const float* in_chw, int width, int height,
                                 uint8_t* y, int y_stride,
                                 uint8_t* u, int u_stride,
                                 uint8_t* v, int v_stride)
{
    int hw = width * height;
    const float* yi = in_chw;
    const float* ui = in_chw + hw;
    const float* vi = in_chw + 2 * hw;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            y[j * y_stride + i] = (uint8_t)(clampf(yi[j * width + i], 0.f, 1.f) * 255.f + 0.5f);
        }
    }
    int uh = height / 2;
    int uw = width / 2;
    for (int j = 0; j < uh; j++) {
        for (int i = 0; i < uw; i++) {
            float us = 0.f, vs = 0.f;
            for (int dj = 0; dj < 2; dj++) {
                for (int di = 0; di < 2; di++) {
                    int jj = j * 2 + dj;
                    int ii = i * 2 + di;
                    us += ui[jj * width + ii];
                    vs += vi[jj * width + ii];
                }
            }
            u[j * u_stride + i] = (uint8_t)(clampf(us * 0.25f, 0.f, 1.f) * 255.f + 0.5f);
            v[j * v_stride + i] = (uint8_t)(clampf(vs * 0.25f, 0.f, 1.f) * 255.f + 0.5f);
        }
    }
}

void dcvc_rgb_to_ycbcr444_f32(const uint8_t* rgb, int width, int height, float* out_chw)
{
    int hw = width * height;
    float* yo = out_chw;
    float* cbo = out_chw + hw;
    float* cro = out_chw + 2 * hw;
    for (int i = 0; i < hw; i++) {
        float r = rgb[i * 3 + 0] / 255.0f;
        float g = rgb[i * 3 + 1] / 255.0f;
        float b = rgb[i * 3 + 2] / 255.0f;
        float yy = KR * r + KG * g + KB * b;
        float cb = 0.5f * (b - yy) / (1.0f - KB) + 0.5f;
        float cr = 0.5f * (r - yy) / (1.0f - KR) + 0.5f;
        yo[i] = clampf(yy, 0.f, 1.f);
        cbo[i] = clampf(cb, 0.f, 1.f);
        cro[i] = clampf(cr, 0.f, 1.f);
    }
}

void dcvc_ycbcr444_f32_to_rgb(const float* in_chw, int width, int height, uint8_t* rgb)
{
    int hw = width * height;
    const float* yi = in_chw;
    const float* cbi = in_chw + hw;
    const float* cri = in_chw + 2 * hw;
    for (int i = 0; i < hw; i++) {
        float y = yi[i];
        float cb = cbi[i];
        float cr = cri[i];
        float r = y + (2.0f - 2.0f * KR) * (cr - 0.5f);
        float b = y + (2.0f - 2.0f * KB) * (cb - 0.5f);
        float g = (y - KR * r - KB * b) / KG;
        rgb[i * 3 + 0] = (uint8_t)(clampf(r, 0.f, 1.f) * 255.f + 0.5f);
        rgb[i * 3 + 1] = (uint8_t)(clampf(g, 0.f, 1.f) * 255.f + 0.5f);
        rgb[i * 3 + 2] = (uint8_t)(clampf(b, 0.f, 1.f) * 255.f + 0.5f);
    }
}

/* Soft-float FP16 conversion (IEEE 754 binary16). */
static uint16_t f32_to_f16_bits(float f)
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

static float f16_bits_to_f32(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
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

void dcvc_f32_to_f16(const float* src, uint16_t* dst, size_t n)
{
    for (size_t i = 0; i < n; i++) dst[i] = f32_to_f16_bits(src[i]);
}

void dcvc_f16_to_f32(const uint16_t* src, float* dst, size_t n)
{
    for (size_t i = 0; i < n; i++) dst[i] = f16_bits_to_f32(src[i]);
}
