/* Standalone bit-exactness test for int8 depthwise 3x3 conv kernels.
 *
 * Compares fxp_dw3x3_pack_i8 + fxp_dw3x3_oc_range_i8 against a scalar golden
 * reference that uses the identical int8 quantization scheme:
 *   - activations quantized to int8 then offset to uint8 (q_i8 + 128)
 *   - border / out-of-bounds cells == 128 (NOT 0)
 *   - 9-tap int32 accumulation, subtract w8_comp[oc] = 128*sum(w8[oc]),
 *     then (float)*scale + bias
 *
 * Build:
 *   g++ -O2 -mavx2 -mavx512bw -mavx512vnni -I src \
 *       tests/test_dw3x3_i8_bitexact.cpp src/fxp/fxp_conv.c -lm -o /tmp/test_dw3x3_i8
 */
#include "fxp/fxp_conv.h"
#include "fxp/fxp_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* Scalar golden reference (byte-identical quantization to the SIMD kernels). */
static void golden_dw3x3_i8(const float* x, float* y, int c, int h, int w,
                            const int8_t* w8, const float* w_scale,
                            const float* bias, float x_scale,
                            const int32_t* w_comp,
                            std::vector<uint8_t>& xq_pad)
{
    const float inv_x = 1.0f / x_scale;
    const int Wpad = w + 2;
    const size_t cpad = (size_t)(h + 2) * (size_t)Wpad;
    xq_pad.assign((size_t)c * cpad, 128);   /* border cells == 128 */
    for (int ic = 0; ic < c; ic++) {
        const float* xc = x + (size_t)ic * h * w;
        for (int r = 0; r < h; r++)
            for (int cc = 0; cc < w; cc++) {
                int8_t q = fxp_quantize_act_i8(xc[(size_t)r * w + cc], inv_x);
                xq_pad[(size_t)ic * cpad + (size_t)(r + 1) * Wpad + (cc + 1)] =
                    (uint8_t)(q + 128);
            }
    }
    const int hw = h * w;
    for (int oc = 0; oc < c; oc++) {
        const int8_t* wk = w8 + (size_t)oc * 9;
        const float scale = x_scale * w_scale[oc];
        const float b = bias[oc];
        const int32_t comp = w_comp[oc];
        const uint8_t* qi = xq_pad.data() + (size_t)oc * cpad;
        for (int yh = 0; yh < h; yh++) {
            const uint8_t* r0 = qi + (size_t)(yh + 0) * Wpad;
            const uint8_t* r1 = qi + (size_t)(yh + 1) * Wpad;
            const uint8_t* r2 = qi + (size_t)(yh + 2) * Wpad;
            for (int xw = 0; xw < w; xw++) {
                int32_t acc =
                    (int32_t)r0[xw + 0] * (int32_t)wk[0] +
                    (int32_t)r0[xw + 1] * (int32_t)wk[1] +
                    (int32_t)r0[xw + 2] * (int32_t)wk[2] +
                    (int32_t)r1[xw + 0] * (int32_t)wk[3] +
                    (int32_t)r1[xw + 1] * (int32_t)wk[4] +
                    (int32_t)r1[xw + 2] * (int32_t)wk[5] +
                    (int32_t)r2[xw + 0] * (int32_t)wk[6] +
                    (int32_t)r2[xw + 1] * (int32_t)wk[7] +
                    (int32_t)r2[xw + 2] * (int32_t)wk[8];
                y[(size_t)oc * hw + (size_t)yh * w + xw] =
                    (float)(acc - comp) * scale + b;
            }
        }
    }
}

static uint32_t rng_state = 0x12345678u;
static uint32_t xorshift32(void)
{
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x;
    return x;
}
static float frand(void) { return (float)((int32_t)xorshift32() & 0xffff) / 2048.0f - 16.0f; }

static bool run_case(int c, int h, int w)
{
    const int hw = h * w;
    std::vector<float> x((size_t)c * hw);
    std::vector<int8_t> w8((size_t)c * 9);
    std::vector<float> w_scale((size_t)c), bias((size_t)c);
    std::vector<int32_t> w_comp((size_t)c);

    for (size_t i = 0; i < x.size(); i++) x[i] = frand();
    for (int oc = 0; oc < c; oc++) {
        int32_t sum = 0;
        for (int t = 0; t < 9; t++) {
            int v = (int)(xorshift32() % 255) - 127;   /* [-127, 127] */
            w8[(size_t)oc * 9 + t] = (int8_t)v;
            sum += v;
        }
        w_comp[oc] = 128 * sum;
        w_scale[oc] = 0.5f + (float)(xorshift32() % 1000) / 1000.0f;
        bias[oc] = frand();
    }
    const float x_scale = 0.25f;

    /* golden */
    std::vector<float> y_gold((size_t)c * hw);
    std::vector<uint8_t> xq_gold;
    golden_dw3x3_i8(x.data(), y_gold.data(), c, h, w, w8.data(),
                    w_scale.data(), bias.data(), x_scale, w_comp.data(), xq_gold);

    /* SIMD implementation */
    const int Wpad = w + 2;
    const size_t cpad = (size_t)(h + 2) * (size_t)Wpad;
    std::vector<uint8_t> xq_simd((size_t)c * cpad);
    std::vector<float> y_simd((size_t)c * hw);
    fxp_dw3x3_pack_i8(x.data(), xq_simd.data(), c, h, w, x_scale);
    fxp_dw3x3_oc_range_i8(xq_simd.data(), y_simd.data(), c, h, w,
                          w8.data(), w_scale.data(), w_comp.data(),
                          bias.data(), x_scale, 0, c);

    /* padded buffers must be byte-identical */
    bool pad_ok = (xq_gold.size() == xq_simd.size()) &&
                  (memcmp(xq_gold.data(), xq_simd.data(),
                          xq_gold.size() * sizeof(uint8_t)) == 0);
    /* outputs must be byte-identical (float memcmp) */
    bool out_ok = (memcmp(y_gold.data(), y_simd.data(),
                          y_gold.size() * sizeof(float)) == 0);

    if (!pad_ok)
        std::printf("  [FAIL] c=%d h=%d w=%d : padded buffer mismatch\n", c, h, w);
    if (!out_ok) {
        int first = -1;
        for (size_t i = 0; i < y_gold.size(); i++)
            if (y_gold[i] != y_simd[i]) { first = (int)i; break; }
        std::printf("  [FAIL] c=%d h=%d w=%d : output mismatch first idx=%d "
                    "gold=%f simd=%f\n", c, h, w, first,
                    first >= 0 ? y_gold[first] : 0.f,
                    first >= 0 ? y_simd[first] : 0.f);
    }
    if (pad_ok && out_ok)
        std::printf("  [PASS] c=%-3d h=%-3d w=%-3d (Wpad=%d)\n", c, h, w, Wpad);
    return pad_ok && out_ok;
}

int main(void)
{
    std::printf("dw3x3 int8 bit-exactness test\n");
    bool ok = true;
    ok &= run_case(16, 8, 8);
    ok &= run_case(64, 16, 16);
    ok &= run_case(128, 13, 17);   /* odd width -> exercises SIMD tails */
    ok &= run_case(32, 1, 1);      /* degenerate: single pixel (all border) */
    ok &= run_case(48, 32, 33);    /* width 33: 16 + 16 + 1 scalar tail */
    ok &= run_case(7, 5, 9);       /* small, c<h */
    std::printf("%s\n", ok ? "OVERALL PASS" : "OVERALL FAIL");
    return ok ? 0 : 1;
}
