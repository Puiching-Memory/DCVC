#include "fxp_scale_index.h"
#include "npy_reader.h"

#include <math.h>
#include <stdlib.h>

/* Optional IEEE float16 round-trip on scales before CDF indexing.
 * Set DCVC_SCALE_FP16=1 to measure fp16 quantization on the rANS scale path
 * (both enc and dec must use the same setting). Default off. */
static int scale_fp16_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("DCVC_SCALE_FP16");
        cached = (e && e[0] == '1' && e[1] == '\0') ? 1 : 0;
    }
    return cached;
}

static float scale_maybe_fp16(float s)
{
    if (!scale_fp16_enabled()) return s;
    return dcvc_f16_to_f32(dcvc_f32_to_f16(s));
}

/* Log-spaced scale table in Q12: exp(linspace(log(0.11), log(16), 128)). */
static const int32_t k_scale_table_q12[DCVC_SCALE_LEVELS] = {
    451, 469, 487, 507, 527, 548, 570, 593,
    617, 641, 667, 694, 721, 750, 780, 811,
    844, 878, 913, 949, 987, 1027, 1068, 1110,
    1155, 1201, 1249, 1299, 1351, 1405, 1461, 1519,
    1580, 1643, 1709, 1777, 1848, 1922, 1999, 2079,
    2162, 2249, 2339, 2432, 2530, 2631, 2736, 2845,
    2959, 3077, 3201, 3329, 3462, 3600, 3744, 3894,
    4049, 4211, 4380, 4555, 4737, 4927, 5124, 5328,
    5542, 5763, 5994, 6233, 6483, 6742, 7011, 7292,
    7583, 7887, 8202, 8530, 8871, 9226, 9595, 9979,
    10378, 10793, 11224, 11673, 12140, 12625, 13130, 13655,
    14202, 14769, 15360, 15974, 16613, 17278, 17968, 18687,
    19434, 20212, 21020, 21860, 22735, 23644, 24589, 25573,
    26595, 27659, 28765, 29915, 31112, 32356, 33650, 34995,
    36395, 37850, 39364, 40938, 42575, 44278, 46049, 47890,
    49805, 51797, 53868, 56023, 58263, 60593, 63016, 65536,
};

int32_t dcvc_float_to_q18(float x)
{
    float v = x * (float)DCVC_Q18_ONE;
    v = (v >= 0.f) ? floorf(v + 0.5f) : ceilf(v - 0.5f);
    if (v > 2147483647.f) return 2147483647;
    if (v < -2147483648.f) return (int32_t)(-2147483647 - 1);
    return (int32_t)v;
}

int32_t dcvc_q18_to_q12_round(int32_t q18)
{
    /* >>6 with round-to-nearest (ties away from zero via +32). Scales are >=0. */
    if (q18 >= 0)
        return (q18 + 32) >> 6;
    return -(((-q18) + 32) >> 6);
}

uint8_t dcvc_scale_q12_to_index(int32_t s_q12)
{
    if (s_q12 < DCVC_SCALE_MIN_Q12) s_q12 = DCVC_SCALE_MIN_Q12;
    if (s_q12 > DCVC_SCALE_MAX_Q12) s_q12 = DCVC_SCALE_MAX_Q12;

    /* Largest i with table[i] <= s_q12. Monotonic table → binary search. */
    int lo = 0, hi = DCVC_SCALE_LEVELS - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (k_scale_table_q12[mid] <= s_q12)
            lo = mid;
        else
            hi = mid - 1;
    }
    return (uint8_t)lo;
}

uint8_t dcvc_scale_float_to_index(float scale)
{
    scale = scale_maybe_fp16(scale);
    int32_t q18 = dcvc_float_to_q18(scale);
    int32_t q12 = dcvc_q18_to_q12_round(q18);
    return dcvc_scale_q12_to_index(q12);
}

void dcvc_build_index_dec_i(const float* scales, uint8_t* out, int n)
{
    for (int i = 0; i < n; i++)
        out[i] = dcvc_scale_float_to_index(scales[i]);
}

void dcvc_build_index_enc_i(const float* symbols, const float* scales,
                            int16_t* out, int n)
{
    for (int i = 0; i < n; i++) {
        int idx = (int)dcvc_scale_float_to_index(scales[i]);
        float sv = symbols[i];
        int sym = (int)((sv >= 0.f) ? floorf(sv + 0.5f) : ceilf(sv - 0.5f));
        out[i] = (int16_t)((sym << 8) + idx);
    }
}
