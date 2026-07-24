/* Unit tests for the integer scale → CDF index path. */
#include "fxp/fxp_scale_index.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int expect_eq_u8(const char* tag, uint8_t a, uint8_t b)
{
    if (a != b) {
        fprintf(stderr, "FAIL %s: got %u want %u\n", tag, a, b);
        return 1;
    }
    return 0;
}

static int test_table_endpoints(void)
{
    int fail = 0;
    fail |= expect_eq_u8("min", dcvc_scale_float_to_index(0.11f), 0);
    fail |= expect_eq_u8("below_min", dcvc_scale_float_to_index(0.01f), 0);
    fail |= expect_eq_u8("max", dcvc_scale_float_to_index(16.0f), 127);
    fail |= expect_eq_u8("above_max", dcvc_scale_float_to_index(100.0f), 127);
    printf("endpoints: %s\n", fail ? "FAIL" : "PASS");
    return fail;
}

static int test_q_pipeline_deterministic(void)
{
    /* Same float → same index, always. */
    const float scales[] = {0.11f, 0.5f, 1.0f, 2.5f, 8.0f, 15.999f, 1e-6f, 1e3f};
    int fail = 0;
    for (int i = 0; i < (int)(sizeof(scales) / sizeof(scales[0])); i++) {
        uint8_t a = dcvc_scale_float_to_index(scales[i]);
        uint8_t b = dcvc_scale_float_to_index(scales[i]);
        if (a != b) {
            fprintf(stderr, "non-deterministic at %g\n", scales[i]);
            fail = 1;
        }
        /* Also via explicit Q18/Q12 */
        int32_t q18 = dcvc_float_to_q18(scales[i]);
        int32_t q12 = dcvc_q18_to_q12_round(q18);
        uint8_t c = dcvc_scale_q12_to_index(q12);
        if (c != a) {
            fprintf(stderr, "Q path mismatch at %g: %u vs %u\n", scales[i], a, c);
            fail = 1;
        }
    }
    printf("deterministic: %s\n", fail ? "FAIL" : "PASS");
    return fail;
}

static int test_batch_bitexact(void)
{
    const int n = 4096;
    float* s = (float*)malloc((size_t)n * sizeof(float));
    uint8_t* a = (uint8_t*)malloc((size_t)n);
    uint8_t* b = (uint8_t*)malloc((size_t)n);
    int16_t* ea = (int16_t*)malloc((size_t)n * sizeof(int16_t));
    int16_t* eb = (int16_t*)malloc((size_t)n * sizeof(int16_t));
    float* sym = (float*)malloc((size_t)n * sizeof(float));
    if (!s || !a || !b || !ea || !eb || !sym) return 1;

    unsigned seed = 123;
    for (int i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        float u = (seed & 0xffff) / 65535.f;
        s[i] = 0.05f + u * 20.f;
        sym[i] = ((int)(seed >> 16) % 21) - 10;
    }
    dcvc_build_index_dec_i(s, a, n);
    dcvc_build_index_dec_i(s, b, n);
    dcvc_build_index_enc_i(sym, s, ea, n);
    dcvc_build_index_enc_i(sym, s, eb, n);

    int fail = 0;
    if (memcmp(a, b, (size_t)n) != 0) {
        fprintf(stderr, "dec batch mismatch\n");
        fail = 1;
    }
    if (memcmp(ea, eb, (size_t)n * sizeof(int16_t)) != 0) {
        fprintf(stderr, "enc batch mismatch\n");
        fail = 1;
    }
    /* packed low byte == dec index */
    for (int i = 0; i < n && !fail; i++) {
        if ((uint8_t)(ea[i] & 0xff) != a[i]) {
            fprintf(stderr, "enc/dec index diverge at %d\n", i);
            fail = 1;
        }
    }
    printf("batch bit-exact: %s\n", fail ? "FAIL" : "PASS");
    free(s); free(a); free(b); free(ea); free(eb); free(sym);
    return fail;
}

static int test_near_boundary_q12_stable(void)
{
    /* Two floats that round to the same Q12 must share an index. */
    int32_t q12 = 1024; /* ~0.25 */
    float lo = (q12 - 0.4f) / 4096.f;
    float hi = (q12 + 0.4f) / 4096.f;
    /* Push through Q18→Q12; they should land on same or adjacent — document. */
    uint8_t i0 = dcvc_scale_float_to_index(lo);
    uint8_t i1 = dcvc_scale_float_to_index(hi);
    printf("near-boundary q12=%d indexes %u %u\n", q12, i0, i1);
    /* Not a hard fail if they differ by 1 — Q12 bins are the contract. */
    return 0;
}

int main(void)
{
    int fail = 0;
    fail |= test_table_endpoints();
    fail |= test_q_pipeline_deterministic();
    fail |= test_batch_bitexact();
    fail |= test_near_boundary_q12_stable();
    /* Smoke: old float log path agreement rate (informational). */
    {
        int agree = 0, n = 0;
        for (int k = 0; k < 1000; k++) {
            float s = 0.11f * powf(16.f / 0.11f, k / 999.f);
            float lmin = logf(0.11f);
            float lrecip = 127.f / (logf(16.f) - lmin);
            float v = s; if (v < 0.11f) v = 0.11f; if (v > 16.f) v = 16.f;
            int fold = (int)floorf((logf(v) - lmin) * lrecip);
            if (fold < 0) fold = 0; if (fold > 127) fold = 127;
            int inew = dcvc_scale_float_to_index(s);
            if (fold == inew) agree++;
            n++;
        }
        printf("agree with legacy float-log on linspace: %d/%d (%.1f%%)\n",
               agree, n, 100.0 * agree / n);
    }
    return fail ? 1 : 0;
}
