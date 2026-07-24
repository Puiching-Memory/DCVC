#include "fxp_conv.h"
#include "fxp_common.h"
#include "fxp_wsrelu.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#define FXP_TLS __declspec(thread)
#else
#define FXP_TLS _Thread_local
#endif

#if defined(__AVX2__)
#include <immintrin.h>

/* Horizontal sum of the 4 int64 lanes in a 256-bit vector. */
static inline int64_t fxp_hsum_epi64(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi64(lo, hi);
    return (int64_t)_mm_extract_epi64(s, 0) + (int64_t)_mm_extract_epi64(s, 1);
}

/* int16 x int16 dot product, AVX2 path (256-bit: 16 elements/iter).
 *
 * Bit-exact with the scalar "acc += (int64_t)a[i]*(int64_t)b[i]" loop:
 * vpmaddwid multiplies 8 int16 pairs and horizontally sums adjacent products
 * into int32 lanes. Each int32 lane holds the sum of exactly 2 products, and
 * |32767 * 32768| * 2 == 2147418112 < INT32_MAX, so no intermediate overflow is
 * possible (weights are clipped to [-32767,32767], activations to [-32768,32767]).
 * Every step is widened to int64 before accumulating, and integer addition is
 * associative, so the summation order does not change the result. */
static inline int64_t fxp_dot_i16_i64_ymm(const int16_t* a, const int16_t* b, int n)
{
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        __m256i p  = _mm256_madd_epi16(va, vb);
        __m128i lo = _mm256_castsi256_si128(p);
        __m128i hi = _mm256_extracti128_si256(p, 1);
        acc0 = _mm256_add_epi64(acc0, _mm256_cvtepi32_epi64(lo));
        acc1 = _mm256_add_epi64(acc1, _mm256_cvtepi32_epi64(hi));
    }
    int64_t acc = fxp_hsum_epi64(acc0) + fxp_hsum_epi64(acc1);
    for (; i < n; i++)
        acc += (int64_t)a[i] * (int64_t)b[i];
    return acc;
}

#if defined(__GNUC__) && defined(__x86_64__)
/* AVX-512BW path (512-bit: 32 elements/iter). Same algorithm, wider lanes,
 * so it is bit-exact with the ymm path. Runtime-dispatched below. */
__attribute__((target("avx512bw")))
static int64_t fxp_dot_i16_i64_zmm(const int16_t* a, const int16_t* b, int n)
{
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512i va = _mm512_loadu_si512(a + i);
        __m512i vb = _mm512_loadu_si512(b + i);
        __m512i p  = _mm512_madd_epi16(va, vb);
        acc0 = _mm512_add_epi64(acc0, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(p)));
        acc1 = _mm512_add_epi64(acc1, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(p, 1)));
    }
    int64_t acc = _mm512_reduce_add_epi64(acc0) + _mm512_reduce_add_epi64(acc1);
    for (; i < n; i++)
        acc += (int64_t)a[i] * (int64_t)b[i];
    return acc;
}

/* Fused oc-group dot (AVX-512BW): MAC one xs vector against 4 consecutive weight
 * rows, traversing cin ONCE so the loaded xs register is reused across the whole
 * oc tile (xs read 1x from memory instead of 4x). Same per-lane int32->int64
 * widen/accumulate as fxp_dot_i16_i64_zmm, so it is bit-exact with 4 separate
 * dots: each out[k] == sum_i xs[i]*wk_base[k*cin+i] (integer MAC, reordered). */
__attribute__((target("avx512bw")))
static void fxp_conv1x1_dot_oc4_zmm(const int16_t* xs, const int16_t* wk_base,
                                    int cin, int64_t* out)
{
    __m512i a0l = _mm512_setzero_si512(), a0h = _mm512_setzero_si512();
    __m512i a1l = _mm512_setzero_si512(), a1h = _mm512_setzero_si512();
    __m512i a2l = _mm512_setzero_si512(), a2h = _mm512_setzero_si512();
    __m512i a3l = _mm512_setzero_si512(), a3h = _mm512_setzero_si512();
    const int16_t* w0 = wk_base;
    const int16_t* w1 = wk_base + (size_t)cin;
    const int16_t* w2 = wk_base + (size_t)2 * (size_t)cin;
    const int16_t* w3 = wk_base + (size_t)3 * (size_t)cin;
    int i = 0;
    for (; i + 32 <= cin; i += 32) {
        __m512i v = _mm512_loadu_si512(xs + i);
        __m512i p;
        p = _mm512_madd_epi16(v, _mm512_loadu_si512(w0 + i));
        a0l = _mm512_add_epi64(a0l, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(p)));
        a0h = _mm512_add_epi64(a0h, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(p, 1)));
        p = _mm512_madd_epi16(v, _mm512_loadu_si512(w1 + i));
        a1l = _mm512_add_epi64(a1l, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(p)));
        a1h = _mm512_add_epi64(a1h, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(p, 1)));
        p = _mm512_madd_epi16(v, _mm512_loadu_si512(w2 + i));
        a2l = _mm512_add_epi64(a2l, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(p)));
        a2h = _mm512_add_epi64(a2h, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(p, 1)));
        p = _mm512_madd_epi16(v, _mm512_loadu_si512(w3 + i));
        a3l = _mm512_add_epi64(a3l, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(p)));
        a3h = _mm512_add_epi64(a3h, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(p, 1)));
    }
    int64_t s0 = _mm512_reduce_add_epi64(a0l) + _mm512_reduce_add_epi64(a0h);
    int64_t s1 = _mm512_reduce_add_epi64(a1l) + _mm512_reduce_add_epi64(a1h);
    int64_t s2 = _mm512_reduce_add_epi64(a2l) + _mm512_reduce_add_epi64(a2h);
    int64_t s3 = _mm512_reduce_add_epi64(a3l) + _mm512_reduce_add_epi64(a3h);
    for (; i < cin; i++) {
        int16_t x = xs[i];
        s0 += (int64_t)x * (int64_t)w0[i];
        s1 += (int64_t)x * (int64_t)w1[i];
        s2 += (int64_t)x * (int64_t)w2[i];
        s3 += (int64_t)x * (int64_t)w3[i];
    }
    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
}

#include <cpuid.h>
static int fxp_have_avx512bw_ = 0;
__attribute__((constructor)) static void fxp_init_cpu_flags(void)
{
    unsigned eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        fxp_have_avx512bw_ = (ebx & (1u << 30)) != 0; /* AVX512BW */
}
#endif

/* Runtime-dispatched int16 dot product (int64 accumulation, bit-exact). */
static inline int64_t fxp_dot_i16_i64(const int16_t* a, const int16_t* b, int n)
{
#if defined(__GNUC__) && defined(__x86_64__)
    if (fxp_have_avx512bw_) return fxp_dot_i16_i64_zmm(a, b, n);
#endif
    return fxp_dot_i16_i64_ymm(a, b, n);
}

/* Depthwise 3x3 spatial-vectorization helpers (AVX2, per-column int64 accumulators).
 *
 * Unlike fxp_dot_i16_i64 (which collapses a channel dot into one scalar), these
 * keep a SEPARATE int64 accumulator per output column. Each int16*int16 product
 * fits in int32 (|32768*32767| = 1073709056 < INT32_MAX), is widened to int64 and
 * added to its own lane, so this is bit-exact with the scalar
 * "acc[i] += (int64_t)act[i]*(int64_t)w" loop (integer addition is associative). */

/* Add 8 activations * w (broadcast int16 weight) into two 4-lane int64 accums. */
static inline void fxp_dw3x3_mac8(__m256i* alo, __m256i* ahi,
                                  const int16_t* p, int w)
{
    __m128i v   = _mm_loadu_si128((const __m128i*)p);        /* 8 int16 */
    __m128i lo4 = _mm_cvtepi16_epi32(v);                     /* low  4 -> int32 */
    __m128i hi4 = _mm_cvtepi16_epi32(_mm_srli_si128(v, 8));  /* high 4 -> int32 */
    __m128i wv  = _mm_set1_epi32(w);
    __m128i pl  = _mm_mullo_epi32(lo4, wv);
    __m128i ph  = _mm_mullo_epi32(hi4, wv);
    *alo = _mm256_add_epi64(*alo, _mm256_cvtepi32_epi64(pl));
    *ahi = _mm256_add_epi64(*ahi, _mm256_cvtepi32_epi64(ph));
}

/* Add 4 activations * w into one 4-lane int64 accumulator. */
static inline void fxp_dw3x3_mac4(__m256i* a4, const int16_t* p, int w)
{
    __m128i v  = _mm_loadl_epi64((const __m128i*)p);         /* 4 int16 */
    __m128i v4 = _mm_cvtepi16_epi32(v);                      /* 4 -> int32 */
    __m128i wv = _mm_set1_epi32(w);
    __m128i p4 = _mm_mullo_epi32(v4, wv);
    *a4 = _mm256_add_epi64(*a4, _mm256_cvtepi32_epi64(p4));
}
/* Add 16 activations * w (broadcast int16 weight) into two 8-lane int64
 * accumulators (a0 = output cols 0..7, a1 = cols 8..15). AVX-512BW; same
 * per-lane int16->int32->int64 widen/accumulate as mac8, just wider, so it is
 * bit-exact with the scalar per-column MAC.  _mm512_cvtepi32_epi64 converts the
 * full 8 int32 of its __m256i argument into 8 int64 (same primitive the zmm dot
 * kernel relies on), so all 16 products land in a0/a1 with no drops. */
#if defined(__GNUC__) && defined(__x86_64__)
__attribute__((target("avx512bw")))
static inline void fxp_dw3x3_mac16(__m512i* a0, __m512i* a1,
                                   const int16_t* p, int w)
{
    __m256i v   = _mm256_loadu_si256((const __m256i*)p);     /* 16 int16 */
    __m512i e32 = _mm512_cvtepi16_epi32(v);                  /* 16 int32 */
    __m512i wv  = _mm512_set1_epi32(w);
    __m512i pr  = _mm512_mullo_epi32(e32, wv);               /* 16 int32 products */
    /* Split into two 8-int32 halves and widen each to 8 int64 (a0=cols0..7,
     * a1=cols8..15) -- same low256/high256 split as the zmm dot kernel. */
    *a0 = _mm512_add_epi64(*a0, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(pr)));
    *a1 = _mm512_add_epi64(*a1, _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(pr, 1)));
}

/* Full 16-output-column MAC for one row: writes 16 int64 accumulators (one per
 * output column) to out16[]. Kept entirely inside a target("avx512bw") function
 * because it issues _mm512_* instructions that the AVX2-only oc_range body cannot
 * emit. It does NOT do the float dequant -- that is done by the caller in the
 * plain (non-target) oc_range body so the (float)acc*scale+b contraction matches
 * the mac8/mac4/scalar paths and the golden reference exactly (no 1-ULP FMA
 * drift between target-attributed and plain code). */
__attribute__((target("avx512bw")))
static inline void fxp_dw3x3_row_mac16_accum(
    const int16_t* r0, const int16_t* r1, const int16_t* r2, int xw,
    int16_t w00, int16_t w01, int16_t w02,
    int16_t w10, int16_t w11, int16_t w12,
    int16_t w20, int16_t w21, int16_t w22,
    int64_t* out16)
{
    __m512i a0 = _mm512_setzero_si512();
    __m512i a1 = _mm512_setzero_si512();
    const int16_t* p0 = r0 + xw;
    const int16_t* p1 = r1 + xw;
    const int16_t* p2 = r2 + xw;
    fxp_dw3x3_mac16(&a0, &a1, p0, w00);
    fxp_dw3x3_mac16(&a0, &a1, p0 + 1, w01);
    fxp_dw3x3_mac16(&a0, &a1, p0 + 2, w02);
    fxp_dw3x3_mac16(&a0, &a1, p1, w10);
    fxp_dw3x3_mac16(&a0, &a1, p1 + 1, w11);
    fxp_dw3x3_mac16(&a0, &a1, p1 + 2, w12);
    fxp_dw3x3_mac16(&a0, &a1, p2, w20);
    fxp_dw3x3_mac16(&a0, &a1, p2 + 1, w21);
    fxp_dw3x3_mac16(&a0, &a1, p2 + 2, w22);
    /* a0 = 8 int64 (cols 0..7); a1 = 8 int64 (cols 8..15). Extract all 16. */
    __m256i a0lo = _mm512_castsi512_si256(a0);
    __m256i a0hi = _mm512_extracti64x4_epi64(a0, 1);
    __m256i a1lo = _mm512_castsi512_si256(a1);
    __m256i a1hi = _mm512_extracti64x4_epi64(a1, 1);
    out16[ 0] = _mm256_extract_epi64(a0lo, 0);
    out16[ 1] = _mm256_extract_epi64(a0lo, 1);
    out16[ 2] = _mm256_extract_epi64(a0lo, 2);
    out16[ 3] = _mm256_extract_epi64(a0lo, 3);
    out16[ 4] = _mm256_extract_epi64(a0hi, 0);
    out16[ 5] = _mm256_extract_epi64(a0hi, 1);
    out16[ 6] = _mm256_extract_epi64(a0hi, 2);
    out16[ 7] = _mm256_extract_epi64(a0hi, 3);
    out16[ 8] = _mm256_extract_epi64(a1lo, 0);
    out16[ 9] = _mm256_extract_epi64(a1lo, 1);
    out16[10] = _mm256_extract_epi64(a1lo, 2);
    out16[11] = _mm256_extract_epi64(a1lo, 3);
    out16[12] = _mm256_extract_epi64(a1hi, 0);
    out16[13] = _mm256_extract_epi64(a1hi, 1);
    out16[14] = _mm256_extract_epi64(a1hi, 2);
    out16[15] = _mm256_extract_epi64(a1hi, 3);
}
#endif

/* Fused oc-group dot (AVX2): same idea as the zmm kernel, 256-bit lanes. */
static void fxp_conv1x1_dot_oc4_ymm(const int16_t* xs, const int16_t* wk_base,
                                    int cin, int64_t* out)
{
    __m256i a0l = _mm256_setzero_si256(), a0h = _mm256_setzero_si256();
    __m256i a1l = _mm256_setzero_si256(), a1h = _mm256_setzero_si256();
    __m256i a2l = _mm256_setzero_si256(), a2h = _mm256_setzero_si256();
    __m256i a3l = _mm256_setzero_si256(), a3h = _mm256_setzero_si256();
    const int16_t* w0 = wk_base;
    const int16_t* w1 = wk_base + (size_t)cin;
    const int16_t* w2 = wk_base + (size_t)2 * (size_t)cin;
    const int16_t* w3 = wk_base + (size_t)3 * (size_t)cin;
    int i = 0;
    for (; i + 16 <= cin; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(xs + i));
        __m256i p;
        p = _mm256_madd_epi16(v, _mm256_loadu_si256((const __m256i*)(w0 + i)));
        a0l = _mm256_add_epi64(a0l, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p)));
        a0h = _mm256_add_epi64(a0h, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(p, 1)));
        p = _mm256_madd_epi16(v, _mm256_loadu_si256((const __m256i*)(w1 + i)));
        a1l = _mm256_add_epi64(a1l, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p)));
        a1h = _mm256_add_epi64(a1h, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(p, 1)));
        p = _mm256_madd_epi16(v, _mm256_loadu_si256((const __m256i*)(w2 + i)));
        a2l = _mm256_add_epi64(a2l, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p)));
        a2h = _mm256_add_epi64(a2h, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(p, 1)));
        p = _mm256_madd_epi16(v, _mm256_loadu_si256((const __m256i*)(w3 + i)));
        a3l = _mm256_add_epi64(a3l, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p)));
        a3h = _mm256_add_epi64(a3h, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(p, 1)));
    }
    int64_t s0 = fxp_hsum_epi64(a0l) + fxp_hsum_epi64(a0h);
    int64_t s1 = fxp_hsum_epi64(a1l) + fxp_hsum_epi64(a1h);
    int64_t s2 = fxp_hsum_epi64(a2l) + fxp_hsum_epi64(a2h);
    int64_t s3 = fxp_hsum_epi64(a3l) + fxp_hsum_epi64(a3h);
    for (; i < cin; i++) {
        int16_t x = xs[i];
        s0 += (int64_t)x * (int64_t)w0[i];
        s1 += (int64_t)x * (int64_t)w1[i];
        s2 += (int64_t)x * (int64_t)w2[i];
        s3 += (int64_t)x * (int64_t)w3[i];
    }
    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
}

/* Runtime-dispatched fused oc-group dot (noc==4 fast path). Reuses the loaded xs
 * vector across the whole oc tile; for partial tiles (noc<4) falls back to plain
 * per-oc dots. Output identical to noc independent fxp_dot_i16_i64 calls. */
static inline void fxp_conv1x1_dot_oc(const int16_t* xs, const int16_t* wk_base,
                                      int cin, int noc, int64_t* out)
{
#if defined(__GNUC__) && defined(__x86_64__)
    if (noc == 4 && fxp_have_avx512bw_) {
        fxp_conv1x1_dot_oc4_zmm(xs, wk_base, cin, out);
        return;
    }
#endif
    if (noc == 4) {
        fxp_conv1x1_dot_oc4_ymm(xs, wk_base, cin, out);
        return;
    }
    for (int oi = 0; oi < noc; oi++)
        out[oi] = fxp_dot_i16_i64(xs, wk_base + (size_t)oi * cin, cin);
}
#endif /* __AVX2__ */

/* TLS scratch for single-threaded entry points (tests / fallback). */
static FXP_TLS int16_t* fxp_tls_i16 = NULL;
static FXP_TLS size_t fxp_tls_i16_cap = 0;
static FXP_TLS int32_t* fxp_tls_i32 = NULL;
static FXP_TLS int16_t* fxp_tls_i16_col = NULL;
static FXP_TLS size_t fxp_tls_i16_col_cap = 0;
static FXP_TLS size_t fxp_tls_i32_cap = 0;

static int16_t* fxp_scratch_i16(size_t n_elem)
{
    if (n_elem > fxp_tls_i16_cap) {
        int16_t* p = (int16_t*)realloc(fxp_tls_i16, n_elem * sizeof(int16_t));
        if (!p) return NULL;
        fxp_tls_i16 = p;
        fxp_tls_i16_cap = n_elem;
    }
    return fxp_tls_i16;
}

static int32_t* fxp_scratch_i32(size_t n_elem)
{
    if (n_elem > fxp_tls_i32_cap) {
        int32_t* p = (int32_t*)realloc(fxp_tls_i32, n_elem * sizeof(int32_t));
        if (!p) return NULL;
        fxp_tls_i32 = p;
        fxp_tls_i32_cap = n_elem;
    }
    return fxp_tls_i32;
}

static int16_t* fxp_scratch_i16_col(size_t n_elem)
{
    if (n_elem > fxp_tls_i16_col_cap) {
        int16_t* p = (int16_t*)realloc(fxp_tls_i16_col, n_elem * sizeof(int16_t));
        if (!p) return NULL;
        fxp_tls_i16_col = p;
        fxp_tls_i16_col_cap = n_elem;
    }
    return fxp_tls_i16_col;
}

enum { FXP_OC_TILE = 4, FXP_S_TILE = 8 };

void fxp_conv1x1_pack_i16(const float* x_nchw, int16_t* xq_hw_cin,
                          int cin, int hw, float x_scale)
{
    const float inv_x = 1.0f / x_scale;
    /* Write NHWC with spatial-major stores (better than ic-outer strided stores). */
    for (int s = 0; s < hw; s++) {
        int16_t* row = xq_hw_cin + (size_t)s * cin;
        for (int ic = 0; ic < cin; ic++)
            row[ic] = fxp_quantize_act(x_nchw[(size_t)ic * hw + s], inv_x);
    }
}

void fxp_conv1x1_oc_range_i16(const int16_t* xq_hw_cin, float* y_nchw,
                              int cin, int cout, int hw,
                              const int16_t* w_int, const float* w_scale,
                              const float* bias, float x_scale,
                              int oc_start, int oc_end)
{
    (void)cout;
    /* OC-blocking with a fused oc-group dot: for each spatial position the packed
     * int16 activation vector xs (length cin) is traversed ONCE and MAC'd against
     * the whole output-channel tile, reusing the loaded xs registers across the oc
     * group (instead of re-reading xs once per output channel). The spatial loop is
     * still innermost so xs streams contiguously; only the memory access pattern
     * changes -- every output element is the identical int64 dot product, so the
     * result is bit-exact with the per-(oc,s) scalar form. */
    for (int oc0 = oc_start; oc0 < oc_end; oc0 += FXP_OC_TILE) {
        const int noc = oc_end - oc0 < FXP_OC_TILE ? oc_end - oc0 : FXP_OC_TILE;
        const int16_t* wk_base = w_int + (size_t)oc0 * cin;
        for (int s0 = 0; s0 < hw; s0 += FXP_S_TILE) {
            const int ns = hw - s0 < FXP_S_TILE ? hw - s0 : FXP_S_TILE;
            for (int si = 0; si < ns; si++) {
                const int s = s0 + si;
                const int16_t* xs = xq_hw_cin + (size_t)s * cin;
                int64_t acc[FXP_OC_TILE];
#if defined(__AVX2__)
                fxp_conv1x1_dot_oc(xs, wk_base, cin, noc, acc);
#else
                for (int oi = 0; oi < noc; oi++) {
                    const int16_t* wk = wk_base + (size_t)oi * cin;
                    int64_t a = 0;
                    for (int ic = 0; ic < cin; ic++)
                        a += (int64_t)xs[ic] * (int64_t)wk[ic];
                    acc[oi] = a;
                }
#endif
                for (int oi = 0; oi < noc; oi++) {
                    const int oc = oc0 + oi;
                    y_nchw[(size_t)oc * hw + s] =
                        (float)acc[oi] * (x_scale * w_scale[oc]) + bias[oc];
                }
            }
        }
    }
}

void fxp_dw3x3_pack_i16(const float* x_nchw, int16_t* xq_pad,
                        int c, int h, int w, float x_scale)
{
    const float inv_x = 1.0f / x_scale;
    const int Wpad = w + 2;
    const size_t cpad = (size_t)(h + 2) * (size_t)Wpad;
    /* realloc does NOT zero memory: clear the whole padded buffer first so every
     * border cell (top/bottom rows, left/right columns) is exactly 0. */
    memset(xq_pad, 0, cpad * (size_t)c * sizeof(int16_t));
    for (int ic = 0; ic < c; ic++) {
        const float* xc = x_nchw + (size_t)ic * (size_t)h * (size_t)w;
        /* qc points at padded row 1, col 1 (first data cell). */
        int16_t* qc = xq_pad + (size_t)ic * cpad + (size_t)Wpad + 1;
        for (int r = 0; r < h; r++) {
            const float* src = xc + (size_t)r * (size_t)w;
            int16_t* dst = qc + (size_t)r * (size_t)Wpad;
            for (int cc = 0; cc < w; cc++)
                dst[cc] = fxp_quantize_act(src[cc], inv_x);
            /* dst[-1] (col 0) and dst[w] (col w+1) stay 0 from the memset above. */
        }
    }
}

void fxp_dw3x3_oc_range_i16(const int16_t* xq_pad, float* y_nchw,
                            int c, int h, int w,
                            const int16_t* w_int, const float* w_scale,
                            const float* bias, float x_scale,
                            int oc_start, int oc_end)
{
    (void)c;
    const int hw = h * w;
    const int Wpad = w + 2;
    const size_t cpad = (size_t)(h + 2) * (size_t)Wpad;
    for (int oc = oc_start; oc < oc_end; oc++) {
        const int16_t* wk = w_int + (size_t)oc * 9;
        float* yo = y_nchw + (size_t)oc * hw;
        const int16_t* qi = xq_pad + (size_t)oc * cpad;  /* padded [h+2][w+2] */
        const float scale = x_scale * w_scale[oc];
        const float b = bias[oc];
        const int16_t w00 = wk[0], w01 = wk[1], w02 = wk[2];
        const int16_t w10 = wk[3], w11 = wk[4], w12 = wk[5];
        const int16_t w20 = wk[6], w21 = wk[7], w22 = wk[8];

        /* Padded layout: output pixel (yh,xw) has its 3x3 window at padded
         * rows [yh, yh+1, yh+2] and columns [xw, xw+1, xw+2]. The window left
         * column is xw (NOT xw-1). With zero padding there are NO bounds checks
         * and NO border special-casing -- border cells are 0, contributing 0*w=0
         * to the int64 sum, identical to the old out-of-bounds->0 rule. */
        for (int yh = 0; yh < h; yh++) {
            const int16_t* r0 = qi + (size_t)(yh + 0) * Wpad;
            const int16_t* r1 = qi + (size_t)(yh + 1) * Wpad;
            const int16_t* r2 = qi + (size_t)(yh + 2) * Wpad;
            float* yor = yo + (size_t)yh * w;
            int xw = 0;
#if defined(__AVX2__)
#if defined(__GNUC__) && defined(__x86_64__)
            if (fxp_have_avx512bw_) {
                for (; xw + 16 <= w; xw += 16) {
                    int64_t acc16[16];
                    fxp_dw3x3_row_mac16_accum(r0, r1, r2, xw,
                                              w00, w01, w02, w10, w11, w12,
                                              w20, w21, w22, acc16);
                    for (int i = 0; i < 16; i++)
                        yor[xw + i] = (float)acc16[i] * scale + b;
                }
            }
#endif
            for (; xw + 8 <= w; xw += 8) {
                __m256i alo = _mm256_setzero_si256();
                __m256i ahi = _mm256_setzero_si256();
                const int16_t* p0 = r0 + xw;
                const int16_t* p1 = r1 + xw;
                const int16_t* p2 = r2 + xw;
                fxp_dw3x3_mac8(&alo, &ahi, p0, w00);
                fxp_dw3x3_mac8(&alo, &ahi, p0 + 1, w01);
                fxp_dw3x3_mac8(&alo, &ahi, p0 + 2, w02);
                fxp_dw3x3_mac8(&alo, &ahi, p1, w10);
                fxp_dw3x3_mac8(&alo, &ahi, p1 + 1, w11);
                fxp_dw3x3_mac8(&alo, &ahi, p1 + 2, w12);
                fxp_dw3x3_mac8(&alo, &ahi, p2, w20);
                fxp_dw3x3_mac8(&alo, &ahi, p2 + 1, w21);
                fxp_dw3x3_mac8(&alo, &ahi, p2 + 2, w22);
                yor[xw + 0] = (float)_mm256_extract_epi64(alo, 0) * scale + b;
                yor[xw + 1] = (float)_mm256_extract_epi64(alo, 1) * scale + b;
                yor[xw + 2] = (float)_mm256_extract_epi64(alo, 2) * scale + b;
                yor[xw + 3] = (float)_mm256_extract_epi64(alo, 3) * scale + b;
                yor[xw + 4] = (float)_mm256_extract_epi64(ahi, 0) * scale + b;
                yor[xw + 5] = (float)_mm256_extract_epi64(ahi, 1) * scale + b;
                yor[xw + 6] = (float)_mm256_extract_epi64(ahi, 2) * scale + b;
                yor[xw + 7] = (float)_mm256_extract_epi64(ahi, 3) * scale + b;
            }
            for (; xw + 4 <= w; xw += 4) {
                __m256i a4 = _mm256_setzero_si256();
                const int16_t* p0 = r0 + xw;
                const int16_t* p1 = r1 + xw;
                const int16_t* p2 = r2 + xw;
                fxp_dw3x3_mac4(&a4, p0, w00);
                fxp_dw3x3_mac4(&a4, p0 + 1, w01);
                fxp_dw3x3_mac4(&a4, p0 + 2, w02);
                fxp_dw3x3_mac4(&a4, p1, w10);
                fxp_dw3x3_mac4(&a4, p1 + 1, w11);
                fxp_dw3x3_mac4(&a4, p1 + 2, w12);
                fxp_dw3x3_mac4(&a4, p2, w20);
                fxp_dw3x3_mac4(&a4, p2 + 1, w21);
                fxp_dw3x3_mac4(&a4, p2 + 2, w22);
                yor[xw + 0] = (float)_mm256_extract_epi64(a4, 0) * scale + b;
                yor[xw + 1] = (float)_mm256_extract_epi64(a4, 1) * scale + b;
                yor[xw + 2] = (float)_mm256_extract_epi64(a4, 2) * scale + b;
                yor[xw + 3] = (float)_mm256_extract_epi64(a4, 3) * scale + b;
            }
#endif
            for (; xw < w; xw++) {
                const int16_t* c0 = r0 + xw;
                const int16_t* c1 = r1 + xw;
                const int16_t* c2 = r2 + xw;
                int64_t acc =
                    (int64_t)c0[0] * w00 + (int64_t)c0[1] * w01 + (int64_t)c0[2] * w02 +
                    (int64_t)c1[0] * w10 + (int64_t)c1[1] * w11 + (int64_t)c1[2] * w12 +
                    (int64_t)c2[0] * w20 + (int64_t)c2[1] * w21 + (int64_t)c2[2] * w22;
                yor[xw] = (float)acc * scale + b;
            }
        }
    }
}

void fxp_wsrelu_nchw_oc_range(float* y_nchw, int hw,
                              int oc_start, int oc_end,
                              const float* lut65536, float wsrelu_x_scale)
{
    for (int oc = oc_start; oc < oc_end; oc++) {
        float* yo = y_nchw + (size_t)oc * hw;
        fxp_wsrelu_f32_range(yo, yo, 0, hw, lut65536, wsrelu_x_scale);
    }
}

static void fxp_conv1x1_fast(const float* x, float* y,
                             int n, int cin, int cout, int h, int w,
                             const int16_t* w_int, const float* w_scale,
                             const float* bias, float x_scale, int act_bits)
{
    if (act_bits >= 24) {
        /* int32 activation path (rare): keep prior int32 pack loop. */
        const float inv_x = 1.0f / x_scale;
        const int hw = h * w;
        int32_t* xq = fxp_scratch_i32((size_t)cin * (size_t)hw);
        if (!xq) return;
        for (int ni = 0; ni < n; ni++) {
            const float* x_n = x + (size_t)ni * cin * hw;
            float* y_n = y + (size_t)ni * cout * hw;
            for (int s = 0; s < hw; s++) {
                int32_t* row = xq + (size_t)s * cin;
                for (int ic = 0; ic < cin; ic++)
                    row[ic] = fxp_quantize_act_i32(x_n[(size_t)ic * hw + s], inv_x);
            }
            for (int oc = 0; oc < cout; oc++) {
                const int16_t* wk = w_int + (size_t)oc * cin;
                float* yo = y_n + (size_t)oc * hw;
                const float scale = x_scale * w_scale[oc];
                const float b = bias[oc];
                for (int s = 0; s < hw; s++) {
                    const int32_t* xs = xq + (size_t)s * cin;
                    int64_t acc = 0;
                    for (int ic = 0; ic < cin; ic++)
                        acc += (int64_t)xs[ic] * (int64_t)wk[ic];
                    yo[s] = (float)acc * scale + b;
                }
            }
        }
        return;
    }

    const int hw = h * w;
    int16_t* xq = fxp_scratch_i16((size_t)cin * (size_t)hw);
    if (!xq) return;
    for (int ni = 0; ni < n; ni++) {
        const float* x_n = x + (size_t)ni * cin * hw;
        float* y_n = y + (size_t)ni * cout * hw;
        fxp_conv1x1_pack_i16(x_n, xq, cin, hw, x_scale);
        fxp_conv1x1_oc_range_i16(xq, y_n, cin, cout, hw, w_int, w_scale, bias,
                                 x_scale, 0, cout);
    }
}

static void fxp_dwconv3x3_fast(const float* x, float* y,
                               int n, int c, int h, int w,
                               const int16_t* w_int, const float* w_scale,
                               const float* bias, float x_scale, int act_bits)
{
    if (act_bits >= 24) {
        /* Fallback: quantize inline as int32 (uncommon). */
        const float inv_x = 1.0f / x_scale;
        const int hw = h * w;
        int32_t* xq = fxp_scratch_i32((size_t)c * (size_t)hw);
        if (!xq) return;
        for (int ni = 0; ni < n; ni++) {
            const float* x_n = x + (size_t)ni * c * hw;
            float* y_n = y + (size_t)ni * c * hw;
            for (int ic = 0; ic < c; ic++) {
                const float* xc = x_n + (size_t)ic * hw;
                int32_t* qc = xq + (size_t)ic * hw;
                for (int s = 0; s < hw; s++)
                    qc[s] = fxp_quantize_act_i32(xc[s], inv_x);
            }
            /* Reuse dw range by temporarily widening — call via float path copy. */
            for (int oc = 0; oc < c; oc++) {
                const int16_t* wk = w_int + (size_t)oc * 9;
                float* yo = y_n + (size_t)oc * hw;
                const int32_t* qi = xq + (size_t)oc * hw;
                const float scale = x_scale * w_scale[oc];
                const float b = bias[oc];
                for (int yh = 0; yh < h; yh++) {
                    for (int xw = 0; xw < w; xw++) {
                        int64_t acc = 0;
                        for (int kh = 0; kh < 3; kh++) {
                            int ih = yh + kh - 1;
                            for (int kw = 0; kw < 3; kw++) {
                                int iw = xw + kw - 1;
                                int32_t v = 0;
                                if ((unsigned)ih < (unsigned)h && (unsigned)iw < (unsigned)w)
                                    v = qi[ih * w + iw];
                                acc += (int64_t)v * (int64_t)wk[kh * 3 + kw];
                            }
                        }
                        yo[yh * w + xw] = (float)acc * scale + b;
                    }
                }
            }
        }
        return;
    }

    const int hw = h * w;
    int16_t* xq = fxp_scratch_i16((size_t)c * (size_t)(h + 2) * (size_t)(w + 2));
    if (!xq) return;
    for (int ni = 0; ni < n; ni++) {
        const float* x_n = x + (size_t)ni * c * hw;
        float* y_n = y + (size_t)ni * c * hw;
        fxp_dw3x3_pack_i16(x_n, xq, c, h, w, x_scale);
        fxp_dw3x3_oc_range_i16(xq, y_n, c, h, w, w_int, w_scale, bias, x_scale, 0, c);
    }
}

/* ── im2col + SIMD GEMM for general (non-1x1, non-dw) convs ──────────────
 *
 * The old general path was scalar int64 MAC with per-access re-quantization
 * (each input value quantized up to kh*kw times). This version:
 *   1. Pre-quantizes the input ONCE (NCHW float → int16)
 *   2. Builds an im2col matrix [oh*ow][cin*k_area] of int16
 *   3. SIMD dot-product each im2col row with the weight vector
 *
 * Bit-exact with the scalar path: the int16 quantized values are identical
 * (same float input, same inv_x, deterministic rounding), and integer addition
 * is associative so the MAC reordering does not change the result.
 */

/* Build im2col for one spatial tile [s0, s0+ns) into a contiguous int16 buffer.
 * Layout per row: [cin][kh][kw] (matches weight [cout][cin][kh][kw] inner order).
 * Out-of-bounds positions are zero (same as the scalar bounds check). */
void fxp_im2col_tile(const int16_t* xq_nchw, int cin, int h, int w,
                            int kh, int kw, int pad_t, int pad_l,
                            int stride_h, int stride_w,
                            int oh, int ow, int s0, int ns,
                            int16_t* col /* [ns][cin*kh*kw] */)
{
    const int x_hw = h * w;
    const int k_area = kh * kw;
    const int row_len = cin * k_area;

    for (int si = 0; si < ns; si++) {
        const int spat = s0 + si;
        const int oh_i = spat / ow;
        const int ow_i = spat - oh_i * ow;
        int16_t* row = col + (size_t)si * row_len;
        int idx = 0;
        for (int ic = 0; ic < cin; ic++) {
            const int16_t* xc = xq_nchw + (size_t)ic * x_hw;
            for (int ki = 0; ki < kh; ki++) {
                const int ih = oh_i * stride_h + ki - pad_t;
                for (int kj = 0; kj < kw; kj++) {
                    const int iw = ow_i * stride_w + kj - pad_l;
                    int16_t v = 0;
                    if ((unsigned)ih < (unsigned)h && (unsigned)iw < (unsigned)w)
                        v = xc[ih * w + iw];
                    row[idx++] = v;
                }
            }
        }
    }
}

/* im2col GEMM over an oc-range: for each output channel in [oc_start,oc_end),
 * dot-product every im2col row with the weight vector.
 * ns = number of spatial positions in the current tile. */
void fxp_conv_im2col_oc_range(const int16_t* col, float* y_nchw,
                              int cin, int cout, int oh, int ow,
                              const int16_t* w_int, const float* w_scale,
                              const float* bias, float x_scale,
                              int kh, int kw,
                              int oc_start, int oc_end)
{
    const int k_area = kh * kw;
    const int row_len = cin * k_area;
    const int y_hw = oh * ow;

    for (int oc = oc_start; oc < oc_end; oc++) {
        const int16_t* wk = w_int + (size_t)oc * row_len;
        float* yo = y_nchw + (size_t)oc * y_hw;
        const float scale = x_scale * w_scale[oc];
        const float b = bias[oc];
        for (int s = 0; s < y_hw; s++) {
            const int16_t* row = col + (size_t)s * row_len;
#if defined(__AVX2__)
            int64_t acc = fxp_dot_i16_i64(row, wk, row_len);
#else
            int64_t acc = 0;
            for (int i = 0; i < row_len; i++)
                acc += (int64_t)row[i] * (int64_t)wk[i];
#endif
            yo[s] = (float)acc * scale + b;
        }
    }
}

/* Exposed build: quantize + im2col into TLS scratch. Returns col pointer. */
int16_t* fxp_conv_im2col_build(const float* x_nchw, int cin, int h, int w,
                               int kh, int kw, int pad_t, int pad_l,
                               int stride_h, int stride_w,
                               int oh, int ow, float x_scale)
{
    const float inv_x = 1.0f / x_scale;
    const int x_hw = h * w;

    int16_t* xq = fxp_scratch_i16((size_t)cin * (size_t)x_hw);
    if (!xq) return NULL;
    for (int i = 0; i < cin * x_hw; i++)
        xq[i] = fxp_quantize_act(x_nchw[i], inv_x);

    const int k_area = kh * kw;
    int16_t* col = fxp_scratch_i16_col((size_t)oh * (size_t)ow *
                                       (size_t)cin * (size_t)k_area);
    if (!col) return NULL;
    fxp_im2col_tile(xq, cin, h, w, kh, kw, pad_t, pad_l,
                    stride_h, stride_w, oh, ow, 0,
                    oh * ow, col);
    return col;
}

/* Full im2col fast path: pre-quantize, build col, SIMD GEMM.
 * Replaces the 7-nested-loop scalar general path. */
static void fxp_conv_im2col_fast(const float* x, float* y,
                                 int n, int cin, int cout, int h, int w,
                                 const int16_t* w_int, const float* w_scale,
                                 const float* bias, float x_scale, int act_bits,
                                 int kh, int kw,
                                 int pad_t, int pad_l, int pad_b, int pad_r,
                                 int stride_h, int stride_w, int group)
{
    /* Only group==1 uses this path (grouped convs are rare in these models). */
    const float inv_x = 1.0f / x_scale;
    const int x_hw = h * w;
    const int oh = (h + pad_t + pad_b - kh) / stride_h + 1;
    const int ow = (w + pad_l + pad_r - kw) / stride_w + 1;
    const int y_hw = oh * ow;
    const int k_area = kh * kw;

    /* Pre-quantize input ONCE: NCHW float → int16. */
    int16_t* xq = fxp_scratch_i16((size_t)cin * (size_t)x_hw);
    if (!xq) return;

    /* im2col buffer: [y_hw][cin*k_area] int16. */
    int16_t* col = fxp_scratch_i16_col((size_t)y_hw * (size_t)cin * (size_t)k_area);
    if (!col) return;

    for (int ni = 0; ni < n; ni++) {
        const float* x_n = x + (size_t)ni * cin * x_hw;
        float* y_n = y + (size_t)ni * cout * y_hw;

        /* Quantize input. */
        for (int i = 0; i < cin * x_hw; i++) {
            if (act_bits >= 24)
                xq[i] = (int16_t)fxp_quantize_act_i32(x_n[i], inv_x);
            else
                xq[i] = fxp_quantize_act(x_n[i], inv_x);
        }

        /* Build full im2col matrix. */
        fxp_im2col_tile(xq, cin, h, w, kh, kw, pad_t, pad_l,
                        stride_h, stride_w, oh, ow, 0, y_hw, col);

        /* SIMD GEMM over all output channels. */
        fxp_conv_im2col_oc_range(col, y_n, cin, cout, oh, ow,
                                 w_int, w_scale, bias, x_scale,
                                 kh, kw, 0, cout);
    }
}

void fxp_conv_f32(const float* x, float* y,
                  int n, int cin, int cout, int h, int w,
                  const int16_t* w_int, const float* w_scale, const float* bias,
                  float x_scale,
                  int kh, int kw,
                  int pad_t, int pad_l, int pad_b, int pad_r,
                  int stride_h, int stride_w, int group, int act_bits)
{
    if (kh == 1 && kw == 1 && group == 1
        && pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0
        && stride_h == 1 && stride_w == 1) {
        fxp_conv1x1_fast(x, y, n, cin, cout, h, w, w_int, w_scale, bias, x_scale, act_bits);
        return;
    }
    if (kh == 3 && kw == 3 && group == cin && cin == cout
        && pad_t == 1 && pad_l == 1 && pad_b == 1 && pad_r == 1
        && stride_h == 1 && stride_w == 1) {
        fxp_dwconv3x3_fast(x, y, n, cin, h, w, w_int, w_scale, bias, x_scale, act_bits);
        return;
    }

    /* im2col + SIMD GEMM for group==1 general convs (was scalar, ~100x slower). */
    if (group == 1 && act_bits < 24) {
        fxp_conv_im2col_fast(x, y, n, cin, cout, h, w, w_int, w_scale, bias,
                             x_scale, act_bits, kh, kw,
                             pad_t, pad_l, pad_b, pad_r, stride_h, stride_w, group);
        return;
    }

    const float inv_x = 1.0f / x_scale;
    const int cin_g = cin / group;
    const int cout_g = cout / group;
    const int oh = (h + pad_t + pad_b - kh) / stride_h + 1;
    const int ow = (w + pad_l + pad_r - kw) / stride_w + 1;
    const int x_hw = h * w;
    const int y_hw = oh * ow;
    const int k_area = kh * kw;
    const int w_inner = cin_g * k_area;

    for (int ni = 0; ni < n; ni++) {
        const float* x_n = x + (size_t)ni * cin * x_hw;
        float* y_n = y + (size_t)ni * cout * y_hw;

        for (int oh_i = 0; oh_i < oh; oh_i++) {
            for (int ow_i = 0; ow_i < ow; ow_i++) {
                const int y_spat = oh_i * ow + ow_i;
                for (int g = 0; g < group; g++) {
                    for (int oc_g = 0; oc_g < cout_g; oc_g++) {
                        const int oc = g * cout_g + oc_g;
                        const int16_t* wk = w_int + (size_t)oc * w_inner;
                        int64_t acc = 0;
                        for (int ic_g = 0; ic_g < cin_g; ic_g++) {
                            const int ic = g * cin_g + ic_g;
                            const float* x_c = x_n + (size_t)ic * x_hw;
                            const int16_t* w_ic = wk + (size_t)ic_g * k_area;
                            for (int kh_i = 0; kh_i < kh; kh_i++) {
                                const int ih = oh_i * stride_h + kh_i - pad_t;
                                for (int kw_i = 0; kw_i < kw; kw_i++) {
                                    const int iw = ow_i * stride_w + kw_i - pad_l;
                                    int32_t xq = 0;
                                    if ((unsigned)ih < (unsigned)h && (unsigned)iw < (unsigned)w) {
                                        if (act_bits >= 24)
                                            xq = fxp_quantize_act_i32(x_c[ih * w + iw], inv_x);
                                        else
                                            xq = fxp_quantize_act(x_c[ih * w + iw], inv_x);
                                    }
                                    acc += (int64_t)xq * (int64_t)w_ic[kh_i * kw + kw_i];
                                }
                            }
                        }
                        float scale = x_scale * w_scale[oc];
                        y_n[(size_t)oc * y_hw + y_spat] = (float)acc * scale + bias[oc];
                    }
                }
            }
        }
    }
}

void fxp_conv1x1_f32(const float* x, float* y,
                     int n, int cin, int cout, int h, int w,
                     const int16_t* w_int, const float* w_scale,
                     const float* bias, float x_scale, int act_bits)
{
    fxp_conv1x1_fast(x, y, n, cin, cout, h, w, w_int, w_scale, bias, x_scale, act_bits);
}

void fxp_conv_f32_range(const float* x, float* y,
                  int n, int cin, int cout, int h, int w,
                  const int16_t* w_int, const float* w_scale, const float* bias,
                  float x_scale,
                  int kh, int kw,
                  int pad_t, int pad_l, int pad_b, int pad_r,
                  int stride_h, int stride_w, int group, int act_bits,
                  int oc_start, int oc_end)
{
    if (kh == 1 && kw == 1 && group == 1
        && pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0
        && stride_h == 1 && stride_w == 1 && act_bits < 24) {
        const int hw = h * w;
        int16_t* xq = fxp_scratch_i16((size_t)cin * (size_t)hw);
        if (!xq) return;
        for (int ni = 0; ni < n; ni++) {
            const float* x_n = x + (size_t)ni * cin * hw;
            float* y_n = y + (size_t)ni * cout * hw;
            fxp_conv1x1_pack_i16(x_n, xq, cin, hw, x_scale);
            fxp_conv1x1_oc_range_i16(xq, y_n, cin, cout, hw, w_int, w_scale, bias,
                                     x_scale, oc_start, oc_end);
        }
        return;
    }
    if (kh == 3 && kw == 3 && group == cin && cin == cout
        && pad_t == 1 && pad_l == 1 && pad_b == 1 && pad_r == 1
        && stride_h == 1 && stride_w == 1 && act_bits < 24) {
        const int hw = h * w;
        int16_t* xq = fxp_scratch_i16((size_t)cin * (size_t)(h + 2) * (size_t)(w + 2));
        if (!xq) return;
        for (int ni = 0; ni < n; ni++) {
            const float* x_n = x + (size_t)ni * cin * hw;
            float* y_n = y + (size_t)ni * cout * hw;
            fxp_dw3x3_pack_i16(x_n, xq, cin, h, w, x_scale);
            fxp_dw3x3_oc_range_i16(xq, y_n, cin, h, w, w_int, w_scale, bias,
                                   x_scale, oc_start, oc_end);
        }
        return;
    }
    /* Slow fallback: full conv (ignores range — caller must be single-threaded). */
    (void)oc_start; (void)oc_end;
    fxp_conv_f32(x, y, n, cin, cout, h, w, w_int, w_scale, bias, x_scale,
                 kh, kw, pad_t, pad_l, pad_b, pad_r,
                 stride_h, stride_w, group, act_bits);
}
