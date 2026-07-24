#ifndef DCVC_FXP_COMMON_H
#define DCVC_FXP_COMMON_H

#include <math.h>
#include <stdint.h>

/* Deterministic half-away-from-zero → int16 (for WsRelu LUT indexing). */
static inline int16_t fxp_quantize_act(float v, float inv_scale)
{
    float q = v * inv_scale;
    q = (q >= 0.f) ? floorf(q + 0.5f) : ceilf(q - 0.5f);
    if (q > 32767.f) q = 32767.f;
    if (q < -32768.f) q = -32768.f;
    return (int16_t)q;
}

/* Dense float→int16 quantize (compiler-friendly contiguous loop). */
static inline void fxp_quantize_act_n(const float* x, int16_t* q, int n, float inv_scale)
{
    for (int i = 0; i < n; i++)
        q[i] = fxp_quantize_act(x[i], inv_scale);
}

/* ── int32 activation quantization (for FxpConv act_bits=32 mode) ──
 *
 * int32 activations give 2^31 levels vs int16's 2^15, reducing the
 * quantization step by ~65536× — well below float32 epsilon (2^-24).
 * The result is near-lossless: the only remaining error is from int16
 * weight quantization (~1.5e-5 per weight, per-channel scaled).
 *
 * Stored as int32_t; the conv MAC loop casts to int64_t for accumulation,
 * so the same loop body handles both int16 and int32 activations.
 */
static inline int32_t fxp_quantize_act_i32(float v, float inv_scale)
{
    float q = v * inv_scale;
    q = (q >= 0.f) ? floorf(q + 0.5f) : ceilf(q - 0.5f);
    if (q > 2147483520.f) q = 2147483520.f; /* largest int32 exactly representable as float */
    if (q < -2147483648.f) q = -2147483648.f;
    return (int32_t)q;
}

static inline void fxp_quantize_act_i32_n(const float* x, int32_t* q, int n, float inv_scale)
{
    for (int i = 0; i < n; i++)
        q[i] = fxp_quantize_act_i32(x[i], inv_scale);
}

/* ── int8 activation quantization (for FxpConv act_bits=8 mode) ──
 *
 * 8-bit activations ([-128,127]) paired with int8 weights enable the
 * AVX-512 VNNI _mm512_dpbusd path: 128 int8 MAC/cycle vs int16's 32.
 * Accumulation stays in int32 (safe: max int32 accum for cin=4096 is
 * 66.6M, far below INT32_MAX), eliminating the int64 widen that throttled
 * the int16 kernel. */
static inline int8_t fxp_quantize_act_i8(float v, float inv_scale)
{
    float q = v * inv_scale;
    q = (q >= 0.f) ? floorf(q + 0.5f) : ceilf(q - 0.5f);
    if (q > 127.f) q = 127.f;
    if (q < -128.f) q = -128.f;
    return (int8_t)q;
}

static inline void fxp_quantize_act_i8_n(const float* x, int8_t* q, int n, float inv_scale)
{
    for (int i = 0; i < n; i++)
        q[i] = fxp_quantize_act_i8(x[i], inv_scale);
}

#endif
