#ifndef DCVC_FXP_SCALE_INDEX_H
#define DCVC_FXP_SCALE_INDEX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Integer scale → CDF index (shared enc/dec rule).
 *
 *   float scale
 *     → Q18 (half-away-from-zero)
 *     → Q12 (round-to-nearest, >>6 with +32)
 *     → clamp to [scale_min, scale_max] in Q12
 *     → largest i in [0,127] with scale_table_q12[i] <= s_q12
 *
 * scale_table_q12 is the log-spaced 128-level table used to build gaussian_cdf,
 * stored as Q12 (value = integer / 4096). This matches the semantics of
 *   floor( (log(s)-log(0.11)) / step )
 * on the continuous table, but without any floating log at runtime.
 */

enum {
    DCVC_SCALE_LEVELS = 128,
    DCVC_Q12_ONE = 4096,
    DCVC_Q18_ONE = 262144,
    DCVC_SCALE_MIN_Q12 = 451,   /* round(0.11 * 4096) */
    DCVC_SCALE_MAX_Q12 = 65536  /* 16.0 * 4096 */
};

/* Single-element helpers (also useful for golden tests). */
int32_t dcvc_float_to_q18(float x);
int32_t dcvc_q18_to_q12_round(int32_t q18);
uint8_t dcvc_scale_q12_to_index(int32_t s_q12);
uint8_t dcvc_scale_float_to_index(float scale);

/* Batch: float scales → CDF indexes. */
void dcvc_build_index_dec_i(const float* scales, uint8_t* out, int n);

/* Batch: pack (round(symbol)<<8 | index) like the legacy encoder. */
void dcvc_build_index_enc_i(const float* symbols, const float* scales,
                            int16_t* out, int n);

#ifdef __cplusplus
}
#endif

#endif
