#include "dcvc_rk/kernels.h"

#include <math.h>
#include <string.h>

#ifdef __aarch64__
#include <arm_neon.h>
#define DCVC_NEON 1
#else
#define DCVC_NEON 0
#endif

static const float Kr = 0.2126f;
static const float Kg = 0.7152f;
static const float Kb = 0.0722f;

static inline float clampf01(float v)
{
    if (v < 0.f) return 0.f;
    if (v > 1.f) return 1.f;
    return v;
}

void dcvc_rk_rgb_to_ycbcr(const float* rgb, float* ycbcr, int n)
{
#pragma omp parallel for schedule(static) if(n > 4096)
    for (int i = 0; i < n; i++) {
        float r = rgb[i], g = rgb[i + n], b = rgb[i + 2 * n];
        float y = Kr * r + Kg * g + Kb * b;
        float cb = 0.5f * (b - y) / (1.0f - Kb) + 0.5f;
        float cr = 0.5f * (r - y) / (1.0f - Kr) + 0.5f;
        ycbcr[i] = clampf01(y);
        ycbcr[i + n] = clampf01(cb);
        ycbcr[i + 2 * n] = clampf01(cr);
    }
}

void dcvc_rk_ycbcr_to_rgb(const float* ycbcr, float* rgb, int n)
{
    const float cr_c = 2.0f - 2.0f * Kr;
    const float cb_c = 2.0f - 2.0f * Kb;
#pragma omp parallel for schedule(static) if(n > 4096)
    for (int i = 0; i < n; i++) {
        float y = ycbcr[i], cb = ycbcr[i + n], cr = ycbcr[i + 2 * n];
        float r = y + cr_c * (cr - 0.5f);
        float b = y + cb_c * (cb - 0.5f);
        float g = (y - Kr * r - Kb * b) / Kg;
        rgb[i] = clampf01(r);
        rgb[i + n] = clampf01(g);
        rgb[i + 2 * n] = clampf01(b);
    }
}

void dcvc_rk_replicate_pad_3(const float* x, int H, int W, float* dst, int Hp, int Wp)
{
#pragma omp parallel for collapse(2) schedule(static) if(Hp * Wp > 4096)
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < Hp; i++) {
            const float* xc = x + (size_t)c * H * W;
            float* dc = dst + (size_t)c * Hp * Wp;
            const float* srow = xc + (size_t)(i < H ? i : H - 1) * W;
            float* drow = dc + (size_t)i * Wp;
            int j = 0;
#if DCVC_NEON
            for (; j + 4 <= W; j += 4)
                vst1q_f32(drow + j, vld1q_f32(srow + j));
#endif
            for (; j < W; j++) drow[j] = srow[j];
            float edge = srow[W - 1];
            for (; j < Wp; j++) drow[j] = edge;
        }
    }
}

void dcvc_rk_crop_3(const float* src, int Hp, int Wp, float* dst, int H, int W)
{
#pragma omp parallel for collapse(2) schedule(static) if(H * W > 4096)
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < H; i++) {
            const float* sc = src + (size_t)c * Hp * Wp;
            float* dc = dst + (size_t)c * H * W;
            memcpy(dc + (size_t)i * W, sc + (size_t)i * Wp, (size_t)W * sizeof(float));
        }
    }
}

void dcvc_rk_round_to_int8(const float* in, float* outf, int8_t* out_i8, int n)
{
#pragma omp parallel for schedule(static) if(n > 16384)
    for (int i = 0; i < n; i++) {
        float v = roundf(in[i]);
        if (v > 127.f) v = 127.f;
        if (v < -128.f) v = -128.f;
        outf[i] = v;
        out_i8[i] = (int8_t)v;
    }
}

void dcvc_rk_int8_to_float(const int8_t* in, float* out, int n)
{
    int i = 0;
#if DCVC_NEON
    for (; i + 16 <= n; i += 16) {
        int8x16_t v = vld1q_s8(in + i);
        int16x8_t lo = vmovl_s8(vget_low_s8(v));
        int16x8_t hi = vmovl_s8(vget_high_s8(v));
        vst1q_f32(out + i,      vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
        vst1q_f32(out + i + 4,  vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
        vst1q_f32(out + i + 8,  vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
        vst1q_f32(out + i + 12, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));
    }
#endif
    for (; i < n; i++) out[i] = (float)in[i];
}

void dcvc_rk_pixel_unshuffle_8(const float* x, float* out, int H, int W)
{
    int fH = H / 8, fW = W / 8;
#pragma omp parallel for collapse(3) schedule(static)
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++) {
                int oc = c * 64 + i * 8 + j;
                for (int fh = 0; fh < fH; fh++)
                    for (int fw = 0; fw < fW; fw++)
                        out[(size_t)oc * fH * fW + fh * fW + fw] =
                            x[(size_t)c * H * W + (fh * 8 + i) * W + (fw * 8 + j)];
            }
}

void dcvc_rk_pixel_shuffle_8(const float* in, float* out, int H, int W)
{
    int fH = H / 8, fW = W / 8;
#pragma omp parallel for collapse(3) schedule(static)
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++) {
                int ic = c * 64 + i * 8 + j;
                for (int fh = 0; fh < fH; fh++)
                    for (int fw = 0; fw < fW; fw++)
                        out[(size_t)c * H * W + (fh * 8 + i) * W + (fw * 8 + j)] =
                            in[(size_t)ic * fH * fW + fh * fW + fw];
            }
}

void dcvc_rk_mul_q_broadcast(const float* x, const float* q, float* out, int C, int HW)
{
#pragma omp parallel for schedule(static) if((size_t)C * HW > 16384)
    for (int c = 0; c < C; c++) {
        float qc = q[c];
        const float* xc = x + (size_t)c * HW;
        float* oc = out + (size_t)c * HW;
        int i = 0;
#if DCVC_NEON
        float32x4_t vq = vdupq_n_f32(qc);
        for (; i + 4 <= HW; i += 4)
            vst1q_f32(oc + i, vmulq_f32(vld1q_f32(xc + i), vq));
#endif
        for (; i < HW; i++) oc[i] = xc[i] * qc;
    }
}

void dcvc_rk_mul_q_pixel(const float* in, const float* q_hw, float* out, int nc, int hw)
{
#pragma omp parallel for schedule(static) if((size_t)nc * hw > 16384)
    for (int c = 0; c < nc; c++) {
        const float* ic = in + (size_t)c * hw;
        float* oc = out + (size_t)c * hw;
        int i = 0;
#if DCVC_NEON
        for (; i + 4 <= hw; i += 4)
            vst1q_f32(oc + i, vmulq_f32(vld1q_f32(ic + i), vld1q_f32(q_hw + i)));
#endif
        for (; i < hw; i++) oc[i] = ic[i] * q_hw[i];
    }
}

void dcvc_rk_separate_prior_intra(const float* pf, float* q_enc, float* q_dec,
                                  float* scales, float* means, int N, int hw)
{
#pragma omp parallel for schedule(static) if(hw > 4096)
    for (int i = 0; i < hw; i++) {
        float v0 = pf[0 * hw + i];
        float v1 = pf[1 * hw + i];
        q_enc[i] = 1.0f / (1.0f + expf(-v0)) * 1.5f + 0.5f;
        q_dec[i] = 1.0f / (1.0f + expf(-v1)) * 1.5f + 0.5f;
    }
    memcpy(scales, pf + 2 * hw, (size_t)N * hw * sizeof(float));
    memcpy(means, pf + (2 + N) * hw, (size_t)N * hw * sizeof(float));
}

void dcvc_rk_separate_prior_video_enc(const float* params, float* qdec,
                                      float* scales, float* means, int nc, int hw)
{
    size_t bw = (size_t)nc * hw * sizeof(float);
    memcpy(qdec, params, bw);
    memcpy(scales, params + nc * hw, bw);
    memcpy(means, params + 2 * nc * hw, bw);
}

void dcvc_rk_separate_prior_video_dec(const float* params, float* qdec,
                                      float* scales, float* means, int nc, int hw)
{
    size_t bw = (size_t)nc * hw * sizeof(float);
    int n = nc * hw;
#pragma omp parallel for schedule(static) if(n > 16384)
    for (int i = 0; i < n; i++) {
        float v = params[i];
        qdec[i] = v < 0.5f ? 0.5f : v;
    }
    memcpy(scales, params + nc * hw, bw);
    memcpy(means, params + 2 * nc * hw, bw);
}

void dcvc_rk_sp4x(const float* x, float* out, int n)
{
    int i = 0;
#if DCVC_NEON
    for (; i + 4 <= n; i += 4) {
        float32x4_t s = vld1q_f32(x + i);
        s = vaddq_f32(s, vld1q_f32(x + i + n));
        s = vaddq_f32(s, vld1q_f32(x + i + 2 * n));
        s = vaddq_f32(s, vld1q_f32(x + i + 3 * n));
        vst1q_f32(out + i, s);
    }
#endif
    for (; i < n; i++)
        out[i] = x[i] + x[i + n] + x[i + 2 * n] + x[i + 3 * n];
}

void dcvc_rk_sp2x(const float* x, float* out, int n)
{
    int i = 0;
#if DCVC_NEON
    for (; i + 4 <= n; i += 4)
        vst1q_f32(out + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(x + i + n)));
#endif
    for (; i < n; i++)
        out[i] = x[i] + x[i + n];
}

void dcvc_rk_process_mask_yq(const float* y, const float* scales, const float* means,
                             const float* mask, float* yq, int n, float skip_thres)
{
#pragma omp parallel for schedule(static) if(n > 16384)
    for (int i = 0; i < n; i++) {
        float fm = mask[i];
        float means_hat = means[i] * fm;
        float q = roundf((y[i] - means_hat) * fm);
        if (skip_thres > 0.f && scales[i] * fm <= skip_thres) q = 0.f;
        if (q > 127.f) q = 127.f;
        if (q < -128.f) q = -128.f;
        yq[i] = q;
    }
}

void dcvc_rk_restore_y_nx(const float* yq_r, const float* means, const float* mask,
                          float* out, int cyhw, int n)
{
#pragma omp parallel for schedule(static) if(n > 16384)
    for (int i = 0; i < n; i++)
        out[i] = (yq_r[i % cyhw] + means[i]) * mask[i];
}

void dcvc_rk_add_inplace(float* out, const float* step, int n)
{
    int i = 0;
#if DCVC_NEON
    for (; i + 4 <= n; i += 4)
        vst1q_f32(out + i, vaddq_f32(vld1q_f32(out + i), vld1q_f32(step + i)));
#endif
    for (; i < n; i++) out[i] += step[i];
}

void dcvc_rk_elem_mul(const float* a, const float* b, float* out, int n)
{
    int i = 0;
#if DCVC_NEON
    for (; i + 4 <= n; i += 4)
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
#endif
    for (; i < n; i++) out[i] = a[i] * b[i];
}

static const int k_mask_pattern[4][4] = {
    {0, 1, 2, 3}, {3, 2, 1, 0}, {2, 3, 0, 1}, {1, 0, 3, 2},
};

void dcvc_rk_fill_masks_4x(float* masks[4], int nc, int H, int W)
{
    int hw = H * W, q = nc / 4;
#pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < 4; m++) {
        for (int ch = 0; ch < nc; ch++) {
            int quarter = ch / q; if (quarter > 3) quarter = 3;
            int target = k_mask_pattern[m][quarter];
            float* row = masks[m] + ch * hw;
            for (int hh = 0; hh < H; hh++)
                for (int ww = 0; ww < W; ww++) {
                    int sp = (hh % 2) * 2 + (ww % 2);
                    row[hh * W + ww] = (sp == target) ? 1.f : 0.f;
                }
        }
    }
}

void dcvc_rk_fill_masks_2x(float* masks[2], int nc, int H, int W)
{
    int hw = H * W, half = nc / 2;
#pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < 2; m++) {
        for (int ch = 0; ch < nc; ch++) {
            int half_idx = ch / half;
            int use_m0 = (m == 0) ? (half_idx == 0) : (half_idx == 1);
            float* row = masks[m] + ch * hw;
            for (int hh = 0; hh < H; hh++)
                for (int ww = 0; ww < W; ww++) {
                    int is_diag = ((hh % 2) == (ww % 2));
                    int val = use_m0 ? is_diag : !is_diag;
                    row[hh * W + ww] = val ? 1.f : 0.f;
                }
        }
    }
}

#include "dcvc_rk/quant.h"
#include <math.h>

void dcvc_rk_quant_nchw_f32_to_i8(const float* src, int8_t* dst,
                                  int N, int C, int H, int W,
                                  float scale, int32_t zp)
{
    float inv = (scale > 1e-12f) ? (1.0f / scale) : 0.f;
    int n = N * C * H * W;
#pragma omp parallel for schedule(static) if(n > 16384)
    for (int i = 0; i < n; i++) {
        int q = (int)lrintf(src[i] * inv) + (int)zp;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        dst[i] = (int8_t)q;
    }
}

void dcvc_rk_dequant_nchw_i8_to_f32(const int8_t* src, float* dst,
                                    int N, int C, int H, int W, int w_stride,
                                    float scale, int32_t zp)
{
    if (w_stride <= 0) w_stride = W;
    if (w_stride == W) {
        int n = N * C * H * W;
#pragma omp parallel for schedule(static) if(n > 16384)
        for (int i = 0; i < n; i++)
            dst[i] = ((float)src[i] - (float)zp) * scale;
        return;
    }
#pragma omp parallel for collapse(3) schedule(static) if((size_t)N * C * H * W > 16384)
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h++) {
                const int8_t* row = src + ((((size_t)n * C + c) * H + h) * (size_t)w_stride);
                float* drow = dst + ((((size_t)n * C + c) * H + h) * (size_t)W);
                for (int w = 0; w < W; w++)
                    drow[w] = ((float)row[w] - (float)zp) * scale;
            }
}

void dcvc_rk_requant_nchw_i8(const int8_t* src, int N, int C, int H, int W, int w_stride,
                             float scale_src, int32_t zp_src,
                             int8_t* dst, float scale_dst, int32_t zp_dst)
{
    if (w_stride <= 0) w_stride = W;
    size_t elems = (size_t)N * C * H * W;
    int same_q = (zp_src == zp_dst) &&
                 (fabsf(scale_src - scale_dst) <= 1e-6f * fmaxf(1.f, fabsf(scale_dst)));
    if (same_q && w_stride == W) {
        if (src != dst) memcpy(dst, src, elems);
        return;
    }
    if (same_q) {
#pragma omp parallel for collapse(3) schedule(static) if(elems > 16384)
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H; h++) {
                    const int8_t* row = src + ((((size_t)n * C + c) * H + h) * (size_t)w_stride);
                    int8_t* drow = dst + ((((size_t)n * C + c) * H + h) * (size_t)W);
                    memcpy(drow, row, (size_t)W);
                }
        return;
    }
    float mul = (scale_dst > 1e-12f) ? (scale_src / scale_dst) : 0.f;
#pragma omp parallel for collapse(3) schedule(static) if(elems > 16384)
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h++) {
                const int8_t* row = src + ((((size_t)n * C + c) * H + h) * (size_t)w_stride);
                int8_t* drow = dst + ((((size_t)n * C + c) * H + h) * (size_t)W);
                for (int w = 0; w < W; w++) {
                    float x = ((float)row[w] - (float)zp_src) * mul;
                    int q = (int)lrintf(x) + (int)zp_dst;
                    if (q > 127) q = 127;
                    if (q < -128) q = -128;
                    drow[w] = (int8_t)q;
                }
            }
}
