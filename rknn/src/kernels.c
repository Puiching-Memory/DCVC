#include "dcvc_rk/kernels.h"

#include <math.h>
#include <string.h>

static const float Kr = 0.2126f;
static const float Kg = 0.7152f;
static const float Kb = 0.0722f;

void dcvc_rk_rgb_to_ycbcr(const float* rgb, float* ycbcr, int n)
{
    for (int i = 0; i < n; i++) {
        float r = rgb[i], g = rgb[i + n], b = rgb[i + 2 * n];
        float y = Kr * r + Kg * g + Kb * b;
        float cb = 0.5f * (b - y) / (1.0f - Kb) + 0.5f;
        float cr = 0.5f * (r - y) / (1.0f - Kr) + 0.5f;
        if (y < 0) y = 0; if (y > 1) y = 1;
        if (cb < 0) cb = 0; if (cb > 1) cb = 1;
        if (cr < 0) cr = 0; if (cr > 1) cr = 1;
        ycbcr[i] = y; ycbcr[i + n] = cb; ycbcr[i + 2 * n] = cr;
    }
}

void dcvc_rk_ycbcr_to_rgb(const float* ycbcr, float* rgb, int n)
{
    for (int i = 0; i < n; i++) {
        float y = ycbcr[i], cb = ycbcr[i + n], cr = ycbcr[i + 2 * n];
        float r = y + (2.0f - 2.0f * Kr) * (cr - 0.5f);
        float b = y + (2.0f - 2.0f * Kb) * (cb - 0.5f);
        float g = (y - Kr * r - Kb * b) / Kg;
        if (r < 0) r = 0; if (r > 1) r = 1;
        if (g < 0) g = 0; if (g > 1) g = 1;
        if (b < 0) b = 0; if (b > 1) b = 1;
        rgb[i] = r; rgb[i + n] = g; rgb[i + 2 * n] = b;
    }
}

void dcvc_rk_replicate_pad_3(const float* x, int H, int W, float* dst, int Hp, int Wp)
{
    for (int c = 0; c < 3; c++) {
        const float* xc = x + (size_t)c * H * W;
        float* dc = dst + (size_t)c * Hp * Wp;
        for (int i = 0; i < Hp; i++) {
            const float* srow = xc + (size_t)(i < H ? i : H - 1) * W;
            float* drow = dc + (size_t)i * Wp;
            for (int j = 0; j < Wp; j++)
                drow[j] = srow[j < W ? j : W - 1];
        }
    }
}

void dcvc_rk_crop_3(const float* src, int Hp, int Wp, float* dst, int H, int W)
{
    for (int c = 0; c < 3; c++) {
        const float* sc = src + (size_t)c * Hp * Wp;
        float* dc = dst + (size_t)c * H * W;
        for (int i = 0; i < H; i++)
            memcpy(dc + (size_t)i * W, sc + (size_t)i * Wp, (size_t)W * sizeof(float));
    }
}

void dcvc_rk_round_to_int8(const float* in, float* outf, int8_t* out_i8, int n)
{
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
    for (int i = 0; i < n; i++) out[i] = (float)in[i];
}

void dcvc_rk_pixel_unshuffle_8(const float* x, float* out, int H, int W)
{
    int fH = H / 8, fW = W / 8;
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
    for (int c = 0; c < C; c++)
        for (int i = 0; i < HW; i++)
            out[c * HW + i] = x[c * HW + i] * q[c];
}

void dcvc_rk_mul_q_pixel(const float* in, const float* q_hw, float* out, int nc, int hw)
{
    for (int c = 0; c < nc; c++)
        for (int i = 0; i < hw; i++)
            out[c * hw + i] = in[c * hw + i] * q_hw[i];
}

void dcvc_rk_separate_prior_intra(const float* pf, float* q_enc, float* q_dec,
                                  float* scales, float* means, int N, int hw)
{
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
    for (int i = 0; i < nc * hw; i++) {
        float v = params[i];
        qdec[i] = v < 0.5f ? 0.5f : v;
    }
    memcpy(scales, params + nc * hw, bw);
    memcpy(means, params + 2 * nc * hw, bw);
}

void dcvc_rk_sp4x(const float* x, float* out, int n)
{
    for (int i = 0; i < n; i++)
        out[i] = x[i] + x[i + n] + x[i + 2 * n] + x[i + 3 * n];
}

void dcvc_rk_sp2x(const float* x, float* out, int n)
{
    for (int i = 0; i < n; i++)
        out[i] = x[i] + x[i + n];
}

void dcvc_rk_process_mask_yq(const float* y, const float* scales, const float* means,
                             const float* mask, float* yq, int n, float skip_thres)
{
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
    for (int i = 0; i < n; i++)
        out[i] = (yq_r[i % cyhw] + means[i]) * mask[i];
}

void dcvc_rk_add_inplace(float* out, const float* step, int n)
{
    for (int i = 0; i < n; i++) out[i] += step[i];
}

void dcvc_rk_elem_mul(const float* a, const float* b, float* out, int n)
{
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}

static const int k_mask_pattern[4][4] = {
    {0, 1, 2, 3}, {3, 2, 1, 0}, {2, 3, 0, 1}, {1, 0, 3, 2},
};

void dcvc_rk_fill_masks_4x(float* masks[4], int nc, int H, int W)
{
    int hw = H * W, q = nc / 4;
    for (int m = 0; m < 4; m++) {
        for (int ch = 0; ch < nc; ch++) {
            int quarter = ch / q; if (quarter > 3) quarter = 3;
            int target = k_mask_pattern[m][quarter];
            for (int hh = 0; hh < H; hh++)
                for (int ww = 0; ww < W; ww++) {
                    int sp = (hh % 2) * 2 + (ww % 2);
                    masks[m][ch * hw + hh * W + ww] = (sp == target) ? 1.f : 0.f;
                }
        }
    }
}

void dcvc_rk_fill_masks_2x(float* masks[2], int nc, int H, int W)
{
    int hw = H * W, half = nc / 2;
    for (int m = 0; m < 2; m++) {
        for (int ch = 0; ch < nc; ch++) {
            int half_idx = ch / half;
            int use_m0 = (m == 0) ? (half_idx == 0) : (half_idx == 1);
            for (int hh = 0; hh < H; hh++)
                for (int ww = 0; ww < W; ww++) {
                    int is_diag = ((hh % 2) == (ww % 2));
                    int val = use_m0 ? is_diag : !is_diag;
                    masks[m][ch * hw + hh * W + ww] = val ? 1.f : 0.f;
                }
        }
    }
}
