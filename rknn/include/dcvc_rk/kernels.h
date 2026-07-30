#ifndef DCVC_RK_KERNELS_H
#define DCVC_RK_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void dcvc_rk_rgb_to_ycbcr(const float* rgb, float* ycbcr, int n);
void dcvc_rk_ycbcr_to_rgb(const float* ycbcr, float* rgb, int n);
void dcvc_rk_replicate_pad_3(const float* x, int H, int W, float* dst, int Hp, int Wp);
void dcvc_rk_crop_3(const float* src, int Hp, int Wp, float* dst, int H, int W);

void dcvc_rk_round_to_int8(const float* in, float* outf, int8_t* out_i8, int n);
void dcvc_rk_int8_to_float(const int8_t* in, float* out, int n);

/* x[3,H,W] -> out[192,H/8,W/8] */
void dcvc_rk_pixel_unshuffle_8(const float* x, float* out, int H, int W);
/* in[192,fH,fW] -> out[3,H,W] */
void dcvc_rk_pixel_shuffle_8(const float* in, float* out, int H, int W);

void dcvc_rk_mul_q_broadcast(const float* x, const float* q, float* out, int C, int HW);
void dcvc_rk_mul_q_pixel(const float* in, const float* q_hw, float* out, int nc, int hw);

void dcvc_rk_separate_prior_intra(const float* pf, float* q_enc, float* q_dec,
                                  float* scales, float* means, int N, int hw);
void dcvc_rk_separate_prior_video_enc(const float* params, float* qdec,
                                      float* scales, float* means, int nc, int hw);
void dcvc_rk_separate_prior_video_dec(const float* params, float* qdec,
                                      float* scales, float* means, int nc, int hw);

void dcvc_rk_sp4x(const float* x, float* out, int n);
void dcvc_rk_sp2x(const float* x, float* out, int n);
void dcvc_rk_process_mask_yq(const float* y, const float* scales, const float* means,
                             const float* mask, float* yq, int n, float skip_thres);
void dcvc_rk_restore_y_nx(const float* yq_r, const float* means, const float* mask,
                          float* out, int cyhw, int n);
void dcvc_rk_add_inplace(float* out, const float* step, int n);
void dcvc_rk_elem_mul(const float* a, const float* b, float* out, int n);

void dcvc_rk_fill_masks_4x(float* masks[4], int nc, int H, int W);
void dcvc_rk_fill_masks_2x(float* masks[2], int nc, int H, int W);

#ifdef __cplusplus
}
#endif

#endif
