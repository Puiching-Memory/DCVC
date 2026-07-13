#ifndef DCVC_DPB_H
#define DCVC_DPB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single-reference DPB (max_dpb_size = 1), feature + optional pixel buffer. */
typedef struct DcvcDpb {
    int pad_w;
    int pad_h;
    int feat_c;          /* g_ch_d = 256 */
    int feat_h;          /* pad_h / 8 */
    int feat_w;          /* pad_w / 8 */
    int has_ref;
    int frame_idx;
    int reset_interval;
    int use_feature_mode; /* 1: store feature; 0: pixel adaptor path */
    uint16_t* feature_fp16; /* feat_c * feat_h * feat_w */
    uint16_t* pixels_fp16;  /* 3 * pad_h * pad_w YCbCr */
} DcvcDpb;

int dcvc_dpb_init(DcvcDpb* dpb, int pad_w, int pad_h, int reset_interval);
void dcvc_dpb_free(DcvcDpb* dpb);
void dcvc_dpb_reset(DcvcDpb* dpb);

/* After I/P recon: store feature (and optionally pixels). */
void dcvc_dpb_update(DcvcDpb* dpb, const uint16_t* feature_fp16, const uint16_t* pixels_fp16);

/* Returns 1 if adaptor_i refresh should run this frame (reset_interval). */
int dcvc_dpb_need_feature_refresh(const DcvcDpb* dpb);

#ifdef __cplusplus
}
#endif

#endif
