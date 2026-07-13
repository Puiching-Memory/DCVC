#include "dcvc_dpb.h"

#include <stdlib.h>
#include <string.h>

#include "dcvc_rt_internal.h"

int dcvc_dpb_init(DcvcDpb* dpb, int pad_w, int pad_h, int reset_interval)
{
    memset(dpb, 0, sizeof(*dpb));
    dpb->pad_w = pad_w;
    dpb->pad_h = pad_h;
    dpb->feat_c = DCVC_RT_CH_D;
    dpb->feat_h = pad_h / 8;
    dpb->feat_w = pad_w / 8;
    dpb->reset_interval = reset_interval > 0 ? reset_interval : 64;
    dpb->use_feature_mode = 1;
    size_t feat_n = (size_t)dpb->feat_c * dpb->feat_h * dpb->feat_w;
    size_t pix_n = (size_t)3 * pad_h * pad_w;
    dpb->feature_fp16 = (uint16_t*)calloc(feat_n, sizeof(uint16_t));
    dpb->pixels_fp16 = (uint16_t*)calloc(pix_n, sizeof(uint16_t));
    if (!dpb->feature_fp16 || !dpb->pixels_fp16) {
        dcvc_dpb_free(dpb);
        return -1;
    }
    return 0;
}

void dcvc_dpb_free(DcvcDpb* dpb)
{
    free(dpb->feature_fp16);
    free(dpb->pixels_fp16);
    memset(dpb, 0, sizeof(*dpb));
}

void dcvc_dpb_reset(DcvcDpb* dpb)
{
    dpb->has_ref = 0;
    dpb->frame_idx = 0;
}

void dcvc_dpb_update(DcvcDpb* dpb, const uint16_t* feature_fp16, const uint16_t* pixels_fp16)
{
    size_t feat_n = (size_t)dpb->feat_c * dpb->feat_h * dpb->feat_w;
    size_t pix_n = (size_t)3 * dpb->pad_h * dpb->pad_w;
    if (feature_fp16) memcpy(dpb->feature_fp16, feature_fp16, feat_n * sizeof(uint16_t));
    if (pixels_fp16) memcpy(dpb->pixels_fp16, pixels_fp16, pix_n * sizeof(uint16_t));
    dpb->has_ref = 1;
    dpb->frame_idx++;
}

int dcvc_dpb_need_feature_refresh(const DcvcDpb* dpb)
{
    if (!dpb->has_ref || dpb->reset_interval <= 0) return 0;
    return (dpb->frame_idx % dpb->reset_interval) == 0;
}
