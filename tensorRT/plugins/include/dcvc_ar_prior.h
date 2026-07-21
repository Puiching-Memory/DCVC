#ifndef DCVC_AR_PRIOR_H
#define DCVC_AR_PRIOR_H

#include "dcvc_plugins.h"
#include "dcvc_rt.h"
#include "dcvc_trt_runner.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DcvcArPriorCtx DcvcArPriorCtx;

DcvcArPriorCtx* dcvc_ar_prior_create(DcvcTrtRunner* runner, int passes);
void dcvc_ar_prior_destroy(DcvcArPriorCtx* ctx);
DcvcRtStatus dcvc_ar_prior_encode_pass(DcvcArPriorCtx* ctx, int pass_idx,
                                       DcvcTensorHalf* y, DcvcTensorHalf* scales,
                                       DcvcTensorHalf* means, DcvcTensorHalf* mask,
                                       DcvcTensorHalf* y_hat, int8_t* y_q,
                                       DcvcTensorHalf* common_params);

#ifdef __cplusplus
}
#endif

#endif
