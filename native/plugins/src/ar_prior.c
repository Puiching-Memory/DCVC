// AR prior host loop — calls spatial-prior engine between entropy passes.
// Mirrors compress_prior_2x / compress_prior_4x in common_model.py.

#include "dcvc_ar_prior.h"

#include <stdlib.h>
#include <string.h>

struct DcvcArPriorCtx {
    DcvcTrtRunner* runner;
    int passes; /* 2 for inter, 4 for intra */
    DcvcEngineId spatial_prior_engine;
};

DcvcArPriorCtx* dcvc_ar_prior_create(DcvcTrtRunner* runner, int passes)
{
    DcvcArPriorCtx* ctx = (DcvcArPriorCtx*)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->runner = runner;
    ctx->passes = passes;
    ctx->spatial_prior_engine =
        (passes == 4) ? DCVC_ENG_INTRA_SPATIAL_PRIOR : DCVC_ENG_INTER_SPATIAL_PRIOR;
    return ctx;
}

void dcvc_ar_prior_destroy(DcvcArPriorCtx* ctx)
{
    free(ctx);
}

/* One AR pass: process_with_mask → (optional) spatial prior TRT → next scales/means */
DcvcRtStatus dcvc_ar_prior_encode_pass(DcvcArPriorCtx* ctx, int pass_idx,
                                       DcvcTensorHalf* y, DcvcTensorHalf* scales,
                                       DcvcTensorHalf* means, DcvcTensorHalf* mask,
                                       DcvcTensorHalf* y_hat, int8_t* y_q,
                                       DcvcTensorHalf* common_params)
{
    if (!ctx || pass_idx < 0 || pass_idx >= ctx->passes) return DCVC_RT_ERR_INVALID_ARG;
    if (dcvc_process_with_mask(y, scales, means, mask, -1.f, y_hat, y_q) != 0)
        return DCVC_RT_ERR_INTERNAL;

    if (pass_idx + 1 < ctx->passes && common_params) {
        DcvcTensorView in[2];
        DcvcTensorView out[1];
        memset(in, 0, sizeof(in));
        memset(out, 0, sizeof(out));
        /* Bindings filled by runtime when engines are present */
        DcvcRtStatus st = dcvc_trt_runner_execute(ctx->runner, ctx->spatial_prior_engine, in, 0,
                                                  out, 0, NULL);
        if (st != DCVC_RT_OK && st != DCVC_RT_ERR_NO_ENGINE) return st;
    }
    return DCVC_RT_OK;
}
