/* DCVC-RK C runtime — shared types. */
#ifndef DCVC_RK_TYPES_H
#define DCVC_RK_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DcvcRkStatus {
    DCVC_RK_OK = 0,
    DCVC_RK_ERR_INVALID_ARG = 1,
    DCVC_RK_ERR_IO = 2,
    DCVC_RK_ERR_RKNN = 3,
    DCVC_RK_ERR_OOM = 4,
    DCVC_RK_ERR_UNSUPPORTED = 5,
    DCVC_RK_ERR_ENTROPY = 6
} DcvcRkStatus;

typedef struct DcvcRkTensorView {
    void* data;   /* FP32 NCHW */
    int n, c, h, w;
} DcvcRkTensorView;

const char* dcvc_rk_status_string(DcvcRkStatus st);

#ifdef __cplusplus
}
#endif

#endif
