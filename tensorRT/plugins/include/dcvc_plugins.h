#ifndef DCVC_PLUGINS_H
#define DCVC_PLUGINS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ATen-free fused op host API (CUDA if available, else CPU reference).
 * Tensor layout: NCHW contiguous, FP16 as uint16_t bits, batch N==1.
 */

typedef struct DcvcTensorHalf {
    uint16_t* data; /* host or device depending on on_device */
    int n, c, h, w;
    int on_device; /* 1 = CUDA device pointer */
} DcvcTensorHalf;

int dcvc_plugin_cuda_available(void);

/* process_with_mask: scales/means/mask FP16; outputs y_q int8 on host */
int dcvc_process_with_mask(const DcvcTensorHalf* y, const DcvcTensorHalf* scales,
                           const DcvcTensorHalf* means, const DcvcTensorHalf* mask,
                           float force_zero_thres, DcvcTensorHalf* y_hat, int8_t* y_q_out);

int dcvc_add_and_multiply(DcvcTensorHalf* y0, const DcvcTensorHalf* y1, const DcvcTensorHalf* q);

int dcvc_round_to_int8(DcvcTensorHalf* z, int8_t* z_int8_out);

int dcvc_replicate_pad(const DcvcTensorHalf* in, int pad_b, int pad_r, DcvcTensorHalf* out);

/* DepthConv / Subpel: weight blobs are opaque host buffers produced by convert_weights.py */
typedef struct DcvcDepthConvHandle DcvcDepthConvHandle;
DcvcDepthConvHandle* dcvc_depthconv_create(const void* weight_blob, size_t blob_bytes);
void dcvc_depthconv_destroy(DcvcDepthConvHandle* h);
int dcvc_depthconv_forward(DcvcDepthConvHandle* h, const DcvcTensorHalf* in, DcvcTensorHalf* out);

#ifdef DCVC_RT_HAS_TENSORRT
/* Register all DCVC plugins with a TensorRT builder/plugin registry. */
int dcvc_register_trt_plugins(void* trt_plugin_registry);
#endif

#ifdef __cplusplus
}
#endif

#endif
