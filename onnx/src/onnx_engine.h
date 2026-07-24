#ifndef DCVC_CPU_ONNX_ENGINE_H
#define DCVC_CPU_ONNX_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DcvcCpuStatus {
    DCVC_CPU_OK = 0,
    DCVC_CPU_ERR_INVALID_ARG = 1,
    DCVC_CPU_ERR_IO = 2,
    DCVC_CPU_ERR_ONNX = 3,
    DCVC_CPU_ERR_OOM = 4,
    DCVC_CPU_ERR_UNSUPPORTED = 5,
    DCVC_CPU_ERR_ENTROPY = 6   /* rANS stream corrupt/desynced */
} DcvcCpuStatus;

/* Generic 4D tensor view. For CPU backend both pointers are host memory. */
typedef struct DcvcCpuTensorView {
    void* data;       /* FP16 or FP32 */
    int is_fp16;      /* 1 = FP16, 0 = FP32 */
    int n, c, h, w;
} DcvcCpuTensorView;

typedef struct DcvcCpuEngine DcvcCpuEngine;

/* use_gpu selects the ONNX Runtime execution provider:
 *   0 = CPU (default; the DCVC_USE_GPU environment variable may override),
 *   1 = CUDA EP, 2 = TensorRT EP.
 * A GPU EP requires an ORT build that ships it (configure with -DDCVC_ORT_GPU=ON);
 * if the requested EP is unavailable the engine falls back to CPU with a warning.
 * DCVC_GPU_DEVICE selects the GPU device index (default 0). */
DcvcCpuEngine* dcvc_cpu_engine_create(const char* onnx_path, int use_gpu, DcvcCpuStatus* out_st);
void dcvc_cpu_engine_destroy(DcvcCpuEngine* eng);

/* Resolve DCVC_USE_GPU env (0/1/2). Used by pipelines so one env var flips all nets. */
int dcvc_cpu_default_use_gpu(void);

/* 1 if CUDA EP can be appended in this ORT build + runtime (cuDNN present). */
int dcvc_cpu_cuda_ep_usable(void);

/* Execute one inference with NCHW inputs/outputs.
 * Inputs and outputs are copied from/into the provided views.
 * Output views must have data allocated by caller. */
DcvcCpuStatus dcvc_cpu_engine_run(DcvcCpuEngine* eng,
                                  DcvcCpuTensorView* inputs, int n_in,
                                  DcvcCpuTensorView* outputs, int n_out);

const char* dcvc_cpu_status_string(DcvcCpuStatus st);

#ifdef __cplusplus
}
#endif

#endif
