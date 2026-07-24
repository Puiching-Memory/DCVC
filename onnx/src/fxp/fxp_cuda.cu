#include "fxp_cuda.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

namespace {

__device__ __forceinline__ int16_t fxp_quant_dev(float v, float inv_scale)
{
    float q = v * inv_scale;
    q = (q >= 0.f) ? floorf(q + 0.5f) : ceilf(q - 0.5f);
    if (q > 32767.f) q = 32767.f;
    if (q < -32768.f) q = -32768.f;
    return (int16_t)q;
}

/* Pack NCHW float → [hw][cin] int16. */
__global__ void k_quant_pack_nhwc(const float* x, int16_t* xq,
                                  int cin, int hw, float inv_x)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = cin * hw;
    if (idx >= total) return;
    int ic = idx / hw;
    int s = idx - ic * hw;
    xq[s * cin + ic] = fxp_quant_dev(x[ic * hw + s], inv_x);
}

/* Quantize NCHW plane in place layout [c][hw]. */
__global__ void k_quant_nchw(const float* x, int16_t* xq, int n_elem, float inv_x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elem)
        xq[i] = fxp_quant_dev(x[i], inv_x);
}

__global__ void k_conv1x1(const int16_t* xq, const int16_t* w_int,
                          const float* w_scale, const float* bias,
                          float* y, int cin, int cout, int hw, float x_scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = cout * hw;
    if (idx >= total) return;
    int oc = idx / hw;
    int s = idx - oc * hw;
    const int16_t* xs = xq + (size_t)s * cin;
    const int16_t* wk = w_int + (size_t)oc * cin;
    long long acc = 0;
    for (int ic = 0; ic < cin; ic++)
        acc += (long long)xs[ic] * (long long)wk[ic];
    /* No FMA: must match CPU (float)acc * scale + bias. */
    float scale = __fmul_rn(x_scale, w_scale[oc]);
    float t = __fmul_rn((float)acc, scale);
    y[oc * hw + s] = __fadd_rn(t, bias[oc]);
}

__global__ void k_dw3x3(const int16_t* xq, const int16_t* w_int,
                        const float* w_scale, const float* bias,
                        float* y, int c, int h, int w, float x_scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hw = h * w;
    int total = c * hw;
    if (idx >= total) return;
    int oc = idx / hw;
    int spat = idx - oc * hw;
    int yh = spat / w;
    int xw = spat - yh * w;
    const int16_t* qi = xq + (size_t)oc * hw;
    const int16_t* wk = w_int + (size_t)oc * 9;
    long long acc = 0;
    for (int kh = 0; kh < 3; kh++) {
        int ih = yh + kh - 1;
        for (int kw = 0; kw < 3; kw++) {
            int iw = xw + kw - 1;
            int16_t v = 0;
            if ((unsigned)ih < (unsigned)h && (unsigned)iw < (unsigned)w)
                v = qi[ih * w + iw];
            acc += (long long)v * (long long)wk[kh * 3 + kw];
        }
    }
    float scale = __fmul_rn(x_scale, w_scale[oc]);
    float t = __fmul_rn((float)acc, scale);
    y[oc * hw + spat] = __fadd_rn(t, bias[oc]);
}

__global__ void k_wsrelu(const float* x, float* y, const float* lut,
                         int n_elem, float inv_x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elem) return;
    int16_t q = fxp_quant_dev(x[i], inv_x);
    y[i] = lut[(int)q + 32768];
}

struct DevScratch {
    void* ptr = nullptr;
    size_t bytes = 0;
};

/* Per-host-thread growable device scratch (simple; custom-op path is single-stream). */
static thread_local DevScratch g_scratch;

static int ensure_scratch(size_t bytes)
{
    if (bytes <= g_scratch.bytes)
        return 0;
    if (g_scratch.ptr)
        cudaFree(g_scratch.ptr);
    g_scratch.ptr = nullptr;
    g_scratch.bytes = 0;
    cudaError_t e = cudaMalloc(&g_scratch.ptr, bytes);
    if (e != cudaSuccess) {
        fprintf(stderr, "fxp_cuda: cudaMalloc(%zu) failed: %s\n",
                bytes, cudaGetErrorString(e));
        return 1;
    }
    g_scratch.bytes = bytes;
    return 0;
}

static inline cudaStream_t as_stream(void* s)
{
    return static_cast<cudaStream_t>(s);
}

static inline int launch_ok(cudaError_t e, const char* what)
{
    if (e == cudaSuccess)
        return 0;
    fprintf(stderr, "fxp_cuda: %s: %s\n", what, cudaGetErrorString(e));
    return 1;
}

}  // namespace

extern "C" int fxp_conv1x1_f32_cuda(const float* x, float* y,
                                    int n, int cin, int cout, int h, int w,
                                    const int16_t* w_int, const float* w_scale,
                                    const float* bias, float x_scale,
                                    void* cuda_stream)
{
    if (n < 1 || cin < 1 || cout < 1 || h < 1 || w < 1)
        return 2;
    const int hw = h * w;
    const size_t xq_bytes = (size_t)cin * (size_t)hw * sizeof(int16_t);
    if (ensure_scratch(xq_bytes))
        return 1;
    int16_t* xq = (int16_t*)g_scratch.ptr;
    cudaStream_t st = as_stream(cuda_stream);
    const float inv_x = 1.0f / x_scale;
    const int threads = 256;

    for (int ni = 0; ni < n; ni++) {
        const float* x_n = x + (size_t)ni * cin * hw;
        float* y_n = y + (size_t)ni * cout * hw;

        int total_q = cin * hw;
        int blocks_q = (total_q + threads - 1) / threads;
        k_quant_pack_nhwc<<<blocks_q, threads, 0, st>>>(x_n, xq, cin, hw, inv_x);
        if (launch_ok(cudaGetLastError(), "k_quant_pack_nhwc"))
            return 3;

        int total = cout * hw;
        int blocks = (total + threads - 1) / threads;
        k_conv1x1<<<blocks, threads, 0, st>>>(xq, w_int, w_scale, bias, y_n,
                                              cin, cout, hw, x_scale);
        if (launch_ok(cudaGetLastError(), "k_conv1x1"))
            return 4;
    }
    return 0;
}

extern "C" int fxp_dwconv3x3_f32_cuda(const float* x, float* y,
                                      int n, int c, int h, int w,
                                      const int16_t* w_int, const float* w_scale,
                                      const float* bias, float x_scale,
                                      void* cuda_stream)
{
    if (n < 1 || c < 1 || h < 1 || w < 1)
        return 2;
    const int hw = h * w;
    const size_t xq_bytes = (size_t)c * (size_t)hw * sizeof(int16_t);
    if (ensure_scratch(xq_bytes))
        return 1;
    int16_t* xq = (int16_t*)g_scratch.ptr;
    cudaStream_t st = as_stream(cuda_stream);
    const float inv_x = 1.0f / x_scale;
    const int threads = 256;

    for (int ni = 0; ni < n; ni++) {
        const float* x_n = x + (size_t)ni * c * hw;
        float* y_n = y + (size_t)ni * c * hw;
        int n_elem = c * hw;
        int blocks_q = (n_elem + threads - 1) / threads;
        k_quant_nchw<<<blocks_q, threads, 0, st>>>(x_n, xq, n_elem, inv_x);
        if (launch_ok(cudaGetLastError(), "k_quant_nchw"))
            return 3;

        int blocks = (n_elem + threads - 1) / threads;
        k_dw3x3<<<blocks, threads, 0, st>>>(xq, w_int, w_scale, bias, y_n,
                                            c, h, w, x_scale);
        if (launch_ok(cudaGetLastError(), "k_dw3x3"))
            return 4;
    }
    return 0;
}

extern "C" int fxp_conv_f32_cuda(const float* x, float* y,
                                 int n, int cin, int cout, int h, int w,
                                 const int16_t* w_int, const float* w_scale, const float* bias,
                                 float x_scale,
                                 int kh, int kw,
                                 int pad_t, int pad_l, int pad_b, int pad_r,
                                 int stride_h, int stride_w, int group,
                                 void* cuda_stream)
{
    if (kh == 1 && kw == 1 && group == 1
        && pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0
        && stride_h == 1 && stride_w == 1)
        return fxp_conv1x1_f32_cuda(x, y, n, cin, cout, h, w,
                                    w_int, w_scale, bias, x_scale, cuda_stream);
    if (kh == 3 && kw == 3 && group == cin && cin == cout
        && pad_t == 1 && pad_l == 1 && pad_b == 1 && pad_r == 1
        && stride_h == 1 && stride_w == 1)
        return fxp_dwconv3x3_f32_cuda(x, y, n, cin, h, w,
                                      w_int, w_scale, bias, x_scale, cuda_stream);
    return -1;
}

extern "C" int fxp_wsrelu_f32_cuda(const float* x, float* y, int n_elem,
                                   const float* lut65536, float x_scale,
                                   void* cuda_stream)
{
    if (n_elem < 1)
        return 2;
    cudaStream_t st = as_stream(cuda_stream);
    const float inv_x = 1.0f / x_scale;
    const int threads = 256;
    int blocks = (n_elem + threads - 1) / threads;
    k_wsrelu<<<blocks, threads, 0, st>>>(x, y, lut65536, n_elem, inv_x);
    return launch_ok(cudaGetLastError(), "k_wsrelu");
}
