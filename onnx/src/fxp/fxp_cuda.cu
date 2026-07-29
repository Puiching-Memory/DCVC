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

/* k_conv1x1 tile configuration (output-stationary tiled GEMM, int64 MAC).
 * OC_TILE output channels x S_TILE spatial positions per CTA; the cin
 * reduction is tiled in chunks of RED so shared-memory use is bounded
 * regardless of cin.  Integer accumulation is associative, so the MAC order
 * is bit-exact with the CPU scalar loop; the dequant below is unchanged
 * (no FMA). */
constexpr int C1_OC_TILE = 16;
constexpr int C1_S_TILE = 16;
constexpr int C1_RED = 32;

__global__ void k_conv1x1(const int16_t* xq, const int16_t* w_int,
                          const float* w_scale, const float* bias,
                          float* y, int cin, int cout, int hw, float x_scale)
{
    extern __shared__ int16_t smem[];
    int16_t* s_act = smem;                       /* [S_TILE][RED] */
    int16_t* s_wgt = smem + C1_S_TILE * C1_RED;  /* [OC_TILE][RED] */

    const int oc0 = blockIdx.x * C1_OC_TILE;
    const int s0 = blockIdx.y * C1_S_TILE;
    const int tid = threadIdx.x;
    const int oc_local = tid / C1_S_TILE;
    const int s_local = tid - oc_local * C1_S_TILE;
    const int oc = oc0 + oc_local;
    const int s = s0 + s_local;
    const bool valid = (oc < cout) && (s < hw);

    long long acc = 0;

    for (int r0 = 0; r0 < cin; r0 += C1_RED) {
        /* Cooperatively load the activation tile (S_TILE x RED) into smem.
         * Out-of-range spatial positions and the cin remainder are padded
         * with 0, which contributes nothing to the int64 sum. */
        for (int k = tid; k < C1_S_TILE * C1_RED; k += C1_OC_TILE * C1_S_TILE) {
            int sl = k / C1_RED;
            int ri = k - sl * C1_RED;
            int s_g = s0 + sl;
            int ic = r0 + ri;
            s_act[sl * C1_RED + ri] =
                ((unsigned)s_g < (unsigned)hw && ic < cin)
                    ? xq[(size_t)s_g * cin + ic] : (int16_t)0;
        }
        /* Cooperatively load the weight tile (OC_TILE x RED) into smem. */
        for (int k = tid; k < C1_OC_TILE * C1_RED; k += C1_OC_TILE * C1_S_TILE) {
            int ol = k / C1_RED;
            int ri = k - ol * C1_RED;
            int oc_g = oc0 + ol;
            int ic = r0 + ri;
            s_wgt[ol * C1_RED + ri] =
                ((unsigned)oc_g < (unsigned)cout && ic < cin)
                    ? w_int[(size_t)oc_g * cin + ic] : (int16_t)0;
        }
        __syncthreads();

        if (valid) {
            const int16_t* arow = s_act + s_local * C1_RED;
            const int16_t* wrow = s_wgt + oc_local * C1_RED;
            for (int ri = 0; ri < C1_RED; ri++)
                acc += (long long)arow[ri] * (long long)wrow[ri];
        }
        __syncthreads();
    }

    if (valid) {
        /* No FMA: must match CPU (float)acc * scale + bias. */
        float scale = __fmul_rn(x_scale, w_scale[oc]);
        float t = __fmul_rn((float)acc, scale);
        y[oc * hw + s] = __fadd_rn(t, bias[oc]);
    }
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

/* General grouped convolution. Each CTA computes an OC_TILE x S_TILE output
 * tile for one group. Activations and weights are staged in reduction tiles,
 * preserving the CPU int64 dot-product semantics while sharing each input
 * value across the output-channel tile. */
__global__ void k_conv_general(const int16_t* xq, const int16_t* w_int,
                               const float* w_scale, const float* bias,
                               float* y, int cin, int cout, int h, int w,
                               int oh, int ow, int kh, int kw,
                               int pad_t, int pad_l,
                               int stride_h, int stride_w, int group,
                               float x_scale)
{
    extern __shared__ int16_t smem[];
    int16_t* s_act = smem;                       /* [S_TILE][RED] */
    int16_t* s_wgt = smem + C1_S_TILE * C1_RED;  /* [OC_TILE][RED] */

    const int cin_g = cin / group;
    const int cout_g = cout / group;
    const int k_area = kh * kw;
    const int red_total = cin_g * k_area;
    const int g = blockIdx.z;
    const int oc0_g = blockIdx.x * C1_OC_TILE;
    const int s0 = blockIdx.y * C1_S_TILE;
    const int tid = threadIdx.x;
    const int oc_local = tid / C1_S_TILE;
    const int s_local = tid - oc_local * C1_S_TILE;
    const int oc_g = oc0_g + oc_local;
    const int oc = g * cout_g + oc_g;
    const int s = s0 + s_local;
    const int y_hw = oh * ow;
    const bool valid = (oc_g < cout_g) && (s < y_hw);

    long long acc = 0;
    for (int r0 = 0; r0 < red_total; r0 += C1_RED) {
        for (int k = tid; k < C1_S_TILE * C1_RED;
             k += C1_OC_TILE * C1_S_TILE) {
            const int sl = k / C1_RED;
            const int ri = k - sl * C1_RED;
            const int s_g = s0 + sl;
            const int red = r0 + ri;
            int16_t v = 0;
            if (s_g < y_hw && red < red_total) {
                const int ic_g = red / k_area;
                const int tap = red - ic_g * k_area;
                const int ky = tap / kw;
                const int kx = tap - ky * kw;
                const int oy = s_g / ow;
                const int ox = s_g - oy * ow;
                const int iy = oy * stride_h + ky - pad_t;
                const int ix = ox * stride_w + kx - pad_l;
                if ((unsigned)iy < (unsigned)h && (unsigned)ix < (unsigned)w) {
                    const int ic = g * cin_g + ic_g;
                    v = xq[((size_t)ic * h + iy) * w + ix];
                }
            }
            s_act[sl * C1_RED + ri] = v;
        }
        for (int k = tid; k < C1_OC_TILE * C1_RED;
             k += C1_OC_TILE * C1_S_TILE) {
            const int ol = k / C1_RED;
            const int ri = k - ol * C1_RED;
            const int oc_g_load = oc0_g + ol;
            const int red = r0 + ri;
            s_wgt[ol * C1_RED + ri] =
                (oc_g_load < cout_g && red < red_total)
                    ? w_int[(size_t)(g * cout_g + oc_g_load) * red_total + red]
                    : (int16_t)0;
        }
        __syncthreads();

        if (valid) {
            const int16_t* arow = s_act + s_local * C1_RED;
            const int16_t* wrow = s_wgt + oc_local * C1_RED;
            for (int ri = 0; ri < C1_RED; ri++)
                acc += (long long)arow[ri] * (long long)wrow[ri];
        }
        __syncthreads();
    }

    if (valid) {
        const float scale = __fmul_rn(x_scale, w_scale[oc]);
        const float t = __fmul_rn((float)acc, scale);
        y[(size_t)oc * y_hw + s] = __fadd_rn(t, bias[oc]);
    }
}

__global__ void k_wsrelu(const float* x, float* y, const float* lut,
                         int n_elem, float inv_x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elem) return;
    /* Match the CPU baseline (fxp_wsrelu_f32_range): linear interpolation
     * between adjacent LUT entries instead of a truncating direct lookup.
     * Plain float ops + floorf are IEEE round-to-nearest -> bit-exact. */
    float scaled = x[i] * inv_x;
    if (scaled >= 32767.0f) {
        y[i] = lut[65535];
    } else if (scaled <= -32768.0f) {
        y[i] = lut[0];
    } else {
        float fl = floorf(scaled);
        int idx = (int)fl + 32768;
        float frac = scaled - fl;
        y[i] = lut[idx] + (lut[idx + 1] - lut[idx]) * frac;
    }
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

        dim3 grid((cout + C1_OC_TILE - 1) / C1_OC_TILE,
                  (hw + C1_S_TILE - 1) / C1_S_TILE);
        const int block = C1_OC_TILE * C1_S_TILE;
        const size_t smem_bytes =
            (size_t)(C1_S_TILE + C1_OC_TILE) * C1_RED * sizeof(int16_t);
        k_conv1x1<<<grid, block, smem_bytes, st>>>(xq, w_int, w_scale, bias,
                                                   y_n, cin, cout, hw, x_scale);
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
    if (n < 1 || cin < 1 || cout < 1 || h < 1 || w < 1 ||
        kh < 1 || kw < 1 || stride_h < 1 || stride_w < 1 ||
        group < 1 || cin % group != 0 || cout % group != 0)
        return 2;
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

    const int oh = (h + pad_t + pad_b - kh) / stride_h + 1;
    const int ow = (w + pad_l + pad_r - kw) / stride_w + 1;
    if (oh < 1 || ow < 1)
        return 2;
    const int x_hw = h * w;
    const int y_hw = oh * ow;
    const size_t xq_bytes = (size_t)cin * (size_t)x_hw * sizeof(int16_t);
    if (ensure_scratch(xq_bytes))
        return 1;
    int16_t* xq = (int16_t*)g_scratch.ptr;
    cudaStream_t st = as_stream(cuda_stream);
    const float inv_x = 1.0f / x_scale;
    const int threads = C1_OC_TILE * C1_S_TILE;
    const size_t smem_bytes =
        (size_t)(C1_S_TILE + C1_OC_TILE) * C1_RED * sizeof(int16_t);

    for (int ni = 0; ni < n; ni++) {
        const float* x_n = x + (size_t)ni * cin * x_hw;
        float* y_n = y + (size_t)ni * cout * y_hw;
        const int n_elem = cin * x_hw;
        const int blocks_q = (n_elem + threads - 1) / threads;
        k_quant_nchw<<<blocks_q, threads, 0, st>>>(x_n, xq, n_elem, inv_x);
        if (launch_ok(cudaGetLastError(), "k_quant_nchw_general"))
            return 3;

        const int cout_g = cout / group;
        dim3 grid((cout_g + C1_OC_TILE - 1) / C1_OC_TILE,
                  (y_hw + C1_S_TILE - 1) / C1_S_TILE,
                  group);
        k_conv_general<<<grid, threads, smem_bytes, st>>>(
            xq, w_int, w_scale, bias, y_n, cin, cout, h, w, oh, ow, kh, kw,
            pad_t, pad_l, stride_h, stride_w, group, x_scale);
        if (launch_ok(cudaGetLastError(), "k_conv_general"))
            return 4;
    }
    return 0;
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
