// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
//
// ATen-free CUDA kernels for DCVC-RT host-side AR (autoregressive) passes.
// All functions take raw __half* device pointers (NCHW contiguous, FP16).
// Called directly by the C runtime between TensorRT engine invocations —
// NOT inside any engine graph, so no IPluginV3 wrapper is needed.
//
// Ported from src/layers/extensions/inference/kernel.cu (ATen → raw pointers).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// launch helpers
// ---------------------------------------------------------------------------
static inline void launch_config(int N, int& grid, int& block) {
    block = 256;
    grid = (N + block - 1) / block;
}

// ===========================================================================
// 1. round_and_to_int8:  z_hat = clamp(round(z), -128, 127); z_int8 = (int8)z_hat
//    PyTorch: torch.clamp(torch.round(z), -128., 127.)
// ===========================================================================
__global__ void round_to_int8_k(const __half* __restrict__ z,
                                __half* __restrict__ z_hat,
                                int8_t* __restrict__ z_int8, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = __half2float(z[i]);
    v = rintf(v);
    if (v > 127.0f) v = 127.0f;
    if (v < -128.0f) v = -128.0f;
    __half h = __float2half_rn(v);
    z_hat[i] = h;
    z_int8[i] = static_cast<int8_t>(v);
}

extern "C" void dcvc_k_round_to_int8(const __half* z, __half* z_hat,
                                     int8_t* z_int8, int N, cudaStream_t stream) {
    int g, b; launch_config(N, g, b);
    round_to_int8_k<<<g, b, 0, stream>>>(z, z_hat, z_int8, N);
}

// ===========================================================================
// 2. add_and_multiply:  out = (x0 + x1) * q
// ===========================================================================
__global__ void add_and_multiply_k(__half* x0, const __half* x1,
                                   const __half* q, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = __half2float(x0[i]) + __half2float(x1[i]);
    x0[i] = __float2half_rn(v * __half2float(q[i]));
}

extern "C" void dcvc_k_add_and_multiply(__half* x0, const __half* x1,
                                        const __half* q, int N, cudaStream_t stream) {
    int g, b; launch_config(N, g, b);
    add_and_multiply_k<<<g, b, 0, stream>>>(x0, x1, q, N);
}

// ===========================================================================
// 3. clamp_reciprocal_with_quant:
//    q_dec_clamp = max(q_dec, min_val); y_out = y * (1 / q_dec_clamp)
// ===========================================================================
__global__ void clamp_recip_quant_k(const __half* q_dec, const __half* y,
                                    __half* q_dec_clamp, __half* y_out,
                                    float min_val, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float qd = __half2float(q_dec[i]);
    if (qd < min_val) qd = min_val;
    q_dec_clamp[i] = __float2half_rn(qd);
    y_out[i] = __float2half_rn(__half2float(y[i]) * (1.0f / qd));
}

extern "C" void dcvc_k_clamp_recip_quant(const __half* q_dec, const __half* y,
                                         __half* q_dec_clamp, __half* y_out,
                                         float min_val, int N, cudaStream_t stream) {
    int g, b; launch_config(N, g, b);
    clamp_recip_quant_k<<<g, b, 0, stream>>>(q_dec, y, q_dec_clamp, y_out, min_val, N);
}

// ===========================================================================
// 4. combine_for_reading_2x:  out[c] = x[c]*mask[c] + x[c+N]*mask[c+N]
//    N = (C/2) * H * W;  x and mask have 2N elements, out has N elements
// ===========================================================================
__global__ void combine_read_2x_k(const __half* x, const __half* mask,
                                  __half* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s1 = __half2float(x[i]) * __half2float(mask[i]);
    float s2 = __half2float(x[i + N]) * __half2float(mask[i + N]);
    out[i] = __float2half_rn(s1 + s2);
}

extern "C" void dcvc_k_combine_read_2x(const __half* x, const __half* mask,
                                       __half* out, int N, cudaStream_t stream) {
    int g, b; launch_config(N, g, b);
    combine_read_2x_k<<<g, b, 0, stream>>>(x, mask, out, N);
}

// ===========================================================================
// 5. restore_y_2x:  out[i] = (y[i % Cy] + means[i]) * mask[i],  i in [0, 2*Cy*HW)
//    y has Cy*HW elements, means/mask/out have 2*Cy*HW elements
// ===========================================================================
__global__ void restore_y_2x_k(const __half* y, const __half* means,
                               const __half* mask, __half* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = (__half2float(y[i % N < N ? i : i]) + __half2float(means[i])) * __half2float(mask[i]);
    out[i] = __float2half_rn(v);
}

// restore_y needs y broadcast: out[i] = (y[i % CyHW] + means[i]) * mask[i]
// where CyHW = total / factor.  We pass CyHW as a separate param.
__global__ void restore_y_2x_k2(const __half* y, const __half* means,
                                const __half* mask, __half* out, int CyHW, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = (__half2float(y[i % CyHW]) + __half2float(means[i])) * __half2float(mask[i]);
    out[i] = __float2half_rn(v);
}

extern "C" void dcvc_k_restore_y_2x(const __half* y, const __half* means,
                                    const __half* mask, __half* out,
                                    int CyHW, int N, cudaStream_t stream) {
    int g, b; launch_config(N, g, b);
    restore_y_2x_k2<<<g, b, 0, stream>>>(y, means, mask, out, CyHW, N);
}

// ===========================================================================
// 6. restore_y_4x:  same but 4x broadcast
// ===========================================================================
extern "C" void dcvc_k_restore_y_4x(const __half* y, const __half* means,
                                    const __half* mask, __half* out,
                                    int CyHW, int N, cudaStream_t stream) {
    int g, b; launch_config(N, g, b);
    restore_y_2x_k2<<<g, b, 0, stream>>>(y, means, mask, out, CyHW, N);
    // restore_y_4x uses same kernel pattern (broadcast + add + mask),
    // just with N = 4*CyHW and CyHW unchanged.  Launch above handles it.
}

// ===========================================================================
// 7. build_index_dec:  index = (log(clamp(s, min, max)) - log_min) * step_recip
//    Matches PyTorch fallback (CUSTOMIZED_CUDA_INFERENCE=False): each step
//    computes in fp32 then rounds to fp16 (torch.log + scalar ops), output
//    truncated (floor) to uint8.
// ===========================================================================
__global__ void build_index_dec_k(const __half* scales, uint8_t* out,
                                  float scale_min, float scale_max,
                                  float log_scale_min, float log_step_recip, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    __half s = scales[i];
    s = __hmax(s, __float2half(scale_min));
    s = __hmin(s, __float2half(scale_max));
    __half log_val  = __float2half_rn(logf(__half2float(s)));
    __half sub_val  = __float2half_rn(__half2float(log_val) - log_scale_min);
    __half mul_val  = __float2half_rn(__half2float(sub_val) * log_step_recip);
    out[i] = __half2uint_rd(mul_val);
}

extern "C" void dcvc_k_build_index_dec(const __half* scales, uint8_t* out,
                                       float scale_min, float scale_max,
                                       float log_scale_min, float log_step_recip,
                                       int N, cudaStream_t stream) {
    int g, b; launch_config(N, g, b);
    build_index_dec_k<<<g, b, 0, stream>>>(scales, out, scale_min, scale_max,
                                           log_scale_min, log_step_recip, N);
}

// ===========================================================================
// 8. build_index_enc:  out = (int16)((symbol << 8) + index)
// ===========================================================================
__global__ void build_index_enc_k(const __half* symbols, const __half* scales,
                                  int16_t* out, float scale_min, float scale_max,
                                  float log_scale_min, float log_step_recip, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    __half s = scales[i];
    s = __hmax(s, __float2half(scale_min));
    s = __hmin(s, __float2half(scale_max));
    __half log_val  = __float2half_rn(logf(__half2float(s)));
    __half sub_val  = __float2half_rn(__half2float(log_val) - log_scale_min);
    __half mul_val  = __float2half_rn(__half2float(sub_val) * log_step_recip);
    int idx = static_cast<int>(__half2uint_rd(mul_val));
    int sym = static_cast<int>(rintf(__half2float(symbols[i])));
    out[i] = static_cast<int16_t>((sym << 8) + idx);
}

extern "C" void dcvc_k_build_index_enc(const __half* symbols, const __half* scales,
                                       int16_t* out, float scale_min, float scale_max,
                                       float log_scale_min, float log_step_recip,
                                       int N, cudaStream_t stream) {
    int g, b; launch_config(N, g, b);
    build_index_enc_k<<<g, b, 0, stream>>>(symbols, scales, out, scale_min, scale_max,
                                           log_scale_min, log_step_recip, N);
}

// ===========================================================================
// 9. process_with_mask (host-callable version, matches process_mask_plugin.cu)
//    out: y_hat = clamp(round((y - means*mask)*mask), -128, 127) + means*mask
// ===========================================================================
__global__ void process_mask_k(const __half* y, const __half* scales,
                               const __half* means, const __half* mask,
                               __half* y_hat, float force_zero_thres, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float fy = __half2float(y[i]);
    float fs = __half2float(scales[i]);
    float fm = __half2float(means[i]);
    float fmask = __half2float(mask[i]);
    float s_hat = fs * fmask;
    float means_hat = fm * fmask;
    float y_res = (fy - means_hat) * fmask;
    float y_q = rintf(y_res);
    if (force_zero_thres > 0.0f && s_hat > force_zero_thres) y_q = 0.0f;
    if (y_q > 127.0f) y_q = 127.0f;
    if (y_q < -128.0f) y_q = -128.0f;
    y_hat[i] = __float2half_rn(y_q + means_hat);
}

extern "C" void dcvc_k_process_mask(const __half* y, const __half* scales,
                                    const __half* means, const __half* mask,
                                    __half* y_hat, float force_zero_thres,
                                    int N, cudaStream_t stream) {
    int g, b; launch_config(N, g, b);
    process_mask_k<<<g, b, 0, stream>>>(y, scales, means, mask, y_hat, force_zero_thres, N);
}

// ===========================================================================
// 9b. process_with_mask_yq: same quantization as (9) but outputs y_q (the
//     rounded integer latent) instead of y_hat. Used by the encoder to build
//     the packed symbols for rANS. y_q = clamp(round((y-means*mask)*mask)).
// ===========================================================================
__global__ void process_mask_yq_k(const __half* __restrict__ y,
                                  const __half* __restrict__ scales,
                                  const __half* __restrict__ means,
                                  const __half* __restrict__ mask,
                                  __half* __restrict__ y_q,
                                  float force_zero_thres, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float fy = __half2float(y[i]);
    float fs = __half2float(scales[i]);
    float fm = __half2float(means[i]);
    float fmask = __half2float(mask[i]);
    float s_hat = fs * fmask;
    float means_hat = fm * fmask;
    float y_res = (fy - means_hat) * fmask;
    float q = rintf(y_res);
    if (force_zero_thres > 0.0f && s_hat > force_zero_thres) q = 0.0f;
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    y_q[i] = __float2half_rn(q);
}

extern "C" void dcvc_k_process_mask_yq(const __half* y, const __half* scales,
                                       const __half* means, const __half* mask,
                                       __half* y_q, float force_zero_thres,
                                       int N, cudaStream_t stream) {
    int g, b; launch_config(N, g, b);
    process_mask_yq_k<<<g, b, 0, stream>>>(y, scales, means, mask, y_q, force_zero_thres, N);
}

// ===========================================================================
// 10. bias_pixel_shuffle_8:  out[3, H*8, W*8] from x[192, H, W] + bias[192]
//     out[0, c_out, h*8+ry, w*8+rx] = clamp(x[0, c, h, w] + bias[c], 0, 1)
//     where c_out = 3, c = c_out*8+ry... actually:
//     pixel_shuffle(x+bias, 8):  C_in=192=3*64, out[3,H*8,W*8]
// ===========================================================================
__global__ void bias_pixel_shuffle_8_k(const __half* x, const __half* bias,
                                       __half* out, int H, int W, int C, int clamp_en) {
    // one thread per (h, w) input position, loops over 64 channels → 8x8 output block
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= H * W) return;
    int h = hw / W;
    int w = hw % W;

    for (int i = 0; i < C; i++) {
        float v = __half2float(x[i * H * W + hw]) + __half2float(bias[i]);
        if (clamp_en) {
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
        }
        int out_c = i / 64;       // 0,1,2
        int ry = (i % 64) / 8;    // 0..7
        int rx = i % 8;           // 0..7
        int oh = h * 8 + ry;
        int ow = w * 8 + rx;
        out[out_c * (H * 8) * (W * 8) + oh * (W * 8) + ow] = __float2half_rn(v);
    }
}

extern "C" void dcvc_k_bias_pixel_shuffle_8(const __half* x, const __half* bias,
                                            __half* out, int H, int W, int C,
                                            int clamp_en, cudaStream_t stream) {
    int g, b; launch_config(H * W, g, b);
    bias_pixel_shuffle_8_k<<<g, b, 0, stream>>>(x, bias, out, H, W, C, clamp_en);
}

// ===========================================================================
// 11. replicate_pad:  out[1, C, H+pb, W+pr] by replicating edge pixels
// ===========================================================================
__global__ void replicate_pad_k(const __half* x, __half* out,
                                int C, int H, int W, int Hp, int Wp) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Hp * Wp;
    if (n >= total) return;
    int dy = n / Wp;
    int dx = n % Wp;
    int sy = dy < H ? dy : H - 1;
    int sx = dx < W ? dx : W - 1;
    for (int c = 0; c < C; c++) {
        out[c * Hp * Wp + dy * Wp + dx] = x[c * H * W + sy * W + sx];
    }
}

extern "C" void dcvc_k_replicate_pad(const __half* x, __half* out,
                                     int C, int H, int W, int Hp, int Wp,
                                     cudaStream_t stream) {
    int g, b; launch_config(Hp * Wp, g, b);
    replicate_pad_k<<<g, b, 0, stream>>>(x, out, C, H, W, Hp, Wp);
}


// ===========================================================================
// 12. single_part_for_writing_4x
// ===========================================================================
__global__ void single_part_writing_4x_k(const __half* x, __half* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = __half2float(x[i]) + __half2float(x[i + N])
            + __half2float(x[i + 2*N]) + __half2float(x[i + 3*N]);
    out[i] = __float2half_rn(s);
}

extern "C" void dcvc_k_single_part_writing_4x(const __half* x, __half* out,
                                              int N, cudaStream_t stream) {
    int g, b; launch_config(N, g, b);
    single_part_writing_4x_k<<<g, b, 0, stream>>>(x, out, N);
}

// ===========================================================================
// 12b. single_part_for_writing_2x:  out[i] = x[i] + x[i+N]  (half-channel sum)
// ===========================================================================
__global__ void single_part_writing_2x_k(const __half* x, __half* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = __half2float(x[i]) + __half2float(x[i + N]);
    out[i] = __float2half_rn(s);
}

extern "C" void dcvc_k_single_part_writing_2x(const __half* x, __half* out,
                                              int N, cudaStream_t stream) {
    int g, b; launch_config(N, g, b);
    single_part_writing_2x_k<<<g, b, 0, stream>>>(x, out, N);
}

// ===========================================================================
// 13. elementwise multiply
// ===========================================================================
__global__ void elem_mul_k(const __half* a, const __half* b, __half* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = __float2half_rn(__half2float(a[i]) * __half2float(b[i]));
}

extern "C" void dcvc_k_elem_mul(const __half* a, const __half* b, __half* out,
                                int N, cudaStream_t stream) {
    int g, blk; launch_config(N, g, blk);
    elem_mul_k<<<g, blk, 0, stream>>>(a, b, out, N);
}

// ===========================================================================
// 14. pixel_unshuffle_8:  out[192, H/8, W/8] = unshuffle(x[3, H, W], 8)
//     out[c*64+i*8+j, fh, fw] = x[c, fh*8+i, fw*8+j]
// ===========================================================================
__global__ void pixel_unshuffle_8_k(const __half* __restrict__ x,
                                    __half* __restrict__ out,
                                    int H, int W) {
    int fW = W / 8, fH = H / 8;
    int fHW = fH * fW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 192 * fHW;
    if (idx >= total) return;
    int c_out = idx / fHW;       // 0..191
    int rem = idx % fHW;
    int fh = rem / fW;
    int fw = rem % fW;
    int c = c_out / 64;          // 0,1,2
    int i = (c_out % 64) / 8;    // 0..7
    int j = c_out % 8;           // 0..7
    int oh = fh * 8 + i;
    int ow = fw * 8 + j;
    out[idx] = x[c * H * W + oh * W + ow];
}

extern "C" void dcvc_k_pixel_unshuffle_8(const __half* x, __half* out,
                                         int H, int W, cudaStream_t stream) {
    int g, b; launch_config(192 * (H/8) * (W/8), g, b);
    pixel_unshuffle_8_k<<<g, b, 0, stream>>>(x, out, H, W);
}

// ===========================================================================
// 15. mul_qfeat_broadcast:  out[c, i] = x[c, i] * q[c]  (channel-broadcast)
// ===========================================================================
__global__ void mul_qfeat_broadcast_k(const __half* __restrict__ x,
                                      const __half* __restrict__ q,
                                      __half* __restrict__ out,
                                      int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * HW) return;
    int c = idx / HW;
    out[idx] = __float2half_rn(__half2float(x[idx]) * __half2float(q[c]));
}

extern "C" void dcvc_k_mul_qfeat_broadcast(const __half* x, const __half* q,
                                           __half* out, int C, int HW,
                                           cudaStream_t stream) {
    int g, b; launch_config(C * HW, g, b);
    mul_qfeat_broadcast_k<<<g, b, 0, stream>>>(x, q, out, C, HW);
}

// ===========================================================================
// 16. broadcast_mul:  out[ch, i] = in[ch, i] * q[i]  (q broadcast across ch)
// ===========================================================================
__global__ void broadcast_mul_k(const __half* __restrict__ in,
                                const __half* __restrict__ q,
                                __half* __restrict__ out,
                                int nc, int hw) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc * hw) return;
    int i = idx % hw;
    out[idx] = __float2half_rn(__half2float(in[idx]) * __half2float(q[i]));
}

extern "C" void dcvc_k_broadcast_mul(const __half* in, const __half* q,
                                     __half* out, int nc, int hw,
                                     cudaStream_t stream) {
    int g, b; launch_config(nc * hw, g, b);
    broadcast_mul_k<<<g, b, 0, stream>>>(in, q, out, nc, hw);
}

// ===========================================================================
// 17. add_inplace:  out[i] += step[i]
// ===========================================================================
__global__ void add_inplace_k(__half* out, const __half* step, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = __float2half_rn(__half2float(out[i]) + __half2float(step[i]));
}

extern "C" void dcvc_k_add_inplace(__half* out, const __half* step,
                                   int N, cudaStream_t stream) {
    int g, b; launch_config(N, g, b);
    add_inplace_k<<<g, b, 0, stream>>>(out, step, N);
}

// ===========================================================================
// 18. separate_prior_intra:  pf[514, HW] → q_enc[HW], q_dec[HW] (sigmoid)
//     sigmoid(x) * 1.5 + 0.5  on pf[0], pf[1]
// ===========================================================================
__global__ void separate_prior_intra_k(const __half* __restrict__ pf,
                                       __half* __restrict__ q_enc,
                                       __half* __restrict__ q_dec,
                                       int HW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HW) return;
    float v0 = __half2float(pf[i]);
    float v1 = __half2float(pf[HW + i]);
    float s0 = 1.0f / (1.0f + expf(-v0)) * 1.5f + 0.5f;
    float s1 = 1.0f / (1.0f + expf(-v1)) * 1.5f + 0.5f;
    q_enc[i] = __float2half_rn(s0);
    q_dec[i] = __float2half_rn(s1);
}

extern "C" void dcvc_k_separate_prior_intra(const __half* pf, __half* q_enc,
                                            __half* q_dec, int HW,
                                            cudaStream_t stream) {
    int g, b; launch_config(HW, g, b);
    separate_prior_intra_k<<<g, b, 0, stream>>>(pf, q_enc, q_dec, HW);
}

// ===========================================================================
// 19. separate_prior_video_enc:  params[3*nc, HW] → qdec[nc,HW], scales, means
//     Pure split (no clamp).
// ===========================================================================
extern "C" void dcvc_k_separate_prior_video_enc(const __half* params,
                                                __half* qdec, __half* scales,
                                                __half* means, int nc, int HW,
                                                cudaStream_t stream) {
    size_t bw = (size_t)nc * HW * 2;
    cudaMemcpyAsync(qdec,   params,                bw, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(scales, params + (size_t)nc * HW, bw, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(means,  params + (size_t)2 * nc * HW, bw, cudaMemcpyDeviceToDevice, stream);
}

// ===========================================================================
// 20. separate_prior_video_dec:  same split but clamp qdec to >= 0.5
// ===========================================================================
__global__ void separate_prior_video_dec_k(const __half* __restrict__ params,
                                          __half* __restrict__ qdec,
                                          int nc, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc * HW) return;
    float v = __half2float(params[idx]);
    if (v < 0.5f) v = 0.5f;
    qdec[idx] = __float2half_rn(v);
}

extern "C" void dcvc_k_separate_prior_video_dec(const __half* params,
                                                __half* qdec, __half* scales,
                                                __half* means, int nc, int HW,
                                                cudaStream_t stream) {
    int g, b; launch_config(nc * HW, g, b);
    separate_prior_video_dec_k<<<g, b, 0, stream>>>(params, qdec, nc, HW);
    size_t bw = (size_t)nc * HW * 2;
    cudaMemcpyAsync(scales, params + (size_t)nc * HW, bw, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(means,  params + (size_t)2 * nc * HW, bw, cudaMemcpyDeviceToDevice, stream);
}

// ===========================================================================
// 21. int8_to_fp16:  out[i] = __float2half_rn((float)in[i])
//     Converts rANS decoded int8 symbols to FP16 on device.
// ===========================================================================
__global__ void int8_to_fp16_k(const int8_t* __restrict__ in,
                               __half* __restrict__ out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = __float2half_rn((float)in[i]);
}

extern "C" void dcvc_k_int8_to_fp16(const int8_t* in, __half* out,
                                    int N, cudaStream_t stream) {
    int g, b; launch_config(N, g, b);
    int8_to_fp16_k<<<g, b, 0, stream>>>(in, out, N);
}
