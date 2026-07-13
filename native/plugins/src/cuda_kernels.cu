// CUDA kernel port targets (from src/layers/extensions/inference/kernel.cu).
// Compile with DCVC_RT_HAS_CUDA=ON. Kernels are ATen-free: raw __half* pointers.
//
// Priority ports:
//   process_with_mask, combine_for_reading_2x, restore_y_2x/4x,
//   build_index_enc/dec, bias_pixel_shuffle_8, bias_wsilu_depthwise_conv2d,
//   round_and_to_int8, add_and_multiply, replicate_pad
//
// Until kernels are linked, plugins/src/plugins_cpu.c provides CPU reference.

#ifdef DCVC_RT_HAS_CUDA

#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" void dcvc_cuda_add_and_multiply_launch(half* y0, const half* y1, const half* q,
                                                  int n, cudaStream_t stream);

__global__ void add_mul_kernel(half* y0, const half* y1, const half* q, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(y0[i]) + __half2float(y1[i]);
    y0[i] = __float2half(v * __half2float(q[i]));
}

extern "C" void dcvc_cuda_add_and_multiply_launch(half* y0, const half* y1, const half* q,
                                                  int n, cudaStream_t stream)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    add_mul_kernel<<<grid, block, 0, stream>>>(y0, y1, q, n);
}

#endif
