/* CPU vs CUDA FXP bit-exactness (device kernels via H2D/D2H).
 *
 * Usage:  test_fxp_cuda_bitexact
 */
#include "fxp/fxp_conv.h"
#include "fxp/fxp_cuda.h"
#include "fxp/fxp_wsrelu.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void fill_f(float* p, int n, unsigned seed)
{
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = ((int)(s >> 8) % 2001 - 1000) / 1000.0f;
    }
}

static void fill_i16(int16_t* p, int n, unsigned seed)
{
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        p[i] = (int16_t)((int)(s >> 8) % 20001 - 10000);
    }
}

template <typename T>
static T* dev_copy(const T* host, size_t n)
{
    T* d = nullptr;
    if (cudaMalloc(&d, n * sizeof(T)) != cudaSuccess)
        return nullptr;
    if (cudaMemcpy(d, host, n * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d);
        return nullptr;
    }
    return d;
}

static int test_1x1(int cin, int cout, int h, int w, unsigned seed)
{
    const int hw = h * w;
    std::vector<float> x((size_t)cin * hw), y_cpu((size_t)cout * hw), y_gpu((size_t)cout * hw);
    std::vector<int16_t> wt((size_t)cout * cin);
    std::vector<float> ws(cout), b(cout);
    fill_f(x.data(), cin * hw, seed);
    fill_i16(wt.data(), cout * cin, seed + 1);
    for (int i = 0; i < cout; i++) {
        ws[i] = 0.0001f * (1 + (i % 7));
        b[i] = 0.001f * (float)i;
    }
    const float xs = 0.02f;
    fxp_conv1x1_f32(x.data(), y_cpu.data(), 1, cin, cout, h, w, wt.data(), ws.data(), b.data(), xs, 16);

    float* dx = dev_copy(x.data(), x.size());
    float* dy = nullptr;
    int16_t* dw = dev_copy(wt.data(), wt.size());
    float* dws = dev_copy(ws.data(), ws.size());
    float* db = dev_copy(b.data(), b.size());
    cudaMalloc(&dy, y_gpu.size() * sizeof(float));
    if (!dx || !dy || !dw || !dws || !db) {
        printf("1x1 %dx%d: CUDA alloc FAIL\n", cin, cout);
        return 1;
    }
    int rc = fxp_conv1x1_f32_cuda(dx, dy, 1, cin, cout, h, w, dw, dws, db, xs, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(y_gpu.data(), dy, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dw);
    cudaFree(dws);
    cudaFree(db);

    int ok = (rc == 0) && (memcmp(y_cpu.data(), y_gpu.data(), y_cpu.size() * sizeof(float)) == 0);
    if (!ok && rc == 0) {
        int mism = 0;
        for (size_t i = 0; i < y_cpu.size(); i++)
            if (y_cpu[i] != y_gpu[i])
                mism++;
        printf("1x1 cin=%d cout=%d %dx%d: FAIL mism=%d/%zu rc=%d\n",
               cin, cout, h, w, mism, y_cpu.size(), rc);
    } else {
        printf("1x1 cin=%d cout=%d %dx%d: %s\n", cin, cout, h, w, ok ? "PASS" : "FAIL");
    }
    return ok ? 0 : 1;
}

static int test_dw(int c, int h, int w, unsigned seed)
{
    const int hw = h * w;
    std::vector<float> x((size_t)c * hw), y_cpu((size_t)c * hw), y_gpu((size_t)c * hw);
    std::vector<int16_t> wt((size_t)c * 9);
    std::vector<float> ws(c), b(c);
    fill_f(x.data(), c * hw, seed);
    fill_i16(wt.data(), c * 9, seed + 2);
    for (int i = 0; i < c; i++) {
        ws[i] = 0.0002f * (1 + (i % 5));
        b[i] = 0.0005f * (float)i;
    }
    const float xs = 0.015f;
    fxp_conv_f32(x.data(), y_cpu.data(), 1, c, c, h, w, wt.data(), ws.data(), b.data(), xs,
                 3, 3, 1, 1, 1, 1, 1, 1, c, 16);

    float* dx = dev_copy(x.data(), x.size());
    float* dy = nullptr;
    int16_t* dw = dev_copy(wt.data(), wt.size());
    float* dws = dev_copy(ws.data(), ws.size());
    float* db = dev_copy(b.data(), b.size());
    cudaMalloc(&dy, y_gpu.size() * sizeof(float));
    int rc = fxp_dwconv3x3_f32_cuda(dx, dy, 1, c, h, w, dw, dws, db, xs, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(y_gpu.data(), dy, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dw);
    cudaFree(dws);
    cudaFree(db);

    int ok = (rc == 0) && (memcmp(y_cpu.data(), y_gpu.data(), y_cpu.size() * sizeof(float)) == 0);
    printf("dw3x3 c=%d %dx%d: %s\n", c, h, w, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int test_general(int cin, int cout, int h, int w,
                        int kh, int kw, int pad_t, int pad_l,
                        int pad_b, int pad_r, int sh, int sw,
                        int group, unsigned seed)
{
    const int oh = (h + pad_t + pad_b - kh) / sh + 1;
    const int ow = (w + pad_l + pad_r - kw) / sw + 1;
    const int cin_g = cin / group;
    std::vector<float> x((size_t)cin * h * w);
    std::vector<float> y_cpu((size_t)cout * oh * ow), y_gpu(y_cpu.size());
    std::vector<int16_t> wt((size_t)cout * cin_g * kh * kw);
    std::vector<float> ws(cout), b(cout);
    fill_f(x.data(), (int)x.size(), seed);
    fill_i16(wt.data(), (int)wt.size(), seed + 1);
    for (int i = 0; i < cout; i++) {
        ws[i] = 0.0001f * (1 + (i % 7));
        b[i] = 0.001f * (float)i;
    }
    const float xs = 0.02f;
    fxp_conv_f32(x.data(), y_cpu.data(), 1, cin, cout, h, w,
                 wt.data(), ws.data(), b.data(), xs, kh, kw,
                 pad_t, pad_l, pad_b, pad_r, sh, sw, group, 16);

    float* dx = dev_copy(x.data(), x.size());
    float* dy = nullptr;
    int16_t* dw = dev_copy(wt.data(), wt.size());
    float* dws = dev_copy(ws.data(), ws.size());
    float* db = dev_copy(b.data(), b.size());
    cudaMalloc(&dy, y_gpu.size() * sizeof(float));
    int rc = fxp_conv_f32_cuda(dx, dy, 1, cin, cout, h, w,
                               dw, dws, db, xs, kh, kw,
                               pad_t, pad_l, pad_b, pad_r, sh, sw, group, nullptr);
    cudaError_t sync_rc = cudaDeviceSynchronize();
    if (sync_rc == cudaSuccess)
        cudaMemcpy(y_gpu.data(), dy, y_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dy); cudaFree(dw); cudaFree(dws); cudaFree(db);

    int mism = 0;
    if (rc == 0 && sync_rc == cudaSuccess) {
        for (size_t i = 0; i < y_cpu.size(); i++)
            if (y_cpu[i] != y_gpu[i]) mism++;
    }
    const int ok = rc == 0 && sync_rc == cudaSuccess && mism == 0;
    printf("general %dx%d k=%dx%d pad=%d,%d,%d,%d stride=%dx%d group=%d: %s",
           cin, cout, kh, kw, pad_t, pad_l, pad_b, pad_r, sh, sw, group,
           ok ? "PASS" : "FAIL");
    if (!ok) printf(" (rc=%d cuda=%s mism=%d/%zu)", rc,
                    cudaGetErrorString(sync_rc), mism, y_cpu.size());
    printf("\n");
    return ok ? 0 : 1;
}

static int test_wsrelu(void)
{
    const int n = 8192;
    std::vector<float> x(n), y_cpu(n), y_gpu(n), lut(65536);
    fill_f(x.data(), n, 99);
    for (int i = 0; i < 65536; i++)
        lut[i] = (float)(i - 32768) * 1e-4f;
    const float xs = 0.01f;
    fxp_wsrelu_f32(x.data(), y_cpu.data(), n, lut.data(), xs);

    float* dx = dev_copy(x.data(), x.size());
    float* dy = nullptr;
    float* dlut = dev_copy(lut.data(), lut.size());
    cudaMalloc(&dy, n * sizeof(float));
    int rc = fxp_wsrelu_f32_cuda(dx, dy, n, dlut, xs, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(y_gpu.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dlut);

    int ok = (rc == 0) && (memcmp(y_cpu.data(), y_gpu.data(), n * sizeof(float)) == 0);
    printf("wsrelu: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main()
{
    int n_gpu = 0;
    cudaGetDeviceCount(&n_gpu);
    if (n_gpu < 1) {
        fprintf(stderr, "no CUDA device\n");
        return 2;
    }
    printf("test_fxp_cuda_bitexact on GPU0 (%d devices)\n", n_gpu);
    int fail = 0;
    fail |= test_1x1(8, 4, 3, 3, 42);
    fail |= test_1x1(256, 512, 16, 16, 7);
    fail |= test_1x1(512, 512, 16, 16, 11);
    fail |= test_1x1(1024, 512, 16, 16, 13);
    fail |= test_dw(64, 16, 16, 23);
    fail |= test_dw(512, 16, 16, 29);
    fail |= test_general(8, 12, 9, 11, 3, 3, 1, 1, 1, 1, 1, 1, 1, 31);
    fail |= test_general(16, 8, 10, 12, 3, 3, 1, 1, 1, 1, 2, 2, 1, 37);
    fail |= test_general(8, 8, 8, 10, 2, 2, 0, 0, 0, 0, 2, 2, 1, 41);
    fail |= test_general(8, 12, 7, 9, 3, 2, 2, 1, 0, 2, 2, 1, 2, 43);
    fail |= test_wsrelu();
    printf(fail ? "OVERALL FAIL\n" : "OVERALL PASS\n");
    return fail ? 1 : 0;
}
