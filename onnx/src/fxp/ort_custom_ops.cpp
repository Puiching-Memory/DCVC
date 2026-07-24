/* ORT custom ops in domain com.dcvc:
 *   FxpConv1x1  — legacy 1x1 (attrs: x_scale; W int16)
 *   FxpConv     — general conv (attrs: x_scale, group, pads[4], strides[2]; W int16)
 *   FxpWsRelu   — LUT WSiLU (attrs: x_scale; input1 = float LUT[65536])
 */

#ifdef __MINGW32__
#ifndef _Frees_ptr_opt_
#define _Frees_ptr_opt_
#endif
#ifndef _Out_z_
#define _Out_z_
#endif
#endif

#include "ort_custom_ops.h"
#include "fxp_conv.h"
#include "det_conv.h"
#include "fxp_wsrelu.h"

#if defined(DCVC_FXP_CUDA)
#include "fxp_cuda.h"
#endif

#include <onnxruntime_cxx_api.h>

#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mutex>

namespace {

static float read_x_scale(const OrtKernelInfo* info)
{
    Ort::ConstKernelInfo ki(info);
    float s = ki.GetAttribute<float>("x_scale");
    if (!(s > 0.f)) {
        fprintf(stderr, "com.dcvc op: x_scale must be > 0 (got %g)\n", s);
        return 1.f;
    }
    return s;
}

/* ---- ORT intra-op ParallelFor helpers (no private thread pool) ---- */

struct Conv1x1Par {
    const int16_t* xq;
    float* y;
    int cin, cout, hw, chunk;
    const int16_t* w_int;
    const float* w_scale;
    const float* bias;
    float x_scale;
};

static void conv1x1_par_fn(void* usr, size_t task)
{
    auto* j = (Conv1x1Par*)usr;
    int oc0 = (int)task * j->chunk;
    int oc1 = std::min(oc0 + j->chunk, j->cout);
    if (oc0 < oc1)
        fxp_conv1x1_oc_range_i16(j->xq, j->y, j->cin, j->cout, j->hw,
                                 j->w_int, j->w_scale, j->bias, j->x_scale,
                                 oc0, oc1);
}

struct Dw3x3Par {
    const int16_t* xq;
    float* y;
    int c, h, w, chunk;
    const int16_t* w_int;
    const float* w_scale;
    const float* bias;
    float x_scale;
};

static void dw3x3_par_fn(void* usr, size_t task)
{
    auto* j = (Dw3x3Par*)usr;
    int oc0 = (int)task * j->chunk;
    int oc1 = std::min(oc0 + j->chunk, j->c);
    if (oc0 < oc1)
        fxp_dw3x3_oc_range_i16(j->xq, j->y, j->c, j->h, j->w,
                               j->w_int, j->w_scale, j->bias, j->x_scale,
                               oc0, oc1);
}

struct Im2colPar {
    const int16_t* col;
    float* y;
    int cin, cout, oh, ow, kh, kw, chunk;
    const int16_t* w_int;
    const float* w_scale;
    const float* bias;
    float x_scale;
};

static void im2col_par_fn(void* usr, size_t task)
{
    auto* j = (Im2colPar*)usr;
    int oc0 = (int)task * j->chunk;
    int oc1 = std::min(oc0 + j->chunk, j->cout);
    if (oc0 < oc1)
        fxp_conv_im2col_oc_range(j->col, j->y, j->cin, j->cout, j->oh, j->ow,
                                 j->w_int, j->w_scale, j->bias, j->x_scale,
                                 j->kh, j->kw, oc0, oc1);
}

struct WsReluPar {
    const float* x;
    float* y;
    const float* lut;
    float x_scale;
    int n, chunk;
};

static void wsrelu_par_fn(void* usr, size_t task)
{
    auto* j = (WsReluPar*)usr;
    int i0 = (int)task * j->chunk;
    int i1 = std::min(i0 + j->chunk, j->n);
    if (i0 < i1)
        fxp_wsrelu_f32_range(j->x, j->y, i0, i1, j->lut, j->x_scale);
}

static int parallel_tasks(int work, int min_per_task)
{
    if (work <= min_per_task) return 1;
    /* Let ORT IntraOp pool decide degree; we only expose coarse tasks. */
    int n = (work + min_per_task - 1) / min_per_task;
    if (n > 64) n = 64;
    return n;
}

struct FxpConv1x1Kernel {
    float x_scale_;
    FxpConv1x1Kernel(const OrtApi&, const OrtKernelInfo* info) : x_scale_(read_x_scale(info)) {}

    void Compute(OrtKernelContext* context)
    {
        Ort::KernelContext ctx(context);
        auto X = ctx.GetInput(0);
        auto W = ctx.GetInput(1);
        auto B = ctx.GetInput(2);
        auto Ws = ctx.GetInput(3);
        auto shape = X.GetTensorTypeAndShapeInfo().GetShape();
        auto w_shape = W.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || w_shape.size() != 4)
            ORT_CXX_API_THROW("FxpConv1x1: bad rank", ORT_INVALID_ARGUMENT);
        const int N = (int)shape[0], Cin = (int)shape[1], H = (int)shape[2], Ww = (int)shape[3];
        const int Cout = (int)w_shape[0];
        const int hw = H * Ww;
        auto Y = ctx.GetOutput(0, std::vector<int64_t>{N, Cout, H, Ww});

        std::vector<int16_t> xq((size_t)Cin * (size_t)hw);
        const int ntasks = parallel_tasks(Cout, 8);
        const int chunk = (Cout + ntasks - 1) / ntasks;

        for (int ni = 0; ni < N; ni++) {
            const float* x_n = X.GetTensorData<float>() + (size_t)ni * Cin * hw;
            float* y_n = Y.GetTensorMutableData<float>() + (size_t)ni * Cout * hw;
            fxp_conv1x1_pack_i16(x_n, xq.data(), Cin, hw, x_scale_);
            Conv1x1Par job{xq.data(), y_n, Cin, Cout, hw, chunk,
                           W.GetTensorData<int16_t>(), Ws.GetTensorData<float>(),
                           B.GetTensorData<float>(), x_scale_};
            ctx.ParallelFor(conv1x1_par_fn, (size_t)ntasks, /*num_batch=*/0, &job);
        }
    }
};

struct ConvWsPar {
    const int16_t* xq;
    float* y;
    int cin, cout, hw, h, w, chunk;
    int is_dw; /* 0=1x1, 1=dw3x3 */
    const int16_t* w_int;
    const float* w_scale;
    const float* bias;
    float x_scale;
    const float* lut;
    float wsrelu_x_scale;
};

static void conv_ws_par_fn(void* usr, size_t task)
{
    auto* j = (ConvWsPar*)usr;
    int oc0 = (int)task * j->chunk;
    int oc1 = std::min(oc0 + j->chunk, j->cout);
    if (oc0 >= oc1) return;
    if (j->is_dw)
        fxp_dw3x3_oc_range_i16(j->xq, j->y, j->cout, j->h, j->w,
                               j->w_int, j->w_scale, j->bias, j->x_scale,
                               oc0, oc1);
    else
        fxp_conv1x1_oc_range_i16(j->xq, j->y, j->cin, j->cout, j->hw,
                                 j->w_int, j->w_scale, j->bias, j->x_scale,
                                 oc0, oc1);
    fxp_wsrelu_nchw_oc_range(j->y, j->hw, oc0, oc1, j->lut, j->wsrelu_x_scale);
}

struct FxpConvWsReluKernel {
    float x_scale_;
    float wsrelu_x_scale_;
    int act_bits_;
    int64_t group_;
    int64_t pads_[4];
    int64_t strides_[2];

    FxpConvWsReluKernel(const OrtApi&, const OrtKernelInfo* info)
        : x_scale_(read_x_scale(info))
    {
        Ort::ConstKernelInfo ki(info);
        try {
            wsrelu_x_scale_ = ki.GetAttribute<float>("wsrelu_x_scale");
        } catch (...) {
            wsrelu_x_scale_ = x_scale_;
        }
        if (!(wsrelu_x_scale_ > 0.f)) wsrelu_x_scale_ = x_scale_;
        group_ = 1;
        try { group_ = ki.GetAttribute<int64_t>("group"); } catch (...) {}
        auto pads = ki.GetAttributes<int64_t>("pads");
        auto strides = ki.GetAttributes<int64_t>("strides");
        if (pads.size() != 4 || strides.size() != 2)
            ORT_CXX_API_THROW("FxpConvWsRelu: pads/strides", ORT_INVALID_ARGUMENT);
        for (int i = 0; i < 4; i++) pads_[i] = pads[i];
        strides_[0] = strides[0];
        strides_[1] = strides[1];
        if (group_ < 1) group_ = 1;
        act_bits_ = 16;
        try { act_bits_ = (int)ki.GetAttribute<int64_t>("act_bits"); } catch (...) {}
        if (act_bits_ != 16 && act_bits_ != 32) act_bits_ = 16;
    }

    void Compute(OrtKernelContext* context)
    {
        Ort::KernelContext ctx(context);
        auto X = ctx.GetInput(0);
        auto W = ctx.GetInput(1);
        auto B = ctx.GetInput(2);
        auto Ws = ctx.GetInput(3);
        auto Lut = ctx.GetInput(4);
        auto shape = X.GetTensorTypeAndShapeInfo().GetShape();
        auto w_shape = W.GetTensorTypeAndShapeInfo().GetShape();
        auto lut_shape = Lut.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || w_shape.size() != 4)
            ORT_CXX_API_THROW("FxpConvWsRelu: bad rank", ORT_INVALID_ARGUMENT);
        int64_t lut_n = 1;
        for (auto d : lut_shape) lut_n *= d;
        if (lut_n != 65536)
            ORT_CXX_API_THROW("FxpConvWsRelu: LUT must have 65536 entries", ORT_INVALID_ARGUMENT);

        const int N = (int)shape[0], Cin = (int)shape[1], H = (int)shape[2], Ww = (int)shape[3];
        const int Cout = (int)w_shape[0], kh = (int)w_shape[2], kw = (int)w_shape[3];
        const int pad_t = (int)pads_[0], pad_l = (int)pads_[1], pad_b = (int)pads_[2], pad_r = (int)pads_[3];
        const int sh = (int)strides_[0], sw = (int)strides_[1];
        const int oh = (H + pad_t + pad_b - kh) / sh + 1;
        const int ow = (Ww + pad_l + pad_r - kw) / sw + 1;
        auto Y = ctx.GetOutput(0, std::vector<int64_t>{N, Cout, oh, ow});

        if (act_bits_ != 16)
            ORT_CXX_API_THROW("FxpConvWsRelu: only act_bits=16 fused path", ORT_INVALID_ARGUMENT);

        const float* xp = X.GetTensorData<float>();
        float* yp = Y.GetTensorMutableData<float>();
        const int16_t* wp = W.GetTensorData<int16_t>();
        const float* wsp = Ws.GetTensorData<float>();
        const float* bp = B.GetTensorData<float>();
        const float* lut = Lut.GetTensorData<float>();

        const int is_1x1 = (kh == 1 && kw == 1 && group_ == 1
            && pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0
            && sh == 1 && sw == 1);
        const int is_dw = (kh == 3 && kw == 3 && group_ == Cin && Cin == Cout
            && pad_t == 1 && pad_l == 1 && pad_b == 1 && pad_r == 1
            && sh == 1 && sw == 1);
        if (!is_1x1 && !is_dw)
            ORT_CXX_API_THROW("FxpConvWsRelu: only 1x1 or dw3x3 fused", ORT_INVALID_ARGUMENT);

        const int hw = H * Ww;
        std::vector<int16_t> xq((size_t)Cin * (size_t)(H + 2) * (size_t)(Ww + 2));
        const int ntasks = parallel_tasks(Cout, is_dw ? 4 : 8);
        const int chunk = (Cout + ntasks - 1) / ntasks;

        for (int ni = 0; ni < N; ni++) {
            const float* x_n = xp + (size_t)ni * Cin * hw;
            float* y_n = yp + (size_t)ni * Cout * hw;
            if (is_dw)
                fxp_dw3x3_pack_i16(x_n, xq.data(), Cin, H, Ww, x_scale_);
            else
                fxp_conv1x1_pack_i16(x_n, xq.data(), Cin, hw, x_scale_);
            ConvWsPar job{xq.data(), y_n, Cin, Cout, hw, H, Ww, chunk, is_dw ? 1 : 0,
                          wp, wsp, bp, x_scale_, lut, wsrelu_x_scale_};
            ctx.ParallelFor(conv_ws_par_fn, (size_t)ntasks, 0, &job);
        }
    }
};

struct FxpConvWsReluOp : Ort::CustomOpBase<FxpConvWsReluOp, FxpConvWsReluKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new FxpConvWsReluKernel(api, info);
    }
    const char* GetName() const { return "FxpConvWsRelu"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 5; }
    ONNXTensorElementDataType GetInputType(size_t i) const
    {
        return i == 1 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

struct FxpConv1x1Op : Ort::CustomOpBase<FxpConv1x1Op, FxpConv1x1Kernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new FxpConv1x1Kernel(api, info);
    }
    const char* GetName() const { return "FxpConv1x1"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 4; }
    ONNXTensorElementDataType GetInputType(size_t i) const
    {
        return i == 1 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

struct FxpConvKernel {
    float x_scale_;
    int act_bits_;
    int64_t group_;
    int64_t pads_[4];
    int64_t strides_[2];

    FxpConvKernel(const OrtApi&, const OrtKernelInfo* info) : x_scale_(read_x_scale(info))
    {
        Ort::ConstKernelInfo ki(info);
        group_ = ki.GetAttribute<int64_t>("group");
        auto pads = ki.GetAttributes<int64_t>("pads");
        auto strides = ki.GetAttributes<int64_t>("strides");
        if (pads.size() != 4 || strides.size() != 2)
            ORT_CXX_API_THROW("FxpConv: pads must be 4 ints, strides 2 ints", ORT_INVALID_ARGUMENT);
        for (int i = 0; i < 4; i++) pads_[i] = pads[i];
        strides_[0] = strides[0];
        strides_[1] = strides[1];
        if (group_ < 1) group_ = 1;
        act_bits_ = 16;
        try { act_bits_ = (int)ki.GetAttribute<int64_t>("act_bits"); } catch (...) {}
        if (act_bits_ != 16 && act_bits_ != 32) act_bits_ = 16;
    }

    void Compute(OrtKernelContext* context)
    {
        Ort::KernelContext ctx(context);
        auto X = ctx.GetInput(0);
        auto W = ctx.GetInput(1);
        auto B = ctx.GetInput(2);
        auto Ws = ctx.GetInput(3);
        auto shape = X.GetTensorTypeAndShapeInfo().GetShape();
        auto w_shape = W.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || w_shape.size() != 4)
            ORT_CXX_API_THROW("FxpConv: bad rank", ORT_INVALID_ARGUMENT);

        const int N = (int)shape[0], Cin = (int)shape[1], H = (int)shape[2], Ww = (int)shape[3];
        const int Cout = (int)w_shape[0], kh = (int)w_shape[2], kw = (int)w_shape[3];
        const int pad_t = (int)pads_[0], pad_l = (int)pads_[1], pad_b = (int)pads_[2], pad_r = (int)pads_[3];
        const int sh = (int)strides_[0], sw = (int)strides_[1];
        const int oh = (H + pad_t + pad_b - kh) / sh + 1;
        const int ow = (Ww + pad_l + pad_r - kw) / sw + 1;

        auto Y = ctx.GetOutput(0, std::vector<int64_t>{N, Cout, oh, ow});
        const float* xp = X.GetTensorData<float>();
        float* yp = Y.GetTensorMutableData<float>();
        const int16_t* wp = W.GetTensorData<int16_t>();
        const float* wsp = Ws.GetTensorData<float>();
        const float* bp = B.GetTensorData<float>();

        /* Fast paths: pack once, ParallelFor over oc via ORT intra-op pool. */
        if (act_bits_ == 16 && kh == 1 && kw == 1 && group_ == 1
            && pad_t == 0 && pad_l == 0 && pad_b == 0 && pad_r == 0
            && sh == 1 && sw == 1) {
            const int hw = H * Ww;
            std::vector<int16_t> xq((size_t)Cin * (size_t)hw);
            const int ntasks = parallel_tasks(Cout, 8);
            const int chunk = (Cout + ntasks - 1) / ntasks;
            for (int ni = 0; ni < N; ni++) {
                const float* x_n = xp + (size_t)ni * Cin * hw;
                float* y_n = yp + (size_t)ni * Cout * hw;
                fxp_conv1x1_pack_i16(x_n, xq.data(), Cin, hw, x_scale_);
                Conv1x1Par job{xq.data(), y_n, Cin, Cout, hw, chunk, wp, wsp, bp, x_scale_};
                ctx.ParallelFor(conv1x1_par_fn, (size_t)ntasks, 0, &job);
            }
            return;
        }
        if (act_bits_ == 16 && kh == 3 && kw == 3 && group_ == Cin && Cin == Cout
            && pad_t == 1 && pad_l == 1 && pad_b == 1 && pad_r == 1
            && sh == 1 && sw == 1) {
            const int hw = H * Ww;
            std::vector<int16_t> xq((size_t)Cin * (size_t)(H + 2) * (size_t)(Ww + 2));
            const int ntasks = parallel_tasks(Cout, 4);
            const int chunk = (Cout + ntasks - 1) / ntasks;
            for (int ni = 0; ni < N; ni++) {
                const float* x_n = xp + (size_t)ni * Cin * hw;
                float* y_n = yp + (size_t)ni * Cout * hw;
                fxp_dw3x3_pack_i16(x_n, xq.data(), Cin, H, Ww, x_scale_);
                Dw3x3Par job{xq.data(), y_n, Cin, H, Ww, chunk, wp, wsp, bp, x_scale_};
                ctx.ParallelFor(dw3x3_par_fn, (size_t)ntasks, 0, &job);
            }
            return;
        }

        /* General k×k group==1 conv: im2col + SIMD GEMM with ParallelFor. */
        if (act_bits_ == 16 && group_ == 1) {
            const int hw = H * Ww;
            for (int ni = 0; ni < N; ni++) {
                const float* x_n = xp + (size_t)ni * Cin * hw;
                float* y_n = yp + (size_t)ni * Cout * oh * ow;
                int16_t* col = fxp_conv_im2col_build(x_n, Cin, H, Ww,
                                                     kh, kw, pad_t, pad_l,
                                                     sh, sw, oh, ow, x_scale_);
                if (!col) {
                    fxp_conv_f32(xp, yp, N, Cin, Cout, H, Ww, wp, wsp, bp, x_scale_,
                                 kh, kw, pad_t, pad_l, pad_b, pad_r, sh, sw,
                                 (int)group_, act_bits_);
                    break;
                }
                const int ntasks = parallel_tasks(Cout, 8);
                const int chunk = (Cout + ntasks - 1) / ntasks;
                Im2colPar job{col, y_n, Cin, Cout, oh, ow, kh, kw, chunk,
                              wp, wsp, bp, x_scale_};
                ctx.ParallelFor(im2col_par_fn, (size_t)ntasks, 0, &job);
            }
            return;
        }

        fxp_conv_f32(xp, yp, N, Cin, Cout, H, Ww, wp, wsp, bp, x_scale_,
                     kh, kw, pad_t, pad_l, pad_b, pad_r, sh, sw, (int)group_, act_bits_);
    }
};

struct FxpConvOp : Ort::CustomOpBase<FxpConvOp, FxpConvKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new FxpConvKernel(api, info);
    }
    const char* GetName() const { return "FxpConv"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 4; }
    ONNXTensorElementDataType GetInputType(size_t i) const
    {
        return i == 1 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

struct FxpWsReluKernel {
    float x_scale_;
    FxpWsReluKernel(const OrtApi&, const OrtKernelInfo* info) : x_scale_(read_x_scale(info)) {}

    void Compute(OrtKernelContext* context)
    {
        Ort::KernelContext ctx(context);
        auto X = ctx.GetInput(0);
        auto Lut = ctx.GetInput(1);
        auto shape = X.GetTensorTypeAndShapeInfo().GetShape();
        auto lut_shape = Lut.GetTensorTypeAndShapeInfo().GetShape();
        int64_t n = 1;
        for (auto d : shape) n *= d;
        int64_t lut_n = 1;
        for (auto d : lut_shape) lut_n *= d;
        if (lut_n != 65536)
            ORT_CXX_API_THROW("FxpWsRelu: LUT must have 65536 entries", ORT_INVALID_ARGUMENT);

        auto Y = ctx.GetOutput(0, shape);
        const float* xp = X.GetTensorData<float>();
        float* yp = Y.GetTensorMutableData<float>();
        const float* lut = Lut.GetTensorData<float>();
        const int ne = (int)n;
        const int ntasks = parallel_tasks(ne, 4096);
        const int chunk = (ne + ntasks - 1) / ntasks;
        WsReluPar job{xp, yp, lut, x_scale_, ne, chunk};
        ctx.ParallelFor(wsrelu_par_fn, (size_t)ntasks, 0, &job);
    }
};

struct FxpWsReluOp : Ort::CustomOpBase<FxpWsReluOp, FxpWsReluKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new FxpWsReluKernel(api, info);
    }
    const char* GetName() const { return "FxpWsRelu"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 2; }
    ONNXTensorElementDataType GetInputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};







struct DetPar {
    const float* x;
    float* y;
    int n, cin, cout, h, ww, kh, kw;
    int pad_t, pad_l, pad_b, pad_r, sh, sw, group;
    const float* weight;
    const float* bias;
    int chunk;
};

static void det_par_fn(void* usr, size_t task)
{
    auto* j = (DetPar*)usr;
    int oc0 = (int)task * j->chunk;
    int oc1 = std::min(oc0 + j->chunk, j->cout);
    if (oc0 < oc1)
        det_conv_f32_range(j->x, j->y, j->n, j->cin, j->cout, j->h, j->ww,
                           j->weight, j->bias, j->kh, j->kw,
                           j->pad_t, j->pad_l, j->pad_b, j->pad_r,
                           j->sh, j->sw, j->group, oc0, oc1);
}

/* ── DetConv: deterministic float32 Conv (float64 accumulation) ── */

struct DetConvKernel {
    int64_t group_{};
    int64_t pads_[4]{};
    int64_t strides_[2]{};

    DetConvKernel(const OrtApi&, const OrtKernelInfo* info)
    {
        Ort::ConstKernelInfo ki(info);
        group_ = ki.GetAttribute<int64_t>("group");
        auto pads = ki.GetAttributes<int64_t>("pads");
        auto strides = ki.GetAttributes<int64_t>("strides");
        if (pads.size() != 4 || strides.size() != 2)
            ORT_CXX_API_THROW("DetConv: pads must be 4 ints, strides 2 ints", ORT_INVALID_ARGUMENT);
        for (int i = 0; i < 4; i++) pads_[i] = pads[i];
        strides_[0] = strides[0];
        strides_[1] = strides[1];
        if (group_ < 1) group_ = 1;
    }

    void Compute(OrtKernelContext* context)
    {
        Ort::KernelContext ctx(context);
        auto X = ctx.GetInput(0);
        auto W = ctx.GetInput(1);
        auto B = ctx.GetInput(2);
        auto shape = X.GetTensorTypeAndShapeInfo().GetShape();
        auto w_shape = W.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || w_shape.size() != 4)
            ORT_CXX_API_THROW("DetConv: bad rank", ORT_INVALID_ARGUMENT);
        const int N = (int)shape[0], Cin = (int)shape[1], H = (int)shape[2], Ww = (int)shape[3];
        const int Cout = (int)w_shape[0], kh = (int)w_shape[2], kw = (int)w_shape[3];
        const int pad_t = (int)pads_[0], pad_l = (int)pads_[1];
        const int pad_b = (int)pads_[2], pad_r = (int)pads_[3];
        const int sh = (int)strides_[0], sw = (int)strides_[1];
        const int oh = (H + pad_t + pad_b - kh) / sh + 1;
        const int ow = (Ww + pad_l + pad_r - kw) / sw + 1;
        auto Y = ctx.GetOutput(0, std::vector<int64_t>{N, Cout, oh, ow});
        const float* xp = X.GetTensorData<float>();
        float* yp = Y.GetTensorMutableData<float>();
        const float* wp = W.GetTensorData<float>();
        const float* bp = B.GetTensorData<float>();
        const int ntasks = parallel_tasks(Cout, 8);
        const int chunk = (Cout + ntasks - 1) / ntasks;
        DetPar job{xp, yp, N, Cin, Cout, H, Ww, kh, kw,
                   pad_t, pad_l, pad_b, pad_r, sh, sw, (int)group_,
                   wp, bp, chunk};
        ctx.ParallelFor(det_par_fn, (size_t)ntasks, 0, &job);
    }
};

struct DetConvOp : Ort::CustomOpBase<DetConvOp, DetConvKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new DetConvKernel(api, info);
    }
    const char* GetName() const { return "DetConv"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 3; }
    ONNXTensorElementDataType GetInputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

static DetConvOp g_det_conv;
static FxpConv1x1Op g_fxp_conv1x1;
static FxpConvOp g_fxp_conv;
static FxpWsReluOp g_fxp_wsrelu;
static FxpConvWsReluOp g_fxp_conv_wsrelu;

#if defined(DCVC_FXP_CUDA)

struct FxpConv1x1CudaKernel {
    float x_scale_;
    FxpConv1x1CudaKernel(const OrtApi&, const OrtKernelInfo* info) : x_scale_(read_x_scale(info)) {}

    void Compute(OrtKernelContext* context)
    {
        Ort::KernelContext ctx(context);
        auto X = ctx.GetInput(0);
        auto W = ctx.GetInput(1);
        auto B = ctx.GetInput(2);
        auto Ws = ctx.GetInput(3);
        auto shape = X.GetTensorTypeAndShapeInfo().GetShape();
        auto w_shape = W.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || w_shape.size() != 4)
            ORT_CXX_API_THROW("FxpConv1x1 CUDA: bad rank", ORT_INVALID_ARGUMENT);
        const int64_t N = shape[0], Cin = shape[1], H = shape[2], Ww = shape[3], Cout = w_shape[0];
        auto Y = ctx.GetOutput(0, std::vector<int64_t>{N, Cout, H, Ww});
        void* stream = ctx.GetGPUComputeStream();
        int rc = fxp_conv1x1_f32_cuda(X.GetTensorData<float>(), Y.GetTensorMutableData<float>(),
                                      (int)N, (int)Cin, (int)Cout, (int)H, (int)Ww,
                                      W.GetTensorData<int16_t>(), Ws.GetTensorData<float>(),
                                      B.GetTensorData<float>(), x_scale_, stream);
        if (rc)
            ORT_CXX_API_THROW("FxpConv1x1 CUDA kernel failed", ORT_FAIL);
    }
};

struct FxpConv1x1CudaOp : Ort::CustomOpBase<FxpConv1x1CudaOp, FxpConv1x1CudaKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new FxpConv1x1CudaKernel(api, info);
    }
    const char* GetName() const { return "FxpConv1x1"; }
    const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
    size_t GetInputTypeCount() const { return 4; }
    ONNXTensorElementDataType GetInputType(size_t i) const
    {
        return i == 1 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

struct FxpConvCudaKernel {
    float x_scale_;
    int64_t group_;
    int64_t pads_[4];
    int64_t strides_[2];

    FxpConvCudaKernel(const OrtApi&, const OrtKernelInfo* info) : x_scale_(read_x_scale(info))
    {
        Ort::ConstKernelInfo ki(info);
        group_ = ki.GetAttribute<int64_t>("group");
        auto pads = ki.GetAttributes<int64_t>("pads");
        auto strides = ki.GetAttributes<int64_t>("strides");
        if (pads.size() != 4 || strides.size() != 2)
            ORT_CXX_API_THROW("FxpConv CUDA: pads/strides", ORT_INVALID_ARGUMENT);
        for (int i = 0; i < 4; i++) pads_[i] = pads[i];
        strides_[0] = strides[0];
        strides_[1] = strides[1];
        if (group_ < 1) group_ = 1;
    }

    void Compute(OrtKernelContext* context)
    {
        Ort::KernelContext ctx(context);
        auto X = ctx.GetInput(0);
        auto W = ctx.GetInput(1);
        auto B = ctx.GetInput(2);
        auto Ws = ctx.GetInput(3);
        auto shape = X.GetTensorTypeAndShapeInfo().GetShape();
        auto w_shape = W.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || w_shape.size() != 4)
            ORT_CXX_API_THROW("FxpConv CUDA: bad rank", ORT_INVALID_ARGUMENT);

        const int N = (int)shape[0], Cin = (int)shape[1], H = (int)shape[2], Ww = (int)shape[3];
        const int Cout = (int)w_shape[0], kh = (int)w_shape[2], kw = (int)w_shape[3];
        const int pad_t = (int)pads_[0], pad_l = (int)pads_[1], pad_b = (int)pads_[2], pad_r = (int)pads_[3];
        const int sh = (int)strides_[0], sw = (int)strides_[1];
        const int oh = (H + pad_t + pad_b - kh) / sh + 1;
        const int ow = (Ww + pad_l + pad_r - kw) / sw + 1;

        auto Y = ctx.GetOutput(0, std::vector<int64_t>{N, Cout, oh, ow});
        void* stream = ctx.GetGPUComputeStream();
        if (getenv("DCVC_FXP_CUDA_TRACE"))
            fprintf(stderr, "dcvc_ort: FxpConv CUDA Compute stream=%p N=%d Cin=%d Cout=%d\n",
                    stream, N, Cin, Cout);
        int rc = fxp_conv_f32_cuda(X.GetTensorData<float>(), Y.GetTensorMutableData<float>(),
                                   N, Cin, Cout, H, Ww,
                                   W.GetTensorData<int16_t>(), Ws.GetTensorData<float>(),
                                   B.GetTensorData<float>(), x_scale_,
                                   kh, kw, pad_t, pad_l, pad_b, pad_r, sh, sw, (int)group_,
                                   stream);
        if (rc == -1)
            ORT_CXX_API_THROW("FxpConv CUDA: unsupported shape (need 1x1 or dw3x3)", ORT_FAIL);
        if (rc)
            ORT_CXX_API_THROW("FxpConv CUDA kernel failed", ORT_FAIL);
    }
};

struct FxpConvCudaOp : Ort::CustomOpBase<FxpConvCudaOp, FxpConvCudaKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new FxpConvCudaKernel(api, info);
    }
    const char* GetName() const { return "FxpConv"; }
    const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
    size_t GetInputTypeCount() const { return 4; }
    ONNXTensorElementDataType GetInputType(size_t i) const
    {
        return i == 1 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

struct FxpWsReluCudaKernel {
    float x_scale_;
    FxpWsReluCudaKernel(const OrtApi&, const OrtKernelInfo* info) : x_scale_(read_x_scale(info)) {}

    void Compute(OrtKernelContext* context)
    {
        Ort::KernelContext ctx(context);
        auto X = ctx.GetInput(0);
        auto Lut = ctx.GetInput(1);
        auto shape = X.GetTensorTypeAndShapeInfo().GetShape();
        auto lut_shape = Lut.GetTensorTypeAndShapeInfo().GetShape();
        int64_t n = 1;
        for (auto d : shape) n *= d;
        int64_t lut_n = 1;
        for (auto d : lut_shape) lut_n *= d;
        if (lut_n != 65536)
            ORT_CXX_API_THROW("FxpWsRelu CUDA: LUT must have 65536 entries", ORT_INVALID_ARGUMENT);

        auto Y = ctx.GetOutput(0, shape);
        void* stream = ctx.GetGPUComputeStream();
        int rc = fxp_wsrelu_f32_cuda(X.GetTensorData<float>(), Y.GetTensorMutableData<float>(),
                                     (int)n, Lut.GetTensorData<float>(), x_scale_, stream);
        if (rc)
            ORT_CXX_API_THROW("FxpWsRelu CUDA kernel failed", ORT_FAIL);
    }
};

struct FxpWsReluCudaOp : Ort::CustomOpBase<FxpWsReluCudaOp, FxpWsReluCudaKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const
    {
        return new FxpWsReluCudaKernel(api, info);
    }
    const char* GetName() const { return "FxpWsRelu"; }
    const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
    size_t GetInputTypeCount() const { return 2; }
    ONNXTensorElementDataType GetInputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

static FxpConv1x1CudaOp g_fxp_conv1x1_cuda;
static FxpConvCudaOp g_fxp_conv_cuda;
static FxpWsReluCudaOp g_fxp_wsrelu_cuda;

#endif  // DCVC_FXP_CUDA

static OrtCustomOpDomain* g_domain = nullptr;
static std::once_flag g_once;

static void init_domain(const OrtApi* api)
{
    OrtStatus* st = api->CreateCustomOpDomain("com.dcvc", &g_domain);
    if (st) {
        fprintf(stderr, "dcvc_ort: CreateCustomOpDomain failed: %s\n", api->GetErrorMessage(st));
        api->ReleaseStatus(st);
        g_domain = nullptr;
        return;
    }
    const OrtCustomOp* ops[] = {
        &g_det_conv, &g_fxp_conv1x1, &g_fxp_conv, &g_fxp_wsrelu, &g_fxp_conv_wsrelu,
#if defined(DCVC_FXP_CUDA)
        &g_fxp_conv1x1_cuda, &g_fxp_conv_cuda, &g_fxp_wsrelu_cuda,
#endif
    };
    for (auto* op : ops) {
        st = api->CustomOpDomain_Add(g_domain, op);
        if (st) {
            fprintf(stderr, "dcvc_ort: CustomOpDomain_Add failed: %s\n", api->GetErrorMessage(st));
            api->ReleaseStatus(st);
        }
    }
#if defined(DCVC_FXP_CUDA)
    fprintf(stderr, "dcvc_ort: registered com.dcvc FXP ops for CPU + CUDA EP\n");
#endif
}

}  // namespace

extern "C" int dcvc_ort_register_custom_ops(const OrtApi* api, OrtSessionOptions* opts)
{
    if (!api || !opts) return 1;
    std::call_once(g_once, init_domain, api);
    if (!g_domain) return 2;
    OrtStatus* st = api->AddCustomOpDomain(opts, g_domain);
    if (st) {
        fprintf(stderr, "dcvc_ort: AddCustomOpDomain failed: %s\n", api->GetErrorMessage(st));
        api->ReleaseStatus(st);
        return 3;
    }
    return 0;
}
