// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
//
// DcvcSubpelConv2x — fused TensorRT IPluginV3 for SubpelConv2x.
//
// Computation (matches SubpelConv2x.forward_torch in src/layers/layers.py):
//   conv_out = Conv2d(x, weight[Cout*4, Cin, k, k], bias[Cout*4], pad)
//   shuffled = PixelShuffle(conv_out, 2)     // [Cout*4,H,W] -> [Cout, 2H, 2W]
//   if to_cat is not None:
//       out = cat(to_cat, shuffled, dim=1)   // cat_at_front, or reversed
//   else:
//       out = shuffled
//
// k=1: cuBLAS GEMM (pointwise).  k=3: im2col + cuBLAS GEMM.

#include <NvInfer.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace dcvc_subpel {

constexpr char const* kName = "DcvcSubpelConv2x";
constexpr char const* kVersion = "1";

// ===================== CUDA kernels =====================

// im2col for 3x3 conv with padding=1: expands x[Cin,H,W] -> col[Cin*9, H*W]
__global__ void im2col_3x3_k(const __half* in, __half* col, int Cin, int H, int W) {
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= H * W || c >= Cin) return;
    int h = hw / W, w = hw % W;
    for (int k = 0; k < 9; k++) {
        int dy = k / 3 - 1;
        int dw = k % 3 - 1;
        int ih = h + dy, iw = w + dw;
        float val = 0.0f;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
            val = __half2float(in[c * H * W + ih * W + iw]);
        col[(c * 9 + k) * H * W + hw] = __float2half_rn(val);
    }
}

// PixelShuffle(2) with fused bias add.
// conv_out: [Cout*4, H, W], bias: [Cout*4], out: [Cout, 2H, 2W]
// out[c, 2h+i, 2w+j] = conv_out[c*4 + i*2 + j, h, w] + bias[c*4 + i*2 + j]
__global__ void pixel_shuffle_bias_2x_k(const __half* conv_out, const __half* bias,
                                        __half* out, int Cout, int H, int W) {
    int c = blockIdx.z;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= H * W) return;
    int h = hw / W, w = hw % W;
    int OH = H * 2, OW = W * 2;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int idx = c * 4 + i * 2 + j;
            float val = __half2float(conv_out[idx * H * W + hw]) + __half2float(bias[idx]);
            out[c * OH * OW + (h * 2 + i) * OW + (w * 2 + j)] = __float2half_rn(val);
        }
    }
}

// Copy a tensor block (for cat assembly)
__global__ void copy_channels_k(const __half* src, __half* dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] = src[i];
}

// ===================== cuBLAS =====================
static cublasHandle_t get_cublas() {
    static cublasHandle_t h = nullptr;
    if (!h) cublasCreate(&h);
    return h;
}

// pointwise conv (1x1): out[Co,HW] = weight[Co,Ci] @ in[Ci,HW]
static void pw_conv(cublasHandle_t h, const __half* in, const __half* weight,
                    __half* out, int Ci, int Co, int HW) {
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, HW, Co, Ci, &alpha,
                 in, CUDA_R_16F, HW, weight, CUDA_R_16F, Ci, &beta,
                 out, CUDA_R_16F, HW, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
}

// ===================== Weight storage =====================
__global__ void f32_to_f16_k(const float* in, __half* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = __float2half_rn(in[i]);
}

// Reshape 3x3 weight [Co, Ci*9] from PyTorch [Co, Ci, 3, 3] — already row-major.
// For 1x1 weight [Co, Ci] from PyTorch [Co, Ci, 1, 1] — already correct.
struct WeightBuf {
    std::vector<uint8_t> host;
    __half* dev = nullptr;
    int count = 0;
    void store(const void* src, size_t nbytes) {
        count = (int)(nbytes / sizeof(float));
        host.resize(nbytes);
        std::memcpy(host.data(), src, nbytes);
    }
    const __half* dptr() {
        if (!dev && count > 0) {
            float* tmp;
            cudaMalloc(&tmp, count * sizeof(float));
            cudaMemcpy(tmp, host.data(), count * sizeof(float), cudaMemcpyHostToDevice);
            cudaMalloc(&dev, count * sizeof(__half));
            f32_to_f16_k<<<(count+255)/256, 256>>>(tmp, dev, count);
            cudaFree(tmp);
        }
        return dev;
    }
};

// ===================== Plugin =====================
class SubpelConvPlugin : public IPluginV3,
                         public IPluginV3OneCore,
                         public IPluginV3OneBuild,
                         public IPluginV3OneRuntime {
public:
    SubpelConvPlugin() = default;

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        if (type == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return nullptr;
    }

    IPluginV3* clone() noexcept override {
        auto* p = new SubpelConvPlugin(*this);
        p->mW[0].dev = nullptr;
        p->mW[1].dev = nullptr;
        return p;
    }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return mNs.c_str(); }
    void setPluginNamespace(char const* ns) noexcept { mNs = ns ? ns : ""; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbIn,
                            DynamicPluginTensorDesc const*, int32_t) noexcept override {
        if (nbIn > 1) {
            mInputMode = true; mNbInputs = nbIn;
            mKernelSize = in[1].desc.dims.d[2];  // weight [Co*4, Ci, k, k]
            mPadding = mKernelSize / 2;
            mHasCat = (nbIn > 3);
        }
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
                                   int32_t nbI, int32_t nbO) noexcept override {
        if (pos < 0 || pos >= nbI + nbO) return false;
        return inOut[pos].desc.type == DataType::kHALF && inOut[pos].desc.format == TensorFormat::kLINEAR;
    }
    int32_t getOutputDataTypes(DataType* ots, int32_t, DataType const* its, int32_t) const noexcept override {
        ots[0] = its[0]; return 0;
    }
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbI, DimsExprs const*,
                            int32_t, DimsExprs* outputs, int32_t nbO, IExprBuilder& eb) noexcept override {
        if (nbI < 1 || nbO != 1) return -1;
        if (nbI > 1) { mInputMode = true; mNbInputs = nbI; }
        outputs[0].nbDims = 4;
        outputs[0].d[0] = inputs[0].d[0];
        int outC = mOutCh + (mHasCat ? mCatCh : 0);
        outputs[0].d[1] = outC > 0 ? eb.constant(outC) : (nbI > 1 ? inputs[1].d[0] : inputs[0].d[1]);
        outputs[0].d[2] = eb.operation(DimensionOperation::kPROD, *inputs[0].d[2], *eb.constant(2));
        outputs[0].d[3] = eb.operation(DimensionOperation::kPROD, *inputs[0].d[3], *eb.constant(2));
        return 0;
    }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t nbIn,
                            DynamicPluginTensorDesc const*, int32_t) const noexcept override {
        int Cin = mInCh, Cout4 = mOutCh * 4;
        if (nbIn > 1) {
            Cin = in[1].desc.dims.d[1];
            Cout4 = in[1].desc.dims.d[0];
        }
        if (Cin == 0 && nbIn > 0) Cin = in[0].max.d[1] > 0 ? in[0].max.d[1] : in[0].desc.dims.d[1];
        if (Cin == 0) Cin = 256;
        if (Cout4 == 0) Cout4 = 1024;
        int H = 0, W = 0;
        if (nbIn > 0) {
            H = in[0].max.d[2] > 0 ? in[0].max.d[2] : in[0].desc.dims.d[2];
            W = in[0].max.d[3] > 0 ? in[0].max.d[3] : in[0].desc.dims.d[3];
        }
        if (H <= 0) H = 64; if (W <= 0) W = 64;
        int HW = H * W;
        int kSize = (nbIn > 1) ? in[1].desc.dims.d[2] : mKernelSize;
        size_t col_sz = (kSize == 3) ? (size_t)Cin * 9 * HW * sizeof(__half) : 0;
        size_t conv_sz = (size_t)Cout4 * HW * sizeof(__half);
        return col_sz + conv_sz + 4096;
    }
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*,
                          int32_t) noexcept override { return 0; }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setWeight(int idx, const void* data, size_t nbytes) {
        if (idx >= 0 && idx < 2) mW[idx].store(data, nbytes);
    }
    void setConfig(int inCh, int outCh, int kSize, int pad, bool hasCat, bool catFront, int catCh) {
        mInCh = inCh; mOutCh = outCh; mKernelSize = kSize; mPadding = pad;
        mHasCat = hasCat; mCatAtFront = catFront; mCatCh = catCh;
    }
    void setInputMode() { mInputMode = true; }

private:
    std::string mNs;
    int mInCh = 0, mOutCh = 0, mKernelSize = 1, mPadding = 0, mCatCh = 0, mNbInputs = 1;
    bool mHasCat = false, mCatAtFront = true, mInputMode = false;
    WeightBuf mW[2];  // 0=weight, 1=bias
};

PluginFieldCollection const* SubpelConvPlugin::getFieldsToSerialize() noexcept {
    static std::vector<PluginField> fields;
    static PluginFieldCollection fc;
    static float sCfgF[7];
    fields.clear();
    if (mW[0].count > 0)
        fields.push_back({"weight", mW[0].host.data(), PluginFieldType::kFLOAT32, (int32_t)mW[0].count});
    if (mW[1].count > 0)
        fields.push_back({"bias", mW[1].host.data(), PluginFieldType::kFLOAT32, (int32_t)mW[1].count});
    sCfgF[0] = (float)mInCh; sCfgF[1] = (float)mOutCh;
    sCfgF[2] = (float)mKernelSize; sCfgF[3] = (float)mPadding;
    sCfgF[4] = mHasCat ? 1.f : 0.f; sCfgF[5] = mCatAtFront ? 1.f : 0.f;
    sCfgF[6] = (float)mCatCh;
    fields.push_back({"in_ch", &sCfgF[0], PluginFieldType::kFLOAT32, 1});
    fields.push_back({"out_ch", &sCfgF[1], PluginFieldType::kFLOAT32, 1});
    fields.push_back({"kernel_size", &sCfgF[2], PluginFieldType::kFLOAT32, 1});
    fields.push_back({"padding", &sCfgF[3], PluginFieldType::kFLOAT32, 1});
    fields.push_back({"has_cat", &sCfgF[4], PluginFieldType::kFLOAT32, 1});
    fields.push_back({"cat_at_front", &sCfgF[5], PluginFieldType::kFLOAT32, 1});
    fields.push_back({"cat_ch", &sCfgF[6], PluginFieldType::kFLOAT32, 1});
    fc.nbFields = (int32_t)fields.size();
    fc.fields = fields.data();
    return &fc;
}

int32_t SubpelConvPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                                  void const* const* inputs, void* const* outputs, void* workspace,
                                  cudaStream_t stream) noexcept {
    cublasSetStream(get_cublas(), stream);
    int H = inputDesc[0].dims.d[2], W = inputDesc[0].dims.d[3];
    int HW = H * W;

    const __half* weight;
    const __half* bias;
    int Cin, Cout4, kSize;

    if (mInputMode) {
        weight = (const __half*)inputs[1];
        bias = (const __half*)inputs[2];
        Cin = inputDesc[1].dims.d[1];
        Cout4 = inputDesc[1].dims.d[0];
        kSize = inputDesc[1].dims.d[2];
    } else {
        weight = mW[0].dptr();
        bias = mW[1].dptr();
        Cin = mInCh;
        Cout4 = mOutCh * 4;
        kSize = mKernelSize;
    }

    int Cout = Cout4 / 4;
    int OH = H * 2, OW = W * 2;

    __half* ws = (__half*)workspace;
    __half* col = nullptr;
    if (kSize == 3) {
        col = ws;
        ws += (size_t)Cin * 9 * HW;
    }
    __half* conv_out = ws;

    const __half* x_in = (const __half*)inputs[0];

    if (kSize == 1) {
        pw_conv(get_cublas(), x_in, weight, conv_out, Cin, Cout4, HW);
    } else {
        dim3 col_grid((HW + 255) / 256, Cin);
        im2col_3x3_k<<<col_grid, 256, 0, stream>>>(x_in, col, Cin, H, W);
        pw_conv(get_cublas(), col, weight, conv_out, Cin * 9, Cout4, HW);
    }

    __half* out_base = (__half*)outputs[0];
    dim3 ps_grid((HW + 255) / 256, 1, Cout);
    pixel_shuffle_bias_2x_k<<<ps_grid, 256, 0, stream>>>(conv_out, bias, out_base, Cout, H, W);

    if (mInputMode && mNbInputs > 3) {
        const __half* cat_in = (const __half*)inputs[3];
        size_t cat_elems = (size_t)(inputDesc[3].dims.d[1]) * OH * OW;
        if (mCatAtFront) {
            copy_channels_k<<<(cat_elems+255)/256, 256, 0, stream>>>(cat_in, out_base, (int)cat_elems);
        } else {
            __half* cat_dst = out_base + (size_t)Cout * OH * OW;
            copy_channels_k<<<(cat_elems+255)/256, 256, 0, stream>>>(cat_in, cat_dst, (int)cat_elems);
        }
    } else if (!mInputMode && mHasCat) {
        const __half* cat_in = (const __half*)inputs[1];
        size_t cat_elems = (size_t)mCatCh * OH * OW;
        if (mCatAtFront) {
            copy_channels_k<<<(cat_elems+255)/256, 256, 0, stream>>>(cat_in, out_base, (int)cat_elems);
            out_base += cat_elems;
        }
        pixel_shuffle_bias_2x_k<<<ps_grid, 256, 0, stream>>>(conv_out, bias, out_base, mOutCh, H, W);
        if (!mCatAtFront) {
            __half* cat_dst = out_base + (size_t)mOutCh * OH * OW;
            copy_channels_k<<<(cat_elems+255)/256, 256, 0, stream>>>(cat_in, cat_dst, (int)cat_elems);
        }
    }

    return cudaGetLastError();
}

// ===================== Creator =====================
class SubpelCreator : public IPluginCreatorV3One {
public:
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return mNs.c_str(); }
    void setPluginNamespace(char const* ns) noexcept { mNs = ns ? ns : ""; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }
    IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const* fc,
                            TensorRTPhase) noexcept override;
private:
    std::string mNs;
    PluginFieldCollection mFC{0, nullptr};
};

IPluginV3* SubpelCreator::createPlugin(AsciiChar const*, PluginFieldCollection const* fc,
                                       TensorRTPhase) noexcept {
    auto* p = new SubpelConvPlugin();
    int inCh = 0, outCh = 0, kSize = 1, pad = 0, catCh = 0;
    int hasCat = 0, catFront = 1;
    if (fc) for (int32_t i = 0; i < fc->nbFields; ++i) {
        auto const& f = fc->fields[i];
        if (std::strcmp(f.name, "weight")==0) p->setWeight(0, f.data, (size_t)f.length * sizeof(float));
        if (std::strcmp(f.name, "bias")==0) p->setWeight(1, f.data, (size_t)f.length * sizeof(float));
        if (std::strcmp(f.name, "in_ch")==0) inCh = (int)*(const float*)f.data;
        if (std::strcmp(f.name, "out_ch")==0) outCh = (int)*(const float*)f.data;
        if (std::strcmp(f.name, "kernel_size")==0) kSize = (int)*(const float*)f.data;
        if (std::strcmp(f.name, "padding")==0) pad = (int)*(const float*)f.data;
        if (std::strcmp(f.name, "has_cat")==0) hasCat = (int)*(const float*)f.data;
        if (std::strcmp(f.name, "cat_at_front")==0) catFront = (int)*(const float*)f.data;
        if (std::strcmp(f.name, "cat_ch")==0) catCh = (int)*(const float*)f.data;
    }
    p->setConfig(inCh, outCh, kSize, pad, hasCat != 0, catFront != 0, catCh);
    if (fc && fc->nbFields == 0) p->setInputMode();
    return p;
}

static SubpelCreator gCreator;
static bool gRegistered = [] {
    IPluginRegistry* reg = getBuilderPluginRegistry(nvinfer1::EngineCapability::kSTANDARD);
    if (reg) reg->registerCreator(gCreator, "");
    IPluginRegistry* greg = getPluginRegistry();
    if (greg) greg->registerCreator(gCreator, "");
    return reg != nullptr || greg != nullptr;
}();

}  // namespace dcvc_subpel

extern "C" void dcvc_register_subpel_plugin() {
    (void)dcvc_subpel::gRegistered;
}
