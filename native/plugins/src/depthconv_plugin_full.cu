// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
//
// DcvcDepthConv — fused TensorRT IPluginV3 for DepthConvBlock.
//
// Replaces 8+ standard layers with one plugin call. Uses cuBLAS GEMM for
// 1x1 pointwise convs and custom CUDA kernels for WSiLU, depthwise 3x3 conv,
// WSiLUChunkAdd, and residuals.
//
// Computation (matches forward_torch in src/layers/layers.py):
//   if adaptor: x = conv1x1(x, adaptor_w, adaptor_b)
//   dc_out  = (conv1x1→wsilu→dwconv3x3→conv1x1)(x) + x     // residual
//   out     = (conv1x1→wsilu_chunk_add→conv1x1)(dc_out) + dc_out // residual
//   if shortcut: out += x

#include <NvInfer.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace dcvc_depthconv {

constexpr char const* kName = "DcvcDepthConv";
constexpr char const* kVersion = "1";

// ===================== CUDA kernels =====================

// Fused: bias add + WSiLU.  out[i] = (in[i] + bias[ch]) * sigmoid(4*(in[i]+bias[ch]))
__global__ void bias_wsilu_k(const __half* in, const __half* bias,
                             __half* out, int C, int HW) {
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= HW) return;
    float v = __half2float(in[c * HW + hw]) + __half2float(bias[c]);
    float s = 1.0f / (1.0f + __expf(-4.0f * v));
    out[c * HW + hw] = __float2half_rn(v * s);
}

// Fused: bias add + WSiLU + chunk_add.  in has 4C*HW, out has 2C*HW.
// out[i] = wsilu(in[i] + bias[ch]) + wsilu(in[i+2C*HW] + bias[ch+2C])
__global__ void bias_wsilu_chunk_k(const __half* in, const __half* bias,
                                   __half* out, int C2, int HW) {
    int c = blockIdx.y;       // 0..2C-1
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= HW || c >= C2) return;
    int C4 = C2 * 2;
    float a = __half2float(in[c * HW + hw]) + __half2float(bias[c]);
    a = a / (1.0f + __expf(-4.0f * a));
    float b = __half2float(in[(c + C2) * HW + hw]) + __half2float(bias[c + C2]);
    b = b / (1.0f + __expf(-4.0f * b));
    out[c * HW + hw] = __float2half_rn(a + b);
}

// Depthwise 3x3 conv + bias.  weight[C*9], bias[C].
__global__ void dwconv3x3_k(const __half* in, const __half* weight,
                            const __half* bias, __half* out,
                            int C, int H, int W) {
    int c = blockIdx.z;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= H * W) return;
    int h = hw / W, w = hw % W;
    const __half* w9 = weight + c * 9;
    float acc = __half2float(bias[c]);
    for (int dy = -1; dy <= 1; dy++) {
        int ih = h + dy;
        if (ih < 0 || ih >= H) continue;
        for (int dw = -1; dw <= 1; dw++) {
            int iw = w + dw;
            if (iw < 0 || iw >= W) continue;
            acc += __half2float(w9[(dy+1)*3 + (dw+1)]) *
                   __half2float(in[c * H * W + ih * W + iw]);
        }
    }
    out[c * H * W + hw] = __float2half_rn(acc);
}

// out = conv_out + bias + residual.  (dc residual & ffn residual)
__global__ void bias_resid_add_k(const __half* conv_out, const __half* bias,
                                 const __half* residual, __half* out, int C, int HW) {
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= HW) return;
    int idx = c * HW + hw;
    out[idx] = __float2half_rn(__half2float(conv_out[idx]) + __half2float(bias[c]) +
                               __half2float(residual[idx]));
}

// Pre-fill: out = bias + residual (for beta=1 GEMM)
__global__ void bias_resid_prefill_k(const __half* bias, const __half* residual,
                                     __half* out, int C, int HW) {
    int c = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= HW) return;
    int idx = c * HW + hw;
    float r = residual ? __half2float(residual[idx]) : 0.0f;
    out[idx] = __float2half_rn(__half2float(bias[c]) + r);
}

// In-place: out += x_in (shortcut)
__global__ void shortcut_add_k(__half* out, const __half* x_in, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = __float2half_rn(__half2float(out[i]) + __half2float(x_in[i]));
}

// ===================== cuBLAS =====================
static cublasHandle_t get_cublas() {
    static cublasHandle_t h = nullptr;
    if (!h) cublasCreate(&h);
    return h;
}

// pointwise conv: out[Co,HW] = weight[Co,Ci] @ in[Ci,HW]  (beta=0: fresh, beta=1: accumulate)
static void pw_conv(cublasHandle_t h, const __half* in, const __half* weight,
                    __half* out, int Ci, int Co, int HW, float beta) {
    float alpha = 1.0f;
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
struct WeightBuf {
    std::vector<uint8_t> host;   // float32 host data
    __half* dev = nullptr;       // float16 device data
    int count = 0;               // number of float32 elements
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
class DepthConvPlugin : public IPluginV3,
                        public IPluginV3OneCore,
                        public IPluginV3OneBuild,
                        public IPluginV3OneRuntime {
public:
    DepthConvPlugin() = default;

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        if (type == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return nullptr;
    }
    void finalizeConfig() {
        // Derive out_ch from dc0_b element count, in_ch from dc0_w / dc0_b
        if (mOutCh == 0 && mW[1].count > 0) mOutCh = mW[1].count;
        if (mInCh == 0 && mW[0].count > 0 && mOutCh > 0) mInCh = mW[0].count / mOutCh;
        if (mW[10].count > 0) mHasAdaptor = true;
    }
    IPluginV3* clone() noexcept override {
        auto* p = new DepthConvPlugin(*this);
        for (auto& w : p->mW) w.dev = nullptr;  // fresh device allocs
        return p;
    }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return mNs.c_str(); }
    void setPluginNamespace(char const* ns) noexcept { mNs = ns ? ns : ""; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbIn,
                            DynamicPluginTensorDesc const* out, int32_t nbOut) noexcept override {
        mNbInputs = nbIn;
        if (nbIn > 1) { mInputMode = true; mHasAdaptor = (nbIn > 11); }
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
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbI, DimsExprs const*, int32_t,
                            DimsExprs* outputs, int32_t nbO, IExprBuilder& eb) noexcept override {
        if (nbI < 1 || nbO != 1) return -1;
        outputs[0] = inputs[0];
        // For adaptor blocks, override output channels; else keep input channels.
        if (mHasAdaptor && mOutCh > 0)
            outputs[0].d[1] = eb.constant(mOutCh);
        return 0;
    }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t nbIn,
                            DynamicPluginTensorDesc const*, int32_t) const noexcept override {
        int C = mOutCh;
        if (nbIn > 1) C = in[1].desc.dims.d[0];  // input mode: dc0_w[0]
        if (C == 0 && nbIn > 0) C = in[0].max.d[1] > 0 ? in[0].max.d[1] : in[0].desc.dims.d[1];
        if (C == 0) C = 512;
        int H = 0, W = 0;
        if (nbIn > 0) {
            H = in[0].max.d[2] > 0 ? in[0].max.d[2] : in[0].desc.dims.d[2];
            W = in[0].max.d[3] > 0 ? in[0].max.d[3] : in[0].desc.dims.d[3];
        }
        if (H <= 0) H = 512;
        if (W <= 0) W = 512;
        return (size_t)10 * C * H * W * sizeof(__half) + 4096;
    }
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*,
                          int32_t) noexcept override { return 0; }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    void setParam(int idx, const void* data, size_t nbytes) {
        if (idx >= 0 && idx < 12) mW[idx].store(data, nbytes);
    }
    void setConfig(int inCh, int outCh, bool hasAdaptor, bool shortcut) {
        mInCh = inCh; mOutCh = outCh; mHasAdaptor = hasAdaptor; mShortcut = shortcut;
    }
    void setInputMode() { mInputMode = true; }
    void forceShortcut() { mShortcut = true; }

private:
    std::string mNs;
    int mInCh = 0, mOutCh = 0, mNbInputs = 1;
    bool mHasAdaptor = false, mShortcut = false, mInputMode = false;
    // 0=dc0_w 1=dc0_b 2=dc_dw 3=dc_db 4=dc3_w 5=dc3_b
    // 6=ffn0_w 7=ffn0_b 8=ffn2_w 9=ffn2_b 10=ad_w 11=ad_b
    WeightBuf mW[12];
};

PluginFieldCollection const* DepthConvPlugin::getFieldsToSerialize() noexcept {
    static std::vector<PluginField> fields;
    static PluginFieldCollection fc;
    static float sCfgF[4];
    fields.clear();
    const char* names[] = {"dc0_w","dc0_b","dc_dw","dc_db","dc3_w","dc3_b",
                           "ffn0_w","ffn0_b","ffn2_w","ffn2_b","ad_w","ad_b"};
    for (int i = 0; i < 12; i++) {
        if (mW[i].count > 0)
            fields.push_back({names[i], mW[i].host.data(), PluginFieldType::kFLOAT32,
                              (int32_t)(mW[i].count)});
    }
    sCfgF[0] = (float)mInCh; sCfgF[1] = (float)mOutCh; sCfgF[2] = mHasAdaptor?1.0f:0.0f; sCfgF[3] = mShortcut?1.0f:0.0f;
    fields.push_back({"in_ch", &sCfgF[0], PluginFieldType::kFLOAT32, 1});
    fields.push_back({"out_ch", &sCfgF[1], PluginFieldType::kFLOAT32, 1});
    fields.push_back({"has_adaptor", &sCfgF[2], PluginFieldType::kFLOAT32, 1});
    fields.push_back({"shortcut", &sCfgF[3], PluginFieldType::kFLOAT32, 1});
    fc.nbFields = (int32_t)fields.size();
    fc.fields = fields.data();
    return &fc;
}

int32_t DepthConvPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                                 void const* const* inputs, void* const* outputs, void* workspace,
                                 cudaStream_t stream) noexcept {
    cublasSetStream(get_cublas(), stream);
    int H = inputDesc[0].dims.d[2], W = inputDesc[0].dims.d[3];
    int HW = H * W;
    int Ci_in = inputDesc[0].dims.d[1];

    // Resolve weight pointers and config (attribute mode or input mode)
    const __half *dc0_w, *dc0_b, *dc_dw, *dc_db, *dc3_w, *dc3_b;
    const __half *ffn0_w, *ffn0_b, *ffn2_w, *ffn2_b;
    const __half *ad_w = nullptr, *ad_b = nullptr;
    int C;
    bool hasAd;

    if (mInputMode) {
        int wi = 1;
        dc0_w = (const __half*)inputs[wi++]; dc0_b = (const __half*)inputs[wi++];
        dc_dw = (const __half*)inputs[wi++]; dc_db = (const __half*)inputs[wi++];
        dc3_w = (const __half*)inputs[wi++]; dc3_b = (const __half*)inputs[wi++];
        ffn0_w = (const __half*)inputs[wi++]; ffn0_b = (const __half*)inputs[wi++];
        ffn2_w = (const __half*)inputs[wi++]; ffn2_b = (const __half*)inputs[wi++];
        hasAd = (mNbInputs > 11);
        if (hasAd) {
            ad_w = (const __half*)inputs[wi++]; ad_b = (const __half*)inputs[wi++];
        }
        C = inputDesc[1].dims.d[0];  // dc0_w shape[0] = out_ch
    } else {
        dc0_w = mW[0].dptr(); dc0_b = mW[1].dptr();
        dc_dw = mW[2].dptr(); dc_db = mW[3].dptr();
        dc3_w = mW[4].dptr(); dc3_b = mW[5].dptr();
        ffn0_w = mW[6].dptr(); ffn0_b = mW[7].dptr();
        ffn2_w = mW[8].dptr(); ffn2_b = mW[9].dptr();
        ad_w = mW[10].dptr(); ad_b = mW[11].dptr();
        C = outputDesc[0].dims.d[1];
        if (C <= 0) C = mOutCh;
        if (C <= 0) C = inputDesc[0].dims.d[1];
        hasAd = mHasAdaptor;
    }

    dim3 grid_hw((HW + 255) / 256, C), grid_hw_C2((HW + 255) / 256, 2 * C);
    dim3 dw_grid((HW + 255) / 256, 1, C);
    int block = 256;

    __half* ws = (__half*)workspace;
    __half* tmp_dc  = ws;   ws += C * HW;
    __half* tmp_dw  = ws;   ws += C * HW;
    __half* dc_out  = ws;   ws += C * HW;
    __half* tmp_ffn = ws;   ws += 4 * C * HW;
    __half* tmp_f2  = ws;   ws += 2 * C * HW;
    __half* x_ad    = hasAd ? ws : (__half*)inputs[0];

    const __half* x_in = (const __half*)inputs[0];
    __half* out_buf = (__half*)outputs[0];

    if (hasAd) {
        bias_resid_prefill_k<<<grid_hw, block, 0, stream>>>(ad_b, nullptr, x_ad, C, HW);
        pw_conv(get_cublas(), x_in, ad_w, x_ad, Ci_in, C, HW, 1.0f);
        x_in = x_ad;
    }

    pw_conv(get_cublas(), x_in, dc0_w, tmp_dc, C, C, HW, 0.0f);
    bias_wsilu_k<<<grid_hw, block, 0, stream>>>(tmp_dc, dc0_b, tmp_dc, C, HW);
    dwconv3x3_k<<<dw_grid, block, 0, stream>>>(tmp_dc, dc_dw, dc_db, tmp_dw, C, H, W);
    pw_conv(get_cublas(), tmp_dw, dc3_w, tmp_dc, C, C, HW, 0.0f);
    bias_resid_add_k<<<grid_hw, block, 0, stream>>>(tmp_dc, dc3_b, x_in, dc_out, C, HW);

    pw_conv(get_cublas(), dc_out, ffn0_w, tmp_ffn, C, 4 * C, HW, 0.0f);
    bias_wsilu_chunk_k<<<grid_hw_C2, block, 0, stream>>>(tmp_ffn, ffn0_b, tmp_f2, 2 * C, HW);
    bias_resid_prefill_k<<<grid_hw, block, 0, stream>>>(ffn2_b, dc_out, out_buf, C, HW);
    pw_conv(get_cublas(), tmp_f2, ffn2_w, out_buf, 2 * C, C, HW, 1.0f);

    if (mShortcut)
        shortcut_add_k<<<(C*HW+255)/256, block, 0, stream>>>(out_buf, x_in, C * HW);

    return cudaGetLastError();
}

// ===================== Creator =====================
class DepthConvCreator : public IPluginCreatorV3One {
public:
    DepthConvCreator(const char* name = kName) : mName(name) {}
    char const* getPluginName() const noexcept override { return mName.c_str(); }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return mNs.c_str(); }
    void setPluginNamespace(char const* ns) noexcept { mNs = ns ? ns : ""; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }
    IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const* fc,
                            TensorRTPhase) noexcept override;
private:
    std::string mNs, mName;
    PluginFieldCollection mFC{0, nullptr};
};

IPluginV3* DepthConvCreator::createPlugin(AsciiChar const*, PluginFieldCollection const* fc,
                                          TensorRTPhase) noexcept {
    auto* p = new DepthConvPlugin();
    int inCh = 0, outCh = 0, hasAd = 0, sc = 0;
    struct { const char* n; int i; } wm[] = {
        {"dc0_w",0},{"dc0_b",1},{"dc_dw",2},{"dc_db",3},{"dc3_w",4},{"dc3_b",5},
        {"ffn0_w",6},{"ffn0_b",7},{"ffn2_w",8},{"ffn2_b",9},{"ad_w",10},{"ad_b",11}};
    if (fc) for (int32_t i = 0; i < fc->nbFields; ++i) {
        auto const& f = fc->fields[i];
        for (auto& m : wm)
            if (std::strcmp(f.name, m.n) == 0)
                p->setParam(m.i, f.data, (size_t)f.length * sizeof(float));
        if (std::strcmp(f.name, "in_ch")==0) inCh = (int)*(const float*)f.data;
        if (std::strcmp(f.name, "out_ch")==0) outCh = (int)*(const float*)f.data;
        if (std::strcmp(f.name, "has_adaptor")==0) hasAd = (int)*(const float*)f.data;
        if (std::strcmp(f.name, "shortcut")==0) sc = (int)*(const float*)f.data;
    }
    // Derive config from weight sizes (more robust than attributes):
    // dc0_b has out_ch elements, dc0_w has out_ch * in_ch elements.
    int derived_out = outCh, derived_in = inCh;
    // Access weight counts via a helper — setParam stores into mW[idx].count
    // We already set params above; re-derive if fields failed.
    p->setConfig(inCh > 0 ? inCh : derived_in, outCh > 0 ? outCh : derived_out, hasAd != 0, sc != 0);
    if (fc && fc->nbFields == 0) {
        p->setInputMode();
        if (mName == "DcvcDepthConvSC") p->forceShortcut();
    }
    p->finalizeConfig();
    return p;
}

static DepthConvCreator gCreator;
static DepthConvCreator gCreatorSC("DcvcDepthConvSC");
static bool gRegistered = [] {
    IPluginRegistry* reg = getBuilderPluginRegistry(nvinfer1::EngineCapability::kSTANDARD);
    if (reg) {
        reg->registerCreator(gCreator, "");
        reg->registerCreator(gCreatorSC, "");
    }
    IPluginRegistry* greg = getPluginRegistry();
    if (greg) {
        greg->registerCreator(gCreator, "");
        greg->registerCreator(gCreatorSC, "");
    }
    return true;
}();

// ===================== Direct test entry point =====================
// Bypasses TRT entirely — receives raw device pointers for weights and input,
// runs the full DepthConvBlock computation, writes to output.
// Used for isolated parity testing.
#include <cstdio>
extern "C" int dcvc_depthconv_test(
    const __half* x_in,       // [1, Ci, H, W]
    __half* out,              // [1, Co, H, W]
    int Ci, int Co, int H, int W,
    int has_adaptor, int shortcut,
    const __half* dc0_w, const __half* dc0_b,
    const __half* dc_dw, const __half* dc_db,
    const __half* dc3_w, const __half* dc3_b,
    const __half* ffn0_w, const __half* ffn0_b,
    const __half* ffn2_w, const __half* ffn2_b,
    const __half* ad_w, const __half* ad_b)
{
    cudaStream_t stream = 0;
    cublasSetStream(get_cublas(), stream);
    int HW = H * W, C = Co;
    dim3 grid_hw((HW + 255) / 256, C);
    dim3 grid_hw_C2((HW + 255) / 256, 2 * C);
    dim3 dw_grid((HW + 255) / 256, 1, C);
    int block = 256;

    // workspace
    size_t ws_elems = C*HW + C*HW + C*HW + 4*C*HW + 2*C*HW + (has_adaptor ? C*HW : 0);
    __half* ws;
    cudaMalloc(&ws, ws_elems * sizeof(__half));
    __half* tmp_dc = ws;   __half* p = ws + C*HW;
    __half* tmp_dw = p;    p += C*HW;
    __half* dc_out = p;    p += C*HW;
    __half* tmp_ffn = p;   p += 4*C*HW;
    __half* tmp_f2 = p;    p += 2*C*HW;
    __half* x_ad = has_adaptor ? p : (__half*)x_in;

    const __half* x = (const __half*)x_in;
    auto ck = [&](int step) -> int {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { fprintf(stderr, "FAIL step %d: %s\n", step, cudaGetErrorString(e)); return step; }
        return 0;
    };
    int step = 0;

    if (has_adaptor) {
        bias_resid_prefill_k<<<grid_hw, block, 0, stream>>>(ad_b, nullptr, x_ad, C, HW);
        if ((step=ck(1))) { cudaFree(ws); return step; }
        pw_conv(get_cublas(), x, ad_w, x_ad, Ci, C, HW, 1.0f);
        if ((step=ck(2))) { cudaFree(ws); return step; }
        x = x_ad;
    }

    pw_conv(get_cublas(), x, dc0_w, tmp_dc, C, C, HW, 0.0f);
    if ((step=ck(3))) { cudaFree(ws); return step; }
    bias_wsilu_k<<<grid_hw, block, 0, stream>>>(tmp_dc, dc0_b, tmp_dc, C, HW);
    if ((step=ck(4))) { cudaFree(ws); return step; }
    dwconv3x3_k<<<dw_grid, block, 0, stream>>>(tmp_dc, dc_dw, dc_db, tmp_dw, C, H, W);
    if ((step=ck(5))) { cudaFree(ws); return step; }
    pw_conv(get_cublas(), tmp_dw, dc3_w, tmp_dc, C, C, HW, 0.0f);
    if ((step=ck(6))) { cudaFree(ws); return step; }
    bias_resid_add_k<<<grid_hw, block, 0, stream>>>(tmp_dc, dc3_b, x, dc_out, C, HW);
    if ((step=ck(7))) { cudaFree(ws); return step; }

    pw_conv(get_cublas(), dc_out, ffn0_w, tmp_ffn, C, 4*C, HW, 0.0f);
    if ((step=ck(8))) { cudaFree(ws); return step; }
    bias_wsilu_chunk_k<<<grid_hw_C2, block, 0, stream>>>(tmp_ffn, ffn0_b, tmp_f2, 2*C, HW);
    if ((step=ck(9))) { cudaFree(ws); return step; }
    bias_resid_prefill_k<<<grid_hw, block, 0, stream>>>(ffn2_b, dc_out, out, C, HW);
    if ((step=ck(10))) { cudaFree(ws); return step; }
    pw_conv(get_cublas(), tmp_f2, ffn2_w, out, 2*C, C, HW, 1.0f);
    if ((step=ck(11))) { cudaFree(ws); return step; }

    if (shortcut)
        shortcut_add_k<<<(C*HW+255)/256, block, 0, stream>>>(out, x, C*HW);

    cudaError_t err = cudaDeviceSynchronize();
    cudaFree(ws);
    if (err != cudaSuccess) fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    return (int)err;
}
}  // namespace dcvc_depthconv

extern "C" void dcvc_register_depthconv_plugin() {
    (void)dcvc_depthconv::gRegistered;
}
