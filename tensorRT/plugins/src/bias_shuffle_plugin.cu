// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
//
// TensorRT IPluginV3 for DCVC bias_pixel_shuffle_8 — the synthesis endpoint.
// Engine-side op: x[1,192,H,W] + bias[192] → pixel_shuffle(8) → clamp[0,1]
//                 → out[1,3,H*8,W*8]
//
// Reuses the ATen-free kernel pattern proven in dcvc_kernels.cu.
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>
#include <string>

using namespace nvinfer1;

namespace dcvc_bias_shuffle {
constexpr char const* kName = "DcvcBiasPixelShuffle8";
constexpr char const* kVersion = "1";

// ---- ATen-free kernel (port of bias_pixel_shuffle_8 from kernel.cu) ----
// NCHW contiguous: x[C,H,W], bias[C], out[3, H*8, W*8]
__global__ void bias_shuffle8_kernel(__half const* __restrict__ x,
                                     __half const* __restrict__ bias,
                                     __half* __restrict__ out,
                                     int H, int W, int C, int clamp_en) {
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * W;
    if (hw >= HW) return;
    int h = hw / W;
    int w = hw % W;
    int oH = H * 8, oW = W * 8;
    for (int i = 0; i < C; i++) {
        float v = __half2float(x[i * HW + hw]) + __half2float(bias[i]);
        if (clamp_en) { if (v < 0.f) v = 0.f; if (v > 1.f) v = 1.f; }
        int out_c = i / 64;
        int ry = (i % 64) / 8;
        int rx = i % 8;
        out[out_c * oH * oW + (h * 8 + ry) * oW + (w * 8 + rx)] = __float2half_rn(v);
    }
}

class BiasShufflePlugin : public IPluginV3,
                          public IPluginV3OneCore,
                          public IPluginV3OneBuild,
                          public IPluginV3OneRuntime {
public:
    BiasShufflePlugin() = default;
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        if (type == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new BiasShufflePlugin(*this); }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return mNs.c_str(); }
    void setPluginNamespace(char const* ns) noexcept { mNs = ns ? ns : ""; }

    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                            int32_t) noexcept override { return 0; }

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override {
        (void)nbOutputs;
        if (pos < 0 || pos >= nbInputs + 1) return false;
        return inOut[pos].desc.type == DataType::kHALF &&
               inOut[pos].desc.format == TensorFormat::kLINEAR;
    }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
                               DataType const* inputTypes, int32_t nbInputs) const noexcept override {
        (void)nbInputs;
        if (nbOutputs != 1) return -1;
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    // out shape = [1, inC/64, H*8, W*8] — pixel_shuffle(8) with C_in = 3*64
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*,
                            int32_t, DimsExprs* outputs, int32_t nbOutputs,
                            IExprBuilder& exprBuilder) noexcept override {
        if (nbInputs < 1 || nbOutputs != 1) return -1;
        auto const& in = inputs[0];
        outputs[0].nbDims = 4;
        outputs[0].d[0] = in.d[0];                                  // 1
        outputs[0].d[1] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV,
                                                *in.d[1], *exprBuilder.constant(64));  // C/64 = 3
        outputs[0].d[2] = exprBuilder.operation(DimensionOperation::kPROD,
                                                *in.d[2], *exprBuilder.constant(8));   // H*8
        outputs[0].d[3] = exprBuilder.operation(DimensionOperation::kPROD,
                                                *in.d[3], *exprBuilder.constant(8));   // W*8
        return 0;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                            int32_t) const noexcept override { return 0; }

    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*,
                          int32_t) noexcept override { return 0; }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        (void)workspace;
        int C = inputDesc[0].dims.d[1];
        int H = inputDesc[0].dims.d[2];
        int W = inputDesc[0].dims.d[3];
        int block = 256;
        int grid = (H * W + block - 1) / block;
        bias_shuffle8_kernel<<<grid, block, 0, stream>>>(
            (__half const*)inputs[0], (__half const*)inputs[1], (__half*)outputs[0],
            H, W, C, mClamp ? 1 : 0);
        return cudaGetLastError();
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        static int32_t clampBuf;
        static PluginField field;
        clampBuf = mClamp;
        field = PluginField{"clamp", &clampBuf, PluginFieldType::kINT32, 1};
        static PluginFieldCollection fc;
        fc.nbFields = 1;
        fc.fields = &field;
        return &fc;
    }

    void setClamp(int32_t v) noexcept { mClamp = v; }

private:
    std::string mNs;
    int32_t mClamp = 1;
};

class BiasShuffleCreator : public IPluginCreatorV3One {
public:
    BiasShuffleCreator() {
        mField.name = "clamp";
        mField.data = &mDefault;
        mField.type = PluginFieldType::kINT32;
        mField.length = 1;
        mFC.nbFields = 1;
        mFC.fields = &mField;
    }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return mNs.c_str(); }
    void setPluginNamespace(char const* ns) noexcept { mNs = ns ? ns : ""; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }

    IPluginV3* createPlugin(AsciiChar const*, PluginFieldCollection const* fc,
                            TensorRTPhase) noexcept override {
        auto* p = new BiasShufflePlugin();
        if (fc) for (int32_t i = 0; i < fc->nbFields; ++i) {
            auto const& f = fc->fields[i];
            if (std::strcmp(f.name, "clamp") == 0 && f.data)
                p->setClamp(*static_cast<int32_t const*>(f.data));
        }
        return p;
    }
private:
    std::string mNs;
    int32_t mDefault = 1;
    PluginField mField;
    PluginFieldCollection mFC;
};

static BiasShuffleCreator gCreator;
static bool gRegistered = [] {
    IPluginRegistry* reg = getBuilderPluginRegistry(nvinfer1::EngineCapability::kSTANDARD);
    if (reg) reg->registerCreator(gCreator, "");
    return reg != nullptr;
}();
}  // namespace dcvc_bias_shuffle

extern "C" void dcvc_register_bias_shuffle_plugin() {
    (void)dcvc_bias_shuffle::gRegistered;
}
