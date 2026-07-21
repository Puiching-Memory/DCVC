// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
//
// Self-contained TensorRT IPluginV3 for DCVC process_with_mask.
// Closes the loop: ONNX custom op node "DcvcProcessWithMask" -> plugin ->
// ATen-free CUDA kernel -> parity vs PyTorch.
//
// Build into a shared lib, dlopen it to auto-register the creator with the
// global builder plugin registry, then OnnxParser resolves the custom node.

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace dcvc_process_mask {
constexpr char const* kName = "DcvcProcessWithMask";
constexpr char const* kVersion = "1";

// ---- ATen-free CUDA kernel (port of kernel.cu process_with_mask) ----
__global__ void process_mask_hat_kernel(__half const* __restrict__ y,
                                        __half const* __restrict__ scales,
                                        __half const* __restrict__ means,
                                        __half const* __restrict__ mask,
                                        __half* __restrict__ y_hat,
                                        float force_zero_thres, int N)
{
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

class ProcessWithMaskPlugin : public IPluginV3,
                              public IPluginV3OneCore,
                              public IPluginV3OneBuild,
                              public IPluginV3OneRuntime {
public:
    ProcessWithMaskPlugin() = default;

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        if (type == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new ProcessWithMaskPlugin(*this); }

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

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*,
                            int32_t, DimsExprs* outputs, int32_t nbOutputs,
                            IExprBuilder&) noexcept override {
        if (nbInputs < 4 || nbOutputs != 1) return -1;
        outputs[0] = inputs[0];
        return 0;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                            int32_t) const noexcept override {
        return 0;
    }

    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*,
                          int32_t) noexcept override {
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override {
        (void)workspace;
        int32_t const n = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1] *
                          inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];
        int block = 256;
        int grid = (n + block - 1) / block;
        process_mask_hat_kernel<<<grid, block, 0, stream>>>(
            (__half const*)inputs[0], (__half const*)inputs[1], (__half const*)inputs[2],
            (__half const*)inputs[3], (__half*)outputs[0], mForceZeroThres, n);
        return cudaGetLastError();
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        static float thresBuf;
        static PluginField field;
        thresBuf = mForceZeroThres;
        field = PluginField{"force_zero_thres", &thresBuf, PluginFieldType::kFLOAT32, 1};
        static PluginFieldCollection fc;
        fc.nbFields = 1;
        fc.fields = &field;
        return &fc;
    }

    void setForceZeroThres(float v) noexcept { mForceZeroThres = v; }

private:
    std::string mNs;
    float mForceZeroThres = -1.0f;
};

class ProcessWithMaskCreator : public IPluginCreatorV3One {
public:
    ProcessWithMaskCreator() {
        mField.name = "force_zero_thres";
        mField.data = &mDefaultThres;
        mField.type = PluginFieldType::kFLOAT32;
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
        auto* p = new ProcessWithMaskPlugin();
        if (fc) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                auto const& f = fc->fields[i];
                if (std::strcmp(f.name, "force_zero_thres") == 0 && f.data) {
                    p->setForceZeroThres(*static_cast<float const*>(f.data));
                }
            }
        }
        return p;
    }

private:
    std::string mNs;
    float mDefaultThres = -1.0f;
    PluginField mField;
    PluginFieldCollection mFC;
};

static ProcessWithMaskCreator gCreator;
static bool gRegistered = [] {
    IPluginRegistry* reg = getBuilderPluginRegistry(nvinfer1::EngineCapability::kSTANDARD);
    if (reg) reg->registerCreator(gCreator, "");
    return reg != nullptr;
}();
}  // namespace

extern "C" void dcvc_register_process_mask_plugin() {
    // Static registrar (gRegistered) already ran at load; this symbol keeps
    // the translation unit live when linked into a larger shared library.
    (void)dcvc_process_mask::gRegistered;
}
