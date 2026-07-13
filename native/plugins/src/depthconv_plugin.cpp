// TensorRT IPluginV3 scaffolding for DepthConvBlock fusion.
// Built only when DCVC_RT_HAS_TENSORRT=ON.
// Port of DepthConvProxy from src/layers/extensions/inference.

#ifdef DCVC_RT_HAS_TENSORRT

#include "dcvc_plugins.h"

#include <NvInfer.h>
#include <cstring>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace {

const char* kDepthConvPluginName = "DcvcDepthConv";
const char* kDepthConvPluginVersion = "1";

class DepthConvPlugin : public IPluginV3,
                        public IPluginV3OneCore,
                        public IPluginV3OneBuild,
                        public IPluginV3OneRuntime {
public:
    DepthConvPlugin() = default;
    explicit DepthConvPlugin(const void* data, size_t length)
    {
        (void)data;
        (void)length;
    }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        if (type == PluginCapabilityType::kCORE) return static_cast<IPluginV3OneCore*>(this);
        if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return nullptr;
    }

    IPluginV3* clone() noexcept override { return new DepthConvPlugin(); }

    char const* getPluginName() const noexcept override { return kDepthConvPluginName; }
    char const* getPluginVersion() const noexcept override { return kDepthConvPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
    void setPluginNamespace(char const* ns) noexcept { mNamespace = ns ? ns : ""; }

    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                            int32_t) noexcept override
    {
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override
    {
        (void)nbOutputs;
        if (pos < 0 || pos >= nbInputs + 1) return false;
        return inOut[pos].desc.type == DataType::kHALF &&
               inOut[pos].desc.format == TensorFormat::kLINEAR;
    }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes,
                               int32_t nbInputs) const noexcept override
    {
        (void)nbInputs;
        if (nbOutputs != 1) return -1;
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t,
                            DimsExprs* outputs, int32_t nbOutputs, IExprBuilder&) noexcept override
    {
        if (nbInputs < 1 || nbOutputs != 1) return -1;
        outputs[0] = inputs[0];
        return 0;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                            int32_t) const noexcept override
    {
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                    void const* const* inputs, void* const* outputs, void* workspace,
                    cudaStream_t stream) noexcept override
    {
        (void)inputDesc;
        (void)outputDesc;
        (void)inputs;
        (void)outputs;
        (void)workspace;
        (void)stream;
        /* Launch fused DepthConv CUDA kernel (ported from kernel.cu). */
        return 0;
    }

    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*,
                          int32_t) noexcept override
    {
        return 0;
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFC; }

private:
    std::string mNamespace;
    PluginFieldCollection mFC{0, nullptr};
};

class DepthConvPluginCreator : public IPluginCreatorV3One {
public:
    DepthConvPluginCreator()
    {
        mFC.nbFields = 0;
        mFC.fields = nullptr;
    }
    char const* getPluginName() const noexcept override { return kDepthConvPluginName; }
    char const* getPluginVersion() const noexcept override { return kDepthConvPluginVersion; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const*,
                            TensorRTPhase) noexcept override
    {
        return new DepthConvPlugin();
    }
    char const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
    void setPluginNamespace(char const* ns) noexcept { mNamespace = ns ? ns : ""; }

private:
    std::string mNamespace;
    PluginFieldCollection mFC;
};

}  // namespace

static DepthConvPluginCreator gDepthConvCreator;

extern "C" int dcvc_register_depthconv_plugin(void* registry)
{
    auto* reg = static_cast<IPluginRegistry*>(registry);
    if (!reg) return -1;
    return reg->registerCreator(gDepthConvCreator, "") ? 0 : -1;
}

#endif /* DCVC_RT_HAS_TENSORRT */
