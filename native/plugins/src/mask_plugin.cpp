// ProcessWithMask / BuildIndex TensorRT plugin scaffolding.
#ifdef DCVC_RT_HAS_TENSORRT

#include <NvInfer.h>
#include <string>

using namespace nvinfer1;

namespace {

const char* kMaskPluginName = "DcvcProcessWithMask";
const char* kMaskPluginVersion = "1";

class ProcessWithMaskPlugin : public IPluginV3,
                              public IPluginV3OneCore,
                              public IPluginV3OneBuild,
                              public IPluginV3OneRuntime {
public:
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        if (type == PluginCapabilityType::kCORE) return (IPluginV3OneCore*)this;
        if (type == PluginCapabilityType::kBUILD) return (IPluginV3OneBuild*)this;
        if (type == PluginCapabilityType::kRUNTIME) return (IPluginV3OneRuntime*)this;
        return nullptr;
    }
    IPluginV3* clone() noexcept override { return new ProcessWithMaskPlugin(); }
    char const* getPluginName() const noexcept override { return kMaskPluginName; }
    char const* getPluginVersion() const noexcept override { return kMaskPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return mNs.c_str(); }
    void setPluginNamespace(char const* ns) noexcept { mNs = ns ? ns : ""; }
    int32_t getNbOutputs() const noexcept override { return 4; /* y_res,y_q,y_hat,scales_hat */ }
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                            int32_t) noexcept override
    {
        return 0;
    }
    bool supportsFormatCombination(int32_t, DynamicPluginTensorDesc const* inOut, int32_t,
                                   int32_t) noexcept override
    {
        return inOut[0].desc.type == DataType::kHALF;
    }
    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes,
                               int32_t) const noexcept override
    {
        for (int i = 0; i < nbOutputs; i++) outputTypes[i] = inputTypes[0];
        return 0;
    }
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t, DimsExprs const*, int32_t,
                            DimsExprs* outputs, int32_t nbOutputs, IExprBuilder&) noexcept override
    {
        for (int i = 0; i < nbOutputs; i++) outputs[i] = inputs[0];
        return 0;
    }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                            int32_t) const noexcept override
    {
        return 0;
    }
    int32_t enqueue(PluginTensorDesc const*, PluginTensorDesc const*, void const* const*,
                    void* const*, void*, cudaStream_t) noexcept override
    {
        /* Port process_with_mask_cuda from kernel.cu */
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
    std::string mNs;
    PluginFieldCollection mFC{0, nullptr};
};

class ProcessWithMaskPluginCreator : public IPluginCreatorV3One {
public:
    char const* getPluginName() const noexcept override { return kMaskPluginName; }
    char const* getPluginVersion() const noexcept override { return kMaskPluginVersion; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const*,
                            TensorRTPhase) noexcept override
    {
        return new ProcessWithMaskPlugin();
    }
    char const* getPluginNamespace() const noexcept override { return mNs.c_str(); }
    void setPluginNamespace(char const* ns) noexcept { mNs = ns ? ns : ""; }

private:
    std::string mNs;
    PluginFieldCollection mFC{0, nullptr};
};

static ProcessWithMaskPluginCreator gMaskCreator;

}  // namespace

extern "C" int dcvc_register_mask_plugin(void* registry)
{
    auto* reg = static_cast<IPluginRegistry*>(registry);
    return reg && reg->registerCreator(gMaskCreator, "") ? 0 : -1;
}

#endif
