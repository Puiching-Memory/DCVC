// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
//
// C-callable TensorRT engine wrapper: load serialized engine, register
// custom plugins, set dynamic shapes, bind device pointers, and execute.

#include "trt_engine.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace {

// TRT logger — minimal implementation
class TrtLogger : public ILogger {
public:
    void log(Severity sev, char const* msg) noexcept override {
        if (sev <= Severity::kWARNING)
            fprintf(stderr, "[TRT] [%s] %s\n",
                    sev == Severity::kINTERNAL_ERROR ? "I" :
                    sev == Severity::kERROR ? "E" : "W", msg);
    }
};

TrtLogger& logger() {
    static TrtLogger inst;
    return inst;
}

} // namespace

struct DcvcTrtEngine {
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;

    ~DcvcTrtEngine() {
        if (context) delete context;
        if (engine) delete engine;
        if (runtime) delete runtime;
    }
};

extern "C" {

static void load_plugins(const char* plugin_dir) {
    const char* sonames[] = {
        "libdcvc_depthconv.so",
        "libdcvc_subpel.so",
        "libdcvc_bias_shuffle.so",
        "libdcvc_process_mask.so",
    };
    for (auto& so : sonames) {
        std::string path = std::string(plugin_dir) + "/" + so;
        if (void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL)) {
            // Plugin .so registers creators via static initializer.
            // Keep handle open for engine lifetime.
        }
    }
}

DcvcTrtEngine* dcvc_trt_engine_load(const char* engine_path, const char* plugin_dir, DcvcRtStatus* st) {
    // Load plugin shared libraries first so creators are registered.
    if (plugin_dir) load_plugins(plugin_dir);

    // Read serialized engine from file.
    FILE* f = fopen(engine_path, "rb");
    if (!f) {
        if (st) *st = DCVC_RT_ERR_IO;
        return nullptr;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize <= 0) { fclose(f); if (st) *st = DCVC_RT_ERR_IO; return nullptr; }

    std::vector<uint8_t> data(fsize);
    size_t got = fread(data.data(), 1, fsize, f);
    fclose(f);
    if ((long)got != fsize) { if (st) *st = DCVC_RT_ERR_IO; return nullptr; }

    // Deserialize.
    auto* eng = new DcvcTrtEngine();
    eng->runtime = createInferRuntime(logger());
    if (!eng->runtime) { delete eng; if (st) *st = DCVC_RT_ERR_TRT; return nullptr; }

    eng->engine = eng->runtime->deserializeCudaEngine(data.data(), data.size());
    if (!eng->engine) { delete eng; if (st) *st = DCVC_RT_ERR_TRT; return nullptr; }

    eng->context = eng->engine->createExecutionContext();
    if (!eng->context) { delete eng; if (st) *st = DCVC_RT_ERR_TRT; return nullptr; }

    if (st) *st = DCVC_RT_OK;
    return eng;
}

void dcvc_trt_engine_destroy(DcvcTrtEngine* eng) {
    delete eng;
}

int dcvc_trt_engine_num_io(const DcvcTrtEngine* eng) {
    if (!eng || !eng->engine) return 0;
    return eng->engine->getNbIOTensors();
}

const char* dcvc_trt_engine_tensor_name(const DcvcTrtEngine* eng, int idx) {
    if (!eng || !eng->engine || idx < 0 || idx >= eng->engine->getNbIOTensors())
        return nullptr;
    return eng->engine->getIOTensorName(idx);
}

int dcvc_trt_engine_is_input(const DcvcTrtEngine* eng, int idx) {
    if (!eng || !eng->engine || idx < 0 || idx >= eng->engine->getNbIOTensors())
        return -1;
    return eng->engine->getTensorIOMode(eng->engine->getIOTensorName(idx)) == TensorIOMode::kINPUT ? 1 : 0;
}

DcvcRtStatus dcvc_trt_engine_set_shape(DcvcTrtEngine* eng, const char* name, const int32_t* dims, int ndims) {
    if (!eng || !eng->context || !name || !dims || ndims <= 0)
        return DCVC_RT_ERR_INVALID_ARG;

    Dims64 d;
    d.nbDims = ndims;
    for (int i = 0; i < ndims; i++) d.d[i] = dims[i];
    if (!eng->context->setInputShape(name, d))
        return DCVC_RT_ERR_TRT;
    return DCVC_RT_OK;
}

DcvcRtStatus dcvc_trt_engine_get_shape(const DcvcTrtEngine* eng, const char* name,
                                       int32_t* dims_out, int* ndims_out, int max_dims) {
    if (!eng || !eng->context || !name || !dims_out || !ndims_out)
        return DCVC_RT_ERR_INVALID_ARG;

    Dims64 d = eng->context->getTensorShape(name);
    if (d.nbDims <= 0 || d.nbDims > max_dims)
        return DCVC_RT_ERR_TRT;
    *ndims_out = d.nbDims;
    for (int i = 0; i < d.nbDims; i++) dims_out[i] = d.d[i];
    return DCVC_RT_OK;
}

DcvcRtStatus dcvc_trt_engine_set_addr(DcvcTrtEngine* eng, const char* name, void* dev_ptr) {
    if (!eng || !eng->context || !name || !dev_ptr)
        return DCVC_RT_ERR_INVALID_ARG;
    if (!eng->context->setTensorAddress(name, dev_ptr))
        return DCVC_RT_ERR_TRT;
    return DCVC_RT_OK;
}

DcvcRtStatus dcvc_trt_engine_execute(DcvcTrtEngine* eng, void* cuda_stream) {
    if (!eng || !eng->context)
        return DCVC_RT_ERR_INVALID_ARG;

    cudaStream_t stream = cuda_stream ? (cudaStream_t)cuda_stream : 0;
    if (!eng->context->enqueueV3(stream))
        return DCVC_RT_ERR_TRT;
    return DCVC_RT_OK;
}

} // extern "C"
