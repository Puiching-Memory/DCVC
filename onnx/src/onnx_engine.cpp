#include "onnx_engine.h"
#include "fxp/ort_custom_ops.h"

#ifdef __MINGW32__
/* MinGW ships a minimal sal.h that omits several SAL annotations referenced
 * by the ONNX Runtime C API headers. _Frees_ptr_opt_ (used on every Release*
 * function pointer) is undefined there, so ORT_CLASS_RELEASE expands into an
 * ill-formed signature under GCC. These are pure no-op annotations here. */
#ifndef _Frees_ptr_opt_
#define _Frees_ptr_opt_
#endif
#ifndef _Out_z_
#define _Out_z_
#endif
#endif

#include <onnxruntime_c_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h> /* sysconf(_SC_NPROCESSORS_ONLN) */
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> /* MultiByteToWideChar: ORT CreateSession takes wchar_t* on Windows */
#endif

const char* dcvc_cpu_status_string(DcvcCpuStatus st)
{
    switch (st) {
    case DCVC_CPU_OK: return "ok";
    case DCVC_CPU_ERR_INVALID_ARG: return "invalid_arg";
    case DCVC_CPU_ERR_IO: return "io";
    case DCVC_CPU_ERR_ONNX: return "onnx";
    case DCVC_CPU_ERR_OOM: return "oom";
    case DCVC_CPU_ERR_UNSUPPORTED: return "unsupported";
    case DCVC_CPU_ERR_ENTROPY: return "entropy_sync";
    default: return "unknown";
    }
}

#ifdef _WIN32
/* ORT's CreateSession expects ORTCHAR_T* == wchar_t* on Windows. Convert the
 * caller's UTF-8 char path to a UTF-16 wstring. */
static std::wstring dcvc_to_wide(const char* s)
{
    if (!s) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    if (n <= 0) return std::wstring();
    std::wstring w((size_t)n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
    /* n includes the trailing NUL produced with cbMultiByte == -1. */
    if (!w.empty() && w.back() == L'\0') w.pop_back();
    return w;
}
#endif

struct DcvcCpuEngine {
    const OrtApi* api;
    OrtEnv* env;
    OrtSessionOptions* opts;
    OrtSession* session;
    OrtMemoryInfo* mem_info;
    OrtRunOptions* run_opts;
};

static void check_status(const OrtApi* api, OrtStatus* st, DcvcCpuStatus* out_st)
{
    if (st) {
        const char* msg = api->GetErrorMessage(st);
        fprintf(stderr, "ONNX error: %s\n", msg);
        api->ReleaseStatus(st);
        if (out_st) *out_st = DCVC_CPU_ERR_ONNX;
    }
}

/* Resolve the requested execution provider: an explicit use_gpu != 0 wins,
 * otherwise the DCVC_USE_GPU environment variable (0/1/2) is consulted. */
static int dcvc_requested_ep(int use_gpu)
{
    if (use_gpu != 0) return use_gpu;
    const char* env = getenv("DCVC_USE_GPU");
    return (env && env[0]) ? atoi(env) : 0;
}

int dcvc_cpu_default_use_gpu(void)
{
    return dcvc_requested_ep(0);
}

static const char* dcvc_ep_name(int ep)
{
    return ep == 2 ? "TensorrtExecutionProvider" : "CUDAExecutionProvider";
}

/* Try to append the requested GPU EP to the session options.
 * Returns 1 when the EP was appended, 0 when it is unavailable and the
 * caller should continue with the default CPU EP. */
static int dcvc_try_append_gpu_ep(const OrtApi* api, OrtSessionOptions* opts, int ep)
{
    OrtStatus* st = nullptr;

    /* Check the provider is actually compiled into this ORT build. */
    char** avail = nullptr;
    int n_avail = 0;
    st = api->GetAvailableProviders(&avail, &n_avail);
    if (st) { api->ReleaseStatus(st); return 0; }
    int found = 0;
    for (int i = 0; i < n_avail; i++)
        if (avail[i] && strcmp(avail[i], dcvc_ep_name(ep)) == 0) found = 1;
    OrtStatus* rst = api->ReleaseAvailableProviders(avail, n_avail);
    if (rst) api->ReleaseStatus(rst);
    if (!found) return 0;

    const char* dev = getenv("DCVC_GPU_DEVICE");
    const char* keys[] = { "device_id" };
    const char* vals[] = { (dev && dev[0]) ? dev : "0" };

    if (ep == 2) {
        OrtTensorRTProviderOptionsV2* trt = nullptr;
        st = api->CreateTensorRTProviderOptions(&trt);
        if (!st) st = api->UpdateTensorRTProviderOptions(trt, keys, vals, 1);
        if (!st) st = api->SessionOptionsAppendExecutionProvider_TensorRT_V2(opts, trt);
        if (trt) api->ReleaseTensorRTProviderOptions(trt);
    } else {
        OrtCUDAProviderOptionsV2* cuda = nullptr;
        st = api->CreateCUDAProviderOptions(&cuda);
        if (!st) st = api->UpdateCUDAProviderOptions(cuda, keys, vals, 1);
        if (!st) st = api->SessionOptionsAppendExecutionProvider_CUDA_V2(opts, cuda);
        if (cuda) api->ReleaseCUDAProviderOptions(cuda);
    }
    if (st) {
        fprintf(stderr, "dcvc_onnx: failed to append %s: %s\n",
                dcvc_ep_name(ep), api->GetErrorMessage(st));
        api->ReleaseStatus(st);
        return 0;
    }
    return 1;
}

int dcvc_cpu_cuda_ep_usable(void)
{
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api) return 0;
    OrtEnv* env = nullptr;
    OrtStatus* st = api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "dcvc_ep_probe", &env);
    if (st) { api->ReleaseStatus(st); return 0; }
    OrtSessionOptions* opts = nullptr;
    st = api->CreateSessionOptions(&opts);
    if (st) {
        api->ReleaseStatus(st);
        api->ReleaseEnv(env);
        return 0;
    }
    int ok = dcvc_try_append_gpu_ep(api, opts, 1);
    api->ReleaseSessionOptions(opts);
    api->ReleaseEnv(env);
    return ok;
}

/* Log the EP selection outcome once per process (a pipeline creates many
 * engines; per-engine messages would drown the output). */
static void dcvc_log_ep_once(int ep, int appended)
{
    static int logged = 0;
    if (logged) return;
    logged = 1;
    if (appended)
        fprintf(stderr, "dcvc_onnx: using %s\n", dcvc_ep_name(ep));
    else
        fprintf(stderr, "dcvc_onnx: %s unavailable in this ONNX Runtime build; using CPU\n",
                dcvc_ep_name(ep));
}

/* Process-wide shared OrtEnv with a single global thread pool.
 *
 * Previously every engine created its own env, and every session created its
 * own intra-op thread pool. A round-trip run builds ~50 engines, i.e. ~50
 * pools; on Windows each thread commits 1 MB of stack up front, which alone
 * is ~400 MB and OOMs the process after a few dozen frames. ORT supports
 * sharing one env across all sessions, and with global thread pools the
 * whole process keeps a single intra-op pool instead.
 *
 * The env is intentionally never released: ORT recommends one env per
 * process, and repeated CreateEnv/ReleaseEnv cycles would just re-churn the
 * thread pools. Thread counts are governed here; engines opt their sessions
 * into these global pools via DisablePerSessionThreads (sessions would
 * otherwise default to per-session pools even on a global-pool env). */
static OrtEnv* g_dcvc_env = nullptr;
static std::mutex g_dcvc_env_mutex;

static OrtEnv* dcvc_shared_env(const OrtApi* api, OrtStatus** out_st)
{
    std::lock_guard<std::mutex> lk(g_dcvc_env_mutex);
    if (g_dcvc_env) return g_dcvc_env;

    OrtThreadingOptions* topts = nullptr;
    OrtStatus* st = api->CreateThreadingOptions(&topts);
    if (st) { *out_st = st; return nullptr; }

    /* Intra-op threads feed Ort::KernelContext::ParallelFor inside FXP ops.
     * FXP integer MAC partitioned by output channel is bit-exact for any
     * thread count. Default: min(8, HW concurrency). Override with
     * DCVC_ORT_INTRA_OP_THREADS (1 keeps the old single-thread behaviour). */
    int intra = 8;
    const char* e = getenv("DCVC_ORT_INTRA_OP_THREADS");
    if (e && e[0]) {
        intra = atoi(e);
        if (intra < 1) intra = 1;
        if (intra > 64) intra = 64;
    } else {
#ifdef _WIN32
        SYSTEM_INFO si; GetSystemInfo(&si);
        int hw = (int)si.dwNumberOfProcessors;
#else
        int hw = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
        if (hw < 1) hw = 1;
        intra = hw < 8 ? hw : 8;
    }

    st = api->SetGlobalIntraOpNumThreads(topts, intra);
    if (!st) st = api->SetGlobalInterOpNumThreads(topts, 1);
    if (!st) st = api->CreateEnvWithGlobalThreadPools(ORT_LOGGING_LEVEL_WARNING, "dcvc_onnx",
                                                      topts, &g_dcvc_env);
    api->ReleaseThreadingOptions(topts);
    if (st) { *out_st = st; g_dcvc_env = nullptr; return nullptr; }
    return g_dcvc_env;
}

DcvcCpuEngine* dcvc_cpu_engine_create(const char* onnx_path, int use_gpu, DcvcCpuStatus* out_st)
{
    if (out_st) *out_st = DCVC_CPU_OK;

    OrtApi* api = (OrtApi*)OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api) { if (out_st) *out_st = DCVC_CPU_ERR_ONNX; return nullptr; }

    DcvcCpuEngine* eng = (DcvcCpuEngine*)calloc(1, sizeof(DcvcCpuEngine));
    if (!eng) { if (out_st) *out_st = DCVC_CPU_ERR_OOM; return nullptr; }
    eng->api = api;

    OrtStatus* st = nullptr;
    eng->env = dcvc_shared_env(api, &st);
    check_status(api, st, out_st);
    if (st) { dcvc_cpu_engine_destroy(eng); return nullptr; }

    st = api->CreateSessionOptions(&eng->opts);
    check_status(api, st, out_st);
    if (st) { dcvc_cpu_engine_destroy(eng); return nullptr; }

    /* Route this session to the shared env's global thread pool. Sessions
     * default to per-session pools even on a global-pool env
     * (SessionOptions::DEFAULT_USE_PER_SESSION_THREADS == true), so every
     * engine must opt out explicitly; otherwise each session still spawns
     * its own pool (~50 engines * N threads * 1 MB committed stack on
     * Windows was the round-trip OOM). */
    st = api->DisablePerSessionThreads(eng->opts);
    check_status(api, st, out_st);
    if (st) { dcvc_cpu_engine_destroy(eng); return nullptr; }

    /* com.dcvc.FxpConv1x1 etc. — no-op for models that do not use them. */
    if (dcvc_ort_register_custom_ops(api, eng->opts) != 0) {
        fprintf(stderr, "dcvc_onnx: warning: custom op registration failed\n");
    }

    /* Enable deterministic compute mode for cross-platform reproducibility. */
    st = api->AddSessionConfigEntry(eng->opts, "session.use_deterministic_compute", "1");
    check_status(api, st, out_st);
    if (st) { dcvc_cpu_engine_destroy(eng); return nullptr; }

    int ep = dcvc_requested_ep(use_gpu);
    if (ep != 0) dcvc_log_ep_once(ep, dcvc_try_append_gpu_ep(api, eng->opts, ep));

    /* Optional ORT per-op profiling: set DCVC_PROFILE=<prefix> to emit
     * <prefix>.*.json trace files (one per session/engine). */
    {
        const char* prof = getenv("DCVC_PROFILE");
        if (prof && prof[0]) {
            /* EnableProfiling takes ORTCHAR_T* (wchar_t* on Windows). */
#ifdef _WIN32
            st = api->EnableProfiling(eng->opts, dcvc_to_wide(prof).c_str());
#else
            st = api->EnableProfiling(eng->opts, prof);
#endif
            check_status(api, st, out_st);
            if (st) { dcvc_cpu_engine_destroy(eng); return nullptr; }
        }
    }

#ifdef _WIN32
    auto _wpath = dcvc_to_wide(onnx_path);
    st = api->CreateSession(eng->env, _wpath.c_str(), eng->opts, &eng->session);
#else
    st = api->CreateSession(eng->env, onnx_path, eng->opts, &eng->session);
#endif
    check_status(api, st, out_st);
    if (st) { dcvc_cpu_engine_destroy(eng); return nullptr; }

    st = api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &eng->mem_info);
    check_status(api, st, out_st);
    if (st) { dcvc_cpu_engine_destroy(eng); return nullptr; }

    st = api->CreateRunOptions(&eng->run_opts);
    check_status(api, st, out_st);
    if (st) { dcvc_cpu_engine_destroy(eng); return nullptr; }

    return eng;
}

void dcvc_cpu_engine_destroy(DcvcCpuEngine* eng)
{
    if (!eng) return;
    const OrtApi* api = eng->api;
    if (eng->run_opts) api->ReleaseRunOptions(eng->run_opts);
    if (eng->mem_info) api->ReleaseMemoryInfo(eng->mem_info);
    if (eng->session) api->ReleaseSession(eng->session);
    if (eng->opts) api->ReleaseSessionOptions(eng->opts);
    /* eng->env is the process-wide shared env (dcvc_shared_env): not owned
     * by this engine, never released here. */
    free(eng);
}

static ONNXTensorElementDataType element_type(int is_fp16)
{
    return is_fp16 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

static int64_t total_elements(const DcvcCpuTensorView* t)
{
    return (int64_t)t->n * t->c * t->h * t->w;
}

DcvcCpuStatus dcvc_cpu_engine_run(DcvcCpuEngine* eng,
                                    DcvcCpuTensorView* inputs, int n_in,
                                    DcvcCpuTensorView* outputs, int n_out)
{
    if (!eng || !eng->api || !eng->session) return DCVC_CPU_ERR_INVALID_ARG;

    const OrtApi* api = eng->api;
    OrtStatus* st = nullptr;

    std::vector<OrtValue*> in_vals(n_in, nullptr);
    std::vector<OrtValue*> out_vals(n_out, nullptr);
    std::vector<const char*> in_names(n_in, nullptr);
    std::vector<const char*> out_names(n_out, nullptr);

    OrtAllocator* allocator = nullptr;
    st = api->GetAllocatorWithDefaultOptions(&allocator);
    if (st) { check_status(api, st, nullptr); return DCVC_CPU_ERR_ONNX; }

    /* Query input names and create tensors from caller data */
    for (int i = 0; i < n_in; i++) {
        OrtTypeInfo* type_info = nullptr;
        st = api->SessionGetInputTypeInfo(eng->session, i, &type_info);
        if (st) { check_status(api, st, nullptr); goto cleanup; }
        const OrtTensorTypeAndShapeInfo* info = nullptr;
        st = api->CastTypeInfoToTensorInfo(type_info, &info);
        if (st) { check_status(api, st, nullptr); goto cleanup; }
        (void)info;  /* validated above; shape is provided by caller */
        api->ReleaseTypeInfo(type_info);

        st = api->SessionGetInputName(eng->session, i, allocator, (char**)&in_names[i]);
        if (st) { check_status(api, st, nullptr); goto cleanup; }

        int64_t shape[4] = { inputs[i].n, inputs[i].c, inputs[i].h, inputs[i].w };
        size_t bytes = total_elements(&inputs[i]) * (inputs[i].is_fp16 ? 2 : 4);
        st = api->CreateTensorWithDataAsOrtValue(eng->mem_info, inputs[i].data, bytes,
                                                  shape, 4, element_type(inputs[i].is_fp16),
                                                  &in_vals[i]);
        if (st) { check_status(api, st, nullptr); goto cleanup; }
    }

    /* Query output names and create placeholder tensors */
    for (int i = 0; i < n_out; i++) {
        st = api->SessionGetOutputName(eng->session, i, allocator, (char**)&out_names[i]);
        if (st) { check_status(api, st, nullptr); goto cleanup; }

        int64_t shape[4] = { outputs[i].n, outputs[i].c, outputs[i].h, outputs[i].w };
        size_t bytes = total_elements(&outputs[i]) * (outputs[i].is_fp16 ? 2 : 4);
        st = api->CreateTensorWithDataAsOrtValue(eng->mem_info, outputs[i].data, bytes,
                                                  shape, 4, element_type(outputs[i].is_fp16),
                                                  &out_vals[i]);
        if (st) { check_status(api, st, nullptr); goto cleanup; }
    }

    st = api->Run(eng->session, eng->run_opts,
                  in_names.data(), in_vals.data(), n_in,
                  out_names.data(), out_names.size(), out_vals.data());
    if (st) { check_status(api, st, nullptr); goto cleanup; }

cleanup:
    for (OrtValue* v : in_vals) if (v) api->ReleaseValue(v);
    for (OrtValue* v : out_vals) if (v) api->ReleaseValue(v);
    for (const char* n : in_names) if (n) allocator->Free(allocator, (void*)n);
    for (const char* n : out_names) if (n) allocator->Free(allocator, (void*)n);
    /* allocator is the default allocator; do not release it. */

    return st ? DCVC_CPU_ERR_ONNX : DCVC_CPU_OK;
}
