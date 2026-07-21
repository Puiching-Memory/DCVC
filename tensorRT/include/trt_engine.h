/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * C-callable TensorRT engine wrapper. Implemented in C++ (trt_engine.cpp).
 * Handles plugin .so loading, engine deserialization, and inference.
 */
#ifndef DCVC_TRT_ENGINE_H
#define DCVC_TRT_ENGINE_H

#include "dcvc_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DcvcTrtEngine DcvcTrtEngine;

/* Load a serialized engine and register plugins.
 * plugin_dir: directory containing libdcvc_*.so plugin shared libraries.
 * Returns NULL on failure; st (if non-NULL) receives the error code. */
DcvcTrtEngine* dcvc_trt_engine_load(const char* engine_path, const char* plugin_dir, DcvcRtStatus* st);

void dcvc_trt_engine_destroy(DcvcTrtEngine* eng);

/* Query tensor info */
int dcvc_trt_engine_num_io(const DcvcTrtEngine* eng);
const char* dcvc_trt_engine_tensor_name(const DcvcTrtEngine* eng, int idx);
int dcvc_trt_engine_is_input(const DcvcTrtEngine* eng, int idx);

/* Dynamic shape control */
DcvcRtStatus dcvc_trt_engine_set_shape(DcvcTrtEngine* eng, const char* name, const int32_t* dims, int ndims);
DcvcRtStatus dcvc_trt_engine_get_shape(const DcvcTrtEngine* eng, const char* name, int32_t* dims_out, int* ndims_out, int max_dims);

/* Bind device pointers and execute */
DcvcRtStatus dcvc_trt_engine_set_addr(DcvcTrtEngine* eng, const char* name, void* dev_ptr);
DcvcRtStatus dcvc_trt_engine_execute(DcvcTrtEngine* eng, void* cuda_stream);

/* At a fixed resolution every input shape and tensor address is identical
 * across frames. An engine is "bound" once its inputs/outputs have been set
 * so subsequent frames skip the redundant setInputShape/setTensorAddress host
 * calls. Clear the flag when shapes/addresses may have changed (resolution
 * change or buffer reallocation). */
int  dcvc_trt_engine_bound(const DcvcTrtEngine* eng);
void dcvc_trt_engine_set_bound(DcvcTrtEngine* eng, int bound);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_TRT_ENGINE_H */
