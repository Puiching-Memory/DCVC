/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Core public types: version, status codes, logging and configuration.
 * This header is safe to include from C89/C99 and C++.
 */
#ifndef DCVC_TYPES_H
#define DCVC_TYPES_H

#include "dcvc_export.h"
#include "dcvc_version.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- *
 *  Version                                                                   *
 * ------------------------------------------------------------------------- */
DCVC_API const char* DCVC_CALL dcvc_version_string(void);        /* "1.0.0"   */
DCVC_API int         DCVC_CALL dcvc_version_major(void);
DCVC_API int         DCVC_CALL dcvc_version_minor(void);
DCVC_API int         DCVC_CALL dcvc_version_patch(void);

/* ------------------------------------------------------------------------- *
 *  Status codes                                                              *
 * ------------------------------------------------------------------------- */
typedef enum {
    DCVC_OK = 0,
    DCVC_ERR_INVALID_ARG = 1,   /* NULL handle / out-of-range W,H,qp          */
    DCVC_ERR_IO = 2,           /* model or CDF file missing / unreadable      */
    DCVC_ERR_FORMAT = 3,        /* corrupt or unknown bitstream packet         */
    DCVC_ERR_MODEL = 4,         /* ONNX model failed to load / inference error */
    DCVC_ERR_ORT = 5,           /* ONNX Runtime internal failure               */
    DCVC_ERR_OOM = 6,           /* allocation failure                          */
    DCVC_ERR_ENTROPY = 7,       /* rANS desync: stream corrupt beyond recovery */
    DCVC_ERR_UNSUPPORTED = 8    /* e.g. GPU EP requested but unavailable       */
} dcvc_status_t;

/* Human-readable message for a status code (static string, never NULL). */
DCVC_API const char* DCVC_CALL dcvc_status_string(dcvc_status_t st);

/* ------------------------------------------------------------------------- *
 *  Logging                                                                   *
 * ------------------------------------------------------------------------- */
typedef enum {
    DCVC_LOG_ERROR = 0,
    DCVC_LOG_WARN = 1,
    DCVC_LOG_INFO = 2,
    DCVC_LOG_DEBUG = 3
} dcvc_log_level_t;

/* Callback receives a NUL-terminated message (no trailing newline). If no
 * callback is set, messages at WARN/ERROR go to stderr. Pass NULL to restore
 * the default sink. The callback may be invoked from any codec thread. */
typedef void (*dcvc_log_callback_t)(dcvc_log_level_t level,
                                    const char* message, void* user_data);
DCVC_API void DCVC_CALL dcvc_set_log_callback(dcvc_log_callback_t cb,
                                              void* user_data);

/* ------------------------------------------------------------------------- *
 *  Configuration                                                             *
 * ------------------------------------------------------------------------- */
typedef struct dcvc_config dcvc_config_t;

DCVC_API dcvc_config_t* DCVC_CALL dcvc_config_create(void);
DCVC_API void           DCVC_CALL dcvc_config_destroy(dcvc_config_t* cfg);

/* Directory holding the ONNX models + CDF tables (gaussian_cdf.npy, q- banks).
 * Required: a session cannot be created without it. */
DCVC_API dcvc_status_t DCVC_CALL dcvc_config_set_model_dir(dcvc_config_t* cfg,
                                                           const char* path);

/* Number of intra-op threads (feeds ORT + FXP ParallelFor). <=0 lets the
 * library pick min(8, hardware concurrency). */
DCVC_API dcvc_status_t DCVC_CALL dcvc_config_set_threads(dcvc_config_t* cfg, int n);

/* Execution provider: 0 = CPU (default), 1 = CUDA EP, 2 = TensorRT EP.
 * A GPU EP needs an ORT build that ships it; otherwise it falls back to CPU
 * and returns DCVC_ERR_UNSUPPORTED from dcvc_session_create when strict. */
DCVC_API dcvc_status_t DCVC_CALL dcvc_config_set_exec_provider(dcvc_config_t* cfg,
                                                               int mode);

/* When nonzero, append a CRC-32 integrity tag to every emitted packet and
 * verify it on decode (default: on). Cheap insurance against channel flips. */
DCVC_API dcvc_status_t DCVC_CALL dcvc_config_set_crc(dcvc_config_t* cfg, int enable);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_TYPES_H */
