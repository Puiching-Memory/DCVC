/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Session: owns the shared runtime state (ORT environment, the loaded model
 * directory, custom-op registration). A session is cheap to keep alive and
 * may be shared across encoder/decoder instances and threads. The heavyweight
 * ONNX model loading happens when an encoder/decoder is created, not here.
 */
#ifndef DCVC_SESSION_H
#define DCVC_SESSION_H

#include "dcvc_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dcvc_session dcvc_session_t;

/* Create a session from a configuration. The config is copied; it may be
 * destroyed immediately after. On success *out is a new session (refcount 1). */
DCVC_API dcvc_status_t DCVC_CALL dcvc_session_create(const dcvc_config_t* cfg,
                                                     dcvc_session_t** out);

/* Drop a reference; the session is freed when the last reference is gone. */
DCVC_API void DCVC_CALL dcvc_session_destroy(dcvc_session_t* sess);

/* Bump/drop a reference manually (handy for sharing across threads). */
DCVC_API dcvc_session_t* DCVC_CALL dcvc_session_ref(dcvc_session_t* sess);
DCVC_API void            DCVC_CALL dcvc_session_unref(dcvc_session_t* sess);

/* Last error recorded on this session, or NULL if none. Thread-local within
 * the session, valid until the next call from the same thread. */
DCVC_API const char* DCVC_CALL dcvc_session_last_error(dcvc_session_t* sess);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_SESSION_H */
