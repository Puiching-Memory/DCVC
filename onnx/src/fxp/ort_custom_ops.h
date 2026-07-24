#ifndef DCVC_ORT_CUSTOM_OPS_H
#define DCVC_ORT_CUSTOM_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

struct OrtApi;
struct OrtSessionOptions;

/* Register com.dcvc.* custom ops on the session options.
 * The domain is process-lifetime static; safe to call once per engine create.
 * Returns 0 on success, non-zero on failure. */
int dcvc_ort_register_custom_ops(const struct OrtApi* api, struct OrtSessionOptions* opts);

#ifdef __cplusplus
}
#endif

#endif
