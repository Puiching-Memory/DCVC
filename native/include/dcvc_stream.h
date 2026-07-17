#ifndef DCVC_STREAM_H
#define DCVC_STREAM_H
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Shared non-blocking CUDA stream for all DCVC-RT GPU operations.
 * Using a dedicated stream avoids default-stream synchronization overhead
 * (enqueueV3 on stream 0 inserts extra cudaStreamSynchronize calls) and
 * lets GPU work overlap with CPU rANS processing. */
cudaStream_t dcvc_stream(void);
/* Synchronize the shared stream.  Use ONLY when the CPU must read GPU
 * results (e.g. before touching a D2H-copied buffer). */
void dcvc_sync(void);
#ifdef __cplusplus
}
#endif
#endif

/* Event for fine-grained sync: wait only for a specific D2H copy while
 * letting subsequent GPU kernels continue executing in the background. */
cudaEvent_t dcvc_event(void);
