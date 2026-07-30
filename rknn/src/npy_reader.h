/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Minimal cross-platform NumPy (.npy) reader/writer used by the DCVC CPU
 * ONNX codec and its tests. Supports the dtypes this project needs:
 * little-endian float32 ('<f4'), int32 ('<i4') and float16 ('<f2').
 *
 * Pure C89/C99 + binary stdio; works on Linux, macOS and Windows (MSVC).
 */
#ifndef DCVC_NPY_READER_H
#define DCVC_NPY_READER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DCVC_NPY_NONE = 0,
    DCVC_NPY_F32,    /* 4 bytes/elem, data is float*     */
    DCVC_NPY_I32,    /* 4 bytes/elem, data is int32_t*   */
    DCVC_NPY_F16     /* 2 bytes/elem, data is uint16_t*  */
} DcvcNpyDtype;

typedef struct {
    int ndims;
    int dims[8];        /* row-major; unused trailing dims default to 1 */
    size_t elems;       /* product of dims */
    DcvcNpyDtype dtype;
    void* data;         /* malloc'd, caller frees with dcvc_npy_free() */
} DcvcNpy;

/* Parse a .npy file into o. Returns 0 on success, non-zero on error.
 * On error o->data is NULL and o is zeroed. */
int  dcvc_npy_read(const char* path, DcvcNpy* o);

/* Free the buffer held by o and zero it. Safe on a zeroed struct. */
void dcvc_npy_free(DcvcNpy* o);

/* Convenience accessors: return the typed pointer or NULL if dtype mismatches. */
const float*    dcvc_npy_f32(const DcvcNpy* o);
const int32_t*  dcvc_npy_i32(const DcvcNpy* o);
const uint16_t* dcvc_npy_f16(const DcvcNpy* o);

/* Writers. Return 0 on success. nd <= 8. */
int dcvc_npy_write_f32(const char* path, const float* data, const int* dims, int nd);
int dcvc_npy_write_f16(const char* path, const float* data, const int* dims, int nd);

/* Read only the shape (ndims and dims) from a .npy file without allocating
 * the data buffer. Returns 0 on success, non-zero on error. */
int dcvc_npy_read_meta(const char* path, int* out_ndims, int* out_dims, int max_nd);

/* Read one frame from a multi-frame float32 NCHW .npy file. The file is
 * expected to have at least (frame_idx + 1) elements in the leading dimension
 * and shape [..., C, H, W]. The last three dimensions are written to out_dims,
 * and out_data must hold C*H*W floats. Returns 0 on success. */
int dcvc_npy_read_frame_f32(const char* path, int frame_idx, float* out_data,
                              int* out_dims /* 3 entries: C, H, W */);

/* float16 <-> float32 conversion (IEEE 754 binary16). */
float    dcvc_f16_to_f32(uint16_t h);
uint16_t dcvc_f32_to_f16(float f);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_NPY_READER_H */
