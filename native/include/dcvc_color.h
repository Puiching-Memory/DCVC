#ifndef DCVC_COLOR_H
#define DCVC_COLOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* BT.709, matching src/utils/transforms.py
 * yuv420 8-bit planar -> float32 YCbCr444 planar CHW [0,1]
 * out size: 3 * height * width floats
 */
void dcvc_yuv420_to_ycbcr444_f32(const uint8_t* y, int y_stride,
                                 const uint8_t* u, int u_stride,
                                 const uint8_t* v, int v_stride,
                                 int width, int height, float* out_chw);

/* float32 YCbCr444 CHW [0,1] -> yuv420 8-bit (nearest downsample UV) */
void dcvc_ycbcr444_f32_to_yuv420(const float* in_chw, int width, int height,
                                 uint8_t* y, int y_stride,
                                 uint8_t* u, int u_stride,
                                 uint8_t* v, int v_stride);

void dcvc_rgb_to_ycbcr444_f32(const uint8_t* rgb, int width, int height, float* out_chw);
void dcvc_ycbcr444_f32_to_rgb(const float* in_chw, int width, int height, uint8_t* rgb);

/* Convert float32 CHW [0,1] to FP16 buffer (IEEE binary16 as uint16_t). */
void dcvc_f32_to_f16(const float* src, uint16_t* dst, size_t n);
void dcvc_f16_to_f32(const uint16_t* src, float* dst, size_t n);

#ifdef __cplusplus
}
#endif

#endif
