#include "dcvc_color.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    const int w = 4, h = 4;
    uint8_t y[16], u[4], v[4];
    for (int i = 0; i < 16; i++) y[i] = (uint8_t)(i * 10);
    for (int i = 0; i < 4; i++) {
        u[i] = 128;
        v[i] = 128;
    }
    float* yuv = (float*)malloc(3 * w * h * sizeof(float));
    dcvc_yuv420_to_ycbcr444_f32(y, w, u, w / 2, v, w / 2, w, h, yuv);
    if (fabsf(yuv[0] - 0.f) > 1e-5f) {
        fprintf(stderr, "y0 bad\n");
        return 1;
    }
    uint16_t* f16 = (uint16_t*)malloc(3 * w * h * sizeof(uint16_t));
    dcvc_f32_to_f16(yuv, f16, (size_t)3 * w * h);
    float* back = (float*)malloc(3 * w * h * sizeof(float));
    dcvc_f16_to_f32(f16, back, (size_t)3 * w * h);
    if (fabsf(back[5] - yuv[5]) > 1e-2f) {
        fprintf(stderr, "f16 roundtrip\n");
        return 1;
    }

    uint8_t rgb[3] = {255, 0, 0};
    float ycbcr[3];
    dcvc_rgb_to_ycbcr444_f32(rgb, 1, 1, ycbcr);
    uint8_t rgb2[3];
    dcvc_ycbcr444_f32_to_rgb(ycbcr, 1, 1, rgb2);
    if (abs((int)rgb2[0] - 255) > 2) {
        fprintf(stderr, "rgb roundtrip R=%d\n", rgb2[0]);
        return 1;
    }

    free(yuv);
    free(f16);
    free(back);
    printf("test_color OK\n");
    return 0;
}
