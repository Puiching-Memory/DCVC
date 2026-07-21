/* Internal helpers shared across native modules. */
#ifndef DCVC_RT_INTERNAL_H
#define DCVC_RT_INTERNAL_H

#include "dcvc_rt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DCVC_RT_QP_NUM 64
#define DCVC_RT_PAD_ALIGN 16
#define DCVC_RT_CH_Y_INTER 128
#define DCVC_RT_CH_Z 128
#define DCVC_RT_CH_Y_INTRA 256
#define DCVC_RT_CH_D 256
#define DCVC_RT_CH_SRC_D 192
#define DCVC_RT_VERSION_STR "0.1.0-native"

static inline int dcvc_rt_align_up(int v, int a)
{
    return (v + a - 1) / a * a;
}

static inline int dcvc_rt_ec_part_for_res(int width, int height)
{
    return (width * height > 1280 * 720) ? 1 : 0;
}

#ifdef __cplusplus
}
#endif

#endif /* DCVC_RT_INTERNAL_H */
