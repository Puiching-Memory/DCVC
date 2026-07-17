/* Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 *
 * DCVC-RT native C API — high-performance encode/decode runtime.
 */
#ifndef DCVC_RT_H
#define DCVC_RT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(DCVC_RT_BUILD_SHARED)
#  ifdef DCVC_RT_EXPORTS
#    define DCVC_RT_API __declspec(dllexport)
#  else
#    define DCVC_RT_API __declspec(dllimport)
#  endif
#else
#  define DCVC_RT_API
#endif

typedef enum DcvcRtStatus {
    DCVC_RT_OK = 0,
    DCVC_RT_ERR_INVALID_ARG = 1,
    DCVC_RT_ERR_IO = 2,
    DCVC_RT_ERR_NO_ENGINE = 3,
    DCVC_RT_ERR_CUDA = 4,
    DCVC_RT_ERR_TRT = 5,
    DCVC_RT_ERR_ENTROPY = 6,
    DCVC_RT_ERR_BITSTREAM = 7,
    DCVC_RT_ERR_OOM = 8,
    DCVC_RT_ERR_UNSUPPORTED = 9,
    DCVC_RT_ERR_INTERNAL = 10
} DcvcRtStatus;

typedef enum DcvcRtPixelFormat {
    DCVC_RT_FMT_YUV420P = 0,
    DCVC_RT_FMT_YUV444P = 1,
    DCVC_RT_FMT_RGB24 = 2
} DcvcRtPixelFormat;

typedef enum DcvcRtFrameType {
    DCVC_RT_FRAME_I = 0,
    DCVC_RT_FRAME_P = 1
} DcvcRtFrameType;

typedef struct DcvcRtConfig {
    int width;              /* original width */
    int height;             /* original height */
    int qp;                 /* 0..63 */
    int reset_interval;     /* feature refresh period; default 64 */
    int device_id;          /* CUDA device */
    int use_cuda_graph;     /* 0/1 */
    const char* asset_dir;  /* engines, CDF, weights manifest */
    DcvcRtPixelFormat format;
} DcvcRtConfig;

typedef struct DcvcRtFrame {
    int width;
    int height;
    DcvcRtPixelFormat format;
    /* YUV420: y, u, v planes; YUV444/RGB: data[0] contiguous CHW or HWC per format docs */
    const uint8_t* data[3];
    int stride[3];
    /* Optional FP16 planar YCbCr444 [0,1] already prepared (overrides data[]) */
    const void* ycbcr444_fp16; /* size = 3 * padH * padW * 2, may be NULL */
} DcvcRtFrame;

typedef struct DcvcRtPacket {
    uint8_t* data;
    size_t size;
    DcvcRtFrameType frame_type;
    int qp;
    int sps_written; /* 1 if this packet begins with SPS */
} DcvcRtPacket;

typedef struct DcvcRtEncoder DcvcRtEncoder;
typedef struct DcvcRtDecoder DcvcRtDecoder;

DCVC_RT_API const char* dcvc_rt_status_string(DcvcRtStatus st);
DCVC_RT_API const char* dcvc_rt_version(void);

DCVC_RT_API DcvcRtStatus dcvc_rt_config_init(DcvcRtConfig* cfg);

DCVC_RT_API DcvcRtEncoder* dcvc_rt_encoder_create(const DcvcRtConfig* cfg, DcvcRtStatus* out_st);
DCVC_RT_API DcvcRtStatus dcvc_rt_encode_frame(DcvcRtEncoder* enc,
                                              const DcvcRtFrame* in,
                                              DcvcRtPacket* out);
DCVC_RT_API DcvcRtStatus dcvc_rt_encoder_flush(DcvcRtEncoder* enc, DcvcRtPacket* out);
DCVC_RT_API void dcvc_rt_encoder_destroy(DcvcRtEncoder* enc);

DCVC_RT_API DcvcRtDecoder* dcvc_rt_decoder_create(const DcvcRtConfig* cfg, DcvcRtStatus* out_st);
DCVC_RT_API DcvcRtStatus dcvc_rt_decode_packet(DcvcRtDecoder* dec,
                                               const uint8_t* data,
                                               size_t size,
                                               DcvcRtFrame* out);
DCVC_RT_API void dcvc_rt_decoder_destroy(DcvcRtDecoder* dec);

DCVC_RT_API void dcvc_rt_packet_free(DcvcRtPacket* pkt);
DCVC_RT_API void dcvc_rt_frame_free_planes(DcvcRtFrame* frame);


#ifdef __cplusplus
}
#endif

#endif /* DCVC_RT_H */

