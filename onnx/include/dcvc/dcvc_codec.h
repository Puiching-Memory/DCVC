/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Encoder / decoder and the self-describing packet format.
 *
 * Threading model: an encoder and a decoder are bound to a single logical
 * stream and are NOT thread-safe by themselves. Create one per thread. The
 * session they share IS thread-safe.
 *
 * Pixel format: the codec operates on planar FP32 RGB in [0,1], NCHW
 * [1,3,H,W], the same layout the underlying pipelines use. A reference-frame
 * buffer (DPB) is owned internally; the caller only feeds frames and receives
 * packets (encode), or feeds packets and receives frames (decode).
 */
#ifndef DCVC_CODEC_H
#define DCVC_CODEC_H

#include "dcvc_session.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Frame types carried in a packet header. */
typedef enum {
    DCVC_FRAME_INTRA = 0,   /* I-frame: self-contained, no reference needed  */
    DCVC_FRAME_INTER = 1    /* P-frame: references the previous reconstruction*/
} dcvc_frame_type_t;

/* ------------------------------------------------------------------------- *
 *  Encoder                                                                   *
 * ------------------------------------------------------------------------- */
typedef struct dcvc_encoder dcvc_encoder_t;

/* Create an encoder for a fixed spatial size and base QP (0..63). The QP can
 * also be overridden per-frame via dcvc_encoder_encode_qp(). The encoder
 * owns an intra pipeline and an inter pipeline plus an internal DPB.
 * `width` and `height` must each be a multiple of 64. */
DCVC_API dcvc_status_t DCVC_CALL dcvc_encoder_create(dcvc_session_t* sess,
                                                     int width, int height, int qp,
                                                     dcvc_encoder_t** out);
DCVC_API void DCVC_CALL dcvc_encoder_destroy(dcvc_encoder_t* enc);

/* Encode one frame. `frame` is FP32 RGB [1,3,H,W] in [0,1].
 *   force_intra: 1 emits an I-frame; 0 emits I for the very first frame then
 *                P-frames thereafter.
 *   *out_packet is malloc'd by the library; free with dcvc_packet_free().
 *   *out_type (optional) receives the frame type actually emitted. */
DCVC_API dcvc_status_t DCVC_CALL dcvc_encoder_encode(dcvc_encoder_t* enc,
                                                     const float* frame,
                                                     int force_intra,
                                                     uint8_t** out_packet,
                                                     size_t* out_size,
                                                     dcvc_frame_type_t* out_type);

/* Same as dcvc_encoder_encode but with a per-call QP override (0..63). */
DCVC_API dcvc_status_t DCVC_CALL dcvc_encoder_encode_qp(dcvc_encoder_t* enc,
                                                        const float* frame,
                                                        int force_intra, int qp,
                                                        uint8_t** out_packet,
                                                        size_t* out_size,
                                                        dcvc_frame_type_t* out_type);

/* Reset the encoder stream (clears the DPB so the next frame is intra). */
DCVC_API dcvc_status_t DCVC_CALL dcvc_encoder_reset(dcvc_encoder_t* enc);

/* ------------------------------------------------------------------------- *
 *  Decoder                                                                   *
 * ------------------------------------------------------------------------- */
typedef struct dcvc_decoder dcvc_decoder_t;

/* Create a decoder. Spatial size and QP are read from each packet, so no
 * dimensions are needed up front. The decoder owns an internal DPB. */
DCVC_API dcvc_status_t DCVC_CALL dcvc_decoder_create(dcvc_session_t* sess,
                                                     dcvc_decoder_t** out);
DCVC_API void DCVC_CALL dcvc_decoder_destroy(dcvc_decoder_t* dec);

/* Decode one packet. `out_frame` must point to a caller buffer of at least
 * width*height*3 floats; width/height are taken from the packet header.
 *   *out_type (optional) receives the decoded frame type.
 * Returns DCVC_ERR_ENTROPY on an unrecoverable rANS desync; in that case the
 * reconstruction is best-effort and the caller should resync at the next
 * intra frame (dcvc_decoder_reset()). */
DCVC_API dcvc_status_t DCVC_CALL dcvc_decoder_decode(dcvc_decoder_t* dec,
                                                     const uint8_t* packet,
                                                     size_t size,
                                                     float* out_frame,
                                                     int* out_width,
                                                     int* out_height,
                                                     dcvc_frame_type_t* out_type);

DCVC_API dcvc_status_t DCVC_CALL dcvc_decoder_reset(dcvc_decoder_t* dec);

/* ------------------------------------------------------------------------- *
 *  Packet helpers                                                            *
 * ------------------------------------------------------------------------- */

/* Peek a packet header without decoding. Returns DCVC_OK and fills the out
 * fields on a well-formed packet, or DCVC_ERR_FORMAT otherwise. */
DCVC_API dcvc_status_t DCVC_CALL dcvc_packet_probe(const uint8_t* packet, size_t size,
                                                   int* out_width, int* out_height,
                                                   int* out_qp,
                                                   dcvc_frame_type_t* out_type);

/* Free a packet returned by the encoder. Matches the allocator used inside. */
DCVC_API void DCVC_CALL dcvc_packet_free(uint8_t* packet);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_CODEC_H */
