/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * DCVC-SDK — public API umbrella header. Include this single header to use
 * the cross-platform, bit-exact neural video codec built on ONNX Runtime +
 * fixed-point custom ops + rANS entropy coding.
 *
 * Minimal usage:
 *
 *   dcvc_config_t* cfg = dcvc_config_create();
 *   dcvc_config_set_model_dir(cfg, "/path/to/models");
 *   dcvc_session_t* s = NULL;
 *   dcvc_session_create(cfg, &s);
 *   dcvc_config_destroy(cfg);
 *
 *   dcvc_encoder_t* enc = NULL;
 *   dcvc_encoder_create(s, 1920, 1080, 32, &enc);
 *   uint8_t* pkt = NULL; size_t n = 0;
 *   dcvc_encoder_encode(enc, frame_rgb_fp32, 1, &pkt, &n, NULL);
 *   dcvc_packet_free(pkt);
 *
 * See docs/sdk_quickstart.md and examples/dcvc_demo.c for the full walkthrough.
 */
#ifndef DCVC_H
#define DCVC_H

#include "dcvc_export.h"
#include "dcvc_version.h"
#include "dcvc_types.h"
#include "dcvc_session.h"
#include "dcvc_codec.h"

#endif /* DCVC_H */
