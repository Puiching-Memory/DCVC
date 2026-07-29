/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * DCVC-SDK facade: implements the public C API (include/dcvc/dcvc.h) on top
 * of the validated internals — the intra/inter pipelines, the ONNX Runtime
 * engine wrapper, the FXP fixed-point custom ops and the rANS entropy coder.
 *
 * The numerics path is untouched; this layer only adds a session/encoder/
 * decoder abstraction, a self-describing packet container, versioning and a
 * stable error/logging surface.
 */
#include "../include/dcvc/dcvc.h"
#include "dcvc_container.h"

#include "../src/cpu_intra_pipeline.h"
#include "../src/cpu_inter_pipeline.h"
#include "../src/onnx_engine.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <atomic>
#include <cstdio>

#if defined(_WIN32)
#  include <cstdlib>   /* _putenv_s */
#endif

/* ========================================================================= *
 *  Internal: logging                                                         *
 * ========================================================================= */
namespace {
struct LogState {
    std::mutex mu;
    dcvc_log_callback_t cb = nullptr;
    void* user = nullptr;
};
LogState& log_state() { static LogState s; return s; }

void dcvc_emit_log(dcvc_log_level_t lvl, const char* msg) {
    LogState& s = log_state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.cb) { s.cb(lvl, msg, s.user); }
    else if (lvl <= DCVC_LOG_WARN) { std::fprintf(stderr, "dcvc: %s\n", msg); }
}
}

/* ========================================================================= *
 *  Version / status strings                                                  *
 * ========================================================================= */
extern "C" {

DCVC_API const char* DCVC_CALL dcvc_version_string(void) {
    static const char v[] = "1.0.0";
    return v;
}
DCVC_API int DCVC_CALL dcvc_version_major(void) { return DCVC_VERSION_MAJOR; }
DCVC_API int DCVC_CALL dcvc_version_minor(void) { return DCVC_VERSION_MINOR; }
DCVC_API int DCVC_CALL dcvc_version_patch(void) { return DCVC_VERSION_PATCH; }

DCVC_API const char* DCVC_CALL dcvc_status_string(dcvc_status_t st) {
    switch (st) {
        case DCVC_OK:               return "ok";
        case DCVC_ERR_INVALID_ARG:  return "invalid argument";
        case DCVC_ERR_IO:           return "model or CDF file I/O error";
        case DCVC_ERR_FORMAT:       return "corrupt or unknown bitstream packet";
        case DCVC_ERR_MODEL:        return "ONNX model load / inference error";
        case DCVC_ERR_ORT:          return "ONNX Runtime internal error";
        case DCVC_ERR_OOM:          return "out of memory";
        case DCVC_ERR_ENTROPY:      return "entropy stream desynchronized";
        case DCVC_ERR_UNSUPPORTED:  return "requested feature unavailable";
    }
    return "unknown error";
}

DCVC_API void DCVC_CALL dcvc_set_log_callback(dcvc_log_callback_t cb, void* user) {
    LogState& s = log_state();
    std::lock_guard<std::mutex> lk(s.mu);
    s.cb = cb; s.user = user;
}

/* ------------------------------------------------------------------------- *
 *  Status mapping (internal DcvcCpuStatus -> public dcvc_status_t)            */
static dcvc_status_t map_status(DcvcCpuStatus s) {
    switch (s) {
        case DCVC_CPU_OK:              return DCVC_OK;
        case DCVC_CPU_ERR_INVALID_ARG: return DCVC_ERR_INVALID_ARG;
        case DCVC_CPU_ERR_IO:          return DCVC_ERR_IO;
        case DCVC_CPU_ERR_ONNX:        return DCVC_ERR_MODEL;
        case DCVC_CPU_ERR_OOM:         return DCVC_ERR_OOM;
        case DCVC_CPU_ERR_UNSUPPORTED: return DCVC_ERR_UNSUPPORTED;
        case DCVC_CPU_ERR_ENTROPY:     return DCVC_ERR_ENTROPY;
    }
    return DCVC_ERR_ORT;
}

/* ------------------------------------------------------------------------- *
 *  Internal: portable env-var set (used to bridge config -> runtime knobs).  *
 *  These are process-global by nature; guarded against concurrent sessions.   */
static std::mutex g_env_mu;

static void set_env_once(const char* name, const char* value) {
    std::lock_guard<std::mutex> lk(g_env_mu);
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

} /* extern "C" */

/* ========================================================================= *
 *  Config                                                                    *
 * ========================================================================= */
struct dcvc_config {
    std::string model_dir;
    int threads = 0;       /* 0 = auto */
    int exec_provider = 0; /* 0 cpu, 1 cuda, 2 trt */
    int crc = 1;           /* on by default */
};

extern "C" {

DCVC_API dcvc_config_t* DCVC_CALL dcvc_config_create(void) {
    try { return new dcvc_config_t; } catch (...) { return nullptr; }
}
DCVC_API void DCVC_CALL dcvc_config_destroy(dcvc_config_t* cfg) { delete cfg; }

DCVC_API dcvc_status_t DCVC_CALL dcvc_config_set_model_dir(dcvc_config_t* cfg, const char* path) {
    if (!cfg || !path) return DCVC_ERR_INVALID_ARG;
    cfg->model_dir = path;
    return DCVC_OK;
}
DCVC_API dcvc_status_t DCVC_CALL dcvc_config_set_threads(dcvc_config_t* cfg, int n) {
    if (!cfg) return DCVC_ERR_INVALID_ARG;
    cfg->threads = n;
    return DCVC_OK;
}
DCVC_API dcvc_status_t DCVC_CALL dcvc_config_set_exec_provider(dcvc_config_t* cfg, int mode) {
    if (!cfg || mode < 0 || mode > 2) return DCVC_ERR_INVALID_ARG;
    cfg->exec_provider = mode;
    return DCVC_OK;
}
DCVC_API dcvc_status_t DCVC_CALL dcvc_config_set_crc(dcvc_config_t* cfg, int enable) {
    if (!cfg) return DCVC_ERR_INVALID_ARG;
    cfg->crc = enable ? 1 : 0;
    return DCVC_OK;
}

/* ========================================================================= *
 *  Session                                                                   *
 * ========================================================================= */
struct dcvc_session {
    std::mutex mu;
    std::atomic<int> ref{1};
    std::string model_dir;
    int threads = 0;
    int exec_provider = 0;
    int crc = 1;
    std::string last_error;
};

/* Apply config to the process-global runtime knobs (the validated pipelines
 * read these). Done once per session create, before any pipeline loads. */
static void apply_runtime_knobs(const dcvc_session_t* s) {
    if (s->exec_provider != 0) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d", s->exec_provider);
        set_env_once("DCVC_USE_GPU", buf);
    }
    if (s->threads > 0) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d", s->threads);
        set_env_once("DCVC_ORT_INTRA_OP_THREADS", buf);
    }
}

DCVC_API dcvc_status_t DCVC_CALL dcvc_session_create(const dcvc_config_t* cfg,
                                                     dcvc_session_t** out) {
    if (!out) return DCVC_ERR_INVALID_ARG;
    *out = nullptr;
    if (!cfg || cfg->model_dir.empty()) return DCVC_ERR_INVALID_ARG;
    try {
        dcvc_session_t* s = new dcvc_session_t;
        s->model_dir = cfg->model_dir;
        s->threads = cfg->threads;
        s->exec_provider = cfg->exec_provider;
        s->crc = cfg->crc;
        apply_runtime_knobs(s);
        *out = s;
        return DCVC_OK;
    } catch (const std::bad_alloc&) {
        return DCVC_ERR_OOM;
    }
}

DCVC_API void DCVC_CALL dcvc_session_destroy(dcvc_session_t* sess) { dcvc_session_unref(sess); }

DCVC_API dcvc_session_t* DCVC_CALL dcvc_session_ref(dcvc_session_t* sess) {
    if (sess) sess->ref.fetch_add(1, std::memory_order_relaxed);
    return sess;
}
DCVC_API void DCVC_CALL dcvc_session_unref(dcvc_session_t* sess) {
    if (!sess) return;
    if (sess->ref.fetch_sub(1, std::memory_order_acq_rel) == 1) delete sess;
}

DCVC_API const char* DCVC_CALL dcvc_session_last_error(dcvc_session_t* sess) {
    if (!sess) return nullptr;
    std::lock_guard<std::mutex> lk(sess->mu);
    return sess->last_error.empty() ? nullptr : sess->last_error.c_str();
}

} /* extern "C" */

/* ========================================================================= *
 *  Shared encoder/decoder internals                                          *
 * ========================================================================= *
 *  The underlying pipelines are created for a fixed (H, W, qp): the qp drives
 *  which quantizer scale bank is loaded. To support per-frame qp (rate
 *  control) without reloading models every frame, we cache the intra+inter
 *  pair for the most recently used qp. A change of qp triggers a one-time
 *  reload of that pair. The DPB reference frame is owned by the codec object.
 */
namespace {

struct PipelinePair {
    DcvcCpuIntraPipeline* intra = nullptr;
    DcvcCpuInterPipeline* inter = nullptr;
    int qp = -1;

    void destroy() {
        if (intra)  dcvc_cpu_intra_pipeline_destroy(intra);
        if (inter)  dcvc_cpu_inter_pipeline_destroy(inter);
        intra = nullptr; inter = nullptr;
    }
};

/* Ensure pp holds the intra+inter pipelines for qp at the given geometry.
 * Reuses the cached pair when qp matches; otherwise tears down and rebuilds. */
static dcvc_status_t ensure_pipelines(dcvc_session_t* sess, PipelinePair* pp,
                                      int H, int W, int qp) {
    if (pp->qp == qp && pp->intra && pp->inter) return DCVC_OK;
    pp->destroy();

    DcvcCpuStatus st = DCVC_CPU_OK;
    pp->intra = dcvc_cpu_intra_pipeline_create(sess->model_dir.c_str(), H, W, qp, &st);
    if (!pp->intra) {
        sess->last_error = std::string("intra pipeline create failed: ") +
                           dcvc_cpu_status_string(st);
        dcvc_emit_log(DCVC_LOG_ERROR, sess->last_error.c_str());
        return map_status(st);
    }
    pp->inter = dcvc_cpu_inter_pipeline_create(sess->model_dir.c_str(), H, W, qp, &st);
    if (!pp->inter) {
        pp->destroy();
        sess->last_error = std::string("inter pipeline create failed: ") +
                           dcvc_cpu_status_string(st);
        dcvc_emit_log(DCVC_LOG_ERROR, sess->last_error.c_str());
        return map_status(st);
    }
    pp->qp = qp;
    return DCVC_OK;
}

} /* namespace */

/* ========================================================================= *
 *  Encoder                                                                   *
 * ========================================================================= */
struct dcvc_encoder {
    dcvc_session_t* sess;
    int W = 0, H = 0, qp = 0;
    PipelinePair pp;
    float* ref = nullptr;     /* previous reconstruction [3*H*W] */
    int have_ref = 0;
    int bytes_per_frame = 0;

    dcvc_encoder() = default;
    ~dcvc_encoder() {
        pp.destroy();
        std::free(ref);
        if (sess) dcvc_session_unref(sess);
    }
};

extern "C" {

DCVC_API dcvc_status_t DCVC_CALL dcvc_encoder_create(dcvc_session_t* sess,
                                                     int width, int height, int qp,
                                                     dcvc_encoder_t** out) {
    if (!out) return DCVC_ERR_INVALID_ARG;
    *out = nullptr;
    if (!sess || width <= 0 || height <= 0 || qp < 0 || qp > 63)
        return DCVC_ERR_INVALID_ARG;
    try {
        dcvc_encoder_t* e = new dcvc_encoder_t;
        e->sess = dcvc_session_ref(sess);
        e->W = width; e->H = height; e->qp = qp;
        e->bytes_per_frame = width * height * 3;
        dcvc_status_t st = ensure_pipelines(sess, &e->pp, height, width, qp);
        if (st != DCVC_OK) { delete e; return st; }
        e->ref = (float*)std::calloc((size_t)e->bytes_per_frame, sizeof(float));
        if (!e->ref) { delete e; return DCVC_ERR_OOM; }
        *out = e;
        return DCVC_OK;
    } catch (const std::bad_alloc&) {
        return DCVC_ERR_OOM;
    }
}

DCVC_API void DCVC_CALL dcvc_encoder_destroy(dcvc_encoder_t* enc) { delete enc; }

DCVC_API dcvc_status_t DCVC_CALL dcvc_encoder_reset(dcvc_encoder_t* enc) {
    if (!enc) return DCVC_ERR_INVALID_ARG;
    enc->have_ref = 0;
    return DCVC_OK;
}

/* Core encode: emits one self-describing packet. */
static dcvc_status_t encode_impl(dcvc_encoder_t* enc, const float* frame,
                                 int force_intra, int qp,
                                 uint8_t** out_packet, size_t* out_size,
                                 dcvc_frame_type_t* out_type) {
    if (!enc || !frame || !out_packet || !out_size)
        return DCVC_ERR_INVALID_ARG;
    *out_packet = nullptr; *out_size = 0;
    dcvc_session_t* sess = enc->sess;

    /* qp override: rebuild the pipeline pair on change (one-time cost). */
    if (qp != enc->qp) {
        if (qp < 0 || qp > 63) return DCVC_ERR_INVALID_ARG;
        dcvc_status_t st = ensure_pipelines(sess, &enc->pp, enc->H, enc->W, qp);
        if (st != DCVC_OK) return st;
        enc->qp = qp;
        enc->have_ref = 0;   /* quantizer changed: invalidate the DPB */
    }

    int intra = force_intra || !enc->have_ref;
    uint8_t* raw = nullptr;
    size_t raw_size = 0;
    DcvcCpuStatus cst;
    if (intra) {
        cst = dcvc_cpu_intra_pipeline_encode(enc->pp.intra, frame,
                                             &raw, &raw_size, enc->ref);
    } else {
        cst = dcvc_cpu_inter_pipeline_encode(enc->pp.inter, frame, enc->ref,
                                             &raw, &raw_size, enc->ref);
    }
    if (cst != DCVC_CPU_OK) {
        std::free(raw);
        sess->last_error = std::string(intra ? "intra" : "inter") + " encode failed: " +
                           dcvc_cpu_status_string(cst);
        return map_status(cst);
    }
    enc->have_ref = 1;

    dcvc_container_header hdr{};
    hdr.frame_type = (uint8_t)(intra ? DCVC_FRAME_INTRA : DCVC_FRAME_INTER);
    hdr.qp         = (uint8_t)enc->qp;
    hdr.width      = (uint32_t)enc->W;
    hdr.height     = (uint32_t)enc->H;
    hdr.payload_len = (uint32_t)raw_size;

    dcvc_status_t st = dcvc_container_build(&hdr, raw, sess->crc,
                                            out_packet, out_size);
    std::free(raw);
    if (st != DCVC_OK) return st;
    if (out_type) *out_type = (dcvc_frame_type_t)hdr.frame_type;
    return DCVC_OK;
}

DCVC_API dcvc_status_t DCVC_CALL dcvc_encoder_encode(dcvc_encoder_t* enc,
                                                     const float* frame,
                                                     int force_intra,
                                                     uint8_t** out_packet,
                                                     size_t* out_size,
                                                     dcvc_frame_type_t* out_type) {
    return encode_impl(enc, frame, force_intra, enc->qp,
                       out_packet, out_size, out_type);
}

DCVC_API dcvc_status_t DCVC_CALL dcvc_encoder_encode_qp(dcvc_encoder_t* enc,
                                                        const float* frame,
                                                        int force_intra, int qp,
                                                        uint8_t** out_packet,
                                                        size_t* out_size,
                                                        dcvc_frame_type_t* out_type) {
    return encode_impl(enc, frame, force_intra, qp,
                       out_packet, out_size, out_type);
}

} /* extern "C" */

/* ========================================================================= *
 *  Decoder                                                                   *
 * ========================================================================= *
 *  Dimensions/qp come from each packet header, so the decoder builds its
 *  pipeline pair lazily on first decode and whenever the geometry changes.
 */
struct dcvc_decoder {
    dcvc_session_t* sess;
    PipelinePair pp;
    int cur_W = 0, cur_H = 0, cur_qp = -1;
    float* ref = nullptr;
    int ref_cap = 0;        /* floats allocated in ref */
    int have_ref = 0;
    int ref_W = 0, ref_H = 0;

    dcvc_decoder() = default;
    ~dcvc_decoder() {
        pp.destroy();
        std::free(ref);
        if (sess) dcvc_session_unref(sess);
    }

    /* Grow ref to hold 3*H*W floats if needed. */
    dcvc_status_t ensure_ref(int H, int W) {
        int need = H * W * 3;
        if (need > ref_cap) {
            float* p = (float*)std::realloc(ref, (size_t)need * sizeof(float));
            if (!p) return DCVC_ERR_OOM;
            ref = p; ref_cap = need;
        }
        return DCVC_OK;
    }
};

extern "C" {

DCVC_API dcvc_status_t DCVC_CALL dcvc_decoder_create(dcvc_session_t* sess,
                                                     dcvc_decoder_t** out) {
    if (!out) return DCVC_ERR_INVALID_ARG;
    *out = nullptr;
    if (!sess) return DCVC_ERR_INVALID_ARG;
    try {
        dcvc_decoder_t* d = new dcvc_decoder_t;
        d->sess = dcvc_session_ref(sess);
        *out = d;
        return DCVC_OK;
    } catch (const std::bad_alloc&) {
        return DCVC_ERR_OOM;
    }
}

DCVC_API void DCVC_CALL dcvc_decoder_destroy(dcvc_decoder_t* dec) { delete dec; }

DCVC_API dcvc_status_t DCVC_CALL dcvc_decoder_reset(dcvc_decoder_t* dec) {
    if (!dec) return DCVC_ERR_INVALID_ARG;
    dec->have_ref = 0;
    return DCVC_OK;
}

DCVC_API dcvc_status_t DCVC_CALL dcvc_decoder_decode(dcvc_decoder_t* dec,
                                                     const uint8_t* packet, size_t size,
                                                     float* out_frame,
                                                     int* out_width, int* out_height,
                                                     dcvc_frame_type_t* out_type) {
    if (!dec || !packet || !out_frame)
        return DCVC_ERR_INVALID_ARG;
    if (out_width)  *out_width = 0;
    if (out_height) *out_height = 0;
    if (out_type)   *out_type = DCVC_FRAME_INTRA;

    dcvc_session_t* sess = dec->sess;

    dcvc_container_header hdr;
    const uint8_t* payload = nullptr;
    dcvc_status_t st = dcvc_container_parse(packet, size, &hdr, &payload);
    if (st != DCVC_OK) return st;

    /* Integrity: verify CRC (if present) before touching the entropy stream. */
    st = dcvc_container_verify_crc(packet, size, &hdr);
    if (st != DCVC_OK) { sess->last_error = "packet CRC mismatch"; return st; }

    int W = (int)hdr.width, H = (int)hdr.height, qp = (int)hdr.qp;
    if (out_width)  *out_width = W;
    if (out_height) *out_height = H;
    if (out_type)   *out_type = (dcvc_frame_type_t)hdr.frame_type;

    /* Rebuild the pipeline pair when geometry or qp changes. An intra frame
     * also implicitly resyncs the DPB. */
    if (dec->cur_W != W || dec->cur_H != H || dec->cur_qp != qp) {
        st = ensure_pipelines(sess, &dec->pp, H, W, qp);
        if (st != DCVC_OK) return st;
        dec->cur_W = W; dec->cur_H = H; dec->cur_qp = qp;
    }
    st = dec->ensure_ref(H, W);
    if (st != DCVC_OK) return st;

    DcvcCpuStatus cst;
    if (hdr.frame_type == DCVC_FRAME_INTRA) {
        cst = dcvc_cpu_intra_pipeline_decode(dec->pp.intra, payload, hdr.payload_len, dec->ref);
        dec->have_ref = 1;
        dec->ref_W = W; dec->ref_H = H;
    } else {
        if (!dec->have_ref || dec->ref_W != W || dec->ref_H != H) {
            sess->last_error = "inter frame decoded without a matching reference";
            return DCVC_ERR_FORMAT;
        }
        cst = dcvc_cpu_inter_pipeline_decode(dec->pp.inter, payload, hdr.payload_len,
                                             dec->ref, dec->ref);
        dec->ref_W = W; dec->ref_H = H;
    }
    if (cst != DCVC_CPU_OK) {
        if (cst == DCVC_CPU_ERR_ENTROPY)
            sess->last_error = "entropy stream desynchronized; resync at next intra frame";
        return map_status(cst);
    }

    std::memcpy(out_frame, dec->ref, (size_t)W * H * 3 * sizeof(float));
    return DCVC_OK;
}

/* ========================================================================= *
 *  Packet helpers                                                            *
 * ========================================================================= */
DCVC_API dcvc_status_t DCVC_CALL dcvc_packet_probe(const uint8_t* packet, size_t size,
                                                   int* out_width, int* out_height,
                                                   int* out_qp,
                                                   dcvc_frame_type_t* out_type) {
    dcvc_container_header hdr;
    dcvc_status_t st = dcvc_container_parse(packet, size, &hdr, nullptr);
    if (st != DCVC_OK) return st;
    if (out_width)  *out_width  = (int)hdr.width;
    if (out_height) *out_height = (int)hdr.height;
    if (out_qp)     *out_qp     = (int)hdr.qp;
    if (out_type)   *out_type   = (dcvc_frame_type_t)hdr.frame_type;
    return DCVC_OK;
}

DCVC_API void DCVC_CALL dcvc_packet_free(uint8_t* packet) { std::free(packet); }

} /* extern "C" */
