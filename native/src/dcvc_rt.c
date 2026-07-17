#include "dcvc_rt.h"
#include "dcvc_rt_internal.h"
#include "dcvc_stream.h"
#include "dcvc_bitstream.h"
#include "dcvc_color.h"
#include "dcvc_dpb.h"
#include "dcvc_trt_runner.h"
#include "rans_c.h"
#include "dcvc_ar_codec.h"
#include "dcvc_intra_pipeline.h"
#include "dcvc_inter_pipeline.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

const char* dcvc_rt_version(void)
{
    return DCVC_RT_VERSION_STR;
}

const char* dcvc_rt_status_string(DcvcRtStatus st)
{
    switch (st) {
    case DCVC_RT_OK: return "ok";
    case DCVC_RT_ERR_INVALID_ARG: return "invalid_arg";
    case DCVC_RT_ERR_IO: return "io";
    case DCVC_RT_ERR_NO_ENGINE: return "no_engine";
    case DCVC_RT_ERR_CUDA: return "cuda";
    case DCVC_RT_ERR_TRT: return "trt";
    case DCVC_RT_ERR_ENTROPY: return "entropy";
    case DCVC_RT_ERR_BITSTREAM: return "bitstream";
    case DCVC_RT_ERR_OOM: return "oom";
    case DCVC_RT_ERR_UNSUPPORTED: return "unsupported";
    default: return "internal";
    }
}

DcvcRtStatus dcvc_rt_config_init(DcvcRtConfig* cfg)
{
    if (!cfg) return DCVC_RT_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->qp = 32;
    cfg->reset_interval = 64;
    cfg->device_id = 0;
    cfg->format = DCVC_RT_FMT_YUV420P;
    return DCVC_RT_OK;
}

void dcvc_rt_packet_free(DcvcRtPacket* pkt)
{
    if (!pkt) return;
    free(pkt->data);
    pkt->data = NULL;
    pkt->size = 0;
}

void dcvc_rt_frame_free_planes(DcvcRtFrame* frame)
{
    if (!frame) return;
    /* Only free planes we allocated (decode path sets ownership via non-const data). */
    free((void*)frame->data[0]);
    free((void*)frame->data[1]);
    free((void*)frame->data[2]);
    frame->data[0] = frame->data[1] = frame->data[2] = NULL;
}

struct DcvcRtEncoder {
    DcvcRtConfig cfg;
    int pad_w;
    int pad_h;
    int sps_id;
    int sps_written;
    int frame_idx;
    DcvcDpb dpb;
    DcvcTrtRunner* trt;
    DcvcRansEncoder* rans;
    DcvcArCodec* ar_codec;  /* intra AR codec (256ch / 4-pass) */
    DcvcArCodec* ar_inter; /* inter AR codec (128ch / 2-pass) */
    DcvcInterPipeline* inter_pl;
    void* d_image;         /* device FP16 [1,3,padH,padW] */
    void* d_ref_feature;   /* device FP16 [1,256,padH/8,padW/8] ref feature */
    void* d_dec_feature;   /* device FP16 [1,256,padH/8,padW/8] decoder feature */
    void* d_ref_pixels;    /* device FP16 [1,3,padH,padW] ref pixels (P-after-I) */
    int have_ref_feature;  /* 1 once a decoder feature is available as next ref */
    void* d_xhat;          /* device FP16 [1,3,padH,padW] reconstruction */
    float* ycbcr_f32;
    uint16_t* ycbcr_f16;
    /* GPU kernel for FP16→FP32 output conversion */
    void* d_xf32;            /* device FP32 [1,3,padH,padW] for kernel output */
    void* kernel_so;
    void (*k_fp16_to_f32)(const void*, float*, int, cudaStream_t);
};

struct DcvcRtDecoder {
    DcvcRtConfig cfg;
    int pad_w;
    int pad_h;
    DcvcSps sps;
    int have_sps;
    DcvcDpb dpb;
    DcvcTrtRunner* trt;
    DcvcRansDecoder* rans;
    DcvcArCodec* ar_codec;  /* intra AR codec (256ch / 4-pass) */
    DcvcArCodec* ar_inter; /* inter AR codec (128ch / 2-pass) */
    DcvcInterPipeline* inter_pl;
    void* d_xhat;          /* device FP16 [1,3,padH,padW] reconstruction */
    void* d_ref_feature;   /* device FP16 [1,256,padH/8,padW/8] ref feature */
    void* d_dec_feature;   /* device FP16 [1,256,padH/8,padW/8] decoder feature */
    void* d_ref_pixels;    /* device FP16 [1,3,padH,padW] ref pixels (P-after-I) */
    int have_ref_feature;  /* 1 once a decoder feature is available as next ref */
    float* ycbcr_f32;
    uint16_t* ycbcr_f16;
    /* GPU kernel for FP16→FP32 output conversion */
    void* d_xf32;            /* device FP32 [1,3,padH,padW] for kernel output */
    void* kernel_so;
    void (*k_fp16_to_f32)(const void*, float*, int, cudaStream_t);
};

static int prepare_input_f16(DcvcRtEncoder* enc, const DcvcRtFrame* in)
{
    size_t n = (size_t)3 * enc->pad_h * enc->pad_w;
    memset(enc->ycbcr_f32, 0, n * sizeof(float));

    if (in->ycbcr444_fp16) {
        memcpy(enc->ycbcr_f16, in->ycbcr444_fp16, n * sizeof(uint16_t));
        return 0;
    }

    if (in->format == DCVC_RT_FMT_YUV420P) {
        if (!in->data[0] || !in->data[1] || !in->data[2]) return -1;
        /* Convert original resolution then replicate-pad into pad buffer */
        float* tmp = (float*)malloc((size_t)3 * in->height * in->width * sizeof(float));
        if (!tmp) return -1;
        dcvc_yuv420_to_ycbcr444_f32(in->data[0], in->stride[0] ? in->stride[0] : in->width,
                                    in->data[1], in->stride[1] ? in->stride[1] : in->width / 2,
                                    in->data[2], in->stride[2] ? in->stride[2] : in->width / 2,
                                    in->width, in->height, tmp);
        for (int c = 0; c < 3; c++) {
            for (int j = 0; j < enc->pad_h; j++) {
                int js = j < in->height ? j : in->height - 1;
                for (int i = 0; i < enc->pad_w; i++) {
                    int is = i < in->width ? i : in->width - 1;
                    enc->ycbcr_f32[c * enc->pad_h * enc->pad_w + j * enc->pad_w + i] =
                        tmp[c * in->height * in->width + js * in->width + is];
                }
            }
        }
        free(tmp);
    } else if (in->format == DCVC_RT_FMT_RGB24) {
        if (!in->data[0]) return -1;
        float* tmp = (float*)malloc((size_t)3 * in->height * in->width * sizeof(float));
        if (!tmp) return -1;
        dcvc_rgb_to_ycbcr444_f32(in->data[0], in->width, in->height, tmp);
        for (int c = 0; c < 3; c++) {
            for (int j = 0; j < enc->pad_h; j++) {
                int js = j < in->height ? j : in->height - 1;
                for (int i = 0; i < enc->pad_w; i++) {
                    int is = i < in->width ? i : in->width - 1;
                    enc->ycbcr_f32[c * enc->pad_h * enc->pad_w + j * enc->pad_w + i] =
                        tmp[c * in->height * in->width + js * in->width + is];
                }
            }
        }
        free(tmp);
    } else {
        return -1;
    }
    dcvc_f32_to_f16(enc->ycbcr_f32, enc->ycbcr_f16, n);
    return 0;
}

DcvcRtEncoder* dcvc_rt_encoder_create(const DcvcRtConfig* cfg, DcvcRtStatus* out_st)
{
    if (!cfg || cfg->width <= 0 || cfg->height <= 0) {
        if (out_st) *out_st = DCVC_RT_ERR_INVALID_ARG;
        return NULL;
    }
    DcvcRtEncoder* enc = (DcvcRtEncoder*)calloc(1, sizeof(DcvcRtEncoder));
    if (!enc) {
        if (out_st) *out_st = DCVC_RT_ERR_OOM;
        return NULL;
    }
    enc->cfg = *cfg;
    enc->pad_w = dcvc_rt_align_up(cfg->width, DCVC_RT_PAD_ALIGN);
    enc->pad_h = dcvc_rt_align_up(cfg->height, DCVC_RT_PAD_ALIGN);
    enc->sps_id = 0;
    if (dcvc_dpb_init(&enc->dpb, enc->pad_w, enc->pad_h, cfg->reset_interval) != 0) {
        free(enc);
        if (out_st) *out_st = DCVC_RT_ERR_OOM;
        return NULL;
    }
    DcvcRtStatus st = DCVC_RT_OK;
    enc->trt = dcvc_trt_runner_create(cfg->asset_dir, cfg->device_id, &st);
    enc->rans = dcvc_rans_encoder_create();
    size_t n = (size_t)3 * enc->pad_h * enc->pad_w;
    enc->ycbcr_f32 = (float*)calloc(n, sizeof(float));
    enc->ycbcr_f16 = (uint16_t*)calloc(n, sizeof(uint16_t));
    /* Device buffers for pipeline */
    {
        size_t dn = (size_t)3 * enc->pad_h * enc->pad_w * 2;
        cudaMalloc(&enc->d_image, dn);
        cudaMalloc(&enc->d_xhat, dn);
    }
    if (!enc->trt || !enc->rans || !enc->ycbcr_f32 || !enc->ycbcr_f16 || !enc->d_image || !enc->d_xhat) {
        dcvc_rt_encoder_destroy(enc);
        if (out_st) *out_st = DCVC_RT_ERR_OOM;
        return NULL;
    }
    int ec = dcvc_rt_ec_part_for_res(cfg->width, cfg->height);
    dcvc_rans_encoder_set_two(enc->rans, ec);

    /* Create production AR codec when TRT engines are available */
    if (dcvc_trt_runner_has_engine(enc->trt, DCVC_ENG_INTRA_SPATIAL_PRIOR)) {
        char plugin_dir[640];
        const char* ad = cfg->asset_dir ? cfg->asset_dir : "native/assets";
        snprintf(plugin_dir, sizeof(plugin_dir), "%s/../build/plugin_demo", ad);
        int passes = 4;  /* intra I-frame AR: always 256ch / 4-pass */
        enc->ar_codec = dcvc_ar_codec_create(ad, plugin_dir, passes, &st);
        /* Non-fatal: codec is optional, encode_frame works without it */
    }

    /* Inter (P-frame) pipeline + AR codec; non-fatal if engines absent */
    {
        const char* ad2 = cfg->asset_dir ? cfg->asset_dir : "native/assets";
        char plugin_dir2[640];
        snprintf(plugin_dir2, sizeof(plugin_dir2), "%s/../build/plugin_demo", ad2);
        enc->inter_pl = dcvc_inter_pipeline_create(ad2, plugin_dir2, &st);
        enc->ar_inter = dcvc_ar_codec_create(ad2, plugin_dir2, 2, &st);
    }
    {
        size_t fn = (size_t)256 * (enc->pad_h / 8) * (enc->pad_w / 8) * 2;
        size_t pn = (size_t)3 * enc->pad_h * enc->pad_w * 2;
        cudaMalloc(&enc->d_ref_feature, fn);
        cudaMalloc(&enc->d_dec_feature, fn);
        cudaMalloc(&enc->d_ref_pixels, pn);
    }

    if (out_st) *out_st = DCVC_RT_OK;
    return enc;
}

void dcvc_rt_encoder_destroy(DcvcRtEncoder* enc)
{
    if (!enc) return;
    dcvc_dpb_free(&enc->dpb);
    dcvc_trt_runner_destroy(enc->trt);
    dcvc_rans_encoder_destroy(enc->rans);
    dcvc_ar_codec_destroy(enc->ar_codec);
    dcvc_ar_codec_destroy(enc->ar_inter);
    dcvc_inter_pipeline_destroy(enc->inter_pl);
    cudaFree(enc->d_image);
    cudaFree(enc->d_xhat);
    cudaFree(enc->d_ref_feature);
    cudaFree(enc->d_dec_feature);
    cudaFree(enc->d_ref_pixels);
    free(enc->ycbcr_f32);
    free(enc->ycbcr_f16);
    free(enc);
}

/*
 * Full NN encode requires TensorRT engines built by tools/build_engines.py.
 * Until engines are present, we still emit a valid container: SPS (once) +
 * I/P NAL with empty payload marker for pipeline integration tests.
 * When engines exist, this path runs analysis → AR prior → rANS → mux.
 */
DcvcRtStatus dcvc_rt_encode_frame(DcvcRtEncoder* enc, const DcvcRtFrame* in, DcvcRtPacket* out)
{
    if (!enc || !in || !out) return DCVC_RT_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    if (prepare_input_f16(enc, in) != 0) return DCVC_RT_ERR_INVALID_ARG;

    int is_i = (enc->frame_idx == 0) || !enc->dpb.has_ref;
    /* P-frame QP shifting: mirrors PyTorch index_map + qp_shift.
       index_map = [0,1,0,2,0,2,0,2], qp_shift = [0,8,4] */
    int effective_qp = enc->cfg.qp;
    if (!is_i) {
        static const int dcvc_index_map[8] = {0, 1, 0, 2, 0, 2, 0, 2};
        static const int dcvc_qp_shift[3]  = {0, 8, 4};
        int fa_idx = dcvc_index_map[enc->frame_idx % 8];
        effective_qp = enc->cfg.qp + dcvc_qp_shift[fa_idx];
    }
    DcvcEngineId analysis = is_i ? DCVC_ENG_INTRA_ANALYSIS : DCVC_ENG_INTER_ANALYSIS;
    int have_nn = dcvc_trt_runner_has_engine(enc->trt, analysis);

    DcvcBitWriter bw;
    dcvc_bw_init(&bw);

    out->sps_written = 0;
    if (!enc->sps_written) {
        DcvcSps sps;
        sps.sps_id = enc->sps_id;
        sps.height = enc->cfg.height;
        sps.width = enc->cfg.width;
        sps.ec_part = dcvc_rt_ec_part_for_res(enc->cfg.width, enc->cfg.height);
        sps.use_ada_i = 1;
        if (dcvc_bw_write_sps(&bw, &sps) != 0) {
            dcvc_bw_free(&bw);
            return DCVC_RT_ERR_BITSTREAM;
        }
        enc->sps_written = 1;
        out->sps_written = 1;
    }

    uint8_t* payload = NULL;
    size_t payload_len = 0;

    if (have_nn && enc->ar_codec && is_i) {
        /* Full intra pipeline: image → analysis → hyper → AR codec → synthesis */
        size_t img_bytes = (size_t)3 * enc->pad_h * enc->pad_w * 2;
        cudaMemcpyAsync(enc->d_image, enc->ycbcr_f16, img_bytes, cudaMemcpyHostToDevice, dcvc_stream());
        DcvcRtStatus est = dcvc_intra_encode(enc->trt, enc->ar_codec, enc->rans,
                                             enc->d_image, enc->pad_h, enc->pad_w,
                                             enc->cfg.qp, &payload, &payload_len, enc->d_xhat);
        if (est != DCVC_RT_OK) {
            dcvc_bw_free(&bw);
            return est;
        }
        enc->have_ref_feature = 0;  /* after I-frame, next P references recon pixels */
    } else if (enc->inter_pl && enc->ar_inter) {
        /* Full inter P-frame pipeline */
        size_t img_bytes = (size_t)3 * enc->pad_h * enc->pad_w * 2;
        cudaMemcpyAsync(enc->d_image, enc->ycbcr_f16, img_bytes, cudaMemcpyHostToDevice, dcvc_stream());
        const void* d_ref_feat = enc->have_ref_feature ? enc->d_dec_feature : NULL;
        const void* d_ref_pix = enc->have_ref_feature ? NULL : enc->d_xhat; /* I recon */
        DcvcRtStatus est = dcvc_inter_encode(enc->inter_pl, enc->ar_inter, enc->rans,
            enc->d_image, d_ref_feat, d_ref_pix, enc->pad_h, enc->pad_w, effective_qp,
            &payload, &payload_len, enc->d_xhat, enc->d_dec_feature);
        if (est != DCVC_RT_OK) {
            dcvc_bw_free(&bw);
            return est;
        }
        enc->have_ref_feature = 1;
    } else if (have_nn) {
        /* Inter frame or no AR codec: fall back to placeholder payload */
        DcvcRtStatus est = dcvc_trt_runner_execute(enc->trt, analysis, NULL, 0, NULL, 0, NULL);
        if (est != DCVC_RT_OK) {
            dcvc_bw_free(&bw);
            return est;
        }
        dcvc_rans_encoder_flush(enc->rans);
        if (dcvc_rans_encoder_get_stream(enc->rans, &payload, &payload_len) != 0) {
            dcvc_bw_free(&bw);
            return DCVC_RT_ERR_ENTROPY;
        }
        dcvc_rans_encoder_reset(enc->rans);
    } else {
        /* Integration mode without engines: empty payload keeps NAL structure valid */
        payload = (uint8_t*)malloc(1);
        if (!payload) {
            dcvc_bw_free(&bw);
            return DCVC_RT_ERR_OOM;
        }
        payload[0] = 0;
        payload_len = 0;
    }

    if (dcvc_bw_write_ip(&bw, is_i, enc->sps_id, effective_qp, payload, payload_len) != 0) {
        free(payload);
        dcvc_bw_free(&bw);
        return DCVC_RT_ERR_BITSTREAM;
    }
    free(payload);

    /* Update DPB with current input as stand-in recon until NN synthesis is wired */
    dcvc_dpb_update(&enc->dpb, NULL, enc->ycbcr_f16);
    enc->frame_idx++;

    out->data = bw.data;
    out->size = bw.size;
    out->frame_type = is_i ? DCVC_RT_FRAME_I : DCVC_RT_FRAME_P;
    out->qp = effective_qp;
    /* bw.data ownership transferred */
    bw.data = NULL;
    dcvc_bw_free(&bw);
    /* Container always succeeds; NN fidelity requires engines under asset_dir. */
    (void)have_nn;
    return DCVC_RT_OK;
}

DcvcRtStatus dcvc_rt_encoder_flush(DcvcRtEncoder* enc, DcvcRtPacket* out)
{
    if (!enc || !out) return DCVC_RT_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));
    return DCVC_RT_OK;
}

DcvcRtDecoder* dcvc_rt_decoder_create(const DcvcRtConfig* cfg, DcvcRtStatus* out_st)
{
    DcvcRtDecoder* dec = (DcvcRtDecoder*)calloc(1, sizeof(DcvcRtDecoder));
    if (!dec) {
        if (out_st) *out_st = DCVC_RT_ERR_OOM;
        return NULL;
    }
    if (cfg) dec->cfg = *cfg;
    else dcvc_rt_config_init(&dec->cfg);

    DcvcRtStatus st = DCVC_RT_OK;
    dec->trt = dcvc_trt_runner_create(dec->cfg.asset_dir, dec->cfg.device_id, &st);
    dec->rans = dcvc_rans_decoder_create();
    if (!dec->trt || !dec->rans) {
        dcvc_rt_decoder_destroy(dec);
        if (out_st) *out_st = DCVC_RT_ERR_OOM;
        return NULL;
    }

    /* Create production AR codec when TRT engines are available */
    if (dcvc_trt_runner_has_engine(dec->trt, DCVC_ENG_INTRA_SPATIAL_PRIOR)) {
        char plugin_dir[640];
        const char* ad = dec->cfg.asset_dir ? dec->cfg.asset_dir : "native/assets";
        snprintf(plugin_dir, sizeof(plugin_dir), "%s/../build/plugin_demo", ad);
        int passes = 4;  /* intra I-frame AR: always 256ch / 4-pass */
        dec->ar_codec = dcvc_ar_codec_create(ad, plugin_dir, passes, &st);
    }

    /* Inter (P-frame) pipeline + AR codec; non-fatal if engines absent */
    {
        const char* ad2 = dec->cfg.asset_dir ? dec->cfg.asset_dir : "native/assets";
        char plugin_dir2[640];
        snprintf(plugin_dir2, sizeof(plugin_dir2), "%s/../build/plugin_demo", ad2);
        dec->inter_pl = dcvc_inter_pipeline_create(ad2, plugin_dir2, &st);
        dec->ar_inter = dcvc_ar_codec_create(ad2, plugin_dir2, 2, &st);
    }

    /* Load FP16→FP32 output kernel */
    {
        const char* ad_k = dec->cfg.asset_dir ? dec->cfg.asset_dir : "native/assets";
        char kpath[640], kdir[640];
        snprintf(kdir, sizeof(kdir), "%s/../build/plugin_demo", ad_k);
        snprintf(kpath, sizeof(kpath), "%s/libdcvc_kernels.so", kdir);
        dec->kernel_so = dlopen(kpath, RTLD_NOW | RTLD_GLOBAL);
        if (dec->kernel_so)
            *(void**)(&dec->k_fp16_to_f32) = dlsym(dec->kernel_so, "dcvc_k_fp16_to_f32_clamp");
    }

    /* Device buffers (dimensions finalized at first SPS; allocate zero-size now) */
    if (out_st) *out_st = DCVC_RT_OK;
    return dec;
}

void dcvc_rt_decoder_destroy(DcvcRtDecoder* dec)
{
    if (!dec) return;
    dcvc_dpb_free(&dec->dpb);
    dcvc_trt_runner_destroy(dec->trt);
    dcvc_rans_decoder_destroy(dec->rans);
    dcvc_ar_codec_destroy(dec->ar_codec);
    dcvc_ar_codec_destroy(dec->ar_inter);
    dcvc_inter_pipeline_destroy(dec->inter_pl);
    if (dec->kernel_so) dlclose(dec->kernel_so);
    cudaFree(dec->d_xhat);
    cudaFree(dec->d_xf32);
    cudaFree(dec->d_ref_feature);
    cudaFree(dec->d_dec_feature);
    cudaFree(dec->d_ref_pixels);
    free(dec->ycbcr_f32);
    free(dec->ycbcr_f16);
    free(dec);
}

DcvcRtStatus dcvc_rt_decode_packet(DcvcRtDecoder* dec, const uint8_t* data, size_t size,
                                   DcvcRtFrame* out)
{
    if (!dec || !data || !out) return DCVC_RT_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    DcvcBitReader br;
    dcvc_br_init(&br, data, size);

    while (br.pos < br.size) {
        int sps_id = 0;
        int nal = dcvc_br_read_header(&br, &sps_id);
        if (nal < 0) return DCVC_RT_ERR_BITSTREAM;

        if (nal == DCVC_NAL_SPS) {
            if (dcvc_br_read_sps_remaining(&br, sps_id, &dec->sps) != 0)
                return DCVC_RT_ERR_BITSTREAM;
            dec->have_sps = 1;
            dec->pad_w = dcvc_rt_align_up(dec->sps.width, DCVC_RT_PAD_ALIGN);
            dec->pad_h = dcvc_rt_align_up(dec->sps.height, DCVC_RT_PAD_ALIGN);
            dcvc_dpb_free(&dec->dpb);
            if (dcvc_dpb_init(&dec->dpb, dec->pad_w, dec->pad_h, dec->cfg.reset_interval) != 0)
                return DCVC_RT_ERR_OOM;
            free(dec->ycbcr_f32);
            free(dec->ycbcr_f16);
            size_t n = (size_t)3 * dec->pad_h * dec->pad_w;
            dec->ycbcr_f32 = (float*)calloc(n, sizeof(float));
            dec->ycbcr_f16 = (uint16_t*)calloc(n, sizeof(uint16_t));
            /* (Re)allocate device buffers now that dimensions are known */
            {
                size_t dn = (size_t)3 * dec->pad_h * dec->pad_w * 2;
                size_t fn = (size_t)256 * (dec->pad_h / 8) * (dec->pad_w / 8) * 2;
                cudaFree(dec->d_xhat);
    cudaFree(dec->d_xf32);       dec->d_xhat = NULL;
                cudaFree(dec->d_ref_feature);dec->d_ref_feature = NULL;
                cudaFree(dec->d_dec_feature);dec->d_dec_feature = NULL;
                cudaFree(dec->d_ref_pixels); dec->d_ref_pixels = NULL;
                cudaMalloc(&dec->d_xhat, dn);
                cudaMalloc(&dec->d_ref_feature, fn);
                cudaMalloc(&dec->d_dec_feature, fn);
                cudaMalloc(&dec->d_ref_pixels, dn);
                cudaFree(dec->d_xf32);
                cudaMalloc(&dec->d_xf32, dn * sizeof(float));
                dec->have_ref_feature = 0;
            }
            dcvc_rans_decoder_set_two(dec->rans, dec->sps.ec_part);
            continue;
        }

        if (nal == DCVC_NAL_I || nal == DCVC_NAL_P) {
            if (!dec->have_sps) return DCVC_RT_ERR_BITSTREAM;
            int qp = 0;
            uint8_t* payload = NULL;
            size_t plen = 0;
            if (dcvc_br_read_ip_remaining(&br, &qp, &payload, &plen) != 0)
                return DCVC_RT_ERR_BITSTREAM;

            int have_nn = dcvc_trt_runner_has_engine(
                dec->trt, nal == DCVC_NAL_I ? DCVC_ENG_INTRA_SYNTHESIS : DCVC_ENG_INTER_SYNTHESIS);

            if (have_nn && dec->ar_codec && plen > 0 && nal == DCVC_NAL_I) {
                /* Full intra pipeline decode */
                DcvcRtStatus est = dcvc_intra_decode(dec->trt, dec->ar_codec, dec->rans,
                                                     payload, plen, dec->pad_h, dec->pad_w,
                                                     qp, dec->d_xhat);
                free(payload);
                if (est != DCVC_RT_OK) return est;
                /* Convert x_hat FP16→clamped FP32 on GPU, then D2H */
                size_t dn = (size_t)3 * dec->pad_h * dec->pad_w;
                if (dec->k_fp16_to_f32) {
                    dec->k_fp16_to_f32(dec->d_xhat, dec->d_xf32, (int)dn, dcvc_stream());
                    cudaMemcpyAsync(dec->ycbcr_f32, dec->d_xf32, dn * sizeof(float),
                                    cudaMemcpyDeviceToHost, dcvc_stream());
                    dcvc_sync();
                } else {
                    uint16_t* xh16 = (uint16_t*)malloc(dn * 2);
                    cudaMemcpyAsync(xh16, dec->d_xhat, dn * 2, cudaMemcpyDeviceToHost, dcvc_stream());
                    dcvc_sync();
                    for (size_t i = 0; i < dn; i++) {
                        uint16_t h = xh16[i];
                        uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, f;
                        if (e == 0) { if (m == 0) f = s << 31; else { int ex = -1; while (!(m & 0x400)) { m <<= 1; ex--; } m &= 0x3ff; e = 127 + ex - 14; f = (s << 31) | (e << 23) | (m << 13); } }
                        else if (e == 31) { f = (s << 31) | (0xff << 23) | (m << 13); }
                        else { f = (s << 31) | ((e + 127 - 15) << 23) | (m << 13); }
                        float v; memcpy(&v, &f, 4);
                        if (v < 0) v = 0; if (v > 1) v = 1;
                        dec->ycbcr_f32[i] = v;
                    }
                    free(xh16);
                }
                dec->have_ref_feature = 0;  /* after I-frame, next P references recon pixels */
            } else if (dec->inter_pl && dec->ar_inter && plen > 0 && nal == DCVC_NAL_P) {
                /* Full inter P-frame pipeline decode */
                const void* d_ref_feat = dec->have_ref_feature ? dec->d_dec_feature : NULL;
                const void* d_ref_pix = dec->have_ref_feature ? NULL : dec->d_xhat;
                DcvcRtStatus est = dcvc_inter_decode(dec->inter_pl, dec->ar_inter, dec->rans,
                    payload, plen, d_ref_feat, d_ref_pix, dec->pad_h, dec->pad_w, qp,
                    dec->d_xhat, dec->d_dec_feature);
                free(payload);
                if (est != DCVC_RT_OK) return est;
                dec->have_ref_feature = 1;
                /* Convert x_hat FP16→clamped FP32 on GPU, then D2H */
                size_t dn = (size_t)3 * dec->pad_h * dec->pad_w;
                if (dec->k_fp16_to_f32) {
                    dec->k_fp16_to_f32(dec->d_xhat, dec->d_xf32, (int)dn, dcvc_stream());
                    cudaMemcpyAsync(dec->ycbcr_f32, dec->d_xf32, dn * sizeof(float),
                                    cudaMemcpyDeviceToHost, dcvc_stream());
                    dcvc_sync();
                } else {
                    uint16_t* xh16 = (uint16_t*)malloc(dn * 2);
                    cudaMemcpyAsync(xh16, dec->d_xhat, dn * 2, cudaMemcpyDeviceToHost, dcvc_stream());
                    dcvc_sync();
                    for (size_t i = 0; i < dn; i++) {
                        uint16_t h = xh16[i];
                        uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, f;
                        if (e == 0) { if (m == 0) f = s << 31; else { int ex = -1; while (!(m & 0x400)) { m <<= 1; ex--; } m &= 0x3ff; e = 127 + ex - 14; f = (s << 31) | (e << 23) | (m << 13); } }
                        else if (e == 31) { f = (s << 31) | (0xff << 23) | (m << 13); }
                        else { f = (s << 31) | ((e + 127 - 15) << 23) | (m << 13); }
                        float v; memcpy(&v, &f, 4);
                        if (v < 0) v = 0; if (v > 1) v = 1;
                        dec->ycbcr_f32[i] = v;
                    }
                    free(xh16);
                }
            } else if (have_nn && plen > 0) {
                dcvc_rans_decoder_set_stream(dec->rans, payload, plen);
                DcvcRtStatus est = dcvc_trt_runner_execute(
                    dec->trt,
                    nal == DCVC_NAL_I ? DCVC_ENG_INTRA_SYNTHESIS : DCVC_ENG_INTER_SYNTHESIS,
                    NULL, 0, NULL, 0, NULL);
                free(payload);
                if (est != DCVC_RT_OK) return est;
            } else {
                /* Without engines: emit mid-gray frame for container tests */
                size_t n = (size_t)3 * dec->pad_h * dec->pad_w;
                for (size_t i = 0; i < n; i++) dec->ycbcr_f32[i] = 0.5f;
                free(payload);
            }

            out->width = dec->sps.width;
            out->height = dec->sps.height;
            out->format = DCVC_RT_FMT_YUV420P;
            int yw = dec->sps.width;
            int yh = dec->sps.height;
            uint8_t* y = (uint8_t*)malloc((size_t)yw * yh);
            uint8_t* u = (uint8_t*)malloc((size_t)(yw / 2) * (yh / 2));
            uint8_t* v = (uint8_t*)malloc((size_t)(yw / 2) * (yh / 2));
            if (!y || !u || !v) {
                free(y); free(u); free(v);
                return DCVC_RT_ERR_OOM;
            }
            /* Crop pad → original */
            float* crop = (float*)malloc((size_t)3 * yh * yw * sizeof(float));
            if (!crop) {
                free(y); free(u); free(v);
                return DCVC_RT_ERR_OOM;
            }
            for (int c = 0; c < 3; c++) {
                for (int j = 0; j < yh; j++) {
                    for (int i = 0; i < yw; i++) {
                        crop[c * yh * yw + j * yw + i] =
                            dec->ycbcr_f32[c * dec->pad_h * dec->pad_w + j * dec->pad_w + i];
                    }
                }
            }
            dcvc_ycbcr444_f32_to_yuv420(crop, yw, yh, y, yw, u, yw / 2, v, yw / 2);
            free(crop);
            out->data[0] = y;
            out->data[1] = u;
            out->data[2] = v;
            out->stride[0] = yw;
            out->stride[1] = yw / 2;
            out->stride[2] = yw / 2;
            dcvc_dpb_update(&dec->dpb, NULL, NULL);
            (void)have_nn;
            return DCVC_RT_OK;
        }

        return DCVC_RT_ERR_BITSTREAM;
    }
    return DCVC_RT_ERR_BITSTREAM;
}
