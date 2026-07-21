// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// C wrapper around RansEncoderLib / RansDecoderLib (no pybind).

#include "rans_c.h"
#include "rans.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

struct DcvcRansEncoder {
    std::shared_ptr<RansEncoderLib> enc0;
    std::shared_ptr<RansEncoderLib> enc1;
    bool use_two{false};
};

struct DcvcRansDecoder {
    std::shared_ptr<RansDecoderLib> dec0;
    std::shared_ptr<RansDecoderLib> dec1;
    bool use_two{false};
};

static std::shared_ptr<std::vector<std::vector<int32_t>>>
make_cdf_vectors(const int32_t* cdfs, int cdf_num, int per_vector_size)
{
    auto vec = std::make_shared<std::vector<std::vector<int32_t>>>(cdf_num);
    for (int i = 0; i < cdf_num; i++) {
        std::vector<int32_t> row(per_vector_size);
        std::memcpy(row.data(), cdfs + i * per_vector_size, sizeof(int32_t) * per_vector_size);
        (*vec)[i] = std::move(row);
    }
    return vec;
}

extern "C" {

DcvcRansEncoder* dcvc_rans_encoder_create(void)
{
    auto* e = new DcvcRansEncoder();
    e->enc0 = std::make_shared<RansEncoderLibMultiThread>();
    e->enc1 = std::make_shared<RansEncoderLibMultiThread>();
    return e;
}

void dcvc_rans_encoder_destroy(DcvcRansEncoder* enc)
{
    delete enc;
}

void dcvc_rans_encoder_set_two(DcvcRansEncoder* enc, int use_two)
{
    if (enc) enc->use_two = use_two != 0;
}

int dcvc_rans_encoder_add_cdf(DcvcRansEncoder* enc,
                              const int32_t* cdfs, int cdf_num, int per_vector_size,
                              const int32_t* cdf_sizes, const int32_t* offsets)
{
    if (!enc || !cdfs || !cdf_sizes || !offsets || cdf_num <= 0) return -1;
    auto vec_cdfs = make_cdf_vectors(cdfs, cdf_num, per_vector_size);
    auto vec_sizes = std::make_shared<std::vector<int32_t>>(cdf_sizes, cdf_sizes + cdf_num);
    auto vec_off = std::make_shared<std::vector<int32_t>>(offsets, offsets + cdf_num);
    int idx = enc->enc0->add_cdf(vec_cdfs, vec_sizes, vec_off);
    enc->enc1->add_cdf(vec_cdfs, vec_sizes, vec_off);
    return idx;
}

void dcvc_rans_encoder_encode_y(DcvcRansEncoder* enc, const int16_t* symbols, int n,
                                int cdf_group_index)
{
    if (!enc || !symbols || n <= 0) return;
    if (enc->use_two) {
        int n0 = n / 2;
        int n1 = n - n0;
        auto v0 = std::make_shared<std::vector<int16_t>>(symbols, symbols + n0);
        auto v1 = std::make_shared<std::vector<int16_t>>(symbols + n0, symbols + n);
        enc->enc0->encode_y(v0, cdf_group_index);
        enc->enc1->encode_y(v1, cdf_group_index);
    } else {
        auto v0 = std::make_shared<std::vector<int16_t>>(symbols, symbols + n);
        enc->enc0->encode_y(v0, cdf_group_index);
    }
}

void dcvc_rans_encoder_encode_z(DcvcRansEncoder* enc, const int8_t* symbols, int n,
                                int cdf_group_index, int start_offset, int per_channel_size)
{
    if (!enc || !symbols || n <= 0) return;
    if (enc->use_two) {
        int n0 = n / 2;
        int n1 = n - n0;
        int channel_half = n0 / per_channel_size;
        auto v0 = std::make_shared<std::vector<int8_t>>(symbols, symbols + n0);
        auto v1 = std::make_shared<std::vector<int8_t>>(symbols + n0, symbols + n);
        enc->enc0->encode_z(v0, cdf_group_index, start_offset, per_channel_size);
        enc->enc1->encode_z(v1, cdf_group_index, start_offset + channel_half, per_channel_size);
    } else {
        auto v0 = std::make_shared<std::vector<int8_t>>(symbols, symbols + n);
        enc->enc0->encode_z(v0, cdf_group_index, start_offset, per_channel_size);
    }
}

void dcvc_rans_encoder_flush(DcvcRansEncoder* enc)
{
    if (!enc) return;
    enc->enc0->flush();
    enc->enc1->flush();
}

int dcvc_rans_encoder_get_stream(DcvcRansEncoder* enc, uint8_t** out, size_t* out_size)
{
    if (!enc || !out || !out_size) return -1;
    *out = nullptr;
    *out_size = 0;

    if (enc->use_two) {
        auto r0 = enc->enc0->get_encoded_stream();
        auto r1 = enc->enc1->get_encoded_stream();
        int nbytes0 = static_cast<int>(r0->size());
        int nbytes1 = static_cast<int>(r1->size());
        int identical_bytes = 0;
        int check_bytes = std::min({nbytes0, nbytes1, 8});
        for (int i = 0; i < check_bytes; i++) {
            if ((*r0)[nbytes0 - 1 - i] != 0) break;
            if ((*r1)[nbytes1 - 1 - i] != 0) break;
            identical_bytes++;
        }
        if (identical_bytes == 0 && (*r0)[nbytes0 - 1] == (*r1)[nbytes1 - 1]) {
            identical_bytes = 1;
        }
        size_t total = static_cast<size_t>(nbytes0 + nbytes1 - identical_bytes);
        auto* buf = static_cast<uint8_t*>(std::malloc(total));
        if (!buf) return -1;
        std::memcpy(buf, r0->data(), nbytes0);
        std::reverse_copy(r1->begin(), r1->end() - identical_bytes, buf + nbytes0);
        *out = buf;
        *out_size = total;
        return 0;
    }

    auto r0 = enc->enc0->get_encoded_stream();
    size_t total = r0->size();
    auto* buf = static_cast<uint8_t*>(std::malloc(total ? total : 1));
    if (!buf) return -1;
    if (total) std::memcpy(buf, r0->data(), total);
    *out = buf;
    *out_size = total;
    return 0;
}

void dcvc_rans_encoder_reset(DcvcRansEncoder* enc)
{
    if (!enc) return;
    enc->enc0->reset();
    enc->enc1->reset();
}

DcvcRansDecoder* dcvc_rans_decoder_create(void)
{
    auto* d = new DcvcRansDecoder();
    d->dec0 = std::make_shared<RansDecoderLibMultiThread>();
    d->dec1 = std::make_shared<RansDecoderLibMultiThread>();
    return d;
}

void dcvc_rans_decoder_destroy(DcvcRansDecoder* dec)
{
    delete dec;
}

void dcvc_rans_decoder_set_two(DcvcRansDecoder* dec, int use_two)
{
    if (dec) dec->use_two = use_two != 0;
}

int dcvc_rans_decoder_add_cdf(DcvcRansDecoder* dec,
                              const int32_t* cdfs, int cdf_num, int per_vector_size,
                              const int32_t* cdf_sizes, const int32_t* offsets)
{
    if (!dec || !cdfs || !cdf_sizes || !offsets || cdf_num <= 0) return -1;
    auto vec_cdfs = make_cdf_vectors(cdfs, cdf_num, per_vector_size);
    auto vec_sizes = std::make_shared<std::vector<int32_t>>(cdf_sizes, cdf_sizes + cdf_num);
    auto vec_off = std::make_shared<std::vector<int32_t>>(offsets, offsets + cdf_num);
    int idx = dec->dec0->add_cdf(vec_cdfs, vec_sizes, vec_off);
    dec->dec1->add_cdf(vec_cdfs, vec_sizes, vec_off);
    return idx;
}

void dcvc_rans_decoder_set_stream(DcvcRansDecoder* dec, const uint8_t* data, size_t size)
{
    if (!dec || !data) return;
    auto s0 = std::make_shared<std::vector<uint8_t>>(data, data + size);
    dec->dec0->set_stream(s0);
    if (dec->use_two) {
        auto s1 = std::make_shared<std::vector<uint8_t>>(size);
        std::reverse_copy(data, data + size, s1->begin());
        dec->dec1->set_stream(s1);
    }
}

void dcvc_rans_decoder_decode_y(DcvcRansDecoder* dec, const uint8_t* indexes, int n,
                                int cdf_group_index)
{
    if (!dec || !indexes || n <= 0) return;
    if (dec->use_two) {
        int n0 = n / 2;
        int n1 = n - n0;
        auto v0 = std::make_shared<std::vector<uint8_t>>(indexes, indexes + n0);
        auto v1 = std::make_shared<std::vector<uint8_t>>(indexes + n0, indexes + n);
        dec->dec0->decode_y(v0, cdf_group_index);
        dec->dec1->decode_y(v1, cdf_group_index);
    } else {
        auto v0 = std::make_shared<std::vector<uint8_t>>(indexes, indexes + n);
        dec->dec0->decode_y(v0, cdf_group_index);
    }
}

void dcvc_rans_decoder_decode_z(DcvcRansDecoder* dec, int total_size, int cdf_group_index,
                                int start_offset, int per_channel_size)
{
    if (!dec || total_size <= 0) return;
    if (dec->use_two) {
        int n0 = total_size / 2;
        int n1 = total_size - n0;
        int channel_half = n0 / per_channel_size;
        dec->dec0->decode_z(n0, cdf_group_index, start_offset, per_channel_size);
        dec->dec1->decode_z(n1, cdf_group_index, start_offset + channel_half, per_channel_size);
    } else {
        dec->dec0->decode_z(total_size, cdf_group_index, start_offset, per_channel_size);
    }
}

int dcvc_rans_decoder_get_symbols(DcvcRansDecoder* dec, int8_t** out, size_t* out_n)
{
    if (!dec || !out || !out_n) return -1;
    *out = nullptr;
    *out_n = 0;
    if (dec->use_two) {
        auto r0 = dec->dec0->get_decoded_tensor();
        auto r1 = dec->dec1->get_decoded_tensor();
        size_t total = r0->size() + r1->size();
        auto* buf = static_cast<int8_t*>(std::malloc(total ? total : 1));
        if (!buf) return -1;
        std::memcpy(buf, r0->data(), r0->size());
        std::memcpy(buf + r0->size(), r1->data(), r1->size());
        *out = buf;
        *out_n = total;
        return 0;
    }
    auto r0 = dec->dec0->get_decoded_tensor();
    size_t total = r0->size();
    auto* buf = static_cast<int8_t*>(std::malloc(total ? total : 1));
    if (!buf) return -1;
    if (total) std::memcpy(buf, r0->data(), total);
    *out = buf;
    *out_n = total;
    return 0;
}

void dcvc_rans_decoder_reset_cdf(DcvcRansDecoder* dec)
{
    if (!dec) return;
    dec->dec0->empty_cdf_buffer();
    dec->dec1->empty_cdf_buffer();
}

}  // extern "C"
