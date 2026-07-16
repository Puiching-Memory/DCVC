// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
//
// Test: rANS decode z from bitstream using C rANS + exported CDF tables.
// Verifies the decoded z_hat matches the golden z_hat.

#include "rans_c.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int dims[4]; size_t elems; int32_t* i32; uint16_t* fp16; float* fp32; } Npy;
static int read_npy(const char* path, Npy* o) {
    FILE* f=fopen(path,"rb"); if(!f) return -1;
    char m[6]; fread(m,1,6,f); if(memcmp(m,"\x93NUMPY",6)){fclose(f);return -1;}
    uint8_t mj,mn; fread(&mj,1,1,f); fread(&mn,1,1,f);
    uint32_t hl=0; if(mj==1){uint16_t h16;fread(&h16,2,1,f);hl=h16;} else fread(&hl,4,1,f);
    char hdr[512]; if(hl>=sizeof(hdr)){fclose(f);return -1;} fread(hdr,1,hl,f); hdr[hl]=0;
    o->dims[0]=o->dims[1]=o->dims[2]=o->dims[3]=1; int nd=0;
    char* sp=strstr(hdr,"shape"); if(sp){sp=strchr(sp,'('); if(sp){sp++; while(*sp&&*sp!=')'&&nd<4){if(*sp>='0'&&*sp<='9'){o->dims[nd++]=atoi(sp);while(*sp>='0'&&*sp<='9')sp++;}sp++;}}}
    while(nd<4) o->dims[nd++]=1;
    o->elems=1; for(int i=0;i<4;i++) o->elems*=o->dims[i];
    o->i32=NULL; o->fp16=NULL; o->fp32=NULL;
    if(strstr(hdr,"i4")||strstr(hdr,"i8")||strstr(hdr,"int")) {o->i32=malloc(o->elems*4);size_t g=fread(o->i32,4,o->elems,f);fclose(f);return g==o->elems?0:-1;}
    if(strstr(hdr,"f2")) {o->fp16=malloc(o->elems*2);size_t g=fread(o->fp16,2,o->elems,f);fclose(f);return g==o->elems?0:-1;}
    o->fp32=malloc(o->elems*4); size_t g=fread(o->fp32,4,o->elems,f);fclose(f);return g==o->elems?0:-1;
}

int main(int argc, char** argv) {
    const char* decode_dir = argc>1 ? argv[1] : "native/assets/decode";
    const char* golden_dir = argc>2 ? argv[2] : "native/assets/golden";
    char path[512];
    int qp=20, z_ch=128, z_h=4, z_w=4;

    // Load bitstream
    snprintf(path,sizeof(path),"%s/i_256x256_qp20_bitstream.bin",golden_dir);
    FILE* bf=fopen(path,"rb"); if(!bf){fprintf(stderr,"no bitstream\n");return 1;}
    fseek(bf,0,SEEK_END); long bs_size=ftell(bf); fseek(bf,0,SEEK_SET);
    uint8_t* bs=malloc(bs_size); fread(bs,1,bs_size,bf); fclose(bf);
    printf("Bitstream: %ld bytes\n", bs_size);

    // Load z CDF tables
    Npy cdf_npy, cdf_len_npy, offset_npy;
    snprintf(path,sizeof(path),"%s/bitest_cdf.npy",decode_dir);
    if(read_npy(path,&cdf_npy)!=0){fprintf(stderr,"no z cdf\n");return 1;}
    snprintf(path,sizeof(path),"%s/bitest_cdf_length.npy",decode_dir);
    read_npy(path,&cdf_len_npy);
    snprintf(path,sizeof(path),"%s/bitest_offset.npy",decode_dir);
    read_npy(path,&offset_npy);
    printf("z CDF: %zu entries, cdf_length=%zu, offset=%zu\n", cdf_npy.elems, cdf_len_npy.elems, offset_npy.elems);

    // CDF dimensions: cdf_npy.dims = [qp_num*channel, max_length+2]
    // per_vector_size = cdf_npy.dims[1]
    int per_vector_size = cdf_npy.dims[1];
    int cdf_num = cdf_npy.dims[0];
    printf("  cdf_num=%d, per_vector_size=%d\n", cdf_num, per_vector_size);

    // Create rANS decoder
    DcvcRansDecoder* dec = dcvc_rans_decoder_create();
    dcvc_rans_decoder_set_two(dec, 1);  // ec_part=1

    // Add CDF: the C API expects (cdfs_flat, cdf_num, per_vector_size, cdf_sizes, offsets)
    int z_cdf_idx = dcvc_rans_decoder_add_cdf(dec, cdf_npy.i32, cdf_num, per_vector_size,
                                               cdf_len_npy.i32, offset_npy.i32);
    printf("  CDF added: index=%d\n", z_cdf_idx);

    // Set bitstream
    dcvc_rans_decoder_set_stream(dec, bs, bs_size);

    // Decode z: total_size = channel * H * W = 128*4*4 = 2048
    int total_size = z_ch * z_h * z_w;
    int start_offset = qp * z_ch;  // qp=20, z_ch=128 → 2560
    int per_channel_size = z_h * z_w;  // 4*4 = 16
    printf("  decode_z: total=%d, start_offset=%d, per_channel=%d\n", total_size, start_offset, per_channel_size);

    dcvc_rans_decoder_decode_z(dec, total_size, z_cdf_idx, start_offset, per_channel_size);

    // Get decoded symbols
    int8_t* symbols=NULL; size_t sym_n=0;
    dcvc_rans_decoder_get_symbols(dec, (int8_t**)&symbols, &sym_n);
    printf("  Decoded %zu symbols\n", sym_n);

    // Load golden z_hat and compare
    snprintf(path,sizeof(path),"%s/golden_z_hat.npy",decode_dir);
    Npy zhat_npy;
    if(read_npy(path,&zhat_npy)==0 && zhat_npy.fp16) {
        float* gold=malloc(zhat_npy.elems*4);
        for(size_t i=0;i<zhat_npy.elems;i++){
            uint16_t h=zhat_npy.fp16[i];
            uint32_t sign=(h>>15)&1,expo=(h>>10)&0x1f,mant=h&0x3ff,f;
            if(expo==0){if(mant==0)f=sign<<31;else{int e=-1;while(!(mant&0x400)){mant<<=1;e--;}mant&=0x3ff;expo=127+e-14;f=(sign<<31)|(expo<<23)|(mant<<13);}}
            else if(expo==31)f=(sign<<31)|(0xff<<23)|(mant<<13);
            else f=(sign<<31)|((expo+127-15)<<23)|(mant<<13);
            memcpy(&gold[i],&f,4);
        }
        int n = zhat_npy.elems < sym_n ? zhat_npy.elems : sym_n;
        int match=0;
        for(int i=0;i<n;i++){
            float expected = gold[i];
            float decoded = (float)symbols[i];
            if(fabsf(expected - decoded) < 0.5f) match++;
        }
        printf("  z_hat match: %d/%d (%.1f%%)\n", match, n, 100.0*match/n);
        printf("  First 10 golden: ");
        for(int i=0;i<10&&i<n;i++) printf("%.0f ", gold[i]);
        printf("\n  First 10 decoded: ");
        for(int i=0;i<10&&i<n;i++) printf("%d ", symbols[i]);
        printf("\n");
        free(gold);
    }

    dcvc_rans_decoder_destroy(dec);
    free(bs);
    free(cdf_npy.i32); free(cdf_len_npy.i32); free(offset_npy.i32);
    if(zhat_npy.fp16) free(zhat_npy.fp16);
    return 0;
}
