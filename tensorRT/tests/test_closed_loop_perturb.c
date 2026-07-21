// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
//
// Stability test: add uniform noise to y (simulating CPU ONNX vs TRT error)
// before the 4x AR encoder, then decode with the same native prior.  Verifies
// that rANS does not crash and that encoder/decoder remain self-consistent.
//
// Usage: test_closed_loop_perturb <asset_dir> <plugin_dir> [noise_mag] [seed]

#include "dcvc_rt.h"
#include "rans_c.h"
#include "trt_engine.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32h(float f_val) {
    uint32_t x; memcpy(&x, &f_val, 4);
    uint32_t s = (x >> 31) & 1; int e = ((x >> 23) & 0xff) - 127 + 15; uint32_t m = (x >> 13) & 0x3ff;
    if (e <= 0) { m |= 0x400; while (e < 0) { m >>= 1; e++; } e = 0; }
    if (e >= 31) { e = 31; m = 0; }
    return (uint16_t)((s << 15) | (e << 10) | m);
}
static float h2f(uint16_t h){uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,f;
    if(e==0){if(m==0)f=s<<31;else{int ex=-1;while(!(m&0x400)){m<<=1;ex--;}m&=0x3ff;e=127+ex-14;f=(s<<31)|(e<<23)|(m<<13);}}
    else if(e==31)f=(s<<31)|(0xff<<23)|(m<<13);
    else f=(s<<31)|((e+127-15)<<23)|(m<<13);
    float r;memcpy(&r,&f,4);return r;}

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
    o->elems=1; for(int i=0;i<4;i++) o->elems*=(size_t)o->dims[i];
    o->i32=o->fp16=o->fp32=NULL;
    if(strstr(hdr,"i4")){o->i32=malloc(o->elems*4);size_t g=fread(o->i32,4,o->elems,f);fclose(f);return g==o->elems?0:-1;}
    if(strstr(hdr,"f2")){o->fp16=malloc(o->elems*2);size_t g=fread(o->fp16,2,o->elems,f);fclose(f);return g==o->elems?0:-1;}
    o->fp32=malloc(o->elems*4); size_t g=fread(o->fp32,4,o->elems,f);fclose(f);return g==o->elems?0:-1;
}
static void npy_to_dev_fp16(const Npy* n, void* d) {
    if(n->fp16) cudaMemcpy(d,n->fp16,n->elems*2,cudaMemcpyHostToDevice);
    else { uint16_t*t=malloc(n->elems*2);
        for(size_t i=0;i<n->elems;i++) t[i]=f32h(n->fp32[i]);
        cudaMemcpy(d,t,n->elems*2,cudaMemcpyHostToDevice); free(t);
    }
}

typedef void (*build_idx_dec_fn)(const void*, uint8_t*, float, float, float, float, int, cudaStream_t);
typedef void (*build_idx_enc_fn)(const void*, const void*, int16_t*, float, float, float, float, int, cudaStream_t);
typedef void (*restore_y4x_fn)(const void*, const void*, const void*, void*, int, int, cudaStream_t);
typedef void (*sp4x_fn)(const void*, void*, int, cudaStream_t);
typedef void (*mul_fn)(const void*, const void*, void*, int, cudaStream_t);
typedef void (*pmask_yq_fn)(const void*, const void*, const void*, const void*, void*, float, int, cudaStream_t);

#define N_CH 256

int main(int argc, char** argv) {
    const char* asset_dir = argc>1 ? argv[1] : "tensorRT/assets";
    const char* plugin_dir = argc>2 ? argv[2] : "tensorRT/build/plugin_demo";
    float noise_mag = argc>3 ? (float)atof(argv[3]) : 0.007f;
    int seed = argc>4 ? atoi(argv[4]) : 42;
    srand(seed);

    char path[600], decode_dir[512], eng_dir[512];
    snprintf(decode_dir,sizeof(decode_dir),"%s/decode",asset_dir);
    snprintf(eng_dir,sizeof(eng_dir),"%s/engines",asset_dir);

    snprintf(path,sizeof(path),"%s/libdcvc_kernels.so",plugin_dir);
    void* kh = dlopen(path, RTLD_NOW|RTLD_GLOBAL);
    if(!kh){fprintf(stderr,"Cannot load kernels: %s\n",dlerror());return 1;}
    build_idx_dec_fn p_build_idx_dec = (build_idx_dec_fn)dlsym(kh,"dcvc_k_build_index_dec");
    build_idx_enc_fn p_build_idx_enc = (build_idx_enc_fn)dlsym(kh,"dcvc_k_build_index_enc");
    restore_y4x_fn p_restore = (restore_y4x_fn)dlsym(kh,"dcvc_k_restore_y_4x");
    sp4x_fn p_sp4x = (sp4x_fn)dlsym(kh,"dcvc_k_single_part_writing_4x");
    mul_fn p_mul = (mul_fn)dlsym(kh,"dcvc_k_elem_mul");
    pmask_yq_fn p_pmask_yq = (pmask_yq_fn)dlsym(kh,"dcvc_k_process_mask_yq");

    Npy gcdf_npy, glen_npy, goff_npy, zcdf_npy, zlen_npy, zoff_npy;
    snprintf(path,sizeof(path),"%s/gaussian_cdf.npy",decode_dir); read_npy(path,&gcdf_npy);
    snprintf(path,sizeof(path),"%s/gaussian_cdf_length.npy",decode_dir); read_npy(path,&glen_npy);
    snprintf(path,sizeof(path),"%s/gaussian_offset.npy",decode_dir); read_npy(path,&goff_npy);
    snprintf(path,sizeof(path),"%s/bitest_cdf.npy",decode_dir); read_npy(path,&zcdf_npy);
    snprintf(path,sizeof(path),"%s/bitest_cdf_length.npy",decode_dir); read_npy(path,&zlen_npy);
    snprintf(path,sizeof(path),"%s/bitest_offset.npy",decode_dir); read_npy(path,&zoff_npy);
    int g_cdf_num=gcdf_npy.dims[0];

    Npy pf_npy;
    snprintf(path,sizeof(path),"%s/golden_params_fusion_ec0.npy",decode_dir);
    if(read_npy(path,&pf_npy)!=0){fprintf(stderr,"no params_fusion\n");return 1;}
    int pf_C=pf_npy.dims[1], pf_H=pf_npy.dims[2], pf_W=pf_npy.dims[3];
    int HW=pf_H*pf_W;

    void* d_pf; cudaMalloc(&d_pf, pf_npy.elems*2);
    npy_to_dev_fp16(&pf_npy, d_pf);
    uint16_t* pf_host = malloc(pf_npy.elems*2);
    cudaMemcpy(pf_host, d_pf, pf_npy.elems*2, cudaMemcpyDeviceToHost);

    void* d_qenc; cudaMalloc(&d_qenc, HW*2);
    void* d_qdec; cudaMalloc(&d_qdec, HW*2);
    uint16_t *qenc_h=malloc(HW*2), *qdec_h=malloc(HW*2);
    for(int i=0;i<HW;i++){
        float qe=1.0f/(1.0f+expf(-h2f(pf_host[0*HW+i])))*1.5f+0.5f;
        float qd=1.0f/(1.0f+expf(-h2f(pf_host[1*HW+i])))*1.5f+0.5f;
        qenc_h[i]=f32h(qe); qdec_h[i]=f32h(qd);
    }
    cudaMemcpy(d_qenc,qenc_h,HW*2,cudaMemcpyHostToDevice);
    cudaMemcpy(d_qdec,qdec_h,HW*2,cudaMemcpyHostToDevice);

    void* d_scales0; cudaMalloc(&d_scales0, N_CH*HW*2);
    void* d_means0;  cudaMalloc(&d_means0,  N_CH*HW*2);
    cudaMemcpy(d_scales0, pf_host+2*HW,         N_CH*HW*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_means0,  pf_host+(2+N_CH)*HW,  N_CH*HW*2, cudaMemcpyHostToDevice);

    DcvcRtStatus st;
    snprintf(path,sizeof(path),"%s/y_spatial_prior_reduction.engine",eng_dir);
    DcvcTrtEngine* eng_yspr = dcvc_trt_engine_load(path,plugin_dir,&st);
    if(!eng_yspr){fprintf(stderr,"yspr load fail\n");return 1;}
    int32_t pf_dims[4]={1,pf_C,pf_H,pf_W};
    dcvc_trt_engine_set_shape(eng_yspr,"in0",pf_dims,4);
    dcvc_trt_engine_set_addr(eng_yspr,"in0",d_pf);
    const char* yspr_out=dcvc_trt_engine_tensor_name(eng_yspr,dcvc_trt_engine_num_io(eng_yspr)-1);
    int32_t rd[8]; int rnd;
    dcvc_trt_engine_get_shape(eng_yspr,yspr_out,rd,&rnd,8);
    void* d_common; cudaMalloc(&d_common, rd[1]*HW*2);
    dcvc_trt_engine_set_addr(eng_yspr,yspr_out,d_common);
    dcvc_trt_engine_execute(eng_yspr,NULL);
    cudaDeviceSynchronize();

    DcvcTrtEngine* eng_yspa[4]; eng_yspa[0]=NULL;
    for(int i=1;i<=3;i++){
        snprintf(path,sizeof(path),"%s/y_spatial_prior_adaptor_%d.engine",eng_dir,i);
        eng_yspa[i]=dcvc_trt_engine_load(path,plugin_dir,&st);
        if(!eng_yspa[i]){fprintf(stderr,"yspa_%d load fail\n",i);return 1;}
    }
    snprintf(path,sizeof(path),"%s/y_spatial_prior.engine",eng_dir);
    DcvcTrtEngine* eng_ysp=dcvc_trt_engine_load(path,plugin_dir,&st);
    if(!eng_ysp){fprintf(stderr,"ysp load fail\n");return 1;}

    void* d_masks[4];
    for(int i=0;i<4;i++){
        Npy mn; snprintf(path,sizeof(path),"%s/mask_%d.npy",decode_dir,i);
        read_npy(path,&mn); cudaMalloc(&d_masks[i],mn.elems*2); npy_to_dev_fp16(&mn,d_masks[i]);
    }

    void* d_y; cudaMalloc(&d_y, N_CH*HW*2);
    {
        uint16_t* yh=malloc(N_CH*HW*2);
        for(int c=0;c<N_CH;c++)
            for(int hw=0;hw<HW;hw++)
                yh[c*HW+hw]=f32h(sinf(c*0.137f+hw*0.071f)*1.5f);
        cudaMemcpy(d_y,yh,N_CH*HW*2,cudaMemcpyHostToDevice);
        free(yh);
    }

    /* ================= PERTURB y (simulating CPU ONNX vs TRT error) ================= */
    {
        uint16_t* yh=malloc(N_CH*HW*2);
        cudaMemcpy(yh,d_y,N_CH*HW*2,cudaMemcpyDeviceToHost);
        float maxv=0, sumv=0; int n=N_CH*HW;
        for(int i=0;i<n;i++){
            float v=h2f(yh[i]);
            float noise = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * noise_mag;
            yh[i] = f32h(v + noise);
            float ad = fabsf(noise); sumv += ad; if(ad>maxv) maxv=ad;
        }
        cudaMemcpy(d_y,yh,N_CH*HW*2,cudaMemcpyHostToDevice);
        free(yh);
        printf("Perturbed y: noise_mag=%.4f  actual avg_abs=%.6f  max_abs=%.6f\n", noise_mag, sumv/n, maxv);
    }

    void* d_ysc; cudaMalloc(&d_ysc, N_CH*HW*2);
    {
        uint16_t* sc=malloc(N_CH*HW*2), *yh=malloc(N_CH*HW*2);
        cudaMemcpy(yh,d_y,N_CH*HW*2,cudaMemcpyDeviceToHost);
        for(int c=0;c<N_CH;c++)
            for(int hw=0;hw<HW;hw++)
                sc[c*HW+hw]=f32h(h2f(yh[c*HW+hw])*h2f(qenc_h[hw]));
        cudaMemcpy(d_ysc,sc,N_CH*HW*2,cudaMemcpyHostToDevice);
        free(sc);free(yh);
    }

    int R=N_CH/4;
    void *d_smask, *d_sr, *d_yq_full, *d_yq_w, *d_packed, *d_yhat_step, *d_cat, *d_sp_out;
    cudaMalloc(&d_smask,   N_CH*HW*2);
    cudaMalloc(&d_sr,      R*HW*2);
    cudaMalloc(&d_yq_full, N_CH*HW*2);
    cudaMalloc(&d_yq_w,    R*HW*2);
    cudaMalloc(&d_packed,  R*HW*2);
    cudaMalloc(&d_yhat_step,N_CH*HW*2);
    cudaMalloc(&d_cat,     2*N_CH*HW*2);
    cudaMalloc(&d_sp_out,  2*N_CH*HW*2);
    int16_t* packed_h=malloc(R*HW*2);
    uint8_t* indexes_h=malloc(R*HW);
    void* d_indexes; cudaMalloc(&d_indexes, R*HW);

    float scale_min=0.11f, scale_max=16.0f;
    float log_scale_min=logf(scale_min);
    float log_step_recip=1.0f/((logf(scale_max)-logf(scale_min))/127.0f);

    uint16_t (*enc_sr)[16384]=malloc(4*16384*2);
    uint16_t (*dec_sr)[16384]=malloc(4*16384*2);
    int16_t  (*enc_sym)[16384]=malloc(4*16384*2);
    int8_t   (*dec_sym)[16384]=malloc(4*16384);

    /* =========================  ENCODE  ============================ */
    DcvcRansEncoder* enc = dcvc_rans_encoder_create();
    dcvc_rans_encoder_set_two(enc, 0);
    dcvc_rans_encoder_add_cdf(enc, zcdf_npy.i32, zcdf_npy.dims[0], zcdf_npy.dims[1], zlen_npy.i32, zoff_npy.i32);
    int g_cdf_idx = dcvc_rans_encoder_add_cdf(enc, gcdf_npy.i32, g_cdf_num, gcdf_npy.dims[1], glen_npy.i32, goff_npy.i32);

    void* d_yhat_enc; cudaMalloc(&d_yhat_enc, N_CH*HW*2);
    cudaMemset(d_yhat_enc, 0, N_CH*HW*2);

    for(int round=0; round<4; round++){
        void *curr_scales=d_scales0, *curr_means=d_means0;
        if(round>0){
            cudaMemcpy(d_cat, d_yhat_enc, N_CH*HW*2, cudaMemcpyDeviceToDevice);
            cudaMemcpy((char*)d_cat+N_CH*HW*2, d_common, N_CH*HW*2, cudaMemcpyDeviceToDevice);
            int32_t cat_dims[4]={1,2*N_CH,pf_H,pf_W};
            dcvc_trt_engine_set_shape(eng_yspa[round],"in0",cat_dims,4);
            dcvc_trt_engine_set_addr(eng_yspa[round],"in0",d_cat);
            const char* ao=dcvc_trt_engine_tensor_name(eng_yspa[round],dcvc_trt_engine_num_io(eng_yspa[round])-1);
            int32_t ad[8]; int and_;
            dcvc_trt_engine_get_shape(eng_yspa[round],ao,ad,&and_,8);
            dcvc_trt_engine_set_addr(eng_yspa[round],ao,d_sp_out);
            dcvc_trt_engine_execute(eng_yspa[round],NULL);
            dcvc_trt_engine_set_shape(eng_ysp,"in0",ad,4);
            dcvc_trt_engine_set_addr(eng_ysp,"in0",d_sp_out);
            const char* yo=dcvc_trt_engine_tensor_name(eng_ysp,dcvc_trt_engine_num_io(eng_ysp)-1);
            dcvc_trt_engine_get_shape(eng_ysp,yo,rd,&rnd,8);
            dcvc_trt_engine_set_addr(eng_ysp,yo,d_cat);
            dcvc_trt_engine_execute(eng_ysp,NULL);
            cudaDeviceSynchronize();
            curr_scales=d_cat;
            curr_means=(void*)((char*)d_cat+N_CH*HW*2);
        }
        p_mul(curr_scales, d_masks[round], d_smask, N_CH*HW, 0);
        p_sp4x(d_smask, d_sr, R*HW, 0);
        p_pmask_yq(d_ysc, curr_scales, curr_means, d_masks[round], d_yq_full, -1.0f, N_CH*HW, 0);
        p_sp4x(d_yq_full, d_yq_w, R*HW, 0);
        p_build_idx_enc(d_yq_w, d_sr, (int16_t*)d_packed, scale_min, scale_max,
                        log_scale_min, log_step_recip, R*HW, 0);
        cudaMemcpy(packed_h, d_packed, R*HW*2, cudaMemcpyDeviceToHost);
        dcvc_rans_encoder_encode_y(enc, packed_h, R*HW, g_cdf_idx);
        cudaMemcpy(enc_sr[round], d_sr, R*HW*2, cudaMemcpyDeviceToHost);        memcpy(enc_sym[round], packed_h, R*HW*2);
        p_restore(d_yq_w, curr_means, d_masks[round], d_yhat_step, R*HW, N_CH*HW, 0);
        {
            uint16_t *sf=malloc(N_CH*HW*2), *stmp=malloc(N_CH*HW*2);
            cudaMemcpy(sf,d_yhat_enc,N_CH*HW*2,cudaMemcpyDeviceToHost);
            cudaMemcpy(stmp,d_yhat_step,N_CH*HW*2,cudaMemcpyDeviceToHost);
            for(int i=0;i<N_CH*HW;i++) sf[i]=f32h(h2f(sf[i])+h2f(stmp[i]));
            cudaMemcpy(d_yhat_enc,sf,N_CH*HW*2,cudaMemcpyHostToDevice);
            free(sf);free(stmp);
        }
        int nz=0; for(int i=0;i<R*HW;i++) if(enc_sym[round][i]>>8) nz++;
        int oob=0; for(int i=0;i<R*HW;i++){int s=enc_sym[round][i]>>8; if(s<-9||s>9)oob++;}
        printf("  enc round %d: %d/%d nonzero, %d out-of-CDF-range\n",round,nz,R*HW,oob);
    }
    dcvc_rans_encoder_flush(enc);
    uint8_t* stream=NULL; size_t stream_n=0;
    dcvc_rans_encoder_get_stream(enc,&stream,&stream_n);
    printf("encode done: bitstream %zu bytes\n", stream_n);

    {
        uint16_t *yh=malloc(N_CH*HW*2);
        cudaMemcpy(yh,d_yhat_enc,N_CH*HW*2,cudaMemcpyDeviceToHost);
        for(int c=0;c<N_CH;c++) for(int hw=0;hw<HW;hw++)
            yh[c*HW+hw]=f32h(h2f(yh[c*HW+hw])*h2f(qdec_h[hw]));
        cudaMemcpy(d_yhat_enc,yh,N_CH*HW*2,cudaMemcpyHostToDevice);
        free(yh);
    }

    /* =========================  DECODE  ============================ */
    DcvcRansDecoder* dec = dcvc_rans_decoder_create();
    dcvc_rans_decoder_set_two(dec, 0);
    dcvc_rans_decoder_add_cdf(dec, zcdf_npy.i32, zcdf_npy.dims[0], zcdf_npy.dims[1], zlen_npy.i32, zoff_npy.i32);
    dcvc_rans_decoder_add_cdf(dec, gcdf_npy.i32, g_cdf_num, gcdf_npy.dims[1], glen_npy.i32, goff_npy.i32);
    dcvc_rans_decoder_set_stream(dec, stream, stream_n);

    void* d_yhat_dec; cudaMalloc(&d_yhat_dec, N_CH*HW*2);
    cudaMemset(d_yhat_dec, 0, N_CH*HW*2);

    int all_sym_match=1;
    for(int round=0; round<4; round++){
        void *curr_scales=d_scales0, *curr_means=d_means0;
        if(round>0){
            cudaMemcpy(d_cat, d_yhat_dec, N_CH*HW*2, cudaMemcpyDeviceToDevice);
            cudaMemcpy((char*)d_cat+N_CH*HW*2, d_common, N_CH*HW*2, cudaMemcpyDeviceToDevice);
            int32_t cat_dims[4]={1,2*N_CH,pf_H,pf_W};
            dcvc_trt_engine_set_shape(eng_yspa[round],"in0",cat_dims,4);
            dcvc_trt_engine_set_addr(eng_yspa[round],"in0",d_cat);
            const char* ao=dcvc_trt_engine_tensor_name(eng_yspa[round],dcvc_trt_engine_num_io(eng_yspa[round])-1);
            int32_t ad[8]; int and_;
            dcvc_trt_engine_get_shape(eng_yspa[round],ao,ad,&and_,8);
            dcvc_trt_engine_set_addr(eng_yspa[round],ao,d_sp_out);
            dcvc_trt_engine_execute(eng_yspa[round],NULL);
            dcvc_trt_engine_set_shape(eng_ysp,"in0",ad,4);
            dcvc_trt_engine_set_addr(eng_ysp,"in0",d_sp_out);
            const char* yo=dcvc_trt_engine_tensor_name(eng_ysp,dcvc_trt_engine_num_io(eng_ysp)-1);
            dcvc_trt_engine_get_shape(eng_ysp,yo,rd,&rnd,8);
            dcvc_trt_engine_set_addr(eng_ysp,yo,d_cat);
            dcvc_trt_engine_execute(eng_ysp,NULL);
            cudaDeviceSynchronize();
            curr_scales=d_cat;
            curr_means=(void*)((char*)d_cat+N_CH*HW*2);
        }
        p_mul(curr_scales, d_masks[round], d_smask, N_CH*HW, 0);
        p_sp4x(d_smask, d_sr, R*HW, 0);
        cudaMemcpy(dec_sr[round], d_sr, R*HW*2, cudaMemcpyDeviceToHost);
        p_build_idx_dec(d_sr, (uint8_t*)d_indexes, scale_min, scale_max,
                        log_scale_min, log_step_recip, R*HW, 0);
        cudaMemcpy(indexes_h, d_indexes, R*HW, cudaMemcpyDeviceToHost);
        dcvc_rans_decoder_decode_y(dec, indexes_h, R*HW, g_cdf_idx);
        int8_t* syms=NULL; size_t sym_n=0;
        dcvc_rans_decoder_get_symbols(dec, (int8_t**)&syms, &sym_n);        memcpy(dec_sym[round], syms, sym_n<R*HW?sym_n:R*HW);
        {
            int match=0, firstbad=-1;
            for(int i=0;i<R*HW;i++){
                int esym=enc_sym[round][i]>>8;
                if(esym==(int)dec_sym[round][i]) match++; else if(firstbad<0) firstbad=i;
            }
            printf("  dec round %d: sym match=%d/%d (%.1f%%)",round,match,R*HW,100.0*match/(R*HW));
            if(firstbad>=0) printf(" first_diff@%d (enc=%d dec=%d)",firstbad,enc_sym[round][firstbad]>>8,(int)dec_sym[round][firstbad]);
            printf("\n");
            if(match!=R*HW) all_sym_match=0;
        }
        {
            uint16_t* yqh=malloc(R*HW*2);
            for(int i=0;i<R*HW;i++) yqh[i]=f32h((float)dec_sym[round][i]);
            cudaMemcpy(d_yq_w, yqh, R*HW*2, cudaMemcpyHostToDevice);
            free(yqh);
        }
        p_restore(d_yq_w, curr_means, d_masks[round], d_yhat_step, R*HW, N_CH*HW, 0);
        {
            uint16_t *sf=malloc(N_CH*HW*2), *stmp=malloc(N_CH*HW*2);
            cudaMemcpy(sf,d_yhat_dec,N_CH*HW*2,cudaMemcpyDeviceToHost);
            cudaMemcpy(stmp,d_yhat_step,N_CH*HW*2,cudaMemcpyDeviceToHost);
            for(int i=0;i<N_CH*HW;i++) sf[i]=f32h(h2f(sf[i])+h2f(stmp[i]));
            cudaMemcpy(d_yhat_dec,sf,N_CH*HW*2,cudaMemcpyHostToDevice);
            free(sf);free(stmp);
        }
    }

    {
        uint16_t *yh=malloc(N_CH*HW*2);
        cudaMemcpy(yh,d_yhat_dec,N_CH*HW*2,cudaMemcpyDeviceToHost);
        for(int c=0;c<N_CH;c++) for(int hw=0;hw<HW;hw++)
            yh[c*HW+hw]=f32h(h2f(yh[c*HW+hw])*h2f(qdec_h[hw]));
        cudaMemcpy(d_yhat_dec,yh,N_CH*HW*2,cudaMemcpyHostToDevice);
        free(yh);
    }

    /* =======================  COMPARE  ============================= */
    {
        uint16_t *eh=malloc(N_CH*HW*2), *dh=malloc(N_CH*HW*2);
        cudaMemcpy(eh,d_yhat_enc,N_CH*HW*2,cudaMemcpyDeviceToHost);
        cudaMemcpy(dh,d_yhat_dec,N_CH*HW*2,cudaMemcpyDeviceToHost);
        int exact=0, near=0, n=N_CH*HW; float mx=0;
        for(int i=0;i<n;i++){
            uint16_t a=eh[i], b=dh[i];
            if(a==b) exact++;
            float df=fabsf(h2f(a)-h2f(b));
            if(df<0.5f) near++;
            if(df>mx) mx=df;
        }
        printf("y_hat enc-vs-dec: BIT-EXACT=%d/%d (%.1f%%)  near(<0.5)=%d/%d  max_abs=%.3e\n",
               exact,n,100.0*exact/n,near,n,mx);
        printf("symbol round-trip: %s\n", all_sym_match?"ALL 4 ROUNDS EXACT MATCH":"MISMATCH");
        if(exact==n && all_sym_match){
            printf("\n*** PERTURBED CLOSED LOOP: PASS (rANS survived y noise) ***\n");
        } else {
            printf("\n*** PERTURBED CLOSED LOOP: FAIL ***\n");
        }
        free(eh);free(dh);
    }

    free(stream); free(packed_h); free(indexes_h); cudaFree(d_indexes); free(pf_host); free(qenc_h); free(qdec_h);
    free(enc_sr); free(dec_sr); free(enc_sym); free(dec_sym);
    dcvc_rans_encoder_destroy(enc); dcvc_rans_decoder_destroy(dec);
    dcvc_trt_engine_destroy(eng_yspr); dcvc_trt_engine_destroy(eng_ysp);
    for(int i=1;i<=3;i++) dcvc_trt_engine_destroy(eng_yspa[i]);
    cudaFree(d_pf);cudaFree(d_qenc);cudaFree(d_qdec);cudaFree(d_scales0);cudaFree(d_means0);
    cudaFree(d_common);cudaFree(d_y);cudaFree(d_ysc);
    cudaFree(d_smask);cudaFree(d_sr);cudaFree(d_yq_full);cudaFree(d_yq_w);cudaFree(d_yhat_step);
    cudaFree(d_packed);
    cudaFree(d_cat);cudaFree(d_sp_out);cudaFree(d_yhat_enc);cudaFree(d_yhat_dec);
    for(int i=0;i<4;i++) cudaFree(d_masks[i]);
    return 0;
}
