// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
//
// Full 4x spatial prior decode in C.
// Pipeline: params_fusion → separate_prior → 4 autoregressive rounds
//           (rANS y decode + TRT spatial_prior engines + CUDA kernels)
//           → y_hat
//
// Uses golden params_fusion as input. Verifies y_hat vs golden_y_hat.

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

/* ---- FP16 helpers ---- */
static uint16_t f32h(float f_val) {
    uint32_t x; memcpy(&x, &f_val, 4);
    uint32_t s = (x >> 31) & 1; int e = ((x >> 23) & 0xff) - 127 + 15; uint32_t m = (x >> 13) & 0x3ff;
    if (e <= 0) { m |= 0x400; while (e < 0) { m >>= 1; e++; } e = 0; }
    if (e >= 31) { e = 31; m = 0; }
    return (s << 15) | (e << 10) | m;
}

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
    o->i32=o->fp16=o->fp32=NULL;
    if(strstr(hdr,"i4")){o->i32=malloc(o->elems*4);size_t g=fread(o->i32,4,o->elems,f);fclose(f);return g==o->elems?0:-1;}
    if(strstr(hdr,"f2")){o->fp16=malloc(o->elems*2);size_t g=fread(o->fp16,2,o->elems,f);fclose(f);return g==o->elems?0:-1;}
    o->fp32=malloc(o->elems*4); size_t g=fread(o->fp32,4,o->elems,f);fclose(f);return g==o->elems?0:-1;
}

static void npy_to_dev_fp16(const Npy* n, void* d) {
    if(n->fp16) cudaMemcpy(d,n->fp16,n->elems*2,cudaMemcpyHostToDevice);
    else { uint16_t*t=malloc(n->elems*2);
        for(size_t i=0;i<n->elems;i++){float f=n->fp32[i]; uint32_t x; memcpy(&x,&f,4);
            uint32_t s=(x>>31)&1; int e=((x>>23)&0xff)-127+15; uint32_t m=(x>>13)&0x3ff;
            if(e<=0){m|=0x400;while(e<0){m>>=1;e++;}e=0;} if(e>=31){e=31;m=0;}
            t[i]=(s<<15)|(e<<10)|m;}
        cudaMemcpy(d,t,n->elems*2,cudaMemcpyHostToDevice); free(t);
    }
}

static float h2f(uint16_t h){uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,f;
    if(e==0){if(m==0)f=s<<31;else{int ex=-1;while(!(m&0x400)){m<<=1;ex--;}m&=0x3ff;e=127+ex-14;f=(s<<31)|(e<<23)|(m<<13);}}
    else if(e==31)f=(s<<31)|(0xff<<23)|(m<<13);
    else f=(s<<31)|((e+127-15)<<23)|(m<<13);
    float r;memcpy(&r,&f,4);return r;}

/* ---- Kernel function pointers ---- */
typedef void (*build_idx_fn)(const void*, uint8_t*, float, float, float, float, int, cudaStream_t);
typedef void (*restore_y4x_fn)(const void*, const void*, const void*, void*, int, int, cudaStream_t);
typedef void (*sp4x_fn)(const void*, void*, int, cudaStream_t);
typedef void (*mul_fn)(const void*, const void*, void*, int, cudaStream_t);

int main(int argc, char** argv) {
    const char* asset_dir = argc>1 ? argv[1] : "native/assets";
    const char* plugin_dir = argc>2 ? argv[2] : "native/build/plugin_demo";
    char path[600], decode_dir[512], eng_dir[512];
    snprintf(decode_dir,sizeof(decode_dir),"%s/decode",asset_dir);
    snprintf(eng_dir,sizeof(eng_dir),"%s/engines",asset_dir);

    /* Load kernel .so */
    snprintf(path,sizeof(path),"%s/libdcvc_kernels.so",plugin_dir);
    void* kh = dlopen(path, RTLD_NOW|RTLD_GLOBAL);
    if(!kh){fprintf(stderr,"Cannot load kernels: %s\n",dlerror());return 1;}
    build_idx_fn p_build_idx = (build_idx_fn)dlsym(kh,"dcvc_k_build_index_dec");
    restore_y4x_fn p_restore = (restore_y4x_fn)dlsym(kh,"dcvc_k_restore_y_4x");
    sp4x_fn p_sp4x = (sp4x_fn)dlsym(kh,"dcvc_k_single_part_writing_4x");
    mul_fn p_mul = (mul_fn)dlsym(kh,"dcvc_k_elem_mul");
    printf("Kernels: build_idx=%p restore=%p sp4x=%p mul=%p\n",p_build_idx,p_restore,p_sp4x,p_mul);

    /* Load Gaussian CDF */
    Npy gcdf_npy, glen_npy, goff_npy;
    snprintf(path,sizeof(path),"%s/gaussian_cdf.npy",decode_dir);
    read_npy(path,&gcdf_npy);
    snprintf(path,sizeof(path),"%s/gaussian_cdf_length.npy",decode_dir);
    read_npy(path,&glen_npy);
    snprintf(path,sizeof(path),"%s/gaussian_offset.npy",decode_dir);
    read_npy(path,&goff_npy);
    int g_cdf_num = gcdf_npy.dims[0], g_pvs = gcdf_npy.dims[1];
    printf("Gaussian CDF: %d CDFs, per_vector=%d\n",g_cdf_num,g_pvs);

    /* Load z (BitEstimator) CDF first — index 0 */
    Npy zcdf_npy, zlen_npy, zoff_npy;
    snprintf(path,sizeof(path),"%s/bitest_cdf.npy",decode_dir);
    read_npy(path,&zcdf_npy);
    snprintf(path,sizeof(path),"%s/bitest_cdf_length.npy",decode_dir);
    read_npy(path,&zlen_npy);
    snprintf(path,sizeof(path),"%s/bitest_offset.npy",decode_dir);
    read_npy(path,&zoff_npy);

    /* Create rANS decoder: add z CDF (index 0), then Gaussian CDF (index 1) */
    DcvcRansDecoder* rans = dcvc_rans_decoder_create();
    dcvc_rans_decoder_set_two(rans, 0);
    int z_cdf_idx = dcvc_rans_decoder_add_cdf(rans, zcdf_npy.i32, zcdf_npy.dims[0], zcdf_npy.dims[1],
                                               zlen_npy.i32, zoff_npy.i32);
    int g_cdf_idx = dcvc_rans_decoder_add_cdf(rans, gcdf_npy.i32, g_cdf_num, g_pvs,
                                               glen_npy.i32, goff_npy.i32);
    printf("Gaussian CDF added: idx=%d\n", g_cdf_idx);

    /* Load bitstream and set stream (z already decoded in real pipeline, but rANS
       state must advance past z. For this test we use a fresh bitstream.) */
    snprintf(path,sizeof(path),"%s/../golden/i_256x256_qp20_ec0_bitstream.bin",decode_dir);
    FILE* bf=fopen(path,"rb");
    if(bf){
        fseek(bf,0,SEEK_END); long bs=ftell(bf); fseek(bf,0,SEEK_SET);
        uint8_t* bsm=malloc(bs); fread(bsm,1,bs,bf); fclose(bf);
        dcvc_rans_decoder_set_stream(rans, bsm, bs);
        /* Skip z decode (advance rANS state). z is 128*4*4=2048 symbols */
        dcvc_rans_decoder_decode_z(rans, 2048, z_cdf_idx, 20*128, 16);
        printf("z decode skipped (2048 symbols)\n");
    }

    /* Load golden params_fusion */
    Npy pf_npy;
    snprintf(path,sizeof(path),"%s/golden_params_fusion_ec0.npy",decode_dir);
    if(read_npy(path,&pf_npy)!=0){fprintf(stderr,"no params_fusion\n");return 1;}
    int pf_C = pf_npy.dims[1], pf_H = pf_npy.dims[2], pf_W = pf_npy.dims[3];
    printf("params_fusion: [%d,%d,%d,%d]\n",pf_npy.dims[0],pf_C,pf_H,pf_W);

    /* Copy to device */
    void* d_pf; cudaMalloc(&d_pf, pf_npy.elems*2);
    npy_to_dev_fp16(&pf_npy, d_pf);

    /* ---- separate_prior: I-frame path ---- */
    /* params[:, :2] → q_enc, q_dec via sigmoid*1.5+0.5 */
    /* params[:, 2:] → scales, means (chunk 2) */
    int N_ch = 256;  /* y channels */
    int C_q = 2;     /* q channels */
    int C_sm = N_ch; /* scales/means channels */
    int HW = pf_H * pf_W;

    /* Extract q_enc, q_dec, scales, means on host (they're small: 514*16*16) */
    uint16_t* pf_host = malloc(pf_npy.elems*2);
    cudaMemcpy(pf_host, d_pf, pf_npy.elems*2, cudaMemcpyDeviceToHost);

    /* q_dec = sigmoid(params[:,1]) * 1.5 + 0.5, shape [1,1,16,16] */
    void* d_qdec; cudaMalloc(&d_qdec, 1*HW*2);
    {
        uint16_t* tmp=malloc(HW*2);
        for(int i=0;i<HW;i++){
            float q=h2f(pf_host[1*HW+i]); /* channel 1 */
            q=1.0f/(1.0f+expf(-q))*1.5f+0.5f;
            tmp[i]=f32h(q);
        }
        cudaMemcpy(d_qdec,tmp,HW*2,cudaMemcpyHostToDevice);
        free(tmp);
    }

    /* scales = params[:,2:258], means = params[:,258:514] */
    void* d_scales; cudaMalloc(&d_scales, C_sm*HW*2);
    void* d_means; cudaMalloc(&d_means, C_sm*HW*2);
    cudaMemcpy(d_scales, pf_host+2*HW, C_sm*HW*2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, pf_host+(2+C_sm)*HW, C_sm*HW*2, cudaMemcpyHostToDevice);

    /* ---- y_spatial_prior_reduction: Conv2d(514,256,1) applied to full params ---- */
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
    printf("yspr: →[%d,%d,%d,%d]\n",rd[0],rd[1],rd[2],rd[3]);

    void* d_common; cudaMalloc(&d_common, rd[1]*HW*2); /* common_params reduced = N_ch */
    dcvc_trt_engine_set_addr(eng_yspr,yspr_out,d_common);
    dcvc_trt_engine_execute(eng_yspr,NULL);
    cudaDeviceSynchronize();
    printf("yspr executed\n");

    /* Load spatial prior engines */
    DcvcTrtEngine* eng_yspa[4]; eng_yspa[0]=NULL;
    DcvcTrtEngine* eng_ysp=NULL;
    for(int i=1;i<=3;i++){
        snprintf(path,sizeof(path),"%s/y_spatial_prior_adaptor_%d.engine",eng_dir,i);
        eng_yspa[i]=dcvc_trt_engine_load(path,plugin_dir,&st);
        if(!eng_yspa[i]){fprintf(stderr,"yspa_%d load fail\n",i);return 1;}
    }
    snprintf(path,sizeof(path),"%s/y_spatial_prior.engine",eng_dir);
    eng_ysp=dcvc_trt_engine_load(path,plugin_dir,&st);
    if(!eng_ysp){fprintf(stderr,"ysp load fail\n");return 1;}
    printf("Loaded 3 adaptors + ysp\n");

    /* Load masks */
    void* d_masks[4];
    for(int i=0;i<4;i++){
        Npy mask_npy;
        snprintf(path,sizeof(path),"%s/mask_%d.npy",decode_dir,i);
        read_npy(path,&mask_npy);
        d_masks[i]=malloc(mask_npy.elems*2);
        cudaMalloc(&d_masks[i],mask_npy.elems*2);
        npy_to_dev_fp16(&mask_npy,d_masks[i]);
    }

    /* ---- 4x decode loop ---- */
    /* y_hat_so_far starts at zero */
    void* d_yhat_sf; cudaMalloc(&d_yhat_sf, N_ch*HW*2);
    cudaMemset(d_yhat_sf, 0, N_ch*HW*2);

    /* Workspace for intermediates */
    void* d_scales_masked; cudaMalloc(&d_scales_masked, N_ch*HW*2);
    void* d_scales_r; cudaMalloc(&d_scales_r, (N_ch/4)*HW*2);
    void* d_yq_r; cudaMalloc(&d_yq_r, (N_ch/4)*HW*2);
    void* d_indexes; cudaMalloc(&d_indexes, (N_ch/4)*HW);
    void* d_yhat_step; cudaMalloc(&d_yhat_step, N_ch*HW*2);
    void* d_cat; cudaMalloc(&d_cat, (N_ch + N_ch)*HW*2); /* y_hat + common */
    void* d_sp_out; cudaMalloc(&d_sp_out, N_ch*2*HW*2); /* adaptor+sp output: 2*N_ch */

    float scale_min=0.11f, scale_max=16.0f;
    float log_scale_min=logf(scale_min);
    float log_step_recip=1.0f/((logf(scale_max)-logf(scale_min))/127.0f);

    for(int round=0; round<4; round++){
        void* curr_scales = d_scales;
        void* curr_means = d_means;

        if(round>0){
            /* params = cat(y_hat_so_far, common) → [1, 2*N_ch, H, W] */
            /* Copy y_hat_so_far to first half of d_cat, common to second half */
            cudaMemcpy(d_cat, d_yhat_sf, N_ch*HW*2, cudaMemcpyDeviceToDevice);
            cudaMemcpy((char*)d_cat+N_ch*HW*2, d_common, N_ch*HW*2, cudaMemcpyDeviceToDevice);

            /* Run adaptor_{round} */
            int32_t cat_dims[4]={1,2*N_ch,pf_H,pf_W};
            dcvc_trt_engine_set_shape(eng_yspa[round],"in0",cat_dims,4);
            dcvc_trt_engine_set_addr(eng_yspa[round],"in0",d_cat);
            const char* yspa_out=dcvc_trt_engine_tensor_name(eng_yspa[round],dcvc_trt_engine_num_io(eng_yspa[round])-1);
            int32_t ad[8]; int and;
            dcvc_trt_engine_get_shape(eng_yspa[round],yspa_out,ad,&and,8);
            dcvc_trt_engine_set_addr(eng_yspa[round],yspa_out,d_sp_out); /* separate buffer */
            dcvc_trt_engine_execute(eng_yspa[round],NULL);
            cudaDeviceSynchronize();

            /* Run y_spatial_prior on adaptor output → [1, 2*N_ch, H, W] = scales+means */
            dcvc_trt_engine_set_shape(eng_ysp,"in0",ad,4);
            dcvc_trt_engine_set_addr(eng_ysp,"in0",d_sp_out);
            const char* ysp_out=dcvc_trt_engine_tensor_name(eng_ysp,dcvc_trt_engine_num_io(eng_ysp)-1);
            dcvc_trt_engine_get_shape(eng_ysp,ysp_out,rd,&rnd,8);
            /* ysp output also 512 ch → write to d_cat (which was input, now free) */
            dcvc_trt_engine_set_addr(eng_ysp,ysp_out,d_cat);
            dcvc_trt_engine_execute(eng_ysp,NULL);
            cudaDeviceSynchronize();

            /* Split: first N_ch = scales, last N_ch = means */
            curr_scales = d_cat;
            curr_means = (void*)((char*)d_cat + N_ch*HW*2);
        }

        /* Compare curr_scales against golden scales for this round */
        {
            char gpath[600];
            snprintf(gpath,sizeof(gpath),"%s/ec0_round%d_scales.npy",decode_dir,round);
            Npy gsc; if(read_npy(gpath,&gsc)==0 && gsc.fp16){
                uint16_t* s_h=malloc(N_ch*HW*2);
                cudaMemcpy(s_h,curr_scales,N_ch*HW*2,cudaMemcpyDeviceToHost);
                int smatch=0; float smax=0;
                for(int i=0;i<N_ch*HW;i++){float d=fabsf(h2f(s_h[i])-h2f(gsc.fp16[i]));if(d>smax)smax=d;if(d<0.001f)smatch++;}
                printf("  Round %d scales: match=%d/%d max_abs=%.6f\n",round,smatch,N_ch*HW,smax);
                free(s_h);
            }
        }
        /* scales_r = single_part_for_writing_4x(scales * mask_{round}) */
        p_mul(curr_scales, d_masks[round], d_scales_masked, N_ch*HW, 0);
        p_sp4x(d_scales_masked, d_scales_r, (N_ch/4)*HW, 0);

        /* Build indexes via GPU kernel (fp32-log + fp16 intermediate rounding + truncate) */
        p_build_idx(d_scales_r, (uint8_t*)d_indexes, scale_min, scale_max,
                    log_scale_min, log_step_recip, (N_ch/4)*HW, 0);
        uint8_t* indexes_h = malloc((N_ch/4)*HW);
        cudaMemcpy(indexes_h, d_indexes, (N_ch/4)*HW, cudaMemcpyDeviceToHost);

        /* rANS decode y */
        dcvc_rans_decoder_decode_y(rans, indexes_h, (N_ch/4)*HW, g_cdf_idx);
        int8_t* syms=NULL; size_t sym_n=0;
        dcvc_rans_decoder_get_symbols(rans, (int8_t**)&syms, &sym_n);

        /* Compare decoded symbols against golden yq for this round */
        {
            char gpath[600];
            snprintf(gpath,sizeof(gpath),"%s/ec0_round%d_yq.npy",decode_dir,round);
            Npy gyq; if(read_npy(gpath,&gyq)==0 && gyq.fp16){
                int match=0, firstbad=-1;
                for(size_t i=0;i<sym_n && i<(size_t)gyq.elems;i++){
                    int8_t expv=(int8_t)h2f(gyq.fp16[i]);
                    if(syms[i]==expv){match++;} else if(firstbad<0) firstbad=(int)i;
                }
                printf("  Round %d symbols: match=%zu/%zu (%.1f%%)",
                    round,(size_t)match,sym_n,100.0*match/sym_n);
                if(firstbad>=0) printf(" first_diff@%d (got=%d exp=%d)",firstbad,syms[firstbad],(int8_t)h2f(gyq.fp16[firstbad]));
                printf("\n");
            }
        }

        /* Copy decoded y_q to device as FP16 */
        {
            uint16_t* yq_h=malloc((N_ch/4)*HW*2);
            for(int i=0;i<(N_ch/4)*HW;i++) yq_h[i]=f32h((float)syms[i]);
            cudaMemcpy(d_yq_r, yq_h, (N_ch/4)*HW*2, cudaMemcpyHostToDevice);
            free(yq_h);
        }
        free(indexes_h);

        /* restore_y_4x: y_hat_step = (y_q_r broadcast 4x + means) * mask */
        p_restore(d_yq_r, curr_means, d_masks[round], d_yhat_step, (N_ch/4)*HW, N_ch*HW, 0);

        /* y_hat_so_far += y_hat_step */
        /* Use a simple element-wise add kernel or CUDA API */
        /* For simplicity, copy to host and add */
        {
            uint16_t* sf_h=malloc(N_ch*HW*2);
            uint16_t* step_h=malloc(N_ch*HW*2);
            cudaMemcpy(sf_h,d_yhat_sf,N_ch*HW*2,cudaMemcpyDeviceToHost);
            cudaMemcpy(step_h,d_yhat_step,N_ch*HW*2,cudaMemcpyDeviceToHost);
            for(int i=0;i<N_ch*HW;i++)
                sf_h[i]=f32h(h2f(sf_h[i])+h2f(step_h[i]));
            cudaMemcpy(d_yhat_sf,sf_h,N_ch*HW*2,cudaMemcpyHostToDevice);
            free(sf_h);free(step_h);
        }

        /* Debug: print y_hat_so_far stats after this round */
        {
            uint16_t* sf_h=malloc(N_ch*HW*2);
            cudaMemcpy(sf_h,d_yhat_sf,N_ch*HW*2,cudaMemcpyDeviceToHost);
            float mn=1e9f,mx=-1e9f;
            int nonzero=0;
            for(int i=0;i<N_ch*HW;i++){float v=h2f(sf_h[i]); if(v<mn)mn=v; if(v>mx)mx=v; if(v!=0)nonzero++;}
            printf("  Round %d: y_hat_sf range=[%.3f,%.3f] nonzero=%d/%d\n",round,mn,mx,nonzero,N_ch*HW);
            free(sf_h);
        }
    }

    /* y_hat = y_hat_so_far * q_dec */
    {
        uint16_t* yh_h=malloc(N_ch*HW*2);
        uint16_t* qd_h=malloc(HW*2);
        cudaMemcpy(yh_h,d_yhat_sf,N_ch*HW*2,cudaMemcpyDeviceToHost);
        cudaMemcpy(qd_h,d_qdec,HW*2,cudaMemcpyDeviceToHost);
        /* q_dec is [1,1,H,W], broadcast across all 256 channels */
        for(int c=0;c<N_ch;c++){
            for(int hw=0;hw<HW;hw++){
                float v=h2f(yh_h[c*HW+hw])*h2f(qd_h[hw]);
                yh_h[c*HW+hw]=f32h(v);
            }
        }
        cudaMemcpy(d_yhat_sf,yh_h,N_ch*HW*2,cudaMemcpyHostToDevice);
        free(yh_h);free(qd_h);
    }

    /* Compare to golden y_hat */
    Npy gyhat_npy;
    snprintf(path,sizeof(path),"%s/golden_y_hat_ec0.npy",decode_dir);
    if(read_npy(path,&gyhat_npy)==0 && gyhat_npy.fp16){
        uint16_t* got=malloc(N_ch*HW*2);
        cudaMemcpy(got,d_yhat_sf,N_ch*HW*2,cudaMemcpyDeviceToHost);
        float mx=0,mr=0; int match=0;
        int n=N_ch*HW;
        for(int i=0;i<n;i++){
            float d=fabsf(h2f(got[i])-h2f(gyhat_npy.fp16[i]));
            if(d>mx)mx=d;
            if(fabsf(h2f(gyhat_npy.fp16[i]))>mr)mr=fabsf(h2f(gyhat_npy.fp16[i]));
            if(d<0.5f)match++;
        }
        printf("\n=== y_hat result ===\n");
        printf("max_abs=%.4e rel=%.4f match=%d/%d (%.1f%%)\n",mx,mx/(mr+1e-6f),match,n,100.0f*match/n);
        free(got);
    }

    /* Cleanup */
    dcvc_rans_decoder_destroy(rans);
    dcvc_trt_engine_destroy(eng_yspr);
    for(int i=1;i<=3;i++)dcvc_trt_engine_destroy(eng_yspa[i]);
    dcvc_trt_engine_destroy(eng_ysp);
    cudaFree(d_pf);cudaFree(d_qdec);cudaFree(d_scales);cudaFree(d_means);
    cudaFree(d_common);cudaFree(d_scales_masked);cudaFree(d_scales_r);
    cudaFree(d_yq_r);cudaFree(d_indexes);cudaFree(d_yhat_step);cudaFree(d_cat);cudaFree(d_sp_out);
    cudaFree(d_yhat_sf);
    for(int i=0;i<4;i++)cudaFree(d_masks[i]);
    free(pf_host);
    return 0;
}
