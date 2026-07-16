// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
//
// Verify 4x decode pipeline using Python-provided golden per-round data.
// Tests: separate_prior → 4 rounds (sp4x compaction + restore_y_4x + mask) → y_hat
// Skips rANS (uses golden decoded y_q per round).

#include "dcvc_rt.h"
#include "trt_engine.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <math.h>
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
    while(nd<4) o->dims[nd++]=1; o->elems=1; for(int i=0;i<4;i++) o->elems*=o->dims[i];
    o->i32=o->fp16=o->fp32=NULL;
    if(strstr(hdr,"i4")){o->i32=malloc(o->elems*4);size_t g=fread(o->i32,4,o->elems,f);fclose(f);return g==o->elems?0:-1;}
    if(strstr(hdr,"f2")){o->fp16=malloc(o->elems*2);size_t g=fread(o->fp16,2,o->elems,f);fclose(f);return g==o->elems?0:-1;}
    o->fp32=malloc(o->elems*4); size_t g=fread(o->fp32,4,o->elems,f);fclose(f);return g==o->elems?0:-1;
}
static void npy_dev(const Npy* n, void* d) {
    if(n->fp16)cudaMemcpy(d,n->fp16,n->elems*2,cudaMemcpyHostToDevice);
    else{uint16_t*t=malloc(n->elems*2);for(size_t i=0;i<n->elems;i++){float f=n->fp32[i];uint32_t x;memcpy(&x,&f,4);uint32_t s=(x>>31)&1;int e=((x>>23)&0xff)-127+15;uint32_t m=(x>>13)&0x3ff;if(e<=0){m|=0x400;while(e<0){m>>=1;e++;}e=0;}if(e>=31){e=31;m=0;}t[i]=(s<<15)|(e<<10)|m;}cudaMemcpy(d,t,n->elems*2,cudaMemcpyHostToDevice);free(t);}
}
static float h2f(uint16_t h){uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,f;if(e==0){if(m==0)f=s<<31;else{int ex=-1;while(!(m&0x400)){m<<=1;ex--;}m&=0x3ff;e=127+ex-14;f=(s<<31)|(e<<23)|(m<<13);}}else if(e==31)f=(s<<31)|(0xff<<23)|(m<<13);else f=(s<<31)|((e+127-15)<<23)|(m<<13);float r;memcpy(&r,&f,4);return r;}

typedef void (*restore_fn)(const void*,const void*,const void*,void*,int,int,cudaStream_t);
typedef void (*mul_fn)(const void*,const void*,void*,int,cudaStream_t);

int main(int argc, char** argv) {
    const char* dd = argc>1?argv[1]:"native/assets/decode";
    const char* pd = argc>2?argv[2]:"native/build/plugin_demo";
    char path[600];

    snprintf(path,sizeof(path),"%s/libdcvc_kernels.so",pd);
    void* kh=dlopen(path,RTLD_NOW|RTLD_GLOBAL);
    restore_fn p_restore=(restore_fn)dlsym(kh,"dcvc_k_restore_y_4x");
    mul_fn p_mul=(mul_fn)dlsym(kh,"dcvc_k_elem_mul");

    int N_ch=256, HW=16*16;
    void* d_masks[4];
    for(int i=0;i<4;i++){Npy mn;snprintf(path,sizeof(path),"%s/mask_%d.npy",dd,i);read_npy(path,&mn);cudaMalloc(&d_masks[i],mn.elems*2);npy_dev(&mn,d_masks[i]);free(mn.fp16?mn.fp16:mn.fp32);}

    /* Load golden q_dec */
    Npy pf_npy; snprintf(path,sizeof(path),"%s/golden_params_fusion_ec0.npy",dd);
    read_npy(path,&pf_npy);
    uint16_t* pf_h=pf_npy.fp16;
    uint16_t* qd_h=malloc(HW*2);
    for(int i=0;i<HW;i++){float q=h2f(pf_h[1*HW+i]);q=1.0f/(1.0f+expf(-q))*1.5f+0.5f;uint32_t x;memcpy(&x,&q,4);uint32_t s=(x>>31)&1;int e=((x>>23)&0xff)-127+15;uint32_t m=(x>>13)&0x3ff;if(e<=0){m|=0x400;while(e<0){m>>=1;e++;}e=0;}if(e>=31){e=31;m=0;}qd_h[i]=(s<<15)|(e<<10)|m;}

    void* d_yhat_sf; cudaMalloc(&d_yhat_sf,N_ch*HW*2); cudaMemset(d_yhat_sf,0,N_ch*HW*2);
    void* d_yhat_step; cudaMalloc(&d_yhat_step,N_ch*HW*2);

    for(int rnd=0;rnd<4;rnd++){
        /* Load golden y_q, scales, means for this round */
        Npy yq_n, sc_n, mn_n;
        snprintf(path,sizeof(path),"%s/ec0_round%d_yq.npy",dd,rnd); read_npy(path,&yq_n);
        snprintf(path,sizeof(path),"%s/ec0_round%d_scales.npy",dd,rnd); read_npy(path,&sc_n);
        snprintf(path,sizeof(path),"%s/ec0_round%d_means.npy",dd,rnd); read_npy(path,&mn_n);

        void* d_yq; cudaMalloc(&d_yq,yq_n.elems*2); npy_dev(&yq_n,d_yq);
        /* means is [1,N_ch,H,W] */
        void* d_means; cudaMalloc(&d_means,mn_n.elems*2); npy_dev(&mn_n,d_means);

        /* restore_y_4x: (y_q broadcast 4x + means) * mask */
        p_restore(d_yq,d_means,d_masks[rnd],d_yhat_step,(N_ch/4)*HW,N_ch*HW,0);

        /* y_hat_sf += y_hat_step */
        uint16_t* sf=malloc(N_ch*HW*2); uint16_t* st=malloc(N_ch*HW*2);
        cudaMemcpy(sf,d_yhat_sf,N_ch*HW*2,cudaMemcpyDeviceToHost);
        cudaMemcpy(st,d_yhat_step,N_ch*HW*2,cudaMemcpyDeviceToHost);
        for(int i=0;i<N_ch*HW;i++){float v=h2f(sf[i])+h2f(st[i]);uint32_t x;memcpy(&x,&v,4);uint32_t s=(x>>31)&1;int e=((x>>23)&0xff)-127+15;uint32_t m=(x>>13)&0x3ff;if(e<=0){m|=0x400;while(e<0){m>>=1;e++;}e=0;}if(e>=31){e=31;m=0;}sf[i]=(s<<15)|(e<<10)|m;}
        cudaMemcpy(d_yhat_sf,sf,N_ch*HW*2,cudaMemcpyHostToDevice);
        free(sf);free(st);
        cudaFree(d_yq);cudaFree(d_means);
        free(yq_n.fp16?yq_n.fp16:yq_n.fp32);
        free(sc_n.fp16?sc_n.fp16:sc_n.fp32);
        free(mn_n.fp16?mn_n.fp16:mn_n.fp32);
    }

    /* y_hat = y_hat_sf * q_dec */
    uint16_t* yh=malloc(N_ch*HW*2);
    cudaMemcpy(yh,d_yhat_sf,N_ch*HW*2,cudaMemcpyDeviceToHost);
    for(int c=0;c<N_ch;c++)for(int hw=0;hw<HW;hw++){float v=h2f(yh[c*HW+hw])*h2f(qd_h[hw]);uint32_t x;memcpy(&x,&v,4);uint32_t s=(x>>31)&1;int e=((x>>23)&0xff)-127+15;uint32_t m=(x>>13)&0x3ff;if(e<=0){m|=0x400;while(e<0){m>>=1;e++;}e=0;}if(e>=31){e=31;m=0;}yh[c*HW+hw]=(s<<15)|(e<<10)|m;}

    /* Compare to golden y_hat */
    Npy gy; snprintf(path,sizeof(path),"%s/golden_y_hat_ec0.npy",dd);
    if(read_npy(path,&gy)==0&&gy.fp16){
        float mx=0,mr=0; int n=N_ch*HW,match=0;
        for(int i=0;i<n;i++){float d=fabsf(h2f(yh[i])-h2f(gy.fp16[i]));if(d>mx)mx=d;if(fabsf(h2f(gy.fp16[i]))>mr)mr=fabsf(h2f(gy.fp16[i]));if(d<0.5f)match++;}
        printf("=== y_hat (golden symbols) ===\n");
        printf("max_abs=%.4e rel=%.4f match=%d/%d (%.1f%%)\n",mx,mx/(mr+1e-6f),match,n,100.0f*match/n);
    }

    /* Cleanup */
    for(int i=0;i<4;i++)cudaFree(d_masks[i]);
    cudaFree(d_yhat_sf);cudaFree(d_yhat_step);
    free(yh);free(qd_h);free(pf_h);
    return 0;
}
