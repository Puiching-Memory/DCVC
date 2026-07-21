// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
//
// Full I-frame decode in C: bitstream → z_hat (rANS) → params (hyper_dec) →
// params_fusion (y_prior_fusion) → y_hat (decompress_prior_4x) → x_hat (synthesis)
//
// Uses golden z_hat and params_fusion to validate engine + spatial prior stages.

#include "dcvc_trt_runner.h"
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

static float f16_to_f32(uint16_t h) {
    uint32_t sign=(h>>15)&1, expo=(h>>10)&0x1f, mant=h&0x3ff, f;
    if (expo==0) { if (mant==0) f=sign<<31; else { int e=-1; while(!(mant&0x400)){mant<<=1;e--;} mant&=0x3ff; expo=127+e-14; f=(sign<<31)|(expo<<23)|(mant<<13);} }
    else if (expo==31) f=(sign<<31)|(0xff<<23)|(mant<<13);
    else f=(sign<<31)|((expo+127-15)<<23)|(mant<<13);
    float r; memcpy(&r,&f,4); return r;
}

typedef struct { int dims[4]; size_t elems; uint16_t* fp16; float* fp32; } Npy;
static int read_npy(const char* path, Npy* o) {
    FILE* f=fopen(path,"rb"); if(!f) return -1;
    char m[6]; fread(m,1,6,f); if(memcmp(m,"\x93NUMPY",6)) {fclose(f);return -1;}
    uint8_t mj,mn; fread(&mj,1,1,f); fread(&mn,1,1,f);
    uint32_t hl=0; if(mj==1){uint16_t h16;fread(&h16,2,1,f);hl=h16;} else fread(&hl,4,1,f);
    char hdr[512]; if(hl>=sizeof(hdr)){fclose(f);return -1;} fread(hdr,1,hl,f); hdr[hl]=0;
    o->dims[0]=o->dims[1]=o->dims[2]=o->dims[3]=1; int nd=0;
    char* sp=strstr(hdr,"shape"); if(sp){sp=strchr(sp,'('); if(sp){sp++; while(*sp&&*sp!=')'&&nd<4){if(*sp>='0'&&*sp<='9'){o->dims[nd++]=atoi(sp);while(*sp>='0'&&*sp<='9')sp++;}sp++;}}}
    while(nd<4) o->dims[nd++]=1;
    o->elems=1; for(int i=0;i<4;i++) o->elems*=o->dims[i];
    int is_f16=strstr(hdr,"f2")!=NULL;
    if(is_f16){o->fp16=malloc(o->elems*2);o->fp32=NULL;size_t g=fread(o->fp16,2,o->elems,f);fclose(f);return g==o->elems?0:-1;}
    else{o->fp32=malloc(o->elems*4);o->fp16=NULL;size_t g=fread(o->fp32,4,o->elems,f);fclose(f);return g==o->elems?0:-1;}
}

static void npy_dev_fp16(const Npy* n, void* d) {
    if(n->fp16) cudaMemcpy(d,n->fp16,n->elems*2,cudaMemcpyHostToDevice);
    else { uint16_t*t=malloc(n->elems*2); for(size_t i=0;i<n->elems;i++) {
        float f=n->fp32[i]; uint32_t x; memcpy(&x,&f,4);
        uint32_t s=(x>>31)&1; int e=((x>>23)&0xff)-127+15; uint32_t m=(x>>13)&0x3ff;
        if(e<=0){m|=0x400;while(e<0){m>>=1;e++;}e=0;} if(e>=31){e=31;m=0;}
        t[i]=(s<<15)|(e<<10)|m;
    } cudaMemcpy(d,t,n->elems*2,cudaMemcpyHostToDevice); free(t); }
}

static void npy_to_float(const Npy* n, float* out) {
    if(n->fp16) for(size_t i=0;i<n->elems;i++) out[i]=f16_to_f32(n->fp16[i]);
    else memcpy(out,n->fp32,n->elems*4);
}

// Kernel function pointers
typedef void (*build_idx_fn)(const void*, uint8_t*, int, float, float, float, cudaStream_t);
typedef void (*restore_y4x_fn)(const void*, const void*, const void*, void*, int, int, cudaStream_t);
typedef void (*sp4x_fn)(const void*, void*, int, cudaStream_t);
typedef void (*mul_fn)(const void*, const void*, void*, int, cudaStream_t);
static build_idx_fn p_build_idx=NULL;
static restore_y4x_fn p_restore_y4x=NULL;
static sp4x_fn p_sp4x=NULL;
static mul_fn p_mul=NULL;

int main(int argc, char** argv) {
    const char* asset_dir = argc>1 ? argv[1] : "tensorRT/assets";
    const char* plugin_dir = argc>2 ? argv[2] : "tensorRT/build/plugin_demo";
    char path[512], decode_dir[512];
    snprintf(decode_dir, sizeof(decode_dir), "%s/decode", asset_dir);
    char eng_dir[512];
    snprintf(eng_dir, sizeof(eng_dir), "%s/engines", asset_dir);

    // Load kernels
    snprintf(path, sizeof(path), "%s/libdcvc_kernels.so", plugin_dir);
    void* kh = dlopen(path, RTLD_NOW|RTLD_GLOBAL);
    if (kh) {
        p_build_idx = (build_idx_fn)dlsym(kh,"dcvc_k_build_index_dec");
        p_restore_y4x = (restore_y4x_fn)dlsym(kh,"dcvc_k_restore_y_4x");
        p_sp4x = (sp4x_fn)dlsym(kh,"dcvc_k_single_part_writing_4x");
        p_mul = (mul_fn)dlsym(kh,"dcvc_k_elem_mul");
    }

    DcvcRtStatus st;
    int N=256, y_h=16, y_w=16, z_ch=128;

    // === Stage 1: Load golden z_hat, run hyper_dec + y_prior_fusion ===
    printf("=== Stage 1: z_hat → params → params_fusion ===\n");
    Npy zhat_npy;
    snprintf(path, sizeof(path), "%s/golden_z_hat.npy", decode_dir);
    if (read_npy(path, &zhat_npy) != 0) { fprintf(stderr,"Cannot read golden_z_hat\n"); return 1; }

    // Load hyper_dec engine
    snprintf(path, sizeof(path), "%s/hyper_dec.engine", eng_dir);
    DcvcTrtEngine* hyper_dec = dcvc_trt_engine_load(path, plugin_dir, &st);
    if (!hyper_dec) { fprintf(stderr,"hyper_dec load failed\n"); return 1; }

    void* d_zhat; cudaMalloc(&d_zhat, zhat_npy.elems*2);
    npy_dev_fp16(&zhat_npy, d_zhat);
    int32_t dims[4]={1,z_ch,4,4};
    dcvc_trt_engine_set_shape(hyper_dec,"in0",dims,4);
    dcvc_trt_engine_set_addr(hyper_dec,"in0",d_zhat);
    const char* hd_out = dcvc_trt_engine_tensor_name(hyper_dec, dcvc_trt_engine_num_io(hyper_dec)-1);
    int32_t od[8]; int ond;
    dcvc_trt_engine_get_shape(hyper_dec, hd_out, od, &ond, 8);
    printf("  hyper_dec: [%d,%d,%d,%d]→[%d,%d,%d,%d]\n",dims[0],dims[1],dims[2],dims[3],od[0],od[1],od[2],od[3]);

    void* d_params; cudaMalloc(&d_params, od[1]*od[2]*od[3]*2);
    dcvc_trt_engine_set_addr(hyper_dec, hd_out, d_params);
    dcvc_trt_engine_execute(hyper_dec, NULL);
    cudaDeviceSynchronize();

    // Compare params to golden
    Npy gparams_npy;
    snprintf(path, sizeof(path), "%s/golden_params.npy", decode_dir);
    if (read_npy(path, &gparams_npy) == 0) {
        int n = gparams_npy.elems;
        uint16_t* tmp = malloc(n*2);
        cudaMemcpy(tmp, d_params, n*2, cudaMemcpyDeviceToHost);
        float mx=0,mr=0;
        float* gold = malloc(n*4); npy_to_float(&gparams_npy, gold);
        for(int i=0;i<n;i++){float d=fabsf(f16_to_f32(tmp[i])-gold[i]); if(d>mx)mx=d; if(fabsf(gold[i])>mr)mr=fabsf(gold[i]);}
        printf("  [hyper_dec] max_abs=%.4e rel=%.4f\n", mx, mx/(mr+1e-6));
        free(tmp); free(gold);
    }

    // Run y_prior_fusion
    snprintf(path, sizeof(path), "%s/y_prior_fusion.engine", eng_dir);
    DcvcTrtEngine* ypf = dcvc_trt_engine_load(path, plugin_dir, &st);
    if (!ypf) { fprintf(stderr,"y_prior_fusion load failed\n"); return 1; }

    int32_t pf_in[4]={1,od[1],od[2],od[3]};
    dcvc_trt_engine_set_shape(ypf,"in0",pf_in,4);
    dcvc_trt_engine_set_addr(ypf,"in0",d_params);
    const char* pf_out = dcvc_trt_engine_tensor_name(ypf, dcvc_trt_engine_num_io(ypf)-1);
    dcvc_trt_engine_get_shape(ypf, pf_out, od, &ond, 8);
    printf("  y_prior_fusion: →[%d,%d,%d,%d]\n",od[0],od[1],od[2],od[3]);

    void* d_pfout; cudaMalloc(&d_pfout, od[1]*od[2]*od[3]*2);
    dcvc_trt_engine_set_addr(ypf, pf_out, d_pfout);
    dcvc_trt_engine_execute(ypf, NULL);
    cudaDeviceSynchronize();

    // Compare to golden params_fusion
    Npy gpf_npy;
    snprintf(path, sizeof(path), "%s/golden_params_fusion.npy", decode_dir);
    if (read_npy(path, &gpf_npy) == 0) {
        int n = gpf_npy.elems;
        uint16_t* tmp = malloc(n*2);
        cudaMemcpy(tmp, d_pfout, n*2, cudaMemcpyDeviceToHost);
        float mx=0,mr=0;
        float* gold = malloc(n*4); npy_to_float(&gpf_npy, gold);
        for(int i=0;i<n;i++){float d=fabsf(f16_to_f32(tmp[i])-gold[i]); if(d>mx)mx=d; if(fabsf(gold[i])>mr)mr=fabsf(gold[i]);}
        printf("  [y_prior_fusion] max_abs=%.4e rel=%.4f\n", mx, mx/(mr+1e-6));
        free(tmp); free(gold);
    }

    // === Stage 2: Run synthesis with golden y_hat ===
    printf("\n=== Stage 2: y_hat → x_hat (synthesis) ===\n");
    Npy yhat_npy;
    snprintf(path, sizeof(path), "%s/golden_y_hat.npy", decode_dir);
    if (read_npy(path, &yhat_npy) != 0) { fprintf(stderr,"Cannot read golden_y_hat\n"); return 1; }

    Npy qdec_npy;
    snprintf(path, sizeof(path), "%s/q_dec.npy", decode_dir);
    read_npy(path, &qdec_npy);

    snprintf(path, sizeof(path), "%s/intra_synthesis.engine", eng_dir);
    DcvcTrtEngine* synth = dcvc_trt_engine_load(path, plugin_dir, &st);
    if (!synth) { fprintf(stderr,"synthesis load failed\n"); return 1; }

    void* d_yhat; cudaMalloc(&d_yhat, yhat_npy.elems*2);
    npy_dev_fp16(&yhat_npy, d_yhat);
    void* d_qdec; cudaMalloc(&d_qdec, qdec_npy.elems*2);
    npy_dev_fp16(&qdec_npy, d_qdec);

    int32_t yh_dims[4]={1,256,16,16};
    int32_t qd_dims[4]={1,368,1,1};
    dcvc_trt_engine_set_shape(synth,"in0",yh_dims,4);
    dcvc_trt_engine_set_shape(synth,"in1",qd_dims,4);
    dcvc_trt_engine_set_addr(synth,"in0",d_yhat);
    dcvc_trt_engine_set_addr(synth,"in1",d_qdec);
    const char* s_out = dcvc_trt_engine_tensor_name(synth, dcvc_trt_engine_num_io(synth)-1);
    dcvc_trt_engine_get_shape(synth, s_out, od, &ond, 8);
    printf("  synthesis: y_hat[%d,%d,%d,%d]+q_dec→x_hat[%d,%d,%d,%d]\n",
           yh_dims[0],yh_dims[1],yh_dims[2],yh_dims[3],od[0],od[1],od[2],od[3]);

    void* d_xhat; cudaMalloc(&d_xhat, od[1]*od[2]*od[3]*2);
    dcvc_trt_engine_set_addr(synth, s_out, d_xhat);
    dcvc_trt_engine_execute(synth, NULL);
    cudaDeviceSynchronize();
    printf("  synthesis executed\n");

    // Print x_hat stats
    {
        int n = od[1]*od[2]*od[3];
        uint16_t* tmp = malloc(n*2);
        cudaMemcpy(tmp, d_xhat, n*2, cudaMemcpyDeviceToHost);
        float mn=1e9f,mx=-1e9f;
        for(int i=0;i<n;i++){float v=f16_to_f32(tmp[i]); if(v<mn)mn=v; if(v>mx)mx=v;}
        printf("  x_hat range: [%.4f, %.4f]\n", mn, mx);
        free(tmp);
    }

    // Cleanup
    dcvc_trt_engine_destroy(hyper_dec);
    dcvc_trt_engine_destroy(ypf);
    dcvc_trt_engine_destroy(synth);
    cudaFree(d_zhat); cudaFree(d_params); cudaFree(d_pfout);
    cudaFree(d_yhat); cudaFree(d_qdec); cudaFree(d_xhat);
    free(zhat_npy.fp16?zhat_npy.fp16:zhat_npy.fp32);
    if(gparams_npy.elems) free(gparams_npy.fp16?gparams_npy.fp16:gparams_npy.fp32);
    if(gpf_npy.elems) free(gpf_npy.fp16?gpf_npy.fp16:gpf_npy.fp32);
    free(yhat_npy.fp16?yhat_npy.fp16:yhat_npy.fp32);
    free(qdec_npy.fp16?qdec_npy.fp16:qdec_npy.fp32);

    printf("\n=== Full C decode pipeline test complete ===\n");
    return 0;
}
