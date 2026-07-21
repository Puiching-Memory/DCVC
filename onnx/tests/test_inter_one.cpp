/* Debug driver: encode ONE P-frame given ref.npy + x.npy, dump y_hat & x_hat.
 * Usage: test_inter_one <ref.npy> <x.npy> [qp=32]
 * Writes /tmp/cpp_yhat.npy, /tmp/cpp_xhat.npy, /tmp/cpp_feat.npy
 * (env DCVC_DUMP_YHAT etc. are set internally by this driver).
 */
#include "cpu_inter_pipeline.h"
#include "npy_reader.h"
#include "onnx_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int load_f32(const char* path, float** out, int dims[4]) {
    DcvcNpy n; if (dcvc_npy_read(path,&n)!=0) return -1;
    if (n.dtype!=DCVC_NPY_F32){dcvc_npy_free(&n);return -1;}
    for (int i=0;i<4;i++) dims[i]=(i<n.ndims)?n.dims[i]:1;
    *out=(float*)malloc(n.elems*sizeof(float));
    memcpy(*out,dcvc_npy_f32(&n),n.elems*sizeof(float));
    dcvc_npy_free(&n); return 0;
}
int main(int argc,char**argv){
    if (argc<3){fprintf(stderr,"usage: %s ref.npy x.npy [qp] [model_dir] [out.bin]\n",argv[0]);return 1;}
    int qp=argc>3?atoi(argv[3]):32;
    const char* model_dir=argc>4?argv[4]:"../models";
    const char* out_bin=argc>5?argv[5]:NULL;
    if (!getenv("DCVC_DUMP_YHAT")) setenv("DCVC_DUMP_YHAT","/tmp/cpp_yhat.npy",0);
    if (!getenv("DCVC_DUMP_XHAT")) setenv("DCVC_DUMP_XHAT","/tmp/cpp_xhat.npy",0);
    if (!getenv("DCVC_DUMP_FEAT")) setenv("DCVC_DUMP_FEAT","/tmp/cpp_feat.npy",0);
    float*ref=NULL;float*x=NULL;int rd[4],xd[4];
    if(load_f32(argv[1],&ref,rd)||load_f32(argv[2],&x,xd)){fprintf(stderr,"npy read fail\n");return 1;}
    int H=rd[2],W=rd[3];
    DcvcCpuStatus st;
    DcvcCpuInterPipeline*p=dcvc_cpu_inter_pipeline_create(model_dir,H,W,qp,&st);
    if(!p){fprintf(stderr,"create fail: %s\n",dcvc_cpu_status_string(st));return 1;}
    float*xhat=(float*)malloc(3*H*W*sizeof(float));
    uint8_t*str=NULL;size_t strn=0;
    st=dcvc_cpu_inter_pipeline_encode(p,x,ref,&str,&strn,xhat);
    if(st){fprintf(stderr,"encode fail: %s\n",dcvc_cpu_status_string(st));return 1;}
    if (out_bin) {
        FILE* f=fopen(out_bin,"wb"); if(!f){fprintf(stderr,"cannot write %s\n",out_bin);return 1;}
        fwrite(str,1,strn,f); fclose(f);
    }
    printf("P-frame stream=%zu bytes; x_hat dumped\n",strn);
    free(ref);free(x);free(xhat);free(str);dcvc_cpu_inter_pipeline_destroy(p);
    return 0;
}
