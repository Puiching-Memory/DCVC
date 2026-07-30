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

#include "console_pause.h"

#ifdef _WIN32
#include <windows.h>
static void dcvc_setenv_default(const char* key, const char* val) {
    if (getenv(key)) return;
    char buf[256];
    snprintf(buf, sizeof(buf), "%s=%s", key, val);
    _putenv(buf);
}
#else
static void dcvc_setenv_default(const char* key, const char* val) {
    if (getenv(key)) return;
    setenv(key, val, 0);
}
#endif

static int load_f32(const char* path, float** out, int dims[4]) {
    DcvcNpy n; if (dcvc_npy_read(path,&n)!=0) return -1;
    if (n.dtype!=DCVC_NPY_F32){dcvc_npy_free(&n);return -1;}
    for (int i=0;i<4;i++) dims[i]=(i<n.ndims)?n.dims[i]:1;
    *out=(float*)malloc(n.elems*sizeof(float));
    memcpy(*out,dcvc_npy_f32(&n),n.elems*sizeof(float));
    dcvc_npy_free(&n); return 0;
}
int main(int argc,char**argv){
    if (argc<3){fprintf(stderr,"usage: %s ref.npy x.npy [qp] [model_dir] [out.bin]\n",argv[0]);{ dcvc_pause_if_dblclick(); return 1; }}
    int qp=argc>3?atoi(argv[3]):32;
    const char* model_dir=argc>4?argv[4]:"../models";
    const char* out_bin=argc>5?argv[5]:NULL;
    if (!getenv("DCVC_DUMP_YHAT")) dcvc_setenv_default("DCVC_DUMP_YHAT","/tmp/cpp_yhat.npy");
    if (!getenv("DCVC_DUMP_XHAT")) dcvc_setenv_default("DCVC_DUMP_XHAT","/tmp/cpp_xhat.npy");
    if (!getenv("DCVC_DUMP_FEAT")) dcvc_setenv_default("DCVC_DUMP_FEAT","/tmp/cpp_feat.npy");
    float*ref=NULL;float*x=NULL;int rd[4],xd[4];
    if(load_f32(argv[1],&ref,rd)||load_f32(argv[2],&x,xd)){fprintf(stderr,"npy read fail\n");{ dcvc_pause_if_dblclick(); return 1; }}
    int H=rd[2],W=rd[3];
    DcvcCpuStatus st;
    int is_hts = !(getenv("DCVC_IS_HTS") && getenv("DCVC_IS_HTS")[0]=='0');
    DcvcCpuInterPipeline*p=dcvc_cpu_inter_pipeline_create(model_dir,H,W,qp,is_hts,&st);
    printf("inter variant: %s\n", is_hts ? "HT-S" : "HT-L");
    if(!p){fprintf(stderr,"create fail: %s\n",dcvc_cpu_status_string(st));{ dcvc_pause_if_dblclick(); return 1; }}
    /* HT chunk: DCVC_FRAME_DELAY frames. If x is a single frame, replicate it to
     * fill the chunk; if x already holds FRAMES frames, use it directly. */
    int single = !(xd[0]==1 || xd[0]==DCVC_FRAME_DELAY);
    float* chunk=NULL;
    if (!single) { chunk=x; }
    else {
        chunk=(float*)malloc((size_t)DCVC_FRAME_DELAY*3*H*W*sizeof(float));
        for(int i=0;i<DCVC_FRAME_DELAY;i++) memcpy(chunk+i*3*H*W,x,3*H*W*sizeof(float));
    }
    float*xhat=(float*)malloc((size_t)DCVC_FRAME_DELAY*3*H*W*sizeof(float));
    uint8_t*str=NULL;size_t strn=0;
    st=dcvc_cpu_inter_pipeline_encode(p,chunk,1,ref,&str,&strn,xhat);
    if(st){fprintf(stderr,"encode fail: %s\n",dcvc_cpu_status_string(st));{ dcvc_pause_if_dblclick(); return 1; }}
    if (out_bin) {
        FILE* f=fopen(out_bin,"wb"); if(!f){fprintf(stderr,"cannot write %s\n",out_bin);{ dcvc_pause_if_dblclick(); return 1; }}
        fwrite(str,1,strn,f); fclose(f);
    }
    printf("P-chunk stream=%zu bytes; x_hat[0..%d] dumped\n",strn,DCVC_FRAME_DELAY);
    free(ref);free(x);free(xhat);free(str);if(chunk!=x)free(chunk);dcvc_cpu_inter_pipeline_destroy(p);
    { dcvc_pause_if_dblclick(); return 0; }
}
