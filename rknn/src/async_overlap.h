#ifndef DCVC_RK_ASYNC_H
#define DCVC_RK_ASYNC_H

#include "dcvc_rk/profile.h"
#include "dcvc_rk/rknn_engine.h"
#include "dcvc_rk/quant.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef DcvcRkStatus (*DcvcRkCpuFn)(void* ctx);

static inline int dcvc_rk_async_enabled(void)
{
    const char* e = getenv("DCVC_RKNN_ASYNC");
    if (!e) return 1;
    if (e[0] == '0' || !strcmp(e, "off") || !strcmp(e, "false")) return 0;
    return 1;
}

static inline int dcvc_rk_pipe_i8_enabled(void)
{
    const char* e = getenv("DCVC_RKNN_PIPE_I8");
    if (!e) return 1; /* default on: NHWC pass-through + INT8 outs beats float-hop */
    if (e[0] == '0' || !strcmp(e, "off") || !strcmp(e, "false")) return 0;
    return 1;
}

typedef struct {
    DcvcRkEngine* eng;
    DcvcRkTensorView* vin;
    int n_in;
    DcvcRkTensorView* vout;
    int n_out;
    DcvcRkStatus st;
} DcvcRkNpuJob;

static void* dcvc_rk_npu_job_fn(void* arg)
{
    DcvcRkNpuJob* j = (DcvcRkNpuJob*)arg;
    j->st = dcvc_rk_engine_run(j->eng, j->vin, j->n_in, j->vout, j->n_out);
    return NULL;
}

/* Run NPU on a worker thread while cpu_fn runs on the caller thread.
 * Falls back to serial when DCVC_RKNN_ASYNC=0 or pthread_create fails.
 * Stage walls are exclusive (SUM may exceed frame wall — expected with overlap). */
static inline DcvcRkStatus dcvc_rk_overlap_npu_cpu(
    DcvcRkEngine* eng,
    DcvcRkTensorView* vin, int n_in,
    DcvcRkTensorView* vout, int n_out,
    DcvcRkCpuFn cpu_fn, void* cpu_ctx,
    DcvcRkStageProf* s_npu, DcvcRkStageProf* s_cpu,
    int64_t* npu_acc)
{
    if (!eng || !vin || !vout || !cpu_fn) return DCVC_RK_ERR_INVALID_ARG;

    if (!dcvc_rk_async_enabled()) {
        double t0 = dcvc_rk_now_ms();
        DcvcRkStatus st = dcvc_rk_engine_run(eng, vin, n_in, vout, n_out);
        if (s_npu) {
            s_npu->wall_ms += dcvc_rk_now_ms() - t0;
            if (st == DCVC_RK_OK) {
                int64_t u = dcvc_rk_engine_last_run_us(eng);
                int64_t su = dcvc_rk_engine_last_set_us(eng);
                int64_t gu = dcvc_rk_engine_last_get_us(eng);
                if (u > 0) { s_npu->npu_ms += u / 1000.0; if (npu_acc) *npu_acc += u; }
                if (su > 0) s_npu->set_ms += su / 1000.0;
                if (gu > 0) s_npu->get_ms += gu / 1000.0;
                s_npu->calls++;
            }
        }
        if (st != DCVC_RK_OK) return st;
        t0 = dcvc_rk_now_ms();
        st = cpu_fn(cpu_ctx);
        if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t0;
        return st;
    }

    DcvcRkNpuJob job = { eng, vin, n_in, vout, n_out, DCVC_RK_OK };
    pthread_t th;
    double t_npu0 = dcvc_rk_now_ms();
    int rc = pthread_create(&th, NULL, dcvc_rk_npu_job_fn, &job);
    if (rc != 0) {
        double t0 = dcvc_rk_now_ms();
        DcvcRkStatus st = dcvc_rk_engine_run(eng, vin, n_in, vout, n_out);
        if (s_npu) {
            s_npu->wall_ms += dcvc_rk_now_ms() - t0;
            if (st == DCVC_RK_OK) {
                int64_t u = dcvc_rk_engine_last_run_us(eng);
                int64_t su = dcvc_rk_engine_last_set_us(eng);
                int64_t gu = dcvc_rk_engine_last_get_us(eng);
                if (u > 0) { s_npu->npu_ms += u / 1000.0; if (npu_acc) *npu_acc += u; }
                if (su > 0) s_npu->set_ms += su / 1000.0;
                if (gu > 0) s_npu->get_ms += gu / 1000.0;
                s_npu->calls++;
            }
        }
        if (st != DCVC_RK_OK) return st;
        t0 = dcvc_rk_now_ms();
        st = cpu_fn(cpu_ctx);
        if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t0;
        return st;
    }

    double t_cpu0 = dcvc_rk_now_ms();
    DcvcRkStatus cpu_st = cpu_fn(cpu_ctx);
    if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t_cpu0;

    pthread_join(th, NULL);
    if (s_npu) {
        s_npu->wall_ms += dcvc_rk_now_ms() - t_npu0;
        if (job.st == DCVC_RK_OK) {
            int64_t u = dcvc_rk_engine_last_run_us(eng);
            int64_t su = dcvc_rk_engine_last_set_us(eng);
            int64_t gu = dcvc_rk_engine_last_get_us(eng);
            if (u > 0) { s_npu->npu_ms += u / 1000.0; if (npu_acc) *npu_acc += u; }
            if (su > 0) s_npu->set_ms += su / 1000.0;
            if (gu > 0) s_npu->get_ms += gu / 1000.0;
            s_npu->calls++;
        }
    }
    if (job.st != DCVC_RK_OK) return job.st;
    return cpu_st;
}

typedef struct {
    DcvcRkEngine* eng;
    DcvcRkI8View* vin;
    int n_in;
    DcvcRkI8View* vout;
    int n_out;
    DcvcRkStatus st;
} DcvcRkNpuI8Job;

static void* dcvc_rk_npu_i8_job_fn(void* arg)
{
    DcvcRkNpuI8Job* j = (DcvcRkNpuI8Job*)arg;
    j->st = dcvc_rk_engine_run_i8(j->eng, j->vin, j->n_in, j->vout, j->n_out);
    return NULL;
}

static inline void dcvc_rk_acc_eng(DcvcRkStageProf* s, DcvcRkEngine* e, int64_t* npu_acc)
{
    if (!s || !e) return;
    int64_t u = dcvc_rk_engine_last_run_us(e);
    int64_t su = dcvc_rk_engine_last_set_us(e);
    int64_t gu = dcvc_rk_engine_last_get_us(e);
    if (u > 0) { s->npu_ms += u / 1000.0; if (npu_acc) *npu_acc += u; }
    if (su > 0) s->set_ms += su / 1000.0;
    if (gu > 0) s->get_ms += gu / 1000.0;
    s->calls++;
}

static inline DcvcRkStatus dcvc_rk_overlap_npu_i8_cpu(
    DcvcRkEngine* eng,
    DcvcRkI8View* vin, int n_in,
    DcvcRkI8View* vout, int n_out,
    DcvcRkCpuFn cpu_fn, void* cpu_ctx,
    DcvcRkStageProf* s_npu, DcvcRkStageProf* s_cpu,
    int64_t* npu_acc)
{
    if (!eng || !vin || !vout || !cpu_fn) return DCVC_RK_ERR_INVALID_ARG;

    if (!dcvc_rk_async_enabled()) {
        double t0 = dcvc_rk_now_ms();
        DcvcRkStatus st = dcvc_rk_engine_run_i8(eng, vin, n_in, vout, n_out);
        if (s_npu) {
            s_npu->wall_ms += dcvc_rk_now_ms() - t0;
            if (st == DCVC_RK_OK) dcvc_rk_acc_eng(s_npu, eng, npu_acc);
        }
        if (st != DCVC_RK_OK) return st;
        t0 = dcvc_rk_now_ms();
        st = cpu_fn(cpu_ctx);
        if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t0;
        return st;
    }

    DcvcRkNpuI8Job job = { eng, vin, n_in, vout, n_out, DCVC_RK_OK };
    pthread_t th;
    double t_npu0 = dcvc_rk_now_ms();
    if (pthread_create(&th, NULL, dcvc_rk_npu_i8_job_fn, &job) != 0) {
        double t0 = dcvc_rk_now_ms();
        DcvcRkStatus st = dcvc_rk_engine_run_i8(eng, vin, n_in, vout, n_out);
        if (s_npu) {
            s_npu->wall_ms += dcvc_rk_now_ms() - t0;
            if (st == DCVC_RK_OK) dcvc_rk_acc_eng(s_npu, eng, npu_acc);
        }
        if (st != DCVC_RK_OK) return st;
        t0 = dcvc_rk_now_ms();
        st = cpu_fn(cpu_ctx);
        if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t0;
        return st;
    }

    double t_cpu0 = dcvc_rk_now_ms();
    DcvcRkStatus cpu_st = cpu_fn(cpu_ctx);
    if (s_cpu) s_cpu->wall_ms += dcvc_rk_now_ms() - t_cpu0;

    pthread_join(th, NULL);
    if (s_npu) {
        s_npu->wall_ms += dcvc_rk_now_ms() - t_npu0;
        if (job.st == DCVC_RK_OK) dcvc_rk_acc_eng(s_npu, eng, npu_acc);
    }
    if (job.st != DCVC_RK_OK) return job.st;
    return cpu_st;
}

#ifdef __cplusplus
}
#endif

#endif
