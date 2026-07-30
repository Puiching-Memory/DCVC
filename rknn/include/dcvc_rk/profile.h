/* Stage timing for optimization — wall + NPU + RKNN I/O split. */
#ifndef DCVC_RK_PROFILE_H
#define DCVC_RK_PROFILE_H

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline double dcvc_rk_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* One named stage: wall covers the whole stage; npu/set/get from engine runs. */
typedef struct DcvcRkStageProf {
    const char* name;
    double wall_ms;
    double npu_ms; /* RKNN_QUERY_PERF_RUN */
    double set_ms; /* rknn_inputs_set */
    double get_ms; /* rknn_outputs_get */
    int calls;
} DcvcRkStageProf;

#define DCVC_RK_PROF_MAX 24

typedef struct DcvcRkProfile {
    DcvcRkStageProf stages[DCVC_RK_PROF_MAX];
    int n;
} DcvcRkProfile;

static inline void dcvc_rk_profile_reset(DcvcRkProfile* p)
{
    if (p) { p->n = 0; }
}

static inline DcvcRkStageProf* dcvc_rk_profile_add(DcvcRkProfile* p, const char* name)
{
    if (!p || p->n >= DCVC_RK_PROF_MAX) return NULL;
    DcvcRkStageProf* s = &p->stages[p->n++];
    s->name = name;
    s->wall_ms = s->npu_ms = s->set_ms = s->get_ms = 0;
    s->calls = 0;
    return s;
}

static inline void dcvc_rk_profile_print(const DcvcRkProfile* p, const char* title)
{
    if (!p || p->n <= 0) return;
    double tw = 0, tn = 0, ts = 0, tg = 0;
    printf("--- %s ---\n", title ? title : "profile");
    printf("%-22s %8s %8s %8s %8s %8s %5s\n",
           "stage", "wall", "npu", "set", "get", "cpu*", "calls");
    for (int i = 0; i < p->n; i++) {
        const DcvcRkStageProf* s = &p->stages[i];
        double cpu = s->wall_ms - s->npu_ms - s->set_ms - s->get_ms;
        if (cpu < 0) cpu = 0;
        printf("%-22s %7.1f %7.1f %7.1f %7.1f %7.1f %5d\n",
               s->name, s->wall_ms, s->npu_ms, s->set_ms, s->get_ms, cpu, s->calls);
        tw += s->wall_ms; tn += s->npu_ms; ts += s->set_ms; tg += s->get_ms;
    }
    double tcpu = tw - tn - ts - tg;
    if (tcpu < 0) tcpu = 0;
    printf("%-22s %7.1f %7.1f %7.1f %7.1f %7.1f\n",
           "SUM", tw, tn, ts, tg, tcpu);
    printf("  (* cpu = wall - npu - set - get; set/get = FP32 host<->NPU copy)\n");
}

#ifdef __cplusplus
}
#endif

#endif
