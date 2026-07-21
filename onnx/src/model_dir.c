/* Copyright (c) Microsoft Corporation. Licensed under the MIT License. */
#include "model_dir.h"

#include <stdio.h>
#include <string.h>

static int dcvc_file_exists(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

const char* dcvc_resolve_model_dir(const char* hint, const char* probe_filename)
{
    static char result[1024];

    const char* candidates[4];
    int n = 0;
    if (hint && hint[0]) candidates[n++] = hint;
    candidates[n++] = ".";
    candidates[n++] = "../models";
    candidates[n++] = "models";

    char probe[1200];
    for (int i = 0; i < n; i++) {
        snprintf(probe, sizeof(probe), "%s/%s", candidates[i], probe_filename);
        if (dcvc_file_exists(probe)) {
            snprintf(result, sizeof(result), "%s", candidates[i]);
            return result;
        }
    }
    return (hint && hint[0]) ? hint : ".";
}
