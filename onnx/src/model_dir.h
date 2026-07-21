/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Resolve the directory that holds the runtime ONNX models + CDF tables, so
 * the test executables work both when run from an in-tree build/ directory
 * and from the packaged folder (where models sit next to the executable).
 */
#ifndef DCVC_MODEL_DIR_H
#define DCVC_MODEL_DIR_H

#ifdef __cplusplus
extern "C" {
#endif

/* Probes candidate directories (first match wins) and returns the one that
 * contains a file named probe_filename:
 *   1. hint            (the caller's explicit argv[1]; may be NULL)
 *   2. "."             (current dir -- the packaged-folder case)
 *   3. "../models"     (running from an in-tree build/ directory)
 *   4. "models"
 * Pass any model that must live in the dir as probe_filename, e.g.
 * "intra_analysis_standard.onnx".
 *
 * Returns a pointer to a static buffer holding the chosen directory. If no
 * candidate contains the probe file, returns hint (or "." if hint is NULL)
 * so the caller proceeds to a clear "file not found" error. */
const char* dcvc_resolve_model_dir(const char* hint, const char* probe_filename);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_MODEL_DIR_H */
