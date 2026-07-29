/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Library version. The ABI is compatible within a major number; the SONAME
 * embeds the major (libdcvc.so.MAJOR) so a customer linked against 1.x keeps
 * working across minor/patch upgrades without relinking.
 */
#ifndef DCVC_VERSION_H
#define DCVC_VERSION_H

#define DCVC_VERSION_MAJOR 1
#define DCVC_VERSION_MINOR 0
#define DCVC_VERSION_PATCH 0

/* Compose at preprocessor time for compile-time checks. */
#define DCVC_VERSION_COMPOSE(MAJ, MIN, PAT) (((MAJ) << 20) | ((MIN) << 10) | (PAT))
#define DCVC_VERSION_NUMBER \
    DCVC_VERSION_COMPOSE(DCVC_VERSION_MAJOR, DCVC_VERSION_MINOR, DCVC_VERSION_PATCH)

#endif /* DCVC_VERSION_H */
