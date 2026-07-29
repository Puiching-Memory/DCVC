/* Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * Symbol export / import control for the DCVC shared library.
 * Generated-style header: no build-time CMake dependency, portable to
 * MSVC, GCC, Clang and MinGW.
 */
#ifndef DCVC_EXPORT_H
#define DCVC_EXPORT_H

/* On Windows the DCVC_EXPORTS macro is defined only while *building* the
 * library; consumers see the imported symbols. On ELF/Mach-O every public
 * symbol is annotated with default visibility. */
#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef DCVC_EXPORTS
#    define DCVC_API __declspec(dllexport)
#  else
#    define DCVC_API __declspec(dllimport)
#  endif
#else
#  define DCVC_API __attribute__((visibility("default")))
#endif

/* Standard C calling convention tag (no-op except on legacy Win32). */
#if defined(_WIN32) && !defined(__GNUC__)
#  define DCVC_CALL __cdecl
#else
#  define DCVC_CALL
#endif

#endif /* DCVC_EXPORT_H */
