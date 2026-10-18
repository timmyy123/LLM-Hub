// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef CC_PP_H
#define CC_PP_H 1

/*
 * C++ Preprocessor Definitions
 */

#ifdef __cplusplus
#define EXTERN_C_BEGIN extern "C" {
#define EXTERN_C_END   }
#else
#define EXTERN_C_BEGIN /* NOTHING */
#define EXTERN_C_END   /* NOTHING */
#endif

#endif
