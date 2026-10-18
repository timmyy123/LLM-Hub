// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef C_TRICKS_H
#define C_TRICKS_H 1

#define CTRICKS_PASTER2(A, B) A##B
#define CTRICKS_PASTER(A, B)  CTRICKS_PASTER2(A, B)

#define STRINGIFY(x) #x
#define TOSTRING(x)  STRINGIFY(x)

#endif
