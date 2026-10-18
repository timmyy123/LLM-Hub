// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef HEXNN_PUB_DEFINE_DESERIALIZATION_ENABLED
#define HEXNN_PUB_DEFINE_DESERIALIZATION_ENABLED

// Deserialization is enabled by default on all platforms (Linux, Android,
// Windows x86, and Windows ARM64) to ensure op registrations are available
// during aux graph deserialization for rehydration
//
// To *disable* deserialization (e.g., for debugging, or platform-
// specific issues), define DESERIALIZATION_DISABLED. This will prevent registration
// of deserialization-related ops and code paths.

#if !defined(DESERIALIZATION_ENABLED)
#if defined(DESERIALIZATION_DISABLED)
#define DESERIALIZATION_ENABLED 0
#else
#define DESERIALIZATION_ENABLED 1
#endif
#endif

#endif
