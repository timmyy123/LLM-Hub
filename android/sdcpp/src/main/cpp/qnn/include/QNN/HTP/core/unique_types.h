// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef UNIQUE_TYPES_H
#define UNIQUE_TYPES_H 1

// simpler way ... generates smaller code
#define DEFINE_UNIQ_TY()                                                                                               \
    namespace {                                                                                                        \
    template <int K> struct UniqTy {};                                                                                 \
    } // namespace
#define UNIQUE_TYPE UniqTy<__LINE__>
#endif
