// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef OP_FACTORY_H_
#define OP_FACTORY_H_

#include "forward_classes.h"
#include "interface_defs.h"

namespace hnnx {
class OpIoPtrs;

// An op factory function.  Just a bare function poiner in order to keep
// things as small as possible.
using OpFactory = uptr_Op (*)(OpIoPtrs const &, const OpId);
} // namespace hnnx
#endif
