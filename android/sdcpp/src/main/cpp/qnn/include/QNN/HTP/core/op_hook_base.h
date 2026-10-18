// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================
#pragma once
#include "macros_attribute.h"
#include "graph_status.h"

class Op;

namespace hnnx {
class OpIoPtrs;

// this is a base class for adding hooks on construction of Ops.
// May not have data members or dtor - so it's just a vtable pointer, and is constexpr constructable
// All methods must be const, and return GraphStatus; the 'default' methods do nothing and return GraphNotApplicable.
// So, we can allow two or more hooks to be attached to an Op; when calling a method,
// we will call it on the first one, and if it returns NotApplicable, we will try the next
// one, etc (so they are 'layered', in effect).
//
class OpHookBase {
  public:
    API_EXPORT virtual GraphStatus pre_output_prep(OpIoPtrs const &, Op &) const;
    API_EXPORT virtual GraphStatus pre_allocate(OpIoPtrs const &, Op &) const;
};
// if the indicated Op is a SpawnOp, get its inner op ptr, otherwise null.
} // namespace hnnx
