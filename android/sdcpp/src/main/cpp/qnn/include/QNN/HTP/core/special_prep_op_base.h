// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================
#pragma once
#include "deserializer.h"
#include "meta_op_base.h"
#include "macros_attribute.h"
#include "flags.h"
#include "interface_defs.h"
#include "opname_tag.h"

class Graph;

namespace hnnx {
class CostBasedFeatureDesc;

// SpecialPrepOpBase is a shim (based on MetaOpBase) which provides some new virtual methods
// that are queried during GraphDeps stage of preparation.
// This is intended for things like SuperTileOp which want to add these discovery methods.
//
// LCOV_EXCL_START [SAFTYSWCCB-1542]

class SpecialPrepOpBase : public MetaOpBase {
  public:
    SpecialPrepOpBase() {};
    SpecialPrepOpBase(Graph &graph_in, unsigned long long int my_id_in) : MetaOpBase(graph_in, my_id_in) {}
    explicit SpecialPrepOpBase(hnnx::Deserz &dctx) : MetaOpBase(dctx) {}

    // new virtual methods to populate the OpDesc for the op:
    // These return 'true' if the result was changed, and 'false' if unchanged; the caller can set
    // the variable to reasonable default before calling, and then ignore the result.
    API_EXPORT virtual bool get_opdef_name(OpId opid, opname_tag_t &result) const; // {return false} in op.cc
    API_EXPORT virtual bool get_splithist(OpId opid, splithist_t &result) const { return false; }
    API_EXPORT virtual bool get_is_volatile(OpId opid, bool &result) const { return false; }
    API_EXPORT virtual bool get_cost(const Graph &, OpId opid, float &result) const { return false; }
    API_EXPORT virtual bool get_flags_word(OpId opid, Flags_word &result) const { return false; }

    // make a CostBasedFeatureDesc. If 'false' is returned, it should be obtained 'in the usual manner'.
    API_EXPORT virtual bool get_costbased_feature(OpId opid, CostBasedFeatureDesc &result) const { return false; }
};
// LCOV_EXCL_STOP
} // namespace hnnx
