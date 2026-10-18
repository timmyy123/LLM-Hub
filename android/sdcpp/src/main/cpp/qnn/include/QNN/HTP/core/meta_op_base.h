// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================
#pragma once
#include "macros_attribute.h"
#include "op.h"
#include "graph_status.h"
#include "deserializer.h"
#include "interface_defs.h"

#include <stddef.h>
#include <utility>

class Graph;
class Tensor;

namespace hnnx {
class OpIoPtrs;

// MetaOpBase is a shim which provides empty defs for all of the =0 virtual methods,
// so that internal Ops (e.g. PreloadOp) can be based on this and not need to define any they don't need
//
class MetaOpBase : public Op {
  public:
    MetaOpBase() {};
    MetaOpBase(Graph &graph_in, unsigned long long int my_id_in) : Op(graph_in, my_id_in) {}
    explicit MetaOpBase(hnnx::Deserz &dctx) : Op(dctx) {}

    API_EXPORT virtual GraphStatus prepare(hnnx::OpIoPtrs const &,
                                           bool tcm_available) override; //{ return GraphStatus::Success;}
    API_EXPORT virtual GraphStatus allocate(Graph &graph_in) override; // { return GraphStatus::Success;}

    API_EXPORT virtual bool is_valid() const noexcept override; // {return false;}
    API_EXPORT virtual std::pair<size_t, size_t> num_inputs_outputs() const override; //{ return {0,0};}
    API_EXPORT virtual Tensor const *get_input_output(size_t which,
                                                      bool is_input) const override; // {return nullptr;}
    API_EXPORT virtual void serialize(hnnx::SerOpsInterface &) const override; // {}
    API_EXPORT virtual uptr_Op clone_meta(Graph &graph_in, OpId new_opid) const; // {return uptr_Op(nullptr);}
};
} // namespace hnnx
