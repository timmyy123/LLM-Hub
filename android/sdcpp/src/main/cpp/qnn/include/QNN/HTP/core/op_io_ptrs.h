// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef OP_IO_PTRS_H
#define OP_IO_PTRS_H

#include "weak_linkage.h"

PUSH_VISIBILITY(default)

class GraphPrepare;
class Graph;
class Tensor;
class Op;
class OpDef;
struct OutputDef;
namespace fa {
struct FancyAllocator;
}

#include "graph_status.h"
#include "forward_classes.h"
#include "interface_defs.h"

#include <vector>
#include <stddef.h>

namespace hnnx {
class Allocator;
class OpHookBase;

//
// This is use for two purposes:
//  (1) make a new Op based on OpDef, using information in the op_def_map
//  (2) make a new Op which is a clone of another, and with a new ID.
// There are two different ctors for the two cases.
// The (1) ctor takes new_op_id from the OpDef; the (2) ctor has a parameter.
// Both constructors gather arrays of Tensor const * for the inputs,
//  but the (1) gathers OutputDef const for the outputs; and only records
// the # of outputs.
// Note: when cloning an op, 'ctor hook' are disabled.
//
class OpIoPtrs {
    GraphPrepare *graphp_p;
    Graph *graph_p; // pointer to same thing, but compiler doesn't know that here.
    OpId new_op_id;
    GraphStatus stat;
    OpHookBase const *ophook_ptr = nullptr;
    unsigned num_out; // number of outputs
  public:
    enum clonemode { output_realloc, output_dup, output_steal };
    OpDef const *op_def; // exactly  one of these two ...
    Op const *op_to_clone; // ... is null.
    clonemode clone_mode = output_realloc;
    std::vector<Tensor const *> in_tensors;
    std::vector<OutputDef const *> out_defs;

    OpIoPtrs(GraphPrepare &g, OpDef const *op_def);
    OpIoPtrs(GraphPrepare &g, Op const *op_to_clone_in, OpId new_op_id_in, clonemode clone_mode_in = output_realloc);
    // NOTE: should be able to remove this one when prepare code is properly separated - used in  Op::clone
    OpIoPtrs(Graph &g, Op const *op_to_clone_in, OpId new_op_id_in, clonemode clone_mode_in = output_realloc);
    GraphStatus status() const { return stat; }

    // Methods used for Switched Ops
    OpIoPtrs(Graph &g, Op const *op_to_clone_in, OpId new_op_id_in, OpDef const *op_def,
             clonemode clone_mode_in = output_dup);
    OpIoPtrs(GraphPrepare &g, Op const *op_to_clone_in, OpId new_op_id_in, OpDef const *op_def,
             clonemode clone_mode_in = output_dup);

    Graph &graph() const { return *graph_p; }
    GraphPrepare &graphp() const { return *graphp_p; }
    Allocator &allocator() const; // returns graph_p->get_allocator()
    fa::FancyAllocator &full_allocator() const; // returns graphp_p->get_full_allocator()

    // called from a constructor hook, to set an op's self-slicing count.
    void set_op_slicing(Op const &op, unsigned n_slices) const;

    void add_ophook(OpHookBase const *hookp) { ophook_ptr = hookp; }

    // hooks are called via .ophook( mfp, Op& ) where mfp is a pointer to method
    // of OpHookBase. To support methods with more parameters, we will add a template
    // version of ophook (which could pass a lambda->std::function to a non-template protected method)
    inline GraphStatus ophook(GraphStatus (OpHookBase::*mfp)(OpIoPtrs const &, Op &) const, Op &target_op) const
    {
        // hooks are suppressed if we are cloning an op, in a mode other than realloc.
        if (is_clone_mode() && clone_mode != output_realloc) return GraphStatus::Success;
        return (ophook_ptr != nullptr) ? ophook_func(mfp, target_op) : GraphStatus::Success;
    }
    OpId get_id() const { return new_op_id; }
    size_t n_inputs() const { return in_tensors.size(); }
    size_t n_outputs() const { return num_out; }
    inline bool is_clone_mode() const { return op_to_clone != nullptr; }
    // get an output tensor for the new op, using the clone mode.
    uptr_Tensor get_output_for_cloned_op(unsigned idx) const;

  protected:
    GraphStatus ophook_func(GraphStatus (OpHookBase::*mfp)(OpIoPtrs const &, Op &) const, Op &target_op) const;
};

GraphStatus change_output_tensor_shape(Op &op, unsigned out_idx, Graph &gr, unsigned newdims_rank,
                                       size_t const *new_dims);

} // namespace hnnx

POP_VISIBILITY()

#endif
