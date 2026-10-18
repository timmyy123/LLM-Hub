// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef OP_ARG_CATEGORY_H_
#define OP_ARG_CATEGORY_H_

#include "graph_handle_defs.h"

#include <type_traits>

template <typename P> class Vector;

namespace hnnx {
struct OsS; // this is the 'real name of hnnx::op_slice_spec

// Op function parameter categories
// The order of these is important: The operands must
// appear in order of increasing category. Also, no two operands
// can have the same category, unless it's tensor_out or tensor_in
// (see CheckOpFuncArgs below).
enum class OpArgCategory { //
    invalid, // none of the below
    tensor_out, // T &, where T is a Tensor subclass
    vararg_out, // Vector<T*> const &; or Vector<T*>
    tensor_in, // T const &, where T is a Tensor subclass.
    vararg_in, // Vector<T const*> const &; or Vector<T*>
    slice_spec, // op_slice_spec (passed by value)
    graph_handle, // subclass of hnnx::GraphHandleBase; previously 'Graph const &'
};

//////////
// OpArgCat<TYPE>::value is the category of a parameter type, as an 'enum OpArgCategory'
// the 'base' definition only looks at the 'graph_handle' category; all others are
// done by specializing this this struct.
template <typename T> struct OpArgCat {
    // if it's a subclass of GraphHandleBase (without actually being GraphHandleBase) it's 'graph_handle'
    // otherwise invalid.
    static constexpr OpArgCategory value =
            (std::is_base_of_v<hnnx::GraphHandleBase, T> && !std::is_same_v<const hnnx::GraphHandleBase, const T>)
                    ? OpArgCategory::graph_handle
                    : OpArgCategory::invalid;
};

// T& or T const &; Ok if  T subclass of Tensor;
template <typename T> struct OpArgCat<T &> {
    static constexpr OpArgCategory value = !std::is_base_of_v<Tensor, T> ? OpArgCategory::invalid
                                           : std::is_const_v<T>          ? OpArgCategory::tensor_in
                                                                         : OpArgCategory::tensor_out;
};

// Also: Vector<T*> is ok as pass-by-value or pass-by-const-ref.
// Implementation of Vector<P> is just {P const *base, size_t n}
//
template <typename T> struct OpArgCat<Vector<T *> const &> {
    static constexpr OpArgCategory value = !std::is_base_of_v<Tensor, T> ? OpArgCategory::invalid
                                           : std::is_const_v<T>          ? OpArgCategory::vararg_in
                                                                         : OpArgCategory::vararg_out;
};
template <typename T> struct OpArgCat<Vector<T *>> : public OpArgCat<Vector<T *> const &> {};

// op_slice_spec is OK as a parameter
template <> struct OpArgCat<OsS> {
    static constexpr OpArgCategory value = OpArgCategory::slice_spec;
};
} // namespace hnnx
#endif
