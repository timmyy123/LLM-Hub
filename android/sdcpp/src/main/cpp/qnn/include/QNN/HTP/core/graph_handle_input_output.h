// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef GRAPH_HANDLE_INPUT_OUTPUT_H
#define GRAPH_HANDLE_INPUT_OUTPUT_H 1

#include <cstddef>
#include "forward_classes.h"
#include "graph_handle_defs.h"

// These functions are for access to Graph input & output tensors at runtime (within Op functions)
// using a graph handle
//
namespace GHANDLE_NS {
// Each of these must have implementation in graph_handle_input_output_impl.h

// access to 'options.io_dma_only'
GHANDLE_ACCESS_FUNC_QUALIFIER bool get_option_io_dma_only(hnnx::GHandle) noexcept;

// This wraps 'select_input' in graph.h.
// Parameters have been reversed to be consistent with other graph_handle functions, and name has
// been changed to avoid confusion with the other function.
GHANDLE_ACCESS_FUNC_QUALIFIER Tensor const * //
input_select(hnnx::GHandle, TensorShape<4> const &select_shape);

// This wraps 'select_output' in graph.h.
// Parameters have been reversed to be consistent with other graph_handle functions, and name has
// been changed to avoid confusion with the other function.
GHANDLE_ACCESS_FUNC_QUALIFIER Tensor * //
output_select(hnnx::GHandle, TensorShape<4> const &select_shape);

// this is a convenience wrapper which also static-casts the pointer to a given tensor type
// e.g. runtime_graph::input_select_type<TensType>(ghand, select_shape) -> TensType const *
//
template <typename T> //
inline T const *input_select_type(hnnx::GHandle gh, TensorShape<4> const &select_shape)
{
    return static_cast<T const *>(input_select(gh, select_shape));
}
// this is a convenience wrapper which also static-casts the pointer to a given tensor type
template <typename T> //
inline T *output_select_type(hnnx::GHandle gh, TensorShape<4> const &select_shape)
{
    return static_cast<T *>(output_select(gh, select_shape));
}

} // namespace GHANDLE_NS

#if GHANDLE_INCLUDE_IMPL // defined in graph_handle_defs.h"
#include "graph_handle_input_output_impl.h"
#endif

#endif // GRAPH_HANDLE_INPUT_OUTPUT_H
