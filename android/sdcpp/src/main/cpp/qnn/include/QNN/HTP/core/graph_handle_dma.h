// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef GRAPH_HANDLE_DMA_H
#define GRAPH_HANDLE_DMA_H 1

#include <cstddef>
#include <utility>

#include "dma_id.h"
#include "hexagon_nn_types.h"
#include "graph_handle_defs.h"

struct Ubwc_D_Surface_Struct;
// These functions are performing DMA operations at runtime (within Op functions)
// using a graph handle
//
namespace GHANDLE_NS {
// each of these must have implementation in graph_handle_dma_impl.h

//////////////
// dma_xxxx(handle, ...
// are equivalent to graph.dma_manager.do_xxxx( ... )
//
GHANDLE_ACCESS_FUNC_QUALIFIER hnnx::dma_id_t //
dma_memcpy(hnnx::GHandle, void *dst, const void *src, size_t len, unsigned modes = 0);

GHANDLE_ACCESS_FUNC_QUALIFIER hnnx::dma_id_t //
dma_far_memcpy(hnnx::GHandle, hexagon_nn_wide_address_t dst, hexagon_nn_wide_address_t src, size_t len,
               unsigned modes = 0);

GHANDLE_ACCESS_FUNC_QUALIFIER hnnx::dma_id_t //
dma_far_memcpy(hnnx::GHandle, void *dst, hexagon_nn_wide_address_t src, size_t len, unsigned modes = 0);

GHANDLE_ACCESS_FUNC_QUALIFIER hnnx::dma_id_t //
dma_far_memcpy(hnnx::GHandle, hexagon_nn_wide_address_t dst, const void *src, size_t len, unsigned modes = 0);

GHANDLE_ACCESS_FUNC_QUALIFIER hnnx::dma_id_t //
dma_memcpy2d(hnnx::GHandle, void *dst, const void *src, size_t width, size_t height, size_t dst_stride,
             size_t src_stride, unsigned modes = 0);

GHANDLE_ACCESS_FUNC_QUALIFIER void //
dma_memcpy2d_overlapped(hnnx::GHandle, void *dst, const void *src, size_t width, size_t height, size_t dst_stride,
                        size_t src_stride, unsigned modes = 0);

GHANDLE_ACCESS_FUNC_QUALIFIER hnnx::dma_id_t //
dma_far_memcpy2d(hnnx::GHandle, hexagon_nn_wide_address_t dst, hexagon_nn_wide_address_t src, size_t width,
                 size_t height, size_t dst_stride, size_t src_stride, unsigned modes = 0);

GHANDLE_ACCESS_FUNC_QUALIFIER hnnx::dma_id_t dma_last_operation_id(hnnx::GHandle);

GHANDLE_ACCESS_FUNC_QUALIFIER void dma_wait_for(hnnx::GHandle, hnnx::dma_id_t);

/////////////
// Dma-related

// convert 'coded pointer' to wide address and 'flags'
GHANDLE_ACCESS_FUNC_QUALIFIER std::pair<hexagon_nn_wide_address_t, unsigned> decode_farvm_addr(hnnx::GHandle,
                                                                                               void const *);

// convert 'coded dynamic pointer to far_vm_ptr_t and 'flags'
GHANDLE_ACCESS_FUNC_QUALIFIER std::pair<hexagon_nn_wide_address_t, unsigned>
decode_dynamic_addr(hnnx::GHandle, void const *, int32_t, int32_t);

// access to options.slc_alloc_all_weights
GHANDLE_ACCESS_FUNC_QUALIFIER bool get_option_slc_alloc_all_weights(hnnx::GHandle) noexcept;

// access to 'bool graph_input_dma_uses_bypass() const'
GHANDLE_ACCESS_FUNC_QUALIFIER bool graph_input_dma_uses_bypass(hnnx::GHandle);

// access to 'bool graph_output_dma_uses_bypass() const'
GHANDLE_ACCESS_FUNC_QUALIFIER bool graph_output_dma_uses_bypass(hnnx::GHandle);

// Can a const->tcm copy be skipped; based on its start address in TCM (for 'cached weights')
GHANDLE_ACCESS_FUNC_QUALIFIER bool can_skip_copy_to_tcm_at(hnnx::GHandle, void *dst_ptr);

// access to 'ubwcd_get_corresponding_surface(tensor_num, is_input, ptr) const'
GHANDLE_ACCESS_FUNC_QUALIFIER Ubwc_D_Surface_Struct const * //
ubwcd_get_corresponding_surface(hnnx::GHandle, int tensor_num, bool is_input, const void *ptr);

GHANDLE_ACCESS_FUNC_QUALIFIER const std::pair<void *, unsigned int> //
ubwcd_get_corresponding_surface_base_ptr_and_stride(hnnx::GHandle, int tensor_num, bool is_input, const void *ptr);
} // namespace GHANDLE_NS

#if GHANDLE_INCLUDE_IMPL // defined in graph_handle_defs.h"
#include "graph_handle_dma_impl.h"
#endif

#endif // GRAPH_HANDLE_DMA_H
