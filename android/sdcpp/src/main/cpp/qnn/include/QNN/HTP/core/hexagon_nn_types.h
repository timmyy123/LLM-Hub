#pragma once
// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// We need this so, as on Windows, long is just 32-bits.  This way, Long is consistently 64-bits on
// 64-bit architectures (x86, aarch64 on Linux, Android, Windows, QNX, etc.).
typedef ptrdiff_t Long;

///
/// @brief Max number of PMU events HexNN can sample
///
#define HEXAGON_NN_MAX_PMU_EVENTS 8

///
/// @brief Type for 32b (virtual) address
///
typedef uint32_t hexagon_nn_address_t;

///
/// @brief Type for 64b (virtual) address
///
typedef uint64_t hexagon_nn_wide_address_t;

///
/// @brief A visual marker for an address whose contents (the thing this points
/// to) are immutable
/// @details For example a pointer to a shared weights table. The table has a
/// list of near/far pointers whose contents (weights) are considered immutable
///
typedef uint64_t hexagon_nn_wide_address_const_t;

///
/// @brief Type for iovec with 32b pointer/address and size
///
typedef struct {
    hexagon_nn_address_t val;
    uint32_t len;
} hexagon_nn_iovec_t;

///
/// @brief Type for iovec with 64b pointer/address and size
///
typedef struct {
    hexagon_nn_wide_address_t val;
    uint64_t len;
} hexagon_nn_wide_iovec_t;

///
/// @brief Per-package trace address info for a loaded custom op package.
///
/// Used with hexagon_nn_get_trace_oppkg_info() for custom op symbol resolution
/// in optrace_to_chrometrace.py.
///
typedef struct {
    unsigned faddr; ///< Runtime DSP address of op_pkg_init (function load-offset anchor).
    unsigned saddr; ///< Runtime DSP address of the package name string (rodata anchor).
    char filename[512]; ///< Path used to load the package (NUL-terminated, <=511 chars).
    char name[64]; ///< Package self-declared name (NUL-terminated, <=63 chars).
} hexagon_nn_pkg_trace_info_t;

typedef struct {
    ///
    /// @brief DSP-side *NEAR* pointer to I/O tensors
    /// @details This is essentially hexagon_nn_wide_tensordef_t[num_tensors]
    /// @warning @a tensors is hexagon_nn_wide_address_t for testing on x86. Its
    /// value on-device MUST BE near!
    ///
    hexagon_nn_wide_address_t tensors;

    ///
    /// @brief Number of tensors pointed to be @a tensors
    ///
    uint32_t num_tensors;

    ///
    /// @brief Reserved/unused
    ///
    uint32_t reserved;

} hexagon_nn_io_tensors_t;

///
/// @brief Argument type to pass sub-graph execute information
/// @details Typically associated with HEXAGON_NN_ARG_TAG_SUBGRAPH_EXECUTE_PUSH
///
typedef struct {
    ///
    /// @brief Handle to a subgraph to run
    /// @details This is essentially nn_graph_t. It is sized to 64b to hold a
    /// pointer on both 64b (x86) and 32b (Hexagon) architectures
    ///
    uint64_t graph_handle;

    ///
    /// @brief Number of iterations to run
    ///
    uint32_t run_iters;

    ///
    /// @brief Reserved/unused
    /// @warning MUST BE initialized to zero!
    ///
    uint32_t reserved;

} hexagon_nn_subgraph_execute_t;

///
/// @brief Used to specify thread types when calling hexagon_nn_set_thread_count
/// and hexagon_nn_get_thread_count.
///
enum hexagon_nn_thread_type_t {
    // Use these enums to specify the type of thread for hexagon_nn_set_thread_count.
    VecThread = 0,
    MtxThread = 1,
    EltThread = 2,
    // Use this for `count` to specify that the maximum available number of threads should be used.
    MaxOsThreads = 1001,
};

///
/// @brief Type for specifying the preemption scheme
///
typedef enum { COOP, FORCED, DEFERRED, ORDERED_COOP } hexagon_nn_preemption_style_t;

enum MemContentType {
    Standard = 0,
    Weight = 1,
    WeightDLBC = 2,
    WeightReplaceable = 3,
    ExtendedRO, ///< Content mapped to far memory with read-only permissions
    ExtendedRW ///< Content mapped to far memory with read-write permissions
};

///
/// @brief A NULL wide IO vector
///
/// @details Equivalent to nullptr for a pointer instance. Can be used as
/// default value for arguments
///
static hexagon_nn_wide_iovec_t const NULL_IOVEC = {0ull, 0ull};

#ifdef __cplusplus
}
#endif
