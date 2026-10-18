// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef OP_REGISTRY_H
#define OP_REGISTRY_H 1

#include "op.h"
#include "op_def.h"
#include "weak_linkage.h"
#include "compile_time_string.h"
#include "op_factory.h"

#include <map>
#include <memory>
#include <string>

/*
 * We need a way to generate ops.  This known as a "factory", being able to
 * return an appropriate op based on the op type and maybe other factors.
 *
 * How do we know what ops exist?  Maintaining a list is not acceptable: it is
 * inflexible, requires constant code changes, is error prone, doesn't allow
 * dynamic op addition, etc.
 *
 * So we must have some mechanism to register ops with the op factory.
 *
 * The interface is pretty simple: register an op with the registry, or generate
 * an op that exists in the registry
 *
 * We're going to use strings to indicate the type of operation, this is less
 * error prone (mismatching IDs) than a numeric ID.
 *
 * The factory has to be virtualized in some way: Map a string to a function
 * which generates an op.  Using templated classes, however, generates a lot of
 * extra code.  To keep this smaller, a bare function pointer is used.  This
 * generates a unique_ptr which wraps the op.  If the op cannot be created, then
 * the unique_ptr contains nullptr.  Each factory function is actually a static
 * member of a concrete template class, but since static member functions have
 * the same signature as simple functions, we can just use a function pointer.
 *
 */

namespace hnnx {
struct op_reg_info_t {
    OpFactory op_factory{nullptr}; // function pointer for generating ops
    bool is_external{false};
};
using OpRegistry_map_t = std::multimap<opname_tag_t, struct op_reg_info_t>;

PUSH_VISIBILITY(default)
/*
	 * We register an op with the registry by giving it a name, and a std::unique_ptr to a generator for that op.
	 * Returns a reference to the op once emplaced.
	 * Why? Because that allows us to create static variables with the results, causing the functions to be loaded automatically...
	 */
extern API_FUNC_EXPORT OpFactory register_op(opname_tag_t name, OpFactory newop, bool is_external,
                                             std::string_view target_reg = "core");

/*
	 * Generate an op
	 */
API_FUNC_EXPORT uptr_Op op_factory_generate(OpIoPtrs const &op_io_ptrs, OpId id_in);

/**Function returns a reference for the registered ops structure
	 * This function currently enables unit tests on the op registry
	 *
	 */
extern API_FUNC_EXPORT const OpRegistry_map_t &get_registered_ops();
extern API_FUNC_EXPORT const OpRegistry_map_t &get_registered_ops(std::string_view target_reg);

// Function clean up all package(external) ops from op maps
extern API_FUNC_EXPORT void clear_pkg_ops_in_op_maps();

// register_optype_by_factory populates the op_factory_map with type info during
// pre-main static init. Weak no-op by default; overridden in prepare-enabled builds.
//
API_FUNC_EXPORT void register_optype_by_factory(OpFactory fp, hnnx::opname_tag_t opname_tag, std::type_info const &typ,
                                                const std::string_view type_tag);

POP_VISIBILITY()
} // namespace hnnx

#endif /*OP_FACTORY_H*/
