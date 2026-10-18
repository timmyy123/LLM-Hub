// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef OPTIMIZE_FWD_H_
#define OPTIMIZE_FWD_H_

#ifndef PREPARE_DISABLED

#include "macros_attribute.h"
#include "weak_linkage.h"
#include "optimize_flags.h"
#include "entire_defopt.h"

#include <vector>
#include <memory>
#include <map>
#include <string_view>

namespace hnnx {
class GraphOptInfo;
}

API_HIDDEN std::vector<std::unique_ptr<hnnx::GraphOptInfo>> &current_package_opts_storage_vec_func();

//
//

PUSH_VISIBILITY(default)

namespace hnnx {
class GraphOptPass;

API_EXPORT std::map<unsigned int, GraphOptPass> &get_optimization_passes();

API_EXPORT std::map<unsigned int, GraphOptPass> &get_optimization_passes(const std::string_view optimization_registry);

// Returns true if the named registry exists and has at least one registered optimization pass.
// Unlike get_optimization_passes(), this does NOT create an empty entry for unknown names.
API_EXPORT bool has_optimization_passes(const std::string_view optimization_registry);

API_EXPORT void merge_optimization_passes(std::map<unsigned int, GraphOptPass> &merged_registries,
                                          const std::vector<std::string> &registry_list);

API_EXPORT std::map<std::string, std::vector<std::unique_ptr<GraphOptInfo>> *> &get_pkg_opt_tmp_map();

API_EXPORT void add_package_opt(std::vector<std::unique_ptr<GraphOptInfo>> &opts, int priority,
                                OptimFlags::flags_t flags_in, get_entire_defopt_t defopt_in, char const *const fname,
                                const int lineno);
API_EXPORT void add_package_opt(std::vector<std::unique_ptr<GraphOptInfo>> &opts, int priority,
                                OptimFlags::flags_t flags_in, get_entire_defopt_t defopt_in, char const *const fname,
                                const int lineno, std::string_view target_reg);
// This entry is only for backwards ABI compatibility for exising op packages
// compiled when fname and line number were not in the default build.
API_EXPORT void add_package_opt(std::vector<std::unique_ptr<GraphOptInfo>> &opts, int priority,
                                OptimFlags::flags_t flags_in, get_entire_defopt_t defopt_in);

API_EXPORT std::string get_opname_with_default_pkg_prefix(char const *opname);

} // namespace hnnx

POP_VISIBILITY()

#endif
#endif
