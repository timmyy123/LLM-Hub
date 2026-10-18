// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef PACKAGE_OP_STORAGE_BASE_H_
#define PACKAGE_OP_STORAGE_BASE_H_

#include "weak_linkage.h"
#include "macros_attribute.h"
#include "op_factory.h"
#include "op_reg_parms.h"

#include <map>
#include <string>
#include <string_view>
#include <vector>

PUSH_VISIBILITY(default)

namespace hnnx {

class PackageOpStorageBase {
  public:
    const std::string op_name;
    const std::string_view type_tag;
    const bool is_external;
    const op_reg_parms opreg_parms;
    const std::string_view target_reg;

    API_EXPORT PackageOpStorageBase(const std::string_view op_name_in, const std::string_view type_tag_in,
                                    const bool is_external_in, const op_reg_parms opreg_params_in,
                                    const std::string_view target_reg_in = "core");
    API_EXPORT OpFactory make_op_wrapper() const;
};

// The map to store op package ops
API_EXPORT std::map<std::string, std::vector<std::unique_ptr<PackageOpStorageBase>> *> &get_pkg_op_tmp_map();

} // namespace hnnx

POP_VISIBILITY()
#endif
