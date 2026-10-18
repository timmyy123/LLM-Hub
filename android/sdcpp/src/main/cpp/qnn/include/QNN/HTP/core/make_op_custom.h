// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef MAKE_OP_CUSTOM_H_
#define MAKE_OP_CUSTOM_H_

#include "weak_linkage.h"
#include "macros_attribute.h"
#include "op_factory.h"

#include <string_view>

namespace hnnx {
struct op_reg_parms;

PUSH_VISIBILITY(default)

API_EXPORT OpFactory make_op_custom_internal(const std::string_view op_name_in, const std::string_view type_tag,
                                             op_reg_parms const &opreg_parms, bool is_external = false,
                                             const std::string_view file_name = "", int line_number = 0,
                                             const std::string_view target_reg = "core");

API_EXPORT OpFactory make_op_custom(const std::string_view op_name_in, std::string_view const type_tag,
                                    op_reg_parms const &opreg_parmsm, bool is_legacy,
                                    std::string_view const file_name = "", int line_number = 0,
                                    const std::string_view target_reg = "core");

POP_VISIBILITY()
} // namespace hnnx
#endif
