// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef OP_REGISTER_H
#define OP_REGISTER_H 1

#include "c_tricks.h"
#include "op_registry.h"
#include "serialize_register.h"
#include "cost_funcs.h"
#include "op_info.h"
#include "op_register_types.h"
#include "op_package_name.h"
#include "template_help.h"
#include "weak_linkage.h"
#include "op_version.h"
#include "op_register_macros.h"
#include "make_op_custom.h"

#include <memory>
#include <string>
#include <utility>

namespace hnnx {
struct item_return {
    typedef op_reg_parms type;
};

// parms_for is wrapped in this class to avoid if constexpr implementation since
// the AUTOSAR checker doesn't evaluate if constexpr blocks properly
// LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
// used in pub/impl/ops_opts_registration_defs.h for internal ops with constexpr lvalue
class GetParms {
  public:
    template <typename Derived, int I> constexpr static typename item_return::type get()
    {
        return op_reg_parms::parms_for<Derived, FlagCounter<Derived, I>::get()>();
    }

    template <auto FP, int I> constexpr static typename item_return::type get()
    {
        using Derived = typename DerivedType<FP>::type;
        return op_reg_parms::parms_for<Derived, FlagCounter<Derived, I>::get()>();
    }
};

//LCOV_EXCL_STOP

} // namespace hnnx

/** ModifiedDerivedType is used to perform a transformation from
 * Tensor_TCM -> Tensor for different tensor types. Both FLAGS_FOR and
 * APPEND_REG_OP_ELEM use this metafunction to implement TCM folding for execute.
 * For more details, see docs/register-op-tcm-folding.md
 */
namespace fold {
template <auto, int> struct ModifiedDerivedType;
} //namespace fold
#endif
