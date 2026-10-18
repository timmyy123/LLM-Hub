// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef OP_REGISTER_TYPES_H
#define OP_REGISTER_TYPES_H 1

#include "op_registry.h"
#include "serialize_register.h"
#include "cost_funcs.h"
#include "op_info.h"
#include "op_package_name.h"
#include "size_align_code.h"
#include "define_deserialization_enabled.h"

#include <memory>
#include <string>
#include <utility>

#include "op_reg_parms.h"

namespace hnnx {
// LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
// generate an 'op_reg_parms' for a given Op type.
// this should be expanded only once for each Derived, so we want it inlined.
template <typename Derived, int N> [[gnu::always_inline]] constexpr op_reg_parms op_reg_parms::parms_for()
{
    return op_reg_parms{
#ifndef PREPARE_DISABLED
            Derived::create,
            &typeid(Derived),
#endif
#if DESERIALIZATION_ENABLED == 1
            Derived::get_tensor_deserializer_register_func(),
            test_flag_for(flags_for<Derived, N>, Flags::IS_CONST) ? nullptr : alloc_func_for_op<Derived>::alloc_func,
            !std::is_trivially_destructible<Derived>::value ? dealloc_func_for_op<Derived>::func : nullptr,
#endif
#ifndef PREPARE_DISABLED
            get_costf<Derived>(),
            flags_for<Derived, N>,
#endif
#if DESERIALIZATION_ENABLED == 1
            alloc_func_for_op<Derived>::op_size_align,
#endif
    };
}
// LCOV_EXCL_STOP
} // namespace hnnx
#endif
