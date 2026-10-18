// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef OP_REG_PARMS_H_
#define OP_REG_PARMS_H_

#include "define_deserialization_enabled.h"
#include "cost_funcs.h"
#include "flags.h"
#include "op.h"
#include "op_factory.h"
#if DESERIALIZATION_ENABLED == 1
#include "size_align_code.h"
#endif

#include <typeinfo>

namespace hnnx {
// package of info for op construction.
// LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
struct op_reg_parms {
#ifndef PREPARE_DISABLED
    OpFactory newop;
    std::type_info const *tinf;
#endif
#if DESERIALIZATION_ENABLED == 1
    Op::tensor_deserializer_register_func deserializer_reg_func;
    deserialize_op_func deserialize_func;
    deserialize_dtor_func deserialize_dtor;
#endif
#ifndef PREPARE_DISABLED
    cost_function_t cost_f;
    Flags_word flags;
#endif
#if DESERIALIZATION_ENABLED == 1
    size_align_code_t size_align_code;
    inline constexpr size_t get_size() const { return size_align_code.size(); }
    inline constexpr size_t get_align() const { return size_align_code.align(); }
#endif
    template <typename Derived, int N> static constexpr op_reg_parms parms_for();

    constexpr bool operator==(op_reg_parms const &other) const noexcept
    {
#ifndef PREPARE_DISABLED
        if ((newop != other.newop) || (*tinf != *other.tinf)) {
            return false;
        }
#endif

#if DESERIALIZATION_ENABLED == 1
        if ((deserializer_reg_func != other.deserializer_reg_func) || (deserialize_func != other.deserialize_func) ||
            (deserialize_dtor != other.deserialize_dtor)) {
            return false;
        }
#endif

#ifndef PREPARE_DISABLED
        if ((cost_f != other.cost_f) || (flags != other.flags)) {
            return false;
        }
#endif

#if DESERIALIZATION_ENABLED == 1
        if (size_align_code != other.size_align_code) {
            return false;
        }
#endif

        return true;
    }
};
// LCOV_EXCL_STOP
} // namespace hnnx
#endif
