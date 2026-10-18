// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef REPL_FUNCS_H_
#define REPL_FUNCS_H_

#include "dtype_enum.h"
#include "opt_function.h"

namespace constraint_lib {
class Constraint;
}
namespace oExp {
using ECtx = constraint_lib::Constraint;
template <typename T> using sFunction = OptFunction<T(ECtx &)>;
} // namespace oExp

class Replacement;
class OpDef;
class OpRef;
using ReplFunc = OptFunction<OpRef(Replacement &, OpDef const &)>;

namespace hnnx {

using ReplFuncInt = oExp::sFunction<int>;
using ReplFuncBool = oExp::sFunction<bool>;
using ReplFuncDType = oExp::sFunction<DType>;
using ReplFuncFloat = oExp::sFunction<float>;
} // namespace hnnx
#endif
