// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef CONSTEXPR_UTIL_H
#define CONSTEXPR_UTIL_H 1
#include <type_traits>

namespace hnnx {
// An expression like the below will work (in a template function, where DT is a constexpr),
//  but it will result in 'uncovered condition' holes, due to the ||.
//
//   constexpr bool DT_is_flt16 = (DT == DType::Float16 || DT == DType::BFloat16);
//
// This can be written instead as
//   constexpr bool DT_is_flt16 = hnnx::any_true_of<(DT == DType::Float16), (DT == DType::BFloat16)>;
//
// or as
//   constexpr bool DT_is_flt16 = hnnx::equals_any_of<DT, DType::Float16, DType::BFloat16>;
//
// The names all_true_of, any_true_of were chosen by a process of eliminating other names
// which are already defined to mean other things, in std:: or in the project.
//
// Note:
// -  'Empty' cases any_true_of<>, and equals_any_of<ignored> are false; all_true_of<> is true.
// -  If you don't put the parens around each condition,  e.g.
//      ...  = hnnx::any_true_of<DT == DType::Float16, DT == DType::BFloat16>;
//    then it will still work, but clang-format will incorrectly guess the syntax, and mess up the spacing.
//
//
template <bool... X> inline constexpr bool all_true_of = (... && X);
template <bool... X> inline constexpr bool any_true_of = (... || X);

template <auto A, auto... B> inline constexpr bool equals_any_of = (... || (A == B));

//
// const_select<COND>(in1, in0) is equivalent to COND ? in1 : in0
// but the condition must be a compile-time const, and this will not show an uncovered condition.
// It can be used with non-constant in1, in0 values; if they are both constexpr, the function call
// will be eligible as constexpr.
//
// The return value does not depend on COND; is the same as what the the type would be if
// you wrote auto x = COND ? in1 : in0;
// See also const_select_auto
//
// **IMPORTANT** Do not use this to directly replace COND ? A : B
//  if either of A, B have side-effects (including function calls) since the side-effects on both will
//  always happen when const_select is used (semantically, it is a function call).
//  E.g.
//        auto x = WITH_NEG ? -(*inp++) : (*inp++);
//  will *not* work as
//        auto x = hnnx::const_select<WITH_NEG>(-(*inp++), *inp++);
//  it could be done as below, factoring out the side effect.
//        auto const x0 = *inp++;
//        auto x = hnnx::const_select<WITH_NEG>(-x0, x0);
//  More complex cases will likely need "if constexpr (COND) ..."
//
template <bool X, typename A, typename B> //
constexpr std::common_type_t<A, B> const_select(A in1, B in0)
{
    if constexpr (X) {
        return in1;
    } else {
        return in0;
    }
}

// const_select_auto<COND>(a, b) is like COND ? a : b
// but the condition must be a compile-time const, and this will not show an uncovered condition.
// It can be used with non-constant in1, in0 values; if they are both constexpr, the function call
// will be eligible as constexpr.
//
// The return value depends on COND, if the types of in1 and in0 are different;
// E.g. with:
//
//     auto x = hnnx::const_select<IS_FLOAT>(1.0f, -1);
//     auto y = hnnx::const_select_auto<IS_FLOAT>(1.0f, -1);
//
// .. 'y' will be of type float when IS_FLOAT is true, and int when it is false; whereas
// 'x' will always be float (it will be 1.0f or -1.0f)
//
// **IMPORTANT** Do not use this to directly replace COND ? A : B
//  if either of A, B have side-effects (including function calls) since the side-effects on both will
//  always happen when const_select_auto is used (semantically, it is a function call).
//  See comment above const_select for example.
//
template <bool X, typename A, typename B> //
constexpr auto const_select_auto(A in1, B in0)
{
    if constexpr (X) {
        return in1;
    } else {
        return in0;
    }
}

} // namespace hnnx

#endif /*CONSTEXPR_UTIL_H*/
