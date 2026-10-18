// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef ARG_TUP_FILTER_H_
#define ARG_TUP_FILTER_H_

#include "op_arg_category.h"

#include <tuple>
#include <type_traits>

namespace hnnx {
//////////
// ArgTupFilter_t<CAT, Args...> -> tuple<Args...> with only ops of given cat removed.
// Also, refs are removed.
//
template <typename T1, typename TUP> struct TupleBuild {};
template <typename T1, typename... Types> struct TupleBuild<T1, std::tuple<Types...>> {
    using type = std::tuple<T1, Types...>;
};

template <OpArgCategory CAT, typename... Types> struct ArgTupFilterHelper {};

template <OpArgCategory CAT, typename T1, typename... Types> struct ArgTupFilterHelper<CAT, T1, Types...> {
  private:
    using tail = typename ArgTupFilterHelper<CAT, Types...>::type;

  public:
    using type = std::conditional_t<OpArgCat<T1>::value == CAT, // is T1 included?
                                    typename TupleBuild<std::remove_reference_t<T1>, tail>::type, tail>;
};

// just one...
template <OpArgCategory CAT, typename T1> struct ArgTupFilterHelper<CAT, T1> {
    using type = std::conditional_t<OpArgCat<T1>::value == CAT, std::tuple<std::remove_reference_t<T1>>, std::tuple<>>;
};

// empty case...
template <OpArgCategory CAT> struct ArgTupFilterHelper<CAT> {
    using type = std::tuple<>;
};

template <OpArgCategory CAT, typename... Types> using ArgTupFilter_t = typename ArgTupFilterHelper<CAT, Types...>::type;
} // namespace hnnx
#endif
