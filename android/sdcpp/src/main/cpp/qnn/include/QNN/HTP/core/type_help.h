// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef TYPE_HELP_H
#define TYPE_HELP_H 1
#include <type_traits>
#include <typeinfo>
#include <cstddef>

namespace hnnx {

//////////////////////////////////////////////////////////////
// value_proxy<T,UNORDERED>::type
//    maps 'long' to either int or to long long;
//    unsigned long to unsigned or to unsigned long long;
//    Any pointer is mapped to whatever size_t maps to;
//    if UNORDERED: also maps signed integer types to unsigned
//    all other types are unchanged
//
// This is useful to reduce code bloat; e.g. a 'set' class operating
//  on T can actually use type_proxy<T> internally, so that e.g. set
//  of 'void *' and set of 'int const *' will both use the same templated code
//  as size_t.
//

template <typename T, bool UNORDERED> struct value_proxy {
    typedef T type;
};
// map unsigned long to unsigned int or to unsigned long long
template <bool UNORDERED> struct value_proxy<unsigned long, UNORDERED> {
    typedef std::conditional_t<(sizeof(unsigned long) <= 4), unsigned, unsigned long long> type;
};
// likewise for long
template <> struct value_proxy<long, false> {
    typedef std::conditional_t<(sizeof(long) <= 4), int, long long> type;
};
template <> struct value_proxy<long, true> {
    typedef typename value_proxy<unsigned long, true>::type type;
};

// map signed types to unsigned, if unordered
template <> struct value_proxy<short, true> {
    typedef unsigned short type;
};
template <> struct value_proxy<int, true> {
    typedef unsigned type;
};
template <> struct value_proxy<long long, true> {
    typedef unsigned long long type;
};
template <> struct value_proxy<signed char, true> {
    typedef unsigned char type;
};

// all pointer -> size_t-> either unsigned or ull
template <typename T, bool UNORDERED> struct value_proxy<T *, UNORDERED> {
    typedef typename value_proxy<size_t, UNORDERED>::type type;
    static_assert(sizeof(type) == sizeof(T *));
};
} // namespace hnnx

#endif
