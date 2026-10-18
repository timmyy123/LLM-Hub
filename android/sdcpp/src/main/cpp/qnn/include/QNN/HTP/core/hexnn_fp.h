// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef HEXNN_FP_H
#define HEXNN_FP_H 1

#include "constexpr_util.h" // for hnnx::equals_any_of

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

// in the hexnn_fp namespace we define fpclassify, isfinite, isnan etc, which work on float, double,
// Float16, BFloat16; but *unlike* the std:: versions, you can't pass 'int', or anything that
// converts to a float type. This is deliberately done to avoid potential for confusing results from
// e.g. isnormal(x), where the range depends on the specific type of x.
// This is done by mapping e.g. isinf(F x) to cmath_funcs<F>::isinf(x);
// if the parameter type F is not a directly supported one, it will fail at compile time.
namespace hexnn_fp_priv {

enum fp_comp {
    // used as template parameter in 'do_cmp' internal methods.
    // there is no cmp_NE,  since (x!=y) is identical to !(x==y)
    // also (x > y) --> (y < x),  (x >= y) --> (y <= x)
    cmp_EQ,
    cmp_LT,
    cmp_LE,
    cmp_LT_fmin, // '<' operator for fmin (different NaN, zero behaviour
    cmp_GT_fmax // '>' operator for fmax (different NaN, zero behaviour
};
template <fp_comp C> inline constexpr bool cmp_true_when_eq = hnnx::equals_any_of<C, cmp_EQ, cmp_LE>;
template <fp_comp C> inline constexpr bool cmp_is_valid = (C >= cmp_EQ && C <= cmp_GT_fmax);
template <fp_comp C> inline constexpr bool cmp_for_fminmax = hnnx::equals_any_of<C, cmp_LT_fmin, cmp_GT_fmax>;

template <typename F> //
struct cmath_funcs {
    // the 'sizeof(F)' is only there to delay evaluating the expression until
    // the template is instantiated.
    static_assert(sizeof(F) != sizeof(F), "functions hnnx_fp::isnan() etc. cannot be applied to this type");
};

// this is specialized to 'true' for float and double
template <typename F> inline constexpr bool is_builtin_float = false;

template <> inline constexpr bool is_builtin_float<float> = true;

template <> //
struct cmath_funcs<float> {
    using FType = float;
    static inline FType fabs(FType x) { return std::fabs(x); }
    static inline FType copysign(FType mag, FType sign) { return std::copysign(mag, sign); }
    static inline FType fmin(FType x, FType y) { return std::fmin(x, y); }
    static inline FType fmax(FType x, FType y) { return std::fmax(x, y); }
    static inline int fpclassify(FType x) { return std::fpclassify(x); }
    static inline bool isfinite(FType x) { return std::isfinite(x); }
    static inline bool isinf(FType x) { return std::isinf(x); }
    static inline bool isnan(FType x) { return std::isnan(x); }
    static inline bool isnormal(FType x) { return std::isnormal(x); }
    static inline bool signbit(FType x) { return std::signbit(x); }
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    static inline constexpr bool iszero(FType x) noexcept { return x == FType{}; }
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop // -Wfloat-equal
#endif
};

template <> inline constexpr bool is_builtin_float<double> = true;

template <> //
struct cmath_funcs<double> {
    using FType = double;
    static inline FType fabs(FType x) { return std::fabs(x); }
    static inline FType copysign(FType mag, FType sign) { return std::copysign(mag, sign); }
    static inline FType fmin(FType x, FType y) { return std::fmin(x, y); }
    static inline FType fmax(FType x, FType y) { return std::fmax(x, y); }
    static inline int fpclassify(FType x) { return std::fpclassify(x); }
    static inline bool isfinite(FType x) { return std::isfinite(x); }
    static inline bool isinf(FType x) { return std::isinf(x); }
    static inline bool isnan(FType x) { return std::isnan(x); }
    static inline bool isnormal(FType x) { return std::isnormal(x); }
    static inline bool signbit(FType x) { return std::signbit(x); }
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    static inline constexpr bool iszero(FType x) { return x == FType{}; }
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop // -Wfloat-equal
#endif
};
} // namespace hexnn_fp_priv

namespace hexnn_fp {

using float32_t = float; // compliant with #include <stdfloat> in c++23 ...
using float64_t = double; // .. but in hexnn_fp:: instead of std::

// define template functions fpclassify, etc.
#define HXNN_FP_CMATH_WRAPPER(FNAME, RT)                                                                               \
    template <typename F> inline constexpr RT FNAME(F x)                                                               \
    {                                                                                                                  \
        return hexnn_fp_priv::cmath_funcs<F>::FNAME(x);                                                                \
    }
#define HXNN_FP_CMATH_WRAPPER2(FNAME, RT)                                                                              \
    template <typename F> inline constexpr RT FNAME(F x, F y)                                                          \
    {                                                                                                                  \
        return hexnn_fp_priv::cmath_funcs<F>::FNAME(x, y);                                                             \
    }

template <typename F> inline constexpr F frexp(F x, int *exp_p)
{
    if constexpr (hexnn_fp_priv::is_builtin_float<F>) {
        return std::frexp(x, exp_p);
    } else {
        return hexnn_fp_priv::cmath_funcs<F>::frexp(x, exp_p);
    }
}
template <typename F> inline constexpr F ldexp(F x, int ex)
{
    if constexpr (hexnn_fp_priv::is_builtin_float<F>) {
        return std::ldexp(x, ex);
    } else {
        return hexnn_fp_priv::cmath_funcs<F>::ldexp(x, ex);
    }
}
template <typename F> inline constexpr F floor(F x)
{
    if constexpr (hexnn_fp_priv::is_builtin_float<F>) {
        return std::floor(x);
    } else {
        return hexnn_fp_priv::cmath_funcs<F>::template fl_tr_ce<-1>(x);
    }
}
template <typename F> inline constexpr F trunc(F x)
{
    if constexpr (hexnn_fp_priv::is_builtin_float<F>) {
        return std::trunc(x);
    } else {
        return hexnn_fp_priv::cmath_funcs<F>::template fl_tr_ce<0>(x);
    }
}
template <typename F> inline constexpr F ceil(F x)
{
    if constexpr (hexnn_fp_priv::is_builtin_float<F>) {
        return std::ceil(x);
    } else {
        return hexnn_fp_priv::cmath_funcs<F>::template fl_tr_ce<1>(x);
    }
}
// unordered(x,y) is 'self-contained'.
template <typename F> inline constexpr bool isunordered(F x, F y)
{
    if constexpr (hexnn_fp_priv::is_builtin_float<F>) {
        return std::isunordered(x, y);
    } else {
        return hexnn_fp_priv::cmath_funcs<F>::isnan(x) || hexnn_fp_priv::cmath_funcs<F>::isnan(y);
    }
}
HXNN_FP_CMATH_WRAPPER2(fmin, F)
HXNN_FP_CMATH_WRAPPER2(fmax, F)
HXNN_FP_CMATH_WRAPPER2(copysign, F)
HXNN_FP_CMATH_WRAPPER(fabs, F)
HXNN_FP_CMATH_WRAPPER(fpclassify, int)
HXNN_FP_CMATH_WRAPPER(isfinite, bool)
HXNN_FP_CMATH_WRAPPER(isinf, bool)
HXNN_FP_CMATH_WRAPPER(isnan, bool)
HXNN_FP_CMATH_WRAPPER(isnormal, bool)
HXNN_FP_CMATH_WRAPPER(signbit, bool)
HXNN_FP_CMATH_WRAPPER(iszero, bool)

#undef HXNN_FP_CMATH_WRAPPER
#undef HXNN_FP_CMATH_WRAPPER2
} // namespace hexnn_fp

namespace hexnn_fp_priv {

// Helper for fp_from_raw, fp_to_raw
// see examples for fp_raw<float> etc.
// Will also be specialized for Float16, BFloat16, in the headers.
template <typename F> struct raw_methods;

//
// more or less std::bit_cast<TO, TI>; as much as can be done without c++20
template <typename TO, typename TI>
inline std::enable_if_t<
        std::is_trivially_copyable_v<TI> && std::is_trivially_copyable_v<TO> && sizeof(TI) == sizeof(TO), TO>
bit_cast(TI const &in)
{
    TO result{};
    std::memcpy(&result, &in, sizeof(TO));
    return result;
}

template <> //
struct raw_methods<float> {
    using raw_t = unsigned; // should be same as NN_UINT32_T, do not change to uint32_t.
    static inline float from_raw(raw_t val) noexcept { return bit_cast<float, raw_t>(val); }
    static inline raw_t to_raw(float val) noexcept { return bit_cast<raw_t, float>(val); }
};

template <> //
struct raw_methods<double> {
    using raw_t = uint64_t;
    static inline double from_raw(raw_t val) noexcept { return bit_cast<double, raw_t>(val); }
    static inline raw_t to_raw(double val) noexcept { return bit_cast<raw_t, double>(val); }
};

} // namespace hexnn_fp_priv

namespace hexnn_fp {

// Methods for converting float types to/from 'raw' codes.
// e.g.
//  hnnx::fp_from_raw<float>(0x40490FDB) -> 3.1415927f;
//  hnnx::fp_to_raw(3.1415926535897931) ->  0x400921FB54442D18;
//
// - Can be used with Float16, BFloat16 if those headers are included.
// - These are not really constexpr, except for Float16, BFloat16.
// It will be possible to make all of them constexpr if c++20 is used (via std::bit_cast)
//

template <typename F> //
inline constexpr F from_raw(typename hexnn_fp_priv::raw_methods<F>::raw_t rawval) noexcept
{
    return hexnn_fp_priv::raw_methods<F>::from_raw(rawval);
}
template <typename F> //
inline constexpr auto to_raw(F val) noexcept // -> typename hexnn_fp_priv::raw_methods<F>::raw_t
{
    return hexnn_fp_priv::raw_methods<F>::to_raw(val);
}

// ALSO:
// hnnx::fp_from_float_alt<F>(float) -> F
// hnnx::fp_to_float_alt(F val) -> float
// - Only defined for Float16; the 'alt' mode treats exp=31 as extended
//   'normal' range, and doesn't have inf/Nan.
//
// float16.h defines:
//    constexpr Float16 hexnn_fp::float16_from_float_alt(float f) noexcept
//      (same as hexnn_fp::fp_from_float_alt<Float16>(f))
//
template <typename F> //
inline constexpr F fp_from_float_alt(float val) noexcept
{
    return hexnn_fp_priv::raw_methods<F>::from_float_alt(val);
}

// fp_to_float_alt is iplemented this way (instead of as an overload)
// to block implicit conversions; if applied to e.g. a float value,
// it will fail to compile, instead of becoming
// fp_to_float_alt(Float16(floatval)). Hopefully the comment
// below will appear in the error message.
//
template <typename F> //
inline constexpr float fp_to_float_alt(F val) noexcept
{
    return hexnn_fp_priv::raw_methods<F>::to_float_alt(val); // fails if fp_to_float_alt applied to wrong type.
}

} // namespace hexnn_fp

#endif /*HEXNN_FP_H*/
