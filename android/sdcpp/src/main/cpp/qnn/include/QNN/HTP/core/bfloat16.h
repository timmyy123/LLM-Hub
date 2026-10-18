// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef BFLOAT16_H
#define BFLOAT16_H

#include <cassert>
#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>

#include "builtin_intrinsics.h"
#include "hexnn_fp.h"

#include "weak_linkage.h"
#include "macros_attribute.h"

PUSH_VISIBILITY(default)

struct BFloat16;

namespace hexnn_fp {
using bfloat16_t = BFloat16;
} // namespace hexnn_fp

struct API_EXPORT BFloat16 {
    friend struct hexnn_fp_priv::cmath_funcs<BFloat16>;
    friend struct hexnn_fp_priv::raw_methods<BFloat16>;
    friend class std::numeric_limits<BFloat16>;

  protected:
    static constexpr unsigned k_no_sign_mask = 0x7fffu;
    static constexpr unsigned k_sign_mask = 0x8000u;
    static constexpr unsigned k_inf_code = 0x7f80u;
    static constexpr unsigned k_mantissa_mask = 0x007f;
    static constexpr unsigned k_quiet_bit = 0x040u;
    static constexpr unsigned k_smallest_norm_code = k_mantissa_mask + 1u;

    // To support conversion from integer to BFloat16
    // This is a bit tricky: values in +/-(1<<24) can be converted losslessy to float,
    // then to BFloat16; for values outside that range, we use from_big_int and from_big_uint
    // to avoid 'double rounding' errors.
    // The smallest value that fails in the float conversion is 0x1010001, which should
    // round up to a BFloat16 value representing 0x1020000; but if converted to float,
    // that rounds down to float(0x1010000), which will round down to 0x1000000 in BFloat16.
    static BFloat16 from_big_int(int64_t) noexcept;
    static BFloat16 from_big_uint(uint64_t, bool sign = false) noexcept;
    // helper for BFloat16(I) : only valid for I being an integer type
    template <typename I> static inline BFloat16 ctor_from_int(I val) noexcept
    {
        if constexpr (!std::is_signed_v<I>) {
            if (val <= 0x1010000u) { // ok range
                return BFloat16(static_cast<float>(static_cast<unsigned>(val)));
            }
            return from_big_uint(val);
        } else {
            if (val <= 0x1010000 && val >= -0x1010000) { // lossless convert to float
                return BFloat16(static_cast<float>(static_cast<int>(val)));
            }
            return from_big_int(val);
        }
    }

  public:
    constexpr BFloat16() noexcept : d(0) {}
    constexpr BFloat16(float f) noexcept;
    // 'noexcept' on defaulted operations is enforced with static_assert af6er class. Do not add here.
    constexpr BFloat16(const BFloat16 &f) = default;
    constexpr BFloat16 &operator=(const BFloat16 &f) = default;
    constexpr BFloat16(BFloat16 &&f) = default;
    constexpr BFloat16 &operator=(BFloat16 &&f) = default;
    ~BFloat16() = default;

    // Conversion from double -- not always the same as BFloat16(float(val))
    BFloat16(double val) noexcept;
    // 'catch-all' for conversion from integer types.
    template <typename I, std::enable_if_t<std::is_integral_v<I>, bool> = false> //
    BFloat16(I val) noexcept : BFloat16(ctor_from_int<I>(val))
    {
    }

    constexpr operator float() const noexcept;

    // NOTE: move these three to 'protected', once they are removed from Op code.
    inline constexpr bool is_nan() const noexcept { return (d & k_no_sign_mask) > k_inf_code; }
    constexpr uint16_t raw() const noexcept { return d; }
    static constexpr BFloat16 from_raw(uint16_t v) noexcept;

  protected:
    uint16_t d;

    template <enum hexnn_fp_priv::fp_comp CMP_OP> //
    constexpr bool do_cmp(BFloat16 other) noexcept;

  public:
    friend constexpr BFloat16 operator-(BFloat16 a) noexcept;
    friend FUNC_ATTRIB_CONST API_FUNC_EXPORT BFloat16 operator+(BFloat16 a, BFloat16 b) noexcept;
    inline friend BFloat16 operator-(BFloat16 a, BFloat16 b) noexcept { return operator+(a, -b); }
    friend FUNC_ATTRIB_CONST API_FUNC_EXPORT BFloat16 operator*(BFloat16 a, BFloat16 b) noexcept;
    friend FUNC_ATTRIB_CONST API_FUNC_EXPORT BFloat16 operator/(BFloat16 a, BFloat16 b) noexcept;

    friend constexpr bool operator==(BFloat16 a, BFloat16 b) noexcept;
    friend constexpr bool operator!=(BFloat16 a, BFloat16 b) noexcept;
    friend constexpr bool operator<(BFloat16 a, BFloat16 b) noexcept;
    friend constexpr bool operator>(BFloat16 a, BFloat16 b) noexcept;
    friend constexpr bool operator<=(BFloat16 a, BFloat16 b) noexcept;
    friend constexpr bool operator>=(BFloat16 a, BFloat16 b) noexcept;
};

POP_VISIBILITY()

//
// ** If you make a change which causes any of the static_assert below
// ** to be violated, please reconsider the change before removing the assert.
// ** Such a change will affect the size and performance of code using BFloat16
// ** (E.g. it may not be possible to pass a BFloat16 by value, without extra
// ** overhead). Do NOT add 'noexcept' to defaulted methods to make the asserts
// ** go away.
static_assert(std::is_nothrow_copy_constructible<BFloat16>::value);
static_assert(std::is_nothrow_move_constructible<BFloat16>::value);
static_assert(std::is_nothrow_assignable<BFloat16, BFloat16>::value);
static_assert(std::is_nothrow_move_assignable<BFloat16>::value);
static_assert(std::is_nothrow_destructible<BFloat16>::value);
static_assert(std::is_trivially_copyable<BFloat16>::value);
static_assert(std::is_standard_layout<BFloat16>::value);

// specialize hexnn_fp_priv::raw_methods<BFloat16>
// makes hexnn_fp::from_raw<BFloat16>(rawval),
// and hexnn_fp::to_raw(BFloat16 f) work.
namespace hexnn_fp_priv {
template <> struct raw_methods<BFloat16> {
    using raw_t = uint16_t;
    static inline constexpr BFloat16 from_raw(raw_t val) noexcept { return BFloat16::from_raw(val); }
    static inline constexpr raw_t to_raw(BFloat16 val) noexcept { return val.raw(); }
};
} // namespace hexnn_fp_priv

namespace hexnn_fp {
// 'convenience' wrapper ... alternative to hexnn_fp::from_raw<BFloat16>(rawval)
inline constexpr BFloat16 bfloat16_from_raw(uint16_t rawval) noexcept
{
    return hexnn_fp_priv::raw_methods<BFloat16>::from_raw(rawval);
}
} // namespace hexnn_fp

// The following are needed to resolve ambiguities when comparing things to BFloat16,
// e.g (bf16_val == 1) -> float(bf16_val) == 1  -> float(bf16_val) == float(1)
#define BF16_COMPARE_ADAPTER(CMP_OP)                                                                                   \
    template <typename T>                                                                                              \
    inline constexpr std::enable_if_t<std::is_arithmetic_v<T>, bool> operator CMP_OP(T a, BFloat16 b) noexcept         \
    {                                                                                                                  \
        return a CMP_OP float(b);                                                                                      \
    }                                                                                                                  \
    template <typename T>                                                                                              \
    inline constexpr std::enable_if_t<std::is_arithmetic_v<T>, bool> operator CMP_OP(BFloat16 a, T b) noexcept         \
    {                                                                                                                  \
        return float(a) CMP_OP b;                                                                                      \
    }

BF16_COMPARE_ADAPTER(==)
BF16_COMPARE_ADAPTER(!=)
BF16_COMPARE_ADAPTER(<)
BF16_COMPARE_ADAPTER(>)
BF16_COMPARE_ADAPTER(<=)
BF16_COMPARE_ADAPTER(>=)
#undef BF16_COMPARE_ADAPTER

inline constexpr BFloat16 BFloat16::from_raw(uint16_t v) noexcept
{
    BFloat16 f;
    f.d = v;
    return f;
}

inline constexpr BFloat16::BFloat16(float f) noexcept : d(0)
{
    uint32_t val = hexnn_fp::to_raw<float>(f);

    // Preserve NaN values
    // The only potential NaN values that can be lost are the ones that have an exp=0xFF and a non 0 bit in the 16 lsb
    if ((val & 0x7F80FFFFu) > 0x7F800000u) {
        d = k_inf_code | (k_quiet_bit >> 1u); // non-quiet NaN
        return;
    }
    // just round it to nearest multiple of 2^16.
    // Preserves sign, zeros, inf.
    // Some values will round up to next exponent, or from EXP=126 to inf,
    // or from large subnormal to 'smallest normal';
    // or down from small subnormal to 0.0. But all of that is numerically correct.
    //
    // rounding bias is 0x7fff if bit 16 is 0, and 0x8000
    // if bit 16 is 1; so we round-to-even when halfway
    uint32_t const bias = 0x7FFFU + ((val >> 16u) & 1u);
    val += bias;
    // upper 16 bits are the result.
    d = static_cast<uint16_t>(val >> 16u);
}

inline constexpr BFloat16::operator float() const noexcept
{
    unsigned const tmp = static_cast<uint32_t>(raw()) << 16u;
    return hexnn_fp::from_raw<float>(tmp);
}

inline constexpr BFloat16 operator-(BFloat16 x) noexcept
{
    return BFloat16::from_raw(x.d ^ BFloat16::k_sign_mask);
}

// Compares are done by this function. When used as in bf16_val == BFloat16{}, this should
// optimize to just: (bf16_val.d & 0x7fff) == 0.
template <enum hexnn_fp_priv::fp_comp CMP_OP> //
inline constexpr bool BFloat16::do_cmp(BFloat16 other) noexcept
{
    using namespace hexnn_fp_priv;
    static_assert(hexnn_fp_priv::cmp_is_valid<CMP_OP>);

    unsigned a_code = d;
    unsigned b_code = other.d;
    unsigned const a_abs = a_code & k_no_sign_mask;
    unsigned const b_abs = b_code & k_no_sign_mask;
    unsigned const m = std::max(a_abs, b_abs);
    if (m > BFloat16::k_inf_code) {
        if constexpr (cmp_for_fminmax<CMP_OP>) {
            // for fmin/fmax: select the one which is *not* NaN where possible
            // the context is:
            //    fmin(a, b) -> (a LT_fmin b) ? a : b
            //    fmax(a, b) -> (a GT_fmax b) ? a : b
            // So, select a if B is NaN, else b.
            return b_abs > BFloat16::k_inf_code;
        } else {
            return false; // normal compare: always false if any NaN
        }
    }
    if constexpr (!cmp_for_fminmax<CMP_OP>) {
        // special case, both are +/- 0
        // (for fmin/fmax, 0.0 is considered > -0.0)
        if (m == 0) return cmp_true_when_eq<CMP_OP>; // special case, both are +/- 0
    }

    bool result{};
    if constexpr (CMP_OP == cmp_EQ) {
        result = a_code == b_code;
    } else {
        // If either a or b has sign bit, swap them
        if (((a_code | b_code) & k_sign_mask) != 0) {
            std::swap(a_code, b_code);
        }
        // They are now equal iff they were equal before; otherwise they can be compared
        // using unsigned compare, and result will be correct for the represented values.
        // (if a,b had different signs, the one which was + is now larger; if both negative,
        // the reversal has compensated for reverse encoding of values).
        if constexpr (CMP_OP == cmp_LE) {
            result = a_code <= b_code;
        } else if constexpr (CMP_OP == cmp_GT_fmax) {
            result = a_code > b_code;
        } else { // LT or LT_fmin
            result = a_code < b_code;
        }
    }
    return result;
}

// some compilers require BFloat16::do_cmp to be defined before it's used...
constexpr bool operator==(BFloat16 a, BFloat16 b) noexcept
{
    return a.do_cmp<hexnn_fp_priv::cmp_EQ>(b);
}
constexpr bool operator!=(BFloat16 a, BFloat16 b) noexcept
{
    return !a.do_cmp<hexnn_fp_priv::cmp_EQ>(b);
}
constexpr bool operator<(BFloat16 a, BFloat16 b) noexcept
{
    return a.do_cmp<hexnn_fp_priv::cmp_LT>(b);
}
constexpr bool operator>(BFloat16 a, BFloat16 b) noexcept
{
    return b.do_cmp<hexnn_fp_priv::cmp_LT>(a);
}
constexpr bool operator<=(BFloat16 a, BFloat16 b) noexcept
{
    return a.do_cmp<hexnn_fp_priv::cmp_LE>(b);
}
constexpr bool operator>=(BFloat16 a, BFloat16 b) noexcept
{
    return b.do_cmp<hexnn_fp_priv::cmp_LE>(a);
}

PUSH_VISIBILITY(default)
template <> class API_EXPORT std::numeric_limits<BFloat16> {
  public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr auto has_denorm = std::denorm_present;
    static constexpr bool has_denorm_loss = false; // libc++
    static constexpr auto round_style = std::round_to_nearest;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 8;
    static constexpr int digits10 = 2; // floor((digits-1) * log10(2))
    static constexpr int max_digits10 = 4; // ceil(digits * log10(2) + 1)
    static constexpr int radix = 2;
    static constexpr int min_exponent = -125;
    static constexpr int min_exponent10 = -37; // float32 value
    static constexpr int max_exponent = 128;
    static constexpr int max_exponent10 = 38; // largest finite val = 3.3895314E38
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false; // libc++

    static constexpr BFloat16 min() noexcept; // returns min positive normal
    static constexpr BFloat16 lowest() noexcept; // returns true min
    static constexpr BFloat16 max() noexcept; // max positive
    static constexpr BFloat16 epsilon() noexcept; // step at 1.0
    static constexpr BFloat16 round_error() noexcept; // 0.5
    static constexpr BFloat16 infinity() noexcept;
    static constexpr BFloat16 quiet_NaN() noexcept;
    static constexpr BFloat16 signaling_NaN() noexcept;
    static constexpr BFloat16 denorm_min() noexcept; // min positive denorm
};

POP_VISIBILITY()

constexpr BFloat16 std::numeric_limits<BFloat16>::min() noexcept
{
    // 0 0000 0001 0000000
    return BFloat16::from_raw(uint16_t(BFloat16::k_smallest_norm_code));
}

constexpr BFloat16 std::numeric_limits<BFloat16>::lowest() noexcept
{
    // -2^127 * (1.9921875)  ; 1 1111 1110 1111 111
    return BFloat16::from_raw(uint16_t(BFloat16::k_sign_mask | (BFloat16::k_inf_code - 1))); // -3.3895314E38
}

constexpr BFloat16 std::numeric_limits<BFloat16>::max() noexcept
{
    return BFloat16::from_raw(uint16_t(BFloat16::k_inf_code - 1));
}

constexpr BFloat16 std::numeric_limits<BFloat16>::epsilon() noexcept
{
    // 2^-7 * (1)     ; 0 01111000 0000000
    return BFloat16::from_raw(0x3C00u); // next_after_1.0 - 1.0
}

constexpr BFloat16 std::numeric_limits<BFloat16>::round_error() noexcept
{
    // 2^-1 * (1)      ; 0 01111110 0000000
    return BFloat16::from_raw(0x3F00u); // 0.5
}

constexpr BFloat16 std::numeric_limits<BFloat16>::infinity() noexcept
{
    return BFloat16::from_raw(uint16_t(BFloat16::k_inf_code));
}

constexpr BFloat16 std::numeric_limits<BFloat16>::quiet_NaN() noexcept
{
    return BFloat16::from_raw(uint16_t(BFloat16::k_inf_code | BFloat16::k_quiet_bit));
}

constexpr BFloat16 std::numeric_limits<BFloat16>::signaling_NaN() noexcept
{
    return BFloat16::from_raw(uint16_t(BFloat16::k_inf_code | (BFloat16::k_quiet_bit >> 1)));
}

constexpr BFloat16 std::numeric_limits<BFloat16>::denorm_min() noexcept
{
    return BFloat16::from_raw(0x0001u);
}

// this is to make hexnn_fp::fpclassify(BFloat16), etc, work...

namespace hexnn_fp_priv {

template <> struct cmath_funcs<BFloat16> {
    using FType = BFloat16;
    static constexpr unsigned k_no_sign_mask = BFloat16::k_no_sign_mask;
    static constexpr unsigned k_sign_mask = BFloat16::k_sign_mask;
    static constexpr unsigned k_inf_code = BFloat16::k_inf_code;
    static constexpr unsigned k_smallest_norm_code = BFloat16::k_smallest_norm_code;

    API_FUNC_EXPORT static FType frexp(FType x, int *ex_p) noexcept; // defined in bfloat16.cc
    API_FUNC_EXPORT static FType ldexp(FType x, int ex) noexcept; // defined in bfloat16.cc
    static inline constexpr FType fabs(FType x) noexcept { return BFloat16::from_raw(x.d & k_no_sign_mask); }
    static inline constexpr FType copysign(FType mag, FType sgn) noexcept
    {
        return BFloat16::from_raw((mag.d & k_no_sign_mask) | (sgn.d & k_sign_mask));
    }

    template <int DIR> //
    static inline FType fl_tr_ce(FType x) noexcept
    {
        static_assert(DIR >= -1 && DIR <= 1);
        // if code >= 0x4300 (128.0) then it represents an integer (or inf or Nan), so no change
        if ((x.d & k_no_sign_mask) >= 0x4300) {
            return x;
        }
        return fl_tr_ce_inner<DIR>(x);
    }

    static inline constexpr int fpclassify(FType x) noexcept
    {
        unsigned const t = (x.d & k_no_sign_mask);
        if (t < k_inf_code) {
            return (t >= k_smallest_norm_code) ? FP_NORMAL : (t == 0) ? FP_ZERO : FP_SUBNORMAL;
        }
        return (t == k_inf_code) ? FP_INFINITE : FP_NAN;
    }
    static inline constexpr bool isfinite(FType x) noexcept { return (x.d & k_no_sign_mask) < k_inf_code; }
    static inline constexpr bool isinf(FType x) noexcept { return (x.d & k_no_sign_mask) == k_inf_code; }
    static inline constexpr bool isnan(FType x) noexcept { return x.is_nan(); }
    static inline constexpr bool isnormal(FType x) noexcept
    {
        unsigned const t = (x.d & k_no_sign_mask);
        return t >= k_smallest_norm_code && t < k_inf_code;
    }
    static inline constexpr bool signbit(FType x) noexcept { return (x.d & k_sign_mask) != 0; }
    static inline constexpr bool iszero(FType x) noexcept { return (x.d & k_no_sign_mask) == 0; }

    static inline constexpr FType fmin(FType x, FType y) noexcept
    {
        return x.do_cmp<hexnn_fp_priv::cmp_LT_fmin>(y) ? x : y;
    }
    static inline constexpr FType fmax(FType x, FType y) noexcept
    {
        return x.do_cmp<hexnn_fp_priv::cmp_GT_fmax>(y) ? x : y;
    }

  protected:
    template <int DIR> //
    API_FUNC_EXPORT static FType fl_tr_ce_inner(FType x) noexcept; // defined in bfloat16.cc
};
} // namespace hexnn_fp_priv

extern template FUNC_ATTRIB_CONST BFloat16
hexnn_fp_priv::cmath_funcs<BFloat16>::fl_tr_ce_inner<-1>(BFloat16 x) noexcept;
extern template FUNC_ATTRIB_CONST BFloat16 hexnn_fp_priv::cmath_funcs<BFloat16>::fl_tr_ce_inner<0>(BFloat16 x) noexcept;
extern template FUNC_ATTRIB_CONST BFloat16 hexnn_fp_priv::cmath_funcs<BFloat16>::fl_tr_ce_inner<1>(BFloat16 x) noexcept;

#endif // BFLOAT16_H
