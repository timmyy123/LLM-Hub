// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef FLOAT16_H
#define FLOAT16_H

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

struct Float16;

namespace hexnn_fp {
using float16_t = Float16;
} // namespace hexnn_fp

struct API_EXPORT Float16 {
    friend struct hexnn_fp_priv::cmath_funcs<Float16>;
    friend struct hexnn_fp_priv::raw_methods<Float16>;
    friend class std::numeric_limits<Float16>;

    constexpr Float16() noexcept : d(0) {}
    constexpr Float16(float f) noexcept;
    // 'noexcept' on defaulted operations is enforced with static_assert af6er class. Do not add here.
    Float16(const Float16 &f) = default;
    Float16 &operator=(const Float16 &f) = default;
    Float16(Float16 &&f) = default;
    Float16 &operator=(Float16 &&f) = default;
    ~Float16() = default;

    /// All these to be made 'private' >>>
    constexpr bool is_zero() const noexcept;
    constexpr bool is_inf() const noexcept;
    constexpr bool is_nan() const noexcept;
    constexpr bool is_finite() const noexcept;

    constexpr uint16_t raw() const noexcept { return d; }

    static constexpr Float16 qnan() noexcept;
    static constexpr Float16 snan() noexcept;
    static constexpr Float16 inf() noexcept;

    static constexpr Float16 from_raw(uint16_t v) noexcept;

    // same as ->float, but treats max. exp as a normal number
    // instead of inf/nan
    constexpr float to_float_alt() const noexcept;
    // same as from-float, but allows +/- 131008 range, using
    // exp=31 as normal.
    static constexpr Float16 from_float_alt(float v) noexcept;

    // <<<<

    constexpr operator float() const noexcept;

    // 'catch-all' for conversion from integer types.
    // This is needed if we have ctor from float, and from double; to eliminate
    // 'ambiguous' conversions to Float16. Some large ints will get rounded when
    // converted to float, but all of those will be 'inf' when converted to Float16.
    template <typename I, std::enable_if_t<std::is_integral_v<I>, bool> = false> //
    constexpr Float16(I val) noexcept : Float16(static_cast<float>(val))
    {
    }
    constexpr Float16(double v) noexcept : Float16(from_double(v)) {}

  private:
    // Float16 from double (not always the same as what you get from
    // rounding to float first)
    static constexpr Float16 from_double(double x) noexcept;

    static constexpr unsigned k_no_sign_mask = 0x7fffu;
    static constexpr unsigned k_sign_mask = 0x8000u;
    static constexpr unsigned k_inf_code = 0x7c00u;
    static constexpr unsigned k_mantissa_mask = 0x03ff;
    static constexpr unsigned k_quiet_bit = 0x0200u;
    static constexpr unsigned k_smallest_norm_code = k_mantissa_mask + 1u;

    template <bool ALT = false> //
    static constexpr uint16_t float16_from_float(float x) noexcept;
    template <bool ALT = false> //
    static constexpr float float_from_float16(unsigned d) noexcept;

    template <enum hexnn_fp_priv::fp_comp CMP_OP> //
    constexpr bool do_cmp(Float16 other) noexcept;

    uint16_t d;

    friend Float16 operator-(Float16 a) noexcept;
    friend FUNC_ATTRIB_CONST API_FUNC_EXPORT Float16 operator+(Float16 a, Float16 b) noexcept;
    friend Float16 operator-(Float16 a, Float16 b) noexcept;
    friend FUNC_ATTRIB_CONST API_FUNC_EXPORT Float16 operator*(Float16 a, Float16 b) noexcept;
    friend FUNC_ATTRIB_CONST API_FUNC_EXPORT Float16 operator/(Float16 a, Float16 b) noexcept;
    friend constexpr bool operator==(Float16 a, Float16 b) noexcept;
    friend constexpr bool operator!=(Float16 a, Float16 b) noexcept;
    friend constexpr bool operator<(Float16 a, Float16 b) noexcept;
    friend constexpr bool operator<=(Float16 a, Float16 b) noexcept;
    friend constexpr bool operator>(Float16 a, Float16 b) noexcept;
    friend constexpr bool operator>=(Float16 a, Float16 b) noexcept;
};

POP_VISIBILITY()

//
// ** If you make a change which causes any of the static_assert below
// ** to be violated, please reconsider the change before removing the assert.
// ** Such a change will affect the size and performance of code using Float16
// ** (E.g. it may not be possible to pass a Float16 by value, without extra
// ** overhead). Do NOT add 'noexcept' to defaulted methods to make the asserts
// ** go away.
static_assert(std::is_nothrow_copy_constructible<Float16>::value);
static_assert(std::is_nothrow_move_constructible<Float16>::value);
static_assert(std::is_nothrow_assignable<Float16, Float16>::value);
static_assert(std::is_nothrow_move_assignable<Float16>::value);
static_assert(std::is_nothrow_destructible<Float16>::value);
static_assert(std::is_trivially_copyable<Float16>::value);
static_assert(std::is_standard_layout<Float16>::value);

// specialize hexnn_fp_priv::raw_methods<Float16>
// makes hexnn_fp::from_raw<Float16>(rawval),
// and hexnn_fp::to_raw(Float16 f) work.
namespace hexnn_fp_priv {
template <> struct raw_methods<Float16> {
    using raw_t = uint16_t;
    static inline constexpr Float16 from_raw(raw_t val) noexcept { return Float16::from_raw(val); }
    static inline constexpr raw_t to_raw(Float16 val) noexcept { return val.raw(); }

    static inline constexpr Float16 from_float_alt(float val) noexcept { return Float16::from_float_alt(val); }
    static inline constexpr float to_float_alt(Float16 val) noexcept { return val.to_float_alt(); }
};
} // namespace hexnn_fp_priv

namespace hexnn_fp {
// 'convenience' wrapper ... alternative to hexnn_fp::from_raw<Float16>(rawval)
inline constexpr Float16 float16_from_raw(uint16_t rawval) noexcept
{
    return hexnn_fp_priv::raw_methods<Float16>::from_raw(rawval);
}

// 'convenience' wrapper ... alternative to hexnn_fp::from_float_alt<Float16>(flt)
inline constexpr Float16 float16_from_float_alt(float f) noexcept
{
    return hexnn_fp_priv::raw_methods<Float16>::from_float_alt(f);
}
} // namespace hexnn_fp

////////////
// The following are needed to resolve ambiguities when comparing other types to Float16,
// e.g (fp16_val == 1) -> float(fp16_val) == 1  -> float(fp16_val) == float(1)
#define FP16_COMPARE_ADAPTER(CMP_OP)                                                                                   \
    template <typename T>                                                                                              \
    inline constexpr std::enable_if_t<std::is_arithmetic_v<T>, bool> operator CMP_OP(T a, Float16 b) noexcept          \
    {                                                                                                                  \
        return a CMP_OP float(b);                                                                                      \
    }                                                                                                                  \
    template <typename T>                                                                                              \
    inline constexpr std::enable_if_t<std::is_arithmetic_v<T>, bool> operator CMP_OP(Float16 a, T b) noexcept          \
    {                                                                                                                  \
        return float(a) CMP_OP b;                                                                                      \
    }

FP16_COMPARE_ADAPTER(==)
FP16_COMPARE_ADAPTER(!=)
FP16_COMPARE_ADAPTER(<)
FP16_COMPARE_ADAPTER(>)
FP16_COMPARE_ADAPTER(<=)
FP16_COMPARE_ADAPTER(>=)
#undef FP16_COMPARE_ADAPTER

//
// This is a float32->float16 conversion.
// When ALT=true, we get 'alternate' conversion with no inf/nan, but
// extra range up to 131008.0 (twice the normal limit of 65504.0).
// Inf, nan, and all larger values are mapped to that value (with sign).
//
// Since this is used in "constexpr Float16 operator""_f16(long double v)",
// it needs to be constexpr, it can be evaluated at compile time.
// This is intended to be correct, concise, and self-contained, but
// not necessarily so clear in the logic. Comments are your friend...
//
// It written in a way which allows static analysis tools to
// establish that the shift amount in << and >> are never out of range.
//
// This cannot actually be evaluated constexpr, before C++20, due to how
// hexnn_fp::to_raw works. though the compiler can still 'fold' it to a constant result
// for constant input. With C++20, std::bit_cast could be used inside to_raw,
// which will allow constexpr evaluation.
//
template <bool ALT> //
inline constexpr uint16_t Float16::float16_from_float(float x) noexcept
{
    // if a 'normal' float16 value is converted to Float32, exponent is
    // larger by this much.
    constexpr unsigned exp_offset = 112;
    // Encoding of 2^16 in Float32. If converted to float16 via 'normal'
    // method, this will map to infinity. Exponent field is 143.
    constexpr unsigned code32_64k = (143u << 23);
    // float32 encoding of 2^-14 (which is smallest Float16 'normal') in float32  (exp = 113)
    constexpr unsigned code32_2pm14 = (113u << 23);
    constexpr unsigned code32_inf = 0x7F800000u; // infinity
    // The smallest float32 which will round up to smallest float32 code.
    // Exponent is 102.
    constexpr unsigned code32_min_for_float16_subnorm = 0x33000001u;

    // this is float32 encoding of 131008.0; exponent is 143.
    constexpr unsigned code32_alt_limit = 0x47FFE000u;

    constexpr unsigned code16_qnan = k_inf_code | k_quiet_bit; // 'quiet_NaN'

    unsigned const x_as_u32 = hexnn_fp::to_raw<float>(x);
    unsigned x_abs = x_as_u32 & 0x7fffffffu;
    unsigned result = (x_as_u32 >> 16) & 0x8000u; // sign bit only

    if (x_abs >= code32_2pm14) { // too big for subnormal Float16
        if constexpr (!ALT) {
            if (x_abs > code32_64k) {
                // all values >= code32_64k map to code32_64k; first we ned to check for NaN
                if (x_abs > code32_inf) { // is a NaN
                    // Same sign; exp=0x1f, bit 9=1, and bits 8..0 from bits 21..13 of input
                    return static_cast<std::uint16_t>(result | code16_qnan | ((x_abs >> 13) & 0x1ff));
                }
                x_abs = code32_64k;
            }
        } else {
            if (x_abs > code32_alt_limit) {
                if (x_abs > code32_inf) { // is a NaN
                    result = 0; // return +MAX vs -MAX for Nan
                }
                // all other map to +/- 131008
                x_abs = code32_alt_limit;
            }
        }
        // value is in range >=2^14, and not too large to convert to 'normal' range.
        // For ALT = false, this includes values <= code32_64k which will convert to the Float16 'inf'
        // code, either directly or via rounding up.
        // For ALT = true, includes all <= code32_alt_limit, none will exceed 0x7fff result.
        // start by rounding away 13 LSBS of x_abs, using convergent rounding.
        unsigned const rbias = 0x1000 - ((~x_abs >> 13) & 1u);
        unsigned rounded = (x_abs + rbias) >> 13;
        // now, exponent is in bits 17:10, and is too big by exp_offset (range 113 .. 143).
        rounded -= (exp_offset << 10); // all done except for sign.
        result |= rounded;
    } else if (x_abs >= code32_min_for_float16_subnorm) {
        // subnormal range; float32 exponent is 102 .. 112; we need to take 23 LSBs
        // of x_abs, add 'hidden bit', and round away lower 126-exp.
        // (for the '102' case, we are rounding away all 24 bits, result will be 1 unless
        // all 23 bits are 0, which happens only when x_abs == code32_min_for_float16_subnorm-1)

        unsigned const bits_to_round = 126 - (x_abs >> 23);
        // Hopefully static analysis can prove from the the above that 14 <= bits_to_round <= 24
        // if you still get 'shift may be too large,' add: 'bits_to_round &= 31u;' (which compiler
        // will likely optimize out), and maybe file an issue with SA vendor.
        x_abs = (x_abs & 0x7FFFFFu) | 0x800000u;
        unsigned rbias = 1u << bits_to_round;
        // rbias is (1/2) of (1<<bits_to_round); but 1 less when the lowest 'retained'
        // bit in x_abs is 0.
        rbias = (rbias >> 1) - ((~x_abs >> bits_to_round) & 1u);
        x_abs = (x_abs + rbias) >> bits_to_round;
        // that's all - result is 1 .. 0x400; 0x400 is 2^-14; a value rounded up out
        // of subnormal range.
        result |= x_abs;
    }
    // .. else result is just +/- 0.0
    return static_cast<std::uint16_t>(result);
}

inline constexpr Float16::Float16(float f) noexcept : d(Float16::float16_from_float<false>(f)) {}

inline constexpr bool Float16::is_zero() const noexcept
{
    return (d & k_no_sign_mask) == 0x0000;
}

inline constexpr bool Float16::is_inf() const noexcept
{
    return (d & k_no_sign_mask) == k_inf_code;
}

inline constexpr bool Float16::is_nan() const noexcept
{
    return (d & k_no_sign_mask) > k_inf_code;
}

inline constexpr bool Float16::is_finite() const noexcept
{
    // is_norm() || is_subnorm
    return (d & k_no_sign_mask) < k_inf_code;
}

inline constexpr Float16 Float16::qnan() noexcept
{
    return Float16::from_raw(k_inf_code | k_quiet_bit);
}

inline constexpr Float16 Float16::snan() noexcept
{
    return Float16::from_raw(k_inf_code | (k_quiet_bit >> 1));
}

inline constexpr Float16 Float16::inf() noexcept
{
    return Float16::from_raw(k_inf_code);
}

inline constexpr Float16 Float16::from_raw(uint16_t v) noexcept
{
    Float16 f;
    f.d = v;
    return f;
}

inline Float16 operator-(Float16 a) noexcept
{
    return Float16::from_raw(a.raw() ^ Float16::k_sign_mask);
}

inline Float16 operator-(Float16 a, Float16 b) noexcept
{
    return operator+(a, -b);
}

// Compares are done by this function. When used as in f16_val == Float16{}, this should
// optimize to just: (f16_val.d & 0x7fff) == 0.
template <enum hexnn_fp_priv::fp_comp CMP_OP> //
inline constexpr bool Float16::do_cmp(Float16 other) noexcept
{
    using namespace hexnn_fp_priv;
    static_assert(hexnn_fp_priv::cmp_is_valid<CMP_OP>);

    unsigned a_code = d;
    unsigned b_code = other.d;
    unsigned const a_abs = a_code & Float16::k_no_sign_mask;
    unsigned const b_abs = b_code & Float16::k_no_sign_mask;
    unsigned const m = std::max(a_abs, b_abs);
    if (m > Float16::k_inf_code) {
        if constexpr (cmp_for_fminmax<CMP_OP>) {
            // for fmin/fmax: select the one which is *not* NaN where possible
            // the context is:
            //    fmin(a, b) -> (a LT_fmin b) ? a : b
            //    fmax(a, b) -> (a GT_fmax b) ? a : b
            // So, select a if B is NaN, else b.
            return b_abs > Float16::k_inf_code;
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
        // If either a or b has sign bit, invert all 16 bits in both.
        if (((a_code | b_code) & Float16::k_sign_mask) != 0) {
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
// some compilers require Float16::do_cmp to be defined before it's used...

constexpr bool operator==(Float16 a, Float16 b) noexcept
{
    return a.do_cmp<hexnn_fp_priv::cmp_EQ>(b);
}
constexpr bool operator!=(Float16 a, Float16 b) noexcept
{
    return !a.do_cmp<hexnn_fp_priv::cmp_EQ>(b);
}
constexpr bool operator<(Float16 a, Float16 b) noexcept
{
    return a.do_cmp<hexnn_fp_priv::cmp_LT>(b);
}
constexpr bool operator<=(Float16 a, Float16 b) noexcept
{
    return a.do_cmp<hexnn_fp_priv::cmp_LE>(b);
}
constexpr bool operator>(Float16 a, Float16 b) noexcept
{
    return b.do_cmp<hexnn_fp_priv::cmp_LT>(a);
}
constexpr bool operator>=(Float16 a, Float16 b) noexcept
{
    return b.do_cmp<hexnn_fp_priv::cmp_LE>(a);
}

// convert Float16 to float.
template <bool ALT> // 'ALT': treat exp=31 as 'normal' range
inline constexpr float Float16::float_from_float16(unsigned d) noexcept
{
    // Procedure is to transfer the sign, exponent and mantissa bits to
    // the float32 format. The 10 mantissa bits are placed in upper 10
    // bits (22:13) of the result mantissa; exponent bits in the lower
    // 5 of the 8 exponent bits (27::23). Sign is transferred, but other
    // exp and mantissa bits are zero -- *unless* the input is inf/nan
    // (exponent is 11111) -- in which case we will make all 8 exponent bits
    // of the float32 1's. This preserves inf and NaN; for other values,
    // (including 'subnormal' Float16), we need to multiply by 2^112 to
    // get the proper result (and, that operation will preserve the infs
    // and NaNs).
    // For 'ALT' mode, we don't check for inf/nan.

    unsigned res = d & 0x7fffu; // remove sign;
    if (!ALT && res >= 0x7c00) { // is inf/nan range...
        res |= 0x38000u; // 'fill out' 8 exponent bits (currently in 17:10)
    }
    // restore sign, move bits to 31:13 of res
    res = (res | ((d & 0x8000u) << 3u)) << 13u;
    // result just needs multiplication by 2^112.
    return hexnn_fp::from_raw<float>(res) * 0x1.0p+112f;
}

inline constexpr Float16::operator float() const noexcept
{
    return float_from_float16(d);
}
// same as operator float, except that exp=31 is treated as
// extended range
inline constexpr float Float16::to_float_alt() const noexcept
{
    return float_from_float16<true>(d);
}

// same as from-float, but allows +/- 131008 range, using
// exp=31 as normal.
constexpr Float16 Float16::from_float_alt(float v) noexcept
{
    return Float16::from_raw(Float16::float16_from_float<true>(v));
}

// Float16 from double .. not exactly the same as
// double->float->Float16; some cases round improperly if you
// do that.
inline constexpr Float16 Float16::from_double(double x) noexcept
{
    // Start by rounding the 'double' to a value with 0's in 31 lsbs,
    // using 'jam' rounding; bit 31 will be unchanged if those bits
    // were already 0, and will be 1 if they were not.
    // This only ever changes the 32 LSbs.
    // The change preserves inf, nan, zero (though, NaN can be changed
    // to different NaN).
    // The 'jam rounded' value will be converted to float without further
    // numerical rounding (assuming it falls within 'normal' float range;
    // values outside that range won't make it to Float16 anyhow).
    uint64_t x_as_u64 = hexnn_fp::to_raw<double>(x);
    if ((x_as_u64 & 0xFFFFFFFFu) != 0) {
        x_as_u64 = (x_as_u64 & ~uint64_t(0xFFFFFFFFu)) | 0x80000000u;
    }
    // now, convert that to float and then to Float16.
    return Float16(float(hexnn_fp::from_raw<double>(x_as_u64)));
}

constexpr Float16 operator""_f16(long double v) noexcept
{
    return Float16(static_cast<double>(v));
}

PUSH_VISIBILITY(default)

template <> class API_EXPORT std::numeric_limits<Float16> {
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
    static constexpr bool is_iec559 = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = 11;
    static constexpr int digits10 = 3; // floor((digits-1) * log10(2))
    static constexpr int max_digits10 = 5; // ceil(digits * log10(2) + 1)
    static constexpr int radix = 2;
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4; // min normal =~ 0.000061035
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 5; // largest finite val = 65504
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false; // libc++

    static constexpr Float16 min() noexcept; // returns min positive normal
    static constexpr Float16 lowest() noexcept; // returns true min
    static constexpr Float16 max() noexcept; // max positive
    static constexpr Float16 epsilon() noexcept; // step at 1.0
    static constexpr Float16 round_error() noexcept; // 0.5
    static constexpr Float16 infinity() noexcept;
    static constexpr Float16 quiet_NaN() noexcept;
    static constexpr Float16 signaling_NaN() noexcept;
    static constexpr Float16 denorm_min() noexcept; // min positive denorm
};

POP_VISIBILITY()

constexpr Float16 std::numeric_limits<Float16>::min() noexcept
{
    // 2^-14 * (1 + 0/1024)     ; 0 00001 0000000000
    return Float16::from_raw(Float16::k_smallest_norm_code);
}

constexpr Float16 std::numeric_limits<Float16>::lowest() noexcept
{
    // -2^15 * (1 + 1023/1024)  ; 1 11110 1111111111
    return Float16::from_raw(Float16::k_sign_mask | (Float16::k_inf_code - 1)); // -65504
}

constexpr Float16 std::numeric_limits<Float16>::max() noexcept
{
    // 2^15 * (1 + 1023/1024)   ; 0 11110 1111111111
    return Float16::from_raw(Float16::k_inf_code - 1); // 65504
}

constexpr Float16 std::numeric_limits<Float16>::epsilon() noexcept
{
    // 2^-10 * (1 + 0/1024)     ; 0 00101 0000000000
    return Float16::from_raw(0x1400); // next_after_1.0 - 1.0
}

constexpr Float16 std::numeric_limits<Float16>::round_error() noexcept
{
    // 2^-1 * (1 + 0/1024)      ; 0 01110 0000000000
    return Float16::from_raw(0x3800); // 0.5
}

constexpr Float16 std::numeric_limits<Float16>::infinity() noexcept
{
    return Float16::inf();
}

constexpr Float16 std::numeric_limits<Float16>::quiet_NaN() noexcept
{
    return Float16::qnan();
}

constexpr Float16 std::numeric_limits<Float16>::signaling_NaN() noexcept
{
    return Float16::snan();
}

constexpr Float16 std::numeric_limits<Float16>::denorm_min() noexcept
{
    return Float16::from_raw(0x0001);
}

// this is to make hexnn_fp::fpclassify(Float16), etc, work...

namespace hexnn_fp_priv {

template <> struct cmath_funcs<Float16> {
    using FType = Float16;

    API_FUNC_EXPORT static FType frexp(FType x, int *ex_p) noexcept; // defined in float16.cc
    API_FUNC_EXPORT static FType ldexp(FType x, int ex) noexcept; // defined in float16.cc
    static inline constexpr FType fabs(FType x) noexcept { return Float16::from_raw(x.d & 0x7fffu); }
    static inline constexpr FType copysign(FType mag, FType sgn) noexcept
    {
        return Float16::from_raw((mag.d & 0x7fffu) | (sgn.d & 0x8000u));
    }
    template <int DIR> //
    static inline FType fl_tr_ce(FType x) noexcept
    {
        static_assert(DIR >= -1 && DIR <= 1);
        // if code >= 0x6400 (1024.0) then it represents an integer (or inf or Nan), so no change
        if ((x.d & 0x7fffu) >= 0x6400) {
            return x;
        }
        return fl_tr_ce_inner<DIR>(x);
    }

    static inline constexpr int fpclassify(FType x) noexcept
    {
        unsigned const t = (x.d & FType::k_no_sign_mask);
        if (t < FType::k_inf_code) {
            return (t >= FType::k_smallest_norm_code) ? FP_NORMAL : (t == 0) ? FP_ZERO : FP_SUBNORMAL;
        }
        return (t == FType::k_inf_code) ? FP_INFINITE : FP_NAN;
    }
    static inline constexpr bool isfinite(FType x) noexcept { return x.is_finite(); }
    static inline constexpr bool isinf(FType x) noexcept { return x.is_inf(); }
    static inline constexpr bool isnan(FType x) noexcept { return x.is_nan(); }
    static inline constexpr bool isnormal(FType x) noexcept
    {
        unsigned const t = (x.d & FType::k_no_sign_mask);
        return t >= FType::k_smallest_norm_code && t < FType::k_inf_code;
    }
    static inline constexpr bool signbit(FType x) noexcept { return (x.d & FType::k_sign_mask) != 0; }
    static inline constexpr bool iszero(FType x) noexcept { return x.is_zero(); }

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
    API_FUNC_EXPORT static FType fl_tr_ce_inner(FType x) noexcept; // defined in float16.cc
};
} // namespace hexnn_fp_priv

extern template FUNC_ATTRIB_CONST Float16 hexnn_fp_priv::cmath_funcs<Float16>::fl_tr_ce_inner<-1>(Float16 x) noexcept;
extern template FUNC_ATTRIB_CONST Float16 hexnn_fp_priv::cmath_funcs<Float16>::fl_tr_ce_inner<0>(Float16 x) noexcept;
extern template FUNC_ATTRIB_CONST Float16 hexnn_fp_priv::cmath_funcs<Float16>::fl_tr_ce_inner<1>(Float16 x) noexcept;

#endif // FLOAT16_H
