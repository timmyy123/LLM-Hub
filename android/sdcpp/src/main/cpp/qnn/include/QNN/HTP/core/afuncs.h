// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef AFUNCS_H
#define AFUNCS_H 1

#include "dtype.h"
#include "builtin_intrinsics.h"
#include "macros_attribute.h"

#include <algorithm>
#include <cmath>
#ifndef __hexagon__
#include <cstring> // for memcpy etc
#endif

struct tile_data {
    uint8_t **addr;
    uint32_t offset_t_col;
    uint32_t offset_t_row;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
};

// Define order: .addr, .offset_t_col, .offset_t_row, .width, .height, .depth
#define TILEDATA(adrtab, next_tab_col, next_tab_row, h, w, d)                                                          \
    {(uint8_t **)(adrtab),                                                                                             \
     static_cast<uint32_t>(next_tab_col),                                                                              \
     static_cast<uint32_t>(next_tab_row),                                                                              \
     static_cast<uint32_t>(w),                                                                                         \
     static_cast<uint32_t>(h),                                                                                         \
     static_cast<uint32_t>(d)}

struct tile_data_conv1x1 {
    uint8_t **i_addr;
    uint8_t **o_addr;
    uint32_t i_depth;
    uint32_t o_depth;
    uint32_t ncroutons;
};

// Define order: .i_addr, .o_addr, .height, .width, .i_depth, .o_depth
#define TILEDATA_CONV1x1(in_adrtab, out_adrtab, idepth, odepth, ncroutons)                                             \
    {(uint8_t **)(in_adrtab), (uint8_t **)(out_adrtab), static_cast<uint32_t>(idepth), static_cast<uint32_t>(odepth),  \
     static_cast<uint32_t>(ncroutons)}

/*=======================================*/
/* Auxiliary functions                   */
/*=======================================*/
#if defined(__hexagon__)
inline int32_t max_i32(int32_t a, int32_t b)
{
    return Q6_R_max_RR(a, b);
}
inline int32_t min_i32(int32_t a, int32_t b)
{
    return Q6_R_min_RR(a, b);
}
inline uint32_t max_u32(uint32_t a, uint32_t b)
{
    return Q6_R_maxu_RR(a, b);
}
inline uint32_t min_u32(uint32_t a, uint32_t b)
{
    return Q6_R_minu_RR(a, b);
}
#else
inline int32_t max_i32(int32_t a, int32_t b)
{
    return (a < b) ? b : a;
}
inline int32_t min_i32(int32_t a, int32_t b)
{
    return (a < b) ? a : b;
}
inline uint32_t max_u32(uint32_t a, uint32_t b)
{
    return (a < b) ? b : a;
}
inline uint32_t min_u32(uint32_t a, uint32_t b)
{
    return (a < b) ? a : b;
}
#endif

[[maybe_unused]] inline ALWAYSINLINE int64_t roundf_i64(float val)
{
    // add 0.5 (with same sign as val) and then conversion to int truncates toward 0.
    // values exactly halfway will round away from 0 (like roundf).

    return (int64_t)(val + copysignf(0.5f, val));
}

[[maybe_unused]] inline ALWAYSINLINE NN_INT32_T roundf_i32(float val)
{
    // add 0.5 (with same sign as val) and then conversion to int truncates toward 0.
    // values exactly halfway will round away from 0 (like roundf).

    return (int)(val + copysignf(0.5f, val));
}
// same thing for rounding to unsigned range; -ve inputs will give 0.
//
[[maybe_unused]] inline ALWAYSINLINE uint32_t roundf_u32(float val)
{
    // add 0.5f and then convert to uint (trunc towards 0; -ve values are clipped to 0).
#ifdef __hexagon__
    // use intrinsic since conv of -ve float to unsigned is 'undefined behaviour' in C.
    return Q6_R_convert_sf2uw_R_chop(val + 0.5f);
#else
    return (val < 0.5f) ? 0 : (uint32_t)(val + 0.5f);
#endif
}

[[maybe_unused]] inline ALWAYSINLINE NN_INT32_T roundd_i32(double val)
{
    // add 0.5 (with same sign as val) and then conversion to int truncates toward 0.
    // values exactly halfway will round away from 0 (like round).

    return (int)(val + copysign(0.5, val));
}

[[maybe_unused]] inline ALWAYSINLINE NN_INT32_T saturate_u8(NN_INT32_T val)
{
#ifdef __hexagon__
    return Q6_R_satub_R(val);
#else
    return (val < 0) ? 0 : ((val > 255) ? 255 : val);
#endif
}

[[maybe_unused]] inline ALWAYSINLINE NN_INT32_T saturate_u16(NN_INT32_T val)
{
#ifdef __hexagon__
    return Q6_R_satuh_R(val);
#else
    return (val < 0) ? 0 : ((val > 65535) ? 65535 : val);
#endif
}

[[maybe_unused]] static inline ALWAYSINLINE NN_INT32_T saturate_i16(NN_INT32_T val)
{
#ifdef __hexagon__
    return Q6_R_sath_R(val);
#else
    return (val < -32768) ? -32768 : ((val > 32767) ? 32767 : val);
#endif
}

/**
 * @brief low-cost frexpf (but only the exponent result);
 * Generates only a few instructions on hexagon.
 *
 * Input must not be inf,nan, zero, or denormal.
 *
 * returns:
 *        -1 if abs(x) is in range 0.25 ... 0.249999
 *         0 if abs(x) is in range 0.5 ... 0.99999
 *         1 if abs(x) is in range 1.0 .. 1.9999
 *  etc
 *
 *  If the value -126 is returned, x is a zero or denormal;
 *  129 is returned for inf or NaN. for other cases the value is the same
 *  as what frexpf  (in math.h) generates for the exponent.
 */
[[maybe_unused]] inline ALWAYSINLINE constexpr int flt_getexp(float x)
{
    union {
        float f;
        uint32_t u32;
    } const uu = {x};
    return static_cast<int>(((uu.u32 >> 23u) & 0xFFu) - 126u);
}

// Specialized flt_getexp for the case where you want the value
// converted to exponent, and mantissa, with the mantissa normalized
// to a specific number of bits.

// e.g. for 15-bit case:
//     int e = flt_getexp_for_frac<15>(scale);
//     float m_f = flt_ldexp(scale, 15-e);
//     int m_i = roundf_i32(m_f);
//
// If you use 'flt_getexp(scale)', the value of m_f will be >= 16384.0, < 32768.0
// *but* it could fall in range >= 32767.5, < 32768.0; in these cases m_i will be 32768.
// By using flt_getexp_for_frac<16> instead, you will get an an exponent which is larger
// by 1 for those specific cases, and then m_f will be in range > 16383.75, < 16384,
// so for those cases m_i will be rounded up to 16384; this is a more accurate representation
// than you get by saturating the result to 32767.
//
// This can be used for x values that may be negative; but in that case
// W does not count the sign. for W=15, with the
// above code you should have a value 16384 <= abs(m_i) <= 32767
// If you want to normalize over the full signed range (i.e. -32768 <= m_i <= -16385,
// when scale < 0), use flt_getexp_for_signed_frac<16>(scale).
//
// ** In general **:
//  Given an normal float x,  > 0, return an exponent e such that
//      x * 2^(W-e)
//  .. rounded to nearest, is 'normalized' in W unsigned bits:
//              >= (1<<W)/2, and < (1<<W)
// x may also be a normal < 0, the result is the same as for abs(x).
//
// This is usually the same as flt_getexp(x) but will be one large for values
// marginally below a power of 2 (the margin gets larger for smaller w)
//
//
// For W >= 24, result is always the same as flt_getexp(x)
//
template <unsigned W> //
inline int flt_getexp_for_frac(float x)
{
    static_assert(W >= 3 && W < 32);
    // We want to return exponent larger by 1
    // if, and only if, the upper W bits of the mantissa are all 1 (not including the hidden
    // bit) - so, add a 1 to bit 23-W of the 'uint32' image of the value; it will carry
    // into the exponent field if and only if all those bits are 1.
    union {
        float f;
        uint32_t u32;
    } uu = {x};
    uu.u32 += (1u << 23u) >> W;
    return flt_getexp(uu.f);
}
// This is like flt_getexp_for_frac<W>, but for cases where you want
// a fully normalized 'signed; mantissa; e.g.
//
//     int e = flt_getexp_for_signed_frac<16>(scale);
//     float m_f = flt_ldexp(scale, 15-e);
//     int m_i = roundf_i32(m_f);
//     m_i = saturate_i16(m_i); // see note below; could be m_i = std::max(m_i, -32768)
//
// m_i will always be -32768.. -16385   (for scale < 0)
//      or 16384..32767    (for scale > 0)
//
// For x > 0, the result is always the same as flt_getexp_frac<W-1>(x).
// for x < 0, it is usually the same as flt_getexp(x), but sometimes
//  one smaller: this happens when -x is exactly a power of two, or marginally larger.
//  Those are cases where want the 'most negative' signed value -32768.
// NOTE: the saturate_i16 is needed since the rounded m_i result could be -32769;
// in such cases -32768 is sill the best available representation (better than
// -16385 with a larger +1 exponent)
//
template <unsigned W> // W includes sign bit
inline int flt_getexp_for_signed_frac(float x)
{
    static_assert(W >= 4 && W <= 25);
    // for x > 0, same effect as flt_getexp_for_frac<W-1>; add 1 in bit (24-(W-1))
    // for x < 0  we subtract (1<<(24-(W-1))) + 1, so it will carry to exponent
    //    field and reduce by 1 in applicable cases.
    //     Equivalent is to add ~(1<<(24-(W-1))) modularly.
    // (this 'modular add' is defence against 'sanitize' detecting overflow)
    auto modular_add_u32 = [](uint32_t a, uint32_t b) -> uint32_t {
        uint64_t const sum = uint64_t(a) + uint64_t(b);
        return uint32_t(sum);
    };
    union {
        float f;
        uint32_t u32;
    } uu = {x};
    uint32_t constexpr fudge_bit = (1u << 23u) >> (W - 1);
    uu.u32 = modular_add_u32(uu.u32, (uu.u32 & (1u << 31u)) ? ~fudge_bit : fudge_bit);
    return flt_getexp(uu.f);
}

/**
 * @brief low-cost frexpf (but only the 'fraction' result);
 * Generates only a few instructions on hexagon.
 *
 * Input must not be inf,nan, zero, or denormal.
 *
 * returns a value in the range [0.5, 1.0)  (or in (-1.0,-0.5] when x < 0)
 * such that x = flt_getmant(x) * powf2(2.0, flt_getexp(x))
 *
 */
[[maybe_unused]] inline ALWAYSINLINE constexpr float flt_getmant(float x)
{
    union {
        float f;
        uint32_t u32;
    } uu = {x};
    uu.u32 = (uu.u32 & 0x807fffffu) | (uint32_t(126) << 23u); // force exponent = 126
    return uu.f;
}

/**
 * @brief returns the mantissa of x, as a 24-bit number
 * in the range 0x800000 .. 0xFFFFFF
 *
 * Input must not be inf,nan, zero, or denormal.
 *
 * Sign is discarded. same as powf(2,24) * flt_getmant(fabsf(x)).
 */
[[maybe_unused]] inline ALWAYSINLINE constexpr int32_t flt_getfrac(float x)
{
    union {
        float f;
        uint32_t u32;
    } const uu = {x};
    int32_t const m = (uu.u32 & 0x007fffffu) | (uint32_t(1) << 23u);
    return m;
}

//
// This 'normalizes' a float to 0.5 .. 0.9999  (sign is retained)
// Same result as the return value from frexpf, without using a function call
// Results are not valid if x is 0, denormal, or inf/nan
//
[[maybe_unused]] inline ALWAYSINLINE float flt_getfrac_norm(float x)
{
    union {
        float f;
        uint32_t u32;
    } uu = {x};
    uu.u32 = (uu.u32 & 0x807fffffu) | (uint32_t(126) << 23u); // force exponent = 126
    return uu.f;
}
/**
 * @brief low-cost 2.0*n for integer n.
 * Same as powf(2.0f, iexpo) without a function call;
 *
 * Constraint: iexpo must be in range -126..127
 */
[[maybe_unused]] inline ALWAYSINLINE constexpr float flt_power2(uint32_t const iexpo)
{
    uint32_t const a = (iexpo + 127) & 0xFFu;
    union {
        uint32_t u32;
        float f;
    } const uu = {a << 23u};
    return uu.f;
}
/**
 * @brief low-cost ldexpf
 * Same as ldexpf(val, iexpo) without a function call;
 *
 * Constraint: iexpo must be in range -126..127
 */
[[maybe_unused]] inline ALWAYSINLINE constexpr float flt_ldexp(float val, int iexpo)
{
    return val * flt_power2(static_cast<uint32_t>(iexpo));
}
/**
 * @brief low-cost 2.0*n for integer n.
 * Same as pow(2.0d, iexpo) without a function call;
 *
 * Constraint: iexpo must be in range -1022..1023
 */
[[maybe_unused]] inline ALWAYSINLINE constexpr double double_power2(uint32_t const iexpo)
{
    uint64_t const a = (iexpo + 1023) & 0x7FFu;
    union {
        uint64_t u64;
        double d;
    } const uu = {a << 52u};
    return uu.d;
}
/**
 * @brief low-cost ldexpf
 * Same as ldexp(val, iexpo) without a function call;
 *
 * Constraint: iexpo must be in range -1022..1023
 */
[[maybe_unused]] inline ALWAYSINLINE constexpr double double_ldexp(double val, int iexpo)
{
    return val * double_power2(static_cast<uint32_t>(iexpo));
}

/**
 * @brief returns the exponent and mantissa of x, as a n-bit number
 *
 * Constraint: iexpo must be in range -126..127
 * Input must not be negative, inf,nan, zero, or denormal.
 */
template <uint32_t MBITS> inline constexpr std::pair<int32_t, uint32_t> get_scalefactor(float x)
{
    union {
        float f;
        uint32_t u32;
    } const uu = {x};

    uint32_t inval = uu.u32;
    uint32_t const mask = hnnx::safe_lshift(1, MBITS) - 1;
    inval = hnnx::safe_rshift(inval + hnnx::safe_lshift(1, (24 - MBITS - 1)),
                              (24 - MBITS)); // possibly overflows into exponent, but that's OK.
    uint32_t const m = ((inval & mask) | hnnx::safe_lshift(1u, (MBITS - 1)));
    int32_t const e = int32_t(hnnx::safe_rshift(inval, (MBITS - 1)) & 0xFFu) - 126;
    return {e, m};
}

/**
 * @brief returns the parameters for scaling.
 * bit 31-24: left shift amount
 * bit 23-16: right shift amout
 * bit 15- 0: scale factor
 *
 * Input must not be inf,nan, zero, negative or denormal.
 *
 */
[[maybe_unused]] inline ALWAYSINLINE constexpr uint32_t get_scaling_params(float x, int max_sl, int max_sr)
{
    auto [e, m] = get_scalefactor<15>(x);
    // Set a sl or sr amount to perform a multiply of 2^exponent by mantissa.
    int sl = (e > 0) ? e : 0;
    int sr = (e > 0) ? 0 : -e;
    // The max_sl allows the addition of extra left shifts when working with small numbers having negative exponents.
    // For every extra left shift, there is an offsetting right shift added so that the net right shift amount
    // required from the exponent stays the same. The max_sr parameter provides a ceiling to the required offsetting
    // right shifts, preventing the total right shift requirement from being large enough to erase data through shifting.
    if (sl == 0 && sr > 0) {
        sl = min_i32(max_sl, max_i32(max_sr - sr, 0));
        sr = sr + sl;
    }
    return ((uint32_t(sl) & 0x0FFu) << 24u) | ((uint32_t(sr) & 0x0FFu) << 16u) | uint32_t(m);
}

/**
 * @brief given a scale in float and a recip shift amount
 *  return a quantized scale multiplier and change recip shamt inplace
 *
 */
inline uint32_t get_quantized_multipiler(const float scale_f, int &recip_shamt)
{
    recip_shamt = (scale_f <= 1.0f) ? 0 : flt_getexp(scale_f);
    uint32_t scale = static_cast<uint32_t>(roundf(flt_ldexp(scale_f, (31 - recip_shamt))));
    scale = (scale < 0x7fffffffu) ? scale : 0x7FFFFFFFu;
    return scale;
}

/**
 * @brief given a scale in float and a recip shift amount
 *  return a quantized scale multiplier and change recip shamt inplace
 *
 */
//Now with corrected spelling
inline uint32_t get_quantized_multiplier(const float scale_f, int &recip_shamt)
{
    return get_quantized_multipiler(scale_f, recip_shamt);
}
#endif /*AFUNCS_H*/
