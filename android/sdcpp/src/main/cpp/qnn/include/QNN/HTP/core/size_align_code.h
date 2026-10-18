// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef SIZE_ALIGN_CODE_H
#define SIZE_ALIGN_CODE_H

#include "weak_linkage.h"

#include <cstddef>
#include <cstdint>
#include <cassert>

PUSH_VISIBILITY(default)

namespace hnnx {

// LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
template <size_t N> unsigned constexpr log2_floor_of()
{
    if constexpr (N < 4) {
        return N >= 2 ? 1 : 0; // note, 0->0
    }
    size_t x = N;
    unsigned res = 2;
    while (x >= 8) {
        x >>= 1;
        res++;
    }
    return res;
}

// LCOV_EXCL_STOP

// size_align_code_t combines the size and alignment of an op in a size_t word.

class size_align_code_t {
    // this has 'K' in lower 4 bits, and 'W' in upper bits;
    // the alignment is 1 << K, and the size is W << K.
    size_t code;

  public:
    constexpr size_align_code_t() : code(0) {}
    constexpr size_align_code_t(size_align_code_t const &) = default;
    constexpr size_align_code_t(size_align_code_t &&) = default;
    constexpr size_align_code_t &operator=(size_align_code_t const &) = default;
    constexpr size_align_code_t &operator=(size_align_code_t &&) = default;
    ~size_align_code_t() = default;

#ifdef QHPI_ENABLE
    constexpr size_align_code_t(size_t sz, size_t algn) : code(0)
    {
        assert(algn >= 1u && algn <= 32768u && (algn & (algn - 1)) == 0 && "bad alignment");
        assert(sz > 0 && sz % algn == 0 && "bad size");
        assert((algn == 4 || algn == 8) && "bad alignment");
        unsigned log2a = (algn == 4 ? log2_floor_of<4>() : log2_floor_of<8>());
        code = (sz / algn) * 16 | log2a;
    }
#endif

    // construct for a given op type T, e.g. size_align_code_t::for_type<OpType>();
    template <typename T> static constexpr size_align_code_t for_type()
    {
        size_align_code_t result{};
        constexpr size_t sz = sizeof(T);
        constexpr size_t algn = alignof(T);
        static_assert(algn >= 1u && algn <= 32768u && (algn & (algn - 1)) == 0, "bad alignment");
        static_assert(sz > 0 && sz % algn == 0, "bad size");
        constexpr unsigned log2a = log2_floor_of<algn>();
        result.code = (sz / algn) * 16 | log2a;
        return result;
    }
    // construct for a given op type T, but with additional size (caller must ensure
    // this is a multiple of the alignment).
    // extra < 0 is allowed (with appropriate caution); sizeof(T) + extra must be > 0.
    template <typename T> static constexpr size_align_code_t for_type_with_extra(int extra)
    {
        size_align_code_t result{};
        size_t sz = sizeof(T) + extra;
        constexpr size_t algn = alignof(T);
        static_assert(algn >= 1u && algn <= 32768u && (algn & (algn - 1)) == 0, "bad alignment");
        assert(int(sizeof(T)) + extra > 0 && sz % algn == 0);
        constexpr unsigned log2a = log2_floor_of<algn>();
        result.code = (sz / algn) * 16 | log2a;
        return result;
    }
    size_t constexpr size() const { return (code >> 4u) << (code & 0xFu); }
    size_t constexpr align() const { return size_t(1) << (code & 0xFu); }
    bool constexpr is_null() const { return code == 0; }

    // LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
    constexpr bool operator==(size_align_code_t const &other) const noexcept { return code == other.code; }
    constexpr bool operator!=(size_align_code_t const &other) const noexcept { return !operator==(other); }
    // LCOV_EXCL_STOP
};

} // namespace hnnx

POP_VISIBILITY()

#endif
