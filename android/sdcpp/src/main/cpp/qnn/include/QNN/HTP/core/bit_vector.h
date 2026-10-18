// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#pragma once

#include <algorithm>
#include <limits>
#include <cassert>
#include <type_traits>

// A bit vector that can additionally represent an unbounded run of 1s starting
// at some index (used to encode a trailing "..." in a name list). The
// explicit `m_bits` field always mirrors that run for in-range indices, so
// bitwise operators comindexe without special-casing the implicit range.
template <typename bits_t> class BitVector {
    static_assert(std::is_unsigned<bits_t>::value && !std::is_same<bits_t, bool>::value,
                  "bits_t must be an unsigned integer type");

  public:
    static constexpr unsigned NUM_BITS = 8 * sizeof(bits_t);

  private:
    static constexpr unsigned NO_ABOVE = std::numeric_limits<unsigned>::max();

    bits_t m_bits = 0;
    unsigned m_set_at_and_above_index = NO_ABOVE;

    // Mask of all in-range indices at or above `index`. Used to mirror the
    // implicit m_set_at_and_above_index range into `m_bits`.
    static constexpr bits_t mask_at_and_above(unsigned index) { return index >= NUM_BITS ? 0u : (~0u << index); }

  public:
    BitVector() = default;
    BitVector(bits_t bits) : m_bits(bits) {}
    BitVector(bits_t bits, unsigned m_set_at_and_above_index)
        : m_bits(bits | mask_at_and_above(m_set_at_and_above_index)), m_set_at_and_above_index(m_set_at_and_above_index)
    {
    }

    bool operator[](unsigned index) const
    {
        return index >= m_set_at_and_above_index || (index < NUM_BITS && (m_bits & (1u << index)));
    }

    void set(unsigned index)
    {
        // index >= NUM_BITS cannot be represented in `m_bits` and lowering `m_set_at_and_above_index` to mark it
        // would also flip other indices
        assert(index < NUM_BITS);
        m_bits |= 1u << index;
    }
    void set_at_and_above(unsigned index)
    {
        m_set_at_and_above_index = index;
        m_bits |= mask_at_and_above(index);
    }

    bits_t get_bits() const { return m_bits; }

    explicit operator bool() const { return m_bits != 0 || m_set_at_and_above_index != NO_ABOVE; }

    bool operator==(const BitVector &rhs) const
    {
        return m_bits == rhs.m_bits && m_set_at_and_above_index == rhs.m_set_at_and_above_index;
    }
    bool operator!=(const BitVector &rhs) const { return !(*this == rhs); }

    bool operator==(bits_t rhs) const { return m_bits == rhs && m_set_at_and_above_index == NO_ABOVE; }
    bool operator!=(bits_t rhs) const { return !(*this == rhs); }

    BitVector operator|(const BitVector &rhs) const
    {
        return {static_cast<bits_t>(m_bits | rhs.m_bits),
                std::min(m_set_at_and_above_index, rhs.m_set_at_and_above_index)};
    }
    BitVector operator&(const BitVector &rhs) const
    {
        return {static_cast<bits_t>(m_bits & rhs.m_bits),
                std::max(m_set_at_and_above_index, rhs.m_set_at_and_above_index)};
    }

    // One past the index of the highest set bit (0 for `b == 0`). Equivalent
    // to NUM_BITS - countl_zero(b), but written as a loop to stay within the
    // C++17 standard library and avoid the compiler-extension `__builtin_clz`
    // (which AUTOSAR flags as non-portable and is UB on zero input).
    static constexpr unsigned highest_set_bit_plus_one(bits_t b)
    {
        unsigned p = 0;
        while (b) {
            ++p;
            b >>= 1;
        }
        return p;
    }

    BitVector operator~() const
    {
        if (m_set_at_and_above_index == NO_ABOVE) {
            // Input has no implicit-1 range; the complement does. The new
            // boundary is one past the highest set bit of the original (above
            // that, m_bits were 0 → complement is 1). For m_bits == 0, the
            // complement is all 1s (boundary = 0).
            return {static_cast<bits_t>(~m_bits), highest_set_bit_plus_one(m_bits)};
        }
        // Input has an implicit-1 range starting at m_set_at_and_above_index; complement removes it.
        const bits_t low_mask = m_set_at_and_above_index >= NUM_BITS ? static_cast<bits_t>(~0u)
                                                                     : ((1u << m_set_at_and_above_index) - 1);
        return {static_cast<bits_t>(~m_bits & low_mask), NO_ABOVE};
    }
    BitVector operator|(bits_t rhs) const { return *this | BitVector(rhs); }
    BitVector operator&(bits_t rhs) const { return *this & BitVector(rhs); }
    BitVector &operator|=(BitVector rhs)
    {
        *this = *this | rhs;
        return *this;
    }
    BitVector &operator&=(BitVector rhs)
    {
        *this = *this & rhs;
        return *this;
    }
    BitVector &operator|=(bits_t rhs) { return *this |= BitVector(rhs); }
    BitVector &operator&=(bits_t rhs) { return *this &= BitVector(rhs); }
    BitVector operator>>(unsigned num_shift) const
    {
        unsigned new_above =
                m_set_at_and_above_index == NO_ABOVE
                        ? NO_ABOVE
                        : (m_set_at_and_above_index >= num_shift ? m_set_at_and_above_index - num_shift : 0);
        // Right-shifting drops the high implicit-range bits; restore them via the constructor.
        return {static_cast<bits_t>(m_bits >> num_shift), new_above};
    }
    BitVector operator<<(unsigned num_shift) const
    {
        unsigned new_above = m_set_at_and_above_index == NO_ABOVE ? NO_ABOVE : m_set_at_and_above_index + num_shift;
        return {static_cast<bits_t>(m_bits << num_shift), new_above};
    }
};
