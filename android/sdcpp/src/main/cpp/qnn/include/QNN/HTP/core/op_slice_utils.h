// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef OP_SLICE_UTILS_H
#define OP_SLICE_UTILS_H 1

#include <cstdint>
#include <utility>
#include <algorithm>
#include "executable.h"

namespace hnnx {
// This is passed an 'op_slice_spec' and a number of units,
// and returns { first_idx, slice_n_units}
// so that each slice does a roughly equal number of units.
// Caller *must* be prepared for the returned slice_n_units to be 0; this can
// happen if n_units is less than the number of slices (or, if the slice count
// is more than MAX_NSLICE below, in which case the extra will do nothing).
// When n_slice_units is 0, 'first_idx' may be invalid (>= n_units).
//
//  E.g. for n_units = 50, and 4 slices:
//  find_slice_range( {n = 4, idx = 0}, 50)   ->  { 0, 12 }   // #0 does [0, 12)
//  find_slice_range( {n = 4, idx = 1}, 50)   ->  { 12, 13 }  // #1 does [12, 25)
//  find_slice_range( {n = 4, idx = 2}, 50)   ->  { 25, 12 }  // #2 does [25, 37)
//  find_slice_range( {n = 4, idx = 3}, 50)   ->  { 37, 13 }  // #3 does [37, 50)
//
inline constexpr unsigned MAX_NSLICE_in_find_slice_range = 10;
inline std::pair<unsigned, unsigned> //
find_slice_range(hnnx::op_slice_spec slice_spec, unsigned n_units)
{

    static constexpr unsigned div_factors[MAX_NSLICE_in_find_slice_range - 1] = {
            0x80000000, // = (1<<32)       / 2   (all of these are exact divides)
            0x55555556, // = ((1<<32) + 2) / 3
            0x40000000, // = (1<<32)       / 4
            0x33333334, // = ((1<<32) + 4) / 5
            0x2aaaaaab, // = ((1<<32) + 2) / 6
            0x24924925, // = ((1<<32) + 3) / 7
            0x20000000, // = (1<<32)       / 8
            0x1c71c71d, // = ((1<<32) + 5) / 9
            0x1999999a, // = ((1<<32) + 4) / 10
    };

    unsigned n_slices = slice_spec.num_slices();
    if (n_slices <= 1) return {0, n_units}; // special case

    n_slices = std::min(n_slices, MAX_NSLICE_in_find_slice_range);

    // Each does units from        idx_begin = (n_units * this_idx) / n_slices,
    // ...up to but not including  idx_end = (n_units * (this_idx + 1)) / n_slices.

    unsigned recip = div_factors[n_slices - 2];
    // To find nn/n_slices, we can find (recip * nn) >> 32.
    // This is not always correct for *very* large nn, it may be too large (by no more than 1 or 2)
    // but it is monotonic. We guard against 'too large' result by forcing 'idx_end' to be exactly
    // n_units in the last slice. As song as each slice's idx_end is the same as the next slice's
    // idx_begin, it works.
    // Note: unsigned((uint64_t(a) * b) >> 32) is one operation 'Rd = mpyu(Rs, Rt)' on hexagon.

    unsigned const this_idx = slice_spec.slice_idx();

    // idx_begin = this_index * n_units / n_slices;
    unsigned const idx_prod = this_idx * n_units;
    unsigned const idx_begin = unsigned((uint64_t(idx_prod) * recip) >> 32);

    // idx_end = (this_index + 1) * n_units / n_slices;
    unsigned idx_end = unsigned((uint64_t(idx_prod + n_units) * recip) >> 32);
    idx_end = std::min(idx_end, n_units);

    // Detect null slice:
    //   if n_slices < n_units, some of the slices will have idx_begin == idx_end.
    //   if n_slices is > MAX_NSLICE, then all this_idx >= MAX_NSLICE will have idx_end == n_units,
    //     and idx_begin >= n_units.
    n_units = (idx_begin < idx_end) ? (idx_end - idx_begin) : 0;
    return {idx_begin, n_units};
}

} // namespace hnnx

#endif // OP_SLICE_UTILS_H
