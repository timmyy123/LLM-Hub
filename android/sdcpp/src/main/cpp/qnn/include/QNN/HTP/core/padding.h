// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef HEXNN_PADDING_H
#define HEXNN_PADDING_H 1
#include <array>
#include <utility>
#include <functional>
#include <algorithm>

/* 
 * This is a nice experiment, but maybe we should always just have padding
 * and represent nopadding with padding 0,0,0,0
 */

using Idx = size_t;
using SIdx = long;

template <Idx Rank> class NoPadding {
  public:
    static constexpr unsigned int is_padded = 0;
    template <typename PadT>
    inline constexpr std::array<Idx, Rank> pad_coords(const std::array<Idx, Rank> &coords,
                                                      const std::array<PadT, Rank> & /* left_padding */) const
    {
        std::array<Idx, Rank> ret{coords};
        return ret;
    }
    template <typename PadT>
    inline constexpr std::array<Idx, Rank> pad_coords(const std::array<SIdx, Rank> &coords,
                                                      const std::array<PadT, Rank> & /* left_padding */) const
    {
        std::array<Idx, Rank> ret{};
        std::transform(coords.cbegin(), coords.cend(), ret.begin(), [](SIdx x) -> Idx { return static_cast<Idx>(x); });
        return ret;
    }
};

/*
 * This padding is just the left/top padding, which affects the coordinates.
 * But if we want right/bottom padding, that would be extra information needed somewhere else.
 */

template <Idx Rank> class Padding {
  public:
    static constexpr unsigned int is_padded = 1;
    template <typename PadT>
    inline const std::array<Idx, Rank> pad_coords(const std::array<Idx, Rank> &coords,
                                                  const std::array<PadT, Rank> &left_padding) const
    {
        std::array<Idx, Rank> ret{};
        std::transform(left_padding.begin(), left_padding.end(), coords.begin(), ret.begin(), std::plus<Idx>());
        return ret;
    }
    template <typename PadT>
    inline const std::array<Idx, Rank> pad_coords(const std::array<SIdx, Rank> &coords,
                                                  const std::array<PadT, Rank> &left_padding) const
    {
        std::array<Idx, Rank> ret{};
        std::transform(left_padding.begin(), left_padding.end(), coords.begin(), ret.begin(), std::plus<Idx>());
        return ret;
    }
};
#endif
