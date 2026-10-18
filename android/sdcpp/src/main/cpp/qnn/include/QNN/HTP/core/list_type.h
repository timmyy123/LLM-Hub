// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#pragma once
namespace hnnx {
enum ListType { MainList, VecList, MtxList, EltList };
// Element count of ListType -- kept as a free constant (not an enumerator)
// so switches on ListType remain exhaustive under -Wswitch.
constexpr unsigned ListTypeCount = EltList + 1;
} // namespace hnnx
