// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================
#ifndef BUILD_OPTIONS_PUB_H
#define BUILD_OPTIONS_PUB_H 1

#include <cstdint>

#if !defined(HEX_ARCH)
#define HEX_ARCH 0
#endif

namespace build_options_pub {

constexpr std::uint16_t hex_arch{HEX_ARCH};

#ifdef WITH_OPT_DEBUG
#ifndef DEFOPT_LOG
#define DEFOPT_LOG 1
#endif
#endif

#ifdef DEFOPT_LOG
constexpr bool DefOptLog = true;
#else
constexpr bool DefOptLog = false;
#endif

#ifdef DEBUG_REGISTRY
constexpr bool DebugRegistry = true;
#else
constexpr bool DebugRegistry = false;
#endif

#ifdef PREPARE_DISABLED
static constexpr bool WITH_PREPARE = false;
#else
static constexpr bool WITH_PREPARE = true;
#endif

} // namespace build_options_pub

#endif // BUILD_OPTIONS_PUB_H
