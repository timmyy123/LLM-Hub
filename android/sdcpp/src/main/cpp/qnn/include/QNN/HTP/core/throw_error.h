// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

// LCOV_EXCL_START [SAFTYSWCCB-1735]
//
// We can convert tests to coverage once the feature is added
//
#pragma once

#ifdef DEMAND_PAGING

// Only define throw_error if not already provided (e.g., by host/errors.h which
// uses FMT_STRING for std::string support). On the skel/hexagon side this is the
// only definition, so it always takes effect there.
#ifndef throw_error

#include <cstdarg>
#include <cstdio>
#include <stdexcept>

#define throw_error(fmt, ...) throw_error_func(__LINE__, __FILE__, (fmt), ##__VA_ARGS__)

[[noreturn]] static inline void throw_error_func(int line, const char *cur_file, const char *fmt, ...)
{
    constexpr size_t BufSize = 500;
    char str[BufSize];
    va_list args;
    va_start(args, fmt);
    vsnprintf(str, BufSize, fmt, args);
    va_end(args);
    throw std::runtime_error(str);
}

#endif

#endif // DEMAND_PAGING

// LCOV_EXCL_STOP
