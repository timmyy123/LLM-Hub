// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef PERF_TIMING_H
#define PERF_TIMING_H 1

#include "weak_linkage.h"
#include "macros_attribute.h"

#include <cstdint>

PUSH_VISIBILITY(default)

class PcyclePoint {
  public:
    API_EXPORT PcyclePoint(bool enable);
    API_EXPORT void stop();
    API_EXPORT uint64_t get_total() const { return end > start ? (end - start) : 0; }
    API_EXPORT uint64_t get_start() const { return start; }
    API_EXPORT uint64_t get_end() const { return end; }
    //private:
    std::uint64_t start;
    std::uint64_t end;
};

POP_VISIBILITY()

#endif //PERF_TIMING_H
