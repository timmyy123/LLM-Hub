// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef ENTIRE_DEFOPT_H_
#define ENTIRE_DEFOPT_H_

#include "repl_funcs.h"
#include "match_op_fwd.h"

namespace hnnx {
class GraphOptInfo;

class entire_defopt {
  public:
    hnnx::MatchAst_uptr matcher;
    ReplFuncBool constraint;
    ReplFunc replacement;
    ReplFunc replacement_first;
    void (*register_tiling)(GraphOptInfo *);
};

using get_entire_defopt_t = entire_defopt (*)();

template <typename T> entire_defopt get_entire_defopt();
} // namespace hnnx
#endif
