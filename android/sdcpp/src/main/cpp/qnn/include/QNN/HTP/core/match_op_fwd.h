// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef MATCH_OP_FWD_H_
#define MATCH_OP_FWD_H_

#include <memory>

namespace hnnx {
class MatchOpBase;
using MatchOp_uptr = std::unique_ptr<MatchOpBase>;
class MatchAstNode;
using MatchAst_uptr = std::unique_ptr<MatchAstNode>;
} // namespace hnnx
#endif
