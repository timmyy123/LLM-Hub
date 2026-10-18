// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

// We need the specific order for these headers
// clang-format off
#include "typical_op.h"
#include "ops_opts_registration.h"
// clang-format on

/**
 * @brief All external Op source files must invoke this macro at the top of the file,
 * before any COST_OF/REGISTER_OP/DEF_OPT calls.
 *
 */
#define BEGIN_PKG_OP_DEFINITION(NAME) INITIALIZE_TABLES()

/**
 * @brief All external Op source files must invoke this macro at the bottom of the
 * file, after all COST_OF/REGISTER_OP/DEF_OPT calls.
 *
 */
#define END_PKG_OP_DEFINITION(NAME) FINALIZE_TABLES(NAME)
