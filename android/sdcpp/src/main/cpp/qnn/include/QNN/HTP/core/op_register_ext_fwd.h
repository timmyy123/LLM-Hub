// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef OP_REGISTER_EXT_FWD_H_
#define OP_REGISTER_EXT_FWD_H_

#include "macros_attribute.h"

#include <vector>
#include <memory>

namespace hnnx {
class PackageOpStorageBase;
}

API_HIDDEN std::vector<std::unique_ptr<hnnx::PackageOpStorageBase>> &current_package_ops_storage_vec_func();
#endif
