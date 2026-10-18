//==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
//==============================================================================

#ifndef OP_REGISTRIES_H
#define OP_REGISTRIES_H

#include <array>
#include <string_view>

#define REGISTRY_CORE "core"
#define REGISTRY_512B "512b"

namespace hnnx {

inline constexpr std::array<std::string_view, 2> ALLOWED_OP_REGISTRIES = {
        REGISTRY_CORE,
        REGISTRY_512B,
};

} // namespace hnnx

#endif // OP_REGISTRIES_H
