// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================
#pragma once
#include <functional>
#include <typeinfo>
#include <typeindex>

// To suppress "-Wgnu-anonymous-struct, -Wnested-anon-types" warnings from float16.h and "-Wc99-extensions" from "qhpi.h" with clang compiler
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wshadow-field-in-constructor"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include "minihash.h"
#include "qhpi_internal.h"
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#ifndef QHPI_ENABLE
struct QHPI_Kernel;
#endif

namespace hnnx {
extern const char *kernel_name(const QHPI_Kernel *);
struct type_info {
    bool is_std;
    union {
        const std::type_info *std_type_info;
        const QHPI_Kernel *foreign_type_info;
    };
    type_info(const hnnx::type_info &std_type) = default;
    type_info(hnnx::type_info &&std_type) = default;
    // SAFTYSWCCB-1735 removed
    type_info(const std::type_info &std_type) : is_std(true), std_type_info(&std_type) {}
    type_info(const QHPI_Kernel *kernel) : is_std(false), foreign_type_info(kernel) {}
    ~type_info() = default;
    // SAFTYSWCCB-1735 removed end
    type_info &operator=(type_info &&other) noexcept = default;
    type_info &operator=(const type_info &ty) = default;
#ifndef PREPARE_DISABLED
    auto value() const -> void const *
    {
        return (is_std ? static_cast<void const *>(std_type_info) : static_cast<void const *>(foreign_type_info));
    }
#ifdef QHPI_ENABLE
    const char *name() const { return (is_std ? std_type_info->name() : kernel_name(foreign_type_info)); }
#else
    const char *name() const { return std_type_info->name(); }
#endif
    bool operator==(const type_info ty) const
    {
        if (is_std != ty.is_std) return false;
        if (is_std) return *std_type_info == *ty.std_type_info;
        return foreign_type_info == ty.foreign_type_info;
    }
    bool operator!=(const type_info ty) const { return !(*this == ty); }
#endif
};
} // namespace hnnx

#ifndef PREPARE_DISABLED
namespace std {
template <> struct hash<const hnnx::type_info> {
    hash() = default;
    size_t operator()(const hnnx::type_info info) const
    {
        if (info.is_std) return type_index(*info.std_type_info).hash_code();
        return hash<void const *>{}(info.foreign_type_info);
    }
};
} // namespace std
#endif
