// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================
#ifndef OP_REGISTER_MACROS_H_
#define OP_REGISTER_MACROS_H_

#include "compile_time_string.h"

// Need the line number to avoid making the same template specialization
// multiple times
#define MDT(W, LINE)                                                                                                   \
    namespace fold {                                                                                                   \
    template <> struct ModifiedDerivedType<&W, LINE> : public ModifiedDerivedTypeParent {                              \
        using Modified = typename DerivedType<&W>::type;                                                               \
    };                                                                                                                 \
    } //namespace fold

#define COUNT_ARGS_IMPL(_1, _2, _3, count, ...) count
#define COUNT_ARGS(...)                         COUNT_ARGS_IMPL(__VA_ARGS__, 3, 2, 1)

#define PROCESS_NMVRT_VERSION_IMPL(count, ...) PROCESS_NMVRT_VERSION_IMPL2(count, __VA_ARGS__)

#define PROCESS_NMVRT_VERSION_IMPL2(count, ...) PROCESS_NMVRT_VERSION_IMPL2_##count(__VA_ARGS__)

#define PROCESS_NMVRT_VERSION_IMPL2_1(nm)      nm
#define PROCESS_NMVRT_VERSION_IMPL2_2(nm, ver) nm ver

#define PROCESS_NMVRT_VERSION(...) PROCESS_NMVRT_VERSION_IMPL(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)

/** @brief Create an Op type's type suffix from an optional name variant and argument types */
#define TYPE_SUFFIX(OP, NMVRT, ARGS, ...)                                                                              \
    (hnnx::ConcatStr<hnnx::ConstexprStrLen(OP), hnnx::ConstexprStrLen(NMVRT) + hnnx::ConstexprStrLen(ARGS)>(           \
            OP, (hnnx::ConcatStr<hnnx::ConstexprStrLen(NMVRT), hnnx::ConstexprStrLen(ARGS)>(NMVRT, ARGS).data())))

#ifndef OP_REG_COMPILE
#define DEF_NATIVE_OP(F, OP, LINE, ...)        DEF_NATIVE_OP_NMVRT(F, F, OP, LINE, false, "", __VA_ARGS__)
#define DEF_NATIVE_OP_LEGACY(F, OP, LINE, ...) DEF_NATIVE_OP_NMVRT(F, F, OP, LINE, true, "", __VA_ARGS__)

#define DEF_NATIVE_OP_NO_TCM_FOLDING(F, OP, ...) DEF_NATIVE_OP_NMVRT_NO_TCM_FOLDING(&F, &F, OP, false, "", __VA_ARGS__)
#define DEF_NATIVE_OP_NO_TCM_FOLDING_LEGACY(F, OP, ...)                                                                \
    DEF_NATIVE_OP_NMVRT_NO_TCM_FOLDING(&F, &F, OP, true, "", __VA_ARGS__)

#define DEF_NATIVE_OP_NMVRT(F, W, OP, LINE, IS_LEGACY, ...)                                                            \
    DEF_NATIVE_OP_NMVRT_IMPL(F, W, OP, LINE, IS_LEGACY, PROCESS_NMVRT_VERSION(__VA_ARGS__))

#define DEF_NATIVE_OP_NMVRT_NO_TCM_FOLDING(F, W, OP, IS_LEGACY, ...)                                                   \
    DEF_NATIVE_OP_NMVRT_NO_TCM_FOLDING_IMPL(F, W, OP, IS_LEGACY, PROCESS_NMVRT_VERSION(__VA_ARGS__))

#define DEF_NATIVE_OP_NMVRT_IMPL(F, W, OP, LINE, IS_LEGACY, NMVRT)                                                     \
    MDT(F, LINE)                                                                                                       \
    APPEND_REG_OP_ELEM(&W, THIS_PKG_NAME_STR "::" OP NMVRT,                                                            \
                       TYPE_SUFFIX(OP, NMVRT, hnnx::ArgsTuples2<&F>::inputTypeNames), LINE, IS_LEGACY)

/** __VA_ARGS__ contains versioning info, and passing into TYPE_SUFFIX as name variant */
#define DEF_NATIVE_OP_NMVRT_NO_TCM_FOLDING_IMPL(F, W, OP, IS_LEGACY, NMVRT)                                            \
    APPEND_REG_OP_ELEM_NO_TCM_FOLDING(W, THIS_PKG_NAME_STR "::" OP,                                                    \
                                      TYPE_SUFFIX(OP, NMVRT, hnnx::ArgsTuples2<F>::inputTypeNames), false, IS_LEGACY,  \
                                      __LINE__)

#define DEF_NATIVE_OP_NMVRT_NO_TCM_FOLDING_REG(F, W, OP, IS_LEGACY, REG, ...)                                          \
    DEF_NATIVE_OP_NMVRT_NO_TCM_FOLDING_REG_IMPL(F, W, OP, IS_LEGACY, REG, PROCESS_NMVRT_VERSION(__VA_ARGS__))

#define DEF_NATIVE_OP_NMVRT_NO_TCM_FOLDING_REG_IMPL(F, W, OP, IS_LEGACY, REG, NMVRT)                                   \
    APPEND_REG_OP_ELEM_NO_TCM_FOLDING_REG(W, THIS_PKG_NAME_STR "::" OP,                                                \
                                          TYPE_SUFFIX(OP, NMVRT, hnnx::ArgsTuples2<F>::inputTypeNames), false,         \
                                          IS_LEGACY, REG, __LINE__)

#else
#define DEF_NATIVE_OP(F, OP, LINE)                          __reg_op__(F, OP)<<<__FILE__, __LINE__>>>
#define DEF_NATIVE_OP_NO_TCM_FOLDING(F, OP)                 __reg_op__(F, OP)<<<__FILE__, __LINE__>>>
#define DEF_NATIVE_OP_NMVRT(F, W, OP, LINE, NMVRT)          __reg_op__(F, OP)<<<__FILE__, __LINE__>>>
#define DEF_NATIVE_OP_NMVRT_NO_TCM_FOLDING(F, W, OP, NMVRT) __reg_op__(F, OP)<<<__FILE__, __LINE__>>>
#endif

// TCM folding is an optimization to reduce skel size, so we only need it for execute.
#if defined(PREPARE_DISABLED) && !defined(TCM_FOLDING_DISABLED)
#define REGISTER_OP(F, STR, ...)        DEF_NATIVE_OP(F, STR, __LINE__, __VA_ARGS__)
#define REGISTER_OP_LEGACY(F, STR, ...) DEF_NATIVE_OP_LEGACY(F, STR, __LINE__, __VA_ARGS__)
#else
#define REGISTER_OP(F, STR, ...)        DEF_NATIVE_OP_NO_TCM_FOLDING(F, STR, __VA_ARGS__)
#define REGISTER_OP_LEGACY(F, STR, ...) DEF_NATIVE_OP_NO_TCM_FOLDING_LEGACY(F, STR, __VA_ARGS__)
#endif

// see register-op-tcm-folding.md
#define REGISTER_OP_NO_TCM_FOLDING(F, STR, ...)        DEF_NATIVE_OP_NO_TCM_FOLDING(F, STR, __VA_ARGS__)
#define REGISTER_OP_NO_TCM_FOLDING_LEGACY(F, STR, ...) DEF_NATIVE_OP_NO_TCM_FOLDING_LEGACY(F, STR, __VA_ARGS__)

#define REGISTER_OP_WRAPPER(F, W, STR, NMVRT, ...)                                                                     \
    DEF_NATIVE_OP_NMVRT_NO_TCM_FOLDING(&F, W, STR, false, NMVRT, __VA_ARGS__)
#define REGISTER_OP_WRAPPER_LEGACY(F, W, STR, NMVRT, ...)                                                              \
    DEF_NATIVE_OP_NMVRT_NO_TCM_FOLDING(&F, W, STR, true, NMVRT, __VA_ARGS__)

// Registry-scoped op registration macros. These register op implementations into a
// named registry so they only take effect when that registry is active.
// Ops/Opts Registries is the agreed-upon path forward for the project: this enables
// utilization of registries within the same pickle driver without rebuilding.
#define REGISTER_OP_WRAPPER_REG(F, W, STR, REG, NMVRT, ...)                                                            \
    DEF_NATIVE_OP_NMVRT_NO_TCM_FOLDING_REG(&F, W, STR, false, REG, NMVRT, __VA_ARGS__)

#define REGISTER_OP_WRAPPER_REG_LEGACY(F, W, STR, REG, NMVRT, ...)                                                     \
    DEF_NATIVE_OP_NMVRT_NO_TCM_FOLDING_REG(&F, W, STR, true, REG, NMVRT, __VA_ARGS__)

#define REGISTER_OP_EXT(F, STR, NMVRT, ...)        REGISTER_OP_WRAPPER(F, &F, STR, NMVRT, __VA_ARGS__)
#define REGISTER_OP_EXT_LEGACY(F, STR, NMVRT, ...) REGISTER_OP_WRAPPER_LEGACY(F, &F, STR, NMVRT, __VA_ARGS__)

#define REGISTER_OP_EXT_REG(F, STR, REG, NMVRT, ...) REGISTER_OP_WRAPPER_REG(F, &F, STR, REG, NMVRT, __VA_ARGS__)
#define REGISTER_OP_EXT_REG_LEGACY(F, STR, REG, NMVRT, ...)                                                            \
    REGISTER_OP_WRAPPER_REG_LEGACY(F, &F, STR, REG, NMVRT, __VA_ARGS__)

#define REGISTER_OP_HVX_EXT(F, STR, NMVRT, ...)                                                                        \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HVX)                                                                \
    REGISTER_OP_EXT(F, STR, NMVRT, __VA_ARGS__)

// Legacy version
#define REGISTER_OP_HVX_EXT_LEGACY(F, STR, NMVRT, ...)                                                                 \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HVX)                                                                \
    REGISTER_OP_EXT_LEGACY(F, STR, NMVRT, __VA_ARGS__)

#define REGISTER_OP_HVX(F, STR, ...)                                                                                   \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HVX)                                                                               \
    REGISTER_OP(F, STR, __VA_ARGS__)

// Legacy version
#define REGISTER_OP_HVX_LEGACY(F, STR, ...)                                                                            \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HVX)                                                                               \
    REGISTER_OP_LEGACY(F, STR, __VA_ARGS__)

#define REGISTER_OP_HVX_NO_TCM_FOLDING(F, STR, ...)                                                                    \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HVX)                                                                \
    REGISTER_OP_NO_TCM_FOLDING(F, STR, __VA_ARGS__)

// Legacy version
#define REGISTER_OP_HVX_NO_TCM_FOLDING_LEGACY(F, STR, ...)                                                             \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HVX)                                                                \
    REGISTER_OP_NO_TCM_FOLDING_LEGACY(F, STR, __VA_ARGS__)

#define REGISTER_OP_HVX_COPY(F, STR, ...)                                                                              \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HVX, Flags::IS_COPY);                                                              \
    REGISTER_OP(F, STR, __VA_ARGS__)

// Legacy version
#define REGISTER_OP_HVX_COPY_LEGACY(F, STR, ...)                                                                       \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HVX, Flags::IS_COPY);                                                              \
    REGISTER_OP_LEGACY(F, STR, __VA_ARGS__)

#define REGISTER_OP_HVX_COPY_NO_TCM_FOLDING(F, STR, ...)                                                               \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HVX, Flags::IS_COPY)                                                \
    REGISTER_OP_NO_TCM_FOLDING(F, STR, __VA_ARGS__)

// Legacy version
#define REGISTER_OP_HVX_COPY_NO_TCM_FOLDING_LEGACY(F, STR, ...)                                                        \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HVX, Flags::IS_COPY)                                                \
    REGISTER_OP_NO_TCM_FOLDING_LEGACY(F, STR, __VA_ARGS__)

#define REGISTER_OP_HLX_EXT(F, STR, NMVRT, ...)                                                                        \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HLX)                                                                \
    REGISTER_OP_EXT(F, STR, NMVRT, __VA_ARGS__)

// Legacy version
#define REGISTER_OP_HLX_EXT_LEGACY(F, STR, NMVRT, ...)                                                                 \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HLX)                                                                \
    REGISTER_OP_EXT_LEGACY(F, STR, NMVRT, __VA_ARGS__)

#define REGISTER_OP_HLX(F, STR, ...)                                                                                   \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HLX)                                                                               \
    REGISTER_OP(F, STR, __VA_ARGS__)

// Legacy version
#define REGISTER_OP_HLX_LEGACY(F, STR, ...)                                                                            \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HLX)                                                                               \
    REGISTER_OP_LEGACY(F, STR, __VA_ARGS__)

#define REGISTER_OP_HLX_NO_TCM_FOLDING(F, STR, ...)                                                                    \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HLX)                                                                \
    REGISTER_OP_NO_TCM_FOLDING(F, STR, __VA_ARGS__)
//Legacy version
#define REGISTER_OP_HLX_NO_TCM_FOLDING_LEGACY(F, STR, ...)                                                             \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HLX)                                                                \
    REGISTER_OP_NO_TCM_FOLDING_LEGACY(F, STR, __VA_ARGS__)

#define REGISTER_OP_HLX_COPY(F, STR, ...)                                                                              \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HLX, Flags::IS_COPY);                                                              \
    REGISTER_OP(F, STR, __VA_ARGS__)

// Legacy version
#define REGISTER_OP_HLX_COPY_LEGACY(F, STR, ...)                                                                       \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HLX, Flags::IS_COPY);                                                              \
    REGISTER_OP_LEGACY(F, STR, __VA_ARGS__)

#define REGISTER_OP_HLX_COPY_NO_TCM_FOLDING(F, STR, ...)                                                               \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HLX, Flags::IS_COPY)                                                \
    REGISTER_OP_NO_TCM_FOLDING(F, STR, __VA_ARGS__)

// Legacy version
#define REGISTER_OP_HLX_COPY_NO_TCM_FOLDING_LEGACY(F, STR, ...)                                                        \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HLX, Flags::IS_COPY)                                                \
    REGISTER_OP_NO_TCM_FOLDING_LEGACY(F, STR, __VA_ARGS__)

#define REGISTER_OP_HMX(F, STR, ...)                                                                                   \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HMX);                                                                              \
    REGISTER_OP(F, STR, __VA_ARGS__)

// Legacy version
#define REGISTER_OP_HMX_LEGACY(F, STR, ...)                                                                            \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HMX);                                                                              \
    REGISTER_OP_LEGACY(F, STR, __VA_ARGS__)

#define REGISTER_OP_HMX_NO_TCM_FOLDING(F, STR, ...)                                                                    \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HMX);                                                               \
    REGISTER_OP_NO_TCM_FOLDING(F, STR, __VA_ARGS__)

// Legacy version
#define REGISTER_OP_HMX_NO_TCM_FOLDING_LEGACY(F, STR, ...)                                                             \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HMX);                                                               \
    REGISTER_OP_NO_TCM_FOLDING_LEGACY(F, STR, __VA_ARGS__)

#define REGISTER_OP_HVX_SRC_DESTRUCTIVE(F, STR, ...)                                                                   \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HVX, Flags::CAN_BE_SRC_DESTRUCTIVE);                                               \
    REGISTER_OP(F, STR, __VA_ARGS__)

// Legacy version
#define REGISTER_OP_HVX_SRC_DESTRUCTIVE_LEGACY(F, STR, ...)                                                            \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HVX, Flags::CAN_BE_SRC_DESTRUCTIVE);                                               \
    REGISTER_OP_LEGACY(F, STR, __VA_ARGS__)

#define REGISTER_OP_HLX_SRC_DESTRUCTIVE(F, STR, ...)                                                                   \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HLX, Flags::CAN_BE_SRC_DESTRUCTIVE);                                               \
    REGISTER_OP(F, STR, __VA_ARGS__)

// Legacy version
#define REGISTER_OP_HLX_SRC_DESTRUCTIVE_LEGACY(F, STR, ...)                                                            \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HLX, Flags::CAN_BE_SRC_DESTRUCTIVE);                                               \
    REGISTER_OP_LEGACY(F, STR, __VA_ARGS__)

#define REGISTER_OP_HVX_SRC_DESTRUCTIVE_NO_TCM_FOLDING(F, STR, ...)                                                    \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HVX, Flags::CAN_BE_SRC_DESTRUCTIVE);                                \
    REGISTER_OP_NO_TCM_FOLDING(F, STR, __VA_ARGS__)

// Legacy version
#define REGISTER_OP_HVX_SRC_DESTRUCTIVE_NO_TCM_FOLDING_LEGACY(F, STR, ...)                                             \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HVX, Flags::CAN_BE_SRC_DESTRUCTIVE);                                \
    REGISTER_OP_NO_TCM_FOLDING_LEGACY(F, STR, __VA_ARGS__)

#define REGISTER_OP_HLX_SRC_DESTRUCTIVE_NO_TCM_FOLDING(F, STR)                                                         \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HLX, Flags::CAN_BE_SRC_DESTRUCTIVE);                                \
    REGISTER_OP_NO_TCM_FOLDING(F, STR)

// Legacy version
#define REGISTER_OP_HLX_SRC_DESTRUCTIVE_NO_TCM_FOLDING_LEGACY(F, STR)                                                  \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HLX, Flags::CAN_BE_SRC_DESTRUCTIVE);                                \
    REGISTER_OP_NO_TCM_FOLDING_LEGACY(F, STR)

// Register Ops which are typically removed during const propagation, but may
// also appear in the aux graph and therefore must support deserialization
// (and hence op-versioning) like any other registered op.
#define REGISTER_OP_CONST_HVX(F, STR, ...)                                                                             \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HVX, Flags::IS_CONST);                                                             \
    REGISTER_OP(F, STR, __VA_ARGS__)

#define REGISTER_OP_CONST_HVX_NO_TCM_FOLDING(F, STR, ...)                                                              \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HVX, Flags::IS_CONST);                                              \
    REGISTER_OP_NO_TCM_FOLDING(F, STR, __VA_ARGS__)

#define REGISTER_OP_CONST(F, STR, ...)                                                                                 \
    FLAGS_FOR_DT(F, Flags::IS_CONST);                                                                                  \
    REGISTER_OP(F, STR, __VA_ARGS__)

#define REGISTER_OP_HVX_EXT_REG(F, STR, REG, NMVRT, ...)                                                               \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HVX);                                                                              \
    REGISTER_OP_EXT_REG(F, STR, REG, NMVRT, __VA_ARGS__)

#define REGISTER_OP_HVX_NO_TCM_FOLDING_EXT_REG(F, STR, REG, NMVRT, ...)                                                \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HVX)                                                                \
    REGISTER_OP_EXT_REG(F, STR, REG, NMVRT, __VA_ARGS__)

#define REGISTER_OP_HVX_COPY_NO_TCM_FOLDING_EXT_REG(F, STR, REG, NMVRT, ...)                                           \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HVX, Flags::IS_COPY)                                                \
    REGISTER_OP_EXT_REG(F, STR, REG, NMVRT, __VA_ARGS__)

#define REGISTER_OP_HLX_EXT_REG(F, STR, REG, NMVRT, ...)                                                               \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HLX)                                                                               \
    REGISTER_OP_EXT_REG(F, STR, REG, NMVRT, __VA_ARGS__)

#define REGISTER_OP_HLX_NO_TCM_FOLDING_EXT_REG(F, STR, REG, NMVRT, ...)                                                \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HLX)                                                                \
    REGISTER_OP_EXT_REG(F, STR, REG, NMVRT, __VA_ARGS__)

#define REGISTER_OP_HLX_COPY_NO_TCM_FOLDING_EXT_REG(F, STR, REG, NMVRT, ...)                                           \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HLX, Flags::IS_COPY)                                                \
    REGISTER_OP_EXT_REG(F, STR, REG, NMVRT, __VA_ARGS__)

#define REGISTER_OP_HMX_NO_TCM_FOLDING_EXT_REG(F, STR, REG, NMVRT, ...)                                                \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HMX);                                                               \
    REGISTER_OP_EXT_REG(F, STR, REG, NMVRT, __VA_ARGS__)

#define REGISTER_OP_HVX_SRC_DESTRUCTIVE_EXT_REG(F, STR, REG, NMVRT, ...)                                               \
    FLAGS_FOR_DT(F, Flags::RESOURCE_HVX, Flags::CAN_BE_SRC_DESTRUCTIVE);                                               \
    REGISTER_OP_EXT_REG(F, STR, REG, NMVRT, __VA_ARGS__)

#define REGISTER_OP_HVX_SRC_DESTRUCTIVE_NO_TCM_FOLDING_EXT_REG(F, STR, REG, NMVRT, ...)                                \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HVX, Flags::CAN_BE_SRC_DESTRUCTIVE);                                \
    REGISTER_OP_EXT_REG(F, STR, REG, NMVRT, __VA_ARGS__)

#define REGISTER_OP_HLX_SRC_DESTRUCTIVE_NO_TCM_FOLDING_EXT_REG(F, STR, REG, NMVRT)                                     \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HLX, Flags::CAN_BE_SRC_DESTRUCTIVE);                                \
    REGISTER_OP_EXT_REG(F, STR, REG, NMVRT)
#endif
