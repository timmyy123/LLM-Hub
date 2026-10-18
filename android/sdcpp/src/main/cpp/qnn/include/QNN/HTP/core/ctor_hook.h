// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef CTOR_HOOK_H
#define CTOR_HOOK_H 1

class Graph;

namespace hnnx {
class OpIoPtrs;

template <typename T> inline void ctor_hook(Graph &, T & /*ref*/)
{
    return;
}

// the 'pre-hook' can install an OpHookBase pointer into the op_io_ptrs
// default is to do nothing.
template <typename T> inline void ctor_ophook(OpIoPtrs const & /*op_io_ptrs*/)
{
    return;
}

} // namespace hnnx

#ifdef PREPARE_DISABLED
#define CTOR_HOOK(FUNC, VAR, CODE)
#else
#define CTOR_HOOK(FUNC, VAR, CODE)                                                                                     \
    template <>                                                                                                        \
    [[maybe_unused]] inline void hnnx::ctor_hook(Graph &graph_in, typename DerivedType<(&FUNC)>::type &VAR)            \
    {                                                                                                                  \
        CODE                                                                                                           \
    }
#endif

// maybe we could add more than one ophook... just define this with different #'s of parms.
// 'HOOKCLASS' must be a subclass of OpHookBase, which defines the hook.
#define CTOR_OPHOOK(FUNC, HOOKCLASS)                                                                                   \
    template <> inline void hnnx::ctor_ophook<typename DerivedType<(&FUNC)>::type>(OpIoPtrs const &op_io_ptrs)         \
    {                                                                                                                  \
        static constexpr HOOKCLASS hook;                                                                               \
        const_cast<OpIoPtrs &>(op_io_ptrs).add_ophook(&hook);                                                          \
    }

// Macro to register self-slicing hook for ops that use COMPILER_FOR
// This registers the hook for both DerivedType (TypicalOpWithCompiler)
// and TypicalOp (which is what prepare() actually calls)
#define CTOR_OPHOOK_FOR_COMPILER(FUNC, HOOKCLASS)                                                                      \
    CTOR_OPHOOK(FUNC, HOOKCLASS)                                                                                       \
    namespace hnnx {                                                                                                   \
    template <> inline void ctor_ophook<TypicalOp<FUNC>>(OpIoPtrs const &op_io_ptrs)                                   \
    {                                                                                                                  \
        static constexpr HOOKCLASS hook;                                                                               \
        const_cast<OpIoPtrs &>(op_io_ptrs).add_ophook(&hook);                                                          \
    }                                                                                                                  \
    }

#endif
