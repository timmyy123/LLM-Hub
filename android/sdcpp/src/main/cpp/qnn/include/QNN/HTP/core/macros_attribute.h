// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef MACROS_MSVC_CPP17_H
#define MACROS_MSVC_CPP17_H

// Macros to compatible with MSVC and clang
#if defined(_MSC_VER)

// Macros to tells the compiler to never inline a particular member function
#define NOINLINE __declspec(noinline)
// Macros to force a function to be inlined, meaning that the function call is replaced with the function body at compile time
#define ALWAYSINLINE __forceinline

#ifndef API_EXPORT
//Macros to export symbol from a library
#define API_EXPORT __declspec(dllexport)
#endif
#ifndef API_HIDDEN
//Macros to hidden symbol from a library
#define API_HIDDEN

//Macros to tell the compiler that the function returns an object that is not aliased, that is, referenced by any other pointers.
#define RESTRICT_VAR __restrict

#endif
#else

#define NOINLINE     __attribute__((noinline))
#define ALWAYSINLINE __attribute((always_inline))

#ifndef API_EXPORT
/**
 * @brief The definition of API_EXPORT is commented out for
 * now, because it was causing problems with static analysis.
 * As an alternative, use the PUSH_VISIBILITY(default)/POP_VISIBILITY()
 * macros to ensure that symbols are exported on Linux.
*/
#define API_EXPORT /* [[gnu::visibility("default")]] */
#endif
#ifndef API_HIDDEN
#define API_HIDDEN [[gnu::visibility("hidden")]]
#endif

#define RESTRICT_VAR __restrict__

#endif // _MSC_VER

#if defined(__clang__)
// Attributes for functions, which can allow compiler to eliminate calls.
// Use with great care, and only where it matters.
// There's no reason to add these to functions defined as inline, since
// the compiler will determine these properties by analyzing the code.
// Probably best to apply these only to 'noexcept' functions.
//
// PURE
// - Function has no effect except to return a value
// - It can look at memory pointed to by its parameters, and can look at global
//    vars in general, and those can affect what is returned. strcmp is 'pure'.
// - Compiler assumes that if the return value is not used, the call can be eliminated;
//    or, if two calls are made within a function, both with the same parameter values, and
//    nothing between can change aything visible outside the calling function, then the
//    second call will return the same value as the first, and can be eliminated.
#define FUNC_ATTRIB_PURE __attribute__((pure))
//
// CONST (implies PURE)
// - Function has no effect except to return a value; the returned value depends only on
//    values of the passed parameters
// - It will *not* look at memory pointed to by its parameters, and result cannot be affected
//    by changes in global memory values.
// - Compiler assumes that if the return value is not used, the call can be eliminated;
//    and, that any two calls made with the same parameter values will return the same value,
//    with no constraint on what happens between.
#define FUNC_ATTRIB_CONST __attribute__((const))
#else
// Only defined for clang - we really only need these on hexagon, but it's good to have the syntax
// checked on X86-clang.
#define FUNC_ATTRIB_PURE
#define FUNC_ATTRIB_CONST
#endif // !clang

/**
 * @brief The following macros: [PUSH|POP|ENABLE|DISABLE]_WARNING,
 * allow for in-code enabing and disabling of compiler warnings in
 * a portable fashion.
*/
#define DO_PRAGMA(x) _Pragma(#x)

#define MSVC_NO_EQUIV
#define GNU_NO_EQUIV ""

#if defined(__clang__)
#define PUSH_WARNING()             DO_PRAGMA(clang diagnostic push)
#define POP_WARNING()              DO_PRAGMA(clang diagnostic pop)
#define ENABLE_WARNING(gnu, msvc)  DO_PRAGMA(clang diagnostic warning gnu)
#define DISABLE_WARNING(gnu, msvc) DO_PRAGMA(clang diagnostic ignored gnu)
#elif defined(__GNUG__)
#define PUSH_WARNING()             DO_PRAGMA(GCC diagnostic push)
#define POP_WARNING()              DO_PRAGMA(GCC diagnostic pop)
#define ENABLE_WARNING(gnu, msvc)  DO_PRAGMA(GCC diagnostic warning gnu)
#define DISABLE_WARNING(gnu, msvc) DO_PRAGMA(GCC diagnostic ignored gnu)
#elif defined(_MSC_VER)
#define PUSH_WARNING()             DO_PRAGMA(warning(push))
#define POP_WARNING()              DO_PRAGMA(warning(pop))
#define ENABLE_WARNING(gnu, msvc)  DO_PRAGMA(warning(default : msvc))
#define DISABLE_WARNING(gnu, msvc) DO_PRAGMA(warning(disable : msvc))
#else
#define PUSH_WARNING()
#define POP_WARNING()
#define ENABLE_WARNING(gnu, msvc)
#define DISABLE_WARNING(gnu, msvc)
#endif

#endif //MACROS_MSVC_H
