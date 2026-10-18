// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef LOG_H
#define LOG_H 1

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif

// Include HULI API if reachable. Builds that only sync hexagon/ and bindiff/
// (e.g. QNN OpPackage x86) will not have huli_api_log.h in their include path.
// HAVE_HULI_LOG is used as the discriminator for HULI-specific code sections.
#if __has_include("huli_api_log.h")
#include "huli_api_log.h"
#define HAVE_HULI_LOG 1
#else
#define HAVE_HULI_LOG 0
#endif

#include "weak_linkage.h"
#include "macros_attribute.h"

// Include fmt if reachable. Builds that consume only installed SDK headers
// (e.g. QNN MCP OpPackage) may not have fmt in their include path; in that
// case the logmsg_fmt / *logf family is simply not declared.
#if __has_include(<fmt/base.h>)
#include <cstdlib>
#include <fmt/base.h>
#define HAVE_FMT 1
#else
#define HAVE_FMT 0
#endif

#include <cstdarg>
#include <cstdint>
#include <chrono>
#include <string>
#include <utility>

// This should be fixed. Double underscores are illegal in user code
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#endif

#if !defined(__PRETTY_FUNCTION__) && !defined(__GNUC__)
#define __FUNC_INFO__ __FUNCSIG__
#else
#define __FUNC_INFO__ __PRETTY_FUNCTION__
#endif

#if defined(__clang__)
#pragma clang diagnostic pop // -Wreserved-macro-identifier
#endif

// GCC and Clang define a preprocessor macro which is just the basename of the current file.
#if defined(__FILE_NAME__)
#define FILE_BASENAME __FILE_NAME__
#else

// MSVC doesn't have this nice feature, so we have to do it manually.  Note that the entire path
// still ends up in the .rodata section, unfortunately.

// Constexpr that will strip the path off of the file for logging purposes
constexpr char const *stripFilePath(const char *path)
{
    const char *file = path;
    while (*path) {
        if (*path++ == '/') {
            file = path;
        }
    }
    return file;
}

#define FILE_BASENAME stripFilePath(__FILE__)

#endif // defined(__FILE_NAME__)

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X)        STRINGIZE_DETAIL(X)

#include "graph_status.h"
#include "cc_pp.h"

#ifdef __cplusplus
#include <cstdio>
#else
#include <stdio.h>
#endif

#ifndef NN_LOG_MAXLVL
#define NN_LOG_MAXLVL 0
#endif

#ifndef NN_LOG_DYNLVL
#define NN_LOG_DYNLVL 0
#endif

// If log level or the dynamic logging flag are defined but don't have a value,
// then consider them to be undefined.
#if ~(~NN_LOG_MAXLVL + 0) == 0 && ~(~NN_LOG_MAXLVL + 1) == 1
#undef NN_LOG_MAXLVL
#endif

#if ~(~NN_LOG_DYNLVL + 0) == 0 && ~(~NN_LOG_DYNLVL + 1) == 1
#undef NN_LOG_DYNLVL
#endif

/*
 * We have migrated using C++ features like iostream to printf strings.
 * Why?
 * * C++ iostream makes it more difficult to use mixed decimal/hex
 * * C++ iostream isn't easily compatible with on-target logging facilities
 * * C++ iostream is bad for code size, printf is much better
 */

//Log levels macro
#define NN_LOG_ERRORLVL         0 //Error log level is 0
#define NN_LOG_WARNLVL          1 //Warning log level is 1
#define NN_LOG_STATLVL          2 //Stats log level is 2
#define NN_LOG_INFOLVL          3 //Info log level is 3
#define NN_LOG_VERBOSELVL       4 //Verbose log level is from 4-10
#define NN_LOG_STATLVL_INTERNAL 8
#define NN_LOG_INFOLVL_INTERNAL 9
#define NN_LOG_DEBUGLVL         11 //Debug log level is > 10

typedef void (*DspLogCallbackFunc)(int level, const char *fmt, va_list args);

// Dynamically set the logging priority level.
PUSH_VISIBILITY(default)
EXTERN_C_BEGIN
extern "C" {

API_FUNC_EXPORT void SetLogPriorityLevel(int level);
API_FUNC_EXPORT int GetLogPriorityLevel();
API_FUNC_EXPORT void SetLogCallbackFunc(DspLogCallbackFunc fn);
API_FUNC_EXPORT DspLogCallbackFunc GetLogCallbackFunc();

// This prevents preemption if we're using the TID preemption mechanism.
// Enable format checking when we're ready to fix all of the broken formats!
//[[gnu::format(printf, 1, 2)]]
API_FUNC_EXPORT void nn_log_printf(const char *fmt, ...);
}
EXTERN_C_END
POP_VISIBILITY()

#ifdef __cplusplus
extern "C" {
#endif

// special log message for x86 that will log regardless logging level
void qnndsp_x86_log(const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#if defined(NN_LOG_DYNLVL) && (NN_LOG_DYNLVL > 0)

// Dynamic logging level test function.
inline bool log_condition(const int prio)
{
    return (prio <= GetLogPriorityLevel());
};

#elif defined(NN_LOG_MAXLVL)

// Logging level is fixed at compile time.
inline bool log_condition(const int prio)
{
    return ((prio <= NN_LOG_MAXLVL) ? true : false);
}

#else

// LCOV_EXCL_START [SAFTYSWCCB-996]
// Logging is completely disabled.
constexpr bool log_condition(const int /* prio */)
{
    return false;
}
// LCOV_EXCL_STOP

#endif

#ifdef ENABLE_QNNDSP_LOG

PUSH_VISIBILITY(default)
API_FUNC_EXPORT API_C_FUNC void API_FUNC_NAME(SetLogCallback)(DspLogCallbackFunc cbFn, int logPriority);

extern "C" {
API_FUNC_EXPORT void qnndsp_log(int prio, const char *FMT, ...);

API_FUNC_EXPORT void hv3_load_log_functions(decltype(SetLogCallback) **SetLogCallback_f);
}
POP_VISIBILITY()

#define MAKE_LOG_FMT_WITH_PREFIX(FMT, ...)                                                                             \
    "%s"                                                                                                               \
    ":" STRINGIZE(__LINE__) ":" FMT "\n",                                                                              \
            FILE_BASENAME, ##__VA_ARGS__

#define HV3_LOG(PRIO, FMT, ...) qnndsp_log(PRIO, FMT, ##__VA_ARGS__)

// _errlog_ must be usable as an expression (return errlog(...)) so it cannot
// use a do-while macro.  hv3_errlog_impl is a plain inline function that can
// appear in a comma-expression context.
// LCOV_EXCL_START [SAFTYSWCCB-996]
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
#pragma clang diagnostic ignored "-Wformat-security"
#endif
template <typename... Args> inline void hv3_errlog_impl(const char *fmt, Args... args)
{
    qnndsp_log(NN_LOG_ERRORLVL, fmt, args...);
}
#if defined(__clang__)
#pragma clang diagnostic pop // -Wformat-nonliteral -Wformat-security
#endif
// LCOV_EXCL_STOP

#elif HAVE_HULI_LOG // !ENABLE_QNNDSP_LOG, HULI available

#define MAKE_LOG_FMT_WITH_PREFIX(FMT, ...) FILE_BASENAME ":" STRINGIZE(__LINE__) ":" FMT, ##__VA_ARGS__
#define HV3_LOG(PRIO, FMT, ...)            huli_log(static_cast<huli_log_level_t>(PRIO), FMT, ##__VA_ARGS__)

// huli_log_impl has __attribute__((format(printf,...))) which is useful for
// catching format bugs, but _errlog_ needs to be usable as an expression (not
// a statement) so it cannot use the huli_log do-while macro.  This wrapper
// calls huli_log_impl directly and can appear in a comma-expression context.
//
// The pragma suppresses -Wformat-nonliteral/-Wformat-security because the
// format string is assembled by MAKE_LOG_FMT_WITH_PREFIX via token-paste and
// FILE_BASENAME — which is a string literal on GCC/Clang (__FILE_NAME__) but
// not on MSVC (constexpr stripFilePath).  Format correctness is still enforced
// by huli_log_impl's format attribute at its definition site, and by
// FORMAT_TYPE_CHECK at each _errlog_ call site.
// LCOV_EXCL_START [SAFTYSWCCB-996]
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
#pragma clang diagnostic ignored "-Wformat-security"
#endif
template <typename... Args> inline void hv3_errlog_impl(const char *fmt, Args... args)
{
    huli_log_impl(HULI_ERROR, fmt, args...);
}
#if defined(__clang__)
#pragma clang diagnostic pop // -Wformat-nonliteral -Wformat-security
#endif
// LCOV_EXCL_STOP

#else // !ENABLE_QNNDSP_LOG && !HAVE_HULI_LOG — no logging backend (e.g. QNN OpPackage x86)

#define MAKE_LOG_FMT_WITH_PREFIX(FMT, ...) FILE_BASENAME ":" STRINGIZE(__LINE__) ":" FMT, ##__VA_ARGS__
#define HV3_LOG(PRIO, FMT, ...)            ((void)0)

// LCOV_EXCL_START [SAFTYSWCCB-996]
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
#pragma clang diagnostic ignored "-Wformat-security"
#endif
template <typename... Args> inline void hv3_errlog_impl(const char * /*fmt*/, Args... /*args*/) {}
#if defined(__clang__)
#pragma clang diagnostic pop // -Wformat-nonliteral -Wformat-security
#endif
// LCOV_EXCL_STOP

#endif // ENABLE_QNNDSP_LOG

// These are conditional, where the condition is set via compile flags.  Note that these are
// template functions so that we can exclude them from coverage using lcov commands.
//
// __attribute__((noinline)) prevents the compiler from expanding a unique copy of this
// template at every call site (each distinct Args... signature would otherwise produce
// separate inlined code, bloating the hot-path instruction stream on DSP targets).
// The calling macros (_logmsg_, _infolog_, etc.) already guard with log_condition(), so
// the inner check here is intentionally removed to avoid a redundant branch per call site.

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-security"
#pragma clang diagnostic ignored "-Wformat-nonliteral"
#endif
// LCOV_EXCL_START [SAFTYSWCCB-996]
template <typename... Types>
[[gnu::noinline]] void logmsgraw(const int prio, char const *fmt, [[maybe_unused]] Types... args)
{
#ifdef ENABLE_QNNDSP_LOG
    qnndsp_log(prio, fmt, args...);
#elif HAVE_HULI_LOG
    // log_condition() is the sole gate for hexnn log levels; call huli_log_impl
    // directly so huli's own compile-time (HULI_LOG_LEVEL) and runtime
    // (huli_log_runtime_level) filters do not add a second, independent gate
    // that would change behavior relative to the pre-HULI system.
    huli_log_impl(static_cast<huli_log_level_t>(prio), fmt, args...);
#else
    (void)prio;
    (void)fmt;
#endif
}
// LCOV_EXCL_STOP
#if defined(__clang__)
#pragma clang diagnostic pop // -Wformat-nonliteral -Wformat-security
#endif

// These macros are what are used in actual code, so that the line and filename macros will expand
// properly to show where the macro is invoked.

#define _rawlog_(FMT, ...)  HV3_LOG(NN_LOG_ERRORLVL, FMT, ##__VA_ARGS__)
#define _okaylog_(FMT, ...) HV3_LOG(NN_LOG_ERRORLVL, MAKE_LOG_FMT_WITH_PREFIX(FMT, ##__VA_ARGS__))
// _errlog_ must be usable as an expression (return errlog(...)) so it cannot
// use the huli_log do-while macro.  Use hv3_errlog_impl which is a plain
// function call and can appear in a comma-expression.
#define _errlog_(FMT, ...) hv3_errlog_impl(MAKE_LOG_FMT_WITH_PREFIX(":ERROR:" FMT, ##__VA_ARGS__))
#define errlog(...)        (_errlog_(__VA_ARGS__), GraphStatus::ErrorFatal)
// Guard the logmsgraw() call so arguments are not evaluated when the level is
// disabled at runtime (important under NN_LOG_DYNLVL where log_condition() is
// a runtime branch rather than a compile-time constant).
#define _logmsg_(PRIO, FMT, ...)                                                                                       \
    do {                                                                                                               \
        if (log_condition(PRIO)) logmsgraw(PRIO, MAKE_LOG_FMT_WITH_PREFIX(FMT, ##__VA_ARGS__));                        \
    } while (false)
#define _warnlog_(FMT, ...)                                                                                            \
    do {                                                                                                               \
        if (log_condition(NN_LOG_WARNLVL))                                                                             \
            logmsgraw(NN_LOG_WARNLVL, MAKE_LOG_FMT_WITH_PREFIX("WARNING:" FMT, ##__VA_ARGS__));                        \
    } while (false) // LCOV_EXCL_BR_LINE LCOV_EXCL_LINE [SAFTYSWCCB-996]
// Stat logging helpers: keeping the if-branch inside a single inline function
// (instead of expanding it at each macro call site) ensures the branch is
// counted once and waived once with LCOV_EXCL, leaving call-site coverage clean.
// In the ENABLE_QNNDSP_LOG path, HULI stat APIs are not available; use no-ops.
#if HAVE_HULI_LOG
template <typename V> inline void hv3_statlog_impl(int prio, const char *name, V value)
{
    // LCOV_EXCL_START [SAFTYSWCCB-996]
    if (log_condition(prio)) huli_log_stat_u64(name, static_cast<uint64_t>(value));
    // LCOV_EXCL_STOP
}
// LCOV_EXCL_START [SAFTYSWCCB-996]
inline void hv3_statslog_impl(int prio, const char *name, const char *value)
{
    if (log_condition(prio)) huli_log_stat_str(name, value);
}
// LCOV_EXCL_STOP
#else
template <typename V> inline void hv3_statlog_impl(int /*prio*/, const char * /*name*/, V /*value*/) {}
inline void hv3_statslog_impl(int /*prio*/, const char * /*name*/, const char * /*value*/) {}
#endif
#define _infolog_(FMT, ...)                                                                                            \
    do {                                                                                                               \
        if (log_condition(NN_LOG_INFOLVL)) logmsgraw(NN_LOG_INFOLVL, MAKE_LOG_FMT_WITH_PREFIX(FMT, ##__VA_ARGS__));    \
    } while (false) // LCOV_EXCL_BR_LINE LCOV_EXCL_LINE [SAFTYSWCCB-996]
#define _i_infolog_(FMT, ...)                                                                                          \
    do {                                                                                                               \
        if (log_condition(NN_LOG_INFOLVL_INTERNAL))                                                                    \
            logmsgraw(NN_LOG_INFOLVL_INTERNAL, MAKE_LOG_FMT_WITH_PREFIX(FMT, ##__VA_ARGS__));                          \
    } while (false) // LCOV_EXCL_BR_LINE LCOV_EXCL_LINE [SAFTYSWCCB-996]
#define _debuglog_(FMT, ...)                                                                                           \
    do {                                                                                                               \
        if (log_condition(NN_LOG_DEBUGLVL)) /* LCOV_EXCL_BR_LINE SAFTYSWCCB-996 */                                     \
            logmsgraw(NN_LOG_DEBUGLVL, MAKE_LOG_FMT_WITH_PREFIX(FMT, ##__VA_ARGS__));                                  \
    } while (false) // LCOV_EXCL_BR_LINE LCOV_EXCL_LINE [SAFTYSWCCB-996]
#define _verboselog_(FMT, ...)                                                                                         \
    do {                                                                                                               \
        if (log_condition(NN_LOG_VERBOSELVL))                                                                          \
            logmsgraw(NN_LOG_VERBOSELVL, MAKE_LOG_FMT_WITH_PREFIX(FMT, ##__VA_ARGS__));                                \
    } while (false) // LCOV_EXCL_BR_LINE LCOV_EXCL_LINE [SAFTYSWCCB-996]

template <class T> constexpr const char *format_type_check = "";

// This compile-time expression ensures that we always apply the -Wformat type-safety check.
#define FORMAT_TYPE_CHECK(...) (format_type_check<decltype(printf(__VA_ARGS__))>)

#define rawlog(...)        _rawlog_(__VA_ARGS__)
#define okaylog(...)       _okaylog_(__VA_ARGS__)
#define logmsg(PRIO, ...)  _logmsg_(PRIO, __VA_ARGS__)
#define warnlog(...)       _warnlog_(__VA_ARGS__)
#define statlog(sn, sv)    hv3_statlog_impl(NN_LOG_STATLVL, (sn), (sv))
#define i_statlog(sn, sv)  hv3_statlog_impl(NN_LOG_STATLVL_INTERNAL, (sn), (sv))
#define statslog(sn, sv)   hv3_statslog_impl(NN_LOG_STATLVL, (sn), (sv))
#define i_statslog(sn, sv) hv3_statslog_impl(NN_LOG_STATLVL_INTERNAL, (sn), (sv))
#define infolog(...)       _infolog_(__VA_ARGS__)
#define i_infolog(...)     _i_infolog_(__VA_ARGS__)
#define _debuglog(...)     _debuglog_(__VA_ARGS__)
#define verboselog(...)    _verboselog_(__VA_ARGS__)

// Extra hook for debuglog.  This allows files to redefine it in order to add extra compile-time
// hooks for removing it.
#define debuglog(...) _debuglog(__VA_ARGS__)

// Internal formatter: formats via fmtlib then routes through huli_log_impl.
// Defined in log.cc.
#if HAVE_FMT
void vlogmsg_fmt(int prio, fmt::string_view fmt, fmt::format_args args);

template <typename... T> inline void logmsg_fmt(const int prio, fmt::format_string<T...> fmt, T &&...args)
{
    // LCOV_EXCL_START [SAFTYSWCCB-996]
    if (log_condition(prio)) {
        vlogmsg_fmt(prio, fmt, fmt::make_format_args(args...));
    }
    // LCOV_EXCL_STOP
}

#define errlogf(...)     logmsg_fmt(NN_LOG_ERRORLVL, __VA_ARGS__)
#define warnlogf(...)    logmsg_fmt(NN_LOG_WARNLVL, __VA_ARGS__)
#define infologf(...)    logmsg_fmt(NN_LOG_INFOLVL, __VA_ARGS__)
#define verboselogf(...) logmsg_fmt(NN_LOG_VERBOSELVL, __VA_ARGS__)
#define debuglogf(...)   logmsg_fmt(NN_LOG_DEBUGLVL, __VA_ARGS__)
#endif // HAVE_FMT

#ifdef NN_LOG_MAXLVL
#define LOG_STAT()    ((NN_LOG_MAXLVL) >= NN_LOG_STATLVL)
#define LOG_INFO()    ((NN_LOG_MAXLVL) >= NN_LOG_INFOLVL)
#define LOG_DEBUG()   ((NN_LOG_MAXLVL) >= NN_LOG_DEBUGLVL)
#define LOG_VERBOSE() ((NN_LOG_MAXLVL) >= NN_LOG_VERBOSELVL)
#else
#define LOG_STAT()    (1)
#define LOG_INFO()    (1)
#define LOG_DEBUG()   (1)
#define LOG_VERBOSE() (1)
#endif //#ifdef NN_LOG_MAXLVL

class ExternalProgressLogger {

  public:
    static void start(const char *stage_name);

    static void update_progress(unsigned int numerator, unsigned int denominator);

    static void end(const char *stage_name, const char *duration);
};

class ExternalTimePoint {
    using TimePoint = std::chrono::high_resolution_clock::time_point;
    const std::string stage_name;
    const TimePoint start_time;
    unsigned int numerator = 1;
    unsigned int denominator = 1;
    bool done = false;

  public:
    explicit ExternalTimePoint(const std::string &&stage_name);

    void update_progress(unsigned int new_numerator, unsigned int new_denominator);

    std::pair<std::string, uint64_t> close();

    // Custom destructor
    ExternalTimePoint() = delete;
    ExternalTimePoint(const ExternalTimePoint &) = delete;
    ExternalTimePoint &operator=(ExternalTimePoint &t) = delete;
    ExternalTimePoint(ExternalTimePoint &&) = delete;
    ExternalTimePoint &operator=(ExternalTimePoint &&t) = delete;
    ~ExternalTimePoint() { close(); } // LCOV_EXCL_LINE [SAFTYSWCCB-1542]
};

#if defined(__clang__)
#pragma clang diagnostic pop // -Wgnu-zero-variadic-macro-arguments
#endif

#endif //#ifndef LOG_H
