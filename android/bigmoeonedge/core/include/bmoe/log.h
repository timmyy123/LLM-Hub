// Where the engine's diagnostics go: the log port.
//
// The engine says things a person may want to read — a fallback taken ("prefill stays on the
// CPU"), a budget it settled on, a policy it disarmed — and none of them is data. Those lines go
// through here, so a front-end can put them where its user looks: a terminal by default, the
// server's event stream, an app's log view. Machine-readable output never does: the CLI's
// `BMOE_*` protocol lines (docs/telemetry.md) and the CSV/trace sinks are written by their own
// owners and are not diagnostics.
//
// The default sink writes `bmoe: <message>\n` to stderr, byte for byte what the engine printed
// before the port existed, so a consumer reading stderr sees no change unless a front-end installs
// a sink of its own.
//
// Thread-safe: the engine logs from its I/O lanes and compute threads as well as the caller's.
// A sink is called with an internal lock held, one line at a time, so lines never interleave and a
// sink replaced by set_log_sink() is never called again once that call returns. A sink must not
// log or call set_log_sink() itself.
//
// Pure policy (no llama.cpp types); llama.cpp's own log is llama_log_set()'s business.
#pragma once

#include <cstdarg>

namespace bmoe {

enum class LogLevel {
    Info,  // what the engine settled on, and why (a budget, a placement, a policy armed)
    Warn,  // a fallback or a configuration worth a second look; the run goes on
    Error, // why something failed; the failure itself is still reported through its own result
};

// `message` has no "bmoe: " prefix and no trailing newline; `user` is set_log_sink()'s.
using LogSink = void (*)(LogLevel level, const char * message, void * user);

// Install `sink` (null restores the default stderr sink). Process-wide, like the engine's backend.
void set_log_sink(LogSink sink, void * user = nullptr);

// Drop messages below `min` (default Info: everything, as before the port).
void set_log_level(LogLevel min);

// Format and deliver one diagnostic line. Cheap when filtered out; diagnostics are rare, so the
// formatting itself is not on any hot path.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
void log_printf(LogLevel level, const char * fmt, ...);
void log_vprintf(LogLevel level, const char * fmt, va_list ap);

} // namespace bmoe
