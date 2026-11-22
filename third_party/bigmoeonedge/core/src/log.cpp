#include "bmoe/log.h"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>

namespace bmoe {

namespace {

// What the engine printed before the port: one fprintf per line, so the bytes are identical.
void stderr_sink(LogLevel, const char * message, void *) {
    std::fprintf(stderr, "bmoe: %s\n", message);
}

// Function-local statics: a diagnostic logged during another translation unit's static
// initialisation still finds them constructed.
std::mutex & sink_mutex() {
    static std::mutex m;
    return m;
}
LogSink g_sink = &stderr_sink; // guarded by sink_mutex()
void * g_user = nullptr;       // guarded by sink_mutex()
std::atomic<int> g_min{(int) LogLevel::Info};

// Long enough for every line the engine writes; a longer one is formatted again on the heap.
constexpr size_t k_line_stack = 512;

} // namespace

void set_log_sink(LogSink sink, void * user) {
    std::lock_guard<std::mutex> lk(sink_mutex());
    g_sink = sink ? sink : &stderr_sink;
    g_user = sink ? user : nullptr;
}

void set_log_level(LogLevel min) {
    g_min.store((int) min, std::memory_order_relaxed);
}

void log_vprintf(LogLevel level, const char * fmt, va_list ap) {
    if ((int) level < g_min.load(std::memory_order_relaxed)) return;
    char stack[k_line_stack];
    va_list again;
    va_copy(again, ap);
    const int n = std::vsnprintf(stack, sizeof(stack), fmt, ap);
    const char * line = stack;
    std::string heap;
    if (n >= (int) sizeof(stack)) {
        heap.resize((size_t) n + 1);
        std::vsnprintf(&heap[0], heap.size(), fmt, again);
        heap.resize((size_t) n);
        line = heap.c_str();
    } else if (n < 0) {
        line = fmt; // an encoding error: say something rather than nothing
    }
    va_end(again);
    std::lock_guard<std::mutex> lk(sink_mutex());
    g_sink(level, line, g_user);
}

void log_printf(LogLevel level, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    log_vprintf(level, fmt, ap);
    va_end(ap);
}

} // namespace bmoe
