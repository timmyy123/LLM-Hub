#include "llama_log.h"

#include "llama.h"

#include <atomic>
#include <mutex>

namespace bmoe::detail {

namespace {

std::atomic<bool> g_debug{false};
std::atomic<int> g_cancelling{0};
// What was installed before the bridge, to forward to. Written once, under call_once, before the
// bridge is installed; read only by the bridge.
ggml_log_callback g_prev = nullptr;
void * g_prev_user = nullptr;

void bridge(ggml_log_level level, const char * text, void *) {
    thread_local LlamaLogFilter filter;
    if (!filter.pass(level, g_debug.load(std::memory_order_relaxed), g_cancelling.load(std::memory_order_relaxed) > 0))
        return;
    if (g_prev) g_prev(level, text, g_prev_user);
}

} // namespace

void install_llama_log() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_get(&g_prev, &g_prev_user);
        llama_log_set(&bridge, nullptr);
    });
}

void set_llama_log_debug(bool on) {
    g_debug.store(on, std::memory_order_relaxed);
}

void llama_log_cancel_begin() {
    g_cancelling.fetch_add(1, std::memory_order_relaxed);
}

void llama_log_cancel_end() {
    g_cancelling.fetch_sub(1, std::memory_order_relaxed);
}

} // namespace bmoe::detail
