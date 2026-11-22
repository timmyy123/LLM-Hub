#pragma once
// What of llama.cpp's and ggml's own log reaches the process's stderr.
//
// Left alone, both write every line they log, DEBUG included: about 1200 lines of graph-input
// bookkeeping per model load, one load per isolated probe child, so an `--auto` run printed ~9700
// and the engine's own lines were lost among them (#287). The bridge installed here keeps
// llama.cpp's output what it was for everything else - the same text, to wherever the callback it
// replaces wrote it - and drops DEBUG unless RunConfig::llama_debug_log asks for it.
//
// It also knows one thing llama.cpp cannot: when a decode fails because the ENGINE asked it to
// stop. A cancel reaches llama.cpp through the abort callback as GGML_STATUS_ABORTED, which it
// reports as three ERROR lines ("graph_compute: ... failed with error 1", "process_ubatch: failed
// to compute graph", "llama_decode: failed to decode"). While such an abort is in flight, ERROR
// lines are demoted to DEBUG. The window opens only when the abort callback answers yes because
// of a cancel (never for a fatal streaming error, which is a real failure) and closes when the
// call that was cancelled returns, so what it can hide is what that cancelled decode logged.
//
// llama.cpp's public API only: llama_log_get / llama_log_set (which sets ggml's logger too).
#include "ggml.h"

namespace bmoe::detail {

// The line-by-line decision, apart from the process-wide state so it can be tested. One per
// thread: a CONT line continues the message the same thread logged last, so it follows that
// message's fate - shown if it was shown, dropped if it was dropped.
class LlamaLogFilter {
public:
    bool pass(ggml_log_level level, bool debug, bool cancelling) {
        if (level == GGML_LOG_LEVEL_CONT) return last_;
        const bool demoted = cancelling && level == GGML_LOG_LEVEL_ERROR;
        last_ = debug || (level != GGML_LOG_LEVEL_DEBUG && !demoted);
        return last_;
    }

private:
    bool last_ = true;
};

// Install the bridge, once per process; later calls do nothing. The callback it replaces (the
// default stderr writer, or an embedder's own) receives every line that passes.
void install_llama_log();

// Show llama.cpp's DEBUG lines (RunConfig::llama_debug_log). Process-wide, like the log itself.
void set_llama_log_debug(bool on);

// Bracket an abort the engine requested. Counted, so two sessions cancelling at once each close
// only their own window.
void llama_log_cancel_begin();
void llama_log_cancel_end();

} // namespace bmoe::detail
