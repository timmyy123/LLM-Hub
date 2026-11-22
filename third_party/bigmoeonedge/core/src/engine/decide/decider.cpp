#include "decider.h"

#include "choice_scorer.h"
#include "prompt_split.h"

#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace bmoe::detail {

DecideResult run_decide(IDecideBackend & backend, IPrefixCache * cache, const DecideRequest & req) {
    DecideResult r;
    // Emptied first, before anything can refuse: the caller has already dropped whatever conversation
    // the sequence held, and a refused request must not leave its tokens behind for the next
    // generate() to build on at positions it no longer knows about.
    backend.clear();
    auto refuse = [&](RunError kind, std::string msg) {
        r.error = std::move(msg);
        r.error_kind = kind;
        r.prefix_state_bytes = cache ? cache->bytes() : 0;
        return r;
    };

    // Everything that can refuse the request runs before any token is prefilled.
    std::vector<std::vector<Token>> choice_tokens;
    choice_tokens.reserve(req.choices.size());
    for (const std::string & c : req.choices)
        choice_tokens.push_back(backend.tokenize_plain(c));
    std::vector<Token> choice_ids;
    std::string error;
    if (!choice_first_tokens(req.choices, choice_tokens, backend.n_vocab(), choice_ids, error))
        return refuse(RunError::InvalidRequest, error);

    std::vector<Token> tokens;
    if (!backend.render(req.prefix + req.suffix, tokens, error)) return refuse(RunError::TemplateRejected, error);
    const int n = (int) tokens.size();
    if (n < 1) return refuse(RunError::EmptyPrompt, "empty prompt after tokenization");
    if (n > backend.n_ctx())
        return refuse(RunError::ContextFull, "the prompt (" + std::to_string(n) +
                                                 " tokens) is longer than {ctx-size} (" +
                                                 std::to_string(backend.n_ctx()) + " tokens)");
    r.n_tokens = n;

    const bool reuse = cache && req.reuse_prefix && !req.prefix.empty();
    int n_split = 0;
    if (reuse) {
        std::vector<Token> prefix_alone;
        if (!backend.render(req.prefix, prefix_alone, error)) return refuse(RunError::TemplateRejected, error);
        n_split = prefix_split(tokens, prefix_alone);
    }

    int n_restored = 0;
    if (reuse && n_split > 0) {
        const auto t0 = std::chrono::steady_clock::now();
        n_restored = cache->restore(backend, tokens, n_split);
        r.restore_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    auto fail = [&](const char * what) {
        r.error = what;
        r.fatal = true;
        r.error_kind = RunError::Failed;
        backend.clear();
        r.prefix_state_bytes = cache ? cache->bytes() : 0;
        return r;
    };
    // The prefill is measured in two brackets around the prefix snapshot, so prefill_s reads exactly
    // as BMOE_DONE's and the copy of the state is reported on its own, as restore_seconds is.
    auto measured_prefill = [&](int from, int to, PrefillStats & st) {
        const auto t0 = std::chrono::steady_clock::now();
        backend.begin_prefill_measure();
        const bool ok = backend.prefill(tokens, from, to);
        backend.end_prefill_measure(st);
        st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return ok;
    };
    PrefillStats head;
    if (n_restored < n_split) {
        if (!measured_prefill(n_restored, n_split, head)) return fail("prefix prefill failed");
        const auto t0 = std::chrono::steady_clock::now();
        cache->store(backend, tokens, n_split);
        r.store_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    PrefillStats tail;
    if (!measured_prefill(n_split, n, tail)) return fail("prefill decode failed");
    r.prefill.seconds = head.seconds + tail.seconds;
    r.prefill.cpu_seconds = head.cpu_seconds + tail.cpu_seconds;
    r.prefill.read_mib = head.read_mib + tail.read_mib;
    r.prefill.io_seconds = head.io_seconds + tail.io_seconds;
    r.prefill.stall_seconds = head.stall_seconds + tail.stall_seconds;
    r.prefill.mgmt_seconds = head.mgmt_seconds + tail.mgmt_seconds;
    r.prefill.device_tokens = head.device_tokens + tail.device_tokens;
    r.prefill.device_read_mib = head.device_read_mib + tail.device_read_mib;
    r.prefill.device_cache_mib = head.device_cache_mib + tail.device_cache_mib;
    r.prefill.device_stall_seconds = head.device_stall_seconds + tail.device_stall_seconds;
    r.prefill.device_routed = head.device_routed + tail.device_routed;
    r.prefill.device_demand = head.device_demand + tail.device_demand;
    r.n_reused = n_restored;
    r.n_prefilled = n - n_restored;

    const float * logits = backend.last_logits();
    if (!logits) return fail("no logits after the prompt");
    score_choices(logits, backend.n_vocab(), choice_ids, r.choice_logp, r.best);

    backend.clear();
    r.prefix_state_bytes = cache ? cache->bytes() : 0;
    r.ok = true;
    return r;
}

} // namespace bmoe::detail
