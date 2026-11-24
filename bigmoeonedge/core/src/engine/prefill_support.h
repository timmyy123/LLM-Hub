#pragma once

// How the engine feeds a prompt to llama.cpp and measures what that cost. Shared by every Session
// entry point that prefills (generate, perplexity, decide), so they fill batches and attribute
// prefill the same way. Internal header: it includes llama.h.

#include "bmoe/expert_source.h"
#include "../io/platform_io.h"

#include "llama.h"

namespace bmoe::detail {

// Fill an explicitly-allocated batch with `n` tokens at consecutive positions on sequence 0.
//
// The engine otherwise decodes through llama_batch_get_one, which leaves pos/seq_id/logits null and
// lets llama.cpp infer them. Speculation cannot: the driver reads the batch's sequence ids, and a
// verify pass needs logits at EVERY position, not just the last. So every batch on the speculative
// path is spelled out — including prefill, which the driver must see to keep the draft context's
// KV in step with the target's.
inline void batch_fill(llama_batch & b, const llama_token * toks, int n, llama_pos pos0, bool all_logits) {
    b.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        b.token[i] = toks[i];
        b.pos[i] = pos0 + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = (int8_t) (all_logits || i == n - 1);
    }
}

// The prompt phase's measurement, the prefill counterpart of generate()'s per-token tally: the
// source's counters are cumulative across a warm session, so begin() pins them (with the process CPU
// clock) just above the prompt chunk loop and end() closes the deltas just after it — the same
// wall-additive terms the decode fields report, read with the same rules. end()'s sample is the
// phase boundary itself: the decode baseline seeds its cursors from it, so prefill's end and
// decode's start are one reading of the counters, not two that could drift apart.
struct PrefillTally {
    IExpertSource::Stats pre;
    double cpu0 = 0.0;

    // Deltas across this turn's prefill chunks — valid after end().
    double cpu_seconds = 0.0;
    double read_mib = 0.0;
    double io_seconds = 0.0;
    double stall_seconds = 0.0;
    double mgmt_seconds = 0.0;

    // The stats sample end() closed on, for the decode baseline to start from.
    IExpertSource::Stats post;

    void begin(bool moe_on, const IExpertSource & src) {
        pre = moe_on ? src.stats() : IExpertSource::Stats{};
        cpu0 = pio::process_cpu_seconds();
    }
    void end(bool moe_on, const IExpertSource & src) {
        post = moe_on ? src.stats() : IExpertSource::Stats{};
        cpu_seconds = pio::process_cpu_seconds() - cpu0;
        read_mib = (double) ((long long) post.read_bytes - (long long) pre.read_bytes) / (1024.0 * 1024.0);
        io_seconds = post.read_seconds - pre.read_seconds;
        stall_seconds = post.stall_seconds - pre.stall_seconds;
        mgmt_seconds = post.mgmt_seconds - pre.mgmt_seconds;
    }
};

} // namespace bmoe::detail
