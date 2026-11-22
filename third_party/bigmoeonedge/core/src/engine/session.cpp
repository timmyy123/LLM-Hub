#include "bmoe/session.h"
#include "bmoe/expert_layout.h"
#include "bmoe/route_trace.h"
#include "bmoe/decode_trace.h"
#include "bmoe/version.h"
#include "bmoe/probe.h"
#include "bmoe/ngram_draft.h"
#include "bmoe/log.h"
#include "chat_messages.h"
#include "chat_parse.h"
#include "decide/decider.h"
#include "decide/llama_backend.h"
#include "logits.h"
#include "prefill_support.h"
#include "stop_text.h"
#include "thinking_control.h"
#include "prefill_path.h"
#include "llama_log.h"
#include "../moe/router_hook.h"
#include "../moe/decide_probe.h"
#include "../moe/gate_gemv.h"
#include "../moe/expert_stream_source.h"
#include "../moe/gguf_offsets.h"
#include "../moe/tensor_names.h"
#include "cache_brake.h"
#include "context_checkpoints.h"
#include "kv_cache_type.h"
#include "media_input.h"
#include "prompt_stream.h"
#include "knob_trial.h"
#include "device_answer.h"
#include "logit_agreement.h"
#include "../io/platform_io.h"
#include "../io/mapping_release.h"
#include "../io/parallel_copy.h"
#include "../metrics/memory_snapshot.h"

#include "ggml-backend.h"
#include "llama.h"
#include "ggml.h"

// llama.cpp's `common` layer (NOT the stable public API): chat-template rendering and
// reasoning parsing. See the note in the root CMakeLists / docs/seam.md.
#include "chat.h"
#include "common.h"
#include "speculative.h"

#include <algorithm>
#include <limits>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bmoe {

namespace {

using detail::batch_fill;
using detail::PrefillTally;

// The speculative driver reads llama.cpp's own mirror of a batch's entries (common_batch), not the
// llama_batch the engine decodes. It only reads entries (token, position, sequence, output), so a
// mirror without a context is enough: nothing here asks it to render a batch for a decode.
common_batch to_common_batch(const llama_batch & b) {
    common_batch out;
    out.tokens.reserve((size_t) b.n_tokens);
    for (int i = 0; i < b.n_tokens; ++i) {
        const int idx = out.add(b.token[i], b.pos[i], b.seq_id[i][0], b.logits[i] != 0);
        for (int s = 1; s < b.n_seq_id[i]; ++s)
            out.add_seq(idx, b.seq_id[i][s]);
    }
    return out;
}

using clock_t_ = std::chrono::steady_clock;

// Tokens kept free in n_ctx past the answer's budget: the end-of-turn tokens a chat template
// appends after the answer must still fit, or the next turn's prompt would not.
constexpr int k_turn_end_reserve = 8;

// The tokenizer's first guess at a prompt's length: its byte count plus the special tokens
// add_special may put around it. A longer result is reported, and the buffer resized to it.
constexpr int k_tokenize_slack = 8;

double secs(clock_t_::time_point a, clock_t_::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// Graph width for the MTP draft context, and it wants to be SMALL.
//
// llama.cpp reserves a context's compute buffers for its widest ubatch, and the dominant term is the
// output buffer, which scales with ubatch x vocabulary — at 256 on a 152k-vocab model that alone is
// ~156 MB, inside a reservation measured at 493 MiB on device. The draft context never needs it:
// at decode time it evaluates ONE token per draft step, the catch-up hands it at most 1 + draft_max
// positions, and that catch-up asks for no logits at all. Only prefill ever feeds it a wide batch,
// and that is one layer, so splitting it into more ubatches costs very little.
//
// The width was 256 until the device A/B showed what it cost. Half a gigabyte of reservation is what
// tipped the phone past its memory budget: major faults per token went from ~69 without speculation
// to 632 at draft 3, as the kernel swapped and dropped file pages to find the room. On this engine
// memory is never free — it is the expert cache's, and the cache is what decides whether the wider
// verify read set is a hit or a flash read.
//
// Raised to 1 + draft_max when a very wide draft asks for more, and clamped down to the target's own
// ubatch so it is never the wider of the two.

// The chosen token's log-probability at a position and the `n` most likely tokens there, most likely
// first, from the raw logits: the distribution before any sampling stage, as llama.cpp's server
// reports it.
struct PositionLogprobs {
    float chosen = 0.0f;
    std::vector<TokenLogprob> top;
};
PositionLogprobs
position_logprobs(const float * logits, int n_vocab, const llama_vocab * vocab, llama_token chosen, int n) {
    PositionLogprobs out;
    float max_l = -INFINITY;
    for (int v = 0; v < n_vocab; ++v)
        max_l = std::max(max_l, logits[v]);
    double sum = 0.0;
    for (int v = 0; v < n_vocab; ++v)
        sum += std::exp((double) logits[v] - max_l);
    const float log_z = max_l + (float) std::log(sum);
    if (chosen >= 0 && chosen < n_vocab) out.chosen = logits[chosen] - log_z;
    std::vector<int32_t> ids((size_t) n_vocab);
    for (int v = 0; v < n_vocab; ++v)
        ids[(size_t) v] = v;
    const size_t k = (size_t) std::min(std::max(n, 0), n_vocab);
    std::partial_sort(ids.begin(), ids.begin() + (ptrdiff_t) k, ids.end(),
                      [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    for (size_t i = 0; i < k; ++i)
        out.top.push_back({ids[i], common_token_to_piece(vocab, ids[i], true), logits[ids[i]] - log_z});
    return out;
}

llama_token argmax(const float * logits, int n_vocab) {
    llama_token best = 0;
    float best_v = logits[0];
    for (int v = 1; v < n_vocab; ++v)
        if (logits[v] > best_v) {
            best_v = logits[v];
            best = v;
        }
    return best;
}

// The generation phase's running measurement state, and the one place a generated token's cost is
// written down. The cursors hold the source's absolute totals as of the previous token — its stats
// are cumulative across a warm session, so every per-token flash figure is a delta against these —
// and the totals are what the summary averages over n_gen. Grouped into one object so the per-token
// metrics block can be a method instead of ten more locals threaded through generate().
struct GenTally {
    // Fixed for the run; kept here so record() needs only the token's own measurements.
    bool overlap = false;

    long long prev_bytes = 0;
    double prev_io_s = 0.0;
    double prev_mgmt_s = 0.0;
    double prev_stall_s = 0.0;

    uint64_t read_bytes = 0;
    double io_seconds = 0.0;
    double mgmt_seconds = 0.0;
    double stall_seconds = 0.0;
    double drain_seconds = 0.0;
    double adopt_seconds = 0.0;
    double prev_drain_s = 0.0;
    double prev_adopt_s = 0.0;
    uint64_t majflt = 0;
    double cpu_seconds = 0.0;

    // Fill in everything a generated token is measured by — its wall/fault/CPU decomposition, the
    // memory picture, and (when streaming) the flash figures taken as deltas against the cursors
    // above — then advance those cursors and the run totals. The token's TEXT stays with the
    // caller: what a token says depends on chat state, what it cost does not.
    //
    // `wall` is the decode's wall time in seconds and `faults`/`cpu_s` the deltas measured around
    // that same decode; `st` is the expert source's stats, or null when streaming is off.
    void
    record(TokenMetrics & m, double wall, uint64_t faults, double cpu_s, int turn, const IExpertSource::Stats * st) {
        m.wall_ms = wall * 1000.0;
        // Fault/CPU decomposition is independent of streaming — dense-weight faults show up in the
        // mmap baseline too — so record it for every token before the moe/no-moe split below.
        m.majflt = faults;
        m.cpu_ms = cpu_s * 1000.0;
        m.majflt_mib = (double) m.majflt * (double) pio::fault_bytes() / (1024.0 * 1024.0);
        m.turn = turn;
        majflt += m.majflt;
        cpu_seconds += cpu_s;

        // Read the memory picture AFTER the decode, outside the caller's timing bracket: two /proc
        // reads are cheap but they are not this token's work, and billing them to wall_ms would
        // corrupt the very number the reader is here to trust.
        pio::ProcessMemory pm;
        if (pio::process_memory(&pm)) detail::record_process_memory(m, pm);
        pio::DeviceMemory dm;
        if (pio::device_memory(&dm)) {
            m.mem_available_mib = dm.available_bytes / (1024.0 * 1024.0);
            m.mem_free_mib = dm.free_bytes / (1024.0 * 1024.0);
            m.swap_free_mib = dm.swap_free_bytes / (1024.0 * 1024.0);
        }
        if (!st) {
            m.compute_ms = m.wall_ms;
            m.cache_hit_pct = -1.0;
            return;
        }

        m.dense_resident_frac = st->dense_resident_frac;
        m.cache_budget_mib = st->cache_budget_bytes / (1024.0 * 1024.0);
        m.read_bytes = (uint64_t) ((long long) st->read_bytes - prev_bytes);
        m.io_ms = (st->read_seconds - prev_io_s) * 1000.0;
        m.mgmt_ms = (st->mgmt_seconds - prev_mgmt_s) * 1000.0;
        if (overlap) {
            // stall is already wall-additive (the union of stalled intervals), so no thread-count
            // normalization: dividing summed thread time by n_threads was a mean that understated
            // the stall whenever a minority of threads did the waiting, and the difference quietly
            // became "compute".
            m.stall_ms = (st->stall_seconds - prev_stall_s) * 1000.0;
            m.flash_wait_ms = m.stall_ms;
        } else {
            m.flash_wait_ms = m.io_ms;
        }
        m.compute_ms = m.wall_ms - m.flash_wait_ms - m.mgmt_ms;
        if (m.compute_ms < 0) m.compute_ms = 0;
        m.cache_hit_pct = st->cache_lookups > 0 ? 100.0 * st->cache_hits / st->cache_lookups : -1.0;
        m.drain_ms = (st->drain_wait_seconds - prev_drain_s) * 1000.0;
        m.adopt_ms = (st->adopt_wait_seconds - prev_adopt_s) * 1000.0;

        prev_bytes = (long long) st->read_bytes;
        prev_io_s = st->read_seconds;
        prev_mgmt_s = st->mgmt_seconds;
        prev_stall_s = st->stall_seconds;
        prev_drain_s = st->drain_wait_seconds;
        prev_adopt_s = st->adopt_wait_seconds;
        read_bytes += m.read_bytes;
        io_seconds += m.io_ms / 1000.0;
        mgmt_seconds += m.mgmt_ms / 1000.0;
        stall_seconds += m.stall_ms / 1000.0;
        drain_seconds += m.drain_ms / 1000.0;
        adopt_seconds += m.adopt_ms / 1000.0;
    }
};

// Whether the load placed `t` in a buffer that belongs to a device other than the host's. Asked as
// "names a non-CPU device" rather than "names the CPU" because the buffer type over the file
// mapping names no device at all in this llama.cpp (a FIXME upstream); that is the host's buffer,
// and it stays ours whether or not the field is ever filled in.
bool on_other_device(const ggml_tensor * t) {
    if (!t->buffer) return false;
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
    return dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
}

// The names of the expert weight tensors the streamer rebinds. "Dense" is defined by subtraction —
// everything the model has that is NOT one of these — so both consumers below start by asking this
// same question, and used to answer it with their own copy of the same triple loop.
std::unordered_set<std::string> expert_tensor_names(const std::vector<LayerExperts> & layers) {
    std::unordered_set<std::string> names;
    for (const LayerExperts & L : layers) {
        if (!L.bound) continue;
        for (int p = 0; p < ExpertLayout::max_slots; ++p)
            if (L.proj[p].tensor) names.insert(L.proj[p].tensor->name);
    }
    return names;
}

// Each layer's bytes that the streamer does NOT manage: everything under blk.<il>. except the
// expert weight tensors it rebinds — attention, norms, the router, and any per-expert scale left
// mmap-resident. This is what the layer costs to page in, and it is a static property of the
// file: nothing about decoding changes it, which is why the route trace states it once in the
// static block instead of pretending to measure it per step.
std::vector<uint64_t>
dense_bytes_per_layer(const GgufOffsets & offs, const std::vector<LayerExperts> & layers, int n_layer) {
    const std::unordered_set<std::string> streamed = expert_tensor_names(layers);
    std::vector<uint64_t> out((size_t) std::max(0, n_layer), 0);
    for (const auto & kv : offs.size_by_name) {
        const int il = tensor_names::block_layer(kv.first.c_str());
        if (il < 0 || il >= n_layer || streamed.count(kv.first)) continue;
        out[(size_t) il] += kv.second;
    }
    return out;
}

// Width of one prefill piece under a prefill device: one ubatch, so a decode call is one graph.
int prefill_piece(const SessionConfig & cfg) {
    return cfg.n_ubatch > 0 ? std::min(cfg.n_ubatch, cfg.n_batch) : cfg.n_batch;
}

} // namespace

// All native state lives here, behind the pimpl. Built once by Session::open(); every
// generate() reuses the loaded model, the live context and the warm expert cache.
namespace {
// A run that does not stream keeps its routing only where it was asked to, and only for a MoE model.
bool records_unstreamed(const SessionConfig & cfg) {
    return !cfg.moe.enabled && cfg.moe.record_unstreamed && cfg.moe.routing_record_tokens > 0;
}

} // namespace

struct Session::Impl {
    SessionConfig cfg;
    std::string arch;
    double load_seconds = 0.0;

    // Ownership order matters at teardown: the source's I/O pool holds fds into the mmap'd
    // file and its buffers back the rebound expert tensors, so it must be shut down before
    // the context and model are freed. The destructor does that explicitly.
    std::unique_ptr<llama_model, void (*)(llama_model *)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, void (*)(llama_context *)> ctx{nullptr, llama_free};
    std::unique_ptr<RouterHook> hook; // heap: its address is baked into cparams.cb_eval_user_data
    // The experts' copy in a device backend's host memory (MoeStreamConfig::expert_host_buffer):
    // the expert tensors point into it, so it outlives the contexts and the model.
    std::unique_ptr<ggml_backend_buffer, void (*)(ggml_backend_buffer_t)> expert_host_buf{nullptr,
                                                                                          ggml_backend_buffer_free};
    // The same copy where the device's backend pinned memory this session allocated (a registered
    // host buffer, stage_experts_in_host_buffer) instead of allocating its own: unpinned and freed
    // with it, after the model.
    struct RegisteredHostMemory {
        void * p = nullptr;
        size_t size = 0;
        void (*unregister)(void *) = nullptr;
        RegisteredHostMemory() = default;
        RegisteredHostMemory(const RegisteredHostMemory &) = delete;
        RegisteredHostMemory & operator=(const RegisteredHostMemory &) = delete;
        ~RegisteredHostMemory() {
            if (!p) return;
            if (unregister) unregister(p);
            pio::vm_release(p, size);
        }
    };
    std::unique_ptr<RegisteredHostMemory> expert_host_registered;
    ExpertStreamSource source;
    // Kept per session, not per turn: the reference it measures stalls against is this machine's,
    // and a later turn of a chat is compared with what the earlier ones settled to.
    CacheBrake brake;

    // The knob trial (SessionConfig::trial): the lossless knobs measured on the first decodes once
    // the cache has settled, one knob after the other, each winner kept for the rest of the session
    // and returned (Session::measured_knobs). The stages run in this order; after the knobs,
    // the decode rate of the device tier's side this load runs (KnobTrialConfig::device_tier), timed
    // with every knob at its winner so no switch is in its windows.
    static constexpr int stage_threads = 0, stage_overlap = 1, stage_rate = 2, stage_done = 3;
    struct Trials {
        std::vector<int> threads; // arms: the incumbent, then the candidates; empty: not measured
        std::vector<int> overlap; // arms as 0/1, the incumbent first; empty: not measured
        bool rate = false;        // the device tier's or the graph width's side is timed
        // A width trial's usage, over this load's turns: requests, tokens decoded, and the prompts longer
        // than the narrow width (tokens prefilled after reuse, and their seconds).
        int width_requests = 0;
        double width_gen = 0.0, width_long_tokens = 0.0, width_long_seconds = 0.0;
        // The request mix (MeasuredKnobs::mix_*), over every turn of this load whether a trial runs or
        // not: requests, tokens decoded, tokens of the prompts a device would take, and of those the
        // ones the host prefilled, with their seconds.
        int mix_requests = 0;
        double mix_gen = 0.0, mix_wide = 0.0, host_wide_tokens = 0.0, host_wide_seconds = 0.0;
        // What this load decoded over every turn, and the seconds it took: the rate of the side it
        // serves, once the probe has timed its windows (note_request_mix).
        double served_tokens = 0.0, served_seconds = 0.0;
        bool rate_timed = false; // the probe has timed its windows on this load
        bool served_rate_said = false;
        int stage = stage_done;
        std::unique_ptr<KnobTrial> cur;
        std::unique_ptr<RateProbe> rate_probe;
        // The decode's own cost (MeasuredKnobs::decode_compute_s), timed on every load once the trial
        // is done: its windows run with every knob at its winner, and it switches nothing.
        std::unique_ptr<RateProbe> cost_probe;
        MeasuredKnobs result;
        uint64_t lookups_seen = 0, hits_seen = 0; // the cache's counters at the last step
    } trials;
    int threads_now = 0; // the decode thread count the contexts run with
    void arm_trials();
    void start_trial_stage(bool settle);
    void next_trial_stage(bool settled);
    void rate_step(double seconds, int tokens, uint64_t lookups, uint64_t hits);
    void note_width_turn(int n_prompt, double prefill_seconds, int n_generated);
    void note_request_mix(const RunSummary & sum);
    void finish_trial_stage();
    bool apply_trial_arm(int arm);
    void set_decode_threads(int n);
    void trial_step(double seconds, int tokens);
    void cost_step(double compute_seconds, int tokens);
    void pause_trial();
    void resume_trial();
    void snapshot_trial_counters();

    // Prefill on an accelerator (SessionConfig::prefill); null when off. Created before the model
    // load, whose device list it holds, and opened after it.
    std::unique_ptr<detail::PrefillPath> prefill;
    bool prefill_check_refused = false; // its check on this model refused it (confirm_prefill_device)

    // Where the next graph actually goes: a device with no room right now leaves it on the host.
    bool place_prefill(bool on_device, bool hold_slots = false) {
        return prefill ? prefill->place(on_device, hold_slots) : false;
    }
    int decode_placed(llama_context * c, const llama_batch & b) {
        return prefill ? prefill->decode(c, b) : llama_decode(c, b);
    }
    int decode_placed_ext(llama_context * c, llama_batch_ext * b) {
        return prefill ? prefill->decode_ext(c, b) : llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b);
    }
    // Every exit from a prefill (errors and cancellation included) leaves the weights on the host:
    // the next graph may be a decode, and the CPU cannot read a device buffer.
    struct PlacementGuard {
        Impl & im;
        ~PlacementGuard() { im.place_prefill(false); }
    };
    // Every llama_memory_clear(data) of the target context goes through here: llama.cpp clears the
    // buffers it allocated, which a moved model state no longer lives in.
    void clear_memory(llama_context * c) {
        llama_memory_clear(llama_get_memory(c), true);
        if (prefill) prefill->clear_state();
    }

    // The MTP draft source (SpecConfig::source == mtp). A SECOND context over the SAME model,
    // created with ctx_type = MTP so llama.cpp builds the nextn graph instead of the trunk one. It
    // carries the same eval callback as the target — the hook is per-context, not per-model — so the
    // MTP block's expert layer is captured and streamed by the one source both contexts share.
    //
    // The n-gram source has no equivalent state: it drafts from the token history alone, which is
    // why it costs no context, no memory and no expert read. Everything below this pair is shared by
    // both sources, because the verify half of the loop does not care who drafted.
    // The MTP head when it comes in a file of its own (SpecConfig::mtp_head); null when the head is
    // the model's own nextn block. Declared before the draft context so it outlives it.
    std::unique_ptr<llama_model, void (*)(llama_model *)> model_dft{nullptr, llama_model_free};
    std::unique_ptr<llama_context, void (*)(llama_context *)> ctx_dft{nullptr, llama_free};
    common_speculative_ptr mtp;
    // Serves both roles on the speculative path, never both at once: a prefill chunk (up to
    // n_batch tokens, logits on the last) and a verify pass (1 + draft_max tokens, logits on all).
    llama_batch mtp_batch{};
    bool mtp_batch_owned = false; // whether mtp_batch holds a llama_batch_init allocation to free
    std::vector<llama_token> draft_buf;
    long long mtp_drafted = 0;
    long long mtp_accepted = 0;
    long long mtp_decodes = 0;
    // Steps that drafted anything at all. Against mtp_decodes this is the n-gram source's match
    // rate — the fraction of the run where it had evidence and widened the verify batch. For the
    // head it is all of them, since it drafts unconditionally unless p_min stops it.
    long long drafted_steps = 0;
    // Time spent drafting and catching the draft context up, i.e. everything speculation adds
    // OUTSIDE the target decode. It used to land in loop_overhead_ms together with sampling and
    // rendering, where it could only be inferred by differencing against an unspeculated run.
    double mtp_draft_seconds = 0.0;
    // Flash bytes those passes streamed. The MTP block is a MoE layer of its own, so drafting has
    // an I/O cost as well as a compute one — and it is invisible to the route trace, whose framing
    // brackets the target decode only. Without this the growth in bytes/token under speculation
    // cannot be split between the widened verify union on the trunk and the head's own routing,
    // which are attacked in completely different ways.
    uint64_t mtp_draft_read_bytes = 0;

    const llama_vocab * vocab = nullptr;
    int n_vocab = 0;
    int n_expert_used = 0; // effective routing width, after any override (0 = not MoE)
    int n_layer = 0;
    // Trained MTP heads in this gguf (0 = no nextn block). The MTP block occupies layer indices
    // [n_layer, n_layer + n_layer_nextn), contiguous with the trunk and using the same tensor
    // naming, so widening the streamer's layer bound by it is all the streamer needs to reach it.
    int n_layer_nextn = 0;
    bool chat_on = false;
    common_chat_templates_ptr chat_tmpls;
    // How a think=false request can be honoured on this model; probed once at open(). Template
    // (the fail-open default) means the flag alone does the job and generate() adds nothing.
    ThinkControl think_ctl = ThinkControl::Template;
    bool backend_inited = false;
    // What release_file_mappings left behind on purpose; released only after `model` is gone
    // (its teardown still unmaps/closes what it believes is its mapping; see mapping_release.h).
    pio::MappingPlaceholders mapping_placeholders;

    // Sampling chain, built once at open() only when sampling is requested (temp > 0); null on the
    // greedy default, where the decode loop stays on the argmax fast path. See open()/generate().
    llama_sampler * smpl = nullptr;

    // Multi-turn chat state (chat mode only). chat_history is the running conversation the
    // template is re-rendered over each turn; kv mirrors the tokens currently decoded
    // into the context's KV (seq 0), in order, so the next turn can reuse the common prefix
    // and prefill only the diverging suffix instead of re-running the whole conversation.
    std::vector<common_chat_msg> chat_history;
    // The stream, not a plain token list: an image in it is a span of placeholders over fewer
    // positions than tokens, and kv.pos_at() is where the context is cut (prompt_stream.h).
    detail::PromptStream kv;
    // The projector, when the session was opened with one: the prompt's images go through it.
    std::unique_ptr<detail::MediaInput> media;
    // A batch of the session's own for a prompt with images in it: its text runs are placed by
    // explicit positions, since the image before them took fewer than its tokens.
    llama_batch media_batch{};
    bool media_batch_owned = false;
    // Speculation drafts from the token list, and an image's tokens are not in the vocabulary: a
    // conversation that has held an image runs without a draft until it is cleared (said once).
    bool media_in_conversation = false;
    bool media_spec_logged = false;
    // A request's own sampling is not applied while speculation is armed (RequestSampling): said once.
    bool request_sampling_logged = false;
    // What a refused cut of the KV rolls back to (context_checkpoints.h). They describe a prefix of
    // kv, so they go wherever it goes.
    detail::ContextCheckpoints checkpoints;
    bool checkpoint_size_logged = false; // the size is the model's, so said once per session
    int checkpoint_turn = 0;             // which request took a checkpoint (ContextCheckpoints::create)
    detail::CheckpointContexts checkpoint_contexts() { return {ctx.get(), ctx_dft.get(), mtp.get()}; }
    void forget_kv_tokens() {
        kv.clear();
        checkpoints.clear();
        media_in_conversation = false;
    }
    // Keep the first n tokens of the KV and drop the rest. Where the memory refuses the cut, or a
    // sliding window has already let go of what the cut would read (memory_covers), the
    // newest checkpoint at or before n stands in for it; returns how many tokens are still held,
    // 0 with the context cleared when no checkpoint helps. A cut inside an image moves to its
    // start: an image is held whole or not at all.
    size_t keep_kv_prefix(size_t n) {
        n = std::min(n, kv.size());
        if (const detail::MediaSpan * m = kv.media_at(n); m && m->begin < n) n = m->begin;
        size_t kept = n;
        if (!checkpoints.memory_covers(checkpoint_contexts(), (int64_t) n) ||
            !llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, kv.pos_at(n), -1)) {
            kept = (size_t) checkpoints.restore(checkpoint_contexts(), (int64_t) n);
            if (kept == 0) clear_memory(ctx.get());
        }
        kv.truncate(kept);
        checkpoints.drop_after((int64_t) kept);
        return kept;
    }

    // decide()'s kept prefix state. Chosen at the first decide() (the policy needs the backend's
    // answer on prefill cost), null when the policy keeps none.
    std::unique_ptr<detail::IPrefixCache> decide_cache;
    bool decide_cache_chosen = false;
    // The experimental decide() probe, and how many decisions it has written.
    std::unique_ptr<DecideProbe> decide_probe;
    int decide_seq = 0;

    // Route trace (diagnostics): null unless requested AND streaming is on — there is no routing
    // to trace otherwise.
    IRouteTraceSink * route_trace = nullptr;

    // Which generate() this is, 0-based. It labels a trace's rows, and it labels every token's
    // metrics — a multi-turn CSV is unreadable without it, and the two-turn A/B (a fast turn, an
    // idle, then the turn that pays for it) is exactly what this engine is measured by.
    int turn = 0;

    // Decode traces (diagnostics): null unless requested. The compute trace needs no streaming —
    // it measures the graph, which a dense mmap run has too; the I/O trace needs the streamer,
    // since there are no engine-issued reads without it. See bmoe/decode_trace.h.
    IComputeTraceSink * compute_trace = nullptr;
    IIoTraceSink * io_trace = nullptr;
    std::vector<IoTraceRow> io_rows_scratch;

    // Stated once to a metrics sink, before the first token: the model and configuration every row
    // it writes was produced under. info_sent guards a session's many generate() calls from
    // interleaving preambles between turns.
    RunInfo info;
    bool info_sent = false;

    std::atomic<bool> cancel_requested{false};
    const std::atomic<bool> * caller_stop = nullptr; // borrowed only by the current generate()
    bool cancelled() const {
        return cancel_requested.load(std::memory_order_relaxed) ||
               (caller_stop && caller_stop->load(std::memory_order_relaxed));
    }
    // Set while a cancel's abort is in flight, so llama.cpp's report of the aborted decode is not
    // shown as an error (engine/llama_log.h). The abort callback sets it; end_cancel_quiet() clears.
    std::atomic<bool> cancel_quieted{false};
    // Close the window the abort callback opened, if it did, and say in the engine's own words
    // what llama.cpp's demoted lines would have reported as a failure.
    void end_cancel_quiet() {
        if (!cancel_quieted.exchange(false, std::memory_order_relaxed)) return;
        detail::llama_log_cancel_end();
        log_printf(LogLevel::Info, "cancelled: the decode in progress was stopped on request");
    }

    // ── open(), one named step at a time, in the order open() calls them (see Opening) ──
    struct Opening;
    bool load_model(Opening & o, llama_model ** reference = nullptr);
    detail::DeviceAnswer compare_device_answer(Opening & o);
    detail::AnswerWeighing weigh_device_answer(const detail::DeviceAnswer & a) const;
    bool read_model(Opening & o);
    bool build_hook(Opening & o);
    bool resolve_gate_mirror(GateMirror m, Opening & o);
    bool create_contexts(Opening & o);
    void build_sampler(const SessionConfig & c);
    void install_abort_callback();
    bool init_speculation(Opening & o);
    bool init_mtp_driver(Opening & o);
    bool stage_experts_in_host_buffer(Opening & o);
    bool bind_streaming(Opening & o);
    bool capture_warmup(Opening & o);
    bool arm_device_cache(Opening & o);
    bool device_cache_armed = false;
    bool bind_expert_offsets(Opening & o, const GgufOffsets & offs, std::vector<LayerExperts> & layers, int & n_expert);
    void hand_over_dense_weights(const SessionConfig & cfg,
                                 const GgufOffsets & offs,
                                 const std::vector<LayerExperts> & layers,
                                 std::vector<std::string> & unaccounted);
    void
    release_mapping(const SessionConfig & cfg, const GgufOffsets & offs, const std::vector<std::string> & unaccounted);
    void arm_route_trace(Opening & o, int n_expert, std::vector<uint64_t> dense_bytes);
    void arm_routing_record(Opening & o, int n_expert);
    bool arm_unstreamed_record(Opening & o);
    bool open_prefill_device(Opening & o);
    void hand_back_prefill(const std::string & why);
    bool confirm_prefill_device(Opening & o);
    void arm_decode_traces(Opening & o);
    void describe_run(Opening & o);
    void warn_about_load(const SessionConfig & cfg);

    // ── generate(): the steps before a turn, and the turn itself (see Turn) ──
    void begin_generate(const GenerateRequest & req, IMetricsSink * sink);
    struct Turn;

    ~Impl() {
        end_cancel_quiet(); // a window still open would demote every later llama.cpp error
        // Deterministic teardown order: stop the I/O pool (it holds fds into the mmap and its
        // buffers back the rebound expert tensors), then the context (its eval callback points
        // at the hook), then the hook, then unmap the model, then release the backend. The hook's
        // prediction worker goes first of all: it reads gate weights the source frees.
        if (hook) hook->predict_worker_stop();
        source.shutdown();
        // Hands every weight back to its mapping, then frees the device copy: before the model goes.
        prefill.reset();
        if (smpl) llama_sampler_free(smpl); // independent of ctx/model; free before them
        // The speculative driver holds both contexts and detaches the backend samplers it
        // installed on the draft one, so it goes before either context is freed.
        mtp.reset();
        if (mtp_batch_owned) llama_batch_free(mtp_batch);
        if (media_batch_owned) llama_batch_free(media_batch);
        ctx_dft.reset();
        ctx.reset();
        hook.reset();
        model.reset();
        expert_host_buf.reset();
        expert_host_registered.reset();
        mapping_placeholders.release();
        if (backend_inited) llama_backend_free();
    }
};

Session::Session() : impl_(std::make_unique<Impl>()) {}
Session::~Session() = default;

double Session::load_seconds() const {
    return impl_->load_seconds;
}
const std::string & Session::arch() const {
    return impl_->arch;
}
int Session::n_ctx() const {
    return impl_->cfg.n_ctx;
}
int Session::n_expert_used() const {
    return impl_->n_expert_used;
}
RoutingTrace Session::routing_record() const {
    return impl_->hook ? impl_->hook->routing_record() : RoutingTrace{};
}
MeasuredKnobs Session::measured_knobs() const {
    return impl_->trials.result;
}
ThinkControl Session::think_control() const {
    return impl_->think_ctl;
}
bool Session::vision() const {
    return impl_->media && impl_->media->vision();
}
std::vector<int32_t> Session::tokenize(const std::string & text, bool add_special, bool parse_special) const {
    std::vector<llama_token> tokens(text.size() + k_tokenize_slack);
    int n = llama_tokenize(impl_->vocab, text.c_str(), (int) text.size(), tokens.data(), (int) tokens.size(),
                           add_special, parse_special);
    if (n < 0) {
        tokens.resize((size_t) -n);
        n = llama_tokenize(impl_->vocab, text.c_str(), (int) text.size(), tokens.data(), (int) tokens.size(),
                           add_special, parse_special);
    }
    tokens.resize((size_t) std::max(n, 0));
    return tokens;
}
std::string Session::token_piece(int32_t token) const {
    if (token < 0 || token >= impl_->n_vocab) return "";
    char buf[256];
    const int n = llama_token_to_piece(impl_->vocab, token, buf, (int) sizeof(buf), 0, /*special*/ true);
    if (n >= 0) return std::string(buf, (size_t) n);
    std::string big((size_t) -n, '\0');
    const int m = llama_token_to_piece(impl_->vocab, token, big.data(), (int) big.size(), 0, true);
    big.resize((size_t) std::max(m, 0));
    return big;
}
bool Session::detokenize(const std::vector<int32_t> & tokens, std::string & text) const {
    for (const int32_t t : tokens)
        if (t < 0 || t >= impl_->n_vocab) return false;
    std::string out(tokens.size() * 8 + 16, '\0');
    int n = llama_detokenize(impl_->vocab, tokens.data(), (int32_t) tokens.size(), out.data(), (int32_t) out.size(),
                             /*remove_special*/ false, /*unparse_special*/ true);
    if (n < 0) {
        out.resize((size_t) -n);
        n = llama_detokenize(impl_->vocab, tokens.data(), (int32_t) tokens.size(), out.data(), (int32_t) out.size(),
                             false, true);
    }
    out.resize((size_t) std::max(n, 0));
    text = std::move(out);
    return true;
}
bool Session::render_chat(const GenerateRequest & req, std::string & prompt, std::string & error) const {
    const auto & im = *impl_;
    if (!im.chat_on) {
        error = "this session was opened without chat turns: there is no template to render";
        return false;
    }
    try {
        // The same inputs a turn builds (Turn::render_prompt), from the request's messages alone.
        common_chat_templates_inputs inputs;
        detail::build_turn_inputs(inputs, detail::to_common_messages(req.messages), req.think, im.think_ctl);
        detail::add_tools(inputs, req.tools, req.tool_choice);
        inputs.json_schema = req.json_schema;
        prompt = common_chat_templates_apply(im.chat_tmpls.get(), inputs).prompt.str();
    } catch (const std::exception & e) {
        error = std::string("the chat template rejected the conversation: ") + e.what();
        return false;
    }
    return true;
}
void Session::set_cache_budget_mb(int mib) {
    impl_->source.set_cache_budget((size_t) std::max(0, mib) * 1024ull * 1024ull);
}
// Teacher-forced perplexity. See PplRequest for why the batch is labelled decode.
PplResult Session::perplexity(const PplRequest & req) {
    auto & im = *impl_;
    PplResult r;
    const auto t0 = clock_t_::now();
    llama_context * ctx = im.ctx.get();

    std::vector<llama_token> tokens(req.text.size() + 8);
    int n = llama_tokenize(im.vocab, req.text.c_str(), (int) req.text.size(), tokens.data(), (int) tokens.size(),
                           /*add_special*/ true, /*parse_special*/ false);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(im.vocab, req.text.c_str(), (int) req.text.size(), tokens.data(), (int) tokens.size(), true,
                           false);
    }
    if (n < 2) {
        r.error = "text too short to score";
        return r;
    }
    tokens.resize(n);
    if (n > im.cfg.n_ctx) {
        r.error =
            "text of " + std::to_string(n) + " tokens exceeds the session n_ctx (" + std::to_string(im.cfg.n_ctx) + ")";
        return r;
    }
    r.n_tokens = n;

    im.clear_memory(ctx);
    im.forget_kv_tokens();

    // Warm-up graph, discarded. Both routing policies decide at the terminal node of each layer's
    // weight chain, and which node that is gets LEARNED from the first graph of a run — so on a
    // single-batch pass they would never fire at all, and every cell would score the baseline. One
    // throwaway decode teaches the hook the shape; the scoring pass then runs with the policy live.
    {
        llama_batch warm = llama_batch_init(1, 0, 1);
        batch_fill(warm, tokens.data(), 1, 0, false);
        im.hook->set_batch_phase(req.as_decode ? 1 : 0);
        const int rc = llama_decode(ctx, warm);
        llama_batch_free(warm);
        if (rc != 0) {
            r.error = "warm-up decode failed";
            return r;
        }
        im.clear_memory(ctx);
    }
    const long long routed0 = im.hook->experts_routed();
    const long long dropped0 = im.hook->experts_dropped();
    const long long reranked0 = im.hook->experts_reranked();
    const long long substituted0 = im.hook->experts_substituted();

    // Logits at EVERY position, so one pass scores every token — llama_batch_get_one would ask for
    // the last position only.
    llama_batch b = llama_batch_init(im.cfg.n_batch, /*embd*/ 0, /*n_seq_max*/ 1);
    struct BatchGuard {
        llama_batch & b;
        ~BatchGuard() { llama_batch_free(b); }
    } guard{b};

    double nll = 0.0;
    int scored = 0, top1 = 0;
    int last_row = 0; // logits row of the text's final position, for the choices
    // Score the logits at row `row` of the last decode against the token at `pos + 1`.
    auto score = [&](int row, int pos) -> bool {
        const float * lg = llama_get_logits_ith(ctx, row);
        if (!lg) {
            r.error = "no logits at position " + std::to_string(pos);
            return false;
        }
        // log softmax at the token that actually follows.
        const double logp = detail::log_norm(lg, im.n_vocab).logp(lg[tokens[pos + 1]]);
        nll -= logp;
        ++scored;
        if (argmax(lg, im.n_vocab) == tokens[pos + 1]) ++top1;
        return true;
    };

    if (req.step) {
        // The decode regime, token by token (see PplRequest::step). The unscored prefix goes in as
        // one prefill batch — labelled as such, so a decode-only policy stays out of it exactly as
        // it does in generation — and every scored position is its own one-token decode.
        const int prefix = std::max(1, std::min(req.skip, n - 1));
        batch_fill(b, tokens.data(), prefix, /*pos0*/ 0, /*all_logits*/ false);
        im.hook->set_batch_phase(0);
        if (llama_decode(ctx, b) != 0) {
            r.error = "prefill decode failed";
            return r;
        }
        // With choices the last token is fed too, so the final distribution exists.
        const int last = req.choices.empty() ? n - 1 : n;
        for (int pos = prefix; pos < last; ++pos) {
            batch_fill(b, tokens.data() + pos, 1, pos, /*all_logits*/ true);
            im.hook->set_batch_phase(1);
            if (llama_decode(ctx, b) != 0) {
                r.error = "decode failed at position " + std::to_string(pos);
                return r;
            }
            last_row = 0;
            if (pos < req.skip || pos + 1 >= n) continue;
            if (!score(0, pos)) return r;
        }
    } else {
        // Same placement rule as generate()'s prefill, so --ppl scores exactly the numerics a prompt
        // gets: wide pieces on the prefill device, a narrow tail on the CPU.
        // Every row here wants its logits, so a piece is also capped by the output rows the context
        // reserved (PrefillDeviceConfig::max_outputs), with or without a device.
        const int step =
            std::min(im.prefill ? prefill_piece(im.cfg) : im.cfg.n_batch, PrefillDeviceConfig::max_outputs);
        Impl::PlacementGuard placement_guard{im};
        for (int i = 0; i < n; i += step) {
            const int chunk = std::min(step, n - i);
            im.place_prefill(im.prefill && chunk >= im.cfg.prefill.min_tokens);
            batch_fill(b, tokens.data() + i, chunk, /*pos0*/ i, /*all_logits*/ true);
            im.hook->set_batch_phase(req.as_decode ? 1 : 0);
            if (im.decode_placed(ctx, b) != 0) {
                r.error = "decode failed at position " + std::to_string(i);
                return r;
            }
            // Position p predicts token p+1, so the last token of the whole text is never scored
            // and the last row of a chunk predicts the first token of the next one.
            for (int j = 0; j < chunk; ++j) {
                const int pos = i + j;
                if (pos + 1 >= n || pos < req.skip) continue;
                if (!score(j, pos)) return r;
            }
            last_row = chunk - 1;
        }
    }
    if (!req.choices.empty()) {
        const float * lg = llama_get_logits_ith(ctx, last_row);
        if (!lg) {
            r.error = "no logits after the text";
            return r;
        }
        const detail::LogNorm z = detail::log_norm(lg, im.n_vocab);
        for (const std::string & c : req.choices) {
            llama_token ct[8];
            const int nc = llama_tokenize(im.vocab, c.c_str(), (int) c.size(), ct, 8, false, false);
            if (nc < 1) {
                r.error = "choice '" + c + "' does not tokenize";
                return r;
            }
            r.choice_logp.push_back(z.logp(lg[ct[0]]));
        }
    }
    if (scored == 0 && req.choices.empty()) {
        r.error = "nothing scored — text shorter than skip + 1";
        return r;
    }
    r.nll = scored > 0 ? nll / scored : 0.0;
    r.ppl = std::exp(r.nll);
    r.n_scored = scored;
    r.n_top1 = top1;
    r.experts_routed = im.hook->experts_routed() - routed0;
    r.experts_dropped = im.hook->experts_dropped() - dropped0;
    r.experts_reranked = im.hook->experts_reranked() - reranked0;
    r.experts_substituted = im.hook->experts_substituted() - substituted0;
    r.seconds = secs(t0, clock_t_::now());
    r.ok = true;
    return r;
}

DecideResult Session::decide(const DecideRequest & req) {
    Impl & im = *impl_;
    if (!im.cfg.decide.enabled) {
        DecideResult off;
        off.error = "decide is off for this session (open it with decide enabled)";
        off.error_kind = RunError::InvalidRequest;
        return off;
    }
    im.cancel_requested.store(false, std::memory_order_relaxed);
    struct CancelQuiet {
        Impl & im;
        ~CancelQuiet() { im.end_cancel_quiet(); }
    } cancel_quiet{im}; // as in generate(): the window ends with the call

    // A decision is not a conversation turn: whatever generate() was continuing is gone once the
    // context is overwritten, so forget it rather than let a later turn diff against stale tokens.
    im.chat_history.clear();
    im.forget_kv_tokens();

    detail::LlamaDecideDeps deps;
    deps.ctx = im.ctx.get();
    deps.ctx_dft = im.ctx_dft.get();
    deps.vocab = im.vocab;
    deps.n_vocab = im.n_vocab;
    deps.n_ctx = im.cfg.n_ctx;
    // With a prefill device each piece is one graph whose width we choose, as in generate().
    deps.n_batch = im.prefill ? prefill_piece(im.cfg) : im.cfg.n_batch;
    deps.hook = im.hook.get();
    deps.source = &im.source;
    deps.moe_on = im.cfg.moe.enabled;
    deps.tmpls = im.chat_on ? im.chat_tmpls.get() : nullptr;
    deps.think_ctl = im.think_ctl;
    deps.prefill = im.prefill.get();
    deps.prefill_min_tokens = im.cfg.prefill.min_tokens;
    detail::LlamaDecideBackend backend(deps);

    if (!im.decide_cache_chosen) {
        // No kept state with a prefill device: llama.cpp would save and restore it through KV views
        // that still point where the state was before PrefillPath moved it (validation refuses `on`).
        im.decide_cache = detail::make_prefix_cache(im.prefill ? PrefixCacheMode::Off : im.cfg.decide.prefix_cache,
                                                    backend.prefill_cost_scales_with_tokens());
        im.decide_cache_chosen = true;
    }
    if (im.decide_probe) {
        std::vector<int32_t> ids;
        for (const std::string & c : req.choices) {
            const std::vector<detail::Token> t = backend.tokenize_plain(c);
            ids.push_back(t.empty() ? -1 : t[0]);
        }
        im.decide_probe->begin(ids);
    }
    DecideResult r = detail::run_decide(backend, im.decide_cache.get(), req);
    if (im.decide_probe) {
        im.decide_probe->end();
        if (r.ok)
            im.decide_probe->write(im.decide_seq, r.n_tokens, r.prefill.seconds, req.choices, r.choice_logp, r.best);
        ++im.decide_seq;
    }
    // The abort callback turns a cancel into a failed decode; that is a request honoured, not a
    // context gone bad, and run_decide has already emptied the sequence.
    if (!r.ok && r.fatal && im.cancelled()) {
        r.fatal = false;
        r.cancelled = true;
        r.error = "cancelled";
        r.error_kind = RunError::None;
    }
    return r;
}

void Session::cancel() {
    impl_->cancel_requested.store(true, std::memory_order_relaxed);
}

// What open()'s steps hand one another: the request, the facts settled so far, and the caller's
// error out-parameter. A step that fails sets the error through fail() and returns false, and
// open() stops there, so a step never runs after one that failed — the same early returns the
// single function had, now one per step.
struct Session::Impl::Opening {
    const SessionConfig & cfg; // the caller's: im.cfg is a copy a step may correct (a device dropped)
    std::string & error;
    IRouteTraceSink * route_trace;
    IComputeTraceSink * compute_trace;
    IIoTraceSink * io_trace;

    // The gguf header answers several separate questions below — the arch-prefixed key for a
    // top-k override, the route trace's effective top-k, the run info's top-k/expert count, and
    // the streamer's per-tensor file offsets — and each used to reopen and reparse the file for
    // its own answer. Parse it at most once, lazily: the callers are conditional (a run with no
    // override, no trace and no streaming asks nothing), so an eager read would be work the
    // common path never needs. One parse serves both the model info and the offsets.
    GgufMeta gguf_meta{};
    bool gguf_meta_read = false;
    const GgufMeta & meta() {
        if (!gguf_meta_read) {
            gguf_meta = read_gguf_meta(cfg.model_path.c_str());
            gguf_meta_read = true;
        }
        return gguf_meta;
    }
    const GgufModelInfo & gguf() { return meta().info; }

    ExpertLayout layout{};    // set by read_model when streaming; left !ok otherwise, and then no tensor matches it
    int n_layer_streamed = 0; // set by build_hook

    bool fail(std::string msg) {
        error = std::move(msg);
        return false;
    }
};

// Load with the layout the streamer requires: file-backed mmap, no repack (a repacked
// q4_K buffer would break the rebind), and every expert the streamer serves in host memory.
//
// Layers on devices are the first stage's decision (llama.cpp's capacity fitter, via the
// planner), carried in as n_gpu_layers plus the override patterns it wrote. With nothing
// planned this is the historical behaviour, everything on the host.
//
// Every pattern is routed to the CPU buffer type. That is what the fitter itself does for the
// experts it leaves on the host, and it is the one placement the streamer can serve: it rebinds
// `data` onto bytes it read itself, and only a host address can be rebound.
//
// Routing them to a DEVICE's host buffer type instead - the obvious way to let an accelerator
// compute a streamed expert - does not work, and the two reasons are both in this submodule.
// `llama-model-loader.cpp` ("avoid using a host buffer when using mmap") replaces any host
// buffer type with the CPU's whenever the model is mmap'd, which this engine always is; and
// `ggml-cuda.cu`'s `supports_buft` accepts the CUDA host buffer only on an `integrated` device,
// so on a discrete one the op falls back to the CPU regardless. See docs/hardware-planning.md
// for what a working version would have to do instead - wrap our own reservation with
// `ggml_backend_dev_buffer_from_host_ptr`, which CUDA does not implement and Metal does.
//
// With `reference`, the model is handed to the caller rather than kept, and nothing of the session
// is touched: the host-only twin a device's answers are compared with (compare_device_answer), loaded
// by the same rules as the model it stands beside.
bool Session::Impl::load_model(Opening & o, llama_model ** reference) {
    const SessionConfig & cfg = o.cfg;
    llama_model_params mparams = llama_model_default_params();
    // A device we are not using should not be in the room. Left registered, the scheduler hands it
    // every node it can execute - the weightless ones especially - purely because it is there, and
    // each one is a boundary the graph crosses twice. Handing llama.cpp an empty device list is the
    // only way to say "not this run" that the scheduler cannot talk itself out of.
    //
    // Empty, not the CPU: the list names the devices llama.cpp offloads to, and the context adds its
    // own CPU backend after them whatever the list holds. A CPU named here becomes a second CPU
    // backend that the scheduler places the graph on, and a "first device" whose host buffer type
    // the context would take for the CPU's intermediate state. Empty is what a build without a GPU
    // backend gets from llama.cpp's own selection.
    //
    // Unless told otherwise that is decided by what was placed: a layer on a device means the
    // devices are wanted, none means they are not. So a build that carries a GPU backend runs a
    // host-only plan exactly as a build without it does, instead of slower and rounded differently.
    const bool cpu_only =
        cfg.device_use == DeviceUse::CpuOnly || (cfg.device_use == DeviceUse::Auto && cfg.n_gpu_layers == 0);
    ggml_backend_dev_t no_devices[1] = {nullptr};
    if (cpu_only) mparams.devices = no_devices;
    mparams.load_mode = LLAMA_LOAD_MODE_MMAP;
    mparams.use_extra_bufts = false;
    mparams.n_gpu_layers = cfg.n_gpu_layers;
    // llama.cpp reads the rows of arch-marked tensors on demand, and its default turns that on by
    // itself for anything over 4 GiB. The tensors it marks - the per-layer embedding tables of
    // gemma4 and qwen4exp - are exactly the ones `--row-stream` gathers rows from, so the two
    // mechanisms would be paging the same bytes by two different routes, each unaware of the
    // other's residency. Whichever is doing it, only one may: ours when it is asked for, theirs
    // otherwise. Inheriting the default silently is the one option that is never right.
    // A table the plan leaves mapped (dense_mapped) is priced as rows faulted in as the graph asks,
    // and llama.cpp's lazy mode would prefetch the whole of it at load into pages the plan did not
    // charge: off for that case too.
    mparams.lazy_mode =
        cfg.moe.row_stream || !cfg.moe.dense_mapped.empty() ? LLAMA_LAZY_MODE_OFF : LLAMA_LAZY_MODE_AUTO;
    std::vector<llama_model_tensor_buft_override> buft_overrides;
    if (!cfg.buft_overrides.empty()) {
        buft_overrides.reserve(cfg.buft_overrides.size() + 1);
        for (const std::string & pat : cfg.buft_overrides)
            buft_overrides.push_back({pat.c_str(), ggml_backend_cpu_buffer_type()});
        buft_overrides.push_back({nullptr, nullptr}); // terminator
        mparams.tensor_buft_overrides = buft_overrides.data();
    }
    // The nextn/MTP block is skipped at load unless asked for: llama.cpp marks its tensors
    // TENSOR_SKIP by default, and only --mtp builds a graph over them. n_layer_nextn comes from
    // the gguf metadata either way, so the "this model has no trained head" check below is
    // unaffected by this flag.
    mparams.load_mtp = cfg.spec.is_mtp() && cfg.spec.mtp_head.empty();

    // The prefill device joins the scheduler as the model's only non-CPU device, with no layer
    // assigned to it: every weight still loads from the mmap, and the device runs nothing until
    // PrefillDevice moves the layer weights onto it for a wide graph.
    // A device that is missing or does not open leaves the whole run on the CPU, exactly as without
    // the flag: the prefill is slower, the output is the CPU's, and the load does not fail over it.
    std::vector<ggml_backend_dev_t> auto_devices;
    if (cfg.prefill.enabled()) {
        std::string derr;
        if (ggml_backend_dev_t dev = detail::PrefillPath::find_device(cfg.prefill.device, derr)) {
            prefill = std::make_unique<detail::PrefillPath>(dev);
            mparams.devices = prefill->model_devices();
            // With a device listed, llama.cpp would put CPU weights in the device's host buffer type,
            // which is an allocation and a copy, not the file mapping. The mapping is load-bearing.
            mparams.no_host = true;
        } else {
            log_printf(LogLevel::Warn, "%s; prefill stays on the CPU", derr.c_str());
            this->cfg.prefill.device.clear();
        }
    }
    if (!prefill && !cpu_only) {
        if (const ggml_backend_dev_t * devs = detail::PrefillPath::devices_without_prefill(auto_devices)) {
            mparams.devices = const_cast<ggml_backend_dev_t *>(devs);
        }
    }
    // The backends the plan chose, one per card: a name this build does not have is a plan for
    // another machine.
    std::vector<ggml_backend_dev_t> listed_devices;
    if (!prefill && !cpu_only && !cfg.device_list.empty()) {
        for (const std::string & n : cfg.device_list) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(n.c_str());
            if (!dev)
                return o.fail("{device-list} names device '" + n +
                              "', which this build does not have: plan again on this machine");
            listed_devices.push_back(dev);
        }
        listed_devices.push_back(nullptr);
        mparams.devices = listed_devices.data();
    }
    // The fitter's split, applied as it was projected: the devices it put layers on, in its order,
    // each with its own count. Left to llama.cpp the layers are divided by free memory, which is not
    // the division the fitter sized each device's share for - on two cards it handed one of them
    // 17 GB of a placement planned across both (#321). A device the plan names that is not here is
    // a plan for another machine, and loading it anyway would place layers nobody budgeted.
    std::vector<ggml_backend_dev_t> split_devices;
    std::vector<float> split_layers;
    if (!prefill && !cpu_only && !cfg.device_split.empty()) {
        for (const auto & d : cfg.device_split) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(d.first.c_str());
            if (!dev)
                return o.fail("{device-split} names device '" + d.first +
                              "', which this build does not have: plan again on this machine");
            split_devices.push_back(dev);
        }
        split_layers.assign(llama_max_devices(), 0.0f);
        for (size_t i = 0; i < cfg.device_split.size() && i < split_layers.size(); ++i)
            split_layers[i] = (float) cfg.device_split[i].second;
        split_devices.push_back(nullptr);
        mparams.devices = split_devices.data();
        mparams.tensor_split = split_layers.data();
    }

    // Optional active-expert override: reduce the model's top-k routing (e.g. 8 -> 6) to cut
    // per-token compute and — under streaming — flash I/O, at a quality cost. Applied purely
    // through llama.cpp's public kv_overrides on the arch-prefixed expert_used_count key: the
    // graph then routes to fewer experts and the whole streaming path adapts automatically
    // (the router hook reads the top-k width from the graph). The array must outlive the load
    // call below. No llama.cpp patch, no per-arch constants.
    llama_model_kv_override kv_overrides[2];
    std::memset(kv_overrides, 0, sizeof(kv_overrides)); // second entry stays the key[0]==0 terminator
    if (cfg.n_expert_used > 0) {
        const GgufModelInfo & info = o.gguf();
        if (!info.ok) return o.fail("cannot read the model file's metadata: " + cfg.model_path);
        if (info.arch.empty())
            return o.fail("{n-expert-used} cannot be applied: the model file declares no architecture "
                          "(general.architecture), so it is incomplete or not a model");
        if (info.n_expert <= 0)
            return o.fail("{n-expert-used} was set, but this model has no experts (the file declares no expert "
                          "count): leave it at 0");
        if (cfg.n_expert_used > info.n_expert)
            return o.fail("{n-expert-used} (" + std::to_string(cfg.n_expert_used) +
                          ") is more than the model's experts (" + std::to_string(info.n_expert) + ")");
        const std::string key = info.arch + ".expert_used_count";
        kv_overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        std::snprintf(kv_overrides[0].key, sizeof(kv_overrides[0].key), "%s", key.c_str());
        kv_overrides[0].val_i64 = cfg.n_expert_used;
        mparams.kv_overrides = kv_overrides;
    }

    llama_model * m = llama_model_load_from_file(cfg.model_path.c_str(), mparams);
    if (!m) return o.fail("could not load the model: " + cfg.model_path);
    if (reference) {
        *reference = m;
        return true;
    }
    model.reset(m);
    if (cfg.spec.is_mtp() && !cfg.spec.mtp_head.empty()) {
        // The head whole and resident, where the plan put it: on the devices the target's layers
        // are on (its bytes taken from the device expert cache) or on the host. One layer the
        // streamer does not serve, small beside the model, computed for a draft position at a time.
        llama_model_params hp = llama_model_default_params();
        hp.n_gpu_layers = cfg.spec.mtp_head_on_devices ? llama_model_default_params().n_gpu_layers : 0;
        hp.devices = mparams.devices; // the same list, so the head cannot land on a device the plan left out
        hp.load_mtp = true;
        llama_model * h = llama_model_load_from_file(cfg.spec.mtp_head.c_str(), hp);
        if (!h) return o.fail("could not load {mtp-head}: " + cfg.spec.mtp_head);
        model_dft.reset(h);
    }
    return true;
}

// The model's answers on the devices its layers were opened on, against the same file on the cores
// (KnobTrialConfig::answer_device; the comparison is device_answer.h's). Run on the context as
// created, before anything is armed on it: nothing is recorded, traced or timed yet, so the sequence
// leaves no mark on what the session measures.
detail::DeviceAnswer Session::Impl::compare_device_answer(Opening & o) {
    const SessionConfig & cfg = o.cfg;
    const SessionConfig host = detail::on_cores(cfg);
    detail::AnswerReference ref;
    std::string load_error;
    ref.load_model = [&]() -> llama_model * {
        Opening ho{host, load_error, nullptr, nullptr, nullptr};
        llama_model * m = nullptr;
        return load_model(ho, &m) ? m : nullptr;
    };
    // The types the context was created with, after any fallback (create_contexts).
    ref.type_k = detail::to_ggml_type(this->cfg.cache_type_k);
    ref.type_v = detail::to_ggml_type(this->cfg.cache_type_v);
    ref.n_threads = cfg.n_threads;
    ref.n_threads_batch = cfg.n_threads_batch > 0 ? cfg.n_threads_batch : cfg.n_threads;
    const auto t0 = clock_t_::now();
    const bool timed = cfg.trial.answer_prompt_per_token >= 0.0f;
    const detail::DeviceAnswer a = detail::check_device_answer(ctx.get(), ref, cfg.trial.test_answer_differs, timed);
    const double check_s = secs(t0, clock_t_::now());
    const char * dev = cfg.trial.answer_device.c_str();
    if (a.verdict == detail::DeviceAnswer::Verdict::Unchecked) {
        log_printf(LogLevel::Warn,
                   "device answer: %s's answers for this model were not compared with the CPU's (%s%s%s); the "
                   "model is reopened on the CPU",
                   dev, a.unchecked.c_str(), load_error.empty() ? "" : ": ", load_error.c_str());
        return a;
    }
    const bool ok = a.verdict == detail::DeviceAnswer::Verdict::Confirmed;
    log_printf(ok ? LogLevel::Info : LogLevel::Warn,
               "device answer: %s against the CPU on %d tokens in %d graphs from 1 to %d wide, %d rows compared, "
               "top token agrees %.0f%%, logit NMSE %.2e, KL mean %.4f max %.4f, device %.2f s, CPU reference "
               "%.2f s, %.2f s in all - %s",
               dev, a.tokens, a.pieces, a.wide, a.agreement.rows, a.agreement.agree() * 100.0, a.agreement.nmse(),
               a.agreement.kld, a.agreement.kld_max, a.device_s, a.host_s, check_s,
               ok ? "confirmed" : "it does not compute this model: the model is reopened on the CPU");
    if (ok && timed) {
        const detail::AnswerWeighing w = weigh_device_answer(a);
        if (w.side == detail::AnswerSide::Unweighed)
            log_printf(LogLevel::Warn, "device answer: the two sides could not be timed on this model; %s keeps it",
                       dev);
        else
            log_printf(LogLevel::Info,
                       "device answer: this model one token at a time takes %.1f ms on %s and %.1f ms on the CPU "
                       "(%.1f against %.1f tok/s), a %d-token graph %.0f ms against %.0f; with %.1f prompt tokens a "
                       "decoded token a request takes %.2fx the CPU's time on %s - %s",
                       a.device_token_s * 1e3, dev, a.host_token_s * 1e3, 1.0 / a.device_token_s, 1.0 / a.host_token_s,
                       a.wide, a.device_wide_s * 1e3, a.host_wide_s * 1e3, (double) cfg.trial.answer_prompt_per_token,
                       w.device_request_s / w.cores_request_s, dev,
                       w.side == detail::AnswerSide::Cores
                           ? "the CPU is faster by the margin: the model is reopened on it"
                           : "the device keeps the model");
    }
    return a;
}

// The check's timing, weighed as this load was asked to weigh it (KnobTrialConfig::answer_prompt_per_token)
// by the margin every decision between two measured sides here uses (KnobTrialPolicy::min_win).
detail::AnswerWeighing Session::Impl::weigh_device_answer(const detail::DeviceAnswer & a) const {
    if (cfg.trial.answer_prompt_per_token < 0.0f) return {};
    return detail::weigh_answer_sides(a, (double) cfg.trial.answer_prompt_per_token, KnobTrialPolicy{}.min_win);
}

// The loaded model's own facts, the expert layout read from its tensors, and its chat template.
bool Session::Impl::read_model(Opening & o) {
    const SessionConfig & cfg = o.cfg;
    llama_model * m = model.get();
    vocab = llama_model_get_vocab(m);
    n_vocab = llama_vocab_n_tokens(vocab);
    n_layer = llama_model_n_layer(m);
    // Excluded from n_layer by llama.cpp, so ask for it separately. Zero on every model without a
    // trained MTP head — which is most ggufs, including quants of MTP-capable models that were
    // converted without the nextn tensors.
    n_layer_nextn = llama_model_n_layer_nextn(m);
    if (model_dft && llama_model_n_layer_nextn(model_dft.get()) <= 0)
        return o.fail("{mtp-head} names a file with no MTP head in it (its nextn_predict_layers is absent or zero): " +
                      cfg.spec.mtp_head);
    if (cfg.spec.is_mtp() && !model_dft && n_layer_nextn <= 0)
        return o.fail("{spec=mtp} needs a model with a trained MTP head, and this model has none (no nextn block "
                      "in the file): give a head file in {mtp-head}, or use {spec=ngram}, which works with "
                      "any model since it drafts from the text rather than from the weights.");

    char arch_buf[128] = {0};
    llama_model_meta_val_str(m, "general.architecture", arch_buf, sizeof(arch_buf));
    arch = arch_buf;

    // Startup prefills already load experts, so their pin-fit floor needs the same effective
    // routing width as generation before any streamed graph runs, even without a metrics sink.
    n_expert_used = cfg.n_expert_used;
    if (n_expert_used <= 0) {
        const GgufModelInfo & mi = o.gguf();
        n_expert_used = mi.ok ? mi.n_expert_used : 0;
    }

    if (cfg.moe.enabled) {
        // Which tensors are the routed experts, read off the file's shapes. A dense model, or a MoE
        // laid out in a way this engine has not proved byte-identical, is refused here with the reason
        // rather than streamed by a guess.
        o.layout = discover_expert_layout(o.meta().offsets.extents, o.gguf().n_expert);
        if (!o.layout.ok)
            return o.fail("not a MoE this engine streams: " + o.layout.error +
                          ". Turn {moe-stream} off to run it from the mapped file (docs/which-models-stream.md).");
    } else if (!cfg.moe.expert_host_buffer.empty()) {
        // The copy into a device's host memory needs only to know which tensors are experts; a file
        // it cannot read that way is simply not copied.
        o.layout = discover_expert_layout(o.meta().offsets.extents, o.gguf().n_expert);
    }

    // Chat templates are model-bound: initialise once here, apply per prompt in generate().
    if (cfg.chatml) {
        try {
            chat_tmpls = common_chat_templates_init(m, "");
            chat_on = true;
            // Which "thinking off" mechanism this template supports is a property of the model, so
            // it is settled once here rather than re-derived on every turn.
            think_ctl = detail::probe_think_control(chat_tmpls.get());
        } catch (const std::exception & e) {
            log_printf(LogLevel::Warn, "chat template unavailable (%s); using raw prompts", e.what());
            chat_on = false;
        }
    }

    // The projector beside the model: a session opened with one reads images. A projector that
    // does not match the model is refused here, with the reason, not at the first image.
    if (!cfg.mmproj.empty()) {
        std::string merr;
        // On the devices by the rule the model's own placement follows (load_model): a layer placed
        // on one means they are wanted, and -1 is every layer, which is what the fitter leaves where
        // the model fits a device whole. The planner prices the projector by the same rule, and the
        // fitter reserved its room there.
        const bool devices_wanted =
            cfg.device_use == DeviceUse::All || (cfg.device_use == DeviceUse::Auto && cfg.n_gpu_layers != 0);
        media = detail::MediaInput::open(cfg.mmproj, m, cfg.n_threads, devices_wanted, cfg.image_max_tokens,
                                         &cancel_requested, merr);
        if (!media) return o.fail(merr);
        if (!media->vision()) {
            media.reset();
            return o.fail("the projector in " + cfg.mmproj + " has no vision encoder: this engine reads images only");
        }
        log_printf(LogLevel::Info, "projector: %s loaded, images are read", cfg.mmproj.c_str());
    }
    return true;
}

// Route-ahead's gate format. A plan that timed it has already said on or off; left at auto, the same
// timing runs here, on the same kernels and this model's gate shape (gate_gemv::measure), so a run
// without --auto gets the faster path rather than one read off the compiler's target. A tie goes to
// the mirror, because the timing runs alone and a decode beside it only makes bytes dearer.
bool Session::Impl::resolve_gate_mirror(GateMirror m, Opening & o) {
    if (m != GateMirror::Auto) return m == GateMirror::On;
    // Every layer that carries a gate, so the working set is the run's own gate set (see measure()).
    const uint32_t gated = o.layout.layers.empty() ? (uint32_t) n_layer : (uint32_t) o.layout.layers.size();
    const gate_gemv::Timing t =
        gate_gemv::measure((uint32_t) llama_model_n_embd(model.get()), (uint32_t) o.gguf().n_expert, gated);
    if (!t.measured()) {
        log_printf(LogLevel::Info, "gate-mirror: auto -> off (the gate shape is unknown, nothing was timed)");
        return false;
    }
    const bool mirror = t.int8_us <= t.f32_us;
    log_printf(LogLevel::Info, "gate-mirror: auto -> %s (int8 %.1f us vs float %.1f us per layer, %u layers timed)",
               mirror ? "on" : "off", t.int8_us, t.f32_us, t.layers);
    return mirror;
}

// The router hook, configured for every job this session asked of it, and the decide probe.
bool Session::Impl::build_hook(Opening & o) {
    const SessionConfig & cfg = o.cfg;
    // The MTP block lives at layer index n_layer and routes experts of its own, so with speculation
    // on the hook and the streamer must span the trunk PLUS the head. The index space is contiguous
    // and the tensor naming identical, so this bound is the only thing standing between the streamer
    // and the MTP experts — left at n_layer they are silently skipped and stay mmap-resident.
    o.n_layer_streamed = n_layer + (cfg.spec.is_mtp() && !model_dft ? n_layer_nextn : 0);
    hook = std::make_unique<RouterHook>(o.layout, o.n_layer_streamed);
    hook->set_prefetch_layers(cfg.moe.prefetch_layers);
    hook->set_drop_policy(cfg.moe.drop_cold_frac, cfg.moe.drop_renorm, cfg.moe.drop_prefill);
    hook->set_expert_substitute(cfg.moe.substitute_lambda);
    hook->set_predict_log(cfg.moe.predict_log);
    hook->set_predict_prefetch(cfg.moe.predict_prefetch, cfg.moe.predict_spec_max);
    if (cfg.moe.route_ahead > 0) hook->set_gate_mirror(resolve_gate_mirror(cfg.moe.gate_mirror, o));
    hook->set_route_ahead(cfg.moe.route_ahead);
    if (cfg.decide.enabled && !cfg.decide.probe_path.empty()) {
        decide_probe = std::make_unique<DecideProbe>(cfg.model_path, n_layer, o.gguf().n_expert,
                                                     llama_model_n_embd(model.get()), cfg.decide.probe_path);
        if (!decide_probe->ok()) return o.fail("decide probe: " + decide_probe->error());
        hook->set_probe(decide_probe.get());
    }
    return true;
}

// The target context, and the MTP draft context beside it when speculation drafts from the head.
bool Session::Impl::create_contexts(Opening & o) {
    const SessionConfig & cfg = o.cfg;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = cfg.n_ctx;
    cparams.n_batch = cfg.n_batch;
    // The graph is reserved for the widest ubatch, so this is what sets the resident compute
    // buffers — the memory this engine is always short of. 0 keeps the historical behaviour
    // (one graph as wide as the batch); a smaller value chunks prefill to buy that memory back.
    cparams.n_ubatch = cfg.n_ubatch > 0 ? (uint32_t) cfg.n_ubatch : (uint32_t) cfg.n_batch;
    cparams.type_k = detail::to_ggml_type(cfg.cache_type_k);
    cparams.type_v = detail::to_ggml_type(cfg.cache_type_v);
    // A sliding-window layer keeps its window, not the whole context. llama.cpp's own default is a
    // full-size cache on those layers too (common/, which its server and CLI use, turns it off), and
    // on a model whose layers are mostly windowed that is most of the context's memory spent on cells
    // no token reads: Gemma 4 26B-A4B at a 16384 context held ~2 GiB for its context on a 12 GB
    // phone, where that was the memory the expert cache and the prefill device were refused. A
    // cut further back than the window is served from a checkpoint instead (keep_kv_prefix). The
    // placement probe opens its context the same way, so the plan prices the cache the run holds.
    cparams.swa_full = false;
    // The streamer needs the callback to see routing; the compute trace needs it to time nodes.
    // Installing it for the trace alone is what lets a NON-streamed run be measured — the dense
    // mmap baseline the streamed numbers are argued against.
    // The prefill device needs it once, for the capture that finds the layer weights, and so does
    // the copy of the experts into a device's host memory.
    // The copy captures on the context the device cache then makes again, which needs no callback.
    const bool copy_capture = !cfg.moe.expert_host_buffer.empty() && !device_cache_armed;
    if (cfg.moe.enabled || o.compute_trace || prefill || records_unstreamed(cfg) || copy_capture) {
        cparams.cb_eval = &RouterHook::c_eval;
        cparams.cb_eval_user_data = hook.get();
    }
    // Placement is decided here, by moving weights; a backend's own offload heuristic would move
    // ops of CPU-resident weights to the device on its own (GPU backends do for wide batches).
    // The rule, and the runs that showed why, is offloads_wide_matmuls (config.h): a front-end says
    // where a prompt runs by the same function. The prefill device is this session's own, armed.
    if (!offloads_wide_matmuls(cfg.moe, cfg.n_gpu_layers, prefill != nullptr)) cparams.op_offload = false;
    // Logit rows are reserved for as many outputs as a graph may return, and left at llama.cpp's
    // default that is the whole ubatch. A row is the whole vocabulary, and the reservation goes where
    // the output layer is: on a CPU-only run of a 35B-A3B at 2048 tokens it was 1.7 GiB of compute
    // buffer for rows a prompt never reads, and on a 24 GB card holding a 125B model's layers it was
    // the 2.1 GiB that no longer fit, so the context could not be created. The planner projects
    // every run at this cap, and leaving it to the prefill device made the run reserve 2028 MiB where
    // the ledger had 436.
    cparams.n_outputs_max = (uint32_t) PrefillDeviceConfig::max_outputs;
    // Rejecting a draft means rewinding the KV to the last accepted position. With recurrent-state
    // snapshots the rewind is a cheap restore; without them llama.cpp has to fall back to replaying
    // the sequence, which would hand back exactly the decode the speculation just saved.
    if (cfg.spec.enabled()) cparams.n_rs_seq = (uint32_t) cfg.spec.draft_max;
    // The device-side expert cache is armed only on the second creation (see open()): with it on,
    // the expert matmuls read its slots rather than the host tensors, so the capture warm-up, which
    // finds those tensors by the nodes that read them, runs on a context without it.
    if (device_cache_armed) cparams.moe_cache_size = (size_t) cfg.moe.device_cache_mb << 20;
    // Part of the misses computed on the host from the expert's host copy rather than uploaded: the
    // same tensors the streamer serves, so the CPU half reads what the slots would have been filled
    // from (docs/seam.md).
    if (device_cache_armed) cparams.moe_cache_cpu_share = cfg.moe.device_cache_cpu_share;

    llama_context * c = llama_init_from_model(model.get(), cparams);
    // The device cache is sized from what the capacity fitter left free, and the fitter measured a
    // graph without the cache; with it, a wide graph gathers a layer's experts into the device's own
    // compute buffers, which no estimate made before the context exists has counted. Measured on a
    // 24 GB card with DeepSeek V4.1: a 15.3 GB cache left 2.4 GB of prefill buffers nowhere to go, and
    // the run could not start. The context is the instrument: the cache gives up a quarter at a time
    // until it fits, never below one token cycle, under which it would return no hits.
    if (!c && device_cache_armed && cparams.moe_cache_size > 0) {
        const size_t floor = n_expert_used > 0 ? source.worst_cycle_bytes(n_expert_used) : 0;
        size_t size = cparams.moe_cache_size;
        while (!c && size - size / 4 >= std::max<size_t>(floor, 1)) {
            const size_t smaller = (size - size / 4) & ~((size_t(1) << 20) - 1);
            log_printf(LogLevel::Warn,
                       "the %zu MiB device expert cache did not fit beside the context; retrying at %zu MiB",
                       size >> 20, smaller >> 20);
            size = smaller;
            cparams.moe_cache_size = size;
            c = llama_init_from_model(model.get(), cparams);
        }
        if (c) this->cfg.moe.device_cache_mb = (int) (size >> 20);
    }
    // A quantized V cache needs flash attention, and whether a backend has it is only known by asking
    // llama.cpp for the context. The default is quantized, so a backend without it must not lose the
    // run: V goes back to F16 (K stays, it needs nothing). The fitter charged the smaller V, which is
    // why the fallback is said out loud.
    if (!c && ggml_is_quantized(cparams.type_v)) {
        log_printf(LogLevel::Warn,
                   "a %s V cache needs flash attention, which this backend refused; V falls back to f16",
                   kv_cache_type_name(cfg.cache_type_v));
        cparams.type_v = GGML_TYPE_F16;
        c = llama_init_from_model(model.get(), cparams);
        if (c) this->cfg.cache_type_v = KvCacheType::F16;
    }
    // The graph's compute buffers are reserved with the context, and their size is an estimate until
    // then: priced off a slope the probe measured, on a device whose free memory others share. Where
    // the reservation fails, the context is the instrument again: the graph halves until it fits, down
    // to llama.cpp's own physical batch, or the most tokens one image takes in a pass where the run
    // has images. A narrower graph changes how many passes a prompt takes, never what it computes.
    if (!c) {
        const uint32_t image_floor =
            !cfg.mmproj.empty() && cfg.image_max_tokens > 0 ? (uint32_t) cfg.image_max_tokens : 0;
        const uint32_t floor_w = std::max(llama_context_default_params().n_ubatch, image_floor);
        while (!c && cparams.n_ubatch / 2 >= floor_w) {
            log_printf(LogLevel::Warn,
                       "the context's compute buffers did not fit at a graph %u tokens wide; retrying at %u",
                       cparams.n_ubatch, cparams.n_ubatch / 2);
            cparams.n_ubatch /= 2;
            c = llama_init_from_model(model.get(), cparams);
        }
        if (c) this->cfg.n_ubatch = (int) cparams.n_ubatch;
    }
    if (!c) return o.fail("failed to create context"); // llama.cpp's own log says why
    ctx.reset(c);
    // Decode and prefill get their own counts. They are different workloads: prefill is compute-bound
    // and scales with cores, a streamed decode waits on flash. 0 means "the same", which is what
    // every caller got before this existed.
    const int n_threads_batch = cfg.n_threads_batch > 0 ? cfg.n_threads_batch : cfg.n_threads;
    llama_set_n_threads(c, cfg.n_threads, n_threads_batch);

    // The MTP draft context: same model, same eval callback, but ctx_type = MTP so llama.cpp builds
    // the nextn graph. It keeps its own (single-position) KV, hence n_rs_seq = 0 — nothing is ever
    // rolled back on the draft side, the target is the one that speculates.
    if (cfg.spec.is_mtp()) {
        llama_context_params dparams = cparams;
        dparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        // One device cache, the target's: the draft's own expert layer is served from the host, and a
        // second cache would take as much device memory again for a head that routes one position.
        dparams.moe_cache_size = 0;
        dparams.n_rs_seq = 0;
        // The target's output cap is about the target's logits; the draft keeps llama.cpp's own,
        // which its narrow ubatch below already bounds.
        dparams.n_outputs_max = 0;
        // Compute buffers are reserved for the WIDEST ubatch, and left at the target's width the
        // draft context reserves gigabytes for a graph it never runs: the widest thing it ever
        // computes is a prefill chunk, and at decode time it is 1 + draft_max positions. On this
        // engine every MiB reserved is a MiB the expert cache does not get, and the cache is what
        // decides whether the widened verify read set is a hit or a flash read — so an oversized
        // draft context does not merely waste memory, it eats the feature's own prize. Measured on
        // the desktop host: leaving it at the target's width cost ~1 GiB of expert cache and 3
        // points of hit rate. n_batch stays as it is — the speculative driver sizes its internal
        // batch from it and must still accept a whole prefill chunk, which llama.cpp then splits
        // into ubatches of the width below.
        dparams.n_ubatch = (uint32_t) std::max(cfg.spec.draft_max + 1, SpecConfig::mtp_draft_ubatch);
        if (dparams.n_ubatch > cparams.n_ubatch) dparams.n_ubatch = cparams.n_ubatch;
        if (model_dft) {
            // A head of its own: its layer is not the streamer's, and it reads the target's hidden
            // states through the context it drafts for.
            dparams.cb_eval = nullptr;
            dparams.cb_eval_user_data = nullptr;
            dparams.ctx_other = ctx.get();
        }
        llama_context * d = llama_init_from_model(model_dft ? model_dft.get() : model.get(), dparams);
        if (!d) return o.fail("failed to create the MTP draft context");
        ctx_dft.reset(d);
        llama_set_n_threads(d, cfg.n_threads, n_threads_batch);
    }
    return true;
}

// Opt-in sampling. temp <= 0 leaves smpl null and the decode loop on argmax — the deterministic
// default the byte-identity gates rely on. temp > 0 builds the standard chain, using only the
// public llama_sampler_* API (hard rule 1): common_sampler lives in the non-stable common layer.
void Session::Impl::build_sampler(const SessionConfig & c) {
    static_assert(SamplingConfig{}.seed == LLAMA_DEFAULT_SEED,
                  "SamplingConfig::seed default must mirror LLAMA_DEFAULT_SEED");
    if (c.sampling.temp > 0.0f) {
        smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(c.sampling.top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(c.sampling.top_p, /*min_keep*/ 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(c.sampling.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(c.sampling.seed));
    }
}

// One abort callback for the session's whole life, checking two independent predicates:
// an explicit cancel() or the current caller's stop, and a fatal streaming I/O error (overlap only).
// Installing it unconditionally is what lets cancel() interrupt a serial decode too.
void Session::Impl::install_abort_callback() {
    llama_set_abort_callback(
        ctx.get(),
        [](void * ud) -> bool {
            auto * p = static_cast<Impl *>(ud);
            if (p->cancelled()) {
                // The decode this aborts is about to be logged by llama.cpp as a failure; it is a
                // request honoured. Opened once per cancelled call (the callback is asked many
                // times), closed by end_cancel_quiet() when that call returns. A fatal streaming
                // error below is a real failure and opens nothing.
                if (!p->cancel_quieted.exchange(true, std::memory_order_relaxed)) detail::llama_log_cancel_begin();
                return true;
            }
            return p->source.fatal();
        },
        this);
}

// The MTP speculative driver over the two contexts as they are now: it holds both, so a context made
// again (the device cache's, arm_device_cache) needs a driver made again.
bool Session::Impl::init_mtp_driver(Opening & o) {
    const SessionConfig & cfg = o.cfg;
    common_params_speculative sp;
    sp.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    sp.draft.n_max = cfg.spec.draft_max;
    sp.draft.n_min = 0; // never skip a whole draft; width is bounded by n_max and p_min below
    // The head's own confidence floor for continuing to draft. At 0 (the default) it drafts
    // n_max tokens however unsure it is, which is what the host was measured at; above 0 the
    // width becomes adaptive per step. See SpecConfig::draft_p_min for why that pays twice on a
    // streamed device.
    sp.draft.p_min = cfg.spec.draft_p_min;
    sp.draft.ctx_tgt = ctx.get(); // two contexts: over one model, or over the model and its head
    sp.draft.ctx_dft = ctx_dft.get();
    mtp.reset(common_speculative_init(sp, /*n_seq*/ 1));
    if (!mtp) return o.fail("failed to initialise MTP speculative decoding");
    return true;
}

// The speculative driver (MTP) and the verify batch every speculation source shares.
bool Session::Impl::init_speculation(Opening & o) {
    const SessionConfig & cfg = o.cfg;
    // Built before the capture warm-up on purpose: the driver's constructor turns on nextn
    // extraction for both contexts, which is what the target graph looks like for the rest of the
    // session. Capturing the graph it actually runs beats capturing the one it briefly had.
    if (cfg.spec.is_mtp() && !init_mtp_driver(o)) return false;

    // The wide verify batch belongs to the loop, not to a source: whoever drafted, the target is
    // handed 1 + draft_max positions in one decode. Allocated for any speculation, which is what
    // lets the n-gram source reuse the whole verify half without a draft context.
    if (cfg.spec.enabled()) {
        // One batch for the session, wide enough for the larger of its two roles.
        mtp_batch = llama_batch_init(std::max(cfg.n_batch, cfg.spec.draft_max + 1), /*embd*/ 0, /*n_seq_max*/ 1);
        mtp_batch_owned = true;
        draft_buf.reserve((size_t) cfg.spec.draft_max);
    }
    // The window a checkpoint is judged against is the attention's, and a memory with a recurrent
    // part reports that part's one position as its minimum, never the window's start: such a model
    // is judged as one without a window, as llama.cpp's server judges it. A windowed model with a
    // recurrent state otherwise never found a checkpoint it could restore.
    const bool recurrent_state = llama_model_is_recurrent(model.get()) || llama_model_is_hybrid(model.get());
    checkpoints.configure(cfg.ctx_checkpoints, cfg.checkpoint_min_step,
                          recurrent_state ? 0 : llama_model_n_swa(model.get()));
    checkpoints.set_position_map([this](int64_t n) { return kv.pos_at((size_t) n); });
    if (media) {
        media_batch = llama_batch_init(cfg.n_batch, /*embd*/ 0, /*n_seq_max*/ 1);
        media_batch_owned = true;
    }
    return true;
}

// Capture warm-up: one mmap-resident decode so the eval-callback can harvest the expert
// tensor pointers from the graph. Any valid token builds the same graph (the expert
// tensor structure is prompt-independent), so use BOS; KV is wiped afterwards.
bool Session::Impl::capture_warmup(Opening & o) {
    llama_context * c = ctx.get();
    hook->begin_capture();
    llama_token warm_tok = llama_vocab_bos(vocab);
    if (warm_tok < 0) warm_tok = 0;
    if (o.cfg.spec.is_mtp()) {
        batch_fill(mtp_batch, &warm_tok, 1, /*pos0*/ 0, /*all_logits*/ true);
        if (llama_decode(c, mtp_batch) != 0) return o.fail("capture warm-up decode failed");
        // The MTP graph is built only by a decode on the draft context, and process() is what
        // issues it. Without this pass the head's expert layer never reaches the eval callback,
        // so it would stay unbound and silently mmap-resident — the one thing streaming exists
        // to avoid on a model that does not fit.
        if (!common_speculative_process(mtp.get(), to_common_batch(mtp_batch)))
            return o.fail("MTP capture warm-up failed: the draft context could not process the batch");
    } else {
        llama_batch warm = llama_batch_get_one(&warm_tok, 1);
        if (llama_decode(c, warm) != 0) return o.fail("capture warm-up decode failed");
    }
    hook->end_capture();
    return true;
}

// Give every captured expert tensor its place in the file, and check the layers agree on how many
// experts there are.
bool Session::Impl::bind_expert_offsets(Opening & o,
                                        const GgufOffsets & offs,
                                        std::vector<LayerExperts> & layers,
                                        int & n_expert) {
    int n_bound = 0;
    for (size_t il = 0; il < layers.size(); ++il) {
        LayerExperts & L = layers[il];
        if (!L.bound) continue;
        // Half a layer on a device. The planner never places one (a pattern that moves part of a
        // layer's FFN brings all its experts to the host), so this is a placement given by hand, and
        // the streamer cannot serve it: the projections it would load with the host half are on the
        // device. Said in those terms, rather than as the missing tensor it would otherwise trip on.
        if (L.on_device)
            return o.fail("MoE layer " + std::to_string(il) +
                          " has its experts split between a device and the host, which streaming cannot "
                          "serve: in {tensor-overrides}, place all of blk." +
                          std::to_string(il) + "'s experts on one side (" +
                          tensor_names::expert_override_pattern((int) il) + " keeps them on the host)");
        ++n_bound;
        for (int p = 0; p < o.layout.n_slots; ++p) {
            ggml_tensor * t = L.proj[p].tensor;
            if (!t) return o.fail("captured MoE layer is missing expert tensor '" + o.layout.suffix[p] + "'");
            auto it = offs.off_by_name.find(t->name);
            if (it == offs.off_by_name.end()) return o.fail(std::string("no gguf offset for tensor ") + t->name);
            L.proj[p].file_off = it->second;
            L.proj[p].file_idx = offs.file_by_name.at(t->name); // same parse as the offset, so present
            const int ne2 = (int) t->ne[2];
            if (n_expert == 0)
                n_expert = ne2;
            else if (ne2 != n_expert)
                return o.fail(std::string("inconsistent expert count: tensor ") + t->name + " has " +
                              std::to_string(ne2) + ", expected " + std::to_string(n_expert));
        }
    }
    if (n_bound == 0) return o.fail("no MoE expert tensors captured — is this a MoE model?");
    return true;
}

// Hand the streamer the dense (non-expert) model weights: every captured weight leaf that IS
// a gguf tensor (dropping graph inputs and KV, which share the leaf shape) and is NOT one of
// the streamed experts. Called before init consumes `layers`. Anonymous/Pinned read these into
// their own buffers; every mode needs the list to hold back a tensor too large to be
// resident at all (qwen4exp's n-gram table), which must also leave the warm sweep and the
// residency sensor — see DenseWeights::hold_back_oversized.
//
// `unaccounted` receives the file tensors no policy owns, which decides release-mmap later.
void Session::Impl::hand_over_dense_weights(const SessionConfig & cfg,
                                            const GgufOffsets & offs,
                                            const std::vector<LayerExperts> & layers,
                                            std::vector<std::string> & unaccounted) {
    const std::unordered_set<std::string> expert_names = expert_tensor_names(layers);
    // The subset the graph only row-gathers, when the run asked for the row policy. Their
    // membership is the hook's verdict on what the graph did, and this is a filter over the
    // dense list rather than a second list, so a table cannot be both resident and streamed.
    const std::unordered_set<std::string> row_names =
        cfg.moe.row_stream ? hook->row_gathered_weights() : std::unordered_set<std::string>();
    // Walk the captured OBJECTS, not the name map: a tied output head gives two tensors
    // one name over one file range, and a name-keyed walk would rebind only one of them
    // and leave the other reading the mmap for the life of the run. Twins are folded into
    // one entry's alias list so the bytes are still read and allocated exactly once.
    std::vector<DenseTensorRef> dense, rows;
    std::unordered_map<uint64_t, size_t> dense_at; // (shard, file offset) -> index in `dense`
    for (ggml_tensor * t : hook->captured_weight_objects()) {
        if (!t) continue;
        // A weight a device owns is not ours to move, for the reason the capture gives for experts:
        // the device computes from its own buffer, so a tensor pointed at host bytes it cannot see
        // reads nothing (Metal said "buffer is nil" and the run decoded '!'). Left out, it is also
        // unowned below, which keeps release-mmap from dropping a mapping a device may still read.
        if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) continue;
        const std::string name = t->name;
        if (expert_names.count(name)) continue;
        // Only a weight the load left on the host is ours to re-home. A layer the first stage put
        // on a device keeps its dense tensors in that device's buffer - its own memory, or the file
        // mapping wrapped for it - and rebinding `data` onto a host allocation leaves the device
        // reading an address it never registered: garbage output on a unified-memory GPU, worse on
        // a discrete one. Asked of the buffer's device, so no backend is named.
        if (on_other_device(t)) continue;
        auto off = offs.off_by_name.find(name);
        auto sz = offs.size_by_name.find(name);
        if (off == offs.off_by_name.end() || sz == offs.size_by_name.end()) continue; // not a file tensor
        const int file_idx = offs.file_by_name.at(name);
        const uint64_t key = ((uint64_t) (uint32_t) file_idx << 48) ^ off->second;
        auto seen = dense_at.find(key);
        if (seen != dense_at.end()) {
            dense[seen->second].aliases.push_back(t);
            continue;
        }
        DenseTensorRef d;
        d.tensor = t;
        d.file_off = off->second;
        d.size = sz->second;
        d.file_idx = file_idx;
        dense_at.emplace(key, dense.size());
        dense.push_back(d);
        // A table with a twin is by definition also read some other way, so it cannot have
        // qualified as row-gathered; the guard costs nothing and keeps that coupling explicit.
        if (row_names.count(name)) rows.push_back(d);
    }
    for (const DenseTensorRef & d : dense)
        if (!d.aliases.empty())
            log_printf(LogLevel::Info, "'%s' is bound %zu times over one range (tied head) — all rebound",
                       d.tensor->name, d.aliases.size() + 1);
    // Every gguf tensor that is neither streamed nor in the dense list is one the capture
    // decode never touched. With speculation off that is a tensor llama.cpp did not even load
    // (load_mtp is off, so the MTP block stays in the file): nothing in this session can
    // reach it. With the MTP draft on, a second context builds a second graph, and a tensor
    // the capture missed may be one that graph reads, so the list then blocks release-mmap.
    // Only release-mmap reads this list, so only a run that asked for it pays to build it.
    if (cfg.moe.release_mmap) {
        std::unordered_set<std::string> owned;
        owned.reserve(dense.size());
        for (const DenseTensorRef & d : dense)
            if (d.tensor) owned.insert(d.tensor->name);
        for (const auto & kv : offs.off_by_name)
            if (!expert_names.count(kv.first) && !owned.count(kv.first)) unaccounted.push_back(kv.first);
    }
    const uint64_t row_budget = (uint64_t) std::max(0, cfg.moe.row_stream_mb) * 1024ull * 1024ull;
    source.set_dense_tensors(std::move(dense));
    source.set_row_tensors(std::move(rows), row_budget);
}

// Drop the model file's mapping once nothing reads through it (core/src/io/mapping_release.h).
// Safety is decided here, not in the module, and it is decided by looking rather than by
// reasoning: the release is correct only while nothing still dereferences the mapping, so the
// engine asks the OS whether any weight the graph read still points inside it. That covers
// every residency policy at once — a dense set left mmap'd, a table held back as oversized,
// and the twin a tied output head creates, which carries the same name as the embedding table
// and which no name-based accounting can see.
//
// It is a check over the pointers the capture pass observed, not a proof about the ones it
// did not: a graph shape this session never built could hold another. That residue, plus the
// gguf tensors no policy owns once the MTP draft adds a second graph, is why the flag is
// opt-in rather than on by default.
void Session::Impl::release_mapping(const SessionConfig & cfg,
                                    const GgufOffsets & offs,
                                    const std::vector<std::string> & unaccounted) {
    std::vector<const void *> weight_addrs;
    if (cfg.moe.release_mmap) {
        weight_addrs.reserve(hook->captured_weight_objects().size());
        for (const ggml_tensor * t : hook->captured_weight_objects())
            if (t && t->data) weight_addrs.push_back(t->data);
    }
    const size_t still_mapped =
        cfg.moe.release_mmap ? pio::addresses_in_file_mappings(offs.shard_paths, weight_addrs) : 0;
    if (!cfg.moe.release_mmap) return;
    if (still_mapped) {
        log_printf(LogLevel::Info, "release-mmap skipped: %zu weight(s) still read the model's mapping", still_mapped);
    } else if (!unaccounted.empty() && cfg.spec.is_mtp()) {
        log_printf(LogLevel::Info, "release-mmap skipped: %zu file tensor(s) no policy owns (first: %s)",
                   unaccounted.size(), unaccounted.front().c_str());
    } else {
        const pio::MappingReleaseReport r = pio::release_file_mappings(offs.shard_paths, &mapping_placeholders);
        // Lanes opened while the section was alive stay serialised against it even once it
        // is gone (measured: iobench S3 vs S4), so the readers are reopened after the release.
        if (r.supported && (r.views_unmapped || r.sections_closed) && !source.reopen_readers())
            log_printf(LogLevel::Warn, "release-mmap: reader reopen failed; reads stay serialised");
        if (r.supported) // POSIX has nothing to undo and says nothing
            log_printf(LogLevel::Info, "release-mmap: %d view(s) unmapped (%llu MiB), %d section(s) closed%s%s%s",
                       r.views_unmapped, (unsigned long long) (r.bytes >> 20), r.sections_closed,
                       r.plugs_missed ? ", handle slot not reclaimed" : "", r.error.empty() ? "" : "; ",
                       r.error.c_str());
    }
}

// The route trace's static block: the streamed span, its experts, and each layer's bytes.
void Session::Impl::arm_route_trace(Opening & o, int n_expert, std::vector<uint64_t> dense_bytes) {
    const SessionConfig & cfg = o.cfg;
    route_trace = o.route_trace;
    hook->set_trace(true);

    RouteTraceStatic st;
    st.model = cfg.model_path;
    st.arch = arch;
    // The streamed span, not the trunk: with speculation on, the MTP block is a layer the
    // streamer manages like any other, and a trace that stopped at the trunk would drop it.
    st.n_layer = o.n_layer_streamed;
    st.n_expert = n_expert;
    // The effective top-k: an override IS the applied width, otherwise the model's own.
    st.n_expert_used = cfg.n_expert_used;
    if (st.n_expert_used <= 0) {
        const GgufModelInfo & info = o.gguf();
        st.n_expert_used = info.ok ? info.n_expert_used : 0;
    }
    st.dense_bytes_per_layer = std::move(dense_bytes);
    st.expert_bytes_per_layer.resize((size_t) o.n_layer_streamed);
    for (int il = 0; il < o.n_layer_streamed; ++il)
        st.expert_bytes_per_layer[(size_t) il] = source.expert_bytes(il);
    o.route_trace->on_static(st);
}

// The routing record's static facts, in the streamer's own units: what the cache charges per expert of
// each block, and the top-k the routing runs at. The record itself fills during the first tokens.
void Session::Impl::arm_routing_record(Opening & o, int n_expert) {
    const SessionConfig & cfg = o.cfg;
    RoutingTrace shape;
    shape.n_layer = (uint32_t) o.n_layer_streamed;
    shape.n_expert = (uint32_t) n_expert;
    int k = cfg.n_expert_used;
    if (k <= 0) {
        const GgufModelInfo & info = o.gguf();
        k = info.ok ? info.n_expert_used : 0;
    }
    shape.n_expert_used = (uint32_t) std::max(0, k);
    // Sized from the file, not the streamer: the hook records every MoE layer's routing, and the
    // streamer knows only the layers it streams. A record sized by the streamer named no bytes for a
    // layer the plan put on a device, the reader refused the whole record, and the next load planned
    // with no hit curve.
    const GgufOffsets & offs = o.meta().offsets;
    shape.expert_bytes = expert_bytes_per_layer(o.layout, offs.extents, offs.size_by_name, o.n_layer_streamed);
    hook->set_routing_record(cfg.moe.routing_record_tokens, std::move(shape));
}

// The routing record of a run with no streamer: the hook records the routing nodes it sees until
// the record is full, each layer sized from the file as a streamed run's record is. Nothing is
// rebound and nothing is read.
bool Session::Impl::arm_unstreamed_record(Opening & o) {
    if (!capture_warmup(o)) return false;
    RoutingTrace shape;
    shape.n_layer = (uint32_t) o.n_layer_streamed;
    int k = o.cfg.n_expert_used;
    if (k <= 0) k = o.gguf().ok ? o.gguf().n_expert_used : 0;
    shape.n_expert_used = (uint32_t) std::max(0, k);
    shape.n_expert = o.gguf().ok ? (uint32_t) o.gguf().n_expert : 0;
    const GgufOffsets & offs = o.meta().offsets;
    const ExpertLayout layout = o.layout.ok ? o.layout : discover_expert_layout(offs.extents, o.gguf().n_expert);
    shape.expert_bytes = expert_bytes_per_layer(layout, offs.extents, offs.size_by_name, o.n_layer_streamed);
    hook->set_routing_record(o.cfg.moe.routing_record_tokens, std::move(shape));
    clear_memory(ctx.get()); // discard the capture's KV
    return true;
}

// The streamed experts computed on the layers' device: the context is made again with llama.cpp's
// device-side expert cache, which uploads each routed expert from the host tensor the streamer has
// just filled. The tensors and the streamer are the model's and outlive the first context; the
// warm-up's KV went with it. Serial streaming only: the overlap hook waits inside the CPU's matmul,
// which no longer runs the experts.
bool Session::Impl::arm_device_cache(Opening & o) {
    const SessionConfig & cfg = o.cfg;
    if (cfg.moe.overlap)
        return o.fail("{device-cache-mb} does not combine with {overlap}: the cache computes the experts on the "
                      "device, and {overlap} waits for them on the CPU. Turn one of them off.");
    if (prefill)
        return o.fail("{device-cache-mb} does not combine with {prefill-device}: the cache keeps the layers on "
                      "their device, and a prefill device moves them. Turn one of them off.");
    if (cfg.n_gpu_layers == 0) return o.fail("{device-cache-mb} needs layers on a device: set {gpu-layers} above 0");
    // The speculative driver holds both contexts, and both are made again: it goes first, and is
    // made again over the new ones.
    const bool had_mtp = mtp != nullptr;
    mtp.reset();
    ctx_dft.reset();
    ctx.reset();
    device_cache_armed = true;
    if (!create_contexts(o)) return false;
    install_abort_callback();
    return !had_mtp || init_mtp_driver(o);
}

// The experts a device cache uploads from, copied into a buffer of the device backend's own host
// type (MoeStreamConfig::expert_host_buffer): the same bytes in the same layout, and the expert
// tensors pointed at the copy, as the streamer points them at its slots. The device is the one the
// plan named. A device with no host type of its own, or a copy that cannot be allocated, leaves the
// mapping where it is: the copy changes how fast a miss arrives, never what. It runs before the
// device cache is armed, whose context is made again without the callback the capture needs.
bool Session::Impl::stage_experts_in_host_buffer(Opening & o) {
    // The hook holds a slot per layer from the start; a capture is what binds them.
    const std::vector<LayerExperts> & seen = hook->captured();
    const bool captured_here = std::none_of(seen.begin(), seen.end(), [](const LayerExperts & L) { return L.bound; });
    if (captured_here && !capture_warmup(o)) return false;
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(o.cfg.moe.expert_host_buffer.c_str());
    ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
    std::vector<ggml_tensor *> experts;
    for (const LayerExperts & L : hook->captured())
        for (const ExpertTensorRef & p : L.proj)
            if (p.tensor && std::find(experts.begin(), experts.end(), p.tensor) == experts.end())
                experts.push_back(p.tensor);
    if (!buft || experts.empty()) {
        log_printf(LogLevel::Warn, "expert host buffer: %s; the cache uploads from the file's mapping",
                   !dev    ? "no device of that name"
                   : !buft ? "the device's backend has no host memory of its own"
                           : "no expert tensor was found");
        if (captured_here) clear_memory(ctx.get());
        return true;
    }
    const size_t align = ggml_backend_buft_get_alignment(buft);
    size_t total = 0;
    for (ggml_tensor * t : experts)
        total += GGML_PAD(ggml_nbytes(t), align);
    // The copy goes onto every core the plan computes a prompt on: one thread copying out of the
    // mapping is bound by its own page faults (pio::copy_spans).
    const int copy_threads = std::max(o.cfg.n_threads, o.cfg.n_threads_batch);
    const auto spans_at = [&](uint8_t * base) {
        std::vector<pio::CopySpan> spans;
        size_t off = 0;
        for (ggml_tensor * t : experts) {
            spans.push_back({base + off, t->data, ggml_nbytes(t)});
            off += GGML_PAD(ggml_nbytes(t), align);
        }
        return spans;
    };
    // Each tensor's pages are handed back as soon as its copy is made, from the mapping and from the
    // system's cache of the file: the plan charged the experts once (choose_expert_staging), and that
    // has to hold while the copy is being made, not only after it. Copied whole first, the copy grew
    // at memory speed beside the pages it was read from; and unmapped alone, those pages stayed the
    // cache's, counted against a container's limit until the kernel reclaimed them, which it did not
    // do as fast as the copy grew. On a 62 GB model under a 42 GB limit the process was killed 7 GB
    // into the copy, three loads in four. Nothing reads those pages again, and a second copy (below)
    // only reads them back from the file. Aligned inward to whole pages, so a page shared with a
    // tensor still read through the mapping is never dropped (as dense_weights does for its copy).
    const GgufOffsets & file = o.meta().offsets;
    size_t handed_back = 0;
    const auto copy_handing_back = [&](const std::vector<pio::CopySpan> & spans) {
        const uintptr_t page = (uintptr_t) pio::vm_page();
        handed_back = 0;
        for (size_t i = 0; i < spans.size(); ++i) {
            const pio::CopySpan & span = spans[i];
            pio::copy_spans({span}, copy_threads);
            const uintptr_t a0 = ((uintptr_t) span.src + page - 1) & ~(page - 1);
            const uintptr_t a1 = ((uintptr_t) span.src + span.bytes) & ~(page - 1);
            if (a1 <= a0) continue;
            pio::vm_drop_file_pages((void *) a0, (size_t) (a1 - a0));
            handed_back += (size_t) (a1 - a0);
            const char * name = ggml_get_name(experts[i]);
            const auto off = file.off_by_name.find(name);
            const auto shard = file.file_by_name.find(name);
            if (off == file.off_by_name.end() || shard == file.file_by_name.end()) continue;
            if (shard->second < 0 || (size_t) shard->second >= file.shard_paths.size()) continue;
            pio::drop_file_cache_range(file.shard_paths[(size_t) shard->second].c_str(),
                                       off->second + (uint64_t) (a0 - (uintptr_t) span.src), (uint64_t) (a1 - a0));
        }
    };
    const auto point_at = [&](const std::vector<pio::CopySpan> & spans) {
        for (size_t i = 0; i < experts.size(); ++i)
            experts[i]->data = spans[i].dst;
    };
    const auto mib = [](size_t b) { return (double) b / (1024.0 * 1024.0); };

    // Memory the session allocates itself, copied into on every core and then pinned by the device's
    // backend, where the backend offers that (ggml's generic "ggml_backend_register_host_buffer"
    // entry point; the front-ends enable it, frontend/backend_env.h). A backend allocating pinned
    // memory of its own does it a page at a time on one thread: on a rented 3090, 117 GB of
    // GLM-4.7's experts took 97 s to allocate and 2 s to copy, where touching ordinary memory on 16
    // threads and pinning it took 6 s every 32 GB. Where the backend has no such entry point, or
    // declines, the backend's own host memory below, as before.
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    using RegisterFn = bool (*)(void *, size_t);
    using UnregisterFn = void (*)(void *);
    const auto do_register =
        reg ? (RegisterFn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_register_host_buffer") : nullptr;
    const auto do_unregister =
        reg ? (UnregisterFn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_unregister_host_buffer") : nullptr;
    // Asked first on one page, so a backend that declines (its entry point off, as ggml's CUDA backend
    // is unless its environment enables it) costs no copy of the experts made for nothing.
    const auto pins = [&] {
        const size_t page = pio::vm_page();
        void * p = pio::vm_reserve(page);
        const bool ok = p && pio::vm_commit(p, page) && do_register(p, page);
        if (ok) do_unregister(p);
        if (p) pio::vm_release(p, page);
        return ok;
    };
    // A copy larger than the system can still give, or than a container still lets this process take,
    // is not made, whoever would allocate it: memory allocated here is committed only as the copy
    // writes it (pio::vm_reserve) and ends in the out-of-memory killer, and a backend's own allocation
    // is pinned as it is made, which a system short of memory answers no more gracefully. The cache
    // then uploads from the mapping, as it does with no copy. Where the room cannot be read, the
    // backend's own allocation answers, which fails rather than overcommits.
    const auto room = [] {
        uint64_t r = pio::mem_available_bytes();
        const uint64_t contained = pio::cgroup_memory_left();
        if (contained > 0 && (r == 0 || contained < r)) r = contained;
        return r;
    }();
    if (room > 0 && (uint64_t) total > room) {
        log_printf(LogLevel::Warn,
                   "expert host buffer: %.0f MiB is more than the %.0f MiB the system can give; the cache uploads "
                   "from the file's mapping",
                   mib(total), mib((size_t) room));
        if (captured_here) clear_memory(ctx.get());
        return true;
    }
    const bool fits = room > 0;
    // Both entry points or neither: memory pinned that could not be unpinned would stay pinned once freed.
    if (fits && do_register && do_unregister && pins()) {
        const size_t page = pio::vm_page();
        const size_t size = (total + page - 1) / page * page;
        const auto t0 = clock_t_::now();
        void * p = pio::vm_reserve(size);
        if (p && pio::vm_commit(p, size)) {
            const std::vector<pio::CopySpan> spans = spans_at((uint8_t *) p);
            copy_handing_back(spans);
            const auto t_copied = clock_t_::now();
            if (do_register(p, size)) {
                auto held = std::make_unique<RegisteredHostMemory>();
                held->p = p;
                held->size = size;
                held->unregister = do_unregister;
                expert_host_registered = std::move(held);
                point_at(spans);
                log_printf(LogLevel::Info,
                           "expert host buffer: %zu expert tensors, %.0f MiB, copied and pinned for %s (%.1f s "
                           "copying, %.1f s pinning); %.0f MiB of the mapping handed back",
                           experts.size(), mib(total), ggml_backend_dev_name(dev), secs(t0, t_copied),
                           secs(t_copied, clock_t_::now()), mib(handed_back));
                if (captured_here) clear_memory(ctx.get()); // discard the capture's KV
                return true;
            }
        }
        // Pinning a page and not the whole copy is a driver's limit met only now; the copy is made again
        // into the backend's own memory, a few seconds against the minutes that allocation takes.
        if (p) pio::vm_release(p, size);
        log_printf(LogLevel::Warn,
                   "expert host buffer: %s did not pin the %.0f MiB allocated here; it allocates its own",
                   ggml_backend_dev_name(dev), mib(total));
    }

    const auto t_alloc = clock_t_::now();
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, total);
    const auto t_copy = clock_t_::now();
    if (!buf) {
        log_printf(LogLevel::Warn,
                   "expert host buffer: %s could not allocate %.0f MiB; the cache uploads from the "
                   "file's mapping",
                   ggml_backend_buft_name(buft), mib(total));
        if (captured_here) clear_memory(ctx.get());
        return true;
    }
    const std::vector<pio::CopySpan> spans = spans_at((uint8_t *) ggml_backend_buffer_get_base(buf));
    copy_handing_back(spans);
    point_at(spans);
    expert_host_buf.reset(buf);
    log_printf(LogLevel::Info,
               "expert host buffer: %zu expert tensors, %.0f MiB, copied into %s (%.1f s allocating, %.1f s "
               "copying); %.0f MiB of the mapping handed back",
               experts.size(), mib(total), ggml_backend_buft_name(buft), secs(t_alloc, t_copy),
               secs(t_copy, clock_t_::now()), mib(handed_back));
    if (captured_here) clear_memory(ctx.get()); // discard the capture's KV
    return true;
}

// Everything streaming adds to a loaded model: capture the expert tensors, place them in the file,
// hand the dense weights and the row tables to their policies, start the expert source, and arm
// what watches it. Ends with the warm-up's KV discarded.
bool Session::Impl::bind_streaming(Opening & o) {
    const SessionConfig & cfg = o.cfg;
    if (!capture_warmup(o)) return false;

    const GgufOffsets & offs = o.meta().offsets;
    if (!offs.ok) return o.fail("cannot read gguf offsets: " + cfg.model_path);

    std::vector<LayerExperts> layers = hook->captured();
    int n_expert = 0;
    if (!bind_expert_offsets(o, offs, layers, n_expert)) return false;

    // Computed before init consumes `layers`: the dense split needs the captured expert
    // tensor names and the gguf sizes together.
    std::vector<uint64_t> dense_bytes;
    if (o.route_trace) dense_bytes = dense_bytes_per_layer(offs, layers, o.n_layer_streamed);

    std::vector<std::string> unaccounted; // file tensors no policy owns; decides release-mmap below
    hand_over_dense_weights(cfg, offs, layers, unaccounted);

    if (prefill) prefill->keep_stream_layout(layers, offs.shard_paths);
    if (prefill && cfg.prefill.slots_on_demand) prefill->share_memory_with(&source);
    if (prefill) prefill->read_experts_from(&source);
    if (!source.init(offs.shard_paths, n_expert, std::move(layers), cfg.moe))
        return o.fail("expert stream source init failed");
    source.set_pin_floor(source.worst_cycle_bytes(n_expert_used));
    source.set_top_k(n_expert_used);
    hook->set_source(&source);
    // The row policy exists only once dense_.init has taken the tables over; null means nothing
    // qualified or the takeover declined, and the hook then costs exactly nothing per node.
    hook->set_row_source(source.row_source());

    release_mapping(cfg, offs, unaccounted);
    // llama.cpp asks for the whole mapping to be read ahead at load (init_mappings prefetches every
    // non-lazy range), so the model files fill the system's page cache with bytes this run reads
    // otherwise: the experts through the streamer's direct reads, the dense set from its own copy, the
    // tables row by row. Left there, the kernel under pressure swapped the expert cache out beside
    // them, and the cache brake then halved it twice (a 30 GiB box with swap, Qwen3.8-Flash-Next:
    // 23 GB of one shard and 21 GB of the other cached at load). Only where the dense set is held
    // apart from the mapping; under `mmap` the page cache IS the dense set.
    if (cfg.moe.dense_weights == DenseWeightsMode::Anonymous || cfg.moe.dense_weights == DenseWeightsMode::Pinned) {
        for (const std::string & path : offs.shard_paths)
            pio::drop_file_cache(path.c_str());
        log_printf(LogLevel::Info, "page cache: the model files' read-ahead handed back (%zu file(s))",
                   offs.shard_paths.size());
    }

    if (o.route_trace) arm_route_trace(o, n_expert, std::move(dense_bytes));
    arm_routing_record(o, n_expert);

    if (o.io_trace) {
        io_trace = o.io_trace;
        source.set_io_trace(true);
    }

    if (cfg.moe.overlap && !source.enable_overlap_hook())
        return o.fail("{overlap} needs the expert-ready hook of the bmoe llama.cpp fork, and the CPU backend "
                      "loaded here does not expose it: turn {overlap} off (docs/seam.md)");

    clear_memory(ctx.get()); // discard warm-up KV
    if (ctx_dft) llama_memory_clear(llama_get_memory(ctx_dft.get()), true);
    return true;
}

// Open the prefill device on the loaded session; a best-effort device that will not open is undone.
bool Session::Impl::open_prefill_device(Opening & o) {
    const SessionConfig & cfg = o.cfg;
    std::string perr;
    if (prefill->open(ctx.get(), vocab, *hook, cfg.prefill, cfg.moe, o.n_layer_streamed, perr)) return true;
    if (!cfg.prefill.best_effort) return o.fail(perr);
    hand_back_prefill(perr);
    return true;
}

// Asked for as a speed-up, not as a condition: undo it and run on the CPU. Destroying the path
// hands every weight and the model state back to where the load put them, so what is left is the
// session a run without the flag would have had.
void Session::Impl::hand_back_prefill(const std::string & why) {
    log_printf(LogLevel::Warn, "%s; prefill stays on the CPU", why.c_str());
    hook->set_device_arena(nullptr);
    hook->count_device_nodes(false);
    prefill.reset();
    cfg.prefill.device.clear();
    clear_memory(ctx.get());
}

// Confirm the device on this model (PrefillDeviceConfig::verify): the same short prefill on the
// host and on the device, compared row by row. A device that fails is handed back exactly as one
// that failed to open. The two are timed and the times printed, but not judged: at this width the
// cold expert reads dominate, and a laptop iGPU measured faster here (6.5 s against 7.5) prefilled
// a 1015-token prompt 8% slower than the host.
bool Session::Impl::confirm_prefill_device(Opening & o) {
    const SessionConfig & cfg = o.cfg;
    llama_context * c = ctx.get();
    std::string why;
    // At least as wide as the narrowest graph the device is handed, so what is confirmed is a
    // placement the device will actually see; bounded by what one graph may return.
    const int want = std::min({std::max(PrefillDeviceConfig::verify_tokens, cfg.prefill.min_tokens),
                               PrefillDeviceConfig::max_outputs, prefill_piece(this->cfg), cfg.n_ctx - 1});
    const int rows = std::min(want, PrefillDeviceConfig::verify_rows);
    double host_s = 0.0, dev_s = 0.0;
    detail::LogitAgreement cmp;
    // The sequence the whole-model answer check decodes too (device_answer.h): one text, one criterion.
    const std::vector<llama_token> probe = detail::answer_probe_tokens(vocab, want);
    // A check that cannot run is not a verdict: the device stays as the plan or the caller set it,
    // and the line below says it was not confirmed.
    std::string unchecked;
    if (probe.empty()) {
        unchecked = "the confirmation text does not tokenize";
    } else if (want < cfg.prefill.min_tokens) {
        unchecked = "no graph as wide as the narrowest the device is handed fits a confirmation";
    } else {
        llama_batch b = llama_batch_init(want, 0, 1);
        // The device computes every position; verify_rows spread positions come back to compare,
        // which keeps the output head's product narrow on the host (see verify_rows): all 64 rows
        // made a phone end the process at load.
        std::vector<int> at((size_t) rows);
        for (int r = 0; r < rows; ++r)
            at[(size_t) r] = (int) ((int64_t) (r + 1) * want / rows) - 1; // the last position included
        std::vector<float> host_logits((size_t) rows * (size_t) n_vocab);
        // 0 ran, 1 the device had no room, -1 the decode failed.
        auto pass = [&](bool device, bool keep, double & secs) -> int {
            // Two wide graphs of one shape in a row would let llama.cpp reuse the first for the
            // second although the weights moved between them, which crashes (the reason min_tokens
            // keeps the two placements' shapes apart). A one-token host graph between them breaks
            // the shape at the cost of one token.
            clear_memory(c);
            batch_fill(b, probe.data(), 1, 0, /*all_logits*/ false);
            if (decode_placed(c, b) != 0) return -1;
            clear_memory(c);
            const bool placed = place_prefill(device);
            if (device && !placed) return 1;
            batch_fill(b, probe.data(), want, 0, /*all_logits*/ false);
            for (int j : at)
                b.logits[j] = 1;
            hook->set_batch_phase(0);
            const auto t0 = std::chrono::steady_clock::now();
            const int rc = decode_placed(c, b);
            llama_synchronize(c);
            secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            place_prefill(false);
            if (rc != 0) return -1;
            if (keep)
                for (int r = 0; r < rows; ++r)
                    std::memcpy(host_logits.data() + (size_t) r * n_vocab, llama_get_logits_ith(c, at[(size_t) r]),
                                sizeof(float) * (size_t) n_vocab);
            return 0;
        };
        // The device first and untimed: its first graph pays for kernel compilation and the arena's
        // first fill, which a timed pass would charge to every prefill after it.
        double warm = 0.0;
        int rc = pass(true, false, warm);
        if (rc == 0) rc = pass(false, true, host_s);
        if (rc == 0) rc = pass(true, false, dev_s);
        if (rc == 1) {
            why = "the device had no room for the confirmation prefill";
        } else if (rc != 0) {
            why = "the confirmation prefill failed";
        } else {
            // Judged as every device check is (logit_agreement.h): the top token's agreement decides,
            // the distances are logged beside it.
            for (int r = 0; r < rows; ++r)
                cmp.add(host_logits.data() + (size_t) r * n_vocab, llama_get_logits_ith(c, at[(size_t) r]), n_vocab);
            if (!cmp.matches())
                why = "it picked the host's next token in " + std::to_string(cmp.same) + " of " + std::to_string(rows) +
                      " positions of a " + std::to_string(want) + "-token prefill: it does not compute this model";
        }
        llama_batch_free(b);
        clear_memory(c);
    }
    if (!unchecked.empty()) {
        log_printf(LogLevel::Warn, "prefill-device %s not confirmed: %s; kept as set", cfg.prefill.device.c_str(),
                   unchecked.c_str());
        return true;
    }
    log_printf(LogLevel::Info,
               "prefill-device %s confirmation: %d tokens, %d compared, top token agrees %.0f%%, logit NMSE %.2e, "
               "KL mean %.4f max %.4f, host %.2f s, device %.2f s%s%s",
               cfg.prefill.device.c_str(), want, rows, cmp.agree() * 100.0, cmp.nmse(), cmp.kld, cmp.kld_max, host_s,
               dev_s, why.empty() ? "" : " - ", why.c_str());
    if (!why.empty()) {
        prefill_check_refused = true;
        const std::string perr = cfg.prefill.device + " is not kept as the prefill device: " + why;
        if (!cfg.prefill.best_effort) return o.fail(perr);
        hand_back_prefill(perr);
    }
    return true;
}

// Decode traces. Outside the streaming step on purpose: the compute trace measures the graph,
// which exists with or without the streamer, so a dense mmap baseline can be traced and
// compared. The I/O trace was armed by bind_streaming (it needs the source) and only reports here.
void Session::Impl::arm_decode_traces(Opening & o) {
    const SessionConfig & cfg = o.cfg;
    if (!o.compute_trace && !io_trace) return;
    DecodeTraceStatic st;
    st.model = cfg.model_path;
    st.arch = arch;
    st.n_layer = n_layer;
    st.n_threads = cfg.n_threads;
    st.io_threads = cfg.moe.enabled ? cfg.moe.io_threads : 0;
    st.o_direct = cfg.moe.enabled && source.stats().o_direct; // the open's outcome, not the request
    st.overlap = cfg.moe.enabled && cfg.moe.overlap;
    if (o.compute_trace) {
        compute_trace = o.compute_trace;
        hook->set_compute_trace(true, compute_trace->layers_only());
        o.compute_trace->on_static(st);
    }
    if (io_trace) io_trace->on_static(st);
}

// What this session IS, for any metrics sink that will describe what it DOES. Built here, where
// every fact is resolved: cache_mb in particular is what the streamer settled on, which under
// auto-sizing is a number no flag ever mentioned. Then the effective routing width.
void Session::Impl::describe_run(Opening & o) {
    const SessionConfig & cfg = o.cfg;
    {
        RunInfo & ri = info;
        const std::string & p = cfg.model_path;
        const size_t slash = p.find_last_of("/\\");
        ri.model = slash == std::string::npos ? p : p.substr(slash + 1);
        ri.arch = arch;
        ri.n_layer = n_layer;
        ri.engine_version = version();
        ri.n_threads = cfg.n_threads;
        ri.n_ctx = cfg.n_ctx;
        // The width the context was opened with, never the 0 that left it to the default.
        ri.n_ubatch = ctx ? (int) llama_n_ubatch(ctx.get()) : cfg.n_ubatch;
        ri.cache_type_k = kv_cache_type_name(this->cfg.cache_type_k); // what was allocated, after any fallback
        ri.cache_type_v = kv_cache_type_name(this->cfg.cache_type_v);
        ri.chatml = cfg.chatml;
        ri.compute_trace_layers = compute_trace && compute_trace->layers_only();
        ri.spec = cfg.spec.is_mtp() ? "mtp" : cfg.spec.is_ngram() ? "ngram" : "off";
        ri.spec_draft_max = cfg.spec.enabled() ? cfg.spec.draft_max : 0;
        ri.mtp_p_min = cfg.spec.is_mtp() ? cfg.spec.draft_p_min : 0.0f;
        ri.ngram_min_match = cfg.spec.is_ngram() ? cfg.spec.ngram_min_match : 0;
        ri.temp = cfg.sampling.temp;
        ri.top_k = cfg.sampling.top_k;
        ri.top_p = cfg.sampling.top_p;
        ri.seed = cfg.sampling.seed;
        ri.moe_stream = cfg.moe.enabled;
        ri.cache_auto = cfg.moe.cache_auto;
        ri.cache_floor_mb = cfg.moe.cache_floor_mb;
        ri.cache_ceil_mb = cfg.moe.cache_ceil_mb;
        ri.force_cache = cfg.moe.force_cache;
        ri.load_all = cfg.moe.enabled && cfg.moe.load_all;
        ri.io_threads = cfg.moe.enabled ? cfg.moe.io_threads : 0;
        ri.overlap = cfg.moe.enabled && cfg.moe.overlap;
        ri.io_two_wave = cfg.moe.enabled && cfg.moe.io_two_wave;
        ri.prefetch_layers = cfg.moe.enabled ? cfg.moe.prefetch_layers : 0;
        ri.route_ahead = cfg.moe.enabled ? cfg.moe.route_ahead : 0;
        ri.predict_prefetch = cfg.moe.enabled && cfg.moe.predict_prefetch;
        ri.predict_log = cfg.moe.enabled && cfg.moe.predict_log;
        ri.predict_spec_max = cfg.moe.enabled ? cfg.moe.predict_spec_max : 0;
        ri.prefetch_sync = cfg.moe.enabled && cfg.moe.prefetch_sync;
        ri.drop_cold_frac = cfg.moe.enabled ? cfg.moe.drop_cold_frac : 0.0f;
        ri.drop_renorm = cfg.moe.drop_renorm;
        ri.drop_prefill = cfg.moe.drop_prefill;
        ri.substitute_lambda = cfg.moe.enabled ? cfg.moe.substitute_lambda : 0.0f;
        ri.dense_weights = dense_weights_mode_name(cfg.moe.dense_weights);
        if (cfg.moe.enabled) {
            const IExpertSource::Stats st = source.stats();
            ri.cache_mb = (int) (st.cache_budget_bytes / (1024ull * 1024ull));
            // o_direct from the same sample: whether the shard readers actually got cache bypass
            // (O_DIRECT honoured, or F_NOCACHE applied on Apple), never what the flag asked for.
            ri.o_direct = st.o_direct;
        }
        // The EFFECTIVE top-k: an override IS the applied width, otherwise the model's own. Same
        // resolution the route trace does, and worth a header read — a run whose top-k is unknown
        // cannot be compared against one whose top-k differs, which is most of the point.
        ri.n_expert_used = n_expert_used;
        const GgufModelInfo & mi = o.gguf();
        if (mi.ok) ri.n_expert = mi.n_expert;
        // Last, because it needs both numbers above: the budget the streamer settled on and the
        // width that was actually applied. Rounded UP — a cycle reported as smaller than it is
        // would put a budget on the wrong side of the cliff in exactly the borderline case.
        ri.cache_cycle_mb =
            (int) ((source.worst_cycle_bytes(ri.n_expert_used) + 1024ull * 1024ull - 1) / (1024ull * 1024ull));
    }
}

// ── the knob trial (KnobTrialConfig, knob_trial.h) ──────────────────────────────────────────
// Which knobs this session measures, decided once it is open: a knob whose switch this build or
// this run cannot make is left out and said so, and the trial measures what is left.
void Session::Impl::arm_trials() {
    const KnobTrialConfig & t = cfg.trial;
    threads_now = cfg.n_threads;
    {
        // The decode's cost, timed as the rate probe times a side: the trial's windows, once the cache
        // has settled. Where a trial runs first it has already waited for the cache; a device expert
        // cache fills out of this session's counters, so the probe then waits as long as the trial
        // ever does.
        KnobTrialPolicy pol;
        pol.window_tokens = t.window_tokens;
        const bool device_cache = cfg.moe.device_cache_mb > 0;
        trials.cost_probe =
            std::make_unique<RateProbe>(!t.enabled() || device_cache, /*hits_visible*/ !device_cache, pol);
    }
    // The requests are recorded against the narrow width on every load whose plan has the width's
    // trade, a side timed or not (note_width_turn).
    if (t.width_narrow > 0) trials.result.width_narrow = t.width_narrow;
    if (!t.enabled()) return;
    if (!t.threads.empty()) {
        trials.threads.push_back(cfg.n_threads);
        trials.threads.insert(trials.threads.end(), t.threads.begin(), t.threads.end());
    }
    if (t.overlap) {
        if (!cfg.moe.enabled)
            log_printf(LogLevel::Warn, "knob trial: overlap is not measured without expert streaming");
        else if (!cfg.moe.overlap && !expert_ready_hook_available())
            log_printf(LogLevel::Warn, "knob trial: overlap is not measured: the CPU backend loaded here has no "
                                       "expert-ready hook");
        else
            trials.overlap = {cfg.moe.overlap ? 1 : 0, cfg.moe.overlap ? 0 : 1};
    }
    trials.rate = t.device_tier >= 0 || t.width_side >= 0 || t.cpu_share_side >= 0 || t.shared_device_side >= 0;
    trials.stage = stage_done;
    next_trial_stage(/*settled*/ false);
}

// The next stage after the current one, started: a knob trial, the rate probe, or done. `settled`:
// a stage before it already waited for the cache.
void Session::Impl::next_trial_stage(bool settled) {
    const int from = trials.stage == stage_done ? -1 : trials.stage;
    trials.cur.reset();
    trials.rate_probe.reset();
    if (from < stage_threads && !trials.threads.empty())
        trials.stage = stage_threads;
    else if (from < stage_overlap && !trials.overlap.empty())
        trials.stage = stage_overlap;
    else if (from < stage_rate && trials.rate)
        trials.stage = stage_rate;
    else {
        trials.stage = stage_done;
        return;
    }
    if (trials.stage == stage_rate) {
        KnobTrialPolicy pol;
        pol.window_tokens = cfg.trial.window_tokens;
        // A device expert cache fills inside llama.cpp, out of this session's counters: where it is
        // armed, the probe waits as long as the trial ever waits for a cache.
        const bool device_cache = cfg.moe.device_cache_mb > 0;
        trials.rate_probe = std::make_unique<RateProbe>(!settled || device_cache, /*hits_visible*/ !device_cache, pol);
        return;
    }
    start_trial_stage(/*settle*/ !settled);
}

void Session::Impl::start_trial_stage(bool settle) {
    KnobTrialPolicy pol;
    pol.window_tokens = cfg.trial.window_tokens;
    const size_t arms = trials.stage == stage_threads ? trials.threads.size() : trials.overlap.size();
    trials.cur = std::make_unique<KnobTrial>((int) arms - 1, settle, pol);
}

void Session::Impl::set_decode_threads(int n) {
    if (n == threads_now) return;
    // Prefill keeps the count it had: the trial measures decode, and a prompt is a different load.
    const int n_batch = cfg.n_threads_batch > 0 ? cfg.n_threads_batch : cfg.n_threads;
    llama_set_n_threads(ctx.get(), n, n_batch);
    if (ctx_dft) llama_set_n_threads(ctx_dft.get(), n, n_batch);
    threads_now = n;
}

// False when the switch could not be made; the stage is then abandoned on its incumbent.
bool Session::Impl::apply_trial_arm(int arm) {
    if (trials.stage == stage_threads) {
        set_decode_threads(trials.threads[(size_t) arm]);
        return true;
    }
    if (trials.stage == stage_overlap) {
        const bool on = trials.overlap[(size_t) arm] != 0;
        if (source.overlap() == on || source.set_overlap(on)) return true;
        log_printf(LogLevel::Warn, "knob trial: overlap could not be switched %s; it is not measured",
                   on ? "on" : "off");
        source.set_overlap(cfg.moe.overlap);
        next_trial_stage(/*settled*/ true);
        return false;
    }
    return true; // the rate probe switches nothing
}

void Session::Impl::snapshot_trial_counters() {
    if (!cfg.moe.enabled) return;
    const IExpertSource::Stats st = source.stats();
    trials.lookups_seen = (uint64_t) st.cache_lookups;
    trials.hits_seen = (uint64_t) st.cache_hits;
}

void Session::Impl::resume_trial() {
    const bool costing = trials.cost_probe && !trials.cost_probe->done();
    if (trials.stage == stage_done && !costing) return;
    snapshot_trial_counters(); // the prompt's lookups are not the first window's
    if (trials.stage == stage_done) return;
    if (trials.stage == stage_rate) return;
    apply_trial_arm(trials.cur->arm());
}

void Session::Impl::pause_trial() {
    if (trials.stage == stage_done) {
        if (trials.cost_probe) trials.cost_probe->interrupt();
        return;
    }
    if (trials.stage == stage_rate) {
        trials.rate_probe->interrupt();
        return;
    }
    trials.cur->interrupt();
    apply_trial_arm(0);
}

// The side of a decision across loads this load runs (the device tier, the graph width, the device
// cache's CPU share or the shared-memory device), timed; its rate goes in the result, which the
// front-end keeps, and the plan of a later load reads it against the other side's.
void Session::Impl::rate_step(double seconds, int tokens, uint64_t lookups, uint64_t hits) {
    RateProbe & r = *trials.rate_probe;
    const bool settling = r.settling();
    r.record(seconds, tokens, lookups, hits);
    const bool tier = cfg.trial.device_tier >= 0;
    char what[96];
    if (tier)
        std::snprintf(what, sizeof(what), "%s a device expert cache", cfg.trial.device_tier == 1 ? "with" : "without");
    else if (cfg.trial.cpu_share_side >= 0)
        std::snprintf(what, sizeof(what), "%s",
                      cfg.trial.cpu_share_side == 1 ? "with part of the device cache's misses on the CPU"
                                                    : "with every device cache miss uploaded");
    else if (cfg.trial.shared_device_side >= 0)
        std::snprintf(what, sizeof(what), "with every layer on %s",
                      cfg.trial.shared_device_side == 1 && !cfg.device_split.empty()
                          ? cfg.device_split.front().first.c_str()
                          : "the CPU");
    else
        std::snprintf(what, sizeof(what), "with the graph %s (%d tokens wide)",
                      cfg.trial.width_side == 1 ? "at the plan's width" : "narrowed",
                      cfg.trial.width_side == 1 ? (int) llama_n_ubatch(ctx.get()) : cfg.trial.width_narrow);
    if (settling && !r.settling()) log_printf(LogLevel::Info, "knob trial: timing the decode %s", what);
    if (!r.done()) return;
    const double tps = r.tokens_per_second();
    record_side_rate(cfg.trial, tps, trials.result);
    log_printf(LogLevel::Info, "knob trial: %.2f tok/s %s, the median of %d windows", tps, what, r.windows());
    trials.rate_timed = true;
    next_trial_stage(/*settled*/ true);
}

// A turn of a load whose plan has the width's trade (KnobTrialConfig::width_narrow): what it asked of
// the model, for the plan that weighs the two widths, and, where the load times a side, the prefill of
// a prompt the width changes on that side. Recorded with no side timed too: a width decided on the
// decode alone, before any request was seen, is weighed again on the requests (planner.cpp,
// width_across_loads).
void Session::Impl::note_width_turn(int n_prompt, double prefill_seconds, int n_generated) {
    const KnobTrialConfig & t = cfg.trial;
    if (t.width_narrow <= 0) return;
    ++trials.width_requests;
    trials.width_gen += n_generated;
    if (n_prompt > t.width_narrow && prefill_seconds > 0.0) {
        trials.width_long_tokens += n_prompt;
        trials.width_long_seconds += prefill_seconds;
    }
    MeasuredKnobs & r = trials.result;
    r.width_narrow = t.width_narrow;
    // A load that times no side runs the width its plan decided: the one a later decision's margin
    // guards (MeasuredKnobs::width_held). A side timed is a trial, not a decision, and holds nothing.
    if (t.width_side < 0) r.width_held = (int) llama_n_ubatch(ctx.get()) <= t.width_narrow ? 0 : 1;
    r.usage_requests = trials.width_requests;
    r.usage_gen_tokens = trials.width_gen / trials.width_requests;
    r.usage_long_prompt_tokens = trials.width_long_tokens / trials.width_requests;
    if (t.width_side >= 0 && trials.width_long_seconds > 0.0)
        (t.width_side == 1 ? r.wide_prefill_tps : r.narrow_prefill_tps) =
            trials.width_long_tokens / trials.width_long_seconds;
}

// A turn, as this machine's use: what a later plan prices a trade between the prompt and the answer
// by (MeasuredKnobs::mix_*). Recorded on every load, since the trade is decided before one is made.
void Session::Impl::note_request_mix(const RunSummary & sum) {
    ++trials.mix_requests;
    trials.mix_gen += sum.n_generated;
    // A prompt narrower than the narrowest graph a device is handed runs on the host either way, so
    // it is no part of the trade.
    if (sum.n_prompt >= cfg.prefill.min_tokens) {
        trials.mix_wide += sum.n_prompt;
        if (sum.prefill_device_tokens == 0 && sum.prefill_seconds > 0.0) {
            trials.host_wide_tokens += sum.n_prompt;
            trials.host_wide_seconds += sum.prefill_seconds;
        }
    }
    MeasuredKnobs & r = trials.result;
    r.mix_requests = trials.mix_requests;
    r.mix_gen_tokens = trials.mix_gen / trials.mix_requests;
    r.mix_wide_prompt_tokens = trials.mix_wide / trials.mix_requests;
    if (trials.host_wide_seconds > 0.0) r.host_prefill_tps = trials.host_wide_tokens / trials.host_wide_seconds;
    // The side this load serves is kept at what the load decoded at, over every turn so far, in place
    // of the probe's windows. The probe times a few dozen tokens early in the first answer, which is
    // where a load happens to be then: after a knob trial on one load and from the first token on the
    // next, with the host's cache cold. Measured on two 16 GB cards with a 284B model: the fitter's
    // placement was timed at 2.27 tok/s on a load whose answers decoded at 4.1 and 6.5, against a
    // device cache timed at 3.17 on a load that decoded at 2.8 and 3.6, and the plan after kept the
    // slower side. Both sides are a whole load's rate, so they are compared like for like.
    trials.served_tokens += sum.n_generated;
    trials.served_seconds += sum.gen_seconds;
    if (trials.rate && trials.rate_timed && trials.served_seconds > 0.0 && trials.served_tokens > 0.0) {
        const double tps = trials.served_tokens / trials.served_seconds;
        record_side_rate(cfg.trial, tps, r);
        if (!trials.served_rate_said) {
            trials.served_rate_said = true;
            log_printf(LogLevel::Info,
                       "knob trial: this load's side is kept at what it decoded at, %.2f tok/s over %.0f tokens so far",
                       tps, trials.served_tokens);
        }
    }
}

void Session::Impl::trial_step(double seconds, int tokens) {
    if (trials.stage == stage_done) return;
    uint64_t lookups = 0, hits = 0;
    if (cfg.moe.enabled) {
        const uint64_t l0 = trials.lookups_seen, h0 = trials.hits_seen;
        snapshot_trial_counters();
        lookups = trials.lookups_seen - l0;
        hits = trials.hits_seen - h0;
    }
    if (trials.stage == stage_rate) {
        rate_step(seconds, tokens, lookups, hits);
        return;
    }
    KnobTrial & k = *trials.cur;
    const bool settling = k.settling();
    k.record(seconds, tokens, lookups, hits);
    if (settling && !k.settling())
        log_printf(LogLevel::Info, "knob trial: the cache has settled; measuring %s",
                   trials.stage == stage_threads ? "the decode thread count" : "overlap");
    if (const int dead = k.last_abandoned()) {
        const std::vector<double> & w = k.window_seconds_per_token();
        const size_t n = w.size();
        const std::string what = trials.stage == stage_threads
                                     ? std::to_string(trials.threads[(size_t) dead]) + " decode threads"
                                     : std::string("overlap ") + (trials.overlap[(size_t) dead] ? "on" : "off");
        if (k.last_abandoned_cut()) {
            // Cut inside its window: the candidate's window is the last one closed, and the bar it
            // passed is the incumbent's, measured in this trial.
            log_printf(LogLevel::Info,
                       "knob trial: %s settled as no win inside its first window: %.0f ms/token, where the "
                       "window could spend %.0f a token and still stand inside the incumbent's range",
                       what.c_str(), 1e3 * w[n - 1], 1e3 * k.apart_bar());
        } else {
            const int r = k.rounds(dead);
            // Apart: every token of its window past the incumbent's whole range, so one round decided it.
            const char * why = k.last_abandoned_apart() ? ", every token past the incumbent's range" : "";
            log_printf(LogLevel::Info,
                       "knob trial: %s settled as no win after %d round%s%s: %.0f ms/token against %.0f", what.c_str(),
                       r, r == 1 ? "" : "s", why, 1e3 * w[n - 2], 1e3 * w[n - 1]);
        }
    }
    if (k.done()) {
        finish_trial_stage();
        return;
    }
    apply_trial_arm(k.arm());
}

// A decode step once the trial is done, timed for the decode's own cost: its compute (the CSV's
// compute_ms: the wall less the flash waits and the cache's management) and the cache's hits. The
// result carries it with the configuration it ran under, for the next plan (MeasuredKnobs).
void Session::Impl::cost_step(double compute_seconds, int tokens) {
    RateProbe * r = trials.cost_probe.get();
    if (!r || r->done()) return;
    uint64_t lookups = 0, hits = 0;
    if (cfg.moe.enabled) {
        const uint64_t l0 = trials.lookups_seen, h0 = trials.hits_seen;
        snapshot_trial_counters();
        lookups = trials.lookups_seen - l0;
        hits = trials.hits_seen - h0;
    }
    r->record(compute_seconds, tokens, lookups, hits);
    if (!r->done()) return;
    const double tps = r->tokens_per_second();
    if (!(tps > 0.0)) return;
    MeasuredKnobs & k = trials.result;
    k.decode_compute_s = 1.0 / tps;
    k.decode_threads = threads_now;
    k.decode_hit = cfg.moe.enabled ? r->hit_rate() : -1.0;
    k.decode_cache_mib = cfg.moe.enabled ? (double) source.cache_budget() / (1024.0 * 1024.0) : 0.0;
    k.decode_gpu_layers = cfg.n_gpu_layers;
    k.decode_device_cache_mb = cfg.moe.device_cache_mb;
    if (k.decode_hit >= 0.0)
        log_printf(LogLevel::Info,
                   "decode cost: %.1f ms/token of compute at %d thread(s), %.0f%% cache hits at %.0f MiB, the median "
                   "of %d windows",
                   1e3 * k.decode_compute_s, k.decode_threads, 100.0 * k.decode_hit, k.decode_cache_mib, r->windows());
    else
        log_printf(LogLevel::Info, "decode cost: %.1f ms/token of compute at %d thread(s), the median of %d windows",
                   1e3 * k.decode_compute_s, k.decode_threads, r->windows());
}

// A stage's verdict: the winner is applied and stays, the result records it, and the next knob
// starts on the cache this one already settled.
void Session::Impl::finish_trial_stage() {
    KnobTrial & k = *trials.cur;
    const int w = k.winner();
    const int arms = trials.stage == stage_threads ? (int) trials.threads.size() : (int) trials.overlap.size();
    int best = 1;
    for (int c = 2; c < arms; ++c)
        if (k.gain(c) > k.gain(best)) best = c;
    const double gain = k.gain(w ? w : best);
    if (!apply_trial_arm(w)) return;
    char verdict[160];
    if (w)
        std::snprintf(verdict, sizeof(verdict), "decoded %.1f%% faster in the median of %d rounds",
                      100.0 * (gain - 1.0), k.rounds(w));
    else
        std::snprintf(verdict, sizeof(verdict),
                      "held: the best candidate was %+.1f%% in the median of %d rounds, "
                      "inside the %.0f%% margin",
                      100.0 * (gain - 1.0), k.rounds(best), 100.0 * KnobTrialPolicy{}.min_win);
    trials.result.rounds = k.rounds(w ? w : best);
    if (trials.stage == stage_threads) {
        trials.result.threads = trials.threads[(size_t) w];
        trials.result.threads_tried = trials.threads;
        trials.result.threads_gain = gain;
        log_printf(LogLevel::Info, "knob trial: %d decode threads (against %d): %s", trials.threads[(size_t) w],
                   trials.threads[0], verdict);
    } else {
        trials.result.overlap = trials.overlap[(size_t) w];
        trials.result.overlap_incumbent = trials.overlap[0];
        trials.result.overlap_gain = gain;
        log_printf(LogLevel::Info, "knob trial: overlap %s (against %s): %s", trials.result.overlap ? "on" : "off",
                   trials.overlap[0] ? "on" : "off", verdict);
    }
    next_trial_stage(/*settled*/ true);
}

// The two configurations that are legal and still worth a word at load, said once.
void Session::Impl::warn_about_load(const SessionConfig & cfg) {
    // Cache-aware dropping on a narrow routing.
    //
    // The threshold is a fraction of the uniform share 1/top-k, so what it removes scales with how
    // wide the routing is. At top-k 8 (where it was measured) it trims a long tail; at top-k 2 the
    // same fraction can discard the whole minority expert on every miss, which is closer to halving
    // the routing than to trimming it. The engine has no way to know whether that is acceptable for
    // a given model, so it states the fact rather than clamping the flag — a silent adjustment
    // would be worse than a loud caveat.
    if (cfg.moe.enabled && cfg.moe.drop_cold_frac > 0.0f) {
        const int k = n_expert_used;
        if (k > 0 && k <= MoeStreamConfig::drop_low_topk_warn)
            log_printf(LogLevel::Warn,
                       "WARNING drop-cold-experts=%.2f with top-k %d — the threshold is %.1f%% of the "
                       "routing here, against 12.5%% at the top-k 8 this was measured on. Expect it to discard "
                       "much more, and check output quality on your own task.",
                       (double) cfg.moe.drop_cold_frac, k, 100.0 * cfg.moe.drop_cold_frac / k);
    }

    // A cache budget below one token's worst-case cycle.
    //
    // The number is computable from the model's shape alone, so this costs nothing and is available
    // at the only moment it can still be acted on. It is a warning, not a rejection: the budget is
    // legal, the run is byte-correct, and cache_min_mb already rejects the band it was drawn for.
    // What this catches is the case that number cannot see — a budget comfortably above the fixed
    // floor and still under THIS model's cycle, which --n-expert-used widens without touching the
    // budget. Below the cycle LRU keeps no entry to the next token, so it pays management and RAM
    // for a cache that cannot hit; decayed-frequency eviction (the default) keeps what recent tokens
    // shared and still hits, but well under what one more cycle of budget would give. The engine
    // states the fact; what to do about it is docs/cache-sizing.md's business.
    if (info.cache_mb > 0 && info.cache_cycle_mb > 0 && info.cache_mb < info.cache_cycle_mb) {
        if (cfg.moe.cache_freq_half_life > 0)
            log_printf(LogLevel::Warn,
                       "WARNING expert cache %d MiB is below this model's worst-case token cycle of %d MiB "
                       "at top-k %d — a token's experts cannot all stay, so expect a low hit rate.",
                       info.cache_mb, info.cache_cycle_mb, info.n_expert_used);
        else
            log_printf(LogLevel::Warn,
                       "WARNING expert cache %d MiB is below this model's worst-case token cycle of %d MiB "
                       "at top-k %d — no entry can survive to the next token, so expect a hit rate near zero.",
                       info.cache_mb, info.cache_cycle_mb, info.n_expert_used);
    }
}

// What a load's answer check came to, for open() to act on once the placement it judged is gone.
struct Session::AnswerCheck {
    bool ran = false;
    detail::DeviceAnswer result;
    // The device computes the model and the cores serve a request sooner (weigh_answer_sides).
    bool cores_faster = false;
};

namespace {
// The two sides as the check timed them, kept where a plan reads a side's decode rate: both at once,
// from one load, so the plan after this one decides on the model's own measurement.
void keep_answer_rates(const detail::DeviceAnswer & a, MeasuredKnobs & rec) {
    if (a.device_token_s > 0.0) rec.shared_device_tps = 1.0 / a.device_token_s;
    if (a.host_token_s > 0.0) rec.no_shared_device_tps = 1.0 / a.host_token_s;
}
} // namespace

std::unique_ptr<Session> Session::open(const SessionConfig & cfg,
                                       std::string & error,
                                       IRouteTraceSink * route_trace,
                                       IComputeTraceSink * compute_trace,
                                       IIoTraceSink * io_trace) {
    using Verdict = detail::DeviceAnswer::Verdict;
    const std::string & device = cfg.trial.answer_device;
    const auto t0 = clock_t_::now();
    AnswerCheck check;
    std::unique_ptr<Session> self = open_placed(cfg, error, route_trace, compute_trace, io_trace, &check);
    if (!check.ran) {
        if (self || !cfg.fallback || device.empty()) return self;
        // The model could not be opened on the device at all, most likely for memory: the plan's other
        // part serves, and the device has lost this model here (a second try at a load that does not
        // fit is a second failure on a machine already short of room).
        log_printf(LogLevel::Warn,
                   "the model could not be opened on %s (%s); the plan's other configuration is opened instead",
                   device.c_str(), error.c_str());
        const double first_s = secs(t0, clock_t_::now());
        const std::string first_error = error;
        self = open_placed(*cfg.fallback, error, route_trace, compute_trace, io_trace, nullptr);
        if (!self) {
            error = "the model could not be opened on " + device + " (" + first_error + "), nor as the plan's other " +
                    "configuration: " + error;
            return nullptr;
        }
        self->impl_->trials.result.shared_device_unviable = MeasuredKnobs::unviable_max;
        self->impl_->load_seconds += first_s;
        return self;
    }
    if (check.result.verdict == Verdict::Confirmed && !check.cores_faster) {
        if (self) {
            self->impl_->trials.result.set_device_answer(device, 1);
            keep_answer_rates(check.result, self->impl_->trials.result);
        }
        return self;
    }
    // Not confirmed, and the placement is already freed (open_placed returned nothing): the session
    // serves from the cores. A sequence that differed is a verdict, and is kept. A check that could
    // not run is not one, and nothing is kept, but the model does not stay on the device either: a
    // device whose answers for this model were never compared with the cores' is exactly what the
    // check exists to keep from serving, and "could not compare" has not compared them. The next load
    // asks again.
    const double first_s = secs(t0, clock_t_::now());
    // The plan's own other part where it has one (SessionConfig::fallback), else this same
    // configuration on the cores.
    self = open_placed(cfg.fallback ? *cfg.fallback : detail::on_cores(cfg), error, route_trace, compute_trace,
                       io_trace, nullptr);
    if (!self) {
        error = "the model was not kept on " + device +
                (check.cores_faster                         ? ", which the CPU decodes it faster than"
                 : check.result.verdict == Verdict::Differs ? ", whose answers for it differed from the CPU's"
                                                            : ", whose answers for it could not be compared with the "
                                                              "CPU's") +
                ", and it could not be opened on the CPU either: " + error;
        return nullptr;
    }
    if (check.result.verdict == Verdict::Differs) self->impl_->trials.result.set_device_answer(device, -1);
    // The device computes the model and the cores are faster: its answers are confirmed all the same,
    // and both rates are kept, so the next plan opens the cores without opening the device first.
    if (check.cores_faster) {
        self->impl_->trials.result.set_device_answer(device, 1);
        keep_answer_rates(check.result, self->impl_->trials.result);
    }
    self->impl_->load_seconds += first_s; // the load the caller waited for is both
    return self;
}

// One opening of `cfg` as it stands. Where it asks for the device's answers to be compared
// (KnobTrialConfig::answer_device) and `check` is given, the comparison runs once the context exists,
// and an answer that is not confirmed ends this opening with nothing: the caller opens the cores.
std::unique_ptr<Session> Session::open_placed(const SessionConfig & cfg,
                                              std::string & error,
                                              IRouteTraceSink * route_trace,
                                              IComputeTraceSink * compute_trace,
                                              IIoTraceSink * io_trace,
                                              AnswerCheck * check) {
    // Create the session first so its Impl destructor owns backend teardown from this point
    // on: any failure below returns nullptr, destroying `self`, which frees the backend once.
    std::unique_ptr<Session> self(new Session());
    Impl & im = *self->impl_;
    im.cfg = cfg;

    // Before anything reaches llama.cpp: a session opened without the planner (a gate, an embedder)
    // still gets llama.cpp's DEBUG lines dropped and a cancel not reported as a decode failure.
    detail::install_llama_log();
    detail::set_llama_log_debug(cfg.llama_debug_log);
    // llama_backend_init/free is process-global and reference counted; init here, free in ~Impl.
    llama_backend_init();
    im.backend_inited = true;

    const auto t_load0 = clock_t_::now();
    Impl::Opening o{cfg, error, route_trace, compute_trace, io_trace};

    // The order is load-bearing; each step's comment says what it needs from the ones before.
    if (!im.load_model(o)) return nullptr;
    detail::log_memory_snapshot("model-loaded");
    if (!im.read_model(o)) return nullptr;
    if (!im.build_hook(o)) return nullptr;
    if (!im.create_contexts(o)) return nullptr;
    if (check && !cfg.trial.answer_device.empty()) {
        check->ran = true;
        check->result = im.compare_device_answer(o);
        if (check->result.verdict != detail::DeviceAnswer::Verdict::Confirmed) {
            error = "the device's answers for this model were not confirmed";
            return nullptr;
        }
        if (im.weigh_device_answer(check->result).side == detail::AnswerSide::Cores) {
            check->cores_faster = true;
            error = "the cores serve this model's requests faster than the device";
            return nullptr;
        }
        detail::log_memory_snapshot("device-answer-checked");
    }
    im.build_sampler(cfg);
    im.install_abort_callback();
    if (!im.init_speculation(o)) return nullptr;
    if (cfg.moe.enabled && !im.bind_streaming(o)) return nullptr;
    if (records_unstreamed(cfg) && !im.arm_unstreamed_record(o)) return nullptr;
    if (!cfg.moe.expert_host_buffer.empty() && !im.stage_experts_in_host_buffer(o)) return nullptr;
    if (cfg.moe.device_cache_mb > 0 && !im.arm_device_cache(o)) return nullptr;
    if (im.prefill && !im.open_prefill_device(o)) return nullptr;
    if (im.prefill && cfg.prefill.verify && !im.confirm_prefill_device(o)) return nullptr;
    im.arm_decode_traces(o);
    im.arm_trials();
    // What came of the prefill device asked for, for the record of this model (MeasuredKnobs::prefill):
    // kept; refused where its check on this model refused it; or not kept, without a verdict on the
    // device, where it would not open (an allocation another program's memory can fail for a moment):
    // that counts as one more load that did not keep it, not as a refusal.
    if (!cfg.prefill.device.empty())
        im.trials.result.set_prefill_word({cfg.prefill.device, im.prefill ? 1 : im.prefill_check_refused ? -1 : 0, 0});
    im.describe_run(o);
    im.warn_about_load(cfg);

    im.load_seconds = secs(t_load0, clock_t_::now());
    detail::log_memory_snapshot("session-ready");
    return self;
}

// Before a turn touches the context: say what the run is, reset the cancel latch, drop a kept
// decide() prefix, and — on a new chat — the conversation.
void Session::Impl::begin_generate(const GenerateRequest & req, IMetricsSink * sink) {
    // Before anything is written about what this run did, say what it was.
    if (sink && !info_sent) {
        sink->on_run_info(info);
        info_sent = true;
    }

    // Fresh cancel latch for this generation; a stale request from a prior aborted call must
    // not carry over. (cancel() sets it; the abort callback reads it.)
    cancel_requested.store(false, std::memory_order_relaxed);

    // A kept decide() prefix is worth its RAM only while decisions follow one another. Back in a
    // conversation it would sit, until close, in the memory the expert cache lives in, so it goes
    // now; the next decide() builds a fresh one.
    decide_cache.reset();
    decide_cache_chosen = false;

    // clear_kv = "new chat": drop the KV and the engine-held conversation. Otherwise this turn
    // continues the conversation, reusing the KV prefix already decoded from earlier turns.
    if (req.clear_kv) {
        clear_memory(ctx.get());
        // The draft context tracks the target's positions and must be dropped with it, or the first
        // draft of the new conversation is conditioned on the previous one.
        if (ctx_dft) llama_memory_clear(llama_get_memory(ctx_dft.get()), true);
        chat_history.clear();
        forget_kv_tokens();
        // A new chat resets the sampler RNG, so a fixed seed reproduces the same transcript from a
        // fresh conversation. A continued turn (clear_kv=false) keeps the stream going, matching the
        // KV it decodes against.
        if (smpl) llama_sampler_reset(smpl);
    }
}

// One generate() call: everything the turn decides, measures and may have to undo, and the steps
// that run it in order — render the prompt, tokenize it, reuse the KV prefix, prefill, decode,
// account, and commit (or roll back) the conversation. A step that ends the turn early returns
// false with `res` already holding the answer: an error kind, or a cancellation.
struct Session::Impl::Turn {
    Impl & im;
    const GenerateRequest & req;
    RunResult & res;
    const MoeStreamConfig & moe;
    llama_context * ctx;

    // ── the prompt ──
    std::string prompt;
    bool chat_on;
    bool history_pushed = false;                 // did we append this turn's user message to chat_history?
    bool history_replaced = false;               // did we swap chat_history for the request's messages?
    std::vector<common_chat_msg> history_before; // what a replaced conversation goes back to
    bool prefilled_answer = false;               // closed the reasoning span in the prompt (parsed all the same)
    common_chat_parser_params parse_params;
    detail::ToolGrammar grammar;        // the template's grammar, off when it has none
    std::vector<llama_token_data> cand; // the full candidate list, built only when the grammar refuses
    // The prompt as tokens, images included as spans of placeholders (prompt_stream.h); `tokens`
    // is the flat list the text paths read, the same storage.
    detail::MediaInput::Tokenized tk;
    std::vector<llama_token> & tokens = tk.stream.tokens;
    std::vector<ChatImage> images; // the turn's, in marker order
    int n_prompt = 0;
    int n_predict = 0;
    size_t n_common = 0; // leading prompt tokens the KV already held
    // Where each user message starts in `tokens`, from the template's own message delimiters, which
    // llama.cpp reads off the template by rendering variants of it: the prefill stops there for a
    // checkpoint (ContextCheckpoints::Stops). Empty when the template gave no delimiters.
    common_chat_msg_delimiters message_delimiters;
    std::vector<size_t> user_starts;

    // The frame the I/O rows are stamped with at flush; the other traces carry their own.
    int trace_phase = 0, trace_step = 0;

    // ── prefill ──
    PrefillTally prefill_tally;
    // Two predicates, deliberately distinct. spec_on is "the verify loop runs" — a wide batch, an
    // accept pass, a rollback — and both sources need all of it. mtp_on is "the draft comes from the
    // head", which is the only thing that needs the second context and llama.cpp's `common`
    // speculative driver. Conflating them is what would make the n-gram source pay for a draft
    // context it never uses.
    bool spec_on = false;
    bool mtp_on = false;
    int prefill_device_tokens = 0;
    uint64_t arena_read0 = 0;
    uint64_t arena_cache0 = 0;
    double arena_stall0 = 0.0;
    long long prefill_device_nodes = 0;
    double prefill_seconds = 0.0;
    const float * logits = nullptr;

    // ── decode ──
    std::string gen;
    int n_gen = 0;
    double gen_seconds = 0.0;
    double last_compute_s = 0.0; // the last decode's compute, what the cache brake reads
    GenTally tally;
    long long prev_spec_bytes = 0, prev_spec_experts = 0, prev_spec_useful = 0;
    long long prev_cache_lookups = 0, prev_cache_hits = 0; // at the start of the decode
    long long prev_routed = 0, prev_dropped = 0, prev_reranked = 0, prev_substituted = 0;
    long long prev_ra_issue_ns = 0, prev_ra_wd_ns = 0;
    clock_t_::time_point loop_mark;
    double loop_overhead_s = 0.0;
    llama_pos n_past = 0;
    std::vector<llama_token> mtp_ctx;
    std::vector<llama_token> verify_toks; // [confirmed token, drafts...] for the verify batch
    std::vector<llama_token> confirmed;   // what one decode confirmed, in order
    long long mtp0_drafted = 0, mtp0_accepted = 0, mtp0_decodes = 0, mtp0_drafted_steps = 0;
    double mtp0_draft_s = 0.0;
    uint64_t mtp0_draft_bytes = 0;
    llama_token tok = 0;
    common_chat_msg final_msg;
    bool final_parsed = false;

    // One decode of the loop: the confirmed token and whatever was drafted after it, verified in
    // one batch, and what that decode cost.
    struct Step {
        int n_draft = 0;
        double draft_s = 0.0; // this group's drafting + catch-up
        bool wide = false;
        double overhead = 0.0; // everything since the previous decode returned
        clock_t_::time_point s0, s1;
        uint64_t f0 = 0, f1 = 0;
        double c0 = 0.0, c1 = 0.0;
        int n_acc = 0;
        bool eog_hit = false;
    };

    Turn(Impl & im_, const GenerateRequest & req_, RunResult & res_)
        : im(im_), req(req_), res(res_), moe(im_.cfg.moe), ctx(im_.ctx.get()), prompt(req_.prompt),
          chat_on(im_.chat_on) {}

    bool fail(RunError kind, std::string msg) {
        res.ok = false;
        res.error = std::move(msg);
        res.error_kind = kind;
        return false;
    }

    bool structured() const { return !req.json_schema.empty(); }
    // Held to a grammar that is a contract, not a guide: a response schema or the request's own GBNF.
    bool constrained() const { return structured() || !req.grammar.empty(); }

    // ── this request's sampling (GenerateRequest::sampling) ──
    struct SamplerFree {
        void operator()(llama_sampler * p) const {
            if (p) llama_sampler_free(p);
        }
    };
    std::unique_ptr<llama_sampler, SamplerFree> turn_chain;
    bool own_sampling = false; // the request's sampling stands for the session's this turn
    // The chain this turn samples with; null is argmax.
    llama_sampler * sampler() const { return own_sampling ? turn_chain.get() : im.smpl; }

    // Build the request's chain over the session's settings, before the first token is sampled. The
    // penalties look back over the prompt too, as llama.cpp's server feeds them. False for a request
    // the vocabulary cannot honour.
    bool arm_sampler() {
        if (!req.sampling) return true;
        const RequestSampling & r = *req.sampling;
        if (im.cfg.spec.enabled()) {
            if (!im.request_sampling_logged) {
                im.request_sampling_logged = true;
                log_printf(LogLevel::Info, "a request's sampling is not applied while speculation is on: a draft "
                                           "is accepted by the model's own argmax, so the session decodes greedily");
            }
            return true;
        }
        for (const auto & b : r.logit_bias)
            if (b.first < 0 || b.first >= im.n_vocab) {
                rollback_turn();
                return fail(RunError::InvalidRequest,
                            "logit_bias names token " + std::to_string(b.first) + ", outside the model's vocabulary");
            }
        const SamplingConfig & d = im.cfg.sampling;
        const float temp = r.temp.value_or(d.temp);
        const bool penalties = r.repeat_penalty != 1.0f || r.frequency_penalty != 0.0f || r.presence_penalty != 0.0f;
        own_sampling = true;
        if (temp <= 0.0f && !penalties && r.logit_bias.empty()) return true; // argmax
        llama_sampler * c = llama_sampler_chain_init(llama_sampler_chain_default_params());
        turn_chain.reset(c);
        if (!r.logit_bias.empty()) {
            std::vector<llama_logit_bias> bias;
            for (const auto & b : r.logit_bias)
                bias.push_back({b.first, b.second});
            llama_sampler_chain_add(c, llama_sampler_init_logit_bias(im.n_vocab, (int32_t) bias.size(), bias.data()));
        }
        const int last_n = r.penalty_last_n < 0 ? im.cfg.n_ctx : r.penalty_last_n;
        if (penalties)
            llama_sampler_chain_add(c, llama_sampler_init_penalties(im.n_vocab, last_n, r.repeat_penalty,
                                                                    r.frequency_penalty, r.presence_penalty));
        if (temp <= 0.0f) {
            llama_sampler_chain_add(c, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(c, llama_sampler_init_top_k(r.top_k.value_or(d.top_k)));
            llama_sampler_chain_add(c, llama_sampler_init_top_p(r.top_p.value_or(d.top_p), /*min_keep*/ 1));
            if (r.min_p.value_or(0.0f) > 0.0f) llama_sampler_chain_add(c, llama_sampler_init_min_p(*r.min_p, 1));
            llama_sampler_chain_add(c, llama_sampler_init_temp(temp));
            llama_sampler_chain_add(c, llama_sampler_init_dist(r.seed.value_or(d.seed)));
        }
        if (penalties)
            for (int i = std::max(0, n_prompt - last_n); i < n_prompt; ++i)
                if (tokens[(size_t) i] >= 0) llama_sampler_accept(c, tokens[(size_t) i]);
        return true;
    }

    // ── logprobs (GenerateRequest::n_probs) ──
    PositionLogprobs tok_probs;                    // for `tok`, taken when it was sampled
    std::vector<PositionLogprobs> confirmed_probs; // parallel to `confirmed`

    // ── stop text (GenerateRequest::stop) ──
    bool stop_hit = false;
    detail::AnswerStart answer_start; // a stop is matched in the answer, never in the reasoning before it

    bool sample_next(int32_t row) {
        try {
            tok = sample(row);
            if (tok == LLAMA_TOKEN_NULL || (constrained() && !grammar.active())) {
                rollback_turn();
                return fail(constrained() ? RunError::InvalidOutput : RunError::Failed,
                            "the response grammar could not select a valid token");
            }
            return true;
        } catch (const std::exception & e) {
            rollback_turn();
            return fail(constrained() ? RunError::InvalidOutput : RunError::Failed,
                        std::string("sampling failed: ") + e.what());
        }
    }

    // Put the conversation back as it was before this turn touched it. Every way out that is not a
    // finished turn goes through here, or the next turn renders a history this one never answered.
    void undo_history() {
        if (history_replaced) im.chat_history = std::move(history_before);
        if (history_pushed) im.chat_history.pop_back();
        history_replaced = history_pushed = false;
    }

    // Format the prompt. With chat on, render the model's OWN chat template (real Jinja) over the
    // WHOLE conversation so far, and set up reasoning parsing so a thinking model's internal
    // reasoning is stripped from the shown answer. req.think drives enable_thinking, per prompt.
    bool render_prompt() {
        // A message list replaces the engine-held conversation instead of adding one user turn to it.
        const bool replace = !req.messages.empty();
        if (replace && !chat_on)
            return fail(RunError::NeedsChatMode, "a message list needs chat turns: open the session with {chatml} on");
        // The images, in the order the rendered prompt places their markers: a message list renders
        // its messages in order, so their images concatenate.
        if (replace) {
            for (const ChatMessage & m : req.messages)
                images.insert(images.end(), m.images.begin(), m.images.end());
        } else {
            images = req.images;
        }
        if (!images.empty() && !im.media)
            return fail(RunError::NoVision, "the request carries " + std::to_string(images.size()) +
                                                " image(s) and this model was loaded without a projector: load it "
                                                "with {mmproj} (the mmproj file of the same conversion) to read them");
        if (!req.grammar.empty() && (!chat_on || !req.tools.empty() || structured()))
            return fail(RunError::InvalidRequest,
                        !chat_on
                            ? "a grammar needs chat turns: open the session with {chatml} on"
                            : "a grammar cannot be combined with tools or a response format, which build their own");
        if (!chat_on) {
            prompt = req.prompt;
            return true;
        }
        try {
            if (replace) {
                history_before = std::move(im.chat_history);
                im.chat_history = detail::to_common_messages(req.messages);
                history_replaced = true;
            } else {
                common_chat_msg user_msg;
                user_msg.role = "user";
                user_msg.content = req.prompt;
                im.chat_history.push_back(user_msg);
                history_pushed = true;
            }

            // The full conversation, not just this turn. The reasoning span and the turn header a
            // prefilled turn resumes after are rendered by llama.cpp's own handler for this
            // template, so no marker for any family is spelled out here. See thinking_control.h.
            common_chat_templates_inputs inputs;
            // A grammar holds the answer from its first token, so the model is asked not to reason first.
            prefilled_answer =
                detail::build_turn_inputs(inputs, im.chat_history, req.think && req.grammar.empty(), im.think_ctl);
            // The tools go to the template, which writes them in the syntax this model was trained
            // on, and the parser built from the same call reads the model's calls back out.
            if (replace) detail::add_tools(inputs, req.tools, req.tool_choice);

            inputs.json_schema = req.json_schema;
            common_chat_params cp = common_chat_templates_apply(im.chat_tmpls.get(), inputs);
            prompt = cp.prompt.str();
            parse_params = detail::build_parse_params(cp);
            message_delimiters = cp.message_delimiters;
            message_delimiters.tokenize(im.vocab);
            // The request's own grammar stands for the template's: whole, from the answer's first
            // token, which is where the generation prompt ends.
            if (!req.grammar.empty()) {
                cp.grammar = req.grammar;
                cp.grammar_lazy = false;
                cp.grammar_triggers.clear();
                cp.generation_prompt.clear();
            }
            // An explicit response format also constrains a prefilled answer: the template
            // owns the grammar and its generation prefix, including any reasoning envelope.
            std::string grammar_err;
            if (!prefilled_answer || constrained()) {
                const bool ready = grammar.init(cp, im.vocab, grammar_err);
                if (constrained() && (!ready || !grammar.active())) {
                    undo_history();
                    // grammar_err speaks of the template's grammar; this one is the request's.
                    if (!req.grammar.empty())
                        return fail(RunError::InvalidRequest, "the grammar does not parse as GBNF with a root rule");
                    return fail(RunError::TemplateRejected,
                                "the template cannot apply the response schema: " +
                                    (grammar_err.empty() ? "no grammar was produced" : grammar_err));
                }
                if (!ready) log_printf(LogLevel::Warn, "tool-call grammar not applied (%s)", grammar_err.c_str());
            }
        } catch (const std::exception & e) {
            // A lone prompt can still be fed raw. A message list cannot: there is no single string
            // to fall back to, and a conversation the template refuses is the caller's to fix.
            const bool was_replace = history_replaced;
            undo_history();
            if (was_replace || structured())
                return fail(RunError::TemplateRejected,
                            std::string("the chat template rejected the conversation: ") + e.what());
            log_printf(LogLevel::Warn, "chat template apply failed (%s); using raw prompt", e.what());
            chat_on = false;
        }
        return true;
    }

    // Tokenize the prompt and settle the answer's budget against the room the context has left.
    bool tokenize_prompt() {
        if (!images.empty()) {
            // Text and images together: the projector's tokenizer splits the prompt at its markers
            // and sizes each image in tokens and positions. The text-only path below stays as it
            // was, byte for byte, for every prompt without an image.
            std::string err;
            bool invalid = false;
            if (!im.media->tokenize(prompt, images, /*add_special*/ true, tk, err, invalid)) {
                undo_history();
                // Nothing was decoded yet: a request at fault (a marker count, an image the
                // projector cannot read or one too large) leaves the session usable.
                return fail(invalid ? RunError::InvalidRequest : RunError::Failed, err);
            }
            // An image that must go in one pass and is wider than the graph: refused here, with the
            // context untouched, rather than failing half way through the prefill.
            const std::string too_wide = im.media->width_refusal(tk, (int) llama_n_ubatch(im.ctx.get()));
            if (!too_wide.empty()) {
                undo_history();
                return fail(RunError::InvalidRequest, too_wide);
            }
            n_prompt = (int) tokens.size();
        } else {
            tokens.resize(prompt.size() + k_tokenize_slack);
            n_prompt = llama_tokenize(im.vocab, prompt.c_str(), (int) prompt.size(), tokens.data(), (int) tokens.size(),
                                      /*add_special*/ true, /*parse_special*/ true);
            if (n_prompt < 0) {
                tokens.resize(-n_prompt);
                n_prompt = llama_tokenize(im.vocab, prompt.c_str(), (int) prompt.size(), tokens.data(),
                                          (int) tokens.size(), true, true);
            }
            if (n_prompt > 0) tokens.resize(n_prompt);
        }
        if (n_prompt < 1) {
            undo_history();
            return fail(RunError::EmptyPrompt, "empty prompt after tokenization");
        }
        // n_predict is a ceiling on the answer, not a reservation: the answer gets what the context
        // has left after the prompt, and stops there with the budget spent, as a length stop. Only a
        // prompt that leaves no room at all is refused. Reserving the whole ceiling refused a
        // two-word prompt whenever the ceiling was set near the context size (#138).
        const int ctx_room = im.cfg.n_ctx - n_prompt - k_turn_end_reserve;
        if (ctx_room < 1) {
            undo_history();
            return fail(RunError::ContextFull,
                        "the conversation (" + std::to_string(n_prompt) + " tokens) is longer than {ctx-size} (" +
                            std::to_string(im.cfg.n_ctx) +
                            " tokens): start a new conversation, or load the model with a larger {ctx-size}");
        }
        // No ceiling at all (0) is the context's room: the budget then grows with the context.
        n_predict = req.n_predict > 0 ? std::min(req.n_predict, ctx_room) : ctx_room;
        user_starts.clear();
        if (chat_on && !message_delimiters.delimiters.empty()) {
            // An image's placeholders are no tokens a delimiter could match: stepped over, as the server does.
            std::map<size_t, size_t> skips;
            for (const detail::MediaSpan & m : tk.stream.media)
                skips[m.begin] = m.n_tokens;
            for (const common_chat_msg_span & sp : message_delimiters.split(tokens, skips).spans)
                if (sp.role == COMMON_CHAT_ROLE_USER) user_starts.push_back(sp.pos);
        }
        return true;
    }

    // The text to surface: with chat on, parse the raw output so a reasoning model's internal
    // thinking is separated from the answer. The answer is shown inline; the reasoning is handed to
    // the UI as a distinct thinking block rather than dropped, so a Thinking-on run does not sit on a
    // blank screen while the model reasons. Generation always uses the raw tokens.
    struct ShownView {
        std::string content;   // the answer, reasoning stripped
        std::string reasoning; // the thinking span, empty unless the parser split one out
        // The calls written so far, read by the same parse: whichever tool and whatever format the
        // template uses, a call that is still being written is visible (TokenMetrics::tool_calls).
        std::vector<ChatToolCall> tool_calls{};
    };
    // The empty reasoning span a prefill closed is the prompt's, not the model's: a parse that reads
    // the prefill back finds it, and it is no reasoning to show.
    void forget_prefilled_reasoning(common_chat_msg & msg) const {
        if (prefilled_answer && msg.reasoning_content.find_first_not_of(" \t\r\n") == std::string::npos)
            msg.reasoning_content.clear();
    }
    ShownView shown_view(const std::string & raw, bool partial) const {
        if (!chat_on) return {raw, ""};
        // A prefilled turn resumes mid-answer, and is read by the same parse: llama.cpp hands the
        // parser the prefill as the generation prompt (continue_final_message), so the turn header it
        // anchors on is there, and what the format closes the answer with is cut as in any turn. The
        // raw stream is the answer only where that close is an end-of-generation token; where it is
        // text (a block that ends with a marker the vocabulary does not stop on), it showed.
        try {
            common_chat_msg msg = common_chat_parse(common_chat_input(raw), partial, parse_params);
            forget_prefilled_reasoning(msg);
            if (prefilled_answer && !structured() && detail::parsed_nothing(msg) && !raw.empty() && !partial)
                return {raw, ""};
            return {msg.content, msg.reasoning_content, detail::from_common_tool_calls(msg.tool_calls)};
        } catch (const std::exception & e) {
            if (structured()) return {}; // never leak an unparsed reasoning/tool envelope as JSON
            detail::warn_parse_failed_once(e.what());
            return {raw, ""};
        }
    }

    // Reuse the KV prefix already decoded from earlier turns (chat mode only): find how many
    // leading tokens still match the cache, drop the divergent tail, and prefill only the suffix.
    // Keeping at least one token to decode means a turn is never a no-op. clear_kv leaves
    // kv_tokens empty, so n_common = 0 and this reduces to a full prefill — the one-shot path the
    // byte-identity gates exercise stays unchanged.
    void reuse_kv_prefix() {
        if (!chat_on || im.kv.empty()) return;
        const size_t max_common = tokens.size() > 0 ? tokens.size() - 1 : 0;
        n_common = im.kv.common_prefix(tk.stream, max_common);
        // A memory that cannot be cut here (a recurrent state, a sliding window) goes back to the
        // newest checkpoint before the divergence, and only a context with none starts again.
        if (n_common < im.kv.size()) n_common = im.keep_kv_prefix(n_common);
        // The draft context mirrors the target's positions, so it has to be rewound to the SAME
        // point — including when n_common did not move, since the previous turn left it holding
        // everything it generated. Miss this and the first prefill batch of the turn starts at a
        // position the draft context already has, which llama.cpp rejects outright: the second
        // message of a conversation fails while the first always works.
        if (im.ctx_dft && !llama_memory_seq_rm(llama_get_memory(im.ctx_dft.get()), 0, im.kv.pos_at(n_common), -1))
            llama_memory_clear(llama_get_memory(im.ctx_dft.get()), true);
    }

    // Roll this turn back to the state before it started: drop the KV added this turn, forget the
    // tokens we fed, and un-append the user message. Used on cancel so prior turns stay usable.
    void rollback_turn() {
        if (chat_on) {
            // A cleared context cannot leave a reusable prefix in the bookkeeping: n_common follows it.
            n_common = im.keep_kv_prefix(n_common);
            undo_history();
        } else {
            im.clear_memory(ctx);
        }
        // Whatever the target rolled back to, the draft context follows: a cancelled turn that left
        // the two at different positions would fail the NEXT turn, not this one.
        if (im.ctx_dft) {
            const llama_pos keep = chat_on ? im.kv.pos_at(n_common) : 0;
            if (!llama_memory_seq_rm(llama_get_memory(im.ctx_dft.get()), 0, keep, -1)) {
                llama_memory_clear(llama_get_memory(im.ctx_dft.get()), true);
                // Both contexts must resume at the same prefix after an aborted turn.
                im.clear_memory(ctx);
                im.forget_kv_tokens();
                n_common = 0;
            }
        }
    }

    // Route trace: frame one decode's rows, then hand them to the sink once it has returned —
    // never from inside the callback, which runs on a compute thread mid-graph. `base_pos` is
    // the context position of the batch's first token, so prefill rows carry real step numbers.
    void trace_begin(int base_pos, int n_tokens, int phase) {
        // Not a trace concern, but the same per-decode frame: the drop policy is decode-only
        // unless armed for prefill, so it has to be told which phase this batch is.
        im.hook->set_batch_phase(phase);
        if (im.route_trace) im.hook->begin_trace_batch(base_pos, n_tokens, phase, im.turn);
        im.hook->begin_record_batch(phase, n_tokens);
        // A node is computed once for the whole batch, not per token, so a prefill chunk's graph is
        // attributed to its last position rather than pretending to split across the chunk.
        if (im.compute_trace) im.hook->begin_compute_batch(base_pos + n_tokens - 1, phase, im.turn);
        trace_phase = phase;
        trace_step = base_pos + n_tokens - 1;
    }
    void trace_flush() {
        // Close the compute trace's dangling interval FIRST: at layer granularity the "post" row
        // is charged the wall since the last boundary, and everything trace_flush does before the
        // close would be billed to the LM head.
        if (im.compute_trace) im.hook->end_compute_batch();
        im.hook->end_record_batch();
        if (im.route_trace) {
            im.hook->end_trace_batch();
            std::vector<RouteTraceRow> & rows = im.hook->trace_rows();
            if (!rows.empty()) im.route_trace->on_rows(rows.data(), rows.size());
            rows.clear();
        }
        if (im.compute_trace) {
            std::vector<ComputeTraceRow> & rows = im.hook->compute_rows();
            if (!rows.empty()) im.compute_trace->on_rows(rows.data(), rows.size());
            rows.clear();
        }
        if (im.io_trace) {
            // The reads carry no frame of their own — a lane does not know which token it serves —
            // so stamp them with the decode they were drained after.
            im.source.take_io_trace_rows(im.io_rows_scratch);
            for (IoTraceRow & r : im.io_rows_scratch) {
                r.turn = im.turn;
                r.phase = trace_phase;
                r.step = trace_step;
            }
            if (!im.io_rows_scratch.empty()) im.io_trace->on_rows(im.io_rows_scratch.data(), im.io_rows_scratch.size());
            im.io_rows_scratch.clear();
        }
    }

    // ── prefill (chunked by n_batch; positions auto-continue from the reused prefix) ──
    // The caller holds the PlacementGuard for the rest of the turn.
    bool prefill() {
        // Prefill attribution (#173): the cumulative counters are pinned before the prompt chunks so
        // their deltas across prefill carry the same wall-additive terms the decode phase reports.
        // The session layer already holds everything needed; the streamer is untouched.
        prefill_tally.begin(moe.enabled, im.source);
        const auto t_prefill0 = clock_t_::now();
        spec_on = im.cfg.spec.enabled();
        mtp_on = im.mtp != nullptr;
        if (tk.stream.has_media()) im.media_in_conversation = true;
        if (im.media_in_conversation && (spec_on || mtp_on)) {
            if (!im.media_spec_logged) {
                im.media_spec_logged = true;
                log_printf(LogLevel::Info, "speculation is off for a conversation holding an image: the draft "
                                           "reads tokens, and an image's are embeddings. It is back with a new "
                                           "conversation");
            }
            spec_on = false;
            mtp_on = false;
        }
        // With a prefill device the prompt goes in one ubatch per decode, so each graph's width is one
        // we chose and can place: wide pieces on the device, a narrow tail on the CPU. Left to
        // llama_decode, a batch would be split into ubatches whose last one could be one token wide —
        // the shape of the decode graph that follows, which llama.cpp would then reuse without
        // re-scheduling it.
        const int pf_step = im.prefill ? prefill_piece(im.cfg) : im.cfg.n_batch;
        arena_read0 = im.prefill ? im.prefill->arena_read_bytes() : 0;
        arena_cache0 = im.prefill ? im.prefill->arena_cache_bytes() : 0;
        arena_stall0 = im.prefill ? im.prefill->arena_stall_seconds() : 0.0;
        const long long device_nodes0 = im.hook->device_nodes();
        const int turn = ++im.checkpoint_turn;
        const detail::ContextCheckpoints::Stops stops{&user_starts, n_prompt, pf_step, (int) llama_n_ubatch(ctx)};
        size_t next_media = 0; // the first span at or after i
        // Where the prefill stands, for a front-end to show while no token exists yet.
        const auto progress = [&](int i) {
            if (req.on_prefill)
                req.on_prefill(
                    PrefillProgress{n_prompt, (int) n_common, i - (int) n_common, secs(t_prefill0, clock_t_::now())});
        };
        progress((int) n_common);
        for (int i = (int) n_common; i < n_prompt;) {
            while (next_media < tk.stream.media.size() && tk.stream.media[next_media].begin < (size_t) i)
                ++next_media;
            const detail::MediaSpan * span =
                next_media < tk.stream.media.size() && tk.stream.media[next_media].begin == (size_t) i
                    ? &tk.stream.media[next_media]
                    : nullptr;
            // A text run ends where the next image begins; an image goes in whole, as its own graphs.
            const int run_end =
                next_media < tk.stream.media.size() ? (int) tk.stream.media[next_media].begin : n_prompt;
            int chunk = span ? (int) span->n_tokens : std::min(pf_step, run_end - i);
            // A text piece stops where a checkpoint is due (a user message's start, the prompt's last
            // tokens), and the checkpoint is taken at the start of the piece, before it is decoded.
            if (chat_on && !span) chunk = (int) (im.checkpoints.next_stop(stops, i, chunk) - i);
            if (chat_on && i > (int) n_common && im.checkpoints.wanted_at(stops, i, chunk))
                im.checkpoints.create(im.checkpoint_contexts(), i, turn,
                                      !user_starts.empty() && (size_t) i == user_starts.front());
            if (im.prefill) {
                const bool on_dev = im.place_prefill(chunk >= im.cfg.prefill.min_tokens);
                if (on_dev) prefill_device_tokens += chunk;
            }
            trace_begin(i, chunk, /*phase*/ 0);
            int rc;
            llama_batch pf{};
            if (span) {
                // The image: encoded by the projector, decoded as embeddings at the positions the
                // projector gives it, through the same placed decode as a text batch.
                rc = im.media->decode_media(ctx, tk, next_media, tk.stream.pos_at((size_t) i), pf_step,
                                            [&](llama_batch_ext * b) { return im.decode_placed_ext(ctx, b); });
            } else {
                if (mtp_on) {
                    // Positions are absolute here — the prompt token at index i sits at position i,
                    // reused prefix included — which is exactly what llama_batch_get_one would have
                    // inferred.
                    batch_fill(im.mtp_batch, tokens.data() + i, chunk, /*pos0*/ i, /*all_logits*/ false);
                    pf = im.mtp_batch;
                } else if (tk.stream.has_media() || im.kv.has_media()) {
                    // After an image the positions no longer equal the indices: spelled out.
                    batch_fill(im.media_batch, tokens.data() + i, chunk, tk.stream.pos_at((size_t) i), false);
                    pf = im.media_batch;
                } else {
                    pf = llama_batch_get_one(tokens.data() + i, chunk);
                }
                rc = im.decode_placed(ctx, pf);
            }
            i += chunk;
            if (rc != 0) {
                if (im.cancelled()) {
                    rollback_turn();
                    res.ok = true;
                    res.cancelled = true;
                    return false;
                }
                if (im.source.fatal()) return fail(RunError::Failed, "expert stream I/O failed during prefill");
                return fail(RunError::Failed, "prefill decode failed");
            }
            trace_flush();
            progress(i);
            // The end of an image is a prompt end's equal: it is what costs the most to redo, and the
            // next turn can diverge a few tokens past it (a template drops a past turn's empty think
            // block), where a memory that cannot be cut falls back to the newest checkpoint before the
            // divergence. Without one here that was the one before the image, and the image was
            // encoded and prefilled again on every turn after it.
            if (chat_on && span) im.checkpoints.create(im.checkpoint_contexts(), i, turn);
            // The draft context has to walk the prompt too: its KV must reach the last prompt position
            // or the first draft is made from a hidden state that never saw the prompt. It is a second
            // context on the host's backends, so the weights go home for its walk, slots kept, and the
            // next wide chunk takes them out again.
            if (mtp_on && im.prefill) im.place_prefill(false, /*hold_slots*/ true);
            if (mtp_on && !common_speculative_process(im.mtp.get(), to_common_batch(pf)))
                return fail(RunError::Failed, "MTP draft context failed to process the prefill batch");
        }
        im.place_prefill(false); // decode reads the host bindings
        prefill_device_nodes = im.hook->device_nodes() - device_nodes0;
        // The suffix is now in the KV; record it so the next turn can diff against it.
        if (chat_on) im.kv.append(tk.stream, n_common, (size_t) n_prompt);
        // The end of this prompt is where the next request most often diverges: a client sends the
        // answer back as text, and the template's tokens for it need not be the ones sampled.
        if (chat_on && im.checkpoints.create(im.checkpoint_contexts(), n_prompt, turn) && !im.checkpoint_size_logged) {
            im.checkpoint_size_logged = true;
            log_printf(LogLevel::Info, "context checkpoints: %.1f MiB each, up to %d kept",
                       (double) im.checkpoints.bytes() / im.checkpoints.count() / (1024.0 * 1024.0),
                       im.cfg.ctx_checkpoints);
        }
        // Announced only once the prompt is in both contexts: begin() checks how far the draft context
        // has actually got, so calling it before prefill would warn about a gap that is about to close.
        if (mtp_on) common_speculative_begin(im.mtp.get(), /*seq_id*/ 0, tokens);
        llama_synchronize(ctx); // the last chunk may still be queued on a device, as in the decode bracket
        prefill_seconds = secs(t_prefill0, clock_t_::now());
        // Prefill attribution, closed here (the begin() snapshot sits above the chunk loop): deltas
        // of the cumulative counters across the prompt, the quantities the decode phase reports per
        // token. Read with the same rules: io is summed lane busy time under overlap, stall is the
        // union of stalled intervals, cpu is whole-process (upper bound on compute-thread time).
        prefill_tally.end(moe.enabled, im.source);
        logits = llama_get_logits_ith(ctx, -1);
        return true;
    }

    // The decode phase's baselines, taken where prefill left off, and the first token.
    bool begin_decode() {
        // Baseline seeded from prefill's closing sample: the summary reports the generation phase
        // only, from real per-token deltas (prefill routes near the whole bank, so folding it into a
        // per-token average would badly inflate the flash-I/O figure). Nothing runs between the two
        // phases, so end()'s snapshot IS the decode baseline — one reading of the counters at the
        // boundary instead of two that could drift apart. In a warm session these counters carry the
        // prior prompts' totals; the deltas make each prompt self-relative.
        tally.overlap = moe.enabled && im.source.overlap();
        if (moe.enabled) {
            const IExpertSource::Stats & st0 = prefill_tally.post;
            tally.prev_bytes = (long long) st0.read_bytes;
            tally.prev_io_s = st0.read_seconds;
            tally.prev_mgmt_s = st0.mgmt_seconds;
            tally.prev_stall_s = st0.stall_seconds;
            tally.prev_drain_s = st0.drain_wait_seconds;
            tally.prev_adopt_s = st0.adopt_wait_seconds;
        }
        const IExpertSource::Stats st_spec0 = moe.enabled ? im.source.stats() : IExpertSource::Stats{};
        prev_spec_bytes = (long long) st_spec0.spec_read_bytes;
        prev_spec_experts = st_spec0.spec_experts;
        prev_spec_useful = st_spec0.spec_useful;
        prev_cache_lookups = st_spec0.cache_lookups;
        prev_cache_hits = st_spec0.cache_hits;
        // Taken after prefill, so the drop counters describe generation — the phase the policy is
        // armed for and the one the tok/s number is about.
        prev_routed = im.hook->experts_routed();
        prev_dropped = im.hook->experts_dropped();
        prev_reranked = im.hook->experts_reranked();
        prev_substituted = im.hook->experts_substituted();
        // Per-token cursors for the hook's own eval-thread meters (route-ahead issue + watchdog): the
        // hook accumulates for the session, the rows want this token's share.
        prev_ra_issue_ns = im.hook->route_ahead_issue_ns();
        prev_ra_wd_ns = im.hook->route_ahead_wd_ns();

        // The decode bracket below measures llama_decode and nothing else, which is what makes
        // compute_ms a clean residual — but it also means everything BETWEEN two decodes (sampling,
        // detokenization, rendering, the sinks) is charged to nobody and disappears from tok/s. Mark
        // where the last decode ended so each token can report the gap it actually waited through.
        loop_mark = clock_t_::now();

        // Absolute position the next decoded token occupies. Prefill left the context filled up to
        // n_prompt-1; without speculation this simply tracks n_prompt + n_gen, but a verify decode
        // advances it by a whole accepted group, so it is carried explicitly.
        n_past = tk.stream.pos_at((size_t) n_prompt);
        // The token sequence the draft source conditions on: the prompt plus everything confirmed
        // since. Only built when speculating — the plain path has no use for it. The head reads it as
        // the sequence to seed from; the n-gram source searches it, and it IS the whole corpus.
        if (spec_on) mtp_ctx = tokens;
        mtp0_drafted = im.mtp_drafted;
        mtp0_accepted = im.mtp_accepted;
        mtp0_decodes = im.mtp_decodes;
        mtp0_drafted_steps = im.drafted_steps;
        mtp0_draft_s = im.mtp_draft_seconds;
        mtp0_draft_bytes = im.mtp_draft_read_bytes;

        // The token the last decode settled on, not yet in the KV. Greedy stays argmax (byte-identical
        // to the resident reference the gates check); with a sampling chain, draw from the context's
        // last-position logits, which llama_sampler_sample reads at index -1 — the same logits argmax
        // would have read.
        if (!arm_sampler()) return false;
        return sample_next(-1);
    }

    // The next token from logits row `row`: argmax, or a draw from the sampling chain, and then held
    // to the tool-call grammar when the turn has one. The first choice is checked alone and kept when
    // the grammar takes it, so free text and well-formed calls cost one check; only a refused token
    // pays for masking the whole vocabulary and choosing again. Greedy, that is still argmax over what
    // the grammar allows, and with no grammar it is exactly the old path.
    llama_token sample(int32_t row) {
        const float * lg = llama_get_logits_ith(ctx, row);
        llama_sampler * chain = sampler();
        llama_token t = chain ? llama_sampler_sample(chain, ctx, row) : argmax(lg, im.n_vocab);
        if (!grammar.active()) {
            if (req.n_probs > 0) tok_probs = position_logprobs(lg, im.n_vocab, im.vocab, t, req.n_probs);
            return t;
        }
        if (!grammar.allows(t)) {
            cand.resize((size_t) im.n_vocab);
            for (llama_token v = 0; v < im.n_vocab; ++v)
                cand[v] = {v, lg[v], 0.0f};
            llama_token_data_array cur = {cand.data(), cand.size(), -1, false};
            grammar.constrain(&cur);
            const bool dead_end = std::none_of(cur.data, cur.data + cur.size,
                                               [](const llama_token_data & c) { return std::isfinite(c.logit); });
            if (!dead_end && chain) {
                llama_sampler_apply(chain, &cur);
                if (cur.selected >= 0 && (size_t) cur.selected < cur.size &&
                    std::isfinite(cur.data[cur.selected].logit))
                    t = cur.data[cur.selected].id;
            } else if (!dead_end) {
                size_t best = 0;
                for (size_t i = 1; i < cur.size; ++i)
                    if (cur.data[i].logit > cur.data[best].logit) best = i;
                t = cur.data[best].id;
            }
        }
        // A grammar that takes no token is a broken contract for a response schema: the caller fails
        // the turn. A tool grammar is a guide, not a contract (#334): the unconstrained pick stands,
        // and accept() switches the grammar off with a warning while the parser fallback still holds.
        if (constrained() && !grammar.allows(t)) return LLAMA_TOKEN_NULL;
        grammar.accept(t);
        if (req.n_probs > 0) tok_probs = position_logprobs(lg, im.n_vocab, im.vocab, t, req.n_probs);
        return t;
    }

    // ── draft ──
    // The source proposes a continuation of `tok`, capped at the caller's remaining budget: a
    // draft accepted past n_predict would be verified, charged for, and then discarded. The cap
    // goes in BEFORE drafting, so no source is ever asked for tokens with nowhere to go.
    bool draft(Step & s) {
        const int room = n_predict - n_gen - 1; // tokens still wanted after `tok` itself
        if (!spec_on || room <= 0) return true;
        const auto d0 = clock_t_::now();
        const uint64_t db0 = moe.enabled ? im.source.stats().read_bytes : 0;
        im.draft_buf.clear();
        if (mtp_on) {
            common_speculative_draft_params & dp = common_speculative_get_draft_params(im.mtp.get(), /*seq*/ 0);
            dp.drafting = true;
            dp.n_max = std::min(im.cfg.spec.draft_max, room);
            dp.pos0 = n_past; // upstream renamed n_past -> pos0 (#28715); text-only, so the same value
            dp.id_last = tok;
            dp.prompt = &mtp_ctx;
            dp.result = &im.draft_buf;
            common_speculative_draft(im.mtp.get());
            if ((int) im.draft_buf.size() > room) im.draft_buf.resize((size_t) room);
            s.n_draft = (int) im.draft_buf.size();

            // Drafting WROTE into the draft context's KV at the very positions the catch-up
            // below is about to occupy. Rewind to where the draft started, or the second decode
            // collides with the first (llama.cpp requires a batch to begin strictly after the
            // last stored position). The catch-up is what replaces those rows with ones
            // conditioned on the target's own hidden states instead of the head's guesses.
            if (!llama_memory_seq_rm(llama_get_memory(im.ctx_dft.get()), /*seq*/ 0, n_past, -1))
                return fail(RunError::Failed, "the MTP draft context does not support rewinding its KV cache");
        } else {
            // The n-gram source reads the text and nothing else: no draft context, no decode, no
            // expert read, so the bytes bracketed around this arm are zero by construction. When
            // it finds no confident match it returns 0 and the step below is an ordinary
            // single-token decode — the property the head does not have, and the reason the
            // floor of this source is the baseline rather than a loss.
            s.n_draft = ngram_draft(mtp_ctx, tok, std::min(im.cfg.spec.draft_max, room), im.cfg.spec.ngram_min_match,
                                    im.cfg.spec.ngram_max_match, im.draft_buf);
        }
        im.mtp_drafted += s.n_draft;
        if (s.n_draft > 0) ++im.drafted_steps;
        s.draft_s += secs(d0, clock_t_::now());
        if (moe.enabled) im.mtp_draft_read_bytes += im.source.stats().read_bytes - db0;
        return true;
    }

    // ── verify batch: the confirmed token, then every draft, all asking for logits ──
    //
    // With nothing drafted there is nothing to verify, so the step takes the plain path — one
    // token, one logits row, no rollback. That is not an optimisation of the speculative loop,
    // it IS the loop's floor: a step that drafts nothing must cost exactly what it would have
    // cost with speculation off, or a source that abstains would still be paying for the
    // scaffolding. It applies to the head too, whenever p_min stops it drafting.
    //
    // Returns llama_decode's result.
    int verify_decode(Step & s) {
        s.wide = spec_on && s.n_draft > 0;
        llama_batch step;
        if (spec_on) {
            verify_toks.clear();
            verify_toks.push_back(tok);
            // n_draft, not draft_buf.size(): on the last token of a run there is no room to draft
            // and the buffer still holds the previous step's proposal.
            verify_toks.insert(verify_toks.end(), im.draft_buf.begin(), im.draft_buf.begin() + s.n_draft);
        }
        if (s.wide) {
            batch_fill(im.mtp_batch, verify_toks.data(), (int) verify_toks.size(), n_past, /*all_logits*/ true);
            step = im.mtp_batch;
        } else {
            step = llama_batch_get_one(&tok, 1);
        }

        // Bracket ONLY the decode: major faults and CPU-time deltas here decompose this token's
        // compute residual into flash-fault stalls vs. genuine (or throttled) computation.
        s.f0 = pio::major_faults();
        s.c0 = pio::process_cpu_seconds();
        s.s0 = clock_t_::now();
        s.overhead = secs(loop_mark, s.s0); // everything since the previous decode returned
        loop_overhead_s += s.overhead;
        trace_begin(n_past, /*n_tokens*/ 1 + s.n_draft, /*phase*/ 1);
        const int dec = llama_decode(ctx, step);
        // On a device backend llama_decode returns once the graph is queued, not computed: the wait
        // would land in the next logits read and be billed to loop overhead, and a two-GPU run read
        // 1777 tok/s for 241. On the CPU the graph is already done and this returns at once.
        llama_synchronize(ctx);
        s.s1 = clock_t_::now();
        loop_mark = s.s1; // the next token's overhead is measured from here
        s.f1 = pio::major_faults();
        s.c1 = pio::process_cpu_seconds();
        return dec;
    }

    // ── accept ──
    // `tok` was settled before the decode, so it is confirmed unconditionally. Each draft is
    // confirmed only while it matches what the target itself would have produced at that
    // position — no approximation enters here. The batch's arithmetic is still not bit-identical
    // to a single-token pass, so a near-tie can land differently; see docs/mtp.md.
    bool accept(Step & s) {
        confirmed.clear();
        confirmed.push_back(tok);
        confirmed_probs.clear();
        if (req.n_probs > 0) confirmed_probs.push_back(tok_probs);
        for (int i = 0; i < s.n_draft; ++i) {
            const llama_token want = argmax(llama_get_logits_ith(ctx, i), im.n_vocab);
            if (want != im.draft_buf[i]) break;
            // A draft the grammar refuses ends the group here, and the next token is sampled from
            // this same row under the grammar, exactly as it would have been without speculation.
            if (grammar.active()) {
                if (!grammar.allows(want)) break;
                grammar.accept(want);
                if (constrained() && !grammar.active()) {
                    rollback_turn();
                    return fail(RunError::InvalidOutput, "the response grammar rejected a drafted token");
                }
            }
            ++s.n_acc; // accepted: it is in the KV whether or not the caller gets to see it
            if (llama_vocab_is_eog(im.vocab, want)) {
                s.eog_hit = true; // end-of-generation is never emitted, here as on the plain path
                break;
            }
            confirmed.push_back(want);
            if (req.n_probs > 0)
                confirmed_probs.push_back(
                    position_logprobs(llama_get_logits_ith(ctx, i), im.n_vocab, im.vocab, want, req.n_probs));
        }
        if (!spec_on) return true;
        im.mtp_accepted += s.n_acc;

        // Catch the draft context up on the ACCEPTED PREFIX ONLY.
        //
        // The catch-up re-runs the MTP block over the batch so the draft context holds rows
        // conditioned on the target's own hidden states, and accept() below seeds the next draft
        // from the row at index n_acc. Handing it the whole verify batch — the obvious ordering,
        // and the one upstream's own loop uses — computes the rejected tail as well, and that
        // tail is deleted a few statements later: those drafts were wrong and nothing ever reads
        // their rows. Acceptance depends only on the target's logits, which are already in hand
        // by this point, so the tail simply never has to be submitted.
        //
        // Identical state, strictly less work. accept() seeds from row min(n_acc, n_rows-1),
        // which is row n_acc under either batch, and the KV this leaves behind is exactly the
        // range the rollback used to carve out. What it saves is (n_draft - n_acc) positions
        // through the MTP block — and that block routes experts of its own, so on a streamed
        // device the saving is flash reads, not just arithmetic.
        //
        // The n-gram source has no state to catch up: its next draft is read off the token
        // history the emit block appends to, which is already correct by the time it is read.
        if (mtp_on) {
            const auto p0 = clock_t_::now();
            const uint64_t pb0 = moe.enabled ? im.source.stats().read_bytes : 0;
            batch_fill(im.mtp_batch, verify_toks.data(), 1 + s.n_acc, n_past, /*all_logits*/ false);
            if (!common_speculative_process(im.mtp.get(), to_common_batch(im.mtp_batch)))
                return fail(RunError::Failed, "MTP draft context failed to process the verify batch");
            s.draft_s += secs(p0, clock_t_::now());
            if (moe.enabled) im.mtp_draft_read_bytes += im.source.stats().read_bytes - pb0;
        }
        im.mtp_draft_seconds += s.draft_s;

        // Drop the rejected tail from the target; the bounded-rollback snapshots asked for at
        // context creation are what make this a restore rather than a replay. The draft context
        // needs no rollback of its own — it was never given the tail.
        if (s.n_acc < s.n_draft) {
            const llama_pos keep = n_past + 1 + s.n_acc;
            if (!llama_memory_seq_rm(llama_get_memory(ctx), /*seq*/ 0, keep, -1))
                return fail(RunError::Failed, "failed to roll back the rejected draft tokens from the KV cache");
        }
        if (mtp_on) common_speculative_accept(im.mtp.get(), /*seq*/ 0, (uint16_t) s.n_acc);
        return true;
    }

    // ── emit ──
    // One metrics row per confirmed token, but ONE decode produced them all: its cost goes to
    // the first row and the rest carry zeros (see TokenMetrics::mtp_batch). Splitting the wall
    // evenly would read as several equally-cheap tokens, which is not what happened.
    void emit(const Step & s, const std::function<void(const TokenMetrics &)> & on_token, IMetricsSink * sink) {
        const double wall = secs(s.s0, s.s1);
        gen_seconds += wall;
        const IExpertSource::Stats st = moe.enabled ? im.source.stats() : IExpertSource::Stats{};
        // Route-ahead's eval-thread meters accumulate per DECODE, and one decode can confirm a whole
        // group, so they are read once here and charged to the group's first row like every other
        // group cost. Reading them inside the loop would advance the cursors once per token and
        // credit the second and later rows with a delta of zero for the wrong reason.
        double ra_issue_ms = 0.0, ra_wd_ms = 0.0;
        {
            const long long ri = im.hook->route_ahead_issue_ns(), rw = im.hook->route_ahead_wd_ns();
            ra_issue_ms = (ri - prev_ra_issue_ns) / 1e6;
            ra_wd_ms = (rw - prev_ra_wd_ns) / 1e6;
            prev_ra_issue_ns = ri;
            prev_ra_wd_ns = rw;
        }
        for (size_t e = 0; e < confirmed.size() && n_gen < n_predict; ++e) {
            const llama_token out = confirmed[e];
            char piece[256];
            int np = llama_token_to_piece(im.vocab, out, piece, sizeof(piece), 0, true);
            std::string delta = np > 0 ? std::string(piece, np) : std::string();
            if (np < 0 && constrained()) delta = common_token_to_piece(im.vocab, out, true);
            const size_t before = gen.size();
            gen += delta;
            // A stop ends the answer where it begins; text that may yet become one is not shown.
            size_t shown = gen.size();
            if (!req.stop_text.empty()) {
                // Only the answer is searched (AnswerStart): the parse that splits the reasoning off says
                // where it begins, and is asked until it has begun.
                if (!answer_start.open) {
                    const ShownView sv = shown_view(gen, /*partial*/ true);
                    answer_start.note(gen, !sv.content.empty() || sv.tool_calls.empty() ? sv.content : gen);
                }
                const size_t floor = answer_start.from;
                const size_t at =
                    answer_start.open ? detail::find_stop(gen, req.stop_text, before, floor) : std::string::npos;
                if (at != std::string::npos) {
                    for (const std::string & st : req.stop_text)
                        if (!st.empty() && gen.compare(at, st.size(), st) == 0) {
                            res.stop_text = st;
                            break;
                        }
                    gen.resize(at);
                    shown = at;
                    stop_hit = true;
                } else if (answer_start.open) {
                    shown -= detail::stop_prefix_tail(gen, req.stop_text, floor);
                }
            }
            if (chat_on) im.kv.push_token(out); // this token is now in the KV
            if (spec_on) mtp_ctx.push_back(out);
            ++n_gen;

            TokenMetrics m;
            m.step = n_gen;
            m.steps = n_predict;
            m.mtp_batch = (int) confirmed.size();
            m.loop_overhead_ms = e == 0 ? s.overhead * 1000.0 : 0.0;
            // Charged to the group's first row like every other group cost. This is a SLICE of
            // loop_overhead_ms, not an addition to it: both measure time outside the decode.
            m.mtp_draft_ms = e == 0 ? s.draft_s * 1000.0 : 0.0;
            m.ra_issue_ms = e == 0 ? ra_issue_ms : 0.0;
            m.ra_wd_ms = e == 0 ? ra_wd_ms : 0.0;
            m.piece = delta;
            // Only when someone will read it: the parser cannot resume, so this re-parses everything
            // generated so far on every token, and off the chat path it is a full copy of the same.
            if (req.render_text) {
                ShownView sv = shown_view(shown == gen.size() ? gen : gen.substr(0, shown), /*partial*/ true);
                m.text = std::move(sv.content);
                m.reasoning = std::move(sv.reasoning);
                m.tool_calls = std::move(sv.tool_calls);
            }
            if (e == 0) {
                tally.record(m, wall, s.f1 - s.f0, s.c1 - s.c0, im.turn, moe.enabled ? &st : nullptr);
                last_compute_s = m.compute_ms / 1000.0;
            } else
                tally.record(m, 0.0, 0, 0.0, im.turn, moe.enabled ? &st : nullptr);
            if (req.n_probs > 0 && e < confirmed_probs.size()) {
                m.has_logprobs = true;
                m.logprob = confirmed_probs[e].chosen;
                m.top_logprobs = confirmed_probs[e].top;
            }
            if (on_token) on_token(m);
            if (sink) sink->on_token(m);
            if (stop_hit) break; // what the decode confirmed past the stop is not the answer
        }
    }

    // ── the cache brake (cache_brake.h) ──
    // Between two decodes, which is the only time the budget may move. A stalled token is asked
    // about first, because it is the cheap signal; the residency sample is the expensive one and runs
    // only to confirm it, so a run that never stalls never pays for it.
    void brake_cache() {
        if (!moe.enabled || !moe.cache_brake || im.source.cache_budget() == 0) return;
        // A token decoded under a knob trial's candidate is slow by choice, not by reclaim: eight
        // threads on four fast cores ran one at ten times the reference, the sample found a page
        // out, and the cache was halved for the rest of the session (a phone: 18.8 tok/s to 10.7).
        // Neither the stall test nor the reference reads such a token.
        if (im.trials.stage < im.stage_rate && im.trials.cur && im.trials.cur->arm() != 0) return;
        const double ref = im.brake.reference();
        if (!im.brake.stalled(last_compute_s)) return;
        const double lost = im.source.cache_lost_frac(k_brake_sample);
        if (lost <= 0.0) return; // slow, but not because memory went: the cores, the clock, a neighbour
        const size_t budget = im.source.cache_budget();
        const size_t next = CacheBrake::braked(budget, im.source.cache_floor());
        if (next >= budget) return;
        im.source.brake_cache_budget(next);
        log_printf(LogLevel::Info,
                   "cache brake: a token took %.0f ms of compute against %.0f, and %.0f%% of the cache's sampled "
                   "pages were no longer in RAM; cache %zu -> %zu MiB",
                   last_compute_s * 1000.0, ref * 1000.0, lost * 100.0, budget >> 20, next >> 20);
    }
    // Entries sampled per stall: enough that a system holding back a tenth of the cache is seen,
    // few enough to be a few dozen syscalls on a token that already took seconds.
    static constexpr int k_brake_sample = 64;

    // A trial's switch of overlap changes what the streamer's read counter measures (time the eval
    // thread was blocked serially, lane time under overlap), so the cursor the next token's delta is
    // taken against moves to the new counter at the switch: no token is charged the difference
    // between two different clocks.
    template <typename F> void trial_guarded(F && f) {
        const bool before = moe.enabled && im.source.overlap();
        f();
        const bool after = moe.enabled && im.source.overlap();
        if (after == before) return;
        tally.overlap = after;
        tally.prev_io_s = im.source.stats().read_seconds;
    }

    // ── greedy generation ──
    // Returns false only on a failure; a cancel ends the loop with res.cancelled set.
    bool decode(const std::function<void(const TokenMetrics &)> & on_token, IMetricsSink * sink) {
        res.ok = true;
        if (!begin_decode()) return false;
        // A knob trial in progress takes up where the last turn left it.
        trial_guarded([&] { im.resume_trial(); });
        while (n_gen < n_predict) {
            if (llama_vocab_is_eog(im.vocab, tok)) break;
            Step s;
            if (!draft(s)) return false;
            if (verify_decode(s) != 0) {
                if (im.cancelled()) {
                    res.cancelled = true;
                    break;
                }
                if (im.source.fatal()) return fail(RunError::Failed, "expert stream I/O failed during decode");
                return fail(RunError::Failed, "decode failed during generation");
            }
            trace_flush(); // outside the s0..s1 bracket: the trace's own writes must not bill wall_ms
            ++im.mtp_decodes;
            if (!accept(s)) return false;
            emit(s, on_token, sink);
            brake_cache();
            if (im.trials.stage != im.stage_done) {
                // The decode as the knob changes it: the target's and the draft's passes, not the
                // emission around them, which a person's own client paces.
                const double step_s = secs(s.s0, s.s1) + s.draft_s;
                trial_guarded([&] { im.trial_step(step_s, (int) confirmed.size()); });
            } else {
                im.cost_step(last_compute_s, (int) confirmed.size());
            }

            if (constrained() && im.cancelled()) {
                res.cancelled = true;
                break;
            }
            if (s.eog_hit || stop_hit || (constrained() && n_gen >= n_predict)) break;
            // The accepted group is now KV-resident, and the logits at the first unverified position
            // hold the target's own continuation — the next token, arrived at for free. Off the
            // speculative path the row index stays -1, exactly as before: one token, one row.
            n_past += 1 + s.n_acc;
            const int32_t row = s.wide ? s.n_acc : -1;
            logits = llama_get_logits_ith(ctx, row);
            if (!sample_next(row)) return false;
        }
        return true;
    }

    // Speculation can leave the KV ahead of what the caller received: an accepted end-of-generation
    // token is decoded but never emitted, and a group can be cut short by n_predict. The KV and
    // kv_tokens must agree exactly or the next turn's prefix reuse decodes from a state that never
    // produced this answer, so trim back to what was actually emitted.
    void trim_speculative_kv() {
        if (!spec_on) return;
        const llama_pos emitted_end = (llama_pos) n_prompt + n_gen;
        // A memory that refuses the cut keeps what a checkpoint still describes, at worst nothing,
        // rather than leave kv_tokens asserting a prefix the context no longer holds. The
        // bookkeeping is kept in chat mode only: outside it the cut goes to the memory directly,
        // or the prefix clamp would read an empty record as "keep nothing".
        if (chat_on)
            im.keep_kv_prefix((size_t) emitted_end);
        else if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, emitted_end, -1))
            im.clear_memory(ctx);
        // The draft context mirrors the target's positions (process() decodes the same batches into
        // it), so it is trimmed to the same point — not cleared, or a continued chat turn would
        // feed it only the new suffix and draft from a state that never saw the conversation.
        if (mtp_on && !llama_memory_seq_rm(llama_get_memory(im.ctx_dft.get()), 0, emitted_end, -1))
            llama_memory_clear(llama_get_memory(im.ctx_dft.get()), true);
    }

    // ── summary ──
    void summarize(IMetricsSink * sink) {
        RunSummary & s = res.summary;
        s.arch = im.arch;
        s.n_expert = im.info.n_expert;
        s.n_generated = n_gen;
        s.n_predict = n_predict;
        s.gen_seconds = gen_seconds;
        s.s_per_token = n_gen ? gen_seconds / n_gen : 0.0;
        s.tokens_per_second = gen_seconds > 0 ? n_gen / gen_seconds : 0.0;
        // Close the accounting: the tail after the last decode belongs to no row, so add it here.
        loop_overhead_s += secs(loop_mark, clock_t_::now());
        s.loop_overhead_s_per_token = n_gen ? loop_overhead_s / n_gen : 0.0;
        s.n_prompt = n_prompt - (int) n_common; // tokens actually prefilled this turn (after KV reuse)
        s.n_past = chat_on ? (int) im.kv.size() : n_prompt + n_gen; // total context length now
        s.load_seconds = im.load_seconds;
        s.prefill_seconds = prefill_seconds;
        s.prefill_device_tokens = prefill_device_tokens;
        s.prefill_device_nodes = prefill_device_nodes;
        if (im.prefill) {
            s.prefill_device_read_mib = (double) (im.prefill->arena_read_bytes() - arena_read0) / (1024.0 * 1024.0);
            s.prefill_device_stall_seconds = im.prefill->arena_stall_seconds() - arena_stall0;
            s.prefill_device_cache_mib = (double) (im.prefill->arena_cache_bytes() - arena_cache0) / (1024.0 * 1024.0);
        }
        s.prefill_cpu_seconds = prefill_tally.cpu_seconds;
        s.prefill_read_mib = prefill_tally.read_mib;
        s.prefill_io_seconds = prefill_tally.io_seconds;
        s.prefill_stall_seconds = prefill_tally.stall_seconds;
        s.prefill_mgmt_seconds = prefill_tally.mgmt_seconds;
        s.majflt_per_token = !pio::major_faults_counted() ? -1.0 : n_gen ? (double) tally.majflt / n_gen : 0.0;
        s.cpu_s_per_token = n_gen ? tally.cpu_seconds / n_gen : 0.0;
        if (moe.enabled) {
            IExpertSource::Stats st = im.source.stats();
            s.moe_read_mib = tally.read_bytes / (1024.0 * 1024.0);
            s.moe_io_seconds = tally.io_seconds;
            s.moe_io_s_per_token = n_gen ? tally.io_seconds / n_gen : 0.0;
            s.moe_mgmt_s_per_token = n_gen ? tally.mgmt_seconds / n_gen : 0.0;
            s.moe_stall_s_per_token = n_gen ? tally.stall_seconds / n_gen : 0.0;
            s.moe_flash_wait_s_per_token = tally.overlap ? s.moe_stall_s_per_token : s.moe_io_s_per_token;
            s.moe_compute_s_per_token = s.s_per_token - s.moe_flash_wait_s_per_token - s.moe_mgmt_s_per_token;
            if (s.moe_compute_s_per_token < 0) s.moe_compute_s_per_token = 0;
            s.cache_hit_pct = st.cache_lookups > 0 ? 100.0 * st.cache_hits / st.cache_lookups : -1.0;
            const long long dec_lookups = st.cache_lookups - prev_cache_lookups;
            s.cache_hit_decode_pct =
                dec_lookups > 0 ? 100.0 * (double) (st.cache_hits - prev_cache_hits) / (double) dec_lookups : -1.0;
            s.cache_resident_mib = st.cache_resident_bytes / (1024.0 * 1024.0);
            s.cache_budget_mib = st.cache_budget_bytes / (1024.0 * 1024.0);
            s.cache_resizes = st.cache_resizes;
            s.cache_evictions = st.evictions;
            s.cache_rereads = st.rereads;
            s.cache_cold_admits = st.cold_admits;
            s.cache_discarded = st.discarded;
            s.cache_discarded_on_hit = st.discarded_on_hit;
            const RowSourceStats rs = im.source.row_stats();
            s.row_table_mib = rs.table_bytes / (1024.0 * 1024.0);
            s.row_resident_mib = rs.resident_bytes / (1024.0 * 1024.0);
            s.row_read_mib = rs.bytes_read / (1024.0 * 1024.0);
            s.row_rows = (long long) rs.rows;
            s.row_slab_reads = (long long) rs.slab_reads;
            s.row_evictions = (long long) rs.evictions;
            s.row_io_errors = (long long) rs.io_errors;
            s.moe_drain_s_per_token = n_gen ? tally.drain_seconds / n_gen : 0.0;
            s.moe_adopt_s_per_token = n_gen ? tally.adopt_seconds / n_gen : 0.0;
            s.token_demand_mib = st.token_demand_bytes / (1024.0 * 1024.0);
            s.layer_demand_mib = st.layer_demand_bytes / (1024.0 * 1024.0);
            s.moe_spec_read_mib = ((long long) st.spec_read_bytes - prev_spec_bytes) / (1024.0 * 1024.0);
            s.moe_spec_experts = st.spec_experts - prev_spec_experts;
            s.moe_spec_useful = st.spec_useful - prev_spec_useful;
        }
        s.experts_routed = im.hook->experts_routed() - prev_routed;
        s.experts_dropped = im.hook->experts_dropped() - prev_dropped;
        s.experts_reranked = im.hook->experts_reranked() - prev_reranked;
        s.experts_substituted = im.hook->experts_substituted() - prev_substituted;
        // Per-turn deltas, like every other generation figure here: a warm session's counters are
        // cumulative, and an acceptance rate averaged over earlier prompts would describe none of them.
        s.mtp_drafted = im.mtp_drafted - mtp0_drafted;
        s.mtp_accepted = im.mtp_accepted - mtp0_accepted;
        s.mtp_decodes = im.mtp_decodes - mtp0_decodes;
        s.drafted_steps = im.drafted_steps - mtp0_drafted_steps;
        s.mtp_draft_s_per_token = n_gen > 0 ? (im.mtp_draft_seconds - mtp0_draft_s) / n_gen : 0.0;
        s.mtp_draft_read_mib = (double) (im.mtp_draft_read_bytes - mtp0_draft_bytes) / (1024.0 * 1024.0);
        if (moe.predict_log) {
            // Session totals, not a per-generation delta: these are an accuracy estimate, and every
            // turn's routings are equally valid samples of it. See RunSummary.
            s.predict_stale = im.hook->predict_stale();
            s.predict_stale2 = im.hook->predict_stale2();
            s.predict_prev = im.hook->predict_prev();
            s.predict_self = im.hook->predict_self();
            s.predict_stale_by_layer = im.hook->predict_stale_by_layer();
            s.predict_prev_by_layer = im.hook->predict_prev_by_layer();
            s.predict_self_by_layer = im.hook->predict_self_by_layer();
            s.predict_unscored = im.hook->predict_unscored();
        }
        if (moe.route_ahead > 0) {
            // Session totals, like the probe's: every overridden routing is an equally valid sample of
            // the perturbation the flag buys, whichever turn produced it.
            s.route_ahead_overridden = im.hook->route_ahead_overridden();
            s.route_ahead_passthrough = im.hook->route_ahead_passthrough();
            s.route_ahead_slots = im.hook->route_ahead_slots();
            s.route_ahead_hits = im.hook->route_ahead_hits();
            s.route_ahead_gemv_ns = im.hook->route_ahead_gemv_ns();
            s.route_ahead_gemv_jobs = im.hook->route_ahead_gemv_jobs();
            s.route_ahead_issue_ns = im.hook->route_ahead_issue_ns();
            s.route_ahead_wd_ns = im.hook->route_ahead_wd_ns();
        }
        if (sink) sink->on_summary(s);
    }

    // Parse before committing history or publishing a successful summary. A length/cancel stop
    // keeps the parser's partial content; only a completed answer promises valid JSON.
    bool parse_result() {
        const bool complete = !res.cancelled && n_gen < n_predict;
        if (chat_on) {
            try {
                final_msg = common_chat_parse(common_chat_input(gen), structured() && !complete, parse_params);
                forget_prefilled_reasoning(final_msg);
                final_parsed = true;
            } catch (const std::exception & e) {
                if (structured() && complete) {
                    rollback_turn();
                    return fail(RunError::InvalidOutput,
                                std::string("the completed response could not be parsed: ") + e.what());
                }
                if (!structured()) detail::warn_parse_failed_once(e.what());
            }
        }
        if (structured() && complete) {
            std::string error;
            if (!final_parsed || !detail::validate_json_response(final_msg, error)) {
                rollback_turn();
                return fail(RunError::InvalidOutput,
                            error.empty() ? "the completed response could not be parsed" : error);
            }
        }
        // Preserve the ordinary chat fallback (#334). An explicit response format must never
        // silently replace a failed parse with raw text.
        if (!structured() && final_parsed && !gen.empty() && detail::parsed_nothing(final_msg)) {
            log_printf(LogLevel::Warn,
                       "the chat parser found no answer, reasoning or call in %d generated tokens; "
                       "returning the raw output",
                       n_gen);
            final_parsed = false;
        }
        if (final_parsed) {
            res.generated_text = final_msg.content;
            res.reasoning_text = final_msg.reasoning_content;
            res.tool_calls = detail::from_common_tool_calls(final_msg.tool_calls);
        } else if (!structured()) {
            res.generated_text = gen;
        }
        return true;
    }

    // The conversation is committed after its output has passed the requested contract.
    void finish() {
        if (res.cancelled) {
            // Undo the whole turn (KV, fed tokens, and the pushed user message) so the conversation
            // is left exactly as it was before this prompt and stays continuable.
            rollback_turn();
        } else if (chat_on) {
            // Commit the assistant turn to the running conversation. Parsing separates a thinking
            // model's reasoning from the answer; the next turn re-renders history from these
            // messages. Reuses the parse above. A prefilled turn has no turn header in the stream to
            // parse — the generation is the answer verbatim — and is committed without the prefill,
            // so history holds a normal assistant message and the next turn re-renders cleanly
            // whatever this turn's think setting was. A parse that threw or found nothing falls back the same way.
            common_chat_msg assistant;
            if (final_parsed)
                assistant = std::move(final_msg);
            else
                assistant.content = gen;
            assistant.role = "assistant";
            im.chat_history.push_back(assistant);
        }
    }
};

RunResult Session::generate(const GenerateRequest & req,
                            const std::function<void(const TokenMetrics &)> & on_token,
                            IMetricsSink * sink) {
    Impl & im = *impl_;
    RunResult res;
    Impl::Turn turn(im, req, res);
    if (!req.json_schema.empty() && !im.chat_on) {
        res.error_kind = RunError::NeedsChatMode;
        res.error = "a response schema needs a session opened with the chat template on";
        return res;
    }
    // The pinned specialized and differential parsers give a response schema precedence
    // over required tool calls. Refuse this specific conflict rather than silently dropping it.
    if (!req.json_schema.empty() && req.tool_choice == ToolChoice::Required) {
        res.error_kind = RunError::InvalidRequest;
        res.error = "response schemas cannot be combined with tool_choice required in the current template parser";
        return res;
    }
    // Callbacks can run on backend workers, but all finish before this serialized call returns.
    // Clear both borrowed pointers on every exit, before the caller may destroy its flag.
    struct CallerStop {
        Impl & im;
        CallerStop(Impl & owner, const std::atomic<bool> * stop) : im(owner) {
            im.caller_stop = stop;
            if (im.media) im.media->set_stop(stop);
        }
        ~CallerStop() {
            im.caller_stop = nullptr;
            if (im.media) im.media->set_stop(nullptr);
        }
    } caller_stop{im, req.stop};
    im.begin_generate(req, sink);
    // Whichever way this call returns, a cancel's quiet window ends with it.
    struct CancelQuiet {
        Impl & im;
        ~CancelQuiet() { im.end_cancel_quiet(); }
    } cancel_quiet{im};

    if (!turn.render_prompt()) return res;
    if (!turn.tokenize_prompt()) return res;
    turn.reuse_kv_prefix();
    // Every exit from here on leaves the weights on the host (see PlacementGuard), and a knob trial
    // in progress on its incumbent: the next prefill is not part of the trial.
    Impl::PlacementGuard placement_guard{im};
    struct TrialPause {
        Impl & im;
        ~TrialPause() { im.pause_trial(); }
    } trial_pause{im};
    if (!turn.prefill()) return res; // a failure, or a cancel already rolled back
    const bool decoded = turn.decode(on_token, sink);
    if (!decoded) return res;
    if (!turn.parse_result()) return res;
    turn.trim_speculative_kv();
    ++im.turn; // this turn is written; label the next one apart
    turn.summarize(sink);
    im.note_width_turn(res.summary.n_prompt, res.summary.prefill_seconds, res.summary.n_generated);
    im.note_request_mix(res.summary);
    turn.finish();
    return res;
}

} // namespace bmoe
