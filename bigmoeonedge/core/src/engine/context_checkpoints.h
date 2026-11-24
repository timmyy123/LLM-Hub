#pragma once

// Context checkpoints, the way llama.cpp's server keeps them (tools/server, --ctx-checkpoints).
//
// A chat turn reuses the KV prefix it shares with the last one and removes the rest. Not every
// memory can be cut at an arbitrary position: a recurrent state (Mamba, Gated DeltaNet) holds only
// the latest step, and a sliding-window cache has already dropped what is out of its window. When
// the cut is refused the context has to start again from nothing, and a coding agent's request
// diverges from the cached one on most turns: a client resends the answer as text and the template
// renders it back into tokens that need not be the ones the model sampled. On a hybrid model that
// cost a full re-prefill of a 12k-token prompt on five requests of eight.
//
// So the part of the state that cannot be cut is copied out at points of the prompt, through the
// public state API with LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY (common_prompt_checkpoint, the server's
// own type), and a refused cut rolls back to the newest checkpoint at or before it. Nothing here asks
// what kind of model this is: a memory that cuts anywhere is never asked for a checkpoint's help,
// and has no partial state to copy, so its checkpoints are empty and none is kept.

#include "common.h"
#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

struct common_speculative;

namespace bmoe::detail {

// The contexts a checkpoint covers: the target, and, under MTP speculation, the draft context that
// mirrors its positions plus the speculative driver's own state.
struct CheckpointContexts {
    llama_context * tgt = nullptr;
    llama_context * dft = nullptr;
    common_speculative * spec = nullptr;
};

// How many tokens before a prompt's end the last checkpoints sit (llama.cpp's server, upstream #20288):
// a template renders a past turn's header and reasoning envelope again differently, a few tokens
// before the end of the prompt that sampled it, and a checkpoint has to be at or before that point.
inline constexpr int k_checkpoint_tail_tokens = 4;
// The checkpoints a turn needs at its end for the next request to reuse it: the one
// k_checkpoint_tail_tokens before the prompt's end, and the one at the end. The preamble's checkpoint
// is held back from eviction only when the list has room for it and for these.
inline constexpr int k_checkpoint_turn_slots = 2;

class ContextCheckpoints {
public:
    // max_count: checkpoints kept, the oldest dropped first (0 = none). min_step: the least spacing
    // between two of them, in tokens, except for the one at the end of a prompt. n_swa: the model's
    // sliding window (llama_model_n_swa), which decides how far back a checkpoint may reach.
    void configure(int max_count, int min_step, int n_swa);
    // The context position of the first n tokens' end. Identity until set: a prompt with images in
    // it takes fewer positions than tokens, and the session says how many (prompt_stream.h).
    using PositionMap = std::function<llama_pos(int64_t n_tokens)>;
    void set_position_map(PositionMap map) { pos_of_ = std::move(map); }

    bool enabled() const { return max_count_ > 0; }
    // Whether a checkpoint after n_tokens is worth taking: the end of a prompt always is, a point
    // inside one only min_step past the last.
    bool wanted(int64_t n_tokens, bool prompt_end) const;
    // Where the prefill stops for a checkpoint, the way llama.cpp's server places them: at the start
    // of a user message (the last one always, an earlier one only min_step past the newest checkpoint)
    // and n_ubatch + k_checkpoint_tail_tokens and k_checkpoint_tail_tokens before the prompt's end. A template renders
    // a past turn again differently from how it was sampled (Qwen3.6 drops its think block), so the next request
    // diverges a few tokens before the end of this prompt, and a checkpoint only at the end is past
    // the divergence. The start of the first user message is also the end of the preamble a new
    // conversation shares. user_starts: token indices, ascending, from the template's own message
    // delimiters (common_chat_msg_spans); empty when the template gave none.
    struct Stops {
        const std::vector<size_t> * user_starts = nullptr;
        int64_t n_prompt = 0;
        int n_batch = 0;  // the prefill's piece
        int n_ubatch = 0; // the context's ubatch
    };
    // The first stop strictly inside (i, i + chunk), or i + chunk when there is none.
    int64_t next_stop(const Stops & s, int64_t i, int64_t chunk) const;
    // Whether a piece [i, i + chunk) starts where a checkpoint is taken (llama.cpp's do_checkpoint).
    bool wanted_at(const Stops & s, int64_t i, int64_t chunk) const;
    // Copy the cut-resistant state after the first n_tokens tokens of seq 0. Kept only when there is
    // such a state; returns whether one was kept.
    // turn: which request took it. When the list is full, a checkpoint an earlier request took within
    // min_step of the one before it goes first, then the oldest (llama.cpp's create_checkpoint), except
    // the preamble's. preamble: this one is at the start of the prompt's first user message, the end of
    // the system prompt and tools a new conversation shares. It is the most valuable checkpoint and the
    // oldest, so the oldest-first rule dropped it first: with 4 kept, a third new conversation prefilled
    // the preamble again. The newest such checkpoint goes last, when the list holds more than it and
    // k_checkpoint_turn_slots; one at another preamble takes its place.
    bool create(const CheckpointContexts & c, int64_t n_tokens, int turn = -1, bool preamble = false);
    // The context refused to drop everything from position n: restore the newest checkpoint that
    // leaves the first `kept` tokens reusable, kept <= n, and drop the context past it. Returns kept,
    // 0 when no checkpoint helps (the caller then clears the context).
    int64_t restore(const CheckpointContexts & c, int64_t n);
    // Whether the context's own memory still holds what the first token after a cut at n reads.
    // A sliding-window cache accepts any cut (llama.cpp's seq_rm succeeds whatever the position) yet
    // keeps only the window, so a cut further back than the window would leave that token reading
    // cells that are gone, and nothing would say so: the answer would just be wrong. Always true for
    // a model without a window, whose memory refuses the cuts it cannot serve. The same test as
    // llama-server's.
    bool memory_covers(const CheckpointContexts & c, int64_t n) const;
    // The context now holds fewer than n tokens of what the checkpoints describe.
    void drop_after(int64_t n);
    void clear() {
        list_.clear();
        preamble_ = -1;
    }

    size_t count() const { return list_.size(); }
    size_t bytes() const;

private:
    llama_pos pos_of(int64_t n) const { return pos_of_ ? pos_of_(n) : (llama_pos) n; }
    // A state is of use to a token evaluated at pos_eval when it starts below this: that token reads
    // the positions its window still covers, and with no window only a state ending before it.
    llama_pos useful_below(llama_pos pos_eval) const;
    PositionMap pos_of_;
    std::vector<common_prompt_checkpoint> list_;
    int max_count_ = 0;
    int min_step_ = 0;
    int n_swa_ = 0;
    int64_t preamble_ = -1; // n_tokens of the preamble's checkpoint, -1 when there is none
};

} // namespace bmoe::detail
