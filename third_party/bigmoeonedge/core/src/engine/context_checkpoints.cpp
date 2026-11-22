#include "context_checkpoints.h"

#include "speculative.h"

#include <algorithm>

namespace bmoe::detail {

namespace {

constexpr llama_seq_id k_seq = 0;
constexpr llama_state_seq_flags k_partial = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;

bool cut_draft(llama_context * dft, llama_pos p0) {
    return !dft || llama_memory_seq_rm(llama_get_memory(dft), k_seq, p0, -1);
}

} // namespace

void ContextCheckpoints::configure(int max_count, int min_step, int n_swa) {
    max_count_ = std::max(0, max_count);
    min_step_ = std::max(0, min_step);
    n_swa_ = std::max(0, n_swa);
    clear();
}

bool ContextCheckpoints::wanted(int64_t n_tokens, bool prompt_end) const {
    if (!enabled() || n_tokens <= 0) return false;
    if (list_.empty() || prompt_end) return true;
    return n_tokens >= list_.back().n_tokens + min_step_;
}

namespace {

bool is_user_start(const std::vector<size_t> * starts, int64_t i) {
    return starts && std::binary_search(starts->begin(), starts->end(), (size_t) i);
}

int64_t last_user_start(const std::vector<size_t> * starts) {
    return starts && !starts->empty() ? (int64_t) starts->back() : -1;
}

} // namespace

int64_t ContextCheckpoints::next_stop(const Stops & s, int64_t i, int64_t chunk) const {
    if (!enabled()) return i + chunk;
    int64_t stop = i + chunk;
    const auto inside = [&](int64_t p) { return p > i && p < stop; };
    if (s.user_starts) {
        const int64_t last = list_.empty() ? -1 : list_.back().n_tokens;
        for (auto it = std::upper_bound(s.user_starts->begin(), s.user_starts->end(), (size_t) i);
             it != s.user_starts->end() && inside((int64_t) *it); ++it) {
            const int64_t p = (int64_t) *it;
            if (p == last_user_start(s.user_starts) || last < 0 || p > last + min_step_) {
                stop = p;
                break;
            }
        }
    }
    for (const int offset : {k_checkpoint_tail_tokens + s.n_ubatch, k_checkpoint_tail_tokens}) {
        const int64_t p = s.n_prompt - std::min(s.n_batch, offset);
        if (inside(p)) stop = p;
    }
    return stop;
}

bool ContextCheckpoints::wanted_at(const Stops & s, int64_t i, int64_t chunk) const {
    if (!enabled() || i <= 0) return false;
    const bool near_end = s.n_prompt < i + chunk + s.n_ubatch;
    // A template that named no message boundaries: the prompt's last tokens, and the spacing rule
    // inside it, as before.
    if (!s.user_starts || s.user_starts->empty()) return near_end || wanted(i, /*prompt_end*/ false);
    const bool user_start = is_user_start(s.user_starts, i);
    if (!user_start && !near_end) return false;
    return list_.empty() || i == last_user_start(s.user_starts) || near_end || i > list_.back().n_tokens + min_step_;
}

bool ContextCheckpoints::create(const CheckpointContexts & c, int64_t n_tokens, int turn, bool preamble) {
    if (!wanted(n_tokens, /*prompt_end*/ true) || !c.tgt) return false;
    // A memory that cuts anywhere has no partial state: nothing to keep, and nothing it would need.
    if (llama_state_seq_get_size_ext(c.tgt, k_seq, k_partial) == 0) return false;

    // One checkpoint per position: a later prompt that ends where an earlier one did replaces it.
    list_.erase(std::remove_if(list_.begin(), list_.end(),
                               [&](const common_prompt_checkpoint & k) { return k.n_tokens == n_tokens; }),
                list_.end());
    if (preamble) preamble_ = n_tokens;
    const bool hold = preamble_ >= 0 && max_count_ > k_checkpoint_turn_slots;
    const auto held = [&](const common_prompt_checkpoint & k) { return hold && k.n_tokens == preamble_; };
    // Full: first the ones an earlier request took within min_step of the one before them, so the
    // sparse ones stay; then the oldest, the preamble's last.
    if ((int) list_.size() + 1 >= max_count_) {
        int64_t last = -1;
        for (auto it = list_.begin(); it != list_.end();) {
            if (!held(*it) && it->id_task != turn && last >= 0 && it->n_tokens <= last + min_step_) {
                it = list_.erase(it);
                continue;
            }
            last = it->n_tokens;
            ++it;
        }
    }
    while ((int) list_.size() >= max_count_) {
        auto it = std::find_if_not(list_.begin(), list_.end(), held);
        list_.erase(it != list_.end() ? it : list_.begin());
    }

    common_prompt_checkpoint & k = list_.emplace_back();
    k.id_task = turn;
    llama_memory_t mem = llama_get_memory(c.tgt);
    k.update_pos(n_tokens, llama_memory_seq_pos_min(mem, k_seq), llama_memory_seq_pos_max(mem, k_seq));
    k.update_tgt(c.tgt, k_seq, k_partial);
    k.update_dft(c.dft, k_seq, k_partial);
    if (c.spec) common_speculative_get_state(c.spec, k_seq, k.data_spec);
    return true;
}

int64_t ContextCheckpoints::restore(const CheckpointContexts & c, int64_t n) {
    if (!c.tgt) return 0;
    // The first token evaluated after the restore is at position kept, and it reads the positions a
    // sliding window still covers, so the checkpoint's window has to start no later than that; with
    // no window, the checkpoint only has to end before it. The same rule as the server's.
    for (auto it = list_.rbegin(); it != list_.rend(); ++it) {
        const common_prompt_checkpoint & k = *it;
        if (k.n_tokens > n) continue;
        const llama_pos pos_eval = pos_of(k.n_tokens);
        const llama_pos thold = useful_below(pos_eval);
        if (k.pos_max >= pos_eval || !(k.pos_min < thold || k.pos_min == 0)) continue;
        if (!k.load_tgt(c.tgt, k_seq, k_partial) || !k.load_dft(c.dft, k_seq, k_partial)) break;
        if (c.spec) common_speculative_set_state(c.spec, k_seq, k.data_spec);
        // The restored partial state ends at kept; what the rest of the memory holds past it goes.
        if (!llama_memory_seq_rm(llama_get_memory(c.tgt), k_seq, pos_eval, -1)) break;
        if (!cut_draft(c.dft, pos_eval)) break;
        const int64_t kept = k.n_tokens;
        drop_after(kept);
        return kept;
    }
    clear();
    return 0;
}

llama_pos ContextCheckpoints::useful_below(llama_pos pos_eval) const {
    return std::max<llama_pos>(0, n_swa_ > 0 ? pos_eval - n_swa_ + 2 : pos_eval);
}

bool ContextCheckpoints::memory_covers(const CheckpointContexts & c, int64_t n) const {
    if (n_swa_ <= 0 || !c.tgt || n <= 0) return true;
    // A cut past everything the memory holds removes nothing: what is there still reads as it did.
    // Asked first, because a memory with a recurrent part reports the recurrent cell's position as its
    // minimum, the last one, and would read as a window that no longer covers the prefix.
    if (pos_of(n) > llama_memory_seq_pos_max(llama_get_memory(c.tgt), k_seq)) return true;
    const llama_pos pos_min = llama_memory_seq_pos_min(llama_get_memory(c.tgt), k_seq);
    return pos_min <= 0 || pos_min < useful_below(pos_of(n));
}

void ContextCheckpoints::drop_after(int64_t n) {
    list_.erase(
        std::remove_if(list_.begin(), list_.end(), [&](const common_prompt_checkpoint & k) { return k.n_tokens > n; }),
        list_.end());
    if (preamble_ > n) preamble_ = -1;
}

size_t ContextCheckpoints::bytes() const {
    size_t b = 0;
    for (const common_prompt_checkpoint & k : list_)
        b += k.size();
    return b;
}

} // namespace bmoe::detail
