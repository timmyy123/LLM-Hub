#include "prefix_cache.h"

#include <algorithm>
#include <utility>

namespace bmoe::detail {

int LastPrefixCache::restore(IDecideBackend & backend, const std::vector<Token> & tokens, int limit) {
    const int n = (int) tokens_.size();
    if (n == 0 || n > limit || n > (int) tokens.size() || !std::equal(tokens_.begin(), tokens_.end(), tokens.begin()))
        return 0;
    if (backend.load_state(state_)) return n;
    // A state the backend will not take back is no use to any later call either.
    backend.clear();
    tokens_.clear();
    state_.clear();
    return 0;
}

void LastPrefixCache::store(IDecideBackend & backend, const std::vector<Token> & tokens, int n) {
    std::vector<uint8_t> state;
    if (n <= 0 || n > (int) tokens.size() || !backend.save_state(state) || state.empty()) {
        tokens_.clear();
        state_.clear();
        return;
    }
    tokens_.assign(tokens.begin(), tokens.begin() + n);
    state_ = std::move(state);
}

std::unique_ptr<IPrefixCache> make_prefix_cache(PrefixCacheMode mode, bool prefill_cost_scales_with_tokens) {
    const bool on = mode == PrefixCacheMode::On || (mode == PrefixCacheMode::Auto && prefill_cost_scales_with_tokens);
    if (!on) return nullptr;
    return std::make_unique<LastPrefixCache>();
}

} // namespace bmoe::detail
