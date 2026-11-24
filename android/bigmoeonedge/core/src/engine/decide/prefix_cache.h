#pragma once

// Kept model state after a decision's prefix, so the next decision can restore it instead of
// prefilling the prefix again.

#include "backend.h"

#include "bmoe/config.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace bmoe::detail {

class IPrefixCache {
public:
    virtual ~IPrefixCache() = default;

    // Restore the longest kept state that covers a prefix of tokens[0, limit) into the (empty)
    // backend. Returns how many tokens it covers; 0 on a miss, with the backend left empty.
    virtual int restore(IDecideBackend & backend, const std::vector<Token> & tokens, int limit) = 0;
    // Keep the state the backend holds now as the state after tokens[0, n).
    virtual void store(IDecideBackend & backend, const std::vector<Token> & tokens, int n) = 0;
    virtual size_t bytes() const = 0;
};

// One kept state: the most recent prefix. An agent's prefix is its instructions, its task and its
// history so far; the history only grows, so each step's prefix extends the previous one and the
// most recent state is the one the next step can use. Storing replaces it.
class LastPrefixCache final : public IPrefixCache {
public:
    int restore(IDecideBackend & backend, const std::vector<Token> & tokens, int limit) override;
    void store(IDecideBackend & backend, const std::vector<Token> & tokens, int n) override;
    size_t bytes() const override { return state_.size(); }

private:
    std::vector<Token> tokens_;
    std::vector<uint8_t> state_;
};

// The cache a session's decide() uses, or null for none. `prefill_cost_scales_with_tokens` is the
// backend's answer to the one question PrefixCacheMode::Auto depends on.
std::unique_ptr<IPrefixCache> make_prefix_cache(PrefixCacheMode mode, bool prefill_cost_scales_with_tokens);

} // namespace bmoe::detail
