#include "prompt_stream.h"

#include <algorithm>

namespace bmoe::detail {

llama_pos PromptStream::pos_at(size_t i) const {
    llama_pos pos = (llama_pos) i;
    for (const MediaSpan & m : media) {
        if (m.begin + m.n_tokens > i) break;
        pos -= (llama_pos) m.n_tokens - m.n_pos;
    }
    return pos;
}

const MediaSpan * PromptStream::media_at(size_t i) const {
    for (const MediaSpan & m : media) {
        if (i < m.begin) return nullptr;
        if (i < m.begin + m.n_tokens) return &m;
    }
    return nullptr;
}

size_t PromptStream::common_prefix(const PromptStream & next, size_t max_common) const {
    const size_t limit = std::min({max_common, tokens.size(), next.tokens.size()});
    size_t n = 0;
    while (n < limit) {
        const MediaSpan * a = media_at(n);
        const MediaSpan * b = next.media_at(n);
        if (a || b) {
            // Both must hold the same image over the same indices, whole, inside the limit.
            if (!a || !b || a->begin != n || b->begin != n || a->n_tokens != b->n_tokens || a->n_pos != b->n_pos ||
                a->id != b->id || n + a->n_tokens > limit)
                return n;
            n += a->n_tokens;
            continue;
        }
        if (tokens[n] != next.tokens[n]) return n;
        ++n;
    }
    return n;
}

size_t PromptStream::truncate(size_t n) {
    if (n >= tokens.size()) return tokens.size();
    if (const MediaSpan * m = media_at(n); m && m->begin < n) n = m->begin;
    tokens.resize(n);
    media.erase(std::remove_if(media.begin(), media.end(), [n](const MediaSpan & m) { return m.begin >= n; }),
                media.end());
    return n;
}

bool PromptStream::append(const PromptStream & src, size_t from, size_t to) {
    to = std::min(to, src.tokens.size());
    if (from >= to) return true;
    for (const MediaSpan & m : src.media) {
        const bool starts_inside = m.begin >= from && m.begin < to;
        const bool ends_inside = m.begin + m.n_tokens > from && m.begin + m.n_tokens <= to;
        if (starts_inside != ends_inside) return false;
    }
    const size_t shift = tokens.size();
    tokens.insert(tokens.end(), src.tokens.begin() + (long) from, src.tokens.begin() + (long) to);
    for (const MediaSpan & m : src.media)
        if (m.begin >= from && m.begin < to) media.push_back({m.begin - from + shift, m.n_tokens, m.n_pos, m.id});
    return true;
}

void PromptStream::push_media(size_t n_tokens, llama_pos n_pos, std::string id) {
    media.push_back({tokens.size(), n_tokens, n_pos, std::move(id)});
    tokens.insert(tokens.end(), n_tokens, LLAMA_TOKEN_NULL);
}

} // namespace bmoe::detail
