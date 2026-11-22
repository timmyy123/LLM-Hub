// The prompt as the context sees it: tokens, with the images in it as spans of placeholders.
//
// A text prompt is a token list, and the engine's KV bookkeeping (prefix reuse, checkpoints, the
// speculative trim) is written over it: index i of the list is position i of the context. An image
// breaks both halves of that: its tokens are embeddings, not vocabulary ids, so the list holds a
// placeholder for each; and under M-RoPE an image of n tokens takes fewer than n positions, so
// everything after it sits at a position below its index. This type keeps the list honest on both
// counts, as pure data: which indices are an image (and which image, by its content hash), and the
// position any index maps to. Nothing here touches a context; the session does, through pos_at().
#pragma once

#include "llama.h"

#include <cstddef>
#include <string>
#include <vector>

namespace bmoe::detail {

// One image inside the stream: `n_tokens` placeholders from `begin`, taking `n_pos` positions of
// the context, identified by the content hash the projector's reader gave the image.
struct MediaSpan {
    size_t begin = 0;
    size_t n_tokens = 0;
    llama_pos n_pos = 0;
    std::string id;
};

struct PromptStream {
    std::vector<llama_token> tokens; // LLAMA_TOKEN_NULL across every media span
    std::vector<MediaSpan> media;    // by begin, ascending, never overlapping

    size_t size() const { return tokens.size(); }
    bool empty() const { return tokens.empty(); }
    bool has_media() const { return !media.empty(); }
    void clear() {
        tokens.clear();
        media.clear();
    }

    // The context position of index i: i, less what every span wholly before it saved. For i ==
    // size() this is the position the next token goes at.
    llama_pos pos_at(size_t i) const;
    // The span holding index i, or null when i is text (or past the end).
    const MediaSpan * media_at(size_t i) const;

    // How many leading indices this stream shares with `next`: text equal token by token, a span
    // equal by id and extent, and never a cut inside a span (an image is in the KV whole or not at
    // all). At most max_common.
    size_t common_prefix(const PromptStream & next, size_t max_common) const;
    // Keep the first n indices; a cut inside a span moves back to its start. Returns what is kept.
    size_t truncate(size_t n);
    // Append `src`'s indices [from, to) with the spans they hold; a span cut by the range is refused
    // (returns false, nothing appended), since half an image is not a prefix of anything.
    bool append(const PromptStream & src, size_t from, size_t to);
    void push_token(llama_token t) { tokens.push_back(t); }
    // Add a span at the end: n_tokens placeholders.
    void push_media(size_t n_tokens, llama_pos n_pos, std::string id);
};

} // namespace bmoe::detail
