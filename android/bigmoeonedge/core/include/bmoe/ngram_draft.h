#pragma once

#include <cstdint>
#include <vector>

namespace bmoe {

// Prompt-lookup drafting: propose the continuation that followed the last time this exact token
// sequence was seen, in the prompt or in what has been generated so far.
//
// This is a draft SOURCE for the self-speculative loop, alternative to the model's MTP head. It
// exists because of what the flash split measured: at draft 3 on the host, the MTP head's own
// routing was 2.9% of the extra bytes a speculated run streams — the other 97.1% was the widened
// verify batch, which every source pays. So the prize is not in making the draft cheaper, with one
// exception: this source can decide to draft NOTHING, at zero cost, and the head cannot. Below
// min_match the step falls back to a plain single-token decode, which is why the floor of the
// feature is the baseline rather than a loss. The win, where it exists, lives in segments the model
// is copying — code edits, quoting from the context, structured output.
//
// Deliberately free of llama.cpp: tokens are int32_t (which is what llama_token is), so this stays
// pure policy on the `core/include/bmoe` side of the seam, is unit-testable with no model, and adds
// no dependency on llama.cpp's `common` layer — unlike the MTP path, which needs
// common/speculative.h. See docs/seam.md.
//
// The match: let the corpus be `ctx` followed by `last`, and the pattern be its suffix. Every
// position strictly before the corpus end is a candidate match end; the longest common suffix wins,
// and among equal lengths the most recent one does (recency is what a code edit or a repeated
// quotation wants). A match may overlap the pattern, which is what makes a run of one repeated token
// predict itself.
//
// Cost is O(corpus * max_match) with an early exit, on a corpus bounded by n_ctx — microseconds,
// and no incremental structure to keep in step with the KV cache.
//
//   ctx        prompt + tokens generated so far, in order (the caller already maintains it)
//   last       the token just confirmed, not yet appended to ctx; part of the pattern
//   n_max      most tokens to draft (the verify batch is 1 + this)
//   min_match  shortest suffix match allowed to draft at all; below it, draft nothing
//   max_match  longest suffix considered (a scan bound, not a target)
//   out        cleared, then filled with the drafted tokens
//
// Returns the number of tokens drafted, in [0, n_max]. 0 means "no confident match" and the caller
// must take the ordinary non-speculative path for this step.
int ngram_draft(const std::vector<int32_t> & ctx,
                int32_t last,
                int n_max,
                int min_match,
                int max_match,
                std::vector<int32_t> & out);

} // namespace bmoe
