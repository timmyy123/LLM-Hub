#pragma once

// How a "thinking off" request is actually honoured, decided from the model rather than from a
// table of model names. Kept separate from session.cpp so the decision is unit-testable against
// llama.cpp's bundled templates, with no model and no live session.
//
// Internal header: includes llama.cpp's `common` (not the stable public API), so it must not be
// pulled into core/include/bmoe/. See docs/seam.md.

#include "bmoe/session.h"

#include "chat.h"

namespace bmoe::detail {

// Turn a rendered turn into one that starts with the model's reasoning span already closed.
//
// This sets no markers of its own. It appends a synthetic assistant message and asks for a
// CONTENT continuation, which makes llama.cpp's own per-template handler emit whatever "reasoning
// is over, answer now" looks like for that family — `<think></think>` for the LFM2 and Qwen
// handlers, a primed `<|channel|>final<|message|>` for harmony, and so on. The engine never spells
// any of them out.
//
// The reasoning span carries whitespace, not words: a continuation is only rendered when the
// message is non-empty (`has_continuation()`), and whitespace is exactly what a template that
// implements this natively puts there (Qwen3 renders `<think>\n\n</think>` for enable_thinking
// false). Anything language-bearing would be the engine inventing content for the model.
//
// CONTENT is passed explicitly rather than AUTO: AUTO resolves a reasoning-only message to a
// REASONING continuation, which *opens* the span instead of closing it — the opposite request.
void add_no_think_prefill(common_chat_templates_inputs & inputs);

// Decide which mechanism this model's template supports, by rendering it and looking.
//
// Three renders, once per session:
//   1. enable_thinking true vs false — if the prompt changes, the template reads the flag and
//      nothing further is needed. This is the same render-and-diff llama.cpp's own template
//      analyzer performs; `common_chat_params::supports_thinking` cannot answer the question,
//      because per handler it is a hardcoded literal reporting "this model can reason", not
//      "this template reads the variable".
//   2. with the prefill applied — if that leaves the prompt untouched, nothing reaches this model
//      and the request cannot be honoured.
//   3. otherwise the prefill lands, and whether the model must OBEY it is read off the reasoning
//      tags it declares: a declared span is the model's own to open and close, so a pre-closed
//      empty one is only a suggestion (LFM2.5 ignores it — see probe_think_control); no declared
//      span means the format separates reasoning structurally and the prefill places the model
//      past it, which it cannot ignore.
//
// Saying "this model cannot be silenced" is better than offering a control that does nothing — and
// far better than one that makes the output worse.
//
// Fails open to Template (the pre-existing behaviour: pass the flag and let the template decide),
// because a probe that itself failed is no evidence that the flag is inert.
ThinkControl probe_think_control(const common_chat_templates * tmpls);

// The template inputs for the assistant turn that follows `messages`, with a think=false request
// honoured the way this model supports it (`ctl`, from probe_think_control). One definition for
// every caller that renders a turn, so a decision and a generation over the same conversation see
// the same prompt. Returns true when the reasoning span was closed in the prompt, i.e. the output
// starts mid-answer and there is no reasoning to parse out of it.
bool build_turn_inputs(common_chat_templates_inputs & inputs,
                       std::vector<common_chat_msg> messages,
                       bool think,
                       ThinkControl ctl);

} // namespace bmoe::detail
