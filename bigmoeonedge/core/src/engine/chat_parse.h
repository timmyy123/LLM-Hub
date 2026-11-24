#pragma once

// Wiring for llama.cpp's reasoning parser. Kept separate from session.cpp so the seam that
// was silently broken (issue #49) is unit-testable without a model or a live session.
//
// Internal header: includes llama.cpp's `common` (not the stable public API), so it must not
// be pulled into core/include/bmoe/. See docs/seam.md.

#include "chat.h"
#include "llama.h"

#include <string>
#include <vector>

namespace bmoe::detail {

// Build reasoning-parser params from an applied chat template.
//
// The `common_chat_parser_params(const common_chat_params &)` convenience constructor copies
// only `format` and `generation_prompt` (chat.h) — it leaves the PEG `parser` arena empty, so
// `common_chat_parse` falls back to a content-only parser and reasoning markers leak into the answer.
// The arena has to be copied across explicitly (this is what upstream's chat session does).
common_chat_parser_params build_parse_params(const common_chat_params & cp);

// Warn exactly once that reasoning parsing failed and the raw stream is being shown instead.
// Returning the raw text is the right degradation, but doing it silently is how #49 stayed
// invisible; the next wiring regression should be noticed.
void warn_parse_failed_once(const char * what);

// A finished turn the parser read as nothing at all: no answer, no reasoning, no call. Handing that
// back would turn generated tokens into an empty assistant message the client cannot tell from a
// model that chose to say nothing (#334), so the caller falls back to the raw generation instead.
bool parsed_nothing(const common_chat_msg & msg);

// The completed assistant content must be JSON. A tool-only turn has no final answer to
// validate; schema semantics remain the template grammar's responsibility.
bool validate_json_response(const common_chat_msg & msg, std::string & err);

// The grammar a template builds for tools or a response schema, held over sampling the way llama.cpp's own server
// holds it. Without it the model is free to write a call the parser cannot read back, and the turn
// comes out empty or the next request is refused for arguments that are not JSON (#334).
//
// Lazy by default: nothing is constrained until the output reaches one of the template's triggers
// (the opening of a call), so free text costs one check of one candidate per token. A reasoning span
// is never constrained, as llama.cpp's reasoning-budget sampler keeps it free there: a model that
// writes a call's opening while it thinks has not started a call.
class ToolGrammar {
public:
    ToolGrammar() = default;
    ToolGrammar(const ToolGrammar &) = delete;
    ToolGrammar & operator=(const ToolGrammar &) = delete;
    ~ToolGrammar();

    // Arms the grammar `cp` carries, or leaves it off when there is none. False with `err` set when
    // the template's grammar does not build; the caller decides whether fallback is allowed.
    bool init(const common_chat_params & cp, const llama_vocab * vocab, std::string & err);
    bool active() const { return smpl_ != nullptr; }

    // Whether `tok` may come next. Leaves the grammar where it is.
    bool allows(llama_token tok) const;
    // Masks every candidate the grammar refuses to -inf, in place.
    void constrain(llama_token_data_array * cur) const;
    // Advances the grammar past `tok`. A token it cannot take switches the grammar off for the rest
    // of the turn, with a warning. Explicit response-format requests must treat that as an error.
    void accept(llama_token tok);

private:
    void off(const char * why);
    llama_sampler * smpl_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    std::string think_start_;             // the template's reasoning opener, empty when it has none
    std::vector<std::string> think_ends_; // and what closes it
    bool thinking_ = false;               // inside a reasoning span: nothing is constrained
    bool started_ = false;                // any output yet besides whitespace
    std::string span_;                    // the reasoning so far, searched for its close
};

} // namespace bmoe::detail
