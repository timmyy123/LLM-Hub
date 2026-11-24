#include "chat_parse.h"
#include "bmoe/log.h"

#include "common.h"
#include <nlohmann/json.hpp>

#include <cctype>
#include <cmath>
#include <cstdio>
#include <exception>
#include <mutex>

namespace bmoe::detail {

common_chat_parser_params build_parse_params(const common_chat_params & cp) {
    common_chat_parser_params pp(cp);
    // The converting constructor drops the grammar; copy it so the parser can actually match.
    // Whether reasoning is split out was already decided by the template inputs (reasoning_format).
    pp.parser = cp.parser;
    return pp;
}

void warn_parse_failed_once(const char * what) {
    static std::once_flag flag;
    std::call_once(flag, [what] {
        log_printf(LogLevel::Warn, "reasoning parse failed (%s); showing raw model output", what ? what : "unknown");
    });
}

bool parsed_nothing(const common_chat_msg & msg) {
    return msg.content.empty() && msg.reasoning_content.empty() && msg.tool_calls.empty();
}

bool validate_json_response(const common_chat_msg & msg, std::string & err) {
    err.clear();
    if (msg.content.empty() && !msg.tool_calls.empty()) return true;
    if (nlohmann::ordered_json::accept(msg.content)) return true;
    err = "the completed assistant content is not valid JSON";
    return false;
}

ToolGrammar::~ToolGrammar() {
    if (smpl_) llama_sampler_free(smpl_);
}

bool ToolGrammar::init(const common_chat_params & cp, const llama_vocab * vocab, std::string & err) {
    if (cp.grammar.empty() || !vocab) return true;
    // llguidance grammars need a build option this engine does not turn on.
    if (cp.grammar.compare(0, 11, "%llguidance") == 0) {
        err = "the template's grammar needs llguidance";
        return false;
    }
    // The triggers as llama.cpp's sampler reads them (common_sampler_init): words are literal text,
    // full patterns must match the whole output so far.
    std::vector<std::string> patterns;
    std::vector<llama_token> tokens;
    for (const common_grammar_trigger & t : cp.grammar_triggers) {
        switch (t.type) {
        case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
            patterns.push_back(regex_escape(t.value));
            break;
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
            patterns.push_back(t.value);
            break;
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
            const std::string & v = t.value;
            patterns.push_back(v.empty() ? "^$" : (v.front() != '^' ? "^" : "") + v + (v.back() != '$' ? "$" : ""));
            break;
        }
        case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
            if (t.token != LLAMA_TOKEN_NULL) tokens.push_back(t.token);
            break;
        }
    }
    if (cp.grammar_lazy && patterns.empty() && tokens.empty()) {
        err = "the template's lazy grammar has no trigger";
        return false;
    }
    std::vector<const char *> patterns_c;
    for (const std::string & p : patterns)
        patterns_c.push_back(p.c_str());
    smpl_ = cp.grammar_lazy
                ? llama_sampler_init_grammar_lazy_patterns(vocab, cp.grammar.c_str(), "root", patterns_c.data(),
                                                           patterns_c.size(), tokens.data(), tokens.size())
                : llama_sampler_init_grammar(vocab, cp.grammar.c_str(), "root");
    if (!smpl_) {
        err = "the template's grammar does not parse";
        return false;
    }
    vocab_ = vocab;
    // Only a lazy grammar leaves reasoning free. One that is on from the first token is built with
    // the reasoning span in it (that is why it is walked past the generation prompt below).
    if (cp.grammar_lazy) {
        think_start_ = cp.thinking_start_tag;
        think_ends_ = cp.thinking_end_tags;
    }
    // A generation prompt that ends by opening the reasoning span starts the turn inside it.
    if (!think_start_.empty() && !think_ends_.empty()) {
        std::string gp = cp.generation_prompt;
        while (!gp.empty() && std::isspace((unsigned char) gp.back()))
            gp.pop_back();
        thinking_ = gp.size() >= think_start_.size() &&
                    gp.compare(gp.size() - think_start_.size(), think_start_.size(), think_start_) == 0;
    }
    // A grammar that is on from the first token starts where the template's generation prompt
    // leaves off (an opened reasoning span, say), so it is walked past those tokens first, as
    // common_sampler_init does. The leading space some tokenizers add is not part of the prompt.
    if (!cp.grammar_lazy && !cp.generation_prompt.empty()) {
        const std::vector<llama_token> pre = common_tokenize(vocab, cp.generation_prompt, false, true);
        for (size_t i = 0; i < pre.size() && smpl_; ++i) {
            const std::string piece = common_token_to_piece(vocab, pre[i], true);
            if (i == 0 && !piece.empty() && std::isspace((unsigned char) piece[0]) &&
                !std::isspace((unsigned char) cp.generation_prompt[0]))
                continue;
            accept(pre[i]);
        }
        if (!smpl_) {
            err = "the template's grammar refuses its own generation prompt";
            return false;
        }
    }
    return true;
}

bool ToolGrammar::allows(llama_token tok) const {
    if (!smpl_ || thinking_) return true;
    llama_token_data one = {tok, 1.0f, 0.0f};
    llama_token_data_array cur = {&one, 1, -1, false};
    llama_sampler_apply(smpl_, &cur);
    return std::isfinite(one.logit);
}

void ToolGrammar::constrain(llama_token_data_array * cur) const {
    if (smpl_ && !thinking_) llama_sampler_apply(smpl_, cur);
}

void ToolGrammar::accept(llama_token tok) {
    if (!smpl_) return;
    if (!think_start_.empty() && !think_ends_.empty() && vocab_) {
        const std::string piece = common_token_to_piece(vocab_, tok, true);
        if (thinking_) {
            // The grammar never sees the span, its close included: what follows is where a call
            // may begin.
            span_ += piece;
            for (const std::string & end : think_ends_)
                if (!end.empty() && span_.find(end) != std::string::npos) {
                    thinking_ = false;
                    span_.clear();
                    break;
                }
            return;
        }
        // A model may open the span itself, but only as the first thing it writes.
        if (!started_) {
            span_ += piece;
            size_t lead = 0;
            while (lead < span_.size() && std::isspace((unsigned char) span_[lead]))
                ++lead;
            const std::string head = span_.substr(lead);
            if (head.compare(0, think_start_.size(), think_start_) == 0) {
                thinking_ = true;
                span_ = head.substr(think_start_.size());
                return;
            }
            if (head.size() >= think_start_.size() || think_start_.compare(0, head.size(), head) != 0) {
                started_ = true;
                span_.clear();
            }
        }
    }
    try {
        llama_sampler_accept(smpl_, tok);
    } catch (const std::exception & e) {
        off(e.what());
    }
}

void ToolGrammar::off(const char * why) {
    log_printf(LogLevel::Warn, "tool-call grammar switched off for this turn (%s)", why ? why : "unknown");
    llama_sampler_free(smpl_);
    smpl_ = nullptr;
}

} // namespace bmoe::detail
