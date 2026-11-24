#include "thinking_control.h"
#include "bmoe/log.h"

#include <cstdio>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace bmoe {

const char * think_control_name(ThinkControl c) {
    switch (c) {
    case ThinkControl::Template:
        return "template";
    case ThinkControl::Prefill:
        return "prefill";
    case ThinkControl::None:
        return "none";
    }
    return "template";
}

} // namespace bmoe

namespace bmoe::detail {

namespace {

// What a natively-supporting template puts inside the closed span. See add_no_think_prefill:
// whitespace, so the engine contributes no words to the model's own reasoning.
const char * const kEmptyReasoning = "\n\n";

// Render a one-turn conversation the way generate() renders a real one — same jinja path, same
// reasoning format — so what the probe observes is what generation will produce. The message text
// is irrelevant: templates branch on the flag, never on the words.
common_chat_params apply_probe(const common_chat_templates * tmpls, bool enable_thinking, bool prefill) {
    common_chat_msg user;
    user.role = "user";
    user.content = "hi";

    common_chat_templates_inputs inputs;
    inputs.messages = {user};
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.enable_thinking = enable_thinking;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    if (prefill) add_no_think_prefill(inputs);

    return common_chat_templates_apply(const_cast<common_chat_templates *>(tmpls), inputs);
}

} // namespace

void add_no_think_prefill(common_chat_templates_inputs & inputs) {
    common_chat_msg prefill;
    prefill.role = "assistant";
    prefill.reasoning_content = kEmptyReasoning;
    // content stays empty: the handler renders the closing tag and then nothing, leaving the model
    // at the first token of its answer. Nothing is appended that would have to be stripped back off.
    inputs.messages.push_back(prefill);
    inputs.continue_final_message = COMMON_CHAT_CONTINUATION_CONTENT;
}

ThinkControl probe_think_control(const common_chat_templates * tmpls) {
    if (tmpls == nullptr) return ThinkControl::Template;
    try {
        const common_chat_params off_p = apply_probe(tmpls, /*enable_thinking=*/false, /*prefill=*/false);
        const std::string off = off_p.prompt.str();
        const std::string on = apply_probe(tmpls, /*enable_thinking=*/true, /*prefill=*/false).prompt.str();
        if (on != off) return ThinkControl::Template;

        // Does the model own a reasoning span — a `<think>`-style pair it opens and closes itself?
        //
        // Both halves are needed. A declared tag alone proves nothing, because handlers publish the
        // pair for a whole family: the non-reasoning members (LFM2-8B-A1B, LFM2.5-Instruct) advertise
        // `<think>` they never emit. Requiring the template to actually USE it is the same test
        // llama.cpp applies before wiring up reasoning extraction for that family.
        const std::string src = common_chat_templates_source(tmpls);
        const bool owns_span =
            !off_p.thinking_start_tag.empty() && src.find(off_p.thinking_start_tag) != std::string::npos;

        const std::string prefilled = apply_probe(tmpls, /*enable_thinking=*/false, /*prefill=*/true).prompt.str();

        // It does, and the flag is inert. Whether the request can still be honoured depends on WHO
        // closes the span. If the prefilled render moves past the closed span into further structure
        // of the format itself — harmony ends the analysis channel and then opens the final one —
        // the model is placed in its answer section and cannot decline. If the render just ends at
        // the closing tag, closing it was only a suggestion to a model that opens its own — LFM2.5
        // reasons straight past a pre-closed empty span and emits the reasoning untagged into the
        // answer (issue #82), worse than leaving it alone. Say so instead of making it worse.
        if (owns_span) {
            if (prefilled != off) {
                size_t close_end = std::string::npos;
                for (const std::string & tag : off_p.thinking_end_tags) {
                    const size_t at = tag.empty() ? std::string::npos : prefilled.rfind(tag);
                    if (at != std::string::npos && (close_end == std::string::npos || at + tag.size() > close_end)) {
                        close_end = at + tag.size();
                    }
                }
                if (close_end != std::string::npos &&
                    prefilled.find_first_not_of(" \t\r\n", close_end) != std::string::npos) {
                    return ThinkControl::Prefill;
                }
            }
            return ThinkControl::None;
        }

        // No span of its own, and the prefill lands: reasoning is structural — a channel the format
        // itself separates — so starting the turn past that section is not something the model can
        // decline (harmony/gpt-oss).
        if (prefilled != off && off_p.thinking_start_tag.empty()) return ThinkControl::Prefill;

        // Nothing indicates this model reasons at all: no span it uses, no reasoning section to start
        // past. There is nothing to suppress, so pass the flag and add nothing — claiming it "always
        // reasons" would be a louder lie than the silence this probe exists to end.
        return ThinkControl::Template;
    } catch (const std::exception & e) {
        log_printf(LogLevel::Warn, "thinking-control probe failed (%s); assuming the template honours it", e.what());
        return ThinkControl::Template;
    }
}

bool build_turn_inputs(common_chat_templates_inputs & inputs,
                       std::vector<common_chat_msg> messages,
                       bool think,
                       ThinkControl ctl) {
    inputs.messages = std::move(messages);
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.enable_thinking = think;
    // AUTO is what bakes reasoning-stripping into the generated parser grammar. It is set here,
    // before apply — the field defaults to NONE, which produces a content-only grammar that leaves
    // <think> markers in the answer no matter how the parse is wired.
    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

    // Many templates never read enable_thinking (LFM2.5 among them): the flag reaches the jinja
    // context, is discarded, and the model reasons anyway — the setting silently does nothing. For
    // those, close the reasoning span in the prompt instead, so the model resumes at the first token
    // of its answer with the reasoning already behind it. Which models need this was measured at
    // open() (probe_think_control), not assumed.
    if (!think && ctl == ThinkControl::Prefill) {
        add_no_think_prefill(inputs);
        return true;
    }
    return false;
}

} // namespace bmoe::detail
