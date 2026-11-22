// Stop text (GenerateRequest::stop_text): where a stop begins in the text generated so far, what a
// streamed answer holds back while one may be forming, and where in that text the answer itself
// begins. Pure string work, so it is tested without a model (tests/chat_parse_test.cpp).
#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace bmoe::detail {

// Where the earliest of `stops` begins in `text`, looking only at matches that begin at or after
// `floor` and end at or after `from` (text before it was searched already); npos when none does.
inline size_t find_stop(const std::string & text, const std::vector<std::string> & stops, size_t from, size_t floor) {
    size_t best = std::string::npos;
    for (const std::string & st : stops) {
        if (st.empty()) continue;
        const size_t start = std::max(floor, from >= st.size() ? from - st.size() + 1 : 0);
        const size_t at = text.find(st, start);
        if (at != std::string::npos && at < best) best = at;
    }
    return best;
}

// How many trailing bytes of `text`, none before `floor`, begin one of `stops`: what a streamed
// answer holds back, since the next token may complete the stop.
inline size_t stop_prefix_tail(const std::string & text, const std::vector<std::string> & stops, size_t floor) {
    const size_t room = text.size() > floor ? text.size() - floor : 0;
    size_t held = 0;
    for (const std::string & st : stops)
        for (size_t k = std::min(st.size() > 0 ? st.size() - 1 : 0, room); k > held; --k)
            if (text.compare(text.size() - k, k, st, 0, k) == 0) {
                held = k;
                break;
            }
    return held;
}

// Where the answer begins in the generated text. A stop is the caller's word for the end of the
// answer, and a model that reasons first writes its reasoning into the same text: matched there, a
// stop the reasoning happened to contain ended the turn before the answer began, and the client got
// an empty answer. So nothing before the answer is searched. Which part is the answer is the chat
// parser's to say, format by format; this only keeps the offset, fed after every token with the
// answer the parser reads so far.
struct AnswerStart {
    size_t from = 0;   // no stop is matched before this offset of the generated text
    bool open = false; // the answer has begun: `from` no longer moves

    // `text` is everything generated, `answer` what the parser shows of it as the answer (or a call
    // being written, which is past the reasoning too). While there is none, the answer begins after
    // what is written; once there is, it began in the token just written, and exactly where the
    // answer is the text's tail.
    void note(const std::string & text, const std::string & answer) {
        if (open) return;
        if (answer.empty()) {
            from = text.size();
            return;
        }
        open = true;
        if (answer.size() <= text.size() && text.size() - answer.size() > from &&
            text.compare(text.size() - answer.size(), answer.size(), answer) == 0)
            from = text.size() - answer.size();
    }
};

} // namespace bmoe::detail
