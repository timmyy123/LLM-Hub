#include "choice_scorer.h"

#include "../logits.h"

namespace bmoe::detail {

bool choice_first_tokens(const std::vector<std::string> & choices,
                         const std::vector<std::vector<Token>> & tokenized,
                         int n_vocab,
                         std::vector<Token> & out,
                         std::string & error) {
    out.clear();
    if (choices.empty()) {
        error = "no choices to decide between";
        return false;
    }
    for (size_t i = 0; i < choices.size(); ++i) {
        if (i >= tokenized.size() || tokenized[i].empty()) {
            error = "choice '" + choices[i] + "' does not tokenize";
            return false;
        }
        const Token t = tokenized[i][0];
        if (t < 0 || t >= n_vocab) {
            error = "choice '" + choices[i] + "' tokenizes outside the vocabulary";
            return false;
        }
        for (size_t j = 0; j < i; ++j)
            if (out[j] == t) {
                error = "choices '" + choices[j] + "' and '" + choices[i] +
                        "' start with the same token and cannot be told apart";
                return false;
            }
        out.push_back(t);
    }
    return true;
}

void score_choices(
    const float * logits, int n_vocab, const std::vector<Token> & ids, std::vector<double> & logp, int & best) {
    const LogNorm z = log_norm(logits, n_vocab);
    logp.clear();
    best = -1;
    for (size_t i = 0; i < ids.size(); ++i) {
        logp.push_back(z.logp(logits[ids[i]]));
        if (best < 0 || logp[i] > logp[(size_t) best]) best = (int) i;
    }
}

} // namespace bmoe::detail
