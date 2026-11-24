// A decision: which of a fixed set of choices the model would answer next, read from ONE prefill.
//
// Nothing is generated. The answer is the next-token distribution after the prompt, so a decision
// costs its prefill and no decode — on this engine, where decode streams experts from flash token
// by token, that is the difference between seconds and minutes for a large model. The shape fits
// any caller that picks from a list: an agent choosing the next UI action from lettered
// candidates, a router, a multiple-choice benchmark. See docs/decide.md.
//
// Pure policy header (no llama.cpp types). Session::decide() is the entry point.
#pragma once

#include "bmoe/run_error.h"

#include <cstddef>
#include <string>
#include <vector>

namespace bmoe {

// The prompt is split in two because a caller making a sequence of decisions (an agent stepping
// through screens) repeats most of it: `prefix` is the part that stays the same from one call to
// the next (instructions, task, history so far), `suffix` the part that changes (the current screen
// and the question). The chat template is rendered over prefix + suffix as ONE user message, so the
// split changes nothing about what the model sees; it only tells the session which part is worth
// remembering. See DecideConfig for when the state after the prefix is kept.
struct DecideRequest {
    std::string prefix;
    std::string suffix;
    // Each choice is scored by the log-probability of its FIRST token. Two choices whose first
    // tokens coincide cannot be told apart, so such a request is refused rather than answered with a
    // tie; single-token labels ("A", "B", ...) avoid the collision by construction.
    std::vector<std::string> choices;
    // There is no think switch: the template is always rendered with reasoning off. A decision reads
    // the first token of the answer, and with reasoning on that token is the reasoning opener, so
    // every choice would be scored against the model starting to think rather than against its answer.
    // Allow this call to restore and refresh the prefix state. Off, the prompt is prefilled whole and
    // the kept state is left untouched: the reference a restored run is compared against.
    bool reuse_prefix = true;
};

// What the prompt phase cost. The same terms, read by the same rules, as the prefill fields of
// RunSummary (bmoe/metrics.h): io is summed lane busy time under overlap, stall is the union of
// stalled intervals, cpu is whole-process.
struct PrefillStats {
    double seconds = 0.0;
    double cpu_seconds = 0.0;
    double read_mib = 0.0;
    double io_seconds = 0.0;
    double stall_seconds = 0.0;
    double mgmt_seconds = 0.0;
    int device_tokens = 0; // prefilled on the prefill device (PrefillDeviceConfig); 0 without one
    // The device's expert arena, apart from the streamer's counters above as in RunSummary: MiB it
    // read, seconds the graph waited on it, and in its routed mode (PrefillDeviceConfig::routed) the
    // experts routed and those read at the routing node because the prediction missed them. All 0
    // without a device.
    double device_read_mib = 0.0;
    double device_stall_seconds = 0.0;
    double device_cache_mib = 0.0; // copied from the expert cache instead of read (from_cache)
    long long device_routed = 0;
    long long device_demand = 0;
};

struct DecideResult {
    bool ok = false;
    std::string error;
    // The session cannot be trusted after this failure (a decode failed mid-prompt). A refused
    // request — no choices, colliding choices, a prompt past n_ctx — leaves it usable.
    bool fatal = false;
    RunError error_kind = RunError::None; // set with `error`; fatal is error_kind == Failed
    bool cancelled = false;               // Session::cancel() stopped the prefill; the session is usable

    // log p(first token of choices[i]) over the WHOLE vocabulary, in the request's order. Not
    // renormalised over the choices: the mass the model puts outside them says how unsure it is,
    // and a caller calibrating an abstention threshold needs exactly that.
    std::vector<double> choice_logp;
    int best = -1; // index of the highest choice_logp

    int n_tokens = 0;    // the rendered prompt
    int n_reused = 0;    // restored from the kept prefix state instead of prefilled
    int n_prefilled = 0; // prefilled by this call (n_tokens - n_reused)
    double restore_seconds = 0.0;
    double store_seconds = 0.0; // copying the state after the prefix into the kept slot; not in prefill
    PrefillStats prefill;
    // Memory the kept prefix state occupies after this call. On a phone it is taken from the same RAM
    // the expert cache lives in, so it is reported rather than left to be inferred.
    size_t prefix_state_bytes = 0;
};

} // namespace bmoe
