// Does a model open on devices answer as the same file does on the cores?
//
// A plan lets a device through on probes that each compute a part of a model in isolation: a
// synthetic matmul, one layer's matrices. None of them is the model's output, and a device can pass
// every one and still not compute a given model (KnobTrialConfig::answer_device has the measurement).
// The only instrument that sees the whole model is the model: this decodes one short fixed sequence
// on the placement as it was opened and on a host-only instance of the same file, and compares the
// next-token rows with the criterion a prefill device is confirmed by (logit_agreement.h).
//
// Both of a backend's paths are in the sequence: most of it goes through as one graph, the way a
// prompt does, and the last tokens one at a time, the way an answer is decoded. A backend's batched
// kernel is not its one-token path, and a model served from a device runs both there.
//
// Nothing here knows which device, backend, architecture or quantization it is looking at: it takes a
// context whose model has layers on devices, and a way to open the same file with none.
#pragma once

#include "bmoe/config.h"
#include "bmoe/session.h"
#include "logit_agreement.h"

#include "ggml.h"
#include "llama.h"

#include <functional>
#include <string>
#include <vector>

namespace bmoe::detail {

// What is decoded, and how much of it is compared. A backend picks its kernels by a graph's width, and
// where it changes kernel is its own business (one for a single token, others for a few, another for
// many), so the sequence goes through at every width from one token to `k_answer_narrow_max` and then
// once as wide as the prefill device's confirmation: the widths a session's own graphs take, from an
// answer's tokens and a short prompt's to a long prompt's. The narrow pieces come first, so a width
// that computes wrongly also leaves its wrong state in the cache every later piece reads. Of each
// piece the rows a session's graph of that width returns are compared: the last of a narrow one, and
// of the wide one as many as the output head's bound allows (PrefillDeviceConfig::verify_rows).
constexpr int k_answer_narrow_max = 8;
constexpr int k_answer_wide_tokens = PrefillDeviceConfig::verify_tokens;
constexpr int k_answer_rows = PrefillDeviceConfig::verify_rows;
// The tokens each side is then timed on, one at a time, as an answer is decoded: enough for a median
// that the first steps' warm-up (a backend's kernels compiled, pages read) does not decide.
constexpr int k_answer_timed_tokens = 16;
// The tokens in memory before the timed ones, so they attend to a context and not to nothing.
constexpr int k_answer_timed_prefix = 8;
// The pieces' widths in order, for a context of `n_ctx` tokens taking `n_batch` a call.
std::vector<int> answer_piece_widths(int n_ctx, int n_batch);
// Rows compared of a piece `width` tokens wide.
int answer_rows_of(int width);

// The fixed sequence a device is checked on, `want` tokens of it: ordinary prose through the model's
// own tokenizer, repeated to the width asked. Prose, so the model's next-token choices are confident
// in most rows and a device that computes the same model agrees with the host on them; a device whose
// numerics break the model agrees on almost none. Empty where the text does not tokenize.
std::vector<llama_token> answer_probe_tokens(const llama_vocab * vocab, int want);

struct DeviceAnswer {
    enum class Verdict { Unchecked, Differs, Confirmed };
    Verdict verdict = Verdict::Unchecked;
    std::string unchecked; // why nothing was compared, where nothing was
    LogitAgreement agreement;
    int tokens = 0;        // the sequence decoded on each side
    int pieces = 0;        // the graphs it went through as
    int wide = 0;          // the widest of them
    double device_s = 0.0; // the sequence on the placement
    double host_s = 0.0;   // the reference: its model opened, the sequence decoded
    // The two sides' speed on this model, where the check was asked to time them (`timed`): the median
    // seconds of a token decoded alone, and the seconds the wide piece took. 0 where not timed, or
    // where that side's timed decode failed. The comparison is the model's own, on the memory, the
    // kernels and the threads the session will serve with, which no probe of a part of a model is.
    double device_token_s = 0.0, host_token_s = 0.0;
    double device_wide_s = 0.0, host_wide_s = 0.0;
};

// Which side serves a request sooner, by the check's own timing.
//
// A request is its prompt and its answer, so a side is weighed on both: the seconds one decoded token
// takes there, plus the seconds of the prompt tokens that come with it (`prompt_per_token` of them a
// decoded token, at the rate the wide piece ran). The placement is the side already open: the cores
// take the model only where a request is shorter there by `margin`, because a reopening is paid for
// and a few tokens' timing is not a whole answer's. Unweighed where either side was not timed.
enum class AnswerSide { Unweighed, Device, Cores };
struct AnswerWeighing {
    AnswerSide side = AnswerSide::Unweighed;
    double device_request_s = 0.0, cores_request_s = 0.0; // per decoded token, its share of the prompt included
};
AnswerWeighing weigh_answer_sides(const DeviceAnswer & a, double prompt_per_token, double margin);

// How the reference is opened: the same file with no layer on any device, loaded the way the session
// loads (mapped, no repack: hard rule 2), which only the session knows how to spell. The check owns
// what `load_model` returns and frees it before it returns. The cache types are the placement's, so
// the two sides differ in where they compute and in nothing else.
struct AnswerReference {
    std::function<llama_model *()> load_model;
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    int n_threads = 1;
    int n_threads_batch = 1;
};

// Compare `placed` (a context over a model with layers on devices) with the reference. Leaves
// `placed` with an empty memory and nothing of the reference alive. `test_differs`
// (KnobTrialConfig::test_answer_differs) shifts the placement's rows by one token before comparing.
// `timed`: also time each side (DeviceAnswer::device_token_s and the rest), once its answers matched.
DeviceAnswer
check_device_answer(llama_context * placed, const AnswerReference & ref, bool test_differs, bool timed = false);

// The configuration a session reopens with where a device's answers were not confirmed: the same one,
// with no layer on any device and no device in the room, and the load's timed side (if the device was
// one) now the cores'.
SessionConfig on_cores(SessionConfig cfg);

} // namespace bmoe::detail
