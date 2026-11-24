// What a session measured about this model's lossless knobs on this machine, kept between loads.
//
// The decode thread count and whether reads overlap compute change no output, and which setting
// decodes faster is a property of this machine running this model, not something a rule over
// either can predict: the thread rule from core classes was shipped and withdrawn in a day, and
// overlap measured +5% on one phone and +8-14% on one laptop with nothing in the plan able to say
// where it stops paying. So the first real request of a session measures them, interleaved on the
// user's own answer (RunConfig::trial), and hands the winner back as a value. The front-end keeps it
// against the model file's fingerprint, beside the model's routing; the next plan of the same model
// on the same machine and engine starts from it and measures nothing. See
// docs/measured-planning.md (Step 2).
//
// Pure: no llama.cpp, no I/O, no clock. The text format is a string in and a string out; where it
// is stored is the front-end's decision.
#pragma once

#include "bmoe/config.h"
#include "bmoe/hardware_profile.h"

#include <string>
#include <vector>

namespace bmoe {

struct MeasuredKnobs {
    // Whose measurement this is: the machine (hardware_fingerprint) and the engine version
    // (bmoe::version()) it was measured with. The model is the file it is kept under.
    std::string hardware;
    std::string engine;

    // The decode thread count that won, 0 when not measured. `threads_tried` lists every count the
    // trial decoded with, the incumbent first; `threads_gain` is how much faster the winner decoded
    // than the incumbent, or, when the incumbent held, the best candidate's figure (below the margin).
    int threads = 0;
    std::vector<int> threads_tried;
    double threads_gain = 0.0;

    // Overlap: -1 not measured, 0 off, 1 on. `overlap_incumbent` is the setting the trial started
    // from, and `overlap_gain` how much faster the other one decoded than it (below the margin: the
    // incumbent held).
    int overlap = -1;
    int overlap_incumbent = -1;
    double overlap_gain = 0.0;

    int rounds = 0; // timed windows per candidate behind each verdict

    // The device tier, decided across loads rather than inside one: a device expert cache is fixed
    // when a context is created, so the two placements cannot alternate within an answer. The plan
    // prices both and names the faster (planner.cpp, choose_device_tier); the load that serves a side
    // measures its decode rate as it serves, and a plan made after it reads that side at its
    // measurement. Tokens per second, 0 when that side was not measured. A rule could not
    // tell the two apart: a cache holding a tenth of a 355B model's experts on a PCIe 3 link decoded
    // at 1.6 tok/s where the fitter's placement did 3.9, and a cache on a 12 GB card won 17.5 to 9.9.
    double device_cache_tps = 0.0;
    double no_device_cache_tps = 0.0;

    // The device cache's CPU share, decided across loads as the tier is (planner.cpp,
    // choose_device_cache_cpu_share): the decode rate with the share the plan priced, and with every
    // miss uploaded, in tokens per second, 0 when that side was not measured. A price proposes the
    // share and these decide it, because the price cannot see what a share costs on a machine whose
    // cache never fills: on a 32 GB card a share priced at 0.89 took the decode from 114 to 68 tok/s
    // with no miss computed on the host at all, and on a 12 GB card 0.77 took it from 18.5 to 21.2.
    double cpu_share_tps = 0.0;
    double no_cpu_share_tps = 0.0;

    // Where a model held in memory runs on a machine whose devices read the host's own memory, decided
    // across loads (planner.cpp, choose_shared_device): the decode rate with every layer on the device
    // and on the cores, in tokens per second, 0 when that side was not measured. Capacity does not
    // decide it, since the model is in the same memory either way, and no probe of a part of the model
    // predicts the whole: on one unified-memory laptop the one-token probe read the cores 60% ahead while
    // the model itself decoded 54 tok/s on the GPU against 40 on the cores.
    double shared_device_tps = 0.0;
    double no_shared_device_tps = 0.0;
    // The rate the share is held against at graph width `n_ubatch`: its own timing of every miss
    // uploaded, else the width's timing of the cache at that width (the width is decided on the cache
    // before the share is weighed), else the tier's. 0 where no side of the cache was timed.
    double cpu_share_baseline_tps(int n_ubatch) const {
        if (no_cpu_share_tps > 0.0) return no_cpu_share_tps;
        const double at_width =
            width_narrow > 0 ? (n_ubatch == width_narrow ? narrow_decode_tps : wide_decode_tps) : 0.0;
        return at_width > 0.0 ? at_width : device_cache_tps;
    }

    // The graph width, decided across loads where a wide graph's compute buffers take device memory
    // the experts would use (planner.cpp, choose_graph_width). A wide graph prefills a long prompt in
    // one pass; a narrow one leaves the experts more of the card. Which wins is a property of this
    // machine AND of how it is used: on two 24 GB cards with GLM-4.5-Air the narrow graph decoded 37%
    // faster, and the wide one prefilled a 9.5k-token prompt several times faster. So each load times
    // both halves, and the plan weighs them by the requests this machine actually serves.
    // `width_narrow` is the narrow width compared (0: none); the rates are tokens per second, the
    // prefill ones over prompts longer than `width_narrow` only (where the width makes a difference),
    // 0 when that side or that kind of prompt was not seen.
    int width_narrow = 0;
    double wide_decode_tps = 0.0, narrow_decode_tps = 0.0;
    double wide_prefill_tps = 0.0, narrow_prefill_tps = 0.0;
    // What the requests of the last load that measured a side looked like: per request, the tokens
    // decoded and the prompt tokens beyond the narrow width (prefilled after reuse), over `usage_requests`.
    double usage_gen_tokens = 0.0, usage_long_prompt_tokens = 0.0;
    int usage_requests = 0;
    // The width a load ran as decided, not as a side timed: 1 the full width, 0 `width_narrow`, -1 none
    // recorded. The margin a decision asks for guards this one: the other width takes over only when it
    // is faster by the margin, so the same measurements weighed again do not swing the graph from one
    // load to the next (two 16 GB cards, DeepSeek-V4-Flash: narrow 11% ahead on the decode alone, then
    // 9% on the requests served, back to the full width and 4 GB less device cache).
    int width_held = -1;

    // What the decode itself cost here, the coefficient a run's own counters correct
    // (docs/measured-planning.md, "The decode's own cost"). The plan prices decode compute from a
    // probed matmul rate and the cache from a replay of the routing, and neither sees all a token
    // pays: a one-token decode on a phone also pays per-op and thread-synchronisation costs the
    // probe does not. Measured on a 12 GB phone, Ling-3.0-tiny Q8_0: predicted 0.037 s/token at a 95%
    // credited hit rate, measured 0.107 at 77-85%, compute alone ~64 ms. So every load times its
    // decode once the cache has settled (the trial's windows, after any knob trial so no switch is in
    // them), and the next plan calibrates its prices by it (planner.cpp, calibrate_decode_cost).
    //
    // `decode_compute_s`: the median window's compute seconds per token, flash waits and cache
    // management excluded (the CSV's compute_ms); 0 when not measured. `decode_threads` is the decode
    // thread count it ran at, `decode_hit` the expert cache's hit rate over the same windows (-1: not
    // seen) at a host cache of `decode_cache_mib`, and `decode_gpu_layers` / `decode_device_cache_mb`
    // the placement it ran under: a figure only answers for a plan of the same configuration.
    double decode_compute_s = 0.0;
    int decode_threads = 0;
    double decode_hit = -1.0;
    double decode_cache_mib = 0.0;
    int decode_gpu_layers = 0;
    int decode_device_cache_mb = 0;

    // How this machine is used, recorded on every load that answered a request: what a trade between
    // the prompt and the answer is priced by (planner.cpp, fit_prefill_device). A prefill device whose
    // room takes the expert cache away speeds every long prompt and slows every decoded token, and
    // which wins is the requests', not the hardware's: on a 12 GB phone a 1054-token prompt prefilled
    // in 122 s on the cores, where the same phone's NPU prefilled another model 8x faster, and a long
    // answer from a short prompt is the opposite case. Per request, over `mix_requests` of the last
    // load that served any: `mix_gen_tokens` decoded, and `mix_wide_prompt_tokens` prefilled (after
    // reuse) in prompts at least as wide as the narrowest graph a device is handed
    // (PrefillDeviceConfig::min_tokens): the part a device would take, since shorter prompts run on
    // the host either way. `host_prefill_tps` is the host's rate over those prompts where the host ran
    // them, 0 when it ran none.
    int mix_requests = 0;
    double mix_gen_tokens = 0.0, mix_wide_prompt_tokens = 0.0;
    double host_prefill_tps = 0.0;

    // Failed opens of each side of a decision across loads, on this machine and engine (the record's
    // own scope): a served load that could not load that configuration, most likely short of device
    // memory, and opened the plan's other side in its place. Kept between loads, since the next plan
    // would otherwise name the side again and every load would open it, fail, and fall back. A side
    // loses in every plan made on the record once it has failed `unviable_max` times: one failure may
    // be the machine's moment (another program holding the card's memory, a driver's hiccup), so the
    // side is opened once more, with the other still behind it; two are the side. A rate measured for
    // the side clears it, and a new engine version starts a new record. The width's two are for
    // `width_narrow` only, as its rates are.
    static constexpr int unviable_max = 2;
    int device_cache_unviable = 0, no_device_cache_unviable = 0;
    int cpu_share_unviable = 0, no_cpu_share_unviable = 0;
    int wide_unviable = 0, narrow_unviable = 0;
    int shared_device_unviable = 0, no_shared_device_unviable = 0;

    // What came of each prefill device a load armed for this model, on this machine and engine, one
    // word per device: `verdict` 1 where the session kept it, -1 where its check on this model refused
    // it, 0 where a load armed it and never said or could not open it - a front-end writes that before
    // the session opens, so a load that never returns leaves it - with `unconfirmed` counting such loads
    // in a row. A plan does not arm a device refused here, nor one that many loads in a row never kept
    // (planner.cpp, survey_prefill): one may be a person closing the app mid-load or another program
    // holding the card's memory for a moment, two in a row are the device. One word per device, so a
    // second device's word never clears the first's: a machine with two that qualify would otherwise
    // arm the refused one again every other load. Measured on a 12 GB phone: an NPU that gives one
    // model wrong results (its top token in 25 of 31 positions) and, with that backend's host buffer
    // on, never returns from that model's first layer. A new engine starts a new record, and tries the
    // device again.
    struct PrefillWord {
        std::string device;
        int verdict = 0;
        int unconfirmed = 0;
    };
    static constexpr int prefill_unconfirmed_max = 2;
    std::vector<PrefillWord> prefill;
    const PrefillWord * prefill_word(const std::string & device) const {
        for (const PrefillWord & w : prefill)
            if (w.device == device) return &w;
        return nullptr;
    }
    // Replaces `device`'s word, or adds one.
    void set_prefill_word(const PrefillWord & w) {
        for (PrefillWord & x : prefill)
            if (x.device == w.device) {
                x = w;
                return;
            }
        prefill.push_back(w);
    }
    bool prefill_refused(const std::string & device) const {
        const PrefillWord * w = device.empty() ? nullptr : prefill_word(device);
        return w && (w->verdict < 0 || (w->verdict == 0 && w->unconfirmed >= prefill_unconfirmed_max));
    }
    // The record a front-end keeps before opening a session that arms `device`: this device's count of
    // loads that armed it and never kept it, one more.
    MeasuredKnobs armed_unconfirmed(const std::string & device) const {
        MeasuredKnobs r;
        r.hardware = hardware;
        r.engine = engine;
        const PrefillWord * w = prefill_word(device);
        r.prefill.push_back({device, 0, (w && w->verdict == 0 ? w->unconfirmed : 0) + 1});
        return r;
    }

    // Whether this model's answers on a device matched the cores', on this machine and engine, one
    // word per device: `verdict` 1 where a load compared them and they matched, -1 where they differed;
    // a device with no word was never compared. The probes a plan reads each compute a part of a model
    // in isolation (a synthetic matmul, one layer's matrices), and a device can pass every one of them
    // and still not compute a given model: on a 16 GB Apple-silicon laptop one small vision model wrote
    // one repeated character with every layer on the GPU and sane text on the cores, where two other
    // models were correct on the same GPU. So a load whose plan moved the whole model onto a device
    // decodes a short sequence on both before it serves (KnobTrialConfig::answer_device), and the word
    // is kept: a plan does not move this model onto a device that differed here, and does not ask a
    // device that matched to be compared again (planner.cpp, choose_shared_device). A new engine starts
    // a new record, and compares again.
    struct AnswerWord {
        std::string device;
        int verdict = 0;
    };
    std::vector<AnswerWord> device_answers;
    // 1 matched, -1 differed, 0 never compared.
    int device_answer(const std::string & device) const {
        for (const AnswerWord & w : device_answers)
            if (w.device == device) return w.verdict;
        return 0;
    }
    // Replaces `device`'s word, or adds one. A word that says nothing (0) is not kept.
    void set_device_answer(const std::string & device, int verdict) {
        if (device.empty() || verdict == 0) return;
        for (AnswerWord & w : device_answers)
            if (w.device == device) {
                w.verdict = verdict;
                return;
            }
        device_answers.push_back({device, verdict});
    }

    bool empty() const { return verdicts() == 0; }
    // How many measurements this record carries: a front-end keeps a record again when it grows. The
    // requests a width trial has seen count too, so the usage it weighs the two widths by is kept as it
    // grows rather than as the first request left it; so do the requests of the mix.
    int verdicts() const {
        return (threads > 0) + (overlap >= 0) + (device_cache_tps > 0.0) + (no_device_cache_tps > 0.0) +
               (cpu_share_tps > 0.0) + (no_cpu_share_tps > 0.0) + (wide_decode_tps > 0.0) + (narrow_decode_tps > 0.0) +
               (wide_prefill_tps > 0.0) + (narrow_prefill_tps > 0.0) + usage_requests + (width_held >= 0) +
               (decode_compute_s > 0.0) + (decode_hit >= 0.0) + mix_requests + (host_prefill_tps > 0.0) +
               (device_cache_unviable > 0) + (no_device_cache_unviable > 0) + (wide_unviable > 0) +
               (narrow_unviable > 0) + (cpu_share_unviable > 0) + (no_cpu_share_unviable > 0) +
               (shared_device_tps > 0.0) + (no_shared_device_tps > 0.0) + (shared_device_unviable > 0) +
               (no_shared_device_unviable > 0) + (int) prefill.size() + (int) device_answers.size();
    }
};

// The hardware facts a stored measurement is valid for, as 16 hex digits: the core count and
// classes, the host's total memory, and each compute device's name and own memory. Facts that do
// not move between loads, never a rate or a free-memory figure: what was measured about the
// machine stays valid when the memory that happened to be free does not.
std::string hardware_fingerprint(const HardwareProfile & hw);

// The records kept for one model, one line per (machine, engine), and back. False with `error` set
// when the text is not this format.
std::string format_measured_knobs(const std::vector<MeasuredKnobs> & records);
bool parse_measured_knobs(const std::string & text, std::vector<MeasuredKnobs> & out, std::string & error);

// Put `rec` in `records`, merged into the one of the same machine and engine: each measurement `rec`
// carries replaces that one, and what it does not carry is kept. A load measures what its plan left
// unmeasured, so the record of the load before holds the rest.
void keep_measured_knobs(std::vector<MeasuredKnobs> & records, const MeasuredKnobs & rec);

// The decode rate of a load that ran the side `trial` names (KnobTrialConfig::device_tier, width_side,
// cpu_share_side or shared_device_side), put where a plan reads it. One decision per load, in that
// order, so a rate never answers two questions measured under one placement. Nothing when `trial`
// names no side.
void record_side_rate(const KnobTrialConfig & trial, double tokens_per_second, MeasuredKnobs & rec);
// The same side, one more failed open on its count (MeasuredKnobs::*_unviable).
void record_side_unviable(const KnobTrialConfig & trial, MeasuredKnobs & rec);
// The failed-open count of the side `trial` names, null when it names none.
int * unviable_count(const KnobTrialConfig & trial, MeasuredKnobs & rec);
// Whether `trial` names a side of a decision across loads: the load serves that side and times it.
bool names_side(const KnobTrialConfig & trial);

// The record of this machine and engine, or null.
const MeasuredKnobs * find_measured_knobs(const std::vector<MeasuredKnobs> & records,
                                          const std::string & hardware,
                                          const std::string & engine);

} // namespace bmoe
