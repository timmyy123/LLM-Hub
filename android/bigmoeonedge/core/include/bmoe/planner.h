// The planner: (machine, model, request) -> a resolved run, with its reasons.
//
// A pure function. No I/O, no clock, no llama.cpp, no environment — so it can be driven against a
// synthetic machine in a unit test, which is the only way a claim like "this holds on hardware we
// do not own" can be checked at all. The adapters that measure a HardwareProfile and read a
// ModelProfile are separate on purpose; this file must stay testable without either.
//
// The two invariants the rules are written to keep, and which the tests check directly:
//
//   1. Decline instead of guessing. Every knob whose deciding fact is Unknown keeps its default
//      and says so (Source::Unprobed). No rule invents a number from a platform name — there are
//      no platform names in its inputs to invent one from.
//
//   2. Nothing lossy arms itself. Expert dropping, substitution and route-ahead change the output
//      and make a run irreproducible; they are reachable only through PlanRequest::quality_budget,
//      which the caller sets and the planner never raises.
//
// What this deliberately does NOT do: place tensors across compute devices. That is a capacity
// problem, it is exactly solvable from tensor sizes, and llama.cpp already solves it. This planner
// owns the tier below — what to do when the bytes fit nowhere and have to come off flash.
#pragma once

#include "bmoe/hardware_profile.h"
#include "bmoe/model_profile.h"
#include "bmoe/placement.h"
#include "bmoe/plan.h"

namespace bmoe {

// Resolve a run. `base` supplies the caller's starting point (model path, prompt, and any knob it
// set); `request.pinned` names the knobs the planner may not touch. Never fails: an unreadable
// model or an unprofiled machine yields a plan that declines streaming and explains why.
Plan plan_run(const RunConfig & base,
              const HardwareProfile & hw,
              const ModelProfile & model,
              const PlanRequest & request);

// As above, composed over the first stage: `placement` is what llama.cpp's capacity fitter decided
// (which layers keep experts on the host, how much host memory the placed model takes). The
// second stage sizes itself on what is left. A placement with fitted=false is the four-argument
// form: everything on the host, nothing placed.
Plan plan_run(const RunConfig & base,
              const HardwareProfile & hw,
              const ModelProfile & model,
              const Placement & placement,
              const PlanRequest & request);

// Margins the planner applies as POLICY rather than measurement — how much of the machine to leave
// for everything that is not us. They are exposed here so they are visible and overridable instead
// of buried, and so a test can state what it expects. They are not measured constants and must not
// be presented as any: what IS measured is that the cost of getting this wrong is asymmetric, since
// overcommitting a device costs far more than a slightly smaller cache.
struct PlannerPolicy {
    // Fraction of the residency budget kept free, by what losing memory costs on this machine.
    float margin_when_compressed = 0.10f; // a reclaim is cheap: decompression, no I/O
    float margin_when_swapped = 0.04f;    // a reclaim writes to disk, but the process survives
    float margin_when_killed = 0.20f;     // exceeding the share is fatal: stay well inside it
    float margin_when_unknown = 0.15f;    // no fact: the conservative of the three

    // Never leave less than this free, whatever the fractions say on a small machine.
    uint64_t margin_min_bytes = 256ull << 20;

    // Fraction of the LOCKABLE total left unasked-for, and it is not there to absorb a misreading:
    // a refusal is survivable, the budget steps down to what is granted. It is there because the
    // platform's limit is not a place to run. What is locked is taken from everything that is
    // not - this process's own context and compute buffers included - and a run that sits on the
    // limit leaves those to be reclaimed. Measured on one 16 GB machine, same model and cache hit
    // rate: 23.1 tok/s asking for 95% of the total, 7.4 and 14.6 asking for 98%.
    float lock_margin = 0.05f;

    // How much room a model must leave BEYOND itself before residency is called safe, as a multiple
    // of what it would hold. Fitting and being left alone are different claims: on a machine whose
    // reclaim can take pages back, a model that fits just barely is reclaimed from underneath and
    // refaults its weights one page at a time, which is measured at 0.1 tok/s against 5.0 for the
    // same model streamed. "Leave as much as you take" is the cheapest test that separates the two
    // without a measurement; the headroom probe replaces it with one. It does not apply where
    // nothing can take the memory back, nor where the limit is a hard cap the margin already sizes.
    float fits_air_ratio = 1.0f;

    // How much of what the process may hold the compute buffers may take, as a fraction. They are
    // reserved for the widest graph a prompt is computed in, and a prompt is computed once where a
    // token is decoded every time: memory given to the first is taken from the expert cache, which
    // the second reads. Below this share the graph stays as wide as the context; above it the
    // width is halved until it fits. Policy, not measurement, and deliberately on decode's side:
    // on one 16 GB machine a 16384 context cost 5 GB of buffers at full width and the decode a
    // third of its rate, while a quarter of the width kept the rate and nine tenths of the prefill.
    float compute_share_max = 0.10f;
    // The narrowest graph the width is brought down to. llama.cpp's own physical batch: nothing
    // here has been measured below it, and under it a prefill is mostly passes over the experts.
    uint32_t ubatch_min = 512;
    // The widest graph a device with memory of its own is planned at: an evidence boundary, not a
    // measurement of this machine. A device's expert matmul can hold a whole pass's tokens in a
    // per-block scratch whose size the backend fixes, and past it the pass aborts the process (#533:
    // `GGML_ASSERT(nbytes_shared <= smpbo)`, ~25k tokens on 24-32 GB cards, ~16k where the scratch is
    // 64 KB). Measured on a 32 GB card at a 32768-token graph: a 28.6k-token prompt aborted the
    // server. Half of the smallest of those limits, so no shipped backend has been shown to fail at
    // it; a prompt past it takes more passes, which cost little once a pass is this wide.
    //
    // Two bounds hold the width, and this is the planned one. A run nothing planned opens at most
    // RunConfig::ubatch_auto_max wide, whatever computes it; a plan starts from the whole context,
    // priced from the probe's measurement, and narrows for memory, so on a device this bound is what
    // stops it short of the context. It is no wider than that default, so a device's graph is never
    // planned past the width the fitter placed its layers against (planner_test checks the order).
    // "A device computes the graph" is any of: layers on a device of its own, a device expert cache, or
    // a prefill device (one the plan arms is held to it once armed). The image floor
    // (RunConfig::graph_width_floor) does not lift it: an image past it is refused instead.
    uint32_t ubatch_max_device = 8192;

    // The shortest context a plan shortens one to, where the person set none and the one given leaves
    // the expert cache under a token cycle (plan_on_facts). llama.cpp's own: the shortest context its
    // capacity fitter shrinks to before it moves weights instead (common's --fit-ctx default). A
    // context the person set is never touched.
    uint32_t ctx_min_auto = 4096;

    // Whether THIS ENGINE can hand a device the memory a streamed expert is made of. It is a fact
    // about our own streamer rather than about any machine, which is why it lives here and not in
    // the hardware profile - and it is false, because the streamer reserves address space and
    // commits pages itself, so the bytes a rebound tensor points at were never allocated by, or
    // registered with, any device.
    //
    // The rule that would route expert compute to a device reads this first, and that is deliberate
    // rather than defensive: without it the plan would name a device on a machine where the device
    // would then read memory it cannot see. A knob that is right in principle and wrong in practice
    // is worse than one that is off, so the precondition is written down and the day the streamer
    // allocates through `ggml_backend_dev_buffer_from_host_ptr` this becomes true and one condition
    // flips. The tests exercise both settings.
    bool streamer_serves_device_memory = false;

    // ── the cost model's coefficients ───────────────────────────────────────────────
    // How many of a token's expert reads a cache serves, credited as this multiple of the FRACTION
    // OF THE EXPERT SET IT HOLDS. 1.0 is the neutral reading: hold a tenth, serve a tenth. Only where
    // the model's token cycle is unknown: otherwise a cache is credited this model's own curve, or the
    // prior in token cycles that five models' curves agree on (prior_hit_curve, bmoe/routing.h), and
    // the single-point argument below no longer holds.
    //
    // The relationship is really super-linear, because routing has strong temporal locality — the
    // one measurement on record is a cache holding 13% of the set serving 61.9% of the reads — so
    // this default is conservative by roughly 5x and deliberately so: over-crediting the cache
    // spends memory on a saving that cannot be collected, and the error is asymmetric. Fitting a
    // curve to a single point would be inventing the shape, so the shape stays linear and the
    // coefficient is what a closed loop corrects from a run's own counters. An earlier 0.22 here
    // was 20x low against that same measurement and made the cache look nearly worthless, which on
    // a machine where cache and residency compete evenly is a mis-allocation rather than caution.
    float cache_hit_optimism = 1.0f;

    // How much faster than the host a device must be before its win is believed, as a fraction. A
    // tenth, not a hair: the two figures come from the same probe on the same machine, and claiming
    // "more" for a difference inside its own repeatability is how a rationale starts being read as
    // noise. Measured 19 against 19 on a phone, which is not a finding.
    float min_backend_win = 0.10f;

    // How much of its ceiling the capacity fitter's placement is priced at. The ceiling adds what a
    // token reads at each processor's probed rate and leaves out the crossings between the device and
    // the host's experts and every per-op cost, so the placement decodes under it, by a part no probe
    // gives. Measured where a load ran the placement after its ceiling was computed: 46.5 tok/s under
    // a ceiling of 175 on a 32 GB card, 32.1 under 43.5 and 62.1 under 83 on 12 GB cards, 20.6 under
    // 39.4, 2.2 under 13.1, 11.3 under 13.9: from 0.17 to 0.81 of it, a half on average. Priced at the
    // whole ceiling, the placement took the second load from a cache measured at 102 tok/s to decode
    // it at 46.
    float fitter_ceiling_realized = 0.5f;

    // What a request is weighed as on a machine that has served none of this model yet, in tokens: the
    // prompt prefilled and the answer decoded. A decision that trades the prompt against the answer
    // (the graph's width, a device that shares the host's memory) needs both, and a first load has no
    // record of its own to read them from; every load after it uses the requests this machine served
    // (MeasuredKnobs::mix_*). A chat or agent turn with a few pages of context and a short answer: a
    // one-line question alone would weigh the prompt at nothing, and a width that loses a long prompt
    // 22-47% (measured on 8, 12 and 2x16 GB cards) would pass as free.
    float mix_prompt_tokens_default = 4096.0f;
    float mix_gen_tokens_default = 300.0f;
    // How many requests the default mix counts as beside the ones a load recorded. A load's record is
    // the last session's alone and often a handful of requests, and a decision fixed at load that
    // followed it swung with it: on two 16 GB cards four requests, one of them 4.6k tokens long, read
    // as "1181 prompt tokens", the next load's three short ones as "34", and the graph went narrow and
    // lost 41% of the long prompt's prefill. Weighed in with the default as this many requests, a few
    // requests move the mix a little and a machine that serves hundreds of one kind is read as that.
    float mix_prior_requests = 16.0f;

    static PlannerPolicy defaults() { return PlannerPolicy(); }
};

// The widest graph, halving from the width `placement` was projected at and never below
// `policy.ubatch_min` (nor `floor`, the narrowest an image allows: RunConfig::graph_width_floor), whose
// compute buffers on every device with memory of its own stay inside `policy.compute_share_max` of
// that device's memory (each card against its own). The projected width where they already do; 0
// where the device side's slope was not measured or no device has memory of its own. Read by the
// width rule and by probe_and_plan, which asks the fitter what that width buys.
uint32_t device_share_width(const Placement & placement,
                            const HardwareProfile & hw,
                            const PlannerPolicy & policy,
                            uint32_t floor = 0);

// As above, with the margins overridden. The four-argument form calls this with the defaults.
Plan plan_run(const RunConfig & base,
              const HardwareProfile & hw,
              const ModelProfile & model,
              const Placement & placement,
              const PlanRequest & request,
              const PlannerPolicy & policy);

} // namespace bmoe
