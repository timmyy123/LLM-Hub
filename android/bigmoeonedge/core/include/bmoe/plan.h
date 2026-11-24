// A resolved run, and the reason for every part of it.
//
// The rationale is not decoration. An engine that configures itself is only usable if a run can be
// explained afterwards — otherwise two benchmarks of "the same" build are not comparable, and a
// regression cannot be attributed. So a Plan carries, for every knob it touched, the fact that
// decided it; the CLI prints that, and the CSV header records it.
//
// It is also how "decline instead of guessing" stays visible: a knob left at its default because
// nothing was measured produces a Decision with Source::Unprobed, which is a standing list of the
// measurements the project still owes itself.
//
// Pure policy: no llama.cpp, no I/O.
#pragma once

#include "bmoe/allocate.h"
#include "bmoe/config.h"
#include "bmoe/placement.h"

#include <string>
#include <vector>

namespace bmoe {

// Where a decision's authority came from. The ordering is the confidence ordering.
enum class Source {
    Measured, // a probed fact on this machine decided it
    Derived,  // arithmetic over model shapes and machine facts (a cache floor, a token cycle)
    Policy,   // a deliberate margin or preference of ours, not a measurement
    Operator, // the caller pinned it; the planner did not choose
    Unprobed, // nothing was known: the knob keeps its default and this says so
};

const char * source_name(Source s);

// One knob, what it became, and why.
struct Decision {
    std::string knob;  // the flag or field name, as a user would type it
    std::string value; // the chosen value, rendered
    Source source = Source::Unprobed;
    std::string reason; // the full account: what --explain prints, and a screen's "why"

    // One plain sentence for a person, carrying the number that decided it: no flag names, no
    // planner vocabulary. A screen leads with this and folds the reason away; the reason alone is
    // written for whoever is checking the arithmetic.
    std::string brief{};
};

// What the planner was asked for, on top of the two profiles. Everything here is the caller's
// authority rather than the machine's, which is why it is a separate input: a plan must never
// silently overrule a person.
struct PlanRequest {
    // Knobs the caller set explicitly. The planner leaves every one of these alone and records it
    // as Source::Operator. Matching llama.cpp's own fitter, which only touches parameters still at
    // their default — a convention worth inheriting rather than reinventing.
    std::vector<std::string> pinned;

    // Quality the caller is willing to spend, as a fraction of perplexity increase. 0, the default,
    // forbids every lossy policy. Nothing the planner does may raise this on its own: a run that
    // silently traded accuracy for speed is a run nobody can reproduce or trust.
    float quality_budget = 0.0f;

    // Measure the lossless knobs on the first real request (RunConfig::trial) when nothing measured
    // them before (ModelProfile::knobs). Set by a front-end that keeps the result, so the second load
    // starts from it; off, the plan arms no measurement and is the plan it always was.
    bool measure_knobs = false;

    bool is_pinned(const char * knob) const;
};

// The regime a machine and a model land in together. Not a machine property and not a model
// property: the ratio of the two, which is why the same phone is in different regimes for
// different files and the same file is in different regimes on different machines.
enum class Regime {
    Unknown,       // the model could not be read
    NotMoe,        // nothing here to stream
    Fits,          // the whole model sits inside the residency budget: streaming only adds cost
    ExpertsStream, // the dense set fits and the experts do not: what this engine is for
    DenseOversized // not even the dense set fits: the policy has to degrade too
};

const char * regime_name(Regime r);

struct Plan {
    RunConfig config;    // the resolved run, ready to hand to the engine
    Placement placement; // what the first stage decided, for the session to apply at load
    Regime regime = Regime::Unknown;
    std::vector<Decision> decisions;

    // Set when streaming was considered and refused. A refusal is a plan: the config it carries is
    // still runnable, and the reason is what stops the caller from concluding the engine is slow.
    bool streaming_declined = false;
    std::string decline_reason;

    // Numbers the rationale quotes, kept so a caller (the CSV header, the app) can record them
    // without re-deriving them from the profiles.
    uint64_t token_cycle_bytes = 0;
    uint64_t cache_budget_bytes = 0;
    uint64_t dense_pending_bytes = 0;

    // Where every group of weights ended up and what that is predicted to cost. The config above is
    // what the engine can be TOLD; this is what the plan actually decided, and the two differ
    // wherever the engine has no way to express a per-group choice. Keeping both is what makes a
    // prediction checkable against the run it produced.
    Allocation allocation;

    // Human-readable multi-line rationale: one line per decision, plus the regime, the allocation
    // and the decline. This is what `--plan` prints.
    std::string explain() const;

    // The plan as a bmoe-cli command line: the form in which it can be pasted, edited, diffed
    // against a hand-tuned run and put in a bench cell. A plan you cannot re-type by hand is a plan
    // you cannot argue with, and arguing with it one lever at a time is how everything this
    // repository knows about performance was learned.
    //
    // Every parameter of the resolved config that is not at its default, spelled by the parameter
    // table (`to_args`) and quoted for a shell; the model path is left to the caller. A knob left
    // at its default is left out rather than spelled with its default value, so the line reads as
    // the decisions and not as a dump. A plan that declined streaming renders as a comment.
    std::string to_flags() const;

    // The same arguments one per element and unquoted, for a caller that passes them on rather
    // than printing them. Empty for a plan that declined streaming.
    std::vector<std::string> reproduce_args() const;
};

} // namespace bmoe
