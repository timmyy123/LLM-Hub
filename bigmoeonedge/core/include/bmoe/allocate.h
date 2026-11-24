// The cost model and the allocation: which memory each group of weights gets, and who computes on
// it, chosen by what it is worth rather than by what fits.
//
// This is the part of planning that is a TRADE rather than a reading. Most knobs in a plan are
// decided by one fact — the lane count by the storage curve, o-direct by whether uncached reads
// return correct bytes, the thread count by a sweep — and for those a rule that reads the fact is
// the right shape. The four decisions here are not like that: RAM given to the dense set is RAM
// taken from the expert cache, a group moved to a device frees host memory and adds graph
// crossings, and a narrower prefill hands its reservation to the weights. They compete, so they
// have to be priced against each other in ONE unit.
//
// That unit is SECONDS PER BYTE OF WEIGHTS. It is available because batch-1 decode is
// bandwidth-bound: every weight is read once and multiplied once, so a GEMV's cost tracks the bytes
// it sweeps exactly as a read's does, and flash, RAM and device memory become directly comparable.
// It is what lets the flash tier enter the objective function at all — which is the one thing no
// other placement solver does.
//
// Three rules keep this from becoming an oracle, and they are load-bearing rather than decorative:
//
//   1. An unmeasured number may never justify a move. A missing fact makes a candidate ineligible,
//      never cheap. Every decline says which measurement it was owed.
//   2. A non-CPU backend must win by a MARGIN, not merely win. Half the plausible levers this
//      engine has tried lost on measurement; a planner built to believe its own model would
//      rediscover every one of them with confidence.
//   3. The allocation that streams nothing must be reachable, because residency beats streaming
//      whenever residency is available.
//
// Pure policy: no llama.cpp, no I/O, no clock. Driven against synthetic machines in the tests.
#pragma once

#include "bmoe/hardware_profile.h"
#include "bmoe/model_profile.h"

#include <cstdint>
#include <string>
#include <vector>

namespace bmoe {

// How a group's bytes reach the compute. Not every lane can serve every group: GroupDemand's
// `streamable` and `row_gatherable` say which are legal, and the allocator never proposes a lane
// the engine cannot execute.
enum class Lane {
    Mmap,         // left mapped; the kernel serves it and may drop it, refaulting a page at a time
    Resident,     // copied into our own anonymous buffers; a reclaim costs whatever a reclaim costs here
    Pinned,       // as Resident, in a store the kernel may not take back (platform-gated)
    RowStream,    // address space only; the rows the graph gathers are pulled from flash
    ExpertStream, // the routed top-k slices are read from flash into a bounded cache
    count,
};

const char * lane_name(Lane l);

// One group's decision, with the arithmetic that produced it kept attached. `reason` is not a log
// line: it is the part a person reads to decide whether they believe the plan.
struct GroupPlacement {
    WeightGroup group = WeightGroup::Other;
    Lane lane = Lane::Mmap;

    // Index into HardwareProfile::devices, or -1 when the group's compute stays on the host. A
    // non-host index is a RECOMMENDATION until the engine can express it; see `device_applicable`.
    int device = -1;

    uint64_t resident_bytes = 0;   // RAM this placement holds for the whole run
    double seconds_per_token = 0;  // predicted contribution to one decoded token
    double seconds_worst_case = 0; // the same group with nothing resident and every byte refaulted
    std::string reason;
};

// What the allocator is allowed to spend and what it must leave alone.
struct AllocationInputs {
    uint64_t budget_bytes = 0; // RAM this plan may commit, after margins and reservations
    bool can_pin = false;      // a reclaim-exempt store exists; whether the dense set FITS in it is the ledger's

    // ── the ledger's other lines, and its second ceiling ─────────────────────────────────────
    // `budget_bytes` bounds what is HELD. These bound what is held reclaim-exempt, which is a
    // smaller number on every machine that has one and a separate one: the dense set, the expert
    // cache and a device's own buffers can each fit under it and still not fit together, and the
    // part that is refused is ordinary memory on a machine that will then reclaim it.
    uint64_t lockable_bytes = 0;      // everything held reclaim-exempt, after the margin
    uint64_t lock_in_place_bytes = 0; // the part that may be locked where it already is (the cache)
    bool lockable_bytes_known = true;
    bool lock_in_place_bytes_known = true;

    // What the holdable ceiling was before its margin, and the margin's terms. The margin is room
    // left for memory the machine can take back, and locked memory is not that: so where the cache
    // is locked, the margin is recomputed on the part of the budget that stays ordinary, and the
    // lockable ceiling - which has a margin of its own - is what bounds the rest. 0 leaves
    // `budget_bytes` as the only holdable ceiling.
    uint64_t holdable_bytes = 0;
    double margin_frac = 0.0;
    uint64_t margin_min_bytes = 0;

    // Held whatever this concludes: the context and the compute buffers. They are not weights and
    // nothing here chooses them, but they come out of the same memory, so they are a line rather
    // than something the budget is assumed to have left room for.
    uint64_t fixed_bytes = 0;
    // The multimodal projector, where the run has one and it lands in this pool: dense weights held
    // whole for the run to read images, never streamed, and not the planner's to size.
    uint64_t projector_bytes = 0;
    // Held the same way, and filled as a chat goes rather than at load: the context checkpoints the
    // run may keep. A line of its own, so a reader finds what a chat run costs over a one-shot one.
    uint64_t checkpoint_bytes = 0;

    // What a device the run is ARMED to compute on costs this pool, and whether it is charged to
    // the lockable ceiling too. Zero where no device is armed or its memory is its own.
    uint64_t device_bytes = 0;
    bool device_locked = false;

    // The part of an armed device that takes TURNS with the expert cache instead of standing beside
    // it: slots held only while a prefill runs there, when the cache is idle, and given back for
    // decode, when the slots are. It is not added to what is held; it has to fit in what the cache
    // can yield, and where the cache is not locked and the device is, in the lock that is left.
    uint64_t device_turn_bytes = 0;

    // The caller chose the pinned dense store. It is charged and reported like any other line and
    // never overruled: a pin is the caller's authority, including over a ledger that does not close.
    bool pin_forced = false;

    // The expert cache asks to be locked in place. Where there is room for at least one token
    // cycle of it, the cache is sized to what can be locked rather than to what can be held.
    bool cache_lock = false;

    // The expert cache is offered to the kernel instead (MoeStreamConfig::cache_discard, on a
    // platform with the mechanism): an entry no layer used in the last pass is the kernel's to drop
    // under pressure. The margin is room for a reclaim that writes, compressing or swapping, and the
    // kernel takes an offered page back by dropping it, which writes nothing. So, like a locked
    // cache, an offered one stands outside the margin, all but one token cycle of it: the experts
    // the last pass used are not offered, and are ordinary memory.
    bool cache_offered = false;

    // The engine's own fixed cache guard, REPORTED and never enforced here. It is a generic
    // constant that predates the per-model floor and is the weaker of the two: a budget above this
    // model's own token cycle returns hits however small it looks next to a number chosen once for
    // every model at once, and the caller clears the guard with `force-cache` when that is the case.
    // Enforcing it here as a floor declines models that stream perfectly well — measured on a phone
    // where a 35B-A3B has a 581 MiB token cycle and 1077 MiB free, and the guard alone turned a
    // working configuration into a refusal.
    uint64_t engine_cache_min = 0;

    // Bytes the dense policy will hold whatever this ranking concludes. The engine's dense mode is
    // one setting for every non-expert tensor, so per-group dense residency is not something it can
    // be told; the mechanism is chosen upstream from what a reclaim COSTS here, and this is the
    // consequence. The ranking below still prices each group — that analysis is what the plan
    // reports and what a future per-group lane would act on — but the cache is sized against what
    // is actually left, or the plan would promise memory the loader is about to take.
    uint64_t reserved_bytes = 0;

    // The expert bytes the streamer will actually serve: the cache is never worth more than the set
    // it caches, and on a machine where the first stage placed some layers on a device that set is
    // smaller than the model's.
    uint64_t cacheable_bytes = 0;

    // ── the device crossing, as a candidate (docs/measured-planning.md, Step 3) ─────────────────
    // Where the capacity fitter put the dense part of some layers on a device and left their experts
    // on the host, every such layer crosses the host/device boundary twice a token. Bringing those
    // layers home - whole, onto the host - removes the crossings and costs the host what their dense
    // part and context take. `layers_home_bytes` is that RAM, `layers_home_saved` the seconds per
    // token the measured crossings cost more than the device saves on those layers' dense matmuls;
    // both 0 when there is nothing to price, or the host does not win. It is ranked against the
    // cache and the resident groups like any other use of the budget.
    uint64_t layers_home_bytes = 0;
    double layers_home_saved = 0.0;
    uint32_t layers_home_count = 0; // the layers it brings home, for the rationale

    // ── what an earlier load measured of this decode (MeasuredKnobs::decode_*) ─────────────────
    // `compute_scale` multiplies every group's host compute term: the ratio between the compute a load
    // of the same configuration measured and what this cost model priced it at, so the groups move
    // together and stay priced relative to each other. `hit_scale` multiplies the cache's credited
    // hit rate at every size (the curve, or the linear credit), the ratio between the hits measured
    // at that load's budget and what was credited there. 1 is the plan with nothing measured.
    double compute_scale = 1.0;
    double hit_scale = 1.0;
};

// One line of the memory ledger: something the run holds for its whole length, and whether it is
// held reclaim-exempt. The rows are what a reader checks a run against - the two ceilings are only
// meaningful next to what was charged to them.
//
// A row is told apart by its kind, never by its name: the name is what a reader sees (and what the
// server's plan view and --explain print), so it may be reworded without a stage that drops or
// replaces a row silently missing it.
enum class LedgerKind {
    ContextCompute, // the context and the compute buffers
    Projector,      // a vision projector
    Checkpoints,    // the context checkpoints
    Device,         // an armed device's host-side share
    Dense,          // the dense weights the dense policy keeps
    ExpertCache,    // the expert cache
    DeviceSlots,    // the prefill device's slots, held in turn with the cache
};

// The row's display name, the one spelling of each.
const char * ledger_kind_name(LedgerKind k);

struct LedgerRow {
    LedgerKind kind = LedgerKind::ContextCompute;
    std::string name; // for display only; see LedgerKind
    uint64_t bytes = 0;
    bool locked = false;
    // Held in turn with the expert cache rather than with it: not part of `held`, which is the
    // account while decoding. During a prefill the cache has yielded this much.
    bool in_turn = false;
};

struct Ledger {
    std::vector<LedgerRow> rows;
    uint64_t holdable_cap = 0; // what the process may hold, after the margin
    uint64_t lockable_cap = 0; // what it may hold reclaim-exempt, after its own margin
    bool lockable_cap_known = true;
    uint64_t held = 0;   // every row
    uint64_t locked = 0; // the rows held reclaim-exempt
};

// One thing the cost model has to say that is not a setting: a fact it priced by, or a reason it
// declined. The topic is the note's name, so a front-end can title it and a reader can tell two notes
// apart without parsing prose; the brief is what a person reads, the detail what --explain prints.
struct AllocationNote {
    std::string topic;  // stable kebab-case key; notes of the same family share one
    std::string value;  // the number the note turns on, rendered; "-" when there is none
    std::string brief;  // one plain sentence
    std::string detail; // the full account
};

// A size as a person reads it: GB with one decimal from 1 GiB up, whole MiB below. The planner's
// one-line briefs use it so every size on a screen is spelled the same way.
std::string size_brief(uint64_t bytes);

struct Allocation {
    GroupPlacement groups[(int) WeightGroup::count];

    Ledger ledger;

    // Whether the dense set goes in the reclaim-exempt store. It is all or nothing because the
    // engine's dense mode is one setting, and a set that is pinned part way is the case this
    // exists to prevent. When a store exists and the set did not fit, `pinned_shortfall_bytes`
    // says by how much.
    bool dense_pinned = false;
    uint64_t pinned_shortfall_bytes = 0;

    // Whether what takes turns with the cache (AllocationInputs::device_turn_bytes) fits when its
    // turn comes: the cache can yield that much, or what is left over still fits both ceilings.
    // True when nothing takes turns.
    bool turn_fits = true;

    // The part of `cache_bytes` the platform can be asked to keep in place; 0 when the cache is
    // ordinary memory, because it was not asked for or because not even one token cycle fits.
    uint64_t cache_locked_bytes = 0;
    // An unknown ceiling permits a bounded runtime attempt, not a predicted lock grant. This
    // never widens the holdable budget; the run fits the cache to what the kernel actually grants.
    bool cache_lock_unverified = false;

    uint64_t cache_bytes = 0; // expert cache budget; 0 means the expert lane is not used

    // Whether the ranking funded AllocationInputs::layers_home_*: the plan then brings those layers
    // home and is made again with them there.
    bool layers_home = false;
    // The share of a token's expert reads the cost model credits the cache with serving, 0..1: what
    // `seconds_per_token` was priced with, so a front-end shows the figure the plan stands on rather
    // than a cache-to-experts ratio of its own. 0 without a cache.
    double cache_hit_credited = 0;

    // What was available for the cache BEFORE the floor was applied. It is the only way to say how
    // far short a refused cache fell: `cache_bytes` is 0 once the floor rejects it, and a shortfall
    // computed from a zero is the whole floor rather than the gap, which turns an actionable "184
    // MiB short" into a useless "901 MiB short".
    uint64_t cache_available_bytes = 0;
    uint64_t resident_bytes = 0; // what the whole allocation holds
    uint64_t budget_bytes = 0;

    double seconds_per_token = 0; // what the allocation predicts

    // What the same token would cost with nothing resident and every byte refaulted a page at a
    // time. An UPPER BOUND on what the allocation avoids, not a prediction of what mmap would
    // really do — the page cache retains some of the model and readahead amortises some of the rest,
    // both by an amount this cannot know. Reported as a bound, never divided into the prediction to
    // manufacture a speedup figure.
    double seconds_worst_case = 0;

    // True when every non-host device choice above can actually be executed by this build. False
    // means the allocation REPORTS a placement it cannot apply, which is said out loud rather than
    // silently dropped: a recommendation the engine cannot honour is still the most useful thing
    // the probe found.
    bool device_applicable = true;

    std::vector<AllocationNote> notes;

    const GroupPlacement & at(WeightGroup g) const { return groups[(int) g]; }
    GroupPlacement & at(WeightGroup g) { return groups[(int) g]; }
};

struct PlannerPolicy; // bmoe/planner.h

// Re-price an allocation for a run that will not stream. The engine's residency policies and its
// expert lane exist only under streaming: declined, the whole file is left mapped and the kernel
// decides what stays. So nothing is held on the model's behalf and every group costs what it costs
// refaulted - the bound, which is then the prediction, because it is the only figure this can state
// about a mapping it does not manage. Without this a declined plan printed the cost of the plan it
// had just refused: measured on a model five times the machine's memory, 0.23 s/token predicted
// against 1.46 run, where the bound said 1.69.
void price_unstreamed(Allocation & a);

// Price every legal lane for every group, rank the candidates by seconds saved per byte of RAM, and
// spend the budget down that list. Never fails: a machine with no measurements yields an allocation
// that leaves everything mapped and says, per group, which measurement it was owed.
Allocation allocate(const HardwareProfile & hw,
                    const ModelProfile & model,
                    const AllocationInputs & in,
                    const PlannerPolicy & pol);

// The compute this cost model prices one decoded token at on the host, every group of `model` at the
// host's measured rates and no compute_scale: what a measured decode of the same configuration is
// compared with to calibrate it. 0 where a rate is unmeasured.
double host_compute_seconds(const HardwareProfile & hw, const ModelProfile & model);

} // namespace bmoe
