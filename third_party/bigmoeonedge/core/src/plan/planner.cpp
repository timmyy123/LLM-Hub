// The planning rules. See bmoe/planner.h for the contract and the two invariants.
//
// Read this as a table rather than as code: each block below is one knob, the fact that decides it,
// and what happens when that fact is missing. Nothing here reads a platform, an architecture or a
// tensor name — the inputs carry no such thing — so the only way to add hardware is to teach an
// adapter to measure one more fact.
//
// Each block is a stage: a function over the Planning state below, and plan_run() at the bottom is
// the order they run in, readable on one screen. A reason that tells a person what to type spells
// the flag through the parameter table (flag_spelling), so a renamed flag cannot leave the rationale
// advising one that no longer parses.

#include "bmoe/planner.h"
#include "bmoe/backend_choice.h"

#include "bmoe/params.h"
#include "bmoe/routing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace bmoe {

namespace {

uint64_t mib(uint64_t bytes) {
    return bytes >> 20;
}

std::string u64s(uint64_t v) {
    return std::to_string((unsigned long long) v);
}

// What this process may hold. The measured figure wins where it exists, because it answers the
// question the rules are actually asking - what can be KEPT - while the reported one answers what
// could be allocated at this instant. They differ by more than rounding wherever a reclaim
// compresses: the reported number is a floor there, and sizing from it leaves memory unused.
uint64_t budget_bytes(const HardwareProfile & hw) {
    return hw.holdable_bytes ? hw.holdable_bytes : hw.residency_budget;
}

// How much of the residency budget to leave for everything that is not us. The fraction is chosen
// by what losing memory costs here, which is the honest axis: it is cheap where a reclaim only
// compresses, and fatal where exceeding the share ends the process.
float margin_frac(const HardwareProfile & hw, const PlannerPolicy & pol) {
    float frac = pol.margin_when_unknown;
    switch (hw.anon_overflow) {
    case Overflow::Compress:
        frac = pol.margin_when_compressed;
        break;
    case Overflow::Swap:
        frac = pol.margin_when_swapped;
        break;
    case Overflow::Kill:
        frac = pol.margin_when_killed;
        break;
    case Overflow::None:
    case Overflow::Refault:
    case Overflow::Unknown:
        break;
    }
    return frac;
}

uint64_t margin_bytes(const HardwareProfile & hw, const PlannerPolicy & pol) {
    const uint64_t by_frac = (uint64_t) ((double) budget_bytes(hw) * margin_frac(hw, pol));
    return std::max(by_frac, pol.margin_min_bytes);
}

// Whether taking resident pages back from us is CHEAP for this kernel, which is the case where
// fitting stops implying being left alone. It is the narrow reading on purpose. Where a reclaim
// only compresses or only drops a clean page, the kernel does it routinely and to anyone, so a
// model with nothing to spare is reclaimed from underneath continuously - that is the case we
// measured. Where a reclaim has to write to a disk the kernel is far more reluctant, where the
// allocation is exempt nothing can take it, and where the limit is a hard cap nothing is taken at
// all. And `Unknown` keeps the default like every other missing fact: the evidence here is specific
// to cheap reclaim, so an unprofiled machine is not entitled to its conclusion. The headroom probe
// is what generalises this from a class of kernel to a measurement.
bool reclaim_is_cheap(Overflow o) {
    switch (o) {
    case Overflow::Compress:
    case Overflow::Refault:
        return true;
    case Overflow::None:
    case Overflow::Swap:
    case Overflow::Kill:
    case Overflow::Unknown:
        break;
    }
    return false;
}

// Where the budget came from, in the words the rationale prints. The distinction matters to a
// reader deciding how much to trust a plan: an estimate and a measurement are not the same claim.
const char * headroom_name(Headroom h) {
    switch (h) {
    case Headroom::Compressible:
        return "estimated from what this machine's compressor is achieving";
    case Headroom::Measured:
        return "measured by holding memory until the machine took some back";
    case Headroom::Unknown:
        break;
    }
    return "as reported available, unmeasured";
}

const char * overflow_name(Overflow o) {
    switch (o) {
    case Overflow::None:
        return "nothing can reclaim it";
    case Overflow::Compress:
        return "reclaimed into compressed RAM";
    case Overflow::Swap:
        return "swapped to disk";
    case Overflow::Refault:
        return "dropped and re-read from the file";
    case Overflow::Kill:
        return "the process is killed";
    case Overflow::Unknown:
        break;
    }
    return "unprobed";
}

} // namespace

const char * source_name(Source s) {
    switch (s) {
    case Source::Measured:
        return "measured";
    case Source::Derived:
        return "derived";
    case Source::Policy:
        return "policy";
    case Source::Operator:
        return "operator";
    case Source::Unprobed:
        return "unprobed";
    }
    return "?";
}

const char * regime_name(Regime r) {
    switch (r) {
    case Regime::Unknown:
        return "unknown";
    case Regime::NotMoe:
        return "not-moe";
    case Regime::Fits:
        return "fits";
    case Regime::ExpertsStream:
        return "experts-stream";
    case Regime::DenseOversized:
        return "dense-oversized";
    }
    return "?";
}

LayerSlots layer_slots(int32_t n_gpu_layers, uint32_t n_layer) {
    LayerSlots s;
    s.n_layer = n_layer;
    const uint32_t slots = n_layer + 1;
    s.on_device = n_gpu_layers < 0 ? slots : std::min<uint32_t>((uint32_t) n_gpu_layers, slots);
    return s;
}

HostDense host_dense_of(const ModelProfile & m, const LayerSlots & slots) {
    HostDense h;
    h.tables = m.group(WeightGroup::Embedding).bytes;
    const uint64_t output = m.group(WeightGroup::Output).bytes;
    const uint64_t host_blocks = slots.host_blocks();
    uint64_t blocks = 0, blocks_on_host = 0;
    if (m.n_layer && m.layer_dense_bytes.size() == m.n_layer) {
        for (uint32_t il = 0; il < m.n_layer; ++il) {
            blocks += m.layer_dense_bytes[il];
            if (il < host_blocks) blocks_on_host += m.layer_dense_bytes[il];
        }
    } else {
        // Without each block's bytes, the blocks are what the parts outside them leave, by count.
        const uint64_t outside = h.tables + output;
        blocks = m.dense_bytes > outside ? m.dense_bytes - outside : 0;
        blocks_on_host = m.n_layer ? (uint64_t) ((double) blocks * (double) host_blocks / (double) m.n_layer) : blocks;
    }
    // What is neither a block, a table nor the output head (the final norm and the like) is counted on
    // the host: the side that can only over-reserve, by bytes too few to move a plan.
    const uint64_t named = blocks + h.tables + output;
    const uint64_t rest = m.dense_bytes > named ? m.dense_bytes - named : 0;
    h.bytes = blocks_on_host + h.tables + (slots.output_on_device() ? 0 : output) + rest;
    return h;
}

Placement Placement::at_width(uint32_t width) const {
    Placement q = *this;
    if (compute_width == 0 || width == 0 || width == compute_width) return q;
    // Both directions, clamped at zero: a slope is a measurement, and a narrower graph cannot be
    // credited with more than the buffer it is taken from.
    auto move = [&](uint64_t bytes, uint64_t per_token) {
        if (width > compute_width) return bytes + (uint64_t) (width - compute_width) * per_token;
        const uint64_t less = (uint64_t) (compute_width - width) * per_token;
        return bytes > less ? bytes - less : 0;
    };
    const uint64_t host = move(raw_host_compute_bytes, host_compute_per_token);
    const uint64_t dev = move(device_compute_bytes, device_compute_per_token);
    // A windowed cache holds one ubatch of cells beyond its window, so the context moves with the
    // width too: by the cells the two widths hold apart (none without a window).
    const uint64_t had = window_cells(compute_width), has = window_cells(width);
    auto move_ctx = [&](uint64_t bytes, uint64_t per_cell) {
        if (has >= had) return bytes + (has - had) * per_cell;
        const uint64_t less = (had - has) * per_cell;
        return bytes > less ? bytes - less : 0;
    };
    const uint64_t host_ctx = move_ctx(raw_host_context_bytes, host_window_cell_bytes);
    const uint64_t dev_ctx = move_ctx(device_context_bytes, device_window_cell_bytes);
    q.host_resident_bytes = host_resident_bytes - std::min(host_resident_bytes, raw_host_compute_bytes) + host;
    q.host_resident_bytes = q.host_resident_bytes - std::min(q.host_resident_bytes, raw_host_context_bytes) + host_ctx;
    q.device_bytes = device_bytes - std::min(device_bytes, device_compute_bytes) + dev;
    q.device_bytes = q.device_bytes - std::min(q.device_bytes, device_context_bytes) + dev_ctx;
    q.raw_host_compute_bytes = host;
    q.device_compute_bytes = dev;
    q.raw_host_context_bytes = host_ctx;
    q.device_context_bytes = dev_ctx;
    q.compute_width = width;
    return q;
}

uint64_t Placement::device_compute_of(const std::string & device) const {
    // Each device's part is taken at the projected width and kept in proportion as the sum moves
    // along its slope: the slope was measured for the sum, not per device.
    uint64_t part = 0, projected = 0;
    for (size_t i = 0; i < projected_devices.size() && i < projected_compute_bytes.size(); ++i) {
        projected += projected_compute_bytes[i];
        if (projected_devices[i] == device) part += projected_compute_bytes[i];
    }
    if (part == 0 || projected == 0) return 0;
    return (uint64_t) ((double) device_compute_bytes * (double) part / (double) projected);
}

bool PlanRequest::is_pinned(const char * knob) const {
    for (const std::string & p : pinned)
        if (p == knob) return true;
    return false;
}

Plan plan_run(const RunConfig & base, const HardwareProfile & hw, const ModelProfile & model, const PlanRequest & req) {
    return plan_run(base, hw, model, Placement{}, req, PlannerPolicy::defaults());
}

Plan plan_run(const RunConfig & base,
              const HardwareProfile & hw,
              const ModelProfile & model,
              const Placement & placement,
              const PlanRequest & req) {
    return plan_run(base, hw, model, placement, req, PlannerPolicy::defaults());
}

namespace {

// The shortest context the rationale will suggest: llama.cpp's own physical batch. A context
// narrower than one graph is not a run anyone would shorten theirs to, whatever cache it bought.
constexpr uint64_t k_min_suggested_ctx = 512;

// Whether every device here reads this host's own memory: none has memory of its own. False with
// no devices at all, where the question does not arise.
bool devices_share_host_memory(const HardwareProfile & hw) {
    bool shared = !hw.devices.empty();
    for (const ComputeDevice & d : hw.devices)
        if (d.has_own_memory()) shared = false;
    return shared;
}

// One plan_run() call, carried through its stages: the inputs, the plan being written, and the
// figures a stage derives for a later one to read. Each stage below is one block of the table this
// file is meant to be read as - a knob or two, the fact that decides it, and what happens when that
// fact is missing - and runs in the order plan_run() lists them, because a later stage reads what
// an earlier one settled: the width moves the compute buffers, the regime gates everything after
// it, the ledger sizes the cache the I/O policy then reads.
struct Planning {
    Planning(const RunConfig & base,
             const HardwareProfile & hw_,
             const ModelProfile & model_,
             const Placement & projected,
             const PlanRequest & req_,
             const PlannerPolicy & pol_,
             Plan & plan)
        : hw(hw_), model(model_), req(req_), pol(pol_), p(plan), placement(projected), projected(projected) {
        p.config = base;
        p.placement = placement;
    }

    const HardwareProfile & hw;
    const ModelProfile & model;
    const PlanRequest & req;
    const PlannerPolicy & pol;
    Plan & p;

    // The first stage's answer at the graph width this stage chose (choose_graph_width). The
    // width is this stage's to choose, and the compute figures move with it.
    Placement placement;
    // And as the first stage gave it, for a plan made again from an alternative to it.
    const Placement projected;

    // The device's dense-only layers brought home (price_layers_home): the alternative placement,
    // what it costs the host and what it saves a token, and the account of it. `considered` is
    // false in the plan made again from that alternative, so it is priced once. `measured` is the
    // account's first half, what the split measured; the decision's reason is that and the outcome
    // (home_reason), so a later stage that settles the outcome rewrites only its own half.
    struct LayersHome {
        bool considered = true;
        bool priced = false;
        uint32_t layers = 0;
        uint64_t bytes = 0;
        double saved = 0.0;
        Placement whole;
        std::string measured;
        Decision decision;
    } home;

    // The rate a device has to beat before it is said to read this model's weights faster than the
    // host: the policy's margin over the host's own figure, computed once so the lines quoting it
    // cannot drift apart.
    double host_bandwidth_win = 0.0;

    // What the first stage left for the second (share_of_host).
    uint32_t n_moe_layer = 0;
    uint32_t host_layers = 0;
    uint64_t host_expert_bytes = 0;

    // What this process may hold, the margin left for everything that is not us, and the rest.
    uint64_t holdable = 0;
    uint64_t margin = 0;
    uint64_t usable = 0;

    // The host set and the regime it puts the run in (classify_regime).
    bool devices_share_host_memory = false;
    uint64_t device_on_host = 0;
    uint64_t host_fixed = 0;
    // The projector, where the run has one: on a device where the run uses one (the session's own
    // rule, below), on the host otherwise; `projector_host` is the part this pool holds.
    bool projector_on_device = false;
    uint64_t projector_host = 0;
    uint64_t checkpoint_bytes = 0; // the context checkpoints a chat run will keep (checkpoint_reserve)
    uint64_t host_all = 0;         // the host's share of the model, every byte of it
    uint64_t host_dense = 0;       // of that, all but the experts
    // Of both, the embedding tables the run may leave in the mapping (mappable_host_tables). What
    // the regime weighs is what is left: the bytes the run has to hold.
    uint64_t host_tables = 0;
    uint64_t held_all() const { return host_all - std::min(host_all, host_tables); }
    uint64_t held_dense() const { return host_dense - std::min(host_dense, host_tables); }
    uint64_t air = 0;
    bool air_short = false;
    // Set on the second pass of a plan that streams (plan_from): the model is planned as held whole,
    // to see whether a device that reads this memory may be given it, and this is the room it may
    // take there with its compute buffer: what the plan that streams it may hold of this pool, its
    // own ledger's ceiling. 0 on every other pass, where the regime and the margin decide.
    uint64_t whole_room = 0;

    // The model as the allocator sees it: only the host's share of the experts (host_model).
    ModelProfile host_model;

    // What an earlier load measured of this decode, as the ledger's coefficients
    // (calibrate_decode_cost; AllocationInputs::compute_scale, hit_scale). 1: nothing measured applies.
    double compute_scale = 1.0;
    double hit_scale = 1.0;

    // Every decision says itself twice: a brief, one plain sentence a person reads on a screen, and
    // the reason, the full account --explain prints. The brief is required so that no rule can add a
    // decision that only an engineer can read.
    void note(const char * knob, std::string value, Source src, std::string brief, std::string reason) {
        Decision d;
        d.knob = knob;
        d.value = std::move(value);
        d.source = src;
        d.reason = std::move(reason);
        d.brief = std::move(brief);
        p.decisions.push_back(std::move(d));
    }

    // A knob the caller set is a knob we do not own. Records it and reports that it is off-limits.
    bool pinned(const char * knob, std::string value) {
        if (!req.is_pinned(knob)) return false;
        const std::string brief = "Kept at " + value + ", as you set it.";
        note(knob, std::move(value), Source::Operator, brief, "set by the caller; the planner leaves it alone");
        return true;
    }

    void decline(std::string why) {
        p.streaming_declined = true;
        p.decline_reason = std::move(why);
        p.config.moe.enabled = false;
        // What acts only on the streamer's own reads has nothing to act on now. A caller who set one
        // asked for it on a streamed run; this plan does not stream, so it is turned off and said so,
        // where leaving it set made the run refuse to start (`moe.overlap requires moe.enabled`).
        MoeStreamConfig & m = p.config.moe;
        const auto off = [&](const char * knob, bool set) {
            if (!set) return;
            note(knob, "off", Source::Derived, "Off: nothing is streamed, so it has nothing to act on.",
                 "it acts on the expert streamer's own reads, and this plan does not stream: " + p.decline_reason);
        };
        off("overlap", m.overlap);
        off("predict-log", m.predict_log);
        off("predict-prefetch", m.predict_prefetch);
        off("route-ahead", m.route_ahead > 0);
        off("row-stream", m.row_stream);
        m.overlap = m.predict_log = m.predict_prefetch = m.row_stream = false;
        m.route_ahead = 0;
        p.config.trial.overlap = false;
    }
};

// ── what the first stage left for the second ─────────────────────────────────────
// Without a placement every layer's experts are on the host and nothing of the model sits on a
// device; with one, only the host share of the experts is the streamer's to serve, and the dense
// set the fitter kept on the host is what it costs the residency budget.
// Expert bytes prorate over the layers that CARRY experts, not over every block in the file. On an
// architecture with leading dense blocks the two counts differ, and dividing the expert set by the
// larger of them understates what the host still has to stream.
void share_of_host(Planning & s) {
    const Placement & placement = s.placement;
    s.host_bandwidth_win = s.hw.host_bandwidth_gibs * (1.0 + (double) s.pol.min_backend_win);
    s.n_moe_layer = s.model.n_moe_layer;
    // Count the host list against the blocks that carry experts: a placement is expressed in every
    // block of the file, and on a hybrid architecture most blocks carry none.
    uint32_t host_moe = 0;
    for (uint32_t il : placement.host_expert_layers)
        if (s.model.moe_layers.empty() ||
            std::find(s.model.moe_layers.begin(), s.model.moe_layers.end(), il) != s.model.moe_layers.end())
            ++host_moe;
    s.host_layers = placement.fitted ? std::min(host_moe, s.n_moe_layer) : s.n_moe_layer;
    s.host_expert_bytes = (placement.fitted && s.n_moe_layer)
                              ? (uint64_t) ((double) s.model.expert_bytes * s.host_layers / s.n_moe_layer)
                              : s.model.expert_bytes;
}

// ── device memory as a tier of the expert cache (docs/measured-planning.md, Step 4) ──
// The same rule as flash and RAM, one level up: the experts used most live in the memory of the
// processor that computes them fastest. The capacity fitter fills a device by capacity, not by use:
// once the dense part of every layer is there, it gives the memory left to whole layers' experts,
// front to back, and the layers it could not fit keep their experts on the host, where every token
// waits for them. Where a device with memory of its own computes the model's experts faster than the
// host, those same bytes hold a cache of the experts used most, from every layer, computed on the
// device; a miss is uploaded from where the host holds it, as a miss in RAM is read from flash. The
// cache fills from use, so it needs no record of the routing to be right; a record says how much of
// a token it serves.
//
// Measured once both ways on a 24 GB card, Qwen3.8-Flash-Next 125B: 63-81 tok/s from the cache
// against 28-47 for the fitter's layers, where a price of the two (the cache's misses at the
// measured upload rate against the host's compute) had called them a tie and kept the layers.

// The share of a token's routed bytes a cache of `bytes` serves: from this model's own routing where
// a record of it was replayed (Step 1), else from the prior in token cycles (prior_hit_curve), else
// the fraction of the expert set it holds.
double expert_share_at(const ModelProfile & model, uint64_t bytes) {
    if (model.hit_curve.credited()) return model.hit_curve.hit_share(bytes);
    if (model.expert_bytes == 0) return 0.0;
    return std::min(1.0, (double) bytes / (double) model.expert_bytes);
}

// Where expert_share_at's figure came from, as a rationale says it.
const char * expert_share_source(const ModelProfile & model) {
    return model.hit_curve.measured ? "from this model's own routing"
           : model.hit_curve.prior  ? "the prior in token cycles every model without a record is credited"
                                    : "the fraction of the expert set it holds";
}

// Whether every device the fitter put layers on can hold a cache of experts and compute them there,
// and the slowest of them at the experts, which bounds the rest. Nullptr, with the reason, where a
// device cannot or its rate is unmeasured.
// The layers' split among the cards and the device cache's budget, decided together. llama.cpp hands
// each card a share of one `moe_cache_size` pro rata to the layers the plan's split gives it, whatever
// the card has free (its MoE cache splits the budget by the tensor split). So a budget summed over the
// cards and a split left as the fitter made it disagree: the fitter fills a card with whole layers'
// experts, and once the cache sends them to the host, the room they leave is on that card while the
// budget follows the layer counts. Two 16 GB cards: the fitter's 6 whole layers on the first and 38
// dense-only ones on the second, the first card's share 6/44 of the budget, 10 GB of it left empty.
//
// Under the cache a card holds its layers' dense part and context and its share of the budget. Its
// room for those is what it had free beyond the fitter's projection, plus the experts the fitter put
// on it, plus the dense part and context of its own layers. Each card's share per layer is then its
// room over its layers less one layer's dense part, and the budget is the tightest card's share times
// the layers: highest when every card has layers in proportion to its room. The fitter's split is
// kept unless the proportional one gives a larger budget. A layer's dense part and context are taken
// as the model's average; the experts the fitter put on the devices are counted exactly in total and
// shared among the cards by how many of those layers each held. One card is its own split: its budget
// is what it had free plus the experts the fitter gave it.
struct TierSplit {
    std::vector<std::pair<std::string, uint32_t>> split; // the split to load with; the fitter's when kept
    uint64_t budget = 0;
    bool resplit = false; // the split differs from the fitter's
};

TierSplit device_tier_split(const Placement & pl, const ModelProfile & m, uint64_t on_devices) {
    TierSplit out;
    out.split = pl.device_split;
    uint64_t spare_sum = 0;
    for (const auto & sp : pl.device_spare)
        spare_sum += sp.second;
    if (pl.device_split.size() <= 1 || m.n_layer == 0) {
        out.budget = on_devices + spare_sum;
        return out;
    }
    const LayerSlots ls = layer_slots(pl.n_gpu_layers, m.n_layer);
    const uint32_t slots = m.n_layer + 1;
    const auto is_moe = [&](uint32_t il) {
        return m.moe_layers.empty() || std::find(m.moe_layers.begin(), m.moe_layers.end(), il) != m.moe_layers.end();
    };
    const auto experts_home = [&](uint32_t il) {
        return std::find(pl.host_expert_layers.begin(), pl.host_expert_layers.end(), il) != pl.host_expert_layers.end();
    };
    struct Card {
        uint32_t layers = 0, whole = 0, output = 0;
        uint64_t spare = 0;
        double room = 0.0;
    };
    std::vector<Card> cards(pl.device_split.size());
    uint32_t at = ls.first_device_slot(), n_layers = 0, n_whole = 0;
    for (size_t i = 0; i < pl.device_split.size(); ++i) {
        Card & c = cards[i];
        for (const auto & sp : pl.device_spare)
            if (sp.first == pl.device_split[i].first) c.spare = sp.second;
        for (uint32_t k = 0; k < pl.device_split[i].second && at < slots; ++k, ++at) {
            if (at == m.n_layer) {
                c.output = 1;
                continue;
            }
            ++c.layers;
            if (is_moe(at) && !experts_home(at)) ++c.whole;
        }
        n_layers += c.layers;
        n_whole += c.whole;
    }
    if (n_layers == 0) {
        out.budget = on_devices + spare_sum;
        return out;
    }
    // A layer's dense part and context. The split is contiguous from the first device slot, so a
    // split names exactly which layers each card holds, and where the file's per-layer bytes are known
    // a card is charged the layers it takes on and credited the ones it gives up, byte for byte: on
    // GLM-4.5-Air layer 0 is twice the average. The context is the model's average per layer. Where
    // the per-layer bytes are unknown, a layer a card takes on is charged the file's largest dense
    // layer, since llama.cpp loads it before any cache is made and a load past the card's memory has
    // nothing to fall back on; a layer a card gives up is credited the average, since a cache past a
    // card's memory is made smaller by the session (create_contexts) rather than failing the load.
    const double kv_per_layer = (double) pl.device_context_bytes / (double) m.n_layer;
    const double dense_per_layer = (double) (m.group(WeightGroup::Attention).bytes +
                                             m.group(WeightGroup::DenseFfn).bytes + m.group(WeightGroup::Other).bytes) /
                                       (double) m.n_layer +
                                   kv_per_layer;
    const double dense_bound = std::max(dense_per_layer, (double) m.largest_layer_dense_bytes + kv_per_layer);
    const uint32_t first = ls.first_device_slot();
    const bool exact = m.layer_dense_bytes.size() == m.n_layer;
    // The dense part and context of layers [from, from + n), the output's slot not among them.
    const auto dense_of = [&](uint32_t from, uint32_t n) {
        double b = 0.0;
        for (uint32_t il = from; il < from + n && il < m.n_layer; ++il)
            b += (double) m.layer_dense_bytes[il] + kv_per_layer;
        return b;
    };
    // What each card holds for the cache today, beside the layers the fitter gave it.
    for (Card & c : cards)
        c.room = (double) c.spare + (n_whole ? (double) on_devices * c.whole / n_whole : 0.0);
    // The budget a split of the layers lets every card take its share of; 0 where a card cannot hold
    // the layers it takes on.
    const auto budget_of = [&](const std::vector<uint32_t> & n) {
        double b = -1.0;
        uint32_t was = first, now = first; // where each card's layers start, before and after
        for (size_t i = 0; i < cards.size(); ++i) {
            const uint32_t had = cards[i].layers;
            double charge = 0.0;
            if (exact)
                charge = dense_of(now, n[i]) - dense_of(was, had);
            else {
                const double moved = (double) n[i] - (double) had;
                charge = moved > 0 ? moved * dense_bound : moved * dense_per_layer;
            }
            was += had;
            now += n[i];
            if (n[i] == 0) continue;
            const double left = cards[i].room - charge;
            const double share = left <= 0.0 ? 0.0 : left * (double) n_layers / (double) n[i];
            b = b < 0.0 ? share : std::min(b, share);
        }
        return b < 0.0 ? 0.0 : b;
    };
    std::vector<uint32_t> kept(cards.size()), even(cards.size(), 0);
    // Layers in proportion to each card's whole room under the cache (what it holds for the cache and
    // its layers' dense part), the remainder to the largest fractions.
    double room_sum = 0.0;
    std::vector<double> whole_room(cards.size());
    for (size_t i = 0; i < cards.size(); ++i) {
        kept[i] = cards[i].layers;
        whole_room[i] = cards[i].room + dense_per_layer * cards[i].layers;
        room_sum += whole_room[i];
    }
    std::vector<std::pair<double, size_t>> frac;
    uint32_t given = 0;
    for (size_t i = 0; i < cards.size(); ++i) {
        const double want = room_sum > 0.0 ? (double) n_layers * whole_room[i] / room_sum : 0.0;
        even[i] = (uint32_t) std::floor(want);
        given += even[i];
        frac.push_back({want - (double) even[i], i});
    }
    std::sort(frac.begin(), frac.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    for (size_t k = 0; given < n_layers && k < frac.size(); ++k, ++given)
        ++even[frac[k].second];
    // From the better of the two, one layer at a time between any two cards while the budget grows:
    // the charges are not symmetric, so the proportional split is a start, not the answer.
    const double b_kept = budget_of(kept);
    double b_even = budget_of(even);
    if (b_kept > b_even) {
        even = kept;
        b_even = b_kept;
    }
    for (bool moved = true; moved;) {
        moved = false;
        for (size_t from = 0; from < cards.size() && !moved; ++from)
            for (size_t to = 0; to < cards.size() && !moved; ++to) {
                if (from == to || even[from] == 0) continue;
                std::vector<uint32_t> n = even;
                --n[from];
                ++n[to];
                const double b = budget_of(n);
                if (b > b_even) {
                    even = n;
                    b_even = b;
                    moved = true;
                }
            }
    }
    // Rounded down to a MiB, and never more than the cards had: free beyond the fitter's projection
    // and the experts it gave them.
    const auto cap = [&](double b) {
        return std::min<uint64_t>((uint64_t) b, on_devices + spare_sum) & ~(uint64_t) 0xFFFFF;
    };
    out.budget = cap(b_kept);
    if (b_even > b_kept && even != kept) {
        out.budget = cap(b_even);
        out.resplit = true;
        out.split.clear();
        for (size_t i = 0; i < cards.size(); ++i)
            if (even[i] + cards[i].output > 0)
                out.split.emplace_back(pl.device_split[i].first, even[i] + cards[i].output);
    }
    return out;
}

const ComputeDevice * device_tier_device(const Planning & s, std::string & why) {
    const ComputeDevice * worst = nullptr;
    for (const ComputeDevice & d : s.hw.devices) {
        if (d.is_cpu || d.usable == Tri::No) continue;
        if (!d.has_own_memory()) continue; // a shared pool has no second tier to fill
        if (d.runs_expert_op != Tri::Yes) {
            why = d.name + (d.runs_expert_op == Tri::No
                                ? " does not execute this model's expert matmul on its own layout"
                                : " was not asked whether it executes this model's expert matmul");
            return nullptr;
        }
        if (d.identity_ok == Tri::No) {
            why = d.name + " computes this model's expert matmul too far from the exact result";
            return nullptr;
        }
        if (expert_rates(s.hw, d).device <= 0.0) {
            why = "what " + d.name + " computes the experts at is unmeasured";
            return nullptr;
        }
        if (!worst || expert_rates(s.hw, d).device < expert_rates(s.hw, *worst).device) worst = &d;
    }
    if (!worst) why = "no device here has memory of its own";
    return worst;
}

// The fastest the capacity fitter's placement could decode on this machine, in tokens a second: the
// experts it left on the host at the host's expert rate, the rest of the experts and every other byte
// a token reads at the fastest device's rates, one after the other as a token computes them, and
// nothing for the crossings, the misses or any overhead. Called where every dense part is on the
// devices (choose_device_tier's precondition). 0 where a rate or the model's bytes per token is
// unknown: a ceiling that cannot be computed decides nothing. `host_expert_bytes` is what the placement
// leaves on the host: the fitter's own, or what a narrower graph would leave.
double fitter_decode_ceiling(const Planning & s, uint64_t host_expert_bytes) {
    const ModelProfile & m = s.model;
    const double experts_tok = (double) m.group(WeightGroup::Experts).bytes_per_token;
    if (experts_tok <= 0.0 || m.expert_bytes == 0) return 0.0;
    double rest_tok = 0.0;
    for (size_t g = 0; g < (size_t) WeightGroup::count; ++g)
        if ((WeightGroup) g != WeightGroup::Experts) rest_tok += (double) m.group((WeightGroup) g).bytes_per_token;
    double device_experts = 0.0, device_dense = 0.0;
    for (const ComputeDevice & d : s.hw.devices) {
        if (d.is_cpu || d.usable == Tri::No || !d.has_own_memory()) continue;
        device_experts = std::max(device_experts, expert_rates(s.hw, d).device);
        device_dense = std::max(device_dense, dense_rates(s.hw, d).device);
    }
    const double host_share = std::min(1.0, (double) host_expert_bytes / (double) m.expert_bytes);
    const double host_experts = s.hw.host_expert_rate();
    if (device_experts <= 0.0 || device_dense <= 0.0 || (host_share > 0.0 && host_experts <= 0.0)) return 0.0;
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    const double seconds = (experts_tok * host_share / host_experts +
                            experts_tok * (1.0 - host_share) / device_experts + rest_tok / device_dense) /
                           gib;
    return seconds > 0.0 ? 1.0 / seconds : 0.0;
}
double fitter_decode_ceiling(const Planning & s) {
    return fitter_decode_ceiling(s, s.host_expert_bytes);
}

// What a decoded token costs with a cache of `cache_bytes` of experts on `dev`, in seconds, from the
// rates this machine measured: every other byte a token reads at the device's dense rate, the routed
// experts the cache serves at its expert rate, and each miss uploaded from the host and then computed
// there. The hits are the share of a token's routed bytes a cache that size serves, from this model's
// own routing where a record was replayed and the prior otherwise (expert_share_at). 0 where a rate
// or the model's bytes per token is unknown: a price that cannot be computed decides nothing.
double device_cache_decode_seconds(const Planning & s, const ComputeDevice & dev, uint64_t cache_bytes) {
    const ModelProfile & m = s.model;
    const RatePair experts = expert_rates(s.hw, dev);
    const double dense = dense_rates(s.hw, dev).device;
    const double upload = dev.upload_gibs > 0.0 ? dev.upload_gibs : dev.upload_host_buffer_gibs;
    const double routed = (double) m.group(WeightGroup::Experts).bytes_per_token;
    if (experts.device <= 0.0 || dense <= 0.0 || upload <= 0.0 || routed <= 0.0) return 0.0;
    double rest = 0.0;
    for (size_t g = 0; g < (size_t) WeightGroup::count; ++g)
        if ((WeightGroup) g != WeightGroup::Experts) rest += (double) m.group((WeightGroup) g).bytes_per_token;
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    const double hit = std::clamp(expert_share_at(m, cache_bytes), 0.0, 1.0);
    const double miss = (1.0 - hit) * routed / gib;
    return rest / gib / dense + hit * routed / gib / experts.device + miss * (1.0 / upload + 1.0 / experts.device);
}

// What a decoded token waits on storage for the experts the host holds, in seconds. The two prices
// above are compute: where the experts the host is left with do not fit its memory they are streamed,
// and a token also reads from flash the routed bytes neither cache serves, at the rate this storage
// measured at the model's own slice size. `host_expert_bytes` is what the placement leaves on the
// host and `device_hit` the share of a token's routed bytes a device cache serves (0 without one);
// the host's RAM cache is what its memory holds of them, and the two caches keep the same hottest
// experts, so what misses the larger of the two is what flash is read for. Left out, a placement that
// streams was priced as if it did not: measured on an 8 GB card with a model twice the host's memory,
// the fitter's placement priced at 11 tok/s decoded at 3.5, and a 284B model on two 16 GB cards at 2.3
// to 6.5 against 16. 0 where everything fits or the storage rate is unmeasured.
double host_expert_flash_seconds(const Planning & s, uint64_t host_expert_bytes, double device_hit) {
    const ModelProfile & m = s.model;
    const double routed = (double) m.group(WeightGroup::Experts).bytes_per_token;
    if (routed <= 0.0 || m.expert_bytes == 0 || host_expert_bytes == 0) return 0.0;
    const uint64_t holdable = budget_bytes(s.hw), margin = margin_bytes(s.hw, s.pol);
    const uint64_t ram = holdable > margin ? holdable - margin : 0;
    if (host_expert_bytes <= ram) return 0.0;
    const uint32_t slice = (uint32_t) m.expert_slice_bytes;
    const uint32_t lanes = slice ? s.hw.storage.best_lanes(slice) : 0;
    const double mibs = slice && lanes ? s.hw.storage.rate_at(slice, lanes) : 0.0;
    if (mibs <= 0.0) return 0.0;
    // The RAM cache holds `ram` of the host's experts: its share of the whole set's curve.
    const double scaled = (double) ram * (double) m.expert_bytes / (double) host_expert_bytes;
    const double ram_hit = std::clamp(expert_share_at(m, (uint64_t) scaled), 0.0, 1.0);
    const double miss = 1.0 - std::max(ram_hit, std::clamp(device_hit, 0.0, 1.0));
    const double host_share = std::min(1.0, (double) host_expert_bytes / (double) m.expert_bytes);
    return miss * routed * host_share / (1024.0 * 1024.0) / mibs;
}

// What gathering a prompt's experts costs a pass, in seconds: a pass as wide as a prompt routes nearly
// every expert, so each one computes on the device the experts that are not held there, uploaded from
// the host at the rate this machine measured. `off_device` is those bytes: everything past the device
// cache, or the experts the fitter left on the host. It is what a narrower graph pays more than once
// for a long prompt. Measured on rented cards, a 4.6k-token prompt in three 2048-token passes against
// one: 1.3 s a pass more on an 8 GB card with 13 GB of a model's experts off it, 0.44 s on a 12 GB
// card with 9.5 GB, 25 s on two 16 GB cards with some 50 GB of a 284B model's. 0 where the upload
// rate is unmeasured.
double prompt_gather_seconds(const ComputeDevice & dev, uint64_t off_device) {
    const double upload = dev.upload_gibs > 0.0 ? dev.upload_gibs : dev.upload_host_buffer_gibs;
    return upload > 0.0 ? (double) off_device / (1024.0 * 1024.0 * 1024.0) / upload : 0.0;
}

// The request a decision fixed at load is weighed by: the prompt tokens prefilled and the tokens
// decoded per request, and of the prompt tokens those in prompts longer than `narrow` (0: not asked).
// The requests this machine's last load recorded (MeasuredKnobs::mix_*), weighed with the default mix
// as PlannerPolicy::mix_prior_requests more, so a handful of requests does not swing the decision.
struct RequestMix {
    double prompt = 0.0, gen = 0.0, long_prompt = 0.0;
    int served = 0; // requests recorded here
    std::string words() const {
        return served > 0 ? u64s((uint64_t) served) + " requests served here, weighed with the default mix"
                          : std::string("none served yet: the default mix");
    }
};
RequestMix request_mix(const Planning & s, uint32_t narrow) {
    const MeasuredKnobs & mk = s.model.knobs;
    RequestMix r;
    const double prior = (double) s.pol.mix_prior_requests;
    const double prompt_default = (double) s.pol.mix_prompt_tokens_default;
    const bool served = mk.mix_requests > 0 && (mk.mix_gen_tokens > 0.0 || mk.mix_wide_prompt_tokens > 0.0);
    const double n = served ? (double) mk.mix_requests : 0.0;
    const auto blend = [&](double by_default, double recorded) {
        return (prior * by_default + n * recorded) / (prior + n);
    };
    r.served = served ? mk.mix_requests : 0;
    r.prompt = blend(prompt_default, mk.mix_wide_prompt_tokens);
    r.gen = blend((double) s.pol.mix_gen_tokens_default, mk.mix_gen_tokens);
    if (narrow > 0) {
        // The long prompts' tokens where a load counted them against this narrow width; else every
        // recorded prompt token where the mean prompt is itself past it.
        const bool counted = mk.usage_requests > 0 && mk.width_narrow == (int) narrow;
        const double recorded = counted                                       ? mk.usage_long_prompt_tokens
                                : mk.mix_wide_prompt_tokens > (double) narrow ? mk.mix_wide_prompt_tokens
                                                                              : 0.0;
        r.long_prompt = blend(prompt_default > (double) narrow ? prompt_default : 0.0, recorded);
    }
    return r;
}

// A side of a decision across loads that failed to open here as often as a plan lets it
// (MeasuredKnobs::unviable_max): it loses, whatever the other side is priced or measured at.
bool lost(int failed_opens) {
    return failed_opens >= MeasuredKnobs::unviable_max;
}

void choose_device_tier(Planning & s) {
    Placement & pl = s.placement;
    const ModelProfile & m = s.model;
    if (!pl.fitted || pl.n_gpu_layers <= 0 || !m.is_moe || m.n_moe_layer == 0) return;
    for (const char * knob : {"device-cache-mb", "gpu-layers", "tensor-overrides"})
        if (s.req.is_pinned(knob)) return; // the caller placed it; the fitter's or theirs stands
    // The cache computes the experts on the device, and overlap waits for them inside the CPU's own
    // matmul: a caller who asked for overlap keeps the fitter's placement, and overlap they did not
    // ask for (a front-end's default) is turned off below where the cache is armed.
    if (s.req.is_pinned("overlap") && s.p.config.moe.overlap) return;
    // The dense part of every layer must already be on the devices: it is read by every token, so it
    // has the first claim on the fast memory, and the fitter gives it that claim before any expert.
    // A layer whose dense part is left on the host would cross to reach a cache on the device.
    // llama.cpp counts n_layer + 1 slots, the output last, and fills the devices from the top, so
    // every block is there only from n_layer + 1: at n_layer the first block is still the host's,
    // which is the count choose_graph_width calls short too.
    if ((uint32_t) pl.n_gpu_layers <= m.n_layer) {
        s.note("device-cache-mb", "0", Source::Derived,
               "Not every layer's dense part fits on the device: no cache of experts there.",
               "the capacity fitter put " + u64s((uint64_t) pl.n_gpu_layers) + " of " + u64s(m.n_layer + 1) +
                   " layers on the devices, so some dense parts are on the host, and a cache would leave them there");
        return;
    }
    if (s.host_layers == 0) return;             // every expert on the devices: the model fits there
    if (pl.all_expert_patterns.empty()) return; // no adapter said how to name them
    const uint64_t on_devices = m.expert_bytes > s.host_expert_bytes ? m.expert_bytes - s.host_expert_bytes : 0;
    // The cache is not bound to the fitter's granularity: it takes the experts' bytes AND what the
    // fitter left free under one more whole layer, which on a small card is everything past the dense
    // part. The same budget as the RAM cache's: what the memory can hold, floored at one token cycle.
    const TierSplit tier = device_tier_split(pl, m, on_devices);
    // The projector is on this device too (every layer is). The fitter reserved its room before
    // placing layers (Placement::device_projector_bytes, already off the spare and in the device's
    // bytes); whatever it did not reserve - a placement made without the fitter - comes off what the
    // card has before the cache does. Where the card's memory is the host's it is charged there.
    const uint64_t unreserved = m.projector_bytes - std::min(m.projector_bytes, pl.device_projector_bytes);
    const uint64_t projector = s.p.config.device_use != DeviceUse::CpuOnly && !devices_share_host_memory(s.hw)
                                   ? std::min(tier.budget, unreserved)
                                   : 0;
    const uint64_t budget = tier.budget - projector;
    // What the budget holds beyond the fitter's experts, for the rationale.
    const uint64_t spare = tier.budget > on_devices ? tier.budget - on_devices : 0;
    if (budget == 0) return;

    const auto mib_s = [](uint64_t b) { return u64s(mib(b)); };
    const std::string held =
        tier.budget >= on_devices
            ? mib_s(budget) + " MiB (" + mib_s(on_devices) + " the capacity fitter gave " +
                  u64s(s.n_moe_layer - s.host_layers) + " of " + u64s(s.n_moe_layer) + " layers' experts, " +
                  (projector > 0 ? "less the projector's " + mib_s(projector) + " MiB, " : "") + mib_s(spare) +
                  " it left free under one more whole layer)"
            : mib_s(budget) + " MiB (what every card can take its share of, by its layers, of the " +
                  mib_s(on_devices) + " the capacity fitter gave " + u64s(s.n_moe_layer - s.host_layers) + " of " +
                  u64s(s.n_moe_layer) + " layers' experts and what it left free" +
                  (projector > 0 ? ", less the projector's " + mib_s(projector) + " MiB" : "") + ")";
    std::string why;
    const ComputeDevice * dev = device_tier_device(s, why);
    // The host and the device on the same experiment: the model's own expert block where both were
    // measured on it. The synthetic GEMV overstated a 48-core host's expert rate six times over.
    const RatePair rates = dev ? expert_rates(s.hw, *dev) : RatePair{};
    if (!dev || rates.host <= 0.0) {
        s.note("device-cache-mb", "0", dev ? Source::Unprobed : Source::Derived,
               "The device holds whole layers' experts, not a cache of the most used ones.",
               "a cache of experts on the device could hold " + held + ", but " +
                   (dev ? std::string("the host's own expert rate is unmeasured") : why));
        return;
    }
    // Below one token cycle an LRU evicts an expert before the same token needs it again: the cache
    // costs its memory and returns no hits. The RAM cache's floor, for the same reason.
    if (budget < m.token_cycle_bytes) {
        s.note("device-cache-mb", "0", Source::Derived,
               "The device has less room than one token's experts: it keeps the fitter's whole layers.",
               "a cache of experts on " + dev->name + " could hold " + held + ", under this model's token cycle of " +
                   mib_s(m.token_cycle_bytes) +
                   " MiB, below which a cache evicts an expert before the same token "
                   "needs it again");
        return;
    }
    const double share = expert_share_at(m, budget);
    char nums[768];
    std::snprintf(nums, sizeof(nums),
                  "%s computes this model's experts at %.0f GiB/s against the host's %.0f%s; a cache of %s "
                  "serves %.0f%% of a token's routed bytes (%s)",
                  dev->name.c_str(), rates.device, rates.host,
                  rates.real_layer ? ", both measured on this model's own expert block" : "", held.c_str(),
                  100.0 * share, expert_share_source(m));
    if (rates.device <= rates.host * (1.0 + (double) s.pol.min_backend_win)) {
        s.note("device-cache-mb", "0", Source::Measured,
               "The host computes the experts as fast as the device: they stay where the fitter put them.",
               std::string(nums) + ", inside the " + u64s((uint64_t) (s.pol.min_backend_win * 100.0f + 0.5f)) + "%" +
                   " margin of each other, so the fastest processor is not the device and nothing moves");
        return;
    }
    // Decided by calculation, and corrected by what the loads served here measured
    // (MeasuredKnobs::device_cache_tps). The rule above says the device is the faster processor, which
    // is necessary and not sufficient: every miss of the cache crosses the link before the device
    // computes it, and where the cache holds a small part of the experts on a slow link that is the
    // whole token. Measured on two 24 GB cards over PCIe 3, a 355B model: a cache of a tenth of its
    // experts decoded at 1.6 tok/s where the fitter's placement did 3.9; on a 12 GB card over PCIe 4
    // the cache won 17.5 to 9.9. So both sides are priced from this machine's probed rates: the cache
    // with its misses uploaded at the hit rate its size reaches (device_cache_decode_seconds), the
    // fitter's placement from its ceiling, which has no miss to pay (fitter_decode_ceiling), at the
    // part of it such a placement reaches (PlannerPolicy::fitter_ceiling_realized). A load
    // serves the side priced faster and times its own decode as it serves; a side measured stands at
    // its measurement, and the other takes over only where its price, or its own measurement, is past
    // that by the margin. No load is opened to try a side: the load-time calibration that did reopened
    // the model once per side before the first answer, 609 s to ready against 126 s on two 16 GB cards,
    // and a side tried on a served load cost that whole session its rate (37 tok/s against 68).
    {
        const MeasuredKnobs & mk = s.model.knobs;
        const double win = 1.0 + (double) s.pol.min_backend_win;
        const std::string margin = u64s((uint64_t) (s.pol.min_backend_win * 100.0f + 0.5f)) + "%";
        // Each side's compute, and what its host's experts wait on flash for where they stream. With
        // the cache every expert is the host's; the fitter's placement leaves it the layers it did.
        const double cache_compute = device_cache_decode_seconds(s, *dev, budget);
        const double cache_s =
            cache_compute > 0.0
                ? cache_compute + host_expert_flash_seconds(s, m.expert_bytes, expert_share_at(m, budget))
                : 0.0;
        const double cache_priced = cache_s > 0.0 ? 1.0 / cache_s : 0.0;
        // The fitter's placement's compute at the part of its ceiling such a placement was measured to
        // reach; its flash reads are not part of that ceiling and are added whole.
        const double ceiling = fitter_decode_ceiling(s) * (double) s.pol.fitter_ceiling_realized;
        const double fitter_priced =
            ceiling > 0.0 ? 1.0 / (1.0 / ceiling + host_expert_flash_seconds(s, s.host_expert_bytes, 0.0)) : 0.0;
        // Each side at its measurement where a load timed it, at its price otherwise.
        const double cache_at = mk.device_cache_tps > 0.0 ? mk.device_cache_tps : cache_priced;
        const double fitter_at = mk.no_device_cache_tps > 0.0 ? mk.no_device_cache_tps : fitter_priced;
        char sides[320];
        std::snprintf(
            sides, sizeof(sides),
            "%.2f tok/s with a device cache (%s) against %.2f with the fitter's "
            "placement (%s)",
            cache_at, mk.device_cache_tps > 0.0 ? "measured on a load here" : "priced from the probed rates", fitter_at,
            mk.no_device_cache_tps > 0.0 ? "measured on a load here" : "priced from its ceiling at the probed rates");
        // A side that could not be opened here loses, whatever the other is priced at: a cache that
        // cannot be created leaves the fitter's placement, and a placement that cannot be loaded leaves
        // the cache.
        if (lost(mk.device_cache_unviable)) {
            s.note("device-cache-mb", "0", Source::Measured,
                   "The experts stay where the fitter put them: a device cache could not be opened here.",
                   std::string(nums) + "; loads with a device cache failed on this machine, so the fitter's "
                                       "placement stands");
            return;
        }
        // The fitter's placement takes over only past the cache by the margin: the cache is the side in
        // force wherever the device is the faster processor, and a price is not a reason to leave it on
        // a few percent. A side with no price and no measurement decides nothing.
        const bool fitter_wins =
            !lost(mk.no_device_cache_unviable) && cache_at > 0.0 && fitter_at > 0.0 && fitter_at > cache_at * win;
        if (fitter_wins) {
            if (mk.no_device_cache_tps <= 0.0) s.p.config.trial.device_tier = 0; // served, and timed as it serves
            s.note("device-cache-mb", "0", mk.no_device_cache_tps > 0.0 ? Source::Measured : Source::Derived,
                   "The experts stay where the fitter put them: on this machine that decodes faster than a "
                   "device cache.",
                   std::string(nums) + "; " + sides + ", past the " + margin + " margin" +
                       (mk.no_device_cache_tps > 0.0 ? "" : "; this load times its own decode as it serves"));
            return;
        }
        if (lost(mk.no_device_cache_unviable))
            std::snprintf(nums + std::strlen(nums), sizeof(nums) - std::strlen(nums),
                          "; loads with the fitter's placement failed on this machine, so the cache stands");
        else if (cache_at > 0.0 && fitter_at > 0.0)
            std::snprintf(nums + std::strlen(nums), sizeof(nums) - std::strlen(nums), "; %s, not past the %s margin",
                          sides, margin.c_str());
        if (mk.device_cache_tps <= 0.0) s.p.config.trial.device_tier = 1; // served, and timed as it serves
    }
    // Every layer's experts back to the host, where the streamer or the resident set serves them, and
    // the bytes the fitter gave them, with what it left free, become the cache. The regime and the
    // ledger below see the host holding the whole expert set, which is now what it holds.
    pl.host_expert_layers = m.moe_layers;
    if (pl.host_expert_layers.empty())
        for (uint32_t il = 0; il < m.n_layer; ++il)
            pl.host_expert_layers.push_back(il);
    pl.override_patterns = pl.all_expert_patterns;
    if (tier.resplit) {
        const auto list = [](const std::vector<std::pair<std::string, uint32_t>> & v) {
            std::string out;
            for (const auto & e : v)
                out += (out.empty() ? "" : ", ") + e.first + " " + u64s(e.second);
            return out;
        };
        s.note("device-split", list(tier.split), Source::Derived,
               "The layers are spread over the cards by the room each has beside the expert cache.",
               "llama.cpp hands each card a share of the cache's budget by its layers, so the split follows each "
               "card's room under the cache (what it had free, the experts the fitter put on it, and its layers' "
               "dense part and context): " +
                   list(tier.split) + " in place of the fitter's " + list(pl.device_split) +
                   ", so every card can take its share of " + mib_s(budget) +
                   " MiB; each layer a card takes on is charged the file's largest dense layer and its "
                   "context, since it is loaded before the cache and a card it overfills cannot fall back");
        pl.device_split = tier.split;
    }
    pl.device_bytes = pl.device_bytes > on_devices ? pl.device_bytes - on_devices : 0;
    pl.device_bytes += projector; // the part the fitter had not reserved is the device's all the same
    s.host_layers = s.n_moe_layer;
    s.host_expert_bytes = m.expert_bytes;
    // The MTP head of a file of its own follows the experts to the processor that computes them
    // fastest: its block is one more expert block, and drafting on a slower processor was measured
    // losing (35 ms of a 65 ms token on 8 slow cores, against no head at all). It takes its bytes
    // from the cache, never from the fitter's layers, so the target keeps the placement it was
    // fitted, and never below the cache's floor; where it does not fit, it drafts on the host and the
    // plan says so.
    uint64_t cache_bytes = budget;
    RunConfig & c = s.p.config;
    if (c.spec.is_mtp() && !c.spec.mtp_head.empty() && !s.req.is_pinned("mtp-head-on-devices")) {
        const uint64_t head = m.mtp_head_bytes;
        c.spec.mtp_head_on_devices = head > 0 && head + std::max<uint64_t>(m.token_cycle_bytes, 1) <= cache_bytes;
        if (c.spec.mtp_head_on_devices) {
            cache_bytes -= head;
            pl.device_bytes += head;
            s.note("mtp-head-on-devices", "true", Source::Measured,
                   "The MTP head (" + size_brief(head) + ") is loaded on " + dev->name + ", beside the expert cache.",
                   "the head's block is an expert block, and " + dev->name +
                       " computes this model's experts faster than the host by the margin; its " + mib_s(head) +
                       " MiB come out of the cache's " + mib_s(budget) + " MiB");
        } else {
            s.note("mtp-head-on-devices", "false", Source::Derived, "The MTP head drafts on the host.",
                   head == 0 ? "the head file's size is unknown"
                             : "the head's " + mib_s(head) + " MiB do not fit in the cache's " + mib_s(budget) +
                                   " MiB on " + dev->name + " above one token cycle");
        }
    }
    s.p.config.moe.device_cache_mb = (int) mib(cache_bytes);
    // A front-end's own default overlap (the server arms it wherever the hook is available) waits for
    // the experts inside the CPU's matmul, which no longer computes them: the session refuses the pair,
    // and a server left with both failed every load. A caller who asked for it never reaches here.
    MoeStreamConfig & moe = s.p.config.moe;
    if (moe.overlap || moe.io_two_wave || s.p.config.trial.overlap) {
        moe.overlap = moe.io_two_wave = false;
        s.p.config.trial.overlap = false;
        s.note("overlap", "off", Source::Derived, "Off: the experts are computed on the device, beside its cache.",
               "overlap hides the CPU's expert compute behind the streamer's reads, and with the device expert "
               "cache armed the experts are computed on " +
                   dev->name + ", so there is nothing on the CPU for it to hide");
    }
    s.note("device-cache-mb", mib_s(cache_bytes), Source::Measured,
           "A " + size_brief(cache_bytes) + " cache of the most used experts on " + dev->name +
               ", in place of whole layers' experts.",
           std::string("the experts used most live on the processor that computes them fastest, as the RAM "
                       "cache holds what flash would read: ") +
               nums + "; a miss is uploaded from the host");
}

// Where the device cache's misses come from. A resident expert is a page of the file's mapping,
// ordinary memory, and a device's backend copies ordinary memory through a staging buffer of its
// own; from a buffer of the backend's own host type it may copy directly (bmoe/hardware_profile.h,
// `upload_host_buffer_gibs`). The copy is made where the gain was measured on this device, past the
// margin. It takes the place of the experts' pages in the mapping, which host_all already charges:
// once the tensors point at the copy nothing reads those pages, and the session hands them back
// (Session::Impl::stage_experts_in_host_buffer), so the host holds the experts once, not twice. What
// changes is that the copy cannot be paged out, which the regime's margin already covers. Charging
// both sides refused the copy on every 64 GB machine a 32 GB expert set runs on (three rented cards,
// Qwen3.8-Flash-Next Q2_0): misses arrived at half the rate, decode 20.7 tok/s on a 12 GB card where
// a machine with the room for both took the copy and decoded 29-32. The experts streamed instead
// live in the streamer's slots, which this does not move.
void choose_expert_staging(Planning & s) {
    MoeStreamConfig & moe = s.p.config.moe;
    if (moe.device_cache_mb <= 0 || s.p.regime != Regime::Fits || s.req.is_pinned("expert-host-buffer")) return;
    std::string why;
    const ComputeDevice * dev = device_tier_device(s, why);
    if (!dev) return;
    const auto mib_s = [](uint64_t b) { return u64s(mib(b)); };
    const std::string pct = u64s((uint64_t) (s.pol.min_backend_win * 100.0f + 0.5f)) + "%";
    char rates[200];
    std::snprintf(rates, sizeof(rates),
                  "a miss reaches %s at %.1f GiB/s from the file's mapping and at %.1f from "
                  "the backend's own host memory",
                  dev->name.c_str(), dev->upload_gibs, dev->upload_host_buffer_gibs);
    if (dev->upload_host_buffer_gibs <= 0.0 || dev->upload_gibs <= 0.0 ||
        dev->upload_host_buffer_gibs <= dev->upload_gibs * (1.0 + (double) s.pol.min_backend_win)) {
        s.note("expert-host-buffer", "false", dev->upload_host_buffer_gibs > 0.0 ? Source::Measured : Source::Derived,
               "The device cache uploads from the file's mapping.",
               dev->upload_host_buffer_gibs > 0.0
                   ? std::string(rates) + ", inside the " + pct + " margin"
                   : dev->name + " offers no host memory of its own to upload from, or its rate is unmeasured");
        return;
    }
    const uint64_t copy = s.host_expert_bytes;
    // The copy in place of the mapped experts host_all counts (see above): the same total, held once.
    const uint64_t held = s.held_all() - std::min(s.held_all(), s.host_expert_bytes) + copy;
    if (held > s.usable) {
        s.note("expert-host-buffer", "false", Source::Derived, "The device cache uploads from the file's mapping.",
               std::string(rates) + ", but with the copy's " + mib_s(copy) + " MiB in place of the mapped experts " +
                   "the host would hold " + mib_s(held) + " MiB, past the " + mib_s(s.usable) + " MiB it may");
        return;
    }
    moe.expert_host_buffer = dev->name;
    s.note("expert-host-buffer", dev->name, Source::Measured,
           "The experts copied into " + dev->name + "'s own host memory, for the cache's misses.",
           std::string(rates) + "; the copy's " + mib_s(copy) + " MiB takes the place of the experts' pages in " +
               "the file's mapping, which are handed back, so the host holds " + mib_s(held) + " MiB of the " +
               mib_s(s.usable) + " it may");
}

// Who computes the device cache's misses. With every expert on the device the cores compute none, and
// a miss crosses the link before the device computes it; the host can instead compute part of the
// misses from its own copy while the device computes the hits and the rest, and llama.cpp adds the
// two halves of the layer (llama_context_params::moe_cache_cpu_share).
//
// A person's setting, never the plan's. The share is fixed when the context is created, so its two
// sides cannot be switched within a load, and no price of it held: where the cache never fills,
// llama.cpp computes no miss on the host at all and every MoE layer's host half is pure cost, and
// whether it fills is not something a plan can know. On a 32 GB card a share priced at 0.89 took the
// decode from 114 to 68 tok/s with no miss on the host, where on a 12 GB card 0.77 took it from 18.5
// to 21.2. Deciding it meant a load of its own for each side, which a plan no longer opens, so the
// plan uploads every miss and says how to set the share by hand.
//
// Over streamed experts the host half would read experts the streamer has not been asked to hold
// until the layer runs; that pair is not gated, and the plan says so where a share was asked for.
void choose_device_cache_cpu_share(Planning & s) {
    MoeStreamConfig & moe = s.p.config.moe;
    if (moe.device_cache_mb <= 0 || s.req.is_pinned("device-cache-cpu-share")) return;
    moe.device_cache_cpu_share = 0.0f;
    s.note("device-cache-cpu-share", "0", Source::Policy, "Every miss of the device cache is uploaded to the device.",
           "computing part of the misses on the CPU instead is fixed when the model's context is created, and what it "
           "gains or costs depends on whether the cache fills, which only a load of its own can tell; the plan opens "
           "no load to try it. " +
               flag_spelling("device-cache-cpu-share") + " sets it by hand");
}

// The account of the layers home: what the split measured, then how it came out. Each stage that
// settles the outcome passes only its own clause, so none of them has to find the other's in prose.
std::string home_reason(const Planning::LayersHome & h, const std::string & outcome) {
    return h.measured + "; " + outcome;
}

// ── the device crossing: layers on a device for decode, or fewer (measured-planning.md, Step 3) ──
// Where not every dense part fits beside the experts, the capacity fitter puts the dense part of
// some layers on a device and leaves their experts on the host: each such layer then crosses the
// host/device boundary twice a token, its dense matmuls on the device and its experts on the host.
// The fitter decides that by capacity alone, and it is right wherever the device's dense rate saves
// more than two crossings cost. Where it does not - a slow link, a device no faster on those shapes -
// the same layers are cheaper whole on the host, and the device keeps only the layers it holds
// whole, contiguous at the top, crossing once each way.
//
// Priced on measured facts only: each device's rate on the model's own dense matrices against the
// host's on the same experiment, and its measured crossing. The RAM is not free - the host then
// holds those layers' dense part and their context - so the saving enters the ranking (allocate.cpp)
// as one more use of the budget, against the cache; where it is funded the plan is made again from
// the layers brought home.
//
// The device keeps as many layers whole as the fitter gave it, so what it holds can only shrink -
// but they become the top ones, so each is bounded by the largest layer the file has, and a device
// whose bound does not fit its free memory keeps the fitter's placement.
void price_layers_home(Planning & s) {
    const Placement & pl = s.projected;
    const ModelProfile & m = s.model;
    if (!s.home.considered || !pl.fitted || pl.n_gpu_layers == 0 || m.n_layer == 0) return;
    for (const char * knob : {"gpu-layers", "tensor-overrides", "devices", "device-cache-mb"})
        if (s.req.is_pinned(knob)) return;
    if (s.p.config.moe.device_cache_mb > 0) return; // the device tier keeps every layer on the device

    // The device slots, top down as llama.cpp fills them: the layers and, last, the output.
    const LayerSlots ls = layer_slots(pl.n_gpu_layers, m.n_layer);
    const uint32_t slots = m.n_layer + 1, ngl = ls.on_device;
    std::vector<std::pair<std::string, uint32_t>> split = pl.device_split;
    if (split.empty()) {
        for (const ComputeDevice & d : s.hw.devices)
            if (!d.is_cpu && d.has_own_memory()) split.emplace_back(d.name, ngl);
        if (split.size() != 1) return; // which device holds the layers is not said
    }
    const auto is_moe = [&](uint32_t il) {
        return m.moe_layers.empty() || std::find(m.moe_layers.begin(), m.moe_layers.end(), il) != m.moe_layers.end();
    };
    const auto experts_home = [&](uint32_t il) {
        return std::find(pl.host_expert_layers.begin(), pl.host_expert_layers.end(), il) != pl.host_expert_layers.end();
    };
    struct Part {
        const ComputeDevice * dev = nullptr;
        uint32_t dense_only = 0, whole = 0;
        bool output = false;
    };
    std::vector<Part> parts;
    uint32_t at = ls.first_device_slot(), dense_only = 0;
    for (const auto & e : split) {
        Part q;
        for (const ComputeDevice & d : s.hw.devices)
            if (!d.is_cpu && d.name == e.first) q.dev = &d;
        for (uint32_t k = 0; k < e.second && at < slots; ++k, ++at) {
            if (at == m.n_layer)
                q.output = true;
            else if (is_moe(at) && experts_home(at))
                ++q.dense_only;
            else
                ++q.whole;
        }
        dense_only += q.dense_only;
        parts.push_back(q);
    }
    if (dense_only == 0) return; // every device layer is whole: one boundary, nothing to price

    const auto mib_s = [](uint64_t b) { return u64s(mib(b)); };
    std::string unpriced;
    for (const Part & q : parts) {
        if (q.dense_only == 0) continue;
        if (!q.dev || !q.dev->has_own_memory())
            unpriced = "a device holding them is not known to have memory of its own";
        else if (q.dev->identity_ok != Tri::Yes)
            unpriced = q.dev->name + " has not been shown to compute the host's results";
        else if (q.dev->split_seconds <= 0.0 || dense_rates(s.hw, *q.dev).device <= 0.0 ||
                 dense_rates(s.hw, *q.dev).host <= 0.0)
            unpriced = "what a crossing to " + q.dev->name + " or its dense matmuls cost is unmeasured";
    }
    Decision & d = s.home.decision;
    d.knob = "device-layers";
    d.value = u64s(dense_only) + " split";
    if (!unpriced.empty()) {
        d.source = Source::Unprobed;
        d.brief = u64s(dense_only) + " layers are split between a device and the CPU; the hops are unpriced.";
        d.reason = "the capacity fitter put the dense part of " + u64s(dense_only) +
                   " layers on a device and their experts on the host, two crossings a token each, and " + unpriced +
                   ": the placement stands as the fitter made it";
        return;
    }

    // Per token, only what differs: those layers' dense matmuls on the device plus two crossings each,
    // against the same matmuls on the host.
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    const double dense_tok =
        (double) (m.group(WeightGroup::Attention).bytes_per_token + m.group(WeightGroup::DenseFfn).bytes_per_token) /
        (double) m.n_layer;
    double on_device = 0.0, on_host = 0.0;
    for (const Part & q : parts) {
        if (q.dense_only == 0) continue;
        const RatePair r = dense_rates(s.hw, *q.dev);
        on_device += q.dense_only * (dense_tok / (r.device * gib) + 2.0 * q.dev->split_seconds);
        on_host += q.dense_only * dense_tok / (r.host * gib);
    }
    char nums[256];
    std::snprintf(nums, sizeof(nums),
                  "their dense matmuls and two crossings each cost %.2f ms a token on the device, the same matmuls "
                  "%.2f ms on the host",
                  1e3 * on_device, 1e3 * on_host);
    d.source = Source::Measured;
    if (on_host * (1.0 + (double) s.pol.min_backend_win) >= on_device) {
        d.brief = u64s(dense_only) + " layers keep their dense part on the device: it saves more than the hops cost.";
        d.reason = "the capacity fitter put the dense part of " + u64s(dense_only) +
                   " layers on a device and their experts on the host; measured, " + nums + ", so the split stands";
        return;
    }

    // The alternative: each device keeps as many whole layers, at the top; the rest come home whole.
    const uint64_t layer_bound = m.largest_expert_layer_bytes + m.largest_layer_dense_bytes;
    const uint64_t kv = pl.device_context_bytes; // bounded by all of it: per layer it need not be uniform
    const uint64_t dense_home = (uint64_t) dense_only * m.largest_layer_dense_bytes;
    std::string refused = layer_bound == 0 || dense_home == 0 ? "the size of the file's layers is unknown" : "";
    uint32_t whole_total = 0;
    for (const Part & q : parts) {
        const uint32_t keep = q.whole + (q.output ? 1 : 0);
        whole_total += keep;
        if (keep == 0 || !refused.empty()) continue;
        const uint64_t bound = (uint64_t) q.whole * layer_bound + (q.output ? m.group(WeightGroup::Output).bytes : 0) +
                               kv + pl.device_compute_of(q.dev->name);
        if (q.dev->memory_free == 0 || bound > q.dev->memory_free)
            refused = q.dev->name + " could not be shown to hold its whole layers at the top (" + mib_s(bound) +
                      " MiB bound against " + mib_s(q.dev->memory_free) + " MiB free)";
    }
    if (!refused.empty()) {
        d.brief = u64s(dense_only) + " layers stay split: bringing them home cannot be accounted here.";
        d.reason =
            std::string("measured, ") + nums + "; bringing them onto the host is not accounted, because " + refused;
        return;
    }
    Placement w = pl;
    w.n_gpu_layers = (int32_t) whole_total;
    w.device_split.clear();
    if (!pl.device_split.empty())
        for (const Part & q : parts) {
            const uint32_t keep = q.whole + (q.output ? 1 : 0);
            if (keep && q.dev) w.device_split.emplace_back(q.dev->name, keep);
        }
    // Below the devices' slots every block is the host's, experts and all, so no override is needed,
    // and the layers the devices keep are whole.
    w.override_patterns.clear();
    w.host_expert_layers.clear();
    const uint32_t first_device = slots - std::min(whole_total, slots);
    for (uint32_t il = 0; il < m.n_layer && il < first_device; ++il)
        if (is_moe(il)) w.host_expert_layers.push_back(il);
    w.host_resident_bytes += dense_home + kv;
    w.raw_host_context_bytes += kv;
    w.device_bytes -= std::min(w.device_bytes, dense_home + kv);
    w.device_context_bytes = 0;

    s.home.priced = true;
    s.home.layers = dense_only;
    s.home.bytes = dense_home + kv;
    s.home.saved = on_device - on_host;
    s.home.whole = std::move(w);
    d.brief = u64s(dense_only) + " layers keep their dense part on the device: the RAM to bring them home buys more "
                                 "as cache.";
    s.home.measured = std::string("measured, ") + nums;
    d.reason = home_reason(s.home, "bringing them home costs the host " + mib_s(s.home.bytes) +
                                       " MiB (their dense part and, bounded by all of it, the context), which the "
                                       "ranking weighed against the cache and did not fund");
}

// A model with no experts streams nothing, but the capacity fitter's placement still holds: it is
// what puts a model that fits a card on the card. Without it the run keeps n_gpu_layers 0 and, with
// the devices left to the plan, opens on the CPU alone. The fitter's answer is applied as it is, the
// same fields apply_placement sets; nothing below prices it, since nothing streams.
void place_dense(Planning & s) {
    const Placement & placement = s.placement;
    if (!placement.fitted) return;
    RunConfig & c = s.p.config;
    if (placement.n_ctx && !s.req.is_pinned("ctx-size")) c.n_ctx = (int) placement.n_ctx;
    c.n_gpu_layers = placement.n_gpu_layers;
    c.buft_overrides = placement.override_patterns;
    c.device_split.clear();
    for (const auto & d : placement.device_split)
        c.device_split.emplace_back(d.first, (int) d.second);
    const std::string layers =
        placement.n_gpu_layers < 0 ? "every layer" : u64s((uint64_t) std::max(0, placement.n_gpu_layers)) + " layers";
    s.note("placement", layers + " on devices", Source::Measured,
           layers + " on a device, as the capacity fitter placed them.", placement.outcome);
}

// ── the regime: a ratio of the two profiles, never a property of either ─────────
// The three inputs no plan can be made from. True when the plan has declined and is complete.
bool declines_unplannable(Planning & s) {
    if (!s.model.ok) {
        s.p.regime = Regime::Unknown;
        s.decline("the model file could not be read, so nothing about it could be planned");
        s.note("moe-stream", "off", Source::Unprobed, "Not streamed: the model file could not be read.",
               "model profile unavailable");
        return true;
    }
    if (!s.model.is_moe) {
        s.p.regime = Regime::NotMoe;
        s.decline("this model has no routed experts: there is nothing for the expert streamer to do");
        s.note("moe-stream", "off", Source::Derived, "Not streamed: this model has no experts to stream.",
               "the file carries no expert tensors");
        place_dense(s);
        return true;
    }
    if (s.hw.residency_budget == 0) {
        s.p.regime = Regime::Unknown;
        s.decline("this machine reported no residency budget, so no size could be derived from it");
        s.note("moe-stream", "off", Source::Unprobed, "Not streamed: this machine's memory was not measured.",
               "machine not profiled");
        return true;
    }
    return false;
}

void size_budget(Planning & s) {
    s.margin = margin_bytes(s.hw, s.pol);
    s.holdable = budget_bytes(s.hw);
    s.usable = s.holdable > s.margin ? s.holdable - s.margin : 0;
}

// ── the widest graph ────────────────────────────────────────────────────────────
// A prompt is computed in graphs `n_ubatch` tokens wide, and the compute buffers are reserved for
// the widest of them. Left to follow the context that is one graph per context, which is the
// fastest prefill there is and costs nothing at a small context; at a large one it is several GiB
// taken from the cache for a pass that happens once per prompt. So the width is the widest whose
// buffers stay inside their share of what the process may hold, found by halving from the context.
// A width the caller set is the caller's; a slope that was not measured decides nothing, and the
// knob keeps its default.
//
// A device's buffers come out of this pool where its memory is the host's. A device with memory of
// its own pays for its width in that memory, and where the model does not fit the device, every byte
// there is one the expert cache or a layer's dense part would have used: the same trade, on the
// device. Measured on a 12 GB card at a 16384 context: a graph as wide as the context reserved 6.6 GB
// of the card for compute buffers, the fitter left two layers' dense part on the host, no device cache
// was armed, and the decode ran at 13 tok/s, where 2048 tokens of width reserved 0.7 GB and the same
// card decoded at 37 with a 6 GB cache. So where a device is short of room, its buffers are held to
// the same share of its memory as the host's are of the host's; a device the whole model fits keeps
// the widest graph, since its buffers take nothing from anything there.
} // namespace

// From `top`, halve the graph's width while its buffers are past their share, and stop at `floor`
// rather than above it: a width that is not a power of two times the floor (a context of 4092
// tokens) used to stop at the last halving still over it (1023, where 511 is under a 512 floor),
// with the buffers past their share and the floor never reached.
namespace {
template <typename Over> uint32_t narrowed_width(uint32_t top, uint32_t floor, Over over) {
    uint32_t w = top;
    while (w > floor && over(w))
        w = std::max(floor, w / 2);
    return w;
}
} // namespace

uint32_t device_share_width(const Placement & placement,
                            const HardwareProfile & hw,
                            const PlannerPolicy & policy,
                            uint32_t min_width) {
    const uint32_t full = placement.compute_width;
    if (full == 0 || placement.device_compute_per_token == 0) return 0;
    const uint64_t device_memory = hw.device_local_memory();
    if (device_memory == 0) return 0;
    // Each card against its own memory: two cards of different sizes each reserve their own buffers,
    // and the smaller is the one that runs short. The sum against the sum where the fitter did not say
    // which card takes what.
    auto over = [&](uint32_t w) {
        const Placement q = placement.at_width(w);
        bool per_card = false;
        for (const ComputeDevice & d : hw.devices) {
            if (!d.has_own_memory() || d.memory_total == 0) continue;
            const uint64_t part = q.device_compute_of(d.name);
            if (part == 0) continue;
            per_card = true;
            if ((double) part > (double) d.memory_total * policy.compute_share_max) return true;
        }
        return !per_card && (double) q.device_compute_bytes > (double) device_memory * policy.compute_share_max;
    };
    return narrowed_width(full, std::max(policy.ubatch_min, min_width), over);
}

namespace {

// The graph width where the rule above leaves it wide and the devices are short of the room it takes:
// every layer on a card of its own, the experts not all there, and the card's buffers past its share.
// A narrower graph then gives the experts more of the card - a larger cache of them, or more of them
// placed there - so every decoded token is faster, and it prefills a long prompt in more passes, each
// gathering onto the device the experts that are not held there. Which wins is this machine's and this
// use's: the width is the one whose request takes less time,
//
//   t(w) = prompt tokens x prefill seconds a token + passes(w) x gather seconds a pass
//          + decoded tokens x decode seconds a token (w)
//
// priced from the rates this machine probed (device_cache_decode_seconds, fitter_decode_ceiling,
// prompt_gather_seconds) over the requests it served (MeasuredKnobs::mix_*), or
// PlannerPolicy::mix_*_default on a first load. A load serves the width priced faster and times what it
// serves; a side's measured decode or prefill corrects the price of both (the ratio between the two
// stays the price's until both are measured, and is then the measurements'). Taken on the decode
// alone, as it was, the narrow graph won everywhere it decoded faster: +40% decode and -40% on a
// 4.6k-token prompt on an 8 GB card, -28% on a 12 GB one, -47% on two 16 GB cards with a 284B model.
// Measured on two 24 GB cards with GLM-4.5-Air: 14.2 tok/s decoded at 16384 tokens wide against 19.4 at
// 512, where the wide graph prefilled a 9.5k-token prompt at 610 tok/s and a 512-wide one at 104.
// True when it decided the width and said so.
bool width_across_loads(Planning & s, uint32_t full) {
    Plan & p = s.p;
    const Placement & pl = s.placement;
    const ModelProfile & m = s.model;
    if (s.req.is_pinned("ubatch") || devices_share_host_memory(s.hw) || !pl.fitted || pl.n_gpu_layers == 0 ||
        pl.host_expert_layers.empty() || m.expert_bytes == 0)
        return false;
    const uint32_t narrow = device_share_width(pl, s.hw, s.pol, (uint32_t) p.config.graph_width_floor());
    if (narrow == 0 || narrow >= full) return false;
    // The device whose rates price the two widths: the one the cache is on, or with no cache the first
    // card that holds layers.
    const bool cache = p.config.moe.device_cache_mb > 0;
    std::string why;
    const ComputeDevice * dev = cache ? device_tier_device(s, why) : nullptr;
    if (!cache)
        for (const ComputeDevice & d : s.hw.devices)
            if (!dev && !d.is_cpu && d.usable != Tri::No && d.has_own_memory()) dev = &d;
    if (!dev) return false;
    const MeasuredKnobs & mk = s.model.knobs;
    const bool same = mk.width_narrow == (int) narrow;
    const std::string n = u64s(narrow);
    const std::string room = "the devices' compute buffers at the full width are past their share of the cards' "
                             "memory, which a narrower graph gives the experts";
    const auto take_narrow = [&] {
        s.placement = s.placement.at_width(narrow);
        p.placement = s.placement;
        p.config.n_ubatch = (int) narrow;
    };
    // A side that could not be opened here loses.
    if (same && (lost(mk.narrow_unviable) || lost(mk.wide_unviable))) {
        const bool narrow_wins = lost(mk.wide_unviable) && !lost(mk.narrow_unviable);
        if (narrow_wins) take_narrow();
        s.note("ubatch", narrow_wins ? n : u64s((uint64_t) p.config.graph_width()), Source::Measured,
               narrow_wins ? n + " tokens per pass: the full width could not be opened here."
                           : "A prompt runs in one pass: a graph " + n + " tokens wide could not be opened here.",
               room + "; loads " + (narrow_wins ? std::string("at the full width") : n + " tokens wide") +
                   " failed on this machine, so the other width stands");
        return true;
    }
    // What each width leaves the experts: the room the narrower buffers give goes to the device cache
    // where one is armed, and to the experts the fitter places on the devices where none is.
    const uint64_t narrow_buffers = pl.at_width(narrow).device_compute_bytes;
    const uint64_t freed = pl.device_compute_bytes > narrow_buffers ? pl.device_compute_bytes - narrow_buffers : 0;
    uint64_t off_wide = 0, off_narrow = 0; // the experts not on the devices, which a prompt's pass gathers
    double dec_wide = 0.0, dec_narrow = 0.0;
    if (cache) {
        const uint64_t held = std::min<uint64_t>((uint64_t) p.config.moe.device_cache_mb << 20, m.expert_bytes);
        const uint64_t held_narrow = std::min<uint64_t>(held + freed, m.expert_bytes);
        off_wide = m.expert_bytes - held;
        off_narrow = m.expert_bytes - held_narrow;
        dec_wide = device_cache_decode_seconds(s, *dev, held);
        dec_narrow = device_cache_decode_seconds(s, *dev, held_narrow);
        if (dec_wide > 0.0 && dec_narrow > 0.0) {
            dec_wide += host_expert_flash_seconds(s, m.expert_bytes, expert_share_at(m, held));
            dec_narrow += host_expert_flash_seconds(s, m.expert_bytes, expert_share_at(m, held_narrow));
        }
    } else {
        off_wide = s.host_expert_bytes;
        off_narrow = off_wide > freed ? off_wide - freed : 0;
        const double wide_tps = fitter_decode_ceiling(s, off_wide), narrow_tps = fitter_decode_ceiling(s, off_narrow);
        dec_wide = wide_tps > 0.0 ? 1.0 / wide_tps + host_expert_flash_seconds(s, off_wide, 0.0) : 0.0;
        dec_narrow = narrow_tps > 0.0 ? 1.0 / narrow_tps + host_expert_flash_seconds(s, off_narrow, 0.0) : 0.0;
    }
    double token_bytes = 0.0;
    for (size_t g = 0; g < (size_t) WeightGroup::count; ++g)
        token_bytes += (double) m.group((WeightGroup) g).bytes_per_token;
    const double wide_rate = dev->wide_matmul_gibs > 0.0 ? dev->wide_matmul_gibs : dense_rates(s.hw, *dev).device;
    const double gather_wide = prompt_gather_seconds(*dev, off_wide),
                 gather_narrow = prompt_gather_seconds(*dev, off_narrow);
    if (dec_wide <= 0.0 || dec_narrow <= 0.0 || wide_rate <= 0.0 || token_bytes <= 0.0 ||
        (off_wide > 0 && gather_wide <= 0.0))
        return false; // a rate is unmeasured: nothing to price the two widths with
    const double prefill_token = token_bytes / (1024.0 * 1024.0 * 1024.0) / wide_rate;
    // The requests: this machine's own, weighed with the default mix. A prompt of l tokens takes
    // ceil(l / w) passes, about l / w + 1/2, so a request's passes at a width are one and, for its
    // long prompts, their tokens over the width less a half for each. The mean prompt would not do:
    // one 4.6k-token prompt among three short ones is a mean of 1181 tokens, which fits one pass.
    const RequestMix mix = request_mix(s, narrow);
    const double gen = mix.gen, prompt = mix.prompt;
    const double long_share = std::min(1.0, mix.long_prompt / (double) narrow); // of requests, at most
    const auto prefill_s = [&](uint32_t width, double gather) {
        const double passes = 1.0 + std::max(0.0, mix.long_prompt / (double) width - 0.5 * long_share);
        return prompt > 0.0 ? prompt * prefill_token + passes * gather : 0.0;
    };
    double pre_wide = prefill_s(full, gather_wide), pre_narrow = prefill_s(narrow, gather_narrow);
    // What the loads served here measured: a side's own figure where both were timed, and where one
    // was, both prices scaled by how far that side's ran from its measurement.
    const double wd = same ? mk.wide_decode_tps : 0.0, nd = same ? mk.narrow_decode_tps : 0.0;
    const double wp = same ? mk.wide_prefill_tps : 0.0, np = same ? mk.narrow_prefill_tps : 0.0;
    const auto correct = [](double & wide, double & narrow_side, double wide_measured, double narrow_measured) {
        if (wide <= 0.0 || narrow_side <= 0.0) return;
        if (wide_measured > 0.0 && narrow_measured > 0.0) {
            wide = wide_measured;
            narrow_side = narrow_measured;
        } else if (wide_measured > 0.0) {
            narrow_side *= wide_measured / wide;
            wide = wide_measured;
        } else if (narrow_measured > 0.0) {
            wide *= narrow_measured / narrow_side;
            narrow_side = narrow_measured;
        }
    };
    correct(dec_wide, dec_narrow, wd > 0.0 ? 1.0 / wd : 0.0, nd > 0.0 ? 1.0 / nd : 0.0);
    correct(pre_wide, pre_narrow, wp > 0.0 ? prompt / wp : 0.0, np > 0.0 ? prompt / np : 0.0);
    const double t_wide = gen * dec_wide + pre_wide, t_narrow = gen * dec_narrow + pre_narrow;
    // The margin guards the width in force: the one an earlier load ran as decided, else the full
    // width. The other takes over only past it, so the same figures weighed again keep the graph where
    // it is rather than swinging it on a few percent.
    const double win = 1.0 + (double) s.pol.min_backend_win;
    const bool narrow_held = same && mk.width_held == 0;
    const bool narrow_wins = narrow_held ? !(t_narrow > t_wide * win) : t_wide > t_narrow * win;
    const bool measured = wd > 0.0 && nd > 0.0;
    char nums[512];
    std::snprintf(nums, sizeof(nums),
                  "a request here is %.0f prompt tokens and %.0f decoded (%s); at the full width it decodes at %.2f "
                  "tok/s and prefills that prompt in %.2f s, at %u tokens at %.2f tok/s and %.2f s (%s), so %.2f s "
                  "a request against %.2f",
                  prompt, gen, mix.words().c_str(), 1.0 / dec_wide, pre_wide, narrow, 1.0 / dec_narrow, pre_narrow,
                  measured ? "both widths' decode measured on loads here"
                  : wd > 0.0 || nd > 0.0
                      ? "priced from the probed rates, corrected by the width measured on a load here"
                      : "priced from the probed rates",
                  t_wide, t_narrow);
    const std::string margin = u64s((uint64_t) (s.pol.min_backend_win * 100.0f + 0.5f)) + "%";
    // The load times the width it serves where that side has no decode of its own yet, and records the
    // requests it serves against the narrow width either way. One decision's rate a load: the device
    // tier's goes first.
    if (p.config.trial.device_tier < 0) {
        p.config.trial.width_narrow = (int) narrow;
        if ((narrow_wins ? nd : wd) <= 0.0) p.config.trial.width_side = narrow_wins ? 0 : 1;
    }
    const Source src = measured ? Source::Measured : Source::Derived;
    if (narrow_wins) {
        take_narrow();
        s.note("ubatch", n, src, n + " tokens per pass: on this machine's requests that is faster.",
               room + "; " + nums +
                   (narrow_held ? "; the narrow graph is in force, and the full width is not faster by the " + margin +
                                      " margin"
                                : ", past the " + margin + " margin") +
                   ". " + flag_spelling("ubatch") + " sets it by hand");
        return true;
    }
    s.note("ubatch", u64s((uint64_t) p.config.graph_width()), src,
           "A prompt runs in one pass: on this machine's requests the full width is as fast or faster.",
           room + "; " + nums +
               (narrow_held ? ", the full width past the " + margin + " margin over the narrow graph in force"
                            : "; the narrow graph is not faster by the " + margin + " margin") +
               ". " + flag_spelling("ubatch") + " sets it by hand");
    return true;
}

// Whether a device computes this graph's matmuls, from what the plan already holds: layers on a device
// of its own (where llama.cpp's op offload also hands that device the wide host-weight matmuls), a device
// expert cache (the experts computed on the device), or a prefill device the caller named. A device the
// plan itself arms for the prefill is decided later, and bound_prefill_width holds the graph then.
bool device_computes_graph(const Planning & s) {
    const bool own = !devices_share_host_memory(s.hw) && s.hw.device_local_memory() > 0;
    // n_gpu_layers is -1 where the fitter placed every layer (llama.cpp's "all"), so "a layer on a
    // device" is any count but zero. Read as > 0, a model that fills a 24 GB card whole was planned
    // as if no device computed it: its graph went as wide as the context, past the card's memory.
    const bool layers = own && s.placement.fitted && s.placement.n_gpu_layers != 0;
    const bool cache = s.p.config.moe.device_cache_mb > 0 || s.p.config.trial.device_tier == 1;
    return layers || cache || s.p.config.prefill.enabled();
}

// The widest pass a device is planned at (PlannerPolicy::ubatch_max_device), never under ubatch_min.
// The image floor does not lift it: a cap on an image's tokens would otherwise widen every prompt's
// pass on the device past what its expert matmul is known to run (#533).
uint32_t device_pass_bound(const PlannerPolicy & pol) {
    return pol.ubatch_max_device > 0 ? std::max(pol.ubatch_max_device, pol.ubatch_min) : 0;
}

// What a device's bound does to images, said where it is narrower than their cap: an image wider than
// the pass is refused with the error that names the settings, rather than every prompt going wider.
std::string image_past_bound(const RunConfig & c, uint32_t bound) {
    const uint32_t floor_w = (uint32_t) c.graph_width_floor();
    if (bound == 0 || floor_w <= bound) return std::string();
    return " An image may take up to " + u64s(floor_w) + " tokens (" + flag_spelling("image-max-tokens") +
           "), past this bound: such an image, if its tokens attend both ways, is refused with a message naming "
           "the settings, since the bound holds every prompt and an image is the rarer case.";
}

void choose_graph_width(Planning & s) {
    Plan & p = s.p;
    const bool device_in_pool = devices_share_host_memory(s.hw);
    auto buffers = [&](uint32_t w) {
        const Placement q = s.placement.at_width(w);
        return q.raw_host_compute_bytes + (device_in_pool ? q.device_compute_bytes : 0);
    };
    // Short of room: the fitter left a whole layer, and with it that layer's dense part, on the host
    // beside a device of its own. Where every layer is on the device, the wide graph costs the device
    // tier cache, not a dense part: that trade (a larger cache for decode against a narrower graph for
    // prefill) is left to the widest graph until it is priced. Measured on a 24 GB card where every
    // layer fit: narrowing to 4096 tokens grew the cache from 12.4 to 17.6 GB and decode by 7-12%, and
    // cost a 9756-token prompt's prefill 24-28% (1560-1668 to 1189-1201 tok/s). The one narrowing
    // there that is no trade, the whole model fitting the device, is asked of the fitter by
    // probe_and_plan, since only a placement can say it.
    const uint64_t device_memory = device_in_pool ? 0 : s.hw.device_local_memory();
    // No layer on the card at all is the shortest of all: the wide graph's buffers took the room the
    // first layer needed. Measured on a 24 GB card with 12 GiB left free, Qwen3.6-35B Q4_K_M at a 40960
    // context: the full width reserved 11.8 GiB of device buffers, the fitter placed 0 of 41 layers and
    // the decode ran on the host at 12.5 tok/s; at 10240 tokens it placed all 41 and decoded at 75-81.
    const bool device_short = device_memory > 0 && s.placement.fitted && s.placement.device_compute_per_token > 0 &&
                              s.placement.n_gpu_layers >= 0 && (uint32_t) s.placement.n_gpu_layers <= s.model.n_layer;
    // The narrowest graph: llama.cpp's own physical batch, or the most tokens one image takes where
    // its tokens must go in one pass (RunConfig::graph_width_floor).
    // Where a device computes the graph, the image floor is held to the device's bound: the bound wins.
    const uint32_t bound = device_computes_graph(s) ? device_pass_bound(s.pol) : 0;
    const uint32_t image_w = (uint32_t) p.config.graph_width_floor();
    const uint32_t narrowest = std::max(s.pol.ubatch_min, bound > 0 ? std::min(image_w, bound) : image_w);
    const uint32_t device_width = device_short ? device_share_width(s.placement, s.hw, s.pol, narrowest) : 0;
    auto device_buffers = [&](uint32_t w) { return s.placement.at_width(w).device_compute_bytes; };
    auto device_over = [&](uint32_t w) { return device_width > 0 && w > device_width; };
    auto over = [&](uint32_t w) {
        return buffers(w) > s.holdable * (double) s.pol.compute_share_max || (device_short && device_over(w));
    };
    const uint32_t probed = s.placement.compute_width;
    if (s.req.is_pinned("ubatch")) {
        s.note("ubatch", u64s((uint64_t) p.config.n_ubatch), Source::Operator,
               u64s((uint64_t) p.config.n_ubatch) + " tokens per pass, as you set it.",
               "set by the caller; the planner leaves it alone" +
                   (probed
                        ? ", and the compute buffers are counted at that width: " + u64s(mib(buffers(probed))) + " MiB"
                        : std::string()));
        return;
    }
    if (probed == 0 || s.placement.host_compute_per_token == 0) {
        s.note("ubatch", u64s((uint64_t) p.config.graph_width()), Source::Unprobed,
               u64s((uint64_t) p.config.graph_width()) +
                   " tokens per pass, the default: its memory cost was not measured.",
               "what a token of graph width costs in compute buffers was not measured here, so the "
               "width is not this plan's to choose; keeping the default");
        return;
    }
    // The widest candidate is the whole context where the width was left to the default. The probe
    // measured the buffers at the default width (RunConfig::graph_width), which a long context caps,
    // and at half of it, both at this context, so the per-token slope already carries what the
    // attention mask costs a token of width here: a wider graph at the same context is priced by
    // extending that line (Placement::at_width). Where a device's part of the buffers was not
    // measured, a wider graph is not priced, and the candidate stays the probed width.
    uint32_t n_ctx = (uint32_t) p.config.n_ctx;
    if (s.placement.n_ctx > 0 && s.placement.n_ctx < n_ctx) n_ctx = s.placement.n_ctx;
    const bool device_priced = s.placement.device_compute_bytes == 0 || s.placement.device_compute_per_token > 0;
    const bool extend = p.config.n_ubatch == 0 && device_priced && probed < n_ctx;
    const uint32_t full = extend ? n_ctx : probed;
    // A device that computes the graph is bounded by the widest pass its expert matmul is known to
    // run (PlannerPolicy::ubatch_max_device, `bound` above), whatever its memory allows.
    const std::string images = image_past_bound(p.config, bound);
    const uint32_t top = bound > 0 && full > bound ? bound : full;
    const uint32_t w = narrowed_width(top, narrowest, over);
    // The width taken: its buffers charged, and the session told it whenever it is not the width the
    // config opens with anyway (at the context the placement settles, which apply_placement adopts).
    RunConfig opened = p.config;
    opened.n_ctx = (int) n_ctx;
    const uint32_t default_width = (uint32_t) opened.graph_width();
    auto take = [&](uint32_t width) {
        s.placement = s.placement.at_width(width);
        p.placement = s.placement;
        if (width != default_width) p.config.n_ubatch = (int) width;
    };
    const std::string pct = u64s((uint64_t) (s.pol.compute_share_max * 100.0f + 0.5f)) + "%";
    const std::string share = pct + " of the " + u64s(mib(s.holdable)) + " MiB this process may hold";
    const bool device_bound = device_short && device_over(full);
    // What the widest graph is, said as it is, and how its buffers were priced when it is wider than
    // the probe measured.
    const std::string widest =
        full >= n_ctx ? "a graph as wide as the context (" + u64s(full) + " tokens" +
                            (full > probed ? ", its buffers priced by extending the " + u64s(mib(buffers(probed))) +
                                                 " MiB measured " + u64s(probed) +
                                                 " tokens wide at this context along the cost of a token of width "
                                                 "measured there)"
                                           : std::string(")"))
                      : "a graph " + u64s(full) +
                            " tokens wide (the width the placement was measured at; a wider one is not priced here)";
    if (bound > 0 && w == top && top < full) {
        // Narrowed by the bound: inside it, the memory takes the graph.
        const bool full_fits = !over(full);
        take(w);
        if (width_across_loads(s, w)) return;
        s.note("ubatch", u64s(w), Source::Policy,
               u64s(w) + " tokens per pass: the widest a device's expert matmul is known to run.",
               widest + (full_fits ? " would fit in memory" : " would not fit in memory either") +
                   ", but a device computes it, and past " + u64s(bound) +
                   " tokens a pass can abort in the backend's expert matmul (#533); a longer prompt takes more "
                   "passes." +
                   images + " " + flag_spelling("ubatch") + " sets it by hand");
        return;
    }
    if (w == full) {
        take(full);
        if (width_across_loads(s, full)) return;
        s.note("ubatch", u64s(full), Source::Derived,
               (full < n_ctx ? "A prompt up to " + u64s(full) + " tokens runs in one pass: that takes only "
                             : std::string("A prompt runs in one pass: that takes only ")) +
                   size_brief(buffers(full)) + ".",
               widest + " reserves " + u64s(mib(buffers(full))) + " MiB of compute buffers, inside " + share +
                   ": a prompt that wide is computed in one pass." + images);
        return;
    }
    const uint64_t before = device_bound ? device_buffers(full) : buffers(full);
    const uint64_t after = device_bound ? device_buffers(w) : buffers(w);
    const std::string where = device_bound ? pct + " of the " + u64s(mib(device_memory)) +
                                                 " MiB of device memory, where the fitter left layers on the host"
                                           : share;
    // Held at the floor an image sets, past the share: said, since the share alone would go narrower.
    const std::string image_floor =
        narrowest > s.pol.ubatch_min && w == narrowest && over(w)
            ? " It stays at least " + u64s(narrowest) + " tokens wide: an image takes up to that many, in one pass."
            : std::string();
    take(w);
    s.note("ubatch", u64s(w), Source::Policy,
           u64s(w) + " tokens per pass; wider would take " + size_brief(before) +
               (device_bound ? " of the device's memory." : " from the cache."),
           widest + " would reserve " + u64s(mib(before)) + " MiB of compute buffers" +
               (device_bound ? " on the devices" : "") + ", more than " + where + "; at " + u64s(w) + " tokens it is " +
               u64s(mib(after)) +
               " MiB. A prompt takes more passes and the expert cache keeps the rest: a prompt is paid once and a "
               "token every time." +
               image_floor + images + " " + flag_spelling("ubatch") + " sets it by hand");
}

// A prefill device the plan armed computes every prompt's pass, as wide as the graph: held to the
// device's bound like any device that computes the graph (#533). The width rule ran before the device
// was chosen, so a graph it left wider is narrowed here; narrower only gives memory back.
// The width that rule holds the graph to, or 0 where it leaves the graph as it is. Asked before the
// device is armed too (fit_prefill_device), so the device is weighed with the graph it will run.
uint32_t prefill_width_bound(const Planning & s) {
    const uint32_t bound = device_pass_bound(s.pol);
    return s.req.is_pinned("ubatch") || bound == 0 || (uint32_t) s.p.config.graph_width() <= bound ? 0 : bound;
}

void bound_prefill_width(Planning & s) {
    Plan & p = s.p;
    const uint32_t bound = prefill_width_bound(s);
    if (bound == 0) return;
    const uint32_t was = (uint32_t) p.config.graph_width();
    s.placement = s.placement.at_width(bound);
    p.placement = s.placement;
    p.config.n_ubatch = (int) bound;
    for (auto it = p.decisions.begin(); it != p.decisions.end();)
        it = it->knob == "ubatch" ? p.decisions.erase(it) : it + 1;
    s.note("ubatch", u64s(bound), Source::Policy,
           u64s(bound) + " tokens per pass: the widest a device's expert matmul is known to run.",
           "the plan chose " + u64s(was) + " tokens wide, and then armed " + p.config.prefill.device +
               " for the prefill, which computes every pass; past " + u64s(bound) +
               " tokens a pass can abort in the backend's expert matmul (#533), so a longer prompt takes more "
               "passes." +
               image_past_bound(p.config, bound) + " " + flag_spelling("ubatch") + " sets it by hand");
}

// The bytes that would have to be resident on the host if nothing were streamed, and the regime
// they put the run in.
//
// The memory the context checkpoints will take: one copy of the context's fixed part per checkpoint
// kept. Only a chat run makes them (the session keeps them at the end of a templated prompt), and a
// context that is all attention has no fixed part, so both cost nothing.
uint64_t checkpoint_reserve(const RunConfig & c, const Placement & placement) {
    if (!c.chatml || c.ctx_checkpoints <= 0) return 0;
    return placement.context_fixed_bytes * (uint64_t) c.ctx_checkpoints;
}

// The context and the compute buffers a placement leaves on the host (residency_regime says what is
// counted and why). A function of the placement alone, so a graph narrowed later is charged again.
uint64_t host_fixed_of(const Placement & placement) {
    uint64_t host_fixed = placement.raw_host_context_bytes + placement.raw_host_compute_bytes;
    if (placement.shared_memory_placement) host_fixed += placement.device_context_bytes;
    if (placement.fitted) host_fixed = std::min(host_fixed, placement.host_resident_bytes);
    return host_fixed;
}

// The embedding tables inside the host's dense set that the run may leave in the mapping rather than
// hold. The graph gathers a table by row, and the run leaves in the mapping any tensor it is handed by
// name, under every dense policy (MoeStreamConfig::dense_mapped), so a table is never a set the run
// must hold whole: whether it is held is the ledger's call (settle_mapped_tables), made later. Counted
// as the host set counted them: the file's tables without a placement, the placement's own figure
// with one (Placement::host_table_bytes). A caller who pinned the list decides instead: the tables
// are left mapped only where the list names every one of them.
uint64_t mappable_host_tables(const Planning & s) {
    const ModelProfile & m = s.model;
    if (m.embedding_tensors.empty()) return 0;
    if (s.req.is_pinned("dense-mapped")) {
        const std::vector<std::string> & named = s.p.config.moe.dense_mapped;
        for (const std::string & t : m.embedding_tensors)
            if (std::find(named.begin(), named.end(), t) == named.end()) return 0;
    }
    return s.placement.fitted ? s.placement.host_table_bytes : m.group(WeightGroup::Embedding).bytes;
}

// What the host holds against what it may (s.host_all, s.air and the rest), and the regime that puts
// the run in. Read for a model with no experts too, where the only question it answers is whether the
// whole model is held in memory.
Regime residency_regime(Planning & s) {
    const Placement & placement = s.placement;
    // With a placement, what the fitter left here; without one, the whole file.
    //
    // On a machine where every device's memory IS the host's, what the fitter put "on a device" is
    // in this same pool and has to be charged to it. The fitter's own arithmetic treats the two as
    // separate - it was told an integrated accelerator had 15 GB free on an 11 GB machine - and a
    // budget that inherits that mistake will size a cache for memory that is already spoken for.
    // Where devices have memory of their own the term is zero and nothing changes.
    s.devices_share_host_memory = devices_share_host_memory(s.hw);
    s.device_on_host = s.devices_share_host_memory ? placement.device_bytes : 0;

    // The context and the compute buffers, as their own line. A fitted placement already carries
    // them inside its host figure; an unfitted one never charged them at all, which on a machine
    // whose devices share the host's memory - where the fitter's answer is always set aside - meant
    // they were charged nowhere. There the context is counted wherever the fitter put it, because
    // a context is the same size on either side of a boundary this machine does not have; the
    // compute buffer is the host's own, since what a device reserves to compute is not what the
    // cores will.
    const uint64_t host_fixed = host_fixed_of(placement);
    // Context checkpoints are host memory the fitter never saw, and the run fills it as the chat
    // goes: charged up front, at the count the run may keep, so the cache is not sized into it.
    s.checkpoint_bytes = checkpoint_reserve(s.p.config, placement);
    s.host_fixed = host_fixed;
    // The projector goes where llama.cpp's library puts it, by the session's rule for the devices: a
    // layer placed on one means they are wanted, none means they are not. On a device with memory
    // of its own it is that device's cost: the fitter reserved its room there before placing layers
    // (Placement::device_projector_bytes), and choose_device_tier charges what it did not. Here it
    // is the host's, which on a machine whose devices share the host's memory is both.
    {
        const RunConfig & c = s.p.config;
        const bool devices_wanted =
            c.device_use == DeviceUse::All || (c.device_use == DeviceUse::Auto && placement.n_gpu_layers != 0);
        s.projector_on_device = s.model.projector_bytes > 0 && devices_wanted;
        s.projector_host = s.projector_on_device && !s.devices_share_host_memory ? 0 : s.model.projector_bytes;
    }
    const uint64_t unfitted_fixed = placement.fitted ? 0 : host_fixed + s.checkpoint_bytes;

    s.host_all = placement.fitted
                     ? placement.host_resident_bytes + s.host_expert_bytes + s.device_on_host + s.checkpoint_bytes
                     : s.model.file_bytes + unfitted_fixed;
    s.host_all += s.projector_host;
    s.host_dense = placement.fitted ? placement.host_resident_bytes + s.device_on_host + s.checkpoint_bytes
                                    : s.model.dense_bytes + unfitted_fixed;
    s.host_dense += s.projector_host;
    // Both questions are asked of what the run has to hold, so neither counts a table it may leave
    // mapped. Weighing the dense set with one called a model whose per-layer table alone is larger
    // than the machine "not even the dense weights fit" while its dense set held in a quarter of the
    // memory (#664). Weighing the whole model with one streams a model that is held whole beside the
    // table, which on two 16 GB cards also gives up the copy of the experts a device reads them from
    // (choose_expert_staging) and with it the decode that load measured. Held whole, the run reads
    // the rows a token needs from the mapping; a table it reads whole is one llama.cpp keeps small
    // enough to read at load, which the margin below the budget is there for.
    s.host_tables = mappable_host_tables(s);
    Regime regime = s.held_all() <= s.usable     ? Regime::Fits
                    : s.held_dense() <= s.usable ? Regime::ExpertsStream
                                                 : Regime::DenseOversized;

    // Fitting is not the same as being left alone. Where the machine can take pages back, residency
    // has to clear a second bar: room beyond the model itself, so its weights are not reclaimed and
    // refaulted a page at a time under whatever else the machine is doing. The bar is a policy ratio
    // rather than a measurement, and the reason it can be this crude is that the error is asymmetric
    // by two orders of magnitude: streaming a model that would have fitted costs some reads, while
    // residency on a model that does not is 0.1 tok/s against 5.0. Until the headroom probe exists,
    // the cheap side of that asymmetry is the right default, and a caller who disagrees drops --auto.
    s.air = s.usable > s.held_all() ? s.usable - s.held_all() : 0;
    // The pass that plans the model as held whole (whole_room) is asked what that plan is, not
    // whether to make it: the plan that streams was made first and is what it is weighed against.
    if (s.whole_room > 0) return Regime::Fits;
    const uint64_t air_needed = (uint64_t) ((double) s.held_all() * (double) s.pol.fits_air_ratio);
    s.air_short = regime == Regime::Fits && reclaim_is_cheap(s.hw.anon_overflow) && s.air < air_needed;
    if (s.air_short) regime = s.held_dense() <= s.usable ? Regime::ExpertsStream : Regime::DenseOversized;
    return regime;
}

void classify_regime(Planning & s) {
    s.p.regime = residency_regime(s);
}

// What the first stage placed, applied, and said - or why nothing was.
void apply_placement(Planning & s) {
    Plan & p = s.p;
    const Placement & placement = s.placement;
    if (!placement.fitted) {
        s.note("placement", "none", placement.shared_memory_placement ? Source::Derived : Source::Unprobed,
               placement.shared_memory_placement ? "The fitter's placement is set aside: the device shares this memory."
                                                 : "Every layer stays on the CPU: no device placement was tried.",
               placement.shared_memory_placement
                   ? "the capacity fitter ran and its answer was set aside: the device it placed layers on "
                     "reads this host's own memory, so 'what fits where' has one pool and no answer. "
                     "Everything is on the host, which is what it already was, unless the model is held in "
                     "memory whole and a measurement moves it (shared-device)"
                   : "the capacity fitter was not run, so no layer is placed on a device: everything is on "
                     "the host");
        return;
    }
    // The fitter leaves the count at its default, every layer, where the model fits the devices whole.
    const std::string layers =
        placement.n_gpu_layers < 0 ? "every layer" : u64s((uint64_t) std::max(0, placement.n_gpu_layers)) + " layers";
    s.note("placement", layers + " on devices", Source::Measured,
           layers + " on a device; " + u64s(s.host_layers) + " of " + u64s(s.n_moe_layer) +
               " layers keep their experts on the CPU.",
           placement.outcome + "; " + u64s(s.host_layers) + " of " + u64s(s.n_moe_layer) +
               " MoE layers keep their experts on the host, host-resident " + u64s(mib(placement.host_resident_bytes)) +
               " MiB, on devices " + u64s(mib(placement.device_bytes)) + " MiB, context " + u64s(placement.n_ctx) +
               " (fitter's host breakdown: model " + u64s(mib(placement.raw_host_model_bytes)) + ", context " +
               u64s(mib(placement.raw_host_context_bytes)) + ", compute " +
               u64s(mib(placement.raw_host_compute_bytes)) + " MiB; file " + u64s(mib(s.model.file_bytes)) +
               ", experts " + u64s(mib(s.model.expert_bytes)) + ")");
    // The context is NOT the fitter's decision, and this line used to claim it was. Upstream
    // modifies the context if and ONLY if it is handed a zero, and the placement adapter always
    // pins it to the caller's - so what comes back is the number that went in. Reporting that as
    // `[measured]` credited a measurement nobody made, on the one knob whose default has been
    // observed to decide whether a model streams at all: a 26B model was refused for a cache
    // 184 MiB short while its context reserved 440 MiB, and halving it was the difference
    // between declining and running. Nothing has priced this number - not the fitter, which
    // cannot, and not this planner, which believed the fitter had.
    if (placement.n_ctx && !s.req.is_pinned("ctx-size")) {
        p.config.n_ctx = (int) placement.n_ctx;
        s.note("ctx-size", u64s(placement.n_ctx), Source::Policy,
               u64s(placement.n_ctx) + " tokens of context, as given: not weighed against the cache.",
               "the value this run was given, not a choice: the capacity fitter only picks a context when it "
               "is handed a zero and this planner always pins it, so nothing has weighed this number against "
               "what the memory it reserves would buy as expert cache");
    }
    p.config.n_gpu_layers = placement.n_gpu_layers;
    p.config.buft_overrides = placement.override_patterns;
    p.config.device_split.clear();
    for (const auto & d : placement.device_split)
        p.config.device_split.emplace_back(d.first, (int) d.second);

    // The two stages answer different questions, and where they disagree the disagreement is the
    // finding. The fitter solves CAPACITY - what fits where - and its answer is applied because it
    // is the only route to a device this engine has, and because a model that only fits with
    // layers on a device must keep them there. This planner owns the tier below and would not have
    // chosen the move: dense and expert halves alternate per layer, so the placement crosses the
    // host/device boundary twice per layer, and what a crossing costs is unmeasured. Saying so is
    // what stops a slow run from being read as the engine being slow.
    bool any_split_priced = false;
    for (const ComputeDevice & d : s.hw.devices)
        if (!d.is_cpu && d.split_seconds > 0.0) any_split_priced = true;
    if (placement.n_gpu_layers != 0 && !any_split_priced)
        s.note("placement-cost", "unpriced", Source::Unprobed,
               "Layers go to the device, but the cost of each hop to it is unknown.",
               "the capacity fitter placed layers on a device and that placement is applied, but this "
               "plan's cost model did not choose it and could not have: the move crosses the "
               "host/device boundary twice per layer and the price of a crossing is unmeasured here. "
               "If this run is slower than the same model with the device left out, that is the reason "
               "to check first");
}

void note_regime(Planning & s) {
    std::string brief;
    switch (s.p.regime) {
    case Regime::Fits:
        brief = "The whole model (" + size_brief(s.held_all()) + ") fits in the " + size_brief(s.usable) + " usable.";
        break;
    case Regime::ExpertsStream:
        brief = "Experts stream from storage: the model needs " + size_brief(s.held_all()) + ", " +
                size_brief(s.usable) + " is usable.";
        break;
    case Regime::DenseOversized:
        brief = "Even the dense weights (" + size_brief(s.held_dense()) + ") exceed the " + size_brief(s.usable) +
                " usable.";
        break;
    case Regime::NotMoe:
    case Regime::Unknown:
        brief = "Nothing to stream.";
        break;
    }
    const uint64_t tables = s.host_dense - s.held_dense();
    const std::string left_out = tables ? ", of which " + u64s(mib(tables)) +
                                              " MiB are embedding tables the run may leave mapped, weighed without"
                                        : "";
    s.note("regime", regime_name(s.p.regime), Source::Derived, brief,
           "on the host " + u64s(mib(s.host_all)) + " MiB (dense " + u64s(mib(s.host_dense)) + left_out + ") against " +
               u64s(mib(s.usable)) + " MiB usable, which is the " + u64s(mib(s.holdable)) +
               " MiB this process may hold (" + headroom_name(s.hw.holdable_from) + ") less a " + u64s(mib(s.margin)) +
               " MiB margin (" + overflow_name(s.hw.anon_overflow) + ")");

    if (s.air_short)
        s.note("residency", "not verified", Source::Derived,
               "Experts stream anyway: the model would fit with only " + size_brief(s.air) + " to spare.",
               "the host set fits with " + u64s(mib(s.air)) + " MiB to spare against the " + u64s(mib(s.held_all())) +
                   " MiB it would hold, and a reclaim is cheap here (" + overflow_name(s.hw.anon_overflow) +
                   "): a model that fits only just is reclaimed from underneath and refaults its weights one page "
                   "at a time, so the experts stream instead. Run without the planner to keep plain residency");
}

// The two outcomes of the first stage that leave the streamer nothing to do. True when the plan
// has declined and is complete.
bool declines_residency(Planning & s) {
    if (s.placement.fitted && s.host_layers == 0) {
        s.p.regime = Regime::Fits;
        s.decline("the capacity fitter placed every layer's experts on a device; nothing is left on the host for "
                  "the streamer to serve");
        s.note("moe-stream", "off", Source::Measured, "Not streamed: every expert already sits on a device.",
               "no expert tensor remains on the host");
        return true;
    }
    if (s.p.regime == Regime::Fits) {
        s.decline("what is left on the host fits in this machine's residency budget; streaming it from flash "
                  "would only add reads that resident weights do not need");
        s.note("moe-stream", "off", Source::Derived, "Not streamed: the whole model fits in memory.",
               s.placement.fitted ? "the host residual fits: llama.cpp's placement is the whole plan"
                                  : "the model fits: hand the placement to llama.cpp instead");
        return true;
    }
    return false;
}

// ── a model held in memory, on a device that reads that same memory ─────────────
// The capacity fitter's answer is set aside on such a machine because its arithmetic counts one pool
// twice (apply_placement). That settles capacity, and capacity is not the question here: the model is
// in this pool either way, so running it on the device frees nothing and costs nothing. What is left
// is which engine serves this machine's requests faster, and it is priced: a request's decoded tokens
// at the rate each processor applies a real layer of this model one token wide, its prompt at the
// rate each applies it as wide as a prompt, both probed here with the host's results. The cores are
// the side in force; the device takes the model where the request is shorter there by the margin.
// The load times the decode of the side it serves (MeasuredKnobs::shared_device_tps), and once both
// sides have one their ratio replaces the probe's.
//
// A probe of one layer does not always predict the whole model's decode: on a unified-memory laptop
// the one-token probe read the cores 60% ahead (109 GiB/s against 67) while the model itself decoded
// 54 tok/s on the GPU against 40 on the cores; a laptop iGPU through Vulkan decoded a 1.7B model 28%
// faster than its cores, and one through OpenCL 37% slower (#602, #375). Timing both sides as loads of
// their own caught those, and cost every first load the model's reopening, so no load is opened to
// try the device: a machine like that laptop keeps the cores unless its prompts decide it.
//
// The whole model or none of it: a partial placement crosses the boundary twice per split layer, the
// shape measured slower on a phone. One device, named, so llama.cpp does not divide the layers between
// two backends that drive the same silicon. A candidate has to have computed the host's results in
// the probes, one token wide and as wide as a prompt (ComputeDevice::identity_ok, wide_identity_ok),
// since the prompt runs there too, run its expert matmul in the file's own format, take the mapped
// weights without a repack (hard rule 2), and have room for the compute buffer it reserves beside the
// model in this pool. Whether its weights then stay the file's mapping or are copied into the device's
// own buffers is the backend's (host_ptr_verified); either way they occupy this pool once, and the plan
// says which.
//
// The probes are a filter, not the guarantee. Each computes a part of a model in isolation (a
// synthetic matmul, one layer's matrices where the model has experts, nothing of the model at all
// where it has none), and a device can pass every one and still not compute a given model: on a 16 GB
// Apple-silicon laptop SmolVLM-256M Q8_0 wrote one repeated character with every layer on the GPU and
// sane text on the cores, where two other dense models were correct on the same GPU and the probes
// passed for all three. What a model answers on a device is only known by decoding the model there, so
// the load that moves it compares the two before it serves (KnobTrialConfig::answer_device) and the
// record keeps the word (MeasuredKnobs::device_answers): a device whose answers for this model
// differed here is no candidate, and one that matched is not compared again.
namespace {
// What a device applies this model's weights at, one token wide: on one real layer of the model where
// that was probed, its memory bandwidth otherwise.
double layer_rate_of(const Planning & s, const ComputeDevice & d) {
    const double r = s.model.is_moe ? d.expert_gibs : d.dense_gibs;
    return r > 0.0 ? r : d.memory_bandwidth_gibs;
}

struct SharedDeviceCandidate {
    const ComputeDevice * dev = nullptr;
    uint64_t compute = 0; // the compute buffer it reserves, from the fitter's projection
    std::string refused;  // why each device that is not one is not, "; "-separated
    std::string differed; // the ones the record refused: their answers for this model differed, ", "-separated
};

// The device a whole model held in `resident` bytes of this pool could move to, or none and why.
SharedDeviceCandidate shared_device_candidate(const Planning & s, uint64_t resident) {
    SharedDeviceCandidate c;
    const bool cheap = reclaim_is_cheap(s.hw.anon_overflow);
    // The room a move has to leave beside the model where this machine takes pages back cheaply
    // (PlannerPolicy::fits_air_ratio), measured against what the move is weighed against. Where part of
    // the model could stream instead, holding it whole is itself the choice that bar guards, and the
    // move has to clear it for the whole model as residency did. Where nothing could stream, the cores
    // hold the whole model on every plan, so the bar applies to what the move adds to this pool: its
    // compute buffer, and the weights as well where the backend copies them into buffers of its own
    // (they are then this process's memory rather than the file's pages the system reads back). On a
    // 16 GB laptop a 2B dense model was kept on the cores because 1638 MiB left beside it fell short of
    // the 2670 MiB it holds, which the cores held all the same.
    const bool could_stream = s.model.is_moe;
    const auto air_needed = [&](const ComputeDevice & d, uint64_t compute) {
        const bool in_place = is_yes(d.host_ptr_verified) && is_yes(d.host_ptr_buffers);
        const uint64_t added = compute + (in_place ? 0 : resident);
        return (uint64_t) ((double) (could_stream ? resident : added) * (double) s.pol.fits_air_ratio);
    };
    // Among several, the one that applies this model's weights fastest on one real layer at decode
    // width: a ranking only, since the timed decode is what decides against the cores.
    const auto layer_rate = [&](const ComputeDevice & d) { return layer_rate_of(s, d); };
    for (const ComputeDevice & d : s.hw.devices) {
        if (d.is_cpu) continue;
        std::string why;
        const uint64_t compute = s.placement.device_compute_of(d.name);
        if (d.has_own_memory())
            continue; // the capacity fitter's device: not this question
        else if (s.model.knobs.device_answer(d.name) < 0) {
            why = "its answers for this model differed from the cores' on this machine";
            c.differed += (c.differed.empty() ? "" : ", ") + d.name;
        } else if (d.usable == Tri::No)
            why = "it did not survive being probed";
        else if (d.identity_ok != Tri::Yes)
            why = "it has not been shown to compute the host's results on this model";
        else if (d.wide_identity_ok != Tri::Yes)
            // The whole model there runs every prompt there too, and a backend's batched kernel is not its
            // one-token path: on a phone a GPU matched the host one token wide and wrote garbage on a prompt.
            why = "it has not been shown to compute the host's results as wide as a prompt";
        else if (s.model.is_moe && d.runs_expert_op != Tri::Yes)
            why = "it does not run this model's expert matmul in the file's own format";
        else if (d.needs_repack == Tri::Yes)
            why = "it needs the weights repacked into a layout of its own, and the model loads as mapped";
        else if (compute == 0)
            why = "the compute buffer it would reserve was never projected";
        else if (s.whole_room > 0 && resident + compute > s.whole_room)
            // Weighed against the plan that streams: on the device the whole model is this process's
            // memory, as that plan's cache and dense set are, so it may take what that plan would.
            why = "the " + u64s(mib(resident)) + " MiB the model holds whole and its compute buffer of " +
                  u64s(mib(compute)) + " MiB are past the " + u64s(mib(s.whole_room)) +
                  " MiB the plan that streams it may hold of this memory";
        else if (s.whole_room > 0)
            ; // inside what the streaming plan may hold: the margin is already off that ceiling
        else if (resident + compute > s.usable)
            why = "its compute buffer of " + u64s(mib(compute)) + " MiB beside the " + u64s(mib(resident)) +
                  " MiB the model holds is past the " + u64s(mib(s.usable)) + " MiB usable";
        else if (cheap && s.usable - resident - compute < air_needed(d, compute))
            why = "its compute buffer of " + u64s(mib(compute)) + " MiB beside the " + u64s(mib(resident)) +
                  " MiB the model holds leaves " + u64s(mib(s.usable - resident - compute)) + " of the " +
                  u64s(mib(s.usable)) + " MiB usable, short of the " + u64s(mib(air_needed(d, compute))) +
                  " MiB this machine's reclaim asks to be left beside " +
                  (could_stream ? std::string("a model held whole") : std::string("what the move adds"));
        if (!why.empty()) {
            c.refused += (c.refused.empty() ? "" : "; ") + d.name + ": " + why;
            continue;
        }
        if (!c.dev || layer_rate(d) > layer_rate(*c.dev)) {
            c.dev = &d;
            c.compute = compute;
        }
    }
    return c;
}
} // namespace

void choose_shared_device(Planning & s, uint64_t resident) {
    if (s.placement.fitted || !s.devices_share_host_memory) return;
    for (const char * knob : {"gpu-layers", "device-split", "devices", "device-list", "tensor-overrides"})
        if (s.req.is_pinned(knob)) return;
    // One decision is timed a load: another one this plan left pending goes first.
    const KnobTrialConfig & pending = s.p.config.trial;
    if (pending.device_tier >= 0 || pending.width_side >= 0) return;
    const char * knob = "shared-device";
    const SharedDeviceCandidate cand = shared_device_candidate(s, resident);
    if (!cand.dev) {
        // A device the record refused is this machine's own measurement of this model, and the line
        // says so: it is not a device that could not, it is one that did and answered differently.
        if (!cand.differed.empty())
            s.note(knob, "cpu", Source::Measured,
                   "The model runs on the CPU: " + cand.differed + " did not give the CPU's answers for it here.",
                   "a device that reads this memory could run the whole model at no cost in capacity, but " +
                       cand.refused);
        else if (!cand.refused.empty())
            s.note(knob, "cpu", Source::Derived, "The model runs on the CPU: no device that shares this memory can.",
                   "a device that reads this memory could run the whole model at no cost in capacity, but " +
                       cand.refused);
        return;
    }
    const ComputeDevice & dev = *cand.dev;
    RunConfig & c = s.p.config;
    KnobTrialConfig & trial = c.trial;
    const MeasuredKnobs & mk = s.model.knobs;
    const double on = mk.shared_device_tps, off = mk.no_shared_device_tps;
    const double m = (double) s.pol.min_backend_win;
    const std::string margin = u64s((uint64_t) (m * 100.0 + 0.5)) + "%";
    const std::string held = is_yes(dev.host_ptr_verified) && is_yes(dev.host_ptr_buffers)
                                 ? "its weights stay the file's mapping, which the device reads in place"
                                 : "its weights are copied into the device's own buffers, which are this same "
                                   "memory, and the mapping's pages are left for the system to reclaim";
    const std::string place = "every layer could run on " + dev.name +
                              " at no cost in capacity, since it reads this memory: " + held + ", and its compute " +
                              "buffer, " + u64s(mib(cand.compute)) + " MiB, fits beside the model";
    char measured[128];
    std::snprintf(measured, sizeof(measured), "%.2f tok/s on %s against %.2f on the CPU", on, dev.name.c_str(), off);

    const auto on_device = [&](Source src, const std::string & brief, const std::string & reason) {
        const int layers = (int) s.model.n_layer + 1; // every block and the output head
        c.n_gpu_layers = layers;
        c.buft_overrides.clear();
        c.device_split.assign(1, {dev.name, layers});
        // The probes let the device through; what this model answers there is compared by the load
        // that moves it, once: a device the record already confirmed for it is not asked again.
        const bool confirmed = mk.device_answer(dev.name) > 0;
        if (!confirmed) trial.answer_device = dev.name;
        s.note(knob, dev.name, src, brief,
               place + "; " + reason +
                   (confirmed ? "; its answers for this model matched the cores' on this machine"
                              : "; the load compares its answers for this model with the cores' before it serves, "
                                "and reopens on the cores where they differ"));
    };
    const auto on_cores = [&](Source src, const std::string & brief, const std::string & reason) {
        s.note(knob, "cpu", src, brief, place + "; " + reason);
    };

    if (lost(mk.shared_device_unviable)) {
        on_cores(Source::Measured, "The model runs on the CPU: loads with it on " + dev.name + " failed here.",
                 "loads of this model with every layer on " + dev.name + " failed on this machine");
        return;
    }
    if (lost(mk.no_shared_device_unviable)) {
        on_device(Source::Measured, "Every layer runs on " + dev.name + ": loads on the CPU failed here.",
                  "loads of this model on the cores failed on this machine");
        return;
    }
    // Until a load has timed both sides on this model here, the load decides, on the model itself. The
    // probes that let the device through each compute a part of a model, and they do not say which side
    // decodes the whole of it faster: on one unified-memory laptop they read the cores 60% ahead of a
    // GPU that decoded the model 35% faster, and on a laptop with integrated graphics 74% ahead of one
    // that decoded it at more than twice their rate. The load that moves the model compares its answers
    // with the cores' anyway, which decodes it on both; it times the two as it does, keeps the side a
    // request is shorter on (KnobTrialConfig::answer_prompt_per_token), and records both rates, so
    // every plan after this one reads a measurement. The device is the side opened because the
    // comparison needs the placement open; where the cores win the model is reopened on them, once.
    const bool both = on > 0.0 && off > 0.0;
    if (!both) {
        const RequestMix mix = request_mix(s, 0);
        const double per_token = mix.gen > 0.0 ? mix.prompt / mix.gen : 0.0;
        char weighed[200];
        std::snprintf(weighed, sizeof(weighed),
                      "a request here is %.0f prompt tokens and %.0f decoded (%s), so a prompt-wide graph's time "
                      "weighs %.1f tokens a decoded one",
                      mix.prompt, mix.gen, mix.words().c_str(), per_token);
        on_device(Source::Derived,
                  "Every layer opens on " + dev.name + ", and the load keeps whichever of it and the CPU is faster.",
                  "no load has timed this model on both sides here, and no probe of a part of a model says which "
                  "decodes the whole faster: the load times it one token at a time and as wide as a prompt on " +
                      dev.name +
                      " and on the cores, keeps the side a request is shorter on (the cores where they "
                      "are faster by the " +
                      margin + " margin, reopening on them) and records both rates; " + weighed);
        trial.answer_device = dev.name;
        trial.answer_prompt_per_token =
            (float) std::min(per_token, (double) KnobTrialConfig::answer_prompt_per_token_max);
        trial.shared_device_side = 1;
        return;
    }
    // A request's time on each side, in the same unit: its decoded tokens at the rate each processor
    // applies a real layer of this model one token wide, and its prompt at the rate each applies it as
    // wide as a prompt, both probed here with the host's results (a token reads the same bytes on
    // either, so the bytes cancel). The decode's ratio is the loads' own once both sides were timed.
    const double decode_host = s.model.is_moe ? s.hw.host_expert_rate() : s.hw.host_dense_rate();
    if (decode_host <= 0.0) {
        on_cores(Source::Unprobed, "The model runs on the CPU: its rate on this model was not measured.",
                 "the rate the cores apply this model's weights at was not probed, so a prompt's share of a "
                 "request has no unit to be weighed in, and nothing moves");
        return;
    }
    const double decode_dev = decode_host * on / off;
    const bool wide_probed = s.hw.host_wide_matmul_gibs > 0.0 && dev.wide_matmul_gibs > 0.0;
    const RequestMix mix = request_mix(s, 0);
    const double gen = mix.gen, prompt = wide_probed ? mix.prompt : 0.0;
    const double t_cores = gen / decode_host + (prompt > 0.0 ? prompt / s.hw.host_wide_matmul_gibs : 0.0);
    const double t_dev = gen / decode_dev + (prompt > 0.0 ? prompt / dev.wide_matmul_gibs : 0.0);
    char nums[400];
    std::snprintf(nums, sizeof(nums),
                  "a request here is %.0f prompt tokens and %.0f decoded (%s); %s decodes at %.2fx the cores' rate "
                  "(%s) and computes a prompt-wide layer at %.2fx, so a request takes %.2fx the time it takes on "
                  "the cores",
                  prompt, gen, mix.words().c_str(), dev.name.c_str(), decode_dev / decode_host, measured,
                  wide_probed ? dev.wide_matmul_gibs / s.hw.host_wide_matmul_gibs : 0.0, t_dev / t_cores);
    // The rule the first load decided by (weigh_answer_sides), so a plan made on its measurements does
    // not undo it: the device keeps the model unless the cores serve a request faster by the margin.
    const bool cores_win = t_dev > t_cores * (1.0 + m);
    if (cores_win)
        on_cores(Source::Measured, "The model runs on the CPU, which serves this machine's requests faster.",
                 std::string(nums) + ", past the " + margin + " margin");
    else
        on_device(Source::Measured, "Every layer runs on " + dev.name + ": the CPU would not serve a request faster.",
                  std::string(nums) + ", which the cores do not beat by the " + margin + " margin");
}

// The same question for a model with no experts. Nothing streams, so the cores hold the whole model
// whatever the regime says of the room beside it, and the question is asked on every regime: the
// candidate's own room test (shared_device_candidate) decides, and the plan says why where it refuses.
// Asked only where the model is held well, a 4B dense model on a 16 GB laptop with 6 GB available ran
// on the cores with no word about the device at all.
void choose_shared_device_dense(Planning & s) {
    size_budget(s);
    residency_regime(s); // the books the candidate reads: what the model holds here, the projector's side
    choose_shared_device(s, s.held_all());
}

// ── the accelerator axis: what may leave the host BEYOND the fitter's placement ─
// Read the `placement` decision first: layers the fitter put on a device are already there,
// computed there, and none of this touches them. What follows is only about the residue the fitter
// left on the host - the experts this engine streams - and whether anything more can be done with
// them than running them on the CPU cores.
//
// Two facts settle this, and neither is a vendor name. A device whose memory IS the host's has no
// bandwidth of its own to win, and batch-1 decode is a chain of GEMVs that reads every weight once
// for a single multiply-accumulate: it is bandwidth-bound, so on such a device an offload moves the
// work without moving the bottleneck. A device with memory of its own wins in proportion to that
// memory's bandwidth, which is a number this profile does not yet carry.
//
// The streamed experts are separately constrained, and absolutely: the streamer serves a tensor by
// rebinding `data` onto the file's native layout, so a weight it serves must live in a host buffer
// that is not repacked. A device that only executes a repacked layout is excluded here by that
// property rather than by name, which is how an NPU with two native quant formats ends up excluded
// without a single line written for it.
struct DeviceSurvey {
    uint64_t own_memory = 0;                   // total memory of the devices that have their own
    const ComputeDevice * candidate = nullptr; // the first of them
    uint32_t n_accelerators = 0;               // devices that are not the CPU: the only ones an offload means
};

DeviceSurvey survey_devices(const HardwareProfile & hw) {
    DeviceSurvey v;
    for (const ComputeDevice & d : hw.devices) {
        if (!d.is_cpu) ++v.n_accelerators;
        if (!d.has_own_memory()) continue;
        v.own_memory += d.memory_total;
        if (!v.candidate) v.candidate = &d;
    }
    return v;
}

void note_extra_offload(Planning & s, const DeviceSurvey & v) {
    const HardwareProfile & hw = s.hw;
    if (v.n_accelerators == 0) {
        // Three different answers, and only the first is about this machine. A build that compiled
        // no accelerator backend and found none beside itself will report a machine with no
        // accelerator on ANY hardware, which is a fact about the build being read as a fact about
        // the room.
        if (!hw.backends_looked)
            s.note("extra-offload", "none", Source::Unprobed,
                   "No device was looked for, so nothing more moves off the CPU.",
                   "nothing was enumerated because nobody went to look: the backend registry was never "
                   "asked, so this says nothing about what this machine has");
        else if (hw.backends_linked <= 1 && hw.backends_loaded == 0)
            s.note("extra-offload", "none", Source::Unprobed,
                   "This build has no accelerator support, so nothing moves off the CPU.",
                   "this build carries no accelerator backend and found none beside the executable, so it "
                   "would report a machine with no accelerator on any hardware. That is a property of the "
                   "build, not of this machine: configure the backends whose SDK is present, or ship them "
                   "as loadable libraries next to the binary");
        else
            s.note("extra-offload", "none", Source::Measured, "No device beyond the CPU was found.",
                   u64s(hw.backends_linked + hw.backends_loaded) +
                       " backends were registered and none of them offered a compute device beyond the "
                       "host CPU, so there was nothing to consider past what the fitter placed");
    } else if (v.own_memory == 0) {
        // Two claims used to be made here and only one of them was provable. That moving a weight
        // onto a device whose memory IS the host's frees nothing is arithmetic. That it "wins no
        // bandwidth" was an assumption, and a wrong one: on shared memory an accelerator can still
        // reach more of the bus than the cores do - which is measured elsewhere at roughly twice -
        // because a CPU core count is not a memory controller. So the capacity half is stated and
        // the bandwidth half is compared, or admitted unknown.
        const ComputeDevice * fastest = nullptr;
        for (const ComputeDevice & d : hw.devices) {
            if (d.is_cpu || d.memory_bandwidth_gibs <= 0.0) continue;
            if (!fastest || d.memory_bandwidth_gibs > fastest->memory_bandwidth_gibs) fastest = &d;
        }
        const std::string capacity = "every device here reads the host's own memory, so moving a weight onto "
                                     "one frees nothing: the capacity tier is inert. ";
        // "More" only past the policy's margin: see PlannerPolicy::min_backend_win for why a
        // difference inside the probe's own repeatability is not reported as a win.
        if (fastest && hw.host_bandwidth_gibs > 0.0 && fastest->memory_bandwidth_gibs > s.host_bandwidth_win)
            s.note("extra-offload", "none", Source::Measured,
                   "Nothing more moves to the device: it shares this memory, though it reads faster.",
                   capacity + "What is left is bandwidth, and here the device has more of it (" +
                       u64s((uint64_t) fastest->memory_bandwidth_gibs) + " against " +
                       u64s((uint64_t) hw.host_bandwidth_gibs) +
                       " GiB/s): what the fitter placed is worth having there, and the streamed experts "
                       "would be too if they could be handed over");
        else if (fastest && hw.host_bandwidth_gibs > 0.0)
            s.note("extra-offload", "none", Source::Measured,
                   "Nothing to move to the device: it shares this memory and reads no faster.",
                   capacity + "What is left is bandwidth, and here the device has no more of it (" +
                       u64s((uint64_t) fastest->memory_bandwidth_gibs) + " against " +
                       u64s((uint64_t) hw.host_bandwidth_gibs) + " GiB/s), so there is nothing to win");
        else
            s.note("extra-offload", "none", Source::Unprobed,
                   "Nothing more moves to the device: it shares this memory, its speed unmeasured.",
                   capacity + "What is left is bandwidth, and no device's own figure was measured here - "
                              "which is the number that would say whether an offload pays at all");
    } else if (v.candidate && v.candidate->memory_bandwidth_gibs <= 0.0) {
        s.note("extra-offload", "none", Source::Unprobed, "Nothing more moves to the device: its speed is unmeasured.",
               "a device with " + u64s(mib(v.own_memory)) +
                   " MiB of its own is present and the fitter has already used it; moving anything FURTHER "
                   "would turn on its memory bandwidth, which is unmeasured here");
    } else {
        s.note("extra-offload", "none", Source::Policy, "Nothing beyond the layer placement moves to a device.",
               "the fitter has already placed what fits on the devices - see the placement line above - and "
               "placing weights across devices is a capacity problem it solves exactly; this planner owns "
               "the tier below it and does not second-guess it");
    }
}

// Where the streamed experts are COMPUTED, which is a different question from where they live. They
// must live in a host buffer - that is absolute, since serving one means rebinding its pointer onto
// the file's native layout, and a repack replaces exactly that layout. But a device that offers a
// host buffer reads that same memory directly, so it can execute over the bytes we read from flash
// without a copy and without a repack, and the pair of properties is one device's answer rather
// than a class of hardware.
//
// Whether it is worth doing is a second question, and the honest gate is not "is there a device"
// but "is compute even on the critical path". Reading an expert costs far more than multiplying by
// it, so where the stall dominates the engine that computes is irrelevant and moving the work buys
// nothing. Where the cache is large enough that most tokens hit, the compute emerges and the faster
// engine is worth having. Both halves of that comparison need a bandwidth figure for the device, and
// until one is measured this stays a stated unknown rather than a decision - which on the machines
// where it matters most, the ones whose memory is unified, is the whole of the difference between an
// idle accelerator and a used one.
void note_expert_compute(Planning & s, const DeviceSurvey & v) {
    const HardwareProfile & hw = s.hw;
    const ComputeDevice * host_capable = nullptr;
    for (const ComputeDevice & d : hw.devices) {
        if (d.is_cpu) continue; // the CPU is where they already are
        if (!is_yes(d.host_buffer) || d.runs_expert_op != Tri::Yes || d.needs_repack == Tri::Yes) continue;
        host_capable = &d;
        break;
    }
    const bool repack_blocks = v.candidate && v.candidate->needs_repack == Tri::Yes;
    if (host_capable && !s.pol.streamer_serves_device_memory)
        s.note("experts", "host", Source::Derived, "Experts compute on the CPU: no device can reach streamed memory.",
               std::string("this device could compute them on paper - it offers a host buffer and runs the "
                           "model's own layout - and it still cannot. The streamer rebinds every expert onto "
                           "memory it reserved itself, which no device was given access to; and a host buffer "
                           "type would not survive the load anyway, since a mapped model has one substituted "
                           "for the CPU's. ") +
                   (is_yes(host_capable->host_ptr_buffers)
                        ? "This device can wrap memory a caller already owns, which is the one opening: the "
                          "streamer's own reservations would have to be handed over that way"
                    : host_capable->host_ptr_buffers == Tri::No
                        ? "This device cannot wrap memory a caller already owns, so there is no opening here "
                          "at all"
                        : "Whether this device can wrap memory a caller already owns - the one opening left - "
                          "is unprobed. What it advertises is a blanket promise, and a backend whose support "
                          "is conditional declines to make one while implementing the call anyway, so the "
                          "absence of a yes decides nothing here"));
    else if (host_capable && host_capable->memory_bandwidth_gibs <= 0.0)
        s.note("experts", "host", Source::Unprobed, "Experts compute on the CPU: the device's speed is unmeasured.",
               "this device offers a host buffer and executes this model's expert matmul on the file's own "
               "layout, so streamed experts could be computed on it - but what that is worth is the ratio of "
               "its bandwidth to the host's, and its own is unmeasured here");
    else if (host_capable && hw.host_bandwidth_gibs > 0.0 &&
             host_capable->memory_bandwidth_gibs > hw.host_bandwidth_gibs)
        s.note("experts", "device", Source::Measured,
               "Experts belong on " + host_capable->name + ", at " +
                   u64s((uint64_t) host_capable->memory_bandwidth_gibs) + " GiB/s against the CPU's " +
                   u64s((uint64_t) hw.host_bandwidth_gibs) + ".",
               "this device offers a host buffer, executes the model's own layout, and reaches " +
                   u64s((uint64_t) host_capable->memory_bandwidth_gibs) + " GiB/s against the host's " +
                   u64s((uint64_t) hw.host_bandwidth_gibs) +
                   ": once the streamer can hand it their memory, "
                   "this is where they belong");
    else
        s.note("experts", "host", Source::Derived,
               repack_blocks ? "Experts compute on the CPU: the device needs a different weight layout."
                             : "Experts compute on the CPU, from memory the streamer controls.",
               repack_blocks ? "a streamed expert must keep the file's native layout so its pointer can be "
                               "rebound, and the device here executes only a repacked layout"
                             : "a streamed expert must live in a host buffer whose pointer we may rebind onto "
                               "the file's native layout");
}

// ── the dense policy: held in memory; how is what a reclaim COSTS here ─────────
// Every decoded token reads the whole dense set, so a page of it the kernel drops is a page read
// back from flash on the next token, in fault-sized pieces and outside every counter the streamer
// keeps. A mapping is free of whatever limit the process is held to for exactly the reason it is
// dropped: the kernel owes it nothing. There used to be a branch here that left the dense set
// mapped where exceeding the limit is fatal and clean file pages are not charged to it. It held the
// run inside its share and cost it 25581 major faults a token (about 400 MiB read back per token,
// the cores 75% idle): 1.2 tok/s where the same model held in memory makes 5. What a fatal limit
// asks for is a wider margin (margin_frac), which the ledger then sizes against, not a weight the
// kernel may take back. (A tensor larger than the process may hold at all still stays mapped: that
// is the runtime's rule and dense_pending's below, and it is about size, not about the platform.)
//
// One branch is only a candidate at this point. That a reclaim-exempt store exists says the dense
// set MAY be pinned; whether it is depends on what else is charged to the same total, which is the
// ledger's to say. So that branch is settled after the allocation (settle_dense_pin), and this
// returns whether it was taken.
bool choose_dense_policy(Planning & s) {
    Plan & p = s.p;
    const HardwareProfile & hw = s.hw;
    if (s.pinned("dense-weights", dense_weights_mode_name(p.config.moe.dense_weights))) return false;
    if (hw.reclaim_exempt_max > 0 && hw.lockable_bytes_known && hw.anon_overflow == Overflow::Compress) {
        // Anonymous memory keeps the dense set off flash but not out of a compressed swap, where
        // every touch costs a decompression that no I/O counter shows. A store the kernel may not
        // reclaim is the only thing that closes that gap.
        p.config.moe.dense_weights = DenseWeightsMode::Pinned;
        return true;
    } else if (hw.reclaim_exempt_max > 0 && !hw.lockable_bytes_known && hw.anon_overflow == Overflow::Compress) {
        p.config.moe.dense_weights = DenseWeightsMode::Anonymous;
        s.note("dense-weights", dense_weights_mode_name(DenseWeightsMode::Anonymous), Source::Unprobed,
               "Dense weights are copied into RAM: the reclaim-exempt total is unmeasured.",
               "a reclaim-exempt allocation exists, but an unread ceiling cannot establish that the whole dense "
               "set fits it; anonymous buffers keep the dense set off flash without assuming a lock grant");
    } else if (hw.anon_overflow == Overflow::Unknown) {
        p.config.moe.dense_weights = DenseWeightsMode::Anonymous;
        s.note("dense-weights", dense_weights_mode_name(DenseWeightsMode::Anonymous), Source::Policy,
               "Dense weights are copied into RAM, the default where memory pressure is unknown.",
               "what a reclaim costs here was not probed; anonymous buffers are the default that at "
               "least keeps the dense set off a flash refault");
    } else {
        p.config.moe.dense_weights = DenseWeightsMode::Anonymous;
        s.note("dense-weights", dense_weights_mode_name(DenseWeightsMode::Anonymous), Source::Derived,
               "Dense weights are copied into RAM, so they are never re-read from storage.",
               "no reclaim-exempt store here, so anonymous buffers are what keeps the dense set off a "
               "flash refault (a reclaim then means: " +
                   std::string(overflow_name(hw.anon_overflow)) + ")");
    }
    return false;
}

// Bytes the dense policy is about to take out of reclaimable page cache. Mirrors the runtime's own
// rule: a tensor bigger than what this process may hold is never converted, it stays mapped. The
// comparison stays on the REPORTED figure rather than the measured headroom, because it is
// mirroring a decision the runtime makes from the reported one - a plan that predicted a conversion
// the engine will then refuse would be worse than one that is merely conservative.
uint64_t dense_pending_bytes(const Planning & s) {
    const DenseWeightsMode mode = s.p.config.moe.dense_weights;
    const bool converts_dense = mode == DenseWeightsMode::Anonymous || mode == DenseWeightsMode::Pinned;
    uint64_t dense_pending = converts_dense ? s.host_dense - s.host_fixed - s.projector_host - s.checkpoint_bytes : 0;
    if (converts_dense && s.model.largest_dense_tensor > s.hw.residency_budget)
        dense_pending = dense_pending > s.model.largest_dense_tensor ? dense_pending - s.model.largest_dense_tensor : 0;
    return dense_pending;
}

// ── the allocation: what each group is WORTH, not what is left over ─────────────
//
// This is where the residual sizing used to be - cache gets `usable` minus whatever the dense
// policy took - and the residual is the shape that cannot express the two cases that matter. On a
// machine with room it hands the cache more than a hit is worth; on a pressured one it hands the
// dense set memory that would have bought more hits. Both halves now compete in one unit, seconds
// per byte of weights, and the cache is a fractional candidate above its own floor rather than the
// remainder.
//
// The allocator sees the HOST's share: where the first stage put some layers' experts on a device,
// those bytes are not the streamer's to hold and must not be planned for.
ModelProfile host_model(const Planning & s) {
    ModelProfile m = s.model;
    m.group(WeightGroup::Experts).bytes = s.host_expert_bytes;
    if (s.n_moe_layer && s.host_layers < s.n_moe_layer) {
        const double share = (double) s.host_layers / (double) s.n_moe_layer;
        m.token_cycle_bytes = (uint64_t) ((double) s.model.token_cycle_bytes * share);
        m.group(WeightGroup::Experts).bytes_per_token = m.token_cycle_bytes;
    }
    return m;
}

// The decode thread count this plan will run with: what choose_threads, at the end of the plan, will
// settle on (the caller's, a count measured on this model, the sweep's, or the default).
int planned_decode_threads(const Planning & s) {
    const RunConfig & c = s.p.config;
    if (s.req.is_pinned("threads")) return c.n_threads;
    if (s.model.knobs.threads > 0) return s.model.knobs.threads;
    if (s.hw.best_threads > 0) return (int) s.hw.best_threads;
    return c.n_threads;
}

// ── the decode's own cost, measured on an earlier load (MeasuredKnobs::decode_*) ─────────────
// The ledger prices decode compute from a probed matmul rate and the cache's hits from a replay of
// the routing, and a one-token decode pays what neither sees: per-op and thread-synchronisation
// costs, and a cache that serves less than the replay of a few hundred tokens did. Measured on a
// 12 GB phone, Ling-3.0-tiny Q8_0: 0.037 s/token predicted at a 95% credited hit rate, 0.107
// measured at 77-85%, and compute alone ~64 ms. A load of this model on this machine timed what its
// decode cost, so the plan prices with that: the compute terms all scale by the ratio between the
// measured compute and what this cost model priced the same configuration at, so the groups keep
// their relative prices and the ranking still weighs them against each other, and the cache's credit
// scales by the ratio between the hits measured at that load's budget and the credit there - lowered
// toward the measurement, never raised past what the replay justified (the prior, which is no
// measurement of this model, moves both ways). A figure answers only for the configuration it was
// measured under: the same decode thread count and the same placement on the devices.
void calibrate_decode_cost(Planning & s) {
    const MeasuredKnobs & mk = s.model.knobs;
    if (!(mk.decode_compute_s > 0.0) && !(mk.decode_hit >= 0.0)) return;
    const RunConfig & c = s.p.config;
    const int threads = planned_decode_threads(s);
    if (mk.decode_gpu_layers != c.n_gpu_layers || (mk.decode_device_cache_mb > 0) != (c.moe.device_cache_mb > 0)) {
        s.note("decode-cost", "probed", Source::Derived,
               "Decode is priced from the probes: the decode measured here ran with another device placement.",
               "an earlier load measured this decode with " + u64s((uint64_t) mk.decode_gpu_layers) +
                   " layers on devices and " + (mk.decode_device_cache_mb > 0 ? std::string("a") : std::string("no")) +
                   " device expert cache, and this plan places " + u64s((uint64_t) std::max(0, c.n_gpu_layers)) +
                   " layers with " + (c.moe.device_cache_mb > 0 ? "a" : "no") +
                   " device cache: a figure measured under another placement does not answer for this one");
        return;
    }
    char nums[320] = "";
    std::string brief, value;
    // Compute.
    const double predicted = host_compute_seconds(s.hw, s.host_model);
    bool compute_other_threads = false;
    if (mk.decode_compute_s > 0.0 && predicted > 0.0) {
        if (mk.decode_threads == threads) {
            s.compute_scale = mk.decode_compute_s / predicted;
            char b[160];
            std::snprintf(b, sizeof(b), "compute %.0f ms/token against %.0f predicted", 1e3 * mk.decode_compute_s,
                          1e3 * predicted);
            brief = b;
            std::snprintf(b, sizeof(b), "compute x%.2f", s.compute_scale);
            value = b;
            std::snprintf(nums, sizeof(nums),
                          "its decode computed for %.1f ms a token at %d threads, flash waits excluded, where the "
                          "matmul rates price the same placement at %.1f ms, so every group's compute is priced "
                          "%.2fx",
                          1e3 * mk.decode_compute_s, threads, 1e3 * predicted, s.compute_scale);
        } else
            compute_other_threads = true;
    }
    // Hits, at the budget the load ran with, on the same expert set this plan caches.
    std::string hits;
    const uint64_t budget = (uint64_t) (mk.decode_cache_mib * 1024.0 * 1024.0);
    if (mk.decode_hit >= 0.0 && budget > 0 && s.host_expert_bytes > 0 && s.model.is_moe) {
        const HitCurve & curve = s.model.hit_curve;
        const bool from_curve = curve.credited() && s.model.expert_bytes > 0;
        const double credited =
            from_curve
                ? curve.hit_share(
                      (uint64_t) ((double) budget * (double) s.model.expert_bytes / (double) s.host_expert_bytes))
                : std::min(1.0, (double) s.pol.cache_hit_optimism * (double) budget / (double) s.host_expert_bytes);
        if (credited > 0.0) {
            double scale = mk.decode_hit / credited;
            // A load measuring more than this model's own record replays is the cache warming past what a
            // short record saw, not a better model of it; the prior is a guess either way.
            if (curve.measured) scale = std::min(1.0, scale);
            if (scale != 1.0) {
                s.hit_scale = scale;
                char b[200];
                std::snprintf(b, sizeof(b), "hits %.0f%% against %.0f%%", 100.0 * mk.decode_hit, 100.0 * credited);
                brief += (brief.empty() ? "" : ", ") + std::string(b);
                std::snprintf(b, sizeof(b), "hits x%.2f", scale);
                value += (value.empty() ? "" : ", ") + std::string(b);
            }
            char b[320];
            std::snprintf(b, sizeof(b),
                          "its expert cache served %.0f%% of the decode lookups at %.0f MiB, where %s credits "
                          "%.0f%% there, %s",
                          100.0 * mk.decode_hit, mk.decode_cache_mib,
                          from_curve ? "the replay of this model's routing" : "the linear credit", 100.0 * credited,
                          scale == 1.0 ? "so the credit stands: a measurement above the replay is not a reason to "
                                         "credit past it"
                          : from_curve ? "so the curve is credited at that ratio at every size"
                                       : "so the linear credit is scaled to it at every size");
            hits = b;
        }
    }
    if (value.empty()) {
        if (compute_other_threads)
            s.note("decode-cost", "probed", Source::Derived,
                   "Decode is priced from the probes: the decode measured here ran at " +
                       u64s((uint64_t) mk.decode_threads) + " threads, this plan runs " + u64s((uint64_t) threads) +
                       ".",
                   "an earlier load measured this decode's compute at " + u64s((uint64_t) mk.decode_threads) +
                       " threads; this plan decodes at " + u64s((uint64_t) threads) +
                       ", and a figure measured at another count does not answer for it" +
                       (hits.empty() ? std::string() : "; " + hits));
        return;
    }
    std::string reason = "measured on an earlier load of this model here: ";
    reason += nums[0] ? std::string(nums) : std::string();
    if (!hits.empty()) reason += (nums[0] ? "; " : "") + hits;
    if (compute_other_threads)
        reason += "; its compute ran at " + u64s((uint64_t) mk.decode_threads) + " threads, not this plan's " +
                  u64s((uint64_t) threads) + ", so the compute stays priced from the probes";
    s.note("decode-cost", value, Source::Measured, "Decode priced from a load here: " + brief + ".", reason);
}

AllocationInputs ledger_inputs(const Planning & s, uint64_t dense_pending) {
    const RunConfig & c = s.p.config;
    AllocationInputs ai;
    if (s.home.priced) {
        ai.layers_home_bytes = s.home.bytes;
        ai.layers_home_saved = s.home.saved;
        ai.layers_home_count = s.home.layers;
    }
    ai.budget_bytes = s.usable;
    ai.engine_cache_min = (uint64_t) MoeStreamConfig::cache_min_mb << 20;
    ai.can_pin = c.moe.dense_weights == DenseWeightsMode::Pinned;
    ai.pin_forced = ai.can_pin && s.req.is_pinned("dense-weights");
    ai.reserved_bytes = dense_pending;
    ai.cacheable_bytes = s.host_expert_bytes;
    ai.fixed_bytes = s.host_fixed;
    ai.projector_bytes = s.projector_host;
    ai.checkpoint_bytes = s.checkpoint_bytes;
    // A discardable cache is never locked (the engine drops the pin where it can offer instead).
    ai.cache_lock = c.moe.cache_pin && !(c.moe.cache_discard && s.hw.discardable_memory);
    ai.cache_offered = c.moe.cache_discard && s.hw.discardable_memory;
    ai.holdable_bytes = s.holdable;
    ai.margin_frac = margin_frac(s.hw, s.pol);
    ai.margin_min_bytes = s.pol.margin_min_bytes;
    ai.compute_scale = s.compute_scale;
    ai.hit_scale = s.hit_scale;

    // The second ceiling, with its own margin. A store that publishes no total is bounded by what
    // the process may hold, which is the only bound there is on it.
    auto lock_cap = [&](uint64_t reported) {
        const uint64_t bounded = std::min(reported, s.holdable);
        const uint64_t m = std::max((uint64_t) ((double) bounded * (double) s.pol.lock_margin), s.pol.margin_min_bytes);
        return bounded > m ? bounded - m : 0;
    };
    ai.lockable_bytes = lock_cap(s.hw.lockable_bytes);
    ai.lock_in_place_bytes = lock_cap(s.hw.lock_in_place_bytes);
    ai.lockable_bytes_known = s.hw.lockable_bytes_known;
    ai.lock_in_place_bytes_known = s.hw.lock_in_place_bytes_known;
    return ai;
}

// ── the prefill plan: a separate placement, decided on its own axis ─────────────
// Prefill and decode want opposite hardware, so this is not the device question above asked again.
// A decode graph is one token wide and bound by reading weights; a prefill graph is hundreds wide
// and bound by computing on them, and the same device can lose one and win the other several times
// over. What decides it is therefore the WIDE measurement, never the one-token bandwidth - and three
// more things, each a fact rather than a name: the device reproduced the host's answer at that
// width, the run's other settings can coexist with it, and what it costs in memory still leaves the
// ledger closed.
//
// What a device costs is the model's own arithmetic: under streaming it carries a layer's experts
// and a layer's other weights through two slots each - the layer being computed and the one being
// loaded behind it - plus the compute buffer llama.cpp projects for it. It is charged to this pool
// only where the device's memory IS this pool.
//
// The expert slots are the part with a choice in it: two read a layer while the one before it
// computes, one waits for each layer and costs a layer less. Same result, so it is a trade of
// prefill time for memory and the ledger decides it - two where they fit, one where only one does.
uint64_t prefill_need(const ModelProfile & model, int slots) {
    return (uint64_t) slots * model.largest_expert_layer_bytes + 2 * model.largest_layer_dense_bytes;
}

// "Its own memory" is believed only where it comes with a size. A device measured as not drawing on
// the host and reporting no memory of its own is one whose buffers this process cannot see being
// taken - an accelerator fed through a kernel allocator answers exactly so - and a pool with no
// stated size cannot be budgeted against. It is charged here, which is the reading that cannot
// overcommit: seen on a phone whose NPU was armed at a cost of 0 MiB.
bool own_pool(const ComputeDevice * dev) {
    return dev && dev->has_own_memory() && dev->memory_total > 0;
}

// The device as a ledger row. Slots held only during a prefill take turns with the cache instead of
// standing beside it: they are charged as such, and what stays for the session is the rest.
void charge_prefill(const Planning & s, const ComputeDevice * dev, AllocationInputs & in, int slots, bool on_demand) {
    if (own_pool(dev)) return;
    const uint64_t experts = (uint64_t) slots * s.model.largest_expert_layer_bytes;
    // The compute buffer is the one the fitter projected for THIS device. A device it never places
    // layers on has none in the projection, and charging it the others' - a GPU's 467 MiB charged to
    // an NPU beside it on a phone - is memory the expert cache never gets. One the profile does not
    // know is charged the whole sum, the reading that cannot overcommit.
    uint64_t compute = dev ? s.placement.device_compute_of(dev->name) : s.placement.device_compute_bytes;
    // But a device the fitter never placed layers on is still handed a compute buffer when the session
    // arms it, for the same graph the host computes, as wide: the host's own projection at this width
    // is its size. Charged nothing, it was memory the plan never saw: on a 12 GB phone an NPU armed for
    // a 4096 context reserved 450 MiB of compute buffer beside a cache sized as if it had none, the
    // process was reclaimed down to a few MiB, and the load never finished.
    if (dev && compute == 0) compute = s.placement.raw_host_compute_bytes;
    const uint64_t rest = 2 * s.model.largest_layer_dense_bytes + compute;
    in.device_bytes = on_demand ? rest : rest + experts;
    in.device_turn_bytes = on_demand ? experts : 0;
    // Device buffers compete with locks wherever that pool may grant them. A known zero cannot
    // be outbid, so it must not reject a device; an unread ceiling keeps its independent bounds.
    // Every device allocation still consumes the ordinary process memory budget.
    in.device_locked = (in.lockable_bytes > 0 || in.lock_in_place_bytes > 0) && (!dev || dev->charges_lockable());
}

// What the profile says about prefill-width devices: the fastest that is both measured and
// verified, one that answered wrong, and the counts the refusals quote.
struct PrefillSurvey {
    const ComputeDevice * best = nullptr;
    const ComputeDevice * wrong = nullptr;
    const ComputeDevice * refused = nullptr; // what an earlier load of this model here did not keep
    uint32_t n_accel = 0;
    uint32_t n_measured = 0;
    uint32_t n_broken = 0;
    // Each accelerator the rule did not weigh, and why, appended to whatever the rule concludes: a
    // device that drops out without a word reads as a device the planner never saw, which is the
    // wrong lesson when it is the fastest one on the machine for another model.
    std::string unweighed;
};

PrefillSurvey survey_prefill(const HardwareProfile & hw, const MeasuredKnobs & knobs) {
    PrefillSurvey v;
    for (const ComputeDevice & d : hw.devices) {
        if (d.is_cpu) continue;
        ++v.n_accel;
        if (d.usable == Tri::No) { // it did not survive being probed: nothing to weigh
            ++v.n_broken;
            continue;
        }
        // An earlier load of this model here armed it and the session did not keep it: its check on
        // this model refused it, or the load never said (MeasuredKnobs::prefill). The probe
        // graph is not this model, and the session's check is; a load that never returned is the one
        // thing worse than a slow prompt.
        if (knobs.prefill_refused(d.name)) {
            v.refused = &d;
            v.unweighed += "; " + d.name + " was not armed: " +
                           (knobs.prefill_word(d.name)->verdict < 0
                                ? "an earlier load of this model here handed it back, its check on this model "
                                  "refused it"
                                : "earlier loads of this model here armed it and never kept it");
            continue;
        }
        if (d.wide_matmul_gibs <= 0.0) {
            v.unweighed +=
                "; " + d.name + " was not weighed: " +
                (d.runs_expert_op == Tri::No ? "it does not run this model's expert matmul in the model's own format"
                                             : "it gave no measurement at that width");
            continue;
        }
        ++v.n_measured;
        if (d.wide_identity_ok != Tri::Yes) {
            v.wrong = &d;
            continue;
        }
        if (!v.best || d.wide_matmul_gibs > v.best->wide_matmul_gibs) v.best = &d;
    }
    return v;
}

// What else in this run a device prefill cannot share a session with. The same conditions
// validate() refuses, read here so the plan declines them with a reason instead of producing a
// config that does not start. Empty when nothing conflicts.
std::string prefill_conflict(const RunConfig & c) {
    // The prefill device joins the load as the model's only device, with no layer of its own; with
    // layers already placed, that list would hand every one of them to this one device.
    if (c.n_gpu_layers != 0)
        return "the capacity fitter already placed layers on devices, and a prefill device loads as the model's "
               "only device with none of its own";
    if (c.moe.row_stream)
        return "row-streamed tables are gathered on the host, inside a graph that would run on the device";
    if (c.spec.enabled() && c.spec.draft_max + 1 >= c.prefill.min_tokens)
        return "a speculative verify is as wide as the narrowest piece a device is handed, so it would reuse a graph "
               "the device ran";
    if (c.decide.enabled && c.decide.prefix_cache == PrefixCacheMode::On)
        return "a kept prefix state would be saved from where the model state was before it moved";
    if (c.n_ubatch > 0 && c.n_ubatch < c.prefill.min_tokens)
        return "the ubatch is narrower than the narrowest graph a device is handed";
    if (c.prefill.min_tokens > c.n_ctx) return "the context is shorter than the narrowest graph a device is handed";
    return "";
}

// Whether an allocation's expert stream has a price to set against the mapping: what lets a cache
// under the floor be turned off rather than the stream declined (size_cache).
bool stream_priced(const Allocation & a) {
    const GroupPlacement & experts = a.at(WeightGroup::Experts);
    return experts.lane == Lane::ExpertStream && experts.seconds_per_token > 0.0 &&
           experts.seconds_per_token < experts.seconds_worst_case;
}

// A device whose room takes the expert cache away - under one token cycle with it, where without it
// the cache streamed - is a trade, not a refusal: every long prompt runs faster, every decoded token
// slower, and which is worth more is how this machine is used. A chat with long prompts and short
// answers wants the device, a long answer from a short prompt the cache. Measured on a 12 GB phone with
// a 125B model streamed: a 1054-token prompt prefilled on the cores in 122 s and decoded at 1.39
// tok/s, the plan having kept a 988 MiB cache over an NPU it measured at 3.0x the cores, where another
// model's prompts ran 8x faster on that NPU.
//
// So a request of this machine's own (MeasuredKnobs::mix_*) is priced both ways: its long prompts at
// the host's measured prefill rate, or at that rate times the device's measured wide ratio, and its
// answer at the decode each allocation predicts. The device is armed where the request is cheaper with
// it by the margin every device decision uses. Unpriced - nothing recorded of the requests, or of the
// host's prefill, or a decode either allocation cannot price - the cache is kept, as it always was.
struct PrefillTrade {
    bool priced = false;
    bool device_wins = false;
    std::string account; // the figures, or what pricing them is owed
};

PrefillTrade
price_prefill_trade(const Planning & s, double wide_ratio, const Allocation & without, const Allocation & with) {
    PrefillTrade t;
    const MeasuredKnobs & mk = s.model.knobs;
    if (mk.mix_requests <= 0 || mk.host_prefill_tps <= 0.0) {
        t.account = "unpriced: " +
                    std::string(mk.mix_requests <= 0 ? "no request of this machine is recorded yet"
                                                     : "the host's prefill of a long prompt is not measured here yet") +
                    ", and a load records it; the next plan weighs the prompts' prefill against the answers' decode, "
                    "and arms the device where long prompts and short answers make it the cheaper";
        return t;
    }
    if (!stream_priced(with) || !(with.seconds_per_token > 0.0) || !(without.seconds_per_token > 0.0) ||
        !(wide_ratio > 0.0)) {
        t.account = "unpriced: the decode with the device or without it has no price here";
        return t;
    }
    const double host_tps = mk.host_prefill_tps, dev_tps = host_tps * wide_ratio;
    const double prompt = mk.mix_wide_prompt_tokens, gen = mk.mix_gen_tokens;
    const double cost_with = prompt / dev_tps + gen * with.seconds_per_token;
    const double cost_without = prompt / host_tps + gen * without.seconds_per_token;
    t.priced = true;
    t.device_wins = cost_with * (1.0 + (double) s.pol.min_backend_win) < cost_without;
    char b[480];
    std::snprintf(b, sizeof(b),
                  "priced on this machine's requests (%d recorded): %.0f prompt tokens and %.0f decoded per request; "
                  "with the device %.1f s (prefill at %.1f tok/s, %.1fx the host's measured %.1f, decode %.3f s/token "
                  "with no cache), without it %.1f s (decode %.3f s/token with the cache)",
                  mk.mix_requests, prompt, gen, cost_with, dev_tps, wide_ratio, host_tps, with.seconds_per_token,
                  cost_without, without.seconds_per_token);
    t.account = b;
    return t;
}

// A device that pays for itself, and whether it FITS - which is the ledger's to say, with the device
// as a row: the run must not lose what decode had without it - a cache of at least one token cycle,
// and the protection of the dense set and of the cache - since those are paid on every decoded token
// and a prefill is paid once. Each is measured against the allocation WITHOUT the device: a cache
// that was already under a token cycle (and so off) is not something the device takes away, and
// refusing a faster prefill for it costs the prompt and saves decode nothing. A device that fails on the cache
// alone is the exception: it is a trade, priced on this machine's requests where they are recorded
// (price_prefill_trade), and refused as above where they are not. Two slots first, then one: the wider
// scheme where it fits, the narrower where only it does. A caller who set the count has chosen, and only that count is
// tried. Returns the allocation the run gets, armed or not.
Allocation fit_prefill_device(Planning & s,
                              const ComputeDevice & best,
                              const std::string & ratio,
                              double wide_ratio,
                              const std::string & width,
                              AllocationInputs & ai,
                              const Allocation & alloc,
                              const std::string & unweighed) {
    Plan & p = s.p;
    const bool slots_by_caller = s.req.is_pinned("prefill-slots");
    // Slots that take turns with the cache are the scheme a plan arms: the cache is idle exactly
    // when they are busy, so standing them beside it charges decode, on every token, for memory only
    // a prefill uses. A caller who set it either way has chosen.
    const bool turns = s.req.is_pinned("prefill-slots-on-demand") ? p.config.prefill.slots_on_demand : true;
    // Weighed with the graph the device will run. Armed, the device computes every pass and the graph
    // is held to its bound (bound_prefill_width), and a narrower graph reserves smaller buffers, on
    // the host and as the device's own row. The ledger used to be made at the width the plan had
    // before, and the bound applied after it: the wide graph's buffers stayed charged and the cache
    // was sized under them (at a 32768 context, 384 MiB of cache for a graph of 16 KiB a token). So
    // the placement is the bounded one from here on, and put back on every refusal.
    const Placement wide = s.placement;
    const uint64_t wide_fixed = s.host_fixed, wide_all = s.host_all, wide_dense = s.host_dense;
    if (const uint32_t bound = prefill_width_bound(s)) {
        s.placement = wide.at_width(bound);
        s.host_fixed = host_fixed_of(s.placement);
        const uint64_t freed = wide_fixed > s.host_fixed ? wide_fixed - s.host_fixed : 0;
        s.host_all = wide_all - std::min(wide_all, freed);
        s.host_dense = wide_dense - std::min(wide_dense, freed);
    }
    const auto refused = [&]() -> const Allocation & {
        s.placement = wide;
        s.host_fixed = wide_fixed;
        s.host_all = wide_all;
        s.host_dense = wide_dense;
        return alloc;
    };
    AllocationInputs narrow = ai;
    narrow.fixed_bytes = s.host_fixed;
    AllocationInputs with = narrow;
    Allocation armed;
    int slots = 0;
    const uint64_t cycle = s.host_model.token_cycle_bytes;
    const auto cache_streams = [cycle](const Allocation & a) { return a.cache_bytes > 0 && a.cache_bytes >= cycle; };
    const bool streamed = cache_streams(alloc);
    bool keeps_cache = false, keeps_dense = false, fits_held = false, own_short = false;
    // The widest scheme that fails on the cache alone: the trade price_prefill_trade weighs.
    AllocationInputs traded_in;
    Allocation traded;
    int traded_slots = 0;
    for (int n = slots_by_caller ? p.config.prefill.slots : PrefillDeviceConfig::slots_max;
         n >= PrefillDeviceConfig::slots_min; --n) {
        with = narrow;
        charge_prefill(s, &best, with, n, turns);
        own_short = own_pool(&best) && best.memory_free > 0 && best.memory_free < prefill_need(s.model, n);
        armed = allocate(s.hw, s.host_model, with, s.pol);
        keeps_cache = !streamed || cache_streams(armed);
        // "Protection" is the dense set's pin and the cache's lock alike: a device that leaves the
        // cache as ordinary memory where it was locked has taken from decode the very thing the
        // ledger exists to keep.
        const bool keeps_pin = armed.dense_pinned == alloc.dense_pinned;
        keeps_dense = keeps_pin && (armed.cache_locked_bytes > 0) == (alloc.cache_locked_bytes > 0) &&
                      armed.cache_lock_unverified == alloc.cache_lock_unverified;
        // What decode holds with the device must still fit what the process may hold - or, where the
        // ledger did not close without it (a pin the caller forced), hold no more than it did. Where a
        // cache streamed, the ledger shrinks it to make room and this holds by construction; where
        // none did, nothing yields, and the device's row is new memory on a pool that may be full.
        fits_held = armed.ledger.held <= armed.ledger.holdable_cap || armed.ledger.held <= alloc.ledger.held;
        if (!own_short && keeps_cache && keeps_dense && fits_held && armed.turn_fits) {
            slots = n;
            break;
        }
        // The cache's lock goes with the cache: a trade that gives up the one gives up the other.
        if (!own_short && !keeps_cache && keeps_pin && fits_held && armed.turn_fits && traded_slots == 0) {
            traded_in = with;
            traded = armed;
            traded_slots = n;
        }
        if (slots_by_caller) break;
    }
    const uint64_t least =
        prefill_need(s.model, slots_by_caller ? p.config.prefill.slots : PrefillDeviceConfig::slots_min);
    if (slots == 0 && own_short) {
        s.note("prefill-device", "off", Source::Measured,
               "Prompts run on the CPU: " + best.name + " has " + size_brief(best.memory_free) + " free, needs " +
                   size_brief(least) + ".",
               best.name + " computes a wide prefill at " + ratio + " the host's rate and has " +
                   u64s(mib(best.memory_free)) + " MiB free of the " + u64s(mib(least)) +
                   " MiB even one layer of this model takes there: refused on memory" + unweighed);
        return refused();
    }
    // The cache is the only thing the device takes: a trade, settled by the price where there is one.
    PrefillTrade trade;
    if (slots == 0 && traded_slots > 0) {
        trade = price_prefill_trade(s, wide_ratio, alloc, traded);
        // Priced, what is reported - armed or refused - is the scheme that was weighed; unpriced, the
        // refusal reads as it always did.
        if (trade.priced) {
            with = traded_in;
            armed = traded;
            keeps_cache = false;
        }
        if (trade.device_wins) slots = traded_slots;
    }
    if (slots == 0 && trade.priced) {
        s.note("prefill-device", "off", Source::Measured,
               "Prompts run on the CPU: on this machine's requests the expert cache is worth more than " + best.name +
                   " (" + ratio + " faster).",
               best.name + " computes a wide prefill at " + ratio + " the host's rate, and with " +
                   (traded_slots == 1 ? "a single expert slot" : "two expert slots") + " it would hold " +
                   u64s(mib(with.device_bytes + with.device_turn_bytes)) +
                   " MiB of this pool, which leaves the expert cache " + u64s(mib(armed.cache_bytes)) +
                   " MiB against a token cycle of " + u64s(mib(cycle)) + " MiB, where it had " +
                   u64s(mib(alloc.cache_bytes)) + " MiB without the device; " + trade.account +
                   ", so the cache is kept: the device is not cheaper by the " +
                   u64s((uint64_t) (s.pol.min_backend_win * 100.0f + 0.5f)) + "% margin" + unweighed);
        return refused();
    }
    if (slots == 0) {
        s.note("prefill-device", "off", Source::Derived,
               "Prompts run on the CPU: " + best.name + " (" + ratio + " faster) would take " +
                   size_brief(with.device_bytes + with.device_turn_bytes) +
                   (!keeps_cache   ? " the cache needs."
                    : !keeps_dense ? " the dense weights' protection needs."
                                   : ", more than this run may hold."),
               best.name + " computes a wide prefill at " + ratio +
                   " the host's rate and is refused on memory: with a single expert slot it would still hold " +
                   u64s(mib(with.device_bytes + with.device_turn_bytes)) +
                   " MiB of this pool (a layer of this model and its compute buffer), " +
                   (!keeps_cache ? "which leaves the expert cache " + u64s(mib(armed.cache_bytes)) +
                                       " MiB against a token cycle of " + u64s(mib(cycle)) + ", where it had " +
                                       u64s(mib(alloc.cache_bytes)) +
                                       " MiB without the device. That is paid on every decoded token, and a "
                                       "prefill is paid once" +
                                       (trade.account.empty() ? std::string() : "; " + trade.account)
                    : !keeps_dense ? "which leaves the dense set or the cache without the protection it has. That "
                                     "is paid on every decoded token, and a prefill is paid once"
                    : !fits_held   ? "which does not fit what the process may hold while it decodes"
                                   : "which does not fit what the process may hold while a prompt runs") +
                   unweighed);
        return refused();
    }
    ai = with;
    p.config.prefill.device = best.name;
    bound_prefill_width(s);
    p.config.prefill.best_effort = true;
    p.config.prefill.slots = slots;
    p.config.prefill.slots_on_demand = turns;
    const bool cache_traded = trade.device_wins;
    s.note("prefill-device", best.name, Source::Measured,
           "Prompts run on " + best.name + ", " + ratio + " faster than the CPU, same results" +
               (cache_traded ? ": on this machine's requests that is worth more than the expert cache." : "."),
           "computes this model's matmul " + width + " at " + ratio + " the host's rate with the same result; its " +
               u64s(mib(with.device_bytes)) + " MiB" +
               (with.device_turn_bytes ? ", plus " + u64s(mib(with.device_turn_bytes)) +
                                             " MiB of expert slots that take turns with the cache,"
                                       : "") +
               " fit with " +
               (slots == PrefillDeviceConfig::slots_max
                    ? "two expert slots, so a layer is read while the one before it computes"
                    : "one expert slot - two did not fit - so the device waits for each layer's read") +
               ", and the expert cache gives up " +
               u64s(mib(alloc.cache_bytes > armed.cache_bytes ? alloc.cache_bytes - armed.cache_bytes : 0)) +
               " MiB for it" +
               (cache_traded ? ", which leaves it under the token cycle of " + u64s(mib(cycle)) +
                                   " MiB, so it is off: " + trade.account + ", cheaper with the device by the " +
                                   u64s((uint64_t) (s.pol.min_backend_win * 100.0f + 0.5f)) + "% margin"
                             : std::string()) +
               ". Decode stays on the host, and so does the prefill if the device cannot be set up at "
               "load. A device's arithmetic is not the host's to the last bit, so the text can differ from a host "
               "prefill's: pass an empty " +
               flag_spelling("prefill-device") + " to keep the prefill on the host" + unweighed);
    return armed;
}

// The allocation, with the prefill device in it where one is armed - by the caller, or here.
// `ai` is updated to the inputs the returned allocation was made from.
Allocation plan_prefill(Planning & s, AllocationInputs & ai) {
    Plan & p = s.p;
    const HardwareProfile & hw = s.hw;
    if (p.config.prefill.enabled()) {
        // Armed by the caller: charged as asked, and reported. The name is the caller's, matched to
        // find that device's facts and for nothing else; one the profile does not know is charged
        // as if it shared the pool, which is the safe reading.
        const ComputeDevice * dev = nullptr;
        for (const ComputeDevice & d : hw.devices)
            if (!d.is_cpu && d.name == p.config.prefill.device) dev = &d;
        charge_prefill(s, dev, ai, p.config.prefill.slots, p.config.prefill.slots_on_demand);
        Allocation alloc = allocate(hw, s.host_model, ai, s.pol);
        s.note("prefill-device", p.config.prefill.device, Source::Operator,
               "Prompts run on " + p.config.prefill.device + ", as you set it; it takes " +
                   size_brief(ai.device_bytes + ai.device_turn_bytes) + ".",
               "set by the caller; the planner leaves it alone and charges it " +
                   u64s(mib(ai.device_bytes + ai.device_turn_bytes)) + " MiB in the ledger");
        return alloc;
    }
    const Allocation alloc = allocate(hw, s.host_model, ai, s.pol);
    if (s.req.is_pinned("prefill-device")) {
        s.note("prefill-device", "off", Source::Operator, "Prompts run on the CPU, as you set it.",
               "set by the caller; the planner leaves it alone");
        return alloc;
    }
    // A prefill device loads as the model's only device and moves layer weights onto it for a wide
    // graph; computing the experts from a device cache needs the layers on their device for good.
    // The session refuses the pair, so the plan does not arm one beside the other.
    if (p.config.moe.device_cache_mb > 0) {
        s.note("prefill-device", "off", Source::Derived,
               "Prompts are not given a device of their own: the experts are computed on the layers' device.",
               "a device expert cache is set, which keeps the layers on their device and computes the experts "
               "there; a prefill device is not combined with it");
        return alloc;
    }

    const PrefillSurvey v = survey_prefill(hw, s.model.knobs);
    const auto prefill_note = [&](std::string value, Source src, std::string brief, std::string reason) {
        s.note("prefill-device", std::move(value), src, std::move(brief), std::move(reason) + v.unweighed);
    };
    const double host_wide = hw.host_wide_matmul_gibs;
    auto ratio = [&](const ComputeDevice & d) {
        char b[32];
        std::snprintf(b, sizeof(b), "%.1fx", d.wide_matmul_gibs / host_wide);
        return std::string(b);
    };
    const std::string width = u64s(hw.wide_batch) + " tokens wide";
    const std::string conflict = prefill_conflict(p.config);

    if (v.n_accel == 0) {
        // The same three answers as the offload line: only one of them is about this machine.
        const bool build_blind = hw.backends_looked && hw.backends_linked <= 1 && hw.backends_loaded == 0;
        prefill_note("off", hw.backends_looked && !build_blind ? Source::Measured : Source::Unprobed,
                     build_blind ? "Prompts run on the CPU: this build has no accelerator support."
                                 : "Prompts run on the CPU: no other device was found.",
                     build_blind ? "this build carries no accelerator backend, so there was no device to consider: a "
                                   "property of the build, not of this machine"
                                 : "no compute device beyond the host CPU was enumerated, so the prefill runs where "
                                   "the decode does");
    } else if (v.n_broken == v.n_accel) {
        prefill_note("off", Source::Measured, "Prompts run on the CPU: every other device failed its check.",
                     "every device here failed its own probe, so the prefill runs where the decode does; see the "
                     "devices line for what each one did");
    } else if (!v.best && !v.wrong && v.refused) {
        prefill_note("off", Source::Measured,
                     "Prompts run on the CPU: " + v.refused->name +
                         " did not hold up for this model on an earlier load.",
                     "the device that could compute a wide prefill here was not kept by an earlier load of this model "
                     "on this machine and engine; a new engine version tries it again");
    } else if (host_wide <= 0.0 || v.n_measured == 0) {
        prefill_note("off", Source::Unprobed, "Prompts run on the CPU: no device was measured on a prompt.",
                     "a prefill is many tokens wide and bound by compute, not by the bandwidth a one-token graph "
                     "measures, and no device here was measured at that width: an unmeasured device is not armed");
    } else if (!v.best) {
        prefill_note("off", Source::Measured, "Prompts run on the CPU: " + v.wrong->name + " gave wrong results.",
                     v.wrong->name + " ran this model's matmul " + width +
                         " and did not reproduce the host's result: refused on correctness, whatever its speed");
    } else if (v.best->wide_matmul_gibs <= host_wide * (1.0 + (double) s.pol.min_backend_win)) {
        prefill_note("off", Source::Measured,
                     "Prompts run on the CPU: " + v.best->name + " is only " + ratio(*v.best) + " as fast.",
                     v.best->name + " computes this model's matmul " + width + " at " + ratio(*v.best) +
                         " the host's rate, which is not a win worth a second placement");
    } else if (!conflict.empty()) {
        prefill_note("off", Source::Derived,
                     "Prompts run on the CPU: " + v.best->name + " conflicts with another setting of this run.",
                     v.best->name + " computes a wide prefill at " + ratio(*v.best) +
                         " the host's rate and is not armed, because " + conflict);
    } else {
        return fit_prefill_device(s, *v.best, ratio(*v.best), v.best->wide_matmul_gibs / host_wide, width, ai, alloc,
                                  v.unweighed);
    }
    return alloc;
}

// The dense candidate, settled by the ledger.
void settle_dense_pin(Planning & s, bool pin_candidate, const Allocation & alloc, const AllocationInputs & ai) {
    if (!pin_candidate) return;
    const uint64_t dense_pending = s.p.dense_pending_bytes;
    if (alloc.dense_pinned) {
        s.note("dense-weights", dense_weights_mode_name(DenseWeightsMode::Pinned), Source::Derived,
               "Dense weights (" + size_brief(dense_pending) + ") are pinned, so memory pressure cannot take them.",
               "anonymous memory here is " + std::string(overflow_name(s.hw.anon_overflow)) + ", and the " +
                   u64s(mib(dense_pending)) + " MiB dense set fits whole in the " + u64s(mib(ai.lockable_bytes)) +
                   " MiB this process may hold reclaim-exempt (up to " + u64s(mib(s.hw.reclaim_exempt_max)) +
                   " MiB per buffer)");
        return;
    }
    s.p.config.moe.dense_weights = DenseWeightsMode::Anonymous;
    s.note("dense-weights", dense_weights_mode_name(DenseWeightsMode::Anonymous), Source::Derived,
           "Dense weights are copied into RAM: " + size_brief(alloc.pinned_shortfall_bytes) + " too big to pin whole.",
           "anonymous memory here is " + std::string(overflow_name(s.hw.anon_overflow)) +
               " and a reclaim-exempt store exists, but the " + u64s(mib(dense_pending)) + " MiB dense set is " +
               u64s(mib(alloc.pinned_shortfall_bytes)) +
               " MiB more than can still be held that way: pinned part way it would be reclaimed from the "
               "part that was refused, so it is held whole as ordinary memory");
}

// The part of the dense reservation the run will never hold: the embedding tables a copying policy
// is told to leave mapped - by the plan where the ranking did not call them resident
// (settle_mapped_tables), by a list the caller pinned where it names them all. The reservation is the
// dense policy's whole copy, so it carried them; left in, the ledger charges tens of GB nothing holds
// and the cache is sized short by exactly that. Measured on a 96 GB machine: a 27 GB per-layer table
// charged as dense held the cache to 92% of an expert set that fits whole beside what the run really
// holds. The same with a placement, whose host figure carries the tables as llama.cpp puts them
// (host_dense_of). Where the runtime already holds back a tensor too large, it may be one of these
// tables, so it is not taken out twice.
uint64_t unheld_table_bytes(const Planning & s, const Allocation & alloc) {
    const RunConfig & c = s.p.config;
    if (c.moe.dense_weights != DenseWeightsMode::Anonymous && c.moe.dense_weights != DenseWeightsMode::Pinned) return 0;
    const Lane lane = alloc.at(WeightGroup::Embedding).lane;
    if (!s.req.is_pinned("dense-mapped") && (lane == Lane::Resident || lane == Lane::Pinned)) return 0;
    uint64_t bytes = mappable_host_tables(s);
    if (s.model.largest_dense_tensor > s.hw.residency_budget)
        bytes = bytes > s.model.largest_dense_tensor ? bytes - s.model.largest_dense_tensor : 0;
    return std::min(bytes, s.p.dense_pending_bytes);
}

// The plan and the run must agree about the embedding tables too. The allocation prices the
// Embedding group on its own lane, and where that lane is not residency - left mapped or
// row-streamed, because the graph gathers rows from it - the run's dense policy must leave the
// tables where they are: Anonymous and Pinned copy every dense tensor they are handed, and a
// per-layer table of tens of GB that "fits" by the kernel's count was copied beside the cache the
// ledger had sized without it, and the control group ended the run (#424). The names come from
// the file's own grouping; the run holds back exactly those, as it holds back a tensor too large.
void settle_mapped_tables(Planning & s, const Allocation & alloc) {
    RunConfig & c = s.p.config;
    if (s.req.is_pinned("dense-mapped")) return;
    c.moe.dense_mapped.clear();
    const GroupPlacement & emb = alloc.at(WeightGroup::Embedding);
    const bool copies =
        c.moe.dense_weights == DenseWeightsMode::Anonymous || c.moe.dense_weights == DenseWeightsMode::Pinned;
    if (!copies || s.model.embedding_tensors.empty()) return;
    const uint64_t bytes = s.model.group(WeightGroup::Embedding).bytes;
    // The ranking's lane is what residency would be worth; the ledger is what was charged. A table
    // the ranking calls resident is copied by the dense policy all the same, so it may only be
    // copied where its bytes are in the reservation or still fit beside what was charged. Measured
    // where neither held: a 194 GB Engram table ranked resident because a short routing record's
    // flat curve priced the cache past the device's share at nothing, the reservation (the fitter's
    // host-resident 199 MiB) never held it, and the copy ended the run beside a 210 GB cache.
    const bool ranked_resident = emb.lane == Lane::Resident || emb.lane == Lane::Pinned;
    const bool charged = s.p.dense_pending_bytes >= bytes;
    const bool fits_uncharged = alloc.ledger.holdable_cap >= alloc.ledger.held + bytes;
    if (ranked_resident && (charged || fits_uncharged)) return;
    c.moe.dense_mapped = s.model.embedding_tensors;
    if (ranked_resident) {
        // The report follows the run: a placement still claiming the bytes would be a plan that
        // contradicts the ledger it prints beside it.
        GroupPlacement & q = s.p.allocation.at(WeightGroup::Embedding);
        q.lane = Lane::Mmap;
        q.resident_bytes = 0;
        q.seconds_per_token = q.seconds_worst_case;
        q.reason = "left mapped: residency was ranked, but the copy is not in the ledger and does not fit beside it";
    }
    s.note("dense-mapped", u64s(c.moe.dense_mapped.size()) + " table(s)", Source::Derived,
           "The embedding tables (" + size_brief(bytes) + ") stay mapped: the graph gathers rows from them.",
           "the allocation priced the embedding group as " + std::string(lane_name(emb.lane)) + " (" + emb.reason +
               "), and the " + dense_weights_mode_name(c.moe.dense_weights) + " policy would otherwise copy its " +
               u64s(mib(bytes)) + " MiB into memory the ledger did not charge");
}

// The plan and the run must agree about the cache's lock. Where the ledger sized it to what can be
// locked, the run is told to keep it there if the platform stops short; where it could not be
// locked at all, the run is told not to try - a lock that is granted for part of an ordinary cache
// protects nothing, and takes from the total the dense set was sized against. Neither is touched
// when the caller set it.
void settle_cache_lock(Planning & s, const Allocation & alloc) {
    RunConfig & c = s.p.config;
    // A cache the machine can offer is not locked at all (ledger_inputs); there is no lock to settle.
    if (c.moe.cache_discard && s.hw.discardable_memory) return;
    const bool cache_by_caller = s.req.is_pinned("cache-mb");
    const bool attempt = alloc.cache_locked_bytes > 0 || alloc.cache_lock_unverified;
    if (!s.req.is_pinned("cache-pin-fit") && !cache_by_caller) c.moe.cache_pin_fit = attempt;
    if (!s.req.is_pinned("cache-pin") && c.moe.cache_pin && !attempt && !cache_by_caller) {
        c.moe.cache_pin = false;
        s.note("cache-pin", "off", Source::Derived, "The cache is not locked: not even one token's experts could be.",
               "not even one token cycle of the cache can be locked in place here after what is ahead of it, so "
               "the cache is ordinary memory and no lock is asked for");
    }
}

// A cache the caller sized is the row, whatever the ledger would have chosen: the account is of what
// the run will hold, and a figure the run does not use would make it close on paper only.
void charge_caller_cache(Planning & s) {
    if (!s.req.is_pinned("cache-mb")) return;
    Ledger & l = s.p.allocation.ledger;
    const uint64_t asked = (uint64_t) std::max(0, s.p.config.moe.cache_mb) << 20;
    l.rows.erase(std::remove_if(l.rows.begin(), l.rows.end(),
                                [](const LedgerRow & r) { return r.kind == LedgerKind::ExpertCache; }),
                 l.rows.end());
    if (asked) l.rows.push_back({LedgerKind::ExpertCache, ledger_kind_name(LedgerKind::ExpertCache), asked, false});
    l.held = l.locked = 0;
    for (const LedgerRow & r : l.rows) {
        l.held += r.bytes;
        if (r.locked) l.locked += r.bytes;
    }
    if (l.held > l.holdable_cap)
        s.note("memory-ledger", "over", Source::Operator,
               "Your cache size brings the run to " + size_brief(l.held) + ", over the " + size_brief(l.holdable_cap) +
                   " it may hold.",
               "the cache the caller set takes what is held to " + u64s(mib(l.held)) + " MiB, over the " +
                   u64s(mib(l.holdable_cap)) + " MiB this process may hold: charged as asked");
}

// The ledger as charged, a row at a time, and the cost model's own notes.
void note_ledger(Planning & s, const Allocation & alloc) {
    const Ledger & ledger = s.p.allocation.ledger;
    std::string lines;
    for (const LedgerRow & r : ledger.rows)
        lines += (lines.empty() ? "" : ", ") + r.name + " " + u64s(mib(r.bytes)) + (r.locked ? " locked" : "");
    const std::string lock_cap = ledger.lockable_cap_known ? u64s(mib(ledger.lockable_cap)) + " MiB" : "unmeasured";
    s.note("memory-ledger", u64s(mib(ledger.held)) + " MiB", Source::Derived,
           "This run holds " + size_brief(ledger.held) + " of the " + size_brief(ledger.holdable_cap) + " it may use.",
           "held " + u64s(mib(ledger.held)) + " of " + u64s(mib(ledger.holdable_cap)) +
               " MiB holdable, of which locked " + u64s(mib(ledger.locked)) + " MiB of " + lock_cap +
               " lockable: " + lines);

    // Each of the cost model's notes under its own name: a column of identical "cost" rows is a list
    // nobody can tell apart.
    for (const AllocationNote & n : alloc.notes)
        s.note(n.topic.c_str(), n.value, Source::Derived, n.brief, n.detail);
}

// Where the cache's credited hit rate came from. With the model's own routing it is a measurement,
// and says so; a trace that was offered and did not fit says why the plan fell back to the linear
// credit. Without a trace there is nothing to say, and the plan is the one it always was.
void note_hit_curve(Planning & s, const Allocation & alloc) {
    const HitCurve & curve = s.model.hit_curve;
    if (curve.measured && alloc.cache_bytes > 0) {
        char value[16];
        std::snprintf(value, sizeof(value), "%.0f%%", 100.0 * alloc.cache_hit_credited);
        const double linear = s.host_expert_bytes
                                  ? std::min(1.0, (double) s.pol.cache_hit_optimism * (double) alloc.cache_bytes /
                                                      (double) s.host_expert_bytes)
                                  : 0.0;
        char linear_pct[16];
        std::snprintf(linear_pct, sizeof(linear_pct), "%.0f%%", 100.0 * linear);
        s.note("cache-hits", value, Source::Measured,
               "Cache hits are priced from this model's own routing: " + std::string(value) + " at " +
                   size_brief(alloc.cache_bytes) + ".",
               "measured on this model: an LRU replay of the routing it recorded on an earlier run (" +
                   u64s(curve.decode_tokens) + " decoded and " + u64s(curve.prefill_tokens) +
                   " prompt tokens) at every cache size up to the whole expert set; at " +
                   u64s(mib(alloc.cache_bytes)) + " MiB it serves " + value +
                   " of the decode lookups, where the linear credit would have said " + linear_pct +
                   (s.hit_scale != 1.0 ? std::string("; scaled to the hits a load measured here (decode-cost)")
                                       : std::string()));
    } else if (curve.prior && alloc.cache_bytes > 0 && s.host_model.token_cycle_bytes > 0) {
        char value[16];
        std::snprintf(value, sizeof(value), "%.0f%%", 100.0 * alloc.cache_hit_credited);
        const double cycles = (double) alloc.cache_bytes / (double) s.host_model.token_cycle_bytes;
        char nums[64];
        std::snprintf(nums, sizeof(nums), "%.1f", cycles);
        s.note("cache-hits", value, Source::Policy,
               "Cache hits are estimated at " + std::string(value) + " until this model's own routing is recorded.",
               "no routing record of this model" +
                   (curve.rejected.empty() ? std::string() : " fits (" + curve.rejected + ")") +
                   ", so the cache is credited the prior every model starts from: hits rise with the cache counted "
                   "in token cycles (the bytes one token routes through), where the measured models agree, toward "
                   "a ceiling the cold misses keep; " +
                   size_brief(alloc.cache_bytes) + " is " + nums + " token cycles here" +
                   (s.hit_scale != 1.0 ? std::string("; scaled to the hits a load measured here (decode-cost)")
                                       : std::string()));
    } else if (!curve.rejected.empty()) {
        s.note("cache-hits", "linear", Source::Policy,
               "Cache hits are credited linearly: the stored routing is not this run's.",
               "a routing trace was offered and not used (" + curve.rejected +
                   "), so the cache is credited linearly in the share of the expert set it holds, as on a first "
                   "load");
    }
}

// Name the one lever that closes the gap, and do not pull it. The KV reservation is the cheapest
// memory on this machine to give back, and the first stage shrinks the context only until the model
// FITS - which can stop well short of where streaming becomes possible. Measured: a 26B model with a
// 901 MiB token cycle was left 717 MiB at the context the fitter settled on, and 939 MiB at half of
// it, which is the difference between declining and running. But a context length is a capability
// the caller chose, not a speed knob, so a planner that quietly halved it would be changing what the
// run can DO - the same reason nothing lossy arms itself here. It gets named instead: a sentence to
// append to the reason, or nothing where a shorter context would not close the gap.
std::string context_suggestion(const Placement & placement, uint64_t cycle, uint64_t available) {
    const uint64_t kv = placement.raw_host_context_bytes;
    // From what was available BEFORE the floor zeroed it, or the shortfall is the whole floor.
    const uint64_t shortfall = cycle > available ? cycle - available : 0;
    if (kv == 0 || placement.n_ctx == 0 || shortfall == 0 || shortfall >= kv) return "";
    const double keep = 1.0 - (double) shortfall / (double) kv;
    const uint64_t suggest = (uint64_t) ((double) placement.n_ctx * keep);
    // Down to the next power of two: a context is a number a person types, and one that clears the
    // floor by arithmetic but not in practice would be worse than no suggestion. Nothing is named
    // where the arithmetic asks for less than the smallest context worth suggesting: rounding up to
    // it would name a context that frees less than the cache is short of.
    if (suggest < k_min_suggested_ctx) return "";
    uint64_t pow2 = k_min_suggested_ctx;
    while (pow2 * 2 <= suggest)
        pow2 *= 2;
    if (pow2 >= (uint64_t) placement.n_ctx) return "";
    return ". The context reserves " + u64s(mib(kv)) + " MiB here and the cache is " + u64s(mib(shortfall)) +
           " MiB short, so " + flag_spelling("ctx-size", u64s(pow2)) +
           " would clear the floor and keep a cache: the first stage shrinks the context only until the model fits, "
           "which is not the same as until it runs well";
}

// ── the cache budget, and the floor that is the model's own arithmetic ──────────
// Turns streaming on with the cache the ledger sized, or with none where no cache clears the floor.
// False when the plan has declined instead and is complete.
bool size_cache(Planning & s, const Allocation & alloc) {
    Plan & p = s.p;
    const uint64_t cycle = s.host_model.token_cycle_bytes;
    p.token_cycle_bytes = cycle;

    const uint64_t budget = alloc.cache_bytes;
    p.cache_budget_bytes = budget;

    if (cycle == 0) {
        s.decline("the model's expert shapes could not be read, so the cache floor could not be derived");
        price_unstreamed(p.allocation);
        s.note("moe-stream", "off", Source::Unprobed, "Not streamed: the experts' sizes could not be read.",
               "expert slice size unknown");
        return false;
    }
    // A budget under the floor takes the CACHE away, not the stream. The two were one decision here,
    // and the plan fell back to the mapped file whenever the cache could not be afforded - which is
    // the machine where the mapped file is at its worst, since the memory a cache lacks is the
    // memory the page cache lacks too. Measured on a 16 GB machine with a 22 GB model whose cycle is
    // 581 MiB and 536 MiB left for it: declined and mapped, 0.26 tok/s; streamed with no cache, 7.4
    // tok/s. The same model with that 536 MiB cache forced on returned 0.0% hits at 6.3 tok/s against
    // 8.6 with none, so the floor stands and it is the cache alone that it turns off. Whether the
    // uncached stream is worth running is the ledger's own price for it against the mapping, not a
    // rule of this file: where the read rate was never measured there is no such price, and the plan
    // declines as before.
    const GroupPlacement & experts = alloc.at(WeightGroup::Experts);
    bool cache_off = false;
    std::string cache_off_why;

    if (budget < cycle && s.req.is_pinned("cache-mb")) {
        s.note("cache-floor", u64s(mib(cycle)) + " MiB", Source::Derived,
               "Your cache is under the " + size_brief(cycle) + " one token reads; kept as you set it.",
               "the derived budget (" + u64s(mib(budget)) +
                   " MiB) is under this model's token cycle, but the caller "
                   "pinned cache-mb and a pin is the caller's authority: proceeding with the pinned value");
    } else if (budget < cycle) {
        std::string why = "the " + u64s(mib(alloc.cache_available_bytes)) +
                          " MiB left for the expert cache is below this model's worst-case token cycle of " +
                          u64s(mib(cycle)) +
                          " MiB: under that floor the cache evicts what the same token still needs, "
                          "so it costs its memory and returns no hits at all";
        why += context_suggestion(s.placement, cycle, alloc.cache_available_bytes);
        if (!stream_priced(alloc)) {
            s.decline(why + ". This storage's read rate was not measured, so a stream with no cache has no price "
                            "to set against the mapping and is not chosen on faith");
            price_unstreamed(p.allocation);
            s.note("moe-stream", "off", Source::Derived,
                   "Not streamed: no room for a useful cache, and storage speed is unknown.",
                   "budget below the derived cache floor, and no price to stream by");
            return false;
        }
        cache_off = true;
        cache_off_why = std::move(why);
    }

    p.config.moe.enabled = true;
    if (cache_off) {
        char b[160];
        std::snprintf(b, sizeof(b), "%.0f ms/token read as slices against %.0f ms/token left mapped",
                      1e3 * experts.seconds_per_token, 1e3 * experts.seconds_worst_case);
        s.note("moe-stream", "on", Source::Derived,
               "Experts stream with no cache: there is no room for one that would help.",
               "the experts do not fit and no cache clears the floor, so every routed slice is read on every token: " +
                   std::string(std::isfinite(experts.seconds_worst_case)
                                   ? b
                                   : "priced from this storage's measured rate, where the mapping's refault was not "
                                     "measured at all") +
                   ". The cache is what is given up, not the stream");
    } else {
        s.note("moe-stream", "on", Source::Derived,
               p.regime == Regime::ExpertsStream ? "Experts stream from storage; everything else stays in memory."
                                                 : "Experts stream, and the dense weights fit only in part.",
               p.regime == Regime::ExpertsStream
                   ? "the experts do not fit but the dense set does"
                   : "neither set fits, so the experts stream and the dense policy degrades with them "
                     "(a tensor larger than this budget is left mapped rather than made resident)");
    }

    if (cache_off) {
        // Off outright, never left to size itself at load: an automatic budget would land in the band
        // the floor has just refused.
        p.config.moe.cache_auto = false;
        p.config.moe.cache_mb = 0;
        s.note("cache-mb", "0", Source::Derived,
               "No cache: " + size_brief(alloc.cache_available_bytes) + " is left, under the " + size_brief(cycle) +
                   " one token reads.",
               cache_off_why);
    } else if (!s.pinned("cache-mb", u64s((uint64_t) std::max(0, p.config.moe.cache_mb)))) {
        p.config.moe.cache_auto = false;
        p.config.moe.cache_mb = (int) mib(budget);
        s.note("cache-mb", u64s(mib(budget)), Source::Derived,
               size_brief(budget) + ": what the cache earns most with, after the dense weights.",
               "what the cache is worth out of " + u64s(mib(s.usable)) +
                   " MiB usable, ranked against the resident groups by seconds saved per byte of RAM, with "
                   "what none of them wanted added: " +
                   u64s(mib(alloc.resident_bytes)) + " MiB committed in total, of which " + u64s(mib(budget)) +
                   " MiB is cache, over a floor of " + u64s(mib(cycle)) + " MiB");

        // The generic 1500 MiB guard predates the per-model floor and is the weaker of the two: a
        // budget above this model's own token cycle is not pathological however small it looks.
        if (p.config.moe.cache_mb > 0 && p.config.moe.cache_mb < MoeStreamConfig::cache_min_mb) {
            p.config.moe.force_cache = true;
            s.note("force-cache", "on", Source::Derived,
                   "Allows this cache below the engine's usual " +
                       size_brief((uint64_t) MoeStreamConfig::cache_min_mb << 20) + " minimum: it still helps.",
                   "the budget clears this model's token cycle of " + u64s(mib(cycle)) +
                       " MiB but sits under the generic " + u64s(MoeStreamConfig::cache_min_mb) +
                       " MiB guard, so the derived floor is used as the authority");
        }
    }
    return true;
}

// ── the KV, as a budget line rather than a silent subtraction ───────────────────
// Context memory competes with the expert cache for the same bytes, and until now it only appeared
// inside the placement's rationale. On a long context it is the larger of the two, and a reader
// deciding whether to shorten the prompt deserves to see the trade rather than infer it. Nothing is
// DECIDED here: llama.cpp's own defaults already pick flash attention when the build supports it,
// which costs no quality, and the K/V caches' type is the engine's default or the caller's (lossy
// either way below f16), never a plan's.
void note_kv_budget(Planning & s) {
    const Placement & placement = s.placement;
    if (!placement.fitted || placement.raw_host_context_bytes == 0) return;
    const uint64_t kv = placement.raw_host_context_bytes;
    const uint64_t compute = placement.raw_host_compute_bytes;
    const std::string types =
        std::string(kv_cache_type_name(s.p.config.cache_type_k)) + "/" + kv_cache_type_name(s.p.config.cache_type_v);
    s.note("kv-budget", u64s(mib(kv)) + " MiB", Source::Measured,
           "The context takes " + size_brief(kv) + " (K/V " + types + "); a shorter one would give that to the cache.",
           "the context reserves " + u64s(mib(kv)) + " MiB on the host at K/V " + types +
               " and its compute buffers another " + u64s(mib(compute)) + ", against " +
               u64s(mib(s.p.cache_budget_bytes)) +
               " MiB left for the expert cache: the two come out of the same memory, and a shorter context "
               "is the caller's cheapest way to buy cache. The cache type is set by {cache-type-k} and "
               "{cache-type-v}, never by this plan: f16 is lossless, q8_0 (the default) halves the bytes");
}

// The checkpoints' charge, said where it was taken, so a reader who sees a smaller cache on a chat
// run than on a one-shot one finds why. Measured on a fitted placement and an unfitted one alike: the
// fixed part is read off the fitter's projection either way.
void note_checkpoint_budget(Planning & s) {
    if (s.checkpoint_bytes == 0) return;
    const RunConfig & c = s.p.config;
    s.note("checkpoint-budget", u64s(mib(s.checkpoint_bytes)) + " MiB", Source::Measured,
           "Context checkpoints may take " + size_brief(s.checkpoint_bytes) + "; that is kept out of the cache.",
           std::to_string(c.ctx_checkpoints) + " checkpoints ({ctx-checkpoints}) of " +
               u64s(mib(s.placement.context_fixed_bytes)) + " MiB each, " +
               (s.placement.n_swa && (s.placement.host_window_cell_bytes || s.placement.device_window_cell_bytes)
                    ? "the model's sliding window of " + u64s(s.placement.n_swa) +
                          " positions at what a cell of it costs (measured at two graph widths)"
                    : std::string("the part of the context that does not grow with its length (measured at two "
                                  "lengths)")) +
               ", charged before the cache is sized: a chat run fills them as it goes, and memory the cache had "
               "already taken would have to come out of something else");
}

// ── the I/O policy: from the machine's own rate curve, never from a class name ──
void choose_io_lanes(Planning & s) {
    MoeStreamConfig & moe = s.p.config.moe;
    if (s.pinned("io-threads", u64s((uint64_t) moe.io_threads))) return;
    const uint32_t want = s.hw.storage.best_lanes((uint32_t) s.model.expert_slice_bytes);
    if (want > 0) {
        const int lanes = std::min<int>((int) want, MoeStreamConfig::io_threads_max);
        moe.io_threads = std::max(1, lanes);
        s.note("io-threads", u64s((uint64_t) moe.io_threads), Source::Measured,
               u64s((uint64_t) moe.io_threads) + " parallel reads: the fastest rate this storage gave.",
               "the highest measured rate for this model's " + u64s(s.model.expert_slice_bytes >> 10) +
                   " KiB expert slice on this storage");
    } else {
        s.note("io-threads", u64s((uint64_t) moe.io_threads), Source::Unprobed,
               u64s((uint64_t) moe.io_threads) + " parallel reads, the default: storage speed was not measured.",
               "no read-rate curve for this machine; keeping the default lane count");
    }
}

void choose_direct_io(Planning & s) {
    MoeStreamConfig & moe = s.p.config.moe;
    if (s.pinned("o-direct", moe.o_direct ? "on" : "off")) return;
    if (s.hw.storage.direct_ok == Tri::Yes) {
        moe.o_direct = true;
        s.note("o-direct", "on", Source::Measured, "Reads bypass the page cache: checked correct on this storage.",
               "uncached reads return correct bytes on this path");
    } else if (s.hw.storage.direct_ok == Tri::No) {
        moe.o_direct = false;
        s.note("o-direct", "off", Source::Measured, "Reads go through the page cache: uncached reads came back wrong.",
               "uncached reads do not return correct bytes on this path, so they are not used");
    } else {
        s.note("o-direct", moe.o_direct ? "on" : "off", Source::Unprobed,
               "Unchecked here: the reader tests uncached reads itself and falls back.",
               "unverified here; the reader checks at open and falls back to buffered reads by itself");
    }
}

void choose_release_mmap(Planning & s) {
    MoeStreamConfig & moe = s.p.config.moe;
    const StorageFacts & storage = s.hw.storage;
    if (s.pinned("release-mmap", moe.release_mmap ? "on" : "off")) return;
    // Carrying a multi-token-prediction block is not the same as using one: llama.cpp does not load
    // that block unless the run asks for it, and an unloaded block holds no pointer into the
    // mapping. Testing the file instead of the run would silently disable this on every model that
    // merely ships an MTP head, which is most of them.
    const bool mtp_in_use = s.model.has_mtp && s.p.config.spec.is_mtp();
    // A table the plan leaves mapped (dense_mapped) is read through the mapping for the whole run,
    // so it is a live pointer like a tied head: the release would have nothing left to drop but the
    // pages that table is gathered from.
    const bool safe_shape = !s.model.tied_output_head && !mtp_in_use && moe.dense_weights != DenseWeightsMode::Mmap &&
                            moe.dense_mapped.empty();
    // Safety is reported before the probe, because it is the stronger constraint: on a shape that
    // keeps a live pointer into the mapping, measuring the storage would change nothing.
    if (!safe_shape) {
        s.note("release-mmap", "off", Source::Derived, "The model's file mapping stays: this run still reads from it.",
               "this run keeps a live pointer into the mapping (a tied head, a draft block, a dense "
               "policy that stays mapped, or a table left mapped), so the mapping cannot be released whatever "
               "the storage does");
    } else if (storage.mapping_serialises_reads == Tri::Yes) {
        moe.release_mmap = true;
        s.note("release-mmap", "on", Source::Measured,
               "The file mapping is dropped after load: it slowed reads to " +
                   u64s((uint64_t) storage.rate_mapped_mibs) + " from " + u64s((uint64_t) storage.rate_unmapped_mibs) +
                   " MiB/s.",
               "a live mapping of the model serialises this storage's concurrent uncached reads (" +
                   u64s((uint64_t) storage.rate_mapped_mibs) + " vs " + u64s((uint64_t) storage.rate_unmapped_mibs) +
                   " MiB/s), and this model's shape keeps no pointer that would survive the release "
                   "(the session checks again before doing it)");
    } else {
        s.note("release-mmap", moe.release_mmap ? "on" : "off", Source::Unprobed,
               storage.rate_unmapped_mibs > 0.0
                   ? "The file mapping stays: no slowdown was seen, though the test can miss one."
                   : "The file mapping stays: whether it slows reads was not measured.",
               storage.rate_unmapped_mibs > 0.0
                   ? "the storage probe did not see a live mapping slow its reads here (" +
                         u64s((uint64_t) storage.rate_mapped_mibs) + " vs " +
                         u64s((uint64_t) storage.rate_unmapped_mibs) +
                         " MiB/s), but it is an instrument validated only in the positive direction, so "
                         "that is not evidence there is nothing to gain: try " +
                         flag_spelling("release-mmap", "true") + " and measure"
                   : "whether a live mapping serialises reads was not measured on this machine");
    }
}

// ── everything lossy: the caller's authority, never ours ────────────────────────
void disarm_lossy(Planning & s) {
    MoeStreamConfig & moe = s.p.config.moe;
    const bool lossy_pinned =
        s.req.is_pinned("drop-cold-experts") || s.req.is_pinned("expert-substitute") || s.req.is_pinned("route-ahead");
    if (lossy_pinned) {
        s.note("lossy", "as set", Source::Operator, "A setting that changes the output is on, as you set it.",
               "the caller armed a lossy policy explicitly");
        return;
    }
    moe.drop_cold_frac = 0.0f;
    moe.substitute_lambda = 0.0f;
    moe.route_ahead = 0;
    if (s.req.quality_budget <= 0.0f)
        s.note("lossy", "off", Source::Policy, "Nothing that changes the output is turned on.",
               "no quality budget was given, and a policy that changes the output must never arm itself");
    else
        s.note("lossy", "off", Source::Unprobed,
               "Nothing that changes the output is turned on: its cost is not measured yet.",
               "a quality budget was given, but no measured mapping from a perplexity cost to a lever value "
               "exists yet, so spending it would be a guess");
}

// ── route-ahead's gate GEMV: the int8 mirror or the float weights ───────────────
// The mirror reads a quarter of the bytes and pays a quantization of the activation per call, so it
// wins where bytes are scarce and loses where they are not - measured both ways, on a phone and on
// an x86 host. That is a bandwidth-against-arithmetic fact of the machine, timed by the probe on the
// run's own kernels and the model's own gate shape; the faster path is taken, a tie to the mirror,
// because the probe ran alone and a decode beside it only makes bytes dearer. Unmeasured, the knob
// stays auto and the session times it at load the same way.
std::string gate_mirror_name(GateMirror m) {
    return m == GateMirror::On ? "on" : m == GateMirror::Off ? "off" : "auto";
}

std::string us_text(double us) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), us < 10.0 ? "%.1f" : "%.0f", us);
    return buf;
}

void choose_gate_mirror(Planning & s) {
    MoeStreamConfig & moe = s.p.config.moe;
    if (moe.route_ahead <= 0) return; // nothing reads it: no decision to explain
    if (s.pinned("gate-mirror", gate_mirror_name(moe.gate_mirror))) return;
    const double f = s.hw.gate_gemv_f32_us, q = s.hw.gate_gemv_int8_us;
    if (f > 0.0 && q > 0.0) {
        const bool mirror = q <= f;
        moe.gate_mirror = mirror ? GateMirror::On : GateMirror::Off;
        const std::string times = us_text(q) + " us int8 against " + us_text(f) + " us float per layer";
        s.note("gate-mirror", gate_mirror_name(moe.gate_mirror), Source::Measured,
               mirror ? "Route-ahead ranks experts on int8 gates: faster on this machine."
                      : "Route-ahead ranks experts on the gates' own weights: faster on this machine.",
               "the prediction's gate GEMV, timed both ways on this model's gate shape: " + times +
                   (mirror ? "; a quarter of the bytes buys more here than quantizing the activation costs"
                           : "; bytes are cheap enough here that quantizing the activation costs more than "
                             "the mirror saves"));
    } else {
        s.note("gate-mirror", gate_mirror_name(moe.gate_mirror), Source::Unprobed,
               "Route-ahead's gate format is chosen when the model loads.",
               "the gate GEMV was not timed by this plan, so the session times both forms at load and keeps "
               "the faster");
    }
}

// ── which devices the run lists ─────────────────────────────────────────────────
// A device that did not survive its probe stays out of the run, whatever else is decided. This is
// not a preference: what ended the probe - a driver without a function the backend calls, a build
// with no code for that device, a device that never answers - would end the run the same way, at
// load. The devices that did survive carry what the fitter placed on them, and the run lists them by
// name, so llama.cpp's default list does not hand the failed one back. With nothing placed on them,
// the host carries it, unless a different device was armed for the prefill, which is then the only
// one listed.
//
// And where nothing is going on a device, no device should be in the graph. It is not enough to
// decline a placement: a registered backend collects work on its own, and every piece it collects
// is a crossing. Measured with zero layers placed - 61 splits a token, and a build carrying the
// backend losing to one without it. A caller who wants the device anyway pins the knob.
// Returns whether a device failed, which settles the knob. Asked on every path, the resident one
// too: there the fitter's placement is the whole plan, and an unnamed list would load the failed
// device beside the layers.
bool leave_out_failed_devices(Planning & s) {
    RunConfig & c = s.p.config;
    std::string broken;
    for (const ComputeDevice & d : s.hw.devices)
        if (!d.is_cpu && d.usable == Tri::No)
            broken += (broken.empty() ? "" : "; ") + d.name + ": " + d.unusable_reason;
    if (!broken.empty()) {
        bool dropped = false;
        const std::vector<std::string> alive = surviving_devices(s.hw, c.device_list, &dropped);
        if (s.req.is_pinned("devices")) {
            s.note("devices", "as set", Source::Operator, "Devices as you set them, though one failed its check.",
                   "set by the caller, and left alone - but a device here failed its probe (" + broken +
                       "), and a run that lists it may end the same way");
        } else if (!alive.empty() && (c.n_gpu_layers != 0 || c.dense_on_device)) {
            if (dropped) c.device_list = alive;
            std::string names;
            for (const std::string & n : alive)
                names += (names.empty() ? "" : ",") + n;
            s.note("devices", names, Source::Measured,
                   "A device that failed its check is left out; the others run the model.",
                   "a device was probed in a process of its own and did not survive it (" + broken +
                       "): it is kept out of the run, and the devices that survived keep what the fitter "
                       "placed on them");
        } else {
            if (!c.prefill.enabled()) c.device_use = DeviceUse::CpuOnly;
            c.n_gpu_layers = 0;
            c.device_split.clear();
            s.note("devices", c.prefill.enabled() ? "prefill device only" : "cpu only", Source::Measured,
                   "A device that failed its check is left out of the run.",
                   "a device was probed in a process of its own and did not survive it (" + broken +
                       "): it is kept out of the run, which is the only way the run starts");
        }
        return true;
    }
    return false;
}

void choose_devices(Planning & s) {
    if (leave_out_failed_devices(s)) return;
    RunConfig & c = s.p.config;
    if (s.req.is_pinned("devices") || c.n_gpu_layers != 0 || c.dense_on_device || c.prefill.enabled()) return;
    size_t non_cpu = 0;
    for (const ComputeDevice & d : s.hw.devices)
        if (!d.is_cpu) ++non_cpu;
    if (non_cpu == 0) return;
    c.device_use = DeviceUse::CpuOnly;
    s.note("devices", "cpu only", Source::Derived,
           "CPU only: no weight goes to the other " + std::string(non_cpu == 1 ? "device" : "devices") +
               ", so none is loaded.",
           u64s(non_cpu) + " device(s) registered and no weight going to any of them: a backend left in the graph "
                           "takes the nodes it can execute simply for being there, and each is a boundary crossed "
                           "twice. Measured at 61 splits a token with nothing placed");
}

// A placement the fitter made and this planner did not apply, because on shared memory its capacity
// arithmetic counts the same pool twice. Said out loud: "no layers on devices" on a machine that has
// one is a sentence a reader would otherwise take as "there is no device".
void note_shared_pool(Planning & s) {
    const Placement & placement = s.placement;
    if (s.device_on_host > 0)
        s.note("shared-pool", u64s(mib(s.device_on_host)) + " MiB", Source::Derived,
               "The " + size_brief(s.device_on_host) + " meant for the device is counted against this same memory.",
               "every device here reads this host's own memory, so what the fitter placed on one is in the "
               "same pool as what it left here and is charged to it: " +
                   u64s(mib(s.device_on_host)) + " MiB added to the " + u64s(mib(placement.host_resident_bytes)) +
                   " MiB host set before anything was sized, because the fitter counts the two as separate");

    if (placement.shared_memory_placement)
        s.note("placement-declined", u64s(mib(placement.device_bytes)) + " MiB", Source::Derived,
               "Not applied: placing " + size_brief(placement.device_bytes) +
                   " on the device would count this memory twice.",
               "the capacity fitter would have placed " + u64s(mib(placement.device_bytes)) +
                   " MiB on a device whose memory is this host's own, counting it as separate from the " +
                   u64s(mib(placement.host_resident_bytes)) +
                   " MiB it left here - the same pool, twice. Acting on that overcommits the pool; charging it "
                   "instead reads as a model twice its real size and declines a run that works. So the "
                   "placement is quoted and not applied: using such a device is a bandwidth decision, not a "
                   "capacity one");
}

// ── the dense set on a device that shares this memory ──────────────────────────
// The capacity tier said nothing here, so this is the bandwidth decision it left behind. The dense
// set is small - a fraction of the file - and the device reads this model's weights faster than the
// cores do, so computing it there is the whole of what an integrated accelerator can offer. It costs
// no capacity, because the memory is the same memory.
//
// Armed by the caller, not by this plan. What is missing is the device's own compute buffer as a
// budget line: on one phone it reserved 906 MiB against the CPU's 22, and a plan that sizes a cache
// without knowing that is a plan that takes the machine down - which it did, once.
void dense_on_shared_device(Planning & s) {
    RunConfig & c = s.p.config;
    const HardwareProfile & hw = s.hw;
    if (!s.devices_share_host_memory) return;
    if (!c.dense_on_device) {
        const ComputeDevice * fast = nullptr;
        for (const ComputeDevice & d : hw.devices) {
            if (d.is_cpu || d.runs_expert_op != Tri::Yes || d.needs_repack == Tri::Yes) continue;
            if (hw.host_bandwidth_gibs <= 0.0 || d.memory_bandwidth_gibs <= s.host_bandwidth_win) continue;
            fast = &d;
            break;
        }
        if (fast)
            s.note("dense-on-device", "off", Source::Unprobed,
                   "The dense weights could run on the device, faster; not turned on until it is priced.",
                   "this device reads the model's own weights at " + u64s((uint64_t) fast->memory_bandwidth_gibs) +
                       " GiB/s against the host's " + u64s((uint64_t) hw.host_bandwidth_gibs) +
                       ", and the dense set is " + u64s(mib(s.model.dense_bytes)) +
                       " MiB - small enough that computing it there costs no memory on a shared pool. Arm it with " +
                       flag_spelling("dense-on-device", "true") +
                       "; this plan will not, until the device's compute buffer is a budget line");
        return;
    }
    // Refused, and the refusal is the finding. Placing the dense set on a shared-memory device and
    // routing the experts back with an `exps` override loads, reserves, starts the streamer - and
    // then segmentation-faults, three times out of three on the phone it was tried on, while the
    // same run without it is fine. Standing the dense policy down (it would rebind a pointer it does
    // not own) moved the fault later rather than removing it, so something else in the streamed path
    // is still handed a device address. Until that is found this arms nothing: a lever that takes
    // the machine down is worse than one that does not exist.
    c.dense_on_device = false;
    s.note("dense-on-device", "refused", Source::Derived,
           "Refused: this path crashes once streaming starts, and it measured slower.",
           "refused, and twice over. This path faults once the expert streamer starts, reproducibly - the "
           "expert override here is a literal pattern where the retired GPU-offload flag built one from the "
           "model's expert tensors, which is a bug and a small one. The larger reason is that the same "
           "placement was measured three times on a device of this class and LOST: full offload -27% on a "
           "Q4_0 run, because the dense and expert halves interleave, so a two-device split crosses the "
           "boundary twice per layer - 96 splits against 1 - and the boundary tax eats the CPU time the "
           "device frees. The untested shape that could change it is contiguous layer BLOCKS, experts "
           "included and resident, which puts the boundary in one place");
}

// ── overlap: named, not armed ──────────────────────────────────────────────────
// Hiding compute behind the reads is not a quality choice - the output is byte-identical - and it is
// worth about 5% on the phone this was measured on, where a plan without it lost to a hand-written
// recipe that had it. It is still not armed, and the reason is the same one that withdrew the thread
// rule below: one machine's evidence is not a rule. What the plan can do honestly is say the knob
// exists and that the caller is the one who knows.
//
// So the answer is measured, on this model and this machine: the first real request alternates
// decode windows with it on and off (docs/measured-planning.md, Step 2), and a load after that starts
// from the verdict. Only where the trial can switch it - streaming, and every layer on the host,
// since with one on a device overlap changes how the context offloads and that is fixed at creation.
// The same condition holds a stored verdict: one measured on a host-only run says nothing about a
// placement that crosses to a device.
// A device-side expert cache, when there is one, is a second condition of the same kind: it refuses
// overlap, so neither the trial nor a verdict may arm it there.
bool overlap_switchable(const RunConfig & c) {
    return c.moe.enabled && c.n_gpu_layers == 0 && !c.moe.io_two_wave && !c.moe.prefetch_sync;
}

std::string gain_pct(double gain) {
    char b[16];
    std::snprintf(b, sizeof(b), "%+.0f%%", 100.0 * (gain - 1.0));
    return b;
}

std::string margin_pct(const Planning & s) {
    return u64s((uint64_t) (s.pol.min_backend_win * 100.0f + 0.5f)) + "%";
}

// With layers on a device and no device cache, overlap is what keeps a prompt on the host: it turns
// off llama.cpp's offload of host-held weights in a wide graph (offloads_wide_matmuls), so every
// such matmul of a prompt runs on the cores, where serial streaming lets the device take them. The
// placed layers still compute their own weights on the device; what is lost is the copy of the
// host's. Measured on a laptop with a discrete GPU and 25 of 41 layers placed: overlap cost 2.1x of
// the prefill on a 2.7k-token prompt and 4.5x on a 20.7k one, and bought no decode (#645). Read
// from the guard itself, so the plan names exactly the runs the session takes the offload from.
bool overlap_keeps_prompts_on_host(const RunConfig & c) {
    if (!c.moe.overlap || offloads_wide_matmuls(c)) return false;
    MoeStreamConfig serial = c.moe;
    serial.overlap = false;
    return offloads_wide_matmuls(serial, c.n_gpu_layers, c.prefill.enabled());
}

// The devices holding the run's layers, for a line that names them, and how much faster than the
// host the fastest of them measured on a wide matmul, where it was measured.
std::string placed_devices(const RunConfig & c) {
    std::string out;
    for (const auto & d : c.device_split)
        if (d.second > 0) out += (out.empty() ? "" : ", ") + d.first;
    return out.empty() ? std::string("the device") : out;
}

std::string placed_wide_ratio(const Planning & s) {
    const double host = s.hw.host_wide_matmul_gibs;
    double best = 0.0;
    for (const auto & split : s.p.config.device_split)
        for (const ComputeDevice & d : s.hw.devices)
            if (!d.is_cpu && d.name == split.first && split.second > 0) best = std::max(best, d.wide_matmul_gibs);
    if (host <= 0.0 || best <= 0.0) return "";
    char b[96];
    std::snprintf(b, sizeof(b), "; its wide matmuls measured %.1fx the host's", best / host);
    return b;
}

void note_overlap(Planning & s) {
    RunConfig & c = s.p.config;
    if (s.req.is_pinned("overlap")) {
        // The caller's setting stands, and the plan says what it costs them.
        if (overlap_keeps_prompts_on_host(c)) {
            const std::string devs = placed_devices(c);
            s.note("overlap", "on", Source::Operator,
                   "On, as you set it: prompts run on the host's cores, not on " + devs + ".",
                   "set by the caller; with layers on " + devs +
                       ", overlap turns off the offload of the host's weights in a prompt's wide matmuls, so they "
                       "run on the cores, where serial streaming lets " +
                       devs + " take them" + placed_wide_ratio(s) + ": pass " + flag_spelling("overlap", "false") +
                       " for serial streaming");
        }
        return;
    }
    if (c.moe.device_cache_mb > 0) return; // the device tier decided it, and said so
    // A front-end's own default overlap (the server arms it wherever the hook is available), with
    // layers on a device and no cache there: off, so the prompts reach the device. A caller who asked
    // for it never reaches here.
    if (overlap_keeps_prompts_on_host(c)) {
        const std::string devs = placed_devices(c);
        c.moe.overlap = c.moe.io_two_wave = false;
        c.trial.overlap = false;
        s.note("overlap", "off", Source::Derived,
               "Off: with layers on " + devs + ", overlap keeps prompts on the host.",
               "overlap turns off the offload of the host's weights in a prompt's wide matmuls, so with layers on " +
                   devs +
                   " and no expert cache there every one of them would run on the cores; serial streaming "
                   "lets " +
                   devs + " take them" + placed_wide_ratio(s) + ". Pass " + flag_spelling("overlap", "true") +
                   " to keep it");
        return;
    }
    const MeasuredKnobs & mk = s.model.knobs;
    if (mk.overlap >= 0 && overlap_switchable(c)) {
        const bool on = mk.overlap == 1;
        const bool switched = mk.overlap_incumbent >= 0 && mk.overlap_incumbent != mk.overlap;
        const char * value = on ? "on" : "off";
        const char * other = on ? "off" : "on";
        c.moe.overlap = on;
        s.note("overlap", value, Source::Measured,
               std::string(on ? "On" : "Off") + ": measured on this model, " +
                   (switched ? "it decoded " + gain_pct(mk.overlap_gain) + " against " + other + "."
                             : std::string(other) + " was no faster here (" + gain_pct(mk.overlap_gain) + ")."),
               "measured on this model: an earlier load's first answer decoded in alternating windows with overlap "
               "on and off, " +
                   u64s((uint64_t) std::max(0, mk.rounds)) +
                   " rounds of the one tried, each between two of the other; " +
                   (switched ? std::string(value) + " was " + gain_pct(mk.overlap_gain) + " against " + other +
                                   ", past the " + margin_pct(s) + " margin"
                             : std::string(other) + " was " + gain_pct(mk.overlap_gain) + ", inside the " +
                                   margin_pct(s) + " margin, so " + value + " held"));
        return;
    }
    // The trial is symmetric: a caller whose default is on (a server that arms it wherever the hook
    // exists) has it measured against off, by the same margin.
    if (s.req.measure_knobs && overlap_switchable(c) && !s.req.is_pinned("trial-overlap")) {
        c.trial.overlap = true;
        const bool on = c.moe.overlap;
        s.note("overlap", on ? "on" : "off", Source::Unprobed,
               std::string(on ? "On" : "Off") + " until the first answer measures it against " + (on ? "off." : "on."),
               "hiding compute behind the reads changes no output and has measured faster, but one machine is not a "
               "rule: the first real request alternates decode windows with it on and off once the cache has "
               "settled, and switches it only if the other setting decodes " +
                   margin_pct(s) + " faster; the next load starts from the verdict");
        return;
    }
    if (c.moe.overlap) return;
    s.note("overlap", "off", Source::Unprobed, "Off: it changes no output, but its gain here is unmeasured.",
           "hiding compute behind the reads changes no output and has measured faster, but one machine is not "
           "a rule and nothing here predicts where it stops paying: pass " +
               flag_spelling("overlap", "true"));
}

// The rate the thread sweep measured at `n` threads, 0 when it did not ask that count.
double swept_rate(const HardwareProfile & hw, int n) {
    for (const HardwareProfile::ThreadRate & r : hw.thread_rates)
        if ((int) r.threads == n) return r.gibs;
    return 0.0;
}

// The thread counts a trial tries against the incumbent: where a barrier would start or stop
// waiting on a slower class (the running sums of the core classes), every core, and half of them
// where all are alike (a machine with two hardware threads a core reports both). The ones nearest
// the incumbent first, as many as the trial takes.
//
// Less a count the sweep already answered: one above the incumbent that the sweep measured short of
// the trial's margin over it (`min_win`). The sweep ran it with every core to itself; a decode runs
// it beside the expert reads, which take cores of their own, so more threads only meet more
// contention there, and a count that could not win alone cannot win under load. Measured on a
// phone: eight threads against four, every core, ran each decoded token about twenty times slower.
// A count below the incumbent stays: what the reads take from the cores is what the sweep cannot see,
// and leaving them cores is what such a count tries. `skipped` gets the counts left out.
std::vector<int>
thread_candidates(const HardwareProfile & hw, int incumbent, double min_win, std::vector<int> & skipped) {
    std::vector<int> pool;
    uint32_t sum = 0;
    for (uint32_t k : hw.core_classes) {
        sum += k;
        pool.push_back((int) sum);
    }
    if (hw.n_cores) pool.push_back((int) hw.n_cores);
    if (hw.core_classes.size() <= 1 && hw.n_cores >= 2) pool.push_back((int) hw.n_cores / 2);
    std::sort(pool.begin(), pool.end());
    pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
    pool.erase(std::remove_if(pool.begin(), pool.end(),
                              [&](int n) {
                                  // A class sum past the cores this process may use is a count the
                                  // quota or the affinity mask would only time-slice.
                                  return n < 1 || n > RunConfig::threads_max || n == incumbent ||
                                         (hw.n_cores > 0 && n > (int) hw.n_cores);
                              }),
               pool.end());
    const double at_incumbent = swept_rate(hw, incumbent);
    pool.erase(std::remove_if(pool.begin(), pool.end(),
                              [&](int n) {
                                  const double rate = swept_rate(hw, n);
                                  const bool answered = n > incumbent && at_incumbent > 0.0 && rate > 0.0 &&
                                                        rate < at_incumbent * (1.0 + min_win);
                                  if (answered) skipped.push_back(n);
                                  return answered;
                              }),
               pool.end());
    std::stable_sort(pool.begin(), pool.end(),
                     [&](int a, int b) { return std::abs(a - incumbent) < std::abs(b - incumbent); });
    if ((int) pool.size() > KnobTrialConfig::threads_max_candidates)
        pool.resize((size_t) KnobTrialConfig::threads_max_candidates);
    return pool;
}

std::string thread_list(const std::vector<int> & v, const char * last_sep) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i)
        out += (i == 0 ? "" : i + 1 == v.size() ? last_sep : ", ") + std::to_string(v[i]);
    return out;
}

// A count an earlier load measured on this model. Prefill is a different load, compute-bound where
// a streamed decode is not, and was not part of the measurement: it keeps the count it would have had.
void apply_measured_threads(Planning & s, int would_have) {
    RunConfig & c = s.p.config;
    const MeasuredKnobs & mk = s.model.knobs;
    c.n_threads = mk.threads;
    std::string prefill;
    if (c.n_threads_batch == 0 && would_have != mk.threads && !s.req.is_pinned("threads-batch")) {
        c.n_threads_batch = would_have;
        prefill = "; prefill keeps " + u64s((uint64_t) would_have) + ", which the trial did not measure";
    }
    std::vector<int> others;
    for (int n : mk.threads_tried)
        if (n != mk.threads) others.push_back(n);
    const bool held = !mk.threads_tried.empty() && mk.threads_tried.front() == mk.threads;
    const std::string n = u64s((uint64_t) mk.threads);
    s.note("threads", n, Source::Measured,
           n + " threads: measured on this model, the fastest decode " +
               (others.empty() ? std::string("here.") : "against " + thread_list(others, " and ") + "."),
           "measured on this model: an earlier load's first answer decoded in alternating windows with " +
               thread_list(mk.threads_tried, " and ") + " threads, " + u64s((uint64_t) std::max(0, mk.rounds)) +
               " rounds each between two of the incumbent's; " +
               (held ? "the best other count was " + gain_pct(mk.threads_gain) + ", inside the " + margin_pct(s) +
                           " margin, so " + n + " held"
                     : n + " was " + gain_pct(mk.threads_gain) + " against " +
                           u64s((uint64_t) mk.threads_tried.front()) + ", past the " + margin_pct(s) + " margin") +
               prefill);
}

// The trial of the count, armed when the front-end keeps its verdict and nothing measured it yet.
void arm_thread_trial(Planning & s) {
    RunConfig & c = s.p.config;
    if (!s.req.measure_knobs || s.req.is_pinned("trial-threads") || !c.trial.threads.empty()) return;
    std::vector<int> skipped;
    const std::vector<int> cand = thread_candidates(s.hw, c.n_threads, s.pol.min_backend_win, skipped);
    // What the sweep already answered, said in the note that follows or, with nothing left to try, alone.
    std::string answered;
    if (!skipped.empty())
        answered = "; " + thread_list(skipped, " and ") + (skipped.size() == 1 ? " is" : " are") +
                   " not tried: the sweep measured " + (skipped.size() == 1 ? "it" : "them") + " short of " +
                   margin_pct(s) + " over " + u64s((uint64_t) c.n_threads) +
                   " with every core to itself, and a decode's expert reads only take cores away";
    if (cand.empty()) {
        if (!skipped.empty())
            s.note("trial-threads", "none", Source::Measured,
                   "The first answer measures no other thread count: the sweep already answered.",
                   "the counts a trial would try against " + u64s((uint64_t) c.n_threads) + " are answered" + answered);
        return;
    }
    c.trial.threads = cand;
    std::string value;
    for (size_t i = 0; i < cand.size(); ++i)
        value += (i ? "," : "") + std::to_string(cand[i]);
    s.note("trial-threads", value, Source::Policy,
           "The first answer measures " + thread_list(cand, " and ") + " threads against " +
               u64s((uint64_t) c.n_threads) + ".",
           "the thread count changes no output and which one decodes fastest is this machine's answer, not a "
           "rule's: once the cache has settled the first real request alternates decode windows between " +
               u64s((uint64_t) c.n_threads) + " and " + thread_list(cand, " and ") +
               " (where a slower class of core starts, every core, or half of them where all are alike), and "
               "switches only to a count " +
               margin_pct(s) + " faster in the median of its rounds; the next load starts from the verdict" + answered);
}

// ── threads: the classes are a fact, the rule that read them was wrong ─────────
// The rule here used to set the count to the fast class, reasoning that every thread meets the same
// barrier so a thread on a slower core sets the pace rather than adding to it. Sound, and refuted by
// the first heterogeneous machine it met: a phone with two prime cores and six others got two
// threads instead of four, and decode compute went from 0.127 to 0.195 s/token - 4.24 tok/s down to
// 2.68. Losing half the threads costs more than the barrier's imbalance does, and nothing here knows
// where that trade turns over.
//
// So the classes stay as a measured fact, printed because they are worth knowing, and the knob keeps
// its default. Deriving a count from them would need a thread sweep on the machine itself, which is
// a probe this does not have. The same applies to prefill: it is compute-bound and plausibly wants
// every core, and "plausibly" is exactly what this planner does not ship.
void choose_threads_unmeasured(Planning & s);
void choose_threads(Planning & s) {
    const HardwareProfile & hw = s.hw;
    RunConfig & c = s.p.config;
    if (s.req.is_pinned("threads")) return;
    if (s.model.knobs.threads > 0) {
        apply_measured_threads(s, hw.best_threads > 0 ? (int) hw.best_threads : c.n_threads);
        return;
    }
    choose_threads_unmeasured(s);
    arm_thread_trial(s);
}

void choose_threads_unmeasured(Planning & s) {
    const HardwareProfile & hw = s.hw;
    RunConfig & c = s.p.config;
    if (hw.best_threads > 0) {
        c.n_threads = (int) hw.best_threads;
        s.note("threads", u64s(hw.best_threads), Source::Measured,
               u64s(hw.best_threads) + " threads: the fastest count measured on this model's format.",
               "swept over the thread counts this machine has, on a matmul in this model's own quantized format, "
               "which is the material that matters: a sweep in F32 can pick a count the quantized kernels are "
               "much slower at");
    } else if (hw.core_classes.size() > 1) {
        std::string shape;
        for (size_t k = 0; k < hw.core_classes.size(); ++k)
            shape += (k ? " + " : "") + u64s(hw.core_classes[k]);
        s.note("threads", u64s((uint64_t) c.n_threads), Source::Unprobed,
               "The default thread count: the cores differ, and no sweep has measured them.",
               "this machine's cores are not alike (" + shape +
                   " by reported maximum frequency), which is measured - but the rule that turned that "
                   "into a count was refuted on a machine of exactly this shape, so the default stands "
                   "until a thread sweep says otherwise");
    } else if (hw.core_classes.size() == 1) {
        s.note("threads", u64s((uint64_t) c.n_threads), Source::Derived,
               "The default thread count: every core is alike, nothing to tune.",
               "every core here is alike, so there is no slower class to spill onto and the count is not "
               "the planner's to improve");
    } else {
        s.note("threads", u64s((uint64_t) c.n_threads), Source::Unprobed,
               "The default thread count: the kinds of core here are unknown.",
               "this machine does not report per-core maximum frequencies, so its core classes are unknown");
    }
}

} // namespace

namespace {

Plan plan_from(const RunConfig & base,
               const HardwareProfile & hw,
               const ModelProfile & model,
               const Placement & projected,
               const PlanRequest & req,
               const PlannerPolicy & pol,
               const Planning::LayersHome * home_from);

Plan plan_once(const RunConfig & base,
               const HardwareProfile & hw,
               const ModelProfile & model,
               const Placement & projected,
               const PlanRequest & req,
               const PlannerPolicy & pol,
               const Planning::LayersHome * home_from,
               uint64_t whole_room = 0) {
    Plan p;
    Planning s(base, hw, model, projected, req, pol, p);
    s.whole_room = whole_room;
    if (home_from) s.home.considered = false;

    // The machine and the model, against each other.
    share_of_host(s);
    if (declines_unplannable(s)) {
        if (p.regime == Regime::NotMoe) choose_shared_device_dense(s);
        return p;
    }
    choose_device_tier(s);
    size_budget(s);
    choose_graph_width(s);
    classify_regime(s);
    apply_placement(s);
    price_layers_home(s);
    // The account of the device's split layers: said now where nothing is left to weigh, after the
    // ranking where it is.
    if (home_from)
        s.p.decisions.push_back(home_from->decision);
    else if (!s.home.decision.knob.empty() && !s.home.priced)
        s.p.decisions.push_back(s.home.decision);
    note_regime(s);
    choose_expert_staging(s);
    choose_device_cache_cpu_share(s);
    if (declines_residency(s)) {
        // Nothing is streamed and nothing ranked: the layers stay as the fitter put them.
        if (s.home.priced) {
            s.home.decision.brief = u64s(s.home.layers) + " layers keep their dense part on the device: not weighed, "
                                                          "nothing is streamed.";
            s.home.decision.reason = home_reason(s.home, "with nothing streamed there is no ranking to weigh bringing "
                                                         "them home in, so the fitter's placement stands");
            s.p.decisions.push_back(s.home.decision);
        }
        choose_shared_device(s, s.held_all());
        // Resident, the cores still compute whatever the devices do not: the layers the fitter left on
        // the host, or all of them. Leaving the count at its default here ran a 24 GB card's 21 host
        // layers on 4 threads of a 32-core box, at 26 tok/s where 16 threads decoded at 47.5.
        leave_out_failed_devices(s);
        choose_threads(s);
        return p;
    }

    // What could leave the host beyond the first stage's placement, and why it does not.
    const DeviceSurvey devices = survey_devices(hw);
    note_extra_offload(s, devices);
    note_expert_compute(s, devices);

    // The memory ledger: the dense policy, then every row priced against the others.
    const bool pin_candidate = choose_dense_policy(s);
    p.dense_pending_bytes = dense_pending_bytes(s);
    s.host_model = host_model(s);
    calibrate_decode_cost(s);
    // Which lane the tables take is the ranking's answer, and the ranking does not spend the
    // reservation (it only labels a pinned lane), so one pricing settles what to take out of it
    // before the one the plan keeps.
    p.dense_pending_bytes -=
        unheld_table_bytes(s, allocate(hw, s.host_model, ledger_inputs(s, p.dense_pending_bytes), s.pol));
    AllocationInputs ai = ledger_inputs(s, p.dense_pending_bytes);
    const Allocation alloc = plan_prefill(s, ai);
    if (s.home.priced) {
        // Funded: the plan is made again with those layers on the host, so every line below - the
        // regime, the ledger, the cache - is the one of the placement the run will load.
        if (alloc.layers_home) {
            Planning::LayersHome home = s.home;
            home.decision.value = u64s(home.layers) + " home";
            home.decision.brief =
                u64s(home.layers) + " layers come home whole: the hops to the device cost more than it saves.";
            home.decision.reason = home_reason(home, "they come onto the host whole, for " + u64s(mib(home.bytes)) +
                                                         " MiB the ranking funded, and the devices keep the layers "
                                                         "they held whole");
            return plan_from(base, hw, model, home.whole, req, pol, &home);
        }
        s.p.decisions.push_back(s.home.decision);
    }
    p.allocation = alloc;
    settle_dense_pin(s, pin_candidate, alloc, ai);
    settle_mapped_tables(s, alloc);
    settle_cache_lock(s, alloc);
    charge_caller_cache(s);
    note_ledger(s, alloc);
    note_hit_curve(s, alloc);
    if (!size_cache(s, alloc)) return p;
    note_kv_budget(s);
    note_checkpoint_budget(s);

    // The knobs each settled by one fact.
    choose_io_lanes(s);
    choose_direct_io(s);
    choose_release_mmap(s);
    disarm_lossy(s);
    choose_gate_mirror(s);
    choose_devices(s);
    note_shared_pool(s);
    dense_on_shared_device(s);
    note_overlap(s);
    choose_threads(s);
    return p;
}

Plan plan_from(const RunConfig & base,
               const HardwareProfile & hw,
               const ModelProfile & model,
               const Placement & projected,
               const PlanRequest & req,
               const PlannerPolicy & pol,
               const Planning::LayersHome * home_from) {
    Plan p = plan_once(base, hw, model, projected, req, pol, home_from);
    // ── a model the plan streams, on a machine whose device reads this same memory ──
    // llama.cpp would hold such a model whole on the device, and where that fits it can be the faster
    // run by far: on a laptop with integrated graphics a 21 GB model streamed with an 18 GiB cache
    // decoded at 12 tok/s, and llama.cpp with every layer on the GPU at 40. Nothing in the plan
    // above says which is faster, and no probe does (choose_shared_device), so where the model held
    // whole on the device stays inside what the streaming plan itself may hold of this pool, the
    // first load is asked: it opens the model there, compares its answers with the cores' and times
    // both (KnobTrialConfig::answer_device, answer_prompt_per_token), and keeps the device unless the
    // cores decode the model faster by the margin. The cores' rate there is the model held whole on
    // them, which is the most a stream on the same cores can reach, so a device the cores do not
    // beat is one the stream does not beat. Where the device does not keep the model, the session
    // reopens with the plan above, unchanged (RunConfig::fallback): the stream is what this engine
    // adds, and it serves wherever holding the model whole is not the faster run.
    if (p.regime != Regime::ExpertsStream || !p.config.moe.enabled || p.streaming_declined || !model.is_moe) return p;
    // What the streaming plan may hold of this pool: its ledger's ceiling, the margin already off it.
    const uint64_t room = p.allocation.ledger.holdable_cap;
    if (!devices_share_host_memory(hw) || projected.fitted || room == 0) return p;
    if (req.is_pinned("moe-stream") || req.is_pinned("cache-mb")) return p;
    Plan whole = plan_once(base, hw, model, projected, req, pol, home_from, room);
    if (whole.config.n_gpu_layers == 0 || whole.config.device_split.size() != 1 || whole.config.moe.enabled) return p;
    const std::string dev = whole.config.device_split.front().first;
    whole.config.fallback = std::make_shared<const RunConfig>(p.config);
    std::string stream_knobs;
    for (const Decision & x : p.decisions)
        if (x.knob == "moe-stream" || x.knob == "cache-mb" || x.knob == "prefill-device" || x.knob == "dense-weights")
            stream_knobs += (stream_knobs.empty() ? "" : "; ") + x.knob + " " + x.value;
    Decision d;
    d.knob = "held-whole";
    d.value = dev;
    d.source = Source::Derived;
    d.brief = "Held whole on " + dev + " if it runs the model faster; otherwise streamed, with a " +
              size_brief((uint64_t) p.config.moe.cache_mb << 20) + " cache.";
    d.reason = "this model does not fit beside the room this machine asks for, and the plan that streams it may "
               "hold " +
               u64s(mib(room)) + " MiB of this memory, " + u64s(mib(p.allocation.ledger.held)) + " MiB of it with a " +
               u64s((uint64_t) p.config.moe.cache_mb) + " MiB expert cache. " + dev +
               " reads this same memory, and the model held whole there with its compute buffer stays inside that, "
               "so the load decides between the two on the model itself (shared-device says how); where " +
               dev + " does not keep it, the session reopens as the streaming plan (" + stream_knobs + ")";
    whole.decisions.push_back(d);
    return whole;
}

} // namespace

Plan plan_run(const RunConfig & base,
              const HardwareProfile & hw,
              const ModelProfile & model,
              const Placement & projected,
              const PlanRequest & req,
              const PlannerPolicy & pol) {
    // A model with no routing record of its own is priced on the prior curve, so the cache is ranked
    // and credited by one path whichever it has (bmoe/routing.h).
    if (!model.hit_curve.credited() && model.is_moe && model.expert_bytes > 0 && model.token_cycle_bytes > 0) {
        ModelProfile with_prior = model;
        with_prior.hit_curve = prior_hit_curve(model.expert_bytes, model.token_cycle_bytes);
        with_prior.hit_curve.rejected = model.hit_curve.rejected;
        return plan_run(base, hw, with_prior, projected, req, pol);
    }
    // Which backend drives each card, before anything is weighed on the card: the rules below then
    // see one device per card, the one that will run (bmoe/backend_choice.h).
    if (req.is_pinned("device-list") || req.is_pinned("devices"))
        return plan_from(base, hw, model, projected, req, pol, nullptr);
    const BackendChoice choice = choose_backends(hw, (double) pol.min_backend_win);
    if (choice.left_out.empty()) return plan_from(base, hw, model, projected, req, pol, nullptr);
    Plan p = plan_from(base, without_left_out(hw, choice), model, projected, req, pol, nullptr);
    p.config.device_list = choice.keep;
    Decision d;
    d.knob = "device-list";
    for (const std::string & n : choice.keep)
        d.value += (d.value.empty() ? "" : ",") + n;
    d.source = Source::Measured;
    d.brief = "One backend per card, the fastest measured: " + d.value + ".";
    for (const std::string & w : choice.why)
        d.reason += (d.reason.empty() ? "" : "; ") + w;
    d.reason += "; on the same card the backend registered first is what llama.cpp keeps, and another replaces "
                "it only where it measured faster by the margin";
    p.decisions.insert(p.decisions.begin(), std::move(d));
    return p;
}

} // namespace bmoe
