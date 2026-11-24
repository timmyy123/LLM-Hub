// The cost model and the allocation. Read bmoe/allocate.h first: it states the unit everything is
// priced in and the three rules this file exists to obey.
//
// The shape of the algorithm is deliberately small enough to argue with:
//
//   1. Price each group in each LEGAL lane. A lane the engine cannot execute for that group, or one
//      whose deciding measurement is missing, is not cheap — it is absent.
//   2. Rank the candidates by SECONDS SAVED PER BYTE OF RAM and spend the budget down that list.
//      The expert cache is one more candidate on the same list — a fractional one, above its own
//      floor — rather than a special case that receives whatever is left over by accident.
//   3. Offer each RESIDENT group to a device, subject to the margin AND to the price of the graph
//      crossings the move creates. On a streamed group the read dominates the compute by more than
//      an order of magnitude, so a faster engine would hide behind the same I/O and buy nothing.
//
// Each step is one function below, and allocate() at the bottom is the order they run in.

#include "bmoe/allocate.h"

#include "bmoe/planner.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace bmoe {

std::string size_brief(uint64_t bytes) {
    char b[32];
    if (bytes >= (1ull << 30))
        std::snprintf(b, sizeof(b), "%.1f GiB", (double) bytes / (double) (1ull << 30));
    else
        std::snprintf(b, sizeof(b), "%llu MiB", (unsigned long long) (bytes >> 20));
    return b;
}

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kMiB = 1024.0 * 1024.0;

// Fastest-to-slowest ratio across the storage probe's repeats above which a quoted rate is flagged
// as unrepeatable: under a quarter the predicted totals still mean what they say.
constexpr double kRateSpreadWarn = 1.25;

// Relative difference under which two densities are the same density. Wide enough to cover the
// few ulp that computing an equal ratio two ways leaves behind, far below any real difference.
constexpr double kDensityTieTolerance = 1e-9;

std::string mibs(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%llu MiB", (unsigned long long) (bytes >> 20));
    return buf;
}

// Seconds to move `bytes` at `mib_per_s`. An unmeasured rate is infinite rather than zero: a
// candidate priced from a missing fact must never look cheap.
double seconds_at_mibs(uint64_t bytes, double mib_per_s) {
    return mib_per_s > 0.0 ? (double) bytes / kMiB / mib_per_s : kInf;
}

double seconds_at_gibs(uint64_t bytes, double gib_per_s) {
    return gib_per_s > 0.0 ? (double) bytes / (kMiB * 1024.0) / gib_per_s : kInf;
}

double finite_or_zero(double v) {
    return std::isfinite(v) ? v : 0.0;
}

void note(Allocation & a, const char * topic, std::string value, std::string brief, std::string detail) {
    a.notes.push_back({topic, std::move(value), std::move(brief), std::move(detail)});
}

// A read rate as a person reads it, in the units size_brief() uses.
std::string rate_brief(double mib_per_s) {
    char b[32];
    if (mib_per_s >= 1024.0)
        std::snprintf(b, sizeof(b), "%.1f GiB/s", mib_per_s / 1024.0);
    else
        std::snprintf(b, sizeof(b), "%.0f MiB/s", mib_per_s);
    return b;
}

} // namespace

void price_unstreamed(Allocation & a) {
    a.seconds_per_token = 0.0;
    a.resident_bytes = 0;
    for (GroupPlacement & g : a.groups) {
        if (!(g.seconds_worst_case > 0.0) && g.resident_bytes == 0 && !(g.seconds_per_token > 0.0)) continue;
        g.lane = Lane::Mmap;
        g.device = -1;
        g.resident_bytes = 0;
        g.seconds_per_token = g.seconds_worst_case;
        g.reason = "left mapped: streaming was declined, so the kernel decides what stays and this is the cost "
                   "with nothing resident";
        a.seconds_per_token += finite_or_zero(g.seconds_per_token);
    }
    a.dense_pinned = false;
    a.cache_bytes = 0;
    a.cache_hit_credited = 0.0;
    a.cache_locked_bytes = 0;
    a.cache_lock_unverified = false;

    // The ledger keeps what the run holds whatever it does with the weights, and drops the rest.
    Ledger & l = a.ledger;
    l.rows.erase(std::remove_if(l.rows.begin(), l.rows.end(),
                                [](const LedgerRow & r) {
                                    return r.kind == LedgerKind::Dense || r.kind == LedgerKind::ExpertCache ||
                                           r.in_turn;
                                }),
                 l.rows.end());
    l.held = l.locked = 0;
    for (const LedgerRow & r : l.rows) {
        l.held += r.bytes;
        if (r.locked) l.locked += r.bytes;
    }
}

const char * ledger_kind_name(LedgerKind k) {
    switch (k) {
    case LedgerKind::ContextCompute:
        return "context and compute";
    case LedgerKind::Projector:
        return "projector";
    case LedgerKind::Checkpoints:
        return "context checkpoints";
    case LedgerKind::Device:
        return "device";
    case LedgerKind::Dense:
        return "dense";
    case LedgerKind::ExpertCache:
        return "expert cache";
    case LedgerKind::DeviceSlots:
        return "device slots";
    }
    return "?";
}

const char * lane_name(Lane l) {
    switch (l) {
    case Lane::Mmap:
        return "mmap";
    case Lane::Resident:
        return "resident";
    case Lane::Pinned:
        return "pinned";
    case Lane::RowStream:
        return "row-stream";
    case Lane::ExpertStream:
        return "expert-stream";
    default:
        return "?";
    }
}

namespace {

// One allocate() call, carried through its steps: the inputs, the allocation being written, the
// rates every price is built from, and the running account of what is already held and locked.
struct Pricing {
    const HardwareProfile & hw;
    const ModelProfile & model;
    const AllocationInputs & in;
    const PlannerPolicy & pol;
    Allocation & a;

    double host_gibs = 0.0;       // how fast the host consumes this model's expert bytes
    double host_dense_gibs = 0.0; // and its dense ones: the real layer's two halves differ
    double refault_mibs = 0.0;    // what a mapped group costs once pressure has dropped its pages
    uint32_t slice = 0;           // this model's expert slice, the request size the expert lane issues
    uint32_t lanes = 0;           // the lane count that reached the best rate at that size
    double stream_mibs = 0.0;     // the rate the expert lane will really see

    uint64_t fixed = 0;     // the ledger's fixed lines: context, compute buffers, an armed device
    uint64_t spendable = 0; // what the ranking may spend once those are off the top
    uint64_t locked = 0;    // what is already charged to the lockable total
    uint64_t unspent = 0;   // what the ranking left once every candidate had its turn
};

// A group in a lane, as the ranking sees it.
struct Candidate {
    WeightGroup group;
    Lane lane;
    uint64_t bytes;  // RAM it wants
    double saved;    // seconds/token saved against this group's baseline lane
    double density;  // saved per byte of RAM: the ranking key
    bool fractional; // may take part of `bytes` and save proportionally (the expert cache)
    uint64_t floor;  // fractional only: below this it saves nothing at all
    std::string why;
    // A piece of the cache priced from the model's measured hit curve rather than the linear credit:
    // its position along the curve. The pieces are taken in that order and only while each one
    // before was taken whole, since a hit further along the curve is only bought by holding the
    // bytes before it. -1 for every other candidate.
    int curve_piece = -1;
    // The device's dense-only layers brought home (AllocationInputs::layers_home_*): RAM bought to
    // remove crossings rather than reads. It places no group.
    bool layers_home = false;
};

// The cache as pieces of this model's measured hit curve, for the ranking. Each piece is a stretch
// of the curve's upper concave hull, so their densities fall along it and the ranking, taking the
// densest first, walks the curve from the floor up as far as each next byte is worth more than what
// competes with it. The hull, not the curve itself: a stretch where the curve rises late (an LRU
// below one token cycle returns nothing, then a great deal at once) is worth its average to a buyer
// who takes all of it, which is the only way it can be bought. What the cache is finally credited is
// the curve at the size it gets (bound_cache), never the hull.
//
// The curve covers the whole expert set; the cache serves the host's share of it, and a host cache
// of C is read as a whole-set cache of C scaled up by the share it covers.
void push_curve_pieces(std::vector<Candidate> & out,
                       const HitCurve & curve,
                       double scale,
                       uint64_t cacheable,
                       uint64_t floor,
                       double cycle_seconds) {
    struct Vertex {
        double x, y;
    };
    std::vector<Vertex> hull{{0.0, 0.0}};
    for (const HitPoint & pt : curve.points) {
        const double x = (double) pt.cache_bytes * scale;
        if (x < (double) floor || x > (double) cacheable + 0.5) continue;
        const Vertex v{x, pt.hit};
        // Keep the hull concave: drop a vertex that lies on or under the chord past it.
        while (hull.size() >= 2) {
            const Vertex & a = hull[hull.size() - 2];
            const Vertex & b = hull.back();
            if ((b.x - a.x) * (v.y - a.y) - (b.y - a.y) * (v.x - a.x) < 0.0) break;
            hull.pop_back();
        }
        if (v.x > hull.back().x) hull.push_back(v);
    }
    int piece = 0;
    for (size_t i = 1; i < hull.size(); ++i) {
        const uint64_t bytes = (uint64_t) (hull[i].x - hull[i - 1].x);
        const double saved = cycle_seconds * (hull[i].y - hull[i - 1].y);
        if (bytes == 0 || !(saved > 0.0)) break;
        Candidate k{WeightGroup::Experts,   Lane::ExpertStream, bytes, saved, saved / (double) bytes, true,
                    piece == 0 ? floor : 0, "expert cache"};
        k.curve_piece = piece++;
        out.push_back(k);
    }
}

// ── the rates every price is built from ──────────────────────────────────────────────────────
void measure_rates(Pricing & c) {
    const HardwareProfile & hw = c.hw;
    Allocation & a = c.a;
    // Compute: how fast the host consumes this model's own weight bytes. Not a raw bandwidth figure,
    // and the difference is the point — a quantized matmul carries dequantisation per byte.
    // Measured on the model's own layer where it was, each half at its own rate.
    c.host_gibs = hw.host_expert_rate();
    c.host_dense_gibs = hw.host_dense_rate();
    // What a load of this configuration measured the decode's compute at, against what these rates
    // priced it at: the rates the compute runs at here, per-op and synchronisation costs included,
    // which a matmul probe does not see. Both halves together, so their ratio stays as measured.
    if (c.in.compute_scale > 0.0 && c.in.compute_scale != 1.0) {
        c.host_gibs /= c.in.compute_scale;
        c.host_dense_gibs /= c.in.compute_scale;
    }

    // The refault price. What a mapped group costs once pressure has dropped its pages, paid one
    // page at a time. Measured separately because it is not a fraction of the sequential rate.
    c.refault_mibs = hw.storage.refault_mibs;

    // The rate the expert lane will really see: at the request size this model's slices ARE, at the
    // lane count that reached it. Everything about streaming is priced with this rather than with
    // the surface's peak — pricing the lane at the peak of the surface instead of at the real
    // request size overstated it by 2.2x on one host.
    c.slice = (uint32_t) c.model.expert_slice_bytes;
    c.lanes = c.slice ? hw.storage.best_lanes(c.slice) : 0;
    c.stream_mibs = (c.slice && c.lanes) ? hw.storage.rate_at(c.slice, c.lanes) : 0.0;

    if (c.host_gibs <= 0.0)
        note(a, "unmeasured-compute", "-", "Compute speed was not measured, so no weight is kept in RAM for speed.",
             "no compute rate was measured, so residency cannot be priced against a read: every group keeps the "
             "lane it would have had, and nothing is bought");
    if (c.refault_mibs <= 0.0)
        note(a, "unmeasured-refault", "-", "The cost of re-reading weights from storage was not measured.",
             "the price of a refault was not measured on this storage, so what a mapped group costs under pressure "
             "is unknown and no residency can be justified by avoiding it");
    if (c.stream_mibs <= 0.0 && c.model.is_moe)
        note(a, "unmeasured-storage", "-", "Storage speed for expert reads was not measured, so streaming is unpriced.",
             "no read rate for this model's expert slice on this storage: the expert lane cannot be priced, so no "
             "plan that depends on it will be proposed");
    else if (c.slice && c.stream_mibs > 0.0) {
        char b[320];
        std::snprintf(b, sizeof(b),
                      "this model's expert slice is %u KiB and at that request size this storage gives %.0f MiB/s "
                      "over %u lanes: the expert lane is priced there, not at the storage's peak",
                      c.slice >> 10, c.stream_mibs, c.lanes);
        char brief[160];
        std::snprintf(brief, sizeof(brief), "Experts load at %s, in %u KiB reads, %u at a time.",
                      rate_brief(c.stream_mibs).c_str(), c.slice >> 10, c.lanes);
        note(a, "storage-rate", rate_brief(c.stream_mibs), brief, b);
        // A rate that is not repeatable makes every number downstream of it unrepeatable too, and a
        // reader comparing two runs deserves to know that before concluding anything from them.
        if (hw.storage.rate_spread > kRateSpreadWarn) {
            std::snprintf(b, sizeof(b),
                          "that rate varied by %.1fx across the probe's own repeats, so every predicted cost below "
                          "carries at least that much uncertainty: treat the ranking as firmer than the totals",
                          hw.storage.rate_spread);
            char value[32];
            std::snprintf(value, sizeof(value), "%.1fx", hw.storage.rate_spread);
            std::snprintf(brief, sizeof(brief), "Storage speed varied %s between repeats: predictions are rough.",
                          value);
            note(a, "storage-spread", value, brief, b);
        }
    }
}

// ── the ledger's fixed lines, charged before anything is ranked ──────────────────────────────
// The context, the compute buffers and an armed device are held whatever is decided below, so they
// come off the top of both ceilings rather than out of whatever the weights leave.
//
// The dense set is read whole on every token, so it is protected before the cache is - and only
// whole. Pinning what fits and leaving the rest is the shape that was measured to collapse: the
// grant stops part way and the remainder is reclaimed on every token.
void charge_fixed(Pricing & c) {
    const AllocationInputs & in = c.in;
    Allocation & a = c.a;
    c.fixed = in.fixed_bytes + in.projector_bytes + in.checkpoint_bytes + in.device_bytes;
    c.spendable = in.budget_bytes > c.fixed ? in.budget_bytes - c.fixed : 0;
    c.locked = in.device_locked ? in.device_bytes : 0;

    const uint64_t lock_room = in.lockable_bytes > c.locked ? in.lockable_bytes - c.locked : 0;
    a.dense_pinned = in.can_pin && in.reserved_bytes > 0 &&
                     (in.pin_forced || (in.lockable_bytes_known && in.reserved_bytes <= lock_room));
    if (in.can_pin && in.reserved_bytes > lock_room) {
        a.pinned_shortfall_bytes = in.reserved_bytes - lock_room;
        note(a, "dense-pin", size_brief(a.pinned_shortfall_bytes),
             in.pin_forced ? "Pinned dense weights are " + size_brief(a.pinned_shortfall_bytes) +
                                 " more than can be pinned; part will be refused."
                           : "Dense weights stay ordinary RAM: " + size_brief(a.pinned_shortfall_bytes) +
                                 " too big to pin whole.",
             "the dense set is " + mibs(in.reserved_bytes) + " and " + mibs(lock_room) +
                 " can still be held reclaim-exempt, " + mibs(a.pinned_shortfall_bytes) + " short" +
                 (in.pin_forced ? ": the caller chose the pinned store, so it is charged as asked and part of it "
                                  "will be refused"
                                : ": it is held as ordinary memory instead, whole, because a set pinned part way "
                                  "is reclaimed from the part that was not"));
    }
    if (a.dense_pinned) c.locked += in.reserved_bytes;
}

// ── 1. price every legal lane ────────────────────────────────────────────────────────────────
//
// A group left mapped is priced as if pressure WILL drop its pages. That is the pessimistic end of
// a range this cannot narrow without knowing the final allocation, and the bias it introduces
// pushes toward residency — which the budget bounds anyway, so the error cannot spend memory that
// does not exist. Each group gets its baseline lane here; the ones residency would improve are
// returned as candidates for the ranking.
std::vector<Candidate> price_lanes(Pricing & c) {
    const ModelProfile & model = c.model;
    Allocation & a = c.a;
    std::vector<Candidate> candidates;
    const Lane resident_lane = a.dense_pinned ? Lane::Pinned : Lane::Resident;

    for (int i = 0; i < (int) WeightGroup::count; ++i) {
        const WeightGroup g = (WeightGroup) i;
        const GroupDemand & d = model.group(g);
        if (d.bytes == 0) continue;
        GroupPlacement & p = a.at(g);

        const double compute =
            seconds_at_gibs(d.bytes_per_token, g == WeightGroup::Experts ? c.host_gibs : c.host_dense_gibs);
        const double refault = seconds_at_mibs(d.bytes_per_token, c.refault_mibs);

        // The baseline: mapped, and refaulted. Both terms, because the bytes must be read AND
        // multiplied; a group that is resident still pays the compute.
        p.seconds_worst_case = finite_or_zero(compute) + refault;
        p.seconds_per_token = p.seconds_worst_case;
        p.reason = "left mapped: nothing cheaper fit in the budget";

        if (g == WeightGroup::Experts) {
            if (!model.is_moe || c.stream_mibs <= 0.0) continue;
            const uint64_t cycle = model.token_cycle_bytes;
            if (cycle == 0) continue;

            // Streaming with an EMPTY cache already beats faulting the whole set through the page
            // cache: a slice is read once, sequentially, at the size the surface was measured at,
            // instead of a page at a time. So this is the group's new baseline, not a candidate.
            p.lane = Lane::ExpertStream;
            p.seconds_per_token = seconds_at_mibs(cycle, c.stream_mibs) + finite_or_zero(compute);
            p.reason = "streamed, no cache: the routed top-k read as sequential slices";

            // The floor is the model's token cycle and nothing else. Below it an LRU evicts a slice
            // before the same token needs it again, so the cache costs its memory and returns no
            // hits — that is mechanical, and it is the only bound this file may enforce.
            //
            // The engine's fixed guard is NOT a second floor. It is a generic constant chosen once
            // for every model, the caller clears it with `force-cache`, and enforcing it here turns
            // a working configuration into a refusal: measured on a phone where this model's cycle
            // is 581 MiB and 1077 MiB were free, which returns hits and which the guard alone would
            // have declined.
            const uint64_t floor = cycle;
            if (c.in.engine_cache_min > cycle)
                note(a, "cache-floor", size_brief(cycle),
                     "The smallest cache that helps is " + size_brief(cycle) + ", under the engine's usual " +
                         size_brief(c.in.engine_cache_min) + " minimum.",
                     "the engine's fixed cache guard (" + mibs(c.in.engine_cache_min) +
                         ") is above this model's token cycle (" + mibs(cycle) +
                         "), so a budget between the two returns hits and needs force-cache to be accepted; the "
                         "floor enforced here is the model's");

            // With this model's own hit curve, the cache is priced along it.
            if (model.hit_curve.credited() && c.in.cacheable_bytes > 0 && model.expert_bytes > 0) {
                push_curve_pieces(candidates, model.hit_curve,
                                  (double) c.in.cacheable_bytes / (double) model.expert_bytes, d.bytes, floor,
                                  seconds_at_mibs(cycle, c.stream_mibs) * c.in.hit_scale);
                continue;
            }

            // A cache holding the whole expert set saves every read. The credited hit rate is
            // linear in its size, so the saving per byte of RAM is constant above the floor and this
            // candidate can be ranked directly against the read-whole groups.
            const double saved_full = seconds_at_mibs(cycle, c.stream_mibs) *
                                      std::min(1.0, (double) c.pol.cache_hit_optimism * c.in.hit_scale);
            if (saved_full > 0.0)
                candidates.push_back({g, Lane::ExpertStream, d.bytes, saved_full, saved_full / (double) d.bytes, true,
                                      floor, "expert cache"});
            continue;
        }

        if (d.row_gatherable)
            p.reason = "left mapped: the graph gathers rows from it, so a token reads a few KiB of it however "
                       "large the table is, and residency would buy bytes nothing reads";

        if (d.row_gatherable && c.refault_mibs > 0.0) {
            // A table the graph only gathers rows from is resident for bytes that are never read.
            // Streaming its rows costs one small read per token and returns the rest of its
            // footprint to the groups that ARE read whole. It wants a window, not residency, so it
            // never competes for the budget.
            const double row_cost = seconds_at_mibs(d.bytes_per_token, c.refault_mibs) + finite_or_zero(compute);
            if (row_cost < p.seconds_per_token) {
                p.lane = Lane::RowStream;
                p.seconds_per_token = row_cost;
                p.reason = "row-gathered: only the rows the graph asks for are read, so residency would buy bytes "
                           "nothing reads";
                continue;
            }
        }

        // Residency removes the refault and leaves the compute. Where either half is unmeasured the
        // saving is not a number, and a candidate without a number does not enter the ranking.
        if (!std::isfinite(p.seconds_worst_case) || !std::isfinite(compute)) continue;
        const double saved = p.seconds_worst_case - compute;
        if (saved <= 0.0) continue;

        // A table the graph only gathers rows from saves one small refault a token, so its saving per
        // byte is tiny, and it can only win the ranking where the cache's own price has gone flat. A
        // routing record of a few hundred tokens is flat past the experts it touched, which is a
        // record too short to see the rest, not proof that RAM there is worth nothing (the remainder
        // rule below gives that RAM to the cache for the same reason). So such a table competes with
        // the cache at the cache's linear credit, the price it has without a record: measured where it
        // did not, a 194 GB Engram table outranked the cache for 0.1 ms a token.
        if (d.row_gatherable && model.is_moe && c.in.cacheable_bytes > 0 && c.stream_mibs > 0.0) {
            const double linear = seconds_at_mibs(model.token_cycle_bytes, c.stream_mibs) *
                                  std::min(1.0f, c.pol.cache_hit_optimism) / (double) c.in.cacheable_bytes;
            if (std::isfinite(linear) && saved / (double) d.bytes < linear) continue;
        }
        candidates.push_back(
            {g, resident_lane, d.bytes, saved, saved / (double) d.bytes, false, 0, "read whole, every token"});
    }

    // The crossings a device's dense-only layers cost, against the RAM that brings them home.
    if (c.in.layers_home_bytes > 0 && c.in.layers_home_saved > 0.0) {
        Candidate k{WeightGroup::Other,
                    Lane::Resident,
                    c.in.layers_home_bytes,
                    c.in.layers_home_saved,
                    c.in.layers_home_saved / (double) c.in.layers_home_bytes,
                    false,
                    0,
                    "layers home"};
        k.layers_home = true;
        candidates.push_back(k);
    }
    return candidates;
}

// ── 2. spend the budget down the ranking ─────────────────────────────────────────────────────
// Density first, and at equal density the LARGER candidate first. The tie is not a corner case:
// every group that is read whole every token has the same seconds-saved-per-byte, because both
// halves of the ratio scale with its size. With equal density the total saving is proportional to
// the bytes packed, so packing the big one first is strictly better - and packing it last strands
// it. Measured on a phone with a dense-oversized model: three small groups took 1511 MiB and left a
// 2700 MiB attention set to refault at 112 s/token, where taking the big one first fits it and most
// of the rest.
void spend_ranking(Pricing & c, std::vector<Candidate> & candidates) {
    Allocation & a = c.a;
    std::sort(candidates.begin(), candidates.end(), [](const Candidate & x, const Candidate & y) {
        // Pieces of the curve keep their order along it, which the hull already makes their density
        // order: this only stops the tie rule below from swapping two that are nearly collinear.
        if (x.curve_piece >= 0 && y.curve_piece >= 0) return x.curve_piece < y.curve_piece;
        // Ties are compared with a relative tolerance, not exactly, and that is the whole point of
        // this comparator. In exact arithmetic every group that is read whole every token has the
        // SAME density - it is `1/refault_rate - 1/compute_rate`, and the group's size cancels -
        // but computed as (b/r - b/h)/b it comes out differing by a few ulp per group. An exact
        // comparison would then read those ulp as a real ranking and order the groups by rounding
        // noise. With the tie recognised, the larger candidate goes first: at equal density the
        // total saving is proportional to the bytes packed, so packing the big one first is never
        // worse and packing it last can strand it.
        const double scale = x.density > y.density ? x.density : y.density;
        const bool tied = scale > 0.0 && std::fabs(x.density - y.density) <= scale * kDensityTieTolerance;
        if (!tied) return x.density > y.density;
        return x.bytes > y.bytes;
    });

    uint64_t remaining = c.spendable;
    bool curve_open = true; // the next piece of the curve may still be taken
    for (const Candidate & k : candidates) {
        if (k.layers_home) {
            const std::string n = std::to_string(c.in.layers_home_count);
            char saved[32];
            std::snprintf(saved, sizeof(saved), "%.2f ms", 1e3 * k.saved);
            if (k.bytes <= remaining) {
                remaining -= k.bytes;
                a.layers_home = true;
                note(a, "device-crossing", saved,
                     "The " + n + " layers whose experts stay on the CPU come home whole: the hop costs more.",
                     "the " + n +
                         " layers whose dense part is on a device and whose experts are on the host cross "
                         "the boundary twice a token each; measured, those crossings cost " +
                         saved + " a token more than the device saves on their dense matmuls, and the " +
                         mibs(k.bytes) + " that brings them onto the host is worth more there than as cache");
            } else {
                note(a, "device-crossing", saved,
                     "The " + n + " layers on the device stay there: bringing them home needs " + size_brief(k.bytes) +
                         ".",
                     "bringing the " + n + " dense-only device layers onto the host would save " + saved +
                         " a token in crossings, and needs " + mibs(k.bytes) + " where " + mibs(remaining) +
                         " was left at its place in the ranking");
            }
            continue;
        }
        GroupPlacement & p = a.at(k.group);
        if (k.curve_piece > 0 && !curve_open) continue;
        if (k.fractional) {
            if (remaining < k.floor) {
                curve_open = false;
                note(a, "cache-off", size_brief(remaining),
                     "No cache: " + size_brief(remaining) + " is left, under the " + size_brief(k.floor) +
                         " one token reads.",
                     "expert cache left off: " + mibs(remaining) + " remained but one token's routing reads " +
                         mibs(k.floor) +
                         ". A budget below that evicts a slice before the same token needs it again - an eviction "
                         "per read and no hits, measurably slower than no cache at all. The floor is the model's, "
                         "not a constant.");
                continue;
            }
            if (k.curve_piece >= 0) {
                // Along the curve: the piece whole while it fits, the rest of the budget where it
                // does not, and then no further piece. The total is what is rounded, as below.
                const uint64_t want = std::min(remaining, k.bytes);
                curve_open = want == k.bytes;
                const uint64_t total = (a.cache_bytes + want) & ~((1ull << 20) - 1);
                if (total < k.floor) {
                    curve_open = false;
                    continue;
                }
                remaining -= total - a.cache_bytes;
                a.cache_bytes = total;
                p.lane = Lane::ExpertStream;
                p.resident_bytes = total;
                continue;
            }
            // Rounded down to a whole MiB: the flag this becomes is expressed in MiB, and a report
            // that says 1749 while the command line says 1748 is a report nobody can check against
            // the run it produced.
            const uint64_t take = std::min(remaining, k.bytes) & ~((1ull << 20) - 1);
            if (take < k.floor) continue;
            a.cache_bytes = take;
            remaining -= take;

            // The lane is settled here; the SIZE is not final until the reservation below has had
            // its say, so the rationale is written there. A reason quoting a number the placement
            // does not carry is a plan that contradicts itself, which is worse than a terse one.
            p.lane = Lane::ExpertStream;
            p.resident_bytes = take;
            continue;
        }
        if (k.bytes > remaining) continue; // does not fit; a later, smaller candidate still might
        remaining -= k.bytes;
        p.lane = k.lane;
        p.resident_bytes = k.bytes;
        p.seconds_per_token = p.seconds_worst_case - k.saved;
        p.reason = std::string(k.lane == Lane::Pinned ? "pinned: " : "resident: ") + k.why;
    }
    c.unspent = remaining;
}

// What the dense policy holds whatever the ranking concluded. The run's dense mode copies every dense
// tensor it is handed (AllocationInputs::reserved_bytes), so a group read whole every token that the
// ranking found no room for is not left mapped in the run: it is held, beyond the budget, and the
// ledger already charges it there. Priced as mapped it stood in the plan at its refault price while
// the run computed it from RAM: measured on a 12 GB phone, Gemma 4 26B-A4B with a 16384 context
// predicted 10-44 s a token (attention and the dense FFN at a 40 MiB/s refault) and decoded at
// 0.45-0.55. So such a group takes the lane the run gives it, up to the reservation's bytes. A table
// the graph only gathers rows from is the exception the run makes too (settle_mapped_tables): it stays
// where the ranking put it.
void hold_reserved(Pricing & c) {
    const AllocationInputs & in = c.in;
    Allocation & a = c.a;
    if (in.reserved_bytes == 0) return;
    uint64_t held = 0;
    std::vector<int> left;
    for (int i = 0; i < (int) WeightGroup::count; ++i) {
        const WeightGroup g = (WeightGroup) i;
        if (g == WeightGroup::Experts || c.model.group(g).bytes == 0) continue;
        const GroupPlacement & p = a.at(g);
        if (p.lane == Lane::Resident || p.lane == Lane::Pinned)
            held += p.resident_bytes;
        else if (p.lane == Lane::Mmap && p.device < 0 && !c.model.group(g).row_gatherable)
            left.push_back(i);
    }
    std::sort(left.begin(), left.end(), [&](int x, int y) {
        return c.model.group((WeightGroup) x).bytes > c.model.group((WeightGroup) y).bytes;
    });
    uint64_t room = in.reserved_bytes > held ? in.reserved_bytes - held : 0;
    std::string names;
    uint64_t over = 0;
    for (int i : left) {
        const WeightGroup g = (WeightGroup) i;
        const GroupDemand & d = c.model.group(g);
        if (d.bytes > room) continue;
        const double compute = seconds_at_gibs(d.bytes_per_token, c.host_dense_gibs);
        if (!std::isfinite(compute)) continue;
        GroupPlacement & p = a.at(g);
        room -= d.bytes;
        over += d.bytes;
        p.lane = a.dense_pinned ? Lane::Pinned : Lane::Resident;
        p.resident_bytes = d.bytes;
        p.seconds_per_token = compute;
        p.reason = std::string(a.dense_pinned ? "pinned" : "resident") +
                   ": the dense policy copies it whole, though the budget had no room for it";
        names += (names.empty() ? "" : ", ") + std::string(group_name(g));
    }
    if (over == 0) return;
    note(a, "dense-held", size_brief(over),
         "The dense weights are held whole: " + size_brief(over) + " of them past the budget.",
         "the dense policy copies every dense tensor, so " + names + " (" + mibs(over) +
             ") are held in RAM although the ranking found no room for them within the budget: they are priced "
             "as held, at their compute, which is what the run does, and the memory they take is charged to the "
             "ledger past its ceiling, where the machine has to find it");
}

// The cache, sized against what the loader will actually leave. Two paths, and which one ran is the
// difference between a number that was chosen and one that was merely available.
//
// Where the expert lane could be PRICED, the ranking above already decided what the cache is worth
// and `a.cache_bytes` holds it; all that is left is to bound it by what the dense policy is about to
// take, since that mode is one global setting the ranking cannot override.
//
// Where it could NOT be priced, sizing the cache is still not a guess to be refused: the regime
// decided that the experts stream, so the cache must have some size, and the honest answer is what
// remains. That is the old residual - kept, but now clearly labelled as the unpriced fallback rather
// than presented as a decision.
void bound_cache(Pricing & c) {
    const AllocationInputs & in = c.in;
    const ModelProfile & model = c.model;
    Allocation & a = c.a;
    if (in.cacheable_bytes == 0 || !model.is_moe) return;
    const uint64_t after_reserved = c.spendable > in.reserved_bytes ? c.spendable - in.reserved_bytes : 0;
    const uint64_t priced = a.cache_bytes;
    uint64_t sized = std::min(after_reserved, in.cacheable_bytes) & ~((1ull << 20) - 1);

    // The second ceiling. Where the cache can be kept in place, it is sized to what can be kept: a
    // byte past that is a byte the machine is free to take back, and on the machine this was
    // measured on taking it back cost nine tenths of the throughput. Where not even one token cycle
    // can be protected the question does not arise - the cache is ordinary memory, as it is on every
    // platform that grants no lock, and it is bounded by what can be held like everything else.
    const uint64_t cache_lock_room =
        (in.lock_in_place_bytes > c.locked ? in.lock_in_place_bytes - c.locked : 0) & ~((1ull << 20) - 1);
    const bool cache_may_lock = in.cache_lock && cache_lock_room >= model.token_cycle_bytes;
    const bool cache_protected = cache_may_lock && in.lock_in_place_bytes_known;
    // An offered cache is taken back by dropping it, never by a write (AllocationInputs::cache_offered),
    // so it leaves the margin as a locked one does - all but the token cycle the last pass used.
    const bool cache_offered = !cache_protected && in.cache_offered && in.cacheable_bytes > model.token_cycle_bytes;
    uint64_t holdable_cap = in.budget_bytes;
    if ((cache_protected || cache_offered) && in.holdable_bytes > 0) {
        // With the cache locked or offered, most of what this run holds is not memory a reclaim has
        // to write to take back, and a margin sized on all of it would be room left for a reclaim
        // that does not happen. It is sized on what stays ordinary instead, and never lowers the
        // ceiling the caller was given. "Ordinary" is never less than the rows that really are: an
        // unpinned dense set is memory the machine can take back however much lock is left unused
        // beside it, and an offered cache keeps the experts of its last pass as ordinary memory.
        const uint64_t exempt_cache = cache_protected ? std::min(cache_lock_room, in.cacheable_bytes)
                                                      : in.cacheable_bytes - model.token_cycle_bytes;
        const uint64_t locked_all = c.locked + exempt_cache;
        const uint64_t ordinary_rows =
            in.fixed_bytes + in.projector_bytes + in.checkpoint_bytes + (in.device_locked ? 0 : in.device_bytes) +
            (a.dense_pinned ? 0 : in.reserved_bytes) + (cache_offered ? model.token_cycle_bytes : 0);
        const uint64_t ordinary =
            std::max(in.holdable_bytes > locked_all ? in.holdable_bytes - locked_all : 0, ordinary_rows);
        const uint64_t m = std::max((uint64_t) ((double) ordinary * in.margin_frac), in.margin_min_bytes);
        holdable_cap = std::max(holdable_cap, in.holdable_bytes > m ? in.holdable_bytes - m : 0);
        const uint64_t taken = c.fixed + in.reserved_bytes;
        sized = std::min(holdable_cap > taken ? holdable_cap - taken : 0, in.cacheable_bytes) & ~((1ull << 20) - 1);
    }
    a.ledger.holdable_cap = holdable_cap;
    if (cache_may_lock && sized > cache_lock_room) {
        note(a, "cache-lock", size_brief(cache_lock_room),
             in.lock_in_place_bytes_known
                 ? "The cache stops at " + size_brief(cache_lock_room) + ", all that can be locked in RAM."
                 : "A known lock bound limits the cache to " + size_brief(cache_lock_room) + ".",
             "the expert cache is bounded by what can be locked in place (" + mibs(cache_lock_room) +
                 ") rather than by what can be held (" + mibs(sized) +
                 "): past that it would be memory the machine may take back");
        sized = cache_lock_room;
    } else if (in.cache_lock && !cache_may_lock && in.lock_in_place_bytes >= model.token_cycle_bytes) {
        note(a, "cache-lock", size_brief(cache_lock_room),
             "The cache is not locked: " + size_brief(cache_lock_room) + " can be, one token needs " +
                 size_brief(model.token_cycle_bytes) + ".",
             "the expert cache is ordinary memory: " + mibs(cache_lock_room) +
                 " can still be locked in place after what is ahead of it, and one token cycle is " +
                 mibs(model.token_cycle_bytes));
    }
    a.cache_available_bytes = priced > 0 ? std::min(priced, sized) : sized;

    // The floor applies to whatever survived the reservation, not only to what the ranking proposed.
    // A cache under one token cycle evicts what the same token still needs, so it costs its memory
    // and returns no hits - and a bound that only guards one of the two paths into this number is
    // not a floor.
    const uint64_t floor = model.token_cycle_bytes;
    if (sized > 0 && sized < floor) {
        note(a, "cache-off", size_brief(sized),
             "No cache: " + size_brief(sized) + " is left after the dense weights, under the " + size_brief(floor) +
                 " one token reads.",
             "expert cache left off: " + mibs(sized) +
                 " survived the dense reservation but one token's routing reads " + mibs(floor) +
                 ". Under that floor the cache evicts a slice before the same token needs it again - an eviction "
                 "per read and no hits, measurably slower than no cache at all.");
        sized = 0;
    }

    if (priced > 0) {
        // What the ranking priced was priced under the caller's ceiling; where the ceiling was raised
        // for a locked or offered cache, the cache is what the difference is for. And what the ranking left
        // unspent goes to the cache too: the curve it priced on is a short record's, flat past the
        // experts those few tokens touched, so it calls a byte past them worth nothing, while a longer
        // answer routes to experts the record never saw and a byte left idle saves nothing at all.
        // Measured on a 16 GB card with 30 GiB of RAM, Qwen3.8-Flash-Next over 1024 tokens: the
        // curve's 15.7 GB cache held 83% and decoded 11.6 tok/s, 22 GB held 96% and decoded 20.7.
        const uint64_t raised = holdable_cap - in.budget_bytes;
        const uint64_t want = (priced + c.unspent + raised) & ~((1ull << 20) - 1);
        if (want > std::min(priced + raised, sized) && sized > priced + raised)
            note(a, "cache-remainder", size_brief(std::min(want, sized) - priced),
                 "The cache also takes the " + size_brief(std::min(want, sized) - priced) +
                     " nothing else wanted: idle RAM serves nothing.",
                 "this model's routing record prices the cache at " + mibs(priced) +
                     ", where its curve stops rising; a record is a few hundred tokens, and a longer answer routes to "
                     "experts it never saw, so the memory the ranking left unspent is the cache's too");
        sized = std::min(want, sized);
    } else if (sized > 0) {
        note(a, "cache-unpriced", size_brief(sized),
             "The cache gets what is left, " + size_brief(sized) + ": storage speed was not measured.",
             "the expert cache was sized from what remains after the dense policy (" + mibs(sized) +
                 ") rather than from what a hit is worth: this machine's read rate was not measured, so there is no "
                 "price to rank it against. It is the answer that was available, not the answer that was chosen.");
    }
    a.cache_bytes = sized;
    a.cache_locked_bytes = cache_protected ? sized : 0;
    a.cache_lock_unverified = cache_may_lock && !in.lock_in_place_bytes_known && sized > 0;
    if (a.cache_lock_unverified) {
        note(a, "cache-lock", "unmeasured", "The lock ceiling is unmeasured: ask for a runtime grant.",
             "the lock-in-place ceiling could not be read; the cache stays within the ordinary process budget "
             "and every known lock bound, with no margin reduction or predicted lock grant. The runtime asks "
             "to lock its pages; an automatically sized cache enables fitting to the grant, never below one "
             "token cycle. Explicit caller settings retain their authority");
    }
    GroupPlacement & p = a.at(WeightGroup::Experts);
    // The ranking may already have written a size that the reservation or the floor then took away.
    // Leaving it there makes the plan contradict itself - a placement claiming to hold memory the
    // plan says it will not spend - and a plan that disagrees with itself cannot be checked against
    // the run it produced.
    if (sized == 0) {
        p.resident_bytes = 0;
        return;
    }
    p.lane = Lane::ExpertStream;
    p.resident_bytes = sized;

    // The cost and the rationale, both from the size that was actually chosen.
    const double held = (double) sized / (double) in.cacheable_bytes;
    const HitCurve & curve = model.hit_curve;
    const bool from_curve = curve.credited() && model.expert_bytes > 0;
    // Scaled by what a load measured at its own budget (AllocationInputs::hit_scale): the curve as a
    // whole, so the sizes stay priced relative to each other.
    const double hit =
        std::min(1.0, from_curve ? curve.hit_share((uint64_t) ((double) sized * (double) model.expert_bytes /
                                                               (double) in.cacheable_bytes)) *
                                       in.hit_scale
                                 : (double) c.pol.cache_hit_optimism * in.hit_scale * held);
    a.cache_hit_credited = hit;
    p.seconds_per_token =
        seconds_at_mibs((uint64_t) ((1.0 - hit) * (double) model.token_cycle_bytes), c.stream_mibs) +
        finite_or_zero(seconds_at_gibs(model.group(WeightGroup::Experts).bytes_per_token, c.host_gibs));
    char buf[320];
    if (from_curve && curve.measured)
        std::snprintf(buf, sizeof(buf),
                      "streamed with a %s cache, %.0f%% of the expert set; credited hit rate %.0f%% [measured on "
                      "this model]: its own routing, %u decoded tokens, replayed through the cache at this size",
                      mibs(sized).c_str(), 100.0 * held, 100.0 * hit, curve.decode_tokens);
    else if (from_curve)
        std::snprintf(buf, sizeof(buf),
                      "streamed with a %s cache, %.0f%% of the expert set; credited hit rate %.0f%% [prior]: %.1f "
                      "token cycles, on the curve every model without a routing record is credited",
                      mibs(sized).c_str(), 100.0 * held, 100.0 * hit,
                      model.token_cycle_bytes ? (double) sized / (double) model.token_cycle_bytes : 0.0);
    else
        std::snprintf(buf, sizeof(buf),
                      "streamed with a %s cache, %.0f%% of the expert set; credited hit rate %.0f%%, which is "
                      "conservative on purpose and is the coefficient a run's own counters correct",
                      mibs(sized).c_str(), 100.0 * held, 100.0 * hit);
    p.reason = buf;
}

// ── 3. offer each resident group to a device ─────────────────────────────────────────────────
//
// Only RESIDENT groups are candidates. On a streamed group the read dominates the compute by more
// than an order of magnitude, so a faster device would hide behind the same I/O and buy nothing —
// and the memory it consumed would come out of the residency that does pay.
//
// Two gates before speed is even looked at. A device that does not reproduce the CPU's answer is
// excluded on CORRECTNESS. And the crossings the move creates must be PRICED: dense and expert
// halves alternate per layer, so a split placement crosses the boundary twice per layer, and a
// bandwidth ratio cannot see any of it. Where that price is unmeasured this declines, which is the
// whole difference between a planner and an oracle.
void offer_devices(Pricing & c) {
    const HardwareProfile & hw = c.hw;
    const ModelProfile & model = c.model;
    Allocation & a = c.a;
    bool any_device = false;
    bool wanted_device = false;

    for (int i = 0; i < (int) WeightGroup::count; ++i) {
        GroupPlacement & p = a.groups[i];
        const GroupDemand & d = model.group((WeightGroup) i);
        if (d.bytes == 0 || c.host_gibs <= 0.0) continue;
        const bool resident = p.lane == Lane::Resident || p.lane == Lane::Pinned;
        if (!resident) continue;

        // Each device against the host on the same experiment (bmoe/hardware_profile.h, RatePair): the
        // model's own dense matrices where both were measured on them, the synthetic GEMV otherwise.
        const ComputeDevice * best = nullptr;
        double t_host = 0.0, best_t = 0.0, best_ratio = 1.0;
        RatePair best_rates;

        for (size_t di = 0; di < hw.devices.size(); ++di) {
            const ComputeDevice & dev = hw.devices[di];
            if (dev.is_cpu) continue;
            const RatePair r = dense_rates(hw, dev);
            if (r.device <= 0.0 || r.host <= 0.0) continue;
            if (dev.identity_ok != Tri::Yes) continue; // correctness first, and it must be a fact
            if (dev.needs_repack == Tri::Yes) continue;
            // A device that cannot hold the group is not an option; and on a device whose memory IS
            // the host's the group's bytes are the SAME bytes, so counting them twice is how a
            // planner allocates memory that does not exist.
            if (dev.has_own_memory() && dev.memory_free && dev.memory_free < d.bytes) continue;

            const double th = seconds_at_gibs(d.bytes_per_token, r.host);
            const double t = seconds_at_gibs(d.bytes_per_token, r.device);
            if (t < th * best_ratio * (1.0 - c.pol.min_backend_win)) {
                best = &dev;
                best_t = t;
                t_host = th;
                best_ratio = t / th;
                best_rates = r;
            }
        }
        if (!best) continue;
        wanted_device = true;

        // The crossings this creates: the boundary is entered and left once per layer that has a
        // group on each side, every time the token passes it, and the experts are always on the
        // host today.
        const uint32_t crossings = 2 * std::max(1u, model.moe_layer_runs());
        if (best->split_seconds <= 0.0) {
            const std::string rates = std::to_string((int) std::lround(best_rates.device)) + " vs " +
                                      std::to_string((int) std::lround(best_rates.host)) + " GiB/s";
            note(a, "device-crossing", rates,
                 std::string("The ") + group_name((WeightGroup) i) +
                     " weights stay on the CPU: the device is faster, the hop to it "
                     "unmeasured.",
                 std::string("a device would consume this model's ") + group_name((WeightGroup) i) + " faster (" +
                     rates +
                     "), and it is NOT placed there: the move puts the expert half on the host and this half on the "
                     "device, which crosses the boundary twice per layer (" +
                     std::to_string(crossings) +
                     " times a token), and what a crossing costs is unmeasured on this machine. A bandwidth ratio "
                     "cannot see that cost, and an unmeasured number may not justify a move.");
            continue;
        }
        const double split_cost = best->split_seconds * crossings;
        if (t_host - best_t <= split_cost) {
            note(a, "device-crossing", "-",
                 std::string("The ") + group_name((WeightGroup) i) +
                     " weights stay on the CPU: the hop to the device costs more than "
                     "it saves.",
                 std::string("a device consumes this model's ") + group_name((WeightGroup) i) +
                     " faster, and the graph crossings the move creates cost more than it saves");
            continue;
        }
        for (size_t di = 0; di < hw.devices.size(); ++di)
            if (&hw.devices[di] == best) p.device = (int) di;
        // The saving at the ratio the pair measured, on the compute this group was priced with.
        const double t_priced = seconds_at_gibs(d.bytes_per_token, c.host_dense_gibs);
        p.seconds_per_token -= t_priced * (1.0 - best_ratio) - split_cost;
        p.reason += "; computed on " + best->name;
        any_device = true;
    }

    // Every non-host choice above is a recommendation until the engine can bind a tensor to a device
    // buffer for a group the streamer serves. Saying so is more useful than dropping it.
    a.device_applicable = !any_device;
    if (any_device)
        note(a, "device-unapplied", "-", "A device would be faster for some weights; this build cannot use it yet.",
             "a device won at least one group, but this build computes on the host: the placement is reported, not "
             "applied");
    if (!wanted_device && !hw.devices.empty() && hw.device_local_memory() == 0)
        note(a, "shared-memory", "-", "Every device shares the CPU's memory: moving weights there frees nothing.",
             "every device here reads the host's own memory, so moving a weight onto one frees nothing: the capacity "
             "tier is inert and only bandwidth was left to compare");

    for (const ComputeDevice & dev : hw.devices) {
        if (dev.is_cpu) continue;
        if (dev.identity_ok == Tri::No)
            note(a, "device-excluded", dev.name, dev.name + " gave results unlike the CPU's, so it is not used.",
                 dev.name + " did not reproduce the CPU's result on the probe graph, and was excluded on correctness "
                            "rather than on speed");
        if (is_yes(dev.host_ptr_buffers) && dev.host_ptr_verified == Tri::No)
            note(a, "device-host-pointer", dev.name,
                 dev.name + " cannot read memory it is handed, so streamed weights never go there.",
                 dev.name + " advertises host pointers and did not honour them when handed one: no streamed group "
                            "could ever be rebound there, whatever its speed");
    }
}

// ── the ledger, as it was charged ────────────────────────────────────────────────────────────
void write_ledger(Pricing & c) {
    const AllocationInputs & in = c.in;
    Allocation & a = c.a;
    if (a.ledger.holdable_cap == 0) a.ledger.holdable_cap = in.budget_bytes;
    // The two stores are one figure where they are one pool and two where they are not; the ceiling
    // a reader checks `locked` against is whichever the locked rows were charged to.
    a.ledger.lockable_cap = std::max(in.lockable_bytes, in.lock_in_place_bytes);
    a.ledger.lockable_cap_known = in.lockable_bytes_known && in.lock_in_place_bytes_known;
    auto row = [&](LedgerKind kind, uint64_t bytes, bool is_locked) {
        if (bytes == 0) return;
        a.ledger.rows.push_back({kind, ledger_kind_name(kind), bytes, is_locked});
        a.ledger.held += bytes;
        if (is_locked) a.ledger.locked += bytes;
    };
    row(LedgerKind::ContextCompute, in.fixed_bytes, false);
    row(LedgerKind::Projector, in.projector_bytes, false);
    row(LedgerKind::Checkpoints, in.checkpoint_bytes, false);
    row(LedgerKind::Device, in.device_bytes, in.device_locked);
    row(LedgerKind::Dense, in.reserved_bytes, a.dense_pinned);
    row(LedgerKind::ExpertCache, a.cache_bytes, a.cache_locked_bytes > 0);
    if (in.device_turn_bytes > 0) {
        // The prefill-time account: the cache yields up to all of itself, and the slots take their
        // turn. Where the cache is at least as large as they are - and locked like them, or neither
        // is - nothing more is held than while decoding. Otherwise what the slots need beyond what
        // the cache gives back has to fit the ceilings on its own.
        const uint64_t turn = in.device_turn_bytes;
        const uint64_t yields = std::min(a.cache_bytes, turn);
        const uint64_t held_then = a.ledger.held - yields + turn;
        const uint64_t lock_yields = in.device_locked ? std::min(a.cache_locked_bytes, turn) : 0;
        const uint64_t locked_then = a.ledger.locked - lock_yields + (in.device_locked ? turn : 0);
        a.turn_fits = held_then <= a.ledger.holdable_cap && (!in.device_locked || locked_then <= a.ledger.lockable_cap);
        a.ledger.rows.push_back(
            {LedgerKind::DeviceSlots, ledger_kind_name(LedgerKind::DeviceSlots), turn, in.device_locked, true});
    }
}

// ── totals ───────────────────────────────────────────────────────────────────────────────────
void add_totals(Pricing & c) {
    Allocation & a = c.a;
    for (int i = 0; i < (int) WeightGroup::count; ++i) {
        if (c.model.groups[i].bytes == 0) continue;
        a.seconds_per_token += finite_or_zero(a.groups[i].seconds_per_token);
        a.seconds_worst_case += finite_or_zero(a.groups[i].seconds_worst_case);
        a.resident_bytes += a.groups[i].resident_bytes;
    }
}

} // namespace

double host_compute_seconds(const HardwareProfile & hw, const ModelProfile & model) {
    double sum = 0.0;
    for (int i = 0; i < (int) WeightGroup::count; ++i) {
        const WeightGroup g = (WeightGroup) i;
        const GroupDemand & d = model.group(g);
        if (d.bytes == 0) continue;
        const double t = seconds_at_gibs(d.bytes_per_token,
                                         g == WeightGroup::Experts ? hw.host_expert_rate() : hw.host_dense_rate());
        if (!std::isfinite(t)) return 0.0;
        sum += t;
    }
    return sum;
}

Allocation allocate(const HardwareProfile & hw,
                    const ModelProfile & model,
                    const AllocationInputs & in,
                    const PlannerPolicy & pol) {
    Allocation a;
    a.budget_bytes = in.budget_bytes;
    for (int i = 0; i < (int) WeightGroup::count; ++i) {
        a.groups[i].group = (WeightGroup) i;
        a.groups[i].lane = Lane::Mmap;
    }

    Pricing c{hw, model, in, pol, a};
    measure_rates(c);
    charge_fixed(c);
    std::vector<Candidate> candidates = price_lanes(c);
    spend_ranking(c, candidates);
    hold_reserved(c);
    bound_cache(c);
    offer_devices(c);
    write_ledger(c);
    add_totals(c);
    return a;
}

} // namespace bmoe
