// The routing trace, its text form, the LRU replay and what is read from it. Read bmoe/routing.h
// first: it says why the planner reads a model's routing at all.

#include "bmoe/routing.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

namespace bmoe {

uint64_t RoutingTrace::expert_set_bytes() const {
    std::vector<bool> seen(expert_bytes.size(), false);
    uint64_t total = 0;
    for (const RoutingBatch & b : batches) {
        if (b.layer >= expert_bytes.size() || seen[b.layer]) continue;
        seen[b.layer] = true;
        total += expert_bytes[b.layer] * n_expert;
    }
    return total;
}

double HitCurve::hit_share(uint64_t cache_bytes) const {
    if (!credited() || points.empty()) return 0.0;
    if (cache_bytes <= points.front().cache_bytes) return points.front().hit;
    for (size_t i = 1; i < points.size(); ++i) {
        const HitPoint & a = points[i - 1];
        const HitPoint & b = points[i];
        if (cache_bytes > b.cache_bytes) continue;
        const double span = (double) (b.cache_bytes - a.cache_bytes);
        const double t = span > 0.0 ? (double) (cache_bytes - a.cache_bytes) / span : 1.0;
        return a.hit + t * (b.hit - a.hit);
    }
    return points.back().hit;
}

HitCurve prior_hit_curve(uint64_t expert_bytes, uint64_t token_cycle_bytes, uint32_t points) {
    HitCurve c;
    if (expert_bytes == 0 || token_cycle_bytes == 0 || points < 2) return c;
    c.prior = true;
    for (uint32_t i = 0; i < points; ++i) {
        const uint64_t bytes = (uint64_t) ((double) expert_bytes * i / (points - 1));
        const double cycles = (double) bytes / (double) token_cycle_bytes;
        const double held = (double) bytes / (double) expert_bytes;
        const double hit = k_prior_hit_ceiling * (1.0 - std::exp(-cycles / k_prior_hit_cycles));
        c.points.push_back({bytes, std::min(1.0, std::max(held, hit))});
    }
    return c;
}

// ── text ─────────────────────────────────────────────────────────────────────────────────────
namespace {

constexpr const char * k_native_magic = "# bmoe routing v1";
constexpr const char * k_route_trace_magic = "# route_trace v1";

// `key=value` tokens of a comment line, as scripts/trace_io.py reads them.
std::map<std::string, std::string> kv_tokens(const std::string & line) {
    std::map<std::string, std::string> kv;
    std::istringstream in(line);
    std::string tok;
    while (in >> tok) {
        const size_t eq = tok.find('=');
        if (eq == std::string::npos || eq == 0) continue;
        kv[tok.substr(0, eq)] = tok.substr(eq + 1);
    }
    return kv;
}

bool to_u64(const std::string & s, uint64_t & out) {
    if (s.empty()) return false;
    char * end = nullptr;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') return false;
    out = v;
    return true;
}

bool to_i64(const std::string & s, long long & out) {
    if (s.empty()) return false;
    char * end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') return false;
    out = v;
    return true;
}

// The preamble both formats share: the shape, then one line per block with its expert bytes.
bool read_preamble_line(const std::string & line, RoutingTrace & t) {
    const auto kv = kv_tokens(line);
    uint64_t v = 0;
    if (kv.count("layer") && kv.count("expert_bytes")) {
        uint64_t il = 0;
        if (!to_u64(kv.at("layer"), il) || !to_u64(kv.at("expert_bytes"), v) || il > (1u << 20)) return false;
        if (t.expert_bytes.size() <= il) t.expert_bytes.resize(il + 1, 0);
        t.expert_bytes[il] = v;
        return true;
    }
    if (kv.count("n_layer") && to_u64(kv.at("n_layer"), v)) t.n_layer = (uint32_t) v;
    if (kv.count("n_expert") && to_u64(kv.at("n_expert"), v)) t.n_expert = (uint32_t) v;
    if (kv.count("n_expert_used") && to_u64(kv.at("n_expert_used"), v)) t.n_expert_used = (uint32_t) v;
    if (kv.count("prefill_tokens") && to_u64(kv.at("prefill_tokens"), v)) t.prefill_tokens = (uint32_t) v;
    if (kv.count("decode_tokens") && to_u64(kv.at("decode_tokens"), v)) t.decode_tokens = (uint32_t) v;
    return true;
}

bool parse_native(std::istringstream & in, RoutingTrace & t, std::string & error) {
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '#') {
            if (!read_preamble_line(line, t)) {
                error = "unreadable preamble line: " + line;
                return false;
            }
            continue;
        }
        std::istringstream row(line);
        std::string phase;
        long long il = -1;
        if (!(row >> phase >> il) || (phase != "p" && phase != "d") || il < 0) {
            error = "unreadable batch line: " + line;
            return false;
        }
        RoutingBatch b;
        b.layer = (uint32_t) il;
        b.decode = phase == "d";
        long long e = 0;
        while (row >> e) {
            if (e < 0 || (t.n_expert && e >= (long long) t.n_expert)) {
                error = "expert id out of range: " + line;
                return false;
            }
            b.experts.push_back((int32_t) e);
        }
        t.batches.push_back(std::move(b));
    }
    return true;
}

// A --route-trace CSV, as scripts/route-replay.py rebuilds it: every turn together, the prompt as
// one batch per layer (its tokens' ids ordered by step, slot), then each decoded step's layers in
// block order.
bool parse_route_csv(std::istringstream & in, RoutingTrace & t, std::string & error) {
    using Pre = std::tuple<long long, long long, long long>; // step, slot, expert
    std::map<uint32_t, std::vector<Pre>> pre;
    std::map<std::pair<long long, uint32_t>, std::vector<std::pair<long long, long long>>> dec; // slot, expert
    std::set<uint32_t> layers;
    std::set<std::pair<long long, long long>> pre_steps, dec_steps; // (turn, step)
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '#') {
            if (!read_preamble_line(line, t)) {
                error = "unreadable preamble line: " + line;
                return false;
            }
            continue;
        }
        if (line.compare(0, 5, "turn,") == 0) continue;
        std::vector<std::string> f;
        std::string cell;
        std::istringstream row(line);
        while (std::getline(row, cell, ','))
            f.push_back(cell);
        if (f.size() < 9) continue; // the script skips a short row too
        long long turn = 0, phase = 0, step = 0, layer = 0, slot = 0, expert = 0;
        if (!to_i64(f[0], turn) || !to_i64(f[1], phase) || !to_i64(f[2], step) || !to_i64(f[3], layer) ||
            !to_i64(f[4], slot) || !to_i64(f[5], expert) || layer < 0 || layer > (1 << 20) || expert < 0 ||
            expert > INT32_MAX) {
            error = "unreadable row: " + line;
            return false;
        }
        layers.insert((uint32_t) layer);
        if (phase == 0) {
            pre[(uint32_t) layer].emplace_back(step, slot, expert);
            pre_steps.insert({turn, step});
        } else {
            dec[{step, (uint32_t) layer}].emplace_back(slot, expert);
            dec_steps.insert({turn, step});
        }
    }
    for (uint32_t il : layers) {
        auto it = pre.find(il);
        if (it == pre.end()) continue;
        std::sort(it->second.begin(), it->second.end());
        RoutingBatch b;
        b.layer = il;
        for (const Pre & p : it->second)
            b.experts.push_back((int32_t) std::get<2>(p));
        t.batches.push_back(std::move(b));
    }
    std::set<long long> steps;
    for (const auto & kv : dec)
        steps.insert(kv.first.first);
    for (long long step : steps) {
        for (uint32_t il : layers) {
            auto it = dec.find({step, il});
            if (it == dec.end()) continue;
            std::sort(it->second.begin(), it->second.end());
            RoutingBatch b;
            b.layer = il;
            b.decode = true;
            for (const auto & se : it->second)
                b.experts.push_back((int32_t) se.second);
            t.batches.push_back(std::move(b));
        }
    }
    t.prefill_tokens = (uint32_t) pre_steps.size();
    t.decode_tokens = (uint32_t) dec_steps.size();
    if (t.n_layer == 0 && !layers.empty()) t.n_layer = *layers.rbegin() + 1;
    return true;
}

} // namespace

std::string format_routing_trace(const RoutingTrace & t) {
    std::string out = k_native_magic;
    out += "\n";
    char buf[192];
    std::snprintf(buf, sizeof(buf), "# n_layer=%u n_expert=%u n_expert_used=%u prefill_tokens=%u decode_tokens=%u\n",
                  t.n_layer, t.n_expert, t.n_expert_used, t.prefill_tokens, t.decode_tokens);
    out += buf;
    for (size_t il = 0; il < t.expert_bytes.size(); ++il) {
        if (t.expert_bytes[il] == 0) continue;
        std::snprintf(buf, sizeof(buf), "# layer=%zu expert_bytes=%llu\n", il, (unsigned long long) t.expert_bytes[il]);
        out += buf;
    }
    for (const RoutingBatch & b : t.batches) {
        out += b.decode ? "d " : "p ";
        out += std::to_string(b.layer);
        for (int32_t e : b.experts) {
            out += ' ';
            out += std::to_string(e);
        }
        out += '\n';
    }
    return out;
}

bool parse_routing_trace(const std::string & text, RoutingTrace & out, std::string & error) {
    RoutingTrace t;
    std::istringstream in(text);
    std::string first;
    std::getline(in, first);
    if (!first.empty() && first.back() == '\r') first.pop_back();
    bool ok = false;
    if (first == k_native_magic)
        ok = parse_native(in, t, error);
    else if (first == k_route_trace_magic)
        ok = parse_route_csv(in, t, error);
    else
        error = "not a routing trace: it starts with neither '" + std::string(k_native_magic) + "' nor '" +
                k_route_trace_magic + "'";
    if (!ok) return false;
    for (const RoutingBatch & b : t.batches) {
        if (b.layer >= t.expert_bytes.size() || t.expert_bytes[b.layer] == 0) {
            // A layer that costs nothing would be admitted free and never evicted, and the replay would
            // report hits at every size: a wrong answer that looks like a good one.
            error = "no expert bytes for layer " + std::to_string(b.layer) + " in the preamble";
            return false;
        }
        // Checked here, after either format: the CSV's preamble can follow its rows, so the count is
        // only known once the file is read, and the replay indexes its tables by layer and id.
        for (int32_t e : b.experts) {
            if (e < 0 || (t.n_expert && (uint32_t) e >= t.n_expert)) {
                error = "expert id out of range: " + std::to_string(e) + " on layer " + std::to_string(b.layer) +
                        " of a trace over " + std::to_string(t.n_expert) + " experts";
                return false;
            }
        }
    }
    out = std::move(t);
    return true;
}

// ── the replay ───────────────────────────────────────────────────────────────────────────────
ReplayCounts replay_cache(const RoutingTrace & t, uint64_t budget, int half_life) {
    ReplayCounts r;
    const uint64_t n_expert = t.n_expert;
    if (n_expert == 0) return r;
    const size_t n_keys = (size_t) t.expert_bytes.size() * (size_t) n_expert;
    const bool freq = half_life > 0;
    const double growth = freq ? std::exp2(1.0 / half_life) : 1.0;

    // Resident entries in eviction order: (score, recency), coldest first. Under LRU every score is
    // 0 and recency decides; under decayed frequency the score does and recency breaks a tie, which
    // is the engine's order (its LRU list position) too. A use at pass p adds growth^p to the score
    // instead of decaying every score each pass, rescaled exactly where the engine rescales.
    using Key = std::pair<double, uint64_t>;
    using Order = std::set<std::pair<Key, uint32_t>>;
    Order order;
    // An entry that moves or leaves keeps its node: the replay is a move per routed id, and a
    // node freed and allocated again for each was most of what a curve cost.
    std::vector<Order::node_type> spare;
    std::vector<Key> key(n_keys, Key{0.0, 0});
    std::vector<double> score(n_keys, 0.0);
    std::vector<uint8_t> resident(n_keys, 0);
    std::vector<uint64_t> stamp(n_keys, 0); // 1 + index of the batch that last touched the key
    uint64_t used = 0, tick = 0;
    double weight = 1.0;
    int64_t last_layer = -1;

    const auto place = [&](uint32_t k, Key nk) {
        Order::node_type node;
        if (resident[k]) {
            node = order.extract({key[k], k});
        } else if (!spare.empty()) {
            node = std::move(spare.back());
            spare.pop_back();
        }
        key[k] = nk;
        if (node) {
            node.value() = {nk, k};
            order.insert(std::move(node));
        } else {
            order.insert({nk, k});
        }
    };
    for (size_t bi = 0; bi < t.batches.size(); ++bi) {
        const RoutingBatch & b = t.batches[bi];
        if (freq) {
            if ((int64_t) b.layer <= last_layer) {
                weight *= growth;
                if (weight > 1e200) {
                    const double s = 1.0 / weight;
                    for (double & f : score)
                        f *= s;
                    for (uint32_t k = 0; k < (uint32_t) n_keys; ++k)
                        if (resident[k]) place(k, Key{score[k], key[k].second});
                    weight = 1.0;
                }
            }
            last_layer = b.layer;
        }
        const uint64_t cost = t.expert_bytes[b.layer];
        const uint64_t mark = bi + 1;
        for (int32_t e : b.experts) {
            const uint32_t k = (uint32_t) (b.layer * n_expert + (uint64_t) e);
            if (stamp[k] == mark) { // a repeat inside the batch promotes and is not a lookup
                place(k, Key{score[k], ++tick});
                continue;
            }
            stamp[k] = mark;
            ++r.lookups;
            if (b.decode) ++r.decode_lookups;
            if (freq) score[k] += weight;
            if (resident[k]) {
                ++r.hits;
                if (b.decode) ++r.decode_hits;
            } else {
                used += cost;
                r.read_bytes += cost;
            }
            place(k, Key{score[k], ++tick});
            resident[k] = 1;
        }
        // Coldest first, skipping the batch's own entries.
        for (auto it = order.begin(); used > budget && it != order.end();) {
            const uint32_t victim = it->second;
            if (stamp[victim] == mark) {
                ++it;
                continue;
            }
            const auto next = std::next(it);
            spare.push_back(order.extract(it));
            it = next;
            resident[victim] = 0;
            used -= t.expert_bytes[victim / n_expert];
            ++r.evictions;
        }
    }
    return r;
}

HitCurve hit_curve(const RoutingTrace & t, uint32_t points, int half_life) {
    HitCurve c;
    c.decode_tokens = t.decode_tokens;
    c.prefill_tokens = t.prefill_tokens;
    const uint64_t set = t.expert_set_bytes();
    if (set == 0 || points == 0 || t.empty()) return c;
    c.points.push_back({0, 0.0});
    double best = 0.0;
    bool decoded = false;
    ReplayCounts r;
    bool bound = true; // the last size replayed evicted something
    for (uint32_t i = 1; i <= points; ++i) {
        const uint64_t size = (uint64_t) ((double) set * (double) i / (double) points);
        // A budget nothing was evicted under never decided anything, so a larger one replays the
        // same: a record touches a part of the expert set, and past it the curve is that last point.
        if (bound) {
            r = replay_cache(t, size, half_life);
            bound = r.evictions > 0;
        }
        decoded = decoded || r.decode_lookups > 0;
        best = std::max(best, r.decode_hit());
        c.points.push_back({size, best});
    }
    c.measured = decoded;
    if (!decoded) c.points.clear();
    return c;
}

std::vector<HotExpert> hottest_experts(const RoutingTrace & t, size_t limit) {
    std::map<std::pair<uint32_t, int32_t>, uint64_t> uses;
    for (const RoutingBatch & b : t.batches)
        for (int32_t e : b.experts)
            ++uses[{b.layer, e}];
    std::vector<HotExpert> out;
    out.reserve(uses.size());
    for (const auto & kv : uses)
        out.push_back({kv.first.first, kv.first.second, kv.second});
    std::stable_sort(out.begin(), out.end(), [](const HotExpert & a, const HotExpert & b) {
        if (a.uses != b.uses) return a.uses > b.uses;
        if (a.layer != b.layer) return a.layer < b.layer;
        return a.expert < b.expert;
    });
    if (limit && out.size() > limit) out.resize(limit);
    return out;
}

void attach_routing(ModelProfile & m, const RoutingTrace & t, uint32_t n_expert_used, int half_life) {
    m.hit_curve = HitCurve{};
    auto reject = [&](std::string why) { m.hit_curve.rejected = std::move(why); };
    if (t.empty()) return reject("the stored routing trace is empty");
    if (!m.ok || !m.is_moe) return reject("the model has no routed experts to price a cache for");
    if (t.n_expert != m.n_expert)
        return reject("the trace routed over " + std::to_string(t.n_expert) + " experts and this model has " +
                      std::to_string(m.n_expert));
    if (t.n_expert_used != n_expert_used)
        return reject("the trace was recorded at top-" + std::to_string(t.n_expert_used) + " and this run routes top-" +
                      std::to_string(n_expert_used));
    if (t.decode_tokens < k_routing_min_decode_tokens)
        return reject("the trace decoded " + std::to_string(t.decode_tokens) + " tokens, under the " +
                      std::to_string(k_routing_min_decode_tokens) + " a curve is read from");
    std::set<uint32_t> traced;
    for (const RoutingBatch & b : t.batches)
        traced.insert(b.layer);
    if (!m.moe_layers.empty() && traced != std::set<uint32_t>(m.moe_layers.begin(), m.moe_layers.end()))
        return reject("the trace routed through " + std::to_string(traced.size()) + " blocks and this model has " +
                      std::to_string(m.moe_layers.size()) + " carrying experts");
    const double set = (double) t.expert_set_bytes();
    const double file = (double) m.expert_bytes;
    if (file <= 0.0 || set < file * (1.0 - k_routing_bytes_tolerance) || set > file * (1.0 + k_routing_bytes_tolerance))
        return reject("the trace's expert set is " + std::to_string((unsigned long long) (set / (1024.0 * 1024.0))) +
                      " MiB and this model's is " + std::to_string((unsigned long long) (file / (1024.0 * 1024.0))) +
                      " MiB");
    m.hit_curve = hit_curve(t, k_hit_curve_points, half_life);
    if (!m.hit_curve.measured) reject("the trace has no decoded tokens");
}

} // namespace bmoe
