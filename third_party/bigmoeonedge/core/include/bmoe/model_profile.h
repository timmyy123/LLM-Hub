// What this gguf is, and what it demands per token, as facts a planning rule may read.
//
// The companion to HardwareProfile: between them they are the planner's whole input. Everything
// here is discovered from the file's own metadata and tensor shapes — no architecture name decides
// anything, no size threshold is written down, and a model family the engine has never seen
// produces a profile like any other. Where a field cannot be discovered it stays 0, and the rules
// that need it decline.
//
// The model asks for wildly different things from different parts of itself, which is the entire
// reason a single global policy leaves throughput on the table: a 30B MoE reads every byte of its
// attention weights per token and roughly one row of its embedding table, and pricing both as
// "dense weights" prices things that differ by three orders of magnitude identically. So the file
// is decomposed into groups by ACCESS SHAPE, and each carries its own per-token demand.
//
// On not hardcoding: which tensors are EXPERT weights is read from the file's shapes
// (bmoe/expert_layout.h), so a new MoE family needs nothing here. The remaining names —
// `token_embd`, `output`, `blk.N.attn_*`, `blk.N.ffn_*` — are the gguf naming convention itself,
// which every architecture llama.cpp converts shares; they are a property of the FORMAT, not of
// any model, and a name that matches nothing lands in Other and is priced as read-whole, which is
// the conservative direction: an unrecognised tensor can make a plan cautious, never reckless.
//
// Pure policy: no llama.cpp, no I/O. See core/src/plan/model_probe.cpp for the adapter that fills
// this from a gguf, and bmoe/planner.h for what reads it.
#pragma once

#include "bmoe/measured_knobs.h"

#include <cstdint>
#include <string>
#include <vector>

namespace bmoe {

// Weight groups that want different treatment, ordered roughly by how much a plan can win on
// them. The split is by ACCESS SHAPE, not by role: two tensors belong together when the same
// residency decision is right for both.
enum class WeightGroup {
    Embedding, // gathered by row, one row per token — resident for almost nothing
    Attention, // read whole, every token, every layer
    DenseFfn,  // read whole, every token: the shared (non-expert) FFN path
    Experts,   // read whole but SELECTIVELY: only the routed top-k of each layer
    Output,    // read whole, once per token, and only if it is a distinct tensor
    Other,     // norms, biases, anything unrecognised: small, and priced as read-whole
    count,
};

const char * group_name(WeightGroup g);

struct GroupDemand {
    uint64_t bytes = 0; // total size of the group in the file, MTP blocks excluded

    // Bytes one decoded token must obtain, from wherever they live. For Experts this is already
    // the top-k share, not the whole expert set: it is the demand, not the footprint. A block the
    // graph runs more than once per token is read that many times, so on such a model a group's
    // demand can exceed its size.
    uint64_t bytes_per_token = 0;

    // The graph only ever gathers ROWS from this group, so a resident copy buys residency for
    // bytes that will never be read. Only the embedding table can qualify, and only when it is
    // not also serving as the output head.
    bool row_gatherable = false;

    // The engine has a lane that can serve this group from flash today. A group that is not
    // streamable is a residency decision only; the planner may not propose a lane that does not
    // exist, and saying so here is how that stays true as lanes are added.
    bool streamable = false;
};

// How many of a token's expert reads an LRU expert cache of a given size serves, for THIS model: an
// LRU replay of a routing trace this model recorded (bmoe/routing.h), at sizes from nothing to the
// whole expert set. Routing does not depend on where an expert is computed or which machine ran it,
// so the curve is a property of the model (and of the kind of prompt it was recorded on), and it is
// over the WHOLE expert set: a caller sizing a cache for part of it scales by the share it serves.
//
// Absent (`measured` false) is the first load of a model, and every caller keeps the credit it had
// before there was a curve. A curve is never invented from a coefficient.
struct HitPoint {
    uint64_t cache_bytes = 0;
    double hit = 0.0; // decode lookups served from the cache, 0..1: what a decoded token is priced with
};

struct HitCurve {
    bool measured = false;
    // Not this model's routing: the prior every model without a record is credited (prior_hit_curve in
    // bmoe/routing.h). Priced as a curve, said as a prior, and never kept as a record.
    bool prior = false;
    // Ascending in cache_bytes, monotone in hit, starting at (0, 0). Between two points the share is
    // interpolated; past the last it holds the last.
    std::vector<HitPoint> points;
    uint32_t decode_tokens = 0;  // decoded tokens in the trace it was replayed from
    uint32_t prefill_tokens = 0; // prompt tokens in it
    // Why a trace that was offered was not used (a different model, too short): "" otherwise. A
    // rejected trace leaves `measured` false, and the plan is the one it would be without it.
    std::string rejected;

    // A curve the cost model prices by: this model's own, or the prior.
    bool credited() const { return measured || prior; }
    // The share served at `cache_bytes`, 0..1, non-decreasing in it. 0 when there is no curve.
    double hit_share(uint64_t cache_bytes) const;
};

struct ModelProfile {
    std::string arch; // for the rationale and for the record; never branched on
    bool is_moe = false;

    // Two different counts, and conflating them is a bug this header exists to prevent. The
    // absolute block index is llama.cpp's coordinate system — it is what `n_gpu_layers` counts
    // down from and what an override pattern names — while only some of those blocks carry
    // experts. An architecture with leading dense blocks (lfm2moe, bailingmoe3) has fewer of the
    // second than the first, so a pro-rata or a placement computed against the wrong one is
    // silently wrong on exactly the architectures that motivated the distinction.
    //
    // A third count joins them where a model runs some of its blocks more than once per token (a
    // looped core): the engine then numbers more layers than the file carries blocks, the later ones
    // running an earlier block's weights. `n_layer` is the engine's count, since a placement and a
    // context's memory are counted in it; `n_block` is the file's, since the weights are. The two
    // are equal on a model whose graph runs each block once, and nothing reads which model that is.
    uint32_t n_layer = 0;     // layers the engine numbers: llama.cpp's own layer indices
    uint32_t n_block = 0;     // of those, the ones the file carries weights for
    uint32_t n_moe_layer = 0; // of those, the ones carrying expert tensors — the ones that stream
    // Which blocks those are, ascending, in llama.cpp's indices: what a placement expressed in
    // blocks (n_gpu_layers, an override pattern) is read against, so a count of "layers on the
    // host" is a count of expert-carrying ones. Empty when the file was not read.
    std::vector<uint32_t> moe_layers;

    uint32_t n_expert = 0;      // experts per MoE layer
    uint32_t n_expert_used = 0; // routed per token (top-k)
    // The sliding window, in positions, from the file (0 without one): what a context checkpoint of a
    // windowed layer copies, at the cost per cell the placement probe measures.
    uint32_t n_swa = 0;

    // The hidden state's width. Not a size the streamer cares about — it is what a tensor handed
    // across a host/device boundary actually carries, so it is the shape the split probe measures
    // a crossing at. Zero when the file does not state it, and the probe then declines.
    uint32_t n_embd = 0;

    // How many expert tensors a layer has: three where gate/up/down are separate, two where gate
    // and up are fused. Read from which tensors the file actually carries, so a new fusion is a
    // different count rather than a new case.
    uint32_t n_expert_projections = 0;

    // ── bytes ───────────────────────────────────────────────────────────────────────
    // Everything below EXCLUDES the multi-token-prediction block. llama.cpp marks its tensors
    // TENSOR_SKIP and does not load them at all unless --mtp asks, experts and dense alike, so
    // counting them overstates what the run holds — measured at ~850 MiB on one model.
    uint64_t file_bytes = 0;           // the whole model, every shard
    uint64_t expert_bytes = 0;         // every routed-expert tensor
    uint64_t dense_bytes = 0;          // everything else: embeddings, attention, norms, head
    uint64_t largest_dense_tensor = 0; // the one a residency policy may not be able to hold
    // The MTP head file's size, where the run names one (SpecConfig::mtp_head): what placing the
    // head on a device costs that device. 0 where there is no file or it cannot be read.
    uint64_t mtp_head_bytes = 0;
    // The multimodal projector's size, where the run names one (RunConfig::mmproj): what llama.cpp's
    // library holds beside the model to read images, whole, on a device the run uses or on the host.
    // Dense, so never streamed. 0 where there is no file or it cannot be read.
    uint64_t projector_bytes = 0;
    // The tensors of the Embedding group, by name: what a plan that prices the group as left
    // mapped hands the run (MoeStreamConfig::dense_mapped), so the run leaves exactly those.
    std::vector<std::string> embedding_tensors;

    // The file that holds the most routed-expert bytes: where the streamer's reads land, so where a
    // storage probe has to read. A model in one file names that file. A split one does not name
    // its first shard, which can be metadata alone - a few MiB that the drive answers from its
    // own cache at a rate no read of the weights will see.
    std::string expert_file;

    // Every expert tensor of the MoE layer that has the most of them, in bytes. It is the unit a
    // device prefill reserves in: the layer being computed and the one being loaded behind it each
    // need room for a whole layer's experts, so two of these is what arming one costs.
    uint64_t largest_expert_layer_bytes = 0;
    // The same for a block's NON-expert tensors (attention, norms, shared experts): under
    // streaming a prefill device carries those through two slots of their own.
    uint64_t largest_layer_dense_bytes = 0;
    // Each block's non-expert bytes, by layer (`n_layer` long where the file was read, else empty). A
    // plan that moves layers between devices charges the ones it moves, not an average or a bound.
    // A layer that runs another block's weights carries none of its own, and is 0 here.
    std::vector<uint64_t> layer_dense_bytes;

    // The same bytes again, decomposed by access shape. `groups[...].bytes` sums to `file_bytes`,
    // and Experts::bytes equals `expert_bytes`; both spellings are kept because a capacity check
    // wants the totals and the cost model wants the per-group demand.
    GroupDemand groups[(int) WeightGroup::count];

    // One expert's slice of the largest projection in a layer: the granule the streamer actually
    // issues as a read, and therefore the request size to look up on the storage rate curve. This
    // is the bridge between the model and the hardware — the same granule sits at a different point
    // of a different machine's curve, which is why no lane count or request size can be a constant.
    //
    // The LARGEST rather than an average over every expert tensor: the curve is looked up to price
    // the read the engine will actually issue, and on a fused gate_up layout the two projections
    // differ by 2x, so an average prices a request nobody makes.
    uint64_t expert_slice_bytes = 0;

    // Bytes one token can demand at worst: every MoE layer routing a disjoint set. This is the
    // floor any expert cache must clear — under it the cache evicts what the same token still
    // needs, so it costs its memory and returns no hits at all.
    //
    // Summed over the tensors the file actually carries rather than multiplied out, because the
    // factors are not uniform: a fused gate_up projection is twice the stride of a split one, and
    // an architecture with leading dense blocks has layers that demand nothing. A product would be
    // wrong on both, and wrong in the direction that under-sizes the floor.
    //
    // Counted per layer the graph runs, not per block the file carries: each run of a block routes
    // on its own, so a block run three times can ask for three sets.
    uint64_t token_cycle_bytes = 0;

    // The quantization the expert tensors carry, as a ggml type id, or -1 when unknown. Kept as an
    // opaque integer so this header stays free of ggml; it exists to be handed back to a backend
    // when asking whether that backend can execute this model's expert matmul natively — and to
    // make a bandwidth probe measure the material the run is actually made of. Measuring in F32
    // instead answered 2 threads on a phone where the engine is 58% faster at 4: an F32 GEMV
    // saturates the bus at very few threads, a quantized one carries dequantization work per byte
    // and keeps scaling.
    //
    // A mixed quant carries several: `expert_type_id` is the one holding the most expert bytes,
    // `expert_type_ids` every one of them (ascending), so a device is asked about all the layouts
    // it would have to execute and not only the commonest.
    int expert_type_id = -1;
    std::vector<int> expert_type_ids;

    // ── shapes that constrain what may be done ──────────────────────────────────────
    // The output head is the embedding table. Two tensor objects over one range, so a policy that
    // rebinds one and not the other leaves the graph reading the mapping every token — and the
    // table is then projected against in full every token, which disqualifies it from row
    // streaming however large it is. No size threshold or architecture name is involved.
    bool tied_output_head = false;
    bool has_mtp = false; // a trained multi-token-prediction block is present

    // This model's own hit-rate curve, from a routing trace it recorded on an earlier run
    // (attach_routing in bmoe/routing.h). Not read from the file: a front-end that kept a trace hands
    // it back, and without one the curve is absent and the plan is what it always was.
    HitCurve hit_curve;

    // What an earlier load of this model measured about its lossless knobs on this machine and this
    // engine (bmoe/measured_knobs.h), the record a front-end kept and probe_and_plan matched by
    // hardware_fingerprint and engine version. Empty without one, and the plan is then the one a
    // first load gets.
    MeasuredKnobs knobs;

    bool ok = false; // false when the file could not be read; every rule declines
    std::string error;

    const GroupDemand & group(WeightGroup g) const { return groups[(int) g]; }
    GroupDemand & group(WeightGroup g) { return groups[(int) g]; }

    // Bytes a token must obtain if NOTHING is resident: the ceiling the plan works down from.
    uint64_t bytes_per_token() const;

    // How many expert-carrying layers one token passes through: `n_moe_layer` where each block runs
    // once, more where the graph runs blocks again. What a per-token cost paid once per routed layer
    // (a gate product, a boundary crossing) is multiplied by.
    uint32_t moe_layer_runs() const;
};

} // namespace bmoe
