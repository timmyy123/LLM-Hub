// The eval-callback adapter: the only bridge between llama.cpp's compute graph and the
// expert streamer, and it uses only the public callback ABI (llama_context_params.cb_eval).
//
// Two phases share one callback:
//
//   Capture — during a one-token warm-up decode (experts still mmap-resident), every
//   graph node is offered to the callback in its "ask" pass. We scan each node's sources
//   for the expert layout's weight tensors (named blk.<il>.<suffix>.weight) and record
//   the live ggml_tensor pointers per (layer, projection). No node is isolated. The
//   runtime then hands these to ExpertStreamSource::init, which rebinds them.
//
//   Stream — for real generation, we ask for only the routing nodes ffn_moe_topk-<il>.
//   ggml computes and synchronizes each alone, then calls back with the selected expert
//   ids materialized; we gather them respecting the view strides and call load_layer()
//   so the routed slices land before that layer's expert matmul runs.
//
// Streaming carries an optional third job, the route trace (set_trace): the same pass also
// asks for each layer's router-weight node and records which experts were routed, how the
// router weighted them, and whether they were already resident. It observes only — the ids
// handed to load_layer are the same traced or not — but it costs a barrier per layer (the whole
// weight chain on the first graph, then only its terminal node), so it stays off unless asked
// for. See bmoe/route_trace.h.
//
// And an optional fourth (set_predict_log): the expert-prediction probe, which asks for each
// layer's gate matmul so it can rank the NEXT layer's experts before that layer runs, and reports
// how often the ranking was right. It decides nothing — it exists to price a predictor before one
// is wired into the loading path. See bmoe/predict_stats.h.
#pragma once

#include "bmoe/expert_layout.h"
#include "bmoe/route_trace.h"
#include "bmoe/decode_trace.h"
#include "bmoe/predict_stats.h"
#include "bmoe/routing.h"
#include "bmoe/row_source.h"
#include "expert_stream_source.h"

namespace bmoe {
class DeviceExpertArena;
class DecideProbe;
} // namespace bmoe

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct ggml_tensor;

namespace bmoe {

class RouterHook {
public:
    RouterHook(const ExpertLayout & layout, int n_layer);
    ~RouterHook(); // joins the prediction worker, if one was started

    // Install as ctx_params.cb_eval / cb_eval_user_data before context creation.
    static bool c_eval(ggml_tensor * t, bool ask, void * user_data);

    void begin_capture(); // switch to capture; clears prior records
    void end_capture();   // stop recording (called after the warm-up decode)

    // After capture, the tensors found per (layer, projection). Entry.bound is false for
    // layers that produced no expert tensors (dense layers). file_off/nb2 are filled in
    // by the runtime from the gguf; here only .tensor is set.
    const std::vector<LayerExperts> & captured() const { return captured_; }
    std::vector<LayerExperts> & captured() { return captured_; }

    // After capture, the non-expert weight LEAVES seen in the graph, keyed by tensor name. These
    // are the persistent model weights the streamer leaves mmap-resident (embeddings, attention,
    // norms, lm_head); the --dense-weights anon policy rebinds them onto O_DIRECT buffers. The map
    // also holds graph inputs and KV tensors (same op-NONE leaf shape); the runtime filters it
    // against the gguf tensor set, which those do not belong to. Only .tensor is meaningful here.
    const std::unordered_map<std::string, ggml_tensor *> & captured_weights() const { return captured_weights_; }

    // Every distinct weight leaf OBJECT the graph read, deduplicated by address rather than by
    // name. The map above cannot answer this: a model whose output head is tied to its token
    // embedding table has two ggml_tensor objects carrying the same name over the same file
    // bytes, and a name-keyed map keeps one of them. Rebinding only that one leaves the twin
    // reading the model's mmap for the life of the run, which is what --dense-weights anon and
    // pinned exist to prevent. Order is first-seen, so a run is reproducible.
    const std::vector<ggml_tensor *> & captured_weight_objects() const { return captured_weight_objects_; }

    // After capture, the memory module's state tensors the graph read (KV cache, recurrent state):
    // llama.cpp names them cache_*, and the graph only ever reaches them through views, so they are
    // found through view_src. A prefill device needs them in memory it can address.
    const std::vector<ggml_tensor *> & captured_state_objects() const { return captured_state_objects_; }

    // While set, the graph being computed runs on a prefill device fed by `arena`: the hook isolates
    // each layer's routing node only to pace the arena there, and does nothing else of the streaming
    // path — the routing lives on the device and the arena, not the streamer, supplies the experts.
    void set_device_arena(DeviceExpertArena * arena) { device_arena_ = arena; }

    // Experimental decide() probe (decide_probe.h): while it is armed, the routing, weight and layer-output
    // nodes are offered to it on every graph, host or device. Observes only.
    void set_probe(DecideProbe * p) { probe_ = p; }

    // After capture, whether the last node of every layer was seen: what paces a dense arena, which
    // must load a layer's weights before its first node and so waits at the end of the one before.
    // The layers of the capture graph in the order it runs them, by the graph's own index, and the
    // block of weights each computes from (itself where the graph named none). What a consumer that
    // walks the graph layer by layer follows instead of counting up: a graph that runs a block at
    // several depths comes back to it.
    const std::vector<int> & layer_order() const { return layer_order_; }
    std::vector<int> layer_blocks() const {
        std::vector<int> out;
        out.reserve(layer_order_.size());
        for (const int il : layer_order_)
            out.push_back(block_of(il));
        return out;
    }
    bool learned_layer_ends() const {
        for (const std::string & n : last_node_)
            if (n.empty()) return false;
        return !last_node_.empty();
    }

    // After capture, the subset of those weights the graph only ever GATHERS ROWS from — the shape a
    // token embedding table has, and the one residency policy can exploit (see IRowSource). A name is
    // in this set only if EVERY node that referenced the tensor was a row gather taking it as the
    // table, and every such gather's index was materialized before the node ran (a graph input, or a
    // view of one). One reference of any other kind — a matmul over the same tensor, which is what
    // tied embeddings used as the output head look like — disqualifies it, because the whole tensor
    // is then read and row residency would be a fault per row.
    //
    // Derived from what the graph did, not from tensor names: no architecture appears in this rule,
    // and a model whose embedding table is multiplied simply never qualifies.
    std::unordered_set<std::string> row_gathered_weights() const;

    // After capture, the weight leaves some node read OTHER than as the matrix of a matmul (src0 of
    // MUL_MAT / MUL_MAT_ID): a norm's scale, a bias, a per-expert scale broadcast by REPEAT, or any
    // weight reached through a view or reshape first. A prefill device keeps the rest in its weight
    // layout; these are plain data every kernel must be able to address (see DeviceExpertArena).
    // Derived from the graph's ops, so no architecture or tensor name appears in the rule.
    const std::unordered_set<const ggml_tensor *> & non_matrix_weights() const { return non_matrix_weights_; }

    void set_source(IExpertSource * src) { source_ = src; } // non-null → stream mode

    // Install the row-gathered residency policy (see bmoe/row_source.h). Non-null makes the stream
    // pass check every gather node against it and make the rows present before the node runs.
    void set_row_source(IRowSource * src) { row_source_ = src; }

    // Temporal prefetch depth K: while streaming layer l, hint the source to read ahead the
    // experts the previous token used at layers l+1..l+K. 0 (default) disables it.
    void set_prefetch_layers(int k) { prefetch_layers_ = k; }

    // ── expert-prediction accuracy probe (diagnostics; see bmoe/predict_stats.h) ──────
    // Measures, without changing a byte of what gets loaded, how well the next layer's routing can
    // be guessed before that layer runs. Three predictors are scored against the routing the
    // router actually produced:
    //
    //   stale-gate  layer l's gate INPUT run through layer l+1's gate matrix. The residual stream
    //               moves little between layers, so the stale input ranks l+1's experts nearly as
    //               the real one will — a full layer earlier. Costs one GEMV per layer; needs no
    //               training and no model change.
    //   prev-token  what the PREVIOUS token routed at this layer: the predictor --prefetch already
    //               bets on, scored here so the two are comparable on the same run.
    //   fresh-gate  the layer's own matrix on its own gate input: the stale predictor with the
    //               staleness taken out, sharing its every line of code. A control, not a
    //               predictor — it must reproduce the routing the graph computes from those same
    //               two tensors, so anything it misses is this probe being wrong (a GEMV that
    //               disagrees with llama.cpp's, or a selection the architecture does not make by
    //               raw-logit ranking: an additive bias, group-limited routing). Read the other
    //               two against it: a fresh-gate below 100% caps what the stale one could show.
    //
    // Decode only, last token of the batch. Off by default: it isolates one extra node per MoE
    // layer and does a per-layer GEMV on the eval thread, so a probed run is not a benchmark run.
    void set_predict_log(bool on);

    // ── predictive prefetch (see MoeStreamConfig::predict_prefetch) ───────────────────
    // Act on the stale-gate prediction: hand the predicted next-layer experts to the source's
    // speculative read path, exactly as the temporal prefetch does with the previous token's ids.
    // Drop-aware by construction — with the drop policy armed, a predicted expert whose predicted
    // routing weight sits below the drop threshold is not speculated: if it misses, the policy
    // would discard it unread, so prefetching it buys nothing and costs the read. Independent of
    // set_predict_log (the probe adds scoring and the control on top of the same prediction).
    // spec_max: how many predicted misses a layer may speculate (0 = retention only). Retention of
    // predicted residents happens at every value — it costs zero bytes.
    void set_predict_prefetch(bool on, int spec_max = 2);

    // Joins the prediction worker. It reads the gate weights, which the source frees when it shuts
    // down, so the session's teardown stops it before that rather than at ~RouterHook.
    void predict_worker_stop();

    // ── route-ahead: commit to the stale-gate prediction (lossy; see MoeStreamConfig) ─
    // With n > 0, the expert selection of MoE layer L is REPLACED, during decode, by the ranking
    // layer L's own gate matrix produced on the hidden state N layers earlier in the same forward
    // pass — the prediction the probe scores, acted on as the routing itself. The router still
    // computes: its logits feed the weight chain (so the substituted experts carry their true,
    // renormalized routing weights) and the layer's own gate input feeds the prediction for L+N.
    // Layers 0..N-1, prefill, and any layer whose prediction is unavailable route normally.
    // The point: the experts of layer L are known a full N layers of compute before L runs, so a
    // prefetch of them is never late and never wrong — the 100%-hit regime no honest predictor
    // reaches, bought with a quality perturbation this flag exists to measure.
    void set_route_ahead(int n);
    // Whether route-ahead's prediction scores against the int8 mirror of the gates (true) or their
    // own float weights (false). Resolved by the session from MoeStreamConfig::gate_mirror; call
    // before set_route_ahead arms the worker.
    void set_gate_mirror(bool on) { gate_mirror_ = on; }

    // How hard route-ahead actually bit, and how far the committed choice sat from the router's
    // own: `overridden` routings had their ids rewritten, `passthrough` were eligible (decode,
    // layer >= N) but had no prediction to commit — the first decode token, plus any layer whose
    // gate matrix or input the hook could not read. hits/slots is the agreement between the
    // committed selection and what the router would have chosen, the same slot metric the probe
    // reports — here it is the measured routing perturbation, not an accuracy claim.
    long long route_ahead_overridden() const { return ra_overridden_; }
    long long route_ahead_passthrough() const { return ra_passthrough_; }
    long long route_ahead_slots() const { return ra_slots_; }
    long long route_ahead_hits() const { return ra_hits_; }

    // CPU nanoseconds the prediction worker spent ranking, and how many rankings it produced.
    // This is the feature's own arithmetic — one gate GEMV per MoE layer per token, in this
    // file's scalar loop rather than ggml's kernels — and it is the one cost that does NOT
    // appear in wall time on a machine with spare cores, which is exactly why it has to be
    // reported: on a 4-thread phone the same work competes for cores and DRAM bandwidth with
    // the decode it is trying to accelerate.
    long long route_ahead_gemv_ns() const { return ra_gemv_ns_.load(); }
    long long route_ahead_gemv_jobs() const { return ra_gemv_jobs_.load(); }
    // Eval-thread nanoseconds spent ISSUING the early reads (settle, residency query, retain,
    // prefetch — page commits included). Separate from the GEMV because they live on different
    // threads: the GEMV is off the critical path by construction, this is not, and on a phone the
    // two are the whole per-layer tax route-ahead adds to the decode.
    long long route_ahead_issue_ns() const { return ra_issue_ns_; }
    // Eval-thread nanoseconds the sampled fresh-gate watchdog spent on its exact float GEMV — the
    // third and last route-ahead cost on the eval thread, metered like the other two.
    long long route_ahead_wd_ns() const { return ra_wd_ns_; }

    // ── route trace (diagnostics; see bmoe/route_trace.h) ────────────────────────────
    // When on, the hook additionally asks for each layer's router-weight node and records one
    // RouteTraceRow per routed expert. Rows buffer in RAM — the callback runs on a compute
    // thread mid-graph, so it must not do I/O — and the session drains them once llama_decode
    // has returned. Off by default: asking for the extra nodes costs a barrier per layer.
    void set_trace(bool on);

    // ── cache-aware expert dropping (lossy; see MoeStreamConfig::drop_cold_frac) ──────
    // `frac` > 0 arms the policy: a routed expert that is a cache MISS and carries less than
    // frac × (1/n_expert_used) of the routing's weight is discarded — not read, weight zeroed,
    // its slot pointed at the routing's top-weighted expert so the matmul still reads memory
    // that is certainly resident. `renorm` rescales the survivors to preserve the routing's
    // total mass. Off (frac == 0) the hook behaves exactly as before, bit for bit.
    void set_drop_policy(float frac, bool renorm, bool in_prefill);

    // ── cache-aware substitution (lossy; see MoeStreamConfig::substitute_lambda) ──────
    // `lambda` > 0 arms it: at every decode routing, resident experts get their router score
    // raised by lambda × the token's score range and the top-k is re-ranked, so a near-tie goes
    // to the expert already in RAM. Off (lambda == 0) the hook behaves exactly as before.
    void set_expert_substitute(float lambda);

    // Which phase the batch being decoded belongs to (0 = prefill, 1 = decode). The drop policy
    // is decode-only unless armed for prefill, and unlike the traces it must know this on every
    // run, so it cannot ride on begin_trace_batch.
    void set_batch_phase(int phase) { batch_phase_ = phase; }

    // Graph nodes whose output lives in a non-host buffer, i.e. that a device backend computed. The
    // proof a prefill device ran, as opposed to merely having the weights moved onto it: a backend
    // allocates the outputs of the ops IT runs in its own compute buffer. Counted only when armed,
    // since it costs a test per node on every graph.
    void count_device_nodes(bool on) { count_device_nodes_ = on; }
    long long device_nodes() const { return device_nodes_; }

    long long experts_routed() const { return experts_routed_; }
    long long experts_dropped() const { return experts_dropped_; }
    // Substitution's own ledger. Reranked counts every slot the policy examined (its denominator —
    // experts_routed above is the drop policy's, and stays 0 with dropping off); substituted counts
    // the slots whose expert it actually changed for one the router had not selected. A re-ranking
    // that returns the router's own SET, in whatever order, changes nothing and is not counted.
    long long experts_reranked() const { return experts_reranked_; }
    long long experts_substituted() const { return experts_substituted_; }

    // Prediction accuracy, aggregate and per layer (index = layer). Empty/zero unless the probe
    // was armed. `predict_unscored` counts routings the probe had to skip: the first token, whose
    // layers have not yet seen the next layer's gate matrix, plus any layer whose gate or hidden
    // state is a dtype the probe does not read. It is reported rather than folded into the
    // denominator — a skipped routing is not a wrong guess.
    const PredictorStats & predict_stale() const { return agg_stale_; }
    const PredictorStats & predict_stale2() const { return agg_stale2_; }
    const PredictorStats & predict_prev() const { return agg_prev_; }
    const PredictorStats & predict_self() const { return agg_self_; }
    const std::vector<PredictorStats> & predict_stale_by_layer() const { return ps_stale_; }
    const std::vector<PredictorStats> & predict_prev_by_layer() const { return ps_prev_; }
    const std::vector<PredictorStats> & predict_self_by_layer() const { return ps_self_; }
    long long predict_unscored() const { return predict_unscored_; }

    // ── compute trace (diagnostics; see bmoe/decode_trace.h) ────────────────────────
    // When on, the hook asks for EVERY node, which makes ggml compute and synchronize each one
    // alone — so the wall delta between consecutive callbacks is that node's real compute time,
    // and the major-fault delta across it is the flash re-read that time was actually spent on.
    // This is the only way to measure compute from outside llama.cpp, and it is expensive by
    // construction: a barrier per node, and no operator coalescing. Off by default; a traced run
    // is not a benchmark run. Independent of streaming — a dense baseline can be traced too.
    //
    // per_layer trades the per-op detail for fidelity: only the first node of each layer is
    // isolated (~n_layer barriers per token instead of thousands), coalescing and the async
    // expert prefetch survive, and each row aggregates one layer's segment (op "LAYER").
    void set_compute_trace(bool on, bool per_layer = false);

    // Frame the rows of one llama_decode, as begin_trace_batch does for the route trace. Rows are
    // stamped with `step`; a prefill chunk attributes its whole graph to the batch's last position,
    // since a node is computed once for the batch, not per token.
    void begin_compute_batch(int step, int phase, int turn);

    // Close the batch's final interval (layer-granularity only): the last layer's tail plus the
    // final norm and LM head have no successor boundary to observe them, so the session must call
    // this right after llama_decode returns — the row is charged the wall since the last boundary.
    void end_compute_batch();

    std::vector<ComputeTraceRow> & compute_rows() { return compute_rows_; }

    // Frame the rows of one llama_decode. `base_pos` is the context position of the batch's
    // first token and `n_tokens` its length, so a prefill chunk's rows carry real per-token step
    // numbers — and so a layer that saw fewer tokens than the batch can still be placed (see
    // flush_pending).
    void begin_trace_batch(int base_pos, int n_tokens, int phase, int turn);
    void end_trace_batch(); // flush the last layer, which has no successor to trigger it

    std::vector<RouteTraceRow> & trace_rows() { return trace_rows_; }

    // ── routing record (the planner's input; see bmoe/routing.h) ─────────────────────
    // Unlike the route trace this asks the graph for nothing: it keeps the ids each layer hands
    // load_layer, which the hook gathers on every routing anyway, so it costs a copy of a few ints per
    // layer per token and stops by itself. `shape` carries the record's static facts (blocks, experts,
    // top-k, each block's expert bytes); `max_tokens` bounds the decoded tokens kept and, separately,
    // the prompt tokens: a prompt longer than that keeps its tail, which is what the cache holds when
    // the decode starts. 0 records nothing.
    void set_routing_record(int max_tokens, RoutingTrace shape);
    // Frame one llama_decode (0 = prompt chunk, 1 = decode), as begin_trace_batch does for the trace;
    // end_record_batch closes it, so a graph run outside a generation is never recorded.
    void begin_record_batch(int phase, int n_tokens);
    void end_record_batch() { rec_on_ = false; }
    const RoutingTrace & routing_record() const { return rec_; }

private:
    // The eval callback, and the jobs it dispatches to in the order it runs them. Each job is only
    // reached when its feature is on, so a node no job wants costs the same few tests it always did.
    bool on_eval(ggml_tensor * t, bool ask);
    void ctrace_observe(const ggml_tensor * t);                 // compute trace, an isolated node
    void capture_sources(ggml_tensor * t);                      // capture warm-up, every node
    bool arena_eval(ggml_tensor * t, bool ask, bool probe_ask); // device prefill: the arena's pace
    void serve_rows(ggml_tensor * t);                           // row-gathered tables, before a node
    void rows_failed();                                         // a row read failed: the run ends
    void on_topk(ggml_tensor * t, int il);                      // a routing node, ids materialized
    void record_topk(ggml_tensor * t, int il);                  // the same node, recorded only: no streamer
    // Whether the routing record still wants routing nodes: armed, and not yet full.
    bool recording() const { return rec_max_ > 0 && rec_.decode_tokens < (uint32_t) rec_max_; }

    // Row-building state for the layer whose topk we last saw. A layer's weights arrive in a
    // LATER callback than its ids, so its rows can only be built once the whole weight chain has
    // been offered — i.e. when the next layer's topk arrives, or when the batch ends.
    struct PendingLayer {
        int layer = -1;
        int nu = 0, nt = 0;
        std::vector<int32_t> ids;
        std::vector<float> weights;
        std::vector<uint8_t> residency;
        std::vector<uint8_t> dropped; // set by the drop policy; all zero when it is off
    };
    void flush_pending();
    bool drop_armed() const;
    void apply_drop(ggml_tensor * weights);
    bool substitute_armed() const;
    void apply_substitute(ggml_tensor * ids, int il, int nu, int nt);
    void close_drop_layer();
    void ctrace_close_segment(int interval_layer, const char * tail_name);

    // Which tensors are the routed experts, and in which slot: read from the file before the load
    // (bmoe/expert_layout.h). By value, so it outlives whatever the caller read it into; a layout
    // that is not ok (streaming off, or a file refused) matches no tensor and nothing binds.
    ExpertLayout layout_;
    int n_layer_ = 0;
    bool capturing_ = false;
    // Graph layer -> the block whose weights it computes from, learnt at capture (-1: not seen, the
    // layer is its own block). They differ where a model runs a block at several depths.
    std::vector<int> block_of_;
    std::vector<int> layer_order_; // graph layers in the order the capture graph first reached them
    int block_of(int layer) const {
        if (layer < 0 || layer >= (int) block_of_.size()) return layer;
        return block_of_[(size_t) layer] >= 0 ? block_of_[(size_t) layer] : layer;
    }
    IExpertSource * source_ = nullptr;  // non-null → stream mode
    IRowSource * row_source_ = nullptr; // row-gathered dense tables, when the policy is on
    std::vector<LayerExperts> captured_;
    std::unordered_map<std::string, ggml_tensor *> captured_weights_;
    std::vector<ggml_tensor *> captured_weight_objects_; // same leaves, deduplicated by address
    std::unordered_set<const ggml_tensor *> captured_weight_seen_;
    std::vector<ggml_tensor *> captured_state_objects_;
    std::unordered_set<const ggml_tensor *> captured_state_seen_;
    DeviceExpertArena * device_arena_ = nullptr;
    DecideProbe * probe_ = nullptr;
    std::vector<std::string> last_node_; // per layer: name of its last node in the capture graph
    // Capture-time evidence for row_gathered_weights(): every weight seen as the TABLE of a row
    // gather, and every weight seen in any way that rules that out. The verdict is the difference.
    std::unordered_set<std::string> row_gathered_;
    std::unordered_set<std::string> row_disqualified_; // non-expert weight leaves (see captured_weights)
    std::unordered_set<const ggml_tensor *> non_matrix_weights_;
    std::vector<int32_t> gathered_; // reused scratch for stream-mode id gather
    std::vector<char> staged_ids_;  // a device-resident routing node's bytes, copied to the host

    // Temporal prefetch: K, and the previous token's routed experts per layer (last-token row
    // during prefill). Empty when prefetch is off or a layer has not been seen yet.
    int prefetch_layers_ = 0;
    std::vector<std::vector<int32_t>> prev_ids_;

    // Prediction (probe and/or prefetch). All of this is inert unless one of the two is on.
    //
    // gate_w_ is learned from the graph rather than looked up by name: the gate matmul's first
    // source IS the router matrix, whatever the architecture calls its tensor. It is a weight LEAF,
    // so unlike a graph intermediate the pointer stays valid across graphs — which is what lets
    // layer l predict for layer l+1 using a matrix layer l+1 has not reached yet this token.
    bool predict_log_ = false;
    bool predict_prefetch_ = false;
    std::vector<ggml_tensor *> gate_w_;
    std::vector<std::vector<int32_t>> pred_stale_;  // per layer, ranked one layer early
    std::vector<std::vector<int32_t>> pred_stale2_; // per layer, ranked TWO layers early (probe only)
    std::vector<std::vector<int32_t>> pred_self_;   // per layer, ranked from its own logits
    std::vector<PredictorStats> ps_stale_, ps_prev_, ps_self_;
    PredictorStats agg_stale_, agg_stale2_, agg_prev_, agg_self_;
    long long predict_unscored_ = 0;
    std::vector<float> pred_scores_; // scratch: one score per expert
    std::vector<float> pred_row_;    // scratch: one activation row, as float
    std::vector<int32_t> spec_ids_;  // scratch: the filtered prediction handed to prefetch()
    std::vector<uint8_t> pred_res_;  // scratch: residency of the prediction being filtered
    int pred_spec_max_ = 2;          // speculated predicted misses per layer (0 = retention only)

    // ── the prefetch's own prediction path (no barrier, GEMV off the eval thread) ─────
    //
    // The probe pays for its reading with an isolated node per layer; the prefetch cannot afford
    // that (the barrier alone measured ~20 ms/token) so it reads WITHOUT one: the gate matmul's
    // source pointers are stashed in the ask pass — which sees every node isolated or not — and
    // the gate-input row is copied at the topk callback, by which point the graph has certainly
    // computed it. What is NOT certain is that ggml's memory planner has not reused the buffer in
    // between; the watchdog below is what turns that risk into a measured quantity.
    //
    // The GEMV itself runs on a dedicated worker, and the horizon is TWO layers: h at layer l
    // ranks layer l+2. One layer ahead cannot work off-thread — the only eval callbacks between a
    // layer's gate and its own load are microseconds apart, so the result would always miss its
    // deadline. At l+2 the worker has a whole layer (~ms) for a ~0.5M-MAC job, and the result is
    // collected at layer l+1's load (predict_after_load), inheriting the same post-load issue
    // window the temporal prefetch uses. The price is one more layer of staleness; the probe's
    // stale-2 column is what says how much accuracy that costs on a given model. A result that is
    // not ready in time is skipped, never waited for.
    std::vector<ggml_tensor *> h_t_; // per layer, the gate matmul's input, stashed in the ask pass

    struct PredictJob {
        uint64_t seq = 0;
        int nl = 0; // layer the prediction is FOR
        int nu = 0;
        bool commit = false; // route-ahead: rank for commitment instead of building spec lists
        float drop_frac = 0.0f;
        int spec_max = 2;
        const ggml_tensor * gate = nullptr; // layer nl's gate matrix, snapshotted with the job
        std::vector<float> row;             // the gate-input row, copied on the eval thread
        std::vector<uint8_t> resident;      // residency bitmap of layer nl, snapshotted on the eval thread
    };
    struct PredictResult {
        uint64_t seq = 0;
        int nl = -1;
        bool commit = false;       // produced by a commit job; only route-ahead may consume it
        std::vector<int32_t> ids;  // commit jobs: the full ranking the topk of nl will commit
        std::vector<int32_t> spec; // predicted misses to speculate, confidence-ordered, capped
        std::vector<int32_t> keep; // predicted residents to retain (LRU-protect)
        bool ready = false;
    };
    std::thread pred_worker_;
    std::mutex pred_mtx_;
    std::condition_variable pred_cv_;
    PredictJob pred_job_;
    PredictResult pred_result_;
    bool pred_job_pending_ = false;
    bool pred_stop_ = false;
    uint64_t pred_seq_ = 0;

    // Watchdog for the barrier-less read. Every wd_interval routings the freshly-stashed row is
    // run through the layer's OWN gate and checked against the ids the router actually chose —
    // the same zero-staleness control the probe uses, sampled instead of continuous. If it drifts
    // below the threshold the buffer is being reused (or the architecture does not select by
    // raw-logit ranking) and the prefetch disarms itself, once, out loud. Without this, a planner
    // change in a future llama.cpp bump would turn the predictor into a silent noise generator.
    long long wd_slots_ = 0, wd_hits_ = 0, wd_rout_ = 0;
    bool wd_tripped_ = false;
    static constexpr int wd_interval = 512;         // routings between samples (~13 tokens at 40 layers)
    static constexpr double wd_min_frac = 0.98;     // below this, the read is not trustworthy
    static constexpr long long wd_min_slots = 2048; // do not judge before this many slots

    // Route-ahead (inert unless route_ahead_ > 0). ra_pred_[L] is the ranking committed at layer
    // L's topk, produced from the gate input of layer L-N in the same forward pass — filled and
    // consumed within one token, with a seq check so a result the previous token never collected
    // cannot be committed by this one. It rides the prefetch's whole barrier-less machinery: the
    // gate matrix and input pointers stashed in the ask pass, the row copied at the topk callback,
    // the GEMV on the shared prediction worker (predict_prefetch is mutually exclusive, so the
    // single job slot is never contended), and the same sampled fresh-gate watchdog validating the
    // read. The one thing a COMMITTED prediction changes is the failure mode: where a corrupt
    // prefetch wastes a read, a corrupt commit would BE the routing — so on a watchdog trip, and
    // for any layer whose prediction is not ready in time, the layer keeps the router's own
    // choice (counted as passthrough), never a stale or suspect one.
    int route_ahead_ = 0;
    std::vector<std::vector<int32_t>> ra_pred_;
    long long ra_overridden_ = 0, ra_passthrough_ = 0, ra_slots_ = 0, ra_hits_ = 0;
    bool ra_tripped_ = false;                                // watchdog verdict: read not trustworthy
    std::atomic<long long> ra_gemv_ns_{0}, ra_gemv_jobs_{0}; // the prediction's own CPU cost
    long long ra_issue_ns_ = 0;                              // eval-thread cost of issuing the reads
    long long ra_wd_ns_ = 0;                                 // eval-thread cost of the sampled watchdog

    // ── int8 mirror of the gate matrices, for the prediction only ────────────────────
    // The prediction re-reads a whole gate matrix per layer per token (n_embd x n_expert, ~4 MB
    // at F32) and only needs to ORDER 256 experts to take the top few. Measured on a phone, that
    // read is what the GEMV costs: ~160 MB/token of DRAM traffic, ~17-19 ms — bandwidth, not
    // arithmetic, which is why vectorising the loop three times over bought little. An int8
    // mirror with one scale per expert quarters the traffic for a ranking that survives it
    // trivially (the router's own logits differ by far more than an int8 step, and the sampled
    // fresh-gate watchdog is already the alarm if a model ever proves otherwise). Built lazily,
    // once per layer, from the same weight leaf the graph uses; ~1 MB per layer of extra RAM.
    // The graph's own routing never touches this — only the prediction does.
    // Whether commit jobs read the mirror at all: the session's resolved gate_mirror. Set before
    // route-ahead is armed and never while the worker runs.
    bool gate_mirror_ = false;
    std::vector<std::vector<int8_t>> ra_gate_q_; // per layer: n_expert rows of n_embd int8
    std::vector<std::vector<float>> ra_gate_s_;  // per layer: one dequant scale per expert
    std::vector<int8_t> ra_qh_;                  // scratch: the activation row, quantized per call
    // Both run on the prediction worker and take layer il's gate from the job, not from gate_w_.
    bool quantize_gate(int il, const ggml_tensor * w); // build the mirror for layer il; false if unsupported
    bool gate_scores_q(int il, const ggml_tensor * w, const std::vector<float> & h, std::vector<float> & out);
    std::vector<int32_t> ra_ids_, ra_keep_, ra_spec_; // scratch: the committed future, split by residency
    std::vector<uint8_t> ra_res_;                     // scratch: residency of the committed ids
    void route_ahead_submit(ggml_tensor * ids, int il, int nu, int nt); // watchdog + row copy + job
    void apply_route_ahead(ggml_tensor * ids, int il, int nu, int nt);
    void route_ahead_collect(int il); // pop a finished ranking; issue its reads if still ahead
    // Start reading layer nl's committed selection — `cand` is the worker's drop-aware issue list.
    void route_ahead_issue_for(int nl, const std::vector<int32_t> & cand);

    bool predict_on() const { return predict_log_ || predict_prefetch_; }
    void predict_reset();
    void predict_worker_main();

    // Rank the top `k` of `scores` into `out`, and score a prediction against a routing.
    static void rank_top_k(const std::vector<float> & scores, int k, std::vector<int32_t> & out);
    static void score_prediction(
        const std::vector<int32_t> & pred, const int32_t * actual, int k, PredictorStats & layer, PredictorStats & agg);
    // From a full score vector: the top-nu ranking, softmax-filtered against the drop threshold,
    // split into misses to speculate (capped) and residents to retain.
    static void build_spec_lists(const std::vector<float> & scores,
                                 int nu,
                                 float drop_frac,
                                 int spec_max,
                                 const std::vector<uint8_t> & resident,
                                 std::vector<int32_t> & spec,
                                 std::vector<int32_t> & keep);
    void predict_at_logits(ggml_tensor * logits, int il);
    void score_layer(int il, const int32_t * actual, int nu);
    void predict_at_topk(ggml_tensor * t, int il, int nu, int nt); // watchdog + job submission
    void predict_after_load(int il);                               // collect + issue once il's load is behind us

    // The probe's maximum prediction width. A routing wider than this would be scored against a
    // truncated prediction and read as a miss it never was, so the probe declines it instead.
    static constexpr int predict_max_k = 32;

    // Cache-aware dropping. Inert unless drop_frac_ > 0.
    //
    // The decision needs the FINAL router weights, and those are produced several nodes after the
    // topk that opens the layer — so load_layer() is postponed from the topk node to the terminal
    // node of the layer's weight chain, and the ids/weights are edited there, before the expert
    // matmul consumes either. Which node is terminal depends on the model's gating (norm, softmax,
    // scaled, or none of them), so it is LEARNED from the graph rather than tabulated per
    // architecture: term_variant_[il] fills in on the first graph, and until it does the layer loads
    // at its topk node undropped, exactly as with the policy off. That costs the first token of a
    // run its dropping and nothing else.
    //
    // A terminal node is remembered as the INDEX of its name in tensor_names::k_weight_nodes, not as
    // the name: the four chain names are mutually exclusive, so the index identifies it exactly, and
    // recording it costs neither an allocation nor a string compare on a path that runs for every
    // weight node of every layer of every token.
    //
    // Once it IS known, only that node is asked for. The rest of the chain exists to identify it,
    // and each extra ask is a graph split plus a compute-thread synchronization — per layer, per
    // token, on tensors of a few floats. A layer whose terminal is forgotten (below) goes back to
    // asking for the whole chain, so the learning is never one-way.
    float drop_frac_ = 0.0f;
    bool drop_renorm_ = true;
    bool drop_prefill_ = false;
    int batch_phase_ = 1; // 0 prefill, 1 decode
    long long experts_routed_ = 0, experts_dropped_ = 0;
    bool count_device_nodes_ = false;
    long long device_nodes_ = 0;

    // Cache-aware substitution. Inert unless sub_lambda_ > 0.
    //
    // Where dropping decides whether to PAY for a missing expert, this decides whether to need it
    // at all: a resident expert whose router score is close enough is run in its place. It costs
    // no flash read and no extra matmul, and the weights the graph applies still come from the
    // router's own scores for whichever experts end up selected. The scratch below is sized once
    // per layer width and reused for every token, so the steady state allocates nothing.
    float sub_lambda_ = 0.0f;
    std::vector<float> sub_scores_; // this token's scores as the graph ranked them, boosted for residents
    std::vector<int32_t> sub_all_;  // 0..n_expert-1, for the whole-layer residency query
    std::vector<uint8_t> sub_res_;  // residency of every expert in the layer
    std::vector<int32_t> sub_pick_; // the re-ranked selection
    long long experts_reranked_ = 0, experts_substituted_ = 0;

    struct PendingDrop {
        int layer = -1;
        int nu = 0, nt = 0;
        ggml_tensor * ids = nullptr; // the topk view, rewritten in place for dropped slots
        bool deferred = false;       // true when load_layer is waiting for the terminal weight node
    };
    PendingDrop drop_;
    std::vector<int8_t> term_variant_; // per layer, -1 until learned
    int8_t chain_last_ = -1;           // last weight node seen for drop_.layer while its chain runs
    std::vector<int32_t> drop_ids_;    // this layer's routed ids, kept across the deferral
    std::vector<float> drop_w_;        // scratch: the final weights
    std::vector<uint8_t> drop_res_;    // scratch: residency of each routed id
    std::vector<uint8_t> drop_mask_;   // scratch: which slots this layer dropped

    // Route trace. All of this is inert unless trace_on_.
    bool trace_on_ = false;
    int trace_base_pos_ = 0, trace_batch_n_ = 1, trace_phase_ = 0, trace_turn_ = 0;
    PendingLayer pending_;
    std::vector<RouteTraceRow> trace_rows_;
    std::unordered_set<int32_t> charged_; // per-flush scratch: experts already charged for a read

    // Routing record. Inert unless rec_max_ > 0, and once the decoded tokens reach it.
    int rec_max_ = 0;
    bool rec_on_ = false;     // the current llama_decode is being recorded
    bool rec_decode_ = false; // ... and it is a decode, not a prompt chunk
    bool rec_in_prompt_ = false;
    // The prompt chunks of the current prompt: the index of each one's first batch, and its tokens.
    std::vector<std::pair<size_t, int>> rec_chunks_;
    RoutingTrace rec_;

    // Compute trace. All of this is inert unless ctrace_on_. `ctrace_mark_` is the previous
    // isolation boundary: the node reported by the next callback is charged the wall since it.
    // Layer granularity keeps two cursors because ask and observe see different node streams:
    // ask_layer_ decides which nodes to isolate (every node is asked), obs_layer_ names the
    // segment an observed interval belongs to (only isolated nodes are observed). -1 = before
    // layer 0, i.e. the "pre" segment.
    bool ctrace_on_ = false;
    bool ctrace_layers_ = false;
    int ctrace_ask_layer_ = -1, ctrace_obs_layer_ = -1;
    int ctrace_step_ = 0, ctrace_phase_ = 0, ctrace_turn_ = 0;
    int ctrace_seq_ = 0;
    std::chrono::steady_clock::time_point ctrace_mark_;
    uint64_t ctrace_faults_ = 0;
    std::vector<ComputeTraceRow> compute_rows_;
};

} // namespace bmoe
