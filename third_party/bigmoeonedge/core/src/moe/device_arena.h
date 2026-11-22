// Streamed experts for a prefill graph that runs on a device.
//
// A model larger than RAM cannot give the device a resident copy of its experts, and a prefill graph
// hundreds of tokens wide needs nearly every expert of every layer anyway. So the device gets TWO
// layer-sized slots of expert weights and every MoE layer's expert tensors are bound to one of them,
// alternating. While the device computes layer k out of one slot, a pool of loader threads fills the
// other with layer k+1: read each expert's bytes from the gguf, hand them to the backend through
// ggml_backend_tensor_set on a per-expert view (which is where a repacking backend lays them out in
// its own format). One flash pass per graph, hidden behind compute when compute is the longer of the
// two.
//
// The pacing point is the layer's routing node, which the router hook already knows how to find and
// isolate: when the scheduler stops there, every op of layer k-1 has finished, so the slot k+1 goes
// into is free, and nothing of layer k that needs experts has run yet. barrier() queues the next fill
// and waits for the current one. The routing itself is not read: at this width it selects nearly
// every expert, and on the device it is not in host memory to read.
//
// The layer's other matrices (attention, shared experts) can ride the same two-slot scheme
// (init_dense; its few weights that are not a matmul's matrix stay on the device instead). They are
// resident on the host already, so filling them is a copy, not a read; what they cost the device is
// two layers of them instead of all of them, which on unified memory is the difference between
// fitting the device's session and not. They are needed before the layer's routing, so they are
// paced one layer ahead, at the last node of the previous layer (learned from the capture graph,
// since no node name is common to every architecture).
//
// Decode never sees any of this. place(false) hands every expert tensor back to the binding it had
// (the streamer's), and the slots are not a cache: each graph refills them from layer 0.
#pragma once

#include "expert_stream_source.h"
#include "../io/file_reader.h"

#include "ggml-backend.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace bmoe {

class DeviceExpertArena {
public:
    DeviceExpertArena() = default;
    ~DeviceExpertArena();
    DeviceExpertArena(const DeviceExpertArena &) = delete;
    DeviceExpertArena & operator=(const DeviceExpertArena &) = delete;

    // `layers` is indexed by layer id, as the streamer receives it; unbound entries are dense layers
    // and get no slot. Layers may differ in an expert tensor's type (quantisers keep some ffn_down in
    // more bits): each projection gets a region of the slot sized for its largest variant, and one
    // slot tensor per variant at that address. `threads` loader threads, each with its own lane.
    //
    // `n_slots` is how many layers of experts the device holds at once. Two is the scheme above: a
    // layer is read while the one before it computes. One halves that memory and gives up the
    // overlap - each layer is read at its own routing node, with the device waiting - which is what
    // lets a model whose layer is large, or a machine with little to spare, use the device at all.
    // The result is the same either way; only when the reads happen differs.
    static constexpr int max_slots = 2;
    bool init(ggml_backend_dev_t dev,
              const std::vector<std::string> & shard_paths,
              const std::vector<LayerExperts> & layers,
              int threads,
              bool direct,
              int n_slots,
              bool on_demand,
              std::string & err);

    // `on_demand` (init): the expert slots are not held for the session. acquire() builds them
    // before a device graph and release() gives them back after the last one, so between prefills
    // their memory is whoever else's - which on this engine is the expert cache's, the one thing
    // that is idle while a device prefill runs and busy when it does not. Without `on_demand` the
    // slots are built in init and both calls do nothing. acquire() can fail (the device has no room
    // right now), and the caller then runs that graph on the host.
    bool acquire(std::string & err);
    void release();
    bool slots_held() const { return ctx_ != nullptr; }
    bool on_demand() const { return on_demand_; }

    // Also carry each layer's non-expert weights through two slots. `per_layer[il]` lists layer il's
    // weights (host resident, contiguous); layers may differ in what they hold. Call after init.
    //
    // `non_matrix` are the weights some op reads other than as a matmul's matrix (RouterHook::
    // non_matrix_weights). A backend may keep a WEIGHTS buffer in a form only its matmul kernels
    // address (Hexagon maps it for DMA only when DMA64 is on, and most other kernels refuse it), so
    // these live in plain device memory, in their own type. They are not streamed: every layer's
    // are written once, here, and stay. A backend may make a write to plain memory wait for its op
    // queue (Hexagon flushes it, and that queue is not thread-safe), so the loaders, which write while
    // a graph runs, must never write one; and they are norms, biases and convolution kernels, a few
    // MiB in all. The matrices ride the two WEIGHTS slots, where the backend lays them out for its
    // matmul and a write leaves the queue alone.
    bool init_dense(ggml_backend_dev_t dev,
                    const std::vector<std::vector<ggml_tensor *>> & per_layer,
                    const std::unordered_set<const ggml_tensor *> & non_matrix,
                    std::string & err);

    // Bind every expert tensor to its slot (true) or back to what it had before (false). The host
    // binding is taken at the moment of the swap, so whatever the streamer bound stays authoritative.
    void place(bool on_device);

    // The order the graph runs its layers in, and the block of weights each one computes from:
    // `layers[i]` is the i-th layer the graph runs, by the graph's own index, and `blocks[i]` the
    // layer id its weights are filed under (the index `init` and `init_dense` take). Read off the
    // graph by whoever captured it. Without a call the graph is taken to run layer 0, 1, 2... each
    // on its own weights, which is every model whose blocks run once; with one, a routing node of a
    // layer the lists do not name fails its graph. A graph that comes back to a block (a looped core) names it again:
    // what is "next" is then the next entry here, not the next index, and a block the slots still hold is not read
    // twice. Call after init and init_dense, between graphs.
    void set_graph_order(const std::vector<int> & layers, const std::vector<int> & blocks);

    // Start of a device graph: queue the first layers the graph runs. Each graph refills from its
    // first layer.
    void begin_graph();
    // At the routing node of graph layer `il`: queue the layer the graph runs next into the slot the
    // previous one just freed, then wait until this one's experts are in place. A layer with no slot
    // passes straight through. `topk` is that node (the routed ids); only the routed mode reads it.
    void barrier(int il, const ggml_tensor * topk = nullptr);

    // Routed mode (the session's default). Off, every layer's whole expert set is read, ahead of its
    // routing. On, a layer is read in two parts: ahead of its routing, the experts the previous graph routed
    // at that layer (the prediction); at its routing node, whatever the routing needs that the
    // prediction missed, ahead of anything else queued. The matmul reads only routed experts, so the
    // slot's other experts may hold anything and the result is the same bit for bit. A layer whose
    // routing covers more than `full_frac` of its experts gets the next layer read whole, as off does:
    // a long prompt routes nearly everything, and there the prediction can only lose.
    void set_routed(bool on, float full_frac) {
        routed_ = on;
        routed_full_frac_ = full_frac;
    }
    // Routed mode, summed over graphs: experts routed, and experts read at the routing node (the
    // reads the prediction missed, which the graph waited for).
    uint64_t routed_used() const { return routed_used_.load(); }
    uint64_t routed_demand() const { return routed_demand_.load(); }
    // At the last node of graph layer `il`: its dense slot is free, so queue the layer after the
    // next into it, and wait for the next one's. With no dense slots this is a no-op.
    void dense_barrier(int il);
    // End of a device graph: wait out anything still in flight (only a graph that stopped early
    // leaves any), so the next graph starts from empty queues.
    void end_graph();

    // Test hook: sleep before each layer's first upload (PrefillDeviceConfig::test_load_delay_us).
    void set_test_delay_us(int us) { test_delay_us_ = us; }
    // Test hook: routed mode skips its reads at the routing node (PrefillDeviceConfig::test_routed_skip_demand).
    void set_test_skip_demand(bool on) { test_skip_demand_ = on; }

    // Fill from the expert cache where it holds the expert (PrefillDeviceConfig::from_cache): a copy
    // from RAM instead of a flash read, same bytes. Null turns it off. Between graphs only.
    void set_cache(const ExpertStreamSource * cache) { cache_ = cache; }

    bool failed() const { return failed_.load(); }
    uint64_t read_bytes() const { return read_bytes_.load(); }
    uint64_t cache_bytes() const { return cache_bytes_.load(); } // copied from the expert cache
    double stall_seconds() const { return stall_ns_.load() * 1e-9; }
    size_t slot_bytes() const { return slot_bytes_; }
    int n_slots() const { return n_slots_; }
    size_t dense_slot_bytes() const { return dense_slot_bytes_; }         // both slots and the resident buffer
    size_t dense_resident_bytes() const { return dense_resident_bytes_; } // of which the non-matrix weights
    int dense_converted() const { return dense_converted_; }
    size_t dense_converted_bytes() const { return dense_converted_bytes_; }
    bool has_dense() const { return !dense_.empty(); }
    int n_layers() const { return (int) order_.size(); }

private:
    struct Proj {
        uint64_t file_off = 0;
        uint64_t nb2 = 0;
        int file_idx = 0;
    };
    struct Layer {
        int il = -1;
        ggml_tensor * t[ExpertLayout::max_slots] = {};
        Proj proj[ExpertLayout::max_slots];
        ggml_backend_buffer_t host_buffer[ExpertLayout::max_slots] = {};
        void * host_data[ExpertLayout::max_slots] = {};
        void * host_extra[ExpertLayout::max_slots] = {};
        int variant[ExpertLayout::max_slots] = {}; // which slot tensor of the projection this layer uses
    };
    // A slot tensor for one (type, shape) variant of a projection, and a view per expert into it.
    struct Twin {
        ggml_tensor * t = nullptr;
        std::vector<ggml_tensor *> views;
    };
    struct Task {
        bool dense = false;
        int k = 0; // expert task: index into order_; dense task: layer id
        int p = 0; // expert task: projection; dense task: tensor index within the layer
        int e = 0;
    };
    struct DenseLayer {
        std::vector<ggml_tensor *> t;    // the model's tensors
        std::vector<ggml_tensor *> twin; // their places: dense slot il % 2, or the resident buffer
        std::vector<int> streamed;       // which of them a graph writes again (the matrices)
        std::vector<const void *> src;   // host bytes, taken at init: t is rebound while filling
        std::vector<ggml_backend_buffer_t> host_buffer;
        std::vector<void *> host_data, host_extra;
        // A weight whose type the device's matmul does not take is carried in one it does (see
        // init_dense); its converted bytes live here, and the swap changes type and strides too.
        std::vector<std::vector<uint8_t>> converted;
        std::vector<ggml_type> host_type;
        std::vector<std::array<size_t, GGML_MAX_DIMS>> host_nb;
    };

    void worker(int lane);
    // Queue layer k's experts that `want` marks (all when null) and are not queued yet this graph;
    // `front` puts them ahead of everything waiting. Caller holds mu_.
    void schedule(int k, const std::vector<uint8_t> * want = nullptr, bool front = false);
    void schedule_dense(int il); // caller holds mu_
    // The block graph layer `il` computes from, and the one the graph runs `ahead` layers after it
    // (-1 past the end). Identity where set_graph_order named nothing.
    int block_at(int il, int ahead) const;
    // Index into order_ of a block, -1 for one with no experts or out of range.
    int k_of(int block) const {
        return block >= 0 && block < (int) k_of_layer_.size() ? k_of_layer_[(size_t) block] : -1;
    }
    // The next expert-carrying block the graph runs after graph layer `il`, as an index into order_.
    int next_k(int il) const;
    void rebuild_order(); // identity for the layers set_graph_order did not name

    std::vector<Layer> order_;    // bound layers, ascending by layer id
    std::vector<int> k_of_layer_; // layer id -> index into order_, -1 for dense layers
    // The graph's run order (set_graph_order): position -> block, and graph layer -> position.
    std::vector<int> run_block_;
    std::vector<int> pos_of_layer_;
    std::vector<int> given_layers_, given_blocks_; // what set_graph_order was handed, kept for a rebuild
    // Which block's weights each slot holds, or is being filled with (-1: none). A block that is
    // queued into a slot another block held starts from nothing; one queued again into its own slot
    // keeps what this graph already put there.
    int expert_owner_[max_slots] = {-1, -1};
    int dense_owner_[2] = {-1, -1};
    int n_expert_ = 0;
    int n_proj_ = 0;

    ggml_context * ctx_ = nullptr;
    // One or two expert slots (init's n_slots). Layer k lives in slot k % n_slots_.
    int n_slots_ = max_slots;
    ggml_backend_buffer_t slot_buf_[max_slots] = {};
    // twins_[s][p][variant]: every variant of projection p in slot s sits at the same address. Built
    // with the slots, views included, and never per fill: a repacking backend keeps per-tensor state
    // for each, so recreating them per fill would grow without bound. On demand they are rebuilt
    // per prefill together with the buffer that state belongs to.
    std::vector<Twin> twins_[max_slots][ExpertLayout::max_slots];
    size_t slot_bytes_ = 0;
    // The slots' layout, kept so they can be built again (build_slots): what init worked out once.
    bool on_demand_ = false;
    ggml_backend_dev_t dev_ = nullptr;
    ggml_backend_buffer_type_t buft_ = nullptr;
    size_t align_ = 0;
    size_t slot_size_ = 0;
    size_t n_slot_tensors_ = 0;
    size_t region_off_[ExpertLayout::max_slots] = {};
    std::vector<const ggml_tensor *> variants_[ExpertLayout::max_slots];
    bool build_slots(std::string & err);
    void free_slots();

    std::vector<std::unique_ptr<FileReader>> readers_;
    std::vector<void *> staging_; // one per lane
    std::vector<std::thread> threads_;

    std::mutex mu_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    std::deque<Task> queue_;
    std::vector<int> remaining_;              // per order_ index: tasks not yet finished for this graph
    std::vector<std::vector<uint8_t>> sched_; // per order_ index, per expert: queued this graph

    bool routed_ = false;
    float routed_full_frac_ = 0.85f;
    // Per order_ index: the experts the last graph routed there (the prediction for the next graph).
    std::vector<std::vector<uint8_t>> prev_;
    std::vector<bool> have_prev_;
    std::atomic<uint64_t> routed_used_{0}, routed_demand_{0};
    int in_flight_ = 0;
    bool stop_ = false;
    bool on_device_ = false;
    int test_delay_us_ = 0;
    bool test_skip_demand_ = false;

    std::vector<DenseLayer> dense_; // by layer id; empty when init_dense was not called
    ggml_context * dense_ctx_ = nullptr;
    // The dense layer slots, [data][parity] as init_dense sizes them: data 0 holds the matmul
    // matrices (usage WEIGHTS), data 1 every other layer weight (plain memory).
    ggml_backend_buffer_t dense_slot_[2][2] = {};
    size_t dense_slot_bytes_ = 0;
    size_t dense_resident_bytes_ = 0;
    int dense_converted_ = 0;
    size_t dense_converted_bytes_ = 0;
    std::vector<int> dense_remaining_;
    std::vector<bool> dense_scheduled_;
    std::vector<bool> dense_waited_; // this graph: someone waited for the layer before it ran

    std::atomic<bool> failed_{false};
    std::atomic<uint64_t> read_bytes_{0};
    std::atomic<uint64_t> cache_bytes_{0};
    const ExpertStreamSource * cache_ = nullptr;
    std::atomic<uint64_t> stall_ns_{0};
};

} // namespace bmoe
