// What this machine is, as facts a planning rule may read.
//
// The whole point of this header is that it contains no platform names. Windows, Android, iOS,
// CUDA and Metal appear only in the adapters that FILL a HardwareProfile, never in the rules that
// consume one — so a rule can be unit-tested against a synthetic machine, and a new platform is a
// new adapter rather than a new branch. If a rule finds itself needing to know which platform it
// is on, the profile is missing a field; add the field, not the branch.
//
// Every probed fact is a tri-state. `Unknown` is not a defect to be papered over with a default:
// it is the input that makes "decline instead of guessing" mechanical, because a rule that has no
// fact leaves its knob where it was and says so in the plan's rationale.
//
// Pure policy: no llama.cpp, no OS headers, no I/O. See core/src/plan/hardware_probe.cpp for the
// adapter that measures these, and bmoe/planner.h for what reads them.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bmoe {

// A probed yes/no. Absent by default, because a profile that was never filled must not read as a
// machine where everything is false.
enum class Tri { Unknown, No, Yes };

inline bool is_yes(Tri t) {
    return t == Tri::Yes;
}

// What happens to bytes we are holding when the machine wants the memory back. This is the single
// fact behind the dense-residency policy, and it is the axis on which platforms genuinely differ:
// the same anonymous allocation is cheap to lose where it is compressed, expensive where it is
// written to a disk, and fatal where the kernel answers pressure by killing the process.
enum class Overflow {
    Unknown,  // not probed; a rule that depends on this must decline rather than assume
    None,     // nothing can take it back (a reclaim-exempt allocation)
    Compress, // compressed in RAM; a touch costs a minor fault plus decompression, invisible to I/O counters
    Swap,     // written out to a swap device; slow, but the process survives
    Refault,  // dropped, and re-read from the file it came from on the next touch
    Kill,     // the process is terminated for exceeding its share (iOS jetsam)
};

// Where a headroom figure came from. Carried for the rationale, so a reader can weigh the number:
// an estimate from what the compressor is currently achieving is worth less than a measurement, and
// both are worth more than the reported budget. No rule branches on this; rules read the bytes.
enum class Headroom {
    Unknown,      // not probed: rules fall back to the reported residency budget
    Compressible, // derived from the machine's own compression ratio and its reclaimable set
    Measured,     // held until the kernel took pages back, and the boundary observed
};

// A compute device the graph could run on, as ggml reports it. Kept deliberately thin: the planner
// only ever asks "how much memory does it have of its own, and can we address it from the host".
struct ComputeDevice {
    // Backend-reported name. No rule reads meaning into it; it is compared only with the device a
    // caller named themselves (`prefill.device`), to find that device's facts, and it is the name a
    // plan arms a device by.
    std::string name;
    std::string description; //
    // Which physical device this is, as the backend reports it (ggml_backend_dev_props::device_id:
    // the PCI bus id for a PCI device), empty when the backend does not say. Two backends that
    // drive the same card - CUDA and Vulkan on one NVIDIA board, HIP and Vulkan on one AMD board -
    // each register a device of their own, and this is what says they are one card rather than two.
    std::string physical_id;
    uint64_t memory_free = 0;
    uint64_t memory_total = 0;
    // True when this device IS the CPU backend. Kept separate from `host_memory` on purpose: an
    // integrated accelerator also reads host memory, and treating the two as one is how an iGPU
    // gets silently excluded from a question that was only ever meant to exclude the CPU.
    bool is_cpu = false;

    // Whether this device survived being used at all. A registered device is a claim by a driver,
    // and a driver can be wrong in the one way no in-process check survives: a backend that calls
    // into a function the driver does not have ends the process, and one that waits on hardware
    // that never answers ends the run just as surely. So a device can be probed somewhere it
    // cannot take this process with it, and this is what came back: Yes when it answered, No when
    // the probe died or ran out of time - with `unusable_reason` saying which - and Unknown where
    // nobody asked that way. A device that is No is measured by nothing and used by nothing.
    Tri usable = Tri::Unknown;
    std::string unusable_reason;
    // Set once this device's facts came from such a probe (or its failure): the probes that run in
    // this process then leave it alone, which is the whole point of having asked elsewhere.
    bool probed_out_of_process = false;
    bool probe_here() const { return !probed_out_of_process; }

    // Whether this device's memory IS the host's — one physical pool, so moving a tensor onto it
    // frees nothing and the capacity tier is inert, leaving only bandwidth to win.
    //
    // A TRI-STATE, and deliberately not a bool read off the device type. `TYPE_IGPU` is a claim of
    // shared memory and is believed, because believing it errs safe. `TYPE_GPU` is NOT evidence of
    // separate memory: Metal reports it on hardware whose memory is unified, and trusting it there
    // is how the same physical RAM gets counted twice — the accounting that took a phone down when
    // an integrated device was credited with "15195 MiB free" on a machine holding 11 GB. So the
    // type only ever narrows this to Yes; a No needs native topology or an allocation probe.
    // A known shared pool cannot become separate merely because its allocation delta was flat.
    //
    // Read it through the accessors below rather than directly. They encode the asymmetry once: an
    // unknown is treated as SHARED, because counting a shared pool as extra capacity overcommits a
    // machine, while counting separate memory as shared merely leaves it unused.
    Tri shares_host_memory = Tri::Unknown;

    // True only where it is KNOWN this device has memory of its own — the only case where the
    // capacity tier can buy anything.
    bool has_own_memory() const { return shares_host_memory == Tri::No; }
    // True where the device's memory is, or must be assumed to be, the host's.
    bool reads_host_memory() const { return shares_host_memory != Tri::No; }

    // Whether the buffers this device allocates are held reclaim-exempt, and so come out of the
    // same total the dense set and the expert cache are locked against. Where they are, a device
    // that is merely USED shrinks what can be protected, without a byte of it appearing in any
    // figure this process reads about itself. Unknown is charged like a yes, for the reason an
    // unknown `shares_host_memory` is read as shared: the error that overcommits is the costly one.
    Tri buffers_locked = Tri::Unknown;
    bool charges_lockable() const { return buffers_locked != Tri::No; }

    // True when this device can execute over a HOST buffer - pinned memory it reads directly rather
    // than a buffer of its own. It is the property that decides whether streamed experts can be
    // computed here at all, and it is not the same as having host memory: a discrete accelerator
    // across a link can offer one too, and then a weight we read from flash is both writable by us
    // and readable by it. Unknown until asked.
    Tri host_buffer = Tri::Unknown;

    // True when this device can wrap memory the CALLER already owns in a buffer of its own. It is
    // the capability the expert streamer would need, and it is not the same as offering a host
    // buffer: one hands us memory the device can read, the other reads memory we hand it. The
    // streamer allocates its own reservations and rebinds tensors onto them, so a device that can
    // only do the first still cannot see the bytes a streamed expert is actually made of.
    //
    // Yes is a fact; Unknown covers both "declined" and "declared nothing", and a reader must not
    // collapse it to No. The capability this is filled from is a blanket promise, and backends whose
    // support is conditional per device decline to make it while implementing the call anyway - so a
    // negative there is not evidence of absence. See probe_devices().
    Tri host_ptr_buffers = Tri::Unknown;

    // True when this device can copy while it computes. On a device across a link it is the
    // difference between a transfer that costs its own time and one that hides behind the previous
    // operation - a different axis from our own read/compute overlap, which hides compute behind I/O.
    Tri async_copies = Tri::Unknown;

    // True when a weight placed here can still have its `data` pointer rebound by us — the whole
    // mechanism of the expert streamer. Host memory qualifies, and so does a host buffer offered by
    // a device with memory of its own; a device-local buffer does not, because its pointer is not a
    // host address and nothing we read can be written into it.
    bool rebindable = false;

    // Whether this device can execute the model's own expert matmul on the model's own quantized
    // layout. Probed by asking the backend about the real operation rather than by listing formats,
    // so a backend that gains a kernel answers differently without anything here changing.
    // Unknown until asked; a rule that needs it declines.
    Tri runs_expert_op = Tri::Unknown;

    // Whether placing a weight here requires the loader to repack it into a device-specific layout.
    // A repacked expert tensor cannot be served by the streamer at all: the whole mechanism is
    // rebinding `data` onto the file's native layout, and a repack replaces exactly that.
    Tri needs_repack = Tri::Unknown;

    // Memory bandwidth in GiB/s, 0 when unmeasured. This is a NECESSARY input to an offload
    // decision and emphatically not a sufficient one, which is the most expensive lesson on this
    // branch: batch-1 decode is a chain of GEMVs that reads every weight once for a single
    // multiply-accumulate, so a device that shares the host's memory has little to win however fast
    // its ALUs are, and one with memory of its own wins in proportion to this — but a GEMV measured
    // on one buffer sees none of the costs below. Measured 20 GiB/s against the host's 12 on a
    // phone whose engine then ran the offload at 0.53x.
    double memory_bandwidth_gibs = 0.0;

    // The same device on ONE REAL LAYER of the model (docs/measured-planning.md, Step 3), at decode
    // width, in GiB/s of weight bytes applied; 0 when unmeasured. Two figures because the layer has
    // two halves that run different kernels on different shapes: `expert_gibs` is the routed expert
    // block (`mul_mat_id` over the layer's own expert tensors, top-k of them per token), `dense_gibs`
    // the layer's other matrices (attention projections, router, shared experts) as plain matmuls.
    // Built from the file's own tensors and bytes, so a kernel the format falls to, a shape that is
    // small enough to be dominated by per-op cost, and a threading pattern that does not scale are
    // all in the figure - which the synthetic GEMV above, one large matrix, cannot see: on one
    // 48-core host it read 77 GiB/s while the decode's experts ran at about 12.
    double expert_gibs = 0.0;
    double dense_gibs = 0.0;

    // The real layer many tokens wide, as weight bytes applied per second; 0 when unmeasured. It
    // is the figure a PREFILL placement turns on and it is not derivable from the one above: a
    // one-token graph is bound by reading weights and a wide one by computing on them, and a
    // device can lose the first and win the second several times over. Measured on the same real
    // layer as `expert_gibs` (the expert block at the per-expert batch a prompt of this width gives
    // it, the dense matrices at the full width), or on a synthetic matmul in the model's format where
    // no layer could be read. `wide_identity_ok` is whether it also reproduced the CPU's result at
    // that width - asked separately, because a backend's batched kernel is not its one-token path.
    double wide_matmul_gibs = 0.0;
    Tri wide_identity_ok = Tri::Unknown;

    // Whether this device reproduced the CPU's result on the probe graph. A device that computes
    // something else is excluded on CORRECTNESS, before speed is considered at all — and the
    // exclusion has to be a probe rather than a promise, since nothing in a backend's advertised
    // capabilities says "and the answers are right". Bit-equality is the wrong gate: cross-backend
    // float reductions do not associate in the same order, so a correct kernel would fail it.
    Tri identity_ok = Tri::Unknown;

    // Whether this device honoured a host-pointer buffer when actually handed one: the buffer was
    // created AND the graph read the values written through our own pointer. Distinct from
    // `host_ptr_buffers`, which is only the advertisement — and a device that returns a buffer and
    // then reads something else closes the door to streaming silently, with no error anywhere.
    Tri host_ptr_verified = Tri::Unknown;

    // What ONE host/device boundary crossing costs on this machine, in seconds; 0 when unmeasured,
    // and unmeasured means no group may be placed here rather than that crossings are free.
    //
    // This is the term a bandwidth figure structurally cannot see, and the one that decides an
    // offload in practice. It is measured the way the graph crosses at a layer boundary: one token's
    // hidden state copied from host memory into the device, one op run on it there, and copied back,
    // each leg waited for, half of the round trip per crossing. See core/src/plan/device_probe.cpp.
    double split_seconds = 0.0;

    // What a host half costs a layer of this device's graph, in seconds, 0 when unmeasured: a device
    // op, a host op at the decode's thread count, and a device op again, run through the scheduler as
    // a decode runs them, less the same ops all on the device. It is what a device cache's CPU share
    // adds to every MoE layer whatever the share computes (planner.cpp, choose_device_cache_cpu_share),
    // and `split_seconds` does not see it: a copy timed on its own misses the host's thread pool waking
    // and the device's graph being cut. Measured on a 32 GB card: a share took a 48-layer decode graph
    // from 50 to 146 splits, about 58 us each, where the copy alone read 11 us.
    double host_hop_seconds = 0.0;

    // Boundary crossings this device adds to ONE token's graph, 0 when unmeasured (no probe fills
    // it today; it is kept for the record and the facts line). It is the term
    // that decides an offload in practice and the one a bandwidth figure cannot see. Two measured
    // facts sit behind it. Dense and expert halves alternate per layer, so a split placement
    // crosses twice per layer — 96 crossings on a 48-layer model — and contiguous layer blocks,
    // the shape that should have avoided it, failed too: swept 0/2/4/8 on a 40-layer model, 2.398
    // tok/s at N=0 against 0.587 at N=2, with N=4 and N=8 faulting, while the I/O half of the trade
    // did exactly what the cost model predicted (cache hits 48% to 73%, re-reads 84.5 to 5.2 per
    // token). And merely REGISTERING a device costs crossings: 61 splits per token with nothing
    // placed on it, which is why a device the plan does not use must be kept out of the run rather
    // than merely left empty.
    uint32_t graph_splits = 0;

    // How fast this process's ordinary host memory reaches this device's own memory, in GiB/s, 0
    // when unmeasured or when the device has no memory of its own. It is the price of a miss in a
    // cache of experts kept on the device: the expert is uploaded from where the host holds it,
    // one expert at a time. Measured at that granularity from pageable memory, the kind a streamed
    // or resident expert lives in, rather than read off the link's nominal width.
    double upload_gibs = 0.0;

    // The same upload from a buffer of the device's own host buffer type
    // (ggml_backend_dev_host_buffer_type), 0 where the device offers none or it was unmeasured. That
    // type is host memory the device's backend allocates for itself to copy from, page-locked where
    // the backend can lock it, so its copies skip the bounce through a staging buffer an ordinary page
    // needs. What it is underneath is the backend's business; the rate is the fact.
    double upload_host_buffer_gibs = 0.0;
};

// The measured read rate at one (request size, lane count) point. A curve of these is the only
// honest way to answer "how many lanes" and "is our request size in the punished regime", because
// the answer differs by an order of magnitude between storage classes: a 4 KiB read returns 2% of
// peak on one desktop SSD and 6.8% on a phone's UFS.
struct RateSample {
    uint32_t request_bytes = 0;
    uint32_t lanes = 0;
    double mib_per_s = 0.0;
};

// The storage the model file lives on, and how it behaves when we read it the way we read it.
struct StorageFacts {
    uint32_t align = 0; // block size uncached reads must be aligned to; 0 when unknown

    // Whether cache-bypassing reads actually deliver correct bytes on THIS path. It is a per-path
    // fact, not a per-platform one: the same phone answers yes on one filesystem and silently wrong
    // on another, which is why the engine verifies rather than trusting the open's success.
    Tri direct_ok = Tri::Unknown;

    // Whether a live mapping of the file serialises our concurrent uncached reads. Measured, never
    // inferred from the platform: it is true on one desktop filesystem and false on a phone's, and
    // the difference is worth 46% of throughput.
    Tri mapping_serialises_reads = Tri::Unknown;

    std::vector<RateSample> rate_curve; // empty when unprobed

    // The two comparable points behind mapping_serialises_reads, in MiB/s, 0 when not measured.
    // Kept so the rationale can quote them: a verdict a reader can check beats one to be believed.
    double rate_mapped_mibs = 0.0;
    double rate_unmapped_mibs = 0.0;

    // The price of a REFAULT: page-sized random reads, in MiB/s, 0 when not measured. It is what a
    // weight left to the kernel's mapping costs per byte once memory pressure has dropped its
    // pages, and it is not a fraction of the sequential rate but a different order of magnitude —
    // 23 MiB/s against 2609 on the test host, 116x. Without it, leaving a group mapped looks nearly
    // free and the plan under-spends on residency.
    double refault_mibs = 0.0;

    // Spread between the fastest and slowest repeat at the point that decides something, as a
    // ratio; 0 when not measured, 1.0 when perfectly repeatable. A plan quoting a rate whose next
    // run contradicts it is worse than one that admits the range, and this is what lets the
    // rationale say so instead of printing three digits it cannot stand behind.
    double rate_spread = 0.0;

    // Best measured rate at or near `request_bytes` for `lanes`, or 0 when the curve says nothing
    // about that point. Nearest-sample lookup on purpose: interpolating a curve whose whole shape
    // is a latency floor would invent throughput between the samples.
    double rate_at(uint32_t request_bytes, uint32_t lanes) const;

    // The largest lane count whose measured rate at `request_bytes` is within `lane_tie_fraction`
    // of the best one, or 0 when unprobed. This is the only source of a lane number that is not a
    // constant carried over from one device.
    uint32_t best_lanes(uint32_t request_bytes) const;

    // The share of the best rate a lane count must reach to count as tied with it. 5% is the
    // probe's own resolution: lane counts closer than that swap places between repeated runs.
    static constexpr double lane_tie_fraction = 0.95;
};

struct HardwareProfile {
    // ── memory ──────────────────────────────────────────────────────────────────────
    // How many bytes THIS PROCESS may hold resident. On most systems that is the kernel's own
    // "available" figure; where a process has a share of its own it is that share instead. The
    // distinction matters: sizing from system-wide availability is meaningless under a per-process
    // limit, and it is the one number every sizing rule here reads.
    uint64_t residency_budget = 0;
    uint64_t memory_total = 0; // for the rationale, and for margins expressed as a fraction

    // What happens to our anonymous buffers under pressure. Decides how the dense set is held, and the margin.
    Overflow anon_overflow = Overflow::Unknown;

    // How many bytes this process can hold and expect to KEEP. It is a different question from
    // `residency_budget`, and on a machine whose reclaim compresses it has a different answer in
    // both directions: the kernel will compress other processes' idle pages to make room for us, so
    // the reported figure is a floor rather than a cap; and it will just as readily take ours back,
    // so a set that merely fits is not a set that survives. 0 means unmeasured, and every rule that
    // would read it then falls back to the reported budget and says which one it used.
    uint64_t holdable_bytes = 0;
    Headroom holdable_from = Headroom::Unknown;

    // Largest single reclaim-exempt allocation available, 0 where the platform has none. A store
    // the kernel may not take back is the only way to keep the dense set out of a compressed swap.
    uint64_t reclaim_exempt_max = 0;

    // How many bytes this process may still hold reclaim-exempt IN TOTAL, net of whatever is
    // already held that way by anyone the limit counts; 0 where nothing can be. A different fact
    // from the per-buffer figure above, and reading one for the other is how a plan asks for more
    // than the machine will grant: every buffer fits and the sum does not, the grant stops part
    // way, and what was refused is ordinary memory on a machine that then compresses it. A
    // platform whose store publishes no total reports the largest value there is, and the ledger
    // bounds it by what the process may hold at all.
    uint64_t lockable_bytes = 0;
    // A denied query is not a grant of zero. When false, bytes still carries any independently
    // known upper bound (including zero); it cannot establish a reclaim-exempt allocation.
    bool lockable_bytes_known = true;

    // The same question for memory the process ALREADY owns and asks to keep in place - which is
    // how the expert cache is protected, since its pages are committed one slice at a time. Two
    // facts rather than one because the two mechanisms are not the same store everywhere: a
    // machine can offer a reclaim-exempt allocation and refuse to lock a page in place, and then
    // the dense set can be protected and the cache cannot. Where they are one pool the two
    // figures are equal, and the ledger charges both against the smaller.
    uint64_t lock_in_place_bytes = 0;
    bool lock_in_place_bytes_known = true;

    // Whether the process can hand memory to the kernel to DROP under pressure, rather than swap
    // (pio::vm_offer). Where it can, the expert cache is offered instead of locked, so neither it
    // nor anything it would have pushed out is written to the drive.
    bool discardable_memory = false;

    // ── compute ─────────────────────────────────────────────────────────────────────
    uint32_t n_cores = 0; // 0 when unknown

    // The core classes this machine has, fastest first, as counts. A heterogeneous processor is not
    // described by a core count: a barrier waits for the slowest participant, so adding cores from a
    // slower class can cost throughput rather than add it. One entry means every core is alike.
    std::vector<uint32_t> core_classes;

    // The thread count that reached the highest rate on this machine's own memory-bound matmul,
    // 0 when unmeasured. It is a measurement rather than a rule for a reason: the rule that read
    // core classes instead - fast class only, because a barrier waits for its slowest participant -
    // was refuted by the first heterogeneous machine it met, where it cost 37% of throughput. How
    // many threads a machine wants is a property of that machine, and the honest way to have it is
    // to ask the machine.
    uint32_t best_threads = 0;

    // Every rung of that sweep, at the rate it reached: the counts the machine was asked about and
    // its answer for each, empty when unmeasured. A count past `best_threads` whose rate is here was
    // asked and lost, with every core left to it; the knob trial does not ask it again under a
    // decode, where the expert reads take cores of their own (planner.cpp, thread_candidates).
    struct ThreadRate {
        uint32_t threads = 0;
        double gibs = 0.0;
    };
    std::vector<ThreadRate> thread_rates;

    // How fast the HOST consumes weight bytes in THIS MODEL's own format, in GiB/s, 0 when
    // unmeasured. Not raw memory bandwidth, and the difference is the point: a quantized matmul
    // carries dequantisation work per byte, so it reaches a different rate and scales differently
    // with threads than an F32 read of the same memory would. Paired with a device's figure from the
    // same graph, it answers the only question that decides an offload - who consumes this model's
    // weights faster - which is what batch-1 decode is, every weight read once and multiplied once.
    double host_bandwidth_gibs = 0.0;

    // The host on the real layer at decode width (see ComputeDevice::expert_gibs and dense_gibs), at
    // `best_threads`; 0 when unmeasured, and a rule reading either then falls back to the figure
    // above, which is what it read before the layer was measured.
    double host_expert_gibs = 0.0;
    double host_dense_gibs = 0.0;

    // The host on the wide matmul (see ComputeDevice::wide_matmul_gibs), and the width it was
    // measured at, for the rationale. 0 when unmeasured, and then no prefill device is armed.
    double host_wide_matmul_gibs = 0.0;
    uint32_t wide_batch = 0;

    // The router-gate GEMV route-ahead's prediction runs once per layer per token, timed on this
    // model's gate shape over the gate's float weights and over an int8 mirror of them, in
    // microseconds per layer; 0 when unmeasured. The mirror reads a quarter of the bytes and pays a
    // quantization of the activation per call, so which is faster is bandwidth against arithmetic -
    // a property of this machine, measured on one host as the float path winning and on a phone as
    // the mirror winning three times over. The instruction set predicts neither.
    double gate_gemv_f32_us = 0.0;
    double gate_gemv_int8_us = 0.0;

    std::vector<ComputeDevice> devices;

    // What this BUILD could see, which bounds what the machine appears to have. A backend that was
    // not compiled in and was not found beside the executable enumerates no device, so an empty
    // `devices` means one of two entirely different things and a rule that conflates them is
    // reporting a property of the build as a property of the hardware.
    uint32_t backends_linked = 0;
    uint32_t backends_loaded = 0; // found at run time as separate shared libraries
    bool backends_looked = false; // false: nothing was even enumerated
    // The registered backends that could offer a LOCAL device work is placed on: not the host's
    // own, not a library the host calls into, not one whose devices live on another machine. Zero
    // with `devices` empty beyond the host means the build carries no accelerator backend at all;
    // above zero, the build carries one and this machine gave it no device. The two counts above
    // cannot tell those apart, since a remote or helper backend registers like any other.
    uint32_t backends_placeable = 0;

    // ── storage ─────────────────────────────────────────────────────────────────────
    StorageFacts storage;

    // Free-text label for the rationale and the CSV header. Never parsed, never matched: it exists
    // so a committed benchmark says which machine it came from, not so a rule can read it.
    std::string label;

    // Total memory of every device that is NOT host memory. This is the only capacity the upstream
    // capacity fitter can actually buy with, so it is also the test for whether that tier is worth
    // running at all on this machine.
    uint64_t device_local_memory() const;

    // The host's rate on the expert block and on the dense matrices: the real layer where it was
    // measured, the synthetic GEMV otherwise. For a rule pricing the host alone.
    double host_expert_rate() const { return host_expert_gibs > 0.0 ? host_expert_gibs : host_bandwidth_gibs; }
    double host_dense_rate() const { return host_dense_gibs > 0.0 ? host_dense_gibs : host_bandwidth_gibs; }
};

// The host's and a device's rate on the same half of the layer, from the SAME experiment: the real
// layer where both were measured on it, the synthetic GEMV where either was not. A rule compares the
// two, and a ratio between two different experiments would be a ratio between two questions.
struct RatePair {
    double host = 0.0;
    double device = 0.0;
    bool real_layer = false; // both from the model's own layer
};
inline RatePair expert_rates(const HardwareProfile & hw, const ComputeDevice & d) {
    if (hw.host_expert_gibs > 0.0 && d.expert_gibs > 0.0) return {hw.host_expert_gibs, d.expert_gibs, true};
    return {hw.host_bandwidth_gibs, d.memory_bandwidth_gibs, false};
}
inline RatePair dense_rates(const HardwareProfile & hw, const ComputeDevice & d) {
    if (hw.host_dense_gibs > 0.0 && d.dense_gibs > 0.0) return {hw.host_dense_gibs, d.dense_gibs, true};
    return {hw.host_bandwidth_gibs, d.memory_bandwidth_gibs, false};
}

} // namespace bmoe
