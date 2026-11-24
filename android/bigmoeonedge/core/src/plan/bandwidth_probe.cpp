// What each compute engine can do with this model's weights, verified by doing it.
//
// Two questions, one graph. How fast does this engine pull weights out of the memory it will read
// them from — the number a bandwidth-bound batch-1 decode turns on, since every weight is read once
// and multiplied once so arithmetic is never the constraint. And does it produce the RIGHT answer,
// and honour a host pointer when handed one, because a planner that trusts an advertisement makes
// two silent mistakes: it believes `buffer_from_host_ptr` and plans to stream into a buffer that
// cannot be rebound, and it treats a device as a faster CPU without ever checking that it computes
// the same thing.
//
// The measurement is ONE graph run on every backend. That matters more than what the graph is: a
// ratio between a hand-rolled loop on one side and a vendor kernel on the other would measure two
// implementations rather than two paths, and the number is only ever used as a ratio. So the same
// `mul_mat` over the same bytes is scheduled on the CPU backend and on each device.
//
// The weight type is the MODEL's, not F32, and that is not a detail: an F32 matmul is pure
// bandwidth and stops improving as soon as the bus is full, while a quantized one carries
// dequantisation work per byte and keeps using cores past that point. Measured in F32 the thread
// sweep below answered 2 on a phone whose engine is 58% faster at 4.
//
// On the agreement test: cross-backend float reductions do not associate in the same order, so
// bit-equality is the wrong gate — a correct kernel fails it. The gate is the one llama.cpp
// applies to its own matmul kernels, so a device its tests pass is not refused here, and a wrong
// kernel still misses it by orders of magnitude.
//
// Bounded by construction: a buffer sized to outrun any last-level cache and nothing more, a few
// repetitions, everything released before returning. Where a backend will not allocate or will not
// run the op, that device keeps its unmeasured 0 and every rule above declines rather than
// assuming. A zero here is "unmeasured", and it is never to be read as "slow".

#include "bmoe/probe.h"

#include "../io/platform_io.h"
#include "layer_probe.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

namespace bmoe {

namespace {

// Big enough that the read comes from memory rather than from a cache, small enough to allocate
// anywhere this runs. A GEMV over a matrix this size reads every byte of it exactly once.
// The weight has to outrun the LAST cache, not the first, and that is a bigger number than it looks.
// At 4096x4096 a Q4_K weight is about 9 MB, which sits inside the system-level cache of a current
// phone SoC - so the probe measured that cache's bandwidth, a single core saturated it, and the
// thread sweep answered 1 on a device whose engine is 58% faster at 4. Four times that size puts the
// read back in memory where the decode's reads actually come from.
constexpr int64_t k_rows = 16384;
constexpr int64_t k_cols = 4096;

// The second instrument: the same matmul, many tokens wide. A one-token graph is bound by how fast
// weights can be READ, which is why the figure above is a bandwidth; a wide one multiplies every
// weight by a whole batch, so it is bound by how fast the engine can COMPUTE, and the two rank
// devices in opposite orders - measured on one unified-memory machine, the cores read 109 GiB/s
// against the accelerator's 64 and then lose a wide prefill to it seven times over. A prefill
// decision read off the one-token figure would be made on the wrong axis.
//
// The width is several times the narrowest graph a prefill device is ever handed, so the sample
// sits inside the regime it speaks for. The weight is smaller than the bandwidth probe's on
// purpose: nothing here depends on outrunning a cache, and at the full size a slow machine would
// spend seconds on a probe.
constexpr int64_t k_wide_batch = detail::k_wide_batch;
constexpr int64_t k_wide_rows = 4096;

// The weight type to measure with. The model's own where it has one; falling back to F32 makes the
// figure a raw bandwidth number rather than a number about this workload, and the plan says so.
ggml_type weight_type(const ModelProfile & model) {
    if (model.expert_type_id < 0) return GGML_TYPE_F32;
    const ggml_type t = (ggml_type) model.expert_type_id;
    return ggml_blck_size(t) > 0 ? t : GGML_TYPE_F32;
}

// The thread counts worth trying. Powers of two up to the core count, plus the core count itself:
// enough resolution to separate "half the cores" from "all of them", which is where the refuted
// rule went wrong, without turning a probe into a benchmark.
std::vector<int> thread_ladder(uint32_t n_cores) {
    std::vector<int> out;
    for (int t = 1; t <= (int) n_cores; t *= 2)
        out.push_back(t);
    if (n_cores > 0 && (out.empty() || out.back() != (int) n_cores)) out.push_back((int) n_cores);
    return out;
}

// Deterministic bytes, with enough variation that a broken kernel cannot pass by returning a
// constant. Deterministic matters: the agreement test compares devices against the CPU on the SAME
// bytes, and data regenerated per device would compare two different questions.
struct Probe {
    ggml_type type = GGML_TYPE_F32;
    int64_t cols = 0;
    int64_t rows = 0;
    int64_t batch = 1; // tokens the input is wide: 1 is the decode's shape, more is a prefill's
    std::vector<uint8_t> weight;
    std::vector<float> x;
};

bool build_probe(Probe & p, ggml_type wtype, int64_t rows = k_rows, int64_t batch = 1) {
    const int64_t blk = ggml_blck_size(wtype);
    p.type = wtype;
    p.cols = blk > 0 ? (k_cols / blk) * blk : k_cols;
    p.rows = rows;
    p.batch = batch;
    if (p.cols <= 0) return false;

    const size_t row_bytes = ggml_row_size(wtype, p.cols);
    if (row_bytes == 0) return false;

    // Every row is the same row, so one is encoded and copied down the matrix. The grid formats
    // search a codebook per block: encoding all 16384 rows took 13-16 s of a laptop core for the
    // 2- and 3-bit IQ types, paid before every `--auto` run, for bytes identical to the copies.
    // What the probe measures is reading them, and a copy is read exactly like the original.
    std::vector<float> src((size_t) p.cols);
    for (size_t c = 0; c < src.size(); ++c)
        src[c] = std::sin((float) (c % 1024) * 0.017f) * 0.5f;

    // Some formats (the 1- and 2-bit IQ grids) cannot be encoded without per-column importance, and
    // ggml asserts rather than declines when it is missing. These bytes are synthetic, so every
    // column matters equally: a uniform importance is the honest answer, and the kernel the probe
    // then times is exactly the one the model's own tensors run. Asked of ggml, not listed here.
    std::vector<float> importance;
    if (ggml_quantize_requires_imatrix(wtype)) importance.assign((size_t) p.cols, 1.0f);

    p.weight.resize(row_bytes * (size_t) p.rows);
    if (wtype == GGML_TYPE_F32) {
        std::memcpy(p.weight.data(), src.data(), row_bytes);
    } else if (ggml_quantize_chunk(wtype, src.data(), p.weight.data(), 0, 1, p.cols,
                                   importance.empty() ? nullptr : importance.data()) == 0) {
        return false;
    }
    for (int64_t r = 1; r < p.rows; ++r)
        std::memcpy(p.weight.data() + (size_t) r * row_bytes, p.weight.data(), row_bytes);

    // Every column differs from its neighbours, so a kernel that computed one token and repeated
    // it across the batch cannot agree with the reference.
    p.x.resize((size_t) (p.cols * p.batch));
    for (size_t i = 0; i < p.x.size(); ++i)
        p.x[i] = std::cos((float) (i % 512) * 0.011f + (float) (i / (size_t) p.cols) * 0.37f);
    return true;
}

struct RunResult {
    bool ok = false;
    double gibs = 0.0;
    std::vector<float> out;
};

// Run the probe on one backend. `host_ptr_buffer`, when true, asks the device to wrap OUR memory
// instead of allocating its own — the capability the whole streaming design depends on, tested by
// exercising it rather than by reading a flag.
RunResult run_gemv(ggml_backend_dev_t dev, const Probe & p, int n_threads, bool host_ptr_buffer) {
    RunResult r;
    if (!dev) return r;
    // A registered device can still refuse to open, and some say so by throwing: an accelerator
    // registers on any machine that has its driver and only opening a session finds hardware its
    // kernels were not built for. That is "unmeasured" like any other failure here, and a probe
    // must not be the thing that ends a plan.
    ggml_backend_t backend = nullptr;
    try {
        backend = ggml_backend_dev_init(dev, nullptr);
    } catch (const std::exception &) {
        backend = nullptr;
    }
    if (!backend) return r;

    // Ask the backend for its own thread setter rather than calling a CPU-specific function: it is
    // how ggml exposes every backend-specific entry point, so a backend that has one is configured
    // and one that has none simply runs as it is.
    if (n_threads > 0) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto set_threads =
            reg ? (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads")
                : nullptr;
        if (set_threads) set_threads(backend, n_threads);
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        ggml_backend_free(backend);
        return r;
    }

    ggml_backend_buffer_t own_buf = nullptr;
    ggml_backend_buffer_t host_buf = nullptr;
    std::vector<uint8_t> host_copy;

    ggml_tensor * w = ggml_new_tensor_2d(ctx, p.type, p.cols, p.rows);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.cols, p.batch);
    ggml_tensor * y = (w && x) ? ggml_mul_mat(ctx, w, x) : nullptr;

    // Ask before allocating: a backend without a kernel for this shape would otherwise be charged
    // for a fallback that says nothing about its own path to memory.
    if (y && ggml_backend_dev_supports_op(dev, y)) {
        bool placed = true;
        if (host_ptr_buffer) {
            // Our memory, handed to the device. If it really honours host pointers, the graph below
            // reads the values we wrote here without any copy — exactly what the streamer needs in
            // order to rebind a tensor onto a slice it just read from flash.
            host_copy.assign(p.weight.size() + pio::tensor_tail_slack(), 0); // readable past the last block
            std::memcpy(host_copy.data(), p.weight.data(), p.weight.size());
            host_buf = ggml_backend_dev_buffer_from_host_ptr(dev, host_copy.data(), p.weight.size(), 0);
            if (host_buf) {
                w->buffer = host_buf;
                w->data = host_copy.data();
            } else {
                placed = false;
            }
        }

        if (placed) own_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (placed && own_buf) {
            if (!host_ptr_buffer) ggml_backend_tensor_set(w, p.weight.data(), 0, ggml_nbytes(w));
            ggml_backend_tensor_set(x, p.x.data(), 0, ggml_nbytes(x));

            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y);

            // One untimed pass: the first call pays for kernel selection, JIT and lazy device
            // setup, and timing it would measure the setup rather than the device.
            if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
                const auto t0 = std::chrono::steady_clock::now();
                bool all_ok = true;
                int reps = 0;
                // Repeat until the sample is long enough to mean something, not a fixed number of
                // times. The clock is read each pass rather than a rep count computed up front,
                // because the rate this is trying to measure is the very thing such a count would
                // have to assume.
                while (all_ok && reps < detail::k_reps_max) {
                    all_ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
                    ++reps;
                    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() >=
                        detail::k_sample_seconds)
                        break;
                }
                ggml_backend_synchronize(backend);
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

                if (all_ok && secs > 0.0 && reps > 0) {
                    // Weight bytes APPLIED per second: every weight is used once per token of the
                    // batch, so at width 1 this is the bandwidth figure and at any other width the
                    // two sides of a ratio are still the same quantity.
                    r.gibs = (double) ggml_nbytes(w) * (double) p.batch * reps / secs / (1024.0 * 1024.0 * 1024.0);
                    r.out.resize((size_t) (p.rows * p.batch));
                    ggml_backend_tensor_get(y, r.out.data(), 0, r.out.size() * sizeof(float));
                    r.ok = true;
                }
            }
        }
    }

    if (own_buf) ggml_backend_buffer_free(own_buf);
    if (host_buf) ggml_backend_buffer_free(host_buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return r;
}

// Agreement, not bit-equality. See the file header.
//
// The measure and the bound are llama.cpp's: normalized mean squared error, sum((a-b)^2) /
// sum(a^2), against the 5e-4 that tests/test-backend-ops.cpp allows a MUL_MAT. This planner does
// not decide what precision a backend's kernels should have - upstream does, for every backend it
// ships - so a stricter private bound only refuses devices llama.cpp considers correct.
//
// It did. Correct kernels differ from the host by what their arithmetic differs in - the host
// multiplies a quantized weight by an activation it has itself quantized, a device typically by
// the float - measured in this unit at 1.6e-9 on a 4-bit type and 3.6e-8 on a 2-bit one at one
// token, 0.9e-7 to 1.6e-7 at prefill width. A prefill-width shader that accumulates in half
// precision sits much higher: an iGPU's Vulkan measured 4.4e-4 (2.1% RMS, every output on the
// half-precision grid). The previous bound, 1e-4 in this unit, refused it, and both of a phone's
// GPU backends with it, though upstream's own test passes them. A wrong kernel is still off by the
// size of the answer, an NMSE near 1.
using detail::agrees;

} // namespace

void probe_bandwidth(HardwareProfile & hw, const ModelProfile & model) {
    Probe probe;
    if (!build_probe(probe, weight_type(model))) return;

    const uint64_t needed = (uint64_t) probe.weight.size();

    // The CPU first and unconditionally: it is the reference every device is judged against, and a
    // comparison without it is not a comparison. Its own result defines correctness by definition.
    std::vector<float> reference;
    for (ComputeDevice & d : hw.devices) {
        if (!d.is_cpu) continue;
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(d.name.c_str());
        if (!dev) continue;
        const RunResult ref = run_gemv(dev, probe, 0, false);
        if (!ref.ok) continue;
        reference = ref.out;
        d.memory_bandwidth_gibs = ref.gibs;
        d.identity_ok = Tri::Yes;       // the CPU is the reference
        d.host_ptr_verified = Tri::Yes; // it computes on host memory and nothing else
        hw.host_bandwidth_gibs = ref.gibs;

        // How many threads this machine wants, asked rather than reasoned about. The same graph at
        // each rung of the ladder; the rung with the highest rate wins, and ties go to the smaller
        // count because threads are not free elsewhere. What this instrument sees is a memory-bound
        // matmul, which is what a batch-1 decode mostly is — not the whole of it, and the plan says
        // as much rather than presenting the number as a decode measurement.
        double best_rate = 0.0;
        uint32_t best_n = 0;
        for (int t : thread_ladder(hw.n_cores)) {
            const RunResult r = run_gemv(dev, probe, t, false);
            if (r.ok) hw.thread_rates.push_back({(uint32_t) t, r.gibs});
            if (r.ok && r.gibs > best_rate * 1.02) { // a rung has to beat the incumbent by more than noise
                best_rate = r.gibs;
                best_n = (uint32_t) t;
            }
        }
        if (best_n > 0) {
            hw.best_threads = best_n;
            hw.host_bandwidth_gibs = std::max(hw.host_bandwidth_gibs, best_rate);
            d.memory_bandwidth_gibs = hw.host_bandwidth_gibs;
        }
        break;
    }

    for (ComputeDevice & d : hw.devices) {
        if (d.is_cpu || !d.probe_here()) continue;
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(d.name.c_str());
        if (!dev) continue;
        // Do not take memory a device may be short of just to learn how fast it is. A device with
        // its own memory is skipped unless it has room to spare; the answer is worth having, and it
        // is not worth pushing a machine into a failed allocation to get. Host memory is not
        // guarded here because the residency budget already governs it.
        if (d.has_own_memory() && d.memory_free && d.memory_free < needed * 4) continue;

        const RunResult own = run_gemv(dev, probe, 0, false);
        if (!own.ok) continue;
        d.memory_bandwidth_gibs = own.gibs;

        // Correctness before speed. Without a reference nothing can be concluded, and an unproven
        // device stays Unknown rather than being credited with agreement it never demonstrated.
        if (!reference.empty()) d.identity_ok = agrees(own.out, reference) ? Tri::Yes : Tri::No;

        // The advertisement is not the fact. A device that returns a host-pointer buffer and then
        // reads something else closes the door to streaming with no error anywhere, so the only
        // answer worth recording is what happened when one was actually used.
        if (is_yes(d.host_ptr_buffers) && !reference.empty()) {
            const RunResult hp = run_gemv(dev, probe, 0, true);
            d.host_ptr_verified = (hp.ok && agrees(hp.out, reference)) ? Tri::Yes : Tri::No;
        }
    }

    // ── the wide instrument: who computes a prefill faster, and correctly ─────────────────────
    // Same shape of experiment as above - the host first, as the reference and the figure to beat,
    // then every device against it on the same bytes - with nothing carried over from the
    // one-token result: a device that lost there is measured here all the same, because losing
    // there says nothing about here. A device that will not run the wide graph keeps its
    // unmeasured 0 and its Unknown, and no rule arms it.
    //
    // The model's own layer answers this, at the prompt's own per-expert batch (layer_probe.cpp),
    // and decides identity at decode width on its real kernels too. The synthetic matmul below is
    // what a model whose layer cannot be read is measured with, as before.
    const bool real_layer = detail::probe_real_layer(hw, model);
    Probe wide;
    if (!real_layer && build_probe(wide, probe.type, k_wide_rows, k_wide_batch)) {
        std::vector<float> wide_ref;
        for (ComputeDevice & d : hw.devices) {
            if (!d.is_cpu) continue;
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(d.name.c_str());
            if (!dev) continue;
            const RunResult r = run_gemv(dev, wide, (int) hw.best_threads, false);
            if (!r.ok) break;
            wide_ref = r.out;
            d.wide_matmul_gibs = r.gibs;
            d.wide_identity_ok = Tri::Yes; // the reference
            hw.host_wide_matmul_gibs = r.gibs;
            hw.wide_batch = (uint32_t) k_wide_batch;
            break;
        }
        for (ComputeDevice & d : hw.devices) {
            if (d.is_cpu || !d.probe_here() || wide_ref.empty()) continue;
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(d.name.c_str());
            if (!dev) continue;
            if (d.has_own_memory() && d.memory_free && d.memory_free < (uint64_t) wide.weight.size() * 4) continue;
            const RunResult r = run_gemv(dev, wide, 0, false);
            if (!r.ok) continue;
            d.wide_matmul_gibs = r.gibs;
            d.wide_identity_ok = agrees(r.out, wide_ref, detail::k_max_nmse_device_wide) ? Tri::Yes : Tri::No;
        }
    }

    // Put it where a reader will see it. The label is the machine's one line of free text, printed
    // above the plan and carried into a benchmark's CSV header, and a figure that decides an
    // offload belongs in the description of the machine rather than only inside a rule that read it.
    if (hw.host_bandwidth_gibs > 0.0) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), ", %.0f GiB/s on this model's matmul", hw.host_bandwidth_gibs);
        hw.label += buf;
    }
}

} // namespace bmoe
