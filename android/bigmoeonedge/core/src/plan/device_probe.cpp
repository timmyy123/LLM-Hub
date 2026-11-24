// What a device COSTS to use, measured by using it.
//
// Two facts live here, and they have nothing in common except that both are usually taken from a
// declaration and both have been wrong when taken that way.
//
// **Is this device's memory the host's?** The device type is not the answer. `TYPE_IGPU` is a claim
// of shared memory, believed because believing it errs safe; `TYPE_GPU` is not the opposite claim —
// Metal reports it on unified-memory hardware. Reading it as "has its own memory" is how one
// physical pool gets counted twice, which is the accounting that told a capacity fitter an
// integrated GPU had 15195 MiB free on a machine holding 11 GB, and took the machine down. So it is
// settled from the native memory topology where available, otherwise by allocating on the device
// and watching what the host loses. A shared topology is not contradicted by a flat allocation delta.
//
// **What does one host/device boundary crossing cost?** This is the term a bandwidth ratio cannot
// see and the one that decides an offload in practice: dense and expert halves alternate per layer,
// so a split placement crosses twice per layer. A device measured at 20 GiB/s against a host's 12
// still ran the move at 0.53x. Measured here the way the graph crosses at a layer boundary: one
// token's hidden state copied to the device, one op launched on it there, and copied back.
//
// Both are best-effort and both fail closed: where the measurement cannot be made the fact stays
// Unknown or zero, and every rule above declines rather than assuming. Nothing here writes a
// constant into the plan — what is in this file is HOW to ask, never the answer.
//
// Bounded by construction: tensors sized by the model rather than by a guess, samples sized by
// duration rather than by a repetition count, everything released on every path.

#include "bmoe/probe.h"

#include "../io/platform_io.h"
#include "memory_topology.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <vector>

namespace bmoe {

namespace {

using clock_t_ = std::chrono::steady_clock;

double seconds_since(clock_t_::time_point t0) {
    return std::chrono::duration<double>(clock_t_::now() - t0).count();
}

// ── is the device's memory the host's? ───────────────────────────────────────────────────────
//
// Allocate on the device, and see whether the host paid for it. On one pool the host's available
// memory falls by roughly the allocation; on two it does not move.
//
// The test has to survive a noisy signal: `MemAvailable` drifts by hundreds of megabytes on a live
// machine, and has been observed swinging by a gigabyte between consecutive runs on both a phone
// and a desktop. So the allocation is large enough that the answer is not a matter of degree, and
// the verdict is taken on HALF of it — a shared pool loses all of it, a separate one loses none,
// and anything in between is noise rather than a third case.
constexpr uint64_t k_probe_alloc = 256ull << 20;

Tri measure_shared_memory(ggml_backend_dev_t dev) {
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    if (!buft) return Tri::Unknown;

    pio::ProcessMemory own_before, own_after;
    const bool own_known = pio::process_memory(&own_before);
    const uint64_t before = pio::mem_available_bytes();
    if (before == 0 && !own_known) return Tri::Unknown; // neither ledger is published: nothing to compare

    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, (size_t) k_probe_alloc);
    if (!buf) return Tri::Unknown;

    // Touch it through the backend so the pages are real. A reservation the device has not backed
    // yet costs the host nothing, and would read as separate memory on a machine that has none.
    void * base = ggml_backend_buffer_get_base(buf);
    if (base) ggml_backend_buffer_clear(buf, 0);

    const bool own_still_known = own_known && pio::process_memory(&own_after);
    const uint64_t after = pio::mem_available_bytes();
    ggml_backend_buffer_free(buf);

    // This process's own ledger is the quiet witness: nobody else writes to it, so pages that
    // appear in it during the allocation are the allocation. It can only say yes - a driver that
    // keeps a shared pool's pages off the process's books leaves it flat, and then the host's
    // figure below is the one left to ask. On a unified-memory desktop the host figure alone gave
    // both answers on consecutive runs, and the wrong one counts a single pool twice.
    //
    // Resident plus swapped, not resident alone: a kernel that compresses or swaps our own idle pages
    // to make room for the allocation leaves the resident figure flat while the charge still grows,
    // and that flat figure is what read a phone's shared GPU as separate memory under pressure.
    const auto charged = [](const pio::ProcessMemory & m) { return m.rss_bytes + m.swap_bytes; };
    if (own_still_known && charged(own_after) >= charged(own_before) + k_probe_alloc / 2) return Tri::Yes;

    if (before == 0 || after == 0) return Tri::Unknown;
    const uint64_t lost = before > after ? before - after : 0;
    return lost >= k_probe_alloc / 2 ? Tri::Yes : Tri::No;
}

// ── what does one boundary crossing cost? ────────────────────────────────────────────────────
//
// The way the graph crosses at a layer boundary: one token's hidden state copied from host memory
// into the device, the device's work on it launched, and the result copied back, each leg waited
// for. The op is trivial on purpose - what is priced is the CROSSING, so the arithmetic on the far
// side must not show up in it - and it is real, because a crossing is a split the device has to be
// told to run. Half the round trip is one crossing.
//
// This replaces a measurement that differenced a chain of ops run all on the host against the same
// chain alternating between host and device, through the scheduler. It read zero on a discrete card
// whose backend runs asynchronously: the mixed arm came out no slower than the host arm, so a
// crossing that costs tens of microseconds went unpriced and the plan said so on every run. Timing
// the copies themselves, with every leg synchronized, cannot read zero on a device that is there.
constexpr double k_sample_seconds = 0.15;
constexpr int k_reps_max = 100000;

double measure_crossing(ggml_backend_dev_t dev, int64_t n_embd) {
    if (!dev || n_embd <= 0) return 0.0;
    ggml_backend_t backend = nullptr;
    try {
        backend = ggml_backend_dev_init(dev, nullptr);
    } catch (const std::exception &) {
        backend = nullptr;
    }
    if (!backend) return 0.0;
    double per_crossing = 0.0;
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ctx ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd) : nullptr;
    ggml_tensor * y = x ? ggml_scale(ctx, x, 1.0f) : nullptr;
    ggml_cgraph * gf = y ? ggml_new_graph(ctx) : nullptr;
    // A device that cannot run the op never crosses this way, and is left unmeasured.
    ggml_backend_buffer_t buf =
        (gf && ggml_backend_dev_supports_op(dev, y)) ? ggml_backend_alloc_ctx_tensors(ctx, backend) : nullptr;
    if (buf) {
        ggml_build_forward_expand(gf, y);
        std::vector<float> in((size_t) n_embd, 0.5f), out((size_t) n_embd, 0.0f);
        const auto cross = [&] {
            ggml_backend_tensor_set(x, in.data(), 0, ggml_nbytes(x));
            const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
            return ok;
        };
        if (cross()) { // the first pays for setting up the path, untimed
            const auto t0 = clock_t_::now();
            int reps = 0;
            bool ok = true;
            while (ok && reps < k_reps_max) {
                ok = cross();
                ++reps;
                if (seconds_since(t0) >= k_sample_seconds) break;
            }
            const double secs = seconds_since(t0);
            // The copy came back: a device that returned something else did not cross.
            if (ok && reps > 0 && secs > 0.0 && out[0] == in[0]) per_crossing = secs / reps / 2.0;
        }
        ggml_backend_buffer_free(buf);
    }
    if (ctx) ggml_free(ctx);
    ggml_backend_free(backend);
    return per_crossing;
}

// ── how fast does host memory reach the device's own? ───────────────────────────────────────
//
// One expert at a time, from ordinary pageable memory, into one device buffer, for long enough to
// mean something. That is the copy a device-side expert cache makes for every miss, and its rate is
// neither the link's nominal width nor a pinned-memory benchmark: on a rented card the two differ.
constexpr uint64_t k_upload_buffer = 64ull << 20;

// `from_host_buffer`: the source is a buffer of the device's own host buffer type instead of
// ordinary memory; 0 where the device has no such type.
double measure_upload(ggml_backend_dev_t dev, uint64_t granule, bool from_host_buffer) {
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    ggml_backend_t backend = buft ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    if (!backend) return 0.0;
    const uint64_t piece = std::clamp<uint64_t>(granule, 1ull << 16, k_upload_buffer);
    double gibs = 0.0;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 2;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ctx ? ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) k_upload_buffer) : nullptr;
    ggml_backend_buffer_t buf = t ? ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft) : nullptr;
    std::vector<uint8_t> pageable;
    uint8_t * src = nullptr;
    ggml_backend_buffer_t src_buf = nullptr;
    if (from_host_buffer) {
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
        src_buf = host_buft ? ggml_backend_buft_alloc_buffer(host_buft, k_upload_buffer) : nullptr;
        src = src_buf ? (uint8_t *) ggml_backend_buffer_get_base(src_buf) : nullptr;
        if (src) std::fill(src, src + k_upload_buffer, (uint8_t) 1);
    } else {
        pageable.assign(k_upload_buffer, 1);
        src = pageable.data();
    }
    if (buf && src) {
        uint8_t * host = src;
        const uint64_t n_pieces = k_upload_buffer / piece;
        uint64_t moved = 0, i = 0;
        ggml_backend_tensor_set(t, host, 0, piece); // the first copy pays for setting up the path
        ggml_backend_synchronize(backend);
        const auto t0 = clock_t_::now();
        while (seconds_since(t0) < k_sample_seconds) {
            const uint64_t off = (i++ % n_pieces) * piece;
            ggml_backend_tensor_set(t, host + off, off, piece);
            moved += piece;
        }
        ggml_backend_synchronize(backend);
        const double secs = seconds_since(t0);
        if (secs > 0.0 && moved > 0) gibs = (double) moved / secs / (double) (1ull << 30);
    }
    if (buf) ggml_backend_buffer_free(buf);
    if (src_buf) ggml_backend_buffer_free(src_buf);
    if (ctx) ggml_free(ctx);
    ggml_backend_free(backend);
    return gibs;
}

// ── what does a host half cost a layer of the device's graph? ─────────────────────────────────
//
// A device expert cache with a CPU share gives every MoE layer a host half: the layer's graph is cut,
// the host's thread pool wakes to compute its part, and the device's graph resumes. Measured as a
// decode meets it: a chain of `k_hop_layers` layers, each a device op then a host op at the decode's
// thread count, run through ggml_backend_sched, against the same chain all on the device. The ops
// are one token's hidden state scaled, trivial on purpose: what is priced is the hop, not arithmetic.
constexpr int k_hop_layers = 16;

double measure_host_hop(ggml_backend_dev_t dev, int64_t n_embd, int n_threads) {
    if (!dev || n_embd <= 0) return 0.0;
    ggml_backend_t gpu = nullptr, cpu = nullptr;
    try {
        gpu = ggml_backend_dev_init(dev, nullptr);
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    } catch (const std::exception &) {
    }
    double hop = 0.0;
    if (gpu && cpu) {
        if (n_threads > 0) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
            auto set_threads = reg ? (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
                                         reg, "ggml_backend_set_n_threads")
                                   : nullptr;
            if (set_threads) set_threads(cpu, n_threads);
        }
        // One timed chain: `mixed` puts every second op on the host.
        const auto time_chain = [&](bool mixed) -> double {
            ggml_backend_t backends[2] = {gpu, cpu};
            ggml_backend_sched_t sched =
                ggml_backend_sched_new(backends, nullptr, 2, 4 * k_hop_layers + 16, false, false);
            ggml_init_params ip{};
            ip.mem_size = ggml_tensor_overhead() * (4 * k_hop_layers + 16) + ggml_graph_overhead();
            ip.no_alloc = true;
            ggml_context * ctx = sched ? ggml_init(ip) : nullptr;
            double per = 0.0;
            if (ctx) {
                ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
                ggml_set_input(x);
                ggml_tensor * cur = x;
                std::vector<ggml_tensor *> dev_ops, host_ops;
                for (int i = 0; i < k_hop_layers; ++i) {
                    cur = ggml_scale(ctx, cur, 1.0f);
                    dev_ops.push_back(cur);
                    cur = ggml_scale(ctx, cur, 1.0f);
                    (mixed ? host_ops : dev_ops).push_back(cur);
                }
                cur = ggml_scale(ctx, cur, 1.0f);
                dev_ops.push_back(cur);
                ggml_set_output(cur);
                ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4 * k_hop_layers + 16, false);
                ggml_build_forward_expand(gf, cur);
                bool ok = ggml_backend_supports_op(gpu, dev_ops.front());
                for (ggml_tensor * t : dev_ops)
                    ggml_backend_sched_set_tensor_backend(sched, t, gpu);
                for (ggml_tensor * t : host_ops)
                    ggml_backend_sched_set_tensor_backend(sched, t, cpu);
                ok = ok && ggml_backend_sched_alloc_graph(sched, gf);
                if (ok) {
                    std::vector<float> in((size_t) n_embd, 0.5f);
                    const auto run = [&] {
                        ggml_backend_tensor_set(x, in.data(), 0, ggml_nbytes(x));
                        const bool r = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
                        ggml_backend_sched_synchronize(sched);
                        return r;
                    };
                    if (run()) { // the first pays for setting up the path, untimed
                        const auto t0 = clock_t_::now();
                        int reps = 0;
                        while (reps < k_reps_max && run()) {
                            ++reps;
                            if (seconds_since(t0) >= k_sample_seconds) break;
                        }
                        const double secs = seconds_since(t0);
                        if (reps > 0 && secs > 0.0) per = secs / reps;
                    }
                }
                ggml_free(ctx);
            }
            if (sched) ggml_backend_sched_free(sched);
            return per;
        };
        const double all_device = time_chain(false);
        const double with_host = time_chain(true);
        if (all_device > 0.0 && with_host > all_device) hop = (with_host - all_device) / k_hop_layers;
    }
    if (cpu) ggml_backend_free(cpu);
    if (gpu) ggml_backend_free(gpu);
    return hop;
}

} // namespace

void probe_device_costs(HardwareProfile & hw, const ModelProfile & model) {
    for (ComputeDevice & d : hw.devices) {
        if (d.is_cpu) continue;        // the host is not a device to cross to
        if (!d.probe_here()) continue; // asked in another process, or found unusable there
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(d.name.c_str());
        if (!dev) continue;

        // Settle the memory question first: it is cheap, and a device whose memory turns out to be
        // the host's changes what every capacity rule above may conclude.
        // Native topology and an integrated-device declaration are positive evidence. Allocator
        // reuse or compression can flatten this delta, but cannot make their pool separate.
        if (d.shares_host_memory != Tri::Yes)
            d.shares_host_memory = detail::refine_memory_sharing(d.shares_host_memory, measure_shared_memory(dev));

        // Do not take a quarter of a gigabyte from a device that is short of it just to learn what
        // a crossing costs; the answer is worth having and it is not worth a failed allocation.
        if (d.has_own_memory() && d.memory_free && d.memory_free < k_probe_alloc * 4) continue;

        const double per_crossing = measure_crossing(dev, (int64_t) model.n_embd);
        if (per_crossing > 0.0) d.split_seconds = per_crossing;
        // Only where there is a second pool to reach: on a shared one a miss is a copy within it.
        const uint64_t expert = model.expert_slice_bytes * std::max<uint32_t>(1, model.n_expert_projections);
        if (d.has_own_memory() && expert > 0) {
            d.upload_gibs = measure_upload(dev, expert, false);
            d.upload_host_buffer_gibs = measure_upload(dev, expert, true);
            // At the thread count the host's own matmul was fastest at, the one a decode starts from.
            const int threads = (int) (hw.best_threads > 0 ? hw.best_threads : hw.n_cores);
            d.host_hop_seconds = measure_host_hop(dev, (int64_t) model.n_embd, threads);
        }
    }
}

} // namespace bmoe
