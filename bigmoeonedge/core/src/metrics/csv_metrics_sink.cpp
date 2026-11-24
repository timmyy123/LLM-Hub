#include "bmoe/metrics.h"

#include <cstdio>

namespace bmoe {

namespace {

class CsvMetricsSink final : public IMetricsSink {
public:
    explicit CsvMetricsSink(std::FILE * f) : f_(f) {}
    ~CsvMetricsSink() override {
        if (f_) std::fclose(f_);
    }

    // The `#` preamble, mirroring the route/compute traces: what this run was, before what it did.
    // Key=value, whitespace-separated, order-independent — new keys are appended freely and older
    // parsers ignore what they do not know.
    void on_run_info(const RunInfo & r) override {
        if (header_) return; // a session sends this once; a second call would interleave a preamble
        // v2 adds the engine version and every knob v1 left out — the compute-buffer width, the
        // prediction settings, the drop variants, the sampling parameters. A file that does not
        // record a setting cannot be compared against one that changed it, and several of these
        // move the very columns below (n_ubatch sets the compute-buffer reservation; a probed or
        // sampled run is not a benchmark run at all).
        std::fprintf(f_, "# bmoe_metrics v2\n");
        // The engine version gets its own line, and the model line keeps starting with `model=`:
        // the app's CSV reader finds the run's name by looking for a line that BEGINS with
        // "# model=", so prefixing that line with anything would have cost every already-installed
        // build its run titles. A new key on a new line is invisible to a parser that ignores it.
        std::fprintf(f_, "# engine=%s\n", r.engine_version.c_str());
        std::fprintf(f_, "# cache_type_k=%s cache_type_v=%s\n", r.cache_type_k.c_str(), r.cache_type_v.c_str());
        std::fprintf(f_,
                     "# model=%s arch=%s n_layer=%d n_expert=%d n_expert_used=%d threads=%d "
                     "n_ctx=%d n_ubatch=%d chatml=%d\n",
                     r.model.c_str(), r.arch.c_str(), r.n_layer, r.n_expert, r.n_expert_used, r.n_threads, r.n_ctx,
                     r.n_ubatch, r.chatml);
        std::fprintf(f_,
                     "# moe_stream=%d cache_mb=%d cache_auto=%d cache_floor_mb=%d cache_ceil_mb=%d "
                     "cache_cycle_mb=%d force_cache=%d load_all=%d io_threads=%d o_direct=%d "
                     "overlap=%d io_two_wave=%d prefetch=%d "
                     "route_ahead=%d predict_prefetch=%d predict_log=%d predict_spec_max=%d prefetch_sync=%d "
                     "dense_weights=%s drop_cold_frac=%.4g drop_renorm=%d drop_prefill=%d "
                     "substitute_lambda=%.4g\n",
                     r.moe_stream, r.cache_mb, r.cache_auto, r.cache_floor_mb, r.cache_ceil_mb, r.cache_cycle_mb,
                     r.force_cache, r.load_all, r.io_threads, r.o_direct, r.overlap, r.io_two_wave, r.prefetch_layers,
                     r.route_ahead, r.predict_prefetch, r.predict_log, r.predict_spec_max, r.prefetch_sync,
                     r.dense_weights.c_str(), (double) r.drop_cold_frac, r.drop_renorm, r.drop_prefill,
                     (double) r.substitute_lambda);
        std::fprintf(f_,
                     "# temp=%.4g top_k=%d top_p=%.4g seed=%u compute_trace_layers=%d spec=%s "
                     "spec_draft_max=%d mtp_p_min=%.4g ngram_min_match=%d\n",
                     (double) r.temp, r.top_k, (double) r.top_p, r.seed, r.compute_trace_layers, r.spec.c_str(),
                     r.spec_draft_max, (double) r.mtp_p_min, r.ngram_min_match);
        write_header();
    }

    void on_token(const TokenMetrics & m) override {
        write_header(); // a caller that never sent RunInfo still gets a readable file
        std::fprintf(f_,
                     "%d,%d,%.3f,%.3f,%.3f,%llu,%.2f,%.3f,%.3f,%llu,%.3f,%.3f,%d,%.2f,%.1f,%.1f,%.1f,"
                     "%.1f,%.1f,%.1f,%.1f,%.1f,%.3f,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                     m.step, m.steps, m.wall_ms, m.io_ms, m.compute_ms, (unsigned long long) m.read_bytes,
                     m.cache_hit_pct, m.stall_ms, m.mgmt_ms, (unsigned long long) m.majflt, m.cpu_ms,
                     m.dense_resident_frac, m.turn, m.majflt_mib, m.cache_budget_mib, m.rss_mib, m.rss_anon_mib,
                     m.rss_file_mib, m.swap_mib, m.mem_available_mib, m.mem_free_mib, m.swap_free_mib,
                     m.loop_overhead_ms, m.mtp_batch, m.mtp_draft_ms, m.drain_ms, m.adopt_ms, m.ra_issue_ms, m.ra_wd_ms,
                     m.process_footprint_mib, m.process_headroom_mib);
        std::fflush(f_);
    }
    void on_summary(const RunSummary & s) override {
        // Run-level trailer parsed by scripts/bench-analyze.py as whitespace-separated key=value
        // tokens (order-independent). New keys are appended freely; older parsers ignore unknowns.
        std::fprintf(f_,
                     "# summary tokens=%d s/tok=%.3f tok/s=%.3f read_MiB=%.1f "
                     "io_s=%.2f compute_s/tok=%.3f io_s/tok=%.3f cache_hit_pct=%.1f "
                     "n_prompt=%d load_s=%.3f prefill_s=%.3f prefill_tps=%.2f stall_s/tok=%.3f mgmt_s/tok=%.3f "
                     "cache_resident_MiB=%.1f cache_budget_MiB=%.1f cache_resizes=%lld "
                     "spec_read_MiB=%.1f spec_experts=%lld spec_useful=%lld "
                     "majflt/tok=%.2f cpu_s/tok=%.4f token_demand_MiB=%.1f layer_demand_MiB=%.1f "
                     "experts_routed=%lld experts_dropped=%lld experts_reranked=%lld experts_substituted=%lld "
                     "loop_overhead_s/tok=%.4f "
                     "mtp_drafted=%lld mtp_accepted=%lld mtp_decodes=%lld mtp_draft_s/tok=%.4f "
                     "drafted_steps=%lld "
                     "ra_committed=%lld ra_passthrough=%lld ra_agree_pct=%.1f ra_gemv_ms/tok=%.2f "
                     "ra_issue_ms/tok=%.2f ra_wd_ms/tok=%.2f drain_s/tok=%.3f adopt_s/tok=%.3f "
                     "evictions=%lld rereads=%lld "
                     "prefill_cpu_s=%.3f prefill_read_mib=%.1f prefill_io_s=%.3f "
                     "prefill_stall_s=%.3f prefill_mgmt_s=%.3f "
                     "prefill_dev_tokens=%d prefill_dev_nodes=%lld prefill_dev_read_mib=%.1f prefill_dev_stall_s=%.3f "
                     "prefill_dev_cache_mib=%.1f "
                     "row_table_MiB=%.1f row_resident_MiB=%.1f row_rows=%lld row_reads=%lld "
                     "row_read_MiB=%.1f row_evictions=%lld row_io_errors=%lld\n",
                     s.n_generated, s.s_per_token, s.tokens_per_second, s.moe_read_mib, s.moe_io_seconds,
                     s.moe_compute_s_per_token, s.moe_io_s_per_token, s.cache_hit_pct, s.n_prompt, s.load_seconds,
                     s.prefill_seconds, s.prefill_seconds > 0 ? s.n_prompt / s.prefill_seconds : 0.0,
                     s.moe_stall_s_per_token, s.moe_mgmt_s_per_token, s.cache_resident_mib, s.cache_budget_mib,
                     s.cache_resizes, s.moe_spec_read_mib, s.moe_spec_experts, s.moe_spec_useful, s.majflt_per_token,
                     s.cpu_s_per_token, s.token_demand_mib, s.layer_demand_mib, s.experts_routed, s.experts_dropped,
                     s.experts_reranked, s.experts_substituted, s.loop_overhead_s_per_token, s.mtp_drafted,
                     s.mtp_accepted, s.mtp_decodes, s.mtp_draft_s_per_token, s.drafted_steps, s.route_ahead_overridden,
                     s.route_ahead_passthrough,
                     s.route_ahead_slots > 0 ? 100.0 * s.route_ahead_hits / s.route_ahead_slots : 0.0,
                     s.n_generated ? s.route_ahead_gemv_ns / 1e6 / s.n_generated : 0.0,
                     s.n_generated ? s.route_ahead_issue_ns / 1e6 / s.n_generated : 0.0,
                     s.n_generated ? s.route_ahead_wd_ns / 1e6 / s.n_generated : 0.0, s.moe_drain_s_per_token,
                     s.moe_adopt_s_per_token, s.cache_evictions, s.cache_rereads, s.prefill_cpu_seconds,
                     s.prefill_read_mib, s.prefill_io_seconds, s.prefill_stall_seconds, s.prefill_mgmt_seconds,
                     s.prefill_device_tokens, s.prefill_device_nodes, s.prefill_device_read_mib,
                     s.prefill_device_stall_seconds, s.prefill_device_cache_mib, s.row_table_mib, s.row_resident_mib,
                     s.row_rows, s.row_slab_reads, s.row_read_mib, s.row_evictions, s.row_io_errors);
        std::fflush(f_);
    }

private:
    // Deferred so the preamble can land above it: the column header is only correct once we know
    // whether anything is going to describe the run first.
    void write_header() {
        if (header_) return;
        header_ = true;
        // stall_ms (0 when serial), mgmt_ms (cache-management + the dense probe, split out of the
        // compute residual), majflt/cpu_ms (the fault + CPU-time decomposition of what remains of
        // "compute"), dense_resident_frac (how much of the dense set is still in RAM; -1 = unmeasured),
        // then the memory block. Consumers read columns by name, so order is not a contract.
        // The tail names costs that would otherwise hide inside another column: mtp_batch and
        // mtp_draft_ms describe a speculative group (one decode, several rows, the cost on the
        // first), and drain_ms / ra_issue_ms / ra_wd_ms come out of compute_ms while adopt_ms comes
        // out of mgmt_ms — see TokenMetrics. All 0 when the feature that pays them is off.
        std::fprintf(f_, "step,steps,wall_ms,io_ms,compute_ms,read_bytes,cache_hit_pct,stall_ms,mgmt_ms,majflt,cpu_ms,"
                         "dense_resident_frac,turn,majflt_mib,cache_budget_mib,rss_mib,rss_anon_mib,"
                         "rss_file_mib,swap_mib,mem_available_mib,mem_free_mib,swap_free_mib,loop_overhead_ms,"
                         "mtp_batch,mtp_draft_ms,drain_ms,adopt_ms,ra_issue_ms,ra_wd_ms,"
                         "process_footprint_mib,process_headroom_mib\n");
    }

    std::FILE * f_ = nullptr;
    bool header_ = false;
};

} // namespace

std::unique_ptr<IMetricsSink> make_csv_metrics_sink(const std::string & path) {
    std::FILE * f = std::fopen(path.c_str(), "w");
    return f ? std::make_unique<CsvMetricsSink>(f) : nullptr;
}

} // namespace bmoe
