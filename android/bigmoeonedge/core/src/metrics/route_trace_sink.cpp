#include "bmoe/route_trace.h"

#include <cmath>
#include <cstdio>

namespace bmoe {

namespace {

class CsvRouteTraceSink final : public IRouteTraceSink {
public:
    explicit CsvRouteTraceSink(std::FILE * f) : f_(f) {}
    ~CsvRouteTraceSink() override {
        if (f_) std::fclose(f_);
    }

    void on_static(const RouteTraceStatic & s) override {
        // A `#` preamble keeps the trace self-contained — the same spirit as the metrics CSV's
        // `# summary` trailer: the static facts a row cannot carry travel with the rows, so a
        // trace file is analysable without the model that produced it.
        std::fprintf(f_, "# route_trace v1\n");
        std::fprintf(f_, "# model=%s arch=%s n_layer=%d n_expert=%d n_expert_used=%d\n", s.model.c_str(),
                     s.arch.c_str(), s.n_layer, s.n_expert, s.n_expert_used);
        for (size_t il = 0; il < s.expert_bytes_per_layer.size(); ++il) {
            const uint64_t dense = il < s.dense_bytes_per_layer.size() ? s.dense_bytes_per_layer[il] : 0;
            std::fprintf(f_, "# layer=%zu expert_bytes=%llu dense_bytes=%llu\n", il,
                         (unsigned long long) s.expert_bytes_per_layer[il], (unsigned long long) dense);
        }
        std::fprintf(f_, "turn,phase,step,layer,slot,expert,weight,residency,expert_bytes,dropped\n");
        std::fflush(f_);
    }

    void on_rows(const RouteTraceRow * rows, size_t n) override {
        for (size_t i = 0; i < n; ++i) {
            const RouteTraceRow & r = rows[i];
            // A weight the graph never exposed prints as nan, not 0: "unknown" must not read as
            // "the router gave this expert no mass".
            if (std::isnan(r.weight))
                std::fprintf(f_, "%d,%d,%d,%d,%d,%d,nan,%u,%llu,%u\n", r.turn, r.phase, r.step, r.layer, r.slot,
                             (int) r.expert, (unsigned) r.residency, (unsigned long long) r.expert_bytes,
                             (unsigned) r.dropped);
            else
                std::fprintf(f_, "%d,%d,%d,%d,%d,%d,%.6g,%u,%llu,%u\n", r.turn, r.phase, r.step, r.layer, r.slot,
                             (int) r.expert, (double) r.weight, (unsigned) r.residency,
                             (unsigned long long) r.expert_bytes, (unsigned) r.dropped);
        }
        std::fflush(f_); // once per decode, not per row
    }

private:
    std::FILE * f_ = nullptr;
};

} // namespace

std::unique_ptr<IRouteTraceSink> make_csv_route_trace_sink(const std::string & path) {
    std::FILE * f = std::fopen(path.c_str(), "w");
    return f ? std::make_unique<CsvRouteTraceSink>(f) : nullptr;
}

} // namespace bmoe
