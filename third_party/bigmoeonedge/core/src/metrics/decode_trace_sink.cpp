#include "bmoe/decode_trace.h"

#include <cstdio>

namespace bmoe {

namespace {

// The static preamble both traces share: the facts a row cannot carry, so a trace file stays
// analysable without the run that produced it (same spirit as the metrics CSV's `# summary`).
void write_static(std::FILE * f, const char * kind, const DecodeTraceStatic & s) {
    std::fprintf(f, "# %s v1\n", kind);
    std::fprintf(f, "# model=%s arch=%s n_layer=%d n_threads=%d io_threads=%d o_direct=%d overlap=%d\n",
                 s.model.c_str(), s.arch.c_str(), s.n_layer, s.n_threads, s.io_threads, (int) s.o_direct,
                 (int) s.overlap);
}

// A node name can carry anything ggml put there; commas and quotes would break the column count.
void write_csv_field(std::FILE * f, const std::string & v) {
    if (v.find_first_of(",\"\n") == std::string::npos) {
        std::fputs(v.c_str(), f);
        return;
    }
    std::fputc('"', f);
    for (char c : v) {
        if (c == '"') std::fputc('"', f); // RFC4180 doubling
        std::fputc(c == '\n' ? ' ' : c, f);
    }
    std::fputc('"', f);
}

class CsvComputeTraceSink final : public IComputeTraceSink {
public:
    CsvComputeTraceSink(std::FILE * f, bool layers) : f_(f), layers_(layers) {}
    ~CsvComputeTraceSink() override {
        if (f_) std::fclose(f_);
    }

    bool layers_only() const override { return layers_; }

    void on_static(const DecodeTraceStatic & s) override {
        write_static(f_, "compute_trace", s);
        std::fprintf(f_, "turn,phase,step,seq,layer,op,name,wall_ns,majflt\n");
        std::fflush(f_);
    }

    void on_rows(const ComputeTraceRow * rows, size_t n) override {
        for (size_t i = 0; i < n; ++i) {
            const ComputeTraceRow & r = rows[i];
            std::fprintf(f_, "%d,%d,%d,%d,%d,", r.turn, r.phase, r.step, r.seq, r.layer);
            write_csv_field(f_, r.op);
            std::fputc(',', f_);
            write_csv_field(f_, r.name);
            std::fprintf(f_, ",%llu,%llu\n", (unsigned long long) r.wall_ns, (unsigned long long) r.majflt);
        }
        std::fflush(f_); // once per decode, not per row
    }

private:
    std::FILE * f_ = nullptr;
    bool layers_ = false;
};

class CsvIoTraceSink final : public IIoTraceSink {
public:
    explicit CsvIoTraceSink(std::FILE * f) : f_(f) {}
    ~CsvIoTraceSink() override {
        if (f_) std::fclose(f_);
    }

    void on_static(const DecodeTraceStatic & s) override {
        write_static(f_, "io_trace", s);
        std::fprintf(f_, "turn,phase,step,layer,expert,proj,lane,spec,offset,req_bytes,read_bytes,latency_ns\n");
        std::fflush(f_);
    }

    void on_rows(const IoTraceRow * rows, size_t n) override {
        for (size_t i = 0; i < n; ++i) {
            const IoTraceRow & r = rows[i];
            std::fprintf(f_, "%d,%d,%d,%d,%d,%d,%d,%u,%llu,%llu,%llu,%llu\n", r.turn, r.phase, r.step, r.layer,
                         (int) r.expert, (int) r.proj, (int) r.lane, (unsigned) r.spec, (unsigned long long) r.offset,
                         (unsigned long long) r.req_bytes, (unsigned long long) r.read_bytes,
                         (unsigned long long) r.latency_ns);
        }
        std::fflush(f_);
    }

private:
    std::FILE * f_ = nullptr;
};

} // namespace

std::unique_ptr<IComputeTraceSink> make_csv_compute_trace_sink(const std::string & path, bool layers) {
    std::FILE * f = std::fopen(path.c_str(), "wb");
    return f ? std::make_unique<CsvComputeTraceSink>(f, layers) : nullptr;
}

std::unique_ptr<IIoTraceSink> make_csv_io_trace_sink(const std::string & path) {
    std::FILE * f = std::fopen(path.c_str(), "wb");
    return f ? std::make_unique<CsvIoTraceSink>(f) : nullptr;
}

} // namespace bmoe
