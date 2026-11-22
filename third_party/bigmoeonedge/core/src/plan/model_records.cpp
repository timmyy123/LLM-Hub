#include "bmoe/model_records.h"

#include "bmoe/log.h"
#include "bmoe/probe.h"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace bmoe {

std::string ModelRecords::path_for(const std::string & model_path, const char * ext) {
    if (dir_.empty() || model_path.empty()) return "";
    const fs::path p = fs::u8path(model_path);
    std::error_code ec;
    const uint64_t size = fs::is_regular_file(p, ec) ? (uint64_t) fs::file_size(p, ec) : 0;
    if (ec || size == 0) return "";
    const auto mt = fs::last_write_time(p, ec);
    const long long mtime = ec ? 0 : (long long) mt.time_since_epoch().count();
    std::string fp;
    {
        std::lock_guard<std::mutex> lk(m_);
        auto it = fingerprints_.find(model_path);
        if (it != fingerprints_.end() && it->second.size == size && it->second.mtime == mtime)
            fp = it->second.fingerprint;
    }
    if (fp.empty()) {
        fp = model_fingerprint(model_path.c_str());
        if (fp.empty()) return "";
        std::lock_guard<std::mutex> lk(m_);
        fingerprints_[model_path] = {size, mtime, fp};
    }
    return (fs::u8path(dir_) / (fp + ext)).u8string();
}

bool ModelRecords::load(const std::string & model_path, RoutingTrace & out) {
    const std::string path = path_for(model_path);
    if (path.empty()) return false;
    std::ifstream f(fs::u8path(path), std::ios::binary);
    if (!f) return false;
    std::ostringstream text;
    text << f.rdbuf();
    std::string err;
    if (parse_routing_trace(text.str(), out, err)) return true;
    log_printf(LogLevel::Warn, "ignoring the routing kept in %s: %s", path.c_str(), err.c_str());
    return false;
}

void ModelRecords::keep(const std::string & model_path, const RoutingTrace & rec) {
    if (rec.empty() || rec.decode_tokens == 0) return;
    const std::string path = path_for(model_path);
    if (path.empty()) return;
    RoutingTrace old;
    if (load(model_path, old) && old.decode_tokens > rec.decode_tokens) return;
    write(path, format_routing_trace(rec), "routing record");
}

// Written beside and renamed, so a reader never sees half of one.
void ModelRecords::write(const std::string & path, const std::string & text, const char * what) {
    const fs::path p = fs::u8path(path);
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    fs::path tmp = p;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            log_printf(LogLevel::Warn, "cannot write the %s %s", what, path.c_str());
            return;
        }
        f << text;
    }
    fs::rename(tmp, p, ec);
    if (ec) log_printf(LogLevel::Warn, "cannot write the %s %s", what, path.c_str());
}

bool ModelRecords::load_knobs(const std::string & model_path, std::vector<MeasuredKnobs> & out) {
    out.clear();
    const std::string path = path_for(model_path, ".knobs");
    if (path.empty()) return false;
    std::ifstream f(fs::u8path(path), std::ios::binary);
    if (!f) return false;
    std::ostringstream text;
    text << f.rdbuf();
    std::string err;
    if (parse_measured_knobs(text.str(), out, err)) return true;
    log_printf(LogLevel::Warn, "ignoring the knob measurements kept in %s: %s", path.c_str(), err.c_str());
    return false;
}

void ModelRecords::keep_knobs(const std::string & model_path, const MeasuredKnobs & rec) {
    if (rec.empty() || rec.hardware.empty() || rec.engine.empty()) return;
    const std::string path = path_for(model_path, ".knobs");
    if (path.empty()) return;
    std::lock_guard<std::mutex> lk(knobs_m_);
    std::vector<MeasuredKnobs> all;
    load_knobs(model_path, all);
    keep_measured_knobs(all, rec);
    write(path, format_measured_knobs(all), "knob measurements");
}

} // namespace bmoe
