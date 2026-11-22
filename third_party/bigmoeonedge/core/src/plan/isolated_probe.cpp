// Probing a device in another process, so that a driver which ends the process ends that one.
// See bmoe/probe.h for the contract; this file is the text format and the two sides of the pipe.

#include "bmoe/probe.h"

#include "../io/platform_io.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace bmoe {

const char * const device_facts_marker = "BMOE_DEVICE_FACTS";

namespace {

char tri_char(Tri t) {
    return t == Tri::Yes ? 'Y' : t == Tri::No ? 'N' : 'U';
}

bool tri_from(const std::string & v, Tri & out) {
    if (v == "Y")
        out = Tri::Yes;
    else if (v == "N")
        out = Tri::No;
    else if (v == "U")
        out = Tri::Unknown;
    else
        return false;
    return true;
}

// How many fields a complete line carries. A line with fewer is a child that died while printing
// or a different build's idea of the format, and neither may be half-applied.
constexpr int k_fields = 20;

} // namespace

std::string device_facts_to_text(const ComputeDevice & d) {
    char b[512];
    std::snprintf(b, sizeof(b),
                  "v=3 shared=%c hostbuf=%c hostptr=%c async=%c rebind=%d runs=%c repack=%c bw=%.6g ident=%c "
                  "hpok=%c split=%.9g splits=%u wide=%.6g wideok=%c locked=%c up=%.6g uphb=%.6g expert=%.6g dense=%.6g "
                  "hop=%.9g",
                  tri_char(d.shares_host_memory), tri_char(d.host_buffer), tri_char(d.host_ptr_buffers),
                  tri_char(d.async_copies), d.rebindable ? 1 : 0, tri_char(d.runs_expert_op), tri_char(d.needs_repack),
                  d.memory_bandwidth_gibs, tri_char(d.identity_ok), tri_char(d.host_ptr_verified), d.split_seconds,
                  (unsigned) d.graph_splits, d.wide_matmul_gibs, tri_char(d.wide_identity_ok),
                  tri_char(d.buffers_locked), d.upload_gibs, d.upload_host_buffer_gibs, d.expert_gibs, d.dense_gibs,
                  d.host_hop_seconds);
    return b;
}

bool device_facts_from_text(const std::string & text, ComputeDevice & d) {
    ComputeDevice n = d; // applied whole or not at all
    std::istringstream in(text);
    std::string kv;
    int seen = 0;
    bool versioned = false;
    while (in >> kv) {
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) return false;
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        bool ok = true;
        if (k == "v") {
            versioned = v == "3"; // the same binary on both ends: a mismatch is noise
            continue;
        } else if (k == "shared")
            ok = tri_from(v, n.shares_host_memory);
        else if (k == "hostbuf")
            ok = tri_from(v, n.host_buffer);
        else if (k == "hostptr")
            ok = tri_from(v, n.host_ptr_buffers);
        else if (k == "async")
            ok = tri_from(v, n.async_copies);
        else if (k == "rebind")
            n.rebindable = v == "1";
        else if (k == "runs")
            ok = tri_from(v, n.runs_expert_op);
        else if (k == "repack")
            ok = tri_from(v, n.needs_repack);
        else if (k == "bw")
            n.memory_bandwidth_gibs = std::strtod(v.c_str(), nullptr);
        else if (k == "ident")
            ok = tri_from(v, n.identity_ok);
        else if (k == "hpok")
            ok = tri_from(v, n.host_ptr_verified);
        else if (k == "split")
            n.split_seconds = std::strtod(v.c_str(), nullptr);
        else if (k == "splits")
            n.graph_splits = (uint32_t) std::strtoul(v.c_str(), nullptr, 10);
        else if (k == "wide")
            n.wide_matmul_gibs = std::strtod(v.c_str(), nullptr);
        else if (k == "wideok")
            ok = tri_from(v, n.wide_identity_ok);
        else if (k == "locked")
            ok = tri_from(v, n.buffers_locked);
        else if (k == "up")
            n.upload_gibs = std::strtod(v.c_str(), nullptr);
        else if (k == "uphb")
            n.upload_host_buffer_gibs = std::strtod(v.c_str(), nullptr);
        else if (k == "hop")
            n.host_hop_seconds = std::strtod(v.c_str(), nullptr);
        else if (k == "expert")
            n.expert_gibs = std::strtod(v.c_str(), nullptr);
        else if (k == "dense")
            n.dense_gibs = std::strtod(v.c_str(), nullptr);
        else
            return false;
        if (!ok) return false;
        ++seen;
    }
    if (!versioned || seen != k_fields) return false;
    d = n;
    return true;
}

void probe_devices_isolated(HardwareProfile & hw, const DeviceProbeRunner & run) {
    if (!run) return;
    for (ComputeDevice & d : hw.devices) {
        // The host is this process: not something a second process could try on its behalf.
        if (d.is_cpu) continue;
        std::string facts, why;
        d.probed_out_of_process = true;
        if (run(d.name, facts, why) && device_facts_from_text(facts, d)) {
            d.usable = Tri::Yes;
        } else {
            d.usable = Tri::No;
            d.unusable_reason = why.empty() ? "its probe returned nothing that could be read" : why;
        }
    }
}

bool probe_one_device(const char * model_path, const std::string & device, std::string & facts) {
    HardwareProfile hw = probe_hardware(nullptr);
    bool found = false;
    std::vector<ComputeDevice> keep;
    for (ComputeDevice & d : hw.devices) {
        if (d.is_cpu) {
            keep.push_back(d); // the reference every device is judged against
        } else if (d.name == device) {
            keep.push_back(d);
            found = true;
        }
    }
    if (!found) return false;
    hw.devices = std::move(keep);
    const ModelProfile mp = probe_model(model_path, probe_engine_layers(model_path));
    probe_device_support(hw, mp);
    probe_bandwidth(hw, mp);
    probe_device_costs(hw, mp);
    for (const ComputeDevice & d : hw.devices)
        if (!d.is_cpu) facts = device_facts_to_text(d);
    return true;
}

DeviceProbeRunner self_process_runner(std::function<std::vector<std::string>(const std::string & device)> argv_for,
                                      double timeout_seconds) {
    if (!pio::can_run_self()) return nullptr;
    return [argv_for, timeout_seconds](const std::string & device, std::string & facts, std::string & why) {
        std::string out;
        const pio::ChildResult r = pio::run_self(argv_for(device), timeout_seconds, &out);
        if (r.outcome == pio::ChildOutcome::TimedOut) {
            char b[96];
            std::snprintf(b, sizeof(b), "its probe did not finish in %.0f s and was stopped", timeout_seconds);
            why = b;
            return false;
        }
        if (r.outcome == pio::ChildOutcome::Signalled) {
            why = "its probe ended the process that ran it (signal " + std::to_string(r.code) + ")";
            return false;
        }
        if (r.outcome != pio::ChildOutcome::Exited || r.code != 0) {
            why = r.outcome == pio::ChildOutcome::Exited ? "its probe failed (exit " + std::to_string(r.code) + ")"
                                                         : "a second process could not be started to probe it";
            return false;
        }
        // The line after the marker; a backend prints what it likes on the same stream.
        const size_t at = out.rfind(device_facts_marker);
        if (at == std::string::npos) {
            why = "its probe finished without reporting";
            return false;
        }
        size_t b0 = at + std::strlen(device_facts_marker);
        size_t e0 = out.find('\n', b0);
        facts = out.substr(b0, e0 == std::string::npos ? std::string::npos : e0 - b0);
        return true;
    };
}

} // namespace bmoe
