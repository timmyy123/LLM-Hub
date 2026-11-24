#include "memory_snapshot.h"

#include "bmoe/log.h"

namespace bmoe::detail {

namespace {
constexpr double k_mib = 1024.0 * 1024.0;
}

void record_process_memory(TokenMetrics & metrics, const pio::ProcessMemory & sample) {
    metrics.rss_mib = sample.rss_bytes / k_mib;
    metrics.rss_anon_mib = sample.rss_anon_bytes / k_mib;
    metrics.rss_file_mib = sample.rss_file_bytes / k_mib;
    metrics.swap_mib = sample.swap_bytes / k_mib;
    metrics.process_footprint_mib = sample.process_footprint_known ? sample.process_footprint_bytes / k_mib : -1.0;
    metrics.process_headroom_mib = sample.process_headroom_known ? sample.process_headroom_bytes / k_mib : -1.0;
}

void log_memory_snapshot(const char * phase) {
    pio::ProcessMemory pm;
    pio::DeviceMemory dm;
    TokenMetrics metrics;
    const bool process_known = pio::process_memory(&pm);
    if (process_known) record_process_memory(metrics, pm);
    const bool device_known = pio::device_memory(&dm);
    log_printf(LogLevel::Info,
               "memory: phase=%s process_footprint_mib=%.3f process_headroom_mib=%.3f rss_mib=%.3f "
               "compressed_or_swap_mib=%.3f system_available_mib=%.3f",
               phase, metrics.process_footprint_mib, metrics.process_headroom_mib,
               process_known ? metrics.rss_mib : -1.0, process_known ? metrics.swap_mib : -1.0,
               device_known ? dm.available_bytes / k_mib : -1.0);
}

} // namespace bmoe::detail
