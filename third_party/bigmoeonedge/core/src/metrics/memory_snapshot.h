#pragma once

#include "bmoe/metrics.h"
#include "../io/platform_io.h"

namespace bmoe::detail {

void record_process_memory(TokenMetrics & metrics, const pio::ProcessMemory & sample);

// Diagnostics through the ordinary log sink. They never change a budget or allocation decision.
void log_memory_snapshot(const char * phase);

} // namespace bmoe::detail
