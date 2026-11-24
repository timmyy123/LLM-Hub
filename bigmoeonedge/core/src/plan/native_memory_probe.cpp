#include "memory_topology.h"

#include <cstring>

namespace bmoe::detail {

// One line per API whose ggml backend reports every device as GPU, compiled where that API is.
// A backend that reports IGPU for an integrated device (Vulkan, CUDA, SYCL) needs no line: its
// device type already says it.
Tri native_memory_sharing(const char * backend, const char * description) {
    if (!backend || !description || !*description) return Tri::Unknown;
#if defined(BMOE_HAVE_METAL_MEMORY_PROBE)
    if (std::strcmp(backend, "MTL") == 0) return metal_memory_sharing(description);
#endif
#if defined(BMOE_HAVE_OPENCL_MEMORY_PROBE)
    if (std::strcmp(backend, "OpenCL") == 0) return opencl_memory_sharing(description);
#endif
    return Tri::Unknown;
}

} // namespace bmoe::detail
