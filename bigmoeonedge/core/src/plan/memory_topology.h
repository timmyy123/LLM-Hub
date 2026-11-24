#pragma once

#include "bmoe/hardware_profile.h"

namespace bmoe::detail {

// A flat allocation delta can miss shared pages already charged to the process. It cannot
// contradict positive evidence that the device uses the host's physical pool.
inline Tri refine_memory_sharing(Tri known, Tri measured) {
    if (known == Tri::Yes || measured == Tri::Unknown) return known;
    return measured;
}

// What a device's own API says about its memory, for a backend whose ggml device type does not say
// it (a backend that reports every device as GPU, integrated or not). The device's driver answers
// the question as a fixed property, the same on every run, where measuring what an allocation costs
// the host is a reading of a noisy figure. One answer per API, each compiled where its API is
// present; Unknown for any other backend, a device the API does not find by the description ggml
// gave it, or a driver that does not answer. `backend` is the ggml registry's name for the backend.
Tri native_memory_sharing(const char * backend, const char * description);

// The pinned Metal backend uses the system default physical device. A description mismatch
// leaves topology unknown rather than assigning another device's memory to this one.
Tri metal_memory_sharing(const char * description);

// OpenCL's CL_DEVICE_HOST_UNIFIED_MEMORY, for the device whose CL_DEVICE_NAME is the description
// (ggml's OpenCL backend reports that name as the description). Read through the OpenCL library
// the backend already loaded into this process, never one loaded here: on a system without OpenCL
// the backend is not loaded and the answer is Unknown. Several devices of that name must agree.
Tri opencl_memory_sharing(const char * description);

} // namespace bmoe::detail
