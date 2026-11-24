#include "memory_topology.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstring>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace bmoe::detail {

namespace {

// The OpenCL library ggml's backend linked, if it is in this process. Never loaded here: where the
// backend's module was not loaded (no OpenCL on this system) there is nothing to ask, and loading
// another copy could resolve to a different driver than the one the backend computes on.
void * loaded_opencl_symbol(const char * name) {
#if defined(_WIN32)
    HMODULE lib = GetModuleHandleA("OpenCL.dll");
    return lib ? reinterpret_cast<void *>(GetProcAddress(lib, name)) : nullptr;
#else
    for (const char * file : {"libOpenCL.so", "libOpenCL.so.1"}) {
        void * lib = dlopen(file, RTLD_NOW | RTLD_NOLOAD);
        if (!lib) continue;
        void * sym = dlsym(lib, name);
        dlclose(lib); // the reference NOLOAD took; the backend's own keeps the library loaded
        if (sym) return sym;
    }
    return nullptr;
#endif
}

} // namespace

Tri opencl_memory_sharing(const char * description) {
    if (!description || !*description) return Tri::Unknown;
    using platforms_fn = cl_int (*)(cl_uint, cl_platform_id *, cl_uint *);
    using devices_fn = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
    using info_fn = cl_int (*)(cl_device_id, cl_device_info, size_t, void *, size_t *);
    const auto get_platforms = reinterpret_cast<platforms_fn>(loaded_opencl_symbol("clGetPlatformIDs"));
    const auto get_devices = reinterpret_cast<devices_fn>(loaded_opencl_symbol("clGetDeviceIDs"));
    const auto get_info = reinterpret_cast<info_fn>(loaded_opencl_symbol("clGetDeviceInfo"));
    if (!get_platforms || !get_devices || !get_info) return Tri::Unknown;

    cl_uint n_platforms = 0;
    if (get_platforms(0, nullptr, &n_platforms) != CL_SUCCESS || n_platforms == 0) return Tri::Unknown;
    std::vector<cl_platform_id> platforms(n_platforms);
    if (get_platforms(n_platforms, platforms.data(), nullptr) != CL_SUCCESS) return Tri::Unknown;

    Tri found = Tri::Unknown;
    for (cl_platform_id platform : platforms) {
        cl_uint n_devices = 0;
        if (get_devices(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &n_devices) != CL_SUCCESS || n_devices == 0) continue;
        std::vector<cl_device_id> devices(n_devices);
        if (get_devices(platform, CL_DEVICE_TYPE_ALL, n_devices, devices.data(), nullptr) != CL_SUCCESS) continue;
        for (cl_device_id device : devices) {
            char name[256] = {};
            if (get_info(device, CL_DEVICE_NAME, sizeof(name) - 1, name, nullptr) != CL_SUCCESS) continue;
            if (std::strcmp(name, description) != 0) continue;
            cl_bool unified = CL_FALSE;
            if (get_info(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, nullptr) != CL_SUCCESS)
                return Tri::Unknown; // a device of this name the driver does not answer for
            const Tri answer = unified ? Tri::Yes : Tri::No;
            if (found != Tri::Unknown && found != answer) return Tri::Unknown; // same name, different memory
            found = answer;
        }
    }
    return found;
}

} // namespace bmoe::detail
